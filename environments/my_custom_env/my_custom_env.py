"""
A multi-turn, curriculum-based coding environment for Juspay Hyperswitch.

Loads real PR and commit tasks from a JSONL dataset or a Repo2RLEnv-generated
Harbor task directory.  Manages sandboxed execution using a pre-built Docker
image (dumball/hyperswitch-rl:d2457784) that has the hyperswitch repo at
/hyperswitch.

Reward — we train for *mergeable* patches, not merely test-passing ones.
The dense reward is structurally weighted with an ADDITIVE compile term
(see REWARD_DESIGN.md for the full reasoning):

    reward  = (1 - w_compile)·quality + w_compile·compiles
    quality = w_struct·structural + w_style·style + w_judge·judge

    structural = 0.30·file_targeting_F1   (right files?)
               + 0.70·ast                 (right items + right symbols?)
    ast        = 0.5·location_F1          (edits land in the same fns/structs)
               + 0.5·reference_F1         (edits USE the same types/fields/methods)

    compiles   = 1.0 if `cargo check -p <touched crates>` passes else 0.0
                 (ADDITIVE, not a ×0.25 gate — a multiplicative gate squashed
                  structural variance and starved the GRPO gradient)

    style = Rust style checker (no unwrap/panic/unsafe/dbg/as-cast)
    judge = optional LLM merge-quality rating; when no API key is
            available it returns None and the remaining weights are
            renormalized (so the env runs fully offline).

Rationale (see findings_pr11372.md): F2P/P2P `cargo test` is a coarse
oracle blind to structural quality — it scored a known agent patch 1.0
despite a type-safety regression a reviewer would reject.  So `cargo test`
F2P/P2P is kept only as a periodic *eval* gate (grade_test_execution +
parse_cargo_test below), never the dense training signal.  `cargo check`
is the cheapest real correctness oracle and gates the structural reward.

Repo2RLEnv utilities inlined below are from:
  https://github.com/huggingface/Repo2RLEnv  (Apache-2.0)
  reward.py — SWE-RL-inspired diff-similarity reward
  log_parsers/cargo_parser.py — cargo test output parser
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from datasets import Dataset


def _load_dotenv() -> None:
    """Load the project .env into os.environ (existing vars win) so the env
    worker subprocess gets JUDGE_API_KEY etc. without relying on the launch
    shell sourcing it. Dependency-free; runs once at import."""
    for base in (Path.cwd(), *Path(__file__).resolve().parents):
        envf = base / ".env"
        if not envf.exists():
            continue
        for line in envf.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
        return


_load_dotenv()

logger = logging.getLogger("hyperswitch_env")
logger.setLevel(logging.INFO)
logger.propagate = True
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
import verifiers as vf
from verifiers.envs.experimental.composable import (
    SandboxSpec,
    SandboxTaskSet,
    ComposableEnv,
)
from verifiers.envs.experimental.composable.harnesses.opencode import opencode_harness
from verifiers.envs.experimental.sandbox_mixin import SandboxTimeouts

# ── Curriculum System Prompts ──────────────────────────────────────────

EXECUTION_DISCIPLINE = (
    "EXECUTION DISCIPLINE:\n"
    "What you are scored on, in order:\n"
    "1. Your change COMPILES (`cargo check` passes) — a building change is rewarded substantially "
    "more than one that doesn't. An empty diff scores zero.\n"
    "2. You edit the RIGHT code — the same functions, structs, impls and modules the task is about, "
    "and you USE the right types/fields/methods (don't just add empty stubs or unused fields).\n"
    "3. The change is CORRECT and COMPLETE for the task, and the diff is minimal and clean.\n\n"
    "MANDATORY WORKFLOW — follow this exactly:\n"
    "1. Explore for at most 2 turns (skim, don't audit).\n"
    "2. Make your first `edit` by turn 3 at the latest.\n"
    "3. IMMEDIATELY run `cargo check -p <package>` after EVERY edit.\n"
    "4. If cargo check fails: fix the compile error NOW before doing anything else. Repeat until it passes.\n"
    "5. Only proceed to the next edit after the current one compiles cleanly.\n\n"
    "HARD RULES:\n"
    "- NEVER finish without running `cargo check` on your changes.\n"
    "- NEVER stack multiple edits without verifying each compiles first.\n"
    "- NEVER end with an empty diff — reward is 0.0 with no edits.\n"
    "- Do not duplicate `use` imports or `impl` blocks — search first.\n\n"
)


def _read_skill(filename: str) -> str:
    """Read a skill markdown from skills/, stripping YAML frontmatter."""
    skill_path = Path(__file__).parent / "skills" / filename
    if not skill_path.exists():
        return ""
    text = skill_path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2].strip()
    return text


def _load_repo_skill() -> str:
    """Assemble the distilled hyperswitch repo briefing from the PR #11372
    investigation skills.

    Two complementary skills:
      - hyperswitch-architecture: the navigation map — crate ownership, the
        v1/v2 cfg split, layer ownership (build vs consume), typed-ID catalog,
        antipatterns. Tells the agent WHERE the fix lives so it doesn't burn
        its small turn budget grepping.
      - hyperswitch-style: CI-enforced conventions the reward grades (no
        unwrap/panic/unsafe/as-cast, typed wrappers, minimal diff).

    Architecture first (navigation), then style (conventions).
    """
    arch = _read_skill("hyperswitch-architecture.md")
    style = _read_skill("hyperswitch-style.md")
    parts = [p for p in (arch, style) if p]
    return "\n\n".join(parts)


_REPO_SKILL_FULL = _load_repo_skill()
_BRIEF_SEP = "\n\n" + ("-" * 40) + "\n\n"


def repo_briefing_for_level(curriculum_level: int) -> str:
    """Fade the repo briefing across the curriculum so the model internalizes
    the conventions into its WEIGHTS instead of depending on the prompt.

    The reward never changes — only how much briefing is in context:

      level 0  — FULL briefing (crate map + every CI-enforced rule). Scaffolds
                 early learning so rollouts produce mergeable edits at all.
      level 1  — the ABSOLUTES section only (the hard CI rules); the agent must
                 recall the crate map / conventions itself.
      level 2  — a one-line reminder that conventions exist; no specifics.
      level 3+ — NOTHING. The model must have internalized the repo knowledge
                 to keep its reward up. This is the deployment condition.

    Training plan: ramp `curriculum_level` 0→3 over the run (or across runs).
    Reward holding steady as the briefing is removed is the signal that the
    knowledge moved from prompt into weights (findings_pr11372.md §5).
    """
    if not _REPO_SKILL_FULL:
        return ""
    if curriculum_level <= 0:
        return (
            "REPOSITORY BRIEFING (juspay/hyperswitch) — read before editing:\n\n"
            f"{_REPO_SKILL_FULL}{_BRIEF_SEP}"
        )
    if curriculum_level == 1:
        absolutes = _REPO_SKILL_FULL
        start = absolutes.find("## ABSOLUTES")
        end = absolutes.find("## CONVENTIONS")
        if start != -1 and end != -1:
            absolutes = absolutes[start:end].strip()
        return (
            "REPOSITORY BRIEFING (juspay/hyperswitch) — the non-negotiable rules:\n\n"
            f"{absolutes}{_BRIEF_SEP}"
        )
    if curriculum_level == 2:
        return (
            "Reminder: this is the juspay/hyperswitch Rust monorepo. Follow its "
            "CI-enforced conventions (no unwrap/panic/unsafe/as-cast, prefer typed "
            "wrappers, respect the v1/v2 cfg split, keep the diff minimal)."
            f"{_BRIEF_SEP}"
        )
    return ""  # level 3+: no hints — internalized-knowledge condition

SYSTEM_PROMPTS = {
    0: (
        "You are an expert Rust software engineer. You must solve the provided coding task in the `juspay/hyperswitch` repository while strictly adhering to our high-quality development standards.\n\n"
        + EXECUTION_DISCIPLINE +
        "CRITICAL CODING STYLE RULES:\n"
        "1. Avoid `unsafe` blocks completely.\n"
        "2. Avoid `.unwrap()` and `.expect()`. Use error propagation (`?`) or proper error-handling enums instead.\n"
        "3. Do not use `panic!` to abort execution. Return an appropriate `Result` or error response.\n"
        "4. Never include debugging prints like `dbg!` or `println!`. If needed, use structured logging from the tracing framework.\n"
        "5. Avoid isolated `as` type-casts (e.g. `x as u64`) to prevent overflow or precision loss. Use `.into()`, `.try_into()`, or specific conversion methods.\n"
        "6. Keep your git diff minimal: only modify the files and lines that are strictly necessary to implement the requested change. Avoid whitespace-only or unrelated modifications."
    ),
    1: (
        "You are an expert Rust engineer. Solve the task and follow these rules:\n"
        + EXECUTION_DISCIPLINE +
        "- Do not use `unsafe`.\n"
        "- Do not use `.unwrap()` or `.expect()`.\n"
        "- Do not use `panic!`.\n"
        "- Do not use `dbg!` or `println!`.\n"
        "- Avoid `as` type-casts.\n"
        "- Keep your diff minimal."
    ),
    2: (
        EXECUTION_DISCIPLINE +
        "You are a Rust engineer. Make sure to solve the task. Ensure that your code compiles, passes tests, uses robust error handling (no unwraps or panics), is safe (no unsafe), has clean logging (no debug prints), and keeps the diff size minimal."
    ),
    3: (
        EXECUTION_DISCIPLINE +
        "You are a Rust engineer. Solve the task. Write clean, production-grade Rust code that compiles and passes all tests successfully."
    ),
    4: (
        EXECUTION_DISCIPLINE +
        "You are a Rust engineer. Solve the task."
    ),
}

# ── Diff parsing helpers (shared by file/AST structural rewards) ─────────

_HUNK_HEADER_RE = re.compile(r"^@@.*@@")
_FILE_HEADER_RE = re.compile(r"^(?:---|\+\+\+) ")
_INDEX_LINE_RE = re.compile(r"^index ")
_DIFF_GIT_RE = re.compile(r"^diff --git ")


@dataclass(slots=True)
class DiffRewardMetadata:
    similarity: float
    pred_lines: int
    oracle_lines: int
    matched_lines: int
    parse_error: str | None = None


def _normalize_diff(diff: str) -> list[str]:
    """Strip volatile metadata (hunk line numbers, indices, file headers)."""
    lines: list[str] = []
    for line in diff.splitlines():
        if _DIFF_GIT_RE.match(line):
            continue
        if _INDEX_LINE_RE.match(line):
            continue
        if _HUNK_HEADER_RE.match(line):
            lines.append("@@")
            continue
        if _FILE_HEADER_RE.match(line):
            lines.append(line.split("\t")[0].strip())
            continue
        lines.append(line)
    return lines


# ── AST-level structural reward (tree-sitter-rust) ───────────────────────
# Replaces textual diff-line similarity. Matching diff LINES punishes
# equivalent code written differently and is fooled by shifted line numbers.
# Instead we parse the patched Rust with tree-sitter and score WHICH named
# items the change actually touches — functions, structs, enums, unions,
# traits, impl targets, modules, consts, type aliases, macros. This is the
# "what the AST is affecting" signal: did the agent modify the same semantic
# units as the gold solution, regardless of formatting or line positions?

_TS_NAMED_ITEMS = {
    "function_item", "struct_item", "enum_item", "union_item", "trait_item",
    "mod_item", "const_item", "static_item", "type_item", "macro_definition",
}

_ts_language = None


def _rust_language():
    """Cached tree-sitter Rust Language. Safe to share across threads; Parser
    objects are NOT, so callers build a fresh Parser per use."""
    global _ts_language
    if _ts_language is None:
        import tree_sitter_rust as tsr
        from tree_sitter import Language

        _ts_language = Language(tsr.language())
    return _ts_language


def _old_side_touched_lines(diff_text: str) -> dict[str, set[int]]:
    """Per base file (a/ path), the 1-based BASE line numbers the diff touches.

    Tracks old-side line numbers from the hunk headers: removed lines are
    touched directly; added lines are attributed to the base line they're
    inserted after. Both the agent diff (`git diff --cached HEAD`) and the gold
    diff are taken against the same base_commit, so their old-side line numbers
    index the SAME base file — which we parse in full to find the enclosing
    item even when the edit is deep inside a function body.
    """
    touched: dict[str, set[int]] = {}
    current: str | None = None
    old_ln = 0
    for line in (diff_text or "").splitlines():
        hm = _DIFF_HEADER_RE.match(line)
        if hm:
            current = hm.group(1)  # a/ (base) path
            touched.setdefault(current, set())
            continue
        if current is None:
            continue
        m = re.match(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
        if m:
            old_ln = int(m.group(1))
            continue
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("-"):
            touched[current].add(old_ln)
            old_ln += 1
        elif line.startswith("+"):
            touched[current].add(max(old_ln - 1, 1))  # insertion point in base
        else:  # context line
            old_ln += 1
    return touched


def _items_for_lines(base_src: bytes, fpath: str, lines: set[int]) -> set[str]:
    """Map 1-based base line numbers to their innermost enclosing Rust item.

    Parses the FULL base file, so an edit anywhere inside a function/impl/struct
    resolves to that item — fixing the fragment-parse blind spot for in-body
    edits. Returns `file::kind:name` labels.
    """
    if not lines:
        return set()
    from tree_sitter import Parser

    tree = Parser(_rust_language()).parse(base_src)
    spans: list[tuple[int, int, str]] = []  # (start0, end0, label) 0-based
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in _TS_NAMED_ITEMS:
            name_node = node.child_by_field_name("name") or node.child_by_field_name("type")
            name = (
                base_src[name_node.start_byte : name_node.end_byte].decode("utf-8", "replace")
                if name_node else "<anon>"
            )
            kind = node.type.replace("_item", "").replace("_definition", "")
            spans.append((node.start_point[0], node.end_point[0], f"{fpath}::{kind}:{name}"))
        stack.extend(node.children)

    out: set[str] = set()
    for ln in lines:
        i = ln - 1  # 0-based
        best: tuple[int, str] | None = None  # (span_size, label)
        for start, end, label in spans:
            if start <= i <= end:
                size = end - start
                if best is None or size < best[0]:
                    best = (size, label)
        if best is not None:
            out.add(best[1])
    return out


def _f1(gold: set[str], pred: set[str]) -> float | None:
    """F1 over two item sets. Returns None when gold is empty (signal undefined)
    so the caller can fall back instead of awarding a spurious perfect score."""
    if not gold:
        return None
    if not pred:
        return 0.0
    tp = len(gold & pred)
    if not tp:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(gold)
    return 2 * precision * recall / (precision + recall)


# Boilerplate symbols that carry no localization signal — every Rust patch
# mentions these, so counting them would just add noise to reference overlap.
_REF_STOPLIST = {
    "String", "str", "Option", "Some", "None", "Result", "Ok", "Err", "Vec",
    "Box", "Self", "self", "Default", "From", "Into", "clone", "into", "to_string",
    "to_owned", "as_ref", "as_str", "unwrap", "expect", "is_some", "is_none",
    "iter", "collect", "map", "and_then", "unwrap_or", "unwrap_or_default",
    "len", "push", "new", "default", "HashMap", "BTreeMap", "u8", "u16", "u32",
    "u64", "i32", "i64", "f64", "bool", "char", "Vec", "format",
}

# tree-sitter node types that denote a *referenced* symbol (a type, a field or
# method name, or a path segment) — as opposed to a binding/declaration.
_REF_NODE_TYPES = {"type_identifier", "field_identifier"}


def _referenced_symbols(diff_text: str) -> set[str]:
    """Symbols the diff's ADDED code references — to full nesting depth.

    Parses only the added (`+`) lines and walks the whole fragment AST, so a
    chain like `item.router_data.connector_request_reference_id.clone()` yields
    {router_data, connector_request_reference_id, clone}. Captures types, field
    accesses, method names and path segments; drops common boilerplate. This is
    the "did the edit USE the right APIs/types/fields" signal — complementary to
    location (which only says *where* the edit lives).
    """
    added = "\n".join(
        l[1:] for l in (diff_text or "").splitlines()
        if l.startswith("+") and not l.startswith("+++")
    )
    if not added.strip():
        return set()
    from tree_sitter import Parser

    src = added.encode("utf-8", "replace")
    tree = Parser(_rust_language()).parse(src)
    out: set[str] = set()
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in _REF_NODE_TYPES:
            sym = src[node.start_byte : node.end_byte].decode("utf-8", "replace")
            if sym and sym not in _REF_STOPLIST:
                out.add(sym)
        elif node.type == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None and fn.type in ("identifier", "scoped_identifier"):
                sym = src[fn.start_byte : fn.end_byte].decode("utf-8", "replace").split("::")[-1]
                if sym and sym not in _REF_STOPLIST:
                    out.add(sym)
        stack.extend(node.children)
    return out


# ── Repo2RLEnv: Test-Execution Grading (inlined from reward.py) ────────

@dataclass(slots=True)
class ExecutionReport:
    """Per-task report after running the model patch through the test suite."""

    fail_to_pass_success: list[str]
    fail_to_pass_failure: list[str]
    pass_to_pass_success: list[str]
    pass_to_pass_failure: list[str]

    @property
    def f2p_rate(self) -> float:
        total = len(self.fail_to_pass_success) + len(self.fail_to_pass_failure)
        return 1.0 if total == 0 else len(self.fail_to_pass_success) / total

    @property
    def p2p_rate(self) -> float:
        total = len(self.pass_to_pass_success) + len(self.pass_to_pass_failure)
        return 1.0 if total == 0 else len(self.pass_to_pass_success) / total

    @property
    def resolution_status(self) -> str:
        f2p, p2p = self.f2p_rate, self.p2p_rate
        if f2p == 1.0 and p2p == 1.0:
            return "FULL"
        if 0.0 < f2p < 1.0 and p2p == 1.0:
            return "PARTIAL"
        return "NO"


def grade_test_execution(
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    test_status: dict[str, str],
) -> ExecutionReport:
    """Compute the per-task report from the post-prediction test status map.

    Args:
        fail_to_pass: tests that must transition FAIL → PASS
        pass_to_pass: tests that must stay PASSED
        test_status:  {test_name -> PASSED|FAILED|SKIPPED|ERROR}

    Tests absent from test_status count as failures.
    """
    f2p_success, f2p_failure = [], []
    for t in fail_to_pass:
        (f2p_success if test_status.get(t) == "PASSED" else f2p_failure).append(t)
    p2p_success, p2p_failure = [], []
    for t in pass_to_pass:
        (p2p_success if test_status.get(t) == "PASSED" else p2p_failure).append(t)
    return ExecutionReport(
        fail_to_pass_success=f2p_success,
        fail_to_pass_failure=f2p_failure,
        pass_to_pass_success=p2p_success,
        pass_to_pass_failure=p2p_failure,
    )


# ── Repo2RLEnv: Cargo Test Log Parser (inlined from log_parsers/cargo_parser.py) ──
# Source: https://github.com/huggingface/Repo2RLEnv/blob/main/src/repo2rlenv/log_parsers/cargo_parser.py
# License: Apache-2.0

_CARGO_TEST_RE = re.compile(
    r"^test\s+(?P<name>\S+)\s+\.\.\.\s+(?P<status>ok|FAILED|ignored)\b",
)
_CARGO_STATUS_MAP: dict[str, str] = {
    "ok": "PASSED",
    "FAILED": "FAILED",
    "ignored": "SKIPPED",
}


def parse_cargo_test(log: str) -> dict[str, str]:
    """Return {test_name -> status} parsed from `cargo test` output.

    Parses `test NAME ... ok/FAILED/ignored` lines; ignores build output
    and the final summary line.
    """
    out: dict[str, str] = {}
    if not log:
        return out
    for raw in log.split("\n"):
        m = _CARGO_TEST_RE.match(raw)
        if m:
            out[m.group("name")] = _CARGO_STATUS_MAP[m.group("status")]
    return out


# ── Structural Diff Reward (from Repo2RLEnv pr_diff 6-component verifier) ──
# file_targeting (F1 over changed-file sets) + region_overlap (hunk spatial
# overlap) are the two strongest localization signals in
# _pr_diff_verifier.py.  We adopt them as the dense structural reward because
# they reward "right files, right region, right shape" — the merge-quality
# axis that F2P/P2P test execution is blind to (findings_pr11372.md §2.4).

def _files_in_diff(diff_text: str) -> set[str]:
    """Set of b/ file paths touched by a unified diff."""
    files: set[str] = set()
    for m in _DIFF_HEADER_RE.finditer(diff_text or ""):
        files.add(m.group(2))
    return files


def file_targeting_f1(oracle_diff: str, pred_diff: str) -> float:
    """F1 over the changed-file sets.

    F1 (not Jaccard) so missing a gold file is penalized harder than
    touching one extra file — matching pr_diff's choice.
    """
    gold = _files_in_diff(oracle_diff)
    pred = _files_in_diff(pred_diff)
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(gold)
    return 2 * precision * recall / (precision + recall)


def _hunk_regions(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Map each touched file -> list of (start, end) new-side line ranges,
    parsed from `@@ -a,b +c,d @@` hunk headers."""
    regions: dict[str, list[tuple[int, int]]] = {}
    current: str | None = None
    for line in (diff_text or "").splitlines():
        hm = _DIFF_HEADER_RE.match(line)
        if hm:
            current = hm.group(2)
            regions.setdefault(current, [])
            continue
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if m and current is not None:
            start = int(m.group(1))
            length = int(m.group(2)) if m.group(2) else 1
            regions[current].append((start, start + max(length, 1) - 1))
    return regions


def region_overlap(oracle_diff: str, pred_diff: str, *, slack: int = 5) -> float:
    """Fraction of oracle hunks whose file + line-region is matched (within
    `slack` lines) by some predicted hunk.

    Strongest spatial-localization signal — catches "edited the consumer,
    not the builder" (the pr11372 failure mode).
    """
    gold = _hunk_regions(oracle_diff)
    pred = _hunk_regions(pred_diff)
    gold_hunks = [(f, r) for f, rs in gold.items() for r in rs]
    if not gold_hunks:
        return 1.0
    matched = 0
    for f, (gs, ge) in gold_hunks:
        for (ps, pe) in pred.get(f, []):
            if ps - slack <= ge and pe + slack >= gs:  # overlap within slack
                matched += 1
                break
    return matched / len(gold_hunks)


# ── Style Checker ──────────────────────────────────────────────────────

def check_style_compliance(diff_text: str) -> float:
    violations = set()
    added_lines = []

    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])

    cleaned_lines = []
    for line in added_lines:
        line_no_comment = re.sub(r'//.*$', '', line)
        cleaned_lines.append(line_no_comment)

    combined_text = "\n".join(cleaned_lines)

    if re.search(r'\bunsafe\b', combined_text):
        violations.add("unsafe")
    if re.search(r'\.unwrap\b', combined_text):
        violations.add("unwrap")
    if re.search(r'\.expect\b', combined_text):
        violations.add("expect")
    if re.search(r'\bpanic!\b', combined_text):
        violations.add("panic")
    if re.search(r'\bdbg!\b', combined_text):
        violations.add("dbg")
    if re.search(r'\bprintln!\b', combined_text):
        violations.add("println")

    for line in cleaned_lines:
        if re.search(r'\bas\b', line):
            if not re.search(r'\b(use|extern)\b', line):
                violations.add("as_cast")
                break

    return max(0.0, 1.0 - len(violations) * 0.2)


# ── Rubric ─────────────────────────────────────────────────────────────

WORKDIR = "/hyperswitch"

_DIFF_HEADER_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)


class HyperswitchRubric(vf.Rubric):
    """Structurally-weighted reward for mergeable patches, with an ADDITIVE
    compile term (not a multiplicative gate).

        reward = (1 - COMPILE_WEIGHT)·quality  +  COMPILE_WEIGHT·compiles
        quality = 0.60·structural + 0.10·style + 0.30·judge
        structural = 0.30·file_targeting_F1 + 0.70·ast    (ast = location+reference)

    Compile is ADDITIVE so it doesn't squash the gold-anchored structural
    variance: a multiplicative ×0.25 gate collapsed two patches scoring 0.4 vs
    0.8 into ~0.1 vs 0.2, killing the within-group reward variance that GRPO
    advantages (and thus the gradient) feed on. Additive keeps quality at full
    scale while still rewarding code that actually builds — an orthogonal,
    solution-agnostic signal the gold-diff comparison can't provide.

    The judge carries low weight (0.15): it's a noisy LLM signal that in
    practice fails to emit parseable JSON the majority of the time, so we lean
    on the deterministic gold-anchored structural signal and treat the judge as
    a minor tie-breaker. When it returns None (offline or parse failure) the
    structural+style weights renormalize, so the env runs fully offline.
    `cargo test` F2P/P2P is NOT in the dense reward — it is a coarse oracle
    blind to structural quality (findings_pr11372.md §2.4) and belongs in a
    periodic eval gate.
    """

    # Weights for the quality term. Structural (gold-anchored, deterministic,
    # high-variance → learnable gradient) stays primary. The judge — now
    # reliable (forced JSON) and focused on the orthogonal correctness +
    # completeness axes (the real target; credits correct-but-different) — gets
    # a meaningful share. Style is the weakest signal (6-pattern regex), trimmed.
    W_STRUCT = 0.60
    W_STYLE = 0.10
    W_JUDGE = 0.30
    # Additive compile term: this fraction of the reward is "does it build",
    # the rest is quality. Additive (not multiplicative) so structural variance
    # survives — see class docstring.
    COMPILE_WEIGHT = 0.3
    CARGO_CHECK_TIMEOUT = 600

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_reward_func(self._compute_reward)

    # AST reward = location (which items the edit lives in) blended with
    # reference (which types/fields/methods the edit uses). Location says
    # "right place"; reference says "right content" — a patch that declares a
    # field in the correct struct but never wires it up scores high on location
    # and low on reference, which is exactly the signal we want.
    AST_LOCATION_WEIGHT = 0.5
    AST_REFERENCE_WEIGHT = 0.5

    async def _ast_item_f1(self, client, sandbox_id, gold_diff: str, agent_diff: str) -> tuple[float | None, float | None, float | None, dict]:
        """Returns (blended, location_f1, reference_f1, detail).

        location_f1: maps each diff's changed base-line numbers onto the FULL
          base-file tree-sitter AST (`git show HEAD:<f>`, HEAD == base_commit)
          and F1s the enclosing Rust items.
        reference_f1: F1 over the symbols (types/fields/methods, to depth) the
          ADDED code references.
        detail: the underlying item/symbol sets, reused to build the compact
          structural summary handed to the judge.
        Each component is None when its gold side is empty; blended is None only
        when BOTH are (caller then falls back to file-level — no spurious 1.0).
        """
        gold_touched = _old_side_touched_lines(gold_diff)
        agent_touched = _old_side_touched_lines(agent_diff)
        files = [f for f in (set(gold_touched) | set(agent_touched)) if f.endswith(".rs")]

        gold_items: set[str] = set()
        agent_items: set[str] = set()
        if files:
            async def _read_base(f: str) -> tuple[str, str]:
                res = await client.execute_command(
                    sandbox_id, f"git show HEAD:{shlex.quote(f)} 2>/dev/null", working_dir=WORKDIR
                )
                return f, (res.stdout or "")

            for f, content in await asyncio.gather(*(_read_base(f) for f in files)):
                if not content.strip():
                    continue
                src = content.encode("utf-8", "replace")
                gi, ai = await asyncio.gather(
                    asyncio.to_thread(_items_for_lines, src, f, gold_touched.get(f, set())),
                    asyncio.to_thread(_items_for_lines, src, f, agent_touched.get(f, set())),
                )
                gold_items |= gi
                agent_items |= ai

        location = _f1(gold_items, agent_items)
        gold_refs, agent_refs = await asyncio.gather(
            asyncio.to_thread(_referenced_symbols, gold_diff),
            asyncio.to_thread(_referenced_symbols, agent_diff),
        )
        reference = _f1(gold_refs, agent_refs)

        # Blend, renormalizing over whichever components are defined.
        parts = []
        if location is not None:
            parts.append((self.AST_LOCATION_WEIGHT, location))
        if reference is not None:
            parts.append((self.AST_REFERENCE_WEIGHT, reference))
        if not parts:
            blended = None
        else:
            wsum = sum(w for w, _ in parts)
            blended = sum(w * v for w, v in parts) / wsum
        # detail feeds the judge a compact structural diagnosis instead of the
        # raw gold diff — same facts, far fewer tokens.
        detail = {
            "gold_items": gold_items, "agent_items": agent_items,
            "gold_refs": gold_refs, "agent_refs": agent_refs,
        }
        return blended, location, reference, detail

    @staticmethod
    def _touched_packages(agent_diff: str) -> set[str]:
        """Crate names under crates/<pkg>/..., capped at 3. Empty when the diff
        touches no crate code — the caller then skips the compile gate."""
        pkgs: set[str] = set()
        for f in _files_in_diff(agent_diff):
            parts = f.split("/")
            if len(parts) > 1 and parts[0] == "crates":
                pkgs.add(parts[1])
        return set(list(pkgs)[:3])

    async def _cargo_check(self, client, sandbox_id, packages: set[str]) -> tuple[bool, str]:
        """Returns (ok, errors) where errors is a concise compiler error summary (<500 chars)."""
        for pkg in packages:
            res = await client.execute_command(
                sandbox_id,
                f"cargo check -p {pkg} --tests 2>&1",
                working_dir=WORKDIR,
                timeout=self.CARGO_CHECK_TIMEOUT,
            )
            if res.exit_code != 0:
                # Extract just `error[...]` lines to keep it concise
                lines = (res.stdout or res.stderr or "").splitlines()
                error_lines = [l for l in lines if l.strip().startswith("error")][:8]
                summary = "\n".join(error_lines)[:500]
                return False, summary
        return True, ""

    _JUDGE_SYSTEM = (
        "You are a strict Rust code reviewer. Judge ONLY two things the "
        "deterministic checks can't: correctness (does the patch's logic "
        "actually implement the task?) and completeness (are all requirements "
        "addressed?). Style, conventions and diff-tidiness are scored elsewhere "
        "— ignore them. The reference is ONE valid solution; a correct but "
        "DIFFERENT approach must score full marks. Output one JSON object only."
    )

    # Compact: task + the agent patch + a precomputed AST diagnosis vs the
    # reference (so we don't ship the whole gold diff into context). Two
    # 0-10 dimensions only.
    _JUDGE_USER_TEMPLATE = """\
## Task
{task}

## Patch to evaluate
```diff
{agent_diff}
```

## Structural analysis (AST diff vs the reference solution; for context only)
{ast_summary}{compile_line}

## Score 0-10 each
- correctness: 0-1 won't build / inverted logic; 2-3 builds, core logic wrong;
  4-5 partially correct w/ a real bug; 6-7 mostly correct, edge cases off;
  8-9 correct for all stated cases; 10 correct AND equivalent to reference.
- completeness: 0-1 none; 2-3 ≤25%; 4-5 ~50%; 6-7 ~75%; 8-9 75-99%; 10 all
  requirements addressed.

Return ONLY: {{"correctness":{{"score":<int>}},"completeness":{{"score":<int>}}}}"""

    _JUDGE_WEIGHTS = {"correctness": 0.6, "completeness": 0.4}
    # Give the judge enough context to assess completeness: the FULL task
    # (= the agent's instruction) and a generous slice of the patch. The AST
    # summary replaces the raw gold diff, which is where the big savings came
    # from — so we can afford real task + patch context.
    _JUDGE_MAX_TASK_CHARS = 3000
    _JUDGE_MAX_DIFF_CHARS = 5000

    # The judge endpoint caps concurrency; exceeding it makes calls fail. Cap
    # in-flight judge requests with a process-global semaphore. Tested judges:
    # minimax-m2 emits clean JSON (~45s); kimi reasoning models can't be made
    # to stop thinking and never emit JSON on real prompts — don't use them.
    JUDGE_CONCURRENCY = int(os.environ.get("JUDGE_CONCURRENCY", "5"))
    _judge_sem: "asyncio.Semaphore | None" = None

    @classmethod
    def _get_judge_sem(cls) -> "asyncio.Semaphore":
        if cls._judge_sem is None:
            cls._judge_sem = asyncio.Semaphore(cls.JUDGE_CONCURRENCY)
        return cls._judge_sem

    @staticmethod
    def _ast_summary(detail: dict) -> str:
        """Compact, judge-facing diagnosis from the AST item/symbol sets."""
        def short(s, n=12):
            xs = sorted(x.split("::")[-1] for x in s)
            return ", ".join(xs[:n]) + (" …" if len(xs) > n else "") if xs else "(none)"

        gi, ai = detail.get("gold_items", set()), detail.get("agent_items", set())
        gr, ar = detail.get("gold_refs", set()), detail.get("agent_refs", set())
        return (
            f"- reference edits items: {short(gi)}\n"
            f"- patch also edits:      {short(ai & gi)}  (matched)\n"
            f"- reference items MISSED by patch: {short(gi - ai)}\n"
            f"- symbols the reference uses but patch does NOT: {short(gr - ar)}"
        )

    @staticmethod
    def _judge_score_from_verdict(verdict: dict) -> float:
        """Compute weighted [0,1] score from structured verdict."""
        total, w_sum = 0.0, 0.0
        for axis, w in HyperswitchRubric._JUDGE_WEIGHTS.items():
            v = verdict.get(axis)
            if isinstance(v, dict):
                s = v.get("score")
            else:
                s = v
            if isinstance(s, (int, float)):
                total += w * float(s)
                w_sum += w
        return max(0.0, min(1.0, total / (w_sum * 10))) if w_sum > 0 else None

    async def _maybe_judge(self, state, agent_diff: str, ast_detail: dict | None = None, compile_errors: str = "") -> float | None:
        """Compact LLM judge — 2 orthogonal dims (correctness, completeness).

        Sends the task + agent patch + a precomputed AST diagnosis (NOT the raw
        gold diff) to keep context small, and forces a JSON object via
        response_format with a fixed seed for determinism. Returns a weighted
        [0,1] score, or None on any error (caller renormalizes over struct+style).
        """
        base_url = os.environ.get("JUDGE_BASE_URL", "https://grid.ai.juspay.net/v1")
        model = os.environ.get("JUDGE_MODEL", "minimaxai/minimax-m2")
        if not base_url or not model:
            return None

        info = state.get("info") or {}
        task = info.get("task_description") or state.get("question") or ""
        ast_summary = self._ast_summary(ast_detail or {})
        compile_line = (
            f"\n- cargo check FAILED: {compile_errors[:300]}" if compile_errors else ""
        )
        user = self._JUDGE_USER_TEMPLATE.format(
            task=task[: self._JUDGE_MAX_TASK_CHARS],
            agent_diff=agent_diff[: self._JUDGE_MAX_DIFF_CHARS],
            ast_summary=ast_summary,
            compile_line=compile_line,
        )

        import urllib.request

        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": self._JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "seed": 0,  # determinism across identical prompts
            # Force a valid JSON object — eliminates the reasoning-prose / malformed
            # JSON outputs that were failing ~84% of judge calls.
            "response_format": {"type": "json_object"},
            # kimi-k2-6 is a reasoning model that thinks in `content` before the
            # JSON; on a real (full-task) prompt 2048 truncated mid-reasoning
            # (finish=length, no JSON). Give it room to reason AND emit.
            "max_tokens": 4096,
        }).encode()
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('JUDGE_API_KEY', 'EMPTY')}",
            },
        )
        def _blocking_judge_call() -> dict:
            # Synchronous network I/O — MUST run off the event loop. urllib here
            # would otherwise freeze the worker's asyncio loop for the full
            # request (a reasoning judge can take >30s), starving the heartbeat
            # `stats_loop` and getting the whole worker killed + all rollouts
            # cancelled before any reward is ever returned.
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())

        try:
            async with self._get_judge_sem():  # respect the endpoint concurrency cap
                out = await asyncio.to_thread(_blocking_judge_call)
            msg = out["choices"][0]["message"]
            text = msg.get("content") or ""
            # Reasoning models may return the answer after a `reasoning` field or
            # after inline thinking prose. Extract the LAST {...} JSON object in
            # the content (the verdict), tolerating leading reasoning text.
            text = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.MULTILINE)
            verdict = None
            try:
                verdict = json.loads(text)
            except Exception:
                matches = re.findall(r"\{[^{}]*\"correctness\".*?\}\s*\}", text, re.DOTALL)
                if not matches:
                    matches = re.findall(r"\{.*\}", text, re.DOTALL)
                if matches:
                    verdict = json.loads(matches[-1])
            if verdict is None:
                raise ValueError(f"no JSON verdict in judge output: {text[:200]!r}")
            score = self._judge_score_from_verdict(verdict)
            logger.info("judge OK model=%s score=%s verdict=%s", model, score, verdict)
            return score
        except Exception as e:
            # Log loudly — a silent None here is why the judge column looks empty.
            logger.warning(
                "judge FAILED model=%s base_url=%s err=%s: %s",
                model, base_url, type(e).__name__, str(e)[:300],
            )
            return None

    async def _compute_reward(self, state, **kwargs) -> float:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        gold_patch = state.get("answer") or ""
        logger.info(
            "score_rollout START sandbox_id=%s has_client=%s",
            sandbox_id, sandbox_client is not None,
        )

        if not sandbox_client or not sandbox_id:
            logger.warning("score_rollout: no sandbox client/id -> reward 0.0")
            return 0.0

        # Stage everything first so the diff also captures NEW/untracked files
        # the agent created — plain `git diff` only shows tracked modifications,
        # which silently yields an empty diff (reward 0) when the agent's edits
        # are new files. The sandbox is ephemeral, so touching the index is safe.
        diff_res = await sandbox_client.execute_command(
            sandbox_id, "git add -A && git diff --cached HEAD", working_dir=WORKDIR
        )
        agent_diff = diff_res.stdout or ""
        logger.info("score_rollout: agent_diff len=%d", len(agent_diff))
        if not agent_diff.strip():
            # Dump the working-tree state so we can tell WHY the diff is empty:
            # agent made no edits at all vs. edits landed somewhere unexpected
            # vs. git status sees changes that `git diff` somehow didn't.
            status_res = await sandbox_client.execute_command(
                sandbox_id,
                "echo '== git status =='; git status --porcelain; "
                "echo '== HEAD =='; git rev-parse HEAD; "
                "echo '== recent mtimes =='; find . -type f -newermt '-30 minutes' "
                "-not -path './.git/*' 2>/dev/null | head -40",
                working_dir=WORKDIR,
            )
            logger.warning(
                "score_rollout: EMPTY agent diff -> reward 0.0\nWORKTREE STATE:\n%s\n%s",
                status_res.stdout or "", status_res.stderr or "",
            )
            return 0.0

        # ---- structural + style (dense, free) ----
        # file-targeting + style are cheap pure-string ops; AST localization
        # reads the base files from the sandbox (async) and parses them.
        file_f1 = file_targeting_f1(gold_patch, agent_diff)
        style = await asyncio.to_thread(check_style_compliance, agent_diff)
        ast_f1, ast_loc, ast_ref, ast_detail = await self._ast_item_f1(
            sandbox_client, sandbox_id, gold_patch, agent_diff
        )
        if ast_f1 is None:
            # AST signal undefined (gold touches no parseable Rust item, e.g. a
            # non-.rs or new file). Fall back to file-level localization only —
            # never award the old spurious 1.0 for editing unrelated files.
            structural = file_f1
        else:
            structural = 0.30 * file_f1 + 0.70 * ast_f1

        # ---- compile + judge, run CONCURRENTLY ----
        # cargo check is ~10min; the judge is ~15-45s. Running them concurrently
        # (instead of judge-after-cargo) hides the judge latency entirely under
        # the compile. They're independent: compile is its own additive reward
        # term, and the judge scores logic correctness/completeness — it doesn't
        # need the compiler errors (so it can start immediately, not wait).
        # Skip cargo (no credit, no cost) when no crate code is touched.
        packages = self._touched_packages(agent_diff)
        if packages:
            (compile_ok, compile_errors), judge = await asyncio.gather(
                self._cargo_check(sandbox_client, sandbox_id, packages),
                self._maybe_judge(state, agent_diff, ast_detail=ast_detail),
            )
        else:
            compile_ok, compile_errors = None, ""
            judge = await self._maybe_judge(state, agent_diff, ast_detail=ast_detail)
        compiles = 1.0 if compile_ok else 0.0

        if judge is None:
            total = self.W_STRUCT + self.W_STYLE
            quality = (self.W_STRUCT * structural + self.W_STYLE * style) / total
        else:
            quality = (
                self.W_STRUCT * structural
                + self.W_STYLE * style
                + self.W_JUDGE * judge
            )

        # Additive: quality keeps its full variance; compile adds a clean bonus.
        reward = (1 - self.COMPILE_WEIGHT) * quality + self.COMPILE_WEIGHT * compiles
        state["reward_breakdown"] = {
            "file_targeting_f1": round(file_f1, 4),
            "ast_item_f1": round(ast_f1, 4) if ast_f1 is not None else None,
            "ast_location_f1": round(ast_loc, 4) if ast_loc is not None else None,
            "ast_reference_f1": round(ast_ref, 4) if ast_ref is not None else None,
            "structural": round(structural, 4),
            "style": round(style, 4),
            "quality": round(quality, 4),
            "compile_ok": compile_ok,
            "judge": judge,
            "reward": round(reward, 4),
        }
        logger.info("score_rollout DONE breakdown=%s", state["reward_breakdown"])
        return float(max(0.0, min(1.0, reward)))

    @vf.cleanup
    async def cleanup_sandbox(self, state):
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if sandbox_client and sandbox_id:
            try:
                await sandbox_client.delete(sandbox_id)
            except Exception:
                pass


# ── TaskSet ────────────────────────────────────────────────────────────

class HyperswitchTaskSet(SandboxTaskSet):
    # Pre-built image: hyperswitch repo at /hyperswitch, cargo deps fetched,
    # base commit d2457784e3 already checked out.
    SANDBOX_IMAGE = "dumball/hyperswitch-rl:d2457784"

    default_workdir = WORKDIR

    def get_instruction(self, info: dict) -> str:
        task_desc = info.get("task_description", "")
        return f"Task:\n{task_desc}\n\nPlease implement a functional and style-compliant fix in the codebase."

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        return SandboxSpec(
            image=self.SANDBOX_IMAGE,
            cpu_cores=64,
            memory_gb=128,
            disk_size_gb=100,
        )

    def get_rubric(self) -> vf.Rubric:
        return HyperswitchRubric()

    async def setup(self, state) -> None:
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        info = state.get("info") or {}
        base_commit = info.get("base_sha") or "d2457784e3"
        logger.info(
            "setup START sandbox_id=%s base_commit=%s client=%s",
            sandbox_id, base_commit, type(sandbox_client).__name__,
        )

        # Use -fd (not -fdx) to preserve gitignored build artifacts like target/.
        setup_cmd = f"""
set -e
cd {WORKDIR}
git reset --hard HEAD
git clean -fd --quiet
if [ "$(git rev-parse HEAD)" != "$(git rev-parse {base_commit} 2>/dev/null)" ]; then
    git fetch --filter=blob:none origin {base_commit} 2>/dev/null || true
    git checkout {base_commit}
fi
"""
        res = await sandbox_client.execute_command(
            sandbox_id, setup_cmd, timeout=300
        )
        if res.exit_code != 0:
            raise vf.SandboxError(
                f"Failed to prepare base commit {base_commit} (exit {res.exit_code}): "
                f"{res.stderr or res.stdout}"
            )


# ── Dataset Loaders ────────────────────────────────────────────────────

def load_jsonl_dataset(file_path: str, max_examples: int | None = None) -> Dataset:
    """Load tasks from a JSONL file.

    Supports both:
    - The legacy hyperswitch JSONL format (task_description, gold_patch/gold_diff)
    - Repo2RLEnv pr_runtime / commit_runtime output format, which carries
      fail_to_pass / pass_to_pass lists in metadata for graded F2P/P2P scoring.
    """
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_examples is not None and len(examples) >= max_examples:
                break
            try:
                row = json.loads(line)
            except Exception:
                continue

            task_desc = row.get("task_description", row.get("task", row.get("prompt", "")))
            gold_patch = row.get("gold_patch", row.get("gold_diff", ""))
            metadata = row.get("metadata") or row

            # Repo2RLEnv pr_runtime / commit_runtime embed F2P/P2P lists in the
            # row under pr_runtime.fail_to_pass / pr_runtime.pass_to_pass or at
            # the top level as fail_to_pass / pass_to_pass.
            pr_runtime_meta = metadata.get("pr_runtime") or {}
            fail_to_pass: list[str] = (
                row.get("fail_to_pass")
                or pr_runtime_meta.get("fail_to_pass")
                or []
            )
            pass_to_pass: list[str] = (
                row.get("pass_to_pass")
                or pr_runtime_meta.get("pass_to_pass")
                or []
            )

            info = {
                "task_description": task_desc,
                "base_sha": str(metadata.get("base_commit") or metadata.get("base_sha") or "d2457784e3"),
                "head_sha": str(metadata.get("head_commit") or metadata.get("head_sha") or ""),
                "repo": str(metadata.get("repo") or "juspay/hyperswitch"),
                "pr_number": str(metadata.get("pr_number") or ""),
                "pr_url": str(metadata.get("pr_url") or ""),
                "fail_to_pass": fail_to_pass,
                "pass_to_pass": pass_to_pass,
            }

            examples.append({
                "question": task_desc,
                "answer": gold_patch,
                "info": info,
            })

    return Dataset.from_list(examples)


def load_harbor_dataset(
    harbor_dir: str,
    max_examples: int | None = None,
) -> Dataset:
    """Load tasks from a Repo2RLEnv Harbor task directory.

    Repo2RLEnv pipelines (pr_runtime, commit_runtime, cve_patches, …)
    emit one subdirectory per task:

        <harbor_dir>/
          <owner>__<repo>-<pr_number>/
            task.toml          — Harbor metadata + [metadata.repo2env.*]
            instruction.md     — problem statement shown to the agent
            solution/
              patch.diff       — gold patch
            tests/
              f2p.json         — FAIL_TO_PASS test-name list (pr_runtime)
              p2p.json         — PASS_TO_PASS test-name list (pr_runtime)

    The loader reads instruction.md as the task description, patch.diff as
    the gold answer, and f2p.json / p2p.json for F2P/P2P graded scoring.
    task.toml is parsed for the base_commit and pipeline metadata.
    """
    try:
        import tomllib  # stdlib ≥ 3.11
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-reattr]
        except ImportError:
            tomllib = None  # type: ignore[assignment]

    root = Path(harbor_dir)
    task_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "task.toml").exists()
    )

    examples = []
    for task_dir in task_dirs:
        if max_examples is not None and len(examples) >= max_examples:
            break

        # Parse task.toml for base_commit and pipeline metadata.
        toml_data: dict = {}
        toml_path = task_dir / "task.toml"
        if tomllib is not None:
            try:
                with open(toml_path, "rb") as fh:
                    toml_data = tomllib.load(fh)
            except Exception:
                pass

        repo2env = (toml_data.get("metadata") or {}).get("repo2env") or {}
        pipeline_name = repo2env.get("pipeline", "")

        # base_commit: look in pipeline-specific sub-table first, then top-level ref.
        pipeline_meta = repo2env.get(pipeline_name) or {}
        base_commit = (
            pipeline_meta.get("base_commit")
            or repo2env.get("ref")
            or "d2457784e3"
        )

        # Task description from instruction.md.
        instruction_path = task_dir / "instruction.md"
        task_desc = instruction_path.read_text(encoding="utf-8") if instruction_path.exists() else ""

        # Gold patch from solution/patch.diff.
        patch_path = task_dir / "solution" / "patch.diff"
        gold_patch = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""

        # F2P/P2P lists from tests/f2p.json and tests/p2p.json (pr_runtime output).
        fail_to_pass: list[str] = []
        pass_to_pass: list[str] = []
        f2p_path = task_dir / "tests" / "f2p.json"
        p2p_path = task_dir / "tests" / "p2p.json"
        if f2p_path.exists():
            try:
                fail_to_pass = json.loads(f2p_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        if p2p_path.exists():
            try:
                pass_to_pass = json.loads(p2p_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Fall back to inline metadata when json files are absent.
        if not fail_to_pass:
            fail_to_pass = pipeline_meta.get("fail_to_pass") or []
        if not pass_to_pass:
            pass_to_pass = pipeline_meta.get("pass_to_pass") or []

        task_name = task_dir.name
        pr_number = task_name.rsplit("-", 1)[-1] if "-" in task_name else ""
        repo = repo2env.get("repo") or "juspay/hyperswitch"
        reference = repo2env.get("reference") or ""

        info = {
            "task_description": task_desc,
            "base_sha": str(base_commit),
            "head_sha": "",
            "repo": str(repo),
            "pr_number": str(pr_number),
            "pr_url": str(reference),
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "pipeline": pipeline_name,
        }

        examples.append({
            "question": task_desc,
            "answer": gold_patch,
            "info": info,
        })

    return Dataset.from_list(examples)


# ── Main Load Environment Function ─────────────────────────────────────

def load_environment(
    num_train_examples: int = 1000,
    num_eval_examples: int = 100,
    curriculum_level: int = 0,
    train_dataset_path: str = "/data/training/rl_train_dataset (1).jsonl",
    eval_dataset_path: str = "/data/training/rl_eval_dataset (1).jsonl",
    # Optional: paths to Repo2RLEnv Harbor task directories.  When set these
    # override the JSONL paths for the respective split.
    train_harbor_dir: str | None = None,
    eval_harbor_dir: str | None = None,
    provider_timeout_ms: int = 300_000,
    max_turns: int = 100,
    sandbox_poll_timeout: int = 300,
    sandbox_extract_timeout: int = 300,
    # When True (default), run sandboxes as local Docker containers on this
    # host instead of Prime's managed sandbox service. We do not want a
    # remote dependency / PRIME_API_KEY in the reward loop, and this node
    # already has the dumball/hyperswitch-rl image with a warm cargo cache.
    use_local_docker_sandbox: bool = True,
    # Fixed port for the host-side interception server (proxies the agent's
    # OpenAI calls to the inference server + captures token IDs). With a local
    # Docker sandbox we set interception_url to host.docker.internal:<port> so
    # the in-container agent reaches it directly — NO prime_tunnel (that tunnel
    # is for remote sandboxes; locally it leaves the agent unable to reach
    # inference, producing empty trajectories that get filtered → eviction).
    interception_port: int = 8765,
    **kwargs,
) -> vf.Environment:
    # Training split — prefer Harbor dir when provided (richer F2P/P2P data).
    if train_harbor_dir and Path(train_harbor_dir).is_dir():
        train_ds = load_harbor_dataset(train_harbor_dir, max_examples=num_train_examples)
    else:
        if not os.path.exists(train_dataset_path):
            fallback = "/data/training/run1.full (2).jsonl"
            if os.path.exists(fallback):
                train_dataset_path = fallback
        train_ds = load_jsonl_dataset(train_dataset_path, max_examples=num_train_examples)

    # Eval split.
    if eval_harbor_dir and Path(eval_harbor_dir).is_dir():
        eval_ds = load_harbor_dataset(eval_harbor_dir, max_examples=num_eval_examples)
    else:
        eval_ds = load_jsonl_dataset(eval_dataset_path, max_examples=num_eval_examples)

    taskset = HyperswitchTaskSet(
        dataset=train_ds,
        name="hyperswitch_train",
    )
    eval_taskset = HyperswitchTaskSet(
        dataset=eval_ds,
        name="hyperswitch_eval",
    )

    system_prompt = SYSTEM_PROMPTS.get(curriculum_level, SYSTEM_PROMPTS[4])
    # Prepend the repo briefing, FADED by curriculum level: full at level 0,
    # shrinking to nothing by level 3+ so the model must internalize the
    # conventions rather than depend on the prompt.
    briefing = repo_briefing_for_level(curriculum_level)
    if briefing:
        system_prompt = briefing + system_prompt
    harness = opencode_harness(
        system_prompt=system_prompt,
        allow_git=True,
        agent_workdir=WORKDIR,
        provider_timeout_ms=provider_timeout_ms,
    )

    kwargs.setdefault("max_turns", max_turns)
    kwargs.setdefault(
        "timeouts",
        SandboxTimeouts(
            poll=sandbox_poll_timeout,
            extract=sandbox_extract_timeout,
        ),
    )

    # For a local Docker sandbox, point the agent at the host interception
    # server via host.docker.internal (no tunnel). For remote sandboxes leave
    # interception_url unset so the env establishes a prime_tunnel as usual.
    if use_local_docker_sandbox:
        kwargs.setdefault("interception_port", interception_port)
        kwargs.setdefault(
            "interception_url", f"http://host.docker.internal:{interception_port}"
        )

    env = ComposableEnv(
        taskset=taskset,
        harness=harness,
        keep_sandbox_for_scoring=True,
        eval_dataset=eval_taskset.get_dataset(),
        **kwargs,
    )

    # Replace the Prime managed-sandbox client (set by CliAgentEnv.__init__ via
    # SandboxMixin.init_sandbox_client) with our local Docker-backed client.
    # The mixin only ever calls create/wait_for_creation/execute_command/
    # delete/bulk_delete/upload_file/read_file/teardown, all of which
    # LocalDockerSandboxClient implements with the same shapes.
    if use_local_docker_sandbox:
        from local_docker_sandbox import LocalDockerSandboxClient

        try:
            env.sandbox_client.teardown()
        except Exception:
            pass
        env.sandbox_client = LocalDockerSandboxClient()

    return env
