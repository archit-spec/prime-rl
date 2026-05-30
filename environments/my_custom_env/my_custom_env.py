"""
A multi-turn, curriculum-based coding environment for Juspay Hyperswitch.

Loads real PR and commit tasks from a JSONL dataset or a Repo2RLEnv-generated
Harbor task directory.  Manages sandboxed execution using a pre-built Docker
image (dumball/hyperswitch-rl:d2457784) that has the hyperswitch repo at
/hyperswitch.

Reward — we train for *mergeable* patches, not merely test-passing ones.
The dense reward is compile-gated and structurally weighted:

    reward = compile_factor × ( w_struct·structural
                              + w_style ·style
                              + w_judge ·judge )

    compile_factor = 1.0   if `cargo check -p <touched crates>` passes
                   = 0.25  otherwise           (close-but-doesn't-build)

    structural = 0.45·file_targeting_F1   (right files?)
               + 0.35·region_overlap      (right lines/region?)
               + 0.20·diff_similarity     (right change shape?)

    style = Rust style checker (no unwrap/panic/unsafe/dbg/as-cast)
    judge = optional Haiku merge-quality rating; when no API key is
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

import difflib
import logging
import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from datasets import Dataset

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
    "EXECUTION DISCIPLINE — ACT EARLY, ITERATE FAST:\n"
    "- You have a STRICT, SMALL turn budget. You MUST make at least one `edit` within your first 3 turns — a partial edit that compiles beats no edit.\n"
    "- Spend at most 1-2 turns exploring before your first edit. Skim, don't audit.\n"
    "- Identify the smallest file(s) you must change, then `edit` immediately. Refine later if needed.\n"
    "- After every edit, run `cargo check -p <package>` and fix compile errors before continuing. Never stack edits without verifying.\n"
    "- Do not duplicate `use` imports or `impl` blocks. Search the file for an existing import/impl before adding a new one.\n"
    "- Stop exploring once you have what you need. Long file reads or repeated `ls`/`grep` are wasted turns.\n"
    "- NEVER end your turn without having written an edit to disk. An empty diff scores ZERO.\n"
    "- Your reward depends on a compiling, test-passing diff. An imperfect edit that compiles beats a long investigation that doesn't.\n\n"
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

# ── Repo2RLEnv: Diff-Similarity Reward (inlined from reward.py) ────────
# Source: https://github.com/huggingface/Repo2RLEnv/blob/main/src/repo2rlenv/reward.py
# License: Apache-2.0
# Concept inspired by SWE-RL (Wei et al., NeurIPS '25, arXiv:2502.18449);
# this is an independent reimplementation using Python's stdlib difflib.

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


def calculate_diff_similarity_reward(
    oracle_diff: str, predicted_diff: str
) -> tuple[float, DiffRewardMetadata]:
    """Score a predicted diff against an oracle diff.

    Returns (reward, metadata) where reward ∈ [0, 1]:
      - 1.0  if normalized diffs are identical
      - 0.0  if predicted_diff is empty or unparseable
      - else difflib.SequenceMatcher ratio over normalized lines
    """
    if not predicted_diff.strip():
        return 0.0, DiffRewardMetadata(0.0, 0, 0, 0, "empty prediction")

    oracle_lines = _normalize_diff(oracle_diff)
    pred_lines = _normalize_diff(predicted_diff)

    if not oracle_lines:
        return 0.0, DiffRewardMetadata(
            0.0, len(pred_lines), 0, 0, "empty oracle after normalization"
        )

    matcher = difflib.SequenceMatcher(a=oracle_lines, b=pred_lines, autojunk=False)
    ratio = matcher.ratio()
    matched = sum(triple.size for triple in matcher.get_matching_blocks())

    return ratio, DiffRewardMetadata(
        similarity=ratio,
        pred_lines=len(pred_lines),
        oracle_lines=len(oracle_lines),
        matched_lines=matched,
    )


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
    """Compile-gated, structurally-weighted reward for mergeable patches.

        reward = compile_factor × ( 0.55·structural + 0.15·style + 0.30·judge )

    where structural = 0.45·file_targeting_F1 + 0.35·region_overlap
                     + 0.20·diff_similarity, and compile_factor is 1.0 when
    `cargo check` on the touched crates passes else COMPILE_FAIL_FACTOR.

    The LLM judge is optional: when no API key is configured it returns
    None and the remaining (structural+style) weights are renormalized, so
    the env runs fully offline.  `cargo test` F2P/P2P is NOT in the dense
    reward — it is a coarse oracle blind to structural quality
    (findings_pr11372.md §2.4) and belongs in a periodic eval gate.
    """

    # Weights for the quality term.
    W_STRUCT = 0.55
    W_STYLE = 0.15
    W_JUDGE = 0.30
    # Partial credit for a structurally-good patch that doesn't compile —
    # a near-miss should beat a no-op, but never out-score a building patch.
    COMPILE_FAIL_FACTOR = 0.25
    CARGO_CHECK_TIMEOUT = 600

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_reward_func(self._compute_reward)

    @staticmethod
    def _touched_packages(agent_diff: str) -> set[str]:
        """Crate names under crates/<pkg>/..., capped at 3; defaults to router."""
        pkgs: set[str] = set()
        for f in _files_in_diff(agent_diff):
            parts = f.split("/")
            if len(parts) > 1 and parts[0] == "crates":
                pkgs.add(parts[1])
        if not pkgs:
            pkgs.add("router")
        return set(list(pkgs)[:3])

    async def _cargo_check(self, client, sandbox_id, packages: set[str]) -> bool:
        """True iff `cargo check -p <pkg> --tests` passes for every package.

        keep_sandbox_for_scoring=True means the agent's own builds during
        the rollout already warmed target/, so this is usually fast.
        """
        for pkg in packages:
            res = await client.execute_command(
                sandbox_id,
                f"cargo check -p {pkg} --tests",
                working_dir=WORKDIR,
                timeout=self.CARGO_CHECK_TIMEOUT,
            )
            if res.exit_code != 0:
                return False
        return True

    async def _maybe_judge(self, state, agent_diff: str) -> float | None:
        """Optional LLM merge-quality judge against an OpenAI-compatible
        endpoint (e.g. our internally-hosted GLM-5 / Kimi-2.5).

        Configured entirely by env vars so the env stays offline-by-default:
          JUDGE_BASE_URL  — e.g. http://<host>:8000/v1   (unset ⇒ disabled)
          JUDGE_MODEL     — e.g. zai-org/GLM-5-FP8 or moonshotai/Kimi-2.5
          JUDGE_API_KEY   — bearer token if the endpoint needs one (else "EMPTY")

        Returns a merge-quality score in [0,1], or None when disabled / on any
        error — None makes score_rollout renormalize over structural+style so
        a judge outage never zeros a real patch (findings_pr11372.md: judge is
        the biggest reward-hacking surface, so it's a backstop, not the spine).
        """
        base_url = os.environ.get("JUDGE_BASE_URL")
        model = os.environ.get("JUDGE_MODEL")
        if not base_url or not model:
            return None

        gold_patch = state.get("answer") or ""
        info = state.get("info") or {}
        task = info.get("task_description") or state.get("question") or ""
        prompt = (
            "You are a senior Rust reviewer for the juspay/hyperswitch payments "
            "monorepo. Rate whether the CANDIDATE patch is MERGE-QUALITY: it must "
            "resolve the task, be correctly localized, minimal, and follow "
            "hyperswitch conventions (no unwrap/panic/unsafe/as-cast, prefer typed "
            "wrappers over stringly-typed keys, respect v1/v2 cfg split).\n\n"
            f"# Task\n{task[:2000]}\n\n"
            f"# Reference (gold) patch\n{gold_patch[:6000]}\n\n"
            f"# Candidate patch\n{agent_diff[:6000]}\n\n"
            "Reply with ONLY a float in [0,1]: 1.0 = indistinguishable from a "
            "merge-ready patch, 0.0 = wrong or unmergeable. No other text."
        )

        import urllib.request

        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 8,
        }).encode()
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('JUDGE_API_KEY', 'EMPTY')}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                out = json.loads(resp.read())
            text = out["choices"][0]["message"]["content"]
            m = re.search(r"[01](?:\.\d+)?", text)
            return max(0.0, min(1.0, float(m.group(0)))) if m else None
        except Exception:
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

        # ---- structural closeness to gold (dense, free) ----
        f1 = file_targeting_f1(gold_patch, agent_diff)
        region = region_overlap(gold_patch, agent_diff)
        sim, _ = calculate_diff_similarity_reward(gold_patch, agent_diff)
        structural = 0.45 * f1 + 0.35 * region + 0.20 * sim

        # ---- style (merge discipline) ----
        style = check_style_compliance(agent_diff)

        # ---- compile gate ----
        packages = self._touched_packages(agent_diff)
        compile_ok = await self._cargo_check(sandbox_client, sandbox_id, packages)
        compile_factor = 1.0 if compile_ok else self.COMPILE_FAIL_FACTOR

        # ---- optional judge (None offline → renormalize) ----
        judge = await self._maybe_judge(state, agent_diff)
        if judge is None:
            total = self.W_STRUCT + self.W_STYLE
            quality = (self.W_STRUCT * structural + self.W_STYLE * style) / total
        else:
            quality = (
                self.W_STRUCT * structural
                + self.W_STYLE * style
                + self.W_JUDGE * judge
            )

        reward = compile_factor * quality
        state["reward_breakdown"] = {
            "file_targeting_f1": round(f1, 4),
            "region_overlap": round(region, 4),
            "diff_similarity": round(sim, 4),
            "structural": round(structural, 4),
            "style": round(style, 4),
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
            cpu_cores=4,
            memory_gb=16,
            disk_size_gb=20,
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
