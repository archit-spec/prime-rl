# Hyperswitch RL — Reward Design: Findings & Reasoning

This doc records *why* the reward is shaped the way it is, the debugging that
led here, and the alternatives we deliberately rejected. It's written so the
next person (or us, in a month) doesn't re-litigate settled questions.

The reward lives in `HyperswitchRubric._compute_reward` (`my_custom_env.py`).

---

## TL;DR — the current reward

```
reward  = (1 - W_COMPILE)·quality + W_COMPILE·compiles      # W_COMPILE = 0.3
quality = 0.70·structural + 0.15·style + 0.15·judge         # judge optional → renormalize
structural = 0.30·file_f1 + 0.70·ast
ast        = 0.5·location_f1 + 0.5·reference_f1
```

### Judge model & serving (hard-won)

The judge contributes **correctness + completeness** (its orthogonal value) at
`W_JUDGE=0.30`. Getting a usable judge took real testing:

- **kimi-k2-6-dev is UNUSABLE as a judge.** It's a reasoning model whose chat
  template forces "thinking" in `content`. On any non-trivial prompt it burns
  the entire `max_tokens` (tested to 4096) reasoning and **never emits the JSON**
  (`finish=length`). `response_format=json_object`, `reasoning_effort=none`,
  `enable_thinking=false`, assistant-prefill, and even a "skip drafting, FULL
  SEND" anxiety prompt do **not** stop it (the anxiety prompt worked only on a
  toy prompt). → ~84% of judge calls returned None. Do not use reasoning models.
- **minimax-m2 (via `https://grid.ai.juspay.net/v1`) WORKS** — emits clean JSON
  (`finish=stop`) on real prompts, ~15–45s. Configured as the default judge.
- **The grid endpoint hard-caps at 5 concurrent** — firing 12 gave exactly 6 OK
  + 6 instant `HTTP 429` (rejected, not queued). So judge calls go through a
  process-global `Semaphore(JUDGE_CONCURRENCY=5)`. Latency cost: ~`⌈N/5⌉ ×
  per-call` per step — a real bottleneck at 64 rollouts/step; revisit if it
  dominates (judge a subset, or get a higher-concurrency fast instruct model).
- Set `JUDGE_API_KEY` (and optionally `JUDGE_BASE_URL` / `JUDGE_MODEL` /
  `JUDGE_CONCURRENCY`) in the launch env. The key is never hardcoded.
- Prompt is compact: full task + patch + the **AST diagnosis** (matched/missed
  items, missing reference symbols, compile status) instead of the raw gold
  diff. `response_format=json_object` + `seed=0` for reliability/determinism.

- **file_f1** — F1 over the *set of files* the patch changes vs gold.
- **location_f1** — F1 over the *Rust items* (fn/struct/impl/enum/trait/mod/…)
  the patch edits, found by mapping each changed line onto the **full base-file
  AST** (`git show HEAD:<path>`, HEAD == base_commit). Innermost item per line.
- **reference_f1** — F1 over the *symbols the added code references* (types,
  fields, methods, call names) extracted from the `+` lines to **full AST
  depth**, boilerplate filtered.
- **compiles** — `cargo check -p <touched crates>` passes (1.0) or not (0.0).
  **Additive**, not a multiplicative gate. Skipped (and no credit) when the
  patch touches no crate code.

---

## How we got here (the debugging story)

The run produced **0 training steps for hours**. Peeling the onion:

1. **Worker died every ~13 min** (`heartbeat timeout, restarting`). Root cause:
   the LLM judge call was a **synchronous `urllib.urlopen` on the asyncio event
   loop**. A reasoning judge takes >30s; that froze the loop, starved the
   worker's heartbeat `stats_loop`, and the router killed the worker —
   cancelling every in-flight rollout before any reward returned.
   **Fix:** run the judge (and tree-sitter parsing) via `asyncio.to_thread`.

2. **Scoring crashed on every call** for a while (`name 'difflib' is not
   defined`) after a half-applied edit. Lesson: the env module is loaded once
   at worker start — code changes need a **full run restart**, not just a file
   save.

3. **The AST reward was wrong on 57% of tasks.** The first implementation
   parsed *diff fragments* (context + added lines). For an edit **inside** an
   existing function, the `fn foo() {` declaration is far above the hunk and not
   in the fragment, so tree-sitter saw loose statements → **no item found** →
   empty set. Empty-vs-empty returned **1.0**, so a patch that edited the wrong
   files got `ast=1.0, structural=0.7` — the single most common reward (0.1911)
   was this false positive, *actively rewarding wrong-file edits*.
   **Fix:** map changed **line numbers** onto the **full base file** AST
   instead of the fragment; guard empty-gold → `None` → fall back to file_f1.

4. **Flat reward + tiny grad norm** even after the above — see the
   first-principles section.

---

## First principles: why grad norm was tiny

GRPO gradient magnitude per token ≈ `|Aᵢ · ρ_t · ∇log π_θ(a_t)|`, where the
advantage is **normalized within the group of G=8 rollouts**:
`Aᵢ = (rᵢ − mean(r)) / std(r)`. It's a product, so the smallest factor wins.

1. **`Aᵢ` ≈ 0 — within-group reward variance collapses.** Three reasons:
   - **Correlated samples.** 8 rollouts from the same policy on the same
     ~80-turn task converge to similar *outcomes* (the final diff), even at
     temp 0.7 — per-token entropy ≠ outcome diversity over a long horizon.
   - **The multiplicative compile gate compressed variance.** `reward =
     0.25·quality` when compile fails (the common case) scales the quality
     variance by `0.25² = 0.0625` — a **16× compression**.
   - **Uniformly-hard regime.** GRPO only gets signal where the policy
     *sometimes* succeeds. Most hyperswitch tasks are "all 8 produce a partial,
     non-compiling diff" → `std(r) ≈ 0` → group filtered by `zero_advantage`.
   - The old AST bug *manufactured* low variance by pinning structural to a
     false 0.7.

2. **`∇log π_θ(a_t)` is small — the policy is confident about its own
   generations.** Softmax score scales like `(1 − p)`; entropy was only
   ~0.20, i.e. tokens were sampled at high `p`, so `(1 − p)` is small. A strong
   base coder generating "typical-for-itself" trajectories has little per-token
   surprise to learn from.

3. **`ρ_t ≈ 1` — near on-policy.** `Mismatch KL ≈ 0.014` (D_KL between trainer
   and inference policies) confirms generations are essentially on-policy. This
   is *neutral* (the reference point, no off-policy amplification) and means the
   DPPO clip isn't masking tokens — but it's why early RL of a strong base model
   is gentle.

**Conclusion:** low grad norm is a **signal/variance problem, not a trainer or
LR problem.** FA3/throughput changes don't touch it. The levers are: raise
reward variance (the AST + reference + de-squash work) and, secondarily, raise
exploration (entropy/temperature) to attack both `Aᵢ` and `(1−p)`.

---

## Why each design choice (and what we rejected)

### Localization signal: AST line-mapping — not difflib, not line numbers

- **Rejected: `difflib.SequenceMatcher` over diff text** (the original).
  Textual line similarity punishes equivalent code written differently and is
  fooled by shifted line numbers / formatting. It's also O(n·m) and big diffs
  could block the event loop. Removed entirely.
- **Rejected: hunk line-range overlap (`region_overlap`).** Line-number based;
  brittle to any upstream drift between gold's base and the agent's base.
- **Rejected: parsing diff fragments with tree-sitter.** Misses in-body edits
  (57% of tasks) because the enclosing item isn't in the hunk. This was the bug.
- **Chosen: map changed line numbers onto the full base-file AST.** Both the
  agent diff (`git diff --cached HEAD`) and gold diff are against the same
  `base_commit`, so their old-side line numbers index the *same* base file —
  parse it once, map each touched line to its **innermost enclosing item**.
  Robust to formatting/line shifts; correct for in-body edits.

### Reference signal: "what the edit uses", to full depth

- **Observation that motivated it:** on the zift task, the agent added the right
  fields to the right structs (location_f1 high) but **never wired them up** —
  its added code referenced *nothing* (`reference_f1 = 0`), while gold's added
  code referenced `connector_request_reference_id`, `router_data`, …. Location
  alone said "looks right"; it was semantically incomplete.
- **Chosen:** F1 over the set of symbols (types / field accesses / method &
  call names) referenced in the `+` lines, collected to **full AST depth** (a
  chain `a.b.c.d()` contributes `b, c, d`), with a boilerplate stoplist
  (`Option`, `String`, `clone`, `self`, …).
- **Known limitation (documented, accepted):** it's a **flat set** comparison —
  depth-complete in *collecting* symbols but it does **not** encode their
  structural relationship (`a.b.c` == `c.b.a` as sets), arguments, or
  correctness. It's "right vocabulary", not "right program". This is *why we
  kept compile* (below) rather than trusting reference_f1 as the sole arbiter.
- **Blend:** `ast = 0.5·location + 0.5·reference`. Location = right place,
  reference = right content; each renormalized away when its gold side is empty.

### Compile: additive term — not a multiplicative gate, not removed

The big variance lever. With most rollouts failing compile, the form matters:

| Design | reward (all fail) | Var(reward) | Verdict |
|---|---|---|---|
| Multiplicative ×0.25 (old) | `0.25·quality` | `0.0625·Var(q)` (16× squash) | rejected — kills grad |
| "Squash less" ×0.6 | `0.6·quality` | `0.36·Var(q)` (2.8× squash) | helps, still couples |
| **Additive (w=0.3)** | `0.7·quality` | `0.49·Var(q)`, *constant* | **chosen** |

- **Rejected: keep multiplicative, just raise the factor.** Still compresses
  multiplicatively in the all-fail regime we live in, and the closer the factor
  goes to 1.0 the less compile matters. Dominated by additive.
- **Rejected: remove compile entirely** and trust the gold-anchored AST signal.
  AST/reference is set-overlap against *one* reference solution — gameable by
  symbol-dumping and **blind to validity**. Compile is the only
  *solution-agnostic* "is this real Rust" signal; the two catch orthogonal
  failure modes.
- **Chosen: additive.** Quality keeps full variance regardless of compile pass
  rate, *plus* a clean +0.3 when it builds (and `+w²·Var + 2w(1−w)·Cov` of extra
  variance when a group has a pass/fail mix). Accepted tradeoff: a
  structurally-perfect non-building patch scores 0.7 — fine at this stage, where
  we want localization to drive learning first and compile mastery to follow.

### Compile is skipped (no credit, no cost) when no crate is touched

`cargo check` is ~10 min. If the diff touches no `crates/<pkg>/` code there's
nothing to build, so we skip the check and grant no compile credit
(`compiles = 0`). We do **not** grant full quality on skip — that would create a
perverse incentive to edit non-crate files to dodge the compile term.

### `cargo test` (F2P/P2P) is NOT in the dense reward

A coarse oracle blind to structural quality (it scored a known patch 1.0 despite
a type-safety regression a reviewer would reject — see `findings_pr11372.md`).
Kept only as a periodic *eval* gate, never the dense training signal.

### Judge: optional, off the event loop

LLM merge-quality rating (4-dimension anchored rubric). Optional: returns `None`
offline and the structural+style weights renormalize. **Must** run via
`asyncio.to_thread` — a synchronous judge call on the event loop was the
original heartbeat-death root cause.

---

## Open levers (not yet pulled)

- **Entropy / temperature bump** — raise rollout outcome diversity (bigger
  `Aᵢ`) and move tokens off the saturated `p≈1` regime (bigger `∇log π`).
  Hits both grad-norm dampers at once.
- **Difficulty curriculum** — concentrate on tasks the policy solves
  *sometimes* (max variance); all-fail / all-succeed tasks waste compute.
- **Reference_f1 → structural-aware** — if symbol-set proves too gameable,
  upgrade from flat set overlap toward path/relationship-aware matching.

---

## Operational notes (hard-won)

- The env module loads **once** at worker start — reward changes require a full
  `uv run rl @ …` restart, not just a file edit.
- The worker heartbeat timeout is **30s**; never block the asyncio loop in the
  scoring path (judge, cargo, parsing all go through threads / `execute_command`).
- Reward verification ≠ "the numbers look sane" — we validated by **ranking the
  8 rollouts of one task** and confirming the genuinely-closer patch scored
  highest (`struct 0.76 @ 2/3 gold items` > `0.55 @ 1/3` > six `0.0` wrong-file).

---

## Change log — every knob we moved and why

Configs live in `examples/my_custom_env/rl_hyperswitch.toml`; sandbox spec +
reward + judge in `my_custom_env.py`; secrets in `.env` (gitignored).

### Reward shape
| Change | From → To | Why |
|---|---|---|
| Structural localization | difflib SequenceMatcher → **AST line-mapping** | text similarity punishes equivalent code; AST maps changed lines onto the full base-file tree |
| AST signal | item set only → **0.5·location + 0.5·reference** | location = "right place", reference = "uses the right types/fields/methods" (catches stubs that don't wire anything up) |
| Empty-gold guard | returned 1.0 → **None → fall back to file_f1** | killed the false 0.7/0.19 reward for editing wrong files |
| Compile gate | **×0.25 multiplicative → +0.3 additive** | multiplicative squashed within-group variance 16× (killed the gradient); additive keeps quality variance at full scale |
| Compile skip | always ran → **skip when no crate touched** | no point burning ~10min cargo on non-crate diffs |
| Quality weights | 0.55/0.15/0.30 → **0.60 struct / 0.10 style / 0.30 judge** | structural is the reliable signal; style is weak; judge restored once reliable |

### Judge
| Change | Value | Why |
|---|---|---|
| Model/endpoint | **minimax-m2 @ grid.ai.juspay.net** | kimi reasoning models never emit JSON (burn all tokens thinking) — unusable |
| Dimensions | 4 → **2 (correctness, completeness)** | the orthogonal axes; convention/hygiene duplicate style+structural |
| Prompt | full gold diff → **compact AST diagnosis** | matched/missed items + missing symbols + compile status; far fewer tokens |
| Reliability | + `response_format=json_object`, `seed=0`, `temp=0` | forces parseable JSON; temp 0 minimizes same-patch noise |
| Concurrency | **Semaphore(5)** | grid hard-rejects >5 with HTTP 429 (verified) |
| Cargo+judge | sequential → **`asyncio.gather` (concurrent)** | judge (~15-45s) hides under the ~10min cargo |
| Key loading | **`.env` auto-loader at import** | judge was 401 — nothing loaded `.env` into the worker |

### Trainer / infra
| Change | From → To | Why |
|---|---|---|
| Attention | flash_attention_2 → **flash_attention_3** | ~74% higher trainer throughput at seq 102400 (MFU 22%→38%) |
| Sandbox CPU | 4 → **64 cores** | cargo scoring was the throughput gate; host load was ~20% |
| Sandbox mem | 16 → **128 GB** | headroom for 64 parallel rustc (avoid OOM-kill → false compile-fail) |
| `disk_size_gb` | 20 → 100 | **NO-OP** — local docker client never passes it; real disk = `/var/lib/containerd` on `/` (~560G free) |
| `max_inflight_rollouts` | 32 → **64** | run all rollouts/step concurrently → more inference in flight, less trainer idle |
| Stale containers | — | clean up `r2e-rl-*` on client init + always `docker rm` by name in delete() |

### Training dynamics
| Change | From → To | Why |
|---|---|---|
| LR | 1e-6 → 5e-6 → **1e-4** | per-task reward dead flat at 1e-6/1e-5 *despite healthy advantages* — updates were sub-threshold; AdamW step ≈ LR, KL had headroom |
| Rollout temperature | 0.7 → **1.0** | more within-group outcome diversity → more reward variance → bigger advantages; also finds the occasional standout success that drives learning |
| `num_train_examples` | 256 → 8 → **32** | 8 was too small (per-step reward swung on which 4 tasks sampled, masking trend; overfit); 32 = diversity + cleaner signal at **no per-step cost** (still 4 tasks×8 rollouts/step) |
| `max_steps` | 100 → **200** | SWE RL needs a long horizon |
| `max_turns` | **80 (kept)** | deliberately not lowered |
| System prompt | "reward 4× if compiles" → **real priorities** | stale after compile went additive; now: compile, edit right items + use right symbols (no stubs), then correctness |

---

## Diagnostics & how to read this run

**Don't judge by loss or grad norm.** Loss ≈ 0 is the GRPO surrogate (meaningless).
Grad norm ~0.003 is ~scale-invariant under normalized advantages + AdamW — it
won't climb much regardless and isn't the signal.

**Judge by per-task reward trend + leading indicators.** With 37 steps at LR 1e-5
we confirmed:
- within-group reward std **0.14, 0% zero-variance groups** → the advantage
  signal is HEALTHY (reward engineering worked);
- yet per-task reward was **dead flat** (Δ≈0 on all 8 tasks) → the limit was
  **update magnitude (LR), not signal** → hence the jump to 1e-4.

**How learning bootstraps (why temp matters):** temp=1.0 makes the 8 rollouts of
a task explore widely; when one lands a genuinely good solution it's a standout
→ large positive advantage → the policy is pulled toward it. Temp *finds* the
success; LR lets the policy *move* to it.

**Leading indicators that move before the blended reward:** compile pass-rate,
`max_turns_reached` fraction (agent finishing more), file/AST F1. Watch these
over 50–200 steps. SWE/agentic-coding reward is inherently slow & jagged — a flat
early curve is normal; what's *not* acceptable is flat-at-1e-4 over ~50 steps
with KL stable (→ then look at adapter rank / task difficulty / curriculum).

**Abort/dial-back condition for LR 1e-4:** mismatch KL past ~0.05–0.1, or entropy
spike→collapse, or reward crater → halve to 5e-5.
