# Hyperswitch RL — Reward Design: Findings & Reasoning

This doc records *why* the reward is shaped the way it is, the debugging that
led here, and the alternatives we deliberately rejected. It's written so the
next person (or us, in a month) doesn't re-litigate settled questions.

The reward lives in `HyperswitchRubric._compute_reward` (`my_custom_env.py`).

---

## TL;DR — the current reward

```
reward  = (1 - W_COMPILE)·quality + W_COMPILE·compiles      # W_COMPILE = 0.3
quality = 0.55·structural + 0.15·style + 0.30·judge         # judge optional → renormalize
structural = 0.30·file_f1 + 0.70·ast
ast        = 0.5·location_f1 + 0.5·reference_f1
```

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
