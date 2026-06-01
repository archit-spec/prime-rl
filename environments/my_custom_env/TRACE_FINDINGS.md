# Rollout Trace Findings — why rollouts score low (and what to fix)

Analysis of **1,568 saved training rollouts** (`outputs/run_default/rollouts/step_*/train_rollouts.jsonl`)
from the GLM-4.7-Flash hyperswitch RL run. The point: stop guessing, look at what
the agent actually does, and target the *real* failure modes.

## Run-wide signals (wandb, run c6ad1122, 50 logged points)

- **`stop_condition/max_turns_reached`: mean ≈ 0.45** (oscillates 0.19–0.75) —
  ~45% of rollouts hit the 80-turn ceiling. High.
- **`reward/mean`: flat ≈ 0.28** (first 0.30 → last 0.30, no trend) — no learning.
- **`reward/max`: ≈ 0.4–0.5, rarely >0.5** — even the *best* of 8 rollouts in a
  group is mediocre. The "standout success" GRPO needs to learn from is weak/
  capped, because tasks are hard/often-unsolvable, compile mostly fails, and 17%
  of completions are empty-diff give-ups. Lifting `reward/max` (curate solvable
  tasks, fix give-ups, raise compile rate) is what creates a signal to climb.

## Headline numbers

| stop_condition | n | mean reward |
|---|---|---|
| `agent_completed` | 861 | mixed (see below) |
| `max_turns_reached` | ~rest | partial |

Of the **861 `agent_completed`** rollouts:
- `>=0.4` (good): 308
- `0.15–0.4` (mid): 347
- `<0.15` (low): 61
- **`0.0` (empty diff): 145 (17%)** ← agent declared done having produced **no diff**

So ~1 in 6 "completed" rollouts edited *nothing*. `reward 0.0` ⇒ empty agent diff
(the rubric early-returns 0.0 before structural/compile/judge).

## The 145 empty-diff "completed" rollouts

```
125 (86%)  NEVER used the edit/write tool — searched, then stopped
 50 (34%)  ended on a MALFORMED tool call (broken XML: </think><tool_call>…</arg_value></tool_call>)
 20        used edit but diff stayed empty (edits didn't land / were reverted)
```

### Why they give up (final-message analysis of the 125 no-edit cases)
```
39  concluded "NO change needed" — talked itself out of the task
 9  CLAIMS done (writes a summary) but produced no diff
76  "other" — stalled mid-investigation or died on a malformed tool call
```

Representative finals:
- **"No change needed" rationalization (dominant voluntary mode):**
  *"the code already handles PartialCharged correctly… This is exactly the
  behavior the task requires!"* → analyzed, decided nothing to fix, stopped.
  But every task is a real merged PR — a change **was** required.
- **False-environment belief:** *"This dev environment lacks cargo"* — false;
  cargo (and `rg`) ARE in every container. Led to degenerate "safest path."
- **Malformed tool-call death:** final text ends `</think><tool_call>rg …</arg_value></tool_call>`
  — opencode can't parse it, ends the turn as "completed" with no edit.

## Turns & exploration — DO NOT cap turns

```
                 turns(median)  p25  p75  reads(median)
GOOD (≥0.4):         71          53   80      19
MID (0.15–0.4):      79          59   80      —
ZERO (gave up):      32          18   53       8
```

- **Winners explore *more*, not less** — 71 turns / 19 reads vs the give-ups'
  32 turns / 8 reads. Failures **under-explore then quit**; they are NOT
  over-grepping (grep/read counts are *lower* than winners). The "needs a better
  map to grep less" hypothesis is **not supported** — failures aren't grep-heavy.
- **A turn deadline (e.g. "edit by turn 3/5") would gut the winners** — 100% of
  good traces take >10 turns; p75 is pegged at 80. The existing "edit by turn 3"
  prompt rule is both ignored and harmful; **remove it.**
- **`max_turns=80` is a binding ceiling** — good *and* mid traces have p75=80
  (truncated). Headroom exists up to ~110 turns before the `seq_len=102400` cap
  (~825 tokens/turn × 80 ≈ 66k). A modest bump **80 → ~100** is safe and lets
  long-but-productive traces finish; bigger risks `seq_len` truncation (cuts off
  the final diff = catastrophic) and balloons step time. **Secondary lever.**

## What to fix, ranked by leverage

| Fix | Targets | Type |
|---|---|---|
| **Malformed tool-call parsing** for GLM in opencode (tool-parser/chat-template) | ~34% of give-ups (involuntary deaths) | harness/config — biggest mechanical win |
| **"There is always a fix" rule** — counters "no change needed" rationalization (39) | premature voluntary give-ups | system prompt |
| **Never finish on an empty diff** — `git diff` self-check before stopping | all 145 empty-diff | system prompt |
| **"Trust the environment"** — cargo/rg ARE available; don't assume tools missing | false-env-belief give-ups | system prompt |
| **Remove the "edit by turn 3" deadline; add persistence** ("hard fixes take 50–70 turns, keep going") | protects 71-turn winners | system prompt |
| `max_turns` 80 → ~100 (≤ seq_len headroom) | truncated good/mid traces | config (secondary) |
| Strip empty `<details></details>`/`<summary>` from `task_description` at load | gutted prompts | loader (3 lines) |
| Recent-commits hint + connector/co-change skills (done) | localization speed | done |

## What the logs say about stopping

The empty-diff give-ups end **`stop=agent_completed`, `exit_code=0`**, with
`agent_error=0`, `agent_timeout=0`, `sandbox_oom=0`, `sandbox_timeout=0`,
`error=None` (142/148 `is_truncated=False`). **No crash / timeout / OOM** — from
opencode's view the agent finished normally.

Worker-log stop tallies (cumulative): `agent_completed exit0` 1862,
`max_turns_reached` 1528, `has_error` 67, `prompt_too_long` 1.

Implication: even the **malformed-tool-call** cases exit 0 / `agent_completed` —
opencode **silently treats a broken/partial tool call as the agent's final text**
and ends the run, logging no error. The mechanism is therefore invisible in our
persistent logs.

**Gap:** opencode's own per-turn reasoning/stop logs are at `/opencode/logs.txt`
*inside each container*, which is deleted after scoring. To confirm the malformed
mechanism and quantify it precisely, **persist `/opencode/logs.txt` before
container deletion** (copy out in `_compute_reward`/cleanup, or `docker cp`).

## Why the reward isn't going up (full diagnosis)

wandb metrics **rule out** the common GRPO failure modes:
- `filters/zero_advantage = 0`, `is_filtered = 0` → **nothing is filtered**; every
  group contributes a gradient.
- `solve_none = 0`, `solve_all = 0` → no all-fail/all-pass groups (every group has
  reward variance — consistent with within-group std ~0.14).
- `effective_batch_size = 1.0` (full), `empty_rollouts = 0`, `errored ≈ 1.3%` →
  batches fill, rollouts produce output, almost no errors.

So the signal/plumbing is **healthy** — yet `reward/mean` is flat at 0.28. The
causes, ranked:

1. **Updates too small (LR).** Clean gradient, but at LR 1e-5 the per-step update
   is sub-threshold for a 30B policy (per-task reward flat over 37 steps; grad
   norm ~0.003). → LR 1e-4 next run. **Dominant cause.**
2. **The reward CEILING is low (`reward/max` ≈ 0.5), not 1.0.** `reward =
   0.7·quality + 0.3·compiles`; with **compile passing only ~16%** of the time,
   ~84% of rollouts forfeit the +0.3 and cap at `0.7·quality ≈ 0.49`. So even the
   *best* of 8 rollouts is mediocre → the GRPO "standout success" is weak →
   little to climb toward. **Raising compile rate lifts the whole ceiling.**
3. **~44% of tasks are infeasible** (gold >5 files / underspecified) → on those,
   no rollout can score high → drags `reward/max` down, dilutes signal.
4. **17% empty-diff give-ups + ~45% max_turns** → wasted rollouts / low-reward
   mass that holds `reward/mean` down.

**NOT the cause:** zero-advantage filtering, degeneracy, batch starvation, empty
or errored rollouts — all clean.

**What lifts reward (in order):** (a) LR 1e-4 (make updates count); (b) curate to
solvable tasks + fix give-ups (raise `reward/max`); (c) raise compile rate
(unlocks the +0.3 and the ceiling). The reward won't climb until the policy can
actually *produce* a compiling, well-localized patch on a reachable task — and be
updated enough to reinforce it.

## Notes
- These empty-diff→0.0 cases are exactly what RL *should* learn away (edit→reward,
  no-edit→0) — but only once updates are large enough (LR) and the signal is clean.
  The prompt fixes raise the floor; they compound with the LR/data fixes.
- Judge by **per-task reward trend** + **compile pass-rate** + **empty-diff rate**,
  not loss/grad-norm (see REWARD_DESIGN.md).
