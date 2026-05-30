# Xyne SEBI RL Handover

Last updated: 2026-05-22 (UTC)

## Current Operational State

- Unified runtime environment target: `.venv-runtime`
- Pinned dependencies file: `requirements.txt`
- Runtime bootstrap script: `scripts/setup_runtime_env.sh`
- Main launcher and shell helpers auto-detect `.venv-runtime/bin/python`
- Root real-time metric files:
  - `logs/near_online/<ts>/training_metrics.jsonl`
  - `logs/near_online/<ts>/eval_metrics.jsonl`
- Runtime API path in this setup: `http://localhost:23000/v2`.
- SEBI subagent id in use: `nj5ednjmpfx9qt56s6wiocu6`.
- Agent cookie file: `data/cookie_agents.txt` (workspace `g0xufd0v4fcvjqa02x4uror1`).
- Inference model label expected by rollouts: `gemma-4-rl`.
- Near-online training mode is segmented merge+restart (Gemma4 runtime LoRA hot-load is not supported in this vLLM build).
- GRPO geometry is now aligned to the intended outcome-only agent training loop:
  - `per_device_train_batch_size=2`
  - `generation_batch_size=40`
  - `group_size=8`
  - `steps_per_generation=1` after GRPOConfig resolution
  - one optimizer step = `5 unique prompts x 8 rollouts`
- On 6 training GPUs, the launcher can auto-adjust to:
  - `generation_batch_size=48`
  - one optimizer step = `6 unique prompts x 8 rollouts`
- Judge reward path now uses adaptive multi-call aggregation in training:
  - 3 judge calls in parallel initially
  - escalate to 7 total if the first 3 have wide score spread or too few valid parses
  - final reward is `median3` or `trimmed_mean7`

## What Is Implemented

- Rollouts use `POST /v2/chat/conversations/{id}/messages` with `agentId`.
- Final answer extraction uses latest assistant message text blocks from v2 messages (not intermediate stream fragments).
- Rollout logging now preserves prompt-group identity and extraction contract:
  - `prompt_group_id`
  - `rollout_index`
  - `episode_id`
  - `raw_assistant_tail`
  - `extraction_status`, `extraction_length`, `leaked_tool_or_thinking`
- Rollout lifecycle logs now include:
  - `rollout_start`
  - `rollout_progress`
  - `rollout_episode`
- Judge logs include:
  - structured and salvaged scoring paths
  - `reward_source` classification
  - raw judge output in `judge_raw_events.jsonl`
  - ensemble metadata (`judge_escalated`, `judge_aggregate_method`, `judge_aggregate_score`)
- Training logs include progress metrics and ETA:
  - `global_step`, `max_steps`, `%complete`
  - `learning_rate`, `loss`
  - `grad_norm`
  - `reward_mean`, `reward_std`, `success_rate`
  - `elapsed_sec`, `eta_sec`
  - `rollout_latency_p50/p95`, `judge_latency_p50/p95`
  - `step_duration_sec`, `rollout_batch_duration_sec`, `judge_batch_duration_sec`, `backward_estimated_sec`
  - per-step prompt-group summary

## Unified Run Layout (Single Folder Per Run)

Canonical path: `logs/near_online/<timestamp>/`

- `orchestrator.log`
- `progress.log`
- `vllm/` (`base.log`, `after_segment_*.log`)
- `xyne/` (`pre_segment_a.log`, `post_segment_*.log`)
- `training/segment_*.log`
- `merge/segment_*.log`
- `rl_runs/segment_*/*`
- `coverage.json` per segment under `rl_runs/segment_*`
- `rollup/segment_*.json`, `rollup/run_rollup.json`

## Standard Run Flow

1. Start vLLM on base model (GPUs `0,1`).
2. Preflight checks (`/v2/me`, dataset keys, vLLM health).
3. Xyne probe before training.
4. Segment A train.
5. Merge adapter A and restart vLLM on merged A.
6. Xyne probe after segment A.
7. Segment B train.
8. Merge adapter B and restart vLLM on merged B.
9. Additional segments repeat the same pattern if configured.
10. Final Xyne probe and rollup generation.

## Quick Commands

Initialize the runtime environment:

```bash
cd /data/RL-Training/RL-Training
./scripts/setup_runtime_env.sh
source .venv-runtime/bin/activate
```

```bash
cd /data/RL-Training/RL-Training
tmux new -s rl-near-online './scripts/run_near_online_segmented.sh'
```

For full-dataset training, do not rely on inline env vars with `tmux new-session`. Use:

```bash
cd /data/RL-Training/RL-Training
tmux new -s rl-full './scripts/run_full_merged_training.sh'
```

This avoids the tmux env handoff issue that previously caused `SEGMENT_PLAN`, `TRAIN_EPOCHS`, and `MAX_SAMPLES` to fall back to defaults.

Current safest long-run launch:

```bash
cd /data/RL-Training/RL-Training
tmux new -s rl-full 'FULL_DISABLE_EVAL=1 ./scripts/run_full_merged_training.sh'
```

Why:
- full-data epoch-mode is now verified
- in-trainer reward eval caused a multi-rank desync at epoch end
- no-eval training avoids that path
- eval can be done later as a separate pass

Curriculum launcher:

```bash
cd /data/RL-Training/RL-Training
tmux new -s rl-curriculum './scripts/run_full_training_curriculum.sh'
```

Default curriculum:
- stage 1: `group_size=8`, `3` epochs
- stage 2: `group_size=8`, `5` epochs from stage 1 merged model
- stage 3: `group_size=16`, `3` epochs exists but is disabled by default because it doubles judge cost

```bash
tail -f /data/RL-Training/RL-Training/logs/near_online/latest/orchestrator.log
tail -f /data/RL-Training/RL-Training/logs/near_online/latest/run_manifest.json
tail -f /data/RL-Training/RL-Training/logs/near_online/latest/progress.log
tail -f /data/RL-Training/RL-Training/logs/near_online/latest/training_metrics.jsonl
tail -f /data/RL-Training/RL-Training/logs/near_online/latest/eval_metrics.jsonl
```

## Known Constraints

- `trl==1.4.0` + `vllm==0.19.1` is used as-is.
- Retry logs from judge client can appear and are expected transiently.
- If Xyne returns no assistant final text, episode is marked failure (`final_answer_missing`) and logged explicitly.
- Judge latency can still be high because `private-large` is queue-backed; the ensemble improves stability of reward quality, not raw backend latency.
- The current stack is still custom-rollout based. TRL’s native vLLM integration version warning is expected because generation is served externally through Xyne + local vLLM, not through TRL’s built-in server path.
- Reward flow is explicitly pushed to W&B from the callback now.
- `grad_norm` is still under live verification in the distributed GRPO path; check the latest canary before approving a long run.
- The newest full-run naming convention is intentionally descriptive:
  - `noeval_g8_e3_mb1_<timestamp>`
  - `noeval_g8_e5_mb1_<timestamp>`
  - read this as `eval mode / group size / epochs / train microbatch`

## Judge Reward Policy

- Accepted judge outcomes for reward:
  - `judge_structured_ok`
  - `judge_salvaged_text`
- Dropped judge outcomes:
  - `judge_request_failed`
  - unrecoverable parse failures
- Aggregation rules:
  - first 3 valid scores stable -> reward is median of those 3
  - if initial range exceeds `0.25`, or fewer than 2 valid initial scores -> make 4 more calls
  - after escalation -> trimmed mean after dropping global min/max when enough valid scores exist
  - if escalation still leaves too few valid scores -> sample is dropped from reward path

## Readiness Gates

- Geometry gate:
  - a `5`-sample smoke with `generation_batch_size=40` must show exactly `5` unique `prompt_group_id`s and exactly `8` rollouts per group
- Extraction gate:
  - `final_answer_present` must be true for every successful rollout
  - no assistant tool/thinking leakage in extracted final answer
- Judge gate:
  - valid judged outputs at least `99%`
  - dropped samples at most `1%`
- Segment gate:
  - every segment must train, merge, restart vLLM, and pass a post-restart Xyne probe

## Judge Consistency Verification

- Launcher: `scripts/run_judge_consistency.sh`
- This harness now supports:
  - parallel rollouts
  - parallel judge calls
  - optional simulation of the production aggregation policy
- Key knobs:
  - `ROLLOUT_PARALLELISM`
  - `JUDGE_PARALLELISM`
  - `JUDGE_BATCH_PARALLELISM`
  - `SIMULATE_PRODUCTION_AGGREGATION=1`
