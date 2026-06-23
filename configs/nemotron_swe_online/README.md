# Nemotron-30B online GRPO on SWE — local sandboxes

Online (async) GRPO counterpart to the offline reinforce/Polar rig.

## Run
    bash scripts/run_swe_online_local.sh

This sets `PRIME_RL_LOCAL_SANDBOX=1`, which monkeypatches verifiers' SandboxEnv to
`local_sandbox/local_docker_client.py:LocalDockerSandboxClient` — each rollout runs in a
local `docker run` container (reusing the SWE-Gym images), no paid Prime sandbox API.

## Curriculum
`[orchestrator.buffer] easy_threshold=1.0 hard_threshold=0.0` drops zero-variance
groups (all-solved / all-failed). Online => moving learnability frontier.

## SkyRL-293 instead of SWE-Bench-Verified
The env grades via `swebench.harness`, which lacks SWE-Gym/SkyRL instances (same gap we
hit in Polar). To train on SkyRL-293 here you must wire the `swegym` fork into the env's
grading + image keys (see lib note). The dataset is exported at
`/data/datasets/skyrl-293/` for that wiring. Out-of-the-box this config uses
SWE-Bench-Verified-Quick, which the env supports natively.
