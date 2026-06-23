"""SkyRL / SWE-Bench environment for prime-rl.

Agentic SWE tasks graded by the **swebench** harness (deterministic: apply the
agent's diff, run the instance's FAIL_TO_PASS/PASS_TO_PASS tests, reward =
resolved). No LLM judge. Each rollout runs in a **local Docker container** (the
SWE-Bench/SWE-Gym image for that instance) instead of the paid Prime sandbox
API.

We reuse the proven SWE agent loop + swebench grading from `mini-swe-agent-plus`
(`DeepSweSandboxEnv`, a `verifiers.SandboxEnv`), and swap its sandbox client for
`local_docker_sandbox.LocalDockerSandboxClient` by monkeypatching the symbol
that `verifiers.envs.sandbox_env.SandboxEnv` constructs — so every env worker
(including orchestrator-spawned subprocesses) uses local Docker.
"""
from __future__ import annotations

import os

import verifiers as vf


def _enable_local_docker_sandbox() -> None:
    """Point verifiers' SandboxEnv at local Docker instead of the Prime API."""
    from local_docker_sandbox import LocalDockerSandboxClient

    import verifiers.envs.sandbox_env as sbx_mod
    sbx_mod.ThreadedAsyncSandboxClient = LocalDockerSandboxClient
    try:
        import verifiers.utils.threaded_sandbox_client as tmod
        tmod.ThreadedAsyncSandboxClient = LocalDockerSandboxClient
    except Exception:
        pass


def load_environment(
    dataset_name: str = "SWE-bench/SWE-bench_Verified",
    max_turns: int = 40,
    allow_git: bool = True,
    **kwargs,
) -> vf.Environment:
    """Build the SWE env with local Docker sandboxes + swebench grading.

    Args:
        dataset_name: HF dataset of SWE instances (SWE-Bench Verified by default;
            pass a SkyRL/SWE-Gym dataset for within-distribution training).
        max_turns: max agent tool-use turns per rollout.
    """
    # make local_docker_sandbox importable when running from an installed wheel
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    _enable_local_docker_sandbox()

    from mini_swe_agent_plus.mini_swe_agent_plus import load_environment as _swe_load

    return _swe_load(
        dataset_name=dataset_name,
        max_turns=max_turns,
        allow_git=allow_git,
        **kwargs,
    )
