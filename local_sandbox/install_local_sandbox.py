"""Activate local Docker sandboxes for verifiers / mini-swe-agent-plus.

Import this (or set PRIME_RL_LOCAL_SANDBOX=1 and import the sitecustomize hook)
BEFORE loading the env. It monkeypatches verifiers' ThreadedAsyncSandboxClient
to our LocalDockerSandboxClient, so any vf.SandboxEnv subclass (mini-swe-agent-plus,
opencode-swe, ...) runs each rollout in a local container instead of the paid
Prime Intellect sandbox API.

Usage:
    from local_sandbox.install_local_sandbox import enable_local_sandbox
    enable_local_sandbox()
"""
from __future__ import annotations

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)


def enable_local_sandbox() -> None:
    from local_docker_client import LocalDockerSandboxClient

    import verifiers.envs.sandbox_env as sbx_mod
    sbx_mod.ThreadedAsyncSandboxClient = LocalDockerSandboxClient

    # Some code paths import the symbol elsewhere; patch the utils module too.
    try:
        import verifiers.utils.threaded_sandbox_client as tmod
        tmod.ThreadedAsyncSandboxClient = LocalDockerSandboxClient
    except Exception:
        pass

    print("[local_sandbox] verifiers SandboxEnv now uses LocalDockerSandboxClient "
          "(local Docker, no Prime API)", flush=True)


if os.environ.get("PRIME_RL_LOCAL_SANDBOX") == "1":
    enable_local_sandbox()
