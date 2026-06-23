#!/usr/bin/env bash
# Online GRPO on SWE with LOCAL Docker sandboxes (no paid Prime API).
set -e
export PRIME_RL_LOCAL_SANDBOX=1          # monkeypatch SandboxEnv -> LocalDockerSandboxClient
export PRIME_API_BASE_URL=http://localhost:0   # never hit the real API (defensive)
export HF_HOME=${HF_HOME:-/jpaypfsdata2/.cache/huggingface}
# make the local_sandbox package importable + auto-enable via sitecustomize
export PYTHONPATH="$(pwd):$(pwd)/local_sandbox:${PYTHONPATH:-}"
python -c "from local_sandbox.install_local_sandbox import enable_local_sandbox" 2>/dev/null || true
exec uv run rl @ configs/nemotron_swe_online/rl.toml "$@"
