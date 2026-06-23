"""Auto-enable local Docker sandboxes in every Python process when
PRIME_RL_LOCAL_SANDBOX=1. Picked up at interpreter startup because prime-rl's
root is on PYTHONPATH (see scripts/run_swe_online_local.sh), so it also applies
inside env-worker subprocesses spawned by the orchestrator.
"""
import os

if os.environ.get("PRIME_RL_LOCAL_SANDBOX") == "1":
    try:
        import sys
        _here = os.path.dirname(os.path.abspath(__file__))
        for p in (_here, os.path.join(_here, "local_sandbox")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from local_sandbox.install_local_sandbox import enable_local_sandbox
        enable_local_sandbox()
    except Exception as e:  # never break interpreter startup
        import sys
        print(f"[sitecustomize] local sandbox enable failed: {e}", file=sys.stderr)
