"""LocalDockerSandboxClient — a drop-in replacement for prime_sandboxes /
verifiers' ThreadedAsyncSandboxClient that runs each sandbox as a local Docker
container instead of calling the paid Prime Intellect sandbox API.

This is the "adapt the env to your own sandbox implementation" path the
mini-swe-agent-plus README points at. It implements exactly the async surface
that verifiers.envs.sandbox_env.SandboxEnv / mini_swe_agent_plus use:

    create(request)                       -> Sandbox(.id, .status)
    wait_for_creation(sandbox_id, ...)    -> None
    execute_command(id, cmd, ...)         -> CommandResponse(stdout, stderr, exit_code)
    upload_file(id, remote, local, ...)   -> FileUploadResponse(success, ...)
    download_file(id, remote, local, ...) -> None
    delete(id)                            -> dict
    bulk_delete(...)                      -> object with .deleted

Backed by `docker run/exec/cp/rm`. Same SWE-Gym images SkyRL/Polar use.

Enable by monkeypatching it over ThreadedAsyncSandboxClient (see
install_local_sandbox.py) — no change to mini-swe-agent-plus source.
"""
from __future__ import annotations

import asyncio
import os
import shlex
import uuid
from dataclasses import dataclass


# --- lightweight stand-ins for prime_sandboxes.models (only fields the env reads) ---
@dataclass
class _Sandbox:
    id: str
    status: str = "RUNNING"
    docker_image: str | None = None


@dataclass
class _CommandResponse:
    stdout: str
    stderr: str
    exit_code: int
    completed: bool = True


@dataclass
class _FileUploadResponse:
    success: bool
    path: str
    size: int
    timestamp: float = 0.0


@dataclass
class _BulkDeleteResponse:
    deleted: list


@dataclass
class _BackgroundJob:
    job_id: str
    sandbox_id: str
    stdout_log_file: str
    stderr_log_file: str
    exit_file: str


@dataclass
class _BackgroundJobStatus:
    job_id: str
    completed: bool
    exit_code: int | None
    stdout: str
    stderr: str


async def _run(*args: str, input_bytes: bytes | None = None, timeout: float | None = None):
    """Run a subprocess, return (rc, stdout_bytes, stderr_bytes)."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE if input_bytes is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(input=input_bytes), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, b"", b"command timed out"
    return proc.returncode, out, err


class LocalDockerSandboxClient:
    """Async client that maps the prime-sandboxes API onto local Docker."""

    def __init__(self, *args, **kwargs):
        # Accept and ignore prime/threaded client kwargs (max_workers, max_connections, ...)
        self._containers: dict[str, str] = {}  # sandbox_id -> container name
        self._default_timeout = int(os.environ.get("LOCAL_SANDBOX_CMD_TIMEOUT", "120"))

    # ---- image resolution: map Prime/GCP image keys to local/dockerhub ----
    async def _image_local(self, image: str) -> bool:
        rc, _, _ = await _run("docker", "image", "inspect", image)
        return rc == 0

    async def _resolve_image(self, requested: str) -> str:
        # The env prefixes images with Prime's GCP registry path and mangles
        # `__` -> `_1776_`. Strip to the bare key and recover the raw instance id.
        base = requested.split("/")[-1]                          # sweb.eval.x86_64.<inst>:latest
        inst = base.replace("sweb.eval.x86_64.", "").replace(":latest", "")
        inst_raw = inst.replace("_1776_", "__")                  # sympy__sympy-19040

        # already-local images (no pull needed)
        for c in (requested, base, f"polar-swebench-runtime:{inst}"):
            if await self._image_local(c):
                return c

        # public registries (ghcr/epoch-research covers all 500 SWE-bench Verified)
        pull_candidates = [
            f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{inst_raw}",
            f"docker.io/swebench/{base}",            # official _1776_ (subset)
            f"docker.io/xingyaoww/sweb.eval.x86_64.{inst_raw.replace('__', '_s_')}",  # SWE-Gym
        ]
        for c in pull_candidates:
            if await self._image_local(c):
                return c
            rc, _, err = await _run("docker", "pull", c, timeout=1800)
            if rc == 0:
                return c
        raise RuntimeError(f"could not resolve image for {requested!r} (tried {pull_candidates})")

    # ---- lifecycle ----
    async def create(self, request) -> _Sandbox:
        requested = getattr(request, "docker_image", None) or "python:3.12-slim"
        image = await self._resolve_image(requested)
        name = f"vf-sbx-{uuid.uuid4().hex[:12]}"
        cpu = getattr(request, "cpu_cores", None) or 2
        mem = getattr(request, "memory_gb", None) or 4
        # Detached, idle container we exec into. --rm so it auto-cleans on stop.
        rc, out, err = await _run(
            "docker", "run", "-d", "--rm", "--name", name,
            "--cpus", str(cpu), "--memory", f"{mem}g",
            "--entrypoint", "/bin/sh", image, "-c", "sleep infinity",
            timeout=600,
        )
        if rc != 0:
            raise RuntimeError(f"docker run failed for {image}: {err.decode()[-400:]}")
        sid = name
        self._containers[sid] = name
        return _Sandbox(id=sid, status="RUNNING", docker_image=image)

    async def wait_for_creation(self, sandbox_id: str, max_attempts: int = 60, stability_checks: int = 1) -> None:
        # Container is already running after `docker run -d`; just verify it's up.
        for _ in range(max_attempts):
            rc, out, _ = await _run("docker", "inspect", "-f", "{{.State.Running}}", sandbox_id)
            if rc == 0 and out.decode().strip() == "true":
                return
            await asyncio.sleep(0.5)
        raise RuntimeError(f"sandbox {sandbox_id} not running")

    async def delete(self, sandbox_id: str) -> dict:
        await _run("docker", "rm", "-f", sandbox_id, timeout=60)
        self._containers.pop(sandbox_id, None)
        return {"id": sandbox_id, "status": "deleted"}

    async def bulk_delete(self, sandbox_ids=None, labels=None, **kwargs) -> _BulkDeleteResponse:
        ids = list(sandbox_ids or list(self._containers))
        for sid in ids:
            await self.delete(sid)
        return _BulkDeleteResponse(deleted=ids)

    # ---- exec ----
    async def execute_command(self, sandbox_id: str, command: str, working_dir: str | None = None,
                              env: dict | None = None, timeout: int | None = None) -> _CommandResponse:
        args = ["docker", "exec"]
        if working_dir:
            args += ["-w", working_dir]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        args += [sandbox_id, "/bin/sh", "-lc", command]
        rc, out, err = await _run(*args, timeout=timeout or self._default_timeout)
        return _CommandResponse(stdout=out.decode("utf-8", "replace"),
                                stderr=err.decode("utf-8", "replace"),
                                exit_code=rc if rc is not None else -1)

    # ---- files ----
    async def upload_file(self, sandbox_id: str, file_path: str, local_file_path: str,
                          timeout: int | None = None) -> _FileUploadResponse:
        # ensure parent dir exists in container
        parent = os.path.dirname(file_path) or "/"
        await self.execute_command(sandbox_id, f"mkdir -p {shlex.quote(parent)}")
        rc, out, err = await _run("docker", "cp", local_file_path, f"{sandbox_id}:{file_path}", timeout=timeout or 120)
        size = os.path.getsize(local_file_path) if os.path.exists(local_file_path) else 0
        if rc != 0:
            raise RuntimeError(f"docker cp upload failed: {err.decode()[-300:]}")
        return _FileUploadResponse(success=True, path=file_path, size=size)

    async def upload_bytes(self, sandbox_id: str, file_path: str, file_bytes: bytes, filename: str,
                           timeout: int | None = None) -> _FileUploadResponse:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file_bytes); tmp = tf.name
        try:
            return await self.upload_file(sandbox_id, file_path, tmp, timeout=timeout)
        finally:
            os.unlink(tmp)

    async def download_file(self, sandbox_id: str, file_path: str, local_file_path: str,
                            timeout: int | None = None) -> None:
        os.makedirs(os.path.dirname(local_file_path) or ".", exist_ok=True)
        rc, out, err = await _run("docker", "cp", f"{sandbox_id}:{file_path}", local_file_path, timeout=timeout or 120)
        if rc != 0:
            raise RuntimeError(f"docker cp download failed: {err.decode()[-300:]}")

    # ---- background jobs (long-running test runs) ----
    async def start_background_job(self, sandbox_id: str, command: str, working_dir: str | None = None,
                                   env: dict | None = None) -> _BackgroundJob:
        jid = uuid.uuid4().hex[:12]
        out_f = f"/tmp/bgjob_{jid}.out"
        err_f = f"/tmp/bgjob_{jid}.err"
        exit_f = f"/tmp/bgjob_{jid}.rc"
        cd = f"cd {shlex.quote(working_dir)}; " if working_dir else ""
        envs = "".join(f"export {k}={shlex.quote(str(v))}; " for k, v in (env or {}).items())
        # detached: run command, capture streams, write rc last (its existence = done)
        wrapped = f"{envs}{cd}( {command} ) >{out_f} 2>{err_f}; echo $? >{exit_f}"
        await _run("docker", "exec", "-d", sandbox_id, "/bin/sh", "-lc", wrapped, timeout=60)
        return _BackgroundJob(job_id=jid, sandbox_id=sandbox_id,
                              stdout_log_file=out_f, stderr_log_file=err_f, exit_file=exit_f)

    async def get_background_job(self, sandbox_id: str, job, timeout: int | None = None) -> _BackgroundJobStatus:
        exit_f = getattr(job, "exit_file", None)
        jid = getattr(job, "job_id", "")
        # done iff exit_file exists
        rc, out, _ = await _run("docker", "exec", sandbox_id, "/bin/sh", "-lc",
                                f"test -f {exit_f} && cat {exit_f} || echo __running__")
        body = out.decode().strip()
        if body == "__running__" or body == "":
            return _BackgroundJobStatus(job_id=jid, completed=False, exit_code=None, stdout="", stderr="")
        try:
            ec = int(body.splitlines()[-1])
        except Exception:
            ec = -1
        so = await self.execute_command(sandbox_id, f"cat {getattr(job,'stdout_log_file','')} 2>/dev/null")
        se = await self.execute_command(sandbox_id, f"cat {getattr(job,'stderr_log_file','')} 2>/dev/null")
        return _BackgroundJobStatus(job_id=jid, completed=True, exit_code=ec,
                                    stdout=so.stdout, stderr=se.stdout)

    # ---- lifecycle cleanup ----
    def teardown(self) -> None:
        """Sync cleanup of any remaining containers (called at env teardown)."""
        import subprocess
        for sid in list(self._containers):
            subprocess.run(["docker", "rm", "-f", sid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._containers.pop(sid, None)

    # ---- misc no-ops used by some code paths ----
    def clear_auth_cache(self) -> None:
        pass

    async def get(self, sandbox_id: str) -> _Sandbox:
        return _Sandbox(id=sandbox_id, status="RUNNING")
