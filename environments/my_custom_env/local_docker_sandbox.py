"""Local Docker-backed sandbox client.

A drop-in replacement for `prime_sandboxes.AsyncSandboxClient` /
`ThreadedAsyncSandboxClient` that runs each sandbox as a local Docker
container on this host instead of calling Prime's managed sandbox service.

We mirror exactly the surface that verifiers' ComposableEnv + SandboxMixin
call on `self.sandbox_client`:

    create(request)               -> _LocalSandbox (has .id)
    wait_for_creation(id, ...)    -> None
    execute_command(id, cmd, ...) -> _CmdResult (.stdout/.stderr/.exit_code)
    delete(id)                    -> dict
    bulk_delete(ids)              -> dict
    upload_file(id, remote, local)
    upload_bytes(id, remote, data)
    read_file(id, remote, ...)    -> _ReadResult (.content)
    run_background_job(...)       -> _CmdResult
    teardown()

Each method is async (the mixin awaits them) but the Docker work runs in a
thread pool so the event loop isn't blocked by subprocess calls.

Why local Docker instead of Prime sandboxes: this node already has the
`dumball/hyperswitch-rl:d2457784` image with the repo + warm cargo
registry, and we don't want a remote dependency / API key in the reward
loop.  One container per rollout, removed on `delete`.
"""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass(slots=True)
class _CmdResult:
    stdout: str
    stderr: str
    exit_code: int


@dataclass(slots=True)
class _ReadResult:
    content: str


@dataclass(slots=True)
class _LocalSandbox:
    id: str
    container_name: str
    docker_image: str
    status: str = "RUNNING"


@dataclass
class LocalDockerSandboxClient:
    """Runs sandboxes as local Docker containers.

    Mirrors the async AsyncSandboxClient interface used by SandboxMixin.
    """

    max_workers: int = 64
    _executor: ThreadPoolExecutor = field(init=False)
    _containers: dict[str, _LocalSandbox] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="local-docker-sandbox"
        )

    # -- internal helpers ---------------------------------------------------

    async def _run(self, args: list[str], timeout: int | None = None) -> _CmdResult:
        loop = asyncio.get_running_loop()

        def _call() -> _CmdResult:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return _CmdResult(proc.stdout, proc.stderr, proc.returncode)

        return await loop.run_in_executor(self._executor, _call)

    # -- lifecycle ----------------------------------------------------------

    async def create(self, request) -> _LocalSandbox:
        """`docker run -d` a detached container that idles until deleted."""
        sandbox_id = uuid.uuid4().hex[:16]
        container_name = f"r2e-rl-{sandbox_id}"
        image = request.docker_image
        start_command = getattr(request, "start_command", None) or "tail -f /dev/null"

        args = [
            "docker", "run", "-d",
            "--name", container_name,
            "--cpus", str(getattr(request, "cpu_cores", 4) or 4),
            "--memory", f"{int(getattr(request, 'memory_gb', 16) or 16)}g",
            # Let the in-container agent reach the host's interception server
            # (which proxies to the inference server) via host.docker.internal.
            # Without this, the env defaults to spinning a prime_tunnel meant
            # for remote sandboxes; the local container can't use it, the agent
            # gets no completions, and every rollout returns an empty trajectory.
            "--add-host", "host.docker.internal:host-gateway",
        ]
        env_vars = getattr(request, "environment_vars", None) or {}
        for k, v in env_vars.items():
            args += ["-e", f"{k}={v}"]
        if not getattr(request, "network_access", True):
            args += ["--network", "none"]
        args += [image, "bash", "-c", start_command]

        res = await self._run(args, timeout=300)
        if res.exit_code != 0:
            raise RuntimeError(
                f"docker run failed for {image} (exit {res.exit_code}): {res.stderr.strip()}"
            )
        sandbox = _LocalSandbox(
            id=sandbox_id, container_name=container_name, docker_image=image
        )
        self._containers[sandbox_id] = sandbox
        return sandbox

    async def wait_for_creation(self, sandbox_id: str, max_attempts: int = 60, **_) -> None:
        """Poll `docker inspect` until the container reports running."""
        sandbox = self._containers.get(sandbox_id)
        if sandbox is None:
            raise RuntimeError(f"unknown sandbox {sandbox_id}")
        for _attempt in range(max_attempts):
            res = await self._run(
                ["docker", "inspect", "-f", "{{.State.Running}}", sandbox.container_name],
                timeout=30,
            )
            if res.exit_code == 0 and res.stdout.strip() == "true":
                return
            await asyncio.sleep(1.0)
        raise RuntimeError(f"sandbox {sandbox_id} did not become ready")

    async def bulk_wait_for_creation(self, sandbox_ids, **_) -> None:
        await asyncio.gather(*(self.wait_for_creation(sid) for sid in sandbox_ids))

    # -- exec ---------------------------------------------------------------

    async def execute_command(
        self,
        sandbox_id: str,
        command: str,
        working_dir: str | None = None,
        env: dict | None = None,
        timeout: int | None = None,
    ) -> _CmdResult:
        sandbox = self._containers.get(sandbox_id)
        if sandbox is None:
            return _CmdResult("", f"unknown sandbox {sandbox_id}", 127)

        args = ["docker", "exec"]
        if working_dir:
            args += ["-w", working_dir]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        args += [sandbox.container_name, "bash", "-c", command]
        try:
            return await self._run(args, timeout=timeout)
        except subprocess.TimeoutExpired:
            return _CmdResult("", f"command timed out after {timeout}s", 124)

    async def run_background_job(
        self,
        sandbox_id: str,
        command: str,
        working_dir: str | None = None,
        env: dict | None = None,
        timeout: int | None = None,
        **_,
    ) -> _CmdResult:
        # Local impl: just run it synchronously (callers await the result anyway).
        return await self.execute_command(sandbox_id, command, working_dir, env, timeout)

    # -- files --------------------------------------------------------------

    async def upload_file(self, sandbox_id: str, remote_path: str, local_path: str) -> None:
        sandbox = self._containers.get(sandbox_id)
        if sandbox is None:
            raise RuntimeError(f"unknown sandbox {sandbox_id}")
        await self.execute_command(
            sandbox_id, f"mkdir -p {shlex.quote(remote_path.rsplit('/', 1)[0] or '/')}"
        )
        res = await self._run(
            ["docker", "cp", local_path, f"{sandbox.container_name}:{remote_path}"],
            timeout=120,
        )
        if res.exit_code != 0:
            raise RuntimeError(f"docker cp failed: {res.stderr.strip()}")

    async def upload_bytes(self, sandbox_id: str, remote_path: str, data: bytes) -> None:
        import base64

        b64 = base64.b64encode(data).decode()
        parent = remote_path.rsplit("/", 1)[0] or "/"
        cmd = (
            f"mkdir -p {shlex.quote(parent)} && "
            f"echo {shlex.quote(b64)} | base64 -d > {shlex.quote(remote_path)}"
        )
        res = await self.execute_command(sandbox_id, cmd, timeout=120)
        if res.exit_code != 0:
            raise RuntimeError(f"upload_bytes failed: {res.stderr.strip()}")

    async def read_file(
        self, sandbox_id: str, remote_path: str, **_
    ) -> _ReadResult:
        res = await self.execute_command(
            sandbox_id, f"cat {shlex.quote(remote_path)}", timeout=60
        )
        return _ReadResult(content=res.stdout)

    # -- teardown -----------------------------------------------------------

    async def delete(self, sandbox_id: str) -> dict:
        sandbox = self._containers.pop(sandbox_id, None)
        if sandbox is None:
            return {"status": "not_found"}
        await self._run(
            ["docker", "rm", "-f", sandbox.container_name], timeout=120
        )
        return {"status": "deleted"}

    async def bulk_delete(self, sandbox_ids, **_) -> dict:
        await asyncio.gather(*(self.delete(sid) for sid in sandbox_ids))
        return {"status": "deleted", "count": len(sandbox_ids)}

    def teardown(self) -> None:
        """Force-remove any leftover containers and shut the pool down."""
        for sandbox in list(self._containers.values()):
            subprocess.run(
                ["docker", "rm", "-f", sandbox.container_name],
                capture_output=True,
            )
        self._containers.clear()
        self._executor.shutdown(wait=False)
