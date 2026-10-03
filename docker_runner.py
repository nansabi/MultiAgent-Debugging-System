"""Docker-backed command runner for untrusted repository code."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


PYTHON_IMAGE = "python:3.11.11-slim-bookworm"
CONTAINER_WORKSPACE = PurePosixPath("/workspace")


@dataclass
class DockerRunResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    container_id: str | None = None

    @property
    def returncode(self) -> int:
        return self.exit_code


class DockerRunner:
    """Execute one bounded command in a fresh container mounting one workspace."""

    def __init__(
        self,
        workspace: str | Path,
        image: str = PYTHON_IMAGE,
        memory_limit: str = "512m",
        cpu_limit: float = 1.0,
        pids_limit: int = 64,
        client=None,
    ):
        import docker

        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_dir():
            raise ValueError(f"Docker workspace does not exist: {self.workspace}")
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit
        self.client = client or docker.from_env(timeout=300)

    @staticmethod
    def check_available() -> None:
        import docker

        client = docker.from_env(timeout=5)
        try:
            client.ping()
        finally:
            client.close()

    def _ensure_image(self) -> None:
        import docker

        try:
            self.client.images.get(self.image)
        except docker.errors.ImageNotFound:
            self.client.images.pull(self.image)

    def _map_path(self, value: str) -> str:
        root = str(self.workspace).replace("\\", "/").rstrip("/")
        normalized = value.replace("\\", "/")
        if normalized.lower() == root.lower():
            mapped = str(CONTAINER_WORKSPACE)
        elif normalized.lower().startswith((root + "/").lower()):
            relative = normalized[len(root) + 1:]
            mapped = str(CONTAINER_WORKSPACE / relative)
        else:
            return value

        if mapped.lower().endswith("/.venv/scripts/python.exe"):
            return mapped[:-len("Scripts/python.exe")] + "bin/python"
        return mapped

    def _map_command(self, cmd: Sequence[str | Path]) -> list[str]:
        mapped = []
        host_python = str(Path(sys.executable)).replace("\\", "/").lower()
        container_venv_python = str(CONTAINER_WORKSPACE / ".venv" / "bin/python")
        for part in cmd:
            value = str(part)
            normalized = value.replace("\\", "/")
            if normalized.lower() == host_python:
                mapped.append("python")
            elif normalized.lower().endswith("/.venv/scripts/python.exe") or normalized.lower().endswith("/.venv/bin/python"):
                mapped.append(container_venv_python)
            elif "=" in value and value.split("=", 1)[0].startswith("--"):
                option, argument = value.split("=", 1)
                mapped.append(f"{option}={self._map_path(argument)}")
            else:
                mapped.append(self._map_path(value))
        return mapped

    def _map_cwd(self, cwd: str | Path) -> str:
        host_cwd = Path(cwd).resolve()
        try:
            relative = host_cwd.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError("Docker command cwd must be inside the mounted job workspace") from exc
        return str(CONTAINER_WORKSPACE / relative.as_posix())

    @staticmethod
    def _decode(value: bytes | str | None) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        return value or ""

    def run(
        self,
        cmd: Sequence[str | Path],
        cwd: str | Path,
        timeout: int,
        max_bytes: int | None = None,
        monitored_path: Path | None = None,
        network_enabled: bool | None = None,
    ) -> DockerRunResult:
        if isinstance(cmd, (str, bytes)):
            raise TypeError("cmd must be an argument list")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        command = self._map_command(cmd)
        if not command:
            raise ValueError("cmd must not be empty")
        self._ensure_image()
        container = self.client.containers.run(
            self.image,
            entrypoint=command[0],
            command=command[1:],
            detach=True,
            remove=False,
            volumes={str(self.workspace): {"bind": str(CONTAINER_WORKSPACE), "mode": "rw"}},
            working_dir=self._map_cwd(cwd),
            network_mode="none" if network_enabled is False else "bridge",
            mem_limit=self.memory_limit,
            nano_cpus=int(self.cpu_limit * 1_000_000_000),
            pids_limit=self.pids_limit,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            labels={"sentinel.managed": "true"},
        )
        deadline = time.monotonic() + timeout
        timed_out = False
        exit_code = 1
        try:
            while True:
                container.reload()
                if container.status != "running":
                    exit_code = int(container.attrs.get("State", {}).get("ExitCode", 1))
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    try:
                        container.kill()
                    except Exception:
                        pass
                    try:
                        container.wait(timeout=10)
                    except Exception:
                        pass
                    exit_code = 124
                    break
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

            stdout = self._decode(container.logs(stdout=True, stderr=False))
            stderr = self._decode(container.logs(stdout=False, stderr=True))
            return DockerRunResult(stdout, stderr, exit_code, timed_out, container.id)
        finally:
            try:
                container.remove(force=True, v=True)
            except Exception:
                pass