"""Clone Python GitHub repositories and produce a pytest baseline report."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit


BASE_DIR = Path(__file__).resolve().parent
WORKSPACES_DIR = BASE_DIR / "workspaces"
MAX_REPO_BYTES = 200 * 1024 * 1024
STALE_WORKSPACE_AGE = 7 * 24 * 60 * 60
DEPENDENCY_FILES = ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg")
LOGGER = logging.getLogger("sentinel")
_ACTIVE_RUNNER = None
_ACTIVE_RUNNER_LOCK = threading.Lock()


class LocalRunner:
    """Subprocess boundary that can be replaced by a sandboxed runner later."""

    def run(self, cmd: Sequence[str | Path], cwd: Path, timeout: int,
            max_bytes: int | None = None, monitored_path: Path | None = None,
            network_enabled: bool | None = None):
        if isinstance(cmd, (str, bytes)):
            raise TypeError("cmd must be an argument list")
        command = [str(part) for part in cmd]
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=stdout_file,
                stderr=stderr_file,
                stdin=subprocess.DEVNULL,
                shell=False,
            )
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    if max_bytes is not None and monitored_path is not None:
                        size = _workspace_size(monitored_path, stop_after=max_bytes)
                        if size > max_bytes:
                            process.kill()
                            process.wait()
                            stdout, stderr = _read_process_output(stdout_file, stderr_file)
                            raise OutputSizeLimitExceeded(
                                f"Command output exceeded the {max_bytes}-byte size limit.",
                                "\n".join(part for part in (stdout, stderr) if part).strip(),
                            )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        process.kill()
                        process.wait()
                        stdout, stderr = _read_process_output(stdout_file, stderr_file)
                        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
                    time.sleep(min(0.1, remaining))
            except BaseException:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                raise
            stdout, stderr = _read_process_output(stdout_file, stderr_file)
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _read_process_output(stdout_file, stderr_file) -> tuple[str, str]:
    stdout_file.seek(0)
    stderr_file.seek(0)
    stdout = stdout_file.read().decode("utf-8", "replace")
    stderr = stderr_file.read().decode("utf-8", "replace")
    return stdout, stderr


class OutputSizeLimitExceeded(RuntimeError):
    def __init__(self, message: str, output: str = ""):
        super().__init__(message)
        self.output = output


class CommandFailure(RuntimeError):
    def __init__(self, message: str, output: str = "", returncode: int | None = None,
                 timed_out: bool = False):
        super().__init__(message)
        self.output = output
        self.returncode = returncode
        self.timed_out = timed_out


def validate_github_url(url: str) -> bool:
    """Accept only canonical HTTPS GitHub owner/repository URLs."""
    if not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url)
        path = parsed.path
        return (
            parsed.scheme == "https"
            and parsed.netloc == "github.com"
            and parsed.username is None
            and parsed.password is None
            and parsed.port is None
            and not parsed.query
            and not parsed.fragment
            and re.fullmatch(r"/[A-Za-z0-9-]+/[A-Za-z0-9_.-]+(?:\.git)?", path) is not None
            and path.split("/")[-1].removesuffix(".git") not in {".", ".."}
        )
    except ValueError:
        return False


def _command_output(result) -> str:
    return "\n".join(part for part in (result.stdout, result.stderr) if part).strip()


def _execute(runner, cmd, cwd: Path, timeout: int, log_path: Path | None = None,
             max_bytes: int | None = None, monitored_path: Path | None = None,
             network_enabled: bool | None = None) -> str:
    try:
        options = {}
        if max_bytes is not None:
            options.update(max_bytes=max_bytes, monitored_path=monitored_path)
        if network_enabled is not None:
            options["network_enabled"] = network_enabled
        result = runner.run(cmd, cwd=cwd, timeout=timeout, **options)
    except OutputSizeLimitExceeded as exc:
        if log_path:
            _write_log(log_path, exc.output)
        limit_mib = max_bytes / (1024 * 1024) if max_bytes else 0
        raise CommandFailure(
            f"Repository exceeded Sentinel's {limit_mib:g} MiB checkout size limit.", exc.output
        ) from exc
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        output = "\n".join(part for part in (stdout, stderr) if part).strip()
        if log_path:
            _write_log(log_path, output)
        raise CommandFailure(f"Command timed out after {timeout}s: {cmd[0]}", output, timed_out=True) from exc
    output = _command_output(result)
    if log_path:
        _write_log(log_path, output)
    if getattr(result, "timed_out", False):
        raise CommandFailure(f"Command timed out after {timeout}s: {cmd[0]}", output, timed_out=True)
    if result.returncode:
        raise CommandFailure(f"Command exited with status {result.returncode}: {cmd[0]}", output,
                             returncode=result.returncode)
    return output


def _write_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(text + "\n")


def _log_file_tail(path: Path, label: str) -> str:
    try:
        contents = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        contents = ""
    return f"{label}:\n{contents[-12000:]}" if contents else f"{label}: no command output."


def _workspace_size(path: Path, stop_after: int | None = None) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            item = Path(root) / name
            try:
                if item.is_file():
                    total += item.stat().st_size
                    if stop_after is not None and total > stop_after:
                        return total
            except OSError:
                continue
    return total


def cleanup_old_workspaces(workspaces_dir: Path = WORKSPACES_DIR) -> None:
    """Remove only Sentinel-owned workspaces older than seven days."""
    if not workspaces_dir.exists():
        return
    cutoff = time.time() - STALE_WORKSPACE_AGE
    for workspace in workspaces_dir.iterdir():
        marker = workspace / ".sentinel-managed"
        try:
            if workspace.is_dir() and marker.is_file() and workspace.stat().st_mtime < cutoff:
                shutil.rmtree(workspace)
        except OSError:
            continue


def clone_repo(url: str, workspace_dir: str | Path, runner=None) -> Path:
    if not validate_github_url(url):
        raise ValueError("URL must be https://github.com/<owner>/<repo> (optional .git).")

    workspace = Path(workspace_dir)
    repo_path = workspace / "repo"
    if repo_path.exists():
        raise FileExistsError(f"Repository destination already exists: {repo_path}")
    repo_path.mkdir(parents=True)
    log_path = workspace / "logs" / "clone.log"
    try:
        _execute(runner or LocalRunner(), ["git", "clone", "--depth", "1", url, str(repo_path)],
                 cwd=workspace, timeout=60, log_path=log_path,
                 max_bytes=MAX_REPO_BYTES, monitored_path=repo_path)
    except CommandFailure as exc:
        shutil.rmtree(repo_path, ignore_errors=True)
        if exc.timed_out:
            raise CommandFailure(
                "This repository is too large or the connection is slow. Try a smaller repo or increase the timeout.",
                output=exc.output,
            ) from exc
        raise
    size = _workspace_size(repo_path)
    if size > MAX_REPO_BYTES:
        shutil.rmtree(repo_path, ignore_errors=True)
        raise CommandFailure(f"Cloned repository exceeds the 200 MB size limit ({size} bytes).")
    return repo_path


def detect_stack(repo_path: str | Path) -> dict:
    root = Path(repo_path)
    dependencies = [name for name in DEPENDENCY_FILES if (root / name).is_file()]
    excluded_source_dirs = {
        ".git", ".venv", "venv", "env", "site-packages", "__pycache__",
        ".pytest_cache", "node_modules",
    }
    has_python_source = False
    if root.is_dir():
        for current, directories, files in os.walk(root):
            directories[:] = [
                name for name in directories
                if name not in excluded_source_dirs
            ]
            if any(name.endswith(".py") for name in files):
                has_python_source = True
                break
    if not dependencies and not has_python_source:
        return {
            "language": "unsupported",
            "test_framework": None,
            "dependency_files": [],
            "test_command": None,
            "message": "Unsupported repository: no root Python dependency or project file was found.",
        }

    pytest_markers = ("pytest.ini", "tox.ini", "conftest.py")
    has_pytest = any((root / name).exists() for name in pytest_markers)
    has_pytest = has_pytest or (root / "tests").is_dir()
    has_pytest = has_pytest or any(root.glob("test_*.py"))
    return {
        "language": "python",
        "test_framework": "pytest" if has_pytest else None,
        "dependency_files": dependencies,
        "test_command": "python -m pytest --tb=short -q" if has_pytest else None,
        "verification_mode": "test_verified" if has_pytest else "none",
        **({} if has_pytest else {"message": "Python project detected, but no pytest indicators were found."}),
    }


def _runner_prefers_linux_venv(runner=None) -> bool:
    """Return True when the active runner expects a Linux-style .venv layout."""
    if runner is None:
        return get_runner_status().get("mode") == "docker"
    return type(runner).__name__ == "DockerRunner"


def _venv_python(venv_path: Path, runner=None) -> Path:
    linux_python = venv_path / "bin/python"
    if _runner_prefers_linux_venv(runner):
        return linux_python
    if linux_python.exists():
        return linux_python
    windows_python = venv_path / "Scripts/python.exe"
    if windows_python.exists():
        return windows_python
    return linux_python if sys.platform != "win32" else windows_python


def _ensure_venv_ready(workspace: str | Path, runner=None) -> Path:
    """Repair or create the workspace venv so the active runner can execute it."""
    workspace_path = Path(workspace).resolve()
    venv_path = workspace_path / ".venv"
    command_runner = _execution_runner(workspace_path, runner)
    if _runner_prefers_linux_venv(command_runner):
        linux_python = venv_path / "bin/python"
        if linux_python.exists():
            return linux_python
        windows_python = venv_path / "Scripts/python.exe"
        if windows_python.exists():
            shutil.rmtree(venv_path, ignore_errors=True)
        log_path = workspace_path / "logs" / "venv-create.log"
        _execute(command_runner,
                 [sys.executable, "-m", "venv", "--without-pip", str(venv_path)],
                 cwd=workspace_path, timeout=120, log_path=log_path, network_enabled=False)
        return linux_python if linux_python.exists() else _venv_python(venv_path, runner=command_runner)

    python = _venv_python(venv_path, runner=command_runner)
    if python.exists():
        return python

    log_path = workspace_path / "logs" / "venv-create.log"
    _execute(command_runner,
             [sys.executable, "-m", "venv", "--without-pip", str(venv_path)],
             cwd=workspace_path, timeout=120, log_path=log_path, network_enabled=False)
    return _venv_python(venv_path, runner=command_runner)


def configure_runner() -> dict:
    """Select the configured execution runner, falling back safely on startup."""
    global _ACTIVE_RUNNER
    requested = os.environ.get("SENTINEL_RUNNER", "docker").strip().lower()
    with _ACTIVE_RUNNER_LOCK:
        if requested == "local":
            _ACTIVE_RUNNER = "local"
            LOGGER.warning("LocalRunner selected - UNSAFE for untrusted repos")
        elif requested == "docker":
            try:
                from docker_runner import DockerRunner

                DockerRunner.check_available()
                _ACTIVE_RUNNER = "docker"
            except Exception as exc:
                _ACTIVE_RUNNER = "local"
                LOGGER.warning(
                    "Docker not available - falling back to LocalRunner (UNSAFE for untrusted repos): %s",
                    exc,
                )
        else:
            _ACTIVE_RUNNER = "local"
            LOGGER.warning(
                "Unknown SENTINEL_RUNNER=%r - falling back to LocalRunner (UNSAFE for untrusted repos)",
                requested,
            )
    return get_runner_status()


def get_runner_status() -> dict:
    mode = _ACTIVE_RUNNER or os.environ.get("SENTINEL_RUNNER", "docker").strip().lower()
    if mode == "docker":
        return {"mode": "docker", "label": "Docker isolated", "unsafe": False}
    return {"mode": "local", "label": "Local-unsafe", "unsafe": True}


def _execution_runner(workspace: Path, runner=None):
    if runner is not None:
        return runner
    if _ACTIVE_RUNNER is None:
        configure_runner()
    if _ACTIVE_RUNNER == "docker":
        from docker_runner import DockerRunner

        return DockerRunner(workspace)
    return LocalRunner()


def install_dependencies(repo_path: str | Path, runner=None) -> dict:
    repo = Path(repo_path)
    workspace = repo.parent
    venv_path = workspace / ".venv"
    log_path = workspace / "logs" / "install.log"
    command_runner = _execution_runner(workspace, runner)
    deadline = time.monotonic() + 300

    # Phase 2 fix: ensure the workspace venv is available in the layout expected by
    # the active runner before we try to install packages.
    python = _ensure_venv_ready(workspace, runner=command_runner)
    detected = detect_stack(repo)
    dependency_files = detected["dependency_files"]

    def pip_install(arguments: list[str]) -> str:
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise CommandFailure("Dependency installation exceeded its 5-minute timeout.")
        return _execute(command_runner,
                        [sys.executable, "-m", "pip", "--python", python, "install", *arguments],
                        cwd=repo, timeout=remaining, log_path=log_path, network_enabled=True)

    pip_install(["pip"])
    project_installed = False
    for name in dependency_files:
        if name == "requirements.txt":
            pip_install(["-r", repo / name])
        elif name in {"pyproject.toml", "setup.py", "setup.cfg"}:
            if not project_installed:
                pip_install(["."])
                project_installed = True
    pip_install(["pytest"])
    return {"venv_path": str(venv_path), "python": str(python), "log_path": str(log_path)}


def parse_junit_report(xml_text: str, duration: float = 0.0) -> dict:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse pytest JUnit XML: {exc}") from exc

    cases = root.findall(".//testcase")
    failures = []
    failed_count = 0
    error_count = 0
    skipped_count = 0
    for case in cases:
        issue = case.find("failure")
        issue_tag = "failure"
        if issue is None:
            issue = case.find("error")
            issue_tag = "error"
        if issue is None:
            if case.find("skipped") is not None:
                skipped_count += 1
            continue

        if issue_tag == "failure":
            failed_count += 1
        else:
            error_count += 1
        traceback = (issue.text or "").strip()
        file_name = case.get("file")
        line = None
        line_match = re.search(r"([^\s:]+\.py):(\d+)(?::|\b)", traceback)
        if line_match:
            file_name = file_name or line_match.group(1)
            line = int(line_match.group(2))
        elif case.get("line"):
            try:
                line = int(case.get("line"))
            except ValueError:
                line = None
        message = issue.get("message") or (traceback.splitlines()[0] if traceback else "")
        error_type = issue.get("type")
        if not error_type:
            if issue_tag == "failure":
                type_match = re.search(r"(?:E\s+)?([A-Za-z_]\w*(?:Error|Exception)):", traceback)
                error_type = type_match.group(1) if type_match else (
                    "AssertionError" if message.lstrip().startswith("assert") else "TestFailure"
                )
            else:
                error_type = issue_tag
        failures.append({
            "test_id": f"{case.get('classname', '')}::{case.get('name', '')}".lstrip(":"),
            "file": file_name,
            "line": line,
            "error_type": error_type,
            "message": message,
            "traceback": traceback,
        })

    total = len(cases)
    if not cases:
        total = int(root.get("tests", "0"))
        failed_count = int(root.get("failures", "0"))
        error_count = int(root.get("errors", "0"))
    return {
        "total": total,
        "passed": max(0, total - failed_count - error_count - skipped_count),
        "failed": failed_count,
        "errors": error_count,
        "skipped": skipped_count,
        "failures": failures,
        "duration": round(float(duration), 3),
    }


def run_baseline_tests(repo_path: str | Path, runner=None, timeout: int = 120) -> dict:
    repo = Path(repo_path)
    workspace = repo.parent
    python = _ensure_venv_ready(workspace, runner=runner)
    xml_path = workspace / "pytest-report.xml"
    log_path = workspace / "logs" / "pytest.log"
    start = time.monotonic()
    try:
        _execute(_execution_runner(workspace, runner),
                 [python, "-m", "pytest", "--tb=short", "-q", f"--junitxml={xml_path}"],
             cwd=repo, timeout=timeout, log_path=log_path, network_enabled=False)
    except CommandFailure as exc:
        if "timed out" not in str(exc):
            output = xml_path.read_text(encoding="utf-8") if xml_path.exists() else ""
            if not output:
                raise
            report = parse_junit_report(output, time.monotonic() - start)
            report["pytest_xml"] = output
            if not report["failures"]:
                message = next((line.strip() for line in exc.output.splitlines() if line.strip()), str(exc))
                report["errors"] += 1
                report["total"] = max(1, report["total"])
                report["passed"] = max(0, report["total"] - report["failed"] - report["errors"] - report["skipped"])
                report["failures"].append({
                    "test_id": "<pytest collection>",
                    "file": None,
                    "line": None,
                    "error_type": "PytestExitError",
                    "message": message,
                    "traceback": exc.output,
                })
                report["pytest_exit_error"] = str(exc)
            return report
        timeout_output = exc.output or str(exc)
        timeout_xml = ET.Element("testsuites", {"tests": "1", "errors": "1"})
        suite = ET.SubElement(timeout_xml, "testsuite", {"name": "pytest", "tests": "1", "errors": "1"})
        case = ET.SubElement(suite, "testcase", {"classname": "pytest", "name": "<test run>"})
        ET.SubElement(case, "error", {"type": "TimeoutExpired", "message": str(exc)}).text = timeout_output
        output = ET.tostring(timeout_xml, encoding="unicode")
        xml_path.write_text(output, encoding="utf-8")
        report = parse_junit_report(output, time.monotonic() - start)
        report["pytest_xml"] = output
        return report

    output = xml_path.read_text(encoding="utf-8") if xml_path.exists() else ""
    if not output:
        raise CommandFailure("pytest completed without producing a JUnit XML report.")
    report = parse_junit_report(output, time.monotonic() - start)
    report["pytest_xml"] = output
    return report


def _job_guard(lock):
    return lock if lock is not None else nullcontext()


def _log(job: dict, message: str, lock=None) -> None:
    with _job_guard(lock):
        job.setdefault("logs", []).append(message)
        if job.get("steps"):
            job["steps"][-1].setdefault("logs", []).append(message)


def _begin_stage(job: dict, stage: str, lock=None) -> None:
    with _job_guard(lock):
        job["stage"] = stage
        job["status"] = "running"
        job.setdefault("steps", []).append({"stage": stage, "status": "running", "logs": []})


def _finish_stage(job: dict, status: str = "completed", lock=None) -> None:
    with _job_guard(lock):
        if job.get("steps"):
            job["steps"][-1]["status"] = status


def run_intake(url: str, job: dict | None = None, job_lock=None, runner=None) -> dict:
    """Run the full Phase 1 intake and persist the resulting baseline report."""
    active_job = job if job is not None else {
        "job_id": uuid.uuid4().hex[:12], "stage": "cloning", "status": "running", "logs": [], "result": None,
    }
    job_id = active_job["job_id"]
    workspace = WORKSPACES_DIR / job_id
    clone_runner = runner or LocalRunner()
    try:
        if not validate_github_url(url):
            raise ValueError("URL must be https://github.com/<owner>/<repo> (optional .git).")
        cleanup_old_workspaces()
        workspace.mkdir(parents=True, exist_ok=False)
        (workspace / ".sentinel-managed").touch()

        _begin_stage(active_job, "cloning", job_lock)
        repo_path = clone_repo(url, workspace, clone_runner)
        _log(active_job, _log_file_tail(workspace / "logs" / "clone.log", "Clone output"), job_lock)
        _log(active_job, f"Cloned repository into {repo_path}.", job_lock)
        _finish_stage(active_job, lock=job_lock)

        _begin_stage(active_job, "detecting", job_lock)
        stack = detect_stack(repo_path)
        _log(active_job, f"Detected stack: {json.dumps(stack, sort_keys=True)}", job_lock)
        _finish_stage(active_job, lock=job_lock)
        if stack["language"] == "unsupported":
            raise ValueError(stack.get("message", "Repository does not have a supported pytest test suite."))

        if stack["verification_mode"] == "none":
            baseline = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "failures": [],
                "job_id": job_id,
                "repository": url,
                "stack": stack,
                "verification_mode": "none",
                "pytest_xml": "",
                "message": (
                    "Python project detected without pytest indicators. "
                    "No dependencies were installed and no project code was executed. "
                    "Describe an issue to request an unverified fix."
                ),
            }
            report_path = workspace / "baseline_report.json"
            report_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
            _log(active_job, baseline["message"], job_lock)
            _log(active_job, f"Intake report written to {report_path}.", job_lock)
            with _job_guard(job_lock):
                active_job["stage"] = "done"
                active_job["status"] = "done"
                active_job["result"] = baseline
            return active_job

        _begin_stage(active_job, "installing", job_lock)
        execution_runner = _execution_runner(workspace, runner)
        installation = install_dependencies(repo_path, execution_runner)
        _log(active_job, _log_file_tail(Path(installation["log_path"]), "Dependency installation output"), job_lock)
        _log(active_job, f"Dependencies installed. Full output: {installation['log_path']}", job_lock)
        _finish_stage(active_job, lock=job_lock)

        _begin_stage(active_job, "testing", job_lock)
        baseline = run_baseline_tests(repo_path, execution_runner)
        _log(active_job, _log_file_tail(workspace / "logs" / "pytest.log", "Pytest output"), job_lock)
        baseline.update({
            "job_id": job_id,
            "repository": url,
            "stack": stack,
            "verification_mode": "test_verified",
            "pytest_xml_path": str(workspace / "pytest-report.xml"),
        })
        report_path = workspace / "baseline_report.json"
        report_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        _log(active_job, f"Baseline report written to {report_path}.", job_lock)
        _finish_stage(active_job, lock=job_lock)

        with _job_guard(job_lock):
            active_job["stage"] = "done"
            active_job["status"] = "done"
            active_job["result"] = baseline
        return active_job
    except Exception as exc:
        if isinstance(exc, CommandFailure) and exc.output:
            _log(active_job, exc.output[-12000:], job_lock)
        _log(active_job, f"{type(exc).__name__}: {exc}", job_lock)
        with _job_guard(job_lock):
            active_job["stage"] = "failed"
            active_job["status"] = "failed"
            active_job["result"] = {"error": str(exc)}
            if active_job.get("steps"):
                active_job["steps"][-1]["status"] = "failed"
        return active_job
