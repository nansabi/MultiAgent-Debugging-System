"""
repo_fix_pipeline.py — Phase 3 of Sentinel v2.

Given a job's baseline_report.json failure entry, this module:
  1. Locates the actual project source file(s) the failing test exercises.
  2. Builds the source_code/test_code inputs the existing Orchestrator.py expects.
  3. Runs the full Diagnosis -> Fix -> Mutation pipeline via Orchestrator.py (subprocess).
  4. Applies the accepted patch back into the job workspace (NEVER outside it).
  5. Re-runs the full baseline suite to confirm no regressions.

Orchestrator.py and simple_mutation_test.py are invoked as-is; this module only
controls what files get passed to them.
"""

from __future__ import annotations

import ast
import difflib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from repo_intake import (
    WORKSPACES_DIR,
    _ensure_venv_ready,
    detect_stack,
    run_baseline_tests,
    _execution_runner,
)

BASE_DIR = Path(__file__).resolve().parent
ORCHESTRATOR = BASE_DIR / "Orchestrator.py"
ORCHESTRATORS = {
    "custom": BASE_DIR / "Orchestrator.py",
    "crewai": BASE_DIR / "crewai_orchestrator.py",
    "langgraph": BASE_DIR / "langgraph_orchestrator.py",
}


def _configured_orchestrator() -> tuple[str, Path]:
    """Return the configured orchestrator, defaulting to the proven custom one."""
    name = os.environ.get("SENTINEL_ORCHESTRATOR", "custom").strip().lower()
    try:
        return name, ORCHESTRATORS[name]
    except KeyError as exc:
        supported = ", ".join(ORCHESTRATORS)
        raise ValueError(
            f"Unsupported SENTINEL_ORCHESTRATOR={name!r}; choose {supported}."
        ) from exc


def _build_orchestrator_command(
    name: str,
    script: str | Path,
    python_executable: str,
    source: str | Path,
    tests: str | Path,
    source_root: str | Path,
    source_files: list[str | Path],
) -> list[str]:
    command = [
        python_executable, "-X", "utf8", str(script),
        "--source", str(source),
        "--tests", str(tests),
    ]
    if name == "langgraph":
        command.extend(["--source-root", str(source_root)])
        for source_file in source_files:
            command.extend(["--source-file", str(source_file)])
    return command

# ---------------------------------------------------------------------------
# In-memory fix-job store (same pattern as REPO_JOBS in app.py)
# ---------------------------------------------------------------------------
FIX_JOBS: dict[str, dict] = {}
FIX_JOBS_LOCK = threading.Lock()


class AuditRecorder:
    """Persist a truthful, workspace-scoped audit trail for a fix job."""

    def __init__(self, job: dict[str, Any], workspace_root: str | Path):
        self.workspace_root = Path(workspace_root).resolve()
        self.job = job
        self.audit_path = self.workspace_root / "fix_audit.json"
        self.audit = {
            "fix_job_id": job.get("fix_job_id"),
            "intake_job_id": job.get("intake_job_id"),
            "failure_index": job.get("failure_index"),
            "verification_mode": job.get("verification_mode", "test_verified"),
            "status": job.get("status", "running"),
            "stage": job.get("stage", "locating"),
            "started_at": self._utc_now(),
            "updated_at": self._utc_now(),
            "completed_at": None,
            "repo_metadata": {
                "workspace_root": self._relative_to_workspace(self.workspace_root),
                "project_root": None,
            },
            "source_candidates": [],
            "chosen_source": None,
            "diagnosis": None,
            "mutation_verdict": None,
            "diff": "",
            "suite_passed": None,
            "retest_report": None,
            "commands": [],
            "files": {"read": [], "written": [], "patched": []},
            "security": {
                "workspace_boundary_ok": True,
                "absolute_paths_removed": [],
            },
            "logs": [],
            "error": None,
            "final_verdict": None,
        }

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _relative_to_workspace(self, value: str | Path | None) -> str | None:
        if value is None:
            return None
        path = Path(value).resolve()
        try:
            return str(path.relative_to(self.workspace_root))
        except ValueError:
            return str(path)

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, Path):
            return self._relative_to_workspace(value)
        if isinstance(value, str):
            candidate = value
            candidate = re.sub(r'(?i)[A-Za-z]:\\[^\\\r\n]+', '<abs-path>', candidate)
            candidate = re.sub(r'(?i)/workspace/[^ \r\n]+', '<sandbox-path>', candidate)
            candidate = re.sub(r'(?i)http\+docker://[^ \r\n]+', '<docker-api>', candidate)

            workspace_root_str = str(self.workspace_root)
            if workspace_root_str in candidate:
                candidate = candidate.replace(workspace_root_str, "")
                candidate = candidate.replace("\\", "/")
                candidate = candidate.lstrip("/")
                if candidate.startswith("."):
                    candidate = candidate.lstrip(".")
                return candidate or "."

            p = Path(value)
            if p.is_absolute():
                try:
                    p.relative_to(self.workspace_root)
                    return self._relative_to_workspace(p)
                except ValueError:
                    return p.name or "<external>"
            return candidate
        if isinstance(value, list):
            return [self._sanitize(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._sanitize(v) for k, v in value.items()}
        return value

    def record(self, **payload: Any) -> dict[str, Any]:
        for key, value in payload.items():
            if key in {"stage", "status"}:
                self.audit[key] = value
            else:
                self.audit[key] = self._sanitize(value)
        self.audit["updated_at"] = self._utc_now()
        return self.audit

    def persist(self) -> dict[str, Any]:
        payload = json.loads(json.dumps(self.audit, default=str))
        self.audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.job["audit"] = payload
        self.job["audit_path"] = str(self.audit_path)
        return payload


# ---------------------------------------------------------------------------
# locate_source_for_failure
# ---------------------------------------------------------------------------

def locate_source_for_failure(
    failure: dict,
    project_root: str | Path,
) -> list[dict]:
    """Identify project source file(s) likely responsible for a test failure.

    Searches three signals in ranked order:
      1. Traceback frames — file paths inside project_root, outside venv/site-packages.
      2. Import statements in the test file.
      3. Single top-level package heuristic.

    Returns a list of dicts (highest-confidence first), each::

        {"path": str, "confidence": "high"|"medium"|"low", "reason": str}

    Returns an empty list when no project source can be identified.
    """
    root = Path(project_root).resolve()
    candidates: list[dict] = []
    seen: set[Path] = set()

    def _add(p: Path, confidence: str, reason: str) -> None:
        p = p.resolve()
        if p not in seen and p.is_file() and p.suffix == ".py":
            seen.add(p)
            candidates.append({"path": str(p), "confidence": confidence, "reason": reason})

    def _is_project_path(p: Path) -> bool:
        try:
            p.resolve().relative_to(root)
        except ValueError:
            return False
        skip = {".venv", "venv", "env", "site-packages", "__pycache__", ".git"}
        return not any(part in skip for part in p.parts)

    # --- Signal 1: traceback frames ---
    traceback_text = failure.get("traceback", "") or ""
    frame_re = re.compile(r'([\w./ \\-]+\.py)(?::(\d+))?', re.MULTILINE)
    test_file_name = Path(failure.get("file", "")).name if failure.get("file") else ""
    for m in frame_re.finditer(traceback_text):
        raw = m.group(1).strip()
        # Try absolute path first, then relative to project_root, then BASE_DIR
        for base in (Path("/"), root, BASE_DIR):
            candidate_path = (base / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
            if candidate_path.exists() and _is_project_path(candidate_path):
                if test_file_name and candidate_path.name == test_file_name:
                    continue  # skip the test file itself
                _add(candidate_path, "high", f"traceback frame points to {raw}")
                break

    # --- Signal 2: imports in the test file ---
    test_file_rel = failure.get("file", "")
    if test_file_rel:
        for base in (root, BASE_DIR, root.parent):
            abs_test = (base / test_file_rel).resolve()
            if abs_test.exists():
                break
        else:
            abs_test = None
        if abs_test and abs_test.exists():
            try:
                source = abs_test.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source)
            except (OSError, SyntaxError):
                tree = None
            if tree:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            _try_import_path(root, alias.name,
                                             "medium",
                                             f"imported as 'import {alias.name}'",
                                             _add)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        _try_import_path(root, node.module,
                                         "medium",
                                         f"imported as 'from {node.module} import ...'",
                                         _add)

    # --- Signal 3: single top-level package heuristic ---
    if not candidates:
        packages = [
            p for p in root.iterdir()
            if p.is_dir()
            and (p / "__init__.py").exists()
            and p.name not in {".venv", "venv", "env", ".git", "__pycache__"}
        ]
        if len(packages) == 1:
            pkg_init = packages[0] / "__init__.py"
            _add(pkg_init, "low",
                 f"only one top-level package: {packages[0].name}/")

    return candidates


def locate_source_for_description(
    description: str,
    project_root: str | Path,
    suspected_file: str | None = None,
) -> list[dict]:
    """Rank repository Python files by terms from a user-reported issue."""
    root = Path(project_root).resolve()
    skipped_dirs = {
        ".git", ".venv", "venv", "env", "site-packages", "__pycache__",
        ".pytest_cache", "node_modules", "build", "dist",
    }
    if suspected_file:
        relative = Path(suspected_file)
        if relative.is_absolute():
            raise ValueError("suspected_file must be a repository-relative path.")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("suspected_file must point inside the repository.") from exc
        if not candidate.is_file() or candidate.suffix.lower() != ".py":
            raise ValueError("suspected_file must identify an existing Python source file.")
        if any(part in skipped_dirs for part in candidate.relative_to(root).parts):
            raise ValueError("suspected_file cannot point into a generated or dependency directory.")
        return [{
            "path": str(candidate),
            "confidence": "high",
            "reason": "explicitly selected by the user",
            "score": 1,
        }]

    stop_words = {
        "about", "after", "again", "also", "and", "are", "bug", "but", "can",
        "cannot", "does", "file", "from", "have", "into", "issue", "its",
        "not", "only", "please", "should", "that", "the", "then", "there",
        "this", "through", "when", "where", "which", "with", "would",
    }
    terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", description)
        if term.lower() not in stop_words
    }
    if not terms:
        raise ValueError(
            "The issue description has no searchable terms. Add more detail or specify suspected_file."
        )

    ranked: list[dict] = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if any(part in skipped_dirs for part in relative.parts):
            continue
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            continue
        try:
            if path.stat().st_size > 1_000_000:
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        path_terms = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", relative.as_posix().lower()))
        content_terms = Counter(
            term.lower()
            for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", source)
        )
        path_matches = terms & path_terms
        content_matches = terms & content_terms.keys()
        score = 5 * len(path_matches) + sum(
            min(content_terms[term], 4) for term in content_matches
        )
        if score:
            ranked.append({
                "path": str(path.resolve()),
                "confidence": "high" if path_matches else "medium",
                "reason": (
                    "description terms matched source path and content"
                    if path_matches else "description terms matched source content"
                ),
                "score": score,
                "matched_terms": sorted(path_matches | content_matches),
            })

    ranked.sort(key=lambda item: (-item["score"], item["path"].lower()))
    return ranked


def _try_import_path(root: Path, module_name: str, confidence: str,
                     reason: str, add_fn) -> None:
    """Resolve a dotted module name to a .py file under root and call add_fn."""
    rel = Path(*module_name.split("."))
    for candidate in (root / rel / "__init__.py", root / rel.with_suffix(".py")):
        if candidate.exists():
            add_fn(candidate, confidence, reason)
            return


def _expand_imported_source_candidates(
    candidates: list[dict],
    project_root: str | Path,
) -> list[dict]:
    """Add project-local Python dependencies imported by the initial candidates."""
    root = Path(project_root).resolve()
    expanded = list(candidates)
    known = {Path(item["path"]).resolve() for item in expanded}
    pending = [Path(item["path"]).resolve() for item in expanded]

    while pending:
        importer = pending.pop(0)
        try:
            tree = ast.parse(importer.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue

        for node in ast.walk(tree):
            module_names: list[str] = []
            if isinstance(node, ast.Import):
                module_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if node.level:
                    relative = importer.relative_to(root).with_suffix("")
                    package = ".".join(relative.parts[:-1])
                    if not package:
                        continue
                    try:
                        module_name = importlib.util.resolve_name(
                            "." * node.level + module_name, package
                        )
                    except ImportError:
                        continue
                if module_name:
                    module_names.append(module_name)

            for module_name in module_names:
                rel = Path(*module_name.split("."))
                resolved = next(
                    (
                        path.resolve()
                        for path in (
                            root / rel / "__init__.py",
                            root / rel.with_suffix(".py"),
                        )
                        if path.is_file()
                    ),
                    None,
                )
                if resolved is None or resolved in known:
                    continue
                try:
                    resolved.relative_to(root)
                except ValueError:
                    continue
                known.add(resolved)
                pending.append(resolved)
                expanded.append({
                    "path": str(resolved),
                    "confidence": "medium",
                    "reason": f"imported by {importer.relative_to(root).as_posix()}",
                })

    return expanded


def _unified_diff_text(before: str, after: str, fromfile: str, tofile: str) -> str:
    """Build a readable unified diff even when either file lacks a final newline."""
    before_lines = [f"{line}\n" for line in before.splitlines()]
    after_lines = [f"{line}\n" for line in after.splitlines()]
    return "".join(difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=fromfile,
        tofile=tofile,
    ))


# ---------------------------------------------------------------------------
# build_fix_job
# ---------------------------------------------------------------------------

def build_fix_job(
    failure: dict,
    test_file_path: str | Path,
    source_file_path: str | Path,
) -> dict:
    """Read real file content and build the input dict the orchestrator expects.

    Returns dict with keys: source_code, test_code, source_path, test_path, failure.
    Raises FileNotFoundError if either file is missing.
    """
    src = Path(source_file_path)
    tst = Path(test_file_path)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    if not tst.exists():
        raise FileNotFoundError(f"Test file not found: {tst}")
    return {
        "source_code": src.read_text(encoding="utf-8", errors="replace"),
        "test_code": tst.read_text(encoding="utf-8", errors="replace"),
        "source_path": src,
        "test_path": tst,
        "failure": failure,
    }


def _stage_review_project(
    workspace: str | Path,
    project_root: str | Path,
    fix_job_id: str,
    source_path: str | Path,
    test_path: str | Path,
) -> tuple[Path, Path]:
    """Copy the project into an isolated review tree and return staged source/test paths."""
    workspace_path = Path(workspace).resolve()
    project_path = Path(project_root).resolve()
    source = Path(source_path).resolve()
    test = Path(test_path).resolve()
    staged_root = workspace_path / ".review" / fix_job_id / "repo"

    source_relative = source.relative_to(project_path)
    test_relative = test.relative_to(project_path)
    shutil.copytree(
        project_path,
        staged_root,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
        ),
    )
    return staged_root / source_relative, staged_root / test_relative


# ---------------------------------------------------------------------------
# apply_fix_to_workspace
# ---------------------------------------------------------------------------

def apply_fix_to_workspace(
    job_id: str,
    source_file_path: str | Path,
    patched_code: str,
    runner=None,
) -> dict:
    """Write patched_code to source_file_path (must be inside the job workspace).

    Safety check: raises ValueError if source_file_path is outside the workspace.
    Then re-runs the FULL baseline suite.

    Returns dict: {written, baseline, suite_passed, diff}
    """
    workspace = (WORKSPACES_DIR / job_id).resolve()
    src = Path(source_file_path).resolve()
    try:
        src.relative_to(workspace)
    except ValueError:
        raise ValueError(
            f"Security violation: attempted to write {src} which is outside "
            f"the job workspace {workspace}. Patch aborted."
        )

    original_code = src.read_text(encoding="utf-8", errors="replace")
    diff = _unified_diff_text(
        original_code,
        patched_code,
        fromfile=f"{src.name} (before)",
        tofile=f"{src.name} (after)",
    )

    src.write_text(patched_code, encoding="utf-8")

    repo_path = workspace / "repo"
    # Use the explicitly-passed runner; only fall back to _execution_runner when
    # no runner was provided. This lets test code and the fixture setup pass
    # LocalRunner() directly without triggering the Docker path.
    if runner is not None:
        execution_runner = runner
    else:
        execution_runner = _execution_runner(workspace, runner)
    _ensure_venv_ready(workspace, runner=execution_runner)
    baseline = run_baseline_tests(repo_path, runner=execution_runner)
    passed = baseline.get("failed", 1) == 0 and baseline.get("errors", 1) == 0

    return {
        "written": True,
        "baseline": baseline,
        "suite_passed": passed,
        "diff": diff,
    }


def apply_fixes_to_workspace(
    job_id: str,
    proposed_files: dict[str, str],
    runner=None,
) -> dict[str, Any]:
    """Apply a set of repository-relative patches, then run the baseline suite."""
    workspace = (WORKSPACES_DIR / job_id).resolve()
    repo_path = (workspace / "repo").resolve()
    if not proposed_files:
        raise ValueError("No proposed files were supplied for approval.")

    resolved_files: list[tuple[Path, str]] = []
    for relative_path, patched_code in proposed_files.items():
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ValueError(f"Patch path must be relative to the repository: {relative_path}")
        source = (repo_path / relative).resolve()
        try:
            source.relative_to(repo_path)
        except ValueError as exc:
            raise ValueError(
                f"Security violation: attempted to write {source} outside {repo_path}."
            ) from exc
        if not source.is_file():
            raise FileNotFoundError(f"Patch source file not found: {source}")
        if not isinstance(patched_code, str):
            raise ValueError(f"Patch contents for {relative_path} must be text.")
        resolved_files.append((source, patched_code))

    diffs = []
    for source, patched_code in resolved_files:
        original_code = source.read_text(encoding="utf-8", errors="replace")
        diffs.append(_unified_diff_text(
            original_code,
            patched_code,
            fromfile=f"{source.relative_to(repo_path).as_posix()} (before)",
            tofile=f"{source.relative_to(repo_path).as_posix()} (after)",
        ))
    for source, patched_code in resolved_files:
        source.write_text(patched_code, encoding="utf-8")

    if runner is not None:
        execution_runner = runner
    else:
        execution_runner = _execution_runner(workspace, runner)
    _ensure_venv_ready(workspace, runner=execution_runner)
    baseline = run_baseline_tests(repo_path, runner=execution_runner)
    passed = baseline.get("failed", 1) == 0 and baseline.get("errors", 1) == 0
    return {
        "written": True,
        "baseline": baseline,
        "suite_passed": passed,
        "diff": "".join(diffs),
    }


def _run_manual_agent(
    orchestrator_name: str,
    role: str,
    system_prompt: str,
    user_message: str,
) -> dict[str, Any]:
    if orchestrator_name == "crewai":
        crew_agents = importlib.import_module("crewai_orchestrator")
        result = crew_agents.call_crew(
            f"{role} Agent",
            (
                "Diagnose a user-described source-code issue without test evidence."
                if role == "Diagnosis"
                else "Propose a targeted patch for a user-described source-code issue."
            ),
            system_prompt,
            user_message,
            "One JSON object matching the required schema, with no surrounding prose.",
        )
    else:
        custom_agents = importlib.import_module("Orchestrator")
        result = custom_agents.call_agent(role, system_prompt, user_message)
    if not isinstance(result, dict):
        raise RuntimeError(f"{role} Agent did not return a valid JSON object.")
    return result


def run_manual_fix(
    intake_job_id: str,
    description: str,
    suspected_file: str | None = None,
    fix_job_id: str | None = None,
) -> dict[str, Any]:
    """Propose an explicitly unverified patch without executing repository code."""
    if fix_job_id is None:
        fix_job_id = uuid.uuid4().hex[:12]
    job: dict[str, Any] = {
        "fix_job_id": fix_job_id,
        "intake_job_id": intake_job_id,
        "stage": "locating",
        "status": "running",
        "verification_mode": "unverified",
        "logs": [],
        "diagnosis": None,
        "diff": "",
        "proposed_patch": None,
        "proposed_files": None,
        "mutation_verdict": None,
        "suite_passed": None,
        "retest_report": None,
        "final_verdict": None,
        "source_candidates": [],
        "chosen_source": None,
        "error": None,
        "audit": {},
        "audit_path": "",
    }
    with FIX_JOBS_LOCK:
        FIX_JOBS[fix_job_id] = job

    audit: AuditRecorder | None = None
    try:
        workspace = (WORKSPACES_DIR / intake_job_id).resolve()
        repo_root = (workspace / "repo").resolve()
        report_path = workspace / "baseline_report.json"
        if not report_path.is_file() or not repo_root.is_dir():
            raise FileNotFoundError(
                f"Completed intake workspace {intake_job_id} is unavailable."
            )
        baseline = json.loads(report_path.read_text(encoding="utf-8"))
        stack = baseline.get("stack", {})
        if stack.get("language") != "python" or stack.get("verification_mode") != "none":
            raise ValueError("Manual fixes are only available for Python intakes without pytest.")

        audit = AuditRecorder(job, workspace)
        job["audit"] = audit.audit
        job["audit_path"] = str(audit.audit_path)
        audit.record(
            event="manual_fix_started",
            verification_mode="unverified",
            final_verdict=None,
            user_description=description,
            stage="locating",
            status="running",
        )
        _log_fix(job, "Verification mode: unverified; no project code will be executed.")
        candidates = locate_source_for_description(
            description, repo_root, suspected_file=suspected_file
        )
        if not candidates:
            raise RuntimeError(
                "No Python source file matched the issue description. "
                "Provide a suspected_file path or include identifiers from the relevant code."
            )

        chosen = candidates[0]
        source_path = Path(chosen["path"]).resolve()
        source_path.relative_to(repo_root)
        source_code = source_path.read_text(encoding="utf-8", errors="replace")
        relative_path = source_path.relative_to(repo_root).as_posix()
        job["source_candidates"] = candidates
        job["chosen_source"] = str(source_path)
        job["stage"] = "diagnosing"
        job["orchestrator"], _ = _configured_orchestrator()
        _log_fix(
            job,
            f"Selected {relative_path} ({chosen['confidence']}): {chosen['reason']}.",
        )
        _log_fix(job, f"Using {job['orchestrator']} orchestrator agents.")
        audit.record(
            source_candidates=candidates,
            chosen_source=str(source_path),
            files={
                "read": [relative_path],
                "written": [],
                "patched": [],
            },
            stage="diagnosing",
            status="running",
        )

        custom_agents = importlib.import_module("Orchestrator")
        diagnosis_prompt = (
            custom_agents.DIAGNOSIS_SYSTEM_PROMPT
            + "\n\nThis is a MANUAL issue report. There is no test suite or traceback. "
              "Diagnose only from the user's description and the supplied source file. "
              "Do not claim that the issue was reproduced or verified."
        )
        diagnosis_input = (
            f"User's issue description:\n{description}\n\n"
            f"Suspected source file ({relative_path}):\n"
            f"```python\n{source_code}\n```"
        )
        diagnosis = _run_manual_agent(
            job["orchestrator"], "Diagnosis", diagnosis_prompt, diagnosis_input
        )
        if not isinstance(diagnosis.get("root_cause"), str) or not diagnosis["root_cause"].strip():
            raise RuntimeError("Diagnosis Agent returned no root_cause.")
        job["diagnosis"] = diagnosis
        job["stage"] = "fixing"
        _log_fix(job, f"Diagnosis: {diagnosis['root_cause']}")
        audit.record(diagnosis=diagnosis, stage="fixing", status="running")

        fix_prompt = (
            custom_agents.FIX_SYSTEM_PROMPT
            + "\n\nThis is an UNVERIFIED manual-fix proposal. There is no test suite. "
              "Use only the user's description, diagnosis, and supplied source. "
              "Return a proposed complete source file; do not claim tests passed."
        )
        fix_input = (
            f"User's issue description:\n{description}\n\n"
            f"Source file path: {relative_path}\n"
            f"Original source code:\n{source_code}\n\n"
            f"Diagnosis:\n{json.dumps(diagnosis)}"
        )
        fix = _run_manual_agent(job["orchestrator"], "Fix", fix_prompt, fix_input)
        patched_code = fix.get("patched_code")
        if not isinstance(patched_code, str) or not patched_code.strip():
            raise RuntimeError("Fix Agent returned no complete patched_code.")
        diff = _unified_diff_text(
            source_code,
            patched_code,
            fromfile=f"{relative_path} (before)",
            tofile=f"{relative_path} (proposed)",
        )

        job["proposed_patch"] = patched_code
        job["diff"] = diff
        job["final_verdict"] = "UNVERIFIED_FIX"
        job["stage"] = "pending_review"
        job["status"] = "pending_review"
        _log_fix(job, f"Proposed change: {fix.get('change_summary', 'See diff.')}")
        _log_fix(job, "UNVERIFIED_FIX: tests and mutation analysis were not run.")
        _log_fix(job, "Patch is staged until explicit manual confirmation and approval.")
        audit.record(
            diff=diff,
            proposed_patch=patched_code,
            change_summary=fix.get("change_summary"),
            verification_mode="unverified",
            mutation_verdict=None,
            suite_passed=None,
            retest_report=None,
            final_verdict="UNVERIFIED_FIX",
            stage="pending_review",
            status="pending_review",
        )
    except Exception as exc:
        job["stage"] = "failed"
        job["status"] = "failed"
        job["error"] = str(exc)
        job["final_verdict"] = "COULD_NOT_FIX"
        if audit is not None:
            audit.record(
                error=str(exc),
                verification_mode="unverified",
                final_verdict="COULD_NOT_FIX",
                stage="failed",
                status="failed",
            )
        _log_fix(job, f"{type(exc).__name__}: {exc}")
    finally:
        if audit is not None:
            job["audit"] = audit.audit
            job["audit_path"] = str(audit.audit_path)
            audit.persist()
    return job


def approve_fix(
    fix_job_id: str,
    runner=None,
    manual_review_confirmed: bool = False,
) -> dict[str, Any]:
    """Apply a previously proposed patch after human approval.

    This is the Phase 5 gate: the actual source file remains untouched until
    approval is explicitly granted. Once approved, the staged patch is written to
    the workspace and the baseline suite is re-run.
    """
    with FIX_JOBS_LOCK:
        job = FIX_JOBS.get(fix_job_id)
        if job is None:
            raise KeyError(f"Unknown fix_job_id: {fix_job_id}")

    if not job.get("chosen_source"):
        raise ValueError(f"Fix job {fix_job_id} has no chosen source file to patch.")
    if not job.get("proposed_patch") and not job.get("proposed_files"):
        raise ValueError(f"Fix job {fix_job_id} has no proposed patch ready for approval.")

    workspace = WORKSPACES_DIR / str(job.get("intake_job_id"))
    audit = AuditRecorder(job, workspace)
    if isinstance(job.get("audit"), dict):
        audit.audit.update(job["audit"])

    if job.get("verification_mode") == "unverified":
        if job.get("status") != "pending_review":
            raise ValueError("Unverified fixes can only be applied from pending_review.")
        if manual_review_confirmed is not True:
            raise ValueError("Manual review confirmation is required for unverified fixes.")
        if job.get("proposed_files"):
            raise ValueError("Unverified manual fixes must contain a single reviewed source patch.")
        repo_root = (workspace / "repo").resolve()
        source = Path(job["chosen_source"]).resolve()
        try:
            source.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError("Unverified patch source must be inside the intake repository.") from exc
        if not source.is_file():
            raise FileNotFoundError(f"Unverified patch source not found: {source}")
        source.write_text(job["proposed_patch"], encoding="utf-8")
        result = {
            "written": True,
            "baseline": None,
            "suite_passed": None,
            "diff": job.get("diff", ""),
        }
        final_verdict = "UNVERIFIED_FIX"
    elif job.get("proposed_files"):
        result = apply_fixes_to_workspace(
            str(job.get("intake_job_id")), job["proposed_files"], runner=runner,
        )
        final_verdict = "ACCEPT_FIX" if result["suite_passed"] else "REJECT_FIX"
    else:
        result = apply_fix_to_workspace(
            str(job.get("intake_job_id")),
            job["chosen_source"],
            job["proposed_patch"],
            runner=runner,
        )
        final_verdict = "ACCEPT_FIX" if result["suite_passed"] else "REJECT_FIX"

    job["stage"] = "done"
    job["status"] = "done"
    job["written"] = True
    job["suite_passed"] = result["suite_passed"]
    job["retest_report"] = result["baseline"]
    job["final_verdict"] = final_verdict
    job["diff"] = job.get("diff") or result.get("diff", "")
    job["approved_at"] = audit._utc_now()

    audit.record(
        event="fix_approved",
        stage="done",
        status="done",
        final_verdict=job["final_verdict"],
        verification_mode=job.get("verification_mode", "test_verified"),
        suite_passed=result["suite_passed"],
        retest_report=result["baseline"],
        manual_review_confirmed=(
            True if job.get("verification_mode") == "unverified" else None
        ),
    )
    audit.audit["completed_at"] = audit._utc_now()
    job["audit"] = audit.audit
    job["audit_path"] = str(audit.audit_path)
    audit.persist()
    return job


def reject_fix(fix_job_id: str, reason: str | None = None) -> dict[str, Any]:
    """Reject a proposed fix without mutating the workspace.

    The audit record is still persisted so the rejected proposal is visible and
    traceable for product trust and compliance reasons.
    """
    with FIX_JOBS_LOCK:
        job = FIX_JOBS.get(fix_job_id)
        if job is None:
            raise KeyError(f"Unknown fix_job_id: {fix_job_id}")

    workspace = WORKSPACES_DIR / str(job.get("intake_job_id"))
    audit = AuditRecorder(job, workspace)
    if isinstance(job.get("audit"), dict):
        audit.audit.update(job["audit"])
    reason_text = reason or "Manual rejection by operator"

    job["stage"] = "rejected"
    job["status"] = "rejected"
    job["final_verdict"] = "REJECT_FIX"
    job["error"] = reason_text
    job["rejected_at"] = audit._utc_now()

    audit.record(
        event="fix_rejected",
        stage="rejected",
        status="rejected",
        final_verdict="REJECT_FIX",
        reason=reason_text,
        error=reason_text,
        suite_passed=False,
    )
    audit.audit["completed_at"] = audit._utc_now()
    job["audit"] = audit.audit
    job["audit_path"] = str(audit.audit_path)
    audit.persist()
    return job


def export_fix(fix_job_id: str, export_format: str = "patch") -> dict[str, Any]:
    """Export a patch file or local branch without pushing to a remote."""
    with FIX_JOBS_LOCK:
        job = FIX_JOBS.get(fix_job_id)
        if job is None:
            raise KeyError(f"Unknown fix_job_id: {fix_job_id}")

    if export_format not in {"patch", "branch", "json"}:
        raise ValueError(f"Unsupported export format: {export_format}")
    if (
        job.get("verification_mode") == "unverified"
        and job.get("status") != "done"
    ):
        raise ValueError(
            "Unverified proposals cannot be exported until explicitly reviewed and approved."
        )

    workspace = (WORKSPACES_DIR / str(job.get("intake_job_id"))).resolve()
    repo = workspace / "repo"
    exported_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    payload = {
        "fix_job_id": job.get("fix_job_id"),
        "intake_job_id": job.get("intake_job_id"),
        "chosen_source": job.get("chosen_source"),
        "proposed_files": job.get("proposed_files"),
        "diff": job.get("diff", ""),
        "mutation_verdict": job.get("mutation_verdict"),
        "final_verdict": job.get("final_verdict"),
        "status": job.get("status"),
        "export_format": export_format,
        "exported_at": exported_at,
        "pushed": False,
    }
    if export_format == "patch":
        export_dir = workspace / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        patch_path = export_dir / f"{fix_job_id}.patch"
        patch_content = job.get("diff", "")
        patch_path.write_text(patch_content, encoding="utf-8")
        payload["patch"] = patch_content
        payload["export_path"] = str(patch_path)
    elif export_format == "branch":
        source = Path(job.get("chosen_source", "")).resolve()
        try:
            source.relative_to(repo.resolve())
        except ValueError as exc:
            raise ValueError("Branch export source must be inside the job repository.") from exc
        if not source.is_file():
            raise FileNotFoundError(f"Branch export source not found: {source}")

        branch_name = f"sentinel/fix-{fix_job_id}"
        subprocess.run(
            ["git", "switch", "-c", branch_name],
            cwd=str(repo), capture_output=True, text=True, encoding="utf-8", check=True,
        )
        proposed_files = job.get("proposed_files")
        if proposed_files:
            staged_paths = []
            for relative_path, patched_code in proposed_files.items():
                relative = Path(relative_path)
                if relative.is_absolute():
                    raise ValueError(
                        f"Patch path must be relative to the repository: {relative_path}"
                    )
                target = (repo / relative).resolve()
                try:
                    target.relative_to(repo.resolve())
                except ValueError as exc:
                    raise ValueError(
                        f"Branch patch path escapes the repository: {relative_path}"
                    ) from exc
                if not target.is_file():
                    raise FileNotFoundError(f"Branch patch source not found: {target}")
                target.write_text(patched_code, encoding="utf-8")
                staged_paths.append(str(target.relative_to(repo.resolve())))
        else:
            source.write_text(job.get("proposed_patch") or "", encoding="utf-8")
            staged_paths = [str(source.relative_to(repo.resolve()))]
        subprocess.run(
            ["git", "add", "--", *staged_paths],
            cwd=str(repo), capture_output=True, text=True, encoding="utf-8", check=True,
        )
        subprocess.run(
            ["git", "-c", "user.name=Sentinel", "-c", "user.email=sentinel@localhost",
             "commit", "-m", f"Apply proposed fix {fix_job_id}"],
            cwd=str(repo), capture_output=True, text=True, encoding="utf-8", check=True,
        )
        payload["branch_name"] = branch_name
        payload["branch_created_locally"] = True
        payload["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo), capture_output=True, text=True, encoding="utf-8", check=True,
        ).stdout.strip()
    elif export_format == "json":
        payload["proposed_patch"] = job.get("proposed_patch")

    if job.get("audit_path"):
        payload["audit_path"] = job.get("audit_path")

    job["exported_as"] = export_format
    job["exported_at"] = exported_at
    if export_format == "branch":
        job["exported_branch"] = payload["branch_name"]

    audit_path_value = job.get("audit_path")
    if audit_path_value:
        audit_path = Path(audit_path_value)
        if audit_path.exists():
            try:
                audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                audit_payload = {}
            audit_payload.update({
                "exported_as": export_format,
                "exported_at": exported_at,
                "exported_branch": payload.get("branch_name"),
                "pushed": False,
            })
            audit_payload["updated_at"] = exported_at
            audit_path.write_text(json.dumps(audit_payload, indent=2, sort_keys=True), encoding="utf-8")
            job["audit"] = audit_payload
    return payload


# ---------------------------------------------------------------------------
# run_fix_for_failure
# ---------------------------------------------------------------------------

def _log_fix(job: dict, message: str) -> None:
    job.setdefault("logs", []).append(message)


def run_fix_for_failure(
    intake_job_id: str,
    failure_index: int,
    fix_job_id: str | None = None,
    runner=None,
) -> dict:
    """Full pipeline: locate source → build → orchestrate → apply → re-test.

    Designed to run in a background thread. Updates FIX_JOBS[fix_job_id] live.
    Returns the final fix job dict.
    """
    if fix_job_id is None:
        fix_job_id = uuid.uuid4().hex[:12]

    job: dict[str, Any] = {
        "fix_job_id": fix_job_id,
        "intake_job_id": intake_job_id,
        "failure_index": failure_index,
        "verification_mode": "test_verified",
        "stage": "locating",
        "status": "running",
        "logs": [],
        "diagnosis": None,
        "diff": "",
        "proposed_patch": None,
        "mutation_verdict": None,
        "suite_passed": None,
        "source_candidates": [],
        "chosen_source": None,
        "orchestrator_output": "",
        "proposed_files": None,
        "retest_report": None,
        "error": None,
        "audit": {},
        "audit_path": "",
    }

    with FIX_JOBS_LOCK:
        FIX_JOBS[fix_job_id] = job

    audit = None
    try:
        # ---- load workspace artifacts ----
        workspace = WORKSPACES_DIR / intake_job_id
        if not workspace.exists():
            raise FileNotFoundError(
                f"Workspace {workspace} not found. Run /repo/analyze first."
            )
        audit = AuditRecorder(job, workspace)
        job["audit"] = audit.audit
        job["audit_path"] = str(audit.audit_path)
        audit.record(event="job_started", stage="locating", status="running")

        report_path = workspace / "baseline_report.json"
        if not report_path.exists():
            raise FileNotFoundError(
                "baseline_report.json not found in workspace. "
                "The intake job may not have completed successfully."
            )

        baseline_report = json.loads(report_path.read_text(encoding="utf-8"))
        failures = baseline_report.get("failures", [])
        if not (0 <= failure_index < len(failures)):
            raise IndexError(
                f"failure_index {failure_index} is out of range "
                f"(report has {len(failures)} failure(s))."
            )

        failure = failures[failure_index]
        orchestrator_name, orchestrator_path = _configured_orchestrator()
        stack = baseline_report.get("stack", {})
        project_root = Path(
            stack.get("project_root", str(workspace / "repo"))
        ).resolve()

        _log_fix(job, f"Failure #{failure_index}: {failure.get('test_id', '?')}")
        _log_fix(job, f"Error: {failure.get('error_type', '?')}: "
                      f"{failure.get('message', '')[:200]}")

        # ---- Stage: locating ----
        job["stage"] = "locating"
        _log_fix(job, f"Using {orchestrator_name} orchestrator.")
        _log_fix(job, f"Searching for source file under {project_root} …")
        candidates = locate_source_for_failure(failure, project_root)

        if not candidates:
            raise RuntimeError(
                "Could not identify the source file for this failure. "
                "The traceback only points into test or library code, "
                "or no project source files were found under project_root."
            )

        if orchestrator_name == "langgraph":
            candidates = _expand_imported_source_candidates(candidates, project_root)

        job["source_candidates"] = candidates
        chosen = candidates[0]
        job["chosen_source"] = chosen["path"]
        audit.record(
            source_candidates=candidates,
            chosen_source=chosen["path"],
            stage="locating",
            status="running",
        )
        _log_fix(job, f"Source identified ({chosen['confidence']}): {chosen['path']}")
        _log_fix(job, f"Reason: {chosen['reason']}")

        # ---- Stage: building inputs ----
        job["stage"] = "building"
        test_file_rel = failure.get("file", "")
        test_file_abs = None
        for base in (project_root, workspace / "repo"):
            candidate = (base / test_file_rel).resolve()
            if candidate.exists():
                test_file_abs = candidate
                break
        if test_file_abs is None:
            raise FileNotFoundError(
                f"Test file '{test_file_rel}' not found under "
                f"{project_root} or {workspace / 'repo'}."
            )

        fix_inputs = build_fix_job(failure, test_file_abs, chosen["path"])
        orchestrator_sources = (
            candidates if orchestrator_name == "langgraph" else [chosen]
        )
        audit.record(
            files={
                "read": [
                    *(item["path"] for item in orchestrator_sources),
                    str(fix_inputs["test_path"]),
                ],
                "written": [],
                "patched": [],
            },
            stage="building",
            status="running",
        )
        _log_fix(job, f"Inputs ready — source: {fix_inputs['source_path'].name}, "
                      f"{len(orchestrator_sources)} source file(s), "
                      f"test: {fix_inputs['test_path'].name}")

        # ---- Stage: diagnosing / fixing / mutating ----
        job["stage"] = "diagnosing"
        _log_fix(job, f"Launching {orchestrator_path.name} (Diagnosis → Fix → Mutation) …")
        original_sources = {
            Path(item["path"]).resolve(): Path(item["path"]).read_text(
                encoding="utf-8", errors="replace"
            )
            for item in orchestrator_sources
        }
        original_source = original_sources[fix_inputs["source_path"].resolve()]

        review_source, review_test = _stage_review_project(
            workspace,
            project_root,
            fix_job_id,
            fix_inputs["source_path"],
            test_file_abs,
        )
        job["review_source_path"] = str(review_source)
        review_root = workspace / ".review" / fix_job_id / "repo"
        review_sources = [
            review_root / Path(item["path"]).resolve().relative_to(project_root)
            for item in orchestrator_sources
        ]

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        execution_runner = _execution_runner(workspace, runner)
        if type(execution_runner).__name__ == "DockerRunner":
            docker_python = _ensure_venv_ready(workspace, runner=execution_runner)
            env["SENTINEL_DOCKER_WORKSPACE"] = str(workspace)
            env["SENTINEL_DOCKER_PYTHON"] = str(docker_python)
        try:
            command = _build_orchestrator_command(
                orchestrator_name,
                orchestrator_path,
                sys.executable,
                review_source,
                review_test,
                review_root,
                review_sources[1:],
            )
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                cwd=str(BASE_DIR),
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Orchestrator timed out after 10 minutes. "
                "The repository may be too complex for automated fixing."
            )

        orchestrator_output = proc.stdout or ""
        if proc.stderr:
            orchestrator_output += "\n--- stderr ---\n" + proc.stderr
        job["orchestrator_output"] = orchestrator_output
        audit.record(
            commands=[{
                "name": orchestrator_name,
                "returncode": proc.returncode,
                "command": command,
            }],
            orchestrator_output=orchestrator_output,
            stage="diagnosing",
            status="running",
        )
        _log_fix(job, f"Orchestrator exited with code {proc.returncode}.")

        # Detect explicit failure cases
        if "Retry limit" in orchestrator_output:
            job["final_verdict"] = "COULD_NOT_FIX"
            audit.record(final_verdict="COULD_NOT_FIX")
            raise RuntimeError(
                "The Fix Agent could not fix this failure automatically after "
                "3 attempts. Manual review needed."
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Orchestrator exited with status {proc.returncode}. "
                "Check orchestrator_output for details."
            )

        # Parse diagnosis and mutation verdict from output
        diagnosis = _extract_diagnosis(orchestrator_output)
        job["diagnosis"] = diagnosis
        audit.record(diagnosis=diagnosis)
        if diagnosis:
            _log_fix(job, f"Diagnosis: {diagnosis.get('root_cause', '?')}")

        mutation_verdict = _extract_mutation_verdict(orchestrator_output)
        job["mutation_verdict"] = mutation_verdict
        audit.record(mutation_verdict=mutation_verdict)
        _log_fix(job, f"Mutation verdict: {mutation_verdict or 'not available'}")

        # ---- Stage: review gate ----
        patched_sources = {
            original_path: staged_path.read_text(
                encoding="utf-8", errors="replace"
            )
            for original_path, staged_path in zip(
                original_sources, review_sources
            )
        }
        patched_code = patched_sources[fix_inputs["source_path"].resolve()]
        changed_sources = {
            original_path: patched
            for original_path, patched in patched_sources.items()
            if original_sources[original_path] != patched
        }
        proposed_files = None
        if orchestrator_name == "langgraph" and len(orchestrator_sources) > 1:
            if not changed_sources:
                changed_sources = {
                    fix_inputs["source_path"].resolve(): patched_code,
                }
            proposed_files = {
                original_path.relative_to(workspace / "repo").as_posix(): content
                for original_path, content in changed_sources.items()
            }
            diff = "".join(
                _unified_diff_text(
                    original_sources[original_path],
                    content,
                    fromfile=(
                        "repo/" + original_path.relative_to(workspace / "repo").as_posix()
                        + " (before)"
                    ),
                    tofile=(
                        "repo/" + original_path.relative_to(workspace / "repo").as_posix()
                        + " (after)"
                    ),
                )
                for original_path, content in changed_sources.items()
            )
        else:
            diff = _unified_diff_text(
                original_source,
                patched_code,
                fromfile=str(fix_inputs["source_path"].name) + " (before)",
                tofile=str(fix_inputs["source_path"].name) + " (after)",
            )
        job["proposed_patch"] = patched_code
        job["proposed_files"] = proposed_files
        job["diff"] = diff
        job["stage"] = "pending_review"
        job["status"] = "pending_review"
        job["suite_passed"] = None
        job["retest_report"] = None
        audit.record(
            diff=diff,
            proposed_patch=patched_code,
            proposed_files=proposed_files,
            files={
                "read": [str(path) for path in original_sources] + [str(fix_inputs["test_path"])],
                "written": [],
                "patched": [
                    str(path.relative_to(workspace / "repo"))
                    for path in changed_sources
                ],
            },
            suite_passed=None,
            retest_report=None,
            stage="pending_review",
            status="pending_review",
            final_verdict=mutation_verdict,
        )

        _log_fix(job, "Patch proposed and waiting for human approval before writing to workspace.")

    except Exception as exc:
        job["stage"] = "failed"
        job["status"] = "failed"
        job["error"] = str(exc)
        if job.get("final_verdict") is None:
            job["final_verdict"] = "COULD_NOT_FIX"
        if audit is not None:
            audit.record(error=str(exc), stage="failed", status="failed", final_verdict=job["final_verdict"])
        _log_fix(job, f"{type(exc).__name__}: {exc}")
    finally:
        if audit is not None:
            job["audit"] = audit.audit
            job["audit_path"] = str(audit.audit_path)
            audit.persist()

    return job


# ---------------------------------------------------------------------------
# Orchestrator output parsers
# ---------------------------------------------------------------------------

def _extract_diagnosis(output: str) -> dict | None:
    """Pull the diagnosis JSON from orchestrator stdout."""
    for block in _find_json_blocks(output):
        if isinstance(block, dict) and "root_cause" in block:
            return block
    # Fallback: extract from printed lines
    root_cause_m = re.search(r'Root cause:\s*(.+)', output)
    confidence_m = re.search(r'Confidence:\s*(.+)', output)
    if root_cause_m:
        return {
            "root_cause": root_cause_m.group(1).strip(),
            "confidence": (confidence_m.group(1).strip()
                           if confidence_m else "unknown"),
        }
    return None


def _extract_mutation_verdict(output: str) -> str | None:
    """Extract the final_verdict string from the orchestrator output."""
    m = re.search(r'=== Final Verdict:\s*(\S+)', output)
    if m:
        return m.group(1)
    for block in _find_json_blocks(output):
        if isinstance(block, dict) and "final_verdict" in block:
            return block["final_verdict"]
    return None


def _find_json_blocks(text: str) -> list[dict | list]:
    """Scan text for JSON objects/arrays using the stdlib decoder's raw_decode."""
    results = []
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] in ('{', '['):
            try:
                obj, offset = decoder.raw_decode(text, i)
                results.append(obj)
                i += offset - i
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return results
