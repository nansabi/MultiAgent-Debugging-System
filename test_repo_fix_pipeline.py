"""
test_repo_fix_pipeline.py — unit and integration tests for Phase 3.

Tests:
  1. locate_source_for_failure() — correct traceback → source file
  2. locate_source_for_failure() — ambiguous / library-only traceback → empty
  3. apply_fix_to_workspace() — workspace boundary check (file outside workspace raises)
  4. apply_fix_to_workspace() — writes patch and re-runs suite (mocked runner)
  5. End-to-end fixture: broken function + failing test → full run_fix_for_failure()
     runs through the real Orchestrator.py pipeline and confirms suite passes.
  6. Regression: patching never touches files outside the workspace.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repo root is on sys.path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from repo_fix_pipeline import (
    locate_source_for_failure,
    build_fix_job,
    apply_fix_to_workspace,
    approve_fix,
    export_fix,
    reject_fix,
    run_fix_for_failure,
    _stage_review_project,
    _unified_diff_text,
    _extract_diagnosis,
    _extract_mutation_verdict,
    _find_json_blocks,
)
from repo_intake import WORKSPACES_DIR, _ensure_venv_ready


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_git_repo(path: Path) -> None:
    """Create a minimal bare git repo so git commands won't complain."""
    subprocess.run(["git", "init", str(path)], capture_output=True, check=False)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@test"],
                   capture_output=True, check=False)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"],
                   capture_output=True, check=False)


# ---------------------------------------------------------------------------
# Unit tests: locate_source_for_failure
# ---------------------------------------------------------------------------

class LocateSourceTests(unittest.TestCase):

    def test_traceback_frame_identifies_source_file(self):
        """High-confidence hit: traceback frame points directly into project source."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Create a project source file
            src = root / "mymodule.py"
            src.write_text("def add(a, b): return a + b\n")
            # Create a test file
            tst = root / "test_mymodule.py"
            tst.write_text("from mymodule import add\ndef test_add(): assert add(1,2)==3\n")

            failure = {
                "test_id": "test_mymodule::test_add",
                "file": "test_mymodule.py",
                "traceback": f"  {root}/mymodule.py:1: in add\nAssertionError",
                "message": "assert 1 == 3",
                "error_type": "AssertionError",
            }
            candidates = locate_source_for_failure(failure, root)

        # Should find mymodule.py with high confidence
        self.assertTrue(len(candidates) > 0,
                        "Expected at least one candidate, got none")
        paths = [c["path"] for c in candidates]
        self.assertTrue(any(Path(p).name == "mymodule.py" for p in paths),
                        f"Expected mymodule.py in candidates, got: {paths}")
        high_conf = [c for c in candidates if c["confidence"] == "high"]
        self.assertTrue(len(high_conf) > 0,
                        "Expected at least one high-confidence candidate")

    def test_library_only_traceback_returns_empty(self):
        """Ambiguous case: traceback only points into site-packages, not project."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Only a test file, no source file, traceback points to stdlib
            tst = root / "test_something.py"
            tst.write_text("import json\ndef test_json(): assert json.loads('{}') == {}\n")

            failure = {
                "test_id": "test_something::test_json",
                "file": "test_something.py",
                "traceback": (
                    "  /usr/lib/python3.11/site-packages/requests/adapters.py:486: in send\n"
                    "  /usr/lib/python3.11/json/decoder.py:355: in raw_decode\n"
                    "JSONDecodeError: Expecting value"
                ),
                "message": "Expecting value",
                "error_type": "JSONDecodeError",
            }
            candidates = locate_source_for_failure(failure, root)

        # No project source file exists → should return empty
        self.assertEqual(candidates, [],
                         f"Expected empty candidates for library-only traceback, "
                         f"got: {candidates}")

    def test_import_signal_finds_module(self):
        """Medium-confidence: test file imports a project module, no traceback."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "calculator.py"
            src.write_text("def multiply(a, b): return a * b\n")
            tst = root / "test_calc.py"
            tst.write_text("from calculator import multiply\ndef test_mul(): assert multiply(2,3)==6\n")

            failure = {
                "test_id": "test_calc::test_mul",
                "file": "test_calc.py",
                "traceback": "",   # empty traceback — signal 2 should take over
                "message": "assert 5 == 6",
                "error_type": "AssertionError",
            }
            candidates = locate_source_for_failure(failure, root)

        paths = [c["path"] for c in candidates]
        self.assertTrue(any(Path(p).name == "calculator.py" for p in paths),
                        f"Expected calculator.py from import signal, got: {paths}")

    def test_test_file_itself_excluded_from_candidates(self):
        """The test file must never appear as a source candidate."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "buggy.py"
            src.write_text("def f(): return 1\n")
            tst = root / "test_buggy.py"
            tst.write_text("from buggy import f\ndef test_f(): assert f()==2\n")

            # Traceback mentions the test file
            failure = {
                "test_id": "test_buggy::test_f",
                "file": "test_buggy.py",
                "traceback": (
                    f"  {root}/test_buggy.py:2: in test_f\n"
                    f"  {root}/buggy.py:1: in f\n"
                    "AssertionError"
                ),
                "message": "assert 1 == 2",
                "error_type": "AssertionError",
            }
            candidates = locate_source_for_failure(failure, root)

        paths = [Path(c["path"]).name for c in candidates]
        self.assertNotIn("test_buggy.py", paths,
                         "Test file should never appear as a source candidate")
        self.assertIn("buggy.py", paths,
                      "buggy.py should be identified as the source candidate")


# ---------------------------------------------------------------------------
# Unit tests: apply_fix_to_workspace boundary check
# ---------------------------------------------------------------------------

class ApplyFixBoundaryTests(unittest.TestCase):

    def test_review_project_tests_use_staged_source_without_mutating_checkout(self):
        """The staged test tree must import its candidate source, not the original checkout."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "job"
            repo = workspace / "repo"
            repo.mkdir(parents=True)
            source = repo / "multiply_module.py"
            test_file = repo / "test_multiply.py"
            original_code = "def multiply(a, b): return a + b\n"
            source.write_text(original_code, encoding="utf-8")
            test_file.write_text(
                "from multiply_module import multiply\n"
                "def test_product(): assert multiply(3, 4) == 12\n",
                encoding="utf-8",
            )

            staged_source, staged_test = _stage_review_project(
                workspace, repo, "review-copy-test", source, test_file
            )
            staged_source.write_text(
                "def multiply(a, b): return a * b\n", encoding="utf-8"
            )
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", str(staged_test)],
                cwd=staged_test.parent,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(source.read_text(encoding="utf-8"), original_code)

    def test_ensure_venv_ready_rebuilds_linux_layout_for_docker(self):
        """A stale Windows .venv layout must be repaired before Docker can run tests."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            venv = workspace / ".venv"
            scripts = venv / "Scripts"
            scripts.mkdir(parents=True)
            (scripts / "python.exe").write_text("stub", encoding="utf-8")

            class DockerRunner:
                pass

            with patch("repo_intake._execution_runner", return_value=DockerRunner()), \
                 patch("repo_intake._execute") as mock_execute:
                python = _ensure_venv_ready(workspace, runner=DockerRunner())

            self.assertEqual(python, workspace / ".venv" / "bin/python")
            self.assertTrue(mock_execute.called)

    def test_raises_on_path_outside_workspace(self):
        """Writing a file outside the job workspace must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            # Set up a fake workspace directory structure
            ws_parent = Path(tmp) / "workspaces"
            ws_parent.mkdir()

            outside_file = Path(tmp) / "outside.py"
            outside_file.write_text("# untouched\n")

            fake_job_id = "fakejob0001"
            ws = ws_parent / fake_job_id
            ws.mkdir()

            # Patch WORKSPACES_DIR
            import repo_fix_pipeline as rfp
            orig = rfp.WORKSPACES_DIR
            rfp.WORKSPACES_DIR = ws_parent
            try:
                with self.assertRaises(ValueError) as ctx:
                    apply_fix_to_workspace(fake_job_id, outside_file, "# evil\n")
                self.assertIn("Security violation", str(ctx.exception))
            finally:
                rfp.WORKSPACES_DIR = orig

            # Original file must be untouched
            self.assertEqual(outside_file.read_text(), "# untouched\n")

    def test_patch_written_inside_workspace(self):
        """A valid patch stays pending until approval, then is written and re-tested."""
        with tempfile.TemporaryDirectory() as tmp:
            ws_parent = Path(tmp) / "workspaces"
            ws_parent.mkdir()
            fake_job_id = "testjob0001"
            ws = ws_parent / fake_job_id
            ws.mkdir()
            repo = ws / "repo"
            repo.mkdir()

            src = ws / "repo" / "module.py"
            src.write_text("def f(): return 1\n")

            import repo_fix_pipeline as rfp
            orig_ws = rfp.WORKSPACES_DIR
            orig_jobs = dict(rfp.FIX_JOBS)

            # Mock run_baseline_tests to avoid actually running pytest
            mock_baseline = {
                "total": 1, "passed": 1, "failed": 0, "errors": 0,
                "skipped": 0, "failures": [], "duration": 0.1,
            }

            rfp.WORKSPACES_DIR = ws_parent
            rfp.FIX_JOBS.clear()
            try:
                rfp.FIX_JOBS["approval-job-1"] = {
                    "fix_job_id": "approval-job-1",
                    "intake_job_id": fake_job_id,
                    "failure_index": 0,
                    "stage": "pending_review",
                    "status": "pending_review",
                    "chosen_source": str(src),
                    "proposed_patch": "def f(): return 2\n",
                    "diff": "--- module.py (before)\n+++ module.py (after)\n@@\n-def f(): return 1\n+def f(): return 2\n",
                    "audit": {
                        "fix_job_id": "approval-job-1",
                        "diagnosis": {"root_cause": "test diagnosis"},
                        "mutation_verdict": "ACCEPT_FIX",
                        "diff": "existing persisted diff",
                        "started_at": "2026-10-02T00:00:00Z",
                    },
                }

                class RunResult:
                    stdout = ""
                    stderr = ""
                    returncode = 0
                    timed_out = False

                mock_runner = MagicMock()
                mock_runner.run.return_value = RunResult()

                with patch("repo_fix_pipeline.run_baseline_tests",
                           return_value=mock_baseline), \
                     patch("repo_fix_pipeline._execution_runner",
                           return_value=mock_runner):
                    result = approve_fix("approval-job-1")
            finally:
                rfp.WORKSPACES_DIR = orig_ws
                rfp.FIX_JOBS.clear()
                rfp.FIX_JOBS.update(orig_jobs)

            self.assertEqual(result["status"], "done")
            self.assertTrue(result["suite_passed"])
            self.assertTrue(result["written"])
            self.assertEqual(src.read_text(), "def f(): return 2\n")
            self.assertIn("return 2", result["diff"])
            self.assertIn("return 1", result["diff"])
            audit = json.loads((ws / "fix_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["diagnosis"]["root_cause"], "test diagnosis")
            self.assertEqual(audit["mutation_verdict"], "ACCEPT_FIX")
            self.assertTrue(audit["completed_at"])

    def test_reject_fix_keeps_complete_audit_trail(self):
        """Rejected fixes must still persist a full, auditable record."""
        with tempfile.TemporaryDirectory() as tmp:
            ws_parent = Path(tmp) / "workspaces"
            ws_parent.mkdir()
            job_id = "rejectjob0001"
            workspace = ws_parent / job_id
            workspace.mkdir()
            (workspace / "repo").mkdir()

            import repo_fix_pipeline as rfp
            orig_ws = rfp.WORKSPACES_DIR
            orig_jobs = dict(rfp.FIX_JOBS)

            rfp.WORKSPACES_DIR = ws_parent
            rfp.FIX_JOBS.clear()
            try:
                rfp.FIX_JOBS["reject-job-1"] = {
                    "fix_job_id": "reject-job-1",
                    "intake_job_id": job_id,
                    "failure_index": 0,
                    "stage": "pending_review",
                    "status": "pending_review",
                    "chosen_source": str(workspace / "repo" / "module.py"),
                    "proposed_patch": "def f(): return 2\n",
                    "diff": "--- module.py (before)\n+++ module.py (after)\n@@\n-def f(): return 1\n+def f(): return 2\n",
                    "mutation_verdict": "ACCEPT_FIX",
                }

                result = reject_fix("reject-job-1", reason="operator rejected patch")

                self.assertEqual(result["status"], "rejected")
                self.assertEqual(result["final_verdict"], "REJECT_FIX")
                self.assertIn("operator rejected patch", result["error"])

                audit_path = Path(result["audit_path"])
                self.assertTrue(audit_path.exists())
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                self.assertEqual(audit["status"], "rejected")
                self.assertEqual(audit["final_verdict"], "REJECT_FIX")
                self.assertEqual(audit["reason"], "operator rejected patch")
            finally:
                rfp.WORKSPACES_DIR = orig_ws
                rfp.FIX_JOBS.clear()
                rfp.FIX_JOBS.update(orig_jobs)

    def test_export_fix_writes_patch_and_creates_local_branch_without_push(self):
        """Patch export writes a .patch file; branch export commits locally only."""
        with tempfile.TemporaryDirectory() as tmp:
            ws_parent = Path(tmp) / "workspaces"
            workspace = ws_parent / "exportjob0001"
            repo = workspace / "repo"
            repo.mkdir(parents=True)
            source = repo / "module.py"
            original = "def f(): return 1"
            proposed = "def f(): return 2"
            source.write_text(original, encoding="utf-8")
            _make_git_repo(repo)
            subprocess.run(["git", "-C", str(repo), "add", "module.py"], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                 "user.email=test@test", "commit", "-m", "baseline"],
                check=True, capture_output=True,
            )

            diff = _unified_diff_text(
                original,
                proposed,
                fromfile="module.py (before)",
                tofile="module.py (after)",
            )
            import repo_fix_pipeline as rfp
            original_workspaces = rfp.WORKSPACES_DIR
            original_jobs = dict(rfp.FIX_JOBS)
            rfp.WORKSPACES_DIR = ws_parent
            rfp.FIX_JOBS.clear()
            rfp.FIX_JOBS["export-job-1"] = {
                "fix_job_id": "export-job-1",
                "intake_job_id": "exportjob0001",
                "chosen_source": str(source),
                "proposed_patch": proposed,
                "diff": diff,
                "mutation_verdict": "ACCEPT_FIX",
                "status": "pending_review",
            }

            invoked_commands = []
            real_run = subprocess.run

            def record_git_run(args, *run_args, **kwargs):
                command = list(args) if isinstance(args, (list, tuple)) else [str(args)]
                if command and command[0] == "git":
                    invoked_commands.append(command)
                return real_run(args, *run_args, **kwargs)

            try:
                with patch("repo_fix_pipeline.subprocess.run", side_effect=record_git_run):
                    patch_result = export_fix("export-job-1", "patch")
                    branch_result = export_fix("export-job-1", "branch")
            finally:
                rfp.WORKSPACES_DIR = original_workspaces
                rfp.FIX_JOBS.clear()
                rfp.FIX_JOBS.update(original_jobs)

            patch_path = Path(patch_result["export_path"])
            patch_content = patch_path.read_text(encoding="utf-8")
            self.assertIn("-def f(): return 1\n+def f(): return 2\n", patch_content)
            self.assertTrue(branch_result["branch_created_locally"])
            self.assertFalse(branch_result["pushed"])
            self.assertEqual(source.read_text(encoding="utf-8"), proposed)
            current_branch = subprocess.run(
                ["git", "-C", str(repo), "branch", "--show-current"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            self.assertEqual(current_branch, branch_result["branch_name"])
            self.assertFalse(any("push" in command for command in invoked_commands))

            print(f"PATCH_EXPORT_PATH={patch_path}")
            print("PATCH_FILE_CONTENT_START")
            print(patch_content, end="")
            print("PATCH_FILE_CONTENT_END")
            print(f"BRANCH_NAME={branch_result['branch_name']}")
            print(f"BRANCH_HEAD={current_branch}")
            print(f"PUSHED={branch_result['pushed']}")
            print(f"GIT_COMMANDS={invoked_commands}")


# ---------------------------------------------------------------------------
# Unit tests: output parsers
# ---------------------------------------------------------------------------

class OutputParserTests(unittest.TestCase):

    def test_extract_diagnosis_from_json_block(self):
        output = (
            'Some preamble\n'
            '{"root_cause": "off-by-one", "explanation": "loop runs one too many", '
            '"faulty_location": "line 5", "category": "logic_error", '
            '"confidence": "High", "suggested_fix_direction": "use < instead of <="}\n'
            'trailing text'
        )
        result = _extract_diagnosis(output)
        self.assertIsNotNone(result)
        self.assertEqual(result["root_cause"], "off-by-one")

    def test_extract_diagnosis_from_printed_lines(self):
        output = "  Root cause: wrong operator\n  Confidence: High\n"
        result = _extract_diagnosis(output)
        self.assertIsNotNone(result)
        self.assertIn("wrong operator", result["root_cause"])

    def test_extract_mutation_verdict_from_header(self):
        output = "=== Final Verdict: ACCEPT_FIX ===\n"
        self.assertEqual(_extract_mutation_verdict(output), "ACCEPT_FIX")

    def test_extract_mutation_verdict_from_json(self):
        output = '{"final_verdict": "ACCEPT_WITH_ADDED_TESTS", "confidence": "Low"}'
        self.assertEqual(_extract_mutation_verdict(output),
                         "ACCEPT_WITH_ADDED_TESTS")

    def test_find_json_blocks(self):
        text = 'before {"a": 1} middle [1, 2, 3] after'
        blocks = _find_json_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0], {"a": 1})
        self.assertEqual(blocks[1], [1, 2, 3])


# ---------------------------------------------------------------------------
# End-to-end fixture test
# ---------------------------------------------------------------------------

class FixtureE2ETest(unittest.TestCase):
    """
    Creates a minimal fake workspace with a broken Python source file and a
    failing pytest test, then runs run_fix_for_failure() end-to-end through
    the real Orchestrator.py pipeline.

    The fixture bug: multiply(a, b) returns a + b instead of a * b.
    One test: test_multiply() asserts multiply(3, 4) == 12 (fails: returns 7).

    REQUIRES: GEMINI_API_KEY set in the environment.
    Skipped automatically if the API key is absent.
    """

    BUGGY_SOURCE = """\
def multiply(a, b):
    \"\"\"Return the product of a and b.\"\"\"
    return a + b   # BUG: should be a * b
"""

    FIXED_SOURCE = """\
def multiply(a, b):
    \"\"\"Return the product of a and b.\"\"\"
    return a * b
"""

    TEST_CODE = """\
from multiply_module import multiply

def test_multiply_basic():
    assert multiply(3, 4) == 12

def test_multiply_zero():
    assert multiply(0, 5) == 0

def test_multiply_negative():
    assert multiply(-2, 3) == -6
"""

    def setUp(self):
        if not os.environ.get("GEMINI_API_KEY"):
            self.skipTest("GEMINI_API_KEY not set — skipping live E2E test")

    def _build_fixture_workspace(self) -> tuple[str, Path]:
        """Create a workspace that looks like what run_intake() would produce."""
        ws_parent = WORKSPACES_DIR
        ws_parent.mkdir(parents=True, exist_ok=True)

        job_id = "phase3_e2e_fixture"
        workspace = ws_parent / job_id
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir()
        (workspace / ".sentinel-managed").touch()

        repo = workspace / "repo"
        repo.mkdir()
        (repo / "requirements.txt").write_text("pytest\n")

        # Write the buggy source file
        src = repo / "multiply_module.py"
        src.write_text(self.BUGGY_SOURCE)

        # Write the test file
        tst = repo / "test_multiply.py"
        tst.write_text(self.TEST_CODE)

        # Run pytest to get real baseline failures
        venv = workspace / ".venv"
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(venv)],
            check=True, capture_output=True,
        )
        python = (venv / ("Scripts/python.exe" if sys.platform == "win32"
                          else "bin/python"))
        subprocess.run(
            [sys.executable, "-m", "pip", "--python", str(python),
             "install", "pytest"],
            check=True, capture_output=True,
        )

        import xml.etree.ElementTree as ET
        from repo_intake import run_baseline_tests, LocalRunner
        baseline = run_baseline_tests(repo, runner=LocalRunner())

        # Write baseline_report.json
        report = {
            **baseline,
            "job_id": job_id,
            "repository": "fixture",
            "stack": {
                "language": "python",
                "test_framework": "pytest",
                "dependency_files": ["requirements.txt"],
                "project_root": str(repo),
            },
            "pytest_xml_path": str(workspace / "pytest-report.xml"),
        }
        (workspace / "baseline_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

        return job_id, workspace

    def test_e2e_fix_fixture_repo(self):
        """Full pipeline: buggy fixture → locate → diagnose → pending review → approval → re-test."""
        import repo_fix_pipeline as rfp

        job_id, workspace = self._build_fixture_workspace()

        # Verify that the baseline report actually has failures
        report = json.loads(
            (workspace / "baseline_report.json").read_text(encoding="utf-8")
        )
        self.assertGreater(
            len(report.get("failures", [])), 0,
            "Fixture setup failed: baseline should have at least 1 failure"
        )

        # Run the fix pipeline: the patch must remain pending until approval.
        result = run_fix_for_failure(job_id, failure_index=0)
        initial_source = Path(result["chosen_source"]).read_text(encoding="utf-8")

        self.assertEqual(result["status"], "pending_review",
                         f"Expected review gate, got {result['status']}: {result.get('error')}")
        self.assertIn("pending_review", result["stage"])
        self.assertEqual(Path(result["chosen_source"]).read_text(encoding="utf-8"), initial_source)

        approved = approve_fix(result["fix_job_id"])

        print("\n--- E2E Fix Pipeline Result ---")
        print(f"Stage: {approved['stage']}")
        print(f"Status: {approved['status']}")
        print(f"Chosen source: {approved.get('chosen_source')}")
        print(f"Diagnosis: {approved.get('diagnosis')}")
        print(f"Mutation verdict: {approved.get('mutation_verdict')}")
        print(f"Suite passed: {approved.get('suite_passed')}")
        print(f"Diff:\n{approved.get('diff', '(none)')}")
        print(f"Logs:\n" + "\n".join(approved.get("logs", [])))

        # Core assertions
        self.assertNotEqual(approved["status"], "failed",
                            f"Fix pipeline failed: {approved.get('error')}\n"
                            f"Orchestrator output:\n{approved.get('orchestrator_output', '')[:3000]}")
        self.assertIsNotNone(approved.get("chosen_source"),
                             "Should have identified a source file")
        self.assertTrue(
            Path(approved["chosen_source"]).name == "multiply_module.py",
            f"Expected multiply_module.py, got {approved.get('chosen_source')}"
        )
        self.assertIsNotNone(approved.get("diagnosis"),
                             "Diagnosis should have been extracted")
        self.assertTrue(
            approved.get("suite_passed"),
            f"Full suite should pass after the fix. "
            f"Retest: {approved.get('retest_report')}"
        )


# ---------------------------------------------------------------------------
# Regression: patch never touches files outside workspace
# ---------------------------------------------------------------------------

class WorkspaceSandboxTests(unittest.TestCase):

    def test_patch_outside_workspace_raises_and_leaves_file_untouched(self):
        """Verify the workspace boundary is enforced end-to-end."""
        with tempfile.TemporaryDirectory() as tmp:
            ws_parent = Path(tmp) / "workspaces"
            ws_parent.mkdir()

            job_id = "sandboxtest01"
            ws = ws_parent / job_id
            ws.mkdir()
            (ws / "repo").mkdir()

            # A file that lives OUTSIDE the workspace
            sentinel_file = Path(tmp) / "important_file.py"
            sentinel_file.write_text("# original content, must not change\n")

            import repo_fix_pipeline as rfp
            orig_ws = rfp.WORKSPACES_DIR
            rfp.WORKSPACES_DIR = ws_parent
            try:
                with self.assertRaises(ValueError):
                    apply_fix_to_workspace(job_id, sentinel_file, "# HACKED\n")
            finally:
                rfp.WORKSPACES_DIR = orig_ws

            # File must be completely untouched
            self.assertEqual(
                sentinel_file.read_text(),
                "# original content, must not change\n",
                "File outside workspace was modified — sandbox violation!"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
