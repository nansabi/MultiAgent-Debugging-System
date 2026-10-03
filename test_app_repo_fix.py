from __future__ import annotations

import os
import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import app as sentinel_app
import repo_fix_pipeline as pipeline
from repo_fix_pipeline import FIX_JOBS, FIX_JOBS_LOCK


class RepositoryFixRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = sentinel_app.app.test_client()
        self.intake_id = "route-test-intake"
        self.report = {
            "failures": [
                {"test_id": "test_sample::test_failure", "file": "test_sample.py"}
            ]
        }
        with sentinel_app.REPO_JOBS_LOCK:
            sentinel_app.REPO_JOBS[self.intake_id] = {
                "job_id": self.intake_id,
                "stage": "done",
                "status": "done",
                "result": self.report,
            }
        with FIX_JOBS_LOCK:
            for key in [key for key in sentinel_app.ACTIVE_FIXES if key[0] == self.intake_id]:
                fix_id = sentinel_app.ACTIVE_FIXES.pop(key)
                FIX_JOBS.pop(fix_id, None)

    def tearDown(self):
        with sentinel_app.REPO_JOBS_LOCK:
            sentinel_app.REPO_JOBS.pop(self.intake_id, None)
        with FIX_JOBS_LOCK:
            for key in [key for key in sentinel_app.ACTIVE_FIXES if key[0] == self.intake_id]:
                fix_id = sentinel_app.ACTIVE_FIXES.pop(key)
                FIX_JOBS.pop(fix_id, None)

    def test_missing_api_key_returns_clear_message(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            response = self.client.post(
                f"/repo/fix/{self.intake_id}", json={"failure_index": 0}
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.get_json()["error"],
            "Diagnosis requires an API key - not configured.",
        )

    def test_duplicate_start_reuses_active_fix_job(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def blocking_fix(intake_job_id, failure_index, fix_job_id=None, runner=None):
            calls.append((intake_job_id, failure_index, fix_job_id))
            started.set()
            release.wait(2)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), \
             patch.object(sentinel_app, "run_fix_for_failure", side_effect=blocking_fix):
            first = self.client.post(
                f"/repo/fix/{self.intake_id}", json={"failure_index": 0}
            )
            self.assertEqual(first.status_code, 202)
            fix_id = first.get_json()["fix_job_id"]
            self.assertTrue(started.wait(1), "background fix worker did not start")

            second = self.client.post(
                f"/repo/fix/{self.intake_id}", json={"failure_index": 0}
            )
            self.assertEqual(second.status_code, 202)
            self.assertEqual(second.get_json()["fix_job_id"], fix_id)
            self.assertTrue(second.get_json()["existing"])
            self.assertEqual(len(calls), 1)
        release.set()

    def test_status_exposes_final_diagnosis_diff_verdict_and_suite(self):
        fix_id = "route-test-fix"
        with FIX_JOBS_LOCK:
            FIX_JOBS[fix_id] = {
                "fix_job_id": fix_id,
                "intake_job_id": self.intake_id,
                "failure_index": 0,
                "stage": "done",
                "status": "done",
                "logs": ["Full suite passed after patch."],
                "diagnosis": {"root_cause": "Addition used instead of multiplication."},
                "diff": "- return a + b\n+ return a * b",
                "mutation_verdict": "ACCEPT_FIX",
                "suite_passed": True,
                "error": None,
            }
        response = self.client.get(f"/repo/fix/status/{fix_id}")
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["diagnosis"]["root_cause"], "Addition used instead of multiplication.")
        self.assertIn("return a * b", result["diff"])
        self.assertEqual(result["mutation_verdict"], "ACCEPT_FIX")
        self.assertTrue(result["suite_passed"])
        with FIX_JOBS_LOCK:
            FIX_JOBS.pop(fix_id, None)

    def test_unknown_source_error_is_normalized_for_ui(self):
        fix_id = "route-test-no-source"
        with FIX_JOBS_LOCK:
            FIX_JOBS[fix_id] = {
                "fix_job_id": fix_id,
                "stage": "failed",
                "status": "failed",
                "error": "RuntimeError: Could not identify the source file for this failure. traceback only points into libraries.",
            }
        response = self.client.get(f"/repo/fix/status/{fix_id}")
        self.assertEqual(
            response.get_json()["error"],
            "Could not identify the source file for this failure",
        )
        with FIX_JOBS_LOCK:
            FIX_JOBS.pop(fix_id, None)

    def test_manual_fix_route_requires_no_test_intake_and_starts_worker(self):
        self.report = {
            "failures": [],
            "verification_mode": "none",
            "stack": {"language": "python", "verification_mode": "none"},
        }
        with sentinel_app.REPO_JOBS_LOCK:
            sentinel_app.REPO_JOBS[self.intake_id]["result"] = self.report
        started = threading.Event()
        release = threading.Event()
        calls = []

        def blocking_manual(intake_id, description, **kwargs):
            calls.append((intake_id, description, kwargs))
            started.set()
            release.wait(2)

        try:
            with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), \
                 patch.object(sentinel_app, "run_manual_fix", side_effect=blocking_manual):
                response = self.client.post(
                    f"/repo/manual-fix/{self.intake_id}",
                    json={
                        "description": "Conversation history is lost after each response.",
                        "suspected_file": "src/chat.py",
                    },
                )
                self.assertEqual(response.status_code, 202)
                self.assertEqual(response.get_json()["verification_mode"], "unverified")
                self.assertEqual(response.get_json()["final_verdict"], "UNVERIFIED_FIX")
                self.assertTrue(started.wait(1))
                self.assertEqual(calls[0][0], self.intake_id)
                self.assertIn("Conversation history", calls[0][1])
                self.assertEqual(calls[0][2]["suspected_file"], "src/chat.py")
        finally:
            release.set()

    def test_manual_fix_route_rejects_empty_description_and_tested_intake(self):
        self.report = {
            "failures": [{"test_id": "test_sample::test_failure"}],
            "verification_mode": "test_verified",
            "stack": {"language": "python", "verification_mode": "test_verified"},
        }
        with sentinel_app.REPO_JOBS_LOCK:
            sentinel_app.REPO_JOBS[self.intake_id]["result"] = self.report
        empty = self.client.post(
            f"/repo/manual-fix/{self.intake_id}", json={"description": " "}
        )
        self.assertEqual(empty.status_code, 400)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            tested = self.client.post(
                f"/repo/manual-fix/{self.intake_id}",
                json={"description": "A real issue"},
            )
        self.assertEqual(tested.status_code, 409)

    def test_unverified_approval_requires_server_confirmation_and_skips_retest(self):
        with tempfile.TemporaryDirectory() as temp:
            workspaces = Path(temp)
            workspace = workspaces / self.intake_id
            repo = workspace / "repo"
            repo.mkdir(parents=True)
            source = repo / "sample.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            fix_id = "manual-approval-test"
            with FIX_JOBS_LOCK:
                FIX_JOBS[fix_id] = {
                    "fix_job_id": fix_id,
                    "intake_job_id": self.intake_id,
                    "verification_mode": "unverified",
                    "final_verdict": "UNVERIFIED_FIX",
                    "status": "pending_review",
                    "stage": "pending_review",
                    "chosen_source": str(source),
                    "proposed_patch": "VALUE = 2\n",
                    "proposed_files": None,
                    "diff": "-VALUE = 1\n+VALUE = 2\n",
                    "audit": {"verification_mode": "unverified"},
                    "logs": [],
                }

            try:
                with patch.object(pipeline, "WORKSPACES_DIR", workspaces), \
                     patch.object(pipeline, "_ensure_venv_ready") as ensure_venv, \
                     patch.object(pipeline, "run_baseline_tests") as run_tests:
                    denied = self.client.post(
                        f"/repo/fix/{fix_id}/approve", json={}
                    )
                    self.assertEqual(denied.status_code, 400)
                    self.assertEqual(source.read_text(encoding="utf-8"), "VALUE = 1\n")
                    blocked_export = self.client.post(
                        f"/repo/fix/{fix_id}/export", json={"format": "branch"}
                    )
                    self.assertEqual(blocked_export.status_code, 400)

                    approved = self.client.post(
                        f"/repo/fix/{fix_id}/approve",
                        json={"manual_review_confirmed": True},
                    )
                    self.assertEqual(approved.status_code, 200, approved.get_json())
                    job = approved.get_json()["job"]
                    self.assertEqual(job["final_verdict"], "UNVERIFIED_FIX")
                    self.assertEqual(job["verification_mode"], "unverified")
                    self.assertIsNone(job["suite_passed"])
                    self.assertIsNone(job["retest_report"])
                    self.assertEqual(source.read_text(encoding="utf-8"), "VALUE = 2\n")
                    ensure_venv.assert_not_called()
                    run_tests.assert_not_called()
            finally:
                with FIX_JOBS_LOCK:
                    FIX_JOBS.pop(fix_id, None)


class ManualFixPipelineTests(unittest.TestCase):
    def test_description_search_ranks_matching_python_source_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "chatbot.py").write_text(
                "def reply(history):\n    return history\n", encoding="utf-8"
            )
            (root / "billing.py").write_text(
                "def total(items):\n    return sum(items)\n", encoding="utf-8"
            )
            candidates = pipeline.locate_source_for_description(
                "Conversation history disappears after each response.", root
            )
            self.assertEqual(Path(candidates[0]["path"]).name, "chatbot.py")
            with self.assertRaisesRegex(ValueError, "inside the repository"):
                pipeline.locate_source_for_description(
                    "Any issue", root, suspected_file="../outside.py"
                )

    def test_all_orchestrators_create_audited_unverified_proposals_without_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            workspaces = Path(temp)
            fake_agents = SimpleNamespace(
                DIAGNOSIS_SYSTEM_PROMPT="diagnose prompt",
                FIX_SYSTEM_PROMPT="fix prompt",
            )
            calls = []

            def model_call(role, *args, **kwargs):
                calls.append(role)
                if role.startswith("Diagnosis"):
                    return {
                        "root_cause": "Conversation history is not retained between responses.",
                        "confidence": "Medium",
                    }
                return {
                    "patched_code": (
                        "def reply(history, message):\n"
                        "    history.append(message)\n"
                        "    return history\n"
                    ),
                    "change_summary": "Persist each user message in history.",
                }

            fake_agents.call_agent = model_call
            fake_crewai = SimpleNamespace(call_crew=model_call)
            import_module = lambda name: {
                "Orchestrator": fake_agents,
                "crewai_orchestrator": fake_crewai,
            }[name]
            created_fix_ids = []

            try:
                for backend in ("custom", "crewai", "langgraph"):
                    intake_id = f"manual-{backend}"
                    workspace = workspaces / intake_id
                    repo = workspace / "repo"
                    repo.mkdir(parents=True)
                    source = repo / "chatbot.py"
                    source.write_text(
                        "def reply(history, message):\n"
                        "    return [message]\n",
                        encoding="utf-8",
                    )
                    (workspace / "baseline_report.json").write_text(
                        json.dumps({
                            "stack": {
                                "language": "python",
                                "verification_mode": "none",
                            },
                            "verification_mode": "none",
                            "failures": [],
                        }),
                        encoding="utf-8",
                    )
                    with patch.object(pipeline, "WORKSPACES_DIR", workspaces), \
                         patch.object(pipeline.importlib, "import_module", side_effect=import_module), \
                         patch.dict(os.environ, {"SENTINEL_ORCHESTRATOR": backend}):
                        job = pipeline.run_manual_fix(
                            intake_id,
                            "Conversation history is lost after each response.",
                        )

                    created_fix_ids.append(job["fix_job_id"])
                    self.assertEqual(job["status"], "pending_review", backend)
                    self.assertEqual(job["verification_mode"], "unverified")
                    self.assertEqual(job["final_verdict"], "UNVERIFIED_FIX")
                    self.assertIsNone(job["suite_passed"])
                    self.assertIsNone(job["mutation_verdict"])
                    self.assertEqual(source.read_text(encoding="utf-8"), "def reply(history, message):\n    return [message]\n")
                    audit = json.loads(
                        (workspace / "fix_audit.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(audit["verification_mode"], "unverified")
                    self.assertEqual(audit["final_verdict"], "UNVERIFIED_FIX")
                    self.assertEqual(job["orchestrator"], backend)
                    with FIX_JOBS_LOCK:
                        FIX_JOBS.pop(job["fix_job_id"], None)

            finally:
                with FIX_JOBS_LOCK:
                    for fix_id in created_fix_ids:
                        FIX_JOBS.pop(fix_id, None)


if __name__ == "__main__":
    unittest.main()