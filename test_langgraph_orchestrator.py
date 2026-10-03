from __future__ import annotations

import os
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

import langgraph_orchestrator as langgraph_runner
import repo_fix_pipeline as pipeline


class LangGraphRetryTests(unittest.TestCase):

    def test_retry_limit_preserves_four_fix_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "calculator.py"
            tests = root / "test_calculator.py"
            source.write_text("def add(a, b): return a - b\n", encoding="utf-8")
            tests.write_text("def test_add(): assert True\n", encoding="utf-8")
            agent_calls = []

            def fake_agent(name, *args, **kwargs):
                agent_calls.append(name)
                if name == "Diagnosis":
                    return {
                        "root_cause": "injected persistent failure",
                        "confidence": "High",
                    }
                if name == "Fix":
                    return {
                        "patched_code": "def add(a, b): return a - b\n",
                        "change_summary": "No effective change",
                    }
                raise AssertionError(f"Unexpected agent call: {name}")

            with patch.object(langgraph_runner.agents, "run_tests", return_value=(False, "still failing")) as test_run, \
                 patch.object(langgraph_runner.agents, "call_agent", side_effect=fake_agent), \
                 patch.object(langgraph_runner.agents, "print_diff"), \
                 patch.object(langgraph_runner.agents, "print_cost_summary"):
                result = langgraph_runner.run(str(source), str(tests), source_root=str(root))

        self.assertEqual(agent_calls.count("Fix"), 4)
        self.assertEqual(agent_calls.count("Diagnosis"), 4)
        self.assertEqual(test_run.call_count, 4)
        self.assertEqual(result["retry_count"], 4)
        self.assertEqual(result["final_verdict"], "COULD_NOT_FIX")
        self.assertEqual(result["result"]["final_verdict"], "COULD_NOT_FIX")
        self.assertEqual(result["result"]["retry_count"], 4)

    def test_multi_file_labels_and_success_result_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a = root / "module_a.py"
            source_b = root / "module_b.py"
            tests = root / "test_modules.py"
            source_a.write_text("def total(value): return value\n", encoding="utf-8")
            source_b.write_text("from module_a import total\n", encoding="utf-8")
            tests.write_text("def test_placeholder(): assert True\n", encoding="utf-8")
            with patch.object(langgraph_runner.agents, "run_tests", return_value=(True, "passed")), \
                 patch.object(langgraph_runner.agents, "run_mutation_testing",
                              return_value="Total mutants: 0\nSurvived: 0"), \
                 patch.object(langgraph_runner.agents, "print_cost_summary"):
                result = langgraph_runner.run(
                    str(source_b), str(tests), [str(source_a)], source_root=str(root)
                )

        self.assertEqual(result["source_paths"], ["module_b.py", "module_a.py"])
        self.assertEqual(
            set(result["result"]),
            {"mutation_summary", "surviving_mutants_analysis", "final_verdict", "confidence"},
        )
        self.assertEqual(result["result"]["final_verdict"], "ACCEPT_WITH_ADDED_TESTS")


class OrchestratorSelectionTests(unittest.TestCase):

    def test_custom_is_default_and_all_three_names_resolve(self):
        with patch.dict(os.environ, {}, clear=True):
            name, path = pipeline._configured_orchestrator()
        self.assertEqual(name, "custom")
        self.assertEqual(path.name, "Orchestrator.py")

        for requested, expected in (
            ("custom", "Orchestrator.py"),
            ("crewai", "crewai_orchestrator.py"),
            ("langgraph", "langgraph_orchestrator.py"),
        ):
            with self.subTest(requested=requested), patch.dict(
                os.environ, {"SENTINEL_ORCHESTRATOR": requested}
            ):
                name, path = pipeline._configured_orchestrator()
                self.assertEqual(name, requested)
                self.assertEqual(path.name, expected)

    def test_unknown_orchestrator_name_is_rejected(self):
        with patch.dict(os.environ, {"SENTINEL_ORCHESTRATOR": "unknown"}):
            with self.assertRaisesRegex(ValueError, "SENTINEL_ORCHESTRATOR"):
                pipeline._configured_orchestrator()

    def test_only_langgraph_receives_multi_file_arguments(self):
        common = (
            "python.exe",
            "D:/repo/source.py",
            "D:/repo/test_source.py",
            "D:/repo",
            ["D:/repo/dependency.py"],
        )
        langgraph = pipeline._build_orchestrator_command(
            "langgraph", "langgraph_orchestrator.py", *common
        )
        self.assertIn("--source-file", langgraph)
        self.assertIn("D:/repo/dependency.py", langgraph)
        self.assertIn("--source-root", langgraph)

        for name in ("custom", "crewai"):
            command = pipeline._build_orchestrator_command(
                name, f"{name}.py", *common
            )
            self.assertNotIn("--source-file", command)
            self.assertNotIn("--source-root", command)


class LangGraphPipelineMultiFileTests(unittest.TestCase):

    def test_transitive_project_imports_are_added_for_langgraph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_a = root / "module_a.py"
            module_b = root / "module_b.py"
            tests = root / "test_modules.py"
            module_a.write_text("def total(value): return value\n", encoding="utf-8")
            module_b.write_text(
                "from module_a import total\n\ndef checkout(value): return total(value)\n",
                encoding="utf-8",
            )
            tests.write_text("from module_b import checkout\n", encoding="utf-8")
            initial = pipeline.locate_source_for_failure(
                {"file": "test_modules.py", "traceback": ""}, root
            )

            expanded = pipeline._expand_imported_source_candidates(initial, root)

        self.assertEqual(
            {Path(item["path"]).name for item in expanded},
            {"module_a.py", "module_b.py"},
        )

    def test_multi_file_approval_checks_and_applies_all_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspaces = Path(tmp)
            repo = workspaces / "job" / "repo"
            repo.mkdir(parents=True)
            (repo / "module_a.py").write_text("VALUE = 1\n", encoding="utf-8")
            (repo / "module_b.py").write_text("VALUE = 2\n", encoding="utf-8")
            baseline = {"failed": 0, "errors": 0}
            with patch.object(pipeline, "WORKSPACES_DIR", workspaces), \
                 patch.object(pipeline, "_ensure_venv_ready"), \
                 patch.object(pipeline, "run_baseline_tests", return_value=baseline):
                result = pipeline.apply_fixes_to_workspace(
                    "job",
                    {"module_a.py": "VALUE = 10\n", "module_b.py": "VALUE = 20\n"},
                    runner=object(),
                )

            self.assertEqual((repo / "module_a.py").read_text(encoding="utf-8"), "VALUE = 10\n")
            self.assertEqual((repo / "module_b.py").read_text(encoding="utf-8"), "VALUE = 20\n")
            self.assertTrue(result["suite_passed"])
            self.assertIn("module_a.py", result["diff"])
            self.assertIn("module_b.py", result["diff"])

    def test_multi_file_approval_rejects_paths_outside_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspaces = Path(tmp)
            workspace = workspaces / "job"
            repo = workspace / "repo"
            repo.mkdir(parents=True)
            sentinel = workspace / "sentinel.py"
            sentinel.write_text("ORIGINAL = True\n", encoding="utf-8")
            with patch.object(pipeline, "WORKSPACES_DIR", workspaces):
                with self.assertRaisesRegex(ValueError, "outside"):
                    pipeline.apply_fixes_to_workspace(
                        "job", {"../sentinel.py": "ORIGINAL = False\n"},
                        runner=object(),
                    )
            self.assertEqual(
                sentinel.read_text(encoding="utf-8"), "ORIGINAL = True\n"
            )

    def test_pipeline_dispatches_and_approves_langgraph_multi_file_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspaces = Path(tmp)
            workspace = workspaces / "intake"
            repo = workspace / "repo"
            repo.mkdir(parents=True)
            original_a = (
                "def calculate_total(subtotal):\n"
                "    return round(subtotal, 2)\n"
            )
            original_b = (
                "from module_a import calculate_total\n\n"
                "def checkout(subtotal):\n"
                "    return calculate_total(subtotal)\n"
            )
            (repo / "module_a.py").write_text(original_a, encoding="utf-8")
            (repo / "module_b.py").write_text(original_b, encoding="utf-8")
            (repo / "test_checkout.py").write_text(
                "from module_a import calculate_total\n"
                "from module_b import checkout\n\n"
                "def test_calculate_total():\n"
                "    assert calculate_total(100, 0.08) == 108\n\n"
                "def test_checkout():\n"
                "    assert checkout(100, 0.08) == 108\n",
                encoding="utf-8",
            )
            (workspace / "baseline_report.json").write_text(
                json.dumps({
                    "failures": [{
                        "file": "test_checkout.py",
                        "test_id": "test_checkout",
                        "error_type": "TypeError",
                        "message": "unexpected tax-rate argument",
                        "traceback": "",
                    }],
                    "stack": {"project_root": str(repo)},
                }),
                encoding="utf-8",
            )

            def fake_orchestrator(command, **kwargs):
                self.assertIn("--source-root", command)
                self.assertIn("--source-file", command)
                staged_root = Path(command[command.index("--source-root") + 1])
                Path(command[command.index("--source") + 1]).write_text(
                    "def calculate_total(subtotal, tax_rate):\n"
                    "    return round(subtotal * (1 + tax_rate), 2)\n",
                    encoding="utf-8",
                )
                (staged_root / "module_b.py").write_text(
                    "from module_a import calculate_total\n\n"
                    "def checkout(subtotal, tax_rate):\n"
                    "    return calculate_total(subtotal, tax_rate)\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(
                    command, 0,
                    "Root cause: tax rate is not accepted or forwarded.\n"
                    "=== Final Verdict: ACCEPT_FIX ===\n"
                    '{"final_verdict":"ACCEPT_FIX"}',
                    "",
                )

            with patch.object(pipeline, "WORKSPACES_DIR", workspaces), \
                 patch.dict(os.environ, {"SENTINEL_ORCHESTRATOR": "langgraph"}), \
                 patch.object(pipeline, "_execution_runner", return_value=object()), \
                 patch.object(pipeline.subprocess, "run", side_effect=fake_orchestrator):
                proposed = pipeline.run_fix_for_failure("intake", 0)
                self.assertEqual(proposed["status"], "pending_review")
                self.assertEqual(
                    set(proposed["proposed_files"]),
                    {"module_a.py", "module_b.py"},
                )
                self.assertEqual((repo / "module_a.py").read_text(encoding="utf-8"), original_a)
                self.assertEqual((repo / "module_b.py").read_text(encoding="utf-8"), original_b)
                with patch.object(pipeline, "_ensure_venv_ready"), \
                     patch.object(
                         pipeline, "run_baseline_tests",
                         return_value={"failed": 0, "errors": 0},
                     ):
                    approved = pipeline.approve_fix(
                        proposed["fix_job_id"], runner=object()
                    )

            self.assertTrue(approved["suite_passed"])
            self.assertIn("tax_rate", (repo / "module_a.py").read_text(encoding="utf-8"))
            self.assertIn("tax_rate", (repo / "module_b.py").read_text(encoding="utf-8"))


class LiveOrchestratorComparisonTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parent

    def setUp(self):
        load_dotenv(self.ROOT / ".env")
        if not os.environ.get("GEMINI_API_KEY"):
            self.skipTest("GEMINI_API_KEY not set; live orchestrator comparisons require Gemini")

    def _run(self, script_name, root, primary, tests, extra_sources=()):
        script = self.ROOT / script_name
        python_executable = (
            os.environ.get("SENTINEL_CREWAI_PYTHON", sys.executable)
            if script_name == "crewai_orchestrator.py"
            else sys.executable
        )
        command = [python_executable, "-X", "utf8", str(script), "--source", str(primary)]
        if script_name == "langgraph_orchestrator.py":
            command.extend(["--source-root", str(root)])
            for source in extra_sources:
                command.extend(["--source-file", str(source)])
        command.extend(["--tests", str(tests)])
        env = dict(os.environ)
        env.pop("SENTINEL_DOCKER_WORKSPACE", None)
        env.pop("SENTINEL_DOCKER_PYTHON", None)
        env["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run(
            command,
            cwd=self.ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=600,
        )

    @staticmethod
    def _verdict(output):
        match = re.search(r"=== Final Verdict:\s*(\S+)", output)
        return match.group(1) if match else None

    def test_same_multiply_bug_through_all_three_orchestrators(self):
        results = {}
        patches = {}
        diagnoses = {}
        for name, script in (
            ("custom", "Orchestrator.py"),
            ("crewai", "crewai_orchestrator.py"),
            ("langgraph", "langgraph_orchestrator.py"),
        ):
            with tempfile.TemporaryDirectory(prefix=f"sentinel-{name}-multiply-") as tmp:
                root = Path(tmp)
                source = root / "multiply_module.py"
                tests = root / "test_multiply.py"
                source.write_text(
                    "def multiply(a, b):\n    return a + b\n", encoding="utf-8"
                )
                tests.write_text(
                    "from multiply_module import multiply\n\n"
                    "def test_product():\n    assert multiply(3, 4) == 12\n"
                    "def test_zero():\n    assert multiply(0, 5) == 0\n",
                    encoding="utf-8",
                )
                completed = self._run(script, root, source, tests)
                transcript = completed.stdout + completed.stderr
                results[name] = (completed.returncode, self._verdict(transcript), transcript)
                patches[name] = source.read_text(encoding="utf-8")
                diagnosis = re.search(r"Root cause:\s*(.+)", transcript)
                diagnoses[name] = diagnosis.group(1).strip() if diagnosis else ""
                print(
                    f"comparison progress: {name} exit={completed.returncode} "
                    f"verdict={results[name][1]} diagnosis={diagnoses[name]}",
                    flush=True,
                )

        for name, (returncode, verdict, transcript) in results.items():
            self.assertEqual(returncode, 0, f"{name} failed:\n{transcript[-5000:]}")
            self.assertEqual(verdict, "ACCEPT_FIX", f"{name}:\n{transcript[-5000:]}")
            self.assertIn("addition", diagnoses[name].lower())
            self.assertIn("multiplication", diagnoses[name].lower())
        self.assertEqual(len(set(patches.values())), 1, patches)
        self.assertIn("return a * b", next(iter(patches.values())))

        print("\n--- Live Orchestrator Comparison: multiply_module.py ---")
        for name in ("custom", "crewai", "langgraph"):
            print(f"{name}: verdict={results[name][1]}; diagnosis={diagnoses[name]}")
            print(f"{name}: patch={patches[name].strip()!r}")

    def test_langgraph_multifile_signature_migration(self):
        with tempfile.TemporaryDirectory(prefix="sentinel-langgraph-multifile-") as tmp:
            root = Path(tmp)
            module_a = root / "module_a.py"
            module_b = root / "module_b.py"
            tests = root / "test_checkout.py"
            original_a = "def calculate_total(subtotal):\n    return round(subtotal, 2)\n"
            original_b = (
                "from module_a import calculate_total\n\n"
                "def checkout(subtotal):\n    return calculate_total(subtotal)\n"
            )
            module_a.write_text(original_a, encoding="utf-8")
            module_b.write_text(original_b, encoding="utf-8")
            tests.write_text(
                "from module_a import calculate_total\n"
                "from module_b import checkout\n\n"
                "def test_calculate_total_applies_tax_rate():\n"
                "    assert calculate_total(100.0, 0.08) == 108.0\n\n"
                "def test_checkout_applies_tax_rate():\n"
                "    assert checkout(100.0, 0.08) == 108.0\n",
                encoding="utf-8",
            )
            completed = self._run(
                "langgraph_orchestrator.py", root, module_b, tests, [module_a]
            )
            transcript = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, transcript[-6000:])
            self.assertNotIn("Retry limit", transcript)
            changed_a = module_a.read_text(encoding="utf-8")
            changed_b = module_b.read_text(encoding="utf-8")
            self.assertNotEqual(changed_a, original_a)
            self.assertNotEqual(changed_b, original_b)

            proof_root = root / "single-file-proofs"
            proof_root.mkdir()
            single_results = {}
            for filename, content in (("module_a.py", changed_a), ("module_b.py", changed_b)):
                variant = proof_root / Path(filename).stem
                variant.mkdir()
                (variant / "module_a.py").write_text(original_a, encoding="utf-8")
                (variant / "module_b.py").write_text(original_b, encoding="utf-8")
                (variant / "test_checkout.py").write_text(
                    tests.read_text(encoding="utf-8"), encoding="utf-8"
                )
                (variant / filename).write_text(content, encoding="utf-8")
                single_results[filename] = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "test_checkout.py"],
                    cwd=variant, capture_output=True, text=True,
                )
            self.assertNotEqual(single_results["module_a.py"].returncode, 0)
            self.assertNotEqual(single_results["module_b.py"].returncode, 0)
            combined = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", str(tests)],
                cwd=root, capture_output=True, text=True,
            )
            self.assertEqual(combined.returncode, 0, combined.stdout + combined.stderr)

            print("\n--- Live LangGraph Multi-file Proof ---")
            print(f"Only module_a.py changed: pytest exit {single_results['module_a.py'].returncode}")
            print(f"Only module_b.py changed: pytest exit {single_results['module_b.py'].returncode}")
            print(f"Both changed: pytest exit {combined.returncode}; {combined.stdout.strip()}")
            print(f"LangGraph verdict: {self._verdict(transcript)}")

    def test_langgraph_hard_case_matches_archived_custom_retry_limit(self):
        archived_root = self.ROOT / "workspaces" / "66d3db8aba54"
        archived_repo = archived_root / "repo"
        audit = json.loads((archived_root / "fix_audit.json").read_text(encoding="utf-8"))
        archived_output = audit["orchestrator_output"]
        self.assertEqual(audit["final_verdict"], "COULD_NOT_FIX")
        self.assertIn("Retry limit", archived_output)
        self.assertEqual(archived_output.count("Invoking Fix Agent"), 4)

        with tempfile.TemporaryDirectory(prefix="sentinel-langgraph-hard-case-") as tmp:
            root = Path(tmp)
            for filename in ("calculator.py", "main.py", "test_calculator.py"):
                shutil.copy2(archived_repo / filename, root / filename)
            completed = self._run(
                "langgraph_orchestrator.py",
                root,
                root / "main.py",
                root / "test_calculator.py",
                [root / "calculator.py"],
            )
            transcript = completed.stdout + completed.stderr

        self.assertEqual(completed.returncode, 0, transcript[-6000:])
        verdict = self._verdict(transcript)
        if verdict == "COULD_NOT_FIX":
            self.assertIn("Retry limit (3)", transcript)
            self.assertEqual(transcript.count("Invoking Fix Agent"), 4)
            self.assertEqual(transcript.count("Sandbox run (attempt"), 4)
        else:
            self.assertIn(verdict, {"ACCEPT_FIX", "ACCEPT_WITH_ADDED_TESTS"})
            self.assertIn("All tests passed.", transcript)
        print("\n--- Hard-case Retry Comparison ---")
        print("archived custom: COULD_NOT_FIX; 4 Fix calls; Retry limit (3)")
        print(f"LangGraph replay: {verdict}; {transcript.count('Invoking Fix Agent')} Fix call(s)")


if __name__ == "__main__":
    unittest.main(verbosity=2)