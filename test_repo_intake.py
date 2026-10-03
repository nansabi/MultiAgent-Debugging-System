import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from repo_intake import (
  MAX_REPO_BYTES,
  LocalRunner,
    CommandFailure,
  OutputSizeLimitExceeded,
  clone_repo,
  parse_junit_report,
  validate_github_url,
)


class GithubUrlValidationTests(unittest.TestCase):
    def test_valid_repository_urls(self):
        self.assertTrue(validate_github_url("https://github.com/pallets/flask"))
        self.assertTrue(validate_github_url("https://github.com/pallets/flask.git"))

    def test_invalid_repository_urls(self):
        for url in (
            "",
            "http://github.com/pallets/flask",
            "https://github.com/pallets",
            "https://github.com/pallets/flask/tree/main",
            "https://github.com.evil.test/pallets/flask",
            "https://github.com/pallets/flask?tab=readme",
            "https://github.com/a/b; rm -rf /",
            "https://github.com:443/a/b",
            "https://user@github.com/a/b",
        ):
            with self.subTest(url=url):
                self.assertFalse(validate_github_url(url))


class JunitParsingTests(unittest.TestCase):
    def test_parses_pytest_junit_xml(self):
        sample_xml = '''<?xml version="1.0" encoding="utf-8"?>
<testsuites tests="3" failures="1" errors="1" skipped="0">
  <testsuite name="pytest" tests="3" failures="1" errors="1" skipped="0">
    <testcase classname="tests.test_math" name="test_add" file="tests/test_math.py" line="4" />
    <testcase classname="tests.test_math" name="test_subtract" file="tests/test_math.py" line="8">
      <failure message="assert 1 == 2" type="AssertionError">tests/test_math.py:9: AssertionError</failure>
    </testcase>
    <testcase classname="tests.test_math" name="test_setup" file="tests/test_math.py">
      <error message="fixture failed" type="RuntimeError">RuntimeError: fixture failed</error>
    </testcase>
  </testsuite>
</testsuites>'''
        report = parse_junit_report(sample_xml, duration=0.125)

        self.assertEqual(report["total"], 3)
        self.assertEqual(report["passed"], 1)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["errors"], 1)
        self.assertEqual(report["duration"], 0.125)
        self.assertEqual(report["failures"][0]["test_id"], "tests.test_math::test_subtract")
        self.assertEqual(report["failures"][0]["line"], 9)
        self.assertEqual(report["failures"][1]["error_type"], "RuntimeError")


class CloneLimitsTests(unittest.TestCase):
    def test_clone_uses_60_second_timeout_and_200_mb_cap(self):
        runner = Mock()
        runner.run.return_value = subprocess.CompletedProcess([], 0, "", "")
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            clone_repo("https://github.com/example/project", workspace, runner=runner)

        runner.run.assert_called_once()
        kwargs = runner.run.call_args.kwargs
        self.assertEqual(kwargs["timeout"], 60)
        self.assertEqual(kwargs["max_bytes"], MAX_REPO_BYTES)

    def test_clone_timeout_has_friendly_message_and_preserves_git_output(self):
        runner = Mock()
        runner.run.side_effect = subprocess.TimeoutExpired(
            "git", 60, output=b"Filtering content: 35% (5/14)"
        )
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                CommandFailure,
                "This repository is too large or the connection is slow",
            ) as raised:
                clone_repo("https://github.com/example/project", Path(temp), runner=runner)

        self.assertIn("Filtering content", raised.exception.output)

    def test_clone_size_limit_has_friendly_message(self):
        runner = Mock()
        runner.run.side_effect = OutputSizeLimitExceeded(
            "Command output exceeded the size limit.", "LFS content exceeded limit"
        )
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(CommandFailure, "200 MiB checkout size limit"):
                clone_repo("https://github.com/example/project", Path(temp), runner=runner)

    def test_local_runner_kills_process_after_size_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            script = "from pathlib import Path; import time; Path('large.bin').write_bytes(b'x' * 4096); time.sleep(5)"
            with self.assertRaises(OutputSizeLimitExceeded):
                LocalRunner().run(
                    [sys.executable, "-c", script],
                    cwd=workspace,
                    timeout=10,
                    max_bytes=1024,
                    monitored_path=workspace,
                )


class IntakeDockerVenvTests(unittest.TestCase):
    def test_install_dependencies_bootstraps_linux_venv_without_ensurepip(self):
        """Docker intake must create the venv with system Python, then install pip separately."""
        from docker_runner import DockerRunner
        import repo_intake

        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp).resolve()
            repo = workspace / "repo"
            repo.mkdir()
            (repo / "requirements.txt").write_text("", encoding="utf-8")

            runner = object.__new__(DockerRunner)
            runner.workspace = workspace
            mapped_commands = []

            def fake_run(cmd, cwd, timeout, **kwargs):
                mapped = runner._map_command(cmd)
                mapped_commands.append(mapped)
                if mapped[0] == "python" and mapped[1:4] == ["-m", "venv", "--without-pip"]:
                    linux_python = workspace / ".venv" / "bin" / "python"
                    linux_python.parent.mkdir(parents=True, exist_ok=True)
                    linux_python.write_text("docker venv interpreter", encoding="utf-8")
                return subprocess.CompletedProcess(mapped, 0, "", "")

            runner.run = fake_run

            result = repo_intake.install_dependencies(repo, runner=runner)

            self.assertEqual(Path(result["python"]), workspace / ".venv" / "bin" / "python")
            venv_command = mapped_commands[0]
            self.assertEqual(venv_command[0], "python")
            self.assertEqual(venv_command[1:4], ["-m", "venv", "--without-pip"])
            self.assertFalse(any("ensurepip" in part for command in mapped_commands for part in command))

            pip_commands = [command for command in mapped_commands if command[1:3] == ["-m", "pip"]]
            self.assertTrue(pip_commands)
            self.assertEqual(pip_commands[0][5:], ["install", "pip"])
            self.assertEqual(pip_commands[0][0], "python")
            self.assertEqual(pip_commands[0][3:5], ["--python", "/workspace/.venv/bin/python"])


class NoTestIntakeTests(unittest.TestCase):
    def test_detect_stack_marks_python_without_pytest_as_manual_verification(self):
        from repo_intake import detect_stack

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "agents" / "rag_agent.py"
            source.parent.mkdir()
            source.write_text("VALUE = 1\n", encoding="utf-8")
            stack = detect_stack(root)

        self.assertEqual(stack["language"], "python")
        self.assertIsNone(stack["test_framework"])
        self.assertIsNone(stack["test_command"])
        self.assertEqual(stack["verification_mode"], "none")

    def test_python_files_only_inside_generated_directory_remain_unsupported(self):
        from repo_intake import detect_stack

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generated = root / ".venv" / "Lib"
            generated.mkdir(parents=True)
            (generated / "third_party.py").write_text("VALUE = 1\n", encoding="utf-8")
            stack = detect_stack(root)

        self.assertEqual(stack["language"], "unsupported")

    def test_no_test_intake_finishes_without_installing_or_running_code(self):
        import repo_intake

        with tempfile.TemporaryDirectory() as temp:
            workspaces = Path(temp) / "workspaces"
            workspaces.mkdir()
            job = {
                "job_id": "no-test-job",
                "stage": "cloning",
                "status": "running",
                "logs": [],
                "steps": [],
            }

            def fake_clone(url, workspace, runner=None):
                repo = workspace / "repo"
                repo.mkdir()
                (repo / "pyproject.toml").write_text(
                    "[project]\nname='sample'\n", encoding="utf-8"
                )
                (repo / "src.py").write_text("VALUE = 1\n", encoding="utf-8")
                return repo

            with patch.object(repo_intake, "WORKSPACES_DIR", workspaces), \
                 patch.object(repo_intake, "cleanup_old_workspaces"), \
                 patch.object(repo_intake, "clone_repo", side_effect=fake_clone), \
                 patch.object(repo_intake, "install_dependencies") as install, \
                 patch.object(repo_intake, "run_baseline_tests") as run_tests:
                result = repo_intake.run_intake(
                    "https://github.com/example/sample",
                    job=job,
                    runner=object(),
                )

            self.assertEqual(result["status"], "done")
            self.assertEqual(result["result"]["verification_mode"], "none")
            self.assertEqual(result["result"]["failures"], [])
            self.assertIn("No dependencies were installed", result["result"]["message"])
            install.assert_not_called()
            run_tests.assert_not_called()
            self.assertTrue((workspaces / "no-test-job" / "baseline_report.json").is_file())


if __name__ == "__main__":
    unittest.main()