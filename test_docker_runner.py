import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from docker_runner import DockerRunner


class DockerRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="sentinel-docker-test-")
        self.root = Path(self.temporary_directory.name)
        self.workspace = self.root / "job"
        self.workspace.mkdir()
        self.runner = DockerRunner(self.workspace)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_runs_command_and_captures_stdout(self):
        result = self.runner.run(
            ["python", "-c", "print('docker-command-ok')"],
            self.workspace,
            timeout=15,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.stdout.strip(), "docker-command-ok")
        self.assertEqual(result.stderr, "")

    def test_timeout_kills_container(self):
        result = self.runner.run(
            ["python", "-c", "import time; time.sleep(15)"],
            self.workspace,
            timeout=1,
        )
        self.assertTrue(result.timed_out)
        self.assertEqual(result.exit_code, 124)
        print(f"Docker timeout: timed_out={result.timed_out}, exit_code={result.exit_code}")

    def test_memory_limit_kills_or_rejects_overallocation(self):
        result = self.runner.run(
            ["python", "-c", "data = bytearray(2 * 1024**3); print(len(data))"],
            self.workspace,
            timeout=20,
            network_enabled=False,
        )
        print(f"Docker memory test: exit_code={result.exit_code}, timed_out={result.timed_out}")
        self.assertNotEqual(result.exit_code, 0)
        self.assertNotIn(str(2 * 1024**3), result.stdout)

    def test_cannot_read_host_file_outside_mounted_workspace(self):
        secret = self.root / "host-secret.txt"
        secret.write_text("HOST_SECRET_SENTINEL", encoding="utf-8")
        test_file = self.workspace / "test_host_escape.py"
        test_file.write_text(
            "import os\n"
            "import unittest\n"
            "\n"
            "class HostEscapeTest(unittest.TestCase):\n"
            "    def test_read_host_sibling(self):\n"
            "        status = os.system('cat /workspace/../host-secret.txt > /workspace/stolen.txt')\n"
            "        self.assertEqual(status, 0, 'host file read was blocked')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        result = self.runner.run(
            ["python", "-m", "unittest", "-v", "test_host_escape"],
            self.workspace,
            timeout=15,
            network_enabled=False,
        )
        output = result.stdout + result.stderr
        print("Host-file isolation test output:\n" + output.strip())
        self.assertNotEqual(result.exit_code, 0, "the malicious host-file access should fail")
        self.assertIn("host file read was blocked", output)
        self.assertNotIn("HOST_SECRET_SENTINEL", output)

    def test_network_request_is_blocked_during_test(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"host-network-reached")

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("0.0.0.0", 0), Handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        test_file = self.workspace / "test_network_escape.py"
        test_file.write_text(
            "from urllib.request import urlopen\n"
            "import unittest\n"
            "\n"
            "class NetworkEscapeTest(unittest.TestCase):\n"
            "    def test_request_reaches_host(self):\n"
            f"        with urlopen('http://host.docker.internal:{server.server_port}/', timeout=3) as response:\n"
            "            self.assertEqual(response.read(), b'host-network-reached')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        try:
            result = self.runner.run(
                ["python", "-m", "unittest", "-v", "test_network_escape"],
                self.workspace,
                timeout=12,
                network_enabled=False,
            )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

        output = result.stdout + result.stderr
        print("Network isolation test output:\n" + output.strip())
        self.assertNotEqual(result.exit_code, 0, "the network request should be blocked")
        self.assertNotIn("host-network-reached", output)
        self.assertTrue("URLError" in output or "NameResolutionError" in output or "Network is unreachable" in output)


if __name__ == "__main__":
    unittest.main()
