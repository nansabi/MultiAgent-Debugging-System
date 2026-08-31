"""
app.py — Flask wrapper around the existing orchestrator pipeline.

Usage:
    Windows PowerShell:  $env:GROQ_API_KEY = "gsk_..."
    Mac/Linux:           export GROQ_API_KEY="gsk_..."
    python app.py
    Then open http://localhost:5000 in a browser.

The orchestrator.py and simple_mutation_test.py files are invoked as a subprocess —
their logic is not duplicated here.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Resolve paths relative to this file so the server can be started from any cwd.
BASE_DIR = Path(__file__).parent
ORCHESTRATOR = BASE_DIR / "Orchestrator.py"


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/run")
def run_pipeline():
    data = request.get_json(force=True)
    source_code = data.get("source_code", "")
    test_code   = data.get("test_code", "")

    if not source_code.strip() or not test_code.strip():
        return jsonify({"error": "Both source_code and test_code are required."}), 400

    # Write to named temp files that survive long enough for the subprocess to read.
    # delete=False on Windows because subprocess cannot open a file another process
    # has open on that platform.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="source_", delete=False,
        dir=BASE_DIR, encoding="utf-8"
    ) as src_f:
        src_f.write(source_code)
        src_path = src_f.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="tests_", delete=False,
        dir=BASE_DIR, encoding="utf-8"
    ) as tst_f:
        tst_f.write(test_code)
        tst_path = tst_f.name

    try:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(ORCHESTRATOR),
             "--source", src_path, "--tests", tst_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=str(BASE_DIR),
            timeout=600,   # 10-minute hard cap; normal runs finish in 1-3 minutes
        )
        output = result.stdout
        if result.stderr:
            output += "\n--- stderr ---\n" + result.stderr
    except subprocess.TimeoutExpired:
        output = "ERROR: The pipeline timed out after 10 minutes."
    except Exception as e:
        output = f"ERROR: Failed to launch orchestrator: {e}"
    finally:
        # Always clean up temp files.
        for p in (src_path, tst_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    return jsonify({"output": output})


if __name__ == "__main__":
    # Development server — fine for local demo use.
    app.run(debug=False, port=5000)
