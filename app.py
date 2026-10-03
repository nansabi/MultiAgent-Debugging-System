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
import threading
import uuid
from copy import deepcopy
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from repo_fix_pipeline import (
    FIX_JOBS,
    FIX_JOBS_LOCK,
    approve_fix,
    export_fix,
    reject_fix,
    run_manual_fix,
    run_fix_for_failure,
)
from repo_intake import WORKSPACES_DIR, configure_runner, get_runner_status, run_intake, validate_github_url

app = Flask(__name__)
RUNNER_STATUS = configure_runner()
ACTIVE_FIXES = {}
ACTIVE_FIXES_LOCK = threading.Lock()

# Resolve paths relative to this file so the server can be started from any cwd.
BASE_DIR = Path(__file__).parent
ORCHESTRATOR = BASE_DIR / "Orchestrator.py"
REPO_JOBS = {}
REPO_JOBS_LOCK = threading.Lock()


def _run_repo_job(job_id, url):
    with REPO_JOBS_LOCK:
        job = REPO_JOBS[job_id]
    run_intake(url, job=job, job_lock=REPO_JOBS_LOCK)


@app.get("/")
def index():
    return render_template("index.html", runner_status=get_runner_status())


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


@app.post("/repo/analyze")
def analyze_repository():
    data = request.get_json(silent=True)
    url = data.get("url") if isinstance(data, dict) else None
    if not validate_github_url(url):
        return jsonify({"error": "URL must be https://github.com/<owner>/<repo> (optional .git)."}), 400

    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "stage": "cloning",
        "status": "running",
        "logs": ["Intake job accepted."],
        "result": None,
        "steps": [],
    }
    with REPO_JOBS_LOCK:
        REPO_JOBS[job_id] = job
    threading.Thread(target=_run_repo_job, args=(job_id, url), daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.get("/repo/status/<job_id>")
def repository_status(job_id):
    with REPO_JOBS_LOCK:
        job = REPO_JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job_id."}), 404
        snapshot = deepcopy(job)

    response = {
        "job_id": snapshot["job_id"],
        "stage": snapshot["stage"],
        "status": snapshot["status"],
        "logs": snapshot["logs"],
        "steps": snapshot.get("steps", []),
    }
    if snapshot["stage"] == "done":
        response["baseline_report"] = snapshot["result"]
        response["pytest_xml"] = snapshot["result"].get("pytest_xml", "")
        response["verification_mode"] = snapshot["result"].get(
            "verification_mode", "test_verified"
        )
    elif snapshot["stage"] == "failed":
        response["error"] = snapshot["result"].get("error")
    return jsonify(response)


@app.post("/repo/fix/<intake_job_id>")
def start_fix(intake_job_id):
    data = request.get_json(silent=True) or {}
    failure_index = data.get("failure_index")

    if not isinstance(failure_index, int):
        return jsonify({"error": "failure_index must be an integer."}), 400

    if not os.environ.get("GEMINI_API_KEY"):
        return jsonify({"error": "Diagnosis requires an API key - not configured."}), 503

    with REPO_JOBS_LOCK:
        intake_job = REPO_JOBS.get(intake_job_id)
        if intake_job is None:
            return jsonify({"error": "Unknown intake job_id."}), 404
        if intake_job.get("stage") != "done" or intake_job.get("status") != "done":
            return jsonify({"error": "Intake job is not complete yet."}), 409

    fix_key = (intake_job_id, failure_index)
    with ACTIVE_FIXES_LOCK:
        existing_fix_id = ACTIVE_FIXES.get(fix_key)

    if existing_fix_id:
        return jsonify({"fix_job_id": existing_fix_id, "existing": True}), 202

    fix_job_id = uuid.uuid4().hex[:12]
    with ACTIVE_FIXES_LOCK:
        ACTIVE_FIXES[fix_key] = fix_job_id

    threading.Thread(
        target=run_fix_for_failure,
        args=(intake_job_id, failure_index),
        kwargs={"fix_job_id": fix_job_id},
        daemon=True,
    ).start()

    return jsonify({"fix_job_id": fix_job_id, "existing": False}), 202


@app.post("/repo/manual-fix/<intake_job_id>")
def start_manual_fix(intake_job_id):
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required."}), 400
    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
        return jsonify({"error": "description must be a non-empty string."}), 400
    if len(description) > 4000:
        return jsonify({"error": "description must be 4000 characters or fewer."}), 400
    suspected_file = data.get("suspected_file")
    if suspected_file is not None and (
        not isinstance(suspected_file, str) or not suspected_file.strip()
    ):
        return jsonify({"error": "suspected_file must be a non-empty relative path when provided."}), 400
    if not os.environ.get("GEMINI_API_KEY"):
        return jsonify({"error": "Diagnosis requires an API key - not configured."}), 503

    with REPO_JOBS_LOCK:
        intake_job = REPO_JOBS.get(intake_job_id)
        if intake_job is None:
            return jsonify({"error": "Unknown intake job_id."}), 404
        if intake_job.get("stage") != "done" or intake_job.get("status") != "done":
            return jsonify({"error": "Intake job is not complete yet."}), 409
        baseline = intake_job.get("result") or {}
        verification_mode = baseline.get(
            "verification_mode",
            (baseline.get("stack") or {}).get("verification_mode"),
        )
        if verification_mode != "none":
            return jsonify({
                "error": "Manual fixes are only available for Python repositories without pytest.",
            }), 409

    fix_job_id = uuid.uuid4().hex[:12]
    threading.Thread(
        target=run_manual_fix,
        args=(intake_job_id, description.strip()),
        kwargs={
            "suspected_file": suspected_file.strip() if suspected_file else None,
            "fix_job_id": fix_job_id,
        },
        daemon=True,
    ).start()
    return jsonify({
        "fix_job_id": fix_job_id,
        "existing": False,
        "verification_mode": "unverified",
        "final_verdict": "UNVERIFIED_FIX",
    }), 202


@app.get("/repo/fix/status/<fix_job_id>")
def fix_status(fix_job_id):
    with FIX_JOBS_LOCK:
        job = FIX_JOBS.get(fix_job_id)
        if job is None:
            return jsonify({"error": "Unknown fix_job_id."}), 404
        snapshot = deepcopy(job)

    response = {
        "fix_job_id": snapshot.get("fix_job_id"),
        "intake_job_id": snapshot.get("intake_job_id"),
        "failure_index": snapshot.get("failure_index"),
        "verification_mode": snapshot.get("verification_mode", "test_verified"),
        "final_verdict": snapshot.get("final_verdict"),
        "chosen_source": snapshot.get("chosen_source"),
        "source_candidates": snapshot.get("source_candidates", []),
        "manual_review_confirmation_required": (
            snapshot.get("verification_mode") == "unverified"
            and snapshot.get("status") == "pending_review"
        ),
        "stage": snapshot.get("stage"),
        "status": snapshot.get("status"),
        "logs": snapshot.get("logs", []),
        "diagnosis": snapshot.get("diagnosis"),
        "diff": snapshot.get("diff", ""),
        "mutation_verdict": snapshot.get("mutation_verdict"),
        "suite_passed": snapshot.get("suite_passed"),
        "error": snapshot.get("error"),
    }

    if response["error"]:
        error_text = str(response["error"])
        if error_text.startswith("RuntimeError: "):
            error_text = error_text[len("RuntimeError: "):]
        if "Could not identify the source file for this failure" in error_text:
            response["error"] = "Could not identify the source file for this failure"
        else:
            response["error"] = error_text

    return jsonify(response)


@app.get("/repo/fix/audit/<fix_job_id>")
def fix_audit(fix_job_id):
    with FIX_JOBS_LOCK:
        job = FIX_JOBS.get(fix_job_id)
        if job is None:
            return jsonify({"error": "Unknown fix_job_id."}), 404

    audit_path = Path(job.get("audit_path") or "")
    if not audit_path.is_absolute() and job.get("intake_job_id"):
        audit_path = WORKSPACES_DIR / job["intake_job_id"] / "fix_audit.json"

    if audit_path.exists():
        try:
            return jsonify(json.loads(audit_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass

    if job.get("audit"):
        return jsonify(job["audit"])

    return jsonify({"error": "Fix audit not available yet."}), 404


@app.post("/repo/fix/<fix_job_id>/approve")
def approve_fix_route(fix_job_id):
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required."}), 400
    try:
        job = approve_fix(
            fix_job_id,
            manual_review_confirmed=data.get("manual_review_confirmed") is True,
        )
        return jsonify({"status": job["status"], "fix_job_id": fix_job_id, "job": job}), 200
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/repo/fix/<fix_job_id>/reject")
def reject_fix_route(fix_job_id):
    data = request.get_json(silent=True) or {}
    reason = data.get("reason")
    try:
        job = reject_fix(fix_job_id, reason=reason)
        return jsonify({"status": job["status"], "fix_job_id": fix_job_id, "job": job}), 200
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/repo/fix/<fix_job_id>/export")
def export_fix_route(fix_job_id):
    data = request.get_json(silent=True) or {}
    export_format = data.get("format", "patch")
    try:
        payload = export_fix(fix_job_id, export_format=export_format)
        return jsonify(payload), 200
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    # Development server — fine for local demo use.
    app.run(debug=False, port=5000)
