"""
crewai_diagnosis_agent.py
Standalone CrewAI version of the Diagnosis Agent only.
Runs independently of orchestrator.py — does not touch your working system.
"""

import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, LLM

MODEL = "gemini/gemini-3-flash-preview"

DIAGNOSIS_SYSTEM_PROMPT = """You are the Diagnosis Agent in a multi-agent debugging system.
You do NOT write or fix code. You only diagnose.
Identify the ROOT CAUSE of the failure, point to the exact faulty location, and give a
suggested fix direction (not code). Respond with ONLY a JSON object, no prose, no markdown
fences, in this exact schema:
{
  "root_cause": "...",
  "explanation": "...",
  "faulty_location": "...",
  "category": "logic_error | edge_case | wrong_assumption | test_bug | environment_issue",
  "confidence": "High | Medium | Low",
  "suggested_fix_direction": "..."
}"""

def run_diagnosis(source_path, tests_path):
    source_code = Path(source_path).read_text(encoding="utf-8")
    test_code = Path(tests_path).read_text(encoding="utf-8")

    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", tests_path],
        capture_output=True, text=True,
    )
    test_output = result.stdout + result.stderr
    passed = result.returncode == 0

    if passed:
        print("\n✅ All tests passed. No bugs detected — Diagnosis Agent not invoked.")
        return

    print("\n❌ Tests failed. Invoking Diagnosis Agent...\n")

    llm = LLM(model=MODEL)

    diagnosis_agent = Agent(
        role="Diagnosis Agent",
        goal="Identify the root cause of a failing test without writing any fix",
        backstory=DIAGNOSIS_SYSTEM_PROMPT,
        llm=llm,
        verbose=True,
    )

    diagnosis_task = Task(
        description=(
            f"Source code:\n{source_code}\n\n"
            f"Test file:\n{test_code}\n\n"
            f"Pytest output:\n{test_output}\n\n"
            f"Diagnose the failure. Respond ONLY with the JSON schema described in your backstory."
        ),
        expected_output="A single JSON object matching the required schema, nothing else.",
        agent=diagnosis_agent,
    )

    crew = Crew(agents=[diagnosis_agent], tasks=[diagnosis_task], verbose=True)
    result = crew.kickoff()

    raw_text = str(result).strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        diagnosis_json = json.loads(raw_text)
    except json.JSONDecodeError:
        print("Could not parse JSON. Raw output:")
        print(raw_text)
        return

    print("\n=== Diagnosis Result ===")
    print(json.dumps(diagnosis_json, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--tests", required=True)
    args = parser.parse_args()
    run_diagnosis(args.source, args.tests)