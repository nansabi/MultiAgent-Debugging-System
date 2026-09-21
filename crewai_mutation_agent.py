"""
crewai_mutation_agent.py
Standalone CrewAI version of the Mutation Re-check Agent only.
Runs the mutation engine, then asks the agent to triage survivors.
Does not touch orchestrator.py.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, LLM

MODEL = "gemini/gemini-3-flash-preview"

MUTATION_SYSTEM_PROMPT = """You are the Mutation Triage Agent in a multi-agent debugging system.
You are given a NUMBERED list of mutants (small code changes) that survived the test suite after
a fix. For EACH numbered mutant, you must ACTUALLY COMPUTE, not guess: pick one concrete input
from the test file's domain, compute what the ORIGINAL (unmutated) code returns for it, then
compute what the MUTATED code returns for that same input. Report both values. If they differ,
verdict is "real_coverage_gap". If you truly cannot find any input where they differ after
checking at least one realistic case, verdict is "equivalent_mutant".

Use the given number (as a string, e.g. "1", "2") as mutant_id — do not invent your own ids.

Respond with ONLY a JSON object, no prose, no markdown fences, in this exact schema:
{
  "verdicts": [
    {
      "mutant_id": "...",
      "verdict": "equivalent_mutant | real_coverage_gap",
      "test_input": "a short concrete function call, e.g. calculate_shipping(30.00)",
      "orig_output": "the value the original code returns",
      "mut_output": "the value the mutated code returns"
    }
  ]
}"""

def parse_survivors(raw_report):
    marker = "Surviving mutants (tests did NOT catch these):"
    idx = raw_report.find(marker)
    if idx == -1:
        return []
    lines = raw_report[idx:].splitlines()[1:]
    survivors = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            survivors.append(stripped[2:])
        elif stripped.startswith("===") or stripped.startswith("---"):
            break
    return survivors

def run_mutation_check(source_path, tests_path):
    engine_path = Path(__file__).parent / "simple_mutation_test.py"
    if not engine_path.exists():
        print("ERROR: simple_mutation_test.py not found in this folder.")
        return

    print("Running mutation engine...")
    result = subprocess.run(
        [sys.executable, str(engine_path), source_path, tests_path],
        capture_output=True, text=True,
    )
    raw_report = result.stdout.strip()

    if not raw_report:
        print("Mutation engine produced no output.")
        return

    survivors = parse_survivors(raw_report)
    print(f"\nFound {len(survivors)} surviving mutant(s).")

    if not survivors:
        print("All mutants killed. No triage needed.")
        return

    numbered_list = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(survivors))

    llm = LLM(model=MODEL)

    mutation_agent = Agent(
        role="Mutation Triage Agent",
        goal="Determine whether each surviving mutant is a real coverage gap or harmless",
        backstory=MUTATION_SYSTEM_PROMPT,
        llm=llm,
        verbose=True,
    )

    test_code = Path(tests_path).read_text(encoding="utf-8")
    source_code = Path(source_path).read_text(encoding="utf-8")

    mutation_task = Task(
        description=(
            f"Patched source code:\n{source_code}\n\n"
            f"Test file:\n{test_code}\n\n"
            f"Survived mutants (numbered — use these exact numbers as mutant_id):\n{numbered_list}\n\n"
            f"Triage each one. Respond ONLY with the JSON schema described in your backstory."
        ),
        expected_output="A single JSON object matching the required schema, nothing else.",
        agent=mutation_agent,
    )

    crew = Crew(agents=[mutation_agent], tasks=[mutation_task], verbose=True)
    result = crew.kickoff()

    raw_text = str(result).strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        mutation_json = json.loads(raw_text)
    except json.JSONDecodeError:
        print("Could not parse JSON. Raw output:")
        print(raw_text)
        return

    print("\n=== Mutation Triage Result ===")
    print(json.dumps(mutation_json, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--tests", required=True)
    args = parser.parse_args()
    run_mutation_check(args.source, args.tests)