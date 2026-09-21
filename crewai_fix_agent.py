"""
crewai_fix_agent.py
Standalone CrewAI version of the Fix Agent only.
Takes a diagnosis (from crewai_diagnosis_agent.py or manually) and produces a patch.
Does not touch orchestrator.py.
"""

import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, LLM

MODEL = "gemini/gemini-3-flash-preview"

FIX_SYSTEM_PROMPT = """You are the Fix Agent in a multi-agent debugging system.
You take a diagnosis and the original source code and produce a minimal, targeted patch.
Do not refactor unrelated code. Do not touch the test file. Preserve signature and docstring
EXACTLY as given, character-for-character outside of the lines you are actually fixing —
this includes quote style, whitespace, comments, and wording. Every line in "patched_code"
that is not part of the actual bug fix must be byte-identical to the corresponding line in
the original source.
Respond with ONLY a JSON object, no prose, no markdown fences, in this exact schema:
{
  "patched_code": "<the FULL corrected file content>",
  "change_summary": "...",
  "diff_explanation": "...",
  "confidence": "High | Medium | Low",
  "risk_notes": "..."
}"""

def run_fix(source_path, diagnosis_json_str):
    source_code = Path(source_path).read_text(encoding="utf-8")
    diagnosis = json.loads(diagnosis_json_str)

    llm = LLM(model=MODEL)

    fix_agent = Agent(
        role="Fix Agent",
        goal="Produce a minimal, targeted patch based on a diagnosis",
        backstory=FIX_SYSTEM_PROMPT,
        llm=llm,
        verbose=True,
    )

    fix_task = Task(
        description=(
            f"Original source code:\n{source_code}\n\n"
            f"Diagnosis:\n{json.dumps(diagnosis)}\n\n"
            f"Produce a minimal patch. Respond ONLY with the JSON schema described in your backstory."
        ),
        expected_output="A single JSON object matching the required schema, nothing else.",
        agent=fix_agent,
    )

    crew = Crew(agents=[fix_agent], tasks=[fix_task], verbose=True)
    result = crew.kickoff()

    raw_text = str(result).strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        fix_json = json.loads(raw_text)
    except json.JSONDecodeError:
        print("Could not parse JSON. Raw output:")
        print(raw_text)
        return

    print("\n=== Fix Result ===")
    print(f"Change: {fix_json['change_summary']}")
    print(f"Confidence: {fix_json['confidence']}")
    print(f"Risk notes: {fix_json['risk_notes']}")

    out_path = Path(source_path).with_name("crewai_patched_" + Path(source_path).name)
    out_path.write_text(fix_json["patched_code"], encoding="utf-8")
    print(f"\nPatched file written to: {out_path}")
    print("(Original file NOT overwritten — review the patch before replacing it.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--diagnosis", required=True, help="Path to a JSON file with the diagnosis")
    args = parser.parse_args()
    diagnosis_str = Path(args.diagnosis).read_text(encoding="utf-8")
    run_fix(args.source, diagnosis_str)