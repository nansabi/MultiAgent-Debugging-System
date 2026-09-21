"""
crewai_diagnosis_test.py
========================
STEP 2–4 of the CrewAI evaluation:

  - Defines one CrewAI Agent (Bug Diagnosis Specialist)
  - Defines one CrewAI Task (diagnose the root cause, return the exact JSON schema
    the downstream Fix Agent depends on)
  - Runs it against the real buggy_code_for_test.py bug (tax-on-wrong-subtotal)
  - Then runs the EXISTING call_agent() Diagnosis Agent against the SAME input
  - Prints both outputs side by side with schema compatibility analysis

DOES NOT modify Orchestrator.py.

CrewAI version: 1.15.22
Model: openai/gpt-oss-120b via Groq's OpenAI-compatible endpoint

LLM routing note (verified from crewai/llm.py source):
  "openai/gpt-oss-120b" has prefix "openai" — CrewAI maps this to the OpenAI
  native provider. Because base_url is supplied and "gpt-oss-120b" is not in
  CrewAI's validated OpenAI model list, custom_openai_route fires, treating this
  as a custom-endpoint OpenAI call. api_key must be passed explicitly because
  CrewAI's OpenAI provider defaults to OPENAI_API_KEY, not GROQ_API_KEY.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # loads .env from project root if present

# --------------------------------------------------------------------------
# 0. Environment check
# --------------------------------------------------------------------------
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    sys.exit(
        "ERROR: GROQ_API_KEY environment variable not set.\n"
        "  PowerShell:  $env:GROQ_API_KEY = 'gsk_...'\n"
        "  Alternatively: create a .env file with GROQ_API_KEY=gsk_..."
    )

# --------------------------------------------------------------------------
# 1. Collect the inputs (source, test file, pytest output)
# --------------------------------------------------------------------------
SOURCE_FILE = Path("buggy_code_for_test.py")
TEST_FILE   = Path("test_buggy_code_for_test.py")

if not SOURCE_FILE.exists() or not TEST_FILE.exists():
    sys.exit(
        "ERROR: buggy_code_for_test.py and/or test_buggy_code_for_test.py "
        "not found. Run this script from the project root directory."
    )

print("=" * 70)
print("CrewAI Diagnosis Agent — Evaluation Script")
print("=" * 70)
print(f"\n[+] Source file : {SOURCE_FILE}")
print(f"[+] Test file   : {TEST_FILE}")

source_code = SOURCE_FILE.read_text(encoding="utf-8")
test_code   = TEST_FILE.read_text(encoding="utf-8")

# Run pytest to capture real failure output
print("\n[+] Running pytest to capture real failure output...")
pytest_result = subprocess.run(
    [sys.executable, "-m", "pytest", "-v", str(TEST_FILE)],
    capture_output=True, text=True,
)
pytest_output = pytest_result.stdout + pytest_result.stderr
print(f"    Exit code: {pytest_result.returncode}")
print(f"    Output preview (first 400 chars):\n    {pytest_output[:400]!r}")

if pytest_result.returncode == 0:
    print("\n[!] WARNING: pytest returned exit code 0 — all tests passed.")
    print("    The bug may not be present. Proceeding anyway for evaluation.")

# --------------------------------------------------------------------------
# 2. Build the diagnosis task input (matches what Orchestrator.py builds)
# --------------------------------------------------------------------------
DIAG_INPUT_TEMPLATE = (
    "Source file:\n{source_code}\n\n"
    "Test file:\n{test_code}\n\n"
    "Pytest output:\n{pytest_output}"
)

diag_input_str = DIAG_INPUT_TEMPLATE.format(
    source_code=source_code,
    test_code=test_code,
    pytest_output=pytest_output,
)

# --------------------------------------------------------------------------
# 3. CREWAI VERSION
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("APPROACH A — CrewAI Agent/Task/Crew")
print("=" * 70)

from crewai import Agent, Task, Crew, Process, LLM

# Configure LLM for Groq via OpenAI-compatible endpoint.
# base_url triggers custom_openai_route in CrewAI 1.15.22's LLM factory.
diagnosis_llm = LLM(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
    max_tokens=1500,
)

print(f"\n[+] LLM configured: model={diagnosis_llm.model}, type={type(diagnosis_llm).__name__}")

# --- Agent ---
diagnosis_agent = Agent(
    role="Bug Diagnosis Specialist",
    goal=(
        "Identify the root cause of a failing test suite. "
        "Do NOT write any code fix — only diagnose. "
        "Return your diagnosis as a single JSON object matching the exact schema provided."
    ),
    backstory=(
        "You are a senior software debugging expert embedded in an automated "
        "multi-agent repair pipeline. Your job is strictly diagnosis, not repair. "
        "You read source code and failing test output, reason about the root cause, "
        "and identify the exact faulty location and category of the bug. "
        "Another agent downstream reads your JSON output to write the fix — "
        "it depends on your field names being exactly correct."
    ),
    llm=diagnosis_llm,
    verbose=False,
    allow_delegation=False,
)

print("[+] Diagnosis Agent created.")

# --- Task ---
# The expected_output explicitly states the required JSON schema so CrewAI
# cannot silently reformat it. The description contains the actual inputs
# as a single formatted string (since CrewAI 1.x supports {variable} placeholders
# for kickoff inputs, but a pre-formatted string also works when input_keys
# is not needed for routing).
diagnosis_task = Task(
    description=(
        "You are given source code, the test file, and pytest output showing test failures.\n\n"
        "Diagnose the root cause of the test failure(s).\n"
        "Do NOT write any code fixes. Do NOT suggest code. Only diagnose.\n\n"
        "You MUST respond with ONLY a JSON object — no prose, no markdown fences, "
        "no explanation outside the JSON. Use this EXACT schema:\n"
        "{\n"
        '  "root_cause": "...",\n'
        '  "explanation": "...",\n'
        '  "faulty_location": "...",\n'
        '  "category": "logic_error | edge_case | wrong_assumption | test_bug | environment_issue",\n'
        '  "confidence": "High | Medium | Low",\n'
        '  "suggested_fix_direction": "..."\n'
        "}\n\n"
        "Here are the inputs:\n\n"
        "{diag_input}"
    ),
    expected_output=(
        'A single JSON object with exactly these six fields and no other text:\n'
        '  "root_cause"            — one sentence identifying the bug\n'
        '  "explanation"           — 2-4 sentences of reasoning\n'
        '  "faulty_location"       — function name and line reference\n'
        '  "category"              — one of: logic_error, edge_case, wrong_assumption, test_bug, environment_issue\n'
        '  "confidence"            — one of: High, Medium, Low\n'
        '  "suggested_fix_direction" — how to fix it, without writing code\n'
        'No markdown, no prose, no code blocks. Pure JSON only.'
    ),
    agent=diagnosis_agent,
)

print("[+] Task created.")

# --- Crew ---
diagnosis_crew = Crew(
    agents=[diagnosis_agent],
    tasks=[diagnosis_task],
    process=Process.sequential,
    verbose=False,
)

print("[+] Crew created. Kicking off...\n")

crewai_start = time.perf_counter()
crewai_result = diagnosis_crew.kickoff(inputs={"diag_input": diag_input_str})
crewai_elapsed = time.perf_counter() - crewai_start

crewai_raw = str(crewai_result)
print(f"\n[+] CrewAI kickoff complete in {crewai_elapsed:.2f}s")
print(f"\n--- CrewAI Raw Output ---\n{crewai_raw}\n")

# Attempt JSON parse (strip markdown fences if present)
crewai_json_str = re.sub(r"^```json\s*|\s*```$", "", crewai_raw.strip(), flags=re.DOTALL)
try:
    crewai_parsed = json.loads(crewai_json_str)
    crewai_parse_ok = True
except json.JSONDecodeError as e:
    crewai_parsed = {}
    crewai_parse_ok = False
    crewai_parse_error = str(e)

# --------------------------------------------------------------------------
# 4. EXISTING call_agent() VERSION (from Orchestrator.py, unmodified)
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("APPROACH B — Existing call_agent() (Orchestrator.py, unchanged)")
print("=" * 70)

# Import the existing infrastructure directly — we do NOT call main(),
# just reuse call_agent() and DIAGNOSIS_SYSTEM_PROMPT exactly as they are.
sys.path.insert(0, str(Path(__file__).parent))
from Orchestrator import call_agent, DIAGNOSIS_SYSTEM_PROMPT, CALL_STATS

print("\n[+] Calling existing Diagnosis Agent via call_agent()...\n")

existing_start = time.perf_counter()
existing_parsed = call_agent(
    agent_name="Diagnosis",
    system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
    user_message=diag_input_str,
    max_tokens=1500,
)
existing_elapsed = time.perf_counter() - existing_start
existing_parse_ok = isinstance(existing_parsed, dict) and bool(existing_parsed)

print(f"\n[+] Existing agent complete in {existing_elapsed:.2f}s")
print(f"\n--- Existing Agent Raw Output (parsed) ---\n{json.dumps(existing_parsed, indent=2)}\n")

# --------------------------------------------------------------------------
# 5. SIDE-BY-SIDE COMPARISON
# --------------------------------------------------------------------------
REQUIRED_FIELDS = {
    "root_cause", "explanation", "faulty_location",
    "category", "confidence", "suggested_fix_direction"
}

def check_schema(parsed: dict) -> tuple[bool, list[str]]:
    """Returns (schema_ok, list_of_missing_fields)."""
    if not isinstance(parsed, dict):
        return False, list(REQUIRED_FIELDS)
    missing = [f for f in REQUIRED_FIELDS if f not in parsed]
    return len(missing) == 0, missing

crewai_schema_ok, crewai_missing = check_schema(crewai_parsed)
existing_schema_ok, existing_missing = check_schema(existing_parsed)

print("\n" + "=" * 70)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 70)

print("\n┌─────────────────────────────────┬─────────────────────────────────┐")
print("│  APPROACH A (CrewAI)            │  APPROACH B (call_agent)        │")
print("├─────────────────────────────────┼─────────────────────────────────┤")
print(f"│  Latency: {crewai_elapsed:.2f}s{' '*(22-len(f'{crewai_elapsed:.2f}s'))}│  Latency: {existing_elapsed:.2f}s{' '*(22-len(f'{existing_elapsed:.2f}s'))}│")
print(f"│  JSON parse: {'✓ OK' if crewai_parse_ok else '✗ FAILED':<20} │  JSON parse: {'✓ OK' if existing_parse_ok else '✗ FAILED':<20} │")
print(f"│  Schema OK:  {'✓' if crewai_schema_ok else '✗ MISSING: ' + str(crewai_missing):<20} │  Schema OK:  {'✓' if existing_schema_ok else '✗ MISSING: ' + str(existing_missing):<20} │")
print("└─────────────────────────────────┴─────────────────────────────────┘")

print("\n--- Field-by-field comparison ---")
for field in sorted(REQUIRED_FIELDS):
    crewai_val  = crewai_parsed.get(field,  "<MISSING>") if crewai_parse_ok  else "<PARSE FAILED>"
    existing_val = existing_parsed.get(field, "<MISSING>") if existing_parse_ok else "<PARSE FAILED>"
    print(f"\n  [{field}]")
    print(f"    CrewAI   : {crewai_val}")
    print(f"    Existing : {existing_val}")

# --------------------------------------------------------------------------
# 6. SCHEMA COMPATIBILITY VERDICT
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SCHEMA COMPATIBILITY VERDICT")
print("=" * 70)

if crewai_parse_ok and crewai_schema_ok:
    print("\n✅ CrewAI output parsed to valid JSON with all required fields present.")
    print("   The downstream Fix Agent (which reads these exact field names) would")
    print("   be able to consume this output without modification.")
elif crewai_parse_ok and not crewai_schema_ok:
    print(f"\n⚠️  CrewAI output parsed to JSON but is MISSING fields: {crewai_missing}")
    print("   The downstream Fix Agent would fail or produce degraded output.")
    print("   Root cause: CrewAI may have added prose, or the model reformatted the schema.")
else:
    print("\n❌ CrewAI output did NOT parse as JSON.")
    print("   The downstream Fix Agent would receive garbage input.")
    if not crewai_parse_ok:
        print(f"   JSON parse error: {crewai_parse_error}")
    print(f"\n   Full raw CrewAI output:\n{crewai_raw}")

print("\n--- Raw CrewAI output (for manual inspection) ---")
print(repr(crewai_raw))
print("\n--- CrewAI result object type ---")
print(type(crewai_result).__name__, "— fields:", [x for x in dir(crewai_result) if not x.startswith('_')])
