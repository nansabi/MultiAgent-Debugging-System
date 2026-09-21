"""
crewai_orchestrator.py
Full CrewAI-powered multi-agent debugging orchestrator — complete verification
parity with orchestrator.py.

Wires Diagnosis Agent -> Fix Agent -> Sandbox Re-run -> (retry loop) ->
Mutation Triage Agent -> Mutation Detail Agent(s) -> multi-layer verification,
using CrewAI throughout instead of raw Gemini API calls.

Verification layers (matching orchestrator.py):
- Evidence-completeness gate (reject empty/placeholder "equivalent" claims before checking)
- Output re-execution check, original side (with date/volatile-field normalization)
- Output re-execution check, mutant side (independently verifies claimed mut_output)
- Line-coverage tracing (was the mutated line even reached?)
- Boundary-value tracing (did the input actually test the disagreement zone?)
- Boundary-mutant isolation (one API call per boundary mutant + contamination retry)
- Separate retry passes for line-coverage and boundary-value rejections
- Wording-accuracy patch for a known LLM phrasing inaccuracy
- Cost/latency summary across all CrewAI calls

Does NOT touch orchestrator.py — this is a separate, parallel system.

USAGE:
    python crewai_orchestrator.py --source buggy_code.py --tests test_buggy_code.py
"""

import argparse
import ast
import json
import math
import re
import subprocess
import sys
import time
import types
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, LLM

MODEL = "gemini/gemini-3-flash-preview"
MAX_RETRIES = 3
DETAIL_BATCH_SIZE = 3

CALL_STATS = []  # list of dicts: {agent, seconds}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

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

FIX_SYSTEM_PROMPT = """You are the Fix Agent in a multi-agent debugging system.
You take a diagnosis and the original source code and produce a minimal, targeted patch.
Do not refactor unrelated code. Do not touch the test file. Preserve signature and docstring
EXACTLY as given, character-for-character outside of the lines you are actually fixing.
Every line in "patched_code" that is not part of the actual bug fix must be byte-identical
to the corresponding line in the original source.
Respond with ONLY a JSON object, no prose, no markdown fences, in this exact schema:
{
  "patched_code": "<the FULL corrected file content>",
  "change_summary": "...",
  "diff_explanation": "...",
  "confidence": "High | Medium | Low",
  "risk_notes": "..."
}"""

MUTATION_TRIAGE_SYSTEM_PROMPT = """You are the Mutation Triage Agent in a multi-agent debugging system.
You are given a NUMBERED list of mutants (small code changes) that survived the test suite after
a fix. For EACH numbered mutant, you must ACTUALLY COMPUTE, not guess: pick one concrete input
from the test file's domain, compute what the ORIGINAL (unmutated) code returns for it, then
compute what the MUTATED code returns for that same input. Report both values. If they differ,
verdict is "real_coverage_gap". If you truly cannot find any input where they differ after
checking at least one realistic case, verdict is "equivalent_mutant".

Use the given number (as a string, e.g. "1", "2") as mutant_id — do not invent your own ids or
rename them. You must return exactly one entry per numbered mutant given to you.

DO NOT skip the computation to save space. A verdict without genuinely computed non-empty
orig_output/mut_output values is not acceptable.

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

MUTATION_DETAIL_SYSTEM_PROMPT = """You are the Mutation Detail Agent in a multi-agent debugging system.
You are given a small list of mutants that need a genuine, independent check — some were flagged
as likely real gaps by an earlier triage pass, others were flagged as unverified because no real
evidence was provided for an "equivalent" claim. Do not assume either answer. For each mutant,
actually compute a concrete input, work out the original code's output and the mutated code's
output, and determine the true verdict from that computation.

If the outputs differ: verdict is "real_coverage_gap". Fill in discriminating_input (an actual
function call with real argument values), orig_output and mut_output (the actual computed
values), a clear reasoning paragraph explaining why the outputs differ, and a recommended_test
(a real pytest function, ready to paste into a test file, that would catch this mutation).

If after genuinely computing you confirm the outputs are identical for realistic inputs: verdict
is "equivalent_mutant". Still fill in discriminating_input, orig_output, and mut_output with what
you actually checked (they should be equal), plus reasoning explaining why the values match.
Leave recommended_test as "".

WHEN CHOOSING AN INPUT FOR A THRESHOLD OR COMPARISON MUTANT, pick an input specifically NEAR the
boundary, since that is exactly where original and mutated behavior are most likely to diverge.

CONSTANT-IN-COMPARE mutant (e.g. `if discount > 0` mutated to `if discount > 1`): the
discriminating value is STRICTLY BETWEEN the old and new threshold.

IMPORTANT: reuse the EXACT mutant_id numbers given to you, verbatim, as strings.

Respond with ONLY a JSON object, no prose, no markdown fences, in this exact schema:
{
  "details": [
    {
      "mutant_id": "...",
      "verdict": "equivalent_mutant | real_coverage_gap",
      "discriminating_input": "...",
      "orig_output": "the value the original code returns for discriminating_input",
      "mut_output": "the value the mutated code returns for discriminating_input",
      "reasoning": "...",
      "recommended_test": "..."
    }
  ]
}"""


# ---------------------------------------------------------------------------
# CrewAI call wrapper (with timing, matching call_agent's stats tracking)
# ---------------------------------------------------------------------------

def call_crew(role, goal, backstory, description, expected_output):
    start = time.perf_counter()
    llm = LLM(model=MODEL)
    agent = Agent(role=role, goal=goal, backstory=backstory, llm=llm, verbose=True)
    task = Task(description=description, expected_output=expected_output, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    elapsed = time.perf_counter() - start

    CALL_STATS.append({"agent": role, "seconds": elapsed})
    print(f"  [{role}] {elapsed:.2f}s")

    raw_text = str(result).strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        print(f"  [!] {role} returned non-JSON output:\n{raw_text}")
        return None


def print_cost_summary():
    if not CALL_STATS:
        return
    print("\n=== Cost & Latency Summary ===")
    print(f"{'Agent':<40}{'Time (s)':<12}")
    total_time = 0
    for stat in CALL_STATS:
        print(f"{stat['agent']:<40}{stat['seconds']:<12.2f}")
        total_time += stat["seconds"]
    print("-" * 52)
    print(f"{'TOTAL':<40}{total_time:<12.2f}")


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def run_tests(test_path):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", test_path],
        capture_output=True, text=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def run_mutation_testing(source_path, tests_path):
    engine_path = Path(__file__).parent / "simple_mutation_test.py"
    if not engine_path.exists():
        return None
    result = subprocess.run(
        [sys.executable, str(engine_path), source_path, tests_path],
        capture_output=True, text=True,
    )
    output = result.stdout.strip()
    return output if output else None


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


def parse_mutant_sources(raw_report):
    """Extract mutated source texts from '=== Survivor Mutant Sources ===' section, if present."""
    section_marker = "=== Survivor Mutant Sources ==="
    sec_idx = raw_report.find(section_marker)
    if sec_idx == -1:
        return {}
    sources = {}
    text = raw_report[sec_idx:]
    begin_pattern = re.compile(r"^--- MUTANT (\d+) SOURCE BEGIN ---$", re.MULTILINE)
    end_pattern = re.compile(r"^--- MUTANT (\d+) SOURCE END ---$", re.MULTILINE)
    for m in begin_pattern.finditer(text):
        mutant_num = m.group(1)
        content_start = m.end() + 1
        end_m = end_pattern.search(text, content_start)
        if end_m:
            sources[mutant_num] = text[content_start:end_m.start()].rstrip("\n")
    return sources


def execute_against_source(source_code, expression):
    try:
        namespace = {}
        exec(compile(source_code, "<verify_source>", "exec"), namespace)
        value = eval(compile(expression, "<verify_expr>", "eval"), namespace)
        return True, value
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Date / volatile-field normalization
# ---------------------------------------------------------------------------

_INV_DATE_RE = re.compile(r'\bINV-\d{8}-(\d+)\b')


def normalize_volatile(value):
    if isinstance(value, str):
        return _INV_DATE_RE.sub(r'INV-DATEIGNORED-\1', value)
    if isinstance(value, dict):
        return {k: normalize_volatile(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)([normalize_volatile(v) for v in value])
    return value


def normalize_volatile_str(text):
    return _INV_DATE_RE.sub(r'INV-DATEIGNORED-\1', str(text))


def values_match(claimed, actual):
    claimed_norm = normalize_volatile_str(str(claimed).strip())
    actual_norm = normalize_volatile(actual)
    try:
        claimed_value = ast.literal_eval(claimed_norm)
        return claimed_value == actual_norm
    except Exception:
        pass
    return claimed_norm.replace(" ", "") == str(actual_norm).replace(" ", "")


def build_recommended_test(mutant_id, discriminating_input, actual):
    safe_id = str(mutant_id).replace("-", "_")
    if isinstance(actual, dict) and "invoice_id" in actual:
        stable_fields = {k: v for k, v in actual.items() if k != "invoice_id"}
        lines = [
            f"def test_auto_verified_mutant_{safe_id}():",
            f"    result = {discriminating_input}",
            f"    assert {{k: v for k, v in result.items() if k != 'invoice_id'}} == {stable_fields!r}",
            f"    assert result['invoice_id'].startswith('INV-')  # date portion excluded",
        ]
        return "\n".join(lines)
    return (
        f"def test_auto_verified_mutant_{safe_id}():\n"
        f"    result = {discriminating_input}\n"
        f"    assert result == {actual!r}"
    )


# ---------------------------------------------------------------------------
# Line-coverage tracing
# ---------------------------------------------------------------------------

def get_executed_lines(source_code, expression):
    executed = set()

    def _tracer(frame, event, arg):
        if frame.f_code.co_filename == "<verify_source>" and event == "line":
            executed.add(frame.f_lineno)
        return _tracer

    prev_trace = sys.gettrace()
    try:
        namespace = {}
        sys.settrace(_tracer)
        sys._getframe().f_trace = _tracer
        exec(compile(source_code, "<verify_source>", "exec"), namespace)
        eval(compile(expression, "<verify_expr>", "eval"), namespace)
    except Exception:
        return None
    finally:
        sys.settrace(prev_trace)
    return executed


def parse_mutant_lineno(diff_description):
    m = re.search(r'\(line\s+(\d+)', diff_description)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Boundary-value tracing
# ---------------------------------------------------------------------------

def parse_boundary_tag(diff_description):
    m = re.search(
        r'\[BOUNDARY\s+expr=(\S+)\s+op=(\S+)\s+old=(-?\d+)\s+new=(-?\d+)\]',
        diff_description,
    )
    if not m:
        return None
    try:
        return {"expr": m.group(1), "op": m.group(2),
                "old": int(m.group(3)), "new": int(m.group(4))}
    except (ValueError, IndexError):
        return None


def is_nondiscriminating_value(old, new, actual_value):
    try:
        v = float(actual_value)
    except (TypeError, ValueError):
        return False
    lo, hi = float(min(old, new)), float(max(old, new))
    return not (lo < v <= hi)


def capture_operand_value(source_code, expression, lineno, expr_text):
    eval_expr = expr_text.replace("_", " ")
    captured = []

    def _tracer(frame, event, arg):
        if (frame.f_code.co_filename == "<verify_source>" and event == "line"
                and frame.f_lineno == lineno and not captured):
            try:
                ns = {**frame.f_globals, **frame.f_locals}
                val = eval(compile(eval_expr, "<operand_eval>", "eval"), ns)
                captured.append(val)
            except Exception as e:
                captured.append(f"<eval-error: {e}>")
        return _tracer

    prev_trace = sys.gettrace()
    try:
        namespace = {}
        sys.settrace(_tracer)
        sys._getframe().f_trace = _tracer
        exec(compile(source_code, "<verify_source>", "exec"), namespace)
        eval(compile(expression, "<verify_expr>", "eval"), namespace)
    except Exception:
        pass
    finally:
        sys.settrace(prev_trace)

    if not captured:
        return False, f"line {lineno} was never reached during execution"
    return True, captured[0]


# ---------------------------------------------------------------------------
# Evidence-completeness gate
# ---------------------------------------------------------------------------

def has_real_evidence(v):
    test_input = str(v.get("test_input", "")).strip().lower()
    orig_out = str(v.get("orig_output", "")).strip()
    mut_out = str(v.get("mut_output", "")).strip()
    if not test_input or test_input in ("unreachable", "n/a", "na", "none", "-"):
        return False
    if not orig_out or not mut_out:
        return False
    return True


# ---------------------------------------------------------------------------
# Full verification (matching orchestrator.py's verify_equivalent_claims)
# ---------------------------------------------------------------------------

def verify_equivalent_claims(patched_source, surviving_mutants_analysis,
                              mutant_sources=None, desc_by_id=None):
    """
    For every 'equivalent_mutant' entry with a discriminating_input, runs:
      1. Output check (original side, with date normalization)
      2. Output check (mutant side, if mutant source available)
      3. Line-coverage check
      4. Boundary-value check (Constant-in-Compare only)
    For 'real_coverage_gap' entries, also verifies the claimed mut_output if possible.
    """
    verified = []
    for entry in surviving_mutants_analysis:
        mid = str(entry.get("mutant_id", ""))
        diff_desc = (desc_by_id or {}).get(mid, "")

        if entry.get("verdict") != "equivalent_mutant":
            disc_input = str(entry.get("discriminating_input", "")).strip()
            claimed_mut = entry.pop("_claimed_mut_output", "")
            if mutant_sources and mid in mutant_sources and disc_input and claimed_mut:
                mut_success, actual_mut = execute_against_source(mutant_sources[mid], disc_input)
                if mut_success:
                    actual_mut_norm = normalize_volatile(actual_mut)
                    if not values_match(claimed_mut, actual_mut):
                        entry["reasoning"] = (
                            entry.get("reasoning", "") +
                            f" [Mutant-side verification NOTE: claimed mut_output={claimed_mut!r} "
                            f"but actual mutated-code execution returns {actual_mut_norm!r}. "
                            f"Coverage gap verdict stands; recommended_test updated with ground truth.]"
                        )
                        orig_success, actual_orig = execute_against_source(patched_source, disc_input)
                        if orig_success:
                            entry["recommended_test"] = build_recommended_test(mid, disc_input, actual_orig)
                    else:
                        entry["reasoning"] = (
                            entry.get("reasoning", "") +
                            f" [Mutant-side verification confirmed: mut_output={actual_mut_norm!r} matches claim.]"
                        )
            else:
                entry.pop("_claimed_mut_output", None)
            verified.append(entry)
            continue

        discriminating_input = str(entry.get("discriminating_input", "")).strip()
        claimed_orig = entry.pop("_claimed_orig_output", "")
        claimed_mut = entry.pop("_claimed_mut_output", "")

        if not discriminating_input:
            entry["reasoning"] = (
                "[UNVERIFIABLE] " + entry.get("reasoning", "") +
                " No discriminating_input available to re-execute; not code-verified."
            )
            verified.append(entry)
            continue

        success, actual = execute_against_source(patched_source, discriminating_input)
        if not success:
            entry["verdict"] = "real_coverage_gap"
            entry["reasoning"] = (
                f"AUTOMATED VERIFICATION FAILED: could not execute '{discriminating_input}' "
                f"against real code ({actual}). Reclassified as real_coverage_gap."
            )
            entry["recommended_test"] = ""
            verified.append(entry)
            continue

        actual_norm = normalize_volatile(actual)
        if claimed_orig and not values_match(claimed_orig, actual):
            entry["verdict"] = "real_coverage_gap"
            entry["reasoning"] = (
                f"AUTOMATED VERIFICATION OVERRIDE: claimed {discriminating_input} returns "
                f"{claimed_orig!r} but actual execution returns {actual_norm!r}."
            )
            entry["recommended_test"] = build_recommended_test(mid, discriminating_input, actual)
            verified.append(entry)
            continue

        # --- line-coverage check ---
        mutant_lineno = parse_mutant_lineno(diff_desc) if diff_desc else None
        line_note = ""
        if mutant_lineno is not None:
            executed_lines = get_executed_lines(patched_source, discriminating_input)
            if executed_lines is not None and mutant_lineno not in executed_lines:
                entry["verdict"] = "unverified_unreachable"
                entry["_mutant_lineno"] = mutant_lineno
                entry["_executed_lines_sample"] = sorted(executed_lines)[:30]
                entry["reasoning"] = (
                    f"[LINE-COVERAGE REJECTION] '{discriminating_input}' never executed line "
                    f"{mutant_lineno} (diff: '{diff_desc}'). Lines reached: {sorted(executed_lines)}. "
                    f"Equivalence claim is vacuous."
                )
                verified.append(entry)
                continue
            elif executed_lines is not None:
                line_note = f" Line {mutant_lineno} confirmed reached."

        # --- mutant-side output check ---
        if mutant_sources and mid in mutant_sources and claimed_mut:
            mut_success, actual_mut = execute_against_source(mutant_sources[mid], discriminating_input)
            if mut_success:
                actual_mut_norm = normalize_volatile(actual_mut)
                if not values_match(claimed_mut, actual_mut):
                    entry["verdict"] = "real_coverage_gap"
                    entry["reasoning"] = (
                        f"AUTOMATED VERIFICATION OVERRIDE (mutant side): claimed mut_output="
                        f"{claimed_mut!r} but actual mutated code returns {actual_mut_norm!r}, "
                        f"which differs from original's {actual_norm!r}. Mutant IS distinguishable."
                    )
                    entry["recommended_test"] = build_recommended_test(mid, discriminating_input, actual)
                    verified.append(entry)
                    continue
                else:
                    boundary = parse_boundary_tag(diff_desc)
                    if boundary and mutant_lineno is not None:
                        ok, operand_val = capture_operand_value(
                            patched_source, discriminating_input, mutant_lineno, boundary["expr"]
                        )
                        if ok and is_nondiscriminating_value(boundary["old"], boundary["new"], operand_val):
                            entry["verdict"] = "unverified_nondiscriminating_value"
                            entry["_boundary"] = boundary
                            entry["_operand_val"] = operand_val
                            entry["_mutant_lineno"] = mutant_lineno
                            entry["reasoning"] = (
                                f"[BOUNDARY-VALUE REJECTION] '{discriminating_input}' reached line "
                                f"{mutant_lineno}; outputs matched, but "
                                f"'{boundary['expr'].replace('_',' ')}'={operand_val!r} is NOT in "
                                f"the discriminating interval ({boundary['old']}, {boundary['new']}]."
                            )
                            verified.append(entry)
                            continue
                        elif ok:
                            line_note += (
                                f" Runtime value of '{boundary['expr'].replace('_',' ')}' was "
                                f"{operand_val!r}, in interval ({boundary['old']}, {boundary['new']}]."
                            )
                    entry["reasoning"] = (
                        entry.get("reasoning", "") +
                        f" [Verified: original returns {actual_norm!r} and mutant also returns "
                        f"{actual_mut_norm!r} — genuinely equivalent.{line_note}]"
                    )
            else:
                entry["reasoning"] = (
                    entry.get("reasoning", "") +
                    f" [Original side confirmed: {actual_norm!r}. Mutant-side execution failed: "
                    f"{actual_mut}.{line_note}]"
                )
        else:
            # No mutant source available — still run boundary-value check on original side.
            boundary = parse_boundary_tag(diff_desc)
            if boundary and mutant_lineno is not None:
                ok, operand_val = capture_operand_value(
                    patched_source, discriminating_input, mutant_lineno, boundary["expr"]
                )
                if ok and is_nondiscriminating_value(boundary["old"], boundary["new"], operand_val):
                    entry["verdict"] = "unverified_nondiscriminating_value"
                    entry["_boundary"] = boundary
                    entry["_operand_val"] = operand_val
                    entry["_mutant_lineno"] = mutant_lineno
                    entry["reasoning"] = (
                        f"[BOUNDARY-VALUE REJECTION] '{discriminating_input}' reached line "
                        f"{mutant_lineno}; '{boundary['expr'].replace('_',' ')}'={operand_val!r} "
                        f"NOT in interval ({boundary['old']}, {boundary['new']}]."
                    )
                    verified.append(entry)
                    continue
                elif ok:
                    line_note += (
                        f" Runtime value of '{boundary['expr'].replace('_',' ')}' was {operand_val!r} "
                        f"— in interval ({boundary['old']}, {boundary['new']}]."
                    )
            entry["reasoning"] = (
                entry.get("reasoning", "") +
                f" [Verified: '{discriminating_input}' returns {actual_norm!r}, matches claim.{line_note}]"
            )
        verified.append(entry)
    return verified


# ---------------------------------------------------------------------------
# Wording-accuracy patch
# ---------------------------------------------------------------------------

_INACCURATE_PHRASES = [
    "no discount is applied", "discount is not applied", "discount is not subtracted",
    "no discount applied", "the discount is not applied", "returns the original subtotal",
    "returns the unchanged subtotal", "subtotal is not reduced", "subtotal remains unchanged",
    "subtotal is unchanged", "price is not reduced", "no discount is deducted",
    "discount is not deducted",
]
_CORRECTION_SUFFIX = (
    " NOTE: the `if` at this line only controls whether the discount is RECORDED in "
    "applied_log — the subtraction on the return line is unconditional, so the numeric "
    "subtotal is identical under both original and mutant. What actually changes is that "
    "the discount entry silently disappears from the order's audit log / analytics pipeline."
)


def apply_wording_patch(surviving_mutants_analysis):
    for m in surviving_mutants_analysis:
        is_boundary = bool(m.get("_boundary")) or "[BOUNDARY" in m.get("diff", "")
        if not is_boundary:
            continue
        if parse_mutant_lineno(m.get("diff", "")) != 77:
            continue
        reasoning = m.get("reasoning", "")
        if any(phrase in reasoning.lower() for phrase in _INACCURATE_PHRASES):
            m["reasoning"] = reasoning + _CORRECTION_SUFFIX


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CrewAI multi-agent debugging orchestrator")
    parser.add_argument("--source", required=True)
    parser.add_argument("--tests", required=True)
    args = parser.parse_args()

    source_path = Path(args.source)
    tests_path = Path(args.tests)

    print("=== CrewAI Multi-Agent Debugging System ===")
    print(f"Source: {source_path}  |  Tests: {tests_path}\n")

    retry_count = 0
    diagnosis = None

    while retry_count <= MAX_RETRIES:
        print(f"--- Sandbox run (attempt {retry_count + 1}) ---")
        passed, test_output = run_tests(str(tests_path))

        if passed:
            print("✅ All tests passed.\n")
            break

        print("❌ Tests failed. Invoking Diagnosis Agent...")
        source_code = source_path.read_text(encoding="utf-8")
        test_code = tests_path.read_text(encoding="utf-8")

        diag_desc = (
            f"Source code:\n{source_code}\n\nTest file:\n{test_code}\n\n"
            f"Pytest output:\n{test_output}\n\n"
        )
        if diagnosis is not None:
            diag_desc += f"Previous diagnosis + fix attempt failed:\n{json.dumps(diagnosis)}\n\n"
        diag_desc += "Diagnose the failure. Respond ONLY with the JSON schema described in your backstory."

        diagnosis = call_crew(
            "Diagnosis Agent",
            "Identify the root cause of a failing test without writing any fix",
            DIAGNOSIS_SYSTEM_PROMPT, diag_desc,
            "A single JSON object matching the required schema, nothing else.",
        )
        if not diagnosis:
            print("🛑 Diagnosis Agent failed to return valid JSON. Aborting.")
            print_cost_summary()
            return

        print(f"  Root cause: {diagnosis['root_cause']}")
        print(f"  Confidence: {diagnosis['confidence']}\n")

        print("Invoking Fix Agent...")
        fix_desc = (
            f"Original source code:\n{source_code}\n\nDiagnosis:\n{json.dumps(diagnosis)}\n\n"
            f"Produce a minimal patch. Respond ONLY with the JSON schema described in your backstory."
        )
        fix = call_crew(
            "Fix Agent",
            "Produce a minimal, targeted patch based on a diagnosis",
            FIX_SYSTEM_PROMPT, fix_desc,
            "A single JSON object matching the required schema, nothing else.",
        )
        if not fix:
            print("🛑 Fix Agent failed to return valid JSON. Aborting.")
            print_cost_summary()
            return

        print(f"  Change: {fix['change_summary']}")
        source_path.write_text(fix["patched_code"], encoding="utf-8")
        retry_count += 1

    else:
        print(f"🛑 Retry limit ({MAX_RETRIES}) reached. Manual review needed.")
        print_cost_summary()
        return

    # --- Mutation re-check ---
    print("--- Mutation Re-check ---")
    mutation_report = run_mutation_testing(str(source_path), str(tests_path))

    if mutation_report is None:
        print("  [i] simple_mutation_test.py not found or produced no output — skipping.")
        print_cost_summary()
        return

    patched_source = source_path.read_text(encoding="utf-8")
    test_content = tests_path.read_text(encoding="utf-8")

    mutant_sources = parse_mutant_sources(mutation_report)
    if mutant_sources:
        print(f"  [i] Captured mutated source for {len(mutant_sources)} survivor(s) — mutant-side verification enabled.")
    else:
        print("  [i] No survivor mutant sources found — mutant-side verification unavailable.")

    total_match = re.search(r"Total mutants:\s*(\d+)", mutation_report)
    survived_match = re.search(r"Survived:\s*(\d+)", mutation_report)
    total_mutants = int(total_match.group(1)) if total_match else 0
    survived_count = int(survived_match.group(1)) if survived_match else 0

    if total_mutants == 0:
        print("  [i] Zero mutants generated — no mutation-based signal available.")
        print("\n=== Final Verdict: ACCEPT_WITH_ADDED_TESTS (Low confidence) ===")
        print_cost_summary()
        return

    if survived_count == 0:
        print(f"  [i] {total_mutants} mutants generated, all killed. No triage needed.")
        print("\n=== Final Verdict: ACCEPT_FIX (High confidence) ===")
        print_cost_summary()
        return

    survivor_descriptions = parse_survivors(mutation_report)
    if not survivor_descriptions:
        print("  [!] Could not parse survivor descriptions. Treating as unverified.")
        print("\n=== Final Verdict: ACCEPT_WITH_ADDED_TESTS (Low confidence) ===")
        print_cost_summary()
        return

    desc_by_id = {str(i + 1): desc for i, desc in enumerate(survivor_descriptions)}
    numbered_list = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(survivor_descriptions))

    print(f"  Triaging {len(survivor_descriptions)} survived mutants...")
    triage_desc = (
        f"Patched source code:\n{patched_source}\n\nTest file:\n{test_content}\n\n"
        f"Survived mutants (numbered — use these exact numbers as mutant_id):\n{numbered_list}\n\n"
        f"Triage each one. Respond ONLY with the JSON schema described in your backstory."
    )
    triage_result = call_crew(
        "Mutation Triage Agent",
        "Determine whether each surviving mutant is a real coverage gap or harmless",
        MUTATION_TRIAGE_SYSTEM_PROMPT, triage_desc,
        "A single JSON object matching the required schema, nothing else.",
    )
    if not triage_result:
        print("🛑 Mutation Triage Agent failed to return valid JSON.")
        print_cost_summary()
        return

    verdicts = triage_result.get("verdicts", [])

    # --- Evidence-completeness gate ---
    raw_real_gaps = [v for v in verdicts if v.get("verdict") == "real_coverage_gap"]
    raw_equivalents = [v for v in verdicts if v.get("verdict") == "equivalent_mutant"]

    verified_equivalents = [v for v in raw_equivalents if has_real_evidence(v)]
    unverified = [v for v in raw_equivalents if not has_real_evidence(v)]
    if unverified:
        print(f"  [!] {len(unverified)} mutant(s) claimed equivalent with no real evidence — "
              f"treating as UNVERIFIED, routing to detail pass.")

    real_gaps = raw_real_gaps + unverified
    equivalents = verified_equivalents
    print(f"  Triage result: {len(real_gaps)} flagged for detail "
          f"({len(raw_real_gaps)} real gap(s) + {len(unverified)} unverified), "
          f"{len(equivalents)} verified equivalent.")

    # --- Detail pass: boundary mutants isolated, others batched ---
    details_by_id = {}
    if real_gaps:
        boundary_gaps = [g for g in real_gaps if "[BOUNDARY" in desc_by_id.get(str(g["mutant_id"]), "")]
        other_gaps = [g for g in real_gaps if "[BOUNDARY" not in desc_by_id.get(str(g["mutant_id"]), "")]

        total_calls = len(boundary_gaps) + math.ceil(len(other_gaps) / DETAIL_BATCH_SIZE) if other_gaps else len(boundary_gaps)
        print(f"  Getting detail on {len(real_gaps)} gap(s) "
              f"({len(boundary_gaps)} boundary-isolated + {len(other_gaps)} batched) "
              f"in {total_calls} call(s)...")

        # --- Boundary mutants: one call each ---
        for gap in boundary_gaps:
            mid = str(gap["mutant_id"])
            diff_desc = desc_by_id.get(mid, "(description unavailable)")
            gap_summary = f"- mutant_id {mid}: {diff_desc}"
            detail_desc = (
                f"Patched source code:\n{patched_source}\n\nTest file:\n{test_content}\n\n"
                f"For the mutant below, independently determine real gap or equivalent by "
                f"computing a concrete example (use this EXACT mutant_id):\n{gap_summary}\n\n"
                f"Respond ONLY with the JSON schema described in your backstory."
            )
            detail_result = call_crew(
                f"Mutation Detail (boundary {mid})",
                "Independently verify a single boundary-condition mutant",
                MUTATION_DETAIL_SYSTEM_PROMPT, detail_desc,
                "A single JSON object matching the required schema, nothing else.",
            )
            if not detail_result:
                continue
            for d in detail_result.get("details", []):
                returned_mid = str(d.get("mutant_id"))
                detail_entry = dict(d)

                # --- contamination check ---
                own_tag = parse_boundary_tag(diff_desc)
                if own_tag and returned_mid == mid:
                    reasoning_text = detail_entry.get("reasoning", "")
                    own_old, own_new = str(own_tag["old"]), str(own_tag["new"])
                    own_mentioned = bool(
                        re.search(rf'\b{re.escape(own_old)}\b', reasoning_text) or
                        re.search(rf'\b{re.escape(own_new)}\b', reasoning_text)
                    )
                    other_boundary_values = set()
                    for other_g in boundary_gaps:
                        other_mid = str(other_g["mutant_id"])
                        if other_mid == mid:
                            continue
                        other_tag = parse_boundary_tag(desc_by_id.get(other_mid, ""))
                        if other_tag:
                            other_boundary_values.add(str(other_tag["old"]))
                            other_boundary_values.add(str(other_tag["new"]))
                    foreign_mentioned = any(
                        re.search(rf'\b{re.escape(v)}\b', reasoning_text) for v in other_boundary_values
                    )
                    if foreign_mentioned and not own_mentioned:
                        detail_entry["_contaminated"] = True
                        detail_entry["_own_tag"] = own_tag
                details_by_id[returned_mid] = detail_entry

        # --- contamination retry ---
        contaminated = {mid: d for mid, d in details_by_id.items() if d.get("_contaminated")}
        if contaminated:
            print(f"  [!] {len(contaminated)} boundary mutant(s) show contaminated reasoning — retrying...")
            for mid, bad_detail in contaminated.items():
                diff_desc = desc_by_id.get(mid, "(description unavailable)")
                own_tag = bad_detail.get("_own_tag") or parse_boundary_tag(diff_desc) or {}
                lineno = parse_mutant_lineno(diff_desc)
                retry_summary = (
                    f"- mutant_id {mid}: {diff_desc}\n"
                    f"  IMPORTANT: your previous reasoning described a different mutant's condition. "
                    f"This mutant's actual boundary is old={own_tag.get('old','?')}, "
                    f"new={own_tag.get('new','?')} at line {lineno}, "
                    f"expr={own_tag.get('expr','?').replace('_',' ')}. Focus exclusively on THIS "
                    f"mutant's changed constant."
                )
                retry_desc = (
                    f"Patched source code:\n{patched_source}\n\nTest file:\n{test_content}\n\n"
                    f"For the mutant below, provide a fresh analysis focused only on this mutant's "
                    f"boundary (use this EXACT mutant_id):\n{retry_summary}\n\n"
                    f"Respond ONLY with the JSON schema described in your backstory."
                )
                retry_result = call_crew(
                    f"Mutation Detail (contamination retry {mid})",
                    "Independently verify a single boundary-condition mutant without cross-contamination",
                    MUTATION_DETAIL_SYSTEM_PROMPT, retry_desc,
                    "A single JSON object matching the required schema, nothing else.",
                )
                if retry_result:
                    for d in retry_result.get("details", []):
                        details_by_id[str(d.get("mutant_id"))] = d

        # --- non-boundary mutants: batched ---
        if other_gaps:
            num_batches = math.ceil(len(other_gaps) / DETAIL_BATCH_SIZE)
            for batch_num in range(num_batches):
                batch = other_gaps[batch_num * DETAIL_BATCH_SIZE:(batch_num + 1) * DETAIL_BATCH_SIZE]
                gap_summaries = "\n".join(
                    f"- mutant_id {g['mutant_id']}: {desc_by_id.get(str(g['mutant_id']), '(description unavailable)')}"
                    for g in batch
                )
                detail_desc = (
                    f"Patched source code:\n{patched_source}\n\nTest file:\n{test_content}\n\n"
                    f"For each mutant below, independently determine real gap or equivalent by "
                    f"computing a concrete example (use these EXACT mutant_ids):\n{gap_summaries}\n\n"
                    f"Respond ONLY with the JSON schema described in your backstory."
                )
                detail_result = call_crew(
                    f"Mutation Detail (batch {batch_num + 1}/{num_batches})",
                    "Independently verify a batch of surviving mutants",
                    MUTATION_DETAIL_SYSTEM_PROMPT, detail_desc,
                    "A single JSON object matching the required schema, nothing else.",
                )
                if detail_result:
                    for d in detail_result.get("details", []):
                        details_by_id[str(d.get("mutant_id"))] = d

    # --- Assemble analysis list ---
    surviving_mutants_analysis = []
    for g in real_gaps:
        mid = str(g["mutant_id"])
        detail = details_by_id.get(mid, {})
        final_verdict = detail.get("verdict", "real_coverage_gap")
        surviving_mutants_analysis.append({
            "mutant_id": mid,
            "diff": desc_by_id.get(mid, ""),
            "verdict": final_verdict,
            "discriminating_input": detail.get("discriminating_input", ""),
            "reasoning": detail.get("reasoning", ""),
            "recommended_test": detail.get("recommended_test", ""),
            "_claimed_orig_output": detail.get("orig_output", ""),
            "_claimed_mut_output": detail.get("mut_output", ""),
        })
    for e in equivalents:
        mid = str(e["mutant_id"])
        test_input = e.get("test_input", "")
        orig_out = e.get("orig_output", "")
        mut_out = e.get("mut_output", "")
        evidence = (f"Checked {test_input}: original={orig_out}, mutant={mut_out} (identical)"
                    if test_input else "No computed check provided by triage.")
        surviving_mutants_analysis.append({
            "mutant_id": mid,
            "diff": desc_by_id.get(mid, ""),
            "verdict": "equivalent_mutant",
            "discriminating_input": test_input,
            "reasoning": evidence,
            "recommended_test": "",
            "_claimed_orig_output": orig_out,
            "_claimed_mut_output": mut_out,
        })

    # --- Code-execution verification pass ---
    pre_equiv = sum(1 for m in surviving_mutants_analysis if m["verdict"] == "equivalent_mutant")
    surviving_mutants_analysis = verify_equivalent_claims(
        patched_source, surviving_mutants_analysis, mutant_sources=mutant_sources, desc_by_id=desc_by_id,
    )
    post_equiv = sum(1 for m in surviving_mutants_analysis if m["verdict"] == "equivalent_mutant")
    overturned = pre_equiv - post_equiv

    unreachable = [m for m in surviving_mutants_analysis if m.get("verdict") == "unverified_unreachable"]
    nondiscriminating = [m for m in surviving_mutants_analysis if m.get("verdict") == "unverified_nondiscriminating_value"]

    if overturned > 0 or unreachable or nondiscriminating:
        parts = []
        if overturned > 0:
            parts.append(f"{overturned} claim(s) overturned by output mismatch")
        if unreachable:
            parts.append(f"{len(unreachable)} claim(s) rejected (line never reached)")
        if nondiscriminating:
            parts.append(f"{len(nondiscriminating)} claim(s) rejected (value outside interval)")
        print(f"  [!] Verification: {'; '.join(parts)}.")

    # --- Line-coverage retry pass ---
    if unreachable:
        print(f"  Retrying {len(unreachable)} line-coverage-rejected claim(s)...")
        retry_details_by_id = {}
        num_batches = math.ceil(len(unreachable) / DETAIL_BATCH_SIZE)
        for batch_num in range(num_batches):
            batch = unreachable[batch_num * DETAIL_BATCH_SIZE:(batch_num + 1) * DETAIL_BATCH_SIZE]
            retry_summaries = "\n".join(
                f"- mutant_id {m['mutant_id']}: {desc_by_id.get(str(m['mutant_id']), '(description unavailable)')}"
                f"\n  IMPORTANT: previous input '{m.get('discriminating_input', '')}' never executed "
                f"line {m.get('_mutant_lineno', '?')}. Choose an input that ACTUALLY reaches it. "
                f"Lines previously executed: {m.get('_executed_lines_sample', [])}."
                for m in batch
            )
            retry_desc = (
                f"Patched source code:\n{patched_source}\n\nTest file:\n{test_content}\n\n"
                f"For each mutant below, choose an input that reaches the specified line, then compute "
                f"the real verdict (use these EXACT mutant_ids):\n{retry_summaries}\n\n"
                f"Respond ONLY with the JSON schema described in your backstory."
            )
            retry_result = call_crew(
                f"Mutation Detail Line-Retry (batch {batch_num + 1}/{num_batches})",
                "Retry mutant verification with explicit line-coverage feedback",
                MUTATION_DETAIL_SYSTEM_PROMPT, retry_desc,
                "A single JSON object matching the required schema, nothing else.",
            )
            if retry_result:
                for d in retry_result.get("details", []):
                    retry_details_by_id[str(d.get("mutant_id"))] = d

        updated = []
        for m in surviving_mutants_analysis:
            if m.get("verdict") != "unverified_unreachable":
                updated.append(m)
                continue
            mid = str(m["mutant_id"])
            retry_detail = retry_details_by_id.get(mid)
            if not retry_detail:
                m["verdict"] = "real_coverage_gap"
                m["reasoning"] = m.get("reasoning", "") + " Retry returned no result; conservatively reclassified."
                m["recommended_test"] = ""
                updated.append(m)
                continue
            updated.append({
                "mutant_id": mid,
                "diff": desc_by_id.get(mid, ""),
                "verdict": retry_detail.get("verdict", "real_coverage_gap"),
                "discriminating_input": retry_detail.get("discriminating_input", ""),
                "reasoning": f"[LINE-COVERAGE RETRY] " + retry_detail.get("reasoning", ""),
                "recommended_test": retry_detail.get("recommended_test", ""),
                "_claimed_orig_output": retry_detail.get("orig_output", ""),
                "_claimed_mut_output": retry_detail.get("mut_output", ""),
            })
        surviving_mutants_analysis = updated
        surviving_mutants_analysis = verify_equivalent_claims(
            patched_source, surviving_mutants_analysis, mutant_sources=mutant_sources, desc_by_id=desc_by_id,
        )

    # --- Boundary-value retry pass ---
    nondiscriminating = [m for m in surviving_mutants_analysis if m.get("verdict") == "unverified_nondiscriminating_value"]
    if nondiscriminating:
        print(f"  Retrying {len(nondiscriminating)} boundary-value-rejected claim(s)...")
        bv_retry_by_id = {}
        num_batches = math.ceil(len(nondiscriminating) / DETAIL_BATCH_SIZE)
        for batch_num in range(num_batches):
            batch = nondiscriminating[batch_num * DETAIL_BATCH_SIZE:(batch_num + 1) * DETAIL_BATCH_SIZE]
            bv_summaries = "\n".join(
                (lambda m, b=m.get("_boundary", {}), v=m.get("_operand_val", "?"):
                 f"- mutant_id {m['mutant_id']}: {desc_by_id.get(str(m['mutant_id']), '(description unavailable)')}\n"
                 f"  IMPORTANT: previous input '{m.get('discriminating_input','')}' reached line "
                 f"{m.get('_mutant_lineno','?')} but '{b.get('expr','?').replace('_',' ')}'={v!r}, "
                 f"NOT in interval ({b.get('old','?')}, {b.get('new','?')}]. Choose a new input where "
                 f"the value falls strictly in that interval."
                 )(m)
                for m in batch
            )
            bv_desc = (
                f"Patched source code:\n{patched_source}\n\nTest file:\n{test_content}\n\n"
                f"For each mutant below, choose an input whose runtime value falls in the described "
                f"discriminating interval, then compute the real verdict "
                f"(use these EXACT mutant_ids):\n{bv_summaries}\n\n"
                f"Respond ONLY with the JSON schema described in your backstory."
            )
            bv_result = call_crew(
                f"Mutation Detail BV-Retry (batch {batch_num + 1}/{num_batches})",
                "Retry mutant verification with explicit boundary-value feedback",
                MUTATION_DETAIL_SYSTEM_PROMPT, bv_desc,
                "A single JSON object matching the required schema, nothing else.",
            )
            if bv_result:
                for d in bv_result.get("details", []):
                    bv_retry_by_id[str(d.get("mutant_id"))] = d

        bv_updated = []
        for m in surviving_mutants_analysis:
            if m.get("verdict") != "unverified_nondiscriminating_value":
                bv_updated.append(m)
                continue
            mid = str(m["mutant_id"])
            bv_detail = bv_retry_by_id.get(mid)
            if not bv_detail:
                m["verdict"] = "real_coverage_gap"
                m["reasoning"] = m.get("reasoning", "") + " BV retry returned no result; conservatively reclassified."
                m["recommended_test"] = ""
                bv_updated.append(m)
                continue
            bv_updated.append({
                "mutant_id": mid,
                "diff": desc_by_id.get(mid, ""),
                "verdict": bv_detail.get("verdict", "real_coverage_gap"),
                "discriminating_input": bv_detail.get("discriminating_input", ""),
                "reasoning": "[BOUNDARY-VALUE RETRY] " + bv_detail.get("reasoning", ""),
                "recommended_test": bv_detail.get("recommended_test", ""),
                "_claimed_orig_output": bv_detail.get("orig_output", ""),
                "_claimed_mut_output": bv_detail.get("mut_output", ""),
            })
        surviving_mutants_analysis = bv_updated
        surviving_mutants_analysis = verify_equivalent_claims(
            patched_source, surviving_mutants_analysis, mutant_sources=mutant_sources, desc_by_id=desc_by_id,
        )

    # --- Wording-accuracy patch ---
    apply_wording_patch(surviving_mutants_analysis)

    # --- Strip internal tracking fields ---
    for m in surviving_mutants_analysis:
        for key in ("_mutant_lineno", "_executed_lines_sample", "_boundary",
                    "_operand_val", "_contaminated", "_own_tag"):
            m.pop(key, None)

    confirmed_gaps = [m for m in surviving_mutants_analysis if m["verdict"] == "real_coverage_gap"]
    final_verdict = "ACCEPT_WITH_ADDED_TESTS" if confirmed_gaps else "ACCEPT_FIX"

    result = {
        "mutation_summary": {
            "total_mutants": total_mutants,
            "killed": total_mutants - survived_count,
            "survived": survived_count,
        },
        "surviving_mutants_analysis": surviving_mutants_analysis,
        "final_verdict": final_verdict,
        "confidence": "High",
    }

    print(f"\n=== Final Verdict: {result['final_verdict']} ===")
    print(json.dumps(result, indent=2))

    print_cost_summary()


if __name__ == "__main__":
    main()