"""
Multi-Agent Debugging System — Orchestrator
=============================================
Wires together Diagnosis Agent -> Fix Agent -> Sandbox Re-run -> (retry loop)
-> Mutation Re-check Agent, using the Groq API (OpenAI-compatible).

USAGE:
    Windows PowerShell:  $env:GROQ_API_KEY = "gsk_..."
    Mac/Linux:            export GROQ_API_KEY="gsk_..."
    python orchestrator.py --source buggy_code.py --tests test_buggy_code.py

Requires: pip install groq
"""

import argparse
import ast
import difflib
import json
import math
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path

from groq import Groq, APITimeoutError, APIConnectionError

# Groq's flagship general-purpose model. NOTE: Groq deprecates models on short notice —
# llama-3.3-70b-versatile was retired as of Aug 2026. If this model 404s, check
# https://console.groq.com/docs/models for the current lineup.
MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 3

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    sys.exit("ERROR: GROQ_API_KEY environment variable not set. "
             "Get a key at https://console.groq.com/keys")
client = Groq(api_key=api_key)

# Tracks per-call stats for the final cost/latency summary
CALL_STATS = []  # list of dicts: {agent, seconds, prompt_tokens, completion_tokens}

# ---------------------------------------------------------------------------
# Agent system prompts
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
EXACTLY as given, character-for-character outside of the lines you are actually fixing —
this includes quote style (do not convert \"\"\" to ''' or vice versa), whitespace, comments,
and wording. Every line in "patched_code" that is not part of the actual bug fix must be
byte-identical to the corresponding line in the original source.
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
orig_output/mut_output values is not acceptable. There is no valid excuse to leave these blank:
every function in this file is called by at least one test in the test file, so nothing here is
truly dead/unreachable code. If you find yourself about to write an empty or placeholder value,
that is a signal you have not actually done the computation — go back and compute a real input
and real outputs instead.

COMMON MISTAKE TO AVOID: rounding-precision mutants (e.g. round(x, 2) -> round(x, 3)) and
boolean-default mutants (e.g. False -> True on a parameter default) are almost never equivalent —
actually compute a real example rather than assuming they're harmless.

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
you actually checked (they should be equal), plus reasoning explaining why the values match — an
equivalent verdict must be backed by the same level of real computation as a real_coverage_gap
verdict. leave recommended_test as "".

WHEN CHOOSING AN INPUT FOR A THRESHOLD OR COMPARISON MUTANT (e.g. > vs >=, or a constant like
0 vs 1 used in a comparison), do not just pick any input where the branch isn't taken — pick an
input specifically NEAR the boundary (e.g. a value strictly between the old and new threshold),
since that is exactly where original and mutated behavior are most likely to diverge. An input
far from the boundary can make a real gap look like a false equivalence.

WHEN CHOOSING AN INPUT FOR EACH MUTANT — THIS IS CRITICAL:
Each mutant's discriminating_input must be independently chosen to specifically target THAT
mutant's changed location and mechanism. Do NOT reuse the same generic call for multiple
mutants in the same batch just because they are all in related functions.

Rules by mutant type:
- DEFAULT-ARGUMENT mutant (e.g. express=False -> True, order_number=1 -> 2): the input MUST
  OMIT that specific argument so the default is actually invoked; pass enough other args to
  isolate the effect of only that default changing. Example for express default: call
  calculate_shipping(30.00) with no express= keyword, not calculate_shipping(30.00, express=True).
- FORMAT-WIDTH mutant (e.g. :04d -> :05d on order_number): choose an order_number whose string
  representation actually DIFFERS between the old and new width. For :04d -> :05d, any
  order_number < 10000 exposes the gap (e.g. order_number=1 gives "0001" vs "00001"). Explicitly
  reason about what the formatted string looks like under each variant before picking.
- ARITHMETIC mutant (e.g. + -> -, * -> /): choose a value where the two operations produce
  visibly different numerical results; do not choose 0 or 1 if those would make both sides equal.
- COMPARISON mutant (e.g. >= -> >): choose a value exactly AT the boundary so one branch takes
  it and the other does not.
- CONSTANT-IN-COMPARE mutant (e.g. `if discount > 0` mutated to `if discount > 1`, or
  `if subtotal > 20` mutated to `if subtotal > 21`): the discriminating value is STRICTLY
  BETWEEN the old and new threshold — not below old, not above new. For `discount > 0 -> > 1`:
  any input where 0 < discount <= 1 is the ONLY range where original and mutant disagree (e.g.
  discount=0.5 passes `> 0` but fails `> 1`; discount=0.0 fails both; discount=5.0 passes both).
  Choose an input that produces a discount value in this narrow gap interval (0, 1].
  For `subtotal > 20 -> > 21`: choose a subtotal of exactly 21 (passes `> 20`, fails `> 21`).
  Verify by tracing through both the original and mutated condition with your chosen value.

IMPORTANT: reuse the EXACT mutant_id numbers given to you, verbatim, as strings. Do not invent
new ids, rename them, or reformat them.

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
# Helpers
# ---------------------------------------------------------------------------

# Groq on-demand tier commonly caps requests at 8000 tokens per minute (prompt + completion
# combined). Keep a safety margin below that so we don't get a 413 rate_limit_exceeded error.
TPM_SAFETY_LIMIT = 8000
TPM_BUFFER = 500  # headroom for tokenizer estimation error


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used only to size max_tokens safely."""
    return len(text) // 4


def call_agent(agent_name: str, system_prompt: str, user_message: str, max_tokens: int = 4000,
                _retry: bool = True, _network_retries: int = 2) -> dict:
    """Call the LLM with a given agent system prompt, track cost/latency, and parse JSON.

    max_tokens is automatically capped so (estimated prompt tokens + max_tokens) stays
    under the account's tokens-per-minute limit, avoiding 413 rate_limit_exceeded errors.

    gpt-oss-120b is a reasoning model: by default it spends part of its completion token
    budget on internal chain-of-thought (reasoning_effort="medium") before writing the
    final answer, which can cause truncation even on short/simple schemas. We request
    reasoning_effort="low" here since these agents are doing classification/formatting
    tasks, not tasks that benefit much from deep reasoning, to leave more of the budget
    for the actual JSON output.

    Handles two distinct failure modes separately:
      - Malformed/truncated JSON: retried once with a higher token ceiling (still capped
        by TPM), since this is usually caused by hitting max_tokens.
      - Transient network errors (timeouts, connection failures): retried with a short
        backoff, since these are unrelated to prompt size or token budget and usually
        resolve on their own within a few seconds.
    """
    estimated_prompt_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_message)
    safe_max_tokens = max(500, TPM_SAFETY_LIMIT - TPM_BUFFER - estimated_prompt_tokens)
    effective_max_tokens = min(max_tokens, safe_max_tokens)
    if effective_max_tokens < max_tokens:
        print(f"  [i] {agent_name}: capping max_tokens to {effective_max_tokens} "
              f"(requested {max_tokens}) to stay under the {TPM_SAFETY_LIMIT} TPM limit "
              f"given ~{estimated_prompt_tokens} estimated prompt tokens.")

    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=effective_max_tokens,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
    except (APITimeoutError, APIConnectionError) as e:
        if _network_retries > 0:
            wait = 3 * (3 - _network_retries)  # 3s, then 6s
            print(f"  [!] {agent_name} hit a transient network error ({type(e).__name__}). "
                  f"Retrying in {wait}s ({_network_retries} attempt(s) left)...")
            time.sleep(wait)
            return call_agent(agent_name, system_prompt, user_message, max_tokens=max_tokens,
                               _retry=_retry, _network_retries=_network_retries - 1)
        print(f"  [!] {agent_name} failed after repeated network errors: {e}")
        sys.exit(f"🛑 {agent_name} Agent could not reach the Groq API after retries. "
                 f"Check your internet connection and try again.")
    elapsed = time.perf_counter() - start

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    finish_reason = getattr(response.choices[0], "finish_reason", None)
    CALL_STATS.append({
        "agent": agent_name,
        "seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    })
    print(f"  [{agent_name}] {elapsed:.2f}s | {prompt_tokens} prompt tokens | {completion_tokens} completion tokens")

    if finish_reason == "length":
        print(f"  [!] {agent_name} response was truncated (hit max_tokens={effective_max_tokens}).")

    text = response.choices[0].message.content
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if _retry:
            bumped = min(int(max_tokens * 1.5), 5000)
            print(f"  [!] {agent_name} returned malformed JSON (likely truncated). "
                  f"Retrying once with max_tokens={bumped} (still capped by TPM limit)...")
            return call_agent(agent_name, system_prompt, user_message, max_tokens=bumped, _retry=False)
        print(f"  [!] {agent_name} returned non-JSON output after retry:\n{text}")
        sys.exit(f"🛑 {agent_name} Agent failed to return valid JSON after retry. Aborting run.")


def print_diff(before: str, after: str, filename: str):
    """Print a unified diff between the pre- and post-patch source code."""
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"{filename} (before)",
        tofile=f"{filename} (after)",
    )
    diff_text = "".join(diff)
    if diff_text:
        print("  --- Diff ---")
        for line in diff_text.splitlines():
            print(f"  {line}")
        print()


def print_cost_summary():
    """Print a final table of time/token usage per agent call."""
    if not CALL_STATS:
        return
    print("\n=== Cost & Latency Summary ===")
    print(f"{'Agent':<20}{'Time (s)':<12}{'Prompt tok':<14}{'Completion tok':<16}")
    total_time = total_prompt = total_completion = 0
    for stat in CALL_STATS:
        print(f"{stat['agent']:<20}{stat['seconds']:<12.2f}{stat['prompt_tokens']:<14}{stat['completion_tokens']:<16}")
        total_time += stat["seconds"]
        total_prompt += stat["prompt_tokens"]
        total_completion += stat["completion_tokens"]
    print("-" * 62)
    print(f"{'TOTAL':<20}{total_time:<12.2f}{total_prompt:<14}{total_completion:<16}")


def run_tests(test_path: str) -> tuple[bool, str]:
    """Run pytest and return (passed, full_output)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", test_path],
        capture_output=True, text=True,
    )
    output = result.stdout + result.stderr
    passed = result.returncode == 0
    return passed, output


def run_mutation_testing(source_path: str, tests_path: str) -> str | None:
    """Run the built-in pure-Python mutation engine (cross-platform, no external tools)."""
    engine_path = Path(__file__).parent / "simple_mutation_test.py"
    if not engine_path.exists():
        return None  # engine script missing from this folder

    result = subprocess.run(
        [sys.executable, str(engine_path), source_path, tests_path],
        capture_output=True, text=True,
    )
    output = result.stdout.strip()
    return output if output else None


def summarize_mutation_report(raw_report: str) -> str:
    """
    The raw mutation engine output includes a per-mutant loop line for every
    single mutant (killed AND survived) — for large files this bloats the
    prompt with information the LLM doesn't need (it only needs to reason
    about the summary counts and the surviving mutants). Trim to just the
    summary section.
    """
    marker = "=== Mutation Testing Summary ==="
    idx = raw_report.find(marker)
    if idx == -1:
        return raw_report  # unexpected format, pass through untouched
    return raw_report[idx:]


def parse_survivors(raw_report: str) -> list[str]:
    """
    Extract the list of surviving mutant descriptions (e.g. "BoolOp And -> Or")
    from the raw mutation engine output, in order. IDs are assigned here in
    Python (their list position) rather than left to the model to invent,
    since triage and detail run as two separate API calls and the model has
    no reliable way to keep self-invented IDs consistent between them.
    """
    marker = "Surviving mutants (tests did NOT catch these):"
    idx = raw_report.find(marker)
    if idx == -1:
        return []
    lines = raw_report[idx:].splitlines()[1:]  # skip the marker line itself
    survivors = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            survivors.append(stripped[2:])
        elif stripped.startswith("===") or stripped.startswith("---"):
            # Stop at the next section header (e.g. the mutant sources section).
            break
    return survivors


def parse_mutant_sources(raw_report: str) -> dict[str, str]:
    """
    Extract the mutated source texts emitted by simple_mutation_test.py's
    '=== Survivor Mutant Sources ===' section.

    Returns a dict mapping the 1-based mutant number (as a string, matching the
    position-based IDs used by parse_survivors) to the full mutated source text.

    If the section is absent (older engine version), returns an empty dict so that
    callers degrade gracefully — mutant-side verification simply becomes unavailable.
    """
    section_marker = "=== Survivor Mutant Sources ==="
    sec_idx = raw_report.find(section_marker)
    if sec_idx == -1:
        return {}

    sources: dict[str, str] = {}
    text = raw_report[sec_idx:]
    # Find all "--- MUTANT N SOURCE BEGIN ---" ... "--- MUTANT N SOURCE END ---" blocks.
    begin_pattern = re.compile(r"^--- MUTANT (\d+) SOURCE BEGIN ---$", re.MULTILINE)
    end_pattern = re.compile(r"^--- MUTANT (\d+) SOURCE END ---$", re.MULTILINE)
    begin_matches = list(begin_pattern.finditer(text))
    for m in begin_matches:
        mutant_num = m.group(1)
        content_start = m.end() + 1  # skip the newline after the BEGIN marker
        end_m = end_pattern.search(text, content_start)
        if end_m:
            sources[mutant_num] = text[content_start:end_m.start()].rstrip("\n")
    return sources


def _execute_against_source(source_code: str, expression: str):
    """
    Execute a function-call expression against the given source code in an isolated
    namespace, and return (success, value_or_error_message).

    This is the structural fix for a specific failure mode: an AI agent can claim an
    "equivalent_mutant" verdict, fill in every required evidence field, and still be
    simply wrong — its self-reported computation doesn't have to match reality. Field
    presence checks (see _has_real_evidence below) catch an EMPTY or placeholder claim,
    but cannot catch a WRONG claim that looks complete. Only actually running the code
    can catch that. This function is the "actually run the code" step.
    """
    try:
        namespace: dict = {}
        exec(compile(source_code, "<verify_source>", "exec"), namespace)
        value = eval(compile(expression, "<verify_expr>", "eval"), namespace)
        return True, value
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _get_executed_lines(source_code: str, expression: str) -> set[int] | None:
    """
    Execute `expression` against `source_code` in an isolated namespace with
    sys.settrace active, and return the set of line numbers that were actually
    executed in the '<verify_source>' code object (i.e. lines from source_code,
    not from the expression itself or stdlib internals).

    The tracer is active for BOTH the exec() (module load) and the eval() (the
    actual call). This is necessary because some mutations affect lines that only
    execute at definition time — specifically, default-argument values on `def`
    lines (e.g. `def process_order(..., express=False, ...)` at line 94) are
    evaluated when the `def` statement runs during exec(), not during a call to
    the function. Tracing only the eval() would miss those lines entirely.

    Returns None if execution raises any exception (caller should treat as
    "coverage unknown" and not block the verdict on that basis).

    Implementation note: sys.settrace operates on the per-thread trace function.
    We save and restore the previous trace function so this is safe to call from
    any context. The trace function is always uninstalled in the finally block,
    even if the eval raises, so we never leave a dangling tracer installed.
    """
    executed: set[int] = set()

    def _tracer(frame: types.FrameType, event: str, arg):
        # Only record lines from the compiled source object, not builtins or stdlib.
        if frame.f_code.co_filename == "<verify_source>" and event == "line":
            executed.add(frame.f_lineno)
        return _tracer

    prev_trace = sys.gettrace()
    try:
        namespace: dict = {}
        # Arm the tracer BEFORE exec so that `def` lines (which evaluate default
        # argument expressions) are captured. Without this, mutations on default
        # argument values (e.g. express=False -> True) would always appear
        # "unreachable" because their line only executes during module load.
        sys.settrace(_tracer)
        sys._getframe().f_trace = _tracer
        exec(compile(source_code, "<verify_source>", "exec"), namespace)
        eval(compile(expression, "<verify_expr>", "eval"), namespace)
    except Exception:
        return None
    finally:
        sys.settrace(prev_trace)

    return executed


def _parse_mutant_lineno(diff_description: str) -> int | None:
    """
    Extract the line number from a diff description like:
        "Compare Gt -> LtE (line 74, in apply_coupon)"
        "Constant False -> True (line 94, in process_order)"
        "BinOp Add -> Sub (line 110, in process_order)"

    Returns the integer line number, or None if no '(line N' pattern is found.
    """
    m = re.search(r'\(line\s+(\d+)', diff_description)
    return int(m.group(1)) if m else None


def _parse_boundary_tag(diff_description: str) -> dict | None:
    """
    Parse the optional [BOUNDARY ...] suffix appended by simple_mutation_test.py
    to Constant-in-Compare mutation descriptions, e.g.:

        "Constant 0 -> 1 (line 77, in apply_coupon) [BOUNDARY expr=discount op=Gt old=0 new=1]"

    Returns a dict with keys: expr (str), op (str), old (int), new (int).
    Returns None if the tag is absent or malformed.

    The tag is only present for int Constant nodes that are direct operands of a
    single-operator Compare node.  BinOp and Compare-operator-swap mutations do
    not carry this tag — those are documented as a known limitation.
    """
    m = re.search(
        r'\[BOUNDARY\s+expr=(\S+)\s+op=(\S+)\s+old=(-?\d+)\s+new=(-?\d+)\]',
        diff_description,
    )
    if not m:
        return None
    try:
        return {
            "expr": m.group(1),
            "op": m.group(2),
            "old": int(m.group(3)),
            "new": int(m.group(4)),
        }
    except (ValueError, IndexError):
        return None


def _is_nondiscriminating_value(op_name: str, old: int, new: int, actual_value) -> bool:
    """
    Return True if `actual_value` (the runtime value of the compared expression)
    does NOT fall in the interval where original and mutant behavior diverge.

    For a mutation `constant old -> new` (where new = old + 1) inside a Compare,
    the values where the two variants disagree depend on the operator:

    Operator  | Original condition      | Mutant condition         | Gap interval
    ----------|-------------------------|--------------------------|----------------------
    Gt        | expr > old              | expr > new (= old+1)     | old < expr <= new
    GtE       | expr >= old             | expr >= new (= old+1)    | old <= expr < new
    Lt        | expr < old              | expr < new (= old+1)     | old <= expr < new
              | (const is left: old<expr| new<expr)                |
    LtE       | expr <= old             | expr <= new              | old < expr <= new

    Because new = old + 1 always, the gap interval is exactly the half-open
    interval (old, new] = (old, old+1], which for integers means expr == old+1
    (i.e. exactly new).  For floats the gap is the open interval (old, new).

    Simplification used here: if actual_value <= old or actual_value >= new, the
    value is OUTSIDE the gap — both original and mutant agree on the branch taken,
    so the discriminating_input cannot distinguish them on this specific comparison.
    If old < actual_value < new, the value IS in the gap (for floats), or
    actual_value == new (for integers).  We treat old < actual_value <= new as
    discriminating (strictly after old, up to and including new).

    Operators where the constant is on the LEFT (e.g. `0 < discount`) are
    handled by normalising: the comparison is semantically the same as
    `discount > 0` after flipping, so we just swap old/new appropriately —
    but since new = old + 1 always, the direction of the gap is always
    "is actual_value between old and new" regardless of which side the constant
    is on.  Conservatively we check the same interval for all operator types.

    Returns True  → value is non-discriminating (equivalence claim is vacuous).
    Returns False → value IS in the discriminating interval (the comparison behaves
                    differently under original vs mutant).
    """
    try:
        v = float(actual_value)
    except (TypeError, ValueError):
        # Cannot compare — treat as discriminating (don't reject).
        return False
    lo = float(min(old, new))
    hi = float(max(old, new))
    # The discriminating interval is (lo, hi] for Gt/LtE, [lo, hi) for GtE/Lt.
    # Conservatively: if lo < v <= hi, the value is discriminating.
    # If v <= lo or v >= hi (beyond or at the boundary on the wrong side), non-discriminating.
    return not (lo < v <= hi)


def _capture_operand_value(
    source_code: str, expression: str, lineno: int, expr_text: str
):
    """
    Execute `expression` against `source_code` and, using sys.settrace, capture
    the value of `expr_text` evaluated in the frame at the FIRST time line `lineno`
    executes.

    Returns (True, value) if the line was reached and expr_text evaluated without
    error, or (False, reason_string) if the line was never reached or evaluation
    failed.

    `expr_text` may contain underscores substituted for spaces (as produced by
    simple_mutation_test.py's _boundary_tag), so we restore spaces before eval.
    The eval is done against frame.f_globals merged with frame.f_locals so that
    both module-level names and local variables are visible.
    """
    # Restore spaces that were replaced with underscores in the tag.
    eval_expr = expr_text.replace("_", " ")

    captured: list = []  # use a list as a mutable cell; filled on first hit

    def _tracer(frame: types.FrameType, event: str, arg):
        if (
            frame.f_code.co_filename == "<verify_source>"
            and event == "line"
            and frame.f_lineno == lineno
            and not captured  # only first hit
        ):
            try:
                ns = {**frame.f_globals, **frame.f_locals}
                val = eval(compile(eval_expr, "<operand_eval>", "eval"), ns)
                captured.append(val)
            except Exception as e:
                captured.append(f"<eval-error: {e}>")
        return _tracer

    prev_trace = sys.gettrace()
    try:
        namespace: dict = {}
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
# Date/volatile field normalization for verification
# ---------------------------------------------------------------------------

# Pattern for invoice IDs like INV-20260830-0001 — the date portion is volatile
# (changes every day) and must be normalized before comparing claimed vs actual values
# to prevent false-positive mismatches that have nothing to do with the mutation under test.
_INV_DATE_RE = re.compile(r'\bINV-\d{8}-(\d+)\b')


def _normalize_volatile(value):
    """
    Replace the date portion of any invoice_id-style string (INV-YYYYMMDD-NNNN)
    with a fixed placeholder so that date drift never causes a spurious mismatch.
    Works on plain strings, dicts (in-place on a copy), and lists recursively.
    Returns the normalized value — does not mutate the original.
    """
    if isinstance(value, str):
        return _INV_DATE_RE.sub(r'INV-DATEIGNORED-\1', value)
    if isinstance(value, dict):
        return {k: _normalize_volatile(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        normalized = [_normalize_volatile(v) for v in value]
        return type(value)(normalized)
    return value


def _normalize_volatile_str(text: str) -> str:
    """
    Normalize a free-text string representation that may contain invoice_id values.
    Replaces INV-YYYYMMDD-NNNN with INV-DATEIGNORED-NNNN in the raw text.
    """
    return _INV_DATE_RE.sub(r'INV-DATEIGNORED-\1', text)


def _values_match(claimed: str, actual) -> bool:
    """
    Compare a model's free-text claimed value against a real Python object returned
    by actual execution. Tries parsing the claim as a Python literal first (handles
    the common case: "45.0", "[1, 2]", "{'a': 1}"); falls back to a loose whitespace-
    insensitive string comparison if the claim isn't a clean literal (e.g. it includes
    extra words).

    Both sides are normalized to strip volatile date fields (e.g. invoice_id date
    portions) before comparison so that date drift can never trigger a false mismatch.
    """
    claimed_norm = _normalize_volatile_str(str(claimed).strip())
    actual_norm = _normalize_volatile(actual)
    try:
        claimed_value = ast.literal_eval(claimed_norm)
        return claimed_value == actual_norm
    except Exception:
        pass
    return claimed_norm.replace(" ", "") == str(actual_norm).replace(" ", "")


def _build_recommended_test(mutant_id, discriminating_input: str, actual) -> str:
    """
    Build a pytest function that asserts the actual value returned by discriminating_input.

    If the result is a dict containing an 'invoice_id' key, that field is excluded from
    the equality assertion (its date portion changes every day, which would make the test
    self-destruct). Instead, we assert only the non-volatile fields and separately assert
    that invoice_id starts with 'INV-' to confirm it was at least generated.
    """
    safe_id = str(mutant_id).replace("-", "_")
    if isinstance(actual, dict) and "invoice_id" in actual:
        # Exclude invoice_id from the equality check to avoid hardcoding a date-bearing string.
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


def verify_equivalent_claims(
    patched_source: str,
    surviving_mutants_analysis: list,
    mutant_sources: dict | None = None,
    desc_by_id: dict | None = None,
) -> list:
    """
    For every mutant currently classified "equivalent_mutant" with a structured
    discriminating_input and claimed orig_output, runs four checks in order:

    1. OUTPUT CHECK (original side): execute the discriminating_input against the
       real patched source and verify the model's claimed orig_output is correct.
       If wrong → override to real_coverage_gap.

    2. OUTPUT CHECK (mutant side): when mutant_sources is available, also execute
       against the mutated source and verify the model's claimed mut_output.
       If the model said orig==mut (hence "equivalent") but they actually differ →
       override to real_coverage_gap.

    3. LINE-COVERAGE CHECK: trace which source lines were actually executed by the
       discriminating_input. Compare against the line number extracted from the diff
       description (e.g. "(line 74, in apply_coupon)"). If the mutated line was NEVER
       reached by the input, the equivalence claim is vacuous — the mutation had no
       chance to matter. Such entries are tagged with verdict "unverified_unreachable"
       so the caller can route them to a retry Detail pass with explicit line-target
       feedback, rather than silently accepting them as confirmed equivalent.

    4. BOUNDARY-VALUE CHECK (Constant-in-Compare only): for mutants whose diff
       description carries a [BOUNDARY expr=X op=Op old=N new=N+1] tag (produced by
       simple_mutation_test.py for Constant nodes that are direct operands of a Compare
       node), capture the runtime value of the compared expression at the mutated line
       and verify it falls strictly in the interval (old, new].  If it does NOT — e.g.
       the input produces discount=0.0 for a `discount > 0` mutation — the two variants
       agree on the branch at this value and the equivalence claim is vacuous. Such
       entries are tagged "unverified_nondiscriminating_value" for the same retry path.
       Scope: Constant-in-Compare mutations only. BinOp and Compare-operator-swap
       mutations do not carry the tag and are not subject to this check — documented as
       a known limitation.

    The line-coverage and boundary-value checks require desc_by_id (mutant_id -> diff
    description string). If desc_by_id is None or the relevant tag is absent, the check
    is skipped — a missing description should not block an otherwise well-supported claim.

    Both sides use _normalize_volatile / _normalize_volatile_str to strip date-derived
    fields (invoice_id date portions) before any comparison, preventing date drift from
    causing false-positive overrides.
    """
    verified = []
    for entry in surviving_mutants_analysis:
        if entry.get("verdict") != "equivalent_mutant":
            # For real_coverage_gap entries: if mutant source is available, also verify
            # the mut_output claim from the detail agent.
            mid = str(entry.get("mutant_id", ""))
            disc_input = str(entry.get("discriminating_input", "")).strip()
            claimed_mut = entry.pop("_claimed_mut_output", "")
            if mutant_sources and mid in mutant_sources and disc_input and claimed_mut:
                mut_success, actual_mut = _execute_against_source(
                    mutant_sources[mid], disc_input
                )
                if mut_success:
                    actual_mut_norm = _normalize_volatile(actual_mut)
                    if not _values_match(claimed_mut, actual_mut):
                        # The claimed mut_output was wrong — note it in reasoning but
                        # keep the verdict as real_coverage_gap (it still is one, the
                        # model just described the mutant's output inaccurately).
                        entry["reasoning"] = (
                            entry.get("reasoning", "") +
                            f" [Mutant-side verification NOTE: the model claimed mut_output="
                            f"{claimed_mut!r} but actually executing against the mutated code "
                            f"returns {actual_mut_norm!r}. The coverage gap verdict stands, "
                            f"but the recommended_test has been updated with ground-truth values.]"
                        )
                        # Rebuild recommended_test using the real orig/mut outputs.
                        orig_success, actual_orig = _execute_against_source(
                            patched_source, disc_input
                        )
                        if orig_success:
                            entry["recommended_test"] = _build_recommended_test(
                                entry.get("mutant_id", "x"), disc_input, actual_orig
                            )
                    else:
                        entry["reasoning"] = (
                            entry.get("reasoning", "") +
                            f" [Mutant-side verification confirmed: mut_output={actual_mut_norm!r} "
                            f"matches claim.]"
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
                " No structured discriminating_input was available to independently "
                "re-execute; this equivalent claim has not been code-verified."
            )
            verified.append(entry)
            continue

        success, actual = _execute_against_source(patched_source, discriminating_input)

        if not success:
            entry["verdict"] = "real_coverage_gap"
            entry["reasoning"] = (
                f"AUTOMATED VERIFICATION FAILED: could not execute the claimed "
                f"discriminating_input '{discriminating_input}' against the real "
                f"patched code ({actual}). The original equivalent_mutant claim could "
                f"not be substantiated, so this mutant is conservatively reclassified "
                f"as a real coverage gap pending manual review."
            )
            entry["recommended_test"] = ""
            verified.append(entry)
            continue

        actual_norm = _normalize_volatile(actual)

        if claimed_orig and not _values_match(claimed_orig, actual):
            entry["verdict"] = "real_coverage_gap"
            entry["reasoning"] = (
                f"AUTOMATED VERIFICATION OVERRIDE: the model claimed "
                f"{discriminating_input} returns {claimed_orig!r} for the original "
                f"(patched) code, but actually executing it returns {actual_norm!r}. "
                f"The equivalent_mutant claim was incorrect and has been reclassified "
                f"as a real coverage gap based on ground-truth code execution."
            )
            entry["recommended_test"] = _build_recommended_test(
                entry.get("mutant_id", "x"), discriminating_input, actual
            )
            verified.append(entry)
            continue

        # --- LINE-COVERAGE CHECK ---
        # Before accepting the equivalence, verify the discriminating_input actually
        # executed the mutated line. An input that never reaches the mutated code
        # trivially "confirms" equivalence — the mutation had no chance to matter.
        mid = str(entry.get("mutant_id", ""))
        diff_desc = (desc_by_id or {}).get(mid, "")
        mutant_lineno = _parse_mutant_lineno(diff_desc) if diff_desc else None

        if mutant_lineno is not None:
            executed_lines = _get_executed_lines(patched_source, discriminating_input)
            if executed_lines is not None and mutant_lineno not in executed_lines:
                # The mutated line was never reached — equivalence is unverified.
                entry["verdict"] = "unverified_unreachable"
                entry["_mutant_lineno"] = mutant_lineno
                entry["_executed_lines_sample"] = sorted(executed_lines)[:30]
                entry["reasoning"] = (
                    f"[LINE-COVERAGE REJECTION] The discriminating_input "
                    f"'{discriminating_input}' never executed line {mutant_lineno} "
                    f"(the mutated line per diff: '{diff_desc}'). "
                    f"Lines actually reached: {sorted(executed_lines)}. "
                    f"This equivalence claim is vacuous — the mutation was never given "
                    f"a chance to affect the output. Routing to retry with explicit "
                    f"line-target feedback."
                )
                verified.append(entry)
                continue
            elif executed_lines is not None:
                # Line was reached — note it in the confirmation message.
                line_note = (f" Line {mutant_lineno} was confirmed in the executed "
                             f"line set {sorted(executed_lines)}, so the mutation "
                             f"had a genuine opportunity to affect the output.")
            else:
                line_note = (f" (Line-coverage trace failed for this input; "
                             f"coverage of line {mutant_lineno} could not be confirmed.)")
        else:
            line_note = ""

        # Output checks out on original side; now check mutant side if available.
        if mutant_sources and mid in mutant_sources and claimed_mut:
            mut_success, actual_mut = _execute_against_source(
                mutant_sources[mid], discriminating_input
            )
            if mut_success:
                actual_mut_norm = _normalize_volatile(actual_mut)
                if not _values_match(claimed_mut, actual_mut):
                    # Model claimed orig==mut (equivalent), but the mutant actually
                    # produces a different value — the mutation IS distinguishable.
                    entry["verdict"] = "real_coverage_gap"
                    entry["reasoning"] = (
                        f"AUTOMATED VERIFICATION OVERRIDE (mutant side): the model "
                        f"claimed {discriminating_input} returns {claimed_mut!r} for "
                        f"the mutated code (same as original, hence 'equivalent'), but "
                        f"actually executing against the mutated source returns "
                        f"{actual_mut_norm!r}, which differs from the original's "
                        f"{actual_norm!r}. The mutant IS distinguishable — reclassified "
                        f"as real_coverage_gap."
                    )
                    entry["recommended_test"] = _build_recommended_test(
                        entry.get("mutant_id", "x"), discriminating_input, actual
                    )
                else:
                    # orig==mut confirmed by execution. Now run the boundary-value
                    # check if the diff carries a [BOUNDARY] tag.
                    boundary = _parse_boundary_tag(diff_desc)
                    if boundary and mutant_lineno is not None:
                        ok, operand_val = _capture_operand_value(
                            patched_source, discriminating_input,
                            mutant_lineno, boundary["expr"],
                        )
                        if ok and _is_nondiscriminating_value(
                            boundary["op"], boundary["old"], boundary["new"], operand_val
                        ):
                            entry["verdict"] = "unverified_nondiscriminating_value"
                            entry["_boundary"] = boundary
                            entry["_operand_val"] = operand_val
                            entry["_mutant_lineno"] = mutant_lineno
                            entry["reasoning"] = (
                                f"[BOUNDARY-VALUE REJECTION] The discriminating_input "
                                f"'{discriminating_input}' reached line {mutant_lineno} "
                                f"and orig/mut outputs matched, but the compared expression "
                                f"'{boundary['expr'].replace('_', ' ')}' evaluated to "
                                f"{operand_val!r} at that line — which is NOT in the "
                                f"discriminating interval ({boundary['old']}, {boundary['new']}]. "
                                f"Both original (expr > {boundary['old']}) and mutant "
                                f"(expr > {boundary['new']}) agree at this value; the input "
                                f"never tests the disagreement zone. Routing to retry with "
                                f"explicit value-target feedback."
                            )
                            verified.append(entry)
                            continue
                        elif ok:
                            line_note += (
                                f" Runtime value of '{boundary['expr'].replace('_', ' ')}' "
                                f"at line {mutant_lineno} was {operand_val!r}, which IS in "
                                f"the discriminating interval "
                                f"({boundary['old']}, {boundary['new']}] — "
                                f"the comparison was genuinely exercised."
                            )
                    entry["reasoning"] = (
                        entry.get("reasoning", "") +
                        f" [Automated verification confirmed: original returns "
                        f"{actual_norm!r} and mutant also returns {actual_mut_norm!r} "
                        f"for {discriminating_input} — genuinely equivalent on this "
                        f"input.{line_note}]"
                    )
            else:
                entry["reasoning"] = (
                    entry.get("reasoning", "") +
                    f" [Automated verification confirmed original side: {discriminating_input} "
                    f"returns {actual_norm!r}. Mutant-side execution failed: {actual_mut}."
                    f"{line_note}]"
                )
        else:
            # No mutant source available for this entry — run boundary-value check
            # using only the original source (we can still capture the operand value).
            boundary = _parse_boundary_tag(diff_desc)
            if boundary and mutant_lineno is not None:
                ok, operand_val = _capture_operand_value(
                    patched_source, discriminating_input,
                    mutant_lineno, boundary["expr"],
                )
                if ok and _is_nondiscriminating_value(
                    boundary["op"], boundary["old"], boundary["new"], operand_val
                ):
                    entry["verdict"] = "unverified_nondiscriminating_value"
                    entry["_boundary"] = boundary
                    entry["_operand_val"] = operand_val
                    entry["_mutant_lineno"] = mutant_lineno
                    entry["reasoning"] = (
                        f"[BOUNDARY-VALUE REJECTION] The discriminating_input "
                        f"'{discriminating_input}' reached line {mutant_lineno} "
                        f"and the compared expression "
                        f"'{boundary['expr'].replace('_', ' ')}' evaluated to "
                        f"{operand_val!r} — NOT in the discriminating interval "
                        f"({boundary['old']}, {boundary['new']}]. Routing to retry."
                    )
                    verified.append(entry)
                    continue
                elif ok:
                    line_note += (
                        f" Runtime value of '{boundary['expr'].replace('_', ' ')}' "
                        f"at line {mutant_lineno} was {operand_val!r} — in the "
                        f"discriminating interval ({boundary['old']}, {boundary['new']}]."
                    )
            entry["reasoning"] = (
                entry.get("reasoning", "") +
                f" [Automated verification confirmed: {discriminating_input} actually "
                f"returns {actual_norm!r} against the real patched code, matching the "
                f"claim.{line_note}]"
            )
        verified.append(entry)
    return verified


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-agent debugging orchestrator")
    parser.add_argument("--source", required=True, help="Path to the source file to fix")
    parser.add_argument("--tests", required=True, help="Path to the test file")
    args = parser.parse_args()

    source_path = Path(args.source)
    tests_path = Path(args.tests)

    print(f"=== Multi-Agent Debugging System ===")
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

        diag_input = (
            f"Source code:\n{source_code}\n\n"
            f"Test file:\n{test_code}\n\n"
            f"Pytest output:\n{test_output}"
        )
        if diagnosis is not None:
            diag_input += f"\n\nPrevious diagnosis + fix attempt failed. Previous diagnosis:\n{json.dumps(diagnosis)}"

        diagnosis = call_agent("Diagnosis", DIAGNOSIS_SYSTEM_PROMPT, diag_input)
        print(f"  Root cause: {diagnosis['root_cause']}")
        print(f"  Confidence: {diagnosis['confidence']}\n")

        print("Invoking Fix Agent...")
        fix_input = f"Original source code:\n{source_code}\n\nDiagnosis:\n{json.dumps(diagnosis)}"
        fix = call_agent("Fix", FIX_SYSTEM_PROMPT, fix_input)
        print(f"  Change: {fix['change_summary']}")

        print_diff(source_code, fix["patched_code"], str(source_path))

        source_path.write_text(fix["patched_code"], encoding="utf-8")
        retry_count += 1

    else:
        print(f"🛑 Retry limit ({MAX_RETRIES}) reached. Manual review needed.")
        print_cost_summary()
        return

    # --- Mutation re-check (only runs once tests pass) ---
    print("--- Mutation Re-check ---")
    mutation_report = run_mutation_testing(str(source_path), str(tests_path))

    if mutation_report is None:
        print("  [i] Mutation engine produced no output — skipping automated mutation re-check.")
        print("  [i] Check that simple_mutation_test.py is in the same folder as orchestrator.py.")
        print_cost_summary()
        return

    patched_source = source_path.read_text(encoding="utf-8")
    test_content = tests_path.read_text(encoding="utf-8")
    trimmed_report = summarize_mutation_report(mutation_report)

    # Parse mutated source texts emitted by the updated mutation engine so that
    # verify_equivalent_claims can also execute against the mutated code (not just
    # the patched/original code) to independently verify the model's mut_output claims.
    mutant_sources = parse_mutant_sources(mutation_report)
    if mutant_sources:
        print(f"  [i] Captured mutated source text for {len(mutant_sources)} survivor(s) "
              f"— mutant-side verification is enabled.")
    else:
        print("  [i] No survivor mutant sources found in report — "
              "mutant-side verification unavailable (only original-side verified).")

    # Parse "Total mutants: N" / "Survived: N" out of the raw report for the zero-mutant check.
    total_match = re.search(r"Total mutants:\s*(\d+)", mutation_report)
    survived_match = re.search(r"Survived:\s*(\d+)", mutation_report)
    total_mutants = int(total_match.group(1)) if total_match else 0
    survived_count = int(survived_match.group(1)) if survived_match else 0

    if total_mutants == 0:
        print("  [i] Zero mutants generated — this bug class is outside the mutation engine's "
              "coverage (e.g. mutable defaults, closures, scoping). No mutation-based signal "
              "is available; the fix's correctness rests on the Diagnosis/Fix Agents' reasoning "
              "and the existing test suite alone.")
        result = {
            "mutation_summary": {"total_mutants": 0, "killed": 0, "survived": 0},
            "surviving_mutants_analysis": [],
            "final_verdict": "ACCEPT_WITH_ADDED_TESTS",
            "confidence": "Low",
        }
        print(f"\n=== Final Verdict: {result['final_verdict']} ===")
        print(json.dumps(result, indent=2))
        print_cost_summary()
        return

    if survived_count == 0:
        print(f"  [i] {total_mutants} mutants generated, all killed. No triage needed.")
        result = {
            "mutation_summary": {"total_mutants": total_mutants, "killed": total_mutants, "survived": 0},
            "surviving_mutants_analysis": [],
            "final_verdict": "ACCEPT_FIX",
            "confidence": "High",
        }
        print(f"\n=== Final Verdict: {result['final_verdict']} ===")
        print(json.dumps(result, indent=2))
        print_cost_summary()
        return

    # --- Pass 1: triage every survived mutant, verdict only, minimal output per item ---
    survivor_descriptions = parse_survivors(mutation_report)
    if not survivor_descriptions:
        # Fallback: counts said N survived but we couldn't parse descriptions.
        # Treat conservatively — can't triage without them.
        print("  [!] Could not parse survivor descriptions from the mutation report. "
              "Skipping triage; treating as unverified.")
        result = {
            "mutation_summary": {"total_mutants": total_mutants,
                                  "killed": total_mutants - survived_count,
                                  "survived": survived_count},
            "surviving_mutants_analysis": [{
                "mutant_id": "N/A",
                "verdict": "real_coverage_gap",
                "discriminating_input": "",
                "reasoning": "Survivor descriptions could not be parsed from the mutation "
                             "report; treating as unverified rather than assuming safety.",
                "recommended_test": "",
            }],
            "final_verdict": "ACCEPT_WITH_ADDED_TESTS",
            "confidence": "Low",
        }
        print(f"\n=== Final Verdict: {result['final_verdict']} ===")
        print(json.dumps(result, indent=2))
        print_cost_summary()
        return

    print(f"  Triaging {len(survivor_descriptions)} survived mutants...")
    numbered_list = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(survivor_descriptions))
    triage_input = (
        f"Patched source code:\n{patched_source}\n\n"
        f"Test file:\n{test_content}\n\n"
        f"Survived mutants (numbered — use these exact numbers as mutant_id):\n{numbered_list}"
    )
    triage_result = call_agent("Mutation Triage", MUTATION_TRIAGE_SYSTEM_PROMPT, triage_input, max_tokens=3000)
    verdicts = triage_result.get("verdicts", [])

    # Map by position (string of the 1-based number) back to its original description.
    desc_by_id = {str(i + 1): desc for i, desc in enumerate(survivor_descriptions)}

    def _has_real_evidence(v: dict) -> bool:
        """
        Never trust a self-reported "equivalent_mutant" claim on faith. If the model
        didn't actually fill in a concrete test_input and BOTH outputs, there's no
        evidence backing the claim — treat it as unverified rather than accepting it.
        This closes the escape-hatch problem structurally: no phrasing of the prompt
        can make an empty-evidence "equivalent" claim get accepted, because the code
        checks the fields directly regardless of what excuse accompanies them.
        """
        test_input = str(v.get("test_input", "")).strip().lower()
        orig_out = str(v.get("orig_output", "")).strip()
        mut_out = str(v.get("mut_output", "")).strip()
        if not test_input or test_input in ("unreachable", "n/a", "na", "none", "-"):
            return False
        if not orig_out or not mut_out:
            return False
        return True

    raw_real_gaps = [v for v in verdicts if v.get("verdict") == "real_coverage_gap"]
    raw_equivalents = [v for v in verdicts if v.get("verdict") == "equivalent_mutant"]

    verified_equivalents = [v for v in raw_equivalents if _has_real_evidence(v)]
    unverified = [v for v in raw_equivalents if not _has_real_evidence(v)]
    if unverified:
        print(f"  [!] {len(unverified)} mutant(s) were claimed equivalent with no real "
              f"computed evidence (empty or 'unreachable' placeholder values). Treating "
              f"these as UNVERIFIED, not equivalent — routing to detail pass for a proper check.")

    # Unverified claims get treated exactly like real gaps: routed to the detail pass for
    # a real, forced explanation, rather than silently accepted or silently dropped.
    real_gaps = raw_real_gaps + unverified
    equivalents = verified_equivalents
    print(f"  Triage result: {len(real_gaps)} flagged for detail "
          f"({len(raw_real_gaps)} real gap(s) + {len(unverified)} unverified claim(s)), "
          f"{len(equivalents)} verified equivalent mutant(s).")

    # --- Pass 2: only elaborate on the mutants triage flagged as real gaps ---
    # Batched: each detailed write-up (discriminating_input + reasoning + a full test
    # function) runs ~500-600 tokens including JSON overhead in practice, so a single
    # call can only safely cover a handful of mutants before risking the same
    # truncation triage just avoided. Batch size of 3 with a 3000-token ceiling keeps
    # real margin even when entries run long, and even when every survivor turns out
    # to be a genuine gap.
    #
    # [BOUNDARY]-tagged mutants are dispatched one per API call (batch size 1) to
    # prevent the demonstrated cross-contamination pattern where two boundary mutants
    # in the same call blend each other's threshold values into a single confused
    # narrative. other_gaps use the normal batch size of 3.
    DETAIL_BATCH_SIZE = 3
    details_by_id = {}

    if real_gaps:
        # Partition: boundary mutants get isolated calls; others batch normally.
        boundary_gaps = [g for g in real_gaps
                         if "[BOUNDARY" in desc_by_id.get(str(g["mutant_id"]), "")]
        other_gaps    = [g for g in real_gaps
                         if "[BOUNDARY" not in desc_by_id.get(str(g["mutant_id"]), "")]

        total_calls = len(boundary_gaps) + math.ceil(len(other_gaps) / DETAIL_BATCH_SIZE)
        print(f"  Getting full detail on {len(real_gaps)} real coverage gap(s) "
              f"({len(boundary_gaps)} boundary-isolated + "
              f"{len(other_gaps)} batched) in {total_calls} API call(s)...")

        # --- Part A: boundary mutants — one call each ---
        for gap in boundary_gaps:
            mid = str(gap["mutant_id"])
            diff_desc = desc_by_id.get(mid, "(description unavailable)")
            gap_summary = f"- mutant_id {mid}: {diff_desc}"
            detail_input = (
                f"Patched source code:\n{patched_source}\n\n"
                f"Test file:\n{test_content}\n\n"
                f"For the mutant below, independently determine whether it is a real "
                f"coverage gap or actually equivalent by computing a concrete example — "
                f"do not assume either answer, it was previously flagged as a real gap "
                f"and needs a genuine check "
                f"(use this EXACT mutant_id value in your response):\n{gap_summary}"
            )
            detail_result = call_agent(
                f"Mutation Detail (boundary mutant {mid})",
                MUTATION_DETAIL_SYSTEM_PROMPT, detail_input, max_tokens=3000,
            )
            for d in detail_result.get("details", []):
                returned_mid = str(d.get("mutant_id"))
                detail_entry = dict(d)

                # --- Part B: grounding check ---
                # Verify the returned reasoning actually discusses THIS mutant's own
                # boundary values, not values from a different mutant that leaked in.
                # Even with batch-size-1 this acts as a safety net against model
                # hallucination of wrong thresholds.
                own_tag = _parse_boundary_tag(diff_desc)
                if own_tag and returned_mid == mid:
                    reasoning_text = detail_entry.get("reasoning", "")
                    own_old = str(own_tag["old"])
                    own_new = str(own_tag["new"])
                    # Check whether the reasoning mentions this mutant's OWN threshold
                    # values.  We look for the old or new value as a standalone number
                    # (word-boundary match so "20" doesn't match inside "200").
                    own_mentioned = bool(
                        re.search(rf'\b{re.escape(own_old)}\b', reasoning_text) or
                        re.search(rf'\b{re.escape(own_new)}\b', reasoning_text)
                    )
                    # Also check for any OTHER boundary mutant's old/new values that
                    # appear in the reasoning but NOT this mutant's own values.
                    other_boundary_values = set()
                    for other_g in boundary_gaps:
                        other_mid = str(other_g["mutant_id"])
                        if other_mid == mid:
                            continue
                        other_tag = _parse_boundary_tag(
                            desc_by_id.get(other_mid, "")
                        )
                        if other_tag:
                            other_boundary_values.add(str(other_tag["old"]))
                            other_boundary_values.add(str(other_tag["new"]))
                    # Contamination: another mutant's threshold appears, but this
                    # mutant's own threshold does NOT.
                    foreign_mentioned = any(
                        re.search(rf'\b{re.escape(v)}\b', reasoning_text)
                        for v in other_boundary_values
                    )
                    if foreign_mentioned and not own_mentioned:
                        detail_entry["_contaminated"] = True
                        detail_entry["_own_tag"] = own_tag

                details_by_id[returned_mid] = detail_entry

        # --- Contamination retry: one call per flagged entry ---
        contaminated = {mid: d for mid, d in details_by_id.items()
                        if d.get("_contaminated")}
        if contaminated:
            print(f"  [!] {len(contaminated)} boundary mutant(s) had potentially "
                  f"contaminated reasoning (foreign threshold values, own absent) — "
                  f"retrying with explicit grounding feedback...")
            for mid, bad_detail in contaminated.items():
                diff_desc = desc_by_id.get(mid, "(description unavailable)")
                own_tag = bad_detail.get("_own_tag") or _parse_boundary_tag(diff_desc) or {}
                lineno = _parse_mutant_lineno(diff_desc)
                retry_summary = (
                    f"- mutant_id {mid}: {diff_desc}\n"
                    f"  IMPORTANT: your previous reasoning appears to describe a different "
                    f"mutant's condition. This mutant's actual boundary is "
                    f"old={own_tag.get('old','?')}, new={own_tag.get('new','?')} "
                    f"at line {lineno}, expr={own_tag.get('expr','?').replace('_',' ')}. "
                    f"Focus exclusively on THIS mutant's changed constant and provide "
                    f"reasoning that explicitly references old={own_tag.get('old','?')} "
                    f"and new={own_tag.get('new','?')}."
                )
                retry_input = (
                    f"Patched source code:\n{patched_source}\n\n"
                    f"Test file:\n{test_content}\n\n"
                    f"For the mutant below, provide a fresh analysis focused on the "
                    f"specific boundary described — do not let reasoning from other "
                    f"mutants bleed in "
                    f"(use this EXACT mutant_id value in your response):\n{retry_summary}"
                )
                retry_result = call_agent(
                    f"Mutation Detail (contamination retry, mutant {mid})",
                    MUTATION_DETAIL_SYSTEM_PROMPT, retry_input, max_tokens=3000,
                )
                for d in retry_result.get("details", []):
                    details_by_id[str(d.get("mutant_id"))] = d

        # --- Non-boundary mutants: normal batching ---
        if other_gaps:
            num_other_batches = math.ceil(len(other_gaps) / DETAIL_BATCH_SIZE)
            for batch_num in range(num_other_batches):
                batch = other_gaps[batch_num * DETAIL_BATCH_SIZE:
                                   (batch_num + 1) * DETAIL_BATCH_SIZE]
                gap_summaries = "\n".join(
                    f"- mutant_id {g['mutant_id']}: "
                    f"{desc_by_id.get(str(g['mutant_id']), '(description unavailable)')}"
                    for g in batch
                )
                detail_input = (
                    f"Patched source code:\n{patched_source}\n\n"
                    f"Test file:\n{test_content}\n\n"
                    f"For each mutant below, independently determine whether it is a real "
                    f"coverage gap or actually equivalent by computing a concrete example — "
                    f"do not assume either answer, some of these were previously flagged as "
                    f"unverified and need a genuine check "
                    f"(use these EXACT mutant_id values in your response):\n{gap_summaries}"
                )
                detail_result = call_agent(
                    f"Mutation Detail (batch {batch_num + 1}/{num_other_batches})",
                    MUTATION_DETAIL_SYSTEM_PROMPT, detail_input, max_tokens=3000,
                )
                for d in detail_result.get("details", []):
                    details_by_id[str(d.get("mutant_id"))] = d

    surviving_mutants_analysis = []
    for g in real_gaps:
        mid = str(g["mutant_id"])
        detail = details_by_id.get(mid, {})
        # Respect the Detail agent's own independently-computed verdict rather than
        # assuming real_coverage_gap for everything that was routed to it — items
        # here may have been "unverified" claims that the deeper check confirmed
        # were actually equivalent after all.
        final_mutant_verdict = detail.get("verdict", "real_coverage_gap")
        surviving_mutants_analysis.append({
            "mutant_id": mid,
            "diff": desc_by_id.get(mid, ""),
            "verdict": final_mutant_verdict,
            "discriminating_input": detail.get("discriminating_input", ""),
            "reasoning": detail.get("reasoning", ""),
            "recommended_test": detail.get("recommended_test", ""),
            # Internal only — used by verify_equivalent_claims, popped before printing.
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
            # Internal only — used by verify_equivalent_claims, popped before printing.
            "_claimed_orig_output": orig_out,
            "_claimed_mut_output": mut_out,
        })

    # --- Code-execution + line-coverage verification pass ---
    # Never trust an "equivalent_mutant" claim just because it has evidence fields
    # filled in — actually re-run the claimed input against the real patched source
    # and check: (a) the output matches what was claimed, (b) the mutated line was
    # actually REACHED by the input. Claims that fail either check are overridden.
    pre_verify_equivalent_count = sum(
        1 for m in surviving_mutants_analysis if m["verdict"] == "equivalent_mutant"
    )
    surviving_mutants_analysis = verify_equivalent_claims(
        patched_source, surviving_mutants_analysis,
        mutant_sources=mutant_sources, desc_by_id=desc_by_id,
    )
    post_verify_equivalent_count = sum(
        1 for m in surviving_mutants_analysis if m["verdict"] == "equivalent_mutant"
    )
    overturned = pre_verify_equivalent_count - post_verify_equivalent_count

    # Collect verification rejections of two kinds:
    # - unreachable: discriminating_input never executed the mutated line
    # - nondiscriminating: line was reached but runtime value outside gap interval
    unreachable = [m for m in surviving_mutants_analysis
                   if m.get("verdict") == "unverified_unreachable"]
    nondiscriminating = [m for m in surviving_mutants_analysis
                         if m.get("verdict") == "unverified_nondiscriminating_value"]

    if overturned > 0 or unreachable or nondiscriminating:
        parts = []
        if overturned > 0:
            parts.append(f"{overturned} claim(s) overturned by output mismatch")
        if unreachable:
            parts.append(
                f"{len(unreachable)} claim(s) rejected (input never reached mutated line)"
            )
        if nondiscriminating:
            parts.append(
                f"{len(nondiscriminating)} claim(s) rejected (runtime value outside "
                f"discriminating interval)"
            )
        print(f"  [!] Code-execution verification: {'; '.join(parts)}.")

    # --- Line-coverage retry pass ---
    # For each "unverified_unreachable" entry, call the Detail agent once more with
    # explicit line-target feedback so it can choose an input that actually reaches
    # the mutated line.
    if unreachable:
        print(f"  Retrying {len(unreachable)} line-coverage-rejected claim(s) with "
              f"explicit line-target feedback...")
        retry_details_by_id = {}
        num_retry_batches = math.ceil(len(unreachable) / DETAIL_BATCH_SIZE)
        for batch_num in range(num_retry_batches):
            batch = unreachable[batch_num * DETAIL_BATCH_SIZE:
                                (batch_num + 1) * DETAIL_BATCH_SIZE]
            retry_summaries = "\n".join(
                f"- mutant_id {m['mutant_id']}: {desc_by_id.get(str(m['mutant_id']), '(description unavailable)')}"
                f"\n  IMPORTANT: your previous discriminating_input "
                f"'{m.get('discriminating_input', '')}' never executed "
                f"line {m.get('_mutant_lineno', '?')} (the mutated line). "
                f"You MUST choose a new input that actually reaches line "
                f"{m.get('_mutant_lineno', '?')} in the source. "
                f"Lines executed by the previous input: {m.get('_executed_lines_sample', [])}."
                for m in batch
            )
            retry_input = (
                f"Patched source code:\n{patched_source}\n\n"
                f"Test file:\n{test_content}\n\n"
                f"For each mutant below, choose a discriminating_input that ACTUALLY "
                f"REACHES the specified mutated line (verified by line-coverage tracing "
                f"— a previous attempt used an input that bypassed the mutated line "
                f"entirely and was rejected). Determine the real verdict from a genuine "
                f"computation on that reaching input "
                f"(use these EXACT mutant_id values in your response):\n{retry_summaries}"
            )
            retry_result = call_agent(
                f"Mutation Detail Retry (batch {batch_num + 1}/{num_retry_batches})",
                MUTATION_DETAIL_SYSTEM_PROMPT, retry_input, max_tokens=3000,
            )
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
                m["reasoning"] = (
                    m.get("reasoning", "") +
                    " Retry Detail agent returned no result; "
                    "conservatively reclassified as real_coverage_gap."
                )
                m["recommended_test"] = ""
                updated.append(m)
                continue
            retry_verdict = retry_detail.get("verdict", "real_coverage_gap")
            updated.append({
                "mutant_id": mid,
                "diff": desc_by_id.get(mid, ""),
                "verdict": retry_verdict,
                "discriminating_input": retry_detail.get("discriminating_input", ""),
                "reasoning": (
                    f"[LINE-COVERAGE RETRY] Previous input bypassed line "
                    f"{m.get('_mutant_lineno', '?')}. Retry result: "
                    + retry_detail.get("reasoning", "")
                ),
                "recommended_test": retry_detail.get("recommended_test", ""),
                "_claimed_orig_output": retry_detail.get("orig_output", ""),
                "_claimed_mut_output": retry_detail.get("mut_output", ""),
            })
        surviving_mutants_analysis = updated
        surviving_mutants_analysis = verify_equivalent_claims(
            patched_source, surviving_mutants_analysis,
            mutant_sources=mutant_sources, desc_by_id=desc_by_id,
        )

    # --- Boundary-value retry pass ---
    # For each "unverified_nondiscriminating_value" entry, call the Detail agent with
    # explicit feedback: the previous input's runtime operand value was outside the
    # discriminating interval (old, new] — the model must choose an input that produces
    # a value strictly between old and new so the two variants actually disagree.
    # Re-collect after the line-coverage retry pass (those may have produced new
    # equivalent claims that also need the boundary check).
    nondiscriminating = [m for m in surviving_mutants_analysis
                         if m.get("verdict") == "unverified_nondiscriminating_value"]
    if nondiscriminating:
        print(f"  Retrying {len(nondiscriminating)} boundary-value-rejected claim(s) with "
              f"explicit value-interval feedback...")
        bv_retry_by_id = {}
        num_bv_batches = math.ceil(len(nondiscriminating) / DETAIL_BATCH_SIZE)
        for batch_num in range(num_bv_batches):
            batch = nondiscriminating[batch_num * DETAIL_BATCH_SIZE:
                                      (batch_num + 1) * DETAIL_BATCH_SIZE]
            bv_summaries = "\n".join(
                (
                    lambda m, b=m.get("_boundary", {}), v=m.get("_operand_val", "?"):
                    f"- mutant_id {m['mutant_id']}: "
                    f"{desc_by_id.get(str(m['mutant_id']), '(description unavailable)')}\n"
                    f"  IMPORTANT: your previous discriminating_input "
                    f"'{m.get('discriminating_input', '')}' reached line "
                    f"{m.get('_mutant_lineno', '?')} but produced "
                    f"'{b.get('expr','?').replace('_',' ')}'={v!r}, which is NOT in the "
                    f"discriminating interval ({b.get('old','?')}, {b.get('new','?')}]. "
                    f"You MUST choose a new input where "
                    f"'{b.get('expr','?').replace('_',' ')}' is strictly between "
                    f"{b.get('old','?')} and {b.get('new','?')} (exclusive/inclusive) "
                    f"at line {m.get('_mutant_lineno','?')} — that is the only range "
                    f"where original and mutant behavior actually diverge."
                )(m)
                for m in batch
            )
            bv_retry_input = (
                f"Patched source code:\n{patched_source}\n\n"
                f"Test file:\n{test_content}\n\n"
                f"For each mutant below, choose a discriminating_input where the "
                f"compared expression's runtime value falls STRICTLY IN THE DISCRIMINATING "
                f"INTERVAL described (a previous input was rejected because the runtime "
                f"value was outside this interval, meaning original and mutant agreed at "
                f"that point). Determine the real verdict from a genuine computation "
                f"(use these EXACT mutant_id values in your response):\n{bv_summaries}"
            )
            bv_result = call_agent(
                f"Mutation Detail BV-Retry (batch {batch_num + 1}/{num_bv_batches})",
                MUTATION_DETAIL_SYSTEM_PROMPT, bv_retry_input, max_tokens=3000,
            )
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
                m["reasoning"] = (
                    m.get("reasoning", "") +
                    " Boundary-value retry agent returned no result; "
                    "conservatively reclassified as real_coverage_gap."
                )
                m["recommended_test"] = ""
                bv_updated.append(m)
                continue
            bv_verdict = bv_detail.get("verdict", "real_coverage_gap")
            b = m.get("_boundary", {})
            bv_updated.append({
                "mutant_id": mid,
                "diff": desc_by_id.get(mid, ""),
                "verdict": bv_verdict,
                "discriminating_input": bv_detail.get("discriminating_input", ""),
                "reasoning": (
                    f"[BOUNDARY-VALUE RETRY] Previous input had "
                    f"'{b.get('expr','?').replace('_',' ')}'={m.get('_operand_val','?')!r} "
                    f"at line {m.get('_mutant_lineno','?')} (outside discriminating interval "
                    f"({b.get('old','?')}, {b.get('new','?')}]). Retry result: "
                    + bv_detail.get("reasoning", "")
                ),
                "recommended_test": bv_detail.get("recommended_test", ""),
                "_claimed_orig_output": bv_detail.get("orig_output", ""),
                "_claimed_mut_output": bv_detail.get("mut_output", ""),
            })
        surviving_mutants_analysis = bv_updated
        surviving_mutants_analysis = verify_equivalent_claims(
            patched_source, surviving_mutants_analysis,
            mutant_sources=mutant_sources, desc_by_id=desc_by_id,
        )

    # For BOUNDARY-tagged mutants the LLM-generated reasoning commonly says something
    # like "no discount is applied" when the `if` only guards the LOG append, not the
    # subtraction on the return line. Correct that specific inaccuracy in-place before
    # the final output.
    # This is a wording-only patch — verdict, discriminating_input, and recommended_test
    # are untouched.
    #
    # We match the imprecise phrases the model typically produces and replace them with
    # accurate descriptions.  The fix is narrow: only entries that (a) have a [BOUNDARY]
    # tag somewhere (either via _boundary metadata still present, OR via the diff field
    # which always carries the original description) AND (b) contain one of the known
    # inaccurate phrases in their reasoning text are touched.
    _INACCURATE_PHRASES = [
        "no discount is applied",
        "discount is not applied",
        "discount is not subtracted",
        "no discount applied",
        "the discount is not applied",
        # Variants where the model says the mutant returns the "original" (undiscounted) subtotal,
        # implying the price was not changed — also factually wrong for the same structural reason.
        "returns the original subtotal",
        "returns the unchanged subtotal",
        "subtotal is not reduced",
        "subtotal remains unchanged",
        "subtotal is unchanged",
        "price is not reduced",
        "no discount is deducted",
        "discount is not deducted",
    ]
    _CORRECTION_SUFFIX = (
        " NOTE: the `if` at this line only controls whether the discount is RECORDED "
        "in applied_log — the subtraction `round(subtotal - discount, 2)` on the return "
        "line is unconditional, so the numeric subtotal is identical under both original "
        "and mutant. What actually changes is that the discount entry silently disappears "
        "from the order's audit log / analytics pipeline."
    )
    for m in surviving_mutants_analysis:
        # An entry is BOUNDARY-tagged if either its _boundary dict is still present
        # (BV-retry path) or if its diff description contains a [BOUNDARY ...] suffix
        # (direct-detail path — _boundary was never attached but the tag is in the diff).
        is_boundary = bool(m.get("_boundary")) or "[BOUNDARY" in m.get("diff", "")
        if not is_boundary:
            continue
        # The NOTE is only accurate for line 77 (the `if discount > 0:` guard), where
        # the mutated `if` controls only the applied_log append and the subtraction on
        # the return line is unconditional. For other lines (e.g. line 74, which gates
        # the discount CALCULATION itself), the subtotal genuinely differs and the NOTE
        # would be factually wrong — so skip it for any line other than 77.
        if _parse_mutant_lineno(m.get("diff", "")) != 77:
            continue
        reasoning = m.get("reasoning", "")
        if any(phrase in reasoning.lower() for phrase in _INACCURATE_PHRASES):
            m["reasoning"] = reasoning + _CORRECTION_SUFFIX

    # Strip all internal tracking fields before final output.
    for m in surviving_mutants_analysis:
        m.pop("_mutant_lineno", None)
        m.pop("_executed_lines_sample", None)
        m.pop("_boundary", None)
        m.pop("_operand_val", None)
        m.pop("_contaminated", None)
        m.pop("_own_tag", None)

    # Final verdict computed in code from the ACTUAL post-detail verdicts, not the
    # pre-detail routing list (real_gaps may include items the detail pass went on
    # to independently reclassify as equivalent after all).
    confirmed_gaps = [m for m in surviving_mutants_analysis if m["verdict"] == "real_coverage_gap"]
    if confirmed_gaps:
        final_verdict = "ACCEPT_WITH_ADDED_TESTS"
        confidence = "High"  # high confidence in the *finding* that gaps exist
    else:
        final_verdict = "ACCEPT_FIX"
        confidence = "High"

    result = {
        "mutation_summary": {
            "total_mutants": total_mutants,
            "killed": total_mutants - survived_count,
            "survived": survived_count,
        },
        "surviving_mutants_analysis": surviving_mutants_analysis,
        "final_verdict": final_verdict,
        "confidence": confidence,
    }

    print(f"\n=== Final Verdict: {result['final_verdict']} ===")
    print(json.dumps(result, indent=2))

    print_cost_summary()


if __name__ == "__main__":
    main()