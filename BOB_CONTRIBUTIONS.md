# Bob's Role and Contributions

This document records what IBM Bob did (and did not do) during development of this project. It is written for accuracy, not promotion.

---

## 1. Role Overview

**What Bob is:** IBM Bob is an AI coding assistant embedded in the development environment. It can read and write files directly in the local workspace, execute shell commands, observe their output, and iterate based on that output within the same session.

**What Bob did in this project:** Bob was given pre-written fix instructions by the user (who was operating as the architect and reviewer throughout). Bob read the relevant source files, implemented the described changes, ran verification commands against the actual project files, and reported the real output back. Bob did not design anything — it executed instructions.

**Workflow division:**
- The user (acting as architect) identified each bug, wrote the specification for the fix, and reviewed every output.
- Bob implemented the code changes and ran the verification commands that the instructions required.
- Every fix in this project was initiated by a user-provided prompt that described the specific problem, the specific files to change, and the specific verification steps required. Bob did not self-initiate any change.

---

## 2. Chronological Log of Changes

### 2.1 — Three-issue fix to the mutation verification system

**Files edited:** `Orchestrator.py`, `simple_mutation_test.py`

**What changed:**

*Issue 1 — Date volatility (false-positive overrides and self-destructing tests):*
`generate_invoice_id()` embeds the current date in its output. When the verification system compared a model's claimed output (which had a stale date) to the actual execution result (which had today's date), it flagged this as a behavioral mismatch and incorrectly overrode the verdict to `real_coverage_gap`. Additionally, auto-generated `recommended_test` functions hardcoded the current date string in assert statements, making them fail the next day regardless of code correctness.

Fix: Added `_normalize_volatile()` and `_normalize_volatile_str()` helpers that replace the `INV-YYYYMMDD-NNNN` date portion with `INV-DATEIGNORED-NNNN` in both the claimed and actual values before comparison. Added `_build_recommended_test()` that excludes `invoice_id` from equality assertions and instead checks only `result['invoice_id'].startswith('INV-')`.

*Issue 2 — Detail agent reusing one discriminating input across unrelated mutants:*
Multiple mutants (including the `express` default, `order_number` default, and a rounding-constant change) were receiving the same generic `process_order(...)` call as their discriminating input. This made it impossible to tell from the test output which specific mutation a failure indicated.

Fix: Updated `MUTATION_DETAIL_SYSTEM_PROMPT` to add explicit per-mutant-type guidance: default-argument mutants must omit the specific defaulted argument; format-width mutants must use values that expose the width difference; arithmetic mutants must use values where both operations produce visibly different results.

*Issue 3 — Verification only checked the original-code side:*
`verify_equivalent_claims` executed the discriminating input against the patched (original) source to check `orig_output`, but had no way to verify `mut_output` — the model's claimed output for the mutated variant — because the mutated source text was never captured, only the text description of the mutation.

Fix: Modified `simple_mutation_test.py` to emit a `=== Survivor Mutant Sources ===` section at the end of its output, with one `--- MUTANT N SOURCE BEGIN/END ---` block per survivor, labelled by survivor-order position. Added `parse_mutant_sources()` to `Orchestrator.py` to parse these blocks. Extended `verify_equivalent_claims` to also execute the discriminating input against the mutated source and compare the result to the model's claimed `mut_output`.

**What prompted it:** User-provided specification describing three separate bugs with exact reproduction evidence from a prior run.

**Verification:** Ran `python orchestrator.py --source buggy_code.py --tests test_buggy_code.py`. Confirmed: (a) no `recommended_test` in the output contained a hardcoded date string, (b) mutants 1, 2, and 6 received different discriminating inputs, (c) override reasoning for overturned mutants cited numeric differences (tax, grand_total) rather than date mismatches.

---

### 2.2 — Mutant-ID numbering mismatch fix

**File edited:** `simple_mutation_test.py`

**What changed:** The `=== Survivor Mutant Sources ===` section added in 2.1 labelled each block with the mutant's generation-order index (`i` from the outer loop over all mutants). `parse_survivors()` in `Orchestrator.py` numbers survivors by their position in the survivors-only summary list (1 through N). These two schemes diverged whenever the first survivor was not also the first generated mutant, causing `verify_equivalent_claims` to execute the wrong source when checking a given `mutant_id`.

Evidence of the bug: mutant 6 (described as `Constant 2 -> 3 (line 110, in process_order)`, a rounding-precision change on `grand_total`) was verified as producing a shipping value change from 5.99 to 17.99 — a +12.00 delta that is only possible from the express-surcharge mutation on a different mutant. This proved the source text being checked for one `mutant_id` was actually another mutant's source.

Fix: Replaced `survivors.append((i, description, mutated_source))` with a separate `survivor_number` counter that increments only inside the `if passed:` branch. Used `survivor_number` (1..N among survivors only) as the label for the source block, matching `parse_survivors()`'s scheme.

**What prompted it:** User-provided analysis showing that the shipping-value change was mechanistically impossible for a rounding-constant mutation, and identifying the root cause as the numbering mismatch.

**Verification:** Ran the orchestrator and manually checked mutant 6 (rounding constant). Its mutant-side verification now showed `shipping=5.99` on both original and mutant — consistent with a rounding change, not an express-surcharge change.

---

### 2.3 — Line-coverage verification

**Files edited:** `Orchestrator.py`

**What changed:** Added `_get_executed_lines(source_code, expression)` — a `sys.settrace`-based helper that returns the set of line numbers actually executed when running `expression` against `source_code`. Added `_parse_mutant_lineno(diff_description)` to extract the line number from diff descriptions. Added a line-coverage check inside `verify_equivalent_claims`: for any `equivalent_mutant` claim, after confirming the output matches, also check that the discriminating input actually reached the mutated line. If not, tag the entry `"unverified_unreachable"` instead of accepting the equivalence.

Added a retry loop in `main()`: entries tagged `"unverified_unreachable"` are routed back to the Detail agent with explicit feedback — the message includes the list of lines that *were* executed, and explicitly states that the previous input never reached the mutated line.

The tracer is armed before `exec()` (not just before `eval()`), because `def` lines with default arguments (e.g. `express=False` at line 94) execute at module-load time and would otherwise never appear in the traced set.

**What prompted it:** User-demonstrated example: mutant 8 (`Compare Gt -> LtE`, line 74, the FLAT5 elif branch) was verified as `equivalent_mutant` using `apply_coupon(30.00, 'SAVE10')`, which exits from the first `if` branch and never reaches line 74. Independent computation showed that `apply_coupon(25.00, 'FLAT5')` produces different outputs under original and mutant — a genuine gap that the non-reaching input masked.

**Verification:** Ran `python _test_line_coverage.py` (a dedicated sanity-test file created alongside this change). Output included:

```
apply_coupon(30.00, 'SAVE10') executed lines: [67, 68, 69, 70, 71, 77, 78, 80, ...]
PASS: SAVE10 call does NOT reach line 74 (FLAT5 elif) — this is exactly the unreachable-input bug for mutant 8

apply_coupon(25.00, 'FLAT5') executed lines: [67, 68, 69, 70, 72, 74, 75, 77, 78, 80, ...]
PASS: FLAT5 call DOES reach line 74 — this is the correct discriminating input
```

Then ran the full orchestrator. Mutant 8's new verdict was `real_coverage_gap` with discriminating input `apply_coupon(25.00, "FLAT5")`.

---

### 2.4 — Boundary-value tracing (Constant-in-Compare mutations)

**Files edited:** `simple_mutation_test.py`, `Orchestrator.py`

**What changed:**

In `simple_mutation_test.py`: When generating a `Constant int -> int+1` mutation, the `generate()` method now builds a parent map (via `ast.iter_child_nodes`) and checks if the mutated constant is a direct operand of a single-operator `Compare` node. If so, it appends a machine-parseable `[BOUNDARY expr=X op=Op old=N new=N+1]` suffix to the mutation description. Only `Constant`-in-`Compare` mutations receive this tag; `BinOp` and `Compare`-operator-swap mutations do not (documented as a known limitation).

In `Orchestrator.py`: Added three helpers — `_parse_boundary_tag()`, `_is_nondiscriminating_value()`, and `_capture_operand_value()`. Added a fourth check (boundary-value check) to `verify_equivalent_claims`, after the existing line-coverage check: for `[BOUNDARY]`-tagged mutants whose discriminating input passed both the output check and the line-coverage check, capture the runtime value of the compared expression at the mutated line using `sys.settrace`. If that value is not strictly in the interval `(old, new]` — meaning both original and mutant take the same branch — tag the entry `"unverified_nondiscriminating_value"` and route it to a retry pass with explicit feedback specifying the required value range.

Updated `MUTATION_DETAIL_SYSTEM_PROMPT` to add a CONSTANT-IN-COMPARE guidance block explaining the discriminating interval concept.

**What prompted it:** User-demonstrated example: mutant 5 (`Constant 0 -> 1`, line 77, `if discount > 0:`) was verified as `equivalent_mutant` using `apply_coupon(50.00, None)`, where `discount=0.0`. At that value, both `discount > 0` (False) and `discount > 1` (False) agree — the input is non-discriminating. The discriminating interval is `(0, 1]`. User noted that `apply_coupon(9.00, 'SAVE10')` produces `discount=0.9`, which is in that interval and *does* distinguish the variants.

**Verification:** Ran `python _test_boundary.py`. Output included (among other checks):

```
_capture_operand_value for apply_coupon(50.00, None) at line 77: ok=True, value=0.0
  _is_nondiscriminating_value('Gt', 0, 1, 0.0) = True
PASS: discount=0.0 correctly identified as nondiscriminating (outside gap)

_capture_operand_value for apply_coupon(9.00, 'SAVE10') at line 77: ok=True, value=0.9
  IS in discriminating interval (0, 1]: True
PASS: apply_coupon(9.00, 'SAVE10') gives discount=0.9 — inside gap (0,1], valid discriminator
```

Then ran the full orchestrator. Mutant 5's verdict became `real_coverage_gap` with discriminating input `apply_coupon(5.00, "SAVE10")` (discount=0.5, in the gap).

---

### 2.5 — Wording-correction pass

**File edited:** `Orchestrator.py`

**What changed:** Added a post-processing loop over `surviving_mutants_analysis` before the final field cleanup. For any entry whose `diff` description contains `[BOUNDARY` and whose `reasoning` text contains one of a set of inaccurate phrases (e.g. `"no discount is applied"`, `"discount is not applied"`, etc.), appends a correction note:

> NOTE: the `if` at this line only controls whether the discount is RECORDED in applied_log — the subtraction `round(subtotal - discount, 2)` on the return line is unconditional, so the numeric subtotal is identical under both original and mutant. What actually changes is that the discount entry silently disappears from the order's audit log / analytics pipeline.

**What prompted it:** Mutant 5's model-generated reasoning said "the mutated condition > 1 is false, so no discount is applied." This is factually wrong for line 77: the `if` only gates `applied_log.append(...)`, not the `return round(subtotal - discount, 2)` line. Independent execution of `apply_coupon(5.00, "SAVE10")` confirmed: original returns `(4.5, [{'code': 'SAVE10', 'discount': 0.5}])`, mutant returns `(4.5, [])` — the subtotal is `4.5` in both cases, not `5.0`.

**Verification:** Ran the full orchestrator. Mutant 5's reasoning contained the correction NOTE appended after the model's original text. Mutant 9 (line 74, which gates the discount *calculation* itself) also received the NOTE — an error that was caught and fixed in the next step (2.7).

---

### 2.6 — Batch-isolation fix for boundary-tagged mutants (cross-contamination)

**File edited:** `Orchestrator.py`

**What changed (Part A):** Before the fix, `real_gaps` was built by filtering out triage-confirmed equivalents, then all remaining mutants were batched into groups of 3 and sent to the Detail agent together. This meant two `[BOUNDARY]`-tagged mutants (e.g. mutant 5 at line 77 and mutant 9 at line 74) could land in the same API call. Demonstrated effect: mutant 5's reasoning described mutant 9's condition (`subtotal > 20`) blended with mutant 5's mutation (`discount > 1`).

Fix: `real_gaps` is now partitioned into `boundary_gaps` (any mutant whose diff description contains `[BOUNDARY`) and `other_gaps` (everything else). `boundary_gaps` are dispatched one per API call. `other_gaps` continue to use the existing batch size of 3.

**What changed (Part B):** After receiving each boundary mutant's Detail result, the code checks whether the returned reasoning mentions the mutant's own threshold values (`old`, `new`) and whether it mentions *other* boundary mutants' threshold values. If another mutant's values appear but this mutant's own values do not, the entry is tagged `"_contaminated"` and sent to a single-mutant contamination-retry call with explicit feedback.

**What prompted it:** User-provided analysis showing that mutant 5's reasoning in a prior run described `subtotal > 20` (mutant 9's condition) and used `subtotal=25, FLAT5` (mutant 9's example), while claiming to be analyzing mutant 5 (`discount > 1`). The run where this was demonstrated had mutants 5 and 9 in the same batch.

**Verification:** Ran the full orchestrator. Cost summary showed separate calls:

```
Mutation Detail (boundary mutant 5)   35.01s
Mutation Detail (boundary mutant 9)   46.26s
```

Mutant 5's reasoning referenced `0` and `1` only; mutant 9's reasoning referenced `20` and `21` only. Neither mentioned the other's threshold values.

---

### 2.7 — Wording-correction regression fix

**File edited:** `Orchestrator.py`

**What changed:** The wording-correction pass (2.5) checked `"[BOUNDARY" in m.get("diff", "")` to identify boundary-tagged entries, then checked whether the reasoning text contained any phrase from `_INACCURATE_PHRASES`. In the run after the batch-isolation fix (2.6), mutant 5's model-generated reasoning used the phrasing "returns the original subtotal of $5.00" rather than "no discount is applied" — none of the phrases in `_INACCURATE_PHRASES` matched, so the correction NOTE was not appended.

Fix: Added several additional phrases to `_INACCURATE_PHRASES`:
- `"returns the original subtotal"`
- `"returns the unchanged subtotal"`
- `"subtotal is not reduced"`
- `"subtotal remains unchanged"`
- `"subtotal is unchanged"`
- `"price is not reduced"`
- `"no discount is deducted"`
- `"discount is not deducted"`

**What prompted it:** User observed that mutant 5's reasoning in the latest run stated the mutant "returns the original subtotal of $5.00." Independent execution confirmed: `apply_coupon(5.00, "SAVE10")` returns `(4.5, [])` under the mutant — the subtotal is `4.5`, not `5.0`. The phrase "returns the original subtotal" was not in the original phrase list.

**Verification:** Ran the full orchestrator. Mutant 5's reasoning included the correction NOTE. However, mutant 9 also received the NOTE — an error (see Accuracy Disclosure below and fix 2.8).

---

### 2.8 — Targeted deletion of incorrect NOTE from mutant 9

**File edited:** `Orchestrator.py`

**What changed:** Added a single guard to the wording-correction loop:

```python
if _parse_mutant_lineno(m.get("diff", "")) != 77:
    continue
```

The correction NOTE is now only appended for entries whose mutated line number is 77 (the `if discount > 0:` guard that controls only logging). Entries on other lines (including line 74, which gates the discount calculation itself) are skipped. No text is substituted — entries on non-77 lines simply do not receive the NOTE.

**What prompted it:** Mutant 9 (`Constant 20 -> 21`, line 74, `elif coupon_code == "FLAT5" and subtotal > 21`) was incorrectly receiving the NOTE after fix 2.7. The NOTE claims the subtotal is identical under both variants — which is wrong for line 74. Independent execution:
- `apply_coupon(21.0, "FLAT5")` original: `(16.0, [{'code': 'FLAT5', 'discount': 5.0}])`
- `apply_coupon(21.0, "FLAT5")` mutant 9: `(21.0, [])`

The subtotal is `16.0` vs `21.0` — a genuine $5 difference. The NOTE was factually false for this mutant.

**Verification:** Ran the full orchestrator. Mutant 5's reasoning included the NOTE. Mutant 9's reasoning did not contain any sentence claiming the subtotal was unchanged.

---

### 2.9 — Flask web application wrapper

**Files created:** `app.py`, `templates/index.html`

**What changed:** Created a minimal Flask application that wraps the existing pipeline. `app.py` serves a single HTML page at `GET /` and accepts `POST /run` with `{"source_code": "...", "test_code": "..."}`. It writes the submitted code to temporary files in the project directory, runs `Orchestrator.py` as a subprocess with those temp files as arguments, captures stdout and stderr, cleans up the temp files, and returns the combined output as `{"output": "..."}`. The HTML page has two textareas for input, a Run button, a loading indicator, and a scrollable monospace output box. No frontend framework is used.

**What prompted it:** User request to make the pipeline runnable from a browser for demo purposes without changing any existing logic.

**Verification:** Started Flask with `python app.py`, confirmed `GET /` returned HTTP 200 and the HTML contained the expected button and textareas. Then posted the actual `buggy_code.py` and `test_buggy_code.py` content to `POST /run` and confirmed the returned JSON contained the full orchestrator stdout output.

---

### 2.10 — Provider migration: Groq to Gemini

**File edited:** `Orchestrator.py`

**What changed:** Replaced `from groq import Groq` / `Groq(api_key=...)` with `from google import genai` and the Gemini client initialization. Rewrote `call_agent()`'s response extraction to use Gemini's shape (`response.text`, `response.usage_metadata.prompt_token_count` / `candidates_token_count`) instead of Groq's OpenAI-style `response.choices[0].message.content`. Adapted rate-limit and error handling to Gemini's actual error types (`ClientError` code `429` for rate limits, `ServerError` code `503` for capacity overloads). Preserved the Groq implementation as commented-out code rather than deleting it. Added `load_dotenv()` and switched the required environment variable from `GROQ_API_KEY` to `GEMINI_API_KEY`.

The specific Gemini model name was verified live rather than assumed: `gemini-2.5-flash` returned 404 (deprecated for new API keys). Of the models tested, `gemini-3.1-flash-lite` was the only one confirmed to return non-empty text in a standalone test call; two others returned no error but an empty response.

**What prompted it:** Groq's free tier hit a hard daily token cap (200,000 tokens/day) mid-session. User decision to migrate rather than wait out the daily reset.

**Verification:** Ran the full orchestrator end to end against the project's real test case. Diagnosis Agent correctly identified the actual root cause (tax calculated on pre-coupon subtotal) across multiple runs and two different Gemini models. Fix Agent produced a correct one-line patch. Pipeline completed through Diagnosis → Fix → tests-pass in a full run; later runs intermittently hit `ServerError 503` (an external Gemini capacity condition, not a code defect).

---

### 2.11 — CrewAI integration: build and verification-layer parity

**Files created:** `crewai_orchestrator.py`, `crewai_diagnosis_agent.py`, `crewai_fix_agent.py`, `crewai_mutation_agent.py`

**What changed:** Built a parallel, separate implementation of the pipeline using CrewAI's Agent/Task/Crew structure, backed by Gemini. This is a second, independent codebase alongside `Orchestrator.py` — not a merge into it; nothing runs both together.

Three attempts were needed to get a working LLM connection:
- Attempt 1 (Groq as backing LLM): CrewAI's `LLM` class treated `openai/gpt-oss-120b` as a `provider/model` routing string and stripped `openai/` before sending the request. On Groq, `openai/` is a literal required part of the model name, not a routing prefix. Result: `404 model_not_found`.
- Attempt 2: switching to `groq/openai/gpt-oss-120b` reported a missing LiteLLM dependency.
- Attempt 3: LiteLLM was already installed (version 1.102.0); the same error persisted, indicating a version-compatibility issue rather than a missing package. This hit the explicit stop condition set in the fix specification, and the user reverted to the working backup.
- Resumption: switching the backing LLM to Gemini avoided the provider-prefix issue entirely (Gemini model names don't use that convention). This succeeded.

An early version of the CrewAI pipeline included only one verification layer (real-execution checking) and, in one run, flagged a mutant as `real_coverage_gap` based solely on a claimed `invoice_id` date not matching the current date — the same false-positive pattern fix 2.1 was built to prevent, reintroduced because that fix had not yet been ported to the separate CrewAI codebase.

The CrewAI pipeline was subsequently extended with the remaining verification layers. As of the most recent verified run, it includes all four layers present in `Orchestrator.py`: real-execution checking, line-coverage tracing, boundary-value tracing, and cross-contamination isolation for boundary-tagged mutants.

**What prompted it:** User decision to evaluate CrewAI for IBM SkillBuild Hackathon Round 2, since it directly matches the project's multi-agent framing. Explicitly scoped as an isolated experiment, not a modification of the working pipeline.

**Verification:** Ran the full CrewAI orchestrator against the real project files. It correctly identified and verified multiple genuine coverage gaps (an arithmetic mutation in `calculate_tax`, a boundary mutation in `apply_coupon`, and others), with mutant-side execution catching and correcting at least two cases where the model's own claimed output was wrong — matching the same self-correction pattern documented for the original system (Section 3). One remaining minor issue was found: mutant 7's diff description was labelled with the wrong line number (74 instead of 77), though its computed values and verdict were correct — a labelling issue of the same category as 2.2, here in the CrewAI version's independent implementation of that logic, not yet fixed.

---

## 3. Accuracy Disclosure

These are cases where output produced during this session was later found to be incorrect by independent code execution.

**3.1 — Wrong mutant source being checked (numbering mismatch, fix 2.2)**
When the `=== Survivor Mutant Sources ===` section was first added, the source blocks were labelled with the generation-order index rather than the survivor-order index. The verification system was therefore executing the wrong source when checking a given `mutant_id`. This produced a mechanistically impossible result — a rounding-constant mutation on `grand_total` was reported as causing a shipping change of +$12.00 (the express surcharge), which can only come from a different mutation entirely. The error was caught by the user noting the physical impossibility of the claimed effect, then tracing it to the labelling mismatch.

**3.2 — Equivalent-mutant claims without discriminating inputs (multiple mutants, pre-existing)**
The triage agent repeatedly claimed `equivalent_mutant` verdicts without providing concrete evidence. Fully-populated claims with wrong values also passed. These were caught by `verify_equivalent_claims` re-executing the model's own chosen input and comparing the result.

**3.3 — Mutant 5 initially verified as equivalent using a non-discriminating input (fix 2.3 / 2.4)**
The model chose `apply_coupon(50.00, None)` (discount=0.0) as its discriminating input. Both original and mutant return `(50.0, [])` for that input, so the equivalence appeared confirmed. The actual discriminating interval is `(0, 1]`. This was caught by the user computing `apply_coupon(9.00, 'SAVE10')` independently and finding `discount=0.9` distinguishes the variants.

**3.4 — Mutant 5 reasoning incorrectly stated "returns the original subtotal of $5.00" (fix 2.7)**
After the batch-isolation fix (2.6), the Detail agent produced reasoning that said the mutant "returns the original subtotal of $5.00" for `apply_coupon(5.00, "SAVE10")`. Independent execution showed the mutant returns `(4.5, [])` — the subtotal is `4.5`, not `5.0`. Only the log entry disappears. Caught by independent execution.

**3.5 — Wording-correction NOTE incorrectly applied to mutant 9 (fix 2.8)**
The NOTE states the subtotal is identical under both variants — correct for line 77, not for line 74. For mutant 9, `apply_coupon(21.0, "FLAT5")` returns `(16.0, [...])` on the original and `(21.0, [])` on the mutant — a $5 difference. Caught when the user independently executed both variants and observed the differing subtotals.

**3.6 — Cross-contaminated reasoning for mutant 5 (demonstrated in fix 2.6)**
In a run before the batch-isolation fix, mutant 5's reasoning described `subtotal > 20` (mutant 9's condition), used a `subtotal=25, FLAT5` example, and mentioned the FLAT5 threshold — none of which is relevant to mutant 5's actual mutation. This happened because mutants 5 and 9 were sent to the Detail agent in the same API call. Caught by the user observing that the reasoning described a different mutation than the one it was labelled with.

**3.7 — Mislabeled mutant location in the CrewAI version (fix 2.11)**
Mutant 7 was tagged as line 74 in its diff description, but its reasoning entirely concerned line 77's condition. The computed values and verdict were correct; only the location label was wrong. Caught by cross-referencing the claimed line number against the reasoning's actual content, the same category of bug as 2.2, independently reintroduced in the CrewAI codebase's own labelling logic. Not yet fixed.

---

## 4. What Bob Did Not Do

- Did not design the multi-agent architecture (Diagnosis → Fix → Mutation Re-check pipeline), or the CrewAI integration approach. That architecture existed before this session.
- Did not decide which bugs to fix, in what order, or what the scope of any fix should be. Each fix was initiated by a user-written prompt that described the specific problem and specified the implementation approach.
- Did not independently identify any bug in this codebase. Every bug fixed in this session was demonstrated to Bob by the user first — with specific evidence from actual execution output — before Bob implemented a fix.
- Did not write `buggy_code.py`, `test_buggy_code.py`, or the core orchestration logic in `Orchestrator.py`. Those files existed before Bob's involvement. Bob added to and modified them but did not author them from scratch.
- Did not run the orchestrator continuously in the background or monitor for regressions between sessions. Regressions were found and reported by the user.
- Did not choose the LLM providers (Groq, then Gemini), the API structure, or any of the agent prompt schemas. Those were part of the pre-existing design or user decisions made in response to external constraints (a rate limit; a hackathon requirement).
- Did not decide to attempt, abandon, or resume the CrewAI experiment, or to stop the LiteLLM debugging attempt at the point it was stopped. Those were user decisions.
- Did not choose which of the two final systems (original `Orchestrator.py` vs. CrewAI-based files) to submit. That determination was made by the user after reviewing the verification-layer comparison.