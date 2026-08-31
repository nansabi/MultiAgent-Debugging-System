"""
Sanity tests for _get_executed_lines and _parse_mutant_lineno.
Prints the traced line sets so the mutant-8 scenario is visible, not just asserted.
"""
import sys, os
os.environ.setdefault("GROQ_API_KEY", "test_dummy_key")
sys.path.insert(0, ".")

from Orchestrator import _get_executed_lines, _parse_mutant_lineno
from pathlib import Path

SOURCE = Path("buggy_code.py").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# _parse_mutant_lineno
# ---------------------------------------------------------------------------
cases = [
    ("Compare Gt -> LtE (line 74, in apply_coupon)", 74),
    ("Constant False -> True (line 94, in process_order)", 94),
    ("BinOp Add -> Sub (line 110, in process_order)", 110),
    ("Constant 2 -> 3 (line 33, in calculate_item_total)", 33),
    ("BoolOp And -> Or", None),          # no location suffix → None
    ("", None),
]
for desc, expected in cases:
    got = _parse_mutant_lineno(desc)
    assert got == expected, f"_parse_mutant_lineno({desc!r}) = {got!r}, expected {expected!r}"
print("PASS: _parse_mutant_lineno all cases")

# ---------------------------------------------------------------------------
# _get_executed_lines: SAVE10 path — does NOT reach line 74
# ---------------------------------------------------------------------------
expr_save10 = "apply_coupon(30.00, 'SAVE10')"
lines_save10 = _get_executed_lines(SOURCE, expr_save10)
assert lines_save10 is not None, "Trace should succeed for valid expression"
print(f"\napply_coupon(30.00, 'SAVE10') executed lines: {sorted(lines_save10)}")
assert 74 not in lines_save10, (
    f"Line 74 (FLAT5 elif) should NOT be reached by SAVE10 call, "
    f"but found in: {sorted(lines_save10)}"
)
print("PASS: SAVE10 call does NOT reach line 74 (FLAT5 elif) — "
      "this is exactly the unreachable-input bug for mutant 8")

# ---------------------------------------------------------------------------
# _get_executed_lines: FLAT5 path — DOES reach line 74
# ---------------------------------------------------------------------------
expr_flat5 = "apply_coupon(25.00, 'FLAT5')"
lines_flat5 = _get_executed_lines(SOURCE, expr_flat5)
assert lines_flat5 is not None, "Trace should succeed for valid expression"
print(f"\napply_coupon(25.00, 'FLAT5') executed lines: {sorted(lines_flat5)}")
assert 74 in lines_flat5, (
    f"Line 74 (FLAT5 elif) SHOULD be reached by FLAT5 call, "
    f"but not found in: {sorted(lines_flat5)}"
)
print("PASS: FLAT5 call DOES reach line 74 — this is the correct discriminating input")

# ---------------------------------------------------------------------------
# _get_executed_lines: process_order default express path — reaches line 94
# ---------------------------------------------------------------------------
expr_po = "process_order([{'name':'mug','price':10.0,'qty':1}])"
lines_po = _get_executed_lines(SOURCE, expr_po)
assert lines_po is not None
print(f"\nprocess_order([...]) executed lines: {sorted(lines_po)}")
# Line 94 is the `def process_order(...)` line — it runs during exec() (module load),
# where default argument values are evaluated. The tracer is now armed before exec().
assert 94 in lines_po, (
    f"Line 94 should be in executed lines (captured during exec/module-load). "
    f"Got: {sorted(lines_po)}"
)
print("PASS: process_order call includes line 94 (def line, captured during exec/module-load)")

# ---------------------------------------------------------------------------
# _get_executed_lines: bad expression returns None without crashing
# ---------------------------------------------------------------------------
result_bad = _get_executed_lines(SOURCE, "nonexistent_function()")
assert result_bad is None, f"Expected None for bad expression, got {result_bad!r}"
print("\nPASS: bad expression returns None gracefully (no crash, no dangling tracer)")

# ---------------------------------------------------------------------------
# Verify sys.settrace is cleanly restored after each call
# ---------------------------------------------------------------------------
import sys as _sys
prev = _sys.gettrace()
_get_executed_lines(SOURCE, "calculate_item_total([])")
assert _sys.gettrace() is prev, "sys.gettrace() should be restored after _get_executed_lines"
print("PASS: sys.settrace cleanly restored after _get_executed_lines")

print("\nAll line-coverage sanity checks passed.")
