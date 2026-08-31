"""Quick sanity test for the three fixes."""
import sys
sys.path.insert(0, ".")
import datetime
from Orchestrator import (
    _normalize_volatile, _normalize_volatile_str, _values_match,
    _build_recommended_test, parse_mutant_sources, parse_survivors,
)

today = datetime.date.today().strftime("%Y%m%d")

# -------------------------------------------------------------------
# Issue 1a: _normalize_volatile strips date from invoice_id in a dict
# -------------------------------------------------------------------
actual = {"invoice_id": f"INV-{today}-0001", "subtotal": 10.0, "grand_total": 11.0}
normalized = _normalize_volatile(actual)
assert normalized["invoice_id"] == "INV-DATEIGNORED-0001", f"Got {normalized['invoice_id']!r}"
print("PASS: _normalize_volatile strips date portion from invoice_id in a dict")

# -------------------------------------------------------------------
# Issue 1b: _values_match ignores date drift
# -------------------------------------------------------------------
stale_claimed = "{'invoice_id': 'INV-20241231-0001', 'subtotal': 10.0, 'grand_total': 11.0}"
result = _values_match(stale_claimed, actual)
assert result, "Expected stale-date claimed to match today's actual after normalization"
print("PASS: _values_match ignores date drift between claimed (stale) and actual (today)")

# Values that differ on non-date fields should still NOT match
different = {"invoice_id": f"INV-{today}-0001", "subtotal": 99.0, "grand_total": 100.0}
assert not _values_match(stale_claimed, different), "Non-date difference should still mismatch"
print("PASS: _values_match correctly rejects mismatches on non-date fields")

# -------------------------------------------------------------------
# Issue 1c: _build_recommended_test excludes invoice_id from equality
# -------------------------------------------------------------------
test_fn = _build_recommended_test("3", "process_order([{'name':'mug','price':10.0,'qty':1}])", actual)
assert today not in test_fn, f"Today's date should NOT appear in recommended_test, got:\n{test_fn}"
# invoice_id must NOT appear as a key in the asserted stable_fields dict literal.
# It IS expected to appear in the filter expression and in the startswith check.
assert "startswith('INV-')" in test_fn, "Should assert invoice_id starts with INV-"
# The RHS of the == assertion should only contain the stable fields (no invoice_id key).
stable = {"subtotal": 10.0, "grand_total": 11.0}
assert repr(stable) in test_fn, f"Expected stable fields {stable!r} in test body, got:\n{test_fn}"
print("PASS: _build_recommended_test excludes hardcoded date from test")
print("  Generated test:")
for line in test_fn.splitlines():
    print(f"    {line}")

# -------------------------------------------------------------------
# Issue 3: parse_mutant_sources / parse_survivors integration
# -------------------------------------------------------------------
fake_report = """Generated 5 mutants. Running tests against each...

  [1/5] killed   - Compare GtE -> Lt (line 47, in calculate_shipping)
  [2/5] SURVIVED - Compare GtE -> Lt (line 50, in calculate_shipping)
  [3/5] SURVIVED - Constant False -> True (line 94, in process_order)
  [4/5] killed   - BinOp Add -> Sub (line 110, in process_order)
  [5/5] killed   - Constant 2 -> 3 (line 85, in calculate_tax)

=== Mutation Testing Summary ===
Total mutants: 5
Killed: 3
Survived: 2

Surviving mutants (tests did NOT catch these):
  - Compare GtE -> Lt (line 50, in calculate_shipping)
  - Constant False -> True (line 94, in process_order)

=== Survivor Mutant Sources ===
--- MUTANT 2 SOURCE BEGIN ---
SOURCE CODE FOR MUTANT 2
SECOND LINE
--- MUTANT 2 SOURCE END ---
--- MUTANT 3 SOURCE BEGIN ---
SOURCE CODE FOR MUTANT 3
--- MUTANT 3 SOURCE END ---
"""

survivors = parse_survivors(fake_report)
assert survivors == [
    "Compare GtE -> Lt (line 50, in calculate_shipping)",
    "Constant False -> True (line 94, in process_order)",
], f"Got {survivors!r}"
print("PASS: parse_survivors reads survivors correctly and stops before === section")

sources = parse_mutant_sources(fake_report)
assert "2" in sources, f"Expected mutant 2 source, got keys: {list(sources.keys())}"
assert "3" in sources, f"Expected mutant 3 source, got keys: {list(sources.keys())}"
assert sources["2"].strip() == "SOURCE CODE FOR MUTANT 2\nSECOND LINE", f"Got {sources['2']!r}"
assert sources["3"].strip() == "SOURCE CODE FOR MUTANT 3", f"Got {sources['3']!r}"
print("PASS: parse_mutant_sources extracts source blocks correctly")

print()
print("All sanity checks passed.")
