import os, sys, importlib.util
os.environ.setdefault("GROQ_API_KEY", "test_dummy_key")

BASE = r"d:\IBM-MULTI-AGENT"
sys.path.insert(0, BASE)

import ast

# Load modules from explicit absolute paths so cwd doesn't matter.
def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

smt = _load("simple_mutation_test", "simple_mutation_test.py")
orch = _load("Orchestrator", "Orchestrator.py")

Mutator = smt.Mutator
_build_function_ranges = smt._build_function_ranges
_parse_boundary_tag = orch._parse_boundary_tag
_is_nondiscriminating_value = orch._is_nondiscriminating_value
_capture_operand_value = orch._capture_operand_value

from pathlib import Path

SOURCE = Path("buggy_code.py").read_text(encoding="utf-8")

# ------------------------------------------------------------------
# 1. simple_mutation_test.py: check [BOUNDARY] tag is produced for
#    `if discount > 0:` (line 77) but NOT for non-Compare constants
# ------------------------------------------------------------------
tree = ast.parse(SOURCE)
func_ranges = _build_function_ranges(tree)
mutator = Mutator(func_ranges)
mutations = mutator.generate(tree)

# Find the Constant 0 -> 1 mutation on line 77
line77_mutations = [d for d, _ in mutations if "line 77" in d and "Constant 0 -> 1" in d]
print("Mutations on line 77 matching 'Constant 0 -> 1':")
for d in line77_mutations:
    print(f"  {d!r}")
assert line77_mutations, "Expected at least one Constant 0->1 mutation on line 77"
assert any("[BOUNDARY" in d for d in line77_mutations), (
    "Expected [BOUNDARY] tag in line-77 Constant mutation description"
)
print("PASS: [BOUNDARY] tag present in Constant 0->1 mutation at line 77")

# A rounding constant (round(..., 2)) should NOT get the tag
rounding_mutations = [d for d, _ in mutations if "line 33" in d and "[BOUNDARY" in d]
print(f"\nLine-33 mutations with [BOUNDARY] tag: {rounding_mutations}")
assert not rounding_mutations, "round(total, 2) constant should NOT get [BOUNDARY] tag"
print("PASS: round() constant at line 33 has no [BOUNDARY] tag")

# ------------------------------------------------------------------
# 2. _parse_boundary_tag
# ------------------------------------------------------------------
desc = "Constant 0 -> 1 (line 77, in apply_coupon) [BOUNDARY expr=discount op=Gt old=0 new=1]"
tag = _parse_boundary_tag(desc)
assert tag == {"expr": "discount", "op": "Gt", "old": 0, "new": 1}, f"Got {tag}"
print(f"\nPASS: _parse_boundary_tag: {tag}")

assert _parse_boundary_tag("Constant 2 -> 3 (line 33, in calculate_item_total)") is None
print("PASS: _parse_boundary_tag returns None when no [BOUNDARY] tag")

# ------------------------------------------------------------------
# 3. _is_nondiscriminating_value
# ------------------------------------------------------------------
# Gap is (0, 1] for discount > 0 -> > 1
assert     _is_nondiscriminating_value("Gt", 0, 1, 0.0),  "0.0 is NOT in gap — should be nondiscriminating"
assert     _is_nondiscriminating_value("Gt", 0, 1, 5.0),  "5.0 is NOT in gap — should be nondiscriminating"
assert not _is_nondiscriminating_value("Gt", 0, 1, 0.5),  "0.5 IS in gap — should be discriminating"
assert not _is_nondiscriminating_value("Gt", 0, 1, 1.0),  "1.0 IS in gap (inclusive) — should be discriminating"
assert not _is_nondiscriminating_value("Gt", 0, 1, 0.9),  "0.9 IS in gap — should be discriminating"
print("PASS: _is_nondiscriminating_value boundary tests")

# ------------------------------------------------------------------
# 4. _capture_operand_value: apply_coupon(50.00, None) -> discount=0.0 at line 77
# ------------------------------------------------------------------
ok, val = _capture_operand_value(SOURCE, "apply_coupon(50.00, None)", 77, "discount")
print(f"\n_capture_operand_value for apply_coupon(50.00, None) at line 77: ok={ok}, value={val!r}")
assert ok, f"Capture should succeed, got: {val}"
assert val == 0.0, f"discount should be 0.0 for None coupon, got {val!r}"
print(f"PASS: discount={val!r} at line 77 for apply_coupon(50.00, None)")

# Check this value is nondiscriminating for the Gt/0/1 boundary
is_nondiscrim = _is_nondiscriminating_value("Gt", 0, 1, val)
print(f"  _is_nondiscriminating_value('Gt', 0, 1, {val!r}) = {is_nondiscrim}")
assert is_nondiscrim, "discount=0.0 should be nondiscriminating for gap (0, 1]"
print("PASS: discount=0.0 correctly identified as nondiscriminating (outside gap)")

# ------------------------------------------------------------------
# 5. A SAVE10 on $9 gives discount=0.9, which IS in the gap (0, 1]
# ------------------------------------------------------------------
ok2, val2 = _capture_operand_value(SOURCE, "apply_coupon(9.00, 'SAVE10')", 77, "discount")
print(f"\n_capture_operand_value for apply_coupon(9.00, 'SAVE10') at line 77: ok={ok2}, value={val2!r}")
assert ok2
# discount = 9.0 * 0.10 = 0.9
assert abs(val2 - 0.9) < 1e-9, f"Expected 0.9, got {val2!r}"
is_discrim = not _is_nondiscriminating_value("Gt", 0, 1, val2)
print(f"  IS in discriminating interval (0, 1]: {is_discrim}")
assert is_discrim, "0.9 should be IN the discriminating interval"
print("PASS: apply_coupon(9.00, 'SAVE10') gives discount=0.9 — inside gap (0,1], valid discriminator")

print("\nAll boundary-value sanity checks passed.")
