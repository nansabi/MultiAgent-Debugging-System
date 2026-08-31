import os
os.environ.setdefault("GROQ_API_KEY", "test_dummy_key")
from pathlib import Path
from Orchestrator import _get_executed_lines

src = Path("buggy_code.py").read_text(encoding="utf-8")

# patched_source in main() is read directly from the file on disk,
# same text as buggy_code.py here. Line numbers from diff descriptions
# come from the original AST — so they match.

lines_flat5 = _get_executed_lines(src, "apply_coupon(25.00, 'FLAT5')")
print("apply_coupon(25.00, FLAT5) lines:", sorted(lines_flat5))
orig = src.splitlines()
print("Line 74 in original file:", repr(orig[73]))
print("74 in traced set:", 74 in lines_flat5)

lines_po = _get_executed_lines(src, "process_order([{'name':'mug','price':10.0,'qty':1}])")
print("\nprocess_order([...]) lines:", sorted(lines_po))
print("Line 94 in original file:", repr(orig[93]))
print("94 in traced set:", 94 in lines_po)
