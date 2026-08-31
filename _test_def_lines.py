import os, sys, types
os.environ.setdefault("GROQ_API_KEY", "test_dummy_key")
from pathlib import Path

SOURCE = Path("buggy_code.py").read_text(encoding="utf-8")

# Check what lines are executed during exec() itself (module-level, def statements)
executed_exec = set()
def tracer_exec(frame, event, arg):
    if frame.f_code.co_filename == "<verify_source>" and event == "line":
        executed_exec.add(frame.f_lineno)
    return tracer_exec

namespace = {}
code = compile(SOURCE, "<verify_source>", "exec")
prev = sys.gettrace()
sys.settrace(tracer_exec)
sys._getframe().f_trace = tracer_exec
exec(code, namespace)
sys.settrace(prev)
print("Lines executed during exec() of buggy_code.py:", sorted(executed_exec))
print("Line 94 covered during exec:", 94 in executed_exec)
print("Line 42 (def calculate_shipping) covered during exec:", 42 in executed_exec)
print("Line 62 (def apply_coupon) covered during exec:", 62 in executed_exec)
print()

# Now check what AST says about line 94
import ast
tree = ast.parse(SOURCE)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "process_order":
        print(f"process_order: FunctionDef lineno={node.lineno}")
        # Check if any default arg nodes have lineno 94
        for d in node.args.defaults:
            print(f"  default arg: {ast.dump(d)}, lineno={getattr(d,'lineno',None)}")
