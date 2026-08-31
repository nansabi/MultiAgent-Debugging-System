"""
Lightweight Mutation Testing Engine (pure Python, no external tools)
=====================================================================
Cross-platform replacement for mutmut. Works natively on Windows.

It parses the source file into an AST, generates one mutant per meaningful
operator/constant change, writes each mutant to disk temporarily, re-runs
pytest, and records whether the mutant was "killed" (tests failed, good)
or "survived" (tests still passed, meaning that logic isn't covered).

Each mutant description now includes the source line number and enclosing
function name (e.g. "Constant False -> True (line 41, in calculate_shipping)")
instead of just the bare change (e.g. "Constant False -> True"). Bare
descriptions gave downstream LLM analysis nothing to anchor to when a file
has multiple similar constants, which led to plausible-sounding but wrong
location attribution in practice — the model would confidently explain a
mutation as affecting one line when it actually affected a different one
with a similar description. Line number + function name closes that gap.

USAGE:
    python simple_mutation_test.py buggy_code.py test_buggy_code.py
"""

import ast
import copy
import subprocess
import sys
from pathlib import Path


def _build_function_ranges(tree: ast.AST) -> list[tuple[str, int, int]]:
    """
    Return (function_name, start_line, end_line) for every function/method
    definition in the tree, including nested ones. Used to look up which
    function contains a given line number.
    """
    ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", None) or start
            ranges.append((node.name, start, end))
    return ranges


def _enclosing_function(lineno: int, function_ranges: list[tuple[str, int, int]]) -> str | None:
    """
    Find the innermost function containing the given line number. Innermost
    is determined by choosing the match with the smallest line range, since
    nested functions have ranges fully contained within their parent's.
    """
    candidates = [
        (name, start, end) for name, start, end in function_ranges
        if start <= lineno <= end
    ]
    if not candidates:
        return None
    # Smallest range = innermost / most specific function.
    best = min(candidates, key=lambda c: c[2] - c[1])
    return best[0]


def _locate(node: ast.AST, function_ranges: list[tuple[str, int, int]]) -> str:
    """Build a human-readable location suffix like ' (line 41, in calculate_shipping)'."""
    lineno = getattr(node, "lineno", None)
    if lineno is None:
        return ""
    func_name = _enclosing_function(lineno, function_ranges)
    if func_name:
        return f" (line {lineno}, in {func_name})"
    return f" (line {lineno})"


class Mutator(ast.NodeTransformer):
    """Walks the AST and yields one mutated copy per meaningful change."""

    BOOL_OP_SWAP = {ast.And: ast.Or, ast.Or: ast.And}
    COMPARE_SWAP = {
        ast.Lt: ast.GtE, ast.GtE: ast.Lt,
        ast.Gt: ast.LtE, ast.LtE: ast.Gt,
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    }
    ARITH_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}

    def __init__(self, function_ranges: list[tuple[str, int, int]]):
        self.mutations = []  # list of (description, mutated_tree)
        self.function_ranges = function_ranges

    def generate(self, tree):
        # Build a parent map so Constant nodes can look up whether they are a
        # direct operand of a Compare node. ast.walk() gives no parent refs.
        parent_map: dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[id(child)] = parent

        # BoolOp (and/or)
        for node in ast.walk(tree):
            loc = _locate(node, self.function_ranges)

            if isinstance(node, ast.BoolOp) and type(node.op) in self.BOOL_OP_SWAP:
                mutant = copy.deepcopy(tree)
                target = self._find_matching(mutant, tree, node)
                target.op = self.BOOL_OP_SWAP[type(node.op)]()
                self.mutations.append(
                    (f"BoolOp {type(node.op).__name__} -> {type(target.op).__name__}{loc}", mutant)
                )

            elif isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in self.COMPARE_SWAP:
                mutant = copy.deepcopy(tree)
                target = self._find_matching(mutant, tree, node)
                target.ops[0] = self.COMPARE_SWAP[type(node.ops[0])]()
                self.mutations.append(
                    (f"Compare {type(node.ops[0]).__name__} -> {type(target.ops[0]).__name__}{loc}", mutant)
                )

            elif isinstance(node, ast.BinOp) and type(node.op) in self.ARITH_SWAP:
                mutant = copy.deepcopy(tree)
                target = self._find_matching(mutant, tree, node)
                target.op = self.ARITH_SWAP[type(node.op)]()
                self.mutations.append(
                    (f"BinOp {type(node.op).__name__} -> {type(target.op).__name__}{loc}", mutant)
                )

            elif isinstance(node, ast.Attribute) and node.attr in ("isupper", "islower"):
                mutant = copy.deepcopy(tree)
                target = self._find_matching(mutant, tree, node)
                target.attr = "islower" if node.attr == "isupper" else "isupper"
                self.mutations.append(
                    (f"Attribute {node.attr} -> {target.attr}{loc}", mutant)
                )

            elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
                mutant = copy.deepcopy(tree)
                target = self._find_matching(mutant, tree, node)
                target.value = not node.value
                self.mutations.append(
                    (f"Constant {node.value} -> {target.value}{loc}", mutant)
                )

            elif isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                mutant = copy.deepcopy(tree)
                target = self._find_matching(mutant, tree, node)
                target.value = node.value + 1
                # Check whether this constant is a direct operand of a Compare node.
                # If so, append a [BOUNDARY ...] tag so the orchestrator can verify
                # that the discriminating_input actually exercises the distinguishing
                # interval (old, new] rather than a value where both variants agree.
                boundary_tag = self._boundary_tag(node, parent_map)
                self.mutations.append(
                    (f"Constant {node.value} -> {target.value}{loc}{boundary_tag}", mutant)
                )

        return self.mutations

    def _boundary_tag(
        self,
        const_node: ast.Constant,
        parent_map: dict[int, ast.AST],
    ) -> str:
        """
        If `const_node` is a direct operand of a single-comparator Compare node
        (i.e. the constant is either the left-hand operand or one of the comparators'
        right-hand operands), return a machine-parseable suffix like:

            " [BOUNDARY expr=discount op=Gt old=0 new=1]"

        where `expr` is the OTHER side of the comparison (the non-constant operand,
        as source text via ast.unparse), `op` is the comparator class name, `old` is
        the current constant value, and `new` is old+1.

        Returns "" if the constant is not directly inside a Compare, or if the
        Compare has multiple comparators (chained comparisons like `0 < x < 1` are
        not handled — leave those for a future extension).

        Scope: Constant-in-Compare only. BinOp and Compare-operator-swap mutations
        do NOT get this tag (they are documented as a known limitation).
        """
        parent = parent_map.get(id(const_node))
        if not isinstance(parent, ast.Compare):
            return ""
        if len(parent.ops) != 1:
            # Chained comparison — skip.
            return ""
        op_name = type(parent.ops[0]).__name__
        old_val = const_node.value
        new_val = old_val + 1

        # Determine the "other" side: if the constant is the left (parent.left),
        # the expression being compared is one of parent.comparators; otherwise
        # the constant is a comparator operand and parent.left is the expression.
        if const_node is parent.left:
            other_nodes = parent.comparators
        else:
            # Constant is in parent.comparators — other side is parent.left.
            other_nodes = [parent.left]

        if len(other_nodes) != 1:
            return ""

        try:
            expr_text = ast.unparse(other_nodes[0])
        except Exception:
            return ""

        # Sanitize expr_text: strip whitespace and replace spaces with underscores
        # so the tag remains a single parseable token.  We use repr only for safety;
        # typically these are simple names like "discount" or "subtotal".
        expr_text = expr_text.strip().replace(" ", "_")
        return f" [BOUNDARY expr={expr_text} op={op_name} old={old_val} new={new_val}]"

    def _find_matching(self, mutant_tree, original_tree, original_node):
        """Find the node in mutant_tree at the same position as original_node in original_tree."""
        orig_nodes = list(ast.walk(original_tree))
        idx = orig_nodes.index(original_node)
        mutant_nodes = list(ast.walk(mutant_tree))
        return mutant_nodes[idx]


def run_pytest(test_file: str) -> bool:
    """Return True if all tests pass."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-q"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def main():
    if len(sys.argv) != 3:
        print("Usage: python simple_mutation_test.py <source_file> <test_file>")
        sys.exit(1)

    source_path = Path(sys.argv[1])
    test_path = Path(sys.argv[2])
    original_source = source_path.read_text(encoding="utf-8")

    tree = ast.parse(original_source)
    function_ranges = _build_function_ranges(tree)
    mutator = Mutator(function_ranges)
    mutations = mutator.generate(tree)

    if not mutations:
        print("No mutable operators/constants found in this file.")
        return

    print(f"Generated {len(mutations)} mutants. Running tests against each...\n")

    killed, survived = 0, 0
    # survivors is a list of (survivor-order rank, description, mutated_source_text).
    # rank is a SEPARATE counter that only increments for survivors — it matches the
    # 1-based position in the "Surviving mutants" summary list, which is exactly what
    # parse_survivors() in Orchestrator.py uses as mutant_id.  Do NOT use the outer
    # loop's `i` here; `i` is the generation-order index (1..N across ALL mutants,
    # killed + survived) and would produce a different numbering.
    survivors = []
    survivor_number = 0

    for i, (description, mutant_tree) in enumerate(mutations, 1):
        mutated_source = ast.unparse(mutant_tree)
        source_path.write_text(mutated_source, encoding="utf-8")

        passed = run_pytest(str(test_path))

        if passed:
            survived += 1
            survivor_number += 1
            survivors.append((survivor_number, description, mutated_source))
            print(f"  [{i}/{len(mutations)}] SURVIVED — {description}")
        else:
            killed += 1
            print(f"  [{i}/{len(mutations)}] killed   — {description}")

    # restore original
    source_path.write_text(original_source, encoding="utf-8")

    print(f"\n=== Mutation Testing Summary ===")
    print(f"Total mutants: {len(mutations)}")
    print(f"Killed: {killed}")
    print(f"Survived: {survived}")
    if survivors:
        print("\nSurviving mutants (tests did NOT catch these):")
        for _, desc, _ in survivors:
            print(f"  - {desc}")

        # Emit mutant source texts in a structured section so the orchestrator can
        # execute discriminating_inputs against the actual mutated code (not just the
        # original) to independently verify the detail agent's mut_output claims.
        # Format: one block per survivor, delimited by sentinel lines.
        # `rank` here is the survivor-order counter (1..len(survivors)), matching the
        # position-based mutant_id scheme used by parse_survivors() in Orchestrator.py.
        print("\n=== Survivor Mutant Sources ===")
        for rank, _, mutated_src in survivors:
            print(f"--- MUTANT {rank} SOURCE BEGIN ---")
            print(mutated_src)
            print(f"--- MUTANT {rank} SOURCE END ---")


if __name__ == "__main__":
    main()