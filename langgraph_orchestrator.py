"""LangGraph alternative for Sentinel's diagnosis/fix/mutation pipeline.

The provider, prompts, mutation engine, and verification helpers are shared with
Orchestrator.py. Retry and mutation routing are represented as graph edges.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

import Orchestrator as agents


class SentinelState(TypedDict, total=False):
    source_paths: list[str]
    source_file_paths: list[str]
    source_root: str
    test_path: str
    source_contents: dict[str, str]
    retry_count: int
    test_passed: bool
    test_output: str
    diagnosis: dict[str, Any]
    mutation_report: str | None
    mutation_summary: dict[str, int]
    survivor_descriptions: list[str]
    desc_by_id: dict[str, str]
    surviving_mutants_analysis: list[dict[str, Any]]
    result: dict[str, Any]
    final_verdict: str


def test_node(state: SentinelState) -> dict[str, Any]:
    attempt = state.get("retry_count", 0) + 1
    print(f"--- Sandbox run (attempt {attempt}) ---")
    passed, output = agents.run_tests(state["test_path"])
    if passed:
        print("All tests passed.\n")
    else:
        print("Tests failed. Invoking Diagnosis Agent...")
    return {"test_passed": passed, "test_output": output}


def diagnose_node(state: SentinelState) -> dict[str, Any]:
    labels = list(state["source_paths"])
    contents = {label: Path(path).read_text(encoding="utf-8")
                for label, path in zip(labels, state["source_file_paths"])}
    test_code = Path(state["test_path"]).read_text(encoding="utf-8")
    if len(contents) == 1:
        source_section = f"Source code:\n{next(iter(contents.values()))}"
    else:
        source_section = (
            "Source files (each file is labeled by its project-relative path):\n"
            + "\n\n".join(
                f"--- SOURCE FILE: {label} ---\n{content}\n--- END SOURCE FILE: {label} ---"
                for label, content in contents.items()
            )
        )
    diagnosis_input = (
        f"{source_section}\n\nTest file:\n{test_code}\n\n"
        f"Pytest output:\n{state['test_output']}"
    )
    previous = state.get("diagnosis")
    if previous is not None:
        diagnosis_input += (
            "\n\nPrevious diagnosis + fix attempt failed. Previous diagnosis:\n"
            + json.dumps(previous)
        )
    diagnosis = agents.call_agent(
        "Diagnosis", agents.DIAGNOSIS_SYSTEM_PROMPT, diagnosis_input
    )
    print(f"  Root cause: {diagnosis['root_cause']}")
    print(f"  Confidence: {diagnosis['confidence']}\n")
    return {"diagnosis": diagnosis, "source_contents": contents}


def fix_node(state: SentinelState) -> dict[str, Any]:
    print("Invoking Fix Agent...")
    labels = state["source_paths"]
    contents = state["source_contents"]
    primary_label = labels[0]
    source_code = contents[primary_label]
    if len(contents) == 1:
        source_section = f"Source code:\n{source_code}"
        fix_input = (
            f"Original source code:\n{source_code}\n\n"
            f"Diagnosis:\n{json.dumps(state['diagnosis'])}"
        )
    else:
        source_section = "Source files (each file is labeled by its project-relative path):\n" + "\n\n".join(
            f"--- SOURCE FILE: {label} ---\n{contents[label]}\n--- END SOURCE FILE: {label} ---"
            for label in labels
        )
        fix_input = f"Original {source_section}\n\nDiagnosis:\n{json.dumps(state['diagnosis'])}"

    fix_prompt = agents.FIX_SYSTEM_PROMPT
    if len(contents) > 1:
        fix_prompt += """

For this multi-file task, return one JSON object with:
- "change_summary": a concise description of the fix
- "patched_files": a list containing every changed file, each with "path" equal
  to its exact project-relative label shown above and "patched_code" equal to
  the complete replacement contents for that file
Do not omit a file whose interface or implementation must change for the tests
to pass. Do not return unified diffs. Do not invent paths.
"""
    fix = agents.call_agent("Fix", fix_prompt, fix_input)
    print(f"  Change: {fix['change_summary']}")
    if len(contents) == 1:
        patches = {primary_label: fix["patched_code"]}
    else:
        patches = {}
        for item in fix.get("patched_files", []):
            label = str(item.get("path", "")).replace("\\", "/")
            if label not in contents:
                raise ValueError(f"Fix Agent returned an unknown source path: {label!r}")
            patched = item.get("patched_code")
            if not isinstance(patched, str):
                raise ValueError(f"Fix Agent returned no patched_code for {label!r}")
            patches[label] = patched
        if not patches and isinstance(fix.get("patched_code"), str):
            print("  [i] Fix Agent returned the legacy single-file response; "
                  "applying it to the primary source before the full-suite retry.")
            patches[primary_label] = fix["patched_code"]
        if not patches:
            raise ValueError("Fix Agent returned no patched_files for a multi-file fix")

    for label, patched in patches.items():
        agents.print_diff(contents[label], patched, label)
        Path(state["source_file_paths"][labels.index(label)]).write_text(
            patched, encoding="utf-8"
        )
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "source_contents": {
            label: Path(path).read_text(encoding="utf-8")
            for label, path in zip(labels, state["source_file_paths"])
        },
    }


def retry_limit_node(state: SentinelState) -> dict[str, Any]:
    retry_count = state["retry_count"]
    result = {
        "final_verdict": "COULD_NOT_FIX",
        "retry_count": retry_count,
        "max_retries": agents.MAX_RETRIES,
    }
    print(f"Retry limit ({agents.MAX_RETRIES}) reached. Manual review needed.")
    print("\n=== Final Verdict: COULD_NOT_FIX ===")
    print(json.dumps(result, indent=2))
    agents.print_cost_summary()
    return {
        "final_verdict": "COULD_NOT_FIX",
        "retry_count": retry_count,
        "result": result,
    }


def _emit_result(result: dict[str, Any]) -> None:
    print(f"\n=== Final Verdict: {result['final_verdict']} ===")
    print(json.dumps(result, indent=2))
    agents.print_cost_summary()


def mutation_check_node(state: SentinelState) -> dict[str, Any]:
    print("--- Mutation Re-check ---")
    if len(state["source_file_paths"]) > 1:
        print("  [i] Mutation analysis remains scoped to the primary source file; "
              "multi-file mutation analysis is deferred to Phase 8.")
    report = agents.run_mutation_testing(
        state["source_file_paths"][0], state["test_path"]
    )
    if report is None:
        print("  [i] Mutation engine produced no output — skipping automated mutation re-check.")
        return {"mutation_report": None, "result": {}}

    total_match = re.search(r"Total mutants:\s*(\d+)", report)
    survived_match = re.search(r"Survived:\s*(\d+)", report)
    total = int(total_match.group(1)) if total_match else 0
    survived = int(survived_match.group(1)) if survived_match else 0
    summary = {"total_mutants": total, "killed": total - survived, "survived": survived}

    if total == 0:
        result = {
            "mutation_summary": {"total_mutants": 0, "killed": 0, "survived": 0},
            "surviving_mutants_analysis": [],
            "final_verdict": "ACCEPT_WITH_ADDED_TESTS",
            "confidence": "Low",
        }
        _emit_result(result)
        return {"mutation_report": report, "mutation_summary": summary, "result": result}
    if survived == 0:
        result = {
            "mutation_summary": summary,
            "surviving_mutants_analysis": [],
            "final_verdict": "ACCEPT_FIX",
            "confidence": "High",
        }
        _emit_result(result)
        return {"mutation_report": report, "mutation_summary": summary, "result": result}

    descriptions = agents.parse_survivors(report)
    if not descriptions:
        result = {
            "mutation_summary": summary,
            "surviving_mutants_analysis": [{
                "mutant_id": "N/A",
                "verdict": "real_coverage_gap",
                "discriminating_input": "",
                "reasoning": "Survivor descriptions could not be parsed; treating as unverified.",
                "recommended_test": "",
            }],
            "final_verdict": "ACCEPT_WITH_ADDED_TESTS",
            "confidence": "Low",
        }
        _emit_result(result)
        return {"mutation_report": report, "mutation_summary": summary, "result": result}

    return {
        "mutation_report": report,
        "mutation_summary": summary,
        "survivor_descriptions": descriptions,
        "desc_by_id": {str(index + 1): desc for index, desc in enumerate(descriptions)},
    }


def triage_node(state: SentinelState) -> dict[str, Any]:
    report = state["mutation_report"] or ""
    patched_source = Path(state["source_file_paths"][0]).read_text(encoding="utf-8")
    test_content = Path(state["test_path"]).read_text(encoding="utf-8")
    descriptions = state["survivor_descriptions"]
    numbered = "\n".join(f"{index + 1}. {desc}" for index, desc in enumerate(descriptions))
    triage_input = (
        f"Patched source code:\n{patched_source}\n\nTest file:\n{test_content}\n\n"
        f"Survived mutants (numbered — use these exact numbers as mutant_id):\n{numbered}"
    )
    triage = agents.call_agent(
        "Mutation Triage", agents.MUTATION_TRIAGE_SYSTEM_PROMPT,
        triage_input, max_tokens=3000,
    )
    if not isinstance(triage, dict):
        return {"triage_gaps": [
            {"mutant_id": str(index + 1), "verdict": "real_coverage_gap"}
            for index in range(len(state["survivor_descriptions"]))
        ], "triage_equivalents": []}
    verdicts = triage.get("verdicts", [])
    gaps = [entry for entry in verdicts if entry.get("verdict") == "real_coverage_gap"]
    equivalents = [entry for entry in verdicts if entry.get("verdict") == "equivalent_mutant"]

    def has_evidence(entry: dict[str, Any]) -> bool:
        input_value = str(entry.get("test_input", "")).strip().lower()
        return bool(
            input_value and input_value not in {"unreachable", "n/a", "na", "none", "-"}
            and str(entry.get("orig_output", "")).strip()
            and str(entry.get("mut_output", "")).strip()
        )

    verified_equivalents = [entry for entry in equivalents if has_evidence(entry)]
    unverified = [entry for entry in equivalents if not has_evidence(entry)]
    gaps.extend(unverified)
    return {"triage_gaps": gaps, "triage_equivalents": verified_equivalents}


def generate_tests_node(state: SentinelState) -> dict[str, Any]:
    descriptions = state["desc_by_id"]
    gaps = state.get("triage_gaps", [])
    equivalents = state.get("triage_equivalents", [])
    patched_source = Path(state["source_file_paths"][0]).read_text(encoding="utf-8")
    test_content = Path(state["test_path"]).read_text(encoding="utf-8")
    mutant_sources = agents.parse_mutant_sources(state["mutation_report"] or "")
    details_by_id: dict[str, dict[str, Any]] = {}
    batch_size = 3
    boundary_gaps = [entry for entry in gaps
                     if "[BOUNDARY" in descriptions.get(str(entry.get("mutant_id")), "")]
    other_gaps = [entry for entry in gaps if entry not in boundary_gaps]

    for entry in boundary_gaps:
        mutant_id = str(entry.get("mutant_id"))
        request = (
            f"Patched source code:\n{patched_source}\n\nTest file:\n{test_content}\n\n"
            f"For the mutant below, independently compute a concrete example and its true verdict "
            f"(use this exact mutant_id):\n- mutant_id {mutant_id}: "
            f"{descriptions.get(mutant_id, '')}"
        )
        response = agents.call_agent(
            f"Mutation Detail (boundary mutant {mutant_id})",
            agents.MUTATION_DETAIL_SYSTEM_PROMPT, request, max_tokens=3000,
        )
        for detail in response.get("details", []) if isinstance(response, dict) else []:
            details_by_id[str(detail.get("mutant_id"))] = detail

    for start in range(0, len(other_gaps), batch_size):
        batch = other_gaps[start:start + batch_size]
        summaries = "\n".join(
            f"- mutant_id {entry.get('mutant_id')}: "
            f"{descriptions.get(str(entry.get('mutant_id')), '')}"
            for entry in batch
        )
        request = (
            f"Patched source code:\n{patched_source}\n\nTest file:\n{test_content}\n\n"
            f"For each mutant below, independently determine its true verdict by computing a "
            f"concrete example. Use these exact mutant_id values:\n{summaries}"
        )
        response = agents.call_agent(
            f"Mutation Detail (batch {start // batch_size + 1})",
            agents.MUTATION_DETAIL_SYSTEM_PROMPT, request, max_tokens=3000,
        )
        for detail in response.get("details", []) if isinstance(response, dict) else []:
            details_by_id[str(detail.get("mutant_id"))] = detail

    analyses = []
    for gap in gaps:
        mutant_id = str(gap.get("mutant_id"))
        detail = details_by_id.get(mutant_id, {})
        analyses.append({
            "mutant_id": mutant_id,
            "diff": descriptions.get(mutant_id, ""),
            "verdict": detail.get("verdict", "real_coverage_gap"),
            "discriminating_input": detail.get("discriminating_input", ""),
            "reasoning": detail.get("reasoning", ""),
            "recommended_test": detail.get("recommended_test", ""),
            "_claimed_orig_output": detail.get("orig_output", ""),
            "_claimed_mut_output": detail.get("mut_output", ""),
        })
    for equivalent in equivalents:
        mutant_id = str(equivalent.get("mutant_id"))
        test_input = equivalent.get("test_input", "")
        analyses.append({
            "mutant_id": mutant_id,
            "diff": descriptions.get(mutant_id, ""),
            "verdict": "equivalent_mutant",
            "discriminating_input": test_input,
            "reasoning": (
                f"Checked {test_input}: original={equivalent.get('orig_output')}, "
                f"mutant={equivalent.get('mut_output')} (identical)"
            ),
            "recommended_test": "",
            "_claimed_orig_output": equivalent.get("orig_output", ""),
            "_claimed_mut_output": equivalent.get("mut_output", ""),
        })

    analyses = agents.verify_equivalent_claims(
        patched_source, analyses, mutant_sources=mutant_sources,
        desc_by_id=descriptions,
    )
    return {"surviving_mutants_analysis": analyses}


def finalize_node(state: SentinelState) -> dict[str, Any]:
    analyses = state.get("surviving_mutants_analysis", [])
    for entry in analyses:
        for key in (
            "_claimed_orig_output", "_claimed_mut_output", "_mutant_lineno",
            "_executed_lines_sample", "_boundary", "_operand_val",
            "_contaminated", "_own_tag",
        ):
            entry.pop(key, None)
        if entry.get("verdict") in {
            "unverified_unreachable", "unverified_nondiscriminating_value"
        }:
            entry["verdict"] = "real_coverage_gap"
            entry["reasoning"] = (
                entry.get("reasoning", "")
                + " Verification remained inconclusive; conservatively treating this as a real coverage gap."
            )
    gaps = [entry for entry in analyses if entry.get("verdict") == "real_coverage_gap"]
    verdict = "ACCEPT_WITH_ADDED_TESTS" if gaps else "ACCEPT_FIX"
    result = {
        "mutation_summary": state["mutation_summary"],
        "surviving_mutants_analysis": analyses,
        "final_verdict": verdict,
        "confidence": "High",
    }
    print(f"\n=== Final Verdict: {verdict} ===")
    print(json.dumps(result, indent=2))
    agents.print_cost_summary()
    return {"result": result, "final_verdict": verdict}


def route_after_test(state: SentinelState) -> str:
    return "mutation_check" if state["test_passed"] else "diagnose"


def route_after_fix(state: SentinelState) -> str:
    # Deliberately mirrors `while retry_count <= MAX_RETRIES`: this permits
    # Fix attempts 1 through 4, then exits without another sandbox run.
    return "retry_limit" if state["retry_count"] > agents.MAX_RETRIES else "test"


def route_after_mutation_check(state: SentinelState) -> str:
    if state.get("mutation_report") is None or state.get("result"):
        return "end_result"
    return "triage"


def route_after_triage(state: SentinelState) -> str:
    return "generate_tests" if state.get("triage_gaps") else "finalize"


def build_graph():
    graph = StateGraph(SentinelState)
    graph.add_node("test", test_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("fix", fix_node)
    graph.add_node("retry_limit", retry_limit_node)
    graph.add_node("mutation_check", mutation_check_node)
    graph.add_node("triage", triage_node)
    graph.add_node("generate_tests", generate_tests_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "test")
    graph.add_conditional_edges("test", route_after_test, {
        "diagnose": "diagnose",
        "mutation_check": "mutation_check",
    })
    graph.add_edge("diagnose", "fix")
    graph.add_conditional_edges("fix", route_after_fix, {
        "test": "test",
        "retry_limit": "retry_limit",
    })
    graph.add_edge("retry_limit", END)
    graph.add_conditional_edges("mutation_check", route_after_mutation_check, {
        "triage": "triage",
        "end_result": END,
    })
    graph.add_conditional_edges("triage", route_after_triage, {
        "generate_tests": "generate_tests",
        "finalize": "finalize",
    })
    graph.add_edge("generate_tests", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run(source_path: str, tests_path: str, source_paths: list[str] | None = None,
        source_root: str | None = None) -> dict[str, Any]:
    source_file_paths = [source_path, *(source_paths or [])]
    root = Path(source_root).resolve() if source_root else Path(source_path).resolve().parent
    labels: list[str] = []
    for path in source_file_paths:
        resolved = Path(path).resolve()
        try:
            label = resolved.relative_to(root).as_posix()
        except ValueError:
            label = resolved.name
        if label in labels:
            raise ValueError(f"Duplicate source label: {label}")
        labels.append(label)

    print("=== LangGraph Multi-Agent Debugging System ===")
    for label, path in zip(labels, source_file_paths):
        print(f"Source [{label}]: {path}")
    print(f"Tests: {tests_path}\n")
    return build_graph().invoke({
        "source_paths": labels,
        "source_file_paths": source_file_paths,
        "source_root": str(root),
        "test_path": tests_path,
        "retry_count": 0,
        "diagnosis": None,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph Sentinel orchestrator")
    parser.add_argument("--source", required=True, help="Primary source file")
    parser.add_argument("--source-file", action="append", default=[],
                        help="Additional source file; may be repeated")
    parser.add_argument("--source-root", help="Root for project-relative labels")
    parser.add_argument("--tests", required=True, help="Test file")
    args = parser.parse_args()
    run(args.source, args.tests, args.source_file, args.source_root)


if __name__ == "__main__":
    main()