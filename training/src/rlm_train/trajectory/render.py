"""Render a canonical annotated rollout as a readable question/answer recursion tree."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from rlm_train.trajectory.replay import load_annotated_rollout
from rlm_train.trajectory.schema import AnnotatedRollout

SCORE_LABELS = {
    "information_significance": ("significance", "significance"),
    "novelty": ("novelty", "novelty"),
    "uncertainty_reduction": ("uncertainty", "uncertainty_reduction"),
    "evidence_quality": ("evidence", "evidence_quality"),
}


def shorten(text: object, limit: int) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 1)] + "\u2026"


def summarize_value(value: Any, limit: int) -> str:
    return shorten(json.dumps(value, sort_keys=True, ensure_ascii=False), limit)


def format_scores(content: dict[str, Any]) -> str:
    categories = content.get("categories") or {}
    parts: list[str] = []
    for numeric_key, (short, category_key) in SCORE_LABELS.items():
        if numeric_key not in content:
            continue
        numeric = content[numeric_key]
        label = categories.get(category_key)
        try:
            rendered = f"{float(numeric):.2f}"
        except (TypeError, ValueError):
            rendered = str(numeric)
        parts.append(f"{short}={label}({rendered})" if label is not None else f"{short}={rendered}")
    for flag in ("redundant", "misleading"):
        if content.get(flag):
            parts.append(flag)
    return " ".join(parts)


def summarize_assessment(assessment: dict[str, Any]) -> str:
    content = assessment.get("content") or {}
    header = f"judge[{content.get('judge_mode', '?')}] via {assessment.get('provider', '?')}"
    scores = format_scores(content)
    return f"{header}: {scores}" if scores else header


def index_assessments(
    assessments: Sequence[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_edge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assessment in assessments:
        for edge_id in assessment.get("focal_edge_ids", ()):
            by_edge[edge_id].append(assessment)
        for node_id in assessment.get("focal_node_ids", ()):
            by_node[node_id].append(assessment)
    return by_edge, by_node


def assessments_for(
    edge_id: str,
    node_id: str,
    by_edge: dict[str, list[dict[str, Any]]],
    by_node: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    seen: set[str | None] = set()
    ordered: list[dict[str, Any]] = []
    for assessment in (*by_edge.get(edge_id, ()), *by_node.get(node_id, ())):
        key = assessment.get("assessment_id")
        if key in seen:
            continue
        seen.add(key)
        ordered.append(assessment)
    return ordered


def render_recursion_tree(rollout: AnnotatedRollout, *, max_text_chars: int = 200) -> str:
    """Return an indented question/answer tree annotated with attached judge scores."""
    nodes = {node.node_id: node for node in rollout.execution.nodes}
    edges_by_parent: dict[str, list[Any]] = defaultdict(list)
    for edge in rollout.execution.edges:
        edges_by_parent[edge.parent_id].append(edge)
    by_edge, by_node = index_assessments(rollout.feedback.judge_assessments)

    lines = [
        f"Rollout {rollout.rollout_id}  (mode={rollout.mode})",
        f"Task: {summarize_value(rollout.task.public, max_text_chars)}",
    ]
    final_answer = rollout.result.get("final_answer")
    if final_answer is not None:
        lines.append(f"Final answer: {shorten(final_answer, max_text_chars)}")
    if rollout.feedback.overall_assessment:
        lines.append(f"Overall: {summarize_assessment(rollout.feedback.overall_assessment)}")
    lines.append("")

    root = nodes[rollout.execution.root_node_id]
    lines.append(f"{root.node_id}  \u00b7  {root.role.value}  \u00b7  depth {root.depth}")

    def walk(node_id: str, prefix: str) -> None:
        child_edges = edges_by_parent.get(node_id, [])
        for index, edge in enumerate(child_edges):
            is_last = index == len(child_edges) - 1
            connector = "\u2514\u2500 " if is_last else "\u251c\u2500 "
            branch_prefix = prefix + ("   " if is_last else "\u2502  ")
            lines.append(
                f"{prefix}{connector}Q: {shorten(edge.question, max_text_chars)}"
                f"   ({edge.kind} \u2192 {edge.child_id})"
            )
            child = nodes.get(edge.child_id)
            if child is None:
                continue
            status = "  [FAILED]" if child.failed else ""
            if child.result is not None:
                lines.append(f"{branch_prefix}A: {shorten(child.result, max_text_chars)}{status}")
            elif status:
                lines.append(f"{branch_prefix}A:{status.strip()}")
            for assessment in assessments_for(edge.edge_id, edge.child_id, by_edge, by_node):
                lines.append(f"{branch_prefix}{summarize_assessment(assessment)}")
                diagnostic = (assessment.get("content") or {}).get("diagnostic")
                if diagnostic:
                    lines.append(f"{branch_prefix}   \u00b7 {shorten(diagnostic, max_text_chars)}")
            walk(edge.child_id, branch_prefix)

    walk(root.node_id, "")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_path")
    parser.add_argument("--max-text-chars", type=int, default=200)
    args = parser.parse_args(argv)
    rollout = load_annotated_rollout(args.rollout_path)
    print(render_recursion_tree(rollout, max_text_chars=args.max_text_chars))
    return 0


__all__ = ["render_recursion_tree", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
