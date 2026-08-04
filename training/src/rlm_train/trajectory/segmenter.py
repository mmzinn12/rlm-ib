"""Segment depth-1 REPL responses into exclusive character-level decision spans.

Purpose:
    Identify policy-generated routing, call construction, child reasoning, aggregation,
    and final-answer regions before tokenization.
Implementation:
    REPL blocks are parsed with Python's AST to locate helper calls, control-flow
    headers, and ``answer`` assignments. Overlaps are split using fixed precedence;
    parent prose outside new code becomes aggregation after child results are available.
Inputs:
    Raw root or child response strings with response-relative REPL block offsets.
Outputs:
    Sorted, non-overlapping ``DecisionSpan`` objects over the original response.
Example:
    ``spans = RLMResponseSegmenter().segment_root_response(response)``
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass

from rlm.core.trajectory import CallItemSpan, DecisionKind, DecisionSpan
from rlm.utils.parsing import ParsedCodeBlock, find_code_blocks_with_spans

_CALL_HELPERS = frozenset({"llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched"})
_PRECEDENCE = {
    DecisionKind.CALL: 0,
    DecisionKind.FINAL: 1,
    DecisionKind.ROUTE: 2,
    DecisionKind.AGGREGATION: 3,
    DecisionKind.NODE: 4,
}


@dataclass(frozen=True)
class RootSegmentation:
    """Return broad decision spans and narrow, question-addressable call items.

    Attributes:
        spans: Sorted, mutually exclusive decision spans over the root response.
        call_item_spans: Exact spans for statically addressable question expressions.
        question_item_count: Total number of syntactic scalar or batched questions.
        unaddressable_question_item_count: Questions whose values are constructed
            dynamically and therefore cannot be mapped to exact source characters.

    Example:
        ``result = RLMResponseSegmenter().segment_root(response)``
    """

    spans: list[DecisionSpan]
    call_item_spans: list[CallItemSpan]
    question_item_count: int
    unaddressable_question_item_count: int

    @property
    def addressable_question_item_count(self) -> int:
        """Return the number of questions with exact response-relative spans."""
        return len(self.call_item_spans)


class RLMResponseSegmenter:
    """Extract deterministic spans without changing or retokenizing policy output.

    The class is stateless and may be shared across concurrent rollouts.
    """

    def segment_root_response(
        self, response: str, *, has_child_results: bool = False
    ) -> list[DecisionSpan]:
        """Segment one root-policy continuation into exclusive training components.

        Args:
            response: Exact generated root response, including any fenced REPL code.
            has_child_results: Whether prior child results are visible to this root turn;
                when true, non-code prose not otherwise claimed becomes aggregation.

        Returns:
            Sorted, non-overlapping response-relative spans. Invalid Python blocks are
            left unclassified rather than receiving inaccurate AST-derived labels.

        Example:
            ``segmenter.segment_root_response(text, has_child_results=True)``
        """
        return self.segment_root(response, has_child_results=has_child_results).spans

    def segment_root(self, response: str, *, has_child_results: bool = False) -> RootSegmentation:
        """Segment a root continuation and extract literal question-item spans.

        Literal string arguments to scalar calls and literal string elements in lists
        or tuples passed to batched calls are addressable. A simple name bound earlier
        in the same code block to a literal list/tuple is also resolved. Dynamic values
        are counted as unaddressable and never receive an approximate span.

        Args:
            response: Exact root-policy response, including fenced REPL code.
            has_child_results: Whether prior child outputs are visible; when true,
                otherwise-unclaimed prose is classified as aggregation.

        Returns:
            A ``RootSegmentation`` containing exclusive component spans, exact
            question-item spans, and addressability counts.

        Example:
            ``result = segmenter.segment_root(text, has_child_results=True)``
        """
        spans: list[DecisionSpan] = []
        call_item_spans: list[CallItemSpan] = []
        question_item_count = 0
        unaddressable_question_item_count = 0
        code_blocks = find_code_blocks_with_spans(response)
        call_order_offset = 0
        for block in code_blocks:
            (
                block_spans,
                block_item_spans,
                block_call_count,
                block_question_count,
                block_unaddressable_count,
            ) = self._segment_code_block(block, call_order_offset=call_order_offset)
            spans.extend(block_spans)
            call_item_spans.extend(block_item_spans)
            call_order_offset += block_call_count
            question_item_count += block_question_count
            unaddressable_question_item_count += block_unaddressable_count
        spans = resolve_exclusive_spans(spans)
        if has_child_results:
            excluded = [(block.fence_start, block.fence_end) for block in code_blocks]
            spans.extend(
                _unclaimed_non_whitespace(
                    response,
                    spans,
                    DecisionKind.AGGREGATION,
                    excluded=excluded,
                )
            )
        return RootSegmentation(
            spans=sorted(resolve_exclusive_spans(spans), key=lambda span: (span.start, span.end)),
            call_item_spans=sorted(
                call_item_spans,
                key=lambda span: (
                    span.call_order,
                    -1 if span.batch_index is None else span.batch_index,
                ),
            ),
            question_item_count=question_item_count,
            unaddressable_question_item_count=unaddressable_question_item_count,
        )

    def segment_child_response(self, response: str) -> list[DecisionSpan]:
        """Classify a non-empty child continuation as node-reasoning output.

        Args:
            response: Exact child-model response.

        Returns:
            One ``NODE`` span covering trimmed non-whitespace content, or an empty list
            for an empty/whitespace-only response.
        """
        if not response or not response.strip():
            return []
        start = len(response) - len(response.lstrip())
        end = len(response.rstrip())
        return [DecisionSpan(kind=DecisionKind.NODE, start=start, end=end)]

    def _segment_code_block(
        self, block: ParsedCodeBlock, *, call_order_offset: int
    ) -> tuple[list[DecisionSpan], list[CallItemSpan], int, int, int]:
        """Extract candidate call, routing, and final spans from one Python AST.

        Args:
            block: Parsed REPL code and its offsets in the full model response.

        Returns:
            Broad spans, call-item spans, supported-call count, syntactic question-item
            count, and unaddressable-item count. Invalid Python returns empty values.
        """
        try:
            tree = ast.parse(block.code)
        except SyntaxError:
            return [], [], 0, 0, 0

        spans: list[DecisionSpan] = []
        call_item_spans: list[CallItemSpan] = []
        question_item_count = 0
        unaddressable_question_item_count = 0
        calls = sorted(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and _call_name(node.func) in _CALL_HELPERS
            ),
            key=lambda node: (node.lineno, node.col_offset),
        )
        assignments = _name_assignments(tree)
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for local_call_order, node in enumerate(calls):
            helper = _call_name(node.func)
            call_order = call_order_offset + local_call_order
            extracted, item_count, unaddressable_count = _extract_call_items(
                block,
                node,
                helper=helper,
                call_order=call_order,
                assignments=assignments,
                addressable=_has_static_execution_count(node, parents),
            )
            question_item_count += item_count
            unaddressable_question_item_count += unaddressable_count
            call_item_spans.extend(extracted)
            start, end = _absolute_node_span(block, node)
            spans.append(
                DecisionSpan(
                    kind=DecisionKind.CALL,
                    start=start,
                    end=end,
                    metadata={
                        "helper": helper,
                        "call_order": call_order,
                        "question_item_count": item_count,
                        "addressable_question_item_count": len(extracted),
                        "unaddressable_question_item_count": unaddressable_count,
                    },
                )
            )

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node.func) in _CALL_HELPERS:
                continue
            if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While)) and _contains_call(
                node
            ):
                start, end = _absolute_header_span(block, node)
                if end > start:
                    spans.append(DecisionSpan(kind=DecisionKind.ROUTE, start=start, end=end))
            elif _is_final_assignment(node):
                start, end = _absolute_node_span(block, node)
                spans.append(DecisionSpan(kind=DecisionKind.FINAL, start=start, end=end))
        return (
            spans,
            call_item_spans,
            len(calls),
            question_item_count,
            unaddressable_question_item_count,
        )


def _name_assignments(tree: ast.AST) -> dict[str, list[tuple[tuple[int, int], ast.AST]]]:
    """Index top-level assignments for conservative literal-list name resolution."""
    assignments: dict[str, list[tuple[tuple[int, int], ast.AST]]] = {}
    nodes = tree.body if isinstance(tree, ast.Module) else []
    for node in nodes:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                assignments.setdefault(target.id, []).append(
                    ((node.lineno, node.col_offset), value)
                )
    for values in assignments.values():
        values.sort(key=lambda item: item[0])
    return assignments


def _extract_call_items(
    block: ParsedCodeBlock,
    call: ast.Call,
    *,
    helper: str | None,
    call_order: int,
    assignments: dict[str, list[tuple[tuple[int, int], ast.AST]]],
    addressable: bool,
) -> tuple[list[CallItemSpan], int, int]:
    """Extract exact literal question spans without guessing dynamic token ownership."""
    if not call.args:
        return [], 0, 0
    argument = _resolve_literal_argument(call.args[0], call, assignments)
    if helper in {"llm_query_batched", "rlm_query_batched"}:
        if not isinstance(argument, (ast.List, ast.Tuple)):
            return [], 1, 1
        if not addressable:
            return [], len(argument.elts), len(argument.elts)
        spans: list[CallItemSpan] = []
        unaddressable = 0
        for batch_index, element in enumerate(argument.elts):
            if not _is_literal_string(element):
                unaddressable += 1
                continue
            start, end = _absolute_node_span(block, element)
            spans.append(
                CallItemSpan(
                    call_order=call_order,
                    batch_index=batch_index,
                    start=start,
                    end=end,
                )
            )
        return spans, len(argument.elts), unaddressable
    if _is_literal_string(argument) and addressable:
        start, end = _absolute_node_span(block, argument)
        return [CallItemSpan(call_order, None, start, end)], 1, 0
    return [], 1, 1


def _has_static_execution_count(call: ast.Call, parents: dict[ast.AST, ast.AST]) -> bool:
    """Reject item addressing when control flow can change runtime call ordering."""
    dynamic_ancestors = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.BoolOp,
        ast.IfExp,
        ast.Match,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    if any(
        child is not call
        and isinstance(child, ast.Call)
        and _call_name(child.func) in _CALL_HELPERS
        for child in ast.walk(call)
    ):
        return False
    current: ast.AST = call
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.Call) and _call_name(current.func) in _CALL_HELPERS:
            return False
        if isinstance(current, dynamic_ancestors):
            return False
    return True


def _resolve_literal_argument(
    argument: ast.AST,
    call: ast.Call,
    assignments: dict[str, list[tuple[tuple[int, int], ast.AST]]],
) -> ast.AST:
    """Resolve a name only when its latest preceding assignment is a literal container."""
    if not isinstance(argument, ast.Name):
        return argument
    call_position = (call.lineno, call.col_offset)
    preceding = [
        value for position, value in assignments.get(argument.id, []) if position < call_position
    ]
    if not preceding:
        return argument
    candidate = preceding[-1]
    return candidate if isinstance(candidate, (ast.List, ast.Tuple)) else argument


def _is_literal_string(node: ast.AST) -> bool:
    """Return whether a node is a static string literal with exact source offsets."""
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def resolve_exclusive_spans(spans: Iterable[DecisionSpan]) -> list[DecisionSpan]:
    """Resolve overlapping candidate spans using fixed component precedence.

    Higher-priority spans retain their full intervals. Lower-priority spans are split
    into any non-overlapping fragments that remain.

    Args:
        spans: Candidate spans in one response coordinate system.

    Returns:
        Sorted spans in which no pair overlaps.

    Example:
        ``exclusive = resolve_exclusive_spans(candidate_spans)``
    """
    accepted: list[DecisionSpan] = []
    ordered = sorted(
        spans,
        key=lambda span: (_PRECEDENCE.get(span.kind, 99), span.start, -(span.end - span.start)),
    )
    for span in ordered:
        fragments = [(span.start, span.end)]
        for claimed in accepted:
            fragments = _subtract(fragments, claimed.start, claimed.end)
        for start, end in fragments:
            if end > start:
                accepted.append(
                    DecisionSpan(
                        kind=span.kind,
                        start=start,
                        end=end,
                        related_node_id=span.related_node_id,
                        metadata=dict(span.metadata),
                    )
                )
    return sorted(accepted, key=lambda span: (span.start, span.end))


def _contains_call(node: ast.AST) -> bool:
    """Return whether an AST subtree invokes a supported RLM/LLM helper."""
    return any(
        isinstance(child, ast.Call) and _call_name(child.func) in _CALL_HELPERS
        for child in ast.walk(node)
    )


def _call_name(node: ast.AST) -> str | None:
    """Return a direct function name, excluding attribute and dynamic calls."""
    return node.id if isinstance(node, ast.Name) else None


def _is_final_assignment(node: ast.AST) -> bool:
    """Return whether an AST node assigns ``answer['content']`` or ``answer['ready']``."""
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return False
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
            continue
        if target.value.id != "answer":
            continue
        slice_node = target.slice
        if isinstance(slice_node, ast.Constant) and slice_node.value in {"content", "ready"}:
            return True
    return False


def _absolute_node_span(block: ParsedCodeBlock, node: ast.AST) -> tuple[int, int]:
    """Convert an AST node's UTF-8 offsets to full-response character offsets."""
    start = block.start + _source_offset(block.code, node.lineno, node.col_offset)
    end = block.start + _source_offset(block.code, node.end_lineno, node.end_col_offset)
    return start, end


def _absolute_header_span(
    block: ParsedCodeBlock, node: ast.If | ast.For | ast.AsyncFor | ast.While
) -> tuple[int, int]:
    """Return the response-relative interval covering a control-flow header only."""
    start = block.start + _source_offset(block.code, node.lineno, node.col_offset)
    if not node.body:
        return start, start
    body_start = _source_offset(block.code, node.body[0].lineno, node.body[0].col_offset)
    header = block.code[_source_offset(block.code, node.lineno, node.col_offset) : body_start]
    end = block.start + body_start - (len(header) - len(header.rstrip()))
    return start, end


def _source_offset(source: str, lineno: int, utf8_col: int) -> int:
    """Translate an AST line/UTF-8-column pair into a Python character offset."""
    lines = source.splitlines(keepends=True)
    line = lines[lineno - 1]
    char_col = len(line.encode("utf-8")[:utf8_col].decode("utf-8"))
    return sum(len(item) for item in lines[: lineno - 1]) + char_col


def _subtract(
    fragments: list[tuple[int, int]], claimed_start: int, claimed_end: int
) -> list[tuple[int, int]]:
    """Subtract one half-open claimed interval from a list of interval fragments."""
    remaining: list[tuple[int, int]] = []
    for start, end in fragments:
        if claimed_end <= start or claimed_start >= end:
            remaining.append((start, end))
            continue
        if start < claimed_start:
            remaining.append((start, claimed_start))
        if claimed_end < end:
            remaining.append((claimed_end, end))
    return remaining


def _unclaimed_non_whitespace(
    response: str,
    claimed: list[DecisionSpan],
    kind: DecisionKind,
    *,
    excluded: list[tuple[int, int]] | None = None,
) -> list[DecisionSpan]:
    """Convert unclaimed, non-whitespace response fragments into spans.

    Args:
        response: Full generated response.
        claimed: Existing spans to subtract.
        kind: Component assigned to remaining fragments.
        excluded: Optional intervals, such as fenced code, that must remain unclassified.

    Returns:
        Trimmed spans for every remaining non-whitespace fragment.
    """
    fragments = [(0, len(response))]
    for span in claimed:
        fragments = _subtract(fragments, span.start, span.end)
    for start, end in excluded or []:
        fragments = _subtract(fragments, start, end)
    spans: list[DecisionSpan] = []
    for start, end in fragments:
        content = response[start:end]
        left = len(content) - len(content.lstrip())
        right = len(content) - len(content.rstrip())
        span_start = start + left
        span_end = end - right
        if span_end > span_start:
            spans.append(DecisionSpan(kind=kind, start=span_start, end=span_end))
    return spans
