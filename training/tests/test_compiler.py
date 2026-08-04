"""Verify compilation from traced trajectory feedback to node training examples.

Purpose:
    Protect missing-call span reclassification and final-answer feedback attachment.
Implementation:
    Tests build minimal in-memory trajectory and feedback objects, invoke the compiler,
    and inspect the resulting node examples.
Inputs:
    Synthetic ``TrajectoryTree`` and ``TrajectoryFeedback`` instances.
Outputs:
    Pytest assertions over compiled spans and feedback payloads.
Example:
    Run ``pytest training/tests/test_compiler.py`` from the repository root.
"""

from rlm.core.trajectory import (
    CallItemSpan,
    DecisionKind,
    DecisionSpan,
    InvocationKind,
    InvocationNode,
    TrajectoryTree,
)

from rlm_train.judge.schema import (
    FinalAnswerFeedback,
    InformationValueFeedback,
    NodeFeedback,
    RoutingFeedback,
    TrajectoryFeedback,
)
from rlm_train.trajectory.compiler import TrajectoryCompiler


def test_compiler_targets_missing_call_feedback_at_routing_tokens():
    tree = TrajectoryTree(
        trajectory_id="run",
        nodes=[
            InvocationNode(
                node_id="run/root/i000",
                parent_id=None,
                depth=0,
                kind=InvocationKind.ROOT,
                model="model",
                context="context",
                response="if uncertain: stop",
                spans=[DecisionSpan(kind=DecisionKind.ROUTE, start=0, end=13)],
            )
        ],
    )
    feedback = TrajectoryFeedback(
        trajectory_score=0.0,
        nodes=[
            NodeFeedback(
                node_id="run/root/i000",
                routing_feedback=RoutingFeedback(
                    quality="poor",
                    missing_calls=["Ask for the discriminating assay result."],
                ),
            )
        ],
        judge_version="judge-v1",
        rubric_version="rubric-v1",
    )

    example = TrajectoryCompiler().compile(tree, feedback)[0]

    assert example.spans[0].kind is DecisionKind.MISSING_CALL


def test_compiler_attaches_final_feedback_to_the_final_root_node():
    tree = TrajectoryTree(
        trajectory_id="run",
        nodes=[
            InvocationNode(
                node_id="run/root/i000",
                parent_id=None,
                depth=0,
                kind=InvocationKind.ROOT,
                model="model",
                context="context",
                response='answer["ready"] = True',
                spans=[DecisionSpan(kind=DecisionKind.FINAL, start=0, end=22)],
            )
        ],
    )
    feedback = TrajectoryFeedback(
        trajectory_score=1.0,
        final_answer_feedback=FinalAnswerFeedback(outcome="correct"),
        judge_version="judge-v1",
        rubric_version="rubric-v1",
    )

    example = TrajectoryCompiler().compile(tree, feedback)[0]

    assert example.feedback["final_answer_feedback"]["outcome"] == "correct"


def test_compiler_emits_one_restricted_example_per_bound_question():
    response = 'questions = ["first?", "second?"]'
    root_id = "run/root/i000"
    first_id = f"{root_id}/c000/b000"
    second_id = f"{root_id}/c000/b001"
    first_start = response.index('"first?"')
    second_start = response.index('"second?"')
    tree = TrajectoryTree(
        trajectory_id="run",
        nodes=[
            InvocationNode(
                node_id=root_id,
                parent_id=None,
                depth=0,
                kind=InvocationKind.ROOT,
                model="model",
                context="student-only context",
                response=response,
                call_item_spans=[
                    CallItemSpan(0, 0, first_start, first_start + len('"first?"'), first_id),
                    CallItemSpan(
                        0,
                        1,
                        second_start,
                        second_start + len('"second?"'),
                        second_id,
                    ),
                ],
            ),
            InvocationNode(
                node_id=first_id,
                parent_id=root_id,
                depth=1,
                kind=InvocationKind.SUBCALL,
                model="model",
                context="first?",
                call_order=0,
                batch_index=0,
            ),
            InvocationNode(
                node_id=second_id,
                parent_id=root_id,
                depth=1,
                kind=InvocationKind.SUBCALL,
                model="model",
                context="second?",
                call_order=0,
                batch_index=1,
            ),
        ],
    )
    feedback = TrajectoryFeedback(
        trajectory_score=1.0,
        subcalls=[
            InformationValueFeedback(
                parent_node_id=root_id,
                child_node_id=first_id,
                information_significance=0.4,
                novelty=0.5,
                uncertainty_reduction=0.6,
                evidence_quality=0.7,
                information_revealed=["first-only evidence"],
                edge_local_diagnostic="The first question is useful but underspecified.",
                rationale="must not leak",
            ),
            InformationValueFeedback(
                parent_node_id=root_id,
                child_node_id=second_id,
                information_significance=-0.2,
                novelty=0.1,
                uncertainty_reduction=0.0,
                evidence_quality=0.2,
                information_revealed=["second-only evidence"],
            ),
        ],
        judge_version="judge-v1",
        rubric_version="rubric-v1",
    )

    examples = TrajectoryCompiler().compile_questions(tree, feedback)

    assert [example.child_node_id for example in examples] == [first_id, second_id]
    assert all(example.student_continuation == response for example in examples)
    assert examples[0].feedback.child_node_id == first_id
    assert "rationale" not in examples[0].feedback.model_dump()
    assert "information_revealed" not in examples[0].feedback.model_dump()
    assert examples[0].feedback.diagnostic == "The first question is useful but underspecified."
    assert "second-only evidence" not in examples[0].feedback.model_dump_json()


def test_compiler_preserves_100_question_identities_across_completion_order():
    """Keep source-order question identity independent of child completion order."""
    questions = [f"question-{index:03d}?" for index in range(100)]
    response = f"questions = {questions!r}"
    root_id = "run/root/i000"
    child_ids = [f"{root_id}/c000/b{index:03d}" for index in range(100)]
    item_spans = []
    search_start = 0
    for batch_index, (question, child_id) in enumerate(zip(questions, child_ids, strict=True)):
        start = response.index(repr(question), search_start)
        end = start + len(repr(question))
        item_spans.append(CallItemSpan(0, batch_index, start, end, child_id))
        search_start = end
    children = [
        InvocationNode(
            node_id=child_id,
            parent_id=root_id,
            depth=1,
            kind=InvocationKind.SUBCALL,
            model="model",
            context=question,
            response=f"answer-{batch_index:03d}",
            call_order=0,
            batch_index=batch_index,
        )
        for batch_index, (question, child_id) in reversed(
            list(enumerate(zip(questions, child_ids, strict=True)))
        )
    ]
    tree = TrajectoryTree(
        trajectory_id="run",
        nodes=[
            InvocationNode(
                node_id=root_id,
                parent_id=None,
                depth=0,
                kind=InvocationKind.ROOT,
                model="model",
                context="student-only context",
                response=response,
                call_item_spans=list(reversed(item_spans)),
            ),
            *children,
        ],
    )
    feedback = TrajectoryFeedback(
        trajectory_score=1.0,
        subcalls=[
            InformationValueFeedback(
                parent_node_id=root_id,
                child_node_id=child_id,
                information_significance=0.5,
                novelty=0.5,
                uncertainty_reduction=0.5,
                evidence_quality=0.5,
            )
            for child_id in reversed(child_ids)
        ],
        judge_version="judge-v1",
        rubric_version="rubric-v1",
    )

    examples = TrajectoryCompiler().compile_questions(tree, feedback)

    assert len(examples) == 100
    assert [example.batch_index for example in examples] == list(range(100))
    assert [example.child_node_id for example in examples] == child_ids
    assert [
        example.student_continuation[example.question_span.start : example.question_span.end]
        for example in examples
    ] == [repr(question) for question in questions]
    assert [example.feedback.child_node_id for example in examples] == child_ids
