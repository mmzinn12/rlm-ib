"""Connect traced question edges to the judge, projector, fixed teacher, and mask."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlm.core.trajectory import TrajectoryTree

from rlm_train.colab.objectives import RolloutSample
from rlm_train.colab.teacher import TransformersQuestionTeacherProvider
from rlm_train.colab.trainer import PreparedQuestionTarget
from rlm_train.judge import PrivilegedJudgeContext, TaskContext, TrajectoryJudge
from rlm_train.sdpo import build_question_token_mask
from rlm_train.sdpo.masks import TokenOffset
from rlm_train.trajectory import TrajectoryCompiler


class TrajectoryQuestionTargetProvider:
    """Prepare one exact question-local target from an actual recorded rollout edge."""

    def __init__(
        self,
        *,
        judge: TrajectoryJudge,
        compiler: TrajectoryCompiler,
        teacher: TransformersQuestionTeacherProvider,
        privileged_contexts: Mapping[str, PrivilegedJudgeContext | None] | None = None,
    ) -> None:
        self.judge = judge
        self.compiler = compiler
        self.teacher = teacher
        self.privileged_contexts = dict(privileged_contexts or {})

    async def __call__(self, sample: RolloutSample) -> PreparedQuestionTarget:
        """Judge privately, project once, and score only the addressed question tokens."""
        trajectory_value = sample.provenance.get("trajectory")
        if isinstance(trajectory_value, dict):
            trajectory = TrajectoryTree.from_dict(trajectory_value)
        elif isinstance(trajectory_value, TrajectoryTree):
            trajectory = trajectory_value
        else:
            raise ValueError("SDPO rollout is missing its recorded trajectory")
        trajectory.validate()
        context = self.privileged_contexts.get(sample.problem_id)
        feedback = await self.judge.evaluate(
            trajectory,
            TaskContext(
                task_id=sample.problem_id,
                prompt=sample.prompt,
                privileged_context=context,
            ),
        )
        question_examples = self.compiler.compile_questions(trajectory, feedback)
        matching = [
            example
            for example in question_examples
            if example.student_continuation == sample.response
            and self._exact_parent_tokens(trajectory, example.parent_node_id)
            == sample.continuation_token_ids
        ]
        if len(matching) != 1:
            raise ValueError(
                "the initial single-GPU SDPO path requires exactly one addressed question "
                "in the sampled parent continuation"
            )
        question = matching[0]
        offsets = [TokenOffset(start, end) for start, end in sample.continuation_token_offsets]
        mask = tuple(build_question_token_mask(question.question_span, offsets))
        if not any(mask):
            raise ValueError("question span does not activate any exact continuation token")
        target, provenance = self.teacher.score_target(
            original_question=sample.prompt,
            continuation_token_ids=sample.continuation_token_ids,
            feedback=question.feedback,
        )
        return PreparedQuestionTarget(
            feedback=question.feedback,
            mask=mask,
            target=target,
            cache_key=provenance.cache_key,
        )

    def validate_unchanged(self) -> None:
        """Verify the fixed teacher at trainer-configured checkpoint intervals."""
        self.teacher.controller.validate_unchanged()

    @staticmethod
    def _exact_parent_tokens(trajectory: TrajectoryTree, parent_node_id: str) -> tuple[int, ...]:
        parent = next(node for node in trajectory.nodes if node.node_id == parent_node_id)
        values: Any = parent.metadata.get("continuation_token_ids")
        if not isinstance(values, list) or not values:
            raise ValueError("trajectory parent lacks exact continuation token metadata")
        return tuple(int(value) for value in values)


__all__ = ["TrajectoryQuestionTargetProvider"]
