"""Define the node-sample payload expected by a future pinned Prime adapter.

Purpose:
    Isolate tree-aware SDPO data from version-specific Prime trainer internals.
Implementation:
    A dependency-free dataclass carries exclusive masks and validated teacher targets;
    its validator enforces token alignment and the same-tokenizer decision.
Inputs:
    Trajectory/node IDs, token-aligned component masks, and a top-k teacher target.
Outputs:
    A validated transport payload ready to attach to a Prime training sample.
Example:
    ``fields.validate(token_count=len(tokens), student_tokenizer_fingerprint=tokenizer_id)``
"""

from __future__ import annotations

from dataclasses import dataclass

from rlm.core.trajectory import DecisionKind

from rlm_train.sdpo.teacher import TopKTeacherTarget


@dataclass(frozen=True)
class PrimeTreeSDPOFields:
    """Carry all tree-SDPO fields associated with one node continuation.

    Args:
        trajectory_id: Stable rollout identifier.
        node_id: Stable invocation identifier within the trajectory.
        component_masks: Exclusive token masks keyed by decision component.
        teacher_target: Teacher-selected token IDs, log-probabilities, and tail mass.

    Example:
        ``PrimeTreeSDPOFields("run", "root", masks, target)``
    """

    trajectory_id: str
    node_id: str
    component_masks: dict[DecisionKind, list[bool]]
    teacher_target: TopKTeacherTarget

    def validate(self, token_count: int, student_tokenizer_fingerprint: str) -> None:
        """Validate tokenizer identity, token counts, and mask exclusivity.

        Args:
            token_count: Number of continuation positions in the Prime sample.
            student_tokenizer_fingerprint: Stable identity of the student tokenizer.

        Returns:
            ``None`` after all transport invariants pass.

        Raises:
            ValueError: If tokenizer identities differ, masks or targets have the wrong
                length, or multiple components claim one token position.
        """
        if student_tokenizer_fingerprint != self.teacher_target.tokenizer_fingerprint:
            raise ValueError("student and teacher tokenizer fingerprints must match")
        for kind, mask in self.component_masks.items():
            if len(mask) != token_count:
                raise ValueError(
                    f"{kind.value} mask has {len(mask)} positions; expected {token_count}"
                )
        if len(self.teacher_target.token_ids) != token_count:
            raise ValueError("teacher target position count must match the Prime training sample")
        for position in range(token_count):
            active_components = sum(int(mask[position]) for mask in self.component_masks.values())
            if active_components > 1:
                raise ValueError("exclusive component masks overlap at one or more positions")


@dataclass(frozen=True)
class PrimeQuestionSDPOFields:
    """Carry one edge-isolated question mask and teacher target to Prime.

    Attributes:
        trajectory_id: Stable rollout identifier.
        parent_node_id: Node containing the scored question tokens.
        child_node_id: Child whose result supplied edge-local teacher feedback.
        call_order: Zero-based helper-call order within the parent continuation.
        batch_index: Item index for a batched call, or ``None`` for a scalar call.
        question_mask: Token-aligned mask that activates only this question.
        teacher_target: Detached top-k-plus-tail distribution for all token positions.

    Example:
        ``fields = PrimeQuestionSDPOFields("run", "root", "child", 0, None, mask, target)``
    """

    trajectory_id: str
    parent_node_id: str
    child_node_id: str
    call_order: int
    batch_index: int | None
    question_mask: list[bool]
    teacher_target: TopKTeacherTarget

    def validate(self, token_count: int, student_tokenizer_fingerprint: str) -> None:
        """Require exact token/target alignment and a non-empty question mask.

        Args:
            token_count: Number of continuation positions in the Prime sample.
            student_tokenizer_fingerprint: Stable identity of the student tokenizer.

        Returns:
            ``None`` after all question-transport invariants pass.

        Raises:
            ValueError: If call coordinates are negative, tokenizer identities differ,
                the mask is empty or misaligned, or teacher positions are misaligned.

        Example:
            ``fields.validate(len(tokens), student_tokenizer_fingerprint="tok-v1")``
        """
        if self.call_order < 0:
            raise ValueError("question call_order must be non-negative")
        if self.batch_index is not None and self.batch_index < 0:
            raise ValueError("question batch_index must be non-negative")
        if student_tokenizer_fingerprint != self.teacher_target.tokenizer_fingerprint:
            raise ValueError("student and teacher tokenizer fingerprints must match")
        if len(self.question_mask) != token_count:
            raise ValueError("question mask must match the Prime training sample")
        if not any(self.question_mask):
            raise ValueError("question mask must activate at least one token")
        if len(self.teacher_target.token_ids) != token_count:
            raise ValueError("teacher target must match the Prime training sample")
