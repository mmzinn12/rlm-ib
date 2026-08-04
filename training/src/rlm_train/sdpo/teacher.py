"""Run feedback-conditioned EMA teaching and build compact target payloads.

Purpose:
    Separate teacher scoring and EMA lifecycle management from any model framework or
    distributed trainer implementation.
Implementation:
    ``TopKTeacherTarget`` validates full-vocabulary-normalized top-k log-probabilities
    plus an explicit tail bucket. PyTorch utilities extract targets and maintain a
    frozen EMA model, while a provider protocol isolates model-specific tokenization.
Inputs:
    Node training examples, structured feedback context, teacher versions, and model
    parameters supplied by a future integration.
Outputs:
    Validated top-k teacher targets and stable scorer/controller contracts.
Example:
    ``target = await scorer.score(example, feedback_context=example.feedback)``
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rlm_train.judge.schema import QuestionTeacherFeedback
from rlm_train.sdpo.config import TeacherStrategy
from rlm_train.trajectory.compiler import NodeTrainingExample, QuestionTrainingExample

if TYPE_CHECKING:
    from rlm_train.sdpo.cache import TeacherTargetCache


@dataclass(frozen=True)
class TeacherIdentity:
    """Describe an exact frozen or moving teacher in artifacts and checkpoints."""

    strategy: TeacherStrategy
    checkpoint_identity: str
    fingerprint: str
    version: int
    update_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible teacher provenance."""
        payload = asdict(self)
        payload["strategy"] = self.strategy.value
        return payload


def model_state_fingerprint(model: Any) -> str:
    """Hash names, shapes, dtypes, and bytes for a PyTorch module state."""
    try:
        torch = __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for teacher fingerprinting") from exc
    if not isinstance(model, torch.nn.Module):
        raise TypeError("teacher fingerprinting requires a PyTorch module")
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class TopKTeacherTarget(BaseModel):
    """Store full-vocabulary-normalized top-k probabilities for every token position.

    Attributes:
        token_ids: Teacher-selected vocabulary IDs for each continuation position.
        logprobs: Full-softmax log-probabilities aligned with ``token_ids``.
        tail_logprobs: Log-probability mass of all non-selected tokens per position.
        teacher_version: Monotonic EMA teacher version used to produce the target.
        tokenizer_fingerprint: Stable teacher-tokenizer identity checked against student.

    Raises:
        pydantic.ValidationError: If position counts or top-k widths disagree, token IDs
            are invalid, values are non-finite, or explicit plus tail mass is not one.

    Example:
        ``TopKTeacherTarget(token_ids=[[1]], logprobs=[[-0.2231435513]], tail_logprobs=[-1.6094379124], teacher_version=0, tokenizer_fingerprint="tok-v1")``
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    token_ids: list[list[int]]
    logprobs: list[list[float]]
    tail_logprobs: list[float]
    teacher_version: int = Field(ge=0)
    tokenizer_fingerprint: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_shapes(self) -> TopKTeacherTarget:
        """Validate position alignment, token IDs, finiteness, and probability mass.

        Returns:
            This validated target.

        Raises:
            ValueError: If any target-shape or probability invariant fails.
        """
        positions = len(self.token_ids)
        if positions == 0:
            raise ValueError("teacher target must contain at least one token position")
        if len(self.logprobs) != positions or len(self.tail_logprobs) != positions:
            raise ValueError("teacher target fields must have the same token-position count")
        widths = {len(row) for row in self.token_ids}
        if len(widths) > 1:
            raise ValueError("teacher top-k width must be constant across token positions")
        if widths == {0}:
            raise ValueError("teacher target must retain at least one explicit token")
        for ids, logprobs, tail_logprob in zip(
            self.token_ids, self.logprobs, self.tail_logprobs, strict=True
        ):
            if len(ids) != len(logprobs):
                raise ValueError("teacher token IDs and logprobs must align")
            if len(ids) != len(set(ids)) or any(token_id < 0 for token_id in ids):
                raise ValueError("teacher top-k token IDs must be unique and non-negative")
            all_logprobs = [*logprobs, tail_logprob]
            if any(not math.isfinite(value) for value in all_logprobs):
                raise ValueError("teacher target logprobs must be finite")
            mass = sum(math.exp(value) for value in all_logprobs)
            if not math.isclose(mass, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError("teacher top-k and tail probabilities must sum to one")
        return self


class TeacherScorer(Protocol):
    """Specify how a feedback-conditioned teacher scores one node continuation."""

    async def score(
        self,
        example: NodeTrainingExample,
        *,
        feedback_context: dict[str, Any],
    ) -> TopKTeacherTarget:
        """Produce detached top-k targets for a node-level training example.

        Args:
            example: Feedback-free student context and sampled continuation.
            feedback_context: Structured judge feedback visible only to the teacher.

        Returns:
            A normalized top-k-plus-tail target aligned to continuation positions.
        """
        ...


class QuestionTeacherScorer(Protocol):
    """Define feedback-conditioned scoring for exactly one question edge.

    Implementations receive an existing student continuation and must score its token
    positions in place. They must not generate a revised question or expose feedback
    belonging to sibling questions, final outcomes, or global rewards.
    """

    async def score_question(self, example: QuestionTrainingExample) -> TopKTeacherTarget:
        """Score an existing continuation without generating replacement text.

        Args:
            example: One question edge, its exact span, and leakage-restricted feedback.

        Returns:
            Full-vocabulary-normalized top-k-plus-tail targets aligned to every token
            position in ``example.student_continuation``.

        Example:
            ``target = await scorer.score_question(question_example)``
        """
        ...


class QuestionTeacherLogitsProvider(Protocol):
    """Score an existing continuation with one versioned teacher model.

    Model-specific tokenization and feedback rendering live behind this protocol. The
    scorer supplies only the student-visible context, the existing continuation, and
    one restricted ``QuestionTeacherFeedback`` object.
    """

    @property
    def teacher_version(self) -> int:
        """Return the version of the teacher used by the next scoring call."""
        ...

    async def score_existing_continuation(
        self,
        *,
        student_context: Any,
        student_continuation: str,
        feedback: QuestionTeacherFeedback,
    ) -> Any:
        """Return teacher logits shaped ``[continuation_tokens, vocabulary]``."""
        ...


class EMATeacherController(Protocol):
    """Specify lifecycle operations for a versioned EMA copy of the student.

    The trainer must call ``update_after_optimizer_step`` only after a completed student
    optimizer step so teacher targets never receive gradients from the active loss.
    """

    @property
    def version(self) -> int:
        """Return the non-negative teacher version used in cache and target metadata."""
        ...

    def update_after_optimizer_step(self, student: Any, update_rate: float) -> None:
        """Move teacher parameters toward the post-step student parameters.

        Args:
            student: Model whose current parameters are the EMA source.
            update_rate: Interpolation rate in ``(0, 1]`` supplied by ``SDPOConfig``.

        Returns:
            ``None`` after updating teacher parameters and its version.
        """
        ...


def build_question_feedback_context(
    feedback: QuestionTeacherFeedback,
) -> dict[str, Any]:
    """Materialize only the edge-local fields allowed into question teaching.

    Args:
        feedback: The restricted immutable feedback view for the active question.

    Returns:
        A fresh JSON-compatible mapping without judge rationale, sibling feedback,
        global reward, final-answer outcome, or privileged judge context.

    Raises:
        TypeError: If complete judge feedback or another object crosses the boundary.
    """
    if not isinstance(feedback, QuestionTeacherFeedback):
        raise TypeError("question teacher context requires QuestionTeacherFeedback")
    return feedback.model_dump(mode="json")


def validate_question_teacher_example(example: QuestionTrainingExample) -> None:
    """Require exact agreement among the question, span, child, and feedback IDs."""
    if not isinstance(example.feedback, QuestionTeacherFeedback):
        raise TypeError("question teacher example requires QuestionTeacherFeedback")
    if example.feedback.parent_node_id != example.parent_node_id:
        raise ValueError("question feedback parent does not match the training example")
    if example.feedback.child_node_id != example.child_node_id:
        raise ValueError("question feedback child does not match the training example")
    if example.question_span.child_node_id != example.child_node_id:
        raise ValueError("question span child does not match the training example")
    if example.question_span.call_order != example.call_order:
        raise ValueError("question span call order does not match the training example")
    if example.question_span.batch_index != example.batch_index:
        raise ValueError("question span batch index does not match the training example")
    if example.question_span.end > len(example.student_continuation):
        raise ValueError("question span exceeds the student continuation")


def extract_topk_teacher_target(
    teacher_logits: Any,
    *,
    top_k: int,
    teacher_version: int,
    tokenizer_fingerprint: str,
) -> TopKTeacherTarget:
    """Detach teacher logits and extract normalized top-k plus exact tail mass.

    PyTorch's ``log_softmax``, ``topk``, and ``logsumexp`` primitives provide the
    production implementation. The tail is the aggregate probability of every token
    not selected by the teacher at that position.
    """
    try:
        torch = __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for teacher target extraction") from exc
    if not isinstance(teacher_logits, torch.Tensor):
        raise TypeError("teacher logits must be a PyTorch tensor")
    if teacher_logits.ndim != 2:
        raise ValueError("teacher logits must have shape [tokens, vocabulary]")
    token_count, vocabulary_size = teacher_logits.shape
    if token_count == 0:
        raise ValueError("teacher logits must contain at least one token position")
    if top_k <= 0 or top_k >= vocabulary_size:
        raise ValueError("top_k must be positive and smaller than the vocabulary")
    if teacher_version < 0:
        raise ValueError("teacher_version must be non-negative")
    if not tokenizer_fingerprint.strip():
        raise ValueError("tokenizer_fingerprint must not be blank")
    if not torch.isfinite(teacher_logits).all().item():
        raise ValueError("teacher logits must be finite")

    with torch.no_grad():
        logprobs = torch.log_softmax(teacher_logits.detach(), dim=-1)
        topk_logprobs, token_ids = torch.topk(logprobs, k=top_k, dim=-1)
        tail_values = logprobs.clone()
        tail_values.scatter_(dim=-1, index=token_ids, value=-torch.inf)
        tail_logprobs = torch.logsumexp(tail_values, dim=-1)
    return TopKTeacherTarget(
        token_ids=token_ids.cpu().tolist(),
        logprobs=topk_logprobs.cpu().tolist(),
        tail_logprobs=tail_logprobs.cpu().tolist(),
        teacher_version=teacher_version,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )


class TopKQuestionTeacherScorer:
    """Score one existing question continuation and cache its compact target."""

    def __init__(
        self,
        provider: QuestionTeacherLogitsProvider,
        *,
        top_k: int,
        tokenizer_fingerprint: str,
        feedback_version: str,
        cache: TeacherTargetCache | None = None,
    ) -> None:
        """Configure an edge-isolated scorer and optional target cache."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not tokenizer_fingerprint.strip() or not feedback_version.strip():
            raise ValueError("tokenizer and feedback versions must not be blank")
        self.provider = provider
        self.top_k = top_k
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.feedback_version = feedback_version
        self.cache = cache

    async def score_question(self, example: QuestionTrainingExample) -> TopKTeacherTarget:
        """Score the student's existing continuation without generating new text."""
        from rlm_train.sdpo.cache import make_question_teacher_cache_key

        validate_question_teacher_example(example)
        teacher_version = self.provider.teacher_version
        if teacher_version < 0:
            raise ValueError("teacher version must be non-negative")
        cache_key = make_question_teacher_cache_key(
            example,
            teacher_version=teacher_version,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            feedback_version=self.feedback_version,
        )
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        logits = await self.provider.score_existing_continuation(
            student_context=copy.deepcopy(example.student_context),
            student_continuation=example.student_continuation,
            feedback=example.feedback,
        )
        if self.provider.teacher_version != teacher_version:
            raise RuntimeError("teacher version changed during question scoring")
        target = extract_topk_teacher_target(
            logits,
            top_k=self.top_k,
            teacher_version=teacher_version,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
        )
        if self.cache is not None:
            self.cache.put(cache_key, target)
        return target


class TorchFixedTeacherController:
    """Own an immutable deep copy of the initial policy for an entire run."""

    def __init__(
        self,
        teacher: Any,
        *,
        checkpoint_identity: str = "initial-policy",
    ) -> None:
        """Freeze the supplied module and record its exact initial fingerprint."""
        try:
            torch = __import__("torch")
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for fixed teacher management") from exc
        if not isinstance(teacher, torch.nn.Module):
            raise TypeError("teacher must be a PyTorch module")
        if not checkpoint_identity.strip():
            raise ValueError("teacher checkpoint_identity must not be blank")
        self.teacher = teacher
        self.checkpoint_identity = checkpoint_identity
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.version = 0
        self.update_count = 0
        self.initial_fingerprint = model_state_fingerprint(self.teacher)

    @classmethod
    def from_student(
        cls,
        student: Any,
        *,
        checkpoint_identity: str = "initial-policy",
    ) -> TorchFixedTeacherController:
        """Deep-copy the initial student into a frozen fixed teacher."""
        return cls(copy.deepcopy(student), checkpoint_identity=checkpoint_identity)

    def validate_unchanged(self) -> None:
        """Fail loudly if any teacher parameter or buffer has changed."""
        if model_state_fingerprint(self.teacher) != self.initial_fingerprint:
            raise RuntimeError("fixed teacher state changed after initialization")

    def update_after_optimizer_step(self, student: Any, update_rate: float | None = None) -> None:
        """Validate immutability; fixed teachers intentionally perform no update."""
        del student
        if update_rate is not None:
            raise ValueError("fixed teachers do not accept an update rate")
        self.validate_unchanged()

    def identity(self) -> TeacherIdentity:
        """Return stable fixed-teacher provenance."""
        self.validate_unchanged()
        return TeacherIdentity(
            strategy=TeacherStrategy.FIXED,
            checkpoint_identity=self.checkpoint_identity,
            fingerprint=self.initial_fingerprint,
            version=0,
            update_count=0,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return fixed teacher state and immutable identity for checkpointing."""
        self.validate_unchanged()
        return {
            "checkpoint_identity": self.checkpoint_identity,
            "fingerprint": self.initial_fingerprint,
            "teacher_state_dict": self.teacher.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore only a checkpoint whose declared fingerprint matches its bytes."""
        required = {"checkpoint_identity", "fingerprint", "teacher_state_dict"}
        if set(state) != required:
            raise ValueError("fixed teacher state has unknown or missing fields")
        checkpoint_identity = state["checkpoint_identity"]
        fingerprint = state["fingerprint"]
        if not isinstance(checkpoint_identity, str) or not checkpoint_identity.strip():
            raise ValueError("fixed teacher checkpoint identity must not be blank")
        self.teacher.load_state_dict(state["teacher_state_dict"], strict=True)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        actual = model_state_fingerprint(self.teacher)
        if actual != fingerprint:
            raise ValueError("fixed teacher checkpoint fingerprint does not match its state")
        self.checkpoint_identity = checkpoint_identity
        self.initial_fingerprint = actual


class TorchEMATeacherController:
    """Own a frozen PyTorch teacher and update it after student optimizer steps."""

    def __init__(
        self,
        teacher: Any,
        *,
        version: int = 0,
        checkpoint_identity: str = "initial-policy",
    ) -> None:
        """Validate, freeze, and version an existing teacher module."""
        try:
            torch = __import__("torch")
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for EMA teacher management") from exc
        if not isinstance(teacher, torch.nn.Module):
            raise TypeError("teacher must be a PyTorch module")
        if version < 0:
            raise ValueError("teacher version must be non-negative")
        if not checkpoint_identity.strip():
            raise ValueError("teacher checkpoint_identity must not be blank")
        self.teacher = teacher
        self.checkpoint_identity = checkpoint_identity
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.version = version

    @classmethod
    def from_student(
        cls,
        student: Any,
        *,
        version: int = 0,
        checkpoint_identity: str = "initial-policy",
    ) -> TorchEMATeacherController:
        """Deep-copy a student module into a frozen teacher controller."""
        return cls(
            copy.deepcopy(student),
            version=version,
            checkpoint_identity=checkpoint_identity,
        )

    @property
    def update_count(self) -> int:
        """Return the number of EMA updates represented by this controller."""
        return self.version

    def identity(self) -> TeacherIdentity:
        """Return current EMA version, update count, and exact state fingerprint."""
        return TeacherIdentity(
            strategy=TeacherStrategy.EMA,
            checkpoint_identity=self.checkpoint_identity,
            fingerprint=model_state_fingerprint(self.teacher),
            version=self.version,
            update_count=self.update_count,
        )

    def validate_state_alignment(self, student: Any) -> None:
        """Fail before mutation when student and teacher states do not align exactly."""
        teacher_parameters = dict(self.teacher.named_parameters())
        student_parameters = dict(student.named_parameters())
        teacher_buffers = dict(self.teacher.named_buffers())
        student_buffers = dict(student.named_buffers())
        if teacher_parameters.keys() != student_parameters.keys():
            raise ValueError("student and teacher parameter names do not match")
        if teacher_buffers.keys() != student_buffers.keys():
            raise ValueError("student and teacher buffer names do not match")
        for name, teacher_value in [*teacher_parameters.items(), *teacher_buffers.items()]:
            source = student_parameters.get(name, student_buffers.get(name))
            if source is None:
                raise ValueError(f"student state is missing {name!r}")
            if teacher_value.shape != source.shape:
                raise ValueError(f"student and teacher shapes differ for {name!r}")
            if teacher_value.device != source.device or teacher_value.dtype != source.dtype:
                raise ValueError(f"student and teacher device/dtype differ for {name!r}")

    def update_after_optimizer_step(self, student: Any, update_rate: float) -> None:
        """Apply ``teacher = (1-rate) * teacher + rate * student`` and increment version."""
        try:
            torch = __import__("torch")
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for EMA teacher management") from exc
        if not 0.0 < update_rate <= 1.0:
            raise ValueError("EMA update_rate must be in (0, 1]")
        if not isinstance(student, torch.nn.Module):
            raise TypeError("student must be a PyTorch module")
        self.validate_state_alignment(student)
        student_parameters = dict(student.named_parameters())
        student_buffers = dict(student.named_buffers())
        with torch.no_grad():
            for name, teacher_value in self.teacher.named_parameters():
                teacher_value.lerp_(student_parameters[name].detach(), update_rate)
            for name, teacher_value in self.teacher.named_buffers():
                source = student_buffers[name].detach()
                if teacher_value.is_floating_point() or teacher_value.is_complex():
                    teacher_value.lerp_(source, update_rate)
                else:
                    teacher_value.copy_(source)
        self.version += 1

    def state_dict(self) -> dict[str, Any]:
        """Return teacher weights and lifecycle version for checkpointing."""
        return {
            "version": self.version,
            "teacher_state_dict": self.teacher.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore teacher weights and version from a controller checkpoint."""
        if set(state) != {"version", "teacher_state_dict"}:
            raise ValueError("EMA teacher state has unknown or missing fields")
        version = state["version"]
        if not isinstance(version, int) or version < 0:
            raise ValueError("EMA teacher checkpoint version must be non-negative")
        self.teacher.load_state_dict(state["teacher_state_dict"], strict=True)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.version = version


def build_torch_teacher_controller(
    strategy: TeacherStrategy,
    student: Any,
    *,
    checkpoint_identity: str = "initial-policy",
) -> TorchFixedTeacherController | TorchEMATeacherController | None:
    """Construct the configured teacher without importing a trainer framework."""
    strategy = TeacherStrategy(strategy)
    if strategy is TeacherStrategy.NONE:
        return None
    if strategy is TeacherStrategy.FIXED:
        return TorchFixedTeacherController.from_student(
            student,
            checkpoint_identity=checkpoint_identity,
        )
    return TorchEMATeacherController.from_student(
        student,
        checkpoint_identity=checkpoint_identity,
    )
