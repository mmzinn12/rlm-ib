"""Manage feedback-free Gram anchor inputs, outputs, identity, and lifecycle.

Purpose:
    Keep representation anchors independent from judge-conditioned teachers and record
    the exact reference version used by every Gram sample.
Implementation:
    Immutable transport types define aligned model inputs and outputs; protocols specify
    the model boundary; fixed-checkpoint and periodic-EMA controllers own loading,
    refresh timing, no-grad execution, detachment, and identity metadata.
Inputs:
    Student-aligned token tensors, selected block indices, model loader/forward callbacks,
    checkpoints, EMA models, and optimizer steps.
Outputs:
    Detached ``AnchorForwardOutput`` objects and versioned ``AnchorIdentity`` metadata.
Example:
    ``output = anchor.forward(inputs, layer_indices=(15, 23, 31))``
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AlignedSequenceInputs:
    """Carry model inputs shared exactly by student and anchor.

    Attributes:
        input_ids: Token IDs containing the student's prompt and sampled continuation.
        attention_mask: Mask aligned with ``input_ids`` and used by both models.
        position_ids: Optional explicit causal positions shared by both forward passes.

    Example:
        ``inputs = AlignedSequenceInputs(ids, attention, position_ids=positions)``

    The type deliberately has no feedback field, establishing a structural boundary
    against judge-conditioned or otherwise privileged anchor inputs.
    """

    input_ids: Any
    attention_mask: Any
    position_ids: Any | None = None


@dataclass(frozen=True)
class AnchorIdentity:
    """Version an anchor for samples, metrics, caches, and checkpoints.

    Attributes:
        strategy: Lifecycle policy, such as ``fixed_checkpoint``.
        identifier: Stable checkpoint path or logical EMA name.
        version: Monotonic version within the current training run.
        created_step: Optimizer step at which this anchor became active.

    Example:
        ``identity = AnchorIdentity("fixed_checkpoint", "base", 0, 0)``
    """

    strategy: str
    identifier: str
    version: int
    created_step: int

    def age(self, global_step: int) -> int:
        """Return the anchor's non-negative optimizer-step age.

        Args:
            global_step: Current optimizer step.

        Returns:
            ``global_step - created_step``.

        Raises:
            ValueError: If the requested step predates this anchor version.
        """
        if global_step < self.created_step:
            raise ValueError("global_step cannot precede anchor creation")
        return global_step - self.created_step


@dataclass(frozen=True)
class AnchorForwardOutput:
    """Carry detached anchor outputs for one aligned sequence.

    Attributes:
        logits: Next-token logits aligned position-for-position with student logits.
        hidden_states: Mapping from requested zero-based block index to its output.
        identity: Exact anchor version that produced the tensors.

    Example:
        ``output = AnchorForwardOutput(logits, {31: final_states}, identity)``
    """

    logits: Any
    hidden_states: Mapping[int, Any]
    identity: AnchorIdentity


class AlignedAnchorSource(Protocol):
    """Define a feedback-free anchor forward-pass contract.

    Implementations teacher-force the student's exact token sequence, preserve causal
    alignment, return only requested block outputs, and detach all anchor tensors.
    """

    @property
    def identity(self) -> AnchorIdentity:
        """Return the current anchor identity and version."""
        ...

    def forward(
        self, inputs: AlignedSequenceInputs, layer_indices: tuple[int, ...]
    ) -> AnchorForwardOutput:
        """Return detached aligned logits and selected hidden states.

        Args:
            inputs: Feedback-free model inputs shared with the student.
            layer_indices: Unique transformer-block outputs to capture.

        Returns:
            An ``AnchorForwardOutput`` aligned to ``inputs``.
        """
        ...


AnchorForwardFn = Callable[
    [Any, AlignedSequenceInputs, tuple[int, ...]], tuple[Any, Mapping[int, Any]]
]


class FixedCheckpointAnchorController:
    """Load one pre-training checkpoint and keep it fixed for the run.

    Args:
        checkpoint_path: Stable checkpoint identifier or loader path.
        load_checkpoint: Callback that materializes the anchor model once.
        forward_model: Callback that returns aligned logits and requested block outputs.

    Raises:
        ValueError: If ``checkpoint_path`` is blank.

    Example:
        ``anchor = FixedCheckpointAnchorController("base", load_checkpoint=load, forward_model=forward)``
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        load_checkpoint: Callable[[str], Any],
        forward_model: AnchorForwardFn,
    ):
        if not checkpoint_path.strip():
            raise ValueError("checkpoint_path must not be blank")
        self._checkpoint_path = checkpoint_path
        self._load_checkpoint = load_checkpoint
        self._forward_model = forward_model
        self._model: Any | None = None
        self._identity = AnchorIdentity(
            strategy="fixed_checkpoint",
            identifier=checkpoint_path,
            version=0,
            created_step=0,
        )

    @property
    def identity(self) -> AnchorIdentity:
        """Return the immutable checkpoint identity."""
        return self._identity

    def initialize(self) -> None:
        """Load the checkpoint once at the trainer integration boundary.

        Repeated calls are idempotent. Loader errors intentionally propagate so an
        active training run cannot silently continue without its configured anchor.
        """
        if self._model is None:
            self._model = self._load_checkpoint(self._checkpoint_path)

    def forward(
        self, inputs: AlignedSequenceInputs, layer_indices: tuple[int, ...]
    ) -> AnchorForwardOutput:
        """Run the fixed anchor under no-grad and detach its outputs.

        Args:
            inputs: Student-aligned, feedback-free token inputs.
            layer_indices: Block indices whose outputs should be returned.

        Returns:
            Detached aligned logits, selected hidden states, and fixed identity.

        Raises:
            ValueError: If the forward callback returns a different layer set.
        """
        self.initialize()
        with _no_grad():
            logits, hidden_states = self._forward_model(self._model, inputs, layer_indices)
        return _detached_output(logits, hidden_states, layer_indices, self.identity)

    def logits_for(self, inputs: AlignedSequenceInputs) -> Any:
        """Return detached aligned logits without requesting hidden layers.

        Args:
            inputs: Student-aligned, feedback-free token inputs.

        Returns:
            Reference logits suitable for detached JS calculation.
        """
        return self.forward(inputs, ()).logits


class PeriodicEMASnapshotAnchorController:
    """Replace the anchor periodically with a detached EMA snapshot.

    Args:
        update_interval: Positive minimum optimizer steps between snapshots.
        snapshot_model: Callback that freezes or copies the supplied EMA model.
        forward_model: Callback returning aligned logits and selected hidden states.
        identifier: Stable logical name recorded in anchor metadata.

    Raises:
        ValueError: If ``update_interval`` is not positive.

    Example:
        ``anchor = PeriodicEMASnapshotAnchorController(100, snapshot_model=copy_ema, forward_model=forward)``
    """

    def __init__(
        self,
        update_interval: int,
        *,
        snapshot_model: Callable[[Any], Any],
        forward_model: AnchorForwardFn,
        identifier: str = "ema",
    ):
        if update_interval <= 0:
            raise ValueError("EMA anchor update_interval must be positive")
        self._update_interval = update_interval
        self._snapshot_model = snapshot_model
        self._forward_model = forward_model
        self._identifier = identifier
        self._model: Any | None = None
        self._identity: AnchorIdentity | None = None

    @property
    def identity(self) -> AnchorIdentity:
        """Return the current snapshot identity.

        Raises:
            ValueError: If no snapshot has been captured yet.
        """
        if self._identity is None:
            raise ValueError("EMA anchor has not been initialized")
        return self._identity

    def maybe_refresh(self, ema_model: Any, *, global_step: int) -> bool:
        """Capture an EMA snapshot at initialization or a periodic boundary.

        Args:
            ema_model: Current EMA model to copy or freeze.
            global_step: Non-negative optimizer step.

        Returns:
            ``True`` when a new version was captured, otherwise ``False``.

        Raises:
            ValueError: If ``global_step`` is negative.
        """
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        if (
            self._identity is not None
            and global_step - self._identity.created_step < self._update_interval
        ):
            return False
        with _no_grad():
            self._model = self._snapshot_model(ema_model)
        version = 0 if self._identity is None else self._identity.version + 1
        self._identity = AnchorIdentity(
            strategy="periodic_ema_snapshot",
            identifier=self._identifier,
            version=version,
            created_step=global_step,
        )
        return True

    def forward(
        self, inputs: AlignedSequenceInputs, layer_indices: tuple[int, ...]
    ) -> AnchorForwardOutput:
        """Run the current snapshot under no-grad over aligned inputs.

        Args:
            inputs: Student-aligned, feedback-free token inputs.
            layer_indices: Block indices whose outputs should be returned.

        Returns:
            Detached outputs tagged with the current snapshot identity.

        Raises:
            ValueError: If no snapshot exists or returned layers do not match.
        """
        if self._model is None:
            raise ValueError("EMA anchor must be initialized with maybe_refresh")
        with _no_grad():
            logits, hidden_states = self._forward_model(self._model, inputs, layer_indices)
        return _detached_output(logits, hidden_states, layer_indices, self.identity)

    def logits_for(self, inputs: AlignedSequenceInputs) -> Any:
        """Return current-snapshot logits without hidden-state capture.

        Args:
            inputs: Student-aligned, feedback-free token inputs.

        Returns:
            Detached reference logits for JS calculation.
        """
        return self.forward(inputs, ()).logits


def _detached_output(
    logits: Any,
    hidden_states: Mapping[int, Any],
    layer_indices: tuple[int, ...],
    identity: AnchorIdentity,
) -> AnchorForwardOutput:
    """Validate selected layers and sever all anchor autograd paths."""
    if set(hidden_states) != set(layer_indices):
        raise ValueError("anchor output must contain exactly the requested hidden layers")
    detached_logits = logits.detach()
    detached_states = {index: value.detach() for index, value in hidden_states.items()}
    return AnchorForwardOutput(detached_logits, detached_states, identity)


def _no_grad() -> Any:
    """Use PyTorch no-grad when installed while keeping this module import-light."""
    try:
        torch = __import__("torch")
    except ImportError:
        return contextlib.nullcontext()
    return torch.no_grad()


__all__ = [
    "AlignedAnchorSource",
    "AlignedSequenceInputs",
    "AnchorForwardOutput",
    "AnchorIdentity",
    "FixedCheckpointAnchorController",
    "PeriodicEMASnapshotAnchorController",
]
