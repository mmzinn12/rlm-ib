"""Independent training objectives and their sole composer."""

from rlm_train.objectives.composer import ComposedObjectiveResult, ObjectiveComposer
from rlm_train.objectives.protocol import (
    Objective,
    ObjectiveBatch,
    ObjectiveCapabilities,
    ObjectiveResult,
)

__all__ = [
    "ComposedObjectiveResult",
    "Objective",
    "ObjectiveBatch",
    "ObjectiveCapabilities",
    "ObjectiveComposer",
    "ObjectiveResult",
]
