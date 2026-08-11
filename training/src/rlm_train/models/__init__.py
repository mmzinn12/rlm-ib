"""Model protocols and concrete policy adapters."""

from rlm_train.models.identity import ComponentIdentity, PolicyIdentity, TokenizerIdentity
from rlm_train.models.protocol import (
    PolicyScore,
    SampledGeneration,
    TrainablePolicy,
)
from rlm_train.models.transformers import (
    GenerationConfig,
    TransformersPolicy,
    TransformersResponseGenerator,
)

__all__ = [
    "ComponentIdentity",
    "GenerationConfig",
    "PolicyIdentity",
    "PolicyScore",
    "SampledGeneration",
    "TokenizerIdentity",
    "TrainablePolicy",
    "TransformersPolicy",
    "TransformersResponseGenerator",
]
