"""Expose local student generation to the core RLM execution engine."""

from rlm_train.models.transformers_runtime import (
    TransformersCompletionAdapter as BaseStudentRLMClient,
)


class StudentRLMClient(BaseStudentRLMClient):
    """Present the shared student generator to the core RLM as an LM client."""


__all__ = ["StudentRLMClient"]
