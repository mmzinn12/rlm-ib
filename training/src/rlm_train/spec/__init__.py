"""Public declarative configuration surface."""

from rlm_train.spec.artifacts import ArtifactSpec
from rlm_train.spec.evaluation import EvaluationSpec
from rlm_train.spec.feedback import AssessmentScope, FeedbackSpec
from rlm_train.spec.models import JudgeMode, JudgeSpec, StudentSpec, TeacherSpec, TeacherStrategy
from rlm_train.spec.objectives import (
    GramSpec,
    GRPOSpec,
    ObjectivesSpec,
    SDPOSpec,
    TokenScope,
)
from rlm_train.spec.rollout import RolloutSpec
from rlm_train.spec.run import DatasetRefSpec, RunSpec, RuntimeSpec
from rlm_train.spec.uncertainty import UncertaintySpec

__all__ = [
    "ArtifactSpec",
    "AssessmentScope",
    "DatasetRefSpec",
    "EvaluationSpec",
    "FeedbackSpec",
    "GRPOSpec",
    "GramSpec",
    "JudgeMode",
    "JudgeSpec",
    "ObjectivesSpec",
    "RolloutSpec",
    "RunSpec",
    "RuntimeSpec",
    "SDPOSpec",
    "StudentSpec",
    "TeacherSpec",
    "TeacherStrategy",
    "TokenScope",
    "UncertaintySpec",
]
