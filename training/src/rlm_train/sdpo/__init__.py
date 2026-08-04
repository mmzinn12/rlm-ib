"""Expose the stable public API for framework-neutral tree-aware SDPO execution.

Purpose:
    Provide one import surface for configuration, EMA lifecycle, isolated question
    scoring, target caching, top-k/tail gathering, and weighted reverse-KL losses.
Implementation:
    This facade re-exports provider- and trainer-independent types from the SDPO package.
Inputs:
    Python imports from a teacher service or pinned trainer adapter.
Outputs:
    Public configuration, target, controller, scorer, and loss symbols.
Example:
    ``from rlm_train.sdpo import SDPOConfig, reverse_kl_topk_with_tail``
"""

from rlm_train.sdpo.cache import (
    MemoryTeacherTargetCache,
    TeacherTargetCache,
    make_question_teacher_cache_key,
)
from rlm_train.sdpo.config import ComponentWeights, SDPOConfig, TeacherStrategy
from rlm_train.sdpo.loss import (
    StudentTopKDistribution,
    WeightedSDPOLoss,
    gather_student_topk_with_tail,
    reverse_kl_topk_with_tail,
    teacher_target_tensors,
    weighted_component_reverse_kl,
)
from rlm_train.sdpo.masks import build_question_token_mask
from rlm_train.sdpo.prime_adapter import PrimeQuestionSDPOFields, PrimeTreeSDPOFields
from rlm_train.sdpo.teacher import (
    EMATeacherController,
    QuestionTeacherLogitsProvider,
    QuestionTeacherScorer,
    TeacherIdentity,
    TeacherScorer,
    TopKQuestionTeacherScorer,
    TopKTeacherTarget,
    TorchEMATeacherController,
    TorchFixedTeacherController,
    build_question_feedback_context,
    build_torch_teacher_controller,
    extract_topk_teacher_target,
    model_state_fingerprint,
    validate_question_teacher_example,
)

__all__ = [
    "ComponentWeights",
    "EMATeacherController",
    "MemoryTeacherTargetCache",
    "QuestionTeacherScorer",
    "QuestionTeacherLogitsProvider",
    "PrimeQuestionSDPOFields",
    "PrimeTreeSDPOFields",
    "SDPOConfig",
    "StudentTopKDistribution",
    "TeacherScorer",
    "TeacherTargetCache",
    "TeacherIdentity",
    "TeacherStrategy",
    "TopKQuestionTeacherScorer",
    "TopKTeacherTarget",
    "TorchEMATeacherController",
    "TorchFixedTeacherController",
    "WeightedSDPOLoss",
    "build_question_feedback_context",
    "build_torch_teacher_controller",
    "build_question_token_mask",
    "extract_topk_teacher_target",
    "gather_student_topk_with_tail",
    "make_question_teacher_cache_key",
    "model_state_fingerprint",
    "reverse_kl_topk_with_tail",
    "teacher_target_tensors",
    "validate_question_teacher_example",
    "weighted_component_reverse_kl",
]
