"""Feedback-conditioned self-distillation."""

from rlm_train.objectives.sdpo.divergence import reverse_kl_topk_with_tail
from rlm_train.objectives.sdpo.target_support import TopKTeacherTarget
from rlm_train.sdpo.calculate_loss import calculate_loss
from rlm_train.sdpo.feedback_predictions import FeedbackPredictions
from rlm_train.sdpo.prepare_feedback_prompt import prepare_feedback_messages
from rlm_train.sdpo.score_with_feedback import score_with_feedback
from rlm_train.sdpo.settings import SDPOSettings

__all__ = [
    "FeedbackPredictions",
    "SDPOSettings",
    "TopKTeacherTarget",
    "calculate_loss",
    "prepare_feedback_messages",
    "reverse_kl_topk_with_tail",
    "score_with_feedback",
]
