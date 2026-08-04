"""Standalone, Prime-free single-GPU Hugging Face training path."""

from rlm_train.colab.assembly import FixedSDPOComponents, build_fixed_sdpo_components
from rlm_train.colab.benchmark_config import (
    FINAL_ANSWER_PATTERN,
    build_benchmark_sdpo_config,
    write_colab_run_config,
)
from rlm_train.colab.checkpoint import TrainingCheckpointManager
from rlm_train.colab.config import (
    ColabProfile,
    ColabRunConfig,
    DatasetConfig,
    GenerationConfig,
    JudgeConfig,
    ModelConfig,
    OptimizationConfig,
    OutputConfig,
    Precision,
    Quantization,
    SDPORolloutConfig,
    TeacherResidency,
    TeacherRuntimeConfig,
)
from rlm_train.colab.generation import (
    PromptFormatter,
    TokenGenerationResult,
    TransformersCompletionAdapter,
    TransformersResponseGenerator,
)
from rlm_train.colab.gram import TransformersGramLossBuilder
from rlm_train.colab.objectives import ObjectiveComposer, RolloutSample, TrainingBatch
from rlm_train.colab.question_generation import TracedQuestionResponseGenerator
from rlm_train.colab.runtime import load_policy_bundle, validate_colab_runtime
from rlm_train.colab.teacher import (
    FileTeacherTargetCache,
    TransformersQuestionTeacherProvider,
    build_fixed_teacher_controller,
)
from rlm_train.colab.trainer import (
    MaskedQuestionSDPOLossBuilder,
    SingleGPUTrainer,
)
from rlm_train.colab.trajectory_sdpo import TrajectoryQuestionTargetProvider

__all__ = [
    "ColabProfile",
    "ColabRunConfig",
    "DatasetConfig",
    "FileTeacherTargetCache",
    "FINAL_ANSWER_PATTERN",
    "FixedSDPOComponents",
    "GenerationConfig",
    "JudgeConfig",
    "MaskedQuestionSDPOLossBuilder",
    "ModelConfig",
    "ObjectiveComposer",
    "OptimizationConfig",
    "OutputConfig",
    "Precision",
    "PromptFormatter",
    "Quantization",
    "RolloutSample",
    "SDPORolloutConfig",
    "SingleGPUTrainer",
    "TeacherResidency",
    "TeacherRuntimeConfig",
    "TokenGenerationResult",
    "TrainingBatch",
    "TrainingCheckpointManager",
    "TrajectoryQuestionTargetProvider",
    "TracedQuestionResponseGenerator",
    "TransformersCompletionAdapter",
    "TransformersGramLossBuilder",
    "TransformersQuestionTeacherProvider",
    "TransformersResponseGenerator",
    "build_fixed_teacher_controller",
    "build_benchmark_sdpo_config",
    "build_fixed_sdpo_components",
    "load_policy_bundle",
    "validate_colab_runtime",
    "write_colab_run_config",
]
