"""Settings for enabled training methods."""

from rlm_train.spec.objectives import GramSpec, GRPOSpec, ObjectivesSpec, SDPOSpec

GRPOSettings = GRPOSpec
GramSettings = GramSpec
SDPOSettings = SDPOSpec
TrainingMethodsSettings = ObjectivesSpec

__all__ = ["GRPOSettings", "GramSettings", "SDPOSettings", "TrainingMethodsSettings"]
