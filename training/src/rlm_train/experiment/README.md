# Experiment configuration and lifecycle

`ExperimentConfig` is the immutable top-level composition boundary for training,
teacher strategy, teacher-visible feedback, SDPO, Gram anchoring, benchmarks, checkpoint
schedule, and observer diagnostics. JSON and TOML are supported with the standard
library; every default is materialized by `resolved_dict()` and content-addressed by
`fingerprint`.

`resolve_ablation_preset()` supplies the initial base, GRPO, fixed/EMA factual SDPO,
scalar SDPO, diagnostic SDPO, and diagnostic-plus-Gram arms. Presets are conveniences,
not hidden settings: each resolves into the same explicit component fields.

`RunArtifactStore` writes the full resolved configuration into `run.json` and every
checkpoint provenance record. Checkpoints also carry teacher, projector, anchor, and
benchmark versions. Reinitializing a run with different resolved configuration fails.
