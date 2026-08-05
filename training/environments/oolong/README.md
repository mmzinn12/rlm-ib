# OOLONG

This directory retains the original OOLONG synthetic long-context QA environment as a
reference for a future canonical `rlm_train` dataset and scorer integration. Its dataset
filtering, prompt construction, and answer scoring logic are preserved in `oolong/env.py`.

The current `load_environment()` entry point still targets the former Verifiers-based
`RLMTrainEnv` compatibility layer, which was removed when training moved to the canonical
full-RLM architecture. It is therefore a migration example, not a runnable training entry
point today. Before training on OOLONG, adapt `_build_dataset()` to the
`rlm_train.datasets.Dataset` protocol and `_synth_score()` to the
`rlm_train.evaluation.Scorer` protocol, then register those components with the runtime
factory.
