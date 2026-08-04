# Single-GPU Colab launcher

The notebook and launcher use only the local Transformers path. They do not install or
download Prime-RL, AIME24, or a local judge model.

In a fresh GPU runtime, clone the repository, open `rlm_training.ipynb`, and run the cells.
The smoke profile performs two short rollouts and one optimizer step, then writes an atomic
checkpoint. The train profile uses the same importable Python implementation with larger
limits. Resume the latest checkpoint with `--resume`, or provide an explicit checkpoint
directory after that flag.

Google Drive is optional. To use it, mount Drive in the notebook and set
`output.google_drive_root` in a copied config. Credentials are read only from the named
environment variable or Colab secret and are never serialized.

The command-line dataset launcher intentionally selects the policy-only GRPO arm. The
framework-local `SingleGPUTrainer` also accepts `MaskedQuestionSDPOLossBuilder` and
`TransformersGramLossBuilder` for traced question batches; those APIs require exact token
masks and therefore cannot silently turn question-local SDPO into whole-response
distillation. `build_fixed_sdpo_components` constructs the configured fake/OpenAI judge,
content-addressed caches, feedback projector, frozen teacher, and masked loss builder from
the top-level experiment config.
