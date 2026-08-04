# Single-GPU Colab launcher

The notebook and launcher use only the local Transformers path. They do not install or
download Prime-RL or a local judge model. Dataset preparation is an explicit notebook
step; it is not performed by importing the training package or launching a run.

In a fresh GPU runtime, clone the repository, open `rlm_training.ipynb`, and run the cells.
The smoke profile performs two short rollouts and one optimizer step, then writes an atomic
checkpoint. The train profile uses the same importable Python implementation with larger
limits. Resume the latest checkpoint with `--resume`, or provide an explicit checkpoint
directory after that flag.

The optional dataset cells download pinned revisions of `HuggingFaceH4/aime_2024` and
`HuggingFaceH4/MATH-500`, then create deterministic 24/6 and 400/100 train/test splits.
These are project-local partitions of upstream benchmark pools, not official upstream
train/test splits. Training on either pool changes what a score on its held-out partition
means, so manifests retain the exact membership and source revision.
They expose `AIME24_TRAIN_PATH`, `AIME24_TEST_PATH`, `MATH500_TRAIN_PATH`, and
`MATH500_TEST_PATH`, plus count, fingerprint, manifest, repository, and revision variables.
The JSONL files and their deterministic manifests are stored under
`/content/drive/MyDrive/rlm-ib-datasets` by default, so they survive runtime deletion.
Reruns byte-validate existing artifacts instead of silently replacing a changed snapshot.
Preparation never rewrites the active run config; the existing smoke command remains
synthetic until a copied config is pointed at the exposed paths.

The same preparation is available outside the notebook:

```bash
pip install -e './training[hub-datasets]'
rlm-train-prepare-benchmarks all --output-root /path/to/rlm-ib-datasets
```

`colab-train.toml` preserves the sparse exact-match GRPO baseline. For a tiny-model
operational run, `colab-train-shaped.toml` uses a clearly labeled verifier-local numeric
proximity reward and loose last-integer extraction. Repeated problem occurrences receive
fresh deterministic seeds in both profiles so a zero-reward group is not replayed forever.
`colab-sdpo-smoke.toml` is a one-step pure-SDPO integration run: it sets
`policy_weight = 0` and `sdpo_weight = 1` and records one explicit helper-question
parent/child edge per rollout.

Google Drive is optional. To use it, mount Drive in the notebook and set
`output.google_drive_root` in a copied config. Credentials are read only from the named
environment variable or Colab secret and are never serialized.

The command-line launcher selects GRPO or fixed-teacher SDPO from the experiment config.
For SDPO it samples exactly one helper question, executes one child response, records the
real parent/child edge, and applies the reverse-KL only to exact parent question tokens.
Ordinary final-answer generation remains separate and is used for held-out evaluation.
`build_fixed_sdpo_components` constructs the configured fake/OpenAI judge,
content-addressed caches, feedback projector, frozen teacher, and masked loss builder from
the top-level experiment config. The deterministic fake judge is useful only for pipeline
validation because its diagnostic is target-independent; use the API judge when feedback
must assess the verifier-owned target.

Notebook code can create isolated AIME24 and MATH-500 configs without editing TOML:

```python
from rlm_train.colab import build_benchmark_sdpo_config, write_colab_run_config

aime_config = build_benchmark_sdpo_config(AIME24_SPLITS, run_name="aime24-sdpo")
aime_path = write_colab_run_config(aime_config, "/content/aime24-sdpo.json")
```

Every optimizer step is printed immediately as JSON. For a healthy pure-SDPO run,
`loss/sdpo` and `optimizer/gradient_norm` are positive while `loss/policy` and
`tokens/active_policy` are zero.
