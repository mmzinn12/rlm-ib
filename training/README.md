# Training local REPL RLM with `prime-rl`

Verifiers-compatible training harness for `rlm.RLM` at depth=1. Designed to plug directly into [`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl) for end-to-end RL training of RLM policies. Does not require or use sandboxes.

This harness runs rollouts through a local REPL backend (subprocess-isolated, no cloud sandboxes), so it shells out to the same `LocalREPL`-style execution surface described in the [main README](../README.md#local-environments). It corresponds to the `local` environment in `rlm` — sub-LM calls are routed back through a proxy to the trainer's inference server, while Python code execution happens in a subprocess on the training host.

- `src/rlm_train/` — env, rubric, sub-LM proxy, subprocess REPL worker.
- `environments/oolong/` — OOLONG synth long-context QA env (example).
- `configs/rlm-qwen3-30b-example.toml` — example RL config.

## Launching a training run

With `prime-rl` installed and this directory's environment set up, launch with:

```bash
uv run rl @ training/configs/rlm-qwen3-30b-example.toml
```

The config wires the `oolong` environment into `prime-rl`'s orchestrator/trainer/inference loop. See [`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl) for distributed launch options and deployment details.

## Tree-SDPO core

The repository includes modular, opt-in framework-neutral execution for trajectory-aware
SDPO. It does not change the current Prime reward objective or select/launch a model yet.

- `src/rlm_train/trajectory/` records depth-1 root/subcall trees, segments response
  spans, binds literal question-list items to runtime children, and compiles node-level
  or edge-isolated question examples. Versioned JSONL artifacts and offline replay make
  completed rollouts reusable without invoking the student again.
- `src/rlm_train/judge/` defines strict structured feedback schemas, judge prompts,
  a bounded-retry structured judge, persistent SQLite caching, and a privileged-context
  boundary. Subcalls are scored for the significance of information they reveal
  (novelty, uncertainty reduction, and evidence quality), never for how much they
  contributed to the final answer. `information_significance` is the signed
  reward/penalty placeholder; its calibration and weighting are intentionally deferred.
- `src/rlm_train/sdpo/` defines fixed and EMA teacher lifecycles, edge-isolated question
  scoring, teacher top-k/student-tail extraction,
  target caching, exclusive masks, normalized weighted reverse-KL losses, metrics, and
  dependency-free payloads for the Prime integration boundary.
- `src/rlm_train/regularization/` is an independent Gram-anchor objective: detached JS
  drift scoring, deterministic bounded token sampling, selected-layer Gram losses,
  anchor lifecycle contracts, diagnostics, and a thin Prime seam.
- `src/rlm_train/experiment/` composes those pieces into one immutable resolved config,
  validates incompatible combinations, resolves the seven initial ablation presets, and
  persists complete run/checkpoint provenance.
- `src/rlm_train/diagnostics/` records epistemic markers, reconsideration behavior,
  trajectory topology, teacher divergence, truncation, and Gram observations without
  exposing any value to prompts, rewards, sampling, or losses.
- `src/rlm_train/benchmarks/` supplies the generic benchmark protocol, a content-addressed
  JSONL adapter, registry, overlap checks, deterministic sampling, durable resume records,
  and `acc@k`/`pass@k` reports.

The proposed/default configuration uses reverse KL, teacher top-k with an explicit tail
bucket, a fixed copy of the initial policy, diagnostic edge-local feedback, the same
tokenizer for teacher and student, exclusive component masks,
and maximum recursion depth 1. Privileged judge context is optional and unset by
default; when supplied, only its source/version/fingerprint descriptor is persisted.
Additive masks remain a later extension.

Detailed package guides:

- [`trajectory`](src/rlm_train/trajectory/README.md): runtime trace construction, stable
  tree organization, segmentation, and compilation.
- [`judge`](src/rlm_train/judge/README.md): node-addressable feedback, information-value
  semantics, schemas, prompts, and caching.
- [`sdpo`](src/rlm_train/sdpo/README.md): configuration, masks, fixed/EMA teacher contracts,
  top-k-plus-tail targets, reverse KL, and the Prime integration boundary.
- [`regularization`](src/rlm_train/regularization/README.md): Gram configuration,
  reference alignment, sampling, representation loss, anchor lifecycle, and metrics.

Install tensor regularization only when needed:

```bash
uv sync --extra dev --extra regularization
```

Compile stored trajectory artifacts without another student rollout:

```bash
uv run rlm-train-replay path/to/rollouts.jsonl
```

## Download-free OOD experiment dry run

`configs/ood-robust-synthetic.toml` is a fully resolved local experiment using
`benchmarks/synthetic-arithmetic.jsonl`. It exercises configuration, diagnostic,
benchmark, resume, and reporting boundaries without downloading a model or dataset.

The AIME24 adapter/data and the concrete version-pinned Prime trainer bridge are
intentionally deferred. `prime-rl` is not a dependency of these framework-neutral
packages, and no lockbox evaluation is performed by the dry run.

Run the complete synthetic path with:

```bash
uv run rlm-train-ood-dry-run \
  training/configs/ood-robust-synthetic.toml \
  --output training/outputs/ood-robust-synthetic-dry-run
```

## Examples
* [Qwen3-30B-A3B-Instruct-0527] on the original suite of tasks: [https://huggingface.co/mit-oasys/rlm-qwen3-30b-a3b-v0.1](https://huggingface.co/mit-oasys/rlm-qwen3-30b-a3b-v0.1)
