# Full-RLM training

`rlm_train` trains and evaluates the complete recursive policy through the repository's
canonical `rlm.RLM` engine. Root generations, Python/REPL execution, plain helper calls,
recursive child RLMs, state, hierarchy, and final-answer submission all follow the same
runtime path used by inference.

Colab is a thin gateway over this package. It does not own a rollout algorithm, trainer,
objective, trajectory schema, or checkpoint format.

## Architecture

The dependency direction is strict:

```text
CLI / gateways
    -> public API
    -> runtime factory
    -> training or evaluation engine
       -> rollout adapter -> rlm core
       -> judge -> feedback
       -> teachers
       -> objectives
       -> metrics and artifacts
```

The base `rlm` package has no PyTorch or `rlm_train` dependency. It exposes immutable
execution events and a neutral observer protocol in `rlm/core/events.py` and
`rlm/core/observers.py`. The observer is shared by recursive children, so a rollout has
one ordered event stream and one policy-owner identity across the complete invocation
tree.

The main training packages are:

- `api/`: stable `train()` and `evaluate()` entry points.
- `spec/`: immutable, serializable `RunSpec` configuration.
- `models/`: exact sampled-ID generation and rescoring contracts.
- `rollouts/`: canonical RLM adapter, event recorder, structural semantics, and
  per-objective token selectors.
- `trajectory/`: durable annotated rollout schema, validation, projection, and replay.
- `feedback/` and `judge/`: minimal typed evidence views, visibility enforcement,
  scoped assessments, and one-way overall aggregation.
- `teachers/`: current-policy, fixed, and EMA strategies with exact-ID targets.
- `objectives/`: independent GRPO, SDPO, and Gram packages plus the sole composer.
- `datasets/` and `evaluation/`: public/private task separation and whole-policy scoring.
- `engine/`, `runtime/`, `metrics/`, and `artifacts/`: optimization, construction,
  observations, provenance, and safe rollout JSON.
- `gateways/colab/`: CUDA/authentication/storage preflight followed by public API calls.

The former Colab-owned trainer, depth-one rollout path, experiment layer, and duplicate
objective, judge, benchmark, and trajectory implementations have been removed. Each
training concern now has one implementation under the cohesive packages above.

## RunSpec

A run is declared in TOML or JSON and resolved into concrete protocol implementations by
the runtime factory. Direct component injection is supported for tests and research.

```toml
[student]
adapter = "transformers"
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
trainable = true

[rollout]
engine = "rlm"
environment = "local"
max_depth = 2
max_iterations = 20

[teacher]
strategy = "current_policy"
feedback_conditioning = true

[objectives.sdpo]
enabled = true
weight = 1.0
token_scope = "helper_questions"
feedback_scope = "retrospective_local"
divergence = "forward_kl"
target_support = "top_k_with_tail"

[training_dataset]
adapter = "jsonl"
source = "training/benchmarks/synthetic-arithmetic.jsonl"
split = "train"

[evaluation]
recursive_policy = true
benchmarks = ["synthetic-arithmetic"]

[artifacts]
rollout_json = "all"
metrics_jsonl = true
```

Every enabled objective selects its own token scope: `natural_language`,
`helper_questions`, `subcall_natural_language`, or `all_student_tokens`. Selections are
stored as explainable ranges and reconstructed as runtime masks. Only tokens owned by the
configured student can be active.

### Configurable LLM judge output

The judge is a RunSpec component. OpenAI-compatible endpoints support a reliable
categorical contract and the original bounded numeric contract:

```toml
[judge]
provider = "openai"
model = "Qwen/Qwen2.5-7B-Instruct:together"
model_revision = "pinned-revision"
mode = "categorical" # or "full"
schema = "edge-information-v1"
prompt_version = "edge-information-v1"
api_key_environment = "OPENAI_API_KEY"
base_url = "https://router.huggingface.co/v1"
max_attempts = 3
```

`categorical` asks the LLM for enum labels and maps those labels deterministically to
bounded numeric values. `full` asks the LLM to emit the bounded numeric values directly.
Both modes return the same `ScopedAssessment` boundary, so objectives and teachers do
not branch on provider output format. Register the configured component with
`rlm_train.runtime.register_judge_builder(factory)` or inject a `Judge` directly for
research runs.

## Entry points

```python
from rlm_train import RunSpec, evaluate, train

spec = RunSpec.from_file("training/configs/full-rlm-example.toml")
train(spec, components=injected_components)
evaluate(spec, components=injected_components)
```

The corresponding CLIs are `rlm-train` and `rlm-evaluate`. A concrete deployment either
registers component builders on `ComponentFactory` or injects resolved components. The
fully resolved specification and component identities are written before the first
rollout.

Fresh training runs require an empty output directory and always write a final Transformers
checkpoint under `checkpoints/`. Configure `artifacts.checkpoint_interval` and
`artifacts.retain_checkpoints` for periodic retention. Resume and evaluate explicitly from a
saved checkpoint:

```bash
rlm-train run.json --resume-from outputs/run/checkpoints/step-00000025
rlm-evaluate run.json --checkpoint outputs/run/checkpoints/step-00000400
```

## Environment examples

`environments/oolong/` preserves the original OOLONG synthetic long-context dataset,
prompt construction, and answer-scoring implementation. Its loader still targets the
former Verifiers compatibility layer and is retained as an explicit migration example;
see its README for the canonical dataset and scorer protocols it needs before use.

## Tests

From the repository root:

```bash
uv run pytest
cd training && uv run pytest
```

The training suite includes a deterministic full-RLM integration rollout containing a
root generation, code execution, a plain helper response, a recursive child with its own
REPL state, and root final-answer aggregation.
