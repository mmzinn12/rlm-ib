# Full-RLM training

`rlm_train` trains and evaluates the complete recursive policy through the repository's
canonical `rlm.RLM` engine. Root generations, Python/REPL execution, plain helper calls,
recursive child RLMs, state, hierarchy, and final-answer submission all follow the same
runtime path used by inference.

Colab is a thin gateway over this package. It does not own attempt execution, the training
loop, token selection, or checkpoint formats.

## Architecture

The dependency direction is strict:

```text
CLI / gateways
    -> public API
    -> create training run
    -> training loop
       -> attempt runner -> rlm core
       -> judge -> feedback
       -> uncertainty -> per-edge quantitative measurements
       -> token selection
       -> student and feedback-conditioned scoring
       -> training-method loss
       -> optimizer and saved runs
```

The base `rlm` package has no PyTorch or `rlm_train` dependency. It exposes immutable
execution events and a neutral observer protocol in `rlm/core/events.py` and
`rlm/core/observers.py`. The observer is shared by recursive children, so a rollout has
one ordered event stream and one policy-owner identity across the complete invocation
tree.

The main training packages are:

- `api/`: stable `train()` and `evaluate()` entry points.
- `settings/`: immutable, serializable run configuration.
- `student/`: student loading, differentiable exact-token scoring, and model saving.
- `generation/`: chat formatting, exact sampled-ID generation, and the core-RLM client.
- `attempts/`: the full-RLM runner, event recorder, and annotated attempt records.
- `token_selection/`: semantic text regions, character/token matching, and per-method choices.
- `trajectory/`: durable annotated rollout schema, validation, projection, and replay.
- `feedback/` and `judge/`: minimal typed evidence views, visibility enforcement,
  scoped assessments, and one-way overall aggregation.
- `uncertainty/` and `engine/uncertainty_provider.py`: direct-answer student sampling,
  shared semantic clustering, probability-weighted entropy, and causal before/after edge
  measurements. This package never depends on judge feedback or objectives.
- `sdpo/`, `grpo/`, and `gram/`: method-specific settings and loss calculations. SDPO
  additionally owns feedback-prompt construction and detached feedback predictions.
- `datasets/` and `evaluation/`: public/private task separation and whole-policy scoring.
- `training/`, `metrics/`, and `saved_runs/`: the readable training loop, optimization,
  observations, provenance, checkpoints, and safe attempt JSON.
- `gateways/colab/`: CUDA/authentication/storage preflight followed by public API calls.

The former Colab-owned trainer, depth-one rollout path, experiment layer, and duplicate
objective, judge, benchmark, and trajectory implementations have been removed. Each
training concern now has one implementation under the cohesive packages above.

JSONL datasets use the same public boundary expected at production inference time. The user
question is shown directly to the root orchestrator, the evidence context is stored in the REPL,
and the target remains verifier-only:

```json
{"id":"q1","question":"What is being asked?","context":"Supporting evidence","target":"answer"}
```

Combined `prompt` records are rejected because they force the policy to discover the task inside
the evidence payload and create a train–production mismatch.

Hub datasets can be streamed directly with the optional dependency installed via
`uv sync --extra hub-datasets`. For HotpotQA, the adapter maps `id`, `question`, `context`, and
`answer` into the same boundary, renders titled evidence sections, and omits privileged
`supporting_facts`:

```toml
[training_dataset]
adapter = "hotpotqa"
source = "hotpotqa/hotpot_qa"
subset = "distractor"
split = "train"
max_records = 200
```

## RunSpec

A run is declared in TOML or JSON and passed to `create_training_run`. Direct collaborator
injection remains supported for tests and research.

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

[objectives.sdpo]
enabled = true
weight = 1.0
token_scope = "helper_questions"
feedback_scope = "retrospective_local"
divergence = "reverse_kl"
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
checkpoint_interval = 100
retain_checkpoints = 1
save_final_checkpoint = true

[uncertainty]
enabled = true
estimator = "semantic_entropy"
estimator_version = "semantic-entropy-v1-natural-log"
sample_count = 10
temperature = 0.5
top_p = 1.0
max_new_tokens = 32
prompt_version = "direct-answer-v1"
equivalence_provider = "transformers_nli"
equivalence_model = "microsoft/deberta-large-mnli"
# Replace this example with an immutable Hub commit for the selected model.
equivalence_model_revision = "<pinned-hub-commit>"
```

Every enabled objective selects its own token scope: `natural_language`,
`helper_questions`, `subcall_natural_language`, or `all_student_tokens`. Selections are
stored as explainable ranges and reconstructed as runtime masks. Only tokens owned by the
configured student can be active.

### Semantic uncertainty measurements

When enabled, semantic uncertainty is measured before the optimizer step that consumes a
rollout. For each helper edge, the provider asks the frozen student for matched direct-answer
samples under two prompts. The before prompt contains the public task, public context, and the
already-generated helper question; the after prompt differs only by revealing that helper's exact
completed response. Root final answers, future events, sibling results, and verifier references
are never included.

Samples retain exact continuation IDs and per-token behavior log probabilities. Answers from both
conditions are pooled into one question-conditioned semantic partition using bidirectional NLI
entailment. The implementation reports natural-log Shannon entropy reduction and Jensen-Shannon
distribution shift separately. Lower entropy is not treated as correctness or evidence quality;
misleading-information labels remain solely in judge feedback. There is no frequency-only fallback
when behavior log probabilities are unavailable.

The cost per assessed edge is `2 * sample_count` student generations plus semantic-equivalence
comparisons. The deterministic complete-link clustering rule can require quadratically many NLI
comparisons in the number of pooled samples. Use `max_edges_per_rollout` as an explicit, persisted
cost bound when needed. The method follows Kuhn, Gal, and Farquhar's
[semantic uncertainty estimator](https://arxiv.org/abs/2302.09664).

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
Both modes return the same `ScopedAssessment` boundary, so training methods do
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

The corresponding CLIs are `rlm-train` and `rlm-evaluate`. Default training explicitly creates
the student, attempt runner, feedback collector, training methods, optimizer, and saved-run
writers. The resolved settings and student identities are written before the first attempt.

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
