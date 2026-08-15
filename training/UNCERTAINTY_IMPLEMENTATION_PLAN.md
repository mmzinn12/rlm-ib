# Semantic Uncertainty Implementation Plan

## Goal

Add a model-specific, reproducible measurement of how much a helper response changes and
reduces the student's uncertainty about the task answer. The implementation will follow the
semantic-entropy method introduced by Kuhn, Gal, and Farquhar in
[Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation](https://arxiv.org/abs/2302.09664).

The first implementation targets plain and recursive helper edges in annotated RLM rollouts. It
must keep four concepts separate:

- **Semantic uncertainty:** dispersion over distinct answer meanings.
- **Uncertainty reduction:** the change in semantic entropy after observing a helper response.
- **Belief change:** movement between semantic answer distributions, even when entropy is
  unchanged.
- **Evidence quality:** whether the helper response is supported, misleading, or correct. This
  remains a feedback/judge concern and must not be inferred from entropy alone.

## Architectural boundaries

```text
generation/                         Existing model generation machinery
    |
    v
uncertainty/                        Pure uncertainty domain and sampling contract
    |
    v
engine/uncertainty_provider.py      Before/after orchestration for rollout edges
    |
    +--> feedback/                  Qualitative evidence assessment and training projection
    +--> trajectory/                Durable per-edge uncertainty artifacts
    +--> metrics/                   Aggregate observability
```

The dependency rules are:

1. `uncertainty` must not depend on `feedback`, `judge`, objectives, or the training loop.
2. Entropy and clustering functions must not call a model directly.
3. Sampling adapters may use the existing `generation` and model interfaces.
4. The engine provider coordinates sampling and estimation for an edge.
5. Feedback may consume an uncertainty result, but uncertainty must never import feedback.
6. Missing probabilities, malformed clusters, and non-finite values fail loudly. There is no
   implicit fallback from probability-weighted entropy to frequency-only entropy.

## Proposed package layout

```text
training/src/rlm_train/
├── uncertainty/
│   ├── __init__.py
│   ├── schema.py
│   ├── protocols.py
│   ├── semantic_equivalence.py
│   ├── semantic_entropy.py
│   ├── sampling.py
│   └── prompts.py
├── engine/
│   └── uncertainty_provider.py
├── spec/
│   └── uncertainty.py
└── settings/
    └── uncertainty.py
```

### `uncertainty/schema.py`

Define immutable, serializable records:

- `SemanticSample`
  - sample ID
  - condition (`before` or `after`)
  - generated answer text
  - continuation token IDs
  - per-token log probabilities
  - sequence log probability
  - sampling seed
  - model and tokenizer identities
- `SemanticCluster`
  - stable cluster ID
  - member sample IDs
  - representative answer
  - log probability mass for each condition
- `SemanticEntropyEstimate`
  - estimator name and version
  - condition
  - entropy
  - sample count
  - cluster count
  - cluster probability distribution
  - model, tokenizer, sampling, and prompt provenance
- `UncertaintyReduction`
  - rollout and edge IDs
  - before and after estimates
  - absolute entropy reduction
  - normalized entropy reduction, when defined
  - semantic distribution shift
  - shared cluster partition fingerprint

All probability fields must be finite. Cluster probabilities must be non-negative and sum to one
within an explicit numerical tolerance. `normalized_entropy_reduction` should be `None` when the
before entropy is zero rather than silently substituting a misleading value.

### `uncertainty/protocols.py`

Define narrow interfaces so tests and research implementations can be injected:

```python
class AnswerSampler(Protocol):
    def sample(
        self,
        request: AnswerSamplingRequest,
    ) -> tuple[SemanticSample, ...]: ...


class SemanticEquivalenceClassifier(Protocol):
    def equivalent(
        self,
        question: str,
        left: str,
        right: str,
    ) -> bool: ...


class SemanticEntropyEstimator(Protocol):
    def estimate(
        self,
        samples: tuple[SemanticSample, ...],
        clusters: tuple[SemanticCluster, ...],
    ) -> SemanticEntropyEstimate: ...
```

The production sampler must use the student policy whose uncertainty is being measured. Using
the feedback judge would measure judge uncertainty instead of student uncertainty.

### `uncertainty/semantic_equivalence.py`

Implement question-conditioned, bidirectional-entailment clustering:

1. Compare answers in both entailment directions.
2. Treat them as equivalent only when both directions entail.
3. Cluster the pooled union of before and after samples so both distributions share the same
   semantic event space.
4. Assign stable cluster IDs from canonicalized member IDs rather than traversal order.
5. Record the equivalence model identity and revision.

The classifier must be injected behind `SemanticEquivalenceClassifier`. The initial production
adapter should reuse already-installed Transformers infrastructure and a pinned NLI model. Do not
add a new core dependency. Deterministic fake and exact-match classifiers will support unit tests.

The clustering implementation must document that learned pairwise equivalence may not be
perfectly transitive. The chosen deterministic cluster-construction rule and comparison order must
be stable and covered by tests.

### `uncertainty/semantic_entropy.py`

Implement pure numerical functions:

- Sum sequence probability mass within each semantic cluster using log-space operations.
- Normalize observed cluster mass explicitly.
- Calculate Shannon entropy over semantic clusters.
- Calculate before/after absolute entropy reduction:

  ```text
  delta_h = entropy_before - entropy_after
  ```

- Calculate normalized reduction when `entropy_before > 0`:

  ```text
  normalized_delta_h = delta_h / entropy_before
  ```

- Calculate Jensen-Shannon divergence between the aligned before and after cluster distributions
  to measure belief change independently of entropy reduction.

The implementation must define the logarithm base in its estimator version and output. Natural
logarithms should be the default to match the paper.

Frequency-only entropy may be added later as a separately named estimator. It must never be an
automatic fallback when sequence log probabilities are unavailable.

### `uncertainty/sampling.py`

Implement the uncertainty-specific adapter over existing generation machinery. It is responsible
for:

- generating multiple short, direct task answers rather than RLM CodeAct continuations;
- retaining exact continuation IDs and per-token behavior log probabilities;
- calculating sequence log probability without length normalization for the first implementation;
- deriving deterministic per-sample seeds from the run seed, rollout ID, edge ID, and sample index;
- using matched sample indices and seeds across before and after conditions;
- recording all sampling parameters in each request and estimate;
- rejecting providers that cannot return the required token log probabilities.

This module does not decide what evidence belongs in the before and after prompts. That is the
engine provider's responsibility.

### `uncertainty/prompts.py`

Define a versioned, minimal direct-answer probe. The prompt must label fields explicitly:

```text
TASK QUESTION:
{question}

SUPPORTING CONTEXT:
{context}

AVAILABLE HELPER INFORMATION:
{conditioned_helper_information}

Return only the shortest answer that resolves the task question.
```

The same prompt version and formatting must be used in both conditions. The only difference is
the helper response revealed by the intervention.

## Engine-level uncertainty sampling

Create `engine/uncertainty_provider.py` rather than a generic `engine/sampling.py`.

The provider owns the causal before/after experiment for each focal edge:

```python
class UncertaintyProvider(Protocol):
    def assess(
        self,
        record: DatasetRecord,
        rollout: AnnotatedRollout,
        edge_id: str,
    ) -> UncertaintyReduction: ...
```

The concrete `SemanticEntropyUncertaintyProvider` will:

1. Resolve the focal edge and reject unknown IDs.
2. Build a causal evidence view that excludes the root final answer, downstream events, sibling
   results, and verifier-owned references.
3. Construct the **before** condition from the public task and trajectory visible immediately
   before the helper response is received. The helper question may be included because it was
   already generated; its response must not be included.
4. Construct the **after** condition from the identical view plus the exact completed helper
   response.
5. Produce matched before and after sampling requests.
6. Sample both conditions with the same frozen student checkpoint and sampling configuration.
7. Pool and semantically cluster both sample sets.
8. Estimate both semantic distributions on the shared clusters.
9. Calculate entropy reduction and Jensen-Shannon distribution shift.
10. Return a provenance-bearing `UncertaintyReduction`.

This measures the information supplied by the helper response. If the desired research question
later changes to the value of the entire helper action, add a separately named intervention that
compares state before the helper question with state after its response.

## Configuration

Add an immutable `UncertaintySpec` and include it in `RunSpec`:

```toml
[uncertainty]
enabled = true
estimator = "semantic_entropy"
estimator_version = "semantic-entropy-v1"
sample_count = 10
temperature = 0.5
max_new_tokens = 32
prompt_version = "direct-answer-v1"
equivalence_provider = "transformers_nli"
equivalence_model = "pinned-model-id"
equivalence_model_revision = "pinned-revision"
```

Validation requirements:

- `sample_count >= 2`.
- Temperature must be positive.
- Model revisions must be pinned for production runs.
- Semantic entropy requires behavior log probabilities.
- An enabled uncertainty-dependent objective requires uncertainty measurement to be enabled.
- Configuration fingerprints must include prompt, estimator, equivalence model, and sampling
  versions.

The paper's temperature `0.5` and sample count `10` are reasonable initial defaults, but they are
hyperparameters rather than universal constants and must remain configurable.

## Runtime and factory integration

1. Add an optional `uncertainty` component to `ResolvedComponents`.
2. Add a registered builder for the production sampler, equivalence classifier, estimator, and
   engine provider.
3. Permit direct injection of deterministic implementations in tests and research runs.
4. Make the provider use the same student identity and frozen parameter state for both conditions.
5. Run uncertainty sampling outside gradient tracking.
6. Decide explicitly whether measurements are taken before or after an optimizer step; the default
   should be before the step that consumes the rollout.

## Feedback, trajectory, and artifact integration

Uncertainty is quantitative evidence, not judge prose. Add a distinct transport path rather than
placing it in `judge_assessments`:

- Add `uncertainty_assessments` to `FeedbackBundle` or introduce a sibling measurement bundle.
- Add `uncertainty_assessments` to `trajectory.schema.FeedbackRecord` with an empty default for
  compatible loading of existing artifacts.
- Store one serialized `UncertaintyReduction` per rollout edge.
- Preserve the existing qualitative judge fields for evidence quality, misleading information,
  missing information, and improved-question guidance.
- Allow feedback projections and teacher-target builders to read entropy reduction as an
  authoritative quantitative feature.
- Never overwrite judge evidence-quality labels with entropy-derived values.

Before finalizing this boundary, add a schema test proving that old rollout JSON without the new
field still loads and new rollout JSON round-trips exactly.

## Metrics

Record aggregate observations without moving calculation into `metrics`:

- `uncertainty/semantic_entropy_before`
- `uncertainty/semantic_entropy_after`
- `uncertainty/entropy_reduction`
- `uncertainty/normalized_entropy_reduction`
- `uncertainty/semantic_distribution_shift`
- `uncertainty/cluster_count_before`
- `uncertainty/cluster_count_after`
- `uncertainty/sampling_seconds`
- `uncertainty/equivalence_seconds`

Metrics must be tagged or attributable by rollout ID, edge ID, checkpoint identity, estimator
version, and prompt version where the metrics schema permits.

## Caching and cost controls

Uncertainty sampling adds multiple student generations and pairwise equivalence checks per edge.
Implement caching after correctness is established:

- Cache sampled answer sets by model identity, prompt fingerprint, sampling configuration, and
  seeds.
- Cache pairwise equivalence decisions by question fingerprint, normalized answer pair, model
  identity, and revision.
- Cache complete estimates by the fingerprints of samples, clusters, and estimator version.
- Keep cache keys content-addressed and never reuse values across student checkpoints.
- Bound the number of assessed edges per rollout through explicit configuration if needed.

Do not silently skip measurements to save cost. A configured limit must be visible in resolved run
settings and artifact provenance.

## Testing strategy

### Unit tests

Add deterministic tests covering:

- semantically equivalent paraphrases collapsing into one cluster;
- contradictory answers remaining in different clusters;
- bidirectional rather than one-way entailment;
- stable clustering under deterministic input reordering;
- log-space cluster probability aggregation;
- known entropy values for one-, two-, and three-cluster distributions;
- positive, zero, and negative entropy reduction;
- undefined normalized reduction when before entropy is zero;
- Jensen-Shannon divergence symmetry and bounds;
- rejection of missing or non-finite log probabilities;
- exact schema serialization and fingerprints;
- deterministic matched seed derivation;
- strict exclusion of final answers, future events, and verifier references from before/after
  prompts.

### Integration tests

Build a small fake student with predetermined samples:

1. **Useful helper:** before samples disagree; after samples converge on the supported answer.
2. **Redundant helper:** before and after semantic distributions are identical.
3. **Confusing helper:** after samples spread across more meanings.
4. **Confident misinformation:** entropy falls while the existing judge marks the response
   misleading. This proves the two signals remain independent.
5. **Belief replacement:** before and after entropies match but the likely semantic answer changes.
   Jensen-Shannon divergence must detect the shift.

Add one full training-stack test showing that the result is attached to the correct edge, persisted
in the rollout artifact, available to feedback projection, and absent when uncertainty is disabled.

### Empirical validation

Before using entropy reduction as a training target:

1. Run it over a held-out set of successful and failed helper calls.
2. Measure correlation with change in final-answer correctness.
3. Compare semantic entropy against cheaper baselines:
   - number of semantic clusters;
   - lexical disagreement;
   - frequency-only semantic entropy;
   - current judge uncertainty label.
4. Ablate sample counts such as 3, 5, 10, and 20.
5. Ablate temperature while holding other sampling settings fixed.
6. Inspect cases where entropy falls but correctness worsens.
7. Select thresholds or scalarization only after the held-out evaluation.

## Implementation phases

### Phase 1: Pure uncertainty core

- Add schemas and protocols.
- Implement deterministic clustering with an injected equivalence classifier.
- Implement probability-weighted semantic entropy, normalized reduction, and Jensen-Shannon
  divergence.
- Add exhaustive unit tests.

**Exit criterion:** all numerical and clustering tests pass without importing engine, feedback, or
training-loop modules.

### Phase 2: Student answer sampler

- Add the direct-answer prompt contract.
- Adapt existing Transformers generation to return multiple samples with behavior log
  probabilities.
- Add deterministic seed derivation and provenance.
- Add fake sampler integration tests.

**Exit criterion:** repeated requests with identical configuration reproduce the same sample IDs,
tokens, probabilities, and fingerprints.

### Phase 3: Engine provider

- Implement causal before/after view construction.
- Add matched sampling, pooled clustering, estimation, and reduction.
- Add leakage and intervention-boundary tests.

**Exit criterion:** the five integration scenarios produce the expected entropy and distribution
shift behavior.

### Phase 4: Runtime and artifact integration

- Add `UncertaintySpec`, factory construction, direct injection, and resolved provenance.
- Persist per-edge uncertainty assessments separately from judge assessments.
- Emit aggregate metrics.
- Update renderers to display concise before/after entropy and belief-shift summaries.

**Exit criterion:** a saved rollout round-trips with uncertainty provenance and existing rollouts
remain loadable.

### Phase 5: Feedback and training use

- Expose uncertainty measurements to feedback projection and teacher-target construction.
- Keep evidence validity as a hard, separate signal.
- Add any entropy-based weighting or objective only after empirical validation.

**Exit criterion:** training behavior is explicitly configured, tested, and never rewards confident
misinformation solely because entropy decreased.

## Non-goals for the initial implementation

- Treating entropy reduction as correctness or evidence quality.
- Adding a generic sampling subsystem under `engine`.
- Automatically falling back when token probabilities are unavailable.
- Supporting arbitrary hosted APIs without token log-probability access.
- Training or fine-tuning the semantic-equivalence model.
- Adding entropy directly to the base `rlm` inference package.
- Using entropy reduction as an objective before held-out validation.

## Documentation updates

After implementation:

- Add `uncertainty/` and `engine/uncertainty_provider.py` to the architecture section of
  `training/README.md`.
- Add a complete `[uncertainty]` configuration example.
- Document the precise before/after intervention boundary.
- Cite the semantic-entropy paper and distinguish the implemented estimator from any later
  frequency-only approximation.
- Document runtime and memory cost as a function of sample and cluster counts.

## Completion checklist

- [ ] Package boundaries and dependency direction are enforced.
- [ ] Semantic samples retain exact token IDs and behavior log probabilities.
- [ ] Before and after conditions use the same student checkpoint and matched settings.
- [ ] Semantic clustering is question-conditioned and provenance-bearing.
- [ ] Entropy is calculated in log space over aligned semantic clusters.
- [ ] Entropy reduction and belief shift remain separate measurements.
- [ ] Evidence validity remains independent judge feedback.
- [ ] Causal views cannot leak future or verifier-owned information.
- [ ] Configuration, prompts, models, seeds, and estimator versions are fingerprinted.
- [ ] Existing rollout artifacts remain loadable.
- [ ] Unit, integration, full-stack, and empirical validation tests are complete.
- [ ] Ruff, formatting, pre-commit, and the full test suites pass.
