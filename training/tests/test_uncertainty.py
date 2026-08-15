from __future__ import annotations

import math

import pytest

from rlm_train.datasets.records import DatasetRecord
from rlm_train.engine.uncertainty_provider import SemanticEntropyUncertaintyProvider
from rlm_train.spec.uncertainty import UncertaintySpec
from rlm_train.trajectory.schema import (
    AnnotatedRollout,
    ExecutionEdge,
    ExecutionNode,
    ExecutionRecord,
    FeedbackRecord,
    NodeRole,
    TaskPartition,
)
from rlm_train.uncertainty.sampling import matched_seeds
from rlm_train.uncertainty.schema import SemanticSample, SemanticSampleBatch
from rlm_train.uncertainty.semantic_entropy import (
    ProbabilityWeightedSemanticEntropyEstimator,
    calculate_uncertainty_reduction,
    jensen_shannon_divergence,
    shannon_entropy,
)
from rlm_train.uncertainty.semantic_equivalence import (
    ExactMatchEquivalenceClassifier,
    cluster_semantic_samples,
)


def sample(sample_id: str, answer: str, probability: float, seed: int = 0):
    value = math.log(probability)
    return SemanticSample(
        sample_id=sample_id,
        answer=answer,
        continuation_token_ids=(1,),
        token_log_probabilities=(value,),
        sampling_seed=seed,
    )


def batch(condition: str, samples: tuple[SemanticSample, ...]) -> SemanticSampleBatch:
    aligned = tuple(
        item.model_copy(update={"sampling_seed": index}) for index, item in enumerate(samples)
    )
    return SemanticSampleBatch(
        condition=condition,
        samples=aligned,
        model_identity="student-checkpoint",
        tokenizer_identity="tokenizer",
        sampling_parameters={"temperature": 0.5},
        prompt_fingerprint="a" * 64,
        prompt_version="direct-answer-v1",
    )


def test_entropy_known_values_and_invalid_distributions():
    assert shannon_entropy((1.0,)) == 0.0
    assert shannon_entropy((0.5, 0.5)) == pytest.approx(math.log(2))
    assert shannon_entropy((1 / 3, 1 / 3, 1 / 3)) == pytest.approx(math.log(3))
    with pytest.raises(ValueError, match="sum to one"):
        shannon_entropy((0.2, 0.2))


def test_jensen_shannon_is_symmetric_bounded_and_detects_belief_replacement():
    left = {"a": 0.8, "b": 0.2}
    right = {"a": 0.2, "b": 0.8}
    shift = jensen_shannon_divergence(left, right)
    assert shift == pytest.approx(jensen_shannon_divergence(right, left))
    assert 0 < shift <= math.log(2)


def test_clustering_collapses_paraphrases_and_is_stable_under_reordering():
    class ParaphraseClassifier:
        provenance = {"provider": "fake", "model": "fake", "revision": "v1"}

        def equivalent(self, question, left, right):
            del question
            groups = [{"paris", "the city of paris"}, {"london"}]
            return any(left.casefold() in group and right.casefold() in group for group in groups)

    batches = (
        batch(
            "before",
            (sample("b-2", "London", 0.2), sample("a-1", "Paris", 0.4)),
        ),
        batch("after", (sample("a-2", "The city of Paris", 0.4),)),
    )
    forward = cluster_semantic_samples("capital?", batches, ParaphraseClassifier())
    reordered = tuple(
        item.model_copy(update={"samples": tuple(reversed(item.samples))})
        for item in reversed(batches)
    )
    reverse = cluster_semantic_samples("capital?", reordered, ParaphraseClassifier())
    assert forward == reverse
    assert sorted(len(cluster.member_sample_ids) for cluster in forward) == [1, 2]


def test_probability_weighted_estimation_and_reduction_signs():
    classifier = ExactMatchEquivalenceClassifier()
    estimator = ProbabilityWeightedSemanticEntropyEstimator()
    before_batch = batch("before", (sample("before-a", "A", 0.5), sample("before-b", "B", 0.5)))
    after_batch = batch("after", (sample("after-a1", "A", 0.5), sample("after-a2", "A", 0.5)))
    clusters = cluster_semantic_samples("q", (before_batch, after_batch), classifier)
    before = estimator.estimate(before_batch, clusters)
    after = estimator.estimate(after_batch, clusters)
    result = calculate_uncertainty_reduction(
        rollout_id="r",
        edge_id="e",
        before=before,
        after=after,
        clusters=clusters,
        equivalence_provenance=classifier.provenance,
    )
    assert result.absolute_entropy_reduction == pytest.approx(math.log(2))
    assert result.normalized_entropy_reduction == pytest.approx(1.0)
    assert result.semantic_distribution_shift > 0

    zero_batch = batch("before", (sample("only", "A", 1.0),))
    zero = estimator.estimate(zero_batch, cluster_semantic_samples("q", (zero_batch,), classifier))
    zero_result = calculate_uncertainty_reduction(
        rollout_id="r",
        edge_id="z",
        before=zero,
        after=after,
        clusters=clusters,
        equivalence_provenance=classifier.provenance,
    )
    assert zero_result.normalized_entropy_reduction is None
    assert zero_result.absolute_entropy_reduction == 0.0


@pytest.mark.parametrize(
    ("before_answers", "after_answers", "reduction_sign", "shift_positive"),
    [
        (("A", "B"), ("A", "A"), 1, True),  # useful helper
        (("A", "B"), ("A", "B"), 0, False),  # redundant helper
        (("A", "A"), ("A", "B"), -1, True),  # confusing helper
        (("A", "B"), ("C", "D"), 0, True),  # belief replacement
        (("A", "B"), ("WRONG", "WRONG"), 1, True),  # confident misinformation
    ],
)
def test_fake_student_integration_scenarios(
    before_answers, after_answers, reduction_sign, shift_positive
):
    before_batch = batch(
        "before",
        tuple(
            sample(f"before-{index}", answer, 0.5) for index, answer in enumerate(before_answers)
        ),
    )
    after_batch = batch(
        "after",
        tuple(sample(f"after-{index}", answer, 0.5) for index, answer in enumerate(after_answers)),
    )
    classifier = ExactMatchEquivalenceClassifier()
    estimator = ProbabilityWeightedSemanticEntropyEstimator()
    clusters = cluster_semantic_samples("q", (before_batch, after_batch), classifier)
    result = calculate_uncertainty_reduction(
        rollout_id="r",
        edge_id="e",
        before=estimator.estimate(before_batch, clusters),
        after=estimator.estimate(after_batch, clusters),
        clusters=clusters,
        equivalence_provenance=classifier.provenance,
    )
    assert (result.absolute_entropy_reduction > 1e-12) - (
        result.absolute_entropy_reduction < -1e-12
    ) == reduction_sign
    assert (result.semantic_distribution_shift > 1e-12) is shift_positive
    # Evidence quality remains an independent judge concern, including for confident errors.
    judge_feedback = {"misleading_or_invalid": after_answers == ("WRONG", "WRONG")}
    assert judge_feedback["misleading_or_invalid"] is (after_answers == ("WRONG", "WRONG"))


def test_sample_schema_rejects_missing_and_nonfinite_log_probabilities():
    with pytest.raises(ValueError):
        SemanticSample(
            sample_id="bad",
            answer="x",
            continuation_token_ids=(1,),
            token_log_probabilities=(),
            sampling_seed=0,
        )
    with pytest.raises(ValueError, match="finite"):
        sample("bad", "x", math.nan)


def test_sample_keeps_only_per_sample_data_and_derives_sequence_log_probability():
    item = SemanticSample(
        sample_id="sample",
        answer="answer",
        continuation_token_ids=(1, 2),
        token_log_probabilities=(-0.25, -0.75),
        sampling_seed=3,
    )
    assert item.sequence_log_probability == -1.0
    assert set(item.model_dump()) == {
        "sample_id",
        "answer",
        "continuation_token_ids",
        "token_log_probabilities",
        "sampling_seed",
    }


def test_matched_seeds_are_reproducible_and_edge_specific():
    assert matched_seeds(7, "r", "e", 3) == matched_seeds(7, "r", "e", 3)
    assert matched_seeds(7, "r", "e", 3) != matched_seeds(7, "r", "other", 3)


class FakeSampler:
    def __init__(self):
        self.requests = []

    def sample(self, request):
        self.requests.append(request)
        answers = ("yes", "no") if request.condition == "before" else ("yes", "yes")
        return SemanticSampleBatch(
            condition=request.condition,
            samples=tuple(
                sample(f"{request.condition}-{index}", answer, 0.5, seed)
                for index, (answer, seed) in enumerate(zip(answers, request.seeds, strict=True))
            ),
            model_identity="student-checkpoint",
            tokenizer_identity="tokenizer",
            sampling_parameters={"temperature": request.temperature},
            prompt_fingerprint="a" * 64,
            prompt_version=request.prompt_version,
        )


def rollout():
    return AnnotatedRollout(
        rollout_id="rollout",
        mode="training",
        task=TaskPartition(task_id="task", public={"question": "q", "context": "ctx"}),
        policy={"policy_owner": "student"},
        execution=ExecutionRecord(
            root_node_id="root",
            nodes=(
                ExecutionNode(
                    node_id="root", role=NodeRole.ROOT, depth=0, prompt="q", result="SECRET FINAL"
                ),
                ExecutionNode(
                    node_id="child",
                    parent_id="root",
                    role=NodeRole.PLAIN_SUBCALL,
                    depth=1,
                    prompt="helper q",
                    result="useful evidence",
                ),
            ),
            edges=(
                ExecutionEdge(
                    edge_id="edge",
                    parent_id="root",
                    child_id="child",
                    kind="plain",
                    question="helper q",
                ),
            ),
            events=(
                {
                    "event_type": "helper_question_generated",
                    "subcall_id": "edge",
                    "sequence_number": 1,
                },
                {
                    "event_type": "subcall_completed",
                    "subcall_id": "edge",
                    "response": "useful evidence",
                    "sequence_number": 2,
                },
                {
                    "event_type": "final_answer_submitted",
                    "answer": "SECRET FINAL",
                    "sequence_number": 3,
                },
            ),
        ),
        result={"final_answer": "SECRET FINAL"},
    )


def test_provider_uses_matched_prompts_without_final_or_verifier_leakage():
    sampler = FakeSampler()
    provider = SemanticEntropyUncertaintyProvider(
        sampler=sampler,
        classifier=ExactMatchEquivalenceClassifier(),
        estimator=ProbabilityWeightedSemanticEntropyEstimator(),
        configuration=UncertaintySpec(
            enabled=True, equivalence_provider="exact_match", sample_count=2
        ),
        run_seed=11,
    )
    record = DatasetRecord(
        record_id="task",
        public_task={"question": "q", "context": "ctx"},
        verifier_data={"answer": "VERIFIER SECRET"},
    )
    result = provider.assess(record, rollout(), "edge")
    before, after = sampler.requests
    assert before.seeds == after.seeds
    assert "useful evidence" not in before.prompt
    assert "useful evidence" in after.prompt
    assert "SECRET FINAL" not in before.prompt + after.prompt
    assert "VERIFIER SECRET" not in before.prompt + after.prompt
    assert result.absolute_entropy_reduction > 0


def test_legacy_feedback_loads_and_new_feedback_round_trips():
    assert FeedbackRecord.model_validate({}).uncertainty_assessments == ()
    sampler = FakeSampler()
    provider = SemanticEntropyUncertaintyProvider(
        sampler=sampler,
        classifier=ExactMatchEquivalenceClassifier(),
        estimator=ProbabilityWeightedSemanticEntropyEstimator(),
        configuration=UncertaintySpec(
            enabled=True, equivalence_provider="exact_match", sample_count=2
        ),
        run_seed=1,
    )
    result = provider.assess(
        DatasetRecord(record_id="task", public_task={"question": "q", "context": "ctx"}),
        rollout(),
        "edge",
    )
    record = FeedbackRecord(uncertainty_assessments=(result.model_dump(mode="json"),))
    assert FeedbackRecord.model_validate_json(record.model_dump_json()) == record
