"""Causal before/after semantic uncertainty orchestration for rollout edges."""

from __future__ import annotations

import time
from typing import Protocol

from rlm_train.datasets.records import DatasetRecord, require_question_context
from rlm_train.spec.uncertainty import UncertaintySpec
from rlm_train.trajectory.schema import AnnotatedRollout, ExecutionEdge
from rlm_train.uncertainty.prompts import render_direct_answer_prompt
from rlm_train.uncertainty.protocols import (
    AnswerSampler,
    SemanticEntropyEstimator,
    SemanticEquivalenceClassifier,
)
from rlm_train.uncertainty.sampling import matched_seeds
from rlm_train.uncertainty.schema import (
    AnswerSamplingRequest,
    SemanticSampleBatch,
    UncertaintyReduction,
)
from rlm_train.uncertainty.semantic_entropy import calculate_uncertainty_reduction
from rlm_train.uncertainty.semantic_equivalence import cluster_semantic_samples


class UncertaintyProvider(Protocol):
    def assess(
        self, record: DatasetRecord, rollout: AnnotatedRollout, edge_id: str
    ) -> UncertaintyReduction: ...


class SemanticEntropyUncertaintyProvider:
    def __init__(
        self,
        *,
        sampler: AnswerSampler,
        classifier: SemanticEquivalenceClassifier,
        estimator: SemanticEntropyEstimator,
        configuration: UncertaintySpec,
        run_seed: int,
    ) -> None:
        if not configuration.enabled:
            raise ValueError("cannot construct an uncertainty provider from disabled settings")
        self.sampler = sampler
        self.classifier = classifier
        self.estimator = estimator
        self.configuration = configuration
        self.run_seed = run_seed
        self.cache: dict[str, UncertaintyReduction] = {}

    def assess(
        self, record: DatasetRecord, rollout: AnnotatedRollout, edge_id: str
    ) -> UncertaintyReduction:
        edge = _resolve_edge(rollout, edge_id)
        cache_key = f"{record.record_id}:{rollout.fingerprint}:{edge_id}:{configuration_fingerprint(self.configuration)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        question, context = require_question_context(record.public_task)
        if record.record_id != rollout.task.task_id:
            raise ValueError("dataset record and rollout task identities do not match")
        response = _completed_helper_response(rollout, edge)
        prefix = f"Helper question: {edge.question}\nHelper response:"
        seeds = matched_seeds(
            self.run_seed, rollout.rollout_id, edge.edge_id, self.configuration.sample_count
        )
        before = self._request(
            condition="before",
            rollout=rollout,
            edge=edge,
            question=question,
            context=context,
            helper_information=prefix,
            seeds=seeds,
        )
        after = self._request(
            condition="after",
            rollout=rollout,
            edge=edge,
            question=question,
            context=context,
            helper_information=f"{prefix} {response}",
            seeds=seeds,
        )
        sampling_started = time.perf_counter()
        before_batch = self.sampler.sample(before)
        after_batch = self.sampler.sample(after)
        sampling_seconds = time.perf_counter() - sampling_started
        _validate_sample_batch(before_batch, before)
        _validate_sample_batch(after_batch, after)
        if before_batch.model_identity != after_batch.model_identity:
            raise ValueError("before and after samples must use one frozen student checkpoint")
        if before_batch.tokenizer_identity != after_batch.tokenizer_identity:
            raise ValueError("before and after samples must use one tokenizer")
        equivalence_started = time.perf_counter()
        clusters = cluster_semantic_samples(question, (before_batch, after_batch), self.classifier)
        equivalence_seconds = time.perf_counter() - equivalence_started
        before_estimate = self.estimator.estimate(before_batch, clusters)
        after_estimate = self.estimator.estimate(after_batch, clusters)
        result = calculate_uncertainty_reduction(
            rollout_id=rollout.rollout_id,
            edge_id=edge.edge_id,
            before=before_estimate,
            after=after_estimate,
            clusters=clusters,
            equivalence_provenance=self.classifier.provenance,
            sampling_seconds=sampling_seconds,
            equivalence_seconds=equivalence_seconds,
        )
        self.cache[cache_key] = result
        return result

    def assess_rollout(
        self, record: DatasetRecord, rollout: AnnotatedRollout
    ) -> tuple[UncertaintyReduction, ...]:
        edges = rollout.execution.edges
        limit = self.configuration.max_edges_per_rollout
        if limit is not None:
            edges = edges[:limit]
        return tuple(self.assess(record, rollout, edge.edge_id) for edge in edges)

    def _request(
        self,
        *,
        condition: str,
        rollout: AnnotatedRollout,
        edge: ExecutionEdge,
        question: str,
        context: object,
        helper_information: str,
        seeds: tuple[int, ...],
    ) -> AnswerSamplingRequest:
        prompt = render_direct_answer_prompt(
            question=question,
            context=context,
            helper_information=helper_information,
            version=self.configuration.prompt_version,
        )
        return AnswerSamplingRequest(
            condition=condition,
            rollout_id=rollout.rollout_id,
            edge_id=edge.edge_id,
            question=question,
            context=context,
            helper_information=helper_information,
            prompt=prompt,
            prompt_version=self.configuration.prompt_version,
            sample_count=self.configuration.sample_count,
            seeds=seeds,
            temperature=self.configuration.temperature,
            top_p=self.configuration.top_p,
            max_new_tokens=self.configuration.max_new_tokens,
        )


def _resolve_edge(rollout: AnnotatedRollout, edge_id: str) -> ExecutionEdge:
    matches = [edge for edge in rollout.execution.edges if edge.edge_id == edge_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate rollout edge {edge_id!r}")
    return matches[0]


def _completed_helper_response(rollout: AnnotatedRollout, edge: ExecutionEdge) -> str:
    child = next(node for node in rollout.execution.nodes if node.node_id == edge.child_id)
    if child.failed or child.result is None:
        raise ValueError("uncertainty can only assess a successfully completed helper edge")
    helper_events = [
        event
        for event in rollout.execution.events
        if event.get("event_type") == "helper_question_generated"
        and event.get("subcall_id") == edge.edge_id
    ]
    if len(helper_events) != 1:
        raise ValueError("focal edge must have exactly one helper-question event")
    question_sequence = int(helper_events[0]["sequence_number"])
    completed = [
        event
        for event in rollout.execution.events
        if event.get("event_type") == "subcall_completed"
        and event.get("subcall_id") == edge.edge_id
        and int(event["sequence_number"]) > question_sequence
    ]
    if completed:
        response = completed[-1].get("response")
        if response != child.result:
            raise ValueError("helper completion event and child result disagree")
    return child.result


def _validate_sample_batch(batch: SemanticSampleBatch, request: AnswerSamplingRequest) -> None:
    if len(batch.samples) != request.sample_count:
        raise ValueError("answer sampler returned the wrong sample count")
    if tuple(sample.sampling_seed for sample in batch.samples) != request.seeds:
        raise ValueError("answer sampler did not preserve matched seed order")
    if batch.condition != request.condition:
        raise ValueError("answer sampler returned a batch for the wrong condition")


def configuration_fingerprint(configuration: UncertaintySpec) -> str:
    import hashlib

    return hashlib.sha256(configuration.model_dump_json().encode()).hexdigest()


__all__ = ["SemanticEntropyUncertaintyProvider", "UncertaintyProvider"]
