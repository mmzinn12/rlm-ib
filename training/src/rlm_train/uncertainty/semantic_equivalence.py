"""Deterministic question-conditioned semantic clustering."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from typing import Any

from rlm_train.uncertainty.protocols import SemanticEquivalenceClassifier
from rlm_train.uncertainty.schema import SemanticCluster, SemanticSample, SemanticSampleBatch


def normalize_answer(answer: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", answer.casefold()).split())


class ExactMatchEquivalenceClassifier:
    """Deterministic test and lexical-baseline classifier."""

    @property
    def provenance(self) -> dict[str, str]:
        return {"provider": "exact_match", "model": "normalized-text", "revision": "v1"}

    def equivalent(self, question: str, left: str, right: str) -> bool:
        del question
        return normalize_answer(left) == normalize_answer(right)


class TransformersNLIEquivalenceClassifier:
    """Bidirectional-entailment classifier backed by a pinned Transformers NLI model."""

    def __init__(self, model_id: str, revision: str, *, device: Any | None = None) -> None:
        if (
            not model_id.strip()
            or not revision.strip()
            or revision in {"main", "latest", "default"}
        ):
            raise ValueError("production NLI equivalence requires a pinned model and revision")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers NLI equivalence requires torch and transformers"
            ) from exc
        self.model_id = model_id
        self.revision = revision
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
        if device is not None:
            self.model = self.model.to(device)
        self.model.eval()
        labels = {
            int(key): str(value).casefold() for key, value in self.model.config.id2label.items()
        }
        matches = [index for index, label in labels.items() if "entail" in label]
        if len(matches) != 1:
            raise ValueError("NLI model must expose exactly one entailment label")
        self.entailment_label = matches[0]
        self.cache: dict[str, bool] = {}

    @property
    def provenance(self) -> dict[str, str]:
        return {"provider": "transformers_nli", "model": self.model_id, "revision": self.revision}

    def equivalent(self, question: str, left: str, right: str) -> bool:
        pair = sorted((normalize_answer(left), normalize_answer(right)))
        key = hashlib.sha256(
            f"{self.model_id}\0{self.revision}\0{question}\0{pair[0]}\0{pair[1]}".encode()
        ).hexdigest()
        if key not in self.cache:
            self.cache[key] = self.entails(question, left, right) and self.entails(
                question, right, left
            )
        return self.cache[key]

    def entails(self, question: str, premise_answer: str, hypothesis_answer: str) -> bool:
        premise = f"Question: {question}\nAnswer: {premise_answer}"
        hypothesis = f"Question: {question}\nAnswer: {hypothesis_answer}"
        encoded = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        device = next(self.model.parameters()).device
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("NLI model must return one classification row")
        return int(logits.argmax(dim=-1).item()) == self.entailment_label


def cluster_semantic_samples(
    question: str,
    batches: Iterable[SemanticSampleBatch],
    classifier: SemanticEquivalenceClassifier,
) -> tuple[SemanticCluster, ...]:
    """Build stable complete-link clusters over pooled samples.

    Learned equivalence need not be transitive. To make that ambiguity reproducible, samples are
    sorted by ID and a sample joins the first cluster for which it is equivalent to every member.
    This conservative complete-link rule and comparison order are intentionally part of v1.
    """
    conditioned = tuple((batch.condition, sample) for batch in batches for sample in batch.samples)
    ordered = tuple(sorted(conditioned, key=lambda item: item[1].sample_id))
    if not ordered or len({item[1].sample_id for item in ordered}) != len(ordered):
        raise ValueError("semantic clustering requires non-empty, uniquely identified samples")
    groups: list[list[tuple[str, SemanticSample]]] = []
    for condition, sample in ordered:
        destination = next(
            (
                group
                for group in groups
                if all(
                    classifier.equivalent(question, sample.answer, member.answer)
                    for _, member in group
                )
            ),
            None,
        )
        if destination is None:
            groups.append([(condition, sample)])
        else:
            destination.append((condition, sample))
    clusters = []
    for group in groups:
        member_ids = tuple(item.sample_id for _, item in group)
        digest = hashlib.sha256("\0".join(member_ids).encode()).hexdigest()[:16]
        condition_mass = {}
        for condition in ("before", "after"):
            log_probabilities = [
                item.sequence_log_probability
                for item_condition, item in group
                if item_condition == condition
            ]
            if log_probabilities:
                condition_mass[condition] = logsumexp(log_probabilities)
        clusters.append(
            SemanticCluster(
                cluster_id=f"semantic-{digest}",
                member_sample_ids=member_ids,
                representative_answer=group[0][1].answer,
                condition_log_probability_mass=condition_mass,
            )
        )
    return tuple(sorted(clusters, key=lambda item: item.cluster_id))


def logsumexp(values: Iterable[float]) -> float:
    finite = tuple(float(value) for value in values)
    if not finite or not all(math.isfinite(value) for value in finite):
        raise ValueError("logsumexp requires non-empty finite values")
    maximum = max(finite)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in finite))


__all__ = [
    "ExactMatchEquivalenceClassifier",
    "TransformersNLIEquivalenceClassifier",
    "cluster_semantic_samples",
    "logsumexp",
    "normalize_answer",
]
