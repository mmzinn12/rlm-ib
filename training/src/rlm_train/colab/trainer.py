"""Prime-free single-process GRPO trainer with opt-in masked auxiliary objectives."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rlm_train.benchmarks import JSONLBenchmark, Problem, find_prompt_overlaps
from rlm_train.colab.config import (
    ColabRunConfig,
    OptimizationConfig,
    Precision,
)
from rlm_train.colab.generation import (
    TokenGenerationResult,
    TransformersResponseGenerator,
    continuation_logprobs,
    derive_group_seed,
    score_continuation_logits,
)
from rlm_train.colab.objectives import (
    ObjectiveComposer,
    RolloutSample,
    TrainingBatch,
    group_relative_advantages,
    grpo_policy_loss,
)
from rlm_train.judge import QuestionTeacherFeedback
from rlm_train.sdpo import (
    TopKTeacherTarget,
    gather_student_topk_with_tail,
    reverse_kl_topk_with_tail,
    teacher_target_tensors,
)


class RewardRubric(Protocol):
    """Score one public response against verifier-owned problem data."""

    def score(self, problem: Problem, response: str, *, sample_index: int) -> float: ...


class AuxiliaryLossBuilder(Protocol):
    """Prepare detached targets and compute a differentiable masked loss."""

    async def prepare(self, batch: TrainingBatch) -> TrainingBatch: ...

    def loss(
        self,
        batch: TrainingBatch,
        continuation_logits: Mapping[str, Any],
    ) -> tuple[Any, int]: ...

    def validate_unchanged(self) -> None: ...


class BenchmarkRewardRubric:
    """Use the configured generic benchmark extractor and exact verifier."""

    def __init__(self, benchmark: JSONLBenchmark) -> None:
        self.benchmark = benchmark

    def score(self, problem: Problem, response: str, *, sample_index: int) -> float:
        """Return the adapter's bounded numeric reward."""
        del sample_index
        return self.benchmark.score(
            problem,
            self.benchmark.extract_answer(response),
        ).reward


class SmokeIndexRubric:
    """Provide explicit deterministic variance solely for one-step runtime validation."""

    def __init__(self, rollouts_per_prompt: int) -> None:
        if rollouts_per_prompt < 2:
            raise ValueError("smoke gradient validation requires at least two rollouts")
        self.rollouts_per_prompt = rollouts_per_prompt

    def score(self, problem: Problem, response: str, *, sample_index: int) -> float:
        """Return a bounded sample-index signal unrelated to experiment rewards."""
        del problem, response
        if not 0 <= sample_index < self.rollouts_per_prompt:
            raise ValueError("smoke sample index exceeds the configured group")
        return sample_index / (self.rollouts_per_prompt - 1)


@dataclass(frozen=True)
class PreparedQuestionTarget:
    """Carry one restricted projection, exact mask, target, and safe cache key."""

    feedback: QuestionTeacherFeedback
    mask: tuple[bool, ...]
    target: TopKTeacherTarget
    cache_key: str


class QuestionTargetProvider(Protocol):
    """Resolve judge-projected question feedback and exact token mask for a rollout."""

    async def __call__(self, sample: RolloutSample) -> PreparedQuestionTarget: ...

    def validate_unchanged(self) -> None: ...


class MaskedQuestionSDPOLossBuilder:
    """Preserve question-level reverse-KL normalization over variable continuations."""

    def __init__(self, provider: QuestionTargetProvider) -> None:
        self.provider = provider
        self.prepared: dict[str, PreparedQuestionTarget] = {}

    async def prepare(self, batch: TrainingBatch) -> TrainingBatch:
        """Resolve detached teacher targets before the student gradient forward."""
        prepared_items = await asyncio.gather(*(self.provider(sample) for sample in batch.samples))
        feedback: dict[str, QuestionTeacherFeedback] = {}
        targets: dict[str, TopKTeacherTarget] = {}
        masks: dict[str, Any] = {}
        keys = dict(batch.provenance_keys)
        for sample, item in zip(batch.samples, prepared_items, strict=True):
            if len(item.mask) != len(sample.continuation_token_ids):
                raise ValueError("question SDPO mask must align with sampled continuation IDs")
            if not any(item.mask):
                raise ValueError("question SDPO mask must activate at least one token")
            if len(item.target.token_ids) != len(sample.continuation_token_ids):
                raise ValueError("teacher target must align with the complete sampled continuation")
            feedback[sample.trajectory_id] = item.feedback
            targets[sample.trajectory_id] = item.target
            masks[sample.trajectory_id] = item.mask
            keys[f"teacher:{sample.trajectory_id}"] = item.cache_key
            self.prepared[sample.trajectory_id] = item
        prepared_batch = TrainingBatch(
            batch_id=batch.batch_id,
            samples=batch.samples,
            restricted_feedback=feedback,
            teacher_targets=targets,
            sdpo_masks=masks,
            provenance_keys=keys,
        )
        prepared_batch.validate()
        return prepared_batch

    def loss(
        self,
        batch: TrainingBatch,
        continuation_logits: Mapping[str, Any],
    ) -> tuple[Any, int]:
        """Aggregate token-normalized question losses over one explicit batch."""
        torch = _torch()
        weighted_losses: list[Any] = []
        active_total = 0
        for sample in batch.samples:
            trajectory_id = sample.trajectory_id
            logits = continuation_logits[trajectory_id]
            target = batch.teacher_targets[trajectory_id]
            mask = torch.as_tensor(
                batch.sdpo_masks[trajectory_id],
                dtype=torch.bool,
                device=logits.device,
            )
            student = gather_student_topk_with_tail(logits, target)
            teacher_topk, teacher_tail = teacher_target_tensors(
                target,
                reference=student.logprobs,
            )
            active = int(mask.sum().item())
            loss = reverse_kl_topk_with_tail(
                student.logprobs,
                student.tail_logprobs,
                teacher_topk,
                teacher_tail,
                mask,
            )
            weighted_losses.append(loss * active)
            active_total += active
        if active_total == 0:
            raise ValueError("SDPO batch contains no active question tokens")
        return torch.stack(weighted_losses).sum() / active_total, active_total

    def validate_unchanged(self) -> None:
        """Delegate the required fixed-teacher immutability check."""
        self.provider.validate_unchanged()


@dataclass
class TrainerState:
    """Track exact optimizer and deterministic data-loader position."""

    global_step: int = 0
    micro_step: int = 0
    epoch: int = 0
    data_position: int = 0
    consumed_problem_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return checkpoint-safe primitive state."""
        return {
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "epoch": self.epoch,
            "data_position": self.data_position,
            "consumed_problem_ids": list(self.consumed_problem_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainerState:
        """Restore strict state and reject unknown fields."""
        required = {
            "global_step",
            "micro_step",
            "epoch",
            "data_position",
            "consumed_problem_ids",
        }
        if set(payload) != required:
            raise ValueError("trainer state has unknown or missing fields")
        state = cls(
            global_step=int(payload["global_step"]),
            micro_step=int(payload["micro_step"]),
            epoch=int(payload["epoch"]),
            data_position=int(payload["data_position"]),
            consumed_problem_ids=[str(value) for value in payload["consumed_problem_ids"]],
        )
        if min(state.global_step, state.micro_step, state.epoch, state.data_position) < 0:
            raise ValueError("trainer-state counters must be non-negative")
        return state


@dataclass(frozen=True)
class StepMetrics:
    """Emit numerical, throughput, and GPU observations per optimizer step."""

    global_step: int
    problem_ids: tuple[str, ...]
    mean_reward: float
    reward_std: float
    mean_advantage: float
    mean_continuation_length: float
    policy_loss: float
    sdpo_loss: float
    gram_loss: float
    total_loss: float
    approximate_kl: float
    learning_rate: float
    gradient_norm: float
    throughput_tokens_per_second: float
    max_gpu_memory_bytes: int
    active_policy_tokens: int

    def to_dict(self) -> dict[str, Any]:
        """Return tracker-ready metric names."""
        return {
            "step": self.global_step,
            "problem_ids": list(self.problem_ids),
            "reward/mean": self.mean_reward,
            "reward/std": self.reward_std,
            "advantage/mean": self.mean_advantage,
            "continuation/mean_tokens": self.mean_continuation_length,
            "loss/policy": self.policy_loss,
            "loss/sdpo": self.sdpo_loss,
            "loss/gram": self.gram_loss,
            "loss/total": self.total_loss,
            "policy/approximate_kl": self.approximate_kl,
            "optimizer/learning_rate": self.learning_rate,
            "optimizer/gradient_norm": self.gradient_norm,
            "throughput/tokens_per_second": self.throughput_tokens_per_second,
            "gpu/max_memory_bytes": self.max_gpu_memory_bytes,
            "tokens/active_policy": self.active_policy_tokens,
        }


class MetricsJournal:
    """Append one secret-free JSON record per completed optimizer step."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, metrics: StepMetrics) -> None:
        """Persist metrics after the optimizer step succeeds."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(f"{json.dumps(metrics.to_dict(), sort_keys=True, allow_nan=False)}\n")


class SingleGPUTrainer:
    """Generate grouped rollouts and update only trainable student parameters."""

    def __init__(
        self,
        *,
        model: Any,
        generator: TransformersResponseGenerator,
        training_dataset: JSONLBenchmark,
        configuration: ColabRunConfig,
        rubric: RewardRubric | None = None,
        sdpo_builder: AuxiliaryLossBuilder | None = None,
        gram_builder: AuxiliaryLossBuilder | None = None,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        metrics_journal: MetricsJournal | None = None,
    ) -> None:
        torch = _torch()
        self.model = model
        self.generator = generator
        self.training_dataset = training_dataset
        self.configuration = configuration
        self.optimization = configuration.optimization
        self.rubric = rubric or BenchmarkRewardRubric(training_dataset)
        self.sdpo_builder = sdpo_builder
        self.gram_builder = gram_builder
        self.metrics_journal = metrics_journal
        self.state = TrainerState()
        self.trainable_parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not self.trainable_parameters:
            raise ValueError("student model has no trainable parameters")
        self.frozen_parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        }
        self.optimizer = optimizer or torch.optim.AdamW(
            self.trainable_parameters.values(),
            lr=self.optimization.learning_rate,
            betas=(self.optimization.beta1, self.optimization.beta2),
            eps=self.optimization.epsilon,
            weight_decay=self.optimization.weight_decay,
        )
        self.scheduler = scheduler or build_scheduler(self.optimizer, self.optimization)
        self.scaler = build_gradient_scaler(configuration.model.precision)
        self.composer = ObjectiveComposer(
            policy_weight=self.optimization.policy_weight,
            sdpo_weight=self.optimization.sdpo_weight,
            gram_weight=self.optimization.gram_weight,
        )
        if self.optimization.sdpo_weight > 0.0 and sdpo_builder is None:
            raise ValueError("enabled SDPO training requires an SDPO loss builder")
        if self.optimization.gram_weight > 0.0 and gram_builder is None:
            raise ValueError("enabled Gram training requires a Gram loss builder")
        self.optimizer.zero_grad(set_to_none=True)

    async def run(
        self,
        *,
        on_optimizer_step: Callable[[StepMetrics, SingleGPUTrainer], Awaitable[None] | None]
        | None = None,
    ) -> list[StepMetrics]:
        """Train until the configured optimizer-step budget from the current state."""
        reports: list[StepMetrics] = []
        while self.state.global_step < self.optimization.max_optimizer_steps:
            groups: list[tuple[Problem, tuple[RolloutSample, ...]]] = []
            for _ in range(self.optimization.batch_size):
                problem = self.next_problem()
                groups.append((problem, await self.generate_group(problem)))
            batch = self.build_batch(groups)
            report = await self.backward_microbatch(batch)
            if report is not None:
                reports.append(report)
                if self.metrics_journal is not None:
                    self.metrics_journal.append(report)
                if on_optimizer_step is not None:
                    callback_result = on_optimizer_step(report, self)
                    if asyncio.iscoroutine(callback_result):
                        await callback_result
        return reports

    def next_problem(self) -> Problem:
        """Return the next deterministic problem and advance resumable data position."""
        problems = tuple(self.training_dataset.problems())
        if not problems:
            raise ValueError("training dataset is empty")
        if self.state.data_position >= len(problems):
            self.state.epoch += self.state.data_position // len(problems)
            self.state.data_position %= len(problems)
        problem = problems[self.state.data_position]
        self.state.data_position += 1
        self.state.consumed_problem_ids.append(problem.problem_id)
        return problem

    async def generate_group(self, problem: Problem) -> tuple[RolloutSample, ...]:
        """Sample, score, and retain behavior log-probabilities for one prompt group."""
        prompt = self.training_dataset.format_prompt(problem)
        results: list[TokenGenerationResult] = []
        rewards: list[float] = []
        behavior: list[Any] = []
        for sample_index in range(self.configuration.generation.rollouts_per_prompt):
            seed = derive_group_seed(self.configuration.seed, problem.problem_id, sample_index)
            result = self.generator.generate_tokenized(
                prompt,
                seed=seed,
                sample_index=sample_index,
            )
            results.append(result)
            reward = float(self.rubric.score(problem, result.response, sample_index=sample_index))
            if not math.isfinite(reward):
                raise ValueError("rubric returned a non-finite reward")
            rewards.append(reward)
            behavior.append(
                continuation_logprobs(
                    self.model,
                    prompt_token_ids=result.prompt_token_ids,
                    continuation_token_ids=result.continuation_token_ids,
                    require_grad=False,
                ).cpu()
            )
        advantages = group_relative_advantages(rewards)
        samples: list[RolloutSample] = []
        for sample_index, (result, old_logprobs, reward, advantage) in enumerate(
            zip(results, behavior, rewards, advantages, strict=True)
        ):
            trajectory_id = hashlib.sha256(
                f"{self.state.epoch}\0{problem.problem_id}\0{sample_index}\0{result.sampling_metadata['seed']}".encode()
            ).hexdigest()
            mask = _torch().ones(len(result.continuation_token_ids), dtype=_torch().bool)
            sample = RolloutSample(
                trajectory_id=trajectory_id,
                problem_id=problem.problem_id,
                group_index=self.state.data_position - 1,
                sample_index=sample_index,
                prompt=prompt,
                response=result.response,
                prompt_token_ids=result.prompt_token_ids,
                continuation_token_ids=result.continuation_token_ids,
                continuation_token_offsets=result.continuation_token_offsets,
                behavior_logprobs=old_logprobs,
                trainable_token_mask=mask,
                reward=reward,
                advantage=advantage,
                seed=int(result.sampling_metadata["seed"]),
                termination_reason=result.termination_reason,
                truncated=result.truncated,
                provenance=_rollout_provenance(result),
            )
            sample.validate()
            samples.append(sample)
        return tuple(samples)

    def build_batch(
        self,
        groups: Sequence[tuple[Problem, tuple[RolloutSample, ...]]],
    ) -> TrainingBatch:
        """Flatten prompt groups into one replay-complete typed batch."""
        samples = tuple(sample for _, group in groups for sample in group)
        identities = "\0".join(sample.trajectory_id for sample in samples)
        batch = TrainingBatch(
            batch_id=hashlib.sha256(identities.encode()).hexdigest(),
            samples=samples,
            provenance_keys={
                "dataset": self.training_dataset.identity.source_fingerprint,
                "configuration": self.configuration.fingerprint(),
            },
        )
        batch.validate()
        return batch

    async def backward_microbatch(self, batch: TrainingBatch) -> StepMetrics | None:
        """Run exact-token forwards, accumulate gradients, and step when scheduled."""
        torch = _torch()
        started = time.perf_counter()
        batch.validate()
        if self.optimization.sdpo_weight > 0.0:
            assert self.sdpo_builder is not None
            batch = await self.sdpo_builder.prepare(batch)
        if self.optimization.gram_weight > 0.0:
            assert self.gram_builder is not None
            batch = await self.gram_builder.prepare(batch)
        current_logprobs: list[Any] = []
        continuation_logits: dict[str, Any] = {}
        with autocast_context(self.configuration.model.precision):
            for sample in batch.samples:
                logits = score_continuation_logits(
                    self.model,
                    prompt_token_ids=sample.prompt_token_ids,
                    continuation_token_ids=sample.continuation_token_ids,
                    require_grad=True,
                )
                continuation_logits[sample.trajectory_id] = logits
                target_ids = torch.tensor(
                    sample.continuation_token_ids,
                    dtype=torch.long,
                    device=logits.device,
                )
                current_logprobs.append(
                    torch.log_softmax(logits.float(), dim=-1)
                    .gather(-1, target_ids[:, None])
                    .squeeze(-1)
                )
            policy = grpo_policy_loss(
                current_logprobs=current_logprobs,
                behavior_logprobs=[sample.behavior_logprobs for sample in batch.samples],
                advantages=[sample.advantage for sample in batch.samples],
                masks=[sample.trainable_token_mask for sample in batch.samples],
                clip_epsilon=self.optimization.grpo_clip_epsilon,
                kl_coefficient=self.optimization.kl_coefficient,
            )
            result = self.composer.compose(
                policy=lambda: (policy.loss, policy.active_token_count),
                sdpo=(
                    lambda: self.sdpo_builder.loss(batch, continuation_logits)
                    if self.sdpo_builder is not None
                    else None
                ),
                gram=(
                    lambda: self.gram_builder.loss(batch, continuation_logits)
                    if self.gram_builder is not None
                    else None
                ),
            )
            scaled_loss = result.total / self.optimization.gradient_accumulation_steps
        if not torch.isfinite(scaled_loss).item():
            raise FloatingPointError("scaled training loss is non-finite")
        self.scaler.scale(scaled_loss).backward()
        self.state.micro_step += 1
        if self.state.micro_step % self.optimization.gradient_accumulation_steps != 0:
            return None
        self.scaler.unscale_(self.optimizer)
        gradient_norm = finite_gradient_norm(
            self.trainable_parameters,
            self.frozen_parameters,
            max_norm=self.optimization.max_gradient_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.global_step += 1
        if (
            self.state.global_step == 1
            or self.state.global_step % self.configuration.teacher_runtime.fingerprint_interval == 0
        ):
            if self.optimization.sdpo_weight > 0.0:
                assert self.sdpo_builder is not None
                self.sdpo_builder.validate_unchanged()
            if self.optimization.gram_weight > 0.0:
                assert self.gram_builder is not None
                self.gram_builder.validate_unchanged()
        elapsed = max(time.perf_counter() - started, 1e-12)
        token_count = sum(len(sample.continuation_token_ids) for sample in batch.samples)
        rewards = [sample.reward for sample in batch.samples]
        reward_mean = sum(rewards) / len(rewards)
        reward_variance = sum((reward - reward_mean) ** 2 for reward in rewards) / len(rewards)
        metrics = StepMetrics(
            global_step=self.state.global_step,
            problem_ids=tuple(sample.problem_id for sample in batch.samples),
            mean_reward=reward_mean,
            reward_std=math.sqrt(reward_variance),
            mean_advantage=sum(sample.advantage for sample in batch.samples) / len(batch.samples),
            mean_continuation_length=token_count / len(batch.samples),
            policy_loss=float(result.raw["policy"].detach().float().cpu().item()),
            sdpo_loss=float(result.raw["sdpo"].detach().float().cpu().item()),
            gram_loss=float(result.raw["gram"].detach().float().cpu().item()),
            total_loss=float(result.total.detach().float().cpu().item()),
            approximate_kl=float(policy.approximate_kl.float().cpu().item()),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            gradient_norm=gradient_norm,
            throughput_tokens_per_second=token_count / elapsed,
            max_gpu_memory_bytes=(
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
            active_policy_tokens=policy.active_token_count,
        )
        return metrics


def load_training_dataset(
    configuration: ColabRunConfig,
    *,
    base_directory: str | Path | None = None,
    evaluation_benchmarks: Sequence[JSONLBenchmark] = (),
) -> tuple[JSONLBenchmark, dict[str, dict[str, tuple[int, ...]]]]:
    """Load generic JSONL prompts and reject train/evaluation prompt overlap."""
    path = Path(configuration.dataset.path)
    if base_directory is not None and not path.is_absolute():
        path = Path(base_directory) / path
    dataset = JSONLBenchmark(
        path,
        name=configuration.dataset.name,
        version=configuration.dataset.version,
        split=configuration.dataset.split,
        answer_pattern=configuration.dataset.answer_pattern,
        case_sensitive=configuration.dataset.case_sensitive,
    )
    training_inputs = [problem.public_prompt for problem in dataset.problems()]
    overlaps: dict[str, dict[str, tuple[int, ...]]] = {}
    for benchmark in evaluation_benchmarks:
        matched = find_prompt_overlaps(benchmark, training_inputs)
        if matched:
            overlaps[benchmark.identity.key] = matched
    if overlaps:
        raise ValueError(f"training/evaluation prompt overlap detected: {overlaps!r}")
    return dataset, overlaps


def build_scheduler(optimizer: Any, configuration: OptimizationConfig) -> Any:
    """Build a dependency-free constant, linear, or cosine learning-rate schedule."""
    torch = _torch()

    def multiplier(step: int) -> float:
        if configuration.warmup_steps and step < configuration.warmup_steps:
            return (step + 1) / configuration.warmup_steps
        if configuration.scheduler == "constant":
            return 1.0
        decay_steps = max(configuration.max_optimizer_steps - configuration.warmup_steps, 1)
        progress = min(max((step - configuration.warmup_steps) / decay_steps, 0.0), 1.0)
        if configuration.scheduler == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _rollout_provenance(result: TokenGenerationResult) -> dict[str, Any]:
    """Retain public sampling metadata and an optional validated trajectory object."""
    provenance: dict[str, Any] = {
        "sampling": dict(result.sampling_metadata or {}),
        "prompt_truncated": result.prompt_truncated,
    }
    if result.trajectory is not None:
        result.trajectory.validate()
        provenance["trajectory"] = result.trajectory
    return provenance


def build_gradient_scaler(precision: Precision) -> Any:
    """Enable scaling only for CUDA fp16; expose the same stateful interface otherwise."""
    torch = _torch()
    enabled = precision is Precision.FP16 and torch.cuda.is_available()
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(precision: Precision) -> Any:
    """Use configured CUDA mixed precision without affecting CPU unit tests."""
    torch = _torch()
    if not torch.cuda.is_available() or precision is Precision.FP32:
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.float16 if precision is Precision.FP16 else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def finite_gradient_norm(
    trainable_parameters: Mapping[str, Any],
    frozen_parameters: Mapping[str, Any],
    *,
    max_norm: float,
) -> float:
    """Reject missing/non-finite student gradients and any frozen-parameter gradient."""
    torch = _torch()
    if max_norm <= 0.0:
        raise ValueError("maximum gradient norm must be positive")
    if any(parameter.grad is not None for parameter in frozen_parameters.values()):
        offenders = [name for name, value in frozen_parameters.items() if value.grad is not None]
        raise RuntimeError(f"frozen parameters received gradients: {offenders!r}")
    gradients = {
        name: parameter.grad
        for name, parameter in trainable_parameters.items()
        if parameter.grad is not None
    }
    if not gradients:
        raise RuntimeError("optimizer step has no trainable student gradients")
    non_finite = [
        name for name, gradient in gradients.items() if not torch.isfinite(gradient).all()
    ]
    if non_finite:
        raise FloatingPointError(f"non-finite trainable gradients: {non_finite!r}")
    norm = torch.nn.utils.clip_grad_norm_(list(trainable_parameters.values()), max_norm=max_norm)
    if not torch.isfinite(norm).item():
        raise FloatingPointError("gradient norm is non-finite")
    return float(norm.detach().float().cpu().item())


def trainable_parameter_fingerprint(parameters: Mapping[str, Any]) -> str:
    """Hash exact trainable parameter bytes for smoke-update and resume assertions."""
    digest = hashlib.sha256()
    if not parameters:
        raise ValueError("trainable parameter fingerprint requires at least one parameter")
    for name, parameter in sorted(parameters.items()):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.view(_torch().uint8).numpy().tobytes())
    return digest.hexdigest()


def _torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the local trainer") from exc


__all__ = [
    "AuxiliaryLossBuilder",
    "BenchmarkRewardRubric",
    "MaskedQuestionSDPOLossBuilder",
    "MetricsJournal",
    "PreparedQuestionTarget",
    "QuestionTargetProvider",
    "RewardRubric",
    "SingleGPUTrainer",
    "SmokeIndexRubric",
    "StepMetrics",
    "TrainerState",
    "build_gradient_scaler",
    "build_scheduler",
    "finite_gradient_norm",
    "load_training_dataset",
    "trainable_parameter_fingerprint",
]
