"""Exercise the Colab trainer without downloads, network calls, or a GPU."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rlm.core.trajectory import CallItemSpan, InvocationKind, InvocationNode, TrajectoryTree

from rlm_train.benchmarks import JSONLBenchmark
from rlm_train.colab.assembly import build_fixed_sdpo_components
from rlm_train.colab.checkpoint import TrainingCheckpointManager
from rlm_train.colab.config import (
    ColabProfile,
    ColabRunConfig,
    DatasetConfig,
    GenerationConfig,
    ModelConfig,
    OptimizationConfig,
    OutputConfig,
    Precision,
)
from rlm_train.colab.generation import (
    PromptFormatter,
    TransformersCompletionAdapter,
    TransformersResponseGenerator,
    score_continuation_logits,
)
from rlm_train.colab.gram import TransformersGramLossBuilder
from rlm_train.colab.objectives import (
    ObjectiveComposer,
    RolloutSample,
    TrainingBatch,
)
from rlm_train.colab.teacher import (
    FileTeacherTargetCache,
    TransformersQuestionTeacherProvider,
    build_fixed_teacher_controller,
)
from rlm_train.colab.trainer import (
    MaskedQuestionSDPOLossBuilder,
    NumericProximityRewardRubric,
    PreparedQuestionTarget,
    SingleGPUTrainer,
)
from rlm_train.colab.trajectory_sdpo import TrajectoryQuestionTargetProvider
from rlm_train.experiment import resolve_ablation_preset
from rlm_train.judge import (
    DeterministicFakeStructuredJudgeClient,
    DiagnosticQuestionTeacherFeedback,
    MemoryFeedbackCache,
    PrivilegedJudgeContext,
    StructuredOutputTrajectoryJudge,
)
from rlm_train.regularization import (
    GramAnchorConfig,
    GramAnchorSourceConfig,
    GramLayerSelectionConfig,
    JSTokenSamplingConfig,
)
from rlm_train.sdpo import extract_topk_teacher_target
from rlm_train.trajectory import TrajectoryCompiler, TrajectoryRecorder


class ToyTokenizer:
    """Small reversible tokenizer implementing the Transformers surfaces under test."""

    def __init__(self) -> None:
        self.name_or_path = "toy-tokenizer"
        self.chat_template = "toy-chat-template"
        self.special_tokens_map = {"eos_token": "<eos>", "pad_token": "<pad>"}
        self.eos_token_id = 2
        self.pad_token_id = 0

    def get_vocab(self) -> dict[str, int]:
        return {f"t{index}": index for index in range(16)}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        values = [3 + (ord(character) % 10) for character in text]
        if add_special_tokens:
            values.insert(0, 1)
        return values

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize and add_generation_prompt
        text = "|".join(f"{item['role']}:{item['content']}" for item in messages)
        return self.encode(text, add_special_tokens=True)

    def decode(self, token_ids: Any, *, skip_special_tokens: bool) -> str:
        values = [int(value) for value in token_ids]
        if skip_special_tokens:
            values = [value for value in values if value not in {0, 1, 2}]
        return " ".join(str(value) for value in values)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [f"t{value}" for value in token_ids]


class ToyCausalLM:
    """Minimal trainable causal module with deterministic sampled generation."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        torch = pytest.importorskip("torch")

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 8)
                self.lm_head = torch.nn.Linear(8, 16, bias=False)
                self.embedding.weight.requires_grad_(False)
                self.config = SimpleNamespace(vocab_size=16, num_hidden_layers=1)

            def forward(
                self,
                *,
                input_ids: Any,
                attention_mask: Any,
                output_hidden_states: bool = False,
            ) -> Any:
                del attention_mask
                hidden = self.embedding(input_ids)
                preliminary = self.lm_head(hidden)
                block = torch.tanh(hidden + preliminary[..., : hidden.shape[-1]])
                logits = self.lm_head(block)
                states = (hidden, block) if output_hidden_states else None
                return SimpleNamespace(logits=logits, hidden_states=states)

            def generate(self, **kwargs: Any) -> Any:
                input_ids = kwargs["input_ids"]
                assert "generator" not in kwargs
                tokens = torch.randint(
                    3,
                    16,
                    (input_ids.shape[0], 3),
                    device=input_ids.device,
                )
                return torch.cat((input_ids, tokens), dim=1)

        return Module()


class IndexRubric:
    """Create deterministic within-group variance for numerical trainer tests."""

    def score(self, problem: Any, response: str, *, sample_index: int) -> float:
        del problem, response
        return float(sample_index)


def make_config(tmp_path: Path, *, max_steps: int = 1) -> ColabRunConfig:
    profile = ColabProfile.SMOKE if max_steps == 1 else ColabProfile.TRAIN
    return ColabRunConfig(
        profile=profile,
        experiment_preset=None,
        experiment=resolve_ablation_preset("grpo"),
        model=ModelConfig(
            model_id="toy",
            model_revision="revision-v1",
            precision=Precision.FP32,
            max_context_length=256,
            lora_target_modules=("lm_head",),
        ),
        generation=GenerationConfig(
            use_chat_template=False,
            max_prompt_tokens=128,
            max_new_tokens=3,
            rollouts_per_prompt=2,
        ),
        optimization=OptimizationConfig(
            max_optimizer_steps=max_steps,
            scheduler="constant",
            learning_rate=0.05,
        ),
        dataset=DatasetConfig(
            path=str(tmp_path / "train.jsonl"),
            rubric="smoke_index" if profile is ColabProfile.SMOKE else "exact_match",
        ),
        output=OutputConfig(
            output_directory=str(tmp_path / "outputs"),
            run_name="test",
            checkpoint_every_steps=1,
            evaluate_every_steps=1,
        ),
    )


def make_dataset(tmp_path: Path) -> JSONLBenchmark:
    source = tmp_path / "train.jsonl"
    source.write_text(
        '{"id":"one","prompt":"a","target":"x"}\n{"id":"two","prompt":"b","target":"y"}\n',
        encoding="utf-8",
    )
    return JSONLBenchmark(
        source,
        name="toy",
        version="v1",
        split="train",
    )


def make_generator(model: Any, config: ColabRunConfig) -> TransformersResponseGenerator:
    return TransformersResponseGenerator(
        model,
        ToyTokenizer(),
        config.generation,
        model_context_length=config.model.max_context_length,
    )


def one_step(trainer: SingleGPUTrainer) -> Any:
    async def execute() -> Any:
        problem = trainer.next_problem()
        group = await trainer.generate_group(problem)
        batch = trainer.build_batch(((problem, group),))
        return await trainer.backward_microbatch(batch)

    return asyncio.run(execute())


def test_exact_generation_is_seeded_and_completion_adapter_records_root_and_subcall(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    config = make_config(tmp_path)
    model = ToyCausalLM()
    generator = make_generator(model, config)

    rng_before = torch.random.get_rng_state().clone()
    first = generator.generate_tokenized("hello", seed=10)
    second = generator.generate_tokenized("hello", seed=10)

    assert first == second
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert first.prompt_length == len(first.prompt_token_ids)
    assert first.continuation_length == len(first.continuation_token_ids) == 3
    assert len(first.attention_mask) == first.prompt_length + first.continuation_length
    recorder = TrajectoryRecorder("trace")
    adapter = TransformersCompletionAdapter(
        generator,
        model_name="toy",
        base_seed=9,
        recorder=recorder,
    )
    adapter.generate_for_span("root")
    adapter.generate_for_span(
        "child",
        parent_node_id="trace/root/i000",
        depth=1,
        call_order=0,
    )
    tree = recorder.snapshot()

    assert [node.kind.value for node in tree.nodes] == ["root", "subcall"]
    assert tree.nodes[1].parent_id == tree.nodes[0].node_id
    assert tree.nodes[0].metadata["continuation_token_ids"]


def test_policy_smoke_step_changes_only_trainable_student_parameters(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(5)
    config = make_config(tmp_path)
    dataset = make_dataset(tmp_path)
    model = ToyCausalLM()
    frozen_before = model.embedding.weight.detach().clone()
    trainable_before = model.lm_head.weight.detach().clone()
    trainer = SingleGPUTrainer(
        model=model,
        generator=make_generator(model, config),
        training_dataset=dataset,
        configuration=config,
        rubric=IndexRubric(),
    )

    metrics = one_step(trainer)

    assert metrics.global_step == 1
    assert metrics.policy_loss == pytest.approx(0.0, abs=1e-6)
    assert metrics.gradient_norm > 0.0
    assert not torch.equal(model.lm_head.weight, trainable_before)
    torch.testing.assert_close(model.embedding.weight, frozen_before)
    assert model.embedding.weight.grad is None


def test_numeric_proximity_reward_is_dense_and_target_private(tmp_path):
    source = tmp_path / "numeric.jsonl"
    source.write_text(
        '{"id":"one","prompt":"Compute the answer.","target":"42"}\n',
        encoding="utf-8",
    )
    dataset = JSONLBenchmark(
        source,
        name="numeric",
        version="v1",
        split="train",
        answer_pattern=r"(?P<answer>-?\d+)",
    )
    problem = dataset.problems()[0]
    rubric = NumericProximityRewardRubric(dataset)

    assert rubric.score(problem, "FINAL: 42", sample_index=0) == 1.0
    assert rubric.score(problem, "I estimate 41.", sample_index=1) == 0.5
    assert 0.0 < rubric.score(problem, "The result may be 50.", sample_index=2) < 0.5
    assert rubric.score(problem, "No numeric answer", sample_index=3) == 0.0
    assert "42" not in dataset.format_prompt(problem)


def test_repeated_problem_occurrences_use_fresh_deterministic_seeds(tmp_path):
    config = make_config(tmp_path, max_steps=2)
    dataset = make_dataset(tmp_path)
    model = ToyCausalLM()
    trainer = SingleGPUTrainer(
        model=model,
        generator=make_generator(model, config),
        training_dataset=dataset,
        configuration=config,
        rubric=IndexRubric(),
    )

    async def collect() -> tuple[tuple[int, ...], tuple[int, ...]]:
        first_problem = trainer.next_problem()
        first = await trainer.generate_group(first_problem)
        trainer.next_problem()
        repeated_problem = trainer.next_problem()
        repeated = await trainer.generate_group(repeated_problem)
        return tuple(sample.seed for sample in first), tuple(sample.seed for sample in repeated)

    first_seeds, repeated_seeds = asyncio.run(collect())

    assert first_seeds != repeated_seeds


def test_checkpoint_resume_matches_uninterrupted_next_update(tmp_path):
    torch = pytest.importorskip("torch")
    config = make_config(tmp_path, max_steps=2)
    dataset = make_dataset(tmp_path)

    torch.manual_seed(11)
    uninterrupted_model = ToyCausalLM()
    uninterrupted = SingleGPUTrainer(
        model=uninterrupted_model,
        generator=make_generator(uninterrupted_model, config),
        training_dataset=dataset,
        configuration=config,
        rubric=IndexRubric(),
    )
    first_metrics = one_step(uninterrupted)
    assert first_metrics.global_step == 1
    manager = TrainingCheckpointManager(tmp_path / "run", config)
    checkpoint = manager.save(
        uninterrupted,
        model_identity={"model": "toy", "revision": "v1"},
        tokenizer_fingerprint="toy-tokenizer-v1",
        dataset_fingerprint=dataset.identity.source_fingerprint,
    )
    uninterrupted_second = one_step(uninterrupted)
    expected_parameters = {
        name: parameter.detach().clone()
        for name, parameter in uninterrupted.trainable_parameters.items()
    }

    torch.manual_seed(11)
    resumed_model = ToyCausalLM()
    resumed = SingleGPUTrainer(
        model=resumed_model,
        generator=make_generator(resumed_model, config),
        training_dataset=dataset,
        configuration=config,
        rubric=IndexRubric(),
    )
    restored = manager.restore(
        resumed,
        checkpoint,
        expected_model_identity={"model": "toy", "revision": "v1"},
        expected_tokenizer_fingerprint="toy-tokenizer-v1",
        expected_dataset_fingerprint=dataset.identity.source_fingerprint,
    )
    resumed_second = one_step(resumed)

    assert restored.manifest.global_step == 1
    assert resumed_second.problem_ids == uninterrupted_second.problem_ids
    assert resumed_second.total_loss == pytest.approx(uninterrupted_second.total_loss)
    for name, parameter in resumed.trainable_parameters.items():
        torch.testing.assert_close(parameter, expected_parameters[name])


def test_masked_sdpo_and_disabled_objectives_preserve_gradient_boundaries(tmp_path):
    torch = pytest.importorskip("torch")
    feedback = DiagnosticQuestionTeacherFeedback(
        projector_version="v1",
        parent_node_id="root",
        child_node_id="child",
        information_significance=0.5,
        uncertainty_reduction=0.5,
        novelty=0.5,
        evidence_quality=0.5,
        diagnostic="Check the local inference.",
    )
    target = extract_topk_teacher_target(
        torch.tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]]),
        top_k=2,
        teacher_version=0,
        tokenizer_fingerprint="toy",
    )
    sample = RolloutSample(
        trajectory_id="trajectory",
        problem_id="problem",
        group_index=0,
        sample_index=0,
        prompt="prompt",
        response="response",
        prompt_token_ids=(1,),
        continuation_token_ids=(1, 2),
        continuation_token_offsets=((0, 1), (1, 2)),
        behavior_logprobs=torch.zeros(2),
        trainable_token_mask=torch.ones(2, dtype=torch.bool),
        reward=1.0,
        advantage=1.0,
        seed=0,
        termination_reason="eos",
        truncated=False,
    )
    batch = TrainingBatch(batch_id="batch", samples=(sample,))

    async def provider(_: RolloutSample) -> PreparedQuestionTarget:
        return PreparedQuestionTarget(
            feedback=feedback,
            mask=(True, False),
            target=target,
            cache_key="a" * 64,
        )

    builder = MaskedQuestionSDPOLossBuilder(provider)
    prepared = asyncio.run(builder.prepare(batch))
    logits = torch.tensor(
        [[1.0, 1.5, 0.0], [0.5, 0.0, 1.5]],
        requires_grad=True,
    )
    loss, active = builder.loss(prepared, {"trajectory": logits})
    loss.backward()

    assert active == 1
    assert logits.grad[0].abs().sum().item() > 0.0
    assert logits.grad[1].abs().sum().item() == 0.0
    calls = {"sdpo": 0, "gram": 0}
    policy = torch.tensor(2.0, requires_grad=True)
    composed = ObjectiveComposer(policy_weight=1.0, sdpo_weight=0.0, gram_weight=0.0).compose(
        policy=lambda: (policy, 1),
        sdpo=lambda: (calls.__setitem__("sdpo", 1), 1),
        gram=lambda: (calls.__setitem__("gram", 1), 1),
    )
    composed.total.backward()
    assert composed.total.item() == 2.0
    assert calls == {"sdpo": 0, "gram": 0}
    assert policy.grad.item() == 1.0


def test_fixed_teacher_uses_exact_ids_and_feedback_sensitive_cache(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(19)
    student = ToyCausalLM()
    tokenizer = ToyTokenizer()
    generation = GenerationConfig(
        use_chat_template=False,
        max_prompt_tokens=128,
        max_new_tokens=3,
        rollouts_per_prompt=2,
    )
    controller = build_fixed_teacher_controller(student, checkpoint_identity="toy-v1")
    cache = FileTeacherTargetCache(tmp_path / "targets")
    provider = TransformersQuestionTeacherProvider(
        controller,
        tokenizer,
        PromptFormatter(tokenizer, generation),
        student_tokenizer_fingerprint=provider_tokenizer_fingerprint(tokenizer),
        residency="cpu_offload",
        top_k=2,
        cache=cache,
    )
    feedback = DiagnosticQuestionTeacherFeedback(
        projector_version="v1",
        parent_node_id="root",
        child_node_id="child",
        information_significance=0.5,
        uncertainty_reduction=0.5,
        novelty=0.5,
        evidence_quality=0.5,
        diagnostic="Check the local inference without revealing the answer.",
    )
    target, provenance = provider.score_target(
        original_question="public question",
        continuation_token_ids=(3, 4),
        feedback=feedback,
    )
    cached, cached_provenance = provider.score_target(
        original_question="public question",
        continuation_token_ids=(3, 4),
        feedback=feedback,
    )
    changed = feedback.model_copy(update={"diagnostic": "Re-check the question scope."})
    _, changed_provenance = provider.score_target(
        original_question="public question",
        continuation_token_ids=(3, 4),
        feedback=changed,
    )

    assert target == cached
    assert len(target.token_ids) == 2
    assert provenance == cached_provenance
    assert changed_provenance.cache_key != provenance.cache_key
    assert cache.manifest()["count"] == 2
    assert controller.identity().fingerprint == controller.initial_fingerprint


def provider_tokenizer_fingerprint(tokenizer: ToyTokenizer) -> str:
    from rlm_train.colab.runtime import tokenizer_fingerprint

    return tokenizer_fingerprint(tokenizer)


def test_traced_sdpo_provider_connects_private_judge_projection_teacher_and_exact_mask(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    tokenizer = ToyTokenizer()
    student = ToyCausalLM()
    generation = GenerationConfig(
        use_chat_template=False,
        max_prompt_tokens=128,
        max_new_tokens=3,
        rollouts_per_prompt=2,
    )
    teacher = TransformersQuestionTeacherProvider(
        build_fixed_teacher_controller(student, checkpoint_identity="toy-v1"),
        tokenizer,
        PromptFormatter(tokenizer, generation),
        student_tokenizer_fingerprint=provider_tokenizer_fingerprint(tokenizer),
        residency="cpu_offload",
        top_k=2,
    )
    response = tokenizer.decode((3, 4), skip_special_tokens=True)
    tree = TrajectoryTree(
        "trajectory",
        nodes=[
            InvocationNode(
                node_id="trajectory/root/i000",
                parent_id=None,
                depth=0,
                kind=InvocationKind.ROOT,
                model="student",
                context="public prompt",
                response=response,
                call_item_spans=[
                    CallItemSpan(
                        call_order=0,
                        batch_index=None,
                        start=0,
                        end=1,
                        child_node_id="trajectory/root/i000/c000",
                    )
                ],
                metadata={"continuation_token_ids": [3, 4]},
            ),
            InvocationNode(
                node_id="trajectory/root/i000/c000",
                parent_id="trajectory/root/i000",
                depth=1,
                kind=InvocationKind.SUBCALL,
                model="student",
                context="question",
                response="answer",
                call_order=0,
            ),
        ],
    )
    sample = RolloutSample(
        trajectory_id="trajectory",
        problem_id="problem",
        group_index=0,
        sample_index=0,
        prompt="public prompt",
        response=response,
        prompt_token_ids=(1,),
        continuation_token_ids=(3, 4),
        continuation_token_offsets=((0, 1), (1, 3)),
        behavior_logprobs=torch.zeros(2),
        trainable_token_mask=torch.ones(2, dtype=torch.bool),
        reward=0.0,
        advantage=0.0,
        seed=0,
        termination_reason="stopped",
        truncated=False,
        provenance={"trajectory": tree},
    )
    secret = "PRIVILEGED_SENTINEL"
    provider = TrajectoryQuestionTargetProvider(
        judge=StructuredOutputTrajectoryJudge(
            DeterministicFakeStructuredJudgeClient(),
            judge_version="judge-v1",
            rubric_version="rubric-v1",
            cache=MemoryFeedbackCache(),
        ),
        compiler=TrajectoryCompiler(feedback_mode="diagnostic"),
        teacher=teacher,
        privileged_contexts={"problem": PrivilegedJudgeContext("answer", "v1", {"answer": secret})},
    )

    prepared = asyncio.run(provider(sample))

    assert prepared.mask == (True, False)
    assert prepared.feedback.mode.value == "diagnostic"
    assert secret not in prepared.feedback.model_dump_json()
    assert len(prepared.target.token_ids) == 2


def test_transformers_gram_builder_uses_feedback_free_aligned_anchor(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(23)
    student = ToyCausalLM()
    configuration = GramAnchorConfig(
        enabled=True,
        loss_weight=0.1,
        pair_weighting="none",
        anchor=GramAnchorSourceConfig(checkpoint_path="initial-policy"),
        sampling=JSTokenSamplingConfig(
            vocabulary_support="full",
            sample_size=2,
            top_k=2,
        ),
        layers=GramLayerSelectionConfig(indices=(0,)),
    )
    builder = TransformersGramLossBuilder.from_student(
        student,
        configuration=configuration,
    )
    with torch.no_grad():
        student.lm_head.weight.add_(0.05)
    sample = RolloutSample(
        trajectory_id="trajectory",
        problem_id="problem",
        group_index=0,
        sample_index=0,
        prompt="prompt",
        response="response",
        prompt_token_ids=(1, 3),
        continuation_token_ids=(4, 5),
        continuation_token_offsets=((0, 1), (1, 2)),
        behavior_logprobs=torch.zeros(2),
        trainable_token_mask=torch.ones(2, dtype=torch.bool),
        reward=0.0,
        advantage=0.0,
        seed=0,
        termination_reason="stopped",
        truncated=False,
    )
    batch = TrainingBatch(batch_id="batch", samples=(sample,))
    logits = score_continuation_logits(
        student,
        prompt_token_ids=sample.prompt_token_ids,
        continuation_token_ids=sample.continuation_token_ids,
        require_grad=True,
    )

    loss, active = builder.loss(batch, {sample.trajectory_id: logits})
    loss.backward()

    assert active == 2
    assert loss.item() > 0.0
    assert student.lm_head.weight.grad is not None
    assert all(parameter.grad is None for parameter in builder.anchor.parameters())
    builder.validate_unchanged()


def test_fixed_sdpo_assembly_is_selected_entirely_from_configuration(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(29)
    tokenizer = ToyTokenizer()
    config = ColabRunConfig(
        profile="train",
        experiment_preset=None,
        experiment=resolve_ablation_preset("edge_local_sdpo"),
        model=ModelConfig(
            model_id="toy",
            model_revision="v1",
            precision="fp32",
            max_context_length=256,
            lora_target_modules=("lm_head",),
        ),
        generation=GenerationConfig(
            use_chat_template=False,
            max_prompt_tokens=128,
            max_new_tokens=3,
            rollouts_per_prompt=2,
        ),
        optimization=OptimizationConfig(
            max_optimizer_steps=1,
            sdpo_weight=1.0,
        ),
        dataset=DatasetConfig(path=str(tmp_path / "train.jsonl"), rubric="exact_match"),
        output=OutputConfig(output_directory=str(tmp_path / "outputs"), run_name="sdpo"),
    )
    student = ToyCausalLM()

    components = build_fixed_sdpo_components(
        config,
        student=student,
        tokenizer=tokenizer,
        tokenizer_fingerprint=provider_tokenizer_fingerprint(tokenizer),
        output_directory=tmp_path / "sdpo",
    )

    assert components.teacher.identity["strategy"] == "fixed"
    assert components.teacher.identity["residency"] == "cpu_offload"
    assert components.loss_builder.provider.compiler.feedback_mode.value == "diagnostic"
    assert components.judge_cache.manifest()["count"] == 0
    assert components.teacher_cache.manifest()["count"] == 0
