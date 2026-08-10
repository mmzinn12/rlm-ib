"""Memory-bounded Transformers rescoring behavior."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from rlm_train.models.identity import PolicyIdentity, TokenizerIdentity
from rlm_train.models.protocol import SampledGeneration
from rlm_train.models.transformers import TransformersPolicy


def test_selected_logit_scoring_uses_one_forward_without_logprobs():
    torch = pytest.importorskip("torch")

    class SelectiveModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = 0
            self.kept_positions = None

        def forward(
            self,
            *,
            input_ids,
            attention_mask,
            use_cache,
            logits_to_keep,
        ):
            del attention_mask
            assert use_cache is False
            self.calls += 1
            self.kept_positions = tuple(int(value) for value in logits_to_keep.tolist())
            logits = self.scale * torch.ones((1, input_ids.shape[1], 7), device=input_ids.device)
            return SimpleNamespace(logits=logits.index_select(1, logits_to_keep))

    model = SelectiveModel()
    model.eval()
    generator = SimpleNamespace(model=model, tokenizer=object())
    policy = TransformersPolicy(
        generator,
        identity=PolicyIdentity(
            component_id="student",
            revision="v1",
            policy_owner="student",
            checkpoint_id="base",
        ),
        tokenizer_identity=TokenizerIdentity(
            component_id="tokenizer", revision="v1", vocabulary_size=7
        ),
        base_seed=0,
    )
    generation = SampledGeneration(
        text="abcd",
        prompt_token_ids=(5, 6, 7),
        token_ids=(0, 1, 2, 3),
        token_offsets=((0, 1), (1, 2), (2, 3), (3, 4)),
        policy=policy.identity,
        tokenizer=policy.tokenizer_identity,
    )

    score = policy.score_sampled_ids(
        generation,
        require_grad=True,
        return_logits=True,
        return_logprobs=False,
        positions=(1, 3),
    )

    assert model.calls == 1
    assert model.kept_positions == (3, 5)
    assert score.logprobs is None
    assert score.logits.shape == (2, 7)
    assert model.training is False
    score.logits.sum().backward()
    assert model.scale.grad is not None


def test_policy_computes_logits_and_logprobs_from_one_forward():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(5, dtype=torch.float32))
            self.calls = 0

        def forward(
            self,
            *,
            input_ids,
            attention_mask,
            use_cache,
            logits_to_keep,
        ):
            del attention_mask, use_cache
            self.calls += 1
            logits = self.weight.expand(1, input_ids.shape[1], 5)
            return SimpleNamespace(logits=logits.index_select(1, logits_to_keep))

    model = Model()
    policy = TransformersPolicy(
        SimpleNamespace(model=model, tokenizer=object()),
        identity=PolicyIdentity(
            component_id="student",
            revision="v1",
            policy_owner="student",
            checkpoint_id="base",
        ),
        tokenizer_identity=TokenizerIdentity(
            component_id="tokenizer", revision="v1", vocabulary_size=5
        ),
        base_seed=0,
    )
    generation = SampledGeneration(
        text="ab",
        prompt_token_ids=(4,),
        token_ids=(1, 2),
        token_offsets=((0, 1), (1, 2)),
        policy=policy.identity,
        tokenizer=policy.tokenizer_identity,
    )

    score = policy.score_sampled_ids(
        generation,
        require_grad=True,
        return_logits=True,
        return_logprobs=True,
    )

    assert model.calls == 1
    assert score.logits.shape == (2, 5)
    assert score.logprobs.shape == (2,)


def test_trainable_transformers_builder_enables_gradient_checkpointing(monkeypatch):
    torch = pytest.importorskip("torch")
    from rlm_train.models.build import build_transformers_policy
    from rlm_train.spec.models import StudentSpec
    from rlm_train.spec.run import RuntimeSpec

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(max_position_embeddings=4096)
            self.gradient_checkpointing_enabled = False

        def gradient_checkpointing_enable(self):
            self.gradient_checkpointing_enabled = True

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0
        vocab_size = 8
        chat_template = None

    model = Model()
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: model
    )
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: Tokenizer()
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    build_transformers_policy(
        StudentSpec(model_id="student", trainable=True),
        runtime=RuntimeSpec(precision="fp32"),
    )

    assert model.gradient_checkpointing_enabled is True
