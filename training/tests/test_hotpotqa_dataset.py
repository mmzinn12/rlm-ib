"""HotpotQA Hub rows map directly into production-shaped RLM dataset records."""

from __future__ import annotations

import pytest

from rlm_train.datasets.adapters.hotpotqa import HotpotQADataset, render_hotpotqa_context


def hotpot_rows():
    return (
        {
            "id": "q1",
            "question": "Which publication came first?",
            "answer": "Arthur's Magazine",
            "type": "comparison",
            "level": "medium",
            "supporting_facts": {"title": ["Arthur's Magazine"], "sent_id": [0]},
            "context": {
                "title": ["Distractor", "Arthur's Magazine"],
                "sentences": [
                    ["Unrelated evidence."],
                    ["Arthur's Magazine began publication in 1844."],
                ],
            },
        },
        {
            "id": "q2",
            "question": "Where is the company based?",
            "answer": "Delhi",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": {"title": ["Company"], "sent_id": [0]},
            "context": {"title": ["Company"], "sentences": [["It is based in Delhi."]]},
        },
    )


def test_hotpotqa_dataset_streams_maps_and_limits_rows():
    calls = []

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        return hotpot_rows()

    dataset = HotpotQADataset(
        "hotpotqa/hotpot_qa",
        subset="distractor",
        split="train",
        revision="revision-1",
        max_records=1,
        loader=loader,
    )

    records = dataset.records()

    assert calls == [
        (
            ("hotpotqa/hotpot_qa", "distractor"),
            {"split": "train", "revision": "revision-1", "streaming": True},
        )
    ]
    assert len(records) == 1
    assert records[0].record_id == "q1"
    assert records[0].public_task == {
        "question": "Which publication came first?",
        "context": (
            "### Distractor\nUnrelated evidence.\n\n"
            "### Arthur's Magazine\nArthur's Magazine began publication in 1844."
        ),
    }
    assert records[0].verifier_data == "Arthur's Magazine"
    assert records[0].metadata == {"type": "comparison", "level": "medium"}
    assert "supporting_facts" not in records[0].public_payload()
    assert dataset.records() is records


def test_render_hotpotqa_context_rejects_misaligned_columns():
    with pytest.raises(ValueError, match="must align"):
        render_hotpotqa_context({"title": ["one"], "sentences": []})


def test_hotpotqa_dataset_requires_fixed_columns():
    dataset = HotpotQADataset(
        "hotpotqa/hotpot_qa",
        subset="distractor",
        split="train",
        loader=lambda *_args, **_kwargs: (
            {"id": "q", "question": "question", "context": {}, "answer": "a"},
        ),
    )

    with pytest.raises(ValueError, match="missing columns.*level.*type"):
        dataset.records()
