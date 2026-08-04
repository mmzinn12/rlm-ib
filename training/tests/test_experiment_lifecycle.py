"""Verify resolved configurations and component identities persist per checkpoint."""

import json

import pytest

from rlm_train.experiment import RunArtifactStore, resolve_ablation_preset
from rlm_train.experiment.dry_run import run_synthetic_dry_run
from rlm_train.trajectory import TrajectoryCompiler


def test_run_and_checkpoint_artifacts_contain_complete_resolved_configuration(tmp_path):
    config = resolve_ablation_preset("edge_local_sdpo")
    store = RunArtifactStore(tmp_path / "run")
    manifest = store.initialize("run-1", config, preset_name="edge_local_sdpo")
    compiler = TrajectoryCompiler(feedback_mode=config.training.feedback.mode)

    checkpoint = store.append_checkpoint(
        manifest,
        checkpoint_step=20,
        model_checkpoint="checkpoint-20",
        teacher_identity={"strategy": "fixed", "fingerprint": "abc", "version": 0},
        feedback_projector=compiler.projector_provenance,
        benchmark_versions=({"name": "synthetic", "version": "v1"},),
    )

    assert checkpoint.resolved_configuration == config.resolved_dict()
    assert checkpoint.configuration_fingerprint == config.fingerprint
    assert checkpoint.feedback_projector == {
        "name": "edge_local_question_feedback",
        "version": "v1",
        "mode": "diagnostic",
    }
    assert store.initialize("run-1", config, preset_name="edge_local_sdpo") == manifest
    with pytest.raises(ValueError, match="already recorded"):
        store.append_checkpoint(
            manifest,
            checkpoint_step=20,
            model_checkpoint="duplicate",
        )


@pytest.mark.asyncio
async def test_download_free_synthetic_dry_run_is_end_to_end_and_resumable(tmp_path):
    output = tmp_path / "dry-run"
    summaries = await run_synthetic_dry_run(
        "training/configs/ood-robust-synthetic.toml",
        output,
    )

    assert summaries[0]["acc_at_k"] == {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0}
    assert (output / "run.json").is_file()
    assert (output / "synthetic-arithmetic-report.json").is_file()
    report = json.loads((output / "synthetic-arithmetic-report.json").read_text())
    assert report["configuration"]["training"]["feedback"]["mode"] == "diagnostic"
    assert len((output / "synthetic-arithmetic-records.jsonl").read_text().splitlines()) == 12

    resumed = await run_synthetic_dry_run(
        "training/configs/ood-robust-synthetic.toml",
        output,
    )
    assert resumed == summaries
    assert len((output / "synthetic-arithmetic-records.jsonl").read_text().splitlines()) == 12
