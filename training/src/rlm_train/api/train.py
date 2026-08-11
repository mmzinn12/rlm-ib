"""Public training entry point shared by CLI, local callers, and Colab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlm_train.runtime.factory import ComponentFactory, ResolvedComponents
from rlm_train.spec import RunSpec


def train(
    run: RunSpec | str | Path,
    *,
    factory: ComponentFactory | None = None,
    components: ResolvedComponents | None = None,
    verbose: bool = False,
    resume_from: str | Path | None = None,
) -> Any:
    spec = run if isinstance(run, RunSpec) else RunSpec.from_file(run)
    from rlm_train.artifacts.checkpoints import resolve_checkpoint_path
    from rlm_train.artifacts.run_directory import prepare_training_output

    resume_checkpoint = resolve_checkpoint_path(resume_from) if resume_from is not None else None
    output = prepare_training_output(
        spec.artifacts.output_directory,
        resume_checkpoint=resume_checkpoint,
    )
    if factory is None:
        from rlm_train.saved_runs.run_info import RunInfo
        from rlm_train.training.create_training_run import create_default_training_run

        training_loop = create_default_training_run(
            spec,
            checkpoint_path=resume_checkpoint,
            resume_training=resume_checkpoint is not None,
            verbose=verbose,
        )
        source = {}
        if resume_checkpoint is not None:
            source["resume_checkpoint"] = str(resume_checkpoint)
        RunInfo(
            run_spec_fingerprint=spec.fingerprint,
            resolved_spec=spec.resolved_dict(),
            components={
                "student": training_loop.student.model_info.model_dump(mode="json"),
                "tokenizer": training_loop.student.tokenizer_info.model_dump(mode="json"),
            },
            factory_version="plain-language-training-v1",
            source=source,
        ).write(output / "run-provenance.json")
        return training_loop.train()
    if resume_checkpoint is not None:
        raise ValueError("resume_from is only supported by the default component factory")
    resolver = factory
    resolved = resolver.resolve(spec, overrides=components)
    if resolved.trainer is None:
        raise ValueError("runtime factory did not resolve a trainer")
    if hasattr(resolved.trainer, "verbose"):
        resolved.trainer.verbose = verbose
    provenance = resolver.provenance(spec, resolved)
    source = dict(provenance.source)
    if resume_checkpoint is not None:
        source["resume_checkpoint"] = str(resume_checkpoint)
        provenance = provenance.model_copy(update={"source": source})
    provenance.write(output / "run-provenance.json")
    return resolved.trainer.train()


__all__ = ["train"]
