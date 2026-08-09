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
) -> Any:
    spec = run if isinstance(run, RunSpec) else RunSpec.from_file(run)
    if factory is None:
        from rlm_train.runtime.assembly import assemble_default_factory

        factory = assemble_default_factory(spec)
    resolver = factory
    resolved = resolver.resolve(spec, overrides=components)
    if resolved.trainer is None:
        raise ValueError("runtime factory did not resolve a trainer")
    if hasattr(resolved.trainer, "verbose"):
        resolved.trainer.verbose = verbose
    provenance = resolver.provenance(spec, resolved)
    output = Path(spec.artifacts.output_directory)
    provenance.write(output / "run-provenance.json")
    return resolved.trainer.train()


__all__ = ["train"]
