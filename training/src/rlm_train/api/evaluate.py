"""Public whole-recursive-policy evaluation entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlm_train.runtime.factory import ComponentFactory, ResolvedComponents
from rlm_train.spec import RunSpec


def evaluate(
    run: RunSpec | str | Path,
    *,
    factory: ComponentFactory | None = None,
    components: ResolvedComponents | None = None,
) -> Any:
    spec = run if isinstance(run, RunSpec) else RunSpec.from_file(run)
    resolver = factory or ComponentFactory()
    resolved = resolver.resolve(spec, overrides=components)
    if resolved.evaluator is None:
        raise ValueError("runtime factory did not resolve an evaluator")
    provenance = resolver.provenance(spec, resolved)
    output = Path(spec.artifacts.output_directory)
    provenance.write(output / "evaluation-provenance.json")
    return resolved.evaluator.evaluate()


__all__ = ["evaluate"]
