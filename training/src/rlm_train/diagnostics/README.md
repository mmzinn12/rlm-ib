# Observer-only diagnostics

`collect_observer_diagnostics()` reads a completed response and optional trajectory,
detached teacher divergence, and Gram metrics. It records response length, truncation,
epistemic marker counts/rates, reconsideration signals, subcall depth/breadth/retries,
divergence at epistemic-token positions, and representation observations.
`build_gram_observer_metrics()` adds detached total drift, per-layer Gram loss, and
entropy effective rank from supplied singular values.

The package exposes no reward, loss, prompt, sampler, or optimizer hook. Diagnostics live
only in evaluation configuration, which is excluded from the training fingerprint, so
enabling or disabling them cannot change training inputs or objectives.
