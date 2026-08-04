# Gram-anchor regularization

`rlm_train.regularization` is a trainer-neutral auxiliary representation objective. It
does not import or configure SDPO, and `loss_weight=0` (or `enabled=false`) disables its
model work entirely.

The package separates stable math from lifecycle and trainer details:

- `config.py` validates anchor, sampling, and layer-selection policy.
- `divergence.py` computes detached per-position JS using full vocabulary or reference
  top-k plus an explicit tail bucket. Position `t` aligns `H[t]` with `logits[t]`.
- `selectors.py` freezes relative layer depths and builds completion/all/decision masks.
- `sampling.py` deterministically samples at most `sample_size` positions from the
  configured uniform/JS mixture.
- `gram.py` computes FP32 normalized Gram matrices at only the sampled positions and
  aggregates independently weighted layers.
- `anchor.py` provides feedback-free aligned inputs plus fixed-checkpoint and periodic
  EMA-snapshot lifecycle controllers.
- `metrics.py` records full-valid and sampled JS summaries, loss diagnostics, layer
  weights, sampling composition, and anchor identity/age.
- `prime_adapter.py` is the only trainer seam. It provides selected-block hooks, a
  one-sample transport, objective composition, and an adapter that invokes the pure
  modules. A version-pinned Prime integration should translate its batch type here.

The default quadratic work is bounded by `sample_size=512`; full-sequence Gram matrices
are never constructed when more valid positions are available. Anchor outputs and JS
scores are detached, while selected student hidden states preserve autograd.
