# Generic benchmark evaluation

The evaluator depends only on the `Benchmark` protocol. The download-free
`JSONLBenchmark` adapter expects one object per line with `id`, `prompt`, `target`, and
optional `metadata`. Targets remain verifier-owned and are never passed to generators or
stored in public evaluation records.

Sampling seeds are derived from the base seed, benchmark fingerprint, problem ID, and
sample index. Each completed sample is appended immediately to a JSONL resume journal.
Reruns reuse exact matching records and never duplicate completed samples. Reports include
the benchmark fingerprint, full evaluation configuration, response length/truncation,
and observed cumulative `acc@k` and `pass@k`.

Lockbox benchmarks run only at configured checkpoint steps.

## Pinned Hugging Face snapshots

`hub_splits.py` converts pinned Hugging Face dataset revisions into the same generic JSONL
format without coupling the trainer to the Hub client. Install the optional dependency with
`pip install -e './training[hub-datasets]'`, then call `prepare_aime24_splits`,
`prepare_math500_splits`, or `prepare_math_benchmark_splits`. The built-in defaults create
an AIME24 24/6 split and a MATH-500 400/100 split using salted SHA-256 identity ranking.
These are explicitly project-local partitions of benchmark pools rather than official
upstream train/test splits.

Only mapped public metadata is copied. In particular, source `solution` fields are omitted;
answers appear only in the verifier-owned `target` field. Each output directory contains
`train.jsonl`, `test.jsonl`, and a deterministic `manifest.json` recording the source
revision, split membership, algorithm, counts, and artifact fingerprints. Existing files
must match byte-for-byte, which prevents a notebook rerun from silently changing a split.

`PreparedDatasetSplits.notebook_variables()` returns JSON-compatible scalar values such as
`AIME24_TRAIN_PATH`, `AIME24_TEST_PATH`, counts, fingerprints, and the pinned source
revision. The Colab notebook installs these values into its global namespace explicitly.
