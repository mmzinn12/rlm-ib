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

Lockbox benchmarks run only at configured checkpoint steps. AIME24 is intentionally not
implemented in this local stage; adding it later requires only a new adapter registration.
