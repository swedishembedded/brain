# brain-bench — architecture-evaluation benchmark suite

A reusable, **model-agnostic** layer for asking *"does this architecture
actually learn task X?"* the same way across many benchmarks. It is the
foundation sibling agents extend with more benchmarks (MAD, formal languages,
scaling sweeps, …) by copying the MQAR pattern.

- Package: `brain-bench` · lib: `bench`
- Depends on `brain-data` (datasets/loader), `brain-gpt` (the model trained
  today), `brain-eval`, `brain-checkpoint`.

## What's here

| Piece | File | Role |
|---|---|---|
| `Benchmark` trait | `src/lib.rs` | a benchmark owns its **dataset** + **scoring** |
| `Metrics` | `src/metrics.rs` | CE (nats/bits), bits-per-byte, exact-match, associative-recall, distinct-n, repetition-rate |
| registry + runner | `src/lib.rs` | `registry()`, `run_all()`, `run_one()`, comparison table |
| MQAR (reference) | `src/mqar.rs` | multi-query associative recall |
| integration test | `tests/mqar.rs` | learnability guard, gated by `MOE_SKIP_GPU_TESTS` |

## Running

There is **no usable GPU** in CI here; always select the CPU backend.

```bash
BRAIN_DEVICE=cpu make bench          # run every registered benchmark, one table
BRAIN_DEVICE=cpu make bench/mqar     # run a single benchmark
BRAIN_DEVICE=cpu cargo test -p brain-bench   # unit + integration tests

# direct binary
./target/release/brain bench [--device cpu] [<name>] [--seed S]
```

The runner prints one comparison table:

```
benchmark           score        chance     train_ce  threshold result
-------------------------------------------------------------------------
mqar               0.77XX        0.1250       0.5XXX     0.5500   PASS
```

## The `Benchmark` contract

```rust
pub trait Benchmark {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn prepare(&self, dir: &Path, seed: u64) -> std::io::Result<()>;
    fn evaluate(&self, dir: &Path, seed: u64) -> std::io::Result<Metrics>;
    fn threshold(&self) -> f32;
    fn report_fields(&self) -> Vec<&str> { Vec::new() }
}
```

- `prepare` synthesizes the dataset into `dir` (brain's `train.bin`/`val.bin`/
  `meta.json` token layout, so `gpt::train` loads it unchanged).
- `evaluate` trains a model on `dir` and returns `Metrics`; the headline
  `Metrics::score` is what `threshold()` gates.
- Keep the benchmark **model-agnostic**: its dataset and scoring must not name a
  particular model. Only `evaluate`'s body calls `gpt::train` today, behind the
  `// TODO(model-trait)` seam — when a `Model` trait lands, only that body
  changes.

## Adding a benchmark

1. Add `src/<name>.rs` with a type implementing `Benchmark`.
2. Register it in `registry()` in `src/lib.rs`.
3. Run it: `make bench/<name>` (the generic `bench/%` rule needs no Makefile
   edit) or `brain bench <name>`.
4. Add a learnability test in `tests/`, gated by `MOE_SKIP_GPU_TESTS`, asserting
   the score clears a **measured** threshold.

## MQAR (reference benchmark)

Multi-query associative recall. Each sequence is

```
k1 v1  k2 v2 ... km vm   SEP   q1 a1  q2 a2 ... qn an   NL
```

where each `qi` is an earlier key and `ai` the value it was bound to. Predicting
`ai` from the adjacent `qi` is a single induction-head lookup ("attend to the
earlier occurrence of `qi` as a key, copy its successor value") — unsolvable by
n-gram statistics, so it isolates data-dependent in-context lookup.

- **Tokens:** `NL`, `SEP`, then `vocab_content` content tokens split into a
  lower half (keys) and a disjoint upper half (values) so a query key can never
  collide with any value token (clean, unambiguous recall signal). Written as a
  char dataset (`SEP`→`'='`, `NL`→`'\n'`, content → Private-Use-Area chars) so
  the existing loader + masking path is reused.
- **Masking:** loss masked up to & including `SEP`, per line, and windows are
  **line-aligned** (`align_to_lines`) so each training window is exactly one
  sequence — the lookup is impossible otherwise.
- **Scoring:** associative-recall accuracy over answer positions only — at each
  `ai`, check the model's argmax (over the previous position's logits) equals
  `ai`. Chance is `2 / vocab_content` (the answer is one of the upper-half value
  tokens).

Default config (`Mqar::default`): `vocab_content=16` (8 keys + 8 values, chance
0.125), `n_pairs=2`, `n_queries=2`, 6000 sequences, 600 steps, 2-layer /
d_model-64 / 4-head GPT. **Measured recall on the CPU backend is ~0.77**, clear
of the `0.55` threshold; runtime is a few minutes on CPU. Difficulty scales with
`n_pairs` (keys to disambiguate): a same-budget `n_pairs=3` run drops to ~0.41,
so 2 is the calibrated sweet spot.
