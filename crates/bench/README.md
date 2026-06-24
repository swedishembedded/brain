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
| Tool-calling | `src/toolcall.rs` | map a user intent to one structured tool call (id + args) |
| MAD family | `src/mad_*.rs` | recall / fuzzy-recall / noisy-recall / selective-copy / memorize |
| parity | `src/parity.rs` | running-parity state tracking over random bits |
| mod_add | `src/mod_add.rs` | modular addition `a+b=c (mod p)` — the grokking task |
| dyck | `src/dyck.rs` | Dyck-k balanced brackets — hierarchical state |
| scaling sweep | `src/scaling.rs` | multi-size scaling-law sweep + power-law fit (separate entry point, **not** a registry `Benchmark`) |
| integration tests | `tests/*.rs` | learnability guards, gated by `MOE_SKIP_GPU_TESTS` |

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

## Tool-calling (`toolcall`)

The pattern for training **tool-calling** models. Each example maps one
natural-language-shaped *user intent* to exactly one structured **tool call** (a
tool id plus its ordered argument values), laid out as a single masked line:

```
VERB_k  F0 v0  F1 v1 ... Fm vm   =   TOOL_k  a0 a1 ... a(p-1)   NL
└──────────── user intent (prompt) ──────────┘   └─ tool call (assistant) ─┘
```

- `VERB_k` selects the tool (`TOOL_k`). Each tool has a **fixed named
  signature**: `arg_j` of `TOOL_k` always comes from the same field-name token.
  The intent lists those `p` labelled `name value` fields **plus `d` distractor
  fields** (disjoint name pool) the call must ignore, all **shuffled** — so the
  model routes by field *name*, not position.
- The assistant span after `=` is `TOOL_k` then the `p` argument values **in the
  tool's canonical order**, then `NL`. Producing it is verb→tool routing plus, per
  arg, an induction-head copy of the matching field's value past the distractors —
  the shape of real function-call argument filling.
- **Tokens:** `NL`, `SEP` (`=`), then disjoint id ranges for verbs, tool ids,
  per-tool argument field names, distractor field names, and argument values.
  Written as a char dataset (PUA chars) so the existing loader is reused.
- **Masking (the key tool-calling pattern):** loss masked up to & including
  `SEP`, per line, line-aligned (`mask_before='='`, `mask_per_line`,
  `align_to_lines`) — **train only on the assistant/tool-call span, never on the
  prompt**.
- **Scoring:** exact-match of the whole predicted call — from the prompt the model
  greedily decodes `p+1` tokens (tool id + args); a call is correct **only if the
  tool id and every argument value match**. Chance is
  `(1/n_tools)*(1/arg_values)^p`.

Default config (`Toolcall::default`): `n_tools=4`, `args_per_tool=2`,
`n_distractors=2`, `arg_values=12`, 8000 sequences, 800 steps, 2-layer /
d_model-64 / 4-head GPT. **Chance ≈ 0.0017; measured exact-match = 1.00 across
seeds** (train_ce ≈ 0.29), clear of the `0.85` threshold, in ~1-2 min on CPU. An
earlier variant that re-randomized field→slot mapping *per example* was
unlearnable (≈ chance), confirming the metric tracks genuine name-routing.

**Generalizes to** multi-call traces (mask each prompt/result region, supervise
each call span), JSON function-call schemas (`{"name":..,"arguments":{..}}` with
parse-then-compare scoring), and typed/free-form multi-token argument values —
all reuse the same "mask the prompt, train+score only the call" recipe. See the
`toolcall` module doc for details.
## Formal-language / algorithmic benchmarks

State-tracking / hierarchical-structure probes — the tasks a transformer is
known to struggle with. Each writes the same char-dataset layout, masks to its
answer region, and scores next-token accuracy on held-out sequences. All numbers
are measured on the CPU (Cranelift JIT) backend.

| benchmark | structure tested | layout | chance | measured | threshold |
|---|---|---|---|---|---|
| `parity` | single-bit running state | `bits = running_parity NL` | 0.500 | ~1.00 | 0.80 |
| `mod_add` *(info)* | group structure / generalization | `a + b = c NL` | 1/p | ~0.79 @ seed 1337 | 0.25 |
| `dyck` | hierarchical stack / nesting | `( [ ] ) … NL` | 1/k | ~0.99 | 0.70 |

- **parity** — a string of random bits followed by the cumulative XOR at every
  position. Carrying one bit of state along the answer region is the textbook
  non-`AC0` regular language a fixed-depth transformer cannot represent exactly;
  `n_bits` is the state-chain-length difficulty knob. Masked up to `=`, scored at
  every parity position (chance 0.5). Default `n_bits=8`, 6000 seqs, 800 steps,
  d_model-64 → **~1.0**.
- **mod_add** *(informational — does not gate the suite)* — the classic
  *grokking* task: `a+b=c (mod p)` over a small prime, trained on a random
  partition of the `p²` fact table and scored on the held-out facts, so the metric
  is true **generalization** not memorization. Held-out generalization here is a
  sharp grokking phase transition: its single-run value swings with seed and step
  budget (the same engine scored ~0.7 on seeds 1337/42 but *below chance* on seed
  1234 at p=23/3000 steps), so a hard pass/fail bar would be flaky — hence it is
  marked **informational** (reported, never fails the suite). Full grokking
  (test-acc→1.0) needs tens of thousands of steps + weight decay, far over the CPU
  budget. The full d_model-128 width is load-bearing — d_model-96 stays stuck
  memorizing at chance test accuracy. Default `p=17`, `train_frac=0.8`, 2000 steps;
  the `tests/mod_add.rs` guard pins **seed 1337** → ~0.79 test accuracy (chance
  ≈0.059). Difficulty knobs: shrink `train_frac`, crank `steps`/weight-decay.
- **dyck** — well-formed Dyck-`k` words (balanced brackets), scored on predicting
  the correct **close bracket** (determined by the stack top) at every closer —
  the canonical context-free / hierarchical-state probe. `k` (bracket types) and
  `max_depth` (stack depth) are the difficulty knobs. Whole word supervised
  next-token (no `=` mask), line-aligned. Default `k=3`, `max_depth=4`, length 24,
  6000 words, 1000 steps, d_model-96 → **~0.99** (chance 1/3).

## Scaling-law sweep (`src/scaling.rs`)

A **separate entry point** from the `Benchmark` registry. A benchmark answers
"does architecture X learn task T?"; the scaling sweep instead asks "how does
loss on task T improve as the model grows?". It

1. synthesizes one fixed task (it reuses MQAR's dataset — recall improves clearly
   with capacity),
2. trains a GPT (via the architecture-agnostic `model::train::fit`) at a grid of
   increasing sizes (n_layers / d_model),
3. records per size: **parameter count**, a **training-FLOPs proxy**
   (`≈ 6 · params · tokens`, the Kaplan/Chinchilla compute estimate), and the
   **final training loss**, then
4. fits a Chinchilla-style power law `L(N) ≈ E + A · N^(−α)` and reports the
   fitted exponent **α**, the fit **R²**, and the per-size table.

```bash
BRAIN_DEVICE=cpu make bench/scaling          # or: brain bench scaling --device cpu
```

```
size               params          flops   final_loss
-----------------------------------------------------
L1xD32                XXXX      X.XXXe10       X.XXXX
L2xD64               XXXXX      X.XXXe11       X.XXXX
L3xD96              XXXXXX      X.XXXe11       X.XXXX

fitted power law  L(N) ≈ E + A·N^(−α)
  α (exponent) = X.XXXX
  ...
  R² (fit)     = X.XXXX
```

The fit fixes the irreducible floor `E` by a coarse grid search and, for each
candidate `E`, fits the remaining two parameters by ordinary least squares in
log–log space (`log(L − E) = log A − α·log N`), keeping the `E` with the best R².
This is robust with only a few points; the same `run` / `fit_power_law` code
generalizes to larger grids and per-task / per-capability loss slices.

The default grid is `[(1,32,2), (2,64,4), (3,96,6)]` × 400 steps — ≈5 min on the
CPU backend. `tests/scaling.rs` (gated by `MOE_SKIP_GPU_TESTS`) is the *capacity
helps* guard: it asserts final loss is **monotonically non-increasing** with
model size (bigger ≤ smaller + a small fp32/single-run tolerance) and that a
finite power law was fitted.

This sweep is the **foundation** the later per-capability predictive-scaling /
eval-harness work builds on: a reproducible "train a grid of sizes, fit `L(N)`"
loop whose output is a single extrapolatable exponent.
