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
| architecture registry | `src/arch.rs` | named architectures (`gpt`, `gpt-small`, `gpt-wide`) + size descriptors |
| capability axes | `src/axes.rs` | benchmark → axis map (`recall`/`copying`/`memory`/…) |
| eval harness | `src/eval.rs` | run the whole battery vs one arch, aggregate per axis, write/compare artifacts |

## Evaluating a new architecture (the turn-key harness)

The benchmark battery is **architecture-agnostic**: every benchmark trains and
scores through the [`DecoderLm`](src/model.rs) seam, so the *same* battery runs
against any architecture and the results are directly comparable. The harness
makes this turn-key.

### The 3-step recipe

1. **Implement `DecoderLm`** for your model — one `train_decoder` + one
   `load_scorer` (and a `Scorer`). Nothing in any benchmark changes. The dense
   GPT baseline (`model::GptDecoder`) is the reference impl.
2. **Add one line to `arch_registry()`** in `src/arch.rs`: a name, a one-line
   description, a `Size` descriptor (depth/width/heads — drives the artifact's
   param count), and a `factory: || Box::new(MyArch::new())`.
3. **Run it**:

   ```bash
   BRAIN_DEVICE=cpu make bench/eval ARCH=<name>     # whole battery vs your arch
   BRAIN_DEVICE=cpu make bench/compare              # leaderboard of all results/*.json
   ```

   or directly:

   ```bash
   ./target/release/brain bench eval --arch <name> [--seed S] [--out F] [--smoke]
   ./target/release/brain bench compare results/<a>.json results/<b>.json ...
   ```

Today three architectures are registered so `compare` is demonstrable
immediately: `gpt` (size per benchmark), `gpt-small` (fixed 1-layer / d_model 32),
`gpt-wide` (fixed 2-layer / d_model 96). A fixed-size variant is a `DecoderLm`
(`arch::ScaledGpt`) that overrides the depth/width the benchmark requests.

### Capability axes

Each benchmark probes a *capability*; several share one. An axis score is the
**mean of its benchmarks' headline scores**, so architectures are compared on a
small interpretable profile rather than a dozen raw numbers (`axes.rs`):

| axis | benchmarks |
|---|---|
| `recall` | `mqar`, `mad_recall`, `mad_fuzzy_recall`, `mad_noisy_recall` |
| `copying` | `mad_selective_copy`, `toolcall` |
| `memory` | `mad_memorize` |
| `state_tracking` | `parity`, `dyck` |
| `compression` | `mad_compress` (a bottleneck autoencoder — see note) |
| `arithmetic` | `mod_add` *(informational; never gates)* |

> **Note (non-LM benchmark):** `mad_compress` trains its own bottleneck
> autoencoder (MSE head), not a causal next-token decoder, so it **ignores** the
> supplied `DecoderLm`. Its `compression`-axis score reflects the autoencoder,
> not the candidate architecture — a known limitation a future arch-aware
> compression objective can address.

### The results artifact

`eval` writes a structured, diffable JSON artifact to `results/<arch>-<seed>.json`
(`results/` is git-ignored; the dir is kept via `results/.gitkeep`). Schema:

| field | meaning |
|---|---|
| `arch` | architecture name |
| `size` | size label, e.g. `L2xD96xH4` or `(bench-default)` |
| `param_count` | total trainable params at a representative shape… |
| `param_count_basis` | …the `vocab=…,block_size=…` that count was computed at |
| `commit` | `git rev-parse --short HEAD`, or `"unknown"` |
| `seed` | run seed |
| `smoke` | whether the fast reduced-budget battery was used |
| `timestamp` | ISO-8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`) from `SystemTime` |
| `benchmarks[]` | per benchmark: `name`, `axis`, `score`, `threshold`, `passed`, `informational`, and full `metrics` (`score` + named `fields`) |
| `axis_scores` | axis → mean score (only axes with ≥1 benchmark) |
| `gating` | `{passed, total, pass_rate}` over non-informational benchmarks |

Example (abridged — a real `make bench/eval ARCH=gpt SEED=1234` run, the
calibrated battery, committed as `results/gpt-1234.json`):

```json
{
  "arch": "gpt",
  "size": "(bench-default)",
  "param_count": 110336,
  "param_count_basis": "vocab=64,block_size=32",
  "commit": "c2829fb",
  "seed": 1234,
  "smoke": false,
  "timestamp": "2026-06-25T00:26:00Z",
  "benchmarks": [
    { "name": "mqar", "axis": "recall", "score": 0.7775, "threshold": 0.55,
      "passed": true, "informational": false,
      "metrics": { "score": 0.7775, "fields": { "chance": 0.125, "train_ce": 0.59 } } }
  ],
  "axis_scores": { "recall": 0.6244, "copying": 0.94, "memory": 1.0,
    "state_tracking": 0.9958, "compression": 0.9458, "arithmetic": 0.7931 },
  "gating": { "passed": 10, "total": 10, "pass_rate": 1.0 }
}
```

The full `gpt` battery passes **10/10** gating benchmarks at this seed (mod_add is
informational). See the committed `results/gpt-1234.json` for the complete file.

### The compare leaderboard

`compare` loads ≥2 artifacts and prints a side-by-side table — columns are
architectures (`arch@commit`), rows are seed/params/commit, overall gating
pass-rate, then per-axis and per-benchmark scores — so a new architecture is
diffed against every prior at a glance:

```
metric                       gpt@c2829fb gpt-small@c2829fb
----------------------------------------------------------
params                            110336             17888
gating pass-rate                  0.5000            0.1000
axis:recall                       0.3563            0.1437
axis:memory                       0.9500            0.4250
...
```

This harness is the **foundation** the next layer (predictive-scaling +
tuning-advisor) builds on: it reasons per-axis, over comparable artifacts.

### Predictive per-capability scaling (`brain bench scale`)

`eval` tells you *where an architecture stands today*. `scale` answers the
forward-looking question: **how will each capability improve as we grow the
model?** — so when you design a new architecture you can predict its per-axis
returns to capacity *before* paying for the bigger run (`src/capscale.rs`).

It sweeps one architecture across a small **SIZE grid** (3 points, increasing
params via `ScaledGpt`: `L1xD32xH2 → L2xD64xH4 → L3xD96xH6`), and for **each
capability axis** trains+scores *one representative benchmark* (the cheapest
informative one — `mqar` for recall, `mad_selective_copy` for copying,
`mad_memorize` for memory, `parity` for state_tracking, `mad_compress` for
compression, `mod_add` for arithmetic) at every size. Per axis it then fits a
**saturating trend** — scores rise toward a ceiling, so we fit the *gap to a
ceiling* as a power law `score(N) ≈ ceil − A·N^(−β)` (the same OLS-in-log-space
machinery as the loss-side `scaling::fit_power_law`, reused via
`scaling::ols`) — and records:

- the **local slope** `Δscore per doubling of N` (how much the axis *responds*
  to capacity — the advisor's lever),
- the saturating-fit exponent **β** and fit **R²**,
- the **extrapolated predicted score at 2× and 4×** the largest grid `N`,
- a coarse **verdict** ∈ {`improving`, `saturating`, `flat`}.

```bash
BRAIN_DEVICE=cpu make bench/scale ARCH=gpt              # or:
./target/release/brain bench scale --arch gpt --seed 1234
```

writes `results/scale-<arch>-<seed>.json` and prints:

```
axis           bench                        scores@sizes   slope   beta  pred@2x  pred@4x     verdict
-----------------------------------------------------------------------------------------------------
recall         mqar                    0.575/0.558/0.592   0.004   0.85    0.591    0.592        flat
copying        mad_selective_copy      0.467/0.667/0.667   0.047   0.18    0.723    0.752   improving
memory         mad_memorize            1.000/1.000/1.000   0.000  -0.00    1.000    1.000  saturating
state_tracking parity                  0.592/0.710/0.750   0.037   0.42    0.766    0.778   improving
compression    mad_compress            0.364/0.364/0.364   0.000  -0.00    0.364    0.364        flat
arithmetic     mod_add*                0.034/0.034/0.034   0.000  -0.00    0.034    0.034        flat
```

**Reading the curves.** `scores@sizes` is the per-size measured score (small →
large); `slope` is the gain per doubling of params; `pred@2x/@4x` is the score
the fit extrapolates at 2×/4× the largest grid size. **`improving`** = still
climbing, capacity helps; **`saturating`** = near its ceiling, little to gain;
**`flat`** = barely moves with size, so the axis is **architecture-bound**
(changing the *mechanism* helps, not more params).

> **Budget & caveats.** The grid is a *smoke* budget (3 sizes × 6 axes = 18 short
> runs, ~a few minutes on the CPU backend), so the absolute scores are coarser
> than a full `eval` — the deliverable is the **shape** of each curve and its
> extrapolation, not a leaderboard number. `mad_compress` trains its own
> autoencoder (it ignores the swept arch), so its curve reflects budget, not
> capacity, and `mod_add` is a high-variance grokking diagnostic (`*`); both are
> reported for completeness but read them with that in mind.

#### The experts knob (future MoE) — how it slots in

The sweep dimension is a generic `Knob` enum. Today only `Knob::Size` is wired
(the GPT family's depth/width). A future **Mixture-of-Experts** `DecoderLm` will
want the *same* machinery on a **number-of-experts** axis (train at experts ∈
{2,4,8}, fit `score(experts)`). Because the fit + extrapolation + advisor reason
purely over `(N, score)` points, activating it is just:

1. register the MoE arch in `arch_registry()`, and
2. fill the `// TODO(experts)` branch in `capscale::grid_for` to return one
   `DecoderLm` factory per expert count.

No change to `fit_saturating`, the predictions, or the advisor. The MoE *scoring*
itself is deliberately **not** implemented here (no MoE arch is registered yet).

### Tuning advisor (`brain bench advise`)

The advisor turns an `eval` artifact (and, if present, a `scale` artifact) into a
**ranked, concrete** list of *what to tune to improve in the best capability
direction* (`src/advisor.rs`) — the breakdown the harness exists to produce.
`brain bench eval` also prints the **top-3** of this list as a footer, so the eval
output itself carries the tuning recommendations.

```bash
./target/release/brain bench advise results/gpt-1234.json results/scale-gpt-1234.json
BRAIN_DEVICE=cpu make bench/advise ARCH=gpt            # finds both artifacts
```

**Heuristics** (each documented in `advisor.rs`):

1. **Rank lever = headroom × responsiveness.** `headroom = 1 − score` (gated axes
   only) × the capscale **size-slope** → the highest *expected-gain* lever first.
   Without a scale artifact a neutral responsiveness prior is used and the rec
   notes that size-vs-mechanism is unknown.
2. **Signal → action** (per weak axis):
   - low score + **rising** size-slope ⇒ *increase model size / depth* (capacity-bound);
   - low score + **flat** size-slope ⇒ *change the MECHANISM* (attention /
     positional / memory) — **architecture-bound, not capacity-bound**;
   - low eval score but **low `train_ce`** (train fits, eval lags) ⇒ *more data /
     regularization / steps* (generalization gap, not capacity);
   - score ≈ ceiling ⇒ *deprioritize* (saturated).
3. **Compute-efficiency.** Each rec carries **score-per-million-params** so the
   advice weighs cost.

```
[1] axis=copying        current=0.940  pred-if-size-doubled=0.723  → increase model size / depth (more capacity)
      rationale: score 0.940 with a rising size-slope (0.047/doubling, verdict=improving) → capacity-bound; bigger N is predicted to raise this axis [score/Mparam=8.519; lever=0.003]
[2] axis=recall         current=0.624  pred-if-size-doubled=0.591  → change the MECHANISM (attention / positional / memory) — not size
      rationale: score 0.624 with a FLAT size-slope (0.004/doubling, verdict=flat) → architecture-bound, not capacity-bound: more params won't move it [score/Mparam=5.659; lever=0.001]
```

**Reading it.** Each line is `[priority] axis, current score, predicted-if-size-
doubled, → concrete action`, then a `rationale` naming the signals (slope,
verdict, `train_ce`, score/Mparam, lever). In the worked `gpt` run above the
standout is **recall**: it has the lowest gated score *and* a flat size-slope, so
the advisor says more params won't help — **change the attention/recall mechanism**.
That is exactly the "improve in the best capability direction" breakdown: it
separates *capacity-bound* axes (buy more params) from *architecture-bound* ones
(redesign the mechanism).

## Running

There is **no usable GPU** in CI here; always select the CPU backend.

```bash
BRAIN_DEVICE=cpu make bench          # run every registered benchmark, one table
BRAIN_DEVICE=cpu make bench/mqar     # run a single benchmark
BRAIN_DEVICE=cpu cargo test -p brain-bench   # unit + integration tests

# direct binary
./target/release/brain bench [--device cpu] [<name>] [--seed S]

# predictive scaling + tuning advisor
BRAIN_DEVICE=cpu make bench/scale ARCH=gpt    # per-capability size sweep -> scale-<arch>-<seed>.json
BRAIN_DEVICE=cpu make bench/advise ARCH=gpt   # ranked tuning recommendations
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
    fn prepare(&self, dir: &Path, seed: u64) -> io::Result<()>;
    /// Architecture-agnostic core: train + score with ANY `DecoderLm`.
    fn evaluate_with(&self, lm: &dyn DecoderLm, dir: &Path, seed: u64) -> io::Result<Metrics>;
    /// Defaults to `self.evaluate_with(&GptDecoder, dir, seed)`.
    fn evaluate(&self, dir: &Path, seed: u64) -> io::Result<Metrics> { /* GPT baseline */ }
    fn threshold(&self) -> f32;
    fn report_fields(&self) -> Vec<&str> { Vec::new() }
    fn informational(&self) -> bool { false }
}
```

- `prepare` synthesizes the dataset into `dir` (brain's `train.bin`/`val.bin`/
  `meta.json` token layout, so the loader reads it unchanged).
- **`evaluate_with`** is the architecture-agnostic core: it trains the supplied
  `DecoderLm` on `dir` and returns `Metrics`; the headline `Metrics::score` is
  what `threshold()` gates. `evaluate` is just this with the GPT baseline, so the
  single-arch runner (`run_all`/`run_one`) is unchanged while the eval harness
  drives the whole battery through `evaluate_with` against any registered
  architecture.
- Keep the benchmark **model-agnostic**: its dataset and scoring must not name a
  particular model — only the `DecoderLm` seam is. (The `mad_compress`
  autoencoder is the documented exception: it ignores `lm`.)

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
