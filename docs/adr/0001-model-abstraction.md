# ADR 0001 — A unified, architecture-agnostic `Model` abstraction (encoder-decoder as a first-class citizen)

- **Status:** Proposed (awaiting sign-off)
- **Date:** 2026-06-24
- **Deciders:** brain maintainers
- **Supersedes / relates to:** the `DecoderLm`/`Scorer`/`TrainConfig` bench seam in `crates/bench/src/model.rs` (this ADR subsumes it); the `CheckModel` trait in `crates/gradcheck/src/lib.rs` (this ADR builds on it)

> This is a **proposal**. No source code is changed by this document. The
> implementation plan in §10 turns into tasks **#15 (encoder-decoder)** and
> **#16 (gpt migration)** and beyond once signed off.

---

## Decision (summary)

Introduce a small, layered model abstraction in a new crate **`crates/model`**:

1. A low-level **`Module`** trait: a forward/backward unit that enumerates its own
   parameters/grads, written once against the `gpu_core` `Step` seam. Existing
   stages (attention, LayerNorm/RMSNorm, MLP/SwiGLU, MoE) become reusable modules.
2. A **`Model`** trait that composes modules, owns its `ParamStore`/`Optim`,
   defines an **objective** (the scalar loss to differentiate), and exposes a
   uniform forward/backward/param/save/load surface — the union of what
   `gpt::Gpt`, `moe::Trainer`, and `pid::Pid` already implement ad hoc.
3. An **`Objective`/head** notion so the same backbone can serve next-token LM,
   seq2seq (encoder-decoder), and autoencoder/regression without re-deriving the
   training loop.
4. One **generic trainer** (`model::train::fit`) written over `Model` —
   AdamW, grad-clip, cosine-LR, masking, grad-accum, periodic eval, atomic
   checkpointing — replacing the three hand-rolled loops.
5. **`DecoderLm` is reimplemented as a thin adapter** over `Model`, so every
   existing benchmark keeps working unchanged. The autoencoder objective gives
   `mad_compress` (today an `Unsupported` placeholder) a real home.

**Recommended dispatch strategy:** **trait objects (`dyn Model`) at the
orchestration boundary, concrete generics inside each model.** Rationale in §9.

**Cross-attention kernel verdict:** the existing attention kernels **cannot** be
reused for cross-attention or bidirectional self-attention — causality and the
single-fused-QKV / shared-`T` assumption are **hard-coded in the WGSL**. We add a
new attention kernel family (§5). This is the only material new-WGSL cost.

---

## 1. Problem statement

brain is advertised as "architecture-agnostic" (`AGENTS.md`), and the
*accelerator* seam genuinely is: every model is written once against
`gpu_core::{Gpu, DeviceBuffer, Step}` and runs on both backends. But there is **no
model seam**. `gpt::Gpt`, `moe::Trainer`, and `pid::Pid` each hand-roll:

- a `Config` struct + a `param_list()` (`GptConfig::param_list`,
  `moe::param_list`, `PidConfig::param_list`) — three near-identical patterns;
- `new`/`load`/`save`, `set_batch`, `forward`, `backward`, `loss`, `zero_grads`,
  `adamw_step`, `read_weight`/`read_grad` — the **same surface**, three times,
  with gratuitous drift (`gpt::Gpt::forward → f32` vs `pid` splitting
  `forward_submit()` + `loss()`; `moe::adamw_step(t,lr,wd,b1,b2,eps)` vs
  `gpt/pid::adamw_step(t,lr,wd,clip,extra_scale)`);
- a forward `Vec<Step>` and a backward `Vec<Step>` built inline in `model.rs`
  (`Gpt::forward_steps` / `build_backward_steps`, the MoE/PID equivalents);
- a **training loop**: `gpt::train::train` (full-featured: cosine LR, grad accum,
  eval, resumable checkpointing), `moe::train::train` (+ a federated
  `train_expert`), and PID's loop lives outside the model crate entirely.

Consequences:

- **Duplication.** Train/sample/eval/gradcheck logic is copied or re-derived per
  model. Only GPT has the full trainer; MoE's is simpler; PID has none in-crate.
- **Uneven correctness gating.** `gradcheck::CheckModel` is implemented **only for
  `gpt::Gpt`**. MoE and PID are *not* gradient-checked because they lack a
  `write_weight` (TESTING.md §2 calls this out as a known gap). A from-scratch
  WGSL engine that drops its FD gate on two of three models is a real risk.
- **A new architecture means re-deriving everything.** Encoder-decoder, SSM, RNN,
  CNN would each restate config/params/fwd/bwd/train/eval/gradcheck.
- **The new bench seam is decoder-only.** `crates/bench/src/model.rs`'s
  `DecoderLm` abstracts exactly one capability — "train a causal next-token
  decoder, then read per-position logits." It is honest about its scope (its
  doc-comment says so), but it cannot express **encoder-decoder** (no encoder
  input, no cross-attention) or **non-LM objectives**. `mad_compress.rs` is an
  `Err(Unsupported)` placeholder *because* of this; `MadCompress` is deliberately
  left out of `registry()`.

We want one clean seam that hosts all of these uniformly, with encoder-decoder
designed in from day one, **without weakening** any invariant: WGSL is the single
source of truth; fp32 / core-compute-only; one build, two backends selected at
runtime; FD gradient-check gates correctness.

---

## 2. Proposed abstraction

Three layers, smallest-blast-radius first. All live in a new crate
**`crates/model`** (lib name `model`) that depends only on `gpu_core`,
`paramstore`, `optim`, `kernels`, `checkpoint`, `data`. Model crates
(`gpt`/`moe`/`pid`/new `seq2seq`) depend on `model`.

### 2.1 Layer A — `Module`: a forward/backward unit

A `Module` records its forward and backward `Step`s against pre-allocated
buffers, and enumerates the parameters it owns. It is the refactor target for the
inline stage-building in today's `model.rs` files; it is **not** required for v1
to compile a model (a model may build `Step`s directly as GPT does today), but it
is the seam that removes the per-model fwd/bwd duplication over time.

```rust
/// Pre-allocated I/O handles a module reads/writes within the model's SSA graph.
/// Buffers are owned by the Model; the Module only borrows them when recording
/// Steps. `aux` carries stage-specific scratch/cache buffers (e.g. attention
/// scores/probs) that double as the backprop activation cache (SSA discipline).
pub struct Wiring<'a> {
    pub gpu: &'a Gpu,
    pub input: &'a DeviceBuffer,   // x_in  [n, d]
    pub output: &'a DeviceBuffer,  // x_out [n, d]  (a fresh buffer — SSA)
    pub d_output: &'a DeviceBuffer, // upstream grad wrt output (backward)
    pub d_input: &'a DeviceBuffer,  // grad wrt input to produce (backward)
    pub params: &'a ParamStore,
    pub aux: &'a [DeviceBuffer],     // module-private activation/scratch buffers
}

pub trait Module {
    /// Parameter tensors this module owns, as (name, numel). Names are prefixed
    /// by the model (e.g. "blocks.3.attn.qkv.weight"); see §7.
    fn params(&self, prefix: &str) -> Vec<(String, usize)>;

    /// How many `aux` scratch buffers of what sizes this module needs, given the
    /// batch/seq shape. The Model allocates them and passes them back via Wiring.
    fn aux_sizes(&self, shape: Shape) -> Vec<u64>;

    /// Append this module's forward dispatches to `steps`.
    fn forward(&self, w: &Wiring, shape: Shape, steps: &mut Vec<Step>);

    /// Append this module's backward dispatches (grads into ParamStore + d_input).
    fn backward(&self, w: &Wiring, shape: Shape, steps: &mut Vec<Step>);
}

/// Batch/sequence shape threaded through wiring. Encoder-decoder uses two
/// sequence lengths (decoder T_dec attending to encoder memory of length T_enc).
#[derive(Clone, Copy)]
pub struct Shape { pub b: u32, pub t: u32, pub t_kv: u32, pub d_model: u32 }
```

> **Note on the SSA invariant.** AGENTS.md requires "each stage writes a fresh
> buffer that doubles as the backprop activation cache." `Wiring.output` and
> `aux` make this explicit: the Model pre-allocates one output buffer per module
> and the module's forward writes it; backward reads it. This is exactly what
> `Gpt::Layer` (`ln1_out`, `qkv`, `scores`, `probs`, …) does today, lifted into a
> reusable shape.

### 2.2 Layer B — `Model`: composes modules + defines an objective

This is the **primary seam**. It is the union of the surface `gpt::Gpt`,
`moe::Trainer`, `pid::Pid` already expose, normalized to one signature set and
extended with `CheckModel`'s requirements (so every model is gradient-checkable
by construction).

```rust
/// What a batch looks like for a given model. Decoder-LM and seq2seq differ in
/// whether there is a separate source sequence; this enum keeps `set_batch`
/// uniform without forcing every model to accept encoder inputs it ignores.
pub enum Batch<'a> {
    /// Causal LM / single-stream: targets[t] predicts position t+1; IGNORE masks.
    Lm { tokens: &'a [u32], targets: &'a [u32] },
    /// Encoder-decoder: source feeds the encoder, target feeds the decoder,
    /// labels are the decoder's next-token targets (IGNORE masks padding).
    Seq2Seq { src: &'a [u32], tgt: &'a [u32], labels: &'a [u32] },
    /// Non-LM: float inputs + float targets (autoencoder reconstruction,
    /// regression). `tokens` is optional (e.g. token-id inputs reconstructed
    /// against themselves).
    Tensor { tokens: Option<&'a [u32]>, inputs: &'a [f32], targets: &'a [f32] },
}

pub trait Model {
    type Config: ModelConfig;

    /// Build from a config + initial weights, sized for batch `b` × seq `t`
    /// (and, for seq2seq, `t_kv` = encoder length via the config).
    fn new(cfg: Self::Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Self
        where Self: Sized;

    /// Upload one batch (shape must match how the model was constructed).
    fn set_batch(&self, batch: Batch);

    /// Run forward; return the scalar objective loss that `backward` differentiates.
    fn forward(&self) -> f32;
    /// Accumulate analytic gradients for the current batch into the ParamStore.
    fn backward(&self);
    fn zero_grads(&self);

    /// One AdamW step (with optional global-norm clip and a grad-accum scale).
    /// This is the *unified* signature (today gpt/pid match it; moe does not).
    fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32);

    /// Block until submitted device work completes (memory-aperture hygiene).
    fn poll_wait(&self);

    // ---- parameter access (also satisfies gradcheck::CheckModel) ----
    fn param_names(&self) -> Vec<String>;
    fn read_weight(&self, name: &str) -> Vec<f32>;
    fn write_weight(&self, name: &str, data: &[f32]); // NEW for moe/pid; required by gradcheck
    fn read_grad(&self, name: &str) -> Vec<f32>;

    // ---- inference / scoring ----
    /// Per-position logits for one sequence, row-major `[len * vocab]` (decoder
    /// models). Returns `None` for models without a token-classification head
    /// (e.g. a pure regression autoencoder).
    fn logits_all(&self, tokens: &[u32]) -> Option<Vec<f32>>;

    // ---- persistence ----
    fn save(&self, path: &str);
    fn config_json(&self) -> serde_json::Value;
}

/// Config behaviour shared by all models: param layout + (de)serialization.
pub trait ModelConfig: Clone {
    fn param_list(&self) -> Vec<(String, usize)>;
    fn to_json(&self) -> serde_json::Value;
    fn from_json(v: &serde_json::Value) -> Self where Self: Sized;
    fn vocab(&self) -> u32;
    fn block_size(&self) -> u32;
}
```

Two design choices worth flagging for sign-off:

- **`adamw_step` is unified to the gpt/pid signature** (`clip: Option<f32>`,
  `extra_scale: f32`). MoE's current `(t,lr,wd,b1,b2,eps)` becomes a per-model
  detail captured in the config or a fixed default; the unified trainer needs
  clip + accum-scale, not per-call betas.
- **`forward` returns the loss directly** (gpt/moe semantics). PID's
  `forward_submit()`/`loss()` split is a performance optimization (submit without
  reading back); we keep it as an *optional* `forward_submit()` + `loss()` pair on
  the concrete type, but the trait method `forward()` is the simple
  submit-then-read. The generic trainer can call the split form when present via a
  provided default that just does `forward_submit(); self.loss()`.

### 2.3 Layer C — `Objective`: the head + loss

The *objective* is what differs between LM, seq2seq, and autoencoder; the backbone
(embeddings + transformer stack) is shared. Rather than a separate trait object
(which would fragment the SSA graph), the objective is a **variant of the model's
final stage**, selected by config and realized as the model's last module + loss
kernel:

```rust
pub enum Head {
    /// Untied (or tied) projection to `vocab` + masked cross-entropy.
    /// Used by GPT, MoE, PID (u_bins), and the seq2seq decoder.
    TokenClassifier { vocab: u32, tied: bool },
    /// Project to `out_dim` floats + MSE. Used by the MAD compression
    /// autoencoder and future regression heads.
    Regression { out_dim: u32 },
}
```

The model selects the loss kernel by `Head`: `CE_VALUE_MASKED`/`CE_GRAD_MASKED`
(already exist) for `TokenClassifier`; a new `MSE_VALUE`/`MSE_GRAD` pair (small,
trivially gradient-checkable — see §6) for `Regression`. This keeps the objective
inside the one SSA graph (so backward stays a single `Vec<Step>` submit) while
making "what is the loss" an explicit, swappable choice.

### 2.4 How this subsumes the `DecoderLm` bench seam

`DecoderLm`/`Scorer`/`TrainConfig` stay as the **bench-facing** vocabulary (no
benchmark changes), but are re-expressed on top of `Model`:

```rust
// crates/bench/src/model.rs (after migration) — a thin adapter, not a parallel stack.
impl DecoderLm for GptDecoder {
    fn train_decoder(&self, dir: &Path, block: u32, cfg: &TrainConfig, out: &Path)
        -> std::io::Result<(f32, f32)>
    {
        // map TrainConfig -> model::train::FitOpts, run the GENERIC trainer
        model::train::fit::<gpt::Gpt>(dir, gpt_config_from(cfg, block), &fit_opts(cfg), Some(out))
    }
    fn load_scorer(&self, weights: &Path, block: u32) -> Box<dyn Scorer> {
        Box::new(ModelScorer { model: gpt::Gpt::load(weights, 1, block) })
    }
}

/// One blanket Scorer for ANY Model with a token head — replaces per-model GptScorer.
struct ModelScorer<M: Model> { model: M }
impl<M: Model> Scorer for ModelScorer<M> {
    fn vocab(&self) -> usize { self.model.config().vocab() as usize }
    fn logits_all(&self, t: &[u32]) -> Vec<f32> { self.model.logits_all(t).expect("token head") }
}
```

So `DecoderLm` becomes "the subset of `Model` that benchmarks need," and swapping
GPT→MoE→PID for a benchmark is choosing a different `Model` to feed the same
adapter. The encoder-decoder and autoencoder get their **own** thin bench seams
(`Seq2SeqLm`, `Autoencoder`) over the same `Model`, unblocking `mad_compress`.

---

## 3. One generic training/eval/sample loop

Today `gpt::train::train` is the most complete loop; it becomes the **one**
trainer, generic over `Model`. Everything it already does (resume-from-checkpoint,
cosine-with-warmup LR via `cosine_lr`, grad accumulation with averaging, periodic
eval + atomic checkpoint, dataset masking/alignment) is architecture-independent —
it only touches the `Model` surface and the `data` loader.

```rust
// crates/model/src/train.rs
pub struct FitOpts {
    pub steps: u32, pub batch_size: u32, pub block_size: u32,
    pub lr: f32, pub min_lr: f32, pub warmup: u32, pub decay_iters: u32,
    pub weight_decay: f32, pub grad_clip: f32, pub grad_accum: u32,
    pub eval_interval: u32, pub eval_batches: u32, pub seed: u64,
    pub mask_before: Option<char>, pub mask_per_line: bool, pub align_to_lines: bool,
}

/// Train any Model on a token dataset; returns (initial_loss, final_loss).
/// This is `gpt::train::train` lifted to `M: Model` — same control flow, same
/// resume/eval/checkpoint semantics, no GPT-specific code.
pub fn fit<M: Model>(dir: &Path, cfg: M::Config, opts: &FitOpts, out: Option<&Path>)
    -> std::io::Result<(f32, f32)>;

/// Cosine LR with linear warmup (moved verbatim from gpt::train::cosine_lr).
pub fn cosine_lr(it: u32, opts: &FitOpts) -> f32;

/// Greedy/temperature/top-k autoregressive sampling for any token-head Model
/// (moved from gpt::sample::generate; depends only on `logits_all`).
pub fn generate<M: Model>(m: &M, prompt: &[u32], max_new: usize,
                          temperature: f32, top_k: usize, rng: &mut Rng) -> Vec<u32>;
```

Eval (`crates/eval`) generalizes the same way: `gpt_val_perplexity` /
`gpt_exact_match` currently `Gpt::load` directly; they become
`val_perplexity::<M>` / `exact_match::<M>` (or take `&dyn Model`) and depend only
on `forward()`/`logits_all()`. The `Batch` enum lets the autoencoder/seq2seq
objectives flow the right inputs without a second loop.

---

## 4. How each existing model maps on

The surface each model already exposes maps **almost 1:1**; the work is renaming
to the trait and adding the two missing pieces (`write_weight`, unified
`adamw_step`).

| Trait member | `gpt::Gpt` today | `moe::Trainer` today | `pid::Pid` today |
|---|---|---|---|
| `new` | ✅ same sig | ✅ same sig | ✅ same sig |
| `set_batch` | ✅ `Batch::Lm` | ✅ `Batch::Lm` | ✅ `Batch::Lm` (masked) |
| `forward` | ✅ `-> f32` | ✅ `-> f32` | wrap `forward_submit`+`loss` |
| `backward`/`zero_grads` | ✅ | ✅ (`zero_grad` per-name → add `zero_grads`) | ✅ |
| `adamw_step` | ✅ unified sig | ⚠️ change `(b1,b2,eps)`→`(clip,scale)` | ✅ unified sig |
| `write_weight` | ✅ exists | ❌ **add** | ❌ **add** |
| `read_weight`/`read_grad`/`param_names` | ✅ | ✅ | ✅ |
| `logits_all` | ✅ `Some(..)` | add (tied head) | ✅ `logits_all`/`logits_last` |
| `save`/`config_json` | ✅ | ✅ | ✅ |

- **GPT (reference).** LayerNorm + learned positions + causal MHA + GELU MLP +
  untied head + masked CE. It already satisfies the trait (it is the template).
  It is the **proof of the seam** (task #16): make `gpt::Gpt: Model`, route
  `gpt::train` through `model::train::fit`, delete the in-crate loop, keep all GPT
  tests + `check_gpt` green.
- **MoE.** RMSNorm + RoPE + top-k router/experts + aux/z losses + tied head. Maps
  on by: adding `write_weight` (one-liner mirroring GPT's), changing `adamw_step`
  to the unified signature, adding `zero_grads`. **Aux/z losses** are part of its
  objective — they are added into the scalar `forward()` loss (already are), so
  the generic trainer is unaffected. The federated `train_expert` /
  `freeze_grads_except_expert` path stays a MoE-specific method (it is not part of
  the core trait; the federated crate calls it directly).
- **PID.** LayerNorm-with-bias + learned positions + biased linears + SwiGLU +
  separate `u_head`. Maps on by: adding `write_weight`, providing a `forward()`
  that wraps `forward_submit()+loss()`, and (new) an in-crate or generic trainer
  via `fit` instead of external orchestration.

**Migration story:** GPT first (proves the seam, zero behavior change). Then MoE
and PID **incrementally**, one trait member at a time, each landing with its
`CheckModel`-via-`Model` gradient check turned on (closing the TESTING.md gap).
No big-bang rewrite; the trait is additive until the per-model loops are deleted.

---

## 5. How encoder-decoder maps on (the new architecture, task #15)

A new crate **`crates/seq2seq`** implementing `Model`, composed from modules:

- **Embeddings.** Source + target token embeddings + positional embeddings.
  Config flag `shared_embeddings: bool` chooses tied src/tgt embeddings (common
  when src/tgt share a vocab) vs separate. Reuses `EMBED`/`POS_ADD`/`EMB_BWD`/
  `POS_BWD` (or RoPE, by config) — **no new kernels**.
- **Encoder stack** (`n_enc` layers): pre-norm → **bidirectional** self-attention
  → FFN. Reuses LayerNorm/RMSNorm, MLP/GELU, matmul, bias kernels. **Self-attn is
  non-causal** ⇒ needs new attention kernels (below).
- **Decoder stack** (`n_dec` layers): pre-norm → **masked (causal)** self-attn
  (reuses existing `ATTN_*` causal kernels) → **cross-attention** to encoder
  memory → FFN.
- **Cross-attention.** Q from the decoder hidden state, K/V from the **encoder
  memory** (the final encoder output, computed once per forward). Non-causal,
  with `T_dec ≠ T_enc`.
- **Head.** `TokenClassifier { vocab_tgt, tied }` over the target vocab + masked
  CE. Reuses `CE_VALUE_MASKED`/`CE_GRAD_MASKED`.

### 5.1 Kernel verdict (analyzed against the WGSL)

The existing attention kernels are **not** parameterizable into the encoder /
cross-attention cases, because three assumptions are **hard-coded in the WGSL**,
not passed as params:

1. **Causality is baked in.** `attn_scores.wgsl` does `if (j > i) { scores = -inf }`;
   `attn_softmax.wgsl`, `attn_apply.wgsl`, and the backward kernels loop `j <= i`
   (and the transposed `i = j..T` in `attn_bwd_dv/dk`). Encoder self-attn and
   cross-attn are **non-causal**; there is no param to disable the mask.
2. **Single fused QKV buffer.** Q/K/V are read from one buffer at offsets
   `q_off/k_off/v_off` with one `qkv_stride`. Cross-attention needs **two
   buffers**: Q from the decoder (stride `3*d`), K/V from encoder memory (a
   separate buffer, stride `d`).
3. **Shared sequence length `T`.** Both Q and K index as `b*T + pos` with a single
   `tcols=T`. Cross-attention has `T_dec` (queries) and `T_enc` (keys/values).

Therefore add a **new attention kernel family** (fp32, single bind group, ≤4
storage buffers, `@workgroup_size(64)` — same invariants as the rest):

| New kernel | Purpose | Key difference from existing |
|---|---|---|
| `attn_scores_bidir` | encoder self-attn scores | drop the `j>i` mask; loop `0..T` |
| `attn_softmax_bidir` | encoder self-attn softmax | normalize over `0..T` |
| `attn_apply_bidir` | encoder self-attn output | loop `0..T` |
| `attn_bwd_*_bidir` (dscores/dv/dq/dk) | encoder self-attn backward | non-causal loops |
| `attn_scores_cross` | cross-attn scores | Q-buf + KV-buf split; `T_dec×T_enc`; non-causal |
| `attn_softmax_cross` | cross-attn softmax | normalize over `T_enc` |
| `attn_apply_cross` | cross-attn output | V from KV-buf; loop `0..T_enc` |
| `attn_bwd_*_cross` (dscores/dv/dq/dk) | cross-attn backward | grads split across Q-buf (decoder) and KV-buf (encoder memory) |

These follow the existing kernels' structure closely (they are the same math with
the mask removed and a second buffer/length threaded through params), so they are
mechanically derivable and individually gradient-checkable. **Bidirectional and
cross are the only genuinely new WGSL.** Everything else (matmul, norms, FFN/GELU,
embeddings, CE, AdamW) is reused unchanged. Adding `.wgsl` files means
regenerating the const list in `crates/kernels/src/lib.rs` (per the WGSL-source-of-
truth convention).

> **Decision point for sign-off:** new kernels vs. a runtime `is_causal` /
> `cross` flag inside the existing kernels. We recommend **new kernels** (the
> kernels crate is "pure string data, no codegen"; branchy WGSL costs on old GPUs;
> separate kernels stay individually gradient-checkable). Alternative considered
> in §9.

### 5.2 Staying fp32 / two-backend / gradient-checkable

All new kernels obey the core-compute rules, so the `wgsl-cpu` Cranelift JIT
compiles them for the CPU backend exactly like the rest — one build, two backends,
no per-backend seq2seq code. Because cross-attention has two distinct activation
buffers, the SSA discipline is preserved by giving the cross-attn module its own
`aux` buffers (scores/probs over `T_dec×T_enc`) that double as its backprop cache,
mirroring `Gpt::Layer`. The new kernels plug into `gradcheck` via the `Model`
impl (§8) with **no** new gradcheck code.

---

## 6. Non-LM objectives (autoencoder / regression)

In scope for v1 as the **`Regression` head** (§2.3), which unblocks
`mad_compress` (today `Err(Unsupported)`):

- **MAD compression autoencoder** needs encoder → **bottleneck** `z` → decoder
  reconstruction with an MSE-style loss — something a causal LM literally cannot
  express (a decoder at position `t` always sees `x_1..x_t`; masking removes a
  *target*, not the model's *input access*). With the seq2seq backbone + a
  `Regression` head + a bottleneck (a narrow projection between encoder and
  decoder), this is a normal `Model` trained by the generic `fit` over
  `Batch::Tensor`.
- **New kernels needed:** only `mse_value` + `mse_grad` (elementwise: `mean (ŷ−y)²`
  and `2(ŷ−y)/n`), trivially small and gradient-checkable. The bottleneck is a
  matmul (reused).
- A thin **`Autoencoder`** bench seam (mirroring `DecoderLm`) lets `mad_compress`
  register in `bench::registry()` once the model lands.

Classification heads (single-label) are the same `TokenClassifier` with a
sequence-pooled input; deferred but with an obvious home.

**Explicit v1 scope line:** seq2seq (encoder-decoder, token head) and the
regression autoencoder are in v1. SSM/RNN/CNN backbones are **out of scope for
v1** but are *expressible*: they are alternative `Module` stacks behind the same
`Model`/`Head`/trainer, so no abstraction change is needed to add them later.

---

## 7. Parameter / grad / optimizer-state ownership; checkpoint; config

- **Ownership is unchanged.** `ParamStore` already owns weight/grad/Adam-m/Adam-v
  per named tensor and is model-agnostic; `Optim` already drives it from kernel
  indices. The `Model` keeps owning its `ParamStore`/`Optim` exactly as the three
  models do today. The abstraction adds **no new ownership** — it standardizes the
  *names* (`param_names`) and the *access* (`read/write_weight`, `read_grad`).
- **Naming convention.** Module `params(prefix)` produces dotted names
  (`enc.blocks.{l}.self_attn.qkv.weight`, `dec.blocks.{l}.cross_attn.q.weight`,
  …), matching the existing `blocks.{l}.…` scheme so checkpoints stay readable and
  the federated `by_role`/shard logic is untouched.
- **Checkpoint format is unchanged.** `checkpoint::{save,load}` already writes
  `{config, tensors:[{name,shape,offset,numel,role}]}` + an fp32 blob. seq2seq and
  autoencoder just write more/different-named tensors and a richer `config`
  (with `n_enc`, `n_dec`, `t_enc`, `shared_embeddings`, `head`). The
  `config_json()` trait method + `ModelConfig::from_json` make round-tripping
  uniform. Existing GPT/MoE/PID checkpoints remain byte-compatible (their
  `config_json` is what they already emit).
- **Optimizer state** (Adam moments) remains transient/in-`ParamStore`, not
  checkpointed (as today — `gpt::train` notes "weights resume; AdamW moments
  restart"). No change.

---

## 8. Gradient-check integration

`gradcheck::CheckModel` already specifies *exactly* the parameter-access surface
we put on `Model` (`param_names`, `read_weight`, `write_weight`, `read_grad`,
`loss`, `zero_grads`, `backward`). So:

```rust
// crates/gradcheck/src/lib.rs — one blanket impl replaces the per-model impls.
impl<M: model::Model> CheckModel for M {
    fn param_names(&self) -> Vec<String> { self.param_names() }
    fn read_weight(&self, n: &str) -> Vec<f32> { Model::read_weight(self, n) }
    fn write_weight(&self, n: &str, d: &[f32]) { Model::write_weight(self, n, d) }
    fn read_grad(&self, n: &str) -> Vec<f32> { Model::read_grad(self, n) }
    fn loss(&self) -> f32 { self.forward() }
    fn zero_grads(&self) { Model::zero_grads(self) }
    fn backward(&self) { Model::backward(self) }
}
```

Consequences:

- **Every `Model` is gradient-checked for free** the moment it implements the
  trait — including MoE and PID, closing the known TESTING.md gap, and the new
  seq2seq/autoencoder. The only per-model addition was `write_weight`, which the
  trait already requires.
- New `check_*` helpers (`check_seq2seq`, `check_autoencoder`) follow `check_gpt`:
  build a tiny config, set a fixed `Batch`, call `directional_check`. The new
  attention/MSE kernels are validated through the model's forward, exactly as
  GPT's GELU/attention/LN are today. The FD discipline (`directional_check`,
  ε=5e-3, atol/rtol) is unchanged.

---

## 9. Alternatives, tradeoffs, recommendation

### Alt 1 — Enum of models (`enum Arch { Gpt(Gpt), Moe(..), Pid(..), Seq2Seq(..) }`)
- **Pros:** no dynamic dispatch; exhaustive `match`; easy serialization.
- **Cons:** every new architecture edits the central enum and every `match`
  (closed set — the opposite of "host all kinds of ML"); the trainer/eval/bench
  must `match` on arch, re-introducing the duplication we are removing; benches
  can't accept a user-supplied architecture without touching core. **Rejected** —
  it fights the stated goal of an open, extensible foundation.

### Alt 2 — Pure trait objects everywhere (`Box<dyn Model>` in the trainer)
- **Pros:** maximal flexibility; one trainer instance; benches hold `dyn Model`.
- **Cons:** `Model::new`/associated `Config` aren't object-safe as written;
  forces erasing the config type and boxing hot-path calls. The hot loop
  (`forward`/`backward`/`adamw_step`) doesn't need dynamism — it's the same model
  for the whole run.

### Alt 3 (recommended) — Generics inside, trait objects at the boundary
- The **trainer is generic** (`fit::<M: Model>`) so the per-step calls are static
  and the associated `Config` type works naturally — zero hot-path dispatch cost,
  matching today's monomorphic loops.
- The **bench/CLI boundary uses trait objects** (`DecoderLm`/`Seq2SeqLm`/
  `Autoencoder` are already `dyn`-shaped; `ModelScorer` boxes the result) so
  user-facing code stays architecture-agnostic and open.
- **Tradeoff:** some monomorphization bloat (one `fit` per model) — negligible for
  a handful of architectures, and it preserves the "written once against the seam"
  property while keeping the GPU-bound inner loop allocation-free.

**Other tradeoffs noted:** new attention kernels (≈10–14 small WGSL files) vs. a
runtime flag — we choose new kernels (§5.1) to keep the kernels crate codegen-free
and individually gradient-checkable, accepting more files. `Module` (Layer A) is
proposed but **optional for v1**: a model may still build `Step`s inline (as GPT
does) and adopt `Module` incrementally; we recommend introducing `Module` only
once two models share a stage, to avoid premature abstraction.

---

## 10. Implementation plan (reviewable, PR-sized steps)

Each step is independently reviewable, keeps `make test` + `make gradcheck` green,
and changes no behavior unless stated.

1. **PR-1 — `crates/model` skeleton.** New crate: `Model`, `ModelConfig`, `Batch`,
   `Head` traits + `FitOpts`; **no impls yet**. Move `cosine_lr` in (re-exported
   from `gpt::train` for compatibility). Pure additive; compiles, no callers.
2. **PR-2 — Generic trainer.** Port `gpt::train::train` body to
   `model::train::fit::<M>` and `generate` to `model::train::generate::<M>`,
   verbatim control flow, generic over `Model`. Still unused by default.
3. **PR-3 — GPT implements `Model` (task #16, the proof).** `impl Model for
   gpt::Gpt` (it already has the surface). Re-point `gpt::train::train` to call
   `fit::<Gpt>`; delete the duplicated loop body. **Gate:** all `gpt` tests +
   `check_gpt` + `make bench/mqar` unchanged.
4. **PR-4 — Blanket `CheckModel` + bench adapter.** Replace the per-model
   `CheckModel` impl with the blanket `impl<M: Model>`; re-express `GptDecoder`'s
   `DecoderLm` over `fit`/`ModelScorer`. **Gate:** gradcheck + all registered
   benches unchanged.
5. **PR-5 — MoE onto `Model`.** Add `write_weight`, `zero_grads`, unify
   `adamw_step`; `impl Model`; turn on `check_moe`. Route `moe::train` through
   `fit` (keep `train_expert` as a MoE method). **Gate:** new — MoE now
   gradient-checked (closes TESTING.md gap).
6. **PR-6 — PID onto `Model`.** Add `write_weight`; trait `forward` wraps
   `forward_submit()+loss()`; `impl Model`; `check_pid`; route training through
   `fit`. **Gate:** PID now gradient-checked; web demo unaffected (inference path
   unchanged).
7. **PR-7 — New attention kernels (bidir).** Add `attn_*_bidir` forward+backward
   WGSL + regen the const list; unit-test each in isolation (forward determinism +
   FD on a 1-layer wrapper). No model uses them yet.
8. **PR-8 — New attention kernels (cross).** Add `attn_*_cross` forward+backward
   WGSL (two-buffer, `T_dec×T_enc`) + regen; same isolated tests.
9. **PR-9 — `crates/seq2seq` (task #15).** Encoder-decoder `Model`
   (bidir encoder, causal+cross decoder, shared/separate embeddings, token head).
   `check_seq2seq` on a tiny config; an overfit-reduces-loss test; a `Seq2SeqLm`
   bench seam.
10. **PR-10 — `Regression` head + MSE kernels + autoencoder.** Add `mse_value`/
    `mse_grad`; `Head::Regression`; a bottleneck autoencoder `Model`; wire
    `mad_compress` to an `Autoencoder` bench seam and **register it in
    `bench::registry()`**. `check_autoencoder` green.
11. **PR-11 — Generalize `eval`.** `val_perplexity::<M>` / `exact_match::<M>` (or
    `&dyn Model`); keep `gpt_*` as thin wrappers. CLI/bench unchanged.
12. **PR-12 — (optional) `Module` extraction.** Lift the shared stages
    (attention, norm, MLP/SwiGLU) into `Module`s where ≥2 models share them, to
    retire the last inline-Step duplication. Pure refactor, gradcheck-gated.

PRs 1–6 deliver the unified seam with **no new architecture and no behavior
change** (and a correctness *win*: MoE/PID gradient-checked). PRs 7–10 deliver
encoder-decoder and the autoencoder. PRs 11–12 are cleanup.

---

## Consequences

- **Positive:** one model seam; one trainer/eval/sampler; MoE+PID gradient-checked;
  encoder-decoder and non-LM objectives have a principled home; `DecoderLm` becomes
  a thin adapter rather than a parallel stack; new architectures are additive.
- **Negative / cost:** ~10–14 new small WGSL kernels (bidir + cross attention,
  MSE) — the only material new core-compute work; some generic-monomorphization
  bloat; a multi-PR migration (mitigated by GPT-first, incremental, always-green).
- **Risk:** cross-attention backward (grads split across decoder-Q and
  encoder-memory K/V buffers) is the subtlest new math — mitigated by isolated
  per-kernel FD tests (PR-8) before any model depends on it, plus `check_seq2seq`.
