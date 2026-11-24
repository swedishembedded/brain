// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A pre-flight VRAM estimate for a [`crate::Qwen`] build, derived from the
//! checkpoint's own [`QwenConfig`]/[`Shard`]/[`Dtype`]/shape rather than
//! guessed, so a caller can refuse an over-budget checkpoint BEFORE
//! `Qwen::new_impl`'s `Shard::ANY_GPU` branch dispatches an unsized
//! `Gpu::new` straight to the driver (see `crates/qwen3vl/src/footprint.rs`,
//! which this module mirrors almost line for line - that crate's own
//! composite build needed the identical fix for the identical reason: an
//! oversized checkpoint OOM'd a wgpu device and segfaulted on cleanup instead
//! of being refused).
//!
//! Swedish Embedded AB implements pre-flight VRAM sizing for large model
//! deployments. If your team needs a placement policy that refuses a
//! checkpoint that does not fit instead of crashing the driver, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! # What is counted, and why it matches `Qwen::new_impl` byte for byte on
//! the two axes that dominate
//!
//! - The shard's own parameter bytes ([`crate::shard_param_list`]'s own
//!   element counts, the exact set `new_impl` uploads into the `ParamStore`),
//!   with the 7 per-layer linears (`crate::q8::Q8::LINEARS`) repacked at `dt`
//!   using the SAME `per_word`/scale-group arithmetic
//!   [`crate::Qwen::linear_weight_bytes`] uses for an already-built model -
//!   so this estimate and what a build actually allocates cannot drift apart.
//! - The plain (non-paged) KV cache: `new_impl` allocates one `[t, kv_dim]`
//!   key + value buffer per layer of the WHOLE checkpoint (`0..cfg.n_layers`,
//!   "regardless of `shard`" - its own comment) for every inference build. A
//!   training build allocates it on its first decode instead, so it is not
//!   counted there.
//! - The per-layer-boundary residual stream (`res`): one `[n, d_model]`
//!   buffer per boundary this shard owns (`shard.start..=shard.end`), plus,
//!   when `train`, the two `[n, d_model]` buffers the backward's `dres`
//!   alternates between.
//! - The per-layer activation scratch (`Layer`): allocated ONCE and shared by
//!   every layer, training included (the backward recomputes each layer from
//!   its saved `res[l]`), plus the attention scores: `n_heads·T²` for a
//!   batched forward that materialises them, the decode extent where no
//!   `[H,T,T]` slab exists (a decode build, and a batched build - training or
//!   inference - on the flash-attention path).
//! - The backward scratch (`d_*`, the LoRA correction scratch) under `train`.
//! - The head-only logits/`d_logits` buffer (`shard.head`, skipped entirely
//!   on a `decode_only` build - the LM head runs host-side there), one row
//!   chunk of it at the largest binding budget any device can report.
//!
//! Not counted: the handful of `[1]`-sized dummies every non-owned
//! stage carries - a rounding error next to the terms above, and what
//! `gpu_core::capacity::headroom` (reserved on top of every automatic
//! placement) exists to absorb, the same way it already does for
//! every other model that declares a sized [`gpu_core::devices::Need`].

use gpu_core::select::Dtype;

use crate::config::QwenConfig;
use crate::model::Shard;

/// Bytes this shard's 7-per-owned-layer linears occupy at `dt`, packed
/// exactly as [`crate::Qwen::linear_weight_bytes`] packs an already-built
/// model's [`crate::model::Weight`]s: `elems.div_ceil(per_word) * 4` plus the
/// scales its tier carries ([`model::ops::Weight::bytes_for`]).
fn quantized_linear_bytes(cfg: &QwenConfig, shard: &Shard, dt: Dtype) -> u64 {
    let pack = |n: u64, k: u64| model::ops::Weight::bytes_for(dt, n as usize, k as usize);
    let d = cfg.d_model as u64;
    let hq = cfg.q_dim() as u64;
    let hkv = cfg.kv_dim() as u64;
    let ff = cfg.d_ff as u64;
    let per_layer = pack(hq, d) // attn.wq
        + pack(hkv, d) * 2 // attn.wk, attn.wv
        + pack(d, hq) // attn.wo
        + pack(ff, d) * 2 // mlp.gate, mlp.up
        + pack(d, ff); // mlp.down
    let owned_layers = shard.end.saturating_sub(shard.start) as u64;
    per_layer * owned_layers
}

/// This shard's total weight bytes at `dt`: [`crate::shard_param_list`]'s
/// fp32 element count for every entry NOT one of the 7 per-layer linears
/// (embed/norms/head, always fp32 - `Qwen::new_impl`'s own doc on why),
/// plus - only when `dt` actually quantizes - [`quantized_linear_bytes`] for
/// the linears `shard_param_list`'s own fp32 count would otherwise cover.
/// At `Dtype::F32` the linears stay in `shard_param_list`'s fp32 sum
/// unchanged, matching `new_impl`'s own `plist` filter (`!(quantized &&
/// is_i8_linear(name))` - nothing is filtered out when not quantized).
fn weight_bytes(cfg: &QwenConfig, shard: &Shard, dt: Dtype, train: bool) -> u64 {
    let quantized = dt != Dtype::F32;
    // The embedding and the head of a LoRA build over a narrow base are held in bf16.
    let half_tables = train && cfg.lora.is_some() && crate::model::FROZEN_LORA_TIERS.contains(&dt);
    let base: u64 = crate::shard_param_list(cfg, shard)
        .into_iter()
        .filter(|(name, _)| !(quantized && crate::q8::Q8::is_i8_linear(name)))
        .map(|(name, c)| c as u64 * if half_tables && (name == "tok.weight" || name == "lm_head.weight") { 2 } else { 4 })
        .sum();
    if !quantized {
        return base;
    }
    base + quantized_linear_bytes(cfg, shard, dt)
}

/// The plain (non-paged) KV cache: one `[t, kv_dim]` key + value buffer per
/// layer of the WHOLE checkpoint for an inference build; none for a training
/// build until it first decodes - see this module's doc.
fn kv_cache_bytes(cfg: &QwenConfig, t: u32, train: bool) -> u64 {
    if train {
        return 0;
    }
    2 * cfg.n_layers as u64 * t as u64 * cfg.kv_dim() as u64 * 4
}

/// The per-layer-boundary residual stream: one `[n, d_model]` buffer per
/// boundary this shard owns (`shard.start..=shard.end`, inclusive - see
/// `new_impl`'s own `0..=cfg.n_layers` loop), plus under `train` the two
/// buffers the backward's `dres` alternates between.
fn residual_bytes(cfg: &QwenConfig, shard: &Shard, n: u64, train: bool) -> u64 {
    let live_boundaries = (shard.end.saturating_sub(shard.start) + 1) as u64;
    let row = n * cfg.d_model as u64 * 4;
    live_boundaries * row + if train { 2 * row } else { 0 }
}

/// Whether a batched build of `cfg` runs the flash-attention kernels, which
/// keep no `[H,T,T]` slab, training or inference: every GPU a placement
/// targets runs its workgroup-cooperative kernels, and they tile a head of up
/// to 128 channels (`model::block::FlashGqaIds::supported`). A decode-only
/// build has no batched forward to route. `Qwen::new_impl` takes the same
/// decision from the device itself; the CPU JIT, which no placement targets,
/// stays on the materialised chain.
pub fn runs_flash(cfg: &QwenConfig, decode_only: bool) -> bool {
    !decode_only && cfg.head_dim <= 128
}

/// The per-layer activation scratch (`Layer`): ONE copy shared by every layer
/// (`new_impl`'s buffer pooling - only `res[l+1]` crosses a layer boundary,
/// and the backward recomputes a layer from `res[l]` before differentiating
/// it), plus the attention `scores`/`probs` pair: `b·n_heads·T²` each for a
/// batched forward that materialises them, the decode extent otherwise, and
/// the per-row log-sum-exp pair a flash build keeps.
fn activation_scratch_bytes(cfg: &QwenConfig, n: u64, t: u32, b: u32, decode_only: bool, flash: bool) -> u64 {
    let d = cfg.d_model as u64;
    let hq = cfg.q_dim() as u64;
    let hkv = cfg.kv_dim() as u64;
    let ff = cfg.d_ff as u64;
    let bht2 = crate::model::attn_score_elems(decode_only || flash, b, cfg.n_heads, t);
    let lse = if flash { 2 * b as u64 * cfg.n_heads as u64 * t as u64 } else { 0 };
    // xn1, q_pre, q, k_pre, k, v, ctx, xmid, xn2, gate_pre, up, h, and the projections' outputs proj and mlp_out
    let layer = n * (5 * d + 3 * hq + 3 * hkv + 3 * ff);
    // tokens, targets, the unique-token list, positions, chunk blocks and the per-row loss: one word a row each;
    // the key mask, one word a position, where a batched forward can mask padded keys.
    let rows = 6 * n + if decode_only { 0 } else { t as u64 };
    (layer + 2 * bht2 + lse + rows) * 4
}

/// The backward-only scratch of a training build: the `d_*` activation
/// gradients, the materialised chain's `d_scores` where the flash path is not
/// taken, and the LoRA correction's `[n, max_out]` output buffer.
fn backward_scratch_bytes(cfg: &QwenConfig, n: u64, t: u32, b: u32, train: bool, flash: bool) -> u64 {
    if !train {
        return 0;
    }
    let d = cfg.d_model as u64;
    let hq = cfg.q_dim() as u64;
    let hkv = cfg.kv_dim() as u64;
    let ff = cfg.d_ff as u64;
    let d_scores = if flash { 0 } else { crate::model::attn_score_elems(false, b, cfg.n_heads, t) };
    // d_xn, d_tmp, dxmid | d_ctx, d_q, dq_pre | d_k, dk_pre, d_v | d_h, d_gate_pre, d_up | lora_out | the per-head
    // norm statistics
    let grads = n * (3 * d + 3 * hq + 3 * hkv + 3 * ff) + n * hq.max(ff).max(d).max(hkv) + n * cfg.n_heads as u64;
    (grads + d_scores) * 4
}

/// The head-only logits/`d_logits` buffer: one row chunk of `[n, vocab]`
/// (`crate::model::head_chunk_rows`, at the largest binding budget a device
/// can report), doubled under `train` for `d_logits`, skipped entirely on a
/// `decode_only` build (the LM head runs host-side there - `new_impl`'s own
/// `hd_or_dummy` gate) or on a non-head stage.
fn head_bytes(cfg: &QwenConfig, shard: &Shard, n: u64, train: bool, decode_only: bool) -> u64 {
    if !shard.head || decode_only {
        return 0;
    }
    let v = cfg.vocab as u64;
    let one = model::head_ce::rows_for_budget(n, v, model::block::tile_budget_words_upper_bound()) * v * 4;
    // The final norm's output the head reads, `[n, d_model]`.
    let normed = n * cfg.d_model as u64 * 4;
    normed + if train { one * 2 } else { one }
}

/// The device bytes a [`crate::Qwen::new_impl`] build (every constructor in
/// this crate funnels through it) will occupy, derived from `cfg`/`shard`
/// themselves so it scales with whatever checkpoint and shard shape is
/// actually being built - see this module's doc for what is and is not
/// counted.
pub fn estimate_vram_bytes(cfg: &QwenConfig, shard: &Shard, dt: Dtype, b: u32, t: u32, train: bool, decode_only: bool) -> u64 {
    model::memplan::device_total(&footprint_demands(cfg, shard, dt, b, t, train, decode_only))
}

/// [`estimate_vram_bytes`] as the global planner's categories (`model::memplan`): the weights (with the gradient
/// copy of a training build), the KV cache, and the activations (residual stream, step scratch, backward scratch
/// and the head's rows). The one place the figure is summed, so a placement, a refusal and a plan agree.
pub fn footprint_demands(cfg: &QwenConfig, shard: &Shard, dt: Dtype, b: u32, t: u32, train: bool, decode_only: bool) -> Vec<model::memplan::Demand> {
    use model::memplan::{Category, Demand};
    let n = crate::model::activation_rows(b, t, decode_only);
    let flash = runs_flash(cfg, decode_only);
    let activations = residual_bytes(cfg, shard, n, train) + activation_scratch_bytes(cfg, n, t, b, decode_only, flash) + backward_scratch_bytes(cfg, n, t, b, train, flash) + head_bytes(cfg, shard, n, train, decode_only);
    vec![Demand::fixed(Category::Weights, "weights", weight_bytes(cfg, shard, dt, train)), Demand::fixed(Category::Kv, "kv cache", kv_cache_bytes(cfg, t, train)), Demand::fixed(Category::Activations, "activations and scratch", activations)]
}

/// The allowance [`lora_training_demands`] makes for the buffers of a few words a build allocates.
const SMALL_BUFFER_BYTES: u64 = 64 << 10;

/// What a LoRA training build of `cfg` (whose adapter `cfg.lora` names) holds on one device for `b` rows of `t`
/// tokens over a frozen base at `dt`, keeping its layer boundaries as `saved` says, as the planner's categories, each
/// term the buffer [`crate::Qwen`]'s training build allocates:
///
/// * weights: the frozen base at its tier (the tables in bf16 over a narrow base) and the adapter's fp32 values;
/// * gradients and optimiser state: the adapter's fp32 gradients and its two AdamW moments (a frozen base has none);
/// * activations, by block length: the layer-boundary residuals (fp32, or the saved format's stream and its two
///   fp32 working rows), the one layer's scratch every layer shares (the backward rebuilds a layer from its
///   boundary), the attention scores where `flash` is false (`[H, T, T]` each for scores, probabilities and their
///   gradient), the backward scratch, and the logits and their gradient for one row chunk of the head.
///
/// `flash` is whether the device runs the flash attention kernels ([`runs_flash`] on a card, the device's own
/// answer once one is open). Device memory a backend keeps beside what the model holds (freed blocks it caches, a
/// library's workspace) is the caller's to add: it depends on the card, not on the model.
///
/// # Errors
/// `cfg` names no adapter, or `saved` keeps every layer's activations, which this trainer does not implement.
pub fn lora_training_demands(cfg: &QwenConfig, dt: Dtype, b: u32, t: u32, saved: model::saved::SavedPolicy, flash: bool) -> Result<Vec<model::memplan::Demand>, String> {
    use model::memplan::{Category, Demand};
    use model::saved::{SavedFormat, SavedLayout};
    if cfg.lora.is_none() {
        return Err("qwen3: a LoRA training footprint needs an adapter configuration".to_string());
    }
    if !saved.recomputes() {
        return Err("qwen3 training recomputes each layer in the backward; SavedPolicy::Keep is not available".to_string());
    }
    let shard = Shard::whole(cfg.n_layers as usize);
    let n = crate::model::activation_rows(b, t, false);
    let trainable: u64 = crate::shard_param_list(cfg, &shard).into_iter().filter(|(name, _)| model::adapter::device::is_adapter_param(name)).map(|(_, numel)| numel as u64).sum();
    let boundaries = match saved.boundary_format() {
        SavedFormat::F32 => residual_bytes(cfg, &shard, n, true),
        format => {
            let layout = SavedLayout { boundaries: cfg.n_layers as u64 + 1, rows: n, width: cfg.d_model as u64 };
            // The narrow stream, the two fp32 working rows it decodes into, and the backward's pair.
            layout.stream_bytes(format) + layout.working_bytes() + 2 * n * cfg.d_model as u64 * 4
        }
    };
    // The adapter's `[n, r]` projection and its gradient beside the layer scratch, its `[max_out, r]` ones, and the
    // few-word buffers (rope frequencies, decode tables, the dummies of unused bindings) as one allowance.
    let rank = cfg.lora.as_ref().map_or(0, |l| l.rank as u64);
    let max_out = (cfg.q_dim() as u64).max(cfg.d_ff as u64).max(cfg.d_model as u64).max(cfg.kv_dim() as u64);
    let scratch = activation_scratch_bytes(cfg, n, t, b, false, flash) + backward_scratch_bytes(cfg, n, t, b, true, flash) + (2 * n + max_out) * rank * 4 + SMALL_BUFFER_BYTES;
    Ok(vec![
        Demand::fixed(Category::Weights, "frozen base and adapter values", weight_bytes(cfg, &shard, dt, true)),
        Demand::fixed(Category::Grads, "adapter gradients", trainable * 4),
        Demand::fixed(Category::OptimizerState, "adapter AdamW moments", trainable * 8),
        Demand::fixed(Category::Activations, format!("{} boundaries of {t} tokens", saved.label()), boundaries),
        Demand::fixed(Category::Activations, format!("layer and backward scratch of {t} tokens"), scratch),
        Demand::fixed(Category::Activations, format!("logits and their gradient ({} vocabulary)", cfg.vocab), head_bytes(cfg, &shard, n, true, false)),
    ])
}

/// An inference build that encodes up to `t` rows, priced the way a
/// layer-range placement prices it: `(per_layer, per_stage)`. `per_layer` is
/// one layer's weights at `dt` and the residual boundary it adds; `per_stage`
/// is what every stage holds once, whatever its layer count - the activation
/// scratch its layers share, its first residual boundary and the KV cache.
/// The same terms [`footprint_demands`] sums for one shard.
pub fn encode_stage_split(cfg: &QwenConfig, dt: Dtype, t: u32) -> (u64, u64) {
    let one_layer = Shard { start: 0, end: 1, embed: false, head: false, gpu_index: Shard::ANY_GPU };
    let n = crate::model::activation_rows(1, t, false);
    let boundary = n * cfg.d_model as u64 * 4;
    let per_layer = weight_bytes(cfg, &one_layer, dt, false) + boundary;
    let per_stage = activation_scratch_bytes(cfg, n, t, 1, false, runs_flash(cfg, false)) + boundary + kv_cache_bytes(cfg, t, false);
    (per_layer, per_stage)
}

/// The parameter count from which a decoder is served with int8 linears
/// unless its caller says otherwise: a 6-8B decoder is 24-32 GB of fp32
/// weights, more than a 24 GB card holds.
pub const INT8_WEIGHTS_FROM_PARAMS: u64 = 6_000_000_000;

/// Whether `cfg` is large enough ([`INT8_WEIGHTS_FROM_PARAMS`] parameters and
/// up) for [`default_weight_tier`] to fall to int8 where its weights do not fit.
fn int8_by_default(cfg: &QwenConfig) -> bool {
    cfg.param_list().iter().map(|(_, n)| *n as u64).sum::<u64>() >= INT8_WEIGHTS_FROM_PARAMS
}

/// Room beyond a checkpoint's weights that serving them still needs on a device (a KV pool for a working
/// context, row buffers, library workspace).
pub const SERVE_HEADROOM_BYTES: u64 = 8 << 30;

/// Device bytes of `cfg`'s weights with the linears and the head held at `tier` (one of the floating tiers or
/// int8, with its group scales) and everything else (embeddings, norms) at fp32.
pub fn held_weight_bytes(cfg: &QwenConfig, tier: Dtype) -> u64 {
    held_weight_bytes_by(cfg, |_| tier)
}

/// Device bytes the serving engine holds for `cfg`'s weights with each linear and the head at the tier `tier_of`
/// names for it (packed words plus the narrow tiers' group scales) and the embedding and norms at fp32. A tied head
/// is the embedding table a second time: the engine gathers the embedding from the fp32 table and uploads the head
/// from it at the head's tier.
pub fn held_weight_bytes_by(cfg: &QwenConfig, tier_of: impl Fn(&str) -> Dtype) -> u64 {
    let at = |tier: Dtype, n: u64| match tier {
        Dtype::BF16 | Dtype::F16 => n * 2,
        Dtype::I8 => n + (n / 32) * 4,
        Dtype::NF4 | Dtype::Q4 => n / 2 + (n / 32) * 4,
        // M14: the affine K-quant tiers (GGUF Q4_K / Q5_K), priced at the
        // bytes the device buffer actually holds - the packed codes (4.5 /
        // 8 bits per element), the 6-bit scale/min pair per 32-group (18
        // bytes per 32 groups), and one f16 (d, dmin) pair per 8 groups.
        // Pricing these at the fp32 promotion (n * 4) is what made every
        // GGUF plan ~21 GiB on a 4B model.
        Dtype::Q4K => n * 9 / 16 + (n / 32) * 18 / 8 + (n / 256) * 4,
        Dtype::Q8K => n + (n / 32) * 18 / 8 + (n / 256) * 4,
        _ => n * 4,
    };
    let head = cfg.head_weight();
    let tables: u64 = cfg
        .param_list()
        .into_iter()
        .map(|(name, n)| {
            let n = n as u64;
            if crate::q8::Q8::is_i8_linear(&name) || (name == head && !cfg.tie_embeddings) {
                at(tier_of(&name), n)
            } else {
                n * 4
            }
        })
        .sum();
    let tied_head = if cfg.tie_embeddings { at(tier_of(head), u64::from(cfg.vocab) * u64::from(cfg.d_model)) } else { 0 };
    tables + tied_head
}

/// The tier a caller that did not choose one serves `cfg`'s linears at, given the floating tier the checkpoint
/// stores them in ([`crate::stored_weight_tier`]) and `device_bytes`, the memory the weights may use on the
/// largest card (`None` when unknown). The stored tier is kept (a bf16 checkpoint is served bf16, an fp32 one
/// fp32; any other stored encoding is decoded to fp32), so the device reads the checkpoint's own values at their
/// own width. int8 is chosen only for a checkpoint of [`INT8_WEIGHTS_FROM_PARAMS`] parameters or more whose
/// stored-tier weights plus [`SERVE_HEADROOM_BYTES`] do not fit the card, or on an unknown card: it is the least
/// accurate tier (on the 8B, KL 7.5e-2 from the exact next-token distribution against 1.6e-4 for fp32 and 1.9e-4
/// for bf16 weights on tensor cores, knowledge 326). A device that cannot hold a half tier widens it to fp32 at
/// upload and says so. The serving resident, the SDK pipeline and the perf targets all decide by this.
pub fn default_weight_tier(cfg: &QwenConfig, stored: Dtype, device_bytes: Option<u64>) -> Dtype {
    // M14: a checkpoint that already stores a packed serving tier (a GGUF's
    // Q4_K / Q5_K, brain-int4, or brain-int8) keeps it - the same tier the
    // CPU backend can now execute natively (`kquant_ops`) and the fit
    // estimator prices at its packed size (`Weight::bytes_for`), not the
    // 4-bytes-per-parameter fp32 promotion that previously made every GGUF
    // plan ~21 GiB. The floating tiers (BF16/F16) keep their old rule.
    let stored = match stored {
        Dtype::BF16 | Dtype::F16 | Dtype::Q4K | Dtype::Q8K | Dtype::Q4 | Dtype::I8 => stored,
        _ => Dtype::F32,
    };
    let fits = |bytes: u64| bytes >= held_weight_bytes(cfg, stored) + SERVE_HEADROOM_BYTES;
    if int8_by_default(cfg) && !device_bytes.is_some_and(fits) {
        Dtype::I8
    } else {
        stored
    }
}

/// The weight tiers the Qwen3 checkpoint at `path` is served at by default: [`default_weight_tier`] of the tier the
/// checkpoint stores its linears in, for `device_bytes` (the memory the weights may use on the largest card, `None`
/// when unknown). A checkpoint that cannot be opened yet is planned at fp32; opening it later reports the real error.
pub fn default_weight_policy(path: &str, device_bytes: Option<u64>) -> model::ops::TierPolicy {
    match crate::open_checkpoint(path) {
        Ok((checkpoint, src)) => model::ops::TierPolicy::uniform(default_weight_tier(&checkpoint, crate::source_weight_tier(&checkpoint, &*src), device_bytes)),
        Err(_) => model::ops::TierPolicy::uniform(Dtype::F32),
    }
}

/// Declares [`estimate_vram_bytes`]'s figure to [`gpu_core::devices::place`]
/// before `f` builds anything, so an over-budget checkpoint is refused BY
/// NAME (which part, how many bytes, what each card has free - see
/// `crates/cli/src/placement.rs`'s own `an_impossible_model_is_refused_legibly`
/// test for the shape of that refusal) instead of `f` dispatching a doomed
/// device allocation that panics the backend mid-upload.
///
/// Skipped - `f` runs exactly as it always did - when a device is ALREADY
/// pinned ([`gpu_core::devices::current_gpu`] is `Some`): an explicit
/// `--device`, or a caller that already scoped its own placement decision
/// (e.g. a residency `activate()` that ran its own estimate through
/// `residency::place::pick_device` before calling this). Asking the
/// automatic placer a second time there could only disagree with a decision
/// already made and acted on - the same rule `crates/qwen3vl/src/caps.rs`'s
/// own `place_and_build` follows for the identical reason.
pub fn place_and_build<R>(cfg: &QwenConfig, shard: &Shard, dt: Dtype, b: u32, t: u32, train: bool, decode_only: bool, name: &str, f: impl FnOnce() -> R) -> Result<R, String> {
    if gpu_core::devices::current_gpu().is_some() {
        return Ok(f());
    }
    let bytes = estimate_vram_bytes(cfg, shard, dt, b, t, train, decode_only);
    let need = gpu_core::devices::Need::sized(name, bytes, 0);
    let homes = gpu_core::devices::place(&[need])?;
    homes.run(name, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg(n_layers: u32, d_model: u32, d_ff: u32) -> QwenConfig {
        QwenConfig {
            vocab: 32000,
            block_size: 4096,
            n_layers,
            d_model,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: d_model / 32,
            d_ff,
            rope_theta: 1_000_000.0,
            rms_eps: 1e-6,
            max_position_embeddings: 4096,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            lora: None,
            rope_scaling: None,
        }
    }

    /// The whole point of int8 existing: it must estimate smaller than fp32
    /// for the same shape - the weight term is the dominant one at real scale.
    #[test]
    fn int8_estimates_smaller_than_fp32() {
        let cfg = tiny_cfg(8, 2048, 5632);
        let shard = Shard::whole(cfg.n_layers as usize);
        let fp32 = estimate_vram_bytes(&cfg, &shard, Dtype::F32, 1, 512, false, false);
        let int8 = estimate_vram_bytes(&cfg, &shard, Dtype::I8, 1, 512, false, false);
        assert!(int8 < fp32, "int8 ({int8}) must be smaller than fp32 ({fp32})");
    }

    /// THE regression this module exists for: a bigger checkpoint (double the
    /// layers) must estimate a bigger footprint - a hardcoded constant, or no
    /// estimate at all (the bug this fixes), cannot tell an 8B-class
    /// checkpoint apart from a smaller one before dispatching a doomed
    /// device allocation.
    #[test]
    fn a_bigger_decoder_estimates_a_bigger_footprint() {
        let small = tiny_cfg(8, 2048, 5632);
        let big = tiny_cfg(16, 2048, 5632);
        let small_shard = Shard::whole(small.n_layers as usize);
        let big_shard = Shard::whole(big.n_layers as usize);
        let small_bytes = estimate_vram_bytes(&small, &small_shard, Dtype::F32, 1, 512, false, false);
        let big_bytes = estimate_vram_bytes(&big, &big_shard, Dtype::F32, 1, 512, false, false);
        assert!(big_bytes > small_bytes * 3 / 2, "doubling n_layers must noticeably grow the estimate: small={small_bytes} big={big_bytes}");
    }

    /// The KV-cache term alone must scale with `t` (context length) - a
    /// caller sizing a smaller context should not pay for a bigger one.
    #[test]
    fn a_longer_context_estimates_a_bigger_footprint() {
        let cfg = tiny_cfg(8, 2048, 5632);
        let shard = Shard::whole(cfg.n_layers as usize);
        let short = estimate_vram_bytes(&cfg, &shard, Dtype::F32, 1, 512, false, false);
        let long = estimate_vram_bytes(&cfg, &shard, Dtype::F32, 1, 8192, false, false);
        assert!(long > short, "a longer context must estimate a bigger footprint: short={short} long={long}");
    }

    /// Matches [`crate::Qwen::linear_weight_bytes`]'s own packing formula at a
    /// hand-computable shape: `n=k=32` (one scale group exactly) at
    /// `Dtype::I8` packs to `32*32/4` (4 int8 per word) `* 4` bytes/word,
    /// plus one `[1,1]` scale group at 4 bytes, per linear.
    #[test]
    fn quantized_linear_bytes_matches_the_known_int8_packing_formula() {
        // n_heads=n_kv_heads=1, head_dim=32=d_model=d_ff -> every one of the
        // 7 linears is [32,32].
        let cfg = QwenConfig {
            vocab: 1,
            block_size: 1,
            n_layers: 3,
            d_model: 32,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 32,
            d_ff: 32,
            rope_theta: 1.0,
            rms_eps: 1e-6,
            max_position_embeddings: 1,
            tie_embeddings: true,
            qk_norm: false,
            attn_bias: false,
            lora: None,
            rope_scaling: None,
        };
        let one_linear_i8 = (32u64 * 32 / 4) * 4 + model::int8::scale_len(32, 32) as u64 * 4;
        let shard = Shard::whole(3);
        assert_eq!(quantized_linear_bytes(&cfg, &shard, Dtype::I8), one_linear_i8 * 7 * 3, "3 owned layers x 7 linears each");

        // A partial shard (2 of 3 layers) must only pack ITS OWN layers.
        let partial = Shard { start: 0, end: 2, embed: true, head: false, gpu_index: Shard::ANY_GPU };
        assert_eq!(quantized_linear_bytes(&cfg, &partial, Dtype::I8), one_linear_i8 * 7 * 2, "2 owned layers x 7 linears each");
    }

    /// The per-layer activation scratch is ONE shared copy (`new_impl`'s own
    /// `Layer::clone` pooling), training included - the backward recomputes
    /// each layer - so it must NOT grow with `n_layers`, unlike the
    /// weight/KV-cache/residual terms above it.
    #[test]
    fn activation_scratch_does_not_scale_with_layer_count() {
        let cfg8 = tiny_cfg(8, 2048, 5632);
        let cfg16 = tiny_cfg(16, 2048, 5632);
        let a = activation_scratch_bytes(&cfg8, 512, 512, 1, false, runs_flash(&cfg8, false));
        let b = activation_scratch_bytes(&cfg16, 512, 512, 1, false, runs_flash(&cfg16, false));
        assert_eq!(a, b, "scratch must be pooled to one copy regardless of layer count: 8L={a} 16L={b}");
    }

    /// A training build on the flash path keeps nothing `T²`-sized: past the
    /// weights, doubling the block at most doubles the estimate. The
    /// materialised chain's `[H,T,T]` slabs would quadruple it.
    #[test]
    fn training_memory_is_linear_in_the_block() {
        let cfg = QwenConfig { lora: Some(crate::LoraCfg::attn(8, 16.0)), ..QwenConfig::qwen3_0_6b() };
        let shard = Shard::whole(cfg.n_layers as usize);
        let weights = weight_bytes(&cfg, &shard, Dtype::F32, true);
        let at = |t: u32| estimate_vram_bytes(&cfg, &shard, Dtype::F32, 1, t, true, false) - weights;
        for t in [4096u32, 8192] {
            assert!(at(2 * t) <= 2 * at(t) + (1 << 20), "T={t}: {} -> {} bytes past the weights", at(t), at(2 * t));
        }
    }

    /// An inference build on the flash path keeps no `[H,T,T]` slab either:
    /// past the weights, doubling the context at most doubles the estimate (the
    /// KV cache, residual stream and activations are all linear in it), where
    /// the materialised chain's score pair would quadruple it. A head wider
    /// than the kernel tiles still materialises the slab.
    #[test]
    fn inference_memory_is_linear_in_the_context_on_the_flash_path() {
        let cfg = QwenConfig::qwen3_0_6b();
        let shard = Shard::whole(cfg.n_layers as usize);
        let weights = weight_bytes(&cfg, &shard, Dtype::F32, false);
        let at = |cfg: &QwenConfig, t: u32| estimate_vram_bytes(cfg, &shard, Dtype::F32, 1, t, false, false) - weight_bytes(cfg, &shard, Dtype::F32, false);
        for t in [4096u32, 8192] {
            assert!(at(&cfg, 2 * t) <= 2 * at(&cfg, t) + (1 << 20), "T={t}: {} -> {} bytes past the weights {weights}", at(&cfg, t), at(&cfg, 2 * t));
        }
        let wide = QwenConfig { head_dim: 256, ..cfg };
        assert!(at(&wide, 8192) > 2 * at(&wide, 4096) + (1 << 30), "a head past the flash tile keeps the materialised score pair");
    }

    /// A LoRA build over a bf16 base holds the embedding and the head in bf16:
    /// the estimate drops by half of each table, and an inference build or an
    /// fp32 base keeps them at fp32.
    #[test]
    fn a_lora_bf16_build_counts_its_tables_at_two_bytes() {
        let cfg = QwenConfig { lora: Some(crate::LoraCfg::attn(8, 16.0)), tie_embeddings: false, ..QwenConfig::qwen3_0_6b() };
        let shard = Shard::whole(cfg.n_layers as usize);
        let tables = 2 * cfg.vocab as u64 * cfg.d_model as u64;
        assert_eq!(weight_bytes(&cfg, &shard, Dtype::BF16, false) - weight_bytes(&cfg, &shard, Dtype::BF16, true), tables * 2, "training halves both tables");
        assert_eq!(weight_bytes(&cfg, &shard, Dtype::F32, false), weight_bytes(&cfg, &shard, Dtype::F32, true), "fp32 keeps them");
    }

    /// An int8 base holds a decoder at about a quarter of the fp32 bytes (a
    /// byte per weight and a scale per 32), beside bf16 tables.
    #[test]
    fn an_int8_lora_base_is_smaller_than_a_bf16_one() {
        let cfg = QwenConfig { lora: Some(crate::LoraCfg::attn(8, 16.0)), ..QwenConfig::qwen3_0_6b() };
        let shard = Shard::whole(cfg.n_layers as usize);
        let (bf16, int8) = (weight_bytes(&cfg, &shard, Dtype::BF16, true), weight_bytes(&cfg, &shard, Dtype::I8, true));
        let linears = quantized_linear_bytes(&cfg, &shard, Dtype::BF16);
        assert!(int8 < bf16 && bf16 - int8 > linears / 3, "{int8} vs {bf16}");
        assert_eq!(int8 - quantized_linear_bytes(&cfg, &shard, Dtype::I8), bf16 - linears, "the tables and norms are the same under either base");
    }

    /// A `decode_only` build skips the head logits/`d_logits` buffer entirely
    /// (the LM head runs host-side there).
    #[test]
    fn decode_only_has_no_head_buffer() {
        let cfg = tiny_cfg(8, 2048, 5632);
        let shard = Shard::whole(cfg.n_layers as usize);
        assert_eq!(head_bytes(&cfg, &shard, 1, false, true), 0);
        assert!(head_bytes(&cfg, &shard, 1, false, false) > 0);
    }

    /// Qwen3-4B: 36 layers, width 2560, MLP 9728, 32 query and 8 key/value heads of 128, a 151,936-token vocabulary.
    fn qwen3_4b() -> QwenConfig {
        let targets = crate::finetune::LORA_TARGETS.iter().map(|t| t.to_string()).collect();
        QwenConfig { vocab: 151_936, block_size: 40_960, n_layers: 36, d_model: 2560, n_heads: 32, n_kv_heads: 8, head_dim: 128, d_ff: 9728, max_position_embeddings: 40_960, lora: Some(crate::LoraCfg::new(16, 32.0, targets)), ..tiny_cfg(36, 2560, 9728) }
    }

    /// The served LoRA run of the pilot report (Qwen3-4B, bf16 base, every projection at rank 16): what one
    /// 32,768-token row holds, against the 25.9 GiB the calibrated rate admitted it at. Each long-row term is the
    /// trainer's own buffer: the fp32 boundaries of 37 layer edges, the one layer's scratch, the backward scratch,
    /// and the logits of one 3,520-row chunk of the head with their gradient.
    #[test]
    fn a_long_row_of_qwen3_4b_holds_its_boundaries_scratch_and_logits() {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let total = |t: u32| model::memplan::device_total(&lora_training_demands(&qwen3_4b(), Dtype::BF16, 1, t, model::saved::SavedPolicy::Recompute, true).unwrap());
        let (at_16k, at_32k) = (total(16_384), total(32_768));
        assert!((24.0..28.0).contains(&gib(at_16k)), "16K tokens: {:.2} GiB", gib(at_16k));
        assert!((37.0..41.0).contains(&gib(at_32k)), "32K tokens: {:.2} GiB", gib(at_32k));
        // About 0.9 MB a token beyond the head's fixed chunk: what a longer row costs this trainer.
        let per_token = (at_32k - at_16k) / 16_384;
        assert!((800_000..1_000_000).contains(&per_token), "{per_token} B a token");
        let narrow = model::memplan::device_total(&lora_training_demands(&qwen3_4b(), Dtype::BF16, 1, 32_768, model::saved::SavedPolicy::Quantized(model::saved::SavedFormat::Fp8), true).unwrap());
        assert!(at_32k - narrow > 7 << 30, "fp8 boundaries save most of the fp32 stream: {:.2} GiB", gib(at_32k - narrow));
    }
}
