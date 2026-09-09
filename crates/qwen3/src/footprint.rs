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
//!   "regardless of `shard`" - its own comment) for EVERY build, not only a
//!   `decode_only` one, so this term is never skipped here either.
//! - The per-layer-boundary residual stream (`res`/`dres`): one `[n, d_model]`
//!   buffer per boundary this shard owns (`shard.start..=shard.end`), doubled
//!   when `train` (the backward-only `dres` twin).
//! - The per-layer activation scratch (`Layer`): allocated ONCE and shared
//!   across every non-owned-for-backward layer when `!train` (`new_impl`'s
//!   own `shared`/`Layer::clone` pooling), or once PER owned layer when
//!   `train` (backward needs every layer's own saved activations resident at
//!   once).
//! - The head-only logits/`d_logits` buffer (`shard.head`, skipped entirely
//!   on a `decode_only` build - the LM head runs host-side there).
//!
//! Not counted: `tokens`/`targets` (`[n]` id buffers), the LoRA scratch
//! (`r · max_out`), and the handful of `[1]`-sized dummies every non-owned
//! stage carries - all a rounding error next to the terms above, and what
//! `crates/cli/src/placement.rs::HEADROOM` (1 GiB, reserved on top of every
//! automatic placement) exists to absorb, the same way it already does for
//! every other model that declares a sized [`gpu_core::devices::Need`].

use gpu_core::select::Dtype;

use crate::config::QwenConfig;
use crate::model::Shard;

/// Bytes this shard's 7-per-owned-layer linears occupy at `dt`, packed
/// exactly as [`crate::Qwen::linear_weight_bytes`] packs an already-built
/// model's [`crate::model::Weight`]s: `elems.div_ceil(per_word) * 4` plus the
/// `[n, k/32]` f32 group scale the `I8`/`Q4` tiers carry.
fn quantized_linear_bytes(cfg: &QwenConfig, shard: &Shard, dt: Dtype) -> u64 {
    let per_word = dt.per_word() as u64;
    let pack = |n: u64, k: u64| -> u64 {
        let packed = (n * k).div_ceil(per_word) * 4;
        let scale = model::int8::scale_len(n as usize, k as usize) as u64 * 4;
        packed + scale
    };
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
fn weight_bytes(cfg: &QwenConfig, shard: &Shard, dt: Dtype) -> u64 {
    let quantized = dt != Dtype::F32;
    let base: u64 = crate::shard_param_list(cfg, shard)
        .into_iter()
        .filter(|(name, _)| !(quantized && crate::q8::Q8::is_i8_linear(name)))
        .map(|(_, c)| c as u64 * 4)
        .sum();
    if !quantized {
        return base;
    }
    base + quantized_linear_bytes(cfg, shard, dt)
}

/// The plain (non-paged) KV cache: one `[t, kv_dim]` key + value buffer per
/// layer of the WHOLE checkpoint, always - see this module's doc.
fn kv_cache_bytes(cfg: &QwenConfig, t: u32) -> u64 {
    2 * cfg.n_layers as u64 * t as u64 * cfg.kv_dim() as u64 * 4
}

/// The per-layer-boundary residual stream: one `[n, d_model]` buffer per
/// boundary this shard owns (`shard.start..=shard.end`, inclusive - see
/// `new_impl`'s own `0..=cfg.n_layers` loop), doubled under `train` for the
/// backward-only `dres` twin.
fn residual_bytes(cfg: &QwenConfig, shard: &Shard, n: u64, train: bool) -> u64 {
    let live_boundaries = (shard.end.saturating_sub(shard.start) + 1) as u64;
    let one = live_boundaries * n * cfg.d_model as u64 * 4;
    if train {
        one * 2
    } else {
        one
    }
}

/// The per-layer activation scratch (`Layer`): one shared copy when `!train`
/// (`new_impl`'s own buffer-pooling - every layer reads/overwrites the SAME
/// physical buffers since only `res[l+1]` ever crosses a layer boundary), or
/// one copy PER owned layer when `train` (backward needs every layer's own
/// saved forward activations resident at once).
fn activation_scratch_bytes(cfg: &QwenConfig, shard: &Shard, n: u64, t: u32, b: u32, train: bool, decode_only: bool) -> u64 {
    let d = cfg.d_model as u64;
    let hq = cfg.q_dim() as u64;
    let hkv = cfg.kv_dim() as u64;
    let ff = cfg.d_ff as u64;
    let bht2 = if decode_only { cfg.n_heads as u64 * t as u64 } else { b as u64 * cfg.n_heads as u64 * t as u64 * t as u64 };
    // xn1, q_pre, q, k_pre, k, v, probs, ctx, xmid, xn2, gate_pre, up, h
    let one_layer = n * d // xn1
        + n * hq // q_pre
        + n * hq // q
        + n * hkv // k_pre
        + n * hkv // k
        + n * hkv // v
        + bht2 // probs
        + n * hq // ctx
        + n * d // xmid
        + n * d // xn2
        + n * ff // gate_pre
        + n * ff // up
        + n * ff; // h
    let layer_count = if train { shard.end.saturating_sub(shard.start) as u64 } else { 1 };
    one_layer * layer_count * 4
}

/// The head-only logits/`d_logits` buffer: `[n, vocab]`, doubled under
/// `train` for `d_logits`, skipped entirely on a `decode_only` build (the LM
/// head runs host-side there - `new_impl`'s own `hd_or_dummy` gate) or on a
/// non-head stage.
fn head_bytes(cfg: &QwenConfig, shard: &Shard, n: u64, train: bool, decode_only: bool) -> u64 {
    if !shard.head || decode_only {
        return 0;
    }
    let one = n * cfg.vocab as u64 * 4;
    if train {
        one * 2
    } else {
        one
    }
}

/// The device bytes a [`crate::Qwen::new_impl`] build (every constructor in
/// this crate funnels through it) will occupy, derived from `cfg`/`shard`
/// themselves so it scales with whatever checkpoint and shard shape is
/// actually being built - see this module's doc for what is and is not
/// counted.
pub fn estimate_vram_bytes(cfg: &QwenConfig, shard: &Shard, dt: Dtype, b: u32, t: u32, train: bool, decode_only: bool) -> u64 {
    let n = if decode_only { 1u64 } else { b as u64 * t as u64 };
    weight_bytes(cfg, shard, dt)
        + kv_cache_bytes(cfg, t)
        + residual_bytes(cfg, shard, n, train)
        + activation_scratch_bytes(cfg, shard, n, t, b, train, decode_only)
        + head_bytes(cfg, shard, n, train, decode_only)
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
        };
        let one_linear_i8 = (32u64 * 32 / 4) * 4 + model::int8::scale_len(32, 32) as u64 * 4;
        let shard = Shard::whole(3);
        assert_eq!(quantized_linear_bytes(&cfg, &shard, Dtype::I8), one_linear_i8 * 7 * 3, "3 owned layers x 7 linears each");

        // A partial shard (2 of 3 layers) must only pack ITS OWN layers.
        let partial = Shard { start: 0, end: 2, embed: true, head: false, gpu_index: Shard::ANY_GPU };
        assert_eq!(quantized_linear_bytes(&cfg, &partial, Dtype::I8), one_linear_i8 * 7 * 2, "2 owned layers x 7 linears each");
    }

    /// `!train` pools the per-layer activation scratch into ONE shared copy
    /// (`new_impl`'s own `Layer::clone` pooling) - so it must NOT grow with
    /// `n_layers`, unlike the weight/KV-cache/residual terms above it.
    #[test]
    fn inference_activation_scratch_does_not_scale_with_layer_count() {
        let cfg8 = tiny_cfg(8, 2048, 5632);
        let cfg16 = tiny_cfg(16, 2048, 5632);
        let a = activation_scratch_bytes(&cfg8, &Shard::whole(8), 512, 512, 1, false, false);
        let b = activation_scratch_bytes(&cfg16, &Shard::whole(16), 512, 512, 1, false, false);
        assert_eq!(a, b, "inference (!train) scratch must be pooled to one copy regardless of layer count: 8L={a} 16L={b}");
    }

    /// `train` does NOT pool - backward needs every owned layer's own saved
    /// activations resident at once, so doubling layers must double this term.
    #[test]
    fn training_activation_scratch_scales_with_layer_count() {
        let cfg8 = tiny_cfg(8, 2048, 5632);
        let cfg16 = tiny_cfg(16, 2048, 5632);
        let a = activation_scratch_bytes(&cfg8, &Shard::whole(8), 512, 512, 1, true, false);
        let b = activation_scratch_bytes(&cfg16, &Shard::whole(16), 512, 512, 1, true, false);
        assert_eq!(b, a * 2, "training scratch must scale with owned layer count: 8L={a} 16L={b}");
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
}
