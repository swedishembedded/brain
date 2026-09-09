// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A pre-flight VRAM estimate for a [`crate::model::MoondreamModel`] build,
//! derived from the checkpoint's OWN [`MoondreamConfig`] rather than a
//! constant hand-derived once for the released preview config.
//!
//! `crates/cli/src/resident_moondream3.rs::Moondream3Resident::estimate`
//! needs "how many device bytes will this build occupy" before a `Gpu` is
//! built, to budget a device through `residency::place::pick_device`. Before
//! this module existed that number was a flat constant - safe in practice
//! ONLY because [`MoondreamConfig::from_json`] already REJECTS BY NAME any
//! checkpoint whose dimensions disagree with the preview architecture (see
//! that function's own doc), so no checkpoint this resident will ever
//! successfully load could actually need a different figure. This module
//! removes that reliance on a second, unenforced invariant: the estimate now
//! reads the same `cfg` the build itself reads, the same way
//! `crates/qwen3vl/src/footprint.rs` does for a family that (unlike this one)
//! ships several real checkpoint sizes a caller can point `--weights` at.
//!
//! # What is counted, and at what tier
//!
//! - The decoder's attention (`qkv`/`proj`, fused `[3·dim,dim]`/`[dim,dim]`)
//!   and dense-layer FFN (`fc1`/`fc2`, layers `0..moe.start_layer` - GeGLU
//!   shape `[2·ff_dim,dim]`/`[dim,ff_dim]`, see [`crate::import`]'s module
//!   doc) - always fp32, neither is ever quantized by [`crate::model::
//!   Precision::Int8`] (see that variant's own doc: "int8 experts").
//! - The MoE experts (`w_h`/`w_g`/`w_down`, layers `moe.start_layer..n_layers`)
//!   - fp32 normally, one byte per element plus a per-output-channel f32
//!     scale under `Precision::Int8` (`crate::decoder`'s own comment: "
//!     per-channel-quantized and dispatched through `moe_linear_gated_i8`" -
//!     NOT `qwen3`'s `[n,k/32]` GROUP scale, a different packing).
//! - Token embedding / untied `lm_head` (+ bias) - always fp32.
//! - The vision tower (SigLIP ViT) + connector - always fp32, a small
//!   fraction of the weights either way.
//! - Per-block activation scratch (`crate::decoder::BlockScratch`): ONE
//!   shared instance under `Precision::Int8`, but [`crate::model::
//!   build_blocks`] gives EVERY block its own before `Precision::Int8`
//!   replaces them (`MoondreamModel::new_with`'s `share_scratch` call only
//!   runs for that tier) - so fp32 pays `n_layers` copies, not one.
//! - The per-layer KV cache (`crate::decoder::KvCache`) `generate_kv` builds
//!   for a request - not held by the resident right after `activate`, but
//!   real device memory the first `caption` call WILL need, so a placement
//!   decision that ignores it can still OOM on the first real request.
//!
//! Swedish Embedded AB implements pre-flight VRAM sizing for large model
//! deployments. If your team needs a placement policy that refuses a
//! checkpoint that does not fit instead of crashing the driver, you can
//! procure our services by sending an email to info@swedishembedded.com.

use crate::config::MoondreamConfig;
use crate::model::Precision;

/// Bytes one dense decoder layer's weights occupy (attention + dense GeGLU
/// FFN), always fp32.
fn dense_layer_bytes(cfg: &MoondreamConfig) -> u64 {
    let d = cfg.dim as u64;
    let ff = cfg.ff_dim as u64;
    let attn = 3 * d * d + d * d; // fused qkv [3d,d] + proj [d,d]
    let ffn = 2 * ff * d + d * ff; // fc1 [2*ff,d] (GeGLU halves) + fc2 [d,ff]
    (attn + ffn) * 4
}

/// Bytes one MoE decoder layer's weights occupy (attention, always fp32,
/// plus every expert + router at `precision`'s tier).
fn moe_layer_bytes(cfg: &MoondreamConfig, precision: Precision) -> u64 {
    let d = cfg.dim as u64;
    let inner = cfg.moe.inner_dim as u64;
    let e = cfg.moe.num_experts as u64;
    let attn = (3 * d * d + d * d) * 4; // fused qkv + proj, always fp32
    let router = e * d * 4; // [E,d], always fp32
    let per_expert = match precision {
        // w_h [inner,d], w_g [inner,d], w_down [d,inner]: 1 byte/elem plus one
        // f32 scale per output row (per-channel, not `qwen3`'s [n,k/32] group).
        Precision::Int8 => (inner * d + inner * d + d * inner) + (inner + inner + d) * 4,
        Precision::Fp32 => (inner * d + inner * d + d * inner) * 4,
    };
    attn + router + e * per_expert
}

/// Bytes the whole decoder's weights occupy: dense layers `0..moe.start_layer`
/// plus MoE layers `moe.start_layer..n_layers`, plus the (always fp32,
/// untied) token embedding and `lm_head`.
fn decoder_weight_bytes(cfg: &MoondreamConfig, precision: Precision) -> u64 {
    let dense_layers = cfg.moe.start_layer as u64;
    let moe_layers = (cfg.n_layers - cfg.moe.start_layer) as u64;
    let embed = 2 * cfg.vocab as u64 * cfg.dim as u64 * 4; // tok.weight + lm_head.weight, untied
    dense_layers * dense_layer_bytes(cfg) + moe_layers * moe_layer_bytes(cfg, precision) + embed
}

/// Bytes the SigLIP ViT + connector occupy - always fp32.
fn vision_and_connector_bytes(cfg: &MoondreamConfig) -> u64 {
    let v = &cfg.vision;
    let (vd, vff) = (v.dim as u64, v.ff_dim as u64);
    let block = 3 * vd * vd + vd * vd + vff * vd + vd * vff; // fused qkv + proj + fc1 + fc2
    let patch_embed = v.patch_vec() as u64 * vd;
    let blocks = block * v.n_layers as u64;
    let (pin, pout, cin) = (cfg.proj_inner as u64, cfg.proj_out as u64, cfg.connector_in() as u64);
    let connector = pin * cin + pout * pin; // fc1 [proj_inner,connector_in] + fc2 [proj_out,proj_inner]
    (blocks + patch_embed + connector) * 4
}

/// Elements one [`crate::decoder::BlockScratch`] occupies at `(t, d, n_heads,
/// ff)` - mirrors that type's own `new` exactly, so this estimate cannot
/// silently drift from what a real build allocates.
fn block_scratch_elems(t: u64, d: u64, n_heads: u64, ff: u64) -> u64 {
    14 * t * d + 5 * n_heads * t + 2 * n_heads * t * t + 2 * t * ff
}

/// Bytes the decoder's activation scratch occupies at `t` tokens: ONE shared
/// [`crate::decoder::BlockScratch`] under `Precision::Int8`, or `n_layers`
/// separate ones under `Precision::Fp32` (see this module's doc on why).
fn scratch_bytes(cfg: &MoondreamConfig, precision: Precision, t: u32) -> u64 {
    let elems = block_scratch_elems(t as u64, cfg.dim as u64, cfg.n_heads as u64, cfg.ff_dim as u64);
    let copies = match precision {
        Precision::Int8 => 1,
        Precision::Fp32 => cfg.n_layers as u64,
    };
    elems * copies * 4
}

/// Bytes every layer's [`crate::decoder::KvCache`] occupies at `cap` tokens -
/// always fp32, one per decoder layer, dominated by `k`/`v` at `2 * cap *
/// n_heads * head_dim`.
fn kv_cache_bytes(cfg: &MoondreamConfig, cap: u32) -> u64 {
    let hk = cfg.n_heads as u64 * cfg.head_dim as u64;
    let per_layer = 2 * cap as u64 * hk + 2 * cfg.n_heads as u64 * cap as u64 + cfg.dim as u64; // k+v, scores+probs, plus small [d]-ish vectors
    per_layer * cfg.n_layers as u64 * 4
}

/// The device bytes a [`crate::model::MoondreamModel`] build at `precision`
/// occupies at `t` (the resident's built context - `crate::caps::SEQ_LEN` in
/// production) - weights (decoder + vision + connector), the shared or
/// per-block activation scratch, and the per-layer KV cache the first real
/// request allocates.
pub fn estimate_vram_bytes(cfg: &MoondreamConfig, precision: Precision, t: u32) -> u64 {
    decoder_weight_bytes(cfg, precision) + vision_and_connector_bytes(cfg) + scratch_bytes(cfg, precision, t) + kv_cache_bytes(cfg, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// The released preview config lands in the same ballpark this resident's
    /// own hand-derived doc comment already documented (~44 GiB fp32 / ~11 GiB
    /// int8) - a looser bound on purpose, since this function additionally
    /// counts the KV cache and the vision tower/connector at their real shape
    /// rather than folding them into a rounded total.
    #[test]
    fn the_preview_config_lands_near_its_known_hand_derivation() {
        let cfg = MoondreamConfig::preview();
        let fp32 = estimate_vram_bytes(&cfg, Precision::Fp32, crate::caps::SEQ_LEN);
        let int8 = estimate_vram_bytes(&cfg, Precision::Int8, crate::caps::SEQ_LEN);
        assert!((25 * GIB..55 * GIB).contains(&fp32), "fp32 estimate {fp32} ({:.1} GiB) out of the expected 25-55 GiB band", fp32 as f64 / GIB as f64);
        assert!((5 * GIB..20 * GIB).contains(&int8), "int8 estimate {int8} ({:.1} GiB) out of the expected 5-20 GiB band", int8 as f64 / GIB as f64);
    }

    /// The whole point of `Precision::Int8` existing: it must estimate
    /// smaller than fp32, on both the expert weights AND the activation
    /// scratch (one shared `BlockScratch` instead of `n_layers` copies).
    #[test]
    fn int8_estimates_smaller_than_fp32() {
        let cfg = MoondreamConfig::preview();
        let fp32 = estimate_vram_bytes(&cfg, Precision::Fp32, crate::caps::SEQ_LEN);
        let int8 = estimate_vram_bytes(&cfg, Precision::Int8, crate::caps::SEQ_LEN);
        assert!(int8 < fp32, "int8 ({int8}) must be smaller than fp32 ({fp32})");
    }

    /// THE regression this module exists for: a bigger config (more MoE
    /// layers, over the same architecture shape) must estimate a bigger
    /// footprint - a hardcoded constant sized for one config cannot tell the
    /// two apart. `MoondreamConfig::from_json` refuses a mismatched real
    /// checkpoint today, but this function must not ALSO depend on that
    /// guard to be correct.
    #[test]
    fn a_bigger_decoder_estimates_a_bigger_footprint() {
        let small = MoondreamConfig::preview();
        let mut big = small.clone();
        big.n_layers *= 2;
        let small_bytes = estimate_vram_bytes(&small, Precision::Fp32, crate::caps::SEQ_LEN);
        let big_bytes = estimate_vram_bytes(&big, Precision::Fp32, crate::caps::SEQ_LEN);
        assert!(big_bytes > small_bytes * 3 / 2, "doubling n_layers must noticeably grow the estimate: small={small_bytes} big={big_bytes}");
    }

    /// A longer built context must estimate a bigger footprint - the KV cache
    /// and activation scratch both scale with `t`.
    #[test]
    fn a_longer_context_estimates_a_bigger_footprint() {
        let cfg = MoondreamConfig::preview();
        let short = estimate_vram_bytes(&cfg, Precision::Fp32, 256);
        let long = estimate_vram_bytes(&cfg, Precision::Fp32, 2048);
        assert!(long > short, "a longer context must estimate a bigger footprint: short={short} long={long}");
    }

    /// Matches the real per-channel int8 packing formula at a hand-computable
    /// shape: one expert at `inner=k=1` needs one packed byte per `w_h`/`w_g`
    /// element (`k=d=1` too) plus one f32 scale per output row.
    #[test]
    fn moe_layer_bytes_matches_the_known_int8_packing_formula() {
        let mut cfg = MoondreamConfig::preview();
        cfg.dim = 1;
        cfg.n_heads = 1;
        cfg.moe.inner_dim = 1;
        cfg.moe.num_experts = 1;
        // attn: (3*1*1 + 1*1)*4 = 16 bytes. router: 1*1*4 = 4 bytes.
        // one expert: w_h(1*1) + w_g(1*1) + w_down(1*1) = 3 bytes packed,
        // plus 3 scale rows (inner=1, inner=1, d=1) * 4 bytes = 12 bytes.
        let want = 16 + 4 + (3 + 12);
        assert_eq!(moe_layer_bytes(&cfg, Precision::Int8), want);
    }
}
