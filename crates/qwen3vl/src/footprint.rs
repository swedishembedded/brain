// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A pre-flight VRAM estimate for a `Qwen3Vl` decode build, derived from the
//! checkpoint's OWN [`Qwen3VlConfig`] rather than a constant hand-derived for
//! one released size.
//!
//! `crate::caps::load_gguf_resident`/`load_hf_resident` (the direct-provider
//! `brain qwen3vl generate --weights ...` path) and
//! `crate::caps::Precision`'s callers in `crates/cli/src/resident_qwen3vl.rs`
//! (the residency-scheduled path) both need "how many device bytes will this
//! build occupy" BEFORE a `Gpu` is built - the residency path to budget a
//! device, the direct path to refuse a checkpoint that fits nowhere instead
//! of dispatching a doomed allocation. A constant sized for one released
//! checkpoint (the 4B) silently under-reports every other one - an 8B
//! checkpoint's decoder alone is roughly double the 4B's, and nothing about
//! that shows up in a number that does not read `cfg`.
//!
//! Swedish Embedded AB implements pre-flight VRAM sizing for large model
//! deployments. If your team needs a placement policy that refuses a
//! checkpoint that does not fit instead of crashing the driver, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! # What is counted, and at what tier
//!
//! - The decoder's 7 per-layer linears (q/k/v/o + gate/up/down), packed at
//!   `precision`'s tier exactly as [`qwen3::Qwen::linear_weight_bytes`]
//!   packs an already-built [`qwen3::Qwen`] - same `per_word`/scale-group
//!   arithmetic, so this estimate and the bytes a build actually allocates
//!   cannot drift apart.
//! - The token embedding / `lm_head` (fp32, always - `qwen3::Qwen::new_impl`
//!   never quantizes these two, see that constructor's own doc), one copy
//!   when tied, two when not.
//! - The plain (non-paged) fp32 KV cache [`qwen3::Qwen::new_shard_dt_decode`]
//!   allocates at `seq_len` - never quantized by this resident (`Precision`
//!   only selects the decoder linear tier).
//! - The vision tower + PatchMergers (fp32, always - a small fraction of the
//!   weights, see [`crate::model::Qwen3Vl::from_imported`]'s own doc).
//! - The DeepStack/splice scratch buffers, sized at `n_visual_capacity`.
//!
//! Not counted: biases and norm vectors (`[d]`/`[hidden]` vectors, a
//! rounding error next to the GEMM weights above) and the handful of host
//! transient buffers a real build also touches - both are what
//! `crates/cli/src/placement.rs::HEADROOM` (1 GiB, reserved on top of every
//! automatic placement) exists to absorb, the same way it already does for
//! every other model that declares a sized [`gpu_core::devices::Need`].

use gpu_core::select::Dtype;
use qwen3::QwenConfig;

use crate::caps::Precision;
use crate::config::{Qwen3VlConfig, VisionConfig};

/// Bytes the decoder's 7 per-layer linears occupy at `dt`, packed exactly as
/// [`qwen3::Qwen::linear_weight_bytes`] packs an already-built [`qwen3::Qwen`]'s
/// [`qwen3::Weight`]s: `elems.div_ceil(per_word) * 4` plus the `[n, k/32]`
/// f32 group scale the `I8`/`Q4` tiers carry.
fn decoder_linear_bytes(text: &QwenConfig, dt: Dtype) -> u64 {
    let per_word = dt.per_word() as u64;
    let quantized = matches!(dt, Dtype::I8 | Dtype::Q4);
    let pack = |n: u64, k: u64| -> u64 {
        let packed = (n * k).div_ceil(per_word) * 4;
        let scale = if quantized { model::int8::scale_len(n as usize, k as usize) as u64 * 4 } else { 0 };
        packed + scale
    };
    let d = text.d_model as u64;
    let q_n = (text.n_heads * text.head_dim) as u64;
    let kv_n = (text.n_kv_heads * text.head_dim) as u64;
    let ff = text.d_ff as u64;
    let per_layer = pack(q_n, d) // q_proj
        + pack(kv_n, d) * 2 // k_proj, v_proj
        + pack(d, q_n) // o_proj
        + pack(ff, d) * 2 // gate_proj, up_proj
        + pack(d, ff); // down_proj
    per_layer * text.n_layers as u64
}

/// Bytes the token embedding / `lm_head` occupy - always fp32, one copy when
/// tied, two when not (see this module's doc).
fn embed_head_bytes(text: &QwenConfig) -> u64 {
    let one = text.vocab as u64 * text.d_model as u64 * 4;
    if text.tie_embeddings {
        one
    } else {
        one * 2
    }
}

/// Bytes the plain (non-paged) fp32 KV cache occupies at `seq_len`:
/// `n_kv_heads * head_dim * 2 (K,V) * 4 bytes * n_layers * seq_len`.
fn kv_cache_bytes(text: &QwenConfig, seq_len: u32) -> u64 {
    text.n_kv_heads as u64 * text.head_dim as u64 * 2 * 4 * text.n_layers as u64 * seq_len as u64
}

/// Bytes the ViT tower's blocks + patch embed + learned position table
/// occupy - always fp32. Per-block: fused `qkv [3·hidden,hidden]` + `proj
/// [hidden,hidden]` (`4·hidden²`) plus `fc1 [intermediate,hidden]` + `fc2
/// [hidden,intermediate]` (`2·hidden·intermediate`), matching
/// `crate::encoder::BLOCK_LEAVES`'s own shapes.
fn vision_tower_bytes(v: &VisionConfig) -> u64 {
    let hidden = v.hidden as u64;
    let inter = v.intermediate as u64;
    let per_block = 4 * hidden * hidden + 2 * hidden * inter;
    let blocks = per_block * v.depth as u64;
    let patch_embed = v.patch_vec_dim() as u64 * hidden;
    let pos_table = v.num_position_embeddings as u64 * hidden;
    (blocks + patch_embed + pos_table) * 4
}

/// Bytes every PatchMerger occupies - always fp32: `merged = hidden ·
/// spatial_merge_size²`, `fc1 [merged,merged]` + `fc2 [decoder_d_model,
/// merged]`, one main merger plus one per DeepStack tap - matching
/// `crate::encoder::PatchMerger::new`'s own `need` array.
fn merger_bytes(v: &VisionConfig, decoder_d_model: u32) -> u64 {
    let merged = v.hidden as u64 * v.spatial_merge_size as u64 * v.spatial_merge_size as u64;
    let per_merger = merged * merged + decoder_d_model as u64 * merged;
    let n_mergers = 1 + v.deepstack_indexes.len() as u64;
    per_merger * n_mergers * 4
}

/// Bytes the DeepStack/splice scratch buffers occupy: one splice buffer plus
/// one DeepStack-tap buffer per level, each up to `n_visual_capacity` rows at
/// the decoder's own width - see [`crate::model::Qwen3Vl::assemble`]'s
/// `enable_deepstack`/`enable_mm_splice` call.
fn visual_scratch_bytes(v: &VisionConfig, decoder_d_model: u32, n_visual_capacity: u32) -> u64 {
    (1 + v.deepstack_indexes.len() as u64) * n_visual_capacity as u64 * decoder_d_model as u64 * 4
}

/// The device bytes a [`crate::model::Qwen3Vl`] decode build (vision tower +
/// mergers + decoder + KV cache + visual scratch, all on one card - see
/// [`crate::model::Qwen3Vl::assemble`]'s doc on why the vision tower shares
/// the decoder's device) will occupy, derived from `cfg` itself so it scales
/// with whatever checkpoint is actually being loaded.
pub fn estimate_vram_bytes(cfg: &Qwen3VlConfig, precision: Precision, seq_len: u32, n_visual_capacity: u32) -> u64 {
    decoder_linear_bytes(&cfg.text, precision.dtype())
        + embed_head_bytes(&cfg.text)
        + kv_cache_bytes(&cfg.text, seq_len)
        + vision_tower_bytes(&cfg.vision)
        + merger_bytes(&cfg.vision, cfg.text.d_model)
        + visual_scratch_bytes(&cfg.vision, cfg.text.d_model, n_visual_capacity)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// The released 4B config at its own served defaults (1024² max_pixels ->
    /// 256 visual tokens/image · 8 images capacity, ctx 24576) lands in the
    /// same ballpark this resident's own hand-derived doc comment already
    /// documented (~24 GiB fp32 / ~13 GiB int8) - a looser bound than the
    /// hand derivation on purpose, since this function additionally counts
    /// the PatchMerger `fc1`/`fc2` at their REAL shape rather than the doc's
    /// rounded "~17M params each".
    #[test]
    fn a_4b_checkpoint_lands_near_its_known_hand_derivation() {
        let cfg = Qwen3VlConfig::qwen3_vl_4b();
        let n_visual_capacity = 256 * 8;
        let fp32 = estimate_vram_bytes(&cfg, Precision::F32, 24576, n_visual_capacity);
        assert!((20 * GIB..30 * GIB).contains(&fp32), "fp32 estimate {fp32} ({:.1} GiB) out of the expected 20-30 GiB band", fp32 as f64 / GIB as f64);
    }

    /// The whole point of int8 existing: it must estimate smaller, and by
    /// roughly the decoder's own weight-tier ratio (~4x on the linears,
    /// diluted by the KV cache and vision tower which stay fp32 either way).
    #[test]
    fn int8_estimates_smaller_than_fp32() {
        let cfg = Qwen3VlConfig::qwen3_vl_4b();
        let fp32 = estimate_vram_bytes(&cfg, Precision::F32, 24576, 256);
        let int8 = estimate_vram_bytes(&cfg, Precision::I8, 24576, 256);
        assert!(int8 < fp32, "int8 ({int8}) must be smaller than fp32 ({fp32})");
    }

    /// THE regression this module exists for: a bigger checkpoint (double
    /// the layers - roughly an 8B-class decoder over the same 4B vision
    /// tower) must estimate a bigger footprint. A hardcoded constant sized
    /// for one released checkpoint cannot tell the two apart, which is
    /// exactly the gap that let a real 8B GGUF land on a card sized by the
    /// 4B's own figure and OOM the driver instead of being refused.
    #[test]
    fn a_bigger_decoder_estimates_a_bigger_footprint() {
        let small = Qwen3VlConfig::qwen3_vl_4b();
        let mut big = small.clone();
        big.text.n_layers *= 2;
        let small_bytes = estimate_vram_bytes(&small, Precision::F32, 24576, 256);
        let big_bytes = estimate_vram_bytes(&big, Precision::F32, 24576, 256);
        assert!(big_bytes > small_bytes * 3 / 2, "doubling n_layers must noticeably grow the estimate: small={small_bytes} big={big_bytes}");
    }

    /// The KV cache term alone must scale with `seq_len` - a caller sizing a
    /// smaller context should not pay (or budget) for a bigger one.
    #[test]
    fn a_longer_context_estimates_a_bigger_footprint() {
        let cfg = Qwen3VlConfig::qwen3_vl_4b();
        let short = estimate_vram_bytes(&cfg, Precision::F32, 4096, 256);
        let long = estimate_vram_bytes(&cfg, Precision::F32, 32768, 256);
        assert!(long > short, "a longer context must estimate a bigger footprint: short={short} long={long}");
    }

    /// Matches [`qwen3::Qwen::linear_weight_bytes`]'s own packing formula at
    /// a hand-computable shape, so the two cannot silently diverge: `n=k=32`
    /// (one scale group exactly) at `Dtype::I8` packs to `32*32/4` (4 int8
    /// per word) `+ 1*1*4` (one `[1,1]` scale group) bytes.
    #[test]
    fn decoder_linear_bytes_matches_the_known_int8_packing_formula() {
        let mut cfg = QwenConfig {
            vocab: 1,
            block_size: 1,
            n_layers: 1,
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
        // Every one of the 7 linears is [32,32] at this shape (q/k/v/o all
        // n=k=32 since n_heads=n_kv_heads=1, head_dim=32=d_model; d_ff=32=d_model).
        // `per_word` for I8 is 4 (32 bits / 8 bits), so 1024 elements pack
        // into 256 words, at 4 bytes/word = 1024 bytes, plus one `[1,1]`
        // scale group at 4 bytes.
        let one_linear_i8 = (32u64 * 32 / 4) * 4 + model::int8::scale_len(32, 32) as u64 * 4;
        cfg.n_layers = 1;
        assert_eq!(decoder_linear_bytes(&cfg, Dtype::I8), one_linear_i8 * 7);
    }
}
