// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! INT8 STORAGE round-trip parity for the flow-matching DiT, mirroring
//! `crates/ltxv/tests/int8_storage.rs`'s own "the test that matters": one
//! tiny forward pass run twice - once at plain f32, once with every eligible
//! weight round-tripped through `minimaxmusic3::dit_int8` storage first -
//! with the two outputs compared by cosine. This bounds int8's real accuracy
//! cost on an actual model forward, not just per-tensor norm preservation.
//!
//! The int8 test uses [`tiny_quantizable`] rather than [`DitConfig::tiny`]:
//! `tiny`'s inner dim (8) and `ff_inner_dim` (16) are each smaller than ONE
//! `model::int8::GROUP`, so nothing in it is quantizable at all, and `tiny`
//! itself is pinned by `dit_parity.rs` against a diffusers golden and must
//! not move. The f32 round-trip test above keeps using `tiny` unchanged.
//!
//! No fixture dependency (unlike `dit_parity.rs`): [`dit_train::
//! random_weights`] needs no golden, so this test always runs.

use checkpoint::safetensors::StTensor;
use data::rng::Lcg;
use gpu_core::Gpu;
use minimaxmusic3::config::DitConfig;
use minimaxmusic3::dit::{self, DitWeights};
use minimaxmusic3::dit_int8::{dequantize_tensors, quantize_tensors};

/// `DitWeights` -> the raw `Vec<StTensor>` shape `dit::from_tensors`
/// consumes - test-fixture plumbing only, every name/shape pair
/// cross-checked against `dit::from_tensors`'s own `get()` calls.
fn to_tensors(w: &DitWeights, cfg: &DitConfig) -> Vec<StTensor> {
    let inner = cfg.inner_dim() as usize;
    let concat = cfg.concat_channels() as usize;
    let cin = cfg.in_channels as usize;
    let ff_inner = cfg.ff_inner_dim as usize;
    let fed = cfg.fourier_embedding_dim as usize;

    let mut out = Vec::new();
    let mut push = |name: String, shape: Vec<usize>, data: Vec<f32>| {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}: shape/data length mismatch");
        out.push(StTensor { name, shape, data });
    };

    push("time_proj.weight".to_string(), vec![fed / 2, 1], w.time_proj_weight.clone());
    push("time_embed.linear_1.weight".to_string(), vec![inner, fed], w.time_embed_l1_w.clone());
    push("time_embed.linear_1.bias".to_string(), vec![inner], w.time_embed_l1_b.clone());
    push("time_embed.linear_2.weight".to_string(), vec![inner, inner], w.time_embed_l2_w.clone());
    push("time_embed.linear_2.bias".to_string(), vec![inner], w.time_embed_l2_b.clone());
    push("preprocess_conv.weight".to_string(), vec![concat, concat, 1], w.preprocess_conv_w.clone());
    push("proj_in.weight".to_string(), vec![inner, concat], w.proj_in_w.clone());
    push("proj_out.weight".to_string(), vec![cin, inner], w.proj_out_w.clone());
    push("postprocess_conv.weight".to_string(), vec![cin, cin, 1], w.postprocess_conv_w.clone());

    for (i, b) in w.blocks.iter().enumerate() {
        let p = format!("transformer_blocks.{i}");
        push(format!("{p}.norm1.weight"), vec![inner], b.norm1_w.clone());
        push(format!("{p}.norm1.bias"), vec![inner], b.norm1_b.clone());
        push(format!("{p}.attn.to_q.weight"), vec![inner, inner], b.attn.wq.clone());
        push(format!("{p}.attn.to_k.weight"), vec![inner, inner], b.attn.wk.clone());
        push(format!("{p}.attn.to_v.weight"), vec![inner, inner], b.attn.wv.clone());
        push(format!("{p}.attn.to_out.0.weight"), vec![inner, inner], b.attn.wo.clone());
        push(format!("{p}.norm2.weight"), vec![inner], b.norm2_w.clone());
        push(format!("{p}.norm2.bias"), vec![inner], b.norm2_b.clone());
        push(format!("{p}.ff_in.weight"), vec![2 * ff_inner, inner], b.ff_in_w.clone());
        push(format!("{p}.ff_in.bias"), vec![2 * ff_inner], b.ff_in_b.clone());
        push(format!("{p}.ff_out.weight"), vec![inner, ff_inner], b.ff_out_w.clone());
        push(format!("{p}.ff_out.bias"), vec![inner], b.ff_out_b.clone());
    }
    out
}

/// Same fixture shape `dit_train.rs`'s own `#[cfg(test)]` tests use:
/// `length=3`, deterministic `latents`/`condition`/`timestep` via
/// `data::rng::Lcg`.
fn fixture(cfg: &DitConfig, seed: u64) -> (DitWeights, Vec<f32>, Vec<f32>, f32, usize) {
    let w = minimaxmusic3::dit_train::random_weights(cfg, seed);
    let length = 3usize;
    let mut r = Lcg::new(seed + 1);
    let latents = r.vec_scaled(cfg.in_channels as usize * length, 0.3);
    let condition = r.vec_scaled(length * cfg.condition_dim as usize, 0.3);
    let timestep = 0.4f32;
    (w, latents, condition, timestep, length)
}

#[test]
fn to_tensors_round_trips_through_from_tensors() {
    // Sanity check demanded by the plan: to_tensors -> dit::from_tensors
    // must reconstruct the SAME DitWeights, confirmed by an exact
    // (bit-for-bit) forward match - not just "close enough" - before int8
    // is anywhere in the picture.
    let cfg = DitConfig::tiny();
    let (w, latents, condition, timestep, length) = fixture(&cfg, 1);

    let gpu = Gpu::new_cpu(dit::PIPELINES);
    let want = dit::forward(&gpu, &cfg, &w, &latents, &condition, timestep, length);

    let w2 = dit::from_tensors(to_tensors(&w, &cfg), &cfg, "test").unwrap();
    let got = dit::forward(&gpu, &cfg, &w2, &latents, &condition, timestep, length);

    assert_eq!(got, want, "to_tensors/from_tensors round trip must reproduce the exact same forward output");
}

/// [`DitConfig::tiny`]'s shape with the two CONTRACTION widths raised to
/// whole `model::int8::GROUP`s: `inner_dim` (attention q/k/v/out and `ff.0`
/// all contract over it) and `ff_inner_dim` (`ff.2`'s own K). Everything
/// else - layer count, head count, the channel/condition widths, the rotary
/// split - is `tiny`'s, so this stays a toy config and still exercises two
/// blocks' worth of attention and FFN.
fn tiny_quantizable() -> DitConfig {
    DitConfig { attention_head_dim: 16, ff_inner_dim: 64, ..DitConfig::tiny() }
}

#[test]
fn dit_forward_stays_close_after_int8_storage_round_trip() {
    let cfg = tiny_quantizable();
    let (w, latents, condition, timestep, length) = fixture(&cfg, 2);
    let tensors = to_tensors(&w, &cfg);

    let q = quantize_tensors(&tensors);
    println!("int8 storage: {} of {} tensors int8-eligible for tiny_quantizable()", q.int8.len(), tensors.len());
    assert!(!q.int8.is_empty(), "tiny config must have at least one int8-eligible tensor");
    // Exactly the DiT's 6 per-block linears, per block.
    assert_eq!(q.int8.len(), cfg.num_layers as usize * 6, "unexpected int8-eligible tensor count");

    let gpu = Gpu::new_cpu(dit::PIPELINES);
    let w_f32 = dit::from_tensors(tensors, &cfg, "f32").unwrap();
    let out_f32 = dit::forward(&gpu, &cfg, &w_f32, &latents, &condition, timestep, length);

    let w_i8 = dit::from_tensors(dequantize_tensors(&q), &cfg, "int8").unwrap();
    let out_i8 = dit::forward(&gpu, &cfg, &w_i8, &latents, &condition, timestep, length);

    assert_eq!(out_f32.len(), out_i8.len());
    let (cos, max_abs) = brain_testutil::parity::compare(&out_f32, &out_i8);
    println!("int8 storage forward parity: cosine={cos:.9} max_abs={max_abs:.6}");
    // Measured on this fixture (tiny_quantizable, length=3): cosine lands at
    // 0.999999+ - the boundary/conditioning tables (proj_in/proj_out/
    // time_embed) held at full f32 keep int8 noise from ever reaching the
    // output through more than 2 blocks' worth of attention/FFN
    // projections. Matching this crate's own `ltxv::int8` precedent's
    // documented approach: measure first, then pick a threshold with a
    // sane margin below the measured number.
    assert!(cos >= 0.9999, "int8-storage-round-tripped forward diverged too far from f32: cosine {cos:.9}");
}
