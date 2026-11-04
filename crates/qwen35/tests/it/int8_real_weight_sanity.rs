// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-weight-distribution sanity check for the int8 (DP4A) quantization
//! path this milestone adds (`model::ops::Weight::upload(..., Dtype::I8)`,
//! driving `Qwen35::new_i8`/`Qwen35::new_on_i8`): for every quantizable leaf
//! of REAL `Qwen/Qwen3.8-27B-FP8` weights (layer 0, Gated DeltaNet, and
//! layer 3, GQA - both streamed one layer at a time via
//! `crate::import::import_layer`, already proven by `real_weight_streaming.
//! rs`), build an `Ops`, an `F32` and an `I8` `Weight` from the SAME raw
//! `&[f32]` slice, run both through `Ops::act`+`Ops::matmul` against a small
//! fixed-seed random activation batch, and check the two outputs track each
//! other (cosine > 0.99).
//!
//! **This is deliberately NOT a full-model forward parity test** - building
//! an `Ops`/`Qwen35` instance at the real 27B config is far too memory-
//! hungry for this box (`real_weight_streaming.rs`'s own doc explains why a
//! full model can't be constructed here: ~108 GB dequantized). This test
//! instead checks the quantization path itself against a REAL weight-value
//! distribution (fp8-block-dequantized, not synthetic fresh-init noise, the
//! way `model_i8_smoke.rs`'s tiny-config parity test does) - real-model
//! end-to-end int8 parity is out of scope here and left to a future
//! milestone once a real forward at this scale is buildable at all on this
//! box.
//!
//! Self-skips loudly (never silently) without `BRAIN_QWEN35_DIR` or the
//! matching shard file - same pattern as `real_weight_streaming.rs`. Run
//! with:
//!
//! ```text
//! BRAIN_QWEN35_DIR=/path/to/Qwen3.8-27B-FP8 \
//!     cargo test -p brain-qwen35 --test int8_real_weight_sanity -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use checkpoint::mmap::MmapSafetensors;
use data::rng::Lcg;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::{Ops, Weight};
use qwen35::config::{LayerType, Qwen35Config};
use qwen35::import::import_layer;

fn checkpoint_dir() -> Option<PathBuf> {
    std::env::var_os("BRAIN_QWEN35_DIR").map(PathBuf::from)
}

/// The quantizable leaves for layer `l` of `ty`, as `(canonical name, n, k)` -
/// mirrors `qwen35::model::Qwen35::new_impl_on`'s own `upload` loop exactly
/// (same names, same shapes derived from `cfg`), so this test's coverage
/// tracks the real constructor's, not a hand-duplicated shape list.
fn leaves_for(cfg: &Qwen35Config, l: usize, ty: LayerType) -> Vec<(String, usize, usize)> {
    let d = cfg.d_model as usize;
    let ff = cfg.intermediate_size as usize;
    let mut v = Vec::new();
    match ty {
        LayerType::Linear => {
            let conv_dim = cfg.linear_conv_dim() as usize;
            let value_dim = cfg.linear_value_dim() as usize;
            let nvh = cfg.linear_num_value_heads as usize;
            v.push((format!("blocks.{l}.linear_attn.in_proj_qkv.weight"), conv_dim, d));
            v.push((format!("blocks.{l}.linear_attn.in_proj_z.weight"), value_dim, d));
            v.push((format!("blocks.{l}.linear_attn.in_proj_b.weight"), nvh, d));
            v.push((format!("blocks.{l}.linear_attn.in_proj_a.weight"), nvh, d));
            v.push((format!("blocks.{l}.linear_attn.out_proj.weight"), d, value_dim));
        }
        LayerType::Full => {
            let hqp = cfg.q_proj_dim() as usize;
            let hkv = cfg.kv_dim() as usize;
            let hq = cfg.q_dim() as usize;
            v.push((format!("blocks.{l}.self_attn.q_proj.weight"), hqp, d));
            v.push((format!("blocks.{l}.self_attn.k_proj.weight"), hkv, d));
            v.push((format!("blocks.{l}.self_attn.v_proj.weight"), hkv, d));
            v.push((format!("blocks.{l}.self_attn.o_proj.weight"), d, hq));
        }
    }
    v.push((format!("blocks.{l}.mlp.gate.weight"), ff, d));
    v.push((format!("blocks.{l}.mlp.up.weight"), ff, d));
    v.push((format!("blocks.{l}.mlp.down.weight"), d, ff));
    v
}

/// Build `F32`/`I8` weights from the same raw slice, dispatch both against
/// the same quantized-once activation (`rows` random rows of width `k`,
/// fixed-seed `Lcg`), and return the cosine between the two outputs.
fn check_leaf(gpu: &Gpu, ops: &Ops, raw: &[f32], n: usize, k: usize, rows: usize, seed: u64) -> f64 {
    assert_eq!(raw.len(), n * k, "raw weight len {} != n*k ({n}*{k})", raw.len());
    let w_f32 = Weight::upload(ops, raw, n, k, Dtype::F32);
    let w_i8 = Weight::upload(ops, raw, n, k, Dtype::I8);

    let mut rng = Lcg::new(seed);
    let act_raw = rng.vec_scaled(rows * k, 0.5);
    let x = gpu.storage_init("qwen35_int8_sanity.act", &act_raw);

    let mut s = Vec::new();
    let act = ops.act(&mut s, &x, 0, rows as u32, k as u32);
    let y_f32 = gpu.storage((rows * n) as u64);
    let y_i8 = gpu.storage((rows * n) as u64);
    ops.matmul(&mut s, &w_f32, &act, &y_f32, 0);
    ops.matmul(&mut s, &w_i8, &act, &y_i8, 0);
    gpu.submit(&[], &s);

    let got_f32 = gpu.read(&y_f32, rows * n);
    let got_i8 = gpu.read(&y_i8, rows * n);
    assert!(got_f32.iter().all(|v| v.is_finite()), "f32 tier produced a non-finite output");
    assert!(got_i8.iter().all(|v| v.is_finite()), "i8 tier produced a non-finite output");
    let (cos, _max_abs) = brain_testutil::parity::compare(&got_i8, &got_f32);
    cos
}

fn run_layer_sanity(l: usize) {
    let Some(dir) = checkpoint_dir() else {
        brain_testutil::skip("BRAIN_QWEN35_DIR unset (set it to a downloaded Qwen/Qwen3.8-27B-FP8 dir to run this - see this file's own doc)");
        return;
    };
    let shard = dir.join(format!("layers-{l}.safetensors"));
    if !shard.exists() {
        brain_testutil::skip_unavailable(&format!("{} not present under BRAIN_QWEN35_DIR", shard.display()));
        return;
    }

    brain_testutil::mem(&format!("layer {l}: before import"));
    let cfg = Qwen35Config::qwen38_27b();
    let ty = cfg.layer_types()[l];
    let leaves = leaves_for(&cfg, l, ty);

    let reader = MmapSafetensors::open(&shard).unwrap_or_else(|e| panic!("open {}: {e}", shard.display()));
    let w = import_layer(&reader, &cfg, l, 128).unwrap_or_else(|e| panic!("import_layer({l}): {e}"));
    drop(reader);
    brain_testutil::mem(&format!("layer {l}: after import_layer ({} tensors)", w.len()));

    let gpu = Gpu::new(qwen35::model::pipelines());
    let ops = Ops::new(gpu.share()).unwrap_or_else(|e| panic!("Ops::new: {e}"));

    let rows = 8usize;
    for (i, (name, n, k)) in leaves.iter().enumerate() {
        let raw = w.get(name).unwrap_or_else(|| panic!("layer {l}: missing weight {name}"));
        let cos = check_leaf(&gpu, &ops, raw, *n, *k, rows, 1000 + l as u64 * 100 + i as u64);
        eprintln!("layer {l} {name} (n={n}, k={k}): int8 vs fp32 cosine={cos:.9}");
        assert!(cos > 0.99, "layer {l} {name}: int8 vs fp32 cosine={cos:.6} too low (want > 0.99)");
    }

    // Drop this layer's fp32 tensor map before the next layer's - keeps peak
    // RSS to one layer's worth, same memory discipline as
    // `real_weight_streaming.rs`'s own per-layer streaming.
    drop(w);
    brain_testutil::mem(&format!("layer {l}: after leaf checks"));
}

#[test]
#[ignore]
fn layer_0_gated_delta_net_int8_tracks_fp32_on_real_weights() {
    run_layer_sanity(0);
}

#[test]
#[ignore]
fn layer_3_gated_gqa_int8_tracks_fp32_on_real_weights() {
    run_layer_sanity(3);
}
