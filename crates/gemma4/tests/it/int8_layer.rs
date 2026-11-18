// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The int8 tier on whatever device the suite runs on.
//!
//! Swedish Embedded AB implements quantized inference paths that run on every device for its clients. If your team
//! needs expertise in keeping a low-precision tier correct where the hardware has no integer dot product then you
//! can procure our services by sending an email to info@swedishembedded.com.
//!
//! The projections are int8 products the shared façade selects for the device (`model::ops::Ops::matmul`): the
//! packed-dot GEMM where there is one, the float-read tile decode where there is not (the CPU JIT). Either way a
//! layer built at [`Precision::Int8`] must compute what its fp32 twin does to within the weights' rounding, on both
//! attention types. The real-checkpoint gate (`int8_compute_real`) needs the 12B GGUF; this one needs nothing.

use gemma4::block::{Gemma4Layer, Precision, Tensors, KERNELS};
use gemma4::config::Gemma4Config;
use gemma4::rope::{full_table, sliding_table, upload_rope};
use gpu_core::Gpu;

/// The tiny config widened so every projection's reduction width is a whole number of 32-element int8 groups.
fn cfg() -> Gemma4Config {
    Gemma4Config { hidden_size: 64, intermediate_size: 128, head_dim: 16, global_head_dim: 32, ..Gemma4Config::tiny() }
}

/// Deterministic, non-degenerate weights for layer `l`: norm gains near one, the layer scalar one, projections
/// small and varied per tensor.
fn layer_weights(cfg: &Gemma4Config, l: u32) -> Tensors {
    gemma4::model::layer_tensor_names(cfg, l)
        .into_iter()
        .enumerate()
        .map(|(ti, (name, shape))| {
            let n: usize = shape.iter().product();
            let wave = |i: usize| ((i * 7919 + ti * 104_729) % 1009) as f32 / 1009.0 - 0.5;
            let data: Vec<f32> = if name.ends_with("layer_scalar") {
                vec![1.0]
            } else if name.contains("norm") {
                (0..n).map(|i| 1.0 + 0.2 * wave(i)).collect()
            } else {
                (0..n).map(|i| 0.3 * wave(i)).collect()
            };
            (name, (shape, data))
        })
        .collect()
}

/// `(cosine, ‖b - a‖ / ‖a‖)`: cosine cannot see a scale error, the relative L2 can.
fn cosine_and_rel_l2(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut diff) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        diff += (x - y) * (x - y);
    }
    (dot / (na.sqrt() * nb.sqrt()), (diff / na).sqrt())
}

#[test]
fn an_int8_layer_matches_its_fp32_twin_on_both_attention_types() {
    let cfg = cfg();
    let gpu = Gpu::open(None, &KERNELS);
    let t = 8usize;
    let x: Vec<f32> = (0..t * cfg.hidden_size as usize).map(|i| ((i % 97) as f32 / 97.0 - 0.5) * 1.3).collect();
    let sliding = upload_rope(&gpu, &sliding_table(cfg.head_dim, cfg.rope_theta_sliding, t));
    let full = upload_rope(&gpu, &full_table(cfg.global_head_dim, cfg.rope_theta_full, cfg.partial_rotary_factor, t));
    let mut types = Vec::new();
    for l in [0u32, 5] {
        let lt = cfg.layer_type(l);
        let rope = match lt {
            gemma4::LayerType::Sliding => &sliding,
            gemma4::LayerType::Full => &full,
        };
        let w = layer_weights(&cfg, l);
        let (out_f32, attn_f32) = Gemma4Layer::on(&gpu, &cfg, &w, l, Precision::Fp32).forward(&x, rope, t as u32);
        let (out_i8, attn_i8) = Gemma4Layer::on(&gpu, &cfg, &w, l, Precision::Int8).forward(&x, rope, t as u32);
        let (cos_out, rel_out) = cosine_and_rel_l2(&out_f32, &out_i8);
        let (cos_attn, rel_attn) = cosine_and_rel_l2(&attn_f32, &attn_i8);
        eprintln!("layer {l} ({lt:?}) int8 vs fp32 on {}: out cosine {cos_out:.6} rel_l2 {rel_out:.4}, attn cosine {cos_attn:.6} rel_l2 {rel_attn:.4}", gpu.kind());
        // int8 is lossy: these floors catch a broken port (a wrong scale, a mis-packed lane, a transposed operand),
        // and the attention tap's magnitude is the one a scale error cannot hide from behind a normalization.
        assert!(cos_out >= 0.99 && rel_out <= 0.15, "layer {l} ({lt:?}) output diverged: cosine {cos_out:.6} rel_l2 {rel_out:.4}");
        assert!(cos_attn >= 0.99 && rel_attn <= 0.15, "layer {l} ({lt:?}) self-attention diverged: cosine {cos_attn:.6} rel_l2 {rel_attn:.4}");
        types.push(lt);
    }
    assert!(types.contains(&gemma4::LayerType::Sliding) && types.contains(&gemma4::LayerType::Full), "both attention types are covered: {types:?}");
}

/// The int8 projections build a `model::ops::Ops` on this crate's `Gpu`, which needs the whole façade set.
#[test]
fn the_kernel_list_registers_the_whole_ops_facade_set() {
    model::ops::assert_kernel_list_complete(&KERNELS);
}
