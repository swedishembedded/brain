// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched vision-encoder serving for clients who
// read documents at scale. If your team needs expertise in running a transformer
// resampler on tensor cores with a stated accuracy contract then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec of the resampler's serving forward on tensor cores: its seven linears per layer and the projector are TF32 library
//! products of the fp32 weights as stored, and the rows it produces agree with the pinned fp32 reference to TF32's rounding.
//! The training build is always the fp32 reference.

use data::rng::Lcg;
use deepseekocr2::config::{DeepseekOcr2VisionConfig, Qwen2EncoderConfig};
use deepseekocr2::encoder::{Resampler, PIPELINES};
use deepseekocr2::init::init_weights;
use gpu_core::provider::library::GemmPrecision;

/// Every contraction and output width a multiple of 8 so each product is one a tensor core takes.
fn config() -> DeepseekOcr2VisionConfig {
    let sam = sam1::SamViTConfig { compress_out: 32, ..sam1::SamViTConfig::tiny() };
    let encoder = Qwen2EncoderConfig { d_model: 32, n_layers: 2, n_heads: 4, n_kv_heads: 2, ffn_hidden: 64, rms_eps: 1e-6, rope_theta: 10_000.0, n_query_local: 8, n_query_global: 16 };
    DeepseekOcr2VisionConfig { sam, encoder, decoder_hidden: 24, lora: None }
}

fn resampler(precision: GemmPrecision) -> Resampler {
    let cfg = config();
    let mut r = Resampler::new_on(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &init_weights(&cfg, 7), false);
    r.set_gemm_precision(precision);
    r
}

#[test]
fn tf32_linears_track_the_fp32_reference() {
    let (fast, reference) = (resampler(GemmPrecision::Auto), resampler(GemmPrecision::Fp32Reference));
    assert!(!reference.uses_tensor_cores(), "the pinned reference runs the fp32 kernels");
    if !fast.uses_tensor_cores() {
        return brain_testutil::skip_unavailable("this device has no tensor-core GEMM");
    }
    let cfg = config();
    let mut rng = Lcg::new(data::rng::random_seed());
    let views: Vec<Vec<f32>> = (0..3).map(|_| rng.vec_scaled((cfg.encoder.n_query_local * cfg.encoder.d_model) as usize, 1.0)).collect();
    let refs: Vec<&[f32]> = views.iter().map(Vec::as_slice).collect();
    let (got, want) = (fast.encode_views(&refs, true), reference.encode_views(&refs, true));
    let flat = |v: &[Vec<f32>]| v.iter().flatten().copied().collect::<Vec<f32>>();
    let (got, want) = (flat(&got), flat(&want));
    let scale = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("resampler tf32 linears: max abs difference {err:.3e} against scale {scale:.3}");
    assert!(scale > 1e-3, "the comparison needs a signal, got scale {scale}");
    assert!(err > 0.0, "identical output means the products did not run on tensor cores");
    assert!(err <= 5e-3 * scale, "TF32 products drifted {err} from the fp32 reference (scale {scale})");
}

/// **No product of the serving forward falls back to fp32 CUDA cores.** Its attention is the unfused score/softmax/apply chain
/// (CUDA-core attention, reported apart); every linear is a tensor-core product.
#[test]
fn the_serving_forward_runs_its_products_on_tensor_cores() {
    let r = resampler(GemmPrecision::Auto);
    if !r.uses_tensor_cores() {
        return brain_testutil::skip_unavailable("this device has no tensor-core GEMM");
    }
    if !r.gpu().set_kernel_timing(true) {
        return brain_testutil::skip_unavailable("this backend cannot time kernels");
    }
    let cfg = config();
    let view = Lcg::new(data::rng::random_seed()).vec_scaled((cfg.encoder.n_query_local * cfg.encoder.d_model) as usize, 1.0);
    let _ = r.encode_views(&[&view, &view], true);
    let rows = r.gpu().kernel_times().expect("timing was switched on");
    let audit = gpu_core::profile::ArithmeticAudit::of(&rows);
    eprintln!("resampler: {}", audit.summary());
    let fallbacks = audit.fallbacks(&[]);
    assert!(fallbacks.is_empty(), "products on CUDA cores in the resampler: {fallbacks:?}\n{}", audit.summary());
    assert!(audit.tensor_core_ms > 0.0, "no tensor-core kernel ran at all: {rows:?}");
}

#[test]
fn a_training_build_never_uses_tensor_cores() {
    let cfg = config();
    let r = Resampler::new_on(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &init_weights(&cfg, 7), true);
    assert!(!r.uses_tensor_cores());
}
