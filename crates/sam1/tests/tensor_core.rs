// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements document-vision serving for clients who read pages at scale. If your team
// needs expertise in running ViT encoders on tensor cores with a stated accuracy contract then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec of the inference build's linears: on a device with tensor cores the tower's QKV, projection and MLP
//! products run as TF32 library GEMMs (fp32 weights read as stored, fp32 accumulate), and the output agrees
//! with the pinned fp32 reference to TF32's rounding. A device without them runs the fp32 kernels and says so.
//! The training build is always the fp32 reference: the gradient checks are stated against it.

use gpu_core::provider::library::GemmPrecision;
use sam1::init::init_dense;
use sam1::{SamEncoder, SamViTConfig, PIPELINES};

/// Tiny, but with every contraction and output width a multiple of 8 so each product is one a tensor core takes.
fn config() -> SamViTConfig {
    // A 12x8 grid: the global block's 96 keys are a multiple of 4, the alignment a tensor-core product needs of its rows.
    SamViTConfig { ffn_hidden: 32, ..SamViTConfig::tiny().at_grid(12, 8) }
}

fn encoder(precision: GemmPrecision, train: bool) -> SamEncoder {
    let cfg = config();
    let mut enc = SamEncoder::new_on(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &init_dense(&cfg, 5), 0, train);
    enc.set_gemm_precision(precision);
    enc
}

fn output(enc: &SamEncoder) -> Vec<f32> {
    let cfg = config();
    let mut rng = data::rng::Lcg::new(9);
    enc.write_image(&rng.vec_scaled((3 * cfg.image_h() * cfg.image_w()) as usize, 1.0));
    enc.run();
    enc.gpu.read(enc.output(), enc.out_len())
}

#[test]
fn tf32_linears_track_the_fp32_reference() {
    let (fast, reference) = (encoder(GemmPrecision::Auto, false), encoder(GemmPrecision::Fp32Reference, false));
    assert!(!reference.uses_tensor_cores(), "the pinned reference runs the fp32 kernels");
    if !fast.uses_tensor_cores() {
        return brain_testutil::skip_unavailable("this device has no tensor-core GEMM");
    }
    let (got, want) = (output(&fast), output(&reference));
    assert_eq!(fast.tensor_core_attention_blocks() as usize, config().global_attn_layers.len(), "every global block attends on tensor cores");
    assert_eq!(reference.tensor_core_attention_blocks(), 0);
    let scale = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("tf32 linears: max abs difference {err:.3e} against scale {scale:.3}");
    assert!(scale > 1e-3, "the comparison needs a signal, got scale {scale}");
    assert!(err > 0.0, "identical output means the products did not run on tensor cores");
    assert!(err <= 5e-3 * scale, "TF32 products drifted {err} from the fp32 reference (scale {scale})");
}

#[test]
fn the_training_build_never_uses_tensor_cores() {
    assert!(!encoder(GemmPrecision::Auto, true).uses_tensor_cores());
}

/// The convolutions (patch embedding, the neck's 1x1 and 3x3, the compressor's two strided 3x3) are direct CUDA-core kernels:
/// about 4% of the tower's device time at the served size. They are the declared exception; every other product is a
/// tensor-core one.
const CONVOLUTIONS: &[&str] = &["conv2d", "conv2d_fwd_f32", "conv2d_bias_fwd_f32", "conv2d_fwd3x3_f32"];

/// **No product of the inference tower falls back to fp32 CUDA cores** except the declared convolutions: the QKV, projection and
/// MLP linears and the global blocks' attention products are tensor-core kernels on the device's own kernel timer.
#[test]
fn the_tower_runs_its_products_on_tensor_cores() {
    let enc = encoder(GemmPrecision::Auto, false);
    if !enc.uses_tensor_cores() {
        return brain_testutil::skip_unavailable("this device has no tensor-core GEMM");
    }
    if !enc.gpu.set_kernel_timing(true) {
        return brain_testutil::skip_unavailable("this backend cannot time kernels");
    }
    output(&enc);
    let rows = enc.gpu.kernel_times().expect("timing was switched on");
    let audit = gpu_core::profile::ArithmeticAudit::of(&rows);
    eprintln!("sam1 tower: {}", audit.summary());
    let fallbacks = audit.fallbacks(CONVOLUTIONS);
    assert!(fallbacks.is_empty(), "products on CUDA cores in the inference tower: {fallbacks:?}\n{}", audit.summary());
    assert!(audit.tensor_core_ms > 0.0, "no tensor-core kernel ran at all: {rows:?}");
}

/// An inference tower builds its relative-position tables once (they depend on weights alone), so a weight written after a
/// pass must still reach the next one: the tower agrees with a fresh build holding the same weights.
#[test]
fn a_weight_written_between_passes_reaches_the_cached_tables() {
    let name = "vision.sam.blocks.0.attn.rel_pos_h";
    let reused = encoder(GemmPrecision::Fp32Reference, false);
    let before = output(&reused);
    let moved: Vec<f32> = reused.read_weight(name).iter().map(|w| w * 3.0 + 0.5).collect();
    reused.write_weight(name, &moved);
    let got = output(&reused);

    let fresh = encoder(GemmPrecision::Fp32Reference, false);
    fresh.write_weight(name, &moved);
    let want = output(&fresh);
    assert!(got.iter().zip(&before).any(|(a, b)| a != b), "the written table must change the output");
    assert_eq!(got, want, "a stale cached table survived a weight write");
}
