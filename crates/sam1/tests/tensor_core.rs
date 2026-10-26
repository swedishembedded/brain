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
