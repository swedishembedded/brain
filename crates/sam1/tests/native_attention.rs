// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements document-vision serving for clients who read pages at scale. If your team
// needs expertise in running ViT encoders on tensor cores with a stated accuracy contract then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec of the inference tower's attention at SAM's real head width (64) on a CUDA device: the windowed and the
//! global blocks attend through the native fp16 tensor-core kernel, and the tower agrees with the same tower
//! built with native kernels withheld (the generated fp32 tier) to the fp16 operand rounding.
//!
//! One test in its own binary: native kernels are withheld process-wide while the reference is built, which
//! must not race another test's handle.

use sam1::init::init_dense;
use sam1::{SamEncoder, SamViTConfig, PIPELINES};

/// Two heads of 64 channels over a 12x8 grid: a windowed block with ragged windows and a global block of 96 keys.
fn config() -> SamViTConfig {
    SamViTConfig { d_model: 128, ffn_hidden: 256, ..SamViTConfig::tiny().at_grid(12, 8) }
}

fn build(native: bool) -> SamEncoder {
    let cfg = config();
    gpu_core::set_native_kernels(native);
    let enc = SamEncoder::new_on(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &init_dense(&cfg, 5), 0, false);
    gpu_core::set_native_kernels(true);
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
fn the_native_attention_tracks_the_generated_tier() {
    let (fast, reference) = (build(true), build(false));
    if fast.gpu.kind() != "cuda" || fast.gpu.caps().arch.compute_capability.is_none() {
        return brain_testutil::skip_unavailable("native flash attention needs a CUDA device");
    }
    let (got, want) = (output(&fast), output(&reference));
    let scale = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("native attention: max abs difference {err:.3e} against scale {scale:.3}");
    assert!(scale > 1e-3, "the comparison needs a signal, got scale {scale}");
    assert!(err > 0.0, "identical output means the native kernel did not run");
    assert!(err <= 1e-2 * scale, "fp16 attention drifted {err} from the generated tier (scale {scale})");
}
