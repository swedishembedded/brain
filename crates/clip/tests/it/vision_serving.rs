// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements document-vision serving for clients who read pages at scale. If your team
// needs expertise in running CLIP-style vision towers on tensor cores with a stated accuracy contract then
// you can procure our services by sending an email to info@swedishembedded.com.

//! Spec of the serving build ([`ClipVision::new_serving_on_src`]): it computes the tower the tap-keeping build
//! computes, but over one shared block scratch, with the fused attention and with the block linears on tensor
//! cores (TF32 products of the fp32 weights as stored). The two agree to TF32's rounding; pinned to the fp32
//! reference the serving build agrees with the tap build to the fused attention's summation order alone.

use clip::config::{ClipVisionConfig, TextAct, VisionStem};
use clip::model::{serving_pipelines, ClipVision, PatchSource, CLIP_VISION_PIPELINES};
use gpu_core::provider::library::GemmPrecision;

/// Tiny, with every contraction and output width a multiple of 8 so each product is one a tensor core takes.
fn config() -> ClipVisionConfig {
    ClipVisionConfig {
        shape: gguf::deepseek_ocr_vision::ClipConfig {
            d_model: 64,
            n_layers: 2,
            n_heads: 4,
            ffn_hidden: 128,
            patch_size: 2,
            image_size: 16, // native grid 8x8
            n_positions: 65,
            layer_norm_eps: 1e-5,
        },
        act: TextAct::QuickGelu,
        stem: VisionStem::Clip,
    }
}

fn relative_error(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    assert!(scale > 1e-3, "the comparison needs a signal, got scale {scale}");
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / scale
}

#[test]
fn the_serving_build_tracks_the_tap_build() {
    let cfg = config();
    let init = clip::init::init_vision_weights(&cfg, 79);
    let px = clip::init::fixed_pixels(&cfg, 2, 83);
    let build = |serving: bool, precision: GemmPrecision| {
        let mut v = if serving {
            ClipVision::new_serving_on_src(gpu_core::testgpu::dev(serving_pipelines()), cfg.clone(), 2, PatchSource::Pixels, &init)
        } else {
            ClipVision::new_on_src(gpu_core::testgpu::dev(CLIP_VISION_PIPELINES), cfg.clone(), 2, PatchSource::Pixels, &init, false)
        };
        v.set_gemm_precision(precision);
        v.set_pixels(&px);
        v.forward();
        (v.uses_tensor_cores(), v.read_output())
    };
    let (taps_tc, want) = build(false, GemmPrecision::Auto);
    assert!(!taps_tc, "the tap build is the fp32 reference");

    let (reference_tc, reference) = build(true, GemmPrecision::Fp32Reference);
    assert!(!reference_tc);
    let err = relative_error(&reference, &want);
    eprintln!("serving build, fp32 products: max abs difference {err:.2e} of scale");
    assert!(err < 1e-4, "the shared scratch and fused attention changed the answer by {err} of scale");

    let (tc, fast) = build(true, GemmPrecision::Auto);
    if !tc {
        return brain_testutil::skip_unavailable("this device has no tensor-core GEMM");
    }
    let err = relative_error(&fast, &want);
    eprintln!("serving build, TF32 products: max abs difference {err:.2e} of scale");
    assert!(fast != reference, "identical output means the products did not run on tensor cores");
    assert!(err < 5e-3, "TF32 products drifted {err} of scale from the fp32 reference");
}

/// **No product of the serving tower falls back to fp32 CUDA cores.** The tower has no convolution on the served path (the patch
/// tokens come from SAM), so every product is a tensor-core one on the device's own kernel timer.
#[test]
fn the_serving_tower_runs_its_products_on_tensor_cores() {
    let cfg = config();
    let init = clip::init::init_vision_weights(&cfg, 79);
    let gpu = gpu_core::testgpu::dev(serving_pipelines());
    let v = ClipVision::new_serving_on_src(gpu, cfg, 2, PatchSource::Tokens { grid: (8, 8) }, &init);
    if !v.uses_tensor_cores() {
        return brain_testutil::skip_unavailable("this device has no tensor-core GEMM");
    }
    if !v.gpu.set_kernel_timing(true) {
        return brain_testutil::skip_unavailable("this backend cannot time kernels");
    }
    let tokens: Vec<f32> = (0..2 * 64 * 64).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
    v.set_tokens(&tokens);
    v.forward();
    let rows = v.gpu.kernel_times().expect("timing was switched on");
    let audit = gpu_core::profile::ArithmeticAudit::of(&rows);
    eprintln!("clip serving tower: {}", audit.summary());
    let fallbacks = audit.fallbacks(&[]);
    assert!(fallbacks.is_empty(), "products on CUDA cores in the serving tower: {fallbacks:?}\n{}", audit.summary());
    assert!(audit.tensor_core_ms > 0.0, "no tensor-core kernel ran at all: {rows:?}");
}
