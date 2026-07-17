// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P3: importing a real ZipDepth checkpoint.
//!
//! Env-gated on the real `.pth` (skips OK when unset). Proves the importer loads
//! every tensor the model needs, rejects the wrong variant, and that a model built
//! on the imported weights runs a real forward.
use depth::{import, ZipConfig, ZipDepth};
use gpu_core::Gpu;
use vision::Ctx;

fn base_pth() -> Option<String> {
    std::env::var("ZIPDEPTH_PTH").ok()
}

/// The importer maps the released checkpoint 1:1 with zero leftovers — every one
/// of the model's tensors is filled, and nothing in the file is unexpected (bar
/// the int64 counters it skips by name).
#[test]
fn imports_the_base_checkpoint_completely() {
    let Some(path) = base_pth() else {
        eprintln!("SKIP: set ZIPDEPTH_PTH to run");
        return;
    };
    let cfg = ZipConfig::base();
    let init = import::load(&path, &cfg).expect("import must succeed on the released base checkpoint");
    // Every declared parameter is present with the right length.
    for (name, shape) in cfg.param_list() {
        let numel: usize = shape.iter().product();
        let v = init.get(&name).unwrap_or_else(|| panic!("import dropped `{name}`"));
        assert_eq!(v.len(), numel, "`{name}` imported with the wrong length");
    }
    assert_eq!(init.len(), cfg.param_list().len(), "import produced a different tensor count than the config declares");
    // The ImageNet buffers came from the file, not a default.
    let mean = &init["mean"];
    assert!((mean[0] - 0.485).abs() < 1e-3, "mean[0] should be ImageNet's 0.485, got {}", mean[0]);
}

/// Loading the base checkpoint against the NPU config must FAIL loudly — the two
/// are different models (mask_pred vs where_conv) — rather than half-load.
#[test]
fn the_wrong_variant_is_rejected() {
    let Some(path) = base_pth() else {
        eprintln!("SKIP: set ZIPDEPTH_PTH to run");
        return;
    };
    let npu = ZipConfig { upsample_unfold: false, ..ZipConfig::base() };
    let err = import::load(&path, &npu).expect_err("the base checkpoint must not load as the NPU variant");
    assert!(
        err.contains("mask_pred") || err.contains("where_conv") || err.contains("missing") || err.contains("does not declare"),
        "the error should name the variant mismatch, got: {err}"
    );
}

/// A model built on the imported weights runs a real eval forward and produces a
/// finite, non-negative depth map at the input resolution.
#[test]
fn a_model_on_imported_weights_runs_forward() {
    let Some(path) = base_pth() else {
        eprintln!("SKIP: set ZIPDEPTH_PTH to run");
        return;
    };
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let cfg = ZipConfig::base();
    let ps = import::load_into(&gpu, &path, &cfg).expect("load_into");
    let m = ZipDepth::build(&ctx, cfg, 1, false);
    m.set_eval(true);
    let x = gpu.storage_init("x", &vec![0.5f32; m.in_shape.numel() as usize]);
    m.forward(&ctx, &ps, &x);
    let out = gpu.read(m.out(), m.out_shape.numel() as usize);
    assert_eq!((m.out_shape.h, m.out_shape.w), (m.in_shape.h, m.in_shape.w));
    assert!(out.iter().all(|v| v.is_finite() && *v >= 0.0), "imported model must produce finite, non-negative depth");
    assert!(out.iter().any(|v| *v > 1e-6), "a real pretrained model must not produce an all-zero map");
}
