// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Two-way import coverage against the REAL `SUPIR-v0Q_fp32.safetensors`
//! checkpoint - the mapping-units rung on real data rather than the
//! synthetic checkpoint `crates/supir/src/import.rs`'s own tests build by
//! hand.
//!
//! This is the strongest available check on [`supir::import`]'s
//! CompVis/LDM -> diffusers-style rename without a real forward: the
//! synthetic checkpoint in `src/import.rs`'s tests is constructed FROM the
//! same block-schedule walk the importer itself uses, so it cannot catch a
//! rename bug both share - only the real file's independent key set can.
//!
//! Real-checkpoint forward parity (trunk hidden states, adaptor taps, the
//! frozen UNet's raw output) is a SEPARATE, harder gate blocked on the
//! sampler's unsaved churn-noise draw - see `tests/schedule_parity.rs`'s
//! module doc for why, and why this test does not attempt it either.
//!
//! Skips itself (never fails) when the checkpoint is absent - SUPIR's
//! weights are non-commercial-licensed with no `default_ref`/auto-fetch, so
//! a fresh checkout never has them unless a user placed them.

use std::path::PathBuf;

use supir::config::SupirConfig;

/// `BRAIN_SUPIR_WEIGHTS`, else the same `resources/supir/` layout the
/// licence note in `crates/supir/src/lib.rs` and this workspace's other
/// `resources/<model>/fetch.py` scripts use.
fn weights_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BRAIN_SUPIR_WEIGHTS") {
        let pb = PathBuf::from(p);
        return pb.is_file().then_some(pb);
    }
    let p = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../resources/supir/supir_v0q/SUPIR/SUPIR-v0Q_fp32.safetensors"
    ));
    p.is_file().then_some(p)
}

#[test]
fn real_checkpoint_two_way_coverage() {
    let Some(path) = weights_path() else {
        brain_testutil::skip("set BRAIN_SUPIR_WEIGHTS to a SUPIR-v0Q_fp32.safetensors path");
        return;
    };
    println!("reading {} ...", path.display());
    let src = checkpoint::safetensors::read(path.to_str().expect("utf-8 path")).expect("read checkpoint");
    println!("read {} raw tensors", src.len());

    let cfg = SupirConfig::sdxl();
    let manifest = cfg.tensor_manifest();

    // The real checkpoint really does carry `mask_LQ` - confirmed against
    // the file itself, not just the synthetic checkpoint `src/import.rs`'s
    // own tests construct by hand. The REJECTION logic itself is already
    // gated there (`import::tests::mask_lq_is_rejected_by_name`); cloning
    // the full 5.3 GB map a second time here just to re-exercise the same
    // branch would double this test's memory footprint for no new coverage.
    assert!(src.iter().any(|t| t.name == "model.control_model.mask_LQ"), "the real checkpoint no longer carries mask_LQ - update the roadmap");
    println!("confirmed: real checkpoint's mask_LQ is present (rejected-by-name path is gated on the synthetic checkpoint)");

    let mut raw: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)> =
        src.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
    raw.remove("model.control_model.mask_LQ");
    let tensors = supir::import::remap(raw, &cfg).expect("supir::import::remap (mask_LQ removed)");

    assert_eq!(tensors.len(), manifest.len(), "imported tensor count vs the canonical manifest");
    let params: usize = tensors.values().map(|(_, d)| d.len()).sum();
    println!("imported {} tensors, {params} parameters = {:.3} GB fp32", tensors.len(), params as f64 * 4.0 / 1e9);

    for (name, shape) in &manifest {
        let (got_shape, data) = tensors.get(name).unwrap_or_else(|| panic!("import produced no {name}"));
        assert_eq!(got_shape, shape, "{name} shape");
        assert_eq!(data.len(), shape.iter().product::<usize>(), "{name} element count");
    }

    // The zero-init convs really are (near-)zero in the released checkpoint
    // - a sanity check that this loaded the RIGHT file (a randomly
    // initialised or corrupted one would not have this property) distinct
    // from the shape checks above.
    let j = cfg.adaptors.joins[0];
    let (_, zc) = tensors.get(&format!("project_modules.{}.zero_conv.weight", j.pm_idx)).expect("zero_conv");
    let max_abs = zc.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    println!("project_modules.{}.zero_conv.weight max|.| = {max_abs:e}", j.pm_idx);
    assert!(max_abs < 1.0, "zero_conv doesn't look zero-init: max|.|={max_abs}");
}
