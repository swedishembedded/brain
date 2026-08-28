// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-checkpoint int8-storage parity for the SDXL UNet, against the
//! already-committed `resources/supir/sdxl_base/sd_xl_base_1.0_0.9vae.safetensors`
//! (LDM format, read via `sdxlunet::import::load_ldm` - the same file
//! `crates/supir`'s own real-checkpoint test loads, so no separate download
//! is needed to run this).
//!
//! `crates/sdxlunet/tests/parity.rs::sdxl_unet_forward_matches_diffusers`
//! already gates plain fp32 against a real diffusers dump at cosine 0.9999 -
//! that is the "fp32 gated at a tight floor" half of
//! `crates/flux1/tests/dit_parity.rs`'s two-gate convention, already in this
//! crate. What is new here is the int8 half: the SAME real weights, built
//! twice - once fp32 ([`Unet::new`]), once via [`Unet::new_quantized`] with
//! every eligible weight round-tripped through `sdxlunet::int8` storage
//! first - run through the SAME synthetic forward and compared to each
//! other. There is no external int8 reference to dump (nothing produces
//! this packed format but this port), so - like `crates/ltxv`'s own
//! `int8_storage.rs` - the fp32 build IS the reference for this file; the
//! tight-floor-vs-diffusers half already lives in `parity.rs`.
//!
//! int8 is a lossy tier; the floor (0.95, matching `crates/flux1/tests/
//! dit_parity.rs::full_depth_int8_parity`'s) only needs to catch a BROKEN
//! port. The printed per-run cosine/rel_l2 above it is the deliverable.
//!
//! A 32x32 latent (256x256 image), matching `parity.rs`'s own speed budget -
//! this is a plausibility/regression gate on int8 noise, not a resolution or
//! speed claim.

use sdxlunet::config::UNetConfig;
use sdxlunet::model::{Unet, KERNELS};

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch ({} vs {})", a.len(), b.len());
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    let den = na.sqrt() * nb.sqrt();
    if den <= 0.0 {
        0.0
    } else {
        d / den
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let diff = x as f64 - y as f64;
        num += diff * diff;
        den += x as f64 * x as f64;
    }
    if den <= 0.0 {
        0.0
    } else {
        (num / den).sqrt()
    }
}

/// `BRAIN_SUPIR_SDXL_LDM`, else the committed
/// `resources/supir/sdxl_base/sd_xl_base_1.0_0.9vae.safetensors` -
/// deliberately the SAME env var name `crates/supir/tests/parity.rs` uses,
/// since it is the same file.
fn sdxl_ldm_weights_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("BRAIN_SUPIR_SDXL_LDM") {
        let pb = std::path::PathBuf::from(p);
        return pb.is_file().then_some(pb);
    }
    let p = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/supir/sdxl_base/sd_xl_base_1.0_0.9vae.safetensors"));
    p.is_file().then_some(p)
}

#[test]
fn int8_forward_stays_close_to_fp32_on_the_real_sdxl_checkpoint() {
    let Some(sdxl_path) = sdxl_ldm_weights_path() else {
        brain_testutil::skip("set BRAIN_SUPIR_SDXL_LDM to a single-file sd_xl_base_1.0*.safetensors path");
        return;
    };

    let cfg = UNetConfig::sdxl_base();
    println!("importing SDXL UNet from {} ...", sdxl_path.display());
    let w = sdxlunet::import::load_ldm(sdxl_path.to_str().expect("utf-8 path"), &cfg).expect("sdxlunet::import::load_ldm");
    let params: usize = w.values().map(|(_, d)| d.len()).sum();
    println!("{} tensors, {params} parameters = {:.2} GB fp32", w.len(), params as f64 * 4.0 / 1e9);

    let (h, w_lat, t_enc) = (32u32, 32u32, 8u32);
    let sample: Vec<f32> = (0..(cfg.in_channels * h * w_lat) as usize).map(|i| ((i as f32) * 0.013).sin()).collect();
    let enc: Vec<f32> = (0..(t_enc * cfg.cross_attention_dim) as usize).map(|i| ((i as f32) * 0.029).cos()).collect();
    let pooled: Vec<f32> = (0..cfg.pooled_dim() as usize).map(|i| ((i as f32) * 0.07).sin()).collect();
    let time_ids = [1024.0f32, 1024.0, 0.0, 0.0, 1024.0, 1024.0];
    let timestep = 601.0f32;

    println!("building the fp32 model ...");
    let model_f32 = Unet::new(gpu_core::testgpu::dev(&KERNELS), cfg.clone(), &w, h, w_lat, t_enc, false);
    let out_f32 = model_f32.run(&sample, timestep, &enc, &pooled, &time_ids);
    drop(model_f32); // frees this build's device buffers before the int8 build allocates its own

    println!("quantizing every eligible weight to int8 storage ...");
    let q = sdxlunet::int8::quantize_tensors(&w);
    println!(
        "{} of {} tensors quantized (packed ~{:.2} GB vs {:.2} GB fp32 for those tensors alone)",
        q.packed.len(),
        w.len(),
        q.packed.values().map(|p| p.packed.len() as f64 * 4.0 + p.scale.len() as f64 * 4.0).sum::<f64>() / 1e9,
        q.packed.values().map(|p| p.shape.iter().product::<usize>() as f64 * 4.0).sum::<f64>() / 1e9,
    );
    drop(w); // the fp32 source is no longer needed once it is quantized

    println!("building the int8 model ...");
    let model_i8 = Unet::new_quantized(gpu_core::testgpu::dev(&KERNELS), cfg.clone(), &q.full, &q.packed, h, w_lat, t_enc, false);
    let out_i8 = model_i8.run(&sample, timestep, &enc, &pooled, &time_ids);

    let c = cosine(&out_f32, &out_i8);
    let rl2 = rel_l2(&out_f32, &out_i8);
    println!("int8 storage forward parity: out cosine {c:.9} (1-cos {:.3e}), rel_l2 {rl2:.3e}", 1.0 - c);
    // int8 is a lossy tier; the floor only catches a BROKEN port - the
    // number printed above is the deliverable, not this assertion.
    assert!(c >= 0.95, "int8-quantized forward diverged too far from fp32: cosine {c:.9}");
    assert!(out_i8.iter().all(|v| v.is_finite()), "int8 forward produced a non-finite output");
}
