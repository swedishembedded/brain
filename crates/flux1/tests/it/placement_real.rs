// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The placement estimates `flux1::pipeline::part_needs` hands the planner,
//! checked against the device bytes the real parts take on a card: T5-XXL at
//! `MAX_TXT_LEN` tokens through one forward, built the way the pipeline builds
//! it, and the int8 DiT at 256 px through one forward. The peak is the lowest free figure the driver
//! reports while the part is built and run, less the free figure once the
//! card's context exists, so the driver context (which the planner's
//! headroom covers, not the part's estimate) is not charged to the part.
//!
//! Swedish Embedded AB implements solutions for fitting large diffusion
//! pipelines onto the accelerators a machine really has for its clients. If
//! your team needs expertise in device memory planning then you can procure
//! our services by sending an email to info@swedishembedded.com.

use flux1::pipeline::{part_needs, MAX_TXT_LEN, TE_TAPS};
use flux1::{Flux1Config, Flux1Model, Precision};
use gpu_core::Gpu;

const MIB: f64 = (1u64 << 20) as f64;

fn checkpoint() -> Option<String> {
    let root = brain_testutil::model_dir("black-forest-labs/FLUX.1-dev").filter(|d| std::path::Path::new(d).join("transformer").exists());
    if root.is_none() {
        brain_testutil::skip("the flux1 placement estimates need the black-forest-labs/FLUX.1-dev diffusers checkpoint in the model store");
    }
    root
}

/// The idle card with the most free bytes the driver could measure, or a
/// skip when there is none (no NVIDIA driver tool, or every card in use).
fn idle_card(needed: u64) -> Option<u32> {
    let card = gpu_core::capacity::probe_gpus().into_iter().filter(|g| g.measured && g.free >= needed).max_by_key(|g| g.free);
    if card.is_none() {
        brain_testutil::skip_unavailable(&format!("no card with {:.0} MiB free whose memory nvidia-smi can read", needed as f64 / MIB));
    }
    card.map(|g| g.index)
}

fn free_on(card: u32) -> u64 {
    gpu_core::capacity::free_on(card).expect("probed card")
}

/// Run `f` on `card` and return its result with the peak bytes it took past
/// `baseline_free`.
fn peak_while<R>(card: u32, baseline_free: u64, f: impl FnOnce() -> R) -> (R, u64) {
    gpu_core::capacity::peak_taken_while(card, baseline_free, || gpu_core::devices::with_gpu(card, f).expect("card in the registry"))
}

fn assert_within(what: &str, estimate: u64, measured: u64) {
    eprintln!("{what}: estimate {:.0} MiB, measured peak {:.0} MiB", estimate as f64 / MIB, measured as f64 / MIB);
    let bounds = gpu_core::capacity::estimate_bounds(estimate);
    assert!(
        bounds.contains(&measured),
        "{what}: measured peak {:.0} MiB is outside the {:.0}..{:.0} MiB band around the {:.0} MiB estimate",
        measured as f64 / MIB,
        *bounds.start() as f64 / MIB,
        *bounds.end() as f64 / MIB,
        estimate as f64 / MIB
    );
}

fn estimate_of(name: &str, precision: Precision, n_joint: u64) -> u64 {
    part_needs(&Flux1Config::dev(), precision, n_joint, 0).into_iter().find(|n| n.name == name).expect("declared part").vram
}

#[test]
#[ignore = "slow (real): builds T5-XXL from FLUX.1-dev on a GPU and runs one 512-token forward"]
fn the_text_encoder_estimate_matches_its_measured_peak() {
    let Some(root) = checkpoint() else { return };
    let estimate = estimate_of("te", Precision::Int8, 0);
    let Some(card) = idle_card(estimate) else { return };
    let cfg = t5encoder::config::T5Config::xxl();
    let tensors = t5encoder::import::read_encoder(&std::path::Path::new(&root).join("text_encoder_2")).expect("T5-XXL shards");
    let init: std::collections::HashMap<String, Vec<f32>> =
        t5encoder::import::import_hf(tensors, &cfg).expect("T5-XXL import").into_iter().map(|(k, (_, d))| (k, d)).collect();
    let gpu = gpu_core::devices::with_gpu(card, || Gpu::new(t5encoder::model::PIPELINES)).expect("card");
    let baseline = free_on(card);
    let (hidden, measured) = peak_while(card, baseline, move || {
        let t5 = t5encoder::model::T5Encoder::new_with(gpu, cfg, 1, MAX_TXT_LEN, &init, TE_TAPS);
        t5.set_tokens(&vec![1u32; MAX_TXT_LEN as usize]);
        t5.forward();
        t5.read_hidden()
    });
    assert!(hidden.iter().all(|v| v.is_finite()), "T5-XXL forward produced a non-finite value");
    assert_within("te (T5-XXL, 512 tokens)", estimate, measured);
}

#[test]
#[ignore = "slow (real): builds the int8 FLUX.1-dev DiT on a GPU and runs one 256 px forward"]
fn the_int8_dit_estimate_matches_its_measured_peak() {
    let Some(root) = checkpoint() else { return };
    let cfg = Flux1Config::dev();
    let (lh, lw) = (256 / 16, 256 / 16);
    let n_gen = lh * lw;
    let n_joint = (n_gen + MAX_TXT_LEN as usize) as u32;
    let estimate = estimate_of("dit", Precision::Int8, n_joint as u64);
    let Some(card) = idle_card(estimate) else { return };
    let dir = std::path::Path::new(&root).join("transformer");
    let mut files: Vec<_> = std::fs::read_dir(&dir).expect("transformer dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "safetensors")).collect();
    files.sort();
    let mut raw = Vec::new();
    for f in files {
        raw.extend(checkpoint::safetensors::read(f.to_str().expect("UTF-8 path")).expect("DiT shard"));
    }
    let ts = flux1::import_diffusers(raw, &cfg).expect("DiT import");
    let gpu = gpu_core::devices::with_gpu(card, || Gpu::new(flux1::KERNELS)).expect("card");
    let baseline = free_on(card);
    let mut rng = data::rng::Lcg::new(7);
    let lat = rng.vec(n_gen * cfg.in_channels);
    let ctx = rng.vec(MAX_TXT_LEN as usize * cfg.context_in_dim);
    let pooled = rng.vec(cfg.vec_in_dim);
    let ids = flux1::position_ids(MAX_TXT_LEN as usize, lh, lw, &[]);
    let (pred, measured) = peak_while(card, baseline, move || {
        let dit = Flux1Model::new_with(&cfg, &ts, gpu, n_joint, Precision::Int8);
        drop(ts);
        dit.forward(&lat, &ctx, &pooled, 1.0, 3.5, &ids, n_gen)
    });
    assert!(pred.iter().all(|v| v.is_finite()), "DiT forward produced a non-finite value");
    assert_within("dit (int8, 256 px)", estimate, measured);
}
