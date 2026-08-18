// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The load-bearing cross-check the porting playbook's training section
//! demands: the NEW host-only `ltxv::modelgrad::forward` (plain host math,
//! no GPU dispatch at all - see `ltxv::grad`'s module doc) must compute the
//! SAME thing as `ltxv::LtxDit::forward`, the already-parity-proven
//! GPU-dispatched path (`tests/dit_parity.rs`, cosine >= 0.999999 against
//! the real reference). This test replays the SAME golden fixture
//! `dit_parity.rs` loads (real tiny weights, real dumped inputs) through
//! BOTH forwards at f32 and asserts cosine ~1.0 between the two final
//! outputs - proof the from-scratch reimplementation is the same math, not
//! merely "also a plausible-looking DiT forward".
//!
//! Skips loudly without the fixture (`BRAIN_REQUIRE_FIXTURES=1` upgrades a
//! skip to a failure), the same convention `dit_parity.rs`/`vae_parity.rs`
//! use.

use std::path::Path;

use ltxv::modelgrad::{forward, Cfg, ModelWeights};
use ltxv::{load_tiny_weights, LtxDit, LtxDitConfig};

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch ({} vs {})", a.len(), b.len());
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    let den = na.sqrt() * nb.sqrt();
    if den <= 0.0 {
        0.0
    } else {
        d / den
    }
}

#[test]
fn host_training_forward_matches_the_gpu_dispatched_forward() {
    let fx_path = brain_testutil::testdata("golden/ltxv/dit/dit_tiny.safetensors");
    let w_path = brain_testutil::testdata("golden/ltxv/dit/dit_tiny_weights.safetensors");
    if !Path::new(&fx_path).exists() || !Path::new(&w_path).exists() {
        brain_testutil::skip(&format!("fixture {fx_path} absent - run tools/goldens/ltxv_dit_dump_reference.py"));
        return;
    }
    let fx = checkpoint::safetensors::read(&fx_path).expect("read golden");
    let get = |name: &str| -> &[f32] { &fx.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).data };
    let shape = |name: &str| -> &[usize] { &fx.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).shape };

    let ltx_cfg = LtxDitConfig::tiny();
    let t = shape("latent")[0];
    let context_len = shape("context")[0];
    let latent = get("latent");
    let context = get("context");
    let timesteps_f32 = get("timesteps");
    let positions = get("positions");
    let keyframes_mask_f32 = get("keyframes_mask");

    // --- the GPU-dispatched, parity-proven path ---
    let w_gpu = load_tiny_weights(&w_path);
    let gpu_model = LtxDit::new(ltx_cfg, w_gpu.clone(), None);
    let gpu_taps = gpu_model.forward(latent, timesteps_f32, positions, keyframes_mask_f32, context, context_len, t);

    // --- the new host-only training reference, same weights/inputs ---
    let cfg = Cfg::from_ltx(&ltx_cfg, t, context_len);
    let host_w = ModelWeights::from_tensors(&cfg, &w_gpu).expect("host weights from golden tensors");
    let timesteps: Vec<f64> = timesteps_f32.iter().map(|&v| v as f64).collect();
    let keyframes_mask: Vec<f64> = keyframes_mask_f32.iter().map(|&v| v as f64).collect();
    let tables = cfg.rope_tables_f32(positions);
    let (host_pred, _cache) = forward(&cfg, &host_w, latent, &timesteps, &keyframes_mask, context, &tables.cos, &tables.sin);

    let c = cosine(&host_pred, &gpu_taps.out);
    let max_abs = host_pred.iter().zip(&gpu_taps.out).map(|(&a, &b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("host-vs-gpu forward: cosine={c:.10}  max_abs={max_abs:.3e}  n={}", host_pred.len());
    assert!(c >= 0.999999, "host training forward must match the GPU-dispatched forward: cosine {c:.10}");

    // The RoPE table this test built independently must also agree with the
    // GPU path's own table (both derive from the same `crate::rope::ltx_rope_tables`
    // call, but at different call sites - a cheap extra cross-check).
    let rope_cos_match = cosine(&tables.cos, &gpu_taps.rope_cos);
    assert!(rope_cos_match >= 0.999999, "rope table mismatch: cosine {rope_cos_match:.10}");
}
