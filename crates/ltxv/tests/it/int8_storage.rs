// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! INT8 STORAGE format for the video-only DiT's weights (storage only) -
//! three things, matching this crate's own `dit_parity.rs`/`lora_train.rs`
//! style:
//!
//! 1. The never-quantize predicate ([`ltxv::int8::is_never_quantized`])
//!    pinned against `dit::dit_tensor_manifest`'s REAL tensor names - a
//!    typo'd substring that silently excludes nothing (or the wrong thing)
//!    would otherwise pass unnoticed.
//! 2. A per-tensor round trip (quantize -> dequantize) over every eligible
//!    weight of a tiny random-weight [`LtxDit`], at the cosine bar
//!    `model::int8`'s own tests already use.
//! 3. The test that actually matters: the SAME tiny [`LtxDit`] forward pass
//!    run twice - once at plain f32, once with every eligible weight
//!    round-tripped through int8 storage first - and the two outputs
//!    compared. This bounds int8's real accuracy cost on an actual model
//!    forward, not just per-tensor norm preservation.
//!
//! No fixture dependency (unlike `dit_parity.rs`): [`random_tiny_weights`]
//! needs no golden, so these tests always run.

use ltxv::dit::{dit_tensor_manifest, random_tiny_weights};
use ltxv::int8::{dequantize_tensors, is_never_quantized, quantize_tensors};
use ltxv::modelgrad::Cfg;
use ltxv::{LtxDit, LtxDitConfig};

// ------------------------------------------------------------------ metrics

/// Same f64-accumulation formula `dit_parity.rs`/`av_dit_parity.rs`/
/// `model::hostmath::cosine` use.
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

// ------------------------------------------------------- synthetic inputs

/// A deterministic, non-degenerate forward input for `cfg` at `t` tokens /
/// `context_len` context rows - same shapes `LtxDit::forward` documents
/// (`latent[t*in_channels]`, `timesteps[t]`, `positions[3*t*2]`,
/// `keyframes_mask[t]`, `context[context_len*cross_attention_dim]`).
/// Positions reuse `modelgrad::Cfg::simple_positions`'s own 1-D frame grid
/// (already exercised by every training test in this crate); latent/context
/// use a plain deterministic formula, no RNG needed for a fixed-seed
/// comparison between two runs of the exact same inputs.
struct Inputs {
    latent: Vec<f32>,
    timesteps: Vec<f32>,
    positions: Vec<f32>,
    keyframes_mask: Vec<f32>,
    context: Vec<f32>,
    context_len: usize,
    t: usize,
}

fn synthetic_inputs(cfg: &LtxDitConfig, t: usize, context_len: usize) -> Inputs {
    let mcfg = Cfg::from_ltx(cfg, t, context_len);
    let positions = mcfg.simple_positions();
    let latent: Vec<f32> = (0..t * cfg.in_channels as usize).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.1).collect();
    let context: Vec<f32> = (0..context_len * cfg.cross_attention_dim as usize).map(|i| ((i % 7) as f32 / 7.0 - 0.5) * 1.4).collect();
    let timesteps: Vec<f32> = (0..t).map(|i| 0.2 + 0.05 * (i % 5) as f32).collect();
    let mut keyframes_mask = vec![0f32; t];
    keyframes_mask[0] = 1.0;
    Inputs { latent, timesteps, positions, keyframes_mask, context, context_len, t }
}

// ------------------------------------------------------- 1. predicate pin

#[test]
fn never_quantize_predicate_matches_the_real_tensor_manifest() {
    let cfg = LtxDitConfig::tiny();
    let manifest = dit_tensor_manifest(&cfg);
    let names: Vec<&str> = manifest.iter().map(|(n, _)| n.as_str()).collect();

    // Every never-quantize category this crate's manifest can express must
    // hit at least one real name - a category that matches nothing means
    // either the pattern is stale or the naming convention guess was wrong.
    for category in ["patchify_proj", "adaln_single", "proj_out", "scale_shift_table"] {
        assert!(
            names.iter().any(|n| n.contains(category) && is_never_quantized(n)),
            "category '{category}' matched no real tensor name in dit_tensor_manifest"
        );
    }

    // Exact real names the predicate must exclude.
    for must_exclude in [
        "patchify_proj.weight",
        "patchify_proj.bias",
        "adaln_single.emb.timestep_embedder.linear_1.weight",
        "adaln_single.linear.weight",
        "adaln_single.linear.bias",
        "scale_shift_table",
        "proj_out.weight",
        "proj_out.bias",
        "transformer_blocks.0.scale_shift_table",
        "transformer_blocks.0.prompt_scale_shift_table",
        "transformer_blocks.1.scale_shift_table",
    ] {
        assert!(names.contains(&must_exclude), "sanity: '{must_exclude}' should be in dit_tensor_manifest");
        assert!(is_never_quantized(must_exclude), "'{must_exclude}' must be on the never-quantize list");
    }

    // Exact real names that must remain int8-eligible.
    for must_quantize in [
        "transformer_blocks.0.attn1.to_q.weight",
        "transformer_blocks.0.attn1.to_k.weight",
        "transformer_blocks.0.attn1.to_v.weight",
        "transformer_blocks.0.attn1.to_out.0.weight",
        "transformer_blocks.0.attn2.to_q.weight",
        "transformer_blocks.0.ff.net.0.proj.weight",
        "transformer_blocks.0.ff.net.2.weight",
    ] {
        assert!(names.contains(&must_quantize), "sanity: '{must_quantize}' should be in dit_tensor_manifest");
        assert!(!is_never_quantized(must_quantize), "'{must_quantize}' must stay int8-eligible");
    }

    // Full-manifest sanity: 1D non-bias tensors (norm gains, the keyframes
    // embedding) are irrelevant to int8 eligibility either way (rank != 2
    // already excludes them), but none should accidentally collide with a
    // never-quantize substring either - a canary for an over-broad pattern.
    for (name, shape) in &manifest {
        if shape.len() == 1 && !name.ends_with(".bias") {
            assert!(!is_never_quantized(name), "unexpected never-quantize match on a norm/embedding tensor: {name}");
        }
    }
}

// --------------------------------------------------- 2. per-tensor round trip

#[test]
fn quantize_then_dequantize_preserves_every_eligible_tensor() {
    let cfg = LtxDitConfig::tiny();
    let w = random_tiny_weights(&cfg, 0xD17_1234);

    let q = quantize_tensors(&w);
    assert!(!q.int8.is_empty(), "tiny config must have at least one int8-eligible tensor");
    // Every never-quantized / ineligible tensor must survive bit-for-bit in `full`.
    for (name, (shape, data)) in &w {
        if !q.int8.contains_key(name) {
            let (fshape, fdata) = q.full.get(name).unwrap_or_else(|| panic!("'{name}' missing from quantize_tensors output"));
            assert_eq!(fshape, shape);
            assert_eq!(fdata, data, "never-quantized tensor '{name}' must be untouched");
        }
    }

    let deq = dequantize_tensors(&q);
    let mut worst = (1.0f64, String::new());
    for name in q.int8.keys() {
        let (_, orig) = w.get(name).expect("original tensor");
        let (_, got) = deq.get(name).expect("dequantized tensor");
        let c = cosine(orig, got);
        assert!(c >= 0.999, "'{name}': per-tensor round-trip cosine {c:.6} too low");
        if c < worst.0 {
            worst = (c, name.clone());
        }
    }
    println!("int8 storage: {} tensors round-tripped, worst per-tensor cosine {:.9} ({})", q.int8.len(), worst.0, worst.1);
}

// ------------------------------------------------ 3. the test that matters

#[test]
fn dit_forward_stays_close_after_int8_storage_round_trip() {
    let cfg = LtxDitConfig::tiny();
    let w = random_tiny_weights(&cfg, 0xD17_5678);
    let inputs = synthetic_inputs(&cfg, 7, 5);

    let context_valid = vec![1.0f32; inputs.context_len];
    let model_f32 = LtxDit::new(cfg, w.clone(), None);
    let taps_f32 = model_f32.forward(&inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &context_valid);

    let q = quantize_tensors(&w);
    let w_roundtripped = dequantize_tensors(&q);
    let model_i8 = LtxDit::new(cfg, w_roundtripped, None);
    let taps_i8 = model_i8.forward(&inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &context_valid);

    let c_out = cosine(&taps_f32.out, &taps_i8.out);
    println!("int8 storage forward parity: final output cosine = {c_out:.9}");
    // Measured on this fixture: final-output cosine lands at 0.999999+ (a
    // 2-layer, dim-64 tiny config with the modulation/patchify/output tables
    // held at full f32 keeps int8 noise from ever reaching the output through
    // more than a couple of attention/FFN projections). Matching FLUX.2's own
    // int8 tests' documented approach (measure, then pick a threshold with a
    // sane margin below the measured number) rather than assuming exact 1.0.
    assert!(c_out >= 0.9999, "int8-storage-round-tripped forward diverged too far from f32: cosine {c_out:.9}");

    for (i, (a, b)) in taps_f32.block_out.iter().zip(&taps_i8.block_out).enumerate() {
        let c = cosine(a, b);
        println!("int8 storage forward parity: block {i} output cosine = {c:.9}");
        assert!(c >= 0.999, "block {i} output diverged too far: cosine {c:.9}");
    }
}
