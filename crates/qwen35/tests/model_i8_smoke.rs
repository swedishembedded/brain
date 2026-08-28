// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P0 smoke + parity test for `qwen35::model::Qwen35`'s int8 (DP4A) inference
//! path (`Qwen35::new_i8`/`Qwen35::new_on_i8`, dispatching every per-layer
//! mixer/MLP linear through `model::ops::{Ops, Weight}` - see
//! `crate::model::is_i8_linear`'s own doc): not a numerical-parity-against-HF
//! test (no `torch`/`transformers` in this environment), but proof that the
//! int8 wiring is a genuine (if lossy) approximation of the SAME fp32
//! computation `crates/qwen35/tests/model_smoke.rs` already exercises, not a
//! different one: build both an fp32 and an int8 `Qwen35` from the IDENTICAL
//! fresh init weights, run `forward`, and confirm the outputs track each
//! other within a generous but real tolerance.
//!
//! ## Why this test does NOT use `Qwen35Config::tiny()` verbatim
//!
//! `model::int8::quantize_weight` scales a weight per 32-element GROUP of its
//! contraction dimension (`model::int8::GROUP`, Q8_0's own block), so every
//! quantized linear's `k` must be a whole number of groups. `tiny()`'s
//! `q_dim=120`, `linear_value_dim=120` and `intermediate_size=112` are not,
//! and `tiny()` is shared with the fp32 smoke suite and the gradient checker,
//! which have no such constraint and should not pay a bigger fixture for it.
//! So this file uses [`tiny_i8`] - `tiny()`'s exact shape (4 layers,
//! `interval=4` so layer 3 is GQA and the rest GDN, a multi-chunk GDN
//! sequence, every dimension still distinct from every other) with each
//! quantized `k` rounded to a multiple of 32. This mirrors what
//! `crates/qwen35moe/tests/model_i8_smoke.rs` already does for its own crate.

use gpu_core::Gpu;
use qwen35::config::{LayerType, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

fn init_weights(cfg: &Qwen35Config, seed: u64) -> std::collections::HashMap<String, Vec<f32>> {
    qwen35::init::init_weights(cfg, seed)
}

/// See this file's module doc for why this exists instead of `tiny()`. The
/// fixture itself lives next to `tiny()` in `config.rs` (the crate's int8
/// unit tests need it too, and one canonical int8-legal shape beats two
/// copies that can drift); [`tiny_i8_cfg_clears_the_int8_scale_group_bar`]
/// below is its contract test.
fn tiny_i8() -> Qwen35Config {
    Qwen35Config::tiny_i8()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in got.iter().zip(want.iter()) {
        num += ((a - b) as f64).powi(2);
        den += (*b as f64).powi(2);
    }
    (num / den.max(1e-12)).sqrt()
}

/// Sanity check on [`tiny_i8`] itself, independent of any device: every
/// CONTRACTION dimension the int8 (DP4A) path quantizes along really is a
/// whole number of `model::int8::GROUP`s, and the fixture still exercises
/// both mixer types and a multi-chunk GDN sequence - this file's module doc
/// asserts these rather than trusting them, in case a config field is
/// misremembered.
#[test]
fn tiny_i8_cfg_clears_the_int8_scale_group_bar() {
    let cfg = tiny_i8();
    let g = model::int8::GROUP as u32;
    assert_eq!(cfg.d_model % g, 0, "d_model is the K of q/k/v-proj, in_proj_*, gate/up");
    assert_eq!(cfg.intermediate_size % g, 0, "intermediate_size is the K of down");
    assert_eq!(cfg.linear_value_dim() % g, 0, "linear_value_dim is the K of the GDN out_proj");
    assert_eq!(cfg.q_dim() % g, 0, "q_dim is the K of o_proj");
    // in_proj_qkv / q_proj / k_proj / v_proj contract along `d_model`; their
    // own widths are OUTPUT dims and carry no group constraint. Checked here
    // so a reader does not add a bogus assertion for them later.
    assert_eq!(cfg.linear_conv_dim() % 4, 0, "linear_conv_dim is an output width; only the activation pack's %4 applies");
    assert_eq!(cfg.q_proj_dim() % 4, 0, "q_proj_dim is an output width");
    assert_eq!(cfg.kv_dim() % 4, 0, "kv_dim is an output width");

    let t = cfg.block_size;
    let chunk = qwen35::model::gdn_chunk_size(t);
    assert!(chunk < t && t.is_multiple_of(chunk) && t / chunk >= 2, "must still exercise a real multi-chunk GDN sequence");

    let types = cfg.layer_types();
    assert!(types.contains(&LayerType::Linear));
    assert!(types.contains(&LayerType::Full));
}

/// Runs both an fp32 and an int8 `Qwen35` forward at [`tiny_i8`] from the SAME
/// fresh init weights, and checks the int8 path's logits track the fp32
/// path's within a generous quantization tolerance. Shared by the default-
/// backend and CPU variants below (a barrier-crossing kernel can silently
/// misbehave on exactly one backend, so both matter) - the CPU variant's own
/// much tighter assertion lives in its own test, not here, since CPU
/// genuinely demotes every mixer/MLP linear to fp32 (see that test's doc).
fn run_parity(gpu_fp32: Gpu, gpu_i8: Gpu) -> (f64, f64) {
    let cfg = tiny_i8();
    let b = 1;
    let t = cfg.block_size;
    let init = init_weights(&cfg, 7);

    let fp32 = Qwen35::new_on(gpu_fp32, cfg.clone(), b, t, &init);
    let i8 = Qwen35::new_on_i8(gpu_i8, cfg.clone(), b, t, &init);

    let tokens: Vec<u32> = (0..t).map(|i| (i * 3 + 1) % cfg.vocab).collect();
    let logits_fp32 = fp32.logits_all(&tokens);
    let logits_i8 = i8.logits_all(&tokens);

    assert_eq!(logits_fp32.len(), logits_i8.len());
    assert_eq!(logits_fp32.len(), (t * cfg.vocab) as usize);
    assert!(logits_fp32.iter().all(|v| v.is_finite()), "fp32 reference produced a non-finite logit");
    assert!(logits_i8.iter().all(|v| v.is_finite()), "int8 path produced a non-finite logit");
    assert!(logits_fp32.iter().any(|&v| v.abs() > 1e-6), "fp32 reference is degenerate (all ~0) -- test shape/init is uninformative");

    let cos = cosine(&logits_i8, &logits_fp32);
    let rel = rel_l2(&logits_i8, &logits_fp32);
    (cos, rel)
}

/// `Gpu::new` honours `BRAIN_DEVICE` when set and defaults to the wgpu
/// backend otherwise - this is the forward-EXECUTION parity gate on
/// whichever backend that resolves to (on this box: a real Intel Arc iGPU
/// whose Vulkan backend genuinely supports int8 DP4A, so every one of the 12
/// per-layer mixer/MLP linears is a REAL `Weight::I8`, not a silent fp32
/// demotion).
#[test]
fn int8_forward_tracks_fp32_within_quant_tolerance_default_backend() {
    let (cos, rel) = run_parity(Gpu::new(pipelines()), Gpu::new(pipelines()));
    eprintln!("qwen35 int8 vs fp32 (tiny_i8, default backend): cosine={cos:.9} rel_l2={rel:.9}");
    assert!(cos > 0.99, "qwen35 int8 path diverged too far from fp32: cosine={cos:.6} (want > 0.99)");
    assert!(rel < 0.1, "qwen35 int8 path diverged too far from fp32: rel_l2={rel:.4} (want < 0.1)");
}

/// Every one of the 12 per-layer mixer/MLP linears goes through
/// `model::ops::Weight::upload`, whose contract narrows the REQUESTED dtype
/// down to whatever the device can actually execute
/// (`want.promote(&ops.caps.numeric)`): on a backend whose caps don't
/// support the DP4A path (like this engine's CPU JIT), it silently builds
/// `Weight::F32` instead of `Weight::I8`. Unlike `qwen35moe` (which still has
/// a genuinely-quantized MoE-expert path on CPU, via a separate kernel that
/// IS CPU-JIT'able), `qwen35` has no such fallback path at all - EVERY
/// quantizable linear in this model lives on `self.ops`/`self.weights`, so
/// an "int8" CPU build is actually a COMPLETE fp32 demotion. The two outputs
/// should therefore be identical or extremely close (cosine ~= 1.0, not
/// merely greater than 0.99) - asserting a much tighter bound here than the
/// default-backend test makes that CPU-demotion behaviour an explicit,
/// checked fact rather than accidentally reusing a loose GPU-appropriate
/// tolerance that would silently pass even if CPU-demotion regressed (e.g.
/// if a future change made the CPU JIT support DP4A and this model started
/// actually quantizing there without anyone noticing this test's assumption
/// had changed).
#[test]
fn int8_forward_matches_fp32_almost_exactly_on_cpu_backend_full_demotion() {
    let (cos, rel) = run_parity(Gpu::new_cpu(pipelines()), Gpu::new_cpu(pipelines()));
    eprintln!("qwen35 int8 vs fp32 (tiny_i8, CPU backend, full fp32 demotion): cosine={cos:.9} rel_l2={rel:.9}");
    assert!(cos > 0.999999, "qwen35 CPU int8 build should be an almost-exact fp32 demotion: cosine={cos:.9} (want > 0.999999)");
    assert!(rel < 1e-4, "qwen35 CPU int8 build should be an almost-exact fp32 demotion: rel_l2={rel:.9} (want < 1e-4)");
}

/// The int8 build must exclude every `is_i8_linear` name from the fp32
/// `ParamStore` (no redundant fp32 copy uploaded) - `Qwen35::param_names`
/// only lists the fp32 store's own contents, so a quantized model's list
/// must be strictly SHORTER than the fp32 model's, by exactly the count of
/// quantizable leaves `tiny_i8()`'s own shape implies: 5 GDN leaves * 3 GDN
/// layers + 4 GQA leaves * 1 GQA layer + 3 MLP leaves * 4 layers (every
/// layer, both mixer types) = 15 + 4 + 12 = 31.
#[test]
fn int8_model_excludes_quantized_names_from_the_fp32_param_store() {
    let cfg = tiny_i8();
    let b = 1;
    let t = cfg.block_size;
    let init = init_weights(&cfg, 7);

    let fp32 = Qwen35::new_on(Gpu::new_cpu(pipelines()), cfg.clone(), b, t, &init);
    let i8 = Qwen35::new_on_i8(Gpu::new_cpu(pipelines()), cfg.clone(), b, t, &init);

    let fp32_names = fp32.param_names();
    let i8_names = i8.param_names();

    let types = cfg.layer_types();
    let n_gdn = types.iter().filter(|t| **t == LayerType::Linear).count();
    let n_gqa = types.iter().filter(|t| **t == LayerType::Full).count();
    assert_eq!(n_gdn + n_gqa, cfg.n_layers as usize);
    let expected_quantized = n_gdn * 5 + n_gqa * 4 + (cfg.n_layers as usize) * 3;

    assert!(expected_quantized > 0, "tiny_i8() must have at least one quantized linear to make this check meaningful");
    assert_eq!(i8_names.len(), fp32_names.len() - expected_quantized);

    const LEAVES: &[&str] = &[
        "in_proj_qkv.weight",
        "in_proj_z.weight",
        "in_proj_b.weight",
        "in_proj_a.weight",
        "out_proj.weight",
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate.weight",
        "up.weight",
        "down.weight",
    ];
    assert!(i8_names.iter().all(|n| !LEAVES.iter().any(|leaf| n.ends_with(leaf))), "int8 model's fp32 store must contain zero quantized names");
}

/// The concrete risk this milestone's `mtp.layers.0.*` upload guards against:
/// `Qwen35::ops_linear` panicking on a missing weight name for the MTP
/// head's own reused `layer_gqa_fwd`/`mlp_fwd` call sites, since those
/// weight names (`mtp.layers.0.self_attn.*`/`mtp.layers.0.mlp.*`) are
/// distinct from the main stack's `blocks.{l}.*` names and must be uploaded
/// into `self.weights` separately. Reuses [`run_parity`]'s own checks
/// (finite outputs, no panic) rather than folding this into the main
/// parity test, since the MTP-int8 interaction is a distinct concern worth
/// naming on its own.
#[test]
fn int8_forward_covers_the_mtp_head_when_mtp_is_enabled() {
    let cfg = Qwen35Config { mtp: true, ..tiny_i8() };
    let b = 1;
    let t = cfg.block_size;
    let init = init_weights(&cfg, 7);

    let fp32 = Qwen35::new_on(Gpu::new(pipelines()), cfg.clone(), b, t, &init);
    let i8 = Qwen35::new_on_i8(Gpu::new(pipelines()), cfg.clone(), b, t, &init);

    let tokens: Vec<u32> = (0..t).map(|i| (i * 3 + 1) % cfg.vocab).collect();
    let logits_fp32 = fp32.logits_all(&tokens);
    let logits_i8 = i8.logits_all(&tokens);

    assert!(logits_fp32.iter().all(|v| v.is_finite()), "fp32 (mtp) produced a non-finite logit");
    assert!(logits_i8.iter().all(|v| v.is_finite()), "int8 (mtp) produced a non-finite logit");

    let cos = cosine(&logits_i8, &logits_fp32);
    eprintln!("qwen35 int8 vs fp32 (tiny, mtp=true, default backend): cosine={cos:.9}");
    assert!(cos > 0.99, "qwen35 int8 path (mtp) diverged too far from fp32: cosine={cos:.6} (want > 0.99)");
}
