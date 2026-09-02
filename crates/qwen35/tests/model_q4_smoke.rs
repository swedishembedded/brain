// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P0 smoke + parity test for `qwen35::model::Qwen35`'s Q4 (W4A8) inference
//! tier (M24), the sibling of `model_i8_smoke.rs`. Two things this file
//! checks that the int8 smoke test does not need to: a per-leaf
//! [`TierPolicy`] is actually PLACED (not silently collapsed to uniform -
//! [`Qwen35::linear_dtype`] is the accessor that makes this checkable at
//! all), and a MIXED policy (Q4 on the MLP, F32 on the GDN decay/beta gates)
//! forwards without panicking and stays finite.
//!
//! Not a numerical-parity-against-HF test (no `torch`/`transformers` in this
//! environment) - same scope note as `model_i8_smoke.rs`.

use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::config::{LayerType, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

fn init_weights(cfg: &Qwen35Config, seed: u64) -> std::collections::HashMap<String, Vec<f32>> {
    qwen35::init::init_weights(cfg, seed)
}

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

/// Runs both an fp32 and a Q4 `Qwen35` forward at [`tiny_i8`] from the SAME
/// fresh init weights (Q4 shares I8's `k`-multiple-of-32 constraint, hence
/// reusing the int8 fixture rather than `tiny()`) and checks the Q4 path's
/// logits track fp32's within a generous quantization tolerance - Q4 halves
/// the weight levels I8 already has, so this floor is deliberately looser
/// than `model_i8_smoke.rs`'s.
#[test]
fn q4_forward_tracks_fp32_within_quant_tolerance_default_backend() {
    let cfg = tiny_i8();
    let b = 1;
    let t = cfg.block_size;
    let init = init_weights(&cfg, 7);

    let fp32 = Qwen35::new_on(Gpu::new(pipelines()), cfg.clone(), b, t, &init);
    let q4 = Qwen35::new_on_dt(Gpu::new(pipelines()), cfg.clone(), b, t, &init, &TierPolicy::uniform(Dtype::Q4));

    let tokens: Vec<u32> = (0..t).map(|i| (i * 3 + 1) % cfg.vocab).collect();
    let logits_fp32 = fp32.logits_all(&tokens);
    let logits_q4 = q4.logits_all(&tokens);

    assert_eq!(logits_fp32.len(), logits_q4.len());
    assert!(logits_fp32.iter().all(|v| v.is_finite()), "fp32 reference produced a non-finite logit");
    assert!(logits_q4.iter().all(|v| v.is_finite()), "q4 path produced a non-finite logit");

    let cos = cosine(&logits_q4, &logits_fp32);
    eprintln!("qwen35 q4 vs fp32 (tiny_i8, default backend): cosine={cos:.9}");
    assert!(cos > 0.9, "qwen35 q4 path diverged too far from fp32: cosine={cos:.6} (want > 0.9)");
}

/// The Q4 twin of `model_i8_smoke.rs`'s own CPU-demotion test: on a backend
/// without `int8_dot`, `Weight::upload`'s `want.promote(caps)` narrows a
/// requested Q4 tier all the way back to F32, so a "q4" CPU build is
/// actually a complete fp32 demotion and should be near bit-identical to a
/// real fp32 build, not merely close.
#[test]
fn q4_forward_matches_fp32_almost_exactly_on_cpu_backend_full_demotion() {
    let cfg = tiny_i8();
    let b = 1;
    let t = cfg.block_size;
    let init = init_weights(&cfg, 7);

    let fp32 = Qwen35::new_on(Gpu::new_cpu(pipelines()), cfg.clone(), b, t, &init);
    let q4 = Qwen35::new_on_dt(Gpu::new_cpu(pipelines()), cfg.clone(), b, t, &init, &TierPolicy::uniform(Dtype::Q4));

    let tokens: Vec<u32> = (0..t).map(|i| (i * 3 + 1) % cfg.vocab).collect();
    let logits_fp32 = fp32.logits_all(&tokens);
    let logits_q4 = q4.logits_all(&tokens);

    let cos = cosine(&logits_q4, &logits_fp32);
    eprintln!("qwen35 q4 vs fp32 (tiny_i8, CPU backend, full fp32 demotion): cosine={cos:.9}");
    assert!(cos > 0.999999, "qwen35 CPU q4 build should be an almost-exact fp32 demotion: cosine={cos:.9} (want > 0.999999)");
}

/// A MIXED policy (Q4 on the MLP, F32 on the two GDN state-sensitive gates -
/// M24's recommended "policy C") forwards without panicking, stays finite,
/// AND is genuinely placed per-leaf - not silently collapsed to uniform Q4.
/// [`Qwen35::linear_dtype`] is asked directly rather than inferred from the
/// output, so a bug that placed everything at Q4 anyway (ignoring the `with`
/// exception) would fail HERE, not be masked by a forward pass that still
/// happens to be finite.
#[test]
fn a_mixed_policy_is_placed_per_leaf_not_collapsed_to_uniform() {
    let cfg = tiny_i8();
    let b = 1;
    let t = cfg.block_size;
    let init = init_weights(&cfg, 7);
    let policy = TierPolicy::uniform(Dtype::Q4).with(&["in_proj_a.weight", "in_proj_b.weight"], Dtype::F32);

    let mixed = Qwen35::new_on_dt(Gpu::new(pipelines()), cfg.clone(), b, t, &init, &policy);

    let types = cfg.layer_types();
    let gdn_layer = types.iter().position(|t| *t == LayerType::Linear).expect("tiny_i8 must have a GDN layer");
    let gqa_layer = types.iter().position(|t| *t == LayerType::Full).expect("tiny_i8 must have a GQA layer");

    assert_eq!(
        mixed.linear_dtype(&format!("blocks.{gdn_layer}.linear_attn.in_proj_a.weight")),
        Some(Dtype::F32),
        "the exception must be placed at F32"
    );
    assert_eq!(
        mixed.linear_dtype(&format!("blocks.{gdn_layer}.linear_attn.in_proj_b.weight")),
        Some(Dtype::F32),
        "the exception must be placed at F32"
    );
    assert_eq!(
        mixed.linear_dtype(&format!("blocks.{gdn_layer}.mlp.gate.weight")),
        Some(Dtype::Q4),
        "the default must still land at Q4 for a leaf the exception does not name"
    );
    assert_eq!(
        mixed.linear_dtype(&format!("blocks.{gdn_layer}.linear_attn.in_proj_qkv.weight")),
        Some(Dtype::Q4),
        "in_proj_qkv is not named by the exception -- must stay at the policy default"
    );
    assert_eq!(mixed.linear_dtype(&format!("blocks.{gqa_layer}.self_attn.q_proj.weight")), Some(Dtype::Q4));
    assert_eq!(mixed.linear_dtype("nonexistent.leaf"), None, "a name this instance never uploaded must report None, not panic");

    let tokens: Vec<u32> = (0..t).map(|i| (i * 3 + 1) % cfg.vocab).collect();
    let logits = mixed.logits_all(&tokens);
    assert!(logits.iter().all(|v| v.is_finite()), "mixed-policy forward produced a non-finite logit");
}

/// [`Qwen35::linear_dtype`] on a plain fp32 build reports F32 for every
/// quantizable leaf and `None` for a norm/embedding name it never uploaded
/// as an `Ops` weight - the two halves of its own doc contract.
#[test]
fn linear_dtype_on_a_plain_fp32_build_reports_f32_for_every_quantizable_leaf() {
    let cfg = tiny_i8();
    let b = 1;
    let t = cfg.block_size;
    let init = init_weights(&cfg, 7);
    let fp32 = Qwen35::new_on(Gpu::new(pipelines()), cfg.clone(), b, t, &init);
    assert_eq!(fp32.linear_dtype("blocks.0.mlp.gate.weight"), Some(Dtype::F32));
    assert_eq!(fp32.linear_dtype("norm.weight"), None, "a plain norm is never an Ops weight -- it lives in the fp32 ParamStore");
}
