// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The real-weight half of the Qwen3 GGUF name-map gate.
//!
//! `crates/qwen3/src/gguf_import.rs`'s own tests prove the map **bit for bit**,
//! but only at a tiny synthetic shape where both containers hold F32. That is
//! the right place to assert bit-identity and the wrong place to find out that
//! a real llama.cpp release spells something differently, ships a tensor the
//! map has never seen, or declares hyperparameters the config extractor reads
//! wrong. This runs the same comparison against the actual published
//! checkpoints - `Qwen/Qwen3-8B` (bf16 safetensors) versus
//! `Qwen/Qwen3-8B-GGUF`'s `Qwen3-8B-Q8_0.gguf` - where the two are the same
//! model but NOT the same bytes.
//!
//! So this rung is **parity-gated, not bit-identical, and says so**: Q8_0 is a
//! lossy 8-bit tier with a per-32-element scale, so a Q8_0 tensor cannot equal
//! its bf16 original and any test asserting it did would be asserting the
//! wrong thing. Both **cosine and relative L2** are checked - cosine alone is
//! scale-invariant and would pass a tensor that had been uniformly rescaled,
//! which is exactly the kind of damage a wrong dequantization does.
//!
//! What this catches that the synthetic gate cannot: every name the real file
//! actually contains. A swapped `k`/`v` (identical shapes on every GQA layer,
//! so no shape check can see it) lands here as a cosine near zero, not near
//! one - the floors below are set to catch a WRONG TENSOR, and are nowhere
//! near tight enough to be a quantization-quality claim.
//!
//! Set `BRAIN_QWEN3_HF_DIR` and `BRAIN_QWEN3_GGUF`; skips loudly otherwise.

use checkpoint::weightio::WeightReader;
use checkpoint::TensorSource;
use qwen3::import::Naming;

/// Cosine and relative L2 of `a` against reference `b`.
fn agreement(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut diff) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        diff += (x - y) * (x - y);
    }
    let cos = if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 1.0 };
    let rel = if nb > 0.0 { diff.sqrt() / nb.sqrt() } else { diff.sqrt() };
    (cos, rel)
}

#[test]
fn the_real_q8_gguf_and_the_real_bf16_safetensors_are_the_same_model() {
    let (Ok(hf_dir), Ok(gguf)) = (std::env::var("BRAIN_QWEN3_HF_DIR"), std::env::var("BRAIN_QWEN3_GGUF")) else {
        brain_testutil::skip("set BRAIN_QWEN3_HF_DIR (an HF Qwen3 dir) and BRAIN_QWEN3_GGUF (the matching .gguf) to run this");
        return;
    };

    // Both opened through the SAME entry point, given two different kinds of
    // path - the property `BRAIN_FLUX2_TE` rides on.
    let r_hf = WeightReader::open_hf_dir(std::path::Path::new(&hf_dir)).unwrap();
    let r_gg = WeightReader::open_hf_dir(std::path::Path::new(&gguf)).unwrap();
    assert_eq!(Naming::of(&r_hf), Naming::Hf, "{hf_dir} does not look like an HF checkpoint");
    assert_eq!(Naming::of(&r_gg), Naming::Gguf, "{gguf} does not look like a GGUF");

    // First: the two sources must describe the SAME MODEL. A config
    // disagreement here would make every weight comparison below meaningless,
    // and is also the cheapest way to notice a mismatched pair of files.
    let hf_cfg = qwen3::import::config_from_hf(&std::fs::read_to_string(std::path::Path::new(&hf_dir).join("config.json")).unwrap()).unwrap();
    let gg_cfg = qwen3::gguf_import::config_from_gguf(&checkpoint::gguf::MmapGguf::open(&gguf).unwrap()).unwrap();
    assert_eq!(gg_cfg.to_json(), hf_cfg.to_json(), "the GGUF and the HF checkpoint describe different models");
    eprintln!(
        "qwen3 real parity: {} layers, d_model {}, GQA {}/{}, head_dim {}, d_ff {}, tied {}",
        hf_cfg.n_layers, hf_cfg.d_model, hf_cfg.n_heads, hf_cfg.n_kv_heads, hf_cfg.head_dim, hf_cfg.d_ff, hf_cfg.tie_embeddings
    );

    let whole = qwen3::Shard::whole(hf_cfg.n_layers as usize);
    // Two-way coverage runs inside both of these, before a byte is read: every
    // brain parameter resolved exactly once, every element count checked.
    let s_hf = qwen3::import::shard_source(&r_hf, &hf_cfg, &whole).expect("hf coverage");
    let s_gg = qwen3::import::shard_source(&r_gg, &gg_cfg, &whole).expect("gguf coverage");

    // Q8_0 against bf16. These floors are set to catch a WRONG TENSOR (a
    // swapped projection reads back near cosine 0), NOT to certify the
    // quantizer - the measured values are printed, and are far tighter.
    const MIN_COS: f64 = 0.999;
    const MAX_REL_L2: f64 = 0.05;

    let params = hf_cfg.param_list();
    let (mut worst_cos, mut worst_rel) = (1.0f64, 0.0f64);
    let (mut worst_cos_name, mut worst_rel_name) = (String::new(), String::new());
    for (name, numel) in &params {
        // One tensor at a time from each source, both dropped before the next:
        // peak host stays two tensors, not two models.
        let mut a: Option<Vec<f32>> = None;
        let mut b: Option<Vec<f32>> = None;
        assert!(s_gg.with_tensor(name, &mut |d| a = Some(d.to_vec())), "gguf missing {name}");
        assert!(s_hf.with_tensor(name, &mut |d| b = Some(d.to_vec())), "hf missing {name}");
        let (a, b) = (a.unwrap(), b.unwrap());
        assert_eq!(a.len(), *numel, "{name}: gguf element count");
        assert_eq!(b.len(), *numel, "{name}: hf element count");

        let (cos, rel) = agreement(&a, &b);
        assert!(cos.is_finite() && rel.is_finite(), "{name}: non-finite weights (cos {cos}, rel_l2 {rel})");
        assert!(cos >= MIN_COS, "{name}: cosine {cos:.10} < {MIN_COS} - this is a WRONG TENSOR, not a quantization loss");
        assert!(rel <= MAX_REL_L2, "{name}: rel_l2 {rel:.6} > {MAX_REL_L2}");
        if cos < worst_cos {
            worst_cos = cos;
            worst_cos_name = name.clone();
        }
        if rel > worst_rel {
            worst_rel = rel;
            worst_rel_name = name.clone();
        }
    }
    eprintln!(
        "qwen3 real parity: {} parameters compared; worst cosine {worst_cos:.10} ({worst_cos_name}), worst rel_l2 {worst_rel:.6} ({worst_rel_name})",
        params.len()
    );
}
