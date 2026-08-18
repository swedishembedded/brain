// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity of the TRAINING forward against the reference Kronos, plus the
//! structural property that parity is really about.
//!
//! The inference ladder in `parity.rs` cannot reach this. The reference
//! dependency layer attends with `is_causal=self.training`, and every inference
//! entry point (`decode_s1`/`decode_s2`) runs with that flag off *and* with a
//! single query row, so no rung driven through them can see the mask at all.
//! Training is the only regime where the flag is on and `q_len == t` - the only
//! regime where the mask exists. A trainer can therefore attend to the future
//! while every inference rung stays green, and a self-consistency gradient check
//! cannot notice either: it proves the backward matches the forward, not that
//! the forward is the right function. brain's trainer did exactly that.
//!
//! Two gates, because they prove different things:
//!
//! - **TR-mask** (no fixture, always runs): the dependency layer's attention
//!   probabilities are lower-triangular and each row still sums to 1. This is
//!   the property itself - no position conditions on information it will not
//!   have at inference - asserted directly rather than inferred from a distance.
//! - **TR-fwd** (golden-gated): brain's training-mode `s1_logits`, `s2_logits`
//!   and the `(CE_s1 + CE_s2)/2` objective against the reference's, on identical
//!   weights and identical inputs. Training is not bit-comparable to *inference*,
//!   but it is exactly comparable to the reference's own training forward, which
//!   is what this loads. Regenerate with
//!   `tools/goldens/kronos_train_dump_reference.py`.
//!
//! Without the mask the reference's own s2 logits move by ~1.5e-1 relative on
//! this dump and flip 4 of 24 argmaxes, so TR-fwd fails loudly on a non-causal
//! trainer rather than passing on a hair.

use kronos::config::KronosConfig;
use kronos::train::{param_list_c, KronosTrain, CAL};
use std::collections::HashMap;
use std::path::Path;

use brain_testutil::testdata_path;

/// Relative L2 `‖a−b‖ / ‖b‖`, the same measure the inference ladder reports.
fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>().sqrt();
    let den: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    num / (den + 1e-30)
}
fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn read_u32(p: &Path) -> Vec<u32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
/// Same band as the inference ladder: fp32 reassociation across two independent
/// implementations costs ~1e-5 here.
const TOL: f64 = 1e-3;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn sym(&mut self, a: f32) -> f32 {
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        (u * 2.0 - 1.0) * a
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
}

fn skip_gpu() -> bool {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS is set");
        return true;
    }
    false
}

/// TR-mask - the property, asserted structurally and with no fixture at all:
/// in training the dependency layer's row `i` may put weight only on keys
/// `j <= i`. The reference forbids the rest (`is_causal=self.training`), and a
/// trainer that allows them teaches the model to use context it will never have
/// when it is asked to predict.
#[test]
fn dependency_layer_attention_is_causal_in_training() {
    if skip_gpu() {
        return;
    }
    let cfg = KronosConfig::tiny();
    let t = 12u32;
    let mut r = Rng(0xC0FFEE);
    let d = cfg.d_model;
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, numel) in param_list_c(&cfg) {
        if name == "embedding.fusion_l" || name == "embedding.fusion_r" {
            continue;
        }
        let is_norm = name.ends_with("norm.weight") || name.ends_with("norm1.weight") || name.ends_with("norm2.weight");
        init.insert(name, (0..numel).map(|_| if is_norm { 1.0 + r.sym(0.02) } else { r.sym(0.25) }).collect());
    }
    init.insert("embedding.fusion_proj.weight".into(), (0..d * 2 * d).map(|_| r.sym(0.25)).collect());

    let m = KronosTrain::new(cfg.clone(), t, &init);
    let (s1v, s2v) = (cfg.s1_vocab() as u32, cfg.s2_vocab() as u32);
    let mut ids = |n: u32| -> Vec<u32> { (0..t).map(|_| r.below(n)).collect() };
    let cal: [Vec<u32>; 5] = std::array::from_fn(|c| ids(CAL[c].1 as u32));
    let (s1, s2, samp, tg1, tg2) = (ids(s1v), ids(s2v), ids(s1v), ids(s1v), ids(s2v));
    let calr: [&[u32]; 5] = [&cal[0], &cal[1], &cal[2], &cal[3], &cal[4]];
    m.set_batch(&s1, &s2, &calr, &samp, &tg1, &tg2);
    m.forward();

    let probs = m.dep_probs();
    let tt = t as usize;
    let mut worst_future = 0.0f32;
    let mut worst_row_err = 0.0f64;
    for h in 0..cfg.dep_n_heads {
        for i in 0..tt {
            let row = &probs[(h * tt + i) * tt..(h * tt + i) * tt + tt];
            for (j, &p) in row.iter().enumerate() {
                if j > i {
                    worst_future = worst_future.max(p.abs());
                }
            }
            let sum: f64 = row.iter().map(|&x| x as f64).sum();
            worst_row_err = worst_row_err.max((sum - 1.0).abs());
        }
    }
    eprintln!("TR-mask: max weight on a future key {worst_future:e}, max |row sum - 1| {worst_row_err:e}");
    assert_eq!(worst_future, 0.0, "TR-mask: the dependency layer attended to a FUTURE position in training");
    assert!(worst_row_err < 1e-5, "TR-mask: attention rows must still be a distribution (err {worst_row_err:e})");
}

/// TR-fwd - brain's whole training forward against the reference's, on the
/// weights and inputs the reference itself was run on.
#[test]
fn training_forward_matches_the_reference() {
    if skip_gpu() {
        return;
    }
    let golden = testdata_path("golden/kronos");
    const DUMPER: &str = "tools/goldens/kronos_train_dump_reference.py";
    let meta_p = golden.join("tr_meta.json");
    if !meta_p.exists() {
        return brain_testutil::skip(&format!("training golden missing; run {DUMPER}"));
    }
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(&meta_p).unwrap()).unwrap();
    let u = |k: &str| meta[k].as_u64().unwrap() as usize;
    let cfg = KronosConfig {
        d_model: u("d_model"),
        n_layers: u("n_layers"),
        n_heads: u("n_heads"),
        ff_dim: u("ff_dim"),
        s1_bits: u("s1_bits"),
        s2_bits: u("s2_bits"),
        learn_te: meta["learn_te"].as_bool().unwrap(),
        dep_n_heads: u("dep_n_heads"),
        max_context: u("seq_len"),
    };
    let t = u("seq_len") as u32;
    // The pairing check the inference ladder runs too: this dump carries its own
    // weights, so what has to agree is the architecture those weights are for.
    let Some(src) = brain_testutil::golden::Source::open_manifest(&meta_p, DUMPER) else {
        return;
    };
    if !src.require(&[("d_model", cfg.d_model as i64), ("n_layers", cfg.n_layers as i64), ("seq_len", t as i64)]) {
        return;
    }

    // Rebuild the reference state_dict from the flat dump (skipping the
    // non-persistent RoPE buffers, exactly as the checkpoint importer does).
    let blob = read_f32(&golden.join("tr_weights.f32"));
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    let mut off = 0usize;
    for p in meta["params"].as_array().unwrap() {
        let name = p[0].as_str().unwrap().to_string();
        let n = p[1].as_u64().unwrap() as usize;
        assert!(off + n <= blob.len(), "TR: tr_weights.f32 is short of what tr_meta.json declares");
        if !name.contains("rotary") && !name.contains("inv_freq") {
            w.insert(name, blob[off..off + n].to_vec());
        }
        off += n;
    }
    assert_eq!(off, blob.len(), "TR: tr_weights.f32 does not match tr_meta.json's tensor list");

    let s1 = read_u32(&golden.join("tr_s1_ids.u32"));
    let s2 = read_u32(&golden.join("tr_s2_ids.u32"));
    let samp = read_u32(&golden.join("tr_samp_s1.u32"));
    let s1_tgt = read_u32(&golden.join("tr_s1_targets.u32"));
    let s2_tgt = read_u32(&golden.join("tr_s2_targets.u32"));
    // The dump stores the stamp row-major [t, 5]; the trainer takes 5 columns.
    let stamp = read_u32(&golden.join("tr_stamp.u32"));
    let cal: [Vec<u32>; 5] = std::array::from_fn(|c| (0..t as usize).map(|i| stamp[i * 5 + c]).collect());
    let calr: [&[u32]; 5] = [&cal[0], &cal[1], &cal[2], &cal[3], &cal[4]];

    let m = KronosTrain::new(cfg.clone(), t, &w);
    m.set_batch(&s1, &s2, &calr, &samp, &s1_tgt, &s2_tgt);
    let loss = m.forward();

    let ref_s1 = read_f32(&golden.join("tr_s1_logits.f32"));
    let ref_s2 = read_f32(&golden.join("tr_s2_logits.f32"));
    let got_s1 = m.s1_logits();
    let got_s2 = m.s2_logits();
    let r1 = rel_l2(&got_s1, &ref_s1);
    let r2 = rel_l2(&got_s2, &ref_s2);
    let ref_loss = meta["ce_loss"].as_f64().unwrap();
    eprintln!("TR s1_logits: rel_l2={r1:.3e}");
    eprintln!("TR s2_logits: rel_l2={r2:.3e}  (the dependency layer, causal in training)");
    eprintln!("TR loss:      brain={loss:.6} ref={ref_loss:.6}");
    assert_eq!(got_s1.len(), ref_s1.len(), "TR: s1 logit field shape");
    assert_eq!(got_s2.len(), ref_s2.len(), "TR: s2 logit field shape");
    assert!(r1 < TOL, "TR: training-mode s1 logits diverge (rel {r1:.3e})");
    assert!(r2 < TOL, "TR: training-mode s2 logits diverge (rel {r2:.3e})");
    let dl = (loss as f64 - ref_loss).abs() / ref_loss.abs();
    assert!(dl < TOL, "TR: the training objective diverges (brain {loss} vs ref {ref_loss}, rel {dl:.3e})");
}
