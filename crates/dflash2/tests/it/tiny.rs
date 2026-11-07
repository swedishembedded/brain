// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding for clients serving large models on modest hardware.
// If your team needs expertise in gating a ported drafter against its reference then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The device model against the host reference on a tiny random draft: three layers (two sliding with a window of 6,
//! one full NoPE), grouped queries, four conv taps, groups of eight, a block of five rows, a selector over four
//! candidates. The contexts (14 and 17 positions) are longer than the window, so the sliding layers read a strict
//! suffix and their rings wrap.
//!
//! Fixture: `python3 tools/goldens/dflash2_reference_forward.py tiny --out testdata/golden/dflash2/tiny`; without it
//! the tests skip by name.

use dflash2::{BlockJob, Dflash2, Dflash2Config, HfDraft};

use crate::common::{gather, logits, scaled_error, Golden};

const TOLERANCE: f32 = 1e-4;
const DIR: &str = "golden/dflash2/tiny";

struct Fixture {
    cfg: Dflash2Config,
    dir: String,
    gold: Golden,
}

fn fixture() -> Option<Fixture> {
    let dir = brain_testutil::testdata(DIR);
    if !std::path::Path::new(&dir).join("golden.safetensors").exists() {
        brain_testutil::skip(&format!("{dir} absent: run tools/goldens/dflash2_reference_forward.py tiny --out {dir}"));
        return None;
    }
    let cfg = Dflash2Config::from_json(&std::fs::read_to_string(format!("{dir}/draft/config.json")).unwrap()).unwrap();
    let gold = Golden::read(&format!("{dir}/golden.safetensors"));
    Some(Fixture { cfg, dir, gold })
}

fn load(fx: &Fixture, slots: u32, cap: u32) -> Dflash2 {
    let gpu = gpu_core::Gpu::open(None, &dflash2::pipelines(&fx.cfg));
    let src = HfDraft::open(std::path::Path::new(&format!("{}/draft", fx.dir))).unwrap();
    Dflash2::load(gpu, fx.cfg.clone(), Box::new(src), slots, cap).unwrap()
}

/// Compare a round's per-layer hidden states, its post-norm rows and the selector's path with the reference's.
fn check_round(fx: &Fixture, m: &Dflash2, prefix: &str, hidden_rows: &[f32], trace: &[Vec<f32>], job: usize, jobs: usize) {
    let d = fx.cfg.d_model as usize;
    let block = fx.cfg.block_size as usize;
    let per_job = block * d;
    for (l, layer) in trace.iter().enumerate() {
        let got = &layer[job * per_job..(job + 1) * per_job];
        let err = scaled_error(&format!("{prefix}layer {l}"), got, fx.gold.get(&format!("{prefix}layer.{l}")));
        assert!(err <= TOLERANCE, "{prefix}layer {l}: scaled error {err:.3e}");
    }
    let _ = jobs;
    let mask_rows = &hidden_rows[d..];
    let normed = m.normed(mask_rows);
    let err = scaled_error(&format!("{prefix}post-norm hidden"), &normed, fx.gold.get(&format!("{prefix}hidden")));
    assert!(err <= TOLERANCE, "{prefix}post-norm hidden: scaled error {err:.3e}");
    let head = fx.gold.get("head");
    let lg = logits(&normed, head, d);
    let err = scaled_error(&format!("{prefix}logits"), &lg, fx.gold.get(&format!("{prefix}logits")));
    assert!(err <= TOLERANCE, "{prefix}logits: scaled error {err:.3e}");
    let anchor = fx.gold.ids(&format!("{prefix}ids"))[0];
    let path = m.select(mask_rows, &lg, anchor).unwrap();
    assert_eq!(path, fx.gold.ids(&format!("{prefix}path")), "{prefix}selector path");
}

fn noise(fx: &Fixture, prefix: &str) -> Vec<f32> {
    gather(fx.gold.get("embed"), fx.cfg.d_model as usize, &fx.gold.ids(&format!("{prefix}ids")))
}

#[test]
fn one_round_matches_the_reference_layer_by_layer() {
    let Some(fx) = fixture() else { return };
    let m = load(&fx, 3, 32);
    let wide = fx.cfg.tap_width();
    let taps_a = fx.gold.get("a.taps");
    m.append_context(1, taps_a, 0).unwrap();
    let mut trace = Vec::new();
    let out = m.denoise_traced(&[BlockJob { slot: 1, pos: 14, noise: &noise(&fx, "a.") }], Some(&mut trace)).unwrap();
    assert_eq!(taps_a.len(), 14 * wide);
    check_round(&fx, &m, "a.", &out[0], &trace, 0, 1);
}

#[test]
fn context_fed_in_pieces_and_extended_after_a_denoise_matches_the_reference() {
    let Some(fx) = fixture() else { return };
    let m = load(&fx, 3, 32);
    let wide = fx.cfg.tap_width();
    // Round A, the context arriving in three pieces.
    let taps_a = fx.gold.get("a.taps");
    for (from, to) in [(0usize, 5usize), (5, 6), (6, 14)] {
        m.append_context(2, &taps_a[from * wide..to * wide], from as u32).unwrap();
    }
    let mut trace = Vec::new();
    let out = m.denoise_traced(&[BlockJob { slot: 2, pos: 14, noise: &noise(&fx, "a.") }], Some(&mut trace)).unwrap();
    check_round(&fx, &m, "a.", &out[0], &trace, 0, 1);
    // The target then commits three positions; the block rows round A wrote past position 14 are overwritten by the
    // real context and round B reads exactly what a fresh draft would.
    let taps_b = fx.gold.get("b.taps");
    m.append_context(2, &taps_b[14 * wide..], 14).unwrap();
    let mut trace = Vec::new();
    let out = m.denoise_traced(&[BlockJob { slot: 2, pos: 17, noise: &noise(&fx, "b.") }], Some(&mut trace)).unwrap();
    check_round(&fx, &m, "b.", &out[0], &trace, 0, 1);
}

#[test]
fn two_sequences_in_one_forward_are_each_what_they_are_alone() {
    let Some(fx) = fixture() else { return };
    let m = load(&fx, 3, 32);
    m.append_context(0, fx.gold.get("a.taps"), 0).unwrap();
    m.append_context(2, fx.gold.get("b.taps"), 0).unwrap();
    let (na, nb) = (noise(&fx, "a."), noise(&fx, "b."));
    let mut trace = Vec::new();
    let out = m.denoise_traced(&[BlockJob { slot: 0, pos: 14, noise: &na }, BlockJob { slot: 2, pos: 17, noise: &nb }], Some(&mut trace)).unwrap();
    check_round(&fx, &m, "a.", &out[0], &trace, 0, 2);
    check_round(&fx, &m, "b.", &out[1], &trace, 1, 2);
}

#[test]
fn a_draft_that_cannot_run_is_refused_by_name() {
    let Some(fx) = fixture() else { return };
    let m = load(&fx, 1, 20);
    assert!(m.append_context(1, fx.gold.get("a.taps"), 0).unwrap_err().contains("cache slot"));
    assert!(m.append_context(0, fx.gold.get("a.taps"), 10).unwrap_err().contains("slot holds"));
    let n = noise(&fx, "a.");
    assert!(m.denoise_blocks(&[BlockJob { slot: 0, pos: 0, noise: &n }, BlockJob { slot: 0, pos: 0, noise: &n }]).unwrap_err().contains("share a cache slot"));
}

#[test]
fn the_memory_plan_charges_the_weight_bytes_the_load_reports() {
    let Some(fx) = fixture() else { return };
    let src = HfDraft::open(std::path::Path::new(&format!("{}/draft", fx.dir))).unwrap();
    let planned = src.device_weight_bytes(&fx.cfg);
    let m = load(&fx, 1, 20);
    assert_eq!(planned, m.weight_bytes());
    assert_eq!(fx.cfg.cache_bytes(3, 32), (0..fx.cfg.n_layers).map(|l| 2 * 3 * u64::from(fx.cfg.cache_rows(l, 32)) * u64::from(fx.cfg.kv_dim()) * 4).sum::<u64>());
}
