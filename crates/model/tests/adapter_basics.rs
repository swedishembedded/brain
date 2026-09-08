// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Phase-1 gate for the generic PEFT substrate: `LoraPair` must reproduce
//! `Pair` bit-for-bit, `TargetSpec` must express
//! both the whole-tensor and fused-slice cases `Pair::delta`/
//! `delta_strided` already do, and the dependency-free glob matcher must
//! match its own specification exactly.

use data::rng::Lcg;
use model::adapter::select::{glob_match, LayerRange, TargetSelector};
use model::adapter::{AdapterKind, LinearSite, TargetHp, TargetSpec};
use model::lora::{LoraPair, Pair};

#[test]
fn target_spec_whole_matches_pair_delta_offsets() {
    let whole = TargetSpec::whole(4, 3);
    assert_eq!(whole, TargetSpec { out: 4, inn: 3, row0: 0, row_stride: 3, col0: 0 });
    assert_eq!(whole.dest_len(), 12);

    let row = TargetSpec::row_slice(4, 3, 8, 9);
    assert_eq!(row, TargetSpec { out: 4, inn: 3, row0: 8, row_stride: 9, col0: 0 });
    assert_eq!(row.dest_len(), 12 * 9);

    let col = TargetSpec::col_slice(4, 3, 9, 5);
    assert_eq!(col, TargetSpec { out: 4, inn: 3, row0: 0, row_stride: 9, col0: 5 });
}

#[test]
fn pair_and_lora_pair_are_bit_identical_over_many_shapes_and_steps() {
    for seed in 0..50u64 {
        let mut dims_rng = Lcg::new(seed);
        let out = 1 + (dims_rng.next_u32() % 12) as usize;
        let inn = 1 + (dims_rng.next_u32() % 12) as usize;
        let r = 1 + (dims_rng.next_u32() % 4) as usize;

        let mut init_a = Lcg::new(seed ^ 0x9e37_79b9);
        let mut init_a_fn = || init_a.signed() * 0.02;
        let mut pair = Pair::new(out, inn, r, &mut init_a_fn);

        let mut init_b = Lcg::new(seed ^ 0x9e37_79b9);
        let mut init_b_fn = || init_b.signed() * 0.02;
        let hp = TargetHp::new(r, r as f32);
        let spec = TargetSpec::whole(out, inn);
        let mut lp = LoraPair::new(spec, hp, &mut init_b_fn);

        let mut step_rng = Lcg::new(seed ^ 0xdead_beef);
        for t in 1..=20u64 {
            let dw: Vec<f32> = (0..out * inn).map(|_| step_rng.signed()).collect();

            let (da, db) = pair.project(&dw, hp.scale());
            pair.adam_step(&da, &db, 0.01, t);

            let g = lp.project(&dw);
            lp.step(&g, 0.01, t);
        }

        let mut w_pair = vec![0.0f32; out * inn];
        pair.delta(hp.scale(), &mut w_pair);
        let mut w_lora = vec![0.0f32; out * inn];
        lp.delta_into(1.0, &mut w_lora);

        for (i, (a, b)) in w_pair.iter().zip(w_lora.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "seed {seed} out {out} inn {inn} r {r}: index {i} diverged");
        }

        // Round-trip through to_tensors/load_tensors must also be exact.
        let tensors = lp.to_tensors();
        let get = |suffix: &str| -> Option<(Vec<usize>, Vec<f32>)> {
            tensors.iter().find(|(s, _, _)| *s == suffix).map(|(_, shape, data)| (shape.clone(), data.clone()))
        };
        let mut reloaded = LoraPair::new(spec, hp, &mut || 0.0);
        reloaded.load_tensors(&get).expect("load_tensors");
        let mut w_reloaded = vec![0.0f32; out * inn];
        reloaded.delta_into(1.0, &mut w_reloaded);
        for (a, b) in w_lora.iter().zip(w_reloaded.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

#[test]
fn lora_plus_ratio_one_is_bit_identical_to_plain_lora_step() {
    let mut init = Lcg::new(1);
    let mut init_fn = || init.signed() * 0.02;
    let spec = TargetSpec::whole(6, 5);
    let hp_plain = TargetHp::new(2, 2.0);
    let mut plain = LoraPair::new(spec, hp_plain, &mut init_fn);

    let mut init2 = Lcg::new(1);
    let mut init2_fn = || init2.signed() * 0.02;
    let mut hp_ratio = TargetHp::new(2, 2.0);
    hp_ratio.lr_ratio = 1.0;
    let mut ratioed = LoraPair::new(spec, hp_ratio, &mut init2_fn);

    let mut rng = Lcg::new(2);
    for t in 1..=10u64 {
        let dw: Vec<f32> = (0..30).map(|_| rng.signed()).collect();
        let g1 = plain.project(&dw);
        plain.step(&g1, 0.05, t);
        let g2 = ratioed.project(&dw);
        ratioed.step(&g2, 0.05, t);
    }

    let mut w1 = vec![0.0f32; 30];
    plain.delta_into(1.0, &mut w1);
    let mut w2 = vec![0.0f32; 30];
    ratioed.delta_into(1.0, &mut w2);
    for (a, b) in w1.iter().zip(w2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn glob_matcher_handles_literal_star_double_star_question_and_braces() {
    assert!(glob_match("blocks.3.attn.q", "blocks.3.attn.q"));
    assert!(!glob_match("blocks.3.attn.q", "blocks.3.attn.k"));

    assert!(glob_match("blocks.*.attn.q", "blocks.3.attn.q"));
    assert!(!glob_match("blocks.*.attn.q", "blocks.3.deep.attn.q"), "single * must not cross a '.'");

    assert!(glob_match("**.attn.q", "blocks.3.deep.attn.q"));
    assert!(glob_match("x**y", "xy"), "** may also match zero characters");
    assert!(glob_match("x**y", "x.a.b.y"), "** crosses '.' unlike a lone *");

    assert!(glob_match("blocks.?.attn.q", "blocks.3.attn.q"));
    assert!(!glob_match("blocks.?.attn.q", "blocks.30.attn.q"));

    assert!(glob_match("blocks.{0,1,2}.attn.q", "blocks.1.attn.q"));
    assert!(!glob_match("blocks.{0,1,2}.attn.q", "blocks.3.attn.q"));
}

#[test]
fn target_selector_layers_filters_by_layer_field_not_name_substring() {
    let site = |name: &'static str, layer: usize| LinearSite {
        name: name.to_string(),
        leaf: "wq",
        layer: Some(layer),
        spec: TargetSpec::whole(4, 4),
        save_name: None,
    };
    let sel = TargetSelector::Layers {
        inner: Box::new(TargetSelector::Leaves(vec!["wq".to_string()])),
        range: LayerRange { first: Some(2), last: Some(4), every: 1 },
    };
    assert!(!sel.matches(&site("blocks.1.wq", 1)));
    assert!(sel.matches(&site("blocks.2.wq", 2)));
    assert!(sel.matches(&site("blocks.4.wq", 4)));
    assert!(!sel.matches(&site("blocks.5.wq", 5)));
    // A name containing a layer-shaped substring outside the `layer` field
    // must not confuse the selector - it never parses `name`.
    assert!(!sel.matches(&LinearSite { name: "blocks.2.extra.4.wq".to_string(), leaf: "wq", layer: Some(9), spec: TargetSpec::whole(4, 4), save_name: None }));
}

#[test]
fn target_selector_unknown_kind_of_query_returns_false_not_panic() {
    let sel = TargetSelector::Not(Box::new(TargetSelector::All));
    let site = LinearSite { name: "x".to_string(), leaf: "x", layer: None, spec: TargetSpec::whole(1, 1), save_name: None };
    assert!(!sel.matches(&site));
}
