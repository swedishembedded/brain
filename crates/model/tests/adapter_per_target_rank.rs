// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-target rank/alpha (M7): an `AdapterPlan` resolves a default plus
//! selector-scoped overrides into one `TargetHp` per site, and
//! `AdapterSet::build_planned`/`from_tensors_planned` actually build and
//! round-trip a heterogeneous adapter - not just resolve hyperparameters on
//! paper.

use data::rng::Lcg;
use model::adapter::plan::{parse_map, AdapterPlan, TargetHpPatch};
use model::adapter::select::TargetSelector;
use model::adapter::{AdapterKind, KeyStyle, LinearSite, TargetHp, TargetSpec};
use model::lora::LoraPair;
use std::collections::HashMap;

fn site(name: &str, out: usize, inn: usize) -> LinearSite {
    LinearSite { name: name.to_string(), leaf: "wq", layer: None, spec: TargetSpec::whole(out, inn), save_name: None }
}

#[test]
fn plan_resolves_a_default_plus_overrides_last_match_wins() {
    let sites = vec![site("blocks.0.wq", 8, 8), site("blocks.1.wq", 8, 8), site("blocks.2.ffn", 16, 8)];
    let plan = AdapterPlan {
        select: TargetSelector::All,
        default: TargetHp::new(8, 8.0),
        overrides: vec![
            (TargetSelector::Names(vec!["blocks.0.wq".to_string(), "blocks.1.wq".to_string()]), TargetHpPatch { rank: Some(16), ..Default::default() }),
            (TargetSelector::Names(vec!["blocks.1.wq".to_string()]), TargetHpPatch { rank: Some(32), ..Default::default() }),
        ],
    };
    let resolved = plan.resolve(&sites).expect("resolve");
    let ranks: HashMap<&str, usize> = resolved.iter().map(|(s, hp)| (s.name.as_str(), hp.rank)).collect();
    assert_eq!(ranks["blocks.0.wq"], 16);
    assert_eq!(ranks["blocks.1.wq"], 32, "later override must win");
    assert_eq!(ranks["blocks.2.ffn"], 8, "unmatched site keeps the default");
    assert_eq!(plan.max_rank(&sites).unwrap(), 32);
}

#[test]
fn plan_rejects_an_override_that_matches_nothing() {
    let sites = vec![site("blocks.0.wq", 8, 8)];
    let plan = AdapterPlan {
        select: TargetSelector::All,
        default: TargetHp::new(8, 8.0),
        overrides: vec![(TargetSelector::Names(vec!["blocks.99.wq".to_string()]), TargetHpPatch { rank: Some(16), ..Default::default() })],
    };
    let err = plan.resolve(&sites).expect_err("must reject a dead override");
    assert!(err.contains("override"), "{err}");
}

#[test]
fn parse_map_reads_pattern_equals_value_pairs_last_wins_on_overlap() {
    let parsed = parse_map("blocks.0.*=16,blocks.*.ffn=4").expect("parse");
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[1].1, 4.0);
    assert!(parse_map("garbage").is_err());
    assert!(parse_map("blocks.0.*=notanumber").is_err());
    assert_eq!(parse_map("").expect("empty is valid").len(), 0);
}

/// The actual capability this milestone is about: three targets at three
/// DIFFERENT ranks, built through one `AdapterPlan`, each preserving its own
/// scale through a save/load round trip - not merely resolved on paper.
#[test]
fn a_three_distinct_rank_adapter_round_trips_with_every_targets_scale_exact() {
    let sites = vec![site("layer.a", 6, 5), site("layer.b", 6, 5), site("layer.c", 6, 5)];
    let plan = AdapterPlan {
        select: TargetSelector::All,
        default: TargetHp::new(2, 4.0),
        overrides: vec![
            (TargetSelector::Names(vec!["layer.b".to_string()]), TargetHpPatch { rank: Some(3), alpha: Some(6.0), ..Default::default() }),
            (TargetSelector::Names(vec!["layer.c".to_string()]), TargetHpPatch { rank: Some(4), alpha: Some(4.0), ..Default::default() }),
        ],
    };
    let planned = plan.resolve(&sites).expect("resolve");
    assert_eq!(planned.iter().map(|(_, hp)| hp.rank).collect::<Vec<_>>(), vec![2, 3, 4]);

    let mut rng = Lcg::new(1);
    let mut init = || rng.signed() * 0.02;
    let built = model::adapter::AdapterSet::<LoraPair>::build_planned(planned.clone(), KeyStyle::Brain, &mut init);

    // Drive each pair's B off zero (deterministically, per-target) so the
    // round trip below is checking real data, not three no-ops.
    let mut set = built;
    for (i, (_, k)) in set.iter_mut().enumerate() {
        let spec = k.spec();
        let dw: Vec<f32> = (0..spec.out * spec.inn).map(|j| ((i * 7 + j) as f32).sin()).collect();
        k.proj_step(&dw, 0.1, 1);
    }

    let tensors = set.to_tensors();
    let src: HashMap<String, (Vec<usize>, Vec<f32>)> = tensors.into_iter().map(|(n, s, d)| (n, (s, d))).collect();

    // Confirm the shapes on disk really do differ per target - this is the
    // "shapes carry the truth" half of per-target rank.
    assert_eq!(src["layer.a.lora_a"].0, vec![2, 5]);
    assert_eq!(src["layer.b.lora_a"].0, vec![3, 5]);
    assert_eq!(src["layer.c.lora_a"].0, vec![4, 5]);

    let reloaded = model::adapter::AdapterSet::<LoraPair>::from_tensors_planned(planned, KeyStyle::Brain, &src).expect("reload");

    for ((site_a, ka), (site_b, kb)) in set.iter().zip(reloaded.iter()) {
        assert_eq!(site_a.name, site_b.name);
        let mut da = vec![0.0f32; 30];
        ka.delta_into(1.0, &mut da);
        let mut db = vec![0.0f32; 30];
        kb.delta_into(1.0, &mut db);
        for (x, y) in da.iter().zip(db.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "{}: reload did not reproduce the trained delta exactly", site_a.name);
        }
    }
}

/// A metadata rank that disagrees with the tensor's own shape must be
/// rejected, never silently trusted - shapes win.
#[test]
fn a_shape_disagreement_on_reload_is_a_hard_error() {
    let sites = vec![site("layer.a", 6, 5)];
    let hp = TargetHp::new(2, 4.0);
    let mut init = || 0.01f32;
    let set = model::adapter::AdapterSet::<LoraPair>::build(sites.clone(), hp, KeyStyle::Brain, &mut init);
    let mut src: HashMap<String, (Vec<usize>, Vec<f32>)> = set.to_tensors().into_iter().map(|(n, s, d)| (n, (s, d))).collect();
    // Corrupt the shape metadata for .lora_a to claim rank 3 while the data
    // (and the plan asking for rank 2) disagree.
    let (shape, data) = src.get_mut("layer.a.lora_a").unwrap();
    *shape = vec![3, 5];
    let _ = data;
    match model::adapter::AdapterSet::<LoraPair>::from_tensors(sites, hp, KeyStyle::Brain, &src) {
        Ok(_) => panic!("must reject the shape mismatch"),
        Err(e) => assert!(e.contains("layer.a"), "{e}"),
    }
}
