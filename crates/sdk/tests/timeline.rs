// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain::TimelineModel` end to end: train on a synthetic population, save,
//! load, and predict the same curves from the loaded model; predictions rank
//! the subjects the truth says are riskier above the others.
#![cfg(feature = "timeline")]

use brain::timeline::{synthetic, TimelineModel, TimelineSpec};

#[test]
fn train_save_load_predict() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (train, _) = synthetic::population(6000, 1);
    let (held_out, truth) = synthetic::population(1000, 2);
    let codes = ["death:a", "death:b", "onset"];
    let spec = TimelineSpec::new(codes, ["death:a", "death:b"])
        .knots(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0])
        .max_tokens(8)
        .steps(400)
        .batch(128)
        .lr(3e-3);
    let (model, report) = TimelineModel::train(&train, &held_out, &spec).unwrap();
    assert!(report.held_out_event_nll.is_finite());
    assert_eq!(report.truncated_tokens, 0);
    let dir = std::env::temp_dir().join(format!("brain-timeline-sdk-{}", std::process::id()));
    model.save(&dir).unwrap();
    let loaded = TimelineModel::load(&dir).unwrap();
    let (a, b) = (
        model.predict(&held_out[..50]).unwrap(),
        loaded.predict(&held_out[..50]).unwrap(),
    );
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(
            x.cif("death:a", 10.0),
            y.cif("death:a", 10.0),
            "the loaded model predicts the same curves"
        );
    }
    assert_eq!(a[0].cif("no-such-code", 1.0), None);
    assert!(a
        .iter()
        .all(|p| (0.0..=1.0).contains(&p.cif("onset", 10.0).unwrap())));
    // The subjects the truth puts in the riskiest fifth get more predicted risk.
    let all = model.predict(&held_out).unwrap();
    let mut order: Vec<usize> = (0..held_out.len()).collect();
    order.sort_by(|&i, &j| {
        truth[i]
            .cif(0, 10.0)
            .partial_cmp(&truth[j].cif(0, 10.0))
            .unwrap()
    });
    let fifth = held_out.len() / 5;
    let mean = |ix: &[usize]| {
        ix.iter()
            .map(|&i| all[i].cif("death:a", 10.0).unwrap())
            .sum::<f64>()
            / ix.len() as f64
    };
    let (low, high) = (mean(&order[..fifth]), mean(&order[order.len() - fifth..]));
    assert!(
        high > 2.0 * low,
        "riskiest fifth {high:.4} vs safest fifth {low:.4}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Both visit backbones save and load: the loaded model predicts the same
/// curves as the trained one.
#[test]
fn visit_backbones_save_and_load() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    use brain::timeline::synthetic::drifting::{self, Gaps};
    use brain::timeline::Backbone;
    let gaps = Gaps { last: (0.0, 2.0), between: (0.5, 2.0), visits: (1, 4) };
    let (train, _) = drifting::population(2000, 1, &gaps, 10.0);
    let (held_out, _) = drifting::population(300, 2, &gaps, 10.0);
    for backbone in [Backbone::State, Backbone::Attention] {
        let spec = TimelineSpec::new([drifting::CODE], [drifting::CODE])
            .knots(vec![0.0, 2.0, 5.0, 10.0])
            .max_tokens(8)
            .steps(60)
            .batch(64)
            .visits(4)
            .backbone(backbone);
        let (model, _) = TimelineModel::train(&train, &held_out, &spec).unwrap();
        let dir = std::env::temp_dir().join(format!("brain-timeline-visits-{backbone:?}-{}", std::process::id()));
        model.save(&dir).unwrap();
        let loaded = TimelineModel::load(&dir).unwrap();
        assert_eq!(loaded.config().visits, 4);
        assert_eq!(loaded.config().backbone, backbone);
        let (a, b) = (model.predict(&held_out[..40]).unwrap(), loaded.predict(&held_out[..40]).unwrap());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.cif(drifting::CODE, 5.0), y.cif(drifting::CODE, 5.0), "{backbone:?}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
