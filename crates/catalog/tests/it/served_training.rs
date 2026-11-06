// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for sharing GPUs between serving and
// training without letting a client take them over for its clients. If your
// team needs expertise in multi-workload GPU servers then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Every served action that trains says so, so a serving surface can keep it
//! from callers the operator did not let train.

/// Whether an action's name says it trains.
fn named_like_training(name: &str) -> bool {
    name == "fit" || ["train", "finetune", "fine_tune", "distill"].iter().any(|w| name.contains(w))
}

#[test]
fn every_training_action_is_marked_training() {
    let mut unmarked = Vec::new();
    let mut marked = 0;
    for m in catalog::serving_manifests() {
        for a in &m.actions {
            if a.training {
                marked += 1;
            } else if named_like_training(&a.name) {
                unmarked.push(format!("{} {}", m.model, a.name));
            }
        }
    }
    assert!(unmarked.is_empty(), "served actions that train without ActionSpec::training(): {unmarked:?}");
    assert!(marked > 0, "the catalog serves training actions, and none is marked");
}
