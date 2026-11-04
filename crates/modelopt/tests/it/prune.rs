// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for shrinking trained models to
// the compute a deployment has, by removing what they need least, for its
// clients. If your team needs expertise in structured pruning and model
// compression then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Structured pruning, end to end on a trained model.
//!
//! * Keeping half of every layer's MLP channels by estimated importance loses
//!   less held-out loss than the median of keeping the same number at random,
//!   so the ranking, not the count, is what preserves the model.
//! * The exported model is genuinely narrower (`d_ff` and the slices it
//!   sizes) and computes what the zeroed measurement predicted.
//! * A width the dimensions cannot take is refused by name.

use model::Model;
use modelopt::{channel_importance, measure_pruned, plan_prune, pruned_weights, PrunePlan, PruneWidth};

use crate::common;
use common::{as_batches, batches, trained, ROWS};

#[test]
fn importance_pruning_keeps_what_the_model_needs_and_exports_a_narrower_model() {
    let mut rng = data::rng::Rng::random();
    let m = trained(&mut rng);
    let (vocab, seq, d_ff) = (m.config().vocab, m.config().block_size, m.config().d_ff as usize);
    let calibration = batches(&mut rng, 4, vocab, seq);
    let held_out = batches(&mut rng, 8, vocab, seq);
    let (calibration, held_out) = (as_batches(&calibration), as_batches(&held_out));

    let dims = Model::prunable_dims(&m);
    assert_eq!(dims.len(), m.config().n_layers as usize, "one MLP dimension per layer");
    let importance = channel_importance(&m, &dims, &calibration, &mut |_| {}).expect("importance");
    let width = d_ff / 2;
    let plan = plan_prune(&dims, &importance, PruneWidth::Channels(width)).expect("a plan");
    assert_eq!(plan.widths, [("d_ff", d_ff, width)]);
    assert!(plan.keep.iter().all(|k| k.len() == width && k.windows(2).all(|w| w[0] < w[1])));

    let pruned = measure_pruned(&m, &dims, &plan, &held_out).expect("measured");
    let original = modelopt::held_out_loss(&m, &held_out);
    assert_eq!(modelopt::held_out_loss(&m, &held_out), original, "measuring a plan restores every weight");
    let mut random: Vec<f64> = (0..7)
        .map(|_| {
            let keep = dims
                .iter()
                .map(|_| {
                    let mut all: Vec<usize> = (0..d_ff).collect();
                    for i in (1..all.len()).rev() {
                        all.swap(i, rng.gen_range_inclusive(0, i as i64) as usize);
                    }
                    let mut kept = all[..width].to_vec();
                    kept.sort_unstable();
                    kept
                })
                .collect();
            measure_pruned(&m, &dims, &PrunePlan { keep, ..plan.clone() }, &held_out).expect("measured")
        })
        .collect();
    random.sort_by(f64::total_cmp);
    assert!(pruned < random[3], "importance pruning {pruned} must beat the median {} of random channel choices ({random:?}); original {original}", random[3]);

    let narrow = pruned_weights(&m, &dims, &plan).expect("the pruned model");
    assert_eq!(narrow.config.d_ff as usize, width);
    assert_eq!(narrow.weights["blocks.0.mlp.gate.weight"].len(), width * m.config().d_model as usize);
    assert_eq!(narrow.weights["blocks.0.mlp.down.weight"].len(), m.config().d_model as usize * width);
    let rebuilt = qwen3::Qwen::new(narrow.config, ROWS, seq, &narrow.weights);
    let exported = modelopt::held_out_loss(&rebuilt, &held_out);
    assert!((exported - pruned).abs() <= 1e-4 * pruned.abs(), "the narrower model computes what the zeroed one did: {exported} vs {pruned}");
    eprintln!("original {original}, pruned to {width}/{d_ff}: {pruned}, random median {}", random[3]);

    assert!(plan_prune(&dims, &importance, PruneWidth::Channels(d_ff + 1)).unwrap_err().contains("cannot keep"));
    assert!(plan_prune(&dims, &importance, PruneWidth::Channels(0)).unwrap_err().contains("cannot keep"));
    assert!(plan_prune(&dims, &importance, PruneWidth::Fraction(1.5)).unwrap_err().contains("(0, 1]"));
}
