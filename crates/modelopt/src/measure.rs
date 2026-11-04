// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The measurement that accepts or rejects a plan: the whole model's loss on
//! held-out batches with the plan applied.

use model::ops::Weight;
use model::{Batch, Model, IGNORE};

use crate::{Assignment, Site};

/// How much a batch counts in a held-out mean: its supervised positions for a
/// language-model batch (the weighting the trainer's own evaluation uses), one
/// for any other kind.
fn weight(batch: &Batch) -> f64 {
    let supervised = |targets: &[u32]| targets.iter().filter(|&&t| t != IGNORE).count() as f64;
    match batch {
        Batch::Lm { targets, .. } | Batch::LmWeighted { targets, .. } | Batch::LmExamples { targets, .. } | Batch::Multimodal { targets, .. } => supervised(targets),
        Batch::Seq2Seq { labels, .. } => supervised(labels),
        Batch::Tensor { .. } => 1.0,
    }
}

/// The mean loss of `m` over `held_out`, one forward per batch, each batch
/// weighted by its supervised positions: the per-position loss the trainer's
/// evaluation reports, so a measured plan and a training curve are the same
/// quantity. Batches with nothing supervised do not count.
pub fn held_out_loss<M: Model>(m: &M, held_out: &[Batch]) -> f64 {
    let (mut sum, mut total) = (0f64, 0f64);
    for &batch in held_out {
        let w = weight(&batch);
        if w == 0.0 {
            continue;
        }
        m.set_batch(batch);
        sum += m.forward() as f64 * w;
        total += w;
    }
    assert!(total > 0.0, "held_out_loss needs at least one batch with a supervised position");
    sum / total
}

/// [`held_out_loss`] with every assigned site stored at its tier, simulated
/// with the deployed codec ([`Weight::round_trip`]). The original weights are
/// written back before this returns, including when the forward panics, so
/// measuring several plans in turn measures each against the same model.
///
/// This simulates the weights only. A tier whose kernels also quantize the
/// activation operand (int8's W8A8 products) is measured on its deployed
/// kernels by building the model at the plan ([`Plan::policy`](crate::Plan::policy)).
///
/// An assignment naming no site in `sites` is a caller error and panics with
/// its name: a plan and a manifest from different models must not measure.
pub fn measure_policy<M: Model>(m: &M, sites: &[Site], assignments: &[Assignment], held_out: &[Batch]) -> f64 {
    let mut restore = Restore { m, saved: Vec::new() };
    for a in assignments {
        let site = sites.iter().find(|s| s.name == a.site).unwrap_or_else(|| panic!("measure_policy: {} is not a site of this model", a.site));
        for t in &site.tensors {
            let original = m.read_weight(&t.name);
            m.write_weight(&t.name, &Weight::round_trip(&original, t.n, t.k, a.tier));
            restore.saved.push((t.name.clone(), original));
        }
    }
    held_out_loss(m, held_out)
}

/// Writes saved weights back when dropped.
struct Restore<'a, M: Model> {
    m: &'a M,
    saved: Vec<(String, Vec<f32>)>,
}

impl<M: Model> Drop for Restore<'_, M> {
    fn drop(&mut self) {
        for (name, w) in self.saved.drain(..).rev() {
            self.m.write_weight(&name, &w);
        }
    }
}
