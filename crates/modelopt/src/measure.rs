// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The measurement that accepts or rejects a plan: the whole model's loss on
//! held-out batches with the plan applied.

use model::ops::Weight;
use model::{Batch, Model};

use crate::{Assignment, Site};

/// The mean loss of `m` over `held_out`, one forward per batch: the model's
/// own training objective, so a measured plan and a training curve are the
/// same quantity.
pub fn held_out_loss<M: Model>(m: &M, held_out: &[Batch]) -> f64 {
    assert!(!held_out.is_empty(), "held_out_loss needs at least one batch");
    held_out
        .iter()
        .map(|&batch| {
            m.set_batch(batch);
            m.forward() as f64
        })
        .sum::<f64>()
        / held_out.len() as f64
}

/// [`held_out_loss`] with every assigned site stored at its tier, simulated
/// with the deployed codec ([`Weight::round_trip`]). The original weights are
/// written back before this returns, including when the forward panics, so
/// measuring several plans in turn measures each against the same model.
///
/// An assignment naming no site in `sites` is a caller error and panics with
/// its name: a plan and a manifest from different models must not measure.
pub fn measure_policy<M: Model>(m: &M, sites: &[Site], assignments: &[Assignment], held_out: &[Batch]) -> f64 {
    let mut restore = Restore { m, saved: Vec::with_capacity(assignments.len()) };
    for a in assignments {
        let site = sites.iter().find(|s| s.name == a.site).unwrap_or_else(|| panic!("measure_policy: {} is not a site of this model", a.site));
        let original = m.read_weight(&site.name);
        m.write_weight(&site.name, &Weight::round_trip(&original, site.n, site.k, a.tier));
        restore.saved.push((site.name.clone(), original));
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
