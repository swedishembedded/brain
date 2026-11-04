// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The weight precision search as one call: analyze, plan, measure against
//! the baselines, report.

use gpu_core::select::Dtype;
use model::ops::Weight;
use model::{Batch, Model};
use serde::Serialize;

use crate::{assign, held_out_loss, measure_policy, sensitivity, sites, Assignment, Site};

/// How many bytes the searched linears may occupy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Budget {
    /// Physical bytes over every searched tensor (packed words plus scales).
    Bytes(u64),
    /// Mean stored bits per searched weight element, scales included: `8.5`
    /// is uniform group-scaled int8, `32` is f32.
    BitsPerWeight(f64),
}

impl Budget {
    /// The budget in bytes for `sites`.
    pub fn bytes(self, sites: &[Site]) -> Result<u64, String> {
        match self {
            Budget::Bytes(b) => Ok(b),
            Budget::BitsPerWeight(bits) if bits.is_finite() && bits > 0.0 => Ok((bits * sites.iter().map(Site::numel).sum::<usize>() as f64 / 8.0) as u64),
            Budget::BitsPerWeight(bits) => Err(format!("a bits-per-weight budget must be finite and positive, got {bits}")),
        }
    }
}

/// What to search: the candidate tiers, the budget, and two disjoint batch
/// sets, one the estimate ranks on and one that measures the result.
pub struct SearchRequest<'a, 'b> {
    pub tiers: &'a [Dtype],
    pub budget: Budget,
    pub calibration: &'a [Batch<'b>],
    pub held_out: &'a [Batch<'b>],
}

/// One site of the chosen plan.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SiteReport {
    pub site: String,
    pub tensors: Vec<String>,
    pub layer: Option<usize>,
    pub roles: Vec<&'static str>,
    pub elements: usize,
    pub tier: &'static str,
    pub bytes: u64,
    /// The estimated loss increase at the chosen tier.
    pub estimated_error: f64,
}

/// A uniform tier the plan is compared against: every searched site at it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Baseline {
    pub tier: &'static str,
    pub bytes: u64,
    pub loss: f64,
}

/// What a weight precision search decided and measured. Losses are the
/// supervised-position mean on the held-out batches with weights simulated by
/// the deployed codec; [`crate::measure_policy`] says what that covers.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Report {
    /// The plan as a `--weight-format` value.
    pub policy: String,
    pub sites: Vec<SiteReport>,
    pub budget_bytes: u64,
    pub plan_bytes: u64,
    /// The searched linears at f32, for scale.
    pub f32_bytes: u64,
    pub bits_per_weight: f64,
    pub estimated_error: f64,
    pub original_loss: f64,
    pub plan_loss: f64,
    /// Every candidate tier that fits the budget uniformly.
    pub baselines: Vec<Baseline>,
    pub calibration_batches: usize,
    pub held_out_batches: usize,
}

impl Report {
    /// The best uniform tier that fits the budget, if any does.
    pub fn best_baseline(&self) -> Option<&Baseline> {
        self.baselines.iter().min_by(|a, b| a.loss.total_cmp(&b.loss))
    }

    /// Whether the plan measured better than every uniform tier that fits its
    /// budget: what a mixed plan has to show to be worth deploying over one.
    pub fn beats_baselines(&self) -> bool {
        self.best_baseline().is_none_or(|b| self.plan_loss < b.loss)
    }
}

/// Search the storage tier of every quantizable linear of `m` within the
/// budget: estimate on the calibration batches ([`sensitivity`]), solve
/// ([`assign`]), then measure the plan, the original and each uniform tier that
/// fits the budget on the held-out batches. The model's weights are unchanged
/// when this returns.
///
/// The calibration and held-out sets must not be the same batches: a search
/// that is accepted on the data it ranked on has not been checked.
pub fn search_weights<M: Model>(m: &M, req: &SearchRequest) -> Result<Report, String> {
    if req.tiers.is_empty() {
        return Err("no candidate tiers to search".into());
    }
    if req.held_out.is_empty() {
        return Err("no held-out batches to measure the plan on".into());
    }
    if req.calibration.iter().any(|c| req.held_out.iter().any(|h| same_batch(c, h))) {
        return Err("a calibration batch is also a held-out batch; the set that accepts a plan must not be the set it was ranked on".into());
    }
    let sites = sites(m)?;
    let budget_bytes = req.budget.bytes(&sites)?;
    let table = sensitivity(m, &sites, req.tiers, req.calibration)?;
    let plan = assign(&table, budget_bytes)?;
    let original_loss = held_out_loss(m, req.held_out);
    let plan_loss = measure_policy(m, &sites, &plan.assignments, req.held_out);
    let bytes_at = |tier: Dtype| -> Option<u64> { sites.iter().flat_map(|s| &s.tensors).map(|t| Weight::packs(tier, t.n, t.k).then(|| Weight::bytes_for(tier, t.n, t.k))).sum() };
    let baselines = req
        .tiers
        .iter()
        .filter_map(|&tier| bytes_at(tier).filter(|&b| b <= budget_bytes).map(|bytes| (tier, bytes)))
        .map(|(tier, bytes)| {
            let uniform: Vec<Assignment> = sites.iter().map(|s| Assignment { site: s.name.clone(), tier }).collect();
            Baseline { tier: tier.name(), bytes, loss: measure_policy(m, &sites, &uniform, req.held_out) }
        })
        .collect();
    let elements: usize = sites.iter().map(Site::numel).sum();
    let site_reports = table
        .iter()
        .zip(&plan.assignments)
        .map(|(row, a)| {
            let chosen = row.choices.iter().find(|c| c.tier == a.tier).expect("a plan picks one of its site's choices");
            SiteReport {
                site: row.site.name.clone(),
                tensors: row.site.tensors.iter().map(|t| t.name.clone()).collect(),
                layer: row.site.layer,
                roles: row.site.roles.clone(),
                elements: row.site.numel(),
                tier: a.tier.name(),
                bytes: chosen.bytes,
                estimated_error: chosen.error,
            }
        })
        .collect();
    Ok(Report {
        policy: plan.policy(&sites).describe(),
        sites: site_reports,
        budget_bytes,
        plan_bytes: plan.bytes,
        f32_bytes: bytes_at(Dtype::F32).unwrap_or(0),
        bits_per_weight: plan.bytes as f64 * 8.0 / elements as f64,
        estimated_error: plan.estimated_error,
        original_loss,
        plan_loss,
        baselines,
        calibration_batches: req.calibration.len(),
        held_out_batches: req.held_out.len(),
    })
}

/// Whether two batches are the same rows, compared by their token streams.
fn same_batch(a: &Batch, b: &Batch) -> bool {
    let tokens = |batch: &Batch<'_>| -> Option<Vec<u32>> {
        match batch {
            Batch::Lm { tokens, .. } | Batch::LmWeighted { tokens, .. } | Batch::LmExamples { tokens, .. } | Batch::Multimodal { tokens, .. } => Some(tokens.to_vec()),
            Batch::Seq2Seq { src, .. } => Some(src.to_vec()),
            Batch::Tensor { .. } => None,
        }
    };
    matches!((tokens(a), tokens(b)), (Some(x), Some(y)) if x == y)
}
