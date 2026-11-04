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
    /// Mean stored bits per searched weight element, scales included: `9` is
    /// uniform group-scaled int8, `32` is f32.
    BitsPerWeight(f64),
    /// What every searched tensor takes at one tier: a plan at the size of a
    /// uniform deployment, searched to lose less than it does.
    Uniform(Dtype),
    /// Halfway between the two smallest candidate tiers' uniform sizes: the
    /// plan must beat the smaller uniform deployment by spending the extra
    /// bytes where they buy the most.
    Midway,
}

impl Budget {
    /// The budget in bytes for `sites` searched over `tiers`.
    pub fn bytes(self, sites: &[Site], tiers: &[Dtype]) -> Result<u64, String> {
        let uniform = |tier: Dtype| -> Option<u64> { sites.iter().flat_map(|s| &s.tensors).map(|t| Weight::packs(tier, t.n, t.k).then(|| Weight::bytes_for(tier, t.n, t.k))).sum() };
        match self {
            Budget::Bytes(b) => Ok(b),
            Budget::BitsPerWeight(bits) if bits.is_finite() && bits > 0.0 => Ok((bits * sites.iter().map(Site::numel).sum::<usize>() as f64 / 8.0) as u64),
            Budget::BitsPerWeight(bits) => Err(format!("a bits-per-weight budget must be finite and positive, got {bits}")),
            Budget::Uniform(tier) => uniform(tier).ok_or_else(|| format!("a uniform {} budget: some searched tensor does not pack at that tier", tier.name())),
            Budget::Midway => {
                let mut sizes: Vec<u64> = tiers.iter().filter_map(|&t| uniform(t)).collect();
                sizes.sort_unstable();
                sizes.dedup();
                match sizes[..] {
                    [smallest, next, ..] => Ok(smallest + (next - smallest) / 2),
                    _ => Err("a midway budget needs two candidate tiers of different sizes that pack every searched tensor".into()),
                }
            }
        }
    }
}

/// Where a search is, for a caller showing progress: it runs a forward and
/// backward per calibration batch, round-trips every site through every
/// candidate codec, then runs the held-out batches once per measured plan.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Progress {
    /// Calibration batch `batch` (1-based) of `of` has been read.
    Calibrated { batch: usize, of: usize },
    /// The codec errors of site `site` (1-based) of `of` are estimated.
    Estimated { site: usize, of: usize },
    /// Distinct product shape `shape` (1-based) of `of` is timed at every candidate tier.
    Timed { shape: usize, of: usize },
    /// A plan (`"plan"`, `"original"`, a uniform tier's name) is about to be measured on the held-out batches.
    Measuring { label: String },
}

/// What to search: the candidate tiers, the budget, and two disjoint batch
/// sets, one the estimate ranks on and one that measures the result.
pub struct SearchRequest<'a, 'b> {
    pub tiers: &'a [Dtype],
    pub budget: Budget,
    pub calibration: &'a [Batch<'b>],
    pub held_out: &'a [Batch<'b>],
    /// Time every site's product on the device, and optionally limit the plan by it.
    pub timing: Option<Timing<'a>>,
}

/// Seconds of one `rows x k` by `n x k` product at a tier on the target device.
pub type Timer<'a> = &'a dyn Fn(u32, usize, usize, Dtype) -> Result<f64, String>;

/// How the search times the sites and what it limits.
pub struct Timing<'a> {
    /// The rows of the timed product: 1 is single-stream decode, where a
    /// narrower weight streams fewer bytes and its decode cost shows.
    pub rows: u32,
    pub limit: Option<LatencyLimit>,
    pub time: Timer<'a>,
}

/// The latency a plan may take, summed over its sites' products.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LatencyLimit {
    Seconds(f64),
    /// No slower than every site at one tier.
    Uniform(Dtype),
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
    /// The measured seconds of the site's product at its tier, when timed.
    pub seconds: Option<f64>,
    /// The estimated loss increase at the chosen tier.
    pub estimated_error: f64,
}

/// A uniform tier the plan is compared against: every searched site at it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Baseline {
    pub tier: &'static str,
    pub bytes: u64,
    pub seconds: Option<f64>,
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
    /// The summed product seconds of the plan, of the searched linears at f32,
    /// and the limit, when the search timed the sites.
    pub plan_seconds: Option<f64>,
    pub f32_seconds: Option<f64>,
    pub latency_limit_seconds: Option<f64>,
    pub timed_rows: Option<u32>,
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
pub fn search_weights<M: Model>(m: &M, req: &SearchRequest, on_progress: &mut dyn FnMut(Progress)) -> Result<Report, String> {
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
    let budget_bytes = req.budget.bytes(&sites, req.tiers)?;
    let mut table = sensitivity(m, &sites, req.tiers, req.calibration, on_progress)?;
    let site_seconds = match &req.timing {
        Some(t) => Some(time_sites(&sites, req.tiers, t, on_progress)?),
        None => None,
    };
    if let Some(times) = &site_seconds {
        for (row, times) in table.iter_mut().zip(times) {
            for c in &mut row.choices {
                c.seconds = times.iter().find(|(t, _)| *t == c.tier).map_or(0.0, |(_, s)| *s);
            }
        }
    }
    let seconds_at = |tier: Dtype| -> Option<f64> { site_seconds.as_ref().and_then(|times| times.iter().map(|row| row.iter().find(|(t, _)| *t == tier).map(|(_, s)| *s)).sum()) };
    let latency_limit_seconds = match req.timing.as_ref().and_then(|t| t.limit) {
        None => None,
        Some(LatencyLimit::Seconds(s)) => Some(s),
        Some(LatencyLimit::Uniform(tier)) => Some(seconds_at(tier).ok_or_else(|| format!("a latency limit like uniform {}: that tier is not a timed candidate", tier.name()))?),
    };
    let plan = assign(&table, crate::Limits { bytes: budget_bytes, seconds: latency_limit_seconds })?;
    on_progress(Progress::Measuring { label: "original".into() });
    let original_loss = held_out_loss(m, req.held_out);
    on_progress(Progress::Measuring { label: "plan".into() });
    let plan_loss = measure_policy(m, &sites, &plan.assignments, req.held_out);
    let bytes_at = |tier: Dtype| -> Option<u64> { sites.iter().flat_map(|s| &s.tensors).map(|t| Weight::packs(tier, t.n, t.k).then(|| Weight::bytes_for(tier, t.n, t.k))).sum() };
    let baselines = req
        .tiers
        .iter()
        .filter_map(|&tier| bytes_at(tier).filter(|&b| b <= budget_bytes).map(|bytes| (tier, bytes)))
        .filter(|&(tier, _)| latency_limit_seconds.is_none_or(|limit| seconds_at(tier).is_some_and(|s| s <= limit)))
        .map(|(tier, bytes)| {
            let uniform: Vec<Assignment> = sites.iter().map(|s| Assignment { site: s.name.clone(), tier }).collect();
            on_progress(Progress::Measuring { label: tier.name().into() });
            Baseline { tier: tier.name(), bytes, seconds: seconds_at(tier), loss: measure_policy(m, &sites, &uniform, req.held_out) }
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
                seconds: site_seconds.is_some().then_some(chosen.seconds),
                estimated_error: chosen.error,
            }
        })
        .collect();
    Ok(Report {
        policy: plan.policy(&sites).describe(),
        sites: site_reports,
        budget_bytes,
        plan_bytes: plan.bytes,
        plan_seconds: site_seconds.is_some().then_some(plan.seconds),
        f32_seconds: seconds_at(Dtype::F32),
        latency_limit_seconds,
        timed_rows: req.timing.as_ref().map(|t| t.rows),
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

/// Every site's product seconds at every candidate tier its tensors pack,
/// timed once per distinct shape: a site's tensors that share a contraction
/// width are one fused product (`rows x k` by the summed `n`), as the engines
/// that fuse them run it.
fn time_sites(sites: &[Site], tiers: &[Dtype], timing: &Timing, on_progress: &mut dyn FnMut(Progress)) -> Result<Vec<Vec<(Dtype, f64)>>, String> {
    let products = |site: &Site| -> Vec<(usize, usize)> {
        let mut by_k: Vec<(usize, usize)> = Vec::new();
        for t in &site.tensors {
            match by_k.iter_mut().find(|(_, k)| *k == t.k) {
                Some((n, _)) => *n += t.n,
                None => by_k.push((t.n, t.k)),
            }
        }
        by_k
    };
    let mut shapes: Vec<(usize, usize)> = sites.iter().flat_map(products).collect();
    shapes.sort_unstable();
    shapes.dedup();
    let mut timed: std::collections::HashMap<(usize, usize, Dtype), f64> = std::collections::HashMap::new();
    for (i, &(n, k)) in shapes.iter().enumerate() {
        for &tier in tiers.iter().filter(|&&t| Weight::packs(t, n, k)) {
            timed.insert((n, k, tier), (timing.time)(timing.rows, n, k, tier).map_err(|e| format!("timing a {} x {k} product at {}: {e}", n, tier.name()))?);
        }
        on_progress(Progress::Timed { shape: i + 1, of: shapes.len() });
    }
    Ok(sites
        .iter()
        .map(|site| tiers.iter().filter_map(|&tier| products(site).iter().map(|&(n, k)| timed.get(&(n, k, tier)).copied()).sum::<Option<f64>>().map(|s| (tier, s))).collect())
        .collect())
}
