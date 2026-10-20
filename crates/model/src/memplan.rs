// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training and serving of large models for its clients. If your
// team needs expertise in fitting a model, its optimizer, activations and KV cache into a fixed GPU then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The global memory planner: weights, gradients, optimizer state, activations, KV cache, staging and
//! transfer buffers are separate [`Category`]s, each with an ordered list of [`Variant`]s (ways to hold it,
//! largest first), and [`plan`] picks the variants that fit a [`Capacity`] BEFORE anything is loaded.
//!
//! # Decision order
//!
//! Levers are applied in a fixed order, each only while the total still does not fit and only if the caller
//! authorized it ([`Authorized`]):
//!
//! 1. preserve quantized storage - not a lever but a rule: a category's first variant is what is stored on disk,
//!    and no variant may be larger than the one before it, so the planner can never expand weights;
//! 2. [`Lever::OptimizerState`] (narrower moment formats), 3. [`Lever::Recompute`], 4. [`Lever::ActivationPrecision`]
//!    (only variants marked `validated`), 5. [`Lever::Offload`] (device bytes move to host bytes, which must fit),
//!    6. [`Lever::CompactKv`] (a smaller [`crate::kv_tier::KvTier`], for rollouts), 7. [`Lever::Microbatch`].
//!
//! Every choice the planner makes is recorded in the [`PlanReport`] (there is no silent precision change), and
//! when nothing authorized fits it refuses with the arithmetic ([`PlanError::DoesNotFit`]).
//!
//! Staging and transfer demands come from the checkpoint crate's [`checkpoint::packed::MemoryBudget`]
//! ([`staging_variant`]): the planner charges the configured cap, which is the contract's bound.

use std::collections::BTreeSet;
use std::fmt;

use checkpoint::packed::{MemoryBudget, Pool};

use crate::kv_tier::KvTier;

/// What a block of memory is for. Reported separately, never merged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Category {
    Weights,
    Grads,
    OptimizerState,
    Activations,
    Kv,
    Staging,
    Transfers,
}

/// The ways the planner may shrink the device footprint, in the order it tries them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lever {
    /// The category's stored form; never a decision.
    Base,
    OptimizerState,
    Recompute,
    ActivationPrecision,
    Offload,
    CompactKv,
    Microbatch,
}

/// One way to hold a category: bytes on the device and bytes on the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Variant {
    pub lever: Lever,
    pub label: String,
    pub device_bytes: u64,
    pub host_bytes: u64,
    /// Whether accuracy was measured for this variant (required of [`Lever::ActivationPrecision`]).
    pub validated: bool,
}

impl Variant {
    pub fn new(lever: Lever, label: impl Into<String>, device_bytes: u64, host_bytes: u64) -> Variant {
        Variant { lever, label: label.into(), device_bytes, host_bytes, validated: true }
    }

    /// Mark a variant whose accuracy has not been measured; the planner will not pick it.
    pub fn unvalidated(mut self) -> Variant {
        self.validated = false;
        self
    }
}

/// A category and its variants, the first being the preferred (stored / highest precision) one.
#[derive(Clone, Debug)]
pub struct Demand {
    pub category: Category,
    pub variants: Vec<Variant>,
}

impl Demand {
    /// A demand with one fixed variant (weights as stored, gradients, ...).
    pub fn fixed(category: Category, label: impl Into<String>, device_bytes: u64) -> Demand {
        Demand { category, variants: vec![Variant::new(Lever::Base, label, device_bytes, 0)] }
    }

    /// A demand with alternatives. The first must be [`Lever::Base`] and device bytes must not grow along
    /// the list, so applying a lever can only shrink the footprint.
    pub fn with_variants(category: Category, variants: Vec<Variant>) -> Result<Demand, PlanError> {
        let bad = |why: String| PlanError::InvalidDemand { category, why };
        let Some(first) = variants.first() else { return Err(bad("no variants".into())) };
        if first.lever != Lever::Base {
            return Err(bad(format!("the first variant {:?} must be the stored form (Lever::Base)", first.label)));
        }
        if let Some(w) = variants.windows(2).find(|w| w[1].device_bytes > w[0].device_bytes) {
            return Err(bad(format!("variant {:?} ({} B) is larger than {:?} ({} B) before it", w[1].label, w[1].device_bytes, w[0].label, w[0].device_bytes)));
        }
        Ok(Demand { category, variants })
    }
}

/// The KV cache of `n_layers` full-attention layers holding `rows` token rows, one variant per tier of
/// `tiers` (largest first), each a [`Lever::CompactKv`] step after the first.
pub fn kv_demand(n_layers: u64, rows: u64, kv_stride: u64, head_dim: u64, tiers: &[KvTier]) -> Result<Demand, PlanError> {
    let mut tiers: Vec<KvTier> = tiers.to_vec();
    let bytes = |t: &KvTier| n_layers * 2 * t.plane_bytes(rows, kv_stride, head_dim);
    tiers.sort_by_key(|t| std::cmp::Reverse(bytes(t)));
    let variants = tiers.iter().enumerate().map(|(i, t)| Variant::new(if i == 0 { Lever::Base } else { Lever::CompactKv }, t.as_str(), bytes(t), 0)).collect();
    Demand::with_variants(Category::Kv, variants)
}

/// The staging demand of loads sharing `budget`: its configured host cap and pinned cap, charged in full
/// (host bytes are not device bytes, so they count against the host capacity).
pub fn staging_variant(budget: &MemoryBudget) -> Demand {
    let host = budget.stats(Pool::HostStaging).capacity;
    let pinned = budget.stats(Pool::Pinned).capacity;
    Demand { category: Category::Staging, variants: vec![Variant::new(Lever::Base, "bounded staging", 0, host + pinned)] }
}

/// What the machine offers.
#[derive(Clone, Copy, Debug)]
pub struct Capacity {
    pub device_bytes: u64,
    pub host_bytes: u64,
}

/// The levers the caller allows the planner to pull.
#[derive(Clone, Debug)]
pub struct Authorized(BTreeSet<Lever>);

impl Authorized {
    pub fn none() -> Authorized {
        Authorized(BTreeSet::new())
    }

    pub fn only(levers: &[Lever]) -> Authorized {
        Authorized(levers.iter().copied().collect())
    }

    pub fn all() -> Authorized {
        Authorized::only(&[Lever::OptimizerState, Lever::Recompute, Lever::ActivationPrecision, Lever::Offload, Lever::CompactKv, Lever::Microbatch])
    }

    fn allows(&self, lever: Lever) -> bool {
        self.0.contains(&lever)
    }
}

/// One choice the planner made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub category: Category,
    pub lever: Lever,
    pub from: String,
    pub to: String,
    pub device_bytes_saved: u64,
}

/// What a plan holds, per category, and every decision that led there.
#[derive(Clone, Debug)]
pub struct PlanReport {
    pub chosen: Vec<(Category, Variant)>,
    pub decisions: Vec<Decision>,
    pub device_bytes: u64,
    pub host_bytes: u64,
}

impl PlanReport {
    pub fn variant(&self, category: Category) -> Option<&Variant> {
        self.chosen.iter().find(|(c, _)| *c == category).map(|(_, v)| v)
    }
}

impl fmt::Display for PlanReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (c, v) in &self.chosen {
            writeln!(f, "  {c:?}: {} ({} B device, {} B host)", v.label, v.device_bytes, v.host_bytes)?;
        }
        for d in &self.decisions {
            writeln!(f, "  decision: {:?} {:?} {} -> {} (saves {} B)", d.lever, d.category, d.from, d.to, d.device_bytes_saved)?;
        }
        write!(f, "  total: {} B device, {} B host", self.device_bytes, self.host_bytes)
    }
}

#[derive(Clone, Debug)]
pub enum PlanError {
    InvalidDemand { category: Category, why: String },
    /// Nothing authorized fits: the plan with every authorized lever applied, for the message.
    DoesNotFit { capacity: Capacity, best: PlanReport },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanError::InvalidDemand { category, why } => write!(f, "invalid {category:?} demand: {why}"),
            PlanError::DoesNotFit { capacity, best } => write!(
                f,
                "does not fit: the smallest plan the authorized levers reach needs {} B device and {} B host, the machine has {} B and {} B\n{best}",
                best.device_bytes, best.host_bytes, capacity.device_bytes, capacity.host_bytes
            ),
        }
    }
}

impl std::error::Error for PlanError {}

fn totals(chosen: &[usize], demands: &[Demand]) -> (u64, u64) {
    demands.iter().zip(chosen).fold((0, 0), |(d, h), (demand, &i)| (d + demand.variants[i].device_bytes, h + demand.variants[i].host_bytes))
}

/// Choose one variant per demand so the total fits `capacity`, pulling only authorized levers, in the
/// module's order and no further than needed; refuse with the arithmetic when nothing authorized fits.
pub fn plan(demands: &[Demand], capacity: Capacity, authorized: &Authorized) -> Result<PlanReport, PlanError> {
    let mut chosen = vec![0usize; demands.len()];
    let mut decisions = Vec::new();
    let fits = |chosen: &[usize]| {
        let (d, h) = totals(chosen, demands);
        d <= capacity.device_bytes && h <= capacity.host_bytes
    };
    const ORDER: [Lever; 6] = [Lever::OptimizerState, Lever::Recompute, Lever::ActivationPrecision, Lever::Offload, Lever::CompactKv, Lever::Microbatch];
    'levers: for lever in ORDER {
        if !authorized.allows(lever) {
            continue;
        }
        for (ci, demand) in demands.iter().enumerate() {
            loop {
                if fits(&chosen) {
                    break 'levers;
                }
                let next = chosen[ci] + 1;
                let Some(v) = demand.variants.get(next) else { break };
                if v.lever != lever || (lever == Lever::ActivationPrecision && !v.validated) {
                    break;
                }
                let mut trial = chosen.clone();
                trial[ci] = next;
                // A step that overflows the host (offload) is not taken.
                if totals(&trial, demands).1 > capacity.host_bytes {
                    break;
                }
                let from = &demand.variants[chosen[ci]];
                decisions.push(Decision { category: demand.category, lever, from: from.label.clone(), to: v.label.clone(), device_bytes_saved: from.device_bytes - v.device_bytes });
                chosen = trial;
            }
        }
    }
    let (device_bytes, host_bytes) = totals(&chosen, demands);
    let report = PlanReport { chosen: demands.iter().zip(&chosen).map(|(d, &i)| (d.category, d.variants[i].clone())).collect(), decisions, device_bytes, host_bytes };
    if fits(&chosen) {
        Ok(report)
    } else {
        Err(PlanError::DoesNotFit { capacity, best: report })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn variants(category: Category, steps: &[(Lever, &str, u64, u64)]) -> Demand {
        Demand::with_variants(category, steps.iter().map(|&(l, n, d, h)| Variant::new(l, n, d, h)).collect()).unwrap()
    }

    /// A training-shaped problem: 10 GiB weights and 10 GiB gradients (fixed), optimizer state in f32/bf16/int8,
    /// activations keep/recompute/offload, KV f32/int8.
    fn problem() -> Vec<Demand> {
        vec![
            Demand::fixed(Category::Weights, "int8 as stored", 10 * GIB),
            Demand::fixed(Category::Grads, "bf16", 10 * GIB),
            variants(Category::OptimizerState, &[(Lever::Base, "f32", 40 * GIB, 0), (Lever::OptimizerState, "bf16", 20 * GIB, 0), (Lever::OptimizerState, "int8", 10 * GIB, 0)]),
            variants(Category::Activations, &[(Lever::Base, "keep", 30 * GIB, 0), (Lever::Recompute, "recompute", 8 * GIB, 0), (Lever::Offload, "offload", 2 * GIB, 8 * GIB)]),
            variants(Category::Kv, &[(Lever::Base, "f32", 8 * GIB, 0), (Lever::CompactKv, "int8", 2 * GIB, 0)]),
        ]
    }

    fn cap(device_gib: u64) -> Capacity {
        Capacity { device_bytes: device_gib * GIB, host_bytes: 64 * GIB }
    }

    #[test]
    fn a_problem_that_fits_makes_no_decisions() {
        let r = plan(&problem(), cap(200), &Authorized::all()).unwrap();
        assert!(r.decisions.is_empty());
        assert_eq!(r.device_bytes, 98 * GIB);
    }

    /// The optimizer state is narrowed before activations are recomputed, and the planner stops the moment the
    /// total fits: needing 10 GiB back must not touch the activations.
    #[test]
    fn levers_are_pulled_in_order_and_no_further_than_needed() {
        let r = plan(&problem(), cap(90), &Authorized::all()).unwrap();
        assert_eq!(r.decisions.len(), 1);
        assert_eq!((r.decisions[0].lever, r.decisions[0].to.as_str()), (Lever::OptimizerState, "bf16"));
        assert_eq!(r.variant(Category::Activations).unwrap().label, "keep");

        let r = plan(&problem(), cap(38), &Authorized::all()).unwrap();
        let order: Vec<Lever> = r.decisions.iter().map(|d| d.lever).collect();
        assert_eq!(order, vec![Lever::OptimizerState, Lever::OptimizerState, Lever::Recompute, Lever::Offload, Lever::CompactKv]);
        assert!(r.device_bytes <= 38 * GIB);
        let r = plan(&problem(), cap(60), &Authorized::all()).unwrap();
        assert_eq!(r.decisions.last().unwrap().lever, Lever::Recompute, "stops as soon as the total fits");
    }

    #[test]
    fn an_unauthorized_lever_is_never_pulled_and_the_refusal_shows_the_arithmetic() {
        let err = plan(&problem(), cap(60), &Authorized::only(&[Lever::Recompute])).unwrap_err();
        let PlanError::DoesNotFit { best, .. } = &err else { panic!("{err}") };
        assert!(best.decisions.iter().all(|d| d.lever == Lever::Recompute), "only the authorized lever may be pulled");
        let text = err.to_string();
        assert!(text.contains("does not fit") && text.contains("OptimizerState: f32") && text.contains("device"), "{text}");
        assert!(plan(&problem(), cap(60), &Authorized::none()).is_err());
    }

    #[test]
    fn an_unvalidated_activation_precision_is_never_chosen() {
        let d = Demand::with_variants(Category::Activations, vec![Variant::new(Lever::Base, "bf16", 20 * GIB, 0), Variant::new(Lever::ActivationPrecision, "fp8", 10 * GIB, 0).unvalidated()]).unwrap();
        let err = plan(std::slice::from_ref(&d), cap(15), &Authorized::all()).unwrap_err();
        assert!(matches!(err, PlanError::DoesNotFit { .. }));
        let mut ok = d;
        ok.variants[1].validated = true;
        assert_eq!(plan(&[ok], cap(15), &Authorized::all()).unwrap().decisions.len(), 1);
    }

    /// Offload trades device bytes for host bytes; it is not taken when the host cannot hold them.
    #[test]
    fn offload_respects_the_host_capacity() {
        let tight = Capacity { device_bytes: 35 * GIB, host_bytes: 4 * GIB };
        let err = plan(&problem(), tight, &Authorized::all()).unwrap_err();
        let PlanError::DoesNotFit { best, .. } = err else { unreachable!() };
        assert!(best.decisions.iter().all(|d| d.lever != Lever::Offload));
        assert!(best.host_bytes <= 4 * GIB);
        assert!(plan(&problem(), cap(35), &Authorized::all()).unwrap().decisions.iter().any(|d| d.lever == Lever::Offload));
    }

    #[test]
    fn a_demand_that_grows_or_starts_off_base_is_refused() {
        let grows = Demand::with_variants(Category::Kv, vec![Variant::new(Lever::Base, "int8", GIB, 0), Variant::new(Lever::CompactKv, "f32", 4 * GIB, 0)]);
        assert!(matches!(grows, Err(PlanError::InvalidDemand { .. })), "no lever may expand a category");
        let off_base = Demand::with_variants(Category::Kv, vec![Variant::new(Lever::CompactKv, "int8", GIB, 0)]);
        assert!(matches!(off_base, Err(PlanError::InvalidDemand { .. })));
    }

    /// The KV demand is the KvTier descriptor's own byte count: fp8 costs int8's bytes, bf16 half of f32, and
    /// a planner that needs room for 128k-token rollouts walks f32 -> bf16 -> int8/fp8.
    #[test]
    fn kv_demand_is_sized_by_the_tier_descriptor() {
        let d = kv_demand(16, 131_072, 1024, 256, &KvTier::ALL).unwrap();
        let bytes: Vec<(String, u64)> = d.variants.iter().map(|v| (v.label.clone(), v.device_bytes)).collect();
        assert_eq!(bytes[0].0, "f32");
        assert_eq!(bytes[1].1 * 2, bytes[0].1, "bf16 is half of f32");
        assert_eq!(bytes[2].1, bytes[3].1, "int8 and fp8 have one footprint");
        let r = plan(&[Demand::fixed(Category::Weights, "int8", 27 * GIB), d], Capacity { device_bytes: 80 * GIB, host_bytes: 0 }, &Authorized::only(&[Lever::CompactKv])).unwrap();
        assert!(r.decisions.iter().all(|d| d.lever == Lever::CompactKv));
        assert!(r.device_bytes <= 80 * GIB);
    }

    #[test]
    fn staging_is_charged_at_the_loaders_configured_cap() {
        let budget = MemoryBudget::new(3 * GIB, GIB);
        let d = staging_variant(&budget);
        assert_eq!((d.variants[0].device_bytes, d.variants[0].host_bytes), (0, 4 * GIB));
        assert!(plan(std::slice::from_ref(&d), Capacity { device_bytes: GIB, host_bytes: 4 * GIB }, &Authorized::none()).is_ok());
        assert!(plan(&[d], Capacity { device_bytes: GIB, host_bytes: 3 * GIB }, &Authorized::none()).is_err());
    }
}
