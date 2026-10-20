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
//!    6. [`Lever::ConvertWeights`] (an explicit, recorded Convert of the frozen base to a narrower format: only for a
//!    caller that authorized it), 7. [`Lever::CompactKv`] (a smaller [`crate::kv_tier::KvTier`], for rollouts),
//!    8. [`Lever::Microbatch`].
//!
//! Every choice the planner makes is recorded in the [`PlanReport`] (there is no silent precision change), and
//! when nothing authorized fits it refuses with the arithmetic ([`PlanError::DoesNotFit`]).
//!
//! Staging and transfer demands come from the checkpoint crate's [`checkpoint::packed::MemoryBudget`]
//! ([`staging_variant`]): the planner charges the configured cap, which is the contract's bound.

use std::collections::BTreeSet;
use std::fmt;

use checkpoint::packed::{MemoryBudget, Pool};
pub use paramstore::precision::ParamPrecision;

use crate::kv_tier::KvTier;

/// What a block of memory is for. Reported separately, never merged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Category {
    Weights,
    Grads,
    /// Recurrent / decode state that is neither KV nor activations (a hybrid decoder's GDN state).
    DecodeState,
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
    /// Convert the frozen base to a narrower storage format (bf16 -> NF4 / int8). Never implicit.
    ConvertWeights,
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

/// The KV cache as variants over `tiers`: `bytes(tier)` is that tier's whole-cache device bytes
/// (the caller knows its layout: windows per slot, a paged pool, a layer subset). The first tier
/// is the requested one and the [`Lever::Base`]; only tiers strictly smaller than it follow, largest
/// first, each a [`Lever::CompactKv`] step - the planner never proposes a bigger cache.
pub fn kv_demand_with(requested: KvTier, tiers: &[KvTier], bytes: impl Fn(KvTier) -> u64) -> Result<Demand, PlanError> {
    let base = bytes(requested);
    let mut smaller: Vec<KvTier> = tiers.iter().copied().filter(|&t| bytes(t) < base).collect();
    smaller.sort_by_key(|&t| std::cmp::Reverse(bytes(t)));
    smaller.dedup_by_key(|t| bytes(*t));
    let mut variants = vec![Variant::new(Lever::Base, requested.as_str(), base, 0)];
    variants.extend(smaller.iter().map(|t| Variant::new(Lever::CompactKv, t.as_str(), bytes(*t), 0)));
    Demand::with_variants(Category::Kv, variants)
}

/// The KV cache of `n_layers` full-attention layers holding `rows` token rows, starting from the
/// largest of `tiers` and offering each smaller one.
pub fn kv_demand(n_layers: u64, rows: u64, kv_stride: u64, head_dim: u64, tiers: &[KvTier]) -> Result<Demand, PlanError> {
    let bytes = |t: KvTier| n_layers * 2 * t.plane_bytes(rows, kv_stride, head_dim);
    let Some(&largest) = tiers.iter().max_by_key(|&&t| bytes(t)) else { return Err(PlanError::InvalidDemand { category: Category::Kv, why: "no tiers".into() }) };
    kv_demand_with(largest, tiers, bytes)
}

/// The staging demand of loads sharing `budget`: its configured host cap and pinned cap, charged in full
/// (host bytes are not device bytes, so they count against the host capacity).
pub fn staging_variant(budget: &MemoryBudget) -> Demand {
    let host = budget.stats(Pool::HostStaging).capacity;
    let pinned = budget.stats(Pool::Pinned).capacity;
    Demand { category: Category::Staging, variants: vec![Variant::new(Lever::Base, "bounded staging", 0, host + pinned)] }
}

/// Activation bytes per token per unit of `layers * width`, calibrated on the 27B (64 layers, width 5120): 78 GiB at 16K
/// tokens against 67 GiB at 8K for the bf16 base is 1.4 MiB per token. An estimate for planning a step before a model is
/// loaded, not a measurement of any one trainer.
const ACTIVATION_BYTES_PER_TOKEN_PER_LAYER_WIDTH: f64 = 1.4 * 1048576.0 / (64.0 * 5120.0);

/// Estimated activation bytes of one step over `tokens` tokens of a `layers`-deep, `width`-wide decoder.
pub fn calibrated_activation_bytes(tokens: u32, layers: u64, width: u64) -> u64 {
    (ACTIVATION_BYTES_PER_TOKEN_PER_LAYER_WIDTH * tokens as f64 * layers as f64 * width as f64) as u64
}

/// The device bytes of every demand at its preferred (stored) variant: what a sizing costs before
/// any lever is pulled.
pub fn device_total(demands: &[Demand]) -> u64 {
    demands.iter().map(|d| d.variants[0].device_bytes).sum()
}

/// One line per category, for a log: `weights 12.00 GiB + kv 2.50 GiB + ... = 15.2 GiB`.
pub fn summary(demands: &[Demand]) -> String {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let parts: Vec<String> = demands.iter().map(|d| format!("{:?} {:.2} GiB ({})", d.category, gib(d.variants[0].device_bytes), d.variants[0].label)).collect();
    format!("{} = {:.2} GiB", parts.join(" + "), gib(device_total(demands)))
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
        Authorized::only(&[Lever::OptimizerState, Lever::Recompute, Lever::ActivationPrecision, Lever::Offload, Lever::ConvertWeights, Lever::CompactKv, Lever::Microbatch])
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
    const ORDER: [Lever; 7] = [Lever::OptimizerState, Lever::Recompute, Lever::ActivationPrecision, Lever::Offload, Lever::ConvertWeights, Lever::CompactKv, Lever::Microbatch];
    'levers: for lever in ORDER {
        if !authorized.allows(lever) {
            continue;
        }
        for (ci, demand) in demands.iter().enumerate() {
            loop {
                if fits(&chosen) {
                    break 'levers;
                }
                // The next variant of THIS lever after the current choice; variants of other levers between them
                // (an unauthorized rung) are skipped, which cannot grow the device bytes.
                let Some(next) = (chosen[ci] + 1..demand.variants.len()).find(|&i| demand.variants[i].lever == lever) else { break };
                let v = &demand.variants[next];
                if lever == Lever::ActivationPrecision && !v.validated {
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

/// What a training run holds, described the way the trainers know it: the frozen base as placed, the trainable
/// manifest with its [`paramstore::precision::ParamPrecision`] (value, gradient, master copy, optimizer-state
/// format - the physical byte accounting of `paramstore`), and the activations of one step.
#[derive(Clone, Debug)]
pub struct TrainingFootprint {
    /// Physical device bytes of the frozen base (`paramstore::dtype::Destination` accounting), 0 for a full fine-tune.
    pub frozen_weight_bytes: u64,
    /// `(name, elements)` of every trainable parameter.
    pub trainable: Vec<(String, usize)>,
    pub precision: ParamPrecision,
    /// Optimizer state slots per parameter (2 for AdamW, 1 for Lion / SGD momentum).
    pub moment_slots: usize,
    /// Block size of the 8-bit state format offered as the narrowest rung.
    pub int8_block: u32,
    /// The optimizer state already lives in host memory (the full fine-tune's layout); it then costs host bytes only.
    pub optimizer_state_on_host: bool,
    /// Activation bytes of one step when every layer's are kept, and when each layer is rebuilt in the backward
    /// (`None`: this trainer cannot recompute).
    pub activation_bytes: u64,
    pub recompute_activation_bytes: Option<u64>,
}

/// The planner categories of a training run. Optimizer state is offered as the `optimizer_state_ladder` from the
/// requested format down to 8-bit blocks (every rung counts its scales) and, if it is on the device, as an
/// offload to host memory; activations as keep, then recompute when the trainer can.
pub fn training_demands(f: &TrainingFootprint) -> Result<Vec<Demand>, PlanError> {
    use paramstore::precision::StateFormat;
    let numel: usize = f.trainable.iter().map(|(_, n)| *n).sum();
    let bytes = f.precision.bytes(numel, f.moment_slots);
    let mut demands = vec![Demand::fixed(Category::Weights, "frozen base", f.frozen_weight_bytes), Demand::fixed(Category::Weights, "trainable values", bytes.value), Demand::fixed(Category::Grads, "gradients", bytes.grad)];
    if bytes.master > 0 {
        demands.push(Demand::fixed(Category::Weights, "master copy", bytes.master));
    }
    let ladder = paramstore::precision::optimizer_state_ladder(&f.trainable, f.moment_slots, f.int8_block);
    let requested = ladder.iter().position(|(fmt, _)| *fmt == f.precision.state).unwrap_or(0);
    let label = |fmt: &StateFormat| format!("{fmt:?}");
    let state = if f.optimizer_state_on_host {
        let b = ladder[requested].1;
        Demand { category: Category::OptimizerState, variants: vec![Variant::new(Lever::Base, format!("{} on the host", label(&ladder[requested].0)), 0, b)] }
    } else {
        let mut variants = vec![Variant::new(Lever::Base, label(&ladder[requested].0), ladder[requested].1, 0)];
        variants.extend(ladder[requested + 1..].iter().map(|(fmt, b)| Variant::new(Lever::OptimizerState, label(fmt), *b, 0)));
        // Offloading keeps the REQUESTED format on the host (a narrower rung taken before it is not carried over:
        // host memory is the roomier side, and the report shows the change).
        variants.push(Variant::new(Lever::Offload, format!("{} on the host", label(&ladder[requested].0)), 0, ladder[requested].1));
        Demand::with_variants(Category::OptimizerState, variants)?
    };
    demands.push(state);
    let mut act = vec![Variant::new(Lever::Base, "keep activations", f.activation_bytes, 0)];
    if let Some(r) = f.recompute_activation_bytes {
        act.push(Variant::new(Lever::Recompute, "recompute per layer", r, 0));
    }
    demands.push(Demand::with_variants(Category::Activations, act)?);
    Ok(demands)
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
        assert_eq!(bytes.len(), 3, "int8 and fp8 have one footprint, so only one is offered");
        assert_eq!(bytes[2].1, 16 * 2 * KvTier::FP8.plane_bytes(131_072, 1024, 256));
        let r = plan(&[Demand::fixed(Category::Weights, "int8", 27 * GIB), d], Capacity { device_bytes: 80 * GIB, host_bytes: 0 }, &Authorized::only(&[Lever::CompactKv])).unwrap();
        assert!(r.decisions.iter().all(|d| d.lever == Lever::CompactKv));
        assert!(r.device_bytes <= 80 * GIB);
    }

    fn training(trainable: usize, on_host: bool) -> TrainingFootprint {
        TrainingFootprint {
            frozen_weight_bytes: 0,
            trainable: vec![("w".into(), trainable)],
            precision: paramstore::precision::ParamPrecision::FP32,
            moment_slots: 2,
            int8_block: 256,
            optimizer_state_on_host: on_host,
            activation_bytes: 8 * GIB,
            recompute_activation_bytes: Some(GIB),
        }
    }

    /// An 8B full fine-tune at fp32 (the paramstore byte accounting): values 4n, grads 4n, AdamW state 8n. The
    /// planner narrows the state first, offloads it only when authorized, and recomputes only after narrowing.
    #[test]
    fn a_full_finetune_is_planned_from_the_paramstore_byte_accounting() {
        let n = 2_000_000_000usize; // 8 GB of values at fp32
        let demands = training_demands(&training(n, false)).unwrap();
        let total = device_total(&demands);
        assert_eq!(total, 4 * n as u64 + 4 * n as u64 + 8 * n as u64 + 8 * GIB, "values + grads + AdamW f32 moments + activations");
        let ladder = paramstore::precision::optimizer_state_ladder(&[("w".into(), n)], 2, 256);
        let int8 = ladder[2].1;
        // Needs the state narrowed to int8 blocks: 8n -> ~2n.
        let room = total - (8 * n as u64 - int8) + 1;
        let r = plan(&demands, Capacity { device_bytes: room, host_bytes: 0 }, &Authorized::only(&[Lever::OptimizerState])).unwrap();
        assert_eq!(r.variant(Category::OptimizerState).unwrap().label, "Int8Block { block: 256 }");
        assert_eq!(r.variant(Category::Activations).unwrap().label, "keep activations");
        // Not authorized to narrow, but the host has room: the state moves there at the requested format.
        let r = plan(&demands, Capacity { device_bytes: room, host_bytes: 16 * GIB * 4 }, &Authorized::only(&[Lever::Offload])).unwrap();
        assert_eq!(r.variant(Category::OptimizerState).unwrap().label, "F32 on the host");
        assert!(r.decisions.iter().all(|d| d.lever == Lever::Offload));
        // Nothing authorized and nothing fits: the refusal carries the per-category arithmetic.
        let err = plan(&demands, Capacity { device_bytes: GIB, host_bytes: 0 }, &Authorized::none()).unwrap_err().to_string();
        assert!(err.contains("OptimizerState: F32") && err.contains("Grads: gradients"), "{err}");
    }

    /// The full fine-tune's layout: the optimizer state already lives on the host, so it costs host bytes only,
    /// and the device holds values and gradients.
    #[test]
    fn state_already_on_the_host_is_charged_to_the_host() {
        let n = 1_000_000_000usize;
        let demands = training_demands(&training(n, true)).unwrap();
        let r = plan(&demands, Capacity { device_bytes: 8 * n as u64 + 8 * GIB, host_bytes: 8 * n as u64 }, &Authorized::none()).unwrap();
        assert_eq!((r.device_bytes, r.host_bytes), (8 * n as u64 + 8 * GIB, 8 * n as u64));
        assert!(plan(&demands, Capacity { device_bytes: 8 * n as u64 + 8 * GIB, host_bytes: 4 * n as u64 }, &Authorized::none()).is_err(), "the host must hold the state too");
    }

    /// Converting the frozen base is a lever of its own, tried after offload and only when authorized.
    #[test]
    fn converting_the_base_is_a_recorded_authorized_lever() {
        let weights = Demand::with_variants(Category::Weights, vec![Variant::new(Lever::Base, "bf16", 54 * GIB, 0), Variant::new(Lever::ConvertWeights, "nf4", 17 * GIB, 0)]).unwrap();
        let acts = Demand::fixed(Category::Activations, "keep", 20 * GIB);
        let room = Capacity { device_bytes: 60 * GIB, host_bytes: 0 };
        let r = plan(&[weights.clone(), acts.clone()], room, &Authorized::only(&[Lever::ConvertWeights])).unwrap();
        assert_eq!(r.decisions.len(), 1);
        assert_eq!(r.decisions[0].lever, Lever::ConvertWeights);
        assert!(plan(&[weights, acts], room, &Authorized::none()).is_err());
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
