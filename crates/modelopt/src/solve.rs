// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The tier per site that minimizes the estimated loss increase within a
//! byte budget: a multiple-choice knapsack.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use gpu_core::select::Dtype;
use model::ops::TierPolicy;

use crate::{Choice, Site, SiteChoices};

/// One site stored at one tier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assignment {
    pub site: String,
    pub tier: Dtype,
}

/// A tier for every site, its physical bytes, its measured seconds (0 when
/// the search timed nothing) and its estimated loss increase.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub assignments: Vec<Assignment>,
    pub bytes: u64,
    pub seconds: f64,
    pub estimated_error: f64,
}

impl Plan {
    /// The plan as the per-leaf policy `--weight-format` serves: f32 by
    /// default, and a rule naming each tensor of each site that is stored
    /// narrower. The default is f32 rather than the most common tier so that a
    /// leaf the plan does not cover (an embedding, a norm) keeps the precision
    /// it has.
    pub fn policy(&self, sites: &[Site]) -> TierPolicy {
        policy_of(&self.assignments, sites)
    }
}

/// [`Plan::policy`] of any assignment list.
pub fn policy_of(assignments: &[Assignment], sites: &[Site]) -> TierPolicy {
    let mut policy = TierPolicy::uniform(Dtype::F32);
    for a in assignments.iter().filter(|a| a.tier != Dtype::F32) {
        if let Some(site) = sites.iter().find(|s| s.name == a.site) {
            for t in &site.tensors {
                policy = policy.with(&[t.name.as_str()], a.tier);
            }
        }
    }
    policy
}

/// What a plan may spend: physical bytes, and optionally the summed measured
/// seconds of every site's product (the latency of the searched linears).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    pub bytes: u64,
    pub seconds: Option<f64>,
}

impl Limits {
    /// A byte budget alone.
    pub fn bytes(bytes: u64) -> Limits {
        Limits { bytes, seconds: None }
    }
}

/// How a choice's cost weighs the limits: each dimension's share of its
/// limit, times a weight that grows while the starting plan breaks that limit.
#[derive(Clone, Copy)]
struct Scale {
    limits: Limits,
    bytes_weight: f64,
    seconds_weight: f64,
}

impl Scale {
    /// The one scale an upgrade's error reduction is weighed by.
    fn cost(&self, c: &Choice) -> f64 {
        self.bytes_weight * c.bytes as f64 / self.limits.bytes.max(1) as f64 + self.limits.seconds.map_or(0.0, |t| self.seconds_weight * c.seconds / t)
    }
}

/// An upgrade of one site from its current choice to a costlier, more accurate
/// one, ordered by error removed per unit of cost added.
struct Upgrade {
    gain_per_cost: f64,
    site: usize,
    from: usize,
    to: usize,
}

impl PartialEq for Upgrade {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Upgrade {}
impl PartialOrd for Upgrade {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Upgrade {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain_per_cost.total_cmp(&other.gain_per_cost).then(other.site.cmp(&self.site))
    }
}

/// The choices of one site that are worth having: cheapest first by
/// [`Limits::cost`], each strictly more accurate than every cheaper one.
fn frontier(choices: &[Choice], limits: &Scale) -> Vec<Choice> {
    let mut sorted = choices.to_vec();
    sorted.sort_by(|a, b| limits.cost(a).total_cmp(&limits.cost(b)).then(a.error.total_cmp(&b.error)));
    let mut out: Vec<Choice> = Vec::new();
    for c in sorted {
        if out.last().is_none_or(|last| c.error < last.error && limits.cost(&c) > limits.cost(last)) {
            out.push(c);
        }
    }
    out
}

/// Room left on each limited dimension.
#[derive(Clone, Copy)]
struct Room {
    bytes: i128,
    seconds: f64,
}

impl Room {
    /// Whether moving from `from` to `to` fits: a move that frees a dimension always fits it.
    fn fits(&self, from: &Choice, to: &Choice) -> bool {
        to.bytes as i128 - from.bytes as i128 <= self.bytes && to.seconds - from.seconds <= self.seconds
    }

    fn take(&mut self, from: &Choice, to: &Choice) {
        self.bytes -= to.bytes as i128 - from.bytes as i128;
        self.seconds -= to.seconds - from.seconds;
    }
}

/// The best upgrade of `site` from choice `from` that fits in `room`.
fn best_upgrade(site: usize, front: &[Choice], from: usize, room: &Room, limits: &Scale) -> Option<Upgrade> {
    let here = front[from];
    (from + 1..front.len())
        .filter(|&to| room.fits(&here, &front[to]))
        .map(|to| Upgrade { gain_per_cost: (here.error - front[to].error) / (limits.cost(&front[to]) - limits.cost(&here)), site, from, to })
        .max()
}

/// The tier per site that minimizes the summed estimated error with the
/// summed bytes, and the summed seconds when limited, within `limits`.
///
/// Every site starts at its cheapest choice; then, while any upgrade fits
/// every limit, the one removing the most error per unit of added cost is
/// taken, cost being each dimension's share of its limit. With bytes alone
/// this is the greedy solution of the knapsack's linear relaxation, exact
/// whenever each site's choices are convex in (bytes, error) and the budget
/// lands on a breakpoint, and within one upgrade of it otherwise; a second
/// limit makes it a heuristic. The estimate it optimizes is itself
/// approximate, which is why a plan is accepted only by measurement.
///
/// A site with no candidate tier, or limits below the cheapest plan, is an
/// error naming what is missing.
pub fn assign(table: &[SiteChoices], limits: Limits) -> Result<Plan, String> {
    if let Some(empty) = table.iter().find(|s| s.choices.is_empty()) {
        return Err(format!("{}: no candidate tier packs every tensor of the site", empty.site.name));
    }
    let least_bytes: u64 = table.iter().map(|s| s.choices.iter().map(|c| c.bytes).min().unwrap_or(0)).sum();
    let least_seconds: f64 = table.iter().map(|s| s.choices.iter().map(|c| c.seconds).fold(f64::INFINITY, f64::min)).sum();
    if least_bytes > limits.bytes {
        return Err(format!("the budget of {} bytes is below the smallest plan, {least_bytes} bytes with every site at its smallest tier", limits.bytes));
    }
    if let Some(t) = limits.seconds.filter(|&t| least_seconds > t) {
        return Err(format!("the latency limit of {t:.6} s is below the fastest plan's {least_seconds:.6} s"));
    }
    // Start every site at its cheapest choice; while that plan breaks a limit, weigh that limit more and start
    // again, until the start fits both or the weights show the limits cannot be met together.
    let mut scale = Scale { limits, bytes_weight: 1.0, seconds_weight: 1.0 };
    let (fronts, floor_bytes, floor_seconds) = loop {
        let fronts: Vec<Vec<Choice>> = table.iter().map(|s| frontier(&s.choices, &scale)).collect();
        let floor_bytes: u64 = fronts.iter().map(|f| f[0].bytes).sum();
        let floor_seconds: f64 = fronts.iter().map(|f| f[0].seconds).sum();
        let (over_bytes, over_seconds) = (floor_bytes > limits.bytes, limits.seconds.is_some_and(|t| floor_seconds > t));
        if !over_bytes && !over_seconds {
            break (fronts, floor_bytes, floor_seconds);
        }
        if scale.bytes_weight.max(scale.seconds_weight) > 1e12 {
            return Err(format!("no plan meets {} bytes and {:.6} s together; the closest start takes {floor_bytes} bytes and {floor_seconds:.6} s", limits.bytes, limits.seconds.unwrap_or(f64::INFINITY)));
        }
        if over_bytes {
            scale.bytes_weight *= 2.0;
        }
        if over_seconds {
            scale.seconds_weight *= 2.0;
        }
    };
    let mut at: Vec<usize> = vec![0; fronts.len()];
    let mut room = Room { bytes: (limits.bytes - floor_bytes) as i128, seconds: limits.seconds.map_or(f64::INFINITY, |t| t - floor_seconds) };
    let mut heap: BinaryHeap<Upgrade> = fronts.iter().enumerate().filter_map(|(i, f)| best_upgrade(i, f, 0, &room, &scale)).collect();
    while let Some(up) = heap.pop() {
        if at[up.site] != up.from {
            continue;
        }
        let (from, to) = (&fronts[up.site][up.from], &fronts[up.site][up.to]);
        if !room.fits(from, to) {
            // The room shrank since this upgrade was queued; a smaller one may still fit.
            heap.extend(best_upgrade(up.site, &fronts[up.site], up.from, &room, &scale));
            continue;
        }
        room.take(from, to);
        at[up.site] = up.to;
        heap.extend(best_upgrade(up.site, &fronts[up.site], up.to, &room, &scale));
    }
    let mut plan = Plan { assignments: Vec::with_capacity(table.len()), bytes: 0, seconds: 0.0, estimated_error: 0.0 };
    for ((s, f), &i) in table.iter().zip(&fronts).zip(&at) {
        plan.assignments.push(Assignment { site: s.site.name.clone(), tier: f[i].tier });
        plan.bytes += f[i].bytes;
        plan.seconds += f[i].seconds;
        plan.estimated_error += f[i].error;
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site(name: &str, choices: &[(Dtype, u64, f64)]) -> SiteChoices {
        SiteChoices {
            site: Site { name: name.into(), tensors: vec![crate::Tensor { name: name.into(), n: 1, k: 32 }], layer: None, roles: vec![] },
            choices: choices.iter().map(|&(tier, bytes, error)| Choice { tier, bytes, seconds: 0.0, error }).collect(),
        }
    }

    fn tiers(plan: &Plan) -> Vec<Dtype> {
        plan.assignments.iter().map(|a| a.tier).collect()
    }

    /// Two sites with the same size and tiers: the sensitive one gets the
    /// bytes, and a budget between "both small" and "both large" buys exactly
    /// one upgrade.
    #[test]
    fn the_budget_goes_to_the_sensitive_site() {
        let table = [site("robust", &[(Dtype::Q4, 10, 1.0), (Dtype::I8, 20, 0.5)]), site("sensitive", &[(Dtype::Q4, 10, 9.0), (Dtype::I8, 20, 1.0)])];
        let plan = assign(&table, Limits::bytes(30)).unwrap();
        assert_eq!(tiers(&plan), [Dtype::Q4, Dtype::I8]);
        assert_eq!((plan.bytes, plan.estimated_error), (30, 2.0));
        assert_eq!(tiers(&assign(&table, Limits::bytes(40)).unwrap()), [Dtype::I8, Dtype::I8]);
        assert_eq!(tiers(&assign(&table, Limits::bytes(20)).unwrap()), [Dtype::Q4, Dtype::Q4]);
    }

    /// A tier that costs more and is no more accurate is never chosen, and an
    /// upgrade that skips a step is taken when only the far step pays.
    #[test]
    fn dominated_tiers_are_never_chosen() {
        let table = [site("a", &[(Dtype::Q4, 10, 4.0), (Dtype::NF4, 12, 4.0), (Dtype::I8, 20, 3.9), (Dtype::BF16, 40, 0.0)])];
        let plan = assign(&table, Limits::bytes(40)).unwrap();
        assert_eq!(tiers(&plan), [Dtype::BF16]);
        assert_eq!(tiers(&assign(&table, Limits::bytes(39)).unwrap()), [Dtype::I8]);
    }

    /// When the best upgrade no longer fits, a smaller one still does.
    #[test]
    fn leftover_room_is_still_spent() {
        let table = [site("a", &[(Dtype::Q4, 10, 10.0), (Dtype::I8, 20, 5.0), (Dtype::F32, 80, 0.0)]), site("b", &[(Dtype::Q4, 10, 1.0), (Dtype::I8, 20, 0.0)])];
        let plan = assign(&table, Limits::bytes(45)).unwrap();
        assert_eq!(tiers(&plan), [Dtype::I8, Dtype::I8]);
        assert!(plan.bytes <= 45);
    }

    /// A latency limit steers the plan away from a tier that is small but slow
    /// on the device (a narrow format whose decode costs more than it saves),
    /// even when the bytes would allow it.
    #[test]
    fn a_latency_limit_rules_out_a_small_but_slow_tier() {
        let timed = |name: &str, choices: &[(Dtype, u64, f64, f64)]| SiteChoices {
            site: Site { name: name.into(), tensors: vec![crate::Tensor { name: name.into(), n: 1, k: 32 }], layer: None, roles: vec![] },
            choices: choices.iter().map(|&(tier, bytes, seconds, error)| Choice { tier, bytes, seconds, error }).collect(),
        };
        // nf4: smallest, slowest; int8: middling; bf16: largest, fastest, exact.
        let row = |n: &str| timed(n, &[(Dtype::NF4, 10, 5.0, 2.0), (Dtype::I8, 20, 2.0, 1.0), (Dtype::BF16, 40, 1.0, 0.0)]);
        let table = [row("a"), row("b")];
        assert_eq!(tiers(&assign(&table, Limits::bytes(40)).unwrap()), [Dtype::I8, Dtype::I8], "bytes alone");
        let plan = assign(&table, Limits { bytes: 40, seconds: Some(4.0) }).unwrap();
        assert!(plan.seconds <= 4.0 && plan.bytes <= 40, "{plan:?}");
        assert_eq!(tiers(&plan), [Dtype::I8, Dtype::I8]);
        let err = assign(&table, Limits { bytes: 40, seconds: Some(1.5) }).unwrap_err();
        assert!(err.contains("latency"), "{err}");
        // The cheapest start by shares (nf4 here: small, and the latency limit loose enough to weigh little) breaks
        // the limit; the solver weighs latency up until a start fits rather than refusing a feasible plan.
        let plan = assign(&table, Limits { bytes: 80, seconds: Some(9.0) }).unwrap();
        assert!(plan.seconds <= 9.0 && plan.bytes <= 80, "{plan:?}");
        // A tight latency limit with room in bytes buys the fast tier.
        let plan = assign(&table, Limits { bytes: 80, seconds: Some(2.0) }).unwrap();
        assert_eq!(tiers(&plan), [Dtype::BF16, Dtype::BF16], "{plan:?}");
    }

    #[test]
    fn infeasible_inputs_are_refused_by_name() {
        let table = [site("a", &[(Dtype::Q4, 10, 1.0)])];
        assert!(assign(&table, Limits::bytes(9)).unwrap_err().contains("below the smallest plan, 10 bytes"));
        assert!(assign(&[site("unpackable", &[])], Limits::bytes(100)).unwrap_err().contains("unpackable"));
    }

    #[test]
    fn the_policy_names_only_narrowed_sites_and_keeps_the_rest_f32() {
        let table = [site("blocks.1.wq", &[(Dtype::I8, 10, 1.0), (Dtype::F32, 40, 0.0)]), site("blocks.11.wq", &[(Dtype::I8, 10, 1.0), (Dtype::F32, 40, 0.0)])];
        let sites: Vec<Site> = table.iter().map(|s| s.site.clone()).collect();
        let plan = Plan {
            assignments: vec![Assignment { site: "blocks.1.wq".into(), tier: Dtype::I8 }, Assignment { site: "blocks.11.wq".into(), tier: Dtype::F32 }],
            bytes: 50,
            seconds: 0.0,
            estimated_error: 1.0,
        };
        let policy = plan.policy(&sites);
        assert_eq!((policy.want("blocks.1.wq"), policy.want("blocks.11.wq"), policy.want("embed")), (Dtype::I8, Dtype::F32, Dtype::F32));
    }
}
