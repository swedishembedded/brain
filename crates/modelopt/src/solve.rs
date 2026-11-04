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

/// A tier for every site, its physical bytes and its estimated loss increase.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub assignments: Vec<Assignment>,
    pub bytes: u64,
    pub estimated_error: f64,
}

impl Plan {
    /// The plan as the per-leaf policy `--weight-format` serves: f32 by
    /// default, and a rule naming each site that is stored narrower. The
    /// default is f32 rather than the most common tier so that a leaf the plan
    /// does not cover (an embedding, a norm) keeps the precision it has.
    pub fn policy(&self, sites: &[Site]) -> TierPolicy {
        self.assignments
            .iter()
            .filter(|a| a.tier != Dtype::F32 && sites.iter().any(|s| s.name == a.site))
            .fold(TierPolicy::uniform(Dtype::F32), |p, a| p.with(&[a.site.as_str()], a.tier))
    }
}

/// An upgrade of one site from its current choice to a larger, more accurate
/// one, ordered by error removed per byte added.
struct Upgrade {
    gain_per_byte: f64,
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
        self.gain_per_byte.total_cmp(&other.gain_per_byte).then(other.site.cmp(&self.site))
    }
}

/// The choices of one site that are worth having: cheapest first, each
/// strictly more accurate than every cheaper one.
fn frontier(choices: &[Choice]) -> Vec<Choice> {
    let mut sorted = choices.to_vec();
    sorted.sort_by(|a, b| a.bytes.cmp(&b.bytes).then(a.error.total_cmp(&b.error)));
    let mut out: Vec<Choice> = Vec::new();
    for c in sorted {
        if out.last().is_none_or(|last| c.error < last.error && c.bytes > last.bytes) {
            out.push(c);
        }
    }
    out
}

/// The best upgrade of `site` from choice `from` that fits in `room` bytes.
fn best_upgrade(site: usize, front: &[Choice], from: usize, room: u64) -> Option<Upgrade> {
    let here = front[from];
    (from + 1..front.len())
        .filter(|&to| front[to].bytes - here.bytes <= room)
        .map(|to| Upgrade { gain_per_byte: (here.error - front[to].error) / (front[to].bytes - here.bytes) as f64, site, from, to })
        .max()
}

/// The tier per site that minimizes the summed estimated error with the
/// summed bytes at most `budget`.
///
/// Every site starts at its cheapest tier; then, while any upgrade fits, the
/// one removing the most error per added byte is taken. This is the greedy
/// solution of the knapsack's linear relaxation, exact whenever each site's
/// choices are convex in (bytes, error) and the budget lands on a breakpoint,
/// and within one upgrade of it otherwise. The estimate it optimizes is
/// itself approximate, which is why a plan is accepted only by measurement.
///
/// A site with no candidate tier, or a budget below the cheapest plan, is an
/// error naming what is missing.
pub fn assign(table: &[SiteChoices], budget: u64) -> Result<Plan, String> {
    let fronts: Vec<Vec<Choice>> = table.iter().map(|s| frontier(&s.choices)).collect();
    if let Some(empty) = table.iter().zip(&fronts).find(|(_, f)| f.is_empty()) {
        return Err(format!("{}: no candidate tier packs its [{}, {}] shape", empty.0.site.name, empty.0.site.n, empty.0.site.k));
    }
    let mut at: Vec<usize> = vec![0; fronts.len()];
    let floor: u64 = fronts.iter().map(|f| f[0].bytes).sum();
    if floor > budget {
        return Err(format!("the budget of {budget} bytes is below the smallest plan, {floor} bytes with every site at its cheapest tier"));
    }
    let mut room = budget - floor;
    let mut heap: BinaryHeap<Upgrade> = fronts.iter().enumerate().filter_map(|(i, f)| best_upgrade(i, f, 0, room)).collect();
    while let Some(up) = heap.pop() {
        if at[up.site] != up.from {
            continue;
        }
        let added = fronts[up.site][up.to].bytes - fronts[up.site][up.from].bytes;
        if added > room {
            // The room shrank since this upgrade was queued; a smaller one may still fit.
            heap.extend(best_upgrade(up.site, &fronts[up.site], up.from, room));
            continue;
        }
        room -= added;
        at[up.site] = up.to;
        heap.extend(best_upgrade(up.site, &fronts[up.site], up.to, room));
    }
    let chosen = table.iter().zip(&fronts).zip(&at).map(|((s, f), &i)| (s, f[i]));
    let (mut assignments, mut bytes, mut estimated_error) = (Vec::with_capacity(table.len()), 0u64, 0f64);
    for (s, c) in chosen {
        assignments.push(Assignment { site: s.site.name.clone(), tier: c.tier });
        bytes += c.bytes;
        estimated_error += c.error;
    }
    Ok(Plan { assignments, bytes, estimated_error })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site(name: &str, choices: &[(Dtype, u64, f64)]) -> SiteChoices {
        SiteChoices {
            site: Site { name: name.into(), n: 1, k: 32, layer: None, roles: vec![] },
            choices: choices.iter().map(|&(tier, bytes, error)| Choice { tier, bytes, error }).collect(),
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
        let plan = assign(&table, 30).unwrap();
        assert_eq!(tiers(&plan), [Dtype::Q4, Dtype::I8]);
        assert_eq!((plan.bytes, plan.estimated_error), (30, 2.0));
        assert_eq!(tiers(&assign(&table, 40).unwrap()), [Dtype::I8, Dtype::I8]);
        assert_eq!(tiers(&assign(&table, 20).unwrap()), [Dtype::Q4, Dtype::Q4]);
    }

    /// A tier that costs more and is no more accurate is never chosen, and an
    /// upgrade that skips a step is taken when only the far step pays.
    #[test]
    fn dominated_tiers_are_never_chosen() {
        let table = [site("a", &[(Dtype::Q4, 10, 4.0), (Dtype::NF4, 12, 4.0), (Dtype::I8, 20, 3.9), (Dtype::BF16, 40, 0.0)])];
        let plan = assign(&table, 40).unwrap();
        assert_eq!(tiers(&plan), [Dtype::BF16]);
        assert_eq!(tiers(&assign(&table, 39).unwrap()), [Dtype::I8]);
    }

    /// When the best upgrade no longer fits, a smaller one still does.
    #[test]
    fn leftover_room_is_still_spent() {
        let table = [site("a", &[(Dtype::Q4, 10, 10.0), (Dtype::I8, 20, 5.0), (Dtype::F32, 80, 0.0)]), site("b", &[(Dtype::Q4, 10, 1.0), (Dtype::I8, 20, 0.0)])];
        let plan = assign(&table, 45).unwrap();
        assert_eq!(tiers(&plan), [Dtype::I8, Dtype::I8]);
        assert!(plan.bytes <= 45);
    }

    #[test]
    fn infeasible_inputs_are_refused_by_name() {
        let table = [site("a", &[(Dtype::Q4, 10, 1.0)])];
        assert!(assign(&table, 9).unwrap_err().contains("below the smallest plan, 10 bytes"));
        assert!(assign(&[site("unpackable", &[])], 100).unwrap_err().contains("unpackable"));
    }

    #[test]
    fn the_policy_names_only_narrowed_sites_and_keeps_the_rest_f32() {
        let table = [site("blocks.1.wq", &[(Dtype::I8, 10, 1.0), (Dtype::F32, 40, 0.0)]), site("blocks.11.wq", &[(Dtype::I8, 10, 1.0), (Dtype::F32, 40, 0.0)])];
        let sites: Vec<Site> = table.iter().map(|s| s.site.clone()).collect();
        let plan = Plan {
            assignments: vec![Assignment { site: "blocks.1.wq".into(), tier: Dtype::I8 }, Assignment { site: "blocks.11.wq".into(), tier: Dtype::F32 }],
            bytes: 50,
            estimated_error: 1.0,
        };
        let policy = plan.policy(&sites);
        assert_eq!((policy.want("blocks.1.wq"), policy.want("blocks.11.wq"), policy.want("embed")), (Dtype::I8, Dtype::F32, Dtype::F32));
    }
}
