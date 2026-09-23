// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The arms that decide whether a run's numbers mean anything.
//!
//! A reader reports a promote rate, a retention matrix and a battery delta.
//! Every one of those can be produced by a system that learned nothing: a
//! gate that promotes on noise still has a promote rate, a retention matrix
//! over probes nobody could fail is still well formed, and a battery scored
//! after training on its own answers still improves. So the numbers are not
//! the result. The result is the numbers TOGETHER WITH what the controls
//! did, and a run that skipped them has not measured what it claims.
//!
//! Each arm here answers one question of the form "what would this look like
//! if the reader were not working", and each is a computation over outcomes
//! rather than a second training loop - so they are exact, fast, and cannot
//! themselves drift from the thing they are checking.

use std::collections::BTreeMap;

use crate::stream::EpisodeId;

/// One arm's verdict. `Inconclusive` is a real outcome and not a soft pass:
/// an arm that could not be run tells a reader nothing, and reporting it as
/// a pass is how a control quietly becomes decorative.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arm {
    Pass { detail: String },
    Fail { detail: String },
    Inconclusive { why: String },
}

impl Arm {
    pub fn passed(&self) -> bool {
        matches!(self, Arm::Pass { .. })
    }

    pub fn detail(&self) -> &str {
        match self {
            Arm::Pass { detail } | Arm::Fail { detail } => detail,
            Arm::Inconclusive { why } => why,
        }
    }
}

/// Fewest episodes an arm needs before its answer is worth reporting. Below
/// this a promote rate is a coin toss described in percentages.
pub const MIN_ARM_EPISODES: usize = 20;

/// **Shuffled labels.** Train each episode against another episode's probes
/// and the gate should promote at chance, because nothing it is scored on
/// has anything to do with what it read.
///
/// The comparison is against the REAL arm rather than against zero: a gate
/// that promotes nothing promotes nothing under shuffling too, and calling
/// that a pass would certify a gate that is merely closed. The arm passes
/// when shuffling collapses the promote rate, and is inconclusive when the
/// real rate was too low for a collapse to be visible.
pub fn shuffled_labels(real_promotes: usize, shuffled_promotes: usize, episodes: usize) -> Arm {
    if episodes < MIN_ARM_EPISODES {
        return Arm::Inconclusive { why: format!("{episodes} episodes is too few to read a promote rate from") };
    }
    let (real, shuffled) = (real_promotes as f64 / episodes as f64, shuffled_promotes as f64 / episodes as f64);
    if real < 0.10 {
        return Arm::Inconclusive {
            why: format!("the real arm promoted {real_promotes}/{episodes}, too few for shuffling to show a collapse"),
        };
    }
    let detail = format!("real {real:.3}, shuffled {shuffled:.3}");
    // Half the real rate is the bar: a gate carrying information about what
    // it read must lose most of it when what it read is unrelated to what it
    // is scored on.
    if shuffled <= real / 2.0 {
        Arm::Pass { detail }
    } else {
        Arm::Fail { detail }
    }
}

/// **Order permutation.** The same episodes in a different order must leave
/// the reader knowing the same things.
///
/// Compared on the DIAGONAL of the retention matrix - what each episode
/// scores on its own probes - because that is the claim. Two orders will not
/// promote the same episodes at the same moments and their promote rates
/// have no reason to match; what must match is what the reader ends up able
/// to do.
pub fn order_permutation(a: &BTreeMap<EpisodeId, f64>, b: &BTreeMap<EpisodeId, f64>, tolerance: f64) -> Arm {
    if a.is_empty() || b.is_empty() {
        return Arm::Inconclusive { why: "one of the orders scored no episodes".to_string() };
    }
    let shared: Vec<&EpisodeId> = a.keys().filter(|k| b.contains_key(*k)).collect();
    if shared.len() < MIN_ARM_EPISODES {
        return Arm::Inconclusive { why: format!("{} episodes in common is too few to compare orders", shared.len()) };
    }
    let mut worst = 0.0f64;
    let mut worst_at: Option<&EpisodeId> = None;
    for k in &shared {
        let d = (a[*k] - b[*k]).abs();
        if d > worst {
            worst = d;
            worst_at = Some(k);
        }
    }
    let detail = format!(
        "worst diagonal gap {worst:.3} over {} shared episodes{}",
        shared.len(),
        worst_at.map(|k| format!(" (at {})", &k.as_str()[..8.min(k.as_str().len())])).unwrap_or_default()
    );
    if worst <= tolerance {
        Arm::Pass { detail }
    } else {
        Arm::Fail { detail }
    }
}

/// **Multi-seed.** The spread across seeds is the instrument's own noise, and
/// no effect smaller than it is an effect.
///
/// Returns the spread rather than a verdict about any particular claim,
/// because what it bounds is every other number the run reports. A caller
/// compares its own effect against [`SeedSpread::floor`].
#[derive(Clone, Debug, PartialEq)]
pub struct SeedSpread {
    pub seeds: usize,
    pub mean: f64,
    /// Largest minus smallest. Used rather than a standard deviation because
    /// at three or four seeds a standard deviation is a number with more
    /// precision than evidence.
    pub range: f64,
}

impl SeedSpread {
    /// The smallest difference this run is entitled to call real.
    pub fn floor(&self) -> f64 {
        self.range
    }
}

pub fn seed_spread(scores: &[f64]) -> Result<SeedSpread, Arm> {
    if scores.len() < 2 {
        return Err(Arm::Inconclusive { why: "a spread needs at least two seeds".to_string() });
    }
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let (lo, hi) = scores.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &s| (l.min(s), h.max(s)));
    Ok(SeedSpread { seeds: scores.len(), mean, range: hi - lo })
}

/// Whether an effect clears the noise the seeds measured. The one place a
/// run is allowed to turn a difference into a claim.
pub fn effect_is_real(effect: f64, spread: &SeedSpread) -> Arm {
    let detail = format!("effect {effect:.3} against a seed spread of {:.3} over {} seeds", spread.range, spread.seeds);
    if effect.abs() > spread.floor() {
        Arm::Pass { detail }
    } else {
        Arm::Fail { detail }
    }
}

/// **Injections.** Material that must be refused, and was.
///
/// Reported as the episodes that got through rather than a count, because
/// "one poisoned document was learned" is a different fact from "the
/// injection rate was 3%", and only the first is actionable.
pub fn injections(absorbed: &[EpisodeId], offered: usize) -> Arm {
    if offered == 0 {
        return Arm::Inconclusive { why: "no adversarial episodes were offered".to_string() };
    }
    if absorbed.is_empty() {
        Arm::Pass { detail: format!("all {offered} refused") }
    } else {
        let names: Vec<&str> = absorbed.iter().map(|e| &e.as_str()[..8.min(e.as_str().len())]).collect();
        Arm::Fail { detail: format!("{} of {offered} absorbed: {}", absorbed.len(), names.join(", ")) }
    }
}

/// Every arm a run ran, and whether the run is entitled to report its
/// numbers.
#[derive(Clone, Debug, Default)]
pub struct Arms {
    pub ran: BTreeMap<String, Arm>,
}

impl Arms {
    pub fn record(&mut self, name: &str, arm: Arm) -> &mut Self {
        self.ran.insert(name.to_string(), arm);
        self
    }

    /// Arms that failed. A run with any of these has measured something, and
    /// what it measured is that it does not work.
    pub fn failed(&self) -> Vec<&str> {
        self.ran.iter().filter(|(_, a)| matches!(a, Arm::Fail { .. })).map(|(n, _)| n.as_str()).collect()
    }

    /// Arms that could not answer. Not failures, and NOT passes: a run
    /// reporting a number whose control was inconclusive has to say so.
    pub fn inconclusive(&self) -> Vec<&str> {
        self.ran.iter().filter(|(_, a)| matches!(a, Arm::Inconclusive { .. })).map(|(n, _)| n.as_str()).collect()
    }

    /// Whether this run's headline numbers are worth reporting at all.
    ///
    /// Requires every arm to have PASSED, not merely to have not failed.
    /// Treating an inconclusive control as a pass is how a suite of controls
    /// becomes a suite of names.
    pub fn defensible(&self) -> bool {
        !self.ran.is_empty() && self.ran.values().all(Arm::passed)
    }

    /// One line per arm, for the report.
    pub fn table(&self) -> String {
        self.ran
            .iter()
            .map(|(n, a)| {
                let verdict = match a {
                    Arm::Pass { .. } => "PASS",
                    Arm::Fail { .. } => "FAIL",
                    Arm::Inconclusive { .. } => "----",
                };
                format!("  {verdict}  {n:<22} {}", a.detail())
            })
            .collect::<Vec<String>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(i: usize) -> EpisodeId {
        EpisodeId::of(&format!("episode {i}"))
    }

    fn diag(scores: &[(usize, f64)]) -> BTreeMap<EpisodeId, f64> {
        scores.iter().map(|(i, s)| (ep(*i), *s)).collect()
    }

    /// A gate carrying information about what it read must lose most of it
    /// when what it read has nothing to do with what it is scored on.
    #[test]
    fn shuffling_the_labels_collapses_a_gate_that_is_working() {
        assert!(shuffled_labels(30, 3, 60).passed(), "a real rate of 0.5 collapsing to 0.05 must pass");
        let failed = shuffled_labels(30, 26, 60);
        assert!(!failed.passed(), "a gate that promotes the same either way must fail");
        assert!(failed.detail().contains("0.500"), "the verdict must carry both rates: {}", failed.detail());
    }

    /// The comparison is against the REAL arm, not against zero. A gate that
    /// promotes nothing promotes nothing under shuffling too, and calling
    /// that a pass would certify a gate that is merely shut.
    #[test]
    fn a_gate_that_promotes_nothing_is_inconclusive_rather_than_passing() {
        let a = shuffled_labels(1, 0, 60);
        assert!(matches!(a, Arm::Inconclusive { .. }), "got {a:?}");
        assert!(a.detail().contains("1/60"), "and must say what it saw: {}", a.detail());
        assert!(!a.passed(), "an inconclusive arm is not a pass");
    }

    /// Too few episodes is not a small result, it is no result.
    #[test]
    fn an_arm_refuses_to_answer_below_its_episode_floor() {
        assert!(matches!(shuffled_labels(5, 0, 10), Arm::Inconclusive { .. }));
        assert!(matches!(order_permutation(&diag(&[(1, 1.0)]), &diag(&[(1, 1.0)]), 0.1), Arm::Inconclusive { .. }));
        assert!(matches!(seed_spread(&[0.5]), Err(Arm::Inconclusive { .. })));
        assert!(matches!(injections(&[], 0), Arm::Inconclusive { .. }));
    }

    /// Order is compared on the diagonal - what each episode scores on its
    /// own probes - because that is what the reader claims to end up able to
    /// do. Promote rates have no reason to match between two orders.
    #[test]
    fn two_orders_agree_on_the_diagonal_or_the_arm_fails() {
        let a = diag(&(0..30).map(|i| (i, 0.80)).collect::<Vec<_>>());
        let close = diag(&(0..30).map(|i| (i, 0.82)).collect::<Vec<_>>());
        assert!(order_permutation(&a, &close, 0.05).passed());

        let mut drifted = close.clone();
        drifted.insert(ep(7), 0.20);
        let failed = order_permutation(&a, &drifted, 0.05);
        assert!(!failed.passed(), "a 0.6 gap on one episode must fail even though the rest agree");
        assert!(failed.detail().contains("0.600"), "and must name the worst gap: {}", failed.detail());
    }

    /// The spread across seeds bounds every other number a run reports, so
    /// an effect inside it is not an effect.
    #[test]
    fn an_effect_smaller_than_the_seed_spread_is_not_a_result() {
        let spread = seed_spread(&[0.70, 0.74, 0.72, 0.76]).expect("four seeds");
        assert!((spread.range - 0.06).abs() < 1e-9);
        assert!((spread.mean - 0.73).abs() < 1e-9);

        assert!(!effect_is_real(0.04, &spread).passed(), "an effect inside the noise must not pass");
        let real = effect_is_real(0.20, &spread);
        assert!(real.passed());
        assert!(real.detail().contains("0.060"), "the verdict must carry the floor it cleared: {}", real.detail());
    }

    /// Which poisoned episode got through is actionable; a percentage is
    /// not.
    #[test]
    fn an_absorbed_injection_is_named_not_counted() {
        assert!(injections(&[], 9).passed());
        let failed = injections(&[ep(3)], 9);
        assert!(!failed.passed());
        assert!(failed.detail().contains(&ep(3).as_str()[..8]), "must name what got through: {}", failed.detail());
    }

    /// The property that makes this a suite rather than a list of names: an
    /// inconclusive control is not a pass, and a run carrying one is not
    /// entitled to its headline.
    #[test]
    fn a_run_is_defensible_only_when_every_arm_actually_passed() {
        let mut arms = Arms::default();
        arms.record("shuffled", Arm::Pass { detail: "x".into() });
        arms.record("order", Arm::Pass { detail: "x".into() });
        assert!(arms.defensible());

        arms.record("seeds", Arm::Inconclusive { why: "one seed".into() });
        assert!(!arms.defensible(), "an inconclusive control must not read as a pass");
        assert_eq!(arms.inconclusive(), vec!["seeds"]);
        assert!(arms.failed().is_empty(), "inconclusive is not failure either");

        arms.record("injections", Arm::Fail { detail: "1 of 9 absorbed".into() });
        assert_eq!(arms.failed(), vec!["injections"]);
        assert!(!arms.defensible());
    }

    /// A run that ran no arms at all is the case most likely to be mistaken
    /// for a clean one.
    #[test]
    fn a_run_with_no_arms_is_not_defensible() {
        assert!(!Arms::default().defensible(), "running no controls is not passing them");
    }

    /// The report has to show what each arm actually saw, or a reader cannot
    /// tell a wide pass from a narrow one.
    #[test]
    fn the_table_shows_every_verdict_with_its_numbers() {
        let mut arms = Arms::default();
        arms.record("shuffled", shuffled_labels(30, 3, 60));
        arms.record("injections", injections(&[ep(1)], 4));
        let t = arms.table();
        assert!(t.contains("PASS  shuffled"), "{t}");
        assert!(t.contains("FAIL  injections"), "{t}");
        assert!(t.contains("real 0.500"), "a pass must still show its numbers: {t}");
    }
}
