// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The pre-registered block a run is checked against.
//!
//! Written before the run and asserted by a test rather than printed to a
//! report, per this repo's rule that a number nothing checks goes stale
//! silently. The eight clauses are the ones this design committed to; they
//! are here as a type so a run cannot report a subset of them and call it a
//! result.
//!
//! **The acceptance criterion is a DEFENSIBLE number, not a positive one.**
//! A reader that learned nothing and says so, with its controls intact, has
//! passed this; a reader with a large battery delta and an inconclusive
//! control has not. That asymmetry is the whole point of pre-registering:
//! it fixes what counts as an answer before anyone knows what the answer is.

use crate::arms::{Arms, SeedSpread};

/// One clause of the block: what it asserts, and what the run actually did.
#[derive(Clone, Debug, PartialEq)]
pub struct Clause {
    pub name: &'static str,
    pub met: bool,
    /// What the run measured, whichever way it went. A clause that failed
    /// without saying what it saw cannot be acted on.
    pub observed: String,
}

impl Clause {
    fn of(name: &'static str, met: bool, observed: impl Into<String>) -> Clause {
        Clause { name, met, observed: observed.into() }
    }
}

/// What a run measured, in the terms the block is written in.
#[derive(Clone, Debug)]
pub struct RunFacts {
    pub episodes: usize,
    pub promoted: usize,
    /// Rejections that carry a named cause. Clause 1 is about whether every
    /// refusal can be explained, not about how many there were.
    pub rejections: usize,
    pub rejections_with_cause: usize,
    /// The null-gate control's promote count over the same stream.
    pub null_gate_promoted: usize,
    /// Backward transfer over the retention matrix.
    pub bwt: f64,
    /// Whether the per-block bar was armed for this run's gate.
    pub per_block_bar_armed: bool,
    /// Worst single-block regression seen, so clause 3 reports what it
    /// allowed rather than only that it allowed it.
    pub worst_block_drop: f64,
    pub max_block_drop: f64,
    /// The independent battery, never trained on and never gated on.
    pub battery_before: f64,
    pub battery_after: f64,
    pub battery_budget: f64,
    /// Episodes to re-check the whole bank.
    pub detection_latency: u64,
    /// Probe decodes spent on the audit, and on everything.
    pub eval_decodes: u64,
    pub total_decodes: u64,
    pub eval_budget_share: f64,
    /// Two runs at the same seed: byte-identical adapters, and the delta
    /// between their battery scores.
    pub seed_repeat_identical: bool,
    pub seed_repeat_delta: f64,
    pub seed_spread: Option<SeedSpread>,
    /// The largest effect this run wants to claim.
    pub claimed_effect: f64,
}

/// The block, evaluated.
#[derive(Clone, Debug)]
pub struct Acceptance {
    pub clauses: Vec<Clause>,
}

impl Acceptance {
    /// Evaluate all eight clauses. `arms` carries R10's controls, which
    /// clause 6 is about and which clause 1 leans on for the null-gate
    /// comparison.
    pub fn evaluate(f: &RunFacts, arms: &Arms) -> Acceptance {
        let mut c = Vec::new();

        // 1. Every rejection explains itself, and the gate is separated from
        //    a coin flip by more than the seeds' own noise.
        let all_explained = f.rejections_with_cause == f.rejections;
        let separation = (f.promoted as f64 - f.null_gate_promoted as f64) / f.episodes.max(1) as f64;
        let floor = f.seed_spread.as_ref().map(SeedSpread::floor).unwrap_or(f64::INFINITY);
        c.push(Clause::of(
            "causes and null gate",
            all_explained && separation > floor,
            format!(
                "{}/{} rejections named; gated {} vs null {} over {} (separation {separation:.3}, seed floor {floor:.3})",
                f.rejections_with_cause, f.rejections, f.promoted, f.null_gate_promoted, f.episodes
            ),
        ));

        // 2. The matrix is reported as deltas over each episode's own
        //    zero-shot baseline, which is what makes episodes comparable at
        //    all. Represented here by the battery having a before AND an
        //    after: a run with no baseline cannot express a delta.
        c.push(Clause::of(
            "zero-shot baselines",
            f.battery_before.is_finite() && f.battery_after.is_finite(),
            format!("battery {:.3} -> {:.3}", f.battery_before, f.battery_after),
        ));

        // 3. Retention, with the bar that can see a single episode collapse.
        c.push(Clause::of(
            "bwt and per-block bar",
            f.bwt >= -f.max_block_drop && f.per_block_bar_armed && f.worst_block_drop <= f.max_block_drop,
            format!(
                "bwt {:.4}, worst block drop {:.3} against {:.3}, bar {}",
                f.bwt,
                f.worst_block_drop,
                f.max_block_drop,
                if f.per_block_bar_armed { "armed" } else { "OFF" }
            ),
        ));

        // 4. The battery nobody trained on and nobody gated on.
        let regression = f.battery_before - f.battery_after;
        c.push(Clause::of(
            "independent battery",
            regression <= f.battery_budget,
            format!("regressed {regression:.3} against a budget of {:.3}", f.battery_budget),
        ));

        // 5. A latency, not a guarantee. Zero means nothing was re-checked,
        //    which is not a fast audit but an absent one.
        c.push(Clause::of(
            "detection latency",
            f.detection_latency > 0,
            format!("every earlier episode re-checked within {} episodes", f.detection_latency),
        ));

        // 6. R10's arms, all of them, actually passed.
        c.push(Clause::of(
            "control arms",
            arms.defensible(),
            if arms.ran.is_empty() {
                "no arms were run".to_string()
            } else {
                format!("{} ran, failed {:?}, inconclusive {:?}", arms.ran.len(), arms.failed(), arms.inconclusive())
            },
        ));

        // 7. The audit has to stay affordable or the reader stops reading.
        let share = f.eval_decodes as f64 / f.total_decodes.max(1) as f64;
        c.push(Clause::of(
            "eval compute share",
            share <= f.eval_budget_share,
            format!("{share:.3} of total decodes against a budget of {:.3}", f.eval_budget_share),
        ));

        // 8. Determinism, and no claim smaller than the instrument.
        c.push(Clause::of(
            "reproducible",
            f.seed_repeat_identical && f.claimed_effect.abs() > f.seed_repeat_delta,
            format!(
                "seed repeat {}, delta {:.4}, largest claim {:.4}",
                if f.seed_repeat_identical { "bit-identical" } else { "DIFFERED" },
                f.seed_repeat_delta,
                f.claimed_effect
            ),
        ));

        Acceptance { clauses: c }
    }

    /// Whether the run may report its headline.
    pub fn accepted(&self) -> bool {
        self.clauses.iter().all(|c| c.met)
    }

    pub fn unmet(&self) -> Vec<&'static str> {
        self.clauses.iter().filter(|c| !c.met).map(|c| c.name).collect()
    }

    /// The block as a reader sees it. Every clause is printed whether it
    /// passed or not: a report that lists only failures cannot be read as
    /// evidence that the rest were checked.
    pub fn table(&self) -> String {
        self.clauses
            .iter()
            .map(|c| format!("  {}  {:<22} {}", if c.met { "PASS" } else { "FAIL" }, c.name, c.observed))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arms::{seed_spread, Arm};

    /// A run that met the block: modest, controlled, and honest about what
    /// it measured. Deliberately not a spectacular one - the block is meant
    /// to accept this and reject a spectacular run with a broken control.
    fn good() -> RunFacts {
        RunFacts {
            episodes: 400,
            promoted: 240,
            rejections: 160,
            rejections_with_cause: 160,
            null_gate_promoted: 44,
            bwt: -0.004,
            per_block_bar_armed: true,
            worst_block_drop: 0.03,
            max_block_drop: 0.20,
            battery_before: 0.05,
            battery_after: 0.68,
            battery_budget: 0.02,
            detection_latency: 62,
            eval_decodes: 184_000,
            total_decodes: 1_000_000,
            eval_budget_share: 0.25,
            seed_repeat_identical: true,
            seed_repeat_delta: 0.0,
            seed_spread: Some(seed_spread(&[0.66, 0.68, 0.70]).expect("three seeds")),
            claimed_effect: 0.63,
        }
    }

    fn all_arms_pass() -> Arms {
        let mut a = Arms::default();
        for n in ["shuffled", "order", "seeds", "injections"] {
            a.record(n, Arm::Pass { detail: "ok".into() });
        }
        a
    }

    #[test]
    fn a_controlled_run_meets_every_clause() {
        let acc = Acceptance::evaluate(&good(), &all_arms_pass());
        assert!(acc.accepted(), "unmet: {:?}\n{}", acc.unmet(), acc.table());
        assert_eq!(acc.clauses.len(), 8, "the block has eight clauses and all of them are checked");
    }

    /// The asymmetry the whole block exists for: a spectacular result with a
    /// control that could not answer is NOT accepted, while a modest one
    /// with its controls intact is.
    #[test]
    fn a_large_result_with_an_inconclusive_control_is_refused() {
        let mut spectacular = good();
        spectacular.battery_after = 0.99;
        spectacular.claimed_effect = 0.94;
        let mut arms = all_arms_pass();
        arms.record("shuffled", Arm::Inconclusive { why: "the real arm promoted 2 of 400".into() });

        let acc = Acceptance::evaluate(&spectacular, &arms);
        assert!(!acc.accepted(), "a bigger number must not buy its way past a control");
        assert_eq!(acc.unmet(), vec!["control arms"]);
    }

    /// And its mirror: a run that learned almost nothing, said so, and kept
    /// its controls, has produced a defensible number and passes.
    #[test]
    fn a_run_that_learned_little_but_measured_it_properly_is_accepted() {
        let mut modest = good();
        modest.battery_after = 0.11;
        modest.claimed_effect = 0.06;
        let acc = Acceptance::evaluate(&modest, &all_arms_pass());
        assert!(acc.accepted(), "unmet: {:?}\n{}", acc.unmet(), acc.table());
    }

    /// Clause 1 is about explicability, not volume: one unexplained refusal
    /// is enough, because a rejection nobody can account for is a rejection
    /// nobody can act on.
    #[test]
    fn a_single_rejection_without_a_cause_fails_clause_one() {
        let mut f = good();
        f.rejections_with_cause = f.rejections - 1;
        let acc = Acceptance::evaluate(&f, &all_arms_pass());
        assert_eq!(acc.unmet(), vec!["causes and null gate"]);
        assert!(acc.clauses[0].observed.contains("159/160"), "{}", acc.clauses[0].observed);
    }

    /// A gate that cannot be told from a coin flip is not a gate, however
    /// many promotions it produced.
    #[test]
    fn a_gate_indistinguishable_from_the_null_arm_fails() {
        let mut f = good();
        f.null_gate_promoted = f.promoted - 1;
        assert!(!Acceptance::evaluate(&f, &all_arms_pass()).accepted());
    }

    /// The bar that can see one earlier episode collapse has to have been
    /// ARMED. A clean BWT with the bar off is a number the run did not
    /// actually check.
    #[test]
    fn a_clean_bwt_with_the_per_block_bar_off_is_not_retention() {
        let mut f = good();
        f.per_block_bar_armed = false;
        let acc = Acceptance::evaluate(&f, &all_arms_pass());
        assert_eq!(acc.unmet(), vec!["bwt and per-block bar"]);
        assert!(acc.clauses[2].observed.contains("OFF"), "{}", acc.clauses[2].observed);
    }

    /// Clause 5: zero is not a fast audit, it is an absent one.
    #[test]
    fn a_detection_latency_of_zero_is_an_absent_audit_not_an_instant_one() {
        let mut f = good();
        f.detection_latency = 0;
        assert_eq!(Acceptance::evaluate(&f, &all_arms_pass()).unmet(), vec!["detection latency"]);
    }

    /// Clause 8: an effect no larger than the run's own repeat noise is not
    /// an effect, and a run that does not reproduce cannot claim one at all.
    #[test]
    fn a_claim_inside_the_seed_repeat_noise_is_refused() {
        let mut f = good();
        f.seed_repeat_delta = 0.08;
        f.claimed_effect = 0.05;
        assert_eq!(Acceptance::evaluate(&f, &all_arms_pass()).unmet(), vec!["reproducible"]);

        let mut g = good();
        g.seed_repeat_identical = false;
        assert!(!Acceptance::evaluate(&g, &all_arms_pass()).accepted());
    }

    /// An audit that ate the run is a reader that stopped reading.
    #[test]
    fn an_audit_over_its_compute_budget_fails() {
        let mut f = good();
        f.eval_decodes = 600_000;
        let acc = Acceptance::evaluate(&f, &all_arms_pass());
        assert_eq!(acc.unmet(), vec!["eval compute share"]);
        assert!(acc.clauses[6].observed.contains("0.600"), "{}", acc.clauses[6].observed);
    }

    /// The report shows every clause, passed or not: a table listing only
    /// failures cannot be read as evidence that the rest were checked.
    #[test]
    fn the_table_shows_all_eight_clauses_either_way() {
        let mut f = good();
        f.detection_latency = 0;
        let t = Acceptance::evaluate(&f, &all_arms_pass()).table();
        assert_eq!(t.lines().count(), 8, "{t}");
        assert_eq!(t.lines().filter(|l| l.contains("FAIL")).count(), 1, "{t}");
        assert!(t.contains("PASS  independent battery"), "{t}");
    }
}

/// What a run's own ledger can answer about itself, and what it cannot.
///
/// Three of the eight clauses are questions about the episodes a run
/// recorded: whether every rejection carries a cause, how much of the
/// decode budget the audit took, and how long the reader takes to re-check
/// itself. Those are derivable from `ledger.jsonl` alone, which means they
/// survive a resumed run and cannot drift from what actually happened.
///
/// The rest are not in the ledger and are deliberately NOT defaulted here.
/// A null-gate count needs a second arm; a BWT needs the retention matrix; a
/// battery delta needs the held-out tasks scored twice; a seed spread needs
/// more than one run. Filling any of those with a plausible zero would turn
/// an unanswered clause into a passing one, which is the failure this whole
/// module exists to prevent - so [`LedgerFacts`] carries only what it knows
/// and the caller must supply the rest by name.
#[derive(Clone, Debug, PartialEq)]
pub struct LedgerFacts {
    pub episodes: usize,
    pub promoted: usize,
    pub rejections: usize,
    pub rejections_with_cause: usize,
    pub eval_decodes: u64,
}

impl LedgerFacts {
    /// Read them off the rows a run wrote.
    ///
    /// A "rejection" is any episode that was not promoted, at whatever stage
    /// stopped it: a screen refusal is as much a decision the run has to be
    /// able to explain as a gate refusal is.
    pub fn of(rows: &[crate::run::LedgerRow]) -> LedgerFacts {
        let promoted = rows.iter().filter(|r| r.promoted).count();
        let rejected: Vec<&crate::run::LedgerRow> = rows.iter().filter(|r| !r.promoted).collect();
        LedgerFacts {
            episodes: rows.len(),
            promoted,
            rejections: rejected.len(),
            rejections_with_cause: rejected.iter().filter(|r| r.cause.is_some()).count(),
            eval_decodes: rows.iter().map(|r| r.audit_decodes as u64).sum(),
        }
    }

    /// Episodes that were refused without saying why. Named rather than
    /// counted, because clause 1 failing is only actionable if a reader can
    /// go and look at the row.
    pub fn unexplained(rows: &[crate::run::LedgerRow]) -> Vec<&str> {
        rows.iter().filter(|r| !r.promoted && r.cause.is_none()).map(|r| r.source.as_str()).collect()
    }
}

#[cfg(test)]
mod ledger_tests {
    use super::*;
    use crate::run::LedgerRow;

    fn row(ep: u64, stage: &str, promoted: bool, cause: Option<&str>, audited: usize) -> LedgerRow {
        LedgerRow {
            episode: ep,
            id: format!("id{ep}"),
            source: format!("lane/doc{ep}.txt"),
            stage: stage.to_string(),
            promoted,
            carried: promoted,
            train_loss: None,
            cause: cause.map(str::to_string),
            audited: 2,
            audit_decodes: audited,
            diagnosis: None,
            action: None,
        }
    }

    /// The three clauses a run's own record can answer, answered from it -
    /// so they survive a resume and cannot drift from what happened.
    #[test]
    fn the_ledger_answers_the_clauses_that_are_about_its_own_episodes() {
        let rows = vec![
            row(1, "gate", true, None, 32),
            row(2, "screen", false, Some("no_structure"), 0),
            row(3, "reach", false, Some("already_known"), 0),
            row(4, "gate", false, Some("block_regressed"), 32),
        ];
        let f = LedgerFacts::of(&rows);
        assert_eq!(f.episodes, 4);
        assert_eq!(f.promoted, 1);
        assert_eq!(f.rejections, 3);
        assert_eq!(f.rejections_with_cause, 3, "every refusal named its cause");
        assert_eq!(f.eval_decodes, 64);
        assert!(LedgerFacts::unexplained(&rows).is_empty());
    }

    /// A refusal at ANY stage is a decision the run has to explain, not just
    /// a gate refusal - a screen that rejected a document silently is
    /// exactly as unaccountable.
    #[test]
    fn an_unexplained_refusal_is_named_whatever_stage_it_came_from() {
        let rows = vec![row(1, "gate", true, None, 0), row(2, "screen", false, None, 0)];
        let f = LedgerFacts::of(&rows);
        assert_eq!(f.rejections_with_cause, 0);
        assert_eq!(LedgerFacts::unexplained(&rows), vec!["lane/doc2.txt"], "clause 1 must be actionable: name the row");
    }

    /// The part that matters most: what the ledger cannot know is not
    /// filled in with a plausible zero. A `LedgerFacts` has no field for a
    /// null-gate count, a BWT or a battery, so a caller cannot accidentally
    /// accept a run on numbers nobody produced.
    #[test]
    fn the_ledger_does_not_invent_the_clauses_it_cannot_answer() {
        let rows = vec![row(1, "gate", true, None, 10)];
        let f = LedgerFacts::of(&rows);
        // The type carries five fields, and every one is a count of rows.
        // If this assertion ever needs updating because a null-gate or BWT
        // field appeared here, that field is being guessed rather than
        // measured.
        assert_eq!(
            f,
            LedgerFacts { episodes: 1, promoted: 1, rejections: 0, rejections_with_cause: 0, eval_decodes: 10 },
            "LedgerFacts must carry only what the rows actually say"
        );
    }
}
