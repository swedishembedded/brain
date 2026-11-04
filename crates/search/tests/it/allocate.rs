// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where the next unit of search budget should be spent.
//!
//! A campaign has several ways to produce a candidate and no way to know in
//! advance which of them is currently worth running - and the answer moves as
//! the search progresses. These are the properties that make the allocator an
//! answer to that rather than a fixed schedule with extra steps.

use search::allocate::{Allocator, Gain};

#[test]
fn every_operator_is_tried_before_any_is_repeated() {
    let mut a = Allocator::new(&["one", "two", "three"]);
    let mut seen = [false; 3];
    for _ in 0..3 {
        let arm = a.choose();
        assert!(!seen[arm], "an operator was repeated before all had been tried");
        seen[arm] = true;
        a.credit(arm, Gain { fresh: 0.0, improved: 0.0, seconds: 1.0 });
    }
    assert!(seen.iter().all(|s| *s));
}

/// The point of the thing. An operator that keeps finding new niches gets
/// more of the budget than one that finds nothing.
#[test]
fn a_productive_operator_earns_more_of_the_budget() {
    let mut a = Allocator::new(&["good", "useless"]);
    let mut picks = [0u32; 2];
    for _ in 0..600 {
        let arm = a.choose();
        picks[arm] += 1;
        let gain = match arm {
            0 => Gain { fresh: 3.0, improved: 1.0, seconds: 1.0 },
            _ => Gain { fresh: 0.0, improved: 0.0, seconds: 1.0 },
        };
        a.credit(arm, gain);
    }
    assert!(picks[0] > picks[1] * 3, "the productive arm got {} against {}", picks[0], picks[1]);
}

/// ...and the unproductive one is never permanently cut off, because what an
/// operator is worth changes as the archive fills. An arm that has paid
/// nothing for five hundred rounds may be the only one that can open the next
/// region.
#[test]
fn an_unproductive_operator_is_still_tried_occasionally() {
    let mut a = Allocator::new(&["good", "useless"]);
    let mut picks = [0u32; 2];
    for _ in 0..600 {
        let arm = a.choose();
        picks[arm] += 1;
        let gain = match arm {
            0 => Gain { fresh: 3.0, improved: 1.0, seconds: 1.0 },
            _ => Gain { fresh: 0.0, improved: 0.0, seconds: 1.0 },
        };
        a.credit(arm, gain);
    }
    assert!(picks[1] > 5, "the unproductive arm was starved: {} draws in 600", picks[1]);
}

/// The metric is gain per SECOND, not gain per call. Two operators finding
/// equally much are not equally good if one takes twenty times as long, and
/// an allocator that cannot see that spends a campaign on the expensive one.
#[test]
fn credit_is_per_second_not_per_call() {
    let mut a = Allocator::new(&["quick", "slow"]);
    let mut picks = [0u32; 2];
    for _ in 0..600 {
        let arm = a.choose();
        picks[arm] += 1;
        // Identical gain; the slow one takes twenty times as long to get it.
        let seconds = if arm == 0 { 1.0 } else { 20.0 };
        a.credit(arm, Gain { fresh: 2.0, improved: 0.0, seconds });
    }
    assert!(picks[0] > picks[1] * 3, "the quick arm got {} against {}", picks[0], picks[1]);
}

/// The defect a real campaign exposed. An operator that produces a great many
/// SHALLOW cells must not outrank one producing fewer DEEP ones.
///
/// Measured on `samples/decision/doom`: a wall-pressing operator generated
/// 44 513 steps of cheap cells against a playing operator's 23 272, scored a
/// comparable gain rate on a metric that counted every fresh niche as one,
/// and took a third of the budget while the playing operator was the one
/// actually advancing the frontier. Weighting an admission by how far along
/// it is, relative to the best the archive holds, is what tells them apart.
#[test]
fn a_deep_niche_outranks_a_shallow_one() {
    let deep = Gain { fresh: 1.0, improved: 0.0, seconds: 1.0 };
    let shallow = Gain { fresh: 4.0 * 0.05, improved: 0.0, seconds: 1.0 };
    assert!(
        deep.value() > shallow.value(),
        "four cells at the spawn ({:.3}) outranked one at the frontier ({:.3})",
        shallow.value(),
        deep.value()
    );
}

/// An improved elite is worth something - it is how a time comes down once
/// coverage has stopped growing - but a fresh niche is worth more, because
/// that is the search reaching somewhere it has never been.
#[test]
fn a_fresh_niche_outranks_an_improved_one() {
    let quicker = Gain { fresh: 1.0, improved: 0.0, seconds: 1.0 };
    let refiner = Gain { fresh: 0.0, improved: 1.0, seconds: 1.0 };
    assert!(quicker.value() > refiner.value());
    assert!(refiner.value() > Gain { fresh: 0.0, improved: 0.0, seconds: 1.0 }.value());
}

/// A zero-second measurement is a clock resolution artifact, not an infinitely
/// productive operator. Left unhandled it divides by zero and that arm takes
/// the entire remaining budget.
#[test]
fn a_zero_second_call_does_not_produce_an_infinite_rate() {
    let mut a = Allocator::new(&["instant", "normal"]);
    a.credit(0, Gain { fresh: 1.0, improved: 0.0, seconds: 0.0 });
    a.credit(1, Gain { fresh: 1.0, improved: 0.0, seconds: 1.0 });
    let rates = a.report();
    assert!(rates[0].rate.is_finite(), "a zero-second call produced {}", rates[0].rate);
}

/// What the campaign prints, so a human can see where the budget went and
/// whether the allocation was sensible.
#[test]
fn the_report_names_every_arm_with_its_draws_and_rate() {
    let mut a = Allocator::new(&["alpha", "beta"]);
    a.credit(0, Gain { fresh: 4.0, improved: 0.0, seconds: 2.0 });
    a.credit(1, Gain { fresh: 0.0, improved: 0.0, seconds: 1.0 });
    let r = a.report();
    assert_eq!(r.len(), 2);
    assert_eq!(r[0].name, "alpha");
    assert_eq!(r[0].draws, 1);
    assert!(r[0].rate > r[1].rate);
    assert_eq!(r[0].seconds, 2.0);
}
