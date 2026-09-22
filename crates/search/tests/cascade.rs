// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The evaluator cascade: cheap checks first, and the expensive one only for
//! what survived them.
//!
//! The property that matters is negative - the expensive rung must NOT run on
//! a candidate a cheap rung already refused. A cascade that evaluates
//! everything at full fidelity is a cascade in name only, and on this campaign
//! the top rung is a whole replayed episode against a real game engine.

use search::cascade::{Cascade, Rung, Verdict};

struct Counting {
    name: &'static str,
    calls: std::rc::Rc<std::cell::Cell<usize>>,
    verdict: Verdict,
}

impl Rung<i32> for Counting {
    fn name(&self) -> &str {
        self.name
    }
    fn check(&mut self, _c: &i32) -> Verdict {
        self.calls.set(self.calls.get() + 1);
        self.verdict.clone()
    }
}

fn counter() -> std::rc::Rc<std::cell::Cell<usize>> {
    std::rc::Rc::new(std::cell::Cell::new(0))
}

#[test]
fn a_candidate_refused_cheaply_never_reaches_the_expensive_rung() {
    let (cheap, dear) = (counter(), counter());
    let mut c: Cascade<i32> = Cascade::new();
    c.push(Box::new(Counting {
        name: "cheap",
        calls: cheap.clone(),
        verdict: Verdict::Reject("not worth it".into()),
    }));
    c.push(Box::new(Counting { name: "dear", calls: dear.clone(), verdict: Verdict::Pass }));
    assert!(matches!(c.admit(&1), Verdict::Reject(_)));
    assert_eq!(cheap.get(), 1);
    assert_eq!(dear.get(), 0, "the expensive rung ran on a candidate already refused");
}

#[test]
fn a_candidate_passing_every_rung_is_admitted() {
    let (a, b) = (counter(), counter());
    let mut c: Cascade<i32> = Cascade::new();
    c.push(Box::new(Counting { name: "cheap", calls: a.clone(), verdict: Verdict::Pass }));
    c.push(Box::new(Counting { name: "dear", calls: b.clone(), verdict: Verdict::Pass }));
    assert!(matches!(c.admit(&1), Verdict::Pass));
    assert_eq!((a.get(), b.get()), (1, 1));
}

/// Where candidates die is the diagnostic. A cascade whose top rung never
/// fires is one whose cheap rungs are too strict, and a cascade where nothing
/// is refused cheaply is one that is paying full price for everything - and
/// neither is visible without this.
#[test]
fn the_cascade_reports_where_candidates_died() {
    let mut c: Cascade<i32> = Cascade::new();
    let refused = counter();
    let passed = counter();
    c.push(Box::new(Counting { name: "cheap", calls: passed.clone(), verdict: Verdict::Pass }));
    c.push(Box::new(Counting {
        name: "dear",
        calls: refused.clone(),
        verdict: Verdict::Reject("replay disagreed".into()),
    }));
    for i in 0..5 {
        let _ = c.admit(&i);
    }
    let tally = c.tally();
    assert_eq!(tally.len(), 2);
    assert_eq!(tally[0].name, "cheap");
    assert_eq!(tally[0].seen, 5);
    assert_eq!(tally[0].refused, 0);
    assert_eq!(tally[1].name, "dear");
    assert_eq!(tally[1].seen, 5);
    assert_eq!(tally[1].refused, 5);
    assert_eq!(c.admitted(), 0);
}

/// An empty cascade admits, rather than refusing everything. A caller that
/// has not configured a verifier yet gets an unverified search, not a search
/// that silently finds nothing.
#[test]
fn an_empty_cascade_admits() {
    let mut c: Cascade<i32> = Cascade::new();
    assert!(matches!(c.admit(&1), Verdict::Pass));
    assert_eq!(c.admitted(), 1);
}
