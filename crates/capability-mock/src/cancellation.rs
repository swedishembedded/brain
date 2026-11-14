// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The behavioural certification of [`capability::CancelToken`]'s contract,
//! one helper every model's tests call for every action they can run.
//!
//! Swedish Embedded AB implements solutions for long-running inference and
//! training jobs that must stop when their caller goes away. If your team
//! needs expertise in cooperative cancellation across model serving stacks
//! then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! A run under test is anything with an action's shape: an [`Action::run`]
//! itself, or a model's library entry point adapted to it (a tiny random
//! config driving the same loop the served action drives). Three checks:
//!
//! * [`admission`]: a token fired before the call returns
//!   `Err(`[`CANCELLED`]`)` and emits no progress, so no work ran.
//! * [`mid_run`]: a token fired at the `fire_after`-th progress event stops
//!   the run within `within` more events, ending either in
//!   `Err(`[`CANCELLED`]`)` or, for a run that already streamed text (`delta`
//!   progress), in `Ok` with `finish_reason: "cancelled"`. The same run
//!   without cancellation must emit more than `fire_after + within` events,
//!   so the check cannot pass on a workload too short to prove anything.
//! * [`certify`]: both, against one run.
//!
//! Each helper panics with the subject's name and what it observed, the way
//! an `assert!` in the caller's test would.
//!
//! [`Action::run`]: capability::Action::run

use capability::failure::CANCELLED;
use capability::{ActionResult, CancelToken, Invocation, Progress};

/// A run under test: an action's `run`, or a library loop in its shape.
pub type Run<'a> = &'a dyn Fn(&Invocation, &mut dyn FnMut(Progress)) -> ActionResult;

/// Where [`mid_run`] fires the token and how much further the run may go.
#[derive(Clone, Copy, Debug)]
pub struct Bound {
    /// Fire the token from inside the callback for this progress event
    /// (1-based), the way a client cancelling over a stream does.
    pub fire_after: u32,
    /// The most progress events the run may emit after the token fired: the
    /// granularity its polling promises (one decode token, one denoise step).
    pub within: u32,
}

impl Bound {
    /// Fire on the first event; at most one more may follow (the step that
    /// was already in flight when the token fired).
    pub const ONE_STEP: Bound = Bound { fire_after: 1, within: 1 };
}

/// What [`mid_run`] observed.
#[derive(Clone, Debug)]
pub struct MidRun {
    /// Progress events the uncancelled reference run emitted.
    pub reference_events: u32,
    /// Progress events after the token fired.
    pub events_after_cancel: u32,
    /// `delta` progress events after the token fired.
    pub deltas_after_cancel: u32,
}

/// Cancel before admission: a fired token stops the run before it emits
/// anything, and the run answers with the contract's literal.
pub fn admission(subject: &str, inv: Invocation, run: Run<'_>) {
    let inv = fired(inv);
    let mut events = 0u32;
    let result = run(&inv, &mut |_| events += 1);
    match result {
        Err(e) if e == CANCELLED => {}
        Err(e) => panic!("{subject}: a call whose token fired before admission must fail with {CANCELLED:?}, got Err({e:?})"),
        Ok(o) => panic!("{subject}: a call whose token fired before admission must fail with {CANCELLED:?}, it ran to Ok({:?})", o.outputs),
    }
    assert_eq!(events, 0, "{subject}: a call cancelled before admission emitted {events} progress event(s): it ran work it should have refused");
}

/// Cancel mid-run: fire the token from the `bound.fire_after`-th progress
/// event and check the run stops within `bound.within` more, with the
/// contract's terminal shape. `make_inv` builds a fresh invocation for each
/// of the two runs (the reference and the cancelled one).
pub fn mid_run(subject: &str, make_inv: &dyn Fn() -> Invocation, run: Run<'_>, bound: Bound) -> MidRun {
    assert!(bound.fire_after >= 1, "{subject}: Bound::fire_after counts progress events from 1");
    let mut reference_events = 0u32;
    let reference = make_inv();
    let reference = Invocation { cancel: CancelToken::default(), ..reference };
    if let Err(e) = run(&reference, &mut |_| reference_events += 1) {
        panic!("{subject}: the uncancelled reference run failed: {e}");
    }
    assert!(
        reference_events > bound.fire_after + bound.within,
        "{subject}: the uncancelled run emits only {reference_events} progress event(s), too few to show a cancel at event {} stops within {}: size the workload up",
        bound.fire_after,
        bound.within
    );

    let token = CancelToken::armed();
    let inv = Invocation { cancel: token.clone(), ..make_inv() };
    let (mut events, mut after, mut deltas_after, mut deltas_before) = (0u32, 0u32, 0u32, 0u32);
    let result = run(&inv, &mut |p| {
        events += 1;
        if token.is_cancelled() {
            after += 1;
            deltas_after += u32::from(p.delta.is_some());
        } else {
            deltas_before += u32::from(p.delta.is_some());
        }
        if events == bound.fire_after {
            token.cancel();
        }
    });
    assert!(token.is_cancelled(), "{subject}: the run emitted only {events} progress event(s), never reaching event {}", bound.fire_after);
    assert!(
        after <= bound.within,
        "{subject}: {after} progress event(s) after cancellation ({deltas_after} of them streamed text), the bound is {}: the run does not poll its token per step",
        bound.within
    );
    match result {
        Err(e) if e == CANCELLED => {}
        Err(e) => panic!("{subject}: a cancelled run must fail with {CANCELLED:?}, got Err({e:?})"),
        Ok(o) => {
            let reason = o.outputs.get("finish_reason").and_then(|r| r.as_str());
            assert_eq!(reason, Some("cancelled"), "{subject}: a cancelled run that returns Ok must report finish_reason \"cancelled\", got {:?}", o.outputs);
            assert!(deltas_before + deltas_after > 0, "{subject}: only a run that streamed partial text may complete Ok on cancellation; this one streamed none");
        }
    }
    MidRun { reference_events, events_after_cancel: after, deltas_after_cancel: deltas_after }
}

/// [`admission`] and [`mid_run`] against one run.
pub fn certify(subject: &str, make_inv: &dyn Fn() -> Invocation, run: Run<'_>, bound: Bound) -> MidRun {
    admission(subject, make_inv(), run);
    mid_run(subject, make_inv, run, bound)
}

/// `inv` carrying a token that has already fired.
fn fired(inv: Invocation) -> Invocation {
    let token = CancelToken::armed();
    token.cancel();
    Invocation { cancel: token, ..inv }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Outcome;
    use serde_json::json;

    fn polling(steps: u32, stream: bool) -> impl Fn(&Invocation, &mut dyn FnMut(Progress)) -> ActionResult {
        move |inv, progress| {
            inv.cancel.check()?;
            let mut text = String::new();
            for i in 0..steps {
                if inv.cancel.is_cancelled() {
                    return if stream && !text.is_empty() { Ok(Outcome::new().set("finish_reason", json!("cancelled"))) } else { inv.cancel.check().map(|_| Outcome::new()) };
                }
                if stream {
                    text.push('x');
                    progress(Progress::token(i, steps, "x"));
                } else {
                    progress(Progress::step(i, steps, ""));
                }
            }
            Ok(Outcome::new().set("finish_reason", json!("stop")))
        }
    }

    fn ignoring(steps: u32) -> impl Fn(&Invocation, &mut dyn FnMut(Progress)) -> ActionResult {
        move |_inv, progress| {
            for i in 0..steps {
                progress(Progress::step(i, steps, ""));
            }
            Ok(Outcome::new())
        }
    }

    fn panics(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
        let err = std::panic::catch_unwind(f).expect_err("the certification must reject this run");
        err.downcast_ref::<String>().cloned().unwrap_or_default()
    }

    #[test]
    fn a_polling_run_is_certified_in_both_terminal_shapes() {
        let report = certify("step", &Invocation::new, &polling(20, false), Bound::ONE_STEP);
        assert_eq!(report.reference_events, 20);
        assert_eq!(report.events_after_cancel, 0);
        let report = certify("stream", &Invocation::new, &polling(20, true), Bound { fire_after: 3, within: 1 });
        assert_eq!(report.deltas_after_cancel, 0);
    }

    #[test]
    fn a_run_that_ignores_its_token_is_rejected_at_admission_and_mid_run() {
        let msg = panics(|| admission("deaf", Invocation::new(), &ignoring(20)));
        assert!(msg.contains("deaf") && msg.contains("ran to Ok"), "{msg}");
        let msg = panics(|| {
            mid_run("deaf", &Invocation::new, &ignoring(20), Bound::ONE_STEP);
        });
        assert!(msg.contains("after cancellation"), "{msg}");
    }

    #[test]
    fn a_workload_too_short_to_prove_a_stop_is_refused() {
        let msg = panics(|| {
            mid_run("short", &Invocation::new, &polling(2, false), Bound::ONE_STEP);
        });
        assert!(msg.contains("too few"), "{msg}");
    }

    #[test]
    fn a_cancelled_run_may_not_claim_a_finished_answer() {
        let lying = |inv: &Invocation, progress: &mut dyn FnMut(Progress)| -> ActionResult {
            inv.cancel.check()?;
            for i in 0..10 {
                if inv.cancel.is_cancelled() {
                    break;
                }
                progress(Progress::token(i, 10, "x"));
            }
            Ok(Outcome::new().set("finish_reason", json!("stop")))
        };
        let msg = panics(|| {
            mid_run("lying", &Invocation::new, &lying, Bound::ONE_STEP);
        });
        assert!(msg.contains("finish_reason"), "{msg}");
    }
}
