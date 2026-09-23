// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a host that keeps ONE [`Registry`] for the life of the process and
//! runs many concurrent callers against it may actually rely on.
//!
//! Every assertion here is a contract a caller builds resource management on,
//! and every one of them was true only by convention before this file existed:
//! nothing prevented a future change from serializing dispatch, from turning a
//! cancelled generation into a silent truncation, or from renaming the error
//! string that another crate matches on. These tests use synthetic actions
//! rather than a model, so they pin the INTERFACE and run in milliseconds; the
//! matching model-side behaviour is pinned where the model lives.
//!
//! Swedish Embedded AB implements model-serving interfaces whose promises hold
//! under real concurrent traffic for its clients. If your team needs expertise
//! in turning an inference stack into a contract a product can be built on,
//! you can procure our services by sending an email to info@swedishembedded.com.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use capability::{Action, ActionResult, ActionSpec, Invocation, Outcome, Progress, Provider, Registry};
use serde_json::json;

/// A provider that answers with one fixed action object, so a test can prove
/// something about the object the registry hands out, not about a fresh clone.
struct One {
    model: &'static str,
    action: Arc<dyn Action>,
}

impl Provider for One {
    fn manifest(&self) -> capability::Manifest {
        capability::Manifest::new(self.model, "a synthetic model", vec![self.action.spec()])
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == self.action.spec().name).then(|| self.action.clone())
    }
}

fn registry_of(model: &'static str, action: Arc<dyn Action>) -> Registry {
    let mut r = Registry::new();
    r.register(Arc::new(One { model, action }));
    r
}

// ===================== 1. concurrency =====================

/// Two callers of the SAME action object are inside `run` at the same time:
/// the registry serializes nothing. A host may therefore not assume its
/// requests queue up somewhere below it - whatever an action touches, it
/// touches concurrently.
struct Rendezvous {
    in_flight: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}

impl Action for Rendezvous {
    fn spec(&self) -> ActionSpec {
        ActionSpec::new("rendezvous", "waits until a second caller joins it")
    }
    fn run(&self, _inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        // Wait on the PEAK, not the live count: the second caller may already
        // have left by the time the first one looks again. Bounded, so if
        // dispatch DID serialize this returns after the deadline with peak == 1
        // and the assertion below reports it, rather than hanging.
        let deadline = Instant::now() + Duration::from_secs(10);
        while self.peak.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(Outcome::new())
    }
}

#[test]
fn the_registry_runs_two_callers_of_one_model_concurrently() {
    let peak = Arc::new(AtomicUsize::new(0));
    let act = Arc::new(Rendezvous { in_flight: Arc::new(AtomicUsize::new(0)), peak: peak.clone() });
    let reg = Arc::new(registry_of("test/rendezvous", act));

    std::thread::scope(|s| {
        for _ in 0..2 {
            let reg = reg.clone();
            s.spawn(move || reg.run("test/rendezvous", "rendezvous", Invocation::new(), &mut |_| {}).unwrap());
        }
    });

    assert_eq!(peak.load(Ordering::SeqCst), 2, "Registry::run must not serialize its callers; an action is responsible for its own shared state");
}

/// The other half, and the pattern every model in this workspace uses: an
/// action that owns resident state behind a mutex serializes its callers, and
/// their runs never interleave. This is what makes one loaded model safe to
/// share - and it is also why a second caller sees the first one's full
/// latency.
struct Resident {
    slot: Arc<Mutex<Option<i64>>>,
    log: Arc<Mutex<Vec<(i64, &'static str)>>>,
}

impl Action for Resident {
    fn spec(&self) -> ActionSpec {
        ActionSpec::new("resident", "holds resident state across calls")
            .param(capability::ParamSpec::new("id", capability::ParamType::Int, "caller id").required())
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let id = inv.get_i64("id").expect("validated");
        let mut slot = capability::lock_resident(&self.slot);
        self.log.lock().unwrap().push((id, "enter"));
        std::thread::sleep(Duration::from_millis(20));
        *slot = Some(id);
        self.log.lock().unwrap().push((id, "exit"));
        Ok(Outcome::new().set("id", json!(id)))
    }
}

#[test]
fn a_model_that_guards_its_resident_state_never_interleaves_two_runs() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let act = Arc::new(Resident { slot: Arc::new(Mutex::new(None)), log: log.clone() });
    let reg = Arc::new(registry_of("test/resident", act));

    std::thread::scope(|s| {
        for id in 0..4i64 {
            let reg = reg.clone();
            s.spawn(move || {
                let out = reg.run("test/resident", "resident", Invocation::new().set("id", json!(id)), &mut |_| {}).unwrap();
                assert_eq!(out.outputs["id"], json!(id));
            });
        }
    });

    let log = log.lock().unwrap();
    assert_eq!(log.len(), 8);
    for pair in log.chunks(2) {
        assert_eq!(pair[0].1, "enter");
        assert_eq!(pair[1].1, "exit");
        assert_eq!(pair[0].0, pair[1].0, "a run began before the previous one finished: {log:?}");
    }
}

// ===================== 2. a panic must not brick the model =====================

/// The hazard, pinned so it stays visible: a resident slot taken with a plain
/// `lock()` is poisoned by ONE panicking request and every later caller of that
/// model fails forever, in a process that is otherwise healthy.
struct PlainLock {
    slot: Arc<Mutex<Option<i64>>>,
}

impl Action for PlainLock {
    fn spec(&self) -> ActionSpec {
        ActionSpec::new("run", "a resident slot taken with a plain lock()")
            .param(capability::ParamSpec::new("boom", capability::ParamType::Bool, "panic while holding the lock").default(json!(false)))
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let mut slot = self.slot.lock().map_err(|_| "resident lock poisoned".to_string())?;
        assert!(!inv.get_bool("boom").unwrap_or(false), "synthetic backend fault");
        *slot = Some(1);
        Ok(Outcome::new())
    }
}

/// Same action, recovering the slot instead ([`capability::lock_resident`]).
struct RecoveredLock {
    slot: Arc<Mutex<Option<i64>>>,
    rebuilds: Arc<AtomicUsize>,
}

impl Action for RecoveredLock {
    fn spec(&self) -> ActionSpec {
        ActionSpec::new("run", "a resident slot that survives a panicking request")
            .param(capability::ParamSpec::new("boom", capability::ParamType::Bool, "panic while holding the lock").default(json!(false)))
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let mut slot = capability::lock_resident(&self.slot);
        if slot.is_none() {
            self.rebuilds.fetch_add(1, Ordering::SeqCst); // stands in for "load the weights"
            *slot = Some(1);
        }
        assert!(!inv.get_bool("boom").unwrap_or(false), "synthetic backend fault");
        Ok(Outcome::new())
    }
}

/// Run `model`'s `run` action, swallowing a panic. The panic message this
/// provokes is EXPECTED test output.
fn run_catching(reg: &Registry, model: &str, boom: bool) -> Option<ActionResult> {
    let inv = Invocation::new().set("boom", json!(boom));
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| reg.run(model, "run", inv, &mut |_| {}))).ok()
}

#[test]
fn a_panicking_request_bricks_a_plainly_locked_resident_for_the_whole_process() {
    let reg = registry_of("test/plain", Arc::new(PlainLock { slot: Arc::new(Mutex::new(None)) }));
    assert!(run_catching(&reg, "test/plain", false).expect("no panic").is_ok());
    assert!(run_catching(&reg, "test/plain", true).is_none(), "the action must have panicked");
    let err = run_catching(&reg, "test/plain", false).expect("no panic").unwrap_err();
    assert!(err.contains("poisoned"), "one panic must not be survivable by accident, got: {err}");
}

#[test]
fn lock_resident_discards_the_resident_after_a_panic_and_serves_the_next_caller() {
    let rebuilds = Arc::new(AtomicUsize::new(0));
    let reg = registry_of("test/recovered", Arc::new(RecoveredLock { slot: Arc::new(Mutex::new(None)), rebuilds: rebuilds.clone() }));
    assert!(run_catching(&reg, "test/recovered", false).expect("no panic").is_ok());
    assert_eq!(rebuilds.load(Ordering::SeqCst), 1, "first call loads");
    assert!(run_catching(&reg, "test/recovered", true).is_none(), "the action must have panicked");
    assert!(run_catching(&reg, "test/recovered", false).expect("no panic").is_ok(), "the model must still serve after a panicking request");
    assert_eq!(rebuilds.load(Ordering::SeqCst), 2, "the resident touched by the panic is discarded and rebuilt, never reused");
}

// ===================== 3. cancellation and time bounds =====================

/// A long action that polls the token between steps, as the contract requires.
struct Stepper {
    step_ms: u64,
    steps_run: Arc<AtomicUsize>,
}

impl Action for Stepper {
    fn spec(&self) -> ActionSpec {
        ActionSpec::new("step", "polls the cancel token between steps").streaming()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        const TOTAL: u32 = 10_000;
        for step in 0..TOTAL {
            if inv.cancel.is_cancelled() {
                return Err("cancelled".into());
            }
            // One indivisible step: the token is NOT observed inside it.
            std::thread::sleep(Duration::from_millis(self.step_ms));
            self.steps_run.fetch_add(1, Ordering::SeqCst);
            progress(Progress::step(step + 1, TOTAL, "step"));
        }
        Ok(Outcome::new())
    }
}

/// A deadline is expressible, and only through the cancel token: `max_new`-style
/// work bounds say nothing about wall-clock time, so a caller that needs one
/// arms a token and fires it from a timer. This is that recipe, end to end.
#[test]
fn a_timer_firing_the_cancel_token_is_the_only_wall_clock_bound_there_is() {
    let steps_run = Arc::new(AtomicUsize::new(0));
    let reg = registry_of("test/stepper", Arc::new(Stepper { step_ms: 2, steps_run: steps_run.clone() }));

    let mut inv = Invocation::new();
    inv.cancel = capability::CancelToken::armed();
    let timer = inv.cancel.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        timer.cancel();
    });

    let start = Instant::now();
    let err = reg.run("test/stepper", "step", inv, &mut |_| {}).unwrap_err();
    let elapsed = start.elapsed();

    assert_eq!(err, "cancelled", "a cancelled action reports cancellation, never a truncated success");
    assert!(elapsed < Duration::from_secs(10), "the whole run is 20s of steps; a fired token must end it promptly, took {elapsed:?}");
    assert!(steps_run.load(Ordering::SeqCst) < 10_000, "the action must stop early, not run to completion into a caller that has gone away");
}

/// And the boundary that recipe does NOT cross: the token is observed BETWEEN
/// steps, so a generation already inside one runs that step to completion no
/// matter when the token fires. On a real model a step is a forward pass, and
/// a forward pass that wedges in a driver call is below anything this interface
/// can interrupt.
#[test]
fn a_fired_token_does_not_shorten_a_step_already_running() {
    let reg = registry_of("test/slowstep", Arc::new(Stepper { step_ms: 300, steps_run: Arc::new(AtomicUsize::new(0)) }));

    let mut inv = Invocation::new();
    inv.cancel = capability::CancelToken::armed();
    let token = inv.cancel.clone();

    let start = Instant::now();
    let err = reg
        .run("test/slowstep", "step", inv, &mut |_| {
            token.cancel(); // fired the instant the first step completes
        })
        .unwrap_err();
    let elapsed = start.elapsed();

    assert_eq!(err, "cancelled");
    assert!(elapsed >= Duration::from_millis(300), "cancellation cannot pre-empt a step in flight; it only prevents the next one (took {elapsed:?})");
}

// ===================== 4. failure =====================

/// An action that fails on request, otherwise succeeds.
struct Flaky;

impl Action for Flaky {
    fn spec(&self) -> ActionSpec {
        ActionSpec::new("flaky", "fails when asked to")
            .param(capability::ParamSpec::new("fail", capability::ParamType::Bool, "fail this call").default(json!(false)))
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if inv.get_bool("fail").unwrap_or(false) {
            return Err("qwen: needs 9000 tokens, engine capacity is 4096".into());
        }
        Ok(Outcome::new().set("ok", json!(true)))
    }
}

#[test]
fn a_failed_call_leaves_the_model_usable_for_the_next_caller() {
    let reg = registry_of("test/flaky", Arc::new(Flaky));
    assert!(reg.run("test/flaky", "flaky", Invocation::new().set("fail", json!(true)), &mut |_| {}).is_err());
    let out = reg.run("test/flaky", "flaky", Invocation::new(), &mut |_| {}).unwrap();
    assert_eq!(out.outputs["ok"], json!(true), "the registry holds no per-call state, so an error cannot wedge it");
}

/// The error type is `String`, so these exact wordings ARE the interface: brain's
/// own HTTP layer turns one of them into a 404 and another into a 400 by matching
/// the prose. Until [`capability::ActionResult`] carries a kind, renaming any of
/// them is a silent breaking change - this test is the tripwire.
#[test]
fn the_error_strings_other_crates_match_on_are_part_of_the_interface() {
    let reg = registry_of("test/flaky", Arc::new(Flaky));

    // apiserve::bridge::map_reply_err -> model_not_found (404)
    let unknown_model = reg.run("test/absent", "flaky", Invocation::new(), &mut |_| {}).unwrap_err();
    assert_eq!(unknown_model, "no action 'flaky' on model 'test/absent'");
    let unknown_action = reg.run("test/flaky", "absent", Invocation::new(), &mut |_| {}).unwrap_err();
    assert_eq!(unknown_action, "no action 'absent' on model 'test/flaky'");
    for e in [&unknown_model, &unknown_action] {
        assert!(e.starts_with("no model") || e.contains("no action"), "apiserve's model_not_found test, verbatim");
    }

    // A rejected request and a failed one are indistinguishable by TYPE: both are
    // Err(String). Only the prose separates "you sent something invalid" from
    // "the device fell over", which is why the gap is worth naming.
    let invalid = reg.run("test/flaky", "flaky", Invocation::new().set("fail", json!("yes")), &mut |_| {}).unwrap_err();
    assert_eq!(invalid, "param 'fail' must be bool (got \"yes\")");

    // apiserve::bridge::parse_exceeds_capacity -> context_length_exceeded (400)
    let over = reg.run("test/flaky", "flaky", Invocation::new().set("fail", json!(true)), &mut |_| {}).unwrap_err();
    let rest = over.split_once("needs ").expect("the phrase apiserve parses").1;
    let (need, capacity) = rest.split_once(" tokens, engine capacity is ").expect("the phrase apiserve parses");
    assert_eq!((need.parse::<u64>().unwrap(), capacity.trim().parse::<u64>().unwrap()), (9000, 4096));
}
