// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared front-door admission policy: the edge concurrency ceiling and the
//! per-request admit deadline every transport applies to a submitted [`Job`]
//! before it starts running on a lane. Previously duplicated between
//! `apiserve` (HTTP) and `dbus` — factored here so both transports gate
//! identically instead of one drifting from the other.
//!
//! [`Job`]: crate::executor::Job

use std::time::Duration;

use crate::Executor;

/// The concurrency ceiling at the edge: at most this many requests are
/// admitted to a transport at once. Overflow is load-shed fast rather than
/// queued, so a saturated server sheds instead of building unbounded latency.
/// Well above the executor's lane count — this guards the transport edge, not
/// the lanes.
pub const EDGE_CONCURRENCY: usize = 256;

/// Default bounded wait for a request to be ADMITTED (work started on a lane)
/// before it is shed. A running job may then take much longer — only the
/// wait-to-start is bounded.
pub const DEFAULT_ADMIT_DEADLINE: Duration = Duration::from_secs(10);

/// The bounded wait applied INSTEAD of [`DEFAULT_ADMIT_DEADLINE`] when a
/// request is queued behind a job for its OWN model that is already running
/// (see [`model_has_running_job`]) — almost always that model's first-ever
/// cold activation (WGSL pipeline compile + weight upload from disk), which
/// on modest hardware measurably takes over a minute and has nothing to do
/// with the server being overloaded. `DEFAULT_ADMIT_DEADLINE` still applies,
/// unchanged, to every other reason a request isn't yet admitted (all lanes
/// genuinely busy with unrelated models, nothing evictable) — that case keeps
/// shedding fast, on purpose.
pub const DEFAULT_COLD_BUILD_ADMIT_DEADLINE: Duration = Duration::from_secs(180);

/// The admission deadline for live servers: `BRAIN_ADMIT_DEADLINE_MS` if set
/// to a positive integer, else [`DEFAULT_ADMIT_DEADLINE`]. An empty/invalid/
/// zero value falls back to the default (never an unbounded or zero-length
/// wait). Shared by every transport so an operator sets it once.
pub fn admit_deadline_from_env() -> Duration {
    parse_deadline_ms(std::env::var("BRAIN_ADMIT_DEADLINE_MS").ok().as_deref(), DEFAULT_ADMIT_DEADLINE)
}

/// The cold-build admission deadline for live servers: `BRAIN_COLD_BUILD_ADMIT_DEADLINE_MS`
/// if set to a positive integer, else [`DEFAULT_COLD_BUILD_ADMIT_DEADLINE`].
pub fn cold_build_admit_deadline_from_env() -> Duration {
    parse_deadline_ms(std::env::var("BRAIN_COLD_BUILD_ADMIT_DEADLINE_MS").ok().as_deref(), DEFAULT_COLD_BUILD_ADMIT_DEADLINE)
}

/// Pure parse of a `*_MS` deadline override: a positive integer of
/// milliseconds, or `default` for any missing/empty/invalid/zero value.
pub fn parse_deadline_ms(raw: Option<&str>, default: Duration) -> Duration {
    match raw.and_then(|v| v.trim().parse::<u64>().ok()).filter(|&ms| ms > 0) {
        Some(ms) => Duration::from_millis(ms),
        None => default,
    }
}

/// Pure parse of the admit-deadline override — kept for callers/tests already
/// naming it; equivalent to `parse_deadline_ms(raw, DEFAULT_ADMIT_DEADLINE)`.
pub fn parse_admit_deadline(raw: Option<&str>) -> Duration {
    parse_deadline_ms(raw, DEFAULT_ADMIT_DEADLINE)
}

/// Is `model`'s currently-running group still inside its deferred
/// activate()/promote() — i.e. genuinely cold-building, not just busy? A
/// caller stuck waiting for admission uses this to tell "my own model's
/// first cold build is still in flight" (same-key jobs serialize onto one
/// lane, see `executor::group_rows`) apart from every OTHER reason it isn't
/// admitted yet — including a normal, already-warm job for the SAME model
/// that's simply taking a while (that's genuine same-model contention, not a
/// cold start, and must keep shedding fast). Only the former deserves
/// [`DEFAULT_COLD_BUILD_ADMIT_DEADLINE`]'s much longer wait; everything else
/// keeps [`DEFAULT_ADMIT_DEADLINE`]'s fast shed. Checks
/// `InFlightJob::phase == "building"` specifically (flips to `"running"` the
/// instant activate()/promote() finishes, however long the job runs after
/// that — see `executor::Msg::Built`'s handling). A blocking round-trip to
/// the dispatcher thread (`Executor::in_flight`) — cheap (an in-process
/// channel call), but callers should still only reach for it AFTER the short
/// deadline has already elapsed, not on every poll tick.
pub fn model_is_cold_building(exec: &Executor, model: &str) -> bool {
    exec.in_flight().iter().any(|j| j.model == model && j.phase == "building")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admit_deadline_override_parses_positive_ms_else_default() {
        assert_eq!(parse_admit_deadline(Some("500")), Duration::from_millis(500));
        assert_eq!(parse_admit_deadline(Some("  250 ")), Duration::from_millis(250));
        assert_eq!(parse_admit_deadline(None), DEFAULT_ADMIT_DEADLINE);
        assert_eq!(parse_admit_deadline(Some("")), DEFAULT_ADMIT_DEADLINE);
        assert_eq!(parse_admit_deadline(Some("0")), DEFAULT_ADMIT_DEADLINE);
        assert_eq!(parse_admit_deadline(Some("nope")), DEFAULT_ADMIT_DEADLINE);
    }

    #[test]
    fn cold_build_deadline_override_parses_positive_ms_else_its_own_default() {
        assert_eq!(parse_deadline_ms(Some("90000"), DEFAULT_COLD_BUILD_ADMIT_DEADLINE), Duration::from_millis(90000));
        assert_eq!(parse_deadline_ms(None, DEFAULT_COLD_BUILD_ADMIT_DEADLINE), DEFAULT_COLD_BUILD_ADMIT_DEADLINE);
        assert_eq!(parse_deadline_ms(Some("0"), DEFAULT_COLD_BUILD_ADMIT_DEADLINE), DEFAULT_COLD_BUILD_ADMIT_DEADLINE);
        assert!(DEFAULT_COLD_BUILD_ADMIT_DEADLINE > DEFAULT_ADMIT_DEADLINE, "a cold build must get strictly more time than the fast-shed default");
    }

    /// A [`ResidentModel`] whose `activate()` blocks on a barrier (simulating a
    /// slow cold build) and whose built [`Instance::run`] blocks on a second
    /// barrier (simulating a slow but already-warm action) -- shared by both
    /// tests below.
    fn slow_activate_resident(name: &'static str, activate_gate: std::sync::Arc<std::sync::Barrier>, run_gate: std::sync::Arc<std::sync::Barrier>) -> std::sync::Arc<dyn crate::ResidentModel> {
        use crate::{Device, Instance, InstanceKey, MemCost, ResidentModel};
        use capability::{ActionResult, ActionSpec, Invocation, Manifest, Progress};

        struct Slow {
            name: &'static str,
            activate_gate: std::sync::Arc<std::sync::Barrier>,
            run_gate: std::sync::Arc<std::sync::Barrier>,
        }
        struct SlowInstance(std::sync::Arc<std::sync::Barrier>);
        impl Instance for SlowInstance {
            fn run(&mut self, _action: &str, _inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
                self.0.wait(); // block "running" until the test releases it
                Err("slow: done".into())
            }
        }
        impl ResidentModel for Slow {
            fn manifest(&self) -> Manifest {
                Manifest::new(self.name, "slow", vec![ActionSpec::new("run", "run")])
            }
            fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
                InstanceKey::new(self.name, "default")
            }
            fn estimate(&self, _k: &InstanceKey) -> MemCost {
                MemCost::new(0, 0)
            }
            fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
                self.activate_gate.wait(); // block "activation" until the test releases it
                Ok(Box::new(SlowInstance(self.run_gate.clone())))
            }
        }
        std::sync::Arc::new(Slow { name, activate_gate, run_gate })
    }

    #[test]
    fn model_is_cold_building_true_only_while_activate_is_still_in_flight() {
        use crate::budget::Budgets;
        use crate::{Device, Job, Policy};
        use capability::Invocation;
        use std::sync::{mpsc, Arc, Barrier};

        let mut budgets = Budgets::new();
        budgets.set(Device::Cpu, 1 << 30, 0);
        let activate_gate = Arc::new(Barrier::new(2));
        let run_gate = Arc::new(Barrier::new(2));
        let exec = Executor::start(vec![slow_activate_resident("slow/model", activate_gate.clone(), run_gate.clone())], budgets, Policy::default());

        assert!(!model_is_cold_building(&exec, "slow/model"), "nothing submitted yet");
        let (tx, rx) = mpsc::channel();
        exec.submit(Job::new("slow/model", "run", Invocation::new()).reply(move |r| {
            let _ = tx.send(r);
        }));
        // Give the dispatcher a moment to claim + hand off to the lane.
        for _ in 0..200 {
            if model_is_cold_building(&exec, "slow/model") {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(model_is_cold_building(&exec, "slow/model"), "must report cold-building while activate() blocks");
        assert!(!model_is_cold_building(&exec, "other/model"), "must not match an unrelated model name");

        activate_gate.wait(); // let activate() return -- Msg::Built should flip building off
        for _ in 0..200 {
            if !model_is_cold_building(&exec, "slow/model") {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!model_is_cold_building(&exec, "slow/model"), "must flip false the instant activate() finishes, even though the job is still running");

        run_gate.wait(); // let the (now warm) run() return
        rx.recv().ok();
    }

    /// Regression guard for the exact scenario `crates/dbus/tests/roundtrip.rs`'s
    /// `admit_deadline_sheds_a_saturated_lane` exercises: a SECOND request for a
    /// model that is already warm and simply busy running a slow action must NOT
    /// be mistaken for a cold build -- it has to keep shedding at the short
    /// deadline, not wait out the much longer cold-build one.
    #[test]
    fn model_is_cold_building_is_false_for_an_already_warm_but_slow_job() {
        use crate::budget::Budgets;
        use crate::{Device, Job, Policy};
        use capability::Invocation;
        use std::sync::{mpsc, Arc, Barrier};

        let mut budgets = Budgets::new();
        budgets.set(Device::Cpu, 1 << 30, 0);
        let activate_gate = Arc::new(Barrier::new(1)); // pre-satisfied: activate() never blocks
        let run_gate = Arc::new(Barrier::new(2));
        let exec = Executor::start(vec![slow_activate_resident("warm/model", activate_gate, run_gate.clone())], budgets, Policy::default());

        let (tx, rx) = mpsc::channel();
        exec.submit(Job::new("warm/model", "run", Invocation::new()).reply(move |r| {
            let _ = tx.send(r);
        }));
        // Wait for the job to actually be running (claimed + activated).
        for _ in 0..200 {
            if exec.in_flight().iter().any(|j| j.model == "warm/model" && j.phase == "running") {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!model_is_cold_building(&exec, "warm/model"), "an already-warm job that's merely slow must not read as a cold build");

        run_gate.wait();
        rx.recv().ok();
    }
}
