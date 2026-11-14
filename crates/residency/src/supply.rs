// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The seam that lets an unknown-but-fetchable model become servable
//! without a process restart: [`ModelSupplier`] classifies a model name and,
//! if it can produce one, fetches/converts/registers it; [`Executor::ensure_model`]
//! is the one blocking entry point a front end calls before submitting a job
//! for a model it does not yet recognize.
//!
//! [`Executor::ensure_model`] runs one fetch per model at a time, on a thread
//! of its own, and every caller asking for that model waits on the same one
//! (single-flight). Each caller waits under its own [`CancelToken`]: a caller
//! whose token fires stops waiting at once, and the fetch itself is cancelled
//! only when no caller waits for it any more. A caller that arrives after a
//! fetch was abandoned starts a new one.
//!
//! This module has no opinion about *where* a model comes from -- no fs, no
//! network, no dependency on `crates/modelstore`. The concrete supplier
//! (model-store backed, bounded) lives in `crates/cli`, which is free to
//! depend on `modelstore`; `residency` stays the leaf it always was.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use capability::CancelToken;

use crate::Executor;

/// What is known about a model name before any resident is built for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Supply {
    /// Already registered on the executor -- immediately usable, nothing to do.
    Resident,
    /// Not registered yet, but the supplier believes it can produce one
    /// (e.g. the model store could fetch + convert it).
    Fetchable,
    /// Not resident and not fetchable, with a reason a caller can surface
    /// directly (a 404 body, a D-Bus error) -- e.g. a reserved vendor with
    /// nothing on disk, or a name that fails the naming grammar.
    Unknown(String),
}

/// Implemented by whatever knows how to turn an unknown model name into a
/// resident one. `classify` MUST be cheap and network-free (it runs on
/// every request's fast path, including for models nobody will ever fetch);
/// `ensure` is the potentially slow, blocking part.
pub trait ModelSupplier: Send + Sync {
    /// Classify `model` without touching the network or filesystem beyond
    /// what is needed for an instant, local answer.
    fn classify(&self, model: &str) -> Supply;

    /// Make `model` resident on `exec` -- fetch/convert as needed, then
    /// call `exec.register(..)` before returning `Ok`. MUST be idempotent.
    /// Front ends reach it through [`Executor::ensure_model`], which runs at
    /// most one call per model at a time and shares it between every caller,
    /// so a supplier needs no deduplication of its own. `progress` receives
    /// `(message, step, total)` updates on a best-effort basis. `cancel` fires
    /// when nobody waits for the result any more: stop at the next point that
    /// leaves nothing half-written (between the chunks of a download) and
    /// return an error.
    fn ensure(&self, model: &str, exec: &Executor, progress: &mut dyn FnMut(&str, u32, u32), cancel: &CancelToken) -> Result<(), String>;
}

/// The error a caller of [`Executor::ensure_model`] gets when its own token
/// fired while it waited.
pub const ENSURE_CANCELLED: &str = "cancelled while waiting for the model";

/// How often a waiting caller looks at its own cancel token and the fetch's
/// latest progress; the outcome itself wakes it at once.
const WAIT_POLL: Duration = Duration::from_millis(100);

/// The fetches in flight on one executor, by model name.
#[derive(Default)]
pub(crate) struct Fetches(Mutex<HashMap<String, Arc<Flight>>>);

/// One fetch, and the callers waiting on it.
struct Flight {
    state: Mutex<FlightState>,
    /// Signalled when the outcome is published.
    done: Condvar,
    /// Handed to [`ModelSupplier::ensure`]; fired when the last waiter leaves.
    cancel: CancelToken,
}

#[derive(Default)]
struct FlightState {
    /// Callers waiting for the outcome right now.
    waiters: usize,
    /// Every waiter left before the outcome: the fetch is being cancelled, and
    /// a new caller must not join it.
    abandoned: bool,
    outcome: Option<Result<(), String>>,
    /// The fetch's latest progress, and a count of updates so a waiter reports
    /// each one it sees once.
    progress: Option<(String, u32, u32)>,
    updates: u64,
}

impl Flight {
    fn lock(&self) -> MutexGuard<'_, FlightState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Waits for the outcome as one caller, relaying the fetch's progress to
    /// `progress` from this caller's own thread. Leaves early when `cancel`
    /// fires; the last caller to leave cancels the fetch.
    fn wait(&self, cancel: &CancelToken, progress: &mut dyn FnMut(&str, u32, u32)) -> Result<(), String> {
        let mut seen = 0u64;
        let mut state = self.lock();
        loop {
            if cancel.is_cancelled() && state.outcome.is_none() {
                state.waiters -= 1;
                if state.waiters == 0 {
                    state.abandoned = true;
                    self.cancel.cancel();
                }
                return Err(ENSURE_CANCELLED.to_string());
            }
            // The latest progress first, so even a fetch that finished within one poll reports where it got to.
            if state.updates != seen {
                seen = state.updates;
                if let Some((name, step, total)) = state.progress.clone() {
                    drop(state);
                    progress(&name, step, total);
                    state = self.lock();
                }
                continue;
            }
            if let Some(outcome) = &state.outcome {
                let outcome = outcome.clone();
                state.waiters -= 1;
                return outcome;
            }
            state = self.done.wait_timeout(state, WAIT_POLL).unwrap_or_else(PoisonError::into_inner).0;
        }
    }
}

impl Fetches {
    fn lock(&self) -> MutexGuard<'_, HashMap<String, Arc<Flight>>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Joins the fetch of `model` in flight, or starts one with `start` when
    /// there is none (or the one there was abandoned). The caller is counted
    /// as a waiter either way.
    fn join(&self, model: &str, start: impl FnOnce(Arc<Flight>)) -> Arc<Flight> {
        let mut flights = self.lock();
        if let Some(flight) = flights.get(model) {
            let mut state = flight.lock();
            if !state.abandoned {
                state.waiters += 1;
                return Arc::clone(flight);
            }
        }
        let flight = Arc::new(Flight { state: Mutex::new(FlightState { waiters: 1, ..FlightState::default() }), done: Condvar::new(), cancel: CancelToken::armed() });
        flights.insert(model.to_string(), Arc::clone(&flight));
        drop(flights);
        start(Arc::clone(&flight));
        flight
    }

    /// Publishes `flight`'s outcome. It is taken out of the map first, so a
    /// caller arriving from here on starts afresh rather than reading a result
    /// that predates its request.
    fn finish(&self, model: &str, flight: &Flight, outcome: Result<(), String>) {
        {
            let mut flights = self.lock();
            if flights.get(model).is_some_and(|f| std::ptr::eq(Arc::as_ptr(f), flight)) {
                flights.remove(model);
            }
        }
        flight.lock().outcome = Some(outcome);
        flight.done.notify_all();
    }
}

impl Executor {
    /// Ensure `model` is registered on this executor, fetching it via
    /// `supplier` if it is not resident yet. Blocking; call this from a
    /// front end's own worker thread (e.g. HTTP's blocking task pool, a
    /// D-Bus stream-setup thread), never from the dispatcher thread itself.
    /// Returns `Ok(())` immediately if `model` is already registered --
    /// including the common case where it was resident from the start, so
    /// every caller can unconditionally call this before submitting a job.
    ///
    /// The fetch is shared with every other caller waiting for the same model
    /// (see the module docs). `cancel` is this caller's own: when it fires the
    /// call returns [`ENSURE_CANCELLED`] within a poll, and the fetch goes on
    /// for the callers still waiting, or stops if there are none.
    pub fn ensure_model(&self, model: &str, supplier: &Arc<dyn ModelSupplier>, progress: &mut dyn FnMut(&str, u32, u32), cancel: &CancelToken) -> Result<(), String> {
        if self.manifests().iter().any(|m| m.model == model) {
            return Ok(());
        }
        match supplier.classify(model) {
            Supply::Resident => return Ok(()),
            Supply::Fetchable => {}
            Supply::Unknown(reason) => return Err(reason),
        }
        let flight = self.fetches.join(model, |flight| self.start_fetch(model, Arc::clone(supplier), flight));
        flight.wait(cancel, progress)
    }

    /// Runs `supplier.ensure` for `flight` on a thread of its own, so the fetch
    /// outlives whichever caller started it.
    fn start_fetch(&self, model: &str, supplier: Arc<dyn ModelSupplier>, flight: Arc<Flight>) {
        let exec = self.clone();
        let name = model.to_string();
        let fetch = Arc::clone(&flight);
        let spawned = std::thread::Builder::new().name("brain-model-fetch".to_string()).spawn(move || {
            let mut report = |message: &str, step: u32, total: u32| {
                let mut state = fetch.lock();
                state.progress = Some((message.to_string(), step, total));
                state.updates += 1;
            };
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| supplier.ensure(&name, &exec, &mut report, &fetch.cancel)))
                .unwrap_or_else(|_| Err(format!("{name}: the fetch panicked")));
            exec.fetches.finish(&name, &fetch, outcome);
        });
        if let Err(e) = spawned {
            self.fetches.finish(model, &flight, Err(format!("{model}: could not start the fetch: {e}")));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::Budgets;
    use crate::{Device, Instance, InstanceKey, MemCost, Policy, ResidentModel};
    use capability::{ActionSpec, Invocation, Manifest};
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Instant;

    struct Toy(&'static str);
    impl ResidentModel for Toy {
        fn manifest(&self) -> Manifest {
            Manifest::new(self.0, "toy", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new(self.0, "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(0, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn Instance>, String> {
            Err("toy: not runnable".into())
        }
    }

    fn exec() -> Executor {
        let mut budgets = Budgets::new();
        budgets.set(Device::Cpu, 1 << 30, 0);
        Executor::start(vec![], budgets, Policy::default())
    }

    /// A supplier whose fetch takes `chunks` steps of 10 ms, checking its
    /// cancel token before each, and records what happened to it.
    #[derive(Default)]
    struct FakeSupplier {
        chunks: u32,
        fetches: AtomicU32,
        cancelled: AtomicBool,
    }
    impl ModelSupplier for FakeSupplier {
        fn classify(&self, model: &str) -> Supply {
            match model {
                "brain/known" => Supply::Resident, // never actually resident; exercises the branch
                "fetchable" => Supply::Fetchable,
                _ => Supply::Unknown(format!("{model}: no such model")),
            }
        }
        fn ensure(&self, model: &str, exec: &Executor, progress: &mut dyn FnMut(&str, u32, u32), cancel: &CancelToken) -> Result<(), String> {
            self.fetches.fetch_add(1, Ordering::SeqCst);
            for step in 0..self.chunks {
                if cancel.is_cancelled() {
                    self.cancelled.store(true, Ordering::SeqCst);
                    return Err("stopped".into());
                }
                progress("fetching", step + 1, self.chunks);
                std::thread::sleep(Duration::from_millis(10));
            }
            exec.register_if_absent(Arc::new(Toy(Box::leak(model.to_string().into_boxed_str()))));
            Ok(())
        }
    }

    fn fake_supplier(chunks: u32) -> (Arc<FakeSupplier>, Arc<dyn ModelSupplier>) {
        let fake = Arc::new(FakeSupplier { chunks, ..FakeSupplier::default() });
        let dynamic: Arc<dyn ModelSupplier> = fake.clone();
        (fake, dynamic)
    }

    fn never() -> CancelToken {
        CancelToken::default()
    }

    #[test]
    fn already_resident_short_circuits_with_no_classify_needed() {
        let e = exec();
        let (fake, supplier) = fake_supplier(2);
        // "totally-unknown-to-the-supplier" would classify Unknown, but since
        // the model is already registered ensure_model must never even ask.
        e.register(Arc::new(Toy("totally-unknown-to-the-supplier")));
        assert!(e.ensure_model("totally-unknown-to-the-supplier", &supplier, &mut |_, _, _| {}, &never()).is_ok());
        assert_eq!(fake.fetches.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn unknown_model_errors_with_the_suppliers_reason() {
        let (_, supplier) = fake_supplier(2);
        let err = exec().ensure_model("brain/nope", &supplier, &mut |_, _, _| {}, &never()).unwrap_err();
        assert!(err.contains("no such model"), "{err}");
    }

    #[test]
    fn fetchable_model_is_fetched_registered_and_reports_progress() {
        let e = exec();
        let (fake, supplier) = fake_supplier(3);
        let mut events = Vec::new();
        e.ensure_model("fetchable", &supplier, &mut |msg, step, total| events.push((msg.to_string(), step, total)), &never()).unwrap();
        assert_eq!(fake.fetches.load(Ordering::SeqCst), 1);
        assert!(!events.is_empty() && events.iter().all(|(m, _, total)| m == "fetching" && *total == 3), "{events:?}");
        let names: Vec<String> = e.manifests().iter().map(|m| m.model.clone()).collect();
        assert_eq!(names, vec!["fetchable".to_string()]);
    }

    /// Callers waiting on the same model share one fetch; one of them leaving
    /// does not cancel it for the others.
    #[test]
    fn one_waiter_leaving_does_not_cancel_the_fetch_for_the_others() {
        let e = exec();
        let (fake, supplier) = fake_supplier(40);
        let leaver = CancelToken::armed();
        let first = {
            let (e, supplier, leaver) = (e.clone(), supplier.clone(), leaver.clone());
            std::thread::spawn(move || e.ensure_model("fetchable", &supplier, &mut |_, _, _| {}, &leaver))
        };
        let second = {
            let (e, supplier) = (e.clone(), supplier.clone());
            std::thread::spawn(move || e.ensure_model("fetchable", &supplier, &mut |_, _, _| {}, &never()))
        };
        std::thread::sleep(Duration::from_millis(80));
        let left = Instant::now();
        leaver.cancel();
        assert_eq!(first.join().unwrap(), Err(ENSURE_CANCELLED.to_string()));
        assert!(left.elapsed() < Duration::from_secs(2), "the leaving caller returns at once");
        assert_eq!(second.join().unwrap(), Ok(()), "the other caller still gets the model");
        assert_eq!(fake.fetches.load(Ordering::SeqCst), 1, "one fetch served both");
        assert!(!fake.cancelled.load(Ordering::SeqCst));
    }

    /// When every waiter has left, the fetch is cancelled; a caller who comes
    /// later starts a new one instead of inheriting the cancellation.
    #[test]
    fn the_fetch_stops_when_no_waiter_remains_and_a_later_caller_starts_afresh() {
        let e = exec();
        let (fake, supplier) = fake_supplier(200);
        let tokens = [CancelToken::armed(), CancelToken::armed()];
        let waiters: Vec<_> = tokens
            .iter()
            .map(|t| {
                let (e, supplier, t) = (e.clone(), supplier.clone(), t.clone());
                std::thread::spawn(move || e.ensure_model("fetchable", &supplier, &mut |_, _, _| {}, &t))
            })
            .collect();
        std::thread::sleep(Duration::from_millis(80));
        for t in &tokens {
            t.cancel();
        }
        for w in waiters {
            assert_eq!(w.join().unwrap(), Err(ENSURE_CANCELLED.to_string()));
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while !fake.cancelled.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "the abandoned fetch never saw its cancel");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(e.manifests().is_empty(), "an abandoned fetch registers nothing");

        let (quick, quick_supplier) = fake_supplier(1);
        e.ensure_model("fetchable", &quick_supplier, &mut |_, _, _| {}, &never()).unwrap();
        assert_eq!(quick.fetches.load(Ordering::SeqCst), 1, "a fresh fetch, not the abandoned one's error");
    }
}
