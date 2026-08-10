// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The seam that lets an unknown-but-fetchable model become servable
//! without a process restart: [`ModelSupplier`] classifies a model name and,
//! if it can produce one, fetches/converts/registers it; [`Executor::ensure_model`]
//! is the one blocking entry point a front end calls before submitting a job
//! for a model it does not yet recognize.
//!
//! This module has no opinion about *where* a model comes from -- no fs, no
//! network, no dependency on `crates/modelstore`. The concrete supplier
//! (model-store backed, single-flight, bounded) lives in `crates/cli`, which
//! is free to depend on `modelstore`; `residency` stays the leaf it always
//! was.

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
    /// call `exec.register(..)` before returning `Ok`. MUST be idempotent
    /// and single-flight: concurrent calls for the same `model` share one
    /// underlying fetch, and all of them return once it completes (or all
    /// return the same error). `progress` receives `(message, step, total)`
    /// updates on a best-effort basis.
    fn ensure(&self, model: &str, exec: &Executor, progress: &mut dyn FnMut(&str, u32, u32)) -> Result<(), String>;
}

impl Executor {
    /// Ensure `model` is registered on this executor, fetching it via
    /// `supplier` if it is not resident yet. Blocking; call this from a
    /// front end's own worker thread (e.g. HTTP's blocking task pool, a
    /// D-Bus stream-setup thread), never from the dispatcher thread itself.
    /// Returns `Ok(())` immediately if `model` is already registered --
    /// including the common case where it was resident from the start, so
    /// every caller can unconditionally call this before submitting a job.
    pub fn ensure_model(&self, model: &str, supplier: &dyn ModelSupplier, progress: &mut dyn FnMut(&str, u32, u32)) -> Result<(), String> {
        if self.manifests().iter().any(|m| m.model == model) {
            return Ok(());
        }
        match supplier.classify(model) {
            Supply::Resident => Ok(()),
            Supply::Fetchable => supplier.ensure(model, self, progress),
            Supply::Unknown(reason) => Err(reason),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::Budgets;
    use crate::{Device, Instance, InstanceKey, MemCost, Policy, ResidentModel};
    use capability::{ActionSpec, Invocation, Manifest};
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

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

    struct FakeSupplier {
        fetches: Arc<AtomicU32>,
    }
    impl ModelSupplier for FakeSupplier {
        fn classify(&self, model: &str) -> Supply {
            match model {
                "brain/known" => Supply::Resident, // never actually resident; exercises the branch
                "fetchable" => Supply::Fetchable,
                _ => Supply::Unknown(format!("{model}: no such model")),
            }
        }
        fn ensure(&self, model: &str, exec: &Executor, progress: &mut dyn FnMut(&str, u32, u32)) -> Result<(), String> {
            self.fetches.fetch_add(1, Ordering::SeqCst);
            progress("fetching", 1, 2);
            progress("registering", 2, 2);
            exec.register(Arc::new(Toy(Box::leak(model.to_string().into_boxed_str()))));
            Ok(())
        }
    }

    #[test]
    fn already_resident_short_circuits_with_no_classify_needed() {
        let e = exec();
        e.register(Arc::new(Toy("already-here")));
        let supplier = FakeSupplier { fetches: Arc::new(AtomicU32::new(0)) };
        // "totally-unknown-to-the-supplier" would classify Unknown, but since
        // the model is already registered ensure_model must never even ask.
        e.register(Arc::new(Toy("totally-unknown-to-the-supplier")));
        assert!(e.ensure_model("totally-unknown-to-the-supplier", &supplier, &mut |_, _, _| {}).is_ok());
        assert_eq!(supplier.fetches.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn unknown_model_errors_with_the_suppliers_reason() {
        let e = exec();
        let supplier = FakeSupplier { fetches: Arc::new(AtomicU32::new(0)) };
        let err = e.ensure_model("brain/nope", &supplier, &mut |_, _, _| {}).unwrap_err();
        assert!(err.contains("no such model"), "{err}");
    }

    #[test]
    fn fetchable_model_is_fetched_and_registered() {
        let e = exec();
        let supplier = FakeSupplier { fetches: Arc::new(AtomicU32::new(0)) };
        let mut events = Vec::new();
        e.ensure_model("fetchable", &supplier, &mut |msg, step, total| events.push((msg.to_string(), step, total))).unwrap();
        assert_eq!(supplier.fetches.load(Ordering::SeqCst), 1);
        assert_eq!(events, vec![("fetching".to_string(), 1, 2), ("registering".to_string(), 2, 2)]);
        let names: Vec<String> = e.manifests().iter().map(|m| m.model.clone()).collect();
        assert_eq!(names, vec!["fetchable".to_string()]);
    }
}
