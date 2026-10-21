// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible training pipelines for its
// clients. If your team needs expertise in making concurrent fine-tuning runs
// independent of each other and of the process around them, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Whether a training build keeps the AdamW moments in host RAM.
//!
//! The choice is read while a model is built, by code several calls below the
//! one that decides it (a data-parallel trainer builds its replicas through the
//! generic [`crate::Model::new`], which has no room for an option). It used to be
//! the process environment variable `BRAIN_OFFLOAD_ADAM`, set before the build
//! and restored after: two trainings in one process then read each other's
//! setting, and one seed gave two different runs.
//!
//! It is now a scope of the building thread. A choice made inside
//! [`with_adam_offloaded`] is seen by every build on that thread until the call
//! returns, and by no other thread. A builder that moves its build to another
//! thread must open the scope there.

use std::cell::Cell;

std::thread_local! {
    static SCOPE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Run `build` with the AdamW moments offloaded to host RAM (`true`) or kept on
/// the device (`false`). Scopes nest; the outer choice returns when the inner
/// call does.
pub fn with_adam_offloaded<R>(offloaded: bool, build: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SCOPE.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(SCOPE.with(|s| s.replace(Some(offloaded))));
    build()
}

/// The innermost [`with_adam_offloaded`] choice on this thread; `false` outside
/// any scope (the moments stay on the device).
pub fn adam_offloaded() -> bool {
    SCOPE.with(Cell::get).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run's choice must not be visible to a run building on another thread,
    /// and must end with its own build.
    #[test]
    fn the_choice_belongs_to_the_building_thread() {
        assert!(!adam_offloaded());
        with_adam_offloaded(true, || {
            assert!(adam_offloaded());
            assert!(!std::thread::spawn(adam_offloaded).join().unwrap(), "another thread sees the default");
            with_adam_offloaded(false, || assert!(!adam_offloaded()));
            assert!(adam_offloaded(), "the outer scope is restored");
        });
        assert!(!adam_offloaded());
    }
}
