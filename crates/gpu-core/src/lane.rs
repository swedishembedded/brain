// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements latency-critical inference serving for its
// clients. If your team needs expertise in keeping interactive generation
// responsive while a GPU is busy with bulk work, you can procure our services
// by sending an email to info@swedishembedded.com.

//! Which stream of a device work runs on.
//!
//! A device runs one stream's kernels in submission order, so a decode step
//! submitted behind a prefill round waits for the whole round. Every [`Gpu`]
//! can open a second, URGENT stream on its device ([`backend_api::Backend::
//! share_urgent`]) whose blocks the device schedules ahead of every other
//! stream's whenever both have work ready. Work issued inside [`urgent`] on
//! this thread goes there - allocations, steps, submissions, reads and waits
//! alike, so a wait inside the scope drains the urgent stream and never the
//! other one. Outside the scope every handle works exactly as before.
//!
//! The lane is a property of the calling thread, not of a handle: the model
//! code that records a decode step is the same code that records a prefill
//! round, and the scheduler that knows which is which decides by wrapping the
//! call. A backend that cannot prioritise ([`Gpu::has_urgent_lane`] false)
//! runs urgent work on the one stream it has.
//!
//! Buffers move freely between lanes; ordering between the two streams is the
//! backend's (reads and writes are fenced per buffer), except for a pool both
//! lanes write in disjoint rows, which [`Gpu::share_rows`] exempts.
//!
//! [`Gpu`]: crate::Gpu
//! [`Gpu::has_urgent_lane`]: crate::Gpu::has_urgent_lane
//! [`Gpu::share_rows`]: crate::Gpu::share_rows

use std::cell::Cell;

/// The streams a device's handle can issue work on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    /// The handle's own stream: everything outside [`urgent`].
    Normal,
    /// The device's high-priority stream: inside [`urgent`].
    Urgent,
}

std::thread_local! {
    static CURRENT: Cell<Lane> = const { Cell::new(Lane::Normal) };
}

/// The lane work issued on this thread goes to right now.
pub fn current() -> Lane {
    CURRENT.with(Cell::get)
}

/// Run `f` with this thread's work on every device going to its urgent lane.
/// Scopes nest; the outer lane returns when `f` does, including by unwinding.
pub fn urgent<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(Lane);
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(CURRENT.with(|c| c.replace(Lane::Urgent)));
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lane_belongs_to_the_scope_and_the_thread() {
        assert_eq!(current(), Lane::Normal);
        urgent(|| {
            assert_eq!(current(), Lane::Urgent);
            assert_eq!(std::thread::spawn(current).join().unwrap(), Lane::Normal, "another thread keeps its own lane");
            urgent(|| assert_eq!(current(), Lane::Urgent));
            assert_eq!(current(), Lane::Urgent, "a nested scope restores the outer lane");
        });
        assert_eq!(current(), Lane::Normal);
        let _ = std::panic::catch_unwind(|| urgent(|| panic!("unwinding out of the scope")));
        assert_eq!(current(), Lane::Normal, "unwinding restores the lane");
    }
}
