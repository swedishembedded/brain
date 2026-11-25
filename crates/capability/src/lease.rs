// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving a model and training it on the same device at once, for its
// clients. If your team needs expertise in scheduling training beside production inference on shared accelerators
// then you can procure our services by sending an email to info@swedishembedded.com.

//! A running job's hold on its device: while a long job waits for something that is not device work (a streaming
//! training run waiting for its next records, between two steps), it lends the device back to the server, which runs
//! other work on it meanwhile, and takes it back before it computes again.
//!
//! The surface that runs the job arms the [`DeviceLease`] on its [`Invocation`](crate::Invocation) with a [`Lender`]
//! (brain's executor does it for a workload's jobs). Like the [`CancelToken`](crate::CancelToken) it rides beside, a
//! lease is unarmed by default: lending it then does nothing, so a job that lends runs the same wherever it is called
//! from. Taking the device back waits until the work that ran on it meanwhile has finished; while it waits, the
//! server starts nothing new on the device.

use std::sync::Arc;

/// What lends a job's device and takes it back.
pub trait Lender: Send + Sync {
    /// The job does no device work until [`Self::reclaim`]: the device may run other work. Returns at once.
    fn lend(&self);
    /// The job wants its device back: returns once nothing else runs on it.
    fn reclaim(&self);
}

/// The device a job runs on, as its invocation carries it. See the module doc.
#[derive(Clone, Default)]
pub struct DeviceLease(Option<Arc<dyn Lender>>);

impl std::fmt::Debug for DeviceLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() { "DeviceLease(armed)" } else { "DeviceLease(unarmed)" })
    }
}

/// Takes the device back when dropped, so a wait that panics still returns it.
struct Lent<'a>(&'a dyn Lender);

impl Drop for Lent<'_> {
    fn drop(&mut self) {
        self.0.reclaim();
    }
}

impl DeviceLease {
    /// A lease lent and reclaimed through `lender`.
    pub fn armed(lender: Arc<dyn Lender>) -> DeviceLease {
        DeviceLease(Some(lender))
    }

    /// Whether lending it reaches a server.
    pub fn is_armed(&self) -> bool {
        self.0.is_some()
    }

    /// Runs `wait`, which must do no device work, with the device lent; returns its result once the device is this
    /// job's again. Unarmed, it only runs `wait`.
    pub fn lent<R>(&self, wait: impl FnOnce() -> R) -> R {
        let Some(lender) = &self.0 else { return wait() };
        lender.lend();
        let _back = Lent(lender.as_ref());
        wait()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Log(Mutex<Vec<&'static str>>);

    impl Lender for Log {
        fn lend(&self) {
            self.0.lock().unwrap().push("lend");
        }
        fn reclaim(&self) {
            self.0.lock().unwrap().push("reclaim");
        }
    }

    #[test]
    fn a_wait_runs_between_lending_and_reclaiming_and_a_panicking_one_still_gives_the_device_back() {
        let log = Arc::new(Log(Mutex::new(Vec::new())));
        let lease = DeviceLease::armed(log.clone());
        assert_eq!(lease.lent(|| log.0.lock().unwrap().push("wait")), ());
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lease.lent(|| panic!("the wait failed")))).is_err());
        assert_eq!(*log.0.lock().unwrap(), ["lend", "wait", "reclaim", "lend", "reclaim"]);
        assert_eq!(DeviceLease::default().lent(|| 7), 7, "an unarmed lease only runs the wait");
    }
}
