// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for seeing where a long training
// job's time goes while it runs, for its clients. If your team needs expertise
// in observable training and serving on shared accelerators then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A running job's own account of itself, for whoever observes the server it runs on: a long job (a fine-tune)
//! publishes what it measures (steps, phases of its wall time) through the [`JobReport`] its
//! [`Invocation`](crate::Invocation) carries, and the server lists the latest one with the job in its metrics.
//!
//! The surface that runs the job arms the report (brain's executor does it for a workload's jobs) and tells it how
//! long the job waited before it got its device ([`JobReport::admitted_after`]). Like the
//! [`CancelToken`](crate::CancelToken) it rides beside, a report is unarmed by default: publishing then keeps nothing,
//! so a job that reports runs the same wherever it is called from.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

#[derive(Default)]
struct Shared {
    /// What the job last published.
    latest: Option<Value>,
    /// How long the job waited for its device before it started.
    admitted_after: Option<Duration>,
}

/// A job's report, as its invocation carries it. See the module doc. Clones share one report.
#[derive(Clone, Default)]
pub struct JobReport(Option<Arc<Mutex<Shared>>>);

impl std::fmt::Debug for JobReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() { "JobReport(armed)" } else { "JobReport(unarmed)" })
    }
}

impl JobReport {
    /// A report the surface reads back.
    pub fn armed() -> JobReport {
        JobReport(Some(Arc::default()))
    }

    /// Whether what the job publishes reaches an observer.
    pub fn is_armed(&self) -> bool {
        self.0.is_some()
    }

    fn shared(&self) -> Option<std::sync::MutexGuard<'_, Shared>> {
        self.0.as_ref().map(|s| s.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
    }

    /// Replace what the job reports with `report` (a JSON object). Unarmed, it keeps nothing.
    pub fn publish(&self, report: Value) {
        if let Some(mut shared) = self.shared() {
            shared.latest = Some(report);
        }
    }

    /// What the job last published.
    pub fn latest(&self) -> Option<Value> {
        self.shared().and_then(|s| s.latest.clone())
    }

    /// The surface tells the job it waited `waited` for its device before it started.
    pub fn set_admitted_after(&self, waited: Duration) {
        if let Some(mut shared) = self.shared() {
            shared.admitted_after = Some(waited);
        }
    }

    /// How long the job waited for its device before it started; `None` when the surface did not say (unarmed, or
    /// run outside a server).
    pub fn admitted_after(&self) -> Option<Duration> {
        self.shared().and_then(|s| s.admitted_after)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_armed_report_keeps_what_the_job_published_and_an_unarmed_one_keeps_nothing() {
        let unarmed = JobReport::default();
        unarmed.publish(serde_json::json!({"step": 1}));
        unarmed.set_admitted_after(Duration::from_secs(3));
        assert_eq!((unarmed.latest(), unarmed.admitted_after()), (None, None));
        let armed = JobReport::armed();
        let seen = armed.clone();
        armed.set_admitted_after(Duration::from_secs(3));
        armed.publish(serde_json::json!({"step": 1}));
        armed.publish(serde_json::json!({"step": 2}));
        assert_eq!(seen.latest(), Some(serde_json::json!({"step": 2})), "clones share one report, the latest wins");
        assert_eq!(seen.admitted_after(), Some(Duration::from_secs(3)));
    }
}
