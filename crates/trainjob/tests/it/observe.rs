// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for seeing where a fine-tune's wall
// time goes on a shared accelerator, for its clients. If your team needs
// expertise in efficient training beside production serving then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Conformance: a training job's account is its `report` in the server's metrics (`training_jobs[].report` of
//! `brain.server.metrics` v1): exactly the fields `report-v1.fields` pins (each documented, `make check/metrics-fields`).

use std::time::Instant;

use capability::report::JobReport;
use trainjob::observe::RunObserver;

#[test]
fn a_jobs_account_has_exactly_the_pinned_fields() {
    let account = RunObserver::new(JobReport::armed(), "train", vec!["gpu0".into()], Instant::now()).account(Instant::now());
    let pinned = brain_testutil::schema::pinned(include_str!("report-v1.fields"), "training_jobs[].report.");
    assert_eq!(brain_testutil::schema::field_paths(&account, &[]), pinned, "a report field went, was renamed, or is new and not pinned");
    let phases: Vec<String> = trainjob::observe::PHASES.iter().map(|p| format!("phases_secs.{p}")).collect();
    assert!(phases.iter().all(|p| pinned.contains(p)), "every phase is a pinned field");
}
