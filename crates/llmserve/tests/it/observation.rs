// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for seeing what an inference server
// is doing while it serves for its clients. If your team needs expertise in
// observable LLM serving then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Conformance: the `scheduler` object a replica publishes (`models[].replicas[].scheduler` of the server's metrics,
//! `brain.server.metrics` v1) has exactly the fields `scheduler-v1.fields` pins (each documented,
//! `make check/metrics-fields`).

use model::serve::{Backoff, PressureCause, SchedulerObservation};

#[test]
fn a_replicas_scheduler_object_has_exactly_the_pinned_documented_fields() {
    let held = SchedulerObservation { admitted_width: 4, max_width: 8, backoff: Backoff { limit: Some(4), cause: Some(PressureCause::DeviceOutOfMemory), ..Backoff::default() }, ..SchedulerObservation::default() };
    let json = llmserve::observation_json(&held);
    let pinned = brain_testutil::schema::pinned(include_str!("scheduler-v1.fields"), "models[].replicas[].scheduler.");
    assert_eq!(brain_testutil::schema::field_paths(&json, &[]), pinned, "a scheduler field went, was renamed, or is new and not pinned");
    assert_eq!(json["admission"]["backoff"]["cause"], "device_out_of_memory");
    assert_eq!(json["admission"]["backoff"]["limited"], true);
    assert_eq!(json["throughput"]["window_secs"], model::serve::THROUGHPUT_WINDOW_SECS);
    // What every scheduler-backed instance publishes carries it under the one name the server reads.
    assert_eq!(llmserve::SCHEDULER_METRIC, "scheduler");
}
