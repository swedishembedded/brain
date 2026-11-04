// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Recording a graph's cost WITHOUT running it.
//!
//! Swedish Embedded AB implements analytic cost models for GPU inference
//! pipelines. If your team needs to price a model on hardware it has not run
//! on yet, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! `Gpu::cost_of` prices a step list offline, but a model that builds its
//! dispatches inside `forward()` and submits them there has no step list to
//! hand out, and a model that opens its own device does not even give the
//! caller the handle whose counters would see them - the only way to price
//! either was to run it. `cost::Recording` closes that: it is scoped to the
//! calling THREAD, so it catches every handle a `forward()` touches, and a DRY
//! recording folds each step in and then drops it on the floor.
//!
//! The contract has two halves and both are gated here, because either alone is
//! satisfiable by a no-op:
//!
//! * the recorded cost must EQUAL a real run's counters - the graph is the
//!   same graph;
//! * the device must be UNTOUCHED - a dry run that only set a flag and still
//!   executed would pass the first half perfectly.

use gpu_core::cost::{CostReport, Recording};
use gpu_core::{DeviceBuffer, Gpu, Step};

const KERNELS: &[(&str, &str)] = &[("add2", kernels::ADD2), ("silu", kernels::SILU)];

/// A small two-kernel graph: y = a + b, then z = silu(y).
fn graph(gpu: &Gpu, a: &DeviceBuffer, b: &DeviceBuffer, y: &DeviceBuffer, z: &DeviceBuffer, n: u32) -> Vec<Step> {
    vec![gpu.step(0, &[a, b, y], &[n], n), gpu.step(1, &[y, z], &[n], n)]
}

#[test]
fn a_dry_run_records_the_same_cost_and_leaves_the_device_untouched() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let n = 256u32;
    let (a, b, y, z) = (gpu.storage(n as u64), gpu.storage(n as u64), gpu.storage(n as u64), gpu.storage(n as u64));
    let ones = vec![1.0f32; n as usize];
    gpu.write_f32(&a, &ones);
    gpu.write_f32(&b, &ones);
    let sentinel = vec![-7.0f32; n as usize];
    gpu.write_f32(&y, &sentinel);
    gpu.write_f32(&z, &sentinel);
    let steps = graph(&gpu, &a, &b, &y, &z, n);

    // OFFLINE: record without executing.
    let rec = Recording::dry();
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    let dry: CostReport = rec.take();

    assert_eq!(
        gpu.read(&z, n as usize),
        sentinel,
        "a dry run must not write to the device: the output buffer moved, so something executed"
    );
    assert_eq!(dry.steps, 2, "both dispatches must still be recorded");
    assert_eq!(dry.coverage(), 1.0);

    // ONLINE: the same graph, actually executed.
    gpu.reset_ops_counters();
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    let wet = gpu.ops_counters();

    assert_ne!(gpu.read(&z, n as usize), sentinel, "the real run must have written the output");
    assert_eq!(dry.total, wet.total, "dry-run cost must equal the executed run's counters");
    assert_eq!(dry.steps, wet.steps);
    assert_eq!(dry.by_kernel.len(), wet.by_kernel.len());
    for (k, v) in &dry.by_kernel {
        let w = wet.by_kernel.get(k).unwrap_or_else(|| panic!("{k} missing from the executed run"));
        assert_eq!((v.calls, v.cost), (w.calls, w.cost), "kernel {k}");
    }
}

/// Recording is off by default and ends with the guard - a dry mode that
/// leaked would silently turn every later submit in the process into a no-op,
/// which is a far worse defect than the missing feature it replaced.
#[test]
fn recording_is_off_by_default_and_ends_with_its_guard() {
    assert!(!gpu_core::cost::is_recording(), "a thread must start un-recorded");
    {
        let _r = Recording::dry();
        assert!(gpu_core::cost::is_recording());
    }
    assert!(!gpu_core::cost::is_recording(), "the guard must close the recording when dropped");

    let gpu = gpu_core::testgpu::dev(KERNELS);
    let n = 64u32;
    let (a, b, y, z) = (gpu.storage(n as u64), gpu.storage(n as u64), gpu.storage(n as u64), gpu.storage(n as u64));
    gpu.write_f32(&a, &vec![1.0f32; n as usize]);
    gpu.write_f32(&b, &vec![1.0f32; n as usize]);
    gpu.write_f32(&z, &vec![-7.0f32; n as usize]);
    gpu.submit(&[], &graph(&gpu, &a, &b, &y, &z, n));
    gpu.poll_wait();
    assert_ne!(gpu.read(&z, n as usize), vec![-7.0f32; n as usize], "execution must resume after a dry recording ends");
}
