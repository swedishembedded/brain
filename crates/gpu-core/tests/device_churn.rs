// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How many `Gpu`s can one process build and drop in sequence?
//!
//! `crates/bench/tests/capscale.rs` fails on this box with
//! `physical GPU "Tesla P40" not found among 0 wgpu adapter(s)` AFTER six
//! devices in the same process succeeded — so the question is not whether the
//! card is visible but whether building and dropping devices in a loop
//! exhausts something.
use std::sync::atomic::{AtomicUsize, Ordering};

static N: AtomicUsize = AtomicUsize::new(0);

#[test]
fn building_and_dropping_devices_in_sequence_does_not_exhaust_the_driver() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    const KERNELS: [(&str, &str); 1] = [("add2", kernels::ADD2)];
    for i in 0..12 {
        let gpu = gpu_core::Gpu::new(&KERNELS);
        let a = gpu.storage(4);
        gpu.write_f32(&a, &[1.0, 2.0, 3.0, 4.0]);
        let b = gpu.storage(4);
        gpu.write_f32(&b, &[1.0, 1.0, 1.0, 1.0]);
        let c = gpu.storage(4);
        gpu.submit(&[], &[gpu.step(0, &[&a, &b, &c], &[4], 4)]);
        gpu.poll_wait();
        assert_eq!(gpu.read(&c, 4), vec![2.0, 3.0, 4.0, 5.0], "device {i} computed wrong");
        N.store(i + 1, Ordering::Relaxed);
        drop(gpu);
    }
    eprintln!("built {} devices in sequence", N.load(Ordering::Relaxed));
}

/// The same loop, but HOLDING every device alive. `capscale` builds one engine
/// per grid point; if the answer differs from the sequential case, the fault is
/// concurrently-live devices rather than churn.
#[test]
fn building_devices_while_holding_the_previous_ones_does_not_exhaust_the_driver() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    const KERNELS: [(&str, &str); 1] = [("add2", kernels::ADD2)];
    let mut live = Vec::new();
    for i in 0..12 {
        eprintln!("building live device {i}");
        live.push(gpu_core::Gpu::new(&KERNELS));
    }
    eprintln!("held {} devices live", live.len());
}

/// The FAITHFUL repro: real allocation and real submits per device, the way
/// `model::train::fit` uses one. The trivial loops above pass at 12; `capscale`
/// dies at the 7th, and the only difference left is how much each device did.
#[test]
fn devices_that_did_real_work_can_still_be_rebuilt() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    const KERNELS: [(&str, &str); 1] = [("add2", kernels::ADD2)];
    const N: usize = 1 << 22; // 16 MiB per buffer, 48 MiB per device
    for i in 0..12 {
        eprintln!("device {i}: building");
        let gpu = gpu_core::Gpu::new(&KERNELS);
        let a = gpu.storage(N as u64);
        let b = gpu.storage(N as u64);
        let c = gpu.storage(N as u64);
        gpu.write_f32(&a, &vec![1.0f32; N]);
        gpu.write_f32(&b, &vec![2.0f32; N]);
        for _ in 0..40 {
            gpu.submit(&[], &[gpu.step(0, &[&a, &b, &c], &[N as u32], N as u32)]);
        }
        gpu.poll_wait();
        assert_eq!(gpu.read(&c, 4), vec![3.0; 4], "device {i} computed wrong");
        drop(gpu);
    }
    eprintln!("12 working devices built and dropped");
}
