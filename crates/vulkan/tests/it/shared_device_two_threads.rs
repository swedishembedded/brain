// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for GPU backends that stay correct
// under concurrent use for its clients. If your team needs expertise in
// Vulkan and multi-threaded device sharing then you can procure our services
// by sending an email to info@swedishembedded.com.

//! Reproducer for the open shared-device race (knowledge 300): two threads
//! that each hold a handle on the SAME pooled device (`gpu_core::testgpu::dev`,
//! one `new_like` handle per kernel set, exactly how a test binary's tests
//! share it) and dispatch a known-answer kernel concurrently get wrong results
//! or a lost device on the Vulkan backend, in about half of the runs. With one
//! thread, or with `--test-threads=1` in the suites that hit it, it does not
//! happen, so the defect needs concurrency on the shared device.
//!
//! The kernel is a residual add whose exact answer is known on the host, so a
//! wrong result is distinguishable from a lost device (which panics inside
//! the backend). The two threads use two kernel sets so each gets its own
//! handle on the one device; the pool keys on the kernel slice's address.
//!
//! It stays `#[ignore]`d until the race is fixed, because it is red on a
//! fraction of runs by design; run it as
//! `BRAIN_DEVICE=vulkan cargo test -p brain-vulkan --test shared_device_two_threads -- --ignored`
//! and repeat it, since one green run proves nothing about a race. On the CPU
//! backend (no shared device) it is green every time, which is the sanity
//! check that the test itself is sound.

use std::sync::{Arc, Barrier};

static KERNELS_A: &[(&str, &str)] = &[("add", kernels::ADD)];
static KERNELS_B: &[(&str, &str)] = &[("add", kernels::ADD)];

const ROUNDS: usize = 200;
const LEN: usize = 4096;

/// `dst += src` on the device, `ROUNDS` times with fresh buffers each round;
/// returns the first mismatch against the host answer, if any.
fn hammer(kernels: &'static [(&'static str, &'static str)], salt: f32, start: Arc<Barrier>) -> Option<String> {
    start.wait();
    let gpu = gpu_core::testgpu::dev(kernels);
    for round in 0..ROUNDS {
        let a: Vec<f32> = (0..LEN).map(|i| salt + (i % 97) as f32).collect();
        let b: Vec<f32> = (0..LEN).map(|i| (round + i % 13) as f32).collect();
        let dst = gpu.storage_init("dst", &a);
        let src = gpu.storage_init("src", &b);
        let step = gpu.step(0, &[&src, &dst], &[LEN as u32], LEN as u32);
        gpu.submit(&[], &[step]);
        gpu.poll_wait();
        let got = gpu.read(&dst, LEN);
        for i in 0..LEN {
            let want = a[i] + b[i];
            if got[i] != want {
                return Some(format!("round {round}, element {i}: got {}, want {want}", got[i]));
            }
        }
    }
    None
}

#[test]
#[ignore = "reproduces the open race"]
fn two_threads_on_one_shared_device_get_correct_results() {
    let start = Arc::new(Barrier::new(2));
    let (sa, sb) = (start.clone(), start);
    let ta = std::thread::spawn(move || hammer(KERNELS_A, 1.0, sa));
    let tb = std::thread::spawn(move || hammer(KERNELS_B, 1000.0, sb));
    let (ra, rb) = (ta.join(), tb.join());
    let mut failures = Vec::new();
    for (name, r) in [("thread A", ra), ("thread B", rb)] {
        match r {
            Err(_) => failures.push(format!("{name} panicked (device lost?)")),
            Ok(Some(msg)) => failures.push(format!("{name}: {msg}")),
            Ok(None) => {}
        }
    }
    assert!(failures.is_empty(), "shared-device race: {failures:?}");
}
