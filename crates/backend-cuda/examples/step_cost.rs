// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for host-side launch overhead of
// GPU inference servers for its clients. If your team needs expertise in
// making a dispatch thread keep a GPU busy then you can procure our services
// by sending an email to info@swedishembedded.com.

//! The host cost of one recorded step on the CUDA backend: recording it
//! (`Backend::step`) and issuing it (`Backend::submit`), per step, as the best
//! of several repetitions. The steps are tiny elementwise kernels, so the
//! device never bounds the loop: what is measured is the dispatch thread.
//!
//! Usage: `step_cost [steps per submit] [repetitions]`
//!
//! Run it on a card with free memory (it allocates kilobytes); the numbers are
//! wall time of one thread, so a loaded host inflates them - compare runs back
//! to back.

use std::time::{Duration, Instant};

use backend_api::Backend as _;
use backend_cuda::CudaBackend;

const KERNELS: &[(&str, &str)] = &[("add2", kernels::ADD2)];
const ADD2: usize = 0;

fn best<T>(reps: u32, mut f: impl FnMut() -> T) -> Duration {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(f());
            t.elapsed()
        })
        .min()
        .expect("at least one repetition")
}

fn main() {
    let steps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let reps: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let b = CudaBackend::try_new(KERNELS).unwrap_or_else(|e| panic!("no usable CUDA backend: {e}"));
    let bufs: Vec<_> = (0..8).map(|i| b.storage_init("x", &[i as f32; 64])).collect();
    let make = || -> Vec<_> { (0..steps).map(|i| b.step(ADD2, &[&bufs[i % 8], &bufs[(i + 1) % 8], &bufs[(i + 2) % 8]], &[64], 64)).collect() };

    let per = |d: Duration| d.as_nanos() as f64 / steps as f64 / 1e3;
    let record = best(reps, make);
    println!("record          {:7.2} us/step", per(record));

    for (label, capture) in [("submit, no graph", false), ("submit, graph   ", true)] {
        let b = CudaBackend::try_new(KERNELS).expect("backend").with_graph_capture(capture);
        let bufs: Vec<_> = (0..8).map(|i| b.storage_init("x", &[i as f32; 64])).collect();
        let steps_v: Vec<_> = (0..steps).map(|i| b.step(ADD2, &[&bufs[i % 8], &bufs[(i + 1) % 8], &bufs[(i + 2) % 8]], &[64], 64)).collect();
        b.submit(&[], &steps_v);
        b.poll_wait();
        let issue = best(reps, || {
            b.submit(&[], &steps_v);
            b.poll_wait();
        });
        println!("{label} {:7.2} us/step (issue + drain)", per(issue));
    }
}
