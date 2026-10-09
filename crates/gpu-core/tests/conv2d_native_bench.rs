// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device time of the native dense-conv kernels at every conv of a YOLOv8n
//! training step (batch 8, 512 x 512 input): forward, input gradient and
//! weight gradient per unit, the step's total, and the achieved fp32 rate
//! against the device's measured roof.
//!
//! Swedish Embedded AB implements training kernels for convolutional networks.
//! If your team needs expertise in measuring and closing the gap to a GPU's
//! roofline, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Every number is a DEVICE timestamp pair around ONE launch, and each launch
//! is repeated and the fastest kept: on a card shared with other work a launch
//! that straddles another context's time slice reads that slice as its own, and
//! the minimum is the sample that did not. Correctness is the gate's job
//! (`conv2d_native.rs`); this only measures.
//!
//! `cargo test --release -p brain-gpu-core --test conv2d_native_bench -- --ignored --nocapture`

use gpu_core::Gpu;

const KERNELS: &[(&str, &str)] = &[("conv2d", kernels::CONV2D), ("conv2d_dx", kernels::CONV2D_DX)];
const CONV: usize = 0;
const CONV_DX: usize = 1;

/// YOLOv8n (nc = 3) at a 512 input: `(Cin, Cout, K, stride, pad, input side,
/// units)` - every dense conv of the network with how many units share it.
const UNITS: &[(u32, u32, u32, u32, u32, u32, u32)] = &[
    (3, 16, 3, 2, 1, 512, 1),
    (16, 32, 3, 2, 1, 256, 1),
    (32, 32, 1, 1, 0, 128, 1),
    (16, 16, 3, 1, 1, 128, 2),
    (48, 32, 1, 1, 0, 128, 1),
    (32, 64, 3, 2, 1, 128, 1),
    (64, 64, 1, 1, 0, 64, 1),
    (32, 32, 3, 1, 1, 64, 6),
    (128, 64, 1, 1, 0, 64, 1),
    (64, 128, 3, 2, 1, 64, 1),
    (128, 128, 1, 1, 0, 32, 1),
    (64, 64, 3, 1, 1, 32, 10),
    (256, 128, 1, 1, 0, 32, 1),
    (128, 256, 3, 2, 1, 32, 1),
    (256, 256, 1, 1, 0, 16, 1),
    (128, 128, 3, 1, 1, 16, 4),
    (384, 256, 1, 1, 0, 16, 3),
    (256, 128, 1, 1, 0, 16, 1),
    (512, 256, 1, 1, 0, 16, 1),
    (384, 128, 1, 1, 0, 32, 1),
    (192, 128, 1, 1, 0, 32, 3),
    (192, 64, 1, 1, 0, 64, 1),
    (96, 64, 1, 1, 0, 64, 1),
    (64, 64, 3, 2, 1, 64, 1),
    (128, 128, 3, 2, 1, 32, 1),
    (64, 64, 3, 1, 1, 64, 4),
    (64, 64, 1, 1, 0, 64, 1),
    (64, 3, 1, 1, 0, 64, 1),
    (128, 64, 3, 1, 1, 32, 2),
    (64, 64, 1, 1, 0, 32, 1),
    (64, 3, 1, 1, 0, 32, 1),
    (256, 64, 3, 1, 1, 16, 2),
    (64, 64, 3, 1, 1, 16, 2),
    (64, 64, 1, 1, 0, 16, 1),
    (64, 3, 1, 1, 0, 16, 1),
];

const BATCH: u32 = 8;
const REPS: usize = 7;

/// Device time of `steps`, each step timed ALONE in its own submission (in
/// order, so each still reads what the one before wrote) and the fastest of
/// [`REPS`] kept per step, ms.
///
/// One step per submission is deliberate: a step that is not first in its
/// submission is preceded on the stream by its uniform's upload, a copy-engine
/// operation during which a shared device may switch to another context; the
/// switch's whole time slice then lands between that step's timestamps.
fn min_ms(gpu: &Gpu, steps: &dyn Fn() -> Vec<gpu_core::Step>) -> f64 {
    let n = steps().len();
    let mut best = vec![f64::INFINITY; n];
    for _ in 0..REPS {
        for (i, s) in steps().into_iter().enumerate() {
            gpu.poll_wait();
            gpu.reset_kernel_times();
            gpu.submit(&[], &[s]);
            gpu.poll_wait();
            let t: f64 = gpu.kernel_times().expect("this backend times kernels").iter().map(|r| r.1).sum();
            best[i] = best[i].min(t);
        }
    }
    best.iter().sum()
}

#[test]
#[ignore = "benchmark: device timing of every YOLOv8n conv, run by hand"]
fn yolov8n_conv_device_time() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if gpu.kind() != "cuda" || gpu.conv2d_dw_split(&[1, 1, 1, 1, 1, 1, 1, 0, 1, 1]).is_none() {
        brain_testutil::skip_unavailable("the native conv kernels need a CUDA device");
        return;
    }
    let roof = gpu_core::roof::ensure(&gpu).map(|r| f64::from(r.gflops));
    assert!(gpu.set_kernel_timing(true), "kernel timing must be available on this backend");
    let mut r = data::rng::Lcg::new(1);
    let (mut t_fwd, mut t_dx, mut t_dw, mut flop_total) = (0.0, 0.0, 0.0, 0.0);
    eprintln!("{:<28} {:>5} {:>9} {:>9} {:>9} {:>8} {:>8} {:>8}", "unit (cin,cout,k,s,side)", "units", "fwd ms", "dx ms", "dw ms", "fwd GF/s", "dx GF/s", "dw GF/s");
    for &(cin, cout, k, s, pad, side, units) in UNITS {
        let (h, w) = (side, side);
        let (ho, wo) = ((h + 2 * pad - k) / s + 1, (w + 2 * pad - k) / s + 1);
        let p = [BATCH, cin, h, w, cout, k, s, pad, ho, wo];
        let x = gpu.storage_init("x", &r.vec((BATCH * cin * h * w) as usize));
        let wt = gpu.storage_init("w", &r.vec((cout * cin * k * k) as usize));
        let dy = gpu.storage_init("dy", &r.vec((BATCH * cout * ho * wo) as usize));
        let y = gpu.storage(u64::from(BATCH * cout * ho * wo));
        let dx = gpu.storage(u64::from(BATCH * cin * h * w));
        let dw = gpu.storage(u64::from(cout * cin * k * k));
        let words = gpu.conv2d_dw_scratch_words(&p).expect("served");
        let scratch = (words > 0).then(|| gpu.storage(words));
        let fwd = min_ms(&gpu, &|| vec![gpu.step(CONV, &[&x, &wt, &y], &p, BATCH * cout * ho * wo)]);
        let dxt = min_ms(&gpu, &|| vec![gpu.step(CONV_DX, &[&dy, &wt, &dx], &p, BATCH * cin * h * w)]);
        let dwt = min_ms(&gpu, &|| gpu.conv2d_dw_steps(&dy, &x, &dw, scratch.as_ref(), &p).expect("served"));
        let flop = 2.0 * f64::from(BATCH * cout * ho * wo) * f64::from(cin * k * k);
        let rate = |ms: f64| flop / (ms * 1e-3) / 1e9;
        let tag = format!("({cin},{cout},{k},{s},{side})");
        eprintln!("{tag:<28} {units:>5} {fwd:>9.3} {dxt:>9.3} {dwt:>9.3} {:>8.0} {:>8.0} {:>8.0}", rate(fwd), rate(dxt), rate(dwt));
        let u = f64::from(units);
        t_fwd += u * fwd;
        t_dx += u * dxt;
        t_dw += u * dwt;
        flop_total += u * flop;
    }
    let total = t_fwd + t_dx + t_dw;
    let rate = 3.0 * flop_total / (total * 1e-3) / 1e9;
    eprintln!("step: fwd {t_fwd:.2} ms, dx {t_dx:.2} ms, dw {t_dw:.2} ms, total {total:.2} ms for {:.1} GFLOP -> {rate:.0} GFLOP/s", 3.0 * flop_total / 1e9);
    match roof {
        Some(g) => eprintln!("measured fp32 roof {g:.0} GFLOP/s: {:.1}% of it", 100.0 * rate / g),
        None => eprintln!("no measured roof on this device"),
    }
}
