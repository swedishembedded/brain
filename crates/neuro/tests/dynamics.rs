// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The runtime's correctness gates.
//!
//! `gradcheck` cannot gate this crate: it finite-differences a scalar loss
//! against analytic gradients, and a spiking runtime whose learning rule is
//! local has neither. That is a deliberate, recorded exception rather than an
//! omission, and these three checks are what stand in for it:
//!
//!   1. the membrane dynamics match an INDEPENDENTLY DERIVED closed form, to
//!      the fp32 noise floor rather than to a tolerance;
//!   2. the CPU and GPU backends agree on a real, irregular connectome -- the
//!      cross-backend discipline `make parity` applies to every other model;
//!   3. a restored snapshot replays bit-for-bit, which is what makes "stop it,
//!      start it, and it still knows what it learned" checkable.

use neuro::{Csc, DynamicalSystem, LifParams, Port, SpikingNet, KERNELS};

/// A deterministic, irregular connectome: degrees vary per neuron and weights
/// are signed, so neither the gather's edge ranges nor its accumulation order
/// is uniform across neurons.
fn random_csc(n: u32, avg_degree: usize, seed: u64) -> Csc {
    let mut rng = data::rng::Lcg::new(seed);
    let mut edges = Vec::new();
    for post in 0..n {
        // Degree in [1, 2*avg): the p99/max spread of a real connectome is
        // what makes a workgroup-per-neuron gather worth testing at all.
        let d = 1 + (rng.unit() * (2.0 * avg_degree as f32)) as usize;
        for _ in 0..d {
            let pre = (rng.unit() * n as f32) as u32 % n;
            edges.push((pre, post, rng.signed() * 0.5));
        }
    }
    Csc::from_edges(n, &edges).expect("well-formed edge list")
}

#[test]
fn membrane_matches_the_closed_form_for_a_constant_input() {
    // One neuron, no edges: the gather contributes nothing and the membrane is
    // driven purely by the external port, which is the regime the closed form
    // describes. The threshold is placed out of reach so the trajectory is the
    // pure exponential rather than a reset sawtooth.
    let p = LifParams { dt_over_tau: 0.2, v_rest: -0.3, v_reset: -0.6, v_th: 1.0e6, r: 2.0, refrac_ticks: 0 };
    let csc = Csc::from_edges(1, &[]).unwrap();
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let mut net = SpikingNet::new(gpu, &csc, p).unwrap();

    let current = 0.35f32;
    net.drive(Port::Drive, &[current]).unwrap();

    let mut v = [0.0f32; 1];
    for k in 1..=64u32 {
        net.step();
        net.read(Port::Membrane, &mut v).unwrap();
        let want = p.analytic_v(p.v_rest, current, k);
        let err = (v[0] - want).abs();
        // fp32 noise over a 64-step geometric recurrence, not a fitted
        // tolerance: the kernel and the closed form do the same arithmetic in
        // a different order.
        assert!(err < 1e-6, "tick {k}: device {} vs closed form {want} (err {err:e})", v[0]);
    }
}

#[test]
fn a_neuron_fires_at_threshold_and_then_stays_silent_while_refractory() {
    let p = LifParams { dt_over_tau: 1.0, v_rest: 0.0, v_reset: 0.0, v_th: 0.5, r: 1.0, refrac_ticks: 3 };
    let csc = Csc::from_edges(1, &[]).unwrap();
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let mut net = SpikingNet::new(gpu, &csc, p).unwrap();
    net.drive(Port::Drive, &[1.0]).unwrap();

    let mut s = [0.0f32; 1];
    let mut fired = Vec::new();
    for tick in 0..12 {
        net.step();
        net.read(Port::Spike, &mut s).unwrap();
        if s[0] == 1.0 {
            fired.push(tick);
        }
    }
    // a = 1 drives straight to v_inf = 1.0 > 0.5 every tick it is allowed to
    // integrate, so the cell fires, sits out three ticks, and fires again:
    // period 4. A refractory period that only gated the SPIKE rather than
    // clamping the membrane would produce period 1 here.
    assert_eq!(fired, vec![0, 4, 8], "expected one spike every refrac_ticks + 1");
}

#[test]
fn the_gather_matches_a_host_sparse_matvec() {
    // The cross-backend test below cannot catch a gather that is wrong the
    // SAME way on both backends, and the analytic test runs on an empty
    // connectome where the gather contributes nothing. So the gather needs its
    // own oracle: the sparse mat-vec, computed on the host straight from the
    // CSC arrays, with no kernel involved.
    //
    // The spike vector must be NON-ZERO for this to test anything -- an
    // earlier version of this test set an unreachable threshold, so nothing
    // ever fired and it compared zero against zero. It passed happily with
    // the gather's loop bound deliberately broken. Hence the low threshold
    // here, and the assertion that neurons actually fired.
    let csc = random_csc(300, 19, 0xBEEF);
    let p = LifParams { dt_over_tau: 1.0, v_rest: 0.0, v_reset: 0.0, v_th: 0.25, r: 1.0, refrac_ticks: 0 };
    let n = csc.n as usize;

    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let mut net = SpikingNet::new(gpu, &csc, p).unwrap();

    let mut rng = data::rng::Lcg::new(3);
    let drive = rng.vec_scaled(n, 1.0);
    net.drive(Port::Drive, &drive).unwrap();
    net.step();

    let mut spike = vec![0.0f32; n];
    net.read(Port::Spike, &mut spike).unwrap();
    let fired = spike.iter().filter(|&&s| s == 1.0).count();
    assert!(fired > n / 8, "only {fired} of {n} neurons fired - the gather would see an almost-empty spike vector and this test would prove nothing");

    // Second tick: the gather sees the spike vector we just read.
    net.step();
    let mut isyn = vec![0.0f32; n];
    net.read(Port::Current, &mut isyn).unwrap();

    let mut worst = 0.0f32;
    let mut nonzero = 0usize;
    for (post, &got) in isyn.iter().enumerate() {
        let lo = csc.indptr[post] as usize;
        let hi = csc.indptr[post + 1] as usize;
        let want: f32 = (lo..hi).map(|k| csc.w[k] * spike[csc.pre[k] as usize]).sum();
        if want != 0.0 {
            nonzero += 1;
        }
        worst = worst.max((got - want).abs());
    }
    assert!(nonzero > n / 8, "only {nonzero} neurons had non-zero expected current");
    assert!(worst < 1e-5, "gather disagrees with a host sparse mat-vec by {worst:e}");
}

#[test]
fn cpu_and_gpu_agree_on_an_irregular_connectome() {
    let csc = random_csc(512, 24, 0xC0FFEE);
    let p = LifParams { dt_over_tau: 0.25, v_th: 0.4, refrac_ticks: 2, ..LifParams::default() };
    let mut rng = data::rng::Lcg::new(7);
    let drive = rng.vec_scaled(csc.n as usize, 0.6);

    let run = |gpu: gpu_core::Gpu| {
        let mut net = SpikingNet::new(gpu, &csc, p).unwrap();
        net.drive(Port::Drive, &drive).unwrap();
        let mut spikes = Vec::new();
        let mut s = vec![0.0f32; csc.n as usize];
        for _ in 0..40 {
            net.step();
            net.read(Port::Spike, &mut s).unwrap();
            spikes.push(s.iter().sum::<f32>());
        }
        (spikes, net.snapshot().v)
    };

    let (gpu_counts, gpu_v) = run(gpu_core::testgpu::dev(&KERNELS));
    let (cpu_counts, cpu_v) = run(gpu_core::Gpu::new_cpu(&KERNELS));

    // Spike counts are discrete: a backend that disagreed about even one
    // threshold crossing would show up here as an integer difference, not a
    // rounding one.
    assert_eq!(gpu_counts, cpu_counts, "backends disagree on which neurons fired");
    let worst = gpu_v.iter().zip(&cpu_v).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1e-5, "membrane disagreement {worst:e} between backends");
}

#[test]
fn a_restored_snapshot_replays_bit_for_bit() {
    let csc = random_csc(256, 16, 0x5EED);
    let p = LifParams { dt_over_tau: 0.3, v_th: 0.35, refrac_ticks: 1, ..LifParams::default() };
    let mut rng = data::rng::Lcg::new(11);
    let drive = rng.vec_scaled(csc.n as usize, 0.7);

    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let mut net = SpikingNet::new(gpu, &csc, p).unwrap();
    net.drive(Port::Drive, &drive).unwrap();
    for _ in 0..20 {
        net.step();
    }

    let saved = net.snapshot();
    let mut first = Vec::new();
    let mut s = vec![0.0f32; csc.n as usize];
    for _ in 0..20 {
        net.step();
        net.read(Port::Spike, &mut s).unwrap();
        first.push(s.clone());
    }

    net.restore(&saved).unwrap();
    net.drive(Port::Drive, &drive).unwrap();
    let mut second = Vec::new();
    for _ in 0..20 {
        net.step();
        net.read(Port::Spike, &mut s).unwrap();
        second.push(s.clone());
    }

    assert_eq!(saved.tick, 20, "snapshot should carry the tick it was taken at");
    // Bit-for-bit, not approximately: the gather sums each neuron's edges in a
    // fixed order, so a replay from identical state is identical arithmetic.
    for (i, (a, b)) in first.iter().zip(&second).enumerate() {
        assert_eq!(a, b, "replay diverged at tick {i}");
    }
}

#[test]
fn a_malformed_connectome_is_refused_rather_than_silently_misread() {
    // A short indptr does not fail a kernel: it reads another neuron's edge
    // range and produces a plausible number. This is the only layer that can
    // catch it, so it must.
    let bad = Csc { n: 3, indptr: vec![0, 1], pre: vec![0], w: vec![1.0] };
    assert!(bad.validate().is_err(), "a short indptr must be refused");

    let out_of_range = Csc { n: 2, indptr: vec![0, 1, 1], pre: vec![7], w: vec![1.0] };
    assert!(out_of_range.validate().is_err(), "an out-of-range presynaptic index must be refused");

    assert!(Csc::from_edges(2, &[(0, 5, 1.0)]).is_err(), "an edge past the neuron count must be refused");
}
