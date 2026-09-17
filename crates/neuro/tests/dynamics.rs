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
    let p = LifParams { dt_over_tau: 0.2, v_rest: -0.3, v_reset: -0.6, v_th: 1.0e6, r: 2.0, refrac_ticks: 0, dt_over_tau_syn: 1.0, dt_over_tau_inh: 1.0, ..LifParams::default() };
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
    let p = LifParams { dt_over_tau: 1.0, v_rest: 0.0, v_reset: 0.0, v_th: 0.5, r: 1.0, refrac_ticks: 3, dt_over_tau_syn: 1.0, dt_over_tau_inh: 1.0, ..LifParams::default() };
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
    let p = LifParams { dt_over_tau: 1.0, v_rest: 0.0, v_reset: 0.0, v_th: 0.25, r: 1.0, refrac_ticks: 0, dt_over_tau_syn: 1.0, dt_over_tau_inh: 1.0, ..LifParams::default() };
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

    // AGREEMENT IS VACUOUS WITHOUT ACTIVITY. Two backends that both produce
    // an all-zero spike train agree perfectly and prove nothing about the
    // spiking path, so assert the network actually fired before comparing.
    // This test passed that way until this assertion was added.
    let fired: f32 = gpu_counts.iter().sum();
    assert!(fired > 0.0, "neither backend fired at all, so the agreement below is between two empty trains");
    assert!(
        gpu_counts.iter().filter(|&&c| c > 0.0).count() > gpu_counts.len() / 4,
        "only {} of {} ticks had any spike; the comparison is nearly empty",
        gpu_counts.iter().filter(|&&c| c > 0.0).count(),
        gpu_counts.len()
    );

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

    // Same vacuity trap as the cross-backend test: two all-zero spike trains
    // replay each other perfectly. Assert the run had content first.
    let total: f32 = first.iter().flat_map(|v| v.iter()).sum();
    assert!(total > 0.0, "nothing fired in the replayed window, so equality below is between empty trains");
    assert!(
        first.iter().filter(|v| v.iter().any(|&s| s > 0.5)).count() > first.len() / 4,
        "too few ticks had any spike for the replay comparison to mean anything"
    );

    // Bit-for-bit, not approximately: the gather sums each neuron's edges in a
    // fixed order, so a replay from identical state is identical arithmetic.
    for (i, (a, b)) in first.iter().zip(&second).enumerate() {
        assert_eq!(a, b, "replay diverged at tick {i}");
    }
}

#[test]
fn a_shuffled_graph_keeps_its_in_degrees_and_weights_but_loses_its_structure() {
    let csc = random_csc(400, 20, 0xD1CE);
    let shuffled = csc.shuffled_sources(0x5EED);

    // The properties the structural control depends on: same shape, same
    // in-degree per neuron, same multiset of weights. A "shuffle" that changed
    // any of these would be testing something other than structure.
    assert_eq!(shuffled.n, csc.n);
    assert_eq!(shuffled.nnz(), csc.nnz());
    assert_eq!(shuffled.indptr, csc.indptr, "in-degree must be preserved exactly");
    assert_eq!(shuffled.w, csc.w, "weights stay with their postsynaptic slot");
    assert_eq!(shuffled.in_degrees(), csc.in_degrees());
    shuffled.validate().expect("a shuffled graph is still a valid graph");

    // And it must actually be shuffled. A no-op shuffle would pass every
    // assertion above and silently make the control identical to the test.
    let moved = csc.pre.iter().zip(&shuffled.pre).filter(|(a, b)| a != b).count();
    assert!(
        moved > csc.nnz() / 2,
        "only {moved} of {} sources moved; this is not a shuffle",
        csc.nnz()
    );

    // Deterministic: the same seed reproduces the same graph, or a shuffled
    // condition could not be replayed.
    assert_eq!(csc.shuffled_sources(0x5EED).pre, shuffled.pre);
    assert_ne!(csc.shuffled_sources(0x1234).pre, shuffled.pre, "different seeds must differ");
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

#[test]
fn a_synaptic_time_constant_makes_a_spike_outlast_its_tick() {
    // Two neurons, one edge: 0 -> 1. Neuron 0 is driven over threshold for a
    // single tick and then released, so exactly one spike crosses the synapse.
    let csc = Csc::from_edges(2, &[(0, 1, 1.0)]).unwrap();
    let base = LifParams {
        dt_over_tau: 1.0,
        v_rest: 0.0,
        v_reset: 0.0,
        // Out of reach, so neuron 1 integrates the current without ever firing
        // and resetting the thing being measured.
        v_th: 1.0e6,
        r: 1.0,
        refrac_ticks: 0,
        dt_over_tau_syn: 1.0,
        dt_over_tau_inh: 1.0,
        ..LifParams::default()
    };

    /// Neuron 1's synaptic current for the first `n` ticks after one spike.
    fn current_after_one_spike(csc: &Csc, p: LifParams, n: usize) -> Vec<f32> {
        let mut net = SpikingNet::new(gpu_core::testgpu::dev(&KERNELS), csc, p).unwrap();
        let mut out = Vec::new();
        let mut isyn = vec![0.0f32; 2];
        for k in 0..n {
            // Neuron 0 fires on the first tick only. Its threshold is the
            // unreachable one too, so it is driven ABOVE it deliberately.
            net.drive(Port::Drive, &[if k == 0 { 2.0e6 } else { 0.0 }, 0.0]).unwrap();
            net.step();
            net.read(Port::Current, &mut isyn).unwrap();
            out.push(isyn[1]);
        }
        out
    }

    // THE CONTROL: an instantaneous synapse. The current appears for exactly
    // one tick and is gone. Without this, "the decayed one lasted longer" is
    // equally satisfied by a network where nothing ever arrives at all.
    let instant = current_after_one_spike(&csc, base, 6);
    let arrived = instant.iter().filter(|c| **c > 0.0).count();
    assert_eq!(arrived, 1, "an instantaneous synapse should carry current for one tick: {instant:?}");

    // Half of it survives each tick, so the current decays geometrically
    // instead of vanishing.
    let slow = current_after_one_spike(&csc, LifParams { dt_over_tau_syn: 0.5, ..base }, 6);
    let carried = slow.iter().filter(|c| **c > 1e-6).count();
    assert!(carried >= 4, "a decaying synapse should carry current for several ticks: {slow:?}");
    for pair in slow.windows(2).skip(1) {
        assert!(pair[1] < pair[0], "the current should decay monotonically: {slow:?}");
        assert!((pair[1] / pair[0] - 0.5).abs() < 1e-5, "each tick should keep half: {slow:?}");
    }

    // And the peak is the same either way: the time constant spreads the
    // charge, it does not add any.
    assert_eq!(
        slow.iter().cloned().fold(f32::MIN, f32::max),
        instant.iter().cloned().fold(f32::MIN, f32::max),
        "a synaptic time constant must not change how much current a spike delivers"
    );
}

#[test]
fn excitation_and_inhibition_carry_their_own_time_constants() {
    // Two networks that differ ONLY in the sign of their single synapse, run
    // with a fast excitatory and a slow inhibitory time constant. If the split
    // were not real, both would decay at whichever rate the kernel actually
    // used and the two traces would be mirror images.
    let base = LifParams {
        dt_over_tau: 1.0,
        v_rest: 0.0,
        v_reset: 0.0,
        v_th: 1.0e6,
        r: 1.0,
        refrac_ticks: 0,
        // Excitation gone in one tick; inhibition keeps half each tick.
        dt_over_tau_syn: 1.0,
        dt_over_tau_inh: 0.5,
        ..LifParams::default()
    };

    fn trace(weight: f32, p: LifParams) -> Vec<f32> {
        let csc = Csc::from_edges(2, &[(0, 1, weight)]).unwrap();
        let mut net = SpikingNet::new(gpu_core::testgpu::dev(&KERNELS), &csc, p).unwrap();
        let mut isyn = vec![0.0f32; 2];
        (0..6)
            .map(|k| {
                net.drive(Port::Drive, &[if k == 0 { 2.0e6 } else { 0.0 }, 0.0]).unwrap();
                net.step();
                net.read(Port::Current, &mut isyn).unwrap();
                isyn[1]
            })
            .collect()
    }

    let exc = trace(1.0, base);
    let inh = trace(-1.0, base);

    // The excitatory one is instantaneous, which is the control: it fixes what
    // "one tick and gone" looks like on this path.
    assert_eq!(exc.iter().filter(|c| c.abs() > 1e-6).count(), 1, "excitation should last one tick: {exc:?}");
    // The inhibitory one, over the SAME graph shape and the same spike,
    // persists - so the two are not sharing a decay.
    assert!(inh.iter().filter(|c| c.abs() > 1e-6).count() >= 4, "inhibition should persist: {inh:?}");
    for pair in inh.windows(2).skip(1) {
        assert!((pair[1] / pair[0] - 0.5).abs() < 1e-5, "inhibition should keep half each tick: {inh:?}");
    }
    // The first non-zero entry, not the first: the gather reads LAST tick's
    // spikes, so nothing has crossed the synapse when tick 0 is read back.
    let first = *inh.iter().find(|c| c.abs() > 1e-6).expect("something should arrive");
    assert!(first < 0.0, "an inhibitory synapse should deliver negative current, got {first}");
}

/// Inter-spike intervals of one neuron under a constant drive.
fn intervals(p: LifParams, drive: f32, ticks: u32) -> Vec<u32> {
    let csc = Csc::from_edges(1, &[]).unwrap();
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let mut net = SpikingNet::new(gpu, &csc, p).unwrap();
    net.drive(Port::Drive, &[drive]).unwrap();
    let mut s = [0.0f32; 1];
    let mut fired = Vec::new();
    for tick in 0..ticks {
        net.step();
        net.read(Port::Spike, &mut s).unwrap();
        if s[0] == 1.0 {
            fired.push(tick);
        }
    }
    fired.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Spike-frequency adaptation: a cell under a constant current fires fast and
/// then slows to a steady rate.
///
/// This is the ingredient a network oscillator needs and the LIF above does
/// not have. A pair of populations inhibiting each other cannot alternate
/// unless something makes the active one give way on a timescale slower than
/// the synapse: with nothing but a membrane and a threshold, whichever side
/// wins the first tick wins every tick, and what comes out is a cord that
/// coordinates aperiodically - which is exactly what `fly`'s gait analysis
/// measured before this existed.
///
/// The CONTROL is the same neuron with the increment at zero. Without it
/// "the intervals grew" is also satisfied by a neuron that is simply running
/// out of drive.
#[test]
fn adaptation_slows_a_neuron_down_and_its_absence_does_not() {
    let base = LifParams {
        dt_over_tau: 0.5,
        v_th: 1.0,
        r: 1.0,
        refrac_ticks: 0,
        adapt_decay: 0.9,
        adapt_increment: 0.35,
        ..LifParams::default()
    };
    let adapting = intervals(base, 3.0, 400);
    let plain = intervals(LifParams { adapt_increment: 0.0, ..base }, 3.0, 400);

    assert!(adapting.len() > 4 && plain.len() > 4, "both cells have to fire repeatedly: {adapting:?} / {plain:?}");
    assert!(
        plain.iter().all(|&i| i == plain[0]),
        "without adaptation a constant current has to give a constant rate, got {plain:?}"
    );
    assert!(
        adapting[0] < *adapting.last().unwrap(),
        "adaptation has to lengthen the interval, got {adapting:?}"
    );
    // It SETTLES rather than running away: an adaptation current that never
    // reached equilibrium would silence the cell instead of slowing it, and a
    // silenced population cannot take its turn in an alternation.
    let tail = &adapting[adapting.len() - 3..];
    assert!(tail.iter().all(|&i| i == tail[0]), "the adapted rate has to settle, got {adapting:?}");
    assert!(*tail.last().unwrap() > adapting[0], "the settled rate has to be slower than the onset rate");
}

/// The adaptation current is a geometric decay with a closed form, like the
/// membrane it subtracts from - so its OFF state is exact rather than small.
#[test]
fn adaptation_off_is_bit_identical_to_a_cord_that_never_had_it() {
    // Every default is the un-adapted model, so a network built from defaults
    // has to reproduce what this runtime did before adaptation existed, to the
    // bit. Anything less would silently restate every measurement in the fly
    // ledger.
    assert_eq!(LifParams::default().adapt_increment, 0.0);
    let csc = random_csc(64, 6, 0xADA9);
    let p = LifParams { dt_over_tau: 0.4, v_th: 0.3, refrac_ticks: 1, ..LifParams::default() };

    let run = |p: LifParams| {
        let gpu = gpu_core::testgpu::dev(&KERNELS);
        let mut net = SpikingNet::new(gpu, &csc, p).unwrap();
        net.drive(Port::Drive, &vec![0.8; 64]).unwrap();
        let mut v = vec![0.0f32; 64];
        let mut trace = Vec::new();
        for _ in 0..40 {
            net.step();
            net.read(Port::Membrane, &mut v).unwrap();
            trace.extend_from_slice(&v);
        }
        trace
    };
    // A decay of zero with no increment is the same no-op as any other decay:
    // nothing ever enters the variable, so nothing ever leaves it.
    assert_eq!(run(p), run(LifParams { adapt_decay: 0.99, ..p }), "an empty adaptation current is not inert");
}

/// Per-neuron physiology, and the control that says adding it changed nothing
/// until it was asked to.
///
/// A connectome fixes who contacts whom and says nothing about time constants
/// or excitability, so those have to be supplied - and a model where every
/// cell shares one membrane is a CHOICE, not a neutral starting point. These
/// gate the two halves of that: the uniform model must survive exactly, and
/// a cell given its own time constant must actually get one.
#[test]
fn unit_cell_scales_are_bit_identical_to_a_network_that_never_had_them() {
    let csc = random_csc(96, 6, 0x71);
    let p = LifParams { dt_over_tau: 0.4, v_th: 0.5, ..LifParams::default() };
    let drive: Vec<f32> = (0..96).map(|i| 0.2 + (i % 7) as f32 * 0.05).collect();

    let run = |scales: Option<(Vec<f32>, Vec<f32>)>| {
        let mut net = SpikingNet::new(gpu_core::testgpu::dev(&KERNELS), &csc, p).unwrap();
        if let Some((t, g)) = scales {
            net.set_cell_scales(&t, &g).unwrap();
        }
        net.drive(Port::Drive, &drive).unwrap();
        let mut v = vec![0.0f32; 96];
        let mut trace = Vec::new();
        for _ in 0..40 {
            net.step();
            net.read(Port::Membrane, &mut v).unwrap();
            trace.extend_from_slice(&v);
        }
        trace
    };

    let plain = run(None);
    let unit = run(Some((vec![1.0; 96], vec![1.0; 96])));
    assert_eq!(plain, unit, "unit scales must be an exact no-op, not an approximation");
}

#[test]
fn a_cell_given_its_own_time_constant_integrates_at_its_own_rate() {
    let csc = random_csc(64, 4, 0x72);
    // No threshold crossing: this is about the membrane, not about spikes.
    let p = LifParams { dt_over_tau: 0.2, v_th: 1e6, r: 1.0, v_rest: 0.0, ..LifParams::default() };
    let mut net = SpikingNet::new(gpu_core::testgpu::dev(&KERNELS), &csc, p).unwrap();
    let mut tau = vec![1.0f32; 64];
    tau[7] = 2.0; // twice as fast
    tau[9] = 0.5; // twice as slow
    let mut gain = vec![1.0f32; 64];
    gain[11] = 3.0; // three times as excitable
    net.set_cell_scales(&tau, &gain).unwrap();
    net.drive(Port::Drive, &vec![1.0; 64]).unwrap();
    for _ in 0..12 {
        net.step();
    }
    let mut v = vec![0.0f32; 64];
    net.read(Port::Membrane, &mut v).unwrap();

    // The closed form for a constant input, per neuron.
    let want = |a: f32, r: f32| {
        let v_inf = r;
        v_inf * (1.0 - (1.0f32 - a).powi(12))
    };
    for (i, (a, r)) in [(7usize, (0.4f32, 1.0f32)), (9, (0.1, 1.0)), (11, (0.2, 3.0)), (0, (0.2, 1.0))] {
        let e = (v[i] - want(a, r)).abs();
        assert!(e < 1e-4, "neuron {i}: {} against the closed form {}, off by {e:e}", v[i], want(a, r));
    }
    assert!(v[7] > v[0], "a faster membrane is nearer its steady state after the same time");
    assert!(v[9] < v[0], "a slower membrane is further from it");
    assert!(v[11] > 2.0 * v[0], "a more excitable cell reaches a higher steady state");

    assert!(net.set_cell_scales(&vec![1.0; 64], &vec![0.0; 64]).is_err(), "a gain of zero is a deleted cell");
    assert!(net.set_cell_scales(&vec![1.0; 63], &vec![1.0; 64]).is_err(), "the wrong width is refused");
}
