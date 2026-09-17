// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gates for the three-factor learning rule.
//!
//! Two of these are not ordinary unit tests: "plasticity disabled does not
//! learn" and "a zero neuromodulator changes nothing" are entries in the
//! control matrix a learned behaviour has to survive. They are asserted
//! BIT-EXACTLY rather than approximately, which is only possible because the
//! learning rate and the neuromodulator are premultiplied into one scalar, so
//! a zero factor is an exact no-op instead of a small number.

use neuro::{Csc, DynamicalSystem, LifParams, Plastic, PlasticityParams, Port, Sites, SpikingNet, KERNELS};

fn random_csc(n: u32, avg_degree: usize, seed: u64) -> Csc {
    let mut rng = data::rng::Lcg::new(seed);
    let mut edges = Vec::new();
    for post in 0..n {
        let d = 1 + (rng.unit() * (2.0 * avg_degree as f32)) as usize;
        for _ in 0..d {
            let pre = (rng.unit() * n as f32) as u32 % n;
            edges.push((pre, post, rng.signed() * 0.4));
        }
    }
    Csc::from_edges(n, &edges).expect("well-formed edge list")
}

/// A network wired to fire a lot, so the traces have something to accumulate.
fn active_net(n: u32, seed: u64) -> (Csc, SpikingNet, Vec<f32>) {
    let csc = random_csc(n, 14, seed);
    let p = LifParams { dt_over_tau: 0.5, v_th: 0.3, refrac_ticks: 1, ..LifParams::default() };
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let net = SpikingNet::new(gpu, &csc, p).unwrap();
    let mut rng = data::rng::Lcg::new(seed ^ 0xA5);
    let drive = rng.vec_scaled(n as usize, 0.9);
    (csc, net, drive)
}

#[test]
fn the_eligibility_trace_matches_a_host_reference() {
    let (csc, mut net, drive) = active_net(192, 0x11);
    let pp = PlasticityParams { pre_decay: 0.8, post_decay: 0.7, elig_decay: 0.9, eta: 0.0, ..Default::default() };
    net.enable_plasticity(pp).unwrap();
    net.drive(Port::Drive, &drive).unwrap();

    let n = csc.n as usize;
    let (mut x_pre, mut x_post) = (vec![0.0f32; n], vec![0.0f32; n]);
    let mut elig = vec![0.0f32; csc.nnz()];
    let mut spike = vec![0.0f32; n];

    for tick in 0..25 {
        net.step();
        // Drive the host reference with the DEVICE's own spikes, so this
        // isolates the trace/eligibility arithmetic instead of re-testing the
        // forward dynamics the other gates already cover.
        net.read(Port::Spike, &mut spike).unwrap();
        for i in 0..n {
            x_pre[i] = x_pre[i] * pp.pre_decay + spike[i];
            x_post[i] = x_post[i] * pp.post_decay + spike[i];
        }
        for (post, &xp) in x_post.iter().enumerate() {
            let (lo, hi) = (csc.indptr[post] as usize, csc.indptr[post + 1] as usize);
            for k in lo..hi {
                elig[k] = elig[k] * pp.elig_decay + x_pre[csc.pre[k] as usize] * xp;
            }
        }
        let device = net.eligibility();
        let worst = device.iter().zip(&elig).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "tick {tick}: eligibility disagrees with the host reference by {worst:e}");
    }

    let reach = elig.iter().filter(|&&e| e.abs() > 1e-3).count();
    assert!(reach > csc.nnz() / 10, "only {reach} of {} synapses became eligible - this test would prove little", csc.nnz());
}

#[test]
fn a_zero_neuromodulator_leaves_every_weight_bit_identical() {
    let (_csc, mut net, drive) = active_net(160, 0x22);
    net.enable_plasticity(PlasticityParams { eta: 0.5, ..Default::default() }).unwrap();
    net.drive(Port::Drive, &drive).unwrap();

    let before = net.weights();
    for _ in 0..30 {
        // No `modulate` call: delta stays 0, so eta*delta is exactly 0.
        net.step();
    }
    let after = net.weights();

    assert!(net.eligibility().iter().any(|&e| e.abs() > 1e-3), "nothing became eligible, so this proves nothing");
    let moved = before.iter().zip(&after).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(moved, 0, "{moved} weights moved with no neuromodulator");
}

#[test]
fn disabling_plasticity_freezes_the_weights_even_under_reward() {
    let (_csc, mut net, drive) = active_net(160, 0x33);
    // Bounds wide enough that nothing saturates. An earlier version of this
    // test used the default +/-1 with eta 0.5, so the learning phase pinned
    // every eligible weight to the ceiling and the "frozen" phase could not
    // have moved them either -- it passed with `set_plasticity` deliberately
    // ignored. A control that cannot fail is not a control.
    net.enable_plasticity(PlasticityParams { eta: 0.05, w_min: -50.0, w_max: 50.0, ..Default::default() }).unwrap();
    net.drive(Port::Drive, &drive).unwrap();

    let start = net.weights();
    for _ in 0..15 {
        net.modulate(1.0);
        net.step();
    }
    assert!(net.plasticity(), "plasticity should be on by default once enabled");
    let learned = net.weights();

    // The learning phase must actually have moved weights, and must not have
    // parked them on the clamp, or the freeze below proves nothing.
    let moved_while_learning = start.iter().zip(&learned).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert!(moved_while_learning > start.len() / 10, "only {moved_while_learning} of {} weights moved while learning", start.len());
    assert!(learned.iter().all(|w| w.abs() < 49.0), "weights saturated; widen the bounds or lower eta");

    net.set_plasticity(false);
    assert!(!net.plasticity());
    for _ in 0..15 {
        net.modulate(1.0);
        net.step();
    }
    let frozen = net.weights();

    let moved = learned.iter().zip(&frozen).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(moved, 0, "{moved} weights moved while plasticity was disabled");
}

#[test]
fn a_reward_moves_eligible_synapses_and_the_sign_follows_the_modulator() {
    let (_csc, mut net, drive) = active_net(160, 0x44);
    let pp = PlasticityParams { eta: 0.2, w_min: -4.0, w_max: 4.0, ..Default::default() };
    net.enable_plasticity(pp).unwrap();

    let run = |net: &mut SpikingNet, delta: f32, ticks: usize| {
        net.reset(0);
        net.drive(Port::Drive, &drive).unwrap();
        let before = net.weights();
        for _ in 0..ticks {
            net.modulate(delta);
            net.step();
        }
        let after = net.weights();
        before.iter().zip(&after).map(|(a, b)| b - a).sum::<f32>()
    };

    // Over a single tick the two runs see an IDENTICAL spike train - the
    // weight update lands after the forward pass, so it cannot yet have
    // changed anything - and eligibility is a product of two non-negative
    // traces. So the update rule itself must mirror exactly under a mirrored
    // modulator. This is the assertion about the RULE.
    let up1 = run(&mut net, 1.0, 1);
    let down1 = run(&mut net, -1.0, 1);
    assert!((up1 + down1).abs() <= 1e-6 * up1.abs().max(1.0), "one tick should mirror exactly, got {up1:e} and {down1:e}");

    // Over many ticks it must NOT be expected to mirror, and that is a
    // property of the system rather than a defect: the loop is closed.
    // Strengthened synapses make their targets fire more, which makes them
    // more eligible still; weakened ones do the opposite. The two runs'
    // spike trains diverge after the first update. An earlier version of this
    // test asserted cancellation over 25 ticks and failed at 1.7e3 vs
    // -2.0e3 - the assertion was wrong, not the rule.
    let up = run(&mut net, 1.0, 25);
    let down = run(&mut net, -1.0, 25);
    assert!(up > 1e-3, "a positive reward moved nothing (sum {up:e})");
    assert!(down < -1e-3, "a negative reward moved nothing (sum {down:e})");
}

#[test]
fn the_weight_clamp_holds_under_sustained_reward() {
    let (_csc, mut net, drive) = active_net(128, 0x55);
    // A deliberately unstable configuration: a large learning rate and a
    // standing reward. Without the clamp this diverges.
    let pp = PlasticityParams { eta: 5.0, w_min: -0.5, w_max: 0.5, ..Default::default() };
    net.enable_plasticity(pp).unwrap();
    net.drive(Port::Drive, &drive).unwrap();
    for _ in 0..60 {
        net.modulate(1.0);
        net.step();
    }
    let w = net.weights();
    assert!(w.iter().all(|v| v.is_finite()), "weights diverged to a non-finite value");
    let worst = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(worst <= 0.5 + 1e-6, "clamp breached: |w| reached {worst}");
    assert!(w.iter().any(|&v| v >= 0.5 - 1e-6), "nothing reached the ceiling, so the clamp was never exercised");
}

#[test]
fn cpu_and_gpu_agree_on_the_learning_path() {
    let csc = random_csc(256, 12, 0x66);
    let p = LifParams { dt_over_tau: 0.5, v_th: 0.3, refrac_ticks: 1, ..LifParams::default() };
    let pp = PlasticityParams { eta: 0.3, ..Default::default() };
    let mut rng = data::rng::Lcg::new(0x77);
    let drive = rng.vec_scaled(csc.n as usize, 0.9);

    let run = |gpu: gpu_core::Gpu| {
        let mut net = SpikingNet::new(gpu, &csc, p).unwrap();
        net.enable_plasticity(pp).unwrap();
        net.drive(Port::Drive, &drive).unwrap();
        for _ in 0..30 {
            net.modulate(0.8);
            net.step();
        }
        (net.weights(), net.eligibility())
    };

    let (gw, ge) = run(gpu_core::testgpu::dev(&KERNELS));
    let (cw, ce) = run(gpu_core::Gpu::new_cpu(&KERNELS));

    let dw = gw.iter().zip(&cw).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let de = ge.iter().zip(&ce).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(dw < 1e-4, "backends disagree on learned weights by {dw:e}");
    assert!(de < 1e-4, "backends disagree on eligibility by {de:e}");
    assert!(gw.iter().zip(&cw).any(|(a, b)| a != b || *a != 0.0), "both backends learned nothing");
}

// ---------------------------------------------------------------------------
// Sited plasticity: which synapses may learn, and what tells them to.
//
// The rule above is a global one: every synapse in the animal is eligible and
// one broadcast scalar decides whether all of them learn. That is a useful
// instrument and it is not what a fly does. In a fly the best understood
// learning happens at one identified population of synapses, and the third
// factor is not an experimenter's scalar but the firing of identified
// dopaminergic neurons into one compartment. These gate that distinction.
// ---------------------------------------------------------------------------

/// Reserved compartment 0 means "this synapse does not learn", so the mask is
/// exact rather than a small update: a connectome is 15.9M edges and 0.1% of
/// them are plastic, so "the rest barely moved" is not good enough to
/// distinguish a sited rule from a global one that happens to be weak.
#[test]
fn only_the_sited_synapses_learn_and_every_other_weight_is_bit_identical() {
    let (csc, mut net, drive) = active_net(192, 0x81);
    let before = net.weights();

    // Make the edges onto the first 10 neurons plastic, and nothing else.
    let plastic: Vec<u32> = (0..csc.indptr[10]).collect();
    assert!(plastic.len() > 20, "the fixture must have edges to make plastic");
    let mut sites = Sites::new(csc.nnz());
    let c = sites.compartment(&[], 1.0);
    sites.assign(&plastic, c).unwrap();

    net.enable_plasticity_at(PlasticityParams { eta: 0.5, ..Default::default() }, sites).unwrap();
    net.drive(Port::Drive, &drive).unwrap();
    for _ in 0..30 {
        net.modulate(1.0);
        net.step();
    }

    let after = net.weights();
    let moved = plastic.iter().filter(|&&k| after[k as usize] != before[k as usize]).count();
    assert!(moved > 10, "the sited synapses did not learn ({moved} of {} moved)", plastic.len());
    for k in plastic.len()..after.len() {
        assert_eq!(after[k], before[k], "edge {k} is outside every site and must be bit-identical");
    }
}

/// The third factor arrives from named cells, and it stays in its compartment.
///
/// This is the property that makes "sugar" and "shock" different signals
/// rather than the same number with a different sign: two compartments of the
/// same network, each driven by its own neuron, one of which is silent.
#[test]
fn a_compartment_is_modulated_by_its_own_neurons_and_not_by_anothers() {
    let (csc, mut net, _drive) = active_net(192, 0x82);
    let before = net.weights();

    // Two disjoint sets of synapses, each answering to one source neuron.
    let a: Vec<u32> = (csc.indptr[20]..csc.indptr[30]).collect();
    let b: Vec<u32> = (csc.indptr[30]..csc.indptr[40]).collect();
    let mut sites = Sites::new(csc.nnz());
    let ca = sites.compartment(&[0], 1.0);
    let cb = sites.compartment(&[1], 1.0);
    sites.assign(&a, ca).unwrap();
    sites.assign(&b, cb).unwrap();

    net.enable_plasticity_at(PlasticityParams { eta: 0.5, ..Default::default() }, sites).unwrap();

    // Neuron 0 is driven hard and fires; neuron 1 is left alone and does not.
    let mut drive = vec![0.4f32; csc.n as usize];
    drive[0] = 40.0;
    drive[1] = 0.0;
    net.drive(Port::Drive, &drive).unwrap();
    for _ in 0..30 {
        net.step();
    }

    let m = net.modulator();
    assert_eq!(m.len(), 3, "an inert compartment plus the two configured");
    assert_eq!(m[0], 0.0, "compartment 0 is reserved and must never carry a modulator");
    assert!(m[ca as usize] > 0.0, "the driven neuron's compartment saw no dopamine ({})", m[ca as usize]);
    assert_eq!(m[cb as usize], 0.0, "the silent neuron's compartment was modulated anyway");

    let after = net.weights();
    assert!(a.iter().any(|&k| after[k as usize] != before[k as usize]), "the modulated compartment did not learn");
    for &k in &b {
        assert_eq!(after[k as usize], before[k as usize], "edge {k} is in an unmodulated compartment and must not move");
    }
}
