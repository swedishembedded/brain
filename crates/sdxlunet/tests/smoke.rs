// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight-free smoke test at toy dims — the porting playbook's §4 rung.
//!
//! Runs the FULL graph shape (down/mid/up, skip concats, both GroupNorm
//! epsilons, self- and cross-attention, GEGLU) at `UNetConfig::tiny`, so every
//! step KIND is dispatched in well under a second. This is what catches
//! buffer-sizing and binding mistakes; the real-weights parity test
//! (`tests/parity.rs`) catches convention mistakes.
//!
//! Needs no fixture and no checkpoint, so it always runs.

use sdxlunet::config::UNetConfig;
use sdxlunet::init::init_weights;
use sdxlunet::model::{Unet, KERNELS};

/// Is THIS backend run-to-run deterministic on THIS graph? Measured, by
/// submitting the same recorded graph twice and comparing bits.
///
/// It is a measurement and not a capability flag on purpose. An earlier version
/// gated on `DeviceCaps::workgroup_reductions` as a proxy for "is the CPU JIT",
/// after a report that `BRAIN_DEVICE=cpu` diverged run-to-run on this graph.
/// That proxy is wrong twice over: it skips `cpu0`, which the same report called
/// deterministic, and it would keep skipping forever once the backend were
/// fixed. Re-measured on this box (48-thread `BRAIN_DEVICE=cpu`, both `taps`
/// modes, `layers_per_block` 1 and 2, 8 submits each) the divergence does NOT
/// reproduce — 0 of 1024 outputs differ in every configuration — so the proxy
/// was also costing the CPU leg its only comparison coverage.
///
/// This probe restores that coverage wherever the backend earns it and still
/// refuses to assert what a genuinely racy one cannot provide.
fn deterministic_device(m: &Unet, i: &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)) -> bool {
    let a = m.run(&i.0, 601.0, &i.1, &i.2, &i.3);
    let b = m.run(&i.0, 601.0, &i.1, &i.2, &i.3);
    let d = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    if d != 0 {
        eprintln!(
            "SKIP: this backend is not run-to-run deterministic on this graph \
             ({d} of {} outputs differ between two submits of ONE recorded graph)",
            a.len()
        );
    }
    d == 0
}

fn tiny(taps: bool) -> Unet {
    let cfg = UNetConfig::tiny();
    let w = init_weights(&cfg, 7);
    Unet::new(gpu_core::testgpu::dev(&KERNELS), cfg, &w, 16, 16, 9, taps)
}

fn inputs(m: &Unet) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let c = m.config();
    let sample: Vec<f32> = (0..(c.in_channels * 16 * 16) as usize)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();
    let enc: Vec<f32> =
        (0..(9 * c.cross_attention_dim) as usize).map(|i| ((i as f32) * 0.031).cos()).collect();
    let pooled: Vec<f32> = (0..c.pooled_dim() as usize).map(|i| ((i as f32) * 0.11).sin()).collect();
    let time_ids = vec![64.0, 64.0, 0.0, 0.0, 64.0, 64.0];
    (sample, enc, pooled, time_ids)
}

#[test]
fn tiny_forward_runs_and_is_finite() {
    let m = tiny(true);
    let (s, e, p, t) = inputs(&m);
    let out = m.run(&s, 601.0, &e, &p, &t);
    assert_eq!(out.len(), (m.config().out_channels * 16 * 16) as usize);

    // Stages FIRST, in record order: the first bad tap names the block that
    // broke, where a bad output only says "somewhere in 30 blocks".
    let names: Vec<String> = m.tap_names().iter().map(|s| s.to_string()).collect();
    assert!(names.len() > 20, "only {} taps recorded", names.len());
    for n in &names {
        let v = m.read_tap(n).expect("tap exists");
        assert!(v.iter().all(|x| x.is_finite()), "{n}: non-finite");
        // A graph that silently produced zeros (a mis-sized binding, a dispatch
        // with 0 threads) would still be "finite".
        assert!(v.iter().map(|x| x * x).sum::<f32>() > 0.0, "{n}: all zeros");
    }
    assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    let energy: f32 = out.iter().map(|v| v * v).sum();
    assert!(energy > 1e-6, "output is ~zero: energy {energy}");
}

/// The production graph (`taps = false`) enables `vae::blocks::Builder`'s
/// activation pool, which reuses buffers whose last read is already recorded.
/// A wrong `free` there silently clobbers a live activation, so the pooled
/// graph must be **bit-identical** to the pinned one — the same gate
/// `crates/vqgan` and `crates/restore` use.
#[test]
fn pooled_graph_is_bit_identical_to_the_tapped_one() {
    let tapped = tiny(true);
    let i = inputs(&tapped);
    if !deterministic_device(&tapped, &i) {
        return;
    }
    let (s, e, p, t) = i;
    let a = tapped.run(&s, 601.0, &e, &p, &t);
    drop(tapped);
    let pooled = tiny(false);
    let b = pooled.run(&s, 601.0, &e, &p, &t);
    // `f32::max` returns the non-NaN operand, so a NaN pair would make this
    // vacuously pass — compare the bits instead.
    assert!(a.iter().all(|v| v.is_finite()), "tapped graph is non-finite");
    let differing = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    assert_eq!(differing, 0, "{differing} of {} outputs differ between the pooled and tapped graphs", a.len());
}

/// Import must be a pure rename/fuse of a synthetic diffusers-shaped
/// checkpoint — exercised without a 10 GB download so the two-way coverage
/// logic itself is always under test.
#[test]
fn import_round_trips_a_synthetic_checkpoint() {
    let cfg = UNetConfig::tiny();
    let brain = init_weights(&cfg, 3);
    // Re-emit the manifest in DIFFUSERS naming, undoing the three fusions.
    let mut src: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)> = Default::default();
    for (name, (shape, data)) in &brain {
        if let Some(base) = name.strip_suffix(".attn1.qkv.weight") {
            let c = shape[1];
            for (i, leaf) in ["to_q", "to_k", "to_v"].into_iter().enumerate() {
                let piece = data[i * c * c..(i + 1) * c * c].to_vec();
                src.insert(format!("{base}.attn1.{leaf}.weight"), (vec![c, c], piece));
            }
        } else if let Some(base) = name.strip_suffix(".attn2.kv.weight") {
            let (two_c, x) = (shape[0], shape[1]);
            let c = two_c / 2;
            for (i, leaf) in ["to_k", "to_v"].into_iter().enumerate() {
                let piece = data[i * c * x..(i + 1) * c * x].to_vec();
                src.insert(format!("{base}.attn2.{leaf}.weight"), (vec![c, x], piece));
            }
        } else if name.contains(".ff.hidden.") || name.contains(".ff.gate.") {
            // Rebuilt below from the pair.
        } else if let Some(base) = name.strip_suffix(".ff.out.weight") {
            src.insert(format!("{base}.ff.net.2.weight"), (shape.clone(), data.clone()));
        } else if let Some(base) = name.strip_suffix(".ff.out.bias") {
            src.insert(format!("{base}.ff.net.2.bias"), (shape.clone(), data.clone()));
        } else if let Some(base) = name.strip_suffix(".to_out.weight") {
            src.insert(format!("{base}.to_out.0.weight"), (shape.clone(), data.clone()));
        } else if let Some(base) = name.strip_suffix(".to_out.bias") {
            src.insert(format!("{base}.to_out.0.bias"), (shape.clone(), data.clone()));
        } else {
            src.insert(name.clone(), (shape.clone(), data.clone()));
        }
    }
    for (name, (shape, data)) in &brain {
        if let Some(base) = name.strip_suffix(".ff.hidden.weight") {
            let gate = &brain[&format!("{base}.ff.gate.weight")];
            let mut v = data.clone();
            v.extend_from_slice(&gate.1);
            src.insert(format!("{base}.ff.net.0.proj.weight"), (vec![2 * shape[0], shape[1]], v));
        } else if let Some(base) = name.strip_suffix(".ff.hidden.bias") {
            let gate = &brain[&format!("{base}.ff.gate.bias")];
            let mut v = data.clone();
            v.extend_from_slice(&gate.1);
            src.insert(format!("{base}.ff.net.0.proj.bias"), (vec![2 * shape[0]], v));
        }
    }

    let got = sdxlunet::import::remap(src.clone(), &cfg).expect("remap");
    assert_eq!(got.len(), brain.len());
    for (name, (_, want)) in &brain {
        let (_, have) = got.get(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(have, want, "{name} differs after the round trip");
    }

    // Two-way: an extra source tensor must be an ERROR, not ignored.
    let mut extra = src.clone();
    extra.insert("something.unexpected".into(), (vec![1], vec![0.0]));
    assert!(sdxlunet::import::remap(extra, &cfg).is_err(), "an unused source tensor was accepted");
    // ... and so must a missing one.
    let mut missing = src;
    missing.remove("conv_in.weight");
    assert!(sdxlunet::import::remap(missing, &cfg).is_err(), "a missing source tensor was accepted");
}
