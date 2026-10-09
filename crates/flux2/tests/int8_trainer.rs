// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device LoRA trainer over an **int8 frozen base**
//! (`devgrad::BasePrecision::Int8`) computes the gradients the
//! FD-gradchecked host reference computes over the SAME dequantised weights,
//! to within what quantising the activations costs.
//!
//! Swedish Embedded AB implements quantised-base fine-tuning on GPUs for its
//! clients. If your team needs expertise in training adapters against the
//! int8 model you actually serve, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! What is compared, and what is not. The int8 base is quantised on upload
//! with `model::int8::quantize_weight` (the quantiser generation serves
//! through), so its weights ARE the dequantised ones the host differentiates -
//! exactly, not approximately. What remains different is the forward's
//! per-row activation quantisation (the served DP4A arithmetic) and the
//! straight-through treatment of it in the backward. That is a genuine
//! numeric difference, so this gate states a tolerance rather than an
//! identity, and it holds the fp32-base trainer over the same dequantised
//! weights to the host's own bound in the same run - so a regression in the
//! common machinery cannot hide inside the int8 tolerance.
//!
//! Needs a device offering the native int8-weight input gradient (CUDA):
//! the card named on the command line: `cargo test -p brain-flux2 --release --test int8_trainer -- --device gpu1 --backend cuda`.

use flux2::devgrad::BasePrecision;
use flux2::devtrain::{DeviceTrainer, TrainerSpec};
use flux2::lora::{LoraAdapter, LoraCfg};
use flux2::modelgrad::{self, Cfg, ModelWeights};

const RANK: usize = 8;

/// The tolerance the int8 base is held to, per adapter gradient tensor.
///
/// A per-row int8 activation is accurate to half a quantisation step, about
/// 0.4% of the row's largest magnitude, and a block's gradient flows through
/// a chain of such linears. Measured on this fixture (both layouts): worst
/// cosine 0.99994, worst rel_l2 1.1e-2 - the same order as the weight term a
/// real klein double block measured (`tests/int8_base_grads.rs`). The bounds
/// leave about 2.7x on the rel_l2, and sit two orders of magnitude below
/// what a wrong scale group, a dropped stream or a mis-sliced row range
/// produces (cosine near zero, rel_l2 near one).
const INT8_MIN_COSINE: f64 = 0.9995;
const INT8_MAX_REL_L2: f64 = 0.03;

/// A tiny klein-topology config every int8 path accepts: widths are whole
/// 32-element scale groups and the text rows - the row base of the image
/// stream - a whole number of 64-row activation-quantisation blocks.
fn cfg_with(refs: Vec<(usize, usize)>) -> Cfg {
    Cfg {
        in_channels: 8,
        context_in_dim: 12,
        hidden: 64,
        n_heads: 4,
        depth_double: 2,
        depth_single: 2,
        mlp: 192,
        txt_len: 64,
        lh: 4,
        lw: 4,
        axes_dim: [4, 4, 4, 4],
        rope_theta: 2000.0,
        refs,
    }
}

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    }
}

fn vof(n: usize, r: &mut impl FnMut() -> f32, s: f32) -> Vec<f32> {
    (0..n).map(|_| r() * s).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-300)
}

fn rel_l2(dev: &[f32], host: &[f32]) -> f64 {
    let nh: f64 = host.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let diff: f64 = dev.iter().zip(host).map(|(&a, &b)| ((a - b) as f64) * ((a - b) as f64)).sum::<f64>().sqrt();
    diff / nh.max(1e-12)
}

/// `w` through the int8 grid and back - what the int8 engine holds.
fn round_trip(w: &mut Vec<f32>, n: usize, k: usize) {
    let (q, s) = model::int8::quantize_weight(w, n, k);
    *w = model::int8::dequantize_weight(&q, &s, n, k);
}

/// `base` with every linear the int8 engine quantises replaced by its
/// dequantised value; the double blocks' `mlp.2` stays fp32, as served.
fn dequantised(c: &Cfg, base: &ModelWeights<f32>) -> ModelWeights<f32> {
    let (d, mlp) = (c.hidden, c.mlp);
    let mut w = base.clone();
    for b in &mut w.dbl {
        for s in [&mut b.img, &mut b.txt] {
            for t in [&mut s.wq, &mut s.wk, &mut s.wv, &mut s.wo] {
                round_trip(t, d, d);
            }
            round_trip(&mut s.w1, mlp, d);
            round_trip(&mut s.w3, mlp, d);
        }
    }
    for s in &mut w.sgl {
        for t in [&mut s.wq, &mut s.wk, &mut s.wv, &mut s.wo_a] {
            round_trip(t, d, d);
        }
        round_trip(&mut s.w1, mlp, d);
        round_trip(&mut s.w3, mlp, d);
        round_trip(&mut s.wo_b, d, mlp);
    }
    w
}

/// A fresh adapter with a NON-ZERO `B` - the shipped init sets `B = 0`, which
/// makes the up-projection a no-op and hides every bug in it.
fn adapter(c: &Cfg, seed: u64) -> LoraAdapter {
    let mut ad = LoraAdapter::new(c, LoraCfg { seed, ..LoraCfg::new(RANK) });
    let mut r = rng(seed ^ 0x1357);
    for p in ad.pairs_mut() {
        for v in p.b.iter_mut() {
            *v = r() * 0.2;
        }
    }
    ad
}

/// Worst `(cosine, rel_l2)` of the device adapter gradients against the host's.
fn worst(dg: &flux2::devtrain::StepGrads, hg: &modelgrad::ModelGrads<f32>, ad: &LoraAdapter) -> (f64, f64) {
    let mut hdw: Vec<&Vec<f32>> = Vec::new();
    for b in &hg.dbl {
        for s in [&b.img, &b.txt] {
            hdw.extend([&s.wq, &s.wk, &s.wv, &s.wo, &s.w1, &s.w3, &s.w2]);
        }
    }
    for s in &hg.sgl {
        hdw.extend([&s.wq, &s.wk, &s.wv, &s.w1, &s.w3, &s.wo_a, &s.wo_b]);
    }
    assert_eq!(dg.lora.len(), hdw.len(), "pair count");
    let (mut wc, mut wr) = (1.0f64, 0.0f64);
    for (i, ((da, db), dw)) in dg.lora.iter().zip(&hdw).enumerate() {
        let (hda, hdb) = ad.pairs()[i].project(dw, ad.scale());
        for (dev, host) in [(da, &hda), (db, &hdb)] {
            wc = wc.min(cosine(dev, host));
            wr = wr.max(rel_l2(dev, host));
        }
    }
    (wc, wr)
}

/// One card, placed by the test run's own device selection.
fn spec(base: BasePrecision) -> TrainerSpec {
    TrainerSpec { card: None, cards: 1, base }
}

fn skip() -> bool {
    let probe = gpu_core::Gpu::new_gpu(flux2::devgrad::KERNELS);
    if !probe.has_fused(gpu_core::Fused::I8wDx) {
        brain_testutil::skip_unavailable("this device does not offer the native int8-weight input gradient");
        return true;
    }
    false
}

fn an_int8_base_trains_the_gradients_of_its_dequantised_weights() {
    if skip() {
        return;
    }
    for refs in [Vec::new(), vec![(4usize, 4usize)]] {
        let paired = !refs.is_empty();
        let c = cfg_with(refs);
        let base = modelgrad::init_model::<f32>(&c, 0x18b_a5e);
        let deq = dequantised(&c, &base);
        let mut r = rng(0x0b5e_55ed);
        let x0 = vof(c.n_gen() * c.in_channels, &mut r, 1.0);
        let rf = vof(c.n_ref() * c.in_channels, &mut r, 1.0);
        let ctx = vof(c.txt_len * c.context_in_dim, &mut r, 1.0);
        let noise = vof(x0.len(), &mut r, 1.0);
        let batch = modelgrad::make_flow_batch_paired(&c, &x0, &rf, &ctx, 0.37, &noise);
        let ad = adapter(&c, 0x77);

        let (hloss, hg) = modelgrad::grads(&c, &ad.apply(&deq), &batch);

        // The fp32 trainer over the dequantised weights: the common machinery,
        // held to the fp32 gate's own bound.
        let f32_tr = DeviceTrainer::build(&spec(BasePrecision::F32), c.clone(), RANK, &deq).expect("fp32 trainer");
        let (floss, fg) = f32_tr.grads(&ad, &batch);
        drop(f32_tr);
        let (fc, fr) = worst(&fg, &hg, &ad);

        // The int8 trainer, handed the ORIGINAL fp32 weights: it quantises
        // them itself on upload, which must land exactly on `deq`.
        let i8_tr = DeviceTrainer::build(&spec(BasePrecision::Int8), c.clone(), RANK, &base).expect("int8 trainer");
        let (iloss, ig) = i8_tr.grads(&ad, &batch);
        let (ic, ir) = worst(&ig, &hg, &ad);

        eprintln!(
            "int8 trainer ({}): loss host {hloss:.7} fp32 {floss:.7} int8 {iloss:.7}; fp32 base worst cosine {fc:.9} rel_l2 {fr:.3e}; int8 base worst cosine {ic:.6} rel_l2 {ir:.3e}",
            if paired { "paired" } else { "caption-only" }
        );
        assert!((hloss - floss).abs() / hloss.abs().max(1e-12) < 1e-5, "fp32 trainer loss {floss} vs host {hloss}");
        assert!(fc > 0.9999999 && fr < 1e-5, "fp32 trainer over the dequantised base: cosine {fc} rel_l2 {fr}");
        assert!((hloss - iloss).abs() / hloss.abs().max(1e-12) < 1e-2, "int8 trainer loss {iloss} vs host {hloss}");
        assert!(ic > INT8_MIN_COSINE, "int8 base worst cosine {ic:.6} <= {INT8_MIN_COSINE}");
        assert!(ir < INT8_MAX_REL_L2, "int8 base worst rel_l2 {ir:.3e} >= {INT8_MAX_REL_L2}");
    }
}

gpu_core::card_tests!(
    an_int8_base_trains_the_gradients_of_its_dequantised_weights,
);
