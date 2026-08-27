// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device full-model LoRA training for FLUX.2 Klein: (1) the GPU trainer's loss
//! and adapter gradients match the FD-gradchecked host reference
//! (`modelgrad` + `lora`), and (2) with the base frozen, training only `A,B`
//! through it drives the flow-matching loss down on the GPU.
//!
//! This is the end-to-end device training loop - the whole double/single block
//! stack plus the embedders and the final layer on the GPU (persistent
//! `BlockDev` engine), wrapped by the host conditioning front and the
//! rectified-flow velocity-MSE loss.
//!
//! The comparison is against `Pair::project`ed host gradients, because that is
//! the value the host trainer would Adam-step: the device produces `(dA, dB)`
//! straight out of the low-rank intermediates and never forms the dense `dW`
//! the host projects from. Cosine AND rel_l2 are both asserted (an epsilon
//! mutation scores cosine 1.000000 - see `tests/dev_grad.rs`).
//!
//! Needs a GPU: `BRAIN_DEV_GPU=1`.

use flux2::devtrain::DeviceTrainer;
use flux2::lora::{LoraAdapter, LoraCfg};
use flux2::modelgrad::{self, Cfg, ModelWeights};

const RANK: usize = 8;

/// A tiny klein-topology config whose sliced binding offsets all land on the
/// 256-byte (64-float) storage alignment: `txt_len·hidden`, `txt_len·mlp` and
/// `n_max·{rank, n_heads}` are all multiples of 64.
fn cfg() -> Cfg {
    Cfg {
        in_channels: 8,
        context_in_dim: 12,
        hidden: 64,
        n_heads: 4,
        depth_double: 2,
        depth_single: 2,
        mlp: 192,
        txt_len: 8,
        lh: 4,
        lw: 4,
        axes_dim: [4, 4, 4, 4],
        rope_theta: 2000.0,
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

fn skip() -> bool {
    if std::env::var("BRAIN_DEV_GPU").as_deref() != Ok("1") {
        brain_testutil::skip_unavailable("set BRAIN_DEV_GPU=1 (needs a GPU) for the FLUX.2 device training test");
        return true;
    }
    false
}

/// A batch and a base model at `cfg()`'s dims.
fn fixture(seed: u64) -> (Cfg, ModelWeights<f32>, modelgrad::Batch<f32>) {
    let c = cfg();
    let w = modelgrad::init_model::<f32>(&c, seed);
    let mut r = rng(seed ^ 0xbeef);
    let x0 = vof(c.n_img() * c.in_channels, &mut r, 1.0);
    let ctx = vof(c.txt_len * c.context_in_dim, &mut r, 1.0);
    let noise = vof(x0.len(), &mut r, 1.0);
    let b = modelgrad::make_flow_batch(&c, &x0, &ctx, 0.37, &noise);
    (c, w, b)
}

/// A fresh adapter with a NON-ZERO `B` - the shipped init sets `B = 0`, which
/// makes the whole up-projection a no-op and hides every bug in it.
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

#[test]
fn device_lora_grads_match_host() {
    if skip() {
        return;
    }
    let (c, base, batch) = fixture(0x5eed_0001);
    let ad = adapter(&c, 0x77);
    let scale = ad.scale();

    // ---- host reference (FD-gradchecked) on the adapter-applied weights ----
    let w_eff = ad.apply(&base);
    let (hloss, hg) = modelgrad::grads(&c, &w_eff, &batch);
    // The dense dW of every targeted linear, in LoraAdapter::pairs() order.
    let mut hdw: Vec<&Vec<f32>> = Vec::new();
    let mut hqk: Vec<(&Vec<f32>, &Vec<f32>)> = Vec::new();
    for b in &hg.dbl {
        for s in [&b.img, &b.txt] {
            hdw.extend([&s.wq, &s.wk, &s.wv, &s.wo, &s.w1, &s.w3, &s.w2]);
            hqk.push((&s.nq, &s.nk));
        }
    }
    for s in &hg.sgl {
        hdw.extend([&s.wq, &s.wk, &s.wv, &s.w1, &s.w3, &s.wo_a, &s.wo_b]);
        hqk.push((&s.nq, &s.nk));
    }

    // ---- device ----
    let tr = DeviceTrainer::new(c.clone(), RANK, &base);
    let (dloss, dg) = tr.grads(&ad, &batch);

    eprintln!("loss host={hloss:.9} device={dloss:.9}");
    assert!((hloss - dloss).abs() / hloss.abs().max(1e-12) < 1e-5, "loss mismatch: host {hloss} device {dloss}");
    assert_eq!(dg.lora.len(), hdw.len(), "pair count");
    assert_eq!(dg.qk.len(), hqk.len(), "qk-norm site count");

    let (mut wc, mut wr) = (1.0f64, 0.0f64);
    let (mut wcn, mut wrn) = (String::new(), String::new());
    let mut cmp = |name: String, dev: &[f32], host: &[f32]| {
        let cs = cosine(dev, host);
        let rl = rel_l2(dev, host);
        if cs < wc {
            wc = cs;
            wcn = name.clone();
        }
        if rl > wr {
            wr = rl;
            wrn = name;
        }
    };
    for (i, ((da, db), dw)) in dg.lora.iter().zip(&hdw).enumerate() {
        let (hda, hdb) = ad.pairs()[i].project(dw, scale);
        cmp(format!("pair{i}.dA"), da, &hda);
        cmp(format!("pair{i}.dB"), db, &hdb);
    }
    for (i, ((dnq, dnk), (hnq, hnk))) in dg.qk.iter().zip(&hqk).enumerate() {
        cmp(format!("qk{i}.nq"), dnq, hnq);
        cmp(format!("qk{i}.nk"), dnk, hnk);
    }
    eprintln!("FLUX.2 device model grads: {} tensors, worst cosine {wc:.9} ({wcn}), worst rel_l2 {wr:.3e} ({wrn})", 2 * (dg.lora.len() + dg.qk.len()));
    assert!(wc > 0.9999999, "worst cosine {wc:.9} on {wcn}");
    assert!(wr < 1e-5, "worst rel_l2 {wr:.3e} on {wrn}");
}

#[test]
fn a_fresh_adapter_is_a_device_no_op() {
    if skip() {
        return;
    }
    let (c, base, batch) = fixture(0x5eed_0002);
    // The SHIPPED init (B = 0), not the perturbed one: this is the invariant a
    // LoRA run depends on - step 0 must see exactly the base model.
    let ad = LoraAdapter::new(&c, LoraCfg::new(RANK));
    let (hloss, _) = modelgrad::grads(&c, &base, &batch);
    let tr = DeviceTrainer::new(c, RANK, &base);
    let (dloss, _) = tr.grads(&ad, &batch);
    eprintln!("fresh-adapter loss: host base {hloss:.9}, device adapted {dloss:.9}");
    assert!((hloss - dloss).abs() / hloss.abs().max(1e-12) < 1e-5, "a B=0 adapter must reproduce the base loss ({hloss} vs {dloss})");
}

#[test]
fn device_lora_only_training_drives_the_loss_down() {
    if skip() {
        return;
    }
    let (c, base, batch) = fixture(0x5eed_0003);
    let mut ad = LoraAdapter::new(&c, LoraCfg::new(RANK));
    let tr = DeviceTrainer::new(c, RANK, &base);
    let l0 = tr.step(&mut ad, &batch, 0.02);
    let mut last = l0;
    let mut lmin = l0;
    let mut early = l0;
    for i in 0..60 {
        last = tr.step(&mut ad, &batch, 0.02);
        lmin = lmin.min(last);
        if i == 9 {
            early = last;
        }
    }
    eprintln!("device LoRA-only overfit: loss {l0:.6} -> {last:.6} (min {lmin:.6}, {:.2}x lower)", l0 / lmin.max(1e-12));
    // A rank-8 adapter over the block linears only (embedders, modulation,
    // QK-norms and the head stay frozen) cannot fully fit an arbitrary target;
    // it drops to a capacity floor and holds. The mechanism is what is gated:
    // gradients reach A,B through the whole device stack and Adam moves them.
    assert!(early < 0.95 * l0, "loss did not drop early ({early:.6} after 10 steps vs {l0:.6})");
    assert!(lmin < 0.7 * l0, "device LoRA training barely moved the loss (min {lmin:.6} vs {l0:.6})");
    assert!(last < 0.85 * l0, "final loss did not hold near the floor ({last:.6} vs {l0:.6})");
}

#[test]
fn a_two_card_split_is_the_same_step_as_one_card() {
    if skip() {
        return;
    }
    if gpu_core::discrete_gpu_count() < 2 {
        brain_testutil::skip_unavailable("needs two discrete GPUs for the split-stack test");
        return;
    }
    // Splitting the stack inserts a host round trip at the cut and nothing
    // else: same blocks, same order, same dispatches. So the loss and every
    // adapter gradient must come back BIT-IDENTICAL, not merely close - a
    // tolerance here would hide a block landing on the wrong card.
    let (c, base, batch) = fixture(0x5eed_0004);
    let ad = adapter(&c, 0x99);
    let one = DeviceTrainer::new(c.clone(), RANK, &base);
    let (l1, g1) = one.grads(&ad, &batch);
    drop(one);
    let two = DeviceTrainer::new_multi(2, c, RANK, &base);
    assert_eq!(two.cards(), 2, "the split must actually use two cards");
    let per = two.weight_bytes_per_card();
    assert!(per.iter().all(|&b| b > 0), "every card must hold part of the stack, got {per:?}");
    let (l2, g2) = two.grads(&ad, &batch);
    eprintln!("split stack: loss {l1:.9} (1 card) vs {l2:.9} (2 cards), weights {per:?} bytes");
    assert_eq!(l1.to_bits(), l2.to_bits(), "loss differs across the split");
    for (i, ((a1, b1), (a2, b2))) in g1.lora.iter().zip(&g2.lora).enumerate() {
        assert_eq!(a1, a2, "pair {i}: dA differs across the split");
        assert_eq!(b1, b2, "pair {i}: dB differs across the split");
    }
    for (i, (s1, s2)) in g1.sites.iter().zip(&g2.sites).enumerate() {
        assert_eq!(s1.gate, s2.gate, "site {i}: gate grad differs across the split");
    }
    eprintln!("FLUX.2 two-card split reproduces the single-card step exactly.");
}
