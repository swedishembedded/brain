// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Overfit gates for `cosyvoice::lmgrad` - the "training actually works" proof
//! that complements `gradcheck::check_cosyvoice_lm{,_block}` (those prove the
//! analytic gradients are *correct*; this proves they are *usable for
//! optimization*). Adam over every trainable tensor (embeddings, every decoder
//! layer, the final norm, the `llm_decoder` head) drives the masked
//! cross-entropy toward zero on a single fixed example, then on a small fixed
//! batch of examples - mirroring `wan::tests::overfit`'s shape for this
//! crate's own architecture.
//!
//! All synthetic token ids/inputs come from `data::rng::Lcg`, this repo's
//! sanctioned test-fixture PRNG - never `rand`.

use cosyvoice::lmgrad::{grad_views, grads, init_weights, params_mut, Example, LmDims};
use data::rng::Lcg;

fn synthetic_example(dims: &LmDims, rng: &mut Lcg, n_text: usize, n_speech: usize) -> Example {
    let text_ids = (0..n_text).map(|_| (rng.next_u32() as usize) % dims.text_vocab).collect();
    let speech_tokens = (0..n_speech).map(|_| (rng.next_u32() as usize) % dims.speech_vocab).collect();
    let special_task = if dims.special_vocab > 0 { dims.special_vocab - 1 } else { dims.speech_vocab - 1 };
    Example { text_ids, special_sos: 0, special_task, speech_tokens }
}

/// Flat Adam over every trainable tensor, `f32` - the same shape
/// `wan::tests::overfit` drives its own `params_mut`/`grad_views` pair with.
struct FlatAdam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u64,
}

impl FlatAdam {
    fn new(n: usize) -> FlatAdam {
        FlatAdam { m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }

    fn step(&mut self, w: &mut cosyvoice::lmgrad::LmWeights<f32>, g: &cosyvoice::lmgrad::LmGrads<f32>, lr: f32) {
        self.t += 1;
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        let gv: Vec<Vec<f32>> = grad_views(g).into_iter().map(|(_, x)| x.clone()).collect();
        let mut off = 0usize;
        for ((_, p), gt) in params_mut(w).into_iter().zip(&gv) {
            for (i, (pi, &gi)) in p.iter_mut().zip(gt.iter()).enumerate() {
                let k = off + i;
                self.m[k] = b1 * self.m[k] + (1.0 - b1) * gi;
                self.v[k] = b2 * self.v[k] + (1.0 - b2) * gi * gi;
                let mh = self.m[k] / bc1;
                let vh = self.v[k] / bc2;
                *pi -= lr * mh / (vh.sqrt() + eps);
            }
            off += gt.len();
        }
    }
}

fn nparams(w: &mut cosyvoice::lmgrad::LmWeights<f32>) -> usize {
    params_mut(w).iter().map(|(_, p)| p.len()).sum()
}

#[test]
fn single_example_overfits_to_near_zero_loss() {
    let dims = LmDims::tiny();
    let mut rng = Lcg::new(0xC05E_0001u64);
    let mut w = init_weights::<f32>(&dims, 1234);
    let ex = synthetic_example(&dims, &mut rng, 4, 5);

    let n = nparams(&mut w);
    let mut adam = FlatAdam::new(n);

    let mut first = 0.0f64;
    let mut last = 0.0f64;
    for step in 1..=400u32 {
        let (l, g) = grads(&dims, &w, &ex);
        if step == 1 {
            first = l;
        }
        last = l;
        adam.step(&mut w, &g, 8e-3);
        if step % 100 == 0 || step == 1 {
            println!("  single-example step {step:>4}  loss {l:.8}");
        }
    }
    println!("single-example overfit: loss {first:.8} -> {last:.8} over 400 Adam steps");
    assert!(last < first * 0.02, "the loss must collapse: {first} -> {last}");
    assert!(last < 0.05, "the loss must approach zero on one memorized example, got {last}");
}

#[test]
fn a_small_batch_overfits_to_near_zero_loss() {
    let dims = LmDims::tiny_cv3();
    let mut rng = Lcg::new(0x8A7C_0002u64);
    let mut w = init_weights::<f32>(&dims, 4321);
    let batch: Vec<Example> = (0..4).map(|i| synthetic_example(&dims, &mut rng, 3 + i, 4 + i)).collect();

    let n = nparams(&mut w);
    let mut adam = FlatAdam::new(n);

    let mut first = 0.0f64;
    let mut last = 0.0f64;
    for step in 1..=500u32 {
        // Average loss/grads over the batch - the natural "batch of examples"
        // extension of `grads`'s single-example unit (variable-length
        // sequences, so a shared-tensor sum rather than a stacked batch axis).
        let mut total_loss = 0.0f64;
        let mut acc_grads: Option<cosyvoice::lmgrad::LmGrads<f32>> = None;
        for ex in &batch {
            let (l, g) = grads(&dims, &w, ex);
            total_loss += l;
            acc_grads = Some(match acc_grads {
                None => g,
                Some(mut acc) => {
                    add_grads(&mut acc, &g);
                    acc
                }
            });
        }
        let mean_loss = total_loss / batch.len() as f64;
        let mut g = acc_grads.expect("non-empty batch");
        scale_grads(&mut g, 1.0 / batch.len() as f32);

        if step == 1 {
            first = mean_loss;
        }
        last = mean_loss;
        adam.step(&mut w, &g, 8e-3);
        if step % 100 == 0 || step == 1 {
            println!("  batch step {step:>4}  mean loss {mean_loss:.8}");
        }
    }
    println!("batch overfit: mean loss {first:.8} -> {last:.8} over 500 Adam steps");
    assert!(last < first * 0.05, "the batch loss must collapse: {first} -> {last}");
    assert!(last < 0.1, "the batch loss must approach zero on 4 memorized examples, got {last}");
}

fn add_grads(acc: &mut cosyvoice::lmgrad::LmGrads<f32>, g: &cosyvoice::lmgrad::LmGrads<f32>) {
    let av = params_mut(acc);
    let gv = grad_views(g);
    for ((_, a), (_, b)) in av.into_iter().zip(gv) {
        for (x, y) in a.iter_mut().zip(b) {
            *x += *y;
        }
    }
}

fn scale_grads(g: &mut cosyvoice::lmgrad::LmGrads<f32>, s: f32) {
    for (_, v) in params_mut(g) {
        for x in v.iter_mut() {
            *x *= s;
        }
    }
}
