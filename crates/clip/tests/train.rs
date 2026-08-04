// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The text tower's TRAINING build, checked for the two things a gradient check
//! structurally cannot see.
//!
//! 1. **The training build's forward is the inference build's forward, bit for
//!    bit.** `ClipText::new_train_on` allocates gradient buffers and records a
//!    reverse pass; it must not perturb the graph the 148-stage forward-parity
//!    ladder gates. `gradcheck` compares a model against *itself*, so a forward
//!    that drifted would still pass it — only this test notices.
//! 2. **The gradients actually descend.** A correct-looking backward whose sign
//!    or scale is wrong passes nothing here: one host gradient-descent step on
//!    every parameter must reduce the same proxy objective the gradient was
//!    taken of. (The step is applied on the host deliberately — `optim::Optim`
//!    would drag the AdamW kernel set into this tower's pipeline list for a
//!    smoke test, and AdamW's moment bias-correction would blur the sign.)
//!
//! No fixtures: the weights come from `clip::init`, so this test never skips.

use data::rng::Lcg;
use clip::config::{ClipTextConfig, TextAct};
use clip::model::{ClipText, TEXT_PIPELINES};

fn tiny(act: TextAct, projection: Option<u32>) -> ClipTextConfig {
    ClipTextConfig {
        hidden: 16,
        intermediate: 32,
        layers: 2,
        heads: 2,
        max_positions: 10,
        vocab: 23,
        act,
        eps: 1e-5,
        projection,
        bos_id: 21,
        eos_id: 22,
        pad_id: 22,
    }
}

#[test]
fn training_build_forward_is_identical_to_the_inference_build() {
    let cfg = tiny(TextAct::QuickGelu, None);
    let (b, t) = (2u32, 8u32);
    let init = clip::init::init_text_weights(&cfg, 11);
    let ids = clip::init::fixed_tokens(&cfg, b, t);

    let infer = ClipText::new_on(gpu_core::testgpu::dev(TEXT_PIPELINES), cfg.clone(), b, t, &init);
    infer.set_tokens(&ids);
    infer.forward();
    let (h0, p0) = (infer.read_hidden(), infer.read_pooled());
    assert!(!infer.is_trainable());
    drop(infer);

    let train = ClipText::new_train_on(gpu_core::testgpu::dev(TEXT_PIPELINES), cfg, b, t, &init);
    train.set_tokens(&ids);
    train.forward();
    assert!(train.is_trainable());
    let (h1, p1) = (train.read_hidden(), train.read_pooled());

    // BIT-identical, not merely close: same kernels, same order, same inputs.
    assert_eq!(h0.len(), h1.len(), "hidden length");
    for (i, (a, c)) in h0.iter().zip(&h1).enumerate() {
        assert_eq!(a.to_bits(), c.to_bits(), "hidden[{i}]: {a} vs {c}");
    }
    for (i, (a, c)) in p0.iter().zip(&p1).enumerate() {
        assert_eq!(a.to_bits(), c.to_bits(), "pooled[{i}]: {a} vs {c}");
    }
}

/// One host gradient-descent step must reduce `L = <r_h, hidden> + <r_o, out>`.
/// Run for both activations and both head shapes.
#[test]
fn a_gradient_step_reduces_the_proxy_objective() {
    for (label, act, projection) in [
        ("clip_l (quick_gelu)", TextAct::QuickGelu, None),
        ("bigg (gelu_erf + projection)", TextAct::GeluErf, Some(12)),
    ] {
        let cfg = tiny(act, projection);
        let (b, t) = (2u32, 8u32);
        let n = (b * t) as usize;
        let h = cfg.hidden as usize;
        let out_w = cfg.projection.unwrap_or(cfg.hidden) as usize;
        let init = clip::init::init_text_weights(&cfg, 23);
        let ids = clip::init::fixed_tokens(&cfg, b, t);
        let m = ClipText::new_train_on(
            gpu_core::testgpu::dev(TEXT_PIPELINES),
            cfg.clone(),
            b,
            t,
            &init,
        );
        m.set_tokens(&ids);
        let r_h = Lcg::new(0xC0FFEE | 1).vec(n * h);
        let r_o = Lcg::new(0xBEEF | 1).vec(b as usize * out_w);

        let loss = || -> f32 {
            m.forward();
            let seq: f32 = m.read_hidden().iter().zip(&r_h).map(|(y, r)| y * r).sum();
            let out: Vec<f32> = m.read_text_embeds().unwrap_or_else(|| m.read_pooled());
            seq + out.iter().zip(&r_o).map(|(y, r)| y * r).sum::<f32>()
        };

        let l0 = loss();
        m.zero_grads();
        m.backward(&r_h, &r_o);
        m.poll_wait();

        // Steepest descent with a step small enough that the linearisation holds.
        let names: Vec<String> = m.ps.params.iter().map(|(k, _)| k.clone()).collect();
        let gnorm: f32 = names
            .iter()
            .map(|k| m.read_grad(k).iter().map(|g| g * g).sum::<f32>())
            .sum::<f32>()
            .sqrt();
        assert!(gnorm > 1e-3, "[{label}] gradient is ~zero (norm {gnorm}) — nothing was written");
        let lr = 1e-2 / gnorm;
        for k in &names {
            let w = m.read_weight(k);
            let g = m.read_grad(k);
            let stepped: Vec<f32> = w.iter().zip(&g).map(|(w, g)| w - lr * g).collect();
            m.write_weight(k, &stepped);
        }
        let l1 = loss();
        eprintln!("[{label}] proxy loss {l0:.6} -> {l1:.6} (grad norm {gnorm:.4})");
        assert!(l1 < l0, "[{label}] a descent step did not decrease the loss: {l0} -> {l1}");
    }
}
