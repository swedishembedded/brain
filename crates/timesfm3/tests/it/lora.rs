// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The LoRA contract for `timesfm3::train::Timesfm3Train`.
//!
//! Three properties, none of which a gradient check can see (a gradcheck
//! confirms the adapter's gradients are right, not that the adapter is
//! correctly placed, correctly frozen around, or correctly foldable):
//!
//! 1. **no-op at init** - `B` is zero, so a freshly built LoRA trainer must
//!    reproduce the base model's forward BITWISE, not approximately;
//! 2. **only the adapters move** - every checkpoint tensor is frozen, and
//!    must come out of a real training loop bit-for-bit unchanged;
//! 3. **fold == apply** - folding `(alpha/r)·B·A` into the base weights and
//!    running the ordinary, adapter-free graph must reproduce the live
//!    adapted forward.
//!
//! Swedish Embedded AB implements parameter-efficient fine-tuning for teams
//! adapting large pretrained models on constrained hardware. If your team
//! needs expertise in LoRA placement, adapter folding or training-time
//! freezing, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::collections::HashMap;

use timesfm3::config::Timesfm3Config;
use timesfm3::train::{init_weights, fixed_input, LoraCfg, Timesfm3Train, LORA_TARGETS, TRAIN_PIPELINES};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

const B: usize = 1;
const V: usize = 2;
const N: usize = 3;

struct Fixture {
    cfg: Timesfm3Config,
    weights: HashMap<String, Vec<f32>>,
    x: Vec<f32>,
    mask: Vec<bool>,
    /// The proxy objective's `dL/dlogits`, fixed so two runs are comparable.
    c: Vec<f32>,
}

fn fixture() -> Fixture {
    let cfg = Timesfm3Config::tiny();
    let weights = init_weights(&cfg, 7);
    let x = fixed_input(&cfg, B, V, N, 7);
    let mask = vec![false; B * V * N];
    // A fixed direction on the raw logits: `L = <c, logits>`, so
    // `dL/dlogits == c` exactly and `backward(c)` is the true gradient of a
    // real scalar. The descent test below needs `L` itself, which this makes
    // a plain dot product.
    let c: Vec<f32> = (0..B * V * N * cfg.head_out_dim()).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    Fixture { cfg, weights, x, mask, c }
}

impl Fixture {
    fn base(&self) -> Timesfm3Train {
        Timesfm3Train::new_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), self.cfg.clone(), &self.x, &self.mask, B, V, N, &self.weights)
    }
    fn lora(&self, rank: usize, alpha: f32) -> Timesfm3Train {
        Timesfm3Train::new_lora_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), self.cfg.clone(), LoraCfg::attn(rank, alpha).with_output_head(), &self.x, &self.mask, B, V, N, &self.weights)
    }
    fn logits(&self, m: &Timesfm3Train) -> Vec<f32> {
        m.forward();
        m.poll_wait();
        m.read_logits()
    }
    fn loss(&self, m: &Timesfm3Train) -> f32 {
        self.logits(m).iter().zip(&self.c).map(|(&y, &c)| y as f64 * c as f64).sum::<f64>() as f32
    }
}

/// `B` is initialised to exactly zero, so `W + (alpha/r)·B·A == W` entry for
/// entry and the adapted forward must equal the base forward BITWISE. An
/// "approximately equal" adapter at init is an adapter that has already
/// changed the model before a single step, and every later comparison against
/// the base checkpoint is then measuring the wrong thing.
#[test]
fn a_freshly_built_adapter_is_an_exact_no_op() {
    if skip() {
        return;
    }
    let f = fixture();
    let base = f.logits(&f.base());
    let lora = f.logits(&f.lora(2, 4.0));
    assert_eq!(base.len(), lora.len());
    assert_eq!(base, lora, "a fresh LoRA adapter (B = 0) must not move a single output bit");
}

/// Only the adapters are trainable, and they are the WHOLE trainable set.
/// Ten targets per layer plus the quantile head, each a whole-matrix
/// placement: this architecture fuses no QKV, so there is no packed region
/// any of them could be a slice of.
#[test]
fn only_adapters_train_and_every_target_gets_one() {
    if skip() {
        return;
    }
    let f = fixture();
    let m = f.lora(2, 4.0);
    let names = m.param_names();
    assert!(
        names.iter().all(|n| n.ends_with(".lora_a") || n.ends_with(".lora_b")),
        "only adapters may be trainable under LoRA, got {:?}",
        names.iter().filter(|n| !n.ends_with(".lora_a") && !n.ends_with(".lora_b")).collect::<Vec<_>>()
    );
    // 10 targets per layer + the output head, A and B each.
    let want = 2 * (LORA_TARGETS.len() * f.cfg.num_layers + 1);
    assert_eq!(names.len(), want, "every target gets exactly one A and one B");
    // Norm gains and the PerDimScale halves are never adapted.
    for n in &names {
        assert!(!n.contains("_ln.weight"), "{n}: a norm gain is not a LoRA target");
        assert!(!n.contains("per_dim_scale"), "{n}: per_dim_scale is not a LoRA target");
    }
}

/// A real optimisation step has to move the loss DOWN, and has to do it while
/// leaving every checkpoint tensor bit-for-bit where it was.
///
/// Plain gradient descent, applied host-side through `read_grad`/
/// `write_weight`, rather than an AdamW step: the property under test is that
/// the adapter's gradients point downhill and that the freeze holds, and SGD
/// is the update for which "downhill" is a statement about those gradients
/// alone rather than about a moment estimator's warm-up.
#[test]
fn adapter_descent_lowers_the_loss_and_never_moves_the_base() {
    if skip() {
        return;
    }
    let f = fixture();
    let m = f.lora(2, 4.0);

    let base_before: Vec<(String, Vec<f32>)> =
        f.cfg.param_list().into_iter().map(|(n, _)| (n.clone(), m.read_weight(&n))).collect();
    let l0 = f.loss(&m);

    let lr = 2e-2f32;
    for _ in 0..12 {
        m.forward();
        m.poll_wait();
        m.zero_grads();
        m.backward(&f.c);
        for name in m.param_names() {
            let (w, g) = (m.read_weight(&name), m.read_grad(&name));
            let stepped: Vec<f32> = w.iter().zip(&g).map(|(&wi, &gi)| wi - lr * gi).collect();
            m.write_weight(&name, &stepped);
        }
    }
    let l1 = f.loss(&m);
    assert!(l1 < l0, "adapter descent did not lower the loss: {l0} -> {l1}");

    for (name, before) in &base_before {
        assert_eq!(&m.read_weight(name), before, "{name}: a frozen base tensor moved during LoRA training");
    }
    // Vacuity guard: if the adapters had not moved either, the freeze above
    // would be trivially satisfied and the descent would be measuring noise.
    let moved = m.param_names().iter().filter(|n| n.ends_with(".lora_b")).any(|n| m.read_weight(n).iter().any(|v| v.abs() > 1e-8));
    assert!(moved, "no B adapter left its zero init, so nothing was actually trained");
}

/// Folding `(alpha/r)·B·A` into the base weights and running the ordinary,
/// adapter-free graph must reproduce the live adapted forward.
///
/// Agreement here is numerical, not bitwise, and that is a property of the
/// device-side adapter family rather than a loosened gate: the live path
/// computes `x·Wᵀ` and then adds `scale·((x·Aᵀ)·Bᵀ)` to the RESULT, while the
/// folded path contracts `x` against `W + scale·B·A` in one accumulation.
/// Those are the same real number and different floating-point summations.
/// (The host-side adapter family in `model::lora` does assert bit-equality,
/// because there "apply" is itself a fold into a cloned weight - the same
/// operation, not a different association of it.) `qwen3`'s
/// `folding_the_adapter_into_the_base_reproduces_the_live_lora_forward` is
/// the precedent this matches.
#[test]
fn folding_the_adapter_reproduces_the_live_adapted_forward() {
    if skip() {
        return;
    }
    let f = fixture();
    let m = f.lora(2, 4.0);

    // Move the adapters off their no-op init first, or the fold is folding
    // zero and this test passes against a broken `fold_delta`.
    let mut state = 0xabcd_ef01_2345_6789u64;
    let mut next = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (((state >> 33) as u32 % 2000) as f32 / 1000.0) - 1.0
    };
    for name in m.param_names() {
        let w = m.read_weight(&name);
        let perturbed: Vec<f32> = w.iter().map(|&wi| wi + 0.05 * next()).collect();
        m.write_weight(&name, &perturbed);
    }

    let live = f.logits(&m);
    let folded_weights = m.to_reference_weights();
    assert_eq!(folded_weights.len(), f.cfg.param_list().len(), "a folded checkpoint carries the reference tensors and nothing else");

    let folded_model = Timesfm3Train::new_on(gpu_core::testgpu::dev(TRAIN_PIPELINES), f.cfg.clone(), &f.x, &f.mask, B, V, N, &folded_weights);
    let folded = f.logits(&folded_model);

    // Vacuity guard: the adapters must actually be doing something, or the
    // agreement below is the agreement of two identical base forwards.
    let base = f.logits(&f.base());
    let adapter_effect: f32 = live.iter().zip(&base).map(|(a, b)| (a - b).abs()).sum::<f32>() / live.len() as f32;
    assert!(adapter_effect > 1e-3, "the adapter barely changed the forward ({adapter_effect:.3e}); fold agreement would be vacuous");

    let mean_abs_diff: f32 = live.iter().zip(&folded).map(|(a, b)| (a - b).abs()).sum::<f32>() / live.len() as f32;
    assert!(
        mean_abs_diff < 1e-5,
        "folded-base forward does not match the live adapted forward: mean abs diff {mean_abs_diff:.3e} against an adapter effect of {adapter_effect:.3e}"
    );
}
