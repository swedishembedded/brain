// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The fine-tune loop for forecasters whose trainable twin runs on the host.
//!
//! A host trainer (Chronos-2, FinCast) is a plain `name -> Vec<f32>` weight map
//! with a forward that records its intermediates and an analytic backward. The
//! loop around it is the same whichever model it is: draw a batch, take the mean
//! of the per-example gradients, clip, step AdamW, and judge the result on held
//! out windows with [`promoted`]. It lives here once, as [`finetune_host`]
//! over the [`HostTrainable`] a model implements, rather than as a copy per
//! model. What stays model-specific is the example (how a window becomes
//! tokens and a target) and the loss.
//!
//! [`HostAdamW`] is `torch.optim.AdamW` (decoupled weight decay, beta
//! 0.9/0.999, eps 1e-8) with the global-norm clip of
//! `torch.nn.utils.clip_grad_norm_`, over named tensors. Tensors are visited in
//! sorted-name order so the global norm, a float sum, is the same on every run.
//!
//! Swedish Embedded AB implements gated fine-tuning for forecasting models for
//! its clients: a model update that ships only when held-out data says it
//! should. If your team needs expertise in time-series fine-tuning and
//! promotion policy you can procure our services by sending an email to
//! info@swedishembedded.com.

use crate::finetune::{promoted, FinetuneOpts, FinetuneOutcome, FinetuneReport};
use std::collections::HashMap;

/// Named tensors: parameters or gradients, keyed by the checkpoint's tensor names.
pub type Tensors = HashMap<String, Vec<f32>>;

/// AdamW over named tensors, with an optional global-norm clip.
#[derive(Default)]
pub struct HostAdamW {
    m: Tensors,
    v: Tensors,
    t: u32,
}

impl HostAdamW {
    pub fn new() -> HostAdamW {
        HostAdamW::default()
    }

    /// Steps taken so far.
    pub fn steps(&self) -> u32 {
        self.t
    }

    /// One update of `params` from `grads`. A tensor with no gradient entry is
    /// left alone. With `clip = Some(c)` the gradients are first scaled so their
    /// global L2 norm is at most `c`.
    pub fn step(&mut self, params: &mut Tensors, grads: &Tensors, lr: f32, wd: f32, clip: Option<f32>) {
        const BETA1: f32 = 0.9;
        const BETA2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        self.t += 1;
        let mut names: Vec<&String> = grads.keys().collect();
        names.sort();

        let scale = match clip {
            Some(c) => {
                let sq: f64 = names.iter().flat_map(|n| grads[*n].iter()).map(|&g| g as f64 * g as f64).sum();
                let norm = sq.sqrt() as f32;
                if norm > c {
                    c / (norm + 1e-6)
                } else {
                    1.0
                }
            }
            None => 1.0,
        };

        let bc1 = 1.0 - BETA1.powi(self.t as i32);
        let bc2 = 1.0 - BETA2.powi(self.t as i32);
        for name in names {
            let Some(p) = params.get_mut(name) else { continue };
            let g = &grads[name];
            let m = self.m.entry(name.clone()).or_insert_with(|| vec![0.0; p.len()]);
            let v = self.v.entry(name.clone()).or_insert_with(|| vec![0.0; p.len()]);
            for i in 0..p.len() {
                let gi = g[i] * scale;
                m[i] = BETA1 * m[i] + (1.0 - BETA1) * gi;
                v[i] = BETA2 * v[i] + (1.0 - BETA2) * gi * gi;
                let update = (m[i] / bc1) / ((v[i] / bc2).sqrt() + EPS);
                p[i] = p[i] * (1.0 - lr * wd) - lr * update;
            }
        }
    }
}

/// A model the host loop can fine-tune.
pub trait HostTrainable {
    /// One training or held-out window, already turned into what the model
    /// consumes.
    type Example;

    /// The weights, in the checkpoint's own tensor names.
    fn params(&self) -> &Tensors;
    fn params_mut(&mut self) -> &mut Tensors;

    /// The loss of one example under the current weights.
    fn loss(&self, ex: &Self::Example) -> f32;

    /// The loss of one example, with its gradient ADDED into `grads` (which
    /// holds an entry per parameter).
    fn loss_and_grad(&self, ex: &Self::Example, grads: &mut Tensors) -> f32;
}

/// Mean loss over `examples`; `NaN` when there are none or one is not finite.
pub fn mean_loss<M: HostTrainable>(model: &M, examples: &[M::Example]) -> f32 {
    if examples.is_empty() {
        return f32::NAN;
    }
    let total: f64 = examples.iter().map(|e| model.loss(e) as f64).sum();
    (total / examples.len() as f64) as f32
}

/// Fine-tune every weight of `model` on `train` and gate the result on `val`.
///
/// Each step averages the gradients of `opts.batch` examples (a trailing short
/// batch is a smaller mean, not dropped). The returned weights are the tuned
/// ones whether or not the gate promoted them; `report.promoted` alone decides
/// whether the caller writes them. `label` prefixes the progress lines.
pub fn finetune_host<M: HostTrainable, L, S>(model: &mut M, train: &[M::Example], val: &[M::Example], opts: &FinetuneOpts<L, S>, label: &str) -> FinetuneOutcome {
    let base_val = mean_loss(model, val);
    let batch = (opts.batch.max(1) as usize).min(train.len().max(1));
    let mut adam = HostAdamW::new();
    for epoch in 0..opts.epochs {
        for group in train.chunks(batch) {
            let mut grads: Tensors = model.params().iter().map(|(k, v)| (k.clone(), vec![0.0; v.len()])).collect();
            for ex in group {
                model.loss_and_grad(ex, &mut grads);
            }
            let inv = 1.0 / group.len() as f32;
            for g in grads.values_mut().flat_map(|g| g.iter_mut()) {
                *g *= inv;
            }
            adam.step(model.params_mut(), &grads, opts.lr, opts.wd, Some(opts.clip));
        }
        if opts.progress {
            eprintln!("{label} finetune: epoch {}/{} ({} steps)", epoch + 1, opts.epochs, adam.steps());
        }
    }
    let ft_val = if adam.steps() == 0 { f32::NAN } else { mean_loss(model, val) };
    let report = FinetuneReport { promoted: promoted(base_val, ft_val), base_val, ft_val, steps: adam.steps(), lora_seed: None };
    (report, Some(model.params().clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(name: &str, v: &[f32]) -> Tensors {
        HashMap::from([(name.to_string(), v.to_vec())])
    }

    #[test]
    fn the_first_adamw_step_moves_each_weight_by_lr_against_its_gradient_sign() {
        // m_hat = g, v_hat = g^2, so the update is lr * g / (|g| + eps) ~ lr * sign(g).
        let mut p = one("w", &[1.0, -1.0, 5.0]);
        HostAdamW::new().step(&mut p, &one("w", &[0.3, -7.0, 0.0]), 0.1, 0.0, None);
        let w = &p["w"];
        assert!((w[0] - 0.9).abs() < 1e-5 && (w[1] + 0.9).abs() < 1e-5, "{w:?}");
        assert_eq!(w[2], 5.0, "a zero gradient moves nothing");
    }

    #[test]
    fn weight_decay_is_decoupled_from_the_gradient() {
        let mut p = one("w", &[2.0]);
        HostAdamW::new().step(&mut p, &one("w", &[0.0]), 0.1, 0.5, None);
        assert!((p["w"][0] - 2.0 * (1.0 - 0.1 * 0.5)).abs() < 1e-6);
    }

    #[test]
    fn the_global_norm_clip_scales_every_tensor_alike() {
        // g = (3, 4) over two tensors has norm 5; clipped to 1 it is (0.6, 0.8).
        // The first Adam step is sign-like whatever the scale, so compare the
        // second moment instead: feed the same step twice and look at v.
        let mut a = HostAdamW::new();
        let mut p = HashMap::from([("a".to_string(), vec![0.0]), ("b".to_string(), vec![0.0])]);
        let g = HashMap::from([("a".to_string(), vec![3.0]), ("b".to_string(), vec![4.0])]);
        a.step(&mut p, &g, 0.0, 0.0, Some(1.0));
        assert!((a.v["a"][0] - 0.001 * 0.36).abs() < 1e-8 && (a.v["b"][0] - 0.001 * 0.64).abs() < 1e-8);
        let mut b = HostAdamW::new();
        b.step(&mut p, &g, 0.0, 0.0, Some(10.0));
        assert!((b.v["a"][0] - 0.001 * 9.0).abs() < 1e-6, "below the clip the gradient is untouched");
    }

    /// `y = w * x`, squared error: the smallest model that exercises the loop.
    struct Line(Tensors);
    impl HostTrainable for Line {
        type Example = (f32, f32);
        fn params(&self) -> &Tensors {
            &self.0
        }
        fn params_mut(&mut self) -> &mut Tensors {
            &mut self.0
        }
        fn loss(&self, &(x, y): &(f32, f32)) -> f32 {
            (self.0["w"][0] * x - y).powi(2)
        }
        fn loss_and_grad(&self, ex: &(f32, f32), grads: &mut Tensors) -> f32 {
            grads.get_mut("w").unwrap()[0] += 2.0 * (self.0["w"][0] * ex.0 - ex.1) * ex.0;
            self.loss(ex)
        }
    }

    #[test]
    fn the_loop_lowers_held_out_loss_and_the_gate_promotes_it() {
        let mut m = Line(one("w", &[0.0]));
        let train: Vec<(f32, f32)> = (1..=8).map(|i| (i as f32 * 0.5, i as f32 * 1.0)).collect(); // y = 2x
        let val = vec![(3.0, 6.0), (0.5, 1.0)];
        let opts: FinetuneOpts<()> = FinetuneOpts { epochs: 60, lr: 0.05, wd: 0.0, clip: 100.0, batch: 3, ..Default::default() };
        let (rep, w) = finetune_host(&mut m, &train, &val, &opts, "line");
        assert!(rep.promoted && rep.ft_val < 0.01 * rep.base_val, "{rep:?}");
        assert_eq!(rep.steps, 60 * 3, "8 examples at batch 3 are 3 steps an epoch, the short batch included");
        assert!((w.unwrap()["w"][0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn a_run_with_no_steps_or_no_held_out_windows_is_never_promoted() {
        let opts: FinetuneOpts<()> = FinetuneOpts { epochs: 0, ..Default::default() };
        let (rep, _) = finetune_host(&mut Line(one("w", &[0.0])), &[(1.0, 2.0)], &[(1.0, 2.0)], &opts, "line");
        assert!(!rep.promoted && rep.ft_val.is_nan());
        let opts: FinetuneOpts<()> = FinetuneOpts { epochs: 3, lr: 0.1, ..Default::default() };
        let (rep, _) = finetune_host(&mut Line(one("w", &[0.0])), &[(1.0, 2.0)], &[], &opts, "line");
        assert!(!rep.promoted, "{rep:?}");
    }
}
