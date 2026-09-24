// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `pixels` arm's own trainable path: a small host-side
//! `Linear -> LayerNorm` turning one grid cell's color identity into one row
//! the encoder never produced, and a plain Adam step to train it - the ONLY
//! new trainable weights this sample adds. Everything downstream of the
//! projected row (the head's cross-attention, its optimizer) is
//! `crates/decide`'s existing kept-features path, untouched.
//!
//! Two fixes were required before this converged, both found by measuring
//! rather than guessing, and both recorded here because either one on its
//! own is exactly the kind of thing that looks fine and silently is not:
//!
//! 1. **The LayerNorm.** Every row `Decide`'s head ever reads (the embedding
//!    table's output, every transformer block's output) is LayerNorm'd
//!    immediately before the next consumer sees it
//!    (`crates/decide/src/model.rs`'s own doc: "POST-LayerNorm... each block
//!    normalizes AFTER its residual add"). A raw `Linear` output has no such
//!    guarantee: measured against a real encoded row from the same
//!    checkpoint, a Xavier-initialized projector's rows read RMS ~0.17-0.27
//!    against the real rows' ~0.36-0.63 - noticeably quieter, and attention's
//!    dot-product score is exactly the computation the literature says is
//!    most sensitive to that (keys with smaller magnitude get smaller, less
//!    distinctive attention logits, so the query has less to select on).
//! 2. **Discrete color identity, not continuous RGB.** A raw-RGB version
//!    trained flat at chance regardless of the LayerNorm fix, for a
//!    structural reason rather than a scale one: unlike a real vision tower
//!    (CLIP's is itself CONTRASTIVELY PRETRAINED to align with text, which is
//!    exactly why a LLaVA-style linear projector on top of it can bootstrap
//!    from a random start), continuous RGB here carries no prior relationship
//!    AT ALL to how the frozen encoder represents the WORD "red" - the head
//!    would have to learn that whole cross-modal alignment from scratch, from
//!    a 16-way softmax's weak supervision. A one-hot color over the SAME
//!    six-color vocabulary `instruction()` already draws from removes that
//!    confound: the projector's job becomes "which of six known symbols is
//!    present here", a closed discrete match rather than an unconstrained
//!    continuous one.
//!
//! Both are verified independently of the sample's own held-out run:
//! `layernorm_matches_a_numerical_gradient` gradient-checks the LayerNorm
//! backward, `layernorm_output_has_unit_scale_at_init` checks the scale claim
//! directly. What neither fix touches is TRAINING BUDGET: `text` converges
//! inside 3000 steps; `pixels` measured flat through 3000, and even 6000,
//! with a clear downward trend only emerging around step 4000-4400 and
//! continuing to a comparable held-out accuracy by 15000 - see
//! `default_train_n` in `main.rs` for why the two arms default differently.
//! That gap is the cost of bootstrapping head AND projector jointly from
//! nothing, against `text`'s head-only bootstrap on an already
//! richly-structured frozen encoder.
//!
//! Deliberately host-side and deliberately tiny (23 inputs, one linear
//! layer, one LayerNorm): the question under test is whether a row from
//! OUTSIDE the encoder can drive the head at all, not how large a vision
//! tower can be bolted on. A GPU projector reading real pixels through a
//! real vision encoder is the natural next step once this is proven, not
//! before.

use crate::scene::{Rng, CELLS, COLORS};

/// 7 (one-hot color identity, six named colors plus "empty") + 16 (one-hot
/// cell identity) - see the module doc on why color identity is a discrete
/// symbol over the SAME closed vocabulary `instruction()` draws from, not a
/// continuous RGB triple, and why cell identity is explicit rather than left
/// for the projector to infer from a continuous position.
pub const N_COLOR_CLASSES: usize = COLORS.len() + 1;
pub const FEAT_DIM: usize = N_COLOR_CLASSES + CELLS;

const LN_EPS: f32 = 1e-5;

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
    lr: f32,
}

impl Adam {
    fn new(n: usize, lr: f32) -> Adam {
        Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0, lr }
    }

    fn step(&mut self, params: &mut [f32], grad: &[f32]) {
        self.t += 1;
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for i in 0..params.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grad[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grad[i] * grad[i];
            let mhat = self.m[i] / bc1;
            let vhat = self.v[i] / bc2;
            params[i] -= self.lr * mhat / (vhat.sqrt() + eps);
        }
    }
}

/// `LayerNorm(Linear(FEAT_DIM, d_model))`, one row per grid cell, applied
/// independently per cell - not a shared spatial kernel, since sixteen cells
/// is the whole image. `forward_rows` caches what `accumulate` needs
/// (`xhat`, `inv_std`), so the two must be called in that order, once each,
/// per training step - exactly how `crates/decide`'s own forward/backward
/// pairing already works.
pub struct Projector {
    d_model: usize,
    w: Vec<f32>,
    b: Vec<f32>,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    gw: Vec<f32>,
    gb: Vec<f32>,
    g_gamma: Vec<f32>,
    g_beta: Vec<f32>,
    adam_w: Adam,
    adam_b: Adam,
    adam_gamma: Adam,
    adam_beta: Adam,
    xhat: Vec<f32>,
    inv_std: Vec<f32>,
}

impl Projector {
    pub fn new(d_model: usize, seed: u64, lr: f32) -> Projector {
        let mut rng = Rng::new(seed);
        // Fan-in scaled uniform init for the linear stage - its own scale no
        // longer matters much downstream, since LayerNorm re-normalizes
        // every row to zero mean / unit variance regardless. `gamma`/`beta`
        // start at the identity LayerNorm (1, 0) and are what the optimizer
        // actually uses to calibrate scale, rather than fighting whatever
        // Xavier happened to produce.
        let scale = (1.0 / FEAT_DIM as f32).sqrt();
        let w = (0..d_model * FEAT_DIM).map(|_| (rng.f32() * 2.0 - 1.0) * scale).collect();
        Projector {
            d_model,
            w,
            b: vec![0.0; d_model],
            gamma: vec![1.0; d_model],
            beta: vec![0.0; d_model],
            gw: vec![0.0; d_model * FEAT_DIM],
            gb: vec![0.0; d_model],
            g_gamma: vec![0.0; d_model],
            g_beta: vec![0.0; d_model],
            adam_w: Adam::new(d_model * FEAT_DIM, lr),
            adam_b: Adam::new(d_model, lr),
            adam_gamma: Adam::new(d_model, lr),
            adam_beta: Adam::new(d_model, lr),
            xhat: vec![0.0; CELLS * d_model],
            inv_std: vec![0.0; CELLS],
        }
    }

    fn feature(cell: usize, color_idx: usize) -> [f32; FEAT_DIM] {
        assert!(color_idx < N_COLOR_CLASSES, "color_idx {color_idx} out of range");
        let mut f = [0.0f32; FEAT_DIM];
        f[color_idx] = 1.0;
        f[N_COLOR_CLASSES + cell] = 1.0;
        f
    }

    /// One row per cell, flattened `[CELLS * d_model]` - exactly the layout
    /// `Features::from_parts` expects for a contiguous run of state rows.
    /// Caches each row's LayerNorm statistics for `accumulate`.
    pub fn forward_rows(&mut self, color_idx: &[usize; CELLS]) -> Vec<f32> {
        let mut out = vec![0.0f32; CELLS * self.d_model];
        for cell in 0..CELLS {
            let feat = Self::feature(cell, color_idx[cell]);
            let mut lin = self.b.clone();
            for (h, acc) in lin.iter_mut().enumerate() {
                let row = h * FEAT_DIM;
                for (f, &x) in feat.iter().enumerate() {
                    *acc += self.w[row + f] * x;
                }
            }
            let mean = lin.iter().sum::<f32>() / self.d_model as f32;
            let var = lin.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / self.d_model as f32;
            let inv_std = 1.0 / (var + LN_EPS).sqrt();
            self.inv_std[cell] = inv_std;
            for h in 0..self.d_model {
                let xhat = (lin[h] - mean) * inv_std;
                self.xhat[cell * self.d_model + h] = xhat;
                out[cell * self.d_model + h] = self.gamma[h] * xhat + self.beta[h];
            }
        }
        out
    }

    /// Accumulate every parameter's gradient from the upstream gradient -
    /// `dL/d(forward_rows output)`, read back out of the encoder's own seed
    /// buffer after `Decide::accumulate_kept`'s reverse pass wrote it there.
    /// Call exactly once per `forward_rows` call it corresponds to, with the
    /// SAME `color_idx`; `step` clears the accumulator.
    pub fn accumulate(&mut self, color_idx: &[usize; CELLS], d_out: &[f32]) {
        assert_eq!(d_out.len(), CELLS * self.d_model, "one gradient row per cell");
        let n = self.d_model as f32;
        for cell in 0..CELLS {
            let feat = Self::feature(cell, color_idx[cell]);
            let inv_std = self.inv_std[cell];
            let xhat = &self.xhat[cell * self.d_model..(cell + 1) * self.d_model];
            let dy = &d_out[cell * self.d_model..(cell + 1) * self.d_model];

            let mut dxhat = vec![0.0f32; self.d_model];
            for h in 0..self.d_model {
                self.g_beta[h] += dy[h];
                self.g_gamma[h] += dy[h] * xhat[h];
                dxhat[h] = dy[h] * self.gamma[h];
            }
            let sum_dxhat: f32 = dxhat.iter().sum();
            let sum_dxhat_xhat: f32 = dxhat.iter().zip(xhat).map(|(a, b)| a * b).sum();

            for h in 0..self.d_model {
                // Standard LayerNorm backward: dL/d(pre-norm) from dL/d(xhat),
                // accounting for both the mean and variance depending on
                // every element of the row.
                let dlin = inv_std / n * (n * dxhat[h] - sum_dxhat - xhat[h] * sum_dxhat_xhat);
                self.gb[h] += dlin;
                let row = h * FEAT_DIM;
                for (f, &x) in feat.iter().enumerate() {
                    self.gw[row + f] += dlin * x;
                }
            }
        }
    }

    pub fn step(&mut self) {
        self.adam_w.step(&mut self.w, &self.gw);
        self.adam_b.step(&mut self.b, &self.gb);
        self.adam_gamma.step(&mut self.gamma, &self.g_gamma);
        self.adam_beta.step(&mut self.beta, &self.g_beta);
        for g in [&mut self.gw, &mut self.gb, &mut self.g_gamma, &mut self.g_beta] {
            g.iter_mut().for_each(|v| *v = 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene_colors(seed: u64) -> [usize; CELLS] {
        let mut rng = Rng::new(seed);
        let mut c = [0usize; CELLS];
        for cell in &mut c {
            *cell = rng.index(N_COLOR_CLASSES);
        }
        c
    }

    /// A closed-form check independent of `crates/decide` entirely: fit the
    /// projector against a FIXED target row for cell 0 and confirm the loss
    /// (squared error against that target) actually decreases. If the
    /// forward/accumulate/step arithmetic disagreed with each other this
    /// would diverge or stall, not just converge slowly.
    #[test]
    fn the_projector_reduces_a_fixed_target_mse_over_training() {
        let d_model = 8;
        let mut p = Projector::new(d_model, 7, 0.05);
        let colors = [1usize; CELLS];
        let target: Vec<f32> = (0..d_model).map(|i| i as f32 * 0.1 - 0.3).collect();

        let mse = |rows: &[f32]| -> f32 {
            rows[..d_model].iter().zip(&target).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / d_model as f32
        };

        let before = mse(&p.forward_rows(&colors));
        for _ in 0..200 {
            let rows = p.forward_rows(&colors);
            let mut d_rows = vec![0.0f32; CELLS * d_model];
            for h in 0..d_model {
                d_rows[h] = 2.0 * (rows[h] - target[h]) / d_model as f32;
            }
            p.accumulate(&colors, &d_rows);
            p.step();
        }
        let after = mse(&p.forward_rows(&colors));
        assert!(after < before * 0.05, "projector barely moved: {before} -> {after}");
    }

    /// `accumulate`'s hand-derived LayerNorm backward against a numerical
    /// gradient of the same scalar loss, the way every other backward pass in
    /// this repo is trusted - see the workspace's own gradcheck crate for the
    /// pattern this borrows. Checks EVERY parameter class (`w`, `b`, `gamma`,
    /// `beta`), because a LayerNorm backward is exactly the kind of thing
    /// that looks plausible and is subtly wrong in one term (the
    /// mean/variance cross terms) without this.
    ///
    /// The loss is a dot product against a FIXED random target, not
    /// `sum(output^2)`. A first version used `sum(output^2)` and it failed
    /// this check by 4-5x - not a backward bug: a LayerNorm'd row has
    /// `sum(xhat^2) ~ d_model` by construction (that is what LayerNorm
    /// guarantees), so `sum(output^2)` is nearly invariant to the very
    /// perturbations a gradient check makes, and both the true gradient AND
    /// any finite-difference estimate of it are dominated by f32 rounding at
    /// that point - confirmed by reproducing the same near-zero-vs-noise gap
    /// on bare LayerNorm arithmetic outside this crate entirely. A fixed
    /// target makes the loss linear in the output, which has no such
    /// degeneracy.
    #[test]
    fn layernorm_matches_a_numerical_gradient() {
        let d_model = 6;
        let mut p = Projector::new(d_model, 3, 0.1);
        let colors = scene_colors(5);
        let mut trng = Rng::new(42);
        let target: Vec<f32> = (0..CELLS * d_model).map(|_| trng.f32() * 2.0 - 1.0).collect();

        let loss = |p: &mut Projector| -> f32 { p.forward_rows(&colors).iter().zip(&target).map(|(v, t)| v * t).sum() };

        p.forward_rows(&colors);
        p.accumulate(&colors, &target); // dL/d(output_i) = target_i for a dot-product loss

        fn numeric_grad(mut loss: impl FnMut(&mut Vec<f32>) -> f32, params: &mut Vec<f32>) -> Vec<f32> {
            let eps = 1e-3f32;
            let mut g = vec![0.0f32; params.len()];
            for i in 0..params.len() {
                let orig = params[i];
                params[i] = orig + eps;
                let plus = loss(params);
                params[i] = orig - eps;
                let minus = loss(params);
                params[i] = orig;
                g[i] = (plus - minus) / (2.0 * eps);
            }
            g
        }

        {
            let mut w = p.w.clone();
            let g = numeric_grad(
                |w| {
                    p.w.copy_from_slice(w);
                    loss(&mut p)
                },
                &mut w,
            );
            p.w.copy_from_slice(&w);
            for (i, (&a, &n)) in p.gw.iter().zip(&g).enumerate() {
                assert!((a - n).abs() < 1e-2 * (n.abs() + 1.0), "w[{i}]: analytic {a} vs numeric {n}");
            }
        }
        {
            let mut b = p.b.clone();
            let g = numeric_grad(
                |b| {
                    p.b.copy_from_slice(b);
                    loss(&mut p)
                },
                &mut b,
            );
            p.b.copy_from_slice(&b);
            for (i, (&a, &n)) in p.gb.iter().zip(&g).enumerate() {
                assert!((a - n).abs() < 1e-2 * (n.abs() + 1.0), "b[{i}]: analytic {a} vs numeric {n}");
            }
        }
        {
            let mut gamma = p.gamma.clone();
            let g = numeric_grad(
                |gamma| {
                    p.gamma.copy_from_slice(gamma);
                    loss(&mut p)
                },
                &mut gamma,
            );
            p.gamma.copy_from_slice(&gamma);
            for (i, (&a, &n)) in p.g_gamma.iter().zip(&g).enumerate() {
                assert!((a - n).abs() < 1e-2 * (n.abs() + 1.0), "gamma[{i}]: analytic {a} vs numeric {n}");
            }
        }
        {
            let mut beta = p.beta.clone();
            let g = numeric_grad(
                |beta| {
                    p.beta.copy_from_slice(beta);
                    loss(&mut p)
                },
                &mut beta,
            );
            p.beta.copy_from_slice(&beta);
            for (i, (&a, &n)) in p.g_beta.iter().zip(&g).enumerate() {
                assert!((a - n).abs() < 1e-2 * (n.abs() + 1.0), "beta[{i}]: analytic {a} vs numeric {n}");
            }
        }
    }

    /// A row's post-LayerNorm scale is now within the same order of
    /// magnitude as a real encoded row regardless of the linear stage's own
    /// (arbitrary) init - the property the module doc measured missing
    /// before this existed.
    #[test]
    fn layernorm_output_has_unit_scale_at_init() {
        let d_model = 32;
        let mut p = Projector::new(d_model, 1, 0.01);
        let colors = scene_colors(9);
        let rows = p.forward_rows(&colors);
        for (cell, chunk) in rows.chunks(d_model).enumerate() {
            let rms = (chunk.iter().map(|v| v * v).sum::<f32>() / d_model as f32).sqrt();
            assert!((0.5..2.0).contains(&rms), "cell {cell}: rms {rms} is not unit-scale");
        }
    }
}
