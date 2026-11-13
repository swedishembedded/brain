// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements codebook-utilisation losses for image tokenizers for its clients. If your team needs
// expertise in training VQ codebooks that stay in use, you can procure our services by sending an email to
// info@swedishembedded.com.

//! LlamaGen's codebook **entropy loss** (`compute_entropy_loss` in `tokenizer/tokenizer_image/vq_model.py`), with its
//! gradient.
//!
//! Per token the quantiser has a squared distance `d_ik = |z_i - e_k|^2` to every code. The loss reads
//! `affinity = -d / temperature` as logits and asks two things of the softmax `p_ik` over codes:
//!
//! ```text
//! L = mean_i H(p_i)  -  H(mean_i p_i)
//! ```
//!
//! each token's assignment should be **confident** (low per-token entropy) while the codebook as a whole is used
//! **evenly** (high entropy of the average assignment). The reference's `avg_entropy` is
//! `-sum_k avg_k log(avg_k + 1e-5)`; its `log_softmax(affinity + 1e-5)` is `log_softmax(affinity)` (a softmax is
//! shift invariant), so the per-token term is the exact `-sum p log p`. The released recipe scales this by
//! `entropy_loss_ratio` (0, so it never ran there; [`crate::Objective::entropy_loss_ratio`]) at `temperature = 0.01`.
//!
//! It is differentiable in both the rows `z` (the encoder) and the codebook `e`, and unlike the commitment terms
//! nothing in it is detached.
//!
//! # Gradient
//!
//! With `H_i = -sum_k p_ik log p_ik`, `avg_k = mean_i p_ik`, `c_k = log(avg_k + 1e-5) + avg_k / (avg_k + 1e-5)`
//! and `R` rows, the logit gradient is
//!
//! ```text
//! dL/da_ik = p_ik / R * ( c_k - sum_j p_ij c_j - log p_ik - H_i )
//! ```
//!
//! and `a_ik = -(|z_i|^2 + |e_k|^2 - 2 z_i . e_k) / temperature` gives `da_ik/dz_i = 2 (e_k - z_i) / temperature`
//! and `da_ik/de_k = -da_ik/dz_i`.
//!
//! # Where it runs
//!
//! The training step runs it on the device: `entropy_logp` (per-token log-softmax and entropy), `entropy_cols`
//! (the per-code average assignment), and the gradient pair `entropy_dz` and `entropy_dcb`
//! (`crates/kernels/wgsl/entropy_*.wgsl`), none of which materialises anything off the device but the two scalar
//! reductions. This module is the **f64 oracle** those kernels are checked against (`tests/it/objective.rs`): the
//! same formulas on the host, one row at a time, plus the finite-difference test of the formulas themselves.

/// The entropy term at one point: its value and its gradient with respect to the rows and the codebook.
#[derive(Clone, Debug)]
pub struct Entropy {
    pub loss: f64,
    /// `[rows, dim]`.
    pub d_rows: Vec<f32>,
    /// `[codes, dim]`.
    pub d_codebook: Vec<f32>,
}

/// The `epsilon` inside the reference's `log(avg_probs + 1e-5)`.
const AVG_EPS: f64 = 1e-5;

/// The LlamaGen entropy loss of `rows` (`[rows, dim]`) against `codebook` (`[codes, dim]`) at `temperature`, and its
/// gradient. Panics on a shape that does not divide, a non-positive temperature, or no rows (a loss over nothing).
pub fn entropy_loss(rows: &[f32], codebook: &[f32], dim: usize, temperature: f32) -> Entropy {
    assert!(dim > 0 && rows.len().is_multiple_of(dim) && codebook.len().is_multiple_of(dim), "entropy_loss: rows {} and codebook {} are not whole rows of {dim}", rows.len(), codebook.len());
    assert!(temperature > 0.0 && temperature.is_finite(), "entropy_loss: temperature {temperature} must be positive");
    let (r, k) = (rows.len() / dim, codebook.len() / dim);
    assert!(r > 0 && k > 0, "entropy_loss: {r} rows against {k} codes");
    let (z, e): (Vec<f64>, Vec<f64>) = (rows.iter().map(|v| *v as f64).collect(), codebook.iter().map(|v| *v as f64).collect());
    let (loss, dz, de) = entropy_f64(&z, &e, dim, temperature as f64);
    Entropy { loss, d_rows: dz.into_iter().map(|v| v as f32).collect(), d_codebook: de.into_iter().map(|v| v as f32).collect() }
}

/// `log p_i` (log-softmax of the affinity) of row `i` into `logp`.
fn log_probs(z: &[f64], e: &[f64], dim: usize, tau: f64, i: usize, logp: &mut [f64]) {
    let zi = &z[i * dim..(i + 1) * dim];
    let zz: f64 = zi.iter().map(|v| v * v).sum();
    for (k, a) in logp.iter_mut().enumerate() {
        let ek = &e[k * dim..(k + 1) * dim];
        let (ee, ze) = ek.iter().zip(zi).fold((0.0, 0.0), |(ee, ze), (c, x)| (ee + c * c, ze + c * x));
        *a = -(zz + ee - 2.0 * ze) / tau;
    }
    let max = logp.iter().fold(f64::NEG_INFINITY, |m, v| m.max(*v));
    let log_sum = max + logp.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
    logp.iter_mut().for_each(|v| *v -= log_sum);
}

fn entropy_f64(z: &[f64], e: &[f64], dim: usize, tau: f64) -> (f64, Vec<f64>, Vec<f64>) {
    let (r, k) = (z.len() / dim, e.len() / dim);
    let mut logp = vec![0.0; k];
    // Pass one: the mean assignment and the mean per-token entropy.
    let mut avg = vec![0.0; k];
    let mut sample = 0.0;
    for i in 0..r {
        log_probs(z, e, dim, tau, i, &mut logp);
        for (a, lp) in avg.iter_mut().zip(&logp) {
            let p = lp.exp();
            *a += p / r as f64;
            sample -= p * lp / r as f64;
        }
    }
    let average: f64 = -avg.iter().map(|a| a * (a + AVG_EPS).ln()).sum::<f64>();
    let c: Vec<f64> = avg.iter().map(|a| (a + AVG_EPS).ln() + a / (a + AVG_EPS)).collect();

    // Pass two: the gradient, row by row.
    let (mut dz, mut de) = (vec![0.0; z.len()], vec![0.0; e.len()]);
    for i in 0..r {
        log_probs(z, e, dim, tau, i, &mut logp);
        let h: f64 = -logp.iter().map(|lp| lp.exp() * lp).sum::<f64>();
        let mean_c: f64 = logp.iter().zip(&c).map(|(lp, c)| lp.exp() * c).sum();
        let zi = &z[i * dim..(i + 1) * dim];
        for (ki, lp) in logp.iter().enumerate() {
            let g = lp.exp() / r as f64 * (c[ki] - mean_c - lp - h);
            // da/dz = 2 (e - z) / tau, da/de = -da/dz.
            let w = 2.0 * g / tau;
            let ek = &e[ki * dim..(ki + 1) * dim];
            for d in 0..dim {
                let step = w * (ek[d] - zi[d]);
                dz[i * dim + d] += step;
                de[ki * dim + d] -= step;
            }
        }
    }
    (sample - average, dz, de)
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_testutil::rounding::U;

    const DIM: usize = 3;

    /// `rows` and `codes` random points in `[-1, 1]^3`, from a fixed stream.
    fn points(n: usize, seed: u64) -> Vec<f32> {
        data::rng::Lcg::new(seed).vec(n * DIM)
    }

    /// `compute_entropy_loss(-d)` transcribed from the reference, term for term, in f64 on the distance matrix:
    /// logits divided by the temperature, `softmax`, `log_softmax(logits + 1e-5)`, the two entropies.
    fn reference(rows: &[f32], codes: &[f32], temperature: f32) -> f64 {
        let (r, k) = (rows.len() / DIM, codes.len() / DIM);
        let mut probs = vec![vec![0.0f64; k]; r];
        let mut log_probs = vec![vec![0.0f64; k]; r];
        for i in 0..r {
            let logits: Vec<f64> = (0..k)
                .map(|j| {
                    let d: f64 = (0..DIM).map(|c| (rows[i * DIM + c] as f64 - codes[j * DIM + c] as f64).powi(2)).sum();
                    -d / temperature as f64
                })
                .collect();
            let softmax = |xs: &[f64]| {
                let m = xs.iter().fold(f64::NEG_INFINITY, |m, v| m.max(*v));
                let s: f64 = xs.iter().map(|v| (v - m).exp()).sum();
                xs.iter().map(|v| (v - m).exp() / s).collect::<Vec<f64>>()
            };
            probs[i] = softmax(&logits);
            let shifted: Vec<f64> = logits.iter().map(|v| v + 1e-5).collect();
            log_probs[i] = softmax(&shifted).iter().map(|p| p.ln()).collect();
        }
        let avg: Vec<f64> = (0..k).map(|j| probs.iter().map(|p| p[j]).sum::<f64>() / r as f64).collect();
        let avg_entropy = -avg.iter().map(|a| a * (a + 1e-5).ln()).sum::<f64>();
        let sample_entropy = -(0..r).map(|i| (0..k).map(|j| probs[i][j] * log_probs[i][j]).sum::<f64>()).sum::<f64>() / r as f64;
        sample_entropy - avg_entropy
    }

    /// Spec: the value is LlamaGen's `compute_entropy_loss`, at the released temperature and at a softer one.
    #[test]
    fn the_value_is_the_reference_loss() {
        let (rows, codes) = (points(7, 1), points(5, 2));
        for tau in [0.01f32, 0.3] {
            let got = entropy_loss(&rows, &codes, DIM, tau).loss;
            let want = reference(&rows, &codes, tau);
            assert!((got - want).abs() <= 1e-9 * want.abs().max(1.0), "temperature {tau}: {got} vs the reference {want}");
        }
    }

    /// Spec: tokens that each sit on a different code are confident and spread evenly over the codebook, so the loss
    /// is `-ln(codes used)`; tokens that all sit on one code are confident and use one code, so it is zero.
    #[test]
    fn confident_even_use_is_rewarded_and_collapse_is_not() {
        let codes = points(4, 3);
        let spread = entropy_loss(&codes, &codes, DIM, 0.01).loss;
        assert!((spread + 4.0f64.ln()).abs() < 1e-3, "spread over 4 codes: {spread}");
        let collapsed: Vec<f32> = codes[..DIM].repeat(4);
        let one = entropy_loss(&collapsed, &codes, DIM, 0.01).loss;
        assert!(one.abs() < 1e-3, "all on one code: {one}");
        assert!(spread < one);
    }

    /// Spec: the gradient is the derivative of the value, in the rows and the codebook alike. Central differences of
    /// the f64 loss, Richardson-extrapolated so the step's `eps^2` truncation cancels and what is left is `O(eps^4)`;
    /// the result is returned as f32, so each entry is within one unit round-off `U` of its f64 value.
    #[test]
    fn the_gradient_is_the_derivative_in_the_rows_and_the_codebook() {
        let (rows, codes) = (points(6, 4), points(5, 5));
        let tau = 0.3;
        let got = entropy_loss(&rows, &codes, DIM, tau);
        let fd = |at: &dyn Fn(f64) -> f64, eps: f64| (at(eps) - at(-eps)) / (2.0 * eps);
        let rich = |at: &dyn Fn(f64) -> f64| (4.0 * fd(at, 5e-4) - fd(at, 1e-3)) / 3.0;
        let check = |name: &str, analytic: &[f32], n: usize, bump: &dyn Fn(usize, f64) -> f64| {
            let scale = analytic.iter().fold(0.0f64, |m, v| m.max((*v as f64).abs()));
            assert!(scale > 1e-3, "{name}: a vanishing gradient proves nothing");
            for j in 0..n {
                let want = rich(&|d| bump(j, d));
                let tol = 2.0 * U * want.abs().max(analytic[j].abs() as f64) + 1e-9 * scale;
                assert!((analytic[j] as f64 - want).abs() <= tol, "{name}[{j}]: analytic {} vs central difference {want} (tolerance {tol:e})", analytic[j]);
            }
        };
        let at_rows = |j: usize, d: f64| {
            let mut z: Vec<f64> = rows.iter().map(|v| *v as f64).collect();
            z[j] += d;
            let e: Vec<f64> = codes.iter().map(|v| *v as f64).collect();
            entropy_f64(&z, &e, DIM, tau as f64).0
        };
        let at_codes = |j: usize, d: f64| {
            let z: Vec<f64> = rows.iter().map(|v| *v as f64).collect();
            let mut e: Vec<f64> = codes.iter().map(|v| *v as f64).collect();
            e[j] += d;
            entropy_f64(&z, &e, DIM, tau as f64).0
        };
        check("d_rows", &got.d_rows, rows.len(), &at_rows);
        check("d_codebook", &got.d_codebook, codes.len(), &at_codes);
    }

    #[test]
    #[should_panic(expected = "temperature")]
    fn a_non_positive_temperature_is_refused() {
        entropy_loss(&points(2, 6), &points(2, 7), DIM, 0.0);
    }
}
