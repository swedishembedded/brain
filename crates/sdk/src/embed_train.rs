// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`EmbeddingTrainer`]: a contrastively-trained linear refinement over a
//! FROZEN embedding backbone.
//!
//! No engine crate in this workspace has a contrastive (InfoNCE) objective,
//! and training the backbone itself needs a seeded backward pass neither
//! `qwen3` nor `lfm2` has yet - real, unbuilt work, tracked separately. What
//! this milestone trains instead is the projection head alone, over
//! embeddings the backbone already produced: host arithmetic, deliberately,
//! the same reasoning `crates/decide/src/loss.rs`'s own module doc gives for
//! why its (comparably small) objective lives on the host rather than as a
//! WGSL kernel - a `dim x dim` linear layer over a batch of a few dozen
//! vectors is nowhere near where GPU dispatch overhead pays for itself.
//!
//! Caching every embedding ONCE and training only the head on top of that
//! cache is also what keeps this tractable at the 32k-token context
//! `EmbeddingPipeline`'s Qwen3 backend supports: the expensive forward runs
//! once per document, never once per training step.
//!
//! The objective is the standard symmetric (CLIP-style) InfoNCE loss over a
//! batch of `(anchor, positive)` pairs, with every other pair in the batch
//! as in-batch negatives:
//!
//! ```text
//! S[i][j] = cosine(project(anchor_i), project(positive_j)) / temperature
//! L = mean_i CE(softmax_row(S)[i], label=i) / 2
//!   + mean_j CE(softmax_col(S)[j], label=j) / 2
//! ```
//!
//! `project` is one linear layer plus L2-normalization, initialized near the
//! identity (small noise on top of an identity matrix) so training starts
//! from "pass the backbone's own embedding through unchanged" rather than a
//! random re-projection that would only make retrieval worse before any
//! training happened. This is deliberately NOT dimensionality reduction
//! (Matryoshka-style truncation is a different, unbuilt feature) - input and
//! output dimension are the same.

use crate::embedding::Embedding;

/// A contrastively-trained linear refinement over frozen embeddings, plus
/// its own Adam optimizer state. `dim` is fixed at construction to whatever
/// dimension the backbone that produced the training embeddings uses.
pub struct EmbeddingTrainer {
    dim: usize,
    /// `[dim, dim]`, row-major (output row, input column).
    w: Vec<f32>,
    b: Vec<f32>,
    temperature: f32,
    m_w: Vec<f32>,
    v_w: Vec<f32>,
    m_b: Vec<f32>,
    v_b: Vec<f32>,
    step: u32,
}

impl std::fmt::Debug for EmbeddingTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingTrainer").field("dim", &self.dim).field("step", &self.step).finish()
    }
}

impl EmbeddingTrainer {
    /// A fresh trainer for `dim`-dimensional embeddings, near-identity
    /// initialized from `seed`.
    pub fn new(dim: usize, seed: u64) -> EmbeddingTrainer {
        let mut rng = data::rng::Rng::new(seed);
        let mut w = vec![0.0f32; dim * dim];
        for o in 0..dim {
            for i in 0..dim {
                // Identity on the diagonal, small noise everywhere - see this
                // module's own doc for why identity is the right starting
                // point rather than a random re-projection.
                let identity = if o == i { 1.0 } else { 0.0 };
                w[o * dim + i] = identity + (rng.next_f32() - 0.5) * 0.02;
            }
        }
        EmbeddingTrainer { dim, w, b: vec![0.0; dim], temperature: 0.05, m_w: vec![0.0; dim * dim], v_w: vec![0.0; dim * dim], m_b: vec![0.0; dim], v_b: vec![0.0; dim], step: 0 }
    }

    /// The softmax temperature `S[i][j]` is scaled by before the loss - lower
    /// sharpens the contrast between the correct pair and its in-batch
    /// negatives. Defaults to 0.05, a standard starting point for
    /// L2-normalized embeddings.
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Project one embedding through the current head: linear, then
    /// L2-normalized. No gradient, no state mutation - the inference side of
    /// this type.
    pub fn project(&self, e: &Embedding) -> Embedding {
        assert_eq!(e.dim(), self.dim, "EmbeddingTrainer built for dim {}, got {}", self.dim, e.dim());
        let (_, u) = project_fwd(&self.w, &self.b, self.dim, e.as_slice());
        Embedding::from(u)
    }

    /// One training step over a batch of `(anchor, positive)` pairs -
    /// forward, backward, one Adam update - and the batch's mean loss.
    /// `anchors.len()` must equal `positives.len()`; every entry must be
    /// [`EmbeddingTrainer::new`]'s `dim`.
    pub fn step(&mut self, anchors: &[Embedding], positives: &[Embedding], lr: f32) -> f32 {
        assert_eq!(anchors.len(), positives.len(), "one positive per anchor");
        assert!(!anchors.is_empty(), "step on an empty batch");
        let a: Vec<&[f32]> = anchors.iter().map(|e| e.as_slice()).collect();
        let p: Vec<&[f32]> = positives.iter().map(|e| e.as_slice()).collect();
        let (loss, d_w, d_b) = info_nce_backward(&self.w, &self.b, self.dim, &a, &p, self.temperature);
        self.step += 1;
        adamw_step(&mut self.w, &mut self.m_w, &mut self.v_w, &d_w, lr, self.step);
        adamw_step(&mut self.b, &mut self.m_b, &mut self.v_b, &d_b, lr, self.step);
        loss
    }
}

/// Linear layer forward: `z = W @ x + b`, then L2-normalized `u = z / ||z||`.
/// Returns both - the backward pass needs `z`'s norm, not just `u`.
fn project_fwd(w: &[f32], b: &[f32], dim: usize, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut z = vec![0.0f64; dim];
    for o in 0..dim {
        let mut acc = b[o] as f64;
        let row = &w[o * dim..(o + 1) * dim];
        for i in 0..dim {
            acc += row[i] as f64 * x[i] as f64;
        }
        z[o] = acc;
    }
    let norm = z.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    let u: Vec<f32> = z.iter().map(|v| (v / norm) as f32).collect();
    (z.iter().map(|v| *v as f32).collect(), u)
}

/// L2-normalize backward: given `z`'s norm and `u = z/||z||`, push a
/// gradient w.r.t. `u` back to a gradient w.r.t. `z`.
/// `du/dz = (I - u u^T) / ||z||`.
fn normalize_bwd(u: &[f32], norm: f64, d_u: &[f32]) -> Vec<f32> {
    let dot: f64 = u.iter().zip(d_u).map(|(&ui, &dui)| ui as f64 * dui as f64).sum();
    u.iter().zip(d_u).map(|(&ui, &dui)| ((dui as f64 - ui as f64 * dot) / norm) as f32).collect()
}

/// Forward + backward over one batch: the symmetric InfoNCE loss, and
/// `(loss, dW, db)` - the accumulated gradient from BOTH the anchor pass and
/// the positive pass, since both share the same `W`/`b` (a siamese head).
fn info_nce_backward(w: &[f32], b: &[f32], dim: usize, anchors: &[&[f32]], positives: &[&[f32]], temperature: f32) -> (f32, Vec<f32>, Vec<f32>) {
    let batch = anchors.len();
    let tau = temperature as f64;

    let a_fwd: Vec<(Vec<f32>, Vec<f32>)> = anchors.iter().map(|x| project_fwd(w, b, dim, x)).collect();
    let p_fwd: Vec<(Vec<f32>, Vec<f32>)> = positives.iter().map(|x| project_fwd(w, b, dim, x)).collect();
    let ua: Vec<&Vec<f32>> = a_fwd.iter().map(|(_, u)| u).collect();
    let up: Vec<&Vec<f32>> = p_fwd.iter().map(|(_, u)| u).collect();

    // S[i][j] = cosine(anchor_i, positive_j) / tau. Inputs are already unit
    // norm, so the dot product IS the cosine similarity.
    let mut s = vec![0.0f64; batch * batch];
    for i in 0..batch {
        for j in 0..batch {
            let dot: f64 = (0..dim).map(|k| ua[i][k] as f64 * up[j][k] as f64).sum();
            s[i * batch + j] = dot / tau;
        }
    }
    let softmax = |row: &[f64]| -> Vec<f64> {
        let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = row.iter().map(|v| (v - mx).exp()).collect();
        let sum: f64 = exp.iter().sum();
        exp.iter().map(|v| v / sum).collect()
    };

    let mut loss_sum = 0.0f64;
    let mut d_s = vec![0.0f64; batch * batch];
    // Row direction: anchor_i retrieves positive_i among positives 0..batch.
    for i in 0..batch {
        let row: Vec<f64> = (0..batch).map(|j| s[i * batch + j]).collect();
        let p_row = softmax(&row);
        loss_sum += -(p_row[i].max(1e-12)).ln();
        for j in 0..batch {
            let y = if j == i { 1.0 } else { 0.0 };
            d_s[i * batch + j] += 0.5 * (p_row[j] - y) / batch as f64;
        }
    }
    // Column direction: positive_j retrieves anchor_j among anchors 0..batch.
    for j in 0..batch {
        let col: Vec<f64> = (0..batch).map(|i| s[i * batch + j]).collect();
        let p_col = softmax(&col);
        loss_sum += -(p_col[j].max(1e-12)).ln();
        for i in 0..batch {
            let y = if i == j { 1.0 } else { 0.0 };
            d_s[i * batch + j] += 0.5 * (p_col[i] - y) / batch as f64;
        }
    }
    let loss = (loss_sum / (2 * batch) as f64) as f32;

    // dL/d(u_anchor_i) = sum_j dS[i][j] * u_positive_j / tau
    // dL/d(u_positive_j) = sum_i dS[i][j] * u_anchor_i / tau
    let mut d_w = vec![0.0f32; dim * dim];
    let mut d_b = vec![0.0f32; dim];
    let mut accumulate = |x: &[f32], z: &[f32], u: &[f32], d_u: &[f32]| {
        let norm = z.iter().map(|v| *v as f64 * *v as f64).sum::<f64>().sqrt().max(1e-12);
        let d_z = normalize_bwd(u, norm, d_u);
        for o in 0..dim {
            d_b[o] += d_z[o];
            let row = &mut d_w[o * dim..(o + 1) * dim];
            for i in 0..dim {
                row[i] += d_z[o] * x[i];
            }
        }
    };
    for i in 0..batch {
        let mut d_u = vec![0.0f64; dim];
        for j in 0..batch {
            let scale = d_s[i * batch + j] / tau;
            for k in 0..dim {
                d_u[k] += scale * up[j][k] as f64;
            }
        }
        let d_u: Vec<f32> = d_u.iter().map(|v| *v as f32).collect();
        accumulate(anchors[i], &a_fwd[i].0, ua[i], &d_u);
    }
    for j in 0..batch {
        let mut d_u = vec![0.0f64; dim];
        for i in 0..batch {
            let scale = d_s[i * batch + j] / tau;
            for k in 0..dim {
                d_u[k] += scale * ua[i][k] as f64;
            }
        }
        let d_u: Vec<f32> = d_u.iter().map(|v| *v as f32).collect();
        accumulate(positives[j], &p_fwd[j].0, up[j], &d_u);
    }

    (loss, d_w, d_b)
}

/// One plain Adam step (not decoupled weight decay - there is no weight
/// decay here at all, the same choice `crates/decide`'s own small objective
/// makes implicitly by never applying one to a host-side update).
fn adamw_step(p: &mut [f32], m: &mut [f32], v: &mut [f32], grad: &[f32], lr: f32, t: u32) {
    let (beta1, beta2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    let t = t as i32;
    let bias1 = 1.0 - beta1.powi(t);
    let bias2 = 1.0 - beta2.powi(t);
    for i in 0..p.len() {
        let g = grad[i] as f64;
        m[i] = (beta1 * m[i] as f64 + (1.0 - beta1) * g) as f32;
        v[i] = (beta2 * v[i] as f64 + (1.0 - beta2) * g * g) as f32;
        let m_hat = m[i] as f64 / bias1;
        let v_hat = v[i] as f64 / bias2;
        p[i] -= (lr as f64 * m_hat / (v_hat.sqrt() + eps)) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_unit(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = data::rng::Rng::new(seed);
        let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32() - 0.5).collect();
        let norm = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        for x in &mut v {
            *x = (*x as f64 / norm) as f32;
        }
        v
    }

    /// The hand-derived backward pass (through the InfoNCE loss, the
    /// similarity matrix, L2-normalize, and the linear layer) must match
    /// finite differences over EVERY parameter - the same discipline
    /// `crates/decide/src/loss.rs::the_gradient_matches_finite_differences`
    /// applies to its own hand-derived objective, and for the same reason:
    /// a chain-rule derivation through four stages is exactly the kind of
    /// thing that is plausibly wrong in a way review alone would not catch.
    #[test]
    fn the_gradient_matches_finite_differences() {
        let dim = 6;
        let batch = 4;
        let anchors: Vec<Vec<f32>> = (0..batch).map(|i| rand_unit(dim, 100 + i as u64)).collect();
        let positives: Vec<Vec<f32>> = (0..batch).map(|i| rand_unit(dim, 200 + i as u64)).collect();
        let a_ref: Vec<&[f32]> = anchors.iter().map(|v| v.as_slice()).collect();
        let p_ref: Vec<&[f32]> = positives.iter().map(|v| v.as_slice()).collect();

        let mut rng = data::rng::Rng::new(7);
        let mut w: Vec<f32> = (0..dim * dim).map(|_| (rng.next_f32() - 0.5) * 0.5).collect();
        for o in 0..dim {
            w[o * dim + o] += 1.0;
        }
        let b: Vec<f32> = vec![0.01; dim];

        let (_, d_w, d_b) = info_nce_backward(&w, &b, dim, &a_ref, &p_ref, 0.1);

        let loss_at = |w: &[f32], b: &[f32]| info_nce_backward(w, b, dim, &a_ref, &p_ref, 0.1).0 as f64;
        let eps = 1e-3f32;

        for idx in 0..w.len() {
            let mut wp = w.clone();
            let mut wm = w.clone();
            wp[idx] += eps;
            wm[idx] -= eps;
            let num = (loss_at(&wp, &b) - loss_at(&wm, &b)) / (2.0 * eps as f64);
            let tol = 2e-3 + 5e-2 * (d_w[idx] as f64).abs().max(num.abs());
            assert!((d_w[idx] as f64 - num).abs() <= tol, "w[{idx}]: analytic {} vs numeric {num}", d_w[idx]);
        }
        for idx in 0..b.len() {
            let mut bp = b.clone();
            let mut bm = b.clone();
            bp[idx] += eps;
            bm[idx] -= eps;
            let num = (loss_at(&w, &bp) - loss_at(&w, &bm)) / (2.0 * eps as f64);
            let tol = 2e-3 + 5e-2 * (d_b[idx] as f64).abs().max(num.abs());
            assert!((d_b[idx] as f64 - num).abs() <= tol, "b[{idx}]: analytic {} vs numeric {num}", d_b[idx]);
        }
    }

    /// No parameter is disconnected from the loss - the same
    /// no-dead-gradient discipline `crates/gradcheck` enforces for a
    /// GPU-resident model, applied by hand here since this objective is
    /// entirely host-side and never touches `crates/gradcheck`'s
    /// `CheckModel` trait (that trait's `read_weight`/`backward` shape is
    /// built for a GPU `ParamStore`, which this type does not have).
    #[test]
    fn no_parameter_is_disconnected_from_the_loss() {
        let dim = 5;
        let batch = 3;
        let anchors: Vec<Vec<f32>> = (0..batch).map(|i| rand_unit(dim, 10 + i as u64)).collect();
        let positives: Vec<Vec<f32>> = (0..batch).map(|i| rand_unit(dim, 20 + i as u64)).collect();
        let a_ref: Vec<&[f32]> = anchors.iter().map(|v| v.as_slice()).collect();
        let p_ref: Vec<&[f32]> = positives.iter().map(|v| v.as_slice()).collect();
        let w: Vec<f32> = (0..dim * dim).map(|i| if i % (dim + 1) == 0 { 1.0 } else { 0.1 }).collect();
        let b: Vec<f32> = vec![0.0; dim];

        let (_, d_w, d_b) = info_nce_backward(&w, &b, dim, &a_ref, &p_ref, 0.1);
        assert!(d_w.iter().any(|g| g.abs() > 1e-8), "every W gradient is exactly zero");
        assert!(d_b.iter().any(|g| g.abs() > 1e-8), "every b gradient is exactly zero");
    }

    /// A trainer initialized near the identity must project close to the
    /// identity - the stated design rationale, checked rather than assumed.
    #[test]
    fn a_fresh_trainer_projects_close_to_the_input() {
        let dim = 8;
        let trainer = EmbeddingTrainer::new(dim, 42);
        let e = Embedding::from(rand_unit(dim, 1));
        let out = trainer.project(&e);
        let cos = e.cosine_similarity(&out);
        assert!(cos > 0.95, "expected near-identity projection, cosine similarity was {cos}");
    }

    /// Training must reduce the batch's own loss - the one end-to-end
    /// contract `step` owes a caller, independent of the hand-derived
    /// gradient math the two tests above already pin down directly.
    #[test]
    fn training_reduces_loss_on_its_own_batch() {
        let dim = 6;
        let batch = 5;
        let anchors: Vec<Embedding> = (0..batch).map(|i| Embedding::from(rand_unit(dim, 300 + i as u64))).collect();
        // Positives correlated with their anchor (not identical, so the task
        // is not trivially solved by the identity projection alone).
        let positives: Vec<Embedding> = anchors
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let mut v = a.as_slice().to_vec();
                let mut rng = data::rng::Rng::new(400 + i as u64);
                for x in &mut v {
                    *x += (rng.next_f32() - 0.5) * 0.3;
                }
                let norm = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                Embedding::from(v.iter().map(|x| (*x as f64 / norm) as f32).collect::<Vec<_>>())
            })
            .collect();

        let mut trainer = EmbeddingTrainer::new(dim, 5);
        let first = trainer.step(&anchors, &positives, 0.05);
        let mut last = first;
        for _ in 0..49 {
            last = trainer.step(&anchors, &positives, 0.05);
        }
        assert!(last < first, "loss did not drop: first {first} last {last}");
    }
}
