// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Distilled latent sparse attention (D-LSA): training an indexer's query and
//! key projections against the frozen model's own attention.
//!
//! Per layer and sampled query `i`, the teacher is the model's head-averaged
//! attention over the keys a sparse read must find - those outside the sinks
//! and the window, `R_i = [sink, i + 1 - window)` - restricted to its `keep`
//! largest entries and renormalised:
//!
//! `A_ij = mean_h softmax_{j in R_i}(q_ih . k_j / sqrt(d_h))`
//!
//! and the student is the indexer's own distribution over `R_i`,
//! `Â_ij = softmax_{j in R_i}(q̃_i . k̃_j / τ)`, with `q̃ = unit(q W_Q)` and
//! `k̃ = unit(k W_K)` over the un-rotated, head-summed inputs the device
//! index reads. The loss is the forward KL divergence `KL(A || Â)`, averaged
//! over queries; only `W_Q`, `W_K` and `τ` (as `ln τ`) train. Clustering
//! plays no part in training: buckets are built from whatever keys the
//! trained projections produce.
//!
//! The arithmetic is host math with analytic gradients (a few hundred
//! million multiply-adds per document and layer), layers in parallel.

use super::indexer::{IndexGeometry, Indexer, LayerProjection};
use super::LsaConfig;

/// One sampled query's teacher: `(key position, probability)` over its
/// long-range keys, summing to one.
pub type Target = Vec<(u32, f32)>;

/// One document's training material for one layer.
#[derive(Clone, Debug)]
pub struct LayerSample {
    /// `[T, input_dim]` un-rotated keys.
    pub keys: Vec<f32>,
    /// `[S, input_dim]` un-rotated, head-summed queries of the sampled
    /// positions.
    pub queries: Vec<f32>,
    /// `[S]` the sampled positions.
    pub positions: Vec<u32>,
    /// `[S]` each query's teacher.
    pub targets: Vec<Target>,
}

/// The teacher of each sampled query from the model's rotated heads.
///
/// `q` is `[S, n_heads * head_dim]` (the sampled rows), `k` is `[T,
/// n_kv_heads * head_dim]` (every key), both post-RoPE as attention reads
/// them. Returns, per query, its `keep` largest head-averaged attention
/// weights over `R_i`, renormalised - or an empty target when `R_i` is empty.
pub fn teacher_targets(q: &[f32], k: &[f32], positions: &[u32], geometry: &IndexGeometry, cfg: &LsaConfig, keep: usize) -> Vec<Target> {
    let (nh, nkv, hd) = (geometry.n_heads, geometry.n_kv_heads, geometry.head_dim);
    let group = nh / nkv;
    let scale = 1.0 / (hd as f64).sqrt();
    positions
        .iter()
        .enumerate()
        .map(|(s, &p)| {
            let end = (p as usize + 1).saturating_sub(cfg.window);
            if end <= cfg.sink {
                return Vec::new();
            }
            let keys = cfg.sink..end;
            let mut mean = vec![0f64; keys.len()];
            let mut logits = vec![0f64; keys.len()];
            for h in 0..nh {
                let qh = &q[s * nh * hd + h * hd..][..hd];
                let kv = (h / group) * hd;
                for (slot, j) in keys.clone().enumerate() {
                    logits[slot] = scale * super::dot(qh, &k[j * nkv * hd + kv..][..hd]);
                }
                let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = logits.iter().map(|z| (z - m).exp()).sum();
                for (acc, z) in mean.iter_mut().zip(&logits) {
                    *acc += (z - m).exp() / total / nh as f64;
                }
            }
            let mut ranked: Vec<(u32, f64)> = keys.zip(mean).map(|(j, a)| (j as u32, a)).collect();
            if ranked.len() > keep {
                ranked.select_nth_unstable_by(keep - 1, |a, b| b.1.total_cmp(&a.1));
                ranked.truncate(keep);
            }
            let total: f64 = ranked.iter().map(|r| r.1).sum();
            ranked.into_iter().map(|(j, a)| (j, (a / total) as f32)).collect()
        })
        .collect()
}

/// One layer's trainable indexer: projections, temperature and Adam state.
#[derive(Clone, Debug)]
pub struct LayerDistiller {
    input_dim: usize,
    index_dim: usize,
    sink: usize,
    window: usize,
    /// `[input_dim, index_dim]`.
    pub wq: Vec<f32>,
    /// `[input_dim, index_dim]`.
    pub wk: Vec<f32>,
    /// `ln τ`.
    pub log_tau: f32,
    adam: Adam,
}

#[derive(Clone, Debug)]
struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
}

/// What one step measured.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepStats {
    /// Mean KL divergence over the queries that had a teacher.
    pub kl: f64,
    /// Queries that had a teacher.
    pub queries: usize,
}

impl LayerDistiller {
    /// Start from `init` (a PCA or random projection) and temperature `tau`,
    /// for a read with `cfg`'s sinks and window.
    pub fn new(init: &LayerProjection, input_dim: usize, cfg: &LsaConfig, tau: f32) -> Result<LayerDistiller, String> {
        let index_dim = cfg.index_dim;
        if init.wq.len() != input_dim * index_dim || init.wk.len() != input_dim * index_dim {
            return Err(format!("D-LSA: projections of {} and {} values for a {input_dim} x {index_dim} map", init.wq.len(), init.wk.len()));
        }
        if !(tau > 0.0 && tau.is_finite()) {
            return Err(format!("D-LSA: temperature {tau} must be positive"));
        }
        let n = 2 * input_dim * index_dim + 1;
        Ok(LayerDistiller { input_dim, index_dim, sink: cfg.sink, window: cfg.window, wq: init.wq.clone(), wk: init.wk.clone(), log_tau: tau.ln(), adam: Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0 } })
    }

    /// The trained projections.
    pub fn projection(&self) -> LayerProjection {
        LayerProjection { wq: self.wq.clone(), wk: self.wk.clone() }
    }

    /// The KL loss on `sample` at the current parameters, and its gradients
    /// `(dW_Q, dW_K, d ln τ)`.
    pub fn loss_and_grads(&self, sample: &LayerSample) -> (StepStats, Vec<f32>, Vec<f32>, f64) {
        let (din, d) = (self.input_dim, self.index_dim);
        let tau = (self.log_tau as f64).exp();
        let (kp, kn) = project(&sample.keys, &self.wk, din, d);
        let (qp, qn) = project(&sample.queries, &self.wq, din, d);
        let t = kn.len();
        let mut dk_unit = vec![0f64; t * d];
        let mut dq_unit = vec![0f64; qn.len() * d];
        let mut dlog_tau = 0f64;
        let mut stats = StepStats::default();
        let mut z: Vec<f64> = Vec::new();
        let mut a: Vec<f64> = Vec::new();
        for (s, target) in sample.targets.iter().enumerate() {
            if target.is_empty() {
                continue;
            }
            // The student normalises over all of R_i, the keys a sparse read
            // must find; the teacher's targets are a subset of it.
            let (lo, hi) = (self.sink, (sample.positions[s] as usize + 1).saturating_sub(self.window));
            if hi <= lo || target.iter().any(|&(j, _)| (j as usize) < lo || j as usize >= hi) {
                continue;
            }
            let qu = &qp[s * d..][..d];
            z.clear();
            z.extend((lo..hi).map(|j| qu.iter().zip(&kp[j * d..][..d]).map(|(x, y)| x * y).sum::<f64>() / tau));
            let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = z.iter().map(|v| (v - m).exp()).sum();
            let log_total = m + total.ln();
            a.clear();
            a.resize(hi - lo, 0.0);
            let mut kl = 0f64;
            for &(j, w) in target {
                let w = w as f64;
                a[j as usize - lo] += w;
                if w > 0.0 {
                    kl += w * (w.ln() - (z[j as usize - lo] - log_total));
                }
            }
            stats.kl += kl;
            stats.queries += 1;
            for (slot, j) in (lo..hi).enumerate() {
                let student = (z[slot] - log_total).exp();
                let gz = student - a[slot];
                if gz == 0.0 {
                    continue;
                }
                dlog_tau -= gz * z[slot];
                let gs = gz / tau;
                for c in 0..d {
                    dq_unit[s * d + c] += gs * kp[j * d + c];
                    dk_unit[j * d + c] += gs * qu[c];
                }
            }
        }
        let n = stats.queries.max(1) as f64;
        stats.kl /= n;
        let scale = 1.0 / n;
        let gk = back_project(&sample.keys, &kp, &kn, &dk_unit, din, d, scale);
        let gq = back_project(&sample.queries, &qp, &qn, &dq_unit, din, d, scale);
        (stats, gq, gk, dlog_tau * scale)
    }

    /// One Adam step on `sample`.
    pub fn step(&mut self, sample: &LayerSample, lr: f32) -> StepStats {
        let (stats, gq, gk, gt) = self.loss_and_grads(sample);
        if stats.queries == 0 {
            return stats;
        }
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        self.adam.t += 1;
        let t = self.adam.t as i32;
        let (c1, c2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
        let n = self.wq.len();
        let mut update = |i: usize, value: &mut f32, g: f32| {
            let m = &mut self.adam.m[i];
            let v = &mut self.adam.v[i];
            *m = b1 * *m + (1.0 - b1) * g;
            *v = b2 * *v + (1.0 - b2) * g * g;
            *value -= lr * (*m / c1) / ((*v / c2).sqrt() + eps);
        };
        for (i, (w, g)) in self.wq.iter_mut().zip(&gq).enumerate() {
            update(i, w, *g);
        }
        for (i, (w, g)) in self.wk.iter_mut().zip(&gk).enumerate() {
            update(n + i, w, *g);
        }
        update(2 * n, &mut self.log_tau, gt as f32);
        stats
    }
}

/// `x W` rows (`[rows, d]`, f64) and their norms.
fn project(x: &[f32], w: &[f32], din: usize, d: usize) -> (Vec<f64>, Vec<f64>) {
    let rows = x.len() / din;
    let mut out = vec![0f64; rows * d];
    for r in 0..rows {
        let xr = &x[r * din..][..din];
        let o = &mut out[r * d..][..d];
        for (i, &xi) in xr.iter().enumerate() {
            if xi == 0.0 {
                continue;
            }
            let wi = &w[i * d..][..d];
            for c in 0..d {
                o[c] += xi as f64 * wi[c] as f64;
            }
        }
    }
    let norms: Vec<f64> = out.chunks_exact(d).map(|r| r.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30)).collect();
    for (r, n) in out.chunks_exact_mut(d).zip(&norms) {
        for v in r {
            *v /= n;
        }
    }
    (out, norms)
}

/// The gradient of `W` from the gradient of the unit rows `u = unit(x W)`:
/// `du_raw = (du - u (u . du)) / |x W|`, then `dW = x^T du_raw`, times `scale`.
fn back_project(x: &[f32], unit: &[f64], norms: &[f64], du: &[f64], din: usize, d: usize, scale: f64) -> Vec<f32> {
    let rows = norms.len();
    let mut raw = vec![0f64; rows * d];
    for r in 0..rows {
        let (u, g) = (&unit[r * d..][..d], &du[r * d..][..d]);
        let along: f64 = u.iter().zip(g).map(|(a, b)| a * b).sum();
        if along == 0.0 && g.iter().all(|v| *v == 0.0) {
            continue;
        }
        for c in 0..d {
            raw[r * d + c] = (g[c] - u[c] * along) / norms[r];
        }
    }
    let mut dw = vec![0f64; din * d];
    for r in 0..rows {
        let g = &raw[r * d..][..d];
        if g.iter().all(|v| *v == 0.0) {
            continue;
        }
        for (i, &xi) in x[r * din..][..din].iter().enumerate() {
            let row = &mut dw[i * d..][..d];
            for c in 0..d {
                row[c] += xi as f64 * g[c];
            }
        }
    }
    dw.into_iter().map(|v| (v * scale) as f32).collect()
}

/// Every sparse layer of `init` trained together: one distiller per layer.
pub struct Distiller {
    base: Indexer,
    layers: Vec<Option<LayerDistiller>>,
}

impl Distiller {
    /// Start from `init` (its projections and configuration) at temperature
    /// `tau`.
    pub fn new(init: &Indexer, tau: f32) -> Result<Distiller, String> {
        init.validate()?;
        let din = init.geometry.input_dim();
        let layers = init.layers.iter().map(|l| l.as_ref().map(|p| LayerDistiller::new(p, din, &init.cfg, tau)).transpose()).collect::<Result<_, _>>()?;
        Ok(Distiller { base: init.clone(), layers })
    }

    /// One Adam step per layer on one document's samples (`samples[l]` for
    /// layer `l`, `None` for a dense layer), layers in parallel. Returns each
    /// layer's statistics.
    pub fn step(&mut self, samples: &[Option<LayerSample>], lr: f32) -> Vec<StepStats> {
        let mut out = vec![StepStats::default(); self.layers.len()];
        let mut work: Vec<(Option<LayerDistiller>, StepStats)> = self.layers.drain(..).map(|l| (l, StepStats::default())).collect();
        backend_cpu::par::chunks_mut(&mut work, 1, |l, slot| {
            if let (Some(d), Some(Some(sample))) = (slot[0].0.as_mut(), samples.get(l)) {
                slot[0].1 = d.step(sample, lr);
            }
        });
        for (l, (d, stats)) in work.into_iter().enumerate() {
            out[l] = stats;
            self.layers.push(d);
        }
        out
    }

    /// The indexer the training produced.
    pub fn indexer(&self) -> Indexer {
        let layers = self.layers.iter().map(|l| l.as_ref().map(LayerDistiller::projection)).collect();
        Indexer { cfg: self.base.cfg, geometry: self.base.geometry, layers, origin: format!("distilled({})", self.base.origin) }
    }

    /// Each layer's current temperature.
    pub fn temperatures(&self) -> Vec<Option<f32>> {
        self.layers.iter().map(|l| l.as_ref().map(|d| d.log_tau.exp())).collect()
    }
}

/// The share of the model's attention a selection reads: per sampled query,
/// the head-averaged dense attention weight (over every causal key) that
/// falls on the positions `selected[s]` lists. `q`, `k` and `positions` are
/// as for [`teacher_targets`]. Returns one share per query.
pub fn retained_mass(q: &[f32], k: &[f32], positions: &[u32], geometry: &IndexGeometry, selected: &[Vec<u32>]) -> Vec<f64> {
    let (nh, nkv, hd) = (geometry.n_heads, geometry.n_kv_heads, geometry.head_dim);
    let group = nh / nkv;
    let scale = 1.0 / (hd as f64).sqrt();
    positions
        .iter()
        .enumerate()
        .map(|(s, &p)| {
            let keys = p as usize + 1;
            let mut keep = vec![false; keys];
            for &j in &selected[s] {
                if (j as usize) < keys {
                    keep[j as usize] = true;
                }
            }
            let mut logits = vec![0f64; keys];
            let mut share = 0f64;
            for h in 0..nh {
                let qh = &q[s * nh * hd + h * hd..][..hd];
                let kv = (h / group) * hd;
                for (j, z) in logits.iter_mut().enumerate() {
                    *z = scale * super::dot(qh, &k[j * nkv * hd + kv..][..hd]);
                }
                let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = logits.iter().map(|z| (z - m).exp()).sum();
                let kept: f64 = logits.iter().zip(&keep).filter(|(_, k)| **k).map(|(z, _)| (z - m).exp()).sum();
                share += kept / total / nh as f64;
            }
            share
        })
        .collect()
}

/// One layer of one captured document, as the evaluation reads it.
pub struct CapturedLayer<'a> {
    /// `[T, input_dim]` index inputs of every key.
    pub key_inputs: &'a [f32],
    /// `[S, input_dim]` index inputs of the sampled queries.
    pub query_inputs: &'a [f32],
    /// `[S, n_heads * head_dim]` rotated queries.
    pub q: &'a [f32],
    /// `[T, n_kv_heads * head_dim]` rotated keys.
    pub k: &'a [f32],
}

/// Per sparse layer of `indexer` (layers in parallel), the share of the
/// model's dense attention its read keeps for each sampled query - the
/// host index and read over the indexer's own index vectors. Dense layers
/// give an empty list.
pub fn retention(indexer: &Indexer, layers: &[CapturedLayer], positions: &[u32]) -> Result<Vec<Vec<f64>>, String> {
    let mut out: Vec<Result<Vec<f64>, String>> = vec![Ok(Vec::new()); layers.len()];
    backend_cpu::par::chunks_mut(&mut out, 1, |l, slot| {
        if indexer.layers.get(l).is_none_or(Option::is_none) {
            return;
        }
        let c = &layers[l];
        slot[0] = (|| {
            let index = super::BucketIndex::build(indexer.cfg, &indexer.index_vectors(l, c.key_inputs, false)?)?;
            let sel = super::select(&index, &indexer.index_vectors(l, c.query_inputs, true)?, positions)?;
            let rows: Vec<Vec<u32>> = (0..positions.len()).map(|r| sel.row(r).iter().map(|&j| j as u32).collect()).collect();
            Ok(retained_mass(c.q, c.k, positions, &indexer.geometry, &rows))
        })();
    });
    out.into_iter().collect()
}
