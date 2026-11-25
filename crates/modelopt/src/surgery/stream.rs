// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Residual-width reduction: a residual stream rotated into an orthonormal
//! basis and cut to its first `k` directions.
//!
//! Every linear that reads the stream does so through an RMSNorm, and an
//! orthogonal rotation `Q` of the stream preserves its norm. Folding each
//! norm's gain `γ` into the linears it feeds leaves norms without a gain,
//! which commute with `Q`. With the basis as rows `B = Qᵀ`, the rotated model
//! is `W diag(γ) Bᵀ` for every reader, `B W` for every writer (an output
//! projection, a bias) and `E Bᵀ` for every embedding table, and computes the
//! same function. Keeping the first `k` rows `B_k` drops the stream's
//! component outside their span; a norm then sees `k` coordinates of a vector
//! whose mean square it took over `d`, so its gain becomes `sqrt(d / k)`,
//! exact when the dropped component is zero. With the principal directions of
//! the stream's states as the basis, the dropped component is the smallest
//! one the calibration saw.
//!
//! A *shared* norm (one whose output enters a projection that also takes raw
//! stream vectors, so the two cannot get different weights) cannot shed its
//! gain: a per-channel gain does not commute with a general rotation. It
//! keeps the gain `γ'` that best matches `diag(γ)` in the rotated basis (the
//! least-squares fit on the calibration states, or the diagonal of
//! `B diag(γ) Bᵀ` without them), the readers that read only the norm take
//! `W diag(γ) Bᵀ diag(1/γ')` and stay exact, and the shared projection reads
//! the norm's output through the fitted gain: the one approximation even at
//! full width. The identity basis (a permutation, which does commute)
//! keeps that path exact too.

use std::collections::{BTreeMap, BTreeSet};

use model::hostmath::linear_rows;
use model::surgery::{Axis, Stream};
use serde::{Deserialize, Serialize};

use super::{Change, Checkpoint, Tensor};

/// How a basis was chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BasisKind {
    /// The covariance's eigenvectors, largest eigenvalue first.
    Pca,
    /// The stream's own coordinates, largest second moment first: a
    /// selection of channels, exact for every norm.
    Identity,
}

/// An orthonormal basis of a stream, most important direction first.
#[derive(Clone, Debug, PartialEq)]
pub struct Basis {
    pub kind: BasisKind,
    pub d: usize,
    /// `[d, d]` row-major, one direction per row.
    pub rows: Vec<f32>,
    /// The covariance's value along each row: how much of the stream it carries.
    pub energy: Vec<f64>,
}

/// The basis of a stream whose second-moment matrix is `cov` (`[d, d]`,
/// symmetric, f64).
pub fn basis_from_covariance(cov: &[f64], d: usize, kind: BasisKind) -> Result<Basis, String> {
    if cov.len() != d * d || d == 0 {
        return Err(format!("a covariance of {} values is not {d} x {d}", cov.len()));
    }
    if cov.iter().any(|v| !v.is_finite()) {
        return Err("the covariance holds a value that is not finite".into());
    }
    match kind {
        BasisKind::Pca => {
            let (values, vectors) = model::eigen::symmetric_eigen(cov, d);
            let rows = vectors.iter().rev().flat_map(|v| v.iter().map(|&x| x as f32)).collect();
            Ok(Basis { kind, d, rows, energy: values.into_iter().rev().collect() })
        }
        BasisKind::Identity => {
            let diag: Vec<f64> = (0..d).map(|i| cov[i * d + i]).collect();
            let mut order: Vec<usize> = (0..d).collect();
            order.sort_by(|&a, &b| diag[b].total_cmp(&diag[a]).then(a.cmp(&b)));
            let mut rows = vec![0f32; d * d];
            for (r, &c) in order.iter().enumerate() {
                rows[r * d + c] = 1.0;
            }
            Ok(Basis { kind, d, rows, energy: order.iter().map(|&c| diag[c]).collect() })
        }
    }
}

/// The calibration states of a shared norm's input, each scaled to unit mean
/// square (what the norm divides by), as a `[d, d]` second-moment matrix: what
/// fits the norm's gain in the rotated basis.
#[derive(Clone, Debug, PartialEq)]
pub struct SharedNormFit {
    pub norm: String,
    pub covariance: Vec<f64>,
}

fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x[r * cols + c];
        }
    }
    out
}

/// The gain a shared norm keeps along each of the `k` rows of `b` (`[k, d]`).
fn fitted_gain(b: &[f32], k: usize, d: usize, gain: &[f32], cov: Option<&[f64]>) -> Vec<f64> {
    (0..k)
        .map(|j| {
            let row = &b[j * d..(j + 1) * d];
            match cov {
                // (B C diag(γ) Bᵀ)_jj / (B C Bᵀ)_jj.
                Some(c) => {
                    let (mut num, mut den) = (0f64, 0f64);
                    for a in 0..d {
                        let m: f64 = (0..d).map(|x| f64::from(row[x]) * c[x * d + a]).sum();
                        num += m * f64::from(gain[a]) * f64::from(row[a]);
                        den += m * f64::from(row[a]);
                    }
                    num / den
                }
                None => row.iter().zip(gain).map(|(&q, &g)| f64::from(q) * f64::from(q) * f64::from(g)).sum(),
            }
        })
        .collect()
}

/// Rotate `stream` into `basis` and keep its first `k` directions: every
/// writer, reader and norm the stream declares, and its width keys. `fits`
/// gives the calibration states of shared norms (a shared norm without one
/// takes the calibration-free gain). Refused, touching nothing: a basis of
/// another width, `k` of 0 or past the width, a declared tensor of the wrong
/// width, a tensor declared twice, a reader's norm the stream does not
/// declare, and a shared gain that fits to zero along a kept direction.
pub fn rotate_stream(ck: &mut Checkpoint, stream: &Stream, basis: &Basis, k: usize, fits: &[SharedNormFit]) -> Result<Change, String> {
    let d = stream.width;
    let what = |e: String| format!("{}: {e}", stream.name);
    if basis.d != d || basis.rows.len() != d * d {
        return Err(what(format!("a basis {} wide for a stream {d} wide", basis.d)));
    }
    if k == 0 || k > d {
        return Err(what(format!("a width of {k} is not in 1..={d}")));
    }
    let mut seen = BTreeSet::new();
    let names = stream.writers.iter().map(|(n, _)| n).chain(stream.readers.iter().map(|r| &r.tensor)).chain(stream.norms.iter().map(|n| &n.tensor));
    for n in names {
        if !seen.insert(n.clone()) {
            return Err(what(format!("{n} is declared twice")));
        }
    }
    let b = &basis.rows[..k * d];
    let mut gains: BTreeMap<&str, Vec<f32>> = BTreeMap::new();
    for norm in &stream.norms {
        let g = ck.get(&norm.tensor).map_err(what)?;
        if g.data.len() != d {
            return Err(what(format!("norm {} holds {} gains", norm.tensor, g.data.len())));
        }
        gains.insert(&norm.tensor, g.data.clone());
    }
    // The gain each shared norm keeps, per kept direction.
    let mut fitted: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for norm in stream.norms.iter().filter(|n| n.shared) {
        let cov = fits.iter().find(|f| f.norm == norm.tensor).map(|f| f.covariance.as_slice());
        if cov.is_some_and(|c| c.len() != d * d) {
            return Err(what(format!("the fit of {} is not {d} x {d}", norm.tensor)));
        }
        let g = fitted_gain(b, k, d, &gains[norm.tensor.as_str()], cov);
        let scale = g.iter().fold(0f64, |m, v| m.max(v.abs()));
        if let Some(j) = g.iter().position(|v| !v.is_finite() || v.abs() <= scale * 1e-6) {
            return Err(what(format!("the gain of shared norm {} fits to {} along direction {j}", norm.tensor, g[j])));
        }
        fitted.insert(&norm.tensor, g);
    }
    let mut out: Vec<(String, Tensor)> = Vec::new();
    for r in &stream.readers {
        let w = ck.get(&r.tensor).map_err(what)?;
        let (rows, cols) = w.rows_cols();
        if cols != d || w.shape.len() != 2 {
            return Err(what(format!("reader {} is {:?}, not [_, {d}]", r.tensor, w.shape)));
        }
        let mut wg = w.data.clone();
        if let Some(norm) = &r.norm {
            let g = gains.get(norm.as_str()).ok_or_else(|| what(format!("reader {} names norm {norm}, which the stream does not declare", r.tensor)))?;
            for row in wg.chunks_mut(d) {
                row.iter_mut().zip(g).for_each(|(v, g)| *v *= g);
            }
        }
        let mut rotated = linear_rows(&wg, b, rows, d, k);
        if let Some(g) = r.norm.as_deref().and_then(|n| fitted.get(n)) {
            for row in rotated.chunks_mut(k) {
                row.iter_mut().zip(g).for_each(|(v, g)| *v = (f64::from(*v) / g) as f32);
            }
        }
        out.push((r.tensor.clone(), Tensor { shape: vec![rows, k], data: rotated }));
    }
    for (name, axis) in &stream.writers {
        let w = ck.get(name).map_err(what)?;
        let (rows, cols) = w.rows_cols();
        let t = match axis {
            Axis::Cols => {
                if cols != d || w.shape.len() != 2 {
                    return Err(what(format!("writer {name} is {:?}, not [_, {d}]", w.shape)));
                }
                Tensor { shape: vec![rows, k], data: linear_rows(&w.data, b, rows, d, k) }
            }
            Axis::Rows => {
                if rows != d {
                    return Err(what(format!("writer {name} is {:?}, not [{d}, ...]", w.shape)));
                }
                let rotated = transpose(&linear_rows(&transpose(&w.data, rows, cols), b, cols, d, k), cols, k);
                let mut shape = w.shape.clone();
                shape[0] = k;
                Tensor { shape, data: rotated }
            }
        };
        out.push((name.clone(), t));
    }
    let scale = (d as f64 / k as f64).sqrt();
    for norm in &stream.norms {
        let data = match fitted.get(norm.tensor.as_str()) {
            Some(g) => g.iter().map(|g| (g * scale) as f32).collect(),
            None => vec![scale as f32; k],
        };
        out.push((norm.tensor.clone(), Tensor { shape: vec![k], data }));
    }
    let (mut change, before) = Change::begin("rotate_stream", ck);
    for (name, t) in out {
        *ck.get_mut(&name)? = t;
    }
    for key in &stream.keys {
        change.set(ck, key, k.into())?;
    }
    change.targets.push(stream.name.clone());
    change.kept.insert(stream.name.clone(), (0..k).collect());
    Ok(change.finish(ck, before))
}
