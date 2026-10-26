// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How much of a pretrained model's residual stream a truncated Fourier basis
//! keeps, against the bases it must beat.
//!
//! The residual stream of Qwen3 after layer `l` is fitted (mean and
//! covariance) on training documents. On held-out documents it is then
//! replaced by its projection onto `m` directions of each basis,
//! `x' = mu + Q Qᵀ (x - mu)`, and the rest of the model runs on the
//! projection. Reported per basis and width: the fraction of centred energy
//! the projection keeps, and the increase in next-token loss it causes. The
//! bases are the lowest `m` real Fourier modes over the coordinates, the top
//! `m` principal components of the fit set, `m` random orthonormal directions
//! and the first `m` coordinates.

use std::time::Instant;

use appopts::Args;
use data::tokenizer::Tokenizer;
use model::score::ByteDoc;
use model::{Batch, Model, Shard};
use qwen3::model::Qwen;

use crate::{die, read_text};

/// `[m, d]` row-major: the orthonormal directions of one basis.
struct Basis {
    name: &'static str,
    m: usize,
    rows: Vec<f32>,
}

/// The two stages of the model split at layer `l`, on one batch shape.
struct Split {
    lower: Qwen,
    upper: Qwen,
    b: usize,
    t: usize,
    d: usize,
}

impl Split {
    fn new(weights: &str, layer: usize, b: u32, t: u32) -> Split {
        let (cfg, src) = qwen3::open_checkpoint(weights).unwrap_or_else(|e| die(&e));
        let n = cfg.n_layers as usize;
        if layer == 0 || layer >= n {
            die(&format!("--layer {layer} must split the {n} layers"));
        }
        let lower = Qwen::new_shard(cfg.clone(), b, t, &*src, false, Shard { start: 0, end: layer, embed: true, head: false, gpu_index: Shard::ANY_GPU });
        let upper = Qwen::new_shard(cfg.clone(), b, t, &*src, false, Shard { start: layer, end: n, embed: false, head: true, gpu_index: Shard::ANY_GPU });
        Split { lower, upper, b: b as usize, t: t as usize, d: cfg.d_model as usize }
    }

    /// One batch of documents as model inputs and next-token targets.
    fn batch(&self, docs: &[ByteDoc]) -> (Vec<u32>, Vec<u32>, usize) {
        let (mut x, mut y) = (vec![0u32; self.b * self.t], vec![model::IGNORE; self.b * self.t]);
        let mut supervised = 0;
        for (r, doc) in docs.iter().enumerate() {
            let n = doc.tokens.len();
            x[r * self.t..r * self.t + n].copy_from_slice(&doc.tokens);
            y[r * self.t..r * self.t + n - 1].copy_from_slice(&doc.tokens[1..]);
            supervised += n - 1;
        }
        (x, y, supervised)
    }

    /// The residual after the lower stage, `[b * t, d]`.
    fn residual(&self, x: &[u32], y: &[u32]) -> Vec<f32> {
        self.lower.set_batch(x, y);
        self.lower.run_forward();
        self.lower.read_out_res()
    }

    /// Summed next-token loss (nats) of the upper stage on residual `res`.
    fn loss(&self, res: &[f32], x: &[u32], y: &[u32], supervised: usize) -> f64 {
        Model::set_batch(&self.upper, Batch::Lm { tokens: x, targets: y });
        self.upper.write_in_res(res);
        Model::forward(&self.upper) as f64 * supervised as f64
    }
}

/// Positions of a batch that hold a document token.
fn live(docs: &[ByteDoc], t: usize) -> Vec<usize> {
    docs.iter().enumerate().flat_map(|(r, d)| (0..d.tokens.len()).map(move |i| r * t + i)).collect()
}

/// The first `count` documents of `path` within the byte limit, tokenized.
/// Selected before tokenizing: the training file is gigabytes.
fn documents(path: &str, tok: &dyn Tokenizer, eot: u32, max_bytes: usize, count: usize) -> Vec<ByteDoc> {
    let text = read_text(path);
    let kept: Vec<&str> = data::prepare::split_documents(&text).filter(|d| d.len() <= max_bytes).take(count).collect();
    model::score::byte_docs(&kept.join(data::bytebpe::END_OF_TEXT), tok, eot, max_bytes)
}

/// `n` orthonormal rows of length `d` from Gaussian draws (Gram-Schmidt).
fn random_orthonormal(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = data::rng::Rng::new(seed);
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    while rows.len() < n {
        let mut v: Vec<f64> = (0..d).map(|_| rng.next_gaussian()).collect();
        for r in &rows {
            let dot: f64 = v.iter().zip(r).map(|(a, b)| a * b).sum();
            v.iter_mut().zip(r).for_each(|(a, b)| *a -= dot * b);
        }
        let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if norm > 1e-6 {
            rows.push(v.into_iter().map(|a| a / norm).collect());
        }
    }
    rows.into_iter().flatten().map(|v| v as f32).collect()
}

pub fn run(a: &mut Args) {
    let weights = a.take_str("--weights").unwrap_or_else(|| die("qwen-spectrum needs --weights <Qwen3 checkpoint>"));
    let (tok, eot) = crate::qwen::tokenizer(a, &weights);
    let fit_text = a.take_str("--fit-text").unwrap_or_else(|| die("qwen-spectrum needs --fit-text <training documents>"));
    let text = a.take_str("--text").unwrap_or_else(|| die("qwen-spectrum needs --text <held-out documents>"));
    let layer = a.usize_or("--layer", 14);
    let widths: Vec<usize> = a.str_or("--widths", "64,128,256,512").split(',').map(|w| w.trim().parse().unwrap_or_else(|_| die(&format!("--widths entry {w:?} is not a number")))).collect();
    let fit_docs = a.usize_or("--fit-docs", 256);
    let docs = a.usize_or("--docs", 256);
    let max_bytes = a.usize_or("--max-doc-bytes", 2000);
    let batch = a.u32_or("--batch", 8);
    let seed = a.seed("spectral-lm qwen-spectrum");
    a.finish();

    let fit = documents(&fit_text, &tok, eot, max_bytes, fit_docs);
    let val = documents(&text, &tok, eot, max_bytes, docs);
    let longest = fit.iter().chain(&val).map(|d| d.tokens.len()).max().unwrap_or(2) as u32;
    let s = Split::new(&weights, layer, batch, longest);
    let d = s.d;
    let start = Instant::now();

    // Mean and covariance of the residual over every live fit position.
    let (mut mean, mut cov, mut n) = (vec![0.0f64; d], vec![0.0f64; d * d], 0usize);
    let mut fit_rows: Vec<Vec<f32>> = Vec::new();
    for chunk in fit.chunks(s.b) {
        let (x, y, _) = s.batch(chunk);
        let res = s.residual(&x, &y);
        for p in live(chunk, s.t) {
            fit_rows.push(res[p * d..(p + 1) * d].to_vec());
        }
    }
    for r in &fit_rows {
        mean.iter_mut().zip(r).for_each(|(m, v)| *m += *v as f64);
        n += 1;
    }
    mean.iter_mut().for_each(|m| *m /= n as f64);
    for r in &fit_rows {
        let c: Vec<f64> = r.iter().zip(&mean).map(|(v, m)| *v as f64 - m).collect();
        for i in 0..d {
            let ci = c[i];
            for j in i..d {
                cov[i * d + j] += ci * c[j];
            }
        }
    }
    for i in 0..d {
        for j in i..d {
            cov[i * d + j] /= n as f64;
            cov[j * d + i] = cov[i * d + j];
        }
    }
    drop(fit_rows);
    let (values, vectors) = model::eigen::symmetric_eigen(&cov, d);
    eprintln!("fitted mean and covariance on {n} positions, eigendecomposed in {:.0}s", start.elapsed().as_secs_f64());
    let top: Vec<usize> = (0..d).rev().collect::<Vec<_>>();
    let explained: f64 = values.iter().sum();

    let mut bases = Vec::new();
    for &m in &widths {
        if m > d {
            die(&format!("width {m} exceeds the residual width {d}"));
        }
        let s_basis = model::hostmath::real_fourier_basis(d, m);
        let fourier: Vec<f32> = (0..m).flat_map(|i| (0..d).map(move |j| (i, j))).map(|(i, j)| s_basis[j * m + i]).collect();
        let pca: Vec<f32> = top[..m].iter().flat_map(|&k| vectors[k].iter().map(|v| *v as f32)).collect();
        let coords: Vec<f32> = (0..m).flat_map(|i| (0..d).map(move |j| if i == j { 1.0 } else { 0.0 })).collect();
        bases.push(Basis { name: "fourier", m, rows: fourier });
        bases.push(Basis { name: "pca", m, rows: pca });
        bases.push(Basis { name: "random", m, rows: random_orthonormal(m, d, seed ^ m as u64) });
        bases.push(Basis { name: "coordinates", m, rows: coords });
    }

    // Held-out: baseline loss, then every basis's projected loss and energy.
    let mean32: Vec<f32> = mean.iter().map(|v| *v as f32).collect();
    let (mut base_nll, mut tokens) = (0.0f64, 0usize);
    let mut nll = vec![0.0f64; bases.len()];
    let mut kept = vec![0.0f64; bases.len()];
    let mut total_energy = 0.0f64;
    for chunk in val.chunks(s.b) {
        let (x, y, supervised) = s.batch(chunk);
        let res = s.residual(&x, &y);
        base_nll += s.loss(&res, &x, &y, supervised);
        tokens += supervised;
        let positions = live(chunk, s.t);
        let centred: Vec<f32> = positions.iter().flat_map(|&p| res[p * d..(p + 1) * d].iter().zip(&mean32).map(|(v, m)| v - m)).collect();
        total_energy += centred.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
        let rows = positions.len();
        for (bi, basis) in bases.iter().enumerate() {
            let coef = model::hostmath::linear_rows(&centred, &basis.rows, rows, d, basis.m);
            kept[bi] += coef.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
            let rows_t: Vec<f32> = (0..d).flat_map(|j| (0..basis.m).map(move |i| (i, j))).map(|(i, j)| basis.rows[i * d + j]).collect();
            let back = model::hostmath::linear_rows(&coef, &rows_t, rows, basis.m, d);
            let mut projected = res.clone();
            for (k, &p) in positions.iter().enumerate() {
                for j in 0..d {
                    projected[p * d + j] = mean32[j] + back[k * d + j];
                }
            }
            nll[bi] += s.loss(&projected, &x, &y, supervised);
        }
    }
    let rows: Vec<serde_json::Value> = bases
        .iter()
        .enumerate()
        .map(|(bi, b)| serde_json::json!({ "basis": b.name, "m": b.m, "energy": kept[bi] / total_energy, "delta_nll": (nll[bi] - base_nll) / tokens as f64 }))
        .collect();
    let pca_energy: Vec<f64> = widths.iter().map(|&m| top[..m].iter().map(|&k| values[k]).sum::<f64>() / explained).collect();
    println!(
        "{}",
        serde_json::json!({
            "event": "spectrum",
            "checkpoint": weights,
            "layer": layer,
            "residual_width": d,
            "fit_documents": fit.len(),
            "fit_positions": n,
            "documents": val.len(),
            "tokens": tokens,
            "base_nll": base_nll / tokens as f64,
            "fit_pca_energy": pca_energy,
            "seed": seed,
            "rows": rows,
            "seconds": start.elapsed().as_secs_f64(),
        })
    );
}
