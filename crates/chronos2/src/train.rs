// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Differentiable Chronos-2 — host (CPU) forward+backward, gradcheck-gated.
//!
//! Chronos-2's inference path ([`crate::model`]) is a device-op forward with no
//! backward, and its attention uses the **unscaled** `_FULL` kernels that have no
//! GPU backward yet. Rather than author new gradient kernels, the trainable twin
//! is built **host-side**: a plain-`Vec<f32>` forward that records its
//! intermediates and an analytic backward, validated by finite differences on a
//! tiny config (the same discipline as the Kronos decoder's gradcheck). Training
//! is CPU-bound but correct; a GPU-speed Step-tape version can follow once the
//! unscaled-attention backward kernels exist.
//!
//! Built milestone by milestone, each gradcheck-gated:
//! - **M2 (this file, now):** the terminal path — final RMSNorm → the quantile
//!   head (a biased `ResidualBlock`) → rearrange → mean pinball loss. Exercises
//!   rmsnorm/matmul/bias/relu/residual + [`forecast::metrics::mean_pinball_grad`]
//!   backward, and the gradient w.r.t. the encoder output (`d_emb`) that the
//!   block stack (M3) will consume.
//! - M3: the encoder blocks (time attention + group degeneration + FFN).
//! - M4: full backbone + a gated fine-tune entry (LoRA / promotion gate).

use crate::config::Chronos2Config;
use std::collections::HashMap;

// ---- host math primitives (fwd + bwd), row-major, weights stored `[out, in]` ---

/// `out[m,n] = sum_k x[m,k] * w[n,k]` (weight is `[n, k]`, i.e. `[out, in]`).
fn matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += x[r * k + i] * w[c * k + i];
            }
            out[r * n + c] = acc;
        }
    }
    out
}

/// Backward of [`matmul`]: `dout[m,n]` → `dx[m,k]` and accumulate into `dw[n,k]`.
fn matmul_bwd(x: &[f32], w: &[f32], dout: &[f32], m: usize, k: usize, n: usize, dw: &mut [f32]) -> Vec<f32> {
    let mut dx = vec![0.0f32; m * k];
    for r in 0..m {
        for c in 0..n {
            let g = dout[r * n + c];
            for i in 0..k {
                dx[r * k + i] += g * w[c * k + i];
                dw[c * k + i] += g * x[r * k + i];
            }
        }
    }
    dx
}

/// Add a bias `b[n]` to each of `m` rows, in place.
fn bias_add(y: &mut [f32], b: &[f32], m: usize, n: usize) {
    for r in 0..m {
        for c in 0..n {
            y[r * n + c] += b[c];
        }
    }
}

/// Backward of [`bias_add`]: accumulate `db[c] += sum_r dout[r,c]`.
fn bias_bwd(dout: &[f32], m: usize, n: usize, db: &mut [f32]) {
    for r in 0..m {
        for c in 0..n {
            db[c] += dout[r * n + c];
        }
    }
}

/// RMSNorm per row with per-feature gain `g[d]`: `y[r,i] = x[r,i]/rms(x[r]) * g[i]`.
/// Returns `(y, inv_rms[rows])` (the reciprocal norms, kept for the backward).
fn rmsnorm(x: &[f32], g: &[f32], rows: usize, d: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0f32; rows * d];
    let mut inv = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &x[r * d..r * d + d];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let iv = 1.0 / (ms + eps).sqrt();
        inv[r] = iv;
        for i in 0..d {
            y[r * d + i] = row[i] * iv * g[i];
        }
    }
    (y, inv)
}

/// Backward of [`rmsnorm`]: `dy` → `dx`, accumulate `dg`. `inv` from the forward.
fn rmsnorm_bwd(x: &[f32], g: &[f32], inv: &[f32], dy: &[f32], rows: usize, d: usize, dg: &mut [f32]) -> Vec<f32> {
    let mut dx = vec![0.0f32; rows * d];
    for r in 0..rows {
        let iv = inv[r];
        // s = Σ_i dy_i * g_i * x_i  (the coupling term through the shared norm)
        let mut s = 0.0f32;
        for i in 0..d {
            let xi = x[r * d + i];
            s += dy[r * d + i] * g[i] * xi;
            dg[i] += dy[r * d + i] * xi * iv;
        }
        let iv3 = iv * iv * iv / d as f32;
        for i in 0..d {
            let xi = x[r * d + i];
            dx[r * d + i] = iv * g[i] * dy[r * d + i] - iv3 * xi * s;
        }
    }
    dx
}

// ---- the biased ResidualBlock: output(relu(hidden(x))) + residual(x) ----------

/// Intermediates a [`residual_block`] backward needs.
struct ResidualCache {
    x: Vec<f32>,
    hid: Vec<f32>,      // pre-ReLU
    hid_relu: Vec<f32>, // post-ReLU
}

/// Names of the three linear layers' `weight`/`bias` under `prefix`.
fn rb_names(prefix: &str) -> [String; 6] {
    [
        format!("{prefix}.hidden_layer.weight"),
        format!("{prefix}.hidden_layer.bias"),
        format!("{prefix}.output_layer.weight"),
        format!("{prefix}.output_layer.bias"),
        format!("{prefix}.residual_layer.weight"),
        format!("{prefix}.residual_layer.bias"),
    ]
}

/// Forward: `[rows, in_dim] -> [rows, out_dim]`, mirroring `model::residual_block`.
fn residual_block(
    w: &HashMap<String, Vec<f32>>,
    prefix: &str,
    x: &[f32],
    rows: usize,
    in_dim: usize,
    h: usize,
    out_dim: usize,
) -> (Vec<f32>, ResidualCache) {
    let n = rb_names(prefix);
    let mut hid = matmul(x, &w[&n[0]], rows, in_dim, h);
    bias_add(&mut hid, &w[&n[1]], rows, h);
    let hid_relu: Vec<f32> = hid.iter().map(|&v| v.max(0.0)).collect();

    let mut o1 = matmul(&hid_relu, &w[&n[2]], rows, h, out_dim);
    bias_add(&mut o1, &w[&n[3]], rows, out_dim);

    let mut res = matmul(x, &w[&n[4]], rows, in_dim, out_dim);
    bias_add(&mut res, &w[&n[5]], rows, out_dim);

    for i in 0..rows * out_dim {
        res[i] += o1[i]; // y = o1 + res
    }
    (res, ResidualCache { x: x.to_vec(), hid, hid_relu })
}

/// Backward of [`residual_block`]: `dy` → `dx`, accumulate weight/bias grads.
#[allow(clippy::too_many_arguments)]
fn residual_block_bwd(
    w: &HashMap<String, Vec<f32>>,
    g: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    cache: &ResidualCache,
    dy: &[f32],
    rows: usize,
    in_dim: usize,
    h: usize,
    out_dim: usize,
) -> Vec<f32> {
    let n = rb_names(prefix);
    // y = o1 + res  →  do1 = dres = dy
    bias_bwd(dy, rows, out_dim, g.get_mut(&n[5]).unwrap());
    let mut dx = matmul_bwd(&cache.x, &w[&n[4]], dy, rows, in_dim, out_dim, g.get_mut(&n[4]).unwrap());

    bias_bwd(dy, rows, out_dim, g.get_mut(&n[3]).unwrap());
    let d_hid_relu = matmul_bwd(&cache.hid_relu, &w[&n[2]], dy, rows, h, out_dim, g.get_mut(&n[2]).unwrap());

    // ReLU
    let mut d_hid = vec![0.0f32; rows * h];
    for i in 0..rows * h {
        d_hid[i] = if cache.hid[i] > 0.0 { d_hid_relu[i] } else { 0.0 };
    }
    bias_bwd(&d_hid, rows, h, g.get_mut(&n[1]).unwrap());
    let dx_hidden = matmul_bwd(&cache.x, &w[&n[0]], &d_hid, rows, in_dim, h, g.get_mut(&n[0]).unwrap());
    for i in 0..rows * in_dim {
        dx[i] += dx_hidden[i];
    }
    dx
}

// ---- the differentiable head path (M2) ----------------------------------------

/// The trainable Chronos-2 (host). Holds a config and a name→weights map; the
/// grad map mirrors the weight names. M2 exposes the terminal head path.
pub struct Chronos2Train {
    pub cfg: Chronos2Config,
    pub w: HashMap<String, Vec<f32>>,
}

/// Everything the head backward needs from the forward.
pub struct HeadCache {
    emb: Vec<f32>,
    inv: Vec<f32>,     // final-norm reciprocal norms
    normed: Vec<f32>,  // [s, d]
    rb: ResidualCache, // the head ResidualBlock's cache
    quantiles: Vec<f32>, // [H, Q]
    s: usize,
    n_out: usize,
}

impl Chronos2Train {
    /// Zeroed grad accumulator with an entry per weight.
    pub fn zero_grads(&self) -> HashMap<String, Vec<f32>> {
        self.w.iter().map(|(k, v)| (k.clone(), vec![0.0f32; v.len()])).collect()
    }

    /// Forward of the head path: encoder output `emb` `[s, d]` → final RMSNorm →
    /// quantile head on the trailing `n_out` patch tokens → rearranged
    /// `quantiles[H, Q]` (`H = n_out*patch`, quantile-minor) → mean pinball loss
    /// vs the standardized `target[H]` at `levels[Q]`. Returns `(loss, cache)`.
    pub fn head_forward(&self, emb: &[f32], n_out: usize, target: &[f32], levels: &[f32]) -> (f32, HeadCache) {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let patch = cfg.output_patch_size;
        let q = cfg.num_quantiles;
        let head_out = cfg.head_out_dim();
        let s = emb.len() / d;
        let eps = cfg.layer_norm_epsilon;

        let (normed, inv) = rmsnorm(emb, &self.w["encoder.final_layer_norm.weight"], s, d, eps);
        let head_in = normed[(s - n_out) * d..].to_vec();
        let (qp, rb) = residual_block(&self.w, "output_patch_embedding", &head_in, n_out, d, cfg.d_ff, head_out);

        // rearrange [n_out, head_out=(q*patch)] -> quantiles[H=n_out*patch, Q]
        let hlen = n_out * patch;
        let mut quantiles = vec![0.0f32; hlen * q];
        for nn in 0..n_out {
            for qi in 0..q {
                for pp in 0..patch {
                    quantiles[(nn * patch + pp) * q + qi] = qp[nn * head_out + qi * patch + pp];
                }
            }
        }
        let loss = forecast::metrics::mean_pinball(&quantiles, levels, target);
        (loss, HeadCache { emb: emb.to_vec(), inv, normed, rb, quantiles, s, n_out })
    }

    /// Backward of [`head_forward`]: accumulate head + final-norm grads into `g`
    /// and return `d_emb` `[s, d]` (the gradient the block stack consumes in M3).
    pub fn head_backward(&self, cache: &HeadCache, target: &[f32], levels: &[f32], g: &mut HashMap<String, Vec<f32>>) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let patch = cfg.output_patch_size;
        let q = cfg.num_quantiles;
        let head_out = cfg.head_out_dim();
        let (s, n_out) = (cache.s, cache.n_out);

        let d_quant = forecast::metrics::mean_pinball_grad(&cache.quantiles, levels, target);
        // rearrange the gradient back to d_qp[n_out, head_out]
        let mut d_qp = vec![0.0f32; n_out * head_out];
        for nn in 0..n_out {
            for qi in 0..q {
                for pp in 0..patch {
                    d_qp[nn * head_out + qi * patch + pp] = d_quant[(nn * patch + pp) * q + qi];
                }
            }
        }
        let d_head_in =
            residual_block_bwd(&self.w, g, "output_patch_embedding", &cache.rb, &d_qp, n_out, d, cfg.d_ff, head_out);

        // scatter d_head_in into the trailing rows of d_normed[s, d]
        let mut d_normed = vec![0.0f32; s * d];
        d_normed[(s - n_out) * d..].copy_from_slice(&d_head_in);

        rmsnorm_bwd(
            &cache.emb,
            &self.w["encoder.final_layer_norm.weight"],
            &cache.inv,
            &d_normed,
            s,
            d,
            g.get_mut("encoder.final_layer_norm.weight").unwrap(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic small-magnitude weights/inputs for a gradcheck.
    fn rng(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4
    }

    fn head_params(cfg: &Chronos2Config, seed: &mut u64) -> HashMap<String, Vec<f32>> {
        let d = cfg.d_model;
        let ff = cfg.d_ff;
        let ho = cfg.head_out_dim();
        let mut w = HashMap::new();
        let mut fill = |name: &str, len: usize, w: &mut HashMap<String, Vec<f32>>, seed: &mut u64| {
            w.insert(name.to_string(), (0..len).map(|_| rng(seed)).collect());
        };
        fill("encoder.final_layer_norm.weight", d, &mut w, seed);
        // final-norm gain near 1 (its default), plus small jitter.
        for v in w.get_mut("encoder.final_layer_norm.weight").unwrap() {
            *v += 1.0;
        }
        for (nm, len) in [
            ("output_patch_embedding.hidden_layer.weight", ff * d),
            ("output_patch_embedding.hidden_layer.bias", ff),
            ("output_patch_embedding.output_layer.weight", ho * ff),
            ("output_patch_embedding.output_layer.bias", ho),
            ("output_patch_embedding.residual_layer.weight", ho * d),
            ("output_patch_embedding.residual_layer.bias", ho),
        ] {
            fill(nm, len, &mut w, seed);
        }
        w
    }

    #[test]
    fn head_path_gradcheck() {
        let cfg = Chronos2Config::tiny();
        let mut seed = 0x51ED_u64;
        let w = head_params(&cfg, &mut seed);
        let model = Chronos2Train { cfg: cfg.clone(), w };

        let d = cfg.d_model;
        let patch = cfg.output_patch_size;
        let q = cfg.num_quantiles;
        let (s, n_out) = (5usize, 2usize);
        let h = n_out * patch;
        let emb: Vec<f32> = (0..s * d).map(|_| rng(&mut seed)).collect();
        let target: Vec<f32> = (0..h).map(|_| rng(&mut seed)).collect();
        let levels: Vec<f32> = (0..q).map(|i| (i as f32 + 0.5) / q as f32).collect();

        // analytic grads
        let (_l0, cache) = model.head_forward(&emb, n_out, &target, &levels);
        let mut g = model.zero_grads();
        let d_emb = model.head_backward(&cache, &target, &levels, &mut g);

        let loss = |m: &Chronos2Train, e: &[f32]| m.head_forward(e, n_out, &target, &levels).0;
        let eps = 4e-3f32;
        let mut checked = 0usize;

        // (a) weight grads: sample a few entries per tensor via directional FD.
        for name in model.w.keys() {
            let len = model.w[name].len();
            for &idx in &[0usize, len / 3, len / 2, len - 1] {
                let mut mp = Chronos2Train { cfg: cfg.clone(), w: model.w.clone() };
                mp.w.get_mut(name).unwrap()[idx] += eps;
                let lp = loss(&mp, &emb);
                let mut mm = Chronos2Train { cfg: cfg.clone(), w: model.w.clone() };
                mm.w.get_mut(name).unwrap()[idx] -= eps;
                let lm = loss(&mm, &emb);
                let numeric = (lp - lm) / (2.0 * eps);
                let analytic = g[name][idx];
                assert!(
                    (numeric - analytic).abs() <= 4e-3 + 8e-2 * analytic.abs().max(numeric.abs()),
                    "grad mismatch {name}[{idx}]: analytic {analytic} vs numeric {numeric}"
                );
                checked += 1;
            }
        }

        // (b) input grad d_emb: a few entries.
        for &idx in &[0usize, s * d / 2, s * d - 1] {
            let mut ep = emb.clone();
            ep[idx] += eps;
            let mut em = emb.clone();
            em[idx] -= eps;
            let numeric = (loss(&model, &ep) - loss(&model, &em)) / (2.0 * eps);
            let analytic = d_emb[idx];
            assert!(
                (numeric - analytic).abs() <= 4e-3 + 8e-2 * analytic.abs().max(numeric.abs()),
                "d_emb[{idx}]: analytic {analytic} vs numeric {numeric}"
            );
            checked += 1;
        }
        assert!(checked > 20, "expected a broad gradcheck, only {checked} params");
    }
}
