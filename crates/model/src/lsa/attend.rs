// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Attention restricted to a [`Selection`]: the numerical oracle the device
//! sparse-attention kernels are gated against.
//!
//! Each query row reads exactly the positions its selection lists, with the
//! model's own q, k, v and scale, and one softmax over them. When every row
//! lists every causal position this *is* dense causal attention, which is the
//! anchor that says the sparse read adds no error of its own.

use super::Selection;

/// Shapes of the attention a selection is applied to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionShape {
    /// Query heads.
    pub n_heads: usize,
    /// Key/value heads; divides `n_heads`.
    pub n_kv_heads: usize,
    /// Channels per head.
    pub head_dim: usize,
}

/// Attention of each query over its selected keys.
///
/// `q` is `[rows, n_heads * head_dim]` (row `r` is the selection's row `r`),
/// `k` and `v` are `[T, n_kv_heads * head_dim]` indexed by logical position.
/// Scores are `scale * q · k`, accumulated in f64; returns `[rows, n_heads *
/// head_dim]`. A position outside `k` is an error, not a skipped key.
pub fn sparse_attention(q: &[f32], k: &[f32], v: &[f32], shape: AttentionShape, scale: f32, selection: &Selection) -> Result<Vec<f32>, String> {
    let AttentionShape { n_heads, n_kv_heads, head_dim } = shape;
    if n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
        return Err(format!("sparse attention: {n_heads} query heads do not group over {n_kv_heads} KV heads"));
    }
    let (q_row, kv_row) = (n_heads * head_dim, n_kv_heads * head_dim);
    if q.len() != selection.rows() * q_row {
        return Err(format!("sparse attention: {} query values for {} rows of {q_row}", q.len(), selection.rows()));
    }
    if k.len() != v.len() || !k.len().is_multiple_of(kv_row) {
        return Err(format!("sparse attention: k ({}) and v ({}) are not equal whole rows of {kv_row}", k.len(), v.len()));
    }
    let keys = k.len() / kv_row;
    let group = n_heads / n_kv_heads;
    let mut out = vec![0f32; q.len()];
    let mut weights: Vec<f64> = Vec::new();
    for r in 0..selection.rows() {
        let ids = selection.row(r);
        if let Some(bad) = ids.iter().find(|&&j| j < 0 || j as usize >= keys) {
            return Err(format!("sparse attention: row {r} selects position {bad} outside the {keys} cached keys"));
        }
        for h in 0..n_heads {
            let qh = &q[r * q_row + h * head_dim..][..head_dim];
            let kv_off = (h / group) * head_dim;
            weights.clear();
            weights.extend(ids.iter().map(|&j| scale as f64 * super::dot(qh, &k[j as usize * kv_row + kv_off..][..head_dim])));
            let m = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0f64;
            for w in weights.iter_mut() {
                *w = (*w - m).exp();
                total += *w;
            }
            let o = &mut out[r * q_row + h * head_dim..][..head_dim];
            for c in 0..head_dim {
                let acc: f64 = ids.iter().zip(&weights).map(|(&j, w)| w * v[j as usize * kv_row + kv_off + c] as f64).sum();
                o[c] = if total > 0.0 { (acc / total) as f32 } else { 0.0 };
            }
        }
    }
    Ok(out)
}

/// The selection of dense causal attention: row `r` reads `0..=positions[r]`.
pub fn dense_selection(positions: &[u32]) -> Selection {
    let width = positions.iter().map(|&p| p as usize + 1).max().unwrap_or(0);
    let mut ids = vec![-1i32; positions.len() * width];
    for (r, &p) in positions.iter().enumerate() {
        for j in 0..=p as usize {
            ids[r * width + j] = j as i32;
        }
    }
    Selection { width, ids, lens: positions.iter().map(|&p| p + 1).collect() }
}
