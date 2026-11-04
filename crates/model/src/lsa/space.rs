// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Index vectors from a model's attention heads.
//!
//! The training-free index space is the position-free part of the attention
//! logit, collapsed over grouped-query heads:
//!
//! `sum_h q_h · k_{h/G} = concat_kv(sum_{h in group} q_h) · concat_kv(k_kv)`
//!
//! so one index query per position scores exactly what the attention heads
//! sum to, before the softmax. The rotated dimensions are dropped because
//! their contribution depends on the distance between query and key, which a
//! clustering of keys cannot know. Which dimensions rotate is the model's
//! fact, never assumed here.
//!
//! A learned (distilled) index applies a further `[Hkv * dh', index_dim]` map
//! to these vectors; the contract downstream - unit rows - is the same.

/// Where the rotary embedding sits in each head vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotaryLayout {
    /// No rotated dimensions (a position-free head, or an MLA latent).
    None,
    /// The first `n` dimensions of each head rotate.
    Prefix(usize),
    /// The last `n` dimensions of each head rotate.
    Suffix(usize),
}

impl RotaryLayout {
    /// The position-free channel range `[start, end)` of a `head_dim` head.
    fn free_range(self, head_dim: usize) -> Result<(usize, usize), String> {
        let rotated = match self {
            RotaryLayout::None => 0,
            RotaryLayout::Prefix(n) | RotaryLayout::Suffix(n) => n,
        };
        if rotated >= head_dim {
            return Err(format!("LSA index space: {rotated} rotated dims of a {head_dim}-dim head leave nothing position-free"));
        }
        Ok(match self {
            RotaryLayout::Prefix(n) => (n, head_dim),
            RotaryLayout::None | RotaryLayout::Suffix(_) => (0, head_dim - rotated),
        })
    }
}

/// Head layout of the attention the index is built over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadLayout {
    /// Query heads.
    pub n_heads: usize,
    /// Key/value heads; divides `n_heads`.
    pub n_kv_heads: usize,
    /// Channels per head.
    pub head_dim: usize,
    /// Rotated channels of each head.
    pub rotary: RotaryLayout,
}

impl HeadLayout {
    /// Width of an index vector in this space: `n_kv_heads` position-free heads.
    pub fn index_dim(&self) -> Result<usize, String> {
        let (start, end) = self.rotary.free_range(self.head_dim)?;
        Ok(self.n_kv_heads * (end - start))
    }

    fn check(&self) -> Result<(usize, usize), String> {
        if self.n_kv_heads == 0 || !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(format!("LSA index space: {} query heads do not group over {} KV heads", self.n_heads, self.n_kv_heads));
        }
        self.rotary.free_range(self.head_dim)
    }

    /// Index queries from pre-rotary queries `[rows, n_heads * head_dim]`:
    /// each KV group's query heads summed, groups concatenated, rows
    /// normalised. A zero row stays zero.
    pub fn index_queries(&self, q: &[f32]) -> Result<Vec<f32>, String> {
        let (start, end) = self.check()?;
        let row_width = self.n_heads * self.head_dim;
        if !q.len().is_multiple_of(row_width) {
            return Err(format!("LSA index queries: {} values is not whole rows of {row_width}", q.len()));
        }
        let group = self.n_heads / self.n_kv_heads;
        let free = end - start;
        let mut out = vec![0f32; q.len() / row_width * self.n_kv_heads * free];
        for (row, dst) in q.chunks_exact(row_width).zip(out.chunks_exact_mut(self.n_kv_heads * free)) {
            for h in 0..self.n_heads {
                let head = &row[h * self.head_dim..][start..end];
                for (o, x) in dst[(h / group) * free..][..free].iter_mut().zip(head) {
                    *o += *x;
                }
            }
            normalise(dst);
        }
        Ok(out)
    }

    /// Index keys from pre-rotary keys `[rows, n_kv_heads * head_dim]`: the
    /// position-free channels of every KV head concatenated, rows normalised.
    pub fn index_keys(&self, k: &[f32]) -> Result<Vec<f32>, String> {
        let (start, end) = self.check()?;
        let row_width = self.n_kv_heads * self.head_dim;
        if !k.len().is_multiple_of(row_width) {
            return Err(format!("LSA index keys: {} values is not whole rows of {row_width}", k.len()));
        }
        let free = end - start;
        let mut out = Vec::with_capacity(k.len() / row_width * self.n_kv_heads * free);
        for row in k.chunks_exact(row_width) {
            let begin = out.len();
            for h in 0..self.n_kv_heads {
                out.extend_from_slice(&row[h * self.head_dim..][start..end]);
            }
            normalise(&mut out[begin..]);
        }
        Ok(out)
    }
}

fn normalise(row: &mut [f32]) {
    let unit = crate::hostmath::l2_normalize(row);
    row.copy_from_slice(&unit);
}
