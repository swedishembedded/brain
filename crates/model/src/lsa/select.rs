// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A query's read: which logical key positions it attends to.
//!
//! A query at position `p` scores every bucket of the cluster blocks closed
//! before its own (`p / cluster_block` blocks), picks the `top_p` best (a tie
//! to the lower bucket), and walks their members best bucket first, keeping
//! keys outside the sinks and the window until it holds `top_k`. The last
//! bucket is cut short. Then it adds the window `[p + 1 - window, p]` and the
//! sinks `[0, sink)` below the window.
//!
//! The window must reach back over the query's whole open block (a validated
//! config guarantees it), so every key at or before `p` is either in a bucket
//! the query can score or in its window.

use super::{check_unit_rows, dot, BucketIndex};

/// Per query, the complete set of key positions it reads.
///
/// Row `r` holds `lens[r]` positions in `ids[r * width..]`, then `-1`
/// padding: first the expanded long-range keys (best bucket first), then the
/// window in ascending order, then the sinks. Positions in a row are unique
/// and none exceeds the query's own position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    /// Row stride of `ids`.
    pub width: usize,
    /// Selected positions, `[rows, width]`, `-1` padded.
    pub ids: Vec<i32>,
    /// Positions each row holds.
    pub lens: Vec<u32>,
}

impl Selection {
    /// Queries in the selection.
    pub fn rows(&self) -> usize {
        self.lens.len()
    }

    /// The positions row `r` reads.
    pub fn row(&self, r: usize) -> &[i32] {
        &self.ids[r * self.width..][..self.lens[r] as usize]
    }
}

/// Select the keys each query reads from `index`.
///
/// `queries` is `[Tq, index_dim]` unit rows and `positions[r]` is query `r`'s
/// position, which must already be in the index (its own key appended), since
/// a query always reads itself through its window. Scores are the f64 dot
/// products of query and centroid, rounded to f32 as a device computes them.
pub fn select(index: &BucketIndex, queries: &[f32], positions: &[u32]) -> Result<Selection, String> {
    let cfg = *index.config();
    let dim = cfg.index_dim;
    check_unit_rows("LSA index queries", queries, dim)?;
    if queries.len() / dim != positions.len() {
        return Err(format!("LSA select: {} queries but {} positions", queries.len() / dim, positions.len()));
    }
    let stride = index.rows();
    let mut scores = vec![0f32; positions.len() * stride];
    for (r, row) in scores.chunks_exact_mut(stride.max(1)).enumerate().take(positions.len()) {
        let query = &queries[r * dim..][..dim];
        for (b, s) in row.iter_mut().enumerate() {
            *s = dot(query, &index.centroids()[b * dim..][..dim]) as f32;
        }
    }
    read_scores(index, &scores, stride, positions, index.closed_blocks())
}

/// Select from given centroid scores: `scores[r * stride + b]` is query `r`'s
/// score of bucket `b`, for at least every bucket the query may see.
///
/// Only the first `built` closed blocks are read from buckets; the keys of a
/// closed block past them are read through the window, which then reaches
/// back to `built * cluster_block` - how the device reads while a block's
/// index page is still being built.
///
/// The device read takes its scores from a kernel; given the same f32 scores
/// and `built` this is its oracle, position for position.
pub fn read_scores(index: &BucketIndex, scores: &[f32], stride: usize, positions: &[u32], built: usize) -> Result<Selection, String> {
    let cfg = *index.config();
    if let Some(&p) = positions.iter().find(|&&p| p as usize >= index.len()) {
        return Err(format!("LSA select: query position {p} is past the {} keys in the index", index.len()));
    }
    let built = built.min(index.closed_blocks());
    let width = cfg.selection_width() + if built < index.closed_blocks() { cfg.cluster_block } else { 0 };
    let mut ids = vec![-1i32; positions.len() * width];
    let mut lens = vec![0u32; positions.len()];
    let mut ranked: Vec<(f32, usize)> = Vec::new();
    for (r, &position) in positions.iter().enumerate() {
        let position = position as usize;
        let (top_k, top_p) = cfg.budget_at(position);
        let visible = (position / cfg.cluster_block).min(built) * cfg.buckets_per_block();
        if visible > stride || (visible > 0 && scores.len() < r * stride + visible) {
            return Err(format!("LSA select: row {r} sees {visible} buckets but the scores hold {stride} a row"));
        }

        ranked.clear();
        ranked.extend((0..visible).map(|b| (scores[r * stride + b], b)));
        // Best score first, a tie to the lower bucket; only the top `top_p`
        // need an order.
        let order = |a: &(f32, usize), b: &(f32, usize)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
        let picked = top_p.min(ranked.len());
        if picked > 0 && picked < ranked.len() {
            ranked.select_nth_unstable_by(picked - 1, order);
        }
        ranked[..picked].sort_unstable_by(order);

        let window_start = (position + 1).saturating_sub(cfg.window).min(built * cfg.cluster_block);
        let row = &mut ids[r * width..][..width];
        let mut n = 0usize;
        'expand: for &(_, bucket) in &ranked[..picked] {
            let members = &index.members()[bucket * cfg.bucket_cap..][..index.sizes()[bucket] as usize];
            for &key in members {
                if n == top_k {
                    break 'expand;
                }
                let key_pos = key as usize;
                if key_pos >= cfg.sink && key_pos < window_start {
                    row[n] = key;
                    n += 1;
                }
            }
        }
        for key in window_start..=position {
            row[n] = key as i32;
            n += 1;
        }
        for key in 0..cfg.sink.min(window_start) {
            row[n] = key as i32;
            n += 1;
        }
        lens[r] = n as u32;
    }
    Ok(Selection { width, ids, lens })
}
