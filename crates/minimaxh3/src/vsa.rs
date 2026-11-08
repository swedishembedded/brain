// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Video sparse attention (VSA) over the DiT's packed sequence: the tile
//! geometry and the selection rule.
//!
//! The method is FastVideo's VSA for MiniMax-H3's packed mixed-modality
//! attention (`fastvideo/attention/backends/video_sparse_attn_h3.py`,
//! Apache-2.0), which a trained `to_gate_compress` per block was distilled
//! against. Rows are permuted into tiles of [`TILE_ROWS`]:
//!
//! - every dense segment ([`Segment::Dense`]: the text, keyframe and image
//!   references, every soundtrack and the target audio) is cut into chunks of
//!   at most [`TILE_ROWS`] rows that never straddle a segment, and all of them
//!   lead, wherever the segment sits in the packed sequence;
//! - each video region ([`Segment::Video`]: a reference video, the target
//!   video last) follows as its own run of [`TILE_SHAPE`] tiles in
//!   `(t, h, w)` tile order, a tile at the grid's edge holding fewer rows.
//!
//! Every tile's live rows come first in its slots. Per head, a query tile
//! scores every key tile by the dot product of their mean (normed, rotated)
//! query and key, over `sqrt(head_dim)`. A prefix query tile attends every
//! key tile; a video query tile attends every prefix tile and the
//! [`keep`] best key tiles of each video region by that score ("exempt"
//! prefix keys, the reference's default). The block output adds to that
//! fine attention a coarse one, `softmax(scores) · mean(v)` over all tiles,
//! broadcast to each row of the query tile and scaled per row and channel by
//! the block's gate projection.
//!
//! Swedish Embedded AB implements sparse attention for long-sequence video
//! diffusion transformers for its clients. If your team needs expertise in
//! cutting the attention cost of video generation on its own hardware then
//! you can procure our services by sending an email to
//! info@swedishembedded.com.

use crate::pipeline::Segment;

/// `(t, h, w)` rows of one video tile.
pub const TILE_SHAPE: [u32; 3] = [4, 4, 4];
/// Rows (slots) of every tile.
pub const TILE_ROWS: u32 = TILE_SHAPE[0] * TILE_SHAPE[1] * TILE_SHAPE[2];
/// The slot value of a padded slot.
pub const PAD: u32 = u32::MAX;

/// One packed sequence's tiles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileGeometry {
    /// The packed row in each slot, [`TILE_ROWS`] slots per tile, tile-major;
    /// [`PAD`] past a tile's live rows.
    pub slots: Vec<u32>,
    /// Live rows of each tile.
    pub live: Vec<u32>,
    /// How many leading tiles hold dense segments.
    pub prefix_tiles: u32,
    /// The `[start, end)` tiles of each video region, in packed order.
    pub video_spans: Vec<(u32, u32)>,
}

impl TileGeometry {
    pub fn new(segments: &[Segment]) -> TileGeometry {
        let mut prefix: Vec<(u32, u32)> = Vec::new();
        let mut videos: Vec<(u32, [u32; 3])> = Vec::new();
        let mut cursor = 0u32;
        for segment in segments {
            match *segment {
                Segment::Dense { rows } => {
                    let mut start = cursor;
                    while start < cursor + rows {
                        let n = TILE_ROWS.min(cursor + rows - start);
                        prefix.push((start, n));
                        start += n;
                    }
                }
                Segment::Video { grid } => videos.push((cursor, grid)),
            }
            cursor += segment.rows();
        }
        let mut geometry = TileGeometry { slots: Vec::new(), live: Vec::new(), prefix_tiles: prefix.len() as u32, video_spans: Vec::with_capacity(videos.len()) };
        for (start, n) in prefix {
            geometry.push_tile((start..start + n).collect());
        }
        let [ts, hs, ws] = TILE_SHAPE;
        for (start, [t, h, w]) in videos {
            let first = geometry.live.len() as u32;
            for t0 in (0..t).step_by(ts as usize) {
                for h0 in (0..h).step_by(hs as usize) {
                    for w0 in (0..w).step_by(ws as usize) {
                        let rows = (t0..(t0 + ts).min(t)).flat_map(|tt| (h0..(h0 + hs).min(h)).flat_map(move |hh| (w0..(w0 + ws).min(w)).map(move |ww| start + (tt * h + hh) * w + ww)));
                        geometry.push_tile(rows.collect());
                    }
                }
            }
            geometry.video_spans.push((first, geometry.live.len() as u32));
        }
        geometry
    }

    fn push_tile(&mut self, rows: Vec<u32>) {
        self.live.push(rows.len() as u32);
        self.slots.extend(rows.iter().copied().chain(std::iter::repeat_n(PAD, (TILE_ROWS as usize) - rows.len())));
    }

    pub fn tiles(&self) -> u32 {
        self.live.len() as u32
    }
}

/// Key tiles a video query tile keeps out of a region of `tiles` at
/// `sparsity`: `ceil((1 - sparsity) * tiles)`, at least one and at most all.
pub fn keep(sparsity: f32, tiles: u32) -> u32 {
    (((1.0 - sparsity as f64) * tiles as f64).ceil() as u32).clamp(1, tiles)
}

/// Each query tile's selected key tiles, per head: `[heads][tiles]` lists in
/// ascending tile order. `scores` is `[heads, tiles, tiles]`; `sparsity` is
/// the target video's and `reference_sparsity` every reference video's.
pub fn select(geometry: &TileGeometry, scores: &[f32], heads: u32, (sparsity, reference_sparsity): (f32, f32)) -> Vec<Vec<Vec<u32>>> {
    let n = geometry.tiles() as usize;
    let spans = &geometry.video_spans;
    (0..heads as usize)
        .map(|h| {
            (0..n)
                .map(|q| {
                    if (q as u32) < geometry.prefix_tiles {
                        return (0..n as u32).collect();
                    }
                    let row = &scores[(h * n + q) * n..][..n];
                    let mut picked: Vec<u32> = (0..geometry.prefix_tiles).collect();
                    for (i, &(start, end)) in spans.iter().enumerate() {
                        let s = if i + 1 == spans.len() { sparsity } else { reference_sparsity };
                        let mut region: Vec<u32> = (start..end).collect();
                        // Highest score first; ties to the lower tile, as a stable top-k.
                        region.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]).then(a.cmp(&b)));
                        picked.extend(&region[..keep(s, end - start) as usize]);
                    }
                    picked.sort_unstable();
                    picked
                })
                .collect()
        })
        .collect()
}

/// Per head, the mean over each tile's live rows of `x` (`[rows, heads *
/// head_dim]`, packed order): `[heads, tiles, head_dim]`.
fn pool(geometry: &TileGeometry, x: &[f32], heads: usize, d: usize) -> Vec<f32> {
    let n = geometry.tiles() as usize;
    let mut out = vec![0.0f32; heads * n * d];
    for (t, &live) in geometry.live.iter().enumerate() {
        for &row in &geometry.slots[t * TILE_ROWS as usize..][..live as usize] {
            for h in 0..heads {
                let src = &x[(row as usize * heads + h) * d..][..d];
                out[(h * n + t) * d..][..d].iter_mut().zip(src).for_each(|(o, v)| *o += v);
            }
        }
        for h in 0..heads {
            out[(h * n + t) * d..][..d].iter_mut().for_each(|o| *o /= live as f32);
        }
    }
    out
}

/// The block's VSA self-attention on the host: the oracle a device
/// implementation is checked against. `q`, `k`, `v` are `[rows, heads *
/// head_dim]` in packed order, `q`/`k` normed and rotated; `gate` is the
/// block's gate projection of its input, same shape, `None` for a block
/// without one. Returns the attention output, same shape, packed order.
pub fn attention_host(geometry: &TileGeometry, (q, k, v): (&[f32], &[f32], &[f32]), gate: Option<&[f32]>, (heads, d): (usize, usize), sparsities: (f32, f32)) -> Vec<f32> {
    let n = geometry.tiles() as usize;
    let scale = 1.0 / (d as f32).sqrt();
    let (qp, kp) = (pool(geometry, q, heads, d), pool(geometry, k, heads, d));
    let mut scores = vec![0.0f32; heads * n * n];
    for h in 0..heads {
        for i in 0..n {
            for j in 0..n {
                scores[(h * n + i) * n + j] = qp[(h * n + i) * d..][..d].iter().zip(&kp[(h * n + j) * d..][..d]).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
        }
    }
    let selected = select(geometry, &scores, heads as u32, sparsities);
    let vp = gate.map(|_| pool(geometry, v, heads, d));
    let mut out = vec![0.0f32; q.len()];
    let tile_rows = |t: usize| &geometry.slots[t * TILE_ROWS as usize..][..geometry.live[t] as usize];
    for h in 0..heads {
        for qt in 0..n {
            let keys: Vec<u32> = selected[h][qt].iter().flat_map(|&kt| tile_rows(kt as usize).iter().copied()).collect();
            let coarse = vp.as_ref().map(|vp| {
                let row = &scores[(h * n + qt) * n..][..n];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let w: Vec<f32> = row.iter().map(|s| (s - max).exp()).collect();
                let total: f32 = w.iter().sum();
                (0..d).map(|c| (0..n).map(|j| w[j] * vp[(h * n + j) * d + c]).sum::<f32>() / total).collect::<Vec<f32>>()
            });
            for &r in tile_rows(qt) {
                let qr = &q[(r as usize * heads + h) * d..][..d];
                let logits: Vec<f32> = keys.iter().map(|&kr| qr.iter().zip(&k[(kr as usize * heads + h) * d..][..d]).map(|(a, b)| a * b).sum::<f32>() * scale).collect();
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let w: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
                let total: f32 = w.iter().sum();
                let o = &mut out[(r as usize * heads + h) * d..][..d];
                for (&kr, wk) in keys.iter().zip(&w) {
                    o.iter_mut().zip(&v[(kr as usize * heads + h) * d..][..d]).for_each(|(o, vv)| *o += wk / total * vv);
                }
                if let (Some(c), Some(g)) = (&coarse, gate) {
                    o.iter_mut().zip(c).zip(&g[(r as usize * heads + h) * d..][..d]).for_each(|((o, c), g)| *o += c * g);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense bidirectional attention over `[rows, heads * d]`, the
    /// independent check of [`attention_host`] with every tile kept.
    fn dense(q: &[f32], k: &[f32], v: &[f32], heads: usize, d: usize, allowed: &dyn Fn(usize, usize, usize) -> bool) -> Vec<f32> {
        let rows = q.len() / (heads * d);
        let mut out = vec![0.0; q.len()];
        for h in 0..heads {
            for i in 0..rows {
                let at = |r: usize, x: &[f32]| x[(r * heads + h) * d..][..d].to_vec();
                let logits: Vec<(usize, f32)> = (0..rows).filter(|&j| allowed(h, i, j)).map(|j| (j, at(i, q).iter().zip(at(j, k)).map(|(a, b)| a * b).sum::<f32>() / (d as f32).sqrt())).collect();
                let max = logits.iter().map(|l| l.1).fold(f32::NEG_INFINITY, f32::max);
                let total: f32 = logits.iter().map(|l| (l.1 - max).exp()).sum();
                for (j, l) in logits {
                    let w = (l - max).exp() / total;
                    out[(i * heads + h) * d..][..d].iter_mut().zip(at(j, v)).for_each(|(o, vv)| *o += w * vv);
                }
            }
        }
        out
    }

    fn close(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    fn qkv(rows: usize, width: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rng = data::rng::Lcg::new(seed);
        (rng.vec_scaled(rows * width, 1.0), rng.vec_scaled(rows * width, 1.0), rng.vec_scaled(rows * width, 1.0))
    }

    /// At zero sparsity and without a gate, VSA is dense attention: every
    /// query keeps every tile, padded slots never take part.
    #[test]
    fn with_every_tile_kept_and_no_gate_it_is_dense_attention() {
        let segments = [Segment::Dense { rows: 11 }, Segment::Video { grid: [2, 3, 5] }, Segment::Dense { rows: 7 }, Segment::Video { grid: [5, 2, 6] }];
        let g = TileGeometry::new(&segments);
        let rows = segments.iter().map(Segment::rows).sum::<u32>() as usize;
        let (heads, d) = (2, 8);
        let (q, k, v) = qkv(rows, heads * d, 5);
        let got = attention_host(&g, (&q, &k, &v), None, (heads, d), (0.0, 0.0));
        assert!(close(&got, &dense(&q, &k, &v, heads, d, &|_, _, _| true)) < 1e-5);
    }

    /// A video query attends only the prefix and its own selection of each
    /// video region; a prefix query attends everything; and the gate adds
    /// the coarse attention over tile means, scaled per row and channel.
    #[test]
    fn video_queries_read_only_their_selection_and_the_gate_adds_the_coarse_branch() {
        let segments = [Segment::Dense { rows: 9 }, Segment::Video { grid: [8, 4, 4] }];
        let g = TileGeometry::new(&segments);
        let rows = segments.iter().map(Segment::rows).sum::<u32>() as usize;
        let (heads, d) = (2, 4);
        let (q, k, v) = qkv(rows, heads * d, 8);
        let sparsity = (0.5, 0.5);

        let (qp, kp) = (pool(&g, &q, heads, d), pool(&g, &k, heads, d));
        let n = g.tiles() as usize;
        let scores: Vec<f32> = (0..heads * n * n).map(|i| {
            let (h, qt, kt) = (i / (n * n), (i / n) % n, i % n);
            qp[(h * n + qt) * d..][..d].iter().zip(&kp[(h * n + kt) * d..][..d]).map(|(a, b)| a * b).sum::<f32>() / (d as f32).sqrt()
        }).collect();
        let sel = select(&g, &scores, heads as u32, sparsity);
        assert_eq!(sel[0][1].len(), 1 + 1, "one prefix tile and one of the two video tiles");
        let tile_of: Vec<usize> = (0..rows).map(|r| g.slots.iter().position(|&s| s == r as u32).unwrap() / TILE_ROWS as usize).collect();
        let want = dense(&q, &k, &v, heads, d, &|h, i, j| sel[h][tile_of[i]].contains(&(tile_of[j] as u32)));
        let got = attention_host(&g, (&q, &k, &v), None, (heads, d), sparsity);
        assert!(close(&got, &want) < 1e-5, "{}", close(&got, &want));

        let gate: Vec<f32> = (0..rows * heads * d).map(|i| 0.01 * (i % 13) as f32).collect();
        let gated = attention_host(&g, (&q, &k, &v), Some(&gate), (heads, d), sparsity);
        let vp = pool(&g, &v, heads, d);
        for r in [0usize, 20, rows - 1] {
            for h in 0..heads {
                let row = &scores[(h * n + tile_of[r]) * n..][..n];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let total: f32 = row.iter().map(|s| (s - max).exp()).sum();
                for c in 0..d {
                    let coarse: f32 = (0..n).map(|j| (row[j] - max).exp() / total * vp[(h * n + j) * d + c]).sum();
                    let i = (r * heads + h) * d + c;
                    assert!((gated[i] - (got[i] + coarse * gate[i])).abs() < 1e-5, "row {r} head {h} channel {c}");
                }
            }
        }
    }

    /// Every packed row sits in exactly one live slot, a dense chunk never
    /// mixes two segments, every dense segment's tiles lead - the target
    /// audio's too, though it sits between two videos - and each video's
    /// tiles are one contiguous span.
    #[test]
    fn every_row_has_one_slot_and_dense_tiles_lead_without_mixing_segments() {
        let segments = [
            Segment::Dense { rows: 70 },
            Segment::Dense { rows: 9 },
            Segment::Video { grid: [2, 3, 5] },
            Segment::Dense { rows: 130 },
            Segment::Video { grid: [5, 4, 6] },
        ];
        let g = TileGeometry::new(&segments);
        let rows: u32 = segments.iter().map(Segment::rows).sum();
        let mut seen = vec![0u32; rows as usize];
        for (tile, &live) in g.live.iter().enumerate() {
            let slots = &g.slots[tile * TILE_ROWS as usize..][..TILE_ROWS as usize];
            assert!(slots[..live as usize].iter().all(|&r| r != PAD) && slots[live as usize..].iter().all(|&r| r == PAD), "tile {tile}: live rows first");
            slots[..live as usize].iter().for_each(|&r| seen[r as usize] += 1);
        }
        assert!(seen.iter().all(|&n| n == 1), "every row in exactly one slot");

        assert_eq!(&g.live[..g.prefix_tiles as usize], &[64, 6, 9, 64, 64, 2], "chunks of 70, 9 and 130 rows");
        let tile_rows = |tile: usize| &g.slots[tile * TILE_ROWS as usize..][..g.live[tile] as usize];
        assert_eq!(tile_rows(1), &(64..70).collect::<Vec<_>>()[..], "the text's tail never shares a tile with the next segment");
        assert_eq!(tile_rows(3)[0], 70 + 9 + 30, "the target audio leads too");
        assert_eq!(g.video_spans, vec![(6, 8), (8, 12)]);
        assert_eq!(g.tiles(), 12);
    }

    /// A `(t, h, w)` grid that is no multiple of the tile shape: tiles in
    /// `(t, h, w)` order, each holding its rows frame-major then row-major,
    /// the edge tiles fewer.
    #[test]
    fn a_video_region_tiles_in_t_h_w_order_with_partial_edge_tiles() {
        let g = TileGeometry::new(&[Segment::Video { grid: [5, 4, 6] }]);
        assert_eq!(g.prefix_tiles, 0);
        assert_eq!(g.live, vec![64, 32, 16, 8], "(t0, w0), (t0, w1), (t1, w0), (t1, w1)");
        let tile = |i: usize| &g.slots[i * TILE_ROWS as usize..][..g.live[i] as usize];
        assert_eq!(&tile(0)[..6], &[0, 1, 2, 3, 6, 7], "row (0, 0, 0..4), then (0, 1, 0..)");
        assert_eq!(&tile(1)[..4], &[4, 5, 10, 11], "the second w tile holds columns 4 and 5");
        assert_eq!(tile(2)[0], 4 * 24, "the second t tile starts at frame 4");
        assert_eq!(*tile(3).last().unwrap(), 5 * 24 - 1);
    }

    #[test]
    fn keep_is_the_ceiling_of_the_kept_fraction_within_one_and_all() {
        assert_eq!(keep(0.75, 10), 3);
        assert_eq!(keep(0.75, 12), 3);
        assert_eq!(keep(0.9, 5), 1);
        assert_eq!(keep(0.0, 7), 7);
        assert_eq!(keep(1.0, 7), 1);
    }
}
