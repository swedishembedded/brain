// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Mirage decoder: an 8-layer stack over four hyper-connection lanes with
//! engram-augmented self-attention, cross-attention over the image memory and
//! a Hadamard MLP, producing logits over the tied embedding.
//!
//! One forward routine serves both modes. It consumes a chunk of tokens
//! appended to a [`DecoderState`]; running the whole sequence as one chunk
//! from an empty state is the full recompute, and feeding one token per call
//! is incremental decoding. The state holds exactly what later tokens read:
//! per-layer key/value caches, the last two raw q/k/v rows for the causal
//! taps, the token history for n-gram hashing, and the last nine engram value
//! rows for its dilated taps. Cross-attention keys and values of the memory
//! are computed once per image ([`CrossMemory`]).
//!
//! Swedish Embedded AB implements solutions for incremental decoding of small
//! multimodal models for its clients. If your team needs expertise in
//! cache-exact autoregressive inference then you can procure our services by
//! sending an email to info@swedishembedded.com.


pub mod generate;
pub mod hadamard;

use backend_cpu::par;
use model::hostmath;

use crate::config::{MirageConfig, ENGRAM_TAPS};
use crate::vision::ops::{self, Linear};
use crate::vision::weights::Weights;
use hadamard::{legacy_permutation, HadamardMlp, PERMUTATION_SEEDS};

const ENGRAM_SEED: u64 = 0x9E37_79B9;
const ENGRAM_PRIME: u64 = 0x0100_0193;
const ENGRAM_DILATION: usize = 3;
const SINKHORN_ITERATIONS: usize = 20;
const DECODER_ROPE_PAIRS_DIV: usize = 2;

/// Threads for token-by-token decoding. One token's matrices are small, so
/// fanning out over every core of a large machine spends far more on waking
/// workers than it saves.
pub const DECODE_THREADS: usize = 16;

/// Run `f` on at most [`DECODE_THREADS`] threads.
pub fn with_decode_threads<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    par::with_threads(DECODE_THREADS.min(par::threads()), f)
}

struct Layer {
    norm: Vec<f32>,
    q: Linear,
    k: Linear,
    v: Linear,
    q_taps: Vec<f32>,
    k_taps: Vec<f32>,
    v_taps: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    gate: Linear,
    out: Linear,
    post_attn: Vec<f32>,
    attn_gate: f32,
    cross_norm: Vec<f32>,
    cq: Linear,
    ck: Linear,
    cv: Linear,
    cgate: Linear,
    cout: Linear,
    cq_norm: Vec<f32>,
    ck_norm: Vec<f32>,
    post_cross: Vec<f32>,
    cross_gate: f32,
    pre_hada: Vec<f32>,
    mlp: HadamardMlp,
    // hyper-connection lane maps
    phi_pre: Linear,
    phi_post: Linear,
    phi_res: Linear,
    a: [f32; 3],
    b_pre: Vec<f32>,
    b_post: Vec<f32>,
    b_res: Vec<f32>,
}

struct Engram {
    table: Vec<f32>,
    key: Linear,
    value: Linear,
    taps: Vec<f32>,
}

/// The decoder's weights, prepared for execution.
pub struct Decoder {
    cfg: MirageConfig,
    layers: Vec<Layer>,
    engrams: Vec<Engram>,
    embedding: Linear,
    final_norm: Vec<f32>,
    perms: (Vec<u32>, Vec<u32>),
}

/// Cross-attention keys and values of one image's memory, per layer.
pub struct CrossMemory {
    rows: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl CrossMemory {
    /// Memory rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
}

#[derive(Default, Clone)]
struct LayerState {
    k: Vec<f32>,
    v: Vec<f32>,
    /// Last two raw (pre-tap) q, k, v rows, oldest first.
    raw: [Vec<f32>; 3],
}

/// Everything earlier tokens leave behind for later ones.
#[derive(Clone)]
pub struct DecoderState {
    tokens: Vec<u32>,
    layers: Vec<LayerState>,
    engram_v: Vec<Vec<f32>>,
}

/// Which positions to return logits for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Positions {
    /// Every token of the chunk.
    All,
    /// Only the last token.
    Last,
}

fn prefix<'a>(w: &'a Weights, name: &str, layer: usize) -> &'a [f32] {
    w.layer(&format!("stack/layers/block/{name}"), layer)
}

impl Decoder {
    /// Prepare from the checkpoint tensors.
    pub fn new(cfg: &MirageConfig, w: &Weights) -> Decoder {
        let (d, n) = (cfg.dim, cfg.lanes);
        let heads = cfg.heads();
        let (qd, kd, vd) = (heads * cfg.qk_dim, cfg.kv_heads * cfg.qk_dim, cfg.kv_heads * cfg.v_dim);
        let lin = |name: &str, layer: usize, inn: usize, out: usize| Linear::from_flax(prefix(w, name, layer), inn, out);
        let vec = |name: &str, layer: usize| prefix(w, name, layer).to_vec();
        let layers = (0..cfg.dec_layers)
            .map(|l| Layer {
                norm: vec("ZCRMSNorm_0/scale", l),
                q: lin("self_attn/q_proj/kernel", l, d, qd),
                k: lin("self_attn/k_proj/kernel", l, d, kd),
                v: lin("self_attn/v_proj/kernel", l, d, vd),
                q_taps: vec("self_attn/q_taps", l),
                k_taps: vec("self_attn/k_taps", l),
                v_taps: vec("self_attn/v_taps", l),
                q_norm: vec("self_attn/q_norm/scale", l),
                k_norm: vec("self_attn/k_norm/scale", l),
                gate: lin("self_attn/gate_proj/kernel", l, d, heads * cfg.v_dim),
                out: lin("self_attn/out_proj/kernel", l, heads * cfg.v_dim, d),
                post_attn: vec("post_attn_norm/scale", l),
                attn_gate: prefix(w, "attn_gate", l)[0],
                cross_norm: vec("cross_norm/scale", l),
                cq: lin("cross_attn/q_proj/kernel", l, d, qd),
                ck: lin("cross_attn/k_proj/kernel", l, d, qd),
                cv: lin("cross_attn/v_proj/kernel", l, d, heads * cfg.v_dim),
                cgate: lin("cross_attn/gate_proj/kernel", l, d, heads * cfg.v_dim),
                cout: lin("cross_attn/out_proj/kernel", l, heads * cfg.v_dim, d),
                cq_norm: vec("cross_attn/q_norm/scale", l),
                ck_norm: vec("cross_attn/k_norm/scale", l),
                post_cross: vec("post_cross_norm/scale", l),
                cross_gate: prefix(w, "cross_gate", l)[0],
                pre_hada: vec("pre_hada_norm/scale", l),
                mlp: HadamardMlp::new(cfg, w, l),
                phi_pre: Linear::from_flax(w.layer("stack/mhc_phi_pre", l), n * d, n),
                phi_post: Linear::from_flax(w.layer("stack/mhc_phi_post", l), n * d, n),
                phi_res: Linear::from_flax(w.layer("stack/mhc_phi_res", l), n * d, n * n),
                a: [w.get("stack/mhc_a_pre")[l], w.get("stack/mhc_a_post")[l], w.get("stack/mhc_a_res")[l]],
                b_pre: w.layer("stack/mhc_b_pre", l).to_vec(),
                b_post: w.layer("stack/mhc_b_post", l).to_vec(),
                b_res: w.layer("stack/mhc_b_res", l).to_vec(),
            })
            .collect();
        let engrams = (0..cfg.sites.len())
            .map(|s| Engram {
                table: w.vec(&format!("engrams_{s}/embedding")),
                key: w.linear(&format!("engrams_{s}/key_proj/kernel")),
                value: w.linear(&format!("engrams_{s}/value_proj/kernel")),
                taps: w.vec(&format!("engrams_{s}/taps")),
            })
            .collect();
        let n_hada = cfg.hadamard_width();
        Decoder {
            cfg: cfg.clone(),
            layers,
            engrams,
            embedding: Linear { wt: w.vec("embedding/embedding"), inn: d, out: cfg.embedding_rows() },
            final_norm: w.vec("stack/final_norm/scale"),
            perms: (legacy_permutation(PERMUTATION_SEEDS.0, n_hada), legacy_permutation(PERMUTATION_SEEDS.1, n_hada)),
        }
    }

    /// Rows of the output vocabulary.
    pub fn vocabulary(&self) -> usize {
        self.embedding.out
    }

    /// The Hadamard permutations in use.
    pub fn permutations(&self) -> &(Vec<u32>, Vec<u32>) {
        &self.perms
    }

    /// An empty state.
    pub fn new_state(&self) -> DecoderState {
        DecoderState { tokens: Vec::new(), layers: vec![LayerState::default(); self.layers.len()], engram_v: vec![Vec::new(); self.engrams.len()] }
    }

    /// Cross-attention keys and values for a memory of `rows` rows.
    pub fn cross_memory(&self, memory: &[f32], rows: usize) -> CrossMemory {
        let c = &self.cfg;
        let (heads, hd) = (c.heads(), c.qk_dim);
        let (mut k, mut v) = (Vec::new(), Vec::new());
        for l in &self.layers {
            k.push(ops::zc_norm(&l.ck.apply(memory, rows), hd, &l.ck_norm));
            v.push(l.cv.apply(memory, rows));
        }
        debug_assert_eq!(k[0].len(), rows * heads * hd);
        CrossMemory { rows, k, v }
    }

    /// N-gram table indices for positions `p0..p0+t`, `[t, tables]`.
    fn engram_indices(&self, tokens: &[u32], p0: usize) -> Vec<usize> {
        let c = &self.cfg;
        let tables = c.engram_tables();
        let mut out = Vec::with_capacity((tokens.len() - p0) * tables);
        for t in p0..tokens.len() {
            for (oi, &order) in c.orders.iter().enumerate() {
                for hh in 0..c.hashes {
                    let mut acc = (ENGRAM_SEED * (oi * c.hashes + hh + 1) as u64) & 0xFFFF_FFFF;
                    for j in 0..order {
                        let tok = if t >= j { u64::from(tokens[t - j]) } else { 0 };
                        acc = ((acc ^ tok) * ENGRAM_PRIME) & 0xFFFF_FFFF;
                    }
                    acc ^= acc >> 15;
                    out.push((acc % c.slots as u64) as usize);
                }
            }
        }
        out
    }

    /// Engram keys and tap-convolved values for the new positions.
    fn engram(&self, st: &mut DecoderState, site: usize, p0: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let (d, tables, sub) = (c.dim, c.engram_tables(), c.engram_sub_dim());
        let e = &self.engrams[site];
        let idx = self.engram_indices(&st.tokens, p0);
        let mut gathered = vec![0f32; t * d];
        for (r, row) in gathered.chunks_exact_mut(d).enumerate() {
            let pos = p0 + r;
            for (tab, &slot) in idx[r * tables..(r + 1) * tables].iter().enumerate() {
                // The first `order - 1` positions have an incomplete n-gram: their row is zero.
                if pos + 1 >= c.orders[tab / c.hashes] {
                    row[tab * sub..(tab + 1) * sub].copy_from_slice(&e.table[(tab * c.slots + slot) * sub..][..sub]);
                }
            }
        }
        let key = e.key.apply(&gathered, t);
        let value = e.value.apply(&gathered, t);
        let history = &mut st.engram_v[site];
        let hist_rows = history.len() / d;
        let at = |pos: isize| -> Option<&[f32]> {
            if pos < 0 {
                return None;
            }
            let pos = pos as usize;
            Some(if pos >= p0 { &value[(pos - p0) * d..][..d] } else { &history[(hist_rows - (p0 - pos)) * d..][..d] })
        };
        let mut conv = vec![0f32; t * d];
        for (r, row) in conv.chunks_exact_mut(d).enumerate() {
            for j in 0..ENGRAM_TAPS {
                if let Some(src) = at((p0 + r) as isize - (j * ENGRAM_DILATION) as isize) {
                    row.iter_mut().zip(src.iter().zip(&e.taps[j * d..(j + 1) * d])).for_each(|(o, (s, w))| *o += w * s);
                }
            }
        }
        let keep = ENGRAM_DILATION * (ENGRAM_TAPS - 1);
        let mut joined = std::mem::take(history);
        joined.extend_from_slice(&value);
        let drop_rows = (joined.len() / d).saturating_sub(keep);
        *history = joined.split_off(drop_rows * d);
        (key, conv)
    }

    /// Causal taps over `raw` (`[t, w]`) with the two previous rows from `prev`.
    fn taps(raw: &[f32], prev: &[f32], taps: &[f32], w: usize, t: usize) -> Vec<f32> {
        let hist = prev.len() / w;
        let mut out = vec![0f32; t * w];
        par::rows_mut(&mut out, w, |r, row| {
            for j in 0..taps.len() / w {
                // Row `r - j` of the history-extended sequence.
                let src: Option<&[f32]> = if r >= j { Some(&raw[(r - j) * w..][..w]) } else if hist >= j - r { Some(&prev[(hist - (j - r)) * w..][..w]) } else { None };
                if let Some(src) = src {
                    row.iter_mut().zip(src.iter().zip(&taps[j * w..(j + 1) * w])).for_each(|(o, (s, k))| *o += k * s);
                }
            }
        });
        out
    }

    fn rope(&self, x: &mut [f32], heads: usize, p0: usize) {
        let hd = self.cfg.qk_dim;
        let half = hd / DECODER_ROPE_PAIRS_DIV;
        let freq: Vec<f32> = (0..half).map(|i| 1.0 / self.cfg.rope.powf((2 * i) as f32 / hd as f32)).collect();
        par::rows_mut(x, heads * hd, |r, row| {
            let pos = (p0 + r) as f32;
            for head in row.chunks_exact_mut(hd) {
                for i in 0..half {
                    let (s, c) = (pos * freq[i]).sin_cos();
                    let (a, b) = (head[i], head[i + half]);
                    head[i] = a * c - b * s;
                    head[i + half] = b * c + a * s;
                }
            }
        });
    }

    /// Causal grouped-query attention of `t` new queries against a cache.
    fn self_attention(&self, q: &[f32], ls: &LayerState, p0: usize, t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let (heads, kvh, hd, vh) = (c.heads(), c.kv_heads, c.qk_dim, c.v_dim);
        let group = heads / kvh;
        let scale = 1.0 / (hd as f32).sqrt();
        let (kw, vw) = (kvh * hd, kvh * vh);
        let mut out = vec![0f32; t * heads * vh];
        par::rows_mut(&mut out, heads * vh, |r, row| {
            let visible = p0 + r + 1;
            let mut scores = vec![0f32; visible];
            for h in 0..heads {
                let g = h / group;
                let qv = &q[(r * heads + h) * hd..][..hd];
                for (s, sc) in scores.iter_mut().enumerate() {
                    *sc = qv.iter().zip(&ls.k[s * kw + g * hd..][..hd]).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                hostmath::softmax(&mut scores);
                let o = &mut row[h * vh..(h + 1) * vh];
                for (s, &p) in scores.iter().enumerate() {
                    o.iter_mut().zip(&ls.v[s * vw + g * vh..][..vh]).for_each(|(a, b)| *a += p * b);
                }
            }
        });
        out
    }

    fn cross_attention(&self, q: &[f32], k: &[f32], v: &[f32], rows: usize, t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let (heads, hd, vh) = (c.heads(), c.qk_dim, c.v_dim);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0f32; t * heads * vh];
        par::rows_mut(&mut out, heads * vh, |r, row| {
            let mut scores = vec![0f32; rows];
            for h in 0..heads {
                let qv = &q[(r * heads + h) * hd..][..hd];
                for (s, sc) in scores.iter_mut().enumerate() {
                    *sc = qv.iter().zip(&k[(s * heads + h) * hd..][..hd]).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                hostmath::softmax(&mut scores);
                let o = &mut row[h * vh..(h + 1) * vh];
                for (s, &p) in scores.iter().enumerate() {
                    o.iter_mut().zip(&v[(s * heads + h) * vh..][..vh]).for_each(|(a, b)| *a += p * b);
                }
            }
        });
        out
    }

    /// One decoder block on the lane-mixed input `u` (`[t, dim]`).
    fn block(&self, l: usize, st: &mut DecoderState, cross: &CrossMemory, u: &[f32], engram: Option<&(Vec<f32>, Vec<f32>)>, p0: usize, t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.dim;
        let ly = &self.layers[l];
        let mut y = u.to_vec();
        if let Some((key, value)) = engram {
            let (ny, nk) = (ops::rms_unit(&y, d), ops::rms_unit(key, d));
            par::rows_mut(&mut y, d, |r, row| {
                let dot: f32 = ny[r * d..(r + 1) * d].iter().zip(&nk[r * d..(r + 1) * d]).map(|(a, b)| a * b).sum();
                let alpha = ops::sigmoid(dot / (d as f32).sqrt());
                row.iter_mut().zip(&value[r * d..(r + 1) * d]).for_each(|(a, v)| *a += alpha * v);
            });
        }
        let skip = y.clone();
        let h = ops::zc_norm(&y, d, &ly.norm);
        let (heads, kvh, hd, vh) = (c.heads(), c.kv_heads, c.qk_dim, c.v_dim);
        let state = &mut st.layers[l];
        let raw = [ly.q.apply(&h, t), ly.k.apply(&h, t), ly.v.apply(&h, t)];
        let widths = [heads * hd, kvh * hd, kvh * vh];
        let taps = [&ly.q_taps, &ly.k_taps, &ly.v_taps];
        let mut conv: Vec<Vec<f32>> = (0..3).map(|i| Self::taps(&raw[i], &state.raw[i], taps[i], widths[i], t)).collect();
        for i in 0..3 {
            let mut joined = std::mem::take(&mut state.raw[i]);
            joined.extend_from_slice(&raw[i]);
            let keep = joined.len().saturating_sub(2 * widths[i]);
            state.raw[i] = joined.split_off(keep);
        }
        let mut v = conv.pop().expect("three projections");
        let mut k = ops::zc_norm(&conv.pop().expect("three projections"), hd, &ly.k_norm);
        let mut q = ops::zc_norm(&conv.pop().expect("three projections"), hd, &ly.q_norm);
        self.rope(&mut q, heads, p0);
        self.rope(&mut k, kvh, p0);
        state.k.append(&mut k);
        state.v.append(&mut v);
        let att = self.self_attention(&q, state, p0, t);
        let o = self.gated(&att, &h, &ly.gate, &ly.out, t);
        let mut y = skip;
        add_gated(&mut y, &ops::zc_norm(&o, d, &ly.post_attn), ops::sigmoid(ly.attn_gate), d);

        let skip = y.clone();
        let h = ops::zc_norm(&y, d, &ly.cross_norm);
        let q = ops::zc_norm(&ly.cq.apply(&h, t), hd, &ly.cq_norm);
        let att = self.cross_attention(&q, &cross.k[l], &cross.v[l], cross.rows, t);
        let o = self.gated(&att, &h, &ly.cgate, &ly.cout, t);
        let mut y = skip;
        add_gated(&mut y, &ops::zc_norm(&o, d, &ly.post_cross), ops::sigmoid(ly.cross_gate), d);

        let mlp = ly.mlp.forward(&ops::zc_norm(&y, d, &ly.pre_hada), t, &self.perms);
        ops::add_inplace(&mut y, &mlp);
        y
    }

    /// `(att * sigmoid(h Wgate)) Wout`.
    fn gated(&self, att: &[f32], h: &[f32], gate: &Linear, out: &Linear, t: usize) -> Vec<f32> {
        let g = gate.apply(h, t);
        let mut a = att.to_vec();
        par::chunks_mut(&mut a, 4096, |i, ch| ch.iter_mut().zip(&g[i * 4096..]).for_each(|(v, g)| *v *= ops::sigmoid(*g)));
        out.apply(&a, t)
    }

    /// Append `tokens` to the state and return logits (`[rows, vocab]`).
    pub fn forward(&self, st: &mut DecoderState, cross: &CrossMemory, tokens: &[u32], positions: Positions) -> Vec<f32> {
        let c = &self.cfg;
        let (d, n) = (c.dim, c.lanes);
        let t = tokens.len();
        assert!(t > 0, "forward needs at least one token");
        assert!(tokens.iter().all(|&k| (k as usize) < c.embedding_rows()), "token id outside the embedding");
        let p0 = st.tokens.len();
        st.tokens.extend_from_slice(tokens);
        let root = (d as f32).sqrt();
        let mut x = vec![0f32; t * d];
        for (row, &tok) in x.chunks_exact_mut(d).zip(tokens) {
            row.iter_mut().zip(&self.embedding.wt[tok as usize * d..][..d]).for_each(|(o, e)| *o = e * root);
        }
        let engrams: Vec<(Vec<f32>, Vec<f32>)> = (0..self.engrams.len()).map(|s| self.engram(st, s, p0, t)).collect();

        // Lanes: every token starts with the same vector in each of the n lanes.
        let mut lanes = vec![0f32; t * n * d];
        for (r, row) in lanes.chunks_exact_mut(n * d).enumerate() {
            for lane in row.chunks_exact_mut(d) {
                lane.copy_from_slice(&x[r * d..(r + 1) * d]);
            }
        }
        for (l, ly) in self.layers.iter().enumerate() {
            let nx = ops::rms_unit(&lanes, n * d);
            let lane_id = l % n;
            let pre = ly.phi_pre.apply(&nx, t);
            let post = ly.phi_post.apply(&nx, t);
            let res = ly.phi_res.apply(&nx, t);
            let h_pre: Vec<f32> = (0..t * n)
                .map(|i| {
                    let lane_one = if i % n == lane_id { 1.0 } else { 0.0 };
                    ops::sigmoid(ly.a[0] * pre[i] + ly.b_pre[i % n] + (8.0 * lane_one - 4.0))
                })
                .collect();
            let mut u = vec![0f32; t * d];
            for r in 0..t {
                for j in 0..n {
                    let w = h_pre[r * n + j];
                    u[r * d..(r + 1) * d].iter_mut().zip(&lanes[(r * n + j) * d..][..d]).for_each(|(o, v)| *o += w * v);
                }
            }
            let site = c.sites.iter().position(|&s| s == l);
            let y = self.block(l, st, cross, &u, site.map(|s| &engrams[s]), p0, t);
            let h_post: Vec<f32> = (0..t * n)
                .map(|i| {
                    let lane_one = if i % n == lane_id { 1.0 } else { 0.0 };
                    2.0 * ops::sigmoid(ly.a[1] * post[i] + ly.b_post[i % n] + (-4.0 * (1.0 - lane_one)))
                })
                .collect();
            let mut next = vec![0f32; t * n * d];
            par::rows_mut(&mut next, n * d, |r, row| {
                let mut logits: Vec<f32> = (0..n * n).map(|i| ly.a[2] * res[r * n * n + i] + ly.b_res[i]).collect();
                let hres = sinkhorn(&mut logits, n);
                for i in 0..n {
                    let out = &mut row[i * d..(i + 1) * d];
                    for j in 0..n {
                        let w = hres[i * n + j];
                        out.iter_mut().zip(&lanes[(r * n + j) * d..][..d]).for_each(|(o, v)| *o += w * v);
                    }
                    let wp = h_post[r * n + i];
                    out.iter_mut().zip(y[r * d..(r + 1) * d].iter().zip(&u[r * d..(r + 1) * d])).for_each(|(o, (y, u))| *o += wp * (y - u));
                }
            });
            lanes = next;
        }

        let first = if positions == Positions::Last { t - 1 } else { 0 };
        let rows = t - first;
        let mut mean = vec![0f32; rows * d];
        for (r, row) in mean.chunks_exact_mut(d).enumerate() {
            for j in 0..n {
                row.iter_mut().zip(&lanes[((first + r) * n + j) * d..][..d]).for_each(|(o, v)| *o += v / n as f32);
            }
        }
        let normed = ops::zc_norm(&mean, d, &self.final_norm);
        self.embedding.apply(&normed, rows)
    }
}

fn add_gated(y: &mut [f32], o: &[f32], gate: f32, d: usize) {
    par::rows_mut(y, d, |r, row| row.iter_mut().zip(&o[r * d..(r + 1) * d]).for_each(|(a, b)| *a += gate * b));
}

/// Sinkhorn normalisation in log space: alternately normalise rows and
/// columns of `exp(logits)` for a fixed number of iterations.
fn sinkhorn(logits: &mut [f32], n: usize) -> Vec<f32> {
    for _ in 0..SINKHORN_ITERATIONS {
        for i in 0..n {
            let row = &mut logits[i * n..(i + 1) * n];
            let lse = log_sum_exp(row.iter().copied());
            row.iter_mut().for_each(|v| *v -= lse);
        }
        for j in 0..n {
            let lse = log_sum_exp((0..n).map(|i| logits[i * n + j]));
            (0..n).for_each(|i| logits[i * n + j] -= lse);
        }
    }
    logits.iter().map(|v| v.exp()).collect()
}

fn log_sum_exp(values: impl Iterator<Item = f32> + Clone) -> f32 {
    let max = values.clone().fold(f32::NEG_INFINITY, f32::max);
    max + values.map(|v| (v - max).exp()).sum::<f32>().ln()
}
