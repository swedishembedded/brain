// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decoder as a training graph: teacher-forced, the whole token sequence as one chunk from an empty state
//! (the full-recompute mode of the inference decoder, which the incremental mode equals).

use std::rc::Rc;

use super::graph::Graph;
use super::net::Net;
use super::ops::Seg;
use super::real::{r, Real};
use super::seq::AttnShape;
use super::tape::Id;
use crate::config::{ENGRAM_TAPS, HADAMARD_COND_RANK};
use crate::decoder::hadamard::{legacy_permutation, PERMUTATION_SEEDS};
use crate::decoder::{engram_indices, ENGRAM_DILATION, SINKHORN_ITERATIONS};

impl<R: Real, T: Graph<R>> Net<'_, R, T> {
    /// `(att * sigmoid(h Wgate)) Wout` with the kernels `gate` and `out` of layer `l` under `prefix`.
    fn gated_out(&mut self, att: Id, h: Id, prefix: &str, l: usize, t: usize) -> Id {
        let c = self.cfg.clone();
        let width = c.heads() * c.v_dim;
        let g = self.proj(h, &format!("stack/layers/block/{prefix}/gate_proj/kernel"), Some(l), t, c.dim, width);
        let g = self.tape.sigmoid(g);
        let a = self.tape.mul(att, g);
        self.proj(a, &format!("stack/layers/block/{prefix}/out_proj/kernel"), Some(l), t, width, c.dim)
    }

    fn block_param(&mut self, name: &str, l: usize) -> Id {
        self.layer(&format!("stack/layers/block/{name}"), l)
    }

    /// The decoder's rotary angles for `t` positions: `[t, qk/2]`.
    fn decoder_angles(&self, t: usize) -> Vec<f64> {
        let hd = self.cfg.qk_dim;
        let half = hd / 2;
        let freq: Vec<f32> = (0..half).map(|i| 1.0 / self.cfg.rope.powf((2 * i) as f32 / hd as f32)).collect();
        (0..t).flat_map(|pos| freq.iter().map(move |f| f64::from(pos as f32 * f))).collect()
    }

    /// The Hadamard MLP of layer `l` on normalised rows `x` (`[t, dim]`).
    fn hadamard_mlp(&mut self, x: Id, l: usize, t: usize) -> Id {
        let c = self.cfg.clone();
        let n = c.hadamard_width();
        assert_eq!(n, c.dim, "the training graph supports a Hadamard width equal to the model width");
        let (fa, fb) = c.hadamard_factors();
        let name = |s: &str| format!("hadamard_mlp/{s}");
        let perms = (legacy_permutation(PERMUTATION_SEEDS.0, n), legacy_permutation(PERMUTATION_SEEDS.1, n));
        let cond_v = self.layer(&format!("stack/layers/block/{}", name("cond_v")), l);
        let logits = self.tape.linear(x, cond_v, t, c.dim, HADAMARD_COND_RANK);
        let probs = self.tape.softmax_rows(logits, HADAMARD_COND_RANK);
        let cond_u = self.layer(&format!("stack/layers/block/{}", name("cond_u")), l);
        let cond = self.tape.linear(probs, cond_u, t, HADAMARD_COND_RANK, n);
        let one = self.constant(vec![R::ONE]);
        let cond = self.tape.bias_add(cond, one);

        let (d1, d2, b2, d3, d4) = (self.block_param(&name("d1"), l), self.block_param(&name("d2"), l), self.block_param(&name("b2"), l), self.block_param(&name("d3"), l), self.block_param(&name("d4"), l));
        let mut factors = Vec::new();
        for stage in 1..=3 {
            let a = self.block_param(&name(&format!("w{stage}a")), l);
            let b = self.block_param(&name(&format!("w{stage}b")), l);
            factors.push((a, b));
        }
        let z = self.tape.chan_mul(x, d1);
        let z = self.tape.kron(z, factors[0].0, factors[0].1, fa, fb);
        let z = self.tape.permute_cols(z, n, &perms.0);
        let z = self.tape.mul(cond, z);
        let z = self.tape.chan_mul(z, d2);
        let z = self.tape.bias_add(z, b2);
        let z = self.tape.silu(z);
        let z = self.tape.kron(z, factors[1].0, factors[1].1, fa, fb);
        let z = self.tape.permute_cols(z, n, &perms.1);
        let z = self.tape.chan_mul(z, d3);
        let z = self.tape.kron(z, factors[2].0, factors[2].1, fa, fb);
        self.tape.chan_mul(z, d4)
    }

    /// Engram key and tap-convolved value rows of `site` for the whole sequence.
    fn engram(&mut self, site: usize, tokens: &[u32]) -> (Id, Id) {
        let c = self.cfg.clone();
        let (d, tables, sub) = (c.dim, c.engram_tables(), c.engram_sub_dim());
        let t = tokens.len();
        let idx = engram_indices(&c, tokens, 0);
        let table = self.p(&format!("engrams_{site}/embedding"));
        let mut segs = Vec::new();
        for pos in 0..t {
            for tab in 0..tables {
                // The first `order - 1` positions have an incomplete n-gram: their row is zero.
                if pos + 1 >= c.orders[tab / c.hashes] {
                    segs.push(Seg::copy(pos * d + tab * sub, (tab * c.slots + idx[pos * tables + tab]) * sub, sub));
                }
            }
        }
        let gathered = self.tape.plan(table, t * d, segs);
        let key = self.proj(gathered, &format!("engrams_{site}/key_proj/kernel"), None, t, d, d);
        let value = self.proj(gathered, &format!("engrams_{site}/value_proj/kernel"), None, t, d, d);
        let taps = self.p(&format!("engrams_{site}/taps"));
        assert_eq!(self.tape.len(taps), ENGRAM_TAPS * d, "engram taps");
        let conv = self.tape.causal_taps(value, taps, d, ENGRAM_DILATION);
        (key, conv)
    }

    /// One decoder block on the lane-mixed input `u` (`[t, dim]`) of layer `l`.
    fn decoder_block(&mut self, l: usize, u: Id, engram: Option<(Id, Id)>, cross: Id, mem_rows: usize, t: usize) -> Id {
        let c = self.cfg.clone();
        let d = c.dim;
        let (heads, kvh, hd, vh) = (c.heads(), c.kv_heads, c.qk_dim, c.v_dim);
        let mut y = u;
        if let Some((key, conv)) = engram {
            let ny = self.tape.rms_unit(y, d);
            let nk = self.tape.rms_unit(key, d);
            let dot = self.tape.row_dot(ny, nk, d);
            let dot = self.tape.scale(dot, 1.0 / (d as f64).sqrt());
            let alpha = self.tape.sigmoid(dot);
            let add = self.tape.row_mul(alpha, conv, d);
            y = self.tape.add(y, add);
        }
        let skip = y;
        let norm = self.block_param("ZCRMSNorm_0/scale", l);
        let h = self.tape.zc_norm(y, d, norm);
        let q = self.proj(h, "stack/layers/block/self_attn/q_proj/kernel", Some(l), t, d, heads * hd);
        let k = self.proj(h, "stack/layers/block/self_attn/k_proj/kernel", Some(l), t, d, kvh * hd);
        let v = self.proj(h, "stack/layers/block/self_attn/v_proj/kernel", Some(l), t, d, kvh * vh);
        let (qt, kt, vt) = (self.block_param("self_attn/q_taps", l), self.block_param("self_attn/k_taps", l), self.block_param("self_attn/v_taps", l));
        let q = self.tape.causal_taps(q, qt, heads * hd, 1);
        let k = self.tape.causal_taps(k, kt, kvh * hd, 1);
        let v = self.tape.causal_taps(v, vt, kvh * vh, 1);
        let (qn, kn) = (self.block_param("self_attn/q_norm/scale", l), self.block_param("self_attn/k_norm/scale", l));
        let q = self.tape.zc_norm(q, hd, qn);
        let k = self.tape.zc_norm(k, hd, kn);
        let angles = self.decoder_angles(t);
        let q = self.tape.rope(q, heads, hd, &angles);
        let k = self.tape.rope(k, kvh, hd, &angles);
        let att = self.tape.attention(q, k, v, AttnShape { tq: t, tk: t, hq: heads, hk: kvh, qk: hd, vh, causal: true });
        let o = self.gated_out(att, h, "self_attn", l, t);
        let post = self.block_param("post_attn_norm/scale", l);
        let o = self.tape.zc_norm(o, d, post);
        let gate = self.block_param("attn_gate", l);
        let gate = self.tape.sigmoid(gate);
        let o = self.tape.scalar_mul(o, gate);
        let y = self.tape.add(skip, o);

        let skip = y;
        let cross_norm = self.block_param("cross_norm/scale", l);
        let h = self.tape.zc_norm(y, d, cross_norm);
        let cq = self.proj(h, "stack/layers/block/cross_attn/q_proj/kernel", Some(l), t, d, heads * hd);
        let cqn = self.block_param("cross_attn/q_norm/scale", l);
        let cq = self.tape.zc_norm(cq, hd, cqn);
        let ck = self.proj(cross, "stack/layers/block/cross_attn/k_proj/kernel", Some(l), mem_rows, d, heads * hd);
        let ckn = self.block_param("cross_attn/k_norm/scale", l);
        let ck = self.tape.zc_norm(ck, hd, ckn);
        let cv = self.proj(cross, "stack/layers/block/cross_attn/v_proj/kernel", Some(l), mem_rows, d, heads * vh);
        let att = self.tape.attention(cq, ck, cv, AttnShape { tq: t, tk: mem_rows, hq: heads, hk: heads, qk: hd, vh, causal: false });
        let o = self.gated_out(att, h, "cross_attn", l, t);
        let post = self.block_param("post_cross_norm/scale", l);
        let o = self.tape.zc_norm(o, d, post);
        let gate = self.block_param("cross_gate", l);
        let gate = self.tape.sigmoid(gate);
        let o = self.tape.scalar_mul(o, gate);
        let y = self.tape.add(skip, o);

        let pre = self.block_param("pre_hada_norm/scale", l);
        let hn = self.tape.zc_norm(y, d, pre);
        let mlp = self.hadamard_mlp(hn, l, t);
        self.tape.add(y, mlp)
    }

    /// Decoder layer `l`: the lane-mixing read, the block, and the lane write, on the lanes `lanes`
    /// (`[t, lanes, dim]`); returns the updated lanes.
    fn decoder_layer(&mut self, l: usize, lanes: Id, engram: Option<(Id, Id)>, memory: Id, mem_rows: usize, t: usize) -> Id {
        let c = self.cfg.clone();
        let (d, n) = (c.dim, c.lanes);
        let lane_id = l % n;
        let nx = self.tape.rms_unit(lanes, n * d);
        let (phi_pre, phi_post, phi_res) = (self.layer("stack/mhc_phi_pre", l), self.layer("stack/mhc_phi_post", l), self.layer("stack/mhc_phi_res", l));
        let pre = self.tape.linear(nx, phi_pre, t, n * d, n);
        let post = self.tape.linear(nx, phi_post, t, n * d, n);
        let res = self.tape.linear(nx, phi_res, t, n * d, n * n);
        let (a_pre, a_post, a_res) = (self.layer("stack/mhc_a_pre", l), self.layer("stack/mhc_a_post", l), self.layer("stack/mhc_a_res", l));
        let (b_pre, b_post, b_res) = (self.layer("stack/mhc_b_pre", l), self.layer("stack/mhc_b_post", l), self.layer("stack/mhc_b_res", l));
        let one_hot = |scale: f64, shift: f64| (0..n).map(|i| r::<R>(scale * if i == lane_id { 1.0 } else { 0.0 } + shift)).collect::<Vec<_>>();

        let p = self.tape.scalar_mul(pre, a_pre);
        let p = self.tape.bias_add(p, b_pre);
        let pc = self.constant(one_hot(8.0, -4.0));
        let p = self.tape.bias_add(p, pc);
        let h_pre = self.tape.sigmoid(p);
        let u = self.tape.row_mix(h_pre, lanes, 1, n, d);

        let y = self.decoder_block(l, u, engram, memory, mem_rows, t);

        let q = self.tape.scalar_mul(post, a_post);
        let q = self.tape.bias_add(q, b_post);
        // -4 * (1 - lane_one): zero on the layer's own lane, -4 elsewhere.
        let qc = self.constant((0..n).map(|i| r::<R>(if i == lane_id { 0.0 } else { -4.0 })).collect());
        let q = self.tape.bias_add(q, qc);
        let q = self.tape.sigmoid(q);
        let h_post = self.tape.scale(q, 2.0);

        let rs = self.tape.scalar_mul(res, a_res);
        let rs = self.tape.bias_add(rs, b_res);
        let h_res = self.tape.sinkhorn(rs, n, SINKHORN_ITERATIONS);
        let keep = self.tape.row_mix(h_res, lanes, n, n, d);
        let neg_u = self.tape.scale(u, -1.0);
        let delta = self.tape.add(y, neg_u);
        let write = self.tape.row_mix(h_post, delta, n, 1, d);
        self.tape.add(keep, write)
    }

    /// Teacher-forced logits `[t, vocab]` for `tokens` over an image memory of `mem_rows` rows.
    pub fn decoder_logits(&mut self, tokens: &[u32], memory: Id, mem_rows: usize) -> Id {
        let c = self.cfg.clone();
        let (d, n) = (c.dim, c.lanes);
        let t = tokens.len();
        assert!(t > 0, "the decoder needs at least one token");
        assert!(tokens.iter().all(|&k| (k as usize) < c.embedding_rows()), "token id outside the embedding");
        let embedding = self.p("embedding/embedding");
        let root = r::<R>((d as f64).sqrt());
        let x = self.tape.plan(embedding, t * d, tokens.iter().enumerate().map(|(i, &tok)| Seg { dst: i * d, src: tok as usize * d, len: d, w: root }).collect());
        let shared_tokens: Rc<Vec<u32>> = Rc::new(tokens.to_vec());
        let engrams: Vec<(Id, Id)> = (0..c.sites.len())
            .map(|site| {
                let tokens = Rc::clone(&shared_tokens);
                let kc = self.block(move |net| {
                    let (key, conv) = net.engram(site, &tokens);
                    vec![key, conv]
                });
                (kc[0], kc[1])
            })
            .collect();

        // Every token starts with the same vector in each of the n lanes.
        let mut lanes = self.tape.plan(x, t * n * d, (0..t * n).map(|i| Seg::copy(i * d, (i / n) * d, d)).collect());
        for l in 0..c.dec_layers {
            let (input, engram) = (lanes, c.sites.iter().position(|&s| s == l).map(|s| engrams[s]));
            lanes = self.block1(move |net| net.decoder_layer(l, input, engram, memory, mem_rows, t));
        }
        let inv = r::<R>(1.0 / n as f64);
        let mean = self.tape.plan(lanes, t * d, (0..t * n).map(|i| Seg { dst: (i / n) * d, src: i * d, len: d, w: inv }).collect());
        let final_norm = self.p("stack/final_norm/scale");
        let normed = self.tape.zc_norm(mean, d, final_norm);
        let vocab = c.embedding_rows();
        self.tape.linear_nt(normed, embedding, t, d, vocab)
    }
}
