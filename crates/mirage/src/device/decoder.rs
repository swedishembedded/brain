// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decoder on the device with a shared batched step loop.
//!
//! A batch of sequences (one per image) advances in lockstep: every step feeds
//! one token per sequence at the same position, so the self-attention cache,
//! the tap histories and the engram value history are `[batch, positions, w]`
//! tables written at one shared position. Memories differ in length per
//! sequence and are handled by a key mask over a common stride. A finished
//! sequence keeps stepping on a filler token until the whole batch is done;
//! its result is frozen when it stops. The host owns what is hash- and
//! text-shaped: engram n-gram hashing, rendering and the loop guard.
//!
//! The Hadamard rotations are multiplied out into three dense matrices per
//! layer (`HadamardMlp::fold`), so a layer's MLP is three GEMMs around one
//! elementwise stage.

use gpu_core::DeviceBuffer;
use model::ops::Weight;

use super::prims::{AttnShape, MhcShape};
use super::reader::MemoryBatch;
use super::Device;
use crate::config::{MirageConfig, ENGRAM_TAPS, HADAMARD_COND_RANK};
use crate::decoder::generate::{guard, render, GenerateParams, Generation, Stop};
use crate::decoder::{Decoder, SINKHORN_ITERATIONS};
use crate::tokenizer::Tokenizer;
use crate::vision::ops;

const ENGRAM_DILATION: u32 = 3;
const BAN: f32 = -1.0e30;

fn lin(dev: &Device, name: &str, l: &ops::Linear) -> Weight {
    dev.weight(name, &l.wt, l.out, l.inn)
}

struct DLayer {
    norm: DeviceBuffer,
    qkv: [Weight; 3],
    taps: [DeviceBuffer; 3],
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    gate: Weight,
    out: Weight,
    post_attn: DeviceBuffer,
    attn_gate: f32,
    cross_norm: DeviceBuffer,
    cq: Weight,
    ck: Weight,
    cv: Weight,
    cgate: Weight,
    cout: Weight,
    cq_norm: DeviceBuffer,
    ck_norm: DeviceBuffer,
    post_cross: DeviceBuffer,
    cross_gate: f32,
    pre_hada: DeviceBuffer,
    cond_v: Weight,
    cond_u: DeviceBuffer,
    d2: DeviceBuffer,
    b2: DeviceBuffer,
    rot: [Weight; 3],
    phi: [Weight; 3],
    a: [f32; 3],
    b: [DeviceBuffer; 3],
}

struct DEngram {
    table: DeviceBuffer,
    key: Weight,
    value: Weight,
    taps: DeviceBuffer,
}

/// The decoder weights resident on a device.
pub struct DeviceDecoder {
    cfg: MirageConfig,
    layers: Vec<DLayer>,
    engrams: Vec<DEngram>,
    embedding: DeviceBuffer,
    logits_w: Weight,
    final_norm: DeviceBuffer,
    ones_lanes: DeviceBuffer,
}

/// Cross-attention keys and values of a batch of memories, per layer.
pub struct DeviceCross {
    k: Vec<DeviceBuffer>,
    v: Vec<DeviceBuffer>,
    mask: DeviceBuffer,
    stride: usize,
}

struct LayerCache {
    k: DeviceBuffer,
    v: DeviceBuffer,
    raw: [DeviceBuffer; 3],
}

/// Everything earlier tokens leave behind, for a batch.
pub struct BatchState {
    batch: usize,
    maxt: usize,
    layers: Vec<LayerCache>,
    engram_v: Vec<DeviceBuffer>,
    ones: DeviceBuffer,
    /// Tokens fed so far, per sequence.
    pub tokens: Vec<Vec<u32>>,
}

impl DeviceDecoder {
    /// Upload the host-prepared decoder.
    pub fn upload(dev: &Device, dec: &Decoder) -> DeviceDecoder {
        let cfg = dec.cfg.clone();
        let d = cfg.dim;
        let layers = dec
            .layers
            .iter()
            .map(|l| {
                let fold = l.mlp.fold(&dec.perms);
                DLayer {
                    norm: dev.norm_weight(&l.norm),
                    qkv: [lin(dev, "dq", &l.q), lin(dev, "dk", &l.k), lin(dev, "dv", &l.v)],
                    taps: [dev.buf(&l.q_taps), dev.buf(&l.k_taps), dev.buf(&l.v_taps)],
                    q_norm: dev.norm_weight(&l.q_norm),
                    k_norm: dev.norm_weight(&l.k_norm),
                    gate: lin(dev, "dgate", &l.gate),
                    out: lin(dev, "dout", &l.out),
                    post_attn: dev.norm_weight(&l.post_attn),
                    attn_gate: ops::sigmoid(l.attn_gate),
                    cross_norm: dev.norm_weight(&l.cross_norm),
                    cq: lin(dev, "cq", &l.cq),
                    ck: lin(dev, "ck", &l.ck),
                    cv: lin(dev, "cv", &l.cv),
                    cgate: lin(dev, "cgate", &l.cgate),
                    cout: lin(dev, "cout", &l.cout),
                    cq_norm: dev.norm_weight(&l.cq_norm),
                    ck_norm: dev.norm_weight(&l.ck_norm),
                    post_cross: dev.norm_weight(&l.post_cross),
                    cross_gate: ops::sigmoid(l.cross_gate),
                    pre_hada: dev.norm_weight(&l.pre_hada),
                    cond_v: lin(dev, "cond_v", &l.mlp.cond_v),
                    cond_u: dev.buf(&l.mlp.cond_u),
                    d2: dev.buf(&l.mlp.d[1]),
                    b2: dev.buf(&l.mlp.b2),
                    rot: [dev.weight("w1", &fold[0], d, d), dev.weight("w2", &fold[1], d, d), dev.weight("w3", &fold[2], d, d)],
                    phi: [lin(dev, "phi_pre", &l.phi_pre), lin(dev, "phi_post", &l.phi_post), lin(dev, "phi_res", &l.phi_res)],
                    a: l.a,
                    b: [dev.buf(&l.b_pre), dev.buf(&l.b_post), dev.buf(&l.b_res)],
                }
            })
            .collect();
        let engrams = dec.engrams.iter().map(|e| DEngram { table: dev.buf(&e.table), key: lin(dev, "ekey", &e.key), value: lin(dev, "evalue", &e.value), taps: dev.buf(&e.taps) }).collect();
        let embedding = dev.buf(&dec.embedding.wt);
        let logits_w = Weight::dense_f32(embedding.clone(), dec.embedding.out as u32, dec.embedding.inn as u32);
        DeviceDecoder { layers, engrams, embedding, logits_w, final_norm: dev.norm_weight(&dec.final_norm), ones_lanes: dev.buf(&vec![1.0; cfg.lanes * d]), cfg }
    }

    /// Cross-attention keys and values of a memory batch.
    pub fn cross(&self, dev: &Device, mem: &MemoryBatch) -> DeviceCross {
        let (heads, hd) = (self.cfg.heads(), self.cfg.qk_dim);
        let rows = (mem.batch * mem.stride) as u32;
        let (mut k, mut v) = (Vec::new(), Vec::new());
        let mut s = Vec::new();
        for l in &self.layers {
            let k0 = dev.linear_new(&mut s, &l.ck, &mem.rows, rows);
            k.push(dev.zc_new(&mut s, &k0, &l.ck_norm, rows * heads as u32, hd as u32));
            v.push(dev.linear_new(&mut s, &l.cv, &mem.rows, rows));
            dev.flush(&mut s);
        }
        DeviceCross { k, v, mask: mem.mask.clone(), stride: mem.stride }
    }

    /// An empty state for `batch` sequences of at most `maxt` positions.
    pub fn new_state(&self, dev: &Device, batch: usize, maxt: usize) -> BatchState {
        let c = &self.cfg;
        let (qd, kd, vd) = (c.heads() * c.qk_dim, c.kv_heads * c.qk_dim, c.kv_heads * c.v_dim);
        let table = |w: usize| dev.alloc(batch * maxt * w);
        BatchState {
            batch,
            maxt,
            layers: (0..self.layers.len()).map(|_| LayerCache { k: table(kd), v: table(vd), raw: [table(qd), table(kd), table(vd)] }).collect(),
            engram_v: (0..self.engrams.len()).map(|_| table(c.dim)).collect(),
            ones: dev.buf(&vec![1.0; batch * maxt]),
            tokens: vec![Vec::new(); batch],
        }
    }

    /// Feed one token per sequence at the next position; returns the logits
    /// `[batch, vocabulary]` when `want_logits`.
    pub fn step(&self, dev: &Device, st: &mut BatchState, cross: &DeviceCross, tokens: &[u32], dec: &Decoder, want_logits: bool) -> Option<DeviceBuffer> {
        let c = &self.cfg;
        let (b, d, n) = (st.batch, c.dim, c.lanes);
        assert_eq!(tokens.len(), b);
        let pos = st.tokens[0].len();
        assert!(pos < st.maxt, "the state holds {} positions", st.maxt);
        for (t, &tok) in st.tokens.iter_mut().zip(tokens) {
            t.push(tok);
        }
        let g = dev.gpu();
        let tok_buf = g.storage(b as u64);
        g.write(&tok_buf, tokens);
        let tables = c.engram_tables();
        let rows_buf = g.storage((b * tables) as u64);
        let engram_rows: Vec<u32> = st.tokens.iter().flat_map(|t| dec.engram_rows(t, pos)).collect();
        g.write(&rows_buf, &engram_rows);

        let mut s = Vec::new();
        let (bu, du, nu) = (b as u32, d as u32, n as u32);
        let mut lanes = dev.alloc(b * n * d);
        dev.embed_lanes(&mut s, &tok_buf, &self.embedding, &lanes, (bu, nu, du), (d as f32).sqrt());
        for (l, ly) in self.layers.iter().enumerate() {
            let nx = dev.zc_new(&mut s, &lanes, &self.ones_lanes, bu, nu * du);
            let pre = dev.linear_new(&mut s, &ly.phi[0], &nx, bu);
            let post = dev.linear_new(&mut s, &ly.phi[1], &nx, bu);
            let res = dev.linear_new(&mut s, &ly.phi[2], &nx, bu);
            let maps = dev.alloc(b * (2 * n + n * n));
            let shape = MhcShape { rows: bu, n: nu, lane_id: (l % n) as u32, iters: SINKHORN_ITERATIONS as u32 };
            dev.mhc_maps(&mut s, shape, ly.a, &pre, &post, &res, [&ly.b[0], &ly.b[1], &ly.b[2]], &maps);
            let u = dev.alloc(b * d);
            dev.mhc_mix(&mut s, &lanes, &maps, &u, (bu, nu, du));
            let y = dev.alloc(b * d);
            dev.copy_rows(&mut s, &u, (1, 0), &y, (1, 0), 1, bu, du);
            self.block(dev, &mut s, l, st, cross, &y, &rows_buf, pos);
            let next = dev.alloc(b * n * d);
            dev.mhc_post(&mut s, &lanes, &y, &u, &maps, &next, (bu, nu, du));
            lanes = next;
            dev.flush(&mut s);
        }
        let out = want_logits.then(|| {
            let mean = dev.alloc(b * d);
            dev.lane_mean(&mut s, &lanes, &mean, (bu, nu, du));
            let normed = dev.zc_new(&mut s, &mean, &self.final_norm, bu, du);
            let logits = dev.linear_new(&mut s, &self.logits_w, &normed, bu);
            dev.flush(&mut s);
            logits
        });
        dev.flush(&mut s);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn block(&self, dev: &Device, s: &mut Vec<gpu_core::Step>, l: usize, st: &BatchState, cross: &DeviceCross, y: &DeviceBuffer, engram_rows: &DeviceBuffer, pos: usize) {
        let c = &self.cfg;
        let (b, d) = (st.batch as u32, c.dim as u32);
        let ly = &self.layers[l];
        let (heads, kvh, hd, vh) = (c.heads() as u32, c.kv_heads as u32, c.qk_dim as u32, c.v_dim as u32);
        let maxt = st.maxt as u32;
        if let Some(site) = c.sites.iter().position(|&x| x == l) {
            let e = &self.engrams[site];
            let (tables, sub) = (c.engram_tables() as u32, c.engram_sub_dim() as u32);
            let gathered = dev.alloc(st.batch * c.dim);
            dev.gather_rows(s, engram_rows, &e.table, &gathered, (b, tables, sub));
            let key = dev.linear_new(s, &e.key, &gathered, b);
            let value = dev.linear_new(s, &e.value, &gathered, b);
            dev.row_write(s, &value, &st.engram_v[site], (b, d, maxt), pos as u32);
            let conv = dev.alloc(st.batch * c.dim);
            dev.taps(s, &st.engram_v[site], &e.taps, &conv, (b, d, maxt), pos as u32, ENGRAM_TAPS as u32, ENGRAM_DILATION);
            dev.engram_gate(s, y, &key, &conv, b, d);
        }
        // Self-attention with causal taps and rotary positions.
        let cache = &st.layers[l];
        let h = dev.zc_new(s, y, &ly.norm, b, d);
        let widths = [heads * hd, kvh * hd, kvh * vh];
        let mut conv = Vec::new();
        for (i, &width) in widths.iter().enumerate() {
            let raw = dev.linear_new(s, &ly.qkv[i], &h, b);
            dev.row_write(s, &raw, &cache.raw[i], (b, width, maxt), pos as u32);
            let out = dev.alloc(st.batch * width as usize);
            dev.taps(s, &cache.raw[i], &ly.taps[i], &out, (b, width, maxt), pos as u32, c.taps as u32, 1);
            conv.push(out);
        }
        let q = dev.zc_new(s, &conv[0], &ly.q_norm, b * heads, hd);
        let k = dev.zc_new(s, &conv[1], &ly.k_norm, b * kvh, hd);
        dev.rope_pos(s, &q, b * heads, hd, pos as u32, c.rope);
        dev.rope_pos(s, &k, b * kvh, hd, pos as u32, c.rope);
        dev.row_write(s, &k, &cache.k, (b, kvh * hd, maxt), pos as u32);
        dev.row_write(s, &conv[2], &cache.v, (b, kvh * vh, maxt), pos as u32);
        let att = dev.alloc(st.batch * (heads * vh) as usize);
        let shape = AttnShape { batch: b, nq: 1, heads, kvh, hd, vd: vh, nk: maxt, visible: pos as u32 + 1 };
        dev.attention(s, shape, &q, &cache.k, &cache.v, &st.ones, &att);
        self.gated_residual(dev, s, y, &h, &att, &ly.gate, &ly.out, &ly.post_attn, ly.attn_gate, b, heads * vh);

        // Cross-attention over the image memory.
        let h = dev.zc_new(s, y, &ly.cross_norm, b, d);
        let q0 = dev.linear_new(s, &ly.cq, &h, b);
        let q = dev.zc_new(s, &q0, &ly.cq_norm, b * heads, hd);
        let att = dev.alloc(st.batch * (heads * vh) as usize);
        let shape = AttnShape { batch: b, nq: 1, heads, kvh: heads, hd, vd: vh, nk: cross.stride as u32, visible: cross.stride as u32 };
        dev.attention(s, shape, &q, &cross.k[l], &cross.v[l], &cross.mask, &att);
        self.gated_residual(dev, s, y, &h, &att, &ly.cgate, &ly.cout, &ly.post_cross, ly.cross_gate, b, heads * vh);

        // Hadamard MLP.
        let h = dev.zc_new(s, y, &ly.pre_hada, b, d);
        let cond = dev.linear_new(s, &ly.cond_v, &h, b);
        let z1 = dev.linear_new(s, &ly.rot[0], &h, b);
        dev.hada_mid(s, &z1, &cond, &ly.cond_u, &ly.d2, &ly.b2, (b, d, HADAMARD_COND_RANK as u32));
        let z2 = dev.linear_new(s, &ly.rot[1], &z1, b);
        let z3 = dev.linear_new(s, &ly.rot[2], &z2, b);
        dev.add(s, &z3, y, b * d);
    }

    /// `y += gate * zc((att * sigmoid(h Wgate)) Wout)`.
    #[allow(clippy::too_many_arguments)]
    fn gated_residual(&self, dev: &Device, s: &mut Vec<gpu_core::Step>, y: &DeviceBuffer, h: &DeviceBuffer, att: &DeviceBuffer, gate: &Weight, out: &Weight, post: &DeviceBuffer, scalar: f32, rows: u32, w: u32) {
        let g = dev.linear_new(s, gate, h, rows);
        let sg = dev.alloc((rows * w) as usize);
        dev.sigmoid(s, &g, &sg, rows * w);
        let gated = dev.alloc((rows * w) as usize);
        dev.mul(s, att, &sg, &gated, rows * w);
        let o = dev.linear_new(s, out, &gated, rows);
        let on = dev.zc_new(s, &o, post, rows, self.cfg.dim as u32);
        dev.axpy(s, scalar, &on, y, rows * self.cfg.dim as u32);
    }

    /// Greedy generation for a batch of sequences, one per memory, sharing the step loop.
    pub fn generate_batch(&self, dev: &Device, dec: &Decoder, mem: &MemoryBatch, tok: &Tokenizer, params: &[GenerateParams]) -> Vec<Generation> {
        let b = params.len();
        let plen = params[0].prompt.len();
        assert!(params.iter().all(|p| p.prompt.len() == plen), "a batch shares one prompt length");
        let maxt = plen + params.iter().map(|p| p.max_tokens).max().unwrap_or(0) + 1;
        let cross = self.cross(dev, mem);
        let mut st = self.new_state(dev, b, maxt);
        let vocab = self.logits_w.n() as usize;
        let ban: Vec<f32> = params
            .iter()
            .flat_map(|p| {
                let mut row = vec![0f32; vocab];
                p.banned.iter().for_each(|&t| row[t as usize] = BAN);
                row
            })
            .collect();
        let ban = dev.buf(&ban);
        let picked = dev.alloc(b);
        for pos in 0..plen {
            let toks: Vec<u32> = params.iter().map(|p| p.prompt[pos]).collect();
            let logits = self.step(dev, &mut st, &cross, &toks, dec, pos + 1 == plen);
            if let Some(logits) = logits {
                self.pick(dev, &logits, &ban, &picked, b, vocab);
            }
        }
        let eos = tok.eos_id();
        let mut out: Vec<Option<Generation>> = (0..b).map(|_| None).collect();
        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); b];
        loop {
            let next: Vec<u32> = dev.read_at(&picked, 0, b).iter().map(|&v| v as u32).collect();
            let mut feed = vec![0u32; b];
            for i in 0..b {
                if out[i].is_some() {
                    continue;
                }
                let p = &params[i];
                let finish = |generated: &Vec<u32>, stop| Generation { tokens: generated.clone(), text: render(tok, generated, p.object_slots), stop };
                if generated[i].len() >= p.max_tokens {
                    out[i] = Some(finish(&generated[i], Stop::MaxTokens));
                    continue;
                }
                generated[i].push(next[i]);
                if next[i] == eos {
                    out[i] = Some(finish(&generated[i], Stop::Eos));
                    continue;
                }
                if p.guard {
                    let text = render(tok, &generated[i], p.object_slots);
                    if let Some(stop) = guard(&text, p.max_rank) {
                        out[i] = Some(finish(&generated[i], stop));
                        continue;
                    }
                }
                feed[i] = next[i];
            }
            if out.iter().all(Option::is_some) {
                break;
            }
            let logits = self.step(dev, &mut st, &cross, &feed, dec, true).expect("logits requested");
            self.pick(dev, &logits, &ban, &picked, b, vocab);
        }
        out.into_iter().map(|g| g.expect("every sequence stopped")).collect()
    }

    /// Mask banned tokens and choose the greedy token of every row.
    fn pick(&self, dev: &Device, logits: &DeviceBuffer, ban: &DeviceBuffer, picked: &DeviceBuffer, b: usize, vocab: usize) {
        let mut s = Vec::new();
        dev.add(&mut s, ban, logits, (b * vocab) as u32);
        dev.argmax(&mut s, logits, picked, b as u32, vocab as u32);
        dev.flush(&mut s);
    }
}
