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

/// Decoder steps between host round trips in a generation loop.
pub const WINDOW: usize = 16;

/// Rows of several `[out, in]` linears sharing one input, as one weight.
fn cat(dev: &Device, name: &str, parts: &[&ops::Linear]) -> Weight {
    let inn = parts[0].inn;
    assert!(parts.iter().all(|p| p.inn == inn));
    let wt: Vec<f32> = parts.iter().flat_map(|p| p.wt.iter().copied()).collect();
    dev.weight(name, &wt, parts.iter().map(|p| p.out).sum(), inn)
}

fn lin(dev: &Device, name: &str, l: &ops::Linear) -> Weight {
    dev.weight(name, &l.wt, l.out, l.inn)
}

struct DLayer {
    norm: DeviceBuffer,
    qkvg: Weight,
    taps: [DeviceBuffer; 3],
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
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
    cond_rot: Weight,
    cond_u: DeviceBuffer,
    d2: DeviceBuffer,
    b2: DeviceBuffer,
    rot: [Weight; 2],
    phi: Weight,
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
    /// Token history `[batch, maxt]` (u32): the prompt or teacher tokens, then the tokens the device picked.
    hist: DeviceBuffer,
    /// Positions fed so far.
    pos: usize,
    /// The position as a device word (`u32`), advanced by every step so a recorded step can be replayed.
    posbuf: DeviceBuffer,
    /// First position whose picked token is appended to the history (the prompt length).
    pub first: u32,
}

impl BatchState {
    /// Count `steps` replayed positions.
    pub fn advance(&mut self, steps: usize) {
        self.pos += steps;
    }

    /// Positions fed so far.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// The token history table as `[batch][maxt]` words.
    pub fn history(&self, dev: &Device) -> Vec<Vec<u32>> {
        let all = dev.read_at(&self.hist, 0, self.batch * self.maxt);
        all.chunks_exact(self.maxt).map(|r| r.iter().map(|w| w.to_bits()).collect()).collect()
    }

    /// Write `tokens[b]` into the first positions of every history row.
    pub fn set_tokens(&self, dev: &Device, tokens: &[Vec<u32>]) {
        let mut flat = vec![0u32; self.batch * self.maxt];
        for (row, t) in flat.chunks_exact_mut(self.maxt).zip(tokens) {
            assert!(t.len() <= self.maxt, "{} tokens exceed the {} positions of the state", t.len(), self.maxt);
            row[..t.len()].copy_from_slice(t);
        }
        dev.gpu().write(&self.hist, &flat);
    }
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
                    qkvg: cat(dev, "dqkvg", &[&l.q, &l.k, &l.v, &l.gate]),
                    taps: [dev.buf(&l.q_taps), dev.buf(&l.k_taps), dev.buf(&l.v_taps)],
                    q_norm: dev.norm_weight(&l.q_norm),
                    k_norm: dev.norm_weight(&l.k_norm),
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
                    cond_rot: dev.weight("cond_rot", &[l.mlp.cond_v.wt.clone(), fold[0].clone()].concat(), HADAMARD_COND_RANK + d, d),
                    cond_u: dev.buf(&l.mlp.cond_u),
                    d2: dev.buf(&l.mlp.d[1]),
                    b2: dev.buf(&l.mlp.b2),
                    rot: [dev.weight("w2", &fold[1], d, d), dev.weight("w3", &fold[2], d, d)],
                    phi: cat(dev, "phi", &[&l.phi_pre, &l.phi_post, &l.phi_res]),
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
            hist: dev.alloc(batch * maxt),
            pos: 0,
            posbuf: dev.gpu().storage(1),
            first: 0,
        }
    }

    /// Feed the history token at the next position for every sequence; returns the logits
    /// `[batch, vocabulary]` when `want_logits`. With `pick = (ban, picked)` the same submission adds `ban` to the
    /// logits, writes each row's greedy token (as `f32`) into `picked` and appends it to the history, so a
    /// step costs one submission and no host round trip. Engram hashing runs on the device from the history.
    pub fn step(&self, dev: &Device, st: &mut BatchState, cross: &DeviceCross, want_logits: bool, pick: Option<(&DeviceBuffer, &DeviceBuffer)>) -> Option<DeviceBuffer> {
        let (mut s, logits) = self.build_step(dev, st, cross, want_logits, pick);
        st.pos += 1;
        dev.flush(&mut s);
        logits
    }

    /// Record one step as a replayable tape: nothing in it depends on the position except through the device
    /// word `posbuf`, which the step itself advances. Replay it once per position with `Gpu::replay_tape`
    /// and count the replays with [`BatchState::advance`].
    pub fn record_step(&self, dev: &Device, st: &BatchState, cross: &DeviceCross, want_logits: bool, pick: Option<(&DeviceBuffer, &DeviceBuffer)>) -> gpu_core::tape::Tape {
        let g = dev.gpu();
        g.set_declines_library(true);
        g.begin_tape();
        let (s, _) = self.build_step(dev, st, cross, want_logits, pick);
        g.submit(&[], &s);
        let tape = g.end_tape();
        g.set_declines_library(false);
        tape
    }

    fn build_step(&self, dev: &Device, st: &BatchState, cross: &DeviceCross, want_logits: bool, pick: Option<(&DeviceBuffer, &DeviceBuffer)>) -> (Vec<gpu_core::Step>, Option<DeviceBuffer>) {
        let c = &self.cfg;
        let (b, d, n) = (st.batch, c.dim, c.lanes);
        let pos = st.pos;
        assert!(pos + 1 < st.maxt, "the state holds {} positions", st.maxt);
        let g = dev.gpu();
        let tables = c.engram_tables();
        let tok_buf = g.storage(b as u64);
        let rows_buf = g.storage((b * tables) as u64);

        let mut s = Vec::new();
        let (bu, du, nu) = (b as u32, d as u32, n as u32);
        dev.tok_at(&mut s, &st.hist, &tok_buf, &st.posbuf, bu, st.maxt as u32);
        dev.engram_rows(&mut s, &st.hist, &rows_buf, &st.posbuf, (bu, st.maxt as u32), (tables as u32, c.hashes as u32, c.slots as u32), &c.orders);
        let mut lanes = dev.alloc(b * n * d);
        dev.embed_lanes(&mut s, &tok_buf, &self.embedding, &lanes, (bu, nu, du), (d as f32).sqrt());
        for (l, ly) in self.layers.iter().enumerate() {
            let nx = dev.zc_new(&mut s, &lanes, &self.ones_lanes, bu, nu * du);
            let phi = dev.linear_new(&mut s, &ly.phi, &nx, bu);
            let maps = dev.alloc(b * (2 * n + n * n));
            let shape = MhcShape { rows: bu, n: nu, lane_id: (l % n) as u32, iters: SINKHORN_ITERATIONS as u32 };
            dev.mhc_maps(&mut s, shape, ly.a, &phi, [&ly.b[0], &ly.b[1], &ly.b[2]], &maps);
            let u = dev.alloc(b * d);
            dev.mhc_mix(&mut s, &lanes, &maps, &u, (bu, nu, du));
            let y = dev.alloc(b * d);
            dev.copy_rows(&mut s, &u, (1, 0), &y, (1, 0), 1, bu, du);
            self.block(dev, &mut s, l, st, cross, &y, &rows_buf);
            let next = dev.alloc(b * n * d);
            dev.mhc_post(&mut s, &lanes, &y, &u, &maps, &next, (bu, nu, du));
            lanes = next;
        }
        let out = want_logits.then(|| {
            let mean = dev.alloc(b * d);
            dev.lane_mean(&mut s, &lanes, &mean, (bu, nu, du));
            let normed = dev.zc_new(&mut s, &mean, &self.final_norm, bu, du);
            let logits = dev.linear_new(&mut s, &self.logits_w, &normed, bu);
            if let Some((ban, picked)) = pick {
                dev.add(&mut s, ban, &logits, bu * self.logits_w.n());
                dev.argmax(&mut s, &logits, picked, bu, self.logits_w.n());
                dev.tok_put(&mut s, picked, &st.hist, &st.posbuf, bu, st.maxt as u32, st.first);
            }
            logits
        });
        dev.pos_inc(&mut s, &st.posbuf);
        (s, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn block(&self, dev: &Device, s: &mut Vec<gpu_core::Step>, l: usize, st: &BatchState, cross: &DeviceCross, y: &DeviceBuffer, engram_rows: &DeviceBuffer) {
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
            let conv = dev.alloc(st.batch * c.dim);
            dev.taps(s, &value, &st.engram_v[site], &e.taps, &conv, &st.posbuf, (b, d, maxt), ENGRAM_TAPS as u32, ENGRAM_DILATION, (d, 0));
            dev.engram_gate(s, y, &key, &conv, b, d);
        }
        // Self-attention with causal taps and rotary positions.
        let cache = &st.layers[l];
        let h = dev.zc_new(s, y, &ly.norm, b, d);
        let widths = [heads * hd, kvh * hd, kvh * vh];
        // One fused projection: q, k, v, then the attention gate.
        let qkvg = dev.linear_new(s, &ly.qkvg, &h, b);
        let stride = widths.iter().sum::<u32>() + heads * vh;
        let mut conv = Vec::new();
        let mut off = 0;
        for (i, &width) in widths.iter().enumerate() {
            let out = dev.alloc(st.batch * width as usize);
            dev.taps(s, &qkvg, &cache.raw[i], &ly.taps[i], &out, &st.posbuf, (b, width, maxt), c.taps as u32, 1, (stride, off));
            conv.push(out);
            off += width;
        }
        let q = dev.zc_new(s, &conv[0], &ly.q_norm, b * heads, hd);
        let k = dev.zc_new(s, &conv[1], &ly.k_norm, b * kvh, hd);
        dev.rope_pos(s, &q, &st.posbuf, b * heads, hd, c.rope);
        dev.rope_pos(s, &k, &st.posbuf, b * kvh, hd, c.rope);
        dev.row_write(s, &k, &cache.k, &st.posbuf, (b, kvh * hd, maxt));
        dev.row_write(s, &conv[2], &cache.v, &st.posbuf, (b, kvh * vh, maxt));
        let att = dev.alloc(st.batch * (heads * vh) as usize);
        let shape = AttnShape { batch: b, nq: 1, heads, kvh, hd, vd: vh, nk: maxt, visible: u32::MAX };
        dev.attention_at(s, shape, &q, &cache.k, &cache.v, &st.ones, &st.posbuf, &att);
        self.gated_residual(dev, s, y, (&qkvg, stride, off), &att, &ly.out, &ly.post_attn, ly.attn_gate, b, heads * vh);

        // Cross-attention over the image memory.
        let h = dev.zc_new(s, y, &ly.cross_norm, b, d);
        let q0 = dev.linear_new(s, &ly.cq, &h, b);
        let q = dev.zc_new(s, &q0, &ly.cq_norm, b * heads, hd);
        let att = dev.alloc(st.batch * (heads * vh) as usize);
        let shape = AttnShape { batch: b, nq: 1, heads, kvh: heads, hd, vd: vh, nk: cross.stride as u32, visible: cross.stride as u32 };
        dev.attention(s, shape, &q, &cross.k[l], &cross.v[l], &cross.mask, &att);
        let cg = dev.linear_new(s, &ly.cgate, &h, b);
        self.gated_residual(dev, s, y, (&cg, heads * vh, 0), &att, &ly.cout, &ly.post_cross, ly.cross_gate, b, heads * vh);

        // Hadamard MLP.
        let h = dev.zc_new(s, y, &ly.pre_hada, b, d);
        let zc = dev.linear_new(s, &ly.cond_rot, &h, b);
        let z1 = dev.alloc(st.batch * c.dim);
        dev.hada_mid(s, &zc, &ly.cond_u, &ly.d2, &ly.b2, &z1, (b, d, HADAMARD_COND_RANK as u32));
        let z2 = dev.linear_new(s, &ly.rot[0], &z1, b);
        let z3 = dev.linear_new(s, &ly.rot[1], &z2, b);
        dev.add(s, &z3, y, b * d);
    }

    /// `y += gate * zc((att * sigmoid(gate)) Wout)`; `gate` is (buffer, row stride, column offset).
    #[allow(clippy::too_many_arguments)]
    fn gated_residual(&self, dev: &Device, s: &mut Vec<gpu_core::Step>, y: &DeviceBuffer, gate: (&DeviceBuffer, u32, u32), att: &DeviceBuffer, out: &Weight, post: &DeviceBuffer, scalar: f32, rows: u32, w: u32) {
        let gated = dev.alloc((rows * w) as usize);
        dev.sigmoid_mul(s, att, gate.0, &gated, rows * w, w, (gate.1, gate.2));
        let o = dev.linear_new(s, out, &gated, rows);
        let on = dev.zc_new(s, &o, post, rows, self.cfg.dim as u32);
        dev.axpy(s, scalar, &on, y, rows * self.cfg.dim as u32);
    }

    /// Greedy generation for a batch of sequences, one per memory, sharing the step loop.
    ///
    /// The device feeds every picked token back into the next step itself ([`WINDOW`] steps per host round
    /// trip). The host then walks the window's tokens in order and applies the stop rules (end token, budget,
    /// loop guard) exactly as a token-by-token loop would; steps taken past a sequence's stop are discarded.
    pub fn generate_batch(&self, dev: &Device, mem: &MemoryBatch, tok: &Tokenizer, params: &[GenerateParams]) -> Vec<Generation> {
        self.generate_batch_until(dev, mem, tok, params, &|_| false)
    }

    /// [`Self::generate_batch`] where `cancelled(i)` is asked for every live sequence after each window: a cancelled
    /// sequence stops where it is (its partial generation is returned, for the caller to drop) and the loop ends once
    /// every sequence has stopped, so a batch that is entirely cancelled stops costing device steps.
    pub fn generate_batch_until(&self, dev: &Device, mem: &MemoryBatch, tok: &Tokenizer, params: &[GenerateParams], cancelled: &dyn Fn(usize) -> bool) -> Vec<Generation> {
        let b = params.len();
        let plen = params[0].prompt.len();
        assert!(params.iter().all(|p| p.prompt.len() == plen), "a batch shares one prompt length");
        let budget = params.iter().map(|p| p.max_tokens).max().unwrap_or(0);
        let maxt = plen + budget + WINDOW + 2;
        let cross = self.cross(dev, mem);
        let mut st = self.new_state(dev, b, maxt);
        st.set_tokens(dev, &params.iter().map(|p| p.prompt.to_vec()).collect::<Vec<_>>());
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
        st.first = plen as u32;
        // Record one full step once and replay it for every position: the position lives in a device word.
        let tape = self.record_step(dev, &st, &cross, true, Some((&ban, &picked)));
        for _ in 0..plen {
            dev.gpu().replay_tape(&tape);
        }
        st.advance(plen);
        let eos = tok.eos_id();
        let mut out: Vec<Option<Generation>> = (0..b).map(|_| None).collect();
        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); b];
        let mut consumed = 0usize;
        loop {
            // Token `consumed` of every sequence is already in the history; run a window of steps, then read it.
            let take = WINDOW.min(budget + 1 - consumed.min(budget));
            for _ in 0..take {
                dev.gpu().replay_tape(&tape);
            }
            st.advance(take);
            let history = st.history(dev);
            for i in 0..b {
                for m in consumed..consumed + take {
                    if out[i].is_some() {
                        break;
                    }
                    let p = &params[i];
                    let finish = |generated: &Vec<u32>, stop| Generation { tokens: generated.clone(), text: render(tok, generated, p.object_slots), stop };
                    if generated[i].len() >= p.max_tokens {
                        out[i] = Some(finish(&generated[i], Stop::MaxTokens));
                        break;
                    }
                    let next = history[i][plen + m];
                    generated[i].push(next);
                    if next == eos {
                        out[i] = Some(finish(&generated[i], Stop::Eos));
                        break;
                    }
                    if p.guard {
                        let text = render(tok, &generated[i], p.object_slots);
                        if let Some(stop) = guard(&text, p.max_rank) {
                            out[i] = Some(finish(&generated[i], stop));
                            break;
                        }
                    }
                }
            }
            consumed += take;
            for i in 0..b {
                if out[i].is_none() && cancelled(i) {
                    out[i] = Some(Generation { tokens: generated[i].clone(), text: render(tok, &generated[i], params[i].object_slots), stop: Stop::MaxTokens });
                }
            }
            if out.iter().all(Option::is_some) {
                break;
            }
            if consumed > budget {
                // Every unfinished sequence has spent its own budget (<= budget): finish them.
                for i in 0..b {
                    if out[i].is_none() {
                        out[i] = Some(Generation { tokens: generated[i].clone(), text: render(tok, &generated[i], params[i].object_slots), stop: Stop::MaxTokens });
                    }
                }
                break;
            }
        }
        out.into_iter().map(|g| g.expect("every sequence stopped")).collect()
    }
}
