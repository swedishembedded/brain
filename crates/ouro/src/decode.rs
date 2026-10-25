// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements incremental decoding for recurrent-depth
// language models for its clients. If your team needs expertise in keeping one
// KV history per recurrent pass, or in proving a cached decode step equals the
// full forward, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Incremental decode for Ouro.
//!
//! A shared stack executed for `T` passes needs `T` attention histories: a
//! key/value row written in pass `p` came from a different residual state than
//! the row at the same position in pass `p-1`, so the reference keys its cache
//! by `pass * layers + layer`. This module keeps exactly that layout: one
//! `[cap, kv_dim]` cache pair per `(pass, layer)`.
//!
//! One decode step processes one new token through every pass. Pass 0 starts
//! from the token's embedding; pass `p` starts from pass `p-1`'s final-normed
//! hidden state, and each pass appends its own K/V at the absolute position.
//! The token-by-token result is gated against the full-sequence forward in
//! `tests/decode.rs`.
//!
//! The decode engine borrows the base model's device and parameter store, so a
//! served model holds one copy of the weights regardless of how many decode
//! states it runs.

use std::cell::Cell;

use gpu_core::{BufUsage, DeviceBuffer, Gpu, Step};
use ::model::block;
use paramstore::ParamStore;

use crate::config::OuroConfig;
use crate::model::{self, Ouro};

struct Scratch {
    xn1: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    proj: DeviceBuffer,
    xn1_2: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    mlp_h: DeviceBuffer,
    mlp_out: DeviceBuffer,
    xn2_2: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
}

pub struct Decode<'a> {
    pub gpu: &'a Gpu,
    pub ps: &'a ParamStore,
    pub cfg: &'a OuroConfig,
    cap: u32,
    /// `(pass, layer)` KV histories, `[cap * kv_dim]` each.
    kv_k: Vec<Vec<DeviceBuffer>>,
    kv_v: Vec<Vec<DeviceBuffer>>,
    token: DeviceBuffer,
    h: [DeviceBuffer; 2],
    hs: Vec<DeviceBuffer>,
    gates: Vec<DeviceBuffer>,
    logits: DeviceBuffer,
    scratch: Scratch,
    /// Pass selected by the last step.
    exit: Cell<u32>,
}

impl<'a> Decode<'a> {
    /// A decode engine over `base`'s weights, with room for `cap` positions.
    pub fn new(base: &'a Ouro, cap: u32) -> Decode<'a> {
        let cfg = &base.cfg;
        assert!(cap > 0);
        let gpu = &base.gpu;
        let ps = &base.ps;
        let (d, ff) = (cfg.d_model as u64, cfg.d_ff as u64);
        let (hq, hkv, nh) = (cfg.q_dim() as u64, cfg.kv_dim() as u64, cfg.n_heads as u64);
        let ncap = cap as u64;
        let st = |x: u64| gpu.storage(x);

        let mut kv_k = Vec::new();
        let mut kv_v = Vec::new();
        for _ in 0..cfg.total_ut_steps {
            let mut pk = Vec::new();
            let mut pv = Vec::new();
            for _ in 0..cfg.n_layers {
                pk.push(st(ncap * hkv));
                pv.push(st(ncap * hkv));
            }
            kv_k.push(pk);
            kv_v.push(pv);
        }

        let token = gpu.buffer("decode_token", 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        let h = [st(d), st(d)];
        let hs = (0..cfg.total_ut_steps).map(|_| st(d)).collect();
        let gates = (0..cfg.total_ut_steps).map(|_| st(1)).collect();
        let logits = st(cfg.vocab as u64);
        let scratch = Scratch {
            xn1: st(d),
            q: st(hq),
            k: st(hkv),
            v: st(hkv),
            ctx: st(hq),
            proj: st(d),
            xn1_2: st(d),
            xmid: st(d),
            xn2: st(d),
            gate_pre: st(ff),
            up: st(ff),
            mlp_h: st(ff),
            mlp_out: st(d),
            xn2_2: st(d),
            scores: st(nh * ncap),
            probs: st(nh * ncap),
        };
        Decode { gpu, ps, cfg, cap, kv_k, kv_v, token, h, hs, gates, logits, scratch, exit: Cell::new(0) }
    }

    pub fn capacity(&self) -> u32 {
        self.cap
    }

    pub fn selected_pass(&self) -> u32 {
        self.exit.get()
    }

    /// One token at absolute position `pos`; returns the selected pass's logits
    /// (`[vocab]`). The KV histories are advanced by this call.
    pub fn step(&self, token: u32, pos: u32) -> Vec<f32> {
        assert!(pos < self.cap, "position {pos} exceeds decode capacity {}", self.cap);
        self.gpu.write(&self.token, &[token]);
        let steps = self.build_steps(pos);
        self.gpu.submit(&[], &steps);
        let pass = self.choose_exit();
        self.exit.set(pass);
        self.gpu.submit(&[], &self.head_steps(pass as usize));
        self.gpu.read(&self.logits, self.cfg.vocab as usize)
    }

    /// The exit-gate logits of `pass` after the last step (pre-sigmoid).
    pub fn read_gate(&self, pass: usize) -> f32 {
        self.gpu.read(&self.gates[pass], 1)[0]
    }

    /// The final-normed hidden state of `pass` after the last step.
    pub fn read_hidden(&self, pass: usize) -> Vec<f32> {
        self.gpu.read(&self.hs[pass], self.cfg.d_model as usize)
    }

    fn build_steps(&self, pos: u32) -> Vec<Step> {
        let cfg = self.cfg;
        let ids = model::kernel_ids();
        let dec_ids = model::decode_ids();
        let gpu = self.gpu;
        let s = &self.scratch;
        let (d, ff) = (cfg.d_model, cfg.d_ff);
        let (hq, hkv, nh, nkv, hd) = (cfg.q_dim(), cfg.kv_dim(), cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let t = cfg.total_ut_steps as usize;
        let l_count = cfg.n_layers as usize;

        let mut steps = vec![gpu.step(model::EMBED_ID, &[&self.token, self.ps.w("tok.weight"), &self.h[0]], &[d, 1], d)];
        for p in 0..t {
            // Layer outputs ping-pong between two buffers. For pass 0 the first
            // input is the embedding buffer, so the first output must be the
            // other one; later passes start from `hs[p-1]` and may write h[0].
            let mut prev: &DeviceBuffer = if p == 0 { &self.h[0] } else { &self.hs[p - 1] };
            for l in 0..l_count {
                let out: &DeviceBuffer = if p == 0 { &self.h[(l + 1) % 2] } else { &self.h[l % 2] };
                let pfx = |name: &str| format!("blocks.{l}.{name}");
                steps.push(block::rmsnorm_fwd(gpu, &ids, prev, self.ps.w(&pfx("ln1.weight")), &s.xn1, d, 1, cfg.norm_eps));
                steps.push(gpu.step(model::MATMUL_ID, &[&s.xn1, self.ps.w(&pfx("attn.wq.weight")), &s.q], &[1, d, hq], hq));
                steps.push(gpu.step(model::MATMUL_ID, &[&s.xn1, self.ps.w(&pfx("attn.wk.weight")), &s.k], &[1, d, hkv], hkv));
                steps.push(gpu.step(model::MATMUL_ID, &[&s.xn1, self.ps.w(&pfx("attn.wv.weight")), &s.v], &[1, d, hkv], hkv));
                steps.push(rope_at(gpu, &s.q, 1, nh, hd, hq, pos, cfg.rope_theta));
                steps.push(rope_at(gpu, &s.k, 1, nkv, hd, hkv, pos, cfg.rope_theta));
                steps.extend(block::gqa_decode_step(
                    gpu, &dec_ids, nh, nkv, hd, pos, self.cap, &s.q, &s.k, &s.v, &self.kv_k[p][l], &self.kv_v[p][l], &s.scores, &s.probs, &s.ctx,
                ));
                steps.push(gpu.step(model::MATMUL_ID, &[&s.ctx, self.ps.w(&pfx("attn.wo.weight")), &s.proj], &[1, hq, d], d));
                steps.push(block::rmsnorm_fwd(gpu, &ids, &s.proj, self.ps.w(&pfx("ln1_2.weight")), &s.xn1_2, d, 1, cfg.norm_eps));
                steps.push(gpu.step(model::ADD2_ID, &[prev, &s.xn1_2, &s.xmid], &[d], d));
                steps.push(block::rmsnorm_fwd(gpu, &ids, &s.xmid, self.ps.w(&pfx("ln2.weight")), &s.xn2, d, 1, cfg.norm_eps));
                steps.push(gpu.step(model::MATMUL_ID, &[&s.xn2, self.ps.w(&pfx("mlp.gate.weight")), &s.gate_pre], &[1, d, ff], ff));
                steps.push(gpu.step(model::MATMUL_ID, &[&s.xn2, self.ps.w(&pfx("mlp.up.weight")), &s.up], &[1, d, ff], ff));
                steps.push(block::swiglu_fwd(gpu, &ids, &s.gate_pre, &s.up, &s.mlp_h, ff));
                steps.push(gpu.step(model::MATMUL_ID, &[&s.mlp_h, self.ps.w(&pfx("mlp.down.weight")), &s.mlp_out], &[1, ff, d], d));
                steps.push(block::rmsnorm_fwd(gpu, &ids, &s.mlp_out, self.ps.w(&pfx("ln2_2.weight")), &s.xn2_2, d, 1, cfg.norm_eps));
                steps.push(gpu.step(model::ADD2_ID, &[&s.xmid, &s.xn2_2, out], &[d], d));
                prev = out;
            }
            steps.push(block::rmsnorm_fwd(gpu, &ids, prev, self.ps.w("norm.weight"), &self.hs[p], d, 1, cfg.norm_eps));
            steps.push(gpu.step(model::MATMUL_ID, &[&self.hs[p], self.ps.w("gate.weight"), &self.gates[p]], &[1, d, 1], 1));
            steps.push(gpu.step(model::BIAS_ADD_ID, &[&self.gates[p], self.ps.w("gate.bias")], &[1, 1], 1));
        }
        steps
    }

    fn head_steps(&self, pass: usize) -> Vec<Step> {
        let (d, v) = (self.cfg.d_model, self.cfg.vocab);
        vec![self.gpu.step(model::MATMUL_ID, &[&self.hs[pass], self.ps.w("lm_head.weight"), &self.logits], &[1, d, v], v)]
    }

    /// The reference's CDF rule over the one row.
    fn choose_exit(&self) -> u32 {
        let t = self.cfg.total_ut_steps;
        let mut remaining = 1.0f64;
        let mut cdf = 0.0f64;
        let mut out = t - 1;
        for pass in 0..t {
            let gate = self.gpu.read(&self.gates[pass as usize], 1)[0] as f64;
            let lambda = 1.0 / (1.0 + (-gate).exp());
            let p = if pass + 1 < t {
                let v = lambda * remaining;
                remaining *= 1.0 - lambda;
                v
            } else {
                remaining
            };
            cdf += p;
            if cdf >= self.cfg.early_exit_threshold as f64 {
                out = pass;
            }
        }
        out
    }
}

fn rope_at(g: &Gpu, buf: &DeviceBuffer, rows: u32, n_heads: u32, head_dim: u32, row_stride: u32, pos: u32, theta: f32) -> Step {
    g.step(model::ROPE_AT_ID, &[buf], &[rows, n_heads, head_dim, row_stride, 0, pos, gpu_core::f(theta)], rows * n_heads * (head_dim / 2))
}
