// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements incremental decoding for recurrent-depth
// decoders for its clients. If your team needs expertise in keeping one KV
// history per prelude block, per (recurrent step, block) and per coda block,
// or in proving a cached decode equals the full forward, then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Incremental decode for Huginn.
//!
//! The attention histories this model needs are indexed by *stage*, not by
//! layer: the core blocks run once per recurrent step over a different state
//! each time, so the keys written in step 3 came from another residual stream
//! than the keys the same block wrote in step 2. The stage list is
//! `prelude.0..`, then `core.<b>` for every step in order, then `coda.<b>`;
//! each stage owns one `[cap, kv_dim]` history pair.
//!
//! A decode step takes one token at position `pos` through that whole list,
//! with the recurrence starting from the model's initial latent state - the
//! same state every position starts from in the full-sequence forward. (The
//! reference's continuous-compute generation carries the previous position's
//! normed latent into the next position instead, an approximation that drifts
//! from the model's own definition; this engine does not use it, so a cached
//! step equals the corresponding row of the full forward.)
//!
//! The engine borrows the model's device and parameter store: one copy of the
//! weights regardless of how many decode states run.

use gpu_core::{BufUsage, DeviceBuffer, Gpu, Step};
use ::model::block::{self, GqaDecodeIds, KernelIds};
use paramstore::ParamStore;

use crate::config::HuginnConfig;
use crate::model::{self, Huginn};

struct Scratch {
    xn1: DeviceBuffer,
    xn2: DeviceBuffer,
    xn3: DeviceBuffer,
    qkv: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    proj: DeviceBuffer,
    tmp: DeviceBuffer,
    fc: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    mh: DeviceBuffer,
    mout: DeviceBuffer,
    concat: DeviceBuffer,
    latent: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
}

pub struct Decode<'a> {
    pub gpu: &'a Gpu,
    pub ps: &'a ParamStore,
    pub cfg: &'a HuginnConfig,
    cap: u32,
    steps: u32,
    stages: Vec<String>,
    kv_k: Vec<DeviceBuffer>,
    kv_v: Vec<DeviceBuffer>,
    token: DeviceBuffer,
    h: [DeviceBuffer; 2],
    /// The state every position's recurrence starts from.
    init_state: DeviceBuffer,
    /// A working copy of it, since the residual pair is overwritten.
    state0: DeviceBuffer,
    /// The embedding scale, one value.
    scale: DeviceBuffer,
    logits: DeviceBuffer,
    scratch: Scratch,
}

impl<'a> Decode<'a> {
    /// A decode engine over `base`'s weights, with room for `cap` positions and
    /// `steps` recurrent iterations per position.
    pub fn new(base: &'a Huginn, cap: u32, steps: u32) -> Decode<'a> {
        assert!(cap > 0 && steps > 0);
        let cfg = &base.cfg;
        let gpu = &base.gpu;
        let ps = &base.ps;
        let (d, ff) = (cfg.d_model as u64, cfg.d_ff as u64);
        let nh = cfg.n_heads as u64;
        let ncap = cap as u64;
        let st = |x: u64| gpu.storage(x);

        let mut stages = Vec::new();
        for i in 0..cfg.n_prelude {
            stages.push(format!("prelude.{i}"));
        }
        for _ in 0..steps {
            for i in 0..cfg.n_recurrent {
                stages.push(format!("core.{i}"));
            }
        }
        for i in 0..cfg.n_coda {
            stages.push(format!("coda.{i}"));
        }
        let kv_k = stages.iter().map(|_| st(ncap * d)).collect();
        let kv_v = stages.iter().map(|_| st(ncap * d)).collect();

        let token = gpu.buffer("huginn_decode_token", 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        let scale = gpu.buffer("huginn_decode_scale", 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        gpu.write_f32(&scale, &[cfg.emb_scale]);
        let init_state = st(d);
        let state0 = st(d);
        let h = [st(d), st(d)];
        let logits = st(cfg.vocab as u64);
        let scratch = Scratch {
            xn1: st(d),
            xn2: st(d),
            xn3: st(d),
            qkv: st(3 * d),
            q: st(d),
            k: st(d),
            v: st(d),
            ctx: st(d),
            proj: st(d),
            tmp: st(d),
            fc: st(2 * ff),
            gate: st(ff),
            up: st(ff),
            mh: st(ff),
            mout: st(d),
            concat: st(2 * d),
            latent: st(d),
            scores: st(nh * ncap),
            probs: st(nh * ncap),
        };
        Decode { gpu, ps, cfg, cap, steps, stages, kv_k, kv_v, token, h, init_state, state0, scale, logits, scratch }
    }

    pub fn capacity(&self) -> u32 {
        self.cap
    }

    pub fn steps(&self) -> u32 {
        self.steps
    }

    /// Attention histories this engine holds (`prelude + steps*core + coda`).
    pub fn histories(&self) -> usize {
        self.kv_k.len()
    }

    /// The latent state every position's recurrence starts from.
    pub fn set_state(&self, state: &[f32]) {
        assert_eq!(state.len(), self.cfg.d_model as usize);
        self.gpu.write_f32(&self.init_state, state);
    }

    /// One token at absolute position `pos`; returns the head's logits
    /// (`[vocab]`). The KV histories are advanced by this call.
    pub fn step(&self, token: u32, pos: u32) -> Vec<f32> {
        assert!(pos < self.cap, "position {pos} exceeds decode capacity {}", self.cap);
        self.gpu.write(&self.token, &[token]);
        let steps = self.build_steps(pos);
        self.gpu.submit(&[], &steps);
        self.gpu.read(&self.logits, self.cfg.vocab as usize)
    }

    fn build_steps(&self, pos: u32) -> Vec<Step> {
        let cfg = self.cfg;
        let gpu = self.gpu;
        let s = &self.scratch;
        let d = cfg.d_model;
        let ids = model::kernel_ids();
        let dec_ids = GqaDecodeIds {
            kv_append: model::KV_APPEND_ID,
            attn_decode_scores: model::ATTN_DECODE_SCORES_ID,
            decode_softmax: model::DECODE_SOFTMAX_ID,
            attn_decode_apply: model::ATTN_DECODE_APPLY_ID,
        };
        let copy = |src: &DeviceBuffer, dst: &DeviceBuffer, dst_stride: u32, dst_off: u32| {
            gpu.step(model::COPY_COLS_ID, &[src, dst], &[1, d, d, 0, dst_stride, dst_off], d)
        };
        let mut out = Vec::new();
        let mut stage = 0usize;

        out.push(gpu.step(model::EMBED_ID, &[&self.token, self.ps.w("tok.weight"), &self.h[0]], &[d, 1], d));
        // Embeddings enter the prelude scaled, as in the full forward.
        out.push(gpu.step(model::SCALE_ROW_ID, &[&self.h[0], &self.scale, &self.h[1]], &[d, d], d));
        // The prelude rewrites the embedding; its output for this position is
        // what every recurrent step re-injects.
        let mut cur = 1usize;
        for _ in 0..cfg.n_prelude {
            self.block(&mut out, &ids, &dec_ids, &self.stages[stage], &self.h[cur], &self.h[1 - cur], stage, pos);
            stage += 1;
            cur = 1 - cur;
        }
        // Both the injection and the starting state need their own buffers:
        // the residual pair is overwritten by the first core block. The latent
        // buffer is free until the recurrence ends, so it holds the injection.
        out.push(copy(&self.h[cur], &s.latent, d, 0));
        out.push(copy(&self.init_state, &self.state0, d, 0));
        out.push(copy(&self.state0, &self.h[0], d, 0));

        let mut state = 0usize;
        for _ in 0..self.steps {
            let mid = 1 - state;
            out.push(gpu.step(model::COPY_COLS_ID, &[&self.h[state], &s.concat], &[1, d, 2 * d, 0, 2 * d, 0], d));
            out.push(gpu.step(model::COPY_COLS_ID, &[&s.latent, &s.concat], &[1, d, 2 * d, 0, 2 * d, d], d));
            out.push(gpu.step(model::MATMUL_ID, &[&s.concat, self.ps.w("adapter.weight"), &self.h[mid]], &[1, 2 * d, d], d));
            for i in 0..cfg.n_recurrent {
                let (x_in, x_out) = if i == 0 { (&self.h[mid], &self.h[state]) } else { (&self.h[state], &self.h[1 - state]) };
                self.block(&mut out, &ids, &dec_ids, &self.stages[stage], x_in, x_out, stage, pos);
                stage += 1;
                if i > 0 {
                    state = 1 - state;
                }
            }
        }

        out.push(block::rmsnorm_fwd(gpu, &ids, &self.h[state], self.ps.w("ln_f.weight"), &s.latent, d, 1, cfg.norm_eps));
        let mut last: &DeviceBuffer = &s.latent;
        let mut ccur = 0usize;
        for _ in 0..cfg.n_coda {
            let x_out = &self.h[ccur];
            self.block(&mut out, &ids, &dec_ids, &self.stages[stage], last, x_out, stage, pos);
            stage += 1;
            last = x_out;
            ccur = 1 - ccur;
        }
        out.push(block::rmsnorm_fwd(gpu, &ids, last, self.ps.w("ln_f.weight"), &s.xn2, d, 1, cfg.norm_eps));
        out.push(gpu.step(model::MATMUL_ID, &[&s.xn2, self.ps.w("tok.weight"), &self.logits], &[1, d, cfg.vocab], cfg.vocab));
        debug_assert_eq!(stage, self.stages.len());
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        out: &mut Vec<Step>,
        ids: &KernelIds,
        dec_ids: &GqaDecodeIds,
        name: &str,
        x_in: &DeviceBuffer,
        x_out: &DeviceBuffer,
        stage: usize,
        pos: u32,
    ) {
        let cfg = self.cfg;
        let gpu = self.gpu;
        let s = &self.scratch;
        let (d, ff) = (cfg.d_model, cfg.d_ff);
        let (nh, hd) = (cfg.n_heads, cfg.head_dim);
        let w = |leaf: &str| self.ps.w(&format!("{name}.{leaf}"));

        out.push(block::rmsnorm_fwd(gpu, ids, x_in, w("ln1.weight"), &s.xn1, d, 1, cfg.norm_eps));
        out.push(gpu.step(model::MATMUL_ID, &[&s.xn1, w("attn.wqkv.weight"), &s.qkv], &[1, d, 3 * d], 3 * d));
        out.push(gpu.step(model::COPY_COLS_ID, &[&s.qkv, &s.q], &[1, d, 3 * d, 0, d, 0], d));
        out.push(gpu.step(model::COPY_COLS_ID, &[&s.qkv, &s.k], &[1, d, 3 * d, d, d, 0], d));
        out.push(gpu.step(model::COPY_COLS_ID, &[&s.qkv, &s.v], &[1, d, 3 * d, 2 * d, d, 0], d));
        out.push(gpu.step(model::BIAS_ADD_ID, &[&s.q, w("attn.q_bias")], &[1, d], d));
        out.push(gpu.step(model::BIAS_ADD_ID, &[&s.k, w("attn.k_bias")], &[1, d], d));
        out.push(rope_at(gpu, &s.q, 1, nh, hd, d, pos, cfg.rope_theta));
        out.push(rope_at(gpu, &s.k, 1, nh, hd, d, pos, cfg.rope_theta));
        out.extend(block::gqa_decode_step(
            gpu, dec_ids, nh, nh, hd, pos, self.cap, &s.q, &s.k, &s.v, &self.kv_k[stage], &self.kv_v[stage], &s.scores, &s.probs, &s.ctx,
        ));
        out.push(gpu.step(model::MATMUL_ID, &[&s.ctx, w("attn.wo.weight"), &s.proj], &[1, d, d], d));
        out.push(gpu.step(model::ADD2_ID, &[&s.proj, x_in, &s.tmp], &[d], d));
        out.push(block::rmsnorm_fwd(gpu, ids, &s.tmp, w("ln2.weight"), &s.xn2, d, 1, cfg.norm_eps));
        out.push(block::rmsnorm_fwd(gpu, ids, &s.xn2, w("ln3.weight"), &s.xn3, d, 1, cfg.norm_eps));
        out.push(gpu.step(model::MATMUL_ID, &[&s.xn3, w("mlp.fc.weight"), &s.fc], &[1, d, 2 * ff], 2 * ff));
        out.push(gpu.step(model::COPY_COLS_ID, &[&s.fc, &s.gate], &[1, ff, 2 * ff, 0, ff, 0], ff));
        out.push(gpu.step(model::COPY_COLS_ID, &[&s.fc, &s.up], &[1, ff, 2 * ff, ff, ff, 0], ff));
        out.push(gpu.step(model::SILU_MUL_ID, &[&s.gate, &s.up, &s.mh], &[ff], ff));
        out.push(gpu.step(model::MATMUL_ID, &[&s.mh, w("mlp.proj.weight"), &s.mout], &[1, ff, d], d));
        out.push(gpu.step(model::ADD2_ID, &[&s.mout, &s.xn2, &s.tmp], &[d], d));
        out.push(block::rmsnorm_fwd(gpu, ids, &s.tmp, w("ln4.weight"), x_out, d, 1, cfg.norm_eps));
    }
}

fn rope_at(g: &Gpu, buf: &DeviceBuffer, rows: u32, n_heads: u32, head_dim: u32, row_stride: u32, pos: u32, theta: f32) -> Step {
    g.step(model::ROPE_AT_ID, &[buf], &[rows, n_heads, head_dim, row_stride, 0, pos, gpu_core::f(theta)], rows * n_heads * (head_dim / 2))
}
