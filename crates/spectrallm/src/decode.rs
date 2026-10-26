// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Incremental decoding: one new token per step against per-layer key/value
//! caches, the same block as the training graph at one row. The caches are
//! `[block_size, kv_dim]` per layer and, the residual stream being `m`
//! coefficients wide, hold `m`-wide coefficient states projected to keys and
//! values.

use gpu_core::{BufUsage, DeviceBuffer, Step};
use model::block;

use crate::kernels::{decode_ids, ADD2, EMBED, ROPE_AT};
use crate::mixer::MixerBufs;
use crate::model::SpectralLm;

pub(crate) struct Decode {
    cap: u32,
    pos: u32,
    tok: DeviceBuffer,
    /// Ping-pong residual stream: layer `l` reads `x[l % 2]`, writes the other.
    x: [DeviceBuffer; 2],
    xn1: DeviceBuffer,
    q_pre: DeviceBuffer,
    k_pre: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    proj: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    mlp_out: DeviceBuffer,
    xf: DeviceBuffer,
    logits: DeviceBuffer,
    kcache: Vec<DeviceBuffer>,
    vcache: Vec<DeviceBuffer>,
    mixer: MixerBufs,
}

impl Decode {
    fn new(m: &SpectralLm) -> Decode {
        let cfg = &m.cfg;
        let g = &m.gpu;
        let st = |x: u32| g.storage(x as u64);
        let cap = cfg.block_size;
        let (d, hq, hkv) = (cfg.d_model, cfg.q_dim(), cfg.kv_dim());
        Decode {
            cap,
            pos: 0,
            tok: g.buffer("decode_token", 4, BufUsage::STORAGE | BufUsage::COPY_DST),
            x: [st(d), st(d)],
            xn1: st(d),
            q_pre: st(hq),
            k_pre: st(hkv),
            q: st(hq),
            k: st(hkv),
            v: st(hkv),
            scores: st(cfg.n_heads * cap),
            probs: st(cfg.n_heads * cap),
            ctx: st(hq),
            proj: st(d),
            xmid: st(d),
            xn2: st(d),
            mlp_out: st(d),
            xf: st(d),
            logits: st(cfg.vocab),
            kcache: (0..cfg.n_layers).map(|_| st(cap * hkv)).collect(),
            vcache: (0..cfg.n_layers).map(|_| st(cap * hkv)).collect(),
            mixer: MixerBufs::new(g, cfg, 1, false),
        }
    }
}

impl SpectralLm {
    /// Forget the cached context; the next [`Self::step`] decodes position 0.
    pub fn reset_cache(&self) {
        if let Some(dec) = self.dec.borrow_mut().as_mut() {
            dec.pos = 0;
        }
    }

    /// Positions currently held in the cache.
    pub fn cache_pos(&self) -> u32 {
        self.dec.borrow().as_ref().map_or(0, |d| d.pos)
    }

    /// Append `token` at the next position and return the logits predicting
    /// the position after it.
    pub fn step(&self, token: u32) -> Vec<f32> {
        assert!(!self.is_streamed(), "spectrallm: a layer-streamed build holds a window of its layers; save it and load the checkpoint to decode");
        if self.dec.borrow().is_none() {
            *self.dec.borrow_mut() = Some(Decode::new(self));
        }
        let mut guard = self.dec.borrow_mut();
        let dec = guard.as_mut().unwrap();
        assert!(dec.pos < dec.cap, "spectrallm: decode position {} exceeds the cache of {}", dec.pos, dec.cap);
        let cfg = &self.cfg;
        let (ids, did) = (&self.ids, decode_ids());
        let (d, eps, pos) = (cfg.d_model, cfg.rms_eps, dec.pos);
        let (hq, hkv, nh, nkv, hd) = (cfg.q_dim(), cfg.kv_dim(), cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let g = &self.gpu;
        g.write(&dec.tok, &[token]);
        let mut s: Vec<Step> = vec![g.step(EMBED, &[&dec.tok, self.ps.w("tok.weight"), &dec.x[0]], &[d, 1], d)];
        for l in 0..cfg.n_layers as usize {
            let w = |name: &str| self.ps.w(&format!("blocks.{l}.{name}"));
            let (x_in, x_out) = (&dec.x[l % 2], &dec.x[(l + 1) % 2]);
            s.push(block::rmsnorm_fwd(g, ids, x_in, w("ln1.weight"), &dec.xn1, d, 1, eps));
            self.linear(&mut s, &dec.xn1, w("attn.wq.weight"), &dec.q_pre, 1, d, hq);
            self.linear(&mut s, &dec.xn1, w("attn.wk.weight"), &dec.k_pre, 1, d, hkv);
            self.linear(&mut s, &dec.xn1, w("attn.wv.weight"), &dec.v, 1, d, hkv);
            s.push(block::rmsnorm_fwd(g, ids, &dec.q_pre, w("attn.q_norm.weight"), &dec.q, hd, nh, eps));
            s.push(block::rmsnorm_fwd(g, ids, &dec.k_pre, w("attn.k_norm.weight"), &dec.k, hd, nkv, eps));
            s.push(g.step(ROPE_AT, &[&dec.q], &[1, nh, hd, hq, 0, pos, gpu_core::f(cfg.rope_theta)], nh * (hd / 2)));
            s.push(g.step(ROPE_AT, &[&dec.k], &[1, nkv, hd, hkv, 0, pos, gpu_core::f(cfg.rope_theta)], nkv * (hd / 2)));
            s.extend(block::gqa_decode_step(
                g, &did, nh, nkv, hd, pos, dec.cap, &dec.q, &dec.k, &dec.v, &dec.kcache[l], &dec.vcache[l], &dec.scores, &dec.probs, &dec.ctx,
            ));
            self.linear(&mut s, &dec.ctx, w("attn.wo.weight"), &dec.proj, 1, hq, d);
            s.push(g.step(ADD2, &[x_in, &dec.proj, &dec.xmid], &[d], d));
            s.push(block::rmsnorm_fwd(g, ids, &dec.xmid, w("ln2.weight"), &dec.xn2, d, 1, eps));
            self.mixer_fwd(&mut s, l, &dec.xn2, &dec.mlp_out, &dec.mixer, 1);
            s.push(g.step(ADD2, &[&dec.xmid, &dec.mlp_out, x_out], &[d], d));
        }
        let last = &dec.x[cfg.n_layers as usize % 2];
        s.push(block::rmsnorm_fwd(g, ids, last, self.ps.w("norm.weight"), &dec.xf, d, 1, eps));
        self.linear(&mut s, &dec.xf, self.head_weight(), &dec.logits, 1, d, cfg.vocab);
        g.submit(&[], &s);
        dec.pos += 1;
        g.read(&dec.logits, cfg.vocab as usize)
    }
}
