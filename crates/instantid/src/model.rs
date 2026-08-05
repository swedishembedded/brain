// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! InstantID's `image_proj` Resampler: one ArcFace embedding -> 16 ID tokens.
//!
//! Built entirely from [`model::rowemit::RowEmit`], which is the same emitter
//! `crates/pulid`'s `IDFormer` records with — the two are the same
//! `PerceiverAttention` from the same IP-Adapter lineage, and this crate adds no
//! second copy of the row arithmetic, the fused-kv strides or the attention
//! dispatch. It also adds **no kernel**.
//!
//! # The concatenation is a row offset, not a copy
//!
//! The reference computes `kv_input = cat(x, latents)` where `x` is the SINGLE
//! projected ArcFace token. Here `norm1(x)` is written into row 0 of one buffer
//! and `norm2(latents)` into rows `1..1+num_queries` of the same buffer, so:
//!
//! * `to_kv` runs over all `1 + num_queries` rows in one dispatch;
//! * `to_q` runs over the SAME buffer sliced from row 1 — the reference applies
//!   it to `norm2(latents)`, which is exactly those rows.
//!
//! Sizing k/v at `num_queries` instead of `1 + num_queries` is the trap
//! [`crate::config::ResamplerConfig::kv_rows`] exists to name: it is shape-legal
//! at one query and silently drops the ArcFace context everywhere else.
//!
//! # Buffers are reused across layers
//!
//! Every scratch buffer is shared by all `depth` layers, so a stage is only
//! readable immediately after the step that wrote it. [`Resampler::read_tap`]
//! therefore REPLAYS the graph up to that step rather than reading after one
//! full forward — the same approach `pulid::IdFormer` takes, and the reason the
//! parity ladder can gate every layer rather than only the output.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::rowemit::{RowEmit, RowKernels};

use crate::config::ResamplerConfig;

/// The kernels this crate dispatches: the shared row-emitter set plus `add2`
/// (the two residual adds) and `gelu_erf` (the feed-forward activation —
/// `nn.GELU()` with PyTorch's default `approximate='none'`, i.e. the erf form,
/// NOT the tanh approximation).
pub const KERNELS: &[(&str, &str)] = &[
    ("layernorm", kernels::LAYERNORM),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_gemv", kernels::MATMUL_GEMV),
    ("bias_add", kernels::BIAS_ADD),
    ("gelu_erf", kernels::GELU_ERF),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("region_copy", kernels::REGION_COPY),
    ("add2", kernels::ADD2),
];

/// LayerNorm epsilon. `nn.LayerNorm`'s default, which the reference does not
/// override.
pub const EPS: f32 = 1e-5;

fn idx(names: &[(&str, &str)], k: &str) -> usize {
    names
        .iter()
        .position(|(n, _)| *n == k)
        .unwrap_or_else(|| panic!("instantid: the Gpu was built without the `{k}` kernel"))
}

/// A named stage snapshot: the buffer, its float offset and length, and the step
/// index just after the step that wrote it.
struct Tap {
    name: String,
    buf: DeviceBuffer,
    off: usize,
    len: usize,
    step: usize,
}

/// `image_proj`: `[1, embedding_dim] -> [num_queries, output_dim]`.
pub struct Resampler {
    gpu: Gpu,
    cfg: ResamplerConfig,
    x_in: DeviceBuffer,
    out: DeviceBuffer,
    steps: Vec<Step>,
    taps: Vec<Tap>,
    /// The uploaded weights. Held so their device buffers outlive the recorded
    /// steps that bind them — the steps keep clones, but owning the map here is
    /// what makes that ownership visible rather than incidental.
    _weights: HashMap<String, DeviceBuffer>,
}

impl Resampler {
    /// Build on an existing device. `w` maps the released `image_proj` tensor
    /// names to their data (see [`crate::import`]).
    pub fn new_on(gpu: Gpu, cfg: ResamplerConfig, names: &[(&str, &str)], w: &HashMap<String, Vec<f32>>) -> Resampler {
        let k = RowKernels::resolve("instantid", names);
        let (k_add2, k_gelu) = (idx(names, "add2"), idx(names, "gelu_erf"));
        let e = RowEmit::new(&gpu, k, EPS);

        let (d, nq, kvr) = (cfg.dim, cfg.num_queries, cfg.kv_rows());
        let (inner, ff) = (cfg.inner_dim(), cfg.ff_inner());
        let od = cfg.output_dim;

        // Weights, uploaded once.
        let up: HashMap<String, DeviceBuffer> = w
            .iter()
            .map(|(n, v)| {
                let b = gpu.storage(v.len() as u64);
                gpu.write_f32(&b, v);
                (n.clone(), b)
            })
            .collect();
        let wb = |n: &str| up.get(n).unwrap_or_else(|| panic!("instantid: missing weight `{n}`"));

        // Buffers, reused across layers.
        let x_in = gpu.storage(cfg.embedding_dim as u64);
        let x1 = gpu.storage(d as u64);
        let lat_a = gpu.storage((nq * d) as u64);
        let lat_b = gpu.storage((nq * d) as u64);
        let nkv = gpu.storage((kvr * d) as u64);
        let q = gpu.storage((nq * inner) as u64);
        let kv = gpu.storage((kvr * 2 * inner) as u64);
        let scores = gpu.storage((cfg.heads * nq * kvr) as u64);
        let probs = gpu.storage((cfg.heads * nq * kvr) as u64);
        let ctx = gpu.storage((nq * inner) as u64);
        let aout = gpu.storage((nq * d) as u64);
        let ffn = gpu.storage((nq * d) as u64);
        let fh = gpu.storage((nq * ff) as u64);
        let fg = gpu.storage((nq * ff) as u64);
        let fo = gpu.storage((nq * d) as u64);
        let pout = gpu.storage((nq * od) as u64);
        let out = gpu.storage((nq * od) as u64);

        let mut s: Vec<Step> = Vec::new();
        let mut taps: Vec<Tap> = Vec::new();
        let tap = |s: &Vec<Step>, taps: &mut Vec<Tap>, name: &str, buf: &DeviceBuffer, off: usize, len: usize| {
            taps.push(Tap { name: name.into(), buf: buf.clone(), off, len, step: s.len() });
        };

        // The learned latents are a weight, copied into the working buffer once.
        e.copy_rows(&mut s, wb("latents"), 0, &lat_a, 0, nq, d);
        tap(&s, &mut taps, "latents_init", &lat_a, 0, nq * d);

        // proj_in: the single ArcFace token.
        e.linear(&mut s, &x_in, 0, wb("proj_in.weight"), Some(wb("proj_in.bias")), &x1, 0, 1, cfg.embedding_dim, d);
        tap(&s, &mut taps, "proj_in", &x1, 0, d);

        for l in 0..cfg.depth {
            let p = |n: &str| format!("layers.{l}.{n}");
            // kv input = cat(norm1(x), norm2(latents)) in ONE buffer: x at row 0,
            // the latents from row 1. `to_q` then slices the SAME buffer from
            // row 1, which is the reference's `to_q(norm2(latents))`.
            e.ln(&mut s, &x1, 0, wb(&p("0.norm1.weight")), wb(&p("0.norm1.bias")), &nkv, 0, 1, d);
            e.ln(&mut s, &lat_a, 0, wb(&p("0.norm2.weight")), wb(&p("0.norm2.bias")), &nkv, 1, nq, d);
            e.linear(&mut s, &nkv, 1, wb(&p("0.to_q.weight")), None, &q, 0, nq, d, inner);
            e.linear(&mut s, &nkv, 0, wb(&p("0.to_kv.weight")), None, &kv, 0, kvr, d, 2 * inner);
            e.cross_attn(&mut s, &q, 0, &kv, &scores, &probs, &ctx, cfg.heads, cfg.dim_head, nq, kvr);
            e.linear(&mut s, &ctx, 0, wb(&p("0.to_out.weight")), None, &aout, 0, nq, inner, d);
            // latents = attn(...) + latents
            s.push(gpu.step(k_add2, &[&lat_a, &aout, &lat_b], &[(nq * d) as u32], (nq * d) as u32));
            tap(&s, &mut taps, &format!("layer{l}_attn"), &lat_b, 0, nq * d);

            // Feed-forward: LayerNorm -> Linear -> GELU -> Linear, all bias-free
            // after the norm (the reference's nn.Sequential, hence the positional
            // 1.0 / 1.1 / 1.3 names).
            e.ln(&mut s, &lat_b, 0, wb(&p("1.0.weight")), wb(&p("1.0.bias")), &ffn, 0, nq, d);
            e.linear(&mut s, &ffn, 0, wb(&p("1.1.weight")), None, &fh, 0, nq, d, ff);
            s.push(gpu.step(k_gelu, &[&fh, &fg], &[(nq * ff) as u32], (nq * ff) as u32));
            e.linear(&mut s, &fg, 0, wb(&p("1.3.weight")), None, &fo, 0, nq, ff, d);
            // latents = ff(...) + latents
            s.push(gpu.step(k_add2, &[&lat_b, &fo, &lat_a], &[(nq * d) as u32], (nq * d) as u32));
            tap(&s, &mut taps, &format!("layer{l}_ff"), &lat_a, 0, nq * d);
        }

        e.linear(&mut s, &lat_a, 0, wb("proj_out.weight"), Some(wb("proj_out.bias")), &pout, 0, nq, d, od);
        tap(&s, &mut taps, "proj_out", &pout, 0, nq * od);
        e.ln(&mut s, &pout, 0, wb("norm_out.weight"), wb("norm_out.bias"), &out, 0, nq, od);
        tap(&s, &mut taps, "id_tokens", &out, 0, nq * od);

        Resampler { gpu, cfg, x_in, out, steps: s, taps, _weights: up }
    }

    /// Build on its own device handle.
    pub fn new(cfg: ResamplerConfig, w: &HashMap<String, Vec<f32>>) -> Resampler {
        let gpu = Gpu::new(KERNELS);
        Resampler::new_on(gpu, cfg, KERNELS, w)
    }

    /// Upload the ArcFace embedding. **Raw, not L2-normalised** — see
    /// `pulid::idcond` for why that distinction silently matters.
    pub fn set_embedding(&self, e: &[f32]) {
        assert_eq!(e.len(), self.cfg.embedding_dim, "instantid: ArcFace embedding width");
        self.gpu.write_f32(&self.x_in, e);
    }

    pub fn forward(&self) {
        self.gpu.submit(&[], &self.steps);
    }

    /// The projected ID tokens `[num_queries, output_dim]`.
    pub fn read_id_tokens(&self) -> Vec<f32> {
        self.gpu.read(&self.out, self.cfg.num_queries * self.cfg.output_dim)
    }

    pub fn tap_names(&self) -> Vec<String> {
        self.taps.iter().map(|t| t.name.clone()).collect()
    }

    /// Re-run the forward up to the step that produced `name`, then read it.
    /// Scratch buffers are reused across layers, so a stage is only readable
    /// immediately after its own step.
    pub fn read_tap(&self, name: &str) -> Vec<f32> {
        let t = self.taps.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("instantid: no tap `{name}`"));
        self.gpu.submit(&[], &self.steps[..t.step]);
        self.gpu.read(&t.buf, t.off + t.len)[t.off..].to_vec()
    }
}
