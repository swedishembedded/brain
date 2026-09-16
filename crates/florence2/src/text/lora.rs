// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA adapter primitives for the BART text side (M6): a rank-`r` delta
//! `y += (alpha/r)*(x@A^T)@B^T` fused onto a targeted linear's base output,
//! composed entirely from the generic `matmul`/`matmul_dx`/`matmul_dw`/
//! `axpy`/`grad_scale` kernels every other LoRA-adapted model in this repo
//! already uses - same seven-step forward/backward derivation as
//! `deepseek2::model::DeepseekV2::lora_fwd`/`lora_bwd`, just exposed as free
//! functions here instead of a decoder-private method, since florence2's
//! text encoder AND decoder both need to fuse the same delta onto their
//! attention projections and FFN linears.
//!
//! Targets: the four attention projections (`q_proj`/`k_proj`/`v_proj`/
//! `out_proj`) and the two FFN linears (`fc1`/`fc2`) - DaViT stays frozen
//! and out of this module entirely (see `crate::train`'s module doc).

use gpu_core::{DeviceBuffer, Gpu, Step};
use paramstore::ParamStore;

fn f(v: f32) -> u32 {
    v.to_bits()
}

#[derive(Clone, Copy, Debug)]
pub struct LoraCfg {
    pub rank: u32,
    pub alpha: f32,
}

impl LoraCfg {
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

pub struct LoraKernelIds {
    pub matmul: usize,
    pub matmul_dw: usize,
    pub matmul_dx: usize,
    pub axpy: usize,
    pub grad_scale: usize,
}

impl LoraKernelIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> LoraKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).expect(name);
        LoraKernelIds { matmul: k("matmul"), matmul_dw: k("matmul_dw"), matmul_dx: k("matmul_dx"), axpy: k("axpy"), grad_scale: k("grad_scale") }
    }
}

/// Shared scratch for every LoRA delta in the model, reused sequentially
/// across every targeted linear in every layer - safe because each call's
/// steps fully drain the scratch before the next call's steps are appended
/// (one command stream, issued in list order), the same reuse discipline
/// `deepseek2::model::DeepseekV2`'s own `self.lora_a`/`lora_out`/`lora_da`
/// fields rely on. Sized at the largest `(m, nout)` any targeted linear in
/// the model uses.
pub struct LoraScratch {
    a: DeviceBuffer,
    out: DeviceBuffer,
    da: DeviceBuffer,
}

impl LoraScratch {
    pub fn new(gpu: &Gpu, max_m: u32, rank: u32, max_nout: u32) -> LoraScratch {
        LoraScratch { a: gpu.storage((max_m * rank) as u64), out: gpu.storage((max_m * max_nout) as u64), da: gpu.storage((max_m * rank) as u64) }
    }
}

/// `name` has a gradient buffer allocated (`Role::Trainable`, or a
/// fine-tune-frozen tensor per `ParamStore::freeze_where` - either way its
/// backward must still run correctly; this only gates whether a WEIGHT
/// gradient is worth writing at all).
pub fn trainable(ps: &ParamStore, name: &str) -> bool {
    ps.trainable.iter().any(|(n, _)| n == name) || ps.frozen.iter().any(|(n, _)| n == name)
}

/// Forward LoRA delta for a targeted linear, fused onto `y` in place: `y +=
/// (alpha/r)*(x@A^T)@B^T`. `y` must already hold the base projection's
/// output for the SAME `wname` (the caller's own `matmul_rows`+`bias_add`,
/// dispatched immediately before this call). `wname` is the base weight's
/// full name (e.g. `"...q_proj.weight"`); the adapter tensors are read as
/// `{wname}.lora_a` (`[rank,k]`) / `{wname}.lora_b` (`[nout,rank]`).
#[allow(clippy::too_many_arguments)]
pub fn lora_fwd(s: &mut Vec<Step>, gpu: &Gpu, k: &LoraKernelIds, ps: &ParamStore, wname: &str, cfg: &LoraCfg, scr: &LoraScratch, x: &DeviceBuffer, y: &DeviceBuffer, m: u32, kk: u32, nout: u32) {
    let a = ps.w(&format!("{wname}.lora_a"));
    let b = ps.w(&format!("{wname}.lora_b"));
    s.push(gpu.step(k.matmul, &[x, a, &scr.a], &[m, kk, cfg.rank], m * cfg.rank));
    s.push(gpu.step(k.matmul, &[&scr.a, b, &scr.out], &[m, cfg.rank, nout], m * nout));
    s.push(gpu.step(k.axpy, &[y, &scr.out], &[m * nout, f(cfg.scale())], m * nout));
}

/// Backward of [`lora_fwd`]: adds the adapter's own `gA`/`gB` (skipped when
/// frozen - never the case for a LoRA adapter itself, but kept consistent
/// with [`super::attn::BartAttn`]'s base-weight gating) and its share of
/// `dx` on top of whatever `dx` already holds - ALWAYS accumulating
/// (`acc=1`), because the base projection's own `matmul_dx` call for the
/// SAME buffer runs first at every call site in this crate's trainer.
#[allow(clippy::too_many_arguments)]
pub fn lora_bwd(s: &mut Vec<Step>, gpu: &Gpu, k: &LoraKernelIds, ps: &ParamStore, wname: &str, cfg: &LoraCfg, scr: &LoraScratch, d_out: &DeviceBuffer, x: &DeviceBuffer, dx: &DeviceBuffer, m: u32, kk: u32, nout: u32) {
    let a = ps.w(&format!("{wname}.lora_a"));
    let b = ps.w(&format!("{wname}.lora_b"));
    let an = format!("{wname}.lora_a");
    let bn = format!("{wname}.lora_b");
    // a = x.A^T ; gB += scale * d_out^T . a   (scale folded into `a`, private scratch)
    s.push(gpu.step(k.matmul, &[x, a, &scr.a], &[m, kk, cfg.rank], m * cfg.rank));
    s.push(gpu.step(k.grad_scale, &[&scr.a], &[m * cfg.rank, f(cfg.scale())], m * cfg.rank));
    if trainable(ps, &bn) {
        s.push(gpu.step(k.matmul_dw, &[d_out, &scr.a, ps.g(&bn)], &[m, cfg.rank, nout], nout * cfg.rank));
    }
    // da = scale*(d_out.B) ; gA += da^T.x ; dx += da.A
    s.push(gpu.step(k.matmul_dx, &[d_out, b, &scr.da], &[m, cfg.rank, nout, 0], m * cfg.rank));
    s.push(gpu.step(k.grad_scale, &[&scr.da], &[m * cfg.rank, f(cfg.scale())], m * cfg.rank));
    if trainable(ps, &an) {
        s.push(gpu.step(k.matmul_dw, &[&scr.da, x, ps.g(&an)], &[m, kk, cfg.rank], cfg.rank * kk));
    }
    s.push(gpu.step(k.matmul_dx, &[&scr.da, a, dx], &[m, kk, cfg.rank, 1], m * kk));
}

/// Bundle of everything a LoRA-fused forward/backward call needs, borrowed
/// together so call sites take one argument instead of three.
pub struct LoraCtx<'a> {
    pub cfg: &'a LoraCfg,
    pub ids: &'a LoraKernelIds,
    pub scratch: &'a LoraScratch,
}
