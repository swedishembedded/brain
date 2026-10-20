// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA adapter primitives for the BART text side (M6): a rank-`r` delta
//! `y += (alpha/r)*(x@A^T)@B^T` fused onto a targeted linear's base output,
//! recorded by the shared adapter runtime ([`model::adapter::exec`]) that every
//! other LoRA-adapted model in this repo uses, exposed as free functions here
//! instead of a decoder-private method, since florence2's text encoder AND
//! decoder both need to fuse the same delta onto their attention projections
//! and FFN linears.
//!
//! Targets: the four attention projections (`q_proj`/`k_proj`/`v_proj`/
//! `out_proj`) and the two FFN linears (`fc1`/`fc2`) - DaViT stays frozen
//! and out of this module entirely (see `crate::train`'s module doc).

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::adapter::exec::AdapterRuntime;
pub use model::adapter::device::LoraCfg;
use paramstore::ParamStore;

/// The adapter runtime (kernels and scratch) shared by every targeted linear in
/// the model, reused sequentially across every layer - safe because each call's
/// steps fully drain the scratch before the next call's are appended (one
/// command stream, issued in list order). Sized at the largest `(m, nout)` any
/// targeted linear uses.
pub struct LoraScratch {
    rt: AdapterRuntime,
}

impl LoraScratch {
    pub fn new(gpu: &Gpu, cfg: &LoraCfg, max_m: u32, max_nout: u32) -> LoraScratch {
        LoraScratch { rt: AdapterRuntime::new(gpu, cfg, max_m, max_nout) }
    }
}

/// `name` has a gradient buffer allocated (`Role::Trainable`, or a
/// fine-tune-frozen tensor per `ParamStore::freeze_where` - either way its
/// backward must still run correctly; this only gates whether a WEIGHT
/// gradient is worth writing at all).
pub fn trainable(ps: &ParamStore, name: &str) -> bool {
    ps.trainable.iter().any(|(n, _)| n == name) || ps.frozen.iter().any(|(n, _)| n == name)
}

/// Forward LoRA delta for a targeted linear, fused onto `y` in place. `y` must
/// already hold the base projection's output for the SAME `wname` (the
/// caller's own `matmul_rows`+`bias_add`, dispatched immediately before this
/// call). `wname` is the base weight's full name (e.g. `"...q_proj.weight"`).
#[allow(clippy::too_many_arguments)]
pub fn lora_fwd(s: &mut Vec<Step>, ps: &ParamStore, wname: &str, scr: &LoraScratch, x: &DeviceBuffer, y: &DeviceBuffer, m: u32, kk: u32, nout: u32) {
    scr.rt.forward(scr.rt.reference_gemm(), s, ps, wname, x, y, m, kk, nout);
}

/// Backward of [`lora_fwd`]: the adapter's own gradients and its share of `dx`,
/// ALWAYS accumulating (`acc=1`), because the base projection's own `matmul_dx`
/// call for the SAME buffer runs first at every call site in this crate's
/// trainer.
#[allow(clippy::too_many_arguments)]
pub fn lora_bwd(s: &mut Vec<Step>, ps: &ParamStore, wname: &str, scr: &LoraScratch, d_out: &DeviceBuffer, x: &DeviceBuffer, dx: &DeviceBuffer, m: u32, kk: u32, nout: u32) {
    scr.rt.backward(scr.rt.reference_gemm(), s, ps, wname, d_out, x, dx, m, kk, nout, &|_, _| {});
}

/// Bundle of everything a LoRA-fused forward/backward call needs, borrowed
/// together so call sites take one argument.
pub struct LoraCtx<'a> {
    pub scratch: &'a LoraScratch,
}
