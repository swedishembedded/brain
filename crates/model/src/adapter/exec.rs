// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training any parameter-
// efficient adapter method on any model architecture and any base weight
// precision from one shared execution path, for its clients. If your team
// needs expertise in PEFT training infrastructure then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Device execution of an adapter: the forward correction and the backward
//! pass of ONE adapted linear, recorded as steps, for every model that trains
//! or serves adapters.
//!
//! A model asks [`AdapterRuntime`] for the correction of a linear after it has
//! recorded the base output, and for the adapter gradients (and the adapter's
//! share of the input gradient) after it has recorded the base input gradient.
//! The runtime reads only the fp32 activation `x` and the incoming gradient, so
//! it is independent of how the base weight is stored: an f32, bf16, NF4 or
//! int8 base all run through the same code, and mixed formats across linears
//! need nothing special. Kernel choice (library fp32 GEMMs where the device has
//! them, else the register-tiled or reference kernels by shape) is made here
//! once, not in each model.
//!
//! Which adapter kind runs is [`DeviceKind`]: a new kind is one more arm in
//! this module (and its definition and optimization roles), and no model crate
//! changes.

use gpu_core::provider::library::{F32Product, GemmPrecision, LibraryGemmProvider, Operand};
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use paramstore::ParamStore;

use super::device::{lora_names, LoraCfg};
use crate::block;

/// Read access to the adapter tensors of a model: weights, and gradients where
/// a tensor trains. Implemented for [`ParamStore`]; a streamed trainer that
/// keeps its adapters in a store of their own implements it for that.
pub trait ParamAccess {
    fn weight(&self, name: &str) -> &DeviceBuffer;
    /// `None` for a frozen tensor (no gradient buffer exists).
    fn grad(&self, name: &str) -> Option<&DeviceBuffer>;
}

impl ParamAccess for ParamStore {
    fn weight(&self, name: &str) -> &DeviceBuffer {
        self.w(name)
    }
    fn grad(&self, name: &str) -> Option<&DeviceBuffer> {
        self.grad.get(name)
    }
}

/// The adapter kinds the device can execute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    Lora,
}

/// Kernel indices the runtime dispatches, resolved by name once. The gemv entry
/// is optional: a model that did not register it keeps the reference kernel.
/// The weight and input gradients use the reference kernels: their contraction
/// dimension is the adapter's small rank, where a register-tiled kernel has
/// nothing to tile, and its shape contract is not one a rank-3 factor meets (a
/// regression on a 4-bit base was the evidence).
struct Kernels {
    matmul: usize,
    gemv: Option<usize>,
    dx: usize,
    dw: usize,
    axpy: usize,
    grad_scale: usize,
}

impl Kernels {
    fn resolve(gpu: &Gpu) -> Kernels {
        let need = |name: &str| gpu.kernel_index(name).unwrap_or_else(|| panic!("adapter runtime: kernel {name:?} is not registered by this model"));
        Kernels {
            matmul: need("matmul"),
            gemv: gpu.kernel_index("matmul_gemv"),
            dx: need("matmul_dx"),
            dw: need("matmul_dw"),
            axpy: need("axpy"),
            grad_scale: need("grad_scale"),
        }
    }
}

fn whole(buf: &DeviceBuffer) -> Operand<'_> {
    Operand { buf, range: (0, 0) }
}

/// The adapter execution state of one model instance: the configuration, the
/// kernels and the activation scratch sized for `rows` rows.
pub struct AdapterRuntime {
    gpu: Gpu,
    cfg: LoraCfg,
    kind: DeviceKind,
    k: Kernels,
    /// `x A^t` of the forward, `[rows, rank]`.
    mid: DeviceBuffer,
    /// Gradient w.r.t. `mid`, `[rows, rank]`.
    dmid: DeviceBuffer,
    /// The correction before it is added to the base output, `[rows, max_out]`.
    out: DeviceBuffer,
    /// The fp32 reference GEMM provider, for a model that has no [`crate::ops::Ops`]
    /// of its own to take one from.
    reference: LibraryGemmProvider,
}

impl AdapterRuntime {
    /// Scratch for forwards and backwards of up to `rows` rows through linears
    /// of up to `max_out` outputs.
    pub fn new(gpu: &Gpu, cfg: &LoraCfg, rows: u32, max_out: u32) -> AdapterRuntime {
        let r = cfg.max_rank().max(1) as u64;
        AdapterRuntime {
            gpu: gpu.share(),
            cfg: cfg.clone(),
            kind: DeviceKind::Lora,
            k: Kernels::resolve(gpu),
            mid: gpu.storage(rows as u64 * r),
            dmid: gpu.storage(rows as u64 * r),
            out: gpu.storage(rows as u64 * max_out as u64),
            reference: LibraryGemmProvider::for_gpu(gpu, GemmPrecision::Fp32Reference),
        }
    }

    /// The fp32 reference provider (never tensor cores) for a model without an
    /// `Ops`; a model with one passes `ops.tensor_gemm()`.
    pub fn reference_gemm(&self) -> &LibraryGemmProvider {
        &self.reference
    }

    pub fn cfg(&self) -> &LoraCfg {
        &self.cfg
    }

    /// Does the adapter correct the linear whose leaf name is `leaf`?
    pub fn hits(&self, leaf: &str) -> bool {
        self.cfg.targets_leaf(leaf)
    }

    /// `y += correction(x)` for the adapted linear `base` (`x` is `[m, k]`, `y`
    /// is `[m, nout]`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, gemm: &LibraryGemmProvider, s: &mut Vec<Step>, p: &dyn ParamAccess, base: &str, x: &DeviceBuffer, y: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        match self.kind {
            DeviceKind::Lora => self.lora_forward(gemm, s, p, base, x, y, m, k, nout),
        }
    }

    /// The adapter's gradients for the adapted linear `base`, and its share of
    /// the input gradient accumulated into `dx` (the base's share is recorded
    /// by the model first). `d_out` is never written.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(&self, gemm: &LibraryGemmProvider, s: &mut Vec<Step>, p: &dyn ParamAccess, base: &str, d_out: &DeviceBuffer, x: &DeviceBuffer, dx: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        match self.kind {
            DeviceKind::Lora => self.lora_backward(gemm, s, p, base, d_out, x, dx, m, k, nout),
        }
    }

    /// `mid = x A^t`, then `y += scale * mid B^t`: fp32 library products where
    /// the device has them, else the serial kernels (the workgroup-per-output
    /// GEMV for the first product where the device runs it, since at decode
    /// `m = 1` the naive kernel would run `r` threads down a long serial loop).
    #[allow(clippy::too_many_arguments)]
    fn lora_forward(&self, gemm: &LibraryGemmProvider, s: &mut Vec<Step>, p: &dyn ParamAccess, base: &str, x: &DeviceBuffer, y: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        let (an, bn) = lora_names(base);
        let (a, b) = (p.weight(&an), p.weight(&bn));
        let hp = self.cfg.hp_for(base);
        let (r, scale) = (hp.rank as u32, hp.scale());
        let g = &self.gpu;
        let mut lib = Vec::new();
        let tg = gemm;
        let ok = tg.gemm_f32(g, &mut lib, &whole(x), &whole(a), &whole(&self.mid), F32Product { m, n: r, k, trans_a: false, trans_b: true, alpha: 1.0, beta: 0.0 })
            && tg.gemm_f32(g, &mut lib, &whole(&self.mid), &whole(b), &whole(y), F32Product { m, n: nout, k: r, trans_a: false, trans_b: true, alpha: scale, beta: 1.0 });
        if ok {
            s.append(&mut lib);
            return;
        }
        let first = match (self.k.gemv, block::gemv_tier(g, m, r)) {
            (Some(gemv), Some(grid)) => g.dispatch(gemv, &[x, a, &self.mid], &[m, k, r], grid),
            _ => g.step(self.k.matmul, &[x, a, &self.mid], &[m, k, r], m * r),
        };
        s.push(first);
        s.push(g.step(self.k.matmul, &[&self.mid, b, &self.out], &[m, r, nout], m * nout));
        s.push(g.step(self.k.axpy, &[y, &self.out], &[m * nout, f(scale)], m * nout));
    }

    /// With `mid = scale * x A^t` and `dmid = scale * d_out B`: `gB += d_out^t
    /// mid`, `gA += dmid^t x` (skipped when `A` is frozen), `dx += dmid A`.
    #[allow(clippy::too_many_arguments)]
    fn lora_backward(&self, gemm: &LibraryGemmProvider, s: &mut Vec<Step>, p: &dyn ParamAccess, base: &str, d_out: &DeviceBuffer, x: &DeviceBuffer, dx: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        let (an, bn) = lora_names(base);
        let (a, b) = (p.weight(&an), p.weight(&bn));
        let gb = p.grad(&bn).unwrap_or_else(|| panic!("adapter runtime: {bn} has no gradient buffer"));
        let ga = p.grad(&an);
        let hp = self.cfg.hp_for(base);
        let (r, scale) = (hp.rank as u32, hp.scale());
        let g = &self.gpu;
        let mut lib = Vec::new();
        let tg = gemm;
        let pr = |m, n, k, trans_a, trans_b, alpha, beta| F32Product { m, n, k, trans_a, trans_b, alpha, beta };
        let ok = tg.gemm_f32(g, &mut lib, &whole(x), &whole(a), &whole(&self.mid), pr(m, r, k, false, true, scale, 0.0))
            && tg.gemm_f32(g, &mut lib, &whole(d_out), &whole(&self.mid), &whole(gb), pr(nout, r, m, true, false, 1.0, 1.0))
            && tg.gemm_f32(g, &mut lib, &whole(d_out), &whole(b), &whole(&self.dmid), pr(m, r, nout, false, false, scale, 0.0))
            && ga.is_none_or(|ga| tg.gemm_f32(g, &mut lib, &whole(&self.dmid), &whole(x), &whole(ga), pr(r, k, m, true, false, 1.0, 1.0)))
            && tg.gemm_f32(g, &mut lib, &whole(&self.dmid), &whole(a), &whole(dx), pr(m, k, r, false, false, 1.0, 1.0));
        if ok {
            s.append(&mut lib);
            return;
        }
        // mid = scale * x A^t -> gB += d_out^t mid
        s.push(g.step(self.k.matmul, &[x, a, &self.mid], &[m, k, r], m * r));
        s.push(g.step(self.k.grad_scale, &[&self.mid], &[m * r, f(scale)], m * r));
        s.push(g.step(self.k.dw, &[d_out, &self.mid, gb], &[m, r, nout], nout * r));
        // dmid = scale * d_out B -> gA += dmid^t x ; dx += dmid A
        s.push(g.step(self.k.dx, &[d_out, b, &self.dmid], &[m, r, nout, 0], m * r));
        s.push(g.step(self.k.grad_scale, &[&self.dmid], &[m * r, f(scale)], m * r));
        if let Some(ga) = ga {
            s.push(g.step(self.k.dw, &[&self.dmid, x, ga], &[m, k, r], r * k));
        }
        s.push(g.step(self.k.dx, &[&self.dmid, a, dx], &[m, k, r, 1], m * k));
    }
}
