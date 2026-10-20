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
//! need nothing special. Kernel choice (library GEMMs with TF32 operands where the tier has
//! tensor cores, else the register-tiled or reference kernels by shape) is made here
//! once, not in each model.
//!
//! Which adapter kind runs is [`DeviceKind`]: a new kind is one more arm in
//! this module (and its definition and groups), and no model crate changes. A
//! model gives the runtime its base operations as closures - the base input
//! gradient for a given output gradient - so a kind that rescales the gradient
//! before it reaches the base (IA3) works on any base weight storage.

use std::cell::RefCell;
use std::collections::HashMap;

use gpu_core::provider::library::{F32Product, GemmPrecision, LibraryGemmProvider, Operand};
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use paramstore::ParamStore;

use super::device::{lora_names, param_name, LoraCfg};
use super::kind::{lokr_factors, DeviceKind, DORA_A, DORA_B, DORA_M, IA3_D, LOKR_W1, LOKR_W2};
use crate::block::{self, pick_gemm};

/// Kernels the non-LoRA kinds dispatch beyond the ones every adapter model
/// registers; a model that trains such a kind appends them to its pipelines
/// with [`add_kernels`].
pub const EXTRA_KERNELS: &[(&str, &str)] =
    &[
    ("scale_chan", kernels::SCALE_CHAN),
    ("swap_axes12_vec", kernels::SWAP_AXES12_VEC),
    ("bias_grad", kernels::BIAS_GRAD),
    ("row_dot", kernels::ROW_DOT),
    ("dora_gain", kernels::DORA_GAIN),
    ("dora_gain_bwd", kernels::DORA_GAIN_BWD),
];

/// Append [`EXTRA_KERNELS`] a pipeline list does not already hold. Appending
/// keeps every existing kernel's index.
pub fn add_kernels(list: &mut Vec<(&'static str, &'static str)>) {
    for &(name, src) in EXTRA_KERNELS {
        if !list.iter().any(|(n, _)| *n == name) {
            list.push((name, src));
        }
    }
}

/// Read access to the adapter tensors of a model: weights, and gradients where
/// a tensor trains. Implemented for [`ParamStore`]; a streamed trainer that
/// keeps its adapters in a store of their own implements it for that.
pub trait ParamAccess {
    fn weight(&self, name: &str) -> &DeviceBuffer;
    /// `None` for a frozen tensor (no gradient buffer exists).
    fn grad(&self, name: &str) -> Option<&DeviceBuffer>;
    /// The base weight `name` as a resident fp32 `[out, in]` buffer, when it
    /// is one. A kind that reads the base itself (DoRA's row norms) needs it;
    /// a quantized base answers `None` and the kind refuses.
    fn base(&self, _name: &str) -> Option<&DeviceBuffer> {
        None
    }
}

impl ParamAccess for ParamStore {
    fn weight(&self, name: &str) -> &DeviceBuffer {
        self.w(name)
    }
    fn grad(&self, name: &str) -> Option<&DeviceBuffer> {
        self.grad.get(name)
    }
    fn base(&self, name: &str) -> Option<&DeviceBuffer> {
        self.weight.get(name)
    }
}

/// Kernel indices the runtime dispatches, resolved by name once. The gemv and
/// the register-tiled gradient entries are optional: a model that did not
/// register them keeps the reference kernels ([`tiled_or_reference`]). The
/// channel/axis kernels are only needed by the kinds that use them.
struct Kernels {
    matmul: usize,
    gemv: Option<usize>,
    dx: usize,
    dx_reg: Option<usize>,
    dw: usize,
    dw_reg: Option<usize>,
    axpy: usize,
    grad_scale: usize,
    scale_chan: Option<usize>,
    swap: Option<usize>,
    bias_grad: Option<usize>,
    row_dot: Option<usize>,
    dora_gain: Option<usize>,
    dora_gain_bwd: Option<usize>,
}

impl Kernels {
    fn resolve(gpu: &Gpu) -> Kernels {
        let need = |name: &str| gpu.kernel_index(name).unwrap_or_else(|| panic!("adapter runtime: kernel {name:?} is not registered by this model"));
        Kernels {
            matmul: need("matmul"),
            gemv: gpu.kernel_index("matmul_gemv"),
            dx: need("matmul_dx"),
            dx_reg: gpu.kernel_index("matmul_dx_reg"),
            dw: need("matmul_dw"),
            dw_reg: gpu.kernel_index("matmul_dw_reg"),
            axpy: need("axpy"),
            grad_scale: need("grad_scale"),
            scale_chan: gpu.kernel_index("scale_chan"),
            swap: gpu.kernel_index("swap_axes12_vec"),
            bias_grad: gpu.kernel_index("bias_grad"),
            row_dot: gpu.kernel_index("row_dot"),
            dora_gain: gpu.kernel_index("dora_gain"),
            dora_gain_bwd: gpu.kernel_index("dora_gain_bwd"),
        }
    }

    /// An optional kernel, or the instruction to register it.
    fn extra(slot: Option<usize>, name: &str, kind: DeviceKind) -> usize {
        slot.unwrap_or_else(|| {
            panic!("adapter runtime: a {} adapter needs kernel {name:?}; append model::adapter::exec::EXTRA_KERNELS to this model's pipelines (add_kernels)", kind.name())
        })
    }
}

/// The kernel and dispatch for a GEMM with a `rows x cols` result: the
/// register-tiled kernel (one workgroup per 128x128 tile) where the shape fills
/// a tile AND the model registered it, else the reference kernel with one thread
/// per output element. The tile grid must never be paired with the reference
/// kernel: its workgroup is 64 threads, so a one-tile grid computes 64 outputs.
fn tiled_or_reference(reference: usize, tiled: Option<usize>, rows: u32, cols: u32) -> (usize, gpu_core::Dispatch) {
    match tiled {
        Some(tiled) => pick_gemm(rows as usize, cols as usize, reference, tiled, false),
        None => (reference, gpu_core::Dispatch::Threads(rows * cols)),
    }
}

fn whole(buf: &DeviceBuffer) -> Operand<'_> {
    Operand { buf, range: (0, 0) }
}

/// A scratch buffer that only ever grows: a handle recorded into an earlier
/// step keeps its buffer alive, so replacing it for a larger request is safe.
struct Scratch {
    len: u64,
    buf: DeviceBuffer,
}

/// The adapter execution state of one model instance: the configuration, the
/// kernels and the activation scratch sized for `rows` rows.
pub struct AdapterRuntime {
    gpu: Gpu,
    cfg: LoraCfg,
    k: Kernels,
    /// `x A^t` of the forward, `[rows, rank]`.
    mid: DeviceBuffer,
    /// Gradient w.r.t. `mid`, `[rows, rank]`.
    dmid: DeviceBuffer,
    /// The correction before it is added to the base output, `[rows, max_out]`.
    out: DeviceBuffer,
    /// A vector of ones (`max_out * max_rank` long), for the channel kernel used
    /// as a copy.
    ones: DeviceBuffer,
    /// General scratch of the kinds that need more than `mid`/`dmid`/`out`.
    scratch: RefCell<Vec<Option<Scratch>>>,
    /// Per-site base outputs saved by a forward for its backward (IA3).
    saved: RefCell<HashMap<String, (u64, DeviceBuffer)>>,
    /// The fp32 reference GEMM provider, for a model that has no [`crate::ops::Ops`]
    /// of its own to take one from.
    reference: LibraryGemmProvider,
}

const SCRATCH_SLOTS: usize = 24;

impl AdapterRuntime {
    /// Scratch for forwards and backwards of up to `rows` rows through linears
    /// of up to `max_out` outputs.
    pub fn new(gpu: &Gpu, cfg: &LoraCfg, rows: u32, max_out: u32) -> AdapterRuntime {
        let r = cfg.max_rank().max(1) as u64;
        AdapterRuntime {
            gpu: gpu.share(),
            cfg: cfg.clone(),
            k: Kernels::resolve(gpu),
            mid: gpu.storage(rows as u64 * r),
            dmid: gpu.storage(rows as u64 * r),
            out: gpu.storage(rows as u64 * max_out as u64),
            ones: gpu.storage_init("adapter_ones", &vec![1.0f32; max_out as usize * r as usize]),
            scratch: RefCell::new((0..SCRATCH_SLOTS).map(|_| None).collect()),
            saved: RefCell::new(HashMap::new()),
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

    /// Scratch slot `slot` of at least `len` elements.
    fn scratch(&self, slot: usize, len: u64) -> DeviceBuffer {
        let mut all = self.scratch.borrow_mut();
        match &all[slot] {
            Some(s) if s.len >= len => s.buf.clone(),
            _ => {
                let buf = self.gpu.storage(len);
                all[slot] = Some(Scratch { len, buf: buf.clone() });
                buf
            }
        }
    }

    /// The buffer a forward saves `base`'s output in, `len` elements.
    fn saved(&self, base: &str, len: u64) -> DeviceBuffer {
        let mut all = self.saved.borrow_mut();
        match all.get(base) {
            Some((have, buf)) if *have >= len => buf.clone(),
            _ => {
                let buf = self.gpu.storage(len);
                all.insert(base.to_string(), (len, buf.clone()));
                buf
            }
        }
    }

    /// `y += correction(x)` for the adapted linear `base` (`x` is `[m, k]`, `y`
    /// is `[m, nout]`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, gemm: &LibraryGemmProvider, s: &mut Vec<Step>, p: &dyn ParamAccess, base: &str, x: &DeviceBuffer, y: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        match self.cfg.kind {
            DeviceKind::Lora => self.lora_forward(gemm, s, p, base, lora_names(base), x, y, m, k, nout),
            DeviceKind::Ia3 => self.ia3_forward(s, p, base, y, m, nout),
            DeviceKind::Lokr => self.lokr_forward(s, p, base, x, y, m, k, nout),
            DeviceKind::Dora => self.dora_forward(gemm, s, p, base, x, y, m, k, nout),
        }
    }

    /// The adapted linear's backward: the base's input gradient (through
    /// `base_dx`, which records it for a given output gradient into the
    /// caller's `dx`), the adapter's own gradients, and the adapter's share of
    /// `dx`. `d_out` is never written. A kind that rescales the gradient before
    /// it reaches the base calls `base_dx` with the rescaled one.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        gemm: &LibraryGemmProvider,
        s: &mut Vec<Step>,
        p: &dyn ParamAccess,
        base: &str,
        d_out: &DeviceBuffer,
        x: &DeviceBuffer,
        dx: &DeviceBuffer,
        m: u32,
        k: u32,
        nout: u32,
        base_dx: &dyn Fn(&mut Vec<Step>, &DeviceBuffer),
    ) {
        match self.cfg.kind {
            DeviceKind::Lora => {
                base_dx(s, d_out);
                self.lora_backward(gemm, s, p, base, lora_names(base), d_out, x, dx, m, k, nout);
            }
            DeviceKind::Ia3 => self.ia3_backward(s, p, base, d_out, m, nout, base_dx),
            DeviceKind::Lokr => {
                base_dx(s, d_out);
                self.lokr_backward(s, p, base, d_out, x, dx, m, k, nout);
            }
            DeviceKind::Dora => self.dora_backward(gemm, s, p, base, d_out, x, dx, m, k, nout, base_dx),
        }
    }

    /// `mid = x A^t`, then `y += scale * mid B^t`: library products (TF32 operands, fp32 accumulation) where
    /// the device has them, else the serial kernels (the workgroup-per-output
    /// GEMV for the first product where the device runs it, since at decode
    /// `m = 1` the naive kernel would run `r` threads down a long serial loop).
    #[allow(clippy::too_many_arguments)]
    fn lora_forward(&self, gemm: &LibraryGemmProvider, s: &mut Vec<Step>, p: &dyn ParamAccess, base: &str, (an, bn): (String, String), x: &DeviceBuffer, y: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        let (a, b) = (p.weight(&an), p.weight(&bn));
        let hp = self.cfg.hp_for(base);
        let (r, scale) = (hp.rank as u32, hp.scale());
        let g = &self.gpu;
        let mut lib = Vec::new();
        let tg = gemm;
        let ok = tg.gemm_f32(g, &mut lib, &whole(x), &whole(a), &whole(&self.mid), F32Product { m, n: r, k, trans_a: false, trans_b: true, alpha: 1.0, beta: 0.0, tf32: true })
            && tg.gemm_f32(g, &mut lib, &whole(&self.mid), &whole(b), &whole(y), F32Product { m, n: nout, k: r, trans_a: false, trans_b: true, alpha: scale, beta: 1.0, tf32: true });
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
    fn lora_backward(&self, gemm: &LibraryGemmProvider, s: &mut Vec<Step>, p: &dyn ParamAccess, base: &str, (an, bn): (String, String), d_out: &DeviceBuffer, x: &DeviceBuffer, dx: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        let (a, b) = (p.weight(&an), p.weight(&bn));
        let gb = p.grad(&bn).unwrap_or_else(|| panic!("adapter runtime: {bn} has no gradient buffer"));
        let ga = p.grad(&an);
        let hp = self.cfg.hp_for(base);
        let (r, scale) = (hp.rank as u32, hp.scale());
        let g = &self.gpu;
        let mut lib = Vec::new();
        let tg = gemm;
        let pr = |m, n, k, trans_a, trans_b, alpha, beta| F32Product { m, n, k, trans_a, trans_b, alpha, beta, tf32: true };
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
        s.push(self.dw(&[d_out, &self.mid, gb], m, r, nout));
        // dmid = scale * d_out B -> gA += dmid^t x ; dx += dmid A
        s.push(self.dx(&[d_out, b, &self.dmid], m, r, nout, 0));
        s.push(g.step(self.k.grad_scale, &[&self.dmid], &[m * r, f(scale)], m * r));
        if let Some(ga) = ga {
            s.push(self.dw(&[&self.dmid, x, ga], m, k, r));
        }
        s.push(self.dx(&[&self.dmid, a, dx], m, k, r, 1));
    }

    /// `dW[n, k] += sum_m dY[m, n] X[m, k]` with `bufs = [dY, X, dW]`.
    fn dw(&self, bufs: &[&DeviceBuffer], m: u32, k: u32, n: u32) -> Step {
        let (kern, grid) = tiled_or_reference(self.k.dw, self.k.dw_reg, n, k);
        self.gpu.dispatch(kern, bufs, &[m, k, n], grid)
    }

    /// `dX[m, k] (+)= sum_n dY[m, n] W[n, k]` with `bufs = [dY, W, dX]`.
    fn dx(&self, bufs: &[&DeviceBuffer], m: u32, k: u32, n: u32, accumulate: u32) -> Step {
        let (kern, grid) = tiled_or_reference(self.k.dx, self.k.dx_reg, m, k);
        self.gpu.dispatch(kern, bufs, &[m, k, n, accumulate], grid)
    }

    /// `out[i] = x[i] * v[(i / inner) % c]` over `total` elements.
    fn chan(&self, kind: DeviceKind, x: &DeviceBuffer, v: &DeviceBuffer, out: &DeviceBuffer, total: u32, c: u32, inner: u32) -> Step {
        let kern = Kernels::extra(self.k.scale_chan, "scale_chan", kind);
        self.gpu.step(kern, &[x, v, out], &[total, c, inner], total)
    }

    /// IA3: `y' = y * (1 + d)` per output channel. The base output is saved
    /// first (the gradient of `d` needs it), the correction `y * d` is formed in
    /// scratch and added.
    fn ia3_forward(&self, s: &mut Vec<Step>, p: &dyn ParamAccess, base: &str, y: &DeviceBuffer, m: u32, nout: u32) {
        let d = p.weight(&param_name(base, IA3_D));
        let total = m * nout;
        let saved = self.saved(base, total as u64);
        let tmp = self.scratch(0, total as u64);
        s.push(self.chan(DeviceKind::Ia3, y, &self.ones, &saved, total, nout, 1));
        s.push(self.chan(DeviceKind::Ia3, &saved, d, &tmp, total, nout, 1));
        s.push(self.gpu.step(self.k.axpy, &[y, &tmp], &[total, f(1.0)], total));
    }

    /// IA3 backward: `d_y_base = d_out * (1 + d)` goes to the base, and
    /// `gd[j] += sum_m d_out[m, j] * y_base[m, j]`.
    #[allow(clippy::too_many_arguments)]
    fn ia3_backward(&self, s: &mut Vec<Step>, p: &dyn ParamAccess, base: &str, d_out: &DeviceBuffer, m: u32, nout: u32, base_dx: &dyn Fn(&mut Vec<Step>, &DeviceBuffer)) {
        let dn = param_name(base, IA3_D);
        let d = p.weight(&dn);
        let total = m * nout;
        let saved = self.saved(base, total as u64);
        if let Some(gd) = p.grad(&dn) {
            let prod = self.scratch(0, total as u64);
            s.push(self.chan(DeviceKind::Ia3, d_out, &saved, &prod, total, total, 1));
            let kern = Kernels::extra(self.k.bias_grad, "bias_grad", DeviceKind::Ia3);
            s.push(self.gpu.step(kern, &[&prod, gd], &[m, nout], nout));
        }
        let dy = self.scratch(1, total as u64);
        s.push(self.chan(DeviceKind::Ia3, d_out, d, &dy, total, nout, 1));
        s.push(self.gpu.step(self.k.axpy, &[&dy, d_out], &[total, f(1.0)], total));
        base_dx(s, &dy);
    }

    /// The Kronecker factors of `base` as `(a, c, b, d)`.
    fn lokr_dims(nout: u32, k: u32) -> (u32, u32, u32, u32) {
        let (a, c, b, d) = lokr_factors(nout as usize, k as usize);
        (a as u32, c as u32, b as u32, d as u32)
    }

    fn swap(&self, src: &DeviceBuffer, dst: &DeviceBuffer, a0: u32, a1: u32, a2: u32) -> Step {
        let kern = Kernels::extra(self.k.swap, "swap_axes12_vec", DeviceKind::Lokr);
        self.gpu.step(kern, &[src, dst], &[a0, a1, a2, 1], a0 * a1 * a2)
    }

    /// LoKr forward: `y += scale * (W1 kron W2) x` as two GEMMs and two axis
    /// swaps over `x` viewed `[m, b, d]` - `T = x W2^t` (`[m, b, c]`), swapped to
    /// `[m, c, b]`, `U = T W1^t` (`[m, c, a]`), swapped to `[m, a, c]`, which is
    /// the output's row-major layout. The Kronecker product is never formed.
    #[allow(clippy::too_many_arguments)]
    fn lokr_forward(&self, s: &mut Vec<Step>, p: &dyn ParamAccess, base: &str, x: &DeviceBuffer, y: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        let (w1, w2) = (p.weight(&param_name(base, LOKR_W1)), p.weight(&param_name(base, LOKR_W2)));
        let (a, c, b, d) = Self::lokr_dims(nout, k);
        let scale = self.cfg.hp_for(base).scale();
        let (t, t2) = (self.scratch(0, (m * b * c) as u64), self.scratch(1, (m * b * c) as u64));
        let (u, v) = (self.scratch(2, (m * c * a) as u64), self.scratch(3, (m * c * a) as u64));
        s.push(self.gpu.step(self.k.matmul, &[x, w2, &t], &[m * b, d, c], m * b * c));
        s.push(self.swap(&t, &t2, m, b, c));
        s.push(self.gpu.step(self.k.matmul, &[&t2, w1, &u], &[m * c, b, a], m * c * a));
        s.push(self.swap(&u, &v, m, c, a));
        s.push(self.gpu.step(self.k.axpy, &[y, &v], &[m * nout, f(scale)], m * nout));
    }

    /// LoKr backward, recomputing the forward's intermediates from `x`.
    #[allow(clippy::too_many_arguments)]
    fn lokr_backward(&self, s: &mut Vec<Step>, p: &dyn ParamAccess, base: &str, d_out: &DeviceBuffer, x: &DeviceBuffer, dx: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        let (n1, n2) = (param_name(base, LOKR_W1), param_name(base, LOKR_W2));
        let (w1, w2) = (p.weight(&n1), p.weight(&n2));
        let (a, c, b, d) = Self::lokr_dims(nout, k);
        let scale = self.cfg.hp_for(base).scale();
        let (t, t2) = (self.scratch(0, (m * b * c) as u64), self.scratch(1, (m * b * c) as u64));
        let (u, v) = (self.scratch(2, (m * c * a) as u64), self.scratch(3, (m * c * a) as u64));
        // Recompute T and T2 (the forward's scratch holds another site's by now).
        s.push(self.gpu.step(self.k.matmul, &[x, w2, &t], &[m * b, d, c], m * b * c));
        s.push(self.swap(&t, &t2, m, b, c));
        // dV = scale * d_out in [m, a, c]; dU = swap -> [m, c, a].
        s.push(self.chan(DeviceKind::Lokr, d_out, &self.ones, &v, m * nout, nout, 1));
        s.push(self.gpu.step(self.k.grad_scale, &[&v], &[m * nout, f(scale)], m * nout));
        s.push(self.swap(&v, &u, m, a, c));
        // gW1 += dU^t T2 ; dT2 = dU W1.
        if let Some(g1) = p.grad(&n1) {
            s.push(self.dw(&[&u, &t2, g1], m * c, b, a));
        }
        s.push(self.dx(&[&u, w1, &t2], m * c, b, a, 0));
        // dT = swap(dT2) -> [m, b, c]; gW2 += dT^t x ; dx += dT W2.
        s.push(self.swap(&t2, &t, m, c, b));
        if let Some(g2) = p.grad(&n2) {
            s.push(self.dw(&[&t, x, g2], m * b, d, c));
        }
        s.push(self.dx(&[&t, w2, dx], m * b, d, c, 1));
    }
    /// `out[r] = alpha * sum_d a[r, d] * b[r, d]` over `rows` rows of width `d`.
    fn row_dot(&self, kind: DeviceKind, a: &DeviceBuffer, b: &DeviceBuffer, out: &DeviceBuffer, rows: u32, d: u32) -> Step {
        let kern = Kernels::extra(self.k.row_dot, "row_dot", kind);
        self.gpu.step(kern, &[a, b, out], &[rows, d, 0, 0, f(1.0)], rows)
    }

    /// The base weight of `base`, resident as fp32: DoRA reads it for row norms.
    fn dora_base<'a>(p: &'a dyn ParamAccess, base: &str) -> &'a DeviceBuffer {
        p.base(base).unwrap_or_else(|| panic!("adapter runtime: a dora adapter needs {base} resident as fp32; a quantized or half-precision base is not supported"))
    }

    /// DoRA's per-channel gain of `base`: `g = (1 + mu) * |W_o| / |W_o + s (BA)_o|`
    /// and the pieces the backward reads, written to scratch. The row norms come
    /// from `|W_o|^2` and the products `C = W A^t`, `G = A A^t`, `B G` - never from
    /// a dense `B A`. Returns `(gain [g | ratio | 1/v2], C [out, r], B G [out, r])`.
    #[allow(clippy::too_many_arguments)]
    fn dora_gain(&self, s: &mut Vec<Step>, p: &dyn ParamAccess, base: &str, r: u32, k: u32, nout: u32, scale: f32) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer) {
        let kind = DeviceKind::Dora;
        let (a, b, mu) = (p.weight(&param_name(base, DORA_A)), p.weight(&param_name(base, DORA_B)), p.weight(&param_name(base, DORA_M)));
        let w = Self::dora_base(p, base);
        let rr = (r * nout) as u64;
        let (ct, gm, bg, cm) = (self.scratch(4, rr), self.scratch(5, (r * r) as u64), self.scratch(6, rr), self.scratch(7, rr));
        let (w2, s1, s2, gain) = (self.scratch(8, nout as u64), self.scratch(9, nout as u64), self.scratch(10, nout as u64), self.scratch(11, 3 * nout as u64));
        let g = &self.gpu;
        s.push(g.step(self.k.matmul, &[a, w, &ct], &[r, k, nout], r * nout));
        s.push(g.step(self.k.matmul, &[a, a, &gm], &[r, k, r], r * r));
        s.push(g.step(self.k.matmul, &[b, &gm, &bg], &[nout, r, r], nout * r));
        s.push(self.swap(&ct, &cm, 1, r, nout));
        s.push(self.row_dot(kind, w, w, &w2, nout, k));
        s.push(self.row_dot(kind, b, &cm, &s1, nout, r));
        s.push(self.row_dot(kind, b, &bg, &s2, nout, r));
        let kern = Kernels::extra(self.k.dora_gain, "dora_gain", kind);
        s.push(g.step(kern, &[&w2, &s1, &s2, mu, &gain], &[nout, f(scale)], nout));
        (gain, cm, bg)
    }

    /// DoRA forward: `y = (x W^t + s (x A^t) B^t) * g`. The pre-scale output is
    /// saved for the gradient of `g`.
    #[allow(clippy::too_many_arguments)]
    fn dora_forward(&self, gemm: &LibraryGemmProvider, s: &mut Vec<Step>, p: &dyn ParamAccess, base: &str, x: &DeviceBuffer, y: &DeviceBuffer, m: u32, k: u32, nout: u32) {
        let hp = self.cfg.hp_for(base);
        let (r, scale) = (hp.rank as u32, hp.scale());
        let names = (param_name(base, DORA_A), param_name(base, DORA_B));
        self.lora_forward(gemm, s, p, base, names, x, y, m, k, nout);
        let total = m * nout;
        let saved = self.saved(base, total as u64);
        s.push(self.chan(DeviceKind::Dora, y, &self.ones, &saved, total, nout, 1));
        let (gain, _, _) = self.dora_gain(s, p, base, r, k, nout, scale);
        s.push(self.chan(DeviceKind::Dora, &saved, &gain, y, total, nout, 1));
    }

    /// DoRA backward. The gradient reaching the pre-scale output is
    /// `d_out * g`: it goes to the base and through the low-rank path like
    /// LoRA's. The gradient of `g` gives the magnitude gradient and, through the
    /// row norms, a direction gradient `dV_o = c_o V_o` that reaches `A` and `B`
    /// as `dB = s diag(c)(C + s B G)` and `dA = s E^t W + s^2 (E^t B) A` with
    /// `E = diag(c) B`.
    #[allow(clippy::too_many_arguments)]
    fn dora_backward(
        &self,
        gemm: &LibraryGemmProvider,
        s: &mut Vec<Step>,
        p: &dyn ParamAccess,
        base: &str,
        d_out: &DeviceBuffer,
        x: &DeviceBuffer,
        dx: &DeviceBuffer,
        m: u32,
        k: u32,
        nout: u32,
        base_dx: &dyn Fn(&mut Vec<Step>, &DeviceBuffer),
    ) {
        let kind = DeviceKind::Dora;
        let hp = self.cfg.hp_for(base);
        let (r, scale) = (hp.rank as u32, hp.scale());
        let (an, bn, mn) = (param_name(base, DORA_A), param_name(base, DORA_B), param_name(base, DORA_M));
        let (a, b) = (p.weight(&an), p.weight(&bn));
        let w = Self::dora_base(p, base);
        let g = &self.gpu;
        let total = m * nout;
        let saved = self.saved(base, total as u64);
        let (gain, cm, bg) = self.dora_gain(s, p, base, r, k, nout, scale);

        // dg_o = sum_m d_out * y_pre; gmu += dg * ratio; c = -dg * g / v2.
        let prod = self.scratch(12, total as u64);
        let dg = self.scratch(13, nout as u64);
        s.push(self.chan(kind, d_out, &saved, &prod, total, total, 1));
        s.push(g.step(self.k.grad_scale, &[&dg], &[nout, f(0.0)], nout));
        s.push(g.step(Kernels::extra(self.k.bias_grad, "bias_grad", kind), &[&prod, &dg], &[m, nout], nout));
        let gmu = match p.grad(&mn) {
            Some(gm) => gm.clone(),
            None => self.scratch(14, nout as u64),
        };
        let cvec = self.scratch(15, nout as u64);
        s.push(g.step(Kernels::extra(self.k.dora_gain_bwd, "dora_gain_bwd", kind), &[&dg, &gain, &gmu, &cvec], &[nout], nout));

        // The pre-scale gradient: to the base, and through the low-rank path.
        let dy = self.scratch(16, total as u64);
        s.push(self.chan(kind, d_out, &gain, &dy, total, nout, 1));
        base_dx(s, &dy);
        self.lora_backward(gemm, s, p, base, (an.clone(), bn.clone()), &dy, x, dx, m, k, nout);

        // The norm path into the direction.
        let rr = (r * nout) as u64;
        let (m1, d_b, e, et) = (self.scratch(17, rr), self.scratch(18, rr), self.scratch(19, rr), self.scratch(20, rr));
        s.push(self.chan(kind, &cm, &self.ones, &m1, nout * r, nout * r, 1));
        s.push(g.step(self.k.axpy, &[&m1, &bg], &[nout * r, f(scale)], nout * r));
        s.push(self.chan(kind, &m1, &cvec, &d_b, nout * r, nout, r));
        s.push(g.step(self.k.grad_scale, &[&d_b], &[nout * r, f(scale)], nout * r));
        let gb = p.grad(&bn).unwrap_or_else(|| panic!("adapter runtime: {bn} has no gradient buffer"));
        s.push(g.step(self.k.axpy, &[gb, &d_b], &[nout * r, f(1.0)], nout * r));
        if let Some(ga) = p.grad(&an) {
            s.push(self.chan(kind, b, &cvec, &e, nout * r, nout, r));
            s.push(self.swap(&e, &et, 1, nout, r));
            let (f1, f2, etb) = (self.scratch(21, (r * k) as u64), self.scratch(22, (r * k) as u64), self.scratch(23, (r * r) as u64));
            s.push(self.gpu.step(self.k.dx, &[&et, w, &f1], &[r, k, nout, 0], r * k));
            s.push(g.step(self.k.grad_scale, &[&etb], &[r * r, f(0.0)], r * r));
            s.push(self.gpu.step(self.k.dw, &[&e, b, &etb], &[nout, r, r], r * r));
            s.push(self.gpu.step(self.k.dx, &[&etb, a, &f2], &[r, k, r, 0], r * k));
            s.push(g.step(self.k.axpy, &[ga, &f1], &[r * k, f(scale)], r * k));
            s.push(g.step(self.k.axpy, &[ga, &f2], &[r * k, f(scale * scale)], r * k));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_core::Dispatch;

    /// A model that did not register the tiled kernel must get the reference
    /// kernel with one thread per output: pairing it with a tile-count grid
    /// computed 64 outputs of a 12 x 128 result (a 4-bit base stopped learning).
    #[test]
    fn an_unregistered_tiled_kernel_falls_back_to_one_thread_per_output() {
        assert_eq!(tiled_or_reference(7, None, 12, 128), (7, Dispatch::Threads(12 * 128)));
        let (kern, grid) = tiled_or_reference(7, Some(9), 512, 512);
        assert_eq!((kern, grid), (9, Dispatch::Workgroups(16)), "a shape that fills tiles takes the tiled kernel");
        assert_eq!(tiled_or_reference(7, Some(9), 12, 64), (7, Dispatch::Threads(12 * 64)), "a thin shape stays on the reference kernel");
    }
}
