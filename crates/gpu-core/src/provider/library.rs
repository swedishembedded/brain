// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dense GEMMs on the vendor library's tensor cores, for the frozen bf16
//! weights of LoRA/QLoRA training, chosen by the device and falling back to
//! brain's own fp32 kernels wherever the device cannot run them.
//!
//! Swedish Embedded AB implements tensor-core training stacks that degrade
//! gracefully on hardware without them. If your team needs expertise in
//! selecting vendor GEMM libraries by queried capability, with an exact
//! reference path behind them, you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! # What decides the path
//!
//! [`GemmPath`] is decided once per device by [`decide`], from facts the
//! device reports and the caller's [`GemmPrecision`]:
//!
//! * the device reports a compute capability at all (a CUDA device);
//! * the backend can serve library products ([`crate::Gpu::library_gemm_unavailable`]:
//!   capability at the bf16 tensor-core floor, `libcublas` present, a handle
//!   that binds to the stream);
//! * the caller did not pin the fp32 reference tier.
//!
//! Anything else is [`GemmPath::Fp32Kernels`] with the reason, and the caller
//! runs the kernels it ran before this provider existed. Nothing here panics or
//! returns a typed error for a missing capability: a request the library path
//! cannot take ([`LibraryGemmProvider::forward`] and
//! [`LibraryGemmProvider::dx`] answer `false`) is served by the fp32 kernels
//! too. The chosen path is logged once per process.
//!
//! # Numerics
//!
//! The library path rounds the activation (or output gradient) operand to bf16
//! (round-to-nearest-even) and multiplies it with the stored bf16 weight in
//! fp32 accumulation, producing fp32: the recipe of mixed-precision LoRA/QLoRA
//! (QLoRA, arXiv 2305.14314). It is a tier chosen by [`GemmPrecision`], not a
//! silent change: gradient checks and cross-backend parity pin
//! [`GemmPrecision::Fp32Reference`] or run on backends that have no library.

use std::sync::atomic::{AtomicBool, Ordering};

use backend_api::{BindKind, LibraryDtype, LibraryGemm};
use kernels::template::DecodeFormat;

use crate::{DeviceBuffer, Gpu, Step};

/// Which arithmetic a training GEMM against a bf16 weight uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum GemmPrecision {
    /// Tensor cores wherever the device offers them, the fp32 kernels
    /// otherwise: bf16 products for bf16/NF4 weights and TF32 products
    /// (operands rounded to a 10-bit mantissa inside the multiply, binary32
    /// accumulation, no copy of either) for fp32 weights. The default: no
    /// flag, no failure on unsupported hardware.
    #[default]
    Auto,
    /// Always the fp32 kernels: the reference tier the gradient checks and
    /// cross-backend parity gates are stated against.
    Fp32Reference,
    /// Ask for the tensor-core path; a device that cannot run it still falls
    /// back (with a warning instead of an info line).
    Bf16TensorCore,
}

/// The arithmetic path a device was assigned.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum GemmPath {
    /// bf16 operands on the vendor library's tensor cores, fp32 accumulation.
    Bf16TensorCore,
    /// brain's own fp32 kernels; carries why the library path was not taken.
    Fp32Kernels {
        /// Human-readable reason, for the log and for tests.
        reason: String,
    },
}

/// Decide the path from what the device reported. Pure, so every fallback is
/// testable without hardware: `compute_capability` is the device's reported
/// capability (`None` on a non-CUDA backend), `library_unavailable` the
/// backend's reason it cannot serve library products (`None` when it can).
pub fn decide(precision: GemmPrecision, compute_capability: Option<(u32, u32)>, library_unavailable: Option<String>) -> GemmPath {
    let fp32 = |reason: String| GemmPath::Fp32Kernels { reason };
    if precision == GemmPrecision::Fp32Reference {
        return fp32("the fp32 reference tier was requested".to_string());
    }
    if compute_capability.is_none() {
        return fp32("the device reports no CUDA compute capability".to_string());
    }
    match library_unavailable {
        Some(why) => fp32(why),
        None => GemmPath::Bf16TensorCore,
    }
}

/// Elements of K (and of every leading dimension) a library product needs to be
/// a multiple of, so every operand row starts on a 16-byte boundary.
const ALIGN_ELEMS: u32 = 8;

/// Largest bf16 operand scratch one product may allocate. A bigger request is
/// declined (the fp32 kernels need no scratch) rather than risking the device
/// running out of memory mid-step.
const MAX_SCRATCH_BYTES: u64 = 2 << 30;

const CAST_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageReadWrite];
const SILU_MUL_BF16_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite, BindKind::StorageReadWrite];
const DEQUANT_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// Serves the bf16-weight GEMMs of one device through the vendor library when
/// [`decide`] chose it; otherwise answers `false` to every request.
#[derive(Clone)]
pub struct LibraryGemmProvider {
    path: GemmPath,
    /// Fp32-stored weights also run on tensor cores as TF32 (the device runs the library path
    /// and the precision is not the pinned fp32 reference).
    tf32: bool,
    /// The caller pinned the exact fp32 tier (it also bars the native fp16 attention kernels).
    reference: bool,
}

/// A window of rows of a packed weight to decode ([`LibraryGemmProvider::dequant_bf16`]): the planes the
/// tier stores, bound at their ranges (a tier leaves the planes it does not use bound to any buffer).
pub struct Dequant<'a> {
    /// Which stored layout `codes` and the planes follow (`weight_decode.wgsl` documents each).
    pub format: DecodeFormat,
    /// The packed codes.
    pub codes: Operand<'a>,
    /// f32 scales (int8/int4/NF4/Q6_K groups, FP8 block grid).
    pub scales: Operand<'a>,
    /// Packed per-group sub-scales (affine K-quant).
    pub sub_scales: Operand<'a>,
    /// Packed per-super-block scales (affine K-quant).
    pub super_scales: Operand<'a>,
    /// First row of the window.
    pub row0: u32,
    /// Rows in the window.
    pub rows: u32,
    /// Row width.
    pub k: u32,
    /// FP8 scale block edge (1 for the other tiers).
    pub block: u32,
    /// The per-tensor f32 scale of NVFP4 (1.0 for every other tier).
    pub global: f32,
}

/// A product operand: a buffer and the `(word offset, word length)` it binds.
pub struct Operand<'a> {
    /// The buffer.
    pub buf: &'a DeviceBuffer,
    /// `(word offset, word length)`; length 0 means to the end.
    pub range: (u64, u64),
}

/// The geometry of a strided batch of row-major products (see [`LibraryGemmProvider::bmm_nt_tf32`]): `m x n` outputs
/// over a contraction of `k`, each matrix's row stride (`ld*`, in elements) and the distance between consecutive
/// matrices of the batch (`stride_*`, in elements).
#[derive(Clone, Copy, Debug)]
pub struct BatchedProduct {
    pub batch: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lda: u32,
    pub ldb: u32,
    pub ldc: u32,
    pub stride_a: i64,
    pub stride_b: i64,
    pub stride_c: i64,
    /// Scale of the product.
    pub alpha: f32,
}

impl LibraryGemmProvider {
    /// The provider for `gpu` under `precision`, reading the backend's own
    /// availability answer. Logs the decision once per process.
    pub fn for_gpu(gpu: &Gpu, precision: GemmPrecision) -> LibraryGemmProvider {
        // The pinned reference tier never asks the backend: probing loads the
        // library and binds a handle, which a model that will not use it (every
        // inference build) should not pay for.
        let unavailable = if precision == GemmPrecision::Fp32Reference { None } else { gpu.library_gemm_unavailable() };
        Self::with_probe(gpu, precision, unavailable)
    }

    /// [`Self::for_gpu`] with the backend's availability answer supplied: how a
    /// test simulates a missing library or a too-old device.
    pub fn with_probe(gpu: &Gpu, precision: GemmPrecision, library_unavailable: Option<String>) -> LibraryGemmProvider {
        let path = decide(precision, gpu.caps().arch.compute_capability, library_unavailable);
        // A pinned reference tier is a choice, not a finding: it must not use
        // up the once-per-process report of what the device can do.
        if precision != GemmPrecision::Fp32Reference {
            report_once(&path, precision);
        }
        let tf32 = path == GemmPath::Bf16TensorCore;
        LibraryGemmProvider { path, tf32, reference: precision == GemmPrecision::Fp32Reference }
    }

    /// The path this provider was assigned.
    pub fn path(&self) -> &GemmPath {
        &self.path
    }

    /// Whether this provider will take requests at all (the device runs the
    /// tensor-core path); a caller preparing an operand for it checks first.
    pub fn is_active(&self) -> bool {
        self.path == GemmPath::Bf16TensorCore
    }

    /// Whether the exact fp32 reference tier was pinned: gradient checks and parity gates pin it, and
    /// every native tensor-core kernel of the same pass (attention included) is then declined too.
    pub fn is_reference_pinned(&self) -> bool {
        self.reference
    }

    /// Whether fp32-stored weights take the TF32 tensor-core products
    /// ([`Self::forward_tf32`], [`Self::dx_tf32`]).
    pub fn tf32_enabled(&self) -> bool {
        self.tf32
    }

    /// Decode `elems` weights of an NF4 window (`packed`: eight codes a word;
    /// `scales`: one fp32 per 32 weights, both bound at their ranges) into a new
    /// bf16 buffer and push the step. The result is a whole-buffer bf16 weight
    /// for [`Self::forward`] / [`Self::dx`], valid until the step has run.
    /// `None` when the device declines the decode kernel or the window is not a
    /// whole number of 32-weight groups. [`Self::dequant_bf16`] with the NF4 format.
    pub fn dequant_nf4(&self, gpu: &Gpu, steps: &mut Vec<Step>, packed: &Operand, scales: &Operand, elems: u64) -> Option<DeviceBuffer> {
        if !elems.is_multiple_of(32) || elems > u32::MAX as u64 || !self.can_dequant_bf16(1, elems as u32) {
            return None;
        }
        let unused = Operand { buf: packed.buf, range: (0, 0) };
        let window = Dequant {
            format: DecodeFormat::Nf4,
            codes: Operand { buf: packed.buf, range: packed.range },
            scales: Operand { buf: scales.buf, range: scales.range },
            sub_scales: Operand { buf: unused.buf, range: unused.range },
            super_scales: Operand { buf: unused.buf, range: unused.range },
            row0: 0,
            rows: 1,
            k: elems as u32,
            block: 1,
            global: 1.0,
        };
        // The decode kernel writes every word, so the clear `storage` would add is skipped.
        let out = gpu.storage_uninit(elems / 2);
        self.dequant_bf16(gpu, steps, &window, &out).then_some(out)
    }

    /// Whether [`Self::dequant_bf16`] takes a window of `rows` rows of width `k`: the device runs the
    /// tensor-core path, `k` is a multiple of 8 and the bf16 window fits the scratch cap.
    pub fn can_dequant_bf16(&self, rows: u32, k: u32) -> bool {
        let elems = rows as u64 * k as u64;
        self.is_active() && elems > 0 && k.is_multiple_of(8) && elems * 2 <= MAX_SCRATCH_BYTES
    }

    /// Decode rows `row0 .. row0 + rows` of a packed weight of any tier ([`DecodeFormat`]) into `out`, a
    /// bf16 buffer of at least `rows * k / 2` words, and push the step: the operand a tensor-core GEMM
    /// reads in place of the packed bytes. The caller owns `out`, so a loop over row tiles reuses one
    /// scratch whatever the size of the tensor. `false` (nothing pushed) when [`Self::can_dequant_bf16`]
    /// says no or the device declines the kernel.
    pub fn dequant_bf16(&self, gpu: &Gpu, steps: &mut Vec<Step>, d: &Dequant, out: &DeviceBuffer) -> bool {
        if !self.can_dequant_bf16(d.rows, d.k) {
            return false;
        }
        let elems = d.rows as u64 * d.k as u64;
        let Some(kernel) = kernels_cuda::get("weight_dequant_bf16") else { return false };
        let Some(id) = gpu.native_kernel(kernel, DEQUANT_BINDINGS) else { return false };
        let params = [d.format as u32, d.row0, d.rows, d.k, d.block, d.k.div_ceil(d.block.max(1)), d.k.div_ceil(32).div_ceil(2), d.k.div_ceil(256), d.global.to_bits()];
        let step = gpu.step_native_sliced(
            id,
            &[d.codes.buf, d.scales.buf, d.sub_scales.buf, d.super_scales.buf, out],
            &[d.codes.range, d.scales.range, d.sub_scales.range, d.super_scales.range, (0, 0)],
            &params,
            kernel.blocks_for(1, (elems / 8) as u32),
        );
        match step {
            Some(step) => {
                steps.push(step);
                true
            }
            None => false,
        }
    }

    /// `y[m, n] = x[m, k] @ W[n, k]^T` for a bf16 `w`, with the activation
    /// rounded to bf16. `false` (nothing pushed) when this device runs the fp32
    /// kernels or the shape is outside what the library path takes.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, gpu: &Gpu, steps: &mut Vec<Step>, x: &Operand, w: &Operand, y: &Operand, m: u32, n: u32, k: u32) -> bool {
        self.forward_with(gpu, steps, ActSource::Fp32(x), w, y, m, n, k)
    }

    /// [`Self::forward`] over an activation already rounded to bf16 by
    /// [`Self::cast_bf16`] (several linears reading one activation share the cast).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_bf16(&self, gpu: &Gpu, steps: &mut Vec<Step>, x: &DeviceBuffer, w: &Operand, y: &Operand, m: u32, n: u32, k: u32) -> bool {
        self.forward_with(gpu, steps, ActSource::Bf16(x), w, y, m, n, k)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with(&self, gpu: &Gpu, steps: &mut Vec<Step>, x: ActSource, w: &Operand, y: &Operand, m: u32, n: u32, k: u32) -> bool {
        // Row-major `y = x W^T` is column-major `y^T[n, m] = W[k, n]^T x^T[k, m]`.
        let gemm = LibraryGemm {
            trans_a: true,
            trans_b: false,
            m: n,
            n: m,
            k,
            lda: k,
            ldb: k,
            ldc: n,
            a: LibraryDtype::Bf16,
            b: LibraryDtype::Bf16,
            c: LibraryDtype::F32,
            alpha: 1.0,
            beta: 0.0,
            batch: 1,
            stride_a: 0,
            stride_b: 0,
            stride_c: 0,
            tf32: false,
        };
        self.product(gpu, steps, x, (m, k), w, y, gemm)
    }

    /// `y[m, n] = x[m, k] @ W[n, k]^T` with fp32 `x` and fp32 `W` multiplied as TF32 on tensor cores
    /// (binary32 accumulation, binary32 output); nothing is copied or cast. `false` (nothing pushed)
    /// unless [`Self::tf32_enabled`] and the operands are 16-byte aligned.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tf32(&self, gpu: &Gpu, steps: &mut Vec<Step>, x: &Operand, w: &Operand, y: &Operand, m: u32, n: u32, k: u32) -> bool {
        // Row-major `y = x W^T` is column-major `y^T[n, m] = W[k, n]^T x^T[k, m]`.
        let gemm = Self::tf32_product(true, m, n, k, k, k, n, 0.0);
        self.tf32_step(gpu, steps, &gemm, &[w, x, y])
    }

    /// `dx[m, k] (+)= dy[m, n] @ W[n, k]` as [`Self::forward_tf32`] multiplies; `accumulate` adds to `dx`.
    #[allow(clippy::too_many_arguments)]
    pub fn dx_tf32(&self, gpu: &Gpu, steps: &mut Vec<Step>, dy: &Operand, w: &Operand, dx: &Operand, m: u32, n: u32, k: u32, accumulate: bool) -> bool {
        // Row-major `dx = dy W` is column-major `dx^T[k, m] = W[k, n] dy^T[n, m]`.
        let gemm = Self::tf32_product(false, m, k, n, k, n, k, if accumulate { 1.0 } else { 0.0 });
        self.tf32_step(gpu, steps, &gemm, &[w, dy, dx])
    }

    /// `dw[n, k] (+)= dy[m, n]^T @ x[m, k]` as [`Self::forward_tf32`] multiplies: the weight gradient of a
    /// linear, fp32 in and out, TF32 on the tensor cores; `accumulate` adds to `dw`.
    #[allow(clippy::too_many_arguments)]
    pub fn dw_tf32(&self, gpu: &Gpu, steps: &mut Vec<Step>, dy: &Operand, x: &Operand, dw: &Operand, m: u32, n: u32, k: u32, accumulate: bool) -> bool {
        // Row-major `dw = dy^T x` is column-major `dw^T[k, n] = x[k, m] dy[n, m]^T` (x and dy as stored are
        // column-major `[k, m]` and `[n, m]`).
        let mut gemm = Self::tf32_product(false, n, k, m, k, n, k, if accumulate { 1.0 } else { 0.0 });
        gemm.trans_b = true;
        self.tf32_step(gpu, steps, &gemm, &[x, dy, dw])
    }

    /// A strided batch of row-major fp32 products `C[i] = alpha . A[i] B[i]^T` on tensor cores as TF32 (binary32
    /// accumulation): `A[i]` is `m x k`, `B[i]` is `n x k`, `C[i]` is `m x n`, and entry `i` of each operand starts
    /// `stride` elements after the first. Attention's `Q K^T` over heads, whose operands are slices of one fused
    /// `[rows, 3 * d]` buffer, is this with `lda == ldb == 3 * d` and a stride of one head. `false` (nothing pushed)
    /// when the device runs no TF32 products or an operand is not 16-byte aligned.
    pub fn bmm_nt_tf32(&self, gpu: &Gpu, steps: &mut Vec<Step>, a: &Operand, b: &Operand, c: &Operand, shape: &BatchedProduct) -> bool {
        // Row-major `C = A B^T` is column-major `C^T[n, m] = B[k, n]^T A[k, m]`.
        let mut gemm = Self::tf32_product(true, shape.m, shape.n, shape.k, shape.ldb, shape.lda, shape.ldc, 0.0);
        Self::batch(&mut gemm, shape, shape.stride_b, shape.stride_a);
        self.tf32_step(gpu, steps, &gemm, &[b, a, c])
    }

    /// [`Self::bmm_nt_tf32`] for `C[i] = alpha . A[i] B[i]` with `B[i]` of shape `k x n`: attention's `P V`.
    pub fn bmm_nn_tf32(&self, gpu: &Gpu, steps: &mut Vec<Step>, a: &Operand, b: &Operand, c: &Operand, shape: &BatchedProduct) -> bool {
        // Row-major `C = A B` is column-major `C^T[n, m] = B[n, k] A[k, m]`.
        let mut gemm = Self::tf32_product(false, shape.m, shape.n, shape.k, shape.ldb, shape.lda, shape.ldc, 0.0);
        Self::batch(&mut gemm, shape, shape.stride_b, shape.stride_a);
        self.tf32_step(gpu, steps, &gemm, &[b, a, c])
    }

    fn batch(gemm: &mut LibraryGemm, shape: &BatchedProduct, stride_a: i64, stride_b: i64) {
        gemm.batch = shape.batch;
        gemm.alpha = shape.alpha;
        gemm.stride_a = stride_a;
        gemm.stride_b = stride_b;
        gemm.stride_c = shape.stride_c;
    }

    /// The column-major product of `forward_tf32`/`dx_tf32`: `C[rows, cols] = op(A) B` over `depth`.
    #[allow(clippy::too_many_arguments)]
    fn tf32_product(trans_a: bool, cols: u32, rows: u32, depth: u32, lda: u32, ldb: u32, ldc: u32, beta: f32) -> LibraryGemm {
        LibraryGemm {
            trans_a,
            trans_b: false,
            m: rows,
            n: cols,
            k: depth,
            lda,
            ldb,
            ldc,
            a: LibraryDtype::F32,
            b: LibraryDtype::F32,
            c: LibraryDtype::F32,
            alpha: 1.0,
            beta,
            batch: 1,
            stride_a: 0,
            stride_b: 0,
            stride_c: 0,
            tf32: true,
        }
    }

    fn tf32_step(&self, gpu: &Gpu, steps: &mut Vec<Step>, gemm: &LibraryGemm, ops: &[&Operand; 3]) -> bool {
        // Sub-ranges must stay 16-byte aligned (4 words), as for the bf16 products.
        if !self.tf32 || ![gemm.m, gemm.n, gemm.k].iter().all(|&d| d > 0) || !ops.iter().all(|o| o.range.0.is_multiple_of(4)) {
            return false;
        }
        match gpu.step_library_gemm(gemm, &[ops[0].buf, ops[1].buf, ops[2].buf], &[ops[0].range, ops[1].range, ops[2].range]) {
            Some(step) => {
                steps.push(step);
                true
            }
            None => false,
        }
    }

    /// Round `count` fp32 values of `x` to a new bf16 buffer (the cast step is
    /// pushed), for [`Self::forward_bf16`]. `None` when the device declines.
    pub fn cast_bf16(&self, gpu: &Gpu, steps: &mut Vec<Step>, x: &Operand, count: u64) -> Option<DeviceBuffer> {
        if !self.is_active() || count == 0 || !count.is_multiple_of(ALIGN_ELEMS as u64) || count * 2 > MAX_SCRATCH_BYTES || !x.range.0.is_multiple_of(4) {
            return None;
        }
        let kernel = kernels_cuda::get("cast_f32_bf16")?;
        let id = gpu.native_kernel(kernel, CAST_BINDINGS)?;
        // The cast writes all `count` values (an even multiple of ALIGN_ELEMS), so the scratch is not cleared.
        let scratch = gpu.storage_uninit(count.div_ceil(2));
        let step = gpu.step_native_sliced(id, &[x.buf, &scratch], &[(x.range.0, count), (0, 0)], &[count as u32], kernel.blocks_for(1, count as u32))?;
        steps.push(step);
        Some(scratch)
    }

    /// `h = silu(gate) * up` and its bf16 rounding (the operand [`Self::forward_bf16`] reads) in one pass over
    /// `count` values: pushes the step and returns the bf16 buffer, with `h` (fp32) filled as `silu_mul` would.
    /// `None` when the device declines or `count` is not one the cast would take; nothing is pushed.
    pub fn silu_mul_bf16(&self, gpu: &Gpu, steps: &mut Vec<Step>, gate: &DeviceBuffer, up: &DeviceBuffer, h: &DeviceBuffer, count: u64) -> Option<DeviceBuffer> {
        if !self.is_active() || count == 0 || !count.is_multiple_of(ALIGN_ELEMS as u64) || count * 2 > MAX_SCRATCH_BYTES {
            return None;
        }
        let kernel = kernels_cuda::get("silu_mul_bf16")?;
        let id = gpu.native_kernel(kernel, SILU_MUL_BF16_BINDINGS)?;
        let bf16 = gpu.storage_uninit(count.div_ceil(2)); // the kernel writes every value, as `cast_bf16` does
        let step = gpu.step_native_sliced(id, &[gate, up, h, &bf16], &[(0, 0); 4], &[count as u32], kernel.blocks_for(1, count as u32))?;
        steps.push(step);
        Some(bf16)
    }

    /// `dx[m, k] (+)= dy[m, n] @ W[n, k]` for a bf16 `w`, with `dy` rounded to
    /// bf16; `accumulate` adds to the existing `dx`. Same `false` contract as
    /// [`Self::forward`].
    #[allow(clippy::too_many_arguments)]
    pub fn dx(&self, gpu: &Gpu, steps: &mut Vec<Step>, dy: &Operand, w: &Operand, dx: &Operand, m: u32, n: u32, k: u32, accumulate: bool) -> bool {
        // Row-major `dx = dy W` is column-major `dx^T[k, m] = W[k, n] dy^T[n, m]`.
        let gemm = LibraryGemm {
            trans_a: false,
            trans_b: false,
            m: k,
            n: m,
            k: n,
            lda: k,
            ldb: n,
            ldc: k,
            a: LibraryDtype::Bf16,
            b: LibraryDtype::Bf16,
            c: LibraryDtype::F32,
            alpha: 1.0,
            beta: if accumulate { 1.0 } else { 0.0 },
            batch: 1,
            stride_a: 0,
            stride_b: 0,
            stride_c: 0,
            tf32: false,
        };
        self.product(gpu, steps, ActSource::Fp32(dy), (m, n), w, dx, gemm)
    }

    /// Run `gemm` over the weight (its `A`), the activation rounded to bf16 (its
    /// `B`) and `out`.
    #[allow(clippy::too_many_arguments)]
    fn product(&self, gpu: &Gpu, steps: &mut Vec<Step>, act: ActSource, act_dims: (u32, u32), w: &Operand, out: &Operand, gemm: LibraryGemm) -> bool {
        if !self.is_active() {
            return false;
        }
        let (rows, cols) = act_dims;
        let count = rows as u64 * cols as u64;
        let eligible = [gemm.m, gemm.n, gemm.k].iter().all(|&d| d > 0)
            // Only the bf16 operands need 16-byte rows; the fp32 output may have any
            // row length (the ragged last window of a vocabulary-tiled head does).
            && [gemm.lda, gemm.ldb].iter().all(|&d| d.is_multiple_of(ALIGN_ELEMS))
            && cols.is_multiple_of(ALIGN_ELEMS)
            && count * 2 <= MAX_SCRATCH_BYTES
            // Bound sub-ranges must keep every operand 16-byte aligned (4 words).
            && [w.range.0, out.range.0].iter().all(|o| o.is_multiple_of(4));
        if !eligible {
            return false;
        }
        let mut prep = Vec::new();
        let scratch = match act {
            ActSource::Fp32(x) => match self.cast_bf16(gpu, &mut prep, x, count) {
                Some(b) => b,
                None => return false,
            },
            ActSource::Bf16(b) => b.clone(),
        };
        let Some(gemm_step) = gpu.step_library_gemm(&gemm, &[w.buf, &scratch, out.buf], &[w.range, (0, 0), out.range]) else {
            return false;
        };
        steps.append(&mut prep);
        steps.push(gemm_step);
        true
    }
}

/// Where a product's activation operand comes from.
enum ActSource<'a> {
    /// fp32 values to round to a bf16 scratch.
    Fp32(&'a Operand<'a>),
    /// A bf16 buffer an earlier [`LibraryGemmProvider::cast_bf16`] produced.
    Bf16(&'a DeviceBuffer),
}

/// A row-major fp32 product `C[m, n] = alpha * op(A) * op(B) + beta * C` for
/// [`LibraryGemmProvider::gemm_f32`]: `op(A)` is `m x k` (stored `[m, k]`, or
/// `[k, m]` when `trans_a`), `op(B)` is `k x n` (stored `[k, n]`, or `[n, k]`
/// when `trans_b`).
#[derive(Clone, Copy, Debug)]
pub struct F32Product {
    /// Rows of `C`.
    pub m: u32,
    /// Columns of `C`.
    pub n: u32,
    /// The reduction length.
    pub k: u32,
    /// `A` is stored transposed.
    pub trans_a: bool,
    /// `B` is stored transposed.
    pub trans_b: bool,
    /// Scale of the product.
    pub alpha: f32,
    /// Scale of the existing `C`: 0 overwrites, 1 accumulates.
    pub beta: f32,
    /// Round both operands to TF32 (10 mantissa bits) and run on tensor cores, accumulating in fp32. Off: no
    /// operand is rounded and the product runs on the fp32 CUDA cores.
    pub tf32: bool,
}

impl LibraryGemmProvider {
    /// A row-major fp32 product with fp32 accumulation: the small low-rank
    /// products of a LoRA adapter, whose serial kernels were the training
    /// step's second-largest cost. Operands are rounded only when
    /// [`F32Product::tf32`] asks for the tensor cores. `false` (nothing
    /// pushed) when this device runs the fp32 kernels.
    pub fn gemm_f32(&self, gpu: &Gpu, steps: &mut Vec<Step>, a: &Operand, b: &Operand, c: &Operand, p: F32Product) -> bool {
        if !self.is_active() || p.m == 0 || p.n == 0 || p.k == 0 {
            return false;
        }
        // Row-major `C = op(A) op(B)` is column-major `C^T = op(B)^T op(A)^T`: the
        // library's first operand is `B`, its second `A`, and a row-major
        // matrix is its column-major transpose with the row length as stride.
        let gemm = LibraryGemm {
            trans_a: p.trans_b,
            trans_b: p.trans_a,
            m: p.n,
            n: p.m,
            k: p.k,
            lda: if p.trans_b { p.k } else { p.n },
            ldb: if p.trans_a { p.m } else { p.k },
            ldc: p.n,
            a: LibraryDtype::F32,
            b: LibraryDtype::F32,
            c: LibraryDtype::F32,
            alpha: p.alpha,
            beta: p.beta,
            batch: 1,
            stride_a: 0,
            stride_b: 0,
            stride_c: 0,
            tf32: p.tf32,
        };
        match gpu.step_library_gemm(&gemm, &[b.buf, a.buf, c.buf], &[b.range, a.range, c.range]) {
            Some(step) => {
                steps.push(step);
                true
            }
            None => false,
        }
    }
}

static REPORTED: AtomicBool = AtomicBool::new(false);

/// Log the chosen path the first time any device decides one: info for the
/// automatic choice, a warning when the tensor-core tier was asked for by name
/// and the device cannot run it.
fn report_once(path: &GemmPath, precision: GemmPrecision) {
    if REPORTED.swap(true, Ordering::Relaxed) {
        return;
    }
    match (path, precision) {
        (GemmPath::Bf16TensorCore, _) => tracing::info!("gemm path: bf16 tensor cores through the vendor library (fp32 accumulation)"),
        (GemmPath::Fp32Kernels { reason }, GemmPrecision::Bf16TensorCore) => {
            tracing::warn!(%reason, "gemm path: the bf16 tensor-core tier was requested but is unavailable; using the fp32 kernels")
        }
        (GemmPath::Fp32Kernels { reason }, _) => tracing::info!(%reason, "gemm path: fp32 kernels"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOPPER: Option<(u32, u32)> = Some((9, 0));

    #[test]
    fn a_capable_device_with_the_library_takes_the_tensor_core_path() {
        assert_eq!(decide(GemmPrecision::Auto, HOPPER, None), GemmPath::Bf16TensorCore);
        assert_eq!(decide(GemmPrecision::Bf16TensorCore, HOPPER, None), GemmPath::Bf16TensorCore);
    }

    /// Each reason the library path is unavailable lands on the fp32 kernels,
    /// with that reason reported, for the automatic tier and the named one.
    #[test]
    fn every_missing_capability_falls_back_with_its_reason() {
        for precision in [GemmPrecision::Auto, GemmPrecision::Bf16TensorCore] {
            let missing_lib = decide(precision, HOPPER, Some("no usable cuBLAS".to_string()));
            assert!(matches!(&missing_lib, GemmPath::Fp32Kernels { reason } if reason.contains("cuBLAS")), "{missing_lib:?}");
            let old = decide(precision, Some((7, 5)), Some("compute capability 7.5 is below the 8.0 bf16 tensor-core floor".to_string()));
            assert!(matches!(&old, GemmPath::Fp32Kernels { reason } if reason.contains("7.5")), "{old:?}");
            let no_cuda = decide(precision, None, None);
            assert!(matches!(&no_cuda, GemmPath::Fp32Kernels { reason } if reason.contains("no CUDA")), "{no_cuda:?}");
        }
    }

    #[test]
    fn the_reference_tier_pins_fp32_even_on_a_capable_device() {
        assert!(matches!(decide(GemmPrecision::Fp32Reference, HOPPER, None), GemmPath::Fp32Kernels { .. }));
    }

    /// On a backend with no library (every one but CUDA) a request is answered
    /// `false` and pushes nothing, so the caller runs its fp32 kernels.
    #[test]
    fn a_backend_without_the_library_declines_every_request() {
        let gpu = crate::testgpu::dev(&[]);
        if gpu.kind() == "cuda" {
            return; // this box's test device is the CUDA backend; covered by the device test
        }
        let provider = LibraryGemmProvider::for_gpu(&gpu, GemmPrecision::Auto);
        assert!(matches!(provider.path(), GemmPath::Fp32Kernels { .. }));
        let buf = gpu.storage(1024);
        let op = Operand { buf: &buf, range: (0, 0) };
        let mut steps = Vec::new();
        assert!(!provider.forward(&gpu, &mut steps, &op, &op, &op, 8, 8, 8));
        assert!(steps.is_empty(), "a declined request pushes no step");
    }
}
