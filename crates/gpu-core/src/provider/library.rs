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

use crate::{DeviceBuffer, Gpu, Step};

/// Which arithmetic a training GEMM against a bf16 weight uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum GemmPrecision {
    /// bf16 tensor cores where the device offers them, the fp32 kernels
    /// otherwise. The default: no flag, no failure on unsupported hardware.
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
const NF4_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// Serves the bf16-weight GEMMs of one device through the vendor library when
/// [`decide`] chose it; otherwise answers `false` to every request.
#[derive(Clone)]
pub struct LibraryGemmProvider {
    path: GemmPath,
}

/// A product operand: a buffer and the `(word offset, word length)` it binds.
pub struct Operand<'a> {
    /// The buffer.
    pub buf: &'a DeviceBuffer,
    /// `(word offset, word length)`; length 0 means to the end.
    pub range: (u64, u64),
}

impl LibraryGemmProvider {
    /// The provider for `gpu` under `precision`, reading the backend's own
    /// availability answer. Logs the decision once per process.
    pub fn for_gpu(gpu: &Gpu, precision: GemmPrecision) -> LibraryGemmProvider {
        Self::with_probe(gpu, precision, gpu.library_gemm_unavailable())
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
        LibraryGemmProvider { path }
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

    /// Decode `elems` weights of an NF4 window (`packed`: eight codes a word;
    /// `scales`: one fp32 per 32 weights, both bound at their ranges) into a new
    /// bf16 buffer and push the step. The result is a whole-buffer bf16 weight
    /// for [`Self::forward`] / [`Self::dx`], valid until the step has run.
    /// `None` when the device declines the decode kernel or the window is not a
    /// whole number of 32-weight groups.
    pub fn dequant_nf4(&self, gpu: &Gpu, steps: &mut Vec<Step>, packed: &Operand, scales: &Operand, elems: u64) -> Option<DeviceBuffer> {
        if !self.is_active() || elems == 0 || !elems.is_multiple_of(32) || elems > MAX_SCRATCH_BYTES / 2 {
            return None;
        }
        let words = elems / 8;
        let kernel = kernels_cuda::get("nf4_dequant_bf16")?;
        let id = gpu.native_kernel(kernel, NF4_BINDINGS)?;
        let out = gpu.storage(elems / 2);
        let step = gpu.step_native_sliced(
            id,
            &[packed.buf, scales.buf, &out],
            &[(packed.range.0, words), (scales.range.0, elems / 32), (0, 0)],
            &[words as u32],
            kernel.blocks_for(1, words as u32),
        )?;
        steps.push(step);
        Some(out)
    }

    /// `y[m, n] = x[m, k] @ W[n, k]^T` for a bf16 `w`, with the activation
    /// rounded to bf16. `false` (nothing pushed) when this device runs the fp32
    /// kernels or the shape is outside what the library path takes.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, gpu: &Gpu, steps: &mut Vec<Step>, x: &Operand, w: &Operand, y: &Operand, m: u32, n: u32, k: u32) -> bool {
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
        };
        self.product(gpu, steps, x, (m, k), w, y, gemm)
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
        };
        self.product(gpu, steps, dy, (m, n), w, dx, gemm)
    }

    /// Round `act` (`rows x cols`, fp32) to a bf16 scratch and run `gemm` over
    /// the weight (its `A`), the scratch (its `B`) and `out`.
    #[allow(clippy::too_many_arguments)]
    fn product(&self, gpu: &Gpu, steps: &mut Vec<Step>, act: &Operand, act_dims: (u32, u32), w: &Operand, out: &Operand, gemm: LibraryGemm) -> bool {
        if self.path != GemmPath::Bf16TensorCore {
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
            && [act.range.0, w.range.0, out.range.0].iter().all(|o| o.is_multiple_of(4));
        if !eligible {
            return false;
        }
        let Some(kernel) = kernels_cuda::get("cast_f32_bf16") else { return false };
        let Some(cast) = gpu.native_kernel(kernel, CAST_BINDINGS) else { return false };
        let scratch = gpu.storage(count.div_ceil(2));
        let Some(cast_step) = gpu.step_native_sliced(
            cast,
            &[act.buf, &scratch],
            &[(act.range.0, count), (0, 0)],
            &[count as u32],
            kernel.blocks_for(1, count as u32),
        ) else {
            return false;
        };
        let Some(gemm_step) = gpu.step_library_gemm(&gemm, &[w.buf, &scratch, out.buf], &[w.range, (0, 0), out.range]) else {
            return false;
        };
        steps.push(cast_step);
        steps.push(gemm_step);
        true
    }
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
}

impl LibraryGemmProvider {
    /// A row-major fp32 product with fp32 accumulation, no rounding of any
    /// operand: the small low-rank products of a LoRA adapter, whose serial
    /// kernels were the training step's second-largest cost. `false` (nothing
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
