// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `CudaProvider` - the [`super::OperatorProvider`] that answers an operator
//! with a **hand-written CUDA kernel** from `kernels_cuda`'s registry,
//! resolved against the compute capability the driver reported for the device
//! it is about to run on.
//!
//! Swedish Embedded AB implements architecture-specialised GPU operator
//! libraries and the dispatch machinery that keeps them honest. If your team
//! needs expertise in getting a hand-tuned kernel onto real silicon without
//! losing the portable path that proves it right, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! # What decides whether this provider runs
//!
//! Three questions, all answered by the DEVICE, none by this file:
//!
//! 1. what compute capability did the driver report
//!    (`DeviceCaps::arch::compute_capability`)? That is met against each
//!    kernel's own declared floor by `kernels_cuda::best_for`, highest
//!    eligible floor first - so a card gets the most specialised kernel it
//!    can run and a card below every floor gets none.
//! 2. can the device host the kernel's launch geometry - threads per block
//!    and `__shared__` bytes, both queried capabilities? `register_native`
//!    checks this and answers `None`, which is a decline, not a failure.
//! 3. is the backend one that can compile CUDA C++ at all? Every other
//!    backend answers `None` to `register_native` for a
//!    `NativeSpec::Cuda`, so this provider declines there by the same
//!    mechanism rather than by asking what backend it is on.
//!
//! Nothing in this module names a card, an architecture or a capability
//! value. [`CudaProvider::new`] takes the capability as an argument for the
//! same reason the tier policy does: a value that is queried cannot be a
//! value that is assumed.
//!
//! # Scope
//!
//! Two requests, both `Op::MatMul`, forward only:
//!
//! * plain f32, `[Act, Weight, Out]` - the operand bundle
//!   `model::ops::Ops::matmul`'s own F32 arm builds, with the same uniform
//!   (`[m, k, n]`) the portable `matmul.wgsl` reads;
//! * the int8 dynamic-activation GEMM, `[Act, Weight, ActScale, WeightScale,
//!   Out]` with group-32 weight scales - the prefill GEMM of every int8
//!   resident model, run on int8 tensor cores (`matmul_i8_mma`). It is taken
//!   only above the GEMV's crossover ([`select::I8_GEMV_MAX_ROWS`] rows -
//!   below that the GEMV family is the right shape), only when K is a
//!   whole number of 64-element tiles, and only on a device whose capability
//!   reaches the kernel's own floor.
//!
//! Anything else (another weight tier or scale group, a backward pass,
//! another operator) is declined by [`CudaProvider::accepts`] and served by
//! the WGSL reference provider exactly as before - a decline, never a forced
//! failure and never a silently different answer.

use backend_api::select::{self, Dtype};
use backend_api::{BindKind, DType, ImplSource};
use kernels_cuda::{CudaKernel, Cc};

use super::{LowerCtx, Lowered, OpRequest, OperatorProvider, Pass, Role};

/// `matmul.wgsl`'s own `@binding` order, which the f32 GEMM mirrors: uniform,
/// activations, weights, output.
const MATMUL_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// `matmul_i8_dyn.wgsl`'s own `@binding` order: uniform, packed activations,
/// packed weights, per-token activation scale, group weight scale, output.
const MATMUL_I8_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// `paged_flash_prefill_hd256.wgsl`'s own `@binding` order: uniform, queries,
/// K pool, V pool, block tables, sequence lengths, context output.
const PAGED_FLASH_PREFILL_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// `gdn_chunk_loop_f32`'s binding order: uniform, then key, query, w, u, g_cs,
/// exp_g_cs, intra scores and the initial state (read), then the output and the
/// final state (written).
const GDN_CHUNK_LOOP_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// [`GDN_CHUNK_LOOP_BINDINGS`] followed by the two history buffers the training
/// entry writes.
const GDN_CHUNK_LOOP_TRAIN_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// Elements of K per weight scale the int8 tensor-core kernel folds at (one
/// `k32` MMA). A layout with a different group is declined, not approximated.
const I8_MMA_GROUP: u32 = 32;

/// Elements of K per staged tile of the int8 tensor-core kernel (two groups).
const I8_MMA_K_TILE: u32 = 64;

pub struct CudaProvider {
    /// The capability the DRIVER reported for the device this provider was
    /// built for. Never a default: a provider with no capability to resolve
    /// against cannot be constructed (see [`CudaProvider::for_gpu`]).
    cc: Cc,
}

impl CudaProvider {
    /// A provider that resolves kernels against compute capability `cc`.
    pub fn new(cc: Cc) -> CudaProvider {
        CudaProvider { cc }
    }

    /// The provider for `gpu`, or `None` when this handle's device reports no
    /// compute capability at all.
    ///
    /// "Reports none" is the honest gate, rather than "is not the CUDA
    /// backend": the capability is a queried device fact, and a backend that
    /// does not publish one has nothing for `kernels_cuda::best_for` to
    /// resolve against. A handle that publishes one but cannot compile CUDA
    /// C++ still declines - one layer down, at `register_native` - so no
    /// dispatch can reach a kernel the device never took.
    pub fn for_gpu(gpu: &crate::Gpu) -> Option<CudaProvider> {
        gpu.caps().arch.compute_capability.map(CudaProvider::new)
    }

    /// The compute capability this provider resolves against.
    pub fn compute_capability(&self) -> Cc {
        self.cc
    }

    /// The registry entry this provider would run for `op` at weight tier
    /// `dtype` on its device, or `None` when the registry offers nothing for
    /// that operator and tier at this capability.
    pub fn kernel(&self, op: select::Op, dtype: Dtype) -> Option<&'static CudaKernel> {
        if dtype == Dtype::I8 {
            // The int8 tensor-core GEMM is asked for by name: the registry's
            // (MatMul, I8) answer stays the decode GEMV `native_upgrade` resolves.
            return kernels_cuda::get("matmul_i8_mma").filter(|k| k.min_cc <= self.cc);
        }
        kernels_cuda::find(op, dtype, self.cc)
    }

    /// The int8 tensor-core GEMM for `rows` activation rows: the 64x64-tile
    /// `matmul_i8_mma` for prefill-sized row counts, and the smaller-tile
    /// siblings (`_m16`, `_m32`, `_m128`) for the row counts of a batched
    /// decode step, where a big tile leaves most of the device without a block
    /// and the weights stream at a fraction of the memory bandwidth. All are
    /// the same source at a different tile, so the result is bit-identical.
    pub fn i8_mma_kernel(&self, rows: u32) -> Option<&'static CudaKernel> {
        let name = match rows {
            0..=16 => "matmul_i8_mma_m16",
            17..=32 => "matmul_i8_mma_m32",
            33..=128 => "matmul_i8_mma_m128",
            _ => "matmul_i8_mma",
        };
        kernels_cuda::get(name).filter(|k| k.min_cc <= self.cc)
    }

    /// [`Self::kernel`]'s registry name - what a test or a diagnostic reports
    /// without having to reach into the registry itself.
    pub fn kernel_name(&self, op: select::Op, dtype: Dtype) -> Option<&'static str> {
        self.kernel(op, dtype).map(|k| k.name)
    }

    /// The weight tier of the kernel `req` is shaped for, or `None` when it
    /// is neither request this provider implements.
    ///
    /// Checked structurally rather than trusted: `shape.dtype` names the
    /// WEIGHT tier, and the operand bundle is what the kernel actually binds.
    /// A quantized bundle has five operands with two scale planes in the
    /// middle; binding those to a three-pointer kernel would read scales as
    /// weights - a wrong answer, not a crash. The same holds for the uniform:
    /// the int8 kernel reads `[m, k/4, n]`, and a request whose uniform
    /// disagrees with its own shape is refused rather than launched.
    fn shape_of(req: &OpRequest) -> Option<Dtype> {
        let roles = |want: &[(Role, DType)]| {
            req.operands.len() == want.len() && req.operands.iter().zip(want).all(|(o, (r, d))| o.role == *r && o.dtype == *d)
        };
        match req.shape.dtype {
            Dtype::F32 if roles(&[(Role::Act, DType::F32), (Role::Weight, DType::F32), (Role::Out, DType::F32)]) => Some(Dtype::F32),
            Dtype::I8
                if roles(&[
                    (Role::Act, DType::I8),
                    (Role::Weight, DType::I8),
                    (Role::ActScale, DType::F32),
                    (Role::WeightScale, DType::F32),
                    (Role::Out, DType::F32),
                ]) && req.group == I8_MMA_GROUP
                    && req.shape.m > select::I8_GEMV_MAX_ROWS
                    && req.shape.k.is_multiple_of(I8_MMA_K_TILE)
                    && req.attrs == [req.shape.m, req.shape.k / 4, req.shape.n] =>
            {
                Some(Dtype::I8)
            }
            _ => None,
        }
    }
}

/// Whether `gpu` runs this provider's tensor-core kernels: its device reports a
/// compute capability at the int8 / fp16 MMA floor
/// ([`kernels_cuda::MMA_S8_MIN_CC`]) and `cuda` is not named in
/// `BRAIN_NO_PROVIDER`. A caller whose numerics depend on the tier (a test
/// stating a tolerance, a parity gate) asks this rather than guessing.
pub fn tensor_core_kernels_enabled(gpu: &crate::Gpu) -> bool {
    !super::disabled_providers().iter().any(|d| d == "cuda") && gpu.caps().arch.compute_capability.is_some_and(|cc| cc >= kernels_cuda::MMA_S8_MIN_CC)
}

/// Value-column blocks `gdn_chunk_loop_f32` splits each head across; its launch
/// is `bh * GDN_CHUNK_LOOP_SPLITS` blocks.
const GDN_CHUNK_LOOP_SPLITS: u32 = 4;

/// The native single-launch form of Gated DeltaNet's across-chunk recurrence, or
/// `None` when the caller should walk the chunks with the portable dispatch
/// sequence it was going to use anyway.
///
/// Like [`paged_flash_prefill_step`] this is a drop-in at one call site
/// (`model::gdn::gdn_chunk_fwd`), not a provider request. `None` means: the
/// head widths are not the kernel's `128`, the chunk is longer than its `64`
/// rows, `cuda` is named in `BRAIN_NO_PROVIDER`, the device reports no
/// capability (or one below the kernel's floor), or the backend declined it.
///
/// `bufs` is, in order, `key, query, w, u, g_cs, exp_g_cs, intra_scores,
/// initial_state, out, final_state` - each in the chunk-major layout
/// `model::gdn` documents. The kernel reads `initial_state` itself and writes
/// `final_state`, so the caller records no state copy before it.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_loop_step(
    gpu: &crate::Gpu,
    bufs: &[&backend_api::DeviceBuffer; 10],
    bh: u32,
    c_len: u32,
    dk: u32,
    dv: u32,
    n_chunks: u32,
    scale: f32,
) -> Option<crate::Step> {
    if dk != 128 || dv != 128 || c_len == 0 || c_len > 64 || super::disabled_providers().iter().any(|d| d == "cuda") {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::find(select::Op::GatedDeltaChunkLoop, Dtype::F32, cc)?;
    let id = gpu.native_kernel(kernel, GDN_CHUNK_LOOP_BINDINGS)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 10], &[bh, c_len, dk, dv, n_chunks, scale.to_bits()], bh * GDN_CHUNK_LOOP_SPLITS)
}

/// [`gdn_chunk_loop_step`] for the training forward: the same single-launch
/// recurrence that also writes `state_hist` (`[n_chunks + 1, bh, dk, dv]`, the
/// state entering each chunk and the final one) and `decay_hist`
/// (`[n_chunks, bh, c_len]`, each row's decay scale), which the backward sweep
/// reads. `bufs` is the inference order followed by those two.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_loop_train_step(
    gpu: &crate::Gpu,
    bufs: &[&backend_api::DeviceBuffer; 12],
    bh: u32,
    c_len: u32,
    dk: u32,
    dv: u32,
    n_chunks: u32,
    scale: f32,
) -> Option<crate::Step> {
    if dk != 128 || dv != 128 || c_len == 0 || c_len > 64 || super::disabled_providers().iter().any(|d| d == "cuda") {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get("gdn_chunk_loop_train_f32").filter(|k| k.min_cc <= cc)?;
    let id = gpu.native_kernel(kernel, GDN_CHUNK_LOOP_TRAIN_BINDINGS)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 12], &[bh, c_len, dk, dv, n_chunks, scale.to_bits()], bh * GDN_CHUNK_LOOP_SPLITS)
}

/// `gdn_bwd_sweep_f32`'s binding order: uniform, then the eight read buffers
/// (`v_new`, `decayed_k`, `q_scaled`, `w`, `d_out`, the state history, `g_cs`,
/// `d_final_state`), then the four written ones (`d_decayed_k`, `d_v_new`,
/// `d_g_cs`, `d_initial_state`).
const GDN_BWD_SWEEP_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// Dynamic shared memory of `gdn_bwd_sweep_f32`: the 128 x 129 state gradient,
/// two 16 x 129 operand tiles and one reduction float per warp (16), all f32.
const GDN_BWD_SWEEP_DYNAMIC_SHARED_BYTES: u32 = (128 * 129 + 2 * 16 * 129 + 16) * 4;

/// The native single-launch form of the reverse sweep of Gated DeltaNet's
/// across-chunk recurrence (`model::gdn::gdn_chunk_bwd`), or `None` when the
/// caller should issue its six dispatches per chunk. Declines on the head
/// widths, a chunk over 64 rows, `cuda` named in `BRAIN_NO_PROVIDER`, a device
/// with no reported capability, or a device that cannot give a block the shared
/// memory the state gradient needs.
///
/// `bufs` is, in order, `v_new, decayed_k, q_scaled, w, d_out, state_history,
/// g_cs, d_final_state` (read) and `d_decayed_k, d_v_new, d_g_cs,
/// d_initial_state` (written; `d_v_new` and `d_g_cs` are accumulated into).
#[allow(clippy::too_many_arguments)]
pub fn gdn_bwd_sweep_step(
    gpu: &crate::Gpu,
    bufs: &[&backend_api::DeviceBuffer; 12],
    bh: u32,
    c_len: u32,
    dk: u32,
    dv: u32,
    n_chunks: u32,
) -> Option<crate::Step> {
    if dk != 128 || dv != 128 || c_len == 0 || c_len > 64 || super::disabled_providers().iter().any(|d| d == "cuda") {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get("gdn_bwd_sweep_f32").filter(|k| k.min_cc <= cc)?;
    let launch = backend_api::CudaLaunch { dynamic_shared_bytes: GDN_BWD_SWEEP_DYNAMIC_SHARED_BYTES, shared_opt_in: true, ..backend_api::CudaLaunch::NONE };
    let id = gpu.native_kernel_launched(kernel, GDN_BWD_SWEEP_BINDINGS, launch)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 12], &[bh, c_len, dk, dv, n_chunks], bh)
}

/// `gdn_chunk_prep_tc`'s binding order: uniform, then query, key, value, raw_g
/// and beta (read), then g_cs, exp_g_cs, u, w and the intra-chunk scores
/// (written).
const GDN_CHUNK_PREP_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// [`GDN_CHUNK_PREP_BINDINGS`] followed by the chunk-major query and key the
/// token-major inference entry writes.
const GDN_CHUNK_PREP_TOK_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// [`GDN_CHUNK_PREP_BINDINGS`] followed by the seven tensors the training entry
/// also writes: k_beta, v_beta, k_beta_decay, decay_mask, raw_attn0, t_mat and
/// raw_intra.
const GDN_CHUNK_PREP_TRAIN_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// [`GDN_CHUNK_PREP_TRAIN_BINDINGS`] followed by the chunk-major query, key,
/// value and beta the token-major entry writes.
const GDN_CHUNK_PREP_TOK_TRAIN_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// Dynamic shared memory of `gdn_chunk_prep_tc`: the 64 x 132 key tile, the
/// 64 x 136 query / value tile, two 64 x 68 matrices (attn0 and the solved T)
/// and three 64-float vectors (cumulative decay, its exponential, beta), all f32.
const GDN_CHUNK_PREP_DYNAMIC_SHARED_BYTES: u32 = (64 * 132 + 64 * 136 + 2 * 64 * 68 + 3 * 64) * 4;

/// What the tensor-core chunk forward needs of the call: the head widths it is
/// written for, a chunk of at most 64 rows, no `cuda` in `BRAIN_NO_PROVIDER`.
fn gdn_chunk_prep_shape_ok(dk: u32, dv: u32, c_len: u32) -> bool {
    dk == 128 && dv == 128 && c_len != 0 && c_len <= 64 && !super::disabled_providers().iter().any(|d| d == "cuda")
}

/// The native tensor-core form of the state-independent half of Gated
/// DeltaNet's chunked forward (`model::gdn`'s steps 1-9 and the intra-chunk
/// score precompute), or `None` when the caller should issue the portable
/// whole-tensor dispatch sequence it was going to use anyway.
///
/// Like [`gdn_chunk_loop_step`] this is a drop-in at one call site, not a
/// provider request. `None` means: the head widths are not `128`, the chunk is
/// longer than `64` rows, `cuda` is named in `BRAIN_NO_PROVIDER`, the device
/// reports no capability or one below the tensor-core floor
/// ([`kernels_cuda::MMA_S8_MIN_CC`]), or the backend declined the kernel (for
/// instance a device that cannot give a block the shared memory it needs).
///
/// `bufs` is, in order, `query, key, value, raw_g, beta` (read) and `g_cs,
/// exp_g_cs, u, w, intra_scores` (written), each in the chunk-major layout
/// `model::gdn` documents; `bhc` is the number of (head, chunk) pairs.
pub fn gdn_chunk_prep_step(gpu: &crate::Gpu, bufs: &[&backend_api::DeviceBuffer; 10], bhc: u32, c_len: u32, dk: u32, dv: u32, scale: f32) -> Option<crate::Step> {
    if !gdn_chunk_prep_shape_ok(dk, dv, c_len) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get("gdn_chunk_prep_tc").filter(|k| k.min_cc <= cc)?;
    let launch = backend_api::CudaLaunch { dynamic_shared_bytes: GDN_CHUNK_PREP_DYNAMIC_SHARED_BYTES, shared_opt_in: true, ..backend_api::CudaLaunch::NONE };
    let id = gpu.native_kernel_launched(kernel, GDN_CHUNK_PREP_BINDINGS, launch)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 10], &[bhc, c_len, dk, dv, 0, scale.to_bits()], bhc)
}

/// [`gdn_chunk_prep_step`] for the training forward: the same launch that also
/// writes `k_beta, v_beta, k_beta_decay, decay_mask, raw_attn0, t_mat,
/// raw_intra`, which the backward reads. `bufs` is the inference order followed
/// by those seven.
pub fn gdn_chunk_prep_train_step(gpu: &crate::Gpu, bufs: &[&backend_api::DeviceBuffer; 17], bhc: u32, c_len: u32, dk: u32, dv: u32, scale: f32) -> Option<crate::Step> {
    if !gdn_chunk_prep_shape_ok(dk, dv, c_len) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get("gdn_chunk_prep_train_tc").filter(|k| k.min_cc <= cc)?;
    let launch = backend_api::CudaLaunch { dynamic_shared_bytes: GDN_CHUNK_PREP_DYNAMIC_SHARED_BYTES, shared_opt_in: true, ..backend_api::CudaLaunch::NONE };
    let id = gpu.native_kernel_launched(kernel, GDN_CHUNK_PREP_TRAIN_BINDINGS, launch)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 17], &[bhc, c_len, dk, dv, 0, scale.to_bits()], bhc)
}

/// Dynamic shared memory of `gdn_chunk_loop_tc`: two 64 x 132 staged tiles, the
/// 128 x 40 state slice, the 64 x 40 `v_new` and one 64-float decay vector, all f32.
const GDN_CHUNK_LOOP_TC_DYNAMIC_SHARED_BYTES: u32 = (2 * 64 * 132 + 128 * 40 + 64 * 40 + 64) * 4;

/// Value-column blocks `gdn_chunk_loop_tc` splits each head across.
const GDN_CHUNK_LOOP_TC_SPLITS: u32 = 4;

/// [`gdn_chunk_loop_step`] on tensor cores (`gdn_chunk_loop_tc`): same buffers and
/// contract, three-pass TF32 products instead of fp32 CUDA-core ones, so it
/// agrees with the fp32 kernel to about 1e-6 relative rather than bit for bit.
/// `None` for the same reasons as [`gdn_chunk_loop_step`] and on a device below
/// the tensor-core floor; the caller then asks for the fp32 kernel.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_loop_tc_step(
    gpu: &crate::Gpu,
    bufs: &[&backend_api::DeviceBuffer; 10],
    bh: u32,
    c_len: u32,
    dk: u32,
    dv: u32,
    n_chunks: u32,
    scale: f32,
) -> Option<crate::Step> {
    if !gdn_chunk_prep_shape_ok(dk, dv, c_len) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get("gdn_chunk_loop_tc").filter(|k| k.min_cc <= cc)?;
    let launch = backend_api::CudaLaunch { dynamic_shared_bytes: GDN_CHUNK_LOOP_TC_DYNAMIC_SHARED_BYTES, shared_opt_in: true, ..backend_api::CudaLaunch::NONE };
    let id = gpu.native_kernel_launched(kernel, GDN_CHUNK_LOOP_BINDINGS, launch)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 10], &[bh, c_len, dk, dv, n_chunks, scale.to_bits()], bh * GDN_CHUNK_LOOP_TC_SPLITS)
}

/// [`gdn_chunk_loop_tc_step`] for the training forward: also writes the state and
/// decay-scale histories (the buffers and layout of [`gdn_chunk_loop_train_step`]).
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_loop_train_tc_step(
    gpu: &crate::Gpu,
    bufs: &[&backend_api::DeviceBuffer; 12],
    bh: u32,
    c_len: u32,
    dk: u32,
    dv: u32,
    n_chunks: u32,
    scale: f32,
) -> Option<crate::Step> {
    if !gdn_chunk_prep_shape_ok(dk, dv, c_len) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get("gdn_chunk_loop_train_tc").filter(|k| k.min_cc <= cc)?;
    let launch = backend_api::CudaLaunch { dynamic_shared_bytes: GDN_CHUNK_LOOP_TC_DYNAMIC_SHARED_BYTES, shared_opt_in: true, ..backend_api::CudaLaunch::NONE };
    let id = gpu.native_kernel_launched(kernel, GDN_CHUNK_LOOP_TRAIN_BINDINGS, launch)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 12], &[bh, c_len, dk, dv, n_chunks, scale.to_bits()], bh * GDN_CHUNK_LOOP_TC_SPLITS)
}

/// Whether the token-major tensor-core prefix can serve this call: the chunk
/// forward's own constraints plus a value-head count that is a multiple of the
/// key-head count.
fn gdn_chunk_prep_tok_shape_ok(dk: u32, dv: u32, c_len: u32, heads: u32, group: u32) -> bool {
    gdn_chunk_prep_shape_ok(dk, dv, c_len) && group != 0 && heads.is_multiple_of(group)
}

/// What the token-major tensor-core prefix is told about the mixer's layout.
#[derive(Clone, Copy)]
pub struct GdnTokenLayout {
    /// Number of (head, chunk) pairs, `batch * heads * n_chunks`.
    pub bhc: u32,
    /// Rows per chunk (at most 64).
    pub c_len: u32,
    /// Chunks per sequence.
    pub n_chunks: u32,
    /// Value heads.
    pub heads: u32,
    /// Value heads per key head.
    pub group: u32,
    /// Head width of query, key and value (128 is the only width served).
    pub head_dim: u32,
    pub scale: f32,
    /// The epsilon of the query/key L2 normalisation.
    pub eps: f32,
}

/// [`gdn_chunk_prep_step`] reading the mixer's TOKEN-MAJOR tensors
/// (`gdn_chunk_prep_tok_tc`), for the forward that saves nothing for a backward:
/// `bufs` is `query, key, value, raw_g, beta` (token-major, as for
/// [`gdn_chunk_prep_tok_train_step`]), `g_cs, exp_g_cs, u, w, intra_scores`, then
/// `query_cm, key_cm` for the recurrence. `l.bhc` etc. as there.
pub fn gdn_chunk_prep_tok_step(gpu: &crate::Gpu, bufs: &[&backend_api::DeviceBuffer; 12], l: GdnTokenLayout) -> Option<crate::Step> {
    if !gdn_chunk_prep_tok_shape_ok(l.head_dim, l.head_dim, l.c_len, l.heads, l.group) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get("gdn_chunk_prep_tok_tc").filter(|k| k.min_cc <= cc)?;
    let launch = backend_api::CudaLaunch { dynamic_shared_bytes: GDN_CHUNK_PREP_DYNAMIC_SHARED_BYTES, shared_opt_in: true, ..backend_api::CudaLaunch::NONE };
    let id = gpu.native_kernel_launched(kernel, GDN_CHUNK_PREP_TOK_BINDINGS, launch)?;
    let params = [l.bhc, l.c_len, l.head_dim, l.head_dim, 0, l.scale.to_bits(), l.heads, l.n_chunks, l.group, l.eps.to_bits()];
    gpu.step_native_sliced(id, bufs, &[(0, 0); 12], &params, l.bhc)
}

/// [`gdn_chunk_prep_train_step`] reading the mixer's TOKEN-MAJOR tensors
/// (`gdn_chunk_prep_tok_train_tc`): the L2 normalisation of query and key, the
/// repeat of their heads from `heads / group` to `heads` and the token-major to
/// chunk-major permutes all happen inside the block's loads. `None` for the
/// reasons of [`gdn_chunk_prep_step`], or when `heads` is not a multiple of `group`.
///
/// `bufs` is, in order, `query` and `key` (`[tokens, heads / group * 128]`,
/// pre-normalisation), `value` (`[tokens, heads * 128]`), `raw_g` and `beta`
/// (`[tokens, heads]`) read; `g_cs, exp_g_cs, u, w, intra_scores, k_beta, v_beta,
/// k_beta_decay, decay_mask, raw_attn0, t_mat, raw_intra` as in
/// [`gdn_chunk_prep_train_step`]; then `query_cm, key_cm, value_cm, beta_cm`, the
/// chunk-major tensors the recurrence and the backward read.
pub fn gdn_chunk_prep_tok_train_step(gpu: &crate::Gpu, bufs: &[&backend_api::DeviceBuffer; 21], l: GdnTokenLayout) -> Option<crate::Step> {
    if !gdn_chunk_prep_tok_shape_ok(l.head_dim, l.head_dim, l.c_len, l.heads, l.group) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get("gdn_chunk_prep_tok_train_tc").filter(|k| k.min_cc <= cc)?;
    let launch = backend_api::CudaLaunch { dynamic_shared_bytes: GDN_CHUNK_PREP_DYNAMIC_SHARED_BYTES, shared_opt_in: true, ..backend_api::CudaLaunch::NONE };
    let id = gpu.native_kernel_launched(kernel, GDN_CHUNK_PREP_TOK_TRAIN_BINDINGS, launch)?;
    let params = [l.bhc, l.c_len, l.head_dim, l.head_dim, 0, l.scale.to_bits(), l.heads, l.n_chunks, l.group, l.eps.to_bits()];
    gpu.step_native_sliced(id, bufs, &[(0, 0); 21], &params, l.bhc)
}

/// Binding order of the tensor-core backward kernels: the uniform, `reads` read-only buffers, then
/// `N - 1 - reads` buffers the kernel writes.
const fn bwd_tc_bindings<const N: usize>(reads: usize) -> [BindKind; N] {
    let mut b = [BindKind::StorageReadWrite; N];
    b[0] = BindKind::Uniform;
    let mut i = 1;
    while i <= reads {
        b[i] = BindKind::StorageRead;
        i += 1;
    }
    b
}

/// `gdn_bwd_sweep_tc`: key, query, w, d_out, intra, g_cs, exp_g_cs, decay_scale_hist, d_final_state
/// (read); d_v_new, d_state_hist, d_initial_state (written).
static GDN_BWD_SWEEP_TC_BINDINGS: [BindKind; 13] = bwd_tc_bindings(9);

/// `gdn_chunk_bwd_post_tc`: query, key, value, beta, w, u, g_cs, exp_g_cs, decay_scale_hist, state_hist,
/// t_mat, raw_intra, raw_attn0, d_out, d_state_hist, d_v_new (read); d_query, d_key, d_value, d_raw_g,
/// d_beta (written).
static GDN_CHUNK_BWD_POST_TC_BINDINGS: [BindKind; 22] = bwd_tc_bindings(16);

/// Dynamic shared memory of `gdn_bwd_sweep_tc`: two 64 x 132 staged tiles, the 128 x 40 state-gradient
/// slice, the 64 x 40 `d_out` slice and the 64 x 40 `d_v_new`, all f32.
const GDN_BWD_SWEEP_TC_DYNAMIC_SHARED_BYTES: u32 = (2 * 64 * 132 + 128 * 40 + 2 * 64 * 40) * 4;

/// Value-column blocks `gdn_bwd_sweep_tc` splits each head across (the state gradient's columns are
/// independent: the transition acts on the key axis).
const GDN_BWD_SWEEP_TC_SPLITS: u32 = 4;

/// Dynamic shared memory of `gdn_chunk_bwd_post_tc`: four 64 x 132 tiles, four 64 x 68 matrices and
/// eight 64-float vectors, all f32.
const GDN_CHUNK_BWD_POST_TC_DYNAMIC_SHARED_BYTES: u32 = (4 * 64 * 132 + 4 * 64 * 68 + 8 * 64 + 16) * 4;

/// The tensor-core reverse sweep of Gated DeltaNet's across-chunk recurrence: the same job as
/// [`gdn_bwd_sweep_step`], restated so that the 48 heads split over 192 blocks. The state gradient's
/// columns are independent, so a block owns 32 of them; the quantity that contracts over the columns
/// (`d_decayed_k`) is not computed here but by [`gdn_chunk_bwd_post_tc_step`] from the history of
/// state gradients this launch writes.
///
/// `bufs` is `key, query, w, d_out, intra, g_cs, exp_g_cs, decay_scale_hist, d_final_state` (read)
/// and `d_v_new` (overwritten, the final value), `d_state_hist` (`[n_chunks, bh, 128, 128]`, the state
/// gradient entering each chunk's reverse step), `d_initial_state` (written). `None` for the reasons
/// of [`gdn_chunk_loop_tc_step`]. `heads_tok` is 0 for chunk-major `d_out`, or the number of value heads
/// when `d_out` is the mixer's token-major `[tokens, heads, 128]` (then `bh = batch * heads_tok`).
#[allow(clippy::too_many_arguments)]
pub fn gdn_bwd_sweep_tc_step(
    gpu: &crate::Gpu,
    bufs: &[&backend_api::DeviceBuffer; 12],
    bh: u32,
    c_len: u32,
    dk: u32,
    dv: u32,
    n_chunks: u32,
    scale: f32,
    heads_tok: u32,
) -> Option<crate::Step> {
    if !gdn_chunk_prep_shape_ok(dk, dv, c_len) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get("gdn_bwd_sweep_tc").filter(|k| k.min_cc <= cc)?;
    let launch = backend_api::CudaLaunch { dynamic_shared_bytes: GDN_BWD_SWEEP_TC_DYNAMIC_SHARED_BYTES, shared_opt_in: true, ..backend_api::CudaLaunch::NONE };
    let id = gpu.native_kernel_launched(kernel, &GDN_BWD_SWEEP_TC_BINDINGS, launch)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 12], &[bh, c_len, dk, dv, n_chunks, scale.to_bits(), heads_tok], bh * GDN_BWD_SWEEP_TC_SPLITS)
}

/// Everything of Gated DeltaNet's chunk backward that is independent across chunks, in one launch per
/// (head, chunk) on tensor cores: the recomputed `v_new`, the intra-chunk and inter-chunk output
/// gradients, `d_decayed_k`, the `w`/`u` product gradients, the closed-form unit-triangular-solve
/// backward, the decay and cumulative-sum gradients, and the row-scale gradients, ending in
/// `d_query`, `d_key`, `d_value`, `d_raw_g` and `d_beta`. It replaces the roughly sixty whole-tensor
/// dispatches of the portable backward that surround the sweep.
///
/// `bufs` is `query, key, value, beta, w, u, g_cs, exp_g_cs, decay_scale_hist, state_hist, t_mat,
/// raw_intra, raw_attn0, d_out, d_state_hist, d_v_new` (read) and `d_query, d_key, d_value, d_raw_g,
/// d_beta` (every element overwritten). With `heads_tok` (the number of value heads) the kernel reads
/// `d_out` and writes `d_value`, `d_raw_g` and `d_beta` in the mixer's token-major layout, `n_chunks`
/// chunks per sequence; with 0 every tensor is chunk-major.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_bwd_post_tc_step(
    gpu: &crate::Gpu,
    bufs: &[&backend_api::DeviceBuffer; 21],
    bhc: u32,
    c_len: u32,
    dk: u32,
    dv: u32,
    bh: u32,
    scale: f32,
    heads_tok: u32,
    n_chunks: u32,
) -> Option<crate::Step> {
    if !gdn_chunk_prep_shape_ok(dk, dv, c_len) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get("gdn_chunk_bwd_post_tc").filter(|k| k.min_cc <= cc)?;
    let launch = backend_api::CudaLaunch { dynamic_shared_bytes: GDN_CHUNK_BWD_POST_TC_DYNAMIC_SHARED_BYTES, shared_opt_in: true, ..backend_api::CudaLaunch::NONE };
    let id = gpu.native_kernel_launched(kernel, &GDN_CHUNK_BWD_POST_TC_BINDINGS, launch)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 21], &[bhc, c_len, dk, dv, bh, scale.to_bits(), heads_tok, n_chunks], bhc)
}

/// `gdn_qk_bwd_tok`: d_query_cm, d_key_cm, query, key (read); d_query, d_key (written).
static GDN_QK_BWD_TOK_BINDINGS: [BindKind; 7] = bwd_tc_bindings(4);

/// Rows (one per token and key head, for query and for key) each block of `gdn_qk_bwd_tok` serves:
/// its eight warps take one row each.
const GDN_QK_BWD_ROWS_PER_BLOCK: u32 = 8;

/// The mixer's layout for [`gdn_qk_bwd_tok_step`].
#[derive(Clone, Copy)]
pub struct GdnQkBwdLayout {
    /// Rows of the token-major tensors, `batch * seq_len`.
    pub tokens: u32,
    /// Key heads.
    pub kv_heads: u32,
    /// Value heads per key head.
    pub group: u32,
    /// Value heads.
    pub heads: u32,
    /// Rows per chunk.
    pub c_len: u32,
    /// Tokens per sequence, a multiple of `c_len`.
    pub seq_len: u32,
    pub batch: u32,
    /// The epsilon of the query/key L2 normalisation the forward applied.
    pub eps: f32,
}

/// The last leg of the mixer backward for query and key as one launch (`gdn_qk_bwd_tok`): the
/// chunk-major, per-value-head `d_query` / `d_key` of the chunk backward are summed over the value
/// heads that share a key head and taken back through the L2 normalisation, landing token-major in
/// the layout of the narrow projections. Replaces two layout permutes, two group sums and two
/// normalisation backwards. `None` for the reasons of [`gdn_chunk_prep_tok_step`] (head width 128).
///
/// `bufs` is `d_query_cm, d_key_cm, query, key` (read; `query` / `key` the raw projections) and
/// `d_query, d_key` (every element overwritten).
pub fn gdn_qk_bwd_tok_step(gpu: &crate::Gpu, bufs: &[&backend_api::DeviceBuffer; 6], l: GdnQkBwdLayout) -> Option<crate::Step> {
    if !gdn_chunk_prep_tok_shape_ok(128, 128, l.c_len, l.heads, l.group) || l.kv_heads * l.group != l.heads || l.seq_len % l.c_len != 0 {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get("gdn_qk_bwd_tok").filter(|k| k.min_cc <= cc)?;
    let id = gpu.native_kernel(kernel, &GDN_QK_BWD_TOK_BINDINGS)?;
    let params = [l.tokens, l.kv_heads, l.group, l.heads, l.c_len, l.seq_len, l.batch, l.eps.to_bits()];
    let blocks = (2 * l.tokens * l.kv_heads).div_ceil(GDN_QK_BWD_ROWS_PER_BLOCK);
    gpu.step_native_sliced(id, bufs, &[(0, 0); 6], &params, blocks)
}

/// The native tensor-core paged flash-prefill step for `head_dim = 256`, or
/// `None` when the caller should dispatch the portable
/// `paged_flash_prefill_hd256` it was going to anyway.
///
/// This is not a provider request: attention's call site
/// (`model::block::gqa_chunk_step`) already chooses its kernel by registered
/// pipeline index, and the native kernel is a drop-in for that index - same
/// buffers, same uniform, same launch geometry - so the seam is "try native
/// first" at that one choice. `None` means any of: the head width is not
/// `256`, the round is no wider than the decode regime (`params[0]` rows), `cuda`
/// is named in `BRAIN_NO_PROVIDER`, the device's compute capability is below
/// the kernel's floor (or not reported at all), or the backend declined to take
/// the kernel. Each is a decline, never a failure.
///
/// `bufs` is the portable kernel's own order - queries, K pool, V pool, block
/// tables, sequence lengths, context - bound whole, and `params` its
/// `[bsz, n_heads, n_kv_heads, head_dim, group, block_size, max_bt]`; `blocks`
/// is its workgroup count.
pub fn paged_flash_prefill_step(gpu: &crate::Gpu, head_dim: u32, bufs: &[&backend_api::DeviceBuffer; 6], params: &[u32], blocks: u32) -> Option<crate::Step> {
    // A round of up to `DECODE_REGIME_MAX_ROWS` rows is a speculative-decoding
    // verify or a ragged tail, not a prefill: there is nothing to win, and the
    // contract that such a round computes what the token-by-token tape computes
    // (to fp32 rounding) is worth more than the few microseconds - fp16 attention
    // would break it.
    let rows = params.first().copied().unwrap_or(0);
    if head_dim != 256 || rows <= select::DECODE_REGIME_MAX_ROWS || !tensor_core_kernels_enabled(gpu) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::find(select::Op::PagedAttentionFused, Dtype::F32, cc)?;
    let id = gpu.native_kernel(kernel, PAGED_FLASH_PREFILL_BINDINGS)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 6], params, blocks)
}

impl OperatorProvider for CudaProvider {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn source(&self, _req: &OpRequest) -> ImplSource {
        // Hand-written CUDA C++ selected against a queried capability. The
        // tier is a property of who wrote the kernel, not of how fast it
        // turned out to be - which is why the speedup is asserted by a test
        // rather than implied by this answer.
        ImplSource::Tuned
    }

    fn requires(&self, _req: &OpRequest) -> select::Requirement {
        // Nothing `select::Requirement` can express. What this provider
        // actually needs - a backend that compiles CUDA C++, a device whose
        // reported threads-per-block and shared-memory limits fit the
        // kernel - is asked of the backend itself at `register_native`, which
        // is the only place those answers exist. Claiming a requirement here
        // that does not gate anything would be worse than claiming none: a
        // decline would then be attributed to the wrong cause in the
        // `ImplChoice` record.
        select::Requirement::default()
    }

    fn accepts(&self, req: &OpRequest, _capture: bool) -> bool {
        // `capture` never disqualifies: this provider pushes an ordinary
        // `Step` onto the caller's tape, replayable exactly like a WGSL one.
        req.op == select::Op::MatMul
            && req.pass == Pass::Forward
            && match Self::shape_of(req) {
                // The tiled fp32 kernel fills a 64x64 output tile per block: at
                // decode row counts that is a few blocks of a 132-SM device and
                // measured 200 us for a 257x2048 router. The decode regime is
                // the skinny-M GEMV's (`select::candidates`' head), so leave it
                // to that.
                Some(Dtype::F32) => req.shape.m > select::DECODE_REGIME_MAX_ROWS && self.kernel(select::Op::MatMul, Dtype::F32).is_some(),
                Some(dt) => self.kernel(select::Op::MatMul, dt).is_some(),
                None => false,
            }
    }

    fn lower(&self, ctx: &mut LowerCtx, req: &OpRequest) -> Result<Lowered, String> {
        let dt = match (req.op, Self::shape_of(req)) {
            (select::Op::MatMul, Some(dt)) => dt,
            _ => {
                return Err(format!(
                    "cuda::CudaProvider::lower: {:?} at dtype {:?} is not implemented (plain f32 MatMul and group-32 int8 MatMul only)",
                    req.op, req.shape.dtype
                ))
            }
        };
        let k = if dt == Dtype::I8 { self.i8_mma_kernel(req.shape.m) } else { self.kernel(select::Op::MatMul, dt) };
        let k = k.ok_or_else(|| {
            format!("cuda::CudaProvider::lower: no native {dt:?} MatMul kernel at compute capability {}.{}", self.cc.0, self.cc.1)
        })?;
        let bindings = if dt == Dtype::I8 { MATMUL_I8_BINDINGS } else { MATMUL_BINDINGS };
        let Some(id) = ctx.gpu.native_kernel(k, bindings) else {
            return Err(format!(
                "cuda::CudaProvider::lower: this device declined the native {dt:?} matmul kernel \
                 (register_native returned None) - falling back to the WGSL reference provider"
            ));
        };

        let bufs: Vec<&backend_api::DeviceBuffer> = req.operands.iter().map(|o| o.buf).collect();
        let offsets: Vec<(u64, u64)> = req.operands.iter().map(|o| o.range).collect();
        // The uniform is the reference kernel's own (`[m, k, n]` for f32,
        // `[m, k/4, n]` for int8), passed through untouched - the hand-written
        // kernel reads the identical layout on purpose, so there is no second
        // place a shape could be spelled differently.
        let blocks = k.blocks_for(req.shape.m, req.shape.n);
        let step = ctx
            .gpu
            .step_native_sliced(id, &bufs, &offsets, req.attrs, blocks)
            .ok_or_else(|| {
                "cuda::CudaProvider::lower: step_native_sliced declined a NativeId this provider \
                 itself registered - the backend has no native slicing path"
                    .to_string()
            })?;
        ctx.steps.push(step);
        // Deliberately not spelled like a WGSL catalogue name: this is a
        // kernel from a different registry, and a bare `[a-z0-9_]+` string
        // here would look to a catalogue-cross-referencing gate like an
        // unregistered WGSL kernel. Same convention `coopmat` uses.
        Ok(Lowered::new(1, vec![k.reported]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Operand;
    use backend_api::DeviceBuffer;

    fn shape(m: u32, n: u32, k: u32, dt: Dtype) -> select::OpShape {
        select::OpShape { m, n, k, dtype: dt }
    }

    fn req<'a>(pass: Pass, operands: &'a [Operand<'a>], shape: select::OpShape) -> OpRequest<'a> {
        OpRequest {
            op: select::Op::MatMul,
            shape,
            pass,
            operands,
            attrs: &[],
            group: 32,
            bind: &|_| panic!("the CUDA provider never calls OpRequest::bind - it has no WGSL name table"),
        }
    }

    fn f32_bundle<'a>(b: &'a DeviceBuffer) -> [Operand<'a>; 3] {
        [
            Operand { role: Role::Act, buf: b, range: (0, 1), dtype: DType::F32 },
            Operand { role: Role::Weight, buf: b, range: (0, 0), dtype: DType::F32 },
            Operand { role: Role::Out, buf: b, range: (0, 1), dtype: DType::F32 },
        ]
    }

    /// A capability below every declared floor resolves to no kernel at all,
    /// and a provider with no kernel accepts nothing - it never falls through
    /// to "the closest thing available". Stated at a capability no device
    /// here has, which is the point: the rule has to be right for hardware
    /// this code cannot be run against.
    #[test]
    fn a_capability_below_every_floor_offers_nothing_and_accepts_nothing() {
        let gpu = crate::testgpu::dev(&[("matmul", kernels::MATMUL)]);
        let x = gpu.storage(1);
        let operands = f32_bundle(&x);
        let below = CudaProvider::new((1, 0));
        assert!(below.kernel(select::Op::MatMul, Dtype::F32).is_none() && below.kernel(select::Op::MatMul, Dtype::I8).is_none());
        assert!(!below.accepts(&req(Pass::Forward, &operands, shape(64, 64, 64, Dtype::F32)), false));
    }

    /// A quantized weight tier binds FIVE operands with two scale planes; a
    /// three-pointer f32 kernel would read a scale plane as weights. The
    /// refusal is structural, on the operand bundle, not merely on the
    /// declared dtype.
    #[test]
    fn a_quantized_or_backward_request_is_declined() {
        let gpu = crate::testgpu::dev(&[("matmul", kernels::MATMUL)]);
        let x = gpu.storage(1);
        // A capability above every declared floor, so the decline below can
        // only come from the request and never from kernel resolution.
        let p = CudaProvider::new((99, 0));
        assert!(p.kernel(select::Op::MatMul, Dtype::F32).is_some() && p.kernel(select::Op::MatMul, Dtype::I8).is_some());

        let operands = f32_bundle(&x);
        assert!(p.accepts(&req(Pass::Forward, &operands, shape(64, 64, 64, Dtype::F32)), false));
        assert!(!p.accepts(&req(Pass::Backward, &operands, shape(64, 64, 64, Dtype::F32)), false));
        // Three f32 operands under an I8 shape: the bundle, not the label, decides.
        assert!(!p.accepts(&req(Pass::Forward, &operands, shape(64, 64, 64, Dtype::I8)), false));
    }

    /// The int8 tensor-core kernel is taken for exactly the layout it
    /// implements: group-32 scales, a K that is a whole number of 64-wide
    /// tiles, a row count above the decode regime, and a uniform that agrees
    /// with the shape. Each deviation is a decline (the portable kernel then
    /// serves it), never a launch that reads the wrong plane.
    #[test]
    fn the_int8_tensor_core_kernel_is_taken_only_for_the_layout_it_implements() {
        let gpu = crate::testgpu::dev(&[("matmul", kernels::MATMUL)]);
        let x = gpu.storage(1);
        let p = CudaProvider::new((99, 0));
        let quantized = [
            Operand { role: Role::Act, buf: &x, range: (0, 1), dtype: DType::I8 },
            Operand { role: Role::Weight, buf: &x, range: (0, 0), dtype: DType::I8 },
            Operand { role: Role::ActScale, buf: &x, range: (0, 1), dtype: DType::F32 },
            Operand { role: Role::WeightScale, buf: &x, range: (0, 0), dtype: DType::F32 },
            Operand { role: Role::Out, buf: &x, range: (0, 1), dtype: DType::F32 },
        ];
        let with = |m: u32, n: u32, k: u32, group: u32, attrs: &[u32]| {
            let sh = shape(m, n, k, Dtype::I8);
            let mut r = req(Pass::Forward, &quantized, sh);
            r.group = group;
            r.attrs = attrs;
            p.accepts(&r, false)
        };
        assert!(with(256, 5120, 5120, 32, &[256, 1280, 5120]), "the real prefill shape is the kernel's own");
        assert!(!with(256, 5120, 5120, 16, &[256, 1280, 5120]), "group-16 (Q6_K) scales are a different layout");
        assert!(!with(8, 5120, 5120, 32, &[8, 1280, 5120]), "decode-regime row counts belong to the GEMV family");
        assert!(!with(256, 5120, 5152, 32, &[256, 1288, 5120]), "K must be a whole number of 64-wide tiles");
        assert!(!with(256, 5120, 5120, 32, &[256, 5120, 5120]), "the uniform carries K/4, not K");
        assert!(!with(256, 5120, 5120, 32, &[]), "a request without its uniform is malformed");
    }

    /// On a backend that cannot compile CUDA C++ (every backend but the CUDA
    /// one), `lower` returns `Err` rather than dispatching something else -
    /// which is what makes `ProviderRegistry::dispatch` record the decline
    /// and fall back to the reference provider. Run against the pooled test
    /// device, whatever backend that is.
    #[test]
    fn a_backend_that_cannot_compile_cuda_declines_at_lower() {
        let gpu = crate::testgpu::dev(&[("matmul", kernels::MATMUL)]);
        if gpu.kind() == "cuda" {
            return; // this box's test device IS the CUDA backend; nothing to decline
        }
        let x = gpu.storage(64 * 64);
        let operands = f32_bundle(&x);
        let p = CudaProvider::new((99, 0));
        let caps = gpu.caps();
        let mut steps = Vec::new();
        let mut ctx = LowerCtx { gpu: &gpu, caps: &caps, steps: &mut steps, capture: false };
        assert!(p.lower(&mut ctx, &req(Pass::Forward, &operands, shape(64, 64, 64, Dtype::F32))).is_err());
        assert!(steps.is_empty(), "a declining provider must push no Step");
    }
}
