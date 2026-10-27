// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `CudaProvider` - the [`super::OperatorProvider`] that answers an operator
//! with a **native CUDA kernel** from `kernels_cuda`'s registry,
//! resolved against the compute capability the driver reported for the device
//! it is about to run on.
//!
//! Swedish Embedded AB implements architecture-specialised GPU operator
//! libraries and the dispatch machinery that keeps them honest. If your team
//! needs expertise in getting a tuned kernel onto real silicon without
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
//!   only above the GEMV's crossover ([`select::i8_gemv_max_rows_at`]: the whole
//!   decode regime on a device with a compute capability, whose GEMV slot picks
//!   its candidate by measurement - below that the GEMV family is the right shape), only when K is a
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

/// The int8-KV form of the kernel above: the same buffers, then the K and V
/// planes' `[pool rows, n_kv_heads]` row scales.
const PAGED_FLASH_PREFILL_KV8_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
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
    fn shape_of(&self, req: &OpRequest) -> Option<Dtype> {
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
                    && req.shape.m > select::i8_gemv_max_rows_at(Some(self.cc))
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
/// contract (`heads_tok` non-zero writes `out` token-major, see the kernel header), three-pass TF32 products instead of fp32 CUDA-core ones, so it
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
    heads_tok: u32,
) -> Option<crate::Step> {
    if !gdn_chunk_prep_shape_ok(dk, dv, c_len) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get("gdn_chunk_loop_tc").filter(|k| k.min_cc <= cc)?;
    let launch = backend_api::CudaLaunch { dynamic_shared_bytes: GDN_CHUNK_LOOP_TC_DYNAMIC_SHARED_BYTES, shared_opt_in: true, ..backend_api::CudaLaunch::NONE };
    let id = gpu.native_kernel_launched(kernel, GDN_CHUNK_LOOP_BINDINGS, launch)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 10], &[bh, c_len, dk, dv, n_chunks, scale.to_bits(), heads_tok], bh * GDN_CHUNK_LOOP_TC_SPLITS)
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
    heads_tok: u32,
) -> Option<crate::Step> {
    if !gdn_chunk_prep_shape_ok(dk, dv, c_len) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get("gdn_chunk_loop_train_tc").filter(|k| k.min_cc <= cc)?;
    let launch = backend_api::CudaLaunch { dynamic_shared_bytes: GDN_CHUNK_LOOP_TC_DYNAMIC_SHARED_BYTES, shared_opt_in: true, ..backend_api::CudaLaunch::NONE };
    let id = gpu.native_kernel_launched(kernel, GDN_CHUNK_LOOP_TRAIN_BINDINGS, launch)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 12], &[bh, c_len, dk, dv, n_chunks, scale.to_bits(), heads_tok], bh * GDN_CHUNK_LOOP_TC_SPLITS)
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
    if !gdn_chunk_prep_tok_shape_ok(128, 128, l.c_len, l.heads, l.group) || l.kv_heads * l.group != l.heads || !l.seq_len.is_multiple_of(l.c_len) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get("gdn_qk_bwd_tok").filter(|k| k.min_cc <= cc)?;
    let id = gpu.native_kernel(kernel, &GDN_QK_BWD_TOK_BINDINGS)?;
    let params = [l.tokens, l.kv_heads, l.group, l.heads, l.c_len, l.seq_len, l.batch, l.eps.to_bits()];
    let blocks = (2 * l.tokens * l.kv_heads).div_ceil(GDN_QK_BWD_ROWS_PER_BLOCK);
    gpu.step_native_sliced(id, bufs, &[(0, 0); 6], &params, blocks)
}

/// `gdn_conv_silu_tok`: x, w, hist (read); query, key, value, pre (written).
static GDN_CONV_SILU_TOK_BINDINGS: [BindKind; 8] = bwd_tc_bindings(3);
/// `gdn_conv_hist_tok`: x (read); hist (read and written).
static GDN_CONV_HIST_TOK_BINDINGS: [BindKind; 3] = bwd_tc_bindings(1);
/// `gdn_conv_silu_bwd_tok`: d_query, d_key, d_value, pre, w (read); d_x, d_hist (written).
static GDN_CONV_SILU_BWD_TOK_BINDINGS: [BindKind; 8] = bwd_tc_bindings(5);

/// Channels each block of the token-major convolution kernels serves, and rows each thread walks.
const GDN_CONV_CHANNELS_PER_BLOCK: u32 = 128;
const GDN_CONV_ROWS_PER_THREAD: u32 = 32;

/// The mixer's convolution layout for [`gdn_conv_silu_tok_step`] and [`gdn_conv_silu_bwd_tok_step`].
#[derive(Clone, Copy)]
pub struct GdnConvLayout {
    /// Rows of the token-major tensors, `batch * seq_len`.
    pub rows: u32,
    /// Channels, `2 * key_dim + value_dim`.
    pub conv_dim: u32,
    /// Tokens per sequence.
    pub seq_len: u32,
    pub key_dim: u32,
    pub value_dim: u32,
    /// Taps of the depthwise convolution (4 is the only size served).
    pub kernel: u32,
}

impl GdnConvLayout {
    fn served(&self, gpu: &crate::Gpu) -> Option<&'static kernels_cuda::CudaKernel> {
        let shape_ok = self.kernel == 4
            && self.rows != 0
            && self.seq_len != 0
            && self.rows.is_multiple_of(self.seq_len)
            && self.conv_dim == 2 * self.key_dim + self.value_dim
            && !super::disabled_providers().iter().any(|d| d == "cuda");
        if !shape_ok {
            return None;
        }
        let cc = gpu.caps().arch.compute_capability?;
        kernels_cuda::get("gdn_conv_silu_tok").filter(|k| k.min_cc <= cc)
    }

    /// Blocks of a launch over `rows` output rows: channel tiles fastest.
    fn blocks(&self, rows: u32) -> u32 {
        self.conv_dim.div_ceil(GDN_CONV_CHANNELS_PER_BLOCK) * rows.div_ceil(GDN_CONV_ROWS_PER_THREAD)
    }
}

/// The causal depthwise convolution of the mixer, its SiLU and the query / key / value split as one launch
/// (`gdn_conv_silu_tok`), on the token-major rows of the projection: replaces two layout transposes, the
/// generic convolution, the activation and three row splits. `bufs` is `x` (`[rows, conv_dim]`), `w`
/// (`[conv_dim, 4]`), `hist` (`[conv_dim, 3]` channel-major, oldest first; read only when `has_hist`, then
/// the history precedes row 0 and the call is a single sequence; any buffer otherwise), then `query`, `key`
/// (`[rows, key_dim]`), `value` (`[rows, value_dim]`) and `pre` (`[rows, conv_dim]`, the pre-activation the
/// backward reads; written only when `save_pre`, any buffer otherwise). `None` when the kernel size is not 4,
/// the shape is inconsistent, `cuda` is named in `BRAIN_NO_PROVIDER`, or the device declines.
pub fn gdn_conv_silu_tok_step(gpu: &crate::Gpu, bufs: &[&backend_api::DeviceBuffer; 7], l: GdnConvLayout, has_hist: bool, save_pre: bool) -> Option<crate::Step> {
    if has_hist && l.rows != l.seq_len {
        return None;
    }
    let kernel = l.served(gpu)?;
    let id = gpu.native_kernel(kernel, &GDN_CONV_SILU_TOK_BINDINGS)?;
    let params = [l.rows, l.conv_dim, l.seq_len, l.key_dim, l.value_dim, has_hist as u32, save_pre as u32];
    gpu.step_native_sliced(id, bufs, &[(0, 0); 7], &params, l.blocks(l.rows))
}

/// `hist <- the last 3 rows of [hist | x]` (`gdn_conv_hist_tok`), the convolution history the next round of a
/// chunked forward starts from. Launch it AFTER [`gdn_conv_silu_tok_step`] has read the old history.
/// `bufs` is `x` (`[rows, conv_dim]`) and `hist` (`[conv_dim, 3]`, channel-major, oldest first).
pub fn gdn_conv_hist_tok_step(gpu: &crate::Gpu, bufs: &[&backend_api::DeviceBuffer; 2], l: GdnConvLayout) -> Option<crate::Step> {
    l.served(gpu)?;
    let kernel = kernels_cuda::get("gdn_conv_hist_tok")?;
    let id = gpu.native_kernel(kernel, &GDN_CONV_HIST_TOK_BINDINGS)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 2], &[l.rows, l.conv_dim], l.conv_dim.div_ceil(GDN_CONV_CHANNELS_PER_BLOCK))
}

/// The backward of [`gdn_conv_silu_tok_step`] with a frozen convolution weight, as one launch
/// (`gdn_conv_silu_bwd_tok`): `bufs` is `d_query`, `d_key` (`[rows, key_dim]`), `d_value` (`[rows, value_dim]`),
/// `pre` (`[rows, conv_dim]`), `w` (`[conv_dim, 4]`) read and `d_x` (`[rows, conv_dim]`), `d_hist`
/// (`[ctx, conv_dim]` token-major, the gradient of the history rows; any buffer when `ctx == 0`) written.
/// `ctx` is 0 for a fresh sequence and 3 for a continued round.
pub fn gdn_conv_silu_bwd_tok_step(gpu: &crate::Gpu, bufs: &[&backend_api::DeviceBuffer; 7], l: GdnConvLayout, ctx: u32) -> Option<crate::Step> {
    if ctx != 0 && (ctx != 3 || l.rows != l.seq_len) {
        return None;
    }
    l.served(gpu)?;
    let kernel = kernels_cuda::get("gdn_conv_silu_bwd_tok")?;
    let id = gpu.native_kernel(kernel, &GDN_CONV_SILU_BWD_TOK_BINDINGS)?;
    let params = [l.rows, l.conv_dim, l.seq_len, l.key_dim, l.value_dim, ctx];
    gpu.step_native_sliced(id, bufs, &[(0, 0); 7], &params, l.blocks(l.rows + ctx))
}

/// `swiglu_bwd_fused`: gate pre-activation, up, incoming gradient (read); gate gradient, up gradient (written).
static SWIGLU_BWD_FUSED_BINDINGS: [BindKind; 6] = bwd_tc_bindings(3);

/// Elements one block of `swiglu_bwd_fused` covers when `total` is a multiple of four (256 threads, four each).
const SWIGLU_BWD_BLOCK_ELEMS: u32 = 1024;

/// The SwiGLU backward as one launch (`swiglu_bwd_fused`) in place of `silu_bwd_da` and `silu_bwd_db`:
/// `bufs` is `gate_pre, up, d_h` (read) and `d_gate, d_up` (every element overwritten), all `total` floats.
/// `None` when `cuda` is named in `BRAIN_NO_PROVIDER`, the device reports no capability, or the backend
/// declines; the caller then issues the two portable dispatches.
pub fn swiglu_bwd_fused_step(gpu: &crate::Gpu, bufs: &[&backend_api::DeviceBuffer; 5], total: u32) -> Option<crate::Step> {
    if total == 0 || super::disabled_providers().iter().any(|d| d == "cuda") {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get("swiglu_bwd_fused").filter(|k| k.min_cc <= cc)?;
    let id = gpu.native_kernel(kernel, &SWIGLU_BWD_FUSED_BINDINGS)?;
    // Four elements a thread when the total allows it, else one.
    let per_block = if total.is_multiple_of(4) { SWIGLU_BWD_BLOCK_ELEMS } else { SWIGLU_BWD_BLOCK_ELEMS / 4 };
    gpu.step_native_sliced(id, bufs, &[(0, 0); 5], &[total], total.div_ceil(per_block))
}

/// `add_rmsnorm_fwd`: a, b, w (read); sum, out (written).
static ADD_RMSNORM_BINDINGS: [BindKind; 6] = bwd_tc_bindings(3);
/// `rmsnorm_dx_add`: x, w, dy, res (read); dx (written).
static RMSNORM_DX_ADD_BINDINGS: [BindKind; 6] = bwd_tc_bindings(4);

/// Shared checks of the fused RMSNorm kernels: rows of a multiple of four, no `cuda` in `BRAIN_NO_PROVIDER`.
fn rmsnorm_fused_kernel(gpu: &crate::Gpu, name: &str, d: u32, rows: u32) -> Option<&'static kernels_cuda::CudaKernel> {
    if d == 0 || rows == 0 || !d.is_multiple_of(4) || super::disabled_providers().iter().any(|x| x == "cuda") {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    kernels_cuda::get(name).filter(|k| k.min_cc <= cc)
}

/// `sum = a + b; out = rmsnorm(sum) * w` as one launch (`add_rmsnorm_fwd`), for `rows` rows of width `d`.
/// `bufs` is `a, b, w` (read) and `sum, out` (every element overwritten). `None` when the width is not a
/// multiple of four, `cuda` is disabled, or the device declines; the caller then issues `add2` and the norm.
pub fn add_rmsnorm_fwd_step(gpu: &crate::Gpu, bufs: &[&backend_api::DeviceBuffer; 5], d: u32, rows: u32, eps: f32) -> Option<crate::Step> {
    let kernel = rmsnorm_fused_kernel(gpu, "add_rmsnorm_fwd", d, rows)?;
    let id = gpu.native_kernel(kernel, &ADD_RMSNORM_BINDINGS)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 5], &[d, rows, eps.to_bits()], rows)
}

/// `dx = res + rmsnorm_dx(x, w, dy)` as one launch (`rmsnorm_dx_add`). `bufs` is `x, w, dy, res` (read) and
/// `dx` (overwritten). `None` for the reasons of [`add_rmsnorm_fwd_step`].
pub fn rmsnorm_dx_add_step(gpu: &crate::Gpu, bufs: &[&backend_api::DeviceBuffer; 5], d: u32, rows: u32, eps: f32) -> Option<crate::Step> {
    let kernel = rmsnorm_fused_kernel(gpu, "rmsnorm_dx_add", d, rows)?;
    let id = gpu.native_kernel(kernel, &RMSNORM_DX_ADD_BINDINGS)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 5], &[d, rows, eps.to_bits()], rows)
}

/// The native tensor-core paged flash-prefill step for `head_dim = 256` or `128`, or
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
    if !flash_prefill_offered(gpu, head_dim, params) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = match head_dim {
        128 => kernels_cuda::get("flash_prefill_f16_hd128").filter(|k| k.min_cc <= cc)?,
        _ => kernels_cuda::find(select::Op::PagedAttentionFused, Dtype::F32, cc)?,
    };
    let id = gpu.native_kernel(kernel, PAGED_FLASH_PREFILL_BINDINGS)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 6], params, blocks)
}

/// Dynamic shared memory of `flash_prefill_f16_hd256_v2_*`: the Q tile (64 rows) and the K and V tiles (`V2_KEYS` = 64
/// keys each, in the kernel source), fp16 rows of 256 + 8 values. Two blocks of it fit one SM (2 x 99 KiB of the 227 KiB).
const FLASH_PREFILL_V2_DYNAMIC_SHARED_BYTES: u32 = (64 + 2 * 64) * (256 + 8) * 2;
const FLASH_PREFILL_V2_LAUNCH: backend_api::CudaLaunch = backend_api::CudaLaunch { dynamic_shared_bytes: FLASH_PREFILL_V2_DYNAMIC_SHARED_BYTES, shared_opt_in: true, ..backend_api::CudaLaunch::NONE };

static FLASH_PREFILL_V2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Choose the head_dim 256 prefill attention over a bf16 or int8 KV cache: version 2 (the default) or the original
/// kernel, which stays as the fallback and as the baseline of an A/B. Process-wide; takes effect for steps built after.
pub fn set_flash_prefill_v2(on: bool) {
    FLASH_PREFILL_V2.store(on, std::sync::atomic::Ordering::Relaxed);
}

static FLASH_PREFILL_PORTABLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Withhold the native fp16 flash prefill kernels (every head width and cache tier) so the portable fp32 kernel runs:
/// the reference an accuracy check scores them against. Process-wide; takes effect for steps built after.
pub fn set_flash_prefill_portable(on: bool) {
    FLASH_PREFILL_PORTABLE.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn flash_prefill_v2() -> bool {
    FLASH_PREFILL_V2.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether a prefill round of `rows` rows at `head_dim` may take a native fp16
/// tensor-core attention kernel on this device. A round in the decode regime is
/// a speculative verify or a ragged tail and keeps the fp32 kernel, whose
/// arithmetic the token-by-token tape is held to.
fn flash_prefill_offered(gpu: &crate::Gpu, head_dim: u32, params: &[u32]) -> bool {
    let rows = params.first().copied().unwrap_or(0);
    matches!(head_dim, 128 | 256) && rows > select::DECODE_REGIME_MAX_ROWS && tensor_core_kernels_enabled(gpu) && !FLASH_PREFILL_PORTABLE.load(std::sync::atomic::Ordering::Relaxed)
}

/// [`paged_flash_prefill_step`] over a bf16 KV cache: `bufs` and `params` are
/// the portable bf16-pool kernel's own, the planes are read in place.
pub fn paged_flash_prefill_bf16_step(gpu: &crate::Gpu, head_dim: u32, bufs: &[&backend_api::DeviceBuffer; 6], params: &[u32], blocks: u32) -> Option<crate::Step> {
    if !flash_prefill_offered(gpu, head_dim, params) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    if head_dim == 256 && flash_prefill_v2() {
        if let Some(kernel) = kernels_cuda::get("flash_prefill_f16_hd256_v2_bf16").filter(|k| k.min_cc <= cc) {
            let id = gpu.native_kernel_launched(kernel, PAGED_FLASH_PREFILL_BINDINGS, FLASH_PREFILL_V2_LAUNCH)?;
            return gpu.step_native_sliced(id, bufs, &[(0, 0); 6], params, blocks);
        }
    }
    let kernel = kernels_cuda::get(if head_dim == 128 { "flash_prefill_f16_hd128_bf16" } else { "flash_prefill_f16_hd256_bf16" }).filter(|k| k.min_cc <= cc)?;
    let id = gpu.native_kernel(kernel, PAGED_FLASH_PREFILL_BINDINGS)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 6], params, blocks)
}

/// The 8-bit element codecs the scaled paged prefill kernels read: the same packed rows and row
/// scales, a different decode of each byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaled8 {
    Int8,
    Fp8E4m3,
}

impl Scaled8 {
    /// The kernel-name suffix of the codec (`flash_prefill_f16_hd256_<suffix>`).
    fn suffix(self) -> &'static str {
        match self {
            Scaled8::Int8 => "kv8",
            Scaled8::Fp8E4m3 => "kvfp8",
        }
    }
}

/// [`paged_flash_prefill_step`] over an int8 KV cache: `bufs` are the portable
/// kv8 kernel's own - queries, K data, V data, block tables, sequence lengths,
/// context, then the K and V row scales.
pub fn paged_flash_prefill_kv8_step(gpu: &crate::Gpu, head_dim: u32, bufs: &[&backend_api::DeviceBuffer; 8], params: &[u32], blocks: u32) -> Option<crate::Step> {
    paged_flash_prefill_scaled8_step(gpu, head_dim, bufs, params, blocks, Scaled8::Int8)
}

/// [`paged_flash_prefill_kv8_step`] over an FP8 E4M3 KV cache (same buffers, same launch geometry): the
/// tensor-core prefill reads the packed bytes in place and decodes them to fp16 as the tiles are staged.
pub fn paged_flash_prefill_kvfp8_step(gpu: &crate::Gpu, head_dim: u32, bufs: &[&backend_api::DeviceBuffer; 8], params: &[u32], blocks: u32) -> Option<crate::Step> {
    paged_flash_prefill_scaled8_step(gpu, head_dim, bufs, params, blocks, Scaled8::Fp8E4m3)
}

fn paged_flash_prefill_scaled8_step(gpu: &crate::Gpu, head_dim: u32, bufs: &[&backend_api::DeviceBuffer; 8], params: &[u32], blocks: u32, codec: Scaled8) -> Option<crate::Step> {
    if !flash_prefill_offered(gpu, head_dim, params) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let suffix = codec.suffix();
    if head_dim == 256 && flash_prefill_v2() {
        if let Some(kernel) = kernels_cuda::get(&format!("flash_prefill_f16_hd256_v2_{suffix}")).filter(|k| k.min_cc <= cc) {
            let id = gpu.native_kernel_launched(kernel, PAGED_FLASH_PREFILL_KV8_BINDINGS, FLASH_PREFILL_V2_LAUNCH)?;
            return gpu.step_native_sliced(id, bufs, &[(0, 0); 8], params, blocks);
        }
    }
    let kernel = kernels_cuda::get(&format!("flash_prefill_f16_hd{head_dim}_{suffix}")).filter(|k| k.min_cc <= cc)?;
    let id = gpu.native_kernel(kernel, PAGED_FLASH_PREFILL_KV8_BINDINGS)?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 8], params, blocks)
}

/// `flash_attn_causal_gqa`'s own `@binding` order: uniform, queries, keys, values, context, log-sum-exp.
const FLASH_GQA_FWD_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// The native tensor-core forward of causal grouped-query flash attention over dense `[B*T]` buffers (the
/// training forward), or `None` when the caller should dispatch the portable `flash_attn_causal_gqa`.
///
/// Same buffers (`q`, `k`, `v`, `ctx`, `lse`) and uniform (`[b, n_heads, n_kv_heads, t, head_dim, group]`) as
/// the portable kernel; the launch geometry is its own (64 query rows per block of one (sample, head)).
/// `None` means: the head width is not 128 or 256, `cuda` is named in `BRAIN_NO_PROVIDER`, the device's compute
/// capability is below the kernel's floor, or the backend declined the kernel.
pub fn flash_causal_gqa_fwd_step(gpu: &crate::Gpu, bufs: &[&backend_api::DeviceBuffer; 5], params: &[u32]) -> Option<crate::Step> {
    let &[b, n_heads, _, t, head_dim, _] = params else { return None };
    if !matches!(head_dim, 128 | 256) || !tensor_core_kernels_enabled(gpu) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get(if head_dim == 128 { "flash_attn_fwd_f16_hd128" } else { "flash_attn_fwd_f16_hd256" }).filter(|k| k.min_cc <= cc)?;
    let id = gpu.native_kernel(kernel, FLASH_GQA_FWD_BINDINGS)?;
    let blocks = u32::try_from(u64::from(b) * u64::from(n_heads) * u64::from(t.div_ceil(64))).ok()?;
    gpu.step_native_sliced(id, bufs, &[(0, 0); 5], params, blocks)
}

/// Blocks of `brain_flash_bwd_absmax` (its `ABSMAX_BLOCKS`): the partial maxima `brain_flash_bwd_scale` reduces.
const FLASH_BWD_ABSMAX_BLOCKS: u32 = 256;

/// Both scale passes bind (uniform, read, read-write): dO then the partials, and the partials then the scale.
const FLASH_BWD_SCALE_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageReadWrite];

/// `flash_attn_causal_gqa_bwd_dq`'s order (uniform, q, k, v, o, dO, lse, dq, dsum) plus the call's scale.
const FLASH_BWD_DQ_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
];

/// `flash_attn_causal_gqa_bwd_dkv`'s order (uniform, q, k, v, dO, lse, dsum, dk, dv) plus the call's scale.
const FLASH_BWD_DKV_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
];

/// The buffers of one flash attention backward: what the portable `flash_attn_causal_gqa_bwd_{dq,dkv}` pair reads
/// and writes.
pub struct FlashBwdBufs<'a> {
    /// Queries `[B*T, n_heads*hd]`.
    pub q: &'a backend_api::DeviceBuffer,
    /// Keys `[B*T, n_kv_heads*hd]`.
    pub k: &'a backend_api::DeviceBuffer,
    /// Values, as `k`.
    pub v: &'a backend_api::DeviceBuffer,
    /// The forward's context output.
    pub o: &'a backend_api::DeviceBuffer,
    /// The context gradient.
    pub d_o: &'a backend_api::DeviceBuffer,
    /// The forward's `[B, n_heads, T]` log-sum-exp.
    pub lse: &'a backend_api::DeviceBuffer,
    /// `[B, n_heads, T]` scratch the dQ kernel fills with `D = dO . O` for the dK/dV kernel.
    pub dsum: &'a backend_api::DeviceBuffer,
    /// Output: the query gradient.
    pub d_q: &'a backend_api::DeviceBuffer,
    /// Output: the key gradient.
    pub d_k: &'a backend_api::DeviceBuffer,
    /// Output: the value gradient.
    pub d_v: &'a backend_api::DeviceBuffer,
}

/// The native tensor-core backward of causal grouped-query flash attention over dense `[B*T]` buffers at
/// `head_dim` 128, as three steps (the call's gradient scale, dQ, dK and dV), or `None` when the caller should
/// dispatch the portable pair. `params` is the portable kernels' uniform
/// (`[b, n_heads, n_kv_heads, t, head_dim, group]`); every output is overwritten. `None` means: the head width is
/// not 128, `cuda` is named in `BRAIN_NO_PROVIDER`, the device's compute capability is below a kernel's floor, or
/// the backend declined one.
pub fn flash_causal_gqa_bwd_steps(gpu: &crate::Gpu, bufs: &FlashBwdBufs<'_>, params: &[u32]) -> Option<Vec<crate::Step>> {
    let &[b, n_heads, n_kv, t, head_dim, _] = params else { return None };
    if head_dim != 128 || !tensor_core_kernels_enabled(gpu) {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let get = |name: &str| kernels_cuda::get(name).filter(|k| k.min_cc <= cc);
    let (absmax_k, scale_k, dq_k, dkv_k) = (get("flash_bwd_absmax")?, get("flash_bwd_scale")?, get("flash_bwd_dq")?, get("flash_bwd_dkv")?);
    let (absmax_id, scale_id, dq_id, dkv_id) = (gpu.native_kernel(absmax_k, FLASH_BWD_SCALE_BINDINGS)?, gpu.native_kernel(scale_k, FLASH_BWD_SCALE_BINDINGS)?, gpu.native_kernel(dq_k, FLASH_BWD_DQ_BINDINGS)?, gpu.native_kernel(dkv_k, FLASH_BWD_DKV_BINDINGS)?);
    let tiles = u64::from(t.div_ceil(64));
    let q_blocks = u32::try_from(u64::from(b) * u64::from(n_heads) * tiles).ok()?;
    let kv_blocks = u32::try_from(u64::from(b) * u64::from(n_kv) * tiles).ok()?;
    let scale = gpu.storage(2);
    let partials = gpu.storage(FLASH_BWD_ABSMAX_BLOCKS.into());
    Some(vec![
        gpu.step_native_sliced(absmax_id, &[bufs.d_o, &partials], &[(0, 0); 2], params, FLASH_BWD_ABSMAX_BLOCKS)?,
        gpu.step_native_sliced(scale_id, &[&partials, &scale], &[(0, 0); 2], params, 1)?,
        gpu.step_native_sliced(dq_id, &[bufs.q, bufs.k, bufs.v, bufs.o, bufs.d_o, bufs.lse, bufs.d_q, bufs.dsum, &scale], &[(0, 0); 9], params, q_blocks)?,
        gpu.step_native_sliced(dkv_id, &[bufs.q, bufs.k, bufs.v, bufs.d_o, bufs.lse, bufs.dsum, bufs.d_k, bufs.d_v, &scale], &[(0, 0); 9], params, kv_blocks)?,
    ])
}

/// `paged_decode_scores_i8_batched`-family order for the batched decode attention over an int8 pool: uniform,
/// queries, K data, V data, K scales, V scales, block tables, sequence lengths, context.
const DECODE_ATTN_I8_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// The native batched paged-KV decode attention (one launch for the whole triad) for `head_dim = 128`, or `None`
/// when the caller should dispatch the portable triad (or the split-key kernel for one sequence).
///
/// `bufs` is queries, K pool, V pool, [K scales, V scales for an int8 pool,] block tables, sequence lengths,
/// context; `params` the triad's `[bsz, n_heads, n_kv_heads, head_dim, group, block_size, max_bt]`. `None` means:
/// the head width is not 128, the group is wider than 8, fewer than two sequences (the split-key kernel owns one),
/// `cuda` is named in `BRAIN_NO_PROVIDER`, the device is below the kernel's floor, or the backend declined it.
pub fn paged_decode_batched_step(gpu: &crate::Gpu, int8: bool, bufs: &[&backend_api::DeviceBuffer], params: &[u32]) -> Option<crate::Step> {
    let &[bsz, _, n_kv, head_dim, group, _, _] = params else { return None };
    if head_dim != 128 || !(1..=8).contains(&group) || bsz < 2 || super::disabled_providers().iter().any(|d| d == "cuda") || bufs.len() != if int8 { 8 } else { 6 } {
        return None;
    }
    let cc = gpu.caps().arch.compute_capability?;
    let kernel = kernels_cuda::get(if int8 { "decode_attn_batched_i8" } else { "decode_attn_batched_f32" }).filter(|k| k.min_cc <= cc)?;
    let id = gpu.native_kernel(kernel, if int8 { DECODE_ATTN_I8_BINDINGS } else { PAGED_FLASH_PREFILL_BINDINGS })?;
    let blocks = bsz.checked_mul(n_kv)?;
    gpu.step_native_sliced(id, bufs, &vec![(0, 0); bufs.len()], params, blocks)
}

impl OperatorProvider for CudaProvider {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn source(&self, _req: &OpRequest) -> ImplSource {
        // Native CUDA C++ selected against a queried capability. The
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
            && match self.shape_of(req) {
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
        let dt = match (req.op, self.shape_of(req)) {
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
        // `[m, k/4, n]` for int8), passed through untouched - the native
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
