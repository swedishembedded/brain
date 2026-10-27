// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Native CUDA C++ kernels and the metadata that says what each one
//! is - a registry of brain's NATIVE kernels, entirely separate from the
//! portable WGSL catalogue in `kernels`.
//!
//! Swedish Embedded AB implements native GPU kernel libraries and the
//! metadata discipline that keeps them honest. If your team needs expertise
//! in native CUDA kernels that stay checkable rather than becoming
//! folklore, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Why this is a separate crate, not a `cuda/` subdirectory of `kernels`
//!
//! The WGSL catalogue is not merely a list of strings - four separate pieces
//! of machinery treat every one of its entries as WGSL text:
//!
//! - the generator that rebuilds `kernels`' const block derives each const
//!   name from the file stem, so a `matmul.cu` next to `matmul.wgsl` is a
//!   name collision, not a new kernel;
//! - the catalogue-validation test compiles EVERY registry entry on the test
//!   device, and CUDA C++ is not a shader the device's WGSL front end can
//!   accept;
//! - the cost-formula ratchet requires a FLOP formula for every registry
//!   entry, keyed by the WGSL kernel's own name and parameter layout;
//! - all five metadata cross-checks (barrier count, declared workgroup size,
//!   packed-int8 dot usage, register-blocking claim, storage dtype) parse
//!   WGSL text and have zero purchase on C++.
//!
//! Putting `.cu` files into that registry would mean excluding them from
//! every one of those - which is a second registry with extra steps. So this
//! crate is that second registry, openly: its own table, its own invariants,
//! its own catalogue gate. Duplication is avoided by ONE discovery path (the
//! two catalogues link to each other), not by one directory.
//!
//! # What a device can do is never written down here
//!
//! [`CudaKernel::min_cc`] is a floor a kernel DECLARES about itself, checked
//! against the instructions its own text uses. It is never a statement about
//! any installed card: which kernel a given device gets is decided by
//! [`best_for`] against a compute capability the caller queried at run time.

use backend_api::select::{Dtype, Op};
use backend_api::ImplSource;

/// A compute capability as `(major, minor)`, exactly as
/// `cuDeviceGetAttribute` reports the two halves. Ordered lexicographically
/// by tuple comparison, which is the ordering NVIDIA's own numbering has.
pub type Cc = (u32, u32);

/// The compute capability that introduced the four-way packed integer dot
/// product (`__dp4a`). This is a property of the CUDA instruction set - the
/// version the instruction first appeared in - not of any device: it exists
/// so a kernel whose body uses the instruction cannot declare a floor below
/// the point at which the instruction exists at all. Devices are asked what
/// they are; source text is held to what it uses.
pub const DP4A_MIN_CC: Cc = (6, 1);

/// The compute capability that introduced the warp-level `mma.sync` int8
/// tensor-core instructions (`m16n8k32`) and `cp.async`. Like [`DP4A_MIN_CC`]
/// this is a property of the instruction set, not of a device: it exists so
/// a kernel whose body issues them cannot declare a lower floor.
pub const MMA_S8_MIN_CC: Cc = (8, 0);

/// The compute capability that introduced the FP8 conversions (`cvt.rn.f16x2.e4m3x2`): Ada.
pub const FP8_CVT_MIN_CC: Cc = (8, 9);

/// The compute capability that introduced thread-block clusters and their distributed shared memory: Hopper.
pub const CLUSTER_MIN_CC: Cc = (9, 0);

/// Rows of x from which `matmul_i8_gemv` runs its int8 tensor-core block (see
/// [`backend_api::select::I8_GEMV_MMA_MIN_ROWS`]); the source's `BRAIN_I8G_MMA_MIN_ROWS` is the
/// same number and a test holds the two together.
pub use backend_api::select::I8_GEMV_MMA_MIN_ROWS;

/// The lowest compute capability a CUDA 12.x toolchain will emit code for at
/// all (`--gpu-architecture=sm_50`). A kernel whose text uses nothing beyond
/// the always-available core - shared memory, `__syncthreads`, fp32
/// arithmetic, 64-bit address arithmetic - declares this floor, which says
/// "there is no capability this toolchain can target where this source is
/// invalid", not "this kernel was written for Maxwell".
///
/// Like [`DP4A_MIN_CC`] this is a property of the TOOLCHAIN and the
/// instruction set, never of a card: the floor a kernel declares is checked
/// against the capability a device was *asked* for, and a device below every
/// floor simply gets no native kernel.
pub const BASELINE_MIN_CC: Cc = (5, 0);

/// Shared memory per block every CUDA compute capability guarantees. A
/// kernel's declared `__shared__` must fit inside this, so that "does this
/// device have room" is a question only about cards that grant MORE - which
/// is asked of the driver, per device, never written down. A card granting
/// more is an opportunity a future kernel may query for; it is never a floor
/// this file may assume.
pub const PORTABLE_SHARED_BYTES: u32 = 48 * 1024;

/// One native CUDA kernel: its source text plus the metadata that
/// decides when it is eligible and what claim it makes.
#[derive(Clone, Copy, Debug)]
pub struct CudaKernel {
    /// Registry key, unique across this table. Never required to match a
    /// WGSL kernel name - the two registries are independent namespaces.
    pub name: &'static str,
    /// The whole operator this kernel implements, in the same vocabulary the
    /// kernel selector and the tier policy use.
    pub op: Op,
    /// The weight storage tier the kernel reads: `Dtype::F32` for a plain
    /// fp32 matmul, `Dtype::I8` for the packed-int8 GEMV. Part of the
    /// identity of an implementation, because two kernels for the same
    /// operator bind different operand bundles and must never be resolved
    /// for one another.
    pub weight: Dtype,
    /// Whether this kernel is asked for BY NAME (`gpu_core::Fused`) rather than
    /// resolved by operator and weight tier. A fused kernel replaces a chain of
    /// dispatches the MODEL builds, so there is no operator a selector could
    /// pick it for; [`best_for`] and [`best_for_any_tier`] never offer one, which
    /// is what keeps the entry for an operator the kernel that implements it
    /// whole - a fused kernel sharing an (operator, tier) pair with it must not
    /// win on table order.
    pub by_name: bool,
    /// The tier this kernel claims. Only [`ImplSource::Tuned`] is meaningful
    /// here: a kernel with a `.cu` file in this tree is native by
    /// definition, and a generated kernel has no file to list (it is emitted
    /// from the WGSL reference at run time). [`check_table`] rejects
    /// anything else rather than letting a generated kernel be filed as if
    /// somebody wrote it.
    pub source: ImplSource,
    /// The lowest compute capability this kernel's TEXT is valid on - the
    /// instructions it uses, not the cards it was tried on. [`best_for`]
    /// picks the highest floor at or below the device's queried capability.
    pub min_cc: Cc,
    /// The `extern "C" __global__` entry point to launch, which must appear
    /// in [`Self::src`].
    pub entry: &'static str,
    /// One line, author-stated, for the generated catalogue.
    pub what: &'static str,
    /// This kernel's name as a DISPATCH RECORD reports it: [`Self::name`]
    /// under a `native:` qualifier.
    ///
    /// Written out rather than formatted at use because a dispatch record
    /// holds `&'static str` and formatting one per lowered request would
    /// have to leak it. [`check_table`] pins the two spellings together, so
    /// the duplication cannot drift.
    ///
    /// The qualifier is not decoration: a bare lowercase name in a dispatch
    /// record is indistinguishable from a WGSL catalogue kernel, and this
    /// registry is a different namespace whose names are under no obligation
    /// to be absent from that one.
    pub reported: &'static str,
    /// Threads per block the kernel's own index arithmetic is written
    /// against - CUDA C++ has no `@workgroup_size` attribute for a backend to
    /// read, so a launcher would otherwise have to guess.
    pub block_dim: u32,
    /// Output elements one block covers, as `(rows, cols)` of the `(m, n)`
    /// output. The launcher turns a shape into a block count with it
    /// (`ceil(m/rows) * ceil(n/cols)`); the kernel reconstructs its own tile
    /// from the flat block index the same way.
    pub tile: (u32, u32),
    /// Static `__shared__` bytes the kernel declares. Checked against the
    /// device's QUERIED shared-memory-per-block limit before dispatch, so a
    /// kernel a card cannot host is declined rather than failing at launch.
    pub shared_bytes: u32,
    /// Whether the kernel is written for programmatic dependent launch: it waits for its predecessor
    /// (`griddepcontrol.wait`) before its first read and lets its own dependents launch at once
    /// (`griddepcontrol.launch_dependents`), so the device overlaps its launch with the kernel before it.
    pub programmatic: bool,
    /// The CUDA C++ source, `include_str!`ed from this crate's `cu/`
    /// directory.
    pub src: &'static str,
}

impl CudaKernel {
    /// How many blocks cover an `(m, n)` output with this kernel's tile.
    pub fn blocks_for(&self, m: u32, n: u32) -> u32 {
        m.div_ceil(self.tile.0.max(1)) * n.div_ceil(self.tile.1.max(1))
    }
}

/// Every native CUDA kernel brain ships.
///
/// Each entry says only what is true of it. A floor is what the kernel's own
/// text needs (the toolchain baseline for plain fp32, the packed dot product
/// for the int8 GEMV), never any card's capability; the capability a device
/// reports is asked of the driver and met against this table by [`best_for`],
/// which is where the architecture-specific decision lives. Entries for the
/// same operator are told apart by the weight tier they read.
pub const ALL: &[CudaKernel] = &[
    CudaKernel {
        name: "matmul_f32_tiled",
        op: Op::MatMul,
        weight: Dtype::F32,
        by_name: false,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_matmul_f32_tiled",
        what: "fp32 out = x @ W^T; 64x64 shared tile, 4x4 register block, reference reduction order",
        reported: "native:matmul_f32_tiled",
        block_dim: 256,
        tile: (64, 64),
        // 2 tiles x 16 staged k x (64 + 1 pad) floats. Stated here because the
        // provider checks it against the device's own queried limit before it
        // ever asks the driver to launch.
        shared_bytes: 2 * 16 * (64 + 1) * 4,
        programmatic: false,
        src: include_str!("../cu/matmul_f32_tiled.cu"),
    },
    CudaKernel {
        name: "matmul_i8_mma",
        // Asked for by name by `CudaProvider` (above the decode regime), never
        // resolved by `find`: the (MatMul, I8) answer is the decode GEMV, and a
        // higher floor here must not win it on `max_by_key`.
        op: Op::MatMul,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_matmul_i8_mma",
        what: "int8 tensor-core GEMM (mma.sync m16n8k32), group-32 activation and weight scales folded per MMA",
        reported: "native:matmul_i8_mma",
        block_dim: 128,
        // 64 activation rows x 64 weight rows per block (four warps of 32x32);
        // the kernel numbers its blocks row-tile-fastest.
        tile: (64, 64),
        // 4 stages x (64x64 + 64x64 int8 tiles + 64x2 f32 group scales).
        shared_bytes: 4 * (64 * 64 + 64 * 64 + 64 * 2 * 4),
        programmatic: false,
        src: include_str!("../cu/matmul_i8_mma.cu"),
    },
    CudaKernel {
        name: "matmul_i8_mma_m16",
        // The decode-batch siblings of `matmul_i8_mma`: the same source and
        // arithmetic (every output element folds the same groups in the same
        // order, so the result is bit-identical), compiled at a smaller tile.
        // A 64x64 tile gives the up to 16-row GEMMs of a batched decode step
        // 80-272 blocks on a 132-SM part that has to stream every weight, and
        // measured 0.75-1.4 TB/s of the weights; smaller tiles put more blocks
        // and more bytes in flight (swept by `cuda_provider_matmul_i8
        // sweep_skinny`). Asked for by name by `CudaProvider` for the row
        // counts they were tuned for.
        op: Op::MatMul,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_matmul_i8_mma_m16",
        what: "int8 tensor-core GEMM at a 16x16 tile for the up to 16-row GEMMs of a batched decode step; bit-identical to matmul_i8_mma",
        reported: "native:matmul_i8_mma_m16",
        block_dim: 32,
        tile: (16, 16),
        shared_bytes: 8 * (16 * 64 + 16 * 64 + 16 * 2 * 4),
        programmatic: false,
        src: concat!(
            "#define BRAIN_I8_BM 16\n#define BRAIN_I8_BN 16\n#define BRAIN_I8_WARPS_M 1\n#define BRAIN_I8_WARPS_N 1\n#define BRAIN_I8_STAGES 8\n",
            "#define brain_matmul_i8_mma brain_matmul_i8_mma_m16\n",
            include_str!("../cu/matmul_i8_mma.cu")
        ),
    },
    CudaKernel {
        name: "matmul_i8_mma_m32",
        // The decode-batch siblings of `matmul_i8_mma`: the same source and
        // arithmetic (every output element folds the same groups in the same
        // order, so the result is bit-identical), compiled at a smaller tile.
        // A 64x64 tile gives the 17 to 32-row GEMMs of a batched decode step
        // 80-272 blocks on a 132-SM part that has to stream every weight, and
        // measured 0.75-1.4 TB/s of the weights; smaller tiles put more blocks
        // and more bytes in flight (swept by `cuda_provider_matmul_i8
        // sweep_skinny`). Asked for by name by `CudaProvider` for the row
        // counts they were tuned for.
        op: Op::MatMul,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_matmul_i8_mma_m32",
        what: "int8 tensor-core GEMM at a 32x32 tile for the 17 to 32-row GEMMs of a batched decode step; bit-identical to matmul_i8_mma",
        reported: "native:matmul_i8_mma_m32",
        block_dim: 64,
        tile: (32, 32),
        shared_bytes: 6 * (32 * 64 + 32 * 64 + 32 * 2 * 4),
        programmatic: false,
        src: concat!(
            "#define BRAIN_I8_BM 32\n#define BRAIN_I8_BN 32\n#define BRAIN_I8_WARPS_M 2\n#define BRAIN_I8_WARPS_N 1\n#define BRAIN_I8_STAGES 6\n",
            "#define brain_matmul_i8_mma brain_matmul_i8_mma_m32\n",
            include_str!("../cu/matmul_i8_mma.cu")
        ),
    },
    CudaKernel {
        name: "matmul_i8_mma_m128",
        // The decode-batch siblings of `matmul_i8_mma`: the same source and
        // arithmetic (every output element folds the same groups in the same
        // order, so the result is bit-identical), compiled at a smaller tile.
        // A 64x64 tile gives the 33 to 128-row GEMMs of a batched decode step
        // 80-272 blocks on a 132-SM part that has to stream every weight, and
        // measured 0.75-1.4 TB/s of the weights; smaller tiles put more blocks
        // and more bytes in flight (swept by `cuda_provider_matmul_i8
        // sweep_skinny`). Asked for by name by `CudaProvider` for the row
        // counts they were tuned for.
        op: Op::MatMul,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_matmul_i8_mma_m128",
        what: "int8 tensor-core GEMM at a 64x32 tile for the 33 to 128-row GEMMs of a batched decode step; bit-identical to matmul_i8_mma",
        reported: "native:matmul_i8_mma_m128",
        block_dim: 64,
        tile: (64, 32),
        shared_bytes: 4 * (64 * 64 + 32 * 64 + 32 * 2 * 4),
        programmatic: false,
        src: concat!(
            "#define BRAIN_I8_BM 64\n#define BRAIN_I8_BN 32\n#define BRAIN_I8_WARPS_M 2\n#define BRAIN_I8_WARPS_N 1\n#define BRAIN_I8_STAGES 4\n",
            "#define brain_matmul_i8_mma brain_matmul_i8_mma_m128\n",
            include_str!("../cu/matmul_i8_mma.cu")
        ),
    },
    CudaKernel {
        name: "flash_prefill_f16_hd256",
        // Looked up by `gpu_core::provider::cuda::paged_flash_prefill_step`
        // through `find`: the only (PagedAttentionFused, F32) entry.
        op: Op::PagedAttentionFused,
        weight: Dtype::F32,
        by_name: false,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_prefill_f16_hd256",
        what: "fused causal paged-attention prefill at head_dim 256 on fp16 tensor cores (fp32 pool and accumulators), 64 query rows per block",
        reported: "native:flash_prefill_f16_hd256",
        block_dim: 256,
        // 64 query rows per block; there is no column tile (the whole head is one block).
        tile: (64, 1),
        // K and V tiles of 32 keys x (256 + 8 pad) fp16.
        shared_bytes: 2 * 32 * (256 + 8) * 2,
        programmatic: false,
        src: include_str!("../cu/flash_prefill_f16_hd256.cu"),
    },
    CudaKernel {
        name: "flash_prefill_f16_hd256_bf16",
        // Asked for by name: `find` answers (PagedAttentionFused, F32) with the fp32-pool kernel.
        op: Op::PagedAttentionFused,
        weight: Dtype::BF16,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_prefill_f16_hd256_bf16",
        what: "flash_prefill_f16_hd256 over a bf16 KV cache: the planes are widened while they are staged",
        reported: "native:flash_prefill_f16_hd256_bf16",
        block_dim: 256,
        tile: (64, 1),
        shared_bytes: 2 * 32 * (256 + 8) * 2,
        programmatic: false,
        src: concat!(
            "#define BRAIN_KV_TIER 1\n#define brain_flash_prefill_f16_hd256 brain_flash_prefill_f16_hd256_bf16\n",
            include_str!("../cu/flash_prefill_f16_hd256.cu")
        ),
    },
    CudaKernel {
        name: "flash_prefill_f16_hd256_kv8",
        op: Op::PagedAttentionFused,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_prefill_f16_hd256_kv8",
        what: "flash_prefill_f16_hd256 over an int8 KV cache with one scale per (token, kv head) row: the planes are dequantised while they are staged",
        reported: "native:flash_prefill_f16_hd256_kv8",
        block_dim: 256,
        tile: (64, 1),
        shared_bytes: 2 * 32 * (256 + 8) * 2,
        programmatic: false,
        src: concat!(
            "#define BRAIN_KV_TIER 2\n#define brain_flash_prefill_f16_hd256 brain_flash_prefill_f16_hd256_kv8\n",
            include_str!("../cu/flash_prefill_f16_hd256.cu")
        ),
    },
    CudaKernel {
        name: "flash_prefill_f16_hd256_kvfp8",
        op: Op::PagedAttentionFused,
        weight: Dtype::F8E4M3,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_prefill_f16_hd256_kvfp8",
        what: "flash_prefill_f16_hd256 over an FP8 E4M3 KV cache with one scale per (token, kv head) row: the planes are decoded to fp16 while they are staged",
        reported: "native:flash_prefill_f16_hd256_kvfp8",
        block_dim: 256,
        tile: (64, 1),
        shared_bytes: 2 * 32 * (256 + 8) * 2,
        programmatic: false,
        src: concat!(
            "#define BRAIN_KV_TIER 3\n#define brain_flash_prefill_f16_hd256 brain_flash_prefill_f16_hd256_kvfp8\n",
            include_str!("../cu/flash_prefill_f16_hd256.cu")
        ),
    },
    CudaKernel {
        name: "flash_prefill_f16_hd256_v2_bf16",
        // Asked for by name by `paged_flash_prefill_bf16_step` for head_dim 256 while the version-2 switch is on.
        op: Op::PagedAttentionFused,
        weight: Dtype::BF16,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_prefill_f16_hd256_v2_bf16",
        what: "flash_prefill_f16_hd256 version 2 over a bf16 KV cache: one warp per 16 query rows over the whole head (no duplicated scores), Q in shared memory, two blocks per SM",
        reported: "native:flash_prefill_f16_hd256_v2_bf16",
        block_dim: 128,
        tile: (64, 1),
        // Everything is dynamic (opted in): Q, K and V tiles, see `FLASH_PREFILL_V2_DYNAMIC_SHARED_BYTES` in gpu-core.
        shared_bytes: 0,
        programmatic: false,
        src: concat!(
            "#define BRAIN_FLASH_HELPERS_ONLY 1\n#define BRAIN_KV_TIER 1\n",
            include_str!("../cu/flash_prefill_f16_hd256.cu"),
            "\n#define brain_flash_prefill_f16_hd256_v2 brain_flash_prefill_f16_hd256_v2_bf16\n",
            include_str!("../cu/flash_prefill_f16_hd256_v2.cu")
        ),
    },
    CudaKernel {
        name: "flash_prefill_f16_hd256_v2_kv8",
        // Asked for by name by `paged_flash_prefill_kv8_step` for head_dim 256 while the version-2 switch is on.
        op: Op::PagedAttentionFused,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_prefill_f16_hd256_v2_kv8",
        what: "flash_prefill_f16_hd256 version 2 over an int8 KV cache with one scale per (token, kv head) row: one warp per 16 query rows over the whole head, raw bytes widened at the shared-memory write, two blocks per SM",
        reported: "native:flash_prefill_f16_hd256_v2_kv8",
        block_dim: 128,
        tile: (64, 1),
        // Everything is dynamic (opted in): Q, K and V tiles, see `FLASH_PREFILL_V2_DYNAMIC_SHARED_BYTES` in gpu-core.
        shared_bytes: 0,
        programmatic: false,
        src: concat!(
            "#define BRAIN_FLASH_HELPERS_ONLY 1\n#define BRAIN_KV_TIER 2\n",
            include_str!("../cu/flash_prefill_f16_hd256.cu"),
            "\n#define brain_flash_prefill_f16_hd256_v2 brain_flash_prefill_f16_hd256_v2_kv8\n",
            include_str!("../cu/flash_prefill_f16_hd256_v2.cu")
        ),
    },
    CudaKernel {
        name: "flash_prefill_f16_hd256_v2_kvfp8",
        // Asked for by name by `paged_flash_prefill_kvfp8_step` for head_dim 256 while the version-2 switch is on.
        op: Op::PagedAttentionFused,
        weight: Dtype::F8E4M3,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_prefill_f16_hd256_v2_kvfp8",
        what: "flash_prefill_f16_hd256 version 2 over an FP8 E4M3 KV cache with one scale per (token, kv head) row: one warp per 16 query rows over the whole head, raw bytes decoded at the shared-memory write, two blocks per SM",
        reported: "native:flash_prefill_f16_hd256_v2_kvfp8",
        block_dim: 128,
        tile: (64, 1),
        // Everything is dynamic (opted in): Q, K and V tiles, see `FLASH_PREFILL_V2_DYNAMIC_SHARED_BYTES` in gpu-core.
        shared_bytes: 0,
        programmatic: false,
        src: concat!(
            "#define BRAIN_FLASH_HELPERS_ONLY 1\n#define BRAIN_KV_TIER 3\n",
            include_str!("../cu/flash_prefill_f16_hd256.cu"),
            "\n#define brain_flash_prefill_f16_hd256_v2 brain_flash_prefill_f16_hd256_v2_kvfp8\n",
            include_str!("../cu/flash_prefill_f16_hd256_v2.cu")
        ),
    },
    CudaKernel {
        name: "flash_prefill_f16_hd128",
        // Asked for by name: `find` answers (PagedAttentionFused, F32) with the head_dim 256 kernel.
        op: Op::PagedAttentionFused,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_prefill_f16_hd128",
        what: "flash_prefill_f16_hd256 at head_dim 128 (Qwen3): fp32 pool, fp16 tensor cores, fp32 accumulators",
        reported: "native:flash_prefill_f16_hd128",
        block_dim: 256,
        tile: (64, 1),
        // K and V tiles of 32 keys x (128 + 8 pad) fp16.
        shared_bytes: 2 * 32 * (128 + 8) * 2,
        programmatic: false,
        src: concat!(
            "#define BRAIN_HD 128\n#define brain_flash_prefill_f16_hd256 brain_flash_prefill_f16_hd128\n",
            include_str!("../cu/flash_prefill_f16_hd256.cu")
        ),
    },
    CudaKernel {
        name: "flash_prefill_f16_hd128_bf16",
        // Asked for by name: `find` answers (PagedAttentionFused, F32) with the head_dim 256 kernel.
        op: Op::PagedAttentionFused,
        weight: Dtype::BF16,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_prefill_f16_hd128_bf16",
        what: "flash_prefill_f16_hd128 over a bf16 KV cache",
        reported: "native:flash_prefill_f16_hd128_bf16",
        block_dim: 256,
        tile: (64, 1),
        // K and V tiles of 32 keys x (128 + 8 pad) fp16.
        shared_bytes: 2 * 32 * (128 + 8) * 2,
        programmatic: false,
        src: concat!(
            "#define BRAIN_HD 128\n#define BRAIN_KV_TIER 1\n#define brain_flash_prefill_f16_hd256 brain_flash_prefill_f16_hd128_bf16\n",
            include_str!("../cu/flash_prefill_f16_hd256.cu")
        ),
    },
    CudaKernel {
        name: "flash_prefill_f16_hd128_kv8",
        // Asked for by name: `find` answers (PagedAttentionFused, F32) with the head_dim 256 kernel.
        op: Op::PagedAttentionFused,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_prefill_f16_hd128_kv8",
        what: "flash_prefill_f16_hd128 over an int8 KV cache with one scale per (token, kv head) row",
        reported: "native:flash_prefill_f16_hd128_kv8",
        block_dim: 256,
        tile: (64, 1),
        // K and V tiles of 32 keys x (128 + 8 pad) fp16.
        shared_bytes: 2 * 32 * (128 + 8) * 2,
        programmatic: false,
        src: concat!(
            "#define BRAIN_HD 128\n#define BRAIN_KV_TIER 2\n#define brain_flash_prefill_f16_hd256 brain_flash_prefill_f16_hd128_kv8\n",
            include_str!("../cu/flash_prefill_f16_hd256.cu")
        ),
    },
    CudaKernel {
        name: "flash_prefill_f16_hd128_kvfp8",
        // Asked for by name: `find` answers (PagedAttentionFused, F32) with the head_dim 256 kernel.
        op: Op::PagedAttentionFused,
        weight: Dtype::F8E4M3,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_prefill_f16_hd128_kvfp8",
        what: "flash_prefill_f16_hd128 over an FP8 E4M3 KV cache with one scale per (token, kv head) row",
        reported: "native:flash_prefill_f16_hd128_kvfp8",
        block_dim: 256,
        tile: (64, 1),
        // K and V tiles of 32 keys x (128 + 8 pad) fp16.
        shared_bytes: 2 * 32 * (128 + 8) * 2,
        programmatic: false,
        src: concat!(
            "#define BRAIN_HD 128\n#define BRAIN_KV_TIER 3\n#define brain_flash_prefill_f16_hd256 brain_flash_prefill_f16_hd128_kvfp8\n",
            include_str!("../cu/flash_prefill_f16_hd256.cu")
        ),
    },
    CudaKernel {
        name: "flash_attn_fwd_f16_hd128",
        // Asked for by name by `gpu_core::provider::cuda::flash_causal_gqa_fwd_step`.
        op: Op::PagedAttentionFused,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_attn_fwd_f16_hd128",
        what: "causal GQA flash attention forward over dense [B*T] buffers at head_dim 128 on fp16 tensor cores (fp32 accumulators), writing the row log-sum-exp the backward rebuilds its weights from",
        reported: "native:flash_attn_fwd_f16_hd128",
        block_dim: 256,
        // 64 query rows per block of one (sample, head).
        tile: (64, 1),
        shared_bytes: 2 * 32 * (128 + 8) * 2,
        programmatic: false,
        src: concat!(
            "#define BRAIN_DENSE 1\n#define BRAIN_HD 128\n#define brain_flash_prefill_f16_hd256 brain_flash_attn_fwd_f16_hd128\n",
            include_str!("../cu/flash_prefill_f16_hd256.cu")
        ),
    },
    CudaKernel {
        name: "flash_attn_fwd_f16_hd256",
        // Asked for by name by `gpu_core::provider::cuda::flash_causal_gqa_fwd_step`.
        op: Op::PagedAttentionFused,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_attn_fwd_f16_hd256",
        what: "causal GQA flash attention forward over dense [B*T] buffers at head_dim 256 on fp16 tensor cores (fp32 accumulators), writing the row log-sum-exp the backward rebuilds its weights from",
        reported: "native:flash_attn_fwd_f16_hd256",
        block_dim: 256,
        // 64 query rows per block of one (sample, head).
        tile: (64, 1),
        shared_bytes: 2 * 32 * (256 + 8) * 2,
        programmatic: false,
        src: concat!(
            "#define BRAIN_DENSE 1\n#define BRAIN_HD 256\n#define brain_flash_prefill_f16_hd256 brain_flash_attn_fwd_f16_hd256\n",
            include_str!("../cu/flash_prefill_f16_hd256.cu")
        ),
    },
    CudaKernel {
        name: "flash_bwd_absmax",
        // Asked for by name by `gpu_core::provider::cuda::flash_causal_gqa_bwd_steps`.
        op: Op::PagedAttentionFused,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_bwd_absmax",
        what: "the largest |dO| of each of 256 blocks of one flash attention backward call (first pass of its power-of-two scale)",
        reported: "native:flash_bwd_absmax",
        block_dim: 512,
        tile: (64, 1),
        shared_bytes: 16 * 4,
        programmatic: false,
        src: include_str!("../cu/flash_attn_bwd_f16.cu"),
    },
    CudaKernel {
        name: "flash_bwd_scale",
        // Asked for by name by `gpu_core::provider::cuda::flash_causal_gqa_bwd_steps`.
        op: Op::PagedAttentionFused,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_bwd_scale",
        what: "the power-of-two scale of one flash attention backward call: the 256 partial maxima of |dO| reduced and mapped to [32, 64)",
        reported: "native:flash_bwd_scale",
        block_dim: 256,
        tile: (64, 1),
        shared_bytes: 8 * 4,
        programmatic: false,
        src: include_str!("../cu/flash_attn_bwd_f16.cu"),
    },
    CudaKernel {
        name: "flash_bwd_dq",
        // Asked for by name by `gpu_core::provider::cuda::flash_causal_gqa_bwd_steps`.
        op: Op::PagedAttentionFused,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_bwd_dq",
        what: "causal GQA flash attention backward, dQ and D, over dense [B*T] buffers at head_dim 128 on fp16 tensor cores (fp32 accumulators)",
        reported: "native:flash_bwd_dq",
        block_dim: 256,
        tile: (64, 1),
        shared_bytes: 2 * 32 * (128 + 8) * 2,
        programmatic: false,
        src: include_str!("../cu/flash_attn_bwd_f16.cu"),
    },
    CudaKernel {
        name: "flash_bwd_dkv",
        // Asked for by name by `gpu_core::provider::cuda::flash_causal_gqa_bwd_steps`.
        op: Op::PagedAttentionFused,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_bwd_dkv",
        what: "causal GQA flash attention backward, dK and dV (summed over the query heads of the group), over dense [B*T] buffers at head_dim 128 on fp16 tensor cores",
        reported: "native:flash_bwd_dkv",
        block_dim: 256,
        tile: (64, 1),
        shared_bytes: 2 * 32 * (128 + 8) * 2 + 2 * 32 * 4,
        programmatic: false,
        src: include_str!("../cu/flash_attn_bwd_f16.cu"),
    },
    CudaKernel {
        name: "decode_attn_batched_f32",
        // Asked for by name by `gpu_core::provider::cuda::paged_decode_batched_step`.
        op: Op::PagedAttentionFused,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_decode_attn_batched_f32",
        what: "batched paged-KV decode attention at head_dim 128 over an fp32 pool: one launch per layer for 2 or more sequences, the query heads of a group sharing each K and V read",
        reported: "native:decode_attn_batched_f32",
        block_dim: 256,
        tile: (1, 1),
        // Shared: 8 warps x 8 heads x (max, sum) plus the merge tile of 8 x 8 x 32 float4.
        shared_bytes: 8 * 8 * 4 * 2 + 8 * 8 * 32 * 16,
        programmatic: false,
        src: concat!(
            "#define brain_decode_attn_batched brain_decode_attn_batched_f32\n",
            include_str!("../cu/decode_attn_batched.cu")
        ),
    },
    CudaKernel {
        name: "decode_attn_batched_i8",
        // Asked for by name by `gpu_core::provider::cuda::paged_decode_batched_step`.
        op: Op::PagedAttentionFused,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_decode_attn_batched_i8",
        what: "batched paged-KV decode attention at head_dim 128 over an int8 pool with one scale per (token, kv head) row",
        reported: "native:decode_attn_batched_i8",
        block_dim: 256,
        tile: (1, 1),
        // Shared: 8 warps x 8 heads x (max, sum) plus the merge tile of 8 x 8 x 32 float4.
        shared_bytes: 8 * 8 * 4 * 2 + 8 * 8 * 32 * 16,
        programmatic: false,
        src: concat!(
            "#define BRAIN_KV_INT8 1\n#define brain_decode_attn_batched brain_decode_attn_batched_i8\n",
            include_str!("../cu/decode_attn_batched.cu")
        ),
    },
    CudaKernel {
        name: "paged_flash_decode_gqa_hd256_kv8",
        // Redirected to by name from the WGSL `paged_flash_decode_gqa_hd256#GRP=6#kv=int8` (`gpu_core::native_upgrade`).
        op: Op::PagedAttentionFused,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        // mma.sync.m16n8k16 with f16 inputs: the floor every fp16 tensor-core kernel here shares.
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_paged_flash_decode_gqa_hd256_kv8",
        what: "split-key GQA decode attention at head_dim 256 and group 6 over an int8 paged pool on f16 tensor cores (int8 exact in f16, query and probabilities split into two f16 parts): each warp runs its own 16-key chunks, every head of the group from one read of each row; the partial records of the WGSL kernel",
        reported: "native:paged_flash_decode_gqa_hd256_kv8",
        block_dim: 128,
        // One block per (sequence, kv head, key split).
        tile: (1, 1),
        // The query fragments (16 k-steps x 32 lanes x 16 bytes) plus the merge of 4 warps x 8 heads: 256 outputs, max, sum.
        shared_bytes: 16 * 32 * 16 + 4 * 8 * (256 + 2) * 4,
        programmatic: true,
        src: concat!("#define BRAIN_GRP 6\n", include_str!("../cu/paged_flash_decode_gqa_hd256_kv8.cu")),
    },
    CudaKernel {
        name: "paged_flash_decode_gqa_hd256_kvfp8",
        // Redirected to by name from the WGSL `paged_flash_decode_gqa_hd256#GRP=6#kv=fp8` (`gpu_core::native_upgrade`).
        op: Op::PagedAttentionFused,
        weight: Dtype::F8E4M3,
        by_name: true,
        source: ImplSource::Tuned,
        // mma.sync.m16n8k16 with f16 inputs, and the E4M3 pair conversion (`cvt.rn.f16x2.e4m3x2`, sm_89).
        min_cc: FP8_CVT_MIN_CC,
        entry: "brain_paged_flash_decode_gqa_hd256_kvfp8",
        what: "paged_flash_decode_gqa_hd256_kv8 over an FP8 E4M3 pool with one scale per (token, kv head) row: the elements widen to f16 exactly",
        reported: "native:paged_flash_decode_gqa_hd256_kvfp8",
        block_dim: 128,
        tile: (1, 1),
        shared_bytes: 16 * 32 * 16 + 4 * 8 * (256 + 2) * 4,
        programmatic: true,
        src: concat!(
            "#define BRAIN_GRP 6\n#define BRAIN_KV_FP8 1\n#define brain_paged_flash_decode_gqa_hd256_kv8 brain_paged_flash_decode_gqa_hd256_kvfp8\n",
            include_str!("../cu/paged_flash_decode_gqa_hd256_kv8.cu")
        ),
    },
    CudaKernel {
        name: "paged_flash_decode_gqa_hd256_f32",
        // Redirected to by name from the WGSL `paged_flash_decode_gqa_hd256#GRP=6` (`gpu_core::native_upgrade`).
        op: Op::PagedAttentionFused,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        // mma.sync.m16n8k8 with TF32 operands.
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_paged_flash_decode_gqa_hd256_f32",
        what: "split-key GQA decode attention at head_dim 256 and group 6 over an f32 paged pool on TF32 tensor cores (operands split hi/lo, three products): each warp runs its own 16-key chunks, every head of the group from one read; the partial records of the WGSL kernel",
        reported: "native:paged_flash_decode_gqa_hd256_f32",
        block_dim: 128,
        tile: (1, 1),
        // The query fragments and, reusing their bytes, the merge of 4 warps x 8 heads x 256 outputs; the maxima and sums.
        shared_bytes: 4 * 8 * 256 * 4 + 2 * 4 * 8 * 4,
        programmatic: true,
        src: concat!(
            "#define BRAIN_GRP 6\n#define brain_paged_flash_decode_gqa_hd256_tf32 brain_paged_flash_decode_gqa_hd256_f32\n",
            include_str!("../cu/paged_flash_decode_gqa_hd256_tf32.cu")
        ),
    },
    CudaKernel {
        name: "paged_flash_decode_gqa_hd256_bf16",
        // Redirected to by name from the WGSL `paged_flash_decode_gqa_hd256#GRP=6#kv=bf16` (`gpu_core::native_upgrade`).
        op: Op::PagedAttentionFused,
        weight: Dtype::BF16,
        by_name: true,
        source: ImplSource::Tuned,
        // mma.sync.m16n8k8 with TF32 operands.
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_paged_flash_decode_gqa_hd256_bf16",
        what: "paged_flash_decode_gqa_hd256_f32 over a bf16 pool: bf16 is exact in TF32, so only the query and probabilities are split",
        reported: "native:paged_flash_decode_gqa_hd256_bf16",
        block_dim: 128,
        tile: (1, 1),
        // The query fragments and, reusing their bytes, the merge of 4 warps x 8 heads x 256 outputs; the maxima and sums.
        shared_bytes: 4 * 8 * 256 * 4 + 2 * 4 * 8 * 4,
        programmatic: true,
        src: concat!(
            "#define BRAIN_GRP 6\n#define BRAIN_KV_BF16 1\n#define brain_paged_flash_decode_gqa_hd256_tf32 brain_paged_flash_decode_gqa_hd256_bf16\n",
            include_str!("../cu/paged_flash_decode_gqa_hd256_tf32.cu")
        ),
    },
    CudaKernel {
        name: "paged_flash_decode_gqa_combine",
        // Redirected to by name from the WGSL `paged_flash_decode_gqa_combine` (`gpu_core::native_upgrade`).
        op: Op::PagedAttentionFused,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_paged_flash_decode_gqa_combine",
        what: "merge of the split-key decode attention's partial records: the global max first, every split's weight once in shared memory, then independent per-channel sums (no serial exp chain)",
        reported: "native:paged_flash_decode_gqa_combine",
        block_dim: 64,
        // One block per 64 channels of a (sequence, head).
        tile: (1, 1),
        // Split weights (at most 4096) + 2 warp partials + the max and the sum.
        shared_bytes: 4 * (4096 + 2 + 2),
        programmatic: true,
        src: include_str!("../cu/paged_flash_decode_gqa_combine.cu"),
    },
    CudaKernel {
        name: "kv_append_i8_clipped",
        // Redirected to by name from the WGSL `paged_kv_append_i8_clipped_batched` (`gpu_core::native_upgrade`).
        op: Op::Elementwise,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_kv_append_i8_clipped",
        what: "int8 paged-KV append with a clipped per-(token, kv head) scale, one warp per head instead of one thread; bit-identical to paged_kv_append_i8_clipped_batched",
        reported: "native:kv_append_i8_clipped",
        block_dim: 128,
        // Four (token, kv head) pairs per block.
        tile: (4, 1),
        shared_bytes: 0,
        programmatic: true,
        src: include_str!("../cu/kv_append_i8.cu"),
    },
    CudaKernel {
        name: "gdn_chunk_loop_f32",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        by_name: false,
        source: ImplSource::Tuned,
        // Plain fp32 arithmetic and shared memory: nothing here needs a newer
        // architecture. Which devices take it is the model's decision.
        min_cc: BASELINE_MIN_CC,
        entry: "brain_gdn_chunk_loop_f32",
        what: "Gated DeltaNet's whole across-chunk recurrence in one launch (fp32, bit-identical to the nine-dispatch-per-chunk sequence), head split over four 32-column blocks",
        reported: "native:gdn_chunk_loop_f32",
        block_dim: 128,
        // One block covers the whole chunk (up to 64 rows) of one head and 32 of its 128 value columns.
        tile: (64, 32),
        // State slice 128x32 + v_new 64x32 + the staged A tile 32x(128+1), all f32.
        shared_bytes: (128 * 32 + 64 * 32 + 32 * 129) * 4,
        programmatic: false,
        src: include_str!("../cu/gdn_chunk_loop_f32.cu"),
    },
    CudaKernel {
        name: "matmul_i8_gemv",
        op: Op::MatMul,
        weight: Dtype::I8,
        by_name: false,
        source: ImplSource::Tuned,
        // `__dp4a` is the only instruction above the toolchain baseline.
        min_cc: DP4A_MIN_CC,
        entry: "brain_matmul_i8_gemv",
        what: "packed-int8 skinny-M GEMV (up to 8 rows of x per weight pass); 16 B weight loads and dp4a, bit-identical to matmul_i8_gemv_reg below 5 rows (and everywhere without int8 tensor cores), int8 tensor cores (mma.sync m16n8k32) from 5 rows on compute capability 8.0+",
        reported: "native:matmul_i8_gemv",
        block_dim: 128,
        // 8 rows of x by 16 weight rows per block.
        tile: (8, 16),
        shared_bytes: 0,
        programmatic: true,
        src: include_str!("../cu/matmul_i8_gemv.cu"),
    },
    CudaKernel {
        name: "matmul_i8_gemv_wide",
        // The same source built with `BRAIN_I8G_WIDE_TILE`, for devices with int8 tensor cores: a block serves up to
        // 32 rows of x against one read of its 16 weight rows, so a batched decode step streams every weight
        // matrix from memory once instead of once per eight rows. One candidate of the int8 decode linear's set
        // beside `matmul_i8_gemv` (`gpu_core::native_tune` chooses among them), asked for by name.
        op: Op::MatMul,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_matmul_i8_gemv_wide",
        what: "packed-int8 skinny-M GEMV for up to 32 rows of x per weight pass on int8 tensor cores (mma.sync m16n8k32, one MMA per scale group and tile of eight rows); bit-identical to matmul_i8_gemv below 5 rows",
        reported: "native:matmul_i8_gemv_wide",
        block_dim: 128,
        // 32 rows of x by 16 weight rows per block.
        tile: (32, 16),
        shared_bytes: 0,
        programmatic: true,
        src: concat!(
            "#define BRAIN_I8G_WIDE_TILE 1\n#define brain_matmul_i8_gemv brain_matmul_i8_gemv_wide\n",
            include_str!("../cu/matmul_i8_gemv.cu")
        ),
    },
    CudaKernel {
        name: "matmul_i8_gemv_multi",
        // Up to four matrices that read one activation, in one launch: a second
        // entry point of the single-matrix kernel's own file, which shares its
        // block function. The operator and tier are the single kernel's, and
        // it is a fused kernel asked for by name (`gpu_core::Fused`), never
        // resolved by `find`, whose answer for (MatMul, I8) stays the
        // single-matrix kernel.
        op: Op::MatMul,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: DP4A_MIN_CC,
        entry: "brain_matmul_i8_gemv_multi",
        what: "up to four int8 projections of one activation in one launch (blocks of every matrix fill the card together); runs the single GEMV's own block function, bit-identical to separate launches",
        reported: "native:matmul_i8_gemv_multi",
        block_dim: 128,
        tile: (8, 16),
        shared_bytes: 0,
        programmatic: true,
        src: include_str!("../cu/matmul_i8_gemv.cu"),
    },
    CudaKernel {
        name: "matmul_q4_gemv",
        op: Op::MatMul,
        weight: Dtype::Q4,
        by_name: false,
        source: ImplSource::Tuned,
        // `__dp4a` is the only instruction above the toolchain baseline.
        min_cc: DP4A_MIN_CC,
        entry: "brain_matmul_q4_gemv",
        what: "packed-int4 (W4A8) skinny-M GEMV (up to 8 rows of x per weight pass); 16 B weight loads, nibbles unpacked to dp4a operands, bit-identical to matmul_q4_gemv_reg",
        reported: "native:matmul_q4_gemv",
        block_dim: 256,
        // 8 rows of x by 16 weight rows per block.
        tile: (8, 16),
        // The double-buffered tile of x the batched rows read (`BrainQ4XTile`).
        shared_bytes: 2 * 8 * 64 * 16,
        programmatic: true,
        src: include_str!("../cu/matmul_q4_gemv.cu"),
    },
    CudaKernel {
        name: "matmul_q4_gemv_multi",
        // Up to four q4 matrices that read one activation, in one launch: the
        // q4 twin of `matmul_i8_gemv_multi`, a second entry point of the
        // single-matrix kernel's file sharing its block function, asked for by
        // name (`gpu_core::Fused`).
        op: Op::MatMul,
        weight: Dtype::Q4,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: DP4A_MIN_CC,
        entry: "brain_matmul_q4_gemv_multi",
        what: "up to four q4 projections of one activation in one launch; runs the single q4 GEMV's own block function, bit-identical to separate launches",
        reported: "native:matmul_q4_gemv_multi",
        block_dim: 256,
        tile: (8, 16),
        shared_bytes: 2 * 8 * 64 * 16,
        programmatic: true,
        src: include_str!("../cu/matmul_q4_gemv.cu"),
    },
    CudaKernel {
        name: "matmul_kq4_gemv",
        op: Op::MatMul,
        weight: Dtype::Q4K,
        by_name: false,
        source: ImplSource::Tuned,
        min_cc: DP4A_MIN_CC,
        entry: "brain_matmul_kq4_gemv",
        what: "affine K-quant (Q4_K) skinny-M GEMV (up to 8 rows of x per weight pass); one 32-element group per 8 B vector load, exact int32 group dot and activation sum, two roundings a group",
        reported: "native:matmul_kq4_gemv",
        block_dim: 256,
        // 8 rows of x by 16 weight rows per block.
        tile: (8, 16),
        // The double-buffered tile of x the batched rows read (`BrainKqXShared<8, 2>`).
        shared_bytes: 2 * 8 * 64 * 16,
        programmatic: true,
        src: include_str!("../cu/matmul_kq_gemv.cu"),
    },
    CudaKernel {
        name: "matmul_kq8_gemv",
        op: Op::MatMul,
        weight: Dtype::Q8K,
        by_name: false,
        source: ImplSource::Tuned,
        min_cc: DP4A_MIN_CC,
        entry: "brain_matmul_kq8_gemv",
        what: "affine K-quant (Q5_K) skinny-M GEMV (up to 8 rows of x per weight pass); one 32-element group per 16 B vector load, exact int32 group dot and activation sum, two roundings a group",
        reported: "native:matmul_kq8_gemv",
        block_dim: 256,
        // 8 rows of x by 16 weight rows per block.
        tile: (8, 16),
        // The double-buffered tile of x the batched rows read (`BrainKqXShared<8, 2>`).
        shared_bytes: 2 * 8 * 64 * 16,
        programmatic: true,
        src: include_str!("../cu/matmul_kq_gemv.cu"),
    },
    CudaKernel {
        name: "matmul_kq4_gemv_wide",
        // The Q4_K GEMV's block with four weight rows a thread, for batched
        // decode; asked for by name by `gpu_core::native_upgrade`.
        op: Op::MatMul,
        weight: Dtype::Q4K,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: DP4A_MIN_CC,
        entry: "brain_matmul_kq4_gemv_wide",
        what: "affine K-quant (Q4_K) GEMV for batched decode (9 to 32 rows of x): four weight rows a thread, so each staged activation vector and its sum serve four weight rows",
        reported: "native:matmul_kq4_gemv_wide",
        block_dim: 256,
        // 8 rows of x by 64 weight rows per block.
        tile: (8, 64),
        // The double-buffered 8-row tile of x (`BrainKqXShared<8, 1>`).
        shared_bytes: 2 * 8 * 32 * 16,
        programmatic: true,
        src: include_str!("../cu/matmul_kq_gemv.cu"),
    },
    CudaKernel {
        name: "matmul_kq8_gemv_wide",
        // The Q5_K GEMV's block with four weight rows a thread, for batched
        // decode; asked for by name by `gpu_core::native_upgrade`.
        op: Op::MatMul,
        weight: Dtype::Q8K,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: DP4A_MIN_CC,
        entry: "brain_matmul_kq8_gemv_wide",
        what: "affine K-quant (Q5_K) GEMV for batched decode (9 to 32 rows of x): four weight rows a thread, so each staged activation vector and its sum serve four weight rows",
        reported: "native:matmul_kq8_gemv_wide",
        block_dim: 256,
        // 8 rows of x by 64 weight rows per block.
        tile: (8, 64),
        // The double-buffered 8-row tile of x (`BrainKqXShared<8, 1>`).
        shared_bytes: 2 * 8 * 32 * 16,
        programmatic: true,
        src: include_str!("../cu/matmul_kq_gemv.cu"),
    },
    CudaKernel {
        name: "matmul_f32_reg",
        // The native twin of `matmul_reg3.wgsl`, redirected to BY NAME by
        // `gpu_core::native_upgrade`; the (MatMul, F32) operator answer stays
        // the provider's `matmul_f32_tiled`.
        op: Op::MatMul,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_matmul_f32_reg",
        what: "fp32 out = x @ W^T; 128x128 tile, 8x8 per thread, product and sum each rounded in ascending k, bit-identical to matmul_reg3",
        reported: "native:matmul_f32_reg",
        block_dim: 256,
        // 128 rows of x by 128 weight rows per block, row-tile-major.
        tile: (128, 128),
        // Double-buffered k-major tiles of 16 k x (128 + 4 pad) floats per operand.
        shared_bytes: 2 * 2 * 16 * (128 + 4) * 4,
        programmatic: false,
        src: include_str!("../cu/matmul_f32_reg.cu"),
    },
    CudaKernel {
        name: "matmul_f32_dx_reg",
        // The native twin of `matmul_dx_reg.wgsl` (the input gradient of
        // `matmul_reg3`), a second entry of the same file; redirected to by
        // name by `gpu_core::native_upgrade`.
        op: Op::MatMul,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_matmul_f32_dx_reg",
        what: "fp32 dX = dY @ W (optionally accumulated); 128x128 tile, 8x8 per thread, product and sum each rounded in ascending n, bit-identical to matmul_dx_reg",
        reported: "native:matmul_f32_dx_reg",
        block_dim: 256,
        // 128 rows of dY by 128 columns of W per block, row-tile-major.
        tile: (128, 128),
        shared_bytes: 2 * 2 * 16 * (128 + 4) * 4,
        programmatic: false,
        src: include_str!("../cu/matmul_f32_reg.cu"),
    },
    CudaKernel {
        name: "matmul_f32_dw_reg",
        // The native twin of `matmul_dw_reg.wgsl` (an accumulating weight
        // gradient), a third entry of the same file.
        op: Op::MatMul,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_matmul_f32_dw_reg",
        what: "fp32 dW += A^T @ B; 128x128 tile, 8x8 per thread, product and sum each rounded in ascending m, bit-identical to matmul_dw_reg",
        reported: "native:matmul_f32_dw_reg",
        block_dim: 256,
        // 128 columns of A by 128 columns of B per block, row-tile-major.
        tile: (128, 128),
        shared_bytes: 2 * 2 * 16 * (128 + 4) * 4,
        programmatic: false,
        src: include_str!("../cu/matmul_f32_reg.cu"),
    },
    CudaKernel {
        name: "matmul_f32_fma",
        // Asked for by name (`gpu_core::Fused`) by a trainer, whose contract is
        // agreement with its host reference rather than bits of a WGSL twin.
        op: Op::MatMul,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_matmul_f32_fma",
        what: "fp32 out = x @ W^T with one fused multiply-add per term (training; matmul_f32_reg's tile and contract)",
        reported: "native:matmul_f32_fma",
        block_dim: 256,
        tile: (128, 128),
        shared_bytes: 2 * 2 * 16 * (128 + 4) * 4,
        programmatic: false,
        src: include_str!("../cu/matmul_f32_reg.cu"),
    },
    CudaKernel {
        name: "matmul_f32_dx_fma",
        // Asked for by name (`gpu_core::Fused`) by a trainer, whose contract is
        // agreement with its host reference rather than bits of a WGSL twin.
        op: Op::MatMul,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_matmul_f32_dx_fma",
        what: "fp32 dX = dY @ W (optionally accumulated) with fused multiply-adds (training; matmul_f32_dx_reg's tile and contract)",
        reported: "native:matmul_f32_dx_fma",
        block_dim: 256,
        tile: (128, 128),
        shared_bytes: 2 * 2 * 16 * (128 + 4) * 4,
        programmatic: false,
        src: include_str!("../cu/matmul_f32_reg.cu"),
    },
    CudaKernel {
        name: "matmul_f32_dw_fma",
        // Asked for by name (`gpu_core::Fused`) by a trainer, whose contract is
        // agreement with its host reference rather than bits of a WGSL twin.
        op: Op::MatMul,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_matmul_f32_dw_fma",
        what: "fp32 dW += A^T @ B with fused multiply-adds (training; matmul_f32_dw_reg's tile and contract)",
        reported: "native:matmul_f32_dw_fma",
        block_dim: 256,
        tile: (128, 128),
        shared_bytes: 2 * 2 * 16 * (128 + 4) * 4,
        programmatic: false,
        src: include_str!("../cu/matmul_f32_reg.cu"),
    },
    CudaKernel {
        name: "flash_bidir_f32",
        // Redirected to by name from the WGSL `flash_attn_bidir_reg2`
        // (`gpu_core::native_upgrade`) at head_dim 128.
        op: Op::FlashAttention,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_flash_bidir_f32",
        what: "bidirectional flash attention at head_dim 128, fp32: 64 query rows per 128-thread block with q resident in shared memory, 32-key k/v tiles, 4x4 score and 4x16 output register blocks, probabilities moved by shuffle",
        reported: "native:flash_bidir_f32",
        block_dim: 128,
        // 64 query rows of one (sample, head) per block, all 128 channels.
        tile: (64, 128),
        // q: 64 x 128 floats; one k/v tile: 32 x 128 floats.
        shared_bytes: (64 * 128 + 32 * 128) * 4,
        programmatic: false,
        src: include_str!("../cu/flash_attn_f32.cu"),
    },
    CudaKernel {
        name: "flash_bidir_f16_hd64",
        // Redirected to by name from the WGSL `flash_attn_bidir_reg2` (`gpu_core::native_upgrade`) at head_dim 64.
        op: Op::FlashAttention,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_bidir_f16_hd64",
        what: "bidirectional flash attention at head_dim 64 on fp16 tensor cores (fp32 inputs, outputs and accumulators): 64 query rows per 128-thread block, one warp per 16 rows over the whole head, 64-key tiles",
        reported: "native:flash_bidir_f16_hd64",
        block_dim: 128,
        tile: (64, 64),
        // K and V tiles of 64 keys x (64 + 8 pad) fp16.
        shared_bytes: 2 * 64 * (64 + 8) * 2,
        programmatic: false,
        src: include_str!("../cu/flash_attn_f16_hd64.cu"),
    },
    CudaKernel {
        name: "flash_cross_relpos_f16_hd64",
        // Redirected to by name from the WGSL `flash_attn_cross_relpos` (`gpu_core::native_upgrade`) at head_dim 64.
        op: Op::FlashAttention,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_flash_cross_relpos_f16_hd64",
        what: "flash attention with SAM's decomposed relative-position bias at head_dim 64 on fp16 tensor cores: flash_bidir_f16_hd64's tiling with rel_h[i][j / kw] + rel_w[i][j % kw] added to each score, spans as the batch axis",
        reported: "native:flash_cross_relpos_f16_hd64",
        block_dim: 128,
        tile: (64, 64),
        shared_bytes: 2 * 64 * (64 + 8) * 2,
        programmatic: false,
        src: include_str!("../cu/flash_attn_f16_hd64.cu"),
    },
    CudaKernel {
        name: "flash_gqa_kmask_f32",
        // Asked for by name (`gpu_core::Fused::FlashGqaKmask`) by a padded
        // encoder in place of its scores -> softmax -> apply chain.
        op: Op::FlashAttention,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_flash_gqa_kmask_f32",
        what: "causal grouped-query flash attention with an additive per-key mask at head_dim 128, fp32: flash_bidir_f32's tiling, key tiles past the causal bound and the last live key never visited",
        reported: "native:flash_gqa_kmask_f32",
        block_dim: 128,
        tile: (64, 128),
        shared_bytes: (64 * 128 + 32 * 128) * 4,
        programmatic: false,
        src: include_str!("../cu/flash_attn_f32.cu"),
    },
    CudaKernel {
        name: "softmax_rows_f32_k8",
        // Redirected to by name from the WGSL `softmax_rows` for rows of at
        // most 512 columns (`gpu_core::native_upgrade`, smallest bucket first).
        op: Op::Softmax,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_softmax_rows_k8",
        what: "row softmax, 64 lanes a row holding up to 8 values each in registers: the row read once and written once, bit-identical to softmax_rows",
        reported: "native:softmax_rows_f32_k8",
        block_dim: 256,
        // Four rows per block.
        tile: (4, 512),
        shared_bytes: 4 * 64 * 4,
        programmatic: false,
        src: include_str!("../cu/softmax_rows_f32.cu"),
    },
    CudaKernel {
        name: "softmax_rows_f32_k16",
        // Redirected to by name from the WGSL `softmax_rows` for rows of at
        // most 1024 columns (`gpu_core::native_upgrade`, smallest bucket first).
        op: Op::Softmax,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_softmax_rows_k16",
        what: "row softmax, 64 lanes a row holding up to 16 values each in registers: the row read once and written once, bit-identical to softmax_rows",
        reported: "native:softmax_rows_f32_k16",
        block_dim: 256,
        // Four rows per block.
        tile: (4, 1024),
        shared_bytes: 4 * 64 * 4,
        programmatic: false,
        src: include_str!("../cu/softmax_rows_f32.cu"),
    },
    CudaKernel {
        name: "softmax_rows_f32_k32",
        // Redirected to by name from the WGSL `softmax_rows` for rows of at
        // most 2048 columns (`gpu_core::native_upgrade`, smallest bucket first).
        op: Op::Softmax,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_softmax_rows_k32",
        what: "row softmax, 64 lanes a row holding up to 32 values each in registers: the row read once and written once, bit-identical to softmax_rows",
        reported: "native:softmax_rows_f32_k32",
        block_dim: 256,
        // Four rows per block.
        tile: (4, 2048),
        shared_bytes: 4 * 64 * 4,
        programmatic: false,
        src: include_str!("../cu/softmax_rows_f32.cu"),
    },
    CudaKernel {
        name: "softmax_rows_f32_k64",
        // Redirected to by name from the WGSL `softmax_rows` for rows of at
        // most 4096 columns (`gpu_core::native_upgrade`, smallest bucket first).
        op: Op::Softmax,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_softmax_rows_k64",
        what: "row softmax, 64 lanes a row holding up to 64 values each in registers: the row read once and written once, bit-identical to softmax_rows",
        reported: "native:softmax_rows_f32_k64",
        block_dim: 256,
        // Four rows per block.
        tile: (4, 4096),
        shared_bytes: 4 * 64 * 4,
        programmatic: false,
        src: include_str!("../cu/softmax_rows_f32.cu"),
    },
    CudaKernel {
        name: "softmax_rows_dx",
        // Asked for by name (`gpu_core::Fused::SoftmaxRowsDx`) by a trainer:
        // `softmax_k_dx`'s result over contiguous rows with a reordered
        // reduction, so never a redirect of that kernel.
        op: Op::Softmax,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_softmax_rows_dx",
        what: "softmax Jacobian dx = y * (dy - sum(dy * y)) over contiguous rows, one warp per row, coalesced 16-byte loads and a shuffle reduction",
        reported: "native:softmax_rows_dx",
        block_dim: 256,
        // Eight rows (one per warp) per block.
        tile: (8, 1),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/softmax_rows_dx.cu"),
    },
    CudaKernel {
        name: "roof_dp4a",
        // The int8 roof probe `gpu_core::roof` runs where the backend compiles
        // native kernels; not a model operator, so asked for by name.
        op: Op::MatMul,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: DP4A_MIN_CC,
        entry: "brain_roof_dp4a",
        what: "peak packed-int8 dot rate: eight loop-carried __dp4a chains per thread, the int8 roof probe",
        reported: "native:roof_dp4a",
        block_dim: 256,
        tile: (1, 256),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/roof_dp4a.cu"),
    },
    CudaKernel {
        name: "matmul_i8w_dx",
        // A training kernel with no WGSL twin, asked for by name
        // (`gpu_core::Fused::I8wDx`): the input gradient of a linear whose
        // frozen base is resident as packed int8. A fourth entry of the fp32
        // GEMM file, sharing its block function.
        op: Op::MatMul,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_matmul_i8w_dx",
        what: "fp32 dX = dY @ deq(W) for a packed int8 group-32 weight dequantised while staged, fused multiply-adds; 128x128 tile, 8x8 per thread",
        reported: "native:matmul_i8w_dx",
        block_dim: 256,
        tile: (128, 128),
        shared_bytes: 2 * 2 * 16 * (128 + 4) * 4,
        programmatic: false,
        src: include_str!("../cu/matmul_f32_reg.cu"),
    },
    CudaKernel {
        name: "matmul_i8_dp4a",
        // The native twin of `matmul_i8_dyn.wgsl`, redirected to BY NAME by
        // `gpu_core::native_upgrade` for every dispatch of that kernel. The
        // (MatMul, I8) operator answer stays the decode GEMV, which `find`
        // must keep returning for the GEMV's own row.
        op: Op::MatMul,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: DP4A_MIN_CC,
        entry: "brain_matmul_i8_dp4a",
        what: "packed-int8 GEMM (dp4a), group-32 activation and weight scales folded per group in ascending order; 128x128 tile, 8x8 per thread, bit-identical to matmul_i8_dyn",
        reported: "native:matmul_i8_dp4a",
        block_dim: 256,
        // 128 rows of x by 128 weight rows per block, row-tile-major.
        tile: (128, 128),
        // Double-buffered k-major tiles of 16 words x (128 + 4 pad) for each
        // operand, and the chunk's two groups of 128 weight scales.
        shared_bytes: 2 * 2 * 16 * (128 + 4) * 4 + 2 * 2 * 128 * 4,
        programmatic: false,
        src: include_str!("../cu/matmul_i8_dp4a.cu"),
    },
    CudaKernel {
        name: "matmul_kq4_dp4a",
        // The int8 GEMM's block over affine K-quant (Q4_K) codes, asked for by
        // name by `gpu_core::native_upgrade` in place of `matmul_kq_dyn`.
        op: Op::MatMul,
        weight: Dtype::Q4K,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: DP4A_MIN_CC,
        entry: "brain_matmul_kq4_dp4a",
        what: "affine K-quant (Q4_K) GEMM: the int8 GEMM's 128x128 dp4a block with codes unpacked while staging, exact group sums of the staged activations, ds*A - dm*S folded per group with two fmas",
        reported: "native:matmul_kq4_dp4a",
        block_dim: 256,
        tile: (128, 128),
        // The int8 tiles plus each column's group mins and each row's group sums.
        shared_bytes: 2 * 2 * 16 * (128 + 4) * 4 + 3 * 2 * 2 * 128 * 4,
        programmatic: false,
        src: include_str!("../cu/matmul_i8_dp4a.cu"),
    },
    CudaKernel {
        name: "matmul_kq8_dp4a",
        // The int8 GEMM's block over affine K-quant (Q5_K) codes, asked for by
        // name by `gpu_core::native_upgrade` in place of `matmul_kq_dyn`.
        op: Op::MatMul,
        weight: Dtype::Q8K,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: DP4A_MIN_CC,
        entry: "brain_matmul_kq8_dp4a",
        what: "affine K-quant (Q5_K) GEMM: the int8 GEMM's 128x128 dp4a block with codes unpacked while staging, exact group sums of the staged activations, ds*A - dm*S folded per group with two fmas",
        reported: "native:matmul_kq8_dp4a",
        block_dim: 256,
        tile: (128, 128),
        // The int8 tiles plus each column's group mins and each row's group sums.
        shared_bytes: 2 * 2 * 16 * (128 + 4) * 4 + 3 * 2 * 2 * 128 * 4,
        programmatic: false,
        src: include_str!("../cu/matmul_i8_dp4a.cu"),
    },
    CudaKernel {
        name: "moe_i8_grouped_mma",
        // The native twin of `moe_i8_grouped.wgsl`, redirected to by
        // `gpu_core::native_upgrade`: the (MoeExpertLinear, I8) answer. The decode
        // regime's expert GEMV (`moe_i8_gemv_gather`) has no twin.
        op: Op::MoeExpertLinear,
        weight: Dtype::I8,
        by_name: false,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_moe_i8_grouped_mma",
        what: "sparse-MoE int8 GEMM over a fused expert bank with each expert's slots grouped, on int8 tensor cores (mma.sync m16n8k32), group-32 weight scale folded per MMA; bit-identical to moe_i8_grouped",
        reported: "native:moe_i8_grouped_mma",
        block_dim: 128,
        // One tile of the router's tables (8 slots of one expert) by 64 weight
        // rows per block (four warps of 16), tile-major.
        tile: (1, 64),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/moe_i8_grouped_mma.cu"),
    },
    CudaKernel {
        name: "matmul_bf16_gemv_m1",
        // The native twin of `matmul_bf16_gemv.wgsl` at one activation row, redirected to by name by
        // `gpu_core::native_upgrade`: the decode token's product of a dense bf16 weight.
        op: Op::MatMul,
        weight: Dtype::BF16,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_matmul_bf16_gemv_m1",
        what: "float-read GEMV of a dense bf16 weight against one f32 activation row: 16-byte streaming weight loads, a warp per row with eight in flight per lane, bf16 widened by a shift or a mask, fused multiply-adds, warp-shuffle row sum; f32 rounding of the running sum against the WGSL tier",
        reported: "native:matmul_bf16_gemv_m1",
        block_dim: 256,
        tile: (1, 8),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/bf16_gemv.cu"),
    },
    CudaKernel {
        name: "matmul_fp8_gemv",
        // The native twin of `matmul_fp8_gemv.wgsl` (every row-capacity variant), redirected to by name by
        // `gpu_core::native_upgrade`: the decode-regime product of a stored FP8 weight.
        op: Op::MatMul,
        weight: Dtype::F8E4M3,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_matmul_fp8_gemv",
        what: "float-read GEMV of an FP8 E4M3 weight with block or tensor scales against up to 8 f32 activation rows: 16-byte streaming weight loads, codes decoded by integer bit moves, fused multiply-adds, warp-shuffle row sum; f32 rounding of the running sum against the WGSL tier",
        reported: "native:matmul_fp8_gemv",
        block_dim: 256,
        // 16 weight rows per block, row-tile order.
        tile: (1, 16),
        shared_bytes: 8 * 1024 * 4,
        programmatic: false,
        src: include_str!("../cu/fp8_nvfp4_gemv.cu"),
    },
    CudaKernel {
        name: "matmul_fp8_gemv_m1",
        // The single-activation-row build of `matmul_fp8_gemv`: the decode token, with registers for a deeper load stage.
        op: Op::MatMul,
        weight: Dtype::F8E4M3,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_matmul_fp8_gemv_m1",
        what: "the one-activation-row form of matmul_fp8_gemv: the same FP8 GEMV with a deeper stage of in-flight weight loads, f32 rounding of the running sum against the WGSL tier",
        reported: "native:matmul_fp8_gemv_m1",
        block_dim: 256,
        tile: (1, 16),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/fp8_nvfp4_gemv.cu"),
    },
    CudaKernel {
        name: "matmul_nvfp4_gemv",
        // The native twin of `matmul_nvfp4_gemv.wgsl`: the dense NVFP4 product (the LM head, any NVFP4 projection).
        op: Op::MatMul,
        weight: Dtype::F4E2M1,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_matmul_nvfp4_gemv",
        what: "float-read GEMV of an NVFP4 weight (E2M1 codes, E4M3 scale per 16, f32 tensor scale) against up to 8 f32 activation rows: 16-byte streaming weight loads, codes decoded by integer bit moves, fused multiply-adds, warp-shuffle row sum; f32 rounding of the running sum against the WGSL tier",
        reported: "native:matmul_nvfp4_gemv",
        block_dim: 256,
        tile: (1, 16),
        shared_bytes: 8 * 1024 * 4,
        programmatic: false,
        src: include_str!("../cu/fp8_nvfp4_gemv.cu"),
    },
    CudaKernel {
        name: "matmul_nvfp4_gemv_m1",
        // The single-activation-row build of `matmul_nvfp4_gemv`: the decode token, with registers for a deeper load stage.
        op: Op::MatMul,
        weight: Dtype::F4E2M1,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_matmul_nvfp4_gemv_m1",
        what: "the one-activation-row form of matmul_nvfp4_gemv: the same NVFP4 GEMV with a deeper stage of in-flight weight loads, f32 rounding of the running sum against the WGSL tier",
        reported: "native:matmul_nvfp4_gemv_m1",
        block_dim: 256,
        tile: (1, 16),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/fp8_nvfp4_gemv.cu"),
    },
    CudaKernel {
        name: "moe_nvfp4_gemv_gather",
        // The native twin of `moe_nvfp4_gemv_gather.wgsl`: the decode regime's expert GEMV over an NVFP4 bank.
        op: Op::MoeExpertLinear,
        weight: Dtype::F4E2M1,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_moe_nvfp4_gemv_gather",
        what: "sparse-MoE NVFP4 GEMV over a fused expert bank: each slot reads its expert's 4-bit rows with 16-byte streaming loads, f32 activations, per-expert tensor scale; f32 rounding of the running sum against the WGSL tier",
        reported: "native:moe_nvfp4_gemv_gather",
        block_dim: 256,
        // 16 weight rows of one slot per block, slot-major.
        tile: (1, 16),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/fp8_nvfp4_gemv.cu"),
    },
    CudaKernel {
        name: "moe_nvfp4_gateup_silu",
        // A fused kernel asked for by name (`gpu_core::Fused::MoeNvfp4GateUp`): no WGSL twin, the model keeps its chain of
        // two gathers and a SwiGLU for devices that are not offered it.
        op: Op::MoeExpertLinear,
        weight: Dtype::F4E2M1,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_moe_nvfp4_gateup_silu",
        what: "the gate and up NVFP4 expert projections of each slot and silu(gate) * up in one launch: both banks stream through the same block, the activation row is read once, and the gate/up intermediates never reach memory",
        reported: "native:moe_nvfp4_gateup_silu",
        block_dim: 256,
        // 16 weight rows of one slot per block, slot-major.
        tile: (1, 16),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/fp8_nvfp4_gemv.cu"),
    },
    CudaKernel {
        name: "moe_router_topk",
        // The native twin of `moe_router_topk.wgsl`, redirected to by name by `gpu_core::native_upgrade`.
        op: Op::ArgMaxRow,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_moe_router_topk",
        what: "MoE router head: softmax over the experts, the top_k largest (ties to the lower expert) renormalised or kept raw and scaled, the shared expert's sigmoid weight last; warp-shuffle reductions, the reference's selection order",
        reported: "native:moe_router_topk",
        block_dim: 256,
        // One block per row.
        tile: (1, 1),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/moe_router_topk.cu"),
    },
    CudaKernel {
        name: "moe_nvfp4_grouped",
        // The native twin of `moe_nvfp4_grouped.wgsl`, redirected to by name by `gpu_core::native_upgrade`: the prefill regime's
        // expert GEMM over an NVFP4 bank.
        op: Op::MoeExpertLinear,
        weight: Dtype::F4E2M1,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_moe_nvfp4_grouped",
        what: "sparse-MoE NVFP4 GEMM over a fused expert bank with each expert's slots grouped: every weight chunk is loaded and decoded once and applied to up to 8 slots whose activations are staged in shared memory; fused multiply-adds, f32 rounding of the running sum against the WGSL tier",
        reported: "native:moe_nvfp4_grouped",
        block_dim: 128,
        // One tile of the router's tables (8 slots of one expert) by 64 weight rows per block, tile-major.
        tile: (1, 64),
        shared_bytes: 8 * 512 * 4 + 8 * 4,
        programmatic: false,
        src: include_str!("../cu/fp8_nvfp4_gemv.cu"),
    },
    CudaKernel {
        name: "add_rms_quant",
        // Not an operator a selector chooses an implementation of: a FUSED
        // kernel a model asks for by name (`gpu_core::Fused`), with no WGSL twin
        // to redirect. The operator and tier say what it produces - the packed
        // int8 activation of an int8 linear, after a RMSNorm - and are unique in
        // this table, so `find` can never confuse it with a matmul.
        op: Op::RmsNorm,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_add_rms_quant",
        what: "residual add + RMSNorm + per-group int8 scales + pack in one launch; 256 threads per row (the reference's 64 lanes, four threads each), bit-identical to add2 + rmsnorm_rows + max_abs_rows + quant_pack",
        reported: "native:add_rms_quant",
        block_dim: 256,
        // One block per row of x.
        tile: (1, 1),
        // 64 carried and 64 final lane sums + the 64 x 80 quantised bytes exchanged between lanes.
        shared_bytes: 4 * (64 + 64) + 64 * 80,
        programmatic: true,
        src: include_str!("../cu/add_rms_quant.cu"),
    },
    CudaKernel {
        name: "quant_epilogue",
        // A fused kernel asked for by name, like `add_rms_quant`: the operator
        // and tier say what it produces (group scales and a packed int8
        // activation), and a fused kernel is always looked up through `get`,
        // never through `find`, which is the matmul providers' selector.
        op: Op::MaxAbsRow,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_quant_epilogue",
        what: "silu_mul / sigmoid-gate / plain producer + per-group int8 scales + pack in one launch; 1024 threads per row, bit-identical to the chain it replaces",
        reported: "native:quant_epilogue",
        block_dim: 1024,
        tile: (1, 1),
        // 1024 x 18 quantised bytes exchanged between threads.
        shared_bytes: 1024 * 18,
        programmatic: true,
        src: include_str!("../cu/quant_epilogue.cu"),
    },
    CudaKernel {
        name: "quant_epilogue_cluster",
        // A fused kernel asked for by name, like `add_rms_quant`: the operator
        // and tier say what it produces (group scales and a packed int8
        // activation), and a fused kernel is always looked up through `get`,
        // never through `find`, which is the matmul providers' selector.
        op: Op::MaxAbsRow,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: CLUSTER_MIN_CC,
        entry: "brain_quant_epilogue_cluster",
        what: "quant_epilogue with each row split over a cluster of four blocks on four SMs, each quantising its own slices' groups; bit-identical to the one-block form",
        reported: "native:quant_epilogue_cluster",
        block_dim: 1024,
        tile: (1, 1),
        // 1024 x 18 quantised bytes exchanged between threads.
        shared_bytes: 1024 * 18,
        programmatic: true,
        src: concat!("#define BRAIN_QE_CLUSTER 4
#define brain_quant_epilogue brain_quant_epilogue_cluster
", include_str!("../cu/quant_epilogue.cu")),
    },
    CudaKernel {
        name: "gdn_decode",
        // A fused kernel asked for by name (see `add_rms_quant`): the whole
        // Gated DeltaNet decode step, which ends in a gated RMSNorm. Its
        // operator and tier are unique in this table.
        op: Op::RmsNorm,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_gdn_decode",
        what: "one Gated DeltaNet decode step in one launch: conv+SiLU, L2 norm, gates, delta-rule state update, gated RMSNorm; a block per key head, bit-identical to the 19-kernel WGSL chain",
        reported: "native:gdn_decode",
        block_dim: 384,
        // One block per key head.
        tile: (1, 1),
        // Per row (at most 8): q and k (128 each), three value heads' 128, and the
        // beta, decay and inverse-RMS of each of the three.
        shared_bytes: 8 * (2 * 128 * 4 + 3 * 128 * 4 + 3 * 3 * 4),
        programmatic: true,
        src: include_str!("../cu/gdn_decode.cu"),
    },
    CudaKernel {
        name: "gdn_decode_pool",
        // `gdn_decode` over a batch whose state lives as rows of two pools: a second
        // entry point of that kernel's own file, which shares its block body. A
        // fused kernel asked for by name (see `add_rms_quant`).
        op: Op::RmsNorm,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_gdn_decode_pool",
        what: "one Gated DeltaNet decode step for every sequence of a batch in one launch, updating each sequence's pool row of state and conv window in place; the single-sequence body per (key head, row), bit-identical to the 19-kernel WGSL chain",
        reported: "native:gdn_decode_pool",
        block_dim: 384,
        // One block per (key head, batch row).
        tile: (1, 1),
        // The shared body's staging, sized for `gdn_decode`'s at most 8 rows
        // (see that entry) though a pooled block walks a single one.
        shared_bytes: 8 * (2 * 128 * 4 + 3 * 128 * 4 + 3 * 3 * 4),
        programmatic: true,
        src: include_str!("../cu/gdn_decode.cu"),
    },
    CudaKernel {
        name: "gdn_verify_pool",
        // The speculative verify form of `gdn_decode_pool`: a third entry point
        // of that kernel's file. A fused kernel asked for by name (see
        // `add_rms_quant`).
        op: Op::RmsNorm,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_gdn_verify_pool",
        what: "up to 8 consecutive Gated DeltaNet decode steps for every sequence of a batch in one launch, each sequence from its pool row, snapshotting the state and conv window after every row but the last so a rejected tail can be undone; the same body as gdn_decode_pool, bit-identical to repeated single steps",
        reported: "native:gdn_verify_pool",
        block_dim: 384,
        // One block per (key head, sequence).
        tile: (1, 1),
        shared_bytes: 8 * (2 * 128 * 4 + 3 * 128 * 4 + 3 * 3 * 4),
        programmatic: true,
        src: include_str!("../cu/gdn_decode.cu"),
    },
    CudaKernel {
        name: "gdn_commit_pool",
        // Restores the snapshot `gdn_verify_pool` took of the last accepted row.
        op: Op::RmsNorm,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_gdn_commit_pool",
        what: "copies, for each sequence that rejected speculated rows, the snapshot of its last accepted row's recurrent state and conv window back over its pool row; a block per 4096-element tile per sequence",
        reported: "native:gdn_commit_pool",
        block_dim: 256,
        tile: (1, 1),
        shared_bytes: 0,
        programmatic: true,
        src: include_str!("../cu/gdn_decode.cu"),
    },
    CudaKernel {
        name: "gqa_decode_prep",
        // A fused kernel asked for by name (see `add_rms_quant`); the operator
        // and tier are unique in this table.
        op: Op::PagedAttention,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_gqa_decode_prep",
        what: "gated-attention decode prep in one launch, for every row of a batch: value-and-gate split, per-head QK RMSNorm, partial rotary, KV append; a block per head per row, bit-identical to the 8-kernel WGSL chain",
        reported: "native:gqa_decode_prep",
        block_dim: 256,
        // One block per head per row (query heads, then key heads).
        tile: (1, 1),
        // The head's 512 staged values and the shared inverse norm.
        shared_bytes: 512 * 4 + 4,
        programmatic: true,
        src: include_str!("../cu/gqa_decode_prep.cu"),
    },
    CudaKernel {
        name: "conv2d_fwd_f32",
        // Redirected to by name from the WGSL `conv2d` (`gpu_core::native_upgrade`);
        // `Op::Conv2d`'s selector arm belongs to a different call site
        // (`vae::blocks`), so `find` must never offer this for it.
        op: Op::Conv2d,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_conv2d_fwd",
        what: "dense fp32 conv2d forward (NCHW, any K/stride/pad) as an implicit GEMM; 16/32/64-channel x 128-position tile, double-buffered 16-deep k slices, register block, explicit FMA",
        reported: "native:conv2d_fwd_f32",
        block_dim: 256,
        // Up to 64 output channels (16 or 32 for narrower layers) x 128 positions.
        tile: (64, 128),
        // Two A slices of 16 x (64 + 4) and two B slices of 16 x 128 floats.
        shared_bytes: (2 * 16 * (64 + 4) + 2 * 16 * 128) * 4,
        programmatic: false,
        src: include_str!("../cu/conv2d_f32.cu"),
    },
    CudaKernel {
        name: "conv2d_bias_fwd_f32",
        // Redirected to by name from the WGSL `conv_bias` and `conv_bias_reg`.
        op: Op::Conv2d,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_conv2d_bias_fwd",
        what: "conv2d_fwd_f32 with a per-output-channel bias added to the finished sum (the conv_bias contract)",
        reported: "native:conv2d_bias_fwd_f32",
        block_dim: 256,
        tile: (64, 128),
        shared_bytes: (2 * 16 * (64 + 4) + 2 * 16 * 128) * 4,
        programmatic: false,
        src: include_str!("../cu/conv2d_f32.cu"),
    },
    CudaKernel {
        name: "conv2d_fwd3x3_f32",
        // Redirected to by name from the WGSL `conv2d` for a 3x3, stride-1,
        // pad-1 conv at least 64 channels wide; every other shape keeps
        // `conv2d_fwd_f32`.
        op: Op::Conv2d,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_conv2d_fwd3x3",
        what: "dense fp32 3x3 stride-1 pad-1 conv2d forward as a direct convolution: 64 channels x a 16x16 patch per block, the input slice staged once with its halo, 8x8 register block, bit-identical to conv2d_fwd_f32",
        reported: "native:conv2d_fwd3x3_f32",
        block_dim: 256,
        // 64 output channels x 256 output positions (a 16 x 16 patch).
        tile: (64, 256),
        // Two stages of a 4 x 18 x 20 input slice and 36 x 68 weights.
        shared_bytes: 2 * (4 * 18 * 20 + 4 * 9 * 68) * 4,
        programmatic: false,
        src: include_str!("../cu/conv2d_f32.cu"),
    },
    CudaKernel {
        name: "conv2d_bias_fwd3x3_f32",
        // Redirected to by name from the WGSL `conv_bias` and `conv_bias_reg`
        // at the shapes `conv2d_fwd3x3_f32` serves.
        op: Op::Conv2d,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_conv2d_bias_fwd3x3",
        what: "conv2d_fwd3x3_f32 with a per-output-channel bias added to the finished sum (the conv_bias contract)",
        reported: "native:conv2d_bias_fwd3x3_f32",
        block_dim: 256,
        tile: (64, 256),
        shared_bytes: 2 * (4 * 18 * 20 + 4 * 9 * 68) * 4,
        programmatic: false,
        src: include_str!("../cu/conv2d_f32.cu"),
    },
    CudaKernel {
        name: "conv2d_dx_f32",
        op: Op::Conv2dBackward,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_conv2d_dx",
        what: "dense fp32 conv2d input gradient as an implicit GEMM per stride class (only the live taps of each (h mod s, w mod s) class are gathered); the forward's tile and register block",
        reported: "native:conv2d_dx_f32",
        block_dim: 256,
        // Up to 64 input channels x 128 positions of one stride class.
        tile: (64, 128),
        shared_bytes: (2 * 16 * (64 + 4) + 2 * 16 * 128) * 4,
        programmatic: false,
        src: include_str!("../cu/conv2d_f32.cu"),
    },
    CudaKernel {
        name: "conv2d_dw_partial_f32",
        op: Op::Conv2dBackward,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_conv2d_dw_partial",
        what: "dense fp32 conv2d weight gradient as an implicit GEMM over a slice of the output positions; 16/32/64 x 64 tile, 32 positions staged per iteration, partial sums to scratch (or straight into dw for one slice)",
        reported: "native:conv2d_dw_partial_f32",
        block_dim: 256,
        // Up to 64 output channels x 64 weight columns (ci, kh, kw).
        tile: (64, 64),
        // Two A slices of 32 x (64 + 4) and two B slices of 32 x (64 + 4) floats.
        shared_bytes: (2 * 32 * (64 + 4) + 2 * 32 * (64 + 4)) * 4,
        programmatic: false,
        src: include_str!("../cu/conv2d_f32.cu"),
    },
    CudaKernel {
        name: "conv2d_dw_reduce_f32",
        op: Op::Conv2dBackward,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_conv2d_dw_reduce",
        what: "folds the weight gradient's per-slice partial sums into dw in ascending slice order (deterministic, no atomics)",
        reported: "native:conv2d_dw_reduce_f32",
        block_dim: 256,
        tile: (1, 256),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/conv2d_f32.cu"),
    },
    CudaKernel {
        name: "bn_stats_f32",
        // Redirected to by name from the WGSL `bn_stats` (`gpu_core::native_upgrade`).
        op: Op::BatchNormReduce,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_bn_stats",
        what: "BatchNorm batch mean and population variance of an NCHW channel: one 512-thread block per channel, coalesced 16-byte loads, two passes, fixed-order tree reduction",
        reported: "native:bn_stats_f32",
        block_dim: 512,
        // One block per channel.
        tile: (1, 1),
        // The warp partials and the block total, two floats each.
        shared_bytes: (512 / 32 + 1) * 8,
        programmatic: false,
        src: include_str!("../cu/bn_f32.cu"),
    },
    CudaKernel {
        name: "bn_dstats_f32",
        op: Op::BatchNormReduce,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_bn_dstats",
        what: "BatchNorm backward sums of an NCHW channel (sum dy, sum dy*xhat) into the packed bp layout: one 512-thread block per channel",
        reported: "native:bn_dstats_f32",
        block_dim: 512,
        tile: (1, 1),
        shared_bytes: (512 / 32 + 1) * 8,
        programmatic: false,
        src: include_str!("../cu/bn_f32.cu"),
    },
    CudaKernel {
        name: "bn_dgamma_f32",
        op: Op::BatchNormReduce,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_bn_dgamma",
        what: "BatchNorm gamma gradient (dgamma += sum dy*xhat) of an NCHW channel: one 512-thread block per channel",
        reported: "native:bn_dgamma_f32",
        block_dim: 512,
        tile: (1, 1),
        shared_bytes: (512 / 32 + 1) * 8,
        programmatic: false,
        src: include_str!("../cu/bn_f32.cu"),
    },
    CudaKernel {
        name: "bn_train_f32",
        op: Op::BatchNormReduce,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_bn_train",
        what: "BatchNorm train-mode normalisation of an NCHW map: a block per run of one (n, c) plane, channel constants once per thread, 16-byte accesses; bit-identical to bn_train",
        reported: "native:bn_train_f32",
        block_dim: 256,
        // 1024 elements of one plane per block.
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/bn_elem_f32.cu"),
    },
    CudaKernel {
        name: "gn_apply_f32",
        // Redirected to by name from the WGSL `gn_apply` (`gpu_core::native_upgrade`).
        op: Op::Elementwise,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_gn_apply",
        what: "GroupNorm apply over an NCHW map from per-(n, group) mean and rstd: a block per run of one (n, c) plane, its group decoded once, 16-byte accesses; bit-identical to gn_apply",
        reported: "native:gn_apply_f32",
        block_dim: 256,
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/bn_elem_f32.cu"),
    },
    CudaKernel {
        name: "bn_dx_f32",
        op: Op::BatchNormReduce,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_bn_dx",
        what: "BatchNorm input gradient of an NCHW map from bn_dstats' packed sums: a block per run of one (n, c) plane; bit-identical to bn_dx",
        reported: "native:bn_dx_f32",
        block_dim: 256,
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/bn_elem_f32.cu"),
    },
    CudaKernel {
        name: "concat_split_f32",
        op: Op::Elementwise,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_concat_split",
        what: "a channel window of an NCHW map, copied plane by plane (a block per run of one plane, 16-byte accesses); bit-identical to concat_split",
        reported: "native:concat_split_f32",
        block_dim: 256,
        // 1024 elements of one plane per block.
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: true,
        src: include_str!("../cu/plumb_f32.cu"),
    },
    CudaKernel {
        name: "chan_place_f32",
        op: Op::Elementwise,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_chan_place",
        what: "an NCHW map written into a channel window of a wider one, plane by plane; bit-identical to chan_place",
        reported: "native:chan_place_f32",
        block_dim: 256,
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/plumb_f32.cu"),
    },
    CudaKernel {
        name: "concat2_f32",
        op: Op::Elementwise,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_concat2",
        what: "two NCHW maps concatenated along channels, plane by plane; bit-identical to concat2",
        reported: "native:concat2_f32",
        block_dim: 256,
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/plumb_f32.cu"),
    },
    CudaKernel {
        name: "bias_add_f32",
        op: Op::Elementwise,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_bias_add",
        what: "bias add in place over rows of a width that is a whole number of vectors, four elements a thread with 16-byte accesses; bit-identical to bias_add",
        reported: "native:bias_add_f32",
        block_dim: 256,
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/elementwise_f32.cu"),
    },
    CudaKernel {
        name: "add2_f32",
        op: Op::Elementwise,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_add2",
        what: "a + b over a flat array, four elements a thread with 16-byte accesses; bit-identical to add2",
        reported: "native:add2_f32",
        block_dim: 256,
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/elementwise_f32.cu"),
    },
    CudaKernel {
        name: "gelu_erf_f32",
        op: Op::Elementwise,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_gelu_erf",
        what: "exact erf GELU over a flat array, four elements a thread with 16-byte accesses; bit-identical to gelu_erf",
        reported: "native:gelu_erf_f32",
        block_dim: 256,
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/elementwise_f32.cu"),
    },
    CudaKernel {
        name: "silu_f32",
        op: Op::Elementwise,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_silu_fwd",
        what: "SiLU over a flat array, four elements a thread with 16-byte accesses; bit-identical to silu",
        reported: "native:silu_f32",
        block_dim: 256,
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/plumb_f32.cu"),
    },
    CudaKernel {
        name: "silu_bwd_f32",
        op: Op::Elementwise,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_silu_bwd",
        what: "SiLU's input gradient over a flat array, four elements a thread with 16-byte accesses; bit-identical to silu_bwd",
        reported: "native:silu_bwd_f32",
        block_dim: 256,
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/plumb_f32.cu"),
    },
    CudaKernel {
        name: "bn_dbeta_f32",
        op: Op::BatchNormReduce,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_bn_dbeta",
        what: "BatchNorm beta gradient (dbeta += sum dy) of an NCHW channel: one 512-thread block per channel",
        reported: "native:bn_dbeta_f32",
        block_dim: 512,
        tile: (1, 1),
        shared_bytes: (512 / 32 + 1) * 8,
        programmatic: false,
        src: include_str!("../cu/bn_f32.cu"),
    },
    CudaKernel {
        name: "cast_f32_bf16",
        op: Op::Elementwise,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_cast_f32_bf16",
        what: "fp32 to packed bf16, round-to-nearest-even, four elements a thread with 16-byte loads: the activation operand of a tensor-core GEMM",
        reported: "native:cast_f32_bf16",
        block_dim: 256,
        // 1024 elements per block.
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/cast_f32_bf16.cu"),
    },
    CudaKernel {
        name: "weight_dequant_bf16",
        op: Op::Elementwise,
        weight: Dtype::BF16,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_weight_dequant_bf16",
        what: "a window of rows of any packed weight tier (bf16, f16, int8, int4, NF4, FP8 E4M3/E5M2, NVFP4, MXFP4, MXFP8, Q4_K/Q5_K/Q6_K) decoded to a bf16 scratch for a tensor-core GEMM, a thread per eight weights",
        reported: "native:weight_dequant_bf16",
        block_dim: 256,
        // 256 threads of eight weights (2048 weights) per block.
        tile: (1, 256),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/weight_dequant_bf16.cu"),
    },
    CudaKernel {
        name: "gdn_chunk_loop_train_f32",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        // Asked for by name: the training forward is the same operator and tier as
        // the inference loop, which `find` keeps resolving.
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_gdn_chunk_loop_train_f32",
        what: "gdn_chunk_loop_f32 plus the training histories (the state entering every chunk and each row's decay scale) the backward sweep reads, in the same single launch",
        reported: "native:gdn_chunk_loop_train_f32",
        block_dim: 128,
        tile: (64, 32),
        shared_bytes: (128 * 32 + 64 * 32 + 32 * 129) * 4,
        programmatic: false,
        src: include_str!("../cu/gdn_chunk_loop_f32.cu"),
    },
    CudaKernel {
        name: "gdn_bwd_sweep_f32",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        // Asked for by name: the backward of the recurrence is a different operator
        // from the loop `find` resolves.
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_gdn_bwd_sweep_f32",
        what: "the reverse sweep of Gated DeltaNet's across-chunk recurrence in one launch: one block per head keeps the running state gradient in shared memory across all chunks (dynamic shared memory, opted in)",
        reported: "native:gdn_bwd_sweep_f32",
        block_dim: 512,
        // One block per head; the whole 128 x 128 state gradient lives in it.
        tile: (64, 128),
        // Everything is dynamic: see `GDN_BWD_SWEEP_DYNAMIC_SHARED_BYTES`.
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/gdn_bwd_sweep_f32.cu"),
    },
    CudaKernel {
        name: "attn_softmax_fwd",
        op: Op::Softmax,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_attn_softmax_fwd",
        what: "causal softmax of one query block's fp32 scores into bf16 probabilities and the row log-sum-exp, a block per (head, row)",
        reported: "native:attn_softmax_fwd",
        block_dim: 256,
        // One block per row.
        tile: (1, 1),
        shared_bytes: 8 * 4,
        programmatic: false,
        src: include_str!("../cu/attn_softmax_f32.cu"),
    },
    CudaKernel {
        name: "attn_dscore",
        op: Op::Softmax,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_attn_dscore",
        what: "attention score gradient p * (dp - D) * scale and the rebuilt probabilities as bf16, from fp32 scores and the saved log-sum-exp",
        reported: "native:attn_dscore",
        block_dim: 256,
        // One block per row.
        tile: (1, 1),
        shared_bytes: 8 * 4,
        programmatic: false,
        src: include_str!("../cu/attn_softmax_f32.cu"),
    },
    CudaKernel {
        name: "attn_rowdot",
        op: Op::Softmax,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_attn_rowdot",
        what: "per-row dot product of two fp32 matrices (D = rowsum(dO * O) of attention's backward), a block per row",
        reported: "native:attn_rowdot",
        block_dim: 256,
        // One block per row.
        tile: (1, 1),
        shared_bytes: 8 * 4,
        programmatic: false,
        src: include_str!("../cu/attn_softmax_f32.cu"),
    },
    CudaKernel {
        name: "ce_stats_f32",
        op: Op::Softmax,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_ce_stats",
        what: "per-row max and sum of exp(logit - max) for the cross-entropy backward, a 256-thread block per row instead of a thread per row; the sum is reassociated",
        reported: "native:ce_stats_f32",
        block_dim: 256,
        // One block per row.
        tile: (1, 1),
        shared_bytes: 8 * 4,
        programmatic: false,
        src: include_str!("../cu/ce_stats_f32.cu"),
    },
    CudaKernel {
        name: "gdn_state_decay_bwd_dscale_f32",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_gdn_state_decay_bwd_dscale",
        what: "the decay-scale gradient of Gated DeltaNet's state update (a dot product over a head's [dk, dv] state), a 256-thread block per head instead of one thread; the sum is reassociated",
        reported: "native:gdn_state_decay_bwd_dscale_f32",
        block_dim: 256,
        // One block per head.
        tile: (1, 1),
        shared_bytes: 8 * 4,
        programmatic: false,
        src: include_str!("../cu/gdn_state_decay_bwd_dscale_f32.cu"),
    },
    // ---- t-gdn: the tensor-core Gated DeltaNet chunk forward ----
    CudaKernel {
        name: "gdn_chunk_prep_tc",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        // Asked for by name: it replaces the model's whole-tensor prefix of the chunked forward,
        // not the recurrence `find` resolves for this operator.
        by_name: true,
        source: ImplSource::Tuned,
        // `mma.sync.m16n8k8` on TF32 operands.
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_gdn_chunk_prep_tc",
        what: "Gated DeltaNet's state-independent chunk forward (cumulative decay, decay mask, Gram product, unit-triangular solve, w, u and the intra-chunk scores) in one launch per (head, chunk): three-pass TF32 tensor-core products with fp32 accumulation, the solve in fp32 shared memory",
        reported: "native:gdn_chunk_prep_tc",
        block_dim: 256,
        // One block per (head, chunk) of up to 64 rows.
        tile: (64, 128),
        // Everything is dynamic: see `GDN_CHUNK_PREP_DYNAMIC_SHARED_BYTES` in gpu-core.
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/gdn_chunk_prep_tc.cu"),
    },
    CudaKernel {
        name: "gdn_chunk_prep_train_tc",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_gdn_chunk_prep_train_tc",
        what: "gdn_chunk_prep_tc plus the tensors the backward reads (k_beta, v_beta, k_beta_decay, the decay mask, raw_attn0, t_mat, raw_intra), in the same single launch",
        reported: "native:gdn_chunk_prep_train_tc",
        block_dim: 256,
        tile: (64, 128),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/gdn_chunk_prep_tc.cu"),
    },
    CudaKernel {
        name: "gdn_chunk_loop_tc",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        // Asked for by name, so `find` keeps resolving the fp32 recurrence for this operator.
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_gdn_chunk_loop_tc",
        what: "Gated DeltaNet's whole across-chunk recurrence in one launch on three-pass TF32 tensor cores (fp32 state and accumulation), head split over four 32-column blocks",
        reported: "native:gdn_chunk_loop_tc",
        block_dim: 128,
        tile: (64, 32),
        // Everything is dynamic: see `GDN_CHUNK_LOOP_TC_DYNAMIC_SHARED_BYTES` in gpu-core.
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/gdn_chunk_loop_tc.cu"),
    },
    CudaKernel {
        name: "gdn_chunk_loop_train_tc",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_gdn_chunk_loop_train_tc",
        what: "gdn_chunk_loop_tc plus the training histories (the state entering every chunk and each row's decay scale) the backward sweep reads, in the same single launch",
        reported: "native:gdn_chunk_loop_train_tc",
        block_dim: 128,
        tile: (64, 32),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/gdn_chunk_loop_tc.cu"),
    },
    CudaKernel {
        name: "gdn_chunk_prep_tok_train_tc",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_gdn_chunk_prep_tok_train_tc",
        what: "gdn_chunk_prep_train_tc reading the mixer's token-major tensors: folds the query/key L2 normalisation, the head repeat and the token-major to chunk-major permutes into the block's loads and writes the chunk-major query, key, value and beta the loop and the backward read",
        reported: "native:gdn_chunk_prep_tok_train_tc",
        block_dim: 256,
        tile: (64, 128),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/gdn_chunk_prep_tc.cu"),
    },
    CudaKernel {
        name: "gdn_chunk_prep_tok_tc",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_gdn_chunk_prep_tok_tc",
        what: "gdn_chunk_prep_tc reading the mixer's token-major tensors (query/key L2 normalisation, head repeat and chunk-major permutes folded into the loads); writes the chunk-major query and key the recurrence reads",
        reported: "native:gdn_chunk_prep_tok_tc",
        block_dim: 256,
        tile: (64, 128),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/gdn_chunk_prep_tc.cu"),
    },
    // ---- t-gdn2: the tensor-core Gated DeltaNet chunk backward ----
    CudaKernel {
        name: "gdn_bwd_sweep_tc",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_gdn_bwd_sweep_tc",
        what: "the reverse sweep of Gated DeltaNet's across-chunk recurrence in one launch on three-pass TF32 tensor cores, the head split over four 32-column blocks; writes d_v_new and the history of the state gradient the post kernel reads",
        reported: "native:gdn_bwd_sweep_tc",
        block_dim: 128,
        tile: (64, 32),
        // Everything is dynamic: see `GDN_BWD_SWEEP_TC_DYNAMIC_SHARED_BYTES` in gpu-core.
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/gdn_bwd_sweep_tc.cu"),
    },
    CudaKernel {
        name: "gdn_chunk_bwd_post_tc",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_gdn_chunk_bwd_post_tc",
        what: "everything of Gated DeltaNet's chunk backward that is independent across chunks (recomputed v_new, the output, w, u and inverse-adjoint products, the decay and cumulative-sum gradients, the row-scale gradients) in one launch per (head, chunk) on three-pass TF32 tensor cores",
        reported: "native:gdn_chunk_bwd_post_tc",
        block_dim: 256,
        tile: (64, 128),
        // Everything is dynamic: see `GDN_CHUNK_BWD_POST_TC_DYNAMIC_SHARED_BYTES` in gpu-core.
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/gdn_chunk_bwd_post_tc.cu"),
    },
    CudaKernel {
        name: "gdn_qk_bwd_tok",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_gdn_qk_bwd_tok",
        what: "the last leg of Gated DeltaNet's mixer backward for query and key in one launch: the chunk-major per-value-head gradients are summed over each key head's value heads, taken back through the L2 normalisation and stored token-major",
        reported: "native:gdn_qk_bwd_tok",
        block_dim: 256,
        tile: (8, 128),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/gdn_qk_bwd_tok.cu"),
    },
    CudaKernel {
        name: "gdn_conv_silu_tok",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_gdn_conv_silu_tok",
        what: "Gated DeltaNet's causal depthwise convolution (4 taps, optional history), its SiLU and the query/key/value split in one pass over the token-major rows: replaces two layout transposes, the generic convolution, the activation and three row splits",
        reported: "native:gdn_conv_silu_tok",
        block_dim: 128,
        tile: (32, 128),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/gdn_conv_tok.cu"),
    },
    CudaKernel {
        name: "gdn_conv_hist_tok",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_gdn_conv_hist_tok",
        what: "the convolution history a chunked forward hands to its next round: the last 3 rows of [history | x], channel-major, updated in place",
        reported: "native:gdn_conv_hist_tok",
        block_dim: 128,
        tile: (1, 128),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/gdn_conv_tok.cu"),
    },
    CudaKernel {
        name: "gdn_conv_silu_bwd_tok",
        op: Op::GatedDeltaChunkLoop,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: MMA_S8_MIN_CC,
        entry: "brain_gdn_conv_silu_bwd_tok",
        what: "the backward of gdn_conv_silu_tok for a frozen convolution weight in one pass: SiLU adjoint, the query/key/value gradient concatenation and the causal-adjoint convolution, token-major, with the history rows' gradient of a continued round",
        reported: "native:gdn_conv_silu_bwd_tok",
        block_dim: 128,
        tile: (32, 128),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/gdn_conv_tok.cu"),
    },
    // ---- t4: element-wise glue outside the mixer ----
    CudaKernel {
        name: "swiglu_bwd_fused",
        op: Op::Elementwise,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_swiglu_bwd_fused",
        what: "the SwiGLU backward in one pass: reads the gate pre-activation, the up projection and the incoming gradient once and writes both the gate and the up gradient, four elements a thread; replaces silu_bwd_da plus silu_bwd_db",
        reported: "native:swiglu_bwd_fused",
        block_dim: 256,
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/swiglu_bwd_fused.cu"),
    },
    CudaKernel {
        name: "add_rmsnorm_fwd",
        op: Op::Elementwise,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_add_rmsnorm_fwd",
        what: "a residual add folded into the RMSNorm that follows it, one block per row: writes both the sum and the normalised sum, replacing add2 plus rmsnorm_rows",
        reported: "native:add_rmsnorm_fwd",
        block_dim: 256,
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/rmsnorm_fused.cu"),
    },
    CudaKernel {
        name: "rmsnorm_dx_add",
        op: Op::Elementwise,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_rmsnorm_dx_add",
        what: "the RMSNorm input gradient with the residual-path gradient added in the same pass, one block per row; replaces rmsnorm_dx_rows plus add2 when the norm weight is frozen",
        reported: "native:rmsnorm_dx_add",
        block_dim: 256,
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/rmsnorm_fused.cu"),
    },
    CudaKernel {
        name: "silu_mul_bf16",
        op: Op::Elementwise,
        weight: Dtype::F32,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_silu_mul_bf16",
        what: "silu(gate) * up written as fp32 and, in the same pass, as the packed bf16 operand of the down projection's tensor-core GEMM: replaces silu_mul plus a cast_f32_bf16 re-read of the product",
        reported: "native:silu_mul_bf16",
        block_dim: 256,
        // 1024 elements per block.
        tile: (1, 1024),
        shared_bytes: 0,
        programmatic: false,
        src: include_str!("../cu/silu_mul_bf16.cu"),
    },
];

/// The kernel `table` offers for `op` over `weight` storage on a device of
/// compute capability `cc`: the eligible entry with the HIGHEST floor, so an
/// architecture-specialised kernel beats a generic one on a device that can
/// run both, and the generic one still serves a device that cannot.
///
/// `None` means this table offers nothing for that operator on that device -
/// the caller falls back to a generated or portable implementation and, per
/// the tier policy, must say so.
///
/// Takes the table as a parameter rather than reading [`ALL`] directly: the
/// selection RULE is the thing worth testing, and a test that can only feed
/// it the shipped table can only test it once.
pub fn best_for(table: &'static [CudaKernel], op: Op, weight: Dtype, cc: Cc) -> Option<&'static CudaKernel> {
    table.iter().filter(|k| !k.by_name && k.op == op && k.weight == weight && k.min_cc <= cc).max_by_key(|k| k.min_cc)
}

/// [`best_for`] for a caller with no weight tier to state: the highest-floor
/// entry for `op` over ANY tier. For ledger checks only (the tier policy has
/// no dtype axis yet); a dispatcher must always say which tier it binds.
pub fn best_for_any_tier(table: &'static [CudaKernel], op: Op, cc: Cc) -> Option<&'static CudaKernel> {
    table.iter().filter(|k| !k.by_name && k.op == op && k.min_cc <= cc).max_by_key(|k| k.min_cc)
}

/// [`best_for`] over the shipped [`ALL`] table.
pub fn find(op: Op, weight: Dtype, cc: Cc) -> Option<&'static CudaKernel> {
    best_for(ALL, op, weight, cc)
}

/// Look a kernel up by registry name.
pub fn get(name: &str) -> Option<&'static CudaKernel> {
    ALL.iter().find(|k| k.name == name)
}

/// `src` with `//` comments removed - what the instruction scans in
/// [`check_table`] must look at.
///
/// A kernel's header PROSE is the natural place to say which instructions it
/// uses and which it deliberately avoids, and a scan over raw text reads
/// those sentences as if they were code: a kernel whose header explains that
/// it carries no aliasing promise gets failed for the word appearing in the
/// explanation. That is the identical mistake `backend_api::
/// workgroup_size_of` documents having made against `@workgroup_size` in
/// WGSL headers, with the same fix - scan the code, not the comments.
///
/// Line comments only. Every kernel in this tree writes its header as `//`
/// lines, and stripping `/* */` correctly needs a real lexer (string
/// literals, nesting) for no gain against source nobody writes that way; a
/// block comment that mentions an instruction is therefore still read as
/// code, which fails LOUDLY and is fixed by rewording, never silently.
fn code_of(src: &str) -> String {
    src.lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n")
}

/// Every invariant this registry's entries must satisfy, checked as data
/// rather than asserted per-entry: unique names, a tier that means what it
/// says, an entry point that exists in the source, and a declared capability
/// floor consistent with the instructions the source actually uses.
///
/// Returns every violation found, not just the first - one run tells you
/// everything to fix. Used by this crate's own test over [`ALL`] and by the
/// catalogue gate; a future NVRTC compile check is an ADDITION to this, not
/// a replacement (it needs a toolkit, this needs nothing).
pub fn check_table(table: &[CudaKernel]) -> Vec<String> {
    let mut errs = Vec::new();
    for (i, k) in table.iter().enumerate() {
        // Instruction scans read the CODE, never the header prose that
        // explains which instructions the kernel uses - see `code_of`.
        let code = code_of(k.src);
        if table.iter().take(i).any(|p| p.name == k.name) {
            errs.push(format!("{}: duplicate kernel name", k.name));
        }
        if k.source != ImplSource::Tuned {
            errs.push(format!(
                "{}: declares tier {:?}; a kernel with source text in this tree is native, \
                 and a generated kernel has no file to list",
                k.name, k.source
            ));
        }
        if k.name.is_empty() || !k.name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
            errs.push(format!("{}: registry names are lowercase ascii/digits/underscore", k.name));
        }
        if k.reported != format!("native:{}", k.name) {
            errs.push(format!(
                "{}: reports itself as {:?}; a dispatch record must name it \"native:{}\" so it cannot \
                 be read as a WGSL catalogue kernel",
                k.name, k.reported, k.name
            ));
        }
        if k.what.trim().is_empty() {
            errs.push(format!("{}: no @what line - the catalogue row would be blank", k.name));
        }
        if !k.src.contains(k.entry) {
            errs.push(format!("{}: declared entry point {:?} does not appear in its source", k.name, k.entry));
        }
        if code.contains("__dp4a") && k.min_cc < DP4A_MIN_CC {
            errs.push(format!(
                "{}: uses __dp4a but declares min_cc {}.{}, below the capability that introduced it ({}.{})",
                k.name, k.min_cc.0, k.min_cc.1, DP4A_MIN_CC.0, DP4A_MIN_CC.1
            ));
        }
        // A kernel may run on older parts too if it gates those instructions behind the capability itself
        // (`__CUDA_ARCH__ >= 800`) and carries a body for the rest.
        let gates_mma_itself = code.contains("__CUDA_ARCH__ >= 800");
        if (code.contains("mma.sync") || code.contains("cp.async")) && k.min_cc < MMA_S8_MIN_CC && !gates_mma_itself {
            errs.push(format!(
                "{}: issues mma.sync/cp.async but declares min_cc {}.{}, below the capability that introduced them ({}.{})",
                k.name, k.min_cc.0, k.min_cc.1, MMA_S8_MIN_CC.0, MMA_S8_MIN_CC.1
            ));
        }
        if k.block_dim == 0 || k.block_dim % 32 != 0 {
            errs.push(format!(
                "{}: block_dim {} is not a non-zero multiple of the warp granularity every CUDA \
                 capability schedules in - a partial warp wastes lanes at every launch",
                k.name, k.block_dim
            ));
        }
        if k.tile.0 == 0 || k.tile.1 == 0 {
            errs.push(format!("{}: a tile of {:?} covers no output, so no block count can be derived", k.name, k.tile));
        }
        if k.shared_bytes > PORTABLE_SHARED_BYTES {
            errs.push(format!(
                "{}: declares {} bytes of __shared__, above the {PORTABLE_SHARED_BYTES} every CUDA \
                 capability guarantees per block - a card that grants more must be QUERIED, never assumed",
                k.name, k.shared_bytes
            ));
        }
        if code.contains("__restrict__") {
            errs.push(format!(
                "{}: uses __restrict__. brain's device buffers alias BY DESIGN (a sliced step binds ranges \
                 of one buffer), so a no-alias promise here is a silent wrong-answer hazard unless the \
                 kernel's own call sites are proven disjoint - state that proof next to the kernel and \
                 exempt it deliberately, never by default",
                k.name
            ));
        }
    }
    errs
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_gemv_source_and_the_rust_constant_agree_on_where_the_tensor_core_block_starts() {
        let src = get("matmul_i8_gemv").expect("registered").src;
        assert!(
            src.contains(&format!("#define BRAIN_I8G_MMA_MIN_ROWS {I8_GEMV_MMA_MIN_ROWS}\n")),
            "matmul_i8_gemv.cu must define BRAIN_I8G_MMA_MIN_ROWS as {I8_GEMV_MMA_MIN_ROWS}"
        );
    }

    use super::*;

    /// The shipped table obeys its own invariants.
    #[test]
    fn the_shipped_registry_is_well_formed() {
        let errs = check_table(ALL);
        assert!(errs.is_empty(), "kernels-cuda registry violations:\n  {}", errs.join("\n  "));
        assert!(!ALL.is_empty(), "the registry ships at least one native kernel");
    }

    /// A block count must COVER the output, at shapes that divide the tile
    /// and at shapes that do not - an under-count leaves real output
    /// elements never written, which is silent corruption rather than a
    /// crash (the same failure mode the WGSL thread-count formula's own
    /// regression test exists for).
    #[test]
    fn the_block_count_covers_every_output_element() {
        for k in ALL {
            for (m, n) in [(1u32, 1u32), (64, 64), (65, 64), (64, 65), (300, 260), (37, 53), (513, 257)] {
                let blocks = k.blocks_for(m, n);
                assert!(
                    (blocks as u64) * (k.tile.0 as u64) * (k.tile.1 as u64) >= (m as u64) * (n as u64),
                    "{}: {blocks} blocks of {:?} do not cover a {m}x{n} output",
                    k.name,
                    k.tile
                );
            }
        }
    }

    const GENERIC: &str = "extern \"C\" __global__ void bk_matmul_generic(const float* a) {}";
    const TUNED: &str = "extern \"C\" __global__ void bk_matmul_dp4a(const int* a) { __dp4a(0, 0, 0); }";

    static FIXTURE: &[CudaKernel] = &[
        CudaKernel {
            name: "matmul_generic",
            op: Op::MatMul,
            weight: Dtype::F32,
            by_name: false,
            source: ImplSource::Tuned,
            min_cc: (5, 0),
            entry: "bk_matmul_generic",
            what: "generic fp32 GEMM",
            reported: "native:matmul_generic",
            block_dim: 64,
            tile: (1, 64),
            shared_bytes: 0,
            programmatic: false,
            src: GENERIC,
        },
        CudaKernel {
            name: "matmul_dp4a",
            op: Op::MatMul,
            weight: Dtype::F32,
            by_name: false,
            source: ImplSource::Tuned,
            min_cc: DP4A_MIN_CC,
            entry: "bk_matmul_dp4a",
            what: "packed-int8 GEMM over the four-way dot product",
            reported: "native:matmul_dp4a",
            block_dim: 128,
            tile: (32, 32),
            shared_bytes: 1024,
            programmatic: false,
            src: TUNED,
        },
    ];

    /// The resolution rule: highest declared floor at or below the device's
    /// queried capability. Asserted at a capability BELOW both floors, at
    /// one that admits only the generic kernel, and at one far above both -
    /// the last standing in for any future architecture, which must keep
    /// getting the most specialised kernel rather than nothing.
    #[test]
    fn the_highest_eligible_floor_wins_at_any_capability() {
        assert!(best_for(FIXTURE, Op::MatMul, Dtype::F32, (3, 5)).is_none());
        assert_eq!(best_for(FIXTURE, Op::MatMul, Dtype::F32, (6, 0)).unwrap().name, "matmul_generic");
        assert_eq!(best_for(FIXTURE, Op::MatMul, Dtype::F32, DP4A_MIN_CC).unwrap().name, "matmul_dp4a");
        assert_eq!(best_for(FIXTURE, Op::MatMul, Dtype::F32, (12, 0)).unwrap().name, "matmul_dp4a");
        // An operator the table says nothing about resolves to nothing, on
        // every device - never to "the closest thing available".
        assert!(best_for(FIXTURE, Op::RmsNorm, Dtype::F32, (12, 0)).is_none());
    }

    /// The weight tier is part of an implementation's identity: the int8 GEMV
    /// has a HIGHER floor than the fp32 matmul for the same operator, so
    /// resolving by operator alone would hand the fp32 provider a kernel that
    /// reads packed int8 - a wrong answer, not a crash.
    #[test]
    fn the_weight_tier_selects_the_kernel_not_just_the_operator() {
        let f32_k = find(Op::MatMul, Dtype::F32, (9, 0)).expect("fp32 matmul ships");
        let i8_k = find(Op::MatMul, Dtype::I8, (9, 0)).expect("int8 gemv ships");
        assert_eq!(f32_k.name, "matmul_f32_tiled");
        assert_eq!(i8_k.name, "matmul_i8_gemv");
        // Below the packed dot product's floor the int8 kernel is not offered
        // at all, while fp32 still is.
        assert!(find(Op::MatMul, Dtype::I8, (6, 0)).is_none());
        assert!(find(Op::MatMul, Dtype::F32, (6, 0)).is_some());
        // The q4 GEMV is its own entry, never the int8 one.
        assert_eq!(find(Op::MatMul, Dtype::Q4, (9, 0)).map(|k| k.name), Some("matmul_q4_gemv"));
        // A tier nothing ships resolves to nothing.
        assert!(find(Op::MatMul, Dtype::NF4, (9, 0)).is_none());
        // The tier-blind lookup the policy ledger uses sees the higher floor
        // (which of the packed-dot GEMVs sharing it answers is table order).
        assert_eq!(best_for_any_tier(ALL, Op::MatMul, (9, 0)).map(|k| k.min_cc), Some(DP4A_MIN_CC));
    }

    /// A kernel's header explaining which instructions it avoids is PROSE,
    /// not a use of them. Before this was fixed, the shipped kernel failed
    /// its own registry check for saying in its header that it carries no
    /// aliasing promise - the scan could not tell a sentence from a
    /// declaration, exactly as the WGSL work-group-size scan once could not.
    #[test]
    fn an_instruction_named_only_in_a_comment_is_not_a_use_of_it() {
        static PROSE: &[CudaKernel] = &[CudaKernel {
            name: "prose_only",
            op: Op::MatMul,
            weight: Dtype::F32,
            by_name: false,
            source: ImplSource::Tuned,
            // Below DP4A's floor on purpose: the header mentions the
            // instruction, the body does not use it, and only the body counts.
            min_cc: BASELINE_MIN_CC,
            entry: "bk_prose",
            what: "mentions __dp4a and __restrict__ in its header and uses neither",
            reported: "native:prose_only",
            block_dim: 64,
            tile: (1, 64),
            shared_bytes: 0,
            programmatic: false,
            src: "// no __restrict__ here, and no __dp4a either\n\
                  extern \"C\" __global__ void bk_prose(const float* a) {}\n",
        }];
        assert!(check_table(PROSE).is_empty(), "{:?}", check_table(PROSE));
    }

    /// A tensor-core instruction below its capability floor is a violation unless the kernel gates it on
    /// the capability itself and carries a body for the older parts (as `matmul_i8_gemv` does).
    #[test]
    fn a_kernel_may_declare_an_older_floor_only_if_it_gates_its_tensor_core_instructions() {
        const UNGATED: &str = "extern \"C\" __global__ void bk_mma(const int* a) { asm volatile(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {}\"); }";
        const GATED: &str = "#if __CUDA_ARCH__ >= 800\nextern \"C\" __global__ void bk_mma(const int* a) { asm volatile(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {}\"); }\n#else\nextern \"C\" __global__ void bk_mma(const int* a) {}\n#endif\n";
        let kernel = |src: &'static str| CudaKernel {
            name: "mma_probe",
            op: Op::MatMul,
            weight: Dtype::I8,
            by_name: false,
            source: ImplSource::Tuned,
            min_cc: DP4A_MIN_CC,
            entry: "bk_mma",
            what: "probe",
            reported: "native:mma_probe",
            block_dim: 64,
            tile: (1, 64),
            shared_bytes: 0,
            programmatic: false,
            src,
        };
        let joined = check_table(&[kernel(UNGATED)]).join("\n");
        assert!(joined.contains("issues mma.sync/cp.async but declares min_cc"), "{joined}");
        assert!(check_table(&[kernel(GATED)]).is_empty(), "{:?}", check_table(&[kernel(GATED)]));
    }

    #[test]
    fn the_invariants_catch_a_mis_declared_entry() {
        static BAD: &[CudaKernel] = &[
            CudaKernel {
                name: "matmul_dp4a",
                op: Op::MatMul,
                weight: Dtype::F32,
                by_name: false,
                source: ImplSource::Generated,
                min_cc: (5, 0),
                entry: "bk_missing",
                what: "",
                reported: "matmul_dp4a",
                block_dim: 33,
                tile: (0, 8),
                shared_bytes: PORTABLE_SHARED_BYTES + 1,
                programmatic: false,
                src: TUNED,
            },
            CudaKernel {
                name: "matmul_dp4a",
                op: Op::MatMul,
                weight: Dtype::F32,
                by_name: false,
                source: ImplSource::Tuned,
                min_cc: (5, 0),
                entry: "bk_matmul_dp4a",
                what: "duplicate name",
                reported: "native:matmul_dp4a",
                block_dim: 64,
                tile: (16, 16),
                shared_bytes: 0,
                programmatic: false,
                src: TUNED,
            },
        ];
        let errs = check_table(BAD);
        let joined = errs.join("\n");
        assert!(joined.contains("duplicate kernel name"), "{joined}");
        assert!(joined.contains("declares tier Generated"), "{joined}");
        assert!(joined.contains("does not appear in its source"), "{joined}");
        assert!(joined.contains("no @what line"), "{joined}");
        assert!(joined.contains("below the capability that introduced it"), "{joined}");
        assert!(joined.contains("is not a non-zero multiple of the warp granularity"), "{joined}");
        assert!(joined.contains("covers no output"), "{joined}");
        assert!(joined.contains("bytes of __shared__, above the"), "{joined}");
        assert!(joined.contains("reports itself as"), "{joined}");
    }
}
