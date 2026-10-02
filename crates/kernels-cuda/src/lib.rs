// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Hand-written CUDA C++ kernels and the metadata that says what each one
//! is - a registry of brain's NATIVE kernels, entirely separate from the
//! portable WGSL catalogue in `kernels`.
//!
//! Swedish Embedded AB implements native GPU kernel libraries and the
//! metadata discipline that keeps them honest. If your team needs expertise
//! in hand-written CUDA kernels that stay checkable rather than becoming
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

/// One hand-written CUDA kernel: its source text plus the metadata that
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
    /// here: a kernel with a `.cu` file in this tree is hand-written by
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

/// Every hand-written CUDA kernel brain ships.
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
        what: "int8 tensor-core GEMM (mma.sync m16n8k32), dynamic per-token activation scale, group-32 weight scale folded per MMA",
        reported: "native:matmul_i8_mma",
        block_dim: 128,
        // 64 activation rows x 64 weight rows per block (four warps of 32x32);
        // the kernel numbers its blocks row-tile-fastest.
        tile: (64, 64),
        // 4 stages x (64x64 + 64x64 int8 tiles + 64x2 f32 group scales).
        shared_bytes: 4 * (64 * 64 + 64 * 64 + 64 * 2 * 4),
        src: include_str!("../cu/matmul_i8_mma.cu"),
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
        src: include_str!("../cu/flash_prefill_f16_hd256.cu"),
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
        what: "packed-int8 skinny-M GEMV (up to 8 rows of x per weight pass); 16 B weight loads, dp4a, bit-identical to matmul_i8_gemv_reg",
        reported: "native:matmul_i8_gemv",
        block_dim: 128,
        // 8 rows of x by 8 weight rows per block.
        tile: (8, 8),
        shared_bytes: 0,
        src: include_str!("../cu/matmul_i8_gemv.cu"),
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
        tile: (8, 8),
        shared_bytes: 0,
        src: include_str!("../cu/matmul_i8_gemv.cu"),
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
        src: include_str!("../cu/moe_i8_grouped_mma.cu"),
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
        what: "residual add + RMSNorm + per-row int8 scale + pack in one launch; 64 threads per row, bit-identical to add2 + rmsnorm_rows + max_abs_rows + quant_pack",
        reported: "native:add_rms_quant",
        block_dim: 64,
        // One block per row of x.
        tile: (1, 1),
        // 64 lane partials + the 64 x 128 quantised bytes exchanged between lanes.
        shared_bytes: 64 * 4 + 64 * 128,
        src: include_str!("../cu/add_rms_quant.cu"),
    },
    CudaKernel {
        name: "quant_epilogue",
        // A fused kernel asked for by name, like `add_rms_quant`: the operator
        // and tier say what it produces (per-row scales and a packed int8
        // activation), and a fused kernel is always looked up through `get`,
        // never through `find`, which is the matmul providers' selector.
        op: Op::MaxAbsRow,
        weight: Dtype::I8,
        by_name: true,
        source: ImplSource::Tuned,
        min_cc: BASELINE_MIN_CC,
        entry: "brain_quant_epilogue",
        what: "silu_mul / sigmoid-gate / plain producer + per-row int8 scale + pack in one launch; 1024 threads per row, bit-identical to the chain it replaces",
        reported: "native:quant_epilogue",
        block_dim: 1024,
        tile: (1, 1),
        // 32 warp maxima + 1024 x 18 quantised bytes exchanged between threads.
        shared_bytes: 32 * 4 + 1024 * 18,
        src: include_str!("../cu/quant_epilogue.cu"),
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
        src: include_str!("../cu/gqa_decode_prep.cu"),
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
                "{}: declares tier {:?}; a kernel with source text in this tree is hand-written, \
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
        if (code.contains("mma.sync") || code.contains("cp.async")) && k.min_cc < MMA_S8_MIN_CC {
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
    use super::*;

    /// The shipped table obeys its own invariants.
    #[test]
    fn the_shipped_registry_is_well_formed() {
        let errs = check_table(ALL);
        assert!(errs.is_empty(), "kernels-cuda registry violations:\n  {}", errs.join("\n  "));
        assert!(!ALL.is_empty(), "the registry ships at least one hand-written kernel");
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
        // A tier nothing ships resolves to nothing.
        assert!(find(Op::MatMul, Dtype::Q4, (9, 0)).is_none());
        // The tier-blind lookup the policy ledger uses sees the higher floor.
        assert_eq!(best_for_any_tier(ALL, Op::MatMul, (9, 0)).map(|k| k.name), Some("matmul_i8_gemv"));
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
            src: "// no __restrict__ here, and no __dp4a either\n\
                  extern \"C\" __global__ void bk_prose(const float* a) {}\n",
        }];
        assert!(check_table(PROSE).is_empty(), "{:?}", check_table(PROSE));
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
