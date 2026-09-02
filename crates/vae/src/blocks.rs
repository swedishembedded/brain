// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The shared convolutional-autoencoder block builder.
//!
//! This is the single implementation of the conv / GroupNorm / SiLU / residual
//! / nearest-upsample / single-head-self-attention graph that both
//! `AutoencoderKL` (diffusers: Z-Image, FLUX.2, HiDream) and the VQGAN family
//! (`crates/vqgan`: CodeFormer) are built from — the two architectures differ
//! only in their **block schedule** and their **tensor names**, not in the
//! blocks themselves.
//!
//! What is parameterised:
//!
//! * [`BlockNames`] — the per-architecture leaf names. diffusers calls a
//!   resnet's projection shortcut `conv_shortcut` and an attention's
//!   projections `to_q/to_k/to_v/to_out.0` over a `group_norm`; VQGAN calls
//!   them `conv_out` and `q/k/v/proj_out` over a `norm`.
//! * `taps_on` — record every block output for parity debugging. Taps pin
//!   buffers, so recording them disables the activation pool.
//!
//! What is NOT parameterised (identical in both): GroupNorm groups/eps come in
//! as constructor arguments; the attention is single-head with `head_dim = C`,
//! scale `C^-0.5`, softmax over the key axis and the residual added to the
//! **pre-norm** input; the strided downsample reproduces the reference's
//! asymmetric `F.pad(x,(0,1,0,1))`.
//!
//! Callers own the block schedule: `crate::decoder` walks the diffusers
//! down/mid/up schedule, `vqgan::model` walks the reference's flat
//! `nn.ModuleList`. Neither owns a copy of a block.

use gpu_core::select::{DefaultSelector, KernelSelector, KernelVariant, Op as SelectOp, OpShape};
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use std::collections::HashMap;

pub mod grad;
pub mod skipfuse;

// Kernel-table indices (order matches KERNELS).
const K_CONV: usize = 0;
const K_GN_APPLY: usize = 1;
const K_SILU: usize = 2;
const K_ADD2: usize = 3;
const K_UPSAMPLE2: usize = 4;
const K_NCHW_NLC: usize = 5;
const K_NLC_NCHW: usize = 6;
const K_ATTN_SCORES: usize = 7;
const K_ATTN_SOFTMAX: usize = 8;
const K_ATTN_APPLY: usize = 9;
const K_GN_STATS_WG: usize = 10;
const K_MATMUL: usize = 11;
const K_IM2COL_AT: usize = 12;
const K_NLC_BIAS_NCHW: usize = 13;
const K_GN_PART: usize = 14;
const K_GN_STATS2: usize = 15;
const K_CONCAT2: usize = 16;

/// Partials per group for the two-stage GroupNorm reduction. 64 is the value
/// `crates/diamond` arrived at, and the measurement below was taken at it.
pub(crate) const GN_P: u32 = 64;

/// The `add2` slot inside [`KERNELS`]. Public for the same reason
/// [`grad::BwdIds::axpy`] is: a caller stitching extra graph onto these blocks
/// must reuse this pipeline rather than register a second `add2`, which the CPU
/// backend's JIT rejects as a duplicate definition.
pub const ADD2_SLOT: usize = K_ADD2;

/// The block builder's kernel set, in slot order. Public so a profiler can name
/// the kernel behind each recorded [`Step`] (`flux2_bench vae`), and so a crate
/// that needs extra kernels alongside these can build its set with
/// [`kernels_with`] instead of restating them.
pub const KERNELS: [(&str, &str); 17] = [
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("gn_apply", kernels::GN_APPLY),
    ("silu", kernels::SILU),
    ("add2", kernels::ADD2),
    ("upsample2", kernels::UPSAMPLE2),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("gn_stats_wg", kernels::GN_STATS_WG),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("im2col_at", kernels::IM2COL_AT),
    ("nlc_bias_nchw", kernels::NLC_BIAS_NCHW),
    ("gn_part", kernels::GN_PART),
    ("gn_stats2", kernels::GN_STATS2),
    ("concat2", kernels::CONCAT2),
];

/// Slot index the first caller-supplied kernel gets when a kernel set is built
/// with [`kernels_with`] — i.e. `KERNELS.len()`. A `Builder` addresses slots
/// `0..NEXT_SLOT`, so a caller's own kernels must start here.
pub const NEXT_SLOT: usize = KERNELS.len();

/// `(scores, softmax, apply)` slots of the shared `attn_*_bidir` trio, in
/// [`model::block::BidirIds`] field order.
///
/// Exported because a crate that layers its own kernels on top of
/// [`kernels_with`] may want that trio too (`crates/sdxlunet` reuses it for the
/// self-attention fallback on devices without workgroup reductions) and the
/// alternative is a literal `8, 9, 10` in the caller — which reorders silently
/// into a wrong pipeline the moment [`KERNELS`] grows an entry in the middle.
pub const ATTN_BIDIR_SLOTS: (usize, usize, usize) = (K_ATTN_SCORES, K_ATTN_SOFTMAX, K_ATTN_APPLY);

/// The tiled-GEMM slot inside [`KERNELS`] — `matmul_reg3`, which the conv
/// lowering dispatches and which is the tiled kernel any caller layering on
/// [`kernels_with`] should hand to [`model::block::pick_gemm`].
///
/// Exported because the alternative is registering a *second* tiled GEMM
/// alongside this one, which is what `crates/sdxlunet` did: it carried both this
/// kernel and its own `matmul_reg2`, and every `nn.Linear` in the model went to
/// the slower of the two. `matmul_reg3` is `matmul_reg2` with the shared-memory
/// bank conflicts removed — same `Params`, same `@workgroup_size(256)`, same
/// dispatch arithmetic, bit-identical output - and it measured faster at all
/// twelve shapes swept from `[1,4096,4096]` to `[8192,320,320]`.
/// There is no shape where preferring `matmul_reg2` is correct.
pub const MATMUL_REG3_SLOT: usize = K_MATMUL;

/// The backward kernels the reverse walk of a train-mode [`Builder`] dispatches,
/// in [`grad::BwdIds`] order. A crate that trains these blocks appends this
/// block to its kernel set and hands `grad::BwdIds::at(base)` to
/// [`grad::Trace::backward`] — the same "ids struct at a caller-chosen base"
/// shape `model::block::BidirIds` uses, so the shared blocks never assume a
/// slot layout their user did not pick.
///
/// Everything here is barrier-free and gather-based: one invocation per element
/// of the buffer it writes. The two per-channel reductions (`gn_dgamma` /
/// `gn_dbeta`, C invocations each) and the per-group `gn_dsum` (N*G) have no
/// cooperative twin anywhere in the tree — that is a documented perf gap,
/// NOT a correctness gate, because none of them
/// uses `workgroupBarrier()` and all three are exact on `backend-cpu`.
pub const BWD_KERNELS: [(&str, &str); 36] = [
    ("conv2d_dx", kernels::CONV2D_DX),
    ("conv2d_dw", kernels::CONV2D_DW),
    ("bias_grad", kernels::BIAS_GRAD),
    ("silu_bwd", kernels::SILU_BWD),
    ("scale_chan", kernels::SCALE_CHAN),
    ("gn_dx", kernels::GN_DX),
    ("gn_dgb_part", kernels::GN_DGB_PART),
    ("gn_dgb2", kernels::GN_DGB2),
    ("upsample2_dx", kernels::UPSAMPLE2_DX),
    ("axpy", kernels::AXPY),
    ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
    ("attn_bwd_dq_bidir", kernels::ATTN_BWD_DQ_BIDIR),
    ("attn_bwd_dk_bidir", kernels::ATTN_BWD_DK_BIDIR),
    // The GEMM-lowered conv input gradient. APPENDED, so every existing
    // `BwdIds::at(base)` offset stays valid.
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("col2im", kernels::COL2IM),
    // ...and the weight gradient's. `im2col_at` is NOT here: the forward set
    // already registers it, and a second definition of the same kernel is what
    // the CPU JIT rejects as a duplicate. The reverse reuses the forward's slot
    // through `super::K_IM2COL_AT`, exactly as it already does for `nchw_nlc`.
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    // The two-stage replacement for `gn_dsum`, which is one lane per group.
    ("gn_dsum_part", kernels::GN_DSUM_PART),
    ("gn_dsum2", kernels::GN_DSUM2),
    // Split-K weight gradient. `matmul_dw_reg`'s tile grid is
    // ceil(Cout/128)*ceil(CinKK/128) workgroups REGARDLESS of how long the
    // contraction is, so a wide-shallow conv leaves most of the card idle:
    // measured at 9 workgroups on a 30-SM P40, i.e. most SMs with nothing to
    // do. These split the contraction instead. See
    // `matmul_dw_reg_splitk.wgsl`.
    ("matmul_dw_reg_splitk", kernels::MATMUL_DW_REG_SPLITK),
    ("dw_splitk_reduce", kernels::DW_SPLITK_REDUCE),
    // ---- the transformer half's adjoints (SDXL's `Transformer2DModel`) ----
    // APPENDED, so every existing `BwdIds::at(base)` offset stays valid. None of
    // these is a new kernel: each already existed for a decoder LM's backward
    // (`t5encoder`, `clip`, `gpt2`), and the SDXL UNet reaches them by recording
    // its transformer stages on this tape instead of `push_step`ing them past it.
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("gelu_erf_bwd", kernels::GELU_ERF_BWD),
    ("add_chan_bcast_dv", kernels::ADD_CHAN_BCAST_DV),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dk_cross_acc", kernels::ATTN_BWD_DK_CROSS_ACC),
    ("attn_bwd_dv_cross_acc", kernels::ATTN_BWD_DV_CROSS_ACC),
    // `layernorm_dgamma` needs the row mean/inv-std the FORWARD had; the
    // forward LayerNorm does not retain them, so the reverse recomputes them.
    // Backward-only, hence safe to register here.
    ("ln_stats", kernels::LN_STATS),
    // NOTE `mul` is deliberately ABSENT: [`Op::Mul`]'s adjoint is two more
    // `mul`s, and the caller that records a `Mul` already registers that kernel
    // in its own FORWARD set. Registering a second definition under the same
    // name is what the CPU JIT rejects outright (`DuplicateDefinition`), so the
    // reverse reaches it through the caller-supplied [`XformerIds::mul`] slot -
    // the same arrangement `im2col_at` already uses via `super::K_IM2COL_AT`.
    // The two-stage replacement for `bias_grad`, which is one lane per output
    // feature walking every row serially - measured at 1.3% of the memory
    // roof on a VQGAN training step (kernel-performance.md M5.7), the same
    // occupancy pathology `gn_dsum_part`/`gn_dgb_part` above already fixed for
    // GroupNorm's own per-channel reductions. APPENDED, so every existing
    // `BwdIds::at(base)` offset stays valid.
    ("bias_grad_part", kernels::BIAS_GRAD_PART),
    ("bias_grad_final", kernels::BIAS_GRAD_FINAL),
];

/// Row-chunks per column for the two-stage `bias_grad_part`/`bias_grad_final`
/// pair - the same fixed, ungated split `GN_P` already uses for the identical
/// barrier-free partial-reduction shape (`gn_dsum_part`/`gn_dgb_part`). No
/// capability gate: neither stage uses `workgroupBarrier`, so `backend-cpu`
/// runs the split unconditionally too.
pub(crate) const BIAS_GRAD_P: u32 = 64;

/// Workgroups the split-K weight gradient aims to launch.
///
/// Swept per shape (`vqgan_bench dwtn`): every VQGAN conv-backward shape's
/// optimum landed on **288** workgroups — 144 tiles x 2 slices, 36 x 8, and
/// 9 x 32 all beat their neighbours — i.e. ~9.6 per SM on the P40's 30. So the
/// slice count is not a constant to guess but `ceil(TARGET / tiles)`.
pub const DW_SPLITK_TARGET_WGS: u32 = 288;

/// Copy [`KERNELS`] into the front of a fixed-size kernel set whose remaining
/// slots the caller fills, so a crate that needs the shared blocks **and** its
/// own kernels never restates the shared list (a restated list that drifts by
/// one entry is silently wrong, not a crash).
///
/// `N` must be `KERNELS.len() + extra.len()`; it is checked at compile time
/// through the const evaluation (an out-of-range write is a const error).
///
/// ```ignore
/// const fn set() -> [(&'static str, &'static str); 17] {
///     let mut k = vae::blocks::kernels_with::<17>();
///     k[vae::blocks::NEXT_SLOT] = ("vq_argmin", kernels::VQ_ARGMIN);
///     k[vae::blocks::NEXT_SLOT + 1] = ("embed", kernels::EMBED);
///     k
/// }
/// pub const KERNELS: [(&str, &str); 17] = set();
/// ```
pub const fn kernels_with<const N: usize>() -> [(&'static str, &'static str); N] {
    let mut out = [("", ""); N];
    let mut i = 0;
    while i < KERNELS.len() {
        out[i] = KERNELS[i];
        i += 1;
    }
    out
}

/// Host tensors by name (checkpoint key, e.g. `decoder.conv_in.weight`) →
/// `(shape, row-major f32 data)`.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// One int8-packed weight - `model::int8::quantize_weight`'s packed `[n,
/// k/4]` u32 words plus its `[n, k/32]` f32 group scale, alongside the logical
/// `[n, k]` shape needed to dequantize it (the packed shape alone cannot
/// recover `k`, the same reason `model::int8::upload_dequantized` takes
/// `n`/`k` as separate arguments).
pub struct PackedWeight {
    pub shape: Vec<usize>,
    pub packed: Vec<u32>,
    pub scale: Vec<f32>,
}

/// Packed int8 weights by name - the storage-tier sibling of [`Tensors`].
/// [`Builder::dev`] consults this AFTER `Tensors` for a name it cannot find
/// there - see [`Builder::set_packed`] for why a name lives in one map or
/// the other, never both.
pub type PackedTensors = HashMap<String, PackedWeight>;

/// The per-architecture leaf tensor names the shared blocks look up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockNames {
    /// Resnet projection shortcut (1×1 conv, present only when `cin != cout`).
    pub shortcut: &'static str,
    /// Attention pre-norm.
    pub attn_norm: &'static str,
    pub attn_q: &'static str,
    pub attn_k: &'static str,
    pub attn_v: &'static str,
    /// Attention output projection (1×1 conv).
    pub attn_proj: &'static str,
    /// Pre-fused qkv projection. `Some(name)` means the checkpoint already
    /// ships one `[3C, C, 1, 1]` weight (DIAMOND's `qkv_proj`), so
    /// `attn_q`/`attn_k`/`attn_v` are unused and the host-side concatenation
    /// is skipped. `None` means three separate tensors to fuse.
    pub attn_qkv: Option<&'static str>,
    /// The attention residual adds the NORMED tensor instead of the block
    /// input. DIAMOND's `SelfAttention2d.forward` reassigns `x = norm(x)`
    /// before the skip (`blocks.py`), so its residual source is the norm
    /// output; diffusers and VQGAN both add the untouched input.
    pub attn_residual_normed: bool,
}

impl BlockNames {
    /// diffusers `ResnetBlock2D` / `Attention` naming (`AutoencoderKL`).
    pub const fn diffusers() -> BlockNames {
        BlockNames {
            shortcut: "conv_shortcut",
            attn_norm: "group_norm",
            attn_q: "to_q",
            attn_k: "to_k",
            attn_v: "to_v",
            attn_proj: "to_out.0",
            attn_qkv: None,
            attn_residual_normed: false,
        }
    }

    /// `basicsr` VQGAN naming (`ResBlock` / `AttnBlock` in `vqgan_arch.py`).
    pub const fn vqgan() -> BlockNames {
        BlockNames {
            shortcut: "conv_out",
            attn_norm: "norm",
            attn_q: "q",
            attn_k: "k",
            attn_v: "v",
            attn_proj: "proj_out",
            attn_qkv: None,
            attn_residual_normed: false,
        }
    }

    /// DIAMOND naming (`SelfAttention2d` / `ResBlock` in the reference
    /// `blocks.py`). Two things set it apart from the VAE architectures above:
    /// the checkpoint ships one pre-fused `qkv_proj` rather than three
    /// projections, and the residual adds the normed tensor (see
    /// [`attn_residual_normed`](BlockNames::attn_residual_normed)).
    pub const fn diamond() -> BlockNames {
        BlockNames {
            shortcut: "proj",
            attn_norm: "norm.norm",
            // Unused: `attn_qkv` is Some, so the three-tensor fuse never runs.
            attn_q: "",
            attn_k: "",
            attn_v: "",
            attn_proj: "out_proj",
            attn_qkv: Some("qkv_proj"),
            attn_residual_normed: true,
        }
    }
}

/// The caller's FORWARD kernel slots for the transformer stages this builder can
/// record but does not itself register.
///
/// [`KERNELS`] is the conv-autoencoder set: conv, GroupNorm, SiLU, add, upsample,
/// the NCHW/NLC permutations, the bidirectional self-attention trio, and
/// `concat2`. A diffusion backbone's `Transformer2DModel` needs LayerNorm, a
/// GEMM, a bias add, erf-GELU, a product and the CROSS-attention trio on top -
/// and the caller already registers all of those in its own set past
/// [`NEXT_SLOT`]. Re-registering them here would give the CPU JIT two
/// definitions of one kernel name, which it rejects outright, so the caller
/// hands its slots over instead. Same arrangement the reverse already uses for
/// `im2col_at`.
///
/// Supply this with [`Builder::set_xformer_ids`] BEFORE recording any
/// transformer stage; the recorders panic by name rather than dispatching slot
/// zero if it is missing.
#[derive(Clone, Copy)]
pub struct XformerIds {
    /// LayerNorm forward (its `_dx`/`_dgamma`/`_dbeta` adjoints ARE in
    /// [`BWD_KERNELS`] - only the forward collides).
    pub layernorm: LayerNormFwd,
    /// The naive and register-tiled GEMMs, as `model::block::pick_gemm` selects
    /// between them.
    pub matmul: usize,
    pub matmul_reg: usize,
    pub bias_add: usize,
    pub gelu_erf: usize,
    /// Elementwise product - reused by [`Op::Mul`]'s own adjoint, which is two
    /// more products.
    pub mul: usize,
    /// The `attn_*_cross` trio.
    pub cross: model::block::CrossIds,
    /// `add_chan_bcast`, the per-(image, channel) broadcast add.
    pub add_chan: usize,
}

/// The three slots `model::block::LayerNormIds` resolves a forward from.
#[derive(Clone, Copy)]
pub struct LayerNormFwd {
    pub ids: model::block::LayerNormIds,
}

/// The caller's `edm_mix`/`scale_row` slots for [`Builder::mix`]/[`Op::Mix`].
///
/// Threaded through rather than registered in [`KERNELS`] for the same
/// reason `mul` is threaded through [`XformerIds`] instead: `crates/diamond`
/// already registers both kernels under its own slots, and the CPU backend's
/// JIT rejects a second definition of the same kernel name. A caller that
/// wants [`Builder::mix`] registers `edm_mix`/`scale_row` in its own kernel
/// set and hands the slots over with [`Builder::set_mix_ids`].
#[derive(Clone, Copy)]
pub struct MixIds {
    /// `edm_mix` - [`Builder::mix`]'s forward dispatch.
    pub fwd: usize,
    /// `scale_row` - [`Op::Mix`]'s backward. `edm_mix.wgsl`'s own header
    /// documents this as its exact adjoint (`dx = scale_row(dy, a)`, `df =
    /// scale_row(dy, b)`); there is no `dab` kernel because the coefficients
    /// are host constants, never trained - the same shape EDM's own
    /// `c_skip`/`c_out` already have.
    pub bwd: usize,
}

/// One recorded forward stage, with exactly the buffers its adjoint reads.
///
/// Recorded only in **train mode** ([`Builder::set_train`]); the tape is what
/// `blocks::grad` walks in reverse. Every variant names its output `y` — the
/// reverse walk looks `y` up in the gradient map and skips the op when nothing
/// downstream consumed it.
#[derive(Clone)]
pub(crate) enum Op {
    /// Direct-lowered conv + bias. `w`/`b` are ParamStore-style tensor names.
    Conv {
        w: String,
        b: String,
        cin: u32,
        cout: u32,
        k: u32,
        stride: u32,
        pad: u32,
        h: u32,
        w_in: u32,
        ho: u32,
        wo: u32,
        x: DeviceBuffer,
        y: DeviceBuffer,
    },
    /// GroupNorm with the fused `gb[2C]` parameter and its retained `stats[2G]`.
    Gn {
        gb: String,
        c: u32,
        h: u32,
        w: u32,
        g: u32,
        x: DeviceBuffer,
        stats: DeviceBuffer,
        y: DeviceBuffer,
    },
    Silu { n: u32, x: DeviceBuffer, y: DeviceBuffer },
    Add2 { n: u32, a: DeviceBuffer, b: DeviceBuffer, y: DeviceBuffer },
    /// Nearest-2x upsample; dims are the INPUT's.
    Up2 { c: u32, h: u32, w: u32, x: DeviceBuffer, y: DeviceBuffer },
    /// `[c,hw] -> [hw,c]` (its adjoint is `NlcNchw` and vice versa).
    NchwNlc { c: u32, hw: u32, x: DeviceBuffer, y: DeviceBuffer },
    NlcNchw { c: u32, hw: u32, x: DeviceBuffer, y: DeviceBuffer },
    /// The bidirectional attention trio over the fused `qkv[t, 3c]` rows.
    /// `probs` is the cached softmax slab; `y` is the context `[t, c]`.
    Attn {
        c: u32,
        t: u32,
        /// Head split, as the forward dispatched it. Recorded rather than
        /// assumed: the adjoint quartet is parameterised by `(n_heads,
        /// head_dim)` exactly like the forward trio, and a tape that dropped
        /// them would compute a single-head gradient for a multi-head forward
        /// — silently, since the shapes still line up.
        heads: u32,
        head_dim: u32,
        qkv: DeviceBuffer,
        probs: DeviceBuffer,
        y: DeviceBuffer,
    },
    /// `y[m,n] = x[m,k] · W[n,k]ᵀ (+ b[n])` - the shape every diffusers
    /// `nn.Linear` has. `b` is `None` for the bias-free projections
    /// (`to_q`/`to_k`/`to_v`).
    Linear { w: String, b: Option<String>, m: u32, k: u32, n: u32, x: DeviceBuffer, y: DeviceBuffer },
    /// LayerNorm over `rows` rows of width `d`, with SEPARATE `gamma`/`beta`
    /// tensors - unlike [`Op::Gn`], whose affine pair is fused into one `gb[2C]`
    /// because `gn_apply` reads that layout. Nothing is fused here, so the two
    /// adjoints write two buffers.
    LayerNorm { gamma: String, beta: String, rows: u32, d: u32, eps: f32, x: DeviceBuffer, y: DeviceBuffer },
    /// Exact (erf) GELU. The tanh approximation has its OWN backward kernel and
    /// the two are not interchangeable, so which one the forward used is
    /// recorded rather than assumed.
    GeluErf { n: u32, x: DeviceBuffer, y: DeviceBuffer },
    /// Elementwise product - GEGLU's `hidden · gelu(gate)`. Its own adjoint is
    /// two more products, so no backward kernel is needed.
    Mul { n: u32, a: DeviceBuffer, b: DeviceBuffer, y: DeviceBuffer },
    /// Per-(image, channel) broadcast add: `y[c,hw] = x[c,hw] + v[c]`. SDXL's
    /// resnets inject the timestep embedding this way (`resnet_time_scale_shift:
    /// "default"`), so `v`'s adjoint is a sum over the broadcast axis and `x`'s
    /// is `dy` itself.
    AddChan { c: u32, hw: u32, x: DeviceBuffer, v: DeviceBuffer, y: DeviceBuffer },
    /// Cross-attention: `tq` query rows against `tkv` key/value rows, with the
    /// queries in their own `[tq, c]` buffer and k/v fused as `[tkv, 2c]`.
    /// Distinct from [`Op::Attn`], which is self-attention over ONE fused
    /// `[t, 3c]` buffer - the two lengths and the two buffers are exactly what
    /// differ, and they are what the adjoint needs.
    Cross {
        c: u32,
        tq: u32,
        tkv: u32,
        heads: u32,
        head_dim: u32,
        q: DeviceBuffer,
        kv: DeviceBuffer,
        probs: DeviceBuffer,
        y: DeviceBuffer,
    },
    /// Channel concat of two NCHW maps - the up path's skip join. Its adjoint is
    /// two slices of `dy`, which `concat_split` performs without a scatter.
    Concat { ca: u32, cb: u32, hw: u32, a: DeviceBuffer, b: DeviceBuffer, y: DeviceBuffer },
    /// `y = a*x + b*f`, `edm_mix.wgsl`'s own contract - SUPIR's ZeroSFT/
    /// ZeroCrossAttn `control_scale` lerp reuses it verbatim. `a`/`b` are
    /// SCALARS (one-element device buffers), kept UNPACKED here (unlike the
    /// forward's packed `ab[2]`) because the backward (`scale_row`) needs
    /// each on its own. Host constants, never trained - like EDM's own
    /// `c_skip`/`c_out`, there is no `dab` kernel.
    Mix { n: u32, x: DeviceBuffer, f: DeviceBuffer, a: DeviceBuffer, b: DeviceBuffer, y: DeviceBuffer },
}

/// Graph-construction state (borrows the device + host tensors).
pub struct Builder<'a> {
    gpu: &'a Gpu,
    t: &'a Tensors,
    eps: f32,
    groups: u32,
    names: BlockNames,
    steps: Vec<Step>,
    taps: Vec<(String, DeviceBuffer, usize)>,
    /// Train mode: record the [`Op`] tape, keep every activation alive (the
    /// pool is disabled, so the forward is SSA and doubles as the backprop
    /// cache), and pin the **direct** conv/attention lowerings — those are the
    /// ones whose adjoints exist (`conv2d_dx`/`conv2d_dw` and the
    /// `attn_bwd_*_bidir` quartet). The `im2col_at + matmul_reg3` conv lowering
    /// would need a `col2im` that does not exist, and the GEMM attention path
    /// folds `1/sqrt(C)` into the q weights, which changes what `qkv.w`'s
    /// gradient means. Selection, not a second block implementation.
    train: bool,
    /// The caller's transformer-stage forward slots - see [`XformerIds`].
    xf: Option<XformerIds>,
    tape: Vec<Op>,
    /// Every weight buffer this builder uploaded, by tensor name, in first-use
    /// order. Memoized so one tensor is one device buffer (and therefore one
    /// gradient buffer) however many times a block asks for it.
    wmemo: HashMap<String, DeviceBuffer>,
    worder: Vec<(String, u64)>,
    /// Free-list of activation buffers keyed by exact length (words). An `act(len)`
    /// reuses a buffer of the same length whose last read is already recorded, so
    /// the resident peak is the max *concurrently-live* activation set instead of
    /// the sum of every activation — the difference between decoding 640² and 1536²
    /// on a 24 GB card. Reuse is bit-exact: the graph runs its steps in order with
    /// barriers (as the qwen/zimage scratch reuse relies on), and a buffer is only
    /// freed after its last consumer step is emitted, so the reusing write always
    /// follows the last read. Disabled when `taps_on` (taps pin buffers).
    pool: HashMap<u64, Vec<DeviceBuffer>>,
    /// Record intermediate taps (for parity debugging via `read_tap`). Off by
    /// default — pins buffers and defeats pooling.
    taps_on: bool,
    /// The device executes workgroup-cooperative reductions (barriers): pick the
    /// workgroup-per-group GroupNorm statistics kernel and the conv/attention
    /// GEMM lowerings. False on the CPU JIT, which keeps the reference kernels
    /// (whose native AVX2 fast paths are the fast CPU route anyway).
    coop: bool,
    /// Bytes uploaded since the last forced staging drain — see
    /// [`Builder::upload`].
    uploaded: u64,
    /// The single im2col scratch (`length, buffer`) shared by every lowered
    /// conv, grown on demand. Bounded by [`COL_BUDGET_MIB`] - a whole-image
    /// im2col operand exceeds the P40's 2047 MiB binding limit, so the GEMM is
    /// chunked over spatial positions instead (see `im2col_at.wgsl`).
    col: Option<(u64, DeviceBuffer)>,
    /// Attention head width. `None` is one head of width `C` — what every VAE
    /// architecture here uses. See [`set_attn_head_dim`](Builder::set_attn_head_dim).
    attn_head_dim: Option<u32>,
    /// Device words this builder has actually allocated: every weight buffer,
    /// plus every activation that MISSED the pool. A pooled reuse allocates
    /// nothing and is not counted, so this total is the graph's resident
    /// high-water set rather than the sum of everything it ever touched.
    ///
    /// It exists to be the ground truth a placement ESTIMATE is gated against
    /// (`crates/vae/tests/footprint.rs`). An estimate that under-reports is
    /// not a slightly-wrong number, it is a plan that says a card has room
    /// and then a driver out-of-memory.
    allocated: u64,
    /// The caller's `edm_mix`/`scale_row` slots - see [`MixIds`]. `None`
    /// until [`Builder::set_mix_ids`] is called; [`Builder::mix`] panics by
    /// name rather than dispatching slot zero if it is missing.
    mix_ids: Option<MixIds>,
    /// An int8-packed fallback weight source - see [`Builder::set_packed`].
    /// `None` for every caller except an int8 build (`sdxlunet::int8`,
    /// `supir::int8`): the plain fp32 path is completely unaffected, since
    /// [`Builder::dev`] only ever consults this after `t` reports a name
    /// missing.
    packed: Option<&'a PackedTensors>,
}

/// Ceiling on the im2col scratch, in f32 words (512 MiB). The lowered conv
/// processes `floor(budget / CinKK)` output positions per GEMM, so this trades
/// scratch for the number of chunks; at 512² the largest operand would be
/// 2.4 GB unchunked, which is both unbindable and hostile to a card shared with
/// a resident DiT. Override with `BRAIN_CONV_COL_MIB` (or its original name
/// `BRAIN_VAE_COL_MIB`) - the knob is shared with every other lowering in the
/// tree, see [`gpu_core::lower`].
const COL_BUDGET_MIB: u64 = 512;

// `conv_s`'s direct-vs-lowered choice now runs through
// `backend_api::select::Op::Conv2d` (`GEMM_CONV2D_MIN_COUT`/
// `GEMM_CONV2D_MIN_HW` in that module carry the measured thresholds and
// their sweep provenance - migrated there verbatim so this campaign's
// selection seam is the one place the decision lives, per that `Op`'s doc).

impl<'a> Builder<'a> {
    /// New builder over `gpu` (built with a kernel set whose first
    /// [`KERNELS`]`.len()` slots are [`KERNELS`]) and the host `tensors`.
    /// `eps`/`groups` configure every GroupNorm; `names` selects the leaf
    /// tensor names; `taps_on` records block outputs (and disables pooling).
    pub fn new(
        gpu: &'a Gpu,
        tensors: &'a Tensors,
        eps: f32,
        groups: u32,
        names: BlockNames,
        taps_on: bool,
    ) -> Builder<'a> {
        Builder {
            gpu,
            t: tensors,
            eps,
            groups,
            names,
            steps: Vec::new(),
            taps: Vec::new(),
            train: false,
            xf: None,
            tape: Vec::new(),
            wmemo: HashMap::new(),
            worder: Vec::new(),
            pool: HashMap::new(),
            allocated: 0,
            taps_on,
            coop: gpu.caps().workgroup_reductions,
            uploaded: 0,
            col: None,
            attn_head_dim: None,
            mix_ids: None,
            packed: None,
        }
    }

    /// Change the GroupNorm epsilon used by every subsequently-recorded block.
    ///
    /// `AutoencoderKL` and VQGAN use one epsilon throughout, which is why it is
    /// a constructor argument. SDXL's `UNet2DConditionModel` does not: its
    /// resnets and `conv_norm_out` use the config's `norm_eps` (1e-5) while the
    /// GroupNorm inside every `Transformer2DModel` is hardcoded to 1e-6
    /// (diffusers `_init_continuous_input`). Both live in one recorded graph,
    /// so the value has to be switchable at the boundary — a single-epsilon
    /// builder would force `crates/sdxlunet` to hand-roll a second GroupNorm, which
    /// is exactly the private copy this module exists to prevent.
    pub fn set_eps(&mut self, eps: f32) {
        self.eps = eps;
    }

    /// Change the GroupNorm group count used by every subsequently-recorded
    /// block.
    ///
    /// `AutoencoderKL` and VQGAN normalise with a fixed `norm_num_groups` at
    /// every width, which is why this is a constructor argument too. DIAMOND
    /// does not: it derives the count from the channel width
    /// (`wm_core::gn::num_groups`, `max(C/32, 1)`), so a graph whose levels are
    /// 8 and 48 channels wide normalises over 1 group at one level and 1 group
    /// at the other — and an SDXL-shaped fixed 32 would be wrong at both. Like
    /// [`set_eps`](Self::set_eps), the alternative to switching it at the
    /// boundary is a second private GroupNorm, which is the copy this module
    /// exists to prevent.
    pub fn set_groups(&mut self, groups: u32) {
        assert!(groups > 0, "GroupNorm needs at least one group");
        self.groups = groups;
    }

    /// Split attention into heads of `d` channels each (`heads = max(C/d, 1)`).
    ///
    /// The VAE architectures attend with a single head of width `C`, which is
    /// the default (`None`). DIAMOND uses `d = 8`. Only the *parameters* of the
    /// `attn_scores_bidir`/`attn_apply_bidir` dispatches change — both kernels
    /// have always taken `heads` and `head_dim`; the builder simply pinned them
    /// to `(1, C)`. Note the scale: those kernels divide by `sqrt(head_dim)`,
    /// so this is not merely a reshape, and it must match the reference's head
    /// count exactly.
    ///
    /// Rejected by the GEMM attention lowering, which contracts over the whole
    /// of `C` in one matmul and so cannot express per-head blocks; a multi-head
    /// graph stays on the per-element trio.
    pub fn set_attn_head_dim(&mut self, d: Option<u32>) {
        if let Some(d) = d {
            assert!(d > 0, "attention head width must be positive");
        }
        self.attn_head_dim = d;
    }

    /// Supply the caller's `edm_mix`/`scale_row` slots. Must precede any
    /// [`Builder::mix`] call - see [`MixIds`].
    pub fn set_mix_ids(&mut self, ids: MixIds) {
        self.mix_ids = Some(ids);
    }

    /// Install a packed int8 weight source: a name [`Builder::dev`] cannot
    /// find in the plain `tensors` map falls back to here, dequantized ONE
    /// TENSOR AT A TIME at upload rather than the whole map at once.
    ///
    /// This is the seam that lets an int8 build avoid ever holding a
    /// whole-model fp32 `Tensors` map in host RAM: `sdxlunet::int8`/
    /// `supir::int8`'s `quantize_tensors` splits a checkpoint into a small
    /// `full` map (never-quantized names, biases, norm gains - everything
    /// that isn't a rank-2 GEMM operand) and a `packed` map roughly a
    /// quarter the bytes (`model::int8::quantize_weight`'s packed `[n,k/4]`
    /// u32 layout). `dev` still uploads a plain fp32 device buffer either
    /// way - this crate dispatches no int8 kernel of its own - so the
    /// SAVING is host-side only: on a unified-memory box (no discrete GPU,
    /// so "device" memory is the same system RAM) that is exactly the
    /// difference between a 15.6 GB resident checkpoint and a ~4 GB one
    /// while the device-side upload climbs toward its own full fp32 size,
    /// which is what closed a real, measured OOM in `crates/supir`'s
    /// combined trunk+adaptors+backbone build (see that crate's `int8.rs`).
    pub fn set_packed(&mut self, p: &'a PackedTensors) {
        self.packed = Some(p);
    }

    /// Record the reverse-mode tape (see [`Builder::train`]). Set this BEFORE
    /// recording any block — it changes both what is kept and which lowering
    /// each block picks.
    pub fn set_train(&mut self, on: bool) {
        assert!(self.steps.is_empty(), "vae::blocks: set_train must precede the first block");
        self.train = on;
    }

    /// Whether this builder is recording the reverse-mode tape.
    ///
    /// A caller whose stage has two lowerings - one differentiable, one not
    /// (flash attention never materialises the softmax its adjoint binds) - asks
    /// here rather than tracking train mode a second time.
    pub fn is_train(&self) -> bool {
        self.train
    }

    /// The recorded forward tape + weight buffers, for [`grad::Trace::backward`].
    /// Empty unless [`Builder::set_train`] was called.
    pub fn trace(&self) -> grad::Trace {
        grad::Trace::new(
            self.tape.clone(),
            self.worder.clone(),
            &self.wmemo,
            self.xf.map(|x| x.mul),
            self.mix_ids.map(|m| m.bwd),
        )
    }

    /// The device the graph is being recorded on.
    pub fn gpu(&self) -> &'a Gpu {
        self.gpu
    }

    /// Whether block-output taps are being recorded.
    pub fn taps_on(&self) -> bool {
        self.taps_on
    }

    /// Append a caller-recorded step (for kernels outside the shared blocks —
    /// e.g. `vqgan`'s codebook assignment). Ordering is the caller's.
    ///
    /// **Records no [`Op`].** On a [`Builder::set_train`] builder that makes the
    /// step invisible to [`grad::Trace::backward`], which walks the tape and
    /// silently skips any producer whose output no consumer claimed — so a
    /// pushed step in the middle of a differentiated chain breaks the chain and
    /// every parameter upstream of it gets a **zero** gradient with no error.
    /// Either keep pushed steps outside the differentiated subgraph (record them
    /// on a separate non-train `Builder`, as `vqgan::train` does for the
    /// codebook assignment and gather), or stitch their adjoint on by hand
    /// around `Trace::backward`, as `vqgan::train` does for the quantiser seam.
    pub fn push_step(&mut self, step: Step) {
        self.steps.push(step);
    }

    /// Number of steps recorded so far — a caller that submits a *prefix* of
    /// the graph (vqgan replays the generator alone, skipping the codebook
    /// gather) records the split point with this.
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Consume the builder, yielding the recorded steps and taps.
    pub fn finish(self) -> (Vec<Step>, Vec<(String, DeviceBuffer, usize)>) {
        (self.steps, self.taps)
    }

    /// A host tensor by name; panics naming the tensor if absent (import
    /// validates coverage up front, so this only fires on a schedule bug).
    pub fn get(&self, name: &str) -> &(Vec<usize>, Vec<f32>) {
        self.t.get(name).unwrap_or_else(|| panic!("vae::blocks: missing tensor {name}"))
    }

    /// Upload a host tensor to the device by name (memoized: one tensor is one
    /// device buffer, so a training build has exactly one gradient buffer per
    /// tensor however many blocks read it).
    ///
    /// Checks the plain fp32 map first, then the packed int8 fallback (see
    /// [`Builder::set_packed`]) - dequantizing there is bounded to THIS ONE
    /// tensor (at most tens of MB for anything in this workspace), never the
    /// whole checkpoint, and the scratch is dropped the moment `upload`
    /// returns.
    pub fn dev(&mut self, name: &str) -> DeviceBuffer {
        if let Some(b) = self.wmemo.get(name) {
            return b.clone();
        }
        // `self.t` is a `&'a Tensors` independent of `&self`, so copying the
        // reference out lets `upload` take `&mut self` without cloning the
        // tensor (the SDXL feed-forward weights are 52 MB each).
        let t = self.t;
        if let Some((_, data)) = t.get(name) {
            let buf = self.upload(data);
            return self.remember(name, buf, data.len() as u64);
        }
        if let Some(pw) = self.packed.and_then(|p| p.get(name)) {
            let (n, k) = (pw.shape[0], pw.shape[1]);
            let data = model::int8::dequantize_weight(&pw.packed, &pw.scale, n, k);
            let buf = self.upload(&data);
            return self.remember(name, buf, data.len() as u64);
        }
        panic!("vae::blocks: missing tensor {name} (checked both the plain and the packed weight source)");
    }

    /// Upload one weight tensor, non-ReBAR-safe.
    ///
    /// **Not `storage_init`**, and both departures are load-bearing on a P40
    /// (`paramstore`'s upload loop, and
    /// `s3dit::BlockWeights::upload` both record the same two):
    ///
    /// 1. `create_buffer_init`'s mapped-at-creation path forces weights into an
    ///    inefficient memory type on a card without resizable BAR, inflating a
    ///    multi-GB model far past its nominal size. `storage()` + `write()`
    ///    gives plain DEVICE_LOCAL, which packs tightly.
    /// 2. wgpu only reclaims `write_buffer` staging on a real submit + drain, so
    ///    a long upload accrues a second copy of the whole model. `poll_wait`
    ///    per tensor plus a 1-element readback every ~1 GiB forces that drain.
    ///
    /// Neither changes a single bit of the uploaded data — this is purely how
    /// the memory is obtained. It became necessary when `crates/sdxlunet` put a
    /// **10.3 GB** model through this builder (the earlier users — the FLUX.2
    /// VAE and the VQGAN pair — are each well under 1 GB and never hit it);
    /// SDXL OOM'd on a P40 with 20 GB free before this.
    fn upload(&mut self, data: &[f32]) -> DeviceBuffer {
        self.allocated += data.len() as u64;
        let buf = self.gpu.storage(data.len() as u64);
        let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
        self.gpu.write(&buf, &bits);
        self.gpu.poll_wait();
        self.uploaded += 4 * data.len() as u64;
        if self.uploaded > (1 << 30) {
            let _ = self.gpu.read(&buf, 1);
            self.uploaded = 0;
        }
        buf
    }

    /// Upload host data the builder synthesised (a fused `gamma|beta` or
    /// `q|k|v`) under a name of its own, memoized like [`Builder::dev`]. That
    /// fused buffer is the trainable tensor: `gn_dgamma`/`gn_dbeta` write the
    /// matching fused `dgb[2C]`, and one `conv2d_dw` covers the fused qkv.
    fn dev_fused(&mut self, name: &str, data: &[f32]) -> DeviceBuffer {
        if let Some(b) = self.wmemo.get(name) {
            return b.clone();
        }
        let buf = self.upload(data);
        self.remember(name, buf, data.len() as u64)
    }

    fn remember(&mut self, name: &str, buf: DeviceBuffer, n: u64) -> DeviceBuffer {
        match self.wmemo.entry(name.to_string()) {
            std::collections::hash_map::Entry::Occupied(e) => e.get().clone(),
            std::collections::hash_map::Entry::Vacant(e) => {
                self.worder.push((name.to_string(), n));
                e.insert(buf).clone()
            }
        }
    }

    /// Allocate an activation buffer of `len` words, reusing a same-length freed
    /// buffer from the pool when one is available (see [`Builder::pool`]).
    pub fn act(&mut self, len: u64) -> DeviceBuffer {
        if let Some(b) = self.pool.get_mut(&len).and_then(Vec::pop) {
            return b;
        }
        self.allocated += len;
        self.gpu.storage(len)
    }

    /// Device BYTES this builder has allocated so far - see [`Builder::
    /// allocated`]. Read before [`Builder::finish`] consumes the builder.
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated * 4
    }

    /// Return an activation buffer to the pool for reuse. MUST be called only after
    /// the buffer's last read step has been pushed (else a later reuse would clobber
    /// data a pending step still needs). No-op when pooling is disabled.
    pub fn free(&mut self, len: u64, buf: DeviceBuffer) {
        // Train mode keeps every activation: the forward buffer IS the backprop
        // cache, so reuse would silently overwrite a cached stage.
        if !self.taps_on && !self.train {
            self.pool.entry(len).or_default().push(buf);
        }
    }

    /// Record a named intermediate for later readback. No-op unless `taps_on`.
    pub fn tap(&mut self, name: String, buf: &DeviceBuffer, len: u32) {
        if self.taps_on {
            self.taps.push((name, buf.clone(), len as usize));
        }
    }

    /// Conv (+bias) `prefix.{weight,bias}`: `x[cin,h,w] → y[cout,ho,wo]`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv(
        &mut self,
        prefix: &str,
        cin: u32,
        cout: u32,
        k: u32,
        pad: u32,
        h: u32,
        w: u32,
        x: &DeviceBuffer,
    ) -> DeviceBuffer {
        let ho = (h + 2 * pad - k) + 1;
        let wo = (w + 2 * pad - k) + 1;
        self.conv_s(prefix, cin, cout, k, 1, pad, h, w, ho, wo, x)
    }

    /// diffusers `Downsample2D` (`use_conv`, `padding=0`) == VQGAN `Downsample`:
    /// F.pad(x,(0,1,0,1)) then a stride-2, k=3, pad=0 conv → `[c, h/2, w/2]`. The
    /// right/bottom zero-pad is reproduced by forcing `ho=wo=h/2` with `pad=0`:
    /// the kernels bounds-check their reads, so the extra bottom/right taps read
    /// 0 — exactly the asymmetric pad.
    pub fn conv_down(&mut self, prefix: &str, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
        self.conv_s(prefix, c, c, 3, 2, 0, h, w, h / 2, w / 2, x)
    }

    /// The im2col scratch, grown on demand and shared by every lowered conv.
    /// An outgrown buffer goes back to the activation pool (its last read is
    /// already recorded, which is exactly the pool's reuse contract).
    fn col_buf(&mut self, need: u64) -> DeviceBuffer {
        if let Some((len, b)) = &self.col {
            if *len >= need {
                return b.clone();
            }
        }
        if let Some((len, b)) = self.col.take() {
            self.free(len, b);
        }
        let b = self.act(need);
        self.col = Some((need, b.clone()));
        b
    }

    /// Ask `backend_api::select` whether `conv_s`'s lowered GEMM
    /// (`Op::Conv2d`) is the one to run for this shape. `RegisterTiled`
    /// names the whole lowering (im2col + GEMM + bias epilogue), and carries
    /// the `workgroup_reductions` requirement the register-tiled GEMM needs,
    /// which is what keeps the CPU JIT, whose split-at-barrier model
    /// mis-executes it, on the direct kernel without a backend-name test.
    /// `m` = output positions (`hw`), `n` = `Cout`, `k` = the contraction
    /// (`Cin*K*K`), same shape convention as `Op::Conv1d`'s callers.
    fn conv2d_lowered(gpu: &Gpu, hw: u32, cout: u32, cinkk: u32) -> bool {
        let shape = OpShape { m: hw, n: cout, k: cinkk, dtype: gpu_core::select::Dtype::F32 };
        DefaultSelector.select(SelectOp::Conv2d, shape, &gpu.caps()) == KernelVariant::RegisterTiled
    }

    /// Conv with an explicit stride and output size. Two lowerings:
    ///
    /// * **direct** - `conv_bias_reg`, the 8x4 register-tiled kernel. Measured
    ///   across every FLUX.2 VAE decode shape it holds a flat, low single-digit
    ///   fraction of the card's fp32 peak, and it was nearly the whole decode.
    ///   Its ceiling is structural rather than a tuning miss: 12 global loads
    ///   per 32 FMAs is 0.75 byte/FLOP, so the arithmetic-intensity roofline
    ///   already sits about where it measures, with caching worth a little more.
    /// * **lowered** (`backend_api::select::Op::Conv2d` picks
    ///   `KernelVariant::RegisterTiled` - capability + shape gated, see
    ///   [`conv2d_lowered`]) - `im2col_at` + `matmul_reg3` + `nlc_bias_nchw`,
    ///   i.e. `y[HW, Cout] = col[HW, CinKK] · Wᵀ`, which runs at the GEMM's
    ///   far higher share of peak. This is the same trade already scoped to
    ///   "a compute-bound discrete GPU" and taken for YOLO's convs; the P40
    ///   is that GPU. The transposed orientation (positions as GEMM ROWS) is
    ///   what makes it chunkable: a spatial chunk is a contiguous row range
    ///   of both `col` and the output, so the 2.4 GB whole-image operand
    ///   becomes a bounded scratch (see `im2col_at.wgsl`).
    #[allow(clippy::too_many_arguments)]
    pub fn conv_s(
        &mut self,
        prefix: &str,
        cin: u32,
        cout: u32,
        k: u32,
        stride: u32,
        pad: u32,
        h: u32,
        w: u32,
        ho: u32,
        wo: u32,
        x: &DeviceBuffer,
    ) -> DeviceBuffer {
        let (wn, bn) = (format!("{prefix}.weight"), format!("{prefix}.bias"));
        let wgt = self.dev(&wn);
        let bias = self.dev(&bn);
        let hw = ho * wo;
        let cinkk = cin * k * k;
        if self.train || !Self::conv2d_lowered(self.gpu, hw, cout, cinkk) {
            return self.conv_direct(wn, bn, &wgt, &bias, cin, cout, k, stride, pad, h, w, ho, wo, x);
        }
        let y = self.act((cout * ho * wo) as u64);
        {
            // Positions per GEMM: a multiple of the 128-row tile, inside the
            // scratch budget, at least one tile.
            let budget = gpu_core::lower::col_budget_floats(COL_BUDGET_MIB);
            let chunk = gpu_core::lower::col_chunk_rows(budget, u64::from(cinkk), 128, hw);
            let col = self.col_buf(chunk as u64 * cinkk as u64);
            let nhwc = self.act((hw * cout) as u64);
            let mut pos = 0u32;
            while pos < hw {
                let cnt = chunk.min(hw - pos);
                self.steps.push(self.gpu.step(
                    K_IM2COL_AT,
                    &[x, &col],
                    &[cin, h, w, k, stride, pad, ho, wo, cinkk, pos, cnt],
                    cnt * cinkk,
                ));
                self.steps.push(self.gpu.step_sliced(
                    K_MATMUL,
                    &[&col, &wgt, &nhwc],
                    &[(0, 0), (0, 0), (pos as u64 * cout as u64, cnt as u64 * cout as u64)],
                    &[cnt, cinkk, cout],
                    cnt.div_ceil(128) * cout.div_ceil(128) * 256,
                ));
                pos += cnt;
            }
            self.steps.push(self.gpu.step(
                K_NLC_BIAS_NCHW,
                &[&nhwc, &bias, &y],
                &[hw * cout, cout, hw],
                cout.div_ceil(64) * hw.div_ceil(64) * 64,
            ));
            self.free((hw * cout) as u64, nhwc);
        }
        y
    }

    /// The direct (`conv_bias_reg`) lowering over already-uploaded weight/bias
    /// buffers, recording an [`Op::Conv`] in train mode. Split out of
    /// [`Builder::conv_s`] so the fused qkv projection inside [`Builder::attn`]
    /// dispatches and back-propagates through the same one implementation.
    #[allow(clippy::too_many_arguments)]
    fn conv_direct(
        &mut self,
        wn: String,
        bn: String,
        wgt: &DeviceBuffer,
        bias: &DeviceBuffer,
        cin: u32,
        cout: u32,
        k: u32,
        stride: u32,
        pad: u32,
        h: u32,
        w: u32,
        ho: u32,
        wo: u32,
        x: &DeviceBuffer,
    ) -> DeviceBuffer {
        let y = self.act((cout * ho * wo) as u64);
        let threads = cout.div_ceil(8) * (ho * wo).div_ceil(4);
        self.steps.push(self.gpu.step(
            K_CONV,
            &[x, wgt, bias, &y],
            &[1, cin, h, w, cout, k, stride, pad, ho, wo],
            threads,
        ));
        if self.train {
            self.tape.push(Op::Conv {
                w: wn,
                b: bn,
                cin,
                cout,
                k,
                stride,
                pad,
                h,
                w_in: w,
                ho,
                wo,
                x: x.clone(),
                y: y.clone(),
            });
        }
        y
    }

    /// Static affine GroupNorm from `prefix.{weight,bias}` (32 groups, eps
    /// 1e-6): `y = gamma·(x-μ)/σ + beta` per group. `gb = [gamma‖beta]`.
    pub fn gn(&mut self, prefix: &str, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let (_, gamma) = self.get(&format!("{prefix}.weight"));
        let (_, beta) = self.get(&format!("{prefix}.bias"));
        let mut gbv = gamma.clone();
        gbv.extend_from_slice(beta);
        let gbn = format!("{prefix}.gb");
        let gb = self.dev_fused(&gbn, &gbv);
        self.gn_gb(&gbn, c, h, w, x, &gb)
    }

    /// GroupNorm with a caller-owned fused `[gamma ‖ beta]` buffer (`2·C`
    /// floats), rather than one loaded from `prefix.{weight,bias}`.
    ///
    /// Exists for **adaptive** GroupNorm: DIAMOND's `AdaGroupNorm` computes
    /// gamma and beta per frame from the conditioning vector, so the affine
    /// parameters are not checkpoint tensors at all — the caller keeps the
    /// buffer and rewrites it before each forward. Everything else (the
    /// statistics kernels, the capability gate, the pooling, the training tape)
    /// is identical, which is exactly why this is a seam on the shared block
    /// rather than a second GroupNorm in `crates/diamond`.
    ///
    /// `gb_name` names the buffer on the training tape. For an adaptive norm it
    /// is not a checkpoint key and carries no gradient — DIAMOND trains the
    /// linear that PRODUCES gamma/beta, on the host.
    pub fn gn_gb(
        &mut self,
        gb_name: &str,
        c: u32,
        h: u32,
        w: u32,
        x: &DeviceBuffer,
        gb: &DeviceBuffer,
    ) -> DeviceBuffer {
        let gbn = gb_name.to_string();
        let g = self.groups;
        let stats = self.act(2 * g as u64);
        let y = self.act((c * h * w) as u64);
        // Statistics: one WORKGROUP per group where the device can run a
        // workgroup reduction (`gn_stats_wg`), else the per-group reference
        // kernel. `gn_stats` dispatches `g` = 32 *invocations* for up to 33 M
        // elements, measured as a large share of a 512² FLUX.2 VAE decode; the
        // cooperative kernel is the same two-pass math, coalesced and 32-way
        // parallel (see `gn_stats_wg.wgsl`).
        if self.coop {
            self.steps.push(self.gpu.step(
                K_GN_STATS_WG,
                &[x, &stats],
                &[1, c, h, w, g, f(self.eps)],
                g * 256,
            ));
        } else {
            // No workgroup reductions (backend-cpu): the SERIAL `gn_stats`
            // dispatches `g` invocations for up to 33 M elements. The
            // barrier-free two-stage reduction is the right fallback and
            // measured several times faster on the CPU JIT at every VAE
            // decoder shape from [512,64,64] to [128,512,512], the margin
            // holding as the shape grows. `crates/wm-diamond` built this pair
            // after profiling put the serial kernel at the bulk of its frame
            // time, and the shared builder never learned about it.
            let part = self.act(2 * g as u64 * GN_P as u64);
            self.steps.push(self.gpu.step(
                K_GN_PART,
                &[x, &part],
                &[1, c, h, w, g, GN_P],
                g * GN_P,
            ));
            self.steps.push(self.gpu.step(
                K_GN_STATS2,
                &[&part, &stats],
                &[1, c, h, w, g, GN_P, f(self.eps)],
                g,
            ));
            self.free(2 * g as u64 * GN_P as u64, part);
        }
        self.steps.push(self.gpu.step(
            K_GN_APPLY,
            &[x, &stats, gb, &y],
            &[1, c, h, w, g],
            c * h * w,
        ));
        if self.train {
            // `stats` is NOT freed in train mode (the pool is off): gn_dsum and
            // gn_dgamma both read it back.
            self.tape.push(Op::Gn {
                gb: gbn,
                c,
                h,
                w,
                g,
                x: x.clone(),
                stats: stats.clone(),
                y: y.clone(),
            });
        }
        self.free(2 * g as u64, stats); // last read was GN_APPLY above
        y
    }

    /// Channel-axis concatenation of two NCHW tensors: `[Ca,H,W] ‖ [Cb,H,W]`
    /// → `[Ca+Cb,H,W]`.
    ///
    /// Every U-shaped net here needs it for skip connections, and before this
    /// existed **eleven** crates each registered `concat2` under a private slot
    /// constant and dispatched it inline — including three that were already
    /// building on this Builder. That is the duplication this module exists to
    /// absorb: not because the four-line dispatch is hard, but because eleven
    /// copies is eleven places to get the `[N, Ca, Cb, H, W]` param order or
    /// the output-element thread count wrong, and each copy has to be found
    /// again the next time the kernel changes.
    pub fn concat(
        &mut self,
        ca: u32,
        cb: u32,
        h: u32,
        w: u32,
        a: &DeviceBuffer,
        b: &DeviceBuffer,
    ) -> DeviceBuffer {
        let n = (ca + cb) as u64 * h as u64 * w as u64;
        let y = self.act(n);
        // `concat2` Params: [N, Ca, Cb, H, W]; one invocation per OUTPUT element.
        self.steps.push(self.gpu.step(K_CONCAT2, &[a, b, &y], &[1, ca, cb, h, w], n as u32));
        if self.train {
            self.tape.push(Op::Concat { ca, cb, hw: h * w, a: a.clone(), b: b.clone(), y: y.clone() });
        }
        y
    }

    /// `y = a·x + b·f`, reusing `edm_mix.wgsl` verbatim - one scalar pair per
    /// call, packed as `ab[2] = [a, b]` for the forward dispatch. SUPIR's
    /// `ZeroSFT`/`ZeroCrossAttn` `control_scale` lerp is exactly this shape:
    /// one host-known scalar pair, never a gradient target.
    ///
    /// `a`/`b` are host floats rather than a caller-owned buffer because they
    /// ARE host constants here (the sigma-derived EDM coefficients are the
    /// same shape) - this uploads the packed pair itself, and in train mode
    /// also the two UNPACKED one-element buffers [`Op::Mix`]'s backward needs
    /// (`scale_row` reads a per-row array, not `edm_mix`'s packed layout).
    pub fn mix(&mut self, n: u32, a: f32, b: f32, x: &DeviceBuffer, f: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act(n as u64);
        let ab = self.gpu.storage(2);
        self.gpu.write_f32(&ab, &[a, b]);
        let slot = self.mix_ids.expect("vae::blocks: Builder::set_mix_ids must precede Builder::mix").fwd;
        self.steps.push(self.gpu.step(slot, &[x, f, &ab, &y], &[n, n], n));
        if self.train {
            let av = self.gpu.storage(1);
            self.gpu.write_f32(&av, &[a]);
            let bv = self.gpu.storage(1);
            self.gpu.write_f32(&bv, &[b]);
            self.tape.push(Op::Mix { n, x: x.clone(), f: f.clone(), a: av, b: bv, y: y.clone() });
        }
        y
    }

    /// Supply the caller's transformer-stage forward slots. Must precede any
    /// [`Builder::linear`] / [`Builder::layernorm`] / [`Builder::gelu_erf`] /
    /// [`Builder::mul`] / [`Builder::add_chan`] / [`Builder::cross_attn`].
    pub fn set_xformer_ids(&mut self, ids: XformerIds) {
        self.xf = Some(ids);
    }

    fn xf(&self) -> &XformerIds {
        self.xf.as_ref().expect("vae::blocks: set_xformer_ids must precede any transformer stage")
    }

    /// `y[m,n] = x[m,k] · W[n,k]ᵀ (+ b[n])` - the diffusers `nn.Linear` shape.
    ///
    /// `bias = false` covers the bias-free attention projections. Recorded on the
    /// tape, so a train-mode builder differentiates through it; the old
    /// caller-side `push_step` did not, which is why every parameter upstream of
    /// a transformer stage used to get a silent ZERO gradient.
    pub fn linear(&mut self, prefix: &str, m: u32, k: u32, n: u32, bias: bool, x: &DeviceBuffer) -> DeviceBuffer {
        let w_name = format!("{prefix}.weight");
        let w = self.dev(&w_name);
        let y = self.act((m as u64) * (n as u64));
        let (kind, threads) = model::block::pick_gemm(m as usize, n as usize, self.xf().matmul, self.xf().matmul_reg, false);
        self.steps.push(self.gpu.step(kind, &[x, &w, &y], &[m, k, n], threads));
        let b_name = bias.then(|| format!("{prefix}.bias"));
        if let Some(bn) = &b_name {
            let bb = self.dev(bn);
            let slot = self.xf().bias_add;
            self.steps.push(self.gpu.step(slot, &[&y, &bb], &[m, n], m * n));
        }
        if self.train {
            self.tape.push(Op::Linear { w: w_name, b: b_name, m, k, n, x: x.clone(), y: y.clone() });
        }
        y
    }

    /// LayerNorm with separate `{prefix}.weight` / `{prefix}.bias`.
    pub fn layernorm(&mut self, prefix: &str, rows: u32, d: u32, eps: f32, x: &DeviceBuffer) -> DeviceBuffer {
        let (gn, bn) = (format!("{prefix}.weight"), format!("{prefix}.bias"));
        let (gamma, beta) = (self.dev(&gn), self.dev(&bn));
        let y = self.act((rows as u64) * (d as u64));
        let ids = self.xf().layernorm.ids;
        let step = model::block::layernorm_fwd(self.gpu, &ids, x, &gamma, &beta, &y, d, rows, eps);
        self.steps.push(step);
        if self.train {
            self.tape.push(Op::LayerNorm { gamma: gn, beta: bn, rows, d, eps, x: x.clone(), y: y.clone() });
        }
        y
    }

    /// Exact (erf) GELU over `n` values.
    pub fn gelu_erf(&mut self, n: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act(n as u64);
        let slot = self.xf().gelu_erf;
        self.steps.push(self.gpu.step(slot, &[x, &y], &[n], n));
        if self.train {
            self.tape.push(Op::GeluErf { n, x: x.clone(), y: y.clone() });
        }
        y
    }

    /// Elementwise product over `n` values - GEGLU's `hidden · gelu(gate)`.
    pub fn mul(&mut self, n: u32, a: &DeviceBuffer, b: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act(n as u64);
        let slot = self.xf().mul;
        self.steps.push(self.gpu.step(slot, &[a, b, &y], &[n], n));
        if self.train {
            self.tape.push(Op::Mul { n, a: a.clone(), b: b.clone(), y: y.clone() });
        }
        y
    }

    /// `y[c,hw] = x[c,hw] + v[c]` - the per-channel broadcast add SDXL's resnets
    /// inject the timestep embedding with.
    pub fn add_chan(&mut self, c: u32, hw: u32, x: &DeviceBuffer, v: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act((c as u64) * (hw as u64));
        let slot = self.xf().add_chan;
        // `add_chan_bcast` Params: [N, C, HW]; bufs [x, v[N*C], y].
        self.steps.push(self.gpu.step(slot, &[x, v, &y], &[1, c, hw], c * hw));
        if self.train {
            self.tape.push(Op::AddChan { c, hw, x: x.clone(), v: v.clone(), y: y.clone() });
        }
        y
    }

    /// Multi-head self-attention over an already-projected fused `[t, 3c]`
    /// buffer, writing the context into `ctx[t, c]`.
    ///
    /// Distinct from [`Builder::attn`], which is the VAE's own attention BLOCK
    /// (GroupNorm + a fused qkv conv + an output projection); this is just the
    /// attention itself, for a caller whose projections are `nn.Linear`s.
    ///
    /// The score/softmax slabs are allocated PER CALL. That matters in train
    /// mode: the adjoint binds the softmax the forward took, so two attention
    /// sites sharing one `probs` buffer would each differentiate against the
    /// other's - a silently wrong gradient, not a shape error. Outside train
    /// mode the activation pool hands the same pair back each time, so this
    /// costs nothing there.
    #[allow(clippy::too_many_arguments)]
    pub fn self_attn(&mut self, heads: u32, head_dim: u32, c: u32, t: u32, qkv: &DeviceBuffer, ctx: &DeviceBuffer) {
        let slab = (heads as u64) * (t as u64) * (t as u64);
        let scores = self.act(slab);
        let probs = self.act(slab);
        let a = model::block::Bidir { b: 1, t, n_heads: heads, head_dim, stride: 3 * c, q_off: 0, k_off: c, v_off: 2 * c };
        let ids = model::block::BidirIds {
            scores: K_ATTN_SCORES,
            softmax: K_ATTN_SOFTMAX,
            apply: K_ATTN_APPLY,
            // Forward-only slots: the reverse reaches the quartet through
            // `BwdIds`, never through this struct.
            dscores: usize::MAX,
            dv: usize::MAX,
            dq: usize::MAX,
            dk: usize::MAX,
        };
        for st in model::block::bidir_fwd(self.gpu, &ids, &a, qkv, &scores, &probs, ctx) {
            self.steps.push(st);
        }
        if self.train {
            self.tape.push(Op::Attn { c, t, heads, head_dim, qkv: qkv.clone(), probs: probs.clone(), y: ctx.clone() });
        }
        self.free(slab, scores);
        self.free(slab, probs);
    }

    /// Cross-attention: `tq` query rows in `q[tq, c]` against `tkv` key/value
    /// rows fused as `kv[tkv, 2c]`, writing the context into `ctx[tq, c]`.
    ///
    /// The score/softmax slabs are allocated here and RETAINED on a train-mode
    /// builder (where `free` is a no-op because the pool is disabled), because
    /// the adjoint needs the softmax the forward actually took. The three
    /// dispatches are copied verbatim from the working SDXL call site rather
    /// than re-derived - a mismatched cross-attention param list is silently
    /// wrong, not a crash.
    #[allow(clippy::too_many_arguments)]
    pub fn cross_attn(
        &mut self,
        heads: u32,
        head_dim: u32,
        c: u32,
        tq: u32,
        tkv: u32,
        q: &DeviceBuffer,
        kv: &DeviceBuffer,
        ctx: &DeviceBuffer,
    ) {
        let ids = self.xf().cross;
        let slab = (heads as u64) * (tq as u64) * (tkv as u64);
        let scores = self.act(slab);
        let probs = self.act(slab);
        // `attn_scores_cross`  Params: [bsz, heads, t_dec, t_enc, head_dim, q_stride, kv_stride, q_off, k_off]
        self.steps.push(self.gpu.step(ids.scores, &[q, kv, &scores], &[1, heads, tq, tkv, head_dim, c, 2 * c, 0, 0], heads * tq * tkv));
        // `attn_softmax_cross` Params: [bsz, heads, t_dec, t_enc]
        self.steps.push(self.gpu.step(ids.softmax, &[&scores, &probs], &[1, heads, tq, tkv], heads * tq));
        // `attn_apply_cross`   Params: [bsz, heads, t_dec, t_enc, head_dim, kv_stride, v_off, d_model]
        self.steps.push(self.gpu.step(ids.apply, &[&probs, kv, ctx], &[1, heads, tq, tkv, head_dim, 2 * c, c, c], heads * tq * head_dim));
        if self.train {
            self.tape.push(Op::Cross {
                c,
                tq,
                tkv,
                heads,
                head_dim,
                q: q.clone(),
                kv: kv.clone(),
                probs: probs.clone(),
                y: ctx.clone(),
            });
        }
        self.free(slab, scores);
        self.free(slab, probs);
    }

    /// SiLU/swish (`x·sigmoid(x)`), elementwise over `n` values.
    pub fn silu(&mut self, n: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act(n as u64);
        self.steps.push(self.gpu.step(K_SILU, &[x, &y], &[n], n));
        if self.train {
            self.tape.push(Op::Silu { n, x: x.clone(), y: y.clone() });
        }
        y
    }

    /// Elementwise sum of two `n`-length buffers.
    pub fn add(&mut self, n: u32, a: &DeviceBuffer, b: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act(n as u64);
        self.steps.push(self.gpu.step(K_ADD2, &[a, b, &y], &[n], n));
        if self.train {
            self.tape.push(Op::Add2 { n, a: a.clone(), b: b.clone(), y: y.clone() });
        }
        y
    }

    /// Nearest-neighbour 2× upsample: `[c,h,w] → [c,2h,2w]`.
    pub fn upsample(&mut self, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act((c * 2 * h * 2 * w) as u64);
        self.steps.push(self.gpu.step(K_UPSAMPLE2, &[x, &y], &[1, c, h, w], c * 4 * h * w));
        if self.train {
            self.tape.push(Op::Up2 { c, h, w, x: x.clone(), y: y.clone() });
        }
        y
    }

    /// One residual block (diffusers `ResnetBlock2D` without temb == VQGAN
    /// `ResBlock`): `x → conv2(silu(norm2(conv1(silu(norm1(x)))))) +
    /// shortcut(x)`, the shortcut a 1×1 conv (named [`BlockNames::shortcut`])
    /// when `cin != cout`, else the identity.
    pub fn resnet(
        &mut self,
        prefix: &str,
        cin: u32,
        cout: u32,
        h: u32,
        w: u32,
        x: &DeviceBuffer,
    ) -> DeviceBuffer {
        let (nin, nout) = ((cin * h * w) as u64, (cout * h * w) as u64);
        // `r` aliases the input `x` when cin==cout (a residual we must NOT free — the
        // caller owns `x`); when cin!=cout it is a fresh shortcut-conv buffer we own.
        let (r, r_owned) = if cin != cout {
            let sc = self.names.shortcut;
            (self.conv(&format!("{prefix}.{sc}"), cin, cout, 1, 0, h, w, x), true)
        } else {
            (x.clone(), false)
        };
        let n1 = self.gn(&format!("{prefix}.norm1"), cin, h, w, x);
        self.tap(format!("{prefix}.norm1"), &n1, cin * h * w);
        let s1 = self.silu(cin * h * w, &n1);
        self.free(nin, n1);
        let c1 = self.conv(&format!("{prefix}.conv1"), cin, cout, 3, 1, h, w, &s1);
        self.tap(format!("{prefix}.conv1"), &c1, cout * h * w);
        self.free(nin, s1);
        let n2 = self.gn(&format!("{prefix}.norm2"), cout, h, w, &c1);
        self.tap(format!("{prefix}.norm2"), &n2, cout * h * w);
        self.free(nout, c1);
        let s2 = self.silu(cout * h * w, &n2);
        self.free(nout, n2);
        let c2 = self.conv(&format!("{prefix}.conv2"), cout, cout, 3, 1, h, w, &s2);
        self.tap(format!("{prefix}.conv2"), &c2, cout * h * w);
        self.free(nout, s2);
        if r_owned {
            let sc = self.names.shortcut;
            self.tap(format!("{prefix}.{sc}"), &r, cout * h * w);
        }
        let out = self.add(cout * h * w, &c2, &r); // last read of c2 and r
        self.free(nout, c2);
        if r_owned {
            self.free(nout, r);
        }
        self.tap(prefix.to_string(), &out, cout * h * w);
        out
    }

    /// Self-attention over the spatial positions:
    /// `x + proj(attn(qkv(norm(x))))`, softmax over the key axis. `q/k/v` are
    /// one 1×1 qkv conv so the bidir attention trio applies unchanged.
    ///
    /// Defaults to the VAE shape (diffusers `Attention` with
    /// `residual_connection=True` == VQGAN `AttnBlock`): a single head of
    /// width C, scale `C^-0.5`, residual added to the **pre-norm** input.
    /// [`set_attn_head_dim`](Builder::set_attn_head_dim) splits it into heads,
    /// and [`BlockNames::attn_residual_normed`] moves the residual onto the
    /// norm output for DIAMOND.
    pub fn attn(&mut self, prefix: &str, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let t = h * w;
        let nnorm = self.names.attn_norm;
        let nproj = self.names.attn_proj;
        let normed = self.gn(&format!("{prefix}.{nnorm}"), c, h, w, x);
        // Taps are named after the tensor they follow, so they must use the
        // ARCHITECTURE's leaf name (as `resnet` does for its shortcut) — a tap
        // called `.norm` on a diffusers graph whose tensor is `.group_norm`
        // sends the next debugger to the wrong module.
        self.tap(format!("{prefix}.{nnorm}"), &normed, c * t);

        // The qkv projection, as one [3C,C,1,1] weight + [3C] bias. Either the
        // checkpoint already ships it fused (DIAMOND's `qkv_proj`), or it holds
        // three [C,C] linears — each a [C,C,1,1] conv — to concatenate here.
        let (mut qkv_w, mut qkv_b) = if let Some(nqkv) = self.names.attn_qkv {
            let (_, w) = self.get(&format!("{prefix}.{nqkv}.weight"));
            let (_, b) = self.get(&format!("{prefix}.{nqkv}.bias"));
            (w.clone(), b.clone())
        } else {
            let (nq, nk, nv) = (self.names.attn_q, self.names.attn_k, self.names.attn_v);
            let (_, qw) = self.get(&format!("{prefix}.{nq}.weight"));
            let (_, kw) = self.get(&format!("{prefix}.{nk}.weight"));
            let (_, vw) = self.get(&format!("{prefix}.{nv}.weight"));
            let mut qkv_w = Vec::with_capacity(qw.len() * 3);
            qkv_w.extend_from_slice(qw);
            qkv_w.extend_from_slice(kw);
            qkv_w.extend_from_slice(vw);
            let (_, qb) = self.get(&format!("{prefix}.{nq}.bias"));
            let (_, kb) = self.get(&format!("{prefix}.{nk}.bias"));
            let (_, vb) = self.get(&format!("{prefix}.{nv}.bias"));
            let mut qkv_b = Vec::with_capacity(qb.len() * 3);
            qkv_b.extend_from_slice(qb);
            qkv_b.extend_from_slice(kb);
            qkv_b.extend_from_slice(vb);
            (qkv_w, qkv_b)
        };
        // Either way the layout is [q | k | v], so a third of each is `q`.
        assert_eq!(qkv_w.len() % 3, 0, "{prefix}: qkv weight is not 3 blocks");
        assert_eq!(qkv_b.len() % 3, 0, "{prefix}: qkv bias is not 3 blocks");
        // GEMM path (below): the 1/√C attention scale lives in
        // `attn_scores_bidir`'s epilogue, and a plain GEMM has no epilogue — so
        // fold it into `q` instead. `q = (Wx+b)/√C = (W/√C)x + b/√C`
        // exactly; only the fp32 rounding of the two orders differs (≈1 ulp on
        // a score of O(1), invisible through the softmax).
        //
        // Train mode takes the per-element trio (its adjoints are the shipped
        // `attn_bwd_*_bidir` quartet), so the fold must be off there too — a
        // folded `qkv.w` would make the reported gradient that of `W/√C`.
        //
        // Multi-head is excluded as well: both GEMMs contract over the whole of
        // `C`, which is only the same arithmetic when there is one head.
        let head_dim = self.attn_head_dim.unwrap_or(c);
        let heads = (c / head_dim).max(1);
        let gemm_attn = self.coop && !self.train && heads == 1;
        let qn = qkv_w.len() / 3;
        let qbn = qkv_b.len() / 3;
        if gemm_attn {
            let sc = 1.0f32 / (head_dim as f32).sqrt();
            for v in qkv_w[..qn].iter_mut() {
                *v *= sc;
            }
            for v in qkv_b[..qbn].iter_mut() {
                *v *= sc;
            }
        }
        // Name the uploaded weight after the real tensor when the checkpoint
        // ships it fused: that name is what the tape reports a gradient under,
        // and `qkv_proj.weight` is a parameter that actually exists. The
        // concatenated case has no such tensor, so it keeps a synthetic name.
        let (wn, bn) = match self.names.attn_qkv {
            Some(nqkv) => (format!("{prefix}.{nqkv}.weight"), format!("{prefix}.{nqkv}.bias")),
            None => (format!("{prefix}.qkv.w"), format!("{prefix}.qkv.b")),
        };
        let qkv_wd = self.dev_fused(&wn, &qkv_w);
        let qkv_bd = self.dev_fused(&bn, &qkv_b);

        // qkv 1×1 conv: [C,h,w] → [3C,h,w].
        let qkv_chw =
            self.conv_direct(wn, bn, &qkv_wd, &qkv_bd, c, 3 * c, 1, 1, 0, h, w, h, w, &normed);
        // When the residual adds the normed tensor, the qkv conv is NOT its
        // last read — it has to survive to the final add.
        let residual_normed = self.names.attn_residual_normed;
        if !residual_normed {
            self.free((c * t) as u64, normed.clone());
        }

        let attn_rows = if gemm_attn {
            // ---- attention as two GEMMs -----------------------------------
            // The per-element trio gives one thread per (i,j) score, each
            // looping head_dim with its `k` reads a whole row apart: at the
            // FLUX.2 mid block (T = 4096, C = 512) `attn_scores_bidir`
            // measured a double-digit share of a 512² decode while achieving a
            // fraction of one percent of the card's peak for the FLOPs it
            // actually does. Both contractions are plain matmuls at shapes
            // `matmul_reg3` runs far closer to peak on, so express them as
            // such. The qkv conv already emits **channel-major**
            // [3C, T], which is qᵀ/kᵀ/vᵀ — so `v` needs no transpose at all
            // (it is directly the `[n, k]` operand of the apply GEMM), and
            // q/k need one cheap `nchw_nlc` each.
            let q_nlc = self.act((c * t) as u64);
            let k_nlc = self.act((c * t) as u64);
            for (i, dst) in [&q_nlc, &k_nlc].into_iter().enumerate() {
                let off = (i as u64) * (c * t) as u64;
                self.steps.push(self.gpu.step_sliced(
                    K_NCHW_NLC,
                    &[&qkv_chw, dst],
                    &[(off, (c * t) as u64), (0, 0)],
                    &[c * t, c, t],
                    c * t,
                ));
            }
            // scores[T,T] = q[T,C] · k[T,C]ᵀ  (the 1/√C is folded into q)
            let scores = self.act((t * t) as u64);
            self.steps.push(self.gpu.step(
                K_MATMUL,
                &[&q_nlc, &k_nlc, &scores],
                &[t, c, t],
                t.div_ceil(128) * t.div_ceil(128) * 256,
            ));
            self.free((c * t) as u64, q_nlc);
            self.free((c * t) as u64, k_nlc);
            let probs = self.act((t * t) as u64);
            self.steps.push(self.gpu.step(K_ATTN_SOFTMAX, &[&scores, &probs], &[1, 1, t], t));
            self.free((t * t) as u64, scores);
            // ctx[T,C] = probs[T,T] · v[T,C], with vᵀ = the third channel block
            // of the conv output, read in place as the [n=C, k=T] operand.
            let rows = self.act((t * c) as u64);
            self.steps.push(self.gpu.step_sliced(
                K_MATMUL,
                &[&probs, &qkv_chw, &rows],
                &[(0, 0), (2 * (c * t) as u64, (c * t) as u64), (0, 0)],
                &[t, t, c],
                t.div_ceil(128) * c.div_ceil(128) * 256,
            ));
            self.free((t * t) as u64, probs);
            self.free((3 * c * t) as u64, qkv_chw);
            rows
        } else {
            // NCHW [3C,h,w] → NLC rows [T, 3C].
            let qkv = self.nchw_to_rows(3 * c, t, &qkv_chw);
            self.free((3 * c * t) as u64, qkv_chw);

            // `heads` heads of `head_dim` each; the kernel applies the
            // 1/√head_dim scale. Defaults to one head of width C.
            let scores = self.act((heads * t * t) as u64);
            self.steps.push(self.gpu.step(
                K_ATTN_SCORES,
                &[&qkv, &scores],
                &[1, heads, t, head_dim, 3 * c, 0, c],
                heads * t * t,
            ));
            let probs = self.act((heads * t * t) as u64);
            self.steps.push(self.gpu.step(
                K_ATTN_SOFTMAX,
                &[&scores, &probs],
                &[1, heads, t],
                heads * t,
            ));
            self.free((heads * t * t) as u64, scores);
            let rows = self.act((t * c) as u64);
            self.steps.push(self.gpu.step(
                K_ATTN_APPLY,
                &[&probs, &qkv, &rows], // last read of both probs and qkv
                &[1, heads, t, head_dim, 3 * c, 2 * c, c],
                heads * t * head_dim,
            ));
            if self.train {
                // `probs` and `qkv` stay live: `attn_bwd_dscores_bidir` /
                // `_dv` / `_dq` / `_dk` read both back (no softmax recompute).
                self.tape.push(Op::Attn {
                    c,
                    t,
                    heads,
                    head_dim,
                    qkv: qkv.clone(),
                    probs: probs.clone(),
                    y: rows.clone(),
                });
            }
            self.free((heads * t * t) as u64, probs);
            self.free((3 * c * t) as u64, qkv);
            rows
        };
        // NLC rows [T, C] → NCHW [C,h,w].
        let attn_chw = self.rows_to_nchw(c, t, &attn_rows);
        self.free((t * c) as u64, attn_rows);

        let proj = self.conv(&format!("{prefix}.{nproj}"), c, c, 1, 0, h, w, &attn_chw);
        self.tap(format!("{prefix}.{nproj}"), &proj, c * t);
        self.free((c * t) as u64, attn_chw);
        // `x` is caller-owned; `normed` is ours, and this is its last read.
        let out = match residual_normed {
            true => {
                let y = self.add(c * h * w, &normed, &proj);
                self.free((c * t) as u64, normed);
                y
            }
            false => self.add(c * h * w, x, &proj),
        };
        self.free((c * h * w) as u64, proj);
        self.tap(prefix.to_string(), &out, c * h * w);
        out
    }

    /// NCHW `[c,h,w]` → NLC rows `[h·w, c]` (the layout the codebook search and
    /// any per-position linear want). Exposed because `vqgan`'s quantizer needs
    /// it outside a block.
    pub fn nchw_to_rows(&mut self, c: u32, hw: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act((c * hw) as u64);
        self.steps.push(self.gpu.step(K_NCHW_NLC, &[x, &y], &[c * hw, c, hw], c * hw));
        if self.train {
            self.tape.push(Op::NchwNlc { c, hw, x: x.clone(), y: y.clone() });
        }
        y
    }

    /// NLC rows `[h·w, c]` → NCHW `[c,h,w]` (the exact inverse of
    /// [`Builder::nchw_to_rows`]).
    pub fn rows_to_nchw(&mut self, c: u32, hw: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act((c * hw) as u64);
        self.steps.push(self.gpu.step(K_NLC_NCHW, &[x, &y], &[c * hw, c, hw], c * hw));
        if self.train {
            self.tape.push(Op::NlcNchw { c, hw, x: x.clone(), y: y.clone() });
        }
        y
    }
}
