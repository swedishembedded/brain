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

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use std::collections::HashMap;

pub mod grad;

// Kernel-table indices (order matches KERNELS).
const K_CONV: usize = 0;
const K_GN_STATS: usize = 1;
const K_GN_APPLY: usize = 2;
const K_SILU: usize = 3;
const K_ADD2: usize = 4;
const K_UPSAMPLE2: usize = 5;
const K_NCHW_NLC: usize = 6;
const K_NLC_NCHW: usize = 7;
const K_ATTN_SCORES: usize = 8;
const K_ATTN_SOFTMAX: usize = 9;
const K_ATTN_APPLY: usize = 10;
const K_GN_STATS_WG: usize = 11;
const K_MATMUL: usize = 12;
const K_IM2COL_AT: usize = 13;
const K_NLC_BIAS_NCHW: usize = 14;

/// The `add2` slot inside [`KERNELS`]. Public for the same reason
/// [`grad::BwdIds::axpy`] is: a caller stitching extra graph onto these blocks
/// must reuse this pipeline rather than register a second `add2`, which the CPU
/// backend's JIT rejects as a duplicate definition.
pub const ADD2_SLOT: usize = K_ADD2;

/// The block builder's kernel set, in slot order. Public so a profiler can name
/// the kernel behind each recorded [`Step`] (`flux2_bench vae`), and so a crate
/// that needs extra kernels alongside these can build its set with
/// [`kernels_with`] instead of restating them.
pub const KERNELS: [(&str, &str); 15] = [
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("gn_stats", kernels::GN_STATS),
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
];

/// Slot index the first caller-supplied kernel gets when a kernel set is built
/// with [`kernels_with`] — i.e. `KERNELS.len()`. A `Builder` addresses slots
/// `0..NEXT_SLOT`, so a caller's own kernels must start here.
pub const NEXT_SLOT: usize = KERNELS.len();

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
/// cooperative twin anywhere in the tree — that is the documented §C.2 perf gap
/// in `docs/kernel-checklist.md`, NOT a correctness gate, because none of them
/// uses `workgroupBarrier()` and all three are exact on `backend-cpu`.
pub const BWD_KERNELS: [(&str, &str); 15] = [
    ("conv2d_dx", kernels::CONV2D_DX),
    ("conv2d_dw", kernels::CONV2D_DW),
    ("bias_grad", kernels::BIAS_GRAD),
    ("silu_bwd", kernels::SILU_BWD),
    ("scale_chan", kernels::SCALE_CHAN),
    ("gn_dsum", kernels::GN_DSUM),
    ("gn_dx", kernels::GN_DX),
    ("gn_dgamma", kernels::GN_DGAMMA),
    ("gn_dbeta", kernels::GN_DBETA),
    ("upsample2_dx", kernels::UPSAMPLE2_DX),
    ("axpy", kernels::AXPY),
    ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
    ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
    ("attn_bwd_dq_bidir", kernels::ATTN_BWD_DQ_BIDIR),
    ("attn_bwd_dk_bidir", kernels::ATTN_BWD_DK_BIDIR),
];

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
        }
    }
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
    Attn { c: u32, t: u32, qkv: DeviceBuffer, probs: DeviceBuffer, y: DeviceBuffer },
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
    /// The single im2col scratch (`length, buffer`) shared by every lowered
    /// conv, grown on demand. Bounded by [`col_budget_floats`] — a whole-image
    /// im2col operand exceeds the P40's 2047 MiB binding limit, so the GEMM is
    /// chunked over spatial positions instead (see `im2col_at.wgsl`).
    col: Option<(u64, DeviceBuffer)>,
}

/// Ceiling on the im2col scratch, in f32 words (512 MiB). The lowered conv
/// processes `floor(budget / CinKK)` output positions per GEMM, so this trades
/// scratch for the number of chunks; at 512² the largest operand would be
/// 2.4 GB unchunked, which is both unbindable and hostile to a card shared with
/// a resident DiT. Override with `BRAIN_VAE_COL_MIB`.
fn col_budget_floats() -> u64 {
    let mib: u64 = std::env::var("BRAIN_VAE_COL_MIB").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    mib * 1024 * 1024 / 4
}

/// Minimum output channels for the lowered conv: `matmul_reg3` computes a
/// 128-wide column tile, so a conv with fewer output channels pays for a full
/// tile and wins nothing (the FLUX.2 `conv_out`, Cout = 3, is 42x wasted). It
/// stays on the direct register-tiled conv.
const GEMM_CONV_MIN_COUT: u32 = 128;

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
            tape: Vec::new(),
            wmemo: HashMap::new(),
            worder: Vec::new(),
            pool: HashMap::new(),
            taps_on,
            coop: gpu.caps().workgroup_reductions,
            col: None,
        }
    }

    /// Record the reverse-mode tape (see [`Builder::train`]). Set this BEFORE
    /// recording any block — it changes both what is kept and which lowering
    /// each block picks.
    pub fn set_train(&mut self, on: bool) {
        assert!(self.steps.is_empty(), "vae::blocks: set_train must precede the first block");
        self.train = on;
    }

    /// The recorded forward tape + weight buffers, for [`grad::Trace::backward`].
    /// Empty unless [`Builder::set_train`] was called.
    pub fn trace(&self) -> grad::Trace {
        grad::Trace::new(self.tape.clone(), self.worder.clone(), &self.wmemo)
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
    pub fn dev(&mut self, name: &str) -> DeviceBuffer {
        if let Some(b) = self.wmemo.get(name) {
            return b.clone();
        }
        let gpu = self.gpu;
        let (buf, n) = {
            let data = &self.get(name).1;
            (gpu.storage_init(name, data), data.len() as u64)
        };
        self.remember(name, buf, n)
    }

    /// Upload host data the builder synthesised (a fused `gamma|beta` or
    /// `q|k|v`) under a name of its own, memoized like [`Builder::dev`]. That
    /// fused buffer is the trainable tensor: `gn_dgamma`/`gn_dbeta` write the
    /// matching fused `dgb[2C]`, and one `conv2d_dw` covers the fused qkv.
    fn dev_fused(&mut self, name: &str, data: &[f32]) -> DeviceBuffer {
        if let Some(b) = self.wmemo.get(name) {
            return b.clone();
        }
        let buf = self.gpu.storage_init(name, data);
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
        self.gpu.storage(len)
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

    /// Conv with an explicit stride and output size. Two lowerings:
    ///
    /// * **direct** — `conv_bias_reg`, the 8x4 register-tiled kernel. Measured
    ///   on a P40 across every FLUX.2 VAE decode shape: a flat **~700 GFLOP/s,
    ///   6% of the card's fp32 peak**, and it was 3535 ms of a 3600 ms decode.
    ///   Its ceiling is structural — 12 global loads per 32 FMAs is
    ///   0.75 byte/FLOP, a 461 GFLOP/s roofline that caching stretches to ~700.
    /// * **lowered** (`self.coop`, `cout >= GEMM_CONV_MIN_COUT`) — `im2col_at` +
    ///   `matmul_reg3` + `nlc_bias_nchw`, i.e. `y[HW, Cout] = col[HW, CinKK] ·
    ///   Wᵀ`, which runs at the GEMM's ~34% of peak. This is the trade
    ///   `docs/performance/overview.md` scoped to "a compute-bound discrete
    ///   GPU" and `docs/performance/p40.md` already took for YOLO's convs; the
    ///   P40 is that GPU. The transposed orientation (positions as GEMM ROWS)
    ///   is what makes it chunkable: a spatial chunk is a contiguous row range
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
        if self.train || !(self.coop && cout >= GEMM_CONV_MIN_COUT && hw >= 128) {
            return self.conv_direct(wn, bn, &wgt, &bias, cin, cout, k, stride, pad, h, w, ho, wo, x);
        }
        let y = self.act((cout * ho * wo) as u64);
        {
            // Positions per GEMM: a multiple of the 128-row tile, inside the
            // scratch budget, at least one tile.
            let budget = col_budget_floats();
            let chunk = (((budget / cinkk as u64) / 128) * 128).clamp(128, hw as u64) as u32;
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
        let g = self.groups;
        let stats = self.act(2 * g as u64);
        let y = self.act((c * h * w) as u64);
        // Statistics: one WORKGROUP per group where the device can run a
        // workgroup reduction (`gn_stats_wg`), else the per-group reference
        // kernel. `gn_stats` dispatches `g` = 32 *invocations* for up to 33 M
        // elements — measured at 35% of a 512² FLUX.2 VAE decode; the
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
            self.steps.push(self.gpu.step(
                K_GN_STATS,
                &[x, &stats],
                &[1, c, h, w, g, f(self.eps)],
                g,
            ));
        }
        self.steps.push(self.gpu.step(
            K_GN_APPLY,
            &[x, &stats, &gb, &y],
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

    /// Single-head self-attention over the spatial positions (diffusers
    /// `Attention` with `residual_connection=True` == VQGAN `AttnBlock`):
    /// `x + proj(attn(qkv(norm(x))))`, head_dim = C, scale `C^-0.5`, softmax
    /// over the key axis, residual added to the **pre-norm** input. `q/k/v`
    /// are fused into one 1×1 qkv conv so the bidir attention trio applies
    /// unchanged.
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

        // Fuse q/k/v (each [C,C] linear = [C,C,1,1] conv) into one
        // [3C,C,1,1] qkv conv weight + [3C] bias.
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
        // GEMM path (below): the 1/√C attention scale lives in
        // `attn_scores_bidir`'s epilogue, and a plain GEMM has no epilogue — so
        // fold it into `q` instead. `q = (Wx+b)/√C = (W/√C)x + b/√C`
        // exactly; only the fp32 rounding of the two orders differs (≈1 ulp on
        // a score of O(1), invisible through the softmax).
        //
        // Train mode takes the per-element trio (its adjoints are the shipped
        // `attn_bwd_*_bidir` quartet), so the fold must be off there too — a
        // folded `qkv.w` would make the reported gradient that of `W/√C`.
        let gemm_attn = self.coop && !self.train;
        let qn = qw.len();
        let qbn = qb.len();
        if gemm_attn {
            let sc = 1.0f32 / (c as f32).sqrt();
            for v in qkv_w[..qn].iter_mut() {
                *v *= sc;
            }
            for v in qkv_b[..qbn].iter_mut() {
                *v *= sc;
            }
        }
        let (wn, bn) = (format!("{prefix}.qkv.w"), format!("{prefix}.qkv.b"));
        let qkv_wd = self.dev_fused(&wn, &qkv_w);
        let qkv_bd = self.dev_fused(&bn, &qkv_b);

        // qkv 1×1 conv: [C,h,w] → [3C,h,w].
        let qkv_chw =
            self.conv_direct(wn, bn, &qkv_wd, &qkv_bd, c, 3 * c, 1, 1, 0, h, w, h, w, &normed);
        self.free((c * t) as u64, normed); // last read was the qkv conv

        let attn_rows = if gemm_attn {
            // ---- attention as two GEMMs -----------------------------------
            // The per-element trio gives one thread per (i,j) score, each
            // looping head_dim with its `k` reads a whole row apart: at the
            // FLUX.2 mid block (T = 4096, C = 512) `attn_scores_bidir`
            // measured **562 ms = 13% of a 512² decode** for 17 GFLOP =
            // 30 GFLOP/s, 0.26% of the P40's peak. Both contractions are plain
            // matmuls at shapes `matmul_reg3` runs at ~34% of peak, so express
            // them as such. The qkv conv already emits **channel-major**
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

            // Single head, head_dim = C, scale 1/√C (applied in the kernel).
            let scores = self.act((t * t) as u64);
            self.steps.push(self.gpu.step(
                K_ATTN_SCORES,
                &[&qkv, &scores],
                &[1, 1, t, c, 3 * c, 0, c],
                t * t,
            ));
            let probs = self.act((t * t) as u64);
            self.steps.push(self.gpu.step(K_ATTN_SOFTMAX, &[&scores, &probs], &[1, 1, t], t));
            self.free((t * t) as u64, scores);
            let rows = self.act((t * c) as u64);
            self.steps.push(self.gpu.step(
                K_ATTN_APPLY,
                &[&probs, &qkv, &rows], // last read of both probs and qkv
                &[1, 1, t, c, 3 * c, 2 * c, c],
                t * c,
            ));
            if self.train {
                // `probs` and `qkv` stay live: `attn_bwd_dscores_bidir` /
                // `_dv` / `_dq` / `_dk` read both back (no softmax recompute).
                self.tape.push(Op::Attn {
                    c,
                    t,
                    qkv: qkv.clone(),
                    probs: probs.clone(),
                    y: rows.clone(),
                });
            }
            self.free((t * t) as u64, probs);
            self.free((3 * c * t) as u64, qkv);
            rows
        };
        // NLC rows [T, C] → NCHW [C,h,w].
        let attn_chw = self.rows_to_nchw(c, t, &attn_rows);
        self.free((t * c) as u64, attn_rows);

        let proj = self.conv(&format!("{prefix}.{nproj}"), c, c, 1, 0, h, w, &attn_chw);
        self.tap(format!("{prefix}.{nproj}"), &proj, c * t);
        self.free((c * t) as u64, attn_chw);
        let out = self.add(c * h * w, x, &proj); // x is the residual input (caller-owned)
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
