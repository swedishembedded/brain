// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! 1D convolution Step-builders over the shared WGSL engine, plus tiny CPU
//! reference implementations used as test oracles.
//!
//! The codec conv encoder/decoder, the ECAPA speaker encoder, and the GAN
//! vocoder are all stacks of (transposed) 1D convolutions; these builders are
//! the audio analogue of `model::block`'s RMSNorm/RoPE/GQA/SwiGLU helpers. They
//! are pure dispatch assembly — shapes + buffers in, `Step`s out — and carry no
//! ParamStore / model concerns. Both NCL convolutions use grouping + dilation;
//! causal convs are expressed as a LEFT pad of `dilation*(K-1)` with `lo == l`.
//!
//! # Two lowerings, one selector
//!
//! [`conv1d_fwd`] / [`convtr1d_fwd`] dispatch the DIRECT kernels
//! (`conv1d.wgsl`, `convtr1d.wgsl`): one thread per output element with a
//! serial reduction over `Cin*K`. That is the wrong kernel rather than a slow
//! one - the pair measured at a low single-digit percent and a fraction of one
//! percent of the card's compute roof respectively, and between them they were
//! essentially all of the MiniMax-Music-3 vocoder's stage time.
//!
//! [`conv1d_bias_fwd`] / [`convtr1d_bias_fwd`] are the seam that also offers
//! the GEMM lowering, choosing per shape and per device through
//! `backend_api::select` ([`Op::Conv1d`] / [`Op::ConvTranspose1d`]). They are
//! ADDITIVE - the direct entry points, [`ConvKernels`] and both backward
//! builders are untouched, so no existing caller has to change - and the
//! choice lives here rather than at a call site, which is what lets the dozen
//! crates convolving in 1D inherit it by switching one function.
//!
//! Both lowerings consume the checkpoint's NATIVE weight layout: `[Cout,
//! Cin/G, K]` is `[Cout, Cin*K]` row-major, and `[Cin, Cout/G, K]` is `[Cin,
//! Cout*K]`. No transpose, no permute, on either side.

use gpu_core::select::{DefaultSelector, KernelSelector, KernelVariant, Op, OpShape};
use gpu_core::{DeviceBuffer, Gpu, Step};

/// Shape + hyperparameters of a 1D convolution (forward and both gradients share
/// this, since the kernels take an identical 10-word uniform).
#[derive(Clone, Copy, Debug)]
pub struct Conv1d {
    pub n: u32,
    pub cin: u32,
    pub l: u32,
    pub cout: u32,
    pub k: u32,
    pub stride: u32,
    pub pad: u32,
    pub dilation: u32,
    pub groups: u32,
    pub lo: u32,
}

/// Kernel-pipeline indices for the conv family a model supplies from its own
/// PIPELINES list (forward + input-grad + weight-grad).
#[derive(Clone, Copy)]
pub struct ConvKernels {
    pub fwd: usize,
    pub dx: usize,
    pub dw: usize,
}

impl Conv1d {
    fn params(&self) -> [u32; 10] {
        [self.n, self.cin, self.l, self.cout, self.k, self.stride, self.pad, self.dilation, self.groups, self.lo]
    }

    /// Output length of a standard (non-transposed) conv with the given low/high
    /// padding. The kernels only apply the LOW pad explicitly (high-side taps
    /// past the input are skipped = zero pad), so callers requesting symmetric
    /// padding pass `pad = pad_low` and size `lo` with this helper.
    pub fn out_len(l: u32, k: u32, stride: u32, pad_low: u32, pad_high: u32, dilation: u32) -> u32 {
        (l + pad_low + pad_high - dilation * (k - 1) - 1) / stride + 1
    }

    /// Output length of a transposed conv (upsampling).
    pub fn out_len_transposed(l: u32, k: u32, stride: u32, pad: u32, out_pad: u32, dilation: u32) -> u32 {
        (l - 1) * stride + dilation * (k - 1) + out_pad + 1 - 2 * pad
    }

    pub fn weight_numel(&self) -> usize {
        (self.cout * (self.cin / self.groups) * self.k) as usize
    }
    pub fn weight_numel_transposed(&self) -> usize {
        (self.cin * (self.cout / self.groups) * self.k) as usize
    }
}

/// `y = conv1d(x, w)` — `x:[N,Cin,L]`, `w:[Cout,Cin/G,K]`, `y:[N,Cout,Lo]`.
pub fn conv1d_fwd(g: &Gpu, k: &ConvKernels, c: &Conv1d, x: &DeviceBuffer, w: &DeviceBuffer, y: &DeviceBuffer) -> Step {
    g.step(k.fwd, &[x, w, y], &c.params(), c.n * c.cout * c.lo)
}

/// conv1d backward: input grad `dx` (overwritten) and/or weight grad `dw`
/// (accumulated — zero it via `submit`'s `clears` first). Pass `None` to skip.
pub fn conv1d_bwd(
    g: &Gpu,
    k: &ConvKernels,
    c: &Conv1d,
    dy: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    dx: Option<&DeviceBuffer>,
    dw: Option<&DeviceBuffer>,
) -> Vec<Step> {
    let mut s = Vec::new();
    if let Some(dx) = dx {
        s.push(g.step(k.dx, &[dy, w, dx], &c.params(), c.n * c.cin * c.l));
    }
    if let Some(dw) = dw {
        s.push(g.step(k.dw, &[dy, x, dw], &c.params(), c.cout * (c.cin / c.groups) * c.k));
    }
    s
}

/// `y = conv_transpose1d(x, w)` — `x:[N,Cin,L]`, `w:[Cin,Cout/G,K]`,
/// `y:[N,Cout,Lo]`.
pub fn convtr1d_fwd(g: &Gpu, k: &ConvKernels, c: &Conv1d, x: &DeviceBuffer, w: &DeviceBuffer, y: &DeviceBuffer) -> Step {
    g.step(k.fwd, &[x, w, y], &c.params(), c.n * c.cout * c.lo)
}

/// Transposed-conv backward (mirrors [`conv1d_bwd`]; weight is `[Cin,Cout/G,K]`).
pub fn convtr1d_bwd(
    g: &Gpu,
    k: &ConvKernels,
    c: &Conv1d,
    dy: &DeviceBuffer,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    dx: Option<&DeviceBuffer>,
    dw: Option<&DeviceBuffer>,
) -> Vec<Step> {
    let mut s = Vec::new();
    if let Some(dx) = dx {
        s.push(g.step(k.dx, &[dy, w, dx], &c.params(), c.n * c.cin * c.l));
    }
    if let Some(dw) = dw {
        s.push(g.step(k.dw, &[dy, x, dw], &c.params(), c.cin * (c.cout / c.groups) * c.k));
    }
    s
}

// ---------------------------------------------------------------------------
// The GEMM lowering - an ADDITIVE second seam, `Conv1d` and `ConvKernels`
// unchanged. See the module doc's "Two lowerings" section.
// ---------------------------------------------------------------------------

/// Kernel-pipeline indices for the GEMM lowering, on top of the direct
/// [`ConvKernels`] a model already registers.
///
/// Additive on purpose: `ConvKernels` is the contract a dozen crates
/// (`mimi`, `lfm2`, `ecapatdnn`, `qwen35*`'s GDN, `ltxv`, `campplus`, …) and
/// every conv backward already pass, so widening *it* would force an edit in
/// each of them for a forward-only win. A caller opts in by registering the
/// six extra pipelines and calling [`conv1d_bias_fwd`] /
/// [`convtr1d_bias_fwd`]; one that does not keeps the direct kernels and
/// compiles unchanged.
///
/// Every slot must name the kernel in the comment beside it - a mismatched
/// pipeline index here is silently wrong output, not a crash.
#[derive(Clone, Copy)]
pub struct ConvGemmKernels {
    /// The direct kernels, used when the selector says so and as the
    /// structural fallback on any device or shape the lowering cannot serve.
    pub direct: ConvKernels,
    /// `add_chan_inplace` - the per-channel bias for the direct path.
    pub bias: usize,
    /// `im2col1d_at`.
    pub im2col: usize,
    /// `matmul_reg3` - `out[m,n] = x[m,k] · W[n,k]ᵀ`.
    pub matmul: usize,
    /// `matmul_dx_reg` - `out[m,k] = a[m,n] · b[n,k]` (NN, assigns).
    pub matmul_nn: usize,
    /// `matmul_dw_reg_splitk` - `out[n,k] = sum_m a[m,n]·b[m,k]` (TN,
    /// assigns; `matmul_dw_reg` is the same GEMM but ACCUMULATES, which a
    /// reused scratch buffer cannot use).
    pub matmul_tn: usize,
    /// `nlc_bias_nchw` - the transpose + bias epilogue, shared with the 2D
    /// and 3D lowerings.
    pub nlc_bias: usize,
    /// `col2im1d_bias`.
    pub col2im: usize,
}

/// Reusable device scratch for the lowered convolutions, owned by the caller
/// and shared by every conv in one recorded graph.
///
/// Two slots, grown on demand and never shrunk:
///
/// * `col` - the chunked `im2col1d_at` operand, bounded by
///   [`gpu_core::lower::col_budget_floats`];
/// * `out` - the GEMM's output before the epilogue (`[Lo, Cout]` for
///   `conv1d`, `[Cout*K, L]` for `convtr1d`). The two never overlap in time,
///   so one buffer serves both and the peak is the larger.
///
/// Reuse across convs is safe because every dispatch in a recorded pass runs
/// in submit order with the backend's inter-dispatch barriers: the epilogue
/// of conv *i* reads the scratch before the GEMM of conv *i+1* writes it.
/// This is the same contract `vae::blocks`'s single `col_buf` relies on. What
/// it does NOT survive is a caller reordering steps, which nothing here does.
pub struct ConvScratch {
    col: Option<(u64, DeviceBuffer)>,
    out: Option<(u64, DeviceBuffer)>,
    budget_mib: u64,
}

impl Default for ConvScratch {
    fn default() -> ConvScratch {
        ConvScratch { col: None, out: None, budget_mib: COL_BUDGET_MIB }
    }
}

impl ConvScratch {
    pub fn new() -> ConvScratch {
        ConvScratch::default()
    }

    /// The same scratch with a different `col` ceiling, in MiB.
    ///
    /// Exists for two reasons, both real: a caller sharing a card with a
    /// resident model has less to spend than the default assumes, and a test
    /// needs to reach the multi-chunk path without allocating half a gigabyte.
    /// The second is not cosmetic - a single-chunk test cannot see a chunking
    /// bug at all (a mutation that ignored `pos0` passed the whole
    /// shape-coverage suite until a case forced three chunks).
    ///
    /// `BRAIN_CONV_COL_MIB` still wins over this, since an operator capping
    /// device memory means it.
    pub fn with_budget_mib(mib: u64) -> ConvScratch {
        ConvScratch { budget_mib: mib.max(1), ..ConvScratch::default() }
    }

    fn grow(slot: &mut Option<(u64, DeviceBuffer)>, g: &Gpu, need: u64) -> DeviceBuffer {
        if let Some((len, b)) = slot {
            if *len >= need {
                return b.clone();
            }
        }
        let b = g.storage(need);
        *slot = Some((need, b.clone()));
        b
    }
}

/// Scratch budget for the 1D `col` operand, in MiB. Shared knob - see
/// [`gpu_core::lower::col_budget_floats`].
///
/// # 128, swept - not the VAE lowering's 512
///
/// The chunk size trades GEMM efficiency (bigger chunks fill more of the
/// 128-row tile and amortise the dispatch) against a scratch buffer that is
/// allocated per recorded graph. Swept with `mm3_bench vocoder 689`,
/// best-of-5, reading the DEVICE-time sum (the whole-pass number on that stage
/// is dominated by a host allocation gap and is far too noisy to read an
/// effect this small off). Re-run that bench rather than trusting a figure
/// here; what it showed is that device time is flat from 96 MiB upward and
/// climbs steadily below it (worst at 32 MiB), so 128
/// is the smallest budget that costs nothing - which matters, because this
/// scratch is live beside a decoder that already peaks in the tens of
/// gigabytes. The VAE lowering keeps 512 for its own 2D operands.
const COL_BUDGET_MIB: u64 = 128;

/// The `matmul_reg*` family's tile edge. Both the GEMM row chunking and the
/// workgroup count derive from it.
const GEMM_TILE: u32 = 128;

/// Storage-buffer binding offsets are byte-aligned by the device's
/// `min_storage_buffer_offset_alignment`. WebGPU's *default* limit is 256
/// bytes and a device may only report a smaller one, so 64 f32 words is the
/// safe requirement on any backend without querying a limit `gpu_core` does
/// not expose. It only bites for `N > 1`, where each batch row is bound as a
/// sub-range; an unaligned row count keeps that conv on the direct kernel
/// rather than risking a validation error.
const SLICE_ALIGN_F32: u64 = 64;

/// Whether the batch axis can be walked as bound sub-ranges of `x` and `y`.
fn batch_sliceable(n: u32, x_row: u64, y_row: u64) -> bool {
    n <= 1 || (x_row.is_multiple_of(SLICE_ALIGN_F32) && y_row.is_multiple_of(SLICE_ALIGN_F32))
}

/// Ask [`gpu_core::select`] whether this conv's lowered GEMM is the one to
/// run. `RegisterTiled` names the whole lowering (im2col/col2im included), and
/// carries the `workgroup_reductions` requirement the register-tiled GEMMs
/// need - which is what keeps the CPU JIT, whose split-at-barrier model
/// mis-executes them, on the direct kernels without a backend-name test.
fn lowered(g: &Gpu, op: Op, m: u32, n: u32, k: u32) -> bool {
    let shape = OpShape { m, n, k, dtype: gpu_core::select::Dtype::F32 };
    DefaultSelector.select(op, shape, &g.caps()) == KernelVariant::RegisterTiled
}

impl Conv1d {
    /// Shapes the 1D lowering cannot express at all, independent of whether it
    /// would be faster: grouping (the lowering assumes one group - a grouped
    /// conv is a block-diagonal GEMM, which is a different kernel, not a
    /// different threshold) and a batch whose per-row binding offsets are not
    /// alignable.
    fn lowerable(&self, x_row: u64, y_row: u64) -> bool {
        self.groups == 1 && batch_sliceable(self.n, x_row, y_row)
    }
}

/// `y = conv1d(x, w) + bias` - the SELECTED lowering.
///
/// `x:[N,Cin,L]`, `w:[Cout,Cin/G,K]`, `bias:[Cout]`, `y:[N,Cout,Lo]`, all NCL,
/// identical to [`conv1d_fwd`]'s contract with the bias folded in (the direct
/// path appends `add_chan_inplace`, the lowered ones fuse it into their
/// epilogue). Three lowerings, chosen by `gpu_core::select`:
///
/// * **direct** - `conv1d` + `add_chan_inplace`. One thread per output element
///   with a serial `Cin*K` reduction: measured at **a low single-digit percent
///   of the card's compute roof** across the MiniMax-Music-3 vocoder's shapes,
///   where it was about half of the stage. It stays for narrow convs (`Cout < GEMM_CONV1D_MIN_COUT`, where
///   the GEMM's 128-wide column tile is mostly idle), for grouped convs, and
///   wherever `workgroup_reductions` is false.
/// * **lowered, `K > 1`** - `im2col1d_at` + `matmul_reg3` + `nlc_bias_nchw`,
///   i.e. `y[Lo, Cout] = col[Lo, Cin*K] · Wᵀ`. The native `[Cout, Cin/G, K]`
///   weight IS `[Cout, Cin*K]` row-major at `G = 1`, so `matmul_reg3` consumes
///   the checkpoint tensor with no permute. Chunked over output positions -
///   see [`gpu_core::lower`] and `im2col1d_at.wgsl` for why that is mandatory
///   rather than tidy.
/// * **lowered, `K == 1`** (stride 1, no pad) - `matmul_dx_reg`'s NN form
///   straight over the native operands: `y[Cout, Lo] = W[Cout, Cin] · x[Cin,
///   L]`. No im2col at all (a `K = 1` col operand is just x transposed) and no
///   epilogue transpose, because the result is already NCL. This covers the
///   1x1 projections that make up half a DAC-style vocoder's convolutions.
///
/// The GEMM reassociates the `Cin*K` reduction, so the lowered paths are NOT
/// bit-identical to the direct one - they are more accurate (a tree of
/// register accumulators rather than one serial `f32` chain), but a gate on
/// this path must be a tolerance, not `assert_eq!`.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_bias_fwd(
    g: &Gpu,
    k: &ConvGemmKernels,
    c: &Conv1d,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    bias: &DeviceBuffer,
    y: &DeviceBuffer,
    scratch: &mut ConvScratch,
) -> Vec<Step> {
    let (x_row, y_row) = (u64::from(c.cin) * u64::from(c.l), u64::from(c.cout) * u64::from(c.lo));
    let cink = c.cin * c.k;
    if !c.lowerable(x_row, y_row) || !lowered(g, Op::Conv1d, c.n * c.lo, c.cout, cink) {
        return conv1d_bias_direct(g, k, c, x, w, bias, y);
    }
    let mut steps = Vec::new();
    if c.k == 1 && c.stride == 1 && c.pad == 0 {
        // y[Cout, Lo] = W[Cout, Cin] · x[Cin, L]; `Lo == L` at this kernel
        // width, so the NN GEMM writes the final NCL tensor and the bias is
        // the one shared `add_chan_inplace` over the whole batch.
        for nn in 0..u64::from(c.n) {
            steps.push(g.step_sliced(
                k.matmul_nn,
                &[w, x, y],
                &[(0, 0), (nn * x_row, x_row), (nn * y_row, y_row)],
                &[c.cout, c.lo, c.cin, 0],
                reg_tiles(c.cout, c.lo) * 256,
            ));
        }
        steps.push(bias_step(g, k, c, y, bias));
        return steps;
    }
    for nn in 0..u64::from(c.n) {
        let (xo, yo) = ((nn * x_row, x_row), (nn * y_row, y_row));
        let budget = gpu_core::lower::col_budget_floats(scratch.budget_mib);
        let chunk = gpu_core::lower::col_chunk_rows(budget, u64::from(cink), GEMM_TILE, c.lo);
        let col = ConvScratch::grow(&mut scratch.col, g, u64::from(chunk) * u64::from(cink));
        let nlc = ConvScratch::grow(&mut scratch.out, g, y_row);
        let mut pos = 0u32;
        while pos < c.lo {
            let cnt = chunk.min(c.lo - pos);
            steps.push(g.step_sliced(
                k.im2col,
                &[x, &col],
                &[xo, (0, 0)],
                &[c.cin, c.l, c.k, c.stride, c.pad, c.dilation, cink, pos, cnt],
                cnt * cink,
            ));
            steps.push(g.step_sliced(
                k.matmul,
                &[&col, w, &nlc],
                &[(0, 0), (0, 0), (u64::from(pos) * u64::from(c.cout), u64::from(cnt) * u64::from(c.cout))],
                &[cnt, cink, c.cout],
                reg_tiles(cnt, c.cout) * 256,
            ));
            pos += cnt;
        }
        steps.push(g.step_sliced(
            k.nlc_bias,
            &[&nlc, bias, y],
            &[(0, 0), (0, 0), yo],
            &[c.lo * c.cout, c.cout, c.lo],
            c.cout.div_ceil(64) * c.lo.div_ceil(64) * 64,
        ));
    }
    steps
}

/// `y = conv_transpose1d(x, w) + bias` - the SELECTED lowering.
///
/// `x:[N,Cin,L]`, `w:[Cin,Cout/G,K]`, `bias:[Cout]`, `y:[N,Cout,Lo]`, the same
/// contract as [`convtr1d_fwd`] with the bias folded in. Two lowerings:
///
/// * **direct** - `convtr1d` + `add_chan_inplace`. One thread per output
///   element, and at an upsampling stride most of its `K` taps are discarded
///   by the divisibility test: measured at **a fraction of one percent of the
///   card's compute roof** in the MiniMax-Music-3 vocoder, where four
///   dispatches of it were about half of the stage.
/// * **lowered** - `matmul_dw_reg_splitk` (`s = 1`) + `col2im1d_bias`:
///   `col[Cout*K, L] = Wᵀ·x` in the TN form, whose contraction index is the
///   LEADING axis of both operands - which is exactly how a transposed conv's
///   native `[Cin, Cout/G, K]` weight and `[Cin, L]` NCL input are already
///   laid out, so this needs no transpose and no permute either. `col2im1d_bias`
///   then gathers the taps that land on each output sample.
///
/// The work does not grow: with `K = 2·stride` the GEMM does `L` rows rather
/// than `Lo = L·stride`, and `L·Cin·Cout·K == Lo·Cin·Cout·(K/stride)` - the
/// taps the direct kernel discards are the ones this form never computes.
///
/// Unlike [`conv1d_bias_fwd`] the `col` operand here is NOT chunkable: the TN
/// GEMM's output rows index `Cout*K`, so a range of `L` is a strided slice of
/// both `col` and the input, not a sub-range. It is instead BOUNDED -
/// `Cout*K*L` floats, `K/stride` times the output - and the lowering is used
/// only where that fits the device's storage-binding limit, falling back to
/// the direct kernel where it does not.
#[allow(clippy::too_many_arguments)]
pub fn convtr1d_bias_fwd(
    g: &Gpu,
    k: &ConvGemmKernels,
    c: &Conv1d,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    bias: &DeviceBuffer,
    y: &DeviceBuffer,
    scratch: &mut ConvScratch,
) -> Vec<Step> {
    let (x_row, y_row) = (u64::from(c.cin) * u64::from(c.l), u64::from(c.cout) * u64::from(c.lo));
    let cok = u64::from(c.cout) * u64::from(c.k);
    let col_len = cok * u64::from(c.l);
    let binds = col_len * 4 <= g.max_storage_binding_bytes();
    if !c.lowerable(x_row, y_row) || !binds || !lowered(g, Op::ConvTranspose1d, c.n * c.l, c.cout, c.cin) {
        return convtr1d_bias_direct(g, k, c, x, w, bias, y);
    }
    let col = ConvScratch::grow(&mut scratch.out, g, col_len);
    let mut steps = Vec::new();
    for nn in 0..u64::from(c.n) {
        let (xo, yo) = ((nn * x_row, x_row), (nn * y_row, y_row));
        steps.push(g.step_sliced(
            k.matmul_tn,
            &[w, x, &col],
            &[(0, 0), xo, (0, 0)],
            &[c.cin, c.l, c.cout * c.k, 1],
            reg_tiles(c.cout * c.k, c.l) * 256,
        ));
        steps.push(g.step_sliced(
            k.col2im,
            &[&col, bias, y],
            &[(0, 0), (0, 0), yo],
            &[c.l, c.cout, c.k, c.stride, c.pad, c.dilation, c.lo],
            c.cout * c.lo,
        ));
    }
    steps
}

/// Workgroups a `matmul_reg*`-family dispatch needs for an `rows x cols`
/// output: one per 128x128 tile. Multiply by the 256-thread workgroup.
fn reg_tiles(rows: u32, cols: u32) -> u32 {
    rows.div_ceil(GEMM_TILE) * cols.div_ceil(GEMM_TILE)
}

fn conv1d_bias_direct(
    g: &Gpu,
    k: &ConvGemmKernels,
    c: &Conv1d,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    bias: &DeviceBuffer,
    y: &DeviceBuffer,
) -> Vec<Step> {
    vec![conv1d_fwd(g, &k.direct, c, x, w, y), bias_step(g, k, c, y, bias)]
}

fn convtr1d_bias_direct(
    g: &Gpu,
    k: &ConvGemmKernels,
    c: &Conv1d,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    bias: &DeviceBuffer,
    y: &DeviceBuffer,
) -> Vec<Step> {
    vec![convtr1d_fwd(g, &k.direct, c, x, w, y), bias_step(g, k, c, y, bias)]
}

fn bias_step(g: &Gpu, k: &ConvGemmKernels, c: &Conv1d, y: &DeviceBuffer, bias: &DeviceBuffer) -> Step {
    let total = c.n * c.cout * c.lo;
    g.step(k.bias, &[y, bias], &[total, c.cout, c.lo], total)
}

/// `weight[i,...] = weight_g[i] * weight_v[i,...] / ||weight_v[i,...]||_2` -
/// PyTorch `nn.utils.weight_norm(dim=0)`. `d0` is `weight_v`'s leading dim
/// (for `Conv1d` that is `Cout`; for `ConvTranspose1d`'s native `[Cin,
/// Cout/G, K]` weight layout it is `Cin` - `weight_norm`'s `dim=0` always
/// means dim 0 of the STORED tensor, whichever axis that happens to be for
/// the layer type; confirmed against a real checkpoint, where
/// `conv_t1.weight_g` has one scalar per `Cin` row, not per `Cout`). A
/// one-time host op at import time, not a hot-path kernel.
pub fn fold_weight_norm(g: &[f32], v: &[f32], d0: usize) -> Vec<f32> {
    assert_eq!(g.len(), d0, "weight_norm: weight_g has {} elements, expected d0={d0}", g.len());
    assert_eq!(v.len() % d0, 0, "weight_norm: weight_v length {} not divisible by d0={d0}", v.len());
    let rest = v.len() / d0;
    let mut out = vec![0.0f32; v.len()];
    for i in 0..d0 {
        let row = &v[i * rest..(i + 1) * rest];
        let norm = row.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        let scale = (g[i] as f64 / norm.max(1e-12)) as f32;
        for (o, &x) in out[i * rest..(i + 1) * rest].iter_mut().zip(row) {
            *o = x * scale;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CPU reference oracles (kept tiny, used by tests; not on any hot path).
// ---------------------------------------------------------------------------

/// Reference forward for `conv1d` (matches `wgsl/conv1d.wgsl`).
pub fn conv1d_ref(c: &Conv1d, x: &[f32], w: &[f32]) -> Vec<f32> {
    let (cin_g, cout_g) = (c.cin / c.groups, c.cout / c.groups);
    let mut y = vec![0.0f32; (c.n * c.cout * c.lo) as usize];
    for n in 0..c.n {
        for co in 0..c.cout {
            let g = co / cout_g;
            for lo in 0..c.lo {
                let mut acc = 0.0;
                for cl in 0..cin_g {
                    let ci = g * cin_g + cl;
                    for kw in 0..c.k {
                        let li_b = lo * c.stride + kw * c.dilation;
                        if li_b >= c.pad {
                            let li = li_b - c.pad;
                            if li < c.l {
                                let xi = ((n * c.cin + ci) * c.l + li) as usize;
                                let wi = ((co * cin_g + cl) * c.k + kw) as usize;
                                acc += x[xi] * w[wi];
                            }
                        }
                    }
                }
                y[((n * c.cout + co) * c.lo + lo) as usize] = acc;
            }
        }
    }
    y
}

/// Reference forward for `convtr1d` (matches `wgsl/convtr1d.wgsl`).
pub fn convtr1d_ref(c: &Conv1d, x: &[f32], w: &[f32]) -> Vec<f32> {
    let (cin_g, cout_g) = (c.cin / c.groups, c.cout / c.groups);
    let mut y = vec![0.0f32; (c.n * c.cout * c.lo) as usize];
    for n in 0..c.n {
        for co in 0..c.cout {
            let g = co / cout_g;
            let co_local = co - g * cout_g;
            for lo in 0..c.lo {
                let mut acc = 0.0;
                for kw in 0..c.k {
                    let num = lo + c.pad;
                    let sub = kw * c.dilation;
                    if num >= sub && (num - sub).is_multiple_of(c.stride) {
                        let li = (num - sub) / c.stride;
                        if li < c.l {
                            for cl in 0..cin_g {
                                let ci = g * cin_g + cl;
                                let xi = ((n * c.cin + ci) * c.l + li) as usize;
                                let wi = ((ci * cout_g + co_local) * c.k + kw) as usize;
                                acc += x[xi] * w[wi];
                            }
                        }
                    }
                }
                y[((n * c.cout + co) * c.lo + lo) as usize] = acc;
            }
        }
    }
    y
}
