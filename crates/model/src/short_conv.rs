// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training and serving models with short causal convolutions for its
// clients. If your team needs expertise in sequence-mixing convolutions then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Short causal depthwise convolutions over token-major rows `[n, c]`, static or with per-token (data-dependent)
//! taps: forward, backward, a single-token decode step over a ring state, and the state a prefill chunk leaves.
//!
//! One kernel family covers both kinds of tap. `dyn_group_conv1d` computes
//! `y[t,c] = sum_j (base[j,c] + dyn[t,j,group(c)]) * x[t-j,c]`, and a [`ShortConvShape`] chooses what surrounds it:
//!
//! - **Static** taps ([`ShortConvShape::static_taps`]): no per-token part; the base tensor can be a published depthwise
//!   convolution weight `[c, 1, taps]` as it is ([`BaseLayout::ChannelMajor`], last tap on the current token).
//! - **Dynamic** taps ([`ShortConvShape::dynamic_taps`]): the per-token taps come from a projection's output.
//! - **Residual and scale**: `y = (conv(x) + x) * s` in the kernel ([`ShortConvShape::residual_scaled`]).
//! - **Segments**: a document id per row; rows of different documents never convolve with each other.
//! - **History**: the previous `taps-1` inputs of a sequence, for a chunk that continues it.
//!
//! Everything is f32. With none of the options on, a convolution is exactly what it was before they existed.
//! `ShortConvIds` names the kernels, resolved by the calling model against its registered pipeline list.

use gpu_core::{DeviceBuffer, Gpu, Step};

const FLAG_RES_SCALE: u32 = 1;
const FLAG_SEG: u32 = 2;
const FLAG_HIST: u32 = 4;
const FLAG_NO_DYN: u32 = 8;
const FLAG_BASE_CK: u32 = 16;

/// Kernel indices of the family: `dyn_group_conv1d` and its `_dx`, `_dbase`, `_ddyn`, `_step`, and `conv_state_tail`.
#[derive(Clone, Copy)]
pub struct ShortConvIds {
    pub fwd: usize,
    pub dx: usize,
    pub dbase: usize,
    pub ddyn: usize,
    pub step: usize,
    pub state_tail: usize,
}

/// The kernels of the family, for a caller to register next to its own list (resolved by [`ShortConvIds::resolve`]).
#[must_use]
pub fn pipelines() -> [(&'static str, &'static str); 6] {
    [
        ("dyn_group_conv1d", kernels::DYN_GROUP_CONV1D),
        ("dyn_group_conv1d_dx", kernels::DYN_GROUP_CONV1D_DX),
        ("dyn_group_conv1d_dbase", kernels::DYN_GROUP_CONV1D_DBASE),
        ("dyn_group_conv1d_ddyn", kernels::DYN_GROUP_CONV1D_DDYN),
        ("dyn_group_conv1d_step", kernels::DYN_GROUP_CONV1D_STEP),
        ("conv_state_tail", kernels::CONV_STATE_TAIL),
    ]
}

impl ShortConvIds {
    /// The indices of [`pipelines`] on `g`. Panics when one is not registered.
    pub fn resolve(g: &Gpu) -> ShortConvIds {
        let at = |name: &str| g.kernel_index(name).unwrap_or_else(|| panic!("{name} is not registered on this Gpu (see model::short_conv::pipelines)"));
        ShortConvIds { fwd: at("dyn_group_conv1d"), dx: at("dyn_group_conv1d_dx"), dbase: at("dyn_group_conv1d_dbase"), ddyn: at("dyn_group_conv1d_ddyn"), step: at("dyn_group_conv1d_step"), state_tail: at("conv_state_tail") }
    }
}

/// How the static taps are laid out in the base tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseLayout {
    /// `[taps, c]`, tap `j` weighs the input `j` rows back (tap 0 is the current row).
    TapMajor,
    /// `[c, taps]`, the last tap on the current row: a depthwise `Conv1d` weight `[c, 1, taps]`, read as published.
    ChannelMajor,
}

/// Where a dynamic convolution reads its per-token taps: `taps * groups` values from column `off` of each row of a
/// `[n, stride]` projection output, laid out `[tap][group]`; `group_size` channels share one value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynTaps {
    pub group_size: u32,
    pub stride: u32,
    pub off: u32,
}

/// The shape and options of one convolution call over `n` rows (or, for the step, `n` sequences) of `c` channels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShortConvShape {
    pub n: u32,
    pub c: u32,
    pub taps: u32,
    /// `None`: static taps only.
    pub dynamic: Option<DynTaps>,
    /// Element offset of the base kernel inside its tensor.
    pub base_off: u32,
    pub base_layout: BaseLayout,
    /// History restarts at every multiple of `block` rows (`n` for one sequence).
    pub block: u32,
    /// `Some(s)`: `y = (conv(x) + x) * s`.
    pub res_scale: Option<f32>,
    /// A document id per row is supplied and rows never convolve across a change of it.
    pub segments: bool,
    /// The call continues a sequence whose previous `taps - 1` inputs are supplied (forward only).
    pub history: bool,
}

impl ShortConvShape {
    /// A static convolution of `n` rows, one sequence, over a base tensor in `layout`.
    pub fn static_taps(n: u32, c: u32, taps: u32, layout: BaseLayout) -> Self {
        ShortConvShape { n, c, taps, dynamic: None, base_off: 0, base_layout: layout, block: n.max(1), res_scale: None, segments: false, history: false }
    }

    /// A convolution whose taps are the static base plus per-token taps from a projection.
    pub fn dynamic_taps(n: u32, c: u32, taps: u32, dynamic: DynTaps) -> Self {
        ShortConvShape { dynamic: Some(dynamic), base_layout: BaseLayout::TapMajor, ..Self::static_taps(n, c, taps, BaseLayout::TapMajor) }
    }

    /// `y = (conv(x) + x) * scale`.
    pub fn residual_scaled(mut self, scale: f32) -> Self {
        self.res_scale = Some(scale);
        self
    }

    /// Rows of different documents (ids supplied per row) never convolve.
    pub fn with_segments(mut self) -> Self {
        self.segments = true;
        self
    }

    /// The call continues a sequence from a supplied history.
    pub fn with_history(mut self) -> Self {
        self.history = true;
        self
    }

    /// History restarts at every multiple of `block` rows; `n` must be a whole number of blocks.
    pub fn in_blocks(mut self, block: u32) -> Self {
        self.block = block;
        self
    }

    fn validate(&self) {
        assert!(self.taps >= 1 && self.c >= 1 && self.block >= 1, "short conv: taps, c and block must be positive");
        assert_eq!(self.n % self.block, 0, "short conv: n ({}) is not a whole number of blocks of {}", self.n, self.block);
        if let Some(d) = self.dynamic {
            assert!(d.group_size >= 1 && self.c.is_multiple_of(d.group_size), "short conv: group_size {} does not divide c {}", d.group_size, self.c);
        }
        assert!(!self.history || self.block == self.n, "short conv: a carried history belongs to one sequence, so block must equal n");
    }

    fn flags(&self, with_history: bool) -> u32 {
        let mut f = 0;
        if self.res_scale.is_some() {
            f |= FLAG_RES_SCALE;
        }
        if self.segments {
            f |= FLAG_SEG;
        }
        if with_history && self.history {
            f |= FLAG_HIST;
        }
        if self.dynamic.is_none() {
            f |= FLAG_NO_DYN;
        }
        if self.base_layout == BaseLayout::ChannelMajor {
            f |= FLAG_BASE_CK;
        }
        f
    }

    /// The `dyn_group_conv1d*` parameter words (the step kernel's lack `block`).
    fn words(&self, with_history: bool, with_block: bool) -> Vec<u32> {
        let d = self.dynamic.unwrap_or(DynTaps { group_size: self.c, stride: 0, off: 0 });
        let mut w = vec![self.n, self.c, self.taps, d.group_size, self.c / d.group_size, d.stride, d.off, self.base_off];
        if with_block {
            w.push(self.block);
        }
        w.push(self.flags(with_history));
        w.push(gpu_core::f(self.res_scale.unwrap_or(1.0)));
        w
    }
}

/// The convolution of `x [n, c]` into `y [n, c]`. `dyn_taps` is the projection output (required for dynamic taps),
/// `seg` the per-row document ids, one `u32` per row (required with segments), `hist` the `[c, taps-1]` history
/// (required with history).
#[allow(clippy::too_many_arguments)]
pub fn short_conv_fwd(
    g: &Gpu,
    ids: &ShortConvIds,
    s: &ShortConvShape,
    x: &DeviceBuffer,
    dyn_taps: Option<&DeviceBuffer>,
    base: &DeviceBuffer,
    seg: Option<&DeviceBuffer>,
    hist: Option<&DeviceBuffer>,
    y: &DeviceBuffer,
) -> Step {
    s.validate();
    assert_eq!(dyn_taps.is_some(), s.dynamic.is_some(), "short_conv_fwd: dyn_taps must be supplied exactly for dynamic taps");
    assert_eq!(seg.is_some(), s.segments, "short_conv_fwd: seg must be supplied exactly with segments");
    assert_eq!(hist.is_some(), s.history, "short_conv_fwd: hist must be supplied exactly with history");
    g.step(ids.fwd, &[x, dyn_taps.unwrap_or(x), base, y, seg.unwrap_or(x), hist.unwrap_or(x)], &s.words(true, true), s.n * s.c)
}

/// The gradients of [`short_conv_fwd`] (without history): `d_x` is overwritten; `d_base` is ADDED into (zero it first);
/// `d_dyn` (`[n, stride]`, required for dynamic taps) has the convolution's window overwritten.
#[allow(clippy::too_many_arguments)]
pub fn short_conv_bwd(
    g: &Gpu,
    ids: &ShortConvIds,
    s: &ShortConvShape,
    x: &DeviceBuffer,
    dy: &DeviceBuffer,
    dyn_taps: Option<&DeviceBuffer>,
    base: &DeviceBuffer,
    seg: Option<&DeviceBuffer>,
    d_x: &DeviceBuffer,
    d_base: &DeviceBuffer,
    d_dyn: Option<&DeviceBuffer>,
) -> Vec<Step> {
    s.validate();
    assert!(!s.history, "short_conv_bwd: a carried history is an inference input and has no gradient");
    assert_eq!(dyn_taps.is_some(), s.dynamic.is_some(), "short_conv_bwd: dyn_taps must be supplied exactly for dynamic taps");
    assert_eq!(d_dyn.is_some(), s.dynamic.is_some(), "short_conv_bwd: d_dyn must be supplied exactly for dynamic taps");
    assert_eq!(seg.is_some(), s.segments, "short_conv_bwd: seg must be supplied exactly with segments");
    let w = s.words(false, true);
    let seg = seg.unwrap_or(x);
    let mut steps = vec![
        g.step(ids.dx, &[dy, dyn_taps.unwrap_or(dy), base, d_x, seg], &w, s.n * s.c),
        g.step(ids.dbase, &[dy, x, d_base, seg], &w, s.taps * s.c),
    ];
    if let (Some(d_dyn), Some(d)) = (d_dyn, s.dynamic) {
        steps.push(g.step(ids.ddyn, &[dy, x, d_dyn, seg], &w, s.n * s.taps * (s.c / d.group_size)));
    }
    steps
}

/// One decode token for each of `s.n` sequences, over the ring state `hist [n, c, taps-1]` (zeroed for a fresh
/// sequence; shifted in place). `s.block`, segments and history do not apply.
pub fn short_conv_step(g: &Gpu, ids: &ShortConvIds, s: &ShortConvShape, x: &DeviceBuffer, dyn_taps: Option<&DeviceBuffer>, base: &DeviceBuffer, hist: &DeviceBuffer, y: &DeviceBuffer) -> Step {
    s.validate();
    assert!(!s.segments && !s.history, "short_conv_step: the ring state is the sequence's own; segments and a supplied history do not apply");
    assert_eq!(dyn_taps.is_some(), s.dynamic.is_some(), "short_conv_step: dyn_taps must be supplied exactly for dynamic taps");
    g.step(ids.step, &[x, dyn_taps.unwrap_or(x), base, hist, y], &s.words(false, false), s.n * s.c)
}

/// The ring state after a prefill chunk of `t` rows of `c` channels (`x [t, c]`) that continued `hist_in`
/// (`[c, taps-1]`, zeroed for a fresh sequence): the last `taps-1` inputs, into the separate `hist_out`.
pub fn conv_state_tail_step(g: &Gpu, ids: &ShortConvIds, t: u32, c: u32, taps: u32, x: &DeviceBuffer, hist_in: &DeviceBuffer, hist_out: &DeviceBuffer) -> Step {
    assert!(taps >= 2, "conv_state_tail_step: a single-tap convolution has no state");
    g.step(ids.state_tail, &[x, hist_in, hist_out], &[t, c, taps], c * (taps - 1))
}
