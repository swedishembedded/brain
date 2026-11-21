// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving and training mixture-of-experts models whose expert
// activation or weight layout differs from the common SwiGLU recipe. If your team needs expertise in gated expert
// FFNs then you can procure our services by sending an email to info@swedishembedded.com.

//! The gated activation of an expert FFN, declared instead of hard-wired: which function combines the gate and
//! up values ([`GatedAct`]) and how the two projections are laid out ([`GluLayout`]), with a forward, a backward and
//! an expert-level pair built on them. [`crate::moe::expert_fwd`] and [`crate::moe::expert_bwd`] keep their own
//! SwiGLU over separate projections, bit for bit; a model with another activation or a fused weight bank calls
//! [`expert_fwd_glu`] / [`expert_bwd_glu`] (or just [`glu_act_step`] / [`glu_act_bwd_step`]).
//!
//! # Layout
//!
//! - [`GluLayout::Separate`]: two projections into two buffers, `gate_pre [rows, ff]` and `up [rows, ff]` (what
//!   [`crate::moe::expert_fwd`] does).
//! - [`GluLayout::Concat`]: one projection of `2 * ff` outputs per row, gate values first and up values after:
//!   `g_0 .. g_{ff-1}, u_0 .. u_{ff-1}` (the Hugging Face `gate_up_proj.chunk(2)` layout).
//! - [`GluLayout::Interleaved`]: one projection of `2 * ff` outputs per row, `g_0, u_0, g_1, u_1, ...`; a checkpoint
//!   that stores the fused weight and its block scales row-interleaved is served as published.
//!
//! The projections themselves are layout-agnostic: a fused bank is `[n_experts, 2 * ff, d]` and the existing gated,
//! NVFP4 and int8 GEMV/GEMM kernels read it with `n = 2 * ff`. The layout only matters where gate and up meet, which
//! is the activation kernel `glu_act` and its backward `glu_act_bwd`.

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::moe::{MoeIds, MoeIdsBwd, MoeShape};

/// The function that combines an expert's gate and up values into the hidden value fed to the down projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GatedAct {
    /// `silu(g) * u`: the default, what every existing expert path computes.
    SwiGlu,
    /// `softcap_cg(g) * sigmoid(g) * softcap_cu(u)` with `softcap_c(v) = c * tanh(v / c)`: the sigmoid sees the raw gate
    /// and the caps bound only the linear factors. Both caps must be positive and finite.
    SoftcapSigmoid { gate_cap: f32, up_cap: f32 },
}

/// How an expert's gate and up values are laid out; see the module doc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GluLayout {
    /// Two buffers (today's expert path).
    Separate,
    /// One fused row, gate block then up block.
    Concat,
    /// One fused row, gate and up values alternating.
    Interleaved,
}

/// A declared expert activation: the function and the layout. `GluSpec::default()` is SwiGLU over separate
/// projections, the computation every existing caller already does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GluSpec {
    pub act: GatedAct,
    pub layout: GluLayout,
}

impl Default for GluSpec {
    fn default() -> Self {
        GluSpec { act: GatedAct::SwiGlu, layout: GluLayout::Separate }
    }
}

impl GluSpec {
    /// Whether gate and up come from one fused projection of `2 * ff` outputs.
    pub fn is_fused(&self) -> bool {
        self.layout != GluLayout::Separate
    }

    /// The kernel's `[mode, act, cap_gate, cap_up]` parameter words.
    fn words(&self) -> [u32; 4] {
        let layout = match self.layout {
            GluLayout::Separate => 0,
            GluLayout::Concat => 1,
            GluLayout::Interleaved => 2,
        };
        match self.act {
            GatedAct::SwiGlu => [layout, 0, gpu_core::f(1.0), gpu_core::f(1.0)],
            GatedAct::SoftcapSigmoid { gate_cap, up_cap } => {
                assert!(gate_cap.is_finite() && gate_cap > 0.0 && up_cap.is_finite() && up_cap > 0.0, "GatedAct::SoftcapSigmoid: caps must be positive and finite (got {gate_cap}, {up_cap})");
                [layout, 1, gpu_core::f(gate_cap), gpu_core::f(up_cap)]
            }
        }
    }
}

impl GatedAct {
    /// The host reference of `h(g, u)`: what `glu_act` computes per element.
    pub fn apply(&self, g: f32, u: f32) -> f32 {
        match *self {
            GatedAct::SwiGlu => crate::hostmath::silu(g) * u,
            GatedAct::SoftcapSigmoid { gate_cap, up_cap } => {
                crate::hostmath::softcap(g, gate_cap) * (1.0 / (1.0 + (-g).exp())) * crate::hostmath::softcap(u, up_cap)
            }
        }
    }

    /// The host reference of `(dh/dg, dh/du)` at `(g, u)`: what `glu_act_bwd` computes per element for `dh = 1`.
    pub fn grad(&self, g: f32, u: f32) -> (f32, f32) {
        match *self {
            GatedAct::SwiGlu => (u * crate::hostmath::dsilu(g), crate::hostmath::silu(g)),
            GatedAct::SoftcapSigmoid { gate_cap, up_cap } => {
                let sig = 1.0 / (1.0 + (-g).exp());
                let (tg, tu) = ((g / gate_cap).tanh(), (u / up_cap).tanh());
                let fprime = (1.0 - tg * tg) * sig + gate_cap * tg * sig * (1.0 - sig);
                (fprime * up_cap * tu, gate_cap * tg * sig * (1.0 - tu * tu))
            }
        }
    }
}

/// Kernel indices of the activation pair, resolved by the calling model against its registered pipeline list:
/// `glu_act.wgsl` and `glu_act_bwd.wgsl`.
#[derive(Clone, Copy)]
pub struct GluIds {
    pub act: usize,
    pub act_bwd: usize,
}

/// `h[rows, ff] = act(gate, up)`. `a` is `gate_pre` for [`GluLayout::Separate`] (then `up` is `Some`) and the fused
/// `[rows, 2 * ff]` projection output otherwise (then `up` is `None`).
pub fn glu_act_step(g: &Gpu, ids: &GluIds, spec: &GluSpec, rows: u32, ff: u32, a: &DeviceBuffer, up: Option<&DeviceBuffer>, h: &DeviceBuffer) -> Step {
    assert_eq!(up.is_some(), !spec.is_fused(), "glu_act_step: `up` is the second buffer of the Separate layout and must be absent for a fused one");
    let w = spec.words();
    g.step(ids.act, &[a, up.unwrap_or(a), h], &[rows, ff, w[0], w[1], w[2], w[3]], rows * ff)
}

/// The gradients of [`glu_act_step`]'s inputs from `dh`, from the saved inputs `a`/`up`. Separate writes `da` and `db`
/// (`d_gate_pre`, `d_up`, both `[rows, ff]`); a fused layout writes the whole `[rows, 2 * ff]` gradient into `da`, in the
/// layout of the forward's input, and leaves `db` untouched: it must still be bound, to any buffer other than `da`
/// (a device refuses one buffer bound read-write twice). The gradients are fully overwritten.
pub fn glu_act_bwd_step(
    g: &Gpu,
    ids: &GluIds,
    spec: &GluSpec,
    rows: u32,
    ff: u32,
    a: &DeviceBuffer,
    up: Option<&DeviceBuffer>,
    dh: &DeviceBuffer,
    da: &DeviceBuffer,
    db: &DeviceBuffer,
) -> Step {
    assert_eq!(up.is_some(), !spec.is_fused(), "glu_act_bwd_step: `up` must be present exactly for the Separate layout");
    let w = spec.words();
    g.step(ids.act_bwd, &[a, up.unwrap_or(a), dh, da, db], &[rows, ff, w[0], w[1], w[2], w[3]], rows * ff)
}

/// One expert's projection weights. `gate_w` is the gate projection `[ff, d]` and `up_w` the up projection (Separate),
/// or the fused `[2 * ff, d]` matrix with `up_w == None` (Concat, Interleaved). `gate_off`/`down_off` are the element
/// offsets of this expert's matrices inside banks shared by every expert (`0` for per-expert buffers): they differ
/// between the fused matrix (`e * 2 * ff * d`) and the down matrix (`e * d * ff`).
#[derive(Clone, Copy)]
pub struct GluWeights<'a> {
    pub gate_w: &'a DeviceBuffer,
    pub up_w: Option<&'a DeviceBuffer>,
    pub down_w: &'a DeviceBuffer,
    pub gate_off: u32,
    pub down_off: u32,
}

/// The activations one expert saves for its backward: `gate_up` is `gate_pre [rows, ff]` (Separate, with `up` the up
/// projection) or the fused projection output `[rows, 2 * ff]`.
#[derive(Clone, Copy)]
pub struct GluScratch<'a> {
    pub gate_up: &'a DeviceBuffer,
    pub up: Option<&'a DeviceBuffer>,
    pub h: &'a DeviceBuffer,
    pub expert_out: &'a DeviceBuffer,
}

/// Gradients of one expert's weights, `None` skipping that dW (a frozen weight).
#[derive(Clone, Copy)]
pub struct GluGrads<'a> {
    pub gate_w: Option<&'a DeviceBuffer>,
    pub up_w: Option<&'a DeviceBuffer>,
    pub down_w: Option<&'a DeviceBuffer>,
}

/// Scratch of one expert's backward: `d_gate_up` is `d_gate_pre [rows, ff]` (Separate, with `d_up` the up gradient)
/// or the fused `[rows, 2 * ff]` gradient (then `d_up` is never written; any buffer other than `d_gate_up`).
#[derive(Clone, Copy)]
pub struct GluBwdScratch<'a> {
    pub d_expert_out: &'a DeviceBuffer,
    pub d_h: &'a DeviceBuffer,
    pub d_gate_up: &'a DeviceBuffer,
    pub d_up: &'a DeviceBuffer,
}

fn check_weights(spec: &GluSpec, w: &GluWeights) {
    assert_eq!(w.up_w.is_some(), !spec.is_fused(), "expert GLU: `up_w` is the second projection of the Separate layout and must be absent for a fused one");
}

/// One expert's forward under a declared activation, combined gate-weighted into `acc` like
/// [`crate::moe::expert_fwd`] (which this reproduces bit for bit for `GluSpec::default()`). Projections run through
/// `ids.linear_gated` (the per-row gate early-exit), so a fused bank uses it with `n = 2 * ff`.
#[allow(clippy::too_many_arguments)]
pub fn expert_fwd_glu(
    g: &Gpu,
    ids: &MoeIds,
    glu: &GluIds,
    spec: &GluSpec,
    shape: &MoeShape,
    x: &DeviceBuffer,
    gate: &DeviceBuffer,
    w: &GluWeights,
    scratch: &GluScratch,
    acc: &DeviceBuffer,
    e_idx: u32,
    accumulate: bool,
) -> Vec<Step> {
    check_weights(spec, w);
    let (m, d, ff, e) = (shape.rows, shape.d_model, shape.moe_ff, shape.n_experts);
    let lin = |x: &DeviceBuffer, wt: &DeviceBuffer, out: &DeviceBuffer, k: u32, n: u32, off: u32| g.step(ids.linear_gated, &[x, wt, gate, out], &[m, k, n, e, e_idx, off], m * n);
    let mut steps = Vec::with_capacity(5);
    if spec.is_fused() {
        steps.push(lin(x, w.gate_w, scratch.gate_up, d, 2 * ff, w.gate_off));
    } else {
        steps.push(lin(x, w.gate_w, scratch.gate_up, d, ff, w.gate_off));
        steps.push(lin(x, w.up_w.expect("checked"), scratch.up.expect("GluScratch::up is required for the Separate layout"), d, ff, w.gate_off));
    }
    steps.push(glu_act_step(g, glu, spec, m, ff, scratch.gate_up, scratch.up, scratch.h));
    steps.push(lin(scratch.h, w.down_w, scratch.expert_out, ff, d, w.down_off));
    steps.push(g.step(ids.scale_add, &[gate, scratch.expert_out, acc], &[m, d, e, e_idx, accumulate as u32], m * d));
    steps
}

/// One expert's backward under a declared activation, accumulating dX into `d_x`: [`crate::moe::expert_bwd`] with the
/// activation and layout of [`expert_fwd_glu`] (bit for bit for `GluSpec::default()`). Needs the gated kernel tier
/// (`ids.linear_gated`); `accumulate` governs the first touch of `d_x` exactly as in `expert_bwd`.
#[allow(clippy::too_many_arguments)]
pub fn expert_bwd_glu(
    g: &Gpu,
    ids: &MoeIdsBwd,
    glu: &GluIds,
    spec: &GluSpec,
    shape: &MoeShape,
    x: &DeviceBuffer,
    gate: &DeviceBuffer,
    w: &GluWeights,
    gr: &GluGrads,
    saved: &GluScratch,
    sb: &GluBwdScratch,
    d_acc: &DeviceBuffer,
    d_x: &DeviceBuffer,
    e_idx: u32,
    accumulate: bool,
    steps: &mut Vec<Step>,
) {
    assert!(ids.linear_gated, "expert_bwd_glu: a fused expert bank needs the gated kernel tier");
    check_weights(spec, w);
    let (m, d, ff, e) = (shape.rows, shape.d_model, shape.moe_ff, shape.n_experts);
    let dx = |dy: &DeviceBuffer, wt: &DeviceBuffer, out: &DeviceBuffer, k: u32, n: u32, acc: u32, off: u32, steps: &mut Vec<Step>| {
        steps.push(g.step(ids.linear_dx, &[dy, wt, gate, out], &[m, k, n, e, e_idx, acc, off], m * k));
    };
    let dw = |dy: &DeviceBuffer, xin: &DeviceBuffer, dwbuf: &DeviceBuffer, k: u32, n: u32, off: u32, steps: &mut Vec<Step>| {
        steps.push(g.step(ids.linear_dw, &[dy, xin, gate, dwbuf], &[m, k, n, e, e_idx, off], n * k));
    };

    steps.push(g.step(ids.scale_add_dexp, &[gate, d_acc, sb.d_expert_out], &[m, d, e, e_idx], m * d));
    if let Some(wg) = gr.down_w {
        dw(sb.d_expert_out, saved.h, wg, ff, d, w.down_off, steps);
    }
    dx(sb.d_expert_out, w.down_w, sb.d_h, ff, d, 0, w.down_off, steps);

    steps.push(glu_act_bwd_step(g, glu, spec, m, ff, saved.gate_up, saved.up, sb.d_h, sb.d_gate_up, sb.d_up));

    if spec.is_fused() {
        if let Some(wg) = gr.gate_w {
            dw(sb.d_gate_up, x, wg, d, 2 * ff, w.gate_off, steps);
        }
        dx(sb.d_gate_up, w.gate_w, d_x, d, 2 * ff, accumulate as u32, w.gate_off, steps);
    } else {
        let d_up = sb.d_up;
        if let Some(wg) = gr.up_w {
            dw(d_up, x, wg, d, ff, w.gate_off, steps);
        }
        dx(d_up, w.up_w.expect("checked"), d_x, d, ff, accumulate as u32, w.gate_off, steps);
        if let Some(wg) = gr.gate_w {
            dw(sb.d_gate_up, x, wg, d, ff, w.gate_off, steps);
        }
        dx(sb.d_gate_up, w.gate_w, d_x, d, ff, 1, w.gate_off, steps);
    }
}
