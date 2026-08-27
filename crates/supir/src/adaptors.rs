// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The 12 `ZeroSFT`/`ZeroCrossAttn` adaptors - the [`SkipFuse`] implementor
//! that REPLACES the frozen SDXL UNet's up-path skip concatenation with the
//! trunk's control signal.
//!
//! ## The injection schedule, derived not hand-listed
//! Upstream's `LightGLVUNet.forward` walks `project_modules` with a single
//! descending `adapter_idx` (11 -> 0): the post-mid `ZeroSFT` first, then for
//! each of the 9 output-block positions a `ZeroSFT` join, with a
//! `ZeroCrossAttn` interleaved right after the LAST join of every up-block
//! that ends in an `Upsample` (2 of SDXL's 3 up-blocks). [`AdaptorConfig::for_backbone`]
//! reproduces that exact schedule from `backbone`'s own `levels()` /
//! `layers_per_block` / `up_skips` - walking the frozen backbone's own skip
//! stack in up-path pop order - and its every `pm_idx`/channel width was
//! cross-checked against the real `SUPIR-v0Q_fp32.safetensors` header (all
//! 12 `project_modules` indices and their exact channel widths match), so
//! this is verified against real shapes, not merely internally consistent.
//!
//! ## Which trunk hidden state feeds which join
//! `GLVControl`'s 10 hidden states (`hs`, push order: the 9 down-path
//! entries then the mid output) mirror the frozen backbone's OWN skip stack
//! index-for-index - same push order, same widths (walking SDXL's skip
//! stack in up-path pop order against SUPIR's channel tables reproduces
//! both exactly, because the trunk mirrors the encoder that produced the
//! skip). So join `k` (0-indexed, pop order) and the frozen backbone's
//! `k`-th popped skip are the SAME stack POSITION in two structurally
//! mirrored stacks, both counted from the end:
//! `control_idx(k) = n_joins - 1 - k`. That identity is why `control_c ==
//! skip_c` always holds without being asserted anywhere - it is the same
//! arithmetic underneath both.
//!
//! ## No new kernel
//! `ZeroSFT` composes `Builder::conv`/`silu`/`concat`/`gn`/`mul`/`add`/`mix`;
//! `ZeroCrossAttn` composes `Builder::gn`/`nchw_to_rows`/`linear`/`cross_attn`/
//! `rows_to_nchw`/`mix`. `(1 + gamma)` never needs a "scalar add" kernel: `x
//! .* (1+gamma) + beta == x + x.*gamma + beta`, expressed with the existing
//! `mul`/`add` pair. Neither module needed a kernel the shared blocks
//! didn't already provide - checked against the kernel catalogue before
//! writing either, per this workspace's "one implementation" convention
//! (a duplicate `rmsnorm`/`rope`/`silu` has shipped more than once here in
//! the past for lack of that check).
//!
//! ## Deferred: flash-attention for `ZeroCrossAttn`
//! `model::block::flash_cross_step` (gated by `flash_cross_supported`) would
//! avoid materialising the `heads·T·T` score/probs pair `ZeroCrossAttn`'s two
//! sites need - the wider one (`project_modules.7`, 20 heads, T up to
//! 1024 at a 128x128 latent) is a real allocation. This implementation takes
//! the materialised `Builder::cross_attn` path unconditionally, matching
//! [`sdxlunet::model::Rec::cross_attention`]'s own non-cooperative fallback
//! shape rather than the flash rung - **explicitly deferred**, not silently
//! skipped: the two sites are small enough (`tkv <= 1024`) that the
//! materialised pair is nowhere near the 2 GiB binding limit the frozen
//! backbone's OWN self-attention needed flash for, so correctness-first is
//! the right trade for this pass. A later optimisation pass can wire the
//! flash rung once parity is proven and frozen, and only then, per this
//! workspace's own performance-ladder discipline (get it correct first,
//! profile before optimising).

use sdxlunet::config::UNetConfig;
use vae::blocks::skipfuse::{Map, SkipFuse};
use vae::blocks::Builder;

/// `SPADE`'s hidden width for the `mlp_shared` trunk inside every `ZeroSFT` -
/// fixed at 128 regardless of the join's own channel widths (verified
/// against every `project_modules.*.mlp_shared.0.weight`'s output channel
/// count in the real checkpoint).
pub const NHIDDEN: u32 = 128;

/// `ZeroCrossAttn`'s fixed per-head width (verified: `query_dim / 64` is
/// exactly 10 and 20 at the two real sites, both integral).
pub const HEAD_DIM: u32 = 64;

/// One `ZeroSFT` join: `project_modules[pm_idx]`, replacing the up path's
/// `k`-th skip concat (`k` = pop order, `0` first).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JoinSpec {
    pub pm_idx: usize,
    /// Index into `Adaptors`' 10 trunk hidden states (`hs`).
    pub control_idx: usize,
    pub h_ori_c: u32,
    /// Also the control tensor's own channel width - see the module doc.
    pub skip_c: u32,
}

impl JoinSpec {
    /// `h_ori_c + skip_c` - the joined width every consuming up-path resnet
    /// reads, identical to a plain concat's (`vae::blocks::skipfuse`'s
    /// shape-preservation contract).
    pub fn c_out(&self) -> u32 {
        self.h_ori_c + self.skip_c
    }
}

/// The post-mid-block `ZeroSFT`: `project_modules[11]`, no concat, no
/// `control_scale` lerp - see [`Adaptors::fuse_mid`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MidSpec {
    pub pm_idx: usize,
    pub control_idx: usize,
    /// Both the control tensor's and the frozen mid output's channel width
    /// (they must agree - there is no concat to make them differ).
    pub c: u32,
}

/// One `ZeroCrossAttn` site: applied to up-block `up_block`'s running hidden
/// state right before that block's `Upsample`, attending against the SAME
/// control tensor the up-block's last `ZeroSFT` join just consumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrossSpec {
    pub pm_idx: usize,
    pub control_idx: usize,
    /// Which up-block (0-indexed, `i` in `Rec::record_into`'s up-path loop).
    pub up_block: usize,
    /// The up-block's own running width - `ZeroCrossAttn`'s query dim.
    pub x_c: u32,
    /// The control tensor's width - `ZeroCrossAttn`'s context dim.
    pub context_c: u32,
}

impl CrossSpec {
    pub fn heads(&self) -> u32 {
        assert_eq!(self.x_c % HEAD_DIM, 0, "cross site {}: {} is not a multiple of head_dim {HEAD_DIM}", self.pm_idx, self.x_c);
        self.x_c / HEAD_DIM
    }
}

/// The 12 adaptors' full channel schedule, derived from the frozen
/// backbone - see the module doc.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptorConfig {
    /// The 9 `ZeroSFT` joins, in pop order (`k = 0` first).
    pub joins: Vec<JoinSpec>,
    pub mid: MidSpec,
    /// The `levels - 1` `ZeroCrossAttn` sites, one per up-block that ends in
    /// an `Upsample`, in up-block order.
    pub cross: Vec<CrossSpec>,
    /// The backbone's own GroupNorm eps/groups - `param_free_norm` and
    /// `ZeroCrossAttn`'s two norms use a FIXED 32 groups regardless (SPADE's
    /// own convention), so every adaptor call must restore these afterward
    /// or the next `Rec::resnet`'s GroupNorm silently runs at the wrong
    /// group count.
    pub backbone_norm_eps: f32,
    pub backbone_norm_groups: u32,
}

impl AdaptorConfig {
    /// Derive the full 12-adaptor schedule from `backbone`'s own skip-stack
    /// arithmetic - see the module doc for the exact walk, and
    /// `crates/supir/src/config.rs`'s tests for the real-checkpoint
    /// cross-check.
    pub fn for_backbone(backbone: &UNetConfig) -> AdaptorConfig {
        let per_up = (backbone.layers_per_block + 1) as usize;
        let levels = backbone.levels();
        let n_joins = levels * per_up; // == backbone.skip_stack().len()
        let n_cross = levels.saturating_sub(1);
        let total = 1 + n_joins + n_cross;

        // `adapter_idx` counts DOWN from `total - 1`: the mid site takes the
        // top index, then one decrement per join, with an extra decrement
        // for each interleaved cross-attn site.
        let mut next_idx = total - 1;
        let mid = MidSpec {
            pm_idx: next_idx,
            control_idx: n_joins, // hs[n_joins] == the trunk's own mid output
            c: *backbone.block_out_channels.last().expect("levels >= 1"),
        };
        next_idx = next_idx.saturating_sub(1);

        let mut joins = Vec::with_capacity(n_joins);
        let mut cross = Vec::with_capacity(n_cross);
        let mut prev = *backbone.block_out_channels.last().expect("levels >= 1");
        let mut k = 0usize; // global join counter, 0-indexed pop order
        for i in 0..levels {
            let level = levels - 1 - i;
            let cout = backbone.block_out_channels[level];
            let skips = backbone.up_skips(i);
            let n_this_block = skips.len();
            for (j, &skip_c) in skips.iter().enumerate() {
                let h_ori_c = if j == 0 { prev } else { cout };
                let control_idx = n_joins - 1 - k;
                joins.push(JoinSpec { pm_idx: next_idx, control_idx, h_ori_c, skip_c });
                next_idx = next_idx.saturating_sub(1);
                prev = cout;
                let last_of_block = j + 1 == n_this_block;
                if last_of_block && i + 1 < levels {
                    cross.push(CrossSpec {
                        pm_idx: next_idx,
                        control_idx, // the join just emitted, same site
                        up_block: i,
                        x_c: cout,
                        context_c: skip_c,
                    });
                    next_idx = next_idx.saturating_sub(1);
                }
                k += 1;
            }
        }
        assert_eq!(joins.len(), n_joins);
        assert_eq!(cross.len(), n_cross);

        AdaptorConfig {
            joins,
            mid,
            cross,
            backbone_norm_eps: backbone.norm_eps,
            backbone_norm_groups: backbone.norm_num_groups,
        }
    }

    /// Canonical tensor manifest for all 12 adaptors, named
    /// `project_modules.{pm_idx}.*` exactly as the checkpoint's own
    /// `model.diffusion_model.project_modules.*` tensors are named (minus
    /// that prefix) - so [`crate::import`]'s remap for this half is a plain
    /// rename, not a restructuring. `ZeroCrossAttn`'s `to_k`/`to_v` are
    /// declared PRE-FUSED as one `attn.kv` `[2*x_c, context_c]` weight - the
    /// same host-side fusion `sdxlunet::config::UNetConfig::tensor_manifest`
    /// already does for `attn2.kv` (`crates/supir/src/import.rs` performs
    /// the concatenation).
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut v = Vec::new();
        let zero_sft = |v: &mut Vec<(String, Vec<usize>)>, pm: usize, control_c: u32, c_out: u32| {
            let p = format!("project_modules.{pm}");
            v.push((format!("{p}.mlp_shared.0.weight"), vec![NHIDDEN as usize, control_c as usize, 3, 3]));
            v.push((format!("{p}.mlp_shared.0.bias"), vec![NHIDDEN as usize]));
            v.push((format!("{p}.param_free_norm.weight"), vec![c_out as usize]));
            v.push((format!("{p}.param_free_norm.bias"), vec![c_out as usize]));
            v.push((format!("{p}.zero_mul.weight"), vec![c_out as usize, NHIDDEN as usize, 3, 3]));
            v.push((format!("{p}.zero_mul.bias"), vec![c_out as usize]));
            v.push((format!("{p}.zero_add.weight"), vec![c_out as usize, NHIDDEN as usize, 3, 3]));
            v.push((format!("{p}.zero_add.bias"), vec![c_out as usize]));
            v.push((format!("{p}.zero_conv.weight"), vec![control_c as usize, control_c as usize, 1, 1]));
            v.push((format!("{p}.zero_conv.bias"), vec![control_c as usize]));
        };
        for j in &self.joins {
            zero_sft(&mut v, j.pm_idx, j.skip_c, j.c_out());
        }
        zero_sft(&mut v, self.mid.pm_idx, self.mid.c, self.mid.c);
        for c in &self.cross {
            let p = format!("project_modules.{}", c.pm_idx);
            v.push((format!("{p}.norm1.weight"), vec![c.x_c as usize]));
            v.push((format!("{p}.norm1.bias"), vec![c.x_c as usize]));
            v.push((format!("{p}.norm2.weight"), vec![c.context_c as usize]));
            v.push((format!("{p}.norm2.bias"), vec![c.context_c as usize]));
            v.push((format!("{p}.attn.to_q.weight"), vec![c.x_c as usize, c.x_c as usize]));
            v.push((format!("{p}.attn.kv.weight"), vec![2 * c.x_c as usize, c.context_c as usize]));
            v.push((format!("{p}.attn.to_out.0.weight"), vec![c.x_c as usize, c.x_c as usize]));
            v.push((format!("{p}.attn.to_out.0.bias"), vec![c.x_c as usize]));
        }
        v
    }
}

/// The recorded [`SkipFuse`] implementor: the trunk's 10 hidden states plus
/// the channel schedule, so every adaptor call knows which control tensor to
/// read without re-deriving the schedule mid-record.
pub struct Adaptors {
    cfg: AdaptorConfig,
    /// The trunk's `hs` - push order, 9 down-path entries then the mid
    /// output (see the module doc's "which trunk hidden state" section).
    hs: Vec<Map>,
    /// `s_stage2` in the upstream sampler's notation - a HOST float, baked
    /// into the recorded graph at `Builder::mix`'s call site (its `a`/`b`
    /// are host constants, not a runtime-writable buffer - see that
    /// method's doc). The reference sampler ramps `control_scale` per step
    /// (`linear_s_stage2`), so a full restoration pipeline built on this
    /// graph needs either a fresh recording per step or a `Builder::mix`
    /// variant that reads a device buffer - an open design question for
    /// whoever writes that pipeline, flagged here rather than silently
    /// assumed away.
    control_scale: f32,
}

impl Adaptors {
    /// `hs.len()` must be `joins.len() + 1` (every `control_idx` in `cfg`
    /// indexes into it) - checked here rather than discovered by an
    /// out-of-bounds panic deep in a join call.
    pub fn new(cfg: AdaptorConfig, hs: Vec<Map>, control_scale: f32) -> Adaptors {
        assert_eq!(
            hs.len(),
            cfg.joins.len() + 1,
            "adaptors: {} trunk hidden states, schedule expects {}",
            hs.len(),
            cfg.joins.len() + 1
        );
        Adaptors { cfg, hs, control_scale }
    }

    pub fn config(&self) -> &AdaptorConfig {
        &self.cfg
    }

    /// `param_free_norm(h1) · (1 + gamma) + beta`, expressed without a
    /// "scalar add" kernel: `x + x.*gamma + beta` - see the module doc.
    fn zero_sft_tail(&self, b: &mut Builder<'_>, pm_idx: usize, control: &Map, h1: &Map, c_out: u32) -> gpu_core::DeviceBuffer {
        let n = c_out * h1.h * h1.w;
        b.set_groups(32);
        b.set_eps(1e-5);
        let normed = b.gn(&format!("project_modules.{pm_idx}.param_free_norm"), c_out, h1.h, h1.w, &h1.buf);
        b.set_groups(self.cfg.backbone_norm_groups);
        b.set_eps(self.cfg.backbone_norm_eps);

        let actv_pre = b.conv(&format!("project_modules.{pm_idx}.mlp_shared.0"), control.c, NHIDDEN, 3, 1, control.h, control.w, &control.buf);
        let actv = b.silu(NHIDDEN * control.h * control.w, &actv_pre);
        let gamma = b.conv(&format!("project_modules.{pm_idx}.zero_mul"), NHIDDEN, c_out, 3, 1, control.h, control.w, &actv);
        let beta = b.conv(&format!("project_modules.{pm_idx}.zero_add"), NHIDDEN, c_out, 3, 1, control.h, control.w, &actv);

        let ng = b.mul(n, &normed, &gamma);
        let n_plus = b.add(n, &normed, &ng);
        b.add(n, &n_plus, &beta)
    }

    /// The zero-init 1x1 conv added onto `skip` before the concat that
    /// produces `h1` - `ZeroConv1x1(c)`, channel-preserving (see
    /// `crates/supir/src/config.rs`'s module doc for why its width is the
    /// control tensor's own).
    fn zero_conv(&self, b: &mut Builder<'_>, pm_idx: usize, control: &Map) -> gpu_core::DeviceBuffer {
        b.conv(&format!("project_modules.{pm_idx}.zero_conv"), control.c, control.c, 1, 0, control.h, control.w, &control.buf)
    }
}

impl SkipFuse for Adaptors {
    fn kernels(&self) -> &'static [(&'static str, &'static str)] {
        &[("edm_mix", kernels::EDM_MIX), ("scale_row", kernels::SCALE_ROW)]
    }

    fn joins(&self) -> usize {
        self.cfg.joins.len()
    }

    fn fuse_skip(&self, b: &mut Builder<'_>, k: usize, h_ori: &Map, skip: &Map) -> Map {
        let j = self.cfg.joins[k];
        assert_eq!((h_ori.c, skip.c), (j.h_ori_c, j.skip_c), "adaptors: join {k} shape disagrees with the schedule");
        let control = &self.hs[j.control_idx];
        assert_eq!(control.c, j.skip_c, "adaptors: join {k}'s control tensor is {} wide, schedule says {}", control.c, j.skip_c);

        // Tapped under `proj{pm_idx}.in{0,1,2}`, matching
        // `tools/goldens/supir_dump_reference.py`'s forward-hook capture of
        // upstream `ZeroSFT.forward(self, c, h, h_ori=None, ...)`'s
        // POSITIONAL args in call order: `c` (the control tensor) is in0,
        // `h` (the skip) is in1, `h_ori` is in2 - tapping only the output
        // would miss a permutation bug among these three same-width-at-some-
        // joins tensors (see the module doc's "which trunk hidden state"
        // section).
        b.tap(format!("proj{}.in0", j.pm_idx), &control.buf, control.c * control.h * control.w);
        b.tap(format!("proj{}.in1", j.pm_idx), &skip.buf, skip.c * skip.h * skip.w);
        b.tap(format!("proj{}.in2", j.pm_idx), &h_ori.buf, h_ori.c * h_ori.h * h_ori.w);

        let c_out = j.c_out();
        let h_raw = b.concat(h_ori.c, skip.c, h_ori.h, h_ori.w, &h_ori.buf, &skip.buf);
        let zc = self.zero_conv(b, j.pm_idx, control);
        let skip_plus = b.add(skip.c * skip.h * skip.w, &skip.buf, &zc);
        let h1 = b.concat(h_ori.c, skip.c, h_ori.h, h_ori.w, &h_ori.buf, &skip_plus);
        let h1_map = Map { buf: h1, c: c_out, h: h_ori.h, w: h_ori.w };
        let out = self.zero_sft_tail(b, j.pm_idx, control, &h1_map, c_out);

        let n = c_out * h_ori.h * h_ori.w;
        let lerped = b.mix(n, self.control_scale, 1.0 - self.control_scale, &out, &h_raw);
        b.tap(format!("proj{}", j.pm_idx), &lerped, n);
        Map { buf: lerped, c: c_out, h: h_ori.h, w: h_ori.w }
    }

    fn fuse_mid(&self, b: &mut Builder<'_>, x: &Map) -> Map {
        let m = self.cfg.mid;
        let control = &self.hs[m.control_idx];
        assert_eq!(control.c, m.c, "adaptors: mid control tensor is {} wide, schedule says {}", control.c, m.c);
        assert_eq!(x.c, m.c, "adaptors: mid input is {} wide, schedule says {}", x.c, m.c);

        // Same call-order convention as `fuse_skip`'s taps, minus `in2`:
        // upstream calls `project_modules[11](control[9], h)` with no
        // `h_ori` argument at all at this site (see the module doc).
        b.tap(format!("proj{}.in0", m.pm_idx), &control.buf, control.c * control.h * control.w);
        b.tap(format!("proj{}.in1", m.pm_idx), &x.buf, x.c * x.h * x.w);

        // No `h_ori`: the general `h1 = concat(h_ori, skip + zero_conv(c))`
        // degenerates to `h1 = x + zero_conv(c)` - see config.rs's module
        // doc for why a `zero_conv` tensor still exists at this site.
        let zc = self.zero_conv(b, m.pm_idx, control);
        let h1 = b.add(m.c * x.h * x.w, &x.buf, &zc);
        let h1_map = Map { buf: h1, c: m.c, h: x.h, w: x.w };
        // No `control_scale` lerp at the post-mid site either.
        let out = self.zero_sft_tail(b, m.pm_idx, control, &h1_map, m.c);
        b.tap(format!("proj{}", m.pm_idx), &out, m.c * x.h * x.w);
        Map { buf: out, c: m.c, h: x.h, w: x.w }
    }

    fn pre_upsample(&self, b: &mut Builder<'_>, i: usize, x: &Map) -> Map {
        let Some(spec) = self.cfg.cross.iter().find(|c| c.up_block == i) else {
            return x.clone();
        };
        let control = &self.hs[spec.control_idx];
        assert_eq!(x.c, spec.x_c, "adaptors: cross site {}: x is {} wide, schedule says {}", spec.pm_idx, x.c, spec.x_c);
        assert_eq!(control.c, spec.context_c, "adaptors: cross site {}: context is {} wide, schedule says {}", spec.pm_idx, control.c, spec.context_c);

        // Same convention, mapped onto upstream `ZeroCrossAttn.forward(self,
        // context, x, ...)`'s positional order: `context` (the control
        // tensor) is in0, `x` (the up-block's own running state) is in1.
        b.tap(format!("proj{}.in0", spec.pm_idx), &control.buf, control.c * control.h * control.w);
        b.tap(format!("proj{}.in1", spec.pm_idx), &x.buf, x.c * x.h * x.w);

        let (tq, tkv) = (x.h * x.w, control.h * control.w);
        let heads = spec.heads();
        b.set_groups(32);
        b.set_eps(1e-5);
        let x_normed = b.gn(&format!("project_modules.{}.norm1", spec.pm_idx), x.c, x.h, x.w, &x.buf);
        let c_normed = b.gn(&format!("project_modules.{}.norm2", spec.pm_idx), control.c, control.h, control.w, &control.buf);
        b.set_groups(self.cfg.backbone_norm_groups);
        b.set_eps(self.cfg.backbone_norm_eps);

        let x_rows = b.nchw_to_rows(x.c, tq, &x_normed);
        let c_rows = b.nchw_to_rows(control.c, tkv, &c_normed);
        let q = b.linear(&format!("project_modules.{}.attn.to_q", spec.pm_idx), tq, x.c, x.c, false, &x_rows);
        let kv = b.linear(&format!("project_modules.{}.attn.kv", spec.pm_idx), tkv, control.c, 2 * x.c, false, &c_rows);
        let ctx = b.act((tq as u64) * (x.c as u64));
        b.cross_attn(heads, HEAD_DIM, x.c, tq, tkv, &q, &kv, &ctx);
        let ao = b.linear(&format!("project_modules.{}.attn.to_out.0", spec.pm_idx), tq, x.c, x.c, true, &ctx);
        let ao_chw = b.rows_to_nchw(x.c, tq, &ao);

        let n = x.c * x.h * x.w;
        let out = b.mix(n, 1.0, self.control_scale, &x.buf, &ao_chw);
        b.tap(format!("proj{}", spec.pm_idx), &out, n);
        Map { buf: out, c: x.c, h: x.h, w: x.w }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The join table (per-join `h_ori`/skip channel widths, control index
    /// and joined width), reproduced by [`AdaptorConfig::for_backbone`]
    /// purely from `UNetConfig::sdxl_base`'s public API - the weight-free
    /// gate that turns this schedule into checked code.
    #[test]
    fn sdxl_join_table_matches_the_roadmap() {
        let cfg = AdaptorConfig::for_backbone(&UNetConfig::sdxl_base());
        let want: [(u32, u32, usize, u32); 9] = [
            (1280, 1280, 8, 2560),
            (1280, 1280, 7, 2560),
            (1280, 640, 6, 1920),
            (1280, 640, 5, 1920),
            (640, 640, 4, 1280),
            (640, 320, 3, 960),
            (640, 320, 2, 960),
            (320, 320, 1, 640),
            (320, 320, 0, 640),
        ];
        assert_eq!(cfg.joins.len(), 9);
        for (k, (h_ori_c, skip_c, control_idx, c_out)) in want.into_iter().enumerate() {
            let j = cfg.joins[k];
            assert_eq!(j.h_ori_c, h_ori_c, "join {k} h_ori_c");
            assert_eq!(j.skip_c, skip_c, "join {k} skip_c");
            assert_eq!(j.control_idx, control_idx, "join {k} control_idx");
            assert_eq!(j.c_out(), c_out, "join {k} c_out");
        }
        assert_eq!(cfg.mid.control_idx, 9);
        assert_eq!(cfg.mid.c, 1280);
        assert_eq!(cfg.cross.len(), 2);
    }

    /// `project_modules` indices, cross-checked against the real checkpoint
    /// header (`crates/supir/src/config.rs`'s module doc): mid=11, joins
    /// 10,9,8,6,5,4,2,1,0, cross 7 (after join 2) and 3 (after join 5).
    #[test]
    fn sdxl_project_modules_indices_match_the_real_checkpoint() {
        let cfg = AdaptorConfig::for_backbone(&UNetConfig::sdxl_base());
        assert_eq!(cfg.mid.pm_idx, 11);
        let want_pm = [10, 9, 8, 6, 5, 4, 2, 1, 0];
        for (k, pm) in want_pm.into_iter().enumerate() {
            assert_eq!(cfg.joins[k].pm_idx, pm, "join {k}");
        }
        assert_eq!(cfg.cross[0].pm_idx, 7);
        assert_eq!(cfg.cross[0].up_block, 0);
        assert_eq!(cfg.cross[0].x_c, 1280);
        assert_eq!(cfg.cross[0].context_c, 640);
        assert_eq!(cfg.cross[1].pm_idx, 3);
        assert_eq!(cfg.cross[1].up_block, 1);
        assert_eq!(cfg.cross[1].x_c, 640);
        assert_eq!(cfg.cross[1].context_c, 320);

        // Every one of the 12 indices 0..11 is used exactly once.
        let mut seen: Vec<usize> = cfg.joins.iter().map(|j| j.pm_idx).chain([cfg.mid.pm_idx]).chain(cfg.cross.iter().map(|c| c.pm_idx)).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn cross_attn_heads_are_integral() {
        let cfg = AdaptorConfig::for_backbone(&UNetConfig::sdxl_base());
        assert_eq!(cfg.cross[0].heads(), 20);
        assert_eq!(cfg.cross[1].heads(), 10);
    }
}
