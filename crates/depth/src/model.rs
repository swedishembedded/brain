// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ZipDepth: the encoder/decoder assembly.
//!
//! Wires the ten blocks of [`crate::blocks`] into the reference's own graph
//! (`architecture.py:472-494` decoder, `:560-572` encoder, `:639-645` top level).
//! Every parameter name here must match [`crate::config::ZipConfig::param_list`],
//! which is itself checked against the released `.pth` — so a divergence in this
//! file fails the layout test rather than importing a silently different model.
//!
//! Resolution ladder at the 384 input: stem /2 -> /4, then one QARepBlock stride-2
//! per stage down to /32, and the decoder climbs back to /2 before the learned
//! 2x convex upsample lands on the input grid exactly.
//!
//! ```text
//!   x [N,3,384,384]
//!     stem_half   -> s_half [N,24,192,192]
//!     stem_quarter-> [N,48,96,96]
//!     stage1      -> c1 [N,48,96,96]
//!     down2/stage2-> c2 [N,96,48,48]
//!     down3/stage3-> c3 [N,192,24,24]
//!     down4/stage4-> [N,384,12,12] -> spp -> cross_scale(c3, ..) -> c3', c4'
//!     proj4(c4')  -> f4 [N,288,12,12]
//!     fuse3(c3',f4)-> f3 [N,192,24,24]
//!     fuse2(c2,f3)-> f2 [N,144,48,48]
//!     fuse1(c1,f2)-> f1 [N,96,96,96]
//!     fuse_half(s_half,f1) -> f_half [N,32,192,192]
//!     head_half   -> depth_half [N,1,192,192]
//!     convex_up(f_half, depth_half, S=2) -> [N,1,384,384]
//! ```

use gpu_core::{DeviceBuffer, Gpu};
use paramstore::ParamStore;
use vision::blocks::{Act, Conv, ConvNames, ConvSpec, Norm};
use vision::{Ctx, Shape, SppfSpec, NameStyle, SPPF};

use crate::blocks::{
    ChannelAttention, FastConvexUpsample, GlobalContextBlock, MinimalCrossScale, MinimalMultiScale, QARepBlock,
    StripPoolingAttention, UltraLightFusion, UpsampleKind,
};
use crate::config::{GlobalMode, ZipConfig};

/// The learned 2x upsample's scale. Fixed by the reference
/// (`architecture.py:465`: `FastConvexUpsample(feat_ch=ch_half, scale=2)`) and
/// pinned by the checkpoint, whose `mask_pred.3` emits `9*2*2 = 36` channels.
const UPSAMPLE_SCALE: u32 = 2;

/// One encoder stage: N QARepBlocks, then the stage's optional attention tail.
///
/// The tails are NOT interchangeable and their ORDER is part of the checkpoint's
/// index numbering — `stage2` gets MinimalMultiScale then StripPoolingAttention,
/// `stage3` gets ChannelAttention then GlobalContextBlock. Appending them in the
/// other order would renumber `stage2.2`/`stage2.3` and load the wrong tensors
/// into the wrong module while still matching every shape.
struct Stage {
    blocks: Vec<QARepBlock>,
    mms: Option<MinimalMultiScale>,
    spa: Option<StripPoolingAttention>,
    se: Option<ChannelAttention>,
    gcb: Option<GlobalContextBlock>,
    /// Grad scratch, one per inter-block link (all at the stage's own shape).
    links: Vec<DeviceBuffer>,
}

impl Stage {
    fn out(&self) -> &DeviceBuffer {
        if let Some(b) = &self.gcb {
            return b.out();
        }
        if let Some(b) = &self.se {
            return b.out();
        }
        if let Some(b) = &self.spa {
            return b.out();
        }
        if let Some(b) = &self.mms {
            return b.out();
        }
        self.blocks.last().expect("a stage has at least one block").out()
    }

    fn set_eval(&self, on: bool) {
        self.blocks.iter().for_each(|b| b.set_eval(on));
        self.mms.iter().for_each(|b| b.set_eval(on));
        self.spa.iter().for_each(|b| b.set_eval(on));
        self.gcb.iter().for_each(|b| b.set_eval(on));
    }
    fn set_update_running(&self, on: bool) {
        self.blocks.iter().for_each(|b| b.set_update_running(on));
        self.mms.iter().for_each(|b| b.set_update_running(on));
        self.spa.iter().for_each(|b| b.set_update_running(on));
        self.gcb.iter().for_each(|b| b.set_update_running(on));
    }
    fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = Vec::new();
        for b in &self.blocks {
            v.extend(b.param_list());
        }
        if let Some(b) = &self.mms {
            v.extend(b.param_list());
        }
        if let Some(b) = &self.spa {
            v.extend(b.param_list());
        }
        if let Some(b) = &self.se {
            v.extend(b.param_list());
        }
        if let Some(b) = &self.gcb {
            v.extend(b.param_list());
        }
        v
    }

    fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) {
        let mut cur = x;
        for b in &self.blocks {
            b.forward(ctx, ps, cur);
            cur = b.out();
        }
        if let Some(b) = &self.mms {
            b.forward(ctx, ps, cur);
            cur = b.out();
        }
        if let Some(b) = &self.spa {
            b.forward(ctx, ps, cur);
            cur = b.out();
        }
        if let Some(b) = &self.se {
            b.forward(ctx, ps, cur);
            cur = b.out();
        }
        if let Some(b) = &self.gcb {
            b.forward(ctx, ps, cur);
        }
    }

    /// Walk the stage in reverse, threading the grad through `links`.
    ///
    /// `links[i]` holds the grad wrt the input of the i-th unit; the last unit's
    /// grad goes to the caller's `d_in`. Every unit here is shape-preserving except
    /// nothing — the stage's blocks all run at one shape — so one scratch size does.
    fn backward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer, d_out: &DeviceBuffer, d_in: &DeviceBuffer) {
        // Inputs, in forward order: x, then each unit's output.
        let mut inputs: Vec<&DeviceBuffer> = vec![x];
        for b in &self.blocks {
            inputs.push(b.out());
        }
        if let Some(b) = &self.mms {
            inputs.push(b.out());
        }
        if let Some(b) = &self.spa {
            inputs.push(b.out());
        }
        if let Some(b) = &self.se {
            inputs.push(b.out());
        }
        if let Some(b) = &self.gcb {
            inputs.push(b.out());
        }
        // `inputs` now has one more entry than there are units; drop the last (the
        // stage output) so inputs[i] is unit i's input.
        inputs.pop();

        let n_units = inputs.len();
        let mut cur_d: &DeviceBuffer = d_out;
        // Reverse order: gcb, se, spa, mms, blocks (reversed).
        let mut unit = n_units;
        macro_rules! step {
            ($b:expr) => {{
                unit -= 1;
                let dst: &DeviceBuffer = if unit == 0 { d_in } else { &self.links[unit - 1] };
                $b.backward(ctx, ps, inputs[unit], cur_d, dst);
                cur_d = dst;
            }};
        }
        if let Some(b) = &self.gcb {
            step!(b);
        }
        if let Some(b) = &self.se {
            step!(b);
        }
        if let Some(b) = &self.spa {
            step!(b);
        }
        if let Some(b) = &self.mms {
            step!(b);
        }
        for b in self.blocks.iter().rev() {
            step!(b);
        }
        debug_assert_eq!(unit, 0, "every unit must have been walked");
        let _ = cur_d;
    }
}

/// ZipDepth.
pub struct ZipDepth {
    pub cfg: ZipConfig,
    pub in_shape: Shape,
    pub out_shape: Shape,
    /// The four encoder feature shapes + the stem's half-res one, kept because the
    /// backward's multi-consumer accumulations need element counts and a buffer
    /// does not carry its own.
    sh_half: Shape,
    sh_q: Shape,
    sh_s2: Shape,
    sh_s3: Shape,

    // ---- normalization (constants, but they ship in the state_dict) ----
    /// `1/std` per input channel, and `-mean/std` broadcast to `[N,3]`. Host-built
    /// from the `mean`/`std` buffers once and cached — see `norm_ready`.
    inv_std: DeviceBuffer,
    neg_mu: DeviceBuffer,
    normed: DeviceBuffer,
    norm_tmp: DeviceBuffer,
    norm_ready: std::cell::Cell<bool>,

    // ---- encoder ----
    stem_half: Conv,
    stem_quarter: Conv,
    stage1: Stage,
    down2: QARepBlock,
    stage2: Stage,
    down3: QARepBlock,
    stage3: Stage,
    down4: QARepBlock,
    stage4: Stage,
    spp: SPPF,
    cross_scale: MinimalCrossScale,

    // ---- decoder ----
    proj4: Conv,
    fuse3: UltraLightFusion,
    fuse2: UltraLightFusion,
    fuse1: UltraLightFusion,
    fuse_half: UltraLightFusion,
    head_half: Conv,
    convex_up: FastConvexUpsample,

    // ---- backward scratch, one per graph edge ----
    /// f_half's grad: the SUM of its two consumers (the head and the upsampler's
    /// feature input). Sized to fuse_half's OUTPUT, NOT the model output — those
    /// differ by 32x here, and reusing the smaller one overflows (silent heap
    /// corruption under the CPU JIT's trusted stores).
    d_fhalf_acc: DeviceBuffer,
    d_fhalf: DeviceBuffer,
    d_fhalf_head: DeviceBuffer,
    d_dhalf: DeviceBuffer,
    d_shalf_a: DeviceBuffer,
    d_shalf_b: DeviceBuffer,
    d_f1: DeviceBuffer,
    d_f2: DeviceBuffer,
    d_f3: DeviceBuffer,
    d_f4: DeviceBuffer,
    d_c1: DeviceBuffer,
    d_c2: DeviceBuffer,
    d_c3_fuse: DeviceBuffer,
    d_c3_cross: DeviceBuffer,
    d_c3: DeviceBuffer,
    d_c4_cross: DeviceBuffer,
    d_spp_in: DeviceBuffer,
    d_s4_pre: DeviceBuffer,
    d_s3_pre: DeviceBuffer,
    d_s2_pre: DeviceBuffer,
    d_s1_pre: DeviceBuffer,
    d_quarter: DeviceBuffer,
    d_shalf: DeviceBuffer,
    d_x: DeviceBuffer,
}

impl ZipDepth {
    pub fn new(gpu: &Gpu, cfg: ZipConfig, n: u32, train: bool) -> ZipDepth {
        let ctx = Ctx::new(gpu, crate::net::ids());
        ZipDepth::build(&ctx, cfg, n, train)
    }

    pub fn build(ctx: &Ctx, cfg: ZipConfig, n: u32, train: bool) -> ZipDepth {
        let d = cfg.dims;
        let half = cfg.half_ch();
        let sz = cfg.input;
        assert_eq!(sz % 32, 0, "the input side must be a multiple of 32 (five stride-2 stages)");
        let in_shape = Shape::new(n, 3, sz, sz);
        let use_global = cfg.global_mode != GlobalMode::None;
        assert_ne!(cfg.global_mode, GlobalMode::Full, "GlobalMode::Full needs EfficientGlobalAttention, which is not implemented (no released checkpoint uses it)");

        // ---- encoder: stem. ConvBN(k=3, s=2) -> pad = (k + (k-1)*(d-1))//2 = 1.
        let cbn = |ctx: &Ctx, p: &str, ins: Shape, cout: u32, k: u32, s: u32| {
            let pad = (k + (k - 1) * 0) / 2;
            Conv::with_names(ctx, p, ConvNames::torch_conv_bn(p), ins, ConvSpec::relu(cout, k, s, pad), train)
        };
        let stem_half = cbn(ctx, "encoder.stem_half", in_shape, half, 3, 2);
        let stem_quarter = cbn(ctx, "encoder.stem_quarter", stem_half.out_shape, d[0], 3, 2);

        let s_half_shape = stem_half.out_shape;
        let q = stem_quarter.out_shape;

        let stage1 = Stage::new(ctx, "encoder.stage1", q, cfg.depths[0], StageTail::None, train);
        let down2 = QARepBlock::new(ctx, "encoder.down2", q, d[1], 2, train);
        let s2 = down2.out_shape;
        let stage2 = Stage::new(
            ctx,
            "encoder.stage2",
            s2,
            cfg.depths[1],
            if use_global { StageTail::MmsAndStrip } else { StageTail::Mms },
            train,
        );
        let down3 = QARepBlock::new(ctx, "encoder.down3", s2, d[2], 2, train);
        let s3 = down3.out_shape;
        let stage3 = Stage::new(
            ctx,
            "encoder.stage3",
            s3,
            cfg.depths[2],
            if use_global { StageTail::SeAndGcb } else { StageTail::Se },
            train,
        );
        let down4 = QARepBlock::new(ctx, "encoder.down4", s3, d[3], 2, train);
        let s4 = down4.out_shape;
        let stage4 = Stage::new(ctx, "encoder.stage4", s4, cfg.depths[3], StageTail::None, train);

        let spp = SPPF::with_spec(
            ctx,
            "encoder.spp",
            s4,
            SppfSpec { hidden: d[3] / 4, c_out: d[3], act: Act::Relu },
            NameStyle::TorchConvBn,
            train,
        );
        // MinimalCrossScale(dim_high=dims[2], dim_low=dims[3]): "high" is the
        // higher-RESOLUTION stage3 map, "low" the stride-32 stage4 map.
        let cross_scale = MinimalCrossScale::new(ctx, "encoder.cross_scale", s3, s4, train);

        // ---- decoder ----
        let (ch4, ch3, ch2, ch1) = cfg.dec_chans();
        let proj4 = cbn(ctx, "decoder.proj4", s4, ch4, 1, 1);
        let f4 = proj4.out_shape;
        let fuse3 = UltraLightFusion::new(ctx, "decoder.fuse3", s3, f4, ch3, train);
        let fuse2 = UltraLightFusion::new(ctx, "decoder.fuse2", s2, fuse3.out_shape, ch2, train);
        let fuse1 = UltraLightFusion::new(ctx, "decoder.fuse1", q, fuse2.out_shape, ch1, train);
        let fuse_half = UltraLightFusion::new(ctx, "decoder.fuse_half", s_half_shape, fuse1.out_shape, cfg.half_dec_ch, train);
        let fh = fuse_half.out_shape;
        // A bare biased 3x3 conv: no BN, no activation. Its bias inits to 0.5.
        let head_half = Conv::with_names(
            ctx,
            "decoder.head_half",
            ConvNames {
                bias: "decoder.head_half.bias".into(),
                weight: "decoder.head_half.weight".into(),
                gamma: String::new(),
                beta: String::new(),
                run_mean: String::new(),
                run_var: String::new(),
            },
            fh,
            ConvSpec::relu(1, 3, 1, 1).with_norm(Norm::None).with_act(Act::None).with_bias(),
            train,
        );
        let dhalf = head_half.out_shape;
        let convex_up = FastConvexUpsample::new(
            ctx,
            "decoder.convex_up",
            if cfg.upsample_unfold { UpsampleKind::Unfold } else { UpsampleKind::Blend },
            fh,
            dhalf,
            UPSAMPLE_SCALE,
            1.0,
            train,
        );
        let out_shape = convex_up.out_shape;
        assert_eq!(
            (out_shape.h, out_shape.w),
            (in_shape.h, in_shape.w),
            "ZipDepth emits [B,1,H,W] at exactly the input resolution"
        );

        ZipDepth {
            cfg,
            in_shape,
            out_shape,
            sh_half: s_half_shape,
            sh_q: q,
            sh_s2: s2,
            sh_s3: s3,
            inv_std: ctx.act(3),
            neg_mu: ctx.act(n * 3),
            normed: ctx.act(in_shape.numel()),
            norm_tmp: ctx.act(in_shape.numel()),
            norm_ready: std::cell::Cell::new(false),
            d_fhalf_acc: ctx.act(fh.numel()),
            d_fhalf: ctx.act(fh.numel()),
            d_fhalf_head: ctx.act(fh.numel()),
            d_dhalf: ctx.act(dhalf.numel()),
            d_shalf_a: ctx.act(s_half_shape.numel()),
            d_shalf_b: ctx.act(s_half_shape.numel()),
            d_f1: ctx.act(fuse1.out_shape.numel()),
            d_f2: ctx.act(fuse2.out_shape.numel()),
            d_f3: ctx.act(fuse3.out_shape.numel()),
            d_f4: ctx.act(f4.numel()),
            d_c1: ctx.act(q.numel()),
            d_c2: ctx.act(s2.numel()),
            d_c3_fuse: ctx.act(s3.numel()),
            d_c3_cross: ctx.act(s3.numel()),
            d_c3: ctx.act(s3.numel()),
            d_c4_cross: ctx.act(s4.numel()),
            d_spp_in: ctx.act(s4.numel()),
            d_s4_pre: ctx.act(s4.numel()),
            d_s3_pre: ctx.act(s3.numel()),
            d_s2_pre: ctx.act(s2.numel()),
            d_s1_pre: ctx.act(q.numel()),
            d_quarter: ctx.act(q.numel()),
            d_shalf: ctx.act(s_half_shape.numel()),
            d_x: ctx.act(in_shape.numel()),
            stem_half,
            stem_quarter,
            stage1,
            down2,
            stage2,
            down3,
            stage3,
            down4,
            stage4,
            spp,
            cross_scale,
            proj4,
            fuse3,
            fuse2,
            fuse1,
            fuse_half,
            head_half,
            convex_up,
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        self.convex_up.out()
    }

    pub fn set_eval(&self, on: bool) {
        for c in [&self.stem_half, &self.stem_quarter, &self.proj4, &self.head_half] {
            c.set_eval(on);
        }
        self.stage1.set_eval(on);
        self.stage2.set_eval(on);
        self.stage3.set_eval(on);
        self.stage4.set_eval(on);
        self.down2.set_eval(on);
        self.down3.set_eval(on);
        self.down4.set_eval(on);
        self.spp.set_eval(on);
        self.cross_scale.set_eval(on);
        self.fuse3.set_eval(on);
        self.fuse2.set_eval(on);
        self.fuse1.set_eval(on);
        self.fuse_half.set_eval(on);
        self.convex_up.set_eval(on);
    }

    pub fn set_update_running(&self, on: bool) {
        for c in [&self.stem_half, &self.stem_quarter, &self.proj4] {
            c.set_update_running(on);
        }
        self.stage1.set_update_running(on);
        self.stage2.set_update_running(on);
        self.stage3.set_update_running(on);
        self.stage4.set_update_running(on);
        self.down2.set_update_running(on);
        self.down3.set_update_running(on);
        self.down4.set_update_running(on);
        self.spp.set_update_running(on);
        self.fuse3.set_update_running(on);
        self.fuse2.set_update_running(on);
        self.fuse1.set_update_running(on);
        self.fuse_half.set_update_running(on);
        self.convex_up.set_update_running(on);
    }

    /// Every parameter, in the reference's `state_dict` order.
    ///
    /// MUST equal `cfg.param_list()`'s names — `p3_layout` asserts exactly that, so
    /// the device-free config and the built graph cannot drift apart.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v: Vec<(String, usize)> = vec![("mean".into(), 3), ("std".into(), 3)];
        v.extend(self.stem_half.param_list());
        v.extend(self.stem_quarter.param_list());
        v.extend(self.stage1.param_list());
        v.extend(self.down2.param_list());
        v.extend(self.stage2.param_list());
        v.extend(self.down3.param_list());
        v.extend(self.stage3.param_list());
        v.extend(self.down4.param_list());
        v.extend(self.stage4.param_list());
        v.extend(self.spp.param_list());
        v.extend(self.cross_scale.param_list());
        v.extend(self.proj4.param_list());
        v.extend(self.fuse3.param_list());
        v.extend(self.fuse2.param_list());
        v.extend(self.fuse1.param_list());
        v.extend(self.fuse_half.param_list());
        v.extend(self.head_half.param_list());
        v.extend(self.convex_up.param_list());
        v
    }

    /// `(x - mean)/std`, the ImageNet normalize the reference does INSIDE forward
    /// (`architecture.py:641`) rather than in preprocessing — which is why `mean`
    /// and `std` are `state_dict` buffers and must be carried.
    ///
    /// Two existing kernels, no new one: `scale_chan` at `(c=3, inner=H*W)` gives
    /// `x/std`, and `add_chan_bcast` adds a per-(image,channel) `-mean/std`. The
    /// `[N,3]` vector is the same row repeated — add_chan_bcast is per-image, and a
    /// per-image constant is a special case of that.
    ///
    /// Both derived tables are host-built from `mean`/`std` ONCE and cached: they
    /// are constants, and a per-frame readback would be a sync point on the demo's
    /// hot path.
    fn normalize(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) {
        if !self.norm_ready.get() {
            let mean = ctx.gpu.read(ps.w("mean"), 3);
            let std = ctx.gpu.read(ps.w("std"), 3);
            let inv: Vec<f32> = std.iter().map(|s| 1.0 / s).collect();
            let mut neg = Vec::with_capacity((self.in_shape.n * 3) as usize);
            for _ in 0..self.in_shape.n {
                for c in 0..3 {
                    neg.push(-mean[c] / std[c]);
                }
            }
            ctx.gpu.write(&self.inv_std, bytemuck::cast_slice(&inv));
            ctx.gpu.write(&self.neg_mu, bytemuck::cast_slice(&neg));
            self.norm_ready.set(true);
        }
        let n = self.in_shape.numel();
        let hw = self.in_shape.h * self.in_shape.w;
        // scale_chan uniform is [total, c, inner]: channel = (idx/inner) % c. Here
        // c=3 over the whole NCHW tensor, inner = H*W.
        let s = ctx.step(ctx.ids.scale_chan, &[x, &self.inv_std, &self.norm_tmp], &[n, 3, hw], n);
        ctx.gpu.submit(&[], &[s]);
        let s = ctx.step(
            ctx.ids.need(ctx.ids.add_chan_bcast, "add_chan_bcast"),
            &[&self.norm_tmp, &self.neg_mu, &self.normed],
            &[self.in_shape.n, 3, hw],
            n,
        );
        ctx.gpu.submit(&[], &[s]);
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) {
        self.normalize(ctx, ps, x);
        self.stem_half.forward(ctx, ps, &self.normed);
        self.stem_quarter.forward(ctx, ps, self.stem_half.out());
        self.stage1.forward(ctx, ps, self.stem_quarter.out());
        self.down2.forward(ctx, ps, self.stage1.out());
        self.stage2.forward(ctx, ps, self.down2.out());
        self.down3.forward(ctx, ps, self.stage2.out());
        self.stage3.forward(ctx, ps, self.down3.out());
        self.down4.forward(ctx, ps, self.stage3.out());
        self.stage4.forward(ctx, ps, self.down4.out());
        self.spp.forward(ctx, ps, self.stage4.out());
        // cross_scale REBINDS both s3 and s4 (`s3, s4 = self.cross_scale(s3, s4)`),
        // so the decoder's `c3` is the exchanged map, not stage3's raw output.
        self.cross_scale.forward(ctx, ps, self.stage3.out(), self.spp.out());

        self.proj4.forward(ctx, ps, self.cross_scale.out_low());
        self.fuse3.forward(ctx, ps, self.cross_scale.out_high(), self.proj4.out());
        self.fuse2.forward(ctx, ps, self.stage2.out(), self.fuse3.out());
        self.fuse1.forward(ctx, ps, self.stage1.out(), self.fuse2.out());
        self.fuse_half.forward(ctx, ps, self.stem_half.out(), self.fuse1.out());
        self.head_half.forward(ctx, ps, self.fuse_half.out());
        self.convex_up.forward(ctx, ps, self.fuse_half.out(), self.head_half.out());
    }

    /// Backward from `d_out` (grad wrt the depth map) to every parameter.
    ///
    /// The input image carries no gradient, so `d_x` is written and discarded —
    /// `Conv::backward` always produces it.
    pub fn backward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer, d_out: &DeviceBuffer) {
        let _ = x;
        // ---- decoder ----
        // f_half feeds BOTH the head and the upsampler's feature input.
        self.convex_up.backward(
            ctx,
            ps,
            self.fuse_half.out(),
            self.head_half.out(),
            d_out,
            &self.d_fhalf,
            &self.d_dhalf,
        );
        self.head_half.backward(ctx, ps, self.fuse_half.out(), &self.d_dhalf, &self.d_fhalf_head);
        let nfh = self.fuse_half.out_shape.numel();
        let s = ctx.step(ctx.ids.add2, &[&self.d_fhalf, &self.d_fhalf_head, &self.d_fhalf_acc], &[nfh], nfh);
        ctx.gpu.submit(&[], &[s]);

        self.fuse_half.backward(ctx, ps, self.stem_half.out(), &self.d_fhalf_acc, &self.d_shalf_a, &self.d_f1);
        self.fuse1.backward(ctx, ps, self.stage1.out(), &self.d_f1, &self.d_c1, &self.d_f2);
        self.fuse2.backward(ctx, ps, self.stage2.out(), &self.d_f2, &self.d_c2, &self.d_f3);
        self.fuse3.backward(ctx, ps, self.cross_scale.out_high(), &self.d_f3, &self.d_c3_fuse, &self.d_f4);
        self.proj4.backward(ctx, ps, self.cross_scale.out_low(), &self.d_f4, &self.d_c4_cross);

        // ---- encoder ----
        self.cross_scale.backward(
            ctx,
            ps,
            self.stage3.out(),
            self.spp.out(),
            &self.d_c3_fuse,
            &self.d_c4_cross,
            &self.d_c3_cross,
            &self.d_spp_in,
        );
        self.spp.backward(ctx, ps, self.stage4.out(), &self.d_spp_in, &self.d_s4_pre);
        self.stage4.backward(ctx, ps, self.down4.out(), &self.d_s4_pre, &self.d_c4_cross);
        self.down4.backward(ctx, ps, self.stage3.out(), &self.d_c4_cross, &self.d_c3);
        // stage3's output has TWO consumers: down4 and cross_scale.
        let n3 = self.sh_s3.numel();
        let s = ctx.step(ctx.ids.add2, &[&self.d_c3, &self.d_c3_cross, &self.d_s3_pre], &[n3], n3);
        ctx.gpu.submit(&[], &[s]);
        self.stage3.backward(ctx, ps, self.down3.out(), &self.d_s3_pre, &self.d_c3_cross);
        self.down3.backward(ctx, ps, self.stage2.out(), &self.d_c3_cross, &self.d_s2_pre);
        // stage2's output has TWO consumers: down3 and fuse2.
        let n2 = self.sh_s2.numel();
        let s = ctx.step(ctx.ids.add2, &[&self.d_s2_pre, &self.d_c2, &self.d_c2_acc()], &[n2], n2);
        ctx.gpu.submit(&[], &[s]);
        self.stage2.backward(ctx, ps, self.down2.out(), &self.d_c2_acc(), &self.d_s2_pre);
        self.down2.backward(ctx, ps, self.stage1.out(), &self.d_s2_pre, &self.d_s1_pre);
        // stage1's output has TWO consumers: down2 and fuse1.
        let n1 = self.sh_q.numel();
        let s = ctx.step(ctx.ids.add2, &[&self.d_s1_pre, &self.d_c1, &self.d_quarter], &[n1], n1);
        ctx.gpu.submit(&[], &[s]);
        self.stage1.backward(ctx, ps, self.stem_quarter.out(), &self.d_quarter, &self.d_s1_pre);
        self.stem_quarter.backward(ctx, ps, self.stem_half.out(), &self.d_s1_pre, &self.d_shalf_b);
        // s_half has TWO consumers: stem_quarter and fuse_half.
        let nh = self.sh_half.numel();
        let s = ctx.step(ctx.ids.add2, &[&self.d_shalf_a, &self.d_shalf_b, &self.d_shalf], &[nh], nh);
        ctx.gpu.submit(&[], &[s]);
        self.stem_half.backward(ctx, ps, &self.normed, &self.d_shalf, &self.d_x);
    }

    fn d_c2_acc(&self) -> &DeviceBuffer {
        &self.d_s2_pre
    }
}

/// Which attention tail a stage carries. See [`Stage`]'s doc: the order is part of
/// the checkpoint's index numbering, not a free choice.
enum StageTail {
    None,
    Mms,
    MmsAndStrip,
    Se,
    SeAndGcb,
}

impl Stage {
    fn new(ctx: &Ctx, prefix: &str, shape: Shape, depth: u32, tail: StageTail, train: bool) -> Stage {
        let mut blocks = Vec::new();
        for i in 0..depth {
            blocks.push(QARepBlock::new(ctx, &format!("{prefix}.{i}"), shape, shape.c, 1, train));
        }
        let mut idx = depth;
        let next = |i: &mut u32| {
            let s = format!("{prefix}.{i}");
            *i += 1;
            s
        };
        let (mms, spa, se, gcb) = match tail {
            StageTail::None => (None, None, None, None),
            StageTail::Mms => (Some(MinimalMultiScale::new(ctx, &next(&mut idx), shape, train)), None, None, None),
            StageTail::MmsAndStrip => {
                let m = MinimalMultiScale::new(ctx, &next(&mut idx), shape, train);
                let s = StripPoolingAttention::new(ctx, &next(&mut idx), shape, train);
                (Some(m), Some(s), None, None)
            }
            StageTail::Se => (None, None, Some(ChannelAttention::new(ctx, &next(&mut idx), shape)), None),
            StageTail::SeAndGcb => {
                let se = ChannelAttention::new(ctx, &next(&mut idx), shape);
                let g = GlobalContextBlock::new(ctx, &next(&mut idx), shape, 4, train);
                (None, None, Some(se), Some(g))
            }
        };
        let n_units = blocks.len()
            + mms.is_some() as usize
            + spa.is_some() as usize
            + se.is_some() as usize
            + gcb.is_some() as usize;
        // One grad link per inter-unit edge.
        let links = (0..n_units.saturating_sub(1)).map(|_| ctx.act(shape.numel())).collect();
        Stage { blocks, mms, spa, se, gcb, links }
    }
}

