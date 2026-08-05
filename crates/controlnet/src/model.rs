// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The SDXL ControlNet forward graph.
//!
//! **Composition, not new math, and not even a new block set.** The trainable
//! copy of the backbone's early blocks IS `unet::model::Rec` — the same
//! conditioning chain, the same `down_path`, the same `mid_block`, recorded
//! from the same tensor prefixes with the same tap names. This crate adds
//! exactly three things:
//!
//! 1. the **conditioning-image embedder** (a conv/SiLU stack that takes a pixel
//!    -resolution image down to the latent grid),
//! 2. the **zero-convs** — one 1x1 conv per injection point, which are plain
//!    `vae::blocks::Builder::conv` calls,
//! 3. `conditioning_scale`, a one-element device buffer read by `scale_chan`.
//!
//! ```text
//! emb   = time_embedding(sinusoid(t)) + add_embedding([pooled ‖ sinusoid(time_ids)])
//! cond  = cond_out(silu(block5(… silu(block0(silu(cond_in(image)))))))
//! h     = conv_in(x) + cond                                ; res = [h]
//! down i: for j in 0..layers:  h = resnet(h, emb) [; h = transformer(h, enc)]
//!                              res.push(h)
//!         if not last:         h = conv_s2(h) ; res.push(h)
//! mid   : h = resnet(h, emb) ; h = transformer(h, enc) ; h = resnet(h, emb)
//! out   : down_k = scale · zero_k(res[k]) ;  mid = scale · zero_mid(h)
//! ```
//!
//! ### The four conventions this graph pins, each verified against diffusers
//! 1. **The conditioning embedder's SiLU is AFTER every conv except the last.**
//!    `ControlNetConditioningEmbedding.forward` is
//!    `silu(conv_in(x))`, then `silu(block(·))` for all six blocks, then
//!    `conv_out(·)` with **no** activation. An activation on `conv_out` is a
//!    plausible-looking graph that changes every residual.
//! 2. **The conditioning embedding is added to `conv_in`'s output**, i.e. in
//!    latent space and after the input convolution — not concatenated to the
//!    latent, and not added to the latent before `conv_in`.
//! 3. **`conditioning_scale` multiplies the ZERO-CONV OUTPUT and nothing
//!    else.** It is applied inside `ControlNetModel.forward` after both the
//!    down and the mid zero-conv, so it never touches the trainable copy's
//!    activations. `tools/goldens/controlnet_dump_reference.py` asserts this by dumping
//!    a second forward at 0.75 and checking it is exactly 0.75x the first.
//! 4. **The residual list is the backbone's skip stack, in the same order.**
//!    `controlnet_down_blocks.k` conditions `skip_stack()[k]`, so the k-th
//!    zero-conv's channel width is `skip_stack()[k]` and NOT
//!    `block_out_channels[k]` — the two agree for the first entry and diverge
//!    immediately after.
//!
//! ### Zero-convs are 1x1 convs, so they are `Builder::conv(k = 1, pad = 0)`
//! Not a GEMM over `[HW, C]` rows, which is the other obvious lowering: that
//! would need a permute in and a permute out for a kernel whose entire work is
//! `C` MACs per position, and `docs/imaging/plan.md` §3.1.1 measured those
//! permutes running at 14–33 % of the roofline.

use gpu_core::{DeviceBuffer, Gpu, Step};
use unet::model::Rec;

use crate::adapter::{ControlSource, InjectionPoint, Residuals};
use crate::config::ControlNetConfig;
use crate::import::Tensors;

/// The one kernel slot appended to `unet::model::KERNELS`.
const K_SCALE: usize = unet::model::KERNELS.len();

/// This model's kernel set: **`unet::model::KERNELS` verbatim** (so the
/// backbone's block recorder finds every slot at the index it resolved) plus
/// `scale_chan` for `conditioning_scale`.
///
/// Being a strict prefix-extension is what lets one `Gpu` drive both models:
/// `unet::Unet::new` requires only that its own slots are a prefix, exactly as
/// `unet::model::KERNELS` extends `vae::blocks::KERNELS`. A UNet + ControlNet
/// pipeline therefore builds ONE device with THIS set, not two devices.
pub const KERNELS: [(&str, &str); unet::model::KERNELS.len() + 1] = kernel_set();

const fn kernel_set() -> [(&'static str, &'static str); unet::model::KERNELS.len() + 1] {
    let mut k = [("", ""); unet::model::KERNELS.len() + 1];
    let mut i = 0;
    while i < unet::model::KERNELS.len() {
        k[i] = unet::model::KERNELS[i];
        i += 1;
    }
    // `scale_chan` with `c = 1, inner = 1` is `out[i] = x[i] * scale[0]` — an
    // out-of-place multiply by a ONE-ELEMENT DEVICE BUFFER. Out-of-place
    // matters: the pre-scale zero-conv output is a parity tap. Device-resident
    // matters for the same reason `restore`'s fidelity dial is a buffer:
    // changing `conditioning_scale` is then a write, not a graph rebuild.
    k[K_SCALE] = ("scale_chan", kernels::SCALE_CHAN);
    k
}

/// A recorded SDXL ControlNet at one latent resolution and one text-token
/// count.
pub struct ControlNet {
    gpu: Gpu,
    cfg: ControlNetConfig,
    hw: (u32, u32),
    cond_hw: (u32, u32),
    t_enc: u32,
    sample_in: DeviceBuffer,
    cond_in: DeviceBuffer,
    enc_in: DeviceBuffer,
    temb_in: DeviceBuffer,
    aug_in: DeviceBuffer,
    scale_in: DeviceBuffer,
    /// `(name, buffer, numel)` per injection point, in `skip_stack()` order
    /// then mid.
    outs: Vec<(String, DeviceBuffer, usize)>,
    steps: Vec<Step>,
    taps: Vec<(String, DeviceBuffer, usize)>,
}

impl ControlNet {
    /// Record the graph for a `h × w` latent and `t_enc` text tokens.
    ///
    /// The conditioning image is at `cond_downscale() ×` that, i.e. 8× for
    /// every released SDXL ControlNet — the same factor as the VAE, which is
    /// what makes the embedding land on the latent grid.
    ///
    /// `taps` records every stage output for the parity ladder; it pins buffers
    /// and therefore disables the activation pool, so a production build passes
    /// `false`.
    pub fn new(
        gpu: Gpu,
        cfg: ControlNetConfig,
        tensors: &Tensors,
        h: u32,
        w: u32,
        t_enc: u32,
        taps: bool,
    ) -> ControlNet {
        cfg.validate().expect("controlnet: invalid config");
        let bb = cfg.backbone.clone();
        let levels = bb.levels();
        let scale = 1u32 << (levels - 1);
        assert!(
            h.is_multiple_of(scale) && w.is_multiple_of(scale),
            "controlnet: latent {h}x{w} is not a multiple of the {scale}x downscale"
        );
        let c0 = bb.block_out_channels[0];
        let ds = cfg.cond_downscale();
        let (ph, pw) = (h * ds, w * ds);

        let sample_in = gpu.storage((bb.in_channels * h * w) as u64);
        let cond_in = gpu.storage((cfg.conditioning_channels * ph * pw) as u64);
        let enc_in = gpu.storage((t_enc * bb.cross_attention_dim) as u64);
        let temb_in = gpu.storage(c0 as u64);
        let aug_in = gpu.storage(bb.projection_class_embeddings_input_dim as u64);
        let scale_in = gpu.storage(1);

        // No up path, so the up term must not even be indexed.
        let s_words = unet::model::attn_slab_words(&bb, h, w, false);
        let mut r = Rec::new(&gpu, &bb, tensors, t_enc, s_words, taps);

        r.conditioning(&bb, &temb_in, &aug_in);

        // ---- the conditioning-image embedder ---------------------------------
        let ce = &cfg.conditioning_embedding_out_channels;
        let (mut eh, mut ew) = (ph, pw);
        let mut e = r.blocks().conv(
            "controlnet_cond_embedding.conv_in",
            cfg.conditioning_channels,
            ce[0],
            3,
            1,
            eh,
            ew,
            &cond_in,
        );
        r.blocks().tap("cond.conv_in".into(), &e, ce[0] * eh * ew);
        let mut ea = r.blocks().silu(ce[0] * eh * ew, &e);
        for i in 0..ce.len() - 1 {
            let (cin, cout) = (ce[i], ce[i + 1]);
            // (a) same-resolution conv
            e = r.blocks().conv(
                &format!("controlnet_cond_embedding.blocks.{}", 2 * i),
                cin,
                cin,
                3,
                1,
                eh,
                ew,
                &ea,
            );
            r.blocks().tap(format!("cond.block{}", 2 * i), &e, cin * eh * ew);
            r.blocks().free((cin as u64) * (eh as u64) * (ew as u64), ea);
            ea = r.blocks().silu(cin * eh * ew, &e);
            r.blocks().free((cin as u64) * (eh as u64) * (ew as u64), e);
            // (b) stride-2 widening conv. `Downsample`-style symmetric pad 1,
            // so `ho = (H + 2 - 3)/2 + 1 = H/2` for even H.
            e = r.blocks().conv_s(
                &format!("controlnet_cond_embedding.blocks.{}", 2 * i + 1),
                cin,
                cout,
                3,
                2,
                1,
                eh,
                ew,
                eh / 2,
                ew / 2,
                &ea,
            );
            r.blocks().free((cin as u64) * (eh as u64) * (ew as u64), ea);
            eh /= 2;
            ew /= 2;
            r.blocks().tap(format!("cond.block{}", 2 * i + 1), &e, cout * eh * ew);
            ea = r.blocks().silu(cout * eh * ew, &e);
            r.blocks().free((cout as u64) * (eh as u64) * (ew as u64), e);
        }
        assert_eq!((eh, ew), (h, w), "controlnet: the embedder lands at {eh}x{ew}, latent is {h}x{w}");
        let clast = *ce.last().expect("validated >= 2 stages");
        // NO activation on `conv_out` — convention 1.
        let cond = r.blocks().conv("controlnet_cond_embedding.conv_out", clast, c0, 3, 1, h, w, &ea);
        r.blocks().tap("cond.out".into(), &cond, c0 * h * w);
        r.blocks().free((clast as u64) * (h as u64) * (w as u64), ea);

        // ---- conv_in + the conditioning add ----------------------------------
        let cin = r.blocks().conv("conv_in", bb.in_channels, c0, 3, 1, h, w, &sample_in);
        r.blocks().tap("conv_in".into(), &cin, c0 * h * w);
        let x = r.blocks().add(c0 * h * w, &cin, &cond);
        r.blocks().tap("sample_cond".into(), &x, c0 * h * w);
        r.blocks().free((c0 as u64) * (h as u64) * (w as u64), cond);

        // ---- the trainable copy ----------------------------------------------
        let (hh, skips, ch, cw) = r.down_path(&bb, h, w, &enc_in, &x);
        let mid = r.mid_block(&bb, ch, cw, &enc_in, &hh);
        let cmid = *bb.block_out_channels.last().expect("levels >= 1");

        // ---- zero-convs + conditioning_scale ---------------------------------
        let points = cfg.injection_points(h, w);
        assert_eq!(points.len(), skips.len() + 1, "controlnet: {} points, {} residuals", points.len(), skips.len() + 1);
        let mut outs: Vec<(String, DeviceBuffer, usize)> = Vec::with_capacity(points.len());
        for (k, (buf, c, sh, sw)) in skips.into_iter().enumerate() {
            assert_eq!(
                (c, sh, sw),
                match points[k].layout {
                    crate::adapter::Layout::Spatial { c, h, w } => (c, h, w),
                    _ => unreachable!("a UNet injection point is spatial"),
                },
                "controlnet: residual {k} shape disagrees with the config"
            );
            // Tap the residual as it ENTERS the zero-conv, not only as it
            // leaves. This is what gates convention 4 — that
            // `controlnet_down_blocks.k` is fed `skip_stack()[k]` — and it is
            // the one thing a tap on the conv's output cannot see: a permuted
            // feed among the four 320-channel points produces the right shapes
            // everywhere and a wrong residual everywhere.
            r.blocks().tap(format!("zero{k}.in"), &buf, c * sh * sw);
            let z = r.blocks().conv(&format!("controlnet_down_blocks.{k}"), c, c, 1, 0, sh, sw, &buf);
            r.blocks().tap(format!("zero{k}"), &z, c * sh * sw);
            outs.push((points[k].name.clone(), scale_buf(&mut r, &z, c * sh * sw, &scale_in), (c * sh * sw) as usize));
        }
        r.blocks().tap("zero_mid.in".into(), &mid, cmid * ch * cw);
        let zm = r.blocks().conv("controlnet_mid_block", cmid, cmid, 1, 0, ch, cw, &mid);
        r.blocks().tap("zero_mid".into(), &zm, cmid * ch * cw);
        outs.push((
            "mid".to_string(),
            scale_buf(&mut r, &zm, cmid * ch * cw, &scale_in),
            (cmid * ch * cw) as usize,
        ));

        let (steps, taps) = r.into_blocks().finish();
        ControlNet {
            gpu,
            cfg,
            hw: (h, w),
            cond_hw: (ph, pw),
            t_enc,
            sample_in,
            cond_in,
            enc_in,
            temb_in,
            aug_in,
            scale_in,
            outs,
            steps,
            taps,
        }
    }

    pub fn config(&self) -> &ControlNetConfig {
        &self.cfg
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// `(h, w)` of the conditioning image this graph was recorded for.
    pub fn cond_size(&self) -> (u32, u32) {
        self.cond_hw
    }

    /// One evaluation, producing the residual at every injection point.
    ///
    /// * `sample` — `[in_channels · H · W]`, NCHW, batch 1: the SAME noisy
    ///   latent the UNet is about to be given.
    /// * `cond` — `[conditioning_channels · 8H · 8W]`, `[0, 1]` CHW; see
    ///   [`crate::cond`].
    /// * `enc`, `pooled`, `time_ids` — identical to the UNet's.
    /// * `scale` — diffusers' `conditioning_scale`.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        sample: &[f32],
        timestep: f32,
        enc: &[f32],
        pooled: &[f32],
        time_ids: &[f32],
        cond: &[f32],
        scale: f32,
    ) -> Residuals {
        let c = &self.cfg.backbone;
        let (h, w) = self.hw;
        let (ph, pw) = self.cond_hw;
        assert_eq!(sample.len(), (c.in_channels * h * w) as usize, "controlnet: sample size");
        assert_eq!(
            cond.len(),
            (self.cfg.conditioning_channels * ph * pw) as usize,
            "controlnet: conditioning image must be {}x{ph}x{pw}",
            self.cfg.conditioning_channels
        );
        assert_eq!(enc.len(), (self.t_enc * c.cross_attention_dim) as usize, "controlnet: encoder_hidden_states size");
        assert_eq!(pooled.len(), c.pooled_dim() as usize, "controlnet: pooled text size");
        assert_eq!(time_ids.len(), unet::config::N_TIME_IDS as usize, "controlnet: time_ids count");

        let temb = model::hostmath::timestep_embedding(
            timestep,
            c.block_out_channels[0] as usize,
            c.flip_sin_to_cos,
            c.freq_shift as f64,
            10_000.0,
        );
        // The added-conditioning concat is `unet::hostemb::added_cond` — the
        // ControlNet's `text_time` chain is the UNet's, module for module.
        let aug =
            unet::hostemb::added_cond(pooled, time_ids, c.addition_time_embed_dim, c.flip_sin_to_cos, c.freq_shift);
        self.gpu.write_f32(&self.sample_in, sample);
        self.gpu.write_f32(&self.cond_in, cond);
        self.gpu.write_f32(&self.enc_in, enc);
        self.gpu.write_f32(&self.temb_in, &temb);
        self.gpu.write_f32(&self.aug_in, &aug);
        self.gpu.write_f32(&self.scale_in, &[scale]);
        self.gpu.submit(&[], &self.steps);

        let mut r = Residuals::new();
        for (name, buf, n) in &self.outs {
            r.insert(name.clone(), self.gpu.read(buf, *n));
        }
        r
    }

    /// A recorded stage output (only when the model was built with `taps`).
    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        let (_, buf, len) = self.taps.iter().find(|(n, _, _)| n == name)?;
        Some(self.gpu.read(buf, *len))
    }

    pub fn tap_names(&self) -> Vec<&str> {
        self.taps.iter().map(|(n, _, _)| n.as_str()).collect()
    }
}

impl ControlSource for ControlNet {
    fn injection_points(&self) -> Vec<InjectionPoint> {
        self.cfg.injection_points(self.hw.0, self.hw.1)
    }
}

/// `y = x · scale[0]` via `scale_chan` with `c = 1, inner = 1`.
fn scale_buf(r: &mut Rec<'_>, x: &DeviceBuffer, n: u32, scale: &DeviceBuffer) -> DeviceBuffer {
    let y = r.blocks().act(n as u64);
    let g = r.blocks().gpu();
    // `scale_chan` Params: [total, c, inner]; bufs [x, scale, out].
    let step = g.step(K_SCALE, &[x, scale, &y], &[n, 1, 1], n);
    r.blocks().push_step(step);
    y
}

#[cfg(test)]
mod tests {
    /// The appended slot holds the kernel its constant names, every inherited
    /// slot is still the UNet's, and nothing is empty. `unet::model::KERNELS`
    /// is copied by index here; an off-by-one would resolve `layernorm` to some
    /// other pipeline and only fail deep inside a recorded graph.
    #[test]
    fn the_kernel_set_extends_the_unets_exactly() {
        assert_eq!(super::KERNELS.len(), unet::model::KERNELS.len() + 1);
        for (i, k) in unet::model::KERNELS.iter().enumerate() {
            assert_eq!(super::KERNELS[i], *k, "slot {i}");
        }
        assert_eq!(super::KERNELS[super::K_SCALE].0, "scale_chan");
        assert!(super::KERNELS.iter().all(|(n, s)| !n.is_empty() && !s.is_empty()));
    }
}
