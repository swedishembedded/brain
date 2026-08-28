// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `GLVControl` (the trunk) plus the frozen SDXL UNet, recorded into ONE
//! `Rec`/`Builder`/`Gpu`/submit.
//!
//! The one-graph design is the whole point of the now-public
//! [`sdxlunet::model::Unet::record_into`] and [`sdxlunet::model::Rec::set_fuse`]:
//! recording the trunk first, taking a plain reference to its 10 hidden
//! states, installing them (through [`crate::adaptors::Adaptors`]) as the
//! frozen backbone's `SkipFuse`, then recording the backbone's own
//! conditioning/`conv_in`/down/mid/up walk into the SAME tape means one
//! `Gpu::submit` runs the whole restoration forward - no host round-trip
//! between the trunk and the backbone, and no second device.
//!
//! ```text
//! r = Rec::new(gpu, backbone_cfg, tensors, t_enc, taps)
//! r.set_prefix("control_model.")
//! hs = trunk::record(&mut r, trunk_cfg, "control_model.", h, w, enc, hint, sample, temb, aug)
//! r.take_temb_act()                 # trunk's own resnets already consumed it
//! r.set_prefix("")
//! adaptors = Adaptors::new(adaptor_cfg, hs, control_scale)
//! r.blocks().set_mix_ids(..)        # edm_mix/scale_row - the ZeroSFT/ZeroCrossAttn lerp
//! r.set_fuse(&adaptors)
//! Unet::record_into(&mut r, backbone_cfg, h, w, &inputs, control = false)
//! ```
//!
//! `sample`/`enc`/`temb`/`aug` are shared, single-write device inputs: the
//! trunk and the frozen backbone are evaluated at the SAME noisy latent,
//! text encoding, timestep and added conditioning within one denoiser call
//! (only their WEIGHTS differ, under different prefixes) - only the hint
//! (`_z`, the degradation-robust encode) is trunk-exclusive.
//!
//! This crate's kernel set is `sdxlunet::model::KERNELS` (a strict
//! prefix-extension, the same `controlnet::model::kernel_set` trick - so one
//! `Gpu` drives trunk + adaptors + backbone) plus `edm_mix`/`scale_row` for
//! [`vae::blocks::Builder::mix`]. `ZeroCrossAttn`'s cross-attention reuses
//! the backbone's OWN `attn_*_cross` slots (already in `sdxlunet::model::KERNELS`
//! via `Rec`'s `XformerIds`) - no separate registration needed.

use gpu_core::{DeviceBuffer, Gpu, Step};
use sdxlunet::import::Tensors;
use sdxlunet::model::{Inputs, Rec, Unet};

use vae::blocks::skipfuse::SkipFuse;
use vae::blocks::PackedTensors;

use crate::adaptors::Adaptors;
use crate::config::SupirConfig;

/// The `edm_mix` slot appended to `sdxlunet::model::KERNELS`.
const K_EDM_MIX: usize = sdxlunet::model::KERNELS.len();
/// The `scale_row` slot - `Op::Mix`'s backward. Registered even though this
/// crate never trains: `vae::blocks::Builder::set_mix_ids` requires both
/// slots to exist, and a training build (Step 6's `finetune.rs`, not yet
/// written) will need it resolved at the same index either way.
const K_SCALE_ROW: usize = sdxlunet::model::KERNELS.len() + 1;

/// This model's kernel set: `sdxlunet::model::KERNELS` verbatim, plus the
/// two `Op::Mix` slots. A strict prefix-extension, so the SAME `Gpu` that
/// records the frozen backbone's blocks also runs the trunk's (identical
/// block shapes, different tensor names) and the adaptors' `mix` calls.
pub const KERNELS: [(&str, &str); sdxlunet::model::KERNELS.len() + 2] = kernel_set();

const fn kernel_set() -> [(&'static str, &'static str); sdxlunet::model::KERNELS.len() + 2] {
    let mut k = [("", ""); sdxlunet::model::KERNELS.len() + 2];
    let mut i = 0;
    while i < sdxlunet::model::KERNELS.len() {
        k[i] = sdxlunet::model::KERNELS[i];
        i += 1;
    }
    k[K_EDM_MIX] = ("edm_mix", kernels::EDM_MIX);
    k[K_SCALE_ROW] = ("scale_row", kernels::SCALE_ROW);
    k
}

/// A recorded SUPIR forward (trunk + adaptors + frozen UNet) at one latent
/// resolution and one text-token count, at a fixed `control_scale` (baked
/// into the graph - see [`crate::adaptors::Adaptors`]'s field doc for why a
/// per-step ramp is a design question for `pipeline.rs`, not this module).
pub struct Supir {
    gpu: Gpu,
    cfg: SupirConfig,
    hw: (u32, u32),
    t_enc: u32,
    sample_in: DeviceBuffer,
    hint_in: DeviceBuffer,
    enc_in: DeviceBuffer,
    temb_in: DeviceBuffer,
    aug_in: DeviceBuffer,
    out: DeviceBuffer,
    /// The reverse-mode tape, present only on a [`Supir::new_train`] build -
    /// mirrors [`sdxlunet::model::Unet`]'s own `trace` field exactly, for the
    /// same reason: [`crate::train::SupirTrainer`] is built on it.
    trace: Option<vae::blocks::grad::Trace>,
    steps: Vec<Step>,
    taps: Vec<(String, DeviceBuffer, usize)>,
}

impl Supir {
    /// Record the full graph. `tensors` must hold BOTH the frozen backbone's
    /// weights (unprefixed, `sdxlunet::import`/`sdxlunet::init` shape) and
    /// SUPIR's own delta (`control_model.`/`project_modules.`-prefixed,
    /// `crate::import`/`crate::init` shape) in one map - the caller merges
    /// them, since [`Rec`] itself takes exactly one [`Tensors`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: Gpu,
        cfg: SupirConfig,
        tensors: &Tensors,
        h: u32,
        w: u32,
        t_enc: u32,
        taps: bool,
        control_scale: f32,
    ) -> Supir {
        Supir::build(gpu, cfg, tensors, None, h, w, t_enc, taps, control_scale, false)
    }

    /// [`Supir::new`], but every weight `tensors` doesn't carry falls back to
    /// `packed` (`supir::int8::quantize_tensors`'s output) - the ACTUAL fix
    /// for the measured OOM `crates/supir/tests/parity.rs`'s full-forward
    /// test documents: `tensors` here is the SMALL residual
    /// (`supir::int8::QuantizedTensors::full` - never-quantized names,
    /// biases, norm gains, every conv), not the whole ~15.6 GB manifest, and
    /// `vae::blocks::Builder::dev` dequantizes `packed`'s entries ONE TENSOR
    /// AT A TIME at upload rather than reconstructing a whole-model fp32 map
    /// first. The device buffers this produces are bit-identical to
    /// [`Supir::new`]'s (same dispatch, same fp32 GEMM) - only the
    /// HOST-resident bytes differ.
    #[allow(clippy::too_many_arguments)]
    pub fn new_quantized(
        gpu: Gpu,
        cfg: SupirConfig,
        tensors: &Tensors,
        packed: &PackedTensors,
        h: u32,
        w: u32,
        t_enc: u32,
        taps: bool,
        control_scale: f32,
    ) -> Supir {
        Supir::build(gpu, cfg, tensors, Some(packed), h, w, t_enc, taps, control_scale, false)
    }

    /// [`Supir::new`], recording the reverse-mode tape - what
    /// [`crate::train::SupirTrainer`] builds on. Mirrors
    /// [`sdxlunet::model::Unet::new_train`] exactly: `taps` off (the SSA
    /// forward the tape needs doubles as its own cache), no int8 packing (a
    /// training build wants every gradient buffer, so the host-memory saving
    /// `set_packed` buys is not the relevant axis here).
    pub fn new_train(gpu: Gpu, cfg: SupirConfig, tensors: &Tensors, h: u32, w: u32, t_enc: u32, control_scale: f32) -> Supir {
        Supir::build(gpu, cfg, tensors, None, h, w, t_enc, false, control_scale, true)
    }

    /// The recorded tape, on a [`Supir::new_train`] build.
    pub fn trace(&self) -> &vae::blocks::grad::Trace {
        self.trace.as_ref().expect("supir: no tape recorded - build with Supir::new_train")
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        gpu: Gpu,
        cfg: SupirConfig,
        tensors: &Tensors,
        packed: Option<&PackedTensors>,
        h: u32,
        w: u32,
        t_enc: u32,
        taps: bool,
        control_scale: f32,
        train: bool,
    ) -> Supir {
        let levels = cfg.backbone.levels();
        let scale = 1u32 << (levels - 1);
        assert!(
            h.is_multiple_of(scale) && w.is_multiple_of(scale),
            "supir: latent {h}x{w} is not a multiple of the {scale}x downscale"
        );
        assert_eq!(cfg.backbone.in_channels, 4, "supir: the hint embedder assumes a 4-channel latent");
        let c0 = cfg.backbone.block_out_channels[0];

        let sample_in = gpu.storage((cfg.backbone.in_channels * h * w) as u64);
        let hint_in = gpu.storage((4 * h * w) as u64);
        let enc_in = gpu.storage((t_enc * cfg.backbone.cross_attention_dim) as u64);
        let temb_in = gpu.storage(c0 as u64);
        let aug_in = gpu.storage(cfg.backbone.projection_class_embeddings_input_dim as u64);

        let mut r = if train {
            Rec::new_train(&gpu, &cfg.backbone, tensors, t_enc, taps)
        } else {
            Rec::new(&gpu, &cfg.backbone, tensors, t_enc, taps)
        };
        if let Some(p) = packed {
            r.set_packed(p);
        }

        r.set_prefix("control_model.");
        let hs = crate::trunk::record(&mut r, &cfg.trunk, "control_model.", h, w, &enc_in, &hint_in, &sample_in, &temb_in, &aug_in);
        // The trunk's own resnets already consumed this during the call
        // above; taking it explicitly documents that the backbone's own
        // `conditioning` call below starts a SECOND, independent chain
        // rather than silently reusing the trunk's (see `Rec::take_temb_act`'s
        // doc).
        let _trunk_temb = r.take_temb_act();
        r.set_prefix("");

        let adaptors = Adaptors::new(cfg.adaptors.clone(), hs, control_scale);
        for (name, _) in adaptors.kernels() {
            assert!(
                gpu.kernel_index(name).is_some(),
                "supir: the adaptors need the `{name}` kernel, but this Gpu was not built with it - \
                 construct it from supir::model::KERNELS"
            );
        }
        r.blocks().set_mix_ids(vae::blocks::MixIds { fwd: K_EDM_MIX, bwd: K_SCALE_ROW });
        r.set_fuse(&adaptors);

        let inputs = Inputs {
            sample_in: sample_in.clone(),
            enc_in: enc_in.clone(),
            temb_in: temb_in.clone(),
            aug_in: aug_in.clone(),
        };
        let recorded = Unet::record_into(&mut r, &cfg.backbone, h, w, &inputs, false);

        let blocks = r.into_blocks();
        let trace = train.then(|| blocks.trace());
        let (steps, taps) = blocks.finish();
        Supir { gpu, cfg, hw: (h, w), t_enc, sample_in, hint_in, enc_in, temb_in, aug_in, out: recorded.out, trace, steps, taps }
    }

    pub fn config(&self) -> &SupirConfig {
        &self.cfg
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// The graph's output buffer - what [`crate::train::SupirTrainer`]'s loss
    /// head reads.
    pub fn out(&self) -> &DeviceBuffer {
        &self.out
    }

    /// Write the graph's four device inputs, without submitting - the shared
    /// half of [`Supir::run`] and [`crate::train::SupirTrainer::set_inputs`],
    /// mirroring [`sdxlunet::model::Unet::write_inputs`]'s own split (one
    /// implementation of the host-side embeddings, not two).
    ///
    /// * `sample` - `[4 · H · W]`, the noisy latent `x_t`.
    /// * `hint` - `[4 · H · W]`, `_z` (the degradation-robust encode).
    /// * `enc`/`pooled`/`time_ids` - identical in shape and convention to
    ///   [`sdxlunet::model::Unet::run`]'s.
    pub fn write_inputs(&self, sample: &[f32], hint: &[f32], timestep: f32, enc: &[f32], pooled: &[f32], time_ids: &[f32]) {
        let c = &self.cfg.backbone;
        let (h, w) = self.hw;
        assert_eq!(sample.len(), (c.in_channels * h * w) as usize, "supir: sample size");
        assert_eq!(hint.len(), (4 * h * w) as usize, "supir: hint size");
        assert_eq!(enc.len(), (self.t_enc * c.cross_attention_dim) as usize, "supir: encoder_hidden_states size");
        assert_eq!(pooled.len(), c.pooled_dim() as usize, "supir: pooled text size");
        assert_eq!(time_ids.len(), sdxlunet::config::N_TIME_IDS as usize, "supir: time_ids count");

        let temb = model::hostmath::timestep_embedding(
            timestep,
            c.block_out_channels[0] as usize,
            c.flip_sin_to_cos,
            c.freq_shift as f64,
            10_000.0,
        );
        let aug = sdxlunet::hostemb::added_cond(pooled, time_ids, c.addition_time_embed_dim, c.flip_sin_to_cos, c.freq_shift);
        self.gpu.write_f32(&self.sample_in, sample);
        self.gpu.write_f32(&self.hint_in, hint);
        self.gpu.write_f32(&self.enc_in, enc);
        self.gpu.write_f32(&self.temb_in, &temb);
        self.gpu.write_f32(&self.aug_in, &aug);
    }

    /// One evaluation: the frozen UNet's raw output (pre EDM `c_skip`/`c_out`
    /// - `diffusion::restore`'s job, not this graph's).
    #[allow(clippy::too_many_arguments)]
    pub fn run(&self, sample: &[f32], hint: &[f32], timestep: f32, enc: &[f32], pooled: &[f32], time_ids: &[f32]) -> Vec<f32> {
        let c = &self.cfg.backbone;
        let (h, w) = self.hw;
        self.write_inputs(sample, hint, timestep, enc, pooled, time_ids);
        self.gpu.submit(&[], &self.steps);
        self.gpu.read(&self.out, (c.out_channels * h * w) as usize)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `KERNELS` extends `sdxlunet::model::KERNELS` exactly - the same
    /// contract `controlnet::model::KERNELS`'s own test pins.
    #[test]
    fn the_kernel_set_extends_the_unets_exactly() {
        assert_eq!(KERNELS.len(), sdxlunet::model::KERNELS.len() + 2);
        for (i, k) in sdxlunet::model::KERNELS.iter().enumerate() {
            assert_eq!(KERNELS[i], *k, "slot {i}");
        }
        assert_eq!(KERNELS[K_EDM_MIX].0, "edm_mix");
        assert_eq!(KERNELS[K_SCALE_ROW].0, "scale_row");
        assert!(KERNELS.iter().all(|(n, s)| !n.is_empty() && !s.is_empty()));
    }

    /// Weight-free tiny-config smoke test (porting.md §4): records trunk +
    /// adaptors + frozen UNet into one graph and asserts the output is
    /// finite and the right length. No real checkpoint needed.
    #[test]
    fn tiny_forward_is_finite() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let cfg = SupirConfig::tiny();
        let mut tensors = sdxlunet::init::init_weights(&cfg.backbone, 11);
        tensors.extend(crate::init::init_weights(&cfg, 13));

        let gpu = gpu_core::testgpu::dev(&KERNELS);
        let (h, w) = (16u32, 16u32);
        let t_enc = 9u32;
        let m = Supir::new(gpu, cfg, &tensors, h, w, t_enc, false, 0.7);

        let c = m.config().backbone.clone();
        let sample: Vec<f32> = (0..(c.in_channels * h * w) as usize).map(|i| ((i as f32) * 0.013).sin()).collect();
        let hint: Vec<f32> = (0..(4 * h * w) as usize).map(|i| ((i as f32) * 0.021).cos()).collect();
        let enc: Vec<f32> = (0..(t_enc * c.cross_attention_dim) as usize).map(|i| ((i as f32) * 0.029).cos()).collect();
        let pooled: Vec<f32> = (0..c.pooled_dim() as usize).map(|i| ((i as f32) * 0.07).sin()).collect();
        let time_ids = vec![64.0, 64.0, 0.0, 0.0, 64.0, 64.0];

        let out = m.run(&sample, &hint, 601.0, &enc, &pooled, &time_ids);
        assert_eq!(out.len(), (c.out_channels * h * w) as usize);
        assert!(out.iter().all(|v| v.is_finite()), "supir forward produced a non-finite output");
        assert!(out.iter().any(|v| v.abs() > 1e-9), "supir forward produced an all-zero output");
    }
}
