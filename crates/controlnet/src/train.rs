// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ControlNet's **training** graph: the trainable copy + zero-convs, the
//! frozen SDXL backbone, and the residual injection between them, all
//! recorded into ONE reverse-mode tape, plus an MSE loss head on the
//! backbone's final output.
//!
//! Swedish Embedded AB implements training-mode backward passes for ported
//! diffusion models, for teams who need their control-adapter's gradient
//! flow proven correct end to end rather than merely "the forward runs". If
//! your team needs expertise in wiring a ControlNet-shaped adapter into a
//! frozen backbone's own reverse-mode tape, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! # Composition, and the one genuinely new seam
//!
//! Every op both halves use - conv, GroupNorm, SiLU, add, the transformer
//! stages - already goes through [`vae::blocks::Builder`], which is
//! tape-recording (see `crate::model::ControlNet`'s own forward, which is
//! built from the SAME [`sdxlunet::model::Rec`] convenience methods this
//! trainer uses). The one addition this file needs is the residual ADD at
//! each injection point landing on the tape between the two halves - a plain
//! pass-through backward (`Op::Add2`'s adjoint copies `dy` to both operands
//! unconditionally, already proven by every other user of
//! [`vae::blocks::Builder::add`]) but genuinely new WIRING: nothing in this
//! tree previously recorded a ControlNet's own zero-conv output and a
//! backbone's own skip connection onto the same tape and added them there.
//! `sdxlunet::model::Unet::record_into`'s own `control: bool` branch does the
//! identical add, but against buffers IT allocates and the caller writes
//! from the host (`Unet::run_with_control`) - a deliberate design for
//! INFERENCE from an already-computed ControlNet, and exactly why that path
//! is not a training-mode injection: a host round trip severs the tape
//! between the two models. This file's up-path loop (after the injection) is
//! therefore a second copy of `record_into`'s tail, not a call to it - there
//! is no way to hand `record_into` externally-produced, still-on-the-tape
//! buffers instead of its own internally-allocated ones, and duplicating the
//! ~40 lines of plain up-path plumbing is cheaper than reshaping a
//! `pub fn` five other builds depend on.
//!
//! # The tensor-name collision, and how it is avoided
//!
//! [`crate::config::ControlNetConfig::tensor_manifest`] is a **filter** of
//! the backbone's own manifest (`is_controlnet_half`), not a re-derivation
//! with a prefix - so `crate::init::init_weights` returns names like
//! `"time_embedding.linear_2.weight"` that are BYTE-IDENTICAL to the frozen
//! backbone's own tensor names. Naively merging `sdxlunet::init::init_weights`
//! (the frozen copy) and `crate::init::init_weights` (the trainable copy)
//! into one [`vae::blocks::Tensors`] map the way `crates/supir` does would
//! collide on every shared name and silently train the wrong (or only one)
//! copy - `crates/supir` avoids this only because ITS OWN trunk tensors
//! already carry a `"control_model."`/`"project_modules."` prefix at
//! `supir::init::init_weights` time, before they ever reach a merge.
//! ControlNet's manifest has no such prefix, so [`tensors_for`] adds one
//! (`"controlnet."`) at merge time, and every subsequent lookup of the
//! trainable copy's own tensors goes through [`sdxlunet::model::Rec::set_prefix`]
//! (for `conditioning`/`down_path`/`mid_block`, which prepend it internally)
//! or a literal `"controlnet."`-prefixed name (for the raw
//! [`vae::blocks::Builder`] calls this file makes directly for the
//! conditioning-image embedder and the zero-convs - the same calls
//! `crate::model::ControlNet::new` makes unprefixed for its own,
//! non-colliding, single-copy graph).
//!
//! # Injection order cannot silently swap
//!
//! `crate::adapter`'s own doc warns that a same-shaped permutation of
//! injection points (SDXL has four 320-channel points) type-checks and runs
//! wrong. That risk lives in code that re-derives injection order from a
//! separate `points`/name list and zips it against a residual list built a
//! different way - this file has no such second derivation. The trainable
//! copy's residual list (`cn_outs`) and the frozen backbone's own skip list
//! (`bskips`) are BOTH produced by the literal same
//! [`sdxlunet::model::Rec::down_path`] call (once under `"controlnet."`,
//! once unprefixed) against the same [`crate::config::ControlNetConfig::backbone`]
//! config, so index `k` means "the `k`-th skip `down_path`'s own internal walk
//! pushed" identically on both sides by construction - there is no name
//! lookup or reordering step between them for a swap to hide in.
//!
//! # The dropped `scale_chan` adjoint direction
//!
//! `conditioning_scale` (the `scale` operand of every
//! [`vae::blocks::Builder::scale_chan`] call below) is a per-request
//! `ParamSpec` (`crate::caps`), not a manifest tensor - `crate::init::
//! init_weights` never emits it and `crate::config::ControlNetConfig::
//! tensor_manifest` never lists it. `vae::blocks::Op::ScaleChan`'s adjoint
//! therefore deliberately computes `dL/dx` only; there is no `dL/d(scale)`
//! kernel because nothing would ever read it.

use gpu_core::{DeviceBuffer, Gpu, Step};
use sdxlunet::config::BlockKind;
use sdxlunet::model::Rec;
use vae::blocks::grad::{BwdIds, Grads, Reverse, Trace};
use vae::blocks::{ScaleChanIds, Tensors};

use crate::config::ControlNetConfig;
use crate::model::K_SCALE;

/// Where [`vae::blocks::BWD_KERNELS`] sits in [`TRAIN_KERNELS`] - right after
/// [`crate::model::KERNELS`] (the inference set, which already carries
/// `scale_chan` at [`crate::model::K_SCALE`]).
const BWD_BASE: usize = crate::model::KERNELS.len();
const TAIL: usize = BWD_BASE + vae::blocks::BWD_KERNELS.len();

const K_MSE_VALUE: usize = TAIL;
const K_MSE_GRAD: usize = TAIL + 1;

/// This model's TRAINING kernel set: [`crate::model::KERNELS`], then the
/// shared block backward set, then the loss pair - the same three-part shape
/// `supir::train::TRAIN_KERNELS`/`sdxlunet::train::TRAIN_KERNELS` use.
pub const TRAIN_KERNELS: [(&str, &str); TAIL + 2] = train_kernel_set();

pub const TRAIN_PIPELINES: &[(&str, &str)] = &TRAIN_KERNELS;

const fn train_kernel_set() -> [(&'static str, &'static str); TAIL + 2] {
    let mut k = [("", ""); TAIL + 2];
    let mut i = 0;
    while i < crate::model::KERNELS.len() {
        k[i] = crate::model::KERNELS[i];
        i += 1;
    }
    let mut j = 0;
    while j < vae::blocks::BWD_KERNELS.len() {
        k[BWD_BASE + j] = vae::blocks::BWD_KERNELS[j];
        j += 1;
    }
    k[K_MSE_VALUE] = ("mse_value", kernels::MSE_VALUE);
    k[K_MSE_GRAD] = ("mse_grad", kernels::MSE_GRAD);
    k
}

/// Merge the frozen backbone's weights with ControlNet's own, PREFIXING the
/// latter with `"controlnet."` - see this module's doc for why the prefix is
/// load-bearing, not cosmetic.
pub fn tensors_for(cfg: &ControlNetConfig, seed: u64) -> Tensors {
    let mut tensors = sdxlunet::init::init_weights(&cfg.backbone, seed);
    for (name, v) in crate::init::init_weights(cfg, seed ^ 0x434E_4554) {
        tensors.insert(format!("controlnet.{name}"), v);
    }
    tensors
}

/// A trainable ControlNet + frozen-backbone graph at one latent resolution
/// and one text-token count.
pub struct ControlNetTrainer {
    gpu: Gpu,
    cfg: ControlNetConfig,
    trace: Trace,
    grads: Grads,
    sample_in: DeviceBuffer,
    cond_in: DeviceBuffer,
    enc_in: DeviceBuffer,
    temb_in: DeviceBuffer,
    aug_in: DeviceBuffer,
    scale_in: DeviceBuffer,
    target: DeviceBuffer,
    loss: DeviceBuffer,
    d_out: DeviceBuffer,
    out: DeviceBuffer,
    fwd: Vec<Step>,
    rev: Vec<Step>,
    rev_clears: Vec<DeviceBuffer>,
    n_out: u32,
}

impl ControlNetTrainer {
    /// Record the forward + reverse. `gpu` must carry [`TRAIN_KERNELS`].
    #[allow(clippy::too_many_lines)]
    pub fn new(gpu: Gpu, cfg: ControlNetConfig, tensors: &Tensors, h: u32, w: u32, t_enc: u32) -> ControlNetTrainer {
        cfg.validate().expect("controlnet train: invalid config");
        let gpu = gpu.share_or_new(TRAIN_PIPELINES);
        let bb = cfg.backbone.clone();
        let levels = bb.levels();
        let scale = 1u32 << (levels - 1);
        assert!(
            h.is_multiple_of(scale) && w.is_multiple_of(scale),
            "controlnet train: latent {h}x{w} is not a multiple of the {scale}x downscale"
        );
        let c0 = bb.block_out_channels[0];
        let ds = cfg.cond_downscale();
        let (ph, pw) = (h * ds, w * ds);
        let n_out = bb.out_channels * h * w;

        let sample_in = gpu.storage((bb.in_channels * h * w) as u64);
        let cond_in = gpu.storage((cfg.conditioning_channels * ph * pw) as u64);
        let enc_in = gpu.storage((t_enc * bb.cross_attention_dim) as u64);
        let temb_in = gpu.storage(c0 as u64);
        let aug_in = gpu.storage(bb.projection_class_embeddings_input_dim as u64);
        let scale_in = gpu.storage(1);
        let target = gpu.storage(n_out as u64);
        let loss = gpu.storage(n_out as u64);
        let d_out = gpu.storage(n_out as u64);

        let mut r = Rec::new_train(&gpu, &bb, tensors, t_enc, false);
        r.blocks().set_scale_chan_ids(ScaleChanIds { fwd: K_SCALE });

        // ==== 1. ControlNet's own trainable copy ============================
        r.set_prefix("controlnet.");
        r.conditioning(&bb, &temb_in, &aug_in);

        let p = |s: &str| format!("controlnet.{s}");
        let ce = &cfg.conditioning_embedding_out_channels;
        let (mut eh, mut ew) = (ph, pw);
        let mut e = r.blocks().conv(&p("controlnet_cond_embedding.conv_in"), cfg.conditioning_channels, ce[0], 3, 1, eh, ew, &cond_in);
        let mut ea = r.blocks().silu(ce[0] * eh * ew, &e);
        for i in 0..ce.len() - 1 {
            let (cin, cout) = (ce[i], ce[i + 1]);
            e = r.blocks().conv(&p(&format!("controlnet_cond_embedding.blocks.{}", 2 * i)), cin, cin, 3, 1, eh, ew, &ea);
            ea = r.blocks().silu(cin * eh * ew, &e);
            e = r.blocks().conv_s(
                &p(&format!("controlnet_cond_embedding.blocks.{}", 2 * i + 1)),
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
            eh /= 2;
            ew /= 2;
            ea = r.blocks().silu(cout * eh * ew, &e);
        }
        assert_eq!((eh, ew), (h, w), "controlnet train: the embedder lands at {eh}x{ew}, latent is {h}x{w}");
        let clast = *ce.last().expect("validated >= 2 stages");
        let cond = r.blocks().conv(&p("controlnet_cond_embedding.conv_out"), clast, c0, 3, 1, h, w, &ea);

        let cn_cin = r.blocks().conv(&p("conv_in"), bb.in_channels, c0, 3, 1, h, w, &sample_in);
        let x = r.blocks().add(c0 * h * w, &cn_cin, &cond);

        let (cn_hh, cn_skips, cn_ch, cn_cw) = r.down_path(&bb, h, w, &enc_in, &x);
        let cn_mid = r.mid_block(&bb, cn_ch, cn_cw, &enc_in, &cn_hh);
        let cmid = *bb.block_out_channels.last().expect("levels >= 1");

        // ---- zero-convs + conditioning_scale, in `down_path`'s own push
        // order then mid - see this module's doc for why that guarantees
        // alignment with the backbone's own skip list below.
        let mut cn_outs: Vec<DeviceBuffer> = Vec::with_capacity(cn_skips.len() + 1);
        for (k, (buf, c, sh, sw)) in cn_skips.into_iter().enumerate() {
            let z = r.blocks().conv(&p(&format!("controlnet_down_blocks.{k}")), c, c, 1, 0, sh, sw, &buf);
            cn_outs.push(r.blocks().scale_chan(c * sh * sw, 1, 1, &z, &scale_in));
        }
        let zm = r.blocks().conv(&p("controlnet_mid_block"), cmid, cmid, 1, 0, cn_ch, cn_cw, &cn_mid);
        cn_outs.push(r.blocks().scale_chan(cmid * cn_ch * cn_cw, 1, 1, &zm, &scale_in));

        // The trainable copy's own conditioning chain is fully consumed
        // (every resnet above already read it) - take it out before
        // recording the backbone's own, so the second `Rec::conditioning`
        // call does not need it back.
        let _ = r.take_temb_act();

        // ==== 2. the frozen backbone, with the residual injected onto the
        // SAME tape ============================================================
        r.set_prefix("");
        r.conditioning(&bb, &temb_in, &aug_in);
        let bb_cin = r.blocks().conv("conv_in", bb.in_channels, c0, 3, 1, h, w, &sample_in);
        let (mut bhh, mut bskips, mut bch, mut bcw) = r.down_path(&bb, h, w, &enc_in, &bb_cin);
        let mut prev = *bb.block_out_channels.last().expect("levels >= 1");
        bhh = r.mid_block(&bb, bch, bcw, &enc_in, &bhh);
        let (bhh_f, prev_f) = r.fuse_mid(&bhh, prev, bch, bcw);
        bhh = bhh_f;
        prev = prev_f;

        assert_eq!(bskips.len() + 1, cn_outs.len(), "controlnet train: {} residuals, {} skip slots", cn_outs.len(), bskips.len() + 1);
        for (k, (buf, c, sh, sw)) in bskips.iter_mut().enumerate() {
            let n = *c * *sh * *sw;
            *buf = r.blocks().add(n, buf, &cn_outs[k]);
        }
        let n = prev * bch * bcw;
        bhh = r.blocks().add(n, &bhh, &cn_outs[bskips.len()]);

        // ---- up path - `sdxlunet::model::Unet::record_into`'s own tail,
        // restated against `bskips`/`bhh` (see this module's doc for why it
        // is not a call to `record_into` itself).
        for i in 0..levels {
            let level = levels - 1 - i;
            let cout = bb.block_out_channels[level];
            for j in 0..=bb.layers_per_block {
                let (skip, cskip, sh, sw) = bskips.pop().expect("the skip stack is sized by UNetConfig::skip_stack");
                assert_eq!((sh, sw), (bch, bcw), "controlnet train up{i}.resnet{j}: skip is {sh}x{sw}, hidden is {bch}x{bcw}");
                let (cat, cin) = r.join_skip(prev, cskip, bch, bcw, &bhh, &skip);
                let next = r.resnet(&format!("up_blocks.{i}.resnets.{j}"), &format!("up{i}.resnet{j}"), cin, cout, bch, bcw, bb.time_embed_dim, &cat);
                bhh = next;
                if bb.up_block_types[i] == BlockKind::CrossAttn {
                    bhh = r.transformer(&format!("up_blocks.{i}.attentions.{j}"), &format!("up{i}.attn{j}"), &bb, level, bch, bcw, &enc_in, &bhh);
                }
                prev = cout;
            }
            if i + 1 < levels {
                let (hh_pre, cout) = r.pre_upsample(i, &bhh, cout, bch, bcw);
                bhh = hh_pre;
                let up = r.blocks().upsample(cout, bch, bcw, &bhh);
                bch *= 2;
                bcw *= 2;
                bhh = r.blocks().conv(&format!("up_blocks.{i}.upsamplers.0.conv"), cout, cout, 3, 1, bch, bcw, &up);
            }
        }
        assert!(bskips.is_empty(), "controlnet train: {} skip tensors left unconsumed", bskips.len());

        // ---- head ------------------------------------------------------------
        let no = r.blocks().gn("conv_norm_out", c0, bch, bcw, &bhh);
        let sa = r.blocks().silu(c0 * bch * bcw, &no);
        let out = r.blocks().conv("conv_out", c0, bb.out_channels, 3, 1, bch, bcw, &sa);

        let trace = r.blocks().trace();
        let grads = trace.alloc_grads(&gpu);
        let (mut fwd, _taps) = r.into_blocks().finish();
        fwd.push(gpu.step(K_MSE_VALUE, &[&out, &target, &loss], &[n_out], n_out));

        let mut rev = vec![gpu.step(K_MSE_GRAD, &[&out, &target, &d_out], &[n_out], n_out)];
        let reverse: Reverse = trace.backward(&gpu, BwdIds::at(BWD_BASE), &grads, &out, &d_out);
        rev.extend(reverse.steps.clone());

        ControlNetTrainer {
            gpu,
            cfg,
            trace,
            grads,
            sample_in,
            cond_in,
            enc_in,
            temb_in,
            aug_in,
            scale_in,
            target,
            loss,
            d_out,
            out,
            fwd,
            rev,
            rev_clears: reverse.clears.clone(),
            n_out,
        }
    }

    pub fn config(&self) -> &ControlNetConfig {
        &self.cfg
    }

    /// Every trainable tensor this graph reads - the frozen backbone
    /// (unprefixed) AND ControlNet's own copy (`"controlnet."`-prefixed) -
    /// in first-use order. `crate::gradcheck` (via `gradcheck::controlnet`)
    /// filters this by prefix; nothing here itself freezes anything, matching
    /// `UnetTrainer::params`'s own "everything recorded is trainable"
    /// contract.
    pub fn params(&self) -> &[(String, u64)] {
        self.trace.params()
    }

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        let len = self.len_of(name);
        self.gpu.read(self.trace.weight(name), len)
    }

    pub fn write_weight(&self, name: &str, data: &[f32]) {
        self.gpu.write_f32(self.trace.weight(name), data);
    }

    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        let len = self.len_of(name);
        self.gpu.read(self.grads.g(name), len)
    }

    fn len_of(&self, name: &str) -> usize {
        self.params()
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, l)| *l as usize)
            .unwrap_or_else(|| panic!("controlnet train: no parameter {name}"))
    }

    /// Write the graph's inputs and the MSE target.
    ///
    /// `sample`/`cond`/`enc`/`pooled`/`time_ids` are laid out exactly like
    /// [`crate::model::ControlNet::run`]'s own arguments - the SAME noisy
    /// latent, conditioning image and text/added conditioning both halves of
    /// the graph consume, `scale` is `conditioning_scale`, and `target` is
    /// `[out_channels, h, w]` against the BACKBONE's final output
    /// (`conv_out`), matching `gradcheck::unet`/`gradcheck::supir`'s own
    /// "MSE against a random, non-model-derived target" anti-vacuous-check
    /// reasoning.
    #[allow(clippy::too_many_arguments)]
    pub fn set_inputs(
        &self,
        sample: &[f32],
        timestep: f32,
        enc: &[f32],
        pooled: &[f32],
        time_ids: &[f32],
        cond: &[f32],
        scale: f32,
        target: &[f32],
    ) {
        assert_eq!(target.len(), self.n_out as usize, "controlnet train: target must be [out_channels, h, w]");
        let c = &self.cfg.backbone;
        let temb = model::hostmath::timestep_embedding(timestep, c.block_out_channels[0] as usize, c.flip_sin_to_cos, c.freq_shift as f64, 10_000.0);
        let aug = sdxlunet::hostemb::added_cond(pooled, time_ids, c.addition_time_embed_dim, c.flip_sin_to_cos, c.freq_shift);
        self.gpu.write_f32(&self.sample_in, sample);
        self.gpu.write_f32(&self.cond_in, cond);
        self.gpu.write_f32(&self.enc_in, enc);
        self.gpu.write_f32(&self.temb_in, &temb);
        self.gpu.write_f32(&self.aug_in, &aug);
        self.gpu.write_f32(&self.scale_in, &[scale]);
        self.gpu.write_f32(&self.target, target);
    }

    pub fn forward(&self) -> f32 {
        self.gpu.submit(&[], &self.fwd);
        self.gpu.read(&self.loss, self.n_out as usize).iter().sum()
    }

    pub fn zero_grads(&self) {
        let zeros: Vec<DeviceBuffer> = self.grads.all().into_iter().cloned().collect();
        self.gpu.submit(&zeros.iter().collect::<Vec<_>>(), &[]);
    }

    pub fn backward(&self) {
        self.gpu.submit(&self.rev_clears.iter().collect::<Vec<_>>(), &self.rev);
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// `dL/d(out)` after a [`Self::backward`], mirroring
    /// `sdxlunet::train::UnetTrainer::d_out`/`supir::train::SupirTrainer::d_out`.
    pub fn d_out(&self) -> &DeviceBuffer {
        &self.d_out
    }

    /// The frozen backbone's final output (`conv_out`) - what [`Self::
    /// set_inputs`]'s `target` is an MSE loss against, exposed for a test
    /// that wants the graph's own prediction rather than only the loss
    /// scalar [`Self::forward`] returns.
    pub fn out(&self) -> &DeviceBuffer {
        &self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_training_kernel_set_has_no_duplicate_names() {
        let mut seen = std::collections::HashSet::new();
        for (name, _) in TRAIN_KERNELS {
            assert!(!name.is_empty(), "TRAIN_KERNELS has an unfilled slot");
            assert!(seen.insert(name), "TRAIN_KERNELS registers '{name}' twice");
        }
    }

    #[test]
    fn the_inference_set_is_a_prefix_of_the_training_set() {
        for (i, (name, _)) in crate::model::KERNELS.iter().enumerate() {
            assert_eq!(TRAIN_KERNELS[i].0, *name, "slot {i} differs between the inference and training sets");
        }
    }

    /// The tensor-name-collision guard the module doc describes: a merge that
    /// forgot the `"controlnet."` prefix would silently drop half the
    /// tensors (a `HashMap` overwrite, not an error).
    #[test]
    fn tensors_for_has_no_lost_entries_from_the_merge() {
        let cfg = ControlNetConfig::tiny();
        let backbone_n = cfg.backbone.tensor_manifest().len();
        let controlnet_n = cfg.tensor_manifest().len();
        let merged = tensors_for(&cfg, 3);
        assert_eq!(merged.len(), backbone_n + controlnet_n, "the merge lost entries to a name collision");
        for (name, _) in cfg.tensor_manifest() {
            assert!(merged.contains_key(&format!("controlnet.{name}")), "missing controlnet.{name}");
        }
    }
}
