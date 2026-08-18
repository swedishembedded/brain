// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `check_sam2` — the finite-difference gate on SAM 2's **mask-decoder**
//! backward (`sam2::train::MaskDecoderTrainer`).
//!
//! Scope, stated up front: this is a **decoder-only** backward with the Hiera
//! trunk and the FPN neck FROZEN — the common SAM 2 finetuning mode, and a
//! self-contained sub-graph off `Encoded`. The exact covered/not-covered split
//! is `sam2::train::TRAINABLE_PREFIXES`; every parameter this report lists is
//! one the backward actually computes, and every parameter it does not list is
//! `Role::Frozen` and carries no gradient at all (rather than a silently wrong
//! one).
//!
//! Style: the DEVICE style (`check_seq2seq`), not the host-f64 style
//! (`check_flux2`). SAM 2's decoder is composed from shipped WGSL, so the only
//! thing worth checking is those kernels; a host oracle would prove nothing
//! about them and would be a second implementation.
//!
//! `eps = 5e-4`, not the workspace-default `5e-3`: a ±1 direction over `numel`
//! entries is an L2 step of `eps*sqrt(numel)`, and the decoder's `[32,32]`
//! tensors at `5e-3` would move 0.16 in weight space — comparable to the init
//! scale, deep into the ReLU/GELU nonlinear regime. This is the same reason
//! `yolo/tests/p3_gradcheck.rs` drops to `5e-4`.
//!
//! Two frozen selections make the objective smooth enough to difference, both
//! for the reason `check_moe` pins `top_k == n_experts`:
//!   * `actual_ious` is computed by the reference from THRESHOLDED masks, so it
//!     is piecewise constant in the weights — it is a fixed target here;
//!   * the reference back-props focal+dice only through the `argmin`-selected
//!     mask channel. That selection is frozen. This check supervises **all
//!     four** channels (`mask_w = 1`), which is the `supervise_all` setting —
//!     with one channel selected the other three hypernetwork MLPs would have
//!     exactly zero gradient and their checks would pass vacuously.

use std::collections::HashMap;

use data::rng::Rng;
use sam2::train::{FrozenEncode, MaskDecoderTrainer, MaskTargets};
use sam2::{Sam2Config, Tensors};

use crate::{directional_check, CheckModel, Report};

/// Orphan-rule wrapper: `CheckModel` has a blanket impl over `model::Model`, so
/// a foreign type cannot implement it directly (the compiler cannot prove
/// `MaskDecoderTrainer: !model::Model`). Same workaround as
/// `crates/tts/tests/talker.rs`.
pub struct Sam2DecoderCheck(pub MaskDecoderTrainer);

impl CheckModel for Sam2DecoderCheck {
    fn param_names(&self) -> Vec<String> {
        self.0.param_names()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.0.read_weight(name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.0.write_weight(name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.0.read_grad(name)
    }
    fn loss(&self) -> f32 {
        self.0.loss()
    }
    fn zero_grads(&self) {
        self.0.zero_grads();
    }
    fn backward(&self) {
        self.0.backward();
    }
}

/// A TINY SAM 2: `d_model = 32`, a `4x4` image-embedding grid, 2 two-way layers,
/// 4 heads. Every structural feature the real decoder has is present (the
/// `skip_first_layer_pe` first layer, both cross-attention directions, the
/// `ConvTranspose`/`LayerNorm2d`/GELU upscaling tail, the hypernetwork dot
/// product, the sigmoid IoU head and the object-score head) — only the sizes
/// are small. The trunk half of the config is never executed here; it only has
/// to be self-consistent so the tensor manifest is well formed.
pub fn tiny_config() -> Sam2Config {
    let mut cfg = Sam2Config::hiera_tiny();
    // ---- trunk (frozen, never run — kept consistent for the manifest) ----
    cfg.image_size = 64;
    cfg.backbone_stride = 16; // -> a 4x4 image-embedding grid
    cfg.embed_dim = 8;
    cfg.num_heads = 1;
    cfg.stages = vec![1, 1, 1, 1];
    cfg.global_att_blocks = vec![];
    cfg.window_spec = vec![2, 2, 2, 2];
    cfg.window_pos_embed_bkg_spatial_size = (2, 2);
    cfg.mlp_ratio = 2;
    cfg.backbone_channel_list = vec![64, 32, 16, 8];
    // ---- decoder (the part under test) ----
    cfg.d_model = 32;
    cfg.transformer_depth = 2;
    cfg.transformer_heads = 4;
    cfg.transformer_mlp_dim = 24;
    cfg.iou_head_hidden_dim = 12;
    cfg.pos_sine_num_pos_feats = 16;
    cfg.mask_in_chans = 8;
    cfg
}

/// Deterministic weights for every tensor the manifest names.
///
/// 1-D `*.weight` tensors are the LayerNorm gains and start near 1 — a
/// zero-mean gain would collapse every normalised branch and make the whole
/// check ill-conditioned rather than wrong.
fn random_weights(cfg: &Sam2Config, seed: u64) -> Tensors {
    let mut rng = Rng::new(seed);
    let mut t: Tensors = HashMap::new();
    let mut manifest = cfg.tensor_manifest();
    manifest.push(("no_mem_embed".into(), vec![1, cfg.d_model as usize]));
    for (name, shape) in manifest {
        let n: usize = shape.iter().product();
        let gain = shape.len() == 1 && name.ends_with(".weight");
        let data: Vec<f32> = (0..n)
            .map(|_| {
                let u = rng.next_f32() * 2.0 - 1.0;
                if gain {
                    1.0 + 0.1 * u
                } else {
                    0.3 * u
                }
            })
            .collect();
        t.insert(name, (shape, data));
    }
    t
}

fn randbuf(g: &gpu_core::Gpu, label: &str, rng: &mut Rng, n: usize, scale: f32) -> gpu_core::DeviceBuffer {
    let v: Vec<f32> = (0..n).map(|_| scale * (rng.next_f32() * 2.0 - 1.0)).collect();
    g.storage_init(label, &v)
}

/// Build the tiny decoder on a fixed batch and gradient-check every trainable
/// tensor. `Gpu` comes from the pooled test device, so this shares one device
/// with the rest of the binary (`gpu_core::testgpu`).
pub fn check_sam2(seed: u64) -> Report {
    check_sam2_on(gpu_core::testgpu::dev(sam2::PIPELINES), seed)
}

/// [`check_sam2`] on a caller-supplied device (so a backend-parity harness can
/// run the same check on CPU and GPU).
pub fn check_sam2_on(gpu: gpu_core::Gpu, seed: u64) -> Report {
    let cfg = tiny_config();
    let weights = random_weights(&cfg, seed);
    let d = cfg.d_model as usize;
    let side = cfg.image_embedding_size() as usize;
    let n_img = side * side;
    let nmt = cfg.num_mask_tokens() as usize;
    let hi = 16 * n_img;
    let n_sparse = 2u32; // one point prompt + the padding row
    let mut rng = Rng::new(seed ^ 0x5A5A_1234);

    // The frozen encoder outputs. Their VALUES do not matter (they are
    // constants of the sub-graph); only that they are fixed across the sweep.
    let enc = FrozenEncode {
        image_embed: randbuf(&gpu, "sam2_ck_img", &mut rng, d * n_img, 1.0),
        high_res: [
            randbuf(&gpu, "sam2_ck_hr0", &mut rng, (d / 8) * 16 * n_img, 1.0),
            randbuf(&gpu, "sam2_ck_hr1", &mut rng, (d / 4) * 4 * n_img, 1.0),
        ],
        dense_pe: randbuf(&gpu, "sam2_ck_pe", &mut rng, d * n_img, 1.0),
        sparse: randbuf(&gpu, "sam2_ck_sparse", &mut rng, n_sparse as usize * d, 1.0),
        n_sparse,
    };

    // A fixed binary ground-truth mask, broadcast over the mask channel exactly
    // as `target_masks.expand_as(src_masks)` does. A quarter-plane blob keeps
    // both classes well represented, which is what the focal alpha/gamma terms
    // need in order to be exercised.
    let mut gt = vec![0.0f32; nmt * hi];
    let hw = 4 * side;
    for y in 0..hw {
        for x in 0..hw {
            let inside = (y * 3 + x * 2) % 7 < 3;
            for m in 0..nmt {
                gt[m * hi + y * hw + x] = if inside { 1.0 } else { 0.0 };
            }
        }
    }
    let ious: Vec<f32> = (0..nmt).map(|_| rng.next_f32()).collect();
    let mask_w = vec![1.0f32; nmt];
    let tgt = MaskTargets {
        masks: gpu.storage_init("sam2_ck_gt", &gt),
        ious: gpu.storage_init("sam2_ck_iou", &ious),
        obj: gpu.storage_init("sam2_ck_obj", &[1.0f32]),
        mask_w: gpu.storage_init("sam2_ck_mw", &mask_w),
        // `sam2.1_hiera_b+_MOSE_finetune.yaml`: loss_mask 20, loss_dice 1,
        // loss_iou 1, loss_class 1.
        w_focal: 20.0,
        w_dice: 1.0,
        w_iou: 1.0,
        w_class: 1.0,
        focal_alpha: 0.25,
        focal_gamma: 2.0,
        mask_w_host: mask_w,
    };

    let m = Sam2DecoderCheck(MaskDecoderTrainer::new(gpu, cfg, &weights, enc, tgt));
    // eps 5e-4 (see the module doc); 3 directions, best-conditioned reported.
    directional_check(&m, 5e-4, 3, seed ^ 0x1234)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gate. `MOE_SKIP_GPU_TESTS` skips it the way every other device-model
    /// gradcheck test does.
    #[test]
    fn sam2_mask_decoder_gradients_match_finite_differences() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            brain_testutil::skip_unavailable("check_sam2 (MOE_SKIP_GPU_TESTS)");
            return;
        }
        let r = check_sam2(7);
        r.print();
        println!("check_sam2: {} tensors, max_rel = {:.3e}", r.checks.len(), r.max_rel());
        let bad = r.failures(4e-3, 8e-2);
        assert!(bad.is_empty(), "{} tensors outside tolerance: {:?}", bad.len(), bad);
    }
}
