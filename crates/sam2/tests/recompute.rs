// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of promptable segmentation models for its clients. If
// your team needs expertise in training mask decoders within a fixed device budget, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Activation recompute for the SAM 2 mask-decoder trainer, held to the contract `model::saved::check_recompute`
//! states: a trainer that keeps only each two-way layer's output tokens and keys and rebuilds the layer from them
//! before its backward computes the loss and every gradient of the trainer that keeps everything, bit for bit, on
//! fewer live device bytes. `MaskDecoderTrainer` has no `model::Model` impl for that function to take, so this test
//! runs the same clauses on the trainer itself. One test in its own binary, so nothing else allocates while the
//! device bytes are compared.

use std::collections::HashMap;

use gpu_core::Gpu;
use model::saved::{SavedFormat, SavedPolicy};
use sam2::train::{FrozenEncode, MaskDecoderTrainer, MaskTargets};
use sam2::{Sam2Config, Tensors};

/// A step's loss, every trained parameter's gradient bits, and the device bytes live after the policy switch and at
/// the step's peak.
struct Measured {
    loss: f32,
    grads: Vec<(String, Vec<u32>)>,
    live: Option<u64>,
    peak: Option<u64>,
}

/// A small decoder with every structural feature of the real one: the first layer without positional add, both
/// cross-attention directions, the upscaling tail, the hypernetwork dot product and both heads.
fn config() -> Sam2Config {
    let mut cfg = Sam2Config::hiera_tiny();
    cfg.image_size = 64;
    cfg.backbone_stride = 16;
    cfg.embed_dim = 8;
    cfg.num_heads = 1;
    cfg.stages = vec![1, 1, 1, 1];
    cfg.global_att_blocks = vec![];
    cfg.window_spec = vec![2, 2, 2, 2];
    cfg.window_pos_embed_bkg_spatial_size = (2, 2);
    cfg.mlp_ratio = 2;
    cfg.backbone_channel_list = vec![64, 32, 16, 8];
    cfg.d_model = 32;
    cfg.transformer_depth = 3;
    cfg.transformer_heads = 4;
    cfg.transformer_mlp_dim = 24;
    cfg.iou_head_hidden_dim = 12;
    cfg.pos_sine_num_pos_feats = 16;
    cfg.mask_in_chans = 8;
    cfg
}

/// Deterministic pseudo-random values in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| scale * self.next()).collect()
    }
}

fn weights(cfg: &Sam2Config) -> Tensors {
    let mut rng = Lcg(7);
    let mut t: Tensors = HashMap::new();
    let mut manifest = cfg.tensor_manifest();
    manifest.push(("no_mem_embed".into(), vec![1, cfg.d_model as usize]));
    for (name, shape) in manifest {
        let n: usize = shape.iter().product();
        let gain = shape.len() == 1 && name.ends_with(".weight");
        let data = (0..n).map(|_| if gain { 1.0 + 0.1 * rng.next() } else { 0.3 * rng.next() }).collect();
        t.insert(name, (shape, data));
    }
    t
}

fn trainer() -> MaskDecoderTrainer {
    let gpu: Gpu = gpu_core::testgpu::dev(sam2::PIPELINES);
    let cfg = config();
    let weights = weights(&cfg);
    let d = cfg.d_model as usize;
    let side = cfg.image_embedding_size() as usize;
    let n_img = side * side;
    let nmt = cfg.num_mask_tokens() as usize;
    let hi = 16 * n_img;
    let n_sparse = 2u32;
    let mut rng = Lcg(99);
    let enc = FrozenEncode {
        image_embed: gpu.storage_init("rc_img", &rng.vec(d * n_img, 1.0)),
        high_res: [
            gpu.storage_init("rc_hr0", &rng.vec((d / 8) * 16 * n_img, 1.0)),
            gpu.storage_init("rc_hr1", &rng.vec((d / 4) * 4 * n_img, 1.0)),
        ],
        dense_pe: gpu.storage_init("rc_pe", &rng.vec(d * n_img, 1.0)),
        sparse: gpu.storage_init("rc_sparse", &rng.vec(n_sparse as usize * d, 1.0)),
        n_sparse,
    };
    let gt: Vec<f32> = (0..nmt * hi).map(|i| if (i % hi * 3 + i % 5) % 7 < 3 { 1.0 } else { 0.0 }).collect();
    let ious: Vec<f32> = (0..nmt).map(|_| rng.next().abs()).collect();
    let mask_w = vec![1.0f32; nmt];
    let tgt = MaskTargets {
        masks: gpu.storage_init("rc_gt", &gt),
        ious: gpu.storage_init("rc_iou", &ious),
        obj: gpu.storage_init("rc_obj", &[1.0f32]),
        mask_w: gpu.storage_init("rc_mw", &mask_w),
        w_focal: 20.0,
        w_dice: 1.0,
        w_iou: 1.0,
        w_class: 1.0,
        focal_alpha: 0.25,
        focal_gamma: 2.0,
        mask_w_host: mask_w,
    };
    MaskDecoderTrainer::new(gpu, cfg, &weights, enc, tgt)
}

/// One training step under `policy`.
fn measured(policy: SavedPolicy) -> Measured {
    let mut t = trainer();
    t.set_saved_policy(policy).unwrap();
    let gpu = &t.sam.gpu;
    let live = gpu.memory_usage().map(|u| u.live);
    gpu.reset_memory_peak();
    t.zero_grads();
    let loss = t.loss();
    t.backward();
    gpu.poll_wait();
    let peak = gpu.memory_usage().map(|u| u.peak_live);
    let mut names = t.param_names();
    names.sort();
    let grads = names.into_iter().map(|n| (n.clone(), t.read_grad(&n).into_iter().map(f32::to_bits).collect())).collect();
    Measured { loss, grads, live, peak }
}

#[test]
fn sam2_recomputes_its_two_way_layers_exactly() {
    let keep = measured(SavedPolicy::Keep);
    let recompute = measured(SavedPolicy::Recompute);
    if let (Some(k), Some(r)) = (keep.live, recompute.live) {
        assert!(r < k, "recomputing holds {r} live device bytes after the switch, keeping every layer {k}");
    }
    if let (Some(k), Some(r)) = (keep.peak, recompute.peak) {
        assert!(r < k, "the recomputing step peaks at {r} live device bytes, keeping every layer {k}");
    }
    assert_eq!(recompute.loss.to_bits(), keep.loss.to_bits(), "recomputing gives loss {}, keeping {}", recompute.loss, keep.loss);
    assert!(keep.grads.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b) != 0.0)), "no parameter received a gradient to compare");
    assert_eq!(keep.grads.len(), recompute.grads.len());
    for ((n, a), (_, b)) in keep.grads.iter().zip(&recompute.grads) {
        assert!(a == b, "{n}: gradient differs recomputing");
    }

    // Narrow saved formats do not apply, and a recomputing trainer does not go back to keeping.
    let mut t = trainer();
    assert!(t.set_saved_policy(SavedPolicy::Quantized(SavedFormat::Bf16)).unwrap_err().contains("sam2"));
    t.set_saved_policy(SavedPolicy::Recompute).unwrap();
    assert!(t.set_saved_policy(SavedPolicy::Keep).unwrap_err().contains("sam2"));
}
