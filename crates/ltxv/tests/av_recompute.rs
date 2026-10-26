// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of audio-video diffusion models for its clients. If your
// team needs expertise in trading activation memory for recomputation in a joint audio-video DiT trainer, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The audio-video DiT's host trainer holds its block stack as a `model::saved::BlockStack`: recomputing keeps only
//! every block's two stream inputs and gives the loss and gradients of keeping every block's cache, bit for bit
//! (`model::saved::check_block_recompute`).

use ltxv::av_modelgrad::{backward, forward, grad_views, init_model, loss, make_av_flow_batch, AvCfg};
use model::saved::{check_block_recompute, BlockStep};

#[test]
fn the_audio_video_trainer_recomputes_its_blocks_exactly() {
    let cfg = AvCfg::tiny();
    let w = init_model::<f64>(&cfg, 0xA7_0BEE);
    let fill = |n: usize, m: usize, scale: f64| -> Vec<f64> { (0..n).map(|i| ((i % m) as f64 / m as f64 - 0.5) * scale).collect() };
    let v_x0 = fill(cfg.tv * cfg.v_in_channels, 19, 1.2);
    let a_x0 = fill(cfg.ta * cfg.a_in_channels, 13, 1.1);
    let v_noise = fill(v_x0.len(), 11, 0.7);
    let a_noise = fill(a_x0.len(), 7, 0.8);
    let v_ctx = fill(cfg.v_context_len * cfg.vdim, 5, 1.5);
    let a_ctx = fill(cfg.a_context_len * cfg.adim, 9, 1.4);
    let b = make_av_flow_batch(&cfg, &v_x0, &a_x0, &v_ctx, &a_ctx, 0.42, 0.71, &v_noise, &a_noise);
    check_block_recompute("ltxv audio-video", |policy| {
        let (v_pred, a_pred, cache) = forward(&cfg, &w, &b.inputs(), policy)?;
        let (l, dv_pred, da_pred) = loss(&v_pred, &b.v_target, &a_pred, &b.a_target);
        let g = backward(&cfg, &w, &cache, &dv_pred, &da_pred);
        let grads = grad_views(&g).into_iter().map(|(n, v)| (n, v.iter().map(|x| x.to_bits()).collect())).collect();
        Ok(BlockStep { loss: l.to_bits(), grads, held_caches: cache.held_block_caches() })
    })
    .unwrap();
}
