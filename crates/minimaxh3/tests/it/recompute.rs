// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of joint audio-video diffusion transformers for its clients.
// If your team needs expertise in trading activation memory for recomputation in a packed-sequence DiT trainer, you
// can procure our services by sending an email to info@swedishembedded.com.

//! MiniMax-H3's host trainer holds its block stack as a `model::saved::BlockStack`: recomputing keeps only every
//! block's input and gives the loss and gradients of keeping every block's cache, bit for bit
//! (`model::saved::check_block_recompute`).

use minimaxh3::grad::{backward, forward, grad_views, init_model, loss, make_flow_batch, Cfg};
use model::saved::{check_block_recompute, BlockStep};

#[test]
fn the_minimax_h3_trainer_recomputes_its_blocks_exactly() {
    let cfg = Cfg::tiny();
    let w = init_model::<f64>(&cfg, 0x4_3B1E);
    let fill = |n: usize, m: usize, scale: f64| -> Vec<f64> { (0..n).map(|i| ((i % m) as f64 / m as f64 - 0.5) * scale).collect() };
    let video_x0 = fill(cfg.num_video * cfg.video_patch_dim(), 19, 1.2);
    let audio_x0 = fill(cfg.num_audio * cfg.audio_in_channels(), 13, 1.1);
    let text_ctx = fill(cfg.num_text * cfg.text_dim(), 5, 1.5);
    let video_noise = fill(video_x0.len(), 11, 0.7);
    let audio_noise = fill(audio_x0.len(), 7, 0.8);
    let b = make_flow_batch(&cfg, &video_x0, &audio_x0, &text_ctx, &video_noise, &audio_noise);
    check_block_recompute("minimaxh3", |policy| {
        let (video_pred, audio_pred, cache) = forward(&cfg, &w, &b, policy)?;
        let (l, dv, da) = loss(&video_pred, &b.video_target, &audio_pred, &b.audio_target);
        let g = backward(&cfg, &w, &cache, &dv, &da);
        let grads = grad_views(&g).into_iter().map(|(n, v)| (n, v.iter().map(|x| x.to_bits()).collect())).collect();
        Ok(BlockStep { loss: l.to_bits(), grads, held_caches: cache.held_block_caches() })
    })
    .unwrap();
}
