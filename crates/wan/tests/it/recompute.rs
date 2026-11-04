// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of video diffusion models for its clients. If your team
// needs expertise in trading activation memory for recomputation in a DiT trainer, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Wan's host trainer holds its block stack as a `model::saved::BlockStack`: recomputing keeps only every block's
//! input and gives the loss and gradients of keeping every block's cache, bit for bit
//! (`model::saved::check_block_recompute`).

use model::saved::{check_block_recompute, BlockStep};
use wan::modelgrad::{backward, forward, grad_views, init_model, loss, make_flow_batch, Cfg};

#[test]
fn the_wan_trainer_recomputes_its_blocks_exactly() {
    let cfg = Cfg::tiny();
    let w = init_model::<f64>(&cfg, 0x3A11_0BEE);
    let x0: Vec<f64> = (0..cfg.latent_len()).map(|i| ((i % 19) as f64 / 19.0 - 0.5) * 1.2).collect();
    let noise: Vec<f64> = (0..x0.len()).map(|i| ((i % 11) as f64 / 11.0 - 0.5) * 0.7).collect();
    let rows = cfg.text_len;
    let ctx: Vec<f64> = (0..rows * cfg.text_dim).map(|i| ((i % 5) as f64 / 5.0 - 0.5) * 1.5).collect();
    let b = make_flow_batch(&cfg, &x0, &ctx, rows, 0.4, &noise);
    check_block_recompute("wan", |policy| {
        let (pred, cache) = forward(&cfg, &w, &b.inputs(), policy)?;
        let (l, dpred) = loss(&pred, &b.target);
        let g = backward(&cfg, &w, &cache, &dpred);
        let grads = grad_views(&g).into_iter().map(|(n, v)| (n, v.iter().map(|x| x.to_bits()).collect())).collect();
        Ok(BlockStep { loss: l.to_bits(), grads, held_caches: cache.held_block_caches() })
    })
    .unwrap();
}
