// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of image diffusion models for its clients. If your team
// needs expertise in trading activation memory for recomputation in a DiT trainer, you can procure our services by
// sending an email to info@swedishembedded.com.

//! FLUX.2's host trainer holds its double- and single-stream stacks as `model::saved::BlockStack`s: recomputing keeps
//! only every block's input and gives the loss and gradients of keeping every block's cache, bit for bit
//! (`model::saved::check_block_recompute`), with a reference image conditioning the sequence.

use flux2::modelgrad::{backward, forward, grad_views, init_model, loss_weighted, make_flow_batch_paired, Cfg};
use model::saved::{check_block_recompute, BlockStep};

#[test]
fn the_flux2_trainer_recomputes_its_blocks_exactly() {
    let cfg = Cfg::tiny_paired();
    let w = init_model::<f64>(&cfg, 0xF1_0BEE);
    let fill = |n: usize, m: usize, scale: f64| -> Vec<f64> { (0..n).map(|i| ((i % m) as f64 / m as f64 - 0.5) * scale).collect() };
    let x0 = fill(cfg.n_gen() * cfg.in_channels, 19, 1.2);
    let refs = fill(cfg.n_ref() * cfg.in_channels, 7, 0.9);
    let noise = fill(x0.len(), 11, 0.7);
    let ctx = fill(cfg.txt_len * cfg.context_in_dim, 5, 1.5);
    let b = make_flow_batch_paired(&cfg, &x0, &refs, &ctx, 0.4, &noise);
    check_block_recompute("flux2", |policy| {
        let (pred, cache) = forward(&cfg, &w, &b.inputs(), policy)?;
        let (l, dpred) = loss_weighted(&pred, &b.target, &b.w);
        let g = backward(&cfg, &w, &cache, &dpred);
        let grads = grad_views(&g).into_iter().map(|(n, v)| (n, v.iter().map(|x| x.to_bits()).collect())).collect();
        Ok(BlockStep { loss: l.to_bits(), grads, held_caches: cache.held_block_caches() })
    })
    .unwrap();
}
