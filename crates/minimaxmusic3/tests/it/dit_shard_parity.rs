// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Structural validation for `dit_shard::DitStage` - the only kind possible
//! on a machine with no discrete GPU (see `dit_shard`'s own module doc for
//! the honest account of what real multi-device execution this does NOT
//! prove). Two things:
//!
//! 1. The single-shard degenerate case (`Shard::whole`) run through
//!    `Shardable::new_shard`/`run_stage_forward` must match `dit::forward`
//!    bit-for-bit-close on identical weights/inputs - proving the block-range
//!    slicing and the embed/head branch selection are not silently wrong.
//! 2. A genuine two-stage split (`DitConfig::tiny()`'s 2 layers, split one
//!    block per stage) with the boundary handed off through
//!    `DitStage::write_in_res`/`read_out_res`, run sequentially on one
//!    device, composed and checked against the same non-sharded reference.

use data::rng::Lcg;
use gpu_core::Gpu;
use minimaxmusic3::config::DitConfig;
use minimaxmusic3::dit;
use minimaxmusic3::dit_shard::{flatten_weights, DitStage, DitStageBatch};
use minimaxmusic3::dit_train;
use model::{Shard, Shardable};

fn fixture(cfg: &DitConfig, seed: u64) -> (dit::DitWeights, Vec<f32>, Vec<f32>, f32, usize) {
    let w = dit_train::random_weights(cfg, seed);
    let length = 3usize;
    let mut r = Lcg::new(seed ^ 0x5EAD_1234);
    let latents = r.vec_scaled(cfg.in_channels as usize * length, 0.3);
    let condition = r.vec_scaled(length * cfg.condition_dim as usize, 0.3);
    let timestep = 0.4f32;
    (w, latents, condition, timestep, length)
}

#[test]
fn single_shard_matches_the_non_sharded_reference() {
    let cfg = DitConfig::tiny();
    let (w, latents, condition, timestep, length) = fixture(&cfg, 81);

    let gpu = Gpu::new_cpu(dit::PIPELINES);
    let want = dit::forward(&gpu, &cfg, &w, &latents, &condition, timestep, length);

    let init = flatten_weights(&w);
    let whole = Shard::whole(cfg.num_layers as usize);
    let stage = <DitStage as Shardable>::new_shard(cfg, 1, length as u32, &init, whole);
    stage.load_shard_batch(DitStageBatch { latents: latents.clone(), condition: condition.clone(), timestep, length, target: None });
    stage.run_stage_forward();
    let got = stage.take_stage_output();

    let (cos, max_abs) = brain_testutil::parity::compare(&got, &want);
    println!("dit_shard[single]: cosine={cos:.9} max_abs={max_abs:.6}");
    assert!(cos >= 0.999999, "single-shard forward diverged from dit::forward: cosine {cos}");
}

#[test]
fn two_stage_boundary_handoff_matches_the_non_sharded_reference() {
    let cfg = DitConfig::tiny();
    assert_eq!(cfg.num_layers, 2, "test assumes exactly 2 layers so a 1-block-per-stage split is meaningful");
    let (w, latents, condition, timestep, length) = fixture(&cfg, 91);

    let gpu = Gpu::new_cpu(dit::PIPELINES);
    let want = dit::forward(&gpu, &cfg, &w, &latents, &condition, timestep, length);

    let init = flatten_weights(&w);
    let s0 = Shard { start: 0, end: 1, embed: true, head: false, gpu_index: Shard::ANY_GPU };
    let s1 = Shard { start: 1, end: 2, embed: false, head: true, gpu_index: Shard::ANY_GPU };
    let stage0 = <DitStage as Shardable>::new_shard(cfg, 1, length as u32, &init, s0);
    let stage1 = <DitStage as Shardable>::new_shard(cfg, 1, length as u32, &init, s1);

    stage0.load_shard_batch(DitStageBatch { latents: latents.clone(), condition: condition.clone(), timestep, length, target: None });
    stage0.run_stage_forward();
    let boundary = stage0.read_out_res();

    // A non-embed stage never reads `latents`/`condition` (see
    // `dit_shard`'s module doc) - empty placeholders, not the real input.
    stage1.load_shard_batch(DitStageBatch { latents: vec![], condition: vec![], timestep, length, target: None });
    stage1.write_in_res(&boundary);
    stage1.run_stage_forward();
    let got = stage1.take_stage_output();

    let (cos, max_abs) = brain_testutil::parity::compare(&got, &want);
    println!("dit_shard[two_stage]: cosine={cos:.9} max_abs={max_abs:.6}");
    assert!(cos >= 0.999999, "two-stage boundary handoff diverged from dit::forward: cosine {cos}");
}
