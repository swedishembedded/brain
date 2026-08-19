// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pipeline-parallel sharding for the AUDIO+VIDEO DiT (`crate::shard`'s
//! `Shardable` impl for [`LtxAvDit`]) - the AV extension of `shard_parity.rs`'s
//! two [`ltxv::LtxDit`] tests, run at [`LtxAvDitConfig::tiny_gated`] (gated
//! attention AND both embeddings connectors ON, per this task's own
//! briefing) rather than the video-only suite's connector-off `tiny()`, so
//! the sharded path's connector replication (`crate::dit::av_shard_owns_
//! weight`'s doc) is actually exercised by this correctness proof, not
//! sidestepped by it.
//!
//! Same method as `shard_parity.rs`: build ONE set of random tiny-config AV
//! weights + inputs, run the ordinary non-sharded [`LtxAvDit::forward`] once
//! as the reference, then run the SAME computation through the `Shardable`
//! seam and compare. Bit-exact bar (`>= 0.999_999_9` cosine, `< 1e-4`
//! max_abs) - deterministic host<->GPU round-tripping, no stochastic kernel
//! involved, same as the video-only suite.
//!
//! Single-process only, same explicit gap `shard_parity.rs`'s own doc
//! records: this proves the TWO-RESIDUAL boundary handoff is correct, not
//! that two real cards agree (a separate, real-hardware check).

use std::collections::HashMap;

use ltxv::dit::{random_av_tiny_weights, AvDitBatch};
use ltxv::{LtxAvDit, LtxAvDitConfig};
use model::{Model, Shard, Shardable};

fn flatten(w: &vae::blocks::Tensors) -> HashMap<String, Vec<f32>> {
    w.iter().map(|(k, (_, v))| (k.clone(), v.clone())).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    let den = na.sqrt() * nb.sqrt();
    if den <= 0.0 {
        1.0
    } else {
        d / den
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// A small, valid AV input batch: video's `(1,2,3)` latent grid (tv=6
/// tokens, matching `shard_parity.rs`'s own video-only shape), audio at
/// ta=4 tokens on its own single time axis, distinct context lengths per
/// stream (lesson #4: catches an accidental video/audio context mixup).
struct TestInputs {
    v_latent: Vec<f32>,
    v_timesteps: Vec<f32>,
    v_positions: Vec<f32>,
    v_keyframes_mask: Vec<f32>,
    v_context: Vec<f32>,
    v_context_len: usize,
    tv: usize,
    v_sigma: f32,
    a_latent: Vec<f32>,
    a_timesteps: Vec<f32>,
    a_positions: Vec<f32>,
    a_context: Vec<f32>,
    a_context_len: usize,
    ta: usize,
    a_sigma: f32,
}

fn build_inputs(cfg: &LtxAvDitConfig, seed: u64) -> TestInputs {
    let mut rng = data::rng::Rng::new(seed);
    let (f, h, w) = (1, 2, 3);
    let tv = f * h * w;
    let ta = 4usize;
    // Both connectors share the video config's own `connector_num_learnable_
    // registers` (`crate::dit::push_connector`'s doc), and `EmbeddingsConnector::
    // forward` requires `seq_len % num_registers == 0` - both context lengths
    // below are chosen as distinct multiples of `tiny_gated`'s 3 registers
    // (lesson #4: still distinct per stream, to catch a video/audio mixup).
    let v_context_len = 6usize;
    let a_context_len = 3usize;

    let vdim = cfg.video.in_channels as usize;
    let adim = cfg.audio.in_channels as usize;
    let v_latent: Vec<f32> = (0..tv * vdim).map(|_| (rng.next_gaussian() * 0.5) as f32).collect();
    let a_latent: Vec<f32> = (0..ta * adim).map(|_| (rng.next_gaussian() * 0.5) as f32).collect();
    let v_context: Vec<f32> = (0..v_context_len * cfg.video.cross_attention_dim as usize).map(|_| (rng.next_gaussian() * 0.5) as f32).collect();
    let a_context: Vec<f32> = (0..a_context_len * cfg.audio.cross_attention_dim as usize).map(|_| (rng.next_gaussian() * 0.5) as f32).collect();
    let v_timesteps: Vec<f32> = (0..tv).map(|_| rng.next_f32()).collect();
    let a_timesteps: Vec<f32> = (0..ta).map(|_| rng.next_f32()).collect();
    let v_positions = ltxv::pipeline::grid_positions(f, h, w);
    let a_positions: Vec<f32> = (0..ta).flat_map(|i| [i as f32, i as f32 + 1.0]).collect(); // [1, ta, 2]
    let mut v_keyframes_mask = vec![0f32; tv];
    v_keyframes_mask[0] = 1.0;
    v_keyframes_mask[tv - 1] = 1.0;

    TestInputs {
        v_latent,
        v_timesteps,
        v_positions,
        v_keyframes_mask,
        v_context,
        v_context_len,
        tv,
        v_sigma: 0.6,
        a_latent,
        a_timesteps,
        a_positions,
        a_context,
        a_context_len,
        ta,
        a_sigma: 0.4,
    }
}

fn batch_for(inp: &TestInputs, v_target: Option<Vec<f32>>, a_target: Option<Vec<f32>>) -> AvDitBatch {
    AvDitBatch {
        v_latent: inp.v_latent.clone(),
        v_timesteps: inp.v_timesteps.clone(),
        v_positions: inp.v_positions.clone(),
        v_keyframes_mask: inp.v_keyframes_mask.clone(),
        v_context: inp.v_context.clone(),
        v_context_len: inp.v_context_len,
        tv: inp.tv,
        v_sigma: inp.v_sigma,
        v_context_valid: vec![1.0; inp.v_context_len],
        a_latent: inp.a_latent.clone(),
        a_timesteps: inp.a_timesteps.clone(),
        a_positions: inp.a_positions.clone(),
        a_context: inp.a_context.clone(),
        a_context_len: inp.a_context_len,
        ta: inp.ta,
        a_sigma: inp.a_sigma,
        a_context_valid: vec![1.0; inp.a_context_len],
        v_target,
        a_target,
    }
}

/// The single-shard degenerate case, [`LtxDit`]'s own `single_shard_matches_
/// the_non_sharded_reference` test's AV counterpart.
#[test]
fn single_shard_matches_the_non_sharded_reference_av() {
    let cfg = LtxAvDitConfig::tiny_gated();
    let weights = random_av_tiny_weights(&cfg, 7);
    let flat = flatten(&weights);
    let inp = build_inputs(&cfg, 11);

    let reference = LtxAvDit::new(cfg, weights, None);
    #[rustfmt::skip]
    let taps = reference.forward(
        &inp.v_latent, &inp.v_timesteps, &inp.v_positions, &inp.v_keyframes_mask, &inp.v_context, inp.v_context_len, inp.tv, inp.v_sigma, &vec![1.0; inp.v_context_len],
        &inp.a_latent, &inp.a_timesteps, &inp.a_positions, &inp.a_context, inp.a_context_len, inp.ta, inp.a_sigma, &vec![1.0; inp.a_context_len],
    );

    let whole = Shard::whole(cfg.video.num_layers as usize);
    let sharded = <LtxAvDit as Shardable>::new_shard(cfg, 1, inp.tv as u32, &flat, whole.clone());
    assert_eq!(sharded.shard(), &whole);
    sharded.load_shard_batch(batch_for(&inp, None, None));
    let loss = <LtxAvDit as Shardable>::run_forward_stage(&sharded);
    assert!(loss.is_none(), "no target was set - run_forward_stage must report no loss, not a fabricated one");

    let (got_v, got_a) = sharded.take_stage_output();
    let (cv, mv) = (cosine(&got_v, &taps.video.out), max_abs(&got_v, &taps.video.out));
    let (ca, ma) = (cosine(&got_a, &taps.audio.out), max_abs(&got_a, &taps.audio.out));
    eprintln!("single-shard vs. non-sharded (video): cosine={cv:.9}  max_abs={mv:.3e}");
    eprintln!("single-shard vs. non-sharded (audio): cosine={ca:.9}  max_abs={ma:.3e}");
    assert_eq!(got_v.len(), taps.video.out.len());
    assert_eq!(got_a.len(), taps.audio.out.len());
    assert!(cv >= 0.999_999_9, "single-shard video output diverges from the non-sharded reference: cosine={cv:.9}");
    assert!(mv < 1e-4, "single-shard video output diverges from the non-sharded reference: max_abs={mv:.3e}");
    assert!(ca >= 0.999_999_9, "single-shard audio output diverges from the non-sharded reference: cosine={ca:.9}");
    assert!(ma < 1e-4, "single-shard audio output diverges from the non-sharded reference: max_abs={ma:.3e}");

    let loss2 = Model::forward(&sharded);
    assert_eq!(loss2, 0.0, "no target was set - Model::forward's placeholder loss must be the documented 0.0, not a fabricated non-zero value");
}

/// A genuine two-stage split - block 0 on stage 0, block 1 on stage 1, the
/// boundary handed off through BOTH streams' residuals concatenated
/// ([`LtxAvDit::write_in_res`]/`read_out_res`'s doc), run sequentially on
/// a single device. [`LtxDit`]'s own `two_stage_boundary_handoff_
/// matches_the_non_sharded_reference` test's AV counterpart.
#[test]
fn two_stage_boundary_handoff_matches_the_non_sharded_reference_av() {
    let cfg = LtxAvDitConfig::tiny_gated();
    assert_eq!(cfg.video.num_layers, 2, "this test assumes the tiny_gated config's 2 layers split 1/1 across 2 stages");
    let weights = random_av_tiny_weights(&cfg, 13);
    let flat = flatten(&weights);
    let inp = build_inputs(&cfg, 17);

    let reference = LtxAvDit::new(cfg, weights, None);
    #[rustfmt::skip]
    let taps = reference.forward(
        &inp.v_latent, &inp.v_timesteps, &inp.v_positions, &inp.v_keyframes_mask, &inp.v_context, inp.v_context_len, inp.tv, inp.v_sigma, &vec![1.0; inp.v_context_len],
        &inp.a_latent, &inp.a_timesteps, &inp.a_positions, &inp.a_context, inp.a_context_len, inp.ta, inp.a_sigma, &vec![1.0; inp.a_context_len],
    );

    let shard0 = Shard { start: 0, end: 1, embed: true, head: false, gpu_index: Shard::ANY_GPU };
    let shard1 = Shard { start: 1, end: 2, embed: false, head: true, gpu_index: Shard::ANY_GPU };
    let stage0 = <LtxAvDit as Shardable>::new_shard(cfg, 1, inp.tv as u32, &flat, shard0);
    let stage1 = <LtxAvDit as Shardable>::new_shard(cfg, 1, inp.tv as u32, &flat, shard1);

    stage0.load_shard_batch(batch_for(&inp, None, None));
    stage1.load_shard_batch(batch_for(&inp, None, None));

    let l0 = <LtxAvDit as Shardable>::run_forward_stage(&stage0);
    assert!(l0.is_none(), "a non-head stage must never report a loss");
    let boundary = <LtxAvDit as Shardable>::read_out_res(&stage0);
    assert_eq!(boundary.len(), inp.tv * cfg.video.inner_dim as usize + inp.ta * cfg.audio.inner_dim as usize, "boundary residual must be video's + audio's residual concatenated");

    <LtxAvDit as Shardable>::write_in_res(&stage1, &boundary);
    let l1 = <LtxAvDit as Shardable>::run_forward_stage(&stage1);
    assert!(l1.is_none(), "no target was set on stage 1 either - no loss to report");

    let (got_v, got_a) = stage1.take_stage_output();
    let (cv, mv) = (cosine(&got_v, &taps.video.out), max_abs(&got_v, &taps.video.out));
    let (ca, ma) = (cosine(&got_a, &taps.audio.out), max_abs(&got_a, &taps.audio.out));
    eprintln!("2-stage composed vs. non-sharded (video): cosine={cv:.9}  max_abs={mv:.3e}");
    eprintln!("2-stage composed vs. non-sharded (audio): cosine={ca:.9}  max_abs={ma:.3e}");
    assert_eq!(got_v.len(), taps.video.out.len());
    assert_eq!(got_a.len(), taps.audio.out.len());
    assert!(cv >= 0.999_999_9, "2-stage composed video output diverges from the non-sharded reference: cosine={cv:.9}");
    assert!(mv < 1e-4, "2-stage composed video output diverges from the non-sharded reference: max_abs={mv:.3e}");
    assert!(ca >= 0.999_999_9, "2-stage composed audio output diverges from the non-sharded reference: cosine={ca:.9}");
    assert!(ma < 1e-4, "2-stage composed audio output diverges from the non-sharded reference: max_abs={ma:.3e}");
}

/// `run_stage_forward`'s combined-MSE-against-target path, both streams -
/// [`LtxDit`]'s own `head_stage_loss_is_a_real_mse_against_the_target` test's
/// AV counterpart.
#[test]
fn head_stage_loss_is_a_real_combined_mse_against_both_targets() {
    let cfg = LtxAvDitConfig::tiny_gated();
    let weights = random_av_tiny_weights(&cfg, 23);
    let flat = flatten(&weights);
    let inp = build_inputs(&cfg, 29);

    let whole = Shard::whole(cfg.video.num_layers as usize);
    let sharded = <LtxAvDit as Shardable>::new_shard(cfg, 1, inp.tv as u32, &flat, whole);
    let v_target = vec![0.0f32; inp.tv * cfg.video.out_channels as usize];
    let a_target = vec![0.0f32; inp.ta * cfg.audio.out_channels as usize];
    sharded.load_shard_batch(batch_for(&inp, Some(v_target), Some(a_target)));
    let loss = <LtxAvDit as Shardable>::run_forward_stage(&sharded).expect("head stage with targets must report a loss");
    assert!(loss.is_finite() && loss >= 0.0, "combined MSE must be finite and non-negative, got {loss}");

    let (out_v, out_a) = sharded.take_stage_output();
    sharded.load_shard_batch(batch_for(&inp, Some(out_v), Some(out_a)));
    let zero_loss = <LtxAvDit as Shardable>::run_forward_stage(&sharded).unwrap();
    assert!(zero_loss.abs() < 1e-8, "combined MSE against both predictions themselves must be ~0, got {zero_loss}");
}
