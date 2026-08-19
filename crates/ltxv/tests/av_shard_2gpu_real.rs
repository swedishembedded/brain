// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The AV pipeline-parallel `Shardable` seam, run for REAL on two distinct
//! physical GPUs - closing the roadmap gap `av_shard_parity.rs`'s own doc
//! names explicitly: "this proves the boundary handoff is correct, not that
//! two real cards agree". `av_shard_parity.rs` already proves the boundary
//! handoff is correct, single-process, at bit-exact precision; this file
//! proves the SAME two stages genuinely dispatch on two SEPARATE physical
//! devices and still agree.
//!
//! ## Mechanism
//!
//! `gpu_core::devices::with_gpu(index, f)` pins `f`'s thread-local device
//! selection to canonical GPU `index` - every `Gpu::new`/`open_device` call
//! `f` makes (including the one buried inside [`LtxAvDit::run_stage_forward`],
//! which opens a fresh device per call rather than caching one - see
//! `crate::dit`'s doc) resolves against that physical card. Wrapping stage
//! 0's `run_stage_forward` in `with_gpu(0, ..)` and stage 1's in
//! `with_gpu(1, ..)` therefore genuinely places the two stages on two
//! different cards; the boundary residual crosses between them as a plain
//! host `Vec<f32>` (`read_out_res` on card 0 -> `write_in_res` on card 1) -
//! a real host round trip, which is what actually crosses the PCIe boundary
//! between two distinct devices (unlike a same-device `DeviceBuffer`, which
//! never leaves VRAM). This is the exact mechanism `model::shard::Pipeline::
//! with_shards` already uses for construction-time placement
//! (`gpu_core::devices::with_gpu(sh.gpu_index as u32, ...)`); this test
//! applies it to FORWARD-time placement instead, since `LtxAvDit` (like
//! `LtxDit`) opens its device fresh per `run_stage_forward` call rather than
//! caching one at construction (see `crate::shard`'s own module doc on why
//! `Pipeline<LtxDit>` is not a usable end-to-end entry point for this model).
//!
//! ## Why a `#[test]`, not a standalone binary
//!
//! The task briefing allows a `ltxv_bench`-style manual binary "if genuinely
//! two separate GPU devices are hard to drive from one test process" -
//! `with_gpu`'s thread-local scoping (already the repo's own established
//! mechanism, see `model::shard::Pipeline`) makes that NOT the case here, so
//! this stays a normal, deterministic, hardware-gated `#[test]` (skips
//! loudly on a box with fewer than 2 GPUs, via [`brain_testutil::skip`]).
//!
//! ## Scale
//!
//! Deliberately the SAME `tiny_gated` synthetic config `av_shard_parity.rs`
//! uses (2 layers, video `inner_dim` 64 / audio `inner_dim` 32, gated
//! attention + both connectors ON) - proving the MECHANISM (two real cards,
//! two stages, one boundary crossing) is what this file is for, not a
//! real-22B-weight claim. A real-checkpoint-weight version of this same
//! mechanism would need a GGUF-streaming int8 shard loader this pass did not
//! build (see this crate's roadmap ledger for the tracked gap) - out of
//! scope for the "start small" budget this task set.

use std::collections::HashMap;

use ltxv::dit::{random_av_tiny_weights, AvDitBatch};
use ltxv::{LtxAvDit, LtxAvDitConfig};
use model::{Shard, Shardable};

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
    let v_context_len = 6usize; // multiple of tiny_gated's 3 connector registers
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
    let a_positions: Vec<f32> = (0..ta).flat_map(|i| [i as f32, i as f32 + 1.0]).collect();
    let mut v_keyframes_mask = vec![0f32; tv];
    v_keyframes_mask[0] = 1.0;
    v_keyframes_mask[tv - 1] = 1.0;

    TestInputs { v_latent, v_timesteps, v_positions, v_keyframes_mask, v_context, v_context_len, tv, v_sigma: 0.6, a_latent, a_timesteps, a_positions, a_context, a_context_len, ta, a_sigma: 0.4 }
}

fn batch_for(inp: &TestInputs) -> AvDitBatch {
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
        v_target: None,
        a_target: None,
    }
}

#[test]
fn two_real_gpus_agree_on_the_av_shardable_seam() {
    let devs = gpu_core::devices::gpus();
    if devs.len() < 2 {
        brain_testutil::skip(&format!("this test needs 2 physical GPUs, only {} available", devs.len()));
        return;
    }
    eprintln!(
        "running the AV 2-stage pipeline on gpu0={} ({:?}) and gpu1={} ({:?})",
        devs[0].identity.name, devs[0].identity.pci_bus, devs[1].identity.name, devs[1].identity.pci_bus
    );

    let cfg = LtxAvDitConfig::tiny_gated();
    let weights = random_av_tiny_weights(&cfg, 41);
    let flat = flatten(&weights);
    let inp = build_inputs(&cfg, 43);

    // Reference: the same non-sharded forward `av_shard_parity.rs` uses,
    // run on whatever the ambient/ANY_GPU selection lands on (card 0, absent
    // an override) - this is the single-process ground truth.
    let reference = LtxAvDit::new(cfg, weights, None);
    #[rustfmt::skip]
    let taps = reference.forward(
        &inp.v_latent, &inp.v_timesteps, &inp.v_positions, &inp.v_keyframes_mask, &inp.v_context, inp.v_context_len, inp.tv, inp.v_sigma, &vec![1.0; inp.v_context_len],
        &inp.a_latent, &inp.a_timesteps, &inp.a_positions, &inp.a_context, inp.a_context_len, inp.ta, inp.a_sigma, &vec![1.0; inp.a_context_len],
    );

    let shard0 = Shard { start: 0, end: 1, embed: true, head: false, gpu_index: 0 };
    let shard1 = Shard { start: 1, end: 2, embed: false, head: true, gpu_index: 1 };
    let stage0 = <LtxAvDit as Shardable>::new_shard(cfg, 1, inp.tv as u32, &flat, shard0);
    let stage1 = <LtxAvDit as Shardable>::new_shard(cfg, 1, inp.tv as u32, &flat, shard1);
    stage0.load_shard_batch(batch_for(&inp));
    stage1.load_shard_batch(batch_for(&inp));

    // Stage 0 for real on physical GPU 0.
    let l0 = gpu_core::devices::with_gpu(0, || <LtxAvDit as Shardable>::run_forward_stage(&stage0)).expect("with_gpu(0, ..): placement on card 0 failed");
    assert!(l0.is_none(), "a non-head stage must never report a loss");
    let boundary = <LtxAvDit as Shardable>::read_out_res(&stage0);

    // Cross the boundary as a plain host Vec<f32> - the real thing that
    // moves between two distinct physical devices (no DeviceBuffer here).
    <LtxAvDit as Shardable>::write_in_res(&stage1, &boundary);

    // Stage 1 for real on physical GPU 1.
    let l1 = gpu_core::devices::with_gpu(1, || <LtxAvDit as Shardable>::run_forward_stage(&stage1)).expect("with_gpu(1, ..): placement on card 1 failed");
    assert!(l1.is_none(), "no target was set on stage 1 either - no loss to report");

    let (got_v, got_a) = stage1.take_stage_output();
    let (cv, mv) = (cosine(&got_v, &taps.video.out), max_abs(&got_v, &taps.video.out));
    let (ca, ma) = (cosine(&got_a, &taps.audio.out), max_abs(&got_a, &taps.audio.out));
    eprintln!("2-real-GPU composed vs. non-sharded (video): cosine={cv:.9}  max_abs={mv:.3e}");
    eprintln!("2-real-GPU composed vs. non-sharded (audio): cosine={ca:.9}  max_abs={ma:.3e}");
    assert_eq!(got_v.len(), taps.video.out.len());
    assert_eq!(got_a.len(), taps.audio.out.len());
    assert!(cv >= 0.999_999_9, "2-real-GPU composed video output diverges from the non-sharded reference: cosine={cv:.9}");
    assert!(mv < 1e-4, "2-real-GPU composed video output diverges from the non-sharded reference: max_abs={mv:.3e}");
    assert!(ca >= 0.999_999_9, "2-real-GPU composed audio output diverges from the non-sharded reference: cosine={ca:.9}");
    assert!(ma < 1e-4, "2-real-GPU composed audio output diverges from the non-sharded reference: max_abs={ma:.3e}");
}
