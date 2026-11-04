// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gates `vae::blocks::skipfuse::SkipFuse` (the seam that lets a trainable
//! prologue REPLACE the up path's skip concat, `Unet::new_fused`).
//!
//! Two properties, the same shape `tests/inject.rs` gates the analogous
//! `CrossAttnInject` seam with:
//!
//! 1. **A no-op `SkipFuse` (plain concat, identity mid/pre-upsample hooks)
//!    is bit-identical to no `SkipFuse` at all.** If merely installing the
//!    seam perturbed the graph, `tests/parity.rs`'s numbers would have been
//!    quietly invalidated by adding it.
//! 2. **The gate actually discriminates.** A deliberately broken `SkipFuse`
//!    (one join's two operands swapped) must FAIL the identity check -
//!    otherwise property 1 would pass trivially no matter what the seam did.
//!
//! Weight-free at `UNetConfig::tiny`, so it always runs.

use sdxlunet::config::UNetConfig;
use sdxlunet::init::init_weights;
use sdxlunet::model::{Unet, KERNELS};
use vae::blocks::skipfuse::{Map, SkipFuse};
use vae::blocks::Builder;

/// Plain `torch.cat([h_ori, skip], dim=1)` via every join, identity
/// elsewhere - reproduces `Rec::join_skip`'s own `None` branch exactly.
struct NoopFuse {
    joins: usize,
}

impl SkipFuse for NoopFuse {
    fn joins(&self) -> usize {
        self.joins
    }
    fn fuse_skip(&self, b: &mut Builder<'_>, _k: usize, h_ori: &Map, skip: &Map) -> Map {
        let buf = b.concat(h_ori.c, skip.c, h_ori.h, h_ori.w, &h_ori.buf, &skip.buf);
        Map { buf, c: h_ori.c + skip.c, h: h_ori.h, w: h_ori.w }
    }
}

/// Deliberately WRONG: swaps the two operands at one join (`k == 1`), which
/// still type-checks (same total width) but concatenates the channels in the
/// opposite order - every downstream resnet weight then reads the wrong
/// physical channels. Exists to prove the identity gate below actually
/// discriminates rather than trivially passing.
struct SwappedJoinFuse {
    joins: usize,
}

impl SkipFuse for SwappedJoinFuse {
    fn joins(&self) -> usize {
        self.joins
    }
    fn fuse_skip(&self, b: &mut Builder<'_>, k: usize, h_ori: &Map, skip: &Map) -> Map {
        if k == 1 {
            let buf = b.concat(skip.c, h_ori.c, h_ori.h, h_ori.w, &skip.buf, &h_ori.buf);
            Map { buf, c: h_ori.c + skip.c, h: h_ori.h, w: h_ori.w }
        } else {
            let buf = b.concat(h_ori.c, skip.c, h_ori.h, h_ori.w, &h_ori.buf, &skip.buf);
            Map { buf, c: h_ori.c + skip.c, h: h_ori.h, w: h_ori.w }
        }
    }
}

fn cfg_w() -> (UNetConfig, sdxlunet::import::Tensors) {
    let cfg = UNetConfig::tiny();
    let w = init_weights(&cfg, 11);
    (cfg, w)
}

fn inputs(m: &Unet) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let c = m.config();
    let sample: Vec<f32> = (0..(c.in_channels * 16 * 16) as usize).map(|i| ((i as f32) * 0.013).sin()).collect();
    let enc: Vec<f32> = (0..(9 * c.cross_attention_dim) as usize).map(|i| ((i as f32) * 0.029).cos()).collect();
    let pooled: Vec<f32> = (0..c.pooled_dim() as usize).map(|i| ((i as f32) * 0.07).sin()).collect();
    (sample, enc, pooled, vec![64.0, 64.0, 0.0, 0.0, 64.0, 64.0])
}

/// The number of `fuse_skip` calls one full forward makes: one per up-path
/// resnet, i.e. one per entry of the down-path skip stack.
fn n_joins(cfg: &UNetConfig) -> usize {
    cfg.skip_stack().len()
}

#[test]
fn a_noop_skipfuse_is_bit_identical_to_none() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (cfg, w) = cfg_w();
    let gpu = gpu_core::testgpu::dev(&KERNELS);

    let plain = Unet::new(gpu.share(), cfg.clone(), &w, 16, 16, 9, false);
    let (s, e, p, t) = inputs(&plain);
    let base = plain.run(&s, 601.0, &e, &p, &t);

    let noop = NoopFuse { joins: n_joins(&cfg) };
    let fused = Unet::new_fused(gpu.share(), cfg, &w, 16, 16, 9, false, &noop);
    let got = fused.run(&s, 601.0, &e, &p, &t);

    assert_eq!(base.len(), got.len());
    assert!(base.iter().all(|v| v.is_finite()), "plain graph is non-finite");
    let diff = base.iter().zip(&got).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    assert_eq!(diff, 0, "a no-op SkipFuse changed {diff} of {} outputs", base.len());
}

#[test]
fn a_broken_skipfuse_fails_the_identity_check() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (cfg, w) = cfg_w();
    let gpu = gpu_core::testgpu::dev(&KERNELS);

    let plain = Unet::new(gpu.share(), cfg.clone(), &w, 16, 16, 9, false);
    let (s, e, p, t) = inputs(&plain);
    let base = plain.run(&s, 601.0, &e, &p, &t);

    let broken = SwappedJoinFuse { joins: n_joins(&cfg) };
    let fused = Unet::new_fused(gpu.share(), cfg, &w, 16, 16, 9, false, &broken);
    let got = fused.run(&s, 601.0, &e, &p, &t);

    let diff = base.iter().zip(&got).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    assert!(diff > 0, "a broken SkipFuse (swapped join operands) produced a bit-identical output - the gate cannot discriminate");
}

/// A count mismatch must fail at construction with two numbers, not
/// mid-forward with a shape - the same contract [`Unet::new_injected`]'s own
/// wrong-site-count test pins.
#[test]
#[should_panic(expected = "skip-fuse joins")]
fn a_wrong_join_count_is_rejected_at_construction() {
    let (cfg, w) = cfg_w();
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let wrong = NoopFuse { joins: 1 };
    let _ = Unet::new_fused(gpu, cfg, &w, 16, 16, 9, false, &wrong);
}
