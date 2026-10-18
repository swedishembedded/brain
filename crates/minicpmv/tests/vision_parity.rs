// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The vision tower, window merger and merger on the real checkpoint against
//! the reference's own taps, stage by stage, on the reference's own pixels:
//! a sliced image at 16x and at 4x, and two images packed in one request.

mod common;

use minicpmv::vision::{Taps, Tower, View, VISION_PIPELINES};
use minicpmv::Downsample;

/// The reference's NaViT-packed pixels (`[3, p, sum(h*w)*p]`) as patch rows.
fn patches_from_navit(px: &[f32], n: usize, p: usize) -> Vec<f32> {
    let width = n * p;
    let mut out = Vec::with_capacity(px.len());
    for i in 0..n {
        for c in 0..3 {
            for ky in 0..p {
                out.extend_from_slice(&px[c * p * width + ky * width + i * p..][..p]);
            }
        }
    }
    out
}

/// One case through `tower`, every stage against the reference's tap.
fn run(fx: &common::Fixture, tower: &Tower, case: &str, mode: Downsample, table: &mut brain_testutil::parity::Table) {
    let golden = fx.tensors(&format!("model/{case}.safetensors"));
    let sizes = &golden["target_sizes"].1;
    let views: Vec<View> = sizes.chunks(2).map(|s| View { h: s[0] as u32, w: s[1] as u32 }).collect();
    let n: u32 = views.iter().map(|v| v.patches()).sum();
    let patches = patches_from_navit(&golden["pixel_values"].1, n as usize, fx.cfg.vision.patch as usize);
    let last = fx.cfg.vision.n_layers - 1;
    let mut taps = Taps { layers: vec![(0, vec![]), (fx.cfg.vision.insert_layer_id, vec![]), (last, vec![])], ..Taps::default() };
    let merged = tower.encode_taps(&patches, &views, mode, Some(&mut taps));
    table.check(&format!("{case} embeddings"), &taps.embeddings, &golden["vision.embeddings"].1);
    for (l, got) in &taps.layers {
        table.check(&format!("{case} layer {l}"), got, &golden[&format!("vision.layer.{l}")].1);
    }
    if mode == Downsample::X16 {
        table.check(&format!("{case} window merger"), &taps.window_merger, &golden["vision.vit_merger"].1);
    }
    table.check(&format!("{case} post-layernorm"), &taps.post_norm, &golden["vision.post_layernorm"].1);
    table.check(&format!("{case} merger"), &merged, &golden["merger"].1);
}

/// A sliced image at 16x and at 4x, and two images packed in one request.
#[test]
fn every_stage_matches_the_reference_at_16x_at_4x_and_packed() {
    let Some(fx) = common::fixture() else { return };
    let weights = minicpmv::import::vision_weights(&fx.dir, &fx.cfg).expect("import the vision tower");
    let tower = Tower::new_on(gpu_core::testgpu::dev(VISION_PIPELINES), fx.cfg.clone(), &weights);
    drop(weights);
    let mut table = brain_testutil::parity::Table::new(0.9999, 1e-3);
    for (case, mode) in [("room_16x", Downsample::X16), ("room_4x", Downsample::X4), ("two_images", Downsample::X16)] {
        run(&fx, &tower, case, mode, &mut table);
    }
    table.print();
    assert!(table.failures.is_empty(), "{:#?}", table.failures);
}
