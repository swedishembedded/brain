// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-path gate for `imaging::tiling::BlendPlan` + `Ctx::blend_accumulate` -
//! the blended-overlap tiling variant this crate's own module doc
//! pre-authorizes ("if a model genuinely needs blended overlap ... that
//! arrives as a kernel plus a plan variant").
//!
//! One property, checked end to end on real device dispatches rather than the
//! plan's own host arithmetic (already gated in `crates/imaging/src/tiling.rs`):
//! a **tiled identity transform** - chop an image into overlapping tiles,
//! blend-recompose with no actual transform applied - round-trips to fp32
//! round-off. That is the only gate a real (transforming) consumer's tiles
//! could not pass by construction: real tiles genuinely disagree in the
//! overlap, so only the identity case isolates a mask/accumulate/divide bug
//! from a legitimate seam.

use gpu_core::testgpu;
use imaging::device::Ctx;
use imaging::{BlendPlan, BlendSpec, Rect, Shape, PIPELINES};

fn ctx() -> gpu_core::Gpu {
    testgpu::dev(PIPELINES)
}

/// Composite a whole image through a [`BlendPlan`] with the identity
/// "transform" (each tile's device round-trip is just upload -> download of
/// the exact source crop), and return the recomposed `[C,H,W]` image.
fn blend_identity(gpu: &gpu_core::Gpu, src: &[f32], c: u32, w: u32, h: u32, plan: &BlendPlan) -> Vec<f32> {
    let ctx = Ctx::new(gpu);
    let canvas_shape = Shape::new(1, c, h, w);
    let canvas = ctx.buf(canvas_shape.numel());
    gpu.submit(&[&canvas], &[]); // zero the accumulator before the first tile

    for t in &plan.tiles {
        // The "model": crop the tile out of the whole-image source on the
        // host (device crop2d is exercised elsewhere; this test isolates the
        // blend/accumulate path) - this tile's content is byte-identical to
        // the source, which is what makes deviation on recomposition provable.
        let mut tile_px = Vec::with_capacity((c * t.src.w * t.src.h) as usize);
        for ci in 0..c {
            for yy in t.src.y..t.src.bottom() {
                for xx in t.src.x..t.src.right() {
                    tile_px.push(src[((ci * h + yy) * w + xx) as usize]);
                }
            }
        }
        let tile_buf = ctx.upload("tile", &tile_px);
        let weight_buf = ctx.upload("weight", &t.weight);
        ctx.blend_accumulate(&canvas, canvas_shape, &tile_buf, &weight_buf, t.src);
    }

    // Normalise: reuse blend_accumulate itself with the reciprocal weight,
    // against a fresh zeroed accumulator, over the whole image - exactly the
    // second use its own doc describes.
    let out = ctx.buf(canvas_shape.numel());
    gpu.submit(&[&out], &[]);
    let recip = ctx.upload("recip", plan.recip_weight());
    ctx.blend_accumulate(&out, canvas_shape, &canvas, &recip, Rect::new(0, 0, w, h));

    ctx.download(&out, canvas_shape.numel())
}

#[test]
fn a_tiled_identity_transform_round_trips_to_fp32_round_off() {
    let gpu = ctx();
    let (c, w, h) = (2u32, 130u32, 97u32);
    let src: Vec<f32> =
        (0..c * w * h).map(|i| ((i as f32) * 0.0173).sin() + ((i as f32) * 0.0091).cos() * 0.5).collect();

    let plan = BlendPlan::new(w, h, BlendSpec::new(48, 12));
    assert!(plan.len() > 1, "expected a genuinely multi-tile plan, got {}", plan.len());
    assert!(plan.unity_error() < 1e-5);

    let got = blend_identity(&gpu, &src, c, w, h, &plan);
    assert_eq!(got.len(), src.len());
    let worst = got.iter().zip(&src).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "tiled identity round-trip worst |delta| = {worst}");
}

#[test]
fn a_single_tile_plan_blends_to_the_identity_bit_close() {
    let gpu = ctx();
    let (c, w, h) = (3u32, 40u32, 30u32);
    let src: Vec<f32> = (0..c * w * h).map(|i| (i as f32) * 0.001).collect();

    let plan = BlendPlan::new(w, h, BlendSpec::new(64, 8));
    assert_eq!(plan.len(), 1, "image fits in one tile");

    let got = blend_identity(&gpu, &src, c, w, h, &plan);
    let worst = got.iter().zip(&src).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1e-5, "single-tile plan must reproduce the source, worst |delta| = {worst}");
}
