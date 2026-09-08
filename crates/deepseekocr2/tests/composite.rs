// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gates [`deepseekocr2::model::DeepseekOcr2`]: the row layout
//! ([`deepseekocr2::rows`]) and the composite splice into `deepseek2`'s
//! unmodified decoder (M5). Fully independent of `crates/deepseekocr2`'s
//! own checkpoint-free golden (`testdata/deepseekocr2/`, M2/M3/M4's fixture),
//! since this suite needs no golden at all, only internal consistency
//! between `gather_rows`/`scatter_rows` and a finite-difference check of the
//! whole composite's backward.
//!
//! Swedish Embedded AB builds from-scratch GPU training/inference stacks for
//! vision-language models. If your team needs a checkpoint-free integration
//! gate for a multi-tower composite, you can procure our services by
//! emailing info@swedishembedded.com.

use std::collections::HashMap;

use data::rng::Rng;
use deepseek2::DeepseekV2Config;
use deepseekocr2::config::{DeepseekOcr2VisionConfig, Qwen2EncoderConfig};
use deepseekocr2::encoder::{gather_rows, scatter_rows};
use deepseekocr2::model::DeepseekOcr2;
use deepseekocr2::rows::TileGrid;

/// A round trip through the pure host math needs no GPU at all: whatever
/// [`gather_rows`] concatenates, [`scatter_rows`] must split back apart
/// identically, over a grid shape where every axis (tile count,
/// `n_query_local`, `n_query_global`, `d_model`) is a different number, so a
/// swapped stride cannot cancel out by coincidence.
#[test]
fn gather_and_scatter_are_exact_inverses() {
    let (n_tiles, nl, ng, d) = (5usize, 3u32, 7u32, 4u32);
    let mut rng = Rng::new(11);
    let mut rnd = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.next_f32() - 0.5).collect() };

    let tiles: Vec<Vec<f32>> = (0..n_tiles).map(|_| rnd((nl * d) as usize)).collect();
    let global = rnd((ng * d) as usize);
    let separator = rnd(d as usize);

    let block = gather_rows(&tiles, &global, &separator);
    let (back_tiles, back_global, back_sep) = scatter_rows(&block, n_tiles, nl, ng, d);

    assert_eq!(back_tiles, tiles, "the tiles did not round-trip");
    assert_eq!(back_global, global, "the global view did not round-trip");
    assert_eq!(back_sep, separator, "the separator did not round-trip");
}

/// One tiny, self-contained composite: a vision config whose `decoder_hidden`
/// matches `DeepseekV2Config::tiny()`'s own `d_model` (12) - the ONE
/// dimension the two towers must agree on - while every other axis stays
/// distinct from both `deepseek2`'s and `deepseekocr2`'s own tiny fixtures,
/// so a transposed or coincidentally-matching axis elsewhere cannot mask a
/// wiring bug.
struct Fixture {
    m: DeepseekOcr2,
    grid: TileGrid,
    vision_cfg: DeepseekOcr2VisionConfig,
    seq: u32,
}

impl Fixture {
    fn new(seed: u64) -> Fixture {
        let sam = sam1::SamViTConfig { compress_out: 6, ..sam1::SamViTConfig::tiny() };
        let encoder = Qwen2EncoderConfig {
            d_model: 6,
            n_layers: 2,
            n_heads: 3,
            n_kv_heads: 1,
            ffn_hidden: 9,
            rms_eps: model::block::RMSNORM_EPS,
            rope_theta: 10_000.0,
            n_query_local: 2,
            n_query_global: 3,
        };
        let decoder_cfg = DeepseekV2Config::tiny();
        let vision_cfg = DeepseekOcr2VisionConfig { sam, encoder, decoder_hidden: decoder_cfg.shape.d_model };
        let grid = TileGrid::new(2, 1);
        let seq = decoder_cfg.block_size;

        let mut rng = Rng::new(seed);
        let mut vision_init: HashMap<String, Vec<f32>> = HashMap::new();
        let mut names = vision_cfg.encoder.param_list();
        names.extend(vision_cfg.projector_param_list());
        for (name, numel) in names {
            vision_init.insert(name, (0..numel).map(|_| (rng.next_f32() - 0.5) * 0.2).collect());
        }
        let decoder_init = deepseek2::init_weights(&decoder_cfg, seed ^ 0xd0d0);

        let gpu_vision = gpu_core::testgpu::dev(deepseekocr2::encoder::PIPELINES);
        let gpu_decoder = gpu_core::testgpu::dev(deepseek2::PIPELINES);
        let m = DeepseekOcr2::new_on(gpu_vision, gpu_decoder, vision_cfg.clone(), decoder_cfg, &vision_init, &decoder_init, grid, seq, 0, true);
        let f = Fixture { m, grid, vision_cfg, seq };
        let (ids, targets) = f.tokens();
        f.m.set_tokens(&ids, &targets);
        f
    }

    fn sam_inputs(&self, seed: u64) -> (Vec<Vec<f32>>, Vec<f32>) {
        let e = &self.vision_cfg.encoder;
        let mut rng = Rng::new(seed);
        let mut rnd = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.next_f32() - 0.5).collect() };
        let tiles: Vec<Vec<f32>> = (0..self.grid.tiles()).map(|_| rnd((e.n_query_local * e.d_model) as usize)).collect();
        let global = rnd((e.n_query_global * e.d_model) as usize);
        (tiles, global)
    }

    fn tokens(&self) -> (Vec<u32>, Vec<u32>) {
        let vocab = 19u32; // DeepseekV2Config::tiny()'s own vocab
        let ids: Vec<u32> = (0..self.seq).map(|i| i % vocab).collect();
        let mut targets: Vec<u32> = ids[1..].to_vec();
        targets.push(deepseek2::IGNORE);
        (ids, targets)
    }
}

/// The separator row of the gathered block is EXACTLY the checkpoint's
/// `vision.view_separator` tensor - a direct copy, so equality, not a
/// tolerance - and it lands at [`deepseekocr2::rows::RowPlan::separator_row`],
/// the very last row of the image block.
#[test]
fn the_separator_row_is_exactly_the_checkpoint_tensor_at_the_planned_position() {
    let f = Fixture::new(101);
    let (tiles, global) = f.sam_inputs(202);
    let (_, st) = f.m.forward(&tiles, &global);
    let d = f.vision_cfg.decoder_hidden as usize;
    let block = f.m.read_decoder_input();
    let sep_row = f.m.row_plan().separator_row() as usize;
    let (row0, n_rows) = f.m.image_run();
    assert_eq!(row0, 0);

    let spliced_separator = &block[(sep_row) * d..(sep_row + 1) * d];
    let checkpoint_separator = f.m.encoder().read_weight("vision.view_separator");
    assert_eq!(spliced_separator, checkpoint_separator.as_slice(), "the spliced separator row is not the checkpoint tensor verbatim");
    assert_eq!(sep_row as u32, n_rows - 1, "the separator is not the image block's last row");
    drop(st); // forward's state is only needed by backward; this test stops at the splice.
}

/// A projector row lands exactly where [`deepseekocr2::rows::RowPlan::runs`]
/// says it does: the first row of tile 1's run is bit-identical to that
/// tile's own projected output, read independently through
/// `Resampler::forward_train` on the same weights.
#[test]
fn projector_rows_land_at_the_row_plans_positions() {
    let f = Fixture::new(303);
    let (tiles, global) = f.sam_inputs(404);
    let (_, st) = f.m.forward(&tiles, &global);
    let d = f.vision_cfg.decoder_hidden as usize;
    let block = f.m.read_decoder_input();

    let (proj_tile1, _) = f.m.encoder().forward_train(&tiles[1], true);
    let runs = f.m.row_plan().runs();
    let (row0, n) = runs[1]; // tile 1's run
    assert_eq!(n, f.vision_cfg.encoder.n_query_local);
    let spliced = &block[row0 as usize * d..(row0 as usize + n as usize) * d];
    assert_eq!(spliced, proj_tile1.as_slice(), "tile 1's rows in the spliced block do not match its own projected output");
    drop(st);
}

/// The whole composite's backward, end to end: perturb one local tile's SAM
/// input along a fixed random direction and confirm the loss's directional
/// derivative matches `backward`'s returned gradient for that tile - proof
/// that `gather_rows`/`scatter_rows`/`deepseek2`'s splice compose correctly,
/// not just that each piece is right in isolation (M4 already gated the
/// resampler alone; this is the NEW integration risk).
#[test]
fn the_composite_backward_matches_a_finite_difference_through_the_whole_splice() {
    let f = Fixture::new(505);
    let (tiles0, global) = f.sam_inputs(606);
    let mut rng = Rng::new(707);
    let dir: Vec<f32> = (0..tiles0[0].len()).map(|_| rng.next_f32() - 0.5).collect();

    f.m.zero_grads();
    let (loss0, st) = f.m.forward(&tiles0, &global);
    let (d_tiles, _d_global) = f.m.backward(st);
    let analytic: f64 = d_tiles[0].iter().zip(&dir).map(|(g, v)| *g as f64 * *v as f64).sum();

    let eps = 5e-2f32; // fp32-quantum headroom, the same order v1's own composite needed
    let perturb = |sign: f32| -> f32 {
        let mut t = tiles0.clone();
        for (x, d) in t[0].iter_mut().zip(&dir) {
            *x += sign * eps * d;
        }
        f.m.zero_grads();
        f.m.forward(&t, &global).0
    };
    let (lp, lm) = (perturb(1.0), perturb(-1.0));
    let numeric = (lp as f64 - lm as f64) / (2.0 * eps as f64);

    let rel = (analytic - numeric).abs() / analytic.abs().max(numeric.abs()).max(1e-9);
    println!("composite backward: analytic {analytic:+.6e} numeric {numeric:+.6e} rel {rel:.3e} (loss0 {loss0:.6e})");
    assert!(rel < 1e-1, "analytic {analytic} vs numeric {numeric} (rel {rel})");
    // The floor only needs to rule out a gradient that never left zero (a
    // dropped splice, a disconnected tile) - not assume any particular
    // scale. This fixture's own true gradient is ~9e-6 (a small, causally
    // distant contribution through the decoder, not a bug): the numeric
    // central difference is the independent witness that this magnitude is
    // real, so the floor sits two decades below it, not above it.
    assert!(analytic.abs() > 1e-7, "a ~zero directional derivative proves nothing - the gradient never reached tile 0");
}

/// A mutation check for the row order itself: swapping which tile's rows
/// [`gather_rows`] places first must move the block a real caller reads back,
/// otherwise this whole suite could pass with a row order nobody checked.
#[test]
fn swapping_tile_order_in_gather_rows_changes_the_assembled_block() {
    let (n_tiles, nl, ng, d) = (3usize, 2u32, 3u32, 4u32);
    let mut rng = Rng::new(909);
    let mut rnd = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.next_f32() - 0.5).collect() };
    let tiles: Vec<Vec<f32>> = (0..n_tiles).map(|_| rnd((nl * d) as usize)).collect();
    let global = rnd((ng * d) as usize);
    let separator = rnd(d as usize);

    let forward_order = gather_rows(&tiles, &global, &separator);
    let mut swapped = tiles.clone();
    swapped.swap(0, 1);
    let swapped_order = gather_rows(&swapped, &global, &separator);
    assert_ne!(forward_order, swapped_order, "swapping two tiles must change the assembled block - the row order is load-bearing");
}
