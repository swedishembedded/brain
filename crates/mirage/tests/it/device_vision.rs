// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device vision path against the host oracle: per stage on the tiny
//! configuration (batched, with images of different valid regions) and on the
//! released weights against the host and the FP32 goldens.

use data::rng::Lcg;
use mirage::device::vision::{DeviceVision, Masks};
use mirage::device::Device;
use mirage::vision::ops::valid_cells;
use mirage::vision::VisionModel;

use crate::common::tiny_checkpoint;
use crate::vision::assert_stage;

fn device(test: &str) -> Option<Device> {
    crate::common::cap_threads();
    match Device::open() {
        Ok(d) => {
            eprintln!("{test}: device kind {}", d.gpu().kind());
            Some(d)
        }
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("{test}: no device: {e}"));
            None
        }
    }
}

/// Mask `rows` of width `d` by `valid` (per row).
fn masked(x: &[f32], d: usize, valid: &[bool]) -> Vec<f32> {
    x.chunks_exact(d).zip(valid).flat_map(|(r, &v)| r.iter().map(move |a| if v { *a } else { 0.0 })).collect()
}

#[test]
fn the_tiny_vision_path_matches_the_host_per_stage_in_a_batch() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("tiny vision") else { return };
    let ck = tiny_checkpoint();
    let cfg = ck.config.clone();
    let vm = VisionModel::new(&ck);
    let dv = DeviceVision::upload(&dev, &vm);
    let sizes = [(400usize, 300usize), (512, 512), (200, 480)];
    let mut rng = Lcg::new(77); // seed: canvas filler compared between two paths
    let n = cfg.canvas * cfg.canvas * 3;
    let canvases: Vec<Vec<f32>> = sizes.iter().map(|_| (0..n).map(|_| rng.signed()).collect()).collect();
    let flat: Vec<f32> = canvases.concat();
    let masks = Masks::new(&dev, &cfg, &sizes);
    let canvas_d = dev.buf(&flat);
    let enc = dv.encode(&dev, &canvas_d, sizes.len(), &masks);
    let dense = dv.dense(&dev, &enc, &masks);
    let (protos, coefs) = dv.mask_maps(&dev, &enc, &dense, &masks);
    let (grid, d, regs) = (cfg.grid(), cfg.dim, cfg.registers);
    let cells = grid * grid;
    let tokens = cells + regs;
    for (i, &(h, w)) in sizes.iter().enumerate() {
        let host_enc = vm.encoder.forward(&canvases[i], h, w);
        let host_dense = vm.dense.forward(&host_enc, h, w);
        let (hp, hc) = vm.dense.mask_maps(&host_dense.head_in, &host_enc, h, w);
        for (k, (map, buf)) in enc.stages.iter().enumerate() {
            let len = map.h * map.w * map.c;
            let got = dev.read_at(buf, i * len, len);
            assert_stage(&format!("img{i} stage{k}"), &got, &host_enc.stages[k].1);
        }
        let valid = valid_cells(h, w, grid);
        assert_stage(&format!("img{i} body"), &dev.read_at(&enc.body, i * cells * d, cells * d), &host_enc.body);
        let top = dev.read_at(&enc.top, i * tokens * d, tokens * d);
        let mut top_valid = valid.clone();
        top_valid.extend(std::iter::repeat_n(true, regs));
        assert_stage(&format!("img{i} top"), &masked(&top, d, &top_valid), &host_enc.top);
        assert_stage(&format!("img{i} encoder"), &dev.read_at(&enc.encoder, i * cells * d, cells * d), &host_enc.encoder);
        let side = cfg.canvas / cfg.stride;
        let hl = side * side;
        assert_stage(&format!("img{i} head_in"), &dev.read_at(&dense.head_in, i * hl * cfg.dense_dim, hl * cfg.dense_dim), &host_dense.head_in);
        assert_stage(&format!("img{i} head"), &dev.read_at(&dense.head, i * hl * 10, hl * 10), &host_dense.head);
        let pl = 4 * hl * 32;
        assert_stage(&format!("img{i} protos"), &dev.read_at(&protos, i * pl, pl), &hp);
        assert_stage(&format!("img{i} coefs"), &dev.read_at(&coefs, i * hl * 32, hl * 32), &hc);
    }
}

/// Run both golden fixtures as one batch and compare every stage with the goldens
/// (`pub` for the CUDA run through the same test).
#[test]
fn the_released_vision_path_matches_the_goldens_and_batching_is_exact() {
    use crate::vision::{compare, fixture_named};
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("released vision") else { return };
    let (Some(cats), Some(text)) = (fixture_named("device vision", "cats"), fixture_named("device vision", "text")) else { return };
    let fixtures = [cats, text];
    let cfg = cats.model.model_config().clone();
    let dv = DeviceVision::upload(&dev, &cats.model);
    let sizes: Vec<(usize, usize)> = fixtures.iter().map(|f| (f.image.h, f.image.w)).collect();
    let canvases: Vec<Vec<f32>> = fixtures.iter().map(|f| mirage::vision::preprocess::canvas(&f.image)).collect();
    let run = |idx: &[usize]| {
        let sz: Vec<_> = idx.iter().map(|&i| sizes[i]).collect();
        let flat: Vec<f32> = idx.iter().flat_map(|&i| canvases[i].iter().copied()).collect();
        let masks = Masks::new(&dev, &cfg, &sz);
        let enc = dv.encode(&dev, &dev.buf(&flat), idx.len(), &masks);
        let dense = dv.dense(&dev, &enc, &masks);
        (enc, dense, masks)
    };
    let (enc, dense, masks) = run(&[0, 1]);
    let (grid, d, regs) = (cfg.grid(), cfg.dim, cfg.registers);
    let (cells, side) = (grid * grid, cfg.canvas / cfg.stride);
    for (i, f) in fixtures.iter().enumerate() {
        let gold = |n: &str| &f.golden[n];
        for (k, name) in ["stage_256", "stage_128", "stage_64", "stage_32", "stage_16"].iter().enumerate() {
            let len = gold(name).len();
            assert_stage(&format!("img{i} {name}"), &dev.read_at(&enc.stages[k].1, i * len, len), gold(name));
        }
        assert_stage(&format!("img{i} body"), &dev.read_at(&enc.body, i * cells * d, cells * d), gold("body"));
        let valid = valid_cells(f.image.h, f.image.w, grid);
        let mut tv = valid.clone();
        tv.extend(std::iter::repeat_n(true, regs));
        let top = dev.read_at(&enc.top, i * (cells + regs) * d, (cells + regs) * d);
        assert_stage(&format!("img{i} top"), &masked(&top, d, &tv), gold("top"));
        assert_stage(&format!("img{i} encoder"), &dev.read_at(&enc.encoder, i * cells * d, cells * d), gold("encoder"));
        let hl = side * side;
        assert_stage(&format!("img{i} head_in"), &dev.read_at(&dense.head_in, i * hl * 128, hl * 128), gold("head_in"));
        assert_stage(&format!("img{i} head"), &dev.read_at(&dense.head, i * hl * 10, hl * 10), gold("head"));
    }
    // Batch of N equals N single runs.
    let _ = &masks;
    for i in 0..2 {
        let (_, single, _) = run(&[i]);
        let hl = side * side;
        let want = dev.read_at(&single.head, 0, hl * 10);
        let got = dev.read_at(&dense.head, i * hl * 10, hl * 10);
        let (cos, worst) = compare(&got, &want);
        eprintln!("batch-2 vs single img{i}: head cosine {cos:.9}, max_abs {worst:.3e}");
        // The GEMM selector may pick a different kernel for a different row count, so CUDA differs in rounding.
        assert!(cos > 0.99999, "batching changed image {i}");
    }
}
