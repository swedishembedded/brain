// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device reader and decoder-memory construction against the host oracle.

use data::rng::Lcg;
use mirage::device::reader::{DeviceMemory, DeviceReader, SlotInputs, SlotItem};
use mirage::device::Device;
use mirage::vision::ops::valid_cells;
use mirage::vision::reader::{WINDOW_H, WINDOW_W};
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

#[test]
fn the_tiny_reader_matches_the_host_over_a_batch_of_windows() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("tiny reader") else { return };
    let vm = VisionModel::new(&tiny_checkpoint());
    let dr = DeviceReader::upload(&dev, &vm.reader);
    let mut rng = Lcg::new(31); // seed: window filler compared between two paths
    let n = 3;
    let rgb: Vec<u8> = (0..n * WINDOW_H * WINDOW_W * 3).map(|_| (rng.signed() * 127.0 + 128.0) as u8).collect();
    let got = dr.forward(&dev, &rgb, n);
    let per = WINDOW_H * WINDOW_W * 3;
    let want: Vec<f32> = (0..n).flat_map(|i| vm.reader.forward(&rgb[i * per..(i + 1) * per])).collect();
    assert_stage("tiny reader logits", &got, &want);
}

#[test]
fn the_released_reader_and_memory_match_the_host() {
    use crate::vision::fixture_named;
    let _g = gpu_core::testgpu::exclusive();
    let Some(dev) = device("released reader") else { return };
    let (Some(cats), Some(text)) = (fixture_named("device reader", "cats"), fixture_named("device reader", "text")) else { return };
    let vm = &cats.model;
    let cfg = vm.model_config().clone();
    // Reader: all windows of the text fixture's boxes, as the golden sizes allow: reuse 4 windows cut from the resized image.
    let dr = DeviceReader::upload(&dev, &vm.reader);
    let img = &text.image;
    let strip = mirage::vision::preprocess::Resized { rgb: img.rgb[..img.w * 48 * 3].to_vec(), h: 48, w: img.w };
    let wins = mirage::vision::reader::windows(&strip);
    let flat: Vec<u8> = wins.concat();
    let got = dr.forward(&dev, &flat, wins.len());
    let want: Vec<f32> = wins.iter().flat_map(|w| vm.reader.forward(w)).collect();
    assert_stage("released reader logits", &got, &want);

    // Memory: both fixtures in one batch against the host builder and the golden.
    let dm = DeviceMemory::upload(&dev, &cfg, &vm.memory);
    let fixtures = [cats, text];
    let valids: Vec<Vec<bool>> = fixtures.iter().map(|f| valid_cells(f.image.h, f.image.w, cfg.grid())).collect();
    let items: Vec<SlotItem> = fixtures
        .iter()
        .zip(&valids)
        .map(|(f, v)| SlotItem { encoder: &f.analysis.encoder.encoder, head_in: &f.analysis.dense.head_in, valid: v, slots: vm.slots(&f.analysis.objects, &f.analysis.text) })
        .collect();
    let slots = SlotInputs::collect(&vm.memory, &items, cfg.detections + cfg.regions);
    let enc: Vec<f32> = fixtures.iter().flat_map(|f| f.analysis.encoder.encoder.iter().copied()).collect();
    let mem = dm.build(&dev, &dev.buf(&enc), &valids, &slots);
    let d = cfg.dim;
    for (i, f) in fixtures.iter().enumerate() {
        let rows = dev.read_at(&mem.rows, i * mem.stride * d, mem.stride * d);
        let mask = dev.read_at(&mem.mask, i * mem.stride, mem.stride);
        let live: Vec<f32> = rows.chunks_exact(d).zip(&mask).filter(|(_, m)| **m > 0.0).flat_map(|(r, _)| r.iter().copied()).collect();
        assert_stage(&format!("img{i} memory vs host"), &live, &vm.decoder_memory(&f.analysis));
        assert_stage(&format!("img{i} memory vs golden"), &live, &f.golden["memory"]);
    }
}
