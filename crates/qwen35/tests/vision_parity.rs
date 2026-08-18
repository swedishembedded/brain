// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Vision tower parity at real dims, random weights (M9):
//! `qwen3vl::encoder::VisionEncoder`/`PatchMerger` (reused unchanged) against
//! the real reference goldens dumped by
//! `tools/goldens/qwen35_vision_dump_reference.py`
//! (`transformers.models.qwen3_5.Qwen3_5VisionModel`, not a secondhand
//! description). Weights come from the golden's own saved
//! `qwen35_vision_weights.safetensors` (already renamed to
//! `crate::vl`/`qwen3vl::encoder`'s conventions - see that dumper's own
//! module doc), so this test needs no checkpoint and no import machinery.
//! `crate::vl::Qwen35Vl` splices this SAME `VisionEncoder`/`PatchMerger`
//! pair into the decoder, so validating them here in isolation, at real
//! dims, against the real reference is what proves the composite's vision
//! half is correct (`crate::vl` itself has no independent oracle - see its
//! own module doc - this is the achievable half of that gap).

use std::collections::HashMap;
use std::path::Path;

use brain_testutil::parity::Table;
use checkpoint::safetensors::StTensor;
use gpu_core::Gpu;
use qwen3vl::config::VisionConfig;
use qwen3vl::encoder::{vision_pipelines, PatchMerger, VisionEncoder};

const OUT_HIDDEN_SIZE: u32 = 5120; // this model's own decoder d_model

fn to_map(tensors: Vec<StTensor>) -> HashMap<String, Vec<f32>> {
    tensors.into_iter().map(|t| (t.name, t.data)).collect()
}

struct Golden {
    encoder_w: HashMap<String, Vec<f32>>,
    merger_w: HashMap<String, Vec<f32>>,
    taps: HashMap<String, Vec<f32>>,
}

fn load() -> Option<Golden> {
    let dir = brain_testutil::testdata("golden/qwen35/vision");
    let w_path = format!("{dir}/qwen35_vision_weights.safetensors");
    let t_path = format!("{dir}/qwen35_vision.safetensors");
    if !Path::new(&w_path).exists() || !Path::new(&t_path).exists() {
        brain_testutil::skip(&format!("fixture {w_path} absent - run tools/goldens/qwen35_vision_dump_reference.py"));
        return None;
    }
    let weights = to_map(checkpoint::safetensors::read(&w_path).expect("read golden weights"));
    let taps = to_map(checkpoint::safetensors::read(&t_path).expect("read golden taps"));

    let mut encoder_w = HashMap::new();
    let mut merger_w = HashMap::new();
    for (name, data) in weights {
        if let Some(rest) = name.strip_prefix("merger.") {
            merger_w.insert(rest.to_string(), data);
        } else {
            encoder_w.insert(name, data);
        }
    }
    Some(Golden { encoder_w, merger_w, taps })
}

/// Real-dims vision config, matching `qwen35_vision_dump_reference.py`'s own
/// pinned-default assertions - byte-for-byte `VisionConfig::qwen3_omni()`
/// apart from `out_hidden_size` (this decoder's `d_model`) and
/// `deepstack_indexes` (empty - this model has no DeepStack).
fn real_vcfg() -> VisionConfig {
    VisionConfig { out_hidden_size: OUT_HIDDEN_SIZE, deepstack_indexes: vec![], ..VisionConfig::qwen3_omni() }
}

fn run(gpu: Gpu) {
    let Some(golden) = load() else { return };
    let vcfg = real_vcfg();
    assert_eq!(vcfg.depth, 27);
    assert_eq!(vcfg.hidden, 1152);

    let grid = &golden.taps["grid_thw"];
    let (t, h, w) = (grid[0].round() as u32, grid[1].round() as u32, grid[2].round() as u32);
    assert_eq!(t, 1, "single-frame image only - the golden's own scope");

    let enc = VisionEncoder::new(&gpu, vcfg.clone(), &golden.encoder_w);
    let tap_indices = [0u32, vcfg.depth - 1];
    let (features, tap_feats) = enc.encode_with_taps(h, w, &golden.taps["patches"], &tap_indices);

    let mut table = Table::new(0.9999, 1e-3);
    table.check("block0", &tap_feats[0], &golden.taps["block0"]);
    table.check(&format!("block{}", vcfg.depth - 1), &tap_feats[1], &golden.taps[&format!("block{}", vcfg.depth - 1)]);
    table.check("hidden (pre-merger)", &features, &golden.taps["hidden"]);

    let merger = PatchMerger::new(&gpu, &golden.merger_w, vcfg.hidden, vcfg.spatial_merge_size, vcfg.out_hidden_size, false);
    let merged = merger.merge(&features, h * w);
    table.check("merged (post-merger)", &merged, &golden.taps["merged"]);

    table.print();
    assert!(table.failures.is_empty(), "vision parity failures: {:#?}", table.failures);
}

#[test]
fn real_dims_vision_tower_matches_the_reference_cpu() {
    run(Gpu::new_cpu(vision_pipelines()));
}

#[test]
fn real_dims_vision_tower_matches_the_reference_default_backend() {
    run(Gpu::new(vision_pipelines()));
}
