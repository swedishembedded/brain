// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded serving of language models for its clients. If your team needs
// expertise in keeping the host footprint of generation within budget then you can procure our services by sending
// an email to info@swedishembedded.com.

//! `generate` applies the LM head on the device: decoding keeps no f32 host copy of the head (at a 152k vocabulary
//! and `d_model` 3584 that copy is 2.2 GB). The model here is mostly head, so a host copy shows as a second file's
//! worth of host memory on top of the weights (which, on a CPU device, are host memory themselves).

use std::sync::Arc;

use capability::{Invocation, Provider, Registry};
use qwen3::caps::{QwenProvider, MODEL};
use qwen3::config::QwenConfig;
use serde_json::json;

fn write(path: &std::path::Path, cfg: &QwenConfig) -> u64 {
    let mut rng = data::rng::Lcg::random();
    brain_testutil::load_peak::write_f32_checkpoint(path, &cfg.param_list(), &cfg.to_json(), None, &mut |name, v| {
        // Norm gains one, everything else small: any weights decode, this checks memory.
        v.iter_mut().for_each(|x| *x = if name.contains("norm") { 1.0 } else { rng.scaled(0.02) })
    })
}

fn generate(path: &std::path::Path) {
    let mut reg = Registry::new();
    reg.register(Arc::new(QwenProvider::new()) as Arc<dyn Provider>);
    let inv = Invocation::new().set("weights", json!(path.to_str().unwrap())).set("prompt", json!("3 17 9 1")).set("max_new", json!(6)).set("temp", json!(0.0));
    reg.run(MODEL, "generate", inv, &mut |_| {}).unwrap_or_else(|e| panic!("generate: {e}"));
}

#[test]
fn generating_keeps_no_host_copy_of_the_head() {
    brain_testutil::run_alone(module_path!(), "host_peak_of_a_generation");
}

#[test]
#[ignore = "run alone by generating_keeps_no_host_copy_of_the_head"]
fn host_peak_of_a_generation() {
    let dir = brain_testutil::Scratch::dir("qwen3-generate-peak");
    let small = QwenConfig { vocab: 64, block_size: 64, max_position_embeddings: 64, ..QwenConfig::tiny() };
    // A tied 65536 x 640 table: about 168 MB of the file's 170.
    let sized = QwenConfig { vocab: 65_536, d_model: 640, ..small.clone() };
    let (path, warm) = (dir.path().join("model.safetensors"), dir.path().join("warm.safetensors"));
    let file = write(&path, &sized);
    // The device, its driver libraries and kernels first, from a generation over a model a few KB in size.
    write(&warm, &small);
    generate(&warm);
    let growth = brain_testutil::load_peak::host_peak_growth(|| generate(&path));
    println!("file {:.1} MB, host peak growth {:.1} MB", file as f64 / 1e6, growth as f64 / 1e6);
    // On a CPU device the weights themselves are host memory: the file's size. An f32 host copy of the head adds
    // nearly the file again.
    assert!(growth < file + file / 2, "host peak grew {growth} bytes generating from a {file}-byte checkpoint");
}
