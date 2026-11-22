// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded loading of large language models onto their cards for its clients.
// If your team needs expertise in serving mixture-of-experts models within a fixed host memory budget then you can
// procure our services by sending an email to info@swedishembedded.com.

//! `generate` streams the checkpoint onto the device one tensor at a time: the host peak of a load stays within the
//! file (the weights themselves, on a CPU device), where reading it whole as f32 and cloning the map took about three
//! times the file. The tokens are the eagerly built model's.

use std::sync::Arc;

use capability::{Invocation, Provider, Registry};
use qwen35moe::caps::{Qwen35Provider, MODEL};
use qwen35moe::config::Qwen35Config;
use serde_json::json;

const PROMPT: [u32; 4] = [3, 17, 250, 9];

/// Tens of MB of f32 weights, mostly routed experts as in the real model.
fn sized() -> Qwen35Config {
    Qwen35Config { vocab: 4096, n_layers: 4, d_model: 256, n_experts: 8, moe_intermediate_size: 512, shared_expert_intermediate_size: 256, ..Qwen35Config::tiny() }
}

/// Every parameter of `cfg` as f32, written one tensor at a time: the process never holds a model.
fn write_checkpoint(path: &std::path::Path, cfg: &Qwen35Config) -> u64 {
    let mut rng = data::rng::Lcg::random();
    let card = checkpoint::st::ModelCard::new("brain/qwen35moe", "qwen35moe");
    // Norm gains near one, everything else small: any weights decode, this checks memory and plumbing.
    brain_testutil::load_peak::write_f32_checkpoint(path, &cfg.param_list(), &cfg.to_json(), Some(&card), &mut |name, v| {
        v.iter_mut().for_each(|x| *x = if name.contains("norm") { 1.0 } else { rng.scaled(0.02) })
    })
}

fn generate(path: &std::path::Path, precision: &str) -> Vec<u64> {
    let mut reg = Registry::new();
    reg.register(Arc::new(Qwen35Provider::new(path.to_str().unwrap(), None)) as Arc<dyn Provider>);
    let prompt = PROMPT.iter().map(u32::to_string).collect::<Vec<_>>().join(" ");
    let inv = Invocation::new().set("prompt", json!(prompt)).set("max_new", json!(5)).set("temp", json!(0.0)).set("precision", json!(precision));
    let out = reg.run(MODEL, "generate", inv, &mut |_| {}).unwrap_or_else(|e| panic!("generate at {precision}: {e}"));
    out.outputs["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect()
}

/// The streamed build decodes what the model built from the whole map decodes.
#[test]
fn a_streamed_load_generates_what_the_eager_model_generates() {
    let dir = brain_testutil::Scratch::dir("qwen35moe-streamed-tokens");
    let cfg = Qwen35Config { vocab: 300, ..Qwen35Config::tiny() };
    let path = dir.path().join("model.safetensors");
    write_checkpoint(&path, &cfg);
    let init = checkpoint::load(path.to_str().unwrap()).into_by_role("");
    let eager = qwen35moe::model::Qwen35::new(cfg.clone(), 1, 64, &init);
    let want: Vec<u64> = qwen35moe::sample::generate_kv(&eager, &PROMPT, 5, 0.0, 0, 1.0, &[], &mut data::rng::Rng::new(0)).into_iter().map(u64::from).collect();
    assert_eq!(generate(&path, "fp32"), want);
}

#[test]
fn loading_a_checkpoint_holds_no_whole_model_host_copy() {
    brain_testutil::run_alone(module_path!(), "host_peak_of_a_streamed_load");
}

#[test]
#[ignore = "run alone by loading_a_checkpoint_holds_no_whole_model_host_copy"]
fn host_peak_of_a_streamed_load() {
    let dir = brain_testutil::Scratch::dir("qwen35moe-streamed-peak");
    let path = dir.path().join("model.safetensors");
    let file = write_checkpoint(&path, &sized());
    // The device, its driver libraries and kernels first, from a load of the same layout a few KB in size.
    let warm = dir.path().join("warm.safetensors");
    write_checkpoint(&warm, &Qwen35Config { vocab: 300, ..Qwen35Config::tiny() });
    generate(&warm, "fp32");
    let growth = brain_testutil::load_peak::host_peak_growth(|| drop(generate(&path, "fp32")));
    println!("file {:.1} MB, host peak growth {:.1} MB", file as f64 / 1e6, growth as f64 / 1e6);
    // On a CPU device the f32 weights themselves are host memory: the file's size. A whole-model f32 read and its
    // clone add twice the file on top.
    assert!(growth < file + file / 2, "host peak grew {growth} bytes loading a {file}-byte checkpoint");
}
