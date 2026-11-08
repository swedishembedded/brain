// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one very large model from a single resident copy of its weights
// for its clients. If your team needs expertise in multi-tenant inference of
// fine-tuned language models then you can procure our services by sending an
// email to info@swedishembedded.com.

#![cfg(feature = "text")]

//! `TextGenerationPipeline` over a Ripple directory: `.adapter(path)` serves a trained adapter, and attaching,
//! switching and detaching one never reloads the base (CPU backend).

use brain::{Device, TextGenerationOptions, TextGenerationPipeline};
use data::rng::Rng;
use model::adapter::device::LoraCfg;
use model::Model;
use ricursive::{open_safetensors, RicursiveConfig, Ripple, TrainConfig, TrainOpts};

const BASE_ID: &str = "local/tiny-ripple";

fn config() -> RicursiveConfig {
    RicursiveConfig::from_json(
        r#"{"model_type":"ricursive","vocab_size":264,"hidden_size":64,"intermediate_size":96,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,
            "head_dim":16,"sliding_window":8,"use_valres":true,"use_gated_attention":true,"ffn_clamp_threshold":20.0,"embed_scale":8.0,"max_position_embeddings":512,
            "layer_types":["sliding_attention","sliding_attention","full_attention"]}"#,
    )
    .unwrap()
}

/// A pulled directory: config, random weights, a byte-level tokenizer.
fn pulled_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-sdk-ripple-{tag}-{}", std::process::id())).join("tiny-ripple");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = config();
    let mut rng = Rng::new(31); // seed: any weights serve, the test reads the adapter, not the model
    let order: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    let tensors: Vec<brain_testutil::hfquant::Raw> = ricursive::manifest(&cfg, &order)
        .into_iter()
        .map(|(name, dims)| {
            let n: usize = dims.iter().product();
            let data: Vec<f32> = if name.contains("norm") { vec![1.0; n] } else if name.ends_with("value_res_lambda") { vec![0.0; n] } else { (0..n).map(|_| rng.next_gaussian() as f32 * 1.7 / (dims.last().copied().unwrap_or(1) as f32).sqrt()).collect() };
            brain_testutil::hfquant::f32_raw(&name, &dims, &data)
        })
        .collect();
    brain_testutil::hfquant::write_safetensors(&dir.join("model.safetensors"), &tensors);
    let mut json: serde_json::Value = serde_json::from_str(&cfg.to_json().to_string()).unwrap();
    json["architectures"] = serde_json::json!(["RicursiveForCausalLM"]);
    json["max_position_embeddings"] = serde_json::json!(512);
    std::fs::write(dir.join("config.json"), json.to_string()).unwrap();
    let vocab: serde_json::Map<String, serde_json::Value> = data::bpe::bytes_to_unicode().iter().enumerate().map(|(i, c)| (c.to_string(), serde_json::json!(i))).collect();
    std::fs::write(dir.join("tokenizer.json"), serde_json::json!({"model": {"vocab": vocab, "merges": []}, "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}}).to_string()).unwrap();
    dir
}

/// A trained-shaped adapter for the directory (random factors) and the model that computes its forward.
fn adapter(dir: &std::path::Path, card: &str, seed: u64) -> (std::path::PathBuf, Ripple) {
    let cfg = config();
    let layers: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    let base = open_safetensors(dir.join("model.safetensors").to_str().unwrap(), &cfg, &layers).unwrap();
    let train = TrainConfig::new(cfg, 64).with_lora(LoraCfg::new(4, 8.0, ricursive::train::default_targets()));
    let init = ricursive::train::AdapterInit::fresh(&train, &base, 5); // seed: overwritten just below
    let trained = Ripple::build(train, 1, 64, &init, &TrainOpts::default(), true).unwrap();
    model::conformance_grad::randomize_adapters(&trained, seed);
    let file = dir.parent().unwrap().join(format!("{seed}.safetensors"));
    trained.save_adapter(file.to_str().unwrap(), card, BASE_ID, None, None).unwrap();
    (file, trained)
}

fn greedy(trained: &Ripple, dir: &std::path::Path, prompt: &str, new: usize) -> String {
    use data::Tokenizer;
    let tok = data::qwen_tokenizer::QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).unwrap();
    let mut ids = tok.encode(prompt);
    let (start, vocab) = (ids.len(), trained.model_config().vocab_size);
    for _ in 0..new {
        let logits = trained.logits_all(&ids).unwrap();
        let last = &logits[(ids.len() - 1) * vocab..ids.len() * vocab];
        ids.push(last.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0 as u32);
    }
    tok.decode(&ids[start..])
}

fn options() -> TextGenerationOptions {
    TextGenerationOptions::new().chat(false).max_new_tokens(6).temperature(0.0).seed(1)
}

/// Spec: a pipeline over a Ripple directory serves the adapter it is given exactly as the training model's forward
/// computes it, switches to another adapter and back to the base without reloading, and names what it serves.
#[test]
fn a_ripple_pipeline_serves_an_adapter_and_switches_without_reloading() {
    let dir = pulled_dir("pipeline");
    let (file_a, train_a) = adapter(&dir, "local/tiny-ripple:lab:alpha:v1", 9);
    let (file_b, train_b) = adapter(&dir, "local/tiny-ripple:lab:beta:v1", 10);
    let prompt = "the vault code is 4817";
    let say = |p: &TextGenerationPipeline| p.generate_with(prompt, options()).unwrap();

    let mut pipe = TextGenerationPipeline::builder(dir.to_str().unwrap()).adapter(file_a.to_str().unwrap()).capacity(256).device(Device::parse("cpu").unwrap()).load().expect("a Ripple directory with its adapter serves");
    assert_eq!(pipe.identity().adapter.as_ref().and_then(|a| a.id.as_deref()), Some("local/tiny-ripple:lab:alpha:v1"));
    let with_a = say(&pipe);
    assert_eq!(with_a.text, greedy(&train_a, &dir, prompt, 6), "the adapter answers what the training forward does");
    assert_eq!(with_a.completion_tokens, 6);

    pipe.attach_adapter(file_b.to_str().unwrap()).unwrap();
    assert_eq!(say(&pipe).text, greedy(&train_b, &dir, prompt, 6), "switched to the other adapter, no reload");
    assert!(pipe.detach_adapter().unwrap());
    assert!(pipe.identity().adapter.is_none());
    let base = say(&pipe);
    assert_ne!(base.text, with_a.text, "the adapter changed the answer");
    assert!(!pipe.detach_adapter().unwrap(), "nothing left to detach");
    let plain = TextGenerationPipeline::builder(dir.to_str().unwrap()).capacity(256).device(Device::parse("cpu").unwrap()).load().unwrap();
    assert_eq!(say(&plain).text, base.text, "detached, exactly the base");
    assert!(pipe.attach_adapter("/nonexistent/adapter.safetensors").is_err(), "a refused adapter is an error and changes nothing");
    assert_eq!(say(&pipe).text, base.text);
    let precision = TextGenerationPipeline::builder(dir.to_str().unwrap()).precision("int8").unwrap().device(Device::parse("cpu").unwrap()).load().expect_err("a weight format is not a pipeline knob for Ripple");
    assert!(precision.to_string().contains("int8"), "{precision}");
    std::fs::remove_dir_all(dir.parent().unwrap()).ok();
}
