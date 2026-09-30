// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The text pipeline runs every checkpoint the qwen3 decoder implements, as
//! downloaded: a Llama-architecture Hugging Face directory (no QK-norm, its
//! own tokenizer beside it) loads and generates, at the precision asked for.
//! The real DeepSeek-R1-Distill-Qwen-1.5B, named by its hub id, answers
//! through the chat surface with its own template and end token.

// The shared fixtures' qwen3 safetensors writer is not needed here.
#[allow(dead_code)]
mod common;

use common::{tiny_tokenizer, PROMPT_WITHIN_VOCAB};

/// `cfg` as a Llama checkpoint directory: HF tensor names, a
/// `LlamaForCausalLM` config, no QK-norm, an untied head.
fn tiny_llama_dir(tag: &str, cfg: qwen3::QwenConfig) -> std::path::PathBuf {
    let dir = common::scratch_path(tag, "d");
    write_tiny_llama(&dir, tag, cfg);
    dir
}

fn write_tiny_llama(dir: &std::path::Path, tag: &str, cfg: qwen3::QwenConfig) {
    let cfg = qwen3::QwenConfig { qk_norm: false, tie_embeddings: false, ..cfg };
    std::fs::create_dir_all(dir).unwrap();
    let names = qwen3::hf::HfNames::CAUSAL_LM;
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, numel)| {
            let data: Vec<f32> = (0..numel).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
            (names.from_brain(&name).unwrap_or_else(|| panic!("no HF name for {name}")), vec![numel as u64], data)
        })
        .collect();
    checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &serde_json::Value::Null, None).unwrap();
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "vocab_size": cfg.vocab, "hidden_size": cfg.d_model, "intermediate_size": cfg.d_ff,
        "num_hidden_layers": cfg.n_layers, "num_attention_heads": cfg.n_heads, "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim, "rms_norm_eps": cfg.rms_eps, "rope_theta": cfg.rope_theta,
        "max_position_embeddings": 2048, "tie_word_embeddings": false,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::copy(&*tiny_tokenizer(tag), dir.join("tokenizer.json")).unwrap();
}

#[test]
fn a_llama_checkpoint_directory_loads_and_generates() {
    let dir = tiny_llama_dir("text-llama-dir", qwen3::QwenConfig::tiny());
    let pipe = brain::TextGenerationPipeline::builder(dir.to_str().unwrap()).load().expect("a llama directory with its tokenizer beside it loads");
    assert_eq!(pipe.precision(), "fp32", "a tiny checkpoint defaults to fp32");
    let out = pipe.generate_with(PROMPT_WITHIN_VOCAB, brain::TextGenerationOptions::new().chat(false).max_new_tokens(3)).unwrap();
    assert_eq!((out.prompt_tokens as usize, out.completion_tokens), (PROMPT_WITHIN_VOCAB.len(), 3));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_precision_asked_for_is_the_one_built() {
    // Row widths a multiple of the int8 tier's 32-weight groups.
    let cfg = qwen3::QwenConfig { d_model: 64, n_heads: 2, n_kv_heads: 2, head_dim: 32, d_ff: 128, ..qwen3::QwenConfig::tiny() };
    let dir = tiny_llama_dir("text-llama-int8", cfg);
    let pipe = brain::TextGenerationPipeline::builder(dir.to_str().unwrap()).precision("int8").unwrap().load().expect("an int8 build");
    assert_eq!(pipe.precision(), "int8");
    let out = pipe.generate_with(PROMPT_WITHIN_VOCAB, brain::TextGenerationOptions::new().chat(false).max_new_tokens(2)).unwrap();
    assert_eq!(out.completion_tokens, 2);
    let err = brain::TextGenerationPipeline::builder(dir.to_str().unwrap()).precision("fp8").map(|_| ()).unwrap_err();
    assert!(err.to_string().contains("fp8"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A hub id names one repo: with two checkpoints of the same architecture in
/// the store, the one asked for is the one loaded, never an ambiguity over
/// (or a silent pick of) the other.
#[test]
fn a_hub_id_loads_that_repo_and_no_other() {
    let _serial = brain_testutil::env_lock();
    let root = common::scratch_path("text-store-two", "d");
    write_tiny_llama(&root.join("acme").join("one"), "text-store-one", qwen3::QwenConfig::tiny());
    write_tiny_llama(&root.join("acme").join("two"), "text-store-two", qwen3::QwenConfig::tiny());
    std::env::set_var("BRAIN_MODELS_DIR", &root);
    let loaded = brain::TextGenerationPipeline::builder("acme/two").download_policy(brain::DownloadPolicy::Offline).load();
    std::env::remove_var("BRAIN_MODELS_DIR");
    let _ = std::fs::remove_dir_all(&root);
    let pipe = loaded.expect("acme/two resolves on its own");
    assert!(pipe.identity().base.path.ends_with("acme/two"), "{:?}", pipe.identity());
}

#[test]
fn deepseek_r1_by_its_hub_id_reasons_and_stops_on_its_own_end_token() {
    let Some(_) = brain_testutil::model_dir("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").filter(|d| std::path::Path::new(d).join("config.json").exists()) else {
        brain_testutil::skip("DeepSeek-R1-Distill-Qwen-1.5B not downloaded");
        return;
    };
    let _serial = brain_testutil::env_lock();
    let pipe = brain::TextGenerationPipeline::builder("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        .download_policy(brain::DownloadPolicy::Offline)
        .capacity(2048)
        .load()
        .expect("the downloaded checkpoint resolves by its hub id");
    let chat = brain::ChatPipeline::from(pipe);
    let req = brain::ChatRequest::new(vec![brain::ChatMessage::user("What is 17 + 25? Reply with just the number.")]).max_tokens(1536).temperature(0.0);
    let reply = chat.generate(&req).expect("a chat reply");
    assert!(!reply.reasoning.is_empty(), "R1 reasons first: {reply:?}");
    assert!(reply.text.contains("42"), "{reply:?}");
    assert_eq!(reply.finish_reason.as_str(), "stop", "ends on its own end token, not the budget");
}
