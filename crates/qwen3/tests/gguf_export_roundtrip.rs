// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A checkpoint exported as a llama.cpp GGUF reads back through brain's own
//! GGUF importer as the same model: every decoder variant gives back its
//! configuration, its RoPE table and every tensor (bit for bit at F32 - a
//! llama's q/k rows permuted on the way out and back), and the embedded
//! tokenizer encodes as the `tokenizer.json` it came from. Every real
//! tokenizer family maps to the pre-tokenizer llama.cpp runs for it.
//!
//! Needs the tokenizers in the model store; skips one not downloaded.

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use model::rope_scaling::RopeScaling;
use qwen3::export::{export_gguf, gguf_pre_tokenizer, GgufDtype, GgufExport};
use qwen3::QwenConfig;

fn store(repo: &str) -> Option<std::path::PathBuf> {
    brain_testutil::model_dir(repo).map(std::path::PathBuf::from).filter(|d| d.join("tokenizer.json").exists())
}

#[test]
fn every_variant_round_trips_through_a_gguf() {
    // Two tokenizers: a Qwen vocabulary for the Qwen shapes, DeepSeek-Coder's
    // for the llama ones.
    let (Some(qwen), Some(coder)) = (store("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"), store("deepseek-ai/deepseek-coder-1.3b-base")) else {
        brain_testutil::skip("R1-Distill-Qwen-1.5B or deepseek-coder-1.3b-base not downloaded");
        return;
    };
    let t = QwenConfig::tiny();
    let cases = [
        ("qwen3", &qwen, QwenConfig { vocab: 151936, ..t.clone() }),
        ("qwen2", &qwen, QwenConfig { vocab: 151936, qk_norm: false, attn_bias: true, tie_embeddings: false, ..t.clone() }),
        ("llama-linear", &coder, QwenConfig { vocab: 32256, qk_norm: false, tie_embeddings: false, rope_scaling: Some(RopeScaling::Linear { factor: 4.0 }), ..t.clone() }),
        (
            "llama3",
            &coder,
            QwenConfig {
                vocab: 32256,
                qk_norm: false,
                tie_embeddings: false,
                rope_scaling: Some(RopeScaling::Llama3 { factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max_position_embeddings: 8 }),
                ..t
            },
        ),
    ];
    for (label, tok_dir, cfg) in cases {
        let weights = qwen3::init_weights(&cfg, 5);
        let path = std::env::temp_dir().join(format!("brain-gguf-export-{label}-{}.gguf", std::process::id()));
        export_gguf(&weights, &cfg, &path, &GgufExport { dtype: GgufDtype::F32, tokenizer_dir: tok_dir, name: label }).unwrap_or_else(|e| panic!("{label}: {e}"));

        let (back, src) = qwen3::open_checkpoint(path.to_str().unwrap()).unwrap_or_else(|e| panic!("{label}: {e}"));
        let same_shape = |c: &QwenConfig| (c.vocab, c.n_layers, c.d_model, c.n_heads, c.n_kv_heads, c.head_dim, c.d_ff, c.qk_norm, c.attn_bias, c.tie_embeddings);
        assert_eq!(same_shape(&back), same_shape(&cfg), "{label}");
        assert_eq!((back.rope_theta, back.rms_eps), (cfg.rope_theta, cfg.rms_eps), "{label}");
        let table = |c: &QwenConfig| c.rope_scaling.as_ref().map(|s| s.inv_freq(c.head_dim, c.rope_theta).0);
        assert_eq!(table(&back), table(&cfg), "{label}: the RoPE table reads back");
        for (name, _) in cfg.param_list() {
            let mut got = Vec::new();
            assert!(src.with_tensor(&name, &mut |x| got.extend_from_slice(x)), "{label}: {name} missing");
            assert!(got == weights[&name], "{label}: {name} differs");
        }

        let embedded = checkpoint::gguf::MmapGguf::open(path.to_str().unwrap()).unwrap().tokenizer().expect("the tokenizer is embedded");
        let from_gguf = QwenBpe::from_gguf(&embedded).unwrap();
        let from_json = QwenBpe::from_file(tok_dir.join("tokenizer.json").to_str().unwrap()).unwrap();
        let text = "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)  # 斐波那契 🙂";
        assert_eq!(from_gguf.encode(text), from_json.encode(text), "{label}: the embedded tokenizer encodes as its source");
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn every_tokenizer_family_maps_to_its_llama_cpp_pre_tokenizer() {
    let cases = [
        ("Qwen/Qwen3-0.6B", "qwen2"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "qwen2"),
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "llama-bpe"),
        ("deepseek-ai/deepseek-coder-1.3b-base", "deepseek-coder"),
        ("deepseek-ai/deepseek-coder-6.7b-base", "deepseek-coder"),
        ("deepseek-ai/deepseek-coder-7b-base-v1.5", "deepseek-llm"),
        ("deepseek-ai/deepseek-llm-7b-base", "deepseek-llm"),
        ("deepseek-ai/deepseek-math-7b-base", "deepseek-llm"),
    ];
    for (repo, want) in cases {
        let Some(dir) = store(repo) else {
            brain_testutil::skip(&format!("{repo} not downloaded"));
            continue;
        };
        let tj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer.json")).unwrap()).unwrap();
        assert_eq!(gguf_pre_tokenizer(&tj).unwrap_or_else(|e| panic!("{repo}: {e}")), want, "{repo}");
    }
    let unknown = serde_json::json!({"pre_tokenizer": {"type": "Whitespace"}});
    assert!(gguf_pre_tokenizer(&unknown).is_err());
}
