// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A llama-architecture GGUF, converted by llama.cpp's own
//! `convert_hf_to_gguf.py`, serves the same model as the checkpoint it was
//! converted from: deepseek-coder-1.3b-instruct (llama, linear RoPE x4)
//! read from its Hugging Face directory and from its f16 and Q8_0 GGUFs
//! gives the same config and, position by position, the same logits within
//! the conversion's own rounding.
//!
//! Needs the checkpoint in the model store and `DEEPSEEK_CODER_GGUF_DIR`
//! naming a directory holding `coder-1.3b-f16.gguf` and
//! `coder-1.3b-q8_0.gguf`; skips otherwise.

use qwen3::model::Qwen;

fn logits(path: &str, ids: &[u32]) -> (qwen3::QwenConfig, Vec<f32>) {
    let (cfg, src) = qwen3::open_checkpoint(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let shard = qwen3::Shard::whole(cfg.n_layers as usize);
    let model = Qwen::new_shard_dt(cfg.clone(), 1, ids.len() as u32, &*src, shard, qwen3::Dtype::F32);
    (cfg, model.logits_all(ids))
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

#[test]
fn a_llama_gguf_serves_the_model_it_was_converted_from() {
    let (Some(dir), Ok(gguf_dir)) = (brain_testutil::model_dir("deepseek-ai/deepseek-coder-1.3b-instruct"), std::env::var("DEEPSEEK_CODER_GGUF_DIR")) else {
        brain_testutil::skip("deepseek-coder-1.3b-instruct or DEEPSEEK_CODER_GGUF_DIR missing");
        return;
    };
    let tok = data::qwen_tokenizer::QwenBpe::from_file(&format!("{dir}/tokenizer.json")).unwrap();
    let ids = data::tokenizer::Tokenizer::encode(&tok, "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n");
    let (hf_cfg, hf) = logits(&dir, &ids);
    let vocab = hf_cfg.vocab as usize;
    for (file, min_cos) in [("coder-1.3b-f16.gguf", 0.9999), ("coder-1.3b-q8_0.gguf", 0.999)] {
        let path = format!("{gguf_dir}/{file}");
        let (cfg, got) = logits(&path, &ids);
        assert_eq!((cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_ff), (hf_cfg.n_layers, hf_cfg.d_model, hf_cfg.n_heads, hf_cfg.n_kv_heads, hf_cfg.head_dim, hf_cfg.d_ff), "{file}");
        assert_eq!((cfg.qk_norm, cfg.attn_bias, cfg.tie_embeddings, cfg.rope_theta, cfg.rope_scaling.clone()), (hf_cfg.qk_norm, hf_cfg.attn_bias, hf_cfg.tie_embeddings, hf_cfg.rope_theta, hf_cfg.rope_scaling.clone()), "{file}");
        for p in 0..ids.len() {
            let (a, b) = (&hf[p * vocab..(p + 1) * vocab], &got[p * vocab..(p + 1) * vocab]);
            let c = cosine(a, b);
            assert!(c >= min_cos, "{file}: position {p} logits cosine {c:.6} < {min_cos}");
            assert_eq!(argmax(a), argmax(b), "{file}: position {p} predicts a different token");
        }
        eprintln!("{file}: {} positions agree with the HF checkpoint", ids.len());
    }
}
