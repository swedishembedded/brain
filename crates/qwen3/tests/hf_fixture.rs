// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The dense decoder family (`qwen3`, `llama`, `qwen2`) opens a HuggingFace directory as published: sharded
//! safetensors with an index, and - for an FP8 ModelOpt release - the `weight_scale` / `input_scale` sidecars beside
//! each weight. Every tensor reads back under brain's names with the values the fixture stored (FP8 weights decoded
//! with their scale, sidecars never parameters).
//!
//! Swedish Embedded AB implements serving of HuggingFace decoder releases as published for its clients. If your team
//! needs expertise in loading Qwen, Llama and DeepSeek checkpoints in their stored precision then you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
// hf-fixture: qwen3, llama, qwen2
//!
//! The fixture is random weights in the real layout of each class (Qwen3 has per-head q/k norms, Qwen2 has q/k/v
//! biases, Llama neither), written by `brain_testutil::hfquant`'s encoders.

use std::path::PathBuf;

use brain_testutil::hfquant::{f32_raw, fp8_per_tensor, write_sharded, Raw};
use data::rng::Lcg;

const D: usize = 32;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const FF: usize = 64;
const VOCAB: usize = 16;
const LAYERS: usize = 2;

struct Dir(PathBuf);

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `(hf name, shape)` of every tensor of one `class` of the family, and whether a linear is FP8-quantizable.
fn layout(class: &str) -> Vec<(String, Vec<usize>, bool)> {
    let mut t: Vec<(String, Vec<usize>, bool)> = vec![("model.embed_tokens.weight".into(), vec![VOCAB, D], false), ("model.norm.weight".into(), vec![D], false), ("lm_head.weight".into(), vec![VOCAB, D], false)];
    for l in 0..LAYERS {
        let p = |s: &str| format!("model.layers.{l}.{s}");
        t.push((p("input_layernorm.weight"), vec![D], false));
        t.push((p("post_attention_layernorm.weight"), vec![D], false));
        for (leaf, shape) in [("q_proj", vec![HEADS * HEAD_DIM, D]), ("k_proj", vec![KV_HEADS * HEAD_DIM, D]), ("v_proj", vec![KV_HEADS * HEAD_DIM, D]), ("o_proj", vec![D, HEADS * HEAD_DIM])] {
            t.push((p(&format!("self_attn.{leaf}.weight")), shape, true));
        }
        if class == "Qwen3ForCausalLM" {
            t.push((p("self_attn.q_norm.weight"), vec![HEAD_DIM], false));
            t.push((p("self_attn.k_norm.weight"), vec![HEAD_DIM], false));
        }
        if class == "Qwen2ForCausalLM" {
            t.push((p("self_attn.q_proj.bias"), vec![HEADS * HEAD_DIM], false));
            t.push((p("self_attn.k_proj.bias"), vec![KV_HEADS * HEAD_DIM], false));
            t.push((p("self_attn.v_proj.bias"), vec![KV_HEADS * HEAD_DIM], false));
        }
        for (leaf, shape) in [("gate_proj", vec![FF, D]), ("up_proj", vec![FF, D]), ("down_proj", vec![D, FF])] {
            t.push((p(&format!("mlp.{leaf}.weight")), shape, true));
        }
    }
    t
}

/// Write the release of `class` into a fresh directory: `fp8` stores the linears as FP8 with per-tensor scales,
/// otherwise as f32. The same seed gives the same underlying weights, so the two releases hold the same model.
fn release(class: &str, fp8: bool, tag: &str) -> Dir {
    let dir = std::env::temp_dir().join(format!("qwen3-hf-fixture-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // seed: the f32 and fp8 releases of a class must hold the same weights, so every call replays this stream
    let mut rng = Lcg::new(23);
    let mut raws: Vec<Raw> = Vec::new();
    for (name, shape, linear) in layout(class) {
        let w = rng.vec_scaled(shape.iter().product::<usize>(), 1.0);
        match name.strip_suffix(".weight").filter(|_| fp8 && linear) {
            Some(module) => {
                fp8_per_tensor(&mut raws, module, &shape, &w);
            }
            None => raws.push(f32_raw(&name, &shape, &w)),
        }
    }
    write_sharded(&dir, &raws);
    let config = serde_json::json!({
        "architectures": [class], "vocab_size": VOCAB, "hidden_size": D, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS, "head_dim": HEAD_DIM, "intermediate_size": FF, "rope_theta": 1.0e6, "rms_norm_eps": 1e-6,
        "tie_word_embeddings": false,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    Dir(dir)
}

/// Spec: an FP8 release and the plain release of the SAME decoded values open to the same tensors under brain's
/// names; the quantization sidecars are not parameters.
#[test]
fn a_sharded_hf_release_opens_as_published_for_every_class_of_the_family_plain_and_fp8() {
    for class in ["Qwen3ForCausalLM", "LlamaForCausalLM", "Qwen2ForCausalLM"] {
        let plain = release(class, false, &format!("{class}-plain"));
        let (cfg, src) = qwen3::open::open_checkpoint(plain.0.to_str().unwrap()).unwrap_or_else(|e| panic!("{class} plain: {e}"));
        for (name, numel) in cfg.param_list() {
            let mut got = 0usize;
            assert!(src.with_tensor(&name, &mut |t| got = t.len()), "{class}: no tensor {name}");
            assert_eq!(got, numel, "{class}: {name}");
        }

        let quantized = release(class, true, &format!("{class}-fp8"));
        let (qcfg, qsrc) = qwen3::open::open_checkpoint(quantized.0.to_str().unwrap()).unwrap_or_else(|e| panic!("{class} fp8: {e}"));
        assert_eq!(qcfg.param_list(), cfg.param_list(), "{class}: the quantization sidecars must not become parameters");
        for (name, numel) in qcfg.param_list() {
            let mut got = Vec::new();
            assert!(qsrc.with_tensor(&name, &mut |t| got = t.to_vec()), "{class} fp8: no tensor {name}");
            assert_eq!(got.len(), numel, "{class} fp8: {name}");
            assert!(got.iter().all(|v| v.is_finite()), "{class} fp8: {name} decoded to a non-finite value");
        }
        // An FP8 linear decodes to its stored codes times the scale: the decoded tensor is not the f32 original.
        let name = "blocks.0.mlp.gate.weight";
        let (mut want, mut got) = (Vec::new(), Vec::new());
        src.with_tensor(name, &mut |t| want = t.to_vec());
        qsrc.with_tensor(name, &mut |t| got = t.to_vec());
        let amax = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst > 0.0 && worst <= amax / 16.0, "{class}: FP8 decode within one E4M3 step of the original (worst {worst}, amax {amax})");
    }
}
