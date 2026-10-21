// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A ModelOpt mixed-precision Qwen3.5/3.6 MoE release (FP8 mixers with per-tensor scales, NVFP4 experts, bf16/f32
//! elsewhere) loads from its own HuggingFace directory as stored, and runs to the logits the decoded weights give.
//!
//! Swedish Embedded AB implements serving of quantized HuggingFace mixture-of-experts releases for its clients. If
//! your team needs expertise in running ModelOpt FP8 and NVFP4 checkpoints as published then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
// hf-fixture: qwen35moe
//!
//! The fixture is random weights in the real release's layout (per-expert tensors, `weight_scale`, `weight_scale_2`,
//! `input_scale`, `hf_quant_config.json`, a multimodal `config.json`), quantized by this test's own encoders, so the
//! reference (the same model built from the decoded values at f32) is independent of brain's codecs.

use std::collections::HashMap;
use std::path::PathBuf;

use brain_testutil::hfquant::{f32_raw, fp8_per_tensor, nvfp4, write_sharded};
use gpu_core::Gpu;
use qwen35moe::config::Qwen35Config;
use qwen35moe::model::{pipelines, Qwen35};

fn cfg() -> Qwen35Config {
    Qwen35Config {
        vocab: 32,
        block_size: 24,
        n_layers: 8,
        d_model: 64,
        rms_eps: 1e-6,
        max_position_embeddings: 24,
        tie_embeddings: false,
        n_heads: 8,
        n_kv_heads: 2,
        head_dim: 8,
        attn_bias: false,
        rope_theta: 1.0e6,
        partial_rotary_factor: 0.5,
        mrope_section: [1, 1, 1],
        full_attention_interval: 4,
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 8,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        n_experts: 6,
        top_k: 2,
        moe_intermediate_size: 64,
        shared_expert_intermediate_size: 64,
        lora: None,
    }
}

fn hf_name(canon: &str) -> String {
    let top = |s: &str| format!("model.language_model.{s}");
    match canon {
        "tok.weight" => return top("embed_tokens.weight"),
        "norm.weight" => return top("norm.weight"),
        "lm_head.weight" => return canon.to_string(),
        _ => {}
    }
    let rest = canon.strip_prefix("blocks.").unwrap();
    let (l, leaf) = rest.split_once('.').unwrap();
    let leaf = match leaf {
        "ln1.weight" => "input_layernorm.weight".to_string(),
        "ln2.weight" => "post_attention_layernorm.weight".to_string(),
        "mlp.router.weight" => "mlp.gate.weight".to_string(),
        l if l.starts_with("mlp.shared_expert.") || l.starts_with("mlp.experts.") => {
            let (head, proj) = l.rsplit_once('.').and_then(|(a, _)| a.rsplit_once('.')).unwrap();
            format!("{head}.{proj}_proj.weight")
        }
        other => other.to_string(),
    };
    top(&format!("layers.{l}.{leaf}"))
}

struct Fixture {
    dir: PathBuf,
    decoded: HashMap<String, Vec<f32>>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Write the release for random weights and return it with the decoded canonical weights.
fn fixture(cfg: &Qwen35Config) -> Fixture {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!("qwen35moe-hf-nvfp4-{}-{}", std::process::id(), NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).unwrap();
    let init = qwen35moe::init::init_weights(cfg, 11);
    let (mut raws, mut decoded, mut layers) = (Vec::new(), HashMap::new(), serde_json::Map::new());
    let mut names: Vec<&String> = init.keys().collect();
    names.sort();
    for name in names {
        let w = &init[name];
        let (hf, shape) = (hf_name(name), shape_of(cfg, name));
        assert_eq!(shape.iter().product::<usize>(), w.len(), "{name}");
        let module = hf.strip_suffix(".weight").unwrap_or(&hf).to_string();
        let leaf = name.rsplit('.').nth(1).unwrap_or("");
        let is_expert = name.contains(".mlp.experts.") || name.contains(".mlp.shared_expert.");
        let is_fp8 = matches!(leaf, "in_proj_qkv" | "in_proj_z" | "out_proj" | "q_proj" | "k_proj" | "v_proj" | "o_proj") && name.ends_with(".weight");
        if is_expert || name == "lm_head.weight" {
            decoded.insert(name.clone(), nvfp4(&mut raws, &module, &shape, w));
            layers.insert(module, serde_json::json!({"quant_algo": "W4A16_NVFP4", "group_size": 16}));
        } else if is_fp8 {
            decoded.insert(name.clone(), fp8_per_tensor(&mut raws, &module, &shape, w));
            layers.insert(module, serde_json::json!({"quant_algo": "FP8"}));
        } else {
            // Norm gains are stored as `w` and applied as `1 + w`; the decoded canonical value is `w + 1`.
            let plain = name == "norm.weight" || name.ends_with(".ln1.weight") || name.ends_with(".ln2.weight") || name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight");
            let stored: Vec<f32> = if plain { w.iter().map(|v| v - 1.0).collect() } else { w.clone() };
            let hf_shape: Vec<usize> = if name.ends_with("conv1d.weight") { vec![shape[0], 1, shape[1]] } else if name.ends_with("shared_expert_gate.weight") { vec![1, shape[0]] } else { shape.clone() };
            raws.push(f32_raw(&hf, &hf_shape, &stored));
            decoded.insert(name.clone(), w.clone());
        }
    }
    // The MTP head and the vision tower are in the release and not part of the text decoder.
    raws.push(f32_raw("mtp.fc.weight", &[4], &[0.0; 4]));
    raws.push(f32_raw("model.visual.patch_embed.proj.weight", &[2], &[0.0; 2]));
    write_sharded(&dir, &raws);
    std::fs::write(
        dir.join("hf_quant_config.json"),
        serde_json::json!({"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "MIXED_PRECISION", "exclude_modules": ["mtp*"], "quantized_layers": layers}}).to_string(),
    )
    .unwrap();
    let text = serde_json::json!({
        "model_type": "qwen3_5_moe_text", "vocab_size": cfg.vocab, "hidden_size": cfg.d_model, "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads, "num_key_value_heads": cfg.n_kv_heads, "head_dim": cfg.head_dim, "rms_norm_eps": cfg.rms_eps,
        "max_position_embeddings": cfg.max_position_embeddings, "full_attention_interval": cfg.full_attention_interval,
        "linear_num_key_heads": cfg.linear_num_key_heads, "linear_num_value_heads": cfg.linear_num_value_heads,
        "linear_key_head_dim": cfg.linear_key_head_dim, "linear_value_head_dim": cfg.linear_value_head_dim, "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
        "num_experts": cfg.n_experts, "num_experts_per_tok": cfg.top_k, "moe_intermediate_size": cfg.moe_intermediate_size,
        "shared_expert_intermediate_size": cfg.shared_expert_intermediate_size, "tie_word_embeddings": false,
        "rope_parameters": {"rope_theta": cfg.rope_theta, "partial_rotary_factor": cfg.partial_rotary_factor, "mrope_section": cfg.mrope_section},
    });
    std::fs::write(dir.join("config.json"), serde_json::json!({"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe", "text_config": text, "tie_word_embeddings": false}).to_string()).unwrap();
    Fixture { dir, decoded }
}

/// The logical shape of parameter `name` of `cfg` (a 1-D parameter is `[n]`).
fn shape_of(cfg: &Qwen35Config, name: &str) -> Vec<usize> {
    let (d, ff, sff, v) = (cfg.d_model as usize, cfg.moe_intermediate_size as usize, cfg.shared_expert_intermediate_size as usize, cfg.vocab as usize);
    let leaf = name.strip_prefix("blocks.").and_then(|r| r.split_once('.')).map_or(name, |(_, l)| l);
    match leaf {
        "tok.weight" | "lm_head.weight" => vec![v, d],
        "norm.weight" | "ln1.weight" | "ln2.weight" | "mlp.shared_expert_gate.weight" => vec![d],
        "mlp.router.weight" => vec![cfg.n_experts as usize, d],
        "mlp.shared_expert.gate.weight" | "mlp.shared_expert.up.weight" => vec![sff, d],
        "mlp.shared_expert.down.weight" => vec![d, sff],
        "linear_attn.in_proj_qkv.weight" => vec![cfg.linear_conv_dim() as usize, d],
        "linear_attn.in_proj_z.weight" => vec![cfg.linear_value_dim() as usize, d],
        "linear_attn.in_proj_a.weight" | "linear_attn.in_proj_b.weight" => vec![cfg.linear_num_value_heads as usize, d],
        "linear_attn.conv1d.weight" => vec![cfg.linear_conv_dim() as usize, cfg.linear_conv_kernel_dim as usize],
        "linear_attn.A_log" | "linear_attn.dt_bias" => vec![cfg.linear_num_value_heads as usize],
        "linear_attn.norm.weight" => vec![cfg.linear_value_head_dim as usize],
        "linear_attn.out_proj.weight" => vec![d, cfg.linear_value_dim() as usize],
        "self_attn.q_proj.weight" => vec![cfg.q_proj_dim() as usize, d],
        "self_attn.k_proj.weight" | "self_attn.v_proj.weight" => vec![cfg.kv_dim() as usize, d],
        "self_attn.o_proj.weight" => vec![d, cfg.q_dim() as usize],
        "self_attn.q_norm.weight" | "self_attn.k_norm.weight" => vec![cfg.head_dim as usize],
        l if l.starts_with("mlp.experts.") => if l.ends_with("down.weight") { vec![d, ff] } else { vec![ff, d] },
        other => panic!("no shape for {other}"),
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Spec: the release loads from its directory as published (no conversion), every FP8 and NVFP4 tensor is held in its
/// stored format (the precision report and the NVFP4 tier say so), and the logits equal those of the same model built
/// from the decoded weights at f32 - prefill rows and the single-token decode path alike.
#[test]
fn a_modelopt_mixed_precision_release_loads_as_stored_and_matches_its_decoded_weights() {
    let gpu = Gpu::new(pipelines());
    if !gpu.caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("the MoE router kernel needs workgroup barriers (not the CPU JIT)");
    }
    let cfg = cfg();
    let fx = fixture(&cfg);
    let before: Vec<_> = std::fs::read_dir(&fx.dir).unwrap().map(|e| (e.unwrap().file_name(), 0)).collect();

    let src = qwen35moe::hf_source::open_dir(&fx.dir, &cfg).unwrap();
    let report = src.precision_report();
    let fmt = |n: &str| report.get(n).unwrap_or_else(|| panic!("no tensor {n}")).format.clone();
    assert_eq!(fmt("blocks.0.mlp.experts.3.gate.weight"), "E2M1 x F32/tensor x E4M3Fn/16");
    assert_eq!(fmt("blocks.0.linear_attn.in_proj_qkv.weight"), "E4M3Fn x F32/tensor");
    assert_eq!(fmt("blocks.3.self_attn.q_proj.weight"), "E4M3Fn x F32/tensor");
    assert_eq!(fmt("lm_head.weight"), "E2M1 x F32/tensor x E4M3Fn/16");
    assert_eq!(fmt("blocks.0.linear_attn.in_proj_a.weight"), "F32");
    assert!(!report.rows().iter().any(|r| r.name.contains("input_scale") || r.name.starts_with("mtp") || r.name.contains("visual")), "sidecars and foreign towers are not parameters");

    let t = cfg.block_size;
    let tokens: Vec<u32> = (0..t).map(|i| (i * 5 + 3) % cfg.vocab).collect();
    let reference = Qwen35::new_on(gpu.share(), cfg.clone(), 1, t, &fx.decoded);
    let stored = Qwen35::new_on_src(gpu.share(), cfg.clone(), 1, t, &src);
    assert!(stored.moe_nvfp4_active(), "NVFP4 experts must run from NVFP4 banks");
    let (want, got) = (reference.logits_all(&tokens), stored.logits_all(&tokens));
    assert!(got.iter().all(|v| v.is_finite()) && want.iter().any(|v| v.abs() > 1e-4), "degenerate fixture");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(cosine(&got, &want) > 0.99995, "cosine {}", cosine(&got, &want));
    // The reference multiplies f32 weights on the TF32 tensor cores; the stored path reads exact decoded values.
    assert!(max_abs_diff(&got, &want) < 5e-3 * scale.max(1.0), "max abs diff {} for logits up to {scale}", max_abs_diff(&got, &want));

    // Decode (one row) takes the gather kernels, prefill the grouped ones: same logits for the last prompt token.
    let step = |m: &Qwen35| {
        m.reset_decode_cache();
        let mut last = Vec::new();
        for tok in &tokens[..6] {
            last = m.step(*tok);
        }
        last
    };
    let (want, got) = (step(&reference), step(&stored));
    assert!(cosine(&got, &want) > 0.99995 && max_abs_diff(&got, &want) < 5e-3 * scale.max(1.0), "decode logits differ: {}", max_abs_diff(&got, &want));

    // Nothing was written next to the weights: pull and load create no weight-sized file.
    let after: Vec<_> = std::fs::read_dir(&fx.dir).unwrap().map(|e| (e.unwrap().file_name(), 0)).collect();
    assert_eq!(before, after, "loading wrote into the checkpoint directory");
}

/// Spec: the serving engine over the stored release (paged KV, recurrent state pool, the batched decode step with its
/// fused norm fronts, the recorded and replayed decode tape) greedy-decodes several sequences of different lengths to
/// exactly the tokens the single-sequence decode of the same stored model produces.
#[test]
fn the_engine_decodes_a_stored_release_as_the_single_sequence_path_does() {
    use data::rng::Rng;
    use model::paged::BlockTable;
    use model::serve::PagedDecoder;
    use qwen35moe::serve::{Engine, EngineOptions};

    let gpu = Gpu::new(pipelines());
    if !gpu.caps().workgroup_reductions {
        return brain_testutil::skip_unavailable("the MoE router kernel needs workgroup barriers (not the CPU JIT)");
    }
    let cfg = cfg();
    let fx = fixture(&cfg);
    let src = qwen35moe::hf_source::open_dir(&fx.dir, &cfg).unwrap();
    let prompts: Vec<Vec<u32>> = vec![vec![1, 5, 3], vec![2, 8], vec![4, 6, 1, 7, 3], vec![9, 2, 5, 8]];
    let steps = 5usize;
    let max_seq_len = 16u32;

    let reference = Qwen35::new_on_src(gpu.share(), cfg.clone(), 1, max_seq_len, &src);
    let want: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| {
            reference.reset_decode_cache();
            qwen35moe::sample::generate_kv(&reference, p, steps, 0.0, 0, 1.0, &[], &mut Rng::new(data::rng::random_seed()))
        })
        .collect();

    let mut engine = Engine::from_source(cfg.clone(), &src, EngineOptions::new(max_seq_len, prompts.len() as u32));
    let mut tables: Vec<BlockTable> = Vec::new();
    let mut cur: Vec<u32> = Vec::new();
    let mut got: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    for p in &prompts {
        let mut t = BlockTable::new();
        let hidden = engine.prefill(&mut t, p);
        cur.push(engine.admit_greedy(&hidden));
        tables.push(t);
    }
    for (o, &tok) in got.iter_mut().zip(&cur) {
        o.push(tok);
    }
    for _ in 1..steps {
        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
        cur = engine.forward_batched_greedy(&mut refs, &cur);
        for (o, &tok) in got.iter_mut().zip(&cur) {
            o.push(tok);
        }
    }
    assert_eq!(got, want, "batched engine decode over the stored release differs from the single-sequence decode");
}
