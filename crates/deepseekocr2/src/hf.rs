// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The upstream `transformers` checkpoint**: `deepseek-ai/DeepSeek-OCR-2`'s
//! own safetensors release, read in place under brain's tensor names.
//!
//! The decoder is DeepSeek-OCR's, tensor for tensor, so it is mapped by
//! `deepseek2::hf`; the SAM tower is `sam1::hf`'s. This module maps the part that
//! is new: the Qwen2 resampler (`model.qwen2_model.*`), its two query banks, the
//! linear projector and the view separator. No transpose, slice or fusion:
//! every tensor is a rename, so nothing is converted or rewritten on disk.
//!
//! The config is derived from the tensors' own shapes and then compared with
//! [`Qwen2EncoderConfig::deepseek_ocr2`]; anything else is refused. The
//! checkpoint's `config.json` cannot be trusted for the tower (its SAM
//! `downsample_channels` is stale and it names no resampler geometry at all).
//!
//! Swedish Embedded AB implements checkpoint importers that derive a model's
//! shape from its weights and refuse anything unrecognized. If your team needs
//! a published vision-language model served from its original files without a
//! conversion step, you can procure our services by emailing
//! info@swedishembedded.com.

use std::collections::{BTreeMap, HashMap};

use checkpoint::remap::{Fetch, RemapSource};
use checkpoint::weightio::WeightReader;
use deepseek2::hf::{agree, cfg_u32, dim, split_indexed, stack_len};
use deepseek2::DeepseekV2Config;
use sam1::SamViTConfig;
use serde_json::Value;

use crate::config::{DeepseekOcr2VisionConfig, Qwen2EncoderConfig};

const LABEL: &str = "deepseek-ocr-2 hf import";
const SAM_PREFIX: &str = "model.sam_model.";
const ENCODER_PREFIX: &str = "model.qwen2_model.model.model.";

fn unknown(full: &str) -> String {
    format!("{LABEL}: unrecognized tensor {full:?}")
}

/// The brain-side name of one upstream encoder-half tensor, or `None` when
/// `name` belongs to the decoder (`deepseek2::hf` owns those).
pub fn encoder_name(name: &str, sam: &SamViTConfig, enc: &Qwen2EncoderConfig) -> Result<Option<String>, String> {
    match name {
        "model.projector.layers.weight" => return Ok(Some("vision.projector.fc.weight".to_string())),
        "model.projector.layers.bias" => return Ok(Some("vision.projector.fc.bias".to_string())),
        // `seperator` is the upstream spelling; brain's is spelled correctly.
        "model.view_seperator" => return Ok(Some("vision.view_separator".to_string())),
        "model.qwen2_model.query_768.weight" => return Ok(Some("vision.query_local.weight".to_string())),
        "model.qwen2_model.query_1024.weight" => return Ok(Some("vision.query_global.weight".to_string())),
        _ => {}
    }
    if let Some(leaf) = name.strip_prefix(SAM_PREFIX) {
        return sam1::hf::brain_name(leaf, sam, sam1::hf::Spelling::DeepseekOcr).map(Some).map_err(|e| format!("{LABEL}: {name}: {e}"));
    }
    let Some(rest) = name.strip_prefix(ENCODER_PREFIX) else { return Ok(None) };
    if rest == "norm.weight" {
        return Ok(Some("vision.encoder.norm.weight".to_string()));
    }
    let rest = rest.strip_prefix("layers.").ok_or_else(|| unknown(name))?;
    let (l, leaf) = split_indexed(rest, name, enc.n_layers, "resampler layer").map_err(|e| format!("{LABEL}: {e}"))?;
    let brain = match leaf {
        "input_layernorm.weight" => "norm1.weight",
        "post_attention_layernorm.weight" => "norm2.weight",
        "self_attn.q_proj.weight" => "attn.q.weight",
        "self_attn.q_proj.bias" => "attn.q.bias",
        "self_attn.k_proj.weight" => "attn.k.weight",
        "self_attn.k_proj.bias" => "attn.k.bias",
        "self_attn.v_proj.weight" => "attn.v.weight",
        "self_attn.v_proj.bias" => "attn.v.bias",
        "self_attn.o_proj.weight" => "attn.out.weight",
        "mlp.gate_proj.weight" => "mlp.gate.weight",
        "mlp.up_proj.weight" => "mlp.up.weight",
        "mlp.down_proj.weight" => "mlp.down.weight",
        _ => return Err(unknown(name)),
    };
    Ok(Some(format!("vision.encoder.blocks.{l}.{brain}")))
}

/// The vision half's fetch plan, keyed by brain-side name: SAM, the resampler,
/// the query banks, the projector and the separator. Every upstream tensor is
/// classified (an unrecognized one stops the import) and every tensor of the
/// vision manifest must be produced.
pub fn encoder_plan(names: &[String], vision: &DeepseekOcr2VisionConfig) -> Result<HashMap<String, Fetch>, String> {
    let mut plan = HashMap::new();
    for src in names {
        if let Some(brain) = encoder_name(src, &vision.sam, &vision.encoder)? {
            if plan.insert(brain.clone(), Fetch::Whole(src.clone())).is_some() {
                return Err(format!("{LABEL}: two source tensors map to {brain}"));
            }
        }
    }
    let want = vision.sam.param_list().into_iter().chain(vision.param_list());
    let missing: Vec<String> = want.map(|(n, _)| n).filter(|n| !plan.contains_key(n)).collect();
    if !missing.is_empty() {
        return Err(format!("{LABEL}: this checkpoint produces no vision tensor for {} parameter(s): {missing:?}", missing.len()));
    }
    Ok(plan)
}

/// The vision half as a streaming source under brain's names, borrowing `reader`.
pub fn vision_source<'a>(reader: &'a WeightReader, vision: &DeepseekOcr2VisionConfig) -> Result<RemapSource<'a>, String> {
    let names: Vec<String> = reader.names().map(str::to_string).collect();
    let src = RemapSource::new(reader, encoder_plan(&names, vision)?);
    let want: Vec<(String, usize)> = vision.sam.param_list().into_iter().chain(vision.param_list()).collect();
    src.validate(&want).map_err(|e| format!("{LABEL}: {e}"))?;
    Ok(src)
}

/// The decoder half as a streaming source under brain's names, borrowing `reader`.
pub fn decoder_source<'a>(reader: &'a WeightReader, dec: &DeepseekV2Config) -> Result<RemapSource<'a>, String> {
    let names: Vec<String> = reader.names().map(str::to_string).collect();
    let src = RemapSource::new(reader, deepseek2::hf::plan(&names, dec).map_err(|e| format!("{LABEL}: {e}"))?);
    src.validate(&dec.param_list()).map_err(|e| format!("{LABEL}: {e}"))?;
    Ok(src)
}

/// The checkpoint's two configs, derived from its own tensor shapes (plus the
/// handful of numbers only `config.json` carries) and then refused unless they
/// are the documented preset: deriving alone would silently serve a differently
/// shaped checkpoint, and a hardcoded preset alone would not notice one.
pub fn config_from_shapes(shapes: &BTreeMap<String, Vec<usize>>, config: &Value, block_size: u32) -> Result<(DeepseekOcr2VisionConfig, DeepseekV2Config), String> {
    let decoder = deepseek2::hf::decoder_from_shapes(shapes, config, block_size).map_err(|e| format!("{LABEL}: {e}"))?;
    let sam_gguf = sam1::hf::config_from_shapes(
        shapes,
        SAM_PREFIX,
        cfg_u32(config, &["vision_config", "width", "sam_vit_b", "heads"]).map_err(|e| format!("{LABEL}: {e}"))?,
        cfg_u32(config, &["vision_config", "width", "sam_vit_b", "layers"]).map_err(|e| format!("{LABEL}: {e}"))?,
    )
    .map_err(|e| format!("{LABEL}: {e}"))?;

    let want = Qwen2EncoderConfig::deepseek_ocr2();
    let layer = |leaf: &str, axis: usize| dim(shapes, &format!("{ENCODER_PREFIX}layers.0.{leaf}"), axis).map_err(|e| format!("{LABEL}: {e}"));
    let d_model = layer("self_attn.q_proj.weight", 0)?;
    agree("resampler width", d_model, cfg_u32(config, &["vision_config", "width", "qwen2-0-5b", "dim"]).map_err(|e| format!("{LABEL}: {e}"))?).map_err(|e| format!("{LABEL}: {e}"))?;
    // The head dimension is the one number no tensor shape carries (a query
    // projection is `[heads * head_dim, width]` at any split); the preset's is
    // used and the comparison below catches a checkpoint that disagrees.
    let head_dim = want.head_dim();
    let kv_dim = layer("self_attn.k_proj.weight", 0)?;
    let encoder = Qwen2EncoderConfig {
        d_model,
        n_layers: stack_len(shapes, &format!("{ENCODER_PREFIX}layers.")),
        n_heads: d_model / head_dim,
        n_kv_heads: kv_dim / head_dim,
        ffn_hidden: layer("mlp.gate_proj.weight", 0)?,
        rms_eps: want.rms_eps,
        rope_theta: want.rope_theta,
        n_query_local: dim(shapes, "model.qwen2_model.query_768.weight", 0).map_err(|e| format!("{LABEL}: {e}"))?,
        n_query_global: dim(shapes, "model.qwen2_model.query_1024.weight", 0).map_err(|e| format!("{LABEL}: {e}"))?,
    };
    if encoder != want {
        return Err(format!("{LABEL}: this checkpoint's resampler is not DeepSeek-OCR-2's documented preset (derived {encoder:?}, want {want:?})"));
    }
    let decoder_hidden = dim(shapes, "model.projector.layers.weight", 0).map_err(|e| format!("{LABEL}: {e}"))?;
    agree("projector output width", decoder_hidden, decoder.shape.d_model).map_err(|e| format!("{LABEL}: {e}"))?;
    let vision = DeepseekOcr2VisionConfig { sam: SamViTConfig::from(&sam_gguf), encoder, decoder_hidden, lora: None };
    vision.check();
    let want_dec = DeepseekV2Config::deepseek_ocr(block_size);
    if decoder != want_dec {
        return Err(format!("{LABEL}: this checkpoint's decoder is not the DeepSeek-OCR family's documented preset"));
    }
    Ok((vision, decoder))
}

#[cfg(test)]
mod tests {
    use super::*;

    // hf-fixture: deepseekocr2
    /// The DeepSeek-OCR-2 tensor header, written as an EMITTER (the matcher
    /// above is the other side of the contract) from the repo's own
    /// `model.safetensors.index.json` name patterns: 2707 tensors.
    fn upstream_shapes() -> BTreeMap<String, Vec<usize>> {
        let mut m: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut put = |n: String, s: Vec<usize>| {
            assert!(m.insert(n.clone(), s).is_none(), "duplicate upstream name {n}");
        };
        put("model.sam_model.pos_embed".into(), vec![1, 64, 64, 768]);
        put("model.sam_model.patch_embed.proj.weight".into(), vec![768, 3, 16, 16]);
        put("model.sam_model.patch_embed.proj.bias".into(), vec![768]);
        for l in 0..12 {
            let b = |leaf: &str| format!("model.sam_model.blocks.{l}.{leaf}");
            let rows = if [2, 5, 8, 11].contains(&l) { 127 } else { 27 };
            put(b("norm1.weight"), vec![768]);
            put(b("norm1.bias"), vec![768]);
            put(b("attn.rel_pos_h"), vec![rows, 64]);
            put(b("attn.rel_pos_w"), vec![rows, 64]);
            put(b("attn.qkv.weight"), vec![2304, 768]);
            put(b("attn.qkv.bias"), vec![2304]);
            put(b("attn.proj.weight"), vec![768, 768]);
            put(b("attn.proj.bias"), vec![768]);
            put(b("norm2.weight"), vec![768]);
            put(b("norm2.bias"), vec![768]);
            put(b("mlp.lin1.weight"), vec![3072, 768]);
            put(b("mlp.lin1.bias"), vec![3072]);
            put(b("mlp.lin2.weight"), vec![768, 3072]);
            put(b("mlp.lin2.bias"), vec![768]);
        }
        put("model.sam_model.neck.0.weight".into(), vec![256, 768, 1, 1]);
        put("model.sam_model.neck.1.weight".into(), vec![256]);
        put("model.sam_model.neck.1.bias".into(), vec![256]);
        put("model.sam_model.neck.2.weight".into(), vec![256, 256, 3, 3]);
        put("model.sam_model.neck.3.weight".into(), vec![256]);
        put("model.sam_model.neck.3.bias".into(), vec![256]);
        put("model.sam_model.net_2.weight".into(), vec![512, 256, 3, 3]);
        put("model.sam_model.net_3.weight".into(), vec![896, 512, 3, 3]);
        for l in 0..24 {
            let b = |leaf: &str| format!("{ENCODER_PREFIX}layers.{l}.{leaf}");
            put(b("input_layernorm.weight"), vec![896]);
            put(b("post_attention_layernorm.weight"), vec![896]);
            put(b("self_attn.q_proj.weight"), vec![896, 896]);
            put(b("self_attn.q_proj.bias"), vec![896]);
            put(b("self_attn.k_proj.weight"), vec![128, 896]);
            put(b("self_attn.k_proj.bias"), vec![128]);
            put(b("self_attn.v_proj.weight"), vec![128, 896]);
            put(b("self_attn.v_proj.bias"), vec![128]);
            put(b("self_attn.o_proj.weight"), vec![896, 896]);
            put(b("mlp.gate_proj.weight"), vec![4864, 896]);
            put(b("mlp.up_proj.weight"), vec![4864, 896]);
            put(b("mlp.down_proj.weight"), vec![896, 4864]);
        }
        put(format!("{ENCODER_PREFIX}norm.weight"), vec![896]);
        put("model.qwen2_model.query_768.weight".into(), vec![144, 896]);
        put("model.qwen2_model.query_1024.weight".into(), vec![256, 896]);
        put("model.projector.layers.weight".into(), vec![1280, 896]);
        put("model.projector.layers.bias".into(), vec![1280]);
        put("model.view_seperator".into(), vec![1280]);

        put("model.embed_tokens.weight".into(), vec![129280, 1280]);
        for l in 0..12 {
            let b = |leaf: &str| format!("model.layers.{l}.{leaf}");
            put(b("input_layernorm.weight"), vec![1280]);
            put(b("post_attention_layernorm.weight"), vec![1280]);
            for p in ["q", "k", "v", "o"] {
                put(b(&format!("self_attn.{p}_proj.weight")), vec![1280, 1280]);
            }
            if l == 0 {
                put(b("mlp.gate_proj.weight"), vec![6848, 1280]);
                put(b("mlp.up_proj.weight"), vec![6848, 1280]);
                put(b("mlp.down_proj.weight"), vec![1280, 6848]);
            } else {
                put(b("mlp.gate.weight"), vec![64, 1280]);
                for e in 0..64 {
                    put(b(&format!("mlp.experts.{e}.gate_proj.weight")), vec![896, 1280]);
                    put(b(&format!("mlp.experts.{e}.up_proj.weight")), vec![896, 1280]);
                    put(b(&format!("mlp.experts.{e}.down_proj.weight")), vec![1280, 896]);
                }
                put(b("mlp.shared_experts.gate_proj.weight"), vec![1792, 1280]);
                put(b("mlp.shared_experts.up_proj.weight"), vec![1792, 1280]);
                put(b("mlp.shared_experts.down_proj.weight"), vec![1280, 1792]);
            }
        }
        put("model.norm.weight".into(), vec![1280]);
        put("lm_head.weight".into(), vec![129280, 1280]);
        m
    }

    fn upstream_config() -> Value {
        serde_json::json!({
            "architectures": ["DeepseekOCR2ForCausalLM"],
            "language_config": {
                "vocab_size": 129280, "hidden_size": 1280, "intermediate_size": 6848, "moe_intermediate_size": 896,
                "num_hidden_layers": 12, "num_attention_heads": 10, "num_key_value_heads": 10, "n_routed_experts": 64,
                "num_experts_per_tok": 6, "n_shared_experts": 2, "first_k_dense_replace": 1,
                "max_position_embeddings": 8192, "n_group": 1, "topk_group": 1
            },
            "vision_config": { "width": {
                "qwen2-0-5b": { "dim": 896 },
                "sam_vit_b": { "downsample_channels": [512, 1024], "global_attn_indexes": [2, 5, 8, 11], "heads": 12, "layers": 12, "width": 768 }
            } }
        })
    }

    #[test]
    fn the_transcribed_header_has_the_count_the_real_index_publishes() {
        assert_eq!(upstream_shapes().len(), 2707, "model.safetensors.index.json lists 2707 tensors");
    }

    /// The config is DERIVED from the checkpoint's shapes and equals the preset;
    /// the stale `downsample_channels` in `config.json` is not what decides the
    /// compressor's width (the 896 tensor shape does).
    #[test]
    fn the_config_is_derived_from_the_checkpoints_own_shapes() {
        let (vision, decoder) = config_from_shapes(&upstream_shapes(), &upstream_config(), 1).expect("the real checkpoint is the preset");
        assert_eq!(vision.encoder, Qwen2EncoderConfig::deepseek_ocr2());
        assert_eq!(vision.sam.compress_out, 896);
        assert_eq!(decoder, DeepseekV2Config::deepseek_ocr(1));
    }

    /// Every upstream tensor lands on exactly one brain-side name, both halves
    /// are covered at the declared sizes, and an incomplete checkpoint is refused.
    #[test]
    fn plans_cover_both_halves_exactly_and_refuse_an_incomplete_checkpoint() {
        let shapes = upstream_shapes();
        let (vision, decoder) = config_from_shapes(&shapes, &upstream_config(), 1).unwrap();
        let names: Vec<String> = shapes.keys().cloned().collect();
        let enc = encoder_plan(&names, &vision).expect("encoder plan");
        let dec = deepseek2::hf::plan(&names, &decoder).expect("decoder plan");
        let n_experts = 11 * 3 * 64;
        assert_eq!(enc.len(), vision.sam.param_list().len() + vision.param_list().len());
        assert_eq!(enc.len() + dec.len() + n_experts - 11 * 3, names.len(), "every source tensor is claimed once (a bank claims 64)");

        let short: Vec<String> = names.iter().filter(|n| !n.starts_with(&format!("{ENCODER_PREFIX}layers.7."))).cloned().collect();
        let e = encoder_plan(&short, &vision).unwrap_err();
        assert!(e.contains("vision.encoder.blocks.7."), "{e}");

        let mut extra = names.clone();
        extra.push(format!("{ENCODER_PREFIX}layers.0.self_attn.q_norm.weight"));
        assert!(encoder_plan(&extra, &vision).is_err(), "an unrecognized tensor must stop the import");
    }

    #[test]
    fn a_checkpoint_that_is_not_the_preset_is_refused() {
        let mut shapes = upstream_shapes();
        shapes.insert("model.qwen2_model.query_1024.weight".into(), vec![400, 896]);
        let e = config_from_shapes(&shapes, &upstream_config(), 1).unwrap_err();
        assert!(e.contains("documented preset"), "{e}");
    }
}
