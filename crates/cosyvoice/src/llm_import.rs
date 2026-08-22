// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import `llm.pt` (the real `FunAudioLLM/CosyVoice2-0.5B` checkpoint - a
//! `torch.save`'d `Qwen2LM.state_dict()`) into the backbone weights
//! `qwen3::Qwen` hosts, plus CosyVoice's own bolted-on tables.
//!
//! **`llm.pt` is self-contained** - it carries a FULL, independently
//! fine-tuned copy of the Qwen2.5-0.5B backbone under `llm.model.model.*` (24
//! layers + `embed_tokens`/`norm`, 295 tensors total), not a delta against
//! `CosyVoice-BlankEN`'s shipped `model.safetensors`. Verified empirically,
//! not assumed - checkpoint headers are free architecture docs, and
//! `llm.model.model.embed_tokens.weight` differs
//! from `CosyVoice-BlankEN/model.safetensors`'s `model.embed_tokens.weight`
//! by up to 0.265 max-abs on the first row - `CosyVoice-BlankEN` is only the
//! *pretrained starting point* `Qwen2Encoder.__init__`'s
//! `Qwen2ForCausalLM.from_pretrained` loads before the reference pipeline's
//! strict `model.load(llm.pt, ...)` overwrites every parameter with the
//! trained one. `CosyVoice-BlankEN` therefore supplies only the tokenizer +
//! architecture identity (`config.json`), never weights, in this crate.
//!
//! `llm.model.lm_head.weight` is bit-exactly equal to
//! `llm.model.model.embed_tokens.weight` (tied, `tie_word_embeddings: true`
//! in `CosyVoice-BlankEN/config.json`), so it is dropped rather than
//! imported - the same "tied -> drop" convention
//! `qwen3::import::hf_to_brain` uses for a released Qwen checkpoint. CosyVoice
//! itself never reads this backbone `lm_head` at all: `Qwen2Encoder.
//! forward_one_step` returns `hidden_states`, and `Qwen2LM`/`CosyVoice3LM`
//! project those through their OWN `llm_decoder`, never through the Qwen
//! backbone's 151936-wide text head.
//!
//! **CosyVoice 3's `llm.pt` has no `llm_embedding.weight` and no
//! `llm_decoder.bias`** - both real, verified-not-assumed absences (see
//! `crate::config`'s module doc for why): `CosyVoice3LM` has no
//! `llm_embedding` table at all, and its `llm_decoder = Linear(896, 6761,
//! bias=False)`. [`import_llm_pt`] branches on
//! `cfg.special_token_source`/`cfg.llm_decoder_has_bias` to require or
//! forbid each accordingly, rather than silently defaulting either to zero.

use std::collections::HashMap;

use crate::config::{CosyVoiceLmConfig, SpecialTokenSource};

/// Backbone weights (`qwen3::QwenConfig::param_list()`-keyed, ready for
/// `qwen3::Qwen::from_tensors_decode`) plus CosyVoice's own bolted-on tables.
pub struct LmWeights {
    pub backbone: HashMap<String, Vec<f32>>,
    /// `[2, d]`: row 0 = `sos`, row 1 = `task_id`. `None` for `CosyVoice3LM`
    /// (`SpecialTokenSource::SpeechEmbedding`), which has no such table.
    pub llm_embedding: Option<Vec<f32>>,
    /// `[speech_vocab, d]`.
    pub speech_embedding: Vec<f32>,
    /// `[speech_vocab, d]`.
    pub llm_decoder_w: Vec<f32>,
    /// `[speech_vocab]`. `None` when `cfg.llm_decoder_has_bias` is `false`
    /// (`CosyVoice3LM`'s `llm_decoder` carries no bias).
    pub llm_decoder_b: Option<Vec<f32>>,
}

/// Map one `llm.pt` tensor name to its `qwen3` backbone parameter name, or
/// `None` if it is not a per-layer/embedding/norm backbone tensor (a
/// bolted-on table or the tied `lm_head.weight` duplicate, both handled by
/// [`import_llm_pt`] directly).
fn backbone_name(name: &str) -> Option<String> {
    if name == "llm.model.model.embed_tokens.weight" {
        return Some("tok.weight".to_string());
    }
    if name == "llm.model.model.norm.weight" {
        return Some("norm.weight".to_string());
    }
    let rest = name.strip_prefix("llm.model.model.layers.")?;
    let (n, rest) = rest.split_once('.')?;
    let leaf = match rest {
        "input_layernorm.weight" => "ln1.weight",
        "post_attention_layernorm.weight" => "ln2.weight",
        "self_attn.q_proj.weight" => "attn.wq.weight",
        "self_attn.q_proj.bias" => "attn.wq.bias",
        "self_attn.k_proj.weight" => "attn.wk.weight",
        "self_attn.k_proj.bias" => "attn.wk.bias",
        "self_attn.v_proj.weight" => "attn.wv.weight",
        "self_attn.v_proj.bias" => "attn.wv.bias",
        "self_attn.o_proj.weight" => "attn.wo.weight",
        "mlp.gate_proj.weight" => "mlp.gate.weight",
        "mlp.up_proj.weight" => "mlp.up.weight",
        "mlp.down_proj.weight" => "mlp.down.weight",
        _ => return None,
    };
    Some(format!("blocks.{n}.{leaf}"))
}

/// Import `llm.pt` into [`LmWeights`], validated with the same two-way
/// coverage discipline `qwen3::import::brain_init_from_hf` uses for a
/// released Qwen checkpoint: every backbone parameter `cfg.qwen.param_list()`
/// names is produced exactly once with the right element count, and every
/// tensor the reader returns is either consumed (backbone or bolted-on) or
/// the one explicit, documented drop (`llm.model.lm_head.weight`) - an
/// unrecognized tensor fails loudly rather than being silently skipped.
pub fn import_llm_pt(path: &str, cfg: &CosyVoiceLmConfig) -> Result<LmWeights, String> {
    let tensors = checkpoint::torchpt::read(path)?;
    let d = cfg.llm_input_size as usize;
    let v = cfg.speech_vocab() as usize;

    let mut backbone_src: HashMap<String, Vec<f32>> = HashMap::new();
    let mut llm_embedding = None;
    let mut speech_embedding = None;
    let mut llm_decoder_w = None;
    let mut llm_decoder_b = None;

    for t in tensors {
        match t.name.as_str() {
            "llm.model.lm_head.weight" => continue, // tied to embed_tokens.weight - dropped
            "llm_embedding.weight" => {
                llm_embedding = Some(t.data);
                continue;
            }
            "speech_embedding.weight" => {
                speech_embedding = Some(t.data);
                continue;
            }
            "llm_decoder.weight" => {
                llm_decoder_w = Some(t.data);
                continue;
            }
            "llm_decoder.bias" => {
                llm_decoder_b = Some(t.data);
                continue;
            }
            _ => {}
        }
        let Some(bn) = backbone_name(&t.name) else {
            return Err(format!("import_llm_pt: unrecognized tensor {}", t.name));
        };
        if backbone_src.insert(bn.clone(), t.data).is_some() {
            return Err(format!("import_llm_pt: duplicate mapping to {bn}"));
        }
    }

    let mut backbone = HashMap::new();
    for (name, numel) in cfg.qwen.param_list() {
        let data = backbone_src
            .remove(&name)
            .ok_or_else(|| format!("import_llm_pt: missing backbone tensor {name}"))?;
        if data.len() != numel {
            return Err(format!("import_llm_pt: {name} element count {} != expected {numel}", data.len()));
        }
        backbone.insert(name, data);
    }
    if !backbone_src.is_empty() {
        let extra: Vec<&String> = backbone_src.keys().collect();
        return Err(format!("import_llm_pt: {} backbone tensors unused: {extra:?}", backbone_src.len()));
    }

    let llm_embedding = match cfg.special_token_source {
        SpecialTokenSource::LlmEmbedding => {
            let e = llm_embedding.ok_or("import_llm_pt: missing llm_embedding.weight")?;
            if e.len() != 2 * d {
                return Err(format!("import_llm_pt: llm_embedding.weight has {} elements, want {}", e.len(), 2 * d));
            }
            Some(e)
        }
        SpecialTokenSource::SpeechEmbedding => {
            if llm_embedding.is_some() {
                return Err("import_llm_pt: unexpected llm_embedding.weight for a SpeechEmbedding-sourced config".to_string());
            }
            None
        }
    };
    let speech_embedding = speech_embedding.ok_or("import_llm_pt: missing speech_embedding.weight")?;
    let llm_decoder_w = llm_decoder_w.ok_or("import_llm_pt: missing llm_decoder.weight")?;
    let llm_decoder_b = if cfg.llm_decoder_has_bias {
        Some(llm_decoder_b.ok_or("import_llm_pt: missing llm_decoder.bias")?)
    } else {
        if llm_decoder_b.is_some() {
            return Err("import_llm_pt: unexpected llm_decoder.bias for a bias-free config".to_string());
        }
        None
    };
    if speech_embedding.len() != v * d {
        return Err(format!(
            "import_llm_pt: speech_embedding.weight has {} elements, want {}",
            speech_embedding.len(),
            v * d
        ));
    }
    if llm_decoder_w.len() != v * d {
        return Err(format!("import_llm_pt: llm_decoder.weight has {} elements, want {}", llm_decoder_w.len(), v * d));
    }
    if let Some(b) = &llm_decoder_b {
        if b.len() != v {
            return Err(format!("import_llm_pt: llm_decoder.bias has {} elements, want {v}", b.len()));
        }
    }

    Ok(LmWeights {
        backbone,
        llm_embedding,
        speech_embedding,
        llm_decoder_w,
        llm_decoder_b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backbone_name_mapping() {
        assert_eq!(backbone_name("llm.model.model.embed_tokens.weight").unwrap(), "tok.weight");
        assert_eq!(backbone_name("llm.model.model.norm.weight").unwrap(), "norm.weight");
        assert_eq!(
            backbone_name("llm.model.model.layers.5.self_attn.q_proj.weight").unwrap(),
            "blocks.5.attn.wq.weight"
        );
        assert_eq!(
            backbone_name("llm.model.model.layers.5.self_attn.q_proj.bias").unwrap(),
            "blocks.5.attn.wq.bias"
        );
        assert_eq!(
            backbone_name("llm.model.model.layers.23.mlp.down_proj.weight").unwrap(),
            "blocks.23.mlp.down.weight"
        );
        assert_eq!(backbone_name("llm.model.lm_head.weight"), None);
        assert_eq!(backbone_name("llm_embedding.weight"), None);
        assert_eq!(backbone_name("speech_embedding.weight"), None);
    }
}
