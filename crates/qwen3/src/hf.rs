// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The HuggingFace `transformers` side of the dense decoder this crate runs:
//! its `config.json` and its tensor names, for every architecture whose
//! implementation this crate is (`qwen3`, and the `qwen2` / `llama` config
//! variants - see `brain_arch::Arch::implementation`).
//!
//! The three share one tensor layout and differ only in switches, so they
//! share one reader and one name map:
//!
//! | architecture | q/k/v bias | QK-norm |
//! |---|---|---|
//! | `qwen3` (`Qwen3ForCausalLM`) | no | yes |
//! | `qwen2` (`Qwen2ForCausalLM`) | yes | no |
//! | `llama` (`LlamaForCausalLM`) | no | no |
//!
//! Anything a config asks for that the decoder does not implement - an
//! `o_proj`/MLP bias, a sliding window, another activation, an unsupported
//! RoPE scaling - is refused by name here, never dropped: a checkpoint that
//! loads with one of its switches ignored computes a different model.
//!
//! Swedish Embedded AB brings model families onto its clients' own inference
//! engines. If your team needs expertise in importing and validating
//! transformer checkpoints without silent configuration loss, you can procure
//! our services by sending an email to info@swedishembedded.com.

use serde::Deserialize;

use crate::config::QwenConfig;

/// The `config.json` fields this decoder reads. Everything else a
/// `transformers` config carries (token ids, `torch_dtype`, `_name_or_path`,
/// ...) is not the model's shape and is ignored; every field that IS the
/// model's shape is either read here or refused in [`decoder_config`].
#[derive(Deserialize)]
struct HfDecoderConfig {
    #[serde(default)]
    architectures: Vec<String>,
    model_type: Option<String>,
    vocab_size: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: Option<u32>,
    head_dim: Option<u32>,
    intermediate_size: u32,
    rope_theta: Option<f64>,
    rms_norm_eps: Option<f64>,
    max_position_embeddings: Option<u32>,
    tie_word_embeddings: Option<bool>,
    attention_bias: Option<bool>,
    mlp_bias: Option<bool>,
    hidden_act: Option<String>,
    use_sliding_window: Option<bool>,
    rope_scaling: Option<serde_json::Value>,
}

/// The sequence length a freshly imported checkpoint is built for. Not
/// `max_position_embeddings`: that is the trained RoPE extent (131072 for
/// the R1 distills), and sizing buffers by it would allocate for contexts no
/// caller asked for. The real length is chosen at load time.
const IMPORT_BLOCK_SIZE: u32 = 2048;

/// Read a `transformers` `config.json` of any architecture this crate
/// implements into a [`QwenConfig`].
///
/// A field the config omits takes the value its own `transformers` class
/// defaults to (the same thing loading the checkpoint in `transformers` would
/// do), so a trimmed config means what it means upstream rather than
/// whatever brain happened to assume.
pub fn decoder_config(json: &str) -> Result<QwenConfig, String> {
    let c: HfDecoderConfig = serde_json::from_str(json).map_err(|e| format!("config.json: {e}"))?;
    let class = c.architectures.first().cloned().or_else(|| c.model_type.clone()).ok_or("config.json names neither `architectures` nor `model_type`")?;
    let arch = brain_arch::by_hf(&class).ok_or_else(|| format!("config.json: architecture {class:?} is not one brain recognizes"))?;
    if arch.implementation().id != "qwen3" {
        return Err(format!("config.json: {class:?} is `{}`, which is not implemented by this decoder", arch.id));
    }
    from_hf(c, arch.id)
}

/// [`decoder_config`] for a composite checkpoint whose `config.json` names
/// the composite (`LlavaQwen2ForCausalLM`, ...) rather than its decoder:
/// the caller says which of this crate's architectures the decoder is
/// (`"qwen2"`, `"llama"`, `"qwen3"`), and the config's decoder fields are
/// read exactly as they would be for that architecture on its own.
pub fn decoder_config_as(json: &str, arch_id: &str) -> Result<QwenConfig, String> {
    let c: HfDecoderConfig = serde_json::from_str(json).map_err(|e| format!("config.json: {e}"))?;
    from_hf(c, arch_id)
}

fn from_hf(c: HfDecoderConfig, arch_id: &str) -> Result<QwenConfig, String> {
    // Per-architecture: which attention switches the class hardwires, and
    // the defaults of the fields it lets a config omit
    // (`transformers.{Qwen3,Qwen2,Llama}Config`).
    let (qk_norm, qkv_bias_fixed, max_pos_default) = match arch_id {
        "qwen3" => (true, false, 32768),
        // Qwen2Attention builds q/k/v with `bias=True` unconditionally and
        // o_proj without; there is no `attention_bias` switch.
        "qwen2" => (false, true, 32768),
        "llama" => (false, false, 2048),
        other => return Err(format!("config.json: no decoder profile for architecture `{other}`")),
    };
    // Llama and Qwen3 put `attention_bias` on ALL FOUR projections, o_proj
    // included; this decoder has q/k/v bias only.
    if c.attention_bias == Some(true) {
        return Err(format!("config.json: `attention_bias: true` biases o_proj too, which the {arch_id} decoder does not implement"));
    }
    if c.mlp_bias == Some(true) {
        return Err("config.json: `mlp_bias: true` is not implemented by this decoder".into());
    }
    if let Some(act) = c.hidden_act.as_deref().filter(|a| *a != "silu") {
        return Err(format!("config.json: hidden_act {act:?} is not implemented (SwiGLU uses silu)"));
    }
    if c.use_sliding_window == Some(true) {
        return Err("config.json: `use_sliding_window: true` (windowed attention) is not implemented by this decoder".into());
    }
    if let Some(rs) = c.rope_scaling.as_ref().filter(|v| !v.is_null()) {
        let kind = rs.get("rope_type").or_else(|| rs.get("type")).and_then(|t| t.as_str()).unwrap_or("?");
        return Err(format!("config.json: rope_scaling type {kind:?} is not implemented yet"));
    }
    let cfg = QwenConfig {
        vocab: c.vocab_size,
        block_size: IMPORT_BLOCK_SIZE,
        n_layers: c.num_hidden_layers,
        d_model: c.hidden_size,
        n_heads: c.num_attention_heads,
        n_kv_heads: c.num_key_value_heads.unwrap_or(c.num_attention_heads),
        head_dim: c.head_dim.unwrap_or(0), // 0 -> derived in with_defaults
        d_ff: c.intermediate_size,
        rope_theta: c.rope_theta.unwrap_or(10_000.0) as f32,
        rms_eps: c.rms_norm_eps.unwrap_or(1e-6) as f32,
        max_position_embeddings: c.max_position_embeddings.unwrap_or(max_pos_default),
        tie_embeddings: c.tie_word_embeddings.unwrap_or(false),
        qk_norm,
        attn_bias: qkv_bias_fixed,
        lora: None,
        rope_scaling: None,
    }
    .with_defaults();
    Ok(cfg)
}

/// What one HF tensor is to this decoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HfTensor {
    /// This brain parameter.
    Param(String),
    /// Part of this model but not a parameter brain stores: a tied
    /// `lm_head`, or a `rotary_emb.inv_freq` buffer (older checkpoints
    /// serialize it; brain derives it from the config).
    Dropped,
    /// Not this decoder's tensor at all (outside its prefix) - another
    /// component of a composite checkpoint.
    Foreign,
    /// Inside this decoder's namespace but not something it has: a bias the
    /// config did not declare, a QK-norm on an architecture without one, an
    /// unrecognized leaf. Importing past it would silently lose weights.
    Unknown,
}

/// Every per-layer leaf of the decoder: `(HF name under layers.N., brain
/// name under blocks.N.)`. The one table both directions of the map read.
const LEAVES: &[(&str, &str)] = &[
    ("input_layernorm.weight", "ln1.weight"),
    ("post_attention_layernorm.weight", "ln2.weight"),
    ("self_attn.q_proj.weight", "attn.wq.weight"),
    ("self_attn.k_proj.weight", "attn.wk.weight"),
    ("self_attn.v_proj.weight", "attn.wv.weight"),
    ("self_attn.o_proj.weight", "attn.wo.weight"),
    ("self_attn.q_proj.bias", "attn.wq.bias"),
    ("self_attn.k_proj.bias", "attn.wk.bias"),
    ("self_attn.v_proj.bias", "attn.wv.bias"),
    ("self_attn.q_norm.weight", "attn.q_norm.weight"),
    ("self_attn.k_norm.weight", "attn.k_norm.weight"),
    ("mlp.gate_proj.weight", "mlp.gate.weight"),
    ("mlp.up_proj.weight", "mlp.up.weight"),
    ("mlp.down_proj.weight", "mlp.down.weight"),
];

/// The brain leaf (under `blocks.N.`) of one HF decoder-layer leaf (under
/// `layers.N.`), for a composite whose top-level names are its own (a talker
/// with a codec embedding instead of `embed_tokens`) but whose layers are
/// this decoder's.
pub fn layer_leaf(hf_leaf: &str) -> Option<&'static str> {
    LEAVES.iter().find(|(hf, _)| *hf == hf_leaf).map(|(_, b)| *b)
}

/// The inverse of [`layer_leaf`].
pub fn hf_layer_leaf(brain_leaf: &str) -> Option<&'static str> {
    LEAVES.iter().find(|(_, b)| *b == brain_leaf).map(|(hf, _)| *hf)
}

/// Where a checkpoint keeps the decoder: the prefix of its body
/// (`model.` for a plain `*ForCausalLM`; a composite nests it) and the name
/// of its output head.
#[derive(Clone, Copy, Debug)]
pub struct HfNames {
    pub prefix: &'static str,
    pub head: &'static str,
}

impl HfNames {
    /// A plain `*ForCausalLM` checkpoint.
    pub const CAUSAL_LM: HfNames = HfNames { prefix: "model.", head: "lm_head.weight" };

    /// Classify one HF tensor name against `cfg`'s switches.
    pub fn to_brain(&self, name: &str, cfg: &QwenConfig) -> HfTensor {
        if name == self.head {
            return if cfg.tie_embeddings { HfTensor::Dropped } else { HfTensor::Param("lm_head.weight".into()) };
        }
        let Some(rest) = name.strip_prefix(self.prefix) else { return HfTensor::Foreign };
        if let Some((n, leaf)) = rest.strip_prefix("layers.").and_then(|r| r.split_once('.')) {
            if leaf == "self_attn.rotary_emb.inv_freq" {
                return HfTensor::Dropped;
            }
            if n.parse::<u32>().map_or(true, |l| l >= cfg.n_layers) {
                return HfTensor::Unknown;
            }
        }
        match self.body_param(name) {
            Some(p) if (p.ends_with("attn.wq.bias") || p.ends_with("attn.wk.bias") || p.ends_with("attn.wv.bias")) && !cfg.attn_bias => HfTensor::Unknown,
            Some(p) if (p.ends_with("attn.q_norm.weight") || p.ends_with("attn.k_norm.weight")) && !cfg.qk_norm => HfTensor::Unknown,
            Some(p) => HfTensor::Param(p),
            None => HfTensor::Unknown,
        }
    }

    /// The brain parameter an HF name under this prefix spells, by name alone
    /// (every leaf any of the three architectures has, bias and QK-norm
    /// included; no head, no switches). For a composite reader that routes
    /// its checkpoint by prefix and validates coverage against its own
    /// `param_list` - [`Self::to_brain`] is the checked form.
    pub fn body_param(&self, name: &str) -> Option<String> {
        let rest = name.strip_prefix(self.prefix)?;
        match rest {
            "embed_tokens.weight" => return Some("tok.weight".into()),
            "norm.weight" => return Some("norm.weight".into()),
            _ => {}
        }
        let (n, leaf) = rest.strip_prefix("layers.")?.split_once('.')?;
        let brain = layer_leaf(leaf)?;
        Some(format!("blocks.{n}.{brain}"))
    }

    /// The HF name brain parameter `name` is stored under, or `None` for a
    /// name that is not one of this decoder's parameters. The inverse of
    /// [`Self::to_brain`] for every [`HfTensor::Param`].
    pub fn from_brain(&self, name: &str) -> Option<String> {
        match name {
            "tok.weight" => return Some(format!("{}embed_tokens.weight", self.prefix)),
            "norm.weight" => return Some(format!("{}norm.weight", self.prefix)),
            "lm_head.weight" => return Some(self.head.to_string()),
            _ => {}
        }
        let (n, leaf) = name.strip_prefix("blocks.")?.split_once('.')?;
        let hf = hf_layer_leaf(leaf)?;
        Some(format!("{}layers.{n}.{hf}", self.prefix))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verbatim shape fields of the released configs (token ids and dtype
    // trimmed; they are not read).
    const R1_QWEN_1_5B: &str = r#"{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2","hidden_size":1536,"intermediate_size":8960,
        "num_hidden_layers":28,"num_attention_heads":12,"num_key_value_heads":2,"vocab_size":151936,"rope_theta":10000,
        "rms_norm_eps":1e-06,"tie_word_embeddings":false,"max_position_embeddings":131072,"sliding_window":4096,
        "max_window_layers":21,"use_sliding_window":false,"hidden_act":"silu"}"#;
    const LLM_7B: &str = r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":4096,"intermediate_size":11008,
        "num_hidden_layers":30,"num_attention_heads":32,"num_key_value_heads":32,"vocab_size":102400,"rope_theta":10000.0,
        "rope_scaling":null,"rms_norm_eps":1e-06,"tie_word_embeddings":false,"max_position_embeddings":4096,"hidden_act":"silu"}"#;
    const QWEN3_0_6B: &str = r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","hidden_size":1024,"head_dim":128,
        "intermediate_size":3072,"num_hidden_layers":28,"num_attention_heads":16,"num_key_value_heads":8,"vocab_size":151936,
        "rope_theta":1000000,"rope_scaling":null,"rms_norm_eps":1e-06,"tie_word_embeddings":true,"max_position_embeddings":40960,
        "attention_bias":false,"use_sliding_window":false,"hidden_act":"silu"}"#;

    #[test]
    fn a_qwen2_checkpoint_is_the_decoder_with_qkv_bias_and_no_qk_norm() {
        let c = decoder_config(R1_QWEN_1_5B).unwrap();
        assert!(c.attn_bias && !c.qk_norm && !c.tie_embeddings);
        assert_eq!((c.d_model, c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim, c.d_ff), (1536, 28, 12, 2, 128, 8960));
        assert_eq!((c.rope_theta, c.rms_eps, c.vocab), (10_000.0, 1e-6, 151936));
    }

    #[test]
    fn a_llama_checkpoint_is_the_decoder_with_neither() {
        let c = decoder_config(LLM_7B).unwrap();
        assert!(!c.attn_bias && !c.qk_norm && !c.tie_embeddings);
        assert_eq!((c.n_heads, c.n_kv_heads, c.head_dim), (32, 32, 128), "full multi-head attention");
    }

    #[test]
    fn a_qwen3_checkpoint_reads_as_it_always_did() {
        let c = decoder_config(QWEN3_0_6B).unwrap();
        assert!(c.qk_norm && !c.attn_bias && c.tie_embeddings);
        assert_eq!((c.head_dim, c.rope_theta, c.max_position_embeddings), (128, 1.0e6, 40960));
    }

    #[test]
    fn omitted_fields_take_their_transformers_class_defaults() {
        let trimmed = r#"{"architectures":["LlamaForCausalLM"],"hidden_size":64,"intermediate_size":96,"num_hidden_layers":2,
            "num_attention_heads":4,"vocab_size":32}"#;
        let c = decoder_config(trimmed).unwrap();
        assert_eq!((c.n_kv_heads, c.rope_theta, c.rms_eps, c.tie_embeddings, c.max_position_embeddings), (4, 10_000.0, 1e-6, false, 2048));
    }

    #[test]
    fn a_switch_this_decoder_does_not_implement_is_refused_by_name() {
        let with = |k: &str, v: &str| LLM_7B.replacen("\"hidden_act\":\"silu\"", &format!("\"{k}\":{v}"), 1);
        for (k, v) in [("use_sliding_window", "true"), ("attention_bias", "true"), ("mlp_bias", "true"), ("hidden_act", "\"gelu\"")] {
            let e = decoder_config(&with(k, v)).unwrap_err();
            assert!(e.contains(k), "{k}: {e}");
        }
        let scaled = LLM_7B.replace("\"rope_scaling\":null", r#""rope_scaling":{"factor":4.0,"type":"linear"}"#);
        assert!(decoder_config(&scaled).unwrap_err().contains("linear"));
        let foreign = LLM_7B.replace("LlamaForCausalLM", "GPT2LMHeadModel");
        assert!(decoder_config(&foreign).is_err());
    }

    #[test]
    fn the_name_map_follows_the_configs_switches() {
        let names = HfNames::CAUSAL_LM;
        let qwen2 = decoder_config(R1_QWEN_1_5B).unwrap();
        let llama = decoder_config(LLM_7B).unwrap();
        let bias = "model.layers.3.self_attn.k_proj.bias";
        assert_eq!(names.to_brain(bias, &qwen2), HfTensor::Param("blocks.3.attn.wk.bias".into()));
        assert_eq!(names.to_brain(bias, &llama), HfTensor::Unknown, "an undeclared bias must not be silently dropped");
        assert_eq!(names.to_brain("model.layers.0.self_attn.q_norm.weight", &llama), HfTensor::Unknown);
        assert_eq!(names.to_brain("model.layers.0.self_attn.rotary_emb.inv_freq", &llama), HfTensor::Dropped);
        assert_eq!(names.to_brain("model.layers.30.mlp.up_proj.weight", &llama), HfTensor::Unknown, "a layer past n_layers");
        assert_eq!(names.to_brain("lm_head.weight", &llama), HfTensor::Param("lm_head.weight".into()));
        assert_eq!(names.to_brain("vision_tower.patch.weight", &llama), HfTensor::Foreign);
        // Every parameter round-trips through the inverse.
        for (p, _) in qwen2.param_list() {
            let hf = names.from_brain(&p).unwrap_or_else(|| panic!("no HF name for {p}"));
            assert_eq!(names.to_brain(&hf, &qwen2), HfTensor::Param(p));
        }
    }
}
