// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The draft model's shape.
//!
//! Swedish Embedded AB implements speculative decoding for clients serving large models on modest hardware.
//! If your team needs expertise in porting drafters to a new target then you can procure our services by
//! sending an email to info@swedishembedded.com.

use checkpoint::gguf::{GgufValue, MmapGguf};

const MODEL: &str = "dflash2";

/// The GGUF architecture string [`Dflash2Config::from_gguf`] loads.
pub const GGUF_ARCHITECTURE: &str = "dflash";

/// How a draft layer attends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Within `window` positions of the query, in both directions (the attention is not causal).
    Sliding,
    /// Every position of the context and the block.
    Full,
}

/// One draft layer's attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerSpec {
    pub kind: LayerKind,
    /// Whether the layer rotates its queries and keys. A NoPE layer reads positions only through the values the
    /// target's hidden states carry.
    pub rope: bool,
}

/// The draft model's shape, read from the checkpoint's own configuration and cross-checked against its tensors.
#[derive(Clone, Debug)]
pub struct Dflash2Config {
    pub n_layers: usize,
    pub d_model: u32,
    pub d_ff: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub eps: f32,
    pub rope_theta: f32,
    /// The window of the [`LayerKind::Sliding`] layers.
    pub window: u32,
    /// Attention of each layer, in order.
    pub layers: Vec<LayerSpec>,
    /// Rows per denoising block: one anchor plus `block_size - 1` masks, so the largest `k` the draft can answer
    /// is `block_size - 1`.
    pub block_size: u32,
    pub conv_kernel: u32,
    pub conv_group: u32,
    pub selector_rank: u32,
    pub selector_top_k: u32,
    pub mask_token_id: u32,
    /// `false` in every published DFlash2 checkpoint; a field because it is configuration and configuration is
    /// what a loader believes.
    pub causal: bool,
    /// **0-based decoder-layer indices** of the target whose residual stream (the output of that layer) feeds `fc`,
    /// in concatenation order.
    pub target_layers: Vec<usize>,
    /// The target's vocabulary, which is the draft's (it reuses the target's head).
    pub vocab: u32,
}

impl Dflash2Config {
    /// The shape from a Hugging Face `config.json` (the text of it).
    ///
    /// Keys: the usual `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`,
    /// `num_key_value_heads`, `head_dim`, `rms_norm_eps`, `sliding_window`; `layer_types` (`sliding_attention` or
    /// `full_attention` per layer; without it every layer is sliding when `use_sliding_window` is true and
    /// `sliding_window` is set, else full); `no_rope_layers` (the layers that apply no RoPE, either as indices or
    /// as one 0/1 flag per layer where 1 means RoPE); the RoPE base from `rope_parameters.rope_theta` or
    /// `rope_theta`; and `dflash_config` (`block_size`, `conv_kernel_size`, `conv_group_size`, `selector_rank`,
    /// `selector_top_k`, `mask_token_id`, `target_layer_ids`).
    pub fn from_json(text: &str) -> Result<Dflash2Config, String> {
        let raw: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("{MODEL}: config.json: {e}"))?;
        let uint = |obj: &serde_json::Value, key: &str| -> Result<u32, String> {
            obj.get(key).and_then(|v| v.as_u64()).and_then(|v| u32::try_from(v).ok()).ok_or_else(|| format!("{MODEL}: config.json has no unsigned `{key}`"))
        };
        let d = raw.get("dflash_config").ok_or_else(|| format!("{MODEL}: config.json has no `dflash_config`: this is not a DFlash2 draft"))?;
        let d_model = uint(&raw, "hidden_size")?;
        let n_layers = uint(&raw, "num_hidden_layers")? as usize;
        let n_heads = uint(&raw, "num_attention_heads")?;
        let head_dim = raw.get("head_dim").and_then(|v| v.as_u64()).map_or(Ok(d_model / n_heads.max(1)), |v| u32::try_from(v).map_err(|_| format!("{MODEL}: head_dim")))?;
        let window = raw.get("sliding_window").and_then(|v| v.as_u64()).map(|v| u32::try_from(v).unwrap_or(u32::MAX));
        let use_window = raw.get("use_sliding_window").and_then(|v| v.as_bool()).unwrap_or(window.is_some());
        let kinds: Vec<LayerKind> = match raw.get("layer_types").and_then(|v| v.as_array()) {
            Some(types) => types
                .iter()
                .map(|t| match t.as_str() {
                    Some("sliding_attention") => Ok(LayerKind::Sliding),
                    Some("full_attention") => Ok(LayerKind::Full),
                    other => Err(format!("{MODEL}: layer_types holds {other:?}, expected sliding_attention or full_attention")),
                })
                .collect::<Result<_, _>>()?,
            None if use_window && window.is_some() => vec![LayerKind::Sliding; n_layers],
            None => vec![LayerKind::Full; n_layers],
        };
        if kinds.len() != n_layers {
            return Err(format!("{MODEL}: layer_types has {} entries for {n_layers} layers", kinds.len()));
        }
        let rope = no_rope_flags(raw.get("no_rope_layers"), n_layers)?;
        let theta = raw
            .get("rope_parameters")
            .and_then(|p| p.get("rope_theta"))
            .or_else(|| raw.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .ok_or_else(|| format!("{MODEL}: config.json has neither rope_parameters.rope_theta nor rope_theta"))?;
        let target_layers: Vec<usize> = d
            .get("target_layer_ids")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("{MODEL}: dflash_config has no target_layer_ids"))?
            .iter()
            .map(|v| v.as_u64().map(|i| i as usize).ok_or_else(|| format!("{MODEL}: target_layer_ids holds a non-integer")))
            .collect::<Result<_, _>>()?;
        let cfg = Dflash2Config {
            n_layers,
            d_model,
            d_ff: uint(&raw, "intermediate_size")?,
            n_heads,
            n_kv_heads: uint(&raw, "num_key_value_heads")?,
            head_dim,
            eps: raw.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32,
            rope_theta: theta as f32,
            window: window.unwrap_or(u32::MAX),
            layers: kinds.into_iter().zip(rope).map(|(kind, rope)| LayerSpec { kind, rope }).collect(),
            block_size: uint(d, "block_size")?,
            conv_kernel: uint(d, "conv_kernel_size")?,
            conv_group: uint(d, "conv_group_size")?,
            selector_rank: uint(d, "selector_rank")?,
            selector_top_k: uint(d, "selector_top_k")?,
            mask_token_id: uint(d, "mask_token_id")?,
            causal: raw.get("is_causal").and_then(|v| v.as_bool()).unwrap_or(true),
            target_layers,
            vocab: raw.get("draft_vocab_size").or_else(|| raw.get("vocab_size")).and_then(|v| v.as_u64()).map(|v| v as u32).ok_or_else(|| format!("{MODEL}: config.json has no draft_vocab_size or vocab_size"))?,
        };
        if let Some(total) = raw.get("num_target_layers").and_then(|v| v.as_u64()) {
            if let Some(&bad) = cfg.target_layers.iter().find(|&&l| l as u64 >= total) {
                return Err(format!("{MODEL}: target layer {bad} is outside the target's {total} layers"));
            }
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// The shape from a DFlash2 GGUF, checking the metadata against the file's own tensor shapes rather than
    /// trusting either alone. Every layer is sliding (the GGUF carries one window) and rotates.
    pub fn from_gguf(mg: &MmapGguf) -> Result<Dflash2Config, String> {
        let kv = gguf::kv::ArchKv::expect_architecture(mg, GGUF_ARCHITECTURE)?;
        let d_model = kv.req_u32("embedding_length")?;
        let n_layers = kv.req_u32("block_count")? as usize;
        let head_dim = kv.req_u32("attention.key_length")?;
        // The GGUF stores the target layers one higher than the HF config does, because the reference indexes the
        // `hidden_states` tuple whose entry 0 is the embedding output and entry `i + 1` decoder layer `i`'s output.
        // This carries the decoder-layer form, which is what a tap needs.
        let target_layers = match mg.kv().get("dflash.target_layers") {
            Some(GgufValue::Array(items)) => items
                .iter()
                .map(|v| v.as_u64().and_then(|i| (i as usize).checked_sub(1)).ok_or_else(|| format!("{MODEL}: dflash.target_layers holds a non-integer or an embedding-output index")))
                .collect::<Result<Vec<usize>, String>>()?,
            _ => return Err(format!("{MODEL}: dflash.target_layers missing or not an array")),
        };
        let mask_token_id = mg
            .kv()
            .get("tokenizer.ggml.mask_token_id")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| format!("{MODEL}: tokenizer.ggml.mask_token_id missing - a masked denoiser cannot draft without it"))? as u32;
        let cfg = Dflash2Config {
            n_layers,
            d_model,
            d_ff: kv.req_u32("feed_forward_length")?,
            n_heads: kv.req_u32("attention.head_count")?,
            n_kv_heads: kv.req_u32("attention.head_count_kv")?,
            head_dim,
            eps: kv.f32_or("attention.layer_norm_rms_epsilon", 1e-6),
            rope_theta: kv.req_f32("rope.freq_base")?,
            window: kv.u32_or("attention.sliding_window", u32::MAX),
            layers: vec![LayerSpec { kind: LayerKind::Sliding, rope: true }; n_layers],
            block_size: kv.req_u32("block_size")?,
            conv_kernel: kv.req_u32("conv_kernel_size")?,
            conv_group: kv.req_u32("conv_group_size")?,
            selector_rank: kv.req_u32("selector_rank")?,
            selector_top_k: kv.req_u32("selector_top_k")?,
            mask_token_id,
            causal: kv.bool("attention.causal").unwrap_or(true),
            target_layers,
            // The draft ships no head, so the vocabulary is whatever the selector codebooks are indexed by.
            vocab: mg.shape("selector_predecessor.weight").map(|s| s[0] as u32).ok_or_else(|| format!("{MODEL}: selector_predecessor.weight missing"))?,
        };
        let want = |name: &str, rows: u32, k: u32| -> Result<(), String> {
            let got = mg.shape(name).ok_or_else(|| format!("{MODEL}: {name} missing"))?;
            if got != vec![rows as usize, k as usize] {
                return Err(format!("{MODEL}: {name} is {got:?}, but the metadata implies [{rows}, {k}]"));
            }
            Ok(())
        };
        want("blk.0.attn_q.weight", cfg.n_heads * head_dim, d_model)?;
        want("blk.0.attn_k.weight", cfg.n_kv_heads * head_dim, d_model)?;
        want("fc.weight", d_model, cfg.target_layers.len() as u32 * d_model)?;
        want("blk.0.attn_conv_proj.weight", 2 * cfg.conv_kernel * (d_model / cfg.conv_group), d_model)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Refuse a shape the forward cannot run, by name, before any device work.
    pub fn validate(&self) -> Result<(), String> {
        let bad = |what: String| Err(format!("{MODEL}: {what}"));
        if self.n_layers == 0 || self.layers.len() != self.n_layers {
            return bad(format!("{} layer specs for {} layers", self.layers.len(), self.n_layers));
        }
        if self.n_heads == 0 || self.n_kv_heads == 0 || !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return bad(format!("{} query heads do not group over {} KV heads", self.n_heads, self.n_kv_heads));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return bad(format!("head_dim {} must be positive and even (RoPE rotates pairs)", self.head_dim));
        }
        if self.conv_group == 0 || !self.d_model.is_multiple_of(self.conv_group) {
            return bad(format!("conv group {} does not divide the width {}", self.conv_group, self.d_model));
        }
        if self.conv_kernel == 0 || self.block_size < 2 || self.selector_top_k == 0 || self.selector_rank == 0 {
            return bad("conv_kernel_size, selector_top_k and selector_rank must be positive and block_size at least 2".into());
        }
        if self.window == 0 {
            return bad("a sliding window of 0 positions".into());
        }
        if self.target_layers.is_empty() {
            return bad("no target layers".into());
        }
        if self.mask_token_id >= self.vocab {
            return bad(format!("mask token {} is outside the vocabulary of {}", self.mask_token_id, self.vocab));
        }
        if self.selector_top_k > self.vocab {
            return bad(format!("selector top-k {} exceeds the vocabulary of {}", self.selector_top_k, self.vocab));
        }
        Ok(())
    }

    /// One dynamic-conv projection row: `2 * taps * groups`, where the leading 2 is "one set of taps for the
    /// convolution entering the sublayer, one for the convolution leaving it".
    pub fn dyn_width(&self) -> u32 {
        2 * self.conv_kernel * self.groups()
    }

    pub fn groups(&self) -> u32 {
        self.d_model / self.conv_group
    }

    pub fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }

    pub fn q_dim(&self) -> u32 {
        self.n_heads * self.head_dim
    }

    /// Width of one row of target taps: the tapped layers' residual streams concatenated.
    pub fn tap_width(&self) -> usize {
        self.target_layers.len() * self.d_model as usize
    }

    /// Rows of one slot's K/V cache for layer `layer` when a sequence holds `cap` positions: a sliding layer keeps
    /// the window plus one block (a query of the block reads back `window - 1` positions and forward to the end
    /// of its block), a full layer keeps every position plus the block.
    pub fn cache_rows(&self, layer: usize, cap: u32) -> u32 {
        let block = u64::from(self.block_size);
        let rows = match self.layers[layer].kind {
            LayerKind::Sliding => u64::from(cap).min(u64::from(self.window)) + block,
            LayerKind::Full => u64::from(cap) + block,
        };
        rows.min(u64::from(u32::MAX)) as u32
    }

    /// Bytes of device memory the K/V caches of `slots` sequences of up to `cap` positions cost at f32.
    pub fn cache_bytes(&self, slots: u32, cap: u32) -> u64 {
        (0..self.n_layers).map(|l| 2 * u64::from(slots) * u64::from(self.cache_rows(l, cap)) * u64::from(self.kv_dim()) * 4).sum()
    }
}

/// `no_rope_layers` as one RoPE flag per layer. The key is spelt two ways: a list of 0/1 flags with one entry per
/// layer (1 = the layer rotates), or the indices of the layers that do not.
fn no_rope_flags(value: Option<&serde_json::Value>, n_layers: usize) -> Result<Vec<bool>, String> {
    let Some(list) = value.and_then(|v| v.as_array()) else { return Ok(vec![true; n_layers]) };
    let nums: Vec<u64> = list.iter().map(|v| v.as_u64().ok_or_else(|| format!("{MODEL}: no_rope_layers holds a non-integer"))).collect::<Result<_, _>>()?;
    if nums.len() == n_layers && nums.iter().all(|&n| n <= 1) {
        return Ok(nums.iter().map(|&n| n == 1).collect());
    }
    let mut rope = vec![true; n_layers];
    for n in nums {
        *rope.get_mut(n as usize).ok_or_else(|| format!("{MODEL}: no_rope_layers names layer {n} of {n_layers}"))? = false;
    }
    Ok(rope)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published Ripple-65B-Preview-DFlash2 `config.json` (trimmed to the keys this reads).
    const RIPPLE: &str = r#"{
      "dflash_config": {"block_size": 16, "conv_group_size": 16, "conv_kernel_size": 4, "mask_token_id": 100278,
                        "selector_rank": 256, "selector_top_k": 32, "target_layer_ids": [1, 20, 39, 58, 77]},
      "draft_vocab_size": 100352, "head_dim": 128, "hidden_size": 5120, "intermediate_size": 40960, "is_causal": false,
      "layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
      "no_rope_layers": [4], "num_attention_heads": 80, "num_hidden_layers": 5, "num_key_value_heads": 8, "num_target_layers": 80,
      "rms_norm_eps": 1e-06, "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}, "rope_theta": 10000.0,
      "sliding_window": 4096, "use_sliding_window": true, "vocab_size": 100352}"#;

    #[test]
    fn the_ripple_draft_has_four_sliding_rope_layers_and_one_full_nope_layer() {
        let c = Dflash2Config::from_json(RIPPLE).unwrap();
        let kinds: Vec<_> = c.layers.iter().map(|l| (l.kind, l.rope)).collect();
        assert_eq!(kinds, [(LayerKind::Sliding, true); 4].into_iter().chain([(LayerKind::Full, false)]).collect::<Vec<_>>());
        assert_eq!((c.window, c.block_size, c.conv_kernel, c.target_layers.clone()), (4096, 16, 4, vec![1, 20, 39, 58, 77]));
        assert_eq!((c.dyn_width(), c.tap_width(), c.rope_theta), (2 * 4 * 320, 5 * 5120, 10000.0));
        assert!(!c.causal);
    }

    #[test]
    fn a_sliding_cache_is_the_window_and_a_full_cache_every_position() {
        let c = Dflash2Config::from_json(RIPPLE).unwrap();
        assert_eq!(c.cache_rows(0, 1 << 20), 4096 + 16);
        assert_eq!(c.cache_rows(4, 1 << 20), (1 << 20) + 16);
        assert_eq!(c.cache_rows(0, 100), 100 + 16, "a short context never needs more than its positions");
    }

    #[test]
    fn no_rope_layers_reads_indices_and_per_layer_flags() {
        assert_eq!(no_rope_flags(Some(&serde_json::json!([4])), 5).unwrap(), [true, true, true, true, false]);
        assert_eq!(no_rope_flags(Some(&serde_json::json!([1, 1, 0, 1, 1])), 5).unwrap(), [true, true, false, true, true]);
        assert_eq!(no_rope_flags(None, 3).unwrap(), [true; 3]);
        assert!(no_rope_flags(Some(&serde_json::json!([9])), 5).is_err());
    }

    #[test]
    fn a_config_the_forward_cannot_run_is_refused_by_name() {
        let broken = RIPPLE.replace("\"num_key_value_heads\": 8", "\"num_key_value_heads\": 7");
        assert!(Dflash2Config::from_json(&broken).unwrap_err().contains("do not group"));
        let no_dflash = r#"{"hidden_size": 8}"#;
        assert!(Dflash2Config::from_json(no_dflash).unwrap_err().contains("dflash_config"));
        let past_target = RIPPLE.replace("77]", "80]");
        assert!(Dflash2Config::from_json(&past_target).unwrap_err().contains("outside the target"));
    }
}
