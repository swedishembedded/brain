// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Gemma-4 text tower's configuration - every FLAG that changes the op
//! sequence, transcribed from `tools/goldens/gemma4_dump_reference.py`'s
//! `TINY_CONFIG` (cross-checked against `manifest.json`'s `run.tiny_config`,
//! authoritative over any transcription here, same discipline `ltxv::config`
//! documents).
//!
//! Every real-LTX-2.5-Gemma-4-config flag is set to its real value even at
//! toy dims: `attention_k_eq_v=true` on the global (full-attention) layers,
//! `global_head_dim` DOUBLE the sliding `head_dim` (real: 512 vs 256),
//! `num_global_key_value_heads` narrower than the sliding `num_key_value_heads`
//! (real: 1 vs 8, i.e. the global layers are MQA where the sliding ones are
//! GQA), a `partial_rotary_factor` of `0.25` on the global layers' RoPE, and
//! the AUTO-DERIVED `sliding_window_pattern=6` layer-type alternation (5
//! sliding then 1 full, repeating - `real 48 layers -> 40 sliding + 8 full`,
//! this crate's tiny config uses the pattern's minimal instance: 6 layers ->
//! 5 sliding + 1 full). `vocab_size`/`hidden_size`/`intermediate_size` are
//! shrunk for a fast test - they do not participate in the structurally
//! interesting logic (RoPE construction, the sliding/full alternation,
//! k_eq_v, the aggregate projection).

/// Whether a decoder layer is a windowed (`sliding_attention`) or a
/// wide/global (`full_attention`) attention layer - `Gemma4TextConfig.
/// layer_types[i]`, real values `"sliding_attention"`/`"full_attention"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    Sliding,
    Full,
}

/// `configuration_gemma4.py`'s `Gemma4TextConfig.__post_init__`: layer `i`
/// (0-indexed) is `full_attention` iff `(i+1) % sliding_window_pattern == 0`,
/// with the LAST layer forced to `full_attention` regardless (already implied
/// here whenever `num_hidden_layers % sliding_window_pattern == 0`, which
/// every config this crate builds satisfies).
const SLIDING_WINDOW_PATTERN: u32 = 6;

/// The Gemma-4 text tower's shape + op-sequence configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gemma4Config {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    /// Query head count - shared by every layer regardless of type (only the
    /// per-head WIDTH and the key/value head count differ by layer type).
    pub num_attention_heads: u32,
    /// KV head count on `sliding_attention` layers (GQA).
    pub num_key_value_heads: u32,
    /// Per-head width on `sliding_attention` layers.
    pub head_dim: u32,
    /// Per-head width on `full_attention` layers - real config: `2 *
    /// head_dim` (512 vs 256).
    pub global_head_dim: u32,
    /// KV head count on `full_attention` layers (real config: MQA, `1`).
    pub num_global_key_value_heads: u32,
    /// `true` for the real LTX-2.5 config (class default `false`) - the
    /// `full_attention` layers reuse their raw `k_proj` output as `v_norm`'s
    /// input instead of having their own `v_proj` at all.
    pub attention_k_eq_v: bool,
    /// Window size for `sliding_attention` layers: key `j` lives iff `i -
    /// window < j <= i`.
    pub sliding_window: u32,
    pub rms_norm_eps: f32,
    /// `sliding_attention`'s RoPE base (real config: `10000.0`, `rope_type
    /// = "default"` - every dim rotates).
    pub rope_theta_sliding: f64,
    /// `full_attention`'s RoPE base (real config: `1000000.0`, `rope_type =
    /// "proportional"`).
    pub rope_theta_full: f64,
    /// `full_attention`'s `partial_rotary_factor` (real config: `0.25`) -
    /// only this fraction of `global_head_dim` actually rotates; the rest
    /// passes through unrotated (`rope2d_partial`'s exact contract).
    pub partial_rotary_factor: f64,
}

impl Gemma4Config {
    /// `configuration_gemma4.py`'s auto-derived `layer_types` - see this
    /// module's doc.
    pub fn layer_type(&self, layer_idx: u32) -> LayerType {
        assert!(layer_idx < self.num_hidden_layers);
        if !(layer_idx + 1).is_multiple_of(SLIDING_WINDOW_PATTERN) {
            LayerType::Sliding
        } else {
            LayerType::Full
        }
    }

    /// This layer's per-head width (`head_dim` or `global_head_dim`).
    pub fn head_dim_for(&self, lt: LayerType) -> u32 {
        match lt {
            LayerType::Sliding => self.head_dim,
            LayerType::Full => self.global_head_dim,
        }
    }

    /// This layer's key/value head count.
    pub fn kv_heads_for(&self, lt: LayerType) -> u32 {
        match lt {
            LayerType::Sliding => self.num_key_value_heads,
            LayerType::Full => self.num_global_key_value_heads,
        }
    }

    /// `num_attention_heads / kv_heads_for(lt)` - `gqa_scores_win`/`gqa_apply`'s
    /// `group` param.
    pub fn groups_for(&self, lt: LayerType) -> u32 {
        let kv = self.kv_heads_for(lt);
        assert_eq!(self.num_attention_heads % kv, 0, "num_attention_heads {} not a multiple of kv_heads {kv} for {lt:?}", self.num_attention_heads);
        self.num_attention_heads / kv
    }

    /// Whether THIS layer reuses its raw `k_proj` output as V (no `v_proj`
    /// at all) - `config.attention_k_eq_v and not is_sliding`.
    pub fn k_eq_v_for(&self, lt: LayerType) -> bool {
        self.attention_k_eq_v && lt == LayerType::Full
    }

    /// `tools/goldens/gemma4_dump_reference.py`'s `TINY_CONFIG` - 6 layers (5
    /// sliding + 1 full, the real 5:1 ratio's minimal instance), every flag
    /// at its real-LTX-2.5 value. Cross-checked field by field against
    /// `testdata/golden/gemma4/manifest.json`'s `run.tiny_config`.
    pub fn tiny() -> Gemma4Config {
        Gemma4Config {
            vocab_size: 48,
            hidden_size: 24,
            intermediate_size: 32,
            num_hidden_layers: 6,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            global_head_dim: 16,
            num_global_key_value_heads: 1,
            attention_k_eq_v: true,
            sliding_window: 3,
            rms_norm_eps: 1e-6,
            rope_theta_sliding: 10_000.0,
            rope_theta_full: 1_000_000.0,
            partial_rotary_factor: 0.25,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_config_layer_types_match_the_real_5_to_1_pattern() {
        let c = Gemma4Config::tiny();
        let types: Vec<LayerType> = (0..c.num_hidden_layers).map(|i| c.layer_type(i)).collect();
        assert_eq!(types, [LayerType::Sliding; 5].into_iter().chain([LayerType::Full]).collect::<Vec<_>>());
    }

    #[test]
    fn full_layers_are_mqa_k_eq_v_and_double_width() {
        let c = Gemma4Config::tiny();
        assert!(c.k_eq_v_for(LayerType::Full));
        assert!(!c.k_eq_v_for(LayerType::Sliding));
        assert_eq!(c.head_dim_for(LayerType::Full), 2 * c.head_dim_for(LayerType::Sliding));
        assert_eq!(c.groups_for(LayerType::Full), c.num_attention_heads); // MQA: 1 kv head
        assert_eq!(c.groups_for(LayerType::Sliding), 2); // GQA: 4 heads / 2 kv heads
    }
}
