// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3 decoder configuration + parameter layout.
//!
//! Qwen3 (0.6B reference): RMSNorm (no bias), RoPE (base 1e6, half-split), GQA
//! (`n_kv_heads < n_heads`), per-head QK-RMSNorm over `head_dim` applied to q/k
//! before RoPE, SwiGLU MLP, tied embeddings, **no** attention/MLP biases, and a
//! **decoupled `head_dim`** (e.g. hidden 1024 but 16 heads × 128 = 2048 ≠ 1024).

use serde_json::Value;

/// LoRA adapter configuration (low-rank fine-tuning). When present, the targeted
/// projections keep a frozen base weight plus trainable `A`/`B` adapters.
///
/// `PartialEq` (on top of the `Clone, Debug` every other config-carried struct
/// in this tree gets): `crates/deepseek2::config::DeepseekV2Config` derives it
/// and round-trips through JSON in its own tests, and reusing this struct as-is
/// rather than redeclaring it (this crate's own `Qwen35Config`/`Qwen35`
/// precedent) means the derive has to hold here too.
#[derive(Clone, Debug, PartialEq)]
pub struct LoraCfg {
    pub rank: u32,
    pub alpha: f32,
    /// Which projections get adapters (matched by the leaf name, e.g. "wq").
    pub targets: Vec<String>,
}

impl LoraCfg {
    /// The default Qwen LoRA target set: the four attention projections.
    pub fn attn(rank: u32, alpha: f32) -> LoraCfg {
        LoraCfg {
            rank,
            alpha,
            targets: ["wq", "wk", "wv", "wo"].iter().map(|s| s.to_string()).collect(),
        }
    }
    pub fn targets_leaf(&self, leaf: &str) -> bool {
        self.targets.iter().any(|t| t == leaf)
    }
}

#[derive(Clone, Debug)]
pub struct QwenConfig {
    pub vocab: u32,
    pub block_size: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub d_ff: u32,
    pub rope_theta: f32,
    pub rms_eps: f32,
    /// HF `max_position_embeddings` (the checkpoint's trained RoPE extent),
    /// carried through for reference — NOT what sizes runtime buffers (that is
    /// `block_size`/the `t` an instance is loaded/built with; see
    /// `import::config_from_hf`'s doc comment on why `block_size` doesn't
    /// default to it). Defaults to `block_size` when absent from the source
    /// (HF `config.json` predating this field, or a brain checkpoint written
    /// before this field existed), so old checkpoints keep loading unchanged.
    pub max_position_embeddings: u32,
    pub tie_embeddings: bool,
    /// Per-head QK-RMSNorm on q/k before RoPE. `true` for Qwen3; `false` for
    /// Qwen2 (which has no QK-norm). Governs both the forward and `param_list`.
    pub qk_norm: bool,
    /// Bias on the q/k/v projections. `false` for Qwen3 (bias-free); `true` for
    /// Qwen2 (q/k/v carry a bias, o/gate/up/down do not).
    pub attn_bias: bool,
    /// `Some` selects LoRA fine-tuning (frozen base + adapters); `None` is a
    /// full (all-parameter) model.
    pub lora: Option<LoraCfg>,
}

impl QwenConfig {
    /// A tiny config for tests / gradient checks. Exercises GQA (4 q / 2 kv
    /// heads, group 2), a decoupled `head_dim` (8, vs d_model/n_heads = 4),
    /// QK-norm, RoPE-base, SwiGLU and the tied head.
    pub fn tiny() -> QwenConfig {
        QwenConfig {
            vocab: 23,
            block_size: 12,
            n_layers: 2,
            d_model: 16,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 8,
            d_ff: 32,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            max_position_embeddings: 12,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        }
    }

    /// [`Self::tiny`] at dims the INT8/W4A8 tiers can actually take.
    ///
    /// `model::int8::quantize_weight` scales a weight per `model::int8::GROUP`
    /// (32) elements of its contraction axis, so every quantized linear's `k`
    /// must be a whole number of groups. `tiny`'s `d_model = 16` is not, and
    /// `tiny` is shared by ~100 fp32 tests and the gradient checker, none of
    /// which should pay a bigger fixture for a constraint only the quantized
    /// tiers have. This raises exactly the three widths that ARE a quantized
    /// `k` - `d_model` (q/k/v-proj, gate, up), `q_dim` (o_proj) and `d_ff`
    /// (down) - to 64 / 32 / 96, keeping all four of `d_model`, `q_dim`,
    /// `kv_dim` and `d_ff` mutually DISTINCT so a transpose between any two
    /// still cannot hide (lesson #4), and leaves everything else `tiny`'s.
    pub fn tiny_i8() -> QwenConfig {
        QwenConfig { d_model: 64, d_ff: 96, ..QwenConfig::tiny() }
    }

    /// The published Qwen3-0.6B shape (from its `config.json`).
    pub fn qwen3_0_6b() -> QwenConfig {
        QwenConfig {
            vocab: 151936,
            block_size: 1024,
            n_layers: 28,
            d_model: 1024,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            d_ff: 3072,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            max_position_embeddings: 1024,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        }
    }

    /// The published Qwen3-1.7B shape (from its `config.json`) — the text
    /// decoder used by Qwen3-ASR (`hidden_size` 2048, 28 layers, GQA 16/8,
    /// `head_dim` 128 so `q_dim` 2048, SwiGLU `d_ff` 6144, tied, θ 1e6).
    pub fn qwen3_1_7b() -> QwenConfig {
        QwenConfig {
            vocab: 151936,
            block_size: 1024,
            n_layers: 28,
            d_model: 2048,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            d_ff: 6144,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            max_position_embeddings: 1024,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        }
    }

    /// The published Qwen3-4B shape (from its `config.json`) — the text encoder
    /// used by Z-Image and FLUX.2 (`hidden_size` 2560, 36 layers, GQA 32/8,
    /// `head_dim` 128 so `q_dim` 4096 ≠ 2560, SwiGLU `d_ff` 9728, tied).
    pub fn qwen3_4b() -> QwenConfig {
        QwenConfig {
            vocab: 151936,
            block_size: 1024,
            n_layers: 36,
            d_model: 2560,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            d_ff: 9728,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            max_position_embeddings: 1024,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        }
    }


    /// The published Qwen3-8B shape (from its `config.json`) — the text encoder
    /// used by FLUX.2 Klein 9B (`hidden_size` 4096, 36 layers, GQA 32/8,
    /// `head_dim` 128, SwiGLU `d_ff` 12288, untied `lm_head`).
    pub fn qwen3_8b() -> QwenConfig {
        QwenConfig {
            vocab: 151936,
            block_size: 1024,
            n_layers: 36,
            d_model: 4096,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            d_ff: 12288,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            max_position_embeddings: 1024,
            tie_embeddings: false,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        }
    }

    /// Vicuna-1.5-13B's decoder half - a LLaMA-2-13B fine-tune with **no**
    /// architecture changes: 40 layers, `hidden_size` 5120, 40 attention heads,
    /// `num_key_value_heads` 40 (plain MHA, unlike Qwen3's GQA), `head_dim` 128
    /// (5120/40), SwiGLU `intermediate_size` 13824, RoPE base 10000 (no
    /// `rope_scaling`), `rms_norm_eps` 1e-5, `max_position_embeddings` 4096,
    /// untied `lm_head`, no QK-norm, no attention bias. Read off the real
    /// `meta-llama/Llama-2-13b-hf` `config.json` (mirrored, ungated, at
    /// `NousResearch/Llama-2-13b-hf`) - vocab 32000 matches `data::llama_bpe`'s
    /// SentencePiece byte-fallback tokenizer.
    pub fn llama2_13b() -> QwenConfig {
        QwenConfig {
            vocab: 32000,
            block_size: 4096,
            n_layers: 40,
            d_model: 5120,
            n_heads: 40,
            n_kv_heads: 40,
            head_dim: 128,
            d_ff: 13824,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            max_position_embeddings: 4096,
            tie_embeddings: false,
            qk_norm: false,
            attn_bias: false,
            lora: None,
        }
    }

    /// Apply derived defaults (head_dim = d_model/n_heads when unset).
    pub fn with_defaults(mut self) -> Self {
        if self.head_dim == 0 {
            self.head_dim = self.d_model / self.n_heads;
        }
        self
    }

    /// Query projection width = `n_heads * head_dim` (may differ from d_model).
    pub fn q_dim(&self) -> u32 {
        self.n_heads * self.head_dim
    }
    /// Key/Value projection width = `n_kv_heads * head_dim`.
    pub fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }
    /// Query heads per kv head (GQA grouping factor).
    pub fn group(&self) -> u32 {
        self.n_heads / self.n_kv_heads
    }
    /// The lm_head parameter name (tied -> the embedding table).
    pub fn head_weight(&self) -> &'static str {
        if self.tie_embeddings {
            "tok.weight"
        } else {
            "lm_head.weight"
        }
    }

    pub fn to_json(&self) -> Value {
        let mut v = serde_json::json!({
            "model": "qwen",
            "vocab_size": self.vocab, "block_size": self.block_size, "n_layers": self.n_layers,
            "d_model": self.d_model, "n_heads": self.n_heads, "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim, "d_ff": self.d_ff,
            "rope_theta": self.rope_theta, "rms_norm_eps": self.rms_eps,
            "max_position_embeddings": self.max_position_embeddings,
            "tie_word_embeddings": self.tie_embeddings
        });
        // A LoRA checkpoint must round-trip its adapter shape, or `param_list()`
        // rebuilds without the `.lora_a`/`.lora_b` names on load and the trained
        // adapters are silently dropped (see crates/qwen3/tests/lora_roundtrip.rs).
        if let Some(l) = &self.lora {
            v["lora"] = serde_json::json!({
                "rank": l.rank, "alpha": l.alpha, "targets": l.targets,
            });
        }
        v
    }

    /// Every JSON key [`Self::from_json`] must find to read this config's real
    /// SHAPE (param counts, FLOPs, tensor sizes) rather than silently
    /// substitute an unrelated hardcoded default for it - see that
    /// function's own `g`/`gf` closures, which do exactly that for any
    /// absent key with no warning. Deliberately excludes
    /// `max_position_embeddings` (documented on that field as legitimately
    /// absent from an old checkpoint, defaulting to `block_size`) and the
    /// boolean/optional fields (`tie_word_embeddings`, `qk_norm`,
    /// `attention_bias`, `lora`), whose defaults are small, sensible,
    /// Qwen3-shaped fallbacks - not a silently-wrong shape parameter the way
    /// a missing `vocab_size` defaulting to `23` is.
    pub const SHAPE_KEYS: &'static [&'static str] =
        &["vocab_size", "block_size", "n_layers", "d_model", "n_heads", "n_kv_heads", "head_dim", "d_ff", "rope_theta", "rms_norm_eps"];

    /// Which of [`Self::SHAPE_KEYS`] `c` is missing - empty means
    /// [`Self::from_json`] on `c` reads `c`'s real shape, not a default
    /// standing in for it.
    pub fn missing_shape_keys(c: &Value) -> Vec<&'static str> {
        Self::SHAPE_KEYS.iter().filter(|k| c.get(**k).is_none()).copied().collect()
    }

    /// [`Self::from_json`], but refuses a config that would silently default
    /// any shape-defining key instead of reading it - the gap that let a
    /// config spelling the vocab key `"vocab"` instead of `"vocab_size"` get
    /// silently priced as `vocab=23` with `brain flops`/`brain models
    /// profile` reporting "100% covered, exact" for the wrong model. Every
    /// caller that prices or serves a REAL checkpoint (as opposed to
    /// building a synthetic `tiny()`-style config by hand, which has no JSON
    /// to mismatch against) should call this, not `from_json` directly.
    pub fn from_json_checked(c: &Value) -> Result<QwenConfig, String> {
        let missing = Self::missing_shape_keys(c);
        if !missing.is_empty() {
            return Err(format!(
                "config is missing shape key(s) {missing:?} - from_json would silently substitute an unrelated default for each rather than this checkpoint's real value"
            ));
        }
        Ok(Self::from_json(c))
    }

    pub fn from_json(c: &Value) -> QwenConfig {
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        let gf = |k: &str, d: f32| c[k].as_f64().map(|v| v as f32).unwrap_or(d);
        let block_size = g("block_size", 12);
        QwenConfig {
            vocab: g("vocab_size", 23),
            block_size,
            n_layers: g("n_layers", 2),
            d_model: g("d_model", 16),
            n_heads: g("n_heads", 4),
            n_kv_heads: g("n_kv_heads", 2),
            head_dim: g("head_dim", 8),
            d_ff: g("d_ff", 32),
            rope_theta: gf("rope_theta", 1.0e6),
            rms_eps: gf("rms_norm_eps", 1e-6),
            // Absent on brain checkpoints written before this field existed —
            // default to `block_size` so those keep loading unchanged.
            max_position_embeddings: g("max_position_embeddings", block_size),
            tie_embeddings: c["tie_word_embeddings"].as_bool().unwrap_or(true),
            // Qwen3 default (QK-norm on, bias-free); a Qwen2 loader sets these.
            qk_norm: c["qk_norm"].as_bool().unwrap_or(true),
            attn_bias: c["attention_bias"].as_bool().unwrap_or(false),
            lora: c.get("lora").and_then(|l| l.as_object()).map(|l| LoraCfg {
                rank: l["rank"].as_u64().unwrap_or(0) as u32,
                alpha: l["alpha"].as_f64().unwrap_or(0.0) as f32,
                targets: l["targets"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect())
                    .unwrap_or_default(),
            }),
        }
        .with_defaults()
    }

    /// Qwen2 shape helper: QK-norm **off**, qkv **bias on** (the deltas vs Qwen3).
    /// `head_dim = d_model / n_heads`. Used by FastVLM's decoder.
    pub fn qwen2(vocab: u32, n_layers: u32, d_model: u32, n_heads: u32, n_kv_heads: u32, d_ff: u32, tie: bool) -> QwenConfig {
        QwenConfig {
            vocab,
            block_size: 2048,
            n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim: d_model / n_heads,
            d_ff,
            max_position_embeddings: 2048,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            tie_embeddings: tie,
            qk_norm: false,
            attn_bias: true,
            lora: None,
        }
    }

    /// `Qwen2-0.5B` (FastVLM-0.5B decoder).
    pub fn qwen2_0_5b() -> QwenConfig {
        Self::qwen2(151936, 24, 896, 14, 2, 4864, true)
    }
    /// `Qwen2-1.5B` (FastVLM-1.5B decoder).
    pub fn qwen2_1_5b() -> QwenConfig {
        Self::qwen2(151936, 28, 1536, 12, 2, 8960, true)
    }
    /// `Qwen2-7B` (FastVLM-7B decoder; untied head, vocab 152064).
    pub fn qwen2_7b() -> QwenConfig {
        Self::qwen2(152064, 28, 3584, 28, 4, 18944, false)
    }

    /// Parameter list: `(name, numel)`. With LoRA, targeted projections become a
    /// frozen base + trainable `A`/`B` adapters (see [`crate::model`]).
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let d = self.d_model as usize;
        let ff = self.d_ff as usize;
        let v = self.vocab as usize;
        let hq = self.q_dim() as usize;
        let hkv = self.kv_dim() as usize;
        let hd = self.head_dim as usize;
        let r = self.lora.as_ref().map(|l| l.rank as usize);

        let mut out: Vec<(String, usize)> = vec![("tok.weight".to_string(), v * d)];
        // A linear `[out, in]` either as a plain trainable weight, or (LoRA on a
        // targeted leaf) as a frozen base + A[r,in] + B[out,r] adapters.
        let lin = |out_v: &mut Vec<(String, usize)>, name: String, leaf: &str, o: usize, i: usize| {
            let lora_here = r.is_some()
                && self
                    .lora
                    .as_ref()
                    .map(|l| l.targets_leaf(leaf))
                    .unwrap_or(false);
            if let (true, Some(rk)) = (lora_here, r) {
                out_v.push((name.clone(), o * i)); // frozen base
                out_v.push((format!("{name}.lora_a"), rk * i));
                out_v.push((format!("{name}.lora_b"), o * rk));
            } else {
                out_v.push((name, o * i));
            }
        };

        for l in 0..self.n_layers {
            let p = |s: &str| format!("blocks.{l}.{s}");
            out.push((p("ln1.weight"), d));
            lin(&mut out, p("attn.wq.weight"), "wq", hq, d);
            lin(&mut out, p("attn.wk.weight"), "wk", hkv, d);
            lin(&mut out, p("attn.wv.weight"), "wv", hkv, d);
            if self.attn_bias {
                // Qwen2: q/k/v projections carry a bias (o/gate/up/down do not).
                out.push((p("attn.wq.bias"), hq));
                out.push((p("attn.wk.bias"), hkv));
                out.push((p("attn.wv.bias"), hkv));
            }
            if self.qk_norm {
                out.push((p("attn.q_norm.weight"), hd));
                out.push((p("attn.k_norm.weight"), hd));
            }
            lin(&mut out, p("attn.wo.weight"), "wo", d, hq);
            out.push((p("ln2.weight"), d));
            lin(&mut out, p("mlp.gate.weight"), "gate", ff, d);
            lin(&mut out, p("mlp.up.weight"), "up", ff, d);
            lin(&mut out, p("mlp.down.weight"), "down", d, ff);
        }
        out.push(("norm.weight".to_string(), d));
        if !self.tie_embeddings {
            out.push(("lm_head.weight".to_string(), v * d));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned against the real `meta-llama/Llama-2-13b-hf` `config.json`:
    /// `hidden_size` 5120, `num_hidden_layers` 40, 40 attention heads, 40
    /// key/value heads (plain MHA), `intermediate_size` 13824, `rms_norm_eps`
    /// 1e-5, `max_position_embeddings` 4096, `tie_word_embeddings` false,
    /// vocab 32000, no `rope_scaling` (base 10000).
    #[test]
    fn llama2_13b_matches_the_published_shape() {
        let c = QwenConfig::llama2_13b().with_defaults();
        assert_eq!(c.vocab, 32000);
        assert_eq!(c.n_layers, 40);
        assert_eq!(c.d_model, 5120);
        assert_eq!(c.n_heads, 40);
        assert_eq!(c.n_kv_heads, 40, "Vicuna/LLaMA-2 is plain MHA, not GQA");
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.q_dim(), c.kv_dim(), "MHA: query and kv widths must match");
        assert_eq!(c.group(), 1, "MHA: one query head per kv head");
        assert_eq!(c.d_ff, 13824);
        assert_eq!(c.rope_theta, 10000.0);
        assert_eq!(c.rms_eps, 1e-5);
        assert_eq!(c.max_position_embeddings, 4096);
        assert!(!c.tie_embeddings, "LLaMA-2-13B has an untied lm_head");
        assert!(!c.qk_norm, "no QK-norm in the LLaMA-2 family");
        assert!(!c.attn_bias, "no attention bias in the LLaMA-2 family");
        assert_eq!(c.head_weight(), "lm_head.weight");

        // 1 embed + 40 layers x 9 (ln1, wq, wk, wv, wo, ln2, gate, up, down -
        // no qk_norm rows, no attn-bias rows) + norm + lm_head.
        let params = c.param_list();
        assert_eq!(params.len(), 1 + 40 * 9 + 1 + 1);
    }

    #[test]
    fn from_json_checked_accepts_a_real_to_json_round_trip() {
        let c = QwenConfig::tiny().to_json();
        assert!(QwenConfig::missing_shape_keys(&c).is_empty(), "to_json's own output must satisfy from_json_checked - anything else means the two have drifted apart");
        assert!(QwenConfig::from_json_checked(&c).is_ok());
    }

    #[test]
    fn from_json_checked_rejects_a_config_using_the_wrong_key_name() {
        // The real bug this guards: "vocab" instead of "vocab_size" reads as
        // present-but-irrelevant to `from_json`'s `c["vocab_size"]` lookup,
        // which silently falls back to its hardcoded default (23) rather
        // than erroring - exactly the gap that let a mis-keyed test fixture
        // get priced as a different model than the one it named.
        let c = serde_json::json!({
            "vocab": 16, "block_size": 32, "n_layers": 2, "d_model": 8,
            "n_heads": 2, "n_kv_heads": 1, "head_dim": 4,
        });
        let err = QwenConfig::from_json_checked(&c).expect_err("a config missing vocab_size/d_ff/rope_theta/rms_norm_eps must be refused");
        for key in ["vocab_size", "d_ff", "rope_theta", "rms_norm_eps"] {
            assert!(err.contains(key), "error {err:?} should name the missing key {key:?}");
        }
        // "vocab" is not one of from_json's real keys, so it must NOT be
        // reported as satisfied.
        assert!(!err.contains("\"vocab\","), "must not be fooled by the similarly-named but wrong key: {err:?}");
    }

    #[test]
    fn missing_shape_keys_does_not_flag_the_deliberately_optional_fields() {
        // max_position_embeddings (documented default: block_size) and the
        // boolean/optional fields must never appear in SHAPE_KEYS - their
        // defaults are legitimate, not a silent-wrong-shape footgun.
        for optional in ["max_position_embeddings", "tie_word_embeddings", "qk_norm", "attention_bias", "lora"] {
            assert!(!QwenConfig::SHAPE_KEYS.contains(&optional), "{optional:?} must stay optional, not become a hard requirement");
        }
    }
}
