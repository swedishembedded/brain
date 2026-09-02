// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.5/3.8-27B decoder configuration + parameter layout - the dense
//! sibling of `crates/qwen35moe::config`.
//!
//! Hybrid decoder: `full_attention_interval` layers are Gated DeltaNet
//! (linear attention - short causal depthwise conv over q/k/v, L2-normalized
//! q/k, a sigmoid `beta` gate and an `exp(-softplus(..))` decay gate feeding a
//! chunked delta-rule recurrence, gated RMSNorm output), the remainder are GQA
//! full attention with a **doubled** `q_proj` (value half + a sigmoid output
//! gate half) and `partial_rotary_factor` RoPE - byte-identical mechanisms to
//! `qwen35moe`, confirmed against the installed `transformers.models.qwen3_5`
//! reference. The one structural difference: every layer's MLP is a **plain
//! dense SwiGLU** (`gate`/`up`/`down`, no router, no experts - HF's own
//! `Qwen3_5TextConfig` deliberately deletes every MoE field from its base
//! class), not `qwen35moe`'s 256-expert MoE. Vision is a separate composed
//! tower (`crate::vl`), not part of this config, mirroring `qwen35moe::vl`'s
//! own split.
//!
//! Field names and defaults are taken directly from the real checkpoint's
//! `config.json` / the installed `transformers.models.qwen3_5.
//! configuration_qwen3_5.Qwen3_5TextConfig.__post_init__`, which hardcodes
//! `partial_rotary_factor = 0.25` and `full_attention_interval = 4` as
//! defaults not always present in a checkpoint's `config.json` - a config
//! default here mirrors the *reference's* default, not "off".

use serde_json::Value;

pub use qwen3::LoraCfg;

/// The 12 LoRA-targetable leaf names for this hybrid decoder: GDN's 5 linear
/// projections (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`,
/// `out_proj`), GQA's 4 (`q_proj`, `k_proj`, `v_proj`, `o_proj`) - same 9 as
/// `qwen35moe::config::lora_targets` - plus the dense MLP's 3
/// (`gate`, `up`, `down`), which `qwen35moe` never includes (that crate's
/// 256-expert MoE linears are deliberately excluded from LoRA; this model has
/// no experts to exclude, and `crates/qwen3`'s own dense-MLP LoRA support
/// already targets these exact leaf names, so extending here is mechanical).
pub fn lora_targets() -> Vec<String> {
    ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// [`LoraCfg`] targeting every one of [`lora_targets`] at the given rank/alpha
/// - the qwen35-dense analogue of `qwen35moe::config::lora_cfg`.
pub fn lora_cfg(rank: u32, alpha: f32) -> LoraCfg {
    LoraCfg { rank, alpha, targets: lora_targets() }
}

/// Which token-mixer a decoder layer uses. Generated from
/// `full_attention_interval` exactly like the reference's `__post_init__`:
/// layer `i` (0-indexed) is `Full` iff `(i + 1) % interval == 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    /// Gated DeltaNet linear attention.
    Linear,
    /// GQA full (softmax) attention with a sigmoid output gate.
    Full,
}

/// Build the layer-type schedule for `n_layers` at the given
/// `full_attention_interval` (reference default 4).
pub fn layer_types(n_layers: u32, interval: u32) -> Vec<LayerType> {
    assert!(interval > 0, "full_attention_interval must be > 0");
    (0..n_layers)
        .map(|i| if (i + 1) % interval == 0 { LayerType::Full } else { LayerType::Linear })
        .collect()
}

#[derive(Clone, Debug)]
pub struct Qwen35Config {
    pub vocab: u32,
    pub block_size: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub rms_eps: f32,
    pub max_position_embeddings: u32,
    pub tie_embeddings: bool,

    // -- full-attention (GQA) layer shape --
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub attn_bias: bool,
    pub rope_theta: f32,
    /// Fraction of `head_dim` that is rotated by RoPE (reference default
    /// 0.25 - only the first `head_dim * partial_rotary_factor` dims rotate,
    /// the rest pass through unrotated). Reuses `rope_partial.wgsl`.
    pub partial_rotary_factor: f32,
    /// Interleaved M-RoPE section sizes `[t, h, w]` (reference default
    /// `[11, 11, 10]`, summing to `head_dim * partial_rotary_factor / 2`).
    pub mrope_section: [u32; 3],

    // -- linear-attention (Gated DeltaNet) layer shape --
    /// Layer-type schedule period (reference default 4): every 4th layer
    /// (1-indexed) is `Full`, the rest are `Linear`.
    pub full_attention_interval: u32,
    pub linear_num_key_heads: u32,
    pub linear_num_value_heads: u32,
    pub linear_key_head_dim: u32,
    pub linear_value_head_dim: u32,
    pub linear_conv_kernel_dim: u32,

    // -- dense MLP (every layer, both mixer types) --
    pub intermediate_size: u32,

    /// `Some` selects LoRA fine-tuning (frozen base + adapters); `None` is a
    /// full (all-parameter) model.
    pub lora: Option<LoraCfg>,

    /// Multi-token prediction head (`mtp_num_hidden_layers: 1` in the real
    /// config - always exactly one extra layer, so this is a plain bool, not
    /// a layer count). The real checkpoint carries `mtp.*` tensors, but
    /// `transformers`' own loader discards them
    /// (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`) - there is no
    /// reference oracle for this head on this box, so it is gradchecked and
    /// overfit-tested but never parity-claimed. `false` by default - matches
    /// `qwen35moe`, whose own MTP support is deferred entirely; `true` opts
    /// into `param_list()`'s `mtp.*` tensors and `Qwen35::run_forward`'s MTP
    /// forward pass.
    pub mtp: bool,
}

impl Qwen35Config {
    /// A tiny hybrid config for tests / gradient checks / the goldens parity
    /// suite - **must match `tools/goldens/qwen35_dump_reference.py`'s
    /// `TINY_TEXT` exactly**, dimension for dimension, so the golden and this
    /// config agree by construction rather than by a hand-maintained parallel
    /// list. Every dimension that is distinct in the real config stays
    /// distinct here (a degenerate/collapsed toy dim hides whole bug classes:
    /// the real config's `head_dim == linear_key_head_dim ==
    /// linear_value_head_dim == 128` would let a head-width/head-count swap
    /// pass at cosine 1.0; this tiny config avoids that coincidence on
    /// purpose, see `qwen35_tiny_dims_are_pairwise_distinct` below).
    /// `full_attention_interval = 4` with `n_layers = 4` exercises both layer
    /// types (only layer 3 is `Full`).
    pub fn tiny() -> Qwen35Config {
        Qwen35Config {
            vocab: 29,
            block_size: 24,
            n_layers: 4,
            d_model: 96,
            rms_eps: 1e-6,
            max_position_embeddings: 24,
            tie_embeddings: false,

            n_heads: 3,
            n_kv_heads: 1,
            head_dim: 40,
            attn_bias: false,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            mrope_section: [2, 2, 1],

            full_attention_interval: 4,
            linear_num_key_heads: 2,
            linear_num_value_heads: 6,
            linear_key_head_dim: 16,
            linear_value_head_dim: 20,
            linear_conv_kernel_dim: 4,

            intermediate_size: 112,

            lora: None,
            mtp: false,
        }
    }

    /// [`Self::tiny`] made int8-legal: the SAME shape (4 layers,
    /// `interval = 4` so layer 3 is GQA and the rest GDN, a multi-chunk GDN
    /// sequence, every dimension still distinct from every other) with each
    /// quantized linear's CONTRACTION dim rounded to a multiple of
    /// `model::int8::GROUP` (32).
    ///
    /// `tiny()` itself cannot be used with the int8 tier:
    /// `model::int8::quantize_weight` scales per 32-element group of the
    /// contraction axis and asserts `k % GROUP == 0`, which `tiny()`'s
    /// `q_dim = 120`, `linear_value_dim = 120` and `intermediate_size = 112`
    /// all fail - and `tiny()` must not grow to satisfy a constraint the fp32
    /// smoke suite, the goldens and the gradient checker do not have (it is
    /// pinned dimension-for-dimension to
    /// `tools/goldens/qwen35_dump_reference.py`'s `TINY_TEXT`).
    ///
    /// `head_dim` 40 -> 32 also halves `rotary_dim` from 10 to 8, so
    /// `mrope_section` (which must sum to `rotary_dim/2`) goes from `[2,2,1]`
    /// to `[2,1,1]`. `crates/qwen35/tests/model_i8_smoke.rs` asserts every one
    /// of those group constraints against this fixture directly.
    pub fn tiny_i8() -> Qwen35Config {
        Qwen35Config {
            head_dim: 32,              // q_dim = 3 * 32 = 96
            mrope_section: [2, 1, 1],  // sums to rotary_dim/2 = 4
            linear_value_head_dim: 16, // linear_value_dim = 6 * 16 = 96
            intermediate_size: 128,
            ..Qwen35Config::tiny()
        }
    }

    /// The published Qwen3.8-27B shape (from its real `config.json` -
    /// `Qwen/Qwen3.8-27B-FP8`; the HF module itself still cites
    /// `Qwen/Qwen3.5-27B` in its docstrings, since `model_type: "qwen3_5"`
    /// predates the "3.8" release naming).
    pub fn qwen38_27b() -> Qwen35Config {
        Qwen35Config {
            vocab: 248320,
            block_size: 4096,
            n_layers: 64,
            d_model: 5120,
            rms_eps: 1e-6,
            max_position_embeddings: 262144,
            tie_embeddings: false,

            n_heads: 24,
            n_kv_heads: 4,
            head_dim: 256,
            attn_bias: false,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            mrope_section: [11, 11, 10],

            full_attention_interval: 4,
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,

            intermediate_size: 17408,

            lora: None,
            mtp: false,
        }
    }

    pub fn layer_types(&self) -> Vec<LayerType> {
        layer_types(self.n_layers, self.full_attention_interval)
    }

    // -- full-attention (GQA) derived shapes --
    /// Query projection width, **doubled** for the value+gate split
    /// (`Qwen3_5Attention.q_proj` emits `num_heads * head_dim * 2`).
    pub fn q_proj_dim(&self) -> u32 {
        self.n_heads * self.head_dim * 2
    }
    /// Query *value* width after the gate split (`num_heads * head_dim`).
    pub fn q_dim(&self) -> u32 {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }
    pub fn group(&self) -> u32 {
        self.n_heads / self.n_kv_heads
    }
    /// Number of rotated dims per head (`head_dim * partial_rotary_factor`).
    pub fn rotary_dim(&self) -> u32 {
        ((self.head_dim as f32) * self.partial_rotary_factor).round() as u32
    }

    // -- linear-attention (Gated DeltaNet) derived shapes --
    pub fn linear_key_dim(&self) -> u32 {
        self.linear_num_key_heads * self.linear_key_head_dim
    }
    pub fn linear_value_dim(&self) -> u32 {
        self.linear_num_value_heads * self.linear_value_head_dim
    }
    /// `in_proj_qkv` output width: `2*key_dim + value_dim` (q and k share
    /// `linear_key_head_dim`, v uses `linear_value_head_dim`).
    pub fn linear_conv_dim(&self) -> u32 {
        2 * self.linear_key_dim() + self.linear_value_dim()
    }
    /// GQA-style repeat factor for the linear-attention heads
    /// (`num_v_heads / num_k_heads`, e.g. 48/16 = 3 at the real scale).
    pub fn linear_group(&self) -> u32 {
        self.linear_num_value_heads / self.linear_num_key_heads
    }

    /// The `(suffix, n, k)` triple for every quantizable per-layer leaf of
    /// `ty`: the 5 GDN or 4 GQA mixer-adjacent leaves, plus the 3 dense-MLP
    /// leaves every layer type owns (`n`, `k` in `Weight::upload`'s `[n, k]`
    /// row-major sense). `suffix` is what follows `blocks.{l}.` in the
    /// checkpoint's own naming.
    ///
    /// The ONE table both `Self::layer_weight_bytes` (the byte-cost formula)
    /// and `Qwen35::new_impl_on`'s upload loop (the actual dispatch) read -
    /// replacing a formula that once drifted from what `Weight::upload`
    /// really placed on the card by 12.5% (lesson #68). A hand-transcribed
    /// SECOND copy is exactly the failure mode this table exists to remove,
    /// not merely re-gate.
    pub fn layer_leaves(&self, ty: LayerType) -> Vec<(&'static str, u32, u32)> {
        let d = self.d_model;
        let ff = self.intermediate_size;
        let mut leaves = match ty {
            LayerType::Linear => {
                let conv_dim = self.linear_conv_dim();
                let value_dim = self.linear_value_dim();
                let nvh = self.linear_num_value_heads;
                vec![
                    ("linear_attn.in_proj_qkv.weight", conv_dim, d),
                    ("linear_attn.in_proj_z.weight", value_dim, d),
                    ("linear_attn.in_proj_b.weight", nvh, d),
                    ("linear_attn.in_proj_a.weight", nvh, d),
                    ("linear_attn.out_proj.weight", d, value_dim),
                ]
            }
            LayerType::Full => {
                let hqp = self.q_proj_dim();
                let hkv = self.kv_dim();
                let hq = self.q_dim();
                vec![
                    ("self_attn.q_proj.weight", hqp, d),
                    ("self_attn.k_proj.weight", hkv, d),
                    ("self_attn.v_proj.weight", hkv, d),
                    ("self_attn.o_proj.weight", d, hq),
                ]
            }
        };
        leaves.push(("mlp.gate.weight", ff, d));
        leaves.push(("mlp.up.weight", ff, d));
        leaves.push(("mlp.down.weight", d, ff));
        leaves
    }

    /// Real on-device byte cost of ONE streamed layer's weights at this
    /// config's dims, under `tier` - folds [`Self::layer_leaves`] through
    /// each leaf's own [`TierPolicy::want`] tier. Mirrored here (rather than
    /// read off a live `Weight`) for a host-side perf model (`crates/perf`'s
    /// `weights` scenario, `crate::int8_gguf_resident`'s cross-GPU placement)
    /// that has no GPU and must never build one just to learn a byte count.
    /// Gated against the real `model::int8::quantize_weight`/
    /// `model::int4::quantize_weight_q4` output by
    /// `layer_weight_bytes_equals_what_weight_upload_really_places_on_the_card`,
    /// so this cannot silently drift from what it mirrors (lesson #68).
    /// Excludes the handful of small fp32 aux tensors (norms, GDN's
    /// `A_log`/`dt_bias`, GQA's `q_norm`/`k_norm`) - negligible next to the
    /// quantized leaves at this config's real dims (well under 0.1% of a
    /// layer's own footprint) and never selected by a `TierPolicy` rule
    /// (they are not in [`Self::layer_leaves`]'s table).
    pub fn layer_weight_bytes(&self, ty: LayerType, tier: &model::ops::TierPolicy) -> u64 {
        let group = model::int8::GROUP as u64;
        self.layer_leaves(ty)
            .into_iter()
            .map(|(suffix, n, k)| {
                let (n, k) = (u64::from(n), u64::from(k));
                match tier.want(suffix) {
                    gpu_core::select::Dtype::F32 => n * k * 4,
                    gpu_core::select::Dtype::BF16 | gpu_core::select::Dtype::F16 => n * k * 2,
                    gpu_core::select::Dtype::I8 => n * k + n * k.div_ceil(group) * 4,
                    gpu_core::select::Dtype::Q4 => n * k / 2 + n * k.div_ceil(group) * 4,
                }
            })
            .sum()
    }

    pub fn head_weight(&self) -> &'static str {
        if self.tie_embeddings {
            "tok.weight"
        } else {
            "lm_head.weight"
        }
    }

    pub fn to_json(&self) -> Value {
        let mut v = serde_json::json!({
            "model": "qwen35",
            "vocab_size": self.vocab, "block_size": self.block_size, "n_layers": self.n_layers,
            "d_model": self.d_model, "rms_norm_eps": self.rms_eps,
            "max_position_embeddings": self.max_position_embeddings,
            "tie_word_embeddings": self.tie_embeddings,
            "n_heads": self.n_heads, "n_kv_heads": self.n_kv_heads, "head_dim": self.head_dim,
            "attention_bias": self.attn_bias, "rope_theta": self.rope_theta,
            "partial_rotary_factor": self.partial_rotary_factor,
            "mrope_section": self.mrope_section,
            "full_attention_interval": self.full_attention_interval,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "intermediate_size": self.intermediate_size,
            "mtp": self.mtp,
        });
        // A LoRA checkpoint must round-trip its adapter shape, or `param_list()`
        // rebuilds without the `.lora_a`/`.lora_b` names on load and the trained
        // adapters are silently dropped.
        if let Some(l) = &self.lora {
            v["lora"] = serde_json::json!({
                "rank": l.rank, "alpha": l.alpha, "targets": l.targets,
            });
        }
        v
    }

    /// Every JSON key [`Self::from_json`] must find to read this config's
    /// real per-checkpoint SHAPE rather than silently substitute an
    /// unrelated hardcoded default - see that function's own `g`/`gf`
    /// closures. Deliberately excludes fields whose defaults are a
    /// documented REFERENCE CONSTANT for this whole architecture family
    /// (`partial_rotary_factor`, `mrope_section`, `full_attention_interval`,
    /// the `linear_*` GDN dims, `max_position_embeddings`), not a
    /// per-checkpoint value that varies by model size - unlike a missing
    /// `vocab_size`/`d_model`/`n_layers`, which produces a DIFFERENT model,
    /// silently.
    pub const SHAPE_KEYS: &'static [&'static str] =
        &["vocab_size", "block_size", "n_layers", "d_model", "n_heads", "n_kv_heads", "head_dim", "rms_norm_eps", "rope_theta"];

    /// Which of [`Self::SHAPE_KEYS`] `c` is missing.
    pub fn missing_shape_keys(c: &Value) -> Vec<&'static str> {
        Self::SHAPE_KEYS.iter().filter(|k| c.get(**k).is_none()).copied().collect()
    }

    /// [`Self::from_json`], but refuses a config that would silently default
    /// any shape-defining key instead of reading it.
    pub fn from_json_checked(c: &Value) -> Result<Qwen35Config, String> {
        let missing = Self::missing_shape_keys(c);
        if !missing.is_empty() {
            return Err(format!(
                "config is missing shape key(s) {missing:?} - from_json would silently substitute an unrelated default for each rather than this checkpoint's real value"
            ));
        }
        Ok(Self::from_json(c))
    }

    pub fn from_json(c: &Value) -> Qwen35Config {
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        let gf = |k: &str, d: f32| c[k].as_f64().map(|v| v as f32).unwrap_or(d);
        let block_size = g("block_size", 24);
        let mrope = c["mrope_section"]
            .as_array()
            .map(|a| {
                let v: Vec<u32> = a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect();
                [v.first().copied().unwrap_or(11), v.get(1).copied().unwrap_or(11), v.get(2).copied().unwrap_or(10)]
            })
            .unwrap_or([11, 11, 10]);
        Qwen35Config {
            vocab: g("vocab_size", 29),
            block_size,
            n_layers: g("n_layers", 4),
            d_model: g("d_model", 96),
            rms_eps: gf("rms_norm_eps", 1e-6),
            max_position_embeddings: g("max_position_embeddings", block_size),
            tie_embeddings: c["tie_word_embeddings"].as_bool().unwrap_or(false),

            n_heads: g("n_heads", 3),
            n_kv_heads: g("n_kv_heads", 1),
            head_dim: g("head_dim", 40),
            attn_bias: c["attention_bias"].as_bool().unwrap_or(false),
            rope_theta: gf("rope_theta", 1.0e7),
            // Reference default -- a config.json that
            // predates this field still means 0.25, not "unset -> full RoPE".
            partial_rotary_factor: gf("partial_rotary_factor", 0.25),
            mrope_section: mrope,

            full_attention_interval: g("full_attention_interval", 4),
            linear_num_key_heads: g("linear_num_key_heads", 2),
            linear_num_value_heads: g("linear_num_value_heads", 6),
            linear_key_head_dim: g("linear_key_head_dim", 16),
            linear_value_head_dim: g("linear_value_head_dim", 20),
            linear_conv_kernel_dim: g("linear_conv_kernel_dim", 4),

            intermediate_size: g("intermediate_size", 112),

            lora: c.get("lora").and_then(|l| l.as_object()).map(|l| LoraCfg {
                rank: l["rank"].as_u64().unwrap_or(0) as u32,
                alpha: l["alpha"].as_f64().unwrap_or(0.0) as f32,
                targets: l["targets"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect())
                    .unwrap_or_default(),
            }),
            mtp: c["mtp"].as_bool().unwrap_or(false),
        }
    }

    /// Parameter list: `(name, numel)`. Mirrors `qwen35moe::config::
    /// Qwen35Config::param_list`'s GDN/GQA shapes exactly (both crates'
    /// hybrid-decoder mixers are the same mechanism), but every layer's MLP
    /// is the plain dense `gate`/`up`/`down` SwiGLU (`crates/qwen3::config`'s
    /// own naming for the same primitive) instead of a router + per-expert
    /// bank - this model has no MoE fields at all (HF's own
    /// `Qwen3_5TextConfig` deletes them from its base class).
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let d = self.d_model as usize;
        let v = self.vocab as usize;
        let r = self.lora.as_ref().map(|l| l.rank as usize);
        // A linear `[out, in]` either as a plain trainable weight, or (LoRA on
        // a targeted leaf) as a frozen base + A[r,in] + B[out,r] adapters -
        // same shape as `qwen35moe::config::Qwen35Config::param_list`'s `lin`.
        let lin = |out_v: &mut Vec<(String, usize)>, name: String, leaf: &str, o: usize, i: usize| {
            let lora_here = r.is_some() && self.lora.as_ref().map(|l| l.targets_leaf(leaf)).unwrap_or(false);
            if let (true, Some(rk)) = (lora_here, r) {
                out_v.push((name.clone(), o * i)); // frozen base
                out_v.push((format!("{name}.lora_a"), rk * i));
                out_v.push((format!("{name}.lora_b"), o * rk));
            } else {
                out_v.push((name, o * i));
            }
        };
        let mut out: Vec<(String, usize)> = vec![("tok.weight".to_string(), v * d)];

        let types = self.layer_types();
        let ff = self.intermediate_size as usize;
        for (l, ty) in types.iter().enumerate() {
            let p = |s: &str| format!("blocks.{l}.{s}");
            out.push((p("ln1.weight"), d));

            match ty {
                LayerType::Linear => {
                    let kdim = self.linear_key_dim() as usize;
                    let vdim = self.linear_value_dim() as usize;
                    let conv_dim = self.linear_conv_dim() as usize;
                    let k = self.linear_conv_kernel_dim as usize;
                    let nvh = self.linear_num_value_heads as usize;
                    let hvd = self.linear_value_head_dim as usize;
                    lin(&mut out, p("linear_attn.in_proj_qkv.weight"), "in_proj_qkv", conv_dim, d);
                    lin(&mut out, p("linear_attn.in_proj_z.weight"), "in_proj_z", vdim, d);
                    lin(&mut out, p("linear_attn.in_proj_b.weight"), "in_proj_b", nvh, d);
                    lin(&mut out, p("linear_attn.in_proj_a.weight"), "in_proj_a", nvh, d);
                    out.push((p("linear_attn.conv1d.weight"), conv_dim * k));
                    out.push((p("linear_attn.A_log"), nvh));
                    out.push((p("linear_attn.dt_bias"), nvh));
                    out.push((p("linear_attn.norm.weight"), hvd));
                    lin(&mut out, p("linear_attn.out_proj.weight"), "out_proj", d, vdim);
                    let _ = kdim; // kept for readability of the shape derivation above
                }
                LayerType::Full => {
                    let hq = self.q_dim() as usize;
                    let hqp = self.q_proj_dim() as usize;
                    let hkv = self.kv_dim() as usize;
                    let hd = self.head_dim as usize;
                    lin(&mut out, p("self_attn.q_proj.weight"), "q_proj", hqp, d);
                    lin(&mut out, p("self_attn.k_proj.weight"), "k_proj", hkv, d);
                    lin(&mut out, p("self_attn.v_proj.weight"), "v_proj", hkv, d);
                    out.push((p("self_attn.q_norm.weight"), hd));
                    out.push((p("self_attn.k_norm.weight"), hd));
                    lin(&mut out, p("self_attn.o_proj.weight"), "o_proj", d, hq);
                }
            }

            out.push((p("ln2.weight"), d));

            lin(&mut out, p("mlp.gate.weight"), "gate", ff, d);
            lin(&mut out, p("mlp.up.weight"), "up", ff, d);
            lin(&mut out, p("mlp.down.weight"), "down", d, ff);
        }

        out.push(("norm.weight".to_string(), d));
        if !self.tie_embeddings {
            out.push(("lm_head.weight".to_string(), v * d));
        }

        // Multi-token prediction head - one extra FULL Gated-Attention
        // decoder layer (same self_attn/mlp shapes as any other `Full`
        // block, real config's own `mtp.layers.0.*`), fed by concatenating
        // the next token's own embedding and the final hidden state
        // (`mtp.fc [d,2d]` split at import into `fc_e`/`fc_h`, each `[d,d]`
        // - see `crate::import`'s module doc). `tok.weight`/`lm_head.weight`
        // are SHARED with the main head, not duplicated here.
        if self.mtp {
            out.push(("mtp.pre_fc_norm_embedding.weight".to_string(), d));
            out.push(("mtp.pre_fc_norm_hidden.weight".to_string(), d));
            lin(&mut out, "mtp.fc_e.weight".to_string(), "fc_e", d, d);
            lin(&mut out, "mtp.fc_h.weight".to_string(), "fc_h", d, d);

            out.push(("mtp.layers.0.ln1.weight".to_string(), d));
            let hq = self.q_dim() as usize;
            let hqp = self.q_proj_dim() as usize;
            let hkv = self.kv_dim() as usize;
            let hd = self.head_dim as usize;
            lin(&mut out, "mtp.layers.0.self_attn.q_proj.weight".to_string(), "q_proj", hqp, d);
            lin(&mut out, "mtp.layers.0.self_attn.k_proj.weight".to_string(), "k_proj", hkv, d);
            lin(&mut out, "mtp.layers.0.self_attn.v_proj.weight".to_string(), "v_proj", hkv, d);
            out.push(("mtp.layers.0.self_attn.q_norm.weight".to_string(), hd));
            out.push(("mtp.layers.0.self_attn.k_norm.weight".to_string(), hd));
            lin(&mut out, "mtp.layers.0.self_attn.o_proj.weight".to_string(), "o_proj", d, hq);
            out.push(("mtp.layers.0.ln2.weight".to_string(), d));
            lin(&mut out, "mtp.layers.0.mlp.gate.weight".to_string(), "gate", ff, d);
            lin(&mut out, "mtp.layers.0.mlp.up.weight".to_string(), "up", ff, d);
            lin(&mut out, "mtp.layers.0.mlp.down.weight".to_string(), "down", d, ff);

            out.push(("mtp.norm.weight".to_string(), d));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_json_checked_accepts_a_real_to_json_round_trip() {
        let c = Qwen35Config::tiny().to_json();
        assert!(Qwen35Config::missing_shape_keys(&c).is_empty());
        assert!(Qwen35Config::from_json_checked(&c).is_ok());
    }

    #[test]
    fn from_json_checked_rejects_a_config_using_the_wrong_key_name() {
        let c = serde_json::json!({"vocab": 29, "block_size": 24, "n_layers": 4});
        let err = Qwen35Config::from_json_checked(&c).expect_err("missing vocab_size and friends must be refused");
        assert!(err.contains("vocab_size"), "error {err:?} should name vocab_size");
    }

    #[test]
    fn layer_type_schedule_matches_reference_formula() {
        // Reference: "linear_attention" if bool((i+1) % interval) else "full_attention"
        // -> full at i = 3, 7, 11, ... (0-indexed) for interval=4.
        let types = layer_types(8, 4);
        let expect = [
            LayerType::Linear,
            LayerType::Linear,
            LayerType::Linear,
            LayerType::Full,
            LayerType::Linear,
            LayerType::Linear,
            LayerType::Linear,
            LayerType::Full,
        ];
        assert_eq!(types, expect);
    }

    #[test]
    fn real_config_layer_types_match_checkpoint() {
        // Cross-checked against the real config.json's explicit `layer_types`
        // list (64 layers, full at 1-indexed multiples of 4).
        let cfg = Qwen35Config::qwen38_27b();
        let types = cfg.layer_types();
        assert_eq!(types.len(), 64);
        let full_idx: Vec<usize> =
            types.iter().enumerate().filter(|(_, t)| **t == LayerType::Full).map(|(i, _)| i).collect();
        assert_eq!(full_idx, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]);
    }

    #[test]
    fn tiny_config_matches_the_golden_dumper_dims_exactly() {
        // Must match tools/goldens/qwen35_dump_reference.py's TINY_TEXT
        // dict, field for field, so the golden and this config agree by
        // construction.
        let cfg = Qwen35Config::tiny();
        assert_eq!(cfg.vocab, 29);
        assert_eq!(cfg.block_size, 24);
        assert_eq!(cfg.n_layers, 4);
        assert_eq!(cfg.d_model, 96);
        assert_eq!(cfg.n_heads, 3);
        assert_eq!(cfg.n_kv_heads, 1);
        assert_eq!(cfg.head_dim, 40);
        assert_eq!(cfg.linear_num_key_heads, 2);
        assert_eq!(cfg.linear_num_value_heads, 6);
        assert_eq!(cfg.linear_key_head_dim, 16);
        assert_eq!(cfg.linear_value_head_dim, 20);
        assert_eq!(cfg.linear_conv_kernel_dim, 4);
        assert_eq!(cfg.intermediate_size, 112);
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(cfg.mrope_section, [2, 2, 1]);
        assert_eq!(cfg.partial_rotary_factor, 0.25);
        assert_eq!(cfg.rope_theta, 10_000_000.0);
        assert!(!cfg.tie_embeddings);
        // The dumper's own `assert_dims_distinct` pins this on the Python
        // side; re-derive it here so a future edit to either file cannot
        // silently reintroduce a collision undetected by the OTHER language.
        let dims = [
            cfg.d_model,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
            cfg.intermediate_size,
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
            cfg.vocab,
            cfg.n_layers,
            cfg.block_size,
        ];
        for i in 0..dims.len() {
            for j in (i + 1)..dims.len() {
                assert_ne!(dims[i], dims[j], "tiny dims collide at indices {i},{j}: {dims:?}");
            }
        }
    }

    #[test]
    fn json_round_trip_preserves_every_field() {
        let cfg = Qwen35Config::qwen38_27b();
        let back = Qwen35Config::from_json(&cfg.to_json());
        assert_eq!(back.vocab, cfg.vocab);
        assert_eq!(back.n_layers, cfg.n_layers);
        assert_eq!(back.d_model, cfg.d_model);
        assert_eq!(back.n_heads, cfg.n_heads);
        assert_eq!(back.n_kv_heads, cfg.n_kv_heads);
        assert_eq!(back.head_dim, cfg.head_dim);
        assert_eq!(back.partial_rotary_factor, cfg.partial_rotary_factor);
        assert_eq!(back.mrope_section, cfg.mrope_section);
        assert_eq!(back.full_attention_interval, cfg.full_attention_interval);
        assert_eq!(back.linear_num_key_heads, cfg.linear_num_key_heads);
        assert_eq!(back.linear_num_value_heads, cfg.linear_num_value_heads);
        assert_eq!(back.linear_key_head_dim, cfg.linear_key_head_dim);
        assert_eq!(back.linear_value_head_dim, cfg.linear_value_head_dim);
        assert_eq!(back.linear_conv_kernel_dim, cfg.linear_conv_kernel_dim);
        assert_eq!(back.intermediate_size, cfg.intermediate_size);
        assert_eq!(back.tie_embeddings, cfg.tie_embeddings);
    }

    #[test]
    fn lora_round_trips_through_json() {
        let mut cfg = Qwen35Config::tiny();
        cfg.lora = Some(lora_cfg(8, 16.0));
        let back = Qwen35Config::from_json(&cfg.to_json());
        let lora = back.lora.expect("lora must round-trip");
        assert_eq!(lora.rank, 8);
        assert_eq!(lora.alpha, 16.0);
        assert_eq!(lora.targets, lora_targets());
    }

    #[test]
    fn dense_mlp_has_no_expert_or_router_tensors() {
        // The defining structural difference from qwen35moe: no
        // `.mlp.experts.`, no `.mlp.router.`, no `.shared_expert` anywhere.
        let cfg = Qwen35Config::qwen38_27b();
        let names = cfg.param_list();
        assert!(names.iter().all(|(n, _)| !n.contains("expert") && !n.contains("router")));
        // Every layer gets exactly one dense gate/up/down triple.
        let gate_tensors = names.iter().filter(|(n, _)| n.ends_with("mlp.gate.weight")).count();
        assert_eq!(gate_tensors, 64);
    }

    #[test]
    fn mtp_off_by_default_and_adds_no_tensors() {
        let cfg = Qwen35Config::tiny();
        assert!(!cfg.mtp);
        assert!(cfg.param_list().iter().all(|(n, _)| !n.starts_with("mtp.")));
    }

    #[test]
    fn mtp_on_adds_exactly_the_expected_tensors_no_duplicate_head_or_embedding() {
        let cfg = Qwen35Config { mtp: true, ..Qwen35Config::tiny() };
        let names: Vec<String> = cfg.param_list().into_iter().map(|(n, _)| n).collect();
        let mtp_names: Vec<&String> = names.iter().filter(|n| n.starts_with("mtp.")).collect();
        let expect = [
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.fc_e.weight",
            "mtp.fc_h.weight",
            "mtp.layers.0.ln1.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.ln2.weight",
            "mtp.layers.0.mlp.gate.weight",
            "mtp.layers.0.mlp.up.weight",
            "mtp.layers.0.mlp.down.weight",
            "mtp.norm.weight",
        ];
        for e in expect {
            assert!(mtp_names.iter().any(|n| n.as_str() == e), "missing {e}, got {mtp_names:?}");
        }
        assert_eq!(mtp_names.len(), expect.len(), "unexpected extra mtp.* tensors: {mtp_names:?}");
        // tok.weight/lm_head.weight stay singular - MTP shares the main head.
        assert_eq!(names.iter().filter(|n| n.as_str() == "tok.weight").count(), 1);
        assert_eq!(names.iter().filter(|n| n.as_str() == "lm_head.weight").count(), 1);
    }

    #[test]
    fn mtp_round_trips_through_json() {
        let cfg = Qwen35Config { mtp: true, ..Qwen35Config::tiny() };
        let back = Qwen35Config::from_json(&cfg.to_json());
        assert!(back.mtp);
    }

    #[test]
    fn mtp_self_attn_leaves_are_lora_targetable_like_a_normal_full_layer() {
        let cfg = Qwen35Config { mtp: true, lora: Some(lora_cfg(4, 8.0)), ..Qwen35Config::tiny() };
        let names: Vec<String> = cfg.param_list().into_iter().map(|(n, _)| n).collect();
        assert!(names.iter().any(|n| n == "mtp.layers.0.self_attn.q_proj.weight.lora_a"));
        assert!(names.iter().any(|n| n == "mtp.layers.0.mlp.down.weight.lora_a"));
    }

    #[test]
    fn lora_on_dense_mlp_leaves_adds_adapter_tensors() {
        // qwen35moe never LoRAs its MoE experts; this model has no experts
        // to exclude, so `gate`/`up`/`down` route through the same `lin`
        // fork as every other targeted leaf.
        let mut cfg = Qwen35Config::tiny();
        cfg.lora = Some(lora_cfg(4, 8.0));
        let names: Vec<String> = cfg.param_list().into_iter().map(|(n, _)| n).collect();
        assert!(names.iter().any(|n| n == "blocks.0.mlp.gate.weight.lora_a"));
        assert!(names.iter().any(|n| n == "blocks.0.mlp.gate.weight.lora_b"));
        assert!(names.iter().any(|n| n == "blocks.0.mlp.down.weight.lora_a"));
    }

    /// Pins [`Qwen35Config::layer_weight_bytes`]'s real numbers at the real
    /// `qwen38_27b` scale, uniform INT8 - `crates/perf`'s `weights` scenario
    /// and `crate::int8_gguf_resident`'s cross-GPU placement both depend on
    /// these being accurate, not merely plausible - a silent drift here would
    /// make that scenario's "real per-layer byte profile" claim false and
    /// would let the resident under-budget a card. GDN (`Linear`) is the
    /// larger of the two layer types at this config's dims (more/wider
    /// mixer-adjacent leaves than GQA's 4), both in the ~419-431 MB range.
    #[test]
    fn layer_weight_bytes_matches_the_real_measured_qwen38_27b_range_at_i8() {
        let cfg = Qwen35Config::qwen38_27b();
        let i8 = model::ops::TierPolicy::uniform(gpu_core::select::Dtype::I8);
        let gdn = cfg.layer_weight_bytes(LayerType::Linear, &i8);
        let gqa = cfg.layer_weight_bytes(LayerType::Full, &i8);
        assert_eq!(gdn, 431_124_480, "GDN (Linear) layer int8 byte cost drifted: {gdn}");
        assert_eq!(gqa, 418_775_040, "GQA (Full) layer int8 byte cost drifted: {gqa}");
        assert!(gdn > gqa, "GDN must be the larger layer type at this config's real dims");
        for b in [gdn, gqa] {
            let mb = b as f64 / 1e6;
            assert!((418.0..=432.0).contains(&mb), "layer_weight_bytes {mb:.1} MB outside the documented 419-431 MB range");
        }
    }

    /// GROUND TRUTH for [`Qwen35Config::layer_weight_bytes`]: the number it
    /// returns must be what `model::ops::Weight::upload` ACTUALLY places on
    /// the card for that layer's leaves, at every tier this policy can name -
    /// not a hand-transcribed formula that once matched. Measured here by
    /// running the real `model::int8::quantize_weight` /
    /// `model::int4::quantize_weight_q4` (the same calls `Weight::upload`
    /// makes) over each leaf's real shape and summing the buffers they
    /// return.
    ///
    /// Uses [`Qwen35Config::tiny_i8`] (the int8/q4-legal tiny fixture) so the
    /// quantization is real and cheap; the formula is dimension-generic, so
    /// agreeing here is agreeing at every scale, and the sibling test above
    /// pins the real 27B numbers the I8 tier follows from.
    ///
    /// The I8 arm is the check that was missing before M24: the scale used to
    /// be `[n]` (one per output row) and the formula still charged `n * 4`
    /// long after `model::int8::GROUP`-wise scales made the real cost
    /// `n * (k/32) * 4`, i.e. 12.5% of the packed weight rather than a
    /// rounding error (lesson #68). The Q4 arm is new coverage: Q4 packs
    /// 8 nibbles/`u32` (half I8's bytes) but shares I8's group-wise scale
    /// layout exactly, so a formula that got the packed-weight halving right
    /// but reused I8's scale count (or vice versa) would still drift
    /// silently without this arm.
    #[test]
    fn layer_weight_bytes_equals_what_weight_upload_really_places_on_the_card() {
        let cfg = Qwen35Config::tiny_i8();
        let (d, ff) = (cfg.d_model as usize, cfg.intermediate_size as usize);
        for (tier, measured_leaf) in [
            (
                model::ops::TierPolicy::uniform(gpu_core::select::Dtype::I8),
                (|n: usize, k: usize| -> u64 {
                    let w = vec![0.5f32; n * k];
                    let (packed, scales) = model::int8::quantize_weight(&w, n, k);
                    (packed.len() + scales.len()) as u64 * 4
                }) as fn(usize, usize) -> u64,
            ),
            (
                model::ops::TierPolicy::uniform(gpu_core::select::Dtype::Q4),
                (|n: usize, k: usize| -> u64 {
                    let w = vec![0.5f32; n * k];
                    let (packed, scales) = model::int4::quantize_weight_q4(&w, n, k);
                    (packed.len() + scales.len()) as u64 * 4
                }) as fn(usize, usize) -> u64,
            ),
        ] {
            let mlp = measured_leaf(ff, d) * 2 + measured_leaf(d, ff);
            let gdn = measured_leaf(cfg.linear_conv_dim() as usize, d)
                + measured_leaf(cfg.linear_value_dim() as usize, d)
                + measured_leaf(cfg.linear_num_value_heads as usize, d) * 2
                + measured_leaf(d, cfg.linear_value_dim() as usize);
            let gqa = measured_leaf(cfg.q_proj_dim() as usize, d)
                + measured_leaf(cfg.kv_dim() as usize, d) * 2
                + measured_leaf(d, cfg.q_dim() as usize);
            assert_eq!(
                cfg.layer_weight_bytes(LayerType::Linear, &tier),
                gdn + mlp,
                "GDN layer cost disagrees with the real quantized footprint at {tier:?}"
            );
            assert_eq!(
                cfg.layer_weight_bytes(LayerType::Full, &tier),
                gqa + mlp,
                "GQA layer cost disagrees with the real quantized footprint at {tier:?}"
            );
        }
    }

    /// Two-way coverage between [`Qwen35Config::layer_leaves`] and
    /// `crate::model::is_quantizable_linear`: every leaf the table names must
    /// be recognised as quantizable by the model's own upload-time filter,
    /// and every suffix that filter recognises must appear in the table for
    /// SOME layer type - so the byte-cost table and the actual upload
    /// decision cannot silently diverge (the second copy of exactly the
    /// drift lesson #68 already caught once).
    #[test]
    fn layer_leaves_and_is_quantizable_linear_agree_on_every_leaf() {
        let cfg = Qwen35Config::qwen38_27b();
        let mut named: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
        for ty in [LayerType::Linear, LayerType::Full] {
            for (suffix, _, _) in cfg.layer_leaves(ty) {
                assert!(
                    crate::model::is_quantizable_linear(&format!("blocks.0.{suffix}")),
                    "layer_leaves names {suffix:?} but is_quantizable_linear does not recognise it"
                );
                named.insert(suffix);
            }
        }
        assert_eq!(named.len(), 12, "GDN's 5 + GQA's 4 + MLP's 3, counted once each (MLP is shared, not duplicated)");
    }
}
