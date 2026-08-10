// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.5-35B-A3B decoder configuration + parameter layout.
//!
//! Hybrid decoder: `full_attention_interval` layers are Gated DeltaNet
//! (linear attention — short causal depthwise conv over q/k/v, L2-normalized
//! q/k, a sigmoid `beta` gate and an `exp(-softplus(..))` decay gate feeding a
//! chunked delta-rule recurrence, gated RMSNorm output), the remainder are GQA
//! full attention with a **doubled** `q_proj` (value half + a sigmoid output
//! gate half) and `partial_rotary_factor` RoPE. Every layer's MLP is a sparse
//! MoE (256 experts, top-8, softmax router) plus a sigmoid-gated shared
//! expert. Vision is a separate composed tower (see `crate::vision`), not part
//! of this config, mirroring `crates/qwenvl`'s `Qwen3Vl`/`Qwen`/`VisionConfig`
//! split.
//!
//! Field names and defaults are taken directly from the real checkpoint's
//! `config.json`/`configuration_qwen3_5_moe.py` — see
//! `/data/workspace/resources/qwen3.5/` for the sources this was built
//! against. Per `docs/lessons.md` #16, a config default must mirror the
//! *reference's* default, not "off": `configuration_qwen3_5_moe.py`'s
//! `__post_init__` hardcodes `partial_rotary_factor = 0.25` and
//! `full_attention_interval = 4` as defaults not always present in a
//! checkpoint's `config.json`.

use serde_json::Value;

pub use qwen3::LoraCfg;

/// The 9 LoRA-targetable leaf names for this hybrid decoder: GDN's 5 linear
/// projections (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`,
/// `out_proj`) and GQA's 4 (`q_proj`, `k_proj`, `v_proj`, `o_proj`). No MoE
/// expert leaf is ever included here — the 256-expert linears are
/// deliberately out of scope for LoRA (see `Qwen35Config::param_list`'s own
/// doc). `LoraCfg` is reused as-is from `qwen3` (a small, model-agnostic
/// `{rank, alpha, targets}` struct) rather than duplicated — this free
/// function is the qwen35-specific piece: `qwen3::LoraCfg::attn` targets its
/// OWN four leaf names (`wq`/`wk`/`wv`/`wo`), which do not exist on this
/// model, so a qwen35-specific constructor is needed instead of an inherent
/// method on the foreign `LoraCfg` type (which the orphan rule would not
/// allow from this crate anyway).
pub fn lora_targets() -> Vec<String> {
    ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// [`LoraCfg`] targeting every one of [`lora_targets`] at the given rank/alpha
/// — the qwen35 analogue of `qwen3::LoraCfg::attn`.
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
    /// 0.25 — only the first `head_dim * partial_rotary_factor` dims rotate,
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

    // -- MoE --
    pub n_experts: u32,
    pub top_k: u32,
    pub moe_intermediate_size: u32,
    pub shared_expert_intermediate_size: u32,

    /// `Some` selects LoRA fine-tuning (frozen base + adapters); `None` is a
    /// full (all-parameter) model.
    pub lora: Option<LoraCfg>,
}

impl Qwen35Config {
    /// A tiny hybrid config for tests / gradient checks. Every dimension that
    /// is distinct in the real config stays distinct here (`docs/lessons.md`
    /// #4 — degenerate/collapsed toy dims hide whole bug classes): d_model,
    /// head_dim, n_heads, n_kv_heads, linear_key_head_dim,
    /// linear_value_head_dim, linear_num_key_heads, linear_num_value_heads,
    /// n_experts and top_k are all pairwise distinct where the real config
    /// keeps them distinct. `full_attention_interval = 4` with `n_layers = 8`
    /// exercises both layer types (layers 3 and 7 are `Full`).
    pub fn tiny() -> Qwen35Config {
        Qwen35Config {
            vocab: 29,
            block_size: 24,
            n_layers: 8,
            d_model: 24,
            rms_eps: 1e-6,
            max_position_embeddings: 24,
            tie_embeddings: false,

            n_heads: 6,
            n_kv_heads: 2,
            head_dim: 12,
            attn_bias: false,
            rope_theta: 1.0e6,
            partial_rotary_factor: 0.5,
            mrope_section: [1, 1, 1],

            full_attention_interval: 4,
            linear_num_key_heads: 3,
            linear_num_value_heads: 6,
            linear_key_head_dim: 4,
            linear_value_head_dim: 5,
            linear_conv_kernel_dim: 4,

            n_experts: 6,
            top_k: 2,
            moe_intermediate_size: 10,
            shared_expert_intermediate_size: 7,

            lora: None,
        }
    }

    /// The published Qwen3.5-35B-A3B shape (from its real `config.json`).
    pub fn qwen35_35b_a3b() -> Qwen35Config {
        Qwen35Config {
            vocab: 248320,
            block_size: 4096,
            n_layers: 40,
            d_model: 2048,
            rms_eps: 1e-6,
            max_position_embeddings: 262144,
            tie_embeddings: false,

            n_heads: 16,
            n_kv_heads: 2,
            head_dim: 256,
            attn_bias: false,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            mrope_section: [11, 11, 10],

            full_attention_interval: 4,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,

            n_experts: 256,
            top_k: 8,
            moe_intermediate_size: 512,
            shared_expert_intermediate_size: 512,

            lora: None,
        }
    }

    pub fn layer_types(&self) -> Vec<LayerType> {
        layer_types(self.n_layers, self.full_attention_interval)
    }

    // -- full-attention (GQA) derived shapes --
    /// Query projection width, **doubled** for the value+gate split
    /// (`Qwen3_5MoeAttention.q_proj` emits `num_heads * head_dim * 2`).
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
    /// (`num_v_heads / num_k_heads`, e.g. 32/16 = 2 at the real scale).
    pub fn linear_group(&self) -> u32 {
        self.linear_num_value_heads / self.linear_num_key_heads
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
            "num_experts": self.n_experts, "num_experts_per_tok": self.top_k,
            "moe_intermediate_size": self.moe_intermediate_size,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
        });
        // A LoRA checkpoint must round-trip its adapter shape, or `param_list()`
        // rebuilds without the `.lora_a`/`.lora_b` names on load and the trained
        // adapters are silently dropped (docs/lessons.md #23).
        if let Some(l) = &self.lora {
            v["lora"] = serde_json::json!({
                "rank": l.rank, "alpha": l.alpha, "targets": l.targets,
            });
        }
        v
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
            n_layers: g("n_layers", 8),
            d_model: g("d_model", 24),
            rms_eps: gf("rms_norm_eps", 1e-6),
            max_position_embeddings: g("max_position_embeddings", block_size),
            tie_embeddings: c["tie_word_embeddings"].as_bool().unwrap_or(false),

            n_heads: g("n_heads", 6),
            n_kv_heads: g("n_kv_heads", 2),
            head_dim: g("head_dim", 12),
            attn_bias: c["attention_bias"].as_bool().unwrap_or(false),
            rope_theta: gf("rope_theta", 1.0e7),
            // Reference default per docs/lessons.md #16 — a config.json that
            // predates this field still means 0.25, not "unset -> full RoPE".
            partial_rotary_factor: gf("partial_rotary_factor", 0.25),
            mrope_section: mrope,

            full_attention_interval: g("full_attention_interval", 4),
            linear_num_key_heads: g("linear_num_key_heads", 3),
            linear_num_value_heads: g("linear_num_value_heads", 6),
            linear_key_head_dim: g("linear_key_head_dim", 4),
            linear_value_head_dim: g("linear_value_head_dim", 5),
            linear_conv_kernel_dim: g("linear_conv_kernel_dim", 4),

            n_experts: g("num_experts", 6),
            top_k: g("num_experts_per_tok", 2),
            moe_intermediate_size: g("moe_intermediate_size", 10),
            shared_expert_intermediate_size: g("shared_expert_intermediate_size", 7),

            lora: c.get("lora").and_then(|l| l.as_object()).map(|l| LoraCfg {
                rank: l["rank"].as_u64().unwrap_or(0) as u32,
                alpha: l["alpha"].as_f64().unwrap_or(0.0) as f32,
                targets: l["targets"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect())
                    .unwrap_or_default(),
            }),
        }
    }

    /// Parameter list: `(name, numel)`. Every expert keeps its own indexed
    /// tensor name (never concatenated into one `[E, ff, d]` buffer) even
    /// though the real checkpoint stores `mlp.experts.{gate_up,down}_proj` as
    /// ONE fused 3-D tensor per layer — `import.rs` splits gate/up/down and
    /// slices per expert on the host at import time (same "split fused
    /// weights at import" rule `docs/porting-playbook.md` §2 uses for qkv/
    /// gate-up fusions elsewhere), because `model::moe`'s dispatch reads one
    /// 2-D expert weight per call.
    ///
    /// With `self.lora` set, a targeted linear (matched by leaf name against
    /// `LoraCfg::targets`, e.g. `"in_proj_qkv"`/`"q_proj"`) becomes a frozen
    /// base weight plus trainable `A[r,in]`/`B[out,r]` adapters (mirrors
    /// `qwen3::config::QwenConfig::param_list`'s own `lin` helper exactly).
    /// Only the linear PROJECTIONS listed in this module's doc are ever
    /// targetable this way — the 256-expert MoE linears are deliberately never
    /// matched (no leaf name below is ever `"gate"`/`"up"`/`"down"` alone, only
    /// the qualified GDN/GQA projection names), matching the standing LoRA
    /// task's own scope note.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let d = self.d_model as usize;
        let v = self.vocab as usize;
        let r = self.lora.as_ref().map(|l| l.rank as usize);
        // A linear `[out, in]` either as a plain trainable weight, or (LoRA on
        // a targeted leaf) as a frozen base + A[r,in] + B[out,r] adapters —
        // same shape as `qwen3::config::QwenConfig::param_list`'s `lin`.
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

            // MoE: every layer, both mixer types.
            let ff = self.moe_intermediate_size as usize;
            let sff = self.shared_expert_intermediate_size as usize;
            out.push((p("mlp.router.weight"), self.n_experts as usize * d));
            for e in 0..self.n_experts {
                let pe = |s: &str| format!("blocks.{l}.mlp.experts.{e}.{s}");
                out.push((pe("gate.weight"), ff * d));
                out.push((pe("up.weight"), ff * d));
                out.push((pe("down.weight"), d * ff));
            }
            out.push((p("mlp.shared_expert.gate.weight"), sff * d));
            out.push((p("mlp.shared_expert.up.weight"), sff * d));
            out.push((p("mlp.shared_expert.down.weight"), d * sff));
            out.push((p("mlp.shared_expert_gate.weight"), d));
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
        // list (40 layers, full at 1-indexed multiples of 4).
        let cfg = Qwen35Config::qwen35_35b_a3b();
        let types = cfg.layer_types();
        assert_eq!(types.len(), 40);
        let full_idx: Vec<usize> =
            types.iter().enumerate().filter(|(_, t)| **t == LayerType::Full).map(|(i, _)| i).collect();
        assert_eq!(full_idx, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);
    }

    #[test]
    fn json_round_trip_preserves_every_field() {
        let cfg = Qwen35Config::qwen35_35b_a3b();
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
        assert_eq!(back.n_experts, cfg.n_experts);
        assert_eq!(back.top_k, cfg.top_k);
        assert_eq!(back.moe_intermediate_size, cfg.moe_intermediate_size);
        assert_eq!(back.shared_expert_intermediate_size, cfg.shared_expert_intermediate_size);
        assert_eq!(back.tie_embeddings, cfg.tie_embeddings);
    }

    #[test]
    fn lora_round_trips_through_json() {
        let mut cfg = Qwen35Config::tiny();
        cfg.lora = Some(LoraCfg::attn(8, 16.0));
        let back = Qwen35Config::from_json(&cfg.to_json());
        let lora = back.lora.expect("lora must round-trip (docs/lessons.md #23)");
        assert_eq!(lora.rank, 8);
        assert_eq!(lora.alpha, 16.0);
    }

    #[test]
    fn tiny_config_has_pairwise_distinct_dims_within_each_layer_type() {
        // docs/lessons.md #4: degenerate/collapsed toy dims hide whole bug
        // classes. Assert the toy config doesn't accidentally collapse any of
        // the dims that are distinct in the real 35B-A3B config.
        let cfg = Qwen35Config::tiny();
        let full = [cfg.d_model, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads];
        for i in 0..full.len() {
            for j in (i + 1)..full.len() {
                assert_ne!(full[i], full[j], "full-attention dims must be pairwise distinct");
            }
        }
        let lin = [
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
            cfg.linear_num_key_heads,
            cfg.linear_num_value_heads,
        ];
        for i in 0..lin.len() {
            for j in (i + 1)..lin.len() {
                assert_ne!(lin[i], lin[j], "linear-attention dims must be pairwise distinct");
            }
        }
        assert!(cfg.top_k < cfg.n_experts, "top_k must not equal n_experts (degenerate MoE)");
    }

    #[test]
    fn real_checkpoint_expert_tensor_count_matches_param_list() {
        // Cross-checked against model.safetensors.index.json: 256 experts *
        // 3 tensors (gate/up/down, split from the fused checkpoint tensors at
        // import) * 40 layers, plus one router + 3 shared-expert + 1
        // shared-expert-gate tensor per layer.
        let cfg = Qwen35Config::qwen35_35b_a3b();
        let names = cfg.param_list();
        let expert_tensors =
            names.iter().filter(|(n, _)| n.contains(".mlp.experts.")).count();
        assert_eq!(expert_tensors, 256 * 3 * 40);
        let router_tensors = names.iter().filter(|(n, _)| n.ends_with("mlp.router.weight")).count();
        assert_eq!(router_tensors, 40);
    }
}
