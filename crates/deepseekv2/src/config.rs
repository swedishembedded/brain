// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-V2-family MHA decoder configuration + parameter layout.
//!
//! The **shape** half is not redefined here: it is
//! [`gguf::deepseek_ocr::DeepseekOcrConfig`], the struct the GGUF loader
//! already derives from the real checkpoint and whose `param_list()` is that
//! import's two-way coverage contract. Re-declaring those twenty fields in this
//! crate would create a second description of the same checkpoint that can
//! drift from the loader's - the loader would import `blocks.N.mlp.shared.*`
//! and a divergent decoder would look for something else, with nothing to catch
//! it. So [`DeepseekV2Config`] *wraps* it and adds only what the loader
//! deliberately does not carry:
//!
//! - `block_size` - the training/eval sequence length, a run parameter, not a
//!   checkpoint fact (the checkpoint's own `max_position_embeddings` stays in
//!   the wrapped shape).
//! - `norm_topk_prob` / `routed_scaling` - **forward-pass** router policy. The
//!   GGUF carries no `expert_weights_norm`/`expert_weights_scale` key at all,
//!   so llama.cpp's compiled-in defaults are what runs: no renormalisation,
//!   scale 1.0. They belong to the decoder, which is why the loader records
//!   them in prose and this struct records them as fields.
//!
//! Every brain-side tensor name comes from the wrapped `param_list()` -- *plus*,
//! when [`DeepseekV2Config::lora`] is set, the `.lora_a`/`.lora_b` adapter
//! tensors [`Self::param_list`] appends on top for the four attention
//! projections (see that method's own doc) -- so this decoder and that
//! importer cannot disagree about the CHECKPOINT'S layout by construction; the
//! adapter tensors are never part of a checkpoint's own manifest (LoRA is
//! trained after import, not shipped by DeepSeek), which is why
//! `import.rs::import_map`/`dry_run` read `shape.param_list()` directly rather
//! than through this wrapper.

use gguf::deepseek_ocr::DeepseekOcrConfig;
use serde_json::Value;

pub use qwen3::LoraCfg;

/// The four LoRA-targetable attention-projection leaves this decoder has --
/// plain MHA, so just `q_proj`/`k_proj`/`v_proj`/`o_proj`; no MoE expert leaf
/// is ever included (mirrors `crates/qwen35moe::config::lora_targets`'s own
/// exclusion of its 256-expert linears -- parameter-efficient fine-tuning of a
/// sparse MoE's *routed* experts is a different, much larger feature this
/// decoder does not attempt).
pub fn lora_targets() -> Vec<String> {
    ["q_proj", "k_proj", "v_proj", "o_proj"].iter().map(|s| s.to_string()).collect()
}

/// [`LoraCfg`] targeting every one of [`lora_targets`] at the given rank/alpha
/// -- the deepseekv2 analogue of `qwen3::LoraCfg::attn`/`qwen35moe::config::lora_cfg`.
pub fn lora_cfg(rank: u32, alpha: f32) -> LoraCfg {
    LoraCfg { rank, alpha, targets: lora_targets() }
}

/// If `name` is one of this decoder's four attention-projection weights
/// (`blocks.{l}.self_attn.{leaf}.weight`), the LoRA-targetable leaf it names.
fn attn_leaf(name: &str) -> Option<&'static str> {
    ["q_proj", "k_proj", "v_proj", "o_proj"]
        .into_iter()
        .find(|leaf| name.ends_with(&format!("self_attn.{leaf}.weight")))
}

/// A DeepSeek-V2-family MHA decoder: the checkpoint's own shape plus the two
/// router-policy knobs and the run's sequence length.
#[derive(Clone, Debug, PartialEq)]
pub struct DeepseekV2Config {
    /// The checkpoint shape, exactly as the GGUF loader derives it.
    pub shape: DeepseekOcrConfig,
    /// Training/eval sequence length (`t`). Distinct from
    /// `shape.max_position_embeddings`, which is the checkpoint's own ceiling.
    pub block_size: u32,
    /// Renormalise the selected top-k softmax probabilities to sum to 1.
    /// **`false` for the real checkpoint** - the raw probabilities are the
    /// combine weights.
    pub norm_topk_prob: bool,
    /// DeepSeek's `routed_scaling_factor` applied on top of the gate.
    /// **`1.0` for the real checkpoint.**
    pub routed_scaling: f32,
    /// `Some` selects LoRA fine-tuning (frozen base + trainable low-rank
    /// adapters on the four attention projections); `None` is a full
    /// (all-parameter) model -- the default, and what every real-weight test
    /// and import path in this crate still builds.
    pub lora: Option<LoraCfg>,
}

impl DeepseekV2Config {
    /// Wrap a loader-derived shape with this architecture's real router policy
    /// (no renormalisation, scale 1.0) and a caller-chosen sequence length.
    pub fn from_shape(shape: DeepseekOcrConfig, block_size: u32) -> DeepseekV2Config {
        DeepseekV2Config { shape, block_size, norm_topk_prob: false, routed_scaling: 1.0, lora: None }
    }

    /// The real `DeepSeek-OCR-Q8_0.gguf` decoder shape, as read off that file's
    /// header (see this crate's lib doc). For scale/sanity only - nothing in
    /// this crate needs the checkpoint to exist.
    pub fn deepseek_ocr(block_size: u32) -> DeepseekV2Config {
        DeepseekV2Config::from_shape(
            DeepseekOcrConfig {
                vocab: 129280,
                n_layers: 12,
                d_model: 1280,
                max_position_embeddings: 8192,
                rms_eps: 1e-6,
                tie_embeddings: false,
                n_heads: 10,
                n_kv_heads: 10,
                head_dim: 128,
                rope_theta: 10_000.0,
                rotary_dim: 128,
                n_dense_layers: 1,
                ffn_hidden: 6848,
                n_experts: 64,
                top_k: 6,
                moe_intermediate_size: 896,
                n_shared_experts: 2,
                n_expert_groups: 1,
                n_expert_groups_used: 1,
            },
            block_size,
        )
    }

    /// The tiny fixture: **the same decoder dimensions the checkpoint-free
    /// golden dumper's own tiny sub-fixture uses** (`d_model` 12 = 3 heads ×
    /// head_dim 4, 2 layers with layer 0 dense, dense ff 21 vs MoE ff 7, 5
    /// routed experts top-2, 2 shared experts fused to 14, vocab 19), so a
    /// later phase can compare this decoder against that dump without first
    /// reconciling two different toy shapes.
    ///
    /// Every dimension that is distinct in the real config stays distinct here,
    /// and several that *coincide* at real scale are deliberately broken apart:
    /// `head_dim != n_heads != d_model`, `moe_ff != dense_ff`,
    /// `shared_ff (14) != moe_ff (7) != dense_ff (21)`, `top_k < n_experts`,
    /// `vocab != d_model`. Collapsed toy dims hide whole bug classes - a
    /// transposed or swapped axis is invisible when the two numbers are equal.
    pub fn tiny() -> DeepseekV2Config {
        DeepseekV2Config::from_shape(
            DeepseekOcrConfig {
                vocab: 19,
                n_layers: 2,
                d_model: 12,
                max_position_embeddings: 13,
                rms_eps: 1e-6,
                tie_embeddings: false,
                n_heads: 3,
                n_kv_heads: 3,
                head_dim: 4,
                rope_theta: 10_000.0,
                rotary_dim: 4,
                n_dense_layers: 1,
                ffn_hidden: 21,
                n_experts: 5,
                top_k: 2,
                moe_intermediate_size: 7,
                n_shared_experts: 2,
                n_expert_groups: 1,
                n_expert_groups_used: 1,
            },
            13,
        )
    }

    // ---- shape accessors (delegated; the wrapped struct is the source of truth) ----

    pub fn vocab(&self) -> u32 {
        self.shape.vocab
    }
    pub fn n_layers(&self) -> u32 {
        self.shape.n_layers
    }
    pub fn d_model(&self) -> u32 {
        self.shape.d_model
    }
    pub fn rms_eps(&self) -> f32 {
        self.shape.rms_eps
    }
    pub fn n_heads(&self) -> u32 {
        self.shape.n_heads
    }
    pub fn n_kv_heads(&self) -> u32 {
        self.shape.n_kv_heads
    }
    pub fn head_dim(&self) -> u32 {
        self.shape.head_dim
    }
    pub fn rope_theta(&self) -> f32 {
        self.shape.rope_theta
    }
    pub fn n_dense_layers(&self) -> u32 {
        self.shape.n_dense_layers
    }
    pub fn ffn_hidden(&self) -> u32 {
        self.shape.ffn_hidden
    }
    pub fn n_experts(&self) -> u32 {
        self.shape.n_experts
    }
    pub fn top_k(&self) -> u32 {
        self.shape.top_k
    }
    pub fn moe_ff(&self) -> u32 {
        self.shape.moe_intermediate_size
    }
    /// The fused shared-expert MLP's width (`n_shared_experts * moe_ff`).
    pub fn shared_ff(&self) -> u32 {
        self.shape.shared_intermediate_size()
    }
    /// Total q projection width (`n_heads * head_dim`).
    pub fn q_dim(&self) -> u32 {
        self.shape.q_dim()
    }
    /// Total k/v projection width (`n_kv_heads * head_dim`) - equal to
    /// [`Self::q_dim`] for this model, which is plain MHA.
    pub fn kv_dim(&self) -> u32 {
        self.shape.n_kv_heads * self.shape.head_dim
    }
    /// Whether block `l` carries the MoE MLP rather than the dense one.
    pub fn is_moe_layer(&self, l: u32) -> bool {
        self.shape.is_moe_layer(l)
    }
    /// The output projection's parameter name (untied on the real checkpoint).
    pub fn head_weight(&self) -> &'static str {
        if self.shape.tie_embeddings {
            "tok.weight"
        } else {
            "lm_head.weight"
        }
    }
    /// The widest per-token feed-forward width in the model - the size the
    /// shared SwiGLU backward scratch must cover for every arm.
    pub fn ff_max(&self) -> u32 {
        self.ffn_hidden().max(self.moe_ff()).max(self.shared_ff())
    }

    /// Every brain-side tensor name and its element count.
    ///
    /// Delegated to the loader's own manifest (`self.shape.param_list()`),
    /// which is what makes an imported checkpoint load into this decoder with
    /// no name translation -- **plus**, when [`Self::lora`] is `Some`, the
    /// `.lora_a`/`.lora_b` adapter pair for every targeted attention
    /// projection the checkpoint's own manifest has no notion of. `A` is
    /// `[rank, d_model]` and `B` is `[d_model, rank]`, both `rank * d_model`
    /// elements -- this decoder's q/k/v/o projections are all square
    /// (`DeepseekV2::new_on` asserts `q_dim == d_model`), so there is no
    /// separate "in" vs "out" width to track. The base tensor's own entry is
    /// left exactly as the loader produced it; whether it ends up
    /// `Role::Frozen` or `Role::Trainable` is `DeepseekV2::new_on`'s role
    /// assignment, not a fact this list carries.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut out = self.shape.param_list();
        if let Some(lora) = &self.lora {
            let d = self.d_model() as usize;
            let r = lora.rank as usize;
            let mut extra = Vec::new();
            for (name, _) in &out {
                let Some(leaf) = attn_leaf(name) else { continue };
                if !lora.targets_leaf(leaf) {
                    continue;
                }
                extra.push((format!("{name}.lora_a"), r * d));
                extra.push((format!("{name}.lora_b"), r * d));
            }
            out.extend(extra);
        }
        out
    }

    pub fn to_json(&self) -> Value {
        let mut v = self.shape.to_json();
        v["model"] = Value::from("deepseekv2");
        v["block_size"] = Value::from(self.block_size);
        // Recorded explicitly even at their defaults: a checkpoint that
        // round-tripped without them would silently reacquire whatever this
        // crate's default is at load time, which is exactly how a forward and
        // its backward come to disagree about the router.
        v["norm_topk_prob"] = Value::from(self.norm_topk_prob);
        v["routed_scaling_factor"] = Value::from(self.routed_scaling);
        // A LoRA checkpoint must round-trip its adapter shape, or `param_list()`
        // rebuilds without the `.lora_a`/`.lora_b` names on load and the trained
        // adapters are silently dropped on the next load -- the same defect
        // `crates/qwen3/tests/lora_roundtrip.rs` gates for that crate's own
        // `LoraCfg` field.
        if let Some(l) = &self.lora {
            v["lora"] = serde_json::json!({ "rank": l.rank, "alpha": l.alpha, "targets": l.targets });
        }
        v
    }

    pub fn from_json(c: &Value) -> DeepseekV2Config {
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        let gf = |k: &str, d: f32| c[k].as_f64().map(|v| v as f32).unwrap_or(d);
        let t = DeepseekV2Config::tiny();
        let head_dim = g("head_dim", t.shape.head_dim);
        let shape = DeepseekOcrConfig {
            vocab: g("vocab_size", t.shape.vocab),
            n_layers: g("n_layers", t.shape.n_layers),
            d_model: g("d_model", t.shape.d_model),
            max_position_embeddings: g("max_position_embeddings", t.shape.max_position_embeddings),
            rms_eps: gf("rms_norm_eps", t.shape.rms_eps),
            tie_embeddings: c["tie_word_embeddings"].as_bool().unwrap_or(false),
            n_heads: g("n_heads", t.shape.n_heads),
            n_kv_heads: g("n_kv_heads", t.shape.n_kv_heads),
            head_dim,
            rope_theta: gf("rope_theta", t.shape.rope_theta),
            // `rotary_dim` absent means "rotate the whole head" - the same
            // resolution `gguf::deepseek_ocr::config_from_gguf` applies to the
            // checkpoint's own `rope.dimension_count = 0`.
            rotary_dim: g("rotary_dim", head_dim),
            n_dense_layers: g("first_k_dense_replace", t.shape.n_dense_layers),
            ffn_hidden: g("intermediate_size", t.shape.ffn_hidden),
            n_experts: g("n_routed_experts", t.shape.n_experts),
            top_k: g("num_experts_per_tok", t.shape.top_k),
            moe_intermediate_size: g("moe_intermediate_size", t.shape.moe_intermediate_size),
            n_shared_experts: g("n_shared_experts", t.shape.n_shared_experts),
            n_expert_groups: g("n_group", 1),
            n_expert_groups_used: g("topk_group", 1),
        };
        DeepseekV2Config {
            block_size: g("block_size", shape.max_position_embeddings),
            norm_topk_prob: c["norm_topk_prob"].as_bool().unwrap_or(false),
            routed_scaling: gf("routed_scaling_factor", 1.0),
            lora: c.get("lora").and_then(|l| l.as_object()).map(|l| LoraCfg {
                rank: l["rank"].as_u64().unwrap_or(0) as u32,
                alpha: l["alpha"].as_f64().unwrap_or(0.0) as f32,
                targets: l["targets"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                    .unwrap_or_default(),
            }),
            shape,
        }
    }
}

impl model::ModelConfig for DeepseekV2Config {
    fn param_list(&self) -> Vec<(String, usize)> {
        DeepseekV2Config::param_list(self)
    }
    fn to_json(&self) -> Value {
        DeepseekV2Config::to_json(self)
    }
    fn from_json(v: &Value) -> Self {
        DeepseekV2Config::from_json(v)
    }
    fn vocab(&self) -> u32 {
        self.shape.vocab
    }
    fn block_size(&self) -> u32 {
        self.block_size
    }
    fn finalize_for_dataset(mut self, vocab: u32, block_size: u32) -> Self {
        self.shape.vocab = vocab;
        self.block_size = block_size;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trip_preserves_every_field() {
        for cfg in [DeepseekV2Config::tiny(), DeepseekV2Config::deepseek_ocr(4096)] {
            let back = DeepseekV2Config::from_json(&cfg.to_json());
            assert_eq!(back, cfg);
        }
    }

    /// A LoRA config must round-trip too, or a reloaded checkpoint silently
    /// drops the trained adapter -- the same defect
    /// `crates/qwen3/tests/lora_roundtrip.rs` gates for `qwen3::QwenConfig`.
    #[test]
    fn lora_config_round_trips() {
        let cfg = DeepseekV2Config { lora: Some(lora_cfg(4, 8.0)), ..DeepseekV2Config::tiny() };
        let back = DeepseekV2Config::from_json(&cfg.to_json());
        assert_eq!(back, cfg);
        assert!(DeepseekV2Config::tiny().to_json().get("lora").is_none(), "a non-LoRA config must not emit a lora key at all");
    }

    /// [`DeepseekV2Config::param_list`] appends exactly one `.lora_a`/`.lora_b`
    /// pair per targeted attention leaf, on top of the UNCHANGED base entries a
    /// real checkpoint import already covers -- and adds nothing at all when
    /// `lora` is `None` (the two-layer tiny fixture: layer 0 dense, layer 1
    /// MoE, both carry the same four attention leaves).
    #[test]
    fn param_list_adds_exactly_the_targeted_adapter_pairs() {
        let base = DeepseekV2Config::tiny();
        let base_list = base.param_list();
        assert_eq!(base_list, base.shape.param_list(), "no lora configured -- param_list must be the loader's manifest verbatim");

        let lora = DeepseekV2Config { lora: Some(lora_cfg(3, 6.0)), ..base.clone() };
        let list = lora.param_list();
        let base_names: std::collections::HashSet<&str> = base_list.iter().map(|(n, _)| n.as_str()).collect();
        assert!(base_names.iter().all(|n| list.iter().any(|(m, _)| m == n)), "a lora config must not drop any base tensor");

        let d = base.d_model() as usize;
        let expect_extra = base.n_layers() as usize * 4 * 2; // 4 leaves x {lora_a, lora_b}, every layer
        assert_eq!(list.len(), base_list.len() + expect_extra);
        for l in 0..base.n_layers() {
            for leaf in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                for suffix in ["lora_a", "lora_b"] {
                    let name = format!("blocks.{l}.self_attn.{leaf}.weight.{suffix}");
                    let (_, numel) = list.iter().find(|(n, _)| n == &name).unwrap_or_else(|| panic!("{name} missing from param_list"));
                    assert_eq!(*numel, 3 * d, "{name}: rank 3 x d_model {d}");
                }
            }
        }
        // A narrower target set adds proportionally fewer tensors.
        let one_leaf = DeepseekV2Config {
            lora: Some(qwen3::LoraCfg { rank: 2, alpha: 4.0, targets: vec!["q_proj".to_string()] }),
            ..base.clone()
        };
        assert_eq!(one_leaf.param_list().len(), base_list.len() + base.n_layers() as usize * 2);
    }

    /// The router policy must survive a round trip even at its defaults, and a
    /// NON-default policy (the gradcheck variant) must too - the pair is what
    /// the backward is differentiated against, so a dropped field is a silently
    /// wrong gradient rather than a load error.
    #[test]
    fn router_policy_round_trips_including_non_defaults() {
        let cfg = DeepseekV2Config { norm_topk_prob: true, routed_scaling: 2.5, ..DeepseekV2Config::tiny() };
        let back = DeepseekV2Config::from_json(&cfg.to_json());
        assert!(back.norm_topk_prob);
        assert_eq!(back.routed_scaling, 2.5);
    }

    /// The parameter layout IS the importer's manifest - assert the delegation
    /// rather than trusting it, and assert the dense/MoE split lands where
    /// `n_dense_layers` says.
    #[test]
    fn param_list_matches_the_loader_manifest_and_splits_dense_from_moe() {
        let cfg = DeepseekV2Config::tiny();
        assert_eq!(cfg.param_list(), cfg.shape.param_list());
        let names: Vec<String> = cfg.param_list().into_iter().map(|(n, _)| n).collect();
        assert!(names.contains(&"blocks.0.mlp.gate.weight".to_string()), "block 0 must be dense");
        assert!(!names.contains(&"blocks.0.mlp.router.weight".to_string()), "block 0 must NOT be MoE");
        assert!(names.contains(&"blocks.1.mlp.router.weight".to_string()), "block 1 must be MoE");
        assert!(names.contains(&"blocks.1.mlp.shared.gate.weight".to_string()), "block 1 needs a fused shared expert");
        assert!(names.contains(&"lm_head.weight".to_string()), "lm_head is untied");
        let experts = names.iter().filter(|n| n.contains(".mlp.experts.")).count();
        assert_eq!(experts, 5 * 3, "5 routed experts x gate/up/down on the one MoE block");
    }

    /// Degenerate toy dims hide bug classes - assert the fixture keeps apart
    /// every pair the real config keeps apart, plus the three feed-forward
    /// widths (which at real scale are 6848 / 896 / 1792, all distinct).
    #[test]
    fn tiny_config_has_pairwise_distinct_dims() {
        let c = DeepseekV2Config::tiny();
        let dims = [c.d_model(), c.head_dim(), c.n_heads(), c.vocab(), c.ffn_hidden(), c.moe_ff(), c.shared_ff()];
        for i in 0..dims.len() {
            for j in (i + 1)..dims.len() {
                assert_ne!(dims[i], dims[j], "tiny dims {i} and {j} collapsed");
            }
        }
        assert_eq!(c.n_heads() * c.head_dim(), c.d_model(), "MHA q width must equal d_model");
        assert_eq!(c.n_kv_heads(), c.n_heads(), "this decoder is plain MHA, not GQA");
        assert!(c.top_k() < c.n_experts(), "top_k must be strictly less (degenerate MoE otherwise)");
        assert!(c.head_dim().is_multiple_of(2), "half-split RoPE needs an even head_dim");
        assert_eq!(c.shape.rotary_dim, c.head_dim(), "the real checkpoint ropes the FULL head_dim");
    }

    /// The real checkpoint's own numbers, re-asserted here so a future edit to
    /// the preset cannot quietly drift from the header this crate was built
    /// against.
    #[test]
    fn real_preset_matches_the_checkpoint_header() {
        let c = DeepseekV2Config::deepseek_ocr(8192);
        assert_eq!((c.n_layers(), c.d_model(), c.n_heads(), c.n_kv_heads(), c.head_dim()), (12, 1280, 10, 10, 128));
        assert_eq!((c.n_experts(), c.top_k(), c.moe_ff(), c.shared_ff()), (64, 6, 896, 1792));
        assert_eq!((c.n_dense_layers(), c.ffn_hidden(), c.vocab()), (1, 6848, 129280));
        assert_eq!(c.head_weight(), "lm_head.weight");
        assert!(!c.norm_topk_prob && c.routed_scaling == 1.0, "llama.cpp's compiled-in defaults for this arch");
    }
}
