// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a Ripple training run is built from: the model's own configuration, the
//! sequence capacity, and an optional adapter, with the parameter list and the
//! seeded initial weights that follow from them.
//!
//! Swedish Embedded AB implements solutions for fine-tuning very large dense
//! decoders on the memory a team actually has, with adapters that export to the
//! formats the rest of the ecosystem reads. If your team needs expertise in
//! parameter-efficient training of hybrid-attention language models then you
//! can procure our services by sending an email to info@swedishembedded.com.
//!
//! Tensor names are the checkpoint's own (`model.layers.N...`, see
//! [`crate::import`]), so a trained model saves under the names the release
//! uses and an adapter's `{base}.lora_a` / `{base}.lora_b` hang off the base
//! weight's full name. The adapter targets are the HF module leaves
//! (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `g_proj`, `gate_proj`, `up_proj`,
//! `down_proj`); the naming of the adapter tensors themselves lives in
//! `model::adapter` only.

use std::collections::HashMap;

use data::rng::Rng;
use model::adapter::device::LoraCfg;
use model::adapter::sites::LoraSites;
use model::adapter::{LinearSite, TargetSpec};
use model::Shard;

use crate::config::RicursiveConfig;
use crate::import::manifest;

/// The leaves an adapter may target, in the order a layer lists its linears.
pub const LORA_LEAVES: [&str; 8] = ["q_proj", "k_proj", "v_proj", "o_proj", "g_proj", "gate_proj", "up_proj", "down_proj"];

/// The tensor name of the linear `leaf` within a layer (`self_attn.q_proj.weight`), the suffix of
/// [`crate::import::layer_tensor`].
///
/// # Panics
///
/// On a leaf that is not one of [`LORA_LEAVES`]: a caller names its leaves from that list.
pub fn linear_suffix(leaf: &str) -> String {
    match leaf {
        "q_proj" | "k_proj" | "v_proj" | "o_proj" | "g_proj" => format!("self_attn.{leaf}.weight"),
        "gate_proj" | "up_proj" | "down_proj" => format!("mlp.{leaf}.weight"),
        other => panic!("ricursive: {other:?} is not an adapter leaf ({LORA_LEAVES:?})"),
    }
}

/// The adapter leaves the standard recipe corrects: attention and the feed-forward.
pub fn default_targets() -> Vec<String> {
    LORA_LEAVES.iter().map(|s| (*s).to_string()).collect()
}

/// A Ripple training run: the model, the longest sequence a row may hold and the adapter (if any).
#[derive(Clone, Debug, PartialEq)]
pub struct TrainConfig {
    pub model: RicursiveConfig,
    /// Tokens per row the run is sized for; positions beyond it have no activation buffers.
    pub block_size: u32,
    /// `Some`: train only the adapter's factors; the base is frozen (and may be held packed).
    pub lora: Option<LoraCfg>,
}

impl TrainConfig {
    /// A full fine-tune of `model` sized for rows of `block_size` tokens.
    pub fn new(model: RicursiveConfig, block_size: u32) -> TrainConfig {
        TrainConfig { model, block_size, lora: None }
    }

    /// This configuration training a rank-`rank` adapter on `targets` instead of the weights.
    pub fn with_lora(mut self, lora: LoraCfg) -> TrainConfig {
        self.lora = Some(lora);
        self
    }

    /// The base linears of layer `layer` as `(leaf, name, out, in)`.
    pub(crate) fn layer_linears(&self, layer: usize) -> Vec<(&'static str, String, usize, usize)> {
        let c = &self.model;
        let name = |leaf: &str| crate::import::layer_tensor(layer, &linear_suffix(leaf));
        let mut v = vec![
            ("q_proj", name("q_proj"), c.q_dim(), c.hidden_size),
            ("k_proj", name("k_proj"), c.kv_dim(), c.hidden_size),
            ("v_proj", name("v_proj"), c.kv_dim(), c.hidden_size),
            ("o_proj", name("o_proj"), c.hidden_size, c.q_dim()),
        ];
        if c.use_gated_attention {
            v.push(("g_proj", name("g_proj"), c.q_dim(), c.hidden_size));
        }
        v.push(("gate_proj", name("gate_proj"), c.intermediate_size, c.hidden_size));
        v.push(("up_proj", name("up_proj"), c.intermediate_size, c.hidden_size));
        v.push(("down_proj", name("down_proj"), c.hidden_size, c.intermediate_size));
        v
    }

    /// Check what the run asks of the model: adapter targets that exist, a block that fits.
    pub fn validate(&self) -> Result<(), String> {
        if self.block_size == 0 {
            return Err("ricursive training: block_size must be positive".into());
        }
        if let Some(lora) = &self.lora {
            if lora.rank == 0 {
                return Err("ricursive training: the adapter rank must be positive".into());
            }
            for t in &lora.targets {
                if !LORA_LEAVES.contains(&t.as_str()) {
                    return Err(format!("ricursive training: adapter target {t:?} is not one of {LORA_LEAVES:?}"));
                }
                if t == "g_proj" && !self.model.use_gated_attention {
                    return Err("ricursive training: g_proj is targeted but the model has no gated attention".into());
                }
            }
        }
        Ok(())
    }

    /// A small Ripple for tests and demonstrations: three layers (sliding, sliding, global), a 3-key window over rows
    /// that outgrow it, every Ripple op on (value residual, gated attention, a clip at 1.0, an embedding scale) and
    /// pairwise-distinct non-square widths (hidden 64, q 128, kv 32, feed-forward 192, vocabulary 32), every
    /// contraction a whole number of the 64-element blocks of the packed tiers.
    pub fn tiny() -> TrainConfig {
        let model = RicursiveConfig::from_json(
            r#"{"model_type":"ricursive","vocab_size":32,"hidden_size":64,"intermediate_size":192,"num_hidden_layers":3,"num_attention_heads":8,"num_key_value_heads":2,
                "head_dim":16,"sliding_window":3,"use_valres":true,"use_gated_attention":true,"ffn_clamp_threshold":1.0,"embed_scale":8.0,
                "layer_types":["sliding_attention","sliding_attention","full_attention"]}"#,
        )
        .expect("the tiny configuration is valid");
        TrainConfig { model, block_size: 8, lora: None }
    }

    /// Weights at the scale of a trained model, not of an initialisation: projections Normal(0, 0.2), norm gains
    /// around one, a nonzero value-residual gate per layer and the head at 0.1, so that attention is not uniform,
    /// both clamps fire on part of the activations and the logits are not saturated. For gradient checks and parity
    /// against a reference, which need every path alive. The adapter, when there is one, starts as a no-op.
    pub fn tiny_weights(&self, seed: u64) -> HashMap<String, Vec<f32>> {
        let mut rng = Rng::new(seed);
        let mut w = self.init_weights(seed);
        // In parameter-list order, so that the one generator draws the same values for the same tensor every run.
        for (name, _) in self.param_list() {
            let v = w.get_mut(&name).expect("init_weights covers the parameter list");
            let scale = if name.ends_with("lm_head.weight") { 0.1 } else { 0.2 };
            if name.ends_with("norm.weight") || name.contains("layernorm") {
                v.iter_mut().for_each(|x| *x = 1.0 + 0.2 * rng.next_gaussian() as f32);
            } else if name.ends_with("value_res_lambda") {
                v.iter_mut().for_each(|x| *x = 1.2 * rng.next_gaussian() as f32);
            } else if model::adapter::device::init_of(&name).is_none() {
                v.iter_mut().for_each(|x| *x = scale * rng.next_gaussian() as f32);
            }
        }
        w
    }

    /// The `(name, numel)` of every tensor the run owns, the adapter's factors last.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let layers: Vec<usize> = (0..self.model.num_hidden_layers).collect();
        let mut list: Vec<(String, usize)> = manifest(&self.model, &layers).into_iter().map(|(n, shape)| (n, shape.iter().product())).collect();
        if let Some(lora) = &self.lora {
            for site in self.lora_sites().into_iter().filter(|s| lora.targets_leaf(s.leaf)) {
                list.extend(lora.param_list(&site.name, site.spec.out, site.spec.inn));
            }
        }
        list
    }

    /// Whether the stage `shard` holds the tensor `name`: its layers' tensors, the final norm and head on the head
    /// stage, and the embedding on the embed stage and on every stage when the value residual reads it (`x0` is
    /// recomputed from the tokens there, so no extra tensor crosses a stage boundary).
    pub fn shard_owns(&self, name: &str, shard: &Shard) -> bool {
        if let Some(rest) = name.strip_prefix("model.layers.") {
            return rest.split('.').next().and_then(|n| n.parse::<usize>().ok()).is_some_and(|l| shard.owns(l));
        }
        match name {
            "model.embed_tokens.weight" => shard.embed || self.model.use_valres,
            "model.norm.weight" | "lm_head.weight" => shard.head,
            _ => true,
        }
    }

    /// [`Self::param_list`] restricted to the tensors the stage `shard` holds.
    pub fn shard_param_list(&self, shard: &Shard) -> Vec<(String, usize)> {
        self.param_list().into_iter().filter(|(name, _)| self.shard_owns(name, shard)).collect()
    }

    /// Seeded initial weights: Normal(0, 0.02) projections (the residual-output ones scaled down by the depth),
    /// unit norm gains, a zero value-residual gate (an even mix) and the adapter's own init.
    pub fn init_weights(&self, seed: u64) -> HashMap<String, Vec<f32>> {
        let mut rng = Rng::new(seed);
        let std = 0.02f32;
        let proj_std = std / (2.0 * self.model.num_hidden_layers as f32).sqrt();
        let mut w = HashMap::new();
        for (name, numel) in self.param_list() {
            let v = if name.ends_with("norm.weight") || name.contains("layernorm") {
                vec![1.0; numel]
            } else if name.ends_with("value_res_lambda") {
                vec![0.0; numel]
            } else if let Some(init) = model::adapter::device::init_of(&name) {
                match init {
                    model::adapter::kind::ParamInit::Zeros => vec![0.0; numel],
                    _ => (0..numel).map(|_| rng.next_gaussian() as f32 * std).collect(),
                }
            } else {
                let s = if name.ends_with("o_proj.weight") || name.ends_with("down_proj.weight") { proj_std } else { std };
                (0..numel).map(|_| rng.next_gaussian() as f32 * s).collect()
            };
            w.insert(name, v);
        }
        w
    }

    /// The configuration as the JSON a checkpoint header carries.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "model": self.model.to_json(),
            "block_size": self.block_size,
            "lora": self.lora.as_ref().map(LoraCfg::to_json),
        })
    }

    /// Read back [`Self::to_json`]; a malformed header is an error naming the field.
    pub fn from_json(v: &serde_json::Value) -> Result<TrainConfig, String> {
        let model = RicursiveConfig::from_json(&v.get("model").ok_or("ricursive training config: no model")?.to_string())?;
        let block_size = v.get("block_size").and_then(|b| b.as_u64()).and_then(|b| u32::try_from(b).ok()).ok_or("ricursive training config: block_size is not a u32")?;
        let lora = match v.get("lora") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::Object(map)) => Some(LoraCfg::from_json(map)),
            Some(_) => return Err("ricursive training config: lora is not an object".into()),
        };
        let cfg = TrainConfig { model, block_size, lora };
        cfg.validate()?;
        Ok(cfg)
    }
}

impl LoraSites for TrainConfig {
    fn lora_sites(&self) -> Vec<LinearSite> {
        (0..self.model.num_hidden_layers)
            .flat_map(|layer| {
                self.layer_linears(layer).into_iter().map(move |(leaf, name, out, inn)| LinearSite { name, leaf, layer: Some(layer), spec: TargetSpec::whole(out, inn), save_name: None })
            })
            .collect()
    }
}

impl model::adapter::bank::AdapterServing for TrainConfig {
    fn attention_leaves(&self) -> Vec<&'static str> {
        vec!["q_proj", "k_proj", "v_proj", "o_proj"]
    }
    /// Ripple's adapters correct all eight linears when trained with the default recipe; a bank offered without
    /// `--adapter-targets` takes them all.
    fn default_slots(&self) -> u32 {
        4
    }
}

impl model::ModelConfig for TrainConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        TrainConfig::param_list(self)
    }
    fn to_json(&self) -> serde_json::Value {
        TrainConfig::to_json(self)
    }
    fn from_json(v: &serde_json::Value) -> Self {
        TrainConfig::from_json(v).unwrap_or_else(|e| panic!("{e}"))
    }
    fn vocab(&self) -> u32 {
        self.model.vocab_size as u32
    }
    fn block_size(&self) -> u32 {
        self.block_size
    }
    fn finalize_for_dataset(mut self, vocab: u32, block_size: u32) -> Self {
        self.model.vocab_size = vocab as usize;
        self.block_size = block_size;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(lora: Option<LoraCfg>) -> TrainConfig {
        let model = RicursiveConfig::from_json(
            r#"{"vocab_size":24,"hidden_size":16,"intermediate_size":24,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":6,
                "layer_types":["sliding_attention","full_attention"],"sliding_window":3,"use_valres":true,"ffn_clamp_threshold":0.5,"embed_scale":4.0}"#,
        )
        .unwrap();
        TrainConfig { model, block_size: 8, lora }
    }

    #[test]
    fn the_parameter_list_is_the_checkpoints_manifest_then_the_adapters() {
        let cfg = tiny(Some(LoraCfg::new(2, 4.0, vec!["q_proj".into(), "down_proj".into()])));
        let list = cfg.param_list();
        let base = tiny(None).param_list();
        assert_eq!(&list[..base.len()], &base[..]);
        let adapters: Vec<&str> = list[base.len()..].iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(adapters.len(), 2 * 2 * 2, "two sites per layer, A and B each");
        assert!(adapters.contains(&"model.layers.1.mlp.down_proj.weight.lora_b"));
        let (_, numel) = list.iter().find(|(n, _)| n == "model.layers.0.self_attn.q_proj.weight.lora_a").unwrap();
        assert_eq!(*numel, 2 * 16, "A is [rank, in]");
    }

    #[test]
    fn a_config_survives_its_json_with_its_adapter() {
        let cfg = tiny(Some(LoraCfg::new(3, 6.0, default_targets())));
        assert_eq!(TrainConfig::from_json(&cfg.to_json()).unwrap(), cfg);
        let plain = tiny(None);
        assert_eq!(TrainConfig::from_json(&plain.to_json()).unwrap(), plain);
    }

    #[test]
    fn an_unknown_adapter_target_is_refused_by_name() {
        let cfg = tiny(Some(LoraCfg::new(2, 2.0, vec!["wq".into()])));
        assert!(cfg.validate().unwrap_err().contains("\"wq\""));
    }

    #[test]
    fn initial_weights_cover_the_list_and_start_the_adapter_as_a_no_op() {
        let cfg = tiny(Some(LoraCfg::new(2, 2.0, default_targets())));
        let w = cfg.init_weights(5);
        for (name, numel) in cfg.param_list() {
            assert_eq!(w[&name].len(), numel, "{name}");
        }
        assert!(w["model.layers.0.self_attn.q_proj.weight.lora_b"].iter().all(|x| *x == 0.0));
        assert!(w["model.layers.0.self_attn.q_proj.weight.lora_a"].iter().any(|x| *x != 0.0));
        assert_eq!(w, cfg.init_weights(5));
        assert_ne!(w["model.embed_tokens.weight"], cfg.init_weights(6)["model.embed_tokens.weight"]);
    }
}
