// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Synthetic Ripple-shaped models for tests that compare two runs of this engine with each other: any layer pattern
//! and attention shape, random weights of a scale that keeps the residual stream tame, no reference implementation.
//!
//! Swedish Embedded AB implements solutions for validating inference engines at shapes no published checkpoint small
//! enough to test has, for its clients. If your team needs expertise in equivalence testing of serving kernels then you
//! can procure our services by sending an email to info@swedishembedded.com.

use data::rng::Lcg;
use model::shard::{Shard, ShardCost, StreamPlan};
use ricursive::{manifest, open_safetensors, RicursiveConfig, RicursiveEngine, ServeConfig};

/// The attention shape of a synthetic model.
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

pub(crate) struct Model {
    pub cfg: RicursiveConfig,
    file: tempfile::TempDir,
}

impl Model {
    fn path(&self) -> String {
        self.file.path().join("synthetic.safetensors").to_string_lossy().into_owned()
    }

    fn source(&self) -> Box<checkpoint::remap::RemapSource<'static>> {
        let layers: Vec<usize> = (0..self.cfg.num_hidden_layers).collect();
        Box::new(open_safetensors(&self.path(), &self.cfg, &layers).expect("open"))
    }

    pub fn engine(&self, serve: ServeConfig) -> RicursiveEngine {
        RicursiveEngine::from_source(self.cfg.clone(), self.source(), serve, None).expect("engine")
    }

    /// The layers cut into two stages on the ambient device.
    pub fn staged(&self, serve: ServeConfig) -> RicursiveEngine {
        let plan = StreamPlan::balanced(&ShardCost { n_layers: self.cfg.num_hidden_layers, per_layer: 1.0, embed: 0.0, head: 0.0, boundary_words: 0 }, &[Shard::ANY_GPU, Shard::ANY_GPU]);
        assert_eq!(plan.n_stages(), 2);
        RicursiveEngine::from_source_staged(self.cfg.clone(), self.source(), serve, None, &plan).expect("staged engine")
    }
}

/// A model of the tiny width (hidden 64, FFN 96, window 4) with `layers` and `shape`.
pub(crate) fn synthetic(layers: &[&str], shape: Shape) -> Model {
    let types = serde_json::to_string(layers).unwrap();
    let cfg = RicursiveConfig::from_json(&format!(
        r#"{{"model_type":"ricursive","vocab_size":128,"hidden_size":64,"intermediate_size":96,"num_hidden_layers":{},"num_attention_heads":{},"num_key_value_heads":{},
            "head_dim":{},"sliding_window":4,"use_valres":true,"use_gated_attention":true,"ffn_clamp_threshold":20.0,"embed_scale":8.0,"layer_types":{types}}}"#,
        layers.len(),
        shape.heads,
        shape.kv_heads,
        shape.head_dim
    ))
    .unwrap();
    let mut rng = Lcg::new(31);
    let order: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    let tensors: Vec<brain_testutil::hfquant::Raw> = manifest(&cfg, &order)
        .into_iter()
        .map(|(name, dims)| {
            let n: usize = dims.iter().product();
            let data: Vec<f32> = if name.ends_with("norm.weight") || name.contains("layernorm") {
                (0..n).map(|_| 1.0 + rng.scaled(0.1)).collect()
            } else if name.ends_with("value_res_lambda") {
                vec![0.0; n]
            } else {
                let fan_in = dims.last().copied().unwrap_or(1) as f32;
                (0..n).map(|_| rng.signed() * 1.7 / fan_in.sqrt()).collect()
            };
            brain_testutil::hfquant::f32_raw(&name, &dims, &data)
        })
        .collect();
    let file = tempfile::tempdir().unwrap();
    brain_testutil::hfquant::write_safetensors(&file.path().join("synthetic.safetensors"), &tensors);
    Model { cfg, file }
}
