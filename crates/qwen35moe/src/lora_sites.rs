// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The adapter sites of a Qwen3.5 mixture-of-experts config: the nine linears
//! a LoRA may correct (`crate::config::lora_targets`). A Gated-DeltaNet layer
//! has `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` and `out_proj`; a
//! full-attention layer has `q_proj`, `k_proj`, `v_proj` and `o_proj`. The
//! router, the routed experts and the shared expert are not adapter sites
//! (hundreds of tiny linears per layer, and a router whose logits an adapter
//! would have to leave alone). The dimensions are the ones
//! `Qwen35Config::param_list` gives an adapter's `.lora_a`/`.lora_b`.
//!
//! Swedish Embedded AB implements solutions for loading and serving adapters
//! across model families without a loader per architecture, for its clients.
//! If your team needs expertise in parameter-efficient fine-tuning
//! infrastructure you can procure our services by sending an email to
//! info@swedishembedded.com.

use model::adapter::sites::LoraSites;
use model::adapter::{LinearSite, TargetSpec};

use crate::config::{LayerType, Qwen35Config};

impl LoraSites for Qwen35Config {
    fn lora_sites(&self) -> Vec<LinearSite> {
        let d = self.d_model as usize;
        let (conv, value) = (self.linear_conv_dim() as usize, self.linear_value_dim() as usize);
        let heads = self.linear_num_value_heads as usize;
        let (q_proj, q, kv) = (self.q_proj_dim() as usize, self.q_dim() as usize, self.kv_dim() as usize);
        let gdn: [(&'static str, &str, usize, usize); 5] = [
            ("in_proj_qkv", "linear_attn.in_proj_qkv.weight", conv, d),
            ("in_proj_z", "linear_attn.in_proj_z.weight", value, d),
            ("in_proj_b", "linear_attn.in_proj_b.weight", heads, d),
            ("in_proj_a", "linear_attn.in_proj_a.weight", heads, d),
            ("out_proj", "linear_attn.out_proj.weight", d, value),
        ];
        let gqa: [(&'static str, &str, usize, usize); 4] = [
            ("q_proj", "self_attn.q_proj.weight", q_proj, d),
            ("k_proj", "self_attn.k_proj.weight", kv, d),
            ("v_proj", "self_attn.v_proj.weight", kv, d),
            ("o_proj", "self_attn.o_proj.weight", d, q),
        ];
        let mut sites = Vec::new();
        for (layer, ty) in self.layer_types().into_iter().enumerate() {
            let mixer: &[(&'static str, &str, usize, usize)] = match ty {
                LayerType::Linear => &gdn,
                LayerType::Full => &gqa,
            };
            for &(leaf, tail, out, inn) in mixer {
                sites.push(LinearSite { name: format!("blocks.{layer}.{tail}"), leaf, layer: Some(layer), spec: TargetSpec::whole(out, inn), save_name: None });
            }
        }
        sites
    }
}

#[cfg(test)]
mod tests {
    use model::adapter::sites::{tensor_names, tensor_shapes};

    use super::*;
    use crate::config::lora_cfg;

    /// The sites are exactly the adapters the config's own parameter list
    /// allocates: same names, same element counts, none missing, none extra.
    #[test]
    fn the_sites_are_the_adapters_param_list_allocates() {
        let cfg = Qwen35Config { lora: Some(lora_cfg(3, 3.0)), ..Qwen35Config::tiny() };
        let params: std::collections::HashMap<String, usize> = cfg.param_list().into_iter().collect();
        let sites = cfg.lora_sites();
        let (gdn_layers, full_layers) = cfg.layer_types().iter().fold((0, 0), |(g, f), t| if *t == LayerType::Linear { (g + 1, f) } else { (g, f + 1) });
        assert_eq!(sites.len(), 5 * gdn_layers + 4 * full_layers);
        for site in &sites {
            let (name_a, name_b) = tensor_names(site);
            let (a, b) = tensor_shapes(site, 3);
            assert_eq!(params.get(&name_a), Some(&(a[0] * a[1])), "{name_a}");
            assert_eq!(params.get(&name_b), Some(&(b[0] * b[1])), "{name_b}");
            assert!(params.contains_key(&site.name), "{} is a base weight", site.name);
        }
        assert_eq!(params.keys().filter(|n| n.ends_with(".lora_a")).count(), sites.len(), "no adapter outside the sites");
    }
}
