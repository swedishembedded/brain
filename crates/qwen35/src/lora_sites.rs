// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The adapter sites of a Qwen3.5 config: the twelve linears a LoRA may
//! correct (`crate::config::lora_targets`), as the generic
//! `model::adapter::sites::LoraSites` lists them. A Gated-DeltaNet layer has
//! `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` and `out_proj`; a
//! full-attention layer has `q_proj`, `k_proj`, `v_proj` and `o_proj`; every
//! layer has the dense MLP's `gate`, `up` and `down`. The dimensions are the
//! ones `Qwen35Config::param_list` gives an adapter's `.lora_a`/`.lora_b`. The
//! multi-token-prediction layer (`cfg.mtp`) is not an adapter site: a build
//! with it offers no per-token objectives either.
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
        let (d, ff) = (self.d_model as usize, self.intermediate_size as usize);
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
        let mlp: [(&'static str, &str, usize, usize); 3] = [("gate", "mlp.gate.weight", ff, d), ("up", "mlp.up.weight", ff, d), ("down", "mlp.down.weight", d, ff)];
        let mut sites = Vec::new();
        for (layer, ty) in self.layer_types().into_iter().enumerate() {
            let mixer: &[(&'static str, &str, usize, usize)] = match ty {
                LayerType::Linear => &gdn,
                LayerType::Full => &gqa,
            };
            for &(leaf, tail, out, inn) in mixer.iter().chain(mlp.iter()) {
                sites.push(LinearSite { name: format!("blocks.{layer}.{tail}"), leaf, layer: Some(layer), spec: TargetSpec::whole(out, inn), save_name: None });
            }
            if self.layer_has_memory(layer) {
                for (leaf, out, inn) in self.memory_linears() {
                    sites.push(LinearSite { name: format!("blocks.{layer}.memory.{leaf}"), leaf: memory_site_leaf(leaf), layer: Some(layer), spec: TargetSpec::whole(out, inn), save_name: None });
                }
            }
        }
        sites
    }
}

/// The LoRA target name of a memory linear (`memory.q_proj` for the `q_proj.weight` leaf).
fn memory_site_leaf(leaf: &str) -> &'static str {
    match leaf {
        "q_proj.weight" => "memory.q_proj",
        "k_proj.weight" => "memory.k_proj",
        "v_proj.weight" => "memory.v_proj",
        "o_proj.weight" => "memory.o_proj",
        "write_gate_proj.weight" => "memory.write_gate_proj",
        "write_up_proj.weight" => "memory.write_up_proj",
        "write_down_proj.weight" => "memory.write_down_proj",
        other => panic!("qwen35: {other} is not an anchor-memory linear"),
    }
}

#[cfg(test)]
mod tests {
    use model::adapter::sites::{tensor_names, tensor_shapes, validate};
    use model::lora::device_adapter::{AdapterSite, DeviceAdapter};

    use super::*;
    use crate::config::lora_cfg;

    /// The tiny config's dimensions are pairwise distinct (see
    /// `Qwen35Config::tiny`), so a swapped or transposed site cannot fit.
    fn config(rank: u32) -> Qwen35Config {
        Qwen35Config { lora: Some(lora_cfg(rank, rank as f32)), ..Qwen35Config::tiny() }
    }

    /// The sites are exactly the adapters the config's own parameter list
    /// allocates: same names, same element counts, none missing, none extra.
    #[test]
    fn the_sites_are_the_adapters_param_list_allocates() {
        let cfg = config(3);
        let params: std::collections::HashMap<String, usize> = cfg.param_list().into_iter().collect();
        let sites = cfg.lora_sites();
        // Three GDN layers of 5 + 3 sites and one full-attention layer of 4 + 3.
        assert_eq!(sites.len(), 3 * 8 + 7);
        for site in &sites {
            let (name_a, name_b) = tensor_names(site);
            let (a, b) = tensor_shapes(site, 3);
            assert_eq!(params.get(&name_a), Some(&(a[0] * a[1])), "{name_a}");
            assert_eq!(params.get(&name_b), Some(&(b[0] * b[1])), "{name_b}");
            assert!(params.contains_key(&site.name), "{} is a base weight", site.name);
        }
        assert_eq!(params.keys().filter(|n| n.ends_with(".lora_a")).count(), sites.len(), "no adapter outside the sites");
    }

    #[test]
    fn an_adapter_of_the_right_shapes_validates_and_a_swapped_site_does_not() {
        let cfg = config(3);
        let sites = cfg.lora_sites();
        let pair = |s: &LinearSite| {
            let (a, b) = tensor_shapes(s, 3);
            AdapterSite { base: s.name.clone(), a: vec![0.0; a[0] * a[1]], b: vec![0.0; b[0] * b[1]] }
        };
        let adapter = DeviceAdapter { rank: 3, alpha: 3.0, sites: sites.iter().map(pair).collect() };
        assert_eq!(validate(&adapter, &sites), Ok(()));
        // The second site's pair filed under the first's name: same rank, wrong widths.
        let mut wrong = adapter.clone();
        wrong.sites[0] = AdapterSite { base: sites[0].name.clone(), ..pair(&sites[1]) };
        assert!(validate(&wrong, &sites).is_err());
    }
}
