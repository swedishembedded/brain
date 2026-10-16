// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The adapter sites of a Qwen3 config: the seven linears of each block
//! (`wq`, `wk`, `wv`, `wo`, `gate`, `up`, `down`), as the generic
//! `model::adapter::sites::LoraSites` lists them. The dimensions are the ones
//! `QwenConfig::param_list` gives an adapter's `.lora_a`/`.lora_b`.
//!
//! Swedish Embedded AB implements solutions for loading and serving adapters
//! across model families without a loader per architecture, for its clients.
//! If your team needs expertise in parameter-efficient fine-tuning
//! infrastructure you can procure our services by sending an email to
//! info@swedishembedded.com.

use model::adapter::bank::AdapterServing;
use model::adapter::sites::LoraSites;
use model::adapter::{LinearSite, TargetSpec};

use crate::config::QwenConfig;

impl LoraSites for QwenConfig {
    fn lora_sites(&self) -> Vec<LinearSite> {
        let (d, ff, hq, hkv) = (self.d_model as usize, self.d_ff as usize, self.q_dim() as usize, self.kv_dim() as usize);
        let linears: [(&'static str, &str, usize, usize); 7] = [
            ("wq", "attn.wq.weight", hq, d),
            ("wk", "attn.wk.weight", hkv, d),
            ("wv", "attn.wv.weight", hkv, d),
            ("wo", "attn.wo.weight", d, hq),
            ("gate", "mlp.gate.weight", ff, d),
            ("up", "mlp.up.weight", ff, d),
            ("down", "mlp.down.weight", d, ff),
        ];
        (0..self.n_layers as usize)
            .flat_map(|layer| {
                linears.iter().map(move |&(leaf, tail, out, inn)| LinearSite {
                    name: format!("blocks.{layer}.{tail}"),
                    leaf,
                    layer: Some(layer),
                    spec: TargetSpec::whole(out, inn),
                    save_name: None,
                })
            })
            .collect()
    }
}

/// The defaults: `attention` is `wq wk wv wo`, 4 slots, rank cap 16.
impl AdapterServing for QwenConfig {}

#[cfg(test)]
mod tests {
    use model::adapter::sites::{tensor_names, tensor_shapes, validate};
    use model::lora::device_adapter::{AdapterSite, DeviceAdapter};

    use super::*;
    use crate::config::LoraCfg;

    /// Distinct, non-square widths: d_model 16, q_dim 24, kv_dim 12, d_ff 40.
    fn config(rank: u32) -> QwenConfig {
        let targets = ["wq", "wk", "wv", "wo", "gate", "up", "down"].iter().map(|s| s.to_string()).collect();
        QwenConfig { head_dim: 6, d_ff: 40, lora: Some(LoraCfg { rank, alpha: rank as f32, targets }), ..QwenConfig::tiny() }
    }

    /// The sites are exactly the adapters the config's own parameter list
    /// allocates: same names, same element counts.
    #[test]
    fn the_sites_are_the_adapters_param_list_allocates() {
        let cfg = config(2);
        let params: std::collections::HashMap<String, usize> = cfg.param_list().into_iter().collect();
        let sites = cfg.lora_sites();
        assert_eq!(sites.len(), 7 * cfg.n_layers as usize);
        for site in &sites {
            let (name_a, name_b) = tensor_names(site);
            let (a, b) = tensor_shapes(site, 2);
            assert_eq!(params.get(&name_a), Some(&(a[0] * a[1])), "{name_a}");
            assert_eq!(params.get(&name_b), Some(&(b[0] * b[1])), "{name_b}");
            assert!(params.contains_key(&site.name), "{} is a base weight", site.name);
        }
        assert_eq!(params.keys().filter(|n| n.ends_with(".lora_a")).count(), sites.len(), "no adapter outside the sites");
    }

    #[test]
    fn an_adapter_of_the_right_shapes_validates_and_a_swapped_site_does_not() {
        let cfg = config(2);
        let sites = cfg.lora_sites();
        let pair = |s: &LinearSite| {
            let (a, b) = tensor_shapes(s, 2);
            AdapterSite { base: s.name.clone(), a: vec![0.0; a[0] * a[1]], b: vec![0.0; b[0] * b[1]] }
        };
        let adapter = DeviceAdapter { rank: 2, alpha: 2.0, sites: sites.iter().map(pair).collect() };
        assert_eq!(validate(&adapter, &sites), Ok(()));
        // A wk pair filed under wq's name: same rank, wrong widths (12 vs 24 outputs).
        let mut wrong = adapter.clone();
        wrong.sites[0] = AdapterSite { base: sites[0].name.clone(), ..pair(&sites[1]) };
        assert!(validate(&wrong, &sites).is_err());
    }
}
