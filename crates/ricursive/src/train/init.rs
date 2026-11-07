// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The starting point of an adapter run: a base checkpoint, streamed as it is stored, with the adapter's factors
//! laid over it.
//!
//! Swedish Embedded AB implements solutions for fine-tuning very large dense decoders on the memory a team
//! actually has, starting adapters over checkpoints that never fit in host memory as f32. If your team needs
//! expertise in parameter-efficient training of large language models then you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::collections::HashMap;

use checkpoint::packed::{LoadReport, LoadRequest, TensorDescriptor, TensorSink};
use checkpoint::TensorSource;

use super::TrainConfig;

/// A [`TensorSource`] that answers the adapter's tensors from memory and every other tensor from the base it lies
/// over. The base is never copied: it stays behind its own (possibly packed, memory-mapped) view, so a bf16 or
/// NVFP4 release uploads in its stored encoding.
pub struct AdapterInit<'a> {
    base: &'a dyn TensorSource,
    adapters: HashMap<String, Vec<f32>>,
}

impl<'a> AdapterInit<'a> {
    /// The factors of `cfg`'s adapter freshly initialised from `seed` (`A` drawn, `B` zero: the adapter starts as
    /// the identity) over `base`.
    pub fn fresh(cfg: &TrainConfig, base: &'a dyn TensorSource, seed: u64) -> AdapterInit<'a> {
        let adapters = cfg.init_weights(seed).into_iter().filter(|(name, _)| model::adapter::device::is_adapter_param(name)).collect();
        AdapterInit { base, adapters }
    }

    /// The factors of `adapter` (an adapter written by a previous run) over `base`, to continue its training. The
    /// adapter must be the model's: its rank and every site are checked against `cfg`.
    pub fn continued(cfg: &TrainConfig, base: &'a dyn TensorSource, adapter: &model::lora::device_adapter::DeviceAdapter) -> Result<AdapterInit<'a>, String> {
        use model::adapter::sites::LoraSites;
        let lora = cfg.lora.as_ref().ok_or("ricursive: the configuration trains no adapter")?;
        if adapter.rank != lora.rank {
            return Err(format!("ricursive: the adapter has rank {}, the run trains rank {}", adapter.rank, lora.rank));
        }
        model::adapter::sites::validate(adapter, &cfg.lora_sites()).map_err(|e| format!("ricursive: {e}"))?;
        let mut adapters = Self::fresh(cfg, base, 0).adapters;
        for pair in &adapter.sites {
            let (a, b) = model::adapter::device::lora_names(&pair.base);
            if !adapters.contains_key(&a) {
                return Err(format!("ricursive: the run's adapter does not target {}", pair.base));
            }
            adapters.insert(a, pair.a.clone());
            adapters.insert(b, pair.b.clone());
        }
        Ok(AdapterInit { base, adapters })
    }

    fn owner(&self, name: &str) -> &dyn TensorSource {
        if self.adapters.contains_key(name) {
            &self.adapters
        } else {
            self.base
        }
    }
}

impl TensorSource for AdapterInit<'_> {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        self.owner(name).with_tensor(name, f)
    }
    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        self.owner(name).raw_words(name)
    }
    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        self.owner(name).with_tensor_chunks(name, max_elems, f)
    }
    fn with_tensor_range(&self, name: &str, start: usize, len: usize, f: &mut dyn FnMut(&[f32])) -> bool {
        self.owner(name).with_tensor_range(name, start, len, f)
    }
    fn with_tensor_u32_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[u32])) -> bool {
        self.owner(name).with_tensor_u32_chunks(name, max_elems, f)
    }
    fn numel(&self, name: &str) -> Option<usize> {
        self.owner(name).numel(name)
    }
    fn raw_blocks(&self, name: &str) -> Option<(checkpoint::gguf::BlockLayout, std::borrow::Cow<'_, [u8]>)> {
        self.owner(name).raw_blocks(name)
    }
    fn raw_block_layout(&self, name: &str) -> Option<checkpoint::gguf::BlockLayout> {
        self.owner(name).raw_block_layout(name)
    }
    fn advise_drop(&self, name: &str) {
        self.owner(name).advise_drop(name)
    }
    fn packed_descriptor(&self, name: &str) -> Option<TensorDescriptor> {
        if self.adapters.contains_key(name) {
            None
        } else {
            self.base.packed_descriptor(name)
        }
    }
    fn load_packed(&self, names: &[&str], req: &LoadRequest<'_>, sink: &mut dyn TensorSink) -> Result<Option<LoadReport>, String> {
        if names.iter().any(|n| self.adapters.contains_key(*n)) {
            return Ok(None);
        }
        self.base.load_packed(names, req, sink)
    }
}

#[cfg(test)]
mod tests {
    use model::adapter::device::LoraCfg;

    use super::*;
    use crate::train::default_targets;

    #[test]
    fn adapter_tensors_come_from_memory_and_the_rest_from_the_base() {
        let cfg = TrainConfig::tiny().with_lora(LoraCfg::new(2, 4.0, default_targets()));
        let base = TrainConfig::tiny().tiny_weights(1);
        let init = AdapterInit::fresh(&cfg, &base, 3);
        let mut b_zero = false;
        assert!(init.with_tensor("model.layers.0.self_attn.q_proj.weight.lora_b", &mut |d| b_zero = d.iter().all(|x| *x == 0.0)));
        assert!(b_zero, "B starts at zero");
        let mut got = Vec::new();
        assert!(init.with_tensor("model.norm.weight", &mut |d| got = d.to_vec()));
        assert_eq!(got, base["model.norm.weight"]);
        assert!(!init.with_tensor("model.layers.0.nope", &mut |_| {}));
    }

    #[test]
    fn a_continued_adapter_must_be_the_runs_rank() {
        let cfg = TrainConfig::tiny().with_lora(LoraCfg::new(2, 4.0, default_targets()));
        let base = TrainConfig::tiny().tiny_weights(1);
        let wrong = model::lora::device_adapter::DeviceAdapter { rank: 3, alpha: 4.0, sites: Vec::new() };
        assert!(AdapterInit::continued(&cfg, &base, &wrong).err().unwrap().contains("rank 3"));
    }
}
