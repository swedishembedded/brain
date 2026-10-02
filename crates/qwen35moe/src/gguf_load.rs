// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.5/3.6-35B-A3B straight from its released GGUF: a streaming
//! [`checkpoint::TensorSource`] over the memory-mapped file, with brain's
//! parameter names, the experts sliced out of llama.cpp's stacked tensors and
//! the Gated-DeltaNet value-head order undone.
//!
//! Swedish Embedded AB implements GPU-resident serving of large sparse MoE
//! models from their released quantised checkpoints for its clients. If your
//! team needs expertise in loading a 35 GB checkpoint onto one accelerator
//! without an fp32 intermediate then you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! No fp32 copy of the model exists anywhere on this path: the offline
//! importer ([`crate::import`]) would write ~140 GB, and a host-side
//! `HashMap<String, Vec<f32>>` the same in RAM. Here the `.gguf` IS the load
//! format - the device builders pull one tensor at a time through
//! `raw_blocks` (a Q8_0 tensor becomes brain's packed int8 as a byte repack,
//! `gguf::int8_direct`) or, where a tensor must be transformed, through
//! `with_tensor`.
//!
//! The name map is [`crate::import::classify`] - the same classifier the
//! offline converter drives - so there is one spelling table, and a rename
//! of a leaf breaks both routes loudly. Each routed expert is a
//! [`Fetch::Slice`] of llama.cpp's `[n_experts, out, in]` stack (expert `e`
//! is the contiguous element range `e*out*in .. (e+1)*out*in`), which keeps a
//! Q8_0 expert on the zero-copy block path; the whole stack is also exposed
//! under one name per layer and projection ([`bank_name`]) for a builder that
//! wants the fused bank.

use std::collections::HashMap;

use checkpoint::gguf::MmapGguf;
use checkpoint::remap::{Fetch, RemapSource};
use gguf::import::Mapped;
use gguf::GdnFixSource;

use crate::config::Qwen35Config;

/// The brain name under which a layer's whole expert stack of one projection
/// (`gate`, `up` or `down`) is served: every expert's `[out, in]` matrix back
/// to back in expert order, exactly llama.cpp's own layout.
pub fn bank_name(layer: usize, proj: &str) -> String {
    format!("blocks.{layer}.mlp.experts.{proj}.bank")
}

/// [`crate::import::config_from_gguf`] with `block_size` set to the
/// per-sequence capacity the caller will serve (purely descriptive here).
pub fn resident_config(mg: &MmapGguf, cap: u32) -> Result<Qwen35Config, String> {
    let mut cfg = crate::import::config_from_gguf(mg)?;
    cfg.block_size = cap;
    Ok(cfg)
}

/// The name-to-[`Fetch`] plan over `mg`: every parameter in
/// `cfg.param_list()` plus one [`bank_name`] per layer and projection.
///
/// Fails by name when the GGUF offers no tensor for a planned parameter.
pub fn fetch_plan(mg: &MmapGguf, cfg: &Qwen35Config) -> Result<HashMap<String, Fetch>, String> {
    let mut plan = HashMap::with_capacity(cfg.param_list().len() + 3 * cfg.n_layers as usize);
    for name in mg.names() {
        match crate::import::classify(name, cfg) {
            Mapped::Simple(brain) | Mapped::Transformed { into: brain, .. } => {
                plan.insert(brain, Fetch::Whole(name.clone()));
            }
            Mapped::Split { into } => {
                let numel = mg.shape(name).map(|s| s.iter().product::<usize>()).ok_or_else(|| format!("{name}: no shape"))?;
                if into.is_empty() || numel % into.len() != 0 {
                    return Err(format!("{name}: {numel} elements do not split into {} equal experts", into.len()));
                }
                let len = numel / into.len();
                for (e, brain) in into.into_iter().enumerate() {
                    plan.insert(brain, Fetch::Slice { name: name.clone(), start: e * len, len });
                }
                if let Some((layer, proj)) = expert_stack_of(name) {
                    plan.insert(bank_name(layer, proj), Fetch::Whole(name.clone()));
                }
            }
            Mapped::Permuted { .. } | Mapped::Dropped(_) => {}
        }
    }
    let missing: Vec<String> = cfg.param_list().into_iter().map(|(n, _)| n).filter(|n| !plan.contains_key(n)).collect();
    if !missing.is_empty() {
        let sample: Vec<&String> = missing.iter().take(5).collect();
        return Err(format!("qwen35moe gguf: no tensor for {} planned parameter(s), e.g. {sample:?}", missing.len()));
    }
    Ok(plan)
}

/// `(layer, "gate"|"up"|"down")` for a llama.cpp `blk.N.ffn_{proj}_exps.weight`.
fn expert_stack_of(gguf_name: &str) -> Option<(usize, &'static str)> {
    let rest = gguf_name.strip_prefix("blk.")?;
    let (layer, leaf) = rest.split_once('.')?;
    let proj = match leaf {
        "ffn_gate_exps.weight" => "gate",
        "ffn_up_exps.weight" => "up",
        "ffn_down_exps.weight" => "down",
        _ => return None,
    };
    Some((layer.parse().ok()?, proj))
}

/// A live, transform-applying [`checkpoint::TensorSource`] over `mg`, with the
/// plan checked against every parameter's declared element count BEFORE a byte
/// is uploaded (`RemapSource::validate` reads shapes only) - a config-vs-file
/// mismatch is one named error, not a panic gigabytes into a load.
pub fn source<'a>(mg: &'a MmapGguf, cfg: &Qwen35Config) -> Result<GdnFixSource<'a>, String> {
    let plan = fetch_plan(mg, cfg)?;
    let remap = RemapSource::new(mg, plan);
    remap.validate(&cfg.param_list())?;
    Ok(GdnFixSource::new(remap, cfg.gdn_head_order()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use checkpoint::TensorSource;

    fn synthetic(tag: &str) -> (String, MmapGguf) {
        let path = std::env::temp_dir().join(format!("qwen35moe-gguf-load-{tag}-{}.gguf", std::process::id())).to_string_lossy().into_owned();
        crate::import::testing::write_synthetic_gguf(&path);
        let mg = MmapGguf::open(&path).unwrap();
        (path, mg)
    }

    /// The source serves every parameter the config lists, at the right size,
    /// and each expert is its own contiguous slice of the stacked tensor.
    #[test]
    fn the_source_covers_every_parameter_and_slices_experts() {
        let (path, mg) = synthetic("cover");
        let cfg = resident_config(&mg, 64).unwrap();
        let src = source(&mg, &cfg).expect("a plan that covers the whole parameter list");
        for (name, numel) in cfg.param_list() {
            assert_eq!(src.numel(&name), Some(numel), "{name}");
        }
        let stack = mg.tensor("blk.1.ffn_down_exps.weight").unwrap().unwrap();
        let chunk = (cfg.d_model * cfg.moe_intermediate_size) as usize;
        for e in 0..cfg.n_experts as usize {
            let mut got = Vec::new();
            assert!(src.with_tensor(&format!("blocks.1.mlp.experts.{e}.down.weight"), &mut |d| got = d.to_vec()));
            assert_eq!(got, &stack[e * chunk..(e + 1) * chunk], "expert {e}");
        }
        std::fs::remove_file(&path).ok();
    }

    /// The fused bank is the whole stack, byte-for-byte llama.cpp's layout.
    #[test]
    fn a_bank_is_the_whole_stack_in_expert_order() {
        let (path, mg) = synthetic("bank");
        let cfg = resident_config(&mg, 64).unwrap();
        let src = source(&mg, &cfg).unwrap();
        let stack = mg.tensor("blk.0.ffn_gate_exps.weight").unwrap().unwrap();
        let mut got = Vec::new();
        assert!(src.with_tensor(&bank_name(0, "gate"), &mut |d| got = d.to_vec()));
        assert_eq!(got, stack);
        assert_eq!(src.numel(&bank_name(0, "gate")), Some((cfg.n_experts * cfg.moe_intermediate_size * cfg.d_model) as usize));
        std::fs::remove_file(&path).ok();
    }

    /// The value-head order is undone through the live source too - the same
    /// transform the offline converter applies (`import_regroups_...`).
    #[test]
    fn the_source_regroups_the_value_heads_like_the_converter() {
        let (path, mg) = synthetic("regroup");
        let cfg = resident_config(&mg, 64).unwrap();
        let src = source(&mg, &cfg).unwrap();
        let mut dt_bias = Vec::new();
        assert!(src.with_tensor("blocks.0.linear_attn.dt_bias", &mut |d| dt_bias = d.to_vec()));
        for (g, w) in dt_bias.iter().zip([0.1f32, 0.3, 0.2, 0.4]) {
            assert!((g - w).abs() < 1e-5, "{dt_bias:?}");
        }
        assert!(src.raw_blocks("blocks.0.linear_attn.in_proj_z.weight").is_none(), "a transformed leaf must not be lent");
        std::fs::remove_file(&path).ok();
    }
}
