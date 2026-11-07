// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight tiers: which format each of Ripple's linear weights is resident in, and the one loader that puts it
//! there from whatever the checkpoint stores.
//!
//! Swedish Embedded AB implements solutions for fitting very large dense decoders into the device memory a
//! deployment actually has, loading quantized releases as published, for its clients. If your team needs
//! expertise in bf16, FP8, int8 and NVFP4 serving of large language models then you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! Nothing here is Ripple-specific machinery: the tier vocabulary is `gpu_core::select::Dtype` and
//! `model::ops::TierPolicy`, the codecs and the tiled load are `checkpoint::packed`, the resident forms are
//! `model::ops::Weight`, and the load is `model::ops::upload_from_source`. This module only names Ripple's
//! linears, decides each one's tier, and refuses what the device cannot hold as asked.
//!
//! # The tier of a weight
//!
//! * `requested == None` (the default): every linear is held **as the checkpoint stores it**: a bf16 release
//!   stays bf16 (no f32 blow-up at load), the official NVFP4 release keeps its MLP in NVFP4 and its attention
//!   and `lm_head` in bf16.
//! * `requested == Some(policy)`: the policy's tier per linear (`--weight-format`), converting from the stored
//!   encoding tile by tile. NVFP4 can only be held as stored: there is no tier to quantize *into* it.
//!
//! A tier the device cannot hold is an error naming the tensor, the tier asked and the tier the device would
//! have used, never a silent widening.

use checkpoint::packed::{BoundedWorkspace, MemoryBudget};
use checkpoint::TensorSource;
use gpu_core::select::Dtype;
use model::ops::{stored_tier, upload_from_source, Ops, TierPolicy, Weight};

use crate::config::RicursiveConfig;
use crate::import::layer_tensor;

/// Host staging one tile of a packed load may take.
const STAGING_BYTES: u64 = 64 << 20;

/// The linear weights of one layer, as `(suffix, rows, columns)`.
pub(crate) fn layer_linears(cfg: &RicursiveConfig) -> Vec<(&'static str, usize, usize)> {
    let (h, f) = (cfg.hidden_size, cfg.intermediate_size);
    let mut v = vec![
        ("self_attn.q_proj.weight", cfg.q_dim(), h),
        ("self_attn.k_proj.weight", cfg.kv_dim(), h),
        ("self_attn.v_proj.weight", cfg.kv_dim(), h),
        ("self_attn.o_proj.weight", h, cfg.q_dim()),
        ("mlp.gate_proj.weight", f, h),
        ("mlp.up_proj.weight", f, h),
        ("mlp.down_proj.weight", h, f),
    ];
    if cfg.use_gated_attention {
        v.push(("self_attn.g_proj.weight", cfg.q_dim(), h));
    }
    v
}

/// Every linear weight of the model by canonical name, with its shape: the layers' linears and `lm_head`.
pub(crate) fn all_linears(cfg: &RicursiveConfig) -> Vec<(String, usize, usize)> {
    let mut v: Vec<(String, usize, usize)> = (0..cfg.num_hidden_layers).flat_map(|l| layer_linears(cfg).into_iter().map(move |(s, n, k)| (layer_tensor(l, s), n, k))).collect();
    v.push(("lm_head.weight".to_string(), cfg.vocab_size, cfg.hidden_size));
    v
}

/// Every linear weight of `cfg` with its shape `(name, rows, columns)`: what a weight format applies to.
pub fn all_linears_of(cfg: &RicursiveConfig) -> Vec<(String, usize, usize)> {
    all_linears(cfg)
}

/// The canonical names of every linear weight of `cfg`, the layers' and `lm_head`: the tensors a weight format applies to.
pub fn linear_names(cfg: &RicursiveConfig) -> Vec<String> {
    all_linears(cfg).into_iter().map(|(name, ..)| name).collect()
}

/// The tier policy a load runs under: `requested`, or the encodings `src` stores its linears in.
pub fn resolve_tiers(cfg: &RicursiveConfig, src: &dyn TensorSource, requested: Option<&TierPolicy>) -> TierPolicy {
    if let Some(p) = requested {
        return p.clone();
    }
    // Per tensor, a rule naming it in full (`model.layers.1.` is not inside `model.layers.11.`), with the tier
    // holding the most weights the default.
    let leaves = all_linears(cfg).into_iter().filter_map(|(name, n, k)| stored_tier(src, &name).map(|tier| (name, tier, (n * k) as u64)));
    let stored = TierPolicy::per_leaf(leaves);
    if stored.dtypes().all(|t| t == Dtype::F32) {
        TierPolicy::uniform(Dtype::F32)
    } else {
        stored
    }
}

/// What a load put on the device.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WeightReport {
    /// Per tier, `(tier, tensors, resident bytes)`, in first-seen order.
    pub tiers: Vec<(Dtype, usize, u64)>,
}

impl WeightReport {
    fn add(&mut self, tier: Dtype, bytes: u64) {
        match self.tiers.iter_mut().find(|(t, ..)| *t == tier) {
            Some((_, n, b)) => {
                *n += 1;
                *b += bytes;
            }
            None => self.tiers.push((tier, 1, bytes)),
        }
    }

    /// Add the tensors and bytes of `other`.
    pub(crate) fn absorb(&mut self, other: &WeightReport) {
        for &(tier, n, bytes) in &other.tiers {
            match self.tiers.iter_mut().find(|(t, ..)| *t == tier) {
                Some((_, count, total)) => {
                    *count += n;
                    *total += bytes;
                }
                None => self.tiers.push((tier, n, bytes)),
            }
        }
    }

    /// Resident bytes of every linear weight.
    pub fn bytes(&self) -> u64 {
        self.tiers.iter().map(|(.., b)| b).sum()
    }

    /// Tensors held at `tier`.
    pub fn count(&self, tier: Dtype) -> usize {
        self.tiers.iter().find(|(t, ..)| *t == tier).map_or(0, |(_, n, _)| *n)
    }

    /// `bf16 x 400 (…), nvfp4 x 240 (…)`, for a log line.
    pub fn describe(&self) -> String {
        self.tiers.iter().map(|(t, n, b)| format!("{} x{n} ({:.2} GiB)", model::policy::format_name(*t), *b as f64 / (1u64 << 30) as f64)).collect::<Vec<_>>().join(", ")
    }
}

/// Uploads linear weights at the tier the policy gives each, from a checkpoint source.
pub(crate) struct Loader<'a> {
    ops: &'a Ops,
    src: &'a dyn TensorSource,
    tiers: &'a TierPolicy,
    staging: BoundedWorkspace,
    report: std::cell::RefCell<WeightReport>,
}

impl<'a> Loader<'a> {
    pub(crate) fn new(ops: &'a Ops, src: &'a dyn TensorSource, tiers: &'a TierPolicy) -> Loader<'a> {
        Loader { ops, src, tiers, staging: BoundedWorkspace::new(MemoryBudget::new(STAGING_BYTES, 0)), report: Default::default() }
    }

    pub(crate) fn ops(&self) -> &'a Ops {
        self.ops
    }

    pub(crate) fn report(&self) -> WeightReport {
        self.report.borrow().clone()
    }

    /// The `[n, k]` linear `name`, resident at the tier the policy gives it.
    pub(crate) fn linear(&self, name: &str, n: usize, k: usize) -> Result<Weight, String> {
        let want = self.tiers.want(name);
        if want == Dtype::F4E2M1 && stored_tier(self.src, name) != Some(Dtype::F4E2M1) {
            return Err(format!("ricursive: {name} is asked for as NVFP4 but the checkpoint does not store it that way (NVFP4 is held as the release stores it; nothing quantizes into it)"));
        }
        if !Weight::packs(want, n, k) && want != Dtype::F4E2M1 {
            return Err(format!("ricursive: {name} [{n}, {k}] cannot be held as {}: its inner width does not tile into that format's blocks", model::policy::format_name(want)));
        }
        let limit = self.ops.gpu().max_storage_binding_bytes();
        let bytes = Weight::bytes_for(want, n, k);
        if bytes > limit {
            return Err(format!("ricursive: {name} needs a {bytes}-byte binding as {} but this device binds at most {limit} bytes; choose a narrower weight format", model::policy::format_name(want)));
        }
        let weight = upload_from_source(self.ops, self.src, name, n, k, want, &self.staging).ok_or_else(|| format!("ricursive: source has no tensor {name}"))?;
        if weight.dtype() != want {
            return Err(format!(
                "ricursive: {name} is asked for as {} but this device holds that tier as {}; use a format this backend executes (the CPU JIT has no packed-int8 path, for one)",
                model::policy::format_name(want),
                model::policy::format_name(weight.dtype())
            ));
        }
        self.report.borrow_mut().add(want, weight.resident_bytes());
        Ok(weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_layer_has_seven_linears_and_an_eighth_with_gated_attention() {
        let cfg = RicursiveConfig::from_json(r#"{"vocab_size":16,"hidden_size":8,"intermediate_size":12,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,"use_gated_attention":false}"#).unwrap();
        assert_eq!(layer_linears(&cfg).len(), 7);
        let gated = RicursiveConfig::from_json(r#"{"vocab_size":16,"hidden_size":8,"intermediate_size":12,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":4}"#).unwrap();
        assert_eq!(layer_linears(&gated).len(), 8);
        assert_eq!(all_linears(&gated).len(), 2 * 8 + 1);
    }

    #[test]
    fn the_report_counts_tensors_and_bytes_per_tier() {
        let mut r = WeightReport::default();
        r.add(Dtype::BF16, 10);
        r.add(Dtype::F4E2M1, 3);
        r.add(Dtype::BF16, 10);
        assert_eq!((r.count(Dtype::BF16), r.count(Dtype::F4E2M1), r.count(Dtype::I8)), (2, 1, 0));
        assert_eq!(r.bytes(), 23);
    }
}
