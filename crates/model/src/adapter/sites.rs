// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The adapter sites of a model config, and the one validation of a loaded
//! device-side adapter against them.
//!
//! A config that implements [`LoraSites`] lists every linear an adapter may
//! correct (a [`LinearSite`]: the base weight's name, its layer and its
//! `[out, in]` dimensions). A device-side LoRA adapter stores, per site,
//! `{site}.lora_a` of shape `[r, in]` and `{site}.lora_b` of shape `[out, r]`
//! (`{site}` is the base weight name, e.g. `blocks.0.attn.wq.weight`);
//! [`validate`] checks a loaded [`DeviceAdapter`] against the sites, so a
//! model that serves or trains adapters needs no checker of its own.
//!
//! Swedish Embedded AB implements solutions for loading and serving adapters
//! across model families without a loader per architecture, for its clients.
//! If your team needs expertise in parameter-efficient fine-tuning
//! infrastructure you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::fmt;

use super::kind::{LORA_A, LORA_B};
use super::LinearSite;
use crate::lora::device_adapter::DeviceAdapter;

/// A model config that can list the linears an adapter may correct, in
/// block order. Independent of which of them a particular adapter targets.
pub trait LoraSites {
    fn lora_sites(&self) -> Vec<LinearSite>;

    /// The stacked expert banks an adapter may correct expert by expert, in block order ([`ExpertSite`]). None by
    /// default: a model with routed experts lists them.
    fn expert_sites(&self) -> Vec<ExpertSite> {
        Vec::new()
    }

    /// The small matrices an operator reads whole (a bank of learned queries, a gate's factors read by a fused
    /// operator), which an adapter corrects with an ordinary pair (`{name}.lora_a [r, in]`, `{name}.lora_b [out, r]`)
    /// but a serving bank cannot correct beside a product: it serves each as a per-slot effective matrix
    /// `W + (alpha / rank) B A` ([`crate::adapter::bank::AdapterBank::gather_merged`]). None by default.
    fn merged_sites(&self) -> Vec<LinearSite> {
        Vec::new()
    }
}

/// One projection of a mixture-of-experts layer stored as a stacked bank of `experts` matrices `[out, in]` (one
/// `[experts, out, in]` tensor, the routed experts' gate/up or down projection), which an adapter corrects with one
/// low-rank pair per expert: `{name}.lora_a` of shape `[experts, r, in]` and `{name}.lora_b` of shape
/// `[experts, out, r]` ([`crate::lora::device_adapter::ExpertPair`]). Expert `e`'s pair corrects expert `e`'s matrix
/// on the tokens routed to it, and only there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertSite {
    /// The bank's name, the stem of its adapter tensors (`layers.20.mlp.experts.w13`).
    pub name: String,
    /// The leaf a target list names it by (`w13`).
    pub leaf: &'static str,
    /// The layer it belongs to.
    pub layer: Option<usize>,
    pub experts: usize,
    /// Output rows of each expert's matrix (`2 * moe_ff` for a fused gate/up bank).
    pub out: usize,
    /// Inputs of each expert's matrix.
    pub inn: usize,
    /// Experts each token routes to: a serving bank holds this many expert slots per row.
    pub top_k: usize,
}

/// The shapes of the two tensors a rank-`rank` adapter stores for an expert `site`: `A [experts, rank, in]` and
/// `B [experts, out, rank]`.
pub fn expert_tensor_shapes(site: &ExpertSite, rank: usize) -> ([usize; 3], [usize; 3]) {
    ([site.experts, rank, site.inn], [site.experts, site.out, rank])
}

/// The names of the two tensors an adapter stores for `site`.
pub fn tensor_names(site: &LinearSite) -> (String, String) {
    (super::device::param_name(&site.name, LORA_A), super::device::param_name(&site.name, LORA_B))
}

/// The shapes of the two tensors a rank-`rank` adapter stores for `site`:
/// `A [rank, in]` and `B [out, rank]`.
pub fn tensor_shapes(site: &LinearSite, rank: usize) -> ([usize; 2], [usize; 2]) {
    ([rank, site.spec.inn], [site.spec.out, rank])
}

/// Why an adapter does not fit a model's sites.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SiteMismatch {
    /// The adapter corrects a weight the model has no site for.
    UnknownSite { name: String },
    /// The adapter carries a full tensor this model does not serve or train in full.
    UnsavableTensor { name: String },
    /// A full tensor is not the size of the base tensor it replaces.
    SavedShape { name: String, expected: usize, got: usize },
    /// A pair's factors are not the rank-`rank` shapes of the site.
    Shape { name: String, rank: u32, expected_a: [usize; 2], expected_b: [usize; 2], got_a: usize, got_b: usize },
    /// A stacked expert pair is not the rank-`rank` shapes of its expert site.
    ExpertShape { name: String, rank: u32, expected_a: [usize; 3], expected_b: [usize; 3], got_experts: usize, got_a: usize, got_b: usize },
}

impl fmt::Display for SiteMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SiteMismatch::UnknownSite { name } => write!(f, "{name} is not an adapter site of this model"),
            SiteMismatch::UnsavableTensor { name } => write!(f, "{name}: the adapter carries this tensor in full (modules_to_save), which this model does not apply; serve the adapter on an architecture that does, or fold it into the base"),
            SiteMismatch::SavedShape { name, expected, got } => write!(f, "{name}: the adapter carries {got} values for a tensor of {expected}"),
            SiteMismatch::Shape { name, rank, expected_a, expected_b, got_a, got_b } => write!(
                f,
                "{name}: a rank-{rank} adapter needs A {expected_a:?} ({} values) and B {expected_b:?} ({} values) but holds {got_a} and {got_b}",
                expected_a[0] * expected_a[1],
                expected_b[0] * expected_b[1]
            ),
            SiteMismatch::ExpertShape { name, rank, expected_a, expected_b, got_experts, got_a, got_b } => write!(
                f,
                "{name}: a rank-{rank} stacked adapter needs A {expected_a:?} and B {expected_b:?} but holds {got_experts} experts of {got_a} and {got_b} values in all",
            ),
        }
    }
}

impl std::error::Error for SiteMismatch {}

/// Check that every pair of `adapter` belongs to one of `sites` and has the
/// shapes of that site at the adapter's rank. The adapter need not cover every
/// site (a rank-r adapter on the attention projections only is valid).
///
/// An adapter that carries full tensors (`modules_to_save`) is refused here: a model that applies them calls
/// [`validate_with_saved`] with the tensors it can, so a consumer that knows nothing of them never drops them silently.
pub fn validate(adapter: &DeviceAdapter, sites: &[LinearSite]) -> Result<(), SiteMismatch> {
    validate_with_saved(adapter, sites, &[])
}

/// A base tensor a model applies in full when an adapter carries one: its name and element count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SavedSite {
    pub name: String,
    pub numel: usize,
    /// The layer it belongs to, `None` for a tensor outside the layers (the final norm).
    pub layer: Option<usize>,
}

/// [`validate`] for a model that applies the full tensors in `saved_sites`: each full tensor of the adapter must be
/// one of them, at its size. A stacked expert pair is refused ([`SiteMismatch::UnknownSite`]):
/// [`validate_with_experts`] is the check of a model that has expert sites.
pub fn validate_with_saved(adapter: &DeviceAdapter, sites: &[LinearSite], saved_sites: &[SavedSite]) -> Result<(), SiteMismatch> {
    validate_with_experts(adapter, sites, &[], saved_sites)
}

/// [`validate_with_saved`] for a model with stacked expert banks: each of the adapter's expert pairs must belong to
/// one of `expert_sites`, with its expert count and the site's rank-`rank` shapes ([`expert_tensor_shapes`]).
pub fn validate_with_experts(adapter: &DeviceAdapter, sites: &[LinearSite], expert_sites: &[ExpertSite], saved_sites: &[SavedSite]) -> Result<(), SiteMismatch> {
    let rank = adapter.rank as usize;
    for pair in &adapter.experts {
        let site = expert_sites.iter().find(|s| s.name == pair.base).ok_or_else(|| SiteMismatch::UnknownSite { name: pair.base.clone() })?;
        let (expected_a, expected_b) = expert_tensor_shapes(site, rank);
        if pair.experts != site.experts || pair.a.len() != expected_a.iter().product::<usize>() || pair.b.len() != expected_b.iter().product::<usize>() {
            return Err(SiteMismatch::ExpertShape { name: pair.base.clone(), rank: adapter.rank, expected_a, expected_b, got_experts: pair.experts, got_a: pair.a.len(), got_b: pair.b.len() });
        }
    }
    for t in &adapter.saved {
        let site = saved_sites.iter().find(|s| s.name == t.base).ok_or_else(|| SiteMismatch::UnsavableTensor { name: t.base.clone() })?;
        if site.numel != t.data.len() {
            return Err(SiteMismatch::SavedShape { name: t.base.clone(), expected: site.numel, got: t.data.len() });
        }
    }
    for pair in &adapter.sites {
        let site = sites.iter().find(|s| s.name == pair.base).ok_or_else(|| SiteMismatch::UnknownSite { name: pair.base.clone() })?;
        let (expected_a, expected_b) = tensor_shapes(site, rank);
        if pair.a.len() != expected_a[0] * expected_a[1] || pair.b.len() != expected_b[0] * expected_b[1] {
            return Err(SiteMismatch::Shape { name: pair.base.clone(), rank: adapter.rank, expected_a, expected_b, got_a: pair.a.len(), got_b: pair.b.len() });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::TargetSpec;
    use crate::lora::device_adapter::{AdapterSite, SavedTensor};

    fn site(name: &str, out: usize, inn: usize) -> LinearSite {
        LinearSite { name: name.to_string(), leaf: "wq", layer: Some(0), spec: TargetSpec::whole(out, inn), save_name: None }
    }

    fn pair(site: &LinearSite, rank: usize) -> AdapterSite {
        let (a, b) = tensor_shapes(site, rank);
        AdapterSite { base: site.name.clone(), a: vec![0.0; a[0] * a[1]], b: vec![0.0; b[0] * b[1]] }
    }

    #[test]
    fn tensor_names_are_the_base_name_plus_the_factor() {
        let (a, b) = tensor_names(&site("blocks.0.attn.wq.weight", 4, 3));
        assert_eq!((a.as_str(), b.as_str()), ("blocks.0.attn.wq.weight.lora_a", "blocks.0.attn.wq.weight.lora_b"));
    }

    /// Distinct, non-square dimensions, so a transposed factor cannot fit.
    #[test]
    fn an_adapter_that_fits_its_sites_validates_and_may_skip_sites() {
        let sites = [site("w0", 6, 5), site("w1", 4, 7)];
        let adapter = DeviceAdapter { rank: 2, alpha: 2.0, sites: vec![pair(&sites[1], 2)], saved: Vec::new(), experts: Vec::new(), leaves: Default::default() };
        assert_eq!(validate(&adapter, &sites), Ok(()));
    }

    #[test]
    fn an_unknown_site_is_named() {
        let sites = [site("w0", 6, 5)];
        let adapter = DeviceAdapter { rank: 2, alpha: 2.0, sites: vec![AdapterSite { base: "nope".into(), a: vec![0.0; 10], b: vec![0.0; 12] }], saved: Vec::new(), experts: Vec::new(), leaves: Default::default() };
        assert_eq!(validate(&adapter, &sites), Err(SiteMismatch::UnknownSite { name: "nope".into() }));
    }

    #[test]
    fn full_tensors_are_refused_unless_the_model_applies_them_at_their_size() {
        let sites = [site("w0", 6, 5)];
        let gain = SavedSite { name: "norm.weight".into(), numel: 6, layer: Some(0) };
        let mut adapter = DeviceAdapter { rank: 2, alpha: 2.0, sites: vec![pair(&sites[0], 2)], saved: vec![SavedTensor { base: "norm.weight".into(), data: vec![1.0; 6] }], experts: Vec::new(), leaves: Default::default() };
        // A model that knows nothing of them never drops them silently.
        let err = validate(&adapter, &sites).unwrap_err();
        assert_eq!(err, SiteMismatch::UnsavableTensor { name: "norm.weight".into() });
        assert!(err.to_string().contains("modules_to_save"), "{err}");
        assert_eq!(validate_with_saved(&adapter, &sites, std::slice::from_ref(&gain)), Ok(()));
        adapter.saved[0].data.push(0.0);
        assert_eq!(validate_with_saved(&adapter, &sites, &[gain]), Err(SiteMismatch::SavedShape { name: "norm.weight".into(), expected: 6, got: 7 }));
    }

    #[test]
    fn a_wrong_rank_or_transposed_factor_is_refused_with_the_expected_shapes() {
        let sites = [site("w0", 6, 5)];
        let mut adapter = DeviceAdapter { rank: 2, alpha: 2.0, sites: vec![pair(&sites[0], 3)], saved: Vec::new(), experts: Vec::new(), leaves: Default::default() };
        let err = validate(&adapter, &sites).unwrap_err();
        assert!(matches!(err, SiteMismatch::Shape { rank: 2, expected_a: [2, 5], expected_b: [6, 2], .. }), "{err:?}");
        assert!(err.to_string().contains("w0") && err.to_string().contains("[2, 5]"), "{err}");
        // Same element counts as rank 2 would need only if A and B were swapped.
        adapter.sites[0] = AdapterSite { base: "w0".into(), a: vec![0.0; 12], b: vec![0.0; 10] };
        assert!(validate(&adapter, &sites).is_err());
    }
}
