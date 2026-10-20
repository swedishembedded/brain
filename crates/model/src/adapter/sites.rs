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
    /// A pair's factors are not the rank-`rank` shapes of the site.
    Shape { name: String, rank: u32, expected_a: [usize; 2], expected_b: [usize; 2], got_a: usize, got_b: usize },
}

impl fmt::Display for SiteMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SiteMismatch::UnknownSite { name } => write!(f, "{name} is not an adapter site of this model"),
            SiteMismatch::Shape { name, rank, expected_a, expected_b, got_a, got_b } => write!(
                f,
                "{name}: a rank-{rank} adapter needs A {expected_a:?} ({} values) and B {expected_b:?} ({} values) but holds {got_a} and {got_b}",
                expected_a[0] * expected_a[1],
                expected_b[0] * expected_b[1]
            ),
        }
    }
}

impl std::error::Error for SiteMismatch {}

/// Check that every pair of `adapter` belongs to one of `sites` and has the
/// shapes of that site at the adapter's rank. The adapter need not cover every
/// site (a rank-r adapter on the attention projections only is valid).
pub fn validate(adapter: &DeviceAdapter, sites: &[LinearSite]) -> Result<(), SiteMismatch> {
    let rank = adapter.rank as usize;
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
    use crate::lora::device_adapter::AdapterSite;

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
        let adapter = DeviceAdapter { rank: 2, alpha: 2.0, sites: vec![pair(&sites[1], 2)] };
        assert_eq!(validate(&adapter, &sites), Ok(()));
    }

    #[test]
    fn an_unknown_site_is_named() {
        let sites = [site("w0", 6, 5)];
        let adapter = DeviceAdapter { rank: 2, alpha: 2.0, sites: vec![AdapterSite { base: "nope".into(), a: vec![0.0; 10], b: vec![0.0; 12] }] };
        assert_eq!(validate(&adapter, &sites), Err(SiteMismatch::UnknownSite { name: "nope".into() }));
    }

    #[test]
    fn a_wrong_rank_or_transposed_factor_is_refused_with_the_expected_shapes() {
        let sites = [site("w0", 6, 5)];
        let mut adapter = DeviceAdapter { rank: 2, alpha: 2.0, sites: vec![pair(&sites[0], 3)] };
        let err = validate(&adapter, &sites).unwrap_err();
        assert!(matches!(err, SiteMismatch::Shape { rank: 2, expected_a: [2, 5], expected_b: [6, 2], .. }), "{err:?}");
        assert!(err.to_string().contains("w0") && err.to_string().contains("[2, 5]"), "{err}");
        // Same element counts as rank 2 would need only if A and B were swapped.
        adapter.sites[0] = AdapterSite { base: "w0".into(), a: vec![0.0; 12], b: vec![0.0; 10] };
        assert!(validate(&adapter, &sites).is_err());
    }
}
