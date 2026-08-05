// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import `ip-adapter.bin`'s two halves with two-way coverage validation.
//!
//! The released file is a torch archive holding `image_proj` (the Resampler) and
//! `ip_adapter` (the per-site decoupled projections). Both are validated the way
//! `crates/flux2` does it: every expected tensor produced exactly once with the
//! right shape, and no source tensor left unused — a mismatch is an error naming
//! the tensor, never a silent zero-fill, because a zero-filled weight passes
//! every shape check and destroys parity with no other symptom.

use std::collections::HashMap;

use crate::config::{ResamplerConfig, SiteConfig};

/// name -> fp32 data, keyed by the released names.
pub type Weights = HashMap<String, Vec<f32>>;

/// Validate `map` against `cfg`'s manifest, both directions.
pub fn validate_resampler(map: Weights, cfg: &ResamplerConfig) -> Result<Weights, String> {
    let manifest = cfg.tensor_manifest();
    for (name, shape) in &manifest {
        let n: usize = shape.iter().product();
        match map.get(name) {
            None => return Err(format!("instantid: missing tensor {name}")),
            Some(d) if d.len() != n => {
                return Err(format!("instantid: {name} has {} values, expected {n} for {shape:?}", d.len()))
            }
            Some(_) => {}
        }
    }
    if map.len() != manifest.len() {
        let expected: std::collections::HashSet<&str> = manifest.iter().map(|(n, _)| n.as_str()).collect();
        let extra: Vec<&String> = map.keys().filter(|k| !expected.contains(k.as_str())).collect();
        return Err(format!("instantid: unused source tensors: {extra:?}"));
    }
    Ok(map)
}

/// Every site's `to_k_ip` / `to_v_ip`, keyed by site index.
pub struct SiteWeights {
    pub cfg: Vec<SiteConfig>,
    pub k: HashMap<usize, Vec<f32>>,
    pub v: HashMap<usize, Vec<f32>>,
}

/// Validate the `ip_adapter` half: every declared site must have both
/// projections at the right element count, and nothing else may be present.
pub fn validate_sites(
    shapes: &HashMap<String, Vec<usize>>,
    data: HashMap<String, Vec<f32>>,
) -> Result<SiteWeights, String> {
    let cfg = SiteConfig::from_tensors(shapes)?;
    let (mut k, mut v) = (HashMap::new(), HashMap::new());
    for s in &cfg {
        for (suffix, dst) in [("to_k_ip", &mut k), ("to_v_ip", &mut v)] {
            let name = format!("{}.{suffix}.weight", s.index);
            let d = data.get(&name).ok_or_else(|| format!("instantid: missing {name}"))?;
            let want = s.hidden * s.token_dim;
            if d.len() != want {
                return Err(format!("instantid: {name} has {} values, expected {want}", d.len()));
            }
            dst.insert(s.index, d.clone());
        }
    }
    if data.len() != cfg.len() * 2 {
        return Err(format!("instantid: ip_adapter has {} tensors for {} sites (expected {})", data.len(), cfg.len(), cfg.len() * 2));
    }
    Ok(SiteWeights { cfg, k, v })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full() -> Weights {
        ResamplerConfig::released()
            .tensor_manifest()
            .into_iter()
            .map(|(n, s)| (n, vec![0.0f32; s.iter().product()]))
            .collect()
    }

    #[test]
    fn a_complete_manifest_validates() {
        assert!(validate_resampler(full(), &ResamplerConfig::released()).is_ok());
    }

    #[test]
    fn a_missing_tensor_is_named_not_zero_filled() {
        let mut w = full();
        w.remove("proj_out.weight");
        let e = validate_resampler(w, &ResamplerConfig::released()).unwrap_err();
        assert!(e.contains("proj_out.weight"), "got: {e}");
    }

    #[test]
    fn an_unused_source_tensor_is_an_error() {
        // An extra tensor means the checkpoint is not the one the config
        // describes — importing anyway would silently ignore real weights.
        let mut w = full();
        w.insert("layers.9.0.to_q.weight".into(), vec![0.0; 4]);
        assert!(validate_resampler(w, &ResamplerConfig::released()).is_err());
    }

    #[test]
    fn a_wrong_element_count_is_an_error() {
        let mut w = full();
        w.insert("proj_in.bias".into(), vec![0.0; 3]);
        let e = validate_resampler(w, &ResamplerConfig::released()).unwrap_err();
        assert!(e.contains("proj_in.bias"), "got: {e}");
    }
}
