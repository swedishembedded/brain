// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deterministic weight initialization for ZipDepth.
//!
//! Deterministic for a fixed `seed` — the FD gradient check depends on it.
//!
//! The param names are the REFERENCE's (`running_var`, not brain's `run_var`),
//! because `config.rs` mirrors the checkpoint's state_dict so import is a 1:1
//! name match. That difference is exactly why BatchNorm is classified
//! structurally here (by the presence of a sibling `running_var`) rather than by
//! matching a spelling or a module path — see [`init_params`].

use std::collections::HashMap;

use data::rng::Rng;

use crate::config::ZipConfig;

/// ImageNet statistics, as the reference bakes them into the state_dict
/// (`architecture.py:616-617`). They are constants, not learned, but they ship
/// with the weights because normalization happens INSIDE the model.
///
/// The arrays themselves come from `imaging` — they were declared byte-identically
/// here and in `worldmirror2::preprocess`. Only the arrays are shared: WHERE they are
/// applied is model-specific and deliberately not unified. ZipDepth folds them
/// into the first BatchNorm below, so its predictor feeds the model `[0,1]` and
/// never normalizes; `mirror` applies them per frame on the host.
use imaging::{IMAGENET_MEAN, IMAGENET_STD};

/// Initialize every parameter in `params` (`(name, numel)`), deterministic for
/// `seed`. `std` scales the conv-weight Gaussian.
///
/// BatchNorm is identified STRUCTURALLY, not by name pattern. The reference wraps
/// conv+BN in `nn.Sequential`, so BN's learnable tensors are named by POSITION
/// (`branch_3x3.1.weight`) as often as by module (`bn.weight`) — and a
/// `.1.weight` could equally be a conv. The unambiguous signal is that a BN group
/// carries running statistics and a conv never does: if `<prefix>.running_var`
/// exists, `<prefix>.weight` is BN's affine scale.
///
/// That matters more than it looks. Misclassifying BN's scale as a conv weight
/// initialises it to a Gaussian around 0 instead of 1 — the model still builds,
/// still runs, and trains to garbage. A hardcoded list of module paths would do
/// exactly that the first time a BN appears somewhere new.
pub fn init_params(params: &[(String, usize)], seed: u64, std: f32) -> HashMap<String, Vec<f32>> {
    let bn_prefixes: std::collections::HashSet<&str> = params
        .iter()
        .filter_map(|(n, _)| n.strip_suffix(".running_var"))
        .collect();
    let is_bn = |name: &str| -> bool {
        name.rsplit_once('.').is_some_and(|(pfx, _)| bn_prefixes.contains(pfx))
    };

    let mut rng = Rng::new(seed);
    let mut w = HashMap::new();
    for (name, numel) in params {
        let v: Vec<f32> = if name == "mean" {
            IMAGENET_MEAN.to_vec()
        } else if name == "std" {
            IMAGENET_STD.to_vec()
        } else if name.ends_with("running_mean") {
            vec![0.0; *numel]
        } else if name.ends_with("running_var") {
            vec![1.0; *numel]
        } else if is_bn(name) && name.ends_with(".weight") {
            // BN affine scale ~ 1, jittered so the FD check does not sit on a
            // degenerate point.
            (0..*numel).map(|_| 1.0 + 0.1 * rng.next_gaussian() as f32).collect()
        } else if is_bn(name) && name.ends_with(".bias") {
            (0..*numel).map(|_| 0.05 * rng.next_gaussian() as f32).collect()
        } else if name == "decoder.head_half.bias" {
            // The reference inits this head bias to 0.5 (`architecture.py:461`),
            // not 0 — the output is ReLU'd inverse depth, so starting at zero puts
            // every pixel on the flat side of the ReLU and the head receives no
            // gradient at all.
            vec![0.5; *numel]
        } else if name.ends_with(".bias") {
            vec![0.0; *numel]
        } else {
            (0..*numel).map(|_| std * rng.next_gaussian() as f32).collect()
        };
        w.insert(name.clone(), v);
    }
    w
}

/// Initialize the whole model from its config, deterministic for `seed`.
///
/// A small conv std keeps the deep stack well-conditioned for the FD check —
/// larger weights drive BN/ReLU into curved regions where the central difference
/// stops agreeing with the analytic directional derivative over a 56-conv cascade.
pub fn init_model(cfg: &ZipConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let params: Vec<(String, usize)> =
        cfg.param_list().into_iter().map(|(n, s)| (n, s.iter().product())).collect();
    init_params(&params, seed, 0.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_covers_every_param_exactly() {
        let cfg = ZipConfig::base();
        let w = init_model(&cfg, 7);
        let p = cfg.param_list();
        assert_eq!(w.len(), p.len(), "init produced a different number of tensors");
        for (name, shape) in &p {
            let numel: usize = shape.iter().product();
            let v = w.get(name).unwrap_or_else(|| panic!("init missing `{name}`"));
            assert_eq!(v.len(), numel, "`{name}` has the wrong length");
            assert!(v.iter().all(|x| x.is_finite()), "`{name}` has non-finite init");
        }
    }

    #[test]
    fn init_is_deterministic_for_a_seed() {
        let cfg = ZipConfig::base();
        let (a, b) = (init_model(&cfg, 7), init_model(&cfg, 7));
        for (k, va) in &a {
            assert_eq!(va, &b[k], "`{k}` differs between two inits at the same seed");
        }
        let c = init_model(&cfg, 8);
        assert!(a["encoder.stem_half.conv.weight"] != c["encoder.stem_half.conv.weight"], "seed is ignored");
    }

    /// BN must start as an identity-ish affine over normalized stats. This is the
    /// test that would catch a BN-classification bug: misclassifying BN's scale as
    /// a conv weight leaves it Gaussian around 0 instead of 1, and the model then
    /// builds, runs, and trains to garbage rather than failing.
    ///
    /// It counts the BN groups too — ZipDepth's base config has 43 (one per
    /// BatchNorm, matching the checkpoint's 43 num_batches_tracked counters) — so
    /// a predicate that silently classified NOTHING as BN would fire here.
    #[test]
    fn batchnorm_starts_as_a_near_identity_affine() {
        let cfg = ZipConfig::base();
        let w = init_model(&cfg, 7);
        for (name, v) in &w {
            if name.ends_with("running_mean") {
                assert!(v.iter().all(|x| *x == 0.0), "`{name}` should start at 0");
            } else if name.ends_with("running_var") {
                assert!(v.iter().all(|x| *x == 1.0), "`{name}` should start at 1");
            }
        }
        // Every BN's scale is ~1, not ~0: check via the running_var siblings.
        let bn_scales: Vec<&String> = w
            .keys()
            .filter(|k| k.ends_with(".weight") && w.contains_key(&k.replace(".weight", ".running_var")))
            .collect();
        assert_eq!(
            bn_scales.len(),
            43,
            "expected 43 BN affine scales (one per BatchNorm, matching the \
             checkpoint's 43 num_batches_tracked); the BN predicate is wrong"
        );
        for k in bn_scales {
            let mean = w[k].iter().sum::<f32>() / w[k].len() as f32;
            assert!(
                (mean - 1.0).abs() < 0.35,
                "`{k}` mean {mean} — BN scale must start near 1, not near 0"
            );
        }
    }

    #[test]
    fn normalization_buffers_are_the_imagenet_constants() {
        let w = init_model(&ZipConfig::base(), 7);
        assert_eq!(w["mean"], IMAGENET_MEAN.to_vec());
        assert_eq!(w["std"], IMAGENET_STD.to_vec());
    }

    /// The head bias starts at 0.5, not 0: the output is ReLU'd, so a zero start
    /// parks every pixel on the flat side and the head gets no gradient.
    #[test]
    fn head_bias_starts_off_the_relu_floor() {
        let w = init_model(&ZipConfig::base(), 7);
        assert_eq!(w["decoder.head_half.bias"], vec![0.5]);
    }
}
