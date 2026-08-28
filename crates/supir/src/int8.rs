// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! INT8 STORAGE format for the combined trunk + adaptors + frozen-backbone
//! checkpoint - the actual fix for a real, measured OOM, not an
//! optimisation.
//!
//! `crates/supir/tests/parity.rs`'s own full-forward test measured the
//! problem directly: the merged host import (frozen SDXL backbone, ~10.27
//! GB fp32, plus the SUPIR delta - trunk + adaptors - ~5.33 GB fp32) is
//! 15.6 GB resident, and `Supir::new`'s device-side upload while that map is
//! STILL live climbs steadily past 29 GB on this box's 30 GB and gets
//! SIGKILL'd by the OOM killer before the graph finishes recording. This
//! crate's own roadmap ledger names int8 a PREREQUISITE for running SUPIR at
//! all on hardware like this (one Intel iGPU + one NPU sharing 30 GB system
//! RAM, no discrete GPU), not a speed optimisation.
//!
//! This module is `sdxlunet::int8` extended with SUPIR's own policy, exactly
//! the shape `ltxv::int8::is_never_quantized`/`checkpoint::quantize::Policy`
//! establish: the quantizer itself (`model::int8::quantize_weight`) is the
//! one shared implementation; only the "which named tensors never get
//! touched" list is per-architecture.
//!
//! Swedish Embedded AB implements memory-constrained inference pipelines for
//! large diffusion checkpoints on edge hardware. If your team needs
//! expertise in fitting multi-billion-parameter models onto boxes with no
//! discrete GPU then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! ## What SUPIR adds to the never-quantize list, and why
//!
//! The whole `project_modules.*` prefix - every one of the 12
//! `ZeroSFT`/`ZeroCrossAttn` adaptors - is excluded outright:
//!
//! * `zero_conv`/`zero_mul`/`zero_add` (`ZeroSFT`'s three zero-init convs)
//!   are rank 4 and are therefore never even CANDIDATES under
//!   [`sdxlunet::int8::is_eligible`]'s structural rank-2 rule - but they are
//!   named here anyway, matching this port's roadmap ledger word for word
//!   ("these are meant to be exactly zero or near-zero at reasonable
//!   `control_scale` values and are numerically sensitive"): a future
//!   change to the eligibility rule (a grouped conv int8 tier, say) must not
//!   silently start touching them just because the rank-2 guard moved.
//! * `ZeroCrossAttn`'s `attn.to_q`/`attn.kv`/`attn.to_out.0` (the only
//!   genuinely rank-2, structurally-eligible tensors this module's 54.8 M
//!   adaptor params contain, at exactly 2 sites) ARE excluded, even though
//!   they are ordinary attention projections and not zero-init. The
//!   adaptors are under 1% of the combined checkpoint's bytes (54.8 M of
//!   1 332 M SUPIR-delta params, dwarfed further by the 2 567 M-param frozen
//!   backbone), so quantizing them buys negligible memory and this is the
//!   one control path where SUPIR's own restoration strength is tuned
//!   (`control_scale`) - not worth the risk for the bytes it saves.
//!
//! Everything else - the trunk's own `time_embed`/`label_emb` (matched by
//! [`sdxlunet::int8::is_never_quantized`]'s `time_embedding`/`add_embedding`
//! substrings, since [`crate::import::remap_trunk`] renames them onto those
//! exact diffusers-style names) and the frozen backbone's own conditioning
//! chain - inherits `sdxlunet::int8`'s policy unchanged, because both the
//! trunk and the backbone are SDXL-UNet-shaped (this port's own roadmap
//! ledger: "both are SDXL-UNet-shaped, both dominate the 15.6 GB host
//! footprint").

use std::collections::HashMap;

use vae::blocks::{PackedTensors, PackedWeight, Tensors};

/// SUPIR's own never-quantize predicate: `sdxlunet::int8`'s (the trunk's and
/// the frozen backbone's conditioning chains - both SDXL-UNet-shaped) plus
/// every `project_modules.*` adaptor tensor. See the module doc for why the
/// adaptors are excluded wholesale rather than name-by-name.
pub fn is_never_quantized(tensor_name: &str) -> bool {
    sdxlunet::int8::is_never_quantized(tensor_name) || tensor_name.contains("project_modules")
}

fn is_eligible(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2 && shape[1].is_multiple_of(model::int8::GROUP) && !is_never_quantized(name)
}

/// The SUPIR-scoped sibling of `sdxlunet::int8::QuantizedTensors`: same
/// split, over the MERGED tensor map (frozen backbone + trunk + adaptors,
/// however the caller assembled it - `crate::import::remap`'s output
/// extended with `sdxlunet::import::load`/`load_ldm`'s, exactly what
/// `crates/supir/tests/parity.rs` already builds for `Supir::new`).
pub struct QuantizedTensors {
    pub full: Tensors,
    pub packed: PackedTensors,
}

/// Split `w` into its int8-eligible weights (quantized via
/// `model::int8::quantize_weight`) and everything else, copied as-is into
/// `full`. Bounded to the input's own size plus the packed output - this
/// function does not touch a device, so it does not by itself close the OOM
/// (see [`crate::model::Supir::new_quantized`] for the half that does: the
/// device-side upload consuming `packed` never expands more than one tensor
/// to fp32 at a time).
pub fn quantize_tensors(w: &Tensors) -> QuantizedTensors {
    let mut full = Tensors::new();
    let mut packed: PackedTensors = HashMap::new();
    for (name, (shape, data)) in w {
        if is_eligible(name, shape) {
            let (n, k) = (shape[0], shape[1]);
            let (p, scale) = model::int8::quantize_weight(data, n, k);
            packed.insert(name.clone(), PackedWeight { shape: shape.clone(), packed: p, scale });
        } else {
            full.insert(name.clone(), (shape.clone(), data.clone()));
        }
    }
    QuantizedTensors { full, packed }
}

/// The inverse of [`quantize_tensors`] - see `sdxlunet::int8::dequantize_
/// tensors`'s doc for why this is a test/debugging convenience, not on the
/// real memory-saving path.
pub fn dequantize_tensors(q: &QuantizedTensors) -> Tensors {
    let mut out = q.full.clone();
    for (name, pw) in &q.packed {
        let (n, k) = (pw.shape[0], pw.shape[1]);
        let data = model::int8::dequantize_weight(&pw.packed, &pw.scale, n, k);
        out.insert(name.clone(), (pw.shape.clone(), data));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SupirConfig;

    #[test]
    fn project_modules_are_excluded_wholesale() {
        for name in [
            "project_modules.0.zero_conv.weight",
            "project_modules.7.zero_mul.weight",
            "project_modules.11.zero_add.bias",
            "project_modules.7.attn.to_q.weight",
            "project_modules.3.attn.kv.weight",
            "project_modules.3.attn.to_out.0.weight",
        ] {
            assert!(is_never_quantized(name), "{name} should be on the never-quantize list");
            assert!(!is_eligible(name, &[64, 64]), "{name} must not be int8-eligible even at a valid [n,k] shape");
        }
    }

    /// `sdxlunet::int8`'s own conditioning-chain exclusion carries through
    /// unchanged for the TRUNK's renamed names (`crate::import::remap_trunk`
    /// renames `time_embed`/`label_emb` onto exactly these).
    #[test]
    fn the_trunks_conditioning_chain_inherits_sdxlunets_exclusion() {
        for name in ["time_embedding.linear_1.weight", "add_embedding.linear_2.bias"] {
            assert!(is_never_quantized(name));
        }
    }

    #[test]
    fn ordinary_backbone_and_trunk_projections_stay_eligible() {
        for name in [
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.qkv.weight",
            "control_model.down_blocks.1.attentions.0.transformer_blocks.0.attn1.qkv.weight",
        ] {
            assert!(!is_never_quantized(name), "{name} must stay int8-eligible");
            assert!(is_eligible(name, &[64, 64]));
        }
    }

    fn tiny_merged_tensors(cfg: &SupirConfig) -> Tensors {
        let mut t = sdxlunet::init::init_weights(&cfg.backbone, 11);
        t.extend(crate::init::init_weights(cfg, 13));
        t
    }

    #[test]
    fn quantize_then_dequantize_preserves_every_eligible_tensor_in_the_merged_map() {
        let cfg = SupirConfig::tiny();
        let w = tiny_merged_tensors(&cfg);
        let q = quantize_tensors(&w);
        assert!(!q.packed.is_empty(), "tiny config must have at least one int8-eligible tensor");
        assert!(
            q.packed.keys().all(|n| !n.contains("project_modules")),
            "no project_modules tensor should ever be packed: {:?}",
            q.packed.keys().filter(|n| n.contains("project_modules")).collect::<Vec<_>>()
        );

        for (name, (shape, data)) in &w {
            if !q.packed.contains_key(name) {
                let (fshape, fdata) = q.full.get(name).unwrap_or_else(|| panic!("'{name}' missing from quantize_tensors output"));
                assert_eq!(fshape, shape);
                assert_eq!(fdata, data, "never-quantized tensor '{name}' must be untouched");
            }
        }

        let deq = dequantize_tensors(&q);
        let mut worst = (1.0f64, String::new());
        for name in q.packed.keys() {
            let (_, orig) = w.get(name).expect("original tensor");
            let (_, got) = deq.get(name).expect("dequantized tensor");
            let c = cosine(orig, got);
            assert!(c >= 0.999, "'{name}': per-tensor round-trip cosine {c:.6} too low");
            if c < worst.0 {
                worst = (c, name.clone());
            }
        }
        println!("supir int8 storage: {} tensors round-tripped, worst per-tensor cosine {:.9} ({})", q.packed.len(), worst.0, worst.1);
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            d += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
        }
        let den = na.sqrt() * nb.sqrt();
        if den <= 0.0 {
            0.0
        } else {
            d / den
        }
    }
}
