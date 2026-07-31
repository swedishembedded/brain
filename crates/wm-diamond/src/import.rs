// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import a reference DIAMOND checkpoint (torch `.pt`, e.g. HF
//! `eloialonso/diamond` `atari_100k/models/<Game>.pt`) into a brain
//! `.safetensors` container, and load `.safetensors` back into the host tensor map.
//!
//! Names are the reference names with the `denoiser.inner_model.` prefix
//! stripped. FULL-COVERAGE discipline (like `glm::import`): every expected
//! tensor must be present with the right shape, and unexpected denoiser
//! tensors are a hard error — layers are never silently skipped. The
//! `rew_end_model.*` / `actor_critic.*` sub-models are not imported (not
//! needed to play).

use crate::config::DiamondConfig;
use crate::model::Tensors;
use std::collections::HashMap;

const PREFIX: &str = "denoiser.inner_model.";

/// Import `src` (.pt) to `out` (.safetensors). `num_actions` must match the game
/// (Breakout: 4); it is validated against the embedding shape.
pub fn import(src: &str, out: &str, num_actions: u32) -> Result<DiamondConfig, String> {
    let report = checkpoint::torchpt::read_report(src)?;
    let mut found: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for t in report.tensors {
        if let Some(stripped) = t.name.strip_prefix(PREFIX) {
            found.insert(stripped.to_string(), (t.shape, t.data));
        }
    }
    if found.is_empty() {
        return Err(format!("{src}: no `{PREFIX}*` tensors — not a DIAMOND agent checkpoint?"));
    }

    let cfg = DiamondConfig::atari(num_actions);
    let expected = cfg.param_list();

    // Validate the action embedding actually matches num_actions.
    if let Some((shape, _)) = found.get("act_emb.0.weight") {
        if shape.first() != Some(&(num_actions as usize)) {
            return Err(format!(
                "act_emb.0.weight is {shape:?} but --actions {num_actions} was given \
                 (wrong game action count)"
            ));
        }
    }

    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::with_capacity(expected.len());
    let mut missing = vec![];
    for (name, shape) in &expected {
        match found.remove(name) {
            Some((got_shape, data)) => {
                if &got_shape != shape {
                    return Err(format!(
                        "{name}: shape mismatch — checkpoint {got_shape:?}, expected {shape:?}"
                    ));
                }
                let numel: usize = shape.iter().product();
                if data.len() != numel {
                    return Err(format!(
                        "{name}: {} values for shape {shape:?} ({numel})",
                        data.len()
                    ));
                }
                tensors.push((name.clone(), shape.iter().map(|&d| d as u64).collect(), data));
            }
            None => missing.push(name.clone()),
        }
    }
    if !missing.is_empty() {
        return Err(format!("missing {} tensors, e.g. {:?}", missing.len(), &missing[..missing.len().min(5)]));
    }
    if !found.is_empty() {
        let extra: Vec<&String> = found.keys().take(5).collect();
        return Err(format!(
            "{} unexpected denoiser tensors (never silently skipped), e.g. {extra:?}",
            found.len()
        ));
    }

    let config: serde_json::Value = serde_json::from_str(&cfg.to_json())
        .map_err(|e| format!("config json: {e}"))?;
    checkpoint::save(out, config, &tensors);
    Ok(cfg)
}

/// Load a brain `.safetensors` DIAMOND checkpoint into (config, host tensors),
/// re-validating full coverage and sizes.
pub fn load(path: &str) -> Result<(DiamondConfig, Tensors), String> {
    let c = checkpoint::load(path);
    let cfg = DiamondConfig::from_json(&c.header["config"].to_string())?;
    let expected = cfg.param_list();
    let mut by_name: HashMap<String, Vec<f32>> =
        c.tensors.into_iter().map(|t| (t.name, t.data)).collect();
    let mut out: Tensors = HashMap::new();
    for (name, shape) in expected {
        let data = by_name
            .remove(&name)
            .ok_or_else(|| format!("{path}: missing tensor {name}"))?;
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(format!("{path}: {name} has {} values, expected {numel}", data.len()));
        }
        out.insert(name, (shape, data));
    }
    if !by_name.is_empty() {
        let extra: Vec<&String> = by_name.keys().take(5).collect();
        return Err(format!("{path}: {} unexpected tensors, e.g. {extra:?}", by_name.len()));
    }
    Ok((cfg, out))
}
