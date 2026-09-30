// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where `MultiModalityCausalLM` checkpoints keep each component, and the
//! aligner's name map. The towers' and the decoder's own importers read their
//! prefixes; this module owns the aligner and the whole-checkpoint coverage
//! check, so a tensor no component reads is an error rather than silently
//! left behind.

use std::collections::HashMap;

use checkpoint::weightio::WeightReader;
use model::projector::{ProjectorConfig, ProjectorKind};

/// DeepSeek-VL's SAM tower (`sam_b_downsample`).
pub const SAM_PREFIX: &str = "vision_model.vision_tower_high.vision_tower.";
/// The decoder: `LlamaForCausalLM` nested under `language_model`.
pub const DECODER: qwen3::hf::HfNames = qwen3::hf::HfNames { prefix: "language_model.model.", head: "language_model.lm_head.weight" };
/// The image aligner.
pub const ALIGNER_PREFIX: &str = "aligner.";

/// Built by the hybrid tower's constructor but never applied in its forward:
/// `high_layer_norm` and `low_layer_norm`.
pub const UNUSED: [&str; 4] = [
    "vision_model.high_layer_norm.weight",
    "vision_model.high_layer_norm.bias",
    "vision_model.low_layer_norm.weight",
    "vision_model.low_layer_norm.bias",
];

/// The brain projector parameter an upstream `MlpProjector` leaf (prefix
/// stripped) is. The reference builds its GELU stack as one `nn.Sequential`,
/// so the linears after each GELU sit at odd indices (hybrid split, whose
/// input linears are separate modules) or at even indices from 2 (plain
/// `mlp_gelu`, whose input linear is index 0).
fn aligner_leaf(leaf: &str, kind: ProjectorKind) -> Option<String> {
    let (module, param) = leaf.rsplit_once('.')?;
    if param != "weight" && param != "bias" {
        return None;
    }
    let brain = match (kind, module) {
        (ProjectorKind::HybridSplit, "high_up_proj") => "in_high".to_string(),
        (ProjectorKind::HybridSplit, "low_up_proj") => "in_low".to_string(),
        (ProjectorKind::Mlp, "layers.0") => "in".to_string(),
        (_, m) => {
            let i: u32 = m.strip_prefix("layers.")?.parse().ok()?;
            let k = match kind {
                ProjectorKind::HybridSplit if i % 2 == 1 => (i + 1) / 2,
                ProjectorKind::Mlp if i % 2 == 0 && i > 0 => i / 2,
                _ => return None,
            };
            format!("layers.{k}")
        }
    };
    Some(format!("{brain}.{param}"))
}

/// The aligner under `prefix` (`aligner.`, or Janus-Pro's `gen_aligner.`),
/// by brain name, with two-way coverage against `cfg`.
pub fn aligner_weights(rd: &WeightReader, prefix: &str, cfg: &ProjectorConfig) -> Result<HashMap<String, Vec<f32>>, String> {
    let want: HashMap<String, usize> = cfg.param_list().into_iter().collect();
    let mut out = HashMap::new();
    for name in rd.names().filter(|n| n.starts_with(prefix)) {
        let brain = aligner_leaf(&name[prefix.len()..], cfg.kind).ok_or_else(|| format!("aligner import: {name} is not a {:?} projector parameter", cfg.kind))?;
        let &n = want.get(&brain).ok_or_else(|| format!("aligner import: {name} maps to {brain}, which a depth-{} projector does not have", cfg.depth))?;
        let data = rd.tensor(name).ok_or_else(|| format!("aligner import: {name} is indexed but unreadable"))?;
        if data.len() != n {
            return Err(format!("aligner import: {name} has {} values, {brain} takes {n}", data.len()));
        }
        out.insert(brain, data);
    }
    if let Some(missing) = want.keys().find(|k| !out.contains_key(*k)) {
        return Err(format!("aligner import: nothing under '{prefix}' provides {missing}"));
    }
    Ok(out)
}

/// Every tensor in the checkpoint belongs to a component that reads it (or is
/// one of [`UNUSED`] or the SigLIP tower's unused head): the check that a
/// component the composite does not know about cannot hide in the file.
pub fn check_coverage(rd: &WeightReader, prefixes: &[&str]) -> Result<(), String> {
    match rd.names().find(|n| !UNUSED.contains(n) && !prefixes.iter().any(|p| n.starts_with(p))) {
        Some(stray) => Err(format!("{stray} belongs to no component of the model")),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hybrid_aligner_maps_both_input_projections_and_the_gelu_stack() {
        let h = ProjectorKind::HybridSplit;
        assert_eq!(aligner_leaf("high_up_proj.weight", h).as_deref(), Some("in_high.weight"));
        assert_eq!(aligner_leaf("low_up_proj.bias", h).as_deref(), Some("in_low.bias"));
        assert_eq!(aligner_leaf("layers.1.weight", h).as_deref(), Some("layers.1.weight"));
        assert_eq!(aligner_leaf("layers.0.weight", h), None, "index 0 is the GELU");
    }

    #[test]
    fn the_plain_aligner_counts_linears_past_the_gelus() {
        let m = ProjectorKind::Mlp;
        assert_eq!(aligner_leaf("layers.0.bias", m).as_deref(), Some("in.bias"));
        assert_eq!(aligner_leaf("layers.2.weight", m).as_deref(), Some("layers.1.weight"));
        assert_eq!(aligner_leaf("layers.1.weight", m), None);
        assert_eq!(aligner_leaf("high_up_proj.weight", m), None);
    }
}
