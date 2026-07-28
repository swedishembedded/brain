// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Map a `moondream/moondream3-preview` checkpoint's 662 tensors onto brain's
//! layout. Three subsystems: `model.vision.*` (SigLIP ViT + `proj_mlp` connector),
//! `model.text.*` (the parallel-block MoE decoder), and `model.region.*` (the
//! region/point/detect heads — deferred, recognized so coverage is exhaustive).
//!
//! The one non-trivial transform is the MoE experts: layers 4–23 store all experts
//! stacked in `mlp.fc1.weight [E, 2·inner, d]` and `mlp.fc2.weight [E, d, inner]`.
//! Per expert, `fc1` splits along its `2·inner` rows into `w_h` (first `inner`,
//! erf-GELU'd) and `w_g` (next `inner`, the `+1` shift) — matching `layers.py`'s
//! `x1, g = x1_full.chunk(2); F.gelu(x1) * (g + 1)` — and `fc2` is `w_down` as-is.

use crate::config::MoondreamConfig;

/// A stacked MoE tensor that splits per-expert at load time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MoePart {
    /// `[E, 2·inner, d]` → per expert `w_h [inner,d]` (0..inner) + `w_g [inner,d]` (inner..2·inner).
    Fc1,
    /// `[E, d, inner]` → per expert `w_down [d, inner]`.
    Fc2,
    /// `[E, d]` router gate.
    Router,
    /// `[E]` router bias (recognized; consumed once the router uses a bias).
    RouterBias,
}

/// Where a text tensor lands in brain's layout.
#[derive(Clone, Debug, PartialEq)]
pub enum TextTarget {
    /// Direct rename to this brain parameter key.
    Key(String),
    /// A stacked MoE tensor for decoder layer `layer`, split per-expert at load.
    Moe { layer: u32, part: MoePart },
}

/// HF `model.text.*` name → brain target (config decides dense vs MoE per layer).
pub fn map_text(hf: &str, cfg: &MoondreamConfig) -> Option<TextTarget> {
    use TextTarget::Key;
    match hf {
        "model.text.wte" => return Some(Key("tok.weight".into())),
        "model.text.lm_head.weight" => return Some(Key("lm_head.weight".into())),
        "model.text.lm_head.bias" => return Some(Key("lm_head.bias".into())),
        "model.text.post_ln.weight" => return Some(Key("post_ln.weight".into())),
        "model.text.post_ln.bias" => return Some(Key("post_ln.bias".into())),
        _ => {}
    }
    let (n, leaf) = hf.strip_prefix("model.text.blocks.")?.split_once('.')?;
    let layer: u32 = n.parse().ok()?;
    let moe = cfg.is_moe_layer(layer);
    let key = |k: &str| Some(Key(format!("blocks.{layer}.{k}")));
    match leaf {
        "ln.weight" | "ln.bias" | "attn.proj.weight" | "attn.proj.bias" | "attn.qkv.weight" | "attn.qkv.bias"
        | "attn.tau.alpha" | "attn.tau.wq" | "attn.tau.wv" => key(leaf),
        "mlp.fc1.bias" | "mlp.fc2.bias" if !moe => key(leaf), // dense layers only
        "mlp.fc1.weight" if !moe => key(leaf),
        "mlp.fc2.weight" if !moe => key(leaf),
        "mlp.fc1.weight" if moe => Some(TextTarget::Moe { layer, part: MoePart::Fc1 }),
        "mlp.fc2.weight" if moe => Some(TextTarget::Moe { layer, part: MoePart::Fc2 }),
        "mlp.router.weight" if moe => Some(TextTarget::Moe { layer, part: MoePart::Router }),
        "mlp.router.bias" if moe => Some(TextTarget::Moe { layer, part: MoePart::RouterBias }),
        _ => None,
    }
}

/// HF `model.vision.*` ViT tensor → [`SiglipEncoder`] key (prefix stripped). Returns
/// `None` for `proj_mlp.*` (the connector, see [`map_connector`]).
///
/// [`SiglipEncoder`]: crate::vision::SiglipEncoder
pub fn map_vision(hf: &str) -> Option<String> {
    let rest = hf.strip_prefix("model.vision.")?;
    if rest.starts_with("proj_mlp.") {
        return None;
    }
    // patch_emb.{weight,bias}, pos_emb, post_ln.{weight,bias}, blocks.N.<leaf> — all
    // already match SiglipEncoder's key scheme verbatim.
    Some(rest.to_string())
}

/// HF `model.vision.proj_mlp.*` → [`Connector`] key.
///
/// [`Connector`]: crate::vision::Connector
pub fn map_connector(hf: &str) -> Option<String> {
    let rest = hf.strip_prefix("model.vision.proj_mlp.")?;
    Some(match rest {
        "fc1.weight" => "fc1.weight",
        "fc1.bias" => "fc1.bias",
        "fc2.weight" => "fc2.weight",
        "fc2.bias" => "fc2.bias",
        _ => return None,
    }
    .to_string())
}

/// The region/point/detect heads (`model.region.*`) — deferred (Phase 3.9).
pub fn is_region(hf: &str) -> bool {
    hf.starts_with("model.region.")
}

/// Split a stacked MoE `fc1.weight[expert]` slice `[2·inner, d]` into `(w_h, w_g)`,
/// each `[inner, d]`. `w_h` is the erf-GELU'd half, `w_g` the `+1`-shifted half.
pub fn split_fc1_expert(slice: &[f32], inner: u32, d: u32) -> (Vec<f32>, Vec<f32>) {
    let half = (inner * d) as usize;
    assert_eq!(slice.len(), 2 * half, "fc1 expert slice must be [2·inner, d]");
    (slice[..half].to_vec(), slice[half..].to_vec())
}

/// The brain keys produced for one MoE decoder layer (router + per-expert triples).
pub fn moe_layer_keys(layer: u32, num_experts: u32) -> Vec<String> {
    let mut keys = vec![format!("blocks.{layer}.moe.router.weight")];
    for e in 0..num_experts {
        for leaf in ["w_h.weight", "w_g.weight", "w_down.weight"] {
            keys.push(format!("blocks.{layer}.moe.experts.{e}.{leaf}"));
        }
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MoondreamConfig {
        MoondreamConfig::preview()
    }

    #[test]
    fn text_names_route_dense_vs_moe() {
        let c = cfg();
        assert_eq!(map_text("model.text.wte", &c), Some(TextTarget::Key("tok.weight".into())));
        // Dense layer 0: fc1.weight is a plain rename with a bias.
        assert_eq!(map_text("model.text.blocks.0.mlp.fc1.weight", &c), Some(TextTarget::Key("blocks.0.mlp.fc1.weight".into())));
        assert_eq!(map_text("model.text.blocks.0.mlp.fc1.bias", &c), Some(TextTarget::Key("blocks.0.mlp.fc1.bias".into())));
        // MoE layer 4: fc1/fc2/router are stacked splits; no dense bias.
        assert_eq!(map_text("model.text.blocks.4.mlp.fc1.weight", &c), Some(TextTarget::Moe { layer: 4, part: MoePart::Fc1 }));
        assert_eq!(map_text("model.text.blocks.4.mlp.router.weight", &c), Some(TextTarget::Moe { layer: 4, part: MoePart::Router }));
        assert_eq!(map_text("model.text.blocks.23.mlp.fc2.weight", &c), Some(TextTarget::Moe { layer: 23, part: MoePart::Fc2 }));
        // tau + attn always direct.
        assert_eq!(map_text("model.text.blocks.7.attn.tau.alpha", &c), Some(TextTarget::Key("blocks.7.attn.tau.alpha".into())));
    }

    #[test]
    fn vision_and_connector_split() {
        assert_eq!(map_vision("model.vision.blocks.3.attn.qkv.weight"), Some("blocks.3.attn.qkv.weight".into()));
        assert_eq!(map_vision("model.vision.patch_emb.weight"), Some("patch_emb.weight".into()));
        assert_eq!(map_vision("model.vision.proj_mlp.fc1.weight"), None); // connector, not ViT
        assert_eq!(map_connector("model.vision.proj_mlp.fc2.bias"), Some("fc2.bias".into()));
    }

    #[test]
    fn fc1_split_halves() {
        // [2·inner=4, d=2]: rows 0..2 → w_h, rows 2..4 → w_g.
        let slice: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let (h, g) = split_fc1_expert(&slice, 2, 2);
        assert_eq!(h, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(g, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn real_checkpoint_fully_covered() {
        use std::io::Read;
        let path = "/data/workspace/resources/vl/moondream3/hf/moondream3-preview/model.safetensors.index.json";
        let Ok(mut f) = std::fs::File::open(path) else {
            eprintln!("skip: moondream3 index not present");
            return;
        };
        let mut s = String::new();
        f.read_to_string(&mut s).unwrap();
        let idx: serde_json::Value = serde_json::from_str(&s).unwrap();
        let names: Vec<String> = idx["weight_map"].as_object().unwrap().keys().cloned().collect();
        let c = cfg();

        // Every source tensor is classified into exactly one subsystem.
        let mut text = 0usize;
        let mut vision = 0usize;
        let mut connector = 0usize;
        let mut region = 0usize;
        for n in &names {
            let hits = map_text(n, &c).is_some() as u8
                + map_vision(n).is_some() as u8
                + map_connector(n).is_some() as u8
                + is_region(n) as u8;
            assert_eq!(hits, 1, "tensor classified {hits}× (want 1): {n}");
            if map_text(n, &c).is_some() {
                text += 1;
            } else if map_vision(n).is_some() {
                vision += 1;
            } else if map_connector(n).is_some() {
                connector += 1;
            } else {
                region += 1;
            }
        }
        assert_eq!(names.len(), 662, "expected 662 tensors");
        assert_eq!(connector, 4, "proj_mlp fc1/fc2 weight+bias");
        assert_eq!(region, 12, "region head tensors (deferred)");
        // Vision: 27 blocks × 12 + patch_emb(2) + pos_emb(1) + post_ln(2) = 329.
        assert_eq!(vision, 27 * 12 + 5);
        assert_eq!(text + vision + connector + region, 662);
        assert!(text > 0);
    }
}
