// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import the mobileclip_l (FastViTHD-L) vision tower from an `apple/FastVLM-*`
//! checkpoint into brain's [`Encoder::mobileclip_l`] weight layout.
//!
//! The checkpoint ships the fused inference form under
//! `model.vision_tower.vision_tower.model.*` with a *flat* `network.{0..10}` index
//! (blocks, downsamples, and RepCPEs interleaved) and torch leaf names. brain uses a
//! structured naming (`network.{stage}.{b}`, `downsample.{i}`, `stage{i}.cpe`) with
//! `ConvNames::brain` (`.conv.weight`, `.bn.gamma/beta/run_mean/run_var`). This maps
//! between them, including the `layer_scale [C] → layer_scale_sb [2C] = [ls-1, 0]`
//! value transform used by brain's `film_chan` LayerScale.
//!
//! [`Encoder::mobileclip_l`]: crate::encoder::Encoder::mobileclip_l

use std::collections::HashMap;

const PREFIX: &str = "model.vision_tower.vision_tower.model.";

/// The flat checkpoint `network.{n}` index → brain block prefix. Even block-stages
/// 0/2/4/7/10 → `network.{0..4}`; downsamples 1/3/5/8 → `downsample.{0..3}`; RepCPEs
/// 6/9 → `stage{3,4}.cpe`. Returns `(brain_prefix, block_kind)`.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Rep,
    Attn,
    Down,
    Cpe,
    Conv,
}

fn stage_prefix(n: u32, rest: &str) -> Option<(String, Kind)> {
    // rest begins after "network.{n}." — for block stages it's "{b}.<leaf>".
    match n {
        0 | 2 | 4 => {
            let (b, _) = rest.split_once('.')?;
            Some((format!("network.{}.{b}", n / 2), Kind::Rep))
        }
        7 | 10 => {
            let (b, _) = rest.split_once('.')?;
            let stage = if n == 7 { 3 } else { 4 };
            Some((format!("network.{stage}.{b}"), Kind::Attn))
        }
        1 | 3 | 5 => Some((format!("downsample.{}", (n - 1) / 2), Kind::Down)),
        8 => Some(("downsample.3".to_string(), Kind::Down)),
        6 => Some(("stage3.cpe".to_string(), Kind::Cpe)),
        9 => Some(("stage4.cpe".to_string(), Kind::Cpe)),
        _ => None,
    }
}

/// A ConvFFN leaf (`conv.conv.weight`, `conv.bn.*`, `fc1.*`, `fc2.*`) → brain leaf.
fn convffn_leaf(leaf: &str) -> Option<String> {
    if let Some(rest) = leaf.strip_prefix("conv.") {
        // depthwise: brain `convffn.dw` (ConvNames::brain).
        return Some(format!("convffn.dw.{}", conv_bn_leaf(rest)?));
    }
    for fc in ["fc1", "fc2"] {
        if let Some(rest) = leaf.strip_prefix(&format!("{fc}.")) {
            return Some(match rest {
                "weight" => format!("convffn.{fc}.conv.weight"),
                "bias" => format!("convffn.{fc}.bias"),
                _ => return None,
            });
        }
    }
    None
}

/// A torch conv+BN leaf (`conv.weight` / `bn.{weight,bias,running_mean,running_var}`)
/// → brain (`conv.weight` / `bn.{gamma,beta,run_mean,run_var}`). Drops the counter.
fn conv_bn_leaf(leaf: &str) -> Option<String> {
    Some(match leaf {
        "conv.weight" => "conv.weight".into(),
        "bn.weight" => "bn.gamma".into(),
        "bn.bias" => "bn.beta".into(),
        "bn.running_mean" => "bn.run_mean".into(),
        "bn.running_var" => "bn.run_var".into(),
        _ => return None, // num_batches_tracked etc. dropped
    })
}

/// Map one checkpoint tensor (name relative to `PREFIX`) to `(brain_name, transform)`
/// where transform is applied to the tensor data. `None` = drop.
enum Out {
    Rename(String),
    LayerScale(String),
}

fn map_leaf(prefix: &str, kind: Kind, leaf: &str) -> Option<Out> {
    let key = |s: String| Some(Out::Rename(format!("{prefix}.{s}")));
    match kind {
        Kind::Conv | Kind::Cpe => match leaf {
            "reparam_conv.weight" => key("conv.weight".into()),
            "reparam_conv.bias" => key("bias".into()),
            // conv_exp SE (1×1 conv weights are flat-identical to brain's matmuls).
            "se.reduce.weight" | "se.reduce.bias" | "se.expand.weight" | "se.expand.bias" => key(leaf.into()),
            _ => None,
        },
        Kind::Down => {
            // proj.0 = large-kernel rlk (lkb_reparam); proj.1 = 1×1 (reparam_conv).
            if let Some(l) = leaf.strip_prefix("proj.0.lkb_reparam.") {
                return key(if l == "weight" { "rlk.conv.weight".into() } else { "rlk.bias".into() });
            }
            if let Some(l) = leaf.strip_prefix("proj.1.reparam_conv.") {
                return key(if l == "weight" { "proj.conv.weight".into() } else { "proj.bias".into() });
            }
            None
        }
        Kind::Rep => {
            if let Some(l) = leaf.strip_prefix("token_mixer.reparam_conv.") {
                return key(if l == "weight" { "token_mixer.conv.weight".into() } else { "token_mixer.bias".into() });
            }
            if leaf == "layer_scale" {
                return Some(Out::LayerScale(format!("{prefix}.layer_scale_sb")));
            }
            if let Some(l) = leaf.strip_prefix("convffn.") {
                return key(convffn_leaf(l)?);
            }
            None
        }
        Kind::Attn => {
            match leaf {
                "norm.weight" | "norm.bias" => return key(leaf.into()),
                "token_mixer.qkv.weight" | "token_mixer.proj.weight" | "token_mixer.proj.bias" => return key(leaf.into()),
                "layer_scale_1" => return Some(Out::LayerScale(format!("{prefix}.layer_scale_1_sb"))),
                "layer_scale_2" => return Some(Out::LayerScale(format!("{prefix}.layer_scale_2_sb"))),
                _ => {}
            }
            if let Some(l) = leaf.strip_prefix("convffn.") {
                return key(convffn_leaf(l)?);
            }
            None
        }
    }
}

/// Build brain's mobileclip_l weight map from the checkpoint tensors. `tensors`:
/// `(name, data)` for every `model.vision_tower.*` tensor. Unmapped tensors (e.g.
/// `num_batches_tracked`, the unused CLIP `head.*`) are dropped.
pub fn build_vision_weights(tensors: &[(String, Vec<f32>)]) -> HashMap<String, Vec<f32>> {
    let mut out = HashMap::new();
    for (name, data) in tensors {
        let Some(rest) = name.strip_prefix(PREFIX) else { continue };
        let mapped = if let Some(l) = rest.strip_prefix("patch_embed.") {
            // stem: patch_embed.{i}.reparam_conv.{weight,bias} → stem.{i}.{conv.weight,bias}
            let (i, leaf) = l.split_once('.').unwrap();
            match leaf {
                "reparam_conv.weight" => Some(Out::Rename(format!("stem.{i}.conv.weight"))),
                "reparam_conv.bias" => Some(Out::Rename(format!("stem.{i}.bias"))),
                _ => None,
            }
        } else if let Some(l) = rest.strip_prefix("conv_exp.") {
            map_leaf("conv_exp", Kind::Conv, l)
        } else if let Some(l) = rest.strip_prefix("network.") {
            let Some((n, tail)) = l.split_once('.') else { continue };
            let Ok(n) = n.parse::<u32>() else { continue };
            let Some((prefix, kind)) = stage_prefix(n, tail) else { continue };
            // For block stages, strip the block index from the tail.
            let leaf = if matches!(kind, Kind::Rep | Kind::Attn) { tail.split_once('.').unwrap().1 } else { tail };
            map_leaf(&prefix, kind, leaf)
        } else {
            None // head.* (CLIP head, unused by LLaVA)
        };
        match mapped {
            Some(Out::Rename(k)) => {
                out.insert(k, data.clone());
            }
            Some(Out::LayerScale(k)) => {
                // brain film sb = [scale = ls-1 (C), shift = 0 (C)].
                let mut sb: Vec<f32> = data.iter().map(|v| v - 1.0).collect();
                sb.extend(std::iter::repeat(0.0).take(data.len()));
                out.insert(k, sb);
            }
            None => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {

#[allow(dead_code)]
use brain_testutil::testdata;
#[allow(dead_code)]
fn repo_path(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}
    use super::*;
    use crate::encoder::{ctx, Encoder, PIPELINES};
    use gpu_core::Gpu;

    #[test]
    fn vision_import_covers_every_brain_param() {
        // Every mobileclip_l parameter brain expects must be produced by the import
        // from the real checkpoint's vision tensors (skips if the checkpoint absent).
        let path = testdata("vl/fastvlm/hf/FastVLM-0.5B/model.safetensors");
        let Ok(tensors) = checkpoint::safetensors::read(&path) else {
            eprintln!("skip: FastVLM checkpoint not present");
            return;
        };
        let vt: Vec<(String, Vec<f32>)> = tensors.into_iter().filter(|t| t.name.contains("vision_tower")).map(|t| (t.name, t.data)).collect();
        let weights = build_vision_weights(&vt);

        let gpu = Gpu::new_cpu(PIPELINES);
        let ctx = ctx(&gpu);
        let enc = Encoder::mobileclip_l(&ctx, 1024);
        let mut missing = Vec::new();
        for (name, sz) in enc.param_list() {
            match weights.get(&name) {
                Some(v) if v.len() == sz => {}
                Some(v) => missing.push(format!("{name}: size {} != {sz}", v.len())),
                None => missing.push(format!("{name}: absent")),
            }
        }
        assert!(missing.is_empty(), "unimported/mismatched mobileclip params ({}): {:?}", missing.len(), &missing[..missing.len().min(12)]);
        eprintln!("vision import: {} brain params all covered from {} checkpoint tensors", enc.param_list().len(), vt.len());
    }
}
