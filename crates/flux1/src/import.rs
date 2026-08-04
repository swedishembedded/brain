// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import: BFL reference names (canonical) and the diffusers
//! `transformer/` layout (remapped + re-fused onto the BFL names).
//!
//! Both paths validate **full two-way coverage** against
//! [`Flux1Config::tensor_manifest`]: every expected tensor produced exactly
//! once with the right shape, and no source tensor left unused — a mismatch is
//! an error naming the tensor, never a silent zero-fill (the same discipline as
//! `flux2::import` and `qwen::import::brain_init_from_hf`).
//!
//! Two fusions happen here, so every device matmul later reads a whole buffer:
//! the double blocks' split `to_q`/`to_k`/`to_v` become one `attn.qkv`, and the
//! single blocks' `to_q`/`to_k`/`to_v`/`proj_mlp` become one `linear1`
//! (`[3D + mlp, D]`, the reference's own fused layout).
//!
//! One semantic remap: diffusers' `AdaLayerNormContinuous` chunks
//! `(scale, shift)` where BFL's `final_layer` chunks `(shift, scale)`, so the
//! two halves of `norm_out.linear` (weight AND bias) are swapped onto the
//! canonical name. The per-block `AdaLayerNormZero{,Single}` modulations do
//! NOT need this — diffusers chunks them `(shift, scale, gate)`, same as BFL.

use std::collections::HashMap;

use checkpoint::safetensors::StTensor;

use crate::config::Flux1Config;

/// name -> (shape, fp32 data), keyed by canonical BFL names.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Partially assembled fused tensors during a diffusers import:
/// fused BFL name -> (arity, one slot per source projection).
type FusedParts = HashMap<String, (usize, Vec<Option<Vec<f32>>>)>;

fn validate(map: Tensors, cfg: &Flux1Config) -> Result<Tensors, String> {
    let manifest = cfg.tensor_manifest();
    for (name, shape) in &manifest {
        match map.get(name) {
            None => return Err(format!("import: missing tensor {name}")),
            Some((s, d)) => {
                if s != shape {
                    return Err(format!("import: {name} shape {s:?}, expected {shape:?}"));
                }
                let n: usize = shape.iter().product();
                if d.len() != n {
                    return Err(format!("import: {name} has {} values, expected {n}", d.len()));
                }
            }
        }
    }
    if map.len() != manifest.len() {
        let expected: std::collections::HashSet<&str> =
            manifest.iter().map(|(n, _)| n.as_str()).collect();
        let mut extra: Vec<&str> =
            map.keys().map(String::as_str).filter(|k| !expected.contains(k)).collect();
        extra.sort_unstable();
        extra.truncate(16);
        return Err(format!(
            "import: {} unused source tensors, e.g. {extra:?}",
            map.len() - manifest.len()
        ));
    }
    Ok(map)
}

/// Import a checkpoint already using the BFL reference names (the single-file
/// `flux1-*.safetensors` releases and the GGUF conversions).
pub fn import_bfl(tensors: Vec<StTensor>, cfg: &Flux1Config) -> Result<Tensors, String> {
    let map: Tensors = tensors.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
    validate(map, cfg)
}

/// Map one diffusers `FluxTransformer2DModel` tensor name to its BFL name.
/// Returns `None` for the fused-projection members, which
/// [`fused_slot`] claims instead.
fn diffusers_to_bfl(name: &str) -> Option<String> {
    let (stem, suffix) = name.rsplit_once('.')?;
    if suffix != "weight" && suffix != "bias" {
        return None;
    }
    let bfl_stem = match stem {
        "x_embedder" => "img_in".to_string(),
        "context_embedder" => "txt_in".to_string(),
        "time_text_embed.timestep_embedder.linear_1" => "time_in.in_layer".into(),
        "time_text_embed.timestep_embedder.linear_2" => "time_in.out_layer".into(),
        "time_text_embed.guidance_embedder.linear_1" => "guidance_in.in_layer".into(),
        "time_text_embed.guidance_embedder.linear_2" => "guidance_in.out_layer".into(),
        "time_text_embed.text_embedder.linear_1" => "vector_in.in_layer".into(),
        "time_text_embed.text_embedder.linear_2" => "vector_in.out_layer".into(),
        "norm_out.linear" => "final_layer.adaLN_modulation.1".into(),
        "proj_out" => "final_layer.linear".into(),
        other => {
            if let Some(rest) = other.strip_prefix("transformer_blocks.") {
                let (n, leaf) = rest.split_once('.')?;
                match leaf {
                    "norm1.linear" => format!("double_blocks.{n}.img_mod.lin"),
                    "norm1_context.linear" => format!("double_blocks.{n}.txt_mod.lin"),
                    "attn.to_out.0" => format!("double_blocks.{n}.img_attn.proj"),
                    "attn.to_add_out" => format!("double_blocks.{n}.txt_attn.proj"),
                    "attn.norm_q" => {
                        format!("double_blocks.{n}.img_attn.norm.query_norm.scale#")
                    }
                    "attn.norm_k" => format!("double_blocks.{n}.img_attn.norm.key_norm.scale#"),
                    "attn.norm_added_q" => {
                        format!("double_blocks.{n}.txt_attn.norm.query_norm.scale#")
                    }
                    "attn.norm_added_k" => {
                        format!("double_blocks.{n}.txt_attn.norm.key_norm.scale#")
                    }
                    "ff.net.0.proj" => format!("double_blocks.{n}.img_mlp.0"),
                    "ff.net.2" => format!("double_blocks.{n}.img_mlp.2"),
                    "ff_context.net.0.proj" => format!("double_blocks.{n}.txt_mlp.0"),
                    "ff_context.net.2" => format!("double_blocks.{n}.txt_mlp.2"),
                    _ => return None,
                }
            } else if let Some(rest) = other.strip_prefix("single_transformer_blocks.") {
                let (n, leaf) = rest.split_once('.')?;
                match leaf {
                    "norm.linear" => format!("single_blocks.{n}.modulation.lin"),
                    "proj_out" => format!("single_blocks.{n}.linear2"),
                    "attn.norm_q" => format!("single_blocks.{n}.norm.query_norm.scale#"),
                    "attn.norm_k" => format!("single_blocks.{n}.norm.key_norm.scale#"),
                    _ => return None,
                }
            } else {
                return None;
            }
        }
    };
    // The QK-norm scales are the only unbiased tensors: their canonical name
    // already ends in `.scale` and takes no `.weight`/`.bias` suffix. The `#`
    // sentinel above marks them so the suffix is dropped exactly here.
    Some(match bfl_stem.strip_suffix('#') {
        Some(scale) => {
            if suffix != "weight" {
                return None;
            }
            scale.to_string()
        }
        None => format!("{bfl_stem}.{suffix}"),
    })
}

/// The slot a split diffusers projection occupies inside a fused BFL tensor:
/// `(fused_bfl_name, slot, arity)`.
///
/// Double blocks fuse q‖k‖v (arity 3); single blocks fuse q‖k‖v‖mlp (arity 4,
/// the reference's own `linear1`).
fn fused_slot(name: &str) -> Option<(String, usize, usize)> {
    let (stem, suffix) = name.rsplit_once('.')?;
    if suffix != "weight" && suffix != "bias" {
        return None;
    }
    if let Some(rest) = stem.strip_prefix("transformer_blocks.") {
        let (n, leaf) = rest.split_once('.')?;
        let (stream, slot) = match leaf {
            "attn.to_q" => ("img", 0),
            "attn.to_k" => ("img", 1),
            "attn.to_v" => ("img", 2),
            "attn.add_q_proj" => ("txt", 0),
            "attn.add_k_proj" => ("txt", 1),
            "attn.add_v_proj" => ("txt", 2),
            _ => return None,
        };
        return Some((format!("double_blocks.{n}.{stream}_attn.qkv.{suffix}"), slot, 3));
    }
    if let Some(rest) = stem.strip_prefix("single_transformer_blocks.") {
        let (n, leaf) = rest.split_once('.')?;
        let slot = match leaf {
            "attn.to_q" => 0,
            "attn.to_k" => 1,
            "attn.to_v" => 2,
            "proj_mlp" => 3,
            _ => return None,
        };
        return Some((format!("single_blocks.{n}.linear1.{suffix}"), slot, 4));
    }
    None
}

/// Import the diffusers `transformer/` folder layout.
pub fn import_diffusers(tensors: Vec<StTensor>, cfg: &Flux1Config) -> Result<Tensors, String> {
    let d = cfg.hidden;
    let mut map: Tensors = HashMap::new();
    let mut fused: FusedParts = HashMap::new();

    for t in tensors {
        if let Some((fused_name, slot, arity)) = fused_slot(&t.name) {
            let e = fused.entry(fused_name).or_insert_with(|| (arity, vec![None; arity]));
            if e.1[slot].is_some() {
                return Err(format!("import: duplicate fused slot for {}", t.name));
            }
            e.1[slot] = Some(t.data);
            continue;
        }
        let Some(bfl) = diffusers_to_bfl(&t.name) else {
            return Err(format!("import: unrecognized diffusers tensor {}", t.name));
        };
        let (shape, data) = if bfl.starts_with("final_layer.adaLN_modulation.1") {
            // diffusers chunks (scale, shift); BFL chunks (shift, scale)
            let half = t.data.len() / 2;
            if t.data.len() % 2 != 0 || (t.shape[0] != 2 * d) {
                return Err(format!("import: {} shape {:?}, expected 2*{d} rows", t.name, t.shape));
            }
            let mut w = Vec::with_capacity(t.data.len());
            w.extend_from_slice(&t.data[half..]);
            w.extend_from_slice(&t.data[..half]);
            (t.shape, w)
        } else {
            (t.shape, t.data)
        };
        if map.insert(bfl.clone(), (shape, data)).is_some() {
            return Err(format!("import: duplicate mapping onto {bfl}"));
        }
    }

    for (name, (arity, parts)) in fused {
        let mut w: Vec<f32> = Vec::new();
        for (i, p) in parts.iter().enumerate() {
            let Some(p) = p else {
                return Err(format!("import: incomplete fused set for {name} (slot {i} of {arity})"));
            };
            w.extend_from_slice(p);
        }
        let shape = if name.ends_with(".bias") { vec![w.len()] } else { vec![w.len() / d, d] };
        if map.insert(name.clone(), (shape, w)).is_some() {
            return Err(format!("import: duplicate mapping onto {name}"));
        }
    }

    validate(map, cfg)
}

/// Drop the blocks a reduced-depth [`Flux1Config`] does not have, so the strict
/// two-way coverage check still applies to everything that is kept.
///
/// This is the ONLY sanctioned way to import a truncated model: the count of
/// dropped tensors is returned so a caller can assert it, and every surviving
/// tensor still goes through the same `validate`. Goldens must be dumped at the
/// SAME depth (`tools/flux1_dump_reference.py --small-double/--small-single`).
pub fn truncate_to_depth(
    tensors: Vec<StTensor>,
    cfg: &Flux1Config,
) -> (Vec<StTensor>, usize) {
    let keep = |name: &str| -> bool {
        for (prefix, depth) in [
            ("transformer_blocks.", cfg.depth_double),
            ("single_transformer_blocks.", cfg.depth_single),
            ("double_blocks.", cfg.depth_double),
            ("single_blocks.", cfg.depth_single),
        ] {
            // `strip_prefix` is exact, so `single_transformer_blocks.…` never
            // matches the `transformer_blocks.` arm.
            if let Some(rest) = name.strip_prefix(prefix) {
                let Some((n, _)) = rest.split_once('.') else { return true };
                let Ok(n) = n.parse::<usize>() else { return true };
                return n < depth;
            }
        }
        true
    };
    let before = tensors.len();
    let kept: Vec<StTensor> = tensors.into_iter().filter(|t| keep(&t.name)).collect();
    let dropped = before - kept.len();
    (kept, dropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// dev topology at toy dims so debug-mode tests stay fast.
    fn tiny() -> Flux1Config {
        Flux1Config {
            in_channels: 8,
            context_in_dim: 24,
            vec_in_dim: 12,
            hidden: 16,
            n_heads: 4,
            depth_double: 2,
            depth_single: 3,
            axes_dim: [1, 1, 2],
            ..Flux1Config::dev()
        }
    }

    fn fake_bfl(cfg: &Flux1Config) -> Vec<StTensor> {
        cfg.tensor_manifest()
            .into_iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                StTensor { name, shape, data: vec![0.5; n] }
            })
            .collect()
    }

    #[test]
    fn bfl_roundtrip_and_coverage() {
        let cfg = tiny();
        let want = cfg.tensor_manifest().len();
        let map = import_bfl(fake_bfl(&cfg), &cfg).unwrap();
        assert_eq!(map.len(), want);

        let mut short = fake_bfl(&cfg);
        short.retain(|t| t.name != "single_blocks.1.linear2.weight");
        let err = import_bfl(short, &cfg).unwrap_err();
        assert!(err.contains("single_blocks.1.linear2.weight"), "{err}");

        let mut extra = fake_bfl(&cfg);
        extra.push(StTensor { name: "junk.weight".into(), shape: vec![1], data: vec![0.0] });
        let err = import_bfl(extra, &cfg).unwrap_err();
        assert!(err.contains("junk.weight"), "{err}");
    }

    /// Build a diffusers-layout source set for `cfg`, tagging each fused slot
    /// with a distinct fill so the fusion ORDER is observable.
    fn fake_diffusers(cfg: &Flux1Config) -> Vec<StTensor> {
        let d = cfg.hidden;
        let mlp = cfg.mlp_hidden();
        let mut v: Vec<StTensor> = Vec::new();
        fn push(v: &mut Vec<StTensor>, name: &str, shape: Vec<usize>, fill: f32) {
            let n: usize = shape.iter().product();
            v.push(StTensor { name: name.into(), shape, data: vec![fill; n] });
        }
        fn linf(v: &mut Vec<StTensor>, name: &str, out: usize, k: usize, fill: f32) {
            push(v, &format!("{name}.weight"), vec![out, k], fill);
            push(v, &format!("{name}.bias"), vec![out], fill);
        }
        macro_rules! push {
            ($name:expr, $shape:expr, $fill:expr) => { push(&mut v, &$name, $shape, $fill) };
        }
        macro_rules! lin {
            ($name:expr, $out:expr, $k:expr, $fill:expr) => { linf(&mut v, &$name, $out, $k, $fill) };
        }
        lin!("x_embedder", d, cfg.in_channels, 0.25);
        lin!("context_embedder", d, cfg.context_in_dim, 0.25);
        lin!("time_text_embed.timestep_embedder.linear_1", d, 256, 0.25);
        lin!("time_text_embed.timestep_embedder.linear_2", d, d, 0.25);
        lin!("time_text_embed.text_embedder.linear_1", d, cfg.vec_in_dim, 0.25);
        lin!("time_text_embed.text_embedder.linear_2", d, d, 0.25);
        lin!("time_text_embed.guidance_embedder.linear_1", d, 256, 0.25);
        lin!("time_text_embed.guidance_embedder.linear_2", d, d, 0.25);
        // diffusers order is (scale, shift): rows [0:D] = 7, rows [D:2D] = 8
        v.push(StTensor {
            name: "norm_out.linear.weight".into(),
            shape: vec![2 * d, d],
            data: [vec![7.0; d * d], vec![8.0; d * d]].concat(),
        });
        v.push(StTensor {
            name: "norm_out.linear.bias".into(),
            shape: vec![2 * d],
            data: [vec![7.0; d], vec![8.0; d]].concat(),
        });
        lin!("proj_out", cfg.in_channels, d, 0.25);
        for n in 0..cfg.depth_double {
            let p = format!("transformer_blocks.{n}");
            lin!(format!("{p}.norm1.linear"), 6 * d, d, 0.25);
            lin!(format!("{p}.norm1_context.linear"), 6 * d, d, 0.25);
            for (i, leaf) in ["attn.to_q", "attn.to_k", "attn.to_v"].iter().enumerate() {
                lin!(format!("{p}.{leaf}"), d, d, i as f32 + 1.0);
            }
            for (i, leaf) in
                ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"].iter().enumerate()
            {
                lin!(format!("{p}.{leaf}"), d, d, i as f32 + 4.0);
            }
            for leaf in ["attn.norm_q", "attn.norm_k", "attn.norm_added_q", "attn.norm_added_k"] {
                push!(format!("{p}.{leaf}.weight"), vec![cfg.head_dim()], 0.25);
            }
            lin!(format!("{p}.attn.to_out.0"), d, d, 0.25);
            lin!(format!("{p}.attn.to_add_out"), d, d, 0.25);
            lin!(format!("{p}.ff.net.0.proj"), mlp, d, 0.25);
            lin!(format!("{p}.ff.net.2"), d, mlp, 0.25);
            lin!(format!("{p}.ff_context.net.0.proj"), mlp, d, 0.25);
            lin!(format!("{p}.ff_context.net.2"), d, mlp, 0.25);
        }
        for n in 0..cfg.depth_single {
            let p = format!("single_transformer_blocks.{n}");
            lin!(format!("{p}.norm.linear"), 3 * d, d, 0.25);
            for (i, leaf) in ["attn.to_q", "attn.to_k", "attn.to_v"].iter().enumerate() {
                lin!(format!("{p}.{leaf}"), d, d, i as f32 + 1.0);
            }
            lin!(format!("{p}.proj_mlp"), mlp, d, 9.0);
            for leaf in ["attn.norm_q", "attn.norm_k"] {
                push!(format!("{p}.{leaf}.weight"), vec![cfg.head_dim()], 0.25);
            }
            lin!(format!("{p}.proj_out"), d, d + mlp, 0.25);
        }
        v
    }

    #[test]
    fn diffusers_remap_fuses_and_swaps_final_adaln() {
        let cfg = tiny();
        let d = cfg.hidden;
        let mlp = cfg.mlp_hidden();
        let src = fake_diffusers(&cfg);
        // the real checkpoint has 1160 tensors for the full model; the toy set
        // must at least be self-consistent with the manifest after fusion
        let map = import_diffusers(src, &cfg).unwrap();
        assert_eq!(map.len(), cfg.tensor_manifest().len());

        // double-block qkv fused q,k,v in that order
        let (s, w) = &map["double_blocks.0.img_attn.qkv.weight"];
        assert_eq!(s, &vec![3 * d, d]);
        assert_eq!((w[0], w[d * d], w[2 * d * d]), (1.0, 2.0, 3.0));
        let (s, w) = &map["double_blocks.0.txt_attn.qkv.bias"];
        assert_eq!(s, &vec![3 * d]);
        assert_eq!((w[0], w[d], w[2 * d]), (4.0, 5.0, 6.0));

        // single-block linear1 fused q,k,v,mlp
        let (s, w) = &map["single_blocks.0.linear1.weight"];
        assert_eq!(s, &vec![3 * d + mlp, d]);
        assert_eq!((w[0], w[d * d], w[2 * d * d], w[3 * d * d]), (1.0, 2.0, 3.0, 9.0));

        // final adaLN halves swapped onto BFL (shift, scale) order
        let (_, w) = &map["final_layer.adaLN_modulation.1.weight"];
        assert_eq!((w[0], w[d * d]), (8.0, 7.0));
        let (_, b) = &map["final_layer.adaLN_modulation.1.bias"];
        assert_eq!((b[0], b[d]), (8.0, 7.0));

        // QK-norm scales carry no bias and keep the `.scale` leaf
        assert!(map.contains_key("single_blocks.0.norm.query_norm.scale"));
    }

    #[test]
    fn truncation_drops_exactly_the_out_of_range_blocks() {
        let full = tiny();
        let src = fake_diffusers(&full);
        let small = full.with_depth(1, 1);
        let (kept, dropped) = truncate_to_depth(src, &small);
        assert!(dropped > 0);
        let map = import_diffusers(kept, &small).unwrap();
        assert_eq!(map.len(), small.tensor_manifest().len());
    }
}
