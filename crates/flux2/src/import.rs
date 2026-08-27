// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import: BFL reference names (canonical) and the diffusers
//! `transformer/` layout (remapped + re-fused onto the BFL names).
//!
//! Both paths validate **full two-way coverage** against
//! [`Flux2Config::tensor_manifest`]: every expected tensor produced exactly
//! once with the right shape, and no source tensor left unused — a mismatch is
//! an error naming the tensor, never a silent zero-fill (the same discipline as
//! `qwen3::import::brain_init_from_hf`).

use std::collections::HashMap;

use checkpoint::safetensors::StTensor;

use crate::config::Flux2Config;

/// name -> (shape, fp32 data), keyed by canonical BFL names.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Two-way coverage against [`Flux2Config::tensor_manifest`], by SHAPE only:
/// every expected tensor present with the right shape, and nothing in
/// `present` that the manifest does not name.
///
/// Split out of [`validate`] so a loader that never materializes fp32 can
/// still run the identical check. This is the check that catches a wrong
/// checkpoint, and it must run BEFORE any weight is read - a streaming loader
/// that validated as it went would be halfway through uploading a mismatched
/// model before it noticed.
pub fn validate_manifest(
    shape_of: &dyn Fn(&str) -> Option<Vec<usize>>,
    present: &[String],
    cfg: &Flux2Config,
) -> Result<(), String> {
    let manifest = cfg.tensor_manifest();
    for (name, shape) in &manifest {
        match shape_of(name) {
            None => return Err(format!("import: missing tensor {name}")),
            Some(s) => {
                if &s != shape {
                    return Err(format!("import: {name} shape {s:?}, expected {shape:?}"));
                }
            }
        }
    }
    if present.len() != manifest.len() {
        let expected: std::collections::HashSet<&str> = manifest.iter().map(|(n, _)| n.as_str()).collect();
        let extra: Vec<&String> = present.iter().filter(|k| !expected.contains(k.as_str())).collect();
        return Err(format!("import: unused source tensors: {extra:?}"));
    }
    Ok(())
}

fn validate(map: Tensors, cfg: &Flux2Config) -> Result<Tensors, String> {
    let names: Vec<String> = map.keys().cloned().collect();
    validate_manifest(&|n| map.get(n).map(|(s, _)| s.clone()), &names, cfg)?;
    // Shapes agree; now check each tensor actually carries that many values -
    // something only a materialized map can be asked.
    for (name, shape) in cfg.tensor_manifest() {
        let n: usize = shape.iter().product();
        let d = &map[&name].1;
        if d.len() != n {
            return Err(format!("import: {name} has {} values, expected {n}", d.len()));
        }
    }
    Ok(map)
}

/// Import a checkpoint already using the BFL reference names (the single-file
/// `flux-2-klein-*.safetensors` releases and the GGUF conversions).
pub fn import_bfl(tensors: Vec<StTensor>, cfg: &Flux2Config) -> Result<Tensors, String> {
    let map: Tensors = tensors
        .into_iter()
        .map(|t| (t.name, (t.shape, t.data)))
        .collect();
    validate(map, cfg)
}

/// Map one diffusers `Flux2Transformer2DModel` tensor name to its BFL name.
///
/// Split q/k/v projections return `Some((bfl_qkv_name, slot))` via the second
/// path in [`import_diffusers`]; this function handles the 1:1 renames only.
fn diffusers_to_bfl(name: &str) -> Option<String> {
    // fixed-name tensors
    let fixed = [
        ("x_embedder.weight", "img_in.weight"),
        ("context_embedder.weight", "txt_in.weight"),
        (
            "time_guidance_embed.timestep_embedder.linear_1.weight",
            "time_in.in_layer.weight",
        ),
        (
            "time_guidance_embed.timestep_embedder.linear_2.weight",
            "time_in.out_layer.weight",
        ),
        (
            "time_guidance_embed.guidance_embedder.linear_1.weight",
            "guidance_in.in_layer.weight",
        ),
        (
            "time_guidance_embed.guidance_embedder.linear_2.weight",
            "guidance_in.out_layer.weight",
        ),
        (
            "double_stream_modulation_img.linear.weight",
            "double_stream_modulation_img.lin.weight",
        ),
        (
            "double_stream_modulation_txt.linear.weight",
            "double_stream_modulation_txt.lin.weight",
        ),
        (
            "single_stream_modulation.linear.weight",
            "single_stream_modulation.lin.weight",
        ),
        ("norm_out.linear.weight", "final_layer.adaLN_modulation.1.weight"),
        ("proj_out.weight", "final_layer.linear.weight"),
    ];
    for (df, bfl) in fixed {
        if name == df {
            return Some(bfl.to_string());
        }
    }

    if let Some(rest) = name.strip_prefix("transformer_blocks.") {
        let (n, leaf) = rest.split_once('.')?;
        let mapped = match leaf {
            "attn.to_out.0.weight" => format!("double_blocks.{n}.img_attn.proj.weight"),
            "attn.to_add_out.weight" => format!("double_blocks.{n}.txt_attn.proj.weight"),
            "attn.norm_q.weight" => {
                format!("double_blocks.{n}.img_attn.norm.query_norm.scale")
            }
            "attn.norm_k.weight" => {
                format!("double_blocks.{n}.img_attn.norm.key_norm.scale")
            }
            "attn.norm_added_q.weight" => {
                format!("double_blocks.{n}.txt_attn.norm.query_norm.scale")
            }
            "attn.norm_added_k.weight" => {
                format!("double_blocks.{n}.txt_attn.norm.key_norm.scale")
            }
            "ff.linear_in.weight" => format!("double_blocks.{n}.img_mlp.0.weight"),
            "ff.linear_out.weight" => format!("double_blocks.{n}.img_mlp.2.weight"),
            "ff_context.linear_in.weight" => format!("double_blocks.{n}.txt_mlp.0.weight"),
            "ff_context.linear_out.weight" => {
                format!("double_blocks.{n}.txt_mlp.2.weight")
            }
            _ => return None,
        };
        return Some(mapped);
    }
    if let Some(rest) = name.strip_prefix("single_transformer_blocks.") {
        let (n, leaf) = rest.split_once('.')?;
        let mapped = match leaf {
            "attn.to_qkv_mlp_proj.weight" => format!("single_blocks.{n}.linear1.weight"),
            "attn.to_out.weight" => format!("single_blocks.{n}.linear2.weight"),
            "attn.norm_q.weight" => format!("single_blocks.{n}.norm.query_norm.scale"),
            "attn.norm_k.weight" => format!("single_blocks.{n}.norm.key_norm.scale"),
            _ => return None,
        };
        return Some(mapped);
    }
    None
}

/// The q/k/v third a split diffusers projection occupies in the fused BFL qkv.
fn qkv_slot(name: &str) -> Option<(String, usize)> {
    let rest = name.strip_prefix("transformer_blocks.")?;
    let (n, leaf) = rest.split_once('.')?;
    let (stream, slot) = match leaf {
        "attn.to_q.weight" => ("img", 0),
        "attn.to_k.weight" => ("img", 1),
        "attn.to_v.weight" => ("img", 2),
        "attn.add_q_proj.weight" => ("txt", 0),
        "attn.add_k_proj.weight" => ("txt", 1),
        "attn.add_v_proj.weight" => ("txt", 2),
        _ => return None,
    };
    Some((format!("double_blocks.{n}.{stream}_attn.qkv.weight"), slot))
}

/// Import the diffusers `transformer/` folder layout: rename, re-fuse the
/// split double-block q/k/v projections (q‖k‖v along dim 0), and swap the
/// halves of `norm_out.linear` — diffusers' `AdaLayerNormContinuous` chunks
/// (scale, shift) where the BFL `final_layer` chunks (shift, scale), so the
/// same rows mean different things in the two layouts.
pub fn import_diffusers(
    tensors: Vec<StTensor>,
    cfg: &Flux2Config,
) -> Result<Tensors, String> {
    let d = cfg.hidden;
    let mut map: Tensors = HashMap::new();
    // fused qkv assembly: name -> [q, k, v] thirds
    let mut qkv: HashMap<String, [Option<Vec<f32>>; 3]> = HashMap::new();

    for t in tensors {
        if let Some((fused_name, slot)) = qkv_slot(&t.name) {
            if t.shape != vec![d, d] {
                return Err(format!(
                    "import: {} shape {:?}, expected [{d}, {d}]",
                    t.name, t.shape
                ));
            }
            let entry = qkv.entry(fused_name.clone()).or_default();
            if entry[slot].is_some() {
                return Err(format!("import: duplicate qkv third {}", t.name));
            }
            entry[slot] = Some(t.data);
            continue;
        }
        let Some(bfl) = diffusers_to_bfl(&t.name) else {
            return Err(format!("import: unrecognized diffusers tensor {}", t.name));
        };
        let (shape, data) = if bfl == "final_layer.adaLN_modulation.1.weight" {
            if t.shape != vec![2 * d, d] {
                return Err(format!(
                    "import: {} shape {:?}, expected [{}, {d}]",
                    t.name,
                    t.shape,
                    2 * d
                ));
            }
            // rows [0:D] (diffusers scale) -> BFL rows [D:2D]; and vice versa
            let half = d * d;
            let mut w = Vec::with_capacity(2 * half);
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

    for (name, thirds) in qkv {
        let [q, k, v] = thirds;
        let (Some(q), Some(k), Some(v)) = (q, k, v) else {
            return Err(format!("import: incomplete q/k/v set for {name}"));
        };
        let mut w = Vec::with_capacity(3 * d * d);
        w.extend_from_slice(&q);
        w.extend_from_slice(&k);
        w.extend_from_slice(&v);
        if map.insert(name.clone(), (vec![3 * d, d], w)).is_some() {
            return Err(format!("import: duplicate mapping onto {name}"));
        }
    }

    validate(map, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// klein-4b topology at toy dims so debug-mode tests stay fast.
    fn tiny() -> Flux2Config {
        Flux2Config {
            in_channels: 8,
            context_in_dim: 24,
            hidden: 16,
            n_heads: 4,
            axes_dim: [1, 1, 1, 1],
            ..Flux2Config::klein_4b()
        }
    }

    fn fake_bfl(cfg: &Flux2Config) -> Vec<StTensor> {
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
        let map = import_bfl(fake_bfl(&cfg), &cfg).unwrap();
        assert_eq!(map.len(), 149);

        // a missing tensor errors by name
        let mut short = fake_bfl(&cfg);
        short.retain(|t| t.name != "single_blocks.7.linear2.weight");
        let err = import_bfl(short, &cfg).unwrap_err();
        assert!(err.contains("single_blocks.7.linear2.weight"), "{err}");

        // an extra tensor is rejected, not ignored
        let mut extra = fake_bfl(&cfg);
        extra.push(StTensor {
            name: "guidance_in.in_layer.weight".into(),
            shape: vec![cfg.hidden, 256],
            data: vec![0.0; cfg.hidden * 256],
        });
        assert!(import_bfl(extra, &cfg).is_err());
    }

    #[test]
    fn diffusers_remap_fuses_qkv_and_swaps_final_adaln() {
        let cfg = tiny();
        let d = cfg.hidden;
        // build a diffusers-layout set from the BFL manifest
        let mut src: Vec<StTensor> = Vec::new();
        for (name, shape) in cfg.tensor_manifest() {
            let n: usize = shape.iter().product();
            if let Some(rest) = name.strip_prefix("double_blocks.") {
                if rest.ends_with("_attn.qkv.weight") {
                    let (blk, leaf) = rest.split_once('.').unwrap();
                    let stream = leaf.split('_').next().unwrap();
                    let names = if stream == "img" {
                        ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"]
                    } else {
                        [
                            "attn.add_q_proj.weight",
                            "attn.add_k_proj.weight",
                            "attn.add_v_proj.weight",
                        ]
                    };
                    for (i, leafname) in names.iter().enumerate() {
                        src.push(StTensor {
                            name: format!("transformer_blocks.{blk}.{leafname}"),
                            shape: vec![d, d],
                            // slot-tagged fill so fusion order is observable
                            data: vec![i as f32 + 1.0; d * d],
                        });
                    }
                    continue;
                }
            }
            let df_name = match name.as_str() {
                "img_in.weight" => "x_embedder.weight".to_string(),
                "txt_in.weight" => "context_embedder.weight".to_string(),
                "time_in.in_layer.weight" => {
                    "time_guidance_embed.timestep_embedder.linear_1.weight".into()
                }
                "time_in.out_layer.weight" => {
                    "time_guidance_embed.timestep_embedder.linear_2.weight".into()
                }
                "double_stream_modulation_img.lin.weight" => {
                    "double_stream_modulation_img.linear.weight".into()
                }
                "double_stream_modulation_txt.lin.weight" => {
                    "double_stream_modulation_txt.linear.weight".into()
                }
                "single_stream_modulation.lin.weight" => {
                    "single_stream_modulation.linear.weight".into()
                }
                "final_layer.adaLN_modulation.1.weight" => "norm_out.linear.weight".into(),
                "final_layer.linear.weight" => "proj_out.weight".into(),
                other => {
                    
                    other
                        .replace("double_blocks.", "transformer_blocks.")
                        .replace("img_attn.proj.weight", "attn.to_out.0.weight")
                        .replace("txt_attn.proj.weight", "attn.to_add_out.weight")
                        .replace("img_attn.norm.query_norm.scale", "attn.norm_q.weight")
                        .replace("img_attn.norm.key_norm.scale", "attn.norm_k.weight")
                        .replace(
                            "txt_attn.norm.query_norm.scale",
                            "attn.norm_added_q.weight",
                        )
                        .replace("txt_attn.norm.key_norm.scale", "attn.norm_added_k.weight")
                        .replace("img_mlp.0.weight", "ff.linear_in.weight")
                        .replace("img_mlp.2.weight", "ff.linear_out.weight")
                        .replace("txt_mlp.0.weight", "ff_context.linear_in.weight")
                        .replace("txt_mlp.2.weight", "ff_context.linear_out.weight")
                        .replace("single_blocks.", "single_transformer_blocks.")
                        .replace("linear1.weight", "attn.to_qkv_mlp_proj.weight")
                        .replace("linear2.weight", "attn.to_out.weight")
                        .replace("norm.query_norm.scale", "attn.norm_q.weight")
                        .replace("norm.key_norm.scale", "attn.norm_k.weight")
                }
            };
            let data = if name == "final_layer.adaLN_modulation.1.weight" {
                // diffusers order: scale rows first (fill 7), then shift (fill 8)
                let mut w = vec![7.0; d * d];
                w.extend(vec![8.0; d * d]);
                w
            } else {
                vec![0.25; n]
            };
            let shape = cfg
                .tensor_manifest()
                .iter()
                .find(|(m, _)| *m == name)
                .unwrap()
                .1
                .clone();
            src.push(StTensor { name: df_name, shape, data });
        }

        let map = import_diffusers(src, &cfg).unwrap();
        assert_eq!(map.len(), 149);
        // qkv fused in q,k,v order
        let (s, w) = &map["double_blocks.0.img_attn.qkv.weight"];
        assert_eq!(s, &vec![3 * d, d]);
        assert_eq!(w[0], 1.0);
        assert_eq!(w[d * d], 2.0);
        assert_eq!(w[2 * d * d], 3.0);
        // final adaLN halves swapped: BFL shift rows (first) = diffusers shift fill
        let (_, w) = &map["final_layer.adaLN_modulation.1.weight"];
        assert_eq!(w[0], 8.0);
        assert_eq!(w[d * d], 7.0);
    }
}
