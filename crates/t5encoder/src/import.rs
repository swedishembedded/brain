// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import for the HuggingFace `T5EncoderModel` layout, with **full
//! two-way coverage** against [`T5Config::tensor_manifest`]: every expected
//! tensor produced exactly once with the right shape, and no source tensor left
//! unused. A mismatch is an error naming the tensor — never a silent zero-fill
//! (the discipline of `qwen3::import::brain_init_from_hf` and
//! `flux2::import::validate`).
//!
//! The only structural change at the boundary is **fusing q/k/v** into one
//! `[3*inner, d_model]` weight (q‖k‖v along dim 0), so the device projection is
//! a single GEMM straight into the fused qkv layout the attention kernels read.

use std::collections::HashMap;

use checkpoint::safetensors::StTensor;

use crate::config::T5Config;

/// name -> (shape, fp32 data), keyed by canonical brain names.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Read the `text_encoder_2/` directory (sharded or single-file safetensors).
pub fn read_encoder(dir: &std::path::Path) -> Result<Vec<StTensor>, String> {
    checkpoint::safetensors::read_model_dir(dir)
}

fn validate(map: Tensors, cfg: &T5Config) -> Result<Tensors, String> {
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
            map.keys().map(|k| k.as_str()).filter(|k| !expected.contains(k)).collect();
        extra.sort_unstable();
        return Err(format!("import: unused source tensors: {extra:?}"));
    }
    Ok(map)
}

/// The q/k/v slot a split HF projection occupies in the fused qkv weight.
fn qkv_slot(name: &str) -> Option<(String, usize)> {
    let rest = name.strip_prefix("encoder.block.")?;
    let (l, leaf) = rest.split_once('.')?;
    let slot = match leaf {
        "layer.0.SelfAttention.q.weight" => 0,
        "layer.0.SelfAttention.k.weight" => 1,
        "layer.0.SelfAttention.v.weight" => 2,
        _ => return None,
    };
    Some((format!("blocks.{l}.qkv.weight"), slot))
}

/// Map one HF `T5EncoderModel` tensor name to its brain name (1:1 renames only;
/// q/k/v go through [`qkv_slot`]).
fn hf_to_brain(name: &str) -> Option<String> {
    match name {
        "shared.weight" => return Some("shared.weight".into()),
        "encoder.final_layer_norm.weight" => return Some("final_norm.weight".into()),
        // NOTE `T5EncoderModel.state_dict()` also carries
        // `encoder.embed_tokens.weight` as a VIEW of `shared.weight`;
        // safetensors refuses to serialise the aliased pair, so a
        // `save_pretrained` checkpoint (the released FLUX.1 `text_encoder_2`:
        // 219 tensors, verified) ships only `shared.weight`. A hand-built
        // checkpoint that does carry both is rejected below as an
        // "unrecognized T5 tensor" rather than silently taking one copy —
        // deliberate, because the two are only interchangeable while they are
        // genuinely equal and nothing here checks that.
        _ => {}
    }
    let rest = name.strip_prefix("encoder.block.")?;
    let (l, leaf) = rest.split_once('.')?;
    let mapped = match leaf {
        "layer.0.SelfAttention.o.weight" => format!("blocks.{l}.o.weight"),
        "layer.0.SelfAttention.relative_attention_bias.weight" => "rel_bias.weight".into(),
        "layer.0.layer_norm.weight" => format!("blocks.{l}.attn_norm.weight"),
        "layer.1.DenseReluDense.wi_0.weight" => format!("blocks.{l}.wi_0.weight"),
        "layer.1.DenseReluDense.wi_1.weight" => format!("blocks.{l}.wi_1.weight"),
        "layer.1.DenseReluDense.wo.weight" => format!("blocks.{l}.wo.weight"),
        "layer.1.layer_norm.weight" => format!("blocks.{l}.ff_norm.weight"),
        _ => return None,
    };
    Some(mapped)
}

/// Import an HF `T5EncoderModel` checkpoint onto the canonical manifest.
pub fn import_hf(tensors: Vec<StTensor>, cfg: &T5Config) -> Result<Tensors, String> {
    let (d, inner) = (cfg.d_model as usize, cfg.inner() as usize);
    let mut map: Tensors = HashMap::new();
    let mut qkv: HashMap<String, [Option<Vec<f32>>; 3]> = HashMap::new();

    for t in tensors {
        if let Some((fused, slot)) = qkv_slot(&t.name) {
            if t.shape != vec![inner, d] {
                return Err(format!(
                    "import: {} shape {:?}, expected [{inner}, {d}]",
                    t.name, t.shape
                ));
            }
            let e = qkv.entry(fused).or_default();
            if e[slot].is_some() {
                return Err(format!("import: duplicate q/k/v third {}", t.name));
            }
            e[slot] = Some(t.data);
            continue;
        }
        let Some(brain) = hf_to_brain(&t.name) else {
            return Err(format!("import: unrecognized T5 tensor {}", t.name));
        };
        if map.insert(brain.clone(), (t.shape, t.data)).is_some() {
            return Err(format!("import: duplicate mapping onto {brain}"));
        }
    }

    for (name, thirds) in qkv {
        let [q, k, v] = thirds;
        let (Some(q), Some(k), Some(v)) = (q, k, v) else {
            return Err(format!("import: incomplete q/k/v set for {name}"));
        };
        let mut w = Vec::with_capacity(3 * inner * d);
        w.extend_from_slice(&q);
        w.extend_from_slice(&k);
        w.extend_from_slice(&v);
        if map.insert(name.clone(), (vec![3 * inner, d], w)).is_some() {
            return Err(format!("import: duplicate mapping onto {name}"));
        }
    }

    validate(map, cfg)
}

/// Drop the shapes — the `HashMap<String, Vec<f32>>` a `ParamStore` init takes.
pub fn to_init(t: Tensors) -> HashMap<String, Vec<f32>> {
    t.into_iter().map(|(k, (_, d))| (k, d)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// XXL topology at toy dims so debug-mode tests stay fast. Widths stay
    /// multiples of 64 (the 256-byte storage-binding alignment), and
    /// `heads * d_kv = 128 != d_model = 64` so the fused-qkv row offsets
    /// (`inner * d`) cannot be confused with `d_model * d` — at XXL the two are
    /// numerically identical and a swap would be invisible.
    fn tiny() -> T5Config {
        T5Config { vocab: 128, d_model: 64, d_ff: 128, d_kv: 64, layers: 2, heads: 2, ..T5Config::xxl() }
    }

    /// The HF-layout source set the tiny config maps from.
    fn fake_hf(cfg: &T5Config) -> Vec<StTensor> {
        let (d, ff, inner) = (cfg.d_model as usize, cfg.d_ff as usize, cfg.inner() as usize);
        let mut v = vec![
            StTensor {
                name: "shared.weight".into(),
                shape: vec![cfg.vocab as usize, d],
                data: vec![0.5; cfg.vocab as usize * d],
            },
            StTensor {
                name: "encoder.final_layer_norm.weight".into(),
                shape: vec![d],
                data: vec![0.5; d],
            },
            StTensor {
                name: "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(),
                shape: vec![cfg.rel_buckets as usize, cfg.heads as usize],
                data: vec![0.5; (cfg.rel_buckets * cfg.heads) as usize],
            },
        ];
        for l in 0..cfg.layers {
            let p = format!("encoder.block.{l}");
            // slot-tagged fills so the fusion ORDER is observable
            for (leaf, fill) in [("q", 1.0f32), ("k", 2.0), ("v", 3.0)] {
                v.push(StTensor {
                    name: format!("{p}.layer.0.SelfAttention.{leaf}.weight"),
                    shape: vec![inner, d],
                    data: vec![fill; inner * d],
                });
            }
            v.push(StTensor {
                name: format!("{p}.layer.0.SelfAttention.o.weight"),
                shape: vec![d, inner],
                data: vec![0.5; d * inner],
            });
            v.push(StTensor {
                name: format!("{p}.layer.0.layer_norm.weight"),
                shape: vec![d],
                data: vec![0.5; d],
            });
            for leaf in ["wi_0", "wi_1"] {
                v.push(StTensor {
                    name: format!("{p}.layer.1.DenseReluDense.{leaf}.weight"),
                    shape: vec![ff, d],
                    data: vec![0.5; ff * d],
                });
            }
            v.push(StTensor {
                name: format!("{p}.layer.1.DenseReluDense.wo.weight"),
                shape: vec![d, ff],
                data: vec![0.5; d * ff],
            });
            v.push(StTensor {
                name: format!("{p}.layer.1.layer_norm.weight"),
                shape: vec![d],
                data: vec![0.5; d],
            });
        }
        v
    }

    #[test]
    fn hf_import_fuses_qkv_with_two_way_coverage() {
        let cfg = tiny();
        let src = fake_hf(&cfg);
        // 3 globals + 9 per block; fuses to 2 + 7 per block + 1.
        assert_eq!(src.len(), 3 + 9 * cfg.layers as usize);
        let map = import_hf(src, &cfg).unwrap();
        assert_eq!(map.len(), cfg.tensor_manifest().len());

        let (inner, d) = (cfg.inner() as usize, cfg.d_model as usize);
        let (s, w) = &map["blocks.1.qkv.weight"];
        assert_eq!(s, &vec![3 * inner, d]);
        assert_eq!(w[0], 1.0, "q third first");
        assert_eq!(w[inner * d], 2.0, "k third second");
        assert_eq!(w[2 * inner * d], 3.0, "v third third");
    }

    #[test]
    fn missing_and_extra_tensors_error_by_name() {
        let cfg = tiny();
        let mut short = fake_hf(&cfg);
        short.retain(|t| t.name != "encoder.block.1.layer.1.DenseReluDense.wo.weight");
        let err = import_hf(short, &cfg).unwrap_err();
        assert!(err.contains("blocks.1.wo.weight"), "{err}");

        let mut nokv = fake_hf(&cfg);
        nokv.retain(|t| t.name != "encoder.block.0.layer.0.SelfAttention.k.weight");
        let err = import_hf(nokv, &cfg).unwrap_err();
        assert!(err.contains("incomplete q/k/v set"), "{err}");

        let mut extra = fake_hf(&cfg);
        extra.push(StTensor { name: "decoder.block.0.layer.0.layer_norm.weight".into(), shape: vec![1], data: vec![0.0] });
        let err = import_hf(extra, &cfg).unwrap_err();
        assert!(err.contains("unrecognized"), "{err}");

        // a wrongly-shaped q/k/v third is an error, not a reshape
        let mut bad = fake_hf(&cfg);
        for t in bad.iter_mut() {
            if t.name.ends_with("block.0.layer.0.SelfAttention.q.weight") {
                t.shape = vec![cfg.d_model as usize, cfg.inner() as usize + 1];
            }
        }
        assert!(import_hf(bad, &cfg).is_err());
    }
}
