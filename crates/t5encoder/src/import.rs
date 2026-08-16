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

/// Read the native umT5 encoder checkpoint, which Wan2.1 ships as a single
/// bf16 `.pth` (`models_t5_umt5-xxl-enc-bf16.pth`) rather than safetensors.
/// [`checkpoint::torchpt`] widens every storage to f32, so the result is the
/// same `StTensor` shape [`import_wan`] takes from a converted file.
pub fn read_encoder_pth(path: &std::path::Path) -> Result<Vec<StTensor>, String> {
    let r = checkpoint::torchpt::read(path.to_str().ok_or("non-utf8 path")?)?;
    Ok(r.into_iter()
        .map(|t| StTensor { name: t.name, shape: t.shape, data: t.data })
        .collect())
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

/// The q/k/v slot a split projection occupies, in EITHER name space.
fn any_qkv_slot(name: &str) -> Option<(String, usize)> {
    qkv_slot(name).or_else(|| wan_qkv_slot(name))
}

/// The Wan/umT5 module-tree spelling of [`qkv_slot`]
/// (`blocks.<l>.attn.{q,k,v}.weight`).
fn wan_qkv_slot(name: &str) -> Option<(String, usize)> {
    let rest = name.strip_prefix("blocks.")?;
    let (l, leaf) = rest.split_once('.')?;
    let slot = match leaf {
        "attn.q.weight" => 0,
        "attn.k.weight" => 1,
        "attn.v.weight" => 2,
        _ => return None,
    };
    Some((format!("blocks.{l}.qkv.weight"), slot))
}

/// Map one tensor of the **native** umT5 checkpoint
/// (`models_t5_umt5-xxl-enc-bf16.pth`, a `T5Encoder` state dict) to its brain
/// name.
///
/// The FFN is where this layout is easy to get backwards. `T5FeedForward` is
/// `gate = Sequential(Linear(dim, dim_ffn), GELU())`, `fc1 = Linear(dim,
/// dim_ffn)`, `fc2 = Linear(dim_ffn, dim)`, and its forward is
/// `fc2(fc1(x) * gate(x))`. So the GELU sits on **`ffn.gate.0`**, which is
/// brain's `wi_0` (the activated half), and `ffn.fc1` is `wi_1` (the linear
/// half) - the numeral in the reference's name is the opposite way round from
/// brain's and from HF's. Swapping them keeps every shape valid and every
/// import check green while computing `gelu(wi_1) * wi_0`.
fn wan_to_brain(name: &str) -> Option<String> {
    match name {
        "token_embedding.weight" => return Some("shared.weight".into()),
        "norm.weight" => return Some("final_norm.weight".into()),
        _ => {}
    }
    let rest = name.strip_prefix("blocks.")?;
    let (l, leaf) = rest.split_once('.')?;
    let mapped = match leaf {
        "attn.o.weight" => format!("blocks.{l}.o.weight"),
        "pos_embedding.embedding.weight" => format!("blocks.{l}.rel_bias.weight"),
        "norm1.weight" => format!("blocks.{l}.attn_norm.weight"),
        "norm2.weight" => format!("blocks.{l}.ff_norm.weight"),
        "ffn.gate.0.weight" => format!("blocks.{l}.wi_0.weight"),
        "ffn.fc1.weight" => format!("blocks.{l}.wi_1.weight"),
        "ffn.fc2.weight" => format!("blocks.{l}.wo.weight"),
        _ => return None,
    };
    Some(mapped)
}

/// Import the native umT5 encoder checkpoint (`T5Encoder.state_dict()`, 242
/// tensors) onto the canonical manifest, with the same two-way coverage and the
/// same q/k/v fusion as [`import_hf`].
pub fn import_wan(tensors: Vec<StTensor>, cfg: &T5Config) -> Result<Tensors, String> {
    if !cfg.per_block_rel_bias {
        return Err("import: the umT5 checkpoint has a per-block relative bias, \
                    but this config shares one table"
            .into());
    }
    import_named(tensors, cfg, wan_to_brain, "umT5")
}

/// Import an HF `T5EncoderModel` checkpoint onto the canonical manifest.
pub fn import_hf(tensors: Vec<StTensor>, cfg: &T5Config) -> Result<Tensors, String> {
    if cfg.per_block_rel_bias {
        return Err("import: an HF T5EncoderModel ships ONE relative bias table, \
                    but this config wants one per block"
            .into());
    }
    import_named(tensors, cfg, hf_to_brain, "T5")
}

/// The shared body of both importers: rename, fuse q/k/v, validate both ways.
fn import_named(
    tensors: Vec<StTensor>,
    cfg: &T5Config,
    rename: fn(&str) -> Option<String>,
    what: &str,
) -> Result<Tensors, String> {
    let (d, inner) = (cfg.d_model as usize, cfg.inner() as usize);
    let mut map: Tensors = HashMap::new();
    let mut qkv: HashMap<String, [Option<Vec<f32>>; 3]> = HashMap::new();

    for t in tensors {
        if let Some((fused, slot)) = any_qkv_slot(&t.name) {
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
        let Some(brain) = rename(&t.name) else {
            return Err(format!("import: unrecognized {what} tensor {}", t.name));
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

    /// umT5 at toy dims: the per-block bias table makes the manifest 1 bigger
    /// per block, and `heads * d_kv != d_model` again so a fused-offset swap
    /// cannot hide.
    fn tiny_umt5() -> T5Config {
        T5Config {
            vocab: 128,
            d_model: 64,
            d_ff: 128,
            d_kv: 64,
            layers: 2,
            heads: 2,
            ..T5Config::umt5_xxl()
        }
    }

    /// The native `T5Encoder.state_dict()` name space.
    fn fake_wan(cfg: &T5Config) -> Vec<StTensor> {
        let (d, ff, inner) = (cfg.d_model as usize, cfg.d_ff as usize, cfg.inner() as usize);
        let mut v = vec![
            StTensor {
                name: "token_embedding.weight".into(),
                shape: vec![cfg.vocab as usize, d],
                data: vec![0.5; cfg.vocab as usize * d],
            },
            StTensor { name: "norm.weight".into(), shape: vec![d], data: vec![0.5; d] },
        ];
        for l in 0..cfg.layers {
            let p = format!("blocks.{l}");
            for (leaf, fill) in [("q", 1.0f32), ("k", 2.0), ("v", 3.0)] {
                v.push(StTensor {
                    name: format!("{p}.attn.{leaf}.weight"),
                    shape: vec![inner, d],
                    data: vec![fill; inner * d],
                });
            }
            v.push(StTensor {
                name: format!("{p}.attn.o.weight"),
                shape: vec![d, inner],
                data: vec![0.5; d * inner],
            });
            for leaf in ["norm1", "norm2"] {
                v.push(StTensor {
                    name: format!("{p}.{leaf}.weight"),
                    shape: vec![d],
                    data: vec![0.5; d],
                });
            }
            // Slot-tagged so the gate/fc1 -> wi_0/wi_1 direction is observable.
            v.push(StTensor {
                name: format!("{p}.ffn.gate.0.weight"),
                shape: vec![ff, d],
                data: vec![7.0; ff * d],
            });
            v.push(StTensor {
                name: format!("{p}.ffn.fc1.weight"),
                shape: vec![ff, d],
                data: vec![9.0; ff * d],
            });
            v.push(StTensor {
                name: format!("{p}.ffn.fc2.weight"),
                shape: vec![d, ff],
                data: vec![0.5; d * ff],
            });
            // Per-block table, filled with the block index so a shared-bias
            // import cannot pass by accident.
            v.push(StTensor {
                name: format!("{p}.pos_embedding.embedding.weight"),
                shape: vec![cfg.rel_buckets as usize, cfg.heads as usize],
                data: vec![l as f32 + 1.0; (cfg.rel_buckets * cfg.heads) as usize],
            });
        }
        v
    }

    #[test]
    fn wan_import_maps_the_native_names_with_two_way_coverage() {
        let cfg = tiny_umt5();
        let src = fake_wan(&cfg);
        // 2 globals + 10 per block; fuses to 1 + 8 per block + 1.
        assert_eq!(src.len(), 2 + 10 * cfg.layers as usize);
        let map = import_wan(src, &cfg).expect("import_wan");
        assert_eq!(map.len(), cfg.tensor_manifest().len());

        let (inner, d, ff) = (cfg.inner() as usize, cfg.d_model as usize, cfg.d_ff as usize);
        let (s, w) = &map["blocks.1.qkv.weight"];
        assert_eq!(s, &vec![3 * inner, d]);
        assert_eq!((w[0], w[inner * d], w[2 * inner * d]), (1.0, 2.0, 3.0), "q|k|v order");
        // `ffn.gate.0` is the ACTIVATED half and must land on wi_0.
        assert_eq!(map["blocks.0.wi_0.weight"].1[0], 7.0, "gate.0 -> wi_0");
        assert_eq!(map["blocks.0.wi_1.weight"].1[0], 9.0, "fc1 -> wi_1");
        assert_eq!(map["blocks.0.wo.weight"].0, vec![d, ff]);
        // Each block keeps its OWN bias table.
        assert_eq!(map["blocks.0.rel_bias.weight"].1[0], 1.0);
        assert_eq!(map["blocks.1.rel_bias.weight"].1[0], 2.0);
    }

    #[test]
    fn wan_import_rejects_missing_extra_and_the_wrong_config() {
        let cfg = tiny_umt5();

        let mut short = fake_wan(&cfg);
        short.retain(|t| t.name != "blocks.1.pos_embedding.embedding.weight");
        let err = import_wan(short, &cfg).unwrap_err();
        assert!(err.contains("blocks.1.rel_bias.weight"), "{err}");

        let mut extra = fake_wan(&cfg);
        extra.push(StTensor { name: "head.weight".into(), shape: vec![1], data: vec![0.0] });
        let err = import_wan(extra, &cfg).unwrap_err();
        assert!(err.contains("unrecognized umT5"), "{err}");

        // The two name spaces do not silently accept each other's files.
        let err = import_wan(fake_hf(&tiny()), &cfg).unwrap_err();
        assert!(err.contains("unrecognized umT5"), "{err}");
        let err = import_hf(fake_wan(&cfg), &tiny()).unwrap_err();
        assert!(err.contains("unrecognized T5"), "{err}");

        // ...nor each other's CONFIG, which is the failure that would otherwise
        // surface as 23 missing tensors instead of one clear sentence.
        let err = import_wan(fake_wan(&cfg), &tiny()).unwrap_err();
        assert!(err.contains("per-block"), "{err}");
        let err = import_hf(fake_hf(&tiny()), &cfg).unwrap_err();
        assert!(err.contains("one per block"), "{err}");
    }
}
