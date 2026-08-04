// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `pulid_flux_v0.9.1.safetensors` → brain's canonical names, with **two-way
//! coverage validation**: every manifest tensor is produced exactly once at the
//! right shape, and every source tensor is consumed. A miss is an error naming
//! the tensor — never a zero-fill, never a skip (the `flux1`/`clip` discipline).
//!
//! The checkpoint is one flat BF16 file whose first name component selects the
//! module (`pulid_encoder.*` / `pulid_ca.*`). Its `nn.Sequential` members are
//! addressed by ordinal, so the remap is mostly de-ordinalisation:
//!
//! | source | brain |
//! |---|---|
//! | `pulid_encoder.id_embedding_mapping.{0,1,3,4,6}` | `id_map.{lin0,ln0,lin1,ln1,lin2}` |
//! | `pulid_encoder.mapping_{i}.{0,1,3,4,6}` | `map{i}.{lin0,ln0,lin1,ln1,lin2}` |
//! | `pulid_encoder.layers.{l}.0.*` | `layers.{l}.attn.*` |
//! | `pulid_encoder.layers.{l}.1.{0,1,3}` | `layers.{l}.ff.{norm,w1,w2}` |
//! | `pulid_encoder.latents` `[1,32,1024]` | `latents` `[32,1024]` |
//! | `pulid_encoder.proj_out` `[1024,2048]` | `proj_out` `[2048,1024]` (**transposed**) |
//! | `pulid_ca.{i}.*` | `ca.{i}.*` |
//!
//! Only two tensors need surgery: `latents` loses its leading batch axis, and
//! `proj_out` is transposed because the reference applies it as a bare
//! `latents @ proj_out` while every brain matmul is `x @ Wᵀ`. `to_kv` is left
//! **fused** `[2·inner, k]` on purpose — `chunk(2, -1)` puts k at column 0 and v
//! at column `inner`, which is byte-for-byte the fused-KV layout
//! `attn_{scores,apply}_cross` binds.

use std::collections::HashMap;

use checkpoint::safetensors::StTensor;

use crate::config::PulidConfig;

/// name -> (shape, fp32 data), keyed by canonical brain names.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// The two halves of the checkpoint, each validated against its own manifest.
pub struct PulidWeights {
    pub encoder: Tensors,
    pub ca: Tensors,
    /// Number of cross-attention modules found (20 in v0.9.1).
    pub num_ca: usize,
}

/// Tensor COUNTS, never the tensors: a derived `Debug` on 570 M parameters is
/// a way to hang a test runner, not a diagnostic.
impl std::fmt::Debug for PulidWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PulidWeights")
            .field("encoder_tensors", &self.encoder.len())
            .field("ca_tensors", &self.ca.len())
            .field("num_ca", &self.num_ca)
            .finish()
    }
}

fn put(map: &mut Tensors, name: String, shape: Vec<usize>, data: Vec<f32>) -> Result<(), String> {
    if map.insert(name.clone(), (shape, data)).is_some() {
        return Err(format!("pulid import: duplicate mapping onto {name}"));
    }
    Ok(())
}

fn validate(map: &Tensors, manifest: &[(String, Vec<usize>)], what: &str) -> Result<(), String> {
    for (name, shape) in manifest {
        match map.get(name) {
            None => return Err(format!("pulid import: {what} missing tensor {name}")),
            Some((s, d)) => {
                if s != shape {
                    return Err(format!("pulid import: {what} {name} shape {s:?}, expected {shape:?}"));
                }
                let n: usize = shape.iter().product();
                if d.len() != n {
                    return Err(format!("pulid import: {what} {name} has {} values, expected {n}", d.len()));
                }
            }
        }
    }
    if map.len() != manifest.len() {
        let expected: std::collections::HashSet<&str> =
            manifest.iter().map(|(n, _)| n.as_str()).collect();
        let mut extra: Vec<&String> = map.keys().filter(|k| !expected.contains(k.as_str())).collect();
        extra.sort();
        return Err(format!("pulid import: {what} unused source tensors: {extra:?}"));
    }
    Ok(())
}

/// `nn.Sequential` ordinal -> brain leaf name, for the 7-member
/// `Linear/LN/LeakyReLU/Linear/LN/LeakyReLU/Linear` mapping MLPs. The activations
/// (ordinals 2 and 5) are parameter-free, hence absent from the checkpoint.
fn mlp_leaf(ord: &str) -> Option<&'static str> {
    match ord {
        "0" => Some("lin0"),
        "1" => Some("ln0"),
        "3" => Some("lin1"),
        "4" => Some("ln1"),
        "6" => Some("lin2"),
        _ => None,
    }
}

/// `FeedForward` = `LayerNorm / Linear / GELU / Linear`.
fn ff_leaf(ord: &str) -> Option<&'static str> {
    match ord {
        "0" => Some("norm"),
        "1" => Some("w1"),
        "3" => Some("w2"),
        _ => None,
    }
}

fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Import the released `pulid_flux_v*.safetensors`.
pub fn import(src: Vec<StTensor>, cfg: &PulidConfig) -> Result<PulidWeights, String> {
    let mut enc: Tensors = HashMap::new();
    let mut ca: Tensors = HashMap::new();
    let mut num_ca = 0usize;

    for t in src {
        let parts: Vec<&str> = t.name.split('.').collect();
        let shape: Vec<usize> = t.shape.clone();
        match parts.as_slice() {
            // ---- the ID encoder ------------------------------------------
            ["pulid_encoder", "latents"] => {
                if shape != [1, cfg.num_queries, cfg.dim] {
                    return Err(format!("pulid import: latents shape {shape:?}"));
                }
                put(&mut enc, "latents".into(), vec![cfg.num_queries, cfg.dim], t.data)?;
            }
            ["pulid_encoder", "proj_out"] => {
                if shape != [cfg.dim, cfg.output_dim] {
                    return Err(format!("pulid import: proj_out shape {shape:?}"));
                }
                let d = transpose(&t.data, cfg.dim, cfg.output_dim);
                put(&mut enc, "proj_out".into(), vec![cfg.output_dim, cfg.dim], d)?;
            }
            ["pulid_encoder", "id_embedding_mapping", ord, leaf] => {
                let l = mlp_leaf(ord)
                    .ok_or_else(|| format!("pulid import: unknown id_embedding_mapping ordinal {ord}"))?;
                put(&mut enc, format!("id_map.{l}.{leaf}"), shape, t.data)?;
            }
            ["pulid_encoder", m, ord, leaf] if m.starts_with("mapping_") => {
                let i = &m["mapping_".len()..];
                let l = mlp_leaf(ord)
                    .ok_or_else(|| format!("pulid import: unknown {m} ordinal {ord}"))?;
                put(&mut enc, format!("map{i}.{l}.{leaf}"), shape, t.data)?;
            }
            // `layers.{l}.0.*` = PerceiverAttention, `layers.{l}.1.*` = FeedForward
            ["pulid_encoder", "layers", l, "0", sub, leaf] => {
                put(&mut enc, format!("layers.{l}.attn.{sub}.{leaf}"), shape, t.data)?;
            }
            ["pulid_encoder", "layers", l, "1", ord, leaf] => {
                let f = ff_leaf(ord)
                    .ok_or_else(|| format!("pulid import: unknown FeedForward ordinal {ord}"))?;
                put(&mut enc, format!("layers.{l}.ff.{f}.{leaf}"), shape, t.data)?;
            }
            // ---- the injected cross-attentions ---------------------------
            ["pulid_ca", i, sub, leaf] => {
                let idx: usize =
                    i.parse().map_err(|_| format!("pulid import: bad pulid_ca index {i}"))?;
                num_ca = num_ca.max(idx + 1);
                put(&mut ca, format!("ca.{idx}.{sub}.{leaf}"), shape, t.data)?;
            }
            _ => return Err(format!("pulid import: unrecognised tensor {}", t.name)),
        }
    }

    validate(&enc, &cfg.encoder_manifest(), "encoder")?;
    validate(&ca, &cfg.ca_manifest(num_ca), "ca")?;
    Ok(PulidWeights { encoder: enc, ca, num_ca })
}

/// Read the released checkpoint from disk and import it.
pub fn read(path: &str, cfg: &PulidConfig) -> Result<PulidWeights, String> {
    import(checkpoint::safetensors::read(path)?, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(name: &str, shape: &[usize]) -> StTensor {
        StTensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            data: vec![0.0; shape.iter().product()],
        }
    }

    /// Every manifest name must be reachable from a source name — built by
    /// inverting the remap, so a typo on either side fails here rather than at
    /// a 1.1 GB load.
    fn synthetic_checkpoint(cfg: &PulidConfig) -> Vec<StTensor> {
        let (d, ff) = (cfg.dim, cfg.ff_hidden());
        let mut v = vec![
            st("pulid_encoder.latents", &[1, cfg.num_queries, d]),
            st("pulid_encoder.proj_out", &[d, cfg.output_dim]),
        ];
        let mut mlp = |pfx: String, k0: usize, n2: usize| {
            v.push(st(&format!("{pfx}.0.weight"), &[d, k0]));
            v.push(st(&format!("{pfx}.0.bias"), &[d]));
            v.push(st(&format!("{pfx}.1.weight"), &[d]));
            v.push(st(&format!("{pfx}.1.bias"), &[d]));
            v.push(st(&format!("{pfx}.3.weight"), &[d, d]));
            v.push(st(&format!("{pfx}.3.bias"), &[d]));
            v.push(st(&format!("{pfx}.4.weight"), &[d]));
            v.push(st(&format!("{pfx}.4.bias"), &[d]));
            v.push(st(&format!("{pfx}.6.weight"), &[n2, d]));
            v.push(st(&format!("{pfx}.6.bias"), &[n2]));
        };
        mlp("pulid_encoder.id_embedding_mapping".into(), cfg.id_cond_dim, cfg.num_id_token * d);
        for i in 0..cfg.scales {
            mlp(format!("pulid_encoder.mapping_{i}"), d, d);
        }
        let inner = cfg.inner_dim();
        for l in 0..cfg.depth {
            let b = format!("pulid_encoder.layers.{l}");
            v.push(st(&format!("{b}.0.norm1.weight"), &[d]));
            v.push(st(&format!("{b}.0.norm1.bias"), &[d]));
            v.push(st(&format!("{b}.0.norm2.weight"), &[d]));
            v.push(st(&format!("{b}.0.norm2.bias"), &[d]));
            v.push(st(&format!("{b}.0.to_q.weight"), &[inner, d]));
            v.push(st(&format!("{b}.0.to_kv.weight"), &[2 * inner, d]));
            v.push(st(&format!("{b}.0.to_out.weight"), &[d, inner]));
            v.push(st(&format!("{b}.1.0.weight"), &[d]));
            v.push(st(&format!("{b}.1.0.bias"), &[d]));
            v.push(st(&format!("{b}.1.1.weight"), &[ff, d]));
            v.push(st(&format!("{b}.1.3.weight"), &[d, ff]));
        }
        let (dm, kvd, ci) = (cfg.ca_dim, cfg.output_dim, cfg.ca_inner_dim());
        for i in 0..20 {
            let b = format!("pulid_ca.{i}");
            v.push(st(&format!("{b}.norm1.weight"), &[kvd]));
            v.push(st(&format!("{b}.norm1.bias"), &[kvd]));
            v.push(st(&format!("{b}.norm2.weight"), &[dm]));
            v.push(st(&format!("{b}.norm2.bias"), &[dm]));
            v.push(st(&format!("{b}.to_q.weight"), &[ci, dm]));
            v.push(st(&format!("{b}.to_kv.weight"), &[2 * ci, kvd]));
            v.push(st(&format!("{b}.to_out.weight"), &[dm, ci]));
        }
        v
    }

    #[test]
    fn two_way_coverage_over_a_synthetic_checkpoint() {
        let cfg = PulidConfig::v0_9_1();
        let src = synthetic_checkpoint(&cfg);
        assert_eq!(src.len(), 312, "v0.9.1 ships 312 tensors");
        let w = import(src, &cfg).unwrap();
        assert_eq!(w.num_ca, 20);
        assert_eq!(w.encoder.len(), cfg.encoder_manifest().len());
        assert_eq!(w.ca.len(), cfg.ca_manifest(20).len());
    }

    #[test]
    fn a_missing_tensor_is_an_error_by_name() {
        let cfg = PulidConfig::v0_9_1();
        let mut src = synthetic_checkpoint(&cfg);
        src.retain(|t| t.name != "pulid_encoder.layers.3.0.to_kv.weight");
        let e = import(src, &cfg).unwrap_err();
        assert!(e.contains("layers.3.attn.to_kv.weight"), "{e}");
    }

    #[test]
    fn an_unused_source_tensor_is_an_error() {
        let cfg = PulidConfig::v0_9_1();
        let mut src = synthetic_checkpoint(&cfg);
        src.push(st("pulid_encoder.layers.11.0.to_q.weight", &[1024, 1024]));
        let e = import(src, &cfg).unwrap_err();
        assert!(e.contains("unused source tensors"), "{e}");
    }

    #[test]
    fn proj_out_is_transposed_at_import() {
        let cfg = PulidConfig { dim: 2, output_dim: 3, ..PulidConfig::v0_9_1() };
        // [dim, output_dim] row-major -> [output_dim, dim]
        let src = [StTensor {
            name: "pulid_encoder.proj_out".into(),
            shape: vec![2, 3],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }];
        // import() validates the whole manifest, so drive `transpose` directly
        let _ = cfg;
        assert_eq!(transpose(&src[0].data, 2, 3), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
