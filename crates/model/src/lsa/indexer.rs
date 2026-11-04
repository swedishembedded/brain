// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The indexer of a model: per attention layer, the projections that map
//! head-summed, un-rotated attention heads into the index space, with the
//! configuration and the model geometry they are only valid for.
//!
//! An indexer is a model-attached artifact (a safetensors file): the base
//! model's weights are never touched. It comes from one of three places -
//! a seeded random projection (training-free, data-free), the principal
//! directions of the model's own keys (training-free, from a calibration
//! pass), or distillation against the model's attention (D-LSA).
//!
//! Index inputs are `n_kv_heads * head_dim` wide: a key's KV heads, or a
//! query's heads summed over each KV group. Both are un-rotated, so the
//! index sees content, never position (the window carries recency).

use super::LsaConfig;
use serde::{Deserialize, Serialize};

/// The attention geometry an indexer was built for.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexGeometry {
    /// Decoder layers.
    pub n_layers: usize,
    /// Query heads.
    pub n_heads: usize,
    /// Key/value heads.
    pub n_kv_heads: usize,
    /// Channels per head.
    pub head_dim: usize,
    /// Half-split RoPE base of the heads (0 for none).
    pub rope_base: f32,
}

impl IndexGeometry {
    /// Width of an index input: `n_kv_heads * head_dim`.
    pub fn input_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

/// One layer's projections, each `[input_dim, index_dim]` row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerProjection {
    /// Projection of head-summed queries.
    pub wq: Vec<f32>,
    /// Projection of keys.
    pub wk: Vec<f32>,
}

/// A model's indexer: configuration, geometry, and the projections of every
/// layer that reads sparsely (`None` keeps that layer dense).
#[derive(Clone, Debug, PartialEq)]
pub struct Indexer {
    /// Index configuration.
    pub cfg: LsaConfig,
    /// What the projections were built for.
    pub geometry: IndexGeometry,
    /// Per layer.
    pub layers: Vec<Option<LayerProjection>>,
    /// Where the projections came from, for the record (`random`, `pca`,
    /// `distilled`, ...).
    pub origin: String,
}

/// The artifact header, validated field by field on load.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    format: String,
    origin: String,
    geometry: IndexGeometry,
    config: ConfigHeader,
    sparse_layers: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigHeader {
    index_dim: usize,
    pool_size: usize,
    bucket_size: usize,
    bucket_cap: usize,
    cluster_block: usize,
    top_p: usize,
    top_k: usize,
    sink: usize,
    window: usize,
    early: Option<(usize, usize, usize)>,
}

const FORMAT: &str = "brain.lsa.indexer.v1";

impl Indexer {
    /// Refuse an indexer whose projections do not match its geometry and
    /// configuration.
    pub fn validate(&self) -> Result<(), String> {
        self.cfg.validate()?;
        let g = &self.geometry;
        if g.n_kv_heads == 0 || !g.n_heads.is_multiple_of(g.n_kv_heads) || g.head_dim == 0 {
            return Err(format!("LSA indexer: {} heads over {} KV heads of {} channels is not an attention layout", g.n_heads, g.n_kv_heads, g.head_dim));
        }
        if self.layers.len() != g.n_layers {
            return Err(format!("LSA indexer: {} layer entries for a {}-layer model", self.layers.len(), g.n_layers));
        }
        let want = g.input_dim() * self.cfg.index_dim;
        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(p) = layer {
                if p.wq.len() != want || p.wk.len() != want {
                    return Err(format!("LSA indexer layer {l}: projections of {} and {} values, {want} expected", p.wq.len(), p.wk.len()));
                }
                if p.wq.iter().chain(&p.wk).any(|x| !x.is_finite()) {
                    return Err(format!("LSA indexer layer {l}: a projection holds a non-finite value"));
                }
            }
        }
        if self.layers.iter().all(Option::is_none) {
            return Err("LSA indexer: no layer reads sparsely".into());
        }
        Ok(())
    }

    /// Refuse to attach to a model whose attention differs from the one the
    /// indexer was built for: projections of another model select nothing
    /// meaningful.
    pub fn check_model(&self, model: &IndexGeometry) -> Result<(), String> {
        if &self.geometry != model {
            return Err(format!("LSA indexer built for {:?} cannot index a model with {:?}", self.geometry, model));
        }
        Ok(())
    }

    /// A seeded random projection for every layer: dot products of index
    /// vectors approximate those of the head-summed inputs
    /// (Johnson-Lindenstrauss), with no data and no training.
    pub fn random(cfg: LsaConfig, geometry: IndexGeometry, seed: u64) -> Result<Indexer, String> {
        let mut rng = data::rng::Lcg::new(seed);
        let n = geometry.input_dim() * cfg.index_dim;
        let scale = 1.0 / (cfg.index_dim as f32).sqrt();
        let layers = (0..geometry.n_layers)
            .map(|_| {
                let w: Vec<f32> = (0..n).map(|_| gaussian(&mut rng) * scale).collect();
                Some(LayerProjection { wq: w.clone(), wk: w })
            })
            .collect();
        let indexer = Indexer { cfg, geometry, layers, origin: format!("random(seed={seed})") };
        indexer.validate()?;
        Ok(indexer)
    }

    /// The training-free PCA indexer: per layer, the `index_dim` leading
    /// principal directions of that layer's index inputs (un-rotated keys,
    /// `[rows, input_dim]` each) serve as both projections.
    pub fn pca(cfg: LsaConfig, geometry: IndexGeometry, keys_per_layer: &[Vec<f32>]) -> Result<Indexer, String> {
        if keys_per_layer.len() != geometry.n_layers {
            return Err(format!("LSA PCA: keys of {} layers for a {}-layer model", keys_per_layer.len(), geometry.n_layers));
        }
        let dim = geometry.input_dim();
        let mut layers = vec![None; geometry.n_layers];
        backend_cpu::par::chunks_mut(&mut layers, 1, |l, slot| {
            slot[0] = Some(pca_layer(&keys_per_layer[l], dim, cfg.index_dim).map(|w| LayerProjection { wq: w.clone(), wk: w }));
        });
        let layers = layers.into_iter().map(|l| l.expect("every layer was computed").map(Some)).collect::<Result<Vec<_>, _>>()?;
        let indexer = Indexer { cfg, geometry, layers, origin: "pca".into() };
        indexer.validate()?;
        Ok(indexer)
    }

    /// Write the indexer to `path` (safetensors).
    pub fn save(&self, path: &str) -> Result<(), String> {
        self.validate()?;
        let c = &self.cfg;
        let header = Header {
            format: FORMAT.into(),
            origin: self.origin.clone(),
            geometry: self.geometry,
            config: ConfigHeader {
                index_dim: c.index_dim,
                pool_size: c.pool_size,
                bucket_size: c.bucket_size,
                bucket_cap: c.bucket_cap,
                cluster_block: c.cluster_block,
                top_p: c.top_p,
                top_k: c.top_k,
                sink: c.sink,
                window: c.window,
                early: c.early.map(|e| (e.position, e.top_k, e.top_p)),
            },
            sparse_layers: self.layers.iter().enumerate().filter(|(_, l)| l.is_some()).map(|(i, _)| i).collect(),
        };
        let shape = vec![self.geometry.input_dim() as u64, c.index_dim as u64];
        let mut tensors = Vec::new();
        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(p) = layer {
                tensors.push((format!("layers.{l}.wq"), shape.clone(), p.wq.clone()));
                tensors.push((format!("layers.{l}.wk"), shape.clone(), p.wk.clone()));
            }
        }
        let config = serde_json::to_value(&header).map_err(|e| e.to_string())?;
        checkpoint::st::save_safetensors(path, &tensors, &config, None).map_err(|e| format!("cannot write LSA indexer {path}: {e}"))
    }

    /// Read an indexer from `path`, validating every field.
    pub fn load(path: &str) -> Result<Indexer, String> {
        let st = checkpoint::st::load_safetensors(path).map_err(|e| format!("cannot read LSA indexer {path}: {e}"))?;
        let header: Header = serde_json::from_value(st.config()).map_err(|e| format!("LSA indexer {path}: bad header: {e}"))?;
        if header.format != FORMAT {
            return Err(format!("LSA indexer {path}: format {:?}, expected {FORMAT:?}", header.format));
        }
        let c = header.config;
        let cfg = LsaConfig {
            index_dim: c.index_dim,
            pool_size: c.pool_size,
            bucket_size: c.bucket_size,
            bucket_cap: c.bucket_cap,
            cluster_block: c.cluster_block,
            top_p: c.top_p,
            top_k: c.top_k,
            sink: c.sink,
            window: c.window,
            early: c.early.map(|(position, top_k, top_p)| super::EarlyBudget { position, top_k, top_p }),
        };
        let mut layers = vec![None; header.geometry.n_layers];
        let mut tensors = st.tensors;
        for &l in &header.sparse_layers {
            if l >= layers.len() {
                return Err(format!("LSA indexer {path}: sparse layer {l} past the model's {}", layers.len()));
            }
            let mut take = |name: String| tensors.remove(&name).ok_or_else(|| format!("LSA indexer {path}: missing {name}"));
            layers[l] = Some(LayerProjection { wq: take(format!("layers.{l}.wq"))?, wk: take(format!("layers.{l}.wk"))? });
        }
        if let Some(name) = tensors.keys().next() {
            return Err(format!("LSA indexer {path}: tensor {name} belongs to no listed layer"));
        }
        let indexer = Indexer { cfg, geometry: header.geometry, layers, origin: header.origin };
        indexer.validate()?;
        Ok(indexer)
    }
}

fn gaussian(rng: &mut data::rng::Lcg) -> f32 {
    // Box-Muller from two uniforms in (0, 1].
    let u1 = 1.0 - rng.unit() as f64;
    let u2 = rng.unit() as f64;
    ((-2.0 * u1.max(1e-300).ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

/// `[dim, k]`: the `k` leading eigenvectors of the uncentred second moment of
/// `rows` (`[n, dim]`) as columns. Uncentred, because the index scores raw dot
/// products, not deviations from a mean.
fn pca_layer(rows: &[f32], dim: usize, k: usize) -> Result<Vec<f32>, String> {
    let n = rows.len() / dim;
    if n < k || !rows.len().is_multiple_of(dim) {
        return Err(format!("LSA PCA: {n} key rows of width {dim} cannot give {k} directions"));
    }
    let mut m = vec![0f64; dim * dim];
    for row in rows.chunks_exact(dim) {
        for i in 0..dim {
            let xi = row[i] as f64;
            for j in 0..=i {
                m[i * dim + j] += xi * row[j] as f64;
            }
        }
    }
    let (_, vectors) = crate::eigen::symmetric_eigen(&m, dim);
    let mut w = vec![0f32; dim * k];
    for c in 0..k {
        let v = &vectors[dim - 1 - c];
        for i in 0..dim {
            w[i * k + c] = v[i] as f32;
        }
    }
    Ok(w)
}
