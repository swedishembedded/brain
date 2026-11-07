// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The contract an engine meets to produce its own latent-sparse-attention
//! indexer, and the generic D-LSA tooling over it: capture the attention
//! inputs of a dense prefill, initialise an indexer (random, PCA), distil it
//! against the model's own attention, and evaluate it.
//!
//! Swedish Embedded AB implements long-context inference that keeps its
//! accuracy while its attention cost stops growing quadratically, for its
//! clients. If your team needs expertise in sparse attention, distillation
//! and long-context evaluation, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! # The contract
//!
//! [`AttentionCapture`] is what an engine says about itself and does for the
//! tooling: its [`IndexGeometry`] (whose `rope_base` is the rotation the index
//! undoes: the base for rotated heads, 0 for heads attention reads unrotated),
//! which layers can read sparsely ([`AttentionCapture::sparse_capable`]), a
//! capture of the queries and keys of those layers exactly as attention (and so
//! the index) reads them, the log-probability the model gives a text, and
//! attaching an indexer. Everything an indexer holds is restricted to the
//! sparse-capable layers: random, PCA and distilled indexers get `None` for the
//! rest ([`Indexer::random_for`], [`Indexer::pca_for`]).
//!
//! The tooling is generic over the contract and knows no model: the same
//! functions distil a Qwen3 indexer over all its layers and a Ripple indexer
//! over its sixteen global ones.

use super::distill::{retained_mass, retention, teacher_targets, CapturedLayer, Distiller, LayerSample};
use super::indexer::{IndexGeometry, Indexer, LayerProjection};
use super::space::index_inputs;
use super::LsaConfig;
use data::rng::Lcg;

/// One layer's attention inputs over a captured text, as attention reads them.
#[derive(Clone, Debug)]
pub struct LayerCapture {
    /// `[S, q_dim]` queries at the requested positions.
    pub q: Vec<f32>,
    /// `[T, kv_dim]` keys.
    pub k: Vec<f32>,
}

/// What an engine provides so its indexer can be built, distilled and
/// evaluated by the generic tooling.
pub trait AttentionCapture {
    /// The attention geometry an indexer for this engine carries. `rope_base`
    /// is the rotation the index undoes (0 for none).
    fn index_geometry(&self) -> IndexGeometry;

    /// Per layer, whether it can read sparsely. Only these layers hold
    /// projections in an indexer; a layer that attends a window by design does
    /// not.
    fn sparse_capable(&self) -> Vec<bool>;

    /// The longest text [`Self::capture_attention`] takes.
    fn max_capture_tokens(&self) -> usize;

    /// A dense prefill of `prompt`, in chunks, with sparse attention set aside
    /// (the teacher is the model's own attention), keeping per sparse-capable
    /// layer every position's keys (`[T, kv_dim]`) and the queries at
    /// `positions` (ascending; `[positions.len(), q_dim]`, so a long text does
    /// not hold every query), as the index reads them:
    /// rotated (the index un-rotates them) where the model rotates its heads,
    /// as they stand where it does not. `None` for the layers that are not
    /// sparse-capable. Does not touch the prefix cache.
    fn capture_attention(&mut self, prompt: &[u32], positions: &[u32]) -> Result<Vec<Option<LayerCapture>>, String>;

    /// The log-probability the model gives each next token of `prompt` at
    /// positions `from..len - 1`, through the engine's current attention.
    fn position_logprobs(&mut self, prompt: &[u32], from: usize) -> Vec<f32>;

    /// Attach `indexer` (replacing any attached one).
    fn set_lsa(&mut self, indexer: &Indexer) -> Result<(), String>;
}

impl Indexer {
    /// [`Indexer::random`] over the layers `capable` marks; the rest read densely.
    pub fn random_for(cfg: LsaConfig, geometry: IndexGeometry, seed: u64, capable: &[bool]) -> Result<Indexer, String> {
        check_capable(&geometry, capable)?;
        let mut ix = Indexer::random(cfg, geometry, seed)?;
        for (layer, &ok) in ix.layers.iter_mut().zip(capable) {
            if !ok {
                *layer = None;
            }
        }
        ix.validate()?;
        Ok(ix)
    }

    /// [`Indexer::pca`] over the layers `capable` marks, from `keys_per_layer`
    /// (empty for a layer that is not capable).
    pub fn pca_for(cfg: LsaConfig, geometry: IndexGeometry, keys_per_layer: &[Vec<f32>], capable: &[bool]) -> Result<Indexer, String> {
        check_capable(&geometry, capable)?;
        let dim = geometry.input_dim();
        let filler: Vec<f32> = (0..(2 * cfg.index_dim + 4) * dim).map(|i| ((i as u64).wrapping_mul(2_654_435_761) % 1000) as f32 / 1000.0 - 0.5).collect();
        // PCA of a layer that is dropped afterwards needs some rows to run on.
        let keys: Vec<Vec<f32>> = keys_per_layer.iter().zip(capable).map(|(k, &ok)| if ok { k.clone() } else { filler.clone() }).collect();
        let mut ix = Indexer::pca(cfg, geometry, &keys)?;
        for (layer, &ok) in ix.layers.iter_mut().zip(capable) {
            if !ok {
                *layer = None;
            }
        }
        ix.validate()?;
        Ok(ix)
    }
}

fn check_capable(geometry: &IndexGeometry, capable: &[bool]) -> Result<(), String> {
    if capable.len() != geometry.n_layers {
        return Err(format!("LSA: {} sparse-capable flags for a {}-layer model", capable.len(), geometry.n_layers));
    }
    if !capable.iter().any(|c| *c) {
        return Err("LSA: the model has no layer that can read sparsely".into());
    }
    Ok(())
}

/// Per sparse-capable layer, a captured text's index inputs and the sampled
/// queries.
pub struct Captured {
    /// Per layer `[T, kv_dim]` keys as attention reads them.
    pub k: Vec<Option<Vec<f32>>>,
    /// Per layer `[S, q_dim]` queries of the sampled positions.
    pub q: Vec<Option<Vec<f32>>>,
    /// Per layer `[T, input_dim]` index inputs of every key.
    pub key_inputs: Vec<Option<Vec<f32>>>,
    /// Per layer `[S, input_dim]` index inputs of the sampled queries.
    pub query_inputs: Vec<Option<Vec<f32>>>,
}

/// Capture `doc` and derive the index inputs of every key and of the queries
/// at `positions`.
pub fn capture<E: AttentionCapture>(e: &mut E, doc: &[u32], positions: &[u32]) -> Result<Captured, String> {
    let g = e.index_geometry();
    let layers = e.capture_attention(doc, positions)?;
    if layers.len() != g.n_layers {
        return Err(format!("LSA capture: {} layers captured for a {}-layer model", layers.len(), g.n_layers));
    }
    let all: Vec<u32> = (0..doc.len() as u32).collect();
    let mut out = Captured { k: Vec::new(), q: Vec::new(), key_inputs: Vec::new(), query_inputs: Vec::new() };
    for layer in layers {
        match layer {
            Some(LayerCapture { q: qs, k }) => {
                out.key_inputs.push(Some(index_inputs(&k, &all, g.n_kv_heads, g.n_kv_heads, g.head_dim, g.rope_base)));
                out.query_inputs.push(Some(index_inputs(&qs, positions, g.n_heads, g.n_kv_heads, g.head_dim, g.rope_base)));
                out.q.push(Some(qs));
                out.k.push(Some(k));
            }
            None => {
                out.key_inputs.push(None);
                out.query_inputs.push(None);
                out.q.push(None);
                out.k.push(None);
            }
        }
    }
    Ok(out)
}

/// Query positions to sample from a text of `tokens`: late enough that the
/// read has long-range keys (past the window, a cluster block and the sinks).
pub fn sample_positions(rng: &mut Lcg, tokens: usize, cfg: &LsaConfig, n: usize) -> Result<Vec<u32>, String> {
    let lo = (cfg.window + cfg.cluster_block + cfg.sink) as u32;
    if tokens as u32 <= lo {
        return Err(format!("LSA: a text of {tokens} tokens is not past the window, cluster block and sinks ({lo}); no query has a long-range key to select"));
    }
    let mut p: Vec<u32> = (0..n).map(|_| lo + rng.next_u32() % (tokens as u32 - lo)).collect();
    p.sort_unstable();
    p.dedup();
    Ok(p)
}

/// A PCA indexer for the engine's sparse-capable layers, from the keys of
/// `docs`.
pub fn pca_indexer<E: AttentionCapture>(e: &mut E, cfg: LsaConfig, docs: &[Vec<u32>]) -> Result<Indexer, String> {
    let g = e.index_geometry();
    let capable = e.sparse_capable();
    let mut keys: Vec<Vec<f32>> = vec![Vec::new(); g.n_layers];
    for doc in docs {
        for (acc, k) in keys.iter_mut().zip(capture(e, doc, &[])?.key_inputs) {
            if let Some(k) = k {
                acc.extend(k);
            }
        }
    }
    Indexer::pca_for(cfg, g, &keys, &capable)
}

/// Settings of [`distill`].
#[derive(Clone, Copy, Debug)]
pub struct DistillOptions {
    pub epochs: usize,
    /// Query positions sampled per text and layer.
    pub queries: usize,
    pub lr: f32,
    /// Initial temperature of the index scores.
    pub tau: f32,
    /// Keys of the teacher's distribution each query keeps (the paper's 2048).
    pub teacher_keep: usize,
}

impl Default for DistillOptions {
    fn default() -> DistillOptions {
        DistillOptions { epochs: 3, queries: 64, lr: 1e-3, tau: 0.1, teacher_keep: 2048 }
    }
}

/// Distil `init` against the engine's attention over `docs`: per text and
/// epoch, capture, build each sparse layer's teacher at sampled positions and
/// take one Adam step. `progress(epoch, doc, mean_kl)` is called after each
/// text.
pub fn distill<E: AttentionCapture>(e: &mut E, init: &Indexer, docs: &[Vec<u32>], opts: &DistillOptions, rng: &mut Lcg, mut progress: impl FnMut(usize, usize, f64)) -> Result<Distiller, String> {
    let g = e.index_geometry();
    let cfg = init.cfg;
    let mut d = Distiller::new(init, opts.tau)?;
    for epoch in 0..opts.epochs {
        for (i, doc) in docs.iter().enumerate() {
            let positions = sample_positions(rng, doc.len(), &cfg, opts.queries)?;
            let c = capture(e, doc, &positions)?;
            let samples: Vec<Option<LayerSample>> = (0..g.n_layers)
                .map(|l| {
                    init.layers[l].as_ref()?;
                    let (q, k) = (c.q[l].as_ref()?, c.k[l].as_ref()?);
                    let targets = teacher_targets(q, k, &positions, &g, &cfg, opts.teacher_keep);
                    Some(LayerSample { keys: c.key_inputs[l].clone()?, queries: c.query_inputs[l].clone()?, positions: positions.clone(), targets })
                })
                .collect();
            let stats = d.step(&samples, opts.lr);
            let sparse: Vec<f64> = stats.iter().zip(&init.layers).filter(|(_, l)| l.is_some()).map(|(s, _)| s.kl).collect();
            progress(epoch, i, sparse.iter().sum::<f64>() / sparse.len().max(1) as f64);
        }
    }
    Ok(d)
}

/// The share of the model's dense attention each indexer's selection reads,
/// averaged over held-out texts, sampled queries and sparse layers, against
/// the window and sinks alone.
pub struct Retention {
    /// Window and sinks only.
    pub local: f64,
    /// One share per indexer, in order.
    pub indexers: Vec<f64>,
}

/// Measure [`Retention`] of `indexers` over `docs`.
pub fn retention_report<E: AttentionCapture>(e: &mut E, docs: &[Vec<u32>], indexers: &[&Indexer], cfg: &LsaConfig, queries: usize, rng: &mut Lcg) -> Result<Retention, String> {
    let g = e.index_geometry();
    let mut local = Vec::new();
    let mut per: Vec<Vec<f64>> = vec![Vec::new(); indexers.len()];
    for doc in docs {
        let positions = sample_positions(rng, doc.len(), cfg, queries)?;
        let c = capture(e, doc, &positions)?;
        let window_only: Vec<Vec<u32>> = positions
            .iter()
            .map(|&p| {
                let ws = (p as usize + 1).saturating_sub(cfg.window);
                (ws as u32..=p).chain(0..cfg.sink.min(ws) as u32).collect()
            })
            .collect();
        for l in 0..g.n_layers {
            if let (Some(q), Some(k)) = (&c.q[l], &c.k[l]) {
                local.extend(retained_mass(q, k, &positions, &g, &window_only));
            }
        }
        for (i, ix) in indexers.iter().enumerate() {
            let layers: Vec<CapturedLayer> = (0..g.n_layers)
                .map(|l| match (&c.key_inputs[l], &c.query_inputs[l], &c.q[l], &c.k[l]) {
                    (Some(ki), Some(qi), Some(q), Some(k)) => CapturedLayer { key_inputs: ki, query_inputs: qi, q, k },
                    _ => CapturedLayer { key_inputs: &[], query_inputs: &[], q: &[], k: &[] },
                })
                .collect();
            per[i].extend(retention(ix, &layers, &positions)?.into_iter().flatten());
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    Ok(Retention { local: mean(&local), indexers: per.iter().map(|v| mean(v)).collect() })
}

/// Mean next-token loss over the last `tail` positions of each text, through
/// the engine's current attention.
pub fn tail_loss<E: AttentionCapture>(e: &mut E, docs: &[Vec<u32>], tail: usize) -> f64 {
    let lp: Vec<f32> = docs.iter().flat_map(|d| e.position_logprobs(d, d.len().saturating_sub(1 + tail))).collect();
    -lp.iter().map(|&x| x as f64).sum::<f64>() / lp.len().max(1) as f64
}

/// A [`LayerProjection`] is a layer's projections; re-exported for tools that
/// inspect an indexer's sparse layers.
pub type Projection = LayerProjection;
