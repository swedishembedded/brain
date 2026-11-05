// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The per-block AdaLN modulation, cached by timestep value - what lets the
//! H3 DiT run without holding `adaln_proj` at all.
//!
//! `adaln_proj` is one `Linear(time_embed_dim, 6*hidden*MODALITY_NUM)` per
//! block: 260M params/block x 50 blocks = 13.0B of the 33B total (real
//! shard header). Its input is `silu(temb(t))`, a function of the scalar
//! timestep `t` alone, so its output for a given `t` is a constant
//! `[MODALITY_NUM*6*hidden]` row per block. A generation only ever evaluates
//! the finite set of timesteps its schedule produces (the video and audio
//! sigmas, the keyframe noise floor, the clean-condition `1.0`), so the whole
//! of `adaln_proj` collapses to `values x blocks` rows computed once, and the
//! weight matrix itself never has to be resident: [`AdalnCache::prepare`]
//! reads each block's `adaln_proj` from the checkpoint, evaluates it at every
//! value not yet cached, and drops it before reading the next block.
//!
//! At the real config one value costs `50 * 18 * 5376 * 4` bytes = 19.4 MB;
//! a 50-step generation touches ~150 values (2.9 GB), against 52 GB of fp32
//! `adaln_proj`. The cache is bounded ([`AdalnCache::new`]'s capacity): past
//! it, values no live [`AdalnPlan`] holds are evicted, so the bound is soft
//! only while concurrent requests between them pin more than it.
//!
//! Rows are keyed by the timestep's exact `f32` bits, never by step index or
//! modality: the forward evaluates every modality at every distinct timestep
//! in the packed sequence, and which rows a modality reads at which value is
//! the layout's business (`crate::pipeline::build_row_timesteps`), not this
//! cache's. A value missing from a plan is an error naming it, never a
//! nearest-neighbour substitute.
//!
//! [`AdalnPlan`] is the request-scoped view a generation holds: `Arc`s of
//! exactly the rows it needs. Eviction never drops a row a live plan holds,
//! so while a request keeps its plan, every forward it runs finds its rows
//! in the cache - a concurrent request cannot pull them out from under it.
//!
//! Swedish Embedded AB implements memory-bounded inference for large
//! diffusion transformers for its clients. If your team needs expertise in
//! fitting oversized models onto fixed hardware, you can procure our services
//! by sending an email to info@swedishembedded.com.

use std::collections::HashMap;
use std::sync::Arc;

use model::hostmath::{linear_rows, silu_slice};

use crate::config::{H3TransformerConfig, MODALITY_NUM};
use crate::model::TimeWeights;

/// One timestep value's modulation for every block: `rows[block]` is
/// `adaln_proj(silu(temb(t)))`, `[MODALITY_NUM*6*hidden]`, in the
/// projection's own output order (modality-major, then the six AdaLN-Zero
/// parameters, then `hidden`).
pub type ValueRows = Arc<Vec<Vec<f32>>>;

/// The shared, bounded cache of [`ValueRows`] by timestep bits.
pub struct AdalnCache {
    capacity: usize,
    rows: HashMap<u32, ValueRows>,
}

/// The rows one forward (or one whole generation) reads, aligned to
/// `values`.
#[derive(Clone)]
pub struct AdalnPlan {
    values: Vec<u32>,
    rows: Vec<ValueRows>,
}

impl AdalnPlan {
    /// Block `block`'s row for each of `timesteps`, in that order - what
    /// `crate::block::block_forward` builds its per-forward tables from.
    pub fn block_rows(&self, block: usize, timesteps: &[f32]) -> Result<Vec<&[f32]>, String> {
        timesteps
            .iter()
            .map(|t| {
                let i = self.values.iter().position(|&v| v == t.to_bits()).ok_or_else(|| format!("AdaLN plan has no row for timestep {t}: prepare it with every value the forward will see"))?;
                Ok(self.rows[i][block].as_slice())
            })
            .collect()
    }
}

/// How many timestep values a resident model keeps by default: ~9.9 GB of
/// rows at the real config, enough for several distinct step counts.
pub const DEFAULT_CAPACITY: usize = 512;

impl AdalnCache {
    pub fn new(capacity: usize) -> AdalnCache {
        assert!(capacity > 0, "AdalnCache capacity must be positive");
        AdalnCache { capacity, rows: HashMap::new() }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// A plan covering `values`, computing whatever is missing from
    /// `tensors`' per-block `adaln_proj` (each block's weight read once,
    /// dropped before the next). Errors when `values` alone exceed the
    /// capacity, or a tensor is missing or mis-sized.
    pub fn prepare(&mut self, tensors: &dyn checkpoint::TensorSource, cfg: &H3TransformerConfig, time: &TimeWeights, values: &[f32]) -> Result<AdalnPlan, String> {
        let mut wanted: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
        wanted.sort_unstable();
        wanted.dedup();
        if let Some(bad) = wanted.iter().map(|&b| f32::from_bits(b)).find(|v| !v.is_finite()) {
            return Err(format!("AdaLN cache: timestep {bad} is not finite"));
        }
        if wanted.len() > self.capacity {
            return Err(format!("AdaLN cache: a request needs {} distinct timesteps, more than the cache's capacity of {}", wanted.len(), self.capacity));
        }
        let missing: Vec<u32> = wanted.iter().copied().filter(|b| !self.rows.contains_key(b)).collect();
        if !missing.is_empty() {
            let overflow = (self.rows.len() + missing.len()).saturating_sub(self.capacity);
            if overflow > 0 {
                let evict: Vec<u32> = self.rows.iter().filter(|(b, rows)| wanted.binary_search(b).is_err() && Arc::strong_count(rows) == 1).map(|(b, _)| *b).take(overflow).collect();
                for b in evict {
                    self.rows.remove(&b);
                }
            }
            for (bits, rows) in missing.iter().zip(compute_rows(tensors, cfg, time, &missing)?) {
                self.rows.insert(*bits, Arc::new(rows));
            }
        }
        let rows = wanted.iter().map(|b| Arc::clone(&self.rows[b])).collect();
        Ok(AdalnPlan { values: wanted, rows })
    }

    /// A plan over `values` from what is already cached, computing nothing -
    /// what a model that does not hold the checkpoint reads. Every value must
    /// have been prepared (and its plan still held) by the caller.
    pub fn cached(&self, values: &[f32]) -> Result<AdalnPlan, String> {
        let mut wanted: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let rows = wanted
            .iter()
            .map(|b| self.rows.get(b).map(Arc::clone).ok_or_else(|| format!("no AdaLN rows prepared for timestep {}: prepare the schedule before the first forward", f32::from_bits(*b))))
            .collect::<Result<_, _>>()?;
        Ok(AdalnPlan { values: wanted, rows })
    }
}

/// `adaln_proj(silu(temb(v)))` for every block and every value in `values`
/// (as bits): result `[value][block][MODALITY_NUM*6*hidden]`. The
/// projection is streamed in row chunks, never materialized: a real block's
/// weight is 2 GB of fp32.
fn compute_rows(tensors: &dyn checkpoint::TensorSource, cfg: &H3TransformerConfig, time: &TimeWeights, values: &[u32]) -> Result<Vec<Vec<Vec<f32>>>, String> {
    let te = cfg.time_embed_dim as usize;
    let width = (MODALITY_NUM * 6 * cfg.hidden_size) as usize;
    let mut x = Vec::with_capacity(values.len() * te);
    for &b in values {
        x.extend(silu_slice(&time.temb(cfg, f32::from_bits(b))));
    }
    let mut out: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(cfg.num_layers as usize); values.len()];
    for block in 0..cfg.num_layers as usize {
        let p = format!("transformer_blocks.{block}.adaln_proj.linear");
        let mut y = project_streamed(tensors, &format!("{p}.weight"), &x, values.len(), te, width)?;
        let bias = read_small(tensors, &format!("{p}.bias"), width)?;
        for (vi, row) in y.iter_mut().enumerate() {
            for (a, c) in row.iter_mut().zip(&bias) {
                *a += c;
            }
            out[vi].push(std::mem::take(row));
        }
    }
    Ok(out)
}

/// Rows of weight per streamed chunk: ~64 MB of fp32 at the real config.
const ROWS_PER_CHUNK: usize = 3072;

/// `x @ W^T` for `x: [n, inn]` and the `[out, inn]` weight `name`, read in
/// chunks of whole rows (a chunk boundary that splits a row is carried into
/// the next chunk). Returns `[n][out]`.
fn project_streamed(tensors: &dyn checkpoint::TensorSource, name: &str, x: &[f32], n: usize, inn: usize, out: usize) -> Result<Vec<Vec<f32>>, String> {
    let mut y = vec![vec![0f32; out]; n];
    let mut pending: Vec<f32> = Vec::with_capacity(ROWS_PER_CHUNK * inn);
    let mut next_row = 0usize;
    let mut overflow = false;
    let found = tensors.with_tensor_chunks(name, ROWS_PER_CHUNK * inn, &mut |_, chunk| {
        pending.extend_from_slice(chunk);
        let rows = pending.len() / inn;
        if rows == 0 {
            return;
        }
        if next_row + rows > out {
            overflow = true;
            return;
        }
        let part = linear_rows(x, &pending[..rows * inn], n, inn, rows);
        for (vi, yrow) in y.iter_mut().enumerate() {
            yrow[next_row..next_row + rows].copy_from_slice(&part[vi * rows..(vi + 1) * rows]);
        }
        next_row += rows;
        pending.drain(..rows * inn);
    });
    if !found {
        return Err(format!("AdaLN cache: missing tensor {name:?}"));
    }
    tensors.advise_drop(name);
    if overflow || next_row != out || !pending.is_empty() {
        return Err(format!("AdaLN cache: {name} is not a [{out}, {inn}] matrix"));
    }
    Ok(y)
}

/// A small tensor (a bias), read through the bounded chunk reader.
fn read_small(tensors: &dyn checkpoint::TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let mut v = Vec::with_capacity(len);
    if !tensors.with_tensor_chunks(name, len.max(1), &mut |_, c| v.extend_from_slice(c)) {
        return Err(format!("AdaLN cache: missing tensor {name:?}"));
    }
    tensors.advise_drop(name);
    if v.len() != len {
        return Err(format!("AdaLN cache: {name} has {} elements, expected {len}", v.len()));
    }
    Ok(v)
}

/// A `Tensors` source that records every tensor name it hands out - how a
/// test proves which weights a load or forward actually reads.
#[cfg(test)]
pub(crate) struct Counting {
    pub inner: vae::blocks::Tensors,
    pub reads: std::cell::RefCell<Vec<String>>,
}

#[cfg(test)]
impl Counting {
    pub fn new(inner: vae::blocks::Tensors) -> Counting {
        Counting { inner, reads: std::cell::RefCell::new(Vec::new()) }
    }

    /// The names read so far that contain `needle`, and forget them all.
    pub fn take_matching(&self, needle: &str) -> Vec<String> {
        std::mem::take(&mut *self.reads.borrow_mut()).into_iter().filter(|n| n.contains(needle)).collect()
    }
}

#[cfg(test)]
impl checkpoint::TensorSource for Counting {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        self.reads.borrow_mut().push(name.to_string());
        self.inner.with_tensor(name, f)
    }

    fn with_tensor_range(&self, name: &str, start: usize, len: usize, f: &mut dyn FnMut(&[f32])) -> bool {
        self.reads.borrow_mut().push(name.to_string());
        self.inner.with_tensor_range(name, start, len, f)
    }

    fn numel(&self, name: &str) -> Option<usize> {
        self.inner.numel(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use vae::blocks::Tensors;

    fn tensors(cfg: &H3TransformerConfig, seed: u64) -> Tensors {
        let mut rng = data::rng::Lcg::new(seed);
        let (hidden, te, thd, freq) = (cfg.hidden_size as usize, cfg.time_embed_dim as usize, cfg.time_embed_hidden_dim as usize, cfg.freq_dim as usize);
        let width = 6 * hidden * MODALITY_NUM as usize;
        let mut t = Tensors::new();
        t.insert("time_embedder.linear_1.weight".into(), (vec![thd, freq], rng.vec_scaled(thd * freq, 0.2)));
        t.insert("time_embedder.linear_1.bias".into(), (vec![thd], rng.vec_scaled(thd, 0.2)));
        t.insert("time_embedder.linear_2.weight".into(), (vec![te, thd], rng.vec_scaled(te * thd, 0.2)));
        t.insert("time_embedder.linear_2.bias".into(), (vec![te], rng.vec_scaled(te, 0.2)));
        for i in 0..cfg.num_layers {
            t.insert(format!("transformer_blocks.{i}.adaln_proj.linear.weight"), (vec![width, te], rng.vec_scaled(width * te, 0.2)));
            t.insert(format!("transformer_blocks.{i}.adaln_proj.linear.bias"), (vec![width], rng.vec_scaled(width, 0.2)));
        }
        t
    }

    /// `adaln_proj`'s forward at one timestep, by plain nested loops in f64
    /// - independent of `linear_rows` and of the cache's batching.
    fn reference_row(cfg: &H3TransformerConfig, t: &Tensors, block: u32, timestep: f32) -> Vec<f32> {
        let get = |n: &str| &t[n].1;
        let temb = crate::model::temb_row(cfg, timestep, get("time_embedder.linear_1.weight"), get("time_embedder.linear_1.bias"), get("time_embedder.linear_2.weight"), get("time_embedder.linear_2.bias"));
        let x = silu_slice(&temb);
        let w = get(&format!("transformer_blocks.{block}.adaln_proj.linear.weight"));
        let b = get(&format!("transformer_blocks.{block}.adaln_proj.linear.bias"));
        b.iter().enumerate().map(|(o, bo)| (x.iter().zip(&w[o * x.len()..(o + 1) * x.len()]).map(|(a, c)| *a as f64 * *c as f64).sum::<f64>() + *bo as f64) as f32).collect()
    }

    /// Every cached row is `adaln_proj`'s own output at that exact
    /// timestep, for every block, and a plan hands back rows in the order
    /// the forward asks for them - including values given in any order or
    /// repeated.
    #[test]
    fn cached_rows_are_the_projection_evaluated_at_each_timestep() {
        let cfg = H3TransformerConfig::tiny();
        let t = tensors(&cfg, 7);
        let time = TimeWeights::load(&t);
        let values = [0.913f32, 0.25, 1.0, 0.0, 0.25];
        let plan = AdalnCache::new(16).prepare(&t, &cfg, &time, &values).unwrap();
        for block in 0..cfg.num_layers {
            let got = plan.block_rows(block as usize, &[1.0, 0.913, 0.0]).unwrap();
            for (row, ts) in got.iter().zip([1.0f32, 0.913, 0.0]) {
                let want = reference_row(&cfg, &t, block, ts);
                let worst = row.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                assert!(worst < 1e-5, "block {block} t={ts}: max abs {worst}");
            }
        }
    }

    /// The whole point: after one `prepare` over a generation's timesteps,
    /// a second `prepare` for any subset reads no tensor at all, and the
    /// first one reads each block's `adaln_proj` exactly once no matter how
    /// many values it covers.
    #[test]
    fn a_prepared_schedule_reads_each_projection_once_and_never_again() {
        let cfg = H3TransformerConfig::tiny();
        let src = Counting::new(tensors(&cfg, 3));
        let time = TimeWeights::load(&src);
        src.reads.borrow_mut().clear();
        let mut cache = AdalnCache::new(64);
        let values: Vec<f32> = (0..20).map(|i| i as f32 / 19.0).collect();
        cache.prepare(&src, &cfg, &time, &values).unwrap();
        let reads = src.reads.borrow().clone();
        assert_eq!(reads.len(), 2 * cfg.num_layers as usize, "one weight + one bias read per block: {reads:?}");
        src.reads.borrow_mut().clear();
        cache.prepare(&src, &cfg, &time, &values[3..9]).unwrap();
        assert!(src.reads.borrow().is_empty(), "a cached value must not re-read the checkpoint");
    }

    /// A value the plan was not prepared with is an error naming it, never a
    /// substitute row.
    #[test]
    fn an_unprepared_timestep_is_an_error() {
        let cfg = H3TransformerConfig::tiny();
        let t = tensors(&cfg, 5);
        let plan = AdalnCache::new(4).prepare(&t, &cfg, &TimeWeights::load(&t), &[0.5]).unwrap();
        let err = plan.block_rows(0, &[0.5, 0.75]).unwrap_err();
        assert!(err.contains("0.75"), "{err}");
    }

    /// The cache stays within its capacity by evicting values the current
    /// request does not need, and a plan handed out earlier keeps its rows.
    #[test]
    fn the_cache_is_bounded_and_eviction_never_breaks_a_live_plan() {
        let cfg = H3TransformerConfig::tiny();
        let t = tensors(&cfg, 9);
        let time = TimeWeights::load(&t);
        let mut cache = AdalnCache::new(4);
        drop(cache.prepare(&t, &cfg, &time, &[0.6, 0.7]).unwrap());
        let first = cache.prepare(&t, &cfg, &time, &[0.1, 0.2, 0.3]).unwrap();
        let second = cache.prepare(&t, &cfg, &time, &[0.3, 0.4, 0.5]).unwrap();
        assert!(cache.len() <= 5, "only rows a live plan pins may exceed the capacity, cache holds {}", cache.len());
        assert!(cache.cached(&[0.1, 0.2, 0.3]).is_ok(), "a live plan's rows are never evicted");
        assert!(cache.cached(&[0.6]).is_err(), "an unpinned row is what goes");
        let want = reference_row(&cfg, &t, 0, 0.1);
        let kept = first.block_rows(0, &[0.1]).unwrap()[0];
        assert!(kept.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-5), "an evicted value's row must survive in the plan that holds it");
        assert!(second.block_rows(0, &[0.3, 0.4, 0.5]).is_ok());
        let Err(err) = cache.prepare(&t, &cfg, &time, &[0.1, 0.2, 0.3, 0.4, 0.6]) else { panic!("over capacity must fail") };
        assert!(err.contains("capacity"), "{err}");
    }

    /// A source whose chunks are 7 elements long, so nearly every boundary
    /// falls inside a weight row.
    struct Ragged(Tensors);

    impl checkpoint::TensorSource for Ragged {
        fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
            self.0.with_tensor(name, f)
        }

        fn with_tensor_chunks(&self, name: &str, _max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
            self.0.with_tensor(name, &mut |d| {
                for (i, c) in d.chunks(7).enumerate() {
                    f((i * 7) as u64, c);
                }
            })
        }
    }

    /// Streaming the projection in chunks that split rows gives the same
    /// rows as reading it whole.
    #[test]
    fn a_chunk_boundary_inside_a_row_does_not_change_the_rows() {
        let cfg = H3TransformerConfig::tiny();
        let t = tensors(&cfg, 13);
        let time = TimeWeights::load(&t);
        let whole = AdalnCache::new(8).prepare(&t, &cfg, &time, &[0.4, 0.7]).unwrap();
        let ragged = AdalnCache::new(8).prepare(&Ragged(t.clone()), &cfg, &time, &[0.4, 0.7]).unwrap();
        for block in 0..cfg.num_layers as usize {
            assert_eq!(whole.block_rows(block, &[0.4, 0.7]).unwrap(), ragged.block_rows(block, &[0.4, 0.7]).unwrap());
        }
    }

    #[test]
    fn a_missing_projection_is_an_error_naming_it() {
        let cfg = H3TransformerConfig::tiny();
        let mut t = tensors(&cfg, 1);
        t.remove("transformer_blocks.1.adaln_proj.linear.bias");
        let Err(err) = AdalnCache::new(4).prepare(&t, &cfg, &TimeWeights::load(&t), &[0.5]) else { panic!("a missing tensor must fail") };
        assert!(err.contains("transformer_blocks.1.adaln_proj.linear.bias"), "{err}");
    }
}
