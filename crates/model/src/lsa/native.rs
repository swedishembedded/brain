// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Native sparse attention for training (N-LSA): attention restricted to each
//! query's [`Selection`], forward and backward, in time and memory linear in
//! the context - never the `[t, t]` scores dense attention under a mask
//! needs.
//!
//! Swedish Embedded AB implements long-context models trained with sparse
//! attention for clients whose sequences outgrow quadratic attention. If your
//! team needs expertise in sparse attention training, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The math is dense causal attention with every unselected key masked out:
//! the softmax runs over the selected keys only, and gradients flow exactly
//! through them (the selection itself is a constant). The forward
//! (`lsa_native_fwd`) keeps each row's log-sum-exp; the backward rebuilds the
//! probabilities from it, `lsa_native_bwd_dq` per query over its selection,
//! `lsa_native_bwd_dkv` per key over the queries that selected it - the
//! selection inverted per key, deterministically and on the device - so no
//! gradient is scattered and nothing needs atomics. Layouts are the causal
//! GQA flash trio's: q and ctx `[b * t, n_heads * head_dim]`, k and v
//! `[b * t, n_kv_heads * head_dim]`.
//!
//! The selection comes from [`NativeSelector`]: the serving index kernels
//! over the layer's own rotated queries and keys in the attention's own index
//! space, recorded onto the training tape with the attention, so a step never
//! reads activations back to the host. [`NativeShape::pack`] builds the same
//! index on the host from [`Selection`]s (fixed selections, tests).

use super::device::{IndexHeads, IndexPool, IndexVectorBuffers, ReadBuffers, ReadGeometry, ReadStats, Rope};
use super::{LsaConfig, Selection};
use crate::paged_sparse::ROW_HEADER;
use gpu_core::sort::{record_sort_pairs, SortKernels, SortScratch};
use gpu_core::{DeviceBuffer, Gpu, Step};

/// The kernels, for a model's pipeline list: forward, query half and key half
/// of the backward, then the selection's inversion (pairs, offsets).
pub const KERNELS: [(&str, &str); 5] = [
    ("lsa_native_fwd", kernels::LSA_NATIVE_FWD),
    ("lsa_native_bwd_dq", kernels::LSA_NATIVE_BWD_DQ),
    ("lsa_native_bwd_dkv", kernels::LSA_NATIVE_BWD_DKV),
    ("lsa_native_pairs", kernels::LSA_NATIVE_PAIRS),
    ("lsa_native_offsets", kernels::LSA_NATIVE_OFFSETS),
];

/// Largest head the kernels hold.
pub const MAX_HEAD_DIM: u32 = 256;
/// Most query heads per KV head the kernels hold.
pub const MAX_GROUP: u32 = 8;
/// Workgroup memory the query-side kernels declare (the group's queries and
/// context gradients, `2 * MAX_GROUP * MAX_HEAD_DIM` floats, and tile state).
pub const WORKGROUP_BYTES: u32 = 17 * 1024;

/// The caller's pipeline indices of the attention kernels.
#[derive(Clone, Copy, Debug)]
pub struct NativeKernels {
    pub fwd: usize,
    pub bwd_dq: usize,
    pub bwd_dkv: usize,
}

/// Keys a selection row holds under `cfg`: what a query may read, plus the
/// cluster block the device read keeps room for (`ReadGeometry::
/// selection_width`), so the read writes its rows straight into the index.
pub fn row_width(cfg: &LsaConfig) -> u32 {
    (cfg.selection_width() + cfg.cluster_block) as u32
}

/// The attention shape: `b` sequences of `t` positions, grouped-query heads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeShape {
    pub b: u32,
    pub t: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// Most keys one selection row holds ([`row_width`]).
    pub width: u32,
}

impl NativeShape {
    fn rows(&self) -> u64 {
        u64::from(self.b) * u64::from(self.t)
    }

    /// Words per selection row: the [`ROW_HEADER`] (count, row, live keys),
    /// then the positions - the device read's own rows.
    fn sel_stride(&self) -> u32 {
        self.width + ROW_HEADER
    }

    /// Words of the index: the selection, the inverse's offsets, its entries.
    pub fn index_words(&self) -> u64 {
        self.rows() * u64::from(self.sel_stride()) + self.rows() + 1 + self.rows() * u64::from(self.width)
    }

    /// Words of the statistics: the forward's log-sum-exp and the backward's
    /// `D = dO . O`, `[2, b, n_heads, t]`.
    pub fn stats_words(&self) -> u64 {
        2 * self.rows() * u64::from(self.n_heads)
    }

    fn params(&self) -> [u32; 7] {
        [self.b, self.n_heads, self.n_kv_heads, self.t, self.head_dim, self.n_heads / self.n_kv_heads, self.sel_stride()]
    }

    fn check(&self) -> Result<(), String> {
        if self.n_kv_heads == 0 || !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(format!("native sparse attention: {} heads over {} KV heads", self.n_heads, self.n_kv_heads));
        }
        if self.head_dim > MAX_HEAD_DIM || self.n_heads / self.n_kv_heads > MAX_GROUP {
            return Err(format!("native sparse attention holds head_dim <= {MAX_HEAD_DIM} and <= {MAX_GROUP} heads per KV head, got {} and {}", self.head_dim, self.n_heads / self.n_kv_heads));
        }
        let widest = self.rows() * u64::from(self.n_heads) * u64::from(self.head_dim);
        if widest.max(self.index_words()) > u64::from(u32::MAX) {
            return Err(format!("native sparse attention: {widest} elements overflow the kernels' u32 indexing"));
        }
        Ok(())
    }

    /// Pack one [`Selection`] per sequence (each `t` rows, causal, unique,
    /// every row non-empty, at most `width` keys) into the index the kernels
    /// read: the rows, then the selection inverted per key.
    pub fn pack(&self, selections: &[Selection]) -> Result<Vec<u32>, String> {
        let (t, width) = (self.t as usize, self.width as usize);
        if selections.len() != self.b as usize || selections.iter().any(|s| s.rows() != t) {
            return Err(format!("native sparse attention: selections for {} sequences of {t} queries expected", self.b));
        }
        let rows = self.rows() as usize;
        let stride = self.sel_stride() as usize;
        let header = ROW_HEADER as usize;
        let mut words = vec![0u32; self.index_words() as usize];
        let (sel, rest) = words.split_at_mut(rows * stride);
        let (off, inv) = rest.split_at_mut(rows + 1);
        let mut per_key = vec![0u32; rows];
        for (r, out) in sel.chunks_exact_mut(stride).enumerate() {
            let (seq, i) = (r / t, r % t);
            let keys = selections[seq].row(i);
            if keys.is_empty() || keys.len() > width {
                return Err(format!("native sparse attention: query {i} of sequence {seq} selects {} keys (1..={width})", keys.len()));
            }
            out[..header].copy_from_slice(&[keys.len() as u32, r as u32, i as u32 + 1]);
            for (slot, &j) in out[header..].iter_mut().zip(keys) {
                if j < 0 || j as usize > i {
                    return Err(format!("native sparse attention: query {i} of sequence {seq} selects key {j}"));
                }
                *slot = j as u32;
                per_key[seq * t + j as usize] += 1;
            }
        }
        for r in 0..rows {
            off[r + 1] = off[r] + per_key[r];
        }
        let mut fill: Vec<u32> = off[..rows].to_vec();
        for (r, row) in sel.chunks_exact(stride).enumerate() {
            let (seq, i) = (r / t, r % t);
            for &j in &row[header..][..row[0] as usize] {
                let key = seq * t + j as usize;
                inv[fill[key] as usize] = i as u32;
                fill[key] += 1;
            }
        }
        Ok(words)
    }

    /// The forward: `ctx` from `q`, `k`, `v` over the packed `index`, and the
    /// row log-sum-exp into the first half of `stats`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, g: &Gpu, ids: &NativeKernels, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, index: &DeviceBuffer, ctx: &DeviceBuffer, stats: &DeviceBuffer) -> Result<Step, String> {
        self.check_on(g)?;
        Ok(g.dispatch(ids.fwd, &[q, k, v, index, ctx, stats], &self.params(), gpu_core::Dispatch::Workgroups(self.kv_groups()?)))
    }

    /// The backward of [`Self::forward`]: `d_q`, `d_k`, `d_v` (overwritten)
    /// from the context gradient, the forward's inputs and output, and its
    /// `stats`.
    pub fn backward(&self, g: &Gpu, ids: &NativeKernels, b: &NativeBackward) -> Result<Vec<Step>, String> {
        self.check_on(g)?;
        let p = self.params();
        let groups = gpu_core::Dispatch::Workgroups(self.kv_groups()?);
        Ok(vec![
            g.dispatch(ids.bwd_dq, &[b.q, b.k, b.v, b.ctx, b.d_ctx, b.stats, b.index, b.d_q], &p, groups),
            g.dispatch(ids.bwd_dkv, &[b.q, b.k, b.v, b.d_ctx, b.stats, b.index, b.d_k, b.d_v], &p, groups),
        ])
    }

    /// [`Self::check`] and the device: the kernels are workgroup-cooperative.
    fn check_on(&self, g: &Gpu) -> Result<(), String> {
        self.check()?;
        let caps = g.caps();
        if !caps.workgroup_reductions || caps.workgroup_mem_bytes < WORKGROUP_BYTES {
            return Err(format!("native sparse attention needs workgroup barriers and {WORKGROUP_BYTES} bytes of workgroup memory"));
        }
        Ok(())
    }

    /// Workgroups of each kernel: one per (row, KV head).
    fn kv_groups(&self) -> Result<u32, String> {
        u32::try_from(self.rows() * u64::from(self.n_kv_heads)).map_err(|_| "native sparse attention: too many workgroups".to_string())
    }
}

/// The buffers [`NativeShape::backward`] binds.
pub struct NativeBackward<'a> {
    pub q: &'a DeviceBuffer,
    pub k: &'a DeviceBuffer,
    pub v: &'a DeviceBuffer,
    pub ctx: &'a DeviceBuffer,
    pub d_ctx: &'a DeviceBuffer,
    pub index: &'a DeviceBuffer,
    pub stats: &'a DeviceBuffer,
    pub d_q: &'a DeviceBuffer,
    pub d_k: &'a DeviceBuffer,
    pub d_v: &'a DeviceBuffer,
}

/// The caller's pipeline indices of the selection's kernels: the index
/// ([`super::device::KERNELS`]), the inversion (the last two of
/// [`KERNELS`]) and the sort it runs on ([`gpu_core::sort::KERNELS`]).
#[derive(Clone, Copy, Debug)]
pub struct SelectKernels {
    pub index_vectors: usize,
    pub read: usize,
    pub ward: usize,
    pub pairs: usize,
    pub offsets: usize,
    pub sort: SortKernels,
}

/// Native selection on the device: from a layer's rotated queries and keys,
/// the index vectors of every position (un-rotated, KV-group-summed, not
/// projected: the attention's own index space), every closed cluster block of
/// every sequence Ward-merged in one dispatch, each query's read written
/// straight into the index, and the selection inverted per key by a stable
/// sort of (key, query) pairs - the [`super::select`] the host reference
/// makes, as recorded steps. Scratch is linear in the context except the
/// read's scores, one per bucket of every closed block per query (a bucket
/// holds about `bucket_size` keys).
pub struct NativeSelector {
    cfg: LsaConfig,
    shape: NativeShape,
    heads: IndexHeads,
    read: ReadGeometry,
    blocks: u32,
    pool: IndexPool,
    ward_scratch: DeviceBuffer,
    q_off: DeviceBuffer,
    k_off: DeviceBuffer,
    pos: DeviceBuffer,
    key_pos: DeviceBuffer,
    seqlen: DeviceBuffer,
    page_tab: DeviceBuffer,
    iq: DeviceBuffer,
    ik: DeviceBuffer,
    unused: DeviceBuffer,
    scores: DeviceBuffer,
    read_stats: DeviceBuffer,
    inversion: Inversion,
}

impl NativeSelector {
    /// Scratch for `shape`'s selections under `cfg` (its `index_dim` is the
    /// native width, `n_kv_heads * head_dim`; `shape.width` is
    /// [`row_width`]), heads rotated as `rope` says (`Rope::None` for an attention that does not rotate them).
    pub fn new(g: &Gpu, cfg: LsaConfig, shape: NativeShape, rope: Rope) -> Result<NativeSelector, String> {
        shape.check()?;
        let heads = IndexHeads { n_kv_heads: shape.n_kv_heads, head_dim: shape.head_dim, rope, index_dim: cfg.index_dim as u32, identity: true };
        heads.check(shape.n_heads)?;
        let cb = cfg.cluster_block as u32;
        let blocks = shape.t / cb;
        let read = ReadGeometry { cfg, max_pages: blocks };
        read.check()?;
        if read.selection_width() != shape.width {
            return Err(format!("native selection: rows of {} keys, the read writes {}", shape.width, read.selection_width()));
        }
        let (b, t) = (shape.b, shape.t);
        let rows = shape.rows();
        let slots = b * blocks;
        let d = cfg.index_dim as u64;
        let pool = IndexPool::new(g, cfg, 1, slots)?;
        let positions: Vec<u32> = (0..b).flat_map(|_| 0..t).collect();
        let storage_with = |words: &[u32]| {
            let buf = g.storage((words.len() as u64).max(1));
            g.write(&buf, words);
            buf
        };
        let q_off: Vec<u32> = (0..b * t).map(|r| r * shape.n_heads * shape.head_dim).collect();
        // Keys of the closed blocks only, block after block, so Ward slot `w`
        // finds its keys at `w * cluster_block`.
        let key_rows: Vec<u32> = (0..b).flat_map(|s| (0..blocks * cb).map(move |p| s * t + p)).collect();
        let k_off: Vec<u32> = key_rows.iter().map(|r| r * shape.n_kv_heads * shape.head_dim).collect();
        let key_pos: Vec<u32> = key_rows.iter().map(|r| r % t).collect();
        let width = blocks as usize + 1;
        let mut tab = vec![0u32; rows as usize * width];
        for (r, row) in tab.chunks_exact_mut(width).enumerate() {
            let seq = r as u32 / t;
            row[0] = blocks;
            for (c, page) in row[1..].iter_mut().enumerate() {
                *page = seq * blocks + c as u32;
            }
        }
        Ok(NativeSelector {
            cfg,
            shape,
            heads,
            read,
            blocks,
            pool,
            ward_scratch: g.storage((u64::from(slots) * IndexPool::ward_scratch_words(&cfg)).max(1)),
            q_off: storage_with(&q_off),
            k_off: storage_with(&k_off),
            pos: storage_with(&positions),
            key_pos: storage_with(&key_pos),
            seqlen: storage_with(&positions.iter().map(|p| p + 1).collect::<Vec<u32>>()),
            page_tab: storage_with(&tab),
            iq: g.storage(rows * d),
            ik: g.storage((key_rows.len() as u64 * d).max(1)),
            unused: g.storage(1),
            scores: g.storage((rows * u64::from(read.score_stride())).max(1)),
            read_stats: g.storage(ReadStats::WORDS),
            inversion: Inversion::new(g, shape, cfg.selection_width() as u32),
        })
    }

    /// Record the selection from rotated `q` and `k` (the attention's
    /// layouts) into `index` ([`NativeShape::index_words`]), ready for
    /// [`NativeShape::forward`].
    pub fn record(&self, g: &Gpu, ids: &SelectKernels, q: &DeviceBuffer, k: &DeviceBuffer, index: &DeviceBuffer) -> Result<Vec<Step>, String> {
        let s = &self.shape;
        let rows = s.rows() as u32;
        let cb = self.cfg.cluster_block as u32;
        let mut steps = vec![self.heads.step(g, ids.index_vectors, s.n_heads, rows, &IndexVectorBuffers { x: q, row_off: &self.q_off, pos: &self.pos, w: &self.unused, out: &self.iq, inv_freq: &self.unused, out0: 0 })?];
        if self.blocks > 0 {
            let key_rows = s.b * self.blocks * cb;
            steps.push(self.heads.step(g, ids.index_vectors, s.n_kv_heads, key_rows, &IndexVectorBuffers { x: k, row_off: &self.k_off, pos: &self.key_pos, w: &self.unused, out: &self.ik, inv_freq: &self.unused, out0: 0 })?);
            steps.push(self.pool.ward_step(g, ids.ward, &self.ik, &self.ward_scratch, 0, 0, self.blocks)?);
        }
        let bufs = ReadBuffers { iq: &self.iq, centroids: &self.pool.centroids, members: &self.pool.members, page_tab: &self.page_tab, seqlen: &self.seqlen, scores: &self.scores, sel: index, row0: 0, stats: &self.read_stats };
        steps.push(self.read.step(g, ids.read, rows, &bufs)?);
        self.inversion.record(g, ids, index, &mut steps);
        Ok(steps)
    }
}

/// The selection of a window-only attention: each query reads its last `cfg.window` keys and the first `cfg.sink`
/// positions, the baseline N-LSA is compared with (nothing here depends on the keys' values).
pub fn window_selection(cfg: &LsaConfig, t: usize) -> Selection {
    let width = cfg.window + cfg.sink;
    let mut ids = vec![-1i32; t * width];
    let mut lens = vec![0u32; t];
    for i in 0..t {
        let start = (i + 1).saturating_sub(cfg.window);
        let row: Vec<i32> = (start..=i).chain(0..cfg.sink.min(start)).map(|j| j as i32).collect();
        ids[i * width..][..row.len()].copy_from_slice(&row);
        lens[i] = row.len() as u32;
    }
    Selection { width, ids, lens }
}

/// The selection inverted per key on the device: a (key row, query) pair per
/// selected key (`lsa_native_pairs`, at most `emit` per row), a stable radix
/// sort by key, then each key's offset and queries (`lsa_native_offsets`).
/// Equal keys keep their row order, so a key's queries ascend and the index
/// is exactly [`NativeShape::pack`]'s.
pub struct Inversion {
    shape: NativeShape,
    emit: u32,
    pairs: [DeviceBuffer; 4],
    sort: SortScratch,
}

impl Inversion {
    /// Scratch for `shape` with at most `emit` selected keys per row.
    pub fn new(g: &Gpu, shape: NativeShape, emit: u32) -> Inversion {
        let pairs = (shape.rows() * u64::from(emit)).max(1);
        Inversion { shape, emit, pairs: std::array::from_fn(|_| g.storage(pairs)), sort: SortScratch::new(g, pairs as usize) }
    }

    /// Record the inverse of `index`'s selection rows into its inverse part.
    pub fn record(&self, g: &Gpu, ids: &SelectKernels, index: &DeviceBuffer, steps: &mut Vec<Step>) {
        let s = &self.shape;
        let rows = s.rows() as u32;
        let pairs = rows * self.emit;
        let [keys_a, vals_a, keys_b, vals_b] = &self.pairs;
        steps.push(g.dispatch(ids.pairs, &[index, keys_a, vals_a], &[rows, s.t, s.sel_stride(), self.emit], gpu_core::Dispatch::Workgroups(rows)));
        // Key rows stay below `rows`, padding (all ones) above them.
        let key_bits = 32 - rows.leading_zeros();
        let in_b = record_sort_pairs(g, &ids.sort, keys_a, vals_a, keys_b, vals_b, pairs as usize, key_bits, &self.sort, steps);
        let (keys, vals) = if in_b { (keys_b, vals_b) } else { (keys_a, vals_a) };
        steps.push(g.dispatch(ids.offsets, &[keys, vals, index], &[rows, pairs, s.sel_stride(), 0], gpu_core::Dispatch::Threads(pairs + 1)));
    }
}
