// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `model::serve::PagedDecoder` for Ripple: sliding layers keep a fixed
//! ring of recent keys per sequence, global layers keep a growing paged KV, and
//! both ride one sequence identity, under the generic continuous-batching
//! `model::serve::Scheduler`.
//!
//! Swedish Embedded AB implements solutions for serving long-context dense
//! decoders whose layers mix windowed and global attention at a fixed memory
//! budget for its clients. If your team needs expertise in hybrid KV layouts
//! and prefix reuse then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The scheduler owns admission, preemption, sampling, the prefix-cache policy
//! and speculation; this module implements the device operations it asks for
//! and nothing else.
//!
//! # One sequence, two kinds of KV
//!
//! * **Global layers** (16 of the 80) keep K/V in a block pool of `num_blocks`
//!   blocks of `block_size` rows per layer, addressed through the sequence's
//!   [`BlockTable`] and the allocator the scheduler also reads (`free_blocks`,
//!   `blocks_for`). They apply no RoPE, so cached rows are position-free.
//! * **Sliding layers** (64 of 80) keep a ring of [`ring_rows`] rows per
//!   sequence per layer: the absolute position `p` lives at ring row
//!   `p mod ring_rows`, RoPE is always applied at the absolute position, and a
//!   query at `i` attends absolute positions `i - window + 1 ..= i`. The ring
//!   is `window + prefill_chunk` rows rather than `window`: a prefill chunk
//!   writes all its rows before any of them attends, and a chunk row at `p`
//!   would otherwise overwrite position `p - window`, which an earlier row of
//!   the same chunk still needs. With the slack no key a row needs is
//!   overwritten by a row of the same chunk.
//! * The ring is the sequence's **state row** ([`BlockTable::state_row`]): the
//!   contract already carries a fixed-size per-sequence state beside the paged
//!   blocks for hybrid decoders, so nothing in `model::serve` changed. Ring
//!   rows are drawn from `max_batch` live slots (the scheduler never runs
//!   more sequences than [`PagedDecoder::max_batch`]) followed by
//!   `prefix_slots` snapshot slots.
//!
//! Attention reuses the paged decode kernel triad unchanged, one query row
//! per kernel "batch" entry with its own block table and key count: a global
//! row passes its block table; a sliding row passes a block table of
//! one-row "blocks" listing its ring rows in key order (`block_size = 1`), so
//! the window and the wrap-around are data, not a new kernel. A prefill chunk
//! is `c` rows of one sequence at consecutive positions; scratch is
//! `rows x heads x keys` and never `[heads, t, t]`.
//!
//! # Prefix reuse
//!
//! A global prefix can be shared as blocks; a ring cannot be rebuilt from
//! shared blocks once the prefix is longer than the window (it depends on
//! the last `window` rows' exact slots). So an entry of the
//! [`SnapshotPrefixCache`] is the pair that makes a prompt resumable: the
//! sequence's global blocks (retained by a fork of its table: shared, with
//! copy-on-write of a partly filled tail), a **snapshot of the ring** taken the
//! moment the prefill ended (one slot copy per sliding layer), and the last
//! hidden row, so an exact repeat needs no compute. A prompt that extends a
//! cached one adopts the blocks, copies the ring over its own, and prefills
//! only the rest. Reuse is therefore at the length a prompt was filed at: what
//! a conversation resent turn after turn needs; a shared document that
//! different prompts leave at different points is not reused (no branch
//! points are filed).
//!
//! # Weights, KV and devices
//!
//! * **Weights** are resident at the tier [`ServeConfig::weights`] names for each linear (`None`: as the
//!   checkpoint stores them, so a bf16 release is bf16 on the device and the official NVFP4 release keeps its
//!   MLP in NVFP4), loaded through `model::ops::upload_from_source` ([`crate::weights`]). The products run
//!   through `model::ops::Ops`, so every tier the shared façade executes is available here.
//! * **KV** is stored at [`ServeConfig::kv_cache`] (`f32`, `bf16`, `int8`, `fp8`; `model::kv_tier`) for both
//!   the global pools and the rings, and the attention kernels widen each element as they load it.
//! * **Devices**: the layers are cut into [`crate::stage`] pipeline stages by a `model::shard::StreamPlan`
//!   (see [`crate::placement`] for the byte-exact plan). Block tables, the allocator, ring slots and the prefix
//!   cache are host-side and shared; each stage holds the KV of its own layers, and only the residual stream
//!   crosses a boundary.
//!
//! [`global_kv_bytes_per_token`] and [`ring_bytes_per_slot`] are the two KV costs the memory plan charges: the
//! global KV grows with tokens held, the rings are fixed per slot (live and snapshot) however long the
//! sequences are. Latent sparse attention ([`RicursiveEngine::set_lsa`]) reads the global layers sparsely over
//! an f32, int8 or fp8 cache, one binding per stage; there is no offload in this engine.

use checkpoint::TensorSource;
use gpu_core::{DeviceBuffer, Gpu};
use model::kv_tier::KvTier;
use model::lsa::binding::{self, Attachment, HeadShape, LsaBinding};
use model::lsa::capture::{AttentionCapture, LayerCapture};
use model::lsa::engine::{KeyPools, KvGeometry, LsaStats};
use model::lsa::indexer::{IndexGeometry, Indexer};
use model::ops::{Ops, TierPolicy};
use model::paged::{BlockAllocator, BlockTable, KvNamespace, BASE_KV_NAMESPACE};
use model::paged_sparse::PackedKv;
use model::serve::{AdapterId, PagedDecoder, PrefillProgress};
use model::shard::StreamPlan;
use model::snapshot_prefix::{Entry, SnapshotPrefixCache};

use crate::config::{LayerType, RicursiveConfig};
use crate::layer::LayerWeights;
use crate::model::embed;
use crate::ops::pipelines;
use crate::stage::{Geometry, HostIndex, RowPlan, Stage, StageHead};
use crate::weights::{resolve_tiers, Loader, WeightReport};

/// The shortest prompt worth a snapshot: copying a slot's rings for a handful
/// of tokens costs more than prefilling them again.
pub const MIN_CACHED_TOKENS: usize = 16;

/// Candidate list width of [`RicursiveEngine::forward_batched_topk`].
const TOPK_CAPACITY: usize = 64;

/// How the engine is sized.
#[derive(Clone, Debug)]
pub struct ServeConfig {
    /// Rows per global KV block.
    pub block_size: u32,
    /// Global KV blocks in the pool (shared by live sequences and cached prefixes).
    pub num_blocks: u32,
    /// The most tokens (prompt plus generated) one sequence may hold.
    pub max_seq_len: u32,
    /// Live sequences, i.e. live ring slots.
    pub max_batch: u32,
    /// Cached prompts beside the live sequences, each costing one snapshot ring slot (0 turns reuse off).
    pub prefix_slots: u32,
    /// Prompt rows per prefill forward; sizes the ring slack.
    pub prefill_chunk: u32,
    /// Register the sparse-attention kernels ([`RicursiveEngine::set_lsa`] needs them; an engine that never reads
    /// sparsely leaves them out of its device).
    pub sparse_attention: bool,
    /// The format each linear weight is resident in (`--weights`); `None`: as the checkpoint stores it. See
    /// [`crate::weights`] and [`crate::POLICY`].
    pub weights: Option<TierPolicy>,
    /// How K and V are stored, in the global pools and in the rings (`--kv-cache`).
    pub kv_cache: KvTier,
    /// With [`Self::sparse_attention`]: the most keys a dense read of a global layer may span, which sizes the
    /// score scratch of the dense paged triad (`2 x rows x heads x keys` words). `None`: `max_seq_len`, which at a
    /// million tokens is not a size anyone allocates; an engine that reads its global layers sparsely sets the
    /// crossover it densely reads below. A dense read past it is an error, never a truncation.
    pub dense_keys: Option<u32>,
    /// Attend through the fused paged flash kernels (`model::kv_tier`) wherever the device runs them, which needs no
    /// score scratch and so no cap on the context a dense read spans; `false` pins the three-kernel triad, the reference
    /// (and what a device without workgroup barriers, the CPU JIT, always runs).
    pub fused_attention: bool,
}

impl ServeConfig {
    /// A config with the given sizes, weights as stored and an f32 KV cache.
    pub fn new(block_size: u32, num_blocks: u32, max_seq_len: u32, max_batch: u32, prefix_slots: u32, prefill_chunk: u32) -> ServeConfig {
        ServeConfig { block_size, num_blocks, max_seq_len, max_batch, prefix_slots, prefill_chunk, sparse_attention: false, weights: None, kv_cache: KvTier::F32, dense_keys: None, fused_attention: true }
    }

    fn validate(&self, cfg: &RicursiveConfig) -> Result<(), String> {
        for (name, v) in [("block_size", self.block_size), ("num_blocks", self.num_blocks), ("max_seq_len", self.max_seq_len), ("max_batch", self.max_batch), ("prefill_chunk", self.prefill_chunk)] {
            if v == 0 {
                return Err(format!("ricursive serve: {name} must be positive"));
            }
        }
        if self.kv_cache.scale() != model::kv_tier::KvScale::None && !cfg.head_dim.is_multiple_of(4) {
            return Err(format!("ricursive serve: a {} KV cache packs four elements to a word, so head_dim {} must be a multiple of 4", self.kv_cache, cfg.head_dim));
        }
        if self.dense_keys.is_some_and(|k| k == 0) || (self.dense_keys.is_some() && !self.sparse_attention) {
            return Err("ricursive serve: dense_keys is a positive size that only applies with sparse_attention (a dense engine reads max_seq_len keys)".into());
        }
        if self.sparse_attention && packed_kv(self.kv_cache).is_err() {
            return Err(format!("ricursive serve: latent sparse attention reads an f32, int8 or fp8 KV cache, not {}", self.kv_cache));
        }
        Ok(())
    }

    /// Refuse, before any device work, an indexer this deployment cannot read through: one built for another model
    /// geometry or for rotated heads, one that reads through a tree, one covering a sliding layer (which already reads
    /// its window) or no global layer, a cluster block that is not a whole number of KV blocks, or a KV cache sparse
    /// attention cannot read. What [`RicursiveEngine::set_lsa`] would refuse, asked at startup instead of at the first
    /// request.
    pub fn check_indexer(&self, cfg: &RicursiveConfig, indexer: &Indexer) -> Result<(), String> {
        indexer.check_model(&RicursiveEngine::index_geometry_of(cfg))?;
        if indexer.tree.is_some() {
            return Err("an indexer that reads through a tree is not supported by this engine".into());
        }
        packed_kv(self.kv_cache).map_err(|e| format!("latent sparse attention reads an f32, int8 or fp8 cache: {e}"))?;
        if let Some(l) = (0..cfg.num_hidden_layers).find(|&l| cfg.layer_type(l) == LayerType::Sliding && indexer.layers.get(l).is_some_and(Option::is_some)) {
            return Err(format!("the indexer covers layer {l}, a sliding layer that already reads its window"));
        }
        if !(0..cfg.num_hidden_layers).any(|l| cfg.layer_type(l) == LayerType::Full && indexer.layers.get(l).is_some_and(Option::is_some)) {
            return Err("the indexer covers no global layer".into());
        }
        if !indexer.cfg.cluster_block.is_multiple_of(self.block_size as usize) {
            return Err(format!("the indexer's cluster_block {} is not a whole number of KV blocks of {}", indexer.cfg.cluster_block, self.block_size));
        }
        Ok(())
    }

    /// Apply the model-agnostic serving policy (`--weights`, `--kv-cache`): a choice this model cannot honour is
    /// refused by name ([`crate::POLICY`]).
    pub fn apply_policy(&mut self, policy: &model::policy::ServePolicy) -> Result<(), String> {
        crate::POLICY.check(policy)?;
        if let Some(w) = &policy.weights {
            self.weights = Some(w.clone());
        }
        if let Some(kv) = policy.kv_cache {
            self.kv_cache = kv;
        }
        if policy.sparse_attention.is_some() {
            self.sparse_attention = true;
        }
        Ok(())
    }
}

/// The packed-KV kind a sparse-attention kernel reads for `tier`: `Ok(None)` for f32.
fn packed_kv(tier: KvTier) -> Result<Option<PackedKv>, String> {
    match tier {
        KvTier::F32 => Ok(None),
        KvTier::INT8 => Ok(Some(PackedKv::Int8)),
        KvTier::FP8 => Ok(Some(PackedKv::Fp8)),
        other => Err(format!("no sparse-attention kernel reads a {other} cache")),
    }
}

/// Rows of one sequence's ring on one sliding layer: the window plus the slack
/// a prefill chunk of `prefill_chunk` rows needs (see the module doc).
pub fn ring_rows(cfg: &RicursiveConfig, prefill_chunk: u32) -> u32 {
    cfg.sliding_window as u32 + prefill_chunk
}

/// Device bytes the global layers cost per cached token (K and V, scales included) at KV tier `tier`.
pub fn global_kv_bytes_per_token(cfg: &RicursiveConfig, tier: KvTier) -> u64 {
    let globals = cfg.layer_types.iter().filter(|t| **t == LayerType::Full).count() as u64;
    globals * 2 * tier.plane_bytes(1, cfg.kv_dim() as u64, cfg.head_dim as u64)
}

/// Device bytes one ring slot (live or snapshot) costs across the sliding layers at KV tier `tier`.
pub fn ring_bytes_per_slot(cfg: &RicursiveConfig, prefill_chunk: u32, tier: KvTier) -> u64 {
    let sliding = cfg.layer_types.iter().filter(|t| **t == LayerType::Sliding).count() as u64;
    sliding * 2 * tier.plane_bytes(u64::from(ring_rows(cfg, prefill_chunk)), cfg.kv_dim() as u64, cfg.head_dim as u64)
}

/// Device bytes [`RicursiveEngine::set_lsa`] allocates for `indexer` on an engine sized by `serve` with every sparse
/// layer on one device: the projections, the index pages for the block pool's cluster blocks and the scratch of the
/// widest forward. What a memory planner charges beside [`global_kv_bytes_per_token`] and [`ring_bytes_per_slot`],
/// before loading.
pub fn lsa_device_bytes(indexer: &Indexer, serve: &ServeConfig) -> u64 {
    let kv = KvGeometry { block_size: serve.block_size, max_blocks_per_seq: serve.max_seq_len.div_ceil(serve.block_size) };
    LsaBinding::device_bytes_for(indexer, kv, serve.max_batch.max(serve.prefill_chunk), serve.num_blocks)
}

/// What latent sparse attention costs when the global layers are spread over stages: per global layer (its
/// projections and its slot of every index page) and per stage that owns any (the read scratch). A stage with `g`
/// global layers is charged `g * per_global_layer + per_stage`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LsaCharge {
    pub per_global_layer: u64,
    pub per_stage: u64,
}

impl LsaCharge {
    /// The charge of attaching `indexer` to an engine sized by `serve`.
    pub fn of(indexer: &Indexer, serve: &ServeConfig) -> LsaCharge {
        let kv = KvGeometry { block_size: serve.block_size, max_blocks_per_seq: serve.max_seq_len.div_ceil(serve.block_size) };
        let (per_global_layer, per_stage) = LsaBinding::device_bytes_split(indexer, kv, serve.max_batch.max(serve.prefill_chunk), serve.num_blocks);
        LsaCharge { per_global_layer, per_stage }
    }
}

/// What resuming from a cached prompt needs beyond its blocks and ring snapshot.
struct Resume {
    /// The final-norm hidden row of the prompt's last token.
    last_hidden: Vec<f32>,
}

/// The key pools latent sparse attention indexes, as one stage holds them (one entry per model layer; the layers
/// another stage owns hold a placeholder no sparse layer of this stage reads).
struct LsaPools {
    keys: Vec<DeviceBuffer>,
    scales: Option<Vec<DeviceBuffer>>,
}

/// The latent sparse attention of one stage: its binding over the global layers the stage owns, and their key pools.
/// Stage `stage`'s index pages ride the shared block tables in index domain `stage`.
struct StageLsa {
    stage: usize,
    binding: LsaBinding,
    pools: LsaPools,
}

/// See the module doc.
pub struct RicursiveEngine {
    cfg: RicursiveConfig,
    serve: ServeConfig,
    src: Box<dyn TensorSource + Send + Sync>,
    plan: StreamPlan,
    stages: Vec<Stage>,
    tiers: TierPolicy,
    weight_report: WeightReport,
    alloc: BlockAllocator,
    ring_rows: u32,
    ring_free: Vec<u32>,
    cache: SnapshotPrefixCache<Resume>,
    max_blocks_per_seq: u32,
    /// Latent sparse attention over the global layers ([`Self::set_lsa`]); single-stage engines only.
    lsa: Vec<StageLsa>,
    /// While capturing attention: the queries of each global layer of the forward in flight.
    capture_q: Option<Vec<DeviceBuffer>>,
}

impl RicursiveEngine {
    /// Load every layer of `src` (canonical names, see [`crate::open_safetensors`]) onto the device selected by
    /// `device` (`None`: the process default) and size the KV pools. One stage holds the whole model.
    pub fn from_source(cfg: RicursiveConfig, src: Box<dyn TensorSource + Send + Sync>, serve: ServeConfig, device: Option<&str>) -> Result<RicursiveEngine, String> {
        let plan = StreamPlan::single(cfg.num_hidden_layers);
        Self::from_source_staged(cfg, src, serve, device, &plan)
    }

    /// [`Self::from_source`] with the layers cut into the pipeline stages of `plan` (see [`crate::placement`] for how
    /// to compute one): stage `i` opens its own device under the plan's placement scope (`device` selects the
    /// backend as in `Gpu::open`), holds only its layers and their KV, and passes the residual stream on. The last
    /// stage owns the final norm and `lm_head`.
    pub fn from_source_staged(cfg: RicursiveConfig, src: Box<dyn TensorSource + Send + Sync>, serve: ServeConfig, device: Option<&str>, plan: &StreamPlan) -> Result<RicursiveEngine, String> {
        serve.validate(&cfg)?;
        if plan.n_stages() == 0 || plan.n_layers() != cfg.num_hidden_layers || plan.stages()[0].start != 0 || plan.stages().windows(2).any(|w| w[0].end != w[1].start) {
            return Err(format!("ricursive serve: the stage plan {} does not tile the model's {} layers", plan.describe(), cfg.num_hidden_layers));
        }
        let mut extra: Vec<(&'static str, &'static str)> = Vec::new();
        if serve.sparse_attention {
            if let Some(shape) = Self::global_shape(&cfg) {
                let packed: Vec<PackedKv> = packed_kv(serve.kv_cache)?.into_iter().collect();
                extra = binding::kernel_pipelines(&[shape], &[], &packed, false);
            }
        }
        let table = pipelines(&cfg, &extra);
        let tiers = resolve_tiers(&cfg, src.as_ref(), serve.weights.as_ref());
        let ring_rows = ring_rows(&cfg, serve.prefill_chunk);
        let max_blocks_per_seq = serve.max_seq_len.div_ceil(serve.block_size);
        let geometry = Geometry { cfg: &cfg, serve: &serve, ring_rows, max_blocks_per_seq };

        let mut stages = Vec::with_capacity(plan.n_stages());
        let mut weight_report = WeightReport::default();
        for (i, shard) in plan.stages().iter().enumerate() {
            let last = i + 1 == plan.n_stages();
            let (stage, report) = plan.open_stage(i, || -> Result<(Stage, WeightReport), String> {
                let ops = Ops::new(Gpu::open(device, &table))?;
                let (layers, head, report) = {
                    let loader = Loader::new(&ops, src.as_ref(), &tiers);
                    let layers = (shard.start..shard.end).map(|l| LayerWeights::load(&loader, &cfg, src.as_ref(), l)).collect::<Result<Vec<_>, _>>()?;
                    let head = if last { Some(Self::load_head(&loader, &ops, &cfg, src.as_ref())?) } else { None };
                    (layers, head, loader.report())
                };
                Ok((Stage::new(ops, &geometry, shard.start..shard.end, layers, head)?, report))
            })??;
            weight_report.absorb(&report);
            stages.push(stage);
        }
        tracing::info!(stages = plan.describe(), weights = weight_report.describe(), kv = %serve.kv_cache, "ricursive engine loaded");
        Ok(RicursiveEngine {
            alloc: BlockAllocator::new(serve.num_blocks, serve.block_size),
            ring_rows,
            ring_free: (0..serve.max_batch).rev().collect(),
            cache: SnapshotPrefixCache::new(serve.prefix_slots),
            max_blocks_per_seq,
            lsa: Vec::new(),
            capture_q: None,
            plan: plan.clone(),
            stages,
            tiers,
            weight_report,
            cfg,
            serve,
            src,
        })
    }

    fn load_head(loader: &Loader<'_>, ops: &Ops, cfg: &RicursiveConfig, src: &dyn TensorSource) -> Result<StageHead, String> {
        let norm = paramstore::upload::Uploader::new(ops.gpu()).tensor(src, "model.norm.weight", cfg.hidden_size)?;
        let lm_head = loader.linear("lm_head.weight", cfg.vocab_size, cfg.hidden_size)?;
        src.advise_drop("lm_head.weight");
        Ok(StageHead { norm, lm_head })
    }

    /// Whether attention runs the fused paged flash kernels on every stage (else the score-slab triad: a device without
    /// workgroup barriers, or [`ServeConfig::fused_attention`] off).
    pub fn fused_attention(&self) -> bool {
        self.stages.iter().all(|s| s.fused)
    }

    pub fn config(&self) -> &RicursiveConfig {
        &self.cfg
    }

    /// The tier each linear weight is resident in.
    pub fn weight_tiers(&self) -> &TierPolicy {
        &self.tiers
    }

    /// What the load put on the devices: tensors and bytes per tier.
    pub fn weight_report(&self) -> &WeightReport {
        &self.weight_report
    }

    /// The pipeline stages: layer ranges and the cards they run on.
    pub fn stage_plan(&self) -> &StreamPlan {
        &self.plan
    }

    fn geometry(&self) -> Geometry<'_> {
        Geometry { cfg: &self.cfg, serve: &self.serve, ring_rows: self.ring_rows, max_blocks_per_seq: self.max_blocks_per_seq }
    }

    fn first(&self) -> &Stage {
        &self.stages[0]
    }

    /// The stage that runs `layer`.
    fn stage_of(&self, layer: usize) -> &Stage {
        self.stages.iter().find(|s| s.range.contains(&layer)).expect("the stages tile the layers")
    }

    fn last(&self) -> &Stage {
        self.stages.last().expect("an engine has a stage")
    }

    /// The attention shape of the global layers (`None` for a model without any).
    fn global_shape(cfg: &RicursiveConfig) -> Option<HeadShape> {
        cfg.layer_types.contains(&LayerType::Full).then_some(HeadShape { n_heads: cfg.num_attention_heads as u32, n_kv_heads: cfg.num_key_value_heads as u32, head_dim: cfg.head_dim as u32 })
    }

    /// The geometry an indexer for this model carries: global layers apply no RoPE, so the heads the index reads
    /// are unrotated (`rope_base` 0), and an indexer built with a RoPE base is built for other heads.
    fn index_geometry(cfg: &RicursiveConfig) -> IndexGeometry {
        IndexGeometry { n_layers: cfg.num_hidden_layers, n_heads: cfg.num_attention_heads, n_kv_heads: cfg.num_key_value_heads, head_dim: cfg.head_dim, rope_base: 0.0 }
    }

    /// [`Self::index_geometry`] for a config, so an indexer file can be checked against the model before any engine
    /// is opened.
    pub fn index_geometry_of(cfg: &RicursiveConfig) -> IndexGeometry {
        Self::index_geometry(cfg)
    }

    fn kv_geometry(&self) -> KvGeometry {
        KvGeometry { block_size: self.serve.block_size, max_blocks_per_seq: self.max_blocks_per_seq }
    }

    /// Read the global layers' attention through latent sparse attention: each query reads the keys `indexer`
    /// selects instead of every cached key. Only global layers can read sparsely (a sliding layer already reads
    /// its window); an indexer that covers one, was built for another geometry or for rotated heads, reads
    /// through a tree, or is attached to an engine opened without [`ServeConfig::sparse_attention`] or over a bf16
    /// cache is refused. On an engine split over stages each stage that owns global layers gets its own binding over
    /// those layers (its device's projections, pages and scratch); the bindings ride the shared block tables, each
    /// in its own index domain. Index pages are built from the cached keys as cluster blocks close, shared over
    /// prefix snapshots with the blocks they index, freed with them, and charged by [`lsa_device_bytes`] (one
    /// device) or [`LsaCharge`] (per global layer and per stage).
    pub fn set_lsa(&mut self, indexer: &Indexer) -> Result<(), String> {
        if !self.serve.sparse_attention {
            return Err("ricursive: this engine was opened without ServeConfig::sparse_attention, so its device holds no sparse-attention kernels".into());
        }
        if indexer.tree.is_some() {
            return Err("ricursive: an indexer that reads through a tree is not supported by this engine (its node pages are not shared over prefix snapshots)".into());
        }
        let shape = Self::global_shape(&self.cfg).ok_or("ricursive: the model has no global layer to read sparsely")?;
        let packed = packed_kv(self.serve.kv_cache).map_err(|e| format!("ricursive: {e}"))?;
        let capable: Vec<bool> = self.cfg.layer_types.iter().map(|t| *t == LayerType::Full).collect();
        let geometry = Self::index_geometry(&self.cfg);
        // Geometry and layer coverage are refused before any device work, whatever the backend.
        indexer.check_model(&geometry)?;
        let mut attached = Vec::new();
        for (i, stage) in self.stages.iter().enumerate() {
            // This stage's share of the indexer: the global layers it owns (a stage of sliding layers has none).
            let mine = indexer.restricted_to(|l| stage.range.contains(&l));
            if mine.layers.iter().all(Option::is_none) {
                continue;
            }
            let own: Vec<bool> = capable.iter().enumerate().map(|(l, c)| *c && stage.range.contains(&l)).collect();
            let attachment = Attachment {
                geometry,
                packed,
                kv: self.kv_geometry(),
                max_rows: self.serve.max_batch.max(self.serve.prefill_chunk),
                page_capacity: binding::page_capacity(self.serve.num_blocks, self.serve.block_size, indexer.cfg.cluster_block as u32),
                yarn_inv_freq: None,
                sparse_capable: Some(&own),
            };
            let kernels = binding::resolve_kernels(stage.gpu(), shape, packed, false)?;
            let binding = LsaBinding::attach_in_domain(stage.gpu(), &mine, kernels, &attachment, i)?;
            let per_layer = |f: &dyn Fn(&model::kv_tier::KvLayer) -> DeviceBuffer| -> Vec<DeviceBuffer> {
                (0..self.cfg.num_hidden_layers).map(|l| f(stage.kv_of(if stage.range.contains(&l) { l } else { stage.range.start }))).collect()
            };
            let keys = per_layer(&|l| l.k.data().clone());
            let scales = packed.map(|_| per_layer(&|l| l.k.scales().expect("a packed plane has scales").clone()));
            attached.push(StageLsa { stage: i, binding, pools: LsaPools { keys, scales } });
        }
        if attached.is_empty() {
            return Err("ricursive: the indexer covers no global layer".into());
        }
        self.lsa = attached;
        Ok(())
    }

    /// What latent sparse attention holds and has done on all stages (one small readback each), or `None` while every
    /// layer reads densely.
    pub fn lsa_stats(&self) -> Option<LsaStats> {
        let mut all = self.lsa.iter().map(|s| s.binding.stats(self.stages[s.stage].gpu()));
        let first = all.next()?;
        Some(all.fold(first, |a, b| LsaStats {
            device_bytes: a.device_bytes + b.device_bytes,
            pages_in_use: a.pages_in_use + b.pages_in_use,
            page_capacity: a.page_capacity + b.page_capacity,
            pages_built: a.pages_built + b.pages_built,
            reads: model::lsa::device::ReadStats { rows: a.reads.rows + b.reads.rows, selected_keys: a.reads.selected_keys + b.reads.selected_keys, causal_keys: a.reads.causal_keys + b.reads.causal_keys },
            dense_rows: a.dense_rows + b.dense_rows,
        }))
    }

    /// The prefill position up to which chunks read densely (0 without sparse attention).
    pub fn lsa_dense_until(&self) -> u32 {
        self.lsa.first().map_or(0, |a| a.binding.dense_until())
    }

    /// Override the prefill position up to which chunks read densely (0, the default: every chunk reads sparsely;
    /// this engine has no fused dense prefill to measure a crossover against).
    pub fn set_lsa_dense_until(&mut self, position: u32) {
        for s in &mut self.lsa {
            s.binding.set_dense_until(position);
        }
    }

    /// Release `table`'s index pages and blocks.
    fn drop_table(&mut self, table: &mut BlockTable) {
        for s in &mut self.lsa {
            s.binding.release(table);
        }
        table.release(&mut self.alloc);
    }

    /// The first stage's device, for profiling.
    pub fn gpu(&self) -> &Gpu {
        self.first().gpu()
    }

    /// Device bytes of the two KV pools across all stages: the block pool (global layers) and the ring slots
    /// (sliding layers).
    pub fn kv_bytes(&self) -> (u64, u64) {
        let global = global_kv_bytes_per_token(&self.cfg, self.serve.kv_cache) * u64::from(self.serve.num_blocks) * u64::from(self.serve.block_size);
        let rings = ring_bytes_per_slot(&self.cfg, self.serve.prefill_chunk, self.serve.kv_cache) * u64::from(self.serve.max_batch + self.serve.prefix_slots);
        (global, rings)
    }

    /// Ring slot a snapshot row of the prefix cache lives in.
    fn snapshot_slot(&self, row: u32) -> u32 {
        self.serve.max_batch + row
    }

    fn ring_slot(table: &BlockTable) -> u32 {
        table.state_row().expect("a sequence that has begun prefilling owns a ring slot")
    }

    // ---- graph ----------------------------------------------------------------------------------------------------

    /// The layer stack over `rows` across the stages: appends their K/V and returns the residual stream after the
    /// last layer, `[rows, hidden]` on the last stage's device, before the final norm.
    fn forward_hidden(&mut self, rows: &[RowPlan<'_>]) -> Result<DeviceBuffer, String> {
        let tokens: Vec<u32> = rows.iter().map(|r| r.token).collect();
        let x0_host = embed(self.src.as_ref(), &self.cfg, &tokens)?;
        let geometry = Geometry { cfg: &self.cfg, serve: &self.serve, ring_rows: self.ring_rows, max_blocks_per_seq: self.max_blocks_per_seq };
        let index = HostIndex::build(&geometry, rows);
        let n = rows.len() as u32;
        let boundary_bytes = u64::from(n) * self.cfg.hidden_size as u64 * 4;
        let mut carry: Option<(usize, DeviceBuffer)> = None;
        for (i, stage) in self.stages.iter().enumerate() {
            let gpu = stage.gpu();
            let x0 = gpu.storage(x0_host.len() as u64);
            gpu.write_f32(&x0, &x0_host);
            let x_in = match carry.take() {
                None => x0.clone(),
                Some((from, x)) => {
                    let into = gpu.storage(x0_host.len() as u64);
                    self.stages[from].gpu().copy_to(&x, 0, gpu, &into, 0, boundary_bytes);
                    into
                }
            };
            let out = stage.forward(&geometry, &index, n, &x_in, &x0, self.lsa.iter().find(|s| s.stage == i).map(|s| &s.binding), self.capture_q.as_mut())?;
            carry = Some((i, out));
        }
        for s in &self.lsa {
            s.binding.end_forward();
        }
        Ok(carry.expect("an engine has a stage").1)
    }

    /// Final norm and logits for `rows` rows of the residual stream `x` (on the last stage): `(normed, logits)`.
    fn head(&self, x: &DeviceBuffer, rows: u32) -> (Vec<f32>, Vec<f32>) {
        self.last().head_rows(self.cfg.rms_norm_eps, x, rows, self.cfg.hidden_size as u32, self.cfg.vocab_size as u32)
    }

    /// Copy ring slot `from` over ring slot `to` on every sliding layer.
    fn copy_ring(&self, from: u32, to: u32) {
        let rows = u64::from(self.ring_rows);
        let geometry = self.geometry();
        for stage in &self.stages {
            stage.copy_rows(&geometry, LayerType::Sliding, u64::from(from) * rows, u64::from(to) * rows, rows);
        }
    }

    /// Copy the first `rows` rows of global block `from` over block `to` on every global layer.
    fn copy_block_rows(&self, from: u32, to: u32, rows: u32) {
        let bs = u64::from(self.serve.block_size);
        let geometry = self.geometry();
        for stage in &self.stages {
            stage.copy_rows(&geometry, LayerType::Full, u64::from(from) * bs, u64::from(to) * bs, u64::from(rows));
        }
    }

    // ---- sequences ------------------------------------------------------------------------------------------------

    /// First slice of a sequence: take its ring slot and resume from the prefix cache when there is a hit.
    /// `Some(hidden)` when the whole prompt was cached.
    fn begin(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Option<Vec<f32>> {
        assert!(!prompt.is_empty(), "ricursive serve: empty prompt (no token to produce a hidden state from)");
        assert!(prompt.len() <= self.serve.max_seq_len as usize, "ricursive serve: prompt of {} tokens exceeds the sequence limit of {}", prompt.len(), self.serve.max_seq_len);
        if let Some(&bad) = prompt.iter().find(|&&t| t as usize >= self.cfg.vocab_size) {
            panic!("ricursive serve: prompt token {bad} is outside the model vocabulary ({})", self.cfg.vocab_size);
        }
        let slot = self.ring_free.pop().expect("ricursive serve: no free ring slot (more live sequences than max_batch)");
        table.set_state_row(slot);
        if prompt.len() < MIN_CACHED_TOKENS {
            return None;
        }
        let hit = self.cache.lookup(table.kv_namespace(), prompt)?;
        let bs = self.serve.block_size;
        let entry = self.cache.entry(hit);
        let rows = entry.tokens.len() as u32;
        let (snap, shared, hidden) = (entry.snap_row, entry.table.blocks()[..rows.div_ceil(bs) as usize].to_vec(), entry.payload.last_hidden.clone());
        table.adopt_rows(&shared, rows, &mut self.alloc);
        if !rows.is_multiple_of(bs) {
            // The entry keeps serving later prompts from this block, so the one this sequence writes next is a copy.
            let (from, to) = table.unshare_tail(&mut self.alloc).expect("a block shared with the prefix cache has a single owner");
            self.copy_block_rows(from, to, rows % bs);
        }
        self.copy_ring(self.snapshot_slot(snap), slot);
        (rows as usize == prompt.len()).then_some(hidden)
    }

    /// File the sequence of `table`, whose state is exactly `prompt`'s, as a prefix entry.
    fn file_prefix(&mut self, table: &BlockTable, prompt: &[u32], hidden: &[f32]) {
        if prompt.len() < MIN_CACHED_TOKENS || !self.cache.enabled() {
            return;
        }
        let (row, evicted) = self.cache.take_row(table.kv_namespace(), prompt);
        if let Some(mut gone) = evicted {
            self.drop_table(&mut gone);
        }
        let Some(row) = row else { return };
        self.copy_ring(Self::ring_slot(table), self.snapshot_slot(row));
        let mut shared = table.fork(&mut self.alloc);
        for s in &mut self.lsa {
            // The snapshot holds the index pages of the blocks it holds (each stage's own), so they outlive the sequence with them.
            s.binding.retain_pages(table, &mut shared).unwrap_or_else(|e| panic!("ricursive: latent sparse attention: {e}"));
        }
        let retired = self.cache.insert(Entry::new(prompt.to_vec(), shared, row, Resume { last_hidden: hidden.to_vec() }));
        for mut gone in retired {
            self.drop_table(&mut gone);
        }
    }

    /// Prefill `prompt[from..to]` (at most one ring-slack chunk) into `table`; the last row's final-norm hidden state
    /// when `last`.
    fn prefill_slice(&mut self, table: &mut BlockTable, prompt: &[u32], from: usize, to: usize, last: bool) -> Option<Vec<f32>> {
        let x = self.forward_slice(table, prompt, from, to);
        last.then(|| {
            // Only the prompt's last row is projected.
            let hidden = self.cfg.hidden_size;
            let gpu = self.last().gpu();
            let all = gpu.read(&x, (to - from) * hidden);
            let row = gpu.storage(hidden as u64);
            gpu.write_f32(&row, &all[(to - from - 1) * hidden..]);
            self.head(&row, 1).0
        })
    }

    /// The layer stack over `prompt[from..to]` of `table` (a chunk of at most the ring slack), appending its K/V:
    /// the residual stream after the last layer, `[to - from, hidden]`.
    fn forward_slice(&mut self, table: &mut BlockTable, prompt: &[u32], from: usize, to: usize) -> DeviceBuffer {
        for s in &mut self.lsa {
            s.binding.prepare_chunk(self.stages[s.stage].gpu(), table, from as u32, (to - from) as u32, s.pools.key_pools()).unwrap_or_else(|e| panic!("ricursive serve: latent sparse attention: {e}"));
        }
        let start = table.reserve((to - from) as u32, &mut self.alloc).unwrap_or_else(|e| panic!("ricursive serve: {e}"));
        assert_eq!(start as usize, from, "ricursive serve: the table holds {start} rows but the prompt resumes at {from}");
        let (blocks, ring) = (table.blocks().to_vec(), Self::ring_slot(table));
        let rows: Vec<RowPlan<'_>> = (from..to).map(|p| RowPlan { token: prompt[p], pos: p as u32, ring, blocks: &blocks }).collect();
        self.forward_hidden(&rows).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Run `prompt` from an empty table in chunks, calling `each(table, chunk start, chunk end, residual stream)` after each;
    /// the sequence is released afterwards. LSA, if attached, stays in effect.
    fn run_chunks(&mut self, prompt: &[u32], mut each: impl FnMut(&mut Self, &BlockTable, usize, usize, &DeviceBuffer)) -> Result<(), String> {
        if prompt.is_empty() || prompt.len() > self.serve.max_seq_len as usize {
            return Err(format!("ricursive: a text of {} tokens: the engine holds 1..={}", prompt.len(), self.serve.max_seq_len));
        }
        if let Some(&bad) = prompt.iter().find(|&&t| t as usize >= self.cfg.vocab_size) {
            return Err(format!("ricursive: token {bad} is outside the vocabulary ({})", self.cfg.vocab_size));
        }
        let slot = self.ring_free.pop().ok_or("ricursive: no free sequence slot for a scoring pass")?;
        let mut table = BlockTable::new();
        table.set_state_row(slot);
        let mut at = 0usize;
        while at < prompt.len() {
            let limit = self.lsa.first().map_or(usize::MAX, |a| a.binding.chunk_limit(at as u32) as usize);
            let to = prompt.len().min(at + (self.serve.prefill_chunk as usize).min(limit));
            let x = self.forward_slice(&mut table, prompt, at, to);
            each(self, &table, at, to, &x);
            at = to;
        }
        self.release_table(&mut table);
        Ok(())
    }

    /// The queries (normed, unrotated: the global layers apply no RoPE) and keys of every global layer over a dense
    /// prefill of `prompt` ([`model::lsa::capture::AttentionCapture`]); `None` for the sliding layers.
    fn capture_global(&mut self, prompt: &[u32], positions: &[u32]) -> Result<Vec<Option<LayerCapture>>, String> {
        if self.serve.kv_cache != KvTier::F32 {
            return Err(format!("ricursive: attention capture reads the unquantized keys; open the engine with an f32 KV cache, not {}", self.serve.kv_cache));
        }
        let c = self.cfg.clone();
        let globals: Vec<usize> = (0..c.num_hidden_layers).filter(|&l| c.layer_type(l) == LayerType::Full).collect();
        let (q_dim, kv_dim) = (c.q_dim(), c.kv_dim());
        let mut queries: Vec<Vec<f32>> = globals.iter().map(|_| Vec::with_capacity(positions.len() * q_dim)).collect();
        let lsa = std::mem::take(&mut self.lsa);
        self.capture_q = Some(Vec::new());
        let mut keys: Vec<Vec<f32>> = globals.iter().map(|_| Vec::with_capacity(prompt.len() * kv_dim)).collect();
        let result = self.run_chunks(prompt, |e, table, from, to, _| {
            let taken = e.capture_q.replace(Vec::new()).unwrap_or_default();
            for ((acc, buf), &l) in queries.iter_mut().zip(&taken).zip(&globals) {
                let rows = e.stage_of(l).gpu().read(buf, (to - from) * q_dim);
                // Only the requested positions are kept: a long text does not hold every query.
                for &p in positions.iter().filter(|&&p| (from..to).contains(&(p as usize))) {
                    acc.extend_from_slice(&rows[(p as usize - from) * q_dim..][..q_dim]);
                }
            }
            // This chunk's global K rows, from the blocks the table holds for `from..to`.
            for (acc, &l) in keys.iter_mut().zip(&globals) {
                for p in from..to {
                    let (block, offset) = table.locate(p as u32, e.serve.block_size);
                    let row = (block * e.serve.block_size + offset) as usize * kv_dim;
                    let mut buf = vec![0f32; kv_dim];
                    e.stage_of(l).gpu().read_into(e.stage_of(l).kv_of(l).k.data(), row as u64, &mut buf);
                    acc.extend(buf);
                }
            }
        });
        self.capture_q = None;
        self.lsa = lsa;
        result?;
        let mut out: Vec<Option<LayerCapture>> = vec![None; c.num_hidden_layers];
        for ((&l, q), k) in globals.iter().zip(queries).zip(keys) {
            out[l] = Some(LayerCapture { q, k });
        }
        Ok(out)
    }

    /// The log-probability the model gives each next token of `prompt` at `from..len - 1`.
    fn score(&mut self, prompt: &[u32], from: usize) -> Result<Vec<f32>, String> {
        let vocab = self.cfg.vocab_size;
        let mut out = Vec::new();
        self.run_chunks(prompt, |e, _, start, to, x| {
            let (_, logits) = e.head(x, (to - start) as u32);
            for (i, row) in logits.chunks(vocab).enumerate() {
                let p = start + i;
                if p >= from && p + 1 < prompt.len() {
                    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let log_total = m as f64 + row.iter().map(|&z| ((z - m) as f64).exp()).sum::<f64>().ln();
                    out.push((row[prompt[p + 1] as usize] as f64 - log_total) as f32);
                }
            }
        })?;
        Ok(out)
    }

    /// One decode step: the logits of each `(table, input)` pair, `[vocab]` per sequence.
    pub fn decode_logits(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<Vec<f32>> {
        assert_eq!(tables.len(), inputs.len(), "ricursive serve: tables/inputs length mismatch");
        if tables.is_empty() {
            return Vec::new();
        }
        assert!(tables.len() <= self.serve.max_batch as usize, "ricursive serve: {} sequences exceed max_batch {}", tables.len(), self.serve.max_batch);
        if !self.lsa.is_empty() {
            let mut rows: Vec<(&mut BlockTable, u32, u32)> = tables
                .iter_mut()
                .map(|t| {
                    let cached = t.len();
                    (&mut **t, cached, 1)
                })
                .collect();
            for s in &mut self.lsa {
                s.binding.prepare_rows(self.stages[s.stage].gpu(), &mut rows, s.pools.key_pools()).unwrap_or_else(|e| panic!("ricursive serve: latent sparse attention: {e}"));
            }
        }
        let mut owned = Vec::with_capacity(tables.len());
        for (t, &input) in tables.iter_mut().zip(inputs) {
            t.append(&mut self.alloc).unwrap_or_else(|e| panic!("ricursive serve: {e}"));
            assert!(t.len() <= self.serve.max_seq_len, "ricursive serve: sequence outgrew the limit of {} tokens", self.serve.max_seq_len);
            owned.push((input, t.len() - 1, Self::ring_slot(t), t.blocks().to_vec()));
        }
        let rows: Vec<RowPlan<'_>> = owned.iter().map(|(token, pos, ring, blocks)| RowPlan { token: *token, pos: *pos, ring: *ring, blocks }).collect();
        let x = self.forward_hidden(&rows).unwrap_or_else(|e| panic!("{e}"));
        let n = rows.len();
        let logits = self.head(&x, n as u32).1;
        logits.chunks(self.cfg.vocab_size).map(<[f32]>::to_vec).collect()
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| if v > best.1 { (i, v) } else { best }).0 as u32
}

fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut ranked: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(k);
    ranked
}

impl PagedDecoder for RicursiveEngine {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        self.serve.prefill_chunk
    }
    fn max_batch(&self) -> Option<usize> {
        Some(self.serve.max_batch as usize)
    }
    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        self.serve.max_seq_len as usize
    }
    fn vocab(&self) -> usize {
        self.cfg.vocab_size
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.serve.block_size)
    }
    /// Whole blocks a prompt adopts from the cache; the partly filled tail is copied, not adopted.
    fn cached_prefix_blocks(&self, prompt: &[u32], adapter: Option<AdapterId>) -> u32 {
        let namespace: KvNamespace = adapter.map_or(BASE_KV_NAMESPACE, AdapterId::kv_namespace);
        if prompt.len() < MIN_CACHED_TOKENS {
            return 0;
        }
        self.cache.peek(namespace, prompt).map_or(0, |i| self.cache.entry(i).tokens.len() as u32 / self.serve.block_size)
    }
    fn reclaim_prefix(&mut self, want: u32) -> u32 {
        let before = self.alloc.free_blocks();
        while self.alloc.free_blocks() - before < want {
            let Some(mut gone) = self.cache.evict_lru() else { break };
            self.drop_table(&mut gone);
        }
        self.alloc.free_blocks() - before
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        if let Some(slot) = t.take_state_row() {
            self.ring_free.push(slot);
        }
        self.drop_table(t);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        loop {
            if let PrefillProgress::Done(hidden) = self.prefill_chunk(table, prompt, usize::MAX) {
                return hidden;
            }
        }
    }
    fn prefill_interleave_tokens(&self) -> Option<usize> {
        Some(self.serve.prefill_chunk as usize)
    }
    /// The next slice of `prompt`, at most `budget` tokens, run as forwards of at most `prefill_chunk` rows. The first
    /// call takes the sequence's ring slot and resumes from the prefix cache; the call that ends the prompt files it.
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        if table.state_row().is_none() {
            if let Some(hidden) = self.begin(table, prompt) {
                return PrefillProgress::Done(hidden);
            }
        }
        let done = table.len() as usize;
        let end = done + budget.min(prompt.len() - done);
        let mut hidden = None;
        let mut at = done;
        while at < end {
            // A chunk stays inside one cluster block, so every query's closed blocks are indexed before it runs.
            let limit = self.lsa.first().map_or(usize::MAX, |a| a.binding.chunk_limit(at as u32) as usize);
            let to = end.min(at.saturating_add(self.serve.prefill_chunk as usize).min(at.saturating_add(limit)));
            hidden = self.prefill_slice(table, prompt, at, to, to == prompt.len());
            at = to;
        }
        if end < prompt.len() {
            return PrefillProgress::Partial { consumed: end - done };
        }
        let hidden = hidden.expect("the last slice of a prompt returns its hidden state");
        self.file_prefix(table, prompt, &hidden);
        PrefillProgress::Done(hidden)
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let d = self.cfg.hidden_size;
        assert_eq!(hidden.len(), d, "ricursive serve::logits: hidden must be one row of {d} floats, got {}", hidden.len());
        self.last().project(hidden, self.cfg.vocab_size)
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        self.decode_logits(tables, inputs).iter().map(|l| argmax(l)).collect()
    }
    fn forward_batched_greedy_window(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<u32>> {
        assert!((1..=self.decode_window_capacity()).contains(&k), "window {k} exceeds this engine's decode_window_capacity {}", self.decode_window_capacity());
        self.forward_batched_greedy(tables, inputs).into_iter().map(|t| vec![t]).collect()
    }
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        let k = k.clamp(1, self.topk_capacity());
        self.decode_logits(tables, inputs).iter().map(|l| top_k(l, k)).collect()
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        self.cache.stats()
    }
    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.first().gpu().stats()
    }
    /// The block pool of the global layers plus every ring slot (live and snapshot) of the sliding layers.
    fn kv_pool_bytes(&self) -> u64 {
        let (global, rings) = self.kv_bytes();
        global + rings
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        u64::from(self.serve.num_blocks) * u64::from(self.serve.block_size)
    }
    fn decode_window_capacity(&self) -> usize {
        1
    }
    fn sparse_attention_stats(&self) -> Option<LsaStats> {
        self.lsa_stats()
    }
    fn topk_capacity(&self) -> usize {
        TOPK_CAPACITY.min(self.cfg.vocab_size)
    }
}

impl AttentionCapture for RicursiveEngine {
    fn index_geometry(&self) -> IndexGeometry {
        Self::index_geometry(&self.cfg)
    }
    /// Only the global layers: a sliding layer already reads its window.
    fn sparse_capable(&self) -> Vec<bool> {
        self.cfg.layer_types.iter().map(|t| *t == LayerType::Full).collect()
    }
    fn max_capture_tokens(&self) -> usize {
        self.serve.max_seq_len as usize
    }
    fn capture_attention(&mut self, prompt: &[u32], positions: &[u32]) -> Result<Vec<Option<LayerCapture>>, String> {
        self.capture_global(prompt, positions)
    }
    fn position_logprobs(&mut self, prompt: &[u32], from: usize) -> Vec<f32> {
        self.score(prompt, from).unwrap_or_else(|e| panic!("{e}"))
    }
    fn set_lsa(&mut self, indexer: &Indexer) -> Result<(), String> {
        RicursiveEngine::set_lsa(self, indexer)
    }
}

impl LsaPools {
    fn key_pools(&self) -> KeyPools<'_> {
        KeyPools { keys: &self.keys, scales: self.scales.as_deref() }
    }
}
