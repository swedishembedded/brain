// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A layer-sharded, int8-MoE-expert resident Thinker spanning as many GPUs as
//! it actually needs - built on `crates/residency/src/multi.rs`'s
//! `MultiDeviceResidentModel`/`claim_multi`, `crate::int8_resident::
//! ThinkerInt8Store`, and the two generic mechanisms this model exists to
//! USE rather than reimplement:
//!
//! * **Placement**: `model::shard::plan_fewest_devices` - a capacity-aware,
//!   exact-DP contiguous layer partitioner over `(device, usable bytes)`
//!   pairs. Nothing here knows how many cards the box has, whether they are
//!   the same size, or how to balance them; it supplies real per-layer byte
//!   costs read off the checkpoint's own header and takes the plan it is
//!   given. A 1-GPU box, a 2xP40 box and a mixed 24/8 GB box all work, and a
//!   model that genuinely does not fit reports that instead of OOMing partway
//!   through a multi-minute load.
//! * **Loading**: `paramstore::upload::Uploader` - the bounded disk→VRAM
//!   mover. Weights are streamed from the mapping to the card with peak host
//!   allocation of one chunk, never one tensor: the packed expert weights are
//!   normally lent zero-copy straight out of the mmap, and the packed-but-
//!   fp32-consumed tensors (attention/router projections, `lm_head`) are
//!   dequantized a row block at a time by `model::int8::upload_dequantized`
//!   rather than expanded whole on the host first.
//!
//! `embed_tokens` is not uploaded at all: [`Self::generate`](
//! Int8ThinkerInstance::generate) only ever needs a per-token ROW, so rows
//! are read (and dequantized) on demand straight from the mapping. At the
//! real Thinker shape that is a ~1.2 GB f32 table that never exists in host
//! RAM or VRAM.
//!
//! The residual stream is handed between shards via a host round-trip
//! (`gpu_a.read` → `gpu_b.write`, `n * d_model` floats per hop - under 10 KiB
//! at any realistic decode batch).
//!
//! Reachable through `residency::Executor` via `Executor::register_multi`
//! (never `register` — [`Int8ThinkerResident::estimate`]/[`activate`](
//! ResidentModel::activate) are deliberately unusable stand-ins, since a
//! multi-device-only model has no meaningful single-device footprint; see
//! `crates/cli/src/resident_omni.rs::int8_thinker_multi_from_env`).
//!
//! # Two request shapes, one action
//!
//! `generate` accepts EITHER a raw `ids` blob (token ids in, token ids out -
//! the original contract, which the multi-GPU and executor tests drive) OR the
//! ordinary chat request every other served text model takes (`messages` /
//! `prompt`, text out). The chat shape is what makes this model reachable over
//! `/v1/chat/completions`, `/v1/messages` and D-Bus the same way `brain/omni`
//! is, rather than being a faster path nobody can call: `apiserve::catalog::
//! api_caps` classifies a model chat-capable only from its manifest, so the
//! declared spec is [`crate::caps::chat_generate_spec`] - the SAME builder
//! `brain/omni`'s own `generate` uses, not a second copy of that param list.
//!
//! Tokenization needs vocab files, and a brain-native int8 checkpoint is a
//! single `.safetensors` with no tokenizer sibling, so the directory to read
//! them from is configured separately (`crates/cli/src/resident_omni.rs::
//! int8_thinker_multi_from_env`). Without one the model still serves the raw
//! `ids` contract and says so on a chat request, rather than failing to load.
//!
//! Still out of scope, deliberately: multimodal input and speech output - this
//! serves Thinker TEXT generation only, and `brain/omni` remains the path with
//! the audio/image/video splice and `speak`.

use std::collections::HashMap;
use std::sync::OnceLock;

use capability::{last_user_text, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, Progress};
use checkpoint::weightio::WeightReader;
use checkpoint::TensorSource;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use gpu_core::{DeviceBuffer, Gpu};
use model::shard::LayerBytes;
use paramstore::upload::Uploader;
use residency::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

use crate::config::MoeTextConfig;
use crate::int8_resident::{expert_bytes, ThinkerInt8Store};
use crate::thinker::{final_norm, layer_decode_step, layer_fwd, lm_head_fwd, thinker_pipelines, ThinkerLayerCache, ThinkerLayerWeights};

pub const MODEL: &str = "brain/omni-int8-thinker-multi";

/// A contiguous layer range `[start, end)` assigned to one device.
type LayerRange = std::ops::Range<usize>;

/// The per-layer NON-expert tensor leaves [`load_layer_bufs`] uploads, under
/// the `thinker.blocks.{l}.` prefix. Named once so the byte accounting
/// ([`layer_resident_bytes`]) and the loader cannot drift: an accounting that
/// silently omits a tensor the loader uploads is exactly the kind of
/// under-reported budget that lets a placement decision overrun a card.
pub const LAYER_LEAVES: &[&str] = &[
    "ln1.weight",
    "attn.wq.weight",
    "attn.wk.weight",
    "attn.wv.weight",
    "attn.wo.weight",
    "attn.q_norm.weight",
    "attn.k_norm.weight",
    "ln2.weight",
    "mlp.router.weight",
];

/// Tensors the LAST shard holds (where `forward`'s own host round-trip
/// already lands the final hidden state, so applying the head there avoids an
/// extra cross-device hop).
pub const HEAD_TENSORS: &[&str] = &["thinker.norm.weight", "thinker.lm_head.weight"];

/// The token embedding - read a row at a time from the mapping, never
/// uploaded and never materialized (see this module's doc).
pub const EMBED_TENSOR: &str = "thinker.embed_tokens.weight";

// ---------------------------------------------------------------- byte accounting

/// Declared element count of `name`, or 0 if absent.
fn numel(reader: &WeightReader, name: &str) -> u64 {
    reader.shape(name).map(|s| s.iter().product::<u64>()).unwrap_or(0)
}

/// Device bytes `name` occupies once loaded **as f32** - i.e. what
/// [`load_mat`]/[`load_vec`] actually place on the card.
///
/// Delegates to [`paramstore::dtype`], the ONE dtype→device-bytes table, which
/// the raw-HF path (`crate::thinker_plan`) charges against too: placement is a
/// question about bytes, and a per-dtype copy of the answer here is how the
/// two paths would drift. It matters most for the case that table exists to
/// get right - a packed `U32` tensor is stored `[n, k/4]` but consumed as
/// `[n, k]` f32 (`thinker::layer_fwd` has no int8 dispatch for attention/
/// router/head, only for the routed experts), so it costs FOUR TIMES its
/// on-disk size in VRAM.
fn f32_resident_bytes(reader: &WeightReader, name: &str) -> u64 {
    paramstore::dtype::device_bytes(reader.dtype(name), numel(reader, name))
}

/// Total device bytes layer `l` occupies on whichever shard owns it: its
/// routed experts ([`expert_bytes`]) plus every [`LAYER_LEAVES`] tensor.
/// `None` if the checkpoint is missing any of them.
pub fn layer_resident_bytes(reader: &WeightReader, cfg: &MoeTextConfig, l: usize) -> Option<u64> {
    let experts = expert_bytes(reader, std::iter::once(l), cfg)?;
    let mut non_expert = 0u64;
    for leaf in LAYER_LEAVES {
        let name = format!("thinker.blocks.{l}.{leaf}");
        let b = f32_resident_bytes(reader, &name);
        if b == 0 {
            return None; // absent (or empty) -- the loader would panic later
        }
        non_expert += b;
    }
    Some(experts + non_expert)
}

/// Total device bytes the LAST shard additionally carries ([`HEAD_TENSORS`]).
pub fn head_resident_bytes(reader: &WeightReader) -> Option<u64> {
    let mut total = 0u64;
    for n in HEAD_TENSORS {
        let b = f32_resident_bytes(reader, n);
        if b == 0 {
            return None;
        }
        total += b;
    }
    Some(total)
}

/// The byte-exact per-stage cost model for this checkpoint - the input
/// `model::shard`'s capacity-aware planner needs. `embed` is 0 because the
/// embedding is never device-resident here (see this module's doc).
///
/// `None` - never a panic - for a checkpoint missing anything the loader
/// would upload: this runs on the `Executor` dispatcher thread (see
/// [`MultiDeviceResidentModel::estimate_multi`]'s contract).
pub fn layer_cost(reader: &WeightReader, cfg: &MoeTextConfig) -> Option<LayerBytes> {
    let mut per_layer = Vec::with_capacity(cfg.n_layers as usize);
    for l in 0..cfg.n_layers as usize {
        per_layer.push(layer_resident_bytes(reader, cfg, l)?);
    }
    Some(LayerBytes { per_layer, embed: 0, head: head_resident_bytes(reader)? })
}

// ---------------------------------------------------------------- loading

pub struct LayerBufs {
    ln1: DeviceBuffer,
    wq: DeviceBuffer,
    wk: DeviceBuffer,
    wv: DeviceBuffer,
    wo: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    ln2: DeviceBuffer,
    router: DeviceBuffer,
}

/// The logical (unpacked) `[n, k]` shape of a tensor that may be stored
/// packed. A packed tensor's own shape is `[n, k/4]`, so `k` cannot be read
/// off it directly - but it is always exactly four times the stored last
/// dimension, which is why no caller has to pass shapes in by hand.
fn logical_shape(reader: &WeightReader, name: &str) -> Result<(usize, usize), String> {
    let shape = reader.shape(name).ok_or_else(|| format!("missing tensor {name}"))?;
    if shape.len() != 2 {
        return Err(format!("{name}: expected a rank-2 tensor, got shape {shape:?}"));
    }
    let n = shape[0] as usize;
    let k = shape[1] as usize * if reader.dtype(name) == Some("U32") { 4 } else { 1 };
    Ok((n, k))
}

/// One tensor, real, host-resident: plain f32 if the checkpoint stored it
/// that way, or unpacked via [`model::int8::dequantize_weight`] if
/// `omni::import` quantized it (`should_quantize`: rank-2, last dim a
/// multiple of 4). `n`/`k` are only consulted on the quantized branch.
///
/// **Prefer [`load_mat`]** for anything going to a device: this materializes
/// the whole f32 expansion on the host (1.2 GB for a real `lm_head`), which
/// is precisely what the streaming loader avoids. Kept public because a
/// caller that genuinely wants the host copy - an independent test oracle,
/// an off-device row gather - has no other way to ask for one.
pub fn load_mat_host(reader: &WeightReader, name: &str, n: u32, k: u32) -> Vec<f32> {
    match reader.dtype(name) {
        Some("U32") => {
            let packed = reader.tensor_u32(name).unwrap_or_else(|| panic!("missing tensor {name}"));
            let scale_name = format!("{name}.scale");
            let scale = reader.tensor(&scale_name).unwrap_or_else(|| panic!("missing tensor {scale_name} (scale sibling of quantized {name})"));
            model::int8::dequantize_weight(&packed, &scale, n as usize, k as usize)
        }
        _ => reader.tensor(name).unwrap_or_else(|| panic!("missing tensor {name}")),
    }
}

/// A rank-2 weight, streamed from `reader` onto `up`'s device as f32, with
/// bounded host use: a plain `F32` tensor goes straight across (zero-copy
/// where the mapping allows), and a packed one is dequantized a row block at
/// a time by [`model::int8::upload_dequantized`]. Shapes come from the
/// checkpoint header, so no caller passes them in.
pub fn load_mat(up: &mut Uploader, reader: &WeightReader, name: &str) -> DeviceBuffer {
    let (n, k) = logical_shape(reader, name).unwrap_or_else(|e| panic!("{e}"));
    match reader.dtype(name) {
        Some("U32") => model::int8::upload_dequantized(up, reader, name, n, k).unwrap_or_else(|e| panic!("{e}")),
        _ => up.tensor(reader, name, n * k).unwrap_or_else(|e| panic!("{e}")),
    }
}

/// A 1-D tensor (norm gains) — `omni::import::should_quantize` never
/// quantizes rank-1 tensors, so these are always plain f32.
pub fn load_vec(up: &mut Uploader, reader: &WeightReader, name: &str) -> DeviceBuffer {
    let n = numel(reader, name) as usize;
    assert!(n > 0, "missing tensor {name}");
    up.tensor(reader, name, n).unwrap_or_else(|e| panic!("{e}"))
}

/// Build layer `l`'s REAL non-expert weights on `up`'s device, streamed from
/// `reader`. Brain-native names (`omni::import::map_thinker`'s output),
/// matching exactly what `crate::int8_resident::ThinkerInt8Store::build`
/// reads for the same layer's expert weights from the same checkpoint, and
/// exactly the [`LAYER_LEAVES`] the byte accounting charges for.
pub fn load_layer_bufs(up: &mut Uploader, reader: &WeightReader, l: usize) -> LayerBufs {
    let p = |leaf: &str| format!("thinker.blocks.{l}.{leaf}");
    LayerBufs {
        ln1: load_vec(up, reader, &p("ln1.weight")),
        wq: load_mat(up, reader, &p("attn.wq.weight")),
        wk: load_mat(up, reader, &p("attn.wk.weight")),
        wv: load_mat(up, reader, &p("attn.wv.weight")),
        wo: load_mat(up, reader, &p("attn.wo.weight")),
        q_norm: load_vec(up, reader, &p("attn.q_norm.weight")),
        k_norm: load_vec(up, reader, &p("attn.k_norm.weight")),
        ln2: load_vec(up, reader, &p("ln2.weight")),
        router: load_mat(up, reader, &p("mlp.router.weight")),
    }
}

pub fn weights(b: &LayerBufs) -> ThinkerLayerWeights<'_> {
    ThinkerLayerWeights { ln1: &b.ln1, wq: &b.wq, wk: &b.wk, wv: &b.wv, wo: &b.wo, q_norm: &b.q_norm, k_norm: &b.k_norm, ln2: &b.ln2, router: &b.router, experts: &[] }
}

/// How this instance reads one embedding row.
///
/// The table is `[vocab, hidden]` - 1.2 GB as f32 at the real shape - and
/// generation only ever needs a per-token ROW gather, never a GEMM. So it is
/// neither uploaded nor expanded: rows are read straight from the mapping.
enum EmbedTable {
    /// Packed int8 on disk: read the row's words from the mapping and scale
    /// them by that row's own scale. Peak allocation is one row.
    Packed { scale: Vec<f32>, k: usize },
    /// Plain f32 on disk: the row is a slice of the mapping.
    Plain { k: usize },
    /// The mapping declined to lend its bytes (a byte range this reader
    /// cannot borrow as words). One host copy, the old behaviour - correct,
    /// just not free. brain-native checkpoints are written 8-byte-aligned
    /// specifically so this branch is not taken.
    Host { table: Vec<f32>, k: usize },
}

impl EmbedTable {
    fn open(reader: &WeightReader) -> Result<EmbedTable, String> {
        let (n, k) = logical_shape(reader, EMBED_TENSOR)?;
        let packed = reader.dtype(EMBED_TENSOR) == Some("U32");
        if reader.raw_words(EMBED_TENSOR).is_some() {
            if !packed {
                return Ok(EmbedTable::Plain { k });
            }
            let scale_name = format!("{EMBED_TENSOR}.scale");
            let scale = reader.tensor(&scale_name).ok_or_else(|| format!("{MODEL}: missing {scale_name}"))?;
            if scale.len() != n {
                return Err(format!("{MODEL}: {scale_name} has {} entries, expected {n}", scale.len()));
            }
            return Ok(EmbedTable::Packed { scale, k });
        }
        Ok(EmbedTable::Host { table: load_mat_host(reader, EMBED_TENSOR, n as u32, k as u32), k })
    }

    fn hidden(&self) -> usize {
        match self {
            EmbedTable::Packed { k, .. } | EmbedTable::Plain { k } | EmbedTable::Host { k, .. } => *k,
        }
    }

    /// Token `t`'s embedding row, `[hidden]`.
    fn row(&self, reader: &WeightReader, t: u32) -> Vec<f32> {
        let t = t as usize;
        match self {
            EmbedTable::Packed { scale, k } => {
                let kg = k / 4;
                let words = reader.raw_words(EMBED_TENSOR).expect("EmbedTable::Packed implies a lendable mapping");
                let mut out = Vec::with_capacity(*k);
                model::int8::dequantize_rows_into(&words[t * kg..(t + 1) * kg], scale, t, 1, *k, &mut out);
                out
            }
            EmbedTable::Plain { k } => {
                let words = reader.raw_words(EMBED_TENSOR).expect("EmbedTable::Plain implies a lendable mapping");
                words[t * k..(t + 1) * k].iter().map(|w| f32::from_bits(*w)).collect()
            }
            EmbedTable::Host { table, k } => table[t * k..(t + 1) * k].to_vec(),
        }
    }
}

/// One device's shard: its own `Gpu`, the absolute layer range it owns, the
/// resident int8 expert store for that range, and each owned layer's
/// non-expert weights.
struct DeviceShard {
    gpu: Gpu,
    range: LayerRange,
    store: ThinkerInt8Store,
    layer_bufs: HashMap<usize, LayerBufs>,
}

pub struct Int8ThinkerInstance {
    cfg: MoeTextConfig,
    shards: Vec<DeviceShard>,
    /// Kept open for the lifetime of the instance so embedding rows can be
    /// read on demand. This is the header-only mmap handle, not data.
    reader: WeightReader,
    embed: EmbedTable,
    /// `thinker.norm.weight` and `thinker.lm_head.weight`, resident on the
    /// LAST shard's `Gpu`.
    final_norm_w: DeviceBuffer,
    lm_head_w: DeviceBuffer,
    /// The tokenizer backing the CHAT request shape, and the stop ids that go
    /// with it. `None` when no tokenizer directory was configured or it could
    /// not be read - the raw `ids` contract still works, so a missing
    /// tokenizer degrades one request shape instead of failing the load.
    tok: Option<(QwenBpe, Vec<u32>)>,
}

/// What one walk through the sharded stack does at each layer.
enum Pass<'a> {
    /// Batched causal forward over `n` positions, optionally bulk-filling the
    /// KV cache (the prefill half of the decode loop, and the cacheless
    /// `forward` action).
    Batched { cache: Option<&'a Int8KvCache> },
    /// One new token attending against `cache` at row `pos`.
    Decode { cache: &'a Int8KvCache, pos: u32 },
}

/// Per-layer incremental-decode KV cache for the sharded int8 Thinker.
///
/// Each layer's cache is allocated on the SAME device as the layer that fills
/// it, so it never crosses a card - the sharding here splits by layer RANGE and
/// never splits a layer, which is exactly what makes that possible.
///
/// This is what turns `generate` from O(T²) into O(T): every step used to
/// re-embed the whole ids-so-far window and re-run the full stack over it.
/// Measured on the real 30B checkpoint at a 9-token prompt, that recompute was
/// the dominant remaining cost once the weights stopped streaming.
struct Int8KvCache {
    /// Indexed by ABSOLUTE layer number, on that layer's own device.
    layers: Vec<(DeviceBuffer, DeviceBuffer)>,
    cap: u32,
}

impl Int8KvCache {
    fn new(shards: &[DeviceShard], cfg: &MoeTextConfig, cap: u32) -> Int8KvCache {
        let hkv = (cfg.n_kv_heads * cfg.head_dim) as u64;
        let mut layers = Vec::with_capacity(cfg.n_layers as usize);
        for l in 0..cfg.n_layers as usize {
            // The shard that owns layer `l`; a plan always covers every layer,
            // and the last shard is a safe home for a degenerate one.
            let shard = shards.iter().find(|s| s.range.contains(&l)).unwrap_or_else(|| shards.last().expect("at least one shard"));
            layers.push((shard.gpu.storage(cap as u64 * hkv), shard.gpu.storage(cap as u64 * hkv)));
        }
        Int8KvCache { layers, cap }
    }

    fn layer(&self, l: usize) -> ThinkerLayerCache<'_> {
        ThinkerLayerCache { kcache: &self.layers[l].0, vcache: &self.layers[l].1 }
    }
}

impl Int8ThinkerInstance {
    /// Run every owned layer, in absolute layer order, across however many
    /// device shards this instance holds — a host round-trip hands the
    /// residual stream to the next shard's `Gpu` at each boundary. `x` is
    /// the initial hidden state `[n, d]`, host-resident (the caller's own
    /// embedding/splice step, out of scope here — see this module's doc).
    /// Returns the final hidden state `[n, d]`, host-resident (no final
    /// norm/lm_head — this is the MoE-bearing-layers validation action, not
    /// a full generate()).
    pub fn forward(&self, x_host: &[f32], n: u32) -> Vec<f32> {
        // M-RoPE table: diagonal (plain text), same construction
        // thinker_decode.rs/thinker_int8_parity.rs use.
        let tokens: Vec<u32> = (0..n).collect();
        let positions = qwenvl::mrope::get_rope_index(&tokens, u32::MAX, &[]);
        let (cos_tab, sin_tab) = self.mrope_tables(&positions);
        self.run_shards(x_host, n, &cos_tab, &sin_tab, Pass::Batched { cache: None })
    }

    /// The M-RoPE cos/sin tables for `positions`, at this Thinker's section
    /// split / head dim / theta.
    fn mrope_tables(&self, positions: &[[u32; 3]]) -> (Vec<f32>, Vec<f32>) {
        let section: [u32; 3] = [self.cfg.mrope_section[0], self.cfg.mrope_section[1], self.cfg.mrope_section[2]];
        qwenvl::mrope::mrope_tables(positions, section, self.cfg.head_dim, self.cfg.rope_theta)
    }

    /// Every owned layer, in absolute layer order, across every device shard,
    /// with a host round-trip handing the residual stream on at each boundary.
    /// Returns the final hidden state `[n, d]`, host-resident.
    ///
    /// The one place the shard walk lives. [`Self::forward`], the prefill and
    /// each decode step differ only in the [`Pass`] handed in, never in how the
    /// stack is traversed - so cross-device handoff is written and reasoned
    /// about once.
    fn run_shards(&self, x_host: &[f32], n: u32, cos_tab: &[f32], sin_tab: &[f32], pass: Pass) -> Vec<f32> {
        let d = self.cfg.hidden;
        assert_eq!(x_host.len(), (n * d) as usize, "x must be [n, d]");
        let mut h_host = x_host.to_vec();
        for shard in &self.shards {
            if shard.range.is_empty() {
                continue; // a capacity-driven plan may legitimately leave a stage empty
            }
            let gpu = &shard.gpu;
            let cos = gpu.storage_init("cos", cos_tab);
            let sin = gpu.storage_init("sin", sin_tab);
            let mut h = gpu.storage_init("h", &h_host);
            for l in shard.range.clone() {
                let lb = &shard.layer_bufs[&l];
                let w = weights(lb);
                let experts8 = shard.store.layer(l);
                h = match &pass {
                    Pass::Batched { cache } => {
                        let lc = cache.map(|c| c.layer(l));
                        layer_fwd(gpu, &self.cfg, &w, &h, &cos, &sin, n, lc.as_ref(), Some(experts8)).0
                    }
                    Pass::Decode { cache, pos } => {
                        layer_decode_step(gpu, &self.cfg, &w, &cache.layer(l), &h, &cos, &sin, *pos, cache.cap, Some(experts8))
                    }
                };
            }
            // Host-mediated handoff to the next shard's device (a no-op
            // read+carry on the LAST shard -- still correct, just an extra
            // round trip nothing further consumes).
            h_host = gpu.read(&h, (n * d) as usize);
        }
        h_host
    }

    /// Greedy (argmax) text generation over the sharded int8 Thinker: real
    /// tokens in, real sampled tokens out, EOS handling, no tokenization
    /// (matches `qwenvl::Qwen3Vl::generate`'s own contract: `prompt_ids`/
    /// `eos_ids` are already token ids, and the return value is the
    /// GENERATED tokens only, prompt excluded).
    ///
    /// KV-cached: the prompt is prefilled ONCE into a per-layer cache sized
    /// `prompt + max_new_tokens`, and each subsequent token is a single-row
    /// decode step against it - the same shape `crate::generate`'s bf16 path
    /// uses, and the reason `thinker::layer_decode_step` already takes the
    /// `int8_experts` argument. Each layer's cache lives on the device that
    /// owns that layer ([`Int8KvCache`]), so nothing crosses a card.
    pub fn generate(&self, prompt_ids: &[u32], max_new_tokens: u32, eos_ids: &[u32]) -> Vec<u32> {
        let d = self.cfg.hidden as usize;
        let last = self.shards.last().expect("Int8ThinkerInstance has no shards");

        let mut out = Vec::with_capacity(max_new_tokens as usize);
        if prompt_ids.is_empty() || max_new_tokens == 0 {
            return out;
        }

        let n0 = prompt_ids.len() as u32;
        let cache = Int8KvCache::new(&self.shards, &self.cfg, n0 + max_new_tokens);

        // Prefill: the whole prompt through the stack once, filling the cache.
        let positions = qwenvl::mrope::get_rope_index(prompt_ids, u32::MAX, &[]);
        let (cos_tab, sin_tab) = self.mrope_tables(&positions);
        let mut x_host = Vec::with_capacity(prompt_ids.len() * d);
        for &t in prompt_ids {
            x_host.extend_from_slice(&self.embed.row(&self.reader, t));
        }
        let hidden = self.run_shards(&x_host, n0, &cos_tab, &sin_tab, Pass::Batched { cache: Some(&cache) });
        let mut next = self.head_argmax(last, &hidden[(n0 as usize - 1) * d..n0 as usize * d]);

        // New tokens are always plain text: continue the prompt's last
        // position diagonally, +1 on every axis per step.
        let mut mrope_pos = positions[positions.len() - 1].map(|p| p + 1);
        let mut cache_row = n0;
        for _ in 0..max_new_tokens {
            if eos_ids.contains(&next) {
                break;
            }
            out.push(next);
            if out.len() as u32 == max_new_tokens {
                break;
            }
            let (cos_tab, sin_tab) = self.mrope_tables(&[mrope_pos]);
            let x_row = self.embed.row(&self.reader, next);
            let hidden = self.run_shards(&x_row, 1, &cos_tab, &sin_tab, Pass::Decode { cache: &cache, pos: cache_row });
            next = self.head_argmax(last, &hidden);
            cache_row += 1;
            mrope_pos = mrope_pos.map(|p| p + 1);
        }
        out
    }

    /// Final norm + `lm_head` over ONE hidden row, on the shard that carries
    /// the head, and the argmax of the resulting logits.
    fn head_argmax(&self, last: &DeviceShard, row: &[f32]) -> u32 {
        let gpu = &last.gpu;
        let h1 = gpu.storage_init("h1", row);
        let normed = final_norm(gpu, &self.cfg, &self.final_norm_w, &h1, 1);
        let logits = lm_head_fwd(gpu, &self.lm_head_w, &normed, 1, self.cfg.hidden, self.cfg.vocab);
        argmax(&gpu.read(&logits, self.cfg.vocab as usize))
    }
}

impl Int8ThinkerInstance {
    /// The chat request shape: `messages`/`prompt` in, generated `text` out.
    ///
    /// Deliberately the same three steps `omni::caps::OmniInner::generate`
    /// takes - `last_user_text` (the shared messages-array extraction, not a
    /// second parser), `encode`, greedy generate, `decode` - so the two models
    /// answer an identical request the same way and differ only in how the
    /// weights are stored and placed.
    fn generate_chat(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let Some((tok, eos_ids)) = &self.tok else {
            return Err(format!(
                "{MODEL}: no tokenizer configured, so this model serves only the raw token-id contract \
                 (a 'generate' call with an 'ids' blob). Point BRAIN_OMNI_INT8_TOKENIZER_DIR (or \
                 BRAIN_OMNI_HF_DIR) at a directory holding tokenizer.json, or vocab.json + merges.txt."
            ));
        };
        let prompt = last_user_text(inv);
        if prompt.trim().is_empty() {
            return Err(format!("{MODEL} generate: empty prompt (need 'messages' with a user turn, or 'prompt')"));
        }
        let max_new = inv.get_i64("max_new").unwrap_or(32).clamp(1, 4096) as u32;
        let prompt_ids = tok.encode(&prompt);

        progress(Progress::step(0, max_new, "generating"));
        let t0 = std::time::Instant::now();
        let new_ids = self.generate(&prompt_ids, max_new, eos_ids);
        gpu_core::profile::stage_time("omni-int8 generate", t0);
        for shard in &self.shards {
            shard.gpu.dump_profile();
        }
        let text = tok.decode(&new_ids);
        progress(Progress::step(max_new, max_new, text.clone()));
        Ok(Outcome::new()
            .set("text", serde_json::json!(text))
            .set("tokens", serde_json::json!(new_ids))
            .blob("text", Blob::new(Media::Text, text.into_bytes())))
    }
}

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).expect("non-empty vocab")
}

impl Instance for Int8ThinkerInstance {
    /// `forward`: input blob `x` (raw LE f32 `[n, d]`, meta `{"n": n}`),
    /// output blob `hidden` (same shape) — internal/validation action, not
    /// real generation, see this module's own doc.
    ///
    /// `generate`, in either of the two shapes this module's doc describes:
    ///
    /// * **raw** - input blob `ids` (raw LE `u32` token ids), meta
    ///   `{"max_new_tokens": u32, "eos_ids": [u32]}`; output blob `ids` (raw
    ///   LE `u32`, the GENERATED tokens only, prompt excluded - matches
    ///   [`Self::generate`]'s own contract).
    /// * **chat** - no `ids` blob: `messages`/`prompt` in, `max_new` tokens,
    ///   `text` out, exactly like `brain/omni`. Requires a tokenizer.
    ///
    /// The `ids` blob is what selects between them, so an existing raw caller
    /// is unaffected by the chat shape existing.
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match action {
            "generate" if inv.get_blob("ids").is_none() => self.generate_chat(inv, _progress),
            "forward" => {
                let blob = inv.get_blob("x").ok_or_else(|| format!("{MODEL}: missing 'x' blob"))?;
                let n = blob.meta.get("n").and_then(|v| v.as_u64()).ok_or_else(|| format!("{MODEL}: 'x' blob missing meta.n"))? as u32;
                let x_host: Vec<f32> = blob.bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
                let out = self.forward(&x_host, n);
                let bytes: Vec<u8> = out.iter().flat_map(|f| f.to_le_bytes()).collect();
                Ok(Outcome::new().blob("hidden", Blob::new(Media::Bytes, bytes).with_meta(serde_json::json!({"n": n}))))
            }
            "generate" => {
                let blob = inv.get_blob("ids").ok_or_else(|| format!("{MODEL}: missing 'ids' blob"))?;
                let prompt_ids: Vec<u32> = blob.bytes.chunks_exact(4).map(|q| u32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
                let max_new_tokens = blob.meta.get("max_new_tokens").and_then(|v| v.as_u64()).ok_or_else(|| format!("{MODEL}: 'ids' blob missing meta.max_new_tokens"))? as u32;
                let eos_ids: Vec<u32> = blob
                    .meta
                    .get("eos_ids")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect())
                    .unwrap_or_default();
                let out = self.generate(&prompt_ids, max_new_tokens, &eos_ids);
                let bytes: Vec<u8> = out.iter().flat_map(|t| t.to_le_bytes()).collect();
                Ok(Outcome::new().blob("ids", Blob::new(Media::Bytes, bytes).with_meta(serde_json::json!({"n": out.len()}))))
            }
            other => Err(format!("{MODEL}: unknown action '{other}' (only 'forward'/'generate' exist)")),
        }
    }
}

/// The placement this resident committed to: which device holds which layer
/// range, and how many of ITS bytes that costs. Computed once (from the
/// checkpoint's declared shapes, no GPU) and reused by both `estimate_multi`
/// and `activate_multi`, so the two can never disagree about the split - the
/// scheduler reserves against exactly what the loader will place.
#[derive(Clone, Debug, Default)]
struct Plan {
    stages: Vec<(Device, LayerRange, u64)>,
    /// Host bytes the instance holds regardless of placement.
    host_ram: u64,
}

/// The [`ResidentModel`]/[`MultiDeviceResidentModel`] adapter: real
/// `estimate_multi` (from the checkpoint's DECLARED shapes, no GPU), real
/// `activate_multi` (streams each shard's real weights onto its card).
pub struct Int8ThinkerResident {
    pub checkpoint_path: String,
    pub cfg: MoeTextConfig,
    /// The candidate devices and each one's USABLE byte capacity - a real
    /// number the caller queried (`nvidia-smi` total minus the configured
    /// reserve), not an assumption.
    ///
    /// Capacity is carried, not just identity, because the split has to
    /// RESPECT it: an even layer split across a 24 GB and an 8 GB card fits
    /// neither the model nor reality. How many of these devices actually get
    /// used is decided by `model::shard::plan_fewest_devices` from the
    /// checkpoint's real per-layer bytes - a model that fits one card stays
    /// on one card, and one that needs three gets three.
    devices: Vec<(Device, u64)>,
    /// Memoizes the placement. Required once this model is reachable through
    /// `Executor`: `estimate_multi` runs on the DISPATCHER thread, once per
    /// scheduling round, for every queued group of this model (see
    /// `MultiDeviceResidentModel::estimate_multi`'s own doc) — it must not
    /// re-open and re-parse a 54k-tensor checkpoint header on every call, and
    /// must never panic there (a panic on the dispatcher thread takes every
    /// OTHER model on the server down with it, not just this one).
    plan: OnceLock<Plan>,
    /// Directory holding `tokenizer.json` (or `vocab.json` + `merges.txt`) for
    /// the chat request shape. Separate from `checkpoint_path` because a
    /// brain-native int8 checkpoint is a single file with no tokenizer
    /// sibling. `None` ⇒ raw token-ids contract only.
    tokenizer_dir: Option<String>,
}

impl Int8ThinkerResident {
    /// `devices` is `(device, usable bytes)` - see the field's own doc for
    /// why the capacity travels with the identity. No tokenizer: the raw
    /// token-ids contract only; see [`Self::with_tokenizer_dir`].
    pub fn new(checkpoint_path: String, cfg: MoeTextConfig, devices: Vec<(Device, u64)>) -> Int8ThinkerResident {
        Int8ThinkerResident { checkpoint_path, cfg, devices, plan: OnceLock::new(), tokenizer_dir: None }
    }

    /// Read the tokenizer for the CHAT request shape from `dir`
    /// (`tokenizer.json`, or `vocab.json` + `merges.txt`). Without it this
    /// model still loads and still serves raw token ids.
    pub fn with_tokenizer_dir(mut self, dir: Option<String>) -> Int8ThinkerResident {
        self.tokenizer_dir = dir;
        self
    }

    /// Total device bytes this checkpoint needs, whatever the split - the sum
    /// of every layer plus the head. Useful to a caller sizing budgets, and
    /// the honest answer to "will this fit at all?" when compared against the
    /// sum of the available cards.
    pub fn total_device_bytes(&self) -> Result<u64, String> {
        let reader = WeightReader::open(&self.checkpoint_path).map_err(|e| format!("{MODEL}: cannot open '{}': {e}", self.checkpoint_path))?;
        let cost = layer_cost(&reader, &self.cfg).ok_or_else(|| format!("{MODEL}: '{}' is missing tensors this model needs", self.checkpoint_path))?;
        Ok(cost.total())
    }

    /// The placement, computed once. Returns a plan naming ZERO devices
    /// (never panics) when there are no candidate devices, the checkpoint
    /// cannot be opened, or the model does not fit across the devices given -
    /// [`MultiDeviceResidentModel::estimate_multi`]'s documented "this model
    /// is unavailable" signal, which `ResidencyManager::claim_multi` turns
    /// into a clean per-job error instead of a dispatcher crash or a silently
    /// stuck queue.
    fn plan(&self) -> Plan {
        if let Some(p) = self.plan.get() {
            return p.clone();
        }
        let computed = self.plan_uncached();
        // A losing racer's freshly-computed value is simply dropped -- every
        // caller ends up seeing the SAME value (whichever `set` won), which is
        // all correctness here depends on; `plan_uncached` is a pure function
        // of `self`, so which racer wins does not matter.
        let _ = self.plan.set(computed.clone());
        computed
    }

    fn plan_uncached(&self) -> Plan {
        if self.devices.is_empty() {
            return Plan::default();
        }
        let reader = match WeightReader::open(&self.checkpoint_path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{MODEL}: cannot open '{}': {e} -- reporting zero devices so the claim fails placement instead of panicking", self.checkpoint_path);
                return Plan::default();
            }
        };
        let Some(cost) = layer_cost(&reader, &self.cfg) else {
            eprintln!("{MODEL}: '{}' is missing tensors this model loads -- reporting zero devices so the claim fails placement instead of panicking", self.checkpoint_path);
            return Plan::default();
        };
        // The embedding is host-resident ONLY on the fallback branch (a
        // mapping that cannot lend its bytes); charge it honestly rather than
        // reporting a flat 0 that would be a lie on exactly that branch.
        let host_ram = match reader.raw_words(EMBED_TENSOR) {
            Some(_) => 0,
            None => f32_resident_bytes(&reader, EMBED_TENSOR),
        };
        // `plan_fewest_devices` wants `(index, capacity)`; map back to the
        // caller's own `Device`s afterwards so a non-GPU device in the list
        // (which this model cannot use) is rejected rather than mis-indexed.
        let mut caps: Vec<(usize, u64)> = Vec::with_capacity(self.devices.len());
        for (i, &(d, cap)) in self.devices.iter().enumerate() {
            match d {
                Device::Gpu(_) => caps.push((i, cap)),
                other => {
                    eprintln!("{MODEL}: ignoring non-GPU device {other:?} (this model is GPU-only)");
                }
            }
        }
        let Some(placements) = model::shard::plan_fewest_devices(&cost, &caps) else {
            eprintln!(
                "{MODEL}: {} does not fit across the {} budgeted device(s) ({} bytes needed, {} available) -- reporting zero devices",
                self.checkpoint_path,
                caps.len(),
                cost.total(),
                caps.iter().map(|&(_, c)| c).sum::<u64>()
            );
            return Plan::default();
        };
        let stages = placements
            .iter()
            .map(|p| (self.devices[p.shard.gpu_index].0, p.shard.start..p.shard.end, p.bytes))
            .collect();
        Plan { stages, host_ram }
    }
}

impl ResidentModel for Int8ThinkerResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            MODEL,
            "Qwen3-Omni Thinker, int8 MoE experts, layer-sharded and GPU-RESIDENT across as many GPUs as its real per-layer bytes need (capacity-aware placement via model::shard). Same chat request contract as brain/omni (text only -- no audio/image/video input and no speak), but the weights stay on the cards instead of streaming from the checkpoint per token, and decode runs against a real per-layer KV cache.",
            vec![
                ActionSpec::new("forward", "internal: run the sharded MoE-bearing layers on a raw hidden-state blob"),
                // The SAME builder brain/omni's generate uses -- what makes
                // this model reachable over /v1/chat/completions and
                // /v1/messages rather than D-Bus-only. Adding the raw `ids`
                // blob keeps the original token-ids contract advertised too.
                crate::caps::chat_generate_spec(
                    "Qwen3-Omni Thinker (int8, GPU-resident): greedy text completion. Also accepts the raw contract -- an 'ids' blob of LE u32 token ids with meta max_new_tokens/eos_ids, answered with an 'ids' blob.",
                )
                .input(BlobSpec::new("ids", Media::Bytes, "optional raw prompt token ids (LE u32), meta {max_new_tokens, eos_ids}; when present it replaces messages/prompt and the reply is an 'ids' blob"))
                .output(BlobSpec::new("ids", Media::Bytes, "generated token ids (LE u32), prompt excluded -- only for a request that supplied the 'ids' blob")),
            ],
        )
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(MODEL, "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(0, 0)
    }
    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Err(format!("{MODEL}: single-device activate is not supported -- this model is multi-device only, claim it via ResidencyManager::claim_multi"))
    }
}

impl MultiDeviceResidentModel for Int8ThinkerResident {
    fn estimate_multi(&self, _key: &InstanceKey) -> MultiDeviceCost {
        let plan = self.plan();
        MultiDeviceCost::new(plan.stages.iter().map(|&(d, _, bytes)| (d, bytes)).collect(), plan.host_ram)
    }

    fn activate_multi(&self, _key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        let plan = self.plan();
        if plan.stages.is_empty() {
            return Err(format!("{MODEL}: no placement (checkpoint unreadable, or it does not fit the budgeted devices)"));
        }
        // `claim_multi` reserves against exactly the devices `estimate_multi`
        // named, so it hands back the same set. Insisting on that here (rather
        // than silently re-planning for whatever arrives) is what makes the
        // reservation and the allocation describe the same bytes.
        if devices.len() != plan.stages.len() || !devices.iter().all(|d| plan.stages.iter().any(|&(pd, _, _)| pd == *d)) {
            return Err(format!(
                "{MODEL}: activate_multi got devices {devices:?} but the plan placed {:?} -- the reservation and the load would describe different cards",
                plan.stages.iter().map(|&(d, _, _)| d).collect::<Vec<_>>()
            ));
        }

        let reader = WeightReader::open(&self.checkpoint_path).map_err(|e| format!("{MODEL}: cannot open '{}': {e}", self.checkpoint_path))?;
        let mut shards = Vec::with_capacity(plan.stages.len());
        for (dev, range, _) in &plan.stages {
            let idx = match dev {
                Device::Gpu(i) => *i,
                other => return Err(format!("{MODEL}: plan named a non-GPU device {other:?}")),
            };
            let gpu = Gpu::new_on_index(idx, thinker_pipelines())?;
            // ONE uploader for this whole card's share: the staging-reclaim
            // budget spans every tensor, which is the part that keeps a
            // multi-GB load from accruing a shadow copy of itself.
            let mut up = Uploader::new(&gpu);
            let store = ThinkerInt8Store::build(&mut up, &reader, range.clone(), &self.cfg);
            let layer_bufs = range.clone().map(|l| (l, load_layer_bufs(&mut up, &reader, l))).collect();
            shards.push(DeviceShard { gpu, range: range.clone(), store, layer_bufs });
        }
        let last = shards.last().ok_or_else(|| format!("{MODEL}: plan has zero stages"))?;
        let mut up = Uploader::new(&last.gpu);
        let final_norm_w = load_vec(&mut up, &reader, "thinker.norm.weight");
        let lm_head_w = load_mat(&mut up, &reader, "thinker.lm_head.weight");

        let embed = EmbedTable::open(&reader)?;
        if embed.hidden() != self.cfg.hidden as usize {
            return Err(format!("{MODEL}: {EMBED_TENSOR} is [_, {}] but the config says hidden={}", embed.hidden(), self.cfg.hidden));
        }
        // A tokenizer that will not load is reported and dropped, not fatal:
        // it costs one request shape, and failing the whole activation over it
        // would take away the raw contract that does work.
        let tok = self.tokenizer_dir.as_ref().and_then(|dir| match QwenBpe::from_dir(dir) {
            Ok(t) => {
                let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].into_iter().filter_map(|s| t.special_id(s)).collect();
                Some((t, eos))
            }
            Err(e) => {
                eprintln!("{MODEL}: cannot read a tokenizer from '{dir}': {e} -- serving the raw token-id contract only");
                None
            }
        });
        Ok(Box::new(Int8ThinkerInstance { cfg: self.cfg.clone(), shards, reader, embed, final_norm_w, lm_head_w, tok }))
    }
}
