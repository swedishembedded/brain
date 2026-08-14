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
//! # Multimodal input
//!
//! `generate`'s chat shape also accepts real audio/image/video, spliced in by
//! the SAME `crate::mm::build_multimodal_prompt` (+ `qwen3vl::mrope::
//! get_rope_index_multi` for the real per-axis M-RoPE positions) `brain/omni`
//! uses - not a second copy of that splicing/position logic. This is why
//! [`Int8ThinkerInstance`] carries a [`crate::config::ThinkerConfig`] rather
//! than the bare [`MoeTextConfig`] `thinker::layer_fwd` et al. actually
//! consume (`cfg.text`): `build_multimodal_prompt` needs the special media
//! token ids (`audio_token_id` etc.) that only `ThinkerConfig` carries, and
//! carrying them alongside a second hand-maintained copy on this struct is
//! exactly the kind of drift-prone duplication this codebase avoids.
//!
//! **Weight source for the vision/audio towers, decided honestly**: the real
//! int8 checkpoint DOES contain `audio.*`/`vision.*` tensors, quantized
//! (`.weight.scale` siblings present) - but `crate::mm::encode_audio`/
//! `encode_image` only know how to read PLAIN f32 tower weights (there is no
//! quantized-execution path through `qwen3asr::encoder`/`qwen3vl::encoder`,
//! only through the Thinker's own MoE experts via `model::moe::expert_fwd_i8`).
//! Building one would mean writing new quantized vision/audio kernels - real,
//! separable follow-up work, not attempted here. So this path reuses the
//! encoders EXACTLY as `brain/omni` does, reading the towers' fp32 weights
//! from a real HF checkpoint directory (`BRAIN_QWEN3OMNIMOE_HF_DIR`, the same
//! directory the tokenizer is often read from already) - see
//! [`Int8ThinkerInstance::generate_chat`]'s multimodal branch. The
//! int8-quantized `audio.*`/`vision.*` tensors already sitting in the int8
//! checkpoint are consequently dead weight today, not a blocker: a deployment
//! that wants multimodal input still needs the full HF checkpoint on disk
//! (~66 GB) alongside the small int8 one, which undercuts a "small quantized
//! deployment" story but is correct and reuses 100% of the validated fp32
//! encoder code. Teaching the encoders to read `audio.*`/`vision.*` straight
//! out of the int8 checkpoint (dequantizing with `model::int8::
//! dequantize_weight`, the same primitive [`load_mat_host`] already uses) is
//! real, scoped follow-up, not this pass's job.
//!
//! Speech output (`speak`) remains out of scope - `brain/omni` remains the
//! only path with Thinker->Talker->MTP->Code2Wav.

use std::collections::HashMap;
use std::sync::OnceLock;

use capability::blob::{decode_image, decode_video_hwc};
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

use crate::config::{MoeTextConfig, ThinkerConfig};
use crate::int8_resident::{expert_bytes, ThinkerInt8Store};
use crate::mm::{build_multimodal_prompt, MultimodalPrompt};
use crate::thinker::{final_norm, layer_decode_step, layer_fwd, lm_head_fwd, thinker_pipelines, ThinkerLayerCache, ThinkerLayerWeights};

/// `W8A16`, not `INT8`/GGUF's `Q8_0`: this checkpoint is per-output-channel
/// symmetric INT8 WEIGHT-ONLY quantization (one f32 scale per output channel,
/// MoE expert linears only -- attention/norms/embed/lm_head/vision-audio
/// towers stay fp32) with full-precision activations at compute time
/// (`crate::int8_resident`'s module doc has the exact scheme). That is
/// Weight-8bit/Activation-16bit-or-higher, the current HF/vLLM-recognized tag
/// for this class of scheme -- not GGUF's `Q8_0` (a different, 32-element
/// block-quantization format llama.cpp uses) and not a bare "INT8" (which
/// says nothing about activations staying full precision).
pub const MODEL: &str = "brain/Qwen3-Omni-30B-A3B-Instruct-W8A16";

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
/// `qwen3omnimoe::import` quantized it (`should_quantize`: rank-2, last dim a
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

/// A 1-D tensor (norm gains) - `qwen3omnimoe::import::should_quantize` never
/// quantizes rank-1 tensors, so these are always plain f32.
pub fn load_vec(up: &mut Uploader, reader: &WeightReader, name: &str) -> DeviceBuffer {
    let n = numel(reader, name) as usize;
    assert!(n > 0, "missing tensor {name}");
    up.tensor(reader, name, n).unwrap_or_else(|e| panic!("{e}"))
}

/// Build layer `l`'s REAL non-expert weights on `up`'s device, streamed from
/// `reader`. Brain-native names (`qwen3omnimoe::import::map_thinker`'s output),
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

    /// The WHOLE table as f32 (~1.2 GB at the real Thinker shape) - what
    /// [`crate::mm::build_multimodal_prompt`] needs to embed the media
    /// start/end/text tokens it wraps around each spliced block. Only ever
    /// called on a multimodal request (see [`Int8ThinkerInstance::generate_chat`]),
    /// never on the plain-text path this type exists to keep cheap for.
    /// `Host` is already materialized (built once at [`Self::open`]) so this
    /// just clones it rather than re-reading/re-dequantizing; the other two
    /// variants go through [`load_mat_host`] - the same dequant primitive
    /// [`Self::open`]'s own `Host` fallback and [`load_mat_host`]'s other
    /// callers already use, not a second implementation of it.
    fn to_host(&self, reader: &WeightReader) -> Vec<f32> {
        match self {
            EmbedTable::Host { table, .. } => table.clone(),
            EmbedTable::Plain { .. } | EmbedTable::Packed { .. } => {
                let (n, k) = logical_shape(reader, EMBED_TENSOR).expect("EmbedTable::open already validated this tensor's shape");
                load_mat_host(reader, EMBED_TENSOR, n as u32, k as u32)
            }
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
    /// The FULL Thinker config, not just the text-decoder subset most call
    /// sites here need (`cfg.text`) - carried whole so the special media
    /// token ids `crate::mm::build_multimodal_prompt` needs never live as a
    /// second, independently-maintained copy on this struct (see this
    /// module's own doc, "Multimodal input").
    cfg: ThinkerConfig,
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
    /// This checkpoint's own Jinja chat template (`crate::caps::
    /// load_chat_template`, read from the SAME directory `tok` came from --
    /// see `crate::caps::chat_prompt`'s doc for why a template-less checkpoint
    /// degrades to `last_user_text` rather than failing). Shared with the bf16
    /// `OmniInner` path (`crate::caps`) rather than a second templating
    /// implementation -- see `crate::caps::render_chat_prompt`'s doc.
    chat_template: Option<data::chat_template::ChatTemplate>,
    /// A REAL HF checkpoint directory's `WeightReader`, open ONLY to feed the
    /// vision/audio tower encoders for a multimodal `generate` request (see
    /// this module's doc, "Multimodal input", for why this is a second
    /// reader rather than reading `audio.*`/`vision.*` out of `reader`
    /// above). `None` when no `BRAIN_QWEN3OMNIMOE_HF_DIR`-equivalent was configured
    /// or it could not be opened - text-only `generate` is unaffected; an
    /// audio/image/video blob is then rejected with a clear error rather
    /// than silently ignored.
    mm_reader: Option<WeightReader>,
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
        let positions = qwen3vl::mrope::get_rope_index(&tokens, u32::MAX, &[]);
        let (cos_tab, sin_tab) = self.mrope_tables(&positions);
        self.run_shards(x_host, n, &cos_tab, &sin_tab, Pass::Batched { cache: None })
    }

    /// Sum of every shard's `Gpu::reclaim_event_count()` - test-observability
    /// for `run_shards`'s periodic-reclaim contract (see that function's
    /// doc): on Vulkan this must scale with `n_layers / FLUSH_EVERY`, not
    /// stay flat at ~1 regardless of layer count (which is what a
    /// `run_shards` that only reclaims once, at its final `gpu.read`, would
    /// show). Deliberately NOT `queue_submits`: that also counts one-off
    /// staging submits (`upload`/`zero`/`download`, one per freshly
    /// allocated scratch buffer) which scale with layer count on their own
    /// and would swamp this signal.
    pub fn total_reclaim_events(&self) -> u64 {
        self.shards.iter().map(|s| s.gpu.reclaim_event_count()).sum()
    }

    /// The M-RoPE cos/sin tables for `positions`, at this Thinker's section
    /// split / head dim / theta.
    fn mrope_tables(&self, positions: &[[u32; 3]]) -> (Vec<f32>, Vec<f32>) {
        let section: [u32; 3] = [self.cfg.text.mrope_section[0], self.cfg.text.mrope_section[1], self.cfg.text.mrope_section[2]];
        qwen3vl::mrope::mrope_tables(positions, section, self.cfg.text.head_dim, self.cfg.text.rope_theta)
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
        let d = self.cfg.text.hidden;
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
            // Each layer's attention (scores/probs/ctx) and MoE (router/gate/
            // expert scratch) buffers are dropped at the end of the layer_fwd/
            // layer_decode_step call below, but backend_vulkan's reclaim is
            // deferred (`VkOwnedBuffer::drop` only buries them; the real
            // `vkFreeMemory` happens later, inside `reclaim_dead`, itself only
            // run from `flush`, which is also a full submit+fence-wait -- see
            // `VulkanBackend::flush`'s doc). With nothing forcing a reclaim
            // between layers, a long prefill (batched, `n` large) or a long
            // decode (many steps) buries every prior layer's/step's scratch
            // for the ENTIRE pass instead of reusing it -- VRAM grows with
            // layer/step count instead of staying flat, reproduced as a real
            // ERROR_OUT_OF_DEVICE_MEMORY on 2x Tesla P40 (the shard's card
            // climbed steadily through the layer loop and OOM'd partway
            // through). Flushing every single layer fixes that but forces a
            // full fence-wait per layer, serializing what was previously
            // pipelined GPU work across the whole shard -- reproduced as a
            // >5x prefill slowdown (a 300s SSE idle timeout that never fired
            // before). Flushing every FLUSH_EVERY layers instead bounds the
            // worst case to that many layers' worth of buried scratch while
            // still letting layers within a window pipeline together.
            const FLUSH_EVERY: usize = 4;
            for (i, l) in shard.range.clone().enumerate() {
                let lb = &shard.layer_bufs[&l];
                let w = weights(lb);
                let experts8 = shard.store.layer(l);
                h = match &pass {
                    Pass::Batched { cache } => {
                        let lc = cache.map(|c| c.layer(l));
                        layer_fwd(gpu, &self.cfg.text, &w, &h, &cos, &sin, n, lc.as_ref(), Some(experts8)).0
                    }
                    Pass::Decode { cache, pos } => {
                        layer_decode_step(gpu, &self.cfg.text, &w, &cache.layer(l), &h, &cos, &sin, *pos, cache.cap, Some(experts8))
                    }
                };
                if (i + 1) % FLUSH_EVERY == 0 {
                    gpu.flush();
                }
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
    /// (matches `qwen3vl::Qwen3Vl::generate`'s own contract: `prompt_ids`/
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
        if prompt_ids.is_empty() {
            return Vec::new();
        }
        let positions = qwen3vl::mrope::get_rope_index(prompt_ids, u32::MAX, &[]);
        self.generate_with_embeds(prompt_ids, None, &positions, max_new_tokens, eos_ids)
    }

    /// [`Self::generate`], multimodal: `prompt.x_host` already has real
    /// audio/image/video embeddings spliced in
    /// (`crate::mm::build_multimodal_prompt`), and `prompt.positions` are the
    /// real per-axis M-RoPE positions that splice implies - this method does
    /// no splicing or positioning of its own, only the prefill/decode/sample
    /// loop [`Self::generate_with_embeds`] shares with the plain-text path.
    pub fn generate_multimodal(&self, prompt: &MultimodalPrompt, max_new_tokens: u32, eos_ids: &[u32]) -> Vec<u32> {
        self.generate_with_embeds(&prompt.token_ids, Some(prompt.x_host.clone()), &prompt.positions, max_new_tokens, eos_ids)
    }

    /// The shared implementation behind [`Self::generate`]/[`Self::generate_multimodal`]:
    /// `x_host_override`, when `Some`, is used as the prompt's embedding
    /// buffer verbatim (already spliced with media, `crate::mm::build_multimodal_prompt`);
    /// when `None`, it's built by a plain per-token gather from `self.embed`
    /// (the pure-text case). Mirrors `crate::generate::generate_greedy_with_embeds`'s
    /// shape exactly - same reason the bf16 path has that split: one loop
    /// serving both request shapes rather than two copies that could drift.
    fn generate_with_embeds(&self, prompt_ids: &[u32], x_host_override: Option<Vec<f32>>, positions: &[[u32; 3]], max_new_tokens: u32, eos_ids: &[u32]) -> Vec<u32> {
        let d = self.cfg.text.hidden as usize;
        let last = self.shards.last().expect("Int8ThinkerInstance has no shards");
        assert_eq!(positions.len(), prompt_ids.len(), "generate: positions/prompt_ids length mismatch");

        let mut out = Vec::with_capacity(max_new_tokens as usize);
        if prompt_ids.is_empty() || max_new_tokens == 0 {
            return out;
        }

        let n0 = prompt_ids.len() as u32;
        let cache = Int8KvCache::new(&self.shards, &self.cfg.text, n0 + max_new_tokens);

        // Prefill: the whole prompt through the stack once, filling the cache.
        let (cos_tab, sin_tab) = self.mrope_tables(positions);
        let x_host = x_host_override.unwrap_or_else(|| {
            let mut x = Vec::with_capacity(prompt_ids.len() * d);
            for &t in prompt_ids {
                x.extend_from_slice(&self.embed.row(&self.reader, t));
            }
            x
        });
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
        let normed = final_norm(gpu, &self.cfg.text, &self.final_norm_w, &h1, 1);
        let logits = lm_head_fwd(gpu, &self.lm_head_w, &normed, 1, self.cfg.text.hidden, self.cfg.text.vocab);
        let logits = gpu.read(&logits, self.cfg.text.vocab as usize);
        debug_log_top_candidates("first-generated-token", &logits);
        argmax(&logits)
    }
}

impl Int8ThinkerInstance {
    /// The chat request shape: `messages`/`prompt` in, generated `text` out.
    ///
    /// Deliberately the same steps `qwen3omnimoe::caps::GenerateAction::run`/
    /// `OmniInner::generate`/`generate_multimodal` take - `last_user_text`
    /// (the shared messages-array extraction, not a second parser), decode
    /// the same three optional blobs the SAME way (`audio::asr_caps::
    /// wav_from_blob`, `capability::blob::decode_image`/`decode_video_hwc`),
    /// splice via the SAME `crate::mm::build_multimodal_prompt` when any are
    /// present, greedy generate, `decode` - so the two Thinker-backed models
    /// answer an identical request the same way and differ only in how the
    /// weights are stored/placed and (for multimodal) which checkpoint the
    /// vision/audio towers read from (see this module's doc).
    fn generate_chat(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let Some((tok, eos_ids)) = &self.tok else {
            return Err(format!(
                "{MODEL}: no tokenizer configured, so this model serves only the raw token-id contract \
                 (a 'generate' call with an 'ids' blob). Point BRAIN_QWEN3OMNIMOE_INT8_TOKENIZER_DIR (or \
                 BRAIN_QWEN3OMNIMOE_HF_DIR) at a directory holding tokenizer.json, or vocab.json + merges.txt."
            ));
        };
        if last_user_text(inv).trim().is_empty() {
            return Err(format!("{MODEL} generate: empty prompt (need 'messages' with a user turn, or 'prompt')"));
        }
        // The real Jinja chat template when this checkpoint has one (see
        // `chat_template`'s field doc) -- `crate::caps::render_chat_prompt`,
        // the SAME implementation `qwen3omnimoe::caps::GenerateAction` uses, not a
        // second copy.
        let prompt = crate::caps::chat_prompt(self.chat_template.as_ref(), inv)?;
        let max_new = inv.get_i64("max_new").unwrap_or(32).clamp(1, 4096) as u32;

        let audio = inv.get_blob("audio").map(audio::asr_caps::wav_from_blob).transpose()?;
        let image = inv.get_blob("image").map(|_| decode_image(inv, "image")).transpose()?;
        let video = inv.get_blob("video").map(|_| decode_video_hwc(inv, "video")).transpose()?;

        progress(Progress::step(0, max_new, "generating"));
        let t0 = std::time::Instant::now();
        let new_ids = if audio.is_some() || image.is_some() || video.is_some() {
            let Some(mm_reader) = &self.mm_reader else {
                return Err(format!(
                    "{MODEL}: got audio/image/video input but no HF checkpoint directory is configured for the \
                     vision/audio towers (the int8 checkpoint's own audio.*/vision.* tensors are quantized and this \
                     model's encoders only read plain f32 -- see this module's own doc). Set BRAIN_QWEN3OMNIMOE_HF_DIR."
                ));
            };
            // Strip the chat template's own inline media-placeholder literals
            // before tokenizing -- real embeddings always splice in as a
            // whole block (never expanded in place), and tokenizing WITH the
            // placeholder literal present splits the surrounding caption
            // text into separate BPE runs at that boundary, which measurably
            // broke real-hardware generation even after stripping the
            // resulting placeholder TOKENS back out post-hoc -- see
            // `crate::mm::strip_media_placeholder_text`'s doc for the full
            // real-hardware account.
            let stripped_prompt = crate::mm::strip_media_placeholder_text(&prompt);
            let text_ids = tok.encode(&stripped_prompt);
            let embed_host = self.embed.to_host(&self.reader);
            let gpu = &self.shards[0].gpu;
            let image_ref = image.as_ref().map(|(hwc, w, h)| (hwc.as_slice(), *w, *h));
            // Splice media right after a leading system turn rather than
            // before it -- see `crate::mm::media_splice_point`'s doc for the
            // real-hardware failure (a long system prompt + media producing
            // exactly one bogus `<|im_start|>` token then immediate EOS) this closes.
            // Leading "\n" -- see the matching comment in caps.rs's call sites.
            let user_open = tok.encode("\n<|im_start|>user\n");
            let splice_at = crate::mm::media_splice_point(&stripped_prompt, &text_ids, tok.special_id("<|im_end|>"), Some(&user_open));
            let mm_prompt = build_multimodal_prompt(mm_reader, gpu, &self.cfg, &embed_host, &text_ids, audio.as_deref(), image_ref, video.as_deref(), splice_at)
                .map_err(|e| format!("{MODEL}: {e}"))?;
            if std::env::var("BRAIN_OMNI_DEBUG_LOGITS").is_ok() {
                let n = mm_prompt.token_ids.len();
                let tail = &mm_prompt.token_ids[n.saturating_sub(8)..];
                let tail_pos = &mm_prompt.positions[n.saturating_sub(8)..];
                eprintln!("{MODEL}: multimodal prompt: {n} token(s), last 8 ids={tail:?} positions={tail_pos:?}");
                eprintln!(
                    "{MODEL}: pre-assembly text_ids: {} token(s), image_token_id({}) occurs {}x, audio_token_id({}) occurs {}x",
                    text_ids.len(),
                    self.cfg.image_token_id,
                    text_ids.iter().filter(|&&t| t == self.cfg.image_token_id).count(),
                    self.cfg.audio_token_id,
                    text_ids.iter().filter(|&&t| t == self.cfg.audio_token_id).count(),
                );
                for (label, tok_id) in [("image", self.cfg.image_token_id), ("audio", self.cfg.audio_token_id)] {
                    if let Some(p) = mm_prompt.token_ids.iter().position(|&t| t == tok_id) {
                        let lo = p.saturating_sub(6);
                        let hi = (p + 6).min(n);
                        eprintln!(
                            "{MODEL}: {label} placeholder first at index {p}: ids[{lo}..{hi}]={:?} positions[{lo}..{hi}]={:?}",
                            &mm_prompt.token_ids[lo..hi],
                            &mm_prompt.positions[lo..hi]
                        );
                    }
                }
            }
            self.generate_multimodal(&mm_prompt, max_new, eos_ids)
        } else {
            let prompt_ids = tok.encode(&prompt);
            self.generate(&prompt_ids, max_new, eos_ids)
        };
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

/// Opt-in (`BRAIN_OMNI_DEBUG_LOGITS=1`) top-3 logit dump, mirroring
/// `crate::generate::debug_log_top_candidates` for the int8 path -- the
/// diagnostic this module's own real-hardware audio-splice investigation
/// used to confirm WHICH token the sharded stack's own logits actually
/// prefer at a given decode step, as opposed to guessing from decoded text
/// alone. Costs nothing when unset.
fn debug_log_top_candidates(label: &str, logits: &[f32]) {
    if std::env::var("BRAIN_OMNI_DEBUG_LOGITS").is_err() {
        return;
    }
    let mut sorted: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    eprintln!("{MODEL}: {label}: top3 (token_id, logit) = {:?}", &sorted[..3.min(sorted.len())]);
}

impl Instance for Int8ThinkerInstance {
    /// `total_reclaim_events`: sum of every shard's
    /// `Gpu::reclaim_event_count()` - see [`Self::total_reclaim_events`]'s
    /// doc. Reachable through the `dyn Instance` trait object the residency
    /// manager hands out (the concrete `Int8ThinkerInstance` isn't
    /// downcastable from there), which is also how a test exercises
    /// `run_shards`'s real periodic-reclaim contract via `Instance::run`
    /// alone.
    fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        vec![("total_reclaim_events".to_string(), serde_json::json!(self.total_reclaim_events()))]
    }

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
    pub cfg: ThinkerConfig,
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
    /// A real HF checkpoint DIRECTORY (`BRAIN_QWEN3OMNIMOE_HF_DIR`) to read the
    /// vision/audio tower weights from for multimodal `generate` requests -
    /// see this module's doc, "Multimodal input", for why: the int8
    /// checkpoint's own `audio.*`/`vision.*` tensors are quantized, and
    /// `crate::mm::encode_audio`/`encode_image` only read plain f32. `None`
    /// ⇒ `generate` still works for text, but an attached audio/image/video
    /// blob is rejected with a clear error instead of silently ignored.
    hf_dir: Option<String>,
}

impl Int8ThinkerResident {
    /// `devices` is `(device, usable bytes)` - see the field's own doc for
    /// why the capacity travels with the identity. No tokenizer, no
    /// multimodal HF dir: the raw token-ids (text-only) contract only; see
    /// [`Self::with_tokenizer_dir`]/[`Self::with_hf_dir`].
    pub fn new(checkpoint_path: String, cfg: ThinkerConfig, devices: Vec<(Device, u64)>) -> Int8ThinkerResident {
        Int8ThinkerResident { checkpoint_path, cfg, devices, plan: OnceLock::new(), tokenizer_dir: None, hf_dir: None }
    }

    /// Read the tokenizer for the CHAT request shape from `dir`
    /// (`tokenizer.json`, or `vocab.json` + `merges.txt`). Without it this
    /// model still loads and still serves raw token ids.
    pub fn with_tokenizer_dir(mut self, dir: Option<String>) -> Int8ThinkerResident {
        self.tokenizer_dir = dir;
        self
    }

    /// Read the vision/audio tower weights for multimodal `generate` requests
    /// from a real HF checkpoint directory - see this struct's `hf_dir` field
    /// doc. Without it the model still loads and still serves text-only
    /// `generate`.
    pub fn with_hf_dir(mut self, dir: Option<String>) -> Int8ThinkerResident {
        self.hf_dir = dir;
        self
    }

    /// Total device bytes this checkpoint needs, whatever the split - the sum
    /// of every layer plus the head. Useful to a caller sizing budgets, and
    /// the honest answer to "will this fit at all?" when compared against the
    /// sum of the available cards.
    pub fn total_device_bytes(&self) -> Result<u64, String> {
        let reader = WeightReader::open(&self.checkpoint_path).map_err(|e| format!("{MODEL}: cannot open '{}': {e}", self.checkpoint_path))?;
        let cost = layer_cost(&reader, &self.cfg.text).ok_or_else(|| format!("{MODEL}: '{}' is missing tensors this model needs", self.checkpoint_path))?;
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
        let Some(cost) = layer_cost(&reader, &self.cfg.text) else {
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
            "Qwen3-Omni Thinker, int8 MoE experts, layer-sharded and GPU-RESIDENT across as many GPUs as its real per-layer bytes need (capacity-aware placement via model::shard). Same chat request contract as brain/omni, including real audio/image/video input (splice via crate::mm::build_multimodal_prompt, the same code brain/omni uses -- see this module's doc for the vision/audio tower weight source), but the weights stay on the cards instead of streaming from the checkpoint per token, and decode runs against a real per-layer KV cache. Still no speak (see this module's doc).",
            vec![
                ActionSpec::new("forward", "internal: run the sharded MoE-bearing layers on a raw hidden-state blob"),
                // The SAME builder brain/omni's generate uses (chat params +
                // the three media inputs) -- what makes this model reachable
                // over /v1/chat/completions and /v1/messages with the SAME
                // multimodal contract, rather than a second hand-synced copy
                // of either. Adding the raw `ids` blob keeps the original
                // token-ids contract advertised too.
                crate::caps::with_multimodal_inputs(crate::caps::chat_generate_spec(
                    "Qwen3-Omni Thinker (int8, GPU-resident): greedy text completion, with real audio/image/video input. Also accepts the raw contract -- an 'ids' blob of LE u32 token ids with meta max_new_tokens/eos_ids, answered with an 'ids' blob.",
                ))
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
            let store = ThinkerInt8Store::build(&mut up, &reader, range.clone(), &self.cfg.text);
            let layer_bufs = range.clone().map(|l| (l, load_layer_bufs(&mut up, &reader, l))).collect();
            shards.push(DeviceShard { gpu, range: range.clone(), store, layer_bufs });
        }
        let last = shards.last().ok_or_else(|| format!("{MODEL}: plan has zero stages"))?;
        let mut up = Uploader::new(&last.gpu);
        let final_norm_w = load_vec(&mut up, &reader, "thinker.norm.weight");
        let lm_head_w = load_mat(&mut up, &reader, "thinker.lm_head.weight");

        let embed = EmbedTable::open(&reader)?;
        if embed.hidden() != self.cfg.text.hidden as usize {
            return Err(format!("{MODEL}: {EMBED_TENSOR} is [_, {}] but the config says hidden={}", embed.hidden(), self.cfg.text.hidden));
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
        // The chat template lives beside the tokenizer (tokenizer_config.json
        // or a standalone chat_template.json/chat_template.jinja, same
        // directory) -- read from the SAME `tokenizer_dir`, not a second env
        // var, so the two can never point at different checkpoints. A missing
        // template degrades to `last_user_text` (see `crate::caps::
        // chat_prompt`), same non-fatal shape as the tokenizer/mm_reader
        // degrades above -- there is no tokenizer to read a template beside
        // when `tokenizer_dir` is `None`.
        let chat_template = self.tokenizer_dir.as_ref().and_then(|dir| crate::caps::load_chat_template(dir));
        // Same non-fatal degrade as the tokenizer above: a bad/missing HF dir
        // costs multimodal input only, not the whole activation.
        let mm_reader = self.hf_dir.as_ref().and_then(|dir| match WeightReader::open_hf_dir(std::path::Path::new(dir)) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("{MODEL}: cannot open HF dir '{dir}' for multimodal input: {e} -- serving text-only generate");
                None
            }
        });
        Ok(Box::new(Int8ThinkerInstance { cfg: self.cfg.clone(), shards, reader, embed, final_norm_w, lm_head_w, tok, chat_template, mm_reader }))
    }
}
