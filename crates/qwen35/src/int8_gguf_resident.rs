// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.8-27B served INT8 and GPU-RESIDENT across as many cards as its real
//! per-layer bytes need, loaded **directly from the released Q8_0 GGUF** -
//! no fp32 intermediate file anywhere on the path.
//!
//! That last clause is the whole reason this module exists next to
//! [`crate::serve`]'s single-GPU `Engine`. `Engine` loads a brain-native
//! `.safetensors` through `checkpoint::load`, which means the deployment
//! story for a real checkpoint is "first convert 29 GB of Q8_0 into ~108 GB
//! of fp32 on disk, then load that" - a conversion most boxes that can RUN
//! this model still cannot STORE. Here the released `.gguf` IS the load
//! format: `checkpoint::gguf::MmapGguf` is a `checkpoint::TensorSource` that
//! dequantizes one tensor at a time out of the mapping, and
//! [`Qwen35::new_i8_shard`] consumes exactly that trait, re-quantizing each
//! leaf to brain's own group-wise INT8 as it uploads. Peak host use is one
//! tensor, never one model.
//!
//! # What this composes (and deliberately does not reimplement)
//!
//! * **Placement** - `model::shard::plan_fewest_devices`, the capacity-aware
//!   exact-DP contiguous layer partitioner. Nothing here knows how many cards
//!   the box has or whether they match: it supplies real per-layer byte costs
//!   ([`layer_cost`]) and takes the plan it is given. One card, two P40s, or a
//!   mixed 24/8 GB pair all work, and a model that genuinely does not fit is
//!   reported as unplaceable rather than OOMing partway through a multi-minute
//!   load.
//! * **Name mapping** - `checkpoint::remap::RemapSource` over a
//!   [`Fetch::Whole`] plan built from [`crate::gguf_import::classify`], the
//!   SAME llama.cpp-name classifier the offline `brain import` converter
//!   drives. There is no second name table here, and no dequantize/copy step
//!   of this module's own: `RemapSource` streams straight through to the
//!   inner `MmapGguf`.
//! * **Stage construction** - [`Qwen35::new_i8_shard`], one instance per card,
//!   each holding only its own `shard.start..shard.end` layers.
//! * **Decode** - `Qwen35::decode_step_stage` per stage per token, with the
//!   boundary residual host-staged from card to card (`d_model` floats, 20 KiB
//!   at this shape). Gated at tiny scale by
//!   `crate::model`'s own `two_shard_int8_decode_matches_the_whole_shard_model`,
//!   which proves the two-stage composition is bit-equal to the whole-shard
//!   model.
//! * **Prefill** - `Qwen35::prefill_chunk_stage` per stage per ROUND, the same
//!   seam widened to a whole round's `[n, d_model]` boundary block (5.2 MB at
//!   this resident's `n = 256`). Gated at tiny scale by `crate::model`'s own
//!   `two_shard_chunked_prefill_matches_token_by_token_replay`, which proves a
//!   chunked replay leaves both stages in exactly the decode state the
//!   per-token replay leaves.
//! * **Head epilogue** - `crate::stream::head_logits_on`, the same final-norm
//!   + int8-`lm_head` projection the streaming real-weight path already uses.
//!
//! # The endpoints do not live in a shard, and cannot
//!
//! Both `[vocab, d_model]` tables are 5_085_593_600 bytes as fp32 at this
//! shape. On a 24 GB Tesla P40 that is over `max_buffer_size` (~4.09 GiB) AND
//! 2.4x `max_storage_buffer_binding_size` (2047 MiB, which wgpu clamps to
//! `i32::MAX` on every backend), so an fp32 endpoint is not a thing that can
//! be allocated OR bound here - it is not a size question. Measured, not
//! assumed: the first real load of this resident died in
//! `paramstore::upload` with "needs a single 5085593600-byte buffer but this
//! device's queried max_buffer_size is 4292870144 bytes".
//!
//! So every stage is built with `embed: false, head: false` and this module
//! owns both ends:
//!
//! * the **embedding** is never uploaded and never materialized - decode needs
//!   one ROW per token, and `checkpoint::gguf::MmapGguf::tensor_range`
//!   dequantizes exactly the quant blocks that row touches, straight out of
//!   the mapping. It enters stage 0 through `run_decode_step`'s own
//!   `input_override` seam, the same seam that carries the residual between
//!   cards, so no new path exists for it.
//! * the **head** is quantized to INT8 (1.42 GB, inside both limits) by
//!   `model::int8::upload_quantized` straight from the mapping, and lives on
//!   the last stage's card.
//!
//! Both choices follow `crate::stream`'s real-weight path rather than
//! inventing anything: that module reached the identical conclusion about the
//! same two tensors on the same hardware.
//!
//! # Scope, deliberately
//!
//! * **No MTP.** The GGUF's `blk.64.*` + `nextn.*` MTP block is not imported
//!   and `cfg.mtp` is forced `false`: `Qwen35::new_impl_on` asserts that MTP
//!   implies a whole shard (the head needs `res[n_layers]` and the shared
//!   `lm_head`), which a multi-card split is not by construction.
//!   Self-speculative decode stays a single-GPU concern.
//! * **No flash-attention prefill.** A prefill round's GQA scratch is
//!   `[chunk, n_heads, pos+chunk]` twice over, the one cost that does not
//!   shrink with the dispatch count, and it is why [`MAX_PREFILL_TOKENS`] is a
//!   bounded round size rather than "the whole prompt at once". Growing this
//!   resident into genuinely long contexts needs a flash-attention prefill
//!   kernel, not a bigger constant - `paged_flash_prefill.wgsl` is that
//!   kernel, but its tiles cap `head_dim` at 128 and this model's is 256.
//!
//!   This bullet used to read "**Per-token prefill.** The prompt is replayed
//!   one token at a time through the same decode path... [that] needs the
//!   seam widened to a whole round's `[n, d_model]` boundary residual...
//!   Worth doing and not done here." It is done: `Qwen35::
//!   run_prefill_chunk_stage` is that widened seam (`run_prefill_chunk`, the
//!   whole-model M25 primitive `crate::serve::Engine` drives, is now a thin
//!   wrapper over it), and [`Qwen35GgufInstance::stack_prefill_chunk`] drives
//!   it round by round across every card. Measured on 2x Tesla P40 at a real
//!   1731-token prompt: 262.8 s (6.6 tok/s) before, 26.5 s (65.4 tok/s)
//!   after - 9.9x. What it did NOT need, contrary to that note, was `t`-sized
//!   per-stage buffers on every card: every buffer a round touches is
//!   allocated per call from `n`, so the stages stay built at `b = t = 1`.
//!   What it DID need, and that note did not predict, was bounding how many
//!   layers' worth of those per-call buffers can be in flight at once - see
//!   `Qwen35::run_prefill_chunk_stage`'s `DRAIN_EVERY_N_LAYERS`.
//! * **Per-token prefill at the Q4 tier, and only there.** A round is only
//!   worth issuing if the tier's `m > 1` kernel is a real tiled GEMM. `Q4`'s
//!   is not (`matmul_q4_dyn.wgsl`, naive by its own header), and on real
//!   weights that makes a chunked replay 7.3x SLOWER than the per-token one,
//!   so [`Qwen35GgufInstance::replay_prompt`] picks the tape by tier. Both
//!   leave identical state, so this is a cost choice and never a behavioural
//!   one. See [`MAX_PREFILL_TOKENS`] for the numbers and for the already-built
//!   kernel that would close it.
//! * **One sequence per dispatch.** Every stage is built at `b = t = 1`, so
//!   `run_batch` is the serial default - see its own doc.
//! * **Text only.** `crate::vl`'s vision front-end is not spliced in here.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::OnceLock;

use capability::{ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, ParamSpec, ParamType, Progress};
use checkpoint::gguf::MmapGguf;
use checkpoint::remap::{Fetch, RemapSource};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use data::rng::Rng;
use gguf::import::{ElemOp, Mapped};
use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use model::ops::TierPolicy;
use model::shard::{LayerBytes, Shard};
use residency::multi::{MultiDeviceCost, MultiDeviceResidentModel};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;

use crate::config::{LayerType, Qwen35Config};
use crate::model::{DecodeCaches, Qwen35};

/// Catalog id. Names the real upstream release this resident loads, exactly
/// (`https://huggingface.co/unsloth/Qwen3.8-27B-GGUF`, the `Q8_0` file) -
/// per AGENTS.md every served model is `<vendor>/<repo>[-<QUANT>]` matching
/// its upstream URL, and `brain/` is reserved for built-ins with no upstream
/// provenance (which is what `crate::caps::MODEL`, the fp32 brain-checkpoint
/// path, is).
pub const MODEL: &str = "unsloth/Qwen3.8-27B-Q8_0";

/// The environment variable naming the `.gguf` to serve - the same name the
/// existing real-checkpoint gate in [`crate::gguf_import`] already uses, so a
/// box configured for one is configured for both.
pub const GGUF_ENV: &str = "BRAIN_QWEN35_GGUF";

/// Default per-sequence `prompt + max_new` cap (also this resident's KV/GDN
/// cache capacity, since every sequence gets the whole cache).
const DEFAULT_CTX: u32 = 2048;

/// How many prompt tokens [`Qwen35GgufInstance::generate`] pushes through the
/// whole pipeline per round.
///
/// The upside is `crate::serve`'s and applies here twice over: a round
/// amortises each stage's per-layer dispatch overhead over `n` rows AND
/// replaces `n` host round trips per stage (`Qwen35::decode_step_stage` reads
/// its result back every call, and there are `n_stages` of those per token)
/// with one.
///
/// The same 256 `crate::serve::Engine` uses - but RE-MEASURED on this model
/// rather than inherited, because the first honest attempt at it did not run
/// at all. Measured on 2x Tesla P40, real Q4_K_M checkpoint at the INT8 tier,
/// 1731-token prompt; `prefill_seconds` from the instance's own metrics, peak
/// card occupancy sampled from `nvidia-smi`:
///
/// | chunk | prefill | tok/s | peak VRAM |
/// |-------|---------|-------|-----------|
/// | (per token) | 262.8 s | 6.6 | - |
/// | 64 | 36.5 s | 47.4 | 16.2 GiB |
/// | 128 | 26.6 s | 65.0 | 17.8 GiB |
/// | 192 | 26.4 s | 65.4 | 20.3 GiB |
/// | 256 | OOM | - | (> 24 GiB) |
///
/// A round's transients are not just its GQA scratch: EVERY intermediate a
/// layer allocates stays alive until something drains the queue, and a round
/// used to drain only at its terminal readback - so a stage held all ~32 of
/// its layers' worth at once, at the real 27B widths (`d_model` 5120,
/// `intermediate_size` 17408, `conv_dim` 10240). That is a DEPTH problem, not
/// a chunk-size problem, and capping the chunk is the wrong lever for it: a
/// single-card plan (the Q4 tier fits one) is 64 layers deep and overruns the
/// same budget at half the chunk.
///
/// So the fix went where the defect was - `Qwen35::run_prefill_chunk_stage`'s
/// `DRAIN_EVERY_N_LAYERS` - and the table became:
///
/// | chunk | prefill | tok/s | peak VRAM |
/// |-------|---------|-------|-----------|
/// | 256 | 26.5 s | 65.4 | 15.3 GiB |
/// | 512 | 27.6 s | 62.8 | 16.6 GiB |
///
/// 256 is now both the fastest measured round size AND cheaper in memory than
/// 64 was before the drain, and 512 is slower - so this is a real optimum
/// rather than a memory-driven compromise.
///
/// One thing that is NOT a consideration here and is in `crate::serve`: the
/// per-stage buffers a round needs are all allocated per call from `n`
/// (`Qwen35::run_prefill_chunk_stage` sizes every one of them), so a stage
/// built at `b = t = 1` carries no FIXED cost for this at all.
///
/// **This constant does not apply to a Q4 build at all** - see
/// [`Qwen35GgufInstance::replay_prompt`] and
/// `Qwen35::chunked_prefill_is_profitable`. The Q4 tier's `m > 1` GEMM
/// (`matmul_q4_dyn.wgsl`) is the naive one-thread-per-output kernel its own
/// header says it is, and a round through it costs far MORE than the same
/// tokens through `matmul_q4_gemv`'s coalesced `m = 1` path. Measured on the
/// real checkpoint, uniform Q4 on ONE P40, 1555-token prompt: **152.1 s
/// (10.2 tok/s) per token, 1108.3 s (1.4 tok/s) in rounds of 256** - a 7.3x
/// REGRESSION, where the same change on the INT8 two-card stack is a 9.9x
/// win. Same host code, same round size; the only difference is which GEMM
/// the weight tier binds.
///
/// So a Q4 build keeps the per-token replay until a tiled q4 GEMM is wired
/// into `model::ops::Ops`. `matmul_q4_dyn_reg` (128x128 register-tiled)
/// already exists and is already proven bit-identical to `matmul_q4_dyn`
/// (kernel-performance ledger, M5.5: 2.02x at `m = 32` rising to 12.56x at
/// `m = 2048`); it is simply not bound by `Ops::bind`, which is that
/// ledger's own scoped follow-up and not this module's to force. Binding it
/// is what would let this constant apply to Q4 too.
const MAX_PREFILL_TOKENS: u32 = 256;

// ------------------------------------------------------------ byte accounting

/// Bytes one sequence's DECODE state costs on the card that owns layer `l` -
/// the per-layer half of what [`ShardCaches`] allocates, charged to the same
/// stage the layer itself is charged to (which is exactly where it lands: a
/// layer's cache lives on that layer's own device, never crossing a card).
///
/// A GQA layer carries a `[cap, kv_dim]` K and V cache; a GDN layer carries a
/// fixed-size recurrent state plus a conv history, both independent of `cap`.
/// Same shapes `crate::serve`'s `Engine` allocates for its own pool.
fn layer_decode_state_bytes(cfg: &Qwen35Config, ty: LayerType, cap: u32) -> u64 {
    match ty {
        LayerType::Full => 2 * cap as u64 * cfg.kv_dim() as u64 * 4,
        LayerType::Linear => {
            let state = cfg.linear_num_value_heads as u64 * cfg.linear_key_head_dim as u64 * cfg.linear_value_head_dim as u64;
            let hist = cfg.linear_conv_dim() as u64 * cfg.linear_conv_kernel_dim.saturating_sub(1) as u64;
            (state + hist) * 4
        }
    }
}

/// Device bytes the INT8 `lm_head` occupies: `model::ops::Weight::I8`'s
/// `[n, k/4]` packed words plus its `[n, k/GROUP]` f32 scales, both 4 bytes
/// an element. 1.42 GB at the real shape - against 5.09 GB for the fp32 table
/// this replaces, which is not merely large but IMPOSSIBLE on a 24 GB P40
/// (past `max_buffer_size`, and 2.4x the 2047 MiB storage-binding limit).
fn head_i8_bytes(cfg: &Qwen35Config) -> u64 {
    let (v, d) = (cfg.vocab as u64, cfg.d_model as u64);
    v * d + v * (d / model::int8::GROUP as u64) * 4
}

/// The byte-exact per-stage cost model this resident hands to
/// `model::shard::plan_fewest_devices`, at `cap` decode positions.
///
/// * `per_layer` - the layer's weights at `tier` (`Qwen35Config::
///   layer_weight_bytes`, itself gated against what `model::ops::Weight::
///   upload` really places) plus that layer's own decode state.
/// * `embed` - **zero**. The `[vocab, d_model]` embedding is never uploaded:
///   decode only ever needs one ROW per token, so [`Qwen35GgufInstance`] reads
///   it straight out of the mapping ([`MmapGguf::tensor_range`], which decodes
///   only the quant blocks that row touches). 5.09 GB that exists neither in
///   VRAM nor in host RAM.
/// * `head` - `norm.weight` plus the INT8 `lm_head` ([`head_i8_bytes`]), both
///   held by this resident on the last stage's card rather than by the shard
///   (the shard is built `head: false`; see [`Qwen35GgufInstance::head`]).
///
/// Charging the endpoints truthfully matters more here than anywhere else: at
/// this vocab a mis-charged endpoint is several GB, i.e. the difference
/// between a plan that fits and a card that OOMs mid-load.
pub fn layer_cost(cfg: &Qwen35Config, cap: u32, tier: &TierPolicy) -> LayerBytes {
    let per_layer = cfg
        .layer_types()
        .into_iter()
        .map(|ty| cfg.layer_weight_bytes(ty, tier) + layer_decode_state_bytes(cfg, ty, cap))
        .collect();
    LayerBytes { per_layer, embed: 0, head: head_i8_bytes(cfg) + cfg.d_model as u64 * 4 }
}

// ------------------------------------------------------------ the fetch plan

/// Every brain-canonical name this GGUF offers for the MAIN decoder stack,
/// mapped to the GGUF tensor it comes from - built by running
/// [`crate::gguf_import::classify`] (the offline converter's OWN llama.cpp
/// name classifier, not a second copy) over `mg.names()`.
///
/// `Mapped::Dropped` names are skipped, which is exactly how the MTP block
/// (`blk.{n_layers}.*` and its `nextn.*` extras) excludes itself: `classify`
/// already drops it wholesale, so this resident never has to special-case it.
///
/// [`Mapped::Transformed`] is a 1:1 rename too and belongs in this map -
/// the VALUE transform it carries is applied on read by [`SsmALogFix`], which
/// checks the destination name, so the two mechanisms agree by construction
/// as long as this function does not silently drop the transformed leaf.
/// (It did, once: every `A_log` vanished from the plan the moment `classify`
/// stopped returning `Simple` for it, and the load failed by name - loudly,
/// which is the only reason that was a five-minute fix rather than a silent
/// hole.) `Mapped::Split` genuinely has no place here: it produces SEVERAL
/// destinations from one source and `Fetch::Whole` cannot express that; this
/// architecture is dense and emits none.
fn gguf_name_map(mg: &MmapGguf, cfg: &Qwen35Config) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for name in mg.names() {
        match crate::gguf_import::classify(name, cfg) {
            Mapped::Simple(brain) | Mapped::Transformed { into: brain, .. } => {
                out.insert(brain, name.clone());
            }
            Mapped::Split { .. } | Mapped::Dropped(_) => {}
        }
    }
    out
}

/// The GGUF tensor names of the three ENDPOINT tensors this resident holds
/// itself rather than inside a shard, resolved through the same
/// [`gguf_name_map`] so no llama.cpp spelling is written down twice:
/// `(embedding table, final norm, lm_head)`.
///
/// They are outside the shards because neither `[vocab, d_model]` table can
/// be an fp32 device buffer at this scale (5.09 GB, past a 24 GB P40's
/// `max_buffer_size`): the embedding is read a row at a time from the
/// mapping, and the head is quantized to INT8. See [`layer_cost`].
pub fn endpoint_names(mg: &MmapGguf, cfg: &Qwen35Config) -> Result<(String, String, String), String> {
    let available = gguf_name_map(mg, cfg);
    let get = |brain: &str| available.get(brain).cloned().ok_or_else(|| format!("{MODEL}: this GGUF offers no tensor for '{brain}'"));
    Ok((get("tok.weight")?, get("norm.weight")?, get(cfg.head_weight())?))
}

/// The `checkpoint::remap::RemapSource` fetch plan for ONE stage: exactly the
/// tensors `Qwen35::new_i8_shard` will ask for on `shard` - its own
/// `blocks.{l}.*` for `l` in `shard.start..shard.end`, plus whichever
/// endpoints `shard` declares it owns - each a 1:1 [`Fetch::Whole`] rename of
/// the GGUF tensor [`crate::gguf_import::classify`] maps to it.
///
/// This resident always builds its stages with `embed: false, head: false`
/// (see [`endpoint_names`]), so in practice the plan is layers only; the
/// function stays honest about `shard` regardless, which is what lets a test
/// pin the endpoint behaviour too.
///
/// The wanted set is [`crate::model::shard_param_list`] itself, not a
/// re-derivation of it, so a plan that satisfies this function is by
/// construction a plan `new_i8_shard` cannot find a hole in.
pub fn shard_fetch_plan(mg: &MmapGguf, cfg: &Qwen35Config, shard: &Shard) -> Result<HashMap<String, Fetch>, String> {
    let available = gguf_name_map(mg, cfg);
    let want = crate::model::shard_param_list(cfg, shard);
    let mut plan = HashMap::with_capacity(want.len());
    let mut missing = Vec::new();
    for (name, _) in &want {
        match available.get(name) {
            Some(src) => {
                plan.insert(name.clone(), Fetch::Whole(src.clone()));
            }
            None => missing.push(name.clone()),
        }
    }
    if !missing.is_empty() {
        return Err(format!("{MODEL}: this GGUF offers no tensor for {} required name(s): {}", missing.len(), missing.join(", ")));
    }
    Ok(plan)
}

/// The brain-canonical leaves whose VALUES and/or ORDER llama.cpp stores
/// differently from brain's own convention - see [`ElemOp::LnNeg`] and
/// [`GdnHeadOrder`]. Named once, so [`SsmALogFix`] and
/// [`crate::gguf_import::classify`] (the offline converter's own arm for the
/// same tensors) cannot disagree about which leaves these are. EVERY leaf a
/// GDN layer owns that is indexed by VALUE head needs the fix - found by
/// direct comparison against the FP8 checkpoint's own weights on real data:
/// each of these EIGHT gave cosine >= 0.9996 (`A_log`/`dt_bias`/
/// `conv1d.weight` exactly 1.0000000 - they are unquantized, so no Q8_0
/// rounding noise sits on top) under the SAME single transform, none under
/// any other candidate tried.
const A_LOG_LEAF: &str = "linear_attn.A_log";
const DT_BIAS_LEAF: &str = "linear_attn.dt_bias";
const CONV1D_LEAF: &str = "linear_attn.conv1d.weight";
const IN_PROJ_QKV_LEAF: &str = "linear_attn.in_proj_qkv.weight";
const IN_PROJ_A_LEAF: &str = "linear_attn.in_proj_a.weight";
const IN_PROJ_B_LEAF: &str = "linear_attn.in_proj_b.weight";
const IN_PROJ_Z_LEAF: &str = "linear_attn.in_proj_z.weight";
const OUT_PROJ_LEAF: &str = "linear_attn.out_proj.weight";

/// The value-head geometry [`GdnHeadOrder`]'s methods need, read once from
/// [`Qwen35Config`] rather than re-derived at every `with_tensor` call.
#[derive(Clone, Copy)]
struct GdnHeadOrder {
    num_k_heads: usize,
    group: usize,
    /// `linear_value_head_dim` - rows/columns per value head in the
    /// row/column-block leaves.
    head_dim: usize,
    /// `linear_key_dim` (`= num_k_heads * linear_key_head_dim`) - the q|k
    /// prefix width `conv1d.weight`/`in_proj_qkv.weight` carry before their
    /// v-portion begins. `2 * key_dim` is that prefix (q then k, each
    /// `key_dim` wide).
    key_dim: usize,
    /// `d_model` - the row width of `in_proj_qkv.weight`/`in_proj_z.weight`
    /// (and, doubling as `conv1d.weight`'s kernel width when swapped in by
    /// the caller for that leaf specifically).
    d_model: usize,
}

impl GdnHeadOrder {
    fn from_cfg(cfg: &Qwen35Config) -> GdnHeadOrder {
        GdnHeadOrder {
            num_k_heads: cfg.linear_num_key_heads as usize,
            group: cfg.linear_group() as usize,
            head_dim: cfg.linear_value_head_dim as usize,
            key_dim: cfg.linear_key_dim() as usize,
            d_model: cfg.d_model as usize,
        }
    }

    /// `src_head` = the GROUP-MAJOR position holding SUB-MAJOR value head
    /// `h`'s data (`h = s*group+g`, `s` outer/slow, `g` inner/fast; stored
    /// at `g*num_k_heads+s`). The ONE formula every leaf-shaped method below
    /// applies at its own granularity (per-element, per-row, or per-column).
    fn src_head(&self, h: usize) -> usize {
        let (s, g) = (h / self.group, h % self.group);
        g * self.num_k_heads + s
    }

    /// `[num_v_heads]` flat vector - `A_log`, `dt_bias`.
    fn degroup_heads(&self, v: &[f32]) -> Vec<f32> {
        let nvh = self.num_k_heads * self.group;
        assert_eq!(v.len(), nvh, "GdnHeadOrder::degroup_heads: length must be num_k_heads * group");
        let mut out = vec![0f32; nvh];
        for h in 0..nvh {
            out[h] = v[self.src_head(h)];
        }
        out
    }

    /// Row-major `[n_rows, row_width]`, value heads occupying `head_dim`
    /// CONSECUTIVE rows each, starting at `row_offset` - `conv1d.weight`
    /// (`row_offset = 2*key_dim`, `row_width = kernel`, `head_dim =
    /// self.head_dim`), `in_proj_qkv.weight` (`row_offset = 2*key_dim`,
    /// `row_width = d_model`, `head_dim = self.head_dim`),
    /// `in_proj_z.weight` (`row_offset = 0`, `row_width = d_model`,
    /// `head_dim = self.head_dim`), `in_proj_a.weight`/`in_proj_b.weight`
    /// (`row_offset = 0`, `row_width = d_model`, `head_dim = 1` - these
    /// project to ONE scalar per head, not a `linear_value_head_dim`-wide
    /// block, so `head_dim` is a caller-supplied PARAMETER rather than
    /// always `self.head_dim`). Rows outside the v-portion (q/k, for the
    /// leaves that have one) pass through untouched.
    fn degroup_rows(&self, v: &[f32], row_offset: usize, row_width: usize, head_dim: usize) -> Vec<f32> {
        let nvh = self.num_k_heads * self.group;
        let mut out = v.to_vec();
        for h in 0..nvh {
            let src_head = self.src_head(h);
            for r in 0..head_dim {
                let dst_row = row_offset + h * head_dim + r;
                let src_row = row_offset + src_head * head_dim + r;
                out[dst_row * row_width..(dst_row + 1) * row_width].copy_from_slice(&v[src_row * row_width..(src_row + 1) * row_width]);
            }
        }
        out
    }

    /// Row-major `[n_rows, total_cols]`, value heads occupying `head_dim`
    /// CONSECUTIVE columns each - `out_proj.weight` (`total_cols =
    /// value_dim`, value heads span the WHOLE width, no q/k prefix since
    /// this leaf's input dimension - the axis being reordered - is
    /// `value_dim` alone).
    fn degroup_cols(&self, v: &[f32], n_rows: usize, total_cols: usize) -> Vec<f32> {
        let nvh = self.num_k_heads * self.group;
        let mut out = v.to_vec();
        for row in 0..n_rows {
            for h in 0..nvh {
                let src_head = self.src_head(h);
                for c in 0..self.head_dim {
                    let dst_col = h * self.head_dim + c;
                    let src_col = src_head * self.head_dim + c;
                    out[row * total_cols + dst_col] = v[row * total_cols + src_col];
                }
            }
        }
        out
    }
}

/// A [`RemapSource`] with the non-renames this checkpoint needs applied on
/// read: `ssm_a` holds `-exp(A_log)` ([`ElemOp::LnNeg`] undoes it, applied to
/// `A_log` only), and EVERY GDN leaf indexed by value head is stored in
/// llama.cpp's GROUP-MAJOR order rather than brain's SUB-MAJOR one
/// ([`GdnHeadOrder`] undoes it, applied to all eight - order relative to
/// `LnNeg` does not matter, both act on `A_log`). [`crate::gguf_import::
/// classify`] expresses the same fixes for the offline converter, calling
/// the SAME [`ElemOp::LnNeg`] and [`GdnHeadOrder`].
///
/// Getting the LnNeg fix wrong is not a crash and not obviously wrong
/// output, it is a decay gate up to 260x too strong (`ElemOp::LnNeg`'s own
/// doc). Getting the head-order fix wrong is worse to notice: every head's
/// own decay/bias/projection is individually plausible, just applied to
/// the WRONG head's key/value state - grammatically fluent, factually
/// wrong output that degrades with context length, exactly the M21 symptom
/// this fix targets.
pub struct SsmALogFix<'a> {
    inner: RemapSource<'a>,
    order: GdnHeadOrder,
}

impl SsmALogFix<'_> {
    fn is_a_log(name: &str) -> bool {
        name.ends_with(A_LOG_LEAF)
    }
    fn is_dt_bias(name: &str) -> bool {
        name.ends_with(DT_BIAS_LEAF)
    }
    fn is_conv1d(name: &str) -> bool {
        name.ends_with(CONV1D_LEAF)
    }
    fn is_in_proj_qkv(name: &str) -> bool {
        name.ends_with(IN_PROJ_QKV_LEAF)
    }
    fn is_in_proj_a(name: &str) -> bool {
        name.ends_with(IN_PROJ_A_LEAF)
    }
    fn is_in_proj_b(name: &str) -> bool {
        name.ends_with(IN_PROJ_B_LEAF)
    }
    fn is_in_proj_z(name: &str) -> bool {
        name.ends_with(IN_PROJ_Z_LEAF)
    }
    fn is_out_proj(name: &str) -> bool {
        name.ends_with(OUT_PROJ_LEAF)
    }
    fn needs_fix(name: &str) -> bool {
        Self::is_a_log(name)
            || Self::is_dt_bias(name)
            || Self::is_conv1d(name)
            || Self::is_in_proj_qkv(name)
            || Self::is_in_proj_a(name)
            || Self::is_in_proj_b(name)
            || Self::is_in_proj_z(name)
            || Self::is_out_proj(name)
    }

    /// The one place every leaf's fix is dispatched by shape - kept apart
    /// from `with_tensor` so `with_tensor_chunks`'s whole-leaf fallback calls
    /// exactly the same code, never a second copy that could drift.
    fn fix(&self, name: &str, d: &[f32]) -> Vec<f32> {
        const KERNEL: usize = 4; // linear_conv_kernel_dim at the real shape; conv1d.weight's own row width
        if Self::is_a_log(name) {
            ElemOp::LnNeg.applied(name, &self.order.degroup_heads(d)).unwrap_or_else(|e| panic!("{MODEL}: {e}"))
        } else if Self::is_dt_bias(name) {
            self.order.degroup_heads(d)
        } else if Self::is_conv1d(name) {
            self.order.degroup_rows(d, 2 * self.order.key_dim, KERNEL, self.order.head_dim)
        } else if Self::is_in_proj_qkv(name) {
            self.order.degroup_rows(d, 2 * self.order.key_dim, self.order.d_model, self.order.head_dim)
        } else if Self::is_in_proj_a(name) || Self::is_in_proj_b(name) {
            // ONE scalar per head (alpha/beta), not a `head_dim`-wide block.
            self.order.degroup_rows(d, 0, self.order.d_model, 1)
        } else if Self::is_in_proj_z(name) {
            self.order.degroup_rows(d, 0, self.order.d_model, self.order.head_dim)
        } else if Self::is_out_proj(name) {
            let value_dim = self.order.num_k_heads * self.order.group * self.order.head_dim;
            let n_rows = d.len() / value_dim;
            self.order.degroup_cols(d, n_rows, value_dim)
        } else {
            unreachable!("SsmALogFix::fix called for a leaf needs_fix did not select: {name}")
        }
    }
}

impl checkpoint::TensorSource for SsmALogFix<'_> {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        if !Self::needs_fix(name) {
            return self.inner.with_tensor(name, f);
        }
        let mut fixed = None;
        let found = self.inner.with_tensor(name, &mut |d| fixed = Some(self.fix(name, d)));
        match (found, fixed) {
            (true, Some(v)) => {
                f(&v);
                true
            }
            _ => false,
        }
    }

    /// Never lend raw words for a transformed leaf - a zero-copy borrow
    /// would hand the caller llama.cpp's untransformed bytes and bypass the
    /// fix entirely, which is exactly the silent-wrong-weights failure this
    /// type exists to prevent.
    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        if Self::needs_fix(name) {
            return None;
        }
        self.inner.raw_words(name)
    }

    /// Same rule as [`Self::raw_words`], written explicitly rather than
    /// inherited from `RemapSource`'s forwarding: a zero-fp32 quantized-block
    /// lend for a transformed leaf would hand a caller llama.cpp's
    /// untransformed bytes, bypassing whichever fix that leaf needs exactly
    /// as an inherited `raw_words` would have. Written out even though the
    /// current default (declining) would happen to be correct here too - so
    /// this rule survives a future change to what "declining" means, rather
    /// than depending on it staying accidentally right.
    fn raw_blocks(&self, name: &str) -> Option<(checkpoint::gguf::BlockLayout, &[u8])> {
        if Self::needs_fix(name) {
            return None;
        }
        self.inner.raw_blocks(name)
    }

    /// Same rule as [`Self::raw_words`]: a transformed leaf is served whole,
    /// never streamed past the transform (every one of the eight fits well
    /// inside a single chunk at the real 27B shape - the largest,
    /// `in_proj_qkv.weight`'s v-portion, is 6144*5120 f32 = 126 MB).
    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        if Self::needs_fix(name) {
            return self.with_tensor(name, &mut |d| f(0, d));
        }
        self.inner.with_tensor_chunks(name, max_elems, f)
    }

    fn numel(&self, name: &str) -> Option<usize> {
        self.inner.numel(name)
    }
}

/// [`shard_fetch_plan`] wrapped as a live, transform-applying source over
/// `mg`, with the plan checked against every wanted tensor's declared element
/// count BEFORE a single byte is uploaded (`RemapSource::validate` reads
/// shapes only).
///
/// Validating up front is what turns a config-vs-checkpoint mismatch into one
/// named error instead of a panic dozens of gigabytes into a load.
pub fn shard_source<'a>(mg: &'a MmapGguf, cfg: &Qwen35Config, shard: &Shard) -> Result<SsmALogFix<'a>, String> {
    let plan = shard_fetch_plan(mg, cfg, shard)?;
    let src = RemapSource::new(mg, plan);
    src.validate(&crate::model::shard_param_list(cfg, shard))?;
    Ok(SsmALogFix { inner: src, order: GdnHeadOrder::from_cfg(cfg) })
}

/// [`crate::gguf_import::config_from_gguf`] with the two adjustments a
/// multi-card resident needs, in one place so the resident and its tests
/// cannot disagree:
///
/// * `mtp = false` - the importer sets `true` because IT imports the MTP
///   block; this path must not (see this module's doc).
/// * `block_size = cap` - purely descriptive here (every stage is built at
///   `t = 1`), kept truthful so a `to_json` dump of this config says what the
///   resident actually serves.
pub fn resident_config(mg: &MmapGguf, cap: u32) -> Result<Qwen35Config, String> {
    let mut cfg = crate::gguf_import::config_from_gguf(mg)?;
    cfg.mtp = false;
    cfg.block_size = cap;
    Ok(cfg)
}

// ------------------------------------------------------------ per-shard state

/// One sequence's per-layer decode state for ONE stage, indexed by ABSOLUTE
/// layer index with a size-1 dummy everywhere the stage/layer-type does not
/// apply - `crate::model::DecodeCaches`' own documented convention, and the
/// same shapes `crate::serve`'s `GdnSlot`/GQA pool allocate.
struct ShardCaches {
    gqa_k: Vec<DeviceBuffer>,
    gqa_v: Vec<DeviceBuffer>,
    gdn_state: Vec<DeviceBuffer>,
    gdn_hist: Vec<DeviceBuffer>,
    cap: u32,
}

impl ShardCaches {
    fn new(gpu: &Gpu, cfg: &Qwen35Config, shard: &Shard, cap: u32) -> ShardCaches {
        let kv = cfg.kv_dim() as u64;
        let state = cfg.linear_num_value_heads as u64 * cfg.linear_key_head_dim as u64 * cfg.linear_value_head_dim as u64;
        let hist = cfg.linear_conv_dim() as u64 * cfg.linear_conv_kernel_dim.saturating_sub(1) as u64;
        let n = cfg.n_layers as usize;
        let (mut gqa_k, mut gqa_v) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut gdn_state, mut gdn_hist) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for (l, ty) in cfg.layer_types().into_iter().enumerate() {
            let mine = shard.owns(l);
            let (k, v) = if mine && ty == LayerType::Full { (cap as u64 * kv, cap as u64 * kv) } else { (1, 1) };
            let (s, h) = if mine && ty == LayerType::Linear { (state, hist) } else { (1, 1) };
            gqa_k.push(gpu.storage(k));
            gqa_v.push(gpu.storage(v));
            gdn_state.push(gpu.storage(s));
            gdn_hist.push(gpu.storage(h));
        }
        let caches = ShardCaches { gqa_k, gqa_v, gdn_state, gdn_hist, cap };
        caches.reset(gpu);
        caches
    }

    /// Zero every GDN recurrent state / conv history for a fresh sequence.
    /// `Gpu::storage` does not guarantee zeroed memory and a fresh sequence's
    /// recurrent state MUST start at zero. The GQA caches are deliberately
    /// left alone - `gqa_decode_step` only ever reads rows `0..=pos`, so a
    /// stale row past the new sequence's length is never read (the same
    /// argument `Qwen35::reset_decode_cache` makes).
    fn reset(&self, gpu: &Gpu) {
        let clears: Vec<&DeviceBuffer> = self.gdn_state.iter().chain(self.gdn_hist.iter()).collect();
        gpu.submit(&clears, &[]);
    }

    fn view(&self) -> DecodeCaches<'_> {
        DecodeCaches {
            gqa_kcache: &self.gqa_k,
            gqa_vcache: &self.gqa_v,
            gqa_cap: self.cap,
            gdn_state: &self.gdn_state,
            gdn_hist: &self.gdn_hist,
        }
    }
}

/// One card's stage: the [`Qwen35`] instance holding that card's layer range,
/// plus this sequence's decode state on the SAME card.
///
/// The stage's `Shard` is always `embed: false, head: false` - the endpoints
/// are the instance's ([`EmbedTable`], [`Head`]), not the shard's, because
/// neither `[vocab, d_model]` table can be an fp32 device buffer at this
/// scale (see [`layer_cost`]).
struct DeviceShard {
    qwen35: Qwen35,
    caches: ShardCaches,
}

/// How this instance reads one embedding row.
///
/// The table is `[vocab, d_model]` - 5.09 GB as f32 - and decode only ever
/// needs a per-token ROW, never a GEMM. So it is neither uploaded nor
/// materialized on the host: [`MmapGguf::tensor_range`] dequantizes exactly
/// the quant blocks that row touches, straight out of the mapping. Peak
/// allocation is one row (20 KiB).
struct EmbedTable {
    /// The GGUF tensor name (`token_embd.weight` on a llama.cpp file, but
    /// resolved via [`endpoint_names`], never spelled here).
    name: String,
    d: usize,
}

/// The head epilogue this resident owns, on the LAST stage's card: the final
/// `norm.weight` and an INT8 `lm_head`, projected with
/// `crate::stream::head_logits_on` (the same implementation the streaming
/// real-weight path uses - there is exactly one "final norm then project to
/// vocab logits" in this crate).
///
/// It lives here rather than inside the last `Qwen35` shard because
/// `Qwen35::new_impl_on` would hold the head as plain fp32, which at
/// `[248320, 5120]` is 5.09 GB - past a 24 GB P40's `max_buffer_size`, so
/// such a shard cannot be built at all. Quantized it is 1.42 GB, inside both
/// that and the 2047 MiB storage-binding limit.
struct Head {
    ops: model::ops::Ops,
    norm: DeviceBuffer,
    w: model::ops::Weight,
}

// ------------------------------------------------------------ the instance

/// A built, multi-card, GGUF-resident Qwen3.8-27B.
pub struct Qwen35GgufInstance {
    cfg: Qwen35Config,
    shards: Vec<DeviceShard>,
    /// Kept open for the instance's lifetime so embedding rows can be read on
    /// demand. This is the header + mapping handle, not data.
    mg: MmapGguf,
    embed: EmbedTable,
    head: Head,
    tok: QwenBpe,
    /// Stop ids from the GGUF's own embedded tokenizer metadata plus the chat
    /// markup this model's template emits.
    eos: Vec<u32>,
    /// `prompt + max_new` ceiling for one sequence (the cache capacity every
    /// stage was built with).
    cap: u32,
    /// The last [`Self::generate`] call's real timings, surfaced through
    /// [`Instance::metrics`]. Prefill and decode are now genuinely different
    /// primitives (a bounded ROUND of tokens per pipeline pass versus one
    /// token per pass), and timing them separately is the only way to see
    /// either: a single tok/s figure over a whole request averages two rates
    /// that differ by more than an order of magnitude, and it was exactly this
    /// split that made the per-token prompt replay's real cost visible.
    /// `Cell`, because `metrics` takes `&self`; an `Instance` is owned by one
    /// worker thread at a time.
    last: Cell<Timings>,
    /// Why the last [`Self::generate`] loop ended - `"eos"`, `"length"` or
    /// `"caller"` (a stop string / cancellation seen by the streamer). Also
    /// reported through [`Instance::metrics`]: "it produced 4 tokens" is
    /// ambiguous between a model that chose to stop and a loop that gave up,
    /// and those need different investigations.
    stop: Cell<&'static str>,
}

/// One `generate` call's measured cost, reported by [`Instance::metrics`].
#[derive(Clone, Copy, Debug, Default)]
struct Timings {
    prefill_s: f64,
    prefill_tokens: u32,
    decode_s: f64,
    decode_tokens: u32,
}

impl Qwen35GgufInstance {
    /// Token `t`'s embedding row, `[d_model]`, read from the mapping.
    fn embed_row(&self, t: u32) -> Result<Vec<f32>, String> {
        self.mg
            .tensor_range(&self.embed.name, t as usize * self.embed.d, self.embed.d)
            .ok_or_else(|| format!("{MODEL}: token id {t} is outside '{}'", self.embed.name))?
    }

    /// A whole prefill round's embeddings, `[n, d_model]` in token order -
    /// [`Self::embed_row`]'s chunked form, and stage 0's `input_override` for
    /// [`Qwen35::prefill_chunk_stage`].
    ///
    /// One [`MmapGguf::tensor_range`] call per row and no more: an arbitrary
    /// set of token ids names an arbitrary set of ROWS, which are not
    /// contiguous in the `[vocab, d_model]` table, so there is no wider range
    /// to ask for. What the chunk actually saves is downstream - one host
    /// round trip per stage per ROUND rather than per token - and the gather
    /// itself stays at `embed_row`'s cost, one row's quant blocks at a time,
    /// never the 5.09 GB table.
    fn embed_rows(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(tokens.len() * self.embed.d);
        for &t in tokens {
            out.extend_from_slice(&self.embed_row(t)?);
        }
        Ok(out)
    }

    /// One token through EVERY stage in order, then the head - returning
    /// `[vocab]` logits. The residual stream crosses each card boundary
    /// host-staged (`d_model` fp32, 20 KiB at this shape), which is also how
    /// the embedding enters stage 0 and how the final hidden state reaches the
    /// head.
    ///
    /// A stage whose layer range is empty (which a capacity-driven plan may
    /// legitimately produce) needs no special case: `run_decode_step` passes
    /// its input straight through.
    fn stack_step(&self, token_id: u32, pos: u32) -> Result<Vec<f32>, String> {
        let mut carry = self.embed_row(token_id)?;
        let debug = std::env::var_os("BRAIN_QWEN35_GGUF_DEBUG").is_some();
        let mut per_stage_rms = Vec::new();
        for s in &self.shards {
            let caches = s.caches.view();
            carry = s.qwen35.decode_step_stage(token_id, pos, &caches, Some(&carry));
            if debug {
                per_stage_rms.push((carry.iter().map(|v| v * v).sum::<f32>() / carry.len() as f32).sqrt());
            }
        }
        let last = self.shards.last().expect("a plan always has at least one stage");
        let logits = crate::stream::head_logits_on(&last.qwen35.gpu, &self.head.ops, &self.cfg, &self.head.norm, &self.head.w, &carry);
        self.debug_step(pos, &per_stage_rms, &logits);
        Ok(logits)
    }

    /// **One PREFILL ROUND through every stage in order**, then the head -
    /// [`Self::stack_step`]'s multi-token sibling, and the reason a real
    /// prompt no longer costs one whole pipeline pass per token.
    ///
    /// `tokens` are `n` consecutive prompt tokens starting at absolute
    /// position `pos_start`. The boundary residual crosses each card boundary
    /// host-staged exactly as in `stack_step`, but as the round's whole
    /// `[n, d_model]` block rather than one `[d_model]` row - so the host
    /// round trips per round are `n_stages`, not `n_stages * n`, and each
    /// stage issues one dispatch shape per layer instead of `n` of them.
    ///
    /// Only the LAST row is projected to logits: the rest of the block exists
    /// to have been computed (its K/V is in the cache, its GDN state is
    /// threaded), not to be sampled from. Returns that last token's `[vocab]`
    /// logits, so a caller can treat a round exactly like the last
    /// `stack_step` of the tokens it consumed.
    fn stack_prefill_chunk(&self, tokens: &[u32], pos_start: u32) -> Result<Vec<f32>, String> {
        let d = self.cfg.d_model as usize;
        let n = tokens.len();
        assert!(n > 0, "{MODEL}: stack_prefill_chunk on an empty round");
        let mut carry = self.embed_rows(tokens)?;
        let debug = std::env::var_os("BRAIN_QWEN35_GGUF_DEBUG").is_some();
        let mut per_stage_rms = Vec::new();
        for s in &self.shards {
            let caches = s.caches.view();
            carry = s.qwen35.prefill_chunk_stage(tokens, pos_start, &caches, Some(&carry));
            if debug {
                // The round's LAST row, so the dump is directly comparable to
                // the per-token one `stack_step` prints.
                let last = &carry[(n - 1) * d..];
                per_stage_rms.push((last.iter().map(|v| v * v).sum::<f32>() / last.len() as f32).sqrt());
            }
        }
        let last = self.shards.last().expect("a plan always has at least one stage");
        let logits =
            crate::stream::head_logits_on(&last.qwen35.gpu, &self.head.ops, &self.cfg, &self.head.norm, &self.head.w, &carry[(n - 1) * d..]);
        self.debug_step(pos_start + n as u32 - 1, &per_stage_rms, &logits);
        Ok(logits)
    }

    /// **Replay a whole prompt**, leaving every stage's GQA cache and GDN
    /// state exactly where a following decode step expects them, and returning
    /// the last prompt token's `[vocab]` logits.
    ///
    /// Rounds of [`MAX_PREFILL_TOKENS`] via [`Self::stack_prefill_chunk`] when
    /// this build's weight tier makes a round profitable, one token at a time
    /// via [`Self::stack_step`] when it does not
    /// ([`Qwen35::chunked_prefill_is_profitable`] owns that judgement and its
    /// reasons; `MAX_PREFILL_TOKENS`'s doc has the numbers). The two leave
    /// IDENTICAL state either way - that equivalence is the whole contract of
    /// `Qwen35::run_prefill_chunk_stage` and is gated bit-for-bit by
    /// `crate::model`'s `two_shard_chunked_prefill_matches_token_by_token_replay`
    /// - so this is purely a cost choice, never a behavioural one.
    ///
    /// One function rather than a loop at each call site, because
    /// [`Self::generate`] and [`Self::profile_decode`] must replay a prompt
    /// the SAME way: a profiler that warms up through a different tape than
    /// the one production uses is measuring the wrong thing.
    fn replay_prompt(&self, prompt: &[u32]) -> Result<(Vec<f32>, u32), String> {
        let mut pos = 0u32;
        let mut logits = Vec::new();
        if self.shards.iter().all(|s| s.qwen35.chunked_prefill_is_profitable()) {
            for round in prompt.chunks(MAX_PREFILL_TOKENS as usize) {
                logits = self.stack_prefill_chunk(round, pos)?;
                pos += round.len() as u32;
            }
        } else {
            for &t in prompt {
                logits = self.stack_step(t, pos)?;
                pos += 1;
            }
        }
        Ok((logits, pos))
    }

    fn reset(&self) {
        for s in &self.shards {
            s.caches.reset(&s.qwen35.gpu);
        }
    }

    /// Opt-in (`BRAIN_QWEN35_GGUF_DEBUG=1`) per-step dump: the RMS of the
    /// residual leaving each card, and the top-5 `(token, logit, text)` the
    /// head prefers.
    ///
    /// This is the diagnostic that distinguishes the two ways a big sharded
    /// stack goes wrong, which decoded text alone cannot: a residual whose
    /// magnitude explodes or collapses across a card boundary (a plumbing
    /// bug) versus a healthy residual whose top logits are simply the wrong
    /// tokens (a weights bug). Costs nothing when unset.
    fn debug_step(&self, pos: u32, per_stage_rms: &[f32], logits: &[f32]) {
        if std::env::var_os("BRAIN_QWEN35_GGUF_DEBUG").is_none() {
            return;
        }
        let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let shown: Vec<String> =
            top.iter().take(5).map(|&(i, l)| format!("({i}, {l:.3}, {:?})", self.tok.decode(&[i as u32]))).collect();
        eprintln!("{MODEL}: pos {pos}: stage residual rms {per_stage_rms:?} | top5 {}", shown.join(" "));
    }

    /// Real generation: CHUNKED prefill of `prompt` (bounded rounds of
    /// [`MAX_PREFILL_TOKENS`] through [`Self::stack_prefill_chunk`], each
    /// continuing from the state the previous one left), then greedy/sampled
    /// per-token decode until `max_new` tokens or a stop id. `on_token` sees
    /// every generated id as it is produced (streaming), and returning `true`
    /// from it stops early.
    ///
    /// The two phases are deliberately different primitives: prefill knows all
    /// its tokens up front and so can fill a round, decode does not and cannot.
    /// The state each leaves is the same either way - `Qwen35::
    /// run_prefill_chunk_stage`'s own contract - so the first decode step
    /// continues from a chunked prefill exactly as it would from a
    /// token-by-token one.
    ///
    /// Returns the GENERATED ids only (prompt excluded) - the same contract
    /// `crate::sample::generate_kv` has.
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new: u32,
        temp: f32,
        top_k: usize,
        top_p: f32,
        seed: u64,
        on_token: &mut dyn FnMut(&[u32]) -> bool,
    ) -> Result<Vec<u32>, String> {
        if prompt.is_empty() {
            return Err(format!("{MODEL}: empty prompt"));
        }
        let need = prompt.len() as u64 + max_new as u64;
        if need > self.cap as u64 {
            return Err(format!("{MODEL}: prompt ({}) + max_new ({max_new}) = {need} exceeds this instance's context capacity {}", prompt.len(), self.cap));
        }
        if let Some(bad) = prompt.iter().find(|&&t| t >= self.cfg.vocab) {
            return Err(format!("{MODEL}: prompt token id {bad} is outside vocab {}", self.cfg.vocab));
        }
        self.reset();
        let mut rng = Rng::new(seed);
        let t0 = std::time::Instant::now();
        let (mut logits, mut pos) = self.replay_prompt(prompt)?;
        let prefill_s = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let mut out = Vec::with_capacity(max_new as usize);
        let mut stop = "length";
        for _ in 0..max_new {
            let next = crate::sample::sample_logits(&logits, temp, top_k, top_p, &mut rng);
            if self.eos.contains(&next) {
                stop = "eos";
                break;
            }
            out.push(next);
            if out.len() as u32 == max_new {
                break;
            }
            if on_token(&out) {
                stop = "caller";
                break;
            }
            logits = self.stack_step(next, pos)?;
            pos += 1;
        }
        self.last.set(Timings {
            prefill_s,
            prefill_tokens: prompt.len() as u32,
            decode_s: t1.elapsed().as_secs_f64(),
            decode_tokens: out.len() as u32,
        });
        self.stop.set(stop);
        Ok(out)
    }

    /// Encode text with the checkpoint's OWN embedded tokenizer, raw (no chat
    /// template). What a profiler needs to build a realistic prompt without
    /// going through `qwen3::chat`'s whole request parser.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tok.encode(text)
    }

    /// Drain every stage's queue, so a wall clock taken around a decode region
    /// measures work that has actually FINISHED rather than work that has been
    /// enqueued.
    ///
    /// [`Self::stack_step`] already ends in a `read` per stage plus the head's
    /// logits read, so in practice the queues are drained when it returns; this
    /// makes that explicit at the region boundary instead of relying on it.
    pub fn poll_wait(&self) {
        for s in &self.shards {
            s.qwen35.gpu.poll_wait();
        }
    }

    /// The `n = 1` decode loop, profiled per kernel kind.
    ///
    /// Two separate measurements, because they answer different questions and
    /// only one of them can be trusted as a cost:
    ///
    /// * `wall_s` is `steps` decode passes on the PRODUCTION flush path,
    ///   `poll_wait`-bracketed. This is the whole-pass number, and the only one
    ///   an optimization is allowed to be judged by.
    /// * `rows` is a second, separately-driven set of passes with the backend's
    ///   timestamp path armed, which runs one compute pass per dispatch and
    ///   reads back per flush. It RANKS the kernels; its total is an upper
    ///   bound on their real cost, not the cost, because each dispatch is
    ///   drained on its own and loses the overlap it would have had in the
    ///   production submit.
    ///
    /// Both regions are preceded by a warm-up that also establishes the
    /// recurrent state: `prompt` is replayed through the real stack first, so
    /// the profiled steps are steady-state decode at a realistic position, not
    /// a cold first token. `rows` merges every stage's table (the head's card
    /// is the last stage's, so it is already included) - a per-card split would
    /// invite tuning one card's share of a pipeline that runs them in series.
    ///
    /// Panics rather than reporting zero if the prompt plus the profiled steps
    /// would run past this instance's capacity: a silently truncated profile is
    /// worse than no profile.
    pub fn profile_decode(&self, prompt: &[u32], steps: u32) -> DecodeProfile {
        assert!(steps > 0, "profile_decode: steps must be > 0");
        let need = prompt.len() as u64 + 2 * steps as u64;
        assert!(need <= self.cap as u64, "profile_decode: prompt ({}) + 2*{steps} profiled steps = {need} exceeds capacity {}", prompt.len(), self.cap);
        assert!(!prompt.is_empty(), "profile_decode: needs a non-empty prompt to establish decode state");

        // Warm-up: real prompt replay, through whichever tape `generate` would
        // have used (`replay_prompt`, so a profile can never warm up through a
        // path production does not take). Establishes the GDN recurrent state
        // and the GQA cache, compiles every pipeline, and leaves `pos` where a
        // real decode step would find it. On a chunked replay the DECODE tape's
        // own pipelines are compiled by the first profiled step instead, which
        // is why the production region below is measured over `steps` passes
        // rather than one.
        self.reset();
        let (_, mut pos) = self.replay_prompt(prompt).expect("profile_decode warm-up");
        let last = prompt[prompt.len() - 1];
        self.poll_wait();

        // Region 1: the production path. The token fed back is `last` rather
        // than a sampled id - the cost of a decode step is its shapes, which do
        // not depend on WHICH token it is, and reusing one id keeps the profile
        // from wandering into an EOS.
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            self.stack_step(last, pos).expect("profile_decode production region");
            pos += 1;
        }
        self.poll_wait();
        let wall_s = t0.elapsed().as_secs_f64();

        // Region 2: the same work with timestamp queries armed. A backend that
        // cannot time kernels reports no rows rather than a table of zeros.
        let timed = self.shards.iter().all(|s| s.qwen35.gpu.set_kernel_timing(true));
        for s in &self.shards {
            s.qwen35.gpu.reset_kernel_times();
        }
        let t1 = std::time::Instant::now();
        for _ in 0..steps {
            self.stack_step(last, pos).expect("profile_decode timed region");
            pos += 1;
        }
        self.poll_wait();
        let timed_wall_s = t1.elapsed().as_secs_f64();

        let mut merged: std::collections::BTreeMap<String, (f64, u64)> = Default::default();
        if timed {
            for s in &self.shards {
                for (name, ms, calls) in s.qwen35.gpu.kernel_times().unwrap_or_default() {
                    let e = merged.entry(name).or_insert((0.0, 0));
                    e.0 += ms;
                    e.1 += calls;
                }
            }
        }
        for s in &self.shards {
            s.qwen35.gpu.set_kernel_timing(false);
        }

        let mut rows: Vec<(String, f64, u64)> = merged.into_iter().map(|(n, (ms, c))| (n, ms, c)).collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        DecodeProfile { steps, wall_s, timed_wall_s, rows }
    }
}

/// What [`Qwen35GgufInstance::profile_decode`] measured. See that method's doc
/// for why the wall clock and the table are kept apart.
#[derive(Clone, Debug)]
pub struct DecodeProfile {
    /// Decode passes in each measured region.
    pub steps: u32,
    /// Wall seconds for `steps` passes on the production flush path - the
    /// whole-pass number an optimization is judged by.
    pub wall_s: f64,
    /// Wall seconds for the same passes with per-dispatch timestamps armed.
    /// Always larger; reported so the inflation is visible rather than implied.
    pub timed_wall_s: f64,
    /// `(kernel, device ms, calls)` over every stage, descending by time.
    /// Empty when the backend cannot time individual kernels.
    pub rows: Vec<(String, f64, u64)>,
}

impl DecodeProfile {
    /// Whole-pass decode rate, tokens per second.
    pub fn tok_per_s(&self) -> f64 {
        if self.wall_s > 0.0 {
            self.steps as f64 / self.wall_s
        } else {
            0.0
        }
    }

    /// Summed device time across the table, in ms.
    pub fn device_ms(&self) -> f64 {
        self.rows.iter().map(|(_, ms, _)| ms).sum()
    }
}

impl Instance for Qwen35GgufInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if action != "generate" {
            return Err(format!("{MODEL}: unknown action '{action}' (only 'generate' exists)"));
        }
        // The SAME request parser/streamer every other Qwen-family served
        // model uses (`messages`/`prompt`/`system`/`tools`/`stop`, chat
        // markup rendering, stop-string scanning, tool-call extraction) -
        // not a second copy of that param handling. The flavor defaults to
        // this model's own template (see `crate::caps::with_template_flavor_default`).
        let inv = crate::caps::with_template_flavor_default(inv);
        let req = qwen3::chat::parse_request(&self.tok, &inv)?;
        let mut seq = qwen3::chat::SeqState::new(&req, inv.cancel.clone());
        progress(Progress::step(0, req.max_new as u32, "generating"));
        let mut stop = false;
        let ids = self.generate(
            &req.ids,
            req.max_new as u32,
            req.temp,
            req.top_k,
            req.top_p,
            req.seed,
            &mut |so_far| {
                stop = seq.advance(&self.tok, so_far, progress);
                stop
            },
        )?;
        Ok(seq.finish(&self.tok, &ids, progress))
    }

    /// Serial, and this is why: every stage of this model is built at
    /// `b = t = 1` and `Qwen35::run_decode_step` is an `n = 1` primitive, so
    /// there is no batch axis to fill - two concurrent sequences would need
    /// two independent KV/GDN cache sets AND a `b > 1` build on every card.
    /// `crate::serve::Engine`, the single-GPU path, makes exactly the same
    /// call for exactly the same reason; genuine multi-sequence batching is
    /// one change for both, not a per-resident one.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        invs.iter().enumerate().map(|(i, inv)| self.run(action, inv, &mut |p| progress(i, p))).collect()
    }

    /// The last request's real split between prompt replay and new tokens
    /// (see [`Qwen35GgufInstance::last`]), plus why the loop ended. Both are
    /// polled by the dispatcher and surface in `Executor::stats().metrics`.
    fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        let t = self.last.get();
        let rate = |tokens: u32, secs: f64| if secs > 0.0 { tokens as f64 / secs } else { 0.0 };
        vec![
            ("prefill_seconds".to_string(), json!(t.prefill_s)),
            ("prefill_tokens".to_string(), json!(t.prefill_tokens)),
            ("prefill_tok_per_s".to_string(), json!(rate(t.prefill_tokens, t.prefill_s))),
            ("decode_seconds".to_string(), json!(t.decode_s)),
            ("decode_tokens".to_string(), json!(t.decode_tokens)),
            ("decode_tok_per_s".to_string(), json!(rate(t.decode_tokens, t.decode_s))),
            ("stop_reason".to_string(), json!(self.stop.get())),
        ]
    }
}

// ------------------------------------------------------------ the resident

/// The placement this resident committed to, computed once from the GGUF
/// header alone (no GPU, no tensor data) and reused by BOTH `estimate_multi`
/// and `activate_multi` - so the bytes the scheduler reserves and the bytes
/// the loader places can never describe different cards.
#[derive(Clone, Debug, Default)]
struct Plan {
    cfg: Option<Qwen35Config>,
    /// `(device, that stage's shard, that stage's bytes)`.
    stages: Vec<(Device, Shard, u64)>,
}

/// The [`ResidentModel`] / [`MultiDeviceResidentModel`] adapter. See this
/// module's doc for what it composes.
pub struct Qwen35GgufResident {
    gguf_path: String,
    /// Candidate devices with each one's USABLE bytes - a real number the
    /// caller queried, not an assumption. Capacity travels with identity
    /// because the split has to RESPECT it (see
    /// `model::shard::plan_by_capacity`).
    devices: Vec<(Device, u64)>,
    cap: u32,
    /// The per-leaf storage tier the decoder body uploads at (M24) - the
    /// `lm_head`/embedding endpoints are unaffected (INT8 and row-mmap-read
    /// respectively, always, regardless of `tier` - see [`head_i8_bytes`]).
    /// Feeds both [`layer_cost`] (so a placement is honest about what a
    /// narrower tier actually saves) and [`Self::activate_owned`]'s
    /// [`Qwen35::new_shard_dt`] call (so the plan and the load agree on what
    /// was budgeted).
    tier: TierPolicy,
    plan: OnceLock<Plan>,
}

impl Qwen35GgufResident {
    pub fn new(gguf_path: String, devices: Vec<(Device, u64)>, cap: u32, tier: TierPolicy) -> Qwen35GgufResident {
        Qwen35GgufResident { gguf_path, devices, cap: cap.max(1), tier, plan: OnceLock::new() }
    }

    /// Per-sequence `prompt + max_new` ceiling, from `BRAIN_QWEN35_GGUF_CTX`
    /// (default [`DEFAULT_CTX`]). It is also every stage's KV/GDN cache
    /// capacity, so it is charged into [`layer_cost`] and therefore into the
    /// placement - raising it makes the model need more cards, honestly,
    /// rather than silently overrunning one.
    pub fn ctx_from_env() -> u32 {
        std::env::var("BRAIN_QWEN35_GGUF_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_CTX).max(1)
    }

    /// The per-leaf tier policy, from `BRAIN_QWEN35_GGUF_TIER`
    /// (`TierPolicy::parse`'s grammar - `"i8"`, `"q4"`, or
    /// `"q4,in_proj_a.weight=f32,in_proj_b.weight=f32"`), default uniform
    /// INT8 (unchanged behaviour). A parse failure panics rather than
    /// silently serving the wrong precision - the same rule
    /// `TierPolicy::parse` documents for itself.
    pub fn tier_from_env() -> TierPolicy {
        match std::env::var("BRAIN_QWEN35_GGUF_TIER") {
            Ok(s) => TierPolicy::parse(&s).unwrap_or_else(|e| panic!("BRAIN_QWEN35_GGUF_TIER={s:?}: {e}")),
            Err(_) => TierPolicy::uniform(Dtype::I8),
        }
    }

    /// The placement, computed once. Returns a plan naming ZERO devices -
    /// never a panic - when there are no candidate devices, the GGUF cannot
    /// be opened or understood, or the model does not fit across the devices
    /// given. That is [`MultiDeviceResidentModel::estimate_multi`]'s
    /// documented "unavailable" signal, which `ResidencyManager::claim_multi`
    /// turns into a clean per-job error instead of a dispatcher crash.
    fn plan(&self) -> Plan {
        if let Some(p) = self.plan.get() {
            return p.clone();
        }
        let computed = self.plan_uncached();
        // A losing racer's value is dropped; `plan_uncached` is a pure
        // function of `self`, so which racer wins cannot matter.
        let _ = self.plan.set(computed.clone());
        computed
    }

    fn plan_uncached(&self) -> Plan {
        if self.devices.is_empty() {
            return Plan::default();
        }
        let mg = match MmapGguf::open(&self.gguf_path) {
            Ok(mg) => mg,
            Err(e) => {
                eprintln!("{MODEL}: cannot open '{}': {e} -- reporting zero devices so the claim fails placement instead of panicking", self.gguf_path);
                return Plan::default();
            }
        };
        let cfg = match resident_config(&mg, self.cap) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{MODEL}: '{}' is not a servable Qwen3.8 GGUF: {e} -- reporting zero devices", self.gguf_path);
                return Plan::default();
            }
        };
        let cost = layer_cost(&cfg, self.cap, &self.tier);
        // `plan_fewest_devices` wants `(index into self.devices, capacity)`;
        // mapping back afterwards is what makes a non-GPU device in the list
        // (which this model cannot use) rejected rather than mis-indexed.
        let mut caps: Vec<(usize, u64)> = Vec::with_capacity(self.devices.len());
        for (i, &(d, cap)) in self.devices.iter().enumerate() {
            match d {
                Device::Gpu(_) => caps.push((i, cap)),
                other => eprintln!("{MODEL}: ignoring non-GPU device {other:?} (this model is GPU-only)"),
            }
        }
        let Some(placements) = model::shard::plan_fewest_devices(&cost, &caps) else {
            eprintln!(
                "{MODEL}: {} does not fit across the {} budgeted device(s) ({} bytes needed, {} available) -- reporting zero devices",
                self.gguf_path,
                caps.len(),
                cost.total(),
                caps.iter().map(|&(_, c)| c).sum::<u64>()
            );
            return Plan::default();
        };
        let stages = placements
            .iter()
            .map(|p| {
                let (device, _) = self.devices[p.shard.gpu_index];
                // `plan_*` indexes `caps`, whose `.0` is an index into
                // `self.devices`; the physical card is that entry's own
                // `Device::Gpu(i)`.
                let physical = match device {
                    Device::Gpu(i) => i as usize,
                    _ => unreachable!("only Gpu devices enter `caps`"),
                };
                (device, Shard { gpu_index: physical, ..p.shard.clone() }, p.bytes)
            })
            .collect();
        Plan { cfg: Some(cfg), stages }
    }

    /// The total device bytes this checkpoint needs whatever the split, and
    /// the per-layer profile behind it - for a caller sizing budgets or
    /// reporting "will this fit at all?".
    pub fn total_device_bytes(&self) -> Result<u64, String> {
        let mg = MmapGguf::open(&self.gguf_path).map_err(|e| format!("{MODEL}: cannot open '{}': {e}", self.gguf_path))?;
        let cfg = resident_config(&mg, self.cap)?;
        Ok(layer_cost(&cfg, self.cap, &self.tier).total())
    }

    /// Which layer range and how many bytes each device holds, as planned -
    /// `(device, start, end, bytes)`. Empty when unplaceable.
    pub fn placement(&self) -> Vec<(Device, usize, usize, u64)> {
        self.plan().stages.iter().map(|(d, s, b)| (*d, s.start, s.end, *b)).collect()
    }
}

/// Stop ids for this checkpoint: the GGUF's own declared EOS plus the chat
/// markup terminators, resolved through the embedded vocabulary rather than
/// hardcoded. Duplicates are dropped so `eos.contains` stays a plain scan.
fn stop_ids(tok: &QwenBpe, declared: Option<u32>) -> Vec<u32> {
    let mut ids: Vec<u32> = declared.into_iter().collect();
    for s in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tok.special_id(s) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}

impl ResidentModel for Qwen35GgufResident {
    fn manifest(&self) -> Manifest {
        let generate = ActionSpec::new(
            "generate",
            "generate text (Qwen3.8-27B dense hybrid Gated-DeltaNet/GQA decoder, INT8 weights loaded straight from the released Q8_0 GGUF, layer-sharded and resident across as many GPUs as its real per-layer bytes need; fp32 KV/GDN state; one sequence per dispatch)",
        )
        .streaming()
        .param(ParamSpec::new("prompt", ParamType::Str, "the prompt to continue (or chat message)"))
        .param(ParamSpec::new("messages", ParamType::Str, "JSON array of {role,content,...} chat turns (overrides prompt)"))
        .param(ParamSpec::new("system", ParamType::Str, "optional system prompt prepended to the chat"))
        .param(ParamSpec::new("chat", ParamType::Bool, "apply the chat template to the prompt").default(json!(true)))
        .param(ParamSpec::new("max_new", ParamType::Int, "number of new tokens to generate").default(json!(128)))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature (<= 0 = greedy)").default(json!(0.0)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k filter (0 or negative = disabled)").default(json!(40)))
        .param(ParamSpec::new("top_p", ParamType::Float, "nucleus sampling threshold (>= 1 = disabled)").default(json!(1.0)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed").default(json!(0)))
        .param(ParamSpec::new("stop", ParamType::Str, "JSON array of stop strings"))
        .param(ParamSpec::new("tools", ParamType::Str, "JSON array of tool definitions (OpenAI function-calling schema)"))
        .param(ParamSpec::new("tool_choice", ParamType::Str, "tool_choice directive, raw JSON text (\"auto\"|\"none\"|\"required\"|{\"type\":\"function\",...}); none withholds tool schemas, required/named are enforced post-generation (finish_reason \"tool_choice_unmet\" when unmet)"))
        .param(ParamSpec::new("enable_thinking", ParamType::Bool, "allow the model to emit a <think> reasoning block").default(json!(true)))
        .param(ParamSpec::new("reasoning_effort", ParamType::Str, "reasoning effort level: xhigh (default, detailed deliberation), medium (no instruction), or low (brief thinking)").default(json!("xhigh")))
        .param(ParamSpec::new("preserve_thinking", ParamType::Bool, "Qwen3.8 chat-template kwarg: keep <think> blocks from prior assistant turns in the rendered history (takes effect on the Qwen3.8-flavor render, this model's default)").default(json!(true)))
        .param(ParamSpec::new("template_flavor", ParamType::Str, "chat template flavor: qwen3.8 (default; XML <function=> tool-call payloads, prefilled open <think>, preserve_thinking kwarg) or qwen3 (JSON <tool_call> payloads, positional think framing)").default(json!("qwen3.8")))
        .output(BlobSpec::new("text", Media::Text, "the generated text"));
        Manifest::new(
            MODEL,
            "Qwen3.8-27B dense hybrid Gated-DeltaNet/GQA decoder, served INT8 and GPU-resident directly from the released Q8_0 GGUF (no fp32 intermediate on disk), layer-sharded across as many cards as its real per-layer bytes need. Text only; no MTP self-speculative decode; one sequence per dispatch.",
            vec![generate],
        )
        .with_max_context_tokens(self.cap as u64)
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(MODEL, "default")
    }

    /// Deliberately unusable: this model occupies real bytes on SEVERAL cards
    /// at once, so it has no meaningful single-device footprint. It is
    /// registered via `Executor::register_multi` and budgeted through
    /// [`MultiDeviceResidentModel::estimate_multi`]; see
    /// `crates/residency/src/multi.rs`' module doc for why a plain `register`
    /// would let it spend VRAM the scheduler never budgeted.
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(0, 0)
    }

    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Err(format!("{MODEL}: single-device activate is not supported -- this model is multi-device only, claim it via ResidencyManager::claim_multi"))
    }
}

impl MultiDeviceResidentModel for Qwen35GgufResident {
    fn estimate_multi(&self, _key: &InstanceKey) -> MultiDeviceCost {
        let plan = self.plan();
        // Host RAM: the mapping is header-only plus whatever one streamed
        // tensor costs at a time (`MmapGguf::with_tensor_chunks` decodes a
        // bounded block window, `Weight::upload` materialises one leaf), so
        // the honest steady-state figure is the largest single leaf.
        let ram = plan
            .cfg
            .as_ref()
            .map(|c| c.intermediate_size as u64 * c.d_model as u64 * 4)
            .unwrap_or(0);
        MultiDeviceCost::new(plan.stages.iter().map(|&(d, _, bytes)| (d, bytes)).collect(), ram)
    }

    fn activate_multi(&self, _key: &InstanceKey, devices: &[Device]) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(self.activate_owned(devices)?))
    }
}

impl Qwen35GgufResident {
    /// [`MultiDeviceResidentModel::activate_multi`] returning the CONCRETE
    /// instance rather than a `Box<dyn Instance>`.
    ///
    /// The trait object is the right surface for the executor, which only ever
    /// calls `run`/`metrics`. A profiler is the other kind of caller: it needs
    /// [`Qwen35GgufInstance::profile_decode`], which is not on `Instance` and
    /// should not be (nothing about serving wants a per-kernel table). So the
    /// build lives here and `activate_multi` boxes what this returns - one
    /// loader, two surfaces, rather than a second copy of the load sequence
    /// that could drift from the one being measured.
    pub fn activate_owned(&self, devices: &[Device]) -> Result<Qwen35GgufInstance, String> {
        let plan = self.plan();
        let Some(cfg) = plan.cfg.clone() else {
            return Err(format!("{MODEL}: no placement (GGUF unreadable, or it does not fit the budgeted devices)"));
        };
        if plan.stages.is_empty() {
            return Err(format!("{MODEL}: no placement (GGUF unreadable, or it does not fit the budgeted devices)"));
        }
        // `claim_multi` reserves against exactly the devices `estimate_multi`
        // named, so it hands back the same set. Insisting on that (rather
        // than silently re-planning for whatever arrives) is what makes the
        // reservation and the allocation describe the same bytes.
        if devices.len() != plan.stages.len() || !devices.iter().all(|d| plan.stages.iter().any(|(pd, _, _)| pd == d)) {
            return Err(format!(
                "{MODEL}: activate_multi got devices {devices:?} but the plan placed {:?} -- the reservation and the load would describe different cards",
                plan.stages.iter().map(|&(d, _, _)| d).collect::<Vec<_>>()
            ));
        }

        let mg = MmapGguf::open(&self.gguf_path).map_err(|e| format!("{MODEL}: cannot open '{}': {e}", self.gguf_path))?;
        let gtok = mg.tokenizer().ok_or_else(|| format!("{MODEL}: '{}' carries no embedded tokenizer (tokenizer.ggml.* KV)", self.gguf_path))?;
        let declared_eos = gtok.eos;
        let tok = QwenBpe::from_gguf(&gtok).map_err(|e| format!("{MODEL}: embedded tokenizer: {e}"))?;
        let eos = stop_ids(&tok, declared_eos);
        let (embed_name, norm_name, head_name) = endpoint_names(&mg, &cfg)?;

        let mut shards = Vec::with_capacity(plan.stages.len());
        for (_, placed, _) in &plan.stages {
            // The planner's `embed`/`head` flags say which stage is CHARGED
            // the endpoint bytes; the stage itself is built holding neither
            // (see `DeviceShard`'s own doc - this instance owns both
            // endpoints, because an fp32 `[vocab, d_model]` cannot be a
            // device buffer here at all).
            let shard = &Shard { embed: false, head: false, ..placed.clone() };
            let src = shard_source(&mg, &cfg, shard)?;
            // `b = t = 1`: this instance's own `res`/`logits`/`tokens` and
            // its per-instance decode state are sized to the smallest legal
            // value (`gdn_chunk_size(1) == 1`, so `t % chunk == 0` holds for
            // every config). The real per-sequence cache is `ShardCaches`
            // below, at `self.cap` - the same split `crate::serve::Engine`
            // makes for the same reason.
            let qwen35 = Qwen35::new_shard_dt(cfg.clone(), 1, 1, &src, shard.clone(), &self.tier);
            let caches = ShardCaches::new(&qwen35.gpu, &cfg, shard, self.cap);
            shards.push(DeviceShard { qwen35, caches });
        }

        // The head epilogue, on the LAST stage's card - see `Head`'s own doc
        // for why it is the instance's and not that shard's.
        let last = shards.last().ok_or_else(|| format!("{MODEL}: plan has zero stages"))?;
        let gpu = &last.qwen35.gpu;
        let (v, d) = (cfg.vocab as usize, cfg.d_model as usize);
        // model::int8::upload_quantized takes the fastest route `mg` can
        // serve (a Q8_0 GGUF byte repack straight from blocks, no fp32
        // anywhere, when it applies) and otherwise bounds host scratch to
        // its own row-chunked fp32 fallback - never the whole 5.09 GB table.
        let (w, s) = model::int8::upload_quantized(&mut paramstore::upload::Uploader::new(gpu), &mg, &head_name, v, d)
            .map_err(|e| format!("{MODEL}: {e}"))?;
        let head_w = model::ops::Weight::I8 { w, s, n: v as u32, k: d as u32 };
        let norm = match mg.tensor(&norm_name) {
            Some(Ok(w)) if w.len() == d => gpu.storage_init("qwen35.gguf.final_norm", &w),
            Some(Ok(w)) => return Err(format!("{MODEL}: '{norm_name}' has {} elements, expected {d}", w.len())),
            Some(Err(e)) => return Err(format!("{MODEL}: '{norm_name}': {e}")),
            None => return Err(format!("{MODEL}: '{norm_name}' vanished between planning and load")),
        };
        let head = Head {
            ops: model::ops::Ops::new(gpu.share()).map_err(|e| format!("{MODEL}: Ops::new on the head card: {e}"))?,
            norm,
            w: head_w,
        };
        let embed = EmbedTable { name: embed_name, d };

        Ok(Qwen35GgufInstance { cfg, shards, mg, embed, head, tok, eos, cap: self.cap, last: Cell::default(), stop: Cell::new("length") })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    /// The streaming loader must apply the SAME `ssm_a -> A_log` transform
    /// the offline converter does, and apply it to NOTHING else - including
    /// through the zero-copy `raw_words` path, which would otherwise hand the
    /// device llama.cpp's untransformed bytes and silently bypass it.
    ///
    /// Driven over a plain `HashMap` source (which really does lend zero-copy
    /// words), so the bypass is a reachable state in this test rather than a
    /// hypothetical.
    /// `GdnHeadOrder::degroup_heads` in isolation: `nkh=2, group=3` (`nvh=6`,
    /// `Qwen35Config::tiny()`'s own dims), hand-computed. Group-major
    /// `v[g*nkh+k]` -> sub-major `out[k*group+g]`. `head_dim`/`key_dim`/
    /// `d_model` are unused by this method - dummy values.
    #[test]
    fn degroup_heads_matches_the_hand_computed_transpose() {
        let order = GdnHeadOrder { num_k_heads: 2, group: 3, head_dim: 0, key_dim: 0, d_model: 0 };
        let v = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]; // group-major: g0=[10,20] g1=[30,40] g2=[50,60]
        let got = order.degroup_heads(&v);
        assert_eq!(got, vec![10.0, 30.0, 50.0, 20.0, 40.0, 60.0], "k=0's 3 repeats first (10,30,50), then k=1's (20,40,60)");
    }

    /// `GdnHeadOrder::degroup_rows` at `row_offset=0` (the `in_proj_z.weight`
    /// shape): `nkh=2, group=2` (`nvh=4`), `head_dim=2` rows/head,
    /// `row_width=1` (one column, so each "row" is a single number - keeps
    /// the hand computation legible). Group-major head order is
    /// `[0,2,1,3]` (`src_head(h)`: h=0->0, h=1->2, h=2->1, h=3->3), so head
    /// h's 2-row block should read from `src_head(h)`'s 2-row block.
    #[test]
    fn degroup_rows_at_zero_offset_matches_the_hand_computed_block_permutation() {
        let order = GdnHeadOrder { num_k_heads: 2, group: 2, head_dim: 2, key_dim: 0, d_model: 0 };
        assert_eq!([order.src_head(0), order.src_head(1), order.src_head(2), order.src_head(3)], [0, 2, 1, 3]);
        // 4 heads x 2 rows/head x 1 col: row r holds value 10*r.
        let v: Vec<f32> = (0..8).map(|r| 10.0 * r as f32).collect();
        let got = order.degroup_rows(&v, 0, 1, order.head_dim);
        // head 0 (rows 0,1) <- src_head 0 (rows 0,1): [0,10]
        // head 1 (rows 2,3) <- src_head 2 (rows 4,5): [40,50]
        // head 2 (rows 4,5) <- src_head 1 (rows 2,3): [20,30]
        // head 3 (rows 6,7) <- src_head 3 (rows 6,7): [60,70]
        assert_eq!(got, vec![0.0, 10.0, 40.0, 50.0, 20.0, 30.0, 60.0, 70.0]);
    }

    /// `GdnHeadOrder::degroup_rows` with a NON-ZERO `row_offset` (the
    /// `conv1d.weight`/`in_proj_qkv.weight` shape): rows before the offset
    /// (a q/k prefix) must pass through completely untouched.
    #[test]
    fn degroup_rows_leaves_the_prefix_before_row_offset_untouched() {
        let order = GdnHeadOrder { num_k_heads: 2, group: 2, head_dim: 2, key_dim: 0, d_model: 0 };
        // 3 prefix rows + 4 heads x 2 rows/head, 1 col.
        let v: Vec<f32> = (0..11).map(|r| 10.0 * r as f32).collect();
        let got = order.degroup_rows(&v, 3, 1, order.head_dim);
        assert_eq!(&got[0..3], &v[0..3], "the prefix must be untouched");
        assert_eq!(&got[3..], &[30.0, 40.0, 70.0, 80.0, 50.0, 60.0, 90.0, 100.0][..], "the v-portion must be block-permuted the same way as the zero-offset case");
    }

    /// `GdnHeadOrder::degroup_rows` at `head_dim=1` (the `in_proj_a.weight`/
    /// `in_proj_b.weight` shape - ONE scalar per head, not a `self.head_dim`
    /// -wide block): `nkh=2, group=2` (`nvh=4`), same group-major head order
    /// `[0,2,1,3]` as the block-shaped tests above, but each head is exactly
    /// one row.
    #[test]
    fn degroup_rows_at_head_dim_one_matches_the_hand_computed_row_permutation() {
        let order = GdnHeadOrder { num_k_heads: 2, group: 2, head_dim: 128, key_dim: 0, d_model: 0 };
        // 4 heads x 1 row/head, 1 col: row r holds value 10*r.
        let v: Vec<f32> = (0..4).map(|r| 10.0 * r as f32).collect();
        let got = order.degroup_rows(&v, 0, 1, 1);
        // head 0 <- src_head 0: 0; head 1 <- src_head 2: 20; head 2 <- src_head 1: 10; head 3 <- src_head 3: 30.
        assert_eq!(got, vec![0.0, 20.0, 10.0, 30.0], "head_dim=1 must ignore self.head_dim (128 here) entirely and use the caller-supplied value");
    }

    /// `GdnHeadOrder::degroup_cols` (the `out_proj.weight` shape): the SAME
    /// permutation, applied per COLUMN block instead of per row, repeated
    /// identically for every row of a multi-row matrix.
    #[test]
    fn degroup_cols_matches_the_hand_computed_block_permutation_per_row() {
        let order = GdnHeadOrder { num_k_heads: 2, group: 2, head_dim: 2, key_dim: 0, d_model: 0 };
        // 2 rows x 8 cols (4 heads x 2 cols/head).
        let v: Vec<f32> = vec![
            0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, // row 0
            100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, // row 1
        ];
        let got = order.degroup_cols(&v, 2, 8);
        assert_eq!(&got[0..8], &[0.0, 10.0, 40.0, 50.0, 20.0, 30.0, 60.0, 70.0][..], "row 0: same block permutation as the row-shaped case");
        assert_eq!(&got[8..16], &[100.0, 110.0, 140.0, 150.0, 120.0, 130.0, 160.0, 170.0][..], "row 1: identical permutation, independently applied");
    }

    #[test]
    fn the_streaming_loader_fixes_a_log_and_dt_bias_and_only_those() {
        use checkpoint::TensorSource;
        let (nkh, group) = (2usize, 3usize);
        // Sub-major TARGET order (what brain/FP8 want): A_log[p], dt_bias[p].
        let a_log_target = [-5.5f32, -3.2, -1.1, -2.0, -4.0, -0.5];
        let dt_bias_target = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        // The group-major ON-DISK layout llama.cpp stores: q = g*nkh+k for
        // sub-major position p = k*group+g. `ssm_a` additionally holds
        // -exp(A_log).
        let regroup = |target: &[f32; 6], transform: fn(f32) -> f32| {
            let mut stored = vec![0f32; 6];
            for (p, &t) in target.iter().enumerate() {
                let (k, g) = (p / group, p % group);
                stored[g * nkh + k] = transform(t);
            }
            stored
        };
        let a_log_stored = regroup(&a_log_target, |x| -x.exp());
        let dt_bias_stored = regroup(&dt_bias_target, |x| x);
        let untouched = vec![0.25f32, -0.5, 2.0];

        let inner: HashMap<String, Vec<f32>> = [
            ("blocks.0.linear_attn.A_log".to_string(), a_log_stored),
            ("blocks.0.linear_attn.dt_bias".to_string(), dt_bias_stored),
            ("blocks.0.linear_attn.norm.weight".to_string(), untouched.clone()),
        ]
        .into_iter()
        .collect();
        let plan: HashMap<String, Fetch> = inner.keys().map(|k| (k.clone(), Fetch::Whole(k.clone()))).collect();
        let order = GdnHeadOrder { num_k_heads: nkh, group, head_dim: 0, key_dim: 0, d_model: 0 };
        let src = SsmALogFix { inner: RemapSource::new(&inner, plan), order };

        let mut got_a_log = Vec::new();
        assert!(src.with_tensor("blocks.0.linear_attn.A_log", &mut |d| got_a_log = d.to_vec()));
        for (g, want) in got_a_log.iter().zip(a_log_target) {
            assert!((g - want).abs() < 1e-5, "A_log must be ln(-ssm_a) AND degrouped: got {g}, want {want}");
        }
        assert!(src.raw_words("blocks.0.linear_attn.A_log").is_none(), "a transformed leaf must never be lent zero-copy");
        let mut chunked = Vec::new();
        assert!(src.with_tensor_chunks("blocks.0.linear_attn.A_log", 1, &mut |_, d| chunked.extend_from_slice(d)));
        assert_eq!(chunked, got_a_log, "the chunked path must deliver the transformed values too");

        let mut got_dt_bias = Vec::new();
        assert!(src.with_tensor("blocks.0.linear_attn.dt_bias", &mut |d| got_dt_bias = d.to_vec()));
        assert_eq!(got_dt_bias, dt_bias_target, "dt_bias must be degrouped (no LnNeg - only A_log has that value transform)");
        assert!(src.raw_words("blocks.0.linear_attn.dt_bias").is_none(), "dt_bias is now also transformed -- must never be lent zero-copy");

        // Everything else passes through untouched, zero-copy included.
        let mut other = Vec::new();
        assert!(src.with_tensor("blocks.0.linear_attn.norm.weight", &mut |d| other = d.to_vec()));
        assert_eq!(other, untouched, "only A_log and dt_bias are transformed");
        assert!(src.raw_words("blocks.0.linear_attn.norm.weight").is_some(), "an untransformed leaf keeps its zero-copy path");
        assert_eq!(src.numel("blocks.0.linear_attn.A_log"), Some(6));
    }

    /// [`SsmALogFix::raw_blocks`]'s explicit refusal, over a REAL quantized
    /// GGUF source so the bypass it prevents is reachable rather than
    /// hypothetical - the same design note `the_streaming_loader_untransforms_
    /// a_log_and_only_a_log` makes about `raw_words`, extended to the
    /// zero-fp32 block path `raw_blocks` opens.
    #[test]
    fn raw_blocks_never_lends_the_untransformed_a_log_blocks() {
        use checkpoint::gguf_write::{write, TensorOut};
        use checkpoint::TensorSource;

        // One Q8_0 block (32 elements) each, built through the real encoder
        // rather than hand-assembled bytes.
        let a_log_block = checkpoint::quant::quantize_par(checkpoint::gguf::TYPE_Q8_0, &[1.0f32; 32]).unwrap();
        let dt_bias_block = checkpoint::quant::quantize_par(checkpoint::gguf::TYPE_Q8_0, &[2.0f32; 32]).unwrap();
        let norm_block = checkpoint::quant::quantize_par(checkpoint::gguf::TYPE_Q8_0, &[3.0f32; 32]).unwrap();
        let path = std::env::temp_dir().join(format!("brain-qwen35-ssmalogfix-rawblocks-{}.gguf", std::process::id()));
        let path = path.to_str().unwrap().to_string();
        write(
            &path,
            &[],
            &[
                TensorOut { name: "blocks.0.linear_attn.A_log".to_string(), shape: vec![32], ty: checkpoint::gguf::TYPE_Q8_0, data: a_log_block },
                TensorOut { name: "blocks.0.linear_attn.dt_bias".to_string(), shape: vec![32], ty: checkpoint::gguf::TYPE_Q8_0, data: dt_bias_block },
                TensorOut { name: "blocks.0.linear_attn.norm.weight".to_string(), shape: vec![32], ty: checkpoint::gguf::TYPE_Q8_0, data: norm_block },
            ],
            32,
        )
        .unwrap();
        let mg = MmapGguf::open(&path).unwrap();
        let names = vec![
            "blocks.0.linear_attn.A_log".to_string(),
            "blocks.0.linear_attn.dt_bias".to_string(),
            "blocks.0.linear_attn.norm.weight".to_string(),
        ];
        let plan: HashMap<String, Fetch> = names.iter().map(|n| (n.clone(), Fetch::Whole(n.clone()))).collect();
        let order = GdnHeadOrder { num_k_heads: 2, group: 2, head_dim: 2, key_dim: 0, d_model: 0 };
        let src = SsmALogFix { inner: RemapSource::new(&mg, plan), order };

        assert!(
            src.raw_blocks("blocks.0.linear_attn.A_log").is_none(),
            "a zero-fp32 block lend for the transformed leaf would bypass ElemOp::LnNeg entirely"
        );
        assert!(src.raw_blocks("blocks.0.linear_attn.dt_bias").is_none(), "dt_bias is now also transformed -- must never be lent zero-copy blocks either");
        assert!(src.raw_blocks("blocks.0.linear_attn.norm.weight").is_some(), "an untransformed leaf keeps its zero-fp32 block path");

        std::fs::remove_file(&path).ok();
    }

    /// The endpoints are charged for what this resident ACTUALLY places, and
    /// what it places is what a 24 GB P40 can hold.
    ///
    /// Both `[vocab, d_model]` tables are 5_085_593_600 bytes as fp32 -
    /// simultaneously over that card's `max_buffer_size` (~4.09 GiB) and 2.4x
    /// its 2047 MiB storage-BINDING limit, so "just allocate it" is not an
    /// option this cost model may describe. The embedding is therefore not on
    /// the card at all (row-at-a-time out of the mapping ⇒ 0 bytes) and the
    /// head is INT8 (1.42 GB, inside both limits). A cost model that charged
    /// either at the fp32 rate would plan a split that cannot load.
    #[test]
    fn the_endpoints_are_charged_for_what_is_really_placed() {
        let cfg = Qwen35Config::qwen38_27b();
        let cost = layer_cost(&cfg, 2048, &TierPolicy::uniform(Dtype::I8));
        let fp32_table = cfg.vocab as u64 * cfg.d_model as u64 * 4;
        assert_eq!(fp32_table, 5_085_593_600, "the fp32 table this resident refuses to place");
        assert_eq!(cost.embed, 0, "the embedding is read from the mapping a row at a time, never uploaded");
        assert_eq!(cost.head, head_i8_bytes(&cfg) + cfg.d_model as u64 * 4, "int8 lm_head + fp32 norm.weight");
        assert!(cost.head < fp32_table / 3, "the int8 head must be far smaller than the fp32 table: {}", cost.head);
        // Inside the 2047 MiB storage-buffer BINDING limit, which is the
        // constraint that actually decides whether this can run at all.
        assert!(head_i8_bytes(&cfg) < 2047 << 20, "the packed head must be bindable: {}", head_i8_bytes(&cfg));
    }

    /// A layer's decode state is charged to the stage that owns the layer -
    /// without it a placement would under-budget every card by the whole KV
    /// pool. GQA layers scale with context, GDN layers do not.
    #[test]
    fn per_layer_cost_includes_this_sequences_decode_state() {
        let cfg = Qwen35Config::qwen38_27b();
        let i8 = TierPolicy::uniform(Dtype::I8);
        let small = layer_cost(&cfg, 128, &i8);
        let large = layer_cost(&cfg, 8192, &i8);
        let types = cfg.layer_types();
        let gqa = types.iter().position(|t| *t == LayerType::Full).unwrap();
        let gdn = types.iter().position(|t| *t == LayerType::Linear).unwrap();
        assert!(small.per_layer[gqa] > cfg.layer_weight_bytes(LayerType::Full, &i8), "a GQA layer must be charged its KV cache");
        assert!(small.per_layer[gdn] > cfg.layer_weight_bytes(LayerType::Linear, &i8), "a GDN layer must be charged its recurrent state");
        assert!(large.per_layer[gqa] > small.per_layer[gqa], "GQA cache scales with context");
        assert_eq!(large.per_layer[gdn], small.per_layer[gdn], "GDN state is O(1) in context, not O(T)");
    }

    /// The real model at the real shape does NOT fit one 24 GB P40 and DOES
    /// fit two - the whole reason this resident exists. Pure arithmetic, no
    /// GPU and no checkpoint: it is a property of the published dims.
    #[test]
    fn the_real_model_needs_two_24gb_cards_and_fits_them() {
        let cfg = Qwen35Config::qwen38_27b();
        let cost = layer_cost(&cfg, 2048, &TierPolicy::uniform(Dtype::I8));
        let p40 = 24 * GB; // 24 GiB usable, i.e. a card with no reserve at all
        assert!(cost.total() > p40, "if it fitted one card this resident would be pointless: {} bytes", cost.total());
        assert!(model::shard::plan_by_capacity(&cost, &[(0, p40)]).is_none(), "one card must be reported infeasible, not planned");
        let two = model::shard::plan_fewest_devices(&cost, &[(0, p40), (1, p40)]).expect("two 24 GiB cards must hold it");
        assert_eq!(two.len(), 2);
        assert_eq!(two[0].shard.start, 0);
        assert!(two[0].shard.embed && !two[0].shard.head);
        assert_eq!(two[1].shard.end, cfg.n_layers as usize);
        assert!(two[1].shard.head && !two[1].shard.embed);
        assert_eq!(two[0].shard.end, two[1].shard.start, "contiguous, no gap");
        for p in &two {
            assert!(p.bytes <= p40, "stage of {} bytes overruns a {p40}-byte card", p.bytes);
        }
    }

    /// M24's headline capacity claim, as a standing gate rather than a
    /// one-off calculation: a per-leaf tier that puts the bulk MLP
    /// projections at Q4 while holding the two GDN state-sensitive gates
    /// (`in_proj_a`/`in_proj_b`, the decay/beta projections - ~94 MB total
    /// across 48 GDN layers, essentially free to keep at full precision)
    /// FITS ONE 24 GB P40, unlike the uniform-INT8 tier above which needs
    /// two. Pure arithmetic, no GPU and no checkpoint - if this regresses,
    /// the single-card cascade the M24 plan is built on (no cross-card
    /// residual handoff, MTP legal again since a whole shard fits one card,
    /// a second card free for a concurrent sequence) silently stops being
    /// true.
    #[test]
    fn a_q4_mlp_tier_with_gdn_gates_held_at_f32_fits_one_24gb_card() {
        let cfg = Qwen35Config::qwen38_27b();
        let policy_c = TierPolicy::uniform(Dtype::Q4).with(&["in_proj_a.weight", "in_proj_b.weight"], Dtype::F32);
        let cost = layer_cost(&cfg, 2048, &policy_c);
        let i8_cost = layer_cost(&cfg, 2048, &TierPolicy::uniform(Dtype::I8));
        assert!(
            cost.total() < i8_cost.total(),
            "a narrower tier must cost fewer bytes than uniform I8: q4 {} >= i8 {}",
            cost.total(),
            i8_cost.total()
        );
        let p40 = 24 * GB;
        let one = model::shard::plan_by_capacity(&cost, &[(0, p40)]);
        assert!(
            one.is_some(),
            "policy C ({} bytes) must fit one {p40}-byte card - it is the whole point of narrowing the tier; \
             uniform I8 needs {} bytes and correctly does not fit",
            cost.total(),
            i8_cost.total()
        );
        let one = one.unwrap();
        assert_eq!(one.len(), 1, "a single-card plan must be exactly one stage");
        assert!(one[0].shard.embed && one[0].shard.head, "the lone stage must own both endpoints");
        assert_eq!(one[0].shard.start, 0);
        assert_eq!(one[0].shard.end, cfg.n_layers as usize);
        assert!(one[0].bytes <= p40);
    }

    /// A [`TierPolicy`] that is uniform Q4 with NO exception also fits one
    /// card, with more headroom than policy C - the reference point that
    /// isolates what holding the GDN gates at F32 costs (a few MB, per
    /// [`Qwen35Config::layer_weight_bytes`]'s own ground-truth test) against
    /// what it buys (see M24's roadmap entry for the position-sweep quality
    /// comparison this arithmetic alone cannot make).
    #[test]
    fn uniform_q4_fits_one_24gb_card_with_more_headroom_than_policy_c() {
        let cfg = Qwen35Config::qwen38_27b();
        let uniform_q4 = TierPolicy::uniform(Dtype::Q4);
        let policy_c = TierPolicy::uniform(Dtype::Q4).with(&["in_proj_a.weight", "in_proj_b.weight"], Dtype::F32);
        let cost_uniform = layer_cost(&cfg, 2048, &uniform_q4);
        let cost_c = layer_cost(&cfg, 2048, &policy_c);
        assert!(
            cost_uniform.total() < cost_c.total(),
            "uniform Q4 must cost fewer bytes than policy C's F32 exception: {} >= {}",
            cost_uniform.total(),
            cost_c.total()
        );
        let p40 = 24 * GB;
        assert!(model::shard::plan_by_capacity(&cost_uniform, &[(0, p40)]).is_some(), "uniform Q4 must fit one card");
    }

    /// `estimate_multi` must never panic and must report ZERO devices for an
    /// unreadable checkpoint - the documented "unavailable" signal. It runs
    /// on the `Executor` dispatcher thread, where a panic takes every other
    /// model on the server down with it.
    #[test]
    fn an_unreadable_gguf_reports_zero_devices_rather_than_panicking() {
        let r = Qwen35GgufResident::new("/nonexistent/qwen35.gguf".to_string(), vec![(Device::Gpu(0), 24 * GB)], 2048, TierPolicy::uniform(Dtype::I8));
        let cost = r.estimate_multi(&InstanceKey::new(MODEL, "default"));
        assert_eq!(cost.devices().count(), 0);
        assert!(r.placement().is_empty());
        assert!(r.activate_multi(&InstanceKey::new(MODEL, "default"), &[]).is_err());
        // The plan is memoized, so a second call must agree with the first
        // rather than re-deciding (and must still not panic).
        assert_eq!(r.estimate_multi(&InstanceKey::new(MODEL, "default")).devices().count(), 0);
    }

    /// No budgeted GPU is also "unavailable", not a panic and not a CPU
    /// fallback (there is no CPU path - the weights are int8 device buffers).
    #[test]
    fn no_gpu_reports_zero_devices() {
        let r = Qwen35GgufResident::new("whatever.gguf".to_string(), vec![], 2048, TierPolicy::uniform(Dtype::I8));
        assert_eq!(r.estimate_multi(&InstanceKey::new(MODEL, "default")).devices().count(), 0);
    }

    /// `reasoning_effort` is declared in `caps.rs`'s manifest but was missing
    /// from this resident's own - `qwen3::chat::parse_request` still defaulted
    /// it to `xhigh` either way, but a caller passing it EXPLICITLY through
    /// this path was rejected by param validation before it ever reached the
    /// parser. `Qwen35GgufResident::new` needs no real file (`manifest()`
    /// touches only `self.cap`, never the checkpoint), so this needs no GPU.
    #[test]
    fn manifest_declares_reasoning_effort() {
        let r = Qwen35GgufResident::new("whatever.gguf".to_string(), vec![], 2048, TierPolicy::uniform(Dtype::I8));
        let m = r.manifest();
        assert_eq!(m.actions.len(), 1);
        let g = &m.actions[0];
        assert!(
            g.params.iter().any(|p| p.name == "reasoning_effort" && !p.required),
            "reasoning_effort must be declared (and optional, defaulting to xhigh) so a caller passing it is not rejected"
        );
    }

    /// Single-device activation is refused, loudly - registering this model
    /// with the plain `Executor::register` would budget only one of the cards
    /// it actually occupies.
    #[test]
    fn single_device_activation_is_refused() {
        let r = Qwen35GgufResident::new("whatever.gguf".to_string(), vec![(Device::Gpu(0), 24 * GB)], 2048, TierPolicy::uniform(Dtype::I8));
        let key = InstanceKey::new(MODEL, "default");
        assert_eq!(r.estimate(&key).vram, 0);
        let err = match r.activate(&key, Device::Gpu(0)) {
            Ok(_) => panic!("single-device activate must be refused"),
            Err(e) => e,
        };
        assert!(err.contains("multi-device only"), "{err}");
    }
}
