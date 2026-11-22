// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapters putting Qwen3.5/3.6-35B-A3B's paged serving engine
//! (`qwen35moe::serve::Engine` under the generic `model::serve::Scheduler`, driven by `llmserve`) behind the
//! residency [`Executor`](residency::Executor). The engine is single-GPU: every resident sequence's decode row goes through one batched
//! forward per scheduler iteration, and the sequences share one paged KV pool, each taking the blocks its tokens cover
//! (see `qwen35moe::serve`'s module doc for its scope).
//!
//! Three ways in, all found by the model-dir scan:
//!
//! - [`gguf_resident`]: a released GGUF (`general.architecture = "qwen35moe"`) served from the file, int8 weights
//!   read straight from its Q8_0 blocks. Its KV tier, batch, context and KV pool are planned from the card it is placed
//!   on ([`plan`]: the shared [`KV_DEFAULTS`] ranking by fit, the widest batch of [`MAX_BATCH_DEFAULTS`] that leaves the
//!   shared [`DEFAULT_CONTEXT_FLOOR`], the longest context one sequence can take, then the largest pool the card holds
//!   beside the rest), configured by
//!   [`Qwen35MoeGgufServeConfig`] and reported line by line ([`Qwen35MoeGgufResident::chosen_defaults`]).
//!   `brain perf run --target qwen35moe:<gguf>` builds exactly this resident.
//! - [`Qwen35Resident::from_card`]: a brain checkpoint (family `"qwen35moe"`, what `brain import-gguf`'s conversion
//!   stamps) under its own card id with its sibling `tokenizer.json`.
//! - [`Qwen35Resident::from_hf_dir`]: a published HuggingFace directory, read in place.
//!
//! The last two are sized by the serving policy ([`Qwen35Sizing`]: `brain serve --ctx`, `--max-batch`, defaults
//! 4096 and 4) with f32 weights or the packed int8 the checkpoint stores, and an fp32 KV cache.
//!
//! Not present: LoRA adapter folding, multi-GPU sharding, a prefix cache (the engine keeps no recurrent-state snapshot
//! to resume a shared prefix from), speculative decoding, and parking a GGUF instance off the card.

use capability::{Invocation, Manifest};
use std::sync::Arc;
use data::qwen_tokenizer::QwenBpe;
use qwen3::chat::parse_request;
use qwen35moe::config::Qwen35Config;
use qwen35moe::serve::{Engine, EngineOptions, Scheduler};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel, Tier};
use gpu_core::defaults::Hardware;
use model::kv_tier::KvTier;
use model::ops::TierPolicy;

use crate::serve_defaults::{given_or_chosen, DEFAULT_CONTEXT_FLOOR, KV_DEFAULTS, MAX_BATCH_DEFAULTS};

use crate::parkable::{Build, HostWeights, ParkableInstance};
use crate::resident_qwen35::Qwen35Sizing;
use crate::resident_llm::{est_vram, open_stored_checkpoint};
use llmserve::generate_spec;
use residency::on_device;
use llmserve::{ScheduledInstance, TextFrontend};

/// The Qwen3.5-35B-A3B hybrid Gated-DeltaNet/GQA sparse-MoE decoder of a brain checkpoint or a HuggingFace
/// directory behind the scheduler. See this module's own doc for its scope.
pub struct Qwen35Resident {
    id: String,
    path: String,
    tokenizer: String,
    /// `path` is a HuggingFace checkpoint directory read in place (ModelOpt FP8/NVFP4, compressed-tensors or bf16
    /// safetensors, as published), not a brain-native file.
    hf_dir: bool,
    sizing: Qwen35Sizing,
}

impl Qwen35Resident {
    /// A checkpoint discovered by `serving::model_dir` (family `"qwen35moe"` -
    /// what `qwen35moe::import`'s conversion stamps on its `ModelCard`), served
    /// under its own card id.
    ///
    /// `tokenizer` is the sibling `tokenizer.json` the scan found. It is
    /// required: `Engine` never opens a `.gguf`, so unlike `QwenResident` there
    /// is no embedded-tokenizer fallback (see this module's own doc) - a
    /// checkpoint without one would activate straight into an error, so it is
    /// declined at discovery instead, exactly like the `lfm` family.
    pub fn from_card(path: &str, card: &checkpoint::st::ModelCard, tokenizer: Option<&str>, sizing: Qwen35Sizing) -> Result<Qwen35Resident, String> {
        let tokenizer = tokenizer.filter(|t| !t.is_empty()).ok_or("qwen35moe: no sibling tokenizer.json")?;
        Ok(Qwen35Resident { id: card.id.clone(), path: path.to_string(), tokenizer: tokenizer.to_string(), hf_dir: false, sizing: sizing.sanitized() })
    }

    /// A HuggingFace checkpoint directory (`config.json`, `tokenizer.json` and safetensors shards, as pulled), served
    /// where it lies: every tensor reaches the card in the format it is stored in.
    pub fn from_hf_dir(dir: &str, card: &checkpoint::st::ModelCard, sizing: Qwen35Sizing) -> Result<Qwen35Resident, String> {
        let tokenizer = std::path::Path::new(dir).join("tokenizer.json");
        if !tokenizer.is_file() {
            return Err(format!("qwen35moe: {dir} has no tokenizer.json"));
        }
        Ok(Qwen35Resident { id: card.id.clone(), path: dir.to_string(), tokenizer: tokenizer.to_string_lossy().into_owned(), hf_dir: true, sizing: sizing.sanitized() })
    }

    /// The hard per-sequence `prompt + max_new` cap, also the engine's
    /// physical block size.
    fn ctx(&self) -> u32 {
        self.sizing.ctx
    }

    /// How many sequences may be resident at once, all decoded in one batched
    /// step.
    fn max_concurrent(&self) -> u32 {
        self.sizing.max_batch
    }
}

impl Qwen35Resident {
    /// Device bytes of a directory-served checkpoint: the stored bytes of its shards (FP8 and NVFP4 stay as stored), the
    /// embedding table widened to f32, plus the KV pool and recurrent state the engine allocates.
    fn estimate_hf_dir(&self) -> MemCost {
        let stored: u64 = std::fs::read_dir(&self.path)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();
        let Ok(cfg) = qwen35moe::hf_source::hf_config(std::path::Path::new(&self.path)) else {
            return MemCost::new(stored, 0);
        };
        let weights = stored + cfg.vocab as u64 * cfg.d_model as u64 * 2;
        let options = qwen35moe::serve::EngineOptions::new(self.ctx(), self.max_concurrent());
        let kv = qwen35moe::serve::kv_pool_bytes(&cfg, options.kv_tier, options.pool_rows(), self.max_concurrent());
        MemCost::new(weights + kv, 0).with_activation(self.activation_bytes(&cfg))
    }

    /// The peak activation of a decode step over every concurrent sequence or one prefill round.
    fn activation_bytes(&self, cfg: &Qwen35Config) -> u64 {
        qwen35moe::serve::activation_bytes(cfg, &qwen35moe::serve::EngineOptions::new(self.ctx(), self.max_concurrent()))
    }

    /// Build the scheduler from the directory, in place: the config and the tensors are read from the published files.
    fn activate_hf_dir(&self, device: Device) -> Result<Box<dyn Instance>, String> {
        let dir = std::path::Path::new(&self.path);
        let cfg = qwen35moe::hf_source::hf_config(dir)?;
        let tok = QwenBpe::from_file(&self.tokenizer)?;
        let stop = data::generation::stop_ids(Some(dir), &tok, None, Some("<|im_end|>")).map_err(|e| format!("qwen35moe: {e}"))?;
        let (ctx, max_concurrent) = (self.ctx(), self.max_concurrent());
        let front = Arc::new(TextFrontend::new(tok, stop, "qwen35moe", Arc::new(parse_request)));
        let src = qwen35moe::hf_source::open_dir(dir, &cfg)?;
        let names = src.names().into_iter().map(str::to_string).collect();
        let host = HostWeights::from_source(Box::new(src), names);
        let config = cfg.to_json();
        let build: Build = Box::new(move |host, device| {
            let sched = on_device(device, || engine_from(host.source(), &config, ctx, max_concurrent).map(|engine| Scheduler::new(engine, max_concurrent as usize)))??;
            Ok(Box::new(ScheduledInstance::new(front.clone(), sched, max_concurrent as usize)))
        });
        Ok(Box::new(ParkableInstance::activate(Arc::new(host), device, build)?))
    }
}

impl ResidentModel for Qwen35Resident {
    fn manifest(&self) -> Manifest {
        // `self.ctx()` is exactly the `max_seq_len` `activate()` below builds
        // the engine with, so advertising it is a safe, never-overstated
        // floor on real serving capacity (`Manifest::max_context_tokens`'s own
        // doc on why this must be real engine capacity, not an architectural
        // maximum) — same reasoning as `QwenResident::manifest`.
        Manifest::new(
            &self.id,
            "text generation (Qwen3.5-35B-A3B hybrid Gated-DeltaNet/GQA sparse-MoE decoder; single-GPU, f32 or packed-int8 weights as the checkpoint stores them + fp32 KV, up to --max-batch sequences per batched decode step -- see crate::resident_qwen35moe's module doc)",
            vec![generate_spec("generate text (Qwen3.5-35B-A3B; chat template optional)", true).params(qwen3::chat::template_params(data::qwen_chat::TemplateFlavor::Qwen3))],
        )
        .with_max_context_tokens(self.ctx() as u64)
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(&self.id, "default")
    }
    fn batch_limit(&self) -> Option<usize> {
        Some(self.max_concurrent() as usize)
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        if self.hf_dir {
            return self.estimate_hf_dir();
        }
        let cost = est_vram(&self.path);
        let Ok(reader) = checkpoint::weightio::WeightReader::open(&self.path) else {
            return cost;
        };
        let cfg = Qwen35Config::from_json(&reader.config());
        // What `engine_from` builds: a pool of a whole context per sequence at the f32 KV tier, and a recurrent-state
        // row per sequence.
        let options = EngineOptions::new(self.ctx(), self.max_concurrent());
        let kv = qwen35moe::serve::kv_pool_bytes(&cfg, options.kv_tier, options.pool_rows(), self.max_concurrent());
        MemCost::new(cost.vram + kv, cost.ram).with_activation(self.activation_bytes(&cfg))
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if self.hf_dir {
            return self.activate_hf_dir(device);
        }
        // Open first so a GGUF can supply its own embedded tokenizer, exactly
        // like crate::resident_llm::QwenResident::activate. The engine is built
        // from the stored checkpoint view ([`engine_from`]); this reader
        // exists for the tokenizer fallback.
        let reader = checkpoint::weightio::WeightReader::open(&self.path).map_err(|e| format!("qwen35moe: {e}"))?;
        // Tokenizer precedence: the sibling tokenizer.json wins; else a .gguf builds from its embedded
        // tokenizer.ggml.* KV; else there is nothing to tokenize with.
        let (tok, tok_dir, gguf_eos) = if !self.tokenizer.is_empty() {
            (QwenBpe::from_file(&self.tokenizer)?, std::path::Path::new(&self.tokenizer).parent(), None)
        } else if let Some(gt) = reader.tokenizer() {
            (QwenBpe::from_gguf(&gt).map_err(|e| format!("qwen35moe: {e}"))?, None, gt.eos)
        } else {
            return Err("qwen35moe: no tokenizer (a sibling tokenizer.json, or a GGUF with an embedded tokenizer)".to_string());
        };
        // The checkpoint's own eos ids, plus ChatML's end-of-turn marker: this
        // resident renders every prompt as ChatML.
        let stop = data::generation::stop_ids(tok_dir, &tok, gguf_eos, Some("<|im_end|>")).map_err(|e| format!("qwen35moe: {e}"))?;
        let ctx = self.ctx();
        let max_concurrent = self.max_concurrent();
        let front = Arc::new(TextFrontend::new(tok, stop, "qwen35moe", Arc::new(parse_request)));
        let (config, host) = HostWeights::open(&self.path, |config| Qwen35Config::from_json(config).param_list().into_iter().map(|(name, _)| name).collect())?;
        let build: Build = Box::new(move |host, device| {
            let sched = on_device(device, || engine_from(host.source(), &config, ctx, max_concurrent).map(|engine| Scheduler::new(engine, max_concurrent as usize)))??;
            Ok(Box::new(ScheduledInstance::new(front.clone(), sched, max_concurrent as usize)))
        });
        Ok(Box::new(ParkableInstance::activate(Arc::new(host), device, build)?))
    }
    /// A demoted instance keeps the checkpoint's packed bytes as a mapping of the file (evictable page cache), no
    /// f32 copy and no device memory.
    fn estimate_at(&self, key: &InstanceKey, tier: Tier) -> MemCost {
        match tier {
            Tier::Hot => self.estimate(key),
            Tier::Warm | Tier::Cold if self.hf_dir => MemCost::new(0, 0).with_mapped(self.estimate_hf_dir().vram),
            Tier::Warm | Tier::Cold => MemCost::new(0, 0).with_mapped(std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0)),
        }
    }
}

/// The serving engine over the brain checkpoint at `path` (see `engine_from`).
pub fn serving_engine(path: &str, ctx: u32, max_concurrent: u32) -> Result<Engine, String> {
    let (config, src) = open_stored_checkpoint(path)?;
    engine_from(&*src, &config, ctx, max_concurrent)
}

/// The serving engine over `src`, a brain checkpoint of config `config`, read one tensor at a time as the file
/// stores it: the routed experts stay packed int8 when the checkpoint holds them so (the engine implements the f32
/// and int8 tiers), and every other tensor is decoded as it is uploaded, never as a whole-model fp32 host copy.
fn engine_from(src: &dyn checkpoint::TensorSource, config: &serde_json::Value, ctx: u32, max_concurrent: u32) -> Result<Engine, String> {
    let cfg = Qwen35Config::from_json(config);
    let experts_stored = model::ops::stored_tier(src, "blocks.0.mlp.experts.0.gate.weight");
    let tier = model::ops::TierPolicy::uniform(if experts_stored == Some(gpu_core::select::Dtype::I8) { gpu_core::select::Dtype::I8 } else { gpu_core::select::Dtype::F32 });
    Ok(Engine::from_source(cfg, src, EngineOptions::new(ctx, max_concurrent).with_tier(tier)))
}

// ------------------------------------------------------------ released GGUF

/// Server-wide settings of every released Qwen3.5/3.6 MoE GGUF the model scan serves, as parsed data (`brain serve`'s
/// serving-policy flags, applied by [`crate::resident_qwen35::Qwen35ServeConfig::apply_policy`]). Every knob left
/// `None` is chosen from the card the resident is planned for ([`plan`]).
#[derive(Clone, Debug, Default)]
pub struct Qwen35MoeGgufServeConfig {
    /// Per-sequence `prompt + max_new` ceiling. `None` takes the longest the card holds for the chosen batch.
    pub ctx: Option<u32>,
    /// Per-leaf weight-tier policy. `None` serves int8, the tier a Q8_0 release stores.
    pub tier: Option<TierPolicy>,
    /// GQA KV-cache tier. `None` takes the best of [`KV_DEFAULTS`] that fits.
    pub kv: Option<KvTier>,
    /// Sequences decoded together in one batched step. `None` takes the most of [`MAX_BATCH_DEFAULTS`] that still
    /// leaves [`DEFAULT_CONTEXT_FLOOR`] of context.
    pub max_batch: Option<u32>,
    /// Why the serving policy cannot be honoured by this resident (set by [`Self::apply_policy`]); building refuses
    /// with it.
    pub refused: Option<String>,
}

/// What the released-GGUF MoE resident honours of a [`model::policy::ServePolicy`].
pub const QWEN35MOE_GGUF_POLICY: model::policy::PolicySupport = model::policy::PolicySupport {
    model: "qwen35moe",
    weights: &[gpu_core::select::Dtype::F32, gpu_core::select::Dtype::I8],
    weight_rules: true,
    kv_cache: &["f32", "bf16", "int8", "fp8"],
    gemm: false,
    fuse_norm: false,
    kv_calibration: false,
    exact_decode: false,
    act_precision: false,
    gdn_state: false,
    linear_activations: false,
    sparse_attention: false,
    draft_model: false,
};

impl Qwen35MoeGgufServeConfig {
    /// Apply the model-agnostic policy. A choice this engine cannot honour records why and refuses to build; a limit
    /// it has no knob for is named in the returned notes, never dropped silently.
    pub fn apply_policy(&mut self, p: &model::policy::ServePolicy) -> Result<Vec<String>, String> {
        if let Err(e) = QWEN35MOE_GGUF_POLICY.check(p) {
            self.refused = Some(e.clone());
            return Err(e);
        }
        if p.weights.is_some() {
            self.tier = p.weights.clone();
        }
        self.kv = p.kv_cache.or(self.kv);
        let l = &p.limits;
        self.ctx = l.ctx.or(self.ctx);
        self.max_batch = l.max_batch.or(self.max_batch);
        let ignored = [
            (l.kv_offload_gb.is_some(), "--kv-offload-gb"),
            (l.kv_tokens.is_some(), "kv-tokens"),
            (l.max_prefill.is_some(), "max-prefill"),
            (l.prefix_slots.is_some(), "prefix slots"),
            (l.prefill_slice.is_some(), "the prefill-slice setting"),
            (l.prefill_batch.is_some(), "the prefill-batch setting"),
            (l.itl_target_ms.is_some(), "--itl-target-ms"),
        ];
        Ok(ignored.iter().filter(|(set, _)| *set).map(|(_, name)| format!("qwen35moe: {name} has no meaning for this engine and is ignored")).collect())
    }
}

/// What the planner settled on for one card.
#[derive(Clone, Debug, PartialEq)]
pub struct MoePlan {
    /// Per-sequence `prompt + max_new` ceiling.
    pub ctx: u32,
    /// Sequences resident at once (the engine's recurrent-state rows).
    pub max_batch: u32,
    pub kv: KvTier,
    /// Rows of the KV pool every sequence draws its blocks from.
    pub pool_rows: u32,
    /// Each default taken, as `knob value: why`.
    pub chosen: Vec<String>,
}

impl MoePlan {
    /// The engine this plan builds, at weight tier `tier`.
    pub fn options(&self, tier: &TierPolicy) -> EngineOptions {
        options(tier, self.kv, self.max_batch, self.ctx, self.pool_rows)
    }
}

fn options(tier: &TierPolicy, kv: KvTier, max_batch: u32, ctx: u32, pool_rows: u32) -> EngineOptions {
    EngineOptions::new(ctx, max_batch).with_tier(tier.clone()).with_kv_tier(kv).with_kv_pool_rows(pool_rows)
}

/// Rows of the pool one sequence of `ctx` tokens takes: its whole blocks.
fn sequence_rows(ctx: u32) -> u32 {
    EngineOptions::new(ctx, 1).pool_rows() as u32
}

/// Device bytes an engine of `max_batch` sequences of at most `ctx` tokens sharing a pool of `pool_rows` KV rows holds
/// (weights, KV pool, recurrent state) and the activation peak beside them - the arithmetic the engine allocates from.
pub fn footprint(cfg: &Qwen35Config, tier: &TierPolicy, kv: KvTier, max_batch: u32, ctx: u32, pool_rows: u32) -> MemCost {
    let opts = options(tier, kv, max_batch, ctx, pool_rows);
    let held = qwen35moe::serve::weight_bytes(cfg, tier) + qwen35moe::serve::kv_pool_bytes(cfg, kv, opts.pool_rows(), max_batch);
    MemCost::new(held, 0).with_activation(qwen35moe::serve::activation_bytes(cfg, &opts))
}

const GIB: f64 = (1u64 << 30) as f64;

/// Whether `max_batch` sequences of at most `ctx` tokens sharing `pool_rows` KV rows fit `usable` device bytes, or why
/// not. The pool must hold one whole sequence.
fn fits(cfg: &Qwen35Config, tier: &TierPolicy, usable: u64, kv: KvTier, max_batch: u32, ctx: u32, pool_rows: u32) -> Result<(), String> {
    if pool_rows < sequence_rows(ctx) {
        return Err(format!("a KV pool of {pool_rows} rows cannot hold one sequence of {ctx} tokens"));
    }
    if !KvTier::fits_addressing(pool_rows as u64, cfg.kv_dim() as u64) {
        return Err(format!("a KV pool of {pool_rows} rows is more than the attention kernels address"));
    }
    let cost = footprint(cfg, tier, kv, max_batch, ctx, pool_rows);
    let need = cost.vram + cost.activation.unwrap_or(0);
    if need > usable {
        return Err(format!(
            "{max_batch} sequences of up to {ctx} tokens sharing a {pool_rows}-row pool at {kv} KV need {:.1} GiB of the card's {:.1} GiB",
            need as f64 / GIB,
            usable as f64 / GIB
        ));
    }
    Ok(())
}

/// Contexts and pools are planned in whole multiples of this many tokens.
const CONTEXT_GRAIN: u32 = 256;

/// The largest `n` in `[lo, hi]` for which `fits(n)`, given `fits(lo)` and that the cost grows with `n`, rounded down to
/// [`CONTEXT_GRAIN`] when it is at least one grain.
fn largest_fitting(lo: u32, hi: u32, fits: impl Fn(u32) -> bool) -> u32 {
    let (mut lo, mut hi) = (lo, hi.max(lo));
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if lo >= CONTEXT_GRAIN {
        lo / CONTEXT_GRAIN * CONTEXT_GRAIN
    } else {
        lo
    }
}

/// The longest context, at most the checkpoint's trained window, that one sequence can take while `max_batch` sequences
/// fit `usable` bytes: the pool is that one sequence's blocks.
pub fn longest_context(cfg: &Qwen35Config, tier: &TierPolicy, usable: u64, kv: KvTier, max_batch: u32) -> Result<u32, String> {
    let alone = |ctx: u32| fits(cfg, tier, usable, kv, max_batch, ctx, sequence_rows(ctx));
    alone(1)?;
    Ok(largest_fitting(1, cfg.max_position_embeddings.max(1), |ctx| alone(ctx).is_ok()))
}

/// The largest KV pool, in rows, that `usable` bytes hold for `max_batch` sequences of at most `ctx` tokens beside the
/// weights, the recurrent state and the activations; at least one sequence's.
pub fn longest_pool(cfg: &Qwen35Config, tier: &TierPolicy, usable: u64, kv: KvTier, max_batch: u32, ctx: u32) -> Result<u32, String> {
    let one = sequence_rows(ctx);
    fits(cfg, tier, usable, kv, max_batch, ctx, one)?;
    let row_bytes = qwen35moe::serve::gqa_pool_bytes(cfg, kv, 1).max(1);
    // No pool is larger than the whole card in rows, or than the kernels address.
    let top = (usable / row_bytes).min(u32::MAX as u64) as u32;
    Ok(largest_fitting(one, top, |rows| fits(cfg, tier, usable, kv, max_batch, ctx, rows).is_ok()).max(one))
}

/// Plan the KV tier, the batch, the context and the KV pool of one engine for a card with `usable` device bytes, in the
/// order they matter to throughput, each with the ones not yet chosen at their leanest: the KV tier (one sequence), then
/// the batch, then the context of one sequence, then the pool every sequence shares - the rest of the card. An
/// operator's value is kept as given; a given context that does not fit is refused with the arithmetic.
pub fn plan(cfg: &Qwen35Config, tier: &TierPolicy, usable: u64, serve: &Qwen35MoeGgufServeConfig, hw: &Hardware) -> Result<MoePlan, String> {
    let mut chosen = Vec::new();
    let floor = DEFAULT_CONTEXT_FLOOR.min(cfg.max_position_embeddings.max(1));
    let holds = |kv: KvTier, max_batch: u32| -> Result<(), String> {
        match serve.ctx {
            Some(ctx) => fits(cfg, tier, usable, kv, max_batch, ctx, sequence_rows(ctx)),
            None => match longest_context(cfg, tier, usable, kv, max_batch)? {
                c if c >= floor => Ok(()),
                c => Err(format!("it leaves a {c}-token context, under {floor}")),
            },
        }
    };
    let kv = given_or_chosen(serve.kv, "--kv-cache", KV_DEFAULTS, hw, &mut chosen, |kv| holds(kv, 1));
    let max_batch = given_or_chosen(serve.max_batch.map(|b| b.max(1)), "--max-batch", MAX_BATCH_DEFAULTS, hw, &mut chosen, |b| holds(kv, b));
    let ctx = match serve.ctx {
        Some(ctx) => ctx.max(1),
        None => {
            let ctx = longest_context(cfg, tier, usable, kv, max_batch)?;
            chosen.push(format!("--ctx {ctx}: the longest context one sequence can take beside {max_batch} sequences' recurrent state and activations"));
            ctx
        }
    };
    let pool_rows = longest_pool(cfg, tier, usable, kv, max_batch, ctx)?;
    chosen.push(format!("KV pool {pool_rows} tokens shared by all {max_batch} sequences: what the card holds beside the weights, the recurrent state and the activations"));
    Ok(MoePlan { ctx, max_batch, kv, pool_rows, chosen })
}

/// The weight tier a released GGUF is served at when `--weight-format` is not given: uniform int8, what a Q8_0
/// release stores (a byte repack of its blocks, no fp32 copy).
pub fn default_tier() -> TierPolicy {
    TierPolicy::uniform(gpu_core::select::Dtype::I8)
}

/// A released Qwen3.5/3.6 MoE GGUF served from the file on one card: int8 weights read straight from its Q8_0
/// blocks, a KV tier, batch and context planned for the card ([`plan`]), and the generic continuous-batching
/// scheduler over `qwen35moe::serve::Engine`. One instance per process: the engine is single-GPU.
pub struct Qwen35MoeGgufResident {
    id: String,
    path: String,
    tier: TierPolicy,
    plan: MoePlan,
    /// The card the plan was made for (the roomiest offered).
    card: u32,
}

/// Build the resident for the released GGUF at `gguf_path`, to be registered under `id`.
///
/// `gpus` is the server's budgeted cards as `(index, total bytes)`, `reserved` the per-card headroom it keeps free.
/// `Err` names why the model could never load (no card, not a qwen35moe GGUF, no embedded tokenizer, weights that do
/// not fit), so the caller leaves it out and says so.
pub fn gguf_resident(gguf_path: &str, id: &str, gpus: &[(u32, u64)], reserved: u64, cfg: &Qwen35MoeGgufServeConfig) -> Result<Qwen35MoeGgufResident, String> {
    if let Some(why) = &cfg.refused {
        return Err(why.clone());
    }
    let mg = checkpoint::gguf::MmapGguf::open(gguf_path).map_err(|e| format!("cannot open '{gguf_path}': {e}"))?;
    let model_cfg = qwen35moe::import::config_from_gguf(&mg).map_err(|e| format!("'{gguf_path}' is not a servable Qwen3.5/3.6 MoE GGUF: {e}"))?;
    if mg.tokenizer().is_none() {
        return Err(format!("'{gguf_path}' carries no embedded tokenizer (tokenizer.ggml.*)"));
    }
    let (card, usable) = gpus.iter().map(|&(i, total)| (i, total.saturating_sub(reserved))).max_by_key(|&(_, usable)| usable).ok_or("no GPU is budgeted for it (its weights are device buffers; there is no CPU path)")?;
    let mut chosen = Vec::new();
    let tier = match &cfg.tier {
        Some(tier) => tier.clone(),
        None => {
            chosen.push("--weight-format int8: the routed experts and linears as the Q8_0 release stores them".to_string());
            default_tier()
        }
    };
    let mut plan = plan(&model_cfg, &tier, usable, cfg, Hardware::detected()).map_err(|e| format!("does not fit gpu{card}: {e}"))?;
    chosen.append(&mut plan.chosen);
    plan.chosen = chosen;
    Ok(Qwen35MoeGgufResident { id: id.to_string(), path: gguf_path.to_string(), tier, plan, card })
}

impl Qwen35MoeGgufResident {
    /// Sequences decoded together in one batched step.
    pub fn max_batch(&self) -> u32 {
        self.plan.max_batch
    }
    /// The per-sequence `prompt + max_new` ceiling.
    pub fn cap(&self) -> u32 {
        self.plan.ctx
    }
    pub fn kv_tier(&self) -> KvTier {
        self.plan.kv
    }
    /// Rows of the KV pool the sequences share.
    pub fn kv_pool_rows(&self) -> u32 {
        self.plan.pool_rows
    }
    /// Each setting the operator left unset, as `knob value: why`.
    pub fn chosen_defaults(&self) -> &[String] {
        &self.plan.chosen
    }
    /// The card the plan was made for.
    pub fn planned_device(&self) -> Device {
        Device::Gpu(self.card)
    }

    /// The scheduler `brain serve` runs for this checkpoint on `device`, with the embedded tokenizer and the stop ids
    /// (the GGUF's eos and ChatML's end of turn).
    pub fn activate_scheduler(&self, device: Device) -> Result<(Scheduler, QwenBpe, Vec<u32>), String> {
        let mg = checkpoint::gguf::MmapGguf::open(&self.path).map_err(|e| format!("cannot open '{}': {e}", self.path))?;
        let tok = qwen35moe::gguf_load::tokenizer(&mg)?;
        let eos = mg.tokenizer().and_then(|t| t.eos);
        let stop = data::generation::stop_ids(None, &tok, eos, Some("<|im_end|>")).map_err(|e| format!("qwen35moe: {e}"))?;
        let cfg = qwen35moe::gguf_load::resident_config(&mg, self.plan.ctx)?;
        let src = qwen35moe::gguf_load::source(&mg, &cfg)?;
        let opts = self.plan.options(&self.tier);
        let engine = on_device(device, || Engine::from_source(cfg, &src, opts))?;
        Ok((Scheduler::new(engine, self.plan.max_batch as usize), tok, stop))
    }
}

impl ResidentModel for Qwen35MoeGgufResident {
    fn manifest(&self) -> Manifest {
        let summary = format!(
            "generate text (Qwen3.5/3.6-35B-A3B hybrid Gated-DeltaNet/GQA sparse-MoE decoder from its released GGUF on one GPU: {} weights, {} KV, up to {} sequences per batched decode step sharing a {}-token KV pool, {} tokens per sequence)",
            self.tier.describe(),
            self.plan.kv,
            self.plan.max_batch,
            self.plan.pool_rows,
            self.plan.ctx
        );
        Manifest::new(&self.id, "text generation (Qwen3.5/3.6-35B-A3B sparse-MoE decoder, released GGUF)", vec![generate_spec(&summary, true).params(qwen3::chat::template_params(data::qwen_chat::TemplateFlavor::Qwen3))])
            .with_max_context_tokens(self.plan.ctx as u64)
            .with_architecture(qwen35moe::spec::CARD_FAMILY)
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(&self.id, "default")
    }
    fn batch_limit(&self) -> Option<usize> {
        Some(self.plan.max_batch as usize)
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        let Ok(mg) = checkpoint::gguf::MmapGguf::open(&self.path) else {
            return MemCost::new(std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0), 0);
        };
        match qwen35moe::import::config_from_gguf(&mg) {
            Ok(cfg) => footprint(&cfg, &self.tier, self.plan.kv, self.plan.max_batch, self.plan.ctx, self.plan.pool_rows),
            Err(_) => MemCost::new(std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0), 0),
        }
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let (sched, tok, stop) = self.activate_scheduler(device)?;
        let front = TextFrontend::new(tok, stop, "qwen35moe", Arc::new(parse_request));
        Ok(Box::new(ScheduledInstance::new(front, sched, self.plan.max_batch as usize)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn int8() -> TierPolicy {
        TierPolicy::uniform(gpu_core::select::Dtype::I8)
    }

    fn need(cfg: &Qwen35Config, p: &MoePlan) -> u64 {
        let cost = footprint(cfg, &int8(), p.kv, p.max_batch, p.ctx, p.pool_rows);
        cost.vram + cost.activation.unwrap_or(0)
    }

    /// A 96 GB card with every knob left to the plan serves the widest batch, one sequence may take the checkpoint's
    /// whole window, and the rest of the card is one KV pool the sequences share: it holds every one of the batch at
    /// half the context floor, and no grain more fits.
    #[test]
    fn a_roomy_card_plans_a_wide_batch_sharing_the_rest_of_the_card_as_one_pool() {
        let cfg = Qwen35Config::qwen35_35b_a3b();
        let usable = 92 * GIB;
        let p = plan(&cfg, &int8(), usable, &Qwen35MoeGgufServeConfig::default(), &Hardware::default()).expect("the 35B fits a 96 GB card");
        assert_eq!((p.kv, p.max_batch), (KvTier::INT8, 64), "{:?}", p.chosen);
        assert_eq!(p.ctx, cfg.max_position_embeddings, "the batch is not bought with context: {:?}", p.chosen);
        assert!(p.pool_rows >= p.max_batch * DEFAULT_CONTEXT_FLOOR / 2, "{} rows", p.pool_rows);
        assert_eq!(p.pool_rows % CONTEXT_GRAIN, 0);
        assert!(need(&cfg, &p) <= usable);
        assert!(fits(&cfg, &int8(), usable, p.kv, p.max_batch, p.ctx, p.pool_rows + CONTEXT_GRAIN).is_err(), "the pool is the largest the card holds");
        for knob in ["--kv-cache int8", "--max-batch 64", "--ctx", "KV pool"] {
            assert!(p.chosen.iter().any(|l| l.starts_with(knob)), "{knob} is reported: {:?}", p.chosen);
        }
    }

    /// What the plan charges is what the engine it builds allocates: the pool rows, the state rows and the activations
    /// of the options it hands the engine.
    #[test]
    fn the_plan_charges_the_pool_the_engine_is_built_with() {
        let cfg = Qwen35Config::qwen35_35b_a3b();
        let p = plan(&cfg, &int8(), 92 * GIB, &Qwen35MoeGgufServeConfig::default(), &Hardware::default()).unwrap();
        let opts = p.options(&int8());
        assert_eq!(opts.pool_rows(), p.pool_rows as u64, "the planned pool is whole blocks");
        let held = qwen35moe::serve::weight_bytes(&cfg, &int8()) + qwen35moe::serve::kv_pool_bytes(&cfg, p.kv, opts.pool_rows(), opts.max_concurrent);
        let cost = footprint(&cfg, &int8(), p.kv, p.max_batch, p.ctx, p.pool_rows);
        assert_eq!((cost.vram, cost.activation), (held, Some(qwen35moe::serve::activation_bytes(&cfg, &opts))));
    }

    /// An operator's KV tier, context or batch is kept as given and the rest is planned around it.
    #[test]
    fn an_operators_choice_is_kept_and_the_rest_planned_around_it() {
        let cfg = Qwen35Config::qwen35_35b_a3b();
        let usable = 92 * GIB;
        let given = Qwen35MoeGgufServeConfig { kv: Some(KvTier::BF16), ctx: Some(8320), ..Default::default() };
        let p = plan(&cfg, &int8(), usable, &given, &Hardware::default()).unwrap();
        assert_eq!((p.kv, p.ctx, p.max_batch), (KvTier::BF16, 8320, 64));
        assert!(!p.chosen.iter().any(|l| l.starts_with("--kv-cache") || l.starts_with("--ctx")), "{:?}", p.chosen);

        let narrow = Qwen35MoeGgufServeConfig { max_batch: Some(4), ..Default::default() };
        let q = plan(&cfg, &int8(), usable, &narrow, &Hardware::default()).unwrap();
        assert_eq!(q.max_batch, 4);
        assert_eq!(q.ctx, cfg.max_position_embeddings, "four sequences take the checkpoint's whole window");
    }

    /// A card that cannot hold the weights, or a given context one sequence of which the card cannot hold beside the
    /// batch, is refused with the arithmetic.
    #[test]
    fn what_the_card_cannot_hold_is_refused_with_the_arithmetic() {
        let cfg = Qwen35Config::qwen35_35b_a3b();
        let e = plan(&cfg, &int8(), 24 * GIB, &Qwen35MoeGgufServeConfig::default(), &Hardware::default()).unwrap_err();
        assert!(e.contains("GiB"), "{e}");
        let too_long = Qwen35MoeGgufServeConfig { ctx: Some(8_000_000), max_batch: Some(64), ..Default::default() };
        let e = plan(&cfg, &int8(), 92 * GIB, &too_long, &Hardware::default()).unwrap_err();
        assert!(e.contains("64 sequences of up to 8000000 tokens"), "{e}");
    }

    /// A choice the engine cannot honour leaves the resident out by name; a limit it has no knob for is a note.
    #[test]
    fn the_serving_policy_reaches_the_moe_resident() {
        let mut p = model::policy::ServePolicy::default();
        p.set("--kv-cache", "bf16").unwrap();
        p.set("--max-batch", "16").unwrap();
        p.set("--itl-target-ms", "40").unwrap();
        let mut cfg = Qwen35MoeGgufServeConfig::default();
        let notes = cfg.apply_policy(&p).unwrap();
        assert_eq!((cfg.kv, cfg.max_batch), (Some(KvTier::BF16), Some(16)));
        assert!(notes.iter().any(|n| n.contains("--itl-target-ms")), "{notes:?}");

        let mut p = model::policy::ServePolicy::default();
        p.set("--gdn-state", "bf16").unwrap();
        let mut cfg = Qwen35MoeGgufServeConfig::default();
        assert!(cfg.apply_policy(&p).is_err());
        assert!(gguf_resident("none.gguf", "t/x", &[(0, 96 * GIB)], 0, &cfg).err().unwrap().contains("--gdn-state"), "a refused resident must not build");
    }

    /// A GGUF the resident could never serve is refused before any device work, naming why.
    #[test]
    fn a_gguf_that_cannot_be_served_is_refused_by_name() {
        let dir = std::env::temp_dir().join(format!("qwen35moe-gguf-resident-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synthetic.gguf").to_string_lossy().into_owned();
        qwen35moe::import::testing::write_synthetic_gguf_without_mtp(&path);
        let cfg = Qwen35MoeGgufServeConfig::default();
        let e = gguf_resident(&path, "t/x", &[(0, 96 * GIB)], 0, &cfg).err().unwrap();
        assert!(e.contains("no embedded tokenizer"), "{e}");
        assert!(gguf_resident("/nonexistent/m.gguf", "t/x", &[(0, 96 * GIB)], 0, &cfg).err().unwrap().contains("cannot open"));
        std::fs::remove_dir_all(&dir).ok();
    }

    const MODEL: &str = qwen35moe::caps::MODEL;

    /// REGRESSION for the whole point of this file: registering
    /// `Qwen35Resident` with the residency `Executor` must be enough, with NO
    /// `crates/apiserve`-side code, for `brain/qwen35moe` to appear on BOTH
    /// the OpenAI and the Anthropic `/v1/models` list with the chat
    /// capability — `apiserve::catalog::api_caps`/`exposed` derive this
    /// purely from `Manifest` shape (`generate_spec("...", true)` declares a
    /// streaming `generate` action with a `messages` param and a `Text`
    /// output — exactly `api_caps`'s chat-capability shape). Drives a REAL
    /// `axum` router over a REAL `Executor`, not a hand-inspected manifest.
    ///
    /// Uses a resident whose weights path does not exist: `GET /v1/models`
    /// never activates a model (`models.rs`'s own doc: "No generation here;
    /// discovery only"), so this proves the auto-exposure wiring without
    /// needing a real checkpoint/tokenizer.
    #[test]
    fn brain_qwen35moe_is_auto_exposed_on_openai_and_anthropic_model_lists() {
        let resident = Qwen35Resident { id: MODEL.to_string(), path: "/nonexistent/qwen35.safetensors".to_string(), tokenizer: String::new(), hf_dir: false, sizing: Qwen35Sizing::default() };
        let models: Vec<std::sync::Arc<dyn ResidentModel>> = vec![std::sync::Arc::new(resident)];
        let mut budgets = residency::budget::Budgets::new();
        budgets.set(Device::Cpu, 8 << 30, 0);
        let exec = residency::Executor::start(models, budgets, residency::scheduler::Policy::default());

        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        for provider in [apiserve::Provider::OpenAI, apiserve::Provider::Anthropic] {
            let key = "sk-brain-test-key".to_string();
            let state = apiserve::AppState::new(exec.clone(), key.clone(), provider);
            let app = apiserve::router(state);
            let mut req = axum::http::Request::builder().method(axum::http::Method::GET).uri("/v1/models");
            req = match provider {
                apiserve::Provider::Anthropic => req.header("x-api-key", &key),
                _ => req.header(axum::http::header::AUTHORIZATION, format!("Bearer {key}")),
            };
            let req = req.body(axum::body::Body::empty()).unwrap();

            let (status, body): (axum::http::StatusCode, serde_json::Value) = rt.block_on(async {
                use tower::ServiceExt;
                let resp = app.oneshot(req).await.unwrap();
                let status = resp.status();
                let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
                (status, serde_json::from_slice(&bytes).unwrap())
            });
            assert_eq!(status, axum::http::StatusCode::OK, "{provider:?} GET /v1/models failed: {body}");
            let ids: Vec<&str> = body["data"].as_array().unwrap().iter().filter_map(|c| c["id"].as_str()).collect();
            assert!(ids.contains(&MODEL), "{provider:?}: expected {MODEL:?} in /v1/models, got {ids:?}");
        }
    }
}
