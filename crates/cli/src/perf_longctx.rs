// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Engines for `brain perf run longctx`: the model families behind the
//! engine-agnostic [`perf::scenarios::longctx::LongContextEngine`] seam.
//!
//! Swedish Embedded AB implements long-context inference serving and the
//! measurement that shows what a given GPU can really sustain for its clients.
//! If your team needs expertise in sizing and tuning LLM serving on a specific
//! accelerator then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! `perf` owns the scenario and depends on no model crate; this module is the
//! wiring, one adapter per family:
//!
//! * [`Qwen35Engine`] - the Qwen3.8 dense hybrid (Gated DeltaNet + GQA) GGUF
//!   resident, layer-sharded over the GPUs it fits on. Its decode context is
//!   synthetic (fresh caches, decode work at a real position).
//! * [`QwenPagedEngine`] - the Qwen3 paged serving engine. Same scenario, same
//!   artifact, different model family; its decode context is synthetic too
//!   (the block table is extended to the position without prefilling).
//!
//! * [`Qwen35MoeEngine`] - Qwen3.6-35B-A3B (256-expert sparse MoE over Gated
//!   DeltaNet + GQA layers) straight from its released GGUF on one GPU, through
//!   the paged serving engine. Its decode context is synthetic too.
//!
//! All are GPU only: `plan` counts device memory and nothing else, and a
//! sizing that does not fit ends the sweep rather than spilling to the host.

use std::time::Instant;

use capability::Invocation;
use checkpoint::gguf::MmapGguf;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use perf::scenarios::longctx::{ContextKind, EngineInfo, Fit, LongContextEngine};
use qwen35::int8_gguf_resident::{serving_cost, Qwen35GgufInstance, Qwen35GgufResident, ServingShape};
use qwen35::resident_checkpoint::ResidentCheckpoint;
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

/// Bytes kept free per card for the driver and for activations the placement
/// models do not count: `brain serve`'s own default reserve.
const RESERVE: u64 = 2 << 30;

/// The GPUs `--device` selected, with the bytes usable after the reserve.
fn gpu_budget() -> Vec<(u32, String, u64)> {
    let all: Vec<(u32, String, u64)> = gpu_core::devices::gpus().iter().map(|d| (d.index, d.identity.name.clone(), d.identity.vram_bytes)).collect();
    selected_budget(&all, crate::compute_set().map(|s| s.gpus.as_slice()))
}

/// `gpus` narrowed to `selected` (every GPU when `--device` has not resolved), less [`RESERVE`] each. A card
/// outside the selection is never planned on: another process may own it, and its bytes counted as usable made
/// the sweep load a model across both cards of a machine whose `--device` named one.
fn selected_budget(gpus: &[(u32, String, u64)], selected: Option<&[u32]>) -> Vec<(u32, String, u64)> {
    gpus.iter()
        .filter(|(i, _, _)| selected.is_none_or(|sel| sel.contains(i)))
        .map(|(i, name, vram)| (*i, name.clone(), vram.saturating_sub(RESERVE)))
        .filter(|&(_, _, usable)| usable > 0)
        .collect()
}

/// Device memory in use right now, summed over the GPUs, as the driver's own
/// tool reports it - or `None` where that tool is absent.
fn driver_used_bytes() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).output().ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    let mut total = 0u64;
    let mut any = false;
    for line in text.lines() {
        total += line.trim().parse::<u64>().ok()? << 20;
        any = true;
    }
    any.then_some(total)
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1e3
}

// ------------------------------------------------------------------ qwen35

/// The Qwen3.8 GGUF resident (`qwen35:<gguf>[:kvf32|:kvbf16|:kvint8]`).
pub struct Qwen35Engine {
    path: String,
    tier: model::ops::TierPolicy,
    /// The tier the lm_head is held at (int8 for a GGUF, as stored for an HF release).
    head: gpu_core::select::Dtype,
    /// How the GQA layers' K/V are stored - the target's `:kv*` suffix.
    kv_tier: model::kv_tier::KvTier,
    gpus: Vec<(u32, String, u64)>,
    max_context: Option<u32>,
    inst: Option<Qwen35GgufInstance>,
    /// Real in-vocab token ids prompts are cut from, tokenised once.
    prompt_seed: Option<Vec<u32>>,
}

/// The artifact's name for a KV tier: `fp32`/`bf16`/`int8`/`fp8`, the spellings the
/// paged engine's own fingerprint and the weight tier already use.
fn kv_label(kv: model::kv_tier::KvTier) -> &'static str {
    use model::kv_tier::KvElement;
    match kv.element() {
        KvElement::F32 => "fp32",
        KvElement::Bf16 => "bf16",
        KvElement::Int8 => "int8",
        KvElement::Fp8E4m3 => "fp8",
    }
}

impl Qwen35Engine {
    /// Open a Qwen3.8 GGUF for long-context measurement. `kv_tier` is the
    /// parsed KV storage choice (f32 when unset - the historical default
    /// the removed `BRAIN_QWEN35_KV` reader used); every weight is held as
    /// the checkpoint stores it, as `brain serve` holds it.
    pub fn open(path: &str, kv_tier: model::kv_tier::KvTier) -> Result<Qwen35Engine, String> {
        let gpus = gpu_budget();
        if gpus.is_empty() {
            return Err("no GPU with queryable memory: longctx is GPU-only and has no host fallback".into());
        }
        // The open error already names the path ("gguf: open …");
        // wrapping it again double-prefixed every message.
        let ckpt = ResidentCheckpoint::open(path).map_err(|e| e.to_string())?;
        let cfg = ckpt.config(1)?;
        Ok(Qwen35Engine {
            path: path.to_string(),
            tier: ckpt.held_tiers(&cfg)?,
            head: ckpt.head_tier(),
            kv_tier,
            gpus,
            max_context: Some(cfg.max_position_embeddings).filter(|&n| n > 0),
            inst: None,
            prompt_seed: None,
        })
    }

    fn devices(&self) -> Vec<(Device, u64)> {
        self.gpus.iter().map(|&(i, _, usable)| (Device::Gpu(i), usable)).collect()
    }

    fn resident(&self, batch: u32, context: u32) -> Qwen35GgufResident {
        Qwen35GgufResident::new(self.path.clone(), self.devices(), context, self.tier.clone()).with_kv_tier(self.kv_tier).with_max_batch(batch)
    }

    /// The devices the placement planner admits this sizing on; empty when it
    /// does not fit.
    fn placement(&self, r: &Qwen35GgufResident) -> Vec<Device> {
        r.estimate_multi(&r.instance_key("generate", &Invocation::new())).devices().collect()
    }
}

/// A checkpoint's name in a result: its last three path components without
/// the extension - `<org>/<repo>/<file>` for a file in the model store - so
/// two releases of one model are told apart.
fn checkpoint_label(path: &str) -> String {
    let p = std::path::Path::new(path);
    // Only a GGUF has an extension to drop: a directory `Qwen3.8-27B-FP8` has dots in its name.
    let p = if p.extension().is_some_and(|e| e == "gguf") { p.with_extension("") } else { p.to_path_buf() };
    let parts: Vec<String> = p.components().rev().take(3).map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    parts.into_iter().rev().collect::<Vec<_>>().join("/")
}

impl LongContextEngine for Qwen35Engine {
    fn describe(&self) -> EngineInfo {
        EngineInfo {
            model: checkpoint_label(&self.path),
            weight_tier: self.tier.summary(),
            // The GQA window is stored in the chosen KV tier; the recurrent
            // state is fp32 words whatever it is.
            kv_precision: kv_label(self.kv_tier).into(),
            backend: gpu_core::backend_name().to_string(),
            devices: self.gpus.iter().map(|(i, name, _)| format!("gpu{i} {name}")).collect(),
            context_kind: ContextKind::Synthetic,
        }
    }

    fn max_context(&self) -> Option<u32> {
        self.max_context
    }

    fn plan(&mut self, batch: u32, context: u32) -> Result<Fit, String> {
        let ckpt = ResidentCheckpoint::open(&self.path).map_err(|e| format!("open {}: {e}", self.path))?;
        let cfg = ckpt.config(context)?;
        let needed = serving_cost(&cfg, &ServingShape { head: self.head, ..ServingShape::new(context, self.tier.clone(), self.kv_tier, batch) }).total();
        let usable = self.gpus.iter().map(|g| g.2).sum();
        let fits = !self.placement(&self.resident(batch, context)).is_empty();
        Ok(Fit { fits, needed_bytes: needed, usable_bytes: usable })
    }

    fn load(&mut self, batch: u32, context: u32) -> Result<(), String> {
        let r = self.resident(batch, context);
        let placed = self.placement(&r);
        if placed.is_empty() {
            return Err("does not fit in GPU memory".into());
        }
        self.inst = Some(r.activate_owned(&placed)?);
        Ok(())
    }

    fn prefill(&mut self, prompt_tokens: u32) -> Result<f64, String> {
        let inst = self.inst.as_ref().ok_or("prefill before load")?;
        let seed = self.prompt_seed.get_or_insert_with(|| {
            inst.tokenize("The quick brown fox jumps over the lazy dog while a kalman filter estimates the state of a noisy system. ")
        });
        let prompt: Vec<u32> = seed.iter().cycle().take(prompt_tokens as usize).copied().collect();
        inst.prefill_timed(&prompt)
    }

    fn decode_step(&mut self, positions: &[u32]) -> Result<f64, String> {
        let inst = self.inst.as_ref().ok_or("decode before load")?;
        let tokens = vec![1u32; positions.len()];
        let t = Instant::now();
        // The step a served greedy request runs: the layers, then the device head,
        // with only the chosen ids read back. Timing `decode_batch_at` instead
        // priced a `[batch, vocab]` logits readback no served step makes, which
        // doubled the 32-row step. The read and the drain make the wall time
        // cover the device's work.
        inst.decode_batch_greedy_at(&tokens, positions)?;
        inst.poll_wait();
        Ok(elapsed_ms(t))
    }

    fn device_used_bytes(&self) -> Option<u64> {
        driver_used_bytes()
    }

    fn devices(&self) -> Vec<&gpu_core::Gpu> {
        self.inst.as_ref().map_or_else(Vec::new, |i| (0..i.stages()).map(|s| i.stage_gpu(s)).collect())
    }

    fn unload(&mut self) {
        self.inst = None;
    }
}

// ------------------------------------------------------------ qwen35moe

/// Positions past the measured context a loaded engine keeps room for: the
/// scenario's warm-up and timed steps each advance every sequence by one.
const MOE_DECODE_HEADROOM: u32 = 256;

/// Device bytes beyond weights and KV that a loaded engine holds: the fixed
/// activation scratch and the output logits (`FIXED`), and the per-sequence
/// scratch of a recorded decode step and of the head's `[rows, vocab]` logits
/// (`PER_SEQUENCE`). Calibrated against the driver's own used-memory reading
/// on a GH200 (see the test below, which pins the shape of the estimate).
const MOE_SCRATCH_FIXED: u64 = 3 << 30;
const MOE_SCRATCH_PER_SEQUENCE: u64 = 24 << 20;

/// The Qwen3.6-35B-A3B GGUF (`qwen35moe:<gguf>` / `qwen35moe-gguf`).
pub struct Qwen35MoeEngine {
    path: String,
    cfg: qwen35moe::config::Qwen35Config,
    tier: model::ops::TierPolicy,
    kv_tier: model::kv_tier::KvTier,
    gpu: (u32, String, u64),
    eng: Option<qwen35moe::serve::Engine>,
    tables: Vec<BlockTable>,
    /// Real in-vocab token ids prompts are cut from, tokenised once.
    prompt_seed: Option<Vec<u32>>,
}

impl Qwen35MoeEngine {
    pub fn open(path: &str) -> Result<Qwen35MoeEngine, String> {
        let gpu = gpu_budget().into_iter().next().ok_or("no GPU with queryable memory: longctx is GPU-only and has no host fallback")?;
        let mg = MmapGguf::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let cfg = qwen35moe::gguf_load::resident_config(&mg, 1)?;
        Ok(Qwen35MoeEngine {
            path: path.to_string(),
            cfg,
            tier: qwen35moe::gguf_load::tier_from_env(),
            kv_tier: qwen35moe::gguf_load::kv_tier_from_env()?,
            gpu,
            eng: None,
            tables: Vec::new(),
            prompt_seed: None,
        })
    }

    /// Per-sequence capacity an engine serving `context`-token decode needs.
    fn capacity(context: u32) -> u32 {
        context.saturating_add(MOE_DECODE_HEADROOM)
    }

    fn needed_bytes(&self, batch: u32, context: u32) -> u64 {
        qwen35moe::serve::weight_bytes(&self.cfg, &self.tier)
            + qwen35moe::serve::kv_pool_bytes(&self.cfg, self.kv_tier, batch, Self::capacity(context))
            + MOE_SCRATCH_FIXED
            + batch as u64 * MOE_SCRATCH_PER_SEQUENCE
    }
}

impl LongContextEngine for Qwen35MoeEngine {
    fn describe(&self) -> EngineInfo {
        EngineInfo {
            model: qwen35moe::gguf_load::MODEL.to_string(),
            weight_tier: self.tier.summary(),
            kv_precision: kv_label(self.kv_tier).into(),
            backend: gpu_core::backend_name().to_string(),
            devices: vec![format!("gpu{} {}", self.gpu.0, self.gpu.1)],
            context_kind: ContextKind::Synthetic,
        }
    }

    fn max_context(&self) -> Option<u32> {
        Some(self.cfg.max_position_embeddings).filter(|&n| n > 0)
    }

    fn plan(&mut self, batch: u32, context: u32) -> Result<Fit, String> {
        let needed = self.needed_bytes(batch, context);
        Ok(Fit { fits: needed <= self.gpu.2, needed_bytes: needed, usable_bytes: self.gpu.2 })
    }

    fn load(&mut self, batch: u32, context: u32) -> Result<(), String> {
        self.eng = None;
        self.tables.clear();
        let mg = MmapGguf::open(&self.path).map_err(|e| format!("open {}: {e}", self.path))?;
        let cfg = qwen35moe::gguf_load::resident_config(&mg, Self::capacity(context))?;
        let src = qwen35moe::gguf_load::source(&mg, &cfg)?;
        let opts = qwen35moe::serve::EngineOptions::new(Self::capacity(context), batch).with_tier(self.tier.clone()).with_kv_tier(self.kv_tier);
        // Allocation failure on the device surfaces as a panic from the engine
        // constructor; it is the out-of-memory boundary, not a crash.
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| qwen35moe::serve::Engine::from_source(cfg, &src, opts))).map_err(|p| {
            p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "engine construction panicked".into())
        })?;
        if self.prompt_seed.is_none() {
            use data::tokenizer::Tokenizer;
            let text = "The quick brown fox jumps over the lazy dog while a kalman filter estimates the state of a noisy system. ";
            self.prompt_seed = Some(qwen35moe::gguf_load::tokenizer(&mg)?.encode(text));
        }
        self.eng = Some(built);
        Ok(())
    }

    fn prefill(&mut self, prompt_tokens: u32) -> Result<f64, String> {
        let eng = self.eng.as_mut().ok_or("prefill before load")?;
        let seed = self.prompt_seed.as_ref().ok_or("prefill before load")?;
        let prompt: Vec<u32> = seed.iter().cycle().take(prompt_tokens as usize).copied().collect();
        let mut table = BlockTable::new();
        let t = Instant::now();
        // Returns the last hidden state to the host, so the clock covers the device.
        eng.prefill(&mut table, &prompt);
        let secs = t.elapsed().as_secs_f64();
        eng.release_table(&mut table);
        Ok(secs)
    }

    fn decode_step(&mut self, positions: &[u32]) -> Result<f64, String> {
        let eng = self.eng.as_mut().ok_or("decode before load")?;
        self.tables.resize_with(positions.len(), BlockTable::new);
        // A sequence is brought to its position without computing the context
        // (`Engine::admit_synthetic`); consecutive steps then find it there.
        for (table, &p) in self.tables.iter_mut().zip(positions) {
            eng.admit_synthetic(table, p)?;
        }
        let tokens = vec![1u32; positions.len()];
        let mut rows: Vec<&mut BlockTable> = self.tables.iter_mut().collect();
        let t = Instant::now();
        // Greedy decode reads each row's next token back to the host.
        let out = eng.forward_batched_greedy(&mut rows, &tokens);
        let ms = elapsed_ms(t);
        if out.len() != positions.len() {
            return Err(format!("engine returned {} tokens for {} rows", out.len(), positions.len()));
        }
        Ok(ms)
    }

    fn device_used_bytes(&self) -> Option<u64> {
        driver_used_bytes()
    }

    fn devices(&self) -> Vec<&gpu_core::Gpu> {
        self.eng.as_ref().map(|e| vec![e.gpu()]).unwrap_or_default()
    }

    fn unload(&mut self) {
        // Tables first: they name blocks of the pool the engine owns.
        self.tables.clear();
        self.eng = None;
    }

    fn release_batch(&mut self) {
        if let Some(eng) = self.eng.as_mut() {
            for t in &mut self.tables {
                eng.release_table(t);
            }
        }
        self.tables.clear();
    }
}

// ------------------------------------------------------------- qwen3 paged

/// Tokens per KV block, and the prefill chunk: `brain serve`'s defaults.
const BLOCK_SIZE: u32 = 16;
const PREFILL_CHUNK: u32 = 512;

/// The Qwen3 paged serving engine (`qwen:<weights>[:i8w][:kvf32]`).
pub struct QwenPagedEngine {
    weights: String,
    cfg: qwen3::QwenConfig,
    weights_int8: bool,
    kv_int8: bool,
    /// Device bytes of the first GPU after the reserve: the engine runs on one
    /// device.
    gpu: (u32, String, u64),
    /// The checkpoint, opened once and reused by every `load` so the ladder
    /// does not reopen it per rung. Each engine reads its tensors from it one
    /// at a time; no whole-model host copy is kept between rungs.
    source: Option<Box<dyn checkpoint::TensorSource>>,
    /// The attention paths a loaded engine fused, which decide its scratch
    /// size; before the first load the planner assumes the larger unfused
    /// worst case.
    fused_attention: Option<qwen3::serve::FusedAttention>,
    eng: Option<qwen3::serve::Engine>,
    tables: Vec<BlockTable>,
}

impl QwenPagedEngine {
    pub fn open(spec: &str) -> Result<QwenPagedEngine, String> {
        let (weights, weights_int8, kv_fp32) = crate::perf_cli::spec_flags(spec);
        let gpu = gpu_budget().into_iter().next().ok_or("no GPU with queryable memory: longctx is GPU-only and has no host fallback")?;
        let cfg = qwen3::checkpoint_config(weights)?;
        let kv_int8 = crate::perf_cli::resolve_kv_int8(&cfg, kv_fp32, weights);
        Ok(QwenPagedEngine { weights: weights.to_string(), cfg, weights_int8, kv_int8, gpu, source: None, fused_attention: None, eng: None, tables: Vec::new() })
    }

    /// `(blocks per sequence, pool blocks, prefill chunk)` for a sizing: room
    /// for `context` tokens in each of `batch` sequences plus one spare block
    /// each.
    fn geometry(batch: u32, context: u32) -> (u32, u32, u32) {
        let per_seq = context.div_ceil(BLOCK_SIZE) + 1;
        (per_seq, per_seq * batch + batch, PREFILL_CHUNK.min(context))
    }

    fn scratch_bytes(&self, batch: u32, context: u32, fused: qwen3::serve::FusedAttention) -> u64 {
        let (per_seq, _, chunk) = Self::geometry(batch, context);
        qwen3::serve::paged_attn_scratch_bytes(&self.cfg, batch, chunk, per_seq * BLOCK_SIZE, fused)
    }
}

impl LongContextEngine for QwenPagedEngine {
    fn describe(&self) -> EngineInfo {
        EngineInfo {
            model: format!("qwen3:{}", std::path::Path::new(&self.weights).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()),
            weight_tier: if self.weights_int8 { "int8" } else { "fp32" }.into(),
            kv_precision: if self.kv_int8 { "int8" } else { "fp32" }.into(),
            backend: gpu_core::backend_name().to_string(),
            devices: vec![format!("gpu{} {}", self.gpu.0, self.gpu.1)],
            context_kind: ContextKind::Synthetic,
        }
    }

    fn max_context(&self) -> Option<u32> {
        Some(self.cfg.max_position_embeddings).filter(|&n| n > 0)
    }

    fn plan(&mut self, batch: u32, context: u32) -> Result<Fit, String> {
        let (_, num_blocks, _) = Self::geometry(batch, context);
        let weights = if self.weights_int8 { catalog::resident_llm::weights_int8_bytes(&self.cfg) } else { catalog::resident_llm::weights_fp32_bytes(&self.cfg) };
        let kv = qwen3::serve::kv_pool_bytes(&self.cfg, BLOCK_SIZE, num_blocks, self.kv_int8);
        let needed = weights + kv + self.scratch_bytes(batch, context, self.fused_attention.unwrap_or_default());
        Ok(Fit { fits: needed <= self.gpu.2, needed_bytes: needed, usable_bytes: self.gpu.2 })
    }

    fn load(&mut self, batch: u32, context: u32) -> Result<(), String> {
        self.eng = None;
        self.tables.clear();
        if self.source.is_none() {
            self.source = Some(qwen3::open_checkpoint(&self.weights)?.1);
        }
        let source = self.source.as_deref().expect("opened above");
        let (per_seq, num_blocks, chunk) = Self::geometry(batch, context);
        let (cfg, kv_int8) = (self.cfg.clone(), self.kv_int8);
        let tier = if self.weights_int8 { qwen3::Dtype::I8 } else { qwen3::Dtype::F32 };
        // Allocation failure on the device surfaces as a panic from the engine
        // constructor; it is the out-of-memory boundary, not a crash.
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            qwen3::serve::Engine::from_source_tier(cfg, source, BLOCK_SIZE, num_blocks, batch, per_seq, chunk, kv_int8, tier)
        }))
        .map_err(|p| {
            p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "engine construction panicked".into())
        })?;
        // Learn which scratch formula the device really took, so later plans
        // are exact instead of worst-case.
        self.fused_attention = Some(built.fused_attention());
        self.eng = Some(built);
        Ok(())
    }

    fn prefill(&mut self, prompt_tokens: u32) -> Result<f64, String> {
        let eng = self.eng.as_mut().ok_or("prefill before load")?;
        let vocab = eng.vocab() as u64;
        // Deterministic in-vocab ids: prefill cost does not depend on which.
        let prompt: Vec<u32> = (0..prompt_tokens as u64).map(|i| ((i * 7919 + 13) % vocab) as u32).collect();
        let mut table = BlockTable::new();
        let t = Instant::now();
        // Returns the last hidden state to the host, so the clock covers the device.
        eng.prefill_for_perf(&mut table, &prompt);
        let secs = t.elapsed().as_secs_f64();
        // Free the blocks and drop the prefix-cache entries this prompt left, so
        // the next prefill is cold and the pool is whole again.
        eng.release_table(&mut table);
        eng.reclaim_prefix(u32::MAX);
        Ok(secs)
    }

    fn decode_step(&mut self, positions: &[u32]) -> Result<f64, String> {
        let eng = self.eng.as_mut().ok_or("decode before load")?;
        self.tables.resize_with(positions.len(), BlockTable::new);
        // The paged engine decodes at the end of each sequence, so a sequence is
        // brought to its position by reserving (or truncating) blocks - the
        // synthetic context. Consecutive steps then find it already there.
        for (table, &p) in self.tables.iter_mut().zip(positions) {
            if table.len() < p {
                table.reserve(p - table.len(), eng.alloc_mut())?;
            } else if table.len() > p {
                table.truncate(p, eng.alloc_mut());
            }
        }
        let tokens = vec![1u32; positions.len()];
        let mut rows: Vec<&mut BlockTable> = self.tables.iter_mut().collect();
        let t = Instant::now();
        // Greedy decode reads each row's next token back to the host.
        let out = eng.forward_batched_greedy(&mut rows, &tokens);
        let ms = elapsed_ms(t);
        if out.len() != positions.len() {
            return Err(format!("engine returned {} tokens for {} rows", out.len(), positions.len()));
        }
        Ok(ms)
    }

    fn device_used_bytes(&self) -> Option<u64> {
        driver_used_bytes()
    }

    fn devices(&self) -> Vec<&gpu_core::Gpu> {
        self.eng.as_ref().map(|e| vec![e.gpu()]).unwrap_or_default()
    }

    fn unload(&mut self) {
        self.tables.clear();
        self.eng = None;
    }

    fn release_batch(&mut self) {
        if let Some(eng) = self.eng.as_mut() {
            for t in &mut self.tables {
                eng.release_table(t);
            }
        }
        self.tables.clear();
    }
}

// -------------------------------------------------------------------- spec

/// The `longctx` flags as given on the command line; `None` means "default".
#[derive(Default)]
pub struct Args {
    pub context: Option<u32>,
    pub ladder: Option<String>,
    pub prefill: Option<String>,
    pub steps: Option<u32>,
    /// `--kernels`: profile each batch size's decode steps per kernel.
    pub kernels: bool,
}

/// Defaults when a flag is absent. A context is always stated in the artifact,
/// so a default only has to be sensible, not authoritative.
const DEFAULT_CONTEXT: u32 = 8192;
const DEFAULT_STEPS: u32 = 8;

/// Resolve flags and `--smoke` into scenario options. Smoke shrinks the run to
/// seconds-per-rung (short context, two rungs, two timed steps, one short
/// prefill length) and the artifact is labelled `smoke`.
pub fn options(args: &Args, device: &str, smoke: bool) -> Result<perf::scenarios::longctx::Options, String> {
    use perf::scenarios::longctx::{parse_list, DEFAULT_LADDER};
    let mut ladder = match &args.ladder {
        Some(spec) => parse_list(spec, "--ladder")?,
        None => DEFAULT_LADDER.to_vec(),
    };
    let mut prefill = match &args.prefill {
        Some(spec) => parse_list(spec, "--prefill")?,
        None => Vec::new(),
    };
    let mut context = args.context.unwrap_or(DEFAULT_CONTEXT);
    let mut steps = args.steps.unwrap_or(DEFAULT_STEPS);
    if smoke {
        context = context.min(256);
        steps = steps.min(2);
        ladder.truncate(2);
        prefill.truncate(1);
        prefill.iter_mut().for_each(|n| *n = (*n).min(128));
    }
    Ok(perf::scenarios::longctx::Options { context, ladder, prefill, steps, device: device.to_string(), smoke, kernels: args.kernels })
}

/// Run the scenario on the engine `spec` names.
pub fn run(spec: &str, args: &Args, device: &str, smoke: bool) -> Result<perf::schema::Artifact, String> {
    let opt = options(args, device, smoke)?;
    let mut engine = build_engine(spec)?;
    perf::scenarios::longctx::run(engine.as_mut(), &opt)
}

/// Build the engine a `--target` names, or say why it cannot run `longctx`.
///
/// The path-bearing targets (`qwen35:<gguf>`, `qwen35moe:<gguf>`,
/// `qwen:<weights>`) replaced the bare `-gguf` forms that read env vars: the
/// checkpoint is named on the command line, as the env-variable phase-out
/// wants, so artifacts record exactly which file was measured.
pub fn build_engine(spec: &str) -> Result<Box<dyn LongContextEngine>, String> {
    if let Some(rest) = spec.strip_prefix("qwen35:") {
        // `:kv<tier>` names the GQA cache storage; f32 when absent.
        let (path, kv) = crate::perf_qwen35::parse_spec(rest)?;
        let kv = kv.unwrap_or(model::kv_tier::KvTier::F32);
        return Ok(Box::new(Qwen35Engine::open(path, kv)?));
    }
    if let Some(path) = spec.strip_prefix("qwen35moe:") {
        return Ok(Box::new(Qwen35MoeEngine::open(path)?));
    }
    if spec == "qwen35moe-gguf" {
        let path = std::env::var(qwen35moe::gguf_load::GGUF_ENV).map_err(|_| format!("target qwen35moe-gguf reads {}; set it to a Qwen3.6-35B-A3B GGUF", qwen35moe::gguf_load::GGUF_ENV))?;
        return Ok(Box::new(Qwen35MoeEngine::open(&path)?));
    }
    if let Some(weights) = spec.strip_prefix("qwen:") {
        return Ok(Box::new(QwenPagedEngine::open(weights)?));
    }
    Err(format!("longctx does not support target {spec:?}: use qwen35:<gguf>[:kvf32|:kvbf16|:kvint8], qwen35moe:<gguf>, qwen35moe-gguf or qwen:<weights>[:i8w][:kvf32]"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sweep_plans_only_on_the_cards_device_selected() {
        let gpus = vec![(0, "a".to_owned(), 96 << 30), (1, "b".to_owned(), 96 << 30)];
        let one = selected_budget(&gpus, Some(&[1]));
        assert_eq!(one.iter().map(|g| g.0).collect::<Vec<_>>(), vec![1], "--device gpu1 never counts gpu0's memory");
        assert_eq!(one[0].2, (96 << 30) - RESERVE);
        assert_eq!(selected_budget(&gpus, None).len(), 2, "no selection yet is every card");
    }

    #[test]
    fn smoke_shrinks_the_run_and_flags_are_validated() {
        let args = Args { context: Some(131_072), ladder: Some("1,2,4,8".into()), prefill: Some("4096,8192".into()), steps: Some(16), kernels: false };
        let o = options(&args, "gpu", true).unwrap();
        assert!(o.smoke && o.context <= 256 && o.steps == 2 && o.ladder == vec![1, 2] && o.prefill == vec![128]);
        let full = options(&args, "gpu", false).unwrap();
        assert_eq!((full.context, full.steps, full.ladder, full.prefill), (131_072, 16, vec![1, 2, 4, 8], vec![4096, 8192]));
        assert!(options(&Args { ladder: Some("1,zero".into()), ..Default::default() }, "gpu", false).is_err());
        let d = options(&Args::default(), "gpu", false).unwrap();
        assert!(d.prefill.is_empty() && d.ladder.first() == Some(&1));
    }

    /// A result names the checkpoint it ran, so two releases of one model
    /// are told apart.
    #[test]
    fn a_qwen35_result_is_named_by_its_checkpoint() {
        assert_eq!(checkpoint_label("/m/Abiray/Qwen3.8-27B-Q4_K_M-GGUF/Q4_K_M.gguf"), "Abiray/Qwen3.8-27B-Q4_K_M-GGUF/Q4_K_M");
        assert_eq!(checkpoint_label("Q8_0.gguf"), "Q8_0");
    }

    /// The artifact's fingerprint must say how the KV cache was stored: a
    /// batch ceiling measured on a compact cache is not comparable to one
    /// measured on `f32`, and `fp32` hard-coded would label them identically.
    #[test]
    fn the_qwen35_engine_reports_the_kv_tier_it_plans_and_runs_with() {
        for kv in model::kv_tier::KvTier::ALL {
            let engine = Qwen35Engine { path: String::new(), tier: model::ops::TierPolicy::uniform(gpu_core::select::Dtype::I8), head: gpu_core::select::Dtype::I8, kv_tier: kv, gpus: Vec::new(), max_context: None, inst: None, prompt_seed: None };
            assert_eq!(engine.describe().kv_precision, kv_label(kv));
            assert!(model::kv_tier::KvTier::ALL.iter().map(|k| kv_label(*k)).collect::<std::collections::HashSet<_>>().len() == model::kv_tier::KvTier::ALL.len(), "every tier needs its own label");
        }
    }

    fn moe_engine(tier: model::ops::TierPolicy, kv: model::kv_tier::KvTier) -> Qwen35MoeEngine {
        let cfg = qwen35moe::config::Qwen35Config::qwen35_35b_a3b();
        Qwen35MoeEngine { path: String::new(), cfg, tier, kv_tier: kv, gpu: (0, "test".into(), 96 << 30), eng: None, tables: Vec::new(), prompt_seed: None }
    }

    /// The MoE planner has to say what the run will hold: the released Q8_0
    /// file is 36.9 GB and brain's int8 layout carries an f32 scale per 32
    /// weights, so the weights are ~39 GB; the KV pool grows with the batch and
    /// the context, and a compact tier is smaller than fp32. The artifact names
    /// the tier and the kv precision it ran with.
    #[test]
    fn the_qwen35moe_planner_counts_weights_kv_and_reports_its_tiers() {
        use model::kv_tier::KvTier;
        let i8 = || model::ops::TierPolicy::uniform(gpu_core::select::Dtype::I8);
        let mut e = moe_engine(i8(), KvTier::F32);
        let weights = qwen35moe::serve::weight_bytes(&e.cfg, &e.tier) as f64 / 1e9;
        assert!((36.0..42.0).contains(&weights), "int8 weights are {weights:.1} GB");
        let small = e.plan(1, 1024).unwrap();
        assert!(small.needed_bytes > qwen35moe::serve::weight_bytes(&e.cfg, &e.tier));
        assert!(e.plan(8, 1024).unwrap().needed_bytes > small.needed_bytes);
        assert!(e.plan(1, 131_072).unwrap().needed_bytes > small.needed_bytes);
        let fp32_kv = e.plan(8, 65_536).unwrap().needed_bytes;
        let mut compact = moe_engine(i8(), KvTier::INT8);
        assert!(compact.plan(8, 65_536).unwrap().needed_bytes < fp32_kv);
        assert_eq!(compact.describe().kv_precision, "int8");
        assert_eq!(compact.describe().weight_tier, "i8");
        assert_eq!(compact.describe().context_kind, ContextKind::Synthetic);
        // An fp32 model of this size cannot fit a card, and the planner says so.
        let mut f32_model = moe_engine(model::ops::TierPolicy::uniform(gpu_core::select::Dtype::F32), KvTier::F32);
        assert!(!f32_model.plan(1, 1024).unwrap().fits);
    }

    #[test]
    fn targets_outside_the_supported_families_are_refused_by_name() {
        let e = build_engine("yolo:/nowhere").err().unwrap();
        assert!(e.contains("yolo:/nowhere") && e.contains("qwen35:<gguf>"), "{e}");
    }

    /// The env-reading `qwen35-gguf` target is gone: the checkpoint is part of
    /// the spec, and the `:kv*` suffix is the KV tier as parsed data.
    #[test]
    fn the_qwen35_target_carries_the_path_and_kv_tier() {
        let e = build_engine("qwen35-gguf").err().unwrap();
        assert!(e.contains("qwen35-gguf") && e.contains("qwen35:<gguf>"), "{e}");
        // `Qwen35Engine::open` propagates `MmapGguf::open`'s error verbatim
        // (its own GPU-only gate runs first, so the propagation is asserted
        // through the mapping constructor directly - this suite must not
        // depend on card visibility; the tier threading has its own test).
        let open_err = MmapGguf::open("/nonexistent/none.gguf").err().unwrap();
        assert!(open_err.contains("gguf: open /nonexistent/none.gguf") && open_err.contains("No such file or directory"), "{open_err}");
    }

    #[test]
    fn the_paged_pool_holds_every_sequence_to_the_full_context() {
        for (batch, ctx) in [(1u32, 1u32), (1, 16), (3, 17), (8, 4096)] {
            let (per_seq, blocks, chunk) = QwenPagedEngine::geometry(batch, ctx);
            assert!(per_seq * BLOCK_SIZE > ctx, "a sequence of {ctx} tokens must fit its blocks");
            assert!(blocks >= per_seq * batch, "every sequence needs its own blocks");
            assert!(chunk >= 1 && chunk <= ctx.max(1));
        }
    }
}
