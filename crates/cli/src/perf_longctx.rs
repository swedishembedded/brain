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
//! Both are GPU only: `plan` counts device memory and nothing else, and a
//! sizing that does not fit ends the sweep rather than spilling to the host.

use std::collections::HashMap;
use std::time::Instant;

use capability::Invocation;
use checkpoint::gguf::MmapGguf;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use perf::scenarios::longctx::{ContextKind, EngineInfo, Fit, LongContextEngine};
use qwen35::int8_gguf_resident::{layer_cost, resident_config, Qwen35GgufInstance, Qwen35GgufResident};
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

/// Bytes kept free per card for the driver and for activations the placement
/// models do not count: `brain serve`'s own default reserve.
const RESERVE: u64 = 2 << 30;

/// Every GPU the process can see, with the bytes usable after the reserve.
fn gpu_budget() -> Vec<(u32, String, u64)> {
    gpu_core::devices::gpus()
        .iter()
        .map(|d| (d.index, d.identity.name.clone(), d.identity.vram_bytes.saturating_sub(RESERVE)))
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

/// The Qwen3.8 GGUF resident (`qwen35:<gguf>` / `qwen35-gguf`).
pub struct Qwen35Engine {
    path: String,
    tier: model::ops::TierPolicy,
    gpus: Vec<(u32, String, u64)>,
    max_context: Option<u32>,
    inst: Option<Qwen35GgufInstance>,
    /// Real in-vocab token ids prompts are cut from, tokenised once.
    prompt_seed: Option<Vec<u32>>,
}

impl Qwen35Engine {
    pub fn open(path: &str) -> Result<Qwen35Engine, String> {
        let gpus = gpu_budget();
        if gpus.is_empty() {
            return Err("no GPU with queryable memory: longctx is GPU-only and has no host fallback".into());
        }
        let mg = MmapGguf::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let cfg = resident_config(&mg, 1)?;
        Ok(Qwen35Engine {
            path: path.to_string(),
            tier: Qwen35GgufResident::tier_from_env(),
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
        Qwen35GgufResident::new(self.path.clone(), self.devices(), context, self.tier.clone()).with_max_batch(batch)
    }

    /// The devices the placement planner admits this sizing on; empty when it
    /// does not fit.
    fn placement(&self, r: &Qwen35GgufResident) -> Vec<Device> {
        r.estimate_multi(&r.instance_key("generate", &Invocation::new())).devices().collect()
    }
}

impl LongContextEngine for Qwen35Engine {
    fn describe(&self) -> EngineInfo {
        EngineInfo {
            model: qwen35::int8_gguf_resident::MODEL.to_string(),
            weight_tier: self.tier.describe(),
            // The GQA window and the recurrent state are fp32 words.
            kv_precision: "fp32".into(),
            backend: gpu_core::backend_name().to_string(),
            devices: self.gpus.iter().map(|(i, name, _)| format!("gpu{i} {name}")).collect(),
            context_kind: ContextKind::Synthetic,
        }
    }

    fn max_context(&self) -> Option<u32> {
        self.max_context
    }

    fn plan(&mut self, batch: u32, context: u32) -> Result<Fit, String> {
        let mg = MmapGguf::open(&self.path).map_err(|e| format!("open {}: {e}", self.path))?;
        let cfg = resident_config(&mg, context)?;
        let needed = layer_cost(&cfg, context, &self.tier, batch).total();
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
        // Reading the logits back is part of a real step, and is also what makes
        // the wall time cover the device's work; the explicit drain makes that
        // true regardless.
        inst.decode_batch_at(&tokens, positions)?;
        inst.poll_wait();
        Ok(elapsed_ms(t))
    }

    fn device_used_bytes(&self) -> Option<u64> {
        driver_used_bytes()
    }

    fn unload(&mut self) {
        self.inst = None;
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
    /// Checkpoint tensors decoded to f32, read once and reused by every
    /// `load` so the ladder does not re-read the checkpoint per rung.
    tensors: Option<HashMap<String, Vec<f32>>>,
    /// `Some(true)` once a loaded engine showed the fused paged-attention
    /// scratch size; before the first load the planner assumes the larger
    /// unfused worst case.
    fused_scratch: Option<bool>,
    eng: Option<qwen3::serve::Engine>,
    tables: Vec<BlockTable>,
}

impl QwenPagedEngine {
    pub fn open(spec: &str) -> Result<QwenPagedEngine, String> {
        let (weights, weights_int8, kv_fp32) = crate::perf_cli::spec_flags(spec);
        let gpu = gpu_budget().into_iter().next().ok_or("no GPU with queryable memory: longctx is GPU-only and has no host fallback")?;
        let cfg = qwen3::checkpoint_config(weights)?;
        let kv_int8 = crate::perf_cli::resolve_kv_int8(&cfg, kv_fp32, weights);
        Ok(QwenPagedEngine { weights: weights.to_string(), cfg, weights_int8, kv_int8, gpu, tensors: None, fused_scratch: None, eng: None, tables: Vec::new() })
    }

    /// `(blocks per sequence, pool blocks, prefill chunk)` for a sizing: room
    /// for `context` tokens in each of `batch` sequences plus one spare block
    /// each.
    fn geometry(batch: u32, context: u32) -> (u32, u32, u32) {
        let per_seq = context.div_ceil(BLOCK_SIZE) + 1;
        (per_seq, per_seq * batch + batch, PREFILL_CHUNK.min(context))
    }

    fn scratch_bytes(&self, batch: u32, context: u32, fused: bool) -> u64 {
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
        let weights = if self.weights_int8 { crate::resident_llm::weights_int8_bytes(&self.cfg) } else { crate::resident_llm::weights_fp32_bytes(&self.cfg) };
        let kv = qwen3::serve::kv_pool_bytes(&self.cfg, BLOCK_SIZE, num_blocks, self.kv_int8);
        let needed = weights + kv + self.scratch_bytes(batch, context, self.fused_scratch == Some(true));
        Ok(Fit { fits: needed <= self.gpu.2, needed_bytes: needed, usable_bytes: self.gpu.2 })
    }

    fn load(&mut self, batch: u32, context: u32) -> Result<(), String> {
        self.eng = None;
        self.tables.clear();
        if self.tensors.is_none() {
            let (cfg, src) = qwen3::open_checkpoint(&self.weights)?;
            self.tensors = Some(qwen3::serve::Engine::tensors_from(&cfg, &*src)?);
        }
        let tensors = self.tensors.as_ref().expect("read above");
        let (per_seq, num_blocks, chunk) = Self::geometry(batch, context);
        let (cfg, kv_int8, weights_int8) = (self.cfg.clone(), self.kv_int8, self.weights_int8);
        // Allocation failure on the device surfaces as a panic from the engine
        // constructor; it is the out-of-memory boundary, not a crash.
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            qwen3::serve::Engine::from_map(cfg, tensors, BLOCK_SIZE, num_blocks, batch, per_seq, chunk, kv_int8, weights_int8)
        }))
        .map_err(|p| {
            p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "engine construction panicked".into())
        })?;
        // Learn which scratch formula the device really took, so later plans
        // are exact instead of worst-case.
        self.fused_scratch = Some(built.paged_attn_scratch_bytes() == self.scratch_bytes(batch, context, true));
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

    fn unload(&mut self) {
        self.tables.clear();
        self.eng = None;
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
    Ok(perf::scenarios::longctx::Options { context, ladder, prefill, steps, device: device.to_string(), smoke })
}

/// Run the scenario on the engine `spec` names.
pub fn run(spec: &str, args: &Args, device: &str, smoke: bool) -> Result<perf::schema::Artifact, String> {
    let opt = options(args, device, smoke)?;
    let mut engine = build_engine(spec)?;
    perf::scenarios::longctx::run(engine.as_mut(), &opt)
}

/// Build the engine a `--target` names, or say why it cannot run `longctx`.
pub fn build_engine(spec: &str) -> Result<Box<dyn LongContextEngine>, String> {
    if let Some(path) = spec.strip_prefix("qwen35:") {
        return Ok(Box::new(Qwen35Engine::open(path)?));
    }
    if spec == "qwen35-gguf" {
        let path = std::env::var(qwen35::int8_gguf_resident::GGUF_ENV).map_err(|_| format!("target qwen35-gguf reads {}; set it to a Qwen3.8 GGUF", qwen35::int8_gguf_resident::GGUF_ENV))?;
        return Ok(Box::new(Qwen35Engine::open(&path)?));
    }
    if let Some(weights) = spec.strip_prefix("qwen:") {
        return Ok(Box::new(QwenPagedEngine::open(weights)?));
    }
    Err(format!("longctx does not support target {spec:?}: use qwen35:<gguf>, qwen35-gguf or qwen:<weights>[:i8w][:kvf32]"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_shrinks_the_run_and_flags_are_validated() {
        let args = Args { context: Some(131_072), ladder: Some("1,2,4,8".into()), prefill: Some("4096,8192".into()), steps: Some(16) };
        let o = options(&args, "gpu", true).unwrap();
        assert!(o.smoke && o.context <= 256 && o.steps == 2 && o.ladder == vec![1, 2] && o.prefill == vec![128]);
        let full = options(&args, "gpu", false).unwrap();
        assert_eq!((full.context, full.steps, full.ladder, full.prefill), (131_072, 16, vec![1, 2, 4, 8], vec![4096, 8192]));
        assert!(options(&Args { ladder: Some("1,zero".into()), ..Default::default() }, "gpu", false).is_err());
        let d = options(&Args::default(), "gpu", false).unwrap();
        assert!(d.prefill.is_empty() && d.ladder.first() == Some(&1));
    }

    #[test]
    fn targets_outside_the_supported_families_are_refused_by_name() {
        let e = build_engine("yolo:/nowhere").err().unwrap();
        assert!(e.contains("yolo:/nowhere") && e.contains("qwen35:<gguf>"), "{e}");
    }

    #[test]
    fn qwen35_gguf_without_its_env_names_the_variable() {
        let _g = brain_testutil::env_lock();
        let saved = std::env::var(qwen35::int8_gguf_resident::GGUF_ENV).ok();
        std::env::remove_var(qwen35::int8_gguf_resident::GGUF_ENV);
        let e = build_engine("qwen35-gguf").err().unwrap();
        if let Some(v) = saved {
            std::env::set_var(qwen35::int8_gguf_resident::GGUF_ENV, v);
        }
        assert!(e.contains("BRAIN_QWEN35_GGUF"), "{e}");
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
