// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5 text-to-video behind the residency scheduler
//! (`resident::build_executor`) - `resident_wan.rs`'s pattern, with the one
//! structural difference LTX's own weight story forces.
//!
//! # What is resident, and where it actually lives
//!
//! `wan`'s resident model holds its built DiT in a field, because a Wan DiT
//! is one 5.7 GB object that costs ~20 s to build. LTX's real DiT is 23.6 GB
//! of Q8_0 GGUF that is never materialized as one object at all: it is
//! streamed block by block, and what is expensive is the per-block GGUF read
//! plus int8 quantize (~86% of a real denoise step; ~250 s of cold disk on
//! this box's ~58-70 MB/s rotational storage for one 48-block pass).
//!
//! So the thing worth holding resident is the already-quantized BLOCK BYTES,
//! and they live in `ltxv::weightcache`'s process-wide, checkpoint-keyed
//! store rather than in a field here - because the pipeline reaches them from
//! deep inside `generate()`, and because their correct identity is the
//! checkpoint, not this instance. What this file holds is a HANDLE onto that
//! store, which is what makes `estimate`/`demote`/`promote` able to report
//! and release the real footprint instead of a derived guess.
//!
//! The VAE, per `wan::pipeline`'s own precedent (VAE deliberately never held
//! resident alongside the DiT), is still read fresh per call, and a
//! `--dit-config tiny` run still builds fresh random weights in microseconds
//! and caches nothing - there is genuinely nothing to hold for that path.
//!
//! # Why `demote(Warm)` releases HOST bytes here
//!
//! `Instance::demote`'s contract is "`Warm`: release device buffers, keep
//! host bytes". For LTX there are no device buffers to release BETWEEN calls:
//! `ltxv::dit::forward_q_streamed` opens a fresh `Gpu` per forward and drops
//! every device buffer before it returns - a design its own doc records as
//! deliberate and measured (reusing one `Gpu` handle across calls was tried
//! and ran out of VRAM). An LTX instance's entire reclaimable resident
//! footprint is therefore host RAM, and a `demote` that released "device
//! buffers" would release nothing at all while the manager charged a Warm
//! cost and believed it had made progress. So `demote` releases the block
//! cache, and `estimate_at` reports the honest post-demote number. That is
//! safe to do at any moment for a reason no other model can claim: the cache
//! holds a pure function of immutable checkpoint bytes, so dropping an entry
//! can only cost time - the next access re-reads and re-quantizes to the same
//! bytes. `promote` is correspondingly a no-op that returns to Hot and lets
//! the next forward re-fill lazily; there is nothing to rebuild eagerly, and
//! rebuilding eagerly would block the manager's worker thread for minutes.

use capability::{ActionResult, Invocation, Manifest, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel, Tier};

/// LTX-2.5 resident model, gated on `BRAIN_LTXV_VAE` (the one real weight
/// role this milestone has - see `ltxv::pipeline::PATH_VARS`).
pub struct LtxvResident {
    /// See `WanResident::id`'s doc for why this is not always the compiled-in
    /// constant: a fetched checkpoint registers under its own reference.
    id: String,
    paths: ltxv::pipeline::Paths,
}

impl LtxvResident {
    /// `None` (not registered) unless `BRAIN_LTXV_VAE` is set.
    pub fn from_env() -> Option<LtxvResident> {
        ltxv::pipeline::Paths::from_env().ok().map(|paths| LtxvResident { id: ltxv::caps::MODEL.to_string(), paths })
    }
}

/// `(dit_config, frames, width, height)` from an instance key
/// (`"{dit_config}:{frames}:{w}x{h}"`) - the fields that would fix a real
/// DiT's graph sizes - and, since the real 22B import landed, the fields that
/// decide how big this instance's resident block-weight cache is.
fn parse_key(config: &str) -> Option<(String, usize, usize, usize)> {
    let mut it = config.split(':');
    let dit_config = it.next()?.to_string();
    let frames: usize = it.next()?.parse().ok()?;
    let (w, h) = it.next()?.split_once('x')?;
    Some((dit_config, frames, w.parse().ok()?, h.parse().ok()?))
}

/// Total bytes of a config's weight manifest, at 4 bytes/element (this
/// repo's safetensors reader is always F32-materialized on read) - the same
/// closed-form-over-the-manifest technique `resident_wan.rs::estimate`
/// documents, just driven by `tensor_manifest`/`dit_tensor_manifest` instead
/// of a hand-derived formula.
fn manifest_bytes(manifest: &[(String, Vec<usize>)]) -> u64 {
    manifest.iter().map(|(_, shape)| shape.iter().product::<usize>() as u64).sum::<u64>() * 4
}

/// Host bytes this instance's resident block-weight cache holds once every
/// layer has been visited, for a real (non-`tiny`) dit-config.
///
/// Deliberately NOT `file_size * 1.3`: `ltxv::block::cached_block_bytes` is
/// the closed form of what `QBlockWeights::quantize_host` really builds, and
/// it is pinned against a really-quantized block by
/// `block_weight_cache.rs::cached_block_bytes_matches_a_real_measured_block`.
/// At the real 22B/Q8_0 config it reproduces the 270.1 MB/block that a real
/// forward measured, i.e. ~13.0 GB for all 48 layers.
///
/// Clamped by the cache's own budget where one is in force: under
/// `--limit-ram-total` the cache evicts rather than growing past it, so the
/// resident footprint the manager should plan against is the smaller of the
/// two, not the model's full size.
fn block_cache_bytes(dit_config: &str, paths: &ltxv::pipeline::Paths) -> u64 {
    let Some(cfg) = streamed_config(dit_config, paths) else { return 0 };
    let full = ltxv::block::cached_block_bytes(&cfg, ltxv::block::QTier::Int8) * cfg.num_layers as u64;
    match ltxv::weightcache::budget_from_limits() {
        Some(b) => full.min(b),
        None => full,
    }
}

/// The DiT config for a key that really streams a checkpoint - `Some` only
/// when a real GGUF is configured AND the named config is not the
/// random-weight `tiny` one, which is built in host fp32 per call and caches
/// nothing.
fn streamed_config(dit_config: &str, paths: &ltxv::pipeline::Paths) -> Option<ltxv::LtxDitConfig> {
    if paths.dit.is_none() || dit_config == "tiny" {
        return None;
    }
    ltxv::pipeline::dit_config_from_name(dit_config).ok()
}

/// Host bytes the DiT itself costs while resident.
///
/// The two paths differ in kind, not in size, so one formula cannot serve
/// both. `tiny` really is materialized as one host fp32 weight map, so its
/// whole manifest is the honest number. A real checkpoint is never
/// materialized at all - `ltxv::dit::forward_q_streamed` streams it block by
/// block - so charging its fp32 manifest (~62 GB at the 22B config) described
/// memory that has never been allocated in this repo's history. What a real
/// run actually holds is: the small head tensors, the int8 block-weight cache
/// (the thing this milestone made resident), and ONE block's transient fp32
/// expansion while that block is being read and quantized.
fn dit_host_bytes(dit_config: &str, paths: &ltxv::pipeline::Paths) -> u64 {
    let Some(cfg) = streamed_config(dit_config, paths) else {
        return ltxv::pipeline::dit_config_from_name(dit_config).map(|c| manifest_bytes(&ltxv::dit::dit_tensor_manifest(&c))).unwrap_or(0);
    };
    let manifest = ltxv::dit::dit_tensor_manifest(&cfg);
    let is_block = |name: &str| name.starts_with("transformer_blocks.");
    let head: u64 = manifest.iter().filter(|(n, _)| !is_block(n)).map(|(_, sh)| sh.iter().product::<usize>() as u64).sum::<u64>() * 4;
    let one_block_fp32: u64 = manifest.iter().filter(|(n, _)| n.starts_with("transformer_blocks.0.")).map(|(_, sh)| sh.iter().product::<usize>() as u64).sum::<u64>() * 4;
    head + one_block_fp32 + block_cache_bytes(dit_config, paths)
}

impl ResidentModel for LtxvResident {
    fn manifest(&self) -> Manifest {
        Manifest { model: self.id.clone(), ..ltxv::caps::manifest() }
    }

    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        let d = ltxv::pipeline::GenOpts::default();
        let dit_config = inv.get_str("dit_config").unwrap_or_else(|| "tiny".into());
        let frames = inv.get_i64("frames").unwrap_or(d.frames as i64);
        let w = inv.get_i64("width").unwrap_or(d.width as i64);
        let h = inv.get_i64("height").unwrap_or(d.height as i64);
        InstanceKey::new(&self.id, format!("{dit_config}:{frames}:{w}x{h}"))
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        let Some((dit_config, frames, w, h)) = parse_key(&key.config) else {
            // An unparseable key must not read as "free" - see
            // `resident_wan.rs::estimate`'s identical reasoning.
            return MemCost::new(1u64 << 30, 512u64 << 20);
        };
        let vae_weights = manifest_bytes(&ltxv::LtxVaeConfig::conv25().tensor_manifest());
        // VRAM keeps the pre-existing (deliberately conservative) manifest
        // figure: a real streamed forward's peak VRAM is dominated by
        // activation buffers whose size follows the latent token count, and
        // deriving that honestly is its own piece of work - tracked on the
        // roadmap ledger, not guessed at here. The HOST figure below is the
        // one this milestone made real.
        let dit_weights_fp32 = ltxv::pipeline::dit_config_from_name(&dit_config).map(|c| manifest_bytes(&ltxv::dit::dit_tensor_manifest(&c))).unwrap_or(0);
        // The VAE decoder's own activations dominate at any real clip size
        // (pixel-space buffers, `3 * frames * h * w` floats at several
        // stages) - the same `* 4` term `resident_wan.rs::estimate` charges
        // for its own VAE decode.
        let pixels = frames as u64 * w as u64 * h as u64 * 3 * 4;
        MemCost::new(vae_weights + dit_weights_fp32 + pixels * 4, vae_weights + dit_host_bytes(&dit_config, &self.paths) + (256u64 << 20))
    }

    /// Below `Hot` the block-weight cache is gone, so the host figure drops
    /// by exactly what `estimate` added for it. `Cold` additionally states
    /// the checkpoint's on-disk footprint as `mapped`: the GGUF stays mmap'd
    /// either way, and those pages are reclaimable by the kernel in a way a
    /// live allocation is not, which is precisely the distinction
    /// `MemCost::mapped` exists to carry.
    fn estimate_at(&self, key: &InstanceKey, tier: Tier) -> MemCost {
        let hot = self.estimate(key);
        if tier == Tier::Hot {
            return hot;
        }
        let Some((dit_config, _, _, _)) = parse_key(&key.config) else { return hot };
        let warm = MemCost::new(0, hot.ram.saturating_sub(block_cache_bytes(&dit_config, &self.paths)));
        match tier {
            Tier::Cold => warm.with_mapped(self.paths.dit.as_deref().map(|p| ltxv::text_cache::encoder_identity(p).0).unwrap_or(0)),
            _ => warm,
        }
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let (dit_config, _, _, _) = parse_key(&key.config).ok_or_else(|| format!("ltxv: bad instance key {:?}", key.config))?;
        // Validate at activation, not at the first request - a key naming a
        // `dit_config` this build cannot construct should fail placement.
        ltxv::pipeline::dit_config_from_name(&dit_config)?;
        // The handle onto the checkpoint's shared weight cache - the SAME
        // store `ltxv::pipeline::RealDit` resolves by path from inside
        // `generate()`, which is what lets `demote` here release memory the
        // pipeline is actually using.
        let cache = self.paths.dit.as_deref().map(ltxv::block::GenerationCache::for_checkpoint);
        Ok(Box::new(LtxvInstance { paths: self.paths.clone(), device, cache }))
    }
}

/// A resident LTX-2.5 instance: the VAE path, the assigned device, and a
/// handle onto the checkpoint's shared block-weight cache (`None` when no
/// real DiT checkpoint is configured). The VAE graph is still rebuilt per
/// request, exactly like a one-shot CLI call - see this module's doc for what
/// is and is not held.
struct LtxvInstance {
    paths: ltxv::pipeline::Paths,
    device: Device,
    cache: Option<ltxv::block::GenerationCache>,
}

impl LtxvInstance {
    fn device_name(&self) -> Option<String> {
        match self.device {
            Device::Cpu => Some("cpu".to_string()),
            Device::Gpu(_) => Some("gpu".to_string()),
            Device::Npu(_) => None,
        }
    }
}

impl Instance for LtxvInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match action {
            "t2v" => {
                let mut p = ltxv::caps::gen_params_from(inv)?;
                p.opts.device = self.device_name();
                crate::resident_llm::on_device(self.device, || ltxv::caps::generate_on(&self.paths, inv, &p, progress))?
            }
            "dfr" => {
                // DFR needs the two latent-upscaler paths on top of the VAE
                // `self.paths` already carries - resolved fresh from the
                // environment per call, the same "nothing worth caching yet"
                // reasoning this module's own doc gives for the VAE path.
                let mut p = ltxv::caps::dfr_params_from(inv)?;
                p.opts.base.device = self.device_name();
                let dfr_paths = ltxv::pipeline::DfrPaths::from_env()?;
                crate::resident_llm::on_device(self.device, || ltxv::caps::dfr_on(&dfr_paths, inv, &p, progress))?
            }
            other => Err(format!("ltxv: unknown action '{other}'")),
        }
    }

    /// Sequential, deliberately: a "batch" here is N independent generations
    /// run one after another, since nothing in this pipeline batches a
    /// denoise loop across prompts. What they DO share - and what makes the
    /// second and later ones cheap rather than merely correct - is the
    /// checkpoint's block-weight cache: request 1 populates it, every later
    /// request in the batch starts warm on its first denoise step.
    /// Per-request cancellation still works: each job's own `inv.cancel` is
    /// polled inside its own denoise loop.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        invs.iter().enumerate().map(|(i, inv)| self.run(action, inv, &mut |p| progress(i, p))).collect()
    }

    /// The block-weight cache's own hit/miss/eviction counters and current
    /// footprint, so an operator can see whether residency is doing anything
    /// without re-deriving it from a trace. Empty when no real checkpoint is
    /// configured, which is the honest report for a random-weight run.
    fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        let Some(c) = &self.cache else { return Vec::new() };
        let s = c.stats();
        vec![
            ("ltxv_block_cache_hits".into(), s.hits.into()),
            ("ltxv_block_cache_misses".into(), s.misses.into()),
            ("ltxv_block_cache_evictions".into(), s.evictions.into()),
            ("ltxv_block_cache_blocks".into(), s.blocks.into()),
            ("ltxv_block_cache_bytes".into(), s.bytes.into()),
        ]
    }

    /// Release the resident block-weight cache. See this module's doc for why
    /// a `Warm` demote releases HOST bytes for this model and why doing so is
    /// always safe: the entries are a pure function of immutable checkpoint
    /// bytes, so an eviction costs time and nothing else.
    fn demote(&mut self, tier: Tier) -> Result<(), String> {
        debug_assert_ne!(tier, Tier::Hot, "demote is never a promotion");
        if tier == Tier::Hot {
            return Err("ltxv: demote(Hot) is not a demotion".into());
        }
        match &self.cache {
            Some(c) => {
                c.clear();
                Ok(())
            }
            // Nothing was ever held, so nothing can be released - and saying
            // `Ok` would let the manager charge a Warm cost against progress
            // it did not make.
            None => Err("ltxv: nothing resident to demote (no real DiT checkpoint configured)".into()),
        }
    }

    /// Return to `Hot`. Deliberately lazy: the next forward re-reads and
    /// re-quantizes exactly the blocks it needs, in the loop that already
    /// knows how, and re-filling ~13 GB eagerly here would block the
    /// manager's worker thread for minutes to do work the request itself does
    /// incrementally.
    fn promote(&mut self, _device: Device) -> Result<(), String> {
        match &self.cache {
            Some(_) => Ok(()),
            None => Err("ltxv: nothing to promote (no real DiT checkpoint configured)".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn resident() -> LtxvResident {
        LtxvResident { id: ltxv::caps::MODEL.to_string(), paths: ltxv::pipeline::Paths { vae: "/vae".into(), dit: None, text_encoder: None } }
    }

    /// The key must fix exactly the fields that would size a real DiT's
    /// graph (dit_config + latent extent) and nothing per-call
    /// (seed/steps/guidance/prompt) - even though nothing is cached against
    /// it today, the shape must already be right for the day something is.
    #[test]
    fn the_instance_key_is_exactly_the_dit_config_and_the_latent_extent() {
        let r = resident();
        let base = Invocation::new().set("prompt", json!("a"));
        let a = r.instance_key("t2v", &base);
        assert_eq!(a.config, "tiny:9:64x64", "the defaults must key on the manifest's own defaults");
        let b = r.instance_key("t2v", &Invocation::new().set("prompt", json!("something else")).set("seed", json!(7)).set("steps", json!(2)).set("guidance", json!(3.0)));
        assert_eq!(a.config, b.config, "per-call params must not split the instance");
        for (k, v) in [("frames", json!(17)), ("width", json!(96)), ("height", json!(96))] {
            assert_ne!(a.config, r.instance_key("t2v", &base.clone().set(k, v)).config, "{k} must split the instance");
        }
    }

    #[test]
    fn key_parsing_round_trips_and_a_bad_key_still_costs_something() {
        assert_eq!(parse_key("tiny:9:64x64"), Some(("tiny".to_string(), 9, 64, 64)));
        assert_eq!(parse_key("garbage"), None);
        let cost = resident().estimate(&InstanceKey::new(ltxv::caps::MODEL, "garbage".to_string()));
        assert!(cost.vram > 0 && cost.ram > 0, "{cost:?}");
    }

    /// The estimate must land somewhere sane (single-digit GB, not zero, not
    /// absurd) for a real clip size - the exact number is not load-bearing,
    /// only that it is a real, nonzero, config-derived figure. The VAE's own
    /// weights dominate even the smallest clip (~726M real parameters,
    /// ~2.9 GB at this reader's f32 materialization - `crate::vae3d::
    /// LtxVaeConfig::manifest_counts_the_shipped_checkpoint`'s 170-tensor
    /// count times 4 bytes/element), NOT the tiny/random DiT (~KB) or the
    /// pixel buffers - so "small" here means "a bigger clip costs more on
    /// top of that fixed VAE floor", not "small in absolute terms".
    #[test]
    fn the_estimate_is_nonzero_and_grows_with_clip_size() {
        let r = resident();
        let small = r.estimate(&InstanceKey::new(ltxv::caps::MODEL, "tiny:9:64x64".to_string()));
        let big = r.estimate(&InstanceKey::new(ltxv::caps::MODEL, "tiny:17:256x256".to_string()));
        assert!(small.vram > 0 && small.ram > 0);
        assert!(big.vram > small.vram, "a bigger clip must cost more");
        let gb = 1u64 << 30;
        assert!(small.vram < 4 * gb, "the VAE-dominated floor should be well under 4 GB, got {}", small.vram);
    }

    /// A real checkpoint's resident cost is the BLOCK-WEIGHT CACHE, and the
    /// estimate must say so - a real, config-derived, ~13 GB figure for the
    /// 48-layer 22B model, absent for the random-weight `tiny` path. Without
    /// this the manager would place a 13 GB host resident against a budget
    /// that never heard of it.
    #[test]
    fn a_real_checkpoint_estimate_carries_the_block_cache_and_tiny_does_not() {
        let dir = std::env::temp_dir().join(format!("ltxv-res-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ck = dir.join("dit.gguf");
        std::fs::write(&ck, b"stand-in checkpoint").unwrap();
        let r = LtxvResident { id: ltxv::caps::MODEL.to_string(), paths: ltxv::pipeline::Paths { vae: "/vae".into(), dit: Some(ck.to_string_lossy().into_owned()), text_encoder: None } };

        let real = r.estimate(&InstanceKey::new(ltxv::caps::MODEL, "ltx25_22b:9:64x64".to_string()));
        let tiny = r.estimate(&InstanceKey::new(ltxv::caps::MODEL, "tiny:9:64x64".to_string()));
        let gb = 1u64 << 30;
        assert!(real.ram > tiny.ram + 10 * gb, "the real config's host cost must carry its block-weight cache ({} vs {})", real.ram, tiny.ram);
        assert!(real.ram < 40 * gb, "and must be the int8 cache footprint, not a whole-model fp32 figure: {}", real.ram);

        // Below Hot the cache is gone, so the host figure drops by exactly
        // that much; Cold additionally reports the mapping.
        let key = InstanceKey::new(ltxv::caps::MODEL, "ltx25_22b:9:64x64".to_string());
        let warm = r.estimate_at(&key, Tier::Warm);
        let cold = r.estimate_at(&key, Tier::Cold);
        assert_eq!(warm.ram, real.ram - block_cache_bytes("ltx25_22b", &r.paths), "a Warm estimate must be the Hot one minus exactly the cache");
        assert_eq!(warm.vram, 0, "nothing device-side survives a forward call here");
        assert_eq!(cold.mapped, std::fs::metadata(&ck).unwrap().len(), "Cold must report the checkpoint's mapped footprint");
        assert_eq!(r.estimate_at(&key, Tier::Hot), real, "estimate_at(Hot) must be estimate");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The demote/promote contract this model is the first production
    /// implementor of: demoting releases the checkpoint's shared cache (the
    /// SAME store the pipeline reads), promoting returns to Hot and lets the
    /// next forward refill lazily - and a model with no real checkpoint
    /// refuses both rather than claiming progress it cannot make.
    #[test]
    fn demote_releases_the_shared_block_cache_and_promote_returns_to_hot() {
        let dir = std::env::temp_dir().join(format!("ltxv-res-dp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ck = dir.join("dit2.gguf");
        std::fs::write(&ck, b"stand-in checkpoint bytes").unwrap();
        let path = ck.to_string_lossy().into_owned();
        let r = LtxvResident { id: ltxv::caps::MODEL.to_string(), paths: ltxv::pipeline::Paths { vae: "/vae".into(), dit: Some(path.clone()), text_encoder: None } };
        let mut inst = r.activate(&InstanceKey::new(ltxv::caps::MODEL, "ltx25_22b:9:64x64".to_string()), Device::Gpu(0)).expect("activate");

        // Populate the store the way a generation does - through the SAME
        // process-wide, checkpoint-keyed handle the pipeline resolves.
        let pipeline_side = ltxv::block::GenerationCache::for_checkpoint(&path);
        let cfg = ltxv::LtxDitConfig { num_layers: 1, ..ltxv::LtxDitConfig::tiny() };
        let w = ltxv::dit::random_tiny_weights(&cfg, 0xD0D0);
        let quantized = ltxv::block::CachedQBlockWeights::quantize(&w, "transformer_blocks.0", &cfg, ltxv::block::QTier::Int8);
        pipeline_side.store_block(0, ltxv::block::QTier::Int8, quantized);
        assert_eq!(pipeline_side.stats().blocks, 1, "test setup: the shared store must be populated");

        assert!(inst.demote(Tier::Warm).is_ok());
        assert_eq!(pipeline_side.stats().blocks, 0, "demote must release the store the PIPELINE reads, not a private copy");
        assert!(inst.promote(Device::Gpu(0)).is_ok(), "promote must succeed: refilling is lazy by design");

        // No real checkpoint -> nothing resident -> both must refuse.
        let bare = LtxvResident { id: ltxv::caps::MODEL.to_string(), paths: ltxv::pipeline::Paths { vae: "/vae".into(), dit: None, text_encoder: None } };
        let mut bare_inst = bare.activate(&InstanceKey::new(ltxv::caps::MODEL, "tiny:9:64x64".to_string()), Device::Cpu).expect("activate");
        assert!(bare_inst.demote(Tier::Warm).is_err(), "a model holding nothing must not report a successful demotion");
        assert!(bare_inst.promote(Device::Cpu).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **The end-to-end proof, on real weights.** Two back-to-back
    /// generations with DIFFERENT prompts through ONE `LtxvResident`
    /// instance - the resident path, not the one-shot CLI - must share the
    /// checkpoint's block-weight cache: generation 2 records zero block
    /// misses and its denoise is dominated by GPU work rather than by
    /// re-reading 22 GB off a rotational disk.
    ///
    /// `#[ignore]`d because it is a real multi-minute generation against real
    /// checkpoints, not a unit test. Run it explicitly:
    ///
    /// ```text
    /// BRAIN_LTXV_VAE=... BRAIN_LTXV_DIT=... BRAIN_LTXV_TEXT_ENCODER=... \
    ///   cargo test --release -p brain-cli --bins -- --ignored --nocapture \
    ///   two_real_generations_share
    /// ```
    ///
    /// It exists because nothing else in this workspace drove `LtxvResident`
    /// end to end with real weights, so "the resident path is warm on its
    /// second request" was previously only checkable by hand.
    #[test]
    #[ignore = "real multi-minute generation against real LTX-2.5 checkpoints"]
    fn two_real_generations_share_one_warm_checkpoint_cache() {
        let Some(r) = LtxvResident::from_env() else {
            eprintln!("skipped: set BRAIN_LTXV_VAE (and BRAIN_LTXV_DIT) to real LTX-2.5 checkpoints");
            return;
        };
        assert!(r.paths.dit.is_some(), "this harness needs BRAIN_LTXV_DIT: it is about the real checkpoint's cache");
        let key = InstanceKey::new(ltxv::caps::MODEL, "ltx25_22b:9:64x64".to_string());
        let mut inst = r.activate(&key, Device::Gpu(0)).expect("activate");

        let request = |prompt: &str| {
            Invocation::new()
                .set("prompt", json!(prompt))
                .set("dit_config", json!("ltx25_22b"))
                .set("frames", json!(9))
                .set("width", json!(64))
                .set("height", json!(64))
                .set("guidance", json!(1.0))
        };
        let metric = |inst: &dyn Instance, name: &str| inst.metrics().into_iter().find(|(k, _)| k == name).and_then(|(_, v)| v.as_u64()).unwrap_or(0);

        let run = |inst: &mut dyn Instance, prompt: &str| {
            let t = std::time::Instant::now();
            inst.run("t2v", &request(prompt), &mut |_| {}).unwrap_or_else(|e| panic!("generation failed: {e}"));
            t.elapsed().as_secs_f64()
        };

        let secs_a = run(inst.as_mut(), "a red kite over a grey harbour");
        let (hits_a, misses_a, blocks_a, bytes_a) = (metric(inst.as_ref(), "ltxv_block_cache_hits"), metric(inst.as_ref(), "ltxv_block_cache_misses"), metric(inst.as_ref(), "ltxv_block_cache_blocks"), metric(inst.as_ref(), "ltxv_block_cache_bytes"));

        let secs_b = run(inst.as_mut(), "a slow pan across a snowbound pine forest");
        let (hits_b, misses_b) = (metric(inst.as_ref(), "ltxv_block_cache_hits"), metric(inst.as_ref(), "ltxv_block_cache_misses"));

        println!("generation 1: {secs_a:.1} s   hits={hits_a} misses={misses_a} blocks={blocks_a} bytes={:.2} GB", bytes_a as f64 / 1e9);
        println!("generation 2: {secs_b:.1} s   hits={} misses={} (delta)", hits_b - hits_a, misses_b - misses_a);

        assert_eq!(misses_b, misses_a, "generation 2 must not miss on a single block - it must start warm from its very first layer");
        assert!(hits_b > hits_a, "generation 2 must actually have read the cache");
        assert!(secs_b < secs_a, "generation 2 must be faster than the cold one: {secs_b:.1} s vs {secs_a:.1} s");
    }

    #[test]
    fn activate_rejects_an_unknown_dit_config() {
        let e = resident().activate(&InstanceKey::new(ltxv::caps::MODEL, "22b:9:64x64".to_string()), Device::Cpu).err().expect("an unknown dit_config must not activate");
        assert!(e.contains("unknown ltxv dit-config"), "{e}");
    }

    #[test]
    fn the_adapter_advertises_the_shared_manifest() {
        let m = resident().manifest();
        assert_eq!(m.model, ltxv::caps::MODEL);
        assert_eq!(m.actions.len(), ltxv::caps::manifest().actions.len());
        let fetched = LtxvResident { id: "Lightricks/LTX-2.5".to_string(), paths: ltxv::pipeline::Paths { vae: "v".into(), dit: None, text_encoder: None } };
        assert_eq!(fetched.manifest().model, "Lightricks/LTX-2.5");
        assert_eq!(fetched.instance_key("t2v", &Invocation::new()).model, "Lightricks/LTX-2.5");
    }
}
