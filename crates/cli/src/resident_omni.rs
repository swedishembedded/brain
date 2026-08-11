// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapter putting Qwen3-Omni's Thinker text generation behind
//! the residency [`Executor`], mirroring [`crate::resident_tts::TtsResident`].
//!
//! **Validation-tier, not production** (`omni::caps`'s own module doc has the
//! full reasoning): `generate` streams every decoder layer's weights fresh
//! from the real HF checkpoint on every call, with no KV-cache and no int8/
//! GPU-sharded residency. `estimate()` reports the checkpoint's on-disk size
//! as a RAM cost (the mmap footprint the streaming reads touch), not a real
//! VRAM budget — this resident does not yet participate in the GPU-residency
//! scheduling `.agents/rules/lessons.md §14` describes for a production Omni; that is
//! `crates/qwen3/src/shard.rs`'s int8-sharded-across-2-GPUs pattern, not yet
//! built for Thinker (`.agents/roadmap/omni.md`'s M9 entry).
//!
//! Config is env-only, mirroring `TtsResident`:
//!   * `BRAIN_OMNI_HF_DIR` — the real HF checkpoint directory (config.json +
//!     `vocab.json`/`merges.txt` or `tokenizer.json` + the sharded
//!     `model.safetensors.index.json` + shards). The primary gate; unset ⇒
//!     not served.

use capability::{ActionResult, Invocation, Manifest, Progress};
use omni::caps::OmniProvider;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use std::sync::Arc;

/// Qwen3-Omni behind the scheduler. Loads directly from a real HF checkpoint
/// directory (`BRAIN_OMNI_HF_DIR`) — no brain-native import step involved yet
/// (`.agents/roadmap/omni.md`'s M9 entry on the two open loader-side
/// naming gaps for Talker/Code2Wav). Dispatches both declared actions
/// (`generate`, `speak`) through `omni::caps::run_action`.
pub struct OmniResident {
    hf_dir: String,
}

impl OmniResident {
    /// Configure from the environment. Returns `None` (not served) when
    /// `BRAIN_OMNI_HF_DIR` is unset/empty, like every other `from_env` resident.
    pub fn from_env() -> Option<OmniResident> {
        let hf_dir = std::env::var("BRAIN_OMNI_HF_DIR").ok().filter(|p| !p.is_empty())?;
        Some(OmniResident { hf_dir })
    }

    /// Rough on-disk size of the checkpoint directory (sum of `*.safetensors`
    /// shards) — the mmap footprint `generate`'s streaming reads touch, used
    /// as `estimate()`'s RAM figure. Not a VRAM budget (see this module's doc).
    fn checkpoint_bytes(&self) -> u64 {
        std::fs::read_dir(&self.hf_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .filter_map(|e| e.metadata().ok().map(|m| m.len()))
            .sum()
    }
}

impl ResidentModel for OmniResident {
    fn manifest(&self) -> Manifest {
        omni::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(omni::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        let bytes = self.checkpoint_bytes();
        let ram = if bytes > 0 { bytes } else { 70u64 << 30 }; // 70 GiB fallback: the real checkpoint's own bf16 size
        MemCost::new(0, ram)
    }

    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        let provider = OmniProvider::load(&self.hf_dir)?;
        Ok(Box::new(OmniInstance { inner: provider.inner() }))
    }
}

struct OmniInstance {
    inner: Arc<omni::caps::OmniInner>,
}

impl Instance for OmniInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        // Single dispatch path shared with `omni::caps::OmniProvider` -- this
        // resident (not that Provider) is what `brain serve` actually routes
        // D-Bus/HTTP requests through, and it previously IGNORED the action
        // name: an advertised `speak` silently ran `generate` (text out, no
        // audio, no error). `run_action` matches on the action and errors on
        // anything the manifest doesn't declare.
        omni::caps::run_action(&self.inner, action, inv, progress)
    }
}

/// The int8 dual-GPU Thinker (`omni::int8_thinker_resident::Int8ThinkerResident`)
/// — layer-sharded across every budgeted GPU, streamed straight from a
/// pre-quantized brain-native checkpoint. Reachable ONLY via
/// `Executor::register_multi` (never `register` — see that type's own doc on
/// why a multi-device-only model must stay out of the plain single-device
/// registry). A SEPARATE resident from [`OmniResident`] above: that one is
/// the validation-tier, single-device, no-KV-cache HF-checkpoint path; this
/// is the sharded int8 path `.agents/roadmap/omni.md`'s M18/M20 entries
/// describe, and the two are not interchangeable.
///
/// Config is env-only:
///   * `BRAIN_OMNI_INT8_CHECKPOINT` — a brain-native int8-quantized checkpoint
///     (the output of `omni::import`'s int8-native path). Unset ⇒ not served.
///
/// `gpus` is `build_executor`'s own budgeted GPU list — the shard count
/// tracks whatever the box actually has (never hardcoded), which is exactly
/// the gap the int8 dual-GPU residency work closed in
/// `Int8ThinkerResident::estimate_multi` itself: hardcoding `[Gpu(0), Gpu(1)]`
/// would silently hang on a 1-GPU box and waste a card on a 3+-GPU one.
pub fn int8_thinker_multi_from_env(gpus: &[(u32, u64)]) -> Option<omni::int8_thinker_resident::Int8ThinkerResident> {
    let checkpoint_path = std::env::var("BRAIN_OMNI_INT8_CHECKPOINT").ok().filter(|p| !p.is_empty())?;
    if gpus.is_empty() {
        eprintln!("brain: omni-int8-thinker-multi not served (no GPU budgeted -- it is GPU-sharded only, no CPU path)");
        return None;
    }
    let devices: Vec<Device> = gpus.iter().map(|&(i, _)| Device::Gpu(i)).collect();
    // Real numbers confirmed against the real checkpoint's config (not
    // assumed) -- validated end to end on two physically separate GPUs
    // during the int8 dual-GPU residency work.
    let cfg = omni::config::MoeTextConfig::thinker_defaults();
    Some(omni::int8_thinker_resident::Int8ThinkerResident::new(checkpoint_path, cfg, devices))
}
