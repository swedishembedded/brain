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
//! scheduling `docs/lessons.md §14` describes for a production Omni; that is
//! `crates/qwen/src/shard.rs`'s int8-sharded-across-2-GPUs pattern, not yet
//! built for Thinker (`docs/models/omni/status.md`'s M9 entry).
//!
//! Config is env-only, mirroring `TtsResident`:
//!   * `BRAIN_OMNI_HF_DIR` — the real HF checkpoint directory (config.json +
//!     `vocab.json`/`merges.txt` or `tokenizer.json` + the sharded
//!     `model.safetensors.index.json` + shards). The primary gate; unset ⇒
//!     not served.

use capability::{ActionResult, Invocation, Manifest, Progress};
use omni::caps::OmniProvider;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;
use std::sync::Arc;

/// Qwen3-Omni Thinker text generation behind the scheduler. Loads directly
/// from a real HF checkpoint directory (`BRAIN_OMNI_HF_DIR`) — no brain-
/// native import step involved yet (`docs/models/omni/status.md`'s M9 entry
/// on the two open loader-side naming gaps for Talker/Code2Wav; Thinker-only
/// generation does not touch either). One action, `generate`.
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
    fn run(&mut self, _action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let prompt = omni::caps::last_user_text(inv);
        if prompt.trim().is_empty() {
            return Err("omni generate: empty prompt (need 'messages' with a user turn, or 'prompt')".to_string());
        }
        let max_new = inv.get_i64("max_new").unwrap_or(32).clamp(1, 4096) as u32;

        // Same optional audio/image/video extraction `omni::caps::GenerateAction::run`
        // does -- this resident (not that Provider) is the path `brain serve`
        // actually dispatches D-Bus/HTTP requests through.
        let audio = inv.get_blob("audio").map(audio::asr_caps::wav_from_blob).transpose()?;
        let image = inv.get_blob("image").map(|_| capability::blob::decode_image(inv, "image")).transpose()?;
        let video = inv.get_blob("video").map(|_| capability::blob::decode_video_hwc(inv, "video")).transpose()?;

        progress(Progress::step(0, max_new, "generating"));
        let (text, new_ids) = if audio.is_some() || image.is_some() || video.is_some() {
            let image_ref = image.as_ref().map(|(hwc, w, h)| (hwc.as_slice(), *w, *h));
            self.inner.generate_multimodal(&prompt, audio.as_deref(), image_ref, video.as_deref(), max_new)
        } else {
            self.inner.generate(&prompt, max_new)
        };
        progress(Progress::step(max_new, max_new, text.clone()));
        Ok(capability::Outcome::new()
            .set("text", json!(text))
            .set("tokens", json!(new_ids))
            .blob("text", capability::Blob::new(capability::Media::Text, text.into_bytes())))
    }
}
