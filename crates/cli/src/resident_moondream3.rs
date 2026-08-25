// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moondream 3 behind the residency scheduler.
//!
//! `activate` builds the whole composite ONCE - the checkpoint load, the
//! quantization of 1280 expert tensors, the ViT and connector upload - and the
//! [`Instance`] owns the resulting [`moondream3::caps::Session`], so dropping it
//! frees every buffer. One action, `caption`; its schema and all of its work
//! come from `moondream3::caps`, so this file holds no second copy of the
//! preprocessing, the prompt assembly or the token accounting.
//!
//! # Why this model is CPU-placed, and why that is a declaration
//!
//! `moondream3::caps::Session::load` builds both towers on
//! `gpu_core::Gpu::new_cpu`. That is not a correctness pin like
//! `crates/deepseek2ocr`'s was - there is no known wgpu defect here - it is that
//! nothing has ever run this model on an accelerator, so claiming a GPU
//! placement would be asserting something untested. `estimate` reports a
//! RAM-only [`MemCost`], which is `residency::place::pick_device`'s own
//! vocabulary for "not GPU-placeable", and the two agree by construction.
//!
//! Moving it is a small, well-scoped change once someone has a machine and a
//! checkpoint to verify on: give `Session::load` a device argument, report
//! `vram` here, and build under `crate::resident_llm::on_device`. Do not do it
//! blind.
//!
//! # The footprint, and why int8 is the default
//!
//! At the released config the decoder is 8.8 B parameters across 20 MoE layers
//! of 64 experts. In fp32 that is ~32.8 GiB of weights plus ~10.3 GiB of
//! activation scratch - no machine runs it. `Precision::Int8` quantizes the
//! experts and puts every block on one shared activation set, together ~8.8 GiB,
//! which is what makes serving it possible at all. This resident therefore
//! serves int8 and keys fp32 as a SEPARATE instance, so a request that asks for
//! fp32 on a machine without room fails placement cleanly instead of evicting
//! the working instance to build one that cannot fit.
//!
//! # Batching: the serial default, and why
//!
//! Every request carries its own image, so the ViT pass (overlap multi-crop,
//! `h·w + 1` encoder passes) is per-request with nothing to share. The decoder
//! has no batch axis wired and no KV cache, so two concurrent requests share no
//! work at all. A real batched forward here is a performance phase of its own,
//! not a wrapper this file could write.

use capability::{ActionResult, Invocation, Manifest, Progress};
use moondream3::caps::{Session, DIR_VAR, MODEL};
use moondream3::model::Precision;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// Host bytes an int8 build holds while hot.
///
/// DERIVED from the released config, not measured - no checkpoint exists on the
/// machine this was written on, and a fabricated "measured" figure is worse than
/// an honest derivation. 8.8 B decoder parameters at one byte each is 8.2 GiB;
/// one shared `BlockScratch` at the built context is ~0.6 GiB; the ViT,
/// connector, embeddings and the `[seq, vocab]` logit slab add the rest. Replace
/// this with a real `VmHWM` the first time it runs on real weights.
const INT8_BYTES: u64 = 11u64 << 30;

/// Host bytes an fp32 build holds while hot: ~32.8 GiB of weights plus ~10.3 GiB
/// of per-block activation scratch, same derivation.
const FP32_BYTES: u64 = 44u64 << 30;

/// Moondream 3 behind the scheduler. `BRAIN_MOONDREAM3_WEIGHTS` names the
/// checkpoint DIRECTORY (`config.json`, the safetensors shards, `tokenizer.json`).
pub struct Moondream3Resident {
    dir: String,
}

impl Moondream3Resident {
    /// `None` when the variable is unset or the directory is absent -
    /// registering a model whose every call would fail is worse than not
    /// serving it.
    pub fn from_env() -> Option<Moondream3Resident> {
        let dir = std::env::var(DIR_VAR).ok().filter(|p| !p.is_empty())?;
        Self::new(dir)
    }

    /// Direct constructor (no env round-trip) - see
    /// `crate::resident_scrfd::ScrfdResident::new`'s rationale.
    pub fn new(dir: impl Into<String>) -> Option<Moondream3Resident> {
        let dir = dir.into();
        if !std::path::Path::new(&dir).is_dir() {
            eprintln!("brain: moondream3 not served ({dir} is not a directory)");
            return None;
        }
        Some(Moondream3Resident { dir })
    }

    /// The precision a request asks for, defaulting to the one that fits.
    fn precision_of(inv: &Invocation) -> Precision {
        inv.get_str("precision")
            .as_deref()
            .and_then(|s| moondream3::caps::parse_precision(s).ok())
            .unwrap_or(Precision::Int8)
    }
}

impl ResidentModel for Moondream3Resident {
    fn manifest(&self) -> Manifest {
        moondream3::caps::manifest()
    }

    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        // Precision is part of the identity: the two builds are different
        // objects with a 4x footprint difference, and sharing one key would
        // make a stray fp32 request silently evict the int8 instance.
        let p = match Self::precision_of(inv) {
            Precision::Int8 => "int8",
            Precision::Fp32 => "fp32",
        };
        InstanceKey::new(MODEL, format!("{}|{p}", self.dir))
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        // RAM, not VRAM -- see this module's header on why the CPU placement is
        // a declaration rather than a pin.
        let bytes = if key.config.ends_with("|fp32") { FP32_BYTES } else { INT8_BYTES };
        MemCost::new(0, bytes)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if device != Device::Cpu {
            // Unreachable while `estimate` reports vram == 0, but a silent
            // wrong-backend build is exactly the failure a sibling model in this
            // directory already paid for once.
            return Err(format!(
                "moondream3: assigned {device:?}, but this model builds on the CPU backend \
                 (Session::load uses Gpu::new_cpu) -- its MemCost declares vram == 0"
            ));
        }
        let (dir, p) = key.config.rsplit_once('|').ok_or("moondream3: malformed instance key")?;
        let precision = moondream3::caps::parse_precision(p)?;
        Ok(Box::new(Moondream3Instance { session: Session::load(dir, precision)? }))
    }
}

/// A resident Moondream 3: the built composite and its tokenizer.
struct Moondream3Instance {
    session: Session,
}

impl Instance for Moondream3Instance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if action != "caption" {
            return Err(format!("moondream3: unknown action '{action}'"));
        }
        self.session.caption(inv, progress)
    }

    // `run_batch` is the serial default: a per-request ViT pass and a decoder
    // with no batch axis share no work between requests -- see this module's
    // header.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unconfigured checkpoint yields no resident at all, rather than one
    /// that fails every call.
    #[test]
    fn a_missing_checkpoint_is_not_registered() {
        assert!(Moondream3Resident::new("/definitely/not/a/moondream/dir").is_none());
    }

    /// Precision is part of the instance identity. Sharing one key would let a
    /// single fp32 request evict a working int8 instance to build one four
    /// times its size.
    #[test]
    fn precision_keys_a_separate_instance() {
        let r = Moondream3Resident { dir: "/tmp".into() };
        let k8 = r.instance_key("caption", &Invocation::new().set("precision", serde_json::json!("int8")));
        let k32 = r.instance_key("caption", &Invocation::new().set("precision", serde_json::json!("fp32")));
        assert_ne!(k8, k32);
        // An absent or unparseable precision falls back to the one that fits.
        assert_eq!(r.instance_key("caption", &Invocation::new()), k8);
        assert_eq!(r.instance_key("caption", &Invocation::new().set("precision", serde_json::json!("bf16"))), k8);
    }

    /// The two builds must be budgeted differently, and fp32 must be the larger
    /// - the whole point of the int8 tier.
    #[test]
    fn the_two_precisions_are_budgeted_apart() {
        let r = Moondream3Resident { dir: "/tmp".into() };
        let c8 = r.estimate(&r.instance_key("caption", &Invocation::new()));
        let c32 = r.estimate(&r.instance_key("caption", &Invocation::new().set("precision", serde_json::json!("fp32"))));
        assert_eq!(c8.vram, 0, "a GPU placement would build a backend nothing has verified");
        assert!(c32.ram > c8.ram * 3, "fp32 should be ~4x int8, got {} vs {}", c32.ram, c8.ram);
    }

    /// A GPU assignment is refused loudly rather than built wrong.
    #[test]
    fn a_gpu_assignment_is_refused() {
        let r = Moondream3Resident { dir: "/tmp".into() };
        let e = r.activate(&r.instance_key("caption", &Invocation::new()), Device::Gpu(0)).err().unwrap_or_default();
        assert!(e.contains("CPU backend"), "{e}");
    }
}
