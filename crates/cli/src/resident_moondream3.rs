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
//! # Placement
//!
//! This model is GPU-placeable. `estimate` reports its footprint as `vram`, so
//! `residency::place::pick_device` prefers a card and falls back to the CPU
//! pool on a machine with no GPU (that fallback is `place`'s own rule for a
//! weight-holding model, not a special case here). `activate` builds on
//! whichever device it was handed, through a SCOPED registry selection
//! (`Session::load_on` -> `gpu_core::devices::with_gpu`) rather than an env
//! write, because a server-lifetime resident must not change the backend every
//! other model builds on afterwards.
//!
//! **What that claim rests on, precisely.** No real Moondream checkpoint has
//! been run on an accelerator anywhere, and none exists on the machine this was
//! written on - so this is NOT a statement that the released weights produce
//! good captions on a GPU. It is a statement that the device PLUMBING is
//! correct, which is checked: `a_gpu_build_computes_the_same_function_as_the_cpu_build`
//! builds a tiny-config model on a real card and on the CPU backend and
//! requires the logits to agree. A scoped selection that silently fell through
//! to the ambient device, or one tower built on a different backend from its
//! own buffers, both run and both fail that test.
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
//! # Batching: real, on the vision half
//!
//! `run_batch` is overridden with a genuine batched forward, not a serial loop.
//! The axis is the VISION tower: `SiglipEncoder::encode` attends within each
//! crop as its own span, so N concurrent requests' crops go through ONE encode
//! call rather than N - and at the released config that is the dominant
//! per-request cost (1 global + up to 12 local crops of 729 patches).
//!
//! The decoder half stays per-request, and that is architectural rather than
//! unfinished: each request has its own prompt, its own image embeddings and
//! its own KV cache, and the block forward has no batch dimension. Adding one
//! is a separate piece of work from this seam.

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
        // Reported as VRAM: both towers go wherever this instance is placed, so
        // on a GPU box the whole footprint is device memory. `place::pick_device`
        // falls a weight-holding model back to the CPU pool at the same figure
        // on a machine with no GPU, which is its own rule and the behaviour this
        // model wants.
        let bytes = if key.config.ends_with("|fp32") { FP32_BYTES } else { INT8_BYTES };
        MemCost::new(bytes, 0)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let (dir, p) = key.config.rsplit_once('|').ok_or("moondream3: malformed instance key")?;
        let precision = moondream3::caps::parse_precision(p)?;
        let gpu = match device {
            Device::Cpu => None,
            Device::Gpu(i) => Some(i),
            Device::Npu(i) => {
                // This model advertises no NPU footprint, so the placer never
                // offers one; refuse by name rather than silently building wgpu.
                return Err(format!("moondream3: assigned Npu({i}), but this model has no NPU export path"));
            }
        };
        Ok(Box::new(Moondream3Instance { session: Session::load_on(dir, precision, gpu)? }))
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

    /// Real batching, on the axis this architecture actually has.
    ///
    /// The DECODER cannot batch: each request has its own prompt, its own image
    /// embeddings and its own KV cache, and the block forward has no batch
    /// dimension. The VISION tower can, and it is the dominant per-request cost:
    /// 1 global plus up to 12 local crops of 729 patches each. `SiglipEncoder`
    /// already attends within each crop as its own span, so N requests' crops
    /// concatenate into ONE encode call instead of N.
    ///
    /// Crops-per-request varies with each image's aspect ratio, so the batched
    /// path slices results back by each request's own tile count rather than a
    /// uniform stride - pinned by `batched_vision_matches_one_image_at_a_time`,
    /// because getting that wrong hands one request another's crops with no
    /// shape error to notice.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        if action != "caption" {
            return invs.iter().map(|_| Err(format!("moondream3: unknown action '{action}'"))).collect();
        }
        self.session.caption_batch(invs, progress)
    }
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
        assert!(c8.vram > 0, "this model is GPU-placeable; a zero vram would hide it from the GPU class");
        assert_eq!(c8.npu, 0, "no NPU export path exists");
        assert!(c32.vram > c8.vram * 3, "fp32 should be ~4x int8, got {} vs {}", c32.vram, c8.vram);
    }

    /// An NPU assignment is refused by name. The placer never offers one (npu
    /// == 0), but a silent wgpu build under an NPU label is the kind of
    /// wrong-backend failure a sibling model in this directory already paid for.
    #[test]
    fn an_npu_assignment_is_refused() {
        let r = Moondream3Resident { dir: "/tmp".into() };
        let e = r.activate(&r.instance_key("caption", &Invocation::new()), Device::Npu(0)).err().unwrap_or_default();
        assert!(e.contains("no NPU export path"), "{e}");
    }
}
