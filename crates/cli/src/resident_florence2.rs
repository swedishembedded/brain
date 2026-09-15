// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Florence-2 visual grounding behind the residency scheduler.
//!
//! `activate` imports the checkpoint from `BRAIN_FLORENCE2_DIR` ONCE and
//! builds the vision tower on the assigned device; the [`Instance`] owns the
//! resulting [`florence2::caps::FlorenceSession`], so dropping it frees the
//! model. One action, `ground` - the schema and the work both come from
//! `florence2::caps`, so this file holds no second copy of the
//! preprocessing or the generation loop.
//!
//! # Batching: deliberately serial, and here is why
//!
//! `Florence2Lm` rebuilds its decoder scratch per `ground` call (its shape
//! depends on that call's tokenized prompt length - see `florence2::text::
//! attn`'s module doc on why there is no KV cache either), so there is no
//! fixed graph to widen with a batch axis the way a conv backbone would be.
//! The default serial [`Instance::run_batch`] therefore stands, same
//! reasoning as `crate::resident_scrfd`'s.

use capability::{ActionResult, Invocation, Manifest, Progress};
use florence2::caps::FlorenceSession;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

use crate::resolver_cli::RoleEnv;

/// Florence-2 behind the scheduler (`BRAIN_FLORENCE2_DIR` = the directory
/// holding `config.json`, `model.safetensors` and `tokenizer.json`).
pub struct Florence2Resident {
    dir: String,
}

impl Florence2Resident {
    /// `BRAIN_FLORENCE2_DIR` if the operator set it, else whatever the
    /// model-store resolver finds for `florence2::spec::Florence2Spec`'s
    /// `weights` role - the same scan and candidate rules the one-shot CLI
    /// uses. `None` (not served, never a daemon startup failure) when the
    /// store holds no released checkpoint, or holds more than one and
    /// nothing says which.
    pub fn from_env() -> Option<Florence2Resident> {
        let assembly = crate::resolver_cli::served_assembly("florence2", &florence2::spec::Florence2Spec, &[RoleEnv { role: "weights", var: "BRAIN_FLORENCE2_DIR" }])?;
        Self::new(assembly.roles.get("weights")?.to_string_lossy().into_owned())
    }

    /// Direct constructor for callers that already hold the directory. Same
    /// validation as `from_env`.
    pub fn new(dir: impl Into<String>) -> Option<Florence2Resident> {
        let dir = dir.into();
        let d = std::path::Path::new(&dir);
        let missing: Vec<&str> = florence2::caps::Florence2Provider::RELEASE_FILES.iter().copied().filter(|f| !d.join(f).exists()).collect();
        if !missing.is_empty() {
            eprintln!("brain: florence2 not served ({dir} is missing {missing:?})");
            return None;
        }
        Some(Florence2Resident { dir })
    }
}

impl ResidentModel for Florence2Resident {
    fn manifest(&self) -> Manifest {
        florence2::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(florence2::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // model.safetensors is ~463 MiB fp16. No KV cache and a full-prefix
        // decoder recompute per generation step (text::attn's own module
        // doc) means activation scratch is real but bounded by the short
        // generation lengths this crate targets - a flat, generous bound
        // rather than a bare file-size sum, same reasoning as
        // `resident_scrfd`'s.
        let weights = std::fs::metadata(std::path::Path::new(&self.dir).join("model.safetensors")).map(|m| m.len()).unwrap_or(0);
        MemCost::new(weights.saturating_mul(3) / 2 + (1u64 << 30), 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let gpu = crate::resident_llm::on_device(device, || gpu_core::Gpu::new(&florence2::caps::SERVING_PIPELINES))?;
        Ok(Box::new(Florence2Instance { session: FlorenceSession::load(&self.dir, gpu)? }))
    }
}

/// A resident grounder.
struct Florence2Instance {
    session: FlorenceSession,
}

impl Instance for Florence2Instance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.session.run(action, inv)
    }
    // `run_batch` is deliberately the serial default - see the module doc.
}
