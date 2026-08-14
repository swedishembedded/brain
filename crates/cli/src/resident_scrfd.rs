// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Face detection (insightface antelopev2 SCRFD-10GF) behind the residency
//! scheduler.
//!
//! `activate` imports the released ONNX graph from `BRAIN_SCRFD_DIR` ONCE and
//! builds it on the assigned device; the [`Instance`] owns the resulting
//! [`scrfd::caps::ScrfdSession`], so dropping it frees the model. One action,
//! `detect` - the schema and the work both come from `scrfd::caps`, so this file
//! holds no second copy of the letterbox or the box decode.
//!
//! # Batching: deliberately serial, and here is why
//!
//! The released graph has no batch axis in brain. `Scrfd::new` builds its whole
//! convolution schedule against `Shape::new(1, 3, side, side)`, and every
//! `Conv`/`BasicBlock`/`Head` buffer in the model is sized from that - so a
//! batched forward is not "pass N rows", it is a different set of built graphs.
//! Rebuilding them per batch size would re-upload the weights and defeat the
//! point.
//!
//! The default serial [`Instance::run_batch`] therefore stands. Widening the
//! *inference* graph to a configurable N (as `yolov8::Yolo::load(path, batch)`
//! does) is the follow-up that would make a genuine `run_batch` possible, and it
//! belongs in `crates/scrfd`, not here.

use capability::{ActionResult, Invocation, Manifest, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use scrfd::caps::ScrfdSession;

/// The antelopev2 face detector behind the scheduler (`BRAIN_SCRFD_DIR` = the
/// directory holding `scrfd_10g_bnkps.onnx`).
pub struct ScrfdResident {
    dir: String,
}

impl ScrfdResident {
    /// `None` when the directory is unset or does not hold the released graph
    /// - registering a model whose every call would fail is worse than not
    /// serving it.
    pub fn from_env() -> Option<ScrfdResident> {
        Self::new(std::env::var("BRAIN_SCRFD_DIR").ok().filter(|p| !p.is_empty())?)
    }

    /// Direct constructor for callers that already hold the directory (e.g.
    /// `brain perf`'s `scrfd:<dir>` target) - the compile-time seam that
    /// avoids round-tripping the path through the process environment (an
    /// env-name mismatch shipped that perf target dead on arrival). Same
    /// validation as `from_env`.
    pub fn new(dir: impl Into<String>) -> Option<ScrfdResident> {
        let dir = dir.into();
        let d = std::path::Path::new(&dir);
        let missing: Vec<&str> =
            scrfd::caps::ScrfdProvider::RELEASE_FILES.iter().copied().filter(|f| !d.join(f).exists()).collect();
        if !missing.is_empty() {
            eprintln!("brain: scrfd not served ({dir} is missing {missing:?})");
            return None;
        }
        Some(ScrfdResident { dir })
    }
}

impl ResidentModel for ScrfdResident {
    fn manifest(&self) -> Manifest {
        scrfd::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(scrfd::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // The graph is uploaded to the device: scrfd_10g_bnkps is ~17 MB of fp32
        // initializers. The activation slack dominates it by two orders of
        // magnitude at 640² (a 58-conv backbone + PAFPN, SSA taps) - hence a
        // flat, generous bound rather than a bare file-size sum.
        let files: u64 = scrfd::caps::ScrfdProvider::RELEASE_FILES
            .iter()
            .map(|f| std::fs::metadata(std::path::Path::new(&self.dir).join(f)).map(|m| m.len()).unwrap_or(0))
            .sum();
        MemCost::new(files.saturating_mul(12) / 10 + (2u64 << 30), 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Build the engine on the card the manager assigned (scoped registry
        // selection - never env mutation), then import once.
        let gpu = crate::resident_llm::on_device(device, || gpu_core::Gpu::new(&scrfd::caps::SERVING_PIPELINES))?;
        Ok(Box::new(ScrfdInstance { session: ScrfdSession::load(&self.dir, gpu)? }))
    }
}

/// A resident detector.
struct ScrfdInstance {
    session: ScrfdSession,
}

impl Instance for ScrfdInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.session.run(action, inv)
    }
    // `run_batch` is deliberately the serial default - see the module docs: the
    // released graph is built for a single image and has no N axis in brain.
}
