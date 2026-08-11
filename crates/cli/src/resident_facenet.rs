// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Face detection + recognition (insightface antelopev2) behind the residency
//! scheduler.
//!
//! `activate` imports both released ONNX graphs from `BRAIN_FACENET_DIR` ONCE
//! and builds them on the assigned device; the [`Instance`] owns the resulting
//! [`facenet::caps::Session`], so dropping it frees both models. Two actions,
//! `detect` and `embed` — the schemas and the work both come from
//! `facenet::caps`, so this file holds no second copy of the letterbox, the
//! alignment or the embedding normalisation.
//!
//! # Batching: deliberately serial, and here is why
//!
//! Neither released graph has a batch axis in brain. `Scrfd::new` and
//! `ArcFace::new` build their whole convolution schedule against
//! `Shape::new(1, 3, side, side)`, and every `Conv`/`BasicBlock`/`Head` buffer
//! in the model is sized from that — so a batched forward is not "pass N rows",
//! it is a different set of built graphs. Rebuilding them per batch size would
//! re-upload 280 MB of weights and defeat the point.
//!
//! The default serial [`Instance::run_batch`] therefore stands, and this comment
//! is the required statement of why (`.agents/rules/serving-contract.md` §3). The batch
//! axis exists in the architecture — `crates/facenet/src/train.rs` trains
//! ArcFace at batch > 1 — so widening the *inference* graphs to a configurable
//! N (as `yolo::Yolo::load(path, batch)` does) is the follow-up that would make
//! a genuine `run_batch` possible, and it belongs in `crates/facenet`, not here.

use capability::{ActionResult, Invocation, Manifest, Progress};
use facenet::caps::Session;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// The antelopev2 face stack behind the scheduler (`BRAIN_FACENET_DIR` = the
/// directory holding `glintr100.onnx` and `scrfd_10g_bnkps.onnx`).
pub struct FacenetResident {
    dir: String,
}

impl FacenetResident {
    /// `None` when the directory is unset or does not hold both released graphs
    /// — registering a model whose every call would fail is worse than not
    /// serving it.
    pub fn from_env() -> Option<FacenetResident> {
        Self::new(std::env::var("BRAIN_FACENET_DIR").ok().filter(|p| !p.is_empty())?)
    }

    /// Direct constructor for callers that already hold the directory (e.g.
    /// `brain perf`'s `facenet:<dir>` target) — the compile-time seam that
    /// avoids round-tripping the path through the process environment (an
    /// env-name mismatch shipped that perf target dead on arrival). Same
    /// validation as `from_env`.
    pub fn new(dir: impl Into<String>) -> Option<FacenetResident> {
        let dir = dir.into();
        let d = std::path::Path::new(&dir);
        let missing: Vec<&str> =
            facenet::caps::FacenetProvider::RELEASE_FILES.iter().copied().filter(|f| !d.join(f).exists()).collect();
        if !missing.is_empty() {
            eprintln!("brain: facenet not served ({dir} is missing {missing:?})");
            return None;
        }
        Some(FacenetResident { dir })
    }
}

impl ResidentModel for FacenetResident {
    fn manifest(&self) -> Manifest {
        facenet::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // One build serves both actions: `detect` and `embed` share the device
        // and `embed` calls the detector, so splitting them would double the
        // resident weights for no gain.
        InstanceKey::new(facenet::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // Both graphs are uploaded to the device: glintr100 is ~260 MB of fp32
        // initializers and scrfd_10g_bnkps ~17 MB. The activation slack is
        // dominated by SCRFD at 640² (a 58-conv backbone + PAFPN, SSA taps) —
        // hence a flat, generous bound rather than a bare file-size sum.
        let files: u64 = facenet::caps::FacenetProvider::RELEASE_FILES
            .iter()
            .map(|f| std::fs::metadata(std::path::Path::new(&self.dir).join(f)).map(|m| m.len()).unwrap_or(0))
            .sum();
        MemCost::new(files.saturating_mul(12) / 10 + (2u64 << 30), 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Build the engine on the card the manager assigned (scoped registry
        // selection — never env mutation), then import once.
        let gpu = crate::resident_llm::on_device(device, || gpu_core::Gpu::new(&facenet::caps::SERVING_PIPELINES))?;
        Ok(Box::new(FacenetInstance { session: Session::load(&self.dir, gpu)? }))
    }
}

/// A resident face stack: SCRFD + ArcFace on one shared device handle.
struct FacenetInstance {
    session: Session,
}

impl Instance for FacenetInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.session.run(action, inv)
    }
    // `run_batch` is deliberately the serial default — see the module docs: both
    // released graphs are built for a single image and have no N axis in brain.
}
