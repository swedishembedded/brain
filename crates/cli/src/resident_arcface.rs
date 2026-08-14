// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Face identity embedding (insightface antelopev2 ArcFace IResNet-100) behind
//! the residency scheduler.
//!
//! `activate` imports the released ONNX graph from `BRAIN_ARCFACE_DIR` ONCE and
//! builds it on the assigned device; the [`Instance`] owns the resulting
//! [`arcface::caps::ArcFaceSession`], so dropping it frees the model. One
//! action, `embed` - the schema and the work both come from `arcface::caps`, so
//! this file holds no second copy of the alignment or the embedding
//! normalisation.
//!
//! # The detector is loaded twice when both models are resident, on purpose
//!
//! `embed`'s default (`align = true`) detects the primary face first, so this
//! session builds its own detector from the same directory. When
//! `brain/scrfd` is *also* resident, the 17 MB detector is therefore on the
//! device twice - once as its own model, once inside this one. That is an
//! accepted trade of a few tens of megabytes for two models that can be
//! scheduled, evicted and served independently; sharing one build across two
//! residents would mean one model's eviction silently breaking the other's
//! default path.
//!
//! # Batching: deliberately serial, and here is why
//!
//! The released graph has no batch axis in brain. `ArcFace::new` builds its
//! whole convolution schedule against `Shape::new(1, 3, side, side)`, and every
//! buffer in the model is sized from that - so a batched forward is not "pass N
//! rows", it is a different set of built graphs. Rebuilding them per batch size
//! would re-upload 260 MB of weights and defeat the point.
//!
//! The default serial [`Instance::run_batch`] therefore stands. The batch axis
//! exists in the architecture - `crates/arcface`'s trainer trains at batch > 1 -
//! so widening the *inference* graph to a configurable N (as
//! `yolov8::Yolo::load(path, batch)` does) is the follow-up that would make a
//! genuine `run_batch` possible, and it belongs in `crates/arcface`, not here.

use arcface::caps::ArcFaceSession;
use capability::{ActionResult, Invocation, Manifest, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// The antelopev2 identity embedder behind the scheduler (`BRAIN_ARCFACE_DIR` =
/// the directory holding `glintr100.onnx`, and `scrfd_10g_bnkps.onnx` beside it
/// for the default aligned path).
pub struct ArcFaceResident {
    dir: String,
}

impl ArcFaceResident {
    /// `None` when the directory is unset or does not hold the released graph
    /// - registering a model whose every call would fail is worse than not
    /// serving it.
    pub fn from_env() -> Option<ArcFaceResident> {
        Self::new(std::env::var("BRAIN_ARCFACE_DIR").ok().filter(|p| !p.is_empty())?)
    }

    /// Direct constructor for callers that already hold the directory (e.g.
    /// `brain perf`'s `arcface:<dir>` target) - see
    /// `crate::resident_scrfd::ScrfdResident::new`'s rationale.
    ///
    /// Only `glintr100.onnx` is required. A directory without the detector still
    /// serves `embed --align false` on a pre-aligned crop, and the action itself
    /// reports the difference, so refusing to register the model would remove a
    /// working capability to prevent an error message.
    pub fn new(dir: impl Into<String>) -> Option<ArcFaceResident> {
        let dir = dir.into();
        let d = std::path::Path::new(&dir);
        let missing: Vec<&str> =
            arcface::caps::ArcFaceProvider::RELEASE_FILES.iter().copied().filter(|f| !d.join(f).exists()).collect();
        if !missing.is_empty() {
            eprintln!("brain: arcface not served ({dir} is missing {missing:?})");
            return None;
        }
        Some(ArcFaceResident { dir })
    }
}

impl ResidentModel for ArcFaceResident {
    fn manifest(&self) -> Manifest {
        arcface::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(arcface::caps::MODEL, "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // Both graphs are uploaded to the device: glintr100 is ~260 MB of fp32
        // initializers and the detector this session builds for `align = true`
        // another ~17 MB. The activation slack is dominated by the detector at
        // 640² - hence a flat, generous bound rather than a bare file-size sum.
        let d = std::path::Path::new(&self.dir);
        let files: u64 = arcface::caps::ArcFaceProvider::RELEASE_FILES
            .iter()
            .chain(scrfd::caps::ScrfdProvider::RELEASE_FILES.iter())
            .map(|f| std::fs::metadata(d.join(f)).map(|m| m.len()).unwrap_or(0))
            .sum();
        MemCost::new(files.saturating_mul(12) / 10 + (2u64 << 30), 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Build the engine on the card the manager assigned (scoped registry
        // selection - never env mutation), then import once. The detector rides
        // on the SAME device with its own kernel set (`Gpu::new_like`, inside
        // `ArcFaceSession::load`).
        let gpu = crate::resident_llm::on_device(device, || gpu_core::Gpu::new(&arcface::caps::SERVING_PIPELINES))?;
        Ok(Box::new(ArcFaceInstance { session: ArcFaceSession::load(&self.dir, gpu)? }))
    }
}

/// A resident embedder, with its own detector for the aligned path.
struct ArcFaceInstance {
    session: ArcFaceSession,
}

impl Instance for ArcFaceInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.session.run(action, inv)
    }
    // `run_batch` is deliberately the serial default - see the module docs: the
    // released graph is built for a single image and has no N axis in brain.
}
