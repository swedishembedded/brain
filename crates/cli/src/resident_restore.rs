// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CodeFormer face restoration and the VQ autoencoder under it, behind the
//! residency scheduler.
//!
//! Two adapters, one file, because they read the SAME released checkpoint
//! directory and their scope only differs by what part of it they use:
//!
//! * [`RestoreResident`] (`BRAIN_RESTORE_WEIGHTS`) — `restore_face`, the code
//!   Transformer + controllable feature transformation + the `w` dial.
//! * [`VqganResident`] (`BRAIN_VQGAN_WEIGHTS`) — `encode`/`decode`, the discrete
//!   autoencoder alone.
//!
//! `activate` imports once and the [`Instance`] owns the built graph, so
//! dropping it frees the device memory. The schemas and the work come from
//! `restore::caps` / `vqgan::caps`; this file holds no second copy of the
//! `[0,1] <-> [-1,1]` conversion or the code packing.
//!
//! # Batching: deliberately serial, and here is why
//!
//! Both graphs are RECORDED step lists over fixed buffers: `CodeFormer::new` and
//! `Vqgan::new` size `img_in`/`z`/`idx_in`/`out` from one `[3, H, W]` image and
//! `submit` replays exactly those steps. There is no N axis to widen at call
//! time — a batched forward would mean recording a second graph at batch N and
//! holding both sets of activations, which for a 512² VQ generator is a worse
//! trade than running twice. So the default serial [`Instance::run_batch`]
//! stands, and this comment is the required statement of why
//! (`.agents/rules/serving-contract.md` §3).
//!
//! What DOES batch here is the fidelity dial: `w` is a one-element device buffer
//! read by `scale_add`, not a recorded constant, so a sweep of N values over one
//! image runs on ONE resident instance with N buffer writes and no rebuild.

use capability::{ActionResult, Invocation, Manifest, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

// ---------------------------------------------------------------- restore

/// CodeFormer behind the scheduler. `BRAIN_RESTORE_WEIGHTS` is `codeformer.pth`
/// or the directory holding it.
pub struct RestoreResident {
    path: String,
}

impl RestoreResident {
    /// `None` when the var is unset or does not resolve to an existing file —
    /// registering a model whose every call would fail is worse than not
    /// serving it. Deliberately NOT falling back to `BRAIN_VQGAN_WEIGHTS`: that
    /// one commonly names `vqgan_code1024.pth`, which carries none of the
    /// CodeFormer tensors.
    pub fn from_env() -> Option<RestoreResident> {
        Self::new(std::env::var("BRAIN_RESTORE_WEIGHTS").ok().filter(|p| !p.is_empty())?)
    }

    /// Direct constructor (no env round-trip) — see
    /// `crate::resident_facenet::FacenetResident::new`'s rationale.
    pub fn new(path: impl Into<String>) -> Option<RestoreResident> {
        let path = path.into();
        std::path::Path::new(&restore::caps::checkpoint_path(&path)).exists().then_some(RestoreResident { path })
    }
}

impl ResidentModel for RestoreResident {
    fn manifest(&self) -> Manifest {
        restore::caps::manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // The graph is fixed at 512² and `w` is a buffer write, so every request
        // shares one build — which is exactly what makes a `w` sweep cheap.
        InstanceKey::new(restore::caps::MODEL, "512")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // `codeformer.pth` is ~377 MB of fp32 params, all uploaded; the 512²
        // generator's activations dominate the rest.
        let file = std::fs::metadata(restore::caps::checkpoint_path(&self.path)).map(|m| m.len()).unwrap_or(0);
        MemCost::new(file.saturating_mul(12) / 10 + (3u64 << 30), 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let gpu = crate::resident_llm::on_device(device, || gpu_core::Gpu::new(&restore::caps::SERVING_PIPELINES))?;
        Ok(Box::new(RestoreInstance { session: restore::caps::Session::new(restore::caps::load(&self.path, gpu)?) }))
    }
}

struct RestoreInstance {
    session: restore::caps::Session,
}

impl Instance for RestoreInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match action {
            "restore_face" => self.session.restore_face(inv),
            other => Err(format!("restore: unknown action '{other}'")),
        }
    }
    // `run_batch` is deliberately the serial default — see the module docs.
}

// ---------------------------------------------------------------- vqgan

/// The VQ autoencoder behind the scheduler (`BRAIN_VQGAN_WEIGHTS` = a released
/// checkpoint or the directory holding one).
pub struct VqganResident {
    path: String,
}

impl VqganResident {
    /// `None` when the var is unset or does not resolve to an existing file.
    pub fn from_env() -> Option<VqganResident> {
        Self::new(std::env::var("BRAIN_VQGAN_WEIGHTS").ok().filter(|p| !p.is_empty())?)
    }

    /// Direct constructor (no env round-trip) — see
    /// `crate::resident_facenet::FacenetResident::new`'s rationale.
    pub fn new(path: impl Into<String>) -> Option<VqganResident> {
        let path = path.into();
        std::path::Path::new(&vqgan::caps::checkpoint_path(&path)).exists().then_some(VqganResident { path })
    }
}

impl ResidentModel for VqganResident {
    fn manifest(&self) -> Manifest {
        vqgan::caps::manifest()
    }

    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        // The graph is built for one square side, so the side IS the config
        // fingerprint: `encode` at 512 and `decode` at 512 share an instance
        // (which is what makes the code round trip cheap), 256 gets its own.
        let size = inv.get_i64("size").unwrap_or(vqgan::caps::DEFAULT_SIZE).max(0);
        InstanceKey::new(vqgan::caps::MODEL, size.to_string())
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        let file = std::fs::metadata(vqgan::caps::checkpoint_path(&self.path)).map(|m| m.len()).unwrap_or(0);
        // Activations scale with the square side; 512² is the released
        // resolution and the bound below is sized for it.
        let side: u64 = key.config.parse().unwrap_or(512);
        let act = (2u64 << 30).saturating_mul(side.max(1) * side.max(1)) / (512 * 512);
        MemCost::new(file.saturating_mul(12) / 10 + act, 0)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let size: u32 = key.config.parse().map_err(|_| format!("vqgan: bad instance size '{}'", key.config))?;
        vqgan::caps::check_size(size)?;
        let gpu = crate::resident_llm::on_device(device, || gpu_core::Gpu::new(&vqgan::caps::SERVING_PIPELINES))?;
        Ok(Box::new(VqganInstance { session: vqgan::caps::Session::new(vqgan::caps::load(&self.path, size, gpu)?) }))
    }
}

struct VqganInstance {
    session: vqgan::caps::Session,
}

impl Instance for VqganInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.session.run(action, inv)
    }
    // `run_batch` is deliberately the serial default — see the module docs.
}
