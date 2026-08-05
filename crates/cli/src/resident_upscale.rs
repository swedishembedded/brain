// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-ESRGAN behind the residency scheduler.
//!
//! `BRAIN_ESRGAN_WEIGHTS` names `RealESRGAN_x4plus.pth` (or any RRDBNet
//! checkpoint — the shape is derived, so the anime and x2 variants work here
//! too).
//!
//! **The instance key carries the input size**, because the graph is recorded
//! for one `(h, w)`: two different image sizes are two different builds, and
//! pretending otherwise would silently hand the second one a graph shaped for
//! the first. `upscale::caps::Session` also caches its most recent build
//! internally, so a run of same-sized images does not rebuild even within one
//! instance — the key exists so the SCHEDULER's memory accounting is right, not
//! to force a rebuild.
//!
//! **`run_batch` is deliberately the serial default.** RRDBNet is a dense conv
//! net: its cost is linear in pixels and so is its peak VRAM, so grouping N
//! images saves no work and multiplies the high-water mark by N. See
//! `crates/upscale/src/caps.rs`.

use capability::{ActionResult, Invocation, Manifest, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

pub struct UpscaleResident {
    path: String,
}

impl UpscaleResident {
    /// `None` when the var is unset or names nothing that exists — registering a
    /// model whose every call would fail is worse than not serving it.
    pub fn from_env() -> Option<UpscaleResident> {
        let path = std::env::var("BRAIN_ESRGAN_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        std::path::Path::new(&path).exists().then_some(UpscaleResident { path })
    }
}

/// The input size a request implies, read from the image blob's own meta.
///
/// Falls back to 0x0 rather than guessing: a request with no size is rejected by
/// the action anyway, and a made-up key would put it in the wrong instance.
fn request_size(inv: &Invocation) -> (u32, u32) {
    inv.get_blob("image")
        .and_then(|b| {
            Some((b.meta.get("w")?.as_u64()? as u32, b.meta.get("h")?.as_u64()? as u32))
        })
        .unwrap_or((0, 0))
}

impl ResidentModel for UpscaleResident {
    fn manifest(&self) -> Manifest {
        upscale::caps::manifest()
    }

    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        let (w, h) = request_size(inv);
        // A tiled request builds at the TILE size, not the image size, so every
        // tiled call of the same tile shares one instance regardless of how big
        // the pictures are.
        let tile = inv.get_i64("tile").unwrap_or(0).max(0) as u32;
        let key = if tile > 0 {
            let side = tile + 2 * upscale::caps::TILE_HALO;
            format!("tile{side}")
        } else {
            format!("{w}x{h}")
        };
        InstanceKey::new(upscale::caps::MODEL, key)
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        // x4plus is ~67 MB of fp32 params. The activations dominate: the trunk
        // holds `num_feat` channels at input resolution across a dense block's
        // running concat, and the upsample stages hold 16x the pixels.
        let file = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
        let cfg = key.config.as_str();
        let px: u64 = match cfg.strip_prefix("tile") {
            Some(side) => side.parse::<u64>().map(|s| s * s).unwrap_or(0),
            None => cfg
                .split_once('x')
                .and_then(|(w, h)| Some(w.parse::<u64>().ok()? * h.parse::<u64>().ok()?))
                .unwrap_or(0),
        };
        // ~64 channels x 4 bytes x a handful of live buffers at 1x, plus the 16x
        // pixels the two upsample stages carry.
        MemCost::new(file.saturating_mul(12) / 10 + px.saturating_mul(64 * 4 * 24), 0)
    }

    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let gpu = crate::resident_llm::on_device(device, || gpu_core::Gpu::new(&upscale::KERNELS))?;
        Ok(Box::new(UpscaleInstance { session: upscale::caps::load(&self.path, gpu)? }))
    }
}

struct UpscaleInstance {
    session: upscale::caps::Session,
}

impl Instance for UpscaleInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match action {
            "upscale" => upscale::caps::run_upscale(&self.session, inv),
            other => Err(format!("upscale: unknown action '{other}'")),
        }
    }
    // `run_batch` is deliberately the serial default — see the module docs.
}
