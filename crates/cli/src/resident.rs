// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapters that put brain's real models behind the residency
//! [`Executor`], and [`build_executor`] which every serving path uses.
//!
//! Heavy, weight-holding models (z-image) get a bespoke adapter whose `activate`
//! builds the model on the assigned GPU and whose `Instance` owns it (so eviction
//! actually reclaims VRAM). Stateless providers (imageops, demo) ride the generic
//! [`residency::bridge::ProviderResident`]. The result is one `Executor` that
//! schedules, batches, and swaps every model uniformly, shared by `brain serve
//! --dbus` and (later) the CLI/event paths.

use std::sync::Arc;

use capability::{ActionResult, Blob, Invocation, Manifest, Media, Outcome, Progress};
use residency::bridge::ProviderResident;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};
use serde_json::json;
use zimage::pipeline::{HotPipeline, Image, Paths};

/// Build the shared executor with every model registered, sized to the given per-GPU
/// budgets. `gpus` is `(index, total_bytes)` per card; `reserved` bytes are kept free
/// on each. The RAM pool bounds CPU-resident models. Falls back gracefully if a heavy
/// model's weights are not configured (it is simply not registered).
pub fn build_executor(gpus: &[(u32, u64)], reserved: u64, ram_total: u64, policy: Policy) -> Executor {
    let mut budgets = residency::budget::Budgets::new();
    for &(i, total) in gpus {
        budgets.set(Device::Gpu(i), total, reserved);
    }
    budgets.set(Device::Cpu, ram_total, 0);

    let mut models: Vec<Arc<dyn ResidentModel>> = Vec::new();
    // z-image if its weights are configured (BRAIN_ZIMAGE_*).
    match ZImageResident::from_env() {
        Ok(z) => models.push(Arc::new(z)),
        Err(e) => eprintln!("brain: z-image not served over the scheduler ({e})"),
    }
    // Stateless helpers (no weights) — always available.
    models.push(Arc::new(ProviderResident::stateless(Arc::new(crate::imageops::ImageOps))));

    Executor::start(models, budgets, policy)
}

// ---------------------------------------------------------------- z-image

/// z-image behind the scheduler. A resident instance is a built [`HotPipeline`] for a
/// `(width, height, precision, adapter)` key — the DiT (and its encoder) on the GPU;
/// dropping it frees the VRAM. `text2image` runs on the resident pipeline; the
/// image-editing actions build fresh per call (they take a variable-size input) via a
/// held provider, so the full manifest still works over the bus.
pub struct ZImageResident {
    paths: Paths,
    provider: Arc<zimage::caps::ZImageProvider>,
}

impl ZImageResident {
    pub fn from_env() -> Result<ZImageResident, String> {
        Ok(ZImageResident { paths: Paths::from_env()?, provider: Arc::new(zimage::caps::ZImageProvider::load()?) })
    }
}

impl ResidentModel for ZImageResident {
    fn manifest(&self) -> Manifest {
        zimage::caps::manifest()
    }

    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        if action == "text2image" {
            let w = inv.get_i64("width").unwrap_or(1024);
            let h = inv.get_i64("height").unwrap_or(1024);
            let prec = if inv.get_str("precision").as_deref() == Some("fp32") { "fp32" } else { "int8" };
            let adapter = inv.get_str("adapter").unwrap_or_default();
            InstanceKey::new(zimage::caps::MODEL, format!("{w}x{h}:{prec}:{adapter}"))
        } else {
            // Editing/training actions build fresh per call — one transient instance.
            InstanceKey::new(zimage::caps::MODEL, format!("edit:{action}"))
        }
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        // int8 DiT (~13 GB) or fp32 sharded (~24 GB/card); edit builds are transient
        // and small-footprint (they build + drop within the call).
        let vram = if key.config.contains(":fp32:") {
            24u64 << 30
        } else if key.config.starts_with("edit:") {
            2u64 << 30
        } else {
            14u64 << 30
        };
        MemCost::new(vram, 0)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if let Device::Gpu(i) = device {
            // Place the DiT on the assigned card; the encoder card is z-image's own
            // (BRAIN_ZIMAGE_ENCODER_GPU) and left as configured.
            std::env::set_var("BRAIN_GPU_INDEX", i.to_string());
        }
        if key.config.starts_with("edit:") {
            // No persistent pipeline — the provider builds fresh per call.
            return Ok(Box::new(ZImageInstance { pipe: None, provider: self.provider.clone() }));
        }
        let (w, h, hifi, adapter) = parse_key(&key.config);
        let adapter = if adapter.is_empty() { None } else { Some(adapter.as_str()) };
        let pipe = HotPipeline::build_adapted(&self.paths, w, h, 64, hifi, adapter, |_| {})?;
        Ok(Box::new(ZImageInstance { pipe: Some(pipe), provider: self.provider.clone() }))
    }
}

/// A resident z-image instance: `pipe` when a text2image pipeline is built; the
/// `provider` handles the fresh-build editing/training actions.
struct ZImageInstance {
    pipe: Option<HotPipeline>,
    provider: Arc<zimage::caps::ZImageProvider>,
}

impl Instance for ZImageInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if action == "text2image" {
            let pipe = self.pipe.as_ref().ok_or("z-image: text2image instance has no pipeline")?;
            let prompt = inv.get_str("prompt").unwrap_or_default();
            let seed = inv.get_i64("seed").unwrap_or(42).max(0) as u64;
            let steps = inv.get_i64("steps").unwrap_or(8).max(1) as u32;
            let img = pipe.generate(&prompt, seed, steps, |s, t, m| progress(Progress { step: s, total: t, message: m.to_string() }));
            return Ok(emit_image(img));
        }
        // Editing / training: delegate to the provider's action (fresh build).
        use capability::Provider;
        let act = self.provider.action(action).ok_or_else(|| format!("z-image: unknown action '{action}'"))?;
        let inv = act.spec().validate(inv.clone())?;
        act.run(&inv, progress)
    }
}

/// Parse a `"WxH:precision:adapter"` instance key.
fn parse_key(config: &str) -> (u32, u32, bool, String) {
    let mut parts = config.splitn(3, ':');
    let wh = parts.next().unwrap_or("1024x1024");
    let prec = parts.next().unwrap_or("int8");
    let adapter = parts.next().unwrap_or("").to_string();
    let (w, h) = wh.split_once('x').unwrap_or(("1024", "1024"));
    (w.parse().unwrap_or(1024), h.parse().unwrap_or(1024), prec == "fp32", adapter)
}

/// Wrap a generated [`Image`] as an image-output [`Outcome`] (raw HWC f32 + `{w,h,c}`).
fn emit_image(img: Image) -> Outcome {
    let bytes: Vec<u8> = img.hwc.iter().flat_map(|f| f.to_le_bytes()).collect();
    Outcome::new()
        .set("width", json!(img.w))
        .set("height", json!(img.h))
        .blob("image", Blob::new(Media::Image, bytes).with_meta(json!({"w": img.w, "h": img.h, "c": 3})))
}
