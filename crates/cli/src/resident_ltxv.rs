// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5 text-to-video behind the residency scheduler
//! (`resident::build_executor`) - `resident_wan.rs`'s pattern, simplified by
//! this milestone's one real difference: **there is nothing worth caching
//! yet**. `wan`'s hot cache exists because its DiT is 5.7+ GB and costs ~20s
//! to load+upload; this crate's DiT is always tiny-config with FRESH RANDOM
//! WEIGHTS (`ltxv::pipeline`'s module doc), so rebuilding it costs
//! microseconds - caching it would add state for no benefit. The VAE, per
//! `wan::pipeline`'s own precedent (VAE deliberately never held resident
//! alongside the DiT), is read fresh per call too. What residency buys here
//! is placement/budgeting/D-Bus reachability, not a warm cache - an honest
//! reflection of this milestone's real weight story, not a gap.

use capability::{ActionResult, Invocation, Manifest, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// LTX-2.5 resident model, gated on `BRAIN_LTXV_VAE` (the one real weight
/// role this milestone has - see `ltxv::pipeline::PATH_VARS`).
pub struct LtxvResident {
    /// See `WanResident::id`'s doc for why this is not always the compiled-in
    /// constant: a fetched checkpoint registers under its own reference.
    id: String,
    paths: ltxv::pipeline::Paths,
}

impl LtxvResident {
    /// `None` (not registered) unless `BRAIN_LTXV_VAE` is set.
    pub fn from_env() -> Option<LtxvResident> {
        ltxv::pipeline::Paths::from_env().ok().map(|paths| LtxvResident { id: ltxv::caps::MODEL.to_string(), paths })
    }
}

/// `(dit_config, frames, width, height)` from an instance key
/// (`"{dit_config}:{frames}:{w}x{h}"`) - the fields that would fix a real
/// DiT's graph sizes, kept as the key shape now so a later milestone's real
/// 22B import does not need to touch this file, even though nothing is
/// actually cached against it today (see this module's doc).
fn parse_key(config: &str) -> Option<(String, usize, usize, usize)> {
    let mut it = config.split(':');
    let dit_config = it.next()?.to_string();
    let frames: usize = it.next()?.parse().ok()?;
    let (w, h) = it.next()?.split_once('x')?;
    Some((dit_config, frames, w.parse().ok()?, h.parse().ok()?))
}

/// Total bytes of a config's weight manifest, at 4 bytes/element (this
/// repo's safetensors reader is always F32-materialized on read) - the same
/// closed-form-over-the-manifest technique `resident_wan.rs::estimate`
/// documents, just driven by `tensor_manifest`/`dit_tensor_manifest` instead
/// of a hand-derived formula.
fn manifest_bytes(manifest: &[(String, Vec<usize>)]) -> u64 {
    manifest.iter().map(|(_, shape)| shape.iter().product::<usize>() as u64).sum::<u64>() * 4
}

impl ResidentModel for LtxvResident {
    fn manifest(&self) -> Manifest {
        Manifest { model: self.id.clone(), ..ltxv::caps::manifest() }
    }

    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        let d = ltxv::pipeline::GenOpts::default();
        let dit_config = inv.get_str("dit_config").unwrap_or_else(|| "tiny".into());
        let frames = inv.get_i64("frames").unwrap_or(d.frames as i64);
        let w = inv.get_i64("width").unwrap_or(d.width as i64);
        let h = inv.get_i64("height").unwrap_or(d.height as i64);
        InstanceKey::new(&self.id, format!("{dit_config}:{frames}:{w}x{h}"))
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        let Some((dit_config, frames, w, h)) = parse_key(&key.config) else {
            // An unparseable key must not read as "free" - see
            // `resident_wan.rs::estimate`'s identical reasoning.
            return MemCost::new(1u64 << 30, 512u64 << 20);
        };
        let vae_weights = manifest_bytes(&ltxv::LtxVaeConfig::conv25().tensor_manifest());
        let dit_weights = ltxv::pipeline::dit_config_from_name(&dit_config).map(|c| manifest_bytes(&ltxv::dit::dit_tensor_manifest(&c))).unwrap_or(0);
        // The VAE decoder's own activations dominate at any real clip size
        // (pixel-space buffers, `3 * frames * h * w` floats at several
        // stages) - the same `* 4` term `resident_wan.rs::estimate` charges
        // for its own VAE decode.
        let pixels = frames as u64 * w as u64 * h as u64 * 3 * 4;
        MemCost::new(vae_weights + dit_weights + pixels * 4, vae_weights + dit_weights + (256u64 << 20))
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let (dit_config, _, _, _) = parse_key(&key.config).ok_or_else(|| format!("ltxv: bad instance key {:?}", key.config))?;
        // Validate at activation, not at the first request - a key naming a
        // `dit_config` this build cannot construct should fail placement.
        ltxv::pipeline::dit_config_from_name(&dit_config)?;
        Ok(Box::new(LtxvInstance { paths: self.paths.clone(), device }))
    }
}

/// A resident LTX-2.5 instance: the VAE path and the assigned device. Holds
/// no weights (see this module's doc) - every request rebuilds its own
/// tiny/random DiT and reads+decodes through its own freshly-imported VAE
/// graph, exactly like a one-shot CLI call.
struct LtxvInstance {
    paths: ltxv::pipeline::Paths,
    device: Device,
}

impl LtxvInstance {
    fn device_name(&self) -> Option<String> {
        match self.device {
            Device::Cpu => Some("cpu".to_string()),
            Device::Gpu(_) => Some("gpu".to_string()),
            Device::Npu(_) => None,
        }
    }
}

impl Instance for LtxvInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match action {
            "t2v" => {
                let mut p = ltxv::caps::gen_params_from(inv)?;
                p.opts.device = self.device_name();
                crate::resident_llm::on_device(self.device, || ltxv::caps::generate_on(&self.paths, inv, &p, progress))?
            }
            "dfr" => {
                // DFR needs the two latent-upscaler paths on top of the VAE
                // `self.paths` already carries - resolved fresh from the
                // environment per call, the same "nothing worth caching yet"
                // reasoning this module's own doc gives for the VAE path.
                let mut p = ltxv::caps::dfr_params_from(inv)?;
                p.opts.base.device = self.device_name();
                let dfr_paths = ltxv::pipeline::DfrPaths::from_env()?;
                crate::resident_llm::on_device(self.device, || ltxv::caps::dfr_on(&dfr_paths, inv, &p, progress))?
            }
            other => Err(format!("ltxv: unknown action '{other}'")),
        }
    }

    /// Sequential, deliberately - and for a stronger reason than `wan`'s own
    /// override: TWO requests here do not even share a resident model to
    /// serialize access to (there is none, see this module's doc), so a
    /// "batch" is just N independent one-shot generations run one after
    /// another. Per-request cancellation still works: each job's own
    /// `inv.cancel` is polled inside its own denoise loop.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        invs.iter().enumerate().map(|(i, inv)| self.run(action, inv, &mut |p| progress(i, p))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn resident() -> LtxvResident {
        LtxvResident { id: ltxv::caps::MODEL.to_string(), paths: ltxv::pipeline::Paths { vae: "/vae".into() } }
    }

    /// The key must fix exactly the fields that would size a real DiT's
    /// graph (dit_config + latent extent) and nothing per-call
    /// (seed/steps/guidance/prompt) - even though nothing is cached against
    /// it today, the shape must already be right for the day something is.
    #[test]
    fn the_instance_key_is_exactly_the_dit_config_and_the_latent_extent() {
        let r = resident();
        let base = Invocation::new().set("prompt", json!("a"));
        let a = r.instance_key("t2v", &base);
        assert_eq!(a.config, "tiny:9:64x64", "the defaults must key on the manifest's own defaults");
        let b = r.instance_key("t2v", &Invocation::new().set("prompt", json!("something else")).set("seed", json!(7)).set("steps", json!(2)).set("guidance", json!(3.0)));
        assert_eq!(a.config, b.config, "per-call params must not split the instance");
        for (k, v) in [("frames", json!(17)), ("width", json!(96)), ("height", json!(96))] {
            assert_ne!(a.config, r.instance_key("t2v", &base.clone().set(k, v)).config, "{k} must split the instance");
        }
    }

    #[test]
    fn key_parsing_round_trips_and_a_bad_key_still_costs_something() {
        assert_eq!(parse_key("tiny:9:64x64"), Some(("tiny".to_string(), 9, 64, 64)));
        assert_eq!(parse_key("garbage"), None);
        let cost = resident().estimate(&InstanceKey::new(ltxv::caps::MODEL, "garbage".to_string()));
        assert!(cost.vram > 0 && cost.ram > 0, "{cost:?}");
    }

    /// The estimate must land somewhere sane (single-digit GB, not zero, not
    /// absurd) for a real clip size - the exact number is not load-bearing,
    /// only that it is a real, nonzero, config-derived figure. The VAE's own
    /// weights dominate even the smallest clip (~726M real parameters,
    /// ~2.9 GB at this reader's f32 materialization - `crate::vae3d::
    /// LtxVaeConfig::manifest_counts_the_shipped_checkpoint`'s 170-tensor
    /// count times 4 bytes/element), NOT the tiny/random DiT (~KB) or the
    /// pixel buffers - so "small" here means "a bigger clip costs more on
    /// top of that fixed VAE floor", not "small in absolute terms".
    #[test]
    fn the_estimate_is_nonzero_and_grows_with_clip_size() {
        let r = resident();
        let small = r.estimate(&InstanceKey::new(ltxv::caps::MODEL, "tiny:9:64x64".to_string()));
        let big = r.estimate(&InstanceKey::new(ltxv::caps::MODEL, "tiny:17:256x256".to_string()));
        assert!(small.vram > 0 && small.ram > 0);
        assert!(big.vram > small.vram, "a bigger clip must cost more");
        let gb = 1u64 << 30;
        assert!(small.vram < 4 * gb, "the VAE-dominated floor should be well under 4 GB, got {}", small.vram);
    }

    #[test]
    fn activate_rejects_an_unknown_dit_config() {
        let e = resident().activate(&InstanceKey::new(ltxv::caps::MODEL, "22b:9:64x64".to_string()), Device::Cpu).err().expect("an unknown dit_config must not activate");
        assert!(e.contains("unknown ltxv dit-config"), "{e}");
    }

    #[test]
    fn the_adapter_advertises_the_shared_manifest() {
        let m = resident().manifest();
        assert_eq!(m.model, ltxv::caps::MODEL);
        assert_eq!(m.actions.len(), ltxv::caps::manifest().actions.len());
        let fetched = LtxvResident { id: "Lightricks/LTX-2.5".to_string(), paths: ltxv::pipeline::Paths { vae: "v".into() } };
        assert_eq!(fetched.manifest().model, "Lightricks/LTX-2.5");
        assert_eq!(fetched.instance_key("t2v", &Invocation::new()).model, "Lightricks/LTX-2.5");
    }
}
