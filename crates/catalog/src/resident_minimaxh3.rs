// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3 `t2va`/`fl2va` behind the residency scheduler
//! (`resident::build_executor`) - `resident_wan.rs`'s pattern, adapted to
//! this model's own real structural difference: see
//! `minimaxh3::caps`'s own module doc for why one resident
//! [`minimaxh3::model::H3Transformer`] serves every request shape (no
//! compiled graph is sized to `(frames, width, height)` the way `wan`'s or
//! `ltxv`'s own DiT residency precedents are), and for why the instance key
//! still names both latent extents anyway (matching this workspace's
//! residency-key convention, and forward-compatible with a future
//! compiled-graph backend) while the actual hot object is shared ACROSS
//! keys rather than one-per-key.
//!
//! The two video/audio VAEs are deliberately NOT resident (`minimaxh3::caps::
//! VaeWeights` is read fresh per call) - `wan::pipeline`'s own precedent for
//! its VAE/T5, restated in `minimaxh3::caps::VaeWeights`'s own doc: hundreds
//! of MB against the DiT's tens of GB, not worth caching.
//!
//! Swedish Embedded AB implements residency-scheduled serving adapters like
//! this one for its clients. If your team needs expertise in putting large
//! diffusion models behind a memory-budgeted scheduler, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::sync::Arc;

use capability::{ActionResult, Invocation, Manifest, Progress};
use minimaxh3::config::H3TransformerConfig;
use minimaxh3::video_vae::VideoVaeConfig;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// MiniMax-H3 resident model family, gated on `BRAIN_MINIMAXH3_DIR`
/// (`minimaxh3::caps::Paths::from_env`) - the out-of-store
/// developer lane (`serving::build_executor` registers this directly, the
/// same shape `WanResident::from_env` is wired there).
pub struct MiniMaxH3Resident {
    /// See `WanResident::id`'s doc (`resident_wan.rs`) for why this is not
    /// always the compiled-in constant.
    id: String,
    paths: minimaxh3::caps::Paths,
    /// The resident DiT, shared across EVERY activated instance/key -
    /// `(device_key, transformer)`. `minimaxh3::model::H3Transformer::
    /// forward` takes a packed sequence of any length per call, so unlike
    /// `WanResident::hot` (one `HotDit` per `(variant, extent, device)` key)
    /// this needs splitting only by DEVICE, not by request shape - see this
    /// module's own doc.
    hot: Arc<capability::hot::Hot<String, minimaxh3::model::H3Transformer>>,
}

impl MiniMaxH3Resident {
    /// The catalog/model-dir lane: one fetched-and-assembled
    /// `MiniMaxAI/MiniMax-H3` checkout whose five role directories
    /// (`minimaxh3::caps::Paths::from_assembly` resolved them) this resident
    /// serves under `id`. `serving::model_dir::resident_for_compound`'s
    /// `"minimaxh3"` arm builds one of these per scanned compound model.
    pub fn from_paths(id: String, paths: minimaxh3::caps::Paths) -> MiniMaxH3Resident {
        MiniMaxH3Resident { id, paths, hot: Arc::new(capability::hot::Hot::new()) }
    }

    /// `None` (not registered) unless `BRAIN_MINIMAXH3_DIR` is set.
    pub fn from_env() -> Option<MiniMaxH3Resident> {
        minimaxh3::caps::Paths::from_env().ok().map(|paths| MiniMaxH3Resident { id: minimaxh3::caps::MODEL.to_string(), paths, hot: Arc::new(capability::hot::Hot::new()) })
    }
}

/// `(task, video_latent_frames, latent_height, latent_width,
/// num_audio_latents, dtype)` from an instance key
/// (`"{task}:{vf}x{vh}x{vw}:{audio}[@{dtype}]"`) - see [`instance_key`]'s own
/// doc for why the video and audio latent extents are named separately
/// rather than one being derivable from the other at THIS layer. `dtype` is
/// the tier the DiT is held at ([`RESIDENT_DTYPE`]), `"f32"` when the
/// `@{dtype}` suffix is absent - `resident_wan.rs::parse_key`'s own
/// precedent.
fn parse_key(config: &str) -> Option<(String, u32, u32, u32, u32, String)> {
    let (rest, dtype) = match config.rsplit_once('@') {
        Some((r, d)) => (r, d.to_string()),
        None => (config, "f32".to_string()),
    };
    let mut it = rest.splitn(3, ':');
    let task = it.next()?.to_string();
    let mut dims = it.next()?.splitn(3, 'x');
    let (vf, vh, vw) = (dims.next()?.parse().ok()?, dims.next()?.parse().ok()?, dims.next()?.parse().ok()?);
    let audio: u32 = it.next()?.parse().ok()?;
    Some((task, vf, vh, vw, audio, dtype))
}

/// Resolve one invocation's `(video latent extent, audio latent extent)`
/// from its params alone - pure config-derived arithmetic
/// (`minimaxh3::pipeline`'s own free functions over `H3TransformerConfig::
/// real()`/`VideoVaeConfig::real()`, both pure data), matching
/// `minimaxh3::caps::decode_gen_params`'s own geometry resolution so the key
/// and the request it describes cannot disagree. Falls back to the
/// canvas/frame-count defaults on anything unparseable rather than erroring -
/// an instance key must always resolve to SOMETHING (`residency::
/// InstanceKey`'s own contract), and a malformed request is rejected properly
/// once `gen_params_from` runs inside `Instance::run`, not here.
fn latent_extents(inv: &Invocation) -> (u32, u32, u32, u32) {
    let dit_cfg = H3TransformerConfig::real();
    let vae_cfg = VideoVaeConfig::real();
    let canvas_multiple = vae_cfg.spatial_compression_ratio() * dit_cfg.patch_size[2];
    let (h, w) = match (inv.get_i64("width"), inv.get_i64("height")) {
        (Some(w), Some(h)) => (h.max(1) as u32, w.max(1) as u32),
        _ => minimaxh3::pipeline::resolve_canvas_size(16.0, 9.0, canvas_multiple, minimaxh3::pipeline::CANVAS_SHORT_EDGE, minimaxh3::pipeline::CANVAS_MAX_PIXELS).unwrap_or((768, 1344)),
    };
    let default_frames = (minimaxh3::pipeline::MIN_DURATION_S * minimaxh3::pipeline::FPS) as u32;
    let num_frames = inv.get_i64("num_frames").unwrap_or(default_frames as i64).max(1) as u32;
    let aligned = minimaxh3::pipeline::align_num_frames(num_frames, vae_cfg.clip_length, vae_cfg.tokens_chunk_size()).unwrap_or(num_frames);
    let num_latent_frames = minimaxh3::pipeline::video_latent_num_frames(aligned, vae_cfg.clip_length, vae_cfg.tokens_chunk_size());
    let ratio = vae_cfg.spatial_compression_ratio();
    let num_audio_latents = minimaxh3::pipeline::audio_latent_num_frames(aligned, minimaxh3::pipeline::FPS, minimaxh3::pipeline::AUDIO_LATENTS_PER_SECOND);
    (num_latent_frames, h / ratio, w / ratio, num_audio_latents)
}

/// Host bytes the resident DiT keeps beside its cards: every block's
/// `adaln_proj` (fp32, projected on the host per forward - 13.0B of the 33B
/// parameters) plus `norm_out`'s projection.
fn dit_host_bytes(cfg: &H3TransformerConfig) -> u64 {
    let te = cfg.time_embed_dim as u64 + 1;
    (cfg.num_layers as u64 * cfg.adaln_out_features() as u64 + cfg.final_adaln_out_features() as u64) * te * 4
}

/// The tier a key names (`@i8`), or `None` for one this adapter never builds.
fn key_tier(dtype: &str) -> Option<gpu_core::select::Dtype> {
    match dtype {
        "i8" => Some(gpu_core::select::Dtype::I8),
        "f32" => Some(gpu_core::select::Dtype::F32),
        _ => None,
    }
}

/// The key spelling of `minimaxh3::pipeline::RESIDENT_TIER`, the tier this
/// adapter holds its DiT at.
const RESIDENT_DTYPE: &str = "i8";

/// Packed rows a request of these latent extents runs (video + audio rows,
/// plus the text prompt at its encoder's longest) - what a forward's
/// workspace is sized from.
fn packed_rows(vf: u32, vh: u32, vw: u32, audio: u32) -> u32 {
    let dit_cfg = H3TransformerConfig::real();
    let (ph, pw) = (dit_cfg.patch_size[1], dit_cfg.patch_size[2]);
    vf * (vh / ph) * (vw / pw) + audio * minimaxh3::pipeline::AUDIO_CHANNELS + MAX_PROMPT_ROWS
}

/// The prompt rows a workspace is reserved for: a caption, never a document.
const MAX_PROMPT_ROWS: u32 = 512;

/// Total bytes of a VAE/vocoder config's own tensor manifest at 4
/// bytes/element (this repo's safetensors reader is always F32-materialized
/// on read) - `resident_wan.rs::estimate`'s own closed-form technique.
fn manifest_bytes(manifest: &[(String, Vec<usize>)]) -> u64 {
    manifest.iter().map(|(_, shape)| shape.iter().product::<usize>() as u64).sum::<u64>() * 4
}

impl ResidentModel for MiniMaxH3Resident {
    fn manifest(&self) -> Manifest {
        Manifest { model: self.id.clone(), ..minimaxh3::caps::manifest() }
    }

    /// Names the task plus BOTH latent extents plus the dtype - see this
    /// module's own doc and [`parse_key`]'s doc for why two extents, not
    /// one: [`minimaxh3::pipeline::video_latent_num_frames`]/
    /// [`minimaxh3::pipeline::audio_latent_num_frames`] are two different
    /// closed forms over the SAME aligned pixel frame count, not a shared
    /// divisor - `ltxv`'s own `parse_key` (`resident_ltxv.rs`, checked before
    /// writing this one) has no audio term at all in its key, because LTX's
    /// own audio latent track carries no extra degree of freedom beyond its
    /// video one; MiniMax-H3's does (audio latents/second vs. video's
    /// chunked VAE grouping are unrelated rates), so this key carries it
    /// explicitly instead of assuming the omission transfers.
    fn instance_key(&self, action: &str, inv: &Invocation) -> InstanceKey {
        let (vf, vh, vw, audio) = latent_extents(inv);
        InstanceKey::new(&self.id, format!("{action}:{vf}x{vh}x{vw}:{audio}@{}", RESIDENT_DTYPE))
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        let Some((_, vf, vh, vw, audio, dtype)) = parse_key(&key.config) else {
            // An unparseable key must not read as "free" - `resident_wan.
            // rs::estimate`'s identical reasoning.
            return MemCost::new(8u64 << 30, 4u64 << 30);
        };
        let dit_cfg = H3TransformerConfig::real();
        // An unknown tier must still cost the most this model could.
        let tier = key_tier(&dtype).unwrap_or(gpu_core::select::Dtype::F32);
        let weights = minimaxh3::model::resident_cost(&dit_cfg, tier).total();
        let vae_weights = manifest_bytes(&VideoVaeConfig::real().tensor_manifest()) + manifest_bytes(&minimaxh3::vocoder::VocoderConfig::h3_32khz().tensor_manifest());
        let activations = minimaxh3::model::block_workspace_bytes(&dit_cfg, packed_rows(vf, vh, vw, audio));
        let (ph, pw) = (dit_cfg.patch_size[1] as u64, dit_cfg.patch_size[2] as u64);
        let pixels = vf as u64 * (vh as u64 * ph) * (vw as u64 * pw) * 3 * 4;

        MemCost::new(weights + vae_weights + activations, dit_host_bytes(&dit_cfg) + vae_weights + pixels + (2u64 << 30))
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let (task, ..) = parse_key(&key.config).ok_or_else(|| format!("minimaxh3: bad instance key {:?}", key.config))?;
        if task != "t2va" && task != "fl2va" {
            return Err(format!("minimaxh3: unknown task {task:?} (t2va, fl2va)"));
        }
        // The DiT itself is built lazily, on the first run against this
        // device (`MiniMaxH3Instance::ensure_hot`) - `activate` only fixes
        // placement, matching `WanResident::activate`'s own "the DiT needs
        // the request's own conditioning ordering first" reasoning, though
        // here the deeper reason is simpler: nothing about the checkpoint or
        // the device is known to be wrong yet, so there is nothing this
        // function could usefully fail on before a real request arrives.
        Ok(Box::new(MiniMaxH3Instance { paths: self.paths.clone(), device, hot: self.hot.clone() }))
    }
}

/// The `device` string `minimaxh3::pipeline::GenOpts`/`H3Transformer::load`
/// take, and this instance's own hot-DiT cache key - `resident_wan.rs::
/// WanInstance::device_name`'s pattern, just also used to invalidate the
/// shared hot slot on a device change.
fn device_key(device: Device) -> String {
    match device {
        Device::Cpu => "cpu".to_string(),
        Device::Gpu(i) => format!("gpu{i}"),
        Device::Npu(i) => format!("npu{i}"),
    }
}

fn device_name(device: Device) -> Option<String> {
    match device {
        Device::Cpu => Some("cpu".to_string()),
        Device::Gpu(_) => Some("gpu".to_string()),
        Device::Npu(_) => None,
    }
}

/// A resident MiniMax-H3 instance: the weight paths, the assigned device, and
/// a handle onto the SHARED hot-DiT slot (see [`MiniMaxH3Resident::hot`]'s
/// doc - every instance this resident activates, regardless of its own key,
/// shares one transformer per device).
struct MiniMaxH3Instance {
    paths: minimaxh3::caps::Paths,
    device: Device,
    hot: Arc<capability::hot::Hot<String, minimaxh3::model::H3Transformer>>,
}

impl MiniMaxH3Instance {
    /// Run `body` against the shared hot [`minimaxh3::model::H3Transformer`],
    /// loading (or reloading, on a device change) it first if needed. Holds
    /// the lock for the WHOLE forward - see [`Instance::run_batch`]'s own doc
    /// for why this model's residency is sequential by design, same as
    /// `resident_wan.rs::WanInstance::run_batch`'s reasoning.
    fn with_hot_dit<R>(&self, rows: u32, body: impl FnOnce(&minimaxh3::model::H3Transformer) -> Result<R, String>) -> Result<R, String> {
        let want = device_key(self.device);
        self.hot.try_get(
            want,
            || {
                let dit_reader = minimaxh3::caps::open_dit_reader(&self.paths.dit)?;
                minimaxh3::model::H3Transformer::load(&dit_reader, H3TransformerConfig::real(), device_name(self.device).as_deref(), minimaxh3::pipeline::RESIDENT_TIER, rows)
            },
            body,
        )
    }
}

impl Instance for MiniMaxH3Instance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        // Params (and the license gate they carry) before the weights read -
        // `wan::caps::WanAction::run`'s own ordering.
        let p = minimaxh3::caps::gen_params_from(inv)?;
        let keyframe_specs = match action {
            "t2va" => Vec::new(),
            "fl2va" => minimaxh3::caps::fl2va_keyframes_from(inv)?,
            other => return Err(format!("minimaxh3: unknown action '{other}'")),
        };
        let vae = minimaxh3::caps::VaeWeights::load(&self.paths)?;
        let mut opts = p.opts.clone();
        opts.device = device_name(self.device);
        let keyframes = minimaxh3::caps::encode_keyframes(&vae, &keyframe_specs, &opts)?;
        let (vf, vh, vw, audio) = latent_extents(inv);
        self.with_hot_dit(packed_rows(vf, vh, vw, audio), |model| {
            let empty = vae::blocks::Tensors::default();
            let ckpt = vae.as_checkpoint(&empty, H3TransformerConfig::real());
            minimaxh3::caps::generate_on(&ckpt, inv, &p, &self.paths, &keyframes, Some(model))
        })
    }

    /// Sequential, deliberately - `resident_wan.rs::WanInstance::run_batch`'s
    /// own reasoning applies unchanged here: every job at this resident
    /// shares the SAME hot `H3Transformer` (now across every key too, not
    /// just one), so N requests pay one load between them, but the forward
    /// itself is not reentrant. Per-request cancellation is not yet real
    /// either way (`minimaxh3::caps::t2va_hot_on`'s own doc names the gap).
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        invs.iter().enumerate().map(|(i, inv)| self.run(action, inv, &mut |p| progress(i, p))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn resident() -> MiniMaxH3Resident {
        MiniMaxH3Resident {
            id: minimaxh3::caps::MODEL.to_string(),
            paths: minimaxh3::caps::Paths::resolve("/test-root"),
            hot: Arc::new(capability::hot::Hot::new()),
        }
    }

    #[test]
    fn key_parsing_round_trips_and_a_bad_key_still_costs_something() {
        assert_eq!(parse_key("t2va:37x32x32:207"), Some(("t2va".to_string(), 37, 32, 32, 207, "f32".to_string())));
        assert_eq!(parse_key("t2va:37x32x32:207@f32"), Some(("t2va".to_string(), 37, 32, 32, 207, "f32".to_string())));
        assert_eq!(parse_key("garbage"), None);
        let cost = resident().estimate(&InstanceKey::new(minimaxh3::caps::MODEL, "garbage".to_string()));
        assert!(cost.vram > 0 && cost.ram > 0, "{cost:?}");
    }

    /// The key must fix the task and BOTH latent extents, and per-call params
    /// (seed/steps/prompt) must not split the instance - `resident_wan.
    /// rs::the_instance_key_is_exactly_the_variant_and_the_latent_extent`'s
    /// own shape, adapted to this model's own two-extent key.
    #[test]
    fn the_instance_key_names_the_task_and_both_latent_extents() {
        let r = resident();
        let base = Invocation::new().set("prompt", json!("a"));
        let a = r.instance_key("t2va", &base);
        let b = r.instance_key("t2va", &Invocation::new().set("prompt", json!("something else")).set("seed", json!(7)).set("steps", json!(3)));
        assert_eq!(a.config, b.config, "per-call params must not split the instance");

        // A different task must split the instance.
        let fl = r.instance_key("fl2va", &base);
        assert_ne!(a.config, fl.config);

        // A different canvas must split the instance (a different video AND
        // audio latent extent - the aligned frame count changes too).
        let big = r.instance_key("t2va", &base.clone().set("width", json!(1024)).set("height", json!(576)));
        assert_ne!(a.config, big.config);

        // A different num_frames must split the instance (video extent
        // AND audio extent both derive from it).
        let longer = r.instance_key("t2va", &base.clone().set("num_frames", json!(240)));
        assert_ne!(a.config, longer.config);

        let (_, vf, vh, vw, audio, dtype) = parse_key(&a.config).expect("a well-formed key");
        assert!(vf > 0 && vh > 0 && vw > 0, "the video latent extent must be real, got {vf}x{vh}x{vw}");
        assert!(audio > 0, "the audio latent extent must be real, got {audio}");
        assert_eq!(dtype, "i8");
        assert_eq!(key_tier(&dtype), Some(minimaxh3::pipeline::RESIDENT_TIER), "the key names the tier the adapter loads");
    }

    /// The resident DiT is ~20 GB of int8 linears on the cards, with its
    /// ~52 GB of fp32 AdaLN projections held on the host.
    #[test]
    fn the_estimate_is_nonzero_and_dominated_by_the_dit_weights() {
        let r = resident();
        let key = r.instance_key("t2va", &Invocation::new().set("prompt", json!("a")));
        let cost = r.estimate(&key);
        let gb = 1u64 << 30;
        assert!((18 * gb..60 * gb).contains(&cost.vram), "int8 linears plus the workspace, got {} GB", cost.vram / gb);
        assert!(cost.ram > 45 * gb, "the host-side AdaLN projections dominate host RAM, got {} GB", cost.ram / gb);

        // A bigger request costs more (bigger activations/pixel buffer) on
        // top of the same fixed DiT-weight floor - two EXPLICIT canvases,
        // neither left to the (already near-max-pixels) default resolution
        // the no-width/height case above uses, so the comparison is not
        // accidentally reversed by the default itself being large.
        let small_key = r.instance_key("t2va", &Invocation::new().set("prompt", json!("a")).set("width", json!(256)).set("height", json!(256)));
        let big_key = r.instance_key("t2va", &Invocation::new().set("prompt", json!("a")).set("width", json!(512)).set("height", json!(512)));
        let (small_cost, big_cost) = (r.estimate(&small_key), r.estimate(&big_key));
        assert!(big_cost.vram > small_cost.vram, "a bigger canvas must cost more ({} vs {})", big_cost.vram, small_cost.vram);
    }

    #[test]
    fn activate_rejects_an_unknown_task() {
        let e = resident().activate(&InstanceKey::new(minimaxh3::caps::MODEL, "ref2va:1x1x1:1".to_string()), Device::Cpu).err().expect("an unknown task must not activate");
        assert!(e.contains("unknown task"), "{e}");
    }

    #[test]
    fn activate_accepts_known_tasks() {
        for task in ["t2va", "fl2va"] {
            assert!(resident().activate(&InstanceKey::new(minimaxh3::caps::MODEL, format!("{task}:1x1x1:1")), Device::Cpu).is_ok());
        }
    }

    #[test]
    fn the_adapter_advertises_the_shared_manifest() {
        let m = resident().manifest();
        assert_eq!(m.model, minimaxh3::caps::MODEL);
        assert_eq!(m.actions.len(), minimaxh3::caps::manifest().actions.len());
        let fetched = MiniMaxH3Resident::from_paths("MiniMaxAI/MiniMax-H3".to_string(), resident().paths);
        assert_eq!(fetched.manifest().model, "MiniMaxAI/MiniMax-H3");
        assert_eq!(fetched.instance_key("t2va", &Invocation::new()).model, "MiniMaxAI/MiniMax-H3");
    }
}