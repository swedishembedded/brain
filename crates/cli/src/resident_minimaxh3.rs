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

use std::sync::{Arc, Mutex};

use capability::{ActionResult, Invocation, Manifest, Progress};
use minimaxh3::config::H3TransformerConfig;
use minimaxh3::video_vae::VideoVaeConfig;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// MiniMax-H3 resident model family, gated on the three `BRAIN_MINIMAXH3_*`
/// weight vars (`minimaxh3::caps::Paths::from_env`).
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
    hot: Arc<Mutex<Option<(String, minimaxh3::model::H3Transformer)>>>,
}

impl MiniMaxH3Resident {
    /// `None` (not registered) unless all three `BRAIN_MINIMAXH3_*` vars are
    /// set.
    pub fn from_env() -> Option<MiniMaxH3Resident> {
        minimaxh3::caps::Paths::from_env().ok().map(|paths| MiniMaxH3Resident { id: minimaxh3::caps::MODEL.to_string(), paths, hot: Arc::new(Mutex::new(None)) })
    }
}

/// `(task, video_latent_frames, latent_height, latent_width,
/// num_audio_latents, dtype)` from an instance key
/// (`"{task}:{vf}x{vh}x{vw}:{audio}[@{dtype}]"`) - see [`instance_key`]'s own
/// doc for why the video and audio latent extents are named separately
/// rather than one being derivable from the other at THIS layer. `dtype`
/// defaults to `"f32"` when the `@{dtype}` suffix is absent - `resident_wan.
/// rs::parse_key`'s own precedent for a key shape that predates a dtype
/// dimension existing at all (there is only one storage tier for this model
/// today; the suffix is reserved for when a second one exists, e.g. an int8
/// AdaLN-precomputed backbone - `minimaxh3`'s own crate doc names this as a
/// planned, not-yet-built path).
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

/// The DiT's own real parameter count from its config numbers alone - no
/// weight has to be read. Cross-checked against the roadmap's own
/// independently-confirmed figure (`adaln_proj` = 13.0B of a 33B total) in
/// this module's own test.
fn h3_dit_param_count(cfg: &H3TransformerConfig) -> u64 {
    let (inner, hidden, ffn, hd, te) = (cfg.inner_dim() as u64, cfg.hidden_size as u64, cfg.ffn_dim as u64, cfg.attention_head_dim as u64, cfg.time_embed_dim as u64);
    // Attention (q/k/v/out, both q/k-norms) + FFN (fused gate/up + down) +
    // the two pre-norms - shared by both the main block and the (no-AdaLN)
    // refiner block.
    let attn_ffn = 4 * inner * hidden + 2 * hd + 2 * hidden + 3 * ffn * hidden;
    let adaln = cfg.adaln_out_features() as u64 * (te + 1); // linear.weight + linear.bias
    let block = attn_ffn + adaln;
    let refiner_block = attn_ffn;
    let top_level = hidden * (cfg.video_patch_dim() as u64 + 1) // proj_in
        + hidden * (cfg.audio_in_channels as u64 + 1) // audio_proj_in
        + hidden * (cfg.text_dim as u64 + 1) // context_embedder
        + cfg.time_embed_hidden_dim as u64 * (cfg.freq_dim as u64 + 1) // time_embedder.linear_1
        + te * (cfg.time_embed_hidden_dim as u64 + 1) // time_embedder.linear_2
        + hidden // refiner_final_norm
        + hidden // norm_out.norm
        + cfg.final_adaln_out_features() as u64 * (te + 1) // norm_out.linear
        + cfg.video_patch_dim() as u64 * (hidden + 1) // proj_out
        + cfg.audio_in_channels as u64 * (hidden + 1); // audio_proj_out
    cfg.num_layers as u64 * block + cfg.num_refiner_layers as u64 * refiner_block + top_level
}

/// Real per-dtype byte count for the DiT's weights. Only `f32` exists today
/// (this crate has no quantized storage tier yet - see [`parse_key`]'s own
/// doc), so every dtype currently maps to the same fp32 figure; the branch
/// exists so a future int8/int4 tier is a one-line addition here, matching
/// `resident_wan.rs::dit_weight_bytes`'s own shape.
fn dit_weight_bytes(cfg: &H3TransformerConfig, _dtype: &str) -> u64 {
    // Every dtype this crate can currently produce is fp32 - the `_dtype`
    // parameter is accepted (not read) purely so the planned int8/int4
    // branch this function's own doc names is a one-line addition to the
    // BODY here, not a signature change, once it exists. An unrecognized
    // `@{dtype}` suffix on a malformed/adversarial key must still cost
    // something rather than panic - this function never rejects `_dtype`.
    h3_dit_param_count(cfg) * 4
}

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
        InstanceKey::new(&self.id, format!("{action}:{vf}x{vh}x{vw}:{audio}"))
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        let Some((_, vf, vh, vw, audio, dtype)) = parse_key(&key.config) else {
            // An unparseable key must not read as "free" - `resident_wan.
            // rs::estimate`'s identical reasoning.
            return MemCost::new(8u64 << 30, 4u64 << 30);
        };
        let dit_cfg = H3TransformerConfig::real();
        let weights = dit_weight_bytes(&dit_cfg, &dtype);
        let vae_weights = manifest_bytes(&VideoVaeConfig::real().tensor_manifest()) + manifest_bytes(&minimaxh3::vocoder::VocoderConfig::h3_32khz().tensor_manifest());

        // Packed-sequence row count this request's shape produces (video
        // rows + audio rows, channel-major so `* AUDIO_CHANNELS`) - the
        // activation-buffer driver, the same role `token_count` plays in
        // `resident_wan.rs::estimate`.
        let (ph, pw) = (dit_cfg.patch_size[1] as u64, dit_cfg.patch_size[2] as u64);
        let video_rows = (vf as u64) * (vh as u64 / ph) * (vw as u64 / pw);
        let audio_rows = audio as u64 * minimaxh3::pipeline::AUDIO_CHANNELS as u64;
        let rows = video_rows + audio_rows;
        let hidden = dit_cfg.hidden_size as u64;
        // A handful of `[rows, hidden]` residual/scratch slabs alive at once
        // through one block's forward - `resident_wan.rs::estimate`'s own
        // `* 8` factor for the identical role.
        let activations = rows * hidden * 4 * 8;
        let pixels = vf as u64 * (vh as u64 * ph) * (vw as u64 * pw) * 3 * 4;

        MemCost::new(weights + vae_weights + activations, weights + vae_weights + pixels + (2u64 << 30))
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
    hot: Arc<Mutex<Option<(String, minimaxh3::model::H3Transformer)>>>,
}

impl MiniMaxH3Instance {
    /// Run `body` against the shared hot [`minimaxh3::model::H3Transformer`],
    /// loading (or reloading, on a device change) it first if needed. Holds
    /// the lock for the WHOLE forward - see [`Instance::run_batch`]'s own doc
    /// for why this model's residency is sequential by design, same as
    /// `resident_wan.rs::WanInstance::run_batch`'s reasoning.
    fn with_hot_dit<R>(&self, body: impl FnOnce(&minimaxh3::model::H3Transformer) -> Result<R, String>) -> Result<R, String> {
        let mut guard = self.hot.lock().map_err(|_| "minimaxh3: hot DiT lock poisoned")?;
        let want = device_key(self.device);
        let needs_load = !matches!(&*guard, Some((dk, _)) if dk == &want);
        if needs_load {
            let dit_tensors = minimaxh3::caps::read_tensors(&self.paths.dit)?;
            let model = minimaxh3::model::H3Transformer::load(&dit_tensors, H3TransformerConfig::real(), device_name(self.device).as_deref());
            *guard = Some((want, model));
        }
        let (_, model) = guard.as_ref().expect("just ensured");
        body(model)
    }
}

impl Instance for MiniMaxH3Instance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        // Params (and the license gate they carry) before the weights read -
        // `wan::caps::WanAction::run`'s own ordering.
        let p = minimaxh3::caps::gen_params_from(inv)?;
        let vae = minimaxh3::caps::VaeWeights::load(&self.paths)?;
        let dit_cfg = H3TransformerConfig::real();
        match action {
            "t2va" => self.with_hot_dit(|model| {
                let empty = vae::blocks::Tensors::new();
                let ckpt = vae.as_checkpoint(&empty, dit_cfg);
                minimaxh3::caps::t2va_hot_on(&ckpt, inv, &p, model)
            }),
            "fl2va" => {
                let specs = minimaxh3::caps::fl2va_keyframes_from(inv)?;
                let (h, w) = p.opts.canvas.unwrap_or_else(|| {
                    let vcfg = minimaxh3::video_vae::VideoVaeConfig::real();
                    let m = vcfg.spatial_compression_ratio() * dit_cfg.patch_size[2];
                    minimaxh3::pipeline::resolve_canvas_size(16.0, 9.0, m, minimaxh3::pipeline::CANVAS_SHORT_EDGE, minimaxh3::pipeline::CANVAS_MAX_PIXELS).unwrap_or((768, 1344))
                });
                let mut keyframes = Vec::with_capacity(specs.len());
                for spec in &specs {
                    let (hwc, w0, h0) = crate::image_io::load_image(&spec.path).map_err(|e| format!("fl2va: loading {}: {e}", spec.path))?;
                    let resized = if (w0, h0) != (w, h) { imaging::host::resize_bilinear_hwc(&hwc, 3, w0, h0, w, h) } else { hwc };
                    let pixels_rgb = hwc_unit_to_chw_255(&resized, w, h);
                    keyframes.push(minimaxh3::pipeline::encode_keyframe_condition(
                        &minimaxh3::video_vae::VideoVaeConfig::real(),
                        &vae.video_vae_tensors,
                        device_name(self.device).as_deref(),
                        &pixels_rgb,
                        h,
                        w,
                        &vae.video_latents_mean,
                        &vae.video_latents_std,
                        dit_cfg.patch_size,
                        spec.anchor,
                        p.opts.seed,
                    ));
                }
                self.with_hot_dit(|model| {
                    let empty = vae::blocks::Tensors::new();
                    let ckpt = vae.as_checkpoint(&empty, dit_cfg);
                    minimaxh3::caps::fl2va_hot_on(&ckpt, inv, &p, &keyframes, model)
                })
            }
            other => Err(format!("minimaxh3: unknown action '{other}'")),
        }
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

/// `imaging::host::resize_bilinear_hwc`'s output is still HWC `[0,1]`;
/// [`minimaxh3::pipeline::encode_keyframe_condition`] wants channel-major
/// `[0,255]` (its own doc, matching `encode_vae_condition`'s `pixels.div(255.0)`
/// - i.e. the INPUT side is `[0,255]`, not the VAE's internal range).
fn hwc_unit_to_chw_255(hwc: &[f32], w: u32, h: u32) -> Vec<f32> {
    let (w, h) = (w as usize, h as usize);
    let mut chw = vec![0f32; 3 * h * w];
    for i in 0..h * w {
        for c in 0..3 {
            chw[c * h * w + i] = (hwc[i * 3 + c] * 255.0).clamp(0.0, 255.0);
        }
    }
    chw
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn resident() -> MiniMaxH3Resident {
        MiniMaxH3Resident {
            id: minimaxh3::caps::MODEL.to_string(),
            paths: minimaxh3::caps::Paths { dit: "/dit".into(), video_vae: "/vae".into(), vocoder: "/vocoder".into() },
            hot: Arc::new(Mutex::new(None)),
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
        assert_eq!(dtype, "f32");
    }

    /// [`h3_dit_param_count`] must land near the roadmap's own independently
    /// confirmed numbers: `adaln_proj` alone is 13.0B of a 33B total.
    #[test]
    fn the_param_count_matches_the_roadmaps_independently_confirmed_numbers() {
        let cfg = H3TransformerConfig::real();
        let (inner, hidden, te) = (cfg.inner_dim() as u64, cfg.hidden_size as u64, cfg.time_embed_dim as u64);
        let adaln_total = cfg.num_layers as u64 * cfg.adaln_out_features() as u64 * (te + 1);
        let g = 1_000_000_000u64;
        assert!((12 * g..14 * g).contains(&adaln_total), "adaln_proj should land near 13.0B, got {adaln_total}");
        assert!(inner > 0 && hidden > 0, "sanity");

        let total = h3_dit_param_count(&cfg);
        assert!((28 * g..38 * g).contains(&total), "the DiT should land near 33B total, got {total}");
    }

    /// A real checkpoint's DiT is single-digit-tens-of-GB at fp32 (no
    /// quantized tier exists yet - see [`dit_weight_bytes`]'s own doc) - not
    /// zero, not absurd.
    #[test]
    fn the_estimate_is_nonzero_and_dominated_by_the_dit_weights() {
        let r = resident();
        let key = r.instance_key("t2va", &Invocation::new().set("prompt", json!("a")));
        let cost = r.estimate(&key);
        let gb = 1u64 << 30;
        assert!(cost.vram > 100 * gb, "the fp32 DiT should dominate at ~132GB, got {} GB", cost.vram / gb);
        assert!(cost.ram > cost.vram / 2, "host RAM should track the same weight figure");

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
        let fetched = MiniMaxH3Resident { id: "MiniMaxAI/MiniMax-H3".to_string(), paths: resident().paths, hot: Arc::new(Mutex::new(None)) };
        assert_eq!(fetched.manifest().model, "MiniMaxAI/MiniMax-H3");
        assert_eq!(fetched.instance_key("t2va", &Invocation::new()).model, "MiniMaxAI/MiniMax-H3");
    }

    #[test]
    fn hwc_unit_to_chw_255_converts_layout_and_scale() {
        // 1x2 HWC, RGB: px0=(1,0,0.5), px1=(0,1,0.25).
        let hwc = vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.25];
        let chw = hwc_unit_to_chw_255(&hwc, 2, 1);
        assert_eq!(chw.len(), 6);
        assert_eq!(chw[0], 255.0, "R plane, px0");
        assert_eq!(chw[1], 0.0, "R plane, px1");
        assert_eq!(chw[2], 0.0, "G plane, px0");
        assert_eq!(chw[3], 255.0, "G plane, px1");
        assert!((chw[4] - 127.5).abs() < 1e-3, "B plane, px0");
        assert!((chw[5] - 63.75).abs() < 1e-3, "B plane, px1");
    }
}
