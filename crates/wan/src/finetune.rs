// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end LoRA fine-tuning for the Wan DiT: turn a folder of **captioned
//! video clips** into a trained [`crate::lora::LoraAdapter`].
//!
//! Pipeline per run:
//!   1. Draw temporal windows from the dataset, VAE-encode each to a normalised
//!      latent volume and umT5-encode each caption (both once, up front; each
//!      encoder is **dropped** before the next stage allocates, the same
//!      sequential residency [`crate::pipeline`] is built around - umT5-XXL is
//!      22.72 GB in fp32 and provably does not fit the card).
//!   2. Flow-matching loop: draw `σ`, build `x_σ = (1−σ)·x₀ + σ·ε` with target
//!      `v = ε − x₀` and model time `σ·1000`
//!      ([`crate::modelgrad::make_flow_batch`] - the convention both samplers
//!      invert), run the adapter-applied frozen base through the gradchecked
//!      whole-model training path and project `dL/dW_eff` into an Adam step on
//!      the low-rank `A,B`.
//!   3. Save the adapter; the unchanged generation path picks it up through
//!      [`crate::lora::LoraAdapter::fold_into_tensors`].
//!
//! ## The dataset is `data::episode`, not a new format
//!
//! A video clip and a recorded episode are the same object: a run of frames
//! that must never be sampled across. `data::episode` already owns that -
//! `sample_window` / `iter_windows` are boundary-safe by construction, and
//! `EpisodeDataset::open` hard-validates `meta.json` against the actual file
//! sizes. So a Wan training set is an episode dataset (one episode per clip)
//! plus ONE new file, `captions.json`: a JSON array of one caption per episode.
//! Nothing about the windowing is re-implemented here; [`ClipSet`] adds the
//! caption lookup and the `[t][c][h][w] -> [c][t][h][w]` transpose the 3D VAE
//! wants, and both of those are tested.
//!
//! ## Which trainer runs the step
//!
//! [`crate::train::Trainer`] picks one of two paths, both computing the same
//! `(loss, ModelGrads<f32>)` from the same FD-gradchecked math:
//!
//! * the **device** trainer ([`crate::train::DeviceTrainer`]), which runs every
//!   DiT block's forward and backward on the GPU through
//!   [`crate::devgrad::BlockDev`] and keeps only the small wrapper ops on the
//!   host; and
//! * the **host** f32 instantiation of [`crate::modelgrad`], CPU-parallel
//!   through `model::hostmath::matvec_par`.
//!
//! [`TrainOpts::device`] selects; with `None` the device path is taken exactly
//! when brain's default backend is a real accelerator, so a machine without one
//! keeps working unchanged. `tests/device_train.rs` pins the two paths to the
//! same gradients and the same loss trajectory.

use std::path::Path;

use crate::config::WanConfig;
use crate::lora::{save_adapter, LoraAdapter, LoraCfg};
use crate::modelgrad::{make_flow_batch, Cfg, ModelWeights};
use crate::train::Trainer;
use crate::pipeline::Paths;
use crate::vae3d::{WanVaeConfig, WanVaeEncoder};
use data::episode::EpisodeDataset;
use data::rng::Rng;

/// One named tensor: `(name, shape, row-major f32 data)`.
pub type NamedTensor = (String, Vec<usize>, Vec<f32>);

/// The name of the captions file a [`ClipSet`] directory carries beside the
/// episode dataset.
pub const CAPTIONS_FILE: &str = "captions.json";

/// One training clip: a `[3, frames, h, w]` video in `[-1, 1]` plus its caption.
#[derive(Clone)]
pub struct Clip {
    /// `[3·frames·h·w]`, channel-outermost - the layout [`WanVaeEncoder`] reads.
    pub video: Vec<f32>,
    pub caption: String,
    pub frames: usize,
}

/// A captioned video-clip dataset: a `data::episode` dataset (one episode per
/// clip) plus [`CAPTIONS_FILE`].
pub struct ClipSet {
    ds: EpisodeDataset,
    captions: Vec<String>,
}

/// `[t][c][h][w]` in `[0,1]` (the episode reader's layout) ->
/// `[c][t][h][w]` in `[-1,1]` (the VAE encoder's).
///
/// Both halves are load-bearing and neither is visible in a forward: the
/// transpose because a video model's frame axis and channel axis are the same
/// size for exactly the shapes a test is tempted to use, and the range because
/// upstream encodes `[-1, 1]` - feeding `[0, 1]` produces a latent that decodes
/// to a washed-out but entirely plausible video.
pub fn frames_to_cthw(frames_f32: &[f32], t: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
    assert_eq!(frames_f32.len(), t * c * h * w, "frames_to_cthw: size");
    let hw = h * w;
    let mut out = vec![0f32; t * c * hw];
    for ti in 0..t {
        for ci in 0..c {
            let src = (ti * c + ci) * hw;
            let dst = (ci * t + ti) * hw;
            for i in 0..hw {
                out[dst + i] = frames_f32[src + i] * 2.0 - 1.0;
            }
        }
    }
    out
}

impl ClipSet {
    /// Open `dir`: an episode dataset plus [`CAPTIONS_FILE`], with one caption
    /// per episode (a count mismatch is an error, not a silent truncation - a
    /// misaligned caption trains the wrong text on every clip).
    pub fn load_dir(dir: &Path) -> Result<ClipSet, String> {
        let ds = EpisodeDataset::open(dir)?;
        let path = dir.join(CAPTIONS_FILE);
        let raw = std::fs::read_to_string(&path).map_err(|e| format!("wan finetune: cannot read {}: {e}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| format!("wan finetune: bad {CAPTIONS_FILE}: {e}"))?;
        let arr = v.as_array().ok_or_else(|| format!("wan finetune: {CAPTIONS_FILE} must be a JSON array of captions"))?;
        let captions: Vec<String> = arr.iter().map(|c| c.as_str().unwrap_or("").to_string()).collect();
        if captions.len() != ds.episodes.len() {
            return Err(format!("wan finetune: {} captions for {} clips", captions.len(), ds.episodes.len()));
        }
        Ok(ClipSet { ds, captions })
    }

    pub fn clips(&self) -> usize {
        self.ds.episodes.len()
    }

    /// `(c, h, w)` of one frame.
    pub fn frame_shape(&self) -> (usize, usize, usize) {
        (self.ds.c as usize, self.ds.h as usize, self.ds.w as usize)
    }

    /// The caption of the clip containing flat frame `i`.
    fn caption_at(&self, i: usize) -> &str {
        let k = self.ds.episodes.iter().position(|e| i >= e.start && i < e.start + e.len).unwrap_or(0);
        &self.captions[k]
    }

    /// One random window of `frames` frames, never crossing a clip boundary
    /// (`data::episode`'s guarantee, not a new one).
    pub fn sample(&self, rng: &mut Rng, frames: usize) -> Result<Clip, String> {
        let win = self.ds.sample_window(rng, frames)?;
        let (c, h, w) = self.frame_shape();
        Ok(Clip {
            video: frames_to_cthw(&win.frames_f32, frames, c, h, w),
            caption: self.caption_at(win.start_index).to_string(),
            frames,
        })
    }

    /// Deterministic sweep: every window of `frames` frames at `stride`-spaced
    /// starts, per clip, clips in order.
    pub fn iter_clips(&self, frames: usize, stride: usize) -> impl Iterator<Item = Clip> + '_ {
        let (c, h, w) = self.frame_shape();
        self.ds.iter_windows(frames, stride).map(move |win| Clip {
            video: frames_to_cthw(&win.frames_f32, frames, c, h, w),
            caption: self.caption_at(win.start_index).to_string(),
            frames,
        })
    }
}

/// A dataset sample after encoding: the clean normalised latent `x₀`
/// (`[z·F·H·W]`) and the caption's `[text_len · 4096]` umT5 features.
#[derive(Clone)]
pub struct Encoded {
    pub latent: Vec<f32>,
    pub ctx: Vec<f32>,
}

/// Encode every clip once: video -> normalised latent (Wan-VAE), caption ->
/// umT5-XXL features. Each encoder is built, used and **dropped** before the
/// next allocates. `progress(done, total, stage)` streams per-item progress and
/// `cancel` is polled per item, because both phases are minutes-scale.
#[allow(clippy::too_many_arguments)]
pub fn encode_samples(
    paths: &Paths,
    clips: &[Clip],
    cfg: &WanConfig,
    height: u32,
    width: u32,
    device: Option<&str>,
    te_device: Option<&str>,
    cancel: &capability::CancelToken,
    mut progress: impl FnMut(usize, usize, &str),
) -> Result<Vec<Encoded>, String> {
    let n = clips.len();
    if n == 0 {
        return Err("wan finetune: no clips to encode".into());
    }
    let frames = clips[0].frames;
    if clips.iter().any(|c| c.frames != frames || c.video.len() != clips[0].video.len()) {
        return Err("wan finetune: every clip must have the same shape".into());
    }

    // --- video -> latents (Wan-VAE, built once for the fixed clip shape) ---
    let latents: Vec<Vec<f32>> = {
        let vcfg = WanVaeConfig::wan21();
        let vraw = crate::pipeline::read_any(&paths.vae)?;
        let vweights = crate::import::import_vae(vraw, &vcfg)?;
        let want = 3 * frames * height as usize * width as usize;
        if clips[0].video.len() != want {
            return Err(format!("wan finetune: a clip is {} values, expected 3x{frames}x{height}x{width}", clips[0].video.len()));
        }
        let enc = WanVaeEncoder::build(&vcfg, &vweights, &vcfg.encode_chunks(frames as u32), height, width, device);
        let mut out = Vec::with_capacity(n);
        for (i, c) in clips.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err("cancelled".into());
            }
            progress(i, n, "encoding clips (Wan-VAE)");
            out.push(enc.encode(&c.video));
        }
        out
    }; // encoder + weights dropped here

    // --- captions -> umT5-XXL features ---
    // A standalone mirror of `pipeline::encode_text`, which encodes a
    // prompt/negative PAIR through a built pipeline; finetune needs N captions
    // and must not hold anything else resident while the 22.72 GB encoder is up.
    let ctxs: Vec<Vec<f32>> = {
        let tok = if Path::new(&paths.tokenizer).is_dir() {
            data::unigram::UnigramTokenizer::from_dir(&paths.tokenizer)
        } else {
            data::unigram::UnigramTokenizer::from_file(&paths.tokenizer)
        }?;
        let t5cfg = t5encoder::config::T5Config::umt5_xxl();
        let imported = t5encoder::import::import_wan(crate::pipeline::read_any(&paths.t5)?, &t5cfg)?;
        // Same precedence as `pipeline::encode_text`'s `--t5-device`: the
        // explicit option beats the environment variable, which beats the
        // CPU default (umT5-XXL is 22.72 GB in fp32).
        let te_env = std::env::var("BRAIN_WAN_T5_DEVICE").ok().filter(|s| !s.is_empty());
        let gpu = match te_device.or(te_env.as_deref()).unwrap_or("cpu") {
            "cpu" => gpu_core::Gpu::new_cpu(t5encoder::model::PIPELINES),
            "gpu" | "wgpu" => gpu_core::Gpu::new_wgpu(t5encoder::model::PIPELINES),
            _ => gpu_core::Gpu::new(t5encoder::model::PIPELINES),
        };
        let enc = t5encoder::model::T5Encoder::new_on(gpu, t5cfg, 1, cfg.text_len as u32, &t5encoder::import::to_init(imported));
        // A real clip set repeats a handful of caption templates across many
        // clips (e.g. `data::gen_clips`'s paraphrase pool), and umT5-XXL is a
        // CPU-minutes-scale forward - so an exact-string cache turns "one
        // forward per clip" into "one forward per UNIQUE caption" with no
        // change in output (the encoder is a pure deterministic function of
        // the token ids/mask).
        let mut cache: std::collections::HashMap<&str, Vec<f32>> = std::collections::HashMap::new();
        let mut out = Vec::with_capacity(n);
        for (i, c) in clips.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err("cancelled".into());
            }
            if let Some(cached) = cache.get(c.caption.as_str()) {
                progress(i, n, "encoding captions (umT5-XXL, cached)");
                out.push(cached.clone());
                continue;
            }
            progress(i, n, "encoding captions (umT5-XXL)");
            let (ids, mask) = tok.encode_padded(&c.caption, cfg.text_len);
            enc.set_tokens(&ids);
            enc.set_mask(&mask);
            enc.forward();
            enc.poll_wait();
            // `read_context` already zeroes the pad rows, which is exactly what
            // `WanModel.forward`'s `new_zeros` re-pad produces.
            let ctx = enc.read_context();
            cache.insert(c.caption.as_str(), ctx.clone());
            out.push(ctx);
        }
        out
    }; // encoder dropped here

    Ok(latents.into_iter().zip(ctxs).map(|(latent, ctx)| Encoded { latent, ctx }).collect())
}

/// LoRA fine-tuning hyper-parameters.
pub struct TrainOpts {
    pub steps: u32,
    pub rank: usize,
    pub lr: f32,
    /// Frames per training window. Must be `1 + 4k` (the causal VAE's rule).
    pub frames: usize,
    /// Windows to draw and encode up front.
    pub samples: usize,
    pub seed: u64,
    pub save_path: String,
    /// Write a checkpoint every N steps (0 = final only).
    pub ckpt_every: u32,
    /// Device for the VAE encode AND the DiT trainer; `None` takes brain's
    /// default backend, which uses the GPU wherever one is present.
    ///
    /// `"cpu"` forces the host f32 reference trainer; `"gpu"` forces the
    /// device one. With `None`, the device trainer is used only when brain's
    /// default backend resolves to a real accelerator, so a machine without
    /// one keeps the host path (see [`crate::train::Trainer::open`]).
    pub device: Option<String>,
    /// Device for the umT5 text encoder; `None` falls through to
    /// `BRAIN_WAN_T5_DEVICE`, then `"cpu"` - the same precedence
    /// `GenOpts::te_device` uses at inference.
    pub te_device: Option<String>,
}

/// Fine-tune a LoRA adapter on `dir` (a [`ClipSet`] folder). Returns the
/// adapter's tensors, ready to save. `progress(step, total, msg)` streams
/// encoding and per-step loss; `cancel` is polled every step, and periodic
/// checkpoints already written survive an abort.
pub fn run(
    paths: &Paths,
    dir: &Path,
    cfg: &WanConfig,
    opts: &TrainOpts,
    cancel: &capability::CancelToken,
    mut progress: impl FnMut(u32, u32, String),
) -> Result<Vec<NamedTensor>, String> {
    // The encode phase (VAE + umT5, two passes over `samples` clips) is
    // routinely the longest part of a short adapter run, so it gets its own
    // slice of the progress budget instead of sitting at `0/total` for its
    // whole duration (D8) - `encode_budget` items, then one training step
    // each, then one final "saved" tick.
    let encode_budget = 2 * opts.samples as u32;
    let total = encode_budget + opts.steps + 1;
    // 1. dataset
    let set = ClipSet::load_dir(dir)?;
    let (_, h, w) = set.frame_shape();
    let (lf, lh, lw) = cfg
        .latent_shape(opts.frames, w, h)
        .ok_or_else(|| format!("wan finetune: {} frames is not of the form 1 + 4k", opts.frames))?;
    progress(0, total, format!("{} clips of {w}x{h} in {}", set.clips(), dir.display()));

    // 2. encode (both encoders dropped before the trainer allocates)
    let mut rng = Rng::new(opts.seed);
    let mut clips = Vec::with_capacity(opts.samples);
    for _ in 0..opts.samples {
        clips.push(set.sample(&mut rng, opts.frames)?);
    }
    let n_samples = opts.samples as u32;
    let encoded = encode_samples(
        paths,
        &clips,
        cfg,
        h as u32,
        w as u32,
        opts.device.as_deref(),
        opts.te_device.as_deref(),
        cancel,
        |i, n, stage| {
            // The VAE pass reports `i` in `[0, n_samples)`; the umT5 pass
            // that follows it is offset by `n_samples` so the two phases
            // occupy disjoint, monotonically increasing slices of `total`.
            let base = if stage.contains("captions") { n_samples } else { 0 };
            progress(base + i as u32, total, format!("{stage} {}/{n}", i + 1));
        },
    )?;
    drop(clips);

    // 3. base weights -> host training format
    progress(encode_budget, total, "loading DiT weights".into());
    let tensors = crate::import::import_dit(crate::pipeline::read_any(&paths.dit)?, cfg)?;
    let tcfg = Cfg::from_wan(cfg, lf, lh, lw);
    let base = ModelWeights::from_tensors(&tcfg, &tensors)?;
    drop(tensors);

    // 4. adapter + flow-matching loop
    let mut trainer = Trainer::open(&tcfg, opts.device.as_deref());
    let mut adapter = LoraAdapter::new(&tcfg, LoraCfg::new(opts.rank));
    // With the base frozen and resident, a step's only weight traffic is the
    // rank-sized adapter itself: `W_eff` is folded and `dL/dW_eff` projected
    // on-device.
    let resident = trainer.begin_lora(&base, opts.rank);
    let route = if resident { " (base resident, on-device LoRA)" } else { "" };
    progress(encode_budget, total, format!("training on {}{route}", trainer.label()));
    for step in 0..opts.steps {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        // The sample index is drawn fresh from `rng` (D7) rather than cycled
        // deterministically (`step % len`): a deterministic cycle correlates
        // sample order with the σ draw below, since both come from walking
        // the same counter in lockstep every `encoded.len()` steps.
        let idx = rng.gen_range_inclusive(0, encoded.len() as i64 - 1) as usize;
        let s = &encoded[idx];
        // σ is drawn uniformly on (0, 1]; the clamp keeps the model time off
        // the exact 0 the samplers never evaluate.
        let sigma = (rng.next_f64() as f32).clamp(1e-3, 1.0) as f64;
        let noise: Vec<f32> = (0..s.latent.len()).map(|_| rng.next_gaussian() as f32).collect();
        let b = make_flow_batch(&tcfg, &s.latent, &s.ctx, cfg.text_len, sigma, &noise);
        let loss = trainer.lora_step(&base, &mut adapter, &b, opts.lr);
        progress(encode_budget + step + 1, total, format!("step {}/{}  loss {loss:.5}", step + 1, opts.steps));
        if opts.ckpt_every > 0 && (step + 1).is_multiple_of(opts.ckpt_every) && step + 1 < opts.steps {
            save_adapter(&opts.save_path, &adapter)?;
        }
    }
    save_adapter(&opts.save_path, &adapter)?;
    progress(total, total, format!("saved adapter -> {}", opts.save_path));
    Ok(adapter.to_tensors())
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::episode::EpisodeWriter;

    /// Build a two-clip dataset: clip 0 has 6 frames, clip 1 has 5, at 2x3
    /// RGB. Deliberately unequal lengths - a window sampler that ignores the
    /// episode table is only wrong near a boundary.
    fn tiny_set(dir: &Path) {
        let (c, h, w) = (3u32, 2u32, 3u32);
        let mut wr = EpisodeWriter::create(dir, c, h, w, 1, 8).expect("writer");
        for (clip, len) in [(0u8, 6usize), (1u8, 5usize)] {
            for f in 0..len {
                let frame: Vec<u8> = (0..(c * h * w) as usize).map(|i| (clip as usize * 100 + f * 10 + i) as u8).collect();
                wr.push(&frame, 0, None).expect("push");
            }
            wr.end_episode();
        }
        wr.finalize().expect("finalize");
        std::fs::write(dir.join(CAPTIONS_FILE), r#"["a red square","a blue square"]"#).expect("captions");
    }

    #[test]
    fn a_clipset_windows_within_clips_and_carries_the_right_caption() {
        let dir = std::env::temp_dir().join(format!("wan-clipset-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        tiny_set(&dir);
        let set = ClipSet::load_dir(&dir).expect("load");
        assert_eq!(set.clips(), 2);
        assert_eq!(set.frame_shape(), (3, 2, 3));

        // 4-frame windows: clip 0 offers starts 0..2, clip 1 offers 6..7. A
        // window starting at 3, 4 or 5 would straddle the boundary.
        let seen: Vec<(String, usize)> = set.iter_clips(4, 1).map(|c| (c.caption.clone(), c.video.len())).collect();
        assert_eq!(seen.len(), 3 + 2);
        assert_eq!(seen.iter().filter(|(cap, _)| cap == "a red square").count(), 3);
        assert_eq!(seen.iter().filter(|(cap, _)| cap == "a blue square").count(), 2);
        assert!(seen.iter().all(|(_, n)| *n == 4 * 3 * 2 * 3));

        // Sampling is boundary-safe for the same reason, and a window longer
        // than every clip is an error rather than a short read.
        let mut rng = Rng::new(4);
        for _ in 0..20 {
            let c = set.sample(&mut rng, 5).expect("sample");
            assert_eq!(c.video.len(), 5 * 3 * 2 * 3);
        }
        assert!(set.sample(&mut rng, 7).is_err(), "no clip has 7 frames");

        // A caption count that does not match the clip count is refused.
        std::fs::write(dir.join(CAPTIONS_FILE), r#"["only one"]"#).expect("captions");
        let Err(e) = ClipSet::load_dir(&dir) else { panic!("mismatched captions must fail") };
        assert!(e.contains("1 captions for 2 clips"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The episode reader hands back `[t][c][h][w]` in `[0,1]`; the 3D VAE
    /// wants `[c][t][h][w]` in `[-1,1]`. Non-square, non-equal t/c/h/w so a
    /// transposition cannot pass by coincidence.
    #[test]
    fn the_vae_transpose_moves_frames_under_channels_and_rescales() {
        let (t, c, h, w) = (2usize, 3usize, 2usize, 5usize);
        let src: Vec<f32> = (0..(t * c * h * w)).map(|i| i as f32 / 60.0).collect();
        let out = frames_to_cthw(&src, t, c, h, w);
        assert_eq!(out.len(), src.len());
        for ti in 0..t {
            for ci in 0..c {
                for i in 0..(h * w) {
                    let want = src[(ti * c + ci) * h * w + i] * 2.0 - 1.0;
                    assert_eq!(out[(ci * t + ti) * h * w + i], want, "t{ti} c{ci} i{i}");
                }
            }
        }
        // [0,1] -> [-1,1], not a pass-through.
        assert_eq!(frames_to_cthw(&vec![0.0; t * c * h * w], t, c, h, w)[0], -1.0);
        assert_eq!(frames_to_cthw(&vec![1.0; t * c * h * w], t, c, h, w)[0], 1.0);
    }
}
