// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end Wan2.1 text-to-video: prompt -> umT5-XXL conditioning -> a
//! flow-matching denoise loop over a 3D latent volume with classifier-free
//! guidance -> Wan-VAE decode -> RGB frames.
//!
//! The four components this composes are each gated against goldens on their
//! own ([`crate::vae3d`], [`crate::dev`], `t5encoder`, `diffusion::flowsolvers`);
//! what lives here is only the composition, and it follows
//! `wan/text2video.py` step for step: two forwards per step (conditional and
//! unconditional), `uncond + guide·(cond - uncond)`, then one scheduler
//! `step()`.
//!
//! ## Memory is the design constraint
//!
//! umT5-XXL is **22.72 GB in fp32** and does not fit the 24 GB card this was
//! written on - an earlier phase verified the OOM rather than assuming it. So
//! the three models are never resident at once. [`generate`] runs them as
//! three phases, each in its own scope:
//!
//! 1. **encode text** (umT5, `GenOpts::te_device`, default CPU) and drop the
//!    encoder before anything else is allocated;
//! 2. **denoise** (the DiT, `GenOpts::device`) and drop it;
//! 3. **decode** (the VAE) into frames.
//!
//! That is the same staging `flux2::pipeline` reaches with
//! `BRAIN_FLUX2_TE_DEVICE`, for the same reason. The text encoding survives
//! phase 1 as two `[512, 4096]` f32 blocks (16 MB), which is the whole point of
//! doing it first.
//!
//! ## Reproducibility
//!
//! `--seed` selects the initial noise through [`data::rng::Rng`] (SplitMix64 +
//! Box-Muller), so the same seed gives the same video on the same build. It is
//! deliberately **not** torch's Philox: reproducing that bit-for-bit is not
//! something brain's goldens ask for anywhere, and pretending otherwise would
//! be a claim no test backs.

use std::time::Instant;

use checkpoint::safetensors::StTensor;

use crate::config::WanConfig;
use crate::dev::WanDitDev;
use crate::vae3d::{WanVaeConfig, WanVaeDecoder};

/// Upstream's `sample_neg_prompt` (`wan/configs/shared_config.py`), the
/// unconditional branch of every Wan2.1 T2V generation. It is Chinese because
/// upstream's is; translating it would change the conditioning, which is a
/// silent quality regression rather than a cosmetic edit.
pub const DEFAULT_NEGATIVE_PROMPT: &str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

/// Which multistep solver drives the schedule. Both are gated bit-exactly
/// against the reference in `diffusion::flowsolvers`; they do **not** share a
/// starting sigma, so this is a real choice and not a label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solver {
    /// `--sample_solver unipc`, upstream's default.
    UniPc,
    /// `--sample_solver dpm++`.
    DpmPp,
}

impl Solver {
    pub fn from_name(s: &str) -> Result<Solver, String> {
        match s {
            "unipc" => Ok(Solver::UniPc),
            "dpm++" | "dpmpp" => Ok(Solver::DpmPp),
            other => Err(format!("unknown solver {other:?} (unipc, dpm++)")),
        }
    }
}

/// Where the four model roles live. Every one has an environment variable and
/// a flag twin at the CLI; [`Paths::resolve`] is the place the precedence is
/// decided, once, so the flag always wins.
#[derive(Clone, Debug)]
pub struct Paths {
    /// The DiT: a directory of safetensors shards, a single safetensors file,
    /// or a `.pth`.
    pub dit: String,
    /// The VAE: `Wan2.1_VAE.pth`, a diffusers `vae/` directory, or the
    /// safetensors inside one.
    pub vae: String,
    /// umT5-XXL: `models_t5_umt5-xxl-enc-bf16.pth` or an HF encoder directory.
    pub t5: String,
    /// The SentencePiece unigram tokenizer: a `tokenizer.json` or the
    /// directory holding one.
    pub tokenizer: String,
}

/// `(variable, human name)` for each role, in the order [`Paths`] declares
/// them. One table so the env reader and the "you are missing X" error cannot
/// disagree about the spelling.
pub const PATH_VARS: [(&str, &str); 4] = [
    ("BRAIN_WAN_DIT", "transformer"),
    ("BRAIN_WAN_VAE", "VAE"),
    ("BRAIN_WAN_T5", "umT5 text encoder"),
    ("BRAIN_WAN_TOKENIZER", "tokenizer"),
];

impl Paths {
    /// Every role from its environment variable.
    pub fn from_env() -> Result<Paths, String> {
        Paths::resolve(None, None, None, None)
    }

    /// Every role from an explicit value if given, else from its environment
    /// variable. A role that resolves to neither is an error naming BOTH the
    /// flag and the variable - a user who has just downloaded four files
    /// should be told the flag, and a user running the served path should be
    /// told the variable.
    pub fn resolve(dit: Option<&str>, vae: Option<&str>, t5: Option<&str>, tokenizer: Option<&str>) -> Result<Paths, String> {
        let pick = |flag: Option<&str>, i: usize, flag_name: &str| -> Result<String, String> {
            let (var, role) = PATH_VARS[i];
            if let Some(v) = flag.filter(|s| !s.is_empty()) {
                return Ok(v.to_string());
            }
            match std::env::var(var) {
                Ok(v) if !v.is_empty() => Ok(v),
                _ => Err(format!("no {role} weights: pass {flag_name} <path> or set {var}")),
            }
        };
        Ok(Paths {
            dit: pick(dit, 0, "--dit")?,
            vae: pick(vae, 1, "--vae")?,
            t5: pick(t5, 2, "--t5")?,
            tokenizer: pick(tokenizer, 3, "--tokenizer")?,
        })
    }
}

/// Everything a single generation varies. Built from a [`WanConfig`] so an
/// invocation that names only a prompt still runs upstream's own defaults.
#[derive(Clone, Debug)]
pub struct GenOpts {
    /// Video frames. Must be `1 + 4k` - the causal VAE gives the first frame
    /// its own latent frame, so 80 frames is not representable and 81 is.
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// Flow-matching sigma shift.
    pub shift: f32,
    /// Classifier-free guidance. `<= 1.0` skips the unconditional forward
    /// entirely, which is exact (the combination collapses to the conditional
    /// prediction) and halves the cost.
    pub guidance: f32,
    pub seed: u64,
    /// `None` uses [`DEFAULT_NEGATIVE_PROMPT`]; `Some("")` means genuinely no
    /// negative prompt.
    pub negative_prompt: Option<String>,
    pub solver: Solver,
    pub fps: usize,
    /// Device for the DiT and the VAE (`None` = the ambient default,
    /// `Some("cpu")`, `Some("gpu")`).
    pub device: Option<String>,
    /// Device for the text encoder. Defaults to `cpu` because umT5-XXL is
    /// 22.72 GB in fp32 and a 24 GB card cannot hold it plus its activations.
    pub te_device: Option<String>,
}

impl GenOpts {
    /// Upstream's defaults for this variant.
    pub fn from_config(cfg: &WanConfig) -> GenOpts {
        let (w, h) = cfg.sizes[0];
        GenOpts {
            frames: cfg.frame_num,
            width: w,
            height: h,
            steps: cfg.sample_steps,
            shift: cfg.sample_shift,
            guidance: cfg.guide_scale,
            seed: 0,
            negative_prompt: None,
            solver: Solver::UniPc,
            fps: cfg.sample_fps,
            device: None,
            te_device: None,
        }
    }
}

/// A generated clip: `frames` interleaved RGB8 images, each `width * height *
/// 3` bytes.
///
/// Deliberately not an `imaging::Rgb8`: this crate has no imaging dependency,
/// and the CLI is what turns frames into a container.
#[derive(Clone)]
pub struct Video {
    pub width: u32,
    pub height: u32,
    pub fps: usize,
    pub frames: Vec<Vec<u8>>,
}

/// Read a checkpoint from a directory of safetensors shards, one safetensors
/// file, or a torch `.pth` - the three forms the Wan repos actually ship.
fn read_any(path: &str) -> Result<Vec<StTensor>, String> {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        return checkpoint::safetensors::read_model_dir(p);
    }
    if !p.exists() {
        return Err(format!("{path} does not exist"));
    }
    if p.extension().is_some_and(|e| e == "pth" || e == "pt" || e == "bin") {
        return Ok(checkpoint::torchpt::read(path)?
            .into_iter()
            .map(|t| StTensor { name: t.name, shape: t.shape, data: t.data })
            .collect());
    }
    checkpoint::safetensors::read(path)
}

/// Phase 1: prompt and negative prompt through umT5-XXL, returned as two
/// `[text_len, 4096]` blocks with the pad rows already hard zero.
///
/// A function rather than inline code so the 22.72 GB encoder is **dropped at
/// the return**, before the DiT allocates anything. Both prompts ride one
/// `B = 2` forward: they run the same graph at the same length, so a second
/// forward would double the cost of the most expensive single phase for
/// nothing.
fn encode_text(
    cfg: &WanConfig,
    paths: &Paths,
    o: &GenOpts,
    prompt: &str,
    negative: &str,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let tok = if std::path::Path::new(&paths.tokenizer).is_dir() {
        data::unigram::UnigramTokenizer::from_dir(&paths.tokenizer)
    } else {
        data::unigram::UnigramTokenizer::from_file(&paths.tokenizer)
    }?;
    let (mut ids, mut mask) = tok.encode_padded(prompt, cfg.text_len);
    let (nid, nmask) = tok.encode_padded(negative, cfg.text_len);
    ids.extend(nid);
    mask.extend(nmask);

    let t5cfg = t5encoder::config::T5Config::umt5_xxl();
    let src = read_any(&paths.t5)?;
    let imported = t5encoder::import::import_wan(src, &t5cfg)?;
    // Placement precedence, matching every weight path in this module: the
    // explicit value (the CLI's `--t5-device`) beats the variable, and the
    // default is the CPU because 22.72 GB of fp32 weights plus ~4 GB of
    // activations do not fit the 24 GB card this was written on.
    let te_env = std::env::var("BRAIN_WAN_T5_DEVICE").ok().filter(|s| !s.is_empty());
    let gpu = match o.te_device.as_deref().or(te_env.as_deref()).unwrap_or("cpu") {
        "cpu" => gpu_core::Gpu::new_cpu(t5encoder::model::PIPELINES),
        "gpu" | "wgpu" => gpu_core::Gpu::new_wgpu(t5encoder::model::PIPELINES),
        // "default" defers to BRAIN_DEVICE, which is what a caller who knows
        // their card can hold 22.72 GB would use.
        _ => gpu_core::Gpu::new(t5encoder::model::PIPELINES),
    };
    let enc = t5encoder::model::T5Encoder::new_on(gpu, t5cfg, 2, cfg.text_len as u32, &t5encoder::import::to_init(imported));
    enc.set_tokens(&ids);
    enc.set_mask(&mask);
    enc.forward();
    enc.poll_wait();
    let ctx = enc.read_context();
    let half = cfg.text_len * 4096;
    if ctx.len() != 2 * half {
        return Err(format!("text encoder returned {} values, expected {}", ctx.len(), 2 * half));
    }
    Ok((ctx[..half].to_vec(), ctx[half..].to_vec()))
}

/// Standard-normal noise for the initial latent, from a seeded SplitMix64
/// stream. See the module doc for why this is not torch's Philox.
fn seeded_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = data::rng::Rng::new(seed);
    (0..n).map(|_| rng.next_gaussian() as f32).collect()
}

/// Text to video. `progress(done, total, phase)` is called before each phase
/// and once per denoise step; a multi-minute silent run reads as a hang, so
/// this is not optional decoration.
pub fn generate(
    cfg: &WanConfig,
    paths: &Paths,
    prompt: &str,
    o: &GenOpts,
    mut progress: impl FnMut(u32, u32, &str),
) -> Result<(Video, Timings), String> {
    let (pt, ph, pw) = cfg.patch_size;
    let (_, sh, sw) = cfg.vae_stride;
    if !o.width.is_multiple_of(sw * pw) || !o.height.is_multiple_of(sh * ph) {
        return Err(format!(
            "{}x{} is not a multiple of {}x{} (VAE stride x patch size)",
            o.width,
            o.height,
            sw * pw,
            sh * ph
        ));
    }
    let (lf, lh, lw) = cfg.latent_shape(o.frames, o.width, o.height).ok_or_else(|| {
        format!("{} frames is not of the form 1 + 4k - the causal VAE gives the first frame its own latent frame, so 1, 5, 9, ... 81 are the representable counts", o.frames)
    })?;
    if o.steps == 0 {
        return Err("--steps must be at least 1".into());
    }
    let tokens = (lf / pt) * (lh / ph) * (lw / pw);
    let n_latent = cfg.in_channels * lf * lh * lw;
    let mut timings = Timings::default();
    let total = o.steps as u32 + 3;

    // ---- phase 1: text ---------------------------------------------------
    progress(0, total, "text encode");
    let negative = o.negative_prompt.clone().unwrap_or_else(|| DEFAULT_NEGATIVE_PROMPT.to_string());
    let t = Instant::now();
    let (ctx_cond, ctx_uncond) = encode_text(cfg, paths, o, prompt, &negative)?;
    timings.text = t.elapsed().as_secs_f32();

    // ---- phase 2: denoise ------------------------------------------------
    progress(1, total, "load transformer");
    let t = Instant::now();
    let raw = read_any(&paths.dit)?;
    let weights = crate::import::import_dit(raw, cfg)?;
    // The text MLP runs on the host from these tensors, so both prompts are
    // embedded HERE, while the map is still alive and before the loop starts.
    let emb_cond = crate::model::text_embed(cfg, &weights, &ctx_cond, cfg.text_len);
    let emb_uncond = crate::model::text_embed(cfg, &weights, &ctx_uncond, cfg.text_len);
    drop(ctx_cond);
    drop(ctx_uncond);
    let dit = WanDitDev::build(cfg, &weights, lf as u32, lh as u32, lw as u32, o.device.as_deref(), &[]);
    drop(weights);
    timings.load_dit = t.elapsed().as_secs_f32();

    let mut latent = seeded_noise(n_latent, o.seed);
    let cfg_on = o.guidance > 1.0;
    let t = Instant::now();
    let timesteps: Vec<f32> = match o.solver {
        Solver::UniPc => {
            let mut s = diffusion::flowsolvers::FlowUniPcScheduler::new(Default::default());
            s.set_timesteps(o.steps, o.shift as f64);
            let ts = s.timesteps().to_vec();
            latent = denoise(&dit, &mut s, latent, &ts, &emb_cond, &emb_uncond, o, cfg_on, total, &mut progress)?;
            ts
        }
        Solver::DpmPp => {
            let mut s = diffusion::flowsolvers::FlowDpmSolverPlusPlusScheduler::new(Default::default());
            s.set_timesteps(o.steps, o.shift as f64);
            let ts = s.timesteps().to_vec();
            latent = denoise(&dit, &mut s, latent, &ts, &emb_cond, &emb_uncond, o, cfg_on, total, &mut progress)?;
            ts
        }
    };
    timings.denoise = t.elapsed().as_secs_f32();
    timings.steps = timesteps.len();
    timings.tokens = tokens;
    timings.forwards_per_step = if cfg_on { 2 } else { 1 };
    drop(dit);

    // ---- phase 3: decode -------------------------------------------------
    progress(total - 1, total, "vae decode");
    let t = Instant::now();
    let vcfg = WanVaeConfig::wan21();
    let vraw = read_any(&paths.vae)?;
    let vweights = crate::import::import_vae(vraw, &vcfg)?;
    let dec = WanVaeDecoder::build(&vcfg, &vweights, lf as u32, lh as u32, lw as u32, o.device.as_deref());
    drop(vweights);
    let chw = dec.decode(&latent);
    let frames = dec.frames() as usize;
    let (w, h) = (o.width, o.height);
    if chw.len() != 3 * frames * h * w {
        return Err(format!("VAE returned {} values, expected {}", chw.len(), 3 * frames * h * w));
    }
    // `WanVAE.decode` clamps to [-1, 1] OUTSIDE the model, so the clamp is the
    // pipeline's job, and it must happen before the rescale (clamping after
    // would leave out-of-range values to wrap in the u8 cast).
    let plane = frames * h * w;
    let out: Vec<Vec<u8>> = (0..frames)
        .map(|f| {
            let mut px = vec![0u8; h * w * 3];
            for c in 0..3 {
                let base = c * plane + f * h * w;
                for i in 0..h * w {
                    px[i * 3 + c] = (127.5 * (chw[base + i].clamp(-1.0, 1.0) + 1.0)) as u8;
                }
            }
            px
        })
        .collect();
    timings.decode = t.elapsed().as_secs_f32();
    progress(total, total, "done");
    Ok((Video { width: w as u32, height: h as u32, fps: o.fps, frames: out }, timings))
}

/// The two schedulers expose the same three calls, and the denoise loop must
/// be ONE implementation: a second copy is how `--sample_solver dpm++` ends up
/// silently running a different CFG fold from `unipc`.
trait FlowStep {
    fn step(&mut self, model_output: &[f32], sample: &[f32]) -> Vec<f32>;
}
impl FlowStep for diffusion::flowsolvers::FlowUniPcScheduler {
    fn step(&mut self, model_output: &[f32], sample: &[f32]) -> Vec<f32> {
        diffusion::flowsolvers::FlowUniPcScheduler::step(self, model_output, sample)
    }
}
impl FlowStep for diffusion::flowsolvers::FlowDpmSolverPlusPlusScheduler {
    fn step(&mut self, model_output: &[f32], sample: &[f32]) -> Vec<f32> {
        diffusion::flowsolvers::FlowDpmSolverPlusPlusScheduler::step(self, model_output, sample)
    }
}

/// The only two things the denoise loop asks of a model. A trait so the CFG
/// fold and the per-branch context upload can be gated by a fake instead of by
/// a 5.7 GB checkpoint and half an hour of GPU time - the two mistakes this
/// loop can make (folding the branches the wrong way round, and hoisting a
/// context upload out of the loop) both still produce plausible video, so
/// "it ran" is not evidence.
trait Denoiser {
    fn set_context_embed(&self, emb: &[f32]);
    fn forward(&self, latent: &[f32], t: f32) -> Vec<f32>;
}

impl Denoiser for WanDitDev {
    fn set_context_embed(&self, emb: &[f32]) {
        WanDitDev::set_context_embed(self, emb)
    }
    fn forward(&self, latent: &[f32], t: f32) -> Vec<f32> {
        WanDitDev::forward(self, latent, t)
    }
}

/// `wan/text2video.py`'s loop: per timestep, one conditional and one
/// unconditional forward at the SAME latent, combined as
/// `uncond + guide·(cond - uncond)`, then one scheduler step.
#[allow(clippy::too_many_arguments)]
fn denoise(
    dit: &dyn Denoiser,
    sched: &mut dyn FlowStep,
    mut latent: Vec<f32>,
    timesteps: &[f32],
    emb_cond: &[f32],
    emb_uncond: &[f32],
    o: &GenOpts,
    cfg_on: bool,
    total: u32,
    progress: &mut impl FnMut(u32, u32, &str),
) -> Result<Vec<f32>, String> {
    let t0 = Instant::now();
    for (i, &t) in timesteps.iter().enumerate() {
        // The context buffer is shared by both forwards, so the two uploads
        // have to bracket their own forward - hoisting either one out of the
        // loop silently conditions every step on whichever prompt was last.
        dit.set_context_embed(emb_cond);
        let cond = dit.forward(&latent, t);
        let pred = if cfg_on {
            dit.set_context_embed(emb_uncond);
            let uncond = dit.forward(&latent, t);
            cond.iter().zip(&uncond).map(|(&c, &u)| u + o.guidance * (c - u)).collect()
        } else {
            cond
        };
        if !pred.iter().all(|v| v.is_finite()) {
            return Err(format!("the denoiser produced non-finite values at step {} (t = {t})", i + 1));
        }
        latent = sched.step(&pred, &latent);
        let per = t0.elapsed().as_secs_f32() / (i + 1) as f32;
        let left = per * (timesteps.len() - i - 1) as f32;
        progress(i as u32 + 2, total, &format!("denoise t={t:.0} {per:.1}s/step, ~{left:.0}s left"));
    }
    Ok(latent)
}

/// Per-phase wall clock, the split any perf claim about this pipeline has to
/// be argued from.
#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub text: f32,
    pub load_dit: f32,
    pub denoise: f32,
    pub decode: f32,
    pub steps: usize,
    pub tokens: usize,
    pub forwards_per_step: usize,
}

impl Timings {
    pub fn total(&self) -> f32 {
        self.text + self.load_dit + self.denoise + self.decode
    }

    /// Seconds per DiT forward - the number that predicts what a bigger clip
    /// will cost, which "total time" does not.
    pub fn secs_per_forward(&self) -> f32 {
        let n = (self.steps * self.forwards_per_step).max(1);
        self.denoise / n as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defaults a bare `--prompt` invocation runs, straight from the
    /// config - if these drift, every default generation silently changes.
    #[test]
    fn gen_opts_defaults_come_from_the_config() {
        let cfg = WanConfig::t2v_1_3b();
        let o = GenOpts::from_config(&cfg);
        assert_eq!((o.frames, o.width, o.height), (81, 832, 480));
        assert_eq!((o.steps, o.shift, o.guidance), (50, 5.0, 5.0));
        assert_eq!(o.fps, 16);
        assert_eq!(o.solver, Solver::UniPc);
        // The text encoder defaults OFF the accelerator on purpose.
        assert_eq!(o.te_device, None);
    }

    /// The flag must win over the variable, in both directions, for every one
    /// of the four roles - this is the whole contract of `--dit` and friends.
    #[test]
    fn an_explicit_path_beats_the_environment_variable() {
        // Set every variable to a marker, then override one role at a time.
        for (var, _) in PATH_VARS {
            std::env::set_var(var, format!("env-{var}"));
        }
        let p = Paths::resolve(None, None, None, None).expect("all four from env");
        assert_eq!(p.dit, "env-BRAIN_WAN_DIT");
        assert_eq!(p.tokenizer, "env-BRAIN_WAN_TOKENIZER");

        let p = Paths::resolve(Some("/flag/dit"), None, Some("/flag/t5"), None).expect("mixed");
        assert_eq!(p.dit, "/flag/dit");
        assert_eq!(p.t5, "/flag/t5");
        assert_eq!(p.vae, "env-BRAIN_WAN_VAE");
        assert_eq!(p.tokenizer, "env-BRAIN_WAN_TOKENIZER");

        // An empty flag is not a value: it must fall through to the variable
        // rather than resolving to "".
        let p = Paths::resolve(Some(""), None, None, None).expect("empty flag falls through");
        assert_eq!(p.dit, "env-BRAIN_WAN_DIT");

        for (var, _) in PATH_VARS {
            std::env::remove_var(var);
        }
        let e = Paths::resolve(None, None, None, None).unwrap_err();
        // The error names BOTH spellings, because the two audiences differ.
        assert!(e.contains("--dit") && e.contains("BRAIN_WAN_DIT"), "{e}");
    }

    /// A frame count that is not `1 + 4k` must be rejected with an explanation,
    /// not rounded: a silently truncated clip is the failure this catches.
    #[test]
    fn a_bad_frame_count_is_rejected_before_any_weight_is_read() {
        let cfg = WanConfig::t2v_1_3b();
        let paths = Paths { dit: "/nope".into(), vae: "/nope".into(), t5: "/nope".into(), tokenizer: "/nope".into() };
        let o = GenOpts { frames: 8, width: 128, height: 128, ..GenOpts::from_config(&cfg) };
        let e = generate(&cfg, &paths, "x", &o, |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("1 + 4k"), "{e}");

        // Same for a size the patch grid cannot tile.
        let o = GenOpts { frames: 9, width: 130, height: 128, ..GenOpts::from_config(&cfg) };
        let e = generate(&cfg, &paths, "x", &o, |_, _, _| {}).err().expect("must be rejected");
        assert!(e.contains("multiple of"), "{e}");
    }

    #[test]
    fn seeded_noise_is_reproducible_and_actually_normal() {
        let a = seeded_noise(4096, 42);
        assert_eq!(a, seeded_noise(4096, 42), "the same seed must give the same noise");
        assert_ne!(a, seeded_noise(4096, 43), "a different seed must give different noise");
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        let var = a.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "variance {var}");
    }

    /// Records which context each forward saw, and answers with it - so the
    /// loop's own bookkeeping is observable without a model.
    struct FakeDit {
        seen: std::cell::RefCell<Vec<f32>>,
        ctx: std::cell::Cell<f32>,
    }
    impl Denoiser for FakeDit {
        fn set_context_embed(&self, emb: &[f32]) {
            self.ctx.set(emb[0]);
        }
        fn forward(&self, latent: &[f32], _t: f32) -> Vec<f32> {
            self.seen.borrow_mut().push(self.ctx.get());
            vec![self.ctx.get(); latent.len()]
        }
    }

    /// Records the (already CFG-folded) prediction each step hands the solver.
    struct FakeSched(std::cell::RefCell<Vec<Vec<f32>>>);
    impl FlowStep for FakeSched {
        fn step(&mut self, model_output: &[f32], sample: &[f32]) -> Vec<f32> {
            self.0.borrow_mut().push(model_output.to_vec());
            sample.to_vec()
        }
    }

    fn run_loop(guidance: f32) -> (Vec<f32>, Vec<Vec<f32>>) {
        let cfg = WanConfig::t2v_1_3b();
        let o = GenOpts { guidance, ..GenOpts::from_config(&cfg) };
        let dit = FakeDit { seen: Default::default(), ctx: std::cell::Cell::new(f32::NAN) };
        let mut sched = FakeSched(Default::default());
        // The context "embeddings" are one marker value each: 1.0 conditional,
        // 0.0 unconditional.
        let (cond, uncond) = (vec![1.0f32; 4], vec![0.0f32; 4]);
        let ts = [999.0f32, 500.0];
        let out = denoise(&dit, &mut sched, vec![0.0; 4], &ts, &cond, &uncond, &o, guidance > 1.0, 9, &mut |_, _, _: &str| {})
            .expect("the fake denoiser is finite");
        assert_eq!(out.len(), 4);
        let seen = dit.seen.borrow().clone();
        let preds = sched.0.borrow().clone();
        (seen, preds)
    }

    /// The CFG fold is `uncond + g·(cond - uncond)`, and each branch must upload
    /// its OWN context immediately before its own forward. With the markers
    /// above that means the forwards alternate 1, 0, 1, 0 - a hoisted upload
    /// would make them constant - and the folded prediction is exactly `g`.
    #[test]
    fn cfg_runs_two_bracketed_forwards_and_folds_them_the_reference_way() {
        let (seen, preds) = run_loop(5.0);
        assert_eq!(seen, vec![1.0, 0.0, 1.0, 0.0], "each branch must upload its own context");
        assert_eq!(preds.len(), 2, "one solver step per timestep");
        for p in &preds {
            assert!(p.iter().all(|&v| v == 5.0), "0 + 5*(1 - 0) = 5, got {p:?}");
        }
    }

    /// Guidance at or below 1 collapses the combination to the conditional
    /// prediction exactly, so the unconditional forward is skipped rather than
    /// computed and multiplied by zero - half the cost, same answer.
    #[test]
    fn guidance_of_one_runs_a_single_forward_per_step() {
        let (seen, preds) = run_loop(1.0);
        assert_eq!(seen, vec![1.0, 1.0], "one forward per step, always the conditional one");
        for p in &preds {
            assert!(p.iter().all(|&v| v == 1.0), "{p:?}");
        }
    }

    #[test]
    fn solver_names_round_trip() {
        assert_eq!(Solver::from_name("unipc"), Ok(Solver::UniPc));
        assert_eq!(Solver::from_name("dpm++"), Ok(Solver::DpmPp));
        assert!(Solver::from_name("euler").is_err());
    }
}
