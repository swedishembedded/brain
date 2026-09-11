// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end LoRA fine-tuning for FLUX.2 Klein: turn a folder of captioned
//! images (`data::imageset`) into a trained [`crate::lora::LoraAdapter`].
//!
//! Pipeline per run:
//!   1. VAE-encode each image to packed latent tokens and Qwen-encode each
//!      caption to the 3-tap conditioning (both once, up front; both encoders
//!      are then **dropped** so their memory is reclaimed before training).
//!   2. Rectified-flow loop: each step draws a σ, builds
//!      `x_σ = (1−σ)·x₀ + σ·ε` with target velocity `v = ε − x₀`
//!      ([`crate::modelgrad::make_flow_batch`] — the exact convention
//!      [`crate::pipeline`]'s Euler integrator inverts), runs it through the
//!      frozen base under the chosen [`Trainer`], and Adam-steps the low-rank
//!      `A,B`. The host trainer gets there via a dense `dL/dW_eff` it then
//!      projects, one block at a time as the backward completes it
//!      ([`crate::modelgrad::grads_into`]) so the whole stack is never
//!      resident; the device trainer produces `(dA, dB)` directly.
//!   3. Save the adapter ([`crate::lora::save_adapter`]); the inference path
//!      picks it up via [`crate::lora::LoraAdapter::fold_into_tensors`].
//!
//! **What a host run costs in RAM.** The host trainer differentiates a dense
//! `W_eff`, so a step holds the frozen base AND its effective copy - two fp32
//! copies of the model - plus the saved activations. That is inherent to the
//! reference; it is reported up front by [`run`] rather than discovered from
//! the OOM killer, and it is why the device trainer, which keeps the base on
//! the card and forms no dense `dW` at all, is the default.
//!
//! **Two trainers, one op sequence.** [`Trainer::Host`] is the f32
//! instantiation of the FD-gradchecked reference math ([`crate::modelgrad`]) -
//! correct, deterministic, and CPU-parallel only through
//! `model::hostmath::matvec_par`. It is the oracle the device path is gated
//! against (`tests/dev_grad.rs`, `tests/device_train.rs`) and it stays.
//! [`Trainer::Device`] replays the same op sequence on the GPU through
//! [`crate::devtrain::DeviceTrainer`], with the base frozen on the card and
//! only the low-rank factors differentiated. Which one runs is a caller
//! decision ([`TrainOpts::trainer`], `brain flux2 finetune --trainer`), never
//! an implicit fallback: a run that silently used the slow path would look
//! like a hang, and one that silently used the fast path would hide a missing
//! GPU.

use std::path::Path;

use crate::config::Flux2Config;
use crate::devtrain::DeviceTrainer;
use crate::lora::{save_adapter, LoraAdapter};
use crate::modelgrad::{self, make_flow_batch_paired, Batch, Cfg, ModelWeights};
use crate::pipeline::{Paths, PAD_TOKEN, TAP_LAYERS};
use model::adapter::TargetHp;
use data::qwen_tokenizer::QwenBpe;
use data::Tokenizer;

/// Build the training [`Cfg`] for a checkpoint at latent grid `lh×lw`
/// (latent tokens = image pixels / 16), caption-only.
pub fn train_cfg(fc: &Flux2Config, lh: usize, lw: usize) -> Cfg {
    Cfg::from_flux2(fc, lh, lw)
}

/// A dataset sample after encoding: clean packed latent tokens `x₀` for the
/// TARGET image (`[n_gen·in_channels]`), the reference image's tokens if the
/// dataset paired one with it (`[n_ref·in_channels]`, empty otherwise), and
/// the caption conditioning (`[txt_len·context_in_dim]`).
#[derive(Clone)]
pub struct Encoded {
    pub x0: Vec<f32>,
    /// The reference image's packed latent tokens, built by the same
    /// [`crate::refcond::pack_tokens`] the `--ref` generation path uses.
    /// Empty for a caption-only sample.
    pub refs: Vec<f32>,
    pub ctx: Vec<f32>,
}

/// The σ values a run at `size` pixels will actually be sampled at when the
/// adapter is deployed: this variant's own inference schedule
/// (`diffusion::scheduler::klein_sigmas`), minus its terminal 0.
///
/// **Why a schedule and not `U(0,1)`.** klein is a step-distilled sampler with
/// a FIXED step count ([`crate::pipeline::resolved_steps`]), so a generation
/// evaluates the DiT at a handful of discrete σ and nowhere else - at 512 px,
/// four of them, all above 0.71. Drawing σ uniformly spent most of every
/// training step optimising a regime the deployed sampler never enters, and
/// under-sampled the band it does. The schedule is resolution-dependent
/// (`empirical_mu` shifts with the token count), so this takes the size the
/// run is configured for rather than a constant.
///
/// The terminal 0 is dropped because it is not a model input: it exists to
/// close the last Euler interval, and `x_σ` at σ = 0 carries no noise for the
/// network to have an opinion about.
pub fn training_sigmas(fc: &Flux2Config, size: u32) -> Vec<f32> {
    let n_gen = ((size / 16) * (size / 16)) as usize;
    let steps = crate::pipeline::resolved_steps(&crate::pipeline::GenOpts::default(), fc.distilled) as usize;
    let mut s = diffusion::scheduler::klein_sigmas(steps, n_gen);
    s.pop();
    s
}

/// Pick one σ out of [`training_sigmas`] from a uniform draw `u ∈ [0,1)`.
///
/// Uniform over the schedule's entries: every σ generation visits is trained
/// at, none more than another. A schedule entry is used verbatim, never
/// jittered - the point is that the training σ IS an inference σ.
pub fn draw_sigma(sched: &[f32], u: f64) -> f64 {
    debug_assert!(!sched.is_empty(), "the schedule always has at least one entry");
    let i = ((u * sched.len() as f64) as usize).min(sched.len() - 1);
    sched[i] as f64
}

/// Which sample a run trains on at `step`, over `n` samples.
///
/// **Random without replacement, per pass.** The old `encoded[step % n]`
/// confounded sample identity with epoch for a whole run: sample `i` was only
/// ever seen at steps `i`, `i+n`, `i+2n`, always in the same order and always
/// at the same phase of every other per-step stream. This keeps the property
/// that made the cycle attractive - each sample is seen exactly once per `n`
/// steps, so none starves and none is over-weighted - and drops the fixed
/// order: each pass is its own permutation, derived from the run's seed and
/// the epoch index.
///
/// Derived rather than stateful, so a resumed run replays exactly the order it
/// would have walked had it never stopped.
pub fn sample_index(n: usize, step: u64, seed: u64) -> usize {
    assert!(n > 0, "a dataset with no samples cannot be trained on");
    let epoch = step / n as u64;
    let k = (step % n as u64) as usize;
    // Fisher-Yates over 0..n, seeded by (run seed, epoch). Only the k-th
    // element is needed, but the shuffle is O(n) on a dataset of a few dozen
    // images and stating it plainly is worth more than the saving.
    let mut rng = data::rng::Rng::new(seed ^ 0x5a4f_1e00 ^ epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_f64() * (i + 1) as f64) as usize;
        perm.swap(i, j.min(i));
    }
    perm[k]
}

/// The text-encoder placement a fine-tune should build for a DiT at `path`,
/// given the precision the adapter will be **deployed** at.
///
/// Resolved exactly as `brain flux2 generate` resolves it
/// ([`crate::pipeline::effective_dit_precision`] then
/// [`crate::pipeline::te_tier_int8`]), so a `.gguf` DiT - which generation
/// always runs int8 - trains against the int8 encoder without the caller
/// having to know that. Training used to hard-code the full f32 encoder
/// regardless, which meant the context vectors an adapter was fitted against
/// were not the ones its deployment produces.
pub fn text_encoder_placement(dit: &str, requested: crate::Precision) -> Result<crate::pipeline::TePlacement, String> {
    let effective = crate::pipeline::effective_dit_precision(dit, requested, false)?;
    Ok(crate::pipeline::TePlacement::here_for(effective))
}

/// Encode every dataset sample once: caption → Qwen 3-tap features (the
/// masked-pad path generation uses), image → packed+normalized latent tokens.
/// Both encoders are built, used, and **dropped** before the caller builds the
/// trainer (sequential residency). `size` is the square image size in pixels
/// (must be a multiple of 16); `progress(done, total, stage)` streams per-item
/// progress. `cancel` is polled per item so a cancelled job aborts during this
/// phase too.
pub fn encode_samples(
    fc: &Flux2Config,
    paths: &Paths,
    samples: &[data::imageset::Sample],
    size: u32,
    te: crate::pipeline::TePlacement,
    cancel: &capability::CancelToken,
    mut progress: impl FnMut(usize, usize, &str),
) -> Result<Vec<Encoded>, String> {
    if !size.is_multiple_of(16) {
        return Err("size must be a multiple of 16".into());
    }
    let n = samples.len();
    let tok = QwenBpe::from_file(&paths.tokenizer)?;

    // --- captions → Qwen taps (layers 9/18/27 concatenated per token) ---
    // The SAME encoder generation builds (`pipeline::build_text_encoder`),
    // built here directly because `Pipeline::encode_prompt` needs the whole
    // built Pipeline, DiT included, which finetune must NOT keep resident
    // while training. Conditioning an adapter on features the generation path
    // would not reproduce is the failure this shares code to avoid - and the
    // copy that used to live here also slurped the whole encoder as an fp32
    // `HashMap` before uploading it, the largest single host allocation the
    // run made.
    // Keyed by the exact prompt STRING, not by sample index: the encoder's
    // output depends on nothing else, so two samples that happen to share a
    // caption (a fixed-prompt/single-concept dataset does this for every
    // sample) get the encode done once and the identical result reused. A
    // dataset where every caption differs pays exactly what it paid before -
    // this is a cache miss on first sight of each string, never a behavior
    // change on what gets encoded.
    let ctxs: Vec<Vec<f32>> = {
        let te = crate::pipeline::build_text_encoder_on(fc, paths, te)?;
        let mut cache: std::collections::HashMap<&str, Vec<f32>> = std::collections::HashMap::new();
        let mut out = Vec::with_capacity(n);
        for (i, s) in samples.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err("cancelled".into());
            }
            if let Some(ctx) = cache.get(s.prompt.as_str()) {
                out.push(ctx.clone());
                continue;
            }
            progress(i, n, "encoding captions (Qwen)");
            let templated = tok.apply_chat_template_no_think(&[("user", s.prompt.as_str())]);
            let mut ids = tok.encode(&templated);
            ids.truncate(fc.txt_len);
            let content = ids.len();
            ids.resize(fc.txt_len, PAD_TOKEN);
            let taps = te.encode_hiddens_padded(&ids, content, &TAP_LAYERS);
            let d = taps[0].len() / fc.txt_len;
            let mut ctx = Vec::with_capacity(fc.txt_len * 3 * d);
            for row in 0..fc.txt_len {
                for tap in &taps {
                    ctx.extend_from_slice(&tap[row * d..(row + 1) * d]);
                }
            }
            cache.insert(s.prompt.as_str(), ctx.clone());
            out.push(ctx);
        }
        out
    }; // text encoder dropped here

    // --- images → packed latent tokens (FLUX.2 VAE + pixel-unshuffle pack) ---
    let vp = Path::new(&paths.vae);
    let (vae_file, vae_json) = if vp.is_dir() {
        (vp.join("diffusion_pytorch_model.safetensors"), std::fs::read_to_string(vp.join("config.json")).ok())
    } else {
        (vp.to_path_buf(), None)
    };
    let vae_cfg = match vae_json {
        Some(j) => vae::VaeConfig::from_json(&serde_json::from_str(&j).map_err(|e| e.to_string())?),
        None => vae::VaeConfig::flux2(),
    };
    let vae_ts = checkpoint::safetensors::read(vae_file.to_str().unwrap())?;
    let mut map = std::collections::HashMap::new();
    let (mut bn_mean, mut bn_var) = (Vec::new(), Vec::new());
    for t in vae_ts {
        if t.name == "bn.running_mean" {
            bn_mean = t.data.clone();
        }
        if t.name == "bn.running_var" {
            bn_var = t.data.clone();
        }
        map.insert(t.name, (t.shape, t.data));
    }
    if bn_mean.is_empty() || bn_var.is_empty() {
        return Err("vae checkpoint missing bn.running_{mean,var}".into());
    }
    let enc = vae::VaeEncoder::from_diffusers(vae_cfg.clone(), &map, size, size, None);
    let (lh8, lw8) = ((size / 8) as usize, (size / 8) as usize);
    let mut encoded = Vec::with_capacity(n);
    // One image → packed DiT tokens. The pixel conversion is
    // `pipeline::ref_from_hwc` (HWC `[0,1]` → CHW `[-1,1]`, the layout
    // `Pipeline::generate` takes a `--ref` in) and the latent packing is
    // `refcond::pack_tokens` - both shared with generation, so a training
    // image and a `--ref` photograph of the same pixels become the same
    // tokens.
    let tokens_of = |hwc: &[f32]| -> Result<Vec<f32>, String> {
        let (chw, h, w) = crate::pipeline::ref_from_hwc(hwc, size, size)?;
        let mean = enc.encode_mean(&chw, (h / 8) as u32, (w / 8) as u32);
        Ok(crate::refcond::pack_tokens(&mean, lh8, lw8, &bn_mean, &bn_var, vae_cfg.batch_norm_eps, fc.in_channels))
    };
    for (i, (s, ctx)) in samples.iter().zip(ctxs).enumerate() {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        progress(i, n, "encoding images (VAE)");
        let x0 = tokens_of(&s.hwc)?;
        let refs = match &s.reference {
            Some(r) => tokens_of(r)?,
            None => Vec::new(),
        };
        encoded.push(Encoded { x0, refs, ctx });
    }
    Ok(encoded)
}

/// The reference grids a dataset trains under, or an error naming the samples
/// that disagree.
///
/// A run builds ONE trainer sized for ONE joint sequence, so every sample has
/// to present the same reference layout. A folder that pairs some of its
/// targets and not others is a dataset mistake, not a mode: training the
/// unpaired ones with a blank reference would teach "no photograph here", and
/// silently dropping them would train on a subset the operator did not choose.
/// Both are worse than saying so.
pub fn dataset_refs(samples: &[data::imageset::Sample], lh: usize, lw: usize) -> Result<Vec<(usize, usize)>, String> {
    let paired = samples.iter().filter(|s| s.reference.is_some()).count();
    if paired == 0 {
        return Ok(Vec::new());
    }
    if paired != samples.len() {
        let missing: Vec<String> = samples
            .iter()
            .filter(|s| s.reference.is_none())
            .take(5)
            .map(|s| s.path.file_name().unwrap_or(s.path.as_os_str()).to_string_lossy().into_owned())
            .collect();
        return Err(format!(
            "{paired} of {} samples are paired: a run trains one joint sequence layout, so pairs.yaml must cover every captioned image or none. Unpaired: {} ...",
            samples.len(),
            missing.join(", ")
        ));
    }
    // Every image in this loader is square at `size`, target and reference
    // alike, so one grid covers them all.
    Ok(vec![(lh, lw)])
}

/// Which of the two gradient implementations a run uses. They compute the same
/// thing; `tests/device_train.rs` is what says so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trainer {
    /// The FD-gradchecked host reference ([`crate::modelgrad`]) - the oracle.
    Host,
    /// The WGSL device trainer ([`crate::devtrain`]) - frozen base resident on
    /// the card, only the adapter differentiated.
    Device,
}

impl Trainer {
    /// Parse the CLI spelling (`host` | `device`).
    pub fn from_name(v: &str) -> Result<Trainer, String> {
        match v {
            "host" => Ok(Trainer::Host),
            "device" | "gpu" => Ok(Trainer::Device),
            other => Err(format!("unknown trainer {other} (host|device)")),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Trainer::Host => "host",
            Trainer::Device => "device",
        }
    }
}

/// LoRA fine-tuning hyper-parameters.
pub struct TrainOpts {
    pub steps: u32,
    pub rank: usize,
    pub lr: f32,
    /// Host reference or device (WGSL) gradients - see [`Trainer`].
    pub trainer: Trainer,
    /// How many GPUs the device trainer spreads the block stack over. One card
    /// holds klein-4B's fp32 frozen base; klein-9B's is larger than a 24 GiB
    /// card and needs two. Never auto-grabbed: taking a second card is a
    /// decision about a shared machine, so a caller has to make it.
    pub cards: usize,
    /// Square training image size in pixels (multiple of 16; latent grid =
    /// size/16 per side).
    pub size: u32,
    pub seed: u64,
    /// Where to write the adapter (final + every `ckpt_every` steps; 0 = only final).
    pub save_path: String,
    pub ckpt_every: u32,
    /// Continue from the adapter already at `save_path`, if one is there.
    ///
    /// A multi-hour run that is cancelled, or whose box is rebooted, should
    /// cost the time since the last checkpoint and not the whole run. With
    /// this set the SAME command re-run picks up where it stopped; with no
    /// file at `save_path` it starts fresh, so one invocation is correct
    /// whether or not it is the first. It is a flag rather than automatic
    /// because silently continuing from whatever happens to be lying at the
    /// output path is how a run inherits an unrelated adapter.
    pub resume: bool,
    /// rsLoRA: scale `alpha/sqrt(rank)` instead of `alpha/rank`, so the
    /// update is not increasingly suppressed as rank grows.
    pub rank_stabilized: bool,
    /// LoRA+: `B`'s effective learning rate is `lr_ratio * lr`. `1.0` (the
    /// default) is plain LoRA.
    pub lr_ratio: f32,
    /// LoRA-FA: freeze `A` at its random init: only `B` trains.
    pub freeze_a: bool,
    /// The DiT precision the trained adapter will be **deployed** at.
    ///
    /// It does not change what is differentiated - the frozen base is fp32 on
    /// both trainers either way - it selects the TEXT ENCODER tier the
    /// captions are embedded through ([`text_encoder_placement`]), which is
    /// the input the adapter is actually keyed on. `generate` picks the
    /// encoder's tier from the DiT's, so training has to resolve it the same
    /// way or the adapter is fitted against conditioning vectors its
    /// deployment does not produce.
    pub precision: crate::Precision,
}

/// Fine-tune a LoRA adapter on `dir` (a captioned-image folder — see
/// `data::imageset` for the caption formats). Returns the trained adapter.
/// `progress(step, total, msg)` streams encoding + per-step loss and step time
/// so a long run is not a black box. `cancel` is polled every step (a
/// multi-hour job must be abortable): a cancelled token returns
/// `Err("cancelled")` — periodic checkpoints already written remain.
pub fn run(
    fc: &Flux2Config,
    paths: &Paths,
    dir: &Path,
    opts: &TrainOpts,
    cancel: &capability::CancelToken,
    mut progress: impl FnMut(u32, u32, String),
) -> Result<LoraAdapter, String> {
    // 1. dataset
    let samples = data::imageset::load_dir(dir, opts.size, |w| progress(0, opts.steps + 1, format!("dataset: {w}")))?;
    let (lh, lw) = ((opts.size / 16) as usize, (opts.size / 16) as usize);
    let refs = dataset_refs(&samples, lh, lw)?;
    progress(
        0,
        opts.steps + 1,
        format!(
            "loaded {} images from {} ({})",
            samples.len(),
            dir.display(),
            if refs.is_empty() { "caption-only".to_string() } else { format!("paired: each target conditioned on its pairs.yaml reference, +{} tokens", lh * lw) }
        ),
    );

    // 2. encode (encoders dropped inside before returning). The text encoder
    //    is built at the tier `generate` would use for this DiT, so the
    //    conditioning the adapter is fitted against is the conditioning its
    //    deployment produces.
    let n_samples = samples.len();
    let te_place = text_encoder_placement(&paths.dit, opts.precision)?;
    progress(
        0,
        opts.steps + 1,
        format!("text encoder: {} (matching what generate builds for this DiT)", if te_place.int8 { "int8" } else { "fp32" }),
    );
    let encoded = encode_samples(fc, paths, &samples, opts.size, te_place, cancel, |i, tot, stage| {
        progress(0, opts.steps + 1, format!("{stage} {}/{tot}", i + 1))
    })?;
    drop(samples);

    // 3. frozen base → host training weights (fused checkpoint split)
    progress(0, opts.steps + 1, "loading DiT weights".into());
    let cfg = Cfg::from_flux2_with_refs(fc, lh, lw, refs);
    // `from_tensors` REMOVES as it converts, so the fused map shrinks while the
    // split one grows: the peak is one copy of the model, not two. At klein-9B
    // that is the difference between fitting this box and not.
    let mut tensors = crate::pipeline::read_dit_tensors(&paths.dit, fc)?;
    let base = ModelWeights::from_tensors(&cfg, &mut tensors)?;
    drop(tensors);

    // 4. the chosen gradient implementation. The device path uploads the frozen
    //    base to the card and then releases the host copy, so the two never
    //    hold the whole model twice.
    let mut dev = None;
    let mut host = Some(base);
    if opts.trainer == Trainer::Host {
        // Say the bill before running it up. A host step differentiates a
        // DENSE `W_eff`, so it holds the frozen base AND its effective copy
        // for the whole forward+backward; a `--trainer host` run at 9B scale
        // is a multi-tens-of-GB resident job that reaches its first step
        // slowly, and finding that out from the OOM killer is the worst way to
        // find it out. Derived from the config, so it cannot go stale.
        let one = host.as_ref().expect("base").param_bytes() as f64 / (1u64 << 30) as f64;
        progress(
            0,
            opts.steps + 1,
            format!(
                "host trainer: fp32 weights {one:.2} GiB; a step holds the frozen base and its effective copy, so expect at least {:.2} GiB resident plus activations",
                2.0 * one
            ),
        );
    }
    if opts.trainer == Trainer::Device {
        progress(0, opts.steps + 1, "uploading the frozen base to the device".into());
        let t = DeviceTrainer::new_multi(opts.cards.max(1), cfg.clone(), opts.rank, host.as_ref().expect("base"));
        // The QK-RMSNorm scales are frozen in a LoRA run, so their gain
        // gradient is work nothing consumes. It stays on under the parity
        // gate, which is what proves turning it off changes no adapter
        // gradient.
        t.set_qk_grads(false);
        let per: Vec<String> = t.weight_bytes_per_card().iter().map(|b| format!("{:.2} GiB", *b as f64 / (1u64 << 30) as f64)).collect();
        progress(0, opts.steps + 1, format!("device base resident on {} card(s): {}", t.cards(), per.join(" + ")));
        dev = Some(t);
        host = None;
    }

    // 5. adapter + rectified-flow loop
    // 5a. the adapter: fresh, or continued from the last checkpoint.
    let resume_path = Path::new(&opts.save_path);
    let (mut adapter, done) = if opts.resume && resume_path.exists() {
        let ad = crate::lora::load_adapter(&opts.save_path, &cfg)?;
        if ad.rank() != opts.rank {
            return Err(format!(
                "{}: adapter is rank {}, this run asks for rank {} - resuming would silently train the file's rank",
                opts.save_path,
                ad.rank(),
                opts.rank
            ));
        }
        let saved_hp = ad.hp();
        if saved_hp.rank_stabilized != opts.rank_stabilized || saved_hp.lr_ratio != opts.lr_ratio || saved_hp.freeze_a != opts.freeze_a {
            return Err(format!(
                "{}: adapter was trained with rs={} lr_ratio={} freeze_a={}, this run asks for rs={} lr_ratio={} freeze_a={} - resuming would silently change the training method mid-run",
                opts.save_path, saved_hp.rank_stabilized, saved_hp.lr_ratio, saved_hp.freeze_a, opts.rank_stabilized, opts.lr_ratio, opts.freeze_a
            ));
        }
        let done = ad.steps_done() as u32;
        if done >= opts.steps {
            return Err(format!(
                "{}: already {done} steps, --steps is {} - nothing left to do",
                opts.save_path, opts.steps
            ));
        }
        progress(
            done,
            opts.steps + 1,
            format!("resuming from {} at step {done} (Adam moments restart; weights continue)", opts.save_path),
        );
        (ad, done)
    } else {
        if opts.resume {
            progress(0, opts.steps + 1, format!("--resume: nothing at {}, starting fresh", opts.save_path));
        }
        let hp = TargetHp {
            rank: opts.rank,
            alpha: opts.rank as f32,
            rank_stabilized: opts.rank_stabilized,
            dropout: 0.0,
            lr_ratio: opts.lr_ratio,
            freeze_a: opts.freeze_a,
        };
        (LoraAdapter::new_with_hp(&cfg, hp, opts.seed), 0)
    };
    let mut rng = data::rng::Rng::new(opts.seed ^ 0x5eed_f10c);
    // Advance the sigma stream past the steps already taken. `sigma` is one
    // draw per step, so a resumed run that restarted this at zero would
    // replay the first steps' sigmas against a different sample phase - the
    // schedule is part of the run, not a per-step detail.
    for _ in 0..done {
        let _ = rng.next_f64();
    }
    // The σ band this run's deployment actually samples at - not U(0,1).
    let sched = training_sigmas(fc, opts.size);
    progress(
        done,
        opts.steps + 1,
        format!(
            "sigma schedule ({} steps at {}px): {}",
            sched.len(),
            opts.size,
            sched.iter().map(|s| format!("{s:.4}")).collect::<Vec<_>>().join(", ")
        ),
    );
    for step in done..opts.steps {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let t0 = std::time::Instant::now();
        let s = &encoded[sample_index(n_samples, step as u64, opts.seed)];
        let sigma = draw_sigma(&sched, rng.next_f64());
        let noise = model::hostmath::randn(s.x0.len(), opts.seed ^ (0xa5a5 + step as u64));
        let batch: Batch<f32> = make_flow_batch_paired(&cfg, &s.x0, &s.refs, &s.ctx, sigma, &noise);
        let loss = match (&dev, &host) {
            (Some(t), _) => t.step(&mut adapter, &batch, opts.lr),
            (None, Some(b)) => {
                let w_eff = adapter.apply(b);
                // Streamed, not collected. A LoRA step reduces each block's
                // dense `dW` to its rank-r projection and is done with it, so
                // the whole-model `ModelGrads` `grads` returns is a third fp32
                // copy of the model held for no reason - at klein-9B that is
                // the difference between a training step that fits this box
                // beside the frozen base and its effective copy, and one that
                // does not. `grads_into` runs the identical backward and hands
                // each block over as it completes.
                let mut step = adapter.stepper(opts.lr);
                let (loss, _globals) = modelgrad::grads_into(&cfg, &w_eff, &batch, &mut step);
                loss
            }
            (None, None) => unreachable!("one of the two trainers is always built"),
        };
        progress(
            step + 1,
            opts.steps + 1,
            format!("step {}/{}  loss {loss:.5}  ({:.1} s)", step + 1, opts.steps, t0.elapsed().as_secs_f64()),
        );
        // periodic checkpoint so a long run is resumable / inspectable mid-flight
        if opts.ckpt_every > 0 && (step + 1) % opts.ckpt_every == 0 && step + 1 < opts.steps {
            save_adapter(&opts.save_path, &adapter);
        }
    }
    save_adapter(&opts.save_path, &adapter);
    progress(opts.steps + 1, opts.steps + 1, format!("saved adapter → {}", opts.save_path));
    Ok(adapter)
}
