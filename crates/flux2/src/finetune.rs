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

/// The σ a run trains `step` at: **systematic (stratified) sampling** over
/// [`training_sigmas`] - each block of `sched.len()` consecutive steps is a
/// fresh random permutation of the whole schedule, so every σ is visited
/// exactly once per block.
///
/// Uniform over the schedule's entries, as the i.i.d. draw this replaced was:
/// every σ generation visits is trained at, none more than another, and a
/// schedule entry is used verbatim rather than jittered - the point is that
/// the training σ IS an inference σ. What changes is only the variance.
///
/// **Why not an i.i.d. draw.** klein at 512 px evaluates four σ, and the loss
/// at the largest of them is close to twice the loss at the smallest (the
/// velocity target `ε − x₀` is far harder to predict when the input still
/// carries some `x₀` to separate out). With one sample per step, the σ mix
/// inside any short window is therefore the single biggest term in what a
/// window of reported losses - or a window of gradients - actually averages
/// to. Drawn i.i.d. that mix is binomial and swings by ±1 count in every
/// window, including the ~10-step window Adam's β₁ = 0.9 momentum is an
/// average over; drawn systematically it is exactly uniform in every aligned
/// block. Same marginal distribution, same "training σ IS an inference σ"
/// property, strictly less variance - textbook stratified sampling, applied to
/// strata the deployed sampler handed us.
///
/// The order **within** a block is shuffled from `(seed, block)` rather than
/// fixed, for the reason [`sample_index`] drops `step % n`: a fixed cycle locks
/// each σ to a fixed phase of every other per-step stream for a whole run.
///
/// Derived, not stateful - so a resumed run replays exactly the σ sequence it
/// would have walked had it never stopped.
pub fn step_sigma(sched: &[f32], step: u64, seed: u64) -> f64 {
    assert!(!sched.is_empty(), "the schedule always has at least one entry");
    let k = sched.len() as u64;
    let block = step / k;
    let i = (step % k) as usize;
    let mut rng = data::rng::Rng::new(seed ^ 0x5169_0a00 ^ block.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let mut perm: Vec<usize> = (0..sched.len()).collect();
    for a in (1..perm.len()).rev() {
        let b = (rng.next_f64() * (a + 1) as f64) as usize;
        perm.swap(a, b.min(a));
    }
    sched[perm[i]] as f64
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

/// Does `step` train with its reference **blanked**? A pure function of the
/// global step, so a resumed run replays the same sequence of dropped steps -
/// the property [`sample_index`] and [`step_sigma`] are written for.
///
/// **The failure this exists for.** A reference image reaches the DiT by being
/// concatenated into the same attention sequence as the noised target. When
/// the two images differ only locally - declutter a room, remove an object -
/// the flow-matching loss can be driven a long way down by an adapter that
/// learns to COPY the reference across and nothing else. That solution is
/// real, it is reachable, and its loss curve looks like training: it descends
/// and then flattens, and the flattening is not a capacity limit, it is the
/// copy running out of room. The deployed adapter then reproduces the
/// reference's own lighting, grain and colour instead of transforming them.
/// This is shortcut learning in the sense of arXiv:2510.20887 - a
/// reconstruction loss "indiscriminately incentiviz[es] the adapter to
/// reproduce all visual factors present in the image" rather than isolating
/// the one the task is about. OminiControl (arXiv:2411.15098) reports the same
/// thing from the data side: pairs too alike make a model "simply reproduce
/// the input".
///
/// Blanking the reference on a fraction of steps removes the copy source on
/// those steps, so the adapter has to represent what a finished target looks
/// like from the caption and its own prior - capacity it otherwise never has
/// to spend, and the half of the task the copy solution never learns.
///
/// **It is not free, and it is not what the field defaults to.** No mainstream
/// reference-conditioned trainer (diffusers' Kontext and FLUX.2 img2img
/// examples, ai-toolkit, SimpleTuner, kohya-ss, musubi-tuner) drops the
/// conditioning IMAGE; the one that does is InstructPix2Pix (arXiv:2211.09800
/// §3.2.1, 5%/5%/5% over text, image, and both), and it does so to train the
/// null branch its classifier-free guidance evaluates at inference. klein is
/// guidance-distilled - it runs one forward at guidance 1.0 and has no null
/// branch - so here the dropped steps buy regularisation only, and they spend
/// part of a step budget on a mode the sampler never enters. That is why this
/// is off by default and why the rate to reach for is small.
pub fn ref_dropped(step: u64, seed: u64, p: f32) -> bool {
    if p <= 0.0 {
        return false;
    }
    if p >= 1.0 {
        return true;
    }
    let mut rng = data::rng::Rng::new(seed ^ 0xd2_0b_0f_00 ^ step.wrapping_mul(0xa076_1d64_78bd_642f));
    (rng.next_f64() as f32) < p
}

/// Blank a batch's reference tokens in place - the `∅_I` of [`ref_dropped`].
///
/// The reference rows are ZEROED rather than removed: a run builds one trainer
/// sized for one joint sequence, and this is also exactly what the reference
/// implementation does (diffusers' `train_instruct_pix2pix.py` multiplies the
/// conditioning latents by a 0/1 mask, it does not shorten anything). The
/// position ids therefore still describe a reference at the same t-axis
/// offset; what is gone is its content.
///
/// The velocity target is untouched - it is a property of the sample, not of
/// what the model was shown.
pub fn blank_references<T: crate::grad::Fp>(b: &mut Batch<T>, cfg: &Cfg) {
    let n_gen = cfg.n_gen() * cfg.in_channels;
    for v in &mut b.img[n_gen..] {
        *v = T::ZERO;
    }
}

/// Per-element flow-loss weights that concentrate a PAIRED run's gradient on
/// the tokens its edit actually changes: `w_j = 1 + α·‖x₀_j − ref_j‖/max_j‖·‖`
/// over tokens, rescaled to mean 1 and broadcast across each token's channels.
/// `α = 0`, a reference of a different length than the target (more than one
/// reference, or a differently-sized grid), or a pair with no change at all all
/// return an EMPTY weight - the plain unweighted mean.
///
/// **What it is for.** In an edit pair - declutter a room, remove an object -
/// the target latent equals the reference latent almost everywhere, so almost
/// every token's velocity can be hit by copying the reference across. Averaged
/// uniformly, that easy sub-problem is most of the loss AND most of the
/// gradient, and a run whose loss has flattened may just have become a
/// competent copier: the tokens carrying the edit are a small enough fraction
/// that fitting them barely moves the mean. arXiv:2604.23763 §3.5 states this
/// defect for instruction editing and measures L1 0.2132 → 0.1483 from the
/// reweight alone, at `α = 2`.
///
/// **It requires a SPATIALLY ALIGNED pair**, which a declutter/relight/removal
/// dataset is and a subject-driven one is not: the mask is `|target −
/// reference|` at matching token positions, so if the two images are not
/// registered, every token "changed" and the weight is noise with no signal in
/// it. That is why this is opt-in rather than the default - the trainer cannot
/// tell the two kinds of dataset apart, but [`change_stats`] reports the number
/// the operator can.
///
/// **Rescaled to mean 1** so a weighted run's reported loss is on the same
/// scale as an unweighted one's. Without that the reweighting would show up as
/// a loss-level shift and be indistinguishable from training progress.
pub fn change_weights(x0: &[f32], refs: &[f32], cin: usize, alpha: f32) -> Vec<f32> {
    if alpha <= 0.0 || refs.len() != x0.len() || x0.is_empty() {
        return Vec::new();
    }
    let n_tok = x0.len() / cin;
    let mut m: Vec<f32> = (0..n_tok)
        .map(|j| {
            let (a, b) = (&x0[j * cin..(j + 1) * cin], &refs[j * cin..(j + 1) * cin]);
            a.iter().zip(b).map(|(&p, &q)| (p - q) * (p - q)).sum::<f32>().sqrt()
        })
        .collect();
    let peak = m.iter().copied().fold(0.0f32, f32::max);
    if peak <= 0.0 {
        // Target and reference are the same image. Nothing to weight toward,
        // and dividing by the peak would be a NaN in every gradient.
        return Vec::new();
    }
    let mut sum = 0.0f32;
    for v in &mut m {
        *v = 1.0 + alpha * (*v / peak);
        sum += *v;
    }
    let mean = sum / n_tok as f32;
    let mut w = Vec::with_capacity(x0.len());
    for v in m {
        let scaled = v / mean;
        for _ in 0..cin {
            w.push(scaled);
        }
    }
    w
}

/// What fraction of a pair's tokens carry a change of at least a tenth of the
/// largest one - the number that says whether a dataset is the spatially
/// aligned, small-region kind [`change_weights`] is for.
///
/// A declutter or removal set reads as a few percent: most of the picture is
/// identical and the edit is local. A set whose pairs are not registered to
/// each other reads as most of the tokens, and weighting toward "the changed
/// region" would then be weighting toward nothing. Reported rather than acted
/// on: the trainer states the measurement, the operator decides.
pub fn change_stats(x0: &[f32], refs: &[f32], cin: usize) -> Option<f32> {
    if refs.len() != x0.len() || x0.is_empty() {
        return None;
    }
    let n_tok = x0.len() / cin;
    let m: Vec<f32> = (0..n_tok)
        .map(|j| {
            let (a, b) = (&x0[j * cin..(j + 1) * cin], &refs[j * cin..(j + 1) * cin]);
            a.iter().zip(b).map(|(&p, &q)| (p - q) * (p - q)).sum::<f32>().sqrt()
        })
        .collect();
    let peak = m.iter().copied().fold(0.0f32, f32::max);
    if peak <= 0.0 {
        return Some(0.0);
    }
    Some(m.iter().filter(|&&v| v >= 0.1 * peak).count() as f32 / n_tok as f32)
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
        let mean = enc.encode_mean(&chw, h / 8, w / 8);
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
    /// The **peak** learning rate - what the warmup ramps to and the decay
    /// starts from, not the rate every step runs at. See [`Self::lr_schedule`].
    pub lr: f32,
    /// Steps of linear LR warmup. `None` selects the default,
    /// [`WARMUP_FRACTION`] of `steps`.
    pub warmup: Option<u32>,
    /// The rate the decay lands on at `steps`. `None` selects the default,
    /// [`MIN_LR_FRACTION`] of `lr`; `Some(lr)` with `warmup: Some(0)` is a
    /// constant rate.
    pub min_lr: Option<f32>,
    /// Region-aware flow loss for a PAIRED run: `α` in
    /// [`change_weights`]'s `1 + α·(normalised per-token change)`. `0.0` (the
    /// default) is the plain unweighted mean; `2.0` is the published value.
    /// Ignored by a caption-only run, which has no reference to difference
    /// against.
    pub edit_weight: f32,
    /// Probability that a step trains with its reference tokens **blanked** -
    /// see [`ref_dropped`] for the copy-shortcut failure this exists for, and
    /// for why it is `0.0` by default on a guidance-distilled variant.
    pub ref_dropout: f32,
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

/// Default cooldown, as a fraction of the run's step budget: the rate is held
/// at the peak until the last fifth, then cosine-decays to the floor.
///
/// A fifth is Hägele et al.'s measured saturation point (arXiv:2405.18392):
/// constant-then-cooldown tracks a full cosine, and lengthening the cooldown
/// past ~20% of the run stops buying anything. The shape matters more than the
/// number here - every step before the cooldown runs at the rate the caller
/// asked for, whatever `--steps` says, so a run that turns out to need three
/// times the budget has not already annealed itself against the short one.
pub const COOLDOWN_FRACTION: f32 = 0.2;

/// Default floor, as a fraction of the peak rate. A tenth: low enough that the
/// last steps are refining rather than exploring, above zero so the run does
/// not stop training some steps before it ends.
pub const MIN_LR_FRACTION: f32 = 0.1;

impl TrainOpts {
    /// The learning-rate curve this run follows: [`Self::lr`] held flat, then
    /// a cosine cooldown to the floor over the last [`COOLDOWN_FRACTION`] of
    /// the run. No warmup by default.
    ///
    /// **Why it cools down.** Adam on a stochastic objective does not converge
    /// to a minimum at a constant step size; it converges to a ball around one
    /// whose radius is set by the step size times the gradient noise. At batch
    /// size 1 - one image, one σ, one noise draw per step - that noise is
    /// large, so the ball is wide, and a run that never drops its rate stays
    /// in it. The cooldown is what closes it, and it is the shape the field
    /// has measured against a full cosine (arXiv:2405.18392) rather than a
    /// curve picked for its looks.
    ///
    /// **Why there is no warmup by default.** Warmup exists to let a net
    /// survive a rate above its instability threshold early on; Kalra &
    /// Barkeshli (arXiv:2406.09405) find it unnecessary below that threshold.
    /// A LoRA is initialised at `B = 0`, so its branch output - and the
    /// gradient through it - starts at exactly zero and the early sharpness
    /// pathology warmup addresses is largely absent. Every mainstream FLUX
    /// LoRA trainer (diffusers, kohya-ss, ai-toolkit) defaults to no warmup
    /// for the same reason. `--warmup` is there for a caller who measures
    /// otherwise.
    ///
    /// The curve is a function of the GLOBAL step, so `--resume` continues it
    /// instead of starting over - the same property [`sample_index`] and
    /// [`step_sigma`] are written for.
    pub fn lr_schedule(&self) -> model::LrSchedule {
        let warmup = self.warmup.unwrap_or(0);
        let cooldown = ((self.steps as f32 * COOLDOWN_FRACTION) as u32).max(1);
        model::LrSchedule {
            peak: self.lr,
            floor: self.min_lr.unwrap_or(self.lr * MIN_LR_FRACTION),
            warmup,
            hold: self.steps.saturating_sub(warmup + cooldown),
            decay_iters: self.steps,
        }
    }
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
    // How many times the run will see each image. At batch size 1 a step IS a
    // sample, so this is the number that decides whether a run is long enough,
    // and it is not the number the operator typed. Said out loud because the
    // published recipes for reference-conditioned FLUX training - diffusers'
    // own FLUX.2 img2img and FLUX.1-Kontext examples, ai-toolkit's Kontext
    // config, In-Context LoRA, BFL's own klein guidance - all sit between 1000
    // and 20000 sample-gradients, and a run an order of magnitude under that
    // has a flat loss curve because it has barely started, not because it has
    // converged.
    if !samples.is_empty() {
        progress(
            0,
            opts.steps + 1,
            format!(
                "{} steps over {} samples = {:.1} passes over the dataset{}",
                opts.steps,
                samples.len(),
                opts.steps as f32 / samples.len() as f32,
                if opts.steps < 1000 {
                    " (the published recipes for reference-conditioned FLUX training use 1000-3000 steps and up)"
                } else {
                    ""
                }
            ),
        );
    }

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

    // How local this dataset's edit is: the share of target tokens that differ
    // measurably from their reference, averaged over the pairs. Reported for
    // every paired run, whether or not `--edit-weight` is on, because it is
    // what says which of the two things a flat loss curve means. A few percent
    // says the edit is a small region of an otherwise-copied picture, so most
    // of a uniform loss - and most of the gradient - is the copy; a large
    // share says the pairs are not spatially registered and the region-aware
    // weighting has nothing to aim at.
    let local: Vec<f32> = encoded.iter().filter_map(|e| change_stats(&e.x0, &e.refs, fc.in_channels)).collect();
    if !local.is_empty() {
        let mean = local.iter().sum::<f32>() / local.len() as f32;
        let mut note = String::new();
        if opts.edit_weight > 0.0 {
            note.push_str(&format!("; --edit-weight {} concentrates the loss there", opts.edit_weight));
        }
        if opts.ref_dropout > 0.0 {
            note.push_str(&format!("; --ref-dropout {} blanks the reference on some steps", opts.ref_dropout));
        }
        // The configuration that produces a copy, named while it can still be
        // changed. A target that is mostly its own reference, weighted
        // uniformly, with the reference always present, is a task an adapter
        // can solve by copying - and the loss curve of that solution looks
        // like training working.
        if mean < 0.3 && opts.edit_weight == 0.0 && opts.ref_dropout == 0.0 {
            note.push_str(
                "; with a uniform loss and the reference on every step, copying it is a low-loss solution here \
                 - see --edit-weight and --ref-dropout",
            );
        }
        progress(
            0,
            opts.steps + 1,
            format!(
                "paired dataset: {:.1}% of target tokens differ from their reference (mean over {} pairs){note}",
                100.0 * mean,
                local.len()
            ),
        );
    }

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
    // The σ band this run's deployment actually samples at - not U(0,1) - and
    // the rate curve it walks. Both are pure functions of the global step
    // ([`step_sigma`], [`model::LrSchedule::at`]), so neither needs advancing
    // past a resume's `done`: they replay what an uninterrupted run would have
    // walked.
    let sched = training_sigmas(fc, opts.size);
    let lr = opts.lr_schedule();
    progress(
        done,
        opts.steps + 1,
        format!(
            "sigma schedule ({} steps at {}px, stratified - each block of {} steps covers all of it): {}",
            sched.len(),
            opts.size,
            sched.len(),
            sched.iter().map(|s| format!("{s:.4}")).collect::<Vec<_>>().join(", ")
        ),
    );
    progress(
        done,
        opts.steps + 1,
        format!(
            "learning rate: {:.3e}{}, held to step {}, then cosine-cooled to {:.3e} at step {}",
            lr.peak,
            if lr.warmup > 0 { format!(" after {} warmup steps", lr.warmup) } else { String::new() },
            lr.decay_start(),
            lr.floor,
            lr.decay_iters
        ),
    );
    for step in done..opts.steps {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let t0 = std::time::Instant::now();
        let s = &encoded[sample_index(n_samples, step as u64, opts.seed)];
        let sigma = step_sigma(&sched, step as u64, opts.seed);
        let lr_now = lr.at(step);
        let noise = model::hostmath::randn(s.x0.len(), opts.seed ^ (0xa5a5 + step as u64));
        let mut batch: Batch<f32> = make_flow_batch_paired(&cfg, &s.x0, &s.refs, &s.ctx, sigma, &noise);
        let dropped = !s.refs.is_empty() && ref_dropped(step as u64, opts.seed, opts.ref_dropout);
        if dropped {
            blank_references(&mut batch, &cfg);
        }
        // Region-aware weights, if this run asked for them. Per step rather
        // than cached per sample: it is one pass over the target latent
        // against a forward+backward through the whole DiT.
        //
        // NOT on a dropped step: the weights say "spend the gradient where
        // this pair's edit is", and a step with no reference has no edit to
        // speak of - it is being asked to produce the whole target, so every
        // token of it counts the same.
        batch.w = if dropped {
            Vec::new()
        } else {
            change_weights(&s.x0, &s.refs, cfg.in_channels, opts.edit_weight)
        };
        let loss = match (&dev, &host) {
            (Some(t), _) => t.step(&mut adapter, &batch, lr_now),
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
                let mut step = adapter.stepper(lr_now);
                let (loss, _globals) = modelgrad::grads_into(&cfg, &w_eff, &batch, &mut step);
                loss
            }
            (None, None) => unreachable!("one of the two trainers is always built"),
        };
        // σ is on the line because a single step's loss is not comparable to
        // the next one's without it. The velocity target `ε − x₀` gets harder
        // to predict as σ falls and the input still carries some x₀ to
        // separate out, so ONE adapter evaluated at klein's four scheduled σ
        // produces four quite different losses - a spread that can rival what
        // a whole short run moves the mean by. A log carrying only the number
        // therefore invites reading a σ draw as progress. With σ on the line,
        // a run's log can be stratified after the fact, which at batch size 1
        // is the only way to see a trend rather than a draw.
        progress(
            step + 1,
            opts.steps + 1,
            format!(
                "step {}/{}  loss {loss:.5}  sigma {sigma:.4}  lr {lr_now:.3e}{}  ({:.1} s)",
                step + 1,
                opts.steps,
                // A dropped step's loss is a different quantity - no reference
                // to condition on - so it is marked, not silently averaged in
                // with the rest by whoever reads the log.
                if dropped { "  [ref dropped]" } else { "" },
                t0.elapsed().as_secs_f64()
            ),
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
