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
use crate::lora::{save_adapter, LoraAdapter, LoraCfg};
use crate::modelgrad::{self, make_flow_batch, Batch, Cfg, ModelWeights};
use crate::pipeline::{Paths, PAD_TOKEN, TAP_LAYERS};
use data::qwen_tokenizer::QwenBpe;
use data::Tokenizer;

/// Build the training [`Cfg`] for a checkpoint at latent grid `lh×lw`
/// (latent tokens = image pixels / 16).
pub fn train_cfg(fc: &Flux2Config, lh: usize, lw: usize) -> Cfg {
    Cfg::from_flux2(fc, lh, lw)
}

/// A dataset sample after encoding: clean packed latent tokens `x₀`
/// (`[n_img·in_channels]`) and caption conditioning
/// (`[txt_len·context_in_dim]`).
#[derive(Clone)]
pub struct Encoded {
    pub x0: Vec<f32>,
    pub ctx: Vec<f32>,
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
    let ctxs: Vec<Vec<f32>> = {
        let te = crate::pipeline::build_text_encoder(fc, paths)?;
        let mut out = Vec::with_capacity(n);
        for (i, s) in samples.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err("cancelled".into());
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
    let (lh, lw) = (lh8 / 2, lw8 / 2);
    let sz = size as usize;
    let mut encoded = Vec::with_capacity(n);
    for (i, (s, ctx)) in samples.iter().zip(ctxs).enumerate() {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        progress(i, n, "encoding images (VAE)");
        // HWC [0,1] → CHW [-1,1] (the VAE encoder's expected input range).
        let mut chw = vec![0f32; 3 * sz * sz];
        for c in 0..3 {
            for y in 0..sz {
                for x in 0..sz {
                    chw[(c * sz + y) * sz + x] = s.hwc[(y * sz + x) * 3 + c] * 2.0 - 1.0;
                }
            }
        }
        let mean = enc.encode_mean(&chw, lh8 as u32, lw8 as u32);
        let packed = vae::latent::pack(&mean, 32, lh8, lw8, &bn_mean, &bn_var, vae_cfg.batch_norm_eps);
        // [128, lh, lw] → tokens [lh·lw, 128] (row-major, matching position_ids)
        let mut x0 = vec![0.0f32; lh * lw * fc.in_channels];
        for c in 0..fc.in_channels {
            for y in 0..lh {
                for x in 0..lw {
                    x0[(y * lw + x) * fc.in_channels + c] = packed[(c * lh + y) * lw + x];
                }
            }
        }
        encoded.push(Encoded { x0, ctx });
    }
    Ok(encoded)
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
    progress(0, opts.steps + 1, format!("loaded {} images from {}", samples.len(), dir.display()));

    // 2. encode (encoders dropped inside before returning)
    let n_samples = samples.len();
    let encoded = encode_samples(fc, paths, &samples, opts.size, cancel, |i, tot, stage| {
        progress(0, opts.steps + 1, format!("{stage} {}/{tot}", i + 1))
    })?;
    drop(samples);

    // 3. frozen base → host training weights (fused checkpoint split)
    progress(0, opts.steps + 1, "loading DiT weights".into());
    let (lh, lw) = ((opts.size / 16) as usize, (opts.size / 16) as usize);
    let cfg = train_cfg(fc, lh, lw);
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
        (LoraAdapter::new(&cfg, LoraCfg { seed: opts.seed, ..LoraCfg::new(opts.rank) }), 0)
    };
    let mut rng = data::rng::Rng::new(opts.seed ^ 0x5eed_f10c);
    // Advance the sigma stream past the steps already taken. `sigma` is one
    // draw per step, so a resumed run that restarted this at zero would
    // replay the first steps' sigmas against a different sample phase - the
    // schedule is part of the run, not a per-step detail.
    for _ in 0..done {
        let _ = rng.next_f64();
    }
    for step in done..opts.steps {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let t0 = std::time::Instant::now();
        let s = &encoded[step as usize % n_samples];
        let sigma = rng.next_f64().clamp(1e-3, 1.0);
        let noise = model::hostmath::randn(s.x0.len(), opts.seed ^ (0xa5a5 + step as u64));
        let batch: Batch<f32> = make_flow_batch(&cfg, &s.x0, &s.ctx, sigma, &noise);
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
