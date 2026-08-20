// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Minimal LoRA fine-tuning loop for the audio+video LTX DiT - `crate::
//! finetune`'s AV twin, over either pure-random synthetic data (the
//! "practical entry point" shape that module's own doc explains) or a
//! caller-supplied list of clips with REAL exact ground truth (the phase-7
//! concept-learning gate, `crates/ltxv/tests/av_concept_learning.rs`, which
//! builds its own clips from `data::gen_clips` and passes them in here -
//! this module stays agnostic to where a [`SyntheticAvClip`] came from).
//!
//! ## Turning a pixel clip into a token latent, for THIS gate only
//!
//! There is no real AV VAE-latent distribution this tiny, randomly
//! initialised AV DiT was ever calibrated against (the real VAE is fit to
//! the real 22B checkpoint's own latent statistics, not to a fresh random
//! init - `crate::pipeline`'s own module doc makes the identical point for
//! the video-only smoke path). [`random_projection`]/[`encode_frame_to_
//! latent`] are a deliberately CRUDE stand-in that exists only so this
//! milestone's synthetic dataset has an EXACT, deterministic, reproducible
//! ground truth to train and measure against: a fixed (seeded, never
//! trained) linear projection of the flattened pixel frame into the DiT's
//! own `in_channels` width, normalised by `1/sqrt(N)` (plain
//! Johnson-Lindenstrauss scaling - it approximately preserves relative
//! distances, which is all a concept-margin measurement needs). This is
//! NOT a VAE and makes no claim about visual fidelity; see this crate's
//! roadmap ledger for why decoding this projection back to pixels was
//! considered and rejected as a demonstration for this milestone.

use data::rng::Rng;

use crate::av_lora::{save_adapter, LoraAdapter, LoraCfg};
use crate::av_modelgrad::{grads, make_av_flow_batch, AvCfg, AvModelWeights};

/// One named tensor: `(name, shape, row-major f32 data)`.
pub type NamedTensor = (String, Vec<usize>, Vec<f32>);

/// One AV training example: each stream's own clean latent token sequence
/// and its own text context, already in [`crate::av_modelgrad::forward`]'s
/// input shapes.
#[derive(Clone)]
pub struct SyntheticAvClip {
    pub v_latent: Vec<f32>,
    pub a_latent: Vec<f32>,
    pub v_ctx: Vec<f32>,
    pub a_ctx: Vec<f32>,
}

/// Draw one standard-normal [`SyntheticAvClip`] at `cfg`'s shape - no
/// ground-truth CONCEPT, just the "wiring proof" shape `crate::finetune::
/// synthetic_clip`'s own doc explains, extended to both streams.
pub fn random_synthetic_av_clip(cfg: &AvCfg, rng: &mut Rng) -> SyntheticAvClip {
    SyntheticAvClip {
        v_latent: (0..cfg.tv * cfg.v_in_channels).map(|_| rng.next_gaussian() as f32).collect(),
        a_latent: (0..cfg.ta * cfg.a_in_channels).map(|_| rng.next_gaussian() as f32).collect(),
        v_ctx: (0..cfg.v_context_len * cfg.vdim).map(|_| rng.next_gaussian() as f32).collect(),
        a_ctx: (0..cfg.a_context_len * cfg.adim).map(|_| rng.next_gaussian() as f32).collect(),
    }
}

/// A fixed, deterministic, NEVER-TRAINED `[out_dim, in_dim]` random
/// projection matrix - see this module's doc for why this exists and what
/// it is (and is not) a stand-in for.
pub fn random_projection(seed: u64, out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..out_dim * in_dim).map(|_| rng.next_gaussian() as f32).collect()
}

/// Apply [`random_projection`]'s matrix to one flattened input vector
/// (e.g. one HWC pixel frame), `1/sqrt(in_dim)`-normalised.
pub fn encode_to_latent(x: &[f32], proj: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    assert_eq!(proj.len(), out_dim * in_dim, "encode_to_latent: projection shape mismatch");
    let scale = 1.0 / (in_dim as f32).sqrt();
    (0..out_dim).map(|o| proj[o * in_dim..(o + 1) * in_dim].iter().zip(x).map(|(&p, &v)| p * v).sum::<f32>() * scale).collect()
}

/// A deterministic per-STRING text-context embedding: `text`'s FNV-1a hash
/// seeds a Gaussian draw of `[rows, dim]`. Same string -> same embedding,
/// different strings -> independent draws - this crate's DiT training scope
/// treats `ctx` as an opaque external input with no real text encoder
/// behind it yet (`crate::modelgrad`'s own doc), so this is not a semantic
/// text embedding; it is a reproducible STAND-IN input a caption string can
/// deterministically produce, exactly as much structure as this milestone's
/// training path can actually use.
pub fn caption_context(text: &str, rows: usize, dim: usize) -> Vec<f32> {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    let mut rng = Rng::new(hash);
    (0..rows * dim).map(|_| rng.next_gaussian() as f32).collect()
}

/// LoRA fine-tuning hyper-parameters - `crate::finetune::TrainOpts`'s AV
/// twin. `v_sigma_range`/`a_sigma_range` let a caller pin BOTH streams to
/// the same draw (equal ranges) or let them float independently
/// (diffusion forcing, `crate::av_modelgrad::AvBatch`'s doc).
pub struct TrainOpts {
    pub steps: u32,
    pub rank: usize,
    pub lr: f32,
    pub seed: u64,
    pub save_path: String,
    /// Write a checkpoint every N steps (0 = final only).
    pub ckpt_every: u32,
}

/// Fine-tune an AV LoRA adapter over `base` on `clips` (cycled in order).
/// Returns the adapter's tensors, ready to save. `progress(step, total,
/// msg)` streams per-step loss; `cancel` is polled every step - the same
/// contract `crate::finetune::run` exposes.
pub fn run(cfg: &AvCfg, base: &AvModelWeights<f32>, clips: &[SyntheticAvClip], opts: &TrainOpts, cancel: &capability::CancelToken, mut progress: impl FnMut(u32, u32, String)) -> Result<Vec<NamedTensor>, String> {
    if clips.is_empty() {
        return Err("ltxv av finetune: clips must be non-empty".into());
    }
    let total = opts.steps + 1;
    let mut rng = Rng::new(opts.seed);
    progress(0, total, format!("{} clip(s) at tv={} ta={}", clips.len(), cfg.tv, cfg.ta));

    let mut adapter = LoraAdapter::new(cfg, LoraCfg::new(opts.rank));
    for step in 0..opts.steps {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let clip = &clips[step as usize % clips.len()];
        // Independent per-stream sigma, uniform on (0,1] - the clamp keeps
        // the model time off the exact 0 the samplers never evaluate.
        let v_sigma = rng.next_f64().clamp(1e-3, 1.0);
        let a_sigma = rng.next_f64().clamp(1e-3, 1.0);
        let v_noise: Vec<f32> = (0..clip.v_latent.len()).map(|_| rng.next_gaussian() as f32).collect();
        let a_noise: Vec<f32> = (0..clip.a_latent.len()).map(|_| rng.next_gaussian() as f32).collect();
        let b = make_av_flow_batch(cfg, &clip.v_latent, &clip.a_latent, &clip.v_ctx, &clip.a_ctx, v_sigma, a_sigma, &v_noise, &a_noise);
        let (loss, g) = grads(cfg, &adapter.apply(base), &b);
        adapter.step(&g, opts.lr);
        progress(step + 1, total, format!("step {}/{}  loss {loss:.6}", step + 1, opts.steps));
        if opts.ckpt_every > 0 && (step + 1).is_multiple_of(opts.ckpt_every) && step + 1 < opts.steps {
            save_adapter(&opts.save_path, &adapter);
        }
    }
    save_adapter(&opts.save_path, &adapter);
    progress(total, total, format!("saved AV adapter -> {}", opts.save_path));
    Ok(adapter.to_tensors())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av_modelgrad::init_model;

    /// End-to-end wiring: the loop runs, saves an adapter with the right
    /// tensor count (`28 leaves * 2 (A,B) * num_layers`), and the loss
    /// trends down over the run - the AV "practical entry point" complement
    /// to `crates/ltxv/tests/av_overfit.rs`'s stronger, dedicated proof.
    #[test]
    fn the_synthetic_av_finetune_loop_runs_and_the_loss_descends() {
        let cfg = AvCfg::tiny();
        let base = init_model::<f32>(&cfg, 0xF17E_A007);
        let dir = std::env::temp_dir().join(format!("ltxv-av-finetune-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let path = dir.join("adapter.brain");
        let mut rng = Rng::new(0xC119);
        let clips: Vec<SyntheticAvClip> = (0..1).map(|_| random_synthetic_av_clip(&cfg, &mut rng)).collect();
        let opts = TrainOpts { steps: 60, rank: 4, lr: 2e-3, seed: 11, save_path: path.to_str().expect("utf-8 path").into(), ckpt_every: 0 };

        let mut losses = Vec::new();
        let cancel = capability::CancelToken::default();
        let tensors = run(&cfg, &base, &clips, &opts, &cancel, |step, total, msg| {
            if step > 0 && step < total {
                let loss: f64 = msg.rsplit(' ').next().expect("loss suffix").parse().expect("loss is a float");
                losses.push(loss);
            }
        })
        .expect("av finetune run");

        assert_eq!(tensors.len(), 28 * 2 * cfg.num_layers, "AV adapter tensor count");
        for (name, _, _) in &tensors {
            assert!(name.starts_with("diffusion_model.transformer_blocks."), "{name}");
        }
        println!("av finetune loop: loss {:.6} -> {:.6} over {} steps", losses[0], losses[losses.len() - 1], losses.len());
        // Compare the mean of the first/last 10 steps rather than the two
        // single endpoints - Adam over a single noisy synthetic example
        // does not monotonically improve step-to-step (the AV model's
        // larger parameter count is noisier here than the video-only
        // path's own single-endpoint check tolerates), but the TREND is
        // still real and this is robust to that noise.
        let head = losses[..10].iter().sum::<f64>() / 10.0;
        let tail = losses[losses.len() - 10..].iter().sum::<f64>() / 10.0;
        assert!(tail < head * 0.97, "loss must trend down: first-10 avg {head:.6} -> last-10 avg {tail:.6}: {losses:?}");

        let reloaded = crate::av_lora::load_adapter(path.to_str().expect("utf-8 path"), &cfg).expect("reload adapter");
        assert_eq!(reloaded.rank(), 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `caption_context` must be a pure function of the string (reproducible
    /// across calls) and distinguish different strings.
    #[test]
    fn caption_context_is_deterministic_and_distinguishes_strings() {
        let a1 = caption_context("a magenta triangle", 3, 5);
        let a2 = caption_context("a magenta triangle", 3, 5);
        let b = caption_context("a cyan square", 3, 5);
        assert_eq!(a1, a2, "same string must give the same embedding");
        assert_ne!(a1, b, "different strings must give different embeddings");
    }

    /// `encode_to_latent` must be a pure, deterministic linear map.
    #[test]
    fn encode_to_latent_is_deterministic() {
        let proj = random_projection(7, 6, 12);
        let x: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let e1 = encode_to_latent(&x, &proj, 6);
        let e2 = encode_to_latent(&x, &proj, 6);
        assert_eq!(e1, e2);
        assert_eq!(e1.len(), 6);
    }
}
