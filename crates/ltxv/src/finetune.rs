// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Minimal LoRA fine-tuning loop for the video-only LTX DiT, over SYNTHETIC
//! data.
//!
//! `wan::finetune` reads captioned video clips off disk (`data::episode`),
//! VAE-encodes them and umT5-encodes their captions before the flow-matching
//! loop starts. None of that exists for `ltxv` yet at this milestone - there
//! is no real 22B checkpoint to fine-tune (`crate::dit::random_tiny_weights`
//! is the only weight source this whole crate has, the same "SMOKE TEST of
//! the wiring, not a quality claim" scope `crate::pipeline`'s own module doc
//! records), no real text encoder, and no dataset loader for this
//! architecture. So this trainer draws its own synthetic "clip" - a random
//! latent TOKEN sequence and a random text context, already in
//! `crate::dit::LtxDit::forward`'s own input shapes (see this crate's
//! module doc: there is no patchify/unpatchify step inside the DiT's own
//! math, unlike Wan's pixel-space VAE latents) - and proves the same thing
//! `wan::finetune`'s loop proves: that a full flow-matching LoRA step, run
//! end to end through the gradchecked host trainer
//! ([`crate::modelgrad::grads`]), actually descends. `crates/ltxv/tests/
//! overfit.rs` is the stronger, dedicated proof-of-life gate (full-model
//! Adam driving the loss to ~0 on one fixed batch); this loop is the
//! "practical training entry point" shape, LoRA-scoped, the way a real
//! fine-tune run would be invoked once a real checkpoint exists.

use crate::lora::{save_adapter, LoraAdapter, LoraCfg};
use crate::modelgrad::{grads, make_flow_batch, Cfg, ModelWeights};
use data::rng::Rng;

/// One named tensor: `(name, shape, row-major f32 data)`.
pub type NamedTensor = (String, Vec<usize>, Vec<f32>);

/// One synthetic training example: a random latent token sequence
/// (`[t*in_channels]`) and a random text context (`[context_len*dim]`),
/// standard-normal per element - the natural scale a normalised VAE latent
/// and an embedded text context both live near.
#[derive(Clone)]
pub struct SyntheticClip {
    pub latent: Vec<f32>,
    pub ctx: Vec<f32>,
}

/// Draw one [`SyntheticClip`] at `cfg`'s shape.
pub fn synthetic_clip(cfg: &Cfg, rng: &mut Rng) -> SyntheticClip {
    let latent: Vec<f32> = (0..cfg.t * cfg.in_channels).map(|_| rng.next_gaussian() as f32).collect();
    let ctx: Vec<f32> = (0..cfg.context_len * cfg.dim).map(|_| rng.next_gaussian() as f32).collect();
    SyntheticClip { latent, ctx }
}

/// LoRA fine-tuning hyper-parameters.
pub struct TrainOpts {
    pub steps: u32,
    pub rank: usize,
    pub lr: f32,
    /// Synthetic clips to draw up front; steps cycle through them.
    pub samples: usize,
    pub seed: u64,
    pub save_path: String,
    /// Write a checkpoint every N steps (0 = final only).
    pub ckpt_every: u32,
}

/// Fine-tune a LoRA adapter over `base` on synthetic data. Returns the
/// adapter's tensors, ready to save. `progress(step, total, msg)` streams
/// per-step loss; `cancel` is polled every step, and periodic checkpoints
/// already written survive an abort - the same contract `wan::finetune::run`
/// exposes.
pub fn run(cfg: &Cfg, base: &ModelWeights<f32>, opts: &TrainOpts, cancel: &capability::CancelToken, mut progress: impl FnMut(u32, u32, String)) -> Result<Vec<NamedTensor>, String> {
    if opts.samples == 0 {
        return Err("ltxv finetune: samples must be >= 1".into());
    }
    let total = opts.steps + 1;
    let mut rng = Rng::new(opts.seed);
    let clips: Vec<SyntheticClip> = (0..opts.samples).map(|_| synthetic_clip(cfg, &mut rng)).collect();
    progress(0, total, format!("{} synthetic clip(s) at t={} context_len={}", clips.len(), cfg.t, cfg.context_len));

    let mut adapter = LoraAdapter::new(cfg, LoraCfg::new(opts.rank));
    for step in 0..opts.steps {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let clip = &clips[step as usize % clips.len()];
        // σ is drawn uniformly on (0, 1]; the clamp keeps the model time off
        // the exact 0 the samplers never evaluate.
        let sigma = rng.next_f64().clamp(1e-3, 1.0);
        let noise: Vec<f32> = (0..clip.latent.len()).map(|_| rng.next_gaussian() as f32).collect();
        let b = make_flow_batch(cfg, &clip.latent, &clip.ctx, sigma, &noise);
        let (loss, g) = grads(cfg, &adapter.apply(base), &b);
        adapter.step(&g, opts.lr);
        progress(step + 1, total, format!("step {}/{}  loss {loss:.6}", step + 1, opts.steps));
        if opts.ckpt_every > 0 && (step + 1).is_multiple_of(opts.ckpt_every) && step + 1 < opts.steps {
            save_adapter(&opts.save_path, &adapter);
        }
    }
    save_adapter(&opts.save_path, &adapter);
    progress(total, total, format!("saved adapter -> {}", opts.save_path));
    Ok(adapter.to_tensors())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modelgrad::init_model;

    /// End-to-end wiring: the loop runs, saves an adapter with the right
    /// tensor count (`10 leaves * 2 (A,B) * num_layers`), and the loss
    /// trends down over the run - the "practical entry point" complement to
    /// `tests/overfit.rs`'s stronger, dedicated proof.
    #[test]
    fn the_synthetic_finetune_loop_runs_and_the_loss_descends() {
        let cfg = Cfg::tiny();
        let base = init_model::<f32>(&cfg, 0xF17E_7007);
        let dir = std::env::temp_dir().join(format!("ltxv-finetune-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let path = dir.join("adapter.brain");
        let opts = TrainOpts { steps: 20, rank: 4, lr: 3e-3, samples: 1, seed: 11, save_path: path.to_str().expect("utf-8 path").into(), ckpt_every: 0 };

        let mut losses = Vec::new();
        let cancel = capability::CancelToken::default();
        let tensors = run(&cfg, &base, &opts, &cancel, |step, total, msg| {
            // Steps 1..=opts.steps carry "... loss <f64>"; step 0 (dataset
            // summary) and the final step==total (save-path message) do not.
            if step > 0 && step < total {
                let loss: f64 = msg.rsplit(' ').next().expect("loss suffix").parse().expect("loss is a float");
                losses.push(loss);
            }
        })
        .expect("finetune run");

        assert_eq!(tensors.len(), 10 * 2 * cfg.num_layers, "adapter tensor count");
        for (name, _, _) in &tensors {
            assert!(name.starts_with("diffusion_model.transformer_blocks."), "{name}");
        }
        println!("finetune loop: loss {:.6} -> {:.6} over {} steps", losses[0], losses[losses.len() - 1], losses.len());
        assert!(losses[losses.len() - 1] < losses[0], "loss must trend down: {losses:?}");

        let reloaded = crate::lora::load_adapter(path.to_str().expect("utf-8 path"), &cfg).expect("reload adapter");
        assert_eq!(reloaded.rank(), 4);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
