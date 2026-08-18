// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.5/3.8-27B fine-tuning: full (with optimizer offload) and LoRA, over
//! brain's masked token datasets. A self-contained training loop so both
//! modes seed correctly from a base checkpoint - full merges the checkpoint
//! weights as-is; LoRA merges them and adds freshly-initialised zero-delta
//! adapters - which `model::fit`'s resume path (checkpoint-config-wins)
//! cannot do. Mirrors `qwen3::finetune` exactly (M8's genuinely new surface
//! for this family - qwen35moe has none).

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use data::rng::Rng;
use gpu_core::Gpu;
use model::{cosine_lr, FitOpts, IGNORE};

use crate::config::{lora_targets, LoraCfg, Qwen35Config};
use crate::model::{Qwen35, PIPELINES};

/// Which fine-tuning scheme.
#[derive(Clone, Debug)]
pub enum Mode {
    /// Every weight trainable; AdamW moments offloaded to system RAM (Role::Offload).
    FullOffload,
    /// Low-rank adapters on the 12 targetable projections (`crate::config::
    /// lora_targets`); base frozen.
    Lora { rank: u32, alpha: f32 },
}

/// Fine-tune `base` on the masked dataset in `dir`, writing `out`. Returns
/// `(initial_loss, final_loss)`.
pub fn finetune(base: &str, dir: &Path, opts: &FitOpts, mode: &Mode, out: &str) -> std::io::Result<(f32, f32)> {
    // Base architecture + weights from the checkpoint.
    let c = checkpoint::load(base);
    let mut cfg = Qwen35Config::from_json(&c.header["config"]);
    let base_w = c.by_role("");

    // Build the init map for THIS mode's parameter set, seeded from the base.
    // BRAIN_OFFLOAD_ADAM is a process-global switch (the same convention
    // `model::parallel`/`model::shard` use) -- save/restore the caller's prior
    // value rather than clobbering it, so a nested or later call in the same
    // process doesn't silently inherit this call's mode.
    let prev_off = std::env::var("BRAIN_OFFLOAD_ADAM").ok();
    match mode {
        Mode::FullOffload => {
            std::env::set_var("BRAIN_OFFLOAD_ADAM", "1");
        }
        Mode::Lora { rank, alpha } => {
            std::env::remove_var("BRAIN_OFFLOAD_ADAM");
            cfg.lora = Some(LoraCfg { rank: *rank, alpha: *alpha, targets: lora_targets() });
        }
    }
    // Fresh init for the (possibly LoRA-extended) param set, then overwrite the
    // base params with the checkpoint's -- adapters stay at their zero-delta init.
    let mut init: HashMap<String, Vec<f32>> = crate::init::init_weights(&cfg, opts.seed);
    for (k, v) in base_w {
        init.insert(k, v);
    }

    let m = Qwen35::new_train_on(Gpu::new(PIPELINES), cfg, opts.batch_size, opts.block_size, &init);
    match prev_off {
        Some(v) => std::env::set_var("BRAIN_OFFLOAD_ADAM", v),
        None => std::env::remove_var("BRAIN_OFFLOAD_ADAM"),
    }
    let (train, _val, bcfg, _vocab) = model::load_dataset(dir, opts)?;
    let mut rng = Rng::new(opts.seed ^ 0xA5A5_5A5A);

    // Initial loss (a few batches) for the return value.
    let mut initial = 0.0f32;
    for _ in 0..3 {
        let (x, y) = train.get_batch(&bcfg, &mut rng);
        let t: Vec<u32> = y.iter().map(|&v| if v < 0 { IGNORE } else { v as u32 }).collect();
        m.set_batch(&x, &t);
        initial += m.forward();
    }
    initial /= 3.0;

    let mut last = initial;
    let mut last_save = Instant::now();
    for step in 0..opts.steps {
        let lr = cosine_lr(step, opts);
        m.zero_grads();
        let mut loss = 0.0;
        for _ in 0..opts.grad_accum.max(1) {
            let (x, y) = train.get_batch(&bcfg, &mut rng);
            let t: Vec<u32> = y.iter().map(|&v| if v < 0 { IGNORE } else { v as u32 }).collect();
            m.set_batch(&x, &t);
            loss += m.forward();
            m.backward();
        }
        let clip = (opts.grad_clip > 0.0).then_some(opts.grad_clip);
        m.adamw_step(step + 1, lr, opts.weight_decay, clip, 1.0 / opts.grad_accum.max(1) as f32);
        m.poll_wait();
        last = loss / opts.grad_accum.max(1) as f32;

        if opts.checkpoint_secs > 0 && last_save.elapsed().as_secs() >= opts.checkpoint_secs {
            let ts = Instant::now();
            m.save(out);
            println!("step {:>6}  saved checkpoint -> {out} ({:.1} s)", step + 1, ts.elapsed().as_secs_f64());
            last_save = Instant::now();
        }
    }
    let ts = Instant::now();
    m.save(out);
    println!("saved checkpoint -> {out} ({:.1} s)", ts.elapsed().as_secs_f64());
    Ok((initial, last))
}
