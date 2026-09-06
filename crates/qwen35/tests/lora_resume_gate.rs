// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gate for self-improve roadmap P17 ("resumable LoRA adapters"), qwen35
//! side: mirrors `crates/qwen3/tests/lora_resume_gate.rs` exactly (same
//! discipline, same reasoning - see that file's own doc comment for why
//! "the adapter changed across the resume boundary" alone is not a check).
//! `qwen35::finetune::finetune` mirrored `qwen3::finetune::finetune`'s
//! always-fresh-init defect verbatim; `finetune_from`'s `resume` switch
//! fixes both the same way.

use std::path::{Path, PathBuf};

use data::binio::{self, Meta};
use data::rng::Rng;
use gpu_core::Gpu;
use model::IGNORE;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-qwen35-lora-resume-gate-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const PROMPT: [u32; 3] = [2, 5, 8];
const TARGET: u32 = 15;
const RANK: u32 = 5;
const ALPHA: f32 = 8.0;
/// Deliberately short: long enough for a real gradient signal, short enough
/// that cycle 1 alone has NOT yet reached this task's loss floor, leaving
/// cycle 2 room to show a real further drop (see `avg_loss`'s doc comment).
const CYCLE_STEPS: u32 = 8;

/// `PROMPT ++ [TARGET]`, repeated `reps` times back to back. Mask is `true`
/// only at each repetition's target position.
fn build_stream(reps: usize) -> (Vec<u32>, Vec<bool>) {
    let mut tokens = Vec::with_capacity(reps * (PROMPT.len() + 1));
    let mut mask = Vec::with_capacity(tokens.capacity());
    for _ in 0..reps {
        for &p in &PROMPT {
            tokens.push(p);
            mask.push(false);
        }
        tokens.push(TARGET);
        mask.push(true);
    }
    (tokens, mask)
}

fn write_dataset(dir: &Path, vocab: u32) {
    std::fs::create_dir_all(dir).unwrap();
    let (tokens, mask) = build_stream(200);
    binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
    binio::write_mask_bin(&dir.join("train.mask.bin"), &mask).unwrap();
    binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
    binio::write_mask_bin(&dir.join("val.mask.bin"), &[]).unwrap();
    std::fs::write(dir.join("meta.json"), Meta::vocab_only(vocab as usize)).unwrap();
}

fn opts(seed: u64, steps: u32, block_size: u32) -> model::FitOpts {
    model::FitOpts {
        steps,
        batch_size: 1,
        block_size,
        lr: 5e-2,
        min_lr: 5e-3,
        warmup: 15,
        decay_iters: steps,
        weight_decay: 0.0,
        grad_clip: 1.0,
        grad_accum: 1,
        eval_interval: 0,
        eval_batches: 0,
        checkpoint_secs: 0,
        mask_before: None,
        mask_per_line: false,
        align_to_lines: false,
        seed,
    }
}

/// Mean CE loss over `n` batches, reloaded from disk. `finetune_from`'s own
/// returned `final` loss is deliberately a single noisy last-training-batch
/// value (see `crates/qwen3/tests/lora_learning_gate.rs`'s own comment on why
/// that is only a weak sanity signal) - too noisy to hang a "loss keeps
/// dropping across the resume boundary" assertion on by itself. Averaging
/// many batches off the reloaded checkpoint gives a stable enough number for
/// that comparison.
fn avg_loss(ckpt_path: &str, dir: &Path, seed: u64, n: usize, t: u32) -> f32 {
    let (train, _val, bcfg, _vocab) = model::load_dataset(dir, &opts(seed, 1, t)).expect("load_dataset");
    let c = checkpoint::load(ckpt_path);
    let cfg = Qwen35Config::from_json(&c.header["config"]);
    let m = Qwen35::new_on(Gpu::new(pipelines()), cfg, 1, t, &c.by_role(""));
    let mut rng = Rng::new(seed ^ 0xA5A5_5A5A);
    let mut total = 0.0f32;
    for _ in 0..n {
        let (x, y) = train.get_batch(&bcfg, &mut rng);
        let targets: Vec<u32> = y.iter().map(|&v| if v < 0 { IGNORE } else { v as u32 }).collect();
        m.set_batch(&x, &targets);
        total += m.forward();
    }
    total / n as f32
}

/// The Frobenius norm of the tracked leaf's LoRA delta `B @ A`, computed from
/// the checkpoint reloaded off disk (not the live in-process model) via the
/// same `checkpoint::load` + `Qwen35::new_on` path `lora_roundtrip.rs` uses
/// (this crate has no streaming `load_inference` helper).
fn tracked_leaf_delta_norm(ckpt_path: &str, t: u32) -> f64 {
    let c = checkpoint::load(ckpt_path);
    let cfg = Qwen35Config::from_json(&c.header["config"]);
    let tensors = c.by_role("");
    // `blocks.0.mlp.gate.weight`: dense MLP, present on every layer
    // regardless of GDN-vs-full-attention type. o = intermediate_size = 112,
    // i = d_model = 96.
    let o = cfg.intermediate_size as usize;
    let i = cfg.d_model as usize;
    let r = RANK as usize;
    let m = Qwen35::new_on(Gpu::new(pipelines()), cfg, 1, t, &tensors);
    let a = m.read_weight("blocks.0.mlp.gate.weight.lora_a"); // [r, i]
    let b = m.read_weight("blocks.0.mlp.gate.weight.lora_b"); // [o, r]
    assert_eq!(a.len(), r * i);
    assert_eq!(b.len(), o * r);
    let mut sumsq = 0.0f64;
    for oi in 0..o {
        for ii in 0..i {
            let mut acc = 0.0f64;
            for rr in 0..r {
                acc += b[oi * r + rr] as f64 * a[rr * i + ii] as f64;
            }
            sumsq += acc * acc;
        }
    }
    sumsq.sqrt()
}

#[test]
fn lora_resume_continues_the_same_adapter_instead_of_resetting_it() {
    if skip() {
        return;
    }

    let scratch = tmp("run");
    let base_cfg = Qwen35Config::tiny();
    let t = base_cfg.block_size;
    let vocab = base_cfg.vocab;
    let base_init = qwen35::init::init_weights(&base_cfg, 7);
    let base_path = scratch.join("base.safetensors");
    Qwen35::new_on(Gpu::new(pipelines()), base_cfg, 1, t, &base_init).save(base_path.to_str().unwrap());
    let base_str = base_path.to_str().unwrap();

    let data_dir = scratch.join("data");
    write_dataset(&data_dir, vocab);

    let mode = qwen35::finetune::Mode::Lora { rank: RANK, alpha: ALPHA };
    let out = scratch.join("adapter.safetensors");
    let out_str = out.to_str().unwrap();

    // Cycle 1: fresh start.
    let (a_init, _a_final) = qwen35::finetune::finetune_from(base_str, &data_dir, &opts(1234, CYCLE_STEPS, t), &mode, out_str, false)
        .expect("cycle 1 finetune_from");
    let eval1 = avg_loss(out_str, &data_dir, 99, 30, t);

    // Control: an independent `resume=false` run into a SEPARATE path, same
    // base/seed/data/steps as cycle 1 - the "what a fresh start looks like on
    // this exact setup" yardstick cycle 2 is measured against below.
    let control_out = scratch.join("control.safetensors");
    let (control_init, _control_final) =
        qwen35::finetune::finetune_from(base_str, &data_dir, &opts(1234, CYCLE_STEPS, t), &mode, control_out.to_str().unwrap(), false)
            .expect("control finetune_from");

    let norm1 = tracked_leaf_delta_norm(out_str, t);

    // Cycle 2: resumes from `out`, NOT from `base`.
    let (b_init, _b_final) = qwen35::finetune::finetune_from(base_str, &data_dir, &opts(5678, CYCLE_STEPS, t), &mode, out_str, true)
        .expect("cycle 2 finetune_from (resume)");
    let eval2 = avg_loss(out_str, &data_dir, 99, 30, t);

    let norm2 = tracked_leaf_delta_norm(out_str, t);

    println!("cycle 1:  init {a_init:.6}  ->  30-batch eval {eval1:.6}");
    println!("control:  init {control_init:.6}");
    println!("cycle 2:  init {b_init:.6}  ->  30-batch eval {eval2:.6}  (resumed)");
    println!("||B*A||:  {norm1:.6} -> {norm2:.6}");

    // Loss keeps dropping in both cycles -- training is real, not a no-op.
    // `avg_loss` (30 batches, off the reloaded checkpoint) is the stable
    // signal this hangs its verdict on; `finetune_from`'s own returned
    // `final` is a single noisy last-training-batch value, not asserted on
    // here (see `avg_loss`'s own doc comment).
    assert!(eval1 < a_init, "cycle 1 loss did not drop: init {a_init:.6} -> 30-batch eval {eval1:.6}");
    assert!(eval2 < eval1, "cycle 2 (resumed) loss did not drop further: {eval1:.6} -> {eval2:.6}");

    // The control run (fresh, same seed/base/data as cycle 1) must land on
    // cycle 1's own initial loss almost exactly -- proving what a fresh start
    // looks like here, so the comparison below actually means something.
    assert!(
        (control_init - a_init).abs() < 1e-3,
        "control run (resume=false, same seed/base/data as cycle 1) did not reproduce cycle 1's initial loss: {control_init:.6} vs {a_init:.6} -- \
         the fresh-start baseline itself is not reproducible, so the resume comparison below cannot be trusted"
    );

    // The decisive check: a resumed cycle's initial loss must pick up near
    // where cycle 1 actually left off (`eval1`), not jump back up near a
    // fresh adapter's initial loss the way a broken (always-fresh-init)
    // resume would.
    assert!(
        b_init < (a_init + eval1) / 2.0,
        "resumed cycle's initial loss ({b_init:.6}) was not below the midpoint of cycle 1's own trajectory ({a_init:.6} -> {eval1:.6}) -- \
         resume did not carry over the trained adapter, it looks like a fresh zero-delta start"
    );

    // The LoRA delta's norm ||B*A|| must grow across the resume boundary.
    assert!(norm2 > norm1, "||B*A|| did not grow across the resume boundary: {norm1:.6} -> {norm2:.6}");
}
