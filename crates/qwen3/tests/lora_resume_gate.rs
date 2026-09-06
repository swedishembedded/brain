// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gate for self-improve roadmap P17 ("resumable LoRA adapters"): before this
//! fix, `qwen3::finetune::finetune` always called `init_weights` fresh, so a
//! LoRA adapter's `lora_b` was reset to its zero-delta init on every cycle -
//! a second `train -> save -> train` cycle could never build on the first
//! one's progress, it could only start over. `finetune_from`'s `resume`
//! switch fixes this by loading `out` (not `base`) for both config and
//! weights when resuming.
//!
//! Mirrors `crates/qwen3/tests/lora_learning_gate.rs`'s discipline: a
//! statistic a broken result also satisfies is not a check. "The adapter
//! changed across the resume boundary" is satisfied by a BROKEN resume too
//! (a fresh zero-delta adapter trained for more steps also changes). What
//! only a WORKING resume explains is: (1) the resumed cycle's initial loss
//! picks up near where the first cycle's final loss left off, rather than
//! jumping back up near the first cycle's own initial loss the way a reset
//! adapter would, and (2) a `resume=false` control run - same base, same
//! seed, same data - reproduces cycle 1's own initial loss almost exactly,
//! proving that control is what "starting fresh" actually looks like on this
//! setup, so the resumed run's very different initial loss cannot be
//! explained by anything other than the adapter's state having carried over.

use std::path::{Path, PathBuf};

use data::binio::{self, Meta};
use data::rng::Rng;
use qwen3::{Qwen, QwenConfig, IGNORE};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-qwen-lora-resume-gate-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const VOCAB: u32 = 24;
const PROMPT: [u32; 3] = [2, 5, 8];
const TARGET: u32 = 15;
const RANK: u32 = 3;
const ALPHA: f32 = 6.0;
/// Deliberately short: long enough for a real gradient signal, short enough
/// that cycle 1 alone has NOT yet reached this task's loss floor, leaving
/// cycle 2 room to show a real further drop (see `avg_loss`'s doc comment).
const CYCLE_STEPS: u32 = 8;

fn tiny_config() -> QwenConfig {
    QwenConfig {
        vocab: VOCAB,
        block_size: 16,
        n_layers: 2,
        d_model: 16,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        d_ff: 32,
        rope_theta: 1.0e6,
        rms_eps: 1e-6,
        max_position_embeddings: 16,
        tie_embeddings: true,
        qk_norm: true,
        attn_bias: false,
        lora: None,
    }
}

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

fn write_dataset(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let (tokens, mask) = build_stream(200);
    binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
    binio::write_mask_bin(&dir.join("train.mask.bin"), &mask).unwrap();
    binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
    binio::write_mask_bin(&dir.join("val.mask.bin"), &[]).unwrap();
    std::fs::write(dir.join("meta.json"), Meta::vocab_only(VOCAB as usize)).unwrap();
}

fn opts(seed: u64, steps: u32) -> model::FitOpts {
    model::FitOpts {
        steps,
        batch_size: 1,
        block_size: 16,
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
/// value (see `lora_learning_gate.rs`'s own comment on why that is only a
/// weak sanity signal) - too noisy to hang a "loss keeps dropping across the
/// resume boundary" assertion on by itself. Averaging many batches off the
/// reloaded checkpoint gives a stable enough number for that comparison.
fn avg_loss(ckpt_path: &str, dir: &Path, seed: u64, n: usize) -> f32 {
    let (train, _val, bcfg, _vocab) = model::load_dataset(dir, &opts(seed, 1)).expect("load_dataset");
    let m = Qwen::load_inference(ckpt_path, 1, 16);
    let mut rng = Rng::new(seed ^ 0xA5A5_5A5A);
    let mut total = 0.0f32;
    for _ in 0..n {
        let (x, y) = train.get_batch(&bcfg, &mut rng);
        let t: Vec<u32> = y.iter().map(|&v| if v < 0 { IGNORE } else { v as u32 }).collect();
        m.set_batch(&x, &t);
        total += m.forward();
    }
    total / n as f32
}

/// The Frobenius norm of the tracked leaf's LoRA delta `B @ A`, computed from
/// the checkpoint reloaded off disk (not the live in-process model) -- the
/// same reload discipline `lora_learning_gate.rs` uses, since a checkpoint
/// that only "looks right" in the process that just wrote it is not the bug
/// this phase fixes.
fn tracked_leaf_delta_norm(ckpt_path: &str) -> f64 {
    // `blocks.0.attn.wq.weight`: o = q_dim = n_heads*head_dim = 32, i = d_model = 16.
    const O: usize = 32;
    const I: usize = 16;
    let r = RANK as usize;
    let m = Qwen::load_inference(ckpt_path, 1, 16);
    let a = m.read_weight("blocks.0.attn.wq.weight.lora_a"); // [r, I]
    let b = m.read_weight("blocks.0.attn.wq.weight.lora_b"); // [O, r]
    assert_eq!(a.len(), r * I);
    assert_eq!(b.len(), O * r);
    let mut sumsq = 0.0f64;
    for oi in 0..O {
        for ii in 0..I {
            let mut acc = 0.0f64;
            for rr in 0..r {
                acc += b[oi * r + rr] as f64 * a[rr * I + ii] as f64;
            }
            sumsq += acc * acc;
        }
    }
    sumsq.sqrt()
}

#[test]
fn lora_resume_continues_the_same_adapter_instead_of_resetting_it() {
    if skip() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
        return;
    }

    let scratch = tmp("run");
    let base_cfg = tiny_config();
    let base_init = qwen3::init_weights(&base_cfg, 7);
    let base_path = scratch.join("base.safetensors");
    Qwen::new(base_cfg, 1, 16, &base_init).save(base_path.to_str().unwrap());
    let base_str = base_path.to_str().unwrap();

    let data_dir = scratch.join("data");
    write_dataset(&data_dir);

    let mode = qwen3::finetune::Mode::Lora { rank: RANK, alpha: ALPHA };
    let out = scratch.join("adapter.safetensors");
    let out_str = out.to_str().unwrap();

    // Cycle 1: fresh start.
    let (a_init, _a_final) = qwen3::finetune::finetune_from(base_str, &data_dir, &opts(1234, CYCLE_STEPS), &mode, out_str, false)
        .expect("cycle 1 finetune_from");
    let eval1 = avg_loss(out_str, &data_dir, 99, 30);

    // Control: an independent `resume=false` run into a SEPARATE path, same
    // base/seed/data/steps as cycle 1. If this reproduces cycle 1's own
    // initial loss almost exactly, that establishes what "a fresh adapter
    // start" looks like on this exact setup -- the yardstick cycle 2 (which
    // DOES resume) is measured against below.
    let control_out = scratch.join("control.safetensors");
    let (control_init, _control_final) =
        qwen3::finetune::finetune_from(base_str, &data_dir, &opts(1234, CYCLE_STEPS), &mode, control_out.to_str().unwrap(), false)
            .expect("control finetune_from");

    let norm1 = tracked_leaf_delta_norm(out_str);

    // Cycle 2: resumes from `out`, NOT from `base`.
    let (b_init, _b_final) = qwen3::finetune::finetune_from(base_str, &data_dir, &opts(5678, CYCLE_STEPS), &mode, out_str, true)
        .expect("cycle 2 finetune_from (resume)");
    let eval2 = avg_loss(out_str, &data_dir, 99, 30);

    let norm2 = tracked_leaf_delta_norm(out_str);

    println!("cycle 1:  init {a_init:.6}  ->  30-batch eval {eval1:.6}");
    println!("control:  init {control_init:.6}");
    println!("cycle 2:  init {b_init:.6}  ->  30-batch eval {eval2:.6}  (resumed)");
    println!("||B*A||:  {norm1:.6} -> {norm2:.6}");

    // Loss keeps dropping in both cycles -- training is real, not a no-op.
    // `avg_loss` (30 batches, off the reloaded checkpoint) is the stable
    // signal this hangs its verdict on; `finetune_from`'s own returned
    // `final` is a single noisy last-training-batch value (see
    // `lora_learning_gate.rs`'s own comment on that), not asserted on here.
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
    // fresh adapter's initial loss. A broken resume (fresh zero-delta
    // adapter every cycle) would put `b_init` close to `a_init`/`control_init`
    // (~ln(24) for this uniform 24-way softmax); a working resume puts it
    // close to `eval1` instead.
    assert!(
        b_init < (a_init + eval1) / 2.0,
        "resumed cycle's initial loss ({b_init:.6}) was not below the midpoint of cycle 1's own trajectory ({a_init:.6} -> {eval1:.6}) -- \
         resume did not carry over the trained adapter, it looks like a fresh zero-delta start"
    );

    // The LoRA delta's norm ||B*A|| must grow across the resume boundary --
    // more optimization steps applied to the SAME accumulated adapter, not a
    // fresh adapter re-discovering a similar-sized delta from zero.
    assert!(norm2 > norm1, "||B*A|| did not grow across the resume boundary: {norm1:.6} -> {norm2:.6}");
}

/// Resuming into a checkpoint whose LoRA rank/alpha do not match the request
/// must fail loudly (not silently re-shape the adapter or drop the mismatch).
#[test]
fn resume_with_mismatched_lora_rank_panics() {
    if skip() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
        return;
    }
    let scratch = tmp("mismatch");
    let base_cfg = tiny_config();
    let base_init = qwen3::init_weights(&base_cfg, 3);
    let base_path = scratch.join("base.safetensors");
    Qwen::new(base_cfg, 1, 16, &base_init).save(base_path.to_str().unwrap());
    let base_str = base_path.to_str().unwrap();

    let data_dir = scratch.join("data");
    write_dataset(&data_dir);

    let out = scratch.join("adapter.safetensors");
    qwen3::finetune::finetune_from(
        base_str,
        &data_dir,
        &opts(1, 2),
        &qwen3::finetune::Mode::Lora { rank: RANK, alpha: ALPHA },
        out.to_str().unwrap(),
        false,
    )
    .expect("initial finetune_from");

    let result = std::panic::catch_unwind(|| {
        qwen3::finetune::finetune_from(
            base_str,
            &data_dir,
            &opts(1, 2),
            &qwen3::finetune::Mode::Lora { rank: RANK + 1, alpha: ALPHA },
            out.to_str().unwrap(),
            true,
        )
    });
    assert!(result.is_err(), "resuming with a mismatched LoRA rank should panic, not silently proceed");
}
