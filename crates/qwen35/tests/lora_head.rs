// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The training build's LM head and embedding: the loss and logit gradient
//! come from the chunked head (a few rows of logits alive, never
//! `tokens x vocab`), and a narrow-base LoRA build holds the frozen embedding
//! table and head in bf16 so neither is a 5 GB fp32 tensor at the real shape.

mod common;

use common::*;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35, TrainOpts};

const LONG: usize = 128;

fn long_config(tie: bool) -> Qwen35Config {
    Qwen35Config { block_size: LONG as u32, tie_embeddings: tie, lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() }
}

fn build(cfg: &Qwen35Config, init: &std::collections::HashMap<String, Vec<f32>>, t: usize, opts: &TrainOpts) -> Qwen35 {
    Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, t as u32, init, opts)
}

/// More tokens than one chunk of logits holds: the loss and every gradient
/// match the single-chunk head (same math, different chunking).
#[test]
fn the_chunked_head_gives_the_same_loss_and_gradients_whatever_the_chunking() {
    let cfg = long_config(false);
    let init = weights(&cfg, 0.3);
    let (x, y) = batch_of(LONG);
    let one = TrainOpts::default();
    let small = TrainOpts { head_budget_words: Some(64 * cfg.vocab as u64), ..TrainOpts::default() };
    let want = loss_and_grads_on(&build(&cfg, &init, LONG, &one), &x, &y);
    let got = loss_and_grads_on(&build(&cfg, &init, LONG, &small), &x, &y);
    assert_same_training_signal(got, want, 1e-5, 0.99999);
}

/// The chunked head's recorded steps follow the layer work on the same
/// command stream: on the default device (a GPU where there is one) the loss
/// and gradients equal the CPU backend's, where a head recorded on another
/// stream read hidden states that were not computed yet and saw all-zero logits.
#[test]
fn the_chunked_head_on_the_default_device_matches_the_cpu_backend() {
    let cfg = long_config(false);
    let init = weights(&cfg, 0.3);
    let (x, y) = batch_of(LONG);
    let on = |gpu: Gpu| loss_and_grads_on(&Qwen35::new_train_opts_on(gpu, cfg.clone(), 1, LONG as u32, &init, &TrainOpts::default()), &x, &y);
    let want = on(Gpu::new_cpu(pipelines()));
    assert!(want.0 < (cfg.vocab as f32).ln() * 1.01 && want.0 > 0.0);
    let got = on(Gpu::new(pipelines()));
    assert!(got.0 != (cfg.vocab as f32).ln(), "uniform loss: the head saw zero logits");
    assert_same_training_signal(got, want, 1e-3, 0.999);
}

/// A narrow base holds the embedding table and head in bf16 (tied or not),
/// and still trains the adapters the fp32 build does.
#[test]
fn a_bf16_base_holds_the_embedding_and_head_in_bf16() {
    for tie in [false, true] {
        let cfg = Qwen35Config { block_size: T as u32, ..long_config(tie) };
        let init = weights(&cfg, 0.3);
        let wide = build(&cfg, &init, T, &TrainOpts::default());
        let half = build(&cfg, &init, T, &TrainOpts { base: TierPolicy::uniform(Dtype::BF16), ..TrainOpts::default() });
        for table in ["tok.weight", cfg.head_weight()] {
            assert_eq!(half.linear_dtype(table), Some(Dtype::BF16), "tie={tie}: {table} is held in bf16");
            assert_eq!(half.read_weight(table), init[table], "tie={tie}: {table} reads back as the values it was built from");
        }
        assert_same_training_signal(loss_and_grads(&half), loss_and_grads(&wide), 1e-4, 0.9999);
    }
}

/// The logits a chunked-head build reports are the same ones the head
/// computes per chunk, whatever the chunk size.
#[test]
fn logits_all_of_a_chunked_head_build_match_across_chunk_sizes() {
    let cfg = long_config(false);
    let init = weights(&cfg, 0.3);
    let (x, _) = batch_of(LONG);
    let a = build(&cfg, &init, LONG, &TrainOpts::default()).logits_all(&x);
    let b = build(&cfg, &init, LONG, &TrainOpts { head_budget_words: Some(64 * cfg.vocab as u64), ..TrainOpts::default() }).logits_all(&x);
    assert_eq!(a.len(), LONG * cfg.vocab as usize);
    assert_eq!(a, b);
}

#[test]
#[ignore = "child process helper, driven by a_training_build_never_holds_tokens_times_vocab_logits"]
fn child_charged_bytes_at_two_lengths_with_a_wide_vocab() {
    publish_ceiling();
    let cfg = Qwen35Config { vocab: 8192, ..long_config(false) };
    let init = qwen35::init::init_weights(&cfg, 5); // seed: only the shapes matter here
    let at = |t: u32| {
        let opts = TrainOpts { head_budget_words: Some(64 * cfg.vocab as u64), ..TrainOpts::default() };
        Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), Qwen35Config { block_size: t, ..cfg.clone() }, 1, t, &init, &opts).gpu.charged_bytes()
    };
    mark("short", at(256));
    mark("long", at(512));
}

/// With 8192 columns, a `tokens x vocab` logits buffer would cost 32 KiB per
/// token; the chunked head's scratch is fixed (64 rows), so the per-token
/// slope of what the build asks for stays far below it.
#[test]
fn a_training_build_never_holds_tokens_times_vocab_logits() {
    let m = measure_in_child("child_charged_bytes_at_two_lengths_with_a_wide_vocab");
    let per_token = (m["long"] - m["short"]) / 256;
    assert!(per_token < 16 * 1024, "{per_token} B per token requested: a tokens x vocab buffer (32 KiB per token) is back");
}

/// A batch supervised only early computes only the tokens up to them (rounded
/// up to whole GDN chunks): everything is causal, so the tail adds no loss and
/// no gradient, and a held-out or training row padded to a 15k-token block costs
/// what its real length costs. The loss and every gradient equal a build sized
/// to the shorter row - also under the long-sequence settings a 27B run uses.
#[test]
fn a_batch_supervised_only_early_runs_only_its_prefix() {
    let (long, short, supervised) = (256usize, 128usize, 100usize);
    let cfg = Qwen35Config { block_size: long as u32, ..long_config(false) };
    let init = weights(&cfg, 0.3);
    let (x, mut y) = batch_of(long);
    y[supervised..].fill(model::IGNORE);
    let settings = [
        TrainOpts::default(),
        TrainOpts { recompute: true, gdn_segment_tokens: Some(64), mlp_rows: Some(64), bf16_residuals: true, ..TrainOpts::default() },
    ];
    for opts in &settings {
        let full = build(&cfg, &init, long, opts);
        let got = loss_and_grads_on(&full, &x, &y);
        assert_eq!(full.run_len(), short as u32, "a row supervised up to {supervised} runs {short} tokens, not {long}");

        let cfg_short = Qwen35Config { block_size: short as u32, ..cfg.clone() };
        let want = loss_and_grads_on(&build(&cfg_short, &init, short, opts), &x[..short], &y[..short]);
        assert_same_training_signal(got, want, 1e-5, 0.99999);

        let (x, y) = batch_of(long);
        loss_and_grads_on(&full, &x, &y);
        assert_eq!(full.run_len(), long as u32, "the next, fully supervised batch runs the whole row again");
    }
}
