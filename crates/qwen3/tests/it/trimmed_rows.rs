// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning pipelines that spend their compute on the tokens that train for its
// clients. If your team needs expertise in efficient supervised fine-tuning then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A training batch computes only as far as its last supervised position (`Qwen::needed_len`): a record padded to a
//! long block trains exactly as the same record in a block of its own length - the same loss, the same adapter
//! gradients - and costs that length. The model is causal and the padding carries no target, so nothing after the
//! last supervised position reaches the loss or the gradient; trimming is skipping work, never changing it.

use std::collections::HashMap;
use std::time::Instant;

use data::rng::Rng;
use gpu_core::provider::library::GemmPrecision;
use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig, IGNORE};

fn config(t: u32) -> QwenConfig {
    let targets = qwen3::finetune::LORA_TARGETS.iter().map(|s| s.to_string()).collect();
    QwenConfig { block_size: t, max_position_embeddings: t, lora: Some(LoraCfg::new(4, 8.0, targets)), ..QwenConfig::tiny() }
}

/// Weights with the adapters' `lora_b` drawn non-zero, so every adapter tensor has a gradient.
fn weights(cfg: &QwenConfig) -> HashMap<String, Vec<f32>> {
    let mut w = qwen3::init_weights(cfg, 3);
    let mut rng = Rng::new(17);
    for (name, v) in w.iter_mut() {
        if name.ends_with(".lora_b") {
            v.iter_mut().for_each(|x| *x = (rng.next_f32() * 2.0 - 1.0) * 0.2);
        }
    }
    w
}

/// `rows` records of the given lengths, each padded to `t`: tokens, and targets supervised on each record's second
/// half (a chat record's answer) and IGNORE everywhere else.
fn padded(rows: &[usize], t: usize, vocab: u32) -> (Vec<u32>, Vec<u32>) {
    let (mut x, mut y) = (vec![0u32; rows.len() * t], vec![IGNORE; rows.len() * t]);
    for (r, &len) in rows.iter().enumerate() {
        for i in 0..len {
            x[r * t + i] = (i as u32 * 13 + r as u32 * 7 + 1) % vocab;
            if i >= len / 2 {
                y[r * t + i] = (i as u32 * 5 + 3) % vocab;
            }
        }
    }
    (x, y)
}

fn step(m: &Qwen, x: &[u32], y: &[u32]) -> (f32, Vec<(String, Vec<f32>)>) {
    m.set_batch(x, y);
    m.zero_grads();
    let loss = m.forward();
    m.backward();
    let grads = model::Model::optimized_params(m).unwrap().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect();
    (loss, grads)
}

fn assert_same(what: &str, (want_loss, want): &(f32, Vec<(String, Vec<f32>)>), (loss, got): &(f32, Vec<(String, Vec<f32>)>)) {
    assert!((loss - want_loss).abs() <= 1e-5 * want_loss.abs().max(1.0), "{what}: loss {loss} vs {want_loss}");
    for ((name, g), (_, w)) in got.iter().zip(want) {
        let scale = w.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = g.iter().zip(w).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst <= 1e-4 * scale, "{what}: {name} differs by {worst} (scale {scale})");
    }
}

/// One record of 100 tokens padded to a 512-token block trains as the record in a 128-token block (the run length
/// it is trimmed to), and computes 128 tokens, not 512.
#[test]
fn a_record_padded_to_a_long_block_trains_as_the_record_alone() {
    let (long, short) = (512u32, 128u32);
    let init = weights(&config(long));
    let build = |t| Qwen::new_lora_dt_gemm(config(t), 1, t, &init, Dtype::F32, GemmPrecision::Fp32Reference);
    let (padded_model, own_model) = (build(long), build(short));
    let (x, y) = padded(&[100], long as usize, padded_model.cfg.vocab);
    let (xs, ys) = padded(&[100], short as usize, own_model.cfg.vocab);
    let trimmed = step(&padded_model, &x, &y);
    assert_eq!(padded_model.run_len(), 128, "trimmed to the record rounded up to whole quanta");
    assert_eq!(model::Model::computed_tokens(&padded_model), Some(128));
    let own = step(&own_model, &xs, &ys);
    assert_eq!(own_model.run_len(), short, "a block the record fills is not trimmed");
    assert_same("one row", &own, &trimmed);
    // A batch that does supervise to the end of the block runs the whole block.
    let (xf, mut yf) = padded(&[100], long as usize, padded_model.cfg.vocab);
    yf[long as usize - 1] = 5;
    step(&padded_model, &xf, &yf);
    assert_eq!(padded_model.run_len(), long);
}

/// Two records of different lengths in one batch run at the longer one's length, each row laid out at it, and train
/// as the same two records in blocks of that length; the per-position log-probabilities come back in the caller's
/// `t`-long rows.
#[test]
fn a_batch_of_rows_runs_at_its_longest_record_and_reads_back_in_its_own_layout() {
    let (long, short) = (384u32, 192u32);
    let init = weights(&config(long));
    let build = |t| Qwen::new_lora_dt_gemm(config(t), 2, t, &init, Dtype::F32, GemmPrecision::Fp32Reference);
    let (padded_model, own_model) = (build(long), build(short));
    let (x, y) = padded(&[150, 70], long as usize, padded_model.cfg.vocab);
    let (xs, ys) = padded(&[150, 70], short as usize, own_model.cfg.vocab);
    let trimmed = step(&padded_model, &x, &y);
    assert_eq!(padded_model.run_len(), 192);
    let logprobs = padded_model.batch_token_logprobs();
    let own = step(&own_model, &xs, &ys);
    assert_same("two rows", &own, &trimmed);
    let own_logprobs = own_model.batch_token_logprobs();
    assert_eq!(logprobs.len(), 2 * long as usize, "one value per position of the caller's rows");
    for r in 0..2 {
        let (a, b) = (&logprobs[r * long as usize..r * long as usize + short as usize], &own_logprobs[r * short as usize..(r + 1) * short as usize]);
        assert!(a.iter().zip(b).all(|(p, q)| (p - q).abs() <= 1e-5), "row {r}: the computed positions agree");
        assert!(logprobs[r * long as usize + short as usize..(r + 1) * long as usize].iter().all(|&p| p == 0.0), "row {r}: the trimmed tail reads as unsupervised");
    }
}

/// Useful (supervised-record) tokens a second of LoRA steps over records of about a fifth of the block, trimmed
/// against the whole block computed: the CPU-backend measurement the roadmap quotes.
#[test]
#[ignore = "a measurement: prints useful tokens a second untrimmed and trimmed"]
fn measure_useful_tokens_a_second_of_padded_records() {
    let t = 1024u32;
    let cfg = QwenConfig { n_layers: 4, d_model: 128, n_heads: 4, n_kv_heads: 2, head_dim: 32, d_ff: 384, vocab: 2048, ..config(t) };
    let init = weights(&cfg);
    let m = Qwen::new_lora_dt_gemm(cfg.clone(), 1, t, &init, Dtype::F32, GemmPrecision::Auto);
    let lengths = [180usize, 240, 150, 210, 200, 230, 170, 220];
    let useful: usize = lengths.iter().sum();
    let timed = |supervise_to_end: bool| {
        let start = Instant::now();
        for &len in &lengths {
            let (x, mut y) = padded(&[len], t as usize, cfg.vocab);
            if supervise_to_end {
                // The untrimmed cost: the same shapes as a batch computed over the whole block.
                y[t as usize - 1] = 1;
            }
            step(&m, &x, &y);
            m.adamw_step(1, 1e-4, 0.0, model::Adam::default(), Some(1.0), 1.0);
        }
        m.poll_wait();
        useful as f64 / start.elapsed().as_secs_f64()
    };
    timed(false);
    let (whole, trimmed) = (timed(true), timed(false));
    eprintln!("useful tokens/s over records of ~{} tokens in a {t}-token block: whole block {whole:.0}, trimmed {trimmed:.0} ({:.2}x)", useful / lengths.len(), trimmed / whole);
}

/// A held-out record is scored on the device: each position's loss and the model's own pick agree with what the full
/// logits read back to the host give, so held-out scoring needs two numbers a position, not the logits.
#[test]
fn a_record_scored_on_the_device_matches_its_logits_read_to_the_host() {
    let cfg = QwenConfig { vocab: 3000, ..config(96) };
    let init = qwen3::init_weights(&cfg, 9);
    let shard = qwen3::Shard::whole(cfg.n_layers as usize);
    let m = Qwen::new_shard_dt(cfg.clone(), 1, 96, &init, shard, Dtype::F32);
    let ids: Vec<u32> = (0..80u32).map(|i| (i * 37 + 11) % cfg.vocab).collect();
    let targets: Vec<u32> = (0..ids.len()).map(|i| if (30..79).contains(&i) { ids[i + 1] } else { IGNORE }).collect();
    let (nll, picked) = m.score_positions(&ids, &targets);
    let logits = m.logits_all(&ids);
    let v = cfg.vocab as usize;
    for (i, &target) in targets.iter().enumerate() {
        let row = &logits[i * v..(i + 1) * v];
        let host_pick = row.iter().enumerate().fold(0, |best, (j, &x)| if x > row[best] { j } else { best }) as u32;
        assert_eq!(picked[i], host_pick, "position {i}: the device's pick is the logits' argmax");
        if target == IGNORE {
            assert_eq!(nll[i], 0.0, "position {i}: an unsupervised position scores nothing");
        } else {
            let host = -model::logprobs::row_logprob(row, target as usize);
            assert!((nll[i] - host).abs() <= 1e-4 * host.abs().max(1.0), "position {i}: device loss {} vs host {host}", nll[i]);
        }
    }
}

/// Seconds to score a long record through the logits read to the host and through the device's reduction, at a
/// Qwen3-sized vocabulary.
#[test]
#[ignore = "a measurement: prints the seconds of host and device scoring of one long record"]
fn measure_scoring_a_long_record_on_the_device_and_through_the_host() {
    let cfg = QwenConfig { vocab: 151_936, d_model: 64, n_heads: 2, n_kv_heads: 1, head_dim: 32, d_ff: 128, ..config(2048) };
    let init = qwen3::init_weights(&cfg, 9);
    let m = Qwen::new_shard_dt(cfg.clone(), 1, 2048, &init, qwen3::Shard::whole(cfg.n_layers as usize), Dtype::F32);
    let ids: Vec<u32> = (0..2048u32).map(|i| (i * 37 + 11) % cfg.vocab).collect();
    let targets: Vec<u32> = (0..ids.len()).map(|i| if i + 1 < ids.len() { ids[i + 1] } else { IGNORE }).collect();
    m.score_positions(&ids, &targets);
    let start = Instant::now();
    let logits = m.logits_all(&ids);
    let v = cfg.vocab as usize;
    let host: f64 = (0..ids.len() - 1).map(|i| -model::logprobs::row_logprob(&logits[i * v..(i + 1) * v], ids[i + 1] as usize) as f64).sum();
    let host_secs = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let device: f64 = m.score_positions(&ids, &targets).0.iter().map(|&l| l as f64).sum();
    let device_secs = start.elapsed().as_secs_f64();
    eprintln!("2048-token record, 151936 vocabulary: logits to the host {host_secs:.2} s (loss {host:.1}), on the device {device_secs:.2} s (loss {device:.1})");
}
