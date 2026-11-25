// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that teach speech
// synthesizers new languages and voices at the speed of the hardware for its
// clients. If your team needs expertise in speech-model fine-tuning
// throughput, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Throughput of the speech fine-tune (`qwen3tts::speech_sft`, the function the
//! served `speech_sft` job runs) on a synthetic shard with a real corpus's
//! length distribution. Throughput does not depend on the values of the codes
//! or the x-vectors, so they are random; the lengths are what matter.
//!
//! ```text
//! speech_sft_bench synth OUT.bttshrd --records N [--seed S] [--d 2048]
//! speech_sft_bench run --base DIR --shard PATH [--config '{"steps": 20, ...}'] [--card N]
//! speech_sft_bench ranks --ranks N --base DIR --shard PATH [--config '{...}']
//! ```
//!
//! `synth` draws utterances of 1.5-30 s (log-normal, mean about 6.4 s, 12.5
//! frames a second) with about four text tokens a second, every one complete,
//! in one language (`english`). `run` trains with the given
//! `SpeechSftConfig` fields over defaults (20 steps, the base's `english` row)
//! and prints each step's frames per second and the mean past the first
//! steps, and the peak of the memory the run takes on card `N` (default 0, as
//! the driver reports it).

use std::path::Path;

use data::rng::Rng;
use qwen3tts::shard::{Record, RecordId, Shard, ShardMeta, ShardWriter, FORMAT, NO_VOICE};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn synth(args: &[String]) -> Result<(), String> {
    let out = args.first().ok_or("synth: name the output shard")?;
    let records: usize = arg(args, "--records").ok_or("synth: --records N")?.parse().map_err(|e| format!("--records: {e}"))?;
    let seed: u64 = arg(args, "--seed").map_or(Ok(data::rng::random_seed()), |s| s.parse().map_err(|e| format!("--seed: {e}")))?;
    let d: u32 = arg(args, "--d").map_or(Ok(2048), |s| s.parse().map_err(|e| format!("--d: {e}")))?;
    let meta = ShardMeta { format: FORMAT.into(), codebooks: 16, codebook_size: 2048, xvector_dim: d, frame_rate: 12.5, languages: vec!["english".into()], voices: Vec::new(), records: 0, producers: Default::default() };
    let mut w = ShardWriter::new(meta);
    let mut rng = Rng::new(seed);
    let (mut frames_total, mut text_total) = (0usize, 0usize);
    for r in 0..records {
        // ln(seconds) ~ N(1.71, 0.55): median 5.5 s, mean about 6.4 s (80 frames).
        let secs = (1.71 + 0.55 * rng.next_gaussian()).exp().clamp(1.5, 30.0);
        let frames = (secs * 12.5).round() as usize;
        let text = ((secs * 4.0).round() as usize).max(2);
        // Three role tokens, the text, five trailing template tokens: the generation tokenisation's shape.
        let ids: Vec<u32> = (0..3 + text + 5).map(|_| 1000 + (rng.next_u64() % 100_000) as u32).collect();
        let codes: Vec<u16> = (0..frames * 16).map(|_| (rng.next_u64() % 2048) as u16).collect();
        let xvector: Vec<f32> = (0..d).map(|_| rng.next_gaussian() as f32 * 0.5).collect();
        w.push(&Record { id: RecordId { id: format!("u{r}"), speaker: format!("s{}", r % 64), source: "synthetic".into(), text: String::new() }, text_ids: &ids, codes: &codes, xvector: &xvector, voice: NO_VOICE, language: 0, complete: true })?;
        frames_total += frames;
        text_total += text;
    }
    w.finish(Path::new(out)).map_err(|e| format!("{out}: {e}"))?;
    println!("{out}: {records} records, {frames_total} frames ({:.1} frames, {:.1} text tokens a record, {:.2} h), seed {seed}", frames_total as f64 / records as f64, text_total as f64 / records as f64, frames_total as f64 / 12.5 / 3600.0);
    Ok(())
}

fn run(args: &[String]) -> Result<(), String> {
    let base = arg(args, "--base").ok_or("run: --base DIR")?;
    let shard_path = arg(args, "--shard").ok_or("run: --shard PATH")?;
    let shard = Shard::open(Path::new(&shard_path)).map_err(|e| format!("{shard_path}: {e}"))?;
    let mut cfg = serde_json::json!({ "base": base, "steps": 20, "rows": {"languages": {"english": "base"}} });
    if let Some(over) = arg(args, "--config") {
        let over: serde_json::Value = serde_json::from_str(&over).map_err(|e| format!("--config: {e}"))?;
        for (k, v) in over.as_object().ok_or("--config: an object")? {
            cfg[k] = v.clone();
        }
    }
    let cfg: qwen3tts::speech_sft::SpeechSftConfig = serde_json::from_value(cfg).map_err(|e| format!("config: {e}"))?;
    let t0 = std::time::Instant::now();
    let mut last = t0;
    let mut walls = Vec::new();
    let mut on_step = |r: &model::train::StepReport| {
        let now = std::time::Instant::now();
        let frames = r.metrics.iter().find(|(n, _)| *n == "frames").map_or(0.0, |(_, v)| *v);
        let wall = (now - last).as_secs_f64();
        println!("step {:4} loss {:.4} frames {:6} {:8.1} frames/s (step) {:8.1} frames/s (wall, {wall:.2} s)", r.step, r.loss, frames, r.work.useful_tokens_per_s(), f64::from(frames) / wall);
        walls.push((frames, wall));
        last = now;
        true
    };
    let control = model::train::FitControl { on_step: Some(&mut on_step), ..model::train::FitControl::default() };
    // The card's memory peak, sampled from the driver while the run lasts (the card the run is placed on).
    let card: u32 = arg(args, "--card").map_or(Ok(0), |s| s.parse().map_err(|e| format!("--card: {e}")))?;
    let baseline = gpu_core::capacity::free_on(card).unwrap_or(0);
    let (r, peak) = gpu_core::capacity::peak_taken_while(card, baseline, || qwen3tts::speech_sft::speech_sft(&cfg, &shard, None, control));
    let r = r?;
    let steps = &r.report.steps;
    let skip = (steps.len() / 5).max(1).min(steps.len().saturating_sub(1));
    let tail = &steps[skip..];
    let mean = tail.iter().map(|s| s.frames_per_s).sum::<f64>() / tail.len().max(1) as f64;
    let (frames, wall) = walls[skip..].iter().fold((0f64, 0f64), |(f, w), &(sf, sw)| (f + f64::from(sf), w + sw));
    println!("{} steps, mean {mean:.1} frames/s over steps {}..{} ({:.1} by the wall clock), device peak {:.1} GB, {:.1} s total", steps.len(), skip + 1, steps.len(), frames / wall, peak as f64 / 1e9, t0.elapsed().as_secs_f64());
    Ok(())
}

/// `run` as `ranks` data-parallel ranks of one process, rank `r` on card `r`: every rank trains its shard of the records
/// with the same seed, and each step's gradients are summed over the group (on the cards over NCCL where every rank
/// can, through the host otherwise, as the run says) before the optimizer step. Prints each rank's mean and the
/// group's frames a second (every rank's frames over the slowest rank's wall clock).
fn ranks(args: &[String]) -> Result<(), String> {
    use model::{Collective, CollectivePath, HostCollective, JobCollective};
    use std::sync::Arc;
    let world: usize = arg(args, "--ranks").ok_or("ranks: --ranks N")?.parse().map_err(|e| format!("--ranks: {e}"))?;
    let base = arg(args, "--base").ok_or("ranks: --base DIR")?;
    let shard_path = arg(args, "--shard").ok_or("ranks: --shard PATH")?;
    let mut cfg = serde_json::json!({ "base": base, "steps": 20, "rows": {"languages": {"english": "base"}} });
    if let Some(over) = arg(args, "--config") {
        let over: serde_json::Value = serde_json::from_str(&over).map_err(|e| format!("--config: {e}"))?;
        for (k, v) in over.as_object().ok_or("--config: an object")? {
            cfg[k] = v.clone();
        }
    }
    // Every rank draws the same initialisation and batch order; each trains its own records of it.
    if cfg.get("seed").is_none() {
        cfg["seed"] = serde_json::json!(data::rng::random_seed());
    }
    let cfg: qwen3tts::speech_sft::SpeechSftConfig = serde_json::from_value(cfg).map_err(|e| format!("config: {e}"))?;
    let host = HostCollective::new(world);
    let group: Vec<Arc<dyn Collective>> = (0..world)
        .map(|_| Arc::new(JobCollective::new(Arc::clone(&host), CollectivePath::Auto, std::time::Duration::from_secs(120), std::time::Duration::from_secs(600))) as Arc<dyn Collective>)
        .collect();
    let t0 = std::time::Instant::now();
    let results: Vec<Result<(f64, f64, f64), String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = group
            .iter()
            .enumerate()
            .map(|(rank, collective)| {
                let (cfg, shard_path) = (&cfg, &shard_path);
                scope.spawn(move || -> Result<(f64, f64, f64), String> {
                    let shard = Shard::open(Path::new(shard_path)).map_err(|e| format!("{shard_path}: {e}"))?;
                    let (mut frames, mut first) = (0f64, None);
                    let mut on_step = |r: &model::train::StepReport| {
                        if r.step >= 4 {
                            first.get_or_insert_with(std::time::Instant::now);
                            frames += r.metrics.iter().find(|(n, _)| *n == "frames").map_or(0.0, |(_, v)| f64::from(*v));
                        }
                        true
                    };
                    let sync = model::GradSync { collective: &**collective, rank, checkpoint: None, observer: None };
                    let control = model::train::FitControl { on_step: Some(&mut on_step), grad_sync: Some(sync), ..model::train::FitControl::default() };
                    let r = gpu_core::devices::with_gpu(rank as u32, || qwen3tts::speech_sft::speech_sft(cfg, &shard, None, control)).map_err(|e| format!("rank {rank}: {e}"))??;
                    let steps = &r.report.steps;
                    let tail = &steps[steps.len().min(4)..];
                    let mean = tail.iter().map(|s| s.frames_per_s).sum::<f64>() / tail.len().max(1) as f64;
                    let wall = first.map_or(0.0, |t: std::time::Instant| t.elapsed().as_secs_f64());
                    Ok((mean, frames, wall))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("a rank panicked".into()))).collect()
    });
    let mut total = 0f64;
    let mut slowest = 0f64;
    for (rank, r) in results.iter().enumerate() {
        let (mean, frames, wall) = r.clone()?;
        println!("rank {rank}: mean {mean:.1} frames/s over its steps past the third, {frames:.0} frames in {wall:.1} s");
        total += frames;
        slowest = slowest.max(wall);
    }
    println!("{world} ranks: {:.1} frames/s together (every rank's frames over the slowest rank's wall clock), {:.1} s total", total / slowest.max(1e-9), t0.elapsed().as_secs_f64());
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("synth") => synth(&args[1..]),
        Some("run") => run(&args[1..]),
        Some("ranks") => ranks(&args[1..]),
        _ => Err("usage: speech_sft_bench synth OUT --records N [--seed S] [--d D] | run --base DIR --shard PATH [--config JSON]".into()),
    };
    if let Err(e) = result {
        eprintln!("speech_sft_bench: {e}");
        std::process::exit(1);
    }
}
