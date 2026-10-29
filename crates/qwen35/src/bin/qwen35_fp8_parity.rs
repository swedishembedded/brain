// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for validating FP8 serving of released language models against a higher
// precision reference for its clients. If your team needs expertise in measuring what native FP8 kernels change in a
// model's output you can procure our services by sending an email to info@swedishembedded.com.

//! What serving the released FP8 checkpoint natively at FP8 changes against the same weights decoded to bf16, on the real
//! prompts, in ONE process.
//!
//! The reference holds every FP8 linear as bf16 (`value = decode(byte) * block scale` rounded to bf16, a vendor-library
//! bf16 GEMM): the arithmetic a bf16 server of this checkpoint runs. The candidate holds them as stored (FP8 bytes with
//! block scales on the device, the native kernels the tuner picked). Both score the SAME token stream, the reference's
//! own greedy continuation of each prompt, so the logits can be compared row for row: mean and worst relative L2 distance,
//! the positions that pick another argmax with the reference margin over its runner-up (a flip at a large margin is a
//! defect, a flip at a near-tie is numeric drift), and the mean KL in nats.
//!
//! The reference is loaded and measured first, dropped, then the candidate is loaded: the two never share a card, so
//! a 27B fits. Usage: `qwen35_fp8_parity <hf-dir> <text file> [--prompts N] [--prompt-tokens N] [--steps N]
//! [--device gpu1] [--backend cuda] [--kv int8] [--assert-agreement F]`.

use model::ops::TierPolicy;
use qwen35::int8_gguf_resident::{budgeted_gpus, Qwen35GgufInstance, Qwen35GgufResident};
use qwen35::resident_checkpoint::ResidentCheckpoint;
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

const RESERVE: u64 = 2 << 30;

/// The served rendering of one user turn with thinking off.
fn render(body: &str) -> String {
    format!("<|im_start|>user\n{body}\n\nSummarise the code above in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

/// The prompt-th source window of the text file, `ptoks` tokens' worth of characters.
fn window(text: &str, p: usize, ptoks: usize) -> &str {
    let chars_per_token = 3usize;
    let start = ((p * 7919 + 13) * 104729) % text.len().saturating_sub(ptoks * chars_per_token).max(1);
    let mut slice = start;
    while !text.is_char_boundary(slice) {
        slice += 1;
    }
    let mut end = (slice + ptoks * chars_per_token).min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[slice..end]
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0)
}

fn margin(v: &[f32]) -> f32 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| b.total_cmp(a));
    s[0] - s[1]
}

/// KL(softmax(p) || softmax(q)) in nats: how far `q` is from the reference distribution `p`.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let lse = |v: &[f32]| {
        let m = v.iter().cloned().fold(f32::MIN, f32::max) as f64;
        m + v.iter().map(|x| ((*x as f64) - m).exp()).sum::<f64>().ln()
    };
    let (lp, lq) = (lse(p), lse(q));
    p.iter()
        .zip(q)
        .map(|(a, b)| {
            let pa = ((*a as f64) - lp).exp();
            if pa < 1e-12 { 0.0 } else { pa * (((*a as f64) - lp) - ((*b as f64) - lq)) }
        })
        .sum()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += ((*x - *y) as f64).powi(2);
        n += (*y as f64).powi(2);
    }
    (d / n.max(1e-30)).sqrt()
}

fn load(path: &str, devices: Vec<(Device, u64)>, cap: u32, tier: TierPolicy, kv: model::kv_tier::KvTier) -> Qwen35GgufInstance {
    let r = Qwen35GgufResident::new(path.to_string(), devices, cap, tier).with_kv_tier(kv);
    let placed: Vec<Device> = r.estimate_multi(&r.instance_key("generate", &capability::Invocation::new())).devices().collect();
    r.activate_owned(&placed).unwrap_or_else(|e| panic!("activate {path}: {e}"))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let flag = |name: &str| a.iter().position(|arg| arg == name).and_then(|i| a.get(i + 1)).cloned();
    let num = |name: &str, default: usize| flag(name).and_then(|v| v.parse().ok()).unwrap_or(default);
    let (Some(path), Some(text_path)) = (a.get(1).cloned(), a.get(2).cloned()) else {
        eprintln!("usage: qwen35_fp8_parity <hf-dir> <text file> [--prompts N] [--prompt-tokens N] [--steps N] [--device D] [--backend B] [--kv K] [--assert-agreement F]");
        std::process::exit(2);
    };
    let (prompts, ptoks, steps) = (num("--prompts", 4), num("--prompt-tokens", 2000), num("--steps", 24));
    let set = gpu_core::select_compute(flag("--device").as_deref().unwrap_or("gpu"), flag("--backend").as_deref()).unwrap_or_else(|e| panic!("{e}"));
    let devices = budgeted_gpus(&set.gpus, RESERVE);
    if devices.is_empty() {
        eprintln!("no GPU with queryable VRAM - this resident is GPU-only");
        std::process::exit(2);
    }
    let kv = model::kv_tier::KvTier::parse(flag("--kv").as_deref().unwrap_or("int8")).unwrap_or_else(|e| panic!("{e}"));
    let cap = (ptoks + steps + 256) as u32;
    let text = std::fs::read_to_string(&text_path).expect("read the text file");

    let ckpt = ResidentCheckpoint::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let held = ckpt.held_tiers(&ckpt.config(1).expect("config")).expect("tiers");
    drop(ckpt);

    // The reference: the same weights as bf16, one prompt's greedy continuation at a time.
    let mut prompt_ids: Vec<Vec<u32>> = Vec::new();
    let mut reference: Vec<(Vec<u32>, Vec<Vec<f32>>)> = Vec::new();
    {
        let inst = load(&path, devices.clone(), cap, TierPolicy::uniform(gpu_core::select::Dtype::BF16), kv);
        for p in 0..prompts {
            let prompt: Vec<u32> = inst.tokenize(&render(window(&text, p, ptoks))).into_iter().take(cap as usize - steps - 8).collect();
            let trace = inst.tape_comparison_trace(&prompt, steps as u32).expect("greedy trace");
            println!("prompt {p}: {} tokens; reference greedy continuation {:?}", prompt.len(), trace.ids);
            reference.push((trace.ids, trace.decode));
            prompt_ids.push(prompt);
        }
    }

    let inst = load(&path, devices, cap, held, kv);
    let (mut agree, mut positions, mut big_flips) = (0usize, 0usize, 0usize);
    let (mut sum_rel, mut worst_rel, mut sum_kl) = (0f64, 0f64, 0f64);
    for (p, (ids, ref_logits)) in reference.iter().enumerate() {
        let got = inst.teacher_forced_logits(&prompt_ids[p], ids).expect("teacher-forced logits");
        for (i, (g, r)) in got.iter().zip(ref_logits).enumerate() {
            let (ga, ra) = (argmax(g), argmax(r));
            let rel = rel_l2(g, r);
            sum_rel += rel;
            worst_rel = worst_rel.max(rel);
            sum_kl += kl(r, g);
            positions += 1;
            if ga == ra {
                agree += 1;
            } else {
                let m = margin(r);
                // A near tie: the other token is within a hundredth of the reference's top logit gap scale.
                if m > 0.5 {
                    big_flips += 1;
                }
                println!("  prompt {p} step {i}: argmax {ra} -> {ga}, reference margin {m:.3}");
            }
        }
    }
    let agreement = agree as f64 / positions as f64;
    println!(
        "fp8 vs bf16-decoded: {positions} positions, greedy agreement {:.2} %, mean rel-L2 {:.3e}, worst rel-L2 {worst_rel:.3e}, mean KL {:.3e} nats, flips at margin > 0.5: {big_flips}",
        100.0 * agreement,
        sum_rel / positions as f64,
        sum_kl / positions as f64
    );
    if let Some(min) = flag("--assert-agreement").and_then(|v| v.parse::<f64>().ok()) {
        assert!(agreement >= min, "greedy agreement {agreement:.4} is under the asserted {min}");
    }
}
