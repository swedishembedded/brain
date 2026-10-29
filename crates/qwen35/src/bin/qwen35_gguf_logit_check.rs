// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for attributing a numerical change in a quantised LLM serving stack
// to the kernel that caused it, for its clients. If your team needs expertise in measuring logit drift between
// inference kernels you can procure our services by sending an email to info@swedishembedded.com.

//! Which of the prefill-path changes moves the real 27B's logits, and by how much, in ONE process on ONE load.
//!
//! The released Q8_0 GGUF scores the same greedy token stream (taken under the first setting) under settings of the
//! serving prefill: the portable fp32 attention kernel (the reference), the original fp16 flash kernel, version 2 of
//! it, then the fused front end and quantisation epilogues, then no per-layer drain. For each setting it prints,
//! against the first: the mean and worst relative L2 distance of the logits, the positions that pick another argmax
//! with the reference margin over its runner-up (a flip at a large margin is a defect, a flip at a near-tie is
//! numeric drift), and whether the logits are bit-identical.
//!
//! Usage: `qwen35_gguf_logit_check <Q8_0.gguf> <text file> [--prompts 4] [--prompt-tokens 4000] [--steps 24]
//!        [--device gpu1] [--backend cuda] [--kv int8] [--dump DIR] [--dry-run] [--raw] [--assert-within-floor M]
//!        [--batched N] [--batched-only] [--kernel SLOT=CANDIDATE]...`
//!
//! The prompt is the served chat rendering (user turn, then the assistant turn with an empty think block, as the
//! server renders `enable_thinking: false`), and the continuation is the greedy decode-tape trace of
//! `tape_comparison_trace`, which runs a fixed number of steps and does NOT stop at an end-of-sequence token: the
//! tool fails loudly if the trace is shorter than `--steps`. `--dry-run` tokenizes on the CPU and prints the
//! rendered prompt's token structure without touching a device. `--raw` renders the plain text without the chat
//! markup (the prompt whose first greedy token can be an end-of-sequence token).
//!
//! `--batched N` (N at most 8, at most `--prompts`) afterwards decodes the first N prompts' greedy streams TOGETHER, one
//! batched decode step per token (the path a server's decode batch runs, whose projections use the batch-sized GEMV
//! kernels), and reports each stream's logits against the same stream decoded alone - mean rel-L2 (also in floors),
//! mean KL and the argmax flips with their margins. `--assert-within-floor M` bounds it like the settings above.
//! `--batched-only` skips the settings comparison (the numerics floor is still measured).
//!
//! `--dump DIR` writes each prompt's continuation ids and the logits rows of every setting (`p<N>.ids`,
//! `p<N>.s<K>.f32`, plus `p<N>.first.f32` and `p<N>.last.f32`, little-endian) so a build of an older commit can be compared offline.

use checkpoint::gguf::MmapGguf;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use qwen35::int8_gguf_resident::{resident_config, Qwen35GgufInstance, Qwen35GgufResident};
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

const RESERVE: u64 = 2 << 30;

/// The served rendering of one user turn with thinking off.
fn render(body: &str, raw: bool) -> String {
    if raw {
        return format!("{body}\n\nSummarise the code above in one sentence.\n");
    }
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

fn dry_run(path: &str, text: &str, prompts: usize, ptoks: usize, raw: bool) {
    let mg = MmapGguf::open(path).unwrap_or_else(|e| panic!("open the checkpoint: {e}"));
    let gtok = mg.tokenizer().expect("the checkpoint carries a tokenizer");
    let tok = QwenBpe::from_gguf(&gtok).unwrap_or_else(|e| panic!("embedded tokenizer: {e}"));
    for name in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"] {
        println!("special {name}: {:?}", tok.special_id(name));
    }
    println!("declared eos: {:?}", gtok.eos);
    let start = tok.special_id("<|im_start|>").expect("<|im_start|> is a special token");
    for p in 0..prompts {
        let ids = tok.encode(&render(window(text, p, ptoks), raw));
        assert!(raw || ids.first() == Some(&start), "prompt {p} does not open with <|im_start|>: the template is not tokenized as markup");
        let tail: Vec<u32> = ids[ids.len().saturating_sub(10)..].to_vec();
        println!("prompt {p}: {} tokens, tail {tail:?} = {:?}", ids.len(), tok.decode(&tail));
    }
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
    p.iter().zip(q).map(|(a, b)| { let pa = ((*a as f64) - lp).exp(); if pa < 1e-12 { 0.0 } else { pa * (((*a as f64) - lp) - ((*b as f64) - lq)) } }).sum()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += ((*x - *y) as f64).powi(2);
        n += (*y as f64).powi(2);
    }
    (d / n.max(1e-30)).sqrt()
}

struct Setting {
    name: &'static str,
    portable: bool,
    v2: bool,
    fusion: bool,
    drain: bool,
}

fn apply(inst: &Qwen35GgufInstance, s: &Setting) {
    gpu_core::provider::cuda::set_flash_prefill_portable(s.portable);
    gpu_core::provider::cuda::set_flash_prefill_v2(s.v2);
    inst.set_decode_fusion(s.fusion);
    inst.set_chunk_arena_min_rows(1);
    inst.set_chunk_arena_drain(Some(s.drain));
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let flag = |name: &str| a.iter().position(|arg| arg == name).and_then(|i| a.get(i + 1)).cloned();
    let num = |name: &str, default: usize| flag(name).and_then(|v| v.parse().ok()).unwrap_or(default);
    let (Some(path), Some(text_path)) = (a.get(1).cloned(), a.get(2).cloned()) else {
        eprintln!("usage: qwen35_gguf_logit_check <Q8_0.gguf> <text file> [--prompts N] [--prompt-tokens N] [--steps N] [--device D] [--backend B] [--kv K]");
        std::process::exit(2);
    };
    let (prompts, ptoks, steps) = (num("--prompts", 4), num("--prompt-tokens", 4000), num("--steps", 24));
    let raw = a.iter().any(|x| x == "--raw");
    // `--kernel <slot>=<candidate>` (repeatable) pins a kernel slot to one candidate, as `brain --kernel` does: the
    // model-level quality check of a candidate the autotuner would not choose on this device.
    for (i, _) in a.iter().enumerate().filter(|(_, x)| *x == "--kernel") {
        match a.get(i + 1).and_then(|v| v.split_once('=')).filter(|(s, c)| !s.is_empty() && !c.is_empty()) {
            Some((slot, candidate)) => gpu_core::force_candidate(slot, candidate),
            None => {
                eprintln!("--kernel takes <slot>=<candidate>, e.g. --kernel matmul_i8_prefill=matmul_i8_dq_stream_q2");
                std::process::exit(2);
            }
        }
    }
    if a.iter().any(|x| x == "--dry-run") {
        dry_run(&path, &std::fs::read_to_string(&text_path).expect("read the text file"), prompts, ptoks, raw);
        return;
    }
    let set = gpu_core::select_compute(flag("--device").as_deref().unwrap_or("gpu"), flag("--backend").as_deref()).unwrap_or_else(|e| panic!("{e}"));
    let devices = qwen35::int8_gguf_resident::budgeted_gpus(&set.gpus, RESERVE);
    if devices.is_empty() {
        eprintln!("no GPU with queryable VRAM - this resident is GPU-only");
        std::process::exit(2);
    }
    let cap = (ptoks + steps + 256) as u32;
    let tier = model::ops::TierPolicy::uniform(gpu_core::select::Dtype::I8);
    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open the checkpoint: {e}"));
    let _cfg = resident_config(&mg, cap).expect("resident_config");
    drop(mg);
    let kv = model::kv_tier::KvTier::parse(flag("--kv").as_deref().unwrap_or("int8")).unwrap_or_else(|e| panic!("{e}"));
    let batched = flag("--batched").and_then(|v| v.parse::<u32>().ok());
    let r = Qwen35GgufResident::new(path, devices, cap, tier).with_kv_tier(kv).with_max_batch(batched.unwrap_or(1));
    let placed: Vec<Device> = r.estimate_multi(&r.instance_key("generate", &capability::Invocation::new())).devices().collect();
    let inst = r.activate_owned(&placed).expect("activate the real checkpoint");

    let text = std::fs::read_to_string(&text_path).expect("read the text file");
    let settings = [
        Setting { name: "portable fp32 attention (reference)", portable: true, v2: false, fusion: false, drain: true },
        Setting { name: "flash v1 (the original kernel)", portable: false, v2: false, fusion: false, drain: true },
        Setting { name: "flash v2", portable: false, v2: true, fusion: false, drain: true },
        Setting { name: "flash v2 + fused front end", portable: false, v2: true, fusion: true, drain: true },
        Setting { name: "flash v2 + fused + no drain (the default)", portable: false, v2: true, fusion: true, drain: false },
    ];

    let mut floors: Vec<f64> = Vec::new();
    let mut means: Vec<Vec<f64>> = vec![Vec::new(); settings.len()];
    for p in 0..prompts {
        let prompt = inst.tokenize(&render(window(&text, p, ptoks), raw));
        let prompt: Vec<u32> = prompt.into_iter().take(cap as usize - steps - 8).collect();
        apply(&inst, &settings[0]);
        let trace = inst.tape_comparison_trace(&prompt, steps as u32).expect("greedy trace");
        let floor = trace.decode.iter().zip(&trace.chunk).map(|(x, y)| rel_l2(x, y)).sum::<f64>() / steps as f64;
        floors.push(floor);
        let ids = trace.ids;
        assert_eq!(ids.len(), steps, "prompt {p}: the greedy trace has {} tokens, {steps} were asked for: refusing to compare over fewer", ids.len());
        println!("prompt {p}: {} tokens; greedy continuation {:?}", prompt.len(), ids);
        println!("  numerics floor (this model's own decode tape vs its chunk tape, same kernels): mean rel-L2 {floor:.2e}");
        if a.iter().any(|x| x == "--batched-only") {
            continue;
        }
        let mut base: Option<Vec<Vec<f32>>> = None;
        for (si, s) in settings.iter().enumerate() {
            apply(&inst, s);
            let rows = inst.teacher_forced_logits(&prompt, &ids).expect("forced logits");
            assert_eq!(rows.len(), steps, "{}: {} logit rows for {steps} forced tokens", s.name, rows.len());
            assert!(rows.iter().all(|r| r.iter().all(|x| x.is_finite())), "{}: non-finite logits", s.name);
            match &base {
                None => println!("  {:<38} reference", s.name),
                Some(b) => {
                    let rels: Vec<f64> = rows.iter().zip(b).map(|(x, y)| rel_l2(x, y)).collect();
                    means[si].push(rels.iter().sum::<f64>() / rels.len() as f64);
                    let identical = rows.iter().zip(b).all(|(x, y)| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()));
                    let flips: Vec<String> = rows
                        .iter()
                        .zip(b)
                        .enumerate()
                        .filter(|(_, (x, y))| argmax(x) != argmax(y))
                        .map(|(i, (_, y))| format!("step {i} (orig margin {:.3})", margin(y)))
                        .collect();
                    let mean_kl = rows.iter().zip(b).map(|(x, y)| kl(y, x)).sum::<f64>() / rows.len() as f64;
                    println!(
                        "  {:<38} identical={identical} step-0 rel-L2 {:.2e} mean rel-L2 {:.2e} worst {:.2e} mean KL {mean_kl:.2e}; argmax flips vs orig: {}",
                        s.name,
                        rels[0],
                        rels.iter().sum::<f64>() / rels.len() as f64,
                        rels.iter().cloned().fold(0.0, f64::max),
                        if flips.is_empty() { "none".to_string() } else { flips.join(", ") }
                    );
                }
            }
            if let Some(dir) = flag("--dump") {
                std::fs::create_dir_all(&dir).expect("dump dir");
                let bytes: Vec<u8> = rows.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
                let last = si + 1 == settings.len();
                for name in [format!("s{si}"), if si == 0 { "first".into() } else { String::new() }, if last { "last".into() } else { String::new() }] {
                    if !name.is_empty() {
                        std::fs::write(format!("{dir}/p{p}.{name}.f32"), &bytes).expect("write logits");
                    }
                }
                std::fs::write(format!("{dir}/p{p}.ids"), ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")).expect("write ids");
            }
            if base.is_none() {
                base = Some(rows);
            }
        }
    }
    let floor = floors.iter().sum::<f64>() / floors.len() as f64;
    if let Some(nb) = flag("--batched").and_then(|v| v.parse::<usize>().ok()) {
        assert!((1..=prompts).contains(&nb), "--batched {nb} must be between 1 and --prompts {prompts}");
        apply(&inst, settings.last().expect("settings"));
        let mut toks: Vec<Vec<u32>> = Vec::new();
        let mut forced: Vec<Vec<u32>> = Vec::new();
        for p in 0..nb {
            let prompt = inst.tokenize(&render(window(&text, p, ptoks), raw));
            let prompt: Vec<u32> = prompt.into_iter().take(cap as usize - steps - 8).collect();
            forced.push(inst.tape_comparison_trace(&prompt, steps as u32).expect("greedy trace").ids);
            toks.push(prompt);
        }
        let alone: Vec<Vec<Vec<f32>>> = (0..nb).map(|p| inst.teacher_forced_logits(&toks[p], &forced[p]).expect("forced logits")).collect();
        let together = inst.teacher_forced_logits_batch(&toks, &forced).expect("batched forced logits");
        let mut means = Vec::new();
        for p in 0..nb {
            assert!(together[p].iter().all(|r| r.iter().all(|x| x.is_finite())), "batched stream {p}: non-finite logits");
            let rels: Vec<f64> = together[p].iter().zip(&alone[p]).map(|(x, y)| rel_l2(x, y)).collect();
            let flips: Vec<String> = together[p]
                .iter()
                .zip(&alone[p])
                .enumerate()
                .filter(|(_, (x, y))| argmax(x) != argmax(y))
                .map(|(i, (_, y))| format!("step {i} (alone margin {:.3})", margin(y)))
                .collect();
            let mean_kl = together[p].iter().zip(&alone[p]).map(|(x, y)| kl(y, x)).sum::<f64>() / rels.len() as f64;
            let mean = rels.iter().sum::<f64>() / rels.len() as f64;
            means.push(mean);
            println!(
                "batched x{nb} stream {p}: mean rel-L2 to the same stream alone {mean:.2e} worst {:.2e} mean KL {mean_kl:.2e}; argmax flips: {}",
                rels.iter().cloned().fold(0.0, f64::max),
                if flips.is_empty() { "none".to_string() } else { flips.join(", ") }
            );
        }
        let mean = means.iter().sum::<f64>() / means.len() as f64;
        println!("SUMMARY batched x{nb} decode vs alone: mean rel-L2 {mean:.3e} = {:.2}x the floor {floor:.3e}", mean / floor);
        if let Some(m) = flag("--assert-within-floor").and_then(|v| v.parse::<f64>().ok()) {
            assert!(mean <= m * floor, "batched x{nb} decode: mean rel-L2 {mean:.3e} exceeds {m}x the numerics floor {floor:.3e}");
        }
    }
    println!("SUMMARY numerics floor {floor:.3e} (mean over {} prompts)", floors.len());
    let bound: Option<f64> = flag("--assert-within-floor").and_then(|v| v.parse().ok());
    for (si, s) in settings.iter().enumerate().skip(1).filter(|_| !a.iter().any(|x| x == "--batched-only")) {
        let mean = means[si].iter().sum::<f64>() / means[si].len().max(1) as f64;
        println!("SUMMARY {:<42} mean rel-L2 to the reference {mean:.3e} = {:.2}x the floor", s.name, mean / floor);
        if let Some(m) = bound {
            assert!(mean <= m * floor, "{}: mean rel-L2 {mean:.3e} exceeds {m}x the numerics floor {floor:.3e}", s.name);
        }
    }
}
