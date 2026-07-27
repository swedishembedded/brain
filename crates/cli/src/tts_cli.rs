// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain tts …` — Qwen3-TTS voice synthesis (Talker + MTP + codec + speaker).
//!
//!   brain tts import --ckpt <hf_dir> [--codec-ckpt <dir> --speaker-ckpt <dir>]
//!                    [--out-dir out/tts]
//!       Import all four components into brain checkpoints:
//!         <out-dir>/talker.weights  <out-dir>/mtp.weights
//!         <out-dir>/codec.weights   <out-dir>/speaker.weights
//!
//!   brain tts clone --text "..." --ref voice.wav --ref-text "..." --out demo.wav
//!                   [--weights-dir out/tts --ckpt <hf_dir> --lang english
//!                    --max-frames N --temp X --top-k K --seed S --ref-codes codes.bin]
//!       Voice clone: x-vector timbre from the reference voice (pure brain). When
//!       --ref-text is given, the reference wav is encoded to [T,16] codes in-tree
//!       (codec encoder) and the in-context (ICL) path runs automatically — no
//!       external --ref-codes needed (an explicit --ref-codes still overrides).
//!
//!   brain tts synth --text "..." --out out.wav
//!                   [--weights-dir out/tts --ckpt <hf_dir> --lang english ...]
//!       Speaker-free text-to-speech.
//!
//!   brain tts finetune --base out/tts/talker.weights --data data/tts
//!                      --out out/tts/talker_lora.weights
//!                      [--steps N --lr X --rank R --alpha A --batch B --block T --seed S]
//!       LoRA fine-tune (single-speaker SFT) the Talker on a `text->codes`
//!       dataset (e.g. `make data/tts`). Freezes the base; trains the attention
//!       adapters only. See `tts::sft` for the aligned multi-codebook loss.

use tts::{GenOpts, TtsPaths};

fn val(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i).cloned().unwrap_or_else(|| {
        eprintln!("{flag} requires a value");
        std::process::exit(2);
    })
}

pub fn run_tts(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("import") => import(&args[1..]),
        Some("clone") => clone(&args[1..]),
        Some("synth") => synth(&args[1..]),
        Some("design") => design(&args[1..]),
        Some("serve") => crate::tts_serve::run_serve(&args[1..]),
        Some("sim") => sim(&args[1..]),
        Some("finetune") => finetune(&args[1..]),
        other => {
            eprintln!("usage: brain tts <import|clone|synth|design|serve|finetune> ...  (got {other:?})");
            std::process::exit(2);
        }
    }
}

/// LoRA fine-tune the Talker on a `text->codes` dataset (single-speaker SFT).
///
///   brain tts finetune --base out/tts/talker.weights --data data/tts --out out/tts/talker_lora.weights
///                      [--steps N --lr X --rank R --alpha A --batch B --block T --seed S]
fn finetune(args: &[String]) {
    let mut base = "out/tts/talker.weights".to_string();
    let mut data_dir = "data/tts".to_string();
    let mut out = "out/tts/talker_lora.weights".to_string();
    let mut o = tts::FinetuneOpts::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--base" => base = val(args, &mut i, "--base"),
            "--data" => data_dir = val(args, &mut i, "--data"),
            "--out" => out = val(args, &mut i, "--out"),
            "--steps" => o.steps = val(args, &mut i, "--steps").parse().unwrap_or(o.steps),
            "--lr" => o.lr = val(args, &mut i, "--lr").parse().unwrap_or(o.lr),
            "--rank" => o.rank = val(args, &mut i, "--rank").parse().unwrap_or(o.rank),
            "--alpha" => o.alpha = val(args, &mut i, "--alpha").parse().unwrap_or(o.alpha),
            "--batch" => o.batch = val(args, &mut i, "--batch").parse().unwrap_or(o.batch),
            "--block" => o.block = val(args, &mut i, "--block").parse().unwrap_or(o.block),
            "--seed" => o.seed = val(args, &mut i, "--seed").parse().unwrap_or(o.seed),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    eprintln!(
        "tts finetune [LoRA r={} α={}]: base={base} data={data_dir} steps={} lr={} -> {out}",
        o.rank, o.alpha, o.steps, o.lr
    );
    match tts::sft::finetune_lora(&base, std::path::Path::new(&data_dir), &out, &o) {
        Ok((i0, i1)) => println!("finetune done: loss {i0:.4} -> {i1:.4}  saved -> {out}"),
        Err(e) => {
            eprintln!("finetune failed: {e}");
            std::process::exit(1);
        }
    }
}

/// `brain tts sim --a A.wav --b B.wav [--speaker out/tts-1b7/speaker.weights]`
/// Speaker-embedding cosine similarity between two utterances (ECAPA x-vectors) —
/// the timbre-preservation metric. Used to validate that a quantized (e.g. INT4)
/// Talker keeps the cloned voice: compare sim(int4_out, ref) vs sim(int8_out, ref).
/// Each wav is embedded at its own sample rate (the encoder resamples internally).
fn sim(args: &[String]) {
    let mut a = String::new();
    let mut b = String::new();
    let mut speaker = "out/tts-1b7/speaker.weights".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--a" | "--pred" => a = val(args, &mut i, "--a"),
            "--b" | "--ref" => b = val(args, &mut i, "--b"),
            "--speaker" => speaker = val(args, &mut i, "--speaker"),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if a.is_empty() || b.is_empty() {
        eprintln!("usage: brain tts sim --a A.wav --b B.wav [--speaker speaker.weights]");
        std::process::exit(2);
    }
    let wa = audio::wav::read(&a).unwrap_or_else(|e| { eprintln!("read {a}: {e}"); std::process::exit(1); });
    let wb = audio::wav::read(&b).unwrap_or_else(|e| { eprintln!("read {b}: {e}"); std::process::exit(1); });
    // Force the speaker encoder onto the CPU JIT (it's a small gpu_core model; avoids
    // the wgpu/GL default when the user isn't otherwise on the GPU).
    gpu_core::set_default_backend(gpu_core::Backend::Cpu);
    let enc = speaker::SpeakerEncoder::load_inference(&speaker);
    let ea = enc.embed_wav(&wa.samples, wa.sample_rate);
    let eb = enc.embed_wav(&wb.samples, wb.sample_rate);
    let dot: f32 = ea.iter().zip(&eb).map(|(x, y)| x * y).sum();
    let na: f32 = ea.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = eb.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 };
    println!("spk-cosine({a}, {b}) = {cos:.4}");
}

fn import(args: &[String]) {
    let mut ckpt = String::new();
    let mut codec_ckpt = String::new();
    let mut speaker_ckpt = String::new();
    let mut out_dir = "out/tts".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ckpt" => ckpt = val(args, &mut i, "--ckpt"),
            "--codec-ckpt" => codec_ckpt = val(args, &mut i, "--codec-ckpt"),
            "--speaker-ckpt" => speaker_ckpt = val(args, &mut i, "--speaker-ckpt"),
            "--out-dir" => out_dir = val(args, &mut i, "--out-dir"),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if ckpt.is_empty() {
        eprintln!("usage: brain tts import --ckpt <hf_dir> [--codec-ckpt D --speaker-ckpt D --out-dir D]");
        std::process::exit(2);
    }
    // The speaker encoder lives in the same checkpoint as the Talker; the codec
    // (speech tokenizer) ships separately but defaults to <ckpt>/speech_tokenizer.
    if codec_ckpt.is_empty() {
        codec_ckpt = format!("{ckpt}/speech_tokenizer");
    }
    if speaker_ckpt.is_empty() {
        speaker_ckpt = ckpt.clone();
    }
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!("create {out_dir}: {e}");
        std::process::exit(2);
    }
    let talker = format!("{out_dir}/talker.weights");
    let mtp = format!("{out_dir}/mtp.weights");
    let codec_out = format!("{out_dir}/codec.weights");
    let speaker_out = format!("{out_dir}/speaker.weights");

    run_step("talker", tts::import::import_talker(&ckpt, &talker));
    run_step("mtp", tts::import::import_mtp(&ckpt, &mtp));
    run_step("codec", codec::import::import(&codec_ckpt, &codec_out));
    // The CustomVoice / VoiceDesign (instruct) checkpoints have no speaker encoder
    // (tts_model_type != base) — they don't clone from reference audio. Skip it with
    // a warning rather than failing the whole import.
    match speaker::import::import(&speaker_ckpt, &speaker_out) {
        Ok(()) => {}
        Err(e) => eprintln!("import speaker: skipped ({e}) — fine for CustomVoice/VoiceDesign"),
    }
    println!("imported Qwen3-TTS components -> {out_dir}/");
}

fn run_step(name: &str, r: Result<(), String>) {
    if let Err(e) = r {
        eprintln!("import {name} failed: {e}");
        std::process::exit(1);
    }
}

struct CommonArgs {
    weights_dir: String,
    ckpt: String,
    out: String,
    lang: String,
    opts: GenOpts,
}

fn parse_common(args: &[String]) -> (CommonArgs, std::collections::HashMap<String, String>) {
    let mut weights_dir = "out/tts".to_string();
    let mut ckpt = "/data/workspace/tmp/qwen3-tts-resources/ckpt/Qwen3-TTS-12Hz-0.6B-Base".to_string();
    let mut out = "out.wav".to_string();
    let mut lang = "english".to_string();
    let mut opts = GenOpts::default();
    let mut extra = std::collections::HashMap::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights-dir" => weights_dir = val(args, &mut i, "--weights-dir"),
            "--ckpt" => ckpt = val(args, &mut i, "--ckpt"),
            "--out" => out = val(args, &mut i, "--out"),
            "--lang" | "--language" => lang = val(args, &mut i, "--lang"),
            "--max-frames" => {
                opts.max_frames = val(args, &mut i, "--max-frames").parse().unwrap_or(opts.max_frames)
            }
            "--temp" => opts.temperature = val(args, &mut i, "--temp").parse().unwrap_or(opts.temperature),
            "--top-k" => opts.top_k = val(args, &mut i, "--top-k").parse().unwrap_or(opts.top_k),
            "--seed" => opts.seed = val(args, &mut i, "--seed").parse().unwrap_or(opts.seed),
            // (`--no-cache` removed: the Talker now always uses the device-agnostic
            // KV-cache step(); select CPU vs GPU with `--device`.)
            "--text" => {
                extra.insert("text".to_string(), val(args, &mut i, "--text"));
            }
            "--ref" => {
                extra.insert("ref".to_string(), val(args, &mut i, "--ref"));
            }
            "--ref-text" => {
                extra.insert("ref-text".to_string(), val(args, &mut i, "--ref-text"));
            }
            "--ref-codes" => {
                extra.insert("ref-codes".to_string(), val(args, &mut i, "--ref-codes"));
            }
            "--instruct" => {
                extra.insert("instruct".to_string(), val(args, &mut i, "--instruct"));
            }
            "--speaker" => {
                extra.insert("speaker".to_string(), val(args, &mut i, "--speaker"));
            }
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    (
        CommonArgs {
            weights_dir,
            ckpt,
            out,
            lang,
            opts,
        },
        extra,
    )
}

fn paths(c: &CommonArgs) -> TtsPaths {
    TtsPaths {
        talker: format!("{}/talker.weights", c.weights_dir),
        mtp: format!("{}/mtp.weights", c.weights_dir),
        codec: format!("{}/codec.weights", c.weights_dir),
        speaker: format!("{}/speaker.weights", c.weights_dir),
        ckpt_dir: c.ckpt.clone(),
    }
}

/// Read a `[T,16]` u32 codes file: 8-byte little-endian count header + u32 data
/// (the format the reference dump scripts write).
fn read_codes(path: &str) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    if bytes.len() < 8 {
        return Err("codes file too short".to_string());
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let data = &bytes[8..];
    if data.len() < n * 4 {
        return Err("codes file truncated".to_string());
    }
    Ok((0..n)
        .map(|k| u32::from_le_bytes(data[k * 4..k * 4 + 4].try_into().unwrap()))
        .collect())
}

fn clone(args: &[String]) {
    let (c, extra) = parse_common(args);
    let text = extra.get("text").cloned().unwrap_or_default();
    let refw = extra.get("ref").cloned().unwrap_or_default();
    let ref_text = extra.get("ref-text").cloned().unwrap_or_default();
    if text.is_empty() || refw.is_empty() {
        eprintln!("usage: brain tts clone --text \"...\" --ref voice.wav [--ref-text \"...\" --ref-codes F] --out demo.wav");
        std::process::exit(2);
    }
    let ref_code = match extra.get("ref-codes") {
        Some(p) => match read_codes(p) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("read ref codes: {e}");
                std::process::exit(1);
            }
        },
        None => None,
    };
    let mode = if ref_code.is_some() {
        "ICL (external ref-codes)"
    } else if !ref_text.trim().is_empty() {
        "ICL (in-tree codec encode)"
    } else {
        "x-vector-only"
    };
    let npu = crate::npu_requested();
    if npu {
        // Talker + codec run on the NPU; force any incidental gpu_core model (the
        // speaker x-vector encoder) onto the CPU JIT instead of the wgpu/GL default.
        gpu_core::set_default_backend(gpu_core::Backend::Cpu);
    }
    eprintln!(
        "tts clone [{mode}]{}: lang={} max_frames={} temp={} -> {}",
        if npu { " on NPU (OpenVINO)" } else { "" },
        c.lang,
        c.opts.max_frames,
        c.opts.temperature,
        c.out
    );
    let result = if npu {
        let cache = format!("{}/npu-cache", c.weights_dir);
        tts::pipeline::clone_npu(
            &paths(&c), &c.opts, &text, &refw, &ref_text, &c.lang, ref_code, Some(&cache),
        )
    } else {
        tts::pipeline::clone(&paths(&c), &c.opts, &text, &refw, &ref_text, &c.lang, ref_code)
    };
    let wav = match result {
        Ok(w) => w,
        Err(e) => {
            eprintln!("clone failed: {e}");
            std::process::exit(1);
        }
    };
    write_wav(&c.out, &wav);
}

fn synth(args: &[String]) {
    let (c, extra) = parse_common(args);
    let text = extra.get("text").cloned().unwrap_or_default();
    if text.is_empty() {
        eprintln!("usage: brain tts synth --text \"...\" --out out.wav [--lang english ...]");
        std::process::exit(2);
    }
    let npu = crate::npu_requested();
    if npu {
        gpu_core::set_default_backend(gpu_core::Backend::Cpu);
    }
    eprintln!(
        "tts synth{}: lang={} max_frames={} -> {}",
        if npu { " on NPU (OpenVINO)" } else { "" },
        c.lang,
        c.opts.max_frames,
        c.out
    );
    let result = if npu {
        let cache = format!("{}/npu-cache", c.weights_dir);
        tts::pipeline::synth_npu(&paths(&c), &c.opts, &text, &c.lang, Some(&cache))
    } else {
        tts::pipeline::synth(&paths(&c), &c.opts, &text, &c.lang)
    };
    let wav = match result {
        Ok(w) => w,
        Err(e) => {
            eprintln!("synth failed: {e}");
            std::process::exit(1);
        }
    };
    write_wav(&c.out, &wav);
}

/// `brain tts design --text "..." --instruct "..." [--speaker NAME] --out out.wav`
/// VoiceDesign (instruct only) / CustomVoice (instruct + preset speaker). Needs a
/// 1.7B CustomVoice/VoiceDesign checkpoint (the 0.6B Base has no instruct control).
fn design(args: &[String]) {
    let (c, extra) = parse_common(args);
    let text = extra.get("text").cloned().unwrap_or_default();
    let instruct = extra.get("instruct").cloned().unwrap_or_default();
    let speaker = extra.get("speaker").map(|s| s.as_str());
    if text.is_empty() {
        eprintln!("usage: brain tts design --text \"...\" --instruct \"...\" [--speaker NAME] --out out.wav");
        std::process::exit(2);
    }
    let npu = crate::npu_requested();
    if npu {
        gpu_core::set_default_backend(gpu_core::Backend::Cpu);
    }
    eprintln!(
        "tts design{}: lang={} speaker={:?} instruct={:?} max_frames={} -> {}",
        if npu { " on NPU (OpenVINO)" } else { "" },
        c.lang,
        speaker,
        instruct,
        c.opts.max_frames,
        c.out
    );
    let result = if npu {
        let cache = format!("{}/npu-cache", c.weights_dir);
        tts::pipeline::design_npu(&paths(&c), &c.opts, &text, &c.lang, &instruct, speaker, Some(&cache))
    } else {
        tts::pipeline::design(&paths(&c), &c.opts, &text, &c.lang, &instruct, speaker)
    };
    let wav = match result {
        Ok(w) => w,
        Err(e) => {
            eprintln!("design failed: {e}");
            std::process::exit(1);
        }
    };
    write_wav(&c.out, &wav);
}

fn write_wav(path: &str, wav: &[f32]) {
    let finite = wav.iter().all(|x| x.is_finite());
    if let Err(e) = audio::wav::write(path, wav, 24000) {
        eprintln!("write {path}: {e}");
        std::process::exit(1);
    }
    println!(
        "wrote {path}: {} samples, {:.2}s @ 24kHz (finite={finite})",
        wav.len(),
        wav.len() as f32 / 24000.0
    );
}
