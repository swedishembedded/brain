// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain qwen3tts …` - Qwen3-TTS voice synthesis (Talker + MTP + codec + speaker).
//!
//!   brain qwen3tts import --ckpt <hf_dir> [--codec-ckpt <dir> --speaker-ckpt <dir>]
//!                    [--out-dir out/tts]
//!       Import all four components into brain checkpoints:
//!         <out-dir>/talker.safetensors  <out-dir>/mtp.safetensors
//!         <out-dir>/codec.safetensors   <out-dir>/speaker.safetensors
//!
//!   brain qwen3tts clone --text "..." --ref voice.wav --ref-text "..." --out demo.wav
//!                   [--weights-dir out/tts --ckpt <hf_dir> --lang english
//!                    --max-frames N --temp X --top-k K --seed S --ref-codes codes.bin]
//!       Voice clone: x-vector timbre from the reference voice (pure brain). When
//!       --ref-text is given, the reference wav is encoded to [T,16] codes in-tree
//!       (codec encoder) and the in-context (ICL) path runs automatically - no
//!       external --ref-codes needed (an explicit --ref-codes still overrides).
//!
//!   brain qwen3tts synth --text "..." --out out.wav
//!                   [--weights-dir out/tts --ckpt <hf_dir> --lang english ...]
//!       Speaker-free text-to-speech.
//!
//!   brain qwen3tts finetune --base out/tts/talker.safetensors --data data/tts
//!                      [--full] [--out out/tts/talker_lora.safetensors]
//!                      [--steps N --lr X --rank R --alpha A --batch B --block T --seed S]
//!       Single-speaker SFT the Talker on a `text->codes` dataset (e.g.
//!       `make data/tts`). Default: LoRA - freezes the base, trains the
//!       attention adapters only. `--full`: every Talker weight trains,
//!       matching Qwen's own documented single-speaker fine-tuning workflow
//!       (`--out` then defaults to `talker_full.safetensors`). See
//!       `qwen3tts::sft` for the aligned multi-codebook loss both modes share.

use qwen3tts::{GenOpts, ResidualOpts, TtsPaths};

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
        Some("synth") | Some("infer") => synth(&args[1..]),
        Some("design") => design(&args[1..]),
        Some("serve") => crate::tts_serve::run_serve(&args[1..]),
        Some("sim") => sim(&args[1..]),
        Some("finetune") => finetune(&args[1..]),
        // Any other verb is forwarded to the GENERIC capability dispatcher,
        // the same way `sam2_cli::run_sam2` hands its non-`track` verbs back.
        // Without this, an action that exists only on `qwen3tts::caps`'s
        // manifest (`batch`) was advertised by `brain caps qwen3tts` and then
        // rejected here as an unknown verb - the CLI contradicting the
        // manifest it prints. `-h`/no verb still get this module's usage line,
        // since those are about the dedicated commands.
        Some(verb) if !verb.starts_with('-') => {
            let mut do_args = vec![qwen3tts::caps::MODEL.to_string()];
            do_args.extend_from_slice(args);
            std::process::exit(crate::caps_cli::run_do(&do_args));
        }
        other => {
            eprintln!("usage: brain qwen3tts <import|clone|synth|design|serve|finetune> ...  (got {other:?})");
            eprintln!("       plus every action on the generic manifest (`brain caps qwen3tts`), e.g. `batch`.");
            std::process::exit(2);
        }
    }
}

/// LoRA fine-tune the Talker on a `text->codes` dataset (single-speaker SFT).
///
///   brain qwen3tts finetune --base out/tts/talker.safetensors --data data/tts --out out/tts/talker_lora.safetensors
///                      [--steps N --lr X --rank R --alpha A --batch B --block T --seed S]
fn finetune(args: &[String]) {
    let mut base = "out/tts/talker.safetensors".to_string();
    let mut data_dir = "data/tts".to_string();
    let mut out = String::new();
    let mut full = false;
    let mut o = qwen3tts::FinetuneOpts::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--base" => base = val(args, &mut i, "--base"),
            "--data" => data_dir = val(args, &mut i, "--data"),
            "--out" => out = val(args, &mut i, "--out"),
            // Every Talker weight trains (matches Qwen's own documented
            // single-speaker SFT workflow), instead of the default LoRA
            // adapters-only path that keeps the base frozen.
            "--full" => full = true,
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
    if out.is_empty() {
        out = if full { "out/tts/talker_full.safetensors".to_string() } else { "out/tts/talker_lora.safetensors".to_string() };
    }
    let mode = if full { "full".to_string() } else { format!("LoRA r={} α={}", o.rank, o.alpha) };
    eprintln!("tts finetune [{mode}]: base={base} data={data_dir} steps={} lr={} -> {out}", o.steps, o.lr);
    let result = if full {
        qwen3tts::sft::finetune_full(&base, std::path::Path::new(&data_dir), &out, &o)
    } else {
        qwen3tts::sft::finetune_lora(&base, std::path::Path::new(&data_dir), &out, &o)
    };
    match result {
        Ok((i0, i1)) => println!("finetune done: loss {i0:.4} -> {i1:.4}  saved -> {out}"),
        Err(e) => {
            eprintln!("finetune failed: {e}");
            std::process::exit(1);
        }
    }
}

/// `brain qwen3tts sim --a A.wav --b B.wav [--speaker out/tts-1b7/speaker.safetensors]`
/// Speaker-embedding cosine similarity between two utterances (ECAPA x-vectors) -
/// the timbre-preservation metric. Used to validate that a quantized (e.g. INT4)
/// Talker keeps the cloned voice: compare sim(int4_out, ref) vs sim(int8_out, ref).
/// Each wav is embedded at its own sample rate (the encoder resamples internally).
fn sim(args: &[String]) {
    let mut a = String::new();
    let mut b = String::new();
    let mut speaker = "out/tts-1b7/speaker.safetensors".to_string();
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
        eprintln!("usage: brain qwen3tts sim --a A.wav --b B.wav [--speaker speaker.safetensors]");
        std::process::exit(2);
    }
    let wa = audio::wav::read(&a).unwrap_or_else(|e| { eprintln!("read {a}: {e}"); std::process::exit(1); });
    let wb = audio::wav::read(&b).unwrap_or_else(|e| { eprintln!("read {b}: {e}"); std::process::exit(1); });
    // Force the speaker encoder onto the CPU JIT (it's a small gpu_core model; avoids
    // the wgpu/GL default when the user isn't otherwise on the GPU).
    gpu_core::set_default_backend(gpu_core::Backend::Cpu);
    let enc = ecapatdnn::SpeakerEncoder::load_inference(&speaker);
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
        eprintln!("usage: brain qwen3tts import --ckpt <hf_dir> [--codec-ckpt D --speaker-ckpt D --out-dir D]");
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
    let talker = format!("{out_dir}/talker.safetensors");
    let mtp = format!("{out_dir}/mtp.safetensors");
    let codec_out = format!("{out_dir}/codec.safetensors");
    let speaker_out = format!("{out_dir}/speaker.safetensors");

    run_step("talker", qwen3tts::import::import_talker(&ckpt, &talker));
    run_step("mtp", qwen3tts::import::import_mtp(&ckpt, &mtp));
    run_step("codec", mimi::import::import(&codec_ckpt, &codec_out));
    // The CustomVoice / VoiceDesign (instruct) checkpoints have no speaker encoder
    // (tts_model_type != base) - they don't clone from reference audio. Skip it with
    // a warning rather than failing the whole import.
    match ecapatdnn::import::import(&speaker_ckpt, &speaker_out) {
        Ok(()) => {}
        Err(e) => eprintln!("import speaker: skipped ({e}) - fine for CustomVoice/VoiceDesign"),
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
    // `--weights-dir` defaults from $BRAIN_QWEN3TTS_WEIGHTS when set, same as
    // `--ckpt` below does from $BRAIN_QWEN3TTS_CKPT -- both are "where do the
    // converted/original checkpoint files live" and both are documented,
    // fetchable env vars, so one silently ignoring its var while the other
    // honors it was a drift, not a choice.
    let mut weights_dir = std::env::var("BRAIN_QWEN3TTS_WEIGHTS").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| "out/tts".to_string());
    // Checkpoint dir comes from $BRAIN_QWEN3TTS_CKPT (or `--ckpt`); never a baked-in
    // absolute path (see AGENTS.md: no absolute paths in source).
    let mut ckpt = std::env::var("BRAIN_QWEN3TTS_CKPT").unwrap_or_default();
    let mut out = "out.wav".to_string();
    let mut lang = "english".to_string();
    let mut opts = GenOpts::default();
    let mut extra = std::collections::HashMap::new();
    let (mut residual_temp, mut residual_top_k, mut residual_top_p) = (None, None, None);
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
            // A sampling flag the user did NOT pass stays `None`, so the
            // checkpoint's own `generation_config.json` (then the reference's
            // defaults) answers for it - see `qwen3tts::genconfig`.
            "--temp" => opts.sampling.temperature = val(args, &mut i, "--temp").parse().ok().or(opts.sampling.temperature),
            "--top-k" => opts.sampling.top_k = val(args, &mut i, "--top-k").parse().ok().or(opts.sampling.top_k),
            "--top-p" => opts.sampling.top_p = val(args, &mut i, "--top-p").parse().ok().or(opts.sampling.top_p),
            "--repetition-penalty" => opts.sampling.repetition_penalty = val(args, &mut i, "--repetition-penalty").parse().ok().or(opts.sampling.repetition_penalty),
            "--seed" => opts.seed = val(args, &mut i, "--seed").parse().unwrap_or(opts.seed),
            // Any `--residual-*` flag opts into independent MTP residual-codebook
            // sampling (default: greedy, matching the reference's own default);
            // unset residual knobs fall back to codebook-0's own resolved
            // temp/top-k/top-p, filled in AFTER the whole command line is parsed
            // so `--residual-temp 1.2 --temp 0.5` and `--temp 0.5
            // --residual-temp 1.2` mean the same thing.
            "--residual-temp" => residual_temp = val(args, &mut i, "--residual-temp").parse().ok(),
            "--residual-top-k" => residual_top_k = val(args, &mut i, "--residual-top-k").parse().ok(),
            "--residual-top-p" => residual_top_p = val(args, &mut i, "--residual-top-p").parse().ok(),
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
    if residual_temp.is_some() || residual_top_k.is_some() || residual_top_p.is_some() {
        // The codebook-0 chain resolved from the flags alone (no checkpoint
        // layer yet - the entry point applies that). It is only a seed for the
        // residual knobs the user did not pin.
        let cb0 = opts.plan().cb0;
        opts.residual = Some(ResidualOpts {
            temperature: residual_temp.unwrap_or(cb0.temperature),
            top_k: residual_top_k.unwrap_or(cb0.top_k),
            top_p: residual_top_p.unwrap_or(cb0.top_p),
        });
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
        talker: format!("{}/talker.safetensors", c.weights_dir),
        mtp: format!("{}/mtp.safetensors", c.weights_dir),
        codec: format!("{}/codec.safetensors", c.weights_dir),
        speaker: format!("{}/speaker.safetensors", c.weights_dir),
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
        eprintln!("usage: brain qwen3tts clone --text \"...\" --ref voice.wav [--ref-text \"...\" --ref-codes F] --out demo.wav");
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
    let npu = crate::npu_explicit();
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
        // Only what the CALLER pinned. The checkpoint's own
        // `generation_config.json` has not been read at this point, so
        // printing a number here would be a guess; `TTS_PLAN=1` prints the
        // real resolved plan once the entry point has read it.
        c.opts.sampling.temperature.map(|t| t.to_string()).unwrap_or_else(|| "auto".to_string()),
        c.out
    );
    let result = if npu {
        let cache = format!("{}/npu-cache", c.weights_dir);
        qwen3tts::pipeline::clone_npu(
            &paths(&c), &c.opts, &text, &refw, &ref_text, &c.lang, ref_code, Some(&cache),
        )
    } else {
        // Unarmed: this is a foreground one-shot command, so Ctrl-C ends the
        // whole process. A live cancel token matters on the surfaces that must
        // SURVIVE the abort (`caps.rs`/`resident_tts.rs`), which pass their
        // invocation's own token instead.
        let cancel = capability::CancelToken::default();
        qwen3tts::pipeline::clone(&paths(&c), &c.opts, &text, &refw, &ref_text, &c.lang, ref_code, &cancel)
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
        eprintln!("usage: brain qwen3tts synth --text \"...\" --out out.wav [--lang english ...]");
        std::process::exit(2);
    }
    let npu = crate::npu_explicit();
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
        qwen3tts::pipeline::synth_npu(&paths(&c), &c.opts, &text, &c.lang, Some(&cache))
    } else {
        let cancel = capability::CancelToken::default(); // unarmed, as in `clone`
        qwen3tts::pipeline::synth(&paths(&c), &c.opts, &text, &c.lang, &cancel)
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

/// `brain qwen3tts design --text "..." --instruct "..." [--speaker NAME] --out out.wav`
/// VoiceDesign (instruct only) / CustomVoice (instruct + preset speaker). Needs a
/// 1.7B CustomVoice/VoiceDesign checkpoint (the 0.6B Base has no instruct control).
fn design(args: &[String]) {
    let (c, extra) = parse_common(args);
    let text = extra.get("text").cloned().unwrap_or_default();
    let instruct = extra.get("instruct").cloned().unwrap_or_default();
    let speaker = extra.get("speaker").map(|s| s.as_str());
    if text.is_empty() {
        eprintln!("usage: brain qwen3tts design --text \"...\" --instruct \"...\" [--speaker NAME] --out out.wav");
        std::process::exit(2);
    }
    let npu = crate::npu_explicit();
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
        qwen3tts::pipeline::design_npu(&paths(&c), &c.opts, &text, &c.lang, &instruct, speaker, Some(&cache))
    } else {
        let cancel = capability::CancelToken::default(); // unarmed, as in `clone`
        qwen3tts::pipeline::design(&paths(&c), &c.opts, &text, &c.lang, &instruct, speaker, &cancel)
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
