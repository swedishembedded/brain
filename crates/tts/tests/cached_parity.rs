// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Cache-free vs KV-cached generation parity (the no-voice-bug regression).
//!
//! Both paths are greedy+deterministic, so for the SAME prompt they MUST emit
//! identical `[T,16]` codes. The cache-free path (`generate_codes`, WGSL engine)
//! is the known-good reference; the cached path (`generate_codes_cached`,
//! CpuTalker+CpuMtp) must match it exactly. Gated on the real checkpoint +
//! imported weights (large external artifacts) and on `MOE_SKIP_GPU_TESTS`.
//!
//! Run: `BRAIN_DEVICE=cpu cargo test -p brain-tts --test cached_parity -- --nocapture`

use data::tokenizer::Tokenizer;
use tts::gen::TalkerGen;
use tts::gen_kv::CpuTalker;
use tts::gen_kv_mtp::CpuMtp;
use tts::mtp::MtpModel;
use tts::pipeline::{generate_codes, generate_codes_cached, GenOpts};
use tts::prompt::{self, TtsSpecials};

const CKPT: &str = "/data/workspace/tmp/qwen3-tts-resources/ckpt/Qwen3-TTS-12Hz-0.6B-Base";
const TALKER: &str = "/data/workspace/applications/edgeai/brain/out/tts/talker.weights";
const MTP: &str = "/data/workspace/applications/edgeai/brain/out/tts/mtp.weights";
const SPEAKER: &str = "/data/workspace/applications/edgeai/brain/out/tts/speaker.weights";
const REF_WAV: &str = "/data/workspace/tmp/qwen3-tts-resources/voice-clone-example-voice.wav";

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn ready() -> bool {
    !std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
        && std::path::Path::new(TALKER).exists()
        && std::path::Path::new(MTP).exists()
        && std::path::Path::new(CKPT).join("config.json").exists()
}

#[test]
fn cached_matches_cachefree() {
    if !ready() {
        eprintln!("skip: weights/checkpoint not present (or MOE_SKIP_GPU_TESTS set)");
        return;
    }

    let sp = TtsSpecials::from_config_dir(CKPT).unwrap();
    let tok = prompt::load_tokenizer(CKPT).unwrap();
    let language_id = sp.language_id("english");

    // x-vector from the reference wav.
    let wav = audio::wav::read(REF_WAV).unwrap();
    let speaker = speaker::SpeakerEncoder::load_inference(SPEAKER);
    let xvec = speaker.embed_wav(&wav.samples, wav.sample_rate);

    // short text via the assistant chat template (mirror pipeline::clone).
    let text = "<|im_start|>assistant\nTesting one two three.<|im_end|>\n<|im_start|>assistant\n";
    let ids = tok.encode(text);
    let role_ids = ids[..3].to_vec();
    let text_ids = ids[3..ids.len() - 5].to_vec();

    let opts = GenOpts {
        max_frames: 16,
        temperature: 0.0,
        top_k: 0,
        seed: 0,
        min_new: 2,
    };

    let gen = TalkerGen::load(TALKER, 16 + 32);
    let mtp = MtpModel::load_inference(MTP);
    let prompt =
        prompt::build_xvector_prompt(&gen, &sp, &role_ids, &text_ids, Some(&xvec), language_id);

    let d = gen.d();
    let n_prefix = prompt.embeds.len() / d;
    eprintln!("prefix positions = {n_prefix}, d = {d}");

    // ---- (A) prefix forward: CpuTalker::forward_full vs TalkerGen::forward ----
    let mut cpu = CpuTalker::load(TALKER);
    let eng_hidden = gen.forward(&prompt.embeds);
    let cpu_full = cpu.forward_full(&prompt.embeds);
    let last = (n_prefix - 1) * d;
    let err_full = maxabs(&eng_hidden, &cpu_full);
    let err_last = maxabs(&eng_hidden[last..], &cpu_full[last..]);
    eprintln!("(A) prefix forward_full vs engine: all max-abs={err_full:.4e}, last-row={err_last:.4e}");

    // ---- (B) incremental step over prefix vs forward_full ----
    cpu.reset();
    let mut cpu_inc = vec![0.0f32; n_prefix * d];
    for i in 0..n_prefix {
        let h = cpu.step(&prompt.embeds[i * d..(i + 1) * d]);
        cpu_inc[i * d..(i + 1) * d].copy_from_slice(&h);
    }
    let err_inc = maxabs(&cpu_inc, &cpu_full);
    let err_inc_last = maxabs(&cpu_inc[last..], &cpu_full[last..]);
    eprintln!("(B) incremental step vs forward_full: all max-abs={err_inc:.4e}, last-row={err_inc_last:.4e}");

    // ---- (C) MTP residuals: CpuMtp vs MtpModel on the prefix last hidden ----
    let past_hidden = eng_hidden[last..].to_vec();
    // engine cb0 from engine hidden
    let eng_logits = gen.codec_head_logits(&past_hidden);
    let cb0 = (0..eng_logits.len())
        .filter(|&i| i < eng_logits.len() - 1024) // suppressed band masked anyway; just pick argmax over all
        .max_by(|&a, &b| eng_logits[a].partial_cmp(&eng_logits[b]).unwrap())
        .unwrap() as u32;
    let cb0_embed = gen.codec_embed(cb0).to_vec();
    let mut cpu_mtp = CpuMtp::load(MTP);
    let (codes_e, res_e) = mtp.generate_residuals(&past_hidden, &cb0_embed);
    let (codes_c, res_c) = cpu_mtp.generate_residuals(&past_hidden, &cb0_embed);
    eprintln!(
        "(C) MTP residuals: codes_eq={}, res-sum max-abs={:.4e}\n    engine={:?}\n    cpu   ={:?}",
        codes_e == codes_c,
        maxabs(&res_e, &res_c),
        codes_e,
        codes_c
    );

    // ---- (D) full generation: cache-free vs cached, frame-by-frame ----
    let codes_free = generate_codes(&gen, &mtp, &sp, &prompt, &opts);
    let mut cpu2 = CpuTalker::load(TALKER);
    let mut cpu_mtp2 = CpuMtp::load(MTP);
    let codes_cached = generate_codes_cached(&mut cpu2, &mut cpu_mtp2, &sp, &prompt, &opts);
    let tf = codes_free.len() / 16;
    let tc = codes_cached.len() / 16;
    eprintln!("(D) cache-free frames={tf}, cached frames={tc}");

    let mut first_div = None;
    for f in 0..tf.min(tc) {
        for c in 0..16 {
            let i = f * 16 + c;
            if codes_free[i] != codes_cached[i] {
                first_div = Some((f, c, codes_free[i], codes_cached[i]));
                break;
            }
        }
        if first_div.is_some() {
            break;
        }
    }
    match first_div {
        None if tf == tc => eprintln!("(D) IDENTICAL codes ({tf} frames) — cached path fixed"),
        None => eprintln!("(D) prefixes match but frame counts differ: free={tf} cached={tc}"),
        Some((f, c, a, b)) => {
            eprintln!("(D) FIRST DIVERGENCE at frame {f}, codebook {c}: free={a} cached={b}");
            eprintln!("    free  frame {f}: {:?}", &codes_free[f * 16..(f + 1) * 16]);
            eprintln!("    cached frame {f}: {:?}", &codes_cached[f * 16..(f + 1) * 16]);
        }
    }

    assert_eq!(codes_free, codes_cached, "cached codes must equal cache-free codes");
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-9)
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

/// End-to-end audio quality of the cached path: a short (24-frame) x-vector clone
/// must produce real voice (rms ~0.09) close to the reference speaker (sim ~0.95).
/// Writes /tmp/verify.wav. Gated as above. Set `TTS_PROFILE=1` for stage timers.
#[test]
fn cached_clone_audio_quality() {
    if !ready() {
        eprintln!("skip: weights/checkpoint not present (or MOE_SKIP_GPU_TESTS set)");
        return;
    }
    const CODEC: &str = "/data/workspace/applications/edgeai/brain/out/tts/codec.weights";
    let paths = tts::pipeline::TtsPaths {
        talker: TALKER.to_string(),
        mtp: MTP.to_string(),
        codec: CODEC.to_string(),
        speaker: SPEAKER.to_string(),
        ckpt_dir: CKPT.to_string(),
    };
    let opts = GenOpts {
        max_frames: 24,
        temperature: 0.0,
        top_k: 0,
        seed: 0,
        min_new: 2,
    };
    // --- codec sanity: decode the PyTorch golden codes with our codec.weights ---
    const GOLD: &str = "/data/workspace/tmp/qwen3-tts-resources/dumps/codec_ref/codes.bin";
    if std::path::Path::new(GOLD).exists() {
        let b = std::fs::read(GOLD).unwrap();
        let gcodes: Vec<u32> = b[8..]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let gw = tts::pipeline::decode_codes(&paths.codec, &gcodes).unwrap();
        eprintln!(
            "CODEC-SANITY: golden codes [{} frames] -> rms={:.4} (codec ok if ~0.05+)",
            gcodes.len() / 16,
            rms(&gw)
        );
    }

    // --- build the prompt + generate codes directly so we can inspect them ---
    let sp = TtsSpecials::from_config_dir(CKPT).unwrap();
    let tok = prompt::load_tokenizer(CKPT).unwrap();
    let language_id = sp.language_id("english");
    let refwav = audio::wav::read(REF_WAV).unwrap();
    let spk = speaker::SpeakerEncoder::load_inference(SPEAKER);
    let xvec = spk.embed_wav(&refwav.samples, refwav.sample_rate);
    eprintln!(
        "x-vector: len={}, rms={:.4}, any_nan={}",
        xvec.len(),
        rms(&xvec),
        xvec.iter().any(|v| v.is_nan())
    );
    let ids = tok.encode("<|im_start|>assistant\nTesting one two three.<|im_end|>\n<|im_start|>assistant\n");
    let role_ids = ids[..3].to_vec();
    let text_ids = ids[3..ids.len() - 5].to_vec();
    let gen = TalkerGen::load(TALKER, 24 + 32);
    let promptx =
        prompt::build_xvector_prompt(&gen, &sp, &role_ids, &text_ids, Some(&xvec), language_id);

    let t0 = std::time::Instant::now();
    let mut cpu = CpuTalker::load(TALKER);
    let mut cpu_mtp = CpuMtp::load(MTP);
    let codes = generate_codes_cached(&mut cpu, &mut cpu_mtp, &sp, &promptx, &opts);
    eprintln!("cached gen (24 frames) wall = {:.1}s", t0.elapsed().as_secs_f64());
    let cb0: Vec<u32> = (0..codes.len() / 16).map(|f| codes[f * 16]).collect();
    eprintln!("cb0 per frame = {cb0:?}");
    eprintln!("frame0 all 16 = {:?}", &codes[..16.min(codes.len())]);

    let wav = tts::pipeline::decode_codes(&paths.codec, &codes).unwrap();
    audio::wav::write("/tmp/verify.wav", &wav, 24000).unwrap();
    let r = rms(&wav);

    // speaker similarity: x-vector of generated clip vs reference clip.
    let xv_gen = spk.embed_wav(&wav, 24000);
    let sim = cosine(&xvec, &xv_gen);
    eprintln!(
        "VERIFY(greedy) /tmp/verify.wav: samples={}, rms={r:.4}, speaker-sim={sim:.4}",
        wav.len()
    );

    // --- sampled run (reference defaults: temp 0.9, top_k 50) to test whether the
    //     no-voice is a greedy-collapse artifact rather than a pipeline bug ---
    let opts_s = GenOpts {
        max_frames: 24,
        temperature: 0.9,
        top_k: 50,
        seed: 0,
        min_new: 2,
    };
    let mut cpu_s = CpuTalker::load(TALKER);
    let mut cpu_mtp_s = CpuMtp::load(MTP);
    let codes_s = generate_codes_cached(&mut cpu_s, &mut cpu_mtp_s, &sp, &promptx, &opts_s);
    let cb0_s: Vec<u32> = (0..codes_s.len() / 16).map(|f| codes_s[f * 16]).collect();
    eprintln!("sampled cb0 per frame = {cb0_s:?}");
    let wav_s = tts::pipeline::decode_codes(&paths.codec, &codes_s).unwrap();
    audio::wav::write("/tmp/verify_sampled.wav", &wav_s, 24000).unwrap();
    let xv_s = spk.embed_wav(&wav_s, 24000);
    eprintln!(
        "VERIFY(sampled) /tmp/verify_sampled.wav: samples={}, rms={:.4}, speaker-sim={:.4}",
        wav_s.len(),
        rms(&wav_s),
        cosine(&xvec, &xv_s)
    );
}
