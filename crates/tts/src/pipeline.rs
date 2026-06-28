// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end Qwen3-TTS inference: `text (+ reference voice) -> 24 kHz waveform`.
//!
//! Wires the parity-verified components into the autoregressive voice-synthesis
//! loop from `qwen3_tts_model.py`:
//!   1. tokenize the target text (Qwen BPE) and assemble the Talker prompt
//!      ([`crate::prompt`]) — x-vector timbre (+ optional ICL reference codes);
//!   2. the Talker ([`crate::gen::TalkerGen`]) autoregressively samples codebook-0
//!      per frame; the MTP ([`crate::mtp::MtpModel`]) fills codebooks 1..15;
//!   3. stop at the codec EOS (or a frame cap), then the codec
//!      ([`codec::Codec`]) decodes the `[T,16]` codes to a waveform.
//!
//! Speaker x-vectors come from [`speaker::SpeakerEncoder`]; audio I/O from
//! [`audio`].

use data::rng::Rng;
use data::tokenizer::Tokenizer;

use crate::gen::TalkerGen;
use crate::mtp::MtpModel;
use crate::prompt::{self, Prompt, TtsSpecials};

/// Brain checkpoint paths + the HF checkpoint dir (for `config.json`, tokenizer).
pub struct TtsPaths {
    pub talker: String,
    pub mtp: String,
    pub codec: String,
    pub speaker: String,
    pub ckpt_dir: String,
}

/// Sampling / length controls for the Talker.
#[derive(Clone, Debug)]
pub struct GenOpts {
    pub max_frames: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub seed: u64,
    pub min_new: usize,
}

impl Default for GenOpts {
    fn default() -> GenOpts {
        GenOpts {
            max_frames: 256,
            temperature: 0.0, // greedy by default (deterministic demo)
            top_k: 0,
            seed: 0,
            min_new: 2,
        }
    }
}

fn add_into(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

fn argmax(s: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..s.len() {
        if s[i] > s[bi] {
            bi = i;
        }
    }
    bi
}

/// Sample codebook-0 from `logits` with the reference's `suppress_tokens`: the
/// top-1024 vocab entries `[v-1024, v)` are masked except the codec EOS, which is
/// itself masked unless `allow_eos` (the `min_new_tokens` guard).
fn sample_cb0(
    mut logits: Vec<f32>,
    eos: u32,
    allow_eos: bool,
    temperature: f32,
    top_k: usize,
    rng: &mut Rng,
) -> u32 {
    let v = logits.len();
    let lo = v - 1024;
    let eos_logit = logits[eos as usize];
    for x in logits[lo..].iter_mut() {
        *x = f32::NEG_INFINITY;
    }
    if allow_eos {
        logits[eos as usize] = eos_logit;
    }
    if temperature <= 0.0 {
        return argmax(&logits) as u32;
    }
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
    if top_k > 0 && top_k < scaled.len() {
        let mut idx: Vec<usize> = (0..scaled.len()).collect();
        idx.sort_unstable_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());
        let threshold = scaled[idx[top_k - 1]];
        for x in scaled.iter_mut() {
            if *x < threshold {
                *x = f32::NEG_INFINITY;
            }
        }
    }
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in scaled.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let r = rng.next_f32() * sum;
    let mut acc = 0.0f32;
    for (i, &p) in scaled.iter().enumerate() {
        acc += p;
        if acc >= r {
            return i as u32;
        }
    }
    (scaled.len() - 1) as u32
}

/// Autoregressively generate codec codes `[n_frames*16]` (row-major, codebooks
/// 0..15 per frame) for an assembled [`Prompt`].
pub fn generate_codes(
    gen: &TalkerGen,
    mtp: &MtpModel,
    sp: &TtsSpecials,
    prompt: &Prompt,
    opts: &GenOpts,
) -> Vec<u32> {
    let d = gen.d();
    let n_trailing = prompt.trailing.len() / d;
    let mut rng = Rng::new(opts.seed);

    let mut ctx = prompt.embeds.clone();
    let mut hidden = gen.forward(&ctx);
    let mut past_hidden = hidden[hidden.len() - d..].to_vec();
    let mut cb0 = sample_cb0(
        gen.codec_head_logits(&past_hidden),
        sp.codec_eos,
        0 >= opts.min_new,
        opts.temperature,
        opts.top_k,
        &mut rng,
    );

    let mut frames: Vec<u32> = Vec::new();
    let mut s = 0usize;
    loop {
        if (cb0 == sp.codec_eos && s >= opts.min_new) || s >= opts.max_frames {
            break;
        }
        let cb0_embed = gen.codec_embed(cb0).to_vec();
        let (residuals, res_sum) = mtp.generate_residuals(&past_hidden, &cb0_embed);
        frames.push(cb0);
        frames.extend_from_slice(&residuals);

        // feedback embedding = Σ codec embeds + (trailing text | tts_pad)
        let mut feed = cb0_embed;
        add_into(&mut feed, &res_sum);
        if s < n_trailing {
            add_into(&mut feed, &prompt.trailing[s * d..(s + 1) * d]);
        } else {
            add_into(&mut feed, &prompt.tts_pad);
        }
        ctx.extend_from_slice(&feed);
        s += 1;
        if ctx.len() / d > gen.cfg.max_position_embeddings as usize {
            break;
        }

        hidden = gen.forward(&ctx);
        past_hidden = hidden[hidden.len() - d..].to_vec();
        cb0 = sample_cb0(
            gen.codec_head_logits(&past_hidden),
            sp.codec_eos,
            s >= opts.min_new,
            opts.temperature,
            opts.top_k,
            &mut rng,
        );
    }
    frames
}

/// Split the assistant chat-template ids into the 3-token role header and the
/// target text content (`input_id[3..len-5]`).
fn split_input_ids(ids: &[u32]) -> Result<(Vec<u32>, Vec<u32>), String> {
    if ids.len() < 3 + 5 + 1 {
        return Err(format!("tokenized prompt too short ({} ids)", ids.len()));
    }
    Ok((ids[..3].to_vec(), ids[3..ids.len() - 5].to_vec()))
}

fn assistant_text(text: &str) -> String {
    format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
}

/// Upper bound on the Talker context length (prefix + generated frames).
fn max_ctx(opts: &GenOpts, ref_frames: usize) -> u32 {
    (opts.max_frames + ref_frames + 32) as u32
}

/// **Voice clone** — synthesize `target_text` in the timbre of `ref_wav_path`.
///
/// By default this uses the **x-vector-only** path (speaker timbre from the
/// reference's speaker embedding), which runs entirely within brain. If
/// `ref_code` is supplied (`[T,16]` reference-audio codec codes, which brain's
/// decode-only codec cannot produce in-tree), the **ICL** path is used instead,
/// additionally conditioning on `ref_text` + the reference codes.
#[allow(clippy::too_many_arguments)]
pub fn clone(
    paths: &TtsPaths,
    opts: &GenOpts,
    target_text: &str,
    ref_wav_path: &str,
    ref_text: &str,
    language: &str,
    ref_code: Option<Vec<u32>>,
) -> Result<Vec<f32>, String> {
    let sp = TtsSpecials::from_config_dir(&paths.ckpt_dir)?;
    let tok = prompt::load_tokenizer(&paths.ckpt_dir)?;
    let language_id = sp.language_id(language);

    // speaker x-vector from the reference audio.
    let wav = audio::wav::read(ref_wav_path).map_err(|e| format!("read {ref_wav_path}: {e}"))?;
    let speaker = speaker::SpeakerEncoder::load_inference(&paths.speaker);
    let xvec = speaker.embed_wav(&wav.samples, wav.sample_rate);

    // tokenize the target text via the assistant chat template.
    let input_ids = tok.encode(&assistant_text(target_text));
    let (role_ids, text_ids) = split_input_ids(&input_ids)?;

    let ref_frames = ref_code.as_ref().map(|c| c.len() / 16).unwrap_or(0);
    let gen = TalkerGen::load(&paths.talker, max_ctx(opts, ref_frames));
    let mtp = MtpModel::load_inference(&paths.mtp);

    let prompt = if let Some(ref_code) = ref_code {
        let ref_ids_full = tok.encode(&format!("<|im_start|>assistant\n{ref_text}<|im_end|>\n"));
        if ref_ids_full.len() < 3 + 2 + 1 {
            return Err("ref_text tokenized too short".to_string());
        }
        let ref_ids = &ref_ids_full[3..ref_ids_full.len() - 2];
        prompt::build_icl_prompt(
            &gen, &mtp, &sp, &role_ids, &text_ids, ref_ids, &ref_code, &xvec, language_id,
        )
    } else {
        prompt::build_xvector_prompt(&gen, &sp, &role_ids, &text_ids, Some(&xvec), language_id)
    };

    let codes = generate_codes(&gen, &mtp, &sp, &prompt, opts);
    decode_codes(&paths.codec, &codes)
}

/// **Synth** — speaker-free text-to-speech (no reference voice).
pub fn synth(
    paths: &TtsPaths,
    opts: &GenOpts,
    target_text: &str,
    language: &str,
) -> Result<Vec<f32>, String> {
    let sp = TtsSpecials::from_config_dir(&paths.ckpt_dir)?;
    let tok = prompt::load_tokenizer(&paths.ckpt_dir)?;
    let language_id = sp.language_id(language);

    let input_ids = tok.encode(&assistant_text(target_text));
    let (role_ids, text_ids) = split_input_ids(&input_ids)?;

    let gen = TalkerGen::load(&paths.talker, max_ctx(opts, 0));
    let mtp = MtpModel::load_inference(&paths.mtp);
    let prompt = prompt::build_xvector_prompt(&gen, &sp, &role_ids, &text_ids, None, language_id);

    let codes = generate_codes(&gen, &mtp, &sp, &prompt, opts);
    decode_codes(&paths.codec, &codes)
}

/// Decode `[T,16]` codes to a 24 kHz waveform (empty -> error).
pub fn decode_codes(codec_path: &str, codes: &[u32]) -> Result<Vec<f32>, String> {
    if codes.is_empty() {
        return Err("no codec frames were generated".to_string());
    }
    let codec = codec::Codec::load_inference(codec_path);
    Ok(codec.decode(codes))
}
