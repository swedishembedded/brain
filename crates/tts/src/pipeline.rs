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
    /// Decode on the bit-exact KV-cache CpuTalker (O(T)/frame, CPU host) when
    /// true; otherwise the full-recompute `TalkerGen` path that runs on the
    /// selected `gpu_core` backend (CPU JIT or the wgpu GPU). The GPU path needs
    /// `cached = false`, since the cache mirror is CPU-only.
    pub cached: bool,
}

impl Default for GenOpts {
    fn default() -> GenOpts {
        // The reference (`Qwen3TTSModel._merge_generate_kwargs`) decodes the Talker
        // codebook-0 stream with sampling — `do_sample=True, top_k=50,
        // temperature=0.9` — never greedily. Greedy (`temperature=0`) is degenerate
        // for this autoregressive acoustic model: codebook-0 collapses into a single
        // repeating token after a few frames, decoding to near-silence (rms ~0.004
        // vs ~0.07 sampled). Default to the reference's sampling so a plain
        // `brain tts clone` yields voice; pass `--temp 0` for the deterministic
        // (greedy) parity path. The fixed `seed` keeps a single run reproducible.
        GenOpts {
            max_frames: 256,
            temperature: 0.9,
            top_k: 50,
            seed: 0,
            min_new: 2,
            cached: true,
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
pub(crate) fn sample_cb0(
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
        opts.min_new == 0,
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

/// KV-cached variant of [`generate_codes`]: identical autoregressive logic and
/// sampling, but the Talker decoder is the incremental, key/value-cached
/// [`CpuTalker`] (`O(T)` per frame) instead of the full-recompute
/// [`TalkerGen::forward`] (`O(T²)`). The frozen weights and the resulting codes
/// are the same (the cache is algebraically exact); only the decoder cost
/// differs. The MTP residual fill is unchanged (it is bounded at
/// `num_code_groups` and re-runs cheaply).
pub fn generate_codes_cached(
    cpu: &mut crate::gen_kv::CpuTalker,
    mtp: &mut crate::gen_kv_mtp::CpuMtp,
    sp: &TtsSpecials,
    prompt: &Prompt,
    opts: &GenOpts,
) -> Vec<u32> {
    use std::time::Instant;
    // Coarse per-stage profiling, gated on `TTS_PROFILE` so normal runs are silent.
    let profile = std::env::var("TTS_PROFILE").is_ok();
    let (mut t_step, mut t_mtp) = (0.0f64, 0.0f64);

    let d = cpu.d();
    let n_trailing = prompt.trailing.len() / d;
    let mut rng = Rng::new(opts.seed);

    // Stream the whole prefix through the cache, keeping the last hidden state.
    let t_prefix0 = Instant::now();
    cpu.reset();
    let n_prefix = prompt.embeds.len() / d;
    let mut past_hidden = vec![0.0f32; d];
    for i in 0..n_prefix {
        past_hidden = cpu.step(&prompt.embeds[i * d..(i + 1) * d]);
    }
    let t_prefix = t_prefix0.elapsed().as_secs_f64() * 1e3;
    let mut cb0 = sample_cb0(
        cpu.codec_head_logits(&past_hidden),
        sp.codec_eos,
        opts.min_new == 0,
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
        let cb0_embed = cpu.codec_embed(cb0).to_vec();
        let tm = Instant::now();
        let (residuals, res_sum) = mtp.generate_residuals(&past_hidden, &cb0_embed);
        t_mtp += tm.elapsed().as_secs_f64() * 1e3;
        frames.push(cb0);
        frames.extend_from_slice(&residuals);

        let mut feed = cb0_embed;
        add_into(&mut feed, &res_sum);
        if s < n_trailing {
            add_into(&mut feed, &prompt.trailing[s * d..(s + 1) * d]);
        } else {
            add_into(&mut feed, &prompt.tts_pad);
        }
        s += 1;
        if cpu.pos() >= cpu.cfg.max_position_embeddings as usize {
            break;
        }
        // one incremental decoder step for the new frame.
        let ts = Instant::now();
        past_hidden = cpu.step(&feed);
        t_step += ts.elapsed().as_secs_f64() * 1e3;
        cb0 = sample_cb0(
            cpu.codec_head_logits(&past_hidden),
            sp.codec_eos,
            s >= opts.min_new,
            opts.temperature,
            opts.top_k,
            &mut rng,
        );
    }
    if profile {
        let nf = s.max(1) as f64;
        eprintln!(
            "[tts-profile] prefix-stream({n_prefix} pos)={t_prefix:.1}ms | \
             talker-step total={t_step:.1}ms ({:.1}ms/frame) | \
             mtp-residuals total={t_mtp:.1}ms ({:.1}ms/frame) | frames={s}",
            t_step / nf,
            t_mtp / nf,
        );
    }
    frames
}

/// Split the assistant chat-template ids into the 3-token role header and the
/// target text content (`input_id[3..len-5]`).
pub(crate) fn split_input_ids(ids: &[u32]) -> Result<(Vec<u32>, Vec<u32>), String> {
    if ids.len() < 3 + 5 + 1 {
        return Err(format!("tokenized prompt too short ({} ids)", ids.len()));
    }
    Ok((ids[..3].to_vec(), ids[3..ids.len() - 5].to_vec()))
}

pub(crate) fn assistant_text(text: &str) -> String {
    format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
}

/// The instruct turn for CustomVoice / VoiceDesign (`_build_instruct_text`).
pub(crate) fn instruct_text(instruct: &str) -> String {
    format!("<|im_start|>user\n{instruct}<|im_end|>\n")
}

/// Resolve a CustomVoice preset speaker name to its codec-token id (or error with
/// the supported list). `None` name -> `None` (VoiceDesign).
pub(crate) fn resolve_speaker(sp: &TtsSpecials, speaker: Option<&str>) -> Result<Option<u32>, String> {
    match speaker {
        None => Ok(None),
        Some(name) if name.trim().is_empty() => Ok(None),
        Some(name) => sp.speaker_id(name).map(Some).ok_or_else(|| {
            let mut names: Vec<&String> = sp.spk_id.keys().collect();
            names.sort();
            format!("unknown speaker {name:?}; supported preset speakers: {names:?}")
        }),
    }
}

/// Upper bound on the Talker context length (prefix + generated frames).
fn max_ctx(opts: &GenOpts, ref_frames: usize) -> u32 {
    (opts.max_frames + ref_frames + 32) as u32
}

/// ICL reference codes for `ref_wav_path`, encoded once and cached under
/// `cache_dir` (keyed by the wav's name + mtime). Encoding the reference is the
/// slow CPU step of a clone; caching it makes repeated clones of the same voice
/// skip straight to generation. Returns the `[T,16]` codes.
pub(crate) fn ref_codes_cached(codec_path: &str, wav: &audio::wav::Wav, ref_wav_path: &str, cache_dir: Option<&str>) -> Vec<u32> {
    let cache_file = cache_dir.map(|d| {
        let stem = std::path::Path::new(ref_wav_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("ref");
        let mt = std::fs::metadata(ref_wav_path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        std::path::PathBuf::from(d).join(format!("refcodes-{stem}-{mt}.bin"))
    });
    if let Some(cf) = &cache_file {
        if let Ok(bytes) = std::fs::read(cf) {
            if bytes.len() >= 8 {
                let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
                if bytes.len() >= 8 + n * 4 {
                    eprintln!("tts: reusing cached reference codes ({n} values) from {}", cf.display());
                    return (0..n)
                        .map(|k| u32::from_le_bytes(bytes[8 + k * 4..12 + k * 4].try_into().unwrap()))
                        .collect();
                }
            }
        }
    }
    let codec = codec::Codec::load_inference(codec_path);
    let sr = codec.cfg.input_sample_rate;
    let wav24 = audio::resample_linear(&wav.samples, wav.sample_rate, sr);
    let codes = codec.encode(&wav24);
    if let Some(cf) = &cache_file {
        if let Some(p) = cf.parent() {
            std::fs::create_dir_all(p).ok();
        }
        let mut bytes = (codes.len() as u64).to_le_bytes().to_vec();
        for c in &codes {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
        if std::fs::write(cf, &bytes).is_ok() {
            eprintln!("tts: cached reference codes -> {}", cf.display());
        }
    }
    codes
}

/// **Voice clone** — synthesize `target_text` in the timbre of `ref_wav_path`.
///
/// When `ref_text` is non-empty, this uses the **ICL** path: the reference wav is
/// encoded to `[T,16]` codec codes in-tree ([`codec::Codec::encode`]) and the
/// Talker is conditioned on `ref_text` + those reference codes (plus the
/// x-vector). An explicit `ref_code` (e.g. an external dump) still overrides the
/// auto-encode. With an empty `ref_text` and no `ref_code`, the **x-vector-only**
/// path is used (speaker timbre only). Both paths run entirely within brain.
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

    // ICL reference codes: an explicit dump wins; otherwise, when `ref_text` is
    // given, encode the reference wav in-tree (no external `--ref-codes` needed).
    let ref_code = match ref_code {
        Some(c) => Some(c),
        None if !ref_text.trim().is_empty() => {
            let codec = codec::Codec::load_inference(&paths.codec);
            let sr = codec.cfg.input_sample_rate;
            let wav24 = audio::resample_linear(&wav.samples, wav.sample_rate, sr);
            Some(codec.encode(&wav24))
        }
        None => None,
    };

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

    let codes = generate(&gen, &mtp, &sp, &prompt, opts, &paths.talker, &paths.mtp);
    decode_codes(&paths.codec, &codes)
}

/// Dispatch generation to the cached CPU path or the backend (CPU/GPU) full
/// recompute, per `opts.cached`. Both yield identical codes (the cache is
/// algebraically exact); only the cost and the engine differ.
fn generate(
    gen: &TalkerGen,
    mtp: &MtpModel,
    sp: &TtsSpecials,
    prompt: &Prompt,
    opts: &GenOpts,
    talker_path: &str,
    mtp_path: &str,
) -> Vec<u32> {
    if opts.cached {
        let mut cpu = crate::gen_kv::CpuTalker::load(talker_path);
        let mut cpu_mtp = crate::gen_kv_mtp::CpuMtp::load(mtp_path);
        generate_codes_cached(&mut cpu, &mut cpu_mtp, sp, prompt, opts)
    } else {
        generate_codes(gen, mtp, sp, prompt, opts)
    }
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

    let codes = generate(&gen, &mtp, &sp, &prompt, opts, &paths.talker, &paths.mtp);
    decode_codes(&paths.codec, &codes)
}

/// **Voice clone on the Intel NPU** — same conditioning as [`clone`], but the
/// Talker decoder and the codec decode run as compiled OpenVINO graphs on the NPU
/// (host-side codebook-0 sampling + MTP residual fill, via [`crate::npu_gen`]).
/// `npu_cache`, when given, persists the exported ONNX + OpenVINO compiled blobs
/// so re-runs skip the export/compile wait.
#[allow(clippy::too_many_arguments)]
pub fn clone_npu(
    paths: &TtsPaths,
    opts: &GenOpts,
    target_text: &str,
    ref_wav_path: &str,
    ref_text: &str,
    language: &str,
    ref_code: Option<Vec<u32>>,
    npu_cache: Option<&str>,
) -> Result<Vec<f32>, String> {
    let sp = TtsSpecials::from_config_dir(&paths.ckpt_dir)?;
    let tok = prompt::load_tokenizer(&paths.ckpt_dir)?;
    let language_id = sp.language_id(language);

    let wav = audio::wav::read(ref_wav_path).map_err(|e| format!("read {ref_wav_path}: {e}"))?;
    let speaker = speaker::SpeakerEncoder::load_inference(&paths.speaker);
    let xvec = speaker.embed_wav(&wav.samples, wav.sample_rate);

    let ref_code = match ref_code {
        Some(c) => Some(c),
        None if !ref_text.trim().is_empty() => {
            Some(ref_codes_cached(&paths.codec, &wav, ref_wav_path, npu_cache))
        }
        None => None,
    };

    let input_ids = tok.encode(&assistant_text(target_text));
    let (role_ids, text_ids) = split_input_ids(&input_ids)?;

    let tables = crate::npu_gen::TalkerTables::load(&paths.talker);
    let mut mtp = crate::gen_kv_mtp::CpuMtp::load(&paths.mtp);

    let prompt = if let Some(ref_code) = ref_code {
        let ref_ids_full = tok.encode(&format!("<|im_start|>assistant\n{ref_text}<|im_end|>\n"));
        if ref_ids_full.len() < 3 + 2 + 1 {
            return Err("ref_text tokenized too short".to_string());
        }
        let ref_ids = &ref_ids_full[3..ref_ids_full.len() - 2];
        prompt::build_icl_prompt(
            &tables, &mtp, &sp, &role_ids, &text_ids, ref_ids, &ref_code, &xvec, language_id,
        )
    } else {
        prompt::build_xvector_prompt(&tables, &sp, &role_ids, &text_ids, Some(&xvec), language_id)
    };

    run_npu(paths, &tables, &mut mtp, &sp, &prompt, opts, npu_cache)
}

/// **Synth on the Intel NPU** — speaker-free text-to-speech (no reference voice).
pub fn synth_npu(
    paths: &TtsPaths,
    opts: &GenOpts,
    target_text: &str,
    language: &str,
    npu_cache: Option<&str>,
) -> Result<Vec<f32>, String> {
    let sp = TtsSpecials::from_config_dir(&paths.ckpt_dir)?;
    let tok = prompt::load_tokenizer(&paths.ckpt_dir)?;
    let language_id = sp.language_id(language);

    let input_ids = tok.encode(&assistant_text(target_text));
    let (role_ids, text_ids) = split_input_ids(&input_ids)?;

    let tables = crate::npu_gen::TalkerTables::load(&paths.talker);
    let mut mtp = crate::gen_kv_mtp::CpuMtp::load(&paths.mtp);
    let prompt = prompt::build_xvector_prompt(&tables, &sp, &role_ids, &text_ids, None, language_id);

    run_npu(paths, &tables, &mut mtp, &sp, &prompt, opts, npu_cache)
}

/// Shared NPU path: run the Talker generation loop on the NPU, then decode the
/// codes on the NPU codec graph. Falls back to CPU/GPU within OpenVINO if the NPU
/// is unavailable (so the same command works on any OpenVINO host).
fn run_npu(
    paths: &TtsPaths,
    tables: &crate::npu_gen::TalkerTables,
    mtp: &mut crate::gen_kv_mtp::CpuMtp,
    sp: &TtsSpecials,
    prompt: &Prompt,
    opts: &GenOpts,
    npu_cache: Option<&str>,
) -> Result<Vec<f32>, String> {
    let cache = npu_cache.map(std::path::Path::new);
    // OpenVINO target device; override with BRAIN_TTS_NPU_DEVICE=cpu|gpu|npu|auto
    // (e.g. `cpu` to validate the fp32 graph, since the NPU runs the graph in fp16).
    let device = std::env::var("BRAIN_TTS_NPU_DEVICE")
        .ok()
        .and_then(|s| npu::openvino::NpuDevice::parse(&s))
        .unwrap_or(npu::openvino::NpuDevice::Npu);
    let allow_fallback = true;

    // Talker placement (`BRAIN_TTS_TALKER`):
    //   cpu          -> CPU KV-cache decoder (codec still on NPU),
    //   npu          -> fp32 cache-free NPU graph,
    //   npu-int8     -> weight-only INT8 cache-free NPU graph,
    //   npu-kv       -> INT8 resident KV-cache decode graph (O(1) proj/frame),
    //   npu-kv-fp32  -> fp32 resident KV-cache decode graph.
    // A large Talker (d_model>=2048) defaults to INT8 cache-free; the KV-cache
    // decode path (the per-frame speed win) is opt-in until it's the proven default.
    #[derive(PartialEq, Clone, Copy)]
    enum Mode {
        Cpu,
        NpuF32,
        NpuI8,
        NpuKvI8,
        NpuKvF32,
    }
    let mode = match std::env::var("BRAIN_TTS_TALKER").ok().as_deref() {
        Some("cpu") => Mode::Cpu,
        Some("npu") | Some("npu-fp32") => Mode::NpuF32,
        Some("npu-int8") | Some("int8") => Mode::NpuI8,
        Some("npu-kv") | Some("kv") | Some("npu-kv-int8") => Mode::NpuKvI8,
        Some("npu-kv-fp32") => Mode::NpuKvF32,
        // Default: the resident KV-cache decode graph (talker ~7-19x faster/frame
        // than cache-free). INT8 for the large 1.7B Talker, fp32 for the 0.6B.
        _ if tables.cfg.d_model >= 2048 => Mode::NpuKvI8,
        _ => Mode::NpuKvF32,
    };

    // MTP placement: the residual code-predictor re-runs its 5-layer decoder 16x
    // per frame. On the host that re-streams the MTP's ~300MB fp32 weights every
    // substep and is memory-bandwidth bound — measured ~580ms/frame on the 1.7B
    // (vs an earlier ~225ms when the decoder was smaller). The resident INT8 NPU
    // decode graph (`KvMtp`) streams 4x-smaller weights from device memory and
    // measures ~257ms/frame — a 2.25x win — so it is now the DEFAULT for the large
    // (d_model>=2048) model. `BRAIN_TTS_MTP=cpu` forces the host path (still the
    // default on the small 0.6B, whose CPU MTP is cheap); `=npu` forces it on.
    // Not used for the CPU-Talker mode (which already uses the host MTP).
    let mtp_npu = match std::env::var("BRAIN_TTS_MTP").ok().as_deref() {
        Some("npu") => true,
        Some("cpu") => false,
        _ => tables.cfg.d_model >= 2048,
    } && mode != Mode::Cpu;
    let mut kvmtp = if mtp_npu {
        Some(crate::npu_gen::KvMtp::load(&paths.mtp, device, allow_fallback, cache, tables.cfg.d_model >= 2048)?)
    } else {
        None
    };

    let codes = match mode {
        Mode::Cpu => {
            eprintln!("tts npu: Talker on CPU KV-cache (d_model={}); codec on NPU", tables.cfg.d_model);
            let mut cpu = crate::gen_kv::CpuTalker::load(&paths.talker);
            generate_codes_cached(&mut cpu, mtp, sp, prompt, opts)
        }
        Mode::NpuKvI8 | Mode::NpuKvF32 => {
            let quant = mode == Mode::NpuKvI8;
            if std::env::var("TTS_NPU_PARITY").is_ok() {
                match crate::npu_gen::kv_prefix_parity(&paths.talker, tables, prompt, cache, device, quant) {
                    Ok(m) => eprintln!(
                        "tts npu parity: KV-cache prefix hidden max-abs ({}, {} vs CPU) = {m:.3e}",
                        device.ov_str(),
                        if quant { "INT8" } else { "fp32" }
                    ),
                    Err(e) => eprintln!("tts npu parity check failed: {e}"),
                }
            }
            let eng: &mut dyn crate::npu_gen::MtpEngine = match &mut kvmtp {
                Some(k) => k,
                None => mtp,
            };
            crate::npu_gen::generate_kv(
                &paths.talker, tables, eng, sp, prompt, opts, device, allow_fallback, cache, quant,
            )?
        }
        Mode::NpuF32 | Mode::NpuI8 => {
            let quant = mode == Mode::NpuI8;
            // Optional deterministic parity gate (NPU/OV-device vs CPU Talker hidden state).
            if std::env::var("TTS_NPU_PARITY").is_ok() {
                match crate::npu_gen::talker_prefix_parity(&paths.talker, tables, prompt, cache, device, quant) {
                    Ok(m) => eprintln!(
                        "tts npu parity: Talker prefix hidden max-abs ({}, {} vs CPU) = {m:.3e}",
                        device.ov_str(),
                        if quant { "INT8" } else { "fp32" }
                    ),
                    Err(e) => eprintln!("tts npu parity check failed: {e}"),
                }
            }
            let eng: &mut dyn crate::npu_gen::MtpEngine = match &mut kvmtp {
                Some(k) => k,
                None => mtp,
            };
            crate::npu_gen::generate_npu(
                &paths.talker, tables, eng, sp, prompt, opts, device, allow_fallback, cache, quant,
            )?
        }
    };
    if codes.is_empty() {
        return Err("no codec frames were generated".to_string());
    }
    eprintln!("tts npu: generated {} frames; decoding on NPU codec graph…", codes.len() / 16);
    let tc = std::time::Instant::now();
    let (wav, codec_dev) = crate::npu_gen::decode_codes_npu(&paths.codec, &codes, device, allow_fallback, cache)?;
    if std::env::var("TTS_PROFILE").is_ok() {
        eprintln!(
            "[tts-npu-profile] codec compile+decode ({} frames) on {codec_dev} = {:.0}ms",
            codes.len() / 16,
            tc.elapsed().as_secs_f64() * 1e3
        );
    } else {
        eprintln!("tts npu: codec decode ran on {codec_dev}");
    }
    Ok(wav)
}

/// **VoiceDesign / CustomVoice** — synthesize `target_text` in a voice described
/// by the natural-language `instruct` (CustomVoice may also pick a preset
/// `speaker`). The 0.6B Base model has no instruct control; use a 1.7B
/// CustomVoice/VoiceDesign checkpoint. Runs on the selected `gpu_core` backend
/// (CPU/GPU); see [`design_npu`] for the NPU path.
pub fn design(
    paths: &TtsPaths,
    opts: &GenOpts,
    target_text: &str,
    language: &str,
    instruct: &str,
    speaker: Option<&str>,
) -> Result<Vec<f32>, String> {
    let sp = TtsSpecials::from_config_dir(&paths.ckpt_dir)?;
    let tok = prompt::load_tokenizer(&paths.ckpt_dir)?;
    let language_id = sp.language_id(language);
    let speaker_id = resolve_speaker(&sp, speaker)?;

    let input_ids = tok.encode(&assistant_text(target_text));
    let (role_ids, text_ids) = split_input_ids(&input_ids)?;
    let instruct_ids = if instruct.trim().is_empty() {
        Vec::new()
    } else {
        tok.encode(&instruct_text(instruct))
    };

    // Generous context bound (instruct + text + frames) for the GPU recompute path.
    let max_t = (instruct_ids.len() + input_ids.len() + opts.max_frames + 64) as u32;
    let gen = TalkerGen::load(&paths.talker, max_t);
    let mtp = MtpModel::load_inference(&paths.mtp);
    let prompt =
        prompt::build_instruct_prompt(&gen, &sp, &role_ids, &text_ids, &instruct_ids, speaker_id, language_id);

    let codes = generate(&gen, &mtp, &sp, &prompt, opts, &paths.talker, &paths.mtp);
    decode_codes(&paths.codec, &codes)
}

/// **VoiceDesign / CustomVoice on the Intel NPU** — as [`design`], with the
/// Talker + codec running as compiled OpenVINO graphs.
pub fn design_npu(
    paths: &TtsPaths,
    opts: &GenOpts,
    target_text: &str,
    language: &str,
    instruct: &str,
    speaker: Option<&str>,
    npu_cache: Option<&str>,
) -> Result<Vec<f32>, String> {
    let sp = TtsSpecials::from_config_dir(&paths.ckpt_dir)?;
    let tok = prompt::load_tokenizer(&paths.ckpt_dir)?;
    let language_id = sp.language_id(language);
    let speaker_id = resolve_speaker(&sp, speaker)?;

    let input_ids = tok.encode(&assistant_text(target_text));
    let (role_ids, text_ids) = split_input_ids(&input_ids)?;
    let instruct_ids = if instruct.trim().is_empty() {
        Vec::new()
    } else {
        tok.encode(&instruct_text(instruct))
    };

    let tables = crate::npu_gen::TalkerTables::load(&paths.talker);
    let mut mtp = crate::gen_kv_mtp::CpuMtp::load(&paths.mtp);
    let prompt =
        prompt::build_instruct_prompt(&tables, &sp, &role_ids, &text_ids, &instruct_ids, speaker_id, language_id);

    run_npu(paths, &tables, &mut mtp, &sp, &prompt, opts, npu_cache)
}

/// Decode `[T,16]` codes to a 24 kHz waveform (empty -> error).
pub fn decode_codes(codec_path: &str, codes: &[u32]) -> Result<Vec<f32>, String> {
    if codes.is_empty() {
        return Err("no codec frames were generated".to_string());
    }
    let codec = codec::Codec::load_inference(codec_path);
    if std::env::var("TTS_PROFILE").is_ok() {
        let t0 = std::time::Instant::now();
        let wav = codec.decode(codes);
        eprintln!(
            "[tts-profile] codec.decode({} frames)={:.1}ms -> {} samples",
            codes.len() / 16,
            t0.elapsed().as_secs_f64() * 1e3,
            wav.len()
        );
        return Ok(wav);
    }
    Ok(codec.decode(codes))
}
