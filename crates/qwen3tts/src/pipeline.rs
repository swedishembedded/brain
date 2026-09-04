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
//!      ([`mimi::Codec`]) decodes the `[T,16]` codes to a waveform.
//!
//! Speaker x-vectors come from [`ecapatdnn::SpeakerEncoder`]; audio I/O from
//! [`audio`].

use capability::CancelToken;
use data::rng::Rng;
use data::tokenizer::Tokenizer;

use crate::gen::TalkerGen;
use crate::genconfig::{GenerationConfig, GenerationPlan, SamplingRequest};
use crate::mtp::MtpModel;
use crate::prompt::{self, Prompt, TtsSpecials};
use crate::sampling::DegenerationWatch;

/// Brain checkpoint paths + the HF checkpoint dir (for `config.json`, tokenizer).
pub struct TtsPaths {
    pub talker: String,
    pub mtp: String,
    pub codec: String,
    pub speaker: String,
    pub ckpt_dir: String,
}

/// Sampling / length controls for the Talker.
///
/// Split into two halves on purpose:
///
/// - the length/determinism knobs (`max_frames`, `min_new`, `seed`) and the
///   residual-codebook opt-in, which only ever come from the caller;
/// - [`GenOpts::sampling`], the caller's EXPLICIT codebook-0/subtalker choices,
///   every one an `Option` because "unspecified" and "explicitly 1.0" are
///   different requests. Whatever the caller left unset is answered by the
///   checkpoint's `generation_config.json`, then by the reference's hard
///   defaults - see [`crate::genconfig`].
///
/// Call [`GenOpts::resolve`] once, at an entry point that knows the checkpoint
/// directory, to bake that resolution into [`GenOpts::resolved`]; the decode
/// loops then read [`GenOpts::plan`]. A `GenOpts` that was never resolved is
/// still correct - `plan()` falls back to the reference recipe - so a direct
/// construction, a unit test or a synthetic-checkpoint loop all keep working.
#[derive(Clone, Debug)]
pub struct GenOpts {
    /// Hard cap on generated codec frames.
    ///
    /// Not the same knob as the reference's `max_new_tokens` even though the
    /// units match; see [`GenerationPlan::max_new_tokens`] for the reasoning.
    pub max_frames: usize,
    pub seed: u64,
    pub min_new: usize,
    /// Independent sampling for the MTP's residual codebooks (1..15). `None`
    /// (the default) keeps the reference's greedy `code_predictor.generate`
    /// behavior; `Some` opts into temperature/top-k/top-p sampling there too --
    /// the residual codebooks carry most of the acoustic detail, so this is a
    /// real quality/expressiveness lever, not just parity with codebook-0's
    /// knobs. See [`ResidualOpts`].
    pub residual: Option<ResidualOpts>,
    /// The caller's explicit sampling choices; unset fields resolve from the
    /// checkpoint and then from the reference.
    pub sampling: SamplingRequest,
    /// The plan resolved for this run, once an entry point has called
    /// [`GenOpts::resolve`]. `None` means "not resolved against a checkpoint
    /// yet", which [`GenOpts::plan`] answers with the reference recipe.
    ///
    /// Set it through [`GenOpts::resolve`], never by hand: it is a CACHE of
    /// resolving [`GenOpts::sampling`], and a hand-written value that disagrees
    /// with `sampling` is exactly the second source of truth this whole module
    /// exists to remove. It is `pub` only because struct-update syntax
    /// (`..GenOpts::default()`) across crate boundaries requires every field to
    /// be visible.
    pub resolved: Option<GenerationPlan>,
}

impl GenOpts {
    /// Resolve this request against `ckpt_dir`'s `generation_config.json` and
    /// trace the result once. Idempotent: resolving twice re-resolves from the
    /// same request, never from an already-resolved plan, so precedence cannot
    /// silently collapse.
    pub fn resolve(&mut self, ckpt_dir: &str) {
        self.resolve_with(GenerationConfig::from_config_dir(ckpt_dir));
    }

    /// [`GenOpts::resolve`] against an ALREADY-PARSED checkpoint config, for a
    /// resident server that read `generation_config.json` once at load rather
    /// than once per request.
    pub fn resolve_with(&mut self, file: GenerationConfig) {
        let plan = GenerationPlan::resolve_with(&self.sampling, file);
        plan.trace(self.max_frames);
        self.resolved = Some(plan);
    }

    /// [`GenOpts::resolve`] as a value-consuming builder, for the entry points
    /// that receive `&GenOpts` and need an owned, resolved copy.
    pub fn resolved_for(mut self, ckpt_dir: &str) -> GenOpts {
        self.resolve(ckpt_dir);
        self
    }

    /// [`GenOpts::resolve_with`] as a value-consuming builder.
    pub fn resolved_with(mut self, file: GenerationConfig) -> GenOpts {
        self.resolve_with(file);
        self
    }

    /// The plan the decode loop runs: the resolution an entry point baked in,
    /// or - for an unresolved `GenOpts` - the caller's explicit choices over
    /// the reference's defaults, with no checkpoint layer.
    pub fn plan(&self) -> GenerationPlan {
        self.resolved.unwrap_or_else(|| GenerationPlan::resolve(&self.sampling, None))
    }
}

/// Sampling controls for the MTP's residual codebooks (1..15), independent of
/// codebook-0's [`GenOpts`] knobs -- mirrors the reference's separate
/// `subtalker_temperature`/`subtalker_top_k`/`subtalker_top_p` config keys.
#[derive(Clone, Debug)]
pub struct ResidualOpts {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for GenOpts {
    fn default() -> GenOpts {
        // Every sampling knob is UNSET here on purpose. `GenOpts::default()`
        // used to be the single source of truth for all of them, hardcoded in
        // Rust, and that is precisely how `repetition_penalty` drifted to `1.0`
        // (disabled) while the checkpoint's own `generation_config.json` and the
        // reference (`Qwen3TTSModel._merge_generate_kwargs`) both said `1.05` -
        // the exact guard that keeps codebook-0 out of a silent repetition loop.
        // Sampling alone does not prevent that collapse, it only makes it
        // seed-dependent: codebook-0's next-token distribution self-reinforces
        // once it repeats a token, top-1 probability climbing 0.92 -> 0.97 ->
        // 0.9998, past the point any temperature/top-k draw escapes.
        //
        // So the defaults now come from the checkpoint, with the reference's
        // recipe (`do_sample=true, temperature=0.9, top_k=50, top_p=1.0,
        // repetition_penalty=1.05`) as the fallback when there is no checkpoint
        // to read. Pass an explicit `sampling.temperature = Some(0.0)` (the
        // CLI's `--temp 0`) for the deterministic greedy parity path. The fixed
        // `seed` keeps a single run reproducible.
        //
        // `max_frames` is a caller knob, not a config-resolved one: it sizes the
        // Talker KV cache and the NPU graph bucket, so it does not inherit the
        // reference's 8192-token budget. See [`GenerationPlan::max_new_tokens`].
        GenOpts {
            max_frames: 256,
            seed: 0,
            min_new: 2,
            residual: None,
            sampling: SamplingRequest::default(),
            resolved: None,
        }
    }
}

fn add_into(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// The codebook-0 filter chain lives in [`crate::sampling`] - one testable unit
/// shared by every decode loop in this crate (this one, the CPU-cached mirror,
/// the batch scheduler and the three NPU loops) instead of a private helper
/// inside whichever loop happened to define it first.
pub(crate) use crate::sampling::sample_cb0;

/// A generation loop that stopped early because its [`CancelToken`] fired,
/// carrying the frames produced before the check that tripped.
///
/// Cancellation is NOT an error in the usual sense - the work done so far is
/// real, in-order codec codes, so this is the return value that keeps it rather
/// than a bare `Err(String)` that throws it away. A caller that wants the
/// partial clip hands `partial` straight to [`decode_codes`]; one that does not
/// simply drops it. The higher-level entry points ([`synth`]/[`clone`]/
/// [`design`]) do the latter and report `Err("cancelled")`, matching how every
/// other cancellable action in this workspace reports an aborted run.
#[derive(Clone, Debug)]
pub struct Cancelled {
    /// Codec codes generated before the cancel was observed, in the same
    /// `[frames*16]` row-major layout a completed run returns. May be empty
    /// (the token was already cancelled when the loop started).
    pub partial: Vec<u32>,
}

impl Cancelled {
    /// Number of complete frames in [`Self::partial`], given the model's
    /// per-frame code-group width (16 for the real checkpoints).
    pub fn frames(&self, group_width: usize) -> usize {
        if group_width == 0 {
            0
        } else {
            self.partial.len() / group_width
        }
    }
}

/// Autoregressively generate codec codes `[n_frames*16]` (row-major, codebooks
/// 0..15 per frame) for an assembled [`Prompt`].
///
/// `cancel` is polled once per frame, between the frames rather than inside
/// one: a frame is a Talker step plus an MTP residual fill, and that boundary
/// is this architecture's natural interruption point - a real 0.6B frame is
/// hundreds of milliseconds, so a finer-grained check would buy no measurable
/// latency and would have to thread a token through the kernel dispatch code.
/// Returns [`Cancelled`] (with the frames already produced) when the token
/// fires; pass an unarmed `CancelToken::default()` to run uninterrupted.
pub fn generate_codes(
    gen: &TalkerGen,
    mtp: &MtpModel,
    sp: &TtsSpecials,
    prompt: &Prompt,
    opts: &GenOpts,
    cancel: &CancelToken,
) -> Result<Vec<u32>, Cancelled> {
    use std::time::Instant;
    // Coarse per-stage profiling, gated on `TTS_PROFILE`, matching
    // `generate_codes_cached`'s. This is the path every default
    // `synth`/`clone`/`design` command runs, and it is the only one whose
    // stage split answers "where does a `--device gpu` run spend its time",
    // so it needs the same instrument the CPU-only mirror already had.
    let profile = std::env::var("TTS_PROFILE").is_ok();
    let (mut t_step, mut t_mtp, mut t_head) = (0.0f64, 0.0f64, 0.0f64);

    // Cancelled before we started: skip the prefix stream too (it is the one
    // unbounded-ish chunk of work outside the frame loop).
    if cancel.is_cancelled() {
        return Err(Cancelled { partial: Vec::new() });
    }

    let d = gen.d();
    let n_trailing = prompt.trailing.len() / d;
    let mut rng = Rng::new(opts.seed);
    // One resolution per generation call: caller override > the checkpoint's
    // generation_config.json > the reference's defaults. Already traced by
    // `GenOpts::resolve` at the entry point that knew the checkpoint dir.
    let cfg = opts.plan().cb0;
    let mut watch = DegenerationWatch::new();

    // Stream the prefix through the incremental KV cache, keeping the last hidden.
    // This is the unified O(T)/frame Talker decode (`TalkerGen::step`) — it runs on
    // whatever backend `Gpu` selected (GPU or the wgsl-cpu JIT), replacing both the
    // old O(T²) `forward` recompute and the CPU-only `CpuTalker` path.
    let t_prefix0 = Instant::now();
    gen.reset_cache();
    let n_prefix = prompt.embeds.len() / d;
    let mut past_hidden = vec![0.0f32; d];
    for i in 0..n_prefix {
        past_hidden = gen.step(&prompt.embeds[i * d..(i + 1) * d]);
    }
    let t_prefix = t_prefix0.elapsed().as_secs_f64() * 1e3;
    let mut cb0_history: Vec<u32> = Vec::new();
    let th = Instant::now();
    let cb0_logits = gen.codec_head_logits(&past_hidden);
    t_head += th.elapsed().as_secs_f64() * 1e3;
    let mut draw = sample_cb0(cb0_logits, sp.codec_eos, opts.min_new == 0, &cfg, &cb0_history, &mut rng);
    let mut cb0 = draw.token;

    let mut frames: Vec<u32> = Vec::new();
    let mut s = 0usize;
    loop {
        if (cb0 == sp.codec_eos && s >= opts.min_new) || s >= opts.max_frames {
            break;
        }
        if cancel.is_cancelled() {
            return Err(Cancelled { partial: frames });
        }
        // Observe the token as it is COMMITTED to frame `s`, so frame 0 counts
        // toward a run like every other frame. Diagnostic only - it never
        // changes what is sampled.
        if let Some(report) = watch.observe(s, draw) {
            eprintln!("{report}");
        }
        cb0_history.push(cb0);
        let cb0_embed = gen.codec_embed(cb0).to_vec();
        let tm = Instant::now();
        let (residuals, res_sum) = mtp.generate_residuals_with(&past_hidden, &cb0_embed, opts.residual.as_ref(), &mut rng);
        t_mtp += tm.elapsed().as_secs_f64() * 1e3;
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
        s += 1;
        if gen.cache_pos() as usize >= gen.cfg.max_position_embeddings as usize {
            break;
        }

        // one incremental decoder step for the new frame.
        let ts = Instant::now();
        past_hidden = gen.step(&feed);
        t_step += ts.elapsed().as_secs_f64() * 1e3;
        let th = Instant::now();
        let cb0_logits = gen.codec_head_logits(&past_hidden);
        t_head += th.elapsed().as_secs_f64() * 1e3;
        draw = sample_cb0(cb0_logits, sp.codec_eos, s >= opts.min_new, &cfg, &cb0_history, &mut rng);
        cb0 = draw.token;
    }
    if profile {
        let nf = s.max(1) as f64;
        eprintln!(
            "[tts-profile] prefix-stream({n_prefix} pos)={t_prefix:.1}ms | \
             talker-step total={t_step:.1}ms ({:.1}ms/frame) | \
             mtp-residuals total={t_mtp:.1}ms ({:.1}ms/frame) | \
             cb0-head total={t_head:.1}ms ({:.1}ms/frame) | frames={s}",
            t_step / nf,
            t_mtp / nf,
            t_head / nf,
        );
    }
    Ok(frames)
}

/// CPU-only KV-cache Talker generation for the **NPU path's `Mode::Cpu`** (Talker
/// on the host, codec/MTP on the NPU). The main `generate_codes` now uses the
/// device-agnostic `TalkerGen::step`; this remains for the NPU-adjacent CPU decode.
/// Identical autoregressive logic and
/// sampling, but the Talker decoder is the incremental, key/value-cached
/// [`CpuTalker`] (`O(T)` per frame) instead of the full-recompute
/// [`TalkerGen::forward`] (`O(T²)`). The frozen weights and the resulting codes
/// are the same (the cache is algebraically exact); only the decoder cost
/// differs. The MTP residual fill is unchanged (it is bounded at
/// `num_code_groups` and re-runs cheaply).
///
/// `cancel` behaves exactly as in [`generate_codes`]: polled once per frame,
/// returning [`Cancelled`] with the frames produced so far.
pub fn generate_codes_cached(
    cpu: &mut crate::gen_kv::CpuTalker,
    mtp: &mut crate::gen_kv_mtp::CpuMtp,
    sp: &TtsSpecials,
    prompt: &Prompt,
    opts: &GenOpts,
    cancel: &CancelToken,
) -> Result<Vec<u32>, Cancelled> {
    use std::time::Instant;
    // Coarse per-stage profiling, gated on `TTS_PROFILE` so normal runs are silent.
    let profile = std::env::var("TTS_PROFILE").is_ok();
    let (mut t_step, mut t_mtp) = (0.0f64, 0.0f64);

    if cancel.is_cancelled() {
        return Err(Cancelled { partial: Vec::new() });
    }

    let d = cpu.d();
    let n_trailing = prompt.trailing.len() / d;
    let mut rng = Rng::new(opts.seed);
    let cfg = opts.plan().cb0;
    let mut watch = DegenerationWatch::new();

    // Stream the whole prefix through the cache, keeping the last hidden state.
    let t_prefix0 = Instant::now();
    cpu.reset();
    let n_prefix = prompt.embeds.len() / d;
    let mut past_hidden = vec![0.0f32; d];
    for i in 0..n_prefix {
        past_hidden = cpu.step(&prompt.embeds[i * d..(i + 1) * d]);
    }
    let t_prefix = t_prefix0.elapsed().as_secs_f64() * 1e3;
    let mut cb0_history: Vec<u32> = Vec::new();
    let mut draw = sample_cb0(cpu.codec_head_logits(&past_hidden), sp.codec_eos, opts.min_new == 0, &cfg, &cb0_history, &mut rng);
    let mut cb0 = draw.token;

    let mut frames: Vec<u32> = Vec::new();
    let mut s = 0usize;
    loop {
        if (cb0 == sp.codec_eos && s >= opts.min_new) || s >= opts.max_frames {
            break;
        }
        if cancel.is_cancelled() {
            return Err(Cancelled { partial: frames });
        }
        if let Some(report) = watch.observe(s, draw) {
            eprintln!("{report}");
        }
        cb0_history.push(cb0);
        let cb0_embed = cpu.codec_embed(cb0).to_vec();
        let tm = Instant::now();
        // `CpuMtp::generate_residuals` has no sampling variant yet, so it stays
        // greedy-only here regardless of `opts.residual` -- that option only
        // takes effect on the full-recompute `MtpModel` path this cached
        // mirror does not use.
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
        draw = sample_cb0(cpu.codec_head_logits(&past_hidden), sp.codec_eos, s >= opts.min_new, &cfg, &cb0_history, &mut rng);
        cb0 = draw.token;
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
    Ok(frames)
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
    let codec = mimi::Codec::load_inference(codec_path);
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
/// encoded to `[T,16]` codec codes in-tree ([`mimi::Codec::encode`]) and the
/// Talker is conditioned on `ref_text` + those reference codes (plus the
/// x-vector). An explicit `ref_code` (e.g. an external dump) still overrides the
/// auto-encode. With an empty `ref_text` and no `ref_code`, the **x-vector-only**
/// path is used (speaker timbre only). Both paths run entirely within brain.
///
/// `cancel` is polled between generated frames; an aborted clone returns
/// `Err("cancelled")` without decoding a wav. Pass an unarmed
/// `CancelToken::default()` to run uninterrupted.
#[allow(clippy::too_many_arguments)]
pub fn clone(
    paths: &TtsPaths,
    opts: &GenOpts,
    target_text: &str,
    ref_wav_path: &str,
    ref_text: &str,
    language: &str,
    ref_code: Option<Vec<u32>>,
    cancel: &CancelToken,
) -> Result<Vec<f32>, String> {
    // Refuse an already-cancelled call before touching any weights.
    if cancel.is_cancelled() {
        return Err("cancelled".to_string());
    }
    let sp = TtsSpecials::from_config_dir(&paths.ckpt_dir)?;
    // Resolve the sampling plan ONCE per generation call, here, because this is
    // the outermost layer that knows the checkpoint directory. Everything below
    // reads `opts.plan()` and never re-reads a config file.
    let opts = &opts.clone().resolved_for(&paths.ckpt_dir);
    let tok = prompt::load_tokenizer(&paths.ckpt_dir)?;
    let language_id = sp.language_id(language);

    // speaker x-vector from the reference audio.
    let wav = audio::wav::read(ref_wav_path).map_err(|e| format!("read {ref_wav_path}: {e}"))?;
    let speaker = ecapatdnn::SpeakerEncoder::load_inference(&paths.speaker);
    let xvec = speaker.embed_wav(&wav.samples, wav.sample_rate);

    // ICL reference codes: an explicit dump wins; otherwise, when `ref_text` is
    // given, encode the reference wav in-tree (no external `--ref-codes` needed).
    let ref_code = match ref_code {
        Some(c) => Some(c),
        None if !ref_text.trim().is_empty() => {
            let codec = mimi::Codec::load_inference(&paths.codec);
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

    let codes = generate(&gen, &mtp, &sp, &prompt, opts, cancel)?;
    decode_codes(&paths.codec, &codes)
}

/// Generate codec codes. Single engine: the incremental KV-cache
/// [`TalkerGen::step`] Talker + the [`MtpModel`] residual fill, both running on
/// whatever backend `Gpu` selected (GPU or the wgsl-cpu JIT). Replaces the former
/// `opts.cached` split between a CPU-only `CpuTalker` and the O(T²) GPU recompute.
///
/// Cancellation collapses to `Err("cancelled")` here: the wav-level entry points
/// return a finished clip or nothing, so the partial codes are dropped. A caller
/// that wants the partial audio calls [`generate_codes`] directly and decodes
/// [`Cancelled::partial`] itself.
fn generate(
    gen: &TalkerGen,
    mtp: &MtpModel,
    sp: &TtsSpecials,
    prompt: &Prompt,
    opts: &GenOpts,
    cancel: &CancelToken,
) -> Result<Vec<u32>, String> {
    generate_codes(gen, mtp, sp, prompt, opts, cancel).map_err(|_| "cancelled".to_string())
}

/// **Synth** — speaker-free text-to-speech (no reference voice).
///
/// `cancel` is polled between generated frames; an aborted synth returns
/// `Err("cancelled")`. Pass an unarmed `CancelToken::default()` to run
/// uninterrupted.
pub fn synth(
    paths: &TtsPaths,
    opts: &GenOpts,
    target_text: &str,
    language: &str,
    cancel: &CancelToken,
) -> Result<Vec<f32>, String> {
    let codes = synth_codes(paths, opts, target_text, language, cancel)?;
    decode_codes(&paths.codec, &codes)
}

/// [`synth`] stopping one stage short: the `[T,16]` codec codes, before the
/// codec turns them into a waveform.
///
/// The seam exists because the waveform hides everything a health check needs.
/// "Did this clip terminate on the codec EOS or on its frame cap?" and "did
/// codebook-0 lock into a repetition run?" are both questions about the CODES,
/// and answering them from RMS alone is guesswork - which is why the
/// silent-collapse bug took a root-cause session rather than a failing
/// assertion. [`synth`] is this plus [`decode_codes`], so nothing can drift
/// between what the gate measures and what a user hears.
pub fn synth_codes(
    paths: &TtsPaths,
    opts: &GenOpts,
    target_text: &str,
    language: &str,
    cancel: &CancelToken,
) -> Result<Vec<u32>, String> {
    if cancel.is_cancelled() {
        return Err("cancelled".to_string());
    }
    let sp = TtsSpecials::from_config_dir(&paths.ckpt_dir)?;
    // Resolve the sampling plan ONCE per generation call, here, because this is
    // the outermost layer that knows the checkpoint directory. Everything below
    // reads `opts.plan()` and never re-reads a config file.
    let opts = &opts.clone().resolved_for(&paths.ckpt_dir);
    let tok = prompt::load_tokenizer(&paths.ckpt_dir)?;
    let language_id = sp.language_id(language);

    let input_ids = tok.encode(&assistant_text(target_text));
    let (role_ids, text_ids) = split_input_ids(&input_ids)?;

    let gen = TalkerGen::load(&paths.talker, max_ctx(opts, 0));
    let mtp = MtpModel::load_inference(&paths.mtp);
    let prompt = prompt::build_xvector_prompt(&gen, &sp, &role_ids, &text_ids, None, language_id);

    generate(&gen, &mtp, &sp, &prompt, opts, cancel)
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
    // Resolve the sampling plan ONCE per generation call, here, because this is
    // the outermost layer that knows the checkpoint directory. Everything below
    // reads `opts.plan()` and never re-reads a config file.
    let opts = &opts.clone().resolved_for(&paths.ckpt_dir);
    let tok = prompt::load_tokenizer(&paths.ckpt_dir)?;
    let language_id = sp.language_id(language);

    let wav = audio::wav::read(ref_wav_path).map_err(|e| format!("read {ref_wav_path}: {e}"))?;
    let speaker = ecapatdnn::SpeakerEncoder::load_inference(&paths.speaker);
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
    // Resolve the sampling plan ONCE per generation call, here, because this is
    // the outermost layer that knows the checkpoint directory. Everything below
    // reads `opts.plan()` and never re-reads a config file.
    let opts = &opts.clone().resolved_for(&paths.ckpt_dir);
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
///
/// **Not cancellable yet**, which is why `synth_npu`/`clone_npu`/`design_npu`
/// take no [`CancelToken`] while their CPU/GPU siblings do: the frame loops here
/// live in `crate::npu_gen` (`generate_codes_npu`/`generate_codes_kv`/
/// `generate_codes_kv_streaming`), each driving a compiled OpenVINO graph this
/// host cannot build or exercise. Offering a token those loops silently ignore
/// would be a lying API; the honest shape is no token until the NPU loops
/// actually poll one.
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
    // OpenVINO target device; override with BRAIN_QWEN3TTS_NPU_DEVICE=cpu|gpu|npu|auto
    // (e.g. `cpu` to validate the fp32 graph, since the NPU runs the graph in fp16).
    let device = std::env::var("BRAIN_QWEN3TTS_NPU_DEVICE")
        .ok()
        .and_then(|s| npu::openvino::NpuDevice::parse(&s))
        .unwrap_or(npu::openvino::NpuDevice::Npu);
    let allow_fallback = true;

    // Talker placement (`BRAIN_QWEN3TTS_TALKER`):
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
        NpuKvI4,
    }
    let mode = match std::env::var("BRAIN_QWEN3TTS_TALKER").ok().as_deref() {
        Some("cpu") => Mode::Cpu,
        Some("npu") | Some("npu-fp32") => Mode::NpuF32,
        Some("npu-int8") | Some("int8") => Mode::NpuI8,
        Some("npu-kv") | Some("kv") | Some("npu-kv-int8") => Mode::NpuKvI8,
        Some("npu-kv-int4") | Some("int4") => Mode::NpuKvI4,
        Some("npu-kv-fp32") => Mode::NpuKvF32,
        // Default: the resident KV-cache decode graph (talker measured an order
        // of magnitude faster per frame than cache-free). INT8 for the large
        // 1.7B Talker, fp32 for the 0.6B.
        _ if tables.cfg.d_model >= 2048 => Mode::NpuKvI8,
        _ => Mode::NpuKvF32,
    };
    // Print the resolved hardware path up front (device + weight precision + whether
    // an INT4 request is native or weight-compression on this device).
    if mode != Mode::Cpu {
        let (q, i4) = (matches!(mode, Mode::NpuI8 | Mode::NpuKvI8), matches!(mode, Mode::NpuKvI4));
        eprintln!("{}", crate::npu_gen::describe_talker_path(device, allow_fallback, q, i4));
    }

    // MTP placement: the residual code-predictor re-runs its 5-layer decoder 16
    // times per frame. On the host that re-streams the MTP's ~300MB fp32 weights every
    // substep and is memory-bandwidth bound, and it got worse as the decoder
    // grew. The resident INT8 NPU decode graph (`KvMtp`) streams a quarter of
    // the weight bytes from device memory and measured more than twice as fast
    // per frame on the 1.7B - so it is now the DEFAULT for the large
    // (d_model>=2048) model. `BRAIN_QWEN3TTS_MTP=cpu` forces the host path (still the
    // default on the small 0.6B, whose CPU MTP is cheap); `=npu` forces it on.
    // Not used for the CPU-Talker mode (which already uses the host MTP).
    // `BRAIN_QWEN3TTS_MTP`: `fused` = the single-infer fused graph (all 15 substeps in one
    // NPU inference — kills the per-substep dispatch overhead); `npu` = the per-substep
    // KvMtp; `cpu` = the host CpuMtp. Default for the large model is `npu` (KvMtp).
    let mut npu_mtp: Option<Box<dyn crate::npu_gen::MtpEngine>> = if mode == Mode::Cpu {
        None
    } else {
        match std::env::var("BRAIN_QWEN3TTS_MTP").ok().as_deref() {
            Some("cpu") => None,
            Some("fused") => Some(Box::new(crate::npu_gen::FusedMtp::load(&paths.mtp, device, allow_fallback, cache)?)),
            Some("npu") => Some(Box::new(crate::npu_gen::KvMtp::load(
                &paths.mtp, device, allow_fallback, cache, tables.cfg.d_model >= 2048,
            )?)),
            _ if tables.cfg.d_model >= 2048 => Some(Box::new(crate::npu_gen::KvMtp::load(
                &paths.mtp, device, allow_fallback, cache, true,
            )?)),
            _ => None,
        }
    };

    let codes = match mode {
        Mode::Cpu => {
            eprintln!("tts npu: Talker on CPU KV-cache (d_model={}); codec on NPU", tables.cfg.d_model);
            let mut cpu = crate::gen_kv::CpuTalker::load(&paths.talker);
            // Unarmed: this path has no token to honor (see `run_npu`'s doc),
            // and an unarmed token never cancels, so the loop runs to EOS.
            generate_codes_cached(&mut cpu, mtp, sp, prompt, opts, &CancelToken::default())
                .expect("an unarmed cancel token never fires")
        }
        Mode::NpuKvI8 | Mode::NpuKvF32 | Mode::NpuKvI4 => {
            let int4 = mode == Mode::NpuKvI4;
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
            let eng: &mut dyn crate::npu_gen::MtpEngine = match &mut npu_mtp {
                Some(k) => k.as_mut(),
                None => mtp,
            };
            crate::npu_gen::generate_kv(
                &paths.talker, tables, eng, sp, prompt, opts, device, allow_fallback, cache, quant, int4,
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
            let eng: &mut dyn crate::npu_gen::MtpEngine = match &mut npu_mtp {
                Some(k) => k.as_mut(),
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
///
/// `cancel` is polled between generated frames; an aborted design returns
/// `Err("cancelled")`. Pass an unarmed `CancelToken::default()` to run
/// uninterrupted.
pub fn design(
    paths: &TtsPaths,
    opts: &GenOpts,
    target_text: &str,
    language: &str,
    instruct: &str,
    speaker: Option<&str>,
    cancel: &CancelToken,
) -> Result<Vec<f32>, String> {
    if cancel.is_cancelled() {
        return Err("cancelled".to_string());
    }
    let sp = TtsSpecials::from_config_dir(&paths.ckpt_dir)?;
    // Resolve the sampling plan ONCE per generation call, here, because this is
    // the outermost layer that knows the checkpoint directory. Everything below
    // reads `opts.plan()` and never re-reads a config file.
    let opts = &opts.clone().resolved_for(&paths.ckpt_dir);
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

    let codes = generate(&gen, &mtp, &sp, &prompt, opts, cancel)?;
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
    // Resolve the sampling plan ONCE per generation call, here, because this is
    // the outermost layer that knows the checkpoint directory. Everything below
    // reads `opts.plan()` and never re-reads a config file.
    let opts = &opts.clone().resolved_for(&paths.ckpt_dir);
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
///
/// Built on the **ambient device** (`--device` / `BRAIN_DEVICE`), like the
/// Talker and the MTP - `mimi::Codec::load_inference` is CPU-pinned by its own
/// contract, which used to make the codec the one stage of a `--device gpu`
/// synth that never reached the GPU. It is also the largest single stage of a
/// short clip's wall time, so pinning it silently capped what `--device` could
/// ever mean here.
pub fn decode_codes(codec_path: &str, codes: &[u32]) -> Result<Vec<f32>, String> {
    if codes.is_empty() {
        return Err("no codec frames were generated".to_string());
    }
    let codec = mimi::Codec::load_inference_on(gpu_core::Gpu::new(mimi::PIPELINES), codec_path);
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

#[cfg(test)]
mod sampling_tests {
    use super::*;

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    // The codebook-0 filter chain's own tests (suppress mask, repetition
    // penalty, temperature, top-k, top-p, the categorical draw) live with the
    // chain in `crate::sampling`, where they can be written against synthetic
    // logits without a decode loop. What stays here is what needs THIS module:
    // the MTP residual sampler's interaction with it.

    #[test]
    fn residual_sampling_can_diverge_from_greedy_while_cb0_logic_is_untouched() {
        // MtpModel's own greedy generate_residuals vs generate_residuals_with at
        // high temperature: same synthetic weights, same inputs -- greedy is
        // deterministic, sampling is seed-dependent, so pushing the seed far
        // enough must eventually produce a different residual codebook stream.
        // Skips cleanly (like every other GPU test here) when no GPU is set up.
        if gpu_disabled() {
            return;
        }
        let cfg = crate::config::MtpConfig::tiny();
        let d = cfg.d_model as usize;
        let m = crate::mtp::MtpModel::new_synthetic_on(gpu_core::testgpu::dev(crate::mtp::PIPELINES), cfg, 3);
        let th = vec![0.3f32; d];
        let cb0 = vec![-0.2f32; d];
        let (greedy_codes, _) = m.generate_residuals(&th, &cb0);
        let ro = ResidualOpts { temperature: 2.0, top_k: 0, top_p: 0.0 };
        let mut found_divergent = false;
        for seed in 0..20u64 {
            let mut rng = Rng::new(seed);
            let (sampled_codes, _) = m.generate_residuals_with(&th, &cb0, Some(&ro), &mut rng);
            assert_eq!(sampled_codes.len(), greedy_codes.len());
            if sampled_codes != greedy_codes {
                found_divergent = true;
                break;
            }
        }
        assert!(found_divergent, "residual sampling never diverged from greedy across 20 seeds");
    }
}

#[cfg(test)]
mod cancel_tests {
    use super::*;
    use crate::gen_kv::CpuTalker;
    use crate::gen_kv_mtp::CpuMtp;
    use crate::testsupport::{synthetic_checkpoints, talker_test_cfg, tiny_prompt, tiny_specials, Scratch};
    use std::time::{Duration, Instant};

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    fn group_width() -> usize {
        // `MtpConfig::tiny()`'s num_code_groups (4: cb0 + 3 residuals), not the
        // real model's 16 - a per-frame width, not a fixed constant.
        crate::config::MtpConfig::tiny().num_code_groups as usize
    }

    /// THE test this whole feature exists for: a generation already in flight
    /// on another thread must stop within roughly one frame of the cancel, and
    /// must NOT have run to `max_frames`.
    ///
    /// Sizing is calibrated on the machine running the test rather than
    /// hardcoded: a short uncancelled run measures the per-frame cost, then
    /// `max_frames` is chosen so an uncancelled run would take seconds of
    /// wall time, and the cancel is fired a couple hundred frames in. "It stopped early" is therefore a
    /// difference of roughly two orders of magnitude, not a coin flip on a
    /// loaded CI box. (The CPU Talker's attention is `O(T)` per frame, so a
    /// per-frame cost calibrated at `T=32` UNDER-estimates the full run - the
    /// bounds below are conservative in the safe direction.)
    #[test]
    fn a_mid_flight_cancel_stops_generation_early_with_partial_codes() {
        if gpu_disabled() {
            return;
        }
        let scratch = Scratch::new("pipeline-cancel");
        let (talker_path, mtp_path) = synthetic_checkpoints(scratch.path(), 7);
        let sp = tiny_specials();
        let d = talker_test_cfg().d_model as usize;
        let prompt = tiny_prompt(d, 3, 2, 77);
        let gw = group_width();

        let mut cpu = CpuTalker::load(&talker_path);
        let mut mtp = CpuMtp::load(&mtp_path);

        // Calibration: weights are already loaded, so this times the frame loop
        // alone (both `generate_codes_cached` and `CpuMtp::generate_residuals`
        // reset their own state per call, so reusing these two is exact). The
        // first run is discarded - a cold-cache measurement would OVER-estimate
        // the per-frame cost, which would size the run below too small and let
        // it finish before the cancel ever lands.
        const CAL: usize = 32;
        let cal_opts = GenOpts { max_frames: CAL, min_new: CAL, seed: 5, ..GenOpts::default() };
        let never = CancelToken::default();
        let warm = generate_codes_cached(&mut cpu, &mut mtp, &sp, &prompt, &cal_opts, &never)
            .expect("an unarmed token never cancels");
        assert_eq!(warm.len() / gw, CAL, "calibration run must produce exactly max_frames frames");
        let t_cal = Instant::now();
        generate_codes_cached(&mut cpu, &mut mtp, &sp, &prompt, &cal_opts, &never)
            .expect("an unarmed token never cancels");
        let per_frame = t_cal.elapsed().div_f64(CAL as f64).max(Duration::from_nanos(1));

        // Enough frames that an uncancelled run needs seconds of wall time,
        // capped so a
        // pathologically fast machine cannot ask for an absurd cache. The
        // cancel lands after a couple hundred frames' worth of work, so
        // "stopped early" is a wide ratio, not a photo finish.
        let max_frames = ((3.0 / per_frame.as_secs_f64()) as usize).clamp(2_000, 20_000);
        let opts = GenOpts { max_frames, min_new: max_frames, seed: 5, ..GenOpts::default() };
        let wait = (per_frame * 200).clamp(Duration::from_millis(20), Duration::from_millis(500));

        let cancel = CancelToken::armed();
        let (c2, sp2, p2, o2) = (cancel.clone(), sp.clone(), prompt.clone(), opts.clone());
        let worker = std::thread::spawn(move || {
            let t = Instant::now();
            let r = generate_codes_cached(&mut cpu, &mut mtp, &sp2, &p2, &o2, &c2);
            (r, t.elapsed())
        });

        std::thread::sleep(wait);
        cancel.cancel();
        let (result, elapsed) = worker.join().expect("generation thread must not panic");

        let stopped = result.expect_err("an armed, cancelled token must abort the run");
        let n = stopped.frames(gw);
        assert!(n > 0, "cancel landed before any frame was produced - the test never reached mid-flight");
        assert!(
            n < max_frames / 2,
            "cancelled run produced {n} of {max_frames} frames - that is not an early stop"
        );
        // Promptly: the uncancelled run needs seconds by construction; this
        // one must come back within a small multiple of the cancel point.
        assert!(
            elapsed < wait * 4,
            "cancel took {elapsed:?} to be observed (cancelled at {wait:?}); \
             it must be seen between frames, not after max_frames"
        );

        // The partial codes are REAL output, not a truncated buffer: running the
        // same request uncancelled but capped at `n` frames yields them exactly.
        let mut cpu2 = CpuTalker::load(&talker_path);
        let mut mtp2 = CpuMtp::load(&mtp_path);
        let prefix_opts = GenOpts { max_frames: n, min_new: max_frames, seed: 5, ..GenOpts::default() };
        let prefix = generate_codes_cached(&mut cpu2, &mut mtp2, &sp, &prompt, &prefix_opts, &CancelToken::default())
            .expect("an unarmed token never cancels");
        assert_eq!(stopped.partial, prefix, "partial codes must be the genuine prefix of the full generation");

        eprintln!(
            "[cancel-test] per_frame={per_frame:?} max_frames={max_frames} cancelled at {wait:?}, \
             returned after {elapsed:?} with {n} frames"
        );
    }

    /// An already-cancelled token is refused before the prefix is streamed:
    /// zero frames, no Talker work at all.
    #[test]
    fn an_already_cancelled_token_produces_no_frames() {
        if gpu_disabled() {
            return;
        }
        let scratch = Scratch::new("pipeline-cancel-pre");
        let (talker_path, mtp_path) = synthetic_checkpoints(scratch.path(), 11);
        let sp = tiny_specials();
        let d = talker_test_cfg().d_model as usize;
        let prompt = tiny_prompt(d, 3, 2, 12);

        let mut cpu = CpuTalker::load(&talker_path);
        let mut mtp = CpuMtp::load(&mtp_path);
        let cancel = CancelToken::armed();
        cancel.cancel();
        let opts = GenOpts { max_frames: 64, min_new: 64, seed: 3, ..GenOpts::default() };
        let stopped = generate_codes_cached(&mut cpu, &mut mtp, &sp, &prompt, &opts, &cancel)
            .expect_err("a pre-cancelled token must abort immediately");
        assert!(stopped.partial.is_empty(), "nothing should have been generated");
        assert_eq!(stopped.frames(group_width()), 0);

        // And the unarmed default still runs to completion - the check must not
        // have become an unconditional early return.
        let full = generate_codes_cached(&mut cpu, &mut mtp, &sp, &prompt, &opts, &CancelToken::default())
            .expect("an unarmed token never cancels");
        assert_eq!(full.len() / group_width(), 64);
    }
}
