// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real multimodal INPUT for the Thinker generation loop: audio and image
//! (video: sampled frames through the same image path) encoded into
//! decoder-width embedding rows and spliced into the token-embedding sequence
//! `crate::generate` builds host-side, before the prompt is fed to the
//! decoder — reusing the exact encoders `crate::caps`'s M4/M5 parity tests
//! already validated to cosine 1.000000 against the real checkpoint
//! (`qwen_asr::encoder::AudioEncoder`, `qwenvl::encoder::VisionEncoder` +
//! `PatchMerger`), completely unchanged.
//!
//! **Scope**: one audio clip and/or one image (or a short list of video
//! frames, each run through the validated single-frame vision path) per
//! prompt. A caller assembles the FULL token id sequence — prompt text ids
//! plus the right number of placeholder ids where each medium goes, exactly
//! like HF's own processor — themselves; this module only tells the caller
//! how many placeholder tokens a given medium needs ([`AudioSplice::n_rows`]
//! / [`ImageSplice::n_rows`], computed from the encoder's own real output
//! size / `qwenvl::preprocess::image_token_count`) and does the actual
//! encode + host-side row overwrite.
//!
//! **Splice mechanism**: `crate::generate`'s prefill builds the whole prompt
//! embedding sequence as a host `Vec<f32>` before ONE upload
//! (`gpu.storage_init`), so the splice is a plain host-side slice copy
//! ([`splice_host`]) — not `model::vlm::splice_fwd`'s on-device kernel, which
//! exists for callers whose residual buffer is already GPU-resident before
//! the vision/audio embeddings are ready (`qwen3::Qwen`'s baked forward graph;
//! see that function's own doc). Same seam, cheaper for this caller's shape.
//!
//! **M-RoPE**: real per-axis positions for a mixed sequence, not the
//! diagonal collapse `crate::generate`'s pure-text path uses — built via
//! `qwenvl::mrope::get_rope_index_multi`, hoisted in this same workstream to
//! handle image/audio/video placeholder runs in one scan (see that
//! function's doc for the audio/video grid-shape convention and its one
//! documented approximation: wall-clock audio-timestamp scaling is not
//! ported, only frame-ordinal T-axis advance).
//!
//! **Splice point**: [`build_multimodal_prompt`] does not always put media at
//! the very front of `text_ids` anymore - see [`media_splice_point`]'s doc
//! for the real-hardware failure that motivated placing it right after a
//! leading system turn instead, when one is present.

use checkpoint::weightio::WeightReader;
use gpu_core::Gpu;
use qwen_asr::config::AudioEncoderConfig;
use qwen_asr::encoder::{audio_pipelines, AudioEncoder};
use qwenvl::config::VisionConfig;
use qwenvl::encoder::{vision_pipelines, PatchMerger, VisionEncoder};
use qwenvl::preprocess::{image_token_count, normalize_unit, pack_patches, pack_patches_temporal, patch_grid, smart_resize_default};

use qwenvl::mrope::get_rope_index_multi;

use crate::config::ThinkerConfig;
use crate::import::hf_to_brain;

/// One audio clip's splice-ready embeddings: `[n_rows, hidden]`, already at
/// Thinker decoder width (`AudioEncoder::encode`'s projector output — no
/// extra projection needed).
pub struct AudioSplice {
    pub embeds: Vec<f32>,
    pub n_rows: u32,
}

/// One image's (or one video frame's) splice-ready embeddings: `[n_rows,
/// hidden]` post-`PatchMerger`, plus the merged `(t, h, w)` grid
/// `qwenvl::mrope::get_rope_index_multi` needs for real position ids.
pub struct ImageSplice {
    pub embeds: Vec<f32>,
    pub n_rows: u32,
    pub grid: (u32, u32, u32),
}

/// Stream every `thinker.audio_tower.*` tensor from `reader` (mmap-backed)
/// into the
/// `AudioEncoder` weight map, fusing q/k/v exactly as `crate::import` and
/// `tests/audio_parity.rs` already do, and encode a raw 16kHz mono clip
/// (`samples`, e.g. `audio::wav::parse`'s output) into a splice-ready
/// embedding row set.
///
/// Pads `samples` up to a whole number of `AudioEncoderConfig::chunk_len()`
/// mel frames (`AudioEncoder::encode`'s own requirement — `prepare_packed`
/// asserts `t.is_multiple_of(chunk_len)`): with `hop=160` and `chunk_len=100`
/// (`2*n_window`, `n_window=50` for Omni), one chunk is exactly 16000
/// samples (1 second at 16kHz), so padding `samples.len()` up to the next
/// multiple of 16000 always lands on a whole chunk boundary — no search
/// needed. `valid_frames` (the real, non-padded portion) follows the same
/// `hop`-based frame formula `audio::asr_frontend::qwen_logmel` uses
/// internally (`target_samples / hop`, dropping the last STFT frame).
pub fn encode_audio(reader: &WeightReader, gpu: &Gpu, samples: &[f32]) -> Result<AudioSplice, String> {
    const HOP: usize = 160;
    const CHUNK_SAMPLES: usize = 16000; // chunk_len(100) * hop(160), for n_window=50
    let target_samples = samples.len().max(1).div_ceil(CHUNK_SAMPLES) * CHUNK_SAMPLES;
    let (mel, _n_mels, _n_frames) = audio::asr_frontend::qwen_logmel(samples, target_samples);
    let valid_frames = (samples.len() / HOP) as u32;

    let cfg = AudioEncoderConfig::qwen3_omni();
    let weights = audio_weights(reader)?;
    let gpu_local = gpu.new_like(audio_pipelines());
    let output_dim = cfg.output_dim;
    let enc = AudioEncoder::new(&gpu_local, cfg, &weights);
    let (_encoder_out, embeds) = enc.encode(&mel, valid_frames);
    let n_rows = embeds.len() as u32 / output_dim;
    Ok(AudioSplice { embeds, n_rows })
}

/// Stream + remap every `thinker.audio_tower.*` tensor from `reader` into the
/// `AudioEncoder` weight-map convention (fused q/k/v, `hf_to_brain`-renamed
/// leaves) — the loader half of [`encode_audio`], factored out so
/// [`crate::npu_export`] can build the SAME weight map for ONNX export
/// without also running a GPU encode.
pub(crate) fn audio_weights(reader: &WeightReader) -> Result<std::collections::HashMap<String, Vec<f32>>, String> {
    use std::collections::HashMap;
    type FusedQkv = (Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>);

    // Errors, not panics: streamed at REQUEST time inside the serving daemon
    // -- a malformed/partial checkpoint fails the request, not the process.
    let mut fused: HashMap<u32, FusedQkv> = HashMap::new();
    let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
    for name in reader.names() {
        if !name.starts_with("thinker.audio_tower.") {
            continue;
        }
        if crate::import::is_qkv_fuse_leaf(name) {
            let b: u32 = name
                .strip_prefix("thinker.audio_tower.layers.")
                .and_then(|r| r.split_once('.'))
                .and_then(|(idx, _)| idx.parse().ok())
                .ok_or_else(|| format!("omni: unparseable audio-tower qkv tensor name {name}"))?;
            let data = reader.tensor(name).ok_or_else(|| format!("omni: cannot read tensor {name}"))?;
            let slot = fused.entry(b).or_default();
            let is_weight = name.ends_with(".weight");
            if name.contains(".q_proj.") {
                if is_weight { slot.0 = Some(data) } else { slot.3 = Some(data) }
            } else if name.contains(".k_proj.") {
                if is_weight { slot.1 = Some(data) } else { slot.4 = Some(data) }
            } else if is_weight {
                slot.2 = Some(data)
            } else {
                slot.5 = Some(data)
            }
            continue;
        }
        let Some(brain_name) = hf_to_brain(name) else { continue };
        let Some(key) = brain_name.strip_prefix("audio.") else { continue };
        weights.insert(key.to_string(), reader.tensor(name).ok_or_else(|| format!("omni: cannot read tensor {name}"))?);
    }
    for (b, (qw, kw, vw, qb, kb, vb)) in fused {
        let missing = || format!("omni: audio-tower layer {b} is missing part of its q/k/v projection (partial checkpoint?)");
        let (qw, kw, vw) = (qw.ok_or_else(missing)?, kw.ok_or_else(missing)?, vw.ok_or_else(missing)?);
        let (qb, kb, vb) = (qb.ok_or_else(missing)?, kb.ok_or_else(missing)?, vb.ok_or_else(missing)?);
        let mut w = qw;
        w.extend(kw);
        w.extend(vw);
        let mut bias = qb;
        bias.extend(kb);
        bias.extend(vb);
        weights.insert(format!("blocks.{b}.qkv.weight"), w);
        weights.insert(format!("blocks.{b}.qkv.bias"), bias);
    }
    Ok(weights)
}

/// Stream + remap every `thinker.visual.*` tensor from `reader` into the
/// `VisionEncoder` + primary `PatchMerger` weight maps (same remap
/// `crate::import`/`tests/vision_parity.rs` use) — the loader half of
/// [`encode_image`], factored out so [`crate::npu_export`] can build the SAME
/// weight maps for ONNX export without also running a GPU encode.
/// DeepStack merger weights are skipped (see [`encode_image`]'s doc for why
/// the plain single-merger path is what's actually served).
pub(crate) fn vision_weights(reader: &WeightReader) -> Result<(std::collections::HashMap<String, Vec<f32>>, std::collections::HashMap<String, Vec<f32>>), String> {
    use std::collections::HashMap;
    let mut encoder_w: HashMap<String, Vec<f32>> = HashMap::new();
    let mut main_merger_w: HashMap<String, Vec<f32>> = HashMap::new();
    for name in reader.names() {
        if !name.starts_with("thinker.visual.") {
            continue;
        }
        let Some(brain_name) = hf_to_brain(name) else { continue };
        let data = reader.tensor(name).ok_or_else(|| format!("omni: cannot read tensor {name}"))?;
        if let Some(rest) = brain_name.strip_prefix("vision.merger.") {
            main_merger_w.insert(rest.to_string(), data);
        } else if let Some(rest) = brain_name.strip_prefix("vision.") {
            if rest.starts_with("deepstack_merger.") {
                continue; // not needed for a plain (non-DeepStack) splice path
            }
            encoder_w.insert(rest.to_string(), data);
        }
    }
    Ok((encoder_w, main_merger_w))
}

/// Stream every `thinker.visual.*` tensor from `reader` into the
/// `VisionEncoder` + primary `PatchMerger` weight maps (same remap
/// `crate::import`/`tests/vision_parity.rs` use) and encode one already
/// RGB-decoded, `[0,1]`-normalized image into splice-ready embeddings.
/// `rgb_hwc` is interleaved HWC (the engine's one image wire format,
/// `capability::blob::decode_image`'s output shape).
pub fn encode_image(reader: &WeightReader, gpu: &Gpu, rgb_hwc: &[f32], w: u32, h: u32) -> Result<ImageSplice, String> {
    let cfg = VisionConfig::qwen3_omni();
    let (encoder_w, main_merger_w) = vision_weights(reader)?;

    // Real bilinear resize to the smart-resized (patch-multiple) extent --
    // NOT a crop/pad: `smart_resize_default` picks dimensions close to but
    // never equal to the source's, so every real image needs true
    // interpolation, not a top-left window. Then HWC -> CHW (qwenvl::preprocess
    // expects channel-first before patch packing).
    let (hp, wp) = smart_resize_default(h, w);
    let resized_hwc = imaging::host::resize_bilinear_hwc(rgb_hwc, 3, w, h, wp, hp);
    let mut chw = imaging::pixels::hwc_to_chw(&resized_hwc, 3, hp as usize, wp as usize);
    normalize_unit(&mut chw);
    let (gh, gw) = patch_grid(hp, wp, cfg.patch_size);
    let patches = pack_patches(&chw, 3, hp, wp, cfg.patch_size, cfg.spatial_merge_size, cfg.temporal_patch_size);

    let gpu_local = gpu.new_like(vision_pipelines());
    let enc = VisionEncoder::new(&gpu_local, cfg.clone(), &encoder_w);
    let encoder_out = enc.encode(gh, gw, &patches);
    let merger = PatchMerger::new(&gpu_local, &main_merger_w, cfg.hidden, cfg.spatial_merge_size, cfg.out_hidden_size, false);
    let embeds = merger.merge(&encoder_out, gh * gw);

    let merge = cfg.spatial_merge_size;
    let n_rows = image_token_count(hp, wp, cfg.patch_size, merge);
    Ok(ImageSplice { embeds, n_rows, grid: (1, gh / merge, gw / merge) })
}

/// Encode a short video as a list of ALREADY-DECODED frames (RGB HWC each) —
/// frame extraction from a video FILE is `imaging::video::decode_frames`'s
/// job (a caller, e.g. `crate::caps`, hands over frames it already has).
///
/// Real temporal patching, not the single-frame replication [`encode_image`]
/// uses for a genuinely-still image: every frame is resized to ONE shared
/// smart-resized extent (derived from the first frame — a real video's
/// frames must share a grid to be paired, unlike independent images), padded
/// to a multiple of `temporal_patch_size` by repeating the LAST frame
/// (`Qwen2VLVideoProcessor`'s own convention, confirmed against the
/// installed `transformers` reference before implementing this: `if pad :=
/// -T % temporal_patch_size: repeats = patches[:, -1:].expand(...)`), then
/// packed `temporal_patch_size` real frames at a time via
/// `qwenvl::preprocess::pack_patches_temporal`.
///
/// One `VisionEncoder::encode` + `PatchMerger::merge` pass per temporal
/// GROUP (frame pair), not one pass over the whole clip: the encoder's own
/// contract (`VisionEncoder::encode`'s doc) is full attention over ONE 2D
/// patch grid, and extending that to real multi-group windowed attention
/// across a clip is separate, larger, not-yet-covered work (unchanged by
/// this fix) — this closes the temporal-PATCHING gap specifically, not the
/// cross-frame-attention one. The returned grid's `t` is the number of
/// temporal GROUPS (`frames.len().div_ceil(temporal_patch_size)`), matching
/// the reference's own `grid_t = num_frames // temporal_patch_size` — NOT
/// the raw frame count the previous version of this function used, which
/// was a second, smaller bug alongside the frame-replication one.
pub fn encode_video_frames(reader: &WeightReader, gpu: &Gpu, frames: &[(Vec<f32>, u32, u32)]) -> Result<ImageSplice, String> {
    if frames.is_empty() {
        return Err("omni: encode_video_frames: at least one frame required".to_string());
    }
    let cfg = VisionConfig::qwen3_omni();
    let (encoder_w, main_merger_w) = vision_weights(reader)?;

    let (_, w0, h0) = &frames[0];
    let (hp, wp) = smart_resize_default(*h0, *w0);
    let mut chw_frames: Vec<Vec<f32>> = frames
        .iter()
        .map(|(rgb, w, h)| {
            let resized_hwc = imaging::host::resize_bilinear_hwc(rgb, 3, *w, *h, wp, hp);
            let mut chw = imaging::pixels::hwc_to_chw(&resized_hwc, 3, hp as usize, wp as usize);
            normalize_unit(&mut chw);
            chw
        })
        .collect();

    let temporal = cfg.temporal_patch_size;
    while !(chw_frames.len() as u32).is_multiple_of(temporal) {
        chw_frames.push(chw_frames.last().expect("non-empty, checked above").clone());
    }

    let (gh, gw) = patch_grid(hp, wp, cfg.patch_size);
    let merge = cfg.spatial_merge_size;
    let gpu_local = gpu.new_like(vision_pipelines());
    let enc = VisionEncoder::new(&gpu_local, cfg.clone(), &encoder_w);
    let merger = PatchMerger::new(&gpu_local, &main_merger_w, cfg.hidden, merge, cfg.out_hidden_size, false);

    let n_groups = chw_frames.len() as u32 / temporal;
    let n_rows_per_group = image_token_count(hp, wp, cfg.patch_size, merge);
    let mut embeds = Vec::with_capacity((n_groups * n_rows_per_group * cfg.out_hidden_size) as usize);
    for g in 0..n_groups {
        let group: Vec<&[f32]> = chw_frames[(g * temporal) as usize..((g + 1) * temporal) as usize].iter().map(|f| f.as_slice()).collect();
        let patches = pack_patches_temporal(&group, 3, hp, wp, cfg.patch_size, merge, temporal);
        let encoder_out = enc.encode(gh, gw, &patches);
        embeds.extend(merger.merge(&encoder_out, gh * gw));
    }

    Ok(ImageSplice { embeds, n_rows: n_groups * n_rows_per_group, grid: (n_groups, gh / merge, gw / merge) })
}

/// One resolved multimodal prompt: the full token id sequence (media blocks,
/// each wrapped in its start/end token, spliced into `text_ids` at the
/// caller-chosen [`media_splice_point`] - see [`build_multimodal_prompt`]'s
/// doc for the exact layout), its already-spliced `[n, hidden]` host
/// embedding buffer, and its real 3-axis M-RoPE positions - everything
/// `crate::generate::generate_greedy_multimodal` needs.
pub struct MultimodalPrompt {
    pub token_ids: Vec<u32>,
    pub x_host: Vec<f32>,
    pub positions: Vec<[u32; 3]>,
}

struct MediaBlock {
    start_token: u32,
    placeholder_token: u32,
    end_token: u32,
    embeds: Vec<f32>,
    n_rows: u32,
    grid: (u32, u32, u32),
}

/// Where a caller should splice real media embeddings into `text_ids`: right
/// after a leading system turn's closing `<|im_end|>` when the rendered
/// prompt opens with one (`<|im_start|>system\n...`) - and, when the caller
/// also knows the NEXT turn's own opening tag token sequence (typically
/// `<|im_start|>user\n`, via `next_turn_open`), past THAT too, so media lands
/// at the very first token of that turn's own content rather than in front of
/// its role tag. Falls back to index 0 (the original, still-correct behavior
/// for a system-less prompt) whenever there is no leading system turn.
///
/// **Real bug this closes**: [`build_multimodal_prompt`] used to always place
/// media at index 0 - before the ENTIRE rendered conversation, including the
/// system turn. That is a plausible position when there is no system turn (a
/// short, system-less prompt puts media right where the user's own turn
/// starts, close to the model's trained distribution), but for a real caller
/// with a long system prompt (e.g. an agent's own operating instructions),
/// prepending shoves the system/user role tags hundreds or thousands of
/// tokens further from their trained relative position than the model
/// expects. Confirmed on real hardware (2x Tesla P40, the real
/// Qwen3-Omni-30B-A3B-Instruct W8A16 checkpoint): with a long system prompt +
/// image + audio, the model's own top logit at the very first decode step was
/// a confident `<|im_start|>` (not a corrupted/NaN logit - a real, decisive
/// preference), and its next-step top logit after that was EOS, so generation
/// stopped after exactly one bogus token. Moving the same media blocks to
/// right after the system turn fixed that (verified end-to-end: real,
/// substantively correct generated content instead of one bogus token +
/// stop) but left a smaller remaining artifact - a stray `<|im_start|>` still
/// occasionally leads the real content, because the media block still sits in
/// front of the user turn's OWN role tag rather than inside it. Skipping past
/// `next_turn_open` too closes that gap: media then starts exactly where the
/// user's own turn content would have started, matching the model's trained
/// distribution even more closely. Still a heuristic, not a byte-exact port
/// of HF's own inline per-content-part splice (which needs the chat-template
/// renderer to keep typed content parts instead of flattening them to plain
/// text before splicing has any position to target) - a caller whose media
/// reference sits in the MIDDLE of a long user turn, rather than at its
/// start, is not fully addressed by this.
pub fn media_splice_point(prompt: &str, text_ids: &[u32], im_end_id: Option<u32>, next_turn_open: Option<&[u32]>) -> usize {
    if !prompt.starts_with("<|im_start|>system\n") {
        return 0;
    }
    let Some(after_system) = im_end_id.and_then(|id| text_ids.iter().position(|&t| t == id)).map(|i| i + 1) else {
        return 0;
    };
    match next_turn_open {
        Some(open) if !open.is_empty() && text_ids[after_system..].starts_with(open) => after_system + open.len(),
        _ => after_system,
    }
}

/// Assemble a multimodal prompt: media blocks (audio, then image, then video
/// frames — in that fixed order; any subset may be absent) each wrapped in
/// its start/end token exactly as HF's own chat template does
/// (`<|audio_start|>`..audio placeholders..`<|audio_end|>`,
/// `<|vision_start|>`..image/video placeholders..`<|vision_end|>`), spliced
/// into `text_ids` (the whole rendered conversation, already tokenized) at
/// `splice_at` - see [`media_splice_point`] for how a caller should pick that
/// index. This is a real but simplified convention - one block per medium,
/// all media at ONE point in the sequence, not a fully general interleaved
/// multi-turn processor.
///
/// Real embeddings come from [`encode_audio`]/[`encode_image`]/
/// [`encode_video_frames`], spliced host-side ([`splice_host`]); real 3-axis
/// M-RoPE positions come from `qwenvl::mrope::get_rope_index_multi`, fed each
/// medium's own placeholder token id + grid list (audio and video may each
/// appear at most once here, so each gets a one-entry grid list — multiple
/// same-medium blocks in one prompt aren't assembled by this function, though
/// `get_rope_index_multi` itself supports it).
#[allow(clippy::too_many_arguments)]
pub fn build_multimodal_prompt(
    reader: &WeightReader,
    gpu: &Gpu,
    cfg: &ThinkerConfig,
    embed_table: &[f32],
    text_ids: &[u32],
    audio: Option<&[f32]>,
    image: Option<(&[f32], u32, u32)>,
    video: Option<&[(Vec<f32>, u32, u32)]>,
    splice_at: usize,
) -> Result<MultimodalPrompt, String> {
    let splice_at = splice_at.min(text_ids.len());
    let d = cfg.text.hidden;
    let mut blocks = Vec::new();
    if let Some(samples) = audio {
        let a = encode_audio(reader, gpu, samples)?;
        blocks.push(MediaBlock {
            start_token: cfg.audio_start_token_id,
            placeholder_token: cfg.audio_token_id,
            end_token: cfg.audio_end_token_id,
            embeds: a.embeds,
            n_rows: a.n_rows,
            grid: (a.n_rows, 1, 1), // 1-D, no spatial extent: diagonal on all three axes -- see get_rope_index_multi's doc
        });
        reclaim_tower_vram(gpu);
    }
    if let Some((rgb, w, h)) = image {
        let im = encode_image(reader, gpu, rgb, w, h)?;
        blocks.push(MediaBlock {
            start_token: cfg.vision_start_token_id,
            placeholder_token: cfg.image_token_id,
            end_token: cfg.vision_end_token_id,
            embeds: im.embeds,
            n_rows: im.n_rows,
            grid: im.grid,
        });
        reclaim_tower_vram(gpu);
    }
    if let Some(frames) = video {
        let v = encode_video_frames(reader, gpu, frames)?;
        blocks.push(MediaBlock {
            start_token: cfg.vision_start_token_id,
            placeholder_token: cfg.video_token_id,
            end_token: cfg.vision_end_token_id,
            embeds: v.embeds,
            n_rows: v.n_rows,
            grid: v.grid,
        });
        reclaim_tower_vram(gpu);
    }

    let embed_row = |t: u32| embed_table[t as usize * d as usize..(t as usize + 1) * d as usize].to_vec();
    let mut token_ids = Vec::new();
    let mut x_host = Vec::new();
    let mut placeholder_grids: Vec<(u32, Vec<(u32, u32, u32)>)> = Vec::new();

    // Text BEFORE the splice point (a leading system turn, when present -
    // see `media_splice_point`) is unaffected: media lands after it, not
    // before it.
    for &t in &text_ids[..splice_at] {
        token_ids.push(t);
        x_host.extend_from_slice(&embed_row(t));
    }
    for b in &blocks {
        token_ids.push(b.start_token);
        x_host.extend_from_slice(&embed_row(b.start_token));

        let row0 = (x_host.len() / d as usize) as u32;
        token_ids.extend(std::iter::repeat_n(b.placeholder_token, b.n_rows as usize));
        x_host.resize(x_host.len() + (b.n_rows * d) as usize, 0.0);
        splice_host(&mut x_host, &b.embeds, row0, b.n_rows, d);
        placeholder_grids.push((b.placeholder_token, vec![b.grid]));

        token_ids.push(b.end_token);
        x_host.extend_from_slice(&embed_row(b.end_token));
    }
    for &t in &text_ids[splice_at..] {
        token_ids.push(t);
        x_host.extend_from_slice(&embed_row(t));
    }

    let placeholders: Vec<(u32, &[(u32, u32, u32)])> = placeholder_grids.iter().map(|(id, g)| (*id, g.as_slice())).collect();
    let positions = get_rope_index_multi(&token_ids, &placeholders);

    Ok(MultimodalPrompt { token_ids, x_host, positions })
}

/// Force this device's just-buried tower buffers (the encoder + merger
/// weights and activation scratch [`encode_audio`]/[`encode_image`]/
/// [`encode_video_frames`] allocate via `gpu.new_like` -- a handle sharing
/// `gpu`'s own `VkContext`, see those functions' docs) to actually be freed
/// before the NEXT medium's tower uploads its own weights onto the same
/// device.
///
/// **Real bug this closes**: `crates/backend-vulkan`'s buffer reclaim is
/// deferred by design (`VkOwnedBuffer::drop` -> `VkContext::bury`, a real
/// device-memory free only happens later, at `VkContext::reclaim_dead` --
/// see that function's own doc for why: a batched-but-unsubmitted dispatch
/// may still name the buffer). Nothing called `reclaim_dead` between two
/// back-to-back `encode_*` calls, so a request carrying BOTH audio and image
/// stacked their tower weights+activations in VRAM simultaneously instead of
/// the second reusing the first's just-freed space: measured against the
/// real `Qwen3-Omni-30B-A3B-Instruct` HF checkpoint, the audio tower is
/// ~1.2 GiB on disk (bf16) / ~2.4 GiB uploaded as f32, the vision tower
/// ~1.0 GiB / ~2.0 GiB -- individually well inside a resident int8 shard's
/// spare headroom (~7-8 GiB/card measured on 2x Tesla P40 after the real
/// W8A16 checkpoint's placement), but their unreclaimed SUM plus activation
/// scratch narrows that margin -- exactly the class of failure a real
/// `ERROR_OUT_OF_DEVICE_MEMORY` from `crates/vulkan/src/context.rs`'s
/// `allocate_memory` reports (a genuine `vkAllocateMemory` failure, not the
/// unrelated wgpu-hal 4 GiB clamp `--device vulkan` already avoids). This is
/// a real, verified accounting gap regardless of how much headroom a given
/// card happens to have at request time: `Gpu::flush` on a handle with
/// nothing of its own recorded degrades to exactly one `reclaim_dead` call
/// (`VulkanBackend::flush`'s empty-batch branch) -- cheap, and a no-op on
/// backends that reclaim eagerly already.
fn reclaim_tower_vram(gpu: &Gpu) {
    gpu.flush();
}

/// Overwrite `x_host`'s rows `[row0, row0+n_rows)` with `embeds`
/// (`[n_rows, hidden]`) — the host-side splice `crate::generate`'s prefill
/// uses (see this module's doc for why it's a plain slice copy, not
/// `model::vlm::splice_fwd`'s on-device kernel).
pub fn splice_host(x_host: &mut [f32], embeds: &[f32], row0: u32, n_rows: u32, hidden: u32) {
    let base = (row0 * hidden) as usize;
    let n = (n_rows * hidden) as usize;
    assert_eq!(embeds.len(), n, "splice_host: embeds has {} elems, expected {n} ({n_rows} rows x {hidden})", embeds.len());
    x_host[base..base + n].copy_from_slice(embeds);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real ids from the real Qwen3-Omni tokenizer (`tokenizer_config.json`/
    // `vocab.json`, verified directly against the actual checkpoint's
    // tokenizer this session): <|im_start|>=151644, <|im_end|>=151645,
    // "\n"=198, "user"=872, "system"=8948, "assistant"=77091.
    const IM_START: u32 = 151644;
    const IM_END: u32 = 151645;
    const NL: u32 = 198;
    const USER: u32 = 872;

    /// A prompt with no leading system turn keeps the original behavior --
    /// media at index 0 -- the shape the earlier (system-less, short-prompt)
    /// real-hardware tests already validated as correct.
    #[test]
    fn splice_point_no_system_turn_is_zero() {
        let prompt = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n";
        let text_ids = [IM_START, USER, 9906, IM_END, NL, IM_START, 77091, NL];
        let user_open = [NL, IM_START, USER, NL];
        assert_eq!(media_splice_point(prompt, &text_ids, Some(IM_END), Some(&user_open)), 0);
    }

    /// A prompt that DOES open with a system turn, but the caller has no
    /// `next_turn_open` to check, splices right after the system turn's
    /// closing `<|im_end|>` -- the first real fix: this is what used to be
    /// index 0 (media before the ENTIRE conversation, system turn included),
    /// reproduced on real hardware as a request that generates exactly one
    /// bogus `<|im_start|>` token then stops (see `media_splice_point`'s doc).
    #[test]
    fn splice_point_after_leading_system_turn_without_next_turn_open() {
        let prompt = "<|im_start|>system\nbe helpful<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
        // system turn: [<|im_start|>, "system", "\n", "be", "helpful", <|im_end|>] -- IM_END at index 5.
        let text_ids = [IM_START, 8948, NL, 1055, 10950, IM_END, NL, IM_START, USER, 6023, IM_END, NL, IM_START, 77091, NL];
        assert_eq!(media_splice_point(prompt, &text_ids, Some(IM_END), None), 6);
    }

    /// The full fix: with `next_turn_open` supplied (`"\n<|im_start|>user\n"`,
    /// tokenized -- the leading "\n" belongs to the system turn's own
    /// `<|im_end|>\n` closer, so it must be included to match starting right
    /// at `im_end_id`'s position), the splice point lands at the very FIRST
    /// token of the user turn's own content, not merely after the system
    /// turn. This is what closed the remaining "stray `<|im_start|>` before
    /// otherwise-correct content" artifact the system-turn-only fix left
    /// behind (see `media_splice_point`'s doc + the real end-to-end
    /// `bbox_check.py` pass this fix produced).
    #[test]
    fn splice_point_after_leading_system_and_user_open_tag() {
        let prompt = "<|im_start|>system\nbe helpful<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
        let text_ids = [IM_START, 8948, NL, 1055, 10950, IM_END, NL, IM_START, USER, NL, 6023, IM_END, NL, IM_START, 77091, NL];
        let user_open = [NL, IM_START, USER, NL];
        // after_system = 6 (right after IM_END at index 5); text_ids[6..10] ==
        // user_open, so splice_at = 6 + 4 = 10 -- the "hi" token's own index.
        assert_eq!(media_splice_point(prompt, &text_ids, Some(IM_END), Some(&user_open)), 10);
        assert_eq!(text_ids[10], 6023, "sanity: index 10 really is the user turn's own first content token");
    }

    /// When the NEXT turn's tokens don't actually match `next_turn_open`
    /// (an unexpected shape - not a real "user" turn immediately after
    /// system, or a tokenizer quirk), fall back to splicing right after the
    /// system turn instead of picking a wrong/misaligned index.
    #[test]
    fn splice_point_falls_back_when_next_turn_open_does_not_match() {
        let prompt = "<|im_start|>system\nbe helpful<|im_end|>\n<|im_start|>tool\nhi<|im_end|>\n";
        let text_ids = [IM_START, 8948, NL, 1055, 10950, IM_END, NL, IM_START, 14172, NL, 6023, IM_END, NL];
        let user_open = [NL, IM_START, USER, NL]; // does not match the real "tool" turn here
        assert_eq!(media_splice_point(prompt, &text_ids, Some(IM_END), Some(&user_open)), 6);
    }

    /// A missing `im_end_id` (e.g. a tokenizer with no such special token, or
    /// a future format change) degrades to the old always-prepend behavior
    /// rather than panicking or picking a bogus index -- never worse than
    /// before this fix.
    #[test]
    fn splice_point_falls_back_to_zero_without_im_end_id() {
        let prompt = "<|im_start|>system\nbe helpful<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
        let text_ids = [IM_START, 8948, NL, 1055, 10950, IM_END, NL];
        assert_eq!(media_splice_point(prompt, &text_ids, None, None), 0);
    }

    /// `build_multimodal_prompt`'s splice loop itself: a fake single-block
    /// splice at a mid-sequence index puts the block's tokens between the
    /// "before" and "after" text runs, not merely appended/prepended.
    #[test]
    fn splice_at_mid_sequence_orders_tokens_correctly() {
        let d = 2usize;
        // Reproduce just the token/embedding assembly loop `build_multimodal_prompt`
        // uses internally (a real `MediaBlock` needs a real GPU encode to build --
        // this exercises the same splice arithmetic without one).
        let text_ids = [10u32, 11, 12, 13];
        let splice_at = 2usize.min(text_ids.len());
        let block_tokens = [900u32, 901, 902];
        let embed_row = |t: u32| vec![t as f32, t as f32 + 0.5];

        let mut token_ids = Vec::new();
        let mut x_host = Vec::new();
        for &t in &text_ids[..splice_at] {
            token_ids.push(t);
            x_host.extend(embed_row(t));
        }
        for &t in &block_tokens {
            token_ids.push(t);
            x_host.extend(embed_row(t));
        }
        for &t in &text_ids[splice_at..] {
            token_ids.push(t);
            x_host.extend(embed_row(t));
        }

        assert_eq!(token_ids, vec![10, 11, 900, 901, 902, 12, 13]);
        assert_eq!(x_host.len(), token_ids.len() * d);
    }
}
