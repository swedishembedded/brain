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

use checkpoint::weightio::WeightReader;
use gpu_core::Gpu;
use qwen_asr::config::AudioEncoderConfig;
use qwen_asr::encoder::{audio_pipelines, AudioEncoder};
use qwenvl::config::VisionConfig;
use qwenvl::encoder::{vision_pipelines, PatchMerger, VisionEncoder};
use qwenvl::preprocess::{image_token_count, normalize_unit, pack_patches, patch_grid, smart_resize_default};

use qwenvl::mrope::get_rope_index_multi;

use crate::config::ThinkerConfig;
use crate::import::hf_to_brain;

/// One audio clip's splice-ready embeddings: `[n_rows, hidden]`, already at
/// Thinker decoder width (`AudioEncoder::encode`'s projector output — no
/// extra projection needed, see `docs/models/omni/status.md` M4).
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

/// Stream every `thinker.audio_tower.*` tensor from `reader` (mmap-backed —
/// `docs/models/omni/status.md`'s mmap/streaming instruction) into the
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
/// frame extraction (mp4 demux) is out of scope for this crate (the same
/// scope line the M14 plan drew: `av`-based extraction belongs in brain-py,
/// upstream of this call), so a caller (`crate::caps`) hands over frames it
/// already has. Each frame runs through the SAME validated single-frame
/// [`encode_image`] path (real multi-frame temporal patching is
/// `qwenvl`'s own not-yet-covered work, per its M5 status entry); embeddings
/// concatenate row-major, and the returned grid's `t` is the frame count —
/// the meshgrid `qwenvl::mrope::get_rope_index_multi` needs is identical in
/// shape to a real multi-frame video grid, only the per-frame encode itself
/// is the (documented) approximation.
pub fn encode_video_frames(reader: &WeightReader, gpu: &Gpu, frames: &[(Vec<f32>, u32, u32)]) -> Result<ImageSplice, String> {
    if frames.is_empty() {
        return Err("omni: encode_video_frames: at least one frame required".to_string());
    }
    let mut embeds = Vec::new();
    let mut n_rows = 0u32;
    let (mut fh, mut fw) = (0u32, 0u32);
    for (rgb, w, h) in frames {
        let one = encode_image(reader, gpu, rgb, *w, *h)?;
        embeds.extend(one.embeds);
        n_rows += one.n_rows;
        (fh, fw) = (one.grid.1, one.grid.2);
    }
    Ok(ImageSplice { embeds, n_rows, grid: (frames.len() as u32, fh, fw) })
}

/// One resolved multimodal prompt: the full token id sequence (media blocks,
/// each wrapped in its start/end token, followed by the user's text — see
/// [`build_multimodal_prompt`]'s doc for the exact layout), its already-
/// spliced `[n, hidden]` host embedding buffer, and its real 3-axis M-RoPE
/// positions — everything `crate::generate::generate_greedy_multimodal` needs.
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

/// Assemble a multimodal prompt: media blocks (audio, then image, then video
/// frames — in that fixed order; any subset may be absent) each wrapped in
/// its start/end token exactly as HF's own chat template does
/// (`<|audio_start|>`..audio placeholders..`<|audio_end|>`,
/// `<|vision_start|>`..image/video placeholders..`<|vision_end|>`), followed
/// by `text_ids` (the user's own already-tokenized text). This is a real but
/// simplified convention — one block per medium, media always before text,
/// not a fully general interleaved multi-turn processor — documented as such
/// in `docs/models/omni/status.md`'s multimodal entry.
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
) -> Result<MultimodalPrompt, String> {
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
            grid: (a.n_rows, 1, 1), // 1-D: T advances, H/W pinned -- see get_rope_index_multi's doc
        });
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
    }

    let embed_row = |t: u32| embed_table[t as usize * d as usize..(t as usize + 1) * d as usize].to_vec();
    let mut token_ids = Vec::new();
    let mut x_host = Vec::new();
    let mut placeholder_grids: Vec<(u32, Vec<(u32, u32, u32)>)> = Vec::new();

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
    for &t in text_ids {
        token_ids.push(t);
        x_host.extend_from_slice(&embed_row(t));
    }

    let placeholders: Vec<(u32, &[(u32, u32, u32)])> = placeholder_grids.iter().map(|(id, g)| (*id, g.as_slice())).collect();
    let positions = get_rope_index_multi(&token_ids, &placeholders);

    Ok(MultimodalPrompt { token_ids, x_host, positions })
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
