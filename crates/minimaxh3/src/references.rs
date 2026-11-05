// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for conditioning generative video
// on reference media for its clients. If your team needs expertise in
// multimodal diffusion pipelines then you can procure our services by sending
// an email to info@swedishembedded.com.

//! `ref2va` references: decoded media in, [`EncodedReference`]s out.
//!
//! Two steps, ported from `MiniMaxH3Ref2VASetupStep` and
//! `MiniMaxH3Ref2VAReferenceEncoderStep`:
//!
//! * [`normalize_references`] puts every reference on the model's own rates
//!   and resolutions - an image onto its own short edge (2048 for the
//!   released checkpoint, upscaling included, no area cap), a video onto
//!   24 fps (whole frames dropped or held, as `ffmpeg`'s `fps` filter does),
//!   truncated to the generated frame count and put on the canvas its own
//!   aspect resolves to, a soundtrack truncated to the generated duration at
//!   its native rate, upmixed to stereo and resampled to the audio VAE's
//!   rate. Both rescales are Pillow's LANCZOS and the resample is
//!   torchaudio's, each reproduced exactly.
//! * [`encode_references`] encodes the frames through the video VAE (noised
//!   to the conditioning level, as keyframes are) and each soundtrack
//!   channel through the audio VAE encoder (posterior mean, normalized,
//!   never noised).
//!
//! The references' order is the request's: it labels them in the
//! presentation and lays them out on the shared rotary clock.

use imaging::host::resize_lanczos_pil;
use imaging::Rgb8;
use vae::blocks::Tensors;

use crate::audio_encoder::AudioEncoderConfig;
use crate::pipeline::{check_reference_kinds, encode_visual_condition, resolve_canvas_size, EncodedReference, ReferenceGeometry, ReferenceKind, VisualConditionRequest, AUDIO_CHANNELS, AUDIO_SAMPLE_RATE, CANVAS_MAX_PIXELS, CANVAS_SHORT_EDGE, FPS, REFERENCE_IMAGE_SHORT_EDGE};

/// A decoded waveform: one `Vec` per channel (mono or stereo) at
/// `sample_rate`.
#[derive(Clone, Debug)]
pub struct Soundtrack {
    pub channels: Vec<Vec<f32>>,
    pub sample_rate: u32,
}

/// One reference as the request supplies it, decoded but not yet
/// normalized.
#[derive(Clone, Debug)]
pub enum ReferenceMedia {
    Image(Rgb8),
    /// Frames at `fps`, all the same size; `soundtrack` is conditioned on as
    /// well when present.
    Video { frames: Vec<Rgb8>, fps: f64, soundtrack: Option<Soundtrack> },
    Audio(Soundtrack),
}

impl ReferenceMedia {
    pub fn kind(&self) -> ReferenceKind {
        match self {
            ReferenceMedia::Image(_) => ReferenceKind::Image,
            ReferenceMedia::Video { .. } => ReferenceKind::Video,
            ReferenceMedia::Audio(_) => ReferenceKind::Audio,
        }
    }
}

/// A stereo waveform at [`ReferenceGeometry`]'s audio rate, channel 0 first.
pub type Stereo = [Vec<f32>; 2];

/// One reference on the model's own rates and resolutions.
#[derive(Clone, Debug)]
pub enum NormalizedReference {
    Image(Rgb8),
    /// 24 fps frames on the reference's own canvas.
    Video { frames: Vec<Rgb8>, soundtrack: Option<Stereo> },
    Audio(Stereo),
}

impl NormalizedReference {
    pub fn kind(&self) -> ReferenceKind {
        match self {
            NormalizedReference::Image(_) => ReferenceKind::Image,
            NormalizedReference::Video { .. } => ReferenceKind::Video,
            NormalizedReference::Audio(_) => ReferenceKind::Audio,
        }
    }
}

/// The geometry references are normalized onto: the canvas rule a video
/// reference follows (the generated video's own), the separate short edge
/// of an image reference, the frame rate and the audio VAE's sample rate.
/// [`ReferenceCanvas::released`] is the released checkpoint's; the fields
/// are the reference implementation's configuration, so a fine-tune trained
/// at another geometry is configured rather than patched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceCanvas {
    /// What both canvas axes round to (`canvas_multiple`: the VAE's spatial
    /// compression times the DiT patch).
    pub multiple: u32,
    pub canvas_short_edge: u32,
    pub canvas_max_pixels: u32,
    pub image_short_edge: u32,
    pub fps: f32,
    pub audio_sample_rate: u32,
}

impl ReferenceCanvas {
    pub fn released(multiple: u32) -> ReferenceCanvas {
        ReferenceCanvas { multiple, canvas_short_edge: CANVAS_SHORT_EDGE, canvas_max_pixels: CANVAS_MAX_PIXELS, image_short_edge: REFERENCE_IMAGE_SHORT_EDGE, fps: FPS, audio_sample_rate: AUDIO_SAMPLE_RATE }
    }
}

/// Validate a request's references and normalize each, in order.
/// `num_frames` is the generated frame count, already aligned to the video
/// VAE's `17 * n + 5`: videos are truncated to it and soundtracks to its
/// duration.
pub fn normalize_references(media: &[ReferenceMedia], num_frames: u32, canvas: &ReferenceCanvas) -> Result<Vec<NormalizedReference>, String> {
    check_reference_kinds(media.iter().map(ReferenceMedia::kind))?;
    let max_duration = num_frames as f64 / canvas.fps as f64;
    media
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let normalized = match m {
                ReferenceMedia::Image(img) => normalize_image(img, canvas).map(NormalizedReference::Image),
                ReferenceMedia::Video { frames, fps, soundtrack } => normalize_video(frames, *fps, num_frames, canvas).and_then(|frames| {
                    let soundtrack = soundtrack.as_ref().map(|s| normalize_soundtrack(s, max_duration, canvas.audio_sample_rate)).transpose()?;
                    Ok(NormalizedReference::Video { frames, soundtrack })
                }),
                ReferenceMedia::Audio(s) => normalize_soundtrack(s, max_duration, canvas.audio_sample_rate).map(NormalizedReference::Audio),
            };
            normalized.map_err(|e| format!("ref2va reference {i}: {e}"))
        })
        .collect()
}

/// An image onto its own short edge, both axes rounded (half to even, as
/// Python's `round`) to the canvas multiple, LANCZOS-resized when that
/// changes its size. Within 1:4 and 4:1.
pub fn normalize_image(img: &Rgb8, canvas: &ReferenceCanvas) -> Result<Rgb8, String> {
    let (width, height) = (img.w as f64, img.h as f64);
    if img.w == 0 || img.h == 0 {
        return Err(format!("a reference image must have a positive size, got {}x{}", img.w, img.h));
    }
    if width > 4.0 * height || height > 4.0 * width {
        return Err(format!("a reference image must be within 1:4 and 4:1, got {}x{}", img.w, img.h));
    }
    let scale = canvas.image_short_edge as f64 / width.min(height);
    let m = canvas.multiple as f64;
    let side = |v: f64| canvas.multiple.max(((v * scale / m).round_ties_even() * m) as u32);
    let (target_w, target_h) = (side(width), side(height));
    Ok(if (target_w, target_h) == (img.w, img.h) { img.clone() } else { resize_lanczos_pil(img, target_w as usize, target_h as usize) })
}

/// A video onto `canvas.fps` - every frame held until the slot of the next,
/// the last until the slot the stream's end rounds to - then truncated to
/// `num_frames` and put on the canvas its own aspect resolves to.
pub fn normalize_video(frames: &[Rgb8], fps: f64, num_frames: u32, canvas: &ReferenceCanvas) -> Result<Vec<Rgb8>, String> {
    let Some(first) = frames.first() else {
        return Err("a reference video has no frames".to_string());
    };
    if let Some(odd) = frames.iter().find(|f| (f.w, f.h) != (first.w, first.h)) {
        return Err(format!("a reference video's frames must share one size, got {}x{} and {}x{}", first.w, first.h, odd.w, odd.h));
    }
    if fps.is_nan() || fps <= 0.0 {
        return Err(format!("a reference video must have a positive frame rate, got {fps}"));
    }
    let target_fps = canvas.fps as f64;
    let mut resampled: Vec<&Rgb8> = Vec::new();
    if fps == target_fps {
        resampled.extend(frames);
    } else {
        let scale = target_fps / fps;
        let slot = |i: usize| (i as f64 * scale + 0.5).floor() as usize;
        for (i, frame) in frames.iter().enumerate() {
            resampled.extend(std::iter::repeat_n(frame, slot(i + 1) - slot(i)));
        }
    }
    resampled.truncate(num_frames as usize);
    if resampled.is_empty() {
        return Err(format!("a reference video of {} frames at {fps} fps holds no frame at {target_fps} fps", frames.len()));
    }
    let (height, width) = resolve_canvas_size(first.w as f64, first.h as f64, canvas.multiple, canvas.canvas_short_edge, canvas.canvas_max_pixels)?;
    Ok(resampled.into_iter().map(|f| if (f.w, f.h) == (width, height) { f.clone() } else { resize_lanczos_pil(f, width as usize, height as usize) }).collect())
}

/// A soundtrack truncated to `max_duration` seconds at its own rate,
/// upmixed to stereo when mono, then resampled to `sample_rate` with
/// torchaudio's resampler.
pub fn normalize_soundtrack(s: &Soundtrack, max_duration: f64, sample_rate: u32) -> Result<Stereo, String> {
    if !(1..=2).contains(&s.channels.len()) {
        return Err(format!("a reference soundtrack must be mono or stereo, got {} channels", s.channels.len()));
    }
    if s.sample_rate == 0 {
        return Err("a reference soundtrack must have a positive sample rate".to_string());
    }
    let len = s.channels[0].len();
    if s.channels.iter().any(|c| c.len() != len) {
        return Err("a reference soundtrack's channels must have one length".to_string());
    }
    let keep = len.min((max_duration * s.sample_rate as f64) as usize);
    if keep == 0 {
        return Err("a reference soundtrack is empty".to_string());
    }
    let channel = |c: usize| audio::resample::sinc_hann(&s.channels[c.min(s.channels.len() - 1)][..keep], s.sample_rate, sample_rate);
    Ok([channel(0), channel(1)])
}

/// The audio VAE encoder a soundtrack is encoded through, and the per-channel
/// latent normalization (`audio_vae.config.latents_mean`/`latents_std`).
pub struct SoundtrackEncoder<'a> {
    pub cfg: &'a AudioEncoderConfig,
    pub tensors: &'a Tensors,
    pub latents_mean: &'a [f32],
    pub latents_std: &'a [f32],
    pub device: Option<&'a str>,
}

impl SoundtrackEncoder<'_> {
    /// Each channel through the mono encoder, as its posterior mean,
    /// normalized and packed channel-major: `[channels * latents,
    /// latent_channels]` rows, and the per-channel latent count.
    fn encode(&self, stereo: &Stereo) -> (Vec<f32>, u32) {
        let c = self.cfg.latent_channels as usize;
        assert_eq!(self.latents_mean.len(), c, "the audio latent mean must have one entry per latent channel");
        assert_eq!(self.latents_std.len(), c, "the audio latent std must have one entry per latent channel");
        let mut rows = Vec::new();
        let mut latents = 0;
        for wave in stereo {
            let z = crate::audio_encoder::encode(self.cfg, self.tensors, wave, self.device);
            let t = z.len() / c;
            latents = t as u32;
            for k in 0..t {
                rows.extend((0..c).map(|ch| (z[ch * t + k] - self.latents_mean[ch]) / self.latents_std[ch]));
            }
        }
        debug_assert_eq!(rows.len(), AUDIO_CHANNELS as usize * latents as usize * c);
        (rows, latents)
    }
}

/// Channel-major `[3, frames, h, w]` pixels in `[0, 255]`.
fn planar_pixels(frames: &[&Rgb8]) -> Vec<f32> {
    let plane = (frames[0].w * frames[0].h) as usize;
    let mut out = vec![0f32; 3 * frames.len() * plane];
    for (t, frame) in frames.iter().enumerate() {
        for (i, px) in frame.px.as_chunks::<3>().0.iter().enumerate() {
            for (c, &v) in px.iter().enumerate() {
                out[(c * frames.len() + t) * plane + i] = v as f32;
            }
        }
    }
    out
}

/// Encode normalized references, in order. `visual.seed` seeds the first
/// reference's conditioning noise and each later one draws from the next
/// seed. A video is snapped DOWN to the `frames_per_chunk * n +
/// latents_per_chunk` frames the video VAE encodes without padding; that
/// only bites when it is shorter than the generated clip.
pub fn encode_references(references: &[NormalizedReference], visual: &VisualConditionRequest, audio: &SoundtrackEncoder, frames_per_chunk: u32, latents_per_chunk: u32) -> Vec<EncodedReference> {
    references
        .iter()
        .enumerate()
        .map(|(i, reference)| {
            let req = VisualConditionRequest { seed: visual.seed.wrapping_add(i as u64), ..*visual };
            match reference {
                NormalizedReference::Image(img) => {
                    let v = encode_visual_condition(&req, &planar_pixels(&[img]), 1, img.h, img.w);
                    EncodedReference { geometry: ReferenceGeometry::Image { latent_height: v.latent_height, latent_width: v.latent_width }, video_rows: v.rows, audio_rows: Vec::new() }
                }
                NormalizedReference::Video { frames, soundtrack } => {
                    let n = frames.len() as u32;
                    let snapped = (n.saturating_sub(latents_per_chunk) / frames_per_chunk).max(1) * frames_per_chunk + latents_per_chunk;
                    let kept: Vec<&Rgb8> = frames.iter().take(snapped.min(n) as usize).collect();
                    let v = encode_visual_condition(&req, &planar_pixels(&kept), kept.len() as u32, kept[0].h, kept[0].w);
                    let (audio_rows, soundtrack_latents) = soundtrack.as_ref().map(|s| audio.encode(s)).unwrap_or_default();
                    EncodedReference { geometry: ReferenceGeometry::Video { latent_frames: v.latent_frames, latent_height: v.latent_height, latent_width: v.latent_width, soundtrack_latents }, video_rows: v.rows, audio_rows }
                }
                NormalizedReference::Audio(stereo) => {
                    let (audio_rows, latents) = audio.encode(stereo);
                    EncodedReference { geometry: ReferenceGeometry::Audio { latents }, video_rows: Vec::new(), audio_rows }
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scaled-down geometry `tools/goldens/minimaxh3_ref2va_references_dump_reference.py`
    /// normalizes at.
    const GOLDEN_CANVAS: ReferenceCanvas = ReferenceCanvas { multiple: 32, canvas_short_edge: 64, canvas_max_pixels: 64 * 112, image_short_edge: 96, fps: 24.0, audio_sample_rate: 32000 };
    const GOLDEN_NUM_FRAMES: u32 = 22;
    const GOLDEN_AUDIO_RATE_IN: u32 = 44100;

    fn rgb8(t: &checkpoint::safetensors::StTensor, frame: usize) -> Rgb8 {
        let (h, w) = (t.shape[t.shape.len() - 3], t.shape[t.shape.len() - 2]);
        let n = h * w * 3;
        Rgb8 { w: w as u32, h: h as u32, px: t.data[frame * n..(frame + 1) * n].iter().map(|&v| v as u8).collect() }
    }

    fn clip(t: &checkpoint::safetensors::StTensor) -> Vec<Rgb8> {
        (0..t.shape[0]).map(|f| rgb8(t, f)).collect()
    }

    /// Every normalizer against the installed diffusers' own: the image's
    /// LANCZOS onto its short edge, both fps directions with the canvas
    /// rescale, and the soundtrack's truncation, upmix and torchaudio
    /// resample. Pixels are byte-exact; audio to float rounding.
    #[test]
    fn references_normalize_exactly_as_the_reference_does() {
        let dir = brain_testutil::testdata_path("golden/minimaxh3/ref2va_references");
        let file = dir.join("minimaxh3_ref2va_references.safetensors");
        if !file.is_file() {
            brain_testutil::skip(&format!("{} not found - run tools/goldens/minimaxh3_ref2va_references_dump_reference.py --out {}", file.display(), dir.display()));
            return;
        }
        let Some(src) = brain_testutil::golden::Source::open(&dir, "tools/goldens/minimaxh3_ref2va_references_dump_reference.py") else {
            return;
        };
        let c = GOLDEN_CANVAS;
        if !src.require(&[
            ("multiple", c.multiple as i64),
            ("canvas_short_edge", c.canvas_short_edge as i64),
            ("canvas_max_pixels", c.canvas_max_pixels as i64),
            ("image_short_edge", c.image_short_edge as i64),
            ("num_frames", GOLDEN_NUM_FRAMES as i64),
            ("audio_rate_in", GOLDEN_AUDIO_RATE_IN as i64),
            ("audio_rate_out", c.audio_sample_rate as i64),
        ]) {
            return;
        }
        let fx: std::collections::HashMap<String, checkpoint::safetensors::StTensor> = checkpoint::safetensors::read(file.to_str().unwrap()).expect("read golden").into_iter().map(|t| (t.name.clone(), t)).collect();

        let image = normalize_image(&rgb8(&fx["image_in"], 0), &c).unwrap();
        assert_eq!(image, rgb8(&fx["image_out"], 0), "image reference");

        for name in ["video_a", "video_b"] {
            let fps = fx[&format!("{name}_fps")].data[0] as f64;
            let got = normalize_video(&clip(&fx[&format!("{name}_in")]), fps, GOLDEN_NUM_FRAMES, &c).unwrap();
            assert_eq!(got, clip(&fx[&format!("{name}_out")]), "{name} at {fps} fps");
        }

        let mono = Soundtrack { channels: vec![fx["audio_in"].data.clone()], sample_rate: GOLDEN_AUDIO_RATE_IN };
        let stereo = normalize_soundtrack(&mono, GOLDEN_NUM_FRAMES as f64 / c.fps as f64, c.audio_sample_rate).unwrap();
        let want = &fx["audio_out"].data;
        let half = want.len() / 2;
        for (ch, got) in stereo.iter().enumerate() {
            assert_eq!(got.len(), half, "soundtrack channel {ch} length");
            let worst = got.iter().zip(&want[ch * half..(ch + 1) * half]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(worst < 2e-6, "soundtrack channel {ch}: max abs error {worst:e}");
        }
    }

    /// Requests the reference refuses are refused here too, by name.
    #[test]
    fn invalid_references_are_refused() {
        let c = GOLDEN_CANVAS;
        let img = |w: u32, h: u32| Rgb8 { w, h, px: vec![0; (w * h * 3) as usize] };
        let sound = |channels: usize| Soundtrack { channels: vec![vec![0.1; 4410]; channels], sample_rate: 44100 };
        assert!(normalize_image(&img(90, 20), &c).unwrap_err().contains("1:4"));
        assert!(normalize_video(&[img(8, 8)], 0.0, 22, &c).unwrap_err().contains("frame rate"));
        assert!(normalize_video(&[img(8, 8), img(9, 8)], 24.0, 22, &c).unwrap_err().contains("one size"));
        assert!(normalize_soundtrack(&sound(3), 1.0, 32000).unwrap_err().contains("mono or stereo"));
        let lone_audio = [ReferenceMedia::Audio(sound(1))];
        assert!(normalize_references(&lone_audio, 22, &c).unwrap_err().contains("paired"));
        let bad_second = [ReferenceMedia::Image(img(64, 64)), ReferenceMedia::Image(img(90, 20))];
        assert!(normalize_references(&bad_second, 22, &c).unwrap_err().contains("reference 1"));
    }

    /// Decoded media of every kind normalize, encode and condition a tiny
    /// `ref2va` generation: each encoded reference's rows agree with the
    /// geometry it declares, and the generation runs on them.
    #[test]
    fn decoded_media_encode_into_a_ref2va_generation() {
        use crate::pipeline::tests::*;
        let dit_cfg = tiny_dit_cfg();
        let vae_cfg = tiny_video_vae_cfg();
        let dit_tensors = rand_dit_tensors(&dit_cfg, 61);
        let vae_tensors = rand_manifest_tensors(vae_cfg.tensor_manifest(), 62);
        let vocoder_tensors = rand_manifest_tensors(tiny_vocoder_cfg().tensor_manifest(), 63);
        let audio_cfg = AudioEncoderConfig { encoder_dim: 4, rates: vec![2, 4, 5], latent_dim: 16, latent_channels: dit_cfg.audio_in_channels, num_attention_heads: 2 };
        let audio_tensors = rand_manifest_tensors(audio_cfg.tensor_manifest(), 64);
        let ckpt = tiny_checkpoint(&dit_tensors, &vae_tensors, &vocoder_tensors);
        let opts = tiny_gen_opts(65);

        let mut rng = data::rng::Lcg::new(66);
        let mut img = |w: u32, h: u32| Rgb8 { w, h, px: rng.vec_scaled((w * h * 3) as usize, 1.0).iter().map(|v| ((v + 1.0) * 127.5) as u8).collect() };
        let sound = |seconds: f64, rate: u32| Soundtrack { channels: vec![(0..(seconds * rate as f64) as usize).map(|i| (i as f32 * 0.05).sin() * 0.5).collect()], sample_rate: rate };
        let media = vec![
            ReferenceMedia::Image(img(40, 24)),
            ReferenceMedia::Video { frames: (0..30).map(|_| img(48, 32)).collect(), fps: 30.0, soundtrack: Some(sound(0.4, 16000)) },
            ReferenceMedia::Audio(sound(0.2, 32000)),
        ];
        let canvas = ReferenceCanvas { canvas_short_edge: 32, canvas_max_pixels: 32 * 64, image_short_edge: 32, ..ReferenceCanvas::released(32) };
        let normalized = normalize_references(&media, opts.num_frames, &canvas).unwrap();
        let visual = VisualConditionRequest { video_vae_cfg: &vae_cfg, video_vae_tensors: &vae_tensors, device: Some("cpu"), video_latents_mean: &ckpt.video_latents_mean, video_latents_std: &ckpt.video_latents_std, patch_size: dit_cfg.patch_size, seed: 67 };
        let audio_mean = vec![0.0; audio_cfg.latent_channels as usize];
        let audio_std = vec![1.0; audio_cfg.latent_channels as usize];
        let audio = SoundtrackEncoder { cfg: &audio_cfg, tensors: &audio_tensors, latents_mean: &audio_mean, latents_std: &audio_std, device: Some("cpu") };
        let encoded = encode_references(&normalized, &visual, &audio, vae_cfg.clip_length, vae_cfg.tokens_chunk_size());

        let patch_values = dit_cfg.video_patch_dim() as usize;
        let audio_values = dit_cfg.audio_in_channels as usize;
        for (r, e) in encoded.iter().enumerate() {
            let (video_rows, audio_latents) = match e.geometry {
                ReferenceGeometry::Image { latent_height, latent_width } => ((latent_height / 2) * (latent_width / 2), 0),
                ReferenceGeometry::Video { latent_frames, latent_height, latent_width, soundtrack_latents } => (latent_frames * (latent_height / 2) * (latent_width / 2), soundtrack_latents),
                ReferenceGeometry::Audio { latents } => (0, latents),
            };
            assert_eq!(e.video_rows.len(), video_rows as usize * patch_values, "reference {r}: video rows");
            assert_eq!(e.audio_rows.len(), AUDIO_CHANNELS as usize * audio_latents as usize * audio_values, "reference {r}: soundtrack rows");
        }
        assert!(matches!(encoded[1].geometry, ReferenceGeometry::Video { soundtrack_latents, .. } if soundtrack_latents > 0), "the video's soundtrack is encoded");

        let text = tiny_text(3, dit_cfg.text_dim as usize, 68);
        let av = crate::pipeline::ref2va(&ckpt, &text, &encoded, &opts, &capability::CancelToken::default(), |_, _, _| {}).expect("ref2va");
        assert!(av.video.iter().all(|v| v.is_finite()));
    }
}
