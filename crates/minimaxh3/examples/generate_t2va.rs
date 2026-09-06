// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Run a real `t2va` (text-to-video+audio) generation against a real,
//! locally-checked-out MiniMax-H3 checkpoint, and write the result to an
//! MP4 - the first genuine end-to-end proof that every real-weight
//! component this port has validated in isolation (text encoder, DiT,
//! video VAE, audio VAE/vocoder) also composes into a real generation.
//!
//! Deliberately requests the SMALLEST geometry `crate::pipeline::generate`'s
//! own validation allows (a small square canvas, the ~5s minimum duration
//! `MIN_DURATION_S` enforces, few inference steps) - this is a
//! proof-of-concept run on CPU-only hardware with a ~33B-param DiT at fp32
//! (no reduced-precision tier exists for it yet), not a quality
//! demonstration. Raises steps/canvas/duration once this runs at all.
//!
//! Requires `BRAIN_MINIMAXH3_ALLOW_COMMUNITY=1` (the community-license
//! opt-in, see `crate::caps::check_license`'s own doc) and
//! `BRAIN_MINIMAXH3_DIR` pointing at a local checkout with `transformer/`,
//! `vae/`, `text_encoder/`, `audio_vae/`, `tokenizer/` all present.
//!
//! Usage: cargo run -p brain-minimaxh3 --release --example generate_t2va -- \
//!     "<prompt>" <out.mp4> [num_frames] [canvas_px] [steps]

use std::process::Command;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: generate_t2va \"<prompt>\" <out.mp4> [num_frames] [canvas_px] [steps]");
        std::process::exit(2);
    }
    let prompt = &a[1];
    let out_mp4 = &a[2];
    let num_frames: u32 = a.get(3).map(|s| s.parse().expect("num_frames")).unwrap_or(124);
    let canvas_px: u32 = a.get(4).map(|s| s.parse().expect("canvas_px")).unwrap_or(128);
    let steps: usize = a.get(5).map(|s| s.parse().expect("steps")).unwrap_or(4);

    minimaxh3::caps::check_license().unwrap_or_else(|e| panic!("{e}"));

    let paths = minimaxh3::caps::Paths::from_env().unwrap_or_else(|e| panic!("Paths::from_env: {e}"));

    eprintln!("[1/5] loading real text encoder + tokenizer ...");
    let t0 = std::time::Instant::now();
    let (qwen, tok) = minimaxh3::caps::build_text_encoder(&paths).unwrap_or_else(|e| panic!("build_text_encoder: {e}"));
    eprintln!("      done in {:.1}s", t0.elapsed().as_secs_f32());

    eprintln!("[2/5] encoding prompt: {prompt:?}");
    let text = minimaxh3::caps::encode_text_real(&qwen, &tok, prompt);
    drop(qwen);
    drop(tok);

    eprintln!("[3/5] loading DiT (streaming) + VAE weights - this is the ~33B-param checkpoint, expect several minutes ...");
    let t1 = std::time::Instant::now();
    let weights = minimaxh3::caps::LoadedWeights::load(&paths).unwrap_or_else(|e| panic!("LoadedWeights::load: {e}"));
    eprintln!("      done in {:.1}s", t1.elapsed().as_secs_f32());

    let ckpt = weights.as_checkpoint();
    // Real-hardware measurement (dit_matches_the_real_reference_numerically_
    // layer_by_layer with BRAIN_MINIMAXH3_TEST_DEVICE=vulkan, uncontended):
    // a full 50-block forward on this box's P40 completes in ~112s, faster
    // than the CPU backend's ~148-190s at the same real dimensions - so
    // "vulkan" is the better default now, overridable via
    // BRAIN_MINIMAXH3_GEN_DEVICE. This is independent of BRAIN_DEVICE (the
    // text encoder's own ambient device selection, kept on CPU separately -
    // its ~63GB checkpoint has no reduced-precision GPU tier and would OOM a
    // single P40).
    let dit_device = std::env::var("BRAIN_MINIMAXH3_GEN_DEVICE").unwrap_or_else(|_| "vulkan".to_string());
    let opts = minimaxh3::pipeline::GenOpts {
        canvas: Some((canvas_px, canvas_px)),
        num_frames,
        num_inference_steps: steps,
        seed: 0,
        device: Some(dit_device),
    };

    eprintln!(
        "[4/5] denoising ({} steps, {canvas_px}x{canvas_px}, {num_frames} frames requested) - this runs the real DiT forward pass repeatedly, expect this to be slow ...",
        opts.num_inference_steps
    );
    let t2 = std::time::Instant::now();
    let av = minimaxh3::pipeline::t2va(&ckpt, &text, &opts).unwrap_or_else(|e| panic!("t2va: {e}"));
    eprintln!("      done in {:.1}s -> {}x{} video, {} frames, {} audio channels @ {}Hz", t2.elapsed().as_secs_f32(), av.width, av.height, av.num_video_frames, av.audio.len(), av.sample_rate);

    eprintln!("[5/5] muxing to {out_mp4} ...");
    mux_mp4(&av, out_mp4);
    eprintln!("wrote {out_mp4}");
}

/// Raw RGB24 video frames + planar f32 audio -> MP4 (h264 + aac), via the
/// ffmpeg binary at `FFMPEG_BIN` (a static build - no distro ffmpeg package
/// was available in this environment, so a vendored/pip-installed one is
/// pointed at explicitly rather than assumed to be on `PATH`). brain itself
/// has no MP4 muxer; this is the standard external tool for it, the same
/// way `audio::wav::write` reaches for a container format's own well-known
/// spec rather than reinventing one.
fn mux_mp4(av: &minimaxh3::pipeline::GeneratedAv, out_mp4: &str) {
    let ffmpeg = std::env::var("FFMPEG_BIN").unwrap_or_else(|_| "ffmpeg".to_string());
    let tmp = std::env::temp_dir();
    let rgb_path = tmp.join(format!("h3_t2va_{}.rgb", std::process::id()));
    let pcm_path = tmp.join(format!("h3_t2va_{}.pcm", std::process::id()));

    // [3, T, H, W] channel-major, [0,1] -> per-frame interleaved RGB24 u8.
    let (t, h, w) = (av.num_video_frames as usize, av.height as usize, av.width as usize);
    let plane = h * w;
    let mut rgb = Vec::with_capacity(t * plane * 3);
    for f in 0..t {
        for px in 0..plane {
            for c in 0..3 {
                let v = av.video[c * t * plane + f * plane + px];
                rgb.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
    }
    std::fs::write(&rgb_path, &rgb).unwrap_or_else(|e| panic!("write {}: {e}", rgb_path.display()));

    // Planar f32 channels -> interleaved f32 PCM (ffmpeg's f32le expects
    // interleaved samples, not one contiguous block per channel).
    let n_channels = av.audio.len().max(1);
    let n_samples = av.audio.first().map(|c| c.len()).unwrap_or(0);
    let mut pcm = Vec::with_capacity(n_samples * n_channels * 4);
    for i in 0..n_samples {
        for ch in &av.audio {
            pcm.extend_from_slice(&ch[i].clamp(-1.0, 1.0).to_le_bytes());
        }
    }
    std::fs::write(&pcm_path, &pcm).unwrap_or_else(|e| panic!("write {}: {e}", pcm_path.display()));

    let status = Command::new(&ffmpeg)
        .args(["-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s"])
        .arg(format!("{w}x{h}"))
        .args(["-r"])
        .arg(format!("{}", minimaxh3::pipeline::FPS as u32))
        .args(["-i"])
        .arg(&rgb_path)
        .args(["-f", "f32le", "-ar"])
        .arg(format!("{}", av.sample_rate))
        .args(["-ac"])
        .arg(format!("{n_channels}"))
        .args(["-i"])
        .arg(&pcm_path)
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"])
        .arg(out_mp4)
        .status()
        .unwrap_or_else(|e| panic!("spawn {ffmpeg}: {e}"));

    let _ = std::fs::remove_file(&rgb_path);
    let _ = std::fs::remove_file(&pcm_path);
    assert!(status.success(), "ffmpeg exited with {status}");
}
