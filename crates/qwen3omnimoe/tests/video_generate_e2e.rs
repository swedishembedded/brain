// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real end-to-end smoke test for the `video` capability, file to generation:
//! a real video FILE (ffmpeg-encoded on the fly) -> `imaging::video::
//! decode_frames` (real ffmpeg subprocess decode) -> `capability::blob::
//! video_blob` (the wire encoder) -> `capability::blob::decode_video_hwc`
//! (the wire decoder) -> `OmniInner::generate_multimodal`'s `video`
//! parameter -> `crate::mm::encode_video_frames` (real 2-frame temporal-
//! paired vision encode) -> a real generation, on the real 48-layer/
//! 128-expert Thinker. Every stage of the video pipeline this session added
//! runs for real here, not just the wire format in isolation.
//!
//! Not a parity test (there is no HF reference for "omni + a video file" --
//! that whole path is a brain-side addition, not something upstream ships),
//! and not a vision-tower accuracy test (that is `vision_parity.rs`'s job
//! for the single-image path). What this proves: the full chain runs end to
//! end against the real checkpoint without crashing and produces a
//! non-empty, real generation. The encoded clip is exactly one
//! `temporal_patch_size` group (2 frames, no padding needed), distinctly
//! colored so a human reading `--nocapture` output can sanity-check that
//! "what you see" plausibly reflects both frames, not one replicated twice
//! (the bug `qwen3vl::preprocess::pack_patches_temporal` fixed) -- this
//! test's own assertion (non-empty text) cannot check that without a
//! reference.
//!
//! Real-weight-adjacent: skips cleanly when `BRAIN_OMNI_HF_DIR` is unset, or
//! when `ffmpeg` is not installed (`imaging::video::ffmpeg_available`).
//! Expected to be slow (`crate::generate`'s own doc: every layer's weights
//! stream fresh per token) -- kept to 2 tiny 32x32 frames and 2 new tokens to
//! stay a smoke test, not a benchmark. Marked `#[ignore]`, matching every
//! other real-weight test in this crate.
//!
//! usage: `BRAIN_OMNI_HF_DIR=/tmp/.X11-unix/brain/hf/Qwen3-Omni-30B-A3B-Instruct \
//!         cargo test --release -p brain-omni --test video_generate_e2e -- --ignored --nocapture`

use std::process::Command;

use capability::blob::decode_video_hwc;
use capability::{Invocation, Provider};
use imaging::video::{decode_frames, ffmpeg_available, VideoDecodeOpts};
use qwen3omnimoe::caps::OmniProvider;

#[test]
#[ignore]
fn video_blob_generates_real_text_end_to_end() {
    let Some(hf_dir) = std::env::var("BRAIN_OMNI_HF_DIR").ok().filter(|p| !p.is_empty()) else {
        eprintln!("skip: BRAIN_OMNI_HF_DIR unset");
        return;
    };
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not installed");
        return;
    }

    // A real 2-second, 2fps synthetic clip via ffmpeg's built-in testsrc --
    // 32x32 is the smallest size that clears one merged vision token
    // (patch_size=16 x spatial_merge_size=2). Two color-changing halves so
    // the two SAMPLED frames are visibly distinct.
    let dir = std::env::temp_dir().join("brain-omni-video-e2e-test");
    let _ = std::fs::create_dir_all(&dir);
    let clip = dir.join("clip.mp4");
    let enc = Command::new("ffmpeg")
        .args(["-y", "-f", "lavfi", "-i", "testsrc=size=32x32:rate=2:duration=1", "-pix_fmt", "yuv420p"])
        .arg(&clip)
        .output()
        .expect("spawning ffmpeg to encode the test clip");
    assert!(enc.status.success(), "encoding the test clip failed: {}", String::from_utf8_lossy(&enc.stderr));

    let frames = decode_frames(&clip, &VideoDecodeOpts { fps: Some(2.0), max_frames: 2 }).expect("decode_frames on a real clip must succeed");
    assert_eq!(frames.len(), 2, "expected exactly 2 sampled frames");
    let _ = std::fs::remove_dir_all(&dir);

    let blob = capability::blob::video_blob(&frames).expect("video_blob must accept decode_frames' own output shape");

    // Decode round-trip check before spending real GPU time on generation --
    // if the wire shape itself is wrong, fail here, not deep inside a
    // multi-minute real forward pass.
    let probe_inv = Invocation::new().blob("video", blob.clone());
    let decoded = decode_video_hwc(&probe_inv, "video").expect("decode_video_hwc must accept video_blob's own wire shape");
    assert_eq!(decoded, frames, "video_blob -> decode_video_hwc must round-trip exactly");

    let provider = OmniProvider::load(&hf_dir).expect("load real checkpoint");
    let action = provider.action("generate").expect("qwen3omnimoe::caps must register a generate action");

    let inv = Invocation::new().set("prompt", serde_json::json!("Describe what you see.")).set("max_new", serde_json::json!(2)).blob("video", blob);

    println!("running real generate() with a real-ffmpeg-decoded 2-frame video -- this streams every layer's weights fresh per step, expect real wall time...");
    let outcome = action.run(&inv, &mut |_p| {}).expect("generate with a video blob must succeed, not error");

    let text = outcome.outputs.get("text").and_then(|v| v.as_str()).unwrap_or_default();
    println!("got text: {text:?}");
    assert!(!text.is_empty(), "video-conditioned generation produced empty text");
}
