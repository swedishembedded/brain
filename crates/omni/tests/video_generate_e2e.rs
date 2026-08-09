// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real end-to-end smoke test for the `video` capability blob wired in this
//! session: `capability::blob::decode_video_hwc` -> `OmniInner::
//! generate_multimodal`'s `video` parameter -> `crate::mm::
//! encode_video_frames` -> a real generation, on the real 48-layer/128-expert
//! Thinker.
//!
//! Not a parity test (there is no HF reference for "omni + concatenated
//! video-frame blob" -- that wire shape is this session's own addition, not
//! something upstream ships), and not a vision-tower accuracy test (that is
//! `vision_parity.rs`'s job, already exact per-frame since a video frame runs
//! through the identical `encode_image` path). What this proves: the NEW
//! plumbing (decode -> per-frame vision encode -> M-RoPE grid -> generate)
//! runs end to end against the real checkpoint without crashing and produces
//! a non-empty, real generation -- the risk surface a wire-shape change adds
//! that a unit test on synthetic `Invocation`s cannot cover.
//!
//! Real-weight-adjacent: skips cleanly when `BRAIN_OMNI_HF_DIR` is unset.
//! Expected to be slow (`crate::generate`'s own doc: every layer's weights
//! stream fresh per token) -- kept to 2 tiny 32x32 frames and 2 new tokens to
//! stay a smoke test, not a benchmark. Marked `#[ignore]`, matching every
//! other real-weight test in this crate.
//!
//! usage: `BRAIN_OMNI_HF_DIR=/tmp/.X11-unix/brain/hf/Qwen3-Omni-30B-A3B-Instruct \
//!         cargo test --release -p brain-omni --test video_generate_e2e -- --ignored --nocapture`

use capability::blob::decode_video_hwc;
use capability::{Invocation, Media, Provider};
use omni::caps::OmniProvider;

#[test]
#[ignore]
fn video_blob_generates_real_text_end_to_end() {
    let Some(hf_dir) = std::env::var("BRAIN_OMNI_HF_DIR").ok().filter(|p| !p.is_empty()) else {
        eprintln!("skip: BRAIN_OMNI_HF_DIR unset");
        return;
    };

    // Two tiny, distinctly-colored 32x32 RGB frames -- the smallest size that
    // clears one merged vision token (patch_size=16 x spatial_merge_size=2).
    let (w, h, frames) = (32u32, 32u32, 2usize);
    let per_frame = (w * h * 3) as usize;
    let mut bytes = Vec::with_capacity(frames * per_frame * 4);
    for f in 0..frames {
        let level = if f == 0 { 0.2f32 } else { 0.8f32 };
        for _ in 0..per_frame {
            bytes.extend_from_slice(&level.to_le_bytes());
        }
    }
    let video_blob = capability::Blob::new(Media::Bytes, bytes)
        .with_meta(serde_json::json!({"frames": frames, "w": w, "h": h, "c": 3}));

    // Decode round-trip check before spending real GPU time on generation --
    // if the wire shape itself is wrong, fail here, not deep inside a
    // multi-minute real forward pass.
    let probe_inv = Invocation::new().blob("video", video_blob.clone());
    let decoded = decode_video_hwc(&probe_inv, "video").expect("decode_video_hwc must accept its own wire shape");
    assert_eq!(decoded.len(), frames);
    for (hwc, fw, fh) in &decoded {
        assert_eq!((*fw, *fh), (w, h));
        assert_eq!(hwc.len(), per_frame);
    }

    let provider = OmniProvider::load(&hf_dir).expect("load real checkpoint");
    let action = provider.action("generate").expect("omni::caps must register a generate action");

    let inv = Invocation::new().set("prompt", serde_json::json!("Describe what you see.")).set("max_new", serde_json::json!(2)).blob("video", video_blob);

    println!("running real generate() with a 2-frame video blob -- this streams every layer's weights fresh per step, expect real wall time...");
    let outcome = action.run(&inv, &mut |_p| {}).expect("generate with a video blob must succeed, not error");

    let text = outcome.outputs.get("text").and_then(|v| v.as_str()).unwrap_or_default();
    println!("got text: {text:?}");
    assert!(!text.is_empty(), "video-conditioned generation produced empty text");
}
