// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `florence2::caps`'s `ground` action, run end to end through the real
//! checkpoint on a synthetic image: not a parity check (no reference
//! `generate()` run exists to gate against - HF's own sampling/beam-search
//! machinery is out of scope to replicate byte-exact, per the roadmap), but
//! the one test that actually exercises the FULL serving path a live
//! `brain do florence2 ground` call takes: image decode -> resize/normalize
//! -> vision tower -> tokenizer encode -> encoder -> the multi-step
//! `generate()` loop (never exercised end to end anywhere else - the parity
//! tests only ever call `decode` for a single fixed-length teacher-forced
//! prefix) -> tokenizer decode -> `grounding::parse_boxes`. A real photo
//! isn't needed for this: the point is that the pipeline runs to completion
//! and produces a well-formed `Outcome`, not a specific answer.

use capability::{Blob, Invocation, Media, Registry};
use serde_json::json;

use florence2::caps::Florence2Provider;

/// A small deterministic synthetic RGB image (smooth gradient, no real photo
/// needed - `ground`'s job here is just to prove the pipeline runs).
fn synthetic_image(w: u32, h: u32) -> Blob {
    let mut hwc = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            hwc.push(x as f32 / w as f32);
            hwc.push(y as f32 / h as f32);
            hwc.push(0.5);
        }
    }
    let bytes: Vec<u8> = hwc.iter().flat_map(|f| f.to_le_bytes()).collect();
    Blob::new(Media::Image, bytes).with_meta(json!({"w": w, "h": h, "c": 3}))
}

#[test]
fn ground_runs_end_to_end_against_the_real_checkpoint() {
    let Some(dir) = std::env::var("FLORENCE2_DIR").ok() else {
        brain_testutil::skip("FLORENCE2_DIR unset");
        return;
    };

    let mut reg = Registry::new();
    reg.register(std::sync::Arc::new(Florence2Provider::new(dir)));

    let inv = Invocation::new().blob("image", synthetic_image(320, 240)).set("target", json!("the login button")).set("max_new_tokens", json!(16));
    let outcome = reg.run(florence2::caps::MODEL, "ground", inv, &mut |_| {}).expect("ground should run to completion");

    assert!(outcome.outputs.get("found").is_some(), "outcome must report whether anything was found");
    let boxes = outcome.outputs.get("boxes").and_then(|v| v.as_array()).expect("boxes must be an array");
    for b in boxes {
        let bbox = b.get("bbox").and_then(|v| v.as_array()).expect("each box has a bbox");
        assert_eq!(bbox.len(), 4, "bbox is [x0,y0,x1,y1]");
        assert!(b.get("phrase").is_some(), "each box carries its phrase");
    }
}
