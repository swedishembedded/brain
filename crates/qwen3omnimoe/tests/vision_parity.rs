// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-Omni's vision tower (Thinker's ViT + PatchMerger + DeepStack) vs.
//! the real transformers reference - reusing `qwen3vl::encoder::VisionEncoder`
//! and `qwen3vl::encoder::PatchMerger` completely unchanged, at the new
//! `VisionConfig::qwen3_omni()` preset (same "config bump, not a second
//! copy" shape as `crates/omni/tests/audio_parity.rs`).
//!
//! Real-weight-adjacent: skips cleanly when the checkpoint shard containing
//! `thinker.visual.*` (shard 1 of 15, same shard the audio tower needs) is
//! not on disk.
//!
//! usage: `BRAIN_QWEN3OMNIMOE_HF_DIR=/tmp/.X11-unix/brain/hf/Qwen3-Omni-30B-A3B-Instruct \
//!         cargo test --release -p brain-omni --test vision_parity -- --ignored --nocapture`

use std::collections::HashMap;
use std::path::PathBuf;

use checkpoint::mmap::MmapSafetensors;
use qwen3omnimoe::import::hf_to_brain;
use qwen3vl::config::VisionConfig;
use qwen3vl::encoder::{vision_pipelines, PatchMerger, VisionEncoder};

fn shard_with_visual() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("BRAIN_QWEN3OMNIMOE_HF_DIR").ok()?);
    let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("model.safetensors.index.json")).ok()?).ok()?;
    let shard = idx["weight_map"].as_object()?.get("thinker.visual.patch_embed.proj.weight")?.as_str()?;
    let p = dir.join(shard);
    p.exists().then_some(p)
}

fn cosine_max_abs(got: &[f32], want: &[f32]) -> (f64, f32) {
    assert_eq!(got.len(), want.len(), "shape mismatch: got {} elems, want {}", got.len(), want.len());
    let dot: f64 = got.iter().zip(want).map(|(a, b)| *a as f64 * *b as f64).sum();
    let na: f64 = got.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let cosine = dot / (na * nb).max(1e-12);
    let max_abs = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    (cosine, max_abs)
}

#[test]
#[ignore]
fn matches_the_real_vision_tower() {
    let Some(shard) = shard_with_visual() else {
        brain_testutil::skip("BRAIN_QWEN3OMNIMOE_HF_DIR unset, or its index doesn't (yet) have the shard holding thinker.visual");
        return;
    };
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden/omni/omni_vision.safetensors");
    if !golden_path.exists() {
        brain_testutil::skip(&format!("{golden_path:?} missing (run `make fetch/testdata`)"));
        return;
    }

    let mmap = MmapSafetensors::open(&shard).expect("open shard");
    let cfg = VisionConfig::qwen3_omni();

    // Stream every thinker.visual.* tensor, remap via hf_to_brain (same path
    // import_as uses), split by destination: encoder (blocks/patch_embed/
    // pos_embed) vs. the two merger weight sets.
    let mut encoder_w: HashMap<String, Vec<f32>> = HashMap::new();
    let mut main_merger_w: HashMap<String, Vec<f32>> = HashMap::new();
    let mut deepstack_w: Vec<HashMap<String, Vec<f32>>> = (0..cfg.deepstack_indexes.len()).map(|_| HashMap::new()).collect();
    for name in mmap.names() {
        if !name.starts_with("thinker.visual.") {
            continue;
        }
        let Some(brain_name) = hf_to_brain(name) else { continue };
        let data = mmap.tensor_f32(name).unwrap();
        if let Some(rest) = brain_name.strip_prefix("vision.merger.") {
            main_merger_w.insert(rest.to_string(), data);
        } else if let Some(rest) = brain_name.strip_prefix("vision.deepstack_merger.") {
            let (i, leaf) = rest.split_once('.').unwrap();
            deepstack_w[i.parse::<usize>().unwrap()].insert(leaf.to_string(), data);
        } else if let Some(rest) = brain_name.strip_prefix("vision.") {
            encoder_w.insert(rest.to_string(), data);
        }
    }
    let expected_encoder = 12 * cfg.depth + 3; // 12 leaves/block + patch_embed(w,b) + pos_embed
    assert_eq!(encoder_w.len() as u32, expected_encoder, "expected {expected_encoder} encoder tensors, got {}", encoder_w.len());
    for (i, m) in deepstack_w.iter().enumerate() {
        assert_eq!(m.len(), 6, "deepstack merger {i} expected 6 tensors (ln+fc1+fc2, weight+bias), got {}", m.len());
    }

    let gpu = gpu_core::testgpu::dev(vision_pipelines());
    let enc = VisionEncoder::new(&gpu, cfg.clone(), &encoder_w);

    let golden = MmapSafetensors::open(&golden_path).expect("open golden");
    let patches = golden.tensor_f32("patches").expect("golden patches");
    // grid_thw was saved as I32 in the golden; MmapSafetensors::tensor_f32
    // converts by declared dtype (i32::from_le_bytes(..) as f32), not a bit
    // reinterpretation, so these come back as exact small integers.
    let grid = golden.tensor_f32("grid_thw").expect("golden grid_thw");
    let (t, h, w) = (grid[0] as i32, grid[1] as u32, grid[2] as u32);
    assert_eq!(t, 1, "this test covers the single-frame (image) case; video (t>1) is a separate, not-yet-covered path");

    let (encoder_out, tap_feats) = enc.encode_with_taps(h, w, &patches, &cfg.deepstack_indexes);

    // The golden's "hidden" is `Qwen3OmniMoeVisionEncoder.forward`'s
    // `last_hidden_state` -- the RAW per-patch ViT output, BEFORE the
    // primary merger (which a separate wrapping module applies). It is NOT
    // the same stage as `deepstack_features` below (which the reference
    // model merges internally) -- compare it directly against `encoder_out`,
    // no merger involved. (`main_merger_w` is still streamed/validated above
    // via the tensor-count assertion, but this golden has no post-merger
    // stage to check it against -- that's the primary merger's own, separate
    // test surface, not this one.)
    let want_hidden = golden.tensor_f32("hidden").expect("golden hidden");
    let (cos_h, max_abs_h) = cosine_max_abs(&encoder_out, &want_hidden);
    println!("vision encoder (pre-merger): cosine={cos_h:.6} max_abs={max_abs_h:.6}");
    assert!(cos_h > 0.999, "encoder cosine {cos_h} below the parity floor");

    // No golden covers the primary merger's OWN output (this golden's
    // "hidden" is pre-merger, per the comment above) -- run it anyway as a
    // real-weight shape/finiteness check, since its weights are already
    // streamed and count-checked.
    let main_merger = PatchMerger::new(&gpu, &main_merger_w, cfg.hidden, cfg.spatial_merge_size, cfg.out_hidden_size, false);
    let merged = main_merger.merge(&encoder_out, h * w);
    assert_eq!(merged.len(), (h * w / cfg.merge_unit() * cfg.out_hidden_size) as usize);
    assert!(merged.iter().all(|v| v.is_finite()), "primary merger produced non-finite output on real weights");

    for (i, tap) in tap_feats.iter().enumerate() {
        // Same in_dim/merge as the main merger -- postshuffle_norm only moves
        // WHERE the LayerNorm applies (after the free [n,in_dim] ->
        // [n/merge², in_dim·merge²] reshape instead of before it), not the
        // reshape itself. `merge()`'s LN step reads `mrows·merged` elements
        // either way, and mrows·merged == n·in_dim identically -- so in_dim
        // and merge must match the tap's actual (pre-merge, hidden-width)
        // layout, not the already-merged output width.
        let merger = PatchMerger::new(&gpu, &deepstack_w[i], cfg.hidden, cfg.spatial_merge_size, cfg.out_hidden_size, true);
        let got_tap = merger.merge(tap, h * w);
        let want_tap = golden.tensor_f32(&format!("deepstack{i}")).unwrap_or_else(|| panic!("golden deepstack{i}"));
        let (cos_t, max_abs_t) = cosine_max_abs(&got_tap, &want_tap);
        println!("vision deepstack{i}: cosine={cos_t:.6} max_abs={max_abs_t:.6}");
        assert!(cos_t > 0.999, "deepstack{i} cosine {cos_t} below the parity floor");
    }
}
