// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-Omni's audio tower (Thinker's AuT) vs. the real transformers
//! reference - reusing `qwen3asr::encoder::AudioEncoder` completely
//! unchanged, at the new `AudioEncoderConfig::qwen3_omni()` preset (M4: "the
//! shared-encoder hoist is a config bump, not a second copy" — see
//! `crates/omni/src/import.rs`'s module doc).
//!
//! Real-weight-adjacent: skips cleanly when the checkpoint shard containing
//! `thinker.audio_tower.*` (shard 1 of 15 for the released model) is not on
//! disk, per the engine's standard opt-in-env-var test pattern.
//!
//! usage: `BRAIN_QWEN3OMNIMOE_HF_DIR=/tmp/.X11-unix/brain/hf/Qwen3-Omni-30B-A3B-Instruct \
//!         cargo test --release -p brain-omni --test audio_parity -- --ignored --nocapture`

use std::collections::HashMap;
use std::path::PathBuf;

use checkpoint::mmap::MmapSafetensors;
use qwen3omnimoe::import::hf_to_brain;
use qwen3asr::config::AudioEncoderConfig;
use qwen3asr::encoder::{audio_pipelines, AudioEncoder};

/// Per-block qkv-fuse accumulator: `(q_w, q_b, k_w, k_b, v_w, v_b)`, each
/// filled in as its tensor arrives (real shards interleave block order).
type FusedQkv = (Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>);

fn shard_with_audio_tower() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("BRAIN_QWEN3OMNIMOE_HF_DIR").ok()?);
    let idx_path = dir.join("model.safetensors.index.json");
    let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&idx_path).ok()?).ok()?;
    let weight_map = idx["weight_map"].as_object()?;
    let shard = weight_map.get("thinker.audio_tower.conv2d1.weight")?.as_str()?;
    let p = dir.join(shard);
    p.exists().then_some(p)
}

/// The exact golden-generation formula from
/// `tools/goldens/qwen3omnimoe_dump_reference.py`'s `dump_audio`: a fixed,
/// deterministic, bounded, non-trivial mel pattern (never random, per the
/// engine's test-PRNG convention) so this reproduces the SAME input the
/// golden was dumped against.
fn golden_mel(num_mel: u32, n_frames: u32) -> Vec<f32> {
    (0..num_mel * n_frames).map(|i| ((i % 23) as f32 - 11.0) / 11.0).collect()
}

#[test]
#[ignore]
fn matches_the_real_audio_tower() {
    let Some(shard) = shard_with_audio_tower() else {
        eprintln!("skip: BRAIN_QWEN3OMNIMOE_HF_DIR unset, or its index doesn't (yet) have the shard holding thinker.audio_tower");
        return;
    };
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden/omni/omni_audio.safetensors");
    if !golden_path.exists() {
        eprintln!("skip: {golden_path:?} missing (run `make fetch/testdata`)");
        return;
    }

    let mmap = MmapSafetensors::open(&shard).expect("open shard");
    let cfg = AudioEncoderConfig::qwen3_omni();

    // Stream every thinker.audio_tower.* tensor this shard has, remap via
    // the same hf_to_brain + fuse_audio_qkv path import_as uses, strip the
    // "audio." prefix AudioEncoder::new's weight map doesn't carry.
    let mut fused: HashMap<u32, FusedQkv> = HashMap::new();
    let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
    for name in mmap.names() {
        if !name.starts_with("thinker.audio_tower.") {
            continue;
        }
        if qwen3omnimoe::import::is_qkv_fuse_leaf(name) {
            let b: u32 = name.strip_prefix("thinker.audio_tower.layers.").unwrap().split_once('.').unwrap().0.parse().unwrap();
            let data = mmap.tensor_f32(name).unwrap();
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
        let key = brain_name.strip_prefix("audio.").unwrap().to_string();
        weights.insert(key, mmap.tensor_f32(name).unwrap());
    }
    for (b, (qw, kw, vw, qb, kb, vb)) in fused {
        let (qw, kw, vw, qb, kb, vb) = (qw.unwrap(), kw.unwrap(), vw.unwrap(), qb.unwrap(), kb.unwrap(), vb.unwrap());
        let mut w = qw;
        w.extend(kw);
        w.extend(vw);
        let mut bias = qb;
        bias.extend(kb);
        bias.extend(vb);
        weights.insert(format!("blocks.{b}.qkv.weight"), w);
        weights.insert(format!("blocks.{b}.qkv.bias"), bias);
    }
    // 12 leaves/block x n_layers, plus 13 stem/head tensors: conv2d{1,2,3}.{weight,bias}
    // (6), conv_out.weight (1, bias-free), ln_post.{weight,bias} (2),
    // multi_modal_projector.linear_{1,2}.{weight,bias} (4).
    let expected = 12 * cfg.n_layers + 13;
    assert_eq!(weights.len() as u32, expected, "expected {expected} tensors, got {}", weights.len());

    let gpu = gpu_core::testgpu::dev(audio_pipelines());
    let enc = AudioEncoder::new(&gpu, cfg, &weights);

    let golden = MmapSafetensors::open(&golden_path).expect("open golden");
    let mel = golden.tensor_f32("mel").expect("golden mel");
    let feature_lens = golden.tensor_f32("feature_lens").expect("golden feature_lens");
    let want = golden.tensor_f32("hidden").expect("golden hidden");
    let n_frames = mel.len() as u32 / cfg.num_mel_bins;
    let regenerated = golden_mel(cfg.num_mel_bins, n_frames);
    assert_eq!(mel, regenerated, "golden mel doesn't match the dumper's own documented formula -- golden regenerated?");

    let (_encoder_out, audio_embeds) = enc.encode(&mel, feature_lens[0] as u32);

    assert_eq!(audio_embeds.len(), want.len(), "shape mismatch: got {} elems, golden has {}", audio_embeds.len(), want.len());
    let dot: f64 = audio_embeds.iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
    let na: f64 = audio_embeds.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let cosine = dot / (na * nb).max(1e-12);
    let max_abs = audio_embeds.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("audio tower parity: cosine={cosine:.6} max_abs={max_abs:.6}");
    assert!(cosine > 0.999, "cosine {cosine} below the parity floor");
}
