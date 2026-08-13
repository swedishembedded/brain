// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `omni::int8_thinker_resident::Int8ThinkerResident` on REAL, physically
//! separate GPUs — proves the cross-device mechanism the dual-GPU residency
//! work asked for actually works: real `estimate_multi`/
//! `claim_multi`/`activate_multi` through `residency::ResidencyManager`,
//! two real `Gpu` handles (`Gpu::new_on_index`, one per physical card), a
//! real `ThinkerInt8Store` shard streamed onto each, and a real forward pass
//! that hands the residual stream between them via a host round-trip.
//!
//! Skipped (not `#[ignore]`d — this is a real, always-attempted CI check
//! that degrades gracefully, matching this repo's own `discrete_gpu_count`
//! convention) when fewer than two discrete GPUs are visible.
//!
//! **What this proves and what it does not**: the comparison is SHARDED
//! (2-device) vs UNSHARDED (1-device) — both run the IDENTICAL int8
//! computation over the SAME REAL streamed weights (attention/norm/router via
//! `int8_thinker_resident::load_layer_bufs`, experts via `ThinkerInt8Store`),
//! read from the SAME on-disk synthetic checkpoint, so any divergence would
//! be a bug in the sharding/handoff plumbing, not a quantization question
//! (already covered separately by `thinker_int8_parity.rs`/`int8_resident`'s
//! own tests). The checkpoint quantizes the attention/router projections
//! exactly like a real `omni::import` output would (`should_quantize`: rank
//! 2, last dim a multiple of 4 — true for every 2-D tensor at this config's
//! shapes), so this also exercises `load_layer_bufs`'s dequantize-on-load
//! path, not just its plain-f32 path.

use data::rng::Lcg;
use omni::int8_resident::ThinkerInt8Store;
use omni::int8_thinker_resident::{load_layer_bufs, weights, Int8ThinkerResident};
use paramstore::upload::Uploader;
use omni::thinker::{layer_fwd, thinker_pipelines};
use residency::budget::Budgets;
use residency::manager::ClaimedMulti;
use residency::{Device, Instance, InstanceKey, MultiDeviceResidentModel, ResidencyManager};
use std::sync::Arc;

#[path = "common/int8_thinker_fixture.rs"]
mod fixture;
use fixture::{caps_for_split, tiny_cfg, write_synthetic_checkpoint};

#[test]
fn sharded_two_gpu_forward_matches_unsharded_single_gpu_forward() {
    if gpu_core::discrete_gpu_count() < 2 {
        eprintln!("skipping: fewer than two discrete GPUs visible");
        return;
    }

    let cfg = tiny_cfg(4); // 2 layers/device at an even 2-way split
    let dir = std::env::temp_dir().join(format!("int8_thinker_multi_gpu_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("thinker.safetensors");
    write_synthetic_checkpoint(path.to_str().unwrap(), &cfg, 424242);

    // ---- sharded: real claim_multi across two real GPUs ----
    let mut budgets = Budgets::new();
    // Generous budgets -- this test is about correctness, not fitting a
    // real card's exact capacity.
    budgets.set(Device::Gpu(0), 8 << 30, 0);
    budgets.set(Device::Gpu(1), 8 << 30, 0);
    let mut mgr = ResidencyManager::new(budgets);
    let caps = caps_for_split(path.to_str().unwrap(), &cfg, &[Device::Gpu(0), Device::Gpu(1)], 2);
    let resident = Arc::new(Int8ThinkerResident::new(path.to_str().unwrap().to_string(), omni::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps));
    mgr.register_multi(resident);

    let (claimed, devices, key) = mgr.claim_multi(omni::int8_thinker_resident::MODEL, "forward", &capability::Invocation::new(), &Default::default()).expect("claim_multi");
    assert_eq!(devices, vec![Device::Gpu(0), Device::Gpu(1)], "even 4-layer split over 2 devices");
    let instance = match claimed {
        ClaimedMulti::Hot(h) => h,
        ClaimedMulti::Build(m) => {
            let inst = m.activate_multi(&key, &devices).expect("activate_multi");
            mgr.adopt_multi(&key, Arc::new(std::sync::Mutex::new(inst)))
        }
    };

    let n = 3u32;
    let mut rng = Lcg::new(99);
    let x_host = rng.vec_scaled((n * cfg.hidden) as usize, 1.0);
    let x_bytes: Vec<u8> = x_host.iter().flat_map(|f| f.to_le_bytes()).collect();
    let inv = capability::Invocation::new().blob("x", capability::Blob::new(capability::Media::Bytes, x_bytes).with_meta(serde_json::json!({"n": n})));
    let out = instance.lock().unwrap().run("forward", &inv, &mut |_| {}).expect("forward");
    let hidden_blob = out.blobs.get("hidden").expect("hidden blob");
    let sharded: Vec<f32> = hidden_blob.bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();

    // ---- unsharded reference: SAME checkpoint, read via the SAME real
    // loaders, one device, no handoff ----
    let reader = checkpoint::weightio::WeightReader::open(path.to_str().unwrap()).expect("open synthetic checkpoint");
    let gpu = gpu_core::testgpu::dev(thinker_pipelines());
    let mut up = Uploader::new(&gpu);
    let store = ThinkerInt8Store::build(&mut up, &reader, 0..cfg.n_layers as usize, &cfg);
    let tokens: Vec<u32> = (0..n).collect();
    let positions = qwenvl::mrope::get_rope_index(&tokens, u32::MAX, &[]);
    let section: [u32; 3] = [cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2]];
    let (cos_tab, sin_tab) = qwenvl::mrope::mrope_tables(&positions, section, cfg.head_dim, cfg.rope_theta);
    let cos = gpu.storage_init("cos", &cos_tab);
    let sin = gpu.storage_init("sin", &sin_tab);
    let mut h = gpu.storage_init("h", &x_host);
    for l in 0..cfg.n_layers as usize {
        let lb = load_layer_bufs(&mut up, &reader, l);
        let w = weights(&lb);
        let experts8 = store.layer(l);
        let (out, ..) = layer_fwd(&gpu, &cfg, &w, &h, &cos, &sin, n, None, Some(experts8));
        h = out;
    }
    let reference = gpu.read(&h, (n * cfg.hidden) as usize);

    let mut worst = 0.0f32;
    for (a, b) in sharded.iter().zip(reference.iter()) {
        worst = worst.max((a - b).abs());
    }
    assert!(worst < 1e-4, "sharded (2-GPU) forward diverged from unsharded (1-GPU) reference: worst_abs={worst:e}\nsharded={sharded:?}\nreference={reference:?}");
    assert!(reference.iter().any(|&v| v.abs() > 1e-9), "reference is all-zero");

    std::fs::remove_dir_all(&dir).ok();
}

/// The real `generate()` action end to end: real tokenization-free greedy
/// sampling (token ids in, token ids out, `Instance::run("generate", ...)`),
/// compared SHARDED (2 real devices) vs UNSHARDED (1 real device) — both
/// built via the SAME `Int8ThinkerResident::activate_multi` (n_devices=1 is
/// a legitimate one-shard "multi" claim, not a special case), so this
/// exercises the full production path — real embed/lm_head loading,
/// `Int8ThinkerInstance::generate`'s recompute loop, the `Instance::run`
/// "generate" blob contract — on both sides identically. Any divergence in
/// the GENERATED TOKEN SEQUENCE (not just raw logits) would mean sharding
/// changes the model's actual output, not just introduces float noise.
#[test]
fn sharded_generate_matches_unsharded_generate() {
    if gpu_core::discrete_gpu_count() < 2 {
        eprintln!("skipping: fewer than two discrete GPUs visible");
        return;
    }

    let cfg = tiny_cfg(4);
    let dir = std::env::temp_dir().join(format!("int8_thinker_multi_gpu_generate_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("thinker.safetensors");
    write_synthetic_checkpoint(path.to_str().unwrap(), &cfg, 777777);

    // Two residents, differing ONLY in the capacity they were told about:
    // one card too small to hold the model (so it must shard across both),
    // and one card big enough (so it must not). Placement is capacity-driven,
    // so this is how a test asks for a specific split.
    let sharded_resident = Int8ThinkerResident::new(path.to_str().unwrap().to_string(), omni::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps_for_split(path.to_str().unwrap(), &cfg, &[Device::Gpu(0), Device::Gpu(1)], 2));
    let whole_resident = Int8ThinkerResident::new(path.to_str().unwrap().to_string(), omni::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps_for_split(path.to_str().unwrap(), &cfg, &[Device::Gpu(0)], 1));
    let key = InstanceKey::new(omni::int8_thinker_resident::MODEL, "default");

    let prompt_ids: Vec<u32> = vec![2, 9, 4];
    let max_new_tokens = 4u32;
    let eos_ids: Vec<u32> = Vec::new(); // none in this synthetic vocab -- forces a full max_new_tokens run for a deterministic length to compare

    let run_generate = |instance: &mut Box<dyn Instance>| -> Vec<u32> {
        let bytes: Vec<u8> = prompt_ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        let inv = capability::Invocation::new().blob(
            "ids",
            capability::Blob::new(capability::Media::Bytes, bytes).with_meta(serde_json::json!({"max_new_tokens": max_new_tokens, "eos_ids": eos_ids})),
        );
        let out = instance.run("generate", &inv, &mut |_| {}).expect("generate");
        let blob = out.blobs.get("ids").expect("ids blob");
        blob.bytes.chunks_exact(4).map(|q| u32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect()
    };

    let mut sharded = sharded_resident.activate_multi(&key, &[Device::Gpu(0), Device::Gpu(1)]).expect("activate_multi sharded");
    let sharded_out = run_generate(&mut sharded);
    drop(sharded); // free both devices before building the single-device reference on Gpu(0)

    let mut unsharded = whole_resident.activate_multi(&key, &[Device::Gpu(0)]).expect("activate_multi unsharded");
    let unsharded_out = run_generate(&mut unsharded);
    drop(unsharded);

    assert_eq!(sharded_out.len(), max_new_tokens as usize, "expected exactly max_new_tokens generated (eos_ids is deliberately empty)");
    assert_eq!(sharded_out, unsharded_out, "sharded vs unsharded generate() diverged: sharded={sharded_out:?} unsharded={unsharded_out:?}");

    std::fs::remove_dir_all(&dir).ok();
}

/// `sharded_generate_matches_unsharded_generate` proves sharding doesn't
/// change `generate()`'s answer, but both sides call the SAME
/// `Int8ThinkerInstance::generate` — a bug shared by both (wrong row
/// extraction, wrong head weight, an off-by-one in the embedding gather)
/// would pass that test undetected. This test instead checks `generate()`'s
/// FIRST token against an INDEPENDENTLY assembled reference: the already-
/// separately-tested `forward()` (`sharded_two_gpu_forward_matches_
/// unsharded_single_gpu_forward`) plus a from-scratch embed/final_norm/
/// lm_head/argmax built directly from the public loaders
/// (`load_mat_host`/`load_vec`/`load_mat`, `thinker::
/// final_norm`/`lm_head_fwd`) -- no code path shared with `generate()`'s
/// own implementation. Single-device only (this is about `generate()`'s own
/// correctness, not the cross-device question the other test already
/// covers), so it runs even on a one-GPU box.
#[test]
fn generate_first_token_matches_independently_assembled_reference() {
    if gpu_core::discrete_gpu_count() < 1 {
        eprintln!("skipping: no discrete GPU visible");
        return;
    }

    let cfg = tiny_cfg(2);
    let dir = std::env::temp_dir().join(format!("int8_thinker_multi_gpu_first_token_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("thinker.safetensors");
    write_synthetic_checkpoint(path.to_str().unwrap(), &cfg, 313131);

    let caps1 = caps_for_split(path.to_str().unwrap(), &cfg, &[Device::Gpu(0)], 1);
    let resident = Int8ThinkerResident::new(path.to_str().unwrap().to_string(), omni::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps1.clone());
    let key = InstanceKey::new(omni::int8_thinker_resident::MODEL, "default");
    let prompt_ids: Vec<u32> = vec![3, 7, 1];

    // ---- via generate() (1 token, empty eos_ids so it always runs) ----
    let mut instance = resident.activate_multi(&key, &[Device::Gpu(0)]).expect("activate_multi");
    let bytes: Vec<u8> = prompt_ids.iter().flat_map(|t| t.to_le_bytes()).collect();
    let inv = capability::Invocation::new().blob(
        "ids",
        capability::Blob::new(capability::Media::Bytes, bytes).with_meta(serde_json::json!({"max_new_tokens": 1u32, "eos_ids": Vec::<u32>::new()})),
    );
    let out = instance.run("generate", &inv, &mut |_| {}).expect("generate");
    let blob = out.blobs.get("ids").expect("ids blob");
    let via_generate: Vec<u32> = blob.bytes.chunks_exact(4).map(|q| u32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
    assert_eq!(via_generate.len(), 1, "expected exactly one generated token");
    drop(instance);

    // ---- independent reference: forward() (own coverage above) + a
    // from-scratch embed/final_norm/lm_head/argmax ----
    let reader = checkpoint::weightio::WeightReader::open(path.to_str().unwrap()).expect("open synthetic checkpoint");
    let d = cfg.hidden as usize;
    let embed_table = omni::int8_thinker_resident::load_mat_host(&reader, "thinker.embed_tokens.weight", cfg.vocab, cfg.hidden);
    let mut x_host = Vec::with_capacity(prompt_ids.len() * d);
    for &t in &prompt_ids {
        x_host.extend_from_slice(&embed_table[t as usize * d..(t as usize + 1) * d]);
    }

    let resident2 = Int8ThinkerResident::new(path.to_str().unwrap().to_string(), omni::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps1);
    let mut instance2 = resident2.activate_multi(&key, &[Device::Gpu(0)]).expect("activate_multi (reference)");
    let bytes2: Vec<u8> = x_host.iter().flat_map(|f| f.to_le_bytes()).collect();
    let inv2 = capability::Invocation::new().blob("x", capability::Blob::new(capability::Media::Bytes, bytes2).with_meta(serde_json::json!({"n": prompt_ids.len() as u32})));
    let fwd_out = instance2.run("forward", &inv2, &mut |_| {}).expect("forward");
    drop(instance2);
    let hidden_blob = fwd_out.blobs.get("hidden").expect("hidden blob");
    let hidden: Vec<f32> = hidden_blob.bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
    let last_row = &hidden[(prompt_ids.len() - 1) * d..prompt_ids.len() * d];

    // Matching what Int8ThinkerInstance::generate() actually dispatches for
    // the head (thinker::lm_head_fwd over a dequantized-to-f32 lm_head --
    // Int8ThinkerInstance keeps attention/router/head in f32, only the routed
    // MoE experts stay int8, see int8_thinker_resident's module doc).
    let gpu = gpu_core::testgpu::dev(omni::thinker::thinker_pipelines());
    let mut up = Uploader::new(&gpu);
    let norm_w = omni::int8_thinker_resident::load_vec(&mut up, &reader, "thinker.norm.weight");
    let lm_head_w = omni::int8_thinker_resident::load_mat(&mut up, &reader, "thinker.lm_head.weight");
    let h1 = gpu.storage_init("h1", last_row);
    let normed = omni::thinker::final_norm(&gpu, &cfg, &norm_w, &h1, 1);
    let logits = omni::thinker::lm_head_fwd(&gpu, &lm_head_w, &normed, 1, cfg.hidden, cfg.vocab);
    let logits_host = gpu.read(&logits, cfg.vocab as usize);
    let reference_token = logits_host.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).expect("non-empty vocab");

    assert_eq!(via_generate[0], reference_token, "generate()'s first token diverged from an independently assembled reference");

    std::fs::remove_dir_all(&dir).ok();
}
