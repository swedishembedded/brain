// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Regression test for a real `ERROR_OUT_OF_DEVICE_MEMORY`: `run_shards`'s
//! per-layer loop (`crates/omni/src/int8_thinker_resident.rs`) used to call
//! `layer_fwd`/`layer_decode_step` for every layer with nothing forcing
//! `backend-vulkan`'s deferred reclaim in between, so every prior layer's
//! attention (scores/probs/ctx) and MoE (router/gate/expert scratch) buffers
//! stayed buried (dropped but not yet `vkFreeMemory`'d) for the REST of the
//! pass - VRAM grew with layer count instead of staying flat, reproduced as
//! a real crash on 2x Tesla P40 running the actual
//! `Qwen3-Omni-30B-A3B-Instruct-W8A16` checkpoint. Fixed by having
//! `run_shards` call `gpu.flush()` every few layers.
//!
//! Exercises the REAL production path (`Instance::run("forward", ...)` ->
//! `Int8ThinkerInstance::forward` -> `run_shards`), not a reimplementation of
//! its loop, so a regression that removes or breaks the periodic flush is
//! caught here regardless of how the surrounding code is refactored.
//!
//! Observable: `Gpu::reclaim_event_count()`, reported through
//! `Instance::metrics()` (the concrete `Int8ThinkerInstance` isn't
//! downcastable from the `dyn Instance` the residency manager hands out).
//! This counts only calls to `VkContext::reclaim_dead` that actually freed
//! something - deliberately NOT a raw queue-submit count, which is also
//! inflated by one-off staging submits (`upload`/`zero`/`download`, one per
//! freshly allocated scratch buffer) that scale with layer count on their
//! own and would swamp the signal (measured: with the periodic flush
//! disabled, a 4-layer forward already issues ~100 queue submits and a
//! 32-layer one ~770 - an ~8x split matching the ~8x layer-count ratio even
//! with NO periodic reclaim at all, purely from per-buffer zero-init
//! submits). `reclaim_dead` is called ONLY from `flush`'s two branches, never
//! from those one-off submits, so it isolates deferred-reclaim activity
//! specifically: a loop that only reclaims once, at its very end (e.g. via
//! `run_shards`'s own closing `gpu.read`), reports ~1 reclaim event
//! REGARDLESS of layer count; a loop that reclaims periodically reports
//! roughly `n_layers / FLUSH_EVERY`. So reclaim events SCALING WITH layer
//! count is exactly the signature a regression back to "only flush once"
//! would lose - this test compares a small and a large layer count and
//! asserts that scaling.
//!
//! Vulkan-specific (skips cleanly on any other backend, matching this
//! repo's own `BRAIN_DEVICE`-gated real-hardware test convention - see
//! `scripts/gates/parity-gate.sh`): `reclaim_event_count` is 0-by-contract on
//! every backend that reclaims eagerly (wgpu, cpu), so this only exercises
//! anything real on Vulkan.
//!
//! usage: `BRAIN_DEVICE=vulkan cargo test --release -p brain-omni --test int8_thinker_periodic_reclaim`

use qwen3omnimoe::config::MoeTextConfig;
use qwen3omnimoe::int8_thinker_resident::Int8ThinkerResident;
use residency::{Device, InstanceKey, MultiDeviceResidentModel};

#[path = "common/int8_thinker_fixture.rs"]
mod fixture;
use fixture::{caps_for_split, tiny_cfg, write_synthetic_checkpoint};

fn total_reclaim_events(instance: &dyn residency::Instance) -> u64 {
    let metrics = instance.metrics();
    metrics
        .iter()
        .find(|(k, _)| k == "total_reclaim_events")
        .unwrap_or_else(|| panic!("Int8ThinkerInstance::metrics() did not report total_reclaim_events: {metrics:?}"))
        .1
        .as_u64()
        .expect("total_reclaim_events must be a u64")
}

/// Claim a real single-device `Int8ThinkerInstance` for `cfg`, run one
/// `forward` pass over `n` tokens, and return the reclaim-event count
/// attributable to THAT forward pass alone (baselined against the count
/// right after `activate_multi`, so any reclaim activity from loading
/// weights doesn't leak into the number this test actually cares about).
fn reclaim_events_for_one_forward(cfg: &MoeTextConfig, n: u32, seed: u64) -> u64 {
    let dir = std::env::temp_dir().join(format!("int8_thinker_periodic_reclaim_{}_{}", std::process::id(), seed));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("thinker.safetensors");
    write_synthetic_checkpoint(path.to_str().unwrap(), cfg, seed);

    let caps = caps_for_split(path.to_str().unwrap(), cfg, &[Device::Gpu(0)], 1);
    let resident = Int8ThinkerResident::new(path.to_str().unwrap().to_string(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps);
    let key = InstanceKey::new(qwen3omnimoe::int8_thinker_resident::MODEL, "default");
    let mut instance = resident.activate_multi(&key, &[Device::Gpu(0)]).expect("activate_multi (single device)");
    let base = total_reclaim_events(instance.as_ref());

    let x_host = vec![0.1f32; (n * cfg.hidden) as usize];
    let x_bytes: Vec<u8> = x_host.iter().flat_map(|f| f.to_le_bytes()).collect();
    let inv = capability::Invocation::new().blob("x", capability::Blob::new(capability::Media::Bytes, x_bytes).with_meta(serde_json::json!({"n": n})));
    instance.run("forward", &inv, &mut |_| {}).expect("forward");

    let after = total_reclaim_events(instance.as_ref());
    std::fs::remove_dir_all(&dir).ok();
    after - base
}

#[test]
fn reclaim_events_scale_with_layer_count_not_flat() {
    if gpu_core::backend_name() != "vulkan" {
        eprintln!("skip: needs BRAIN_DEVICE=vulkan (reclaim_event_count is 0-by-contract on every other backend)");
        return;
    }

    let n = 6u32;
    let few = reclaim_events_for_one_forward(&tiny_cfg(4), n, 1001);
    let many = reclaim_events_for_one_forward(&tiny_cfg(32), n, 1002);

    // A `run_shards` that only reclaims once at the very end (the pre-fix
    // shape) reports ~1 reclaim event for 4 layers AND for 32 -- this is the
    // exact signature that bug had (buried scratch piling up for the whole
    // pass instead of being freed layer by layer).
    assert!(
        many > few * 2,
        "reclaim events did not scale with layer count (4 layers: {few}, 32 layers: {many}) \
         -- run_shards's periodic reclaim looks like it's back to firing once \
         at the end instead of every few layers"
    );
}
