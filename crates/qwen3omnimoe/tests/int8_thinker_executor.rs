// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Int8ThinkerResident` driven through the FULL `residency::Executor`
//! dispatcher/lane machinery — the real reachability gate for the dual-GPU
//! residency work's item 3 (now closed).
//! `int8_thinker_multi_gpu.rs`
//! already proves the sharded MECHANISM is correct by calling
//! `ResidencyManager::claim_multi`/`activate_multi` directly; this file proves
//! the model is reachable the way D-Bus/HTTP would actually reach it — through
//! `Executor::register_multi` + `run_blocking`/`submit`, on real per-device
//! budgets, exercising `residency::multi`'s busy-tracking, home-lane dispatch,
//! and `ResidencyReport` surface for real.
//!
//! Skipped (not `#[ignore]`d — matches this repo's own `discrete_gpu_count`
//! convention) when fewer than two discrete GPUs are visible, and also honours
//! `MOE_SKIP_GPU_TESTS` like every other GPU-gated test in this tree.
//!
//! **New interaction, previously untested**: every prior `residency::Executor`
//! test in this tree (`crates/residency/src/executor.rs`'s own unit tests) uses
//! fake, non-GPU `ResidentModel`s, so `Executor`'s background dispatcher/lane
//! THREADS (spawned by `Executor::start`) had never before been combined with
//! REAL Vulkan device handles inside a short-lived test process. Found here:
//! dropping the last `Executor` handle at the end of a test starts an async
//! teardown cascade (dispatcher's `rx.recv()` returns `Err` → its `lanes` map
//! drops → each lane's `rx.recv()` returns `Err` → the lane thread exits,
//! dropping whatever `Gpu`/`Instance` it was holding) that used to NOT be
//! complete by the time the test function returned - the test binary would
//! proceed straight to process exit and could race a lane thread still
//! mid-teardown on a live Vulkan device, observed as an exit-time SIGSEGV in
//! this sandbox specifically in `a_second_generate_reuses_the_hot_sharded_
//! instance` (the one test whose Executor had done the MOST real GPU work -
//! two full `generate()` round trips - before teardown, widening the race
//! window). This is a DIFFERENT symptom of the same general class of real
//! Vulkan device lifecycle edge case investigated previously in this sandbox
//! (concurrent device creation hangs), not a correctness bug in the
//! scheduling logic under test - every assertion above the `drop` always
//! passed even on a run that went on to crash.
//!
//! **Fixed**: `Executor::shutdown` (`crates/residency/src/executor.rs`) now
//! keeps every spawned thread's `JoinHandle` (dispatcher + one per lane,
//! previously discarded at `start`) and `join`s all of them - a real,
//! deterministic guarantee that every lane's Vulkan device teardown has
//! actually finished, not the fixed-sleep heuristic (`settle_teardown`) this
//! file used before. Call it (not `drop`) at the end of every test in this
//! file.

use qwen3omnimoe::int8_thinker_resident::{Int8ThinkerResident, MODEL};
use residency::budget::Budgets;
use residency::{Device, Executor, InstanceKey, MultiDeviceResidentModel, Policy, ResidentModel};
use std::sync::Arc;

#[path = "common/int8_thinker_fixture.rs"]
mod fixture;
use fixture::{caps_for_split, tiny_cfg, write_synthetic_checkpoint};

fn skip() -> bool {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
        return true;
    }
    if gpu_core::discrete_gpu_count() < 2 {
        brain_testutil::skip_unavailable("fewer than two discrete GPUs visible");
        return true;
    }
    false
}

fn generate_inv(prompt_ids: &[u32], max_new_tokens: u32, eos_ids: &[u32]) -> capability::Invocation {
    let bytes: Vec<u8> = prompt_ids.iter().flat_map(|t| t.to_le_bytes()).collect();
    capability::Invocation::new().blob(
        "ids",
        capability::Blob::new(capability::Media::Bytes, bytes).with_meta(serde_json::json!({"max_new_tokens": max_new_tokens, "eos_ids": eos_ids})),
    )
}

fn decode_ids(out: &capability::Outcome) -> Vec<u32> {
    let blob = out.blobs.get("ids").expect("ids blob");
    blob.bytes.chunks_exact(4).map(|q| u32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect()
}

/// The headline test: `generate` through the Executor must produce the EXACT
/// same token sequence as a direct `activate_multi` reference (the path
/// `int8_thinker_multi_gpu.rs` already validates). The Executor is pure
/// plumbing over the same `Int8ThinkerInstance::generate`, so exact equality
/// is the right bar — not a tolerance — any divergence is a scheduling bug,
/// not float noise.
#[test]
fn generate_through_the_executor_matches_a_direct_activate_multi_reference() {
    if skip() {
        return;
    }
    let cfg = tiny_cfg(4);
    let dir = std::env::temp_dir().join(format!("int8_thinker_executor_generate_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("thinker.safetensors");
    write_synthetic_checkpoint(path.to_str().unwrap(), &cfg, 555111);

    let prompt_ids: Vec<u32> = vec![1, 5, 3];
    let max_new_tokens = 4u32;
    let eos_ids: Vec<u32> = Vec::new(); // empty vocab of EOS ids -- forces a full, deterministic-length run

    // ---- reference: direct activate_multi, no Executor ----
    let key = InstanceKey::new(MODEL, "default");
    let reference_resident = Int8ThinkerResident::new(path.to_str().unwrap().to_string(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps_for_split(path.to_str().unwrap(), &cfg, &[Device::Gpu(0), Device::Gpu(1)], 2));
    let mut reference_instance = reference_resident.activate_multi(&key, &[Device::Gpu(0), Device::Gpu(1)]).expect("direct activate_multi");
    let reference_out = decode_ids(&reference_instance.run("generate", &generate_inv(&prompt_ids, max_new_tokens, &eos_ids), &mut |_| {}).expect("direct generate"));
    drop(reference_instance); // free both cards before the Executor claims them

    // ---- through the Executor: register_multi + run_blocking, real dispatcher/lanes ----
    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 8 << 30, 0);
    budgets.set(Device::Gpu(1), 8 << 30, 0);
    let exec = Executor::start(vec![], budgets, Policy::default());
    let resident = Int8ThinkerResident::new(path.to_str().unwrap().to_string(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps_for_split(path.to_str().unwrap(), &cfg, &[Device::Gpu(0), Device::Gpu(1)], 2));
    exec.register_multi(Arc::new(resident));

    let result = exec.run_blocking(MODEL, "generate", generate_inv(&prompt_ids, max_new_tokens, &eos_ids), |_| {});
    let out = result.expect("generate through the Executor must succeed");
    let executor_out = decode_ids(&out);

    assert_eq!(executor_out.len(), max_new_tokens as usize, "expected exactly max_new_tokens generated (eos_ids is deliberately empty)");
    assert_eq!(executor_out, reference_out, "Executor-dispatched generate() diverged from a direct activate_multi reference: executor={executor_out:?} reference={reference_out:?}");

    exec.shutdown();
    std::fs::remove_dir_all(&dir).ok();
}

/// `Executor::residency()` must report real, non-zero, per-device bytes for
/// the resident multi-device Thinker — the honest-accounting claim, asserted
/// through the SERVED surface (not `ResidencyManager` directly, which
/// `manager.rs`'s own unit tests already cover). A multi-device instance must
/// appear in `multi_placements` and NOT in `placements` (that field is
/// singular-device by construction).
#[test]
fn executor_reports_real_per_device_bytes_for_the_multi_resident() {
    if skip() {
        return;
    }
    let cfg = tiny_cfg(4);
    let dir = std::env::temp_dir().join(format!("int8_thinker_executor_report_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("thinker.safetensors");
    write_synthetic_checkpoint(path.to_str().unwrap(), &cfg, 555222);

    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 8 << 30, 0);
    budgets.set(Device::Gpu(1), 8 << 30, 0);
    let exec = Executor::start(vec![], budgets, Policy::default());
    let resident = Int8ThinkerResident::new(path.to_str().unwrap().to_string(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps_for_split(path.to_str().unwrap(), &cfg, &[Device::Gpu(0), Device::Gpu(1)], 2));
    exec.register_multi(Arc::new(resident));

    let r = exec.run_blocking(MODEL, "generate", generate_inv(&[2, 4], 1, &[]), |_| {});
    assert!(r.is_ok(), "{r:?}");

    let report = exec.residency();
    assert_eq!(report.multi_placements.len(), 1, "exactly one multi-device instance resident");
    assert!(report.placements.iter().all(|p| p.key.model != MODEL), "the multi-device Thinker must not appear in the single-device placements list");
    let devs = &report.multi_placements[0].devices;
    assert_eq!(devs.len(), 2, "sharded across both GPUs");
    for &(d, bytes) in devs {
        assert!(matches!(d, Device::Gpu(_)), "unexpected device {d:?}");
        assert!(bytes > 0, "device {d:?} reports zero bytes -- the accounting must be real, not a placeholder");
    }
    for d in [Device::Gpu(0), Device::Gpu(1)] {
        let b = report.budgets.iter().find(|b| b.device == d).unwrap_or_else(|| panic!("no budget entry for {d:?}"));
        assert!(b.used > 0, "{d:?}.used should reflect the resident shard, got 0");
    }

    exec.shutdown();
    std::fs::remove_dir_all(&dir).ok();
}

/// A trivial, always-succeeding single-device model — the blast-radius check
/// needs a second model on the SAME Executor, independent of anything the
/// multi-device Thinker touches.
struct TrivialModel;
struct TrivialInstance;
impl ResidentModel for TrivialModel {
    fn manifest(&self) -> capability::Manifest {
        capability::Manifest::new("trivial", "a stateless always-succeeding model", vec![capability::ActionSpec::new("run", "run")])
    }
    fn instance_key(&self, _action: &str, _inv: &capability::Invocation) -> InstanceKey {
        InstanceKey::new("trivial", "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> residency::MemCost {
        residency::MemCost::default()
    }
    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn residency::Instance>, String> {
        Ok(Box::new(TrivialInstance))
    }
}
impl residency::Instance for TrivialInstance {
    fn run(&mut self, _action: &str, _inv: &capability::Invocation, _progress: &mut dyn FnMut(capability::Progress)) -> capability::ActionResult {
        Ok(capability::Outcome::new().blob("out", capability::Blob::new(capability::Media::Bytes, vec![1])))
    }
}

/// Blast-radius check on REAL hardware: once the sharded Thinker is resident
/// on both cards, an unrelated single-device model registered on the SAME
/// Executor must still schedule and run — `residency::multi`'s addition to
/// the shared scheduling core must not have broken ordinary single-device
/// dispatch, the risk this whole change carries the most.
#[test]
fn a_single_device_model_still_schedules_beside_a_resident_multi_device_thinker() {
    if skip() {
        return;
    }
    let cfg = tiny_cfg(4);
    let dir = std::env::temp_dir().join(format!("int8_thinker_executor_blast_radius_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("thinker.safetensors");
    write_synthetic_checkpoint(path.to_str().unwrap(), &cfg, 555333);

    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 8 << 30, 0);
    budgets.set(Device::Gpu(1), 8 << 30, 0);
    let exec = Executor::start(vec![Arc::new(TrivialModel)], budgets, Policy::default());
    let resident = Int8ThinkerResident::new(path.to_str().unwrap().to_string(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps_for_split(path.to_str().unwrap(), &cfg, &[Device::Gpu(0), Device::Gpu(1)], 2));
    exec.register_multi(Arc::new(resident));

    // Make the Thinker resident first (occupies both cards).
    let r = exec.run_blocking(MODEL, "generate", generate_inv(&[1, 2], 1, &[]), |_| {});
    assert!(r.is_ok(), "{r:?}");

    // The trivial model runs on the SAME (idle) devices' spare capacity --
    // it's zero-cost, so it schedules regardless of `busy` (its own claim
    // never needs a currently-unbusy device the way a real VRAM-costed model
    // would, per `place::pick_device`'s zero-cost branch) -- what this test
    // actually proves is that the shared dispatcher/manager code path is
    // still alive and correct after the Thinker's claim, not stuck or
    // corrupted.
    let r2 = exec.run_blocking("trivial", "run", capability::Invocation::new(), |_| {});
    assert!(r2.is_ok(), "single-device model failed to run beside a resident multi-device Thinker: {r2:?}");

    exec.shutdown();
    std::fs::remove_dir_all(&dir).ok();
}

/// A second `generate` call reuses the hot sharded instance rather than
/// re-streaming ~tens of MB of weights across both cards again.
#[test]
fn a_second_generate_reuses_the_hot_sharded_instance() {
    if skip() {
        return;
    }
    let cfg = tiny_cfg(4);
    let dir = std::env::temp_dir().join(format!("int8_thinker_executor_reuse_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("thinker.safetensors");
    write_synthetic_checkpoint(path.to_str().unwrap(), &cfg, 555444);

    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 8 << 30, 0);
    budgets.set(Device::Gpu(1), 8 << 30, 0);
    let exec = Executor::start(vec![], budgets, Policy::default());
    let resident = Int8ThinkerResident::new(path.to_str().unwrap().to_string(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps_for_split(path.to_str().unwrap(), &cfg, &[Device::Gpu(0), Device::Gpu(1)], 2));
    exec.register_multi(Arc::new(resident));

    assert!(exec.run_blocking(MODEL, "generate", generate_inv(&[1, 2], 1, &[]), |_| {}).is_ok());
    assert!(exec.run_blocking(MODEL, "generate", generate_inv(&[3, 4], 1, &[]), |_| {}).is_ok());

    assert_eq!(exec.stats().builds, 1, "second generate must reuse the hot instance, not rebuild/re-stream");
    assert_eq!(exec.stats().resident_multi, 1);

    exec.shutdown();
    std::fs::remove_dir_all(&dir).ok();
}
