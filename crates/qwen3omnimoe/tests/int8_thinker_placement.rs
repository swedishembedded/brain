// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Placement and byte accounting for the int8 sharded Thinker - **no GPU**.
//!
//! `int8_thinker_multi_gpu.rs`/`int8_thinker_executor.rs` prove the sharded
//! model computes the right answer on real cards; those need two physical
//! GPUs and therefore skip on most machines. The two properties tested here
//! are pure functions of a checkpoint's header plus a device list, so they
//! run everywhere, every time - and they are the two that decide whether a
//! real 30B load succeeds or dies partway through:
//!
//! 1. **The estimate is complete.** `estimate_multi` must account for every
//!    tensor `activate_multi` actually uploads. An estimate that silently
//!    omits, say, the attention projections reserves less than it allocates,
//!    which is a budget that lies - the scheduler happily places a second
//!    model on memory that is already spoken for.
//! 2. **The split follows real capacity.** More devices than two, uneven
//!    VRAM, and a model that does not fit at all: all decided from real
//!    per-layer bytes, none of it assuming a card count.

use qwen3omnimoe::int8_thinker_resident::{Int8ThinkerResident, EMBED_TENSOR, MODEL};
use residency::{Device, InstanceKey, MultiDeviceResidentModel};

#[path = "common/int8_thinker_fixture.rs"]
mod fixture;
use fixture::{caps_for_split, tiny_cfg, write_synthetic_checkpoint};

struct Ckpt {
    dir: std::path::PathBuf,
    path: String,
}

impl Drop for Ckpt {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).ok();
    }
}

fn checkpoint(tag: &str, cfg: &qwen3omnimoe::config::MoeTextConfig, seed: u64) -> Ckpt {
    let dir = std::env::temp_dir().join(format!("int8_thinker_placement_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("thinker.safetensors").to_str().unwrap().to_string();
    write_synthetic_checkpoint(&path, cfg, seed);
    Ckpt { dir, path }
}

fn key() -> InstanceKey {
    InstanceKey::new(MODEL, "default")
}

/// Every tensor in the checkpoint, classified the way the LOADER treats it,
/// summed independently of the cost model under test.
///
/// A packed expert linear is uploaded as-is (its kernel consumes int8), so it
/// costs its stored size plus its scale. Every other packed tensor is
/// dequantized on load - `thinker::layer_fwd` has no int8 path for
/// attention/router/head - so it costs FOUR times its stored size. The
/// embedding is read a row at a time and never lands on a device at all.
fn expected_device_bytes(path: &str) -> u64 {
    let reader = checkpoint::weightio::WeightReader::open(path).expect("open");
    let mut total = 0u64;
    for name in reader.names() {
        if name == EMBED_TENSOR || name == format!("{EMBED_TENSOR}.scale") {
            continue; // host-side row gather, never uploaded
        }
        let numel: u64 = reader.shape(name).map(|s| s.iter().product()).unwrap_or(0);
        let packed = reader.dtype(name) == Some("U32");
        let is_expert_scale = name.ends_with(".scale") && name.contains(".mlp.experts.");
        if name.contains(".mlp.experts.") {
            // packed weight as-is; its scale counted on its own iteration
            total += numel * 4;
            let _ = is_expert_scale;
        } else if name.ends_with(".scale") {
            // A scale sibling of a DEQUANTIZED tensor is consumed on the host
            // during the dequantize and never uploaded.
        } else if packed {
            total += numel * 4 * 4; // dequantized to f32 on the way in
        } else {
            total += numel * 4;
        }
    }
    total
}

#[test]
fn the_estimate_accounts_for_every_tensor_the_loader_uploads() {
    let cfg = tiny_cfg(4);
    let ck = checkpoint("accounting", &cfg, 90210);

    let caps = caps_for_split(&ck.path, &cfg, &[Device::Gpu(0), Device::Gpu(1)], 2);
    let resident = Int8ThinkerResident::new(ck.path.clone(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps);
    let cost = resident.estimate_multi(&key());

    assert_eq!(
        cost.total_accelerator_bytes(),
        expected_device_bytes(&ck.path),
        "estimate_multi must charge for exactly what activate_multi uploads -- an under-report is a budget that lies"
    );
    // Split across two, and every named device carries real bytes.
    assert_eq!(cost.devices().count(), 2);
    for d in cost.devices() {
        assert!(cost.on(d) > 0, "{d:?} was named but charged nothing");
    }
    // The embedding is genuinely not device-resident and genuinely not a host
    // copy either (the mapping lends its bytes), so neither pool is charged.
    assert_eq!(cost.ram(), 0, "the embedding table must not be materialized on the host");
}

#[test]
fn the_total_is_the_same_however_many_devices_it_is_split_across() {
    let cfg = tiny_cfg(6);
    let ck = checkpoint("invariant", &cfg, 4242);
    let total = Int8ThinkerResident::new(ck.path.clone(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), Vec::new()).total_device_bytes().expect("measurable");

    for n in 1..=3usize {
        let devices: Vec<Device> = (0..n as u32).map(Device::Gpu).collect();
        let caps = caps_for_split(&ck.path, &cfg, &devices, n);
        let cost = Int8ThinkerResident::new(ck.path.clone(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps).estimate_multi(&key());
        assert_eq!(cost.devices().count(), n, "{n}-way capacity should produce {n} stages");
        assert_eq!(cost.total_accelerator_bytes(), total, "sharding must not change what the model costs, only where it sits");
    }
}

/// The generic property, on the model rather than on `model::shard` in
/// isolation: capacity - not card count - decides the split.
#[test]
fn uneven_capacity_puts_more_layers_on_the_bigger_card() {
    let cfg = tiny_cfg(8);
    let ck = checkpoint("uneven", &cfg, 31337);
    let total = Int8ThinkerResident::new(ck.path.clone(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), Vec::new()).total_device_bytes().expect("measurable");

    // Card 0 is three times card 1, and neither alone can hold the model.
    let big = total * 3 / 4;
    let small = total / 2;
    let resident = Int8ThinkerResident::new(ck.path.clone(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), vec![(Device::Gpu(0), big), (Device::Gpu(1), small)]);
    let cost = resident.estimate_multi(&key());

    assert_eq!(cost.devices().count(), 2, "neither card alone fits, so it must shard");
    assert!(cost.on(Device::Gpu(0)) > cost.on(Device::Gpu(1)), "the bigger card must take the bigger share: {} vs {}", cost.on(Device::Gpu(0)), cost.on(Device::Gpu(1)));
    assert!(cost.on(Device::Gpu(0)) <= big, "card 0's share must fit card 0");
    assert!(cost.on(Device::Gpu(1)) <= small, "card 1's share must fit card 1");
}

/// Three devices with three different capacities - proving nothing here is a
/// hardcoded two-card assumption.
#[test]
fn three_devices_with_three_capacities_all_get_a_fitting_share() {
    let cfg = tiny_cfg(9);
    let ck = checkpoint("three", &cfg, 5150);
    let total = Int8ThinkerResident::new(ck.path.clone(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), Vec::new()).total_device_bytes().expect("measurable");

    // Deliberately lopsided, and no two of them together can hold it.
    let caps = vec![(Device::Gpu(0), total / 2), (Device::Gpu(1), total * 2 / 5), (Device::Gpu(2), total / 4)];
    let resident = Int8ThinkerResident::new(ck.path.clone(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps.clone());
    let cost = resident.estimate_multi(&key());

    assert_eq!(cost.devices().count(), 3, "must use all three when no prefix of them fits");
    for (d, cap) in caps {
        assert!(cost.on(d) <= cap, "{d:?} charged {} against a {cap}-byte capacity", cost.on(d));
    }
    assert_eq!(cost.total_accelerator_bytes(), total);
}

/// A model that fits ONE card must not be spread across the others: an extra
/// stage costs a cross-device hop per token and strands capacity other models
/// could use.
#[test]
fn a_model_that_fits_one_card_stays_on_one_card() {
    let cfg = tiny_cfg(4);
    let ck = checkpoint("onecard", &cfg, 606);
    let total = Int8ThinkerResident::new(ck.path.clone(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), Vec::new()).total_device_bytes().expect("measurable");

    let caps = vec![(Device::Gpu(0), total * 4), (Device::Gpu(1), total * 4), (Device::Gpu(2), total * 4)];
    let cost = Int8ThinkerResident::new(ck.path.clone(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps).estimate_multi(&key());
    assert_eq!(cost.devices().count(), 1);
    assert_eq!(cost.on(Device::Gpu(0)), total);
}

/// Does not fit anywhere ⇒ a cost naming ZERO devices, which is
/// `MultiDeviceResidentModel`'s documented "unavailable" signal. Never a
/// panic (this runs on the Executor's dispatcher thread) and never a plan
/// that overruns a card.
#[test]
fn a_model_that_fits_nowhere_reports_zero_devices_rather_than_overrunning() {
    let cfg = tiny_cfg(4);
    let ck = checkpoint("nofit", &cfg, 1);
    let total = Int8ThinkerResident::new(ck.path.clone(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), Vec::new()).total_device_bytes().expect("measurable");

    let caps = vec![(Device::Gpu(0), total / 8), (Device::Gpu(1), total / 8)];
    let resident = Int8ThinkerResident::new(ck.path.clone(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg.clone()), caps);
    let cost = resident.estimate_multi(&key());
    assert_eq!(cost.devices().count(), 0, "an unplaceable model must report zero devices");

    // ...and activating it is a clean error, not a panic or a half-built instance.
    let err = resident.activate_multi(&key(), &[Device::Gpu(0)]).err().expect("activate must refuse");
    assert!(err.contains("no placement"), "{err}");
}

#[test]
fn an_unreadable_checkpoint_reports_zero_devices_rather_than_panicking() {
    let resident = Int8ThinkerResident::new("/nonexistent/thinker.safetensors".to_string(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(tiny_cfg(2)), vec![(Device::Gpu(0), 1 << 40)]);
    assert_eq!(resident.estimate_multi(&key()).devices().count(), 0);
    assert!(resident.total_device_bytes().is_err());
}

#[test]
fn no_budgeted_devices_reports_zero_devices() {
    let cfg = tiny_cfg(2);
    let ck = checkpoint("nodev", &cfg, 7);
    let resident = Int8ThinkerResident::new(ck.path.clone(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg), Vec::new());
    assert_eq!(resident.estimate_multi(&key()).devices().count(), 0);
}

/// `claim_multi` hands `activate_multi` exactly the devices `estimate_multi`
/// named. Anything else means the reservation and the load describe different
/// cards, so it must be refused rather than silently re-planned.
#[test]
fn activate_refuses_a_device_set_the_plan_did_not_choose() {
    let cfg = tiny_cfg(4);
    let ck = checkpoint("mismatch", &cfg, 8080);
    let caps = caps_for_split(&ck.path, &cfg, &[Device::Gpu(0), Device::Gpu(1)], 2);
    let resident = Int8ThinkerResident::new(ck.path.clone(), qwen3omnimoe::config::ThinkerConfig::defaults().with_text(cfg), caps);
    assert_eq!(resident.estimate_multi(&key()).devices().count(), 2);

    let err = resident.activate_multi(&key(), &[Device::Gpu(0)]).err().expect("a one-device set must be refused");
    assert!(err.contains("different cards"), "{err}");
}
