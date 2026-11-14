// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A measured kernel choice outlives the process that measured it, on every GPU backend.
//!
//! Swedish Embedded AB implements reproducible LLM serving for its clients, where the same binary on the same card
//! must give the same tokens after a restart. If your team needs expertise in autotuned kernels whose choices do not
//! drift between launches, you can procure our services by sending an email to info@swedishembedded.com.
//!
//! The candidates an autotuner chooses between (split-K slice counts, the int8 GEMV against the tile) round
//! differently, so the choice is part of what a served model computes. A device whose choices were kept in process
//! memory only measured them again on every launch, and timing noise could pick a different reduction order for the
//! same shape: greedy output changed between two launches of one binary on a CUDA card, where no wgpu adapter is
//! recorded. The store is keyed by the device itself (its backend and physical card) instead.

use backend_api::select::TuneStore;
use gpu_core::tune::FileTuneStore;

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-tune-store-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

#[test]
fn a_choice_saved_on_a_device_is_read_back_by_a_later_store_for_it() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if gpu.identity().is_none() && gpu_core::adapter_info().is_none() {
        brain_testutil::skip_unavailable("the backend names no device to file choices under");
        return;
    }
    let dir = scratch("roundtrip");
    let first = FileTuneStore::in_dir(&dir, &gpu, 0x5eed).expect("every device a backend names has a store");
    first.save("MatMul/F32/64x1024x1024/schedule", "tile=wide128/split_k=4");
    // A store opened afterwards, as the next launch opens it, reads the choice instead of measuring.
    let later = FileTuneStore::in_dir(&dir, &gpu, 0x5eed).expect("the same device's store");
    assert_eq!(later.load("MatMul/F32/64x1024x1024/schedule").as_deref(), Some("tile=wide128/split_k=4"));
    // Other kernel sources are another store: an edited candidate is measured again.
    let edited = FileTuneStore::in_dir(&dir, &gpu, 0x5eee).expect("a store for other sources");
    assert_eq!(edited.load("MatMul/F32/64x1024x1024/schedule"), None);
    let _ = std::fs::remove_dir_all(&dir);
}
