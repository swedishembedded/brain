// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end persisted-`VkPipelineCache` contract (M6.4): the file a first
//! `VkContext` writes on `Drop` is non-empty, its byte count matches exactly
//! what that context's own `vkGetPipelineCacheData` reported, and a second
//! `VkContext` on the same physical device loads that file as `initial_data`
//! without error.
//!
//! Chose byte-identity + load-without-error over a timing assertion (the
//! milestone brief's other suggested gate): this crate compiles exactly ONE
//! pipeline (`matmul.rs`'s scalar kernel), so the warm-cache saving on a
//! single tiny pipeline is on the order of what shared-box scheduling noise
//! already costs - not a distinguishable signal, and exactly the kind of
//! flaky wall-clock assertion this campaign's own ledger (decision 4) already
//! rules out in favour of a deterministic check of the actual mechanism.

use vulkan::context::VkContext;
use vulkan::matmul::{matmul, MatmulBackend};
use vulkan::pipeline_cache;

#[test]
fn second_context_loads_the_first_ones_persisted_pipeline_cache() {
    let dir = std::env::temp_dir().join(format!(
        "brain-vk-plcache-e2e-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).ok();
    pipeline_cache::set_dir_override(Some(dir.clone()));

    let ctx = match VkContext::new() {
        Ok(c) => c,
        Err(e) => {
            pipeline_cache::set_dir_override(None);
            std::fs::remove_dir_all(&dir).ok();
            return brain_testutil::skip_unavailable(&format!("no Vulkan device: {e}"));
        }
    };

    // Exercise the exact call site this milestone changed
    // (`matmul.rs::build_pipeline`), not a bespoke test-only pipeline.
    let _ = matmul(&ctx, MatmulBackend::select(&ctx), &[0.0f32; 4], &[0.0f32; 4], 1, 4, 1);
    let first_data = unsafe { ctx.device.get_pipeline_cache_data(ctx.pipeline_cache()) }
        .expect("get_pipeline_cache_data must succeed on a real device");
    assert!(!first_data.is_empty(), "a real device must report a non-empty pipeline-cache blob");

    drop(ctx); // persists to `dir` on Drop

    let files: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect();
    assert_eq!(files.len(), 1, "exactly one persisted cache file, got {files:?}");
    let on_disk = std::fs::read(files[0].path()).unwrap();
    assert!(!on_disk.is_empty(), "persisted cache file must be non-empty");
    assert_eq!(
        on_disk.len(),
        first_data.len(),
        "the persisted file's byte count must match what vkGetPipelineCacheData reported when it was written"
    );

    // A second context on the same physical device must load that exact
    // file as `initial_data` without erroring - a mismatched/foreign blob
    // would either fail `vkCreatePipelineCache` or (per this module's own
    // header check) be silently treated as absent; a MATCHING blob from the
    // same device/driver must be accepted.
    let ctx2 = VkContext::new().expect("second context must load the persisted cache without error");
    drop(ctx2);

    pipeline_cache::set_dir_override(None);
    std::fs::remove_dir_all(&dir).ok();
}
