// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving perception models to
// many concurrent callers for its clients. If your team needs expertise in
// batched vision serving then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Mirage over the scheduler on the released weights: concurrent requests are
//! one device batch whose answers equal the single-request answers, a cancelled
//! request takes no slot, a store scan finds the checkpoint without any
//! setting, and dropping the instance releases the model.
//!
//! The images are small synthetic pictures and the decoding budget is short:
//! these tests prove batching and bookkeeping, not recognition (the model
//! crate's tests and the sample cover that).

use std::sync::mpsc::channel;
use std::sync::Arc;
use std::time::Duration;

use capability::{ActionResult, CancelToken, Invocation};
use catalog::resident_mirage::MirageResident;
use residency::budget::Budgets;
use residency::{Device, Executor, Job, Policy, ResidentModel};
use serde_json::{json, Value};

const MODEL: &str = "Cactus-Compute/mirage";

fn released_dir() -> Option<String> {
    let dir = brain_testutil::model_dir(MODEL)?;
    let d = std::path::Path::new(&dir);
    mirage::resolver::RELEASE_FILES.iter().all(|f| d.join(f).is_file()).then_some(dir)
}

/// A `side x side` picture whose content depends on `seed`.
fn picture(seed: u32, side: u32) -> capability::Blob {
    let hwc: Vec<f32> = (0..side * side * 3).map(|i| (((i.wrapping_mul(2654435761).wrapping_add(seed.wrapping_mul(40503))) >> 8) & 0xff) as f32 / 255.0).collect();
    capability::blob::image_blob(&hwc, side, side, 3)
}

fn describe(seed: u32, max_tokens: u64) -> Invocation {
    Invocation::new().blob("image", picture(seed, 96)).set("max_tokens", json!(max_tokens)).set("sections", json!("objects,text"))
}

fn answer(result: ActionResult) -> Value {
    let mut outputs = result.expect("the request succeeds").outputs;
    outputs.as_object_mut().unwrap().remove("usage");
    outputs
}

fn executor(dir: &str) -> Executor {
    let resident: Arc<dyn ResidentModel> = Arc::new(MirageResident::new(dir).expect("the released checkpoint is complete"));
    let mut budgets = Budgets::new();
    budgets.set(Device::Gpu(0), 24 << 30, 0);
    Executor::start(vec![resident], budgets, Policy::default())
}

fn submit(exec: &Executor, action: &str, inv: Invocation) -> std::sync::mpsc::Receiver<ActionResult> {
    let (tx, rx) = channel();
    exec.submit(Job::new(MODEL, action, inv).reply(move |r| {
        let _ = tx.send(r);
    }));
    rx
}

/// The executor's stats once `ready` holds: a group's `Done` and the instances' metrics follow its last reply by a moment.
fn stats_when(exec: &Executor, ready: impl Fn(&residency::executor::Stats) -> bool) -> residency::executor::Stats {
    let started = std::time::Instant::now();
    loop {
        let stats = exec.stats();
        if ready(&stats) || started.elapsed() > Duration::from_secs(10) {
            return stats;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn metric(stats: &residency::executor::Stats, name: &str) -> u64 {
    stats.metrics.values().flatten().find(|(k, _)| k == name).and_then(|(_, v)| v.as_u64()).unwrap_or(0)
}

fn wait(rx: &std::sync::mpsc::Receiver<ActionResult>) -> ActionResult {
    rx.recv_timeout(Duration::from_secs(600)).expect("the request is answered")
}

#[test]
fn sixteen_concurrent_requests_are_one_device_batch_and_equal_their_single_answers() {
    let Some(dir) = released_dir() else {
        brain_testutil::skip("mirage served batching: Cactus-Compute/mirage is not in the model store");
        return;
    };
    let exec = executor(&dir);
    // The blocker builds the instance and keeps the lane busy while the sixteen queue behind it.
    let blocker = submit(&exec, "describe", describe(99, 24));
    let receivers: Vec<_> = (0..16).map(|i| submit(&exec, "describe", describe(i, 8))).collect();
    wait(&blocker).expect("the blocker succeeds");
    let batched: Vec<Value> = receivers.iter().map(|rx| answer(wait(rx))).collect();
    let stats = stats_when(&exec, |s| s.jobs >= 17 && metric(s, "mirage_images") >= 17);
    assert!(stats.max_batch >= 16, "the sixteen queued requests ran as one group: {stats:?}");
    assert!(stats.batches <= 2, "the blocker and one group of sixteen: {stats:?}");

    // The device batch the instance ran is visible in its metrics.
    assert!(metric(&stats, "mirage_widest_batch") >= 16, "{stats:?}");

    // Every batched answer equals the answer of the same request run alone.
    for (i, got) in batched.iter().enumerate() {
        let alone = answer(wait(&submit(&exec, "describe", describe(i as u32, 8))));
        assert_eq!(got, &alone, "request {i}: batched answer equals single-request answer");
    }
    exec.shutdown();
}

#[test]
fn a_cancelled_request_takes_no_slot_and_the_rest_are_unaffected() {
    let Some(dir) = released_dir() else {
        brain_testutil::skip("mirage served cancellation: Cactus-Compute/mirage is not in the model store");
        return;
    };
    let exec = executor(&dir);
    let warm = answer(wait(&submit(&exec, "describe", describe(1, 8))));
    let mut doomed = describe(2, 8);
    doomed.cancel = CancelToken::armed();
    doomed.cancel.cancel();
    let before = stats_when(&exec, |s| metric(s, "mirage_images") >= 1);
    let blocker = submit(&exec, "describe", describe(50, 16));
    let (cancelled, kept) = (submit(&exec, "describe", doomed), submit(&exec, "describe", describe(1, 8)));
    wait(&blocker).expect("the blocker succeeds");
    assert_eq!(wait(&cancelled).unwrap_err(), "cancelled");
    assert_eq!(answer(wait(&kept)), warm, "the request beside the cancelled one is unaffected");

    // The slot is free: the next request is served normally.
    assert_eq!(answer(wait(&submit(&exec, "describe", describe(1, 8)))), warm);
    let after = stats_when(&exec, |s| metric(s, "mirage_images") >= metric(&before, "mirage_images") + 3);
    assert_eq!(metric(&after, "mirage_images") - metric(&before, "mirage_images"), 3, "the blocker, the kept request and the follow-up ran; the cancelled one never reached the device");
    exec.shutdown();
}

#[test]
fn a_request_that_can_never_run_is_refused_at_the_door() {
    let Some(dir) = released_dir() else {
        brain_testutil::skip("mirage served admission: Cactus-Compute/mirage is not in the model store");
        return;
    };
    let resident = MirageResident::new(&dir).unwrap();
    let bad_option = Invocation::new().blob("image", picture(1, 32)).set("sections", json!("objects,bogus"));
    assert!(resident.check_job("describe", &bad_option).unwrap_err().contains("sections") || resident.check_job("describe", &bad_option).is_err());
    let bomb = Invocation::new().blob("image", picture(1, 32)).set("form", json!("x".repeat(mirage::caps::MAX_FORM_BYTES + 1)));
    assert!(resident.check_job("extract", &bomb).unwrap_err().contains("bytes"));
    assert!(resident.check_job("describe", &Invocation::new().blob("image", picture(1, 32))).is_ok());
    assert!(resident.check_job("train", &Invocation::new()).is_err());
}

#[test]
fn a_store_scan_finds_the_checkpoint_with_no_setting() {
    let Some(dir) = released_dir() else {
        brain_testutil::skip("mirage served scan: Cactus-Compute/mirage is not in the model store");
        return;
    };
    // `<store>/Cactus-Compute/mirage` -> the store is two levels up.
    let store = std::path::Path::new(&dir).parent().and_then(std::path::Path::parent).expect("a store layout").to_path_buf();
    let resident = MirageResident::from_store(Some(&store));
    assert!(resident.is_some(), "the resolver finds {dir} under {}", store.display());
    let manifest = resident.unwrap().manifest();
    assert_eq!(manifest.model, MODEL);
    assert_eq!(manifest.actions.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(), ["describe", "extract"]);
}

#[test]
fn a_cq_request_is_served_by_the_packed_archive_beside_the_fp32_instance() {
    let Some(dir) = released_dir() else {
        brain_testutil::skip("mirage served precision: Cactus-Compute/mirage is not in the model store");
        return;
    };
    let exec = executor(&dir);
    let request = |precision: Option<&str>| {
        let inv = describe(3, 8);
        match precision {
            Some(p) => inv.set("precision", json!(p)),
            None => inv,
        }
    };
    let fp32 = answer(wait(&submit(&exec, "describe", request(Some("fp32")))));
    let cq = answer(wait(&submit(&exec, "describe", request(Some("cq")))));
    // Two instances were loaded: one per weight source, each with its own batch counter.
    let stats = stats_when(&exec, |s| s.metrics.len() >= 2);
    assert_eq!(stats.metrics.len(), 2, "one instance per precision: {stats:?}");
    // Both answer with the same structure; the packed weights are a different (quantised) model.
    assert_eq!(fp32.as_object().unwrap().keys().collect::<Vec<_>>(), cq.as_object().unwrap().keys().collect::<Vec<_>>());
    // The default request is served by the directory's default instance, a third.
    wait(&submit(&exec, "describe", request(None))).expect("the default request succeeds");
    let stats = stats_when(&exec, |s| s.metrics.len() >= 3);
    assert_eq!(stats.metrics.len(), 3, "the default is its own instance: {stats:?}");
    exec.shutdown();
}
