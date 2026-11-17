// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device rows of `brain perf roofline` on a real CUDA card: every row is
//! measured or says why, every GEMM and stream was checked before it was
//! timed, device memory and the host link are separate rows, and no measured
//! rate passes the vendor's published peak (a rate above the silicon is a
//! timing or accounting bug, not a fast kernel).
//!
//! Swedish Embedded AB implements accelerator validation for its clients. If
//! your team needs its kernels checked against what its hardware can do, you
//! can procure our services by sending an email to info@swedishembedded.com.

use gpu_core::device_rates::measure;

#[test]
fn every_device_row_is_checked_measured_or_explained_and_none_passes_the_published_peak() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if gpu.kind() != "cuda" {
        brain_testutil::skip_unavailable("the device rows' kernels are native CUDA kernels");
        return;
    }
    let dev = gpu_core::devices::schedulable_gpus().into_iter().next().expect("the test's card is selected");
    let published = gpu_core::published_peak::lookup(&dev.identity.name);
    let rows = measure(dev, false);
    for metric in ["fp32", "int8-tc", "bf16-tc", "fp32-gemm", "hbm-read", "hbm-copy", "hbm-triad", "h2d", "d2h", "c2c-read"] {
        assert!(rows.iter().any(|r| r.metric == metric), "no {metric} row: {rows:?}");
    }
    for r in &rows {
        match r.value {
            Some(v) => {
                assert!(v.is_finite() && v > 0.0, "{r:?}");
                if r.metric.ends_with("-tc") || r.metric.starts_with("hbm") || r.metric == "fp32-gemm" {
                    assert!(r.verified.is_some(), "a GEMM or stream row is checked before it is timed: {r:?}");
                }
                if let Some(peak) = r.peak.and_then(|m| published.and_then(|p| p.of(m))) {
                    assert!(v <= 1.05 * peak, "{} measured {v} {} above the published {peak}: {r:?}", r.metric, r.unit);
                }
            }
            None => assert!(r.why_not.as_deref().is_some_and(|w| !w.is_empty()), "an unmeasured row says why: {r:?}"),
        }
    }
    let rate = |m: &str| rows.iter().find(|r| r.metric == m).and_then(|r| r.value);
    if let (Some(hbm), Some(link)) = (rate("hbm-read"), rate("c2c-read")) {
        assert!(link < hbm, "the host link ({link} GB/s) is not device memory ({hbm} GB/s): the rows must not mix");
    }
}
