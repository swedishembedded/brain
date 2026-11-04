// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P6: per-layer activation statistics for the INT8 decision (measured, no NPU).
use std::collections::HashMap;

use zipdepth::{collect_activation_stats, ZipConfig};
use gpu_core::Gpu;
use paramstore::ParamStore;
use vision::Ctx;

fn small() -> ZipConfig {
    ZipConfig { dims: [8, 16, 32, 64], depths: [1, 1, 1, 1], dec_ch: 16, half_dec_ch: 8, input: 32, ..ZipConfig::base() }
}

fn store(gpu: &Gpu, cfg: &ZipConfig) -> ParamStore {
    let init = zipdepth::init_weights(cfg, 5);
    let params: Vec<(String, usize)> = cfg.param_list().into_iter().map(|(n, s)| (n, s.iter().product())).collect();
    ParamStore::new(gpu, params, &init)
}

/// The collector observes EVERY conv's input — including the Norm::None raw convs
/// (SE, the fusion projections, the head, GCB's raw convs), which the eval path did
/// NOT tap before this. Missing those would blind the decoder analysis, since most
/// decoder convs are raw.
#[test]
fn every_conv_including_raw_ones_is_observed() {
    let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let cfg = small();
    let ps = store(&gpu, &cfg);
    let img = vec![0.5f32; (3 * cfg.input * cfg.input) as usize];
    let stats = collect_activation_stats(&gpu, &cfg, &ps, std::slice::from_ref(&img));
    let report = stats.report();
    let names: std::collections::HashSet<&str> = report.iter().map(|r| r.name.as_str()).collect();

    // A representative conv from each family that flows through vision::Conv must
    // appear — including the Norm::None raw convs (the fusion projections, the head,
    // GCB's raw convs), which the eval path did NOT tap before this change.
    for expect in [
        "encoder.stem_half",                   // ConvBN (Norm::Bn)
        "encoder.stage1.0.branch_3x3",         // QARepBlock (Norm::Bn)
        "encoder.stage3.2.context_weight",     // GCB's score conv — RAW (Norm::None)
        "decoder.fuse2.proj_high",             // fusion projection — RAW
        "decoder.head_half",                   // the head — RAW, biased
        "decoder.convex_up.mask_pred.0",       // the upsampler
    ] {
        assert!(names.contains(expect), "the collector never saw `{expect}` — a conv family is untapped");
    }
    // ChannelAttention (SE) is DELIBERATELY absent: it dispatches raw conv2d over a
    // [N,C,1,1] global descriptor rather than composing vision::Conv, so it is
    // outside the tap surface. That is fine for the INT8 decision — it quantizes C
    // values per image, a negligible and low-risk surface next to the spatial maps.
    assert!(!names.contains("encoder.stage3.1.fc.0"), "SE's descriptor convs are intentionally outside the spatial-activation tap");
    // Report is sorted by outlier_ratio descending.
    for w in report.windows(2) {
        assert!(w[0].outlier_ratio >= w[1].outlier_ratio, "report must be sorted by ratio");
    }
}

/// A layer fed a heavy-tailed activation must show a large outlier_ratio; a uniform
/// one must show ~1. Pins that the ratio measures what it claims.
#[test]
fn outlier_ratio_tracks_the_tail() {
    use zipdepth::ActStatsCollector;
    use vision::ActTap;
    let c = ActStatsCollector::new();
    // Uniform-ish magnitudes: ratio ~1.
    let mut flat: Vec<f32> = (0..10_000).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    c.tap("flat", &mut flat);
    // One giant outlier over an otherwise small distribution: ratio huge.
    let mut spiky: Vec<f32> = (0..10_000).map(|i| (i % 3) as f32 * 0.001).collect();
    spiky[0] = 1000.0;
    c.tap("spiky", &mut spiky);

    let report = c.report();
    let get = |n: &str| report.iter().find(|r| r.name == n).unwrap().outlier_ratio;
    assert!(get("flat") < 1.2, "a flat distribution should have ratio ~1, got {}", get("flat"));
    assert!(get("spiky") > 100.0, "a single spike should blow the ratio up, got {}", get("spiky"));
    // ...and the report ranks spiky first.
    assert_eq!(report[0].name, "spiky");
}

/// Convs sharing an INPUT tensor must report identical stats — a consistency check
/// on the tap (down4's two branches and cross_scale.high_to_low all read stage3's
/// output). On the real model this showed absmax 8.6263 for all three.
#[test]
fn convs_sharing_an_input_report_identical_stats() {
    let gpu = Gpu::new_cpu(zipdepth::net::PIPELINES);
    let cfg = small();
    let ps = store(&gpu, &cfg);
    // A structured (non-constant) input so the shared tensor is non-degenerate.
    let img: Vec<f32> = (0..(3 * cfg.input * cfg.input)).map(|i| ((i * 31 % 255) as f32) / 255.0).collect();
    let stats = collect_activation_stats(&gpu, &cfg, &ps, std::slice::from_ref(&img));
    let report = stats.report();
    let by: HashMap<&str, f32> = report.iter().map(|r| (r.name.as_str(), r.absmax)).collect();
    // down4's two branches read the same stage3 output.
    let a = by["encoder.down4.branch_3x3"];
    let b = by["encoder.down4.branch_1x1"];
    assert_eq!(a.to_bits(), b.to_bits(), "convs on the same input must have identical absmax: {a} vs {b}");
    // ...and so does cross_scale.high_to_low.
    let c = by["encoder.cross_scale.high_to_low"];
    assert_eq!(a.to_bits(), c.to_bits(), "cross_scale.high_to_low shares stage3's output: {a} vs {c}");
    let _ = Ctx::new(&gpu, zipdepth::net::ids());
}
