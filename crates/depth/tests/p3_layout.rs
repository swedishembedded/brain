// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P3: the assembled ZipDepth's parameter layout.
//!
//! Two independent things must agree, and the whole design leans on it:
//!   * `ZipConfig::param_list` — device-free, written from the reference's source.
//!   * `ZipDepth::param_list` — emitted by the BUILT graph, block by block.
//! They are written from the same spec but by different routes, so a disagreement
//! means one of them is wrong. `p1_param_layout.rs` separately checks the config
//! against a real `.pth` (env-gated); this makes the graph inherit that guarantee.
use depth::{ZipConfig, ZipDepth};
use gpu_core::Gpu;

fn build(cfg: ZipConfig) -> (Gpu, Vec<(String, usize)>) {
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let m = ZipDepth::new(&gpu, cfg, 1, true);
    let p = m.param_list();
    (gpu, p)
}

/// THE test: the graph's tensors and the config's must be the same set, with the
/// same shapes, in the same order.
#[test]
fn the_built_graph_matches_the_config_exactly() {
    let cfg = ZipConfig::base();
    let want: Vec<(String, usize)> =
        cfg.param_list().into_iter().map(|(n, s)| (n, s.iter().product::<usize>())).collect();
    let (_gpu, got) = build(ZipConfig::base());

    // Report the FIRST divergence by name rather than dumping 278 lines: a missing
    // block shifts everything after it and the raw diff is unreadable.
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        assert_eq!(w.0, g.0, "tensor {i}: config says `{}`, the graph built `{}`", w.0, g.0);
        assert_eq!(w.1, g.1, "tensor {i} `{}`: config says {} elements, the graph built {}", w.0, w.1, g.1);
    }
    assert_eq!(
        want.len(),
        got.len(),
        "tensor COUNT differs: config {} vs graph {} — first extra is `{}`",
        want.len(),
        got.len(),
        want.get(got.len().min(want.len() - 1)).map(|p| p.0.as_str()).unwrap_or("<none>")
    );
}

/// The released `zipdepth_base.pth` has **278 tensors / 6,802,927 elements**, of
/// which 43 are the int64 `num_batches_tracked` counters that brain does not carry.
/// So the graph must have 278 - 43 = 235 tensors, and the elements must match
/// exactly (the counters are scalars, contributing 43 elements).
///
/// These numbers are from an independent Python-side inspection of the real
/// file — not from anything in this repo.
#[test]
fn the_graph_has_the_released_checkpoints_tensor_count_and_size() {
    let (_gpu, got) = build(ZipConfig::base());
    assert_eq!(got.len(), 278 - 43, "235 tensors = the .pth's 278 minus its 43 num_batches_tracked counters");
    let numel: usize = got.iter().map(|(_, n)| n).sum();
    assert_eq!(numel, 6_802_927 - 43, "elements must match the .pth minus its 43 scalar counters");
}

/// The unfused count. The 6.1M headline is POST-fusion (RepVGG's 3x3+1x1+identity
/// collapse); the checkpoint stores the larger form, and so does the training graph.
#[test]
fn the_unfused_parameter_count_is_the_checkpoints_not_the_headline() {
    let (_gpu, got) = build(ZipConfig::base());
    let numel: usize = got.iter().map(|(_, n)| n).sum();
    assert!(numel > 6_700_000, "unfused is ~6.79M, not the 6.1M post-fusion headline; got {numel}");
}

/// The NPU checkpoint is a DIFFERENT model: `where_conv.*` instead of
/// `mask_pred.*`. Its released count is 283 tensors / 6,801,324 elements with 44
/// counters (one more BN than the unfold path).
#[test]
fn the_npu_variant_matches_its_own_checkpoint() {
    let cfg = ZipConfig { upsample_unfold: false, ..ZipConfig::base() };
    let (_gpu, got) = build(cfg);
    assert_eq!(got.len(), 283 - 44, "the npu .pth's 283 tensors minus its 44 counters");
    let numel: usize = got.iter().map(|(_, n)| n).sum();
    assert_eq!(numel, 6_801_324 - 44);
    let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.iter().any(|n| n.contains("where_conv")), "the npu path predicts a blend weight");
    assert!(!names.iter().any(|n| n.contains("mask_pred")), "and must NOT carry the unfold path's mask");
}

/// Every tensor name must be unique. A duplicate would mean two modules share a
/// prefix and silently alias each other's weights — which `ParamStore` would
/// accept, since it is keyed by name.
#[test]
fn every_tensor_name_is_unique() {
    for cfg in [ZipConfig::base(), ZipConfig { upsample_unfold: false, ..ZipConfig::base() }] {
        let (_gpu, got) = build(cfg);
        let mut seen = std::collections::HashSet::new();
        for (n, _) in &got {
            assert!(seen.insert(n.clone()), "`{n}` is emitted twice — two modules share a prefix");
        }
    }
}

/// The stage tails' ORDER is part of the checkpoint's index numbering, not a free
/// choice: stage2 is [QARep x2, MinimalMultiScale, StripPoolingAttention] so the
/// MMS is `.2` and the strip gate `.3`. Swapping them keeps every shape loadable
/// while putting the weights in the wrong modules.
#[test]
fn the_stage_tails_are_numbered_in_the_references_order() {
    let (_gpu, got) = build(ZipConfig::base());
    let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
    // depths[1] == 2 -> the two QARepBlocks are .0/.1, MMS is .2, the gate is .3.
    assert!(names.contains(&"encoder.stage2.2.branch1.weight"), "MinimalMultiScale must be stage2.2");
    assert!(names.contains(&"encoder.stage2.3.gate_conv.0.weight"), "StripPoolingAttention must be stage2.3");
    // depths[2] == 6 -> SE is .6, GCB is .7.
    assert!(names.contains(&"encoder.stage3.6.fc.0.weight"), "ChannelAttention must be stage3.6");
    assert!(names.contains(&"encoder.stage3.7.context_weight.weight"), "GlobalContextBlock must be stage3.7");
}
