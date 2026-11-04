// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight-free rungs of the parity ladder: the import mapping, the tiny-config
//! forward, and the seam itself against a real (tiny) UNet.
//!
//! None of these needs a checkpoint or a golden, so they run everywhere and are
//! the tests that localise a failure before `tests/parity.rs` hits it opaquely
//! at 5 GB of fp32 weights.

use std::collections::HashMap;

use controlnet::adapter::{check_compatible, order_for, ControlAdapter, ControlSource};
use controlnet::config::ControlNetConfig;
use controlnet::model::{ControlNet, KERNELS};

/// A synthetic diffusers-side checkpoint for `cfg`: exactly the source tensor
/// names the importer expects, derived by INVERTING the brain manifest.
fn synthetic_source(cfg: &ControlNetConfig) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let mut raw: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut put = |name: String, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        raw.insert(name, (shape, data));
    };
    for (name, shape) in cfg.tensor_manifest() {
        if let Some(b) = name.strip_suffix(".attn1.qkv.weight") {
            for leaf in ["to_q", "to_k", "to_v"] {
                put(format!("{b}.attn1.{leaf}.weight"), vec![shape[0] / 3, shape[1]]);
            }
        } else if let Some(b) = name.strip_suffix(".attn2.kv.weight") {
            for leaf in ["to_k", "to_v"] {
                put(format!("{b}.attn2.{leaf}.weight"), vec![shape[0] / 2, shape[1]]);
            }
        } else if let Some(b) = name.strip_suffix(".ff.hidden.weight") {
            put(format!("{b}.ff.net.0.proj.weight"), vec![shape[0] * 2, shape[1]]);
        } else if let Some(b) = name.strip_suffix(".ff.hidden.bias") {
            put(format!("{b}.ff.net.0.proj.bias"), vec![shape[0] * 2]);
        } else if name.contains(".ff.gate.") {
            // Both GEGLU halves come from the one `proj` tensor above.
        } else if let Some(b) = name.strip_suffix(".ff.out.weight") {
            put(format!("{b}.ff.net.2.weight"), shape);
        } else if let Some(b) = name.strip_suffix(".ff.out.bias") {
            put(format!("{b}.ff.net.2.bias"), shape);
        } else if let Some(b) = name.strip_suffix(".to_out.weight") {
            put(format!("{b}.to_out.0.weight"), shape);
        } else if let Some(b) = name.strip_suffix(".to_out.bias") {
            put(format!("{b}.to_out.0.bias"), shape);
        } else {
            put(name, shape);
        }
    }
    raw
}

/// Rung 1: mapping units. Two-way coverage means an extra source tensor is an
/// ERROR — that is the check that stops a full SDXL UNet checkpoint (which has
/// `up_blocks.*`) from being loaded as a ControlNet.
#[test]
fn import_round_trips_a_synthetic_checkpoint() {
    let cfg = ControlNetConfig::tiny();
    let raw = synthetic_source(&cfg);
    let t = controlnet::import::remap(raw.clone(), &cfg).expect("remap");
    assert_eq!(t.len(), cfg.tensor_manifest().len());
    for (name, shape) in cfg.tensor_manifest() {
        let (s, d) = t.get(&name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(*s, shape, "{name}");
        assert_eq!(d.len(), shape.iter().product::<usize>(), "{name}");
    }

    let mut extra = raw.clone();
    extra.insert("up_blocks.0.resnets.0.conv1.weight".into(), (vec![1], vec![0.0]));
    let e = controlnet::import::remap(extra, &cfg).expect_err("an up-block tensor must be rejected");
    assert!(e.contains("unused"), "{e}");

    let mut missing = raw;
    missing.remove("controlnet_mid_block.weight");
    let e = controlnet::import::remap(missing, &cfg).expect_err("a missing zero-conv must be rejected");
    assert!(e.contains("controlnet_mid_block"), "{e}");
}

/// Rung 2: the tiny forward runs, every residual is finite, and — because the
/// synthetic zero-convs are deliberately non-zero — none of them is identically
/// zero. An all-zero residual set is what a ControlNet at init produces, so it
/// is exactly the failure this assertion must not accept.
#[test]
fn tiny_forward_produces_finite_nonzero_residuals() {
    let cfg = ControlNetConfig::tiny();
    let t = controlnet::init::init_weights(&cfg, 11);
    let (h, w) = (8u32, 8u32);
    let ds = cfg.cond_downscale();
    let m = ControlNet::new(gpu_core::testgpu::dev(&KERNELS), cfg.clone(), &t, h, w, 5, true);
    assert_eq!(m.cond_size(), (h * ds, w * ds));

    let b = &cfg.backbone;
    let sample: Vec<f32> = (0..(b.in_channels * h * w) as usize).map(|i| (i % 13) as f32 * 0.05 - 0.3).collect();
    let cond: Vec<f32> =
        (0..(cfg.conditioning_channels * h * ds * w * ds) as usize).map(|i| (i % 251) as f32 / 250.0).collect();
    let enc: Vec<f32> = (0..(5 * b.cross_attention_dim) as usize).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
    let pooled: Vec<f32> = (0..b.pooled_dim() as usize).map(|i| (i % 5) as f32 * 0.2).collect();
    let time_ids = [64.0, 64.0, 0.0, 0.0, 64.0, 64.0];

    let r = m.run(&sample, 601.0, &enc, &pooled, &time_ids, &cond, 1.0);
    let points = ControlSource::injection_points(&m);
    assert_eq!(r.len(), points.len());
    for p in &points {
        let v = r.get(&p.name).unwrap_or_else(|| panic!("no residual for {}", p.name));
        assert_eq!(v.len(), p.numel(), "{}", p.name);
        assert!(v.iter().all(|x| x.is_finite()), "{} has non-finite values", p.name);
        assert!(v.iter().any(|x| x.abs() > 1e-9), "{} is identically zero", p.name);
    }

    // `conditioning_scale` is a device buffer, so a different scale is a WRITE,
    // not a graph rebuild. Same graph, same object, exactly 0.5x.
    let half = m.run(&sample, 601.0, &enc, &pooled, &time_ids, &cond, 0.5);
    for p in &points {
        let (a, b) = (r.get(&p.name).expect("full"), half.get(&p.name).expect("half"));
        let worst = a.iter().zip(b).map(|(x, y)| (0.5 * x - y).abs() as f64).fold(0.0, f64::max);
        assert!(worst < 1e-6, "{}: scale is not a pure multiply (max |0.5x - y| = {worst:.3e})", p.name);
    }
}

/// Rung 2b: the pooled (production) graph and the tapped one must agree
/// bit-for-bit. Taps disable the activation pool, so this is what says the
/// pooled buffer reuse is safe.
#[test]
fn pooled_graph_is_bit_identical_to_the_tapped_one() {
    let cfg = ControlNetConfig::tiny();
    let t = controlnet::init::init_weights(&cfg, 3);
    let (h, w) = (8u32, 8u32);
    let ds = cfg.cond_downscale();
    let b = &cfg.backbone;
    let sample: Vec<f32> = (0..(b.in_channels * h * w) as usize).map(|i| (i % 11) as f32 * 0.07 - 0.4).collect();
    let cond: Vec<f32> =
        (0..(cfg.conditioning_channels * h * ds * w * ds) as usize).map(|i| (i % 97) as f32 / 96.0).collect();
    let enc: Vec<f32> = (0..(5 * b.cross_attention_dim) as usize).map(|i| (i % 9) as f32 * 0.05).collect();
    let pooled: Vec<f32> = (0..b.pooled_dim() as usize).map(|i| (i % 3) as f32 * 0.3).collect();
    let time_ids = [64.0, 64.0, 0.0, 0.0, 64.0, 64.0];

    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let tapped = ControlNet::new(gpu.share(), cfg.clone(), &t, h, w, 5, true);
    let plain = ControlNet::new(gpu, cfg.clone(), &t, h, w, 5, false);
    let a = tapped.run(&sample, 601.0, &enc, &pooled, &time_ids, &cond, 0.9);
    let c = plain.run(&sample, 601.0, &enc, &pooled, &time_ids, &cond, 0.9);
    for n in a.names() {
        assert_eq!(a.get(n), c.get(n), "{n} differs between the tapped and pooled graphs");
    }
}

/// Rung 3 of the SEAM (not of the model): a tiny ControlNet and the tiny UNet
/// it was configured from agree on every injection point, by name and by
/// element count, and the UNet actually consumes them.
///
/// This is the test that would fail if `crates/controlnet` had hardcoded SDXL's
/// down-block list instead of deriving points from the backbone.
#[test]
fn the_seam_matches_a_real_unet_and_the_unet_consumes_it() {
    let cfg = ControlNetConfig::tiny();
    let (h, w) = (8u32, 8u32);
    let t_enc = 5u32;
    let gpu = gpu_core::testgpu::dev(&KERNELS);

    let cn = ControlNet::new(
        gpu.share(),
        cfg.clone(),
        &controlnet::init::init_weights(&cfg, 5),
        h,
        w,
        t_enc,
        false,
    );
    // ONE device for both models: `controlnet::model::KERNELS` is a strict
    // prefix-extension of the UNet's set.
    let un = sdxlunet::Unet::new_controlled(
        gpu,
        cfg.backbone.clone(),
        &sdxlunet::init::init_weights(&cfg.backbone, 5),
        h,
        w,
        t_enc,
        false,
        true,
    );
    assert!(ControlAdapter::accepts_control(&un));
    check_compatible(&un, &cn).expect("a ControlNet built from this backbone's config must fit it");

    let b = &cfg.backbone;
    let ds = cfg.cond_downscale();
    let sample: Vec<f32> = (0..(b.in_channels * h * w) as usize).map(|i| (i % 13) as f32 * 0.05 - 0.3).collect();
    let cond: Vec<f32> =
        (0..(cfg.conditioning_channels * h * ds * w * ds) as usize).map(|i| (i % 251) as f32 / 250.0).collect();
    let enc: Vec<f32> = (0..(t_enc * b.cross_attention_dim) as usize).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
    let pooled: Vec<f32> = (0..b.pooled_dim() as usize).map(|i| (i % 5) as f32 * 0.2).collect();
    let time_ids = [64.0, 64.0, 0.0, 0.0, 64.0, 64.0];

    let base = un.run(&sample, 601.0, &enc, &pooled, &time_ids);

    // scale = 0 must be an exact no-op on the backbone: the residuals are all
    // zero, so every add is `+ 0`. This is what pins that the control path is
    // additive and touches nothing else.
    let zero = cn.run(&sample, 601.0, &enc, &pooled, &time_ids, &cond, 0.0);
    let ordered = order_for(&un, &zero).expect("ordering");
    let off = un.run_with_control(&sample, 601.0, &enc, &pooled, &time_ids, &ordered);
    let worst = base.iter().zip(&off).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0.0, f64::max);
    assert!(worst == 0.0, "conditioning_scale = 0 moved the UNet output by {worst:.3e}");

    // ... and a real scale must move it.
    let r = cn.run(&sample, 601.0, &enc, &pooled, &time_ids, &cond, 1.0);
    let ordered = order_for(&un, &r).expect("ordering");
    let on = un.run_with_control(&sample, 601.0, &enc, &pooled, &time_ids, &ordered);
    let moved = base.iter().zip(&on).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0.0, f64::max);
    assert!(moved > 1e-6, "the control residuals changed nothing (max |Δ| = {moved:.3e})");
}

/// Rung 3b: **where** a residual lands in the backbone — the one property
/// `scale = 0 is a no-op` and `scale = 1 moves the output` cannot see between
/// them.
///
/// diffusers adds a `down.k` residual to `down_block_res_samples`, which ONLY
/// the up path consumes; the running hidden state is untouched. Adding it to
/// `hh` as well double-counts it through the mid block and every later down
/// block. At the LAST down point that mistake is even shape-legal — `down.{n-1}`
/// is exactly the tensor entering `mid_block` — so it type-checks, runs, still
/// reduces to a no-op at `scale = 0`, and still "moves the output" at
/// `scale = 1`. Nothing else in this file or in `tests/parity.rs` would fail:
/// the ControlNet's own residuals would still match diffusers to 1e-11, because
/// the bug is in the CONSUMER.
///
/// The gate: inject at exactly one down point at a time and read the
/// `mid.resnet1` tap. The mid block must be **bit-identical** every time, and
/// the final output must not be. The `mid` residual is the complementary case —
/// it is added AFTER `mid_block`, so it too must leave that tap alone while
/// changing the output.
#[test]
fn a_down_residual_reaches_the_output_only_through_the_up_path() {
    let cfg = ControlNetConfig::tiny();
    let (h, w, t_enc) = (8u32, 8u32, 5u32);
    let b = &cfg.backbone;
    let un = sdxlunet::Unet::new_controlled(
        gpu_core::testgpu::dev(&KERNELS),
        b.clone(),
        &sdxlunet::init::init_weights(b, 5),
        h,
        w,
        t_enc,
        true, // taps: `mid.resnet1` is the probe
        true,
    );
    let shapes: Vec<(u32, u32, u32)> = un.control_shapes().to_vec();
    assert!(shapes.len() >= 3, "need at least two down points and a mid");

    let sample: Vec<f32> = (0..(b.in_channels * h * w) as usize).map(|i| (i % 13) as f32 * 0.05 - 0.3).collect();
    let enc: Vec<f32> = (0..(t_enc * b.cross_attention_dim) as usize).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
    let pooled: Vec<f32> = (0..b.pooled_dim() as usize).map(|i| (i % 5) as f32 * 0.2).collect();
    let time_ids = [64.0, 64.0, 0.0, 0.0, 64.0, 64.0];

    let zeros: Vec<Vec<f32>> =
        shapes.iter().map(|&(c, sh, sw)| vec![0.0f32; (c * sh * sw) as usize]).collect();
    let base = un.run_with_control(&sample, 601.0, &enc, &pooled, &time_ids, &zeros);
    let base_mid = un.read_tap("mid.resnet1").expect("the mid block is tapped");
    assert!(base_mid.iter().any(|v| v.abs() > 1e-9), "the mid tap is identically zero, so it proves nothing");

    for k in 0..shapes.len() {
        let mut r = zeros.clone();
        // A varying, order-sensitive pattern: a constant would survive a
        // channel permutation of the injected residual.
        r[k] = (0..r[k].len()).map(|i| (i % 29) as f32 * 0.03 - 0.4).collect();
        let out = un.run_with_control(&sample, 601.0, &enc, &pooled, &time_ids, &r);
        let mid = un.read_tap("mid.resnet1").expect("the mid block is tapped");
        let name = if k + 1 == shapes.len() { "mid".to_string() } else { format!("down.{k}") };
        // A max-|Δ| rather than `assert_eq!` on the vectors: these are 2048
        // floats and a failing `assert_eq!` would print both in full.
        let leaked =
            base_mid.iter().zip(&mid).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0.0, f64::max);
        assert!(
            leaked == 0.0,
            "{name}: the residual moved the mid block by {leaked:.3e} — it must reach the output \
             only through the up path (down.*) or only after mid_block (mid)"
        );
        let moved = base.iter().zip(&out).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0.0, f64::max);
        assert!(moved > 1e-6, "{name}: the residual changed nothing (max |Δ| = {moved:.3e})");
    }
}

/// A ControlNet whose backbone disagrees with the UNet's must be REJECTED by
/// name, not silently zipped. Here the latent size differs, which keeps every
/// name identical and changes only the element counts — the case a name-only
/// check would pass.
#[test]
fn a_mismatched_backbone_is_rejected_by_name() {
    let cfg = ControlNetConfig::tiny();
    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let cn = ControlNet::new(
        gpu.share(),
        cfg.clone(),
        &controlnet::init::init_weights(&cfg, 5),
        8,
        8,
        5,
        false,
    );
    let un = sdxlunet::Unet::new_controlled(
        gpu,
        cfg.backbone.clone(),
        &sdxlunet::init::init_weights(&cfg.backbone, 5),
        16,
        16,
        5,
        false,
        true,
    );
    let e = check_compatible(&un, &cn).expect_err("a 16x16 UNet must reject an 8x8 ControlNet");
    assert!(e.contains("down.0"), "{e}");
}
