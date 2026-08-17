// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Stage-level DPT-head isolation vs the reference on a tiny 4×4 grid with
//! random seeded weights (`tools/goldens/worldmirror2_dump_dpt_tiny.py` → scratchpad bins).
//! Env-gated: MIRROR_DPT_TINY=<dir>. Reports the FIRST diverging stage:
//! rn0..rn3 → fused → full → out.

use std::collections::HashMap;

use gpu_core::Gpu;
use worldmirror2::config::MirrorConfig;
use worldmirror2::dpt::{DptCtx, DptScratch, HeadWeights};

fn read_bin(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

#[test]
fn resize_bilinear_matches_torch() {
    let gpu = Gpu::new_cpu(worldmirror2::model::PIPELINES);
    let dk = worldmirror2::model::dpt_kernels(0);
    let x = gpu.storage_init("x", &[1.0, 2.0, 3.0, 4.0]);
    let y = gpu.storage(16);
    let s = gpu.step(dk.resize_bilinear, &[&x, &y], &[1, 1, 2, 2, 4, 4, 1], 16);
    gpu.submit(&[], &[s]);
    let got = gpu.read(&y, 16);
    let expect = [
        1.0, 1.3333334, 1.6666667, 2.0,
        1.6666667, 2.0, 2.3333335, 2.6666667,
        2.3333333, 2.6666667, 3.0, 3.3333335,
        3.0, 3.3333334, 3.6666667, 4.0,
    ];
    for (i, (&g, &e)) in got.iter().zip(&expect).enumerate() {
        assert!((g - e).abs() < 1e-5, "[{i}] {g} vs {e}");
    }
}

#[test]
fn dpt_tiny_stages() {
    let Ok(dir) = std::env::var("MIRROR_DPT_TINY") else {
        eprintln!("MIRROR_DPT_TINY not set — skipping");
        return;
    };
    let stages: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/stages.json")).unwrap()).unwrap();
    let cfg = MirrorConfig::default();
    let (ph, pw) = (4usize, 4usize);
    let td = 7 + ph * pw;
    let (h, w) = (ph * 14, pw * 14);

    // weights: every depth_head.* param from the config list, loaded from bins
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    let mut plist = Vec::new();
    for (name, shape) in cfg.param_list() {
        if let Some(rest) = name.strip_prefix("depth_head.") {
            let numel: usize = shape.iter().product();
            let v = read_bin(&format!("{dir}/w_{rest}.bin"));
            assert_eq!(v.len(), numel, "{name}");
            init.insert(name.clone(), v);
            plist.push((name, numel));
        }
    }
    let gpu = Gpu::new_cpu(worldmirror2::model::PIPELINES);
    let ps = paramstore::ParamStore::new(&gpu, plist, &init);

    let tokens = read_bin(&format!("{dir}/tokens.bin"));
    assert_eq!(tokens.len(), td * 2048);
    let taps: Vec<gpu_core::DeviceBuffer> =
        (0..4).map(|_| gpu.storage_init("tap", &tokens)).collect();

    let scr = DptScratch::new(&gpu, &cfg, ph, pw);
    let dk = worldmirror2::model::dpt_kernels(0);
    let ctx = DptCtx { gpu: &gpu, k: dk, cfg: &cfg, scr: &scr, eps: 1e-5 };
    let out = gpu.storage((3 * h * w) as u64);
    let hw = HeadWeights { ps: &ps, prefix: "depth_head" };
    let mut steps = Vec::new();
    ctx.head_frame(&hw, &taps, 0, td, (ph, pw), 3, &out, None, &mut steps);
    gpu.submit(&[], &steps);


    let check = |name: &str, got: &[f32]| -> Option<String> { check_stage(&stages, name, got) };
    let _ = &check;
    fn check_stage(stages: &serde_json::Value, name: &str, got: &[f32]) -> Option<String> {
        let s = &stages[name];
        let idx: Vec<usize> =
            s["indices"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let vals: Vec<f32> =
            s["values"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let rms_g = s["rms"].as_f64().unwrap();
        let rms = (got.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / got.len() as f64).sqrt();
        let mut worst = (0.0f32, 0usize);
        for (&i, &v) in idx.iter().zip(&vals) {
            let d = (got[i] - v).abs();
            if d > worst.0 {
                worst = (d, i);
            }
        }
        if (rms - rms_g).abs() > 0.002 * rms_g.abs().max(1e-6) || worst.0 > 5e-3 {
            Some(format!("{name}: rms {rms:.5} vs {rms_g:.5}, worst abs {} at [{}]", worst.0, worst.1))
        } else {
            None
        }
    }

    let spat = [16 * ph * pw, 4 * ph * pw, ph * pw, 4]; // spatial per scale (4x4 grid: s3 = 2x2)
    let mut errs = Vec::new();
    for (i, &sp) in spat.iter().enumerate() {
        let got = gpu.read(&scr.rn[i], 256 * sp);
        if let Some(e) = check(&format!("rn{i}"), &got) {
            errs.push(e);
        }
    }
    // "fused" golden = post-output_conv1 at (128, 8ph, 8pw) == scr.a
    let fused = gpu.read(&scr.a, 128 * 8 * ph * 8 * pw);
    if let Some(e) = check("fused", &fused) {
        errs.push(e);
    }
    let full = gpu.read(&scr.full_a, 128 * h * w);
    if let Some(e) = check("full", &full) {
        errs.push(e);
    }
    let out_v = gpu.read(&out, 3 * h * w);
    if let Some(e) = check("out", &out_v) {
        errs.push(e);
    }
    assert!(errs.is_empty(), "DPT stage mismatches:\n{}", errs.join("\n"));

    // manual fusion replay with per-refinenet probes (same math as head_frame)
    let f2 = 256usize;
    let dims = [
        (f2, 4 * ph, 4 * pw),
        (f2, 2 * ph, 2 * pw),
        (f2, ph, pw),
        (f2, ph.div_ceil(2), pw.div_ceil(2)),
    ];
    let probe = |tag: &str, buf: &gpu_core::DeviceBuffer, n: usize, errs: &mut Vec<String>| {
        let got = gpu.read(buf, n);
        if let Some(e) = check_stage(&stages, tag, &got) {
            errs.push(e);
        }
    };
    let mut errs2 = Vec::new();
    {
        let mut st = Vec::new();
        ctx.rcu(&hw, "scratch.refinenet4.resConfUnit2", &scr.rn[3], &scr.a, &scr.t, &scr.b, dims[3], &mut st);
        ctx.bilinear(&scr.b, &scr.a, dims[3], (dims[2].1, dims[2].2), &mut st);
        ctx.conv(&scr.a, ps.w("depth_head.scratch.refinenet4.out_conv.weight"), Some(ps.w("depth_head.scratch.refinenet4.out_conv.bias")), &scr.b, dims[2], f2, 1, 1, 0, &mut st);
        gpu.submit(&[], &st);
        probe("out4", &scr.b, f2 * dims[2].1 * dims[2].2, &mut errs2);
        for (r, rn_i, tag) in [(3usize, 2usize, "out3"), (2, 1, "out2"), (1, 0, "out1")] {
            let mut st = Vec::new();
            let pre = format!("scratch.refinenet{r}");
            let n = dims[rn_i].0 * dims[rn_i].1 * dims[rn_i].2;
            ctx.rcu(&hw, &format!("{pre}.resConfUnit1"), &scr.rn[rn_i], &scr.a, &scr.t, &scr.u, dims[rn_i], &mut st);
            st.push(gpu.step(dk.add2, &[&scr.b, &scr.u, &scr.a], &[n as u32], n as u32));
            ctx.rcu(&hw, &format!("{pre}.resConfUnit2"), &scr.a, &scr.t, &scr.u, &scr.b, dims[rn_i], &mut st);
            let target = if rn_i == 0 { (8 * ph, 8 * pw) } else { (dims[rn_i - 1].1, dims[rn_i - 1].2) };
            ctx.bilinear(&scr.b, &scr.a, dims[rn_i], target, &mut st);
            ctx.conv(&scr.a, ps.w(&format!("depth_head.{pre}.out_conv.weight")), Some(ps.w(&format!("depth_head.{pre}.out_conv.bias"))), &scr.b, (f2, target.0, target.1), f2, 1, 1, 0, &mut st);
            gpu.submit(&[], &st);
            probe(tag, &scr.b, f2 * target.0 * target.1, &mut errs2);
        }
    }
    assert!(errs2.is_empty(), "fusion probes:\n{}", errs2.join("\n"));
}
