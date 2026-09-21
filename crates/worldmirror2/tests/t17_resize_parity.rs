// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T17 gate: the DPT's spatial kernels, at the shapes this model uses.
//!
//! The head resolves its detail by upsampling a coarse feature map, and the
//! last of those steps is a NON-INTEGER 1.75x (208x272 -> 364x476). A
//! source-coordinate mapping can be exactly right at an integer ratio and
//! wrong at a fractional one, so a 2x2 -> 4x4 check proves very little: at
//! that size `align_corners` has only the two endpoints to place and any
//! plausible formula places them identically.
//!
//! Goldens come from `torch.nn.functional.interpolate(..., mode="bilinear",
//! align_corners=True)` - an oracle outside this workspace, which is the point.
//! Generate with `tools/goldens/worldmirror2_dump_resize.py <dir>` and point
//! `WM_RESIZE_GOLDEN` at it.
//!
//! The same argument applies to the convolutions. A shared kernel can be
//! right for the shapes another model exercises and wrong at a stride, a
//! padding, or an odd extent this one happens to need - and the transposed
//! convolutions that upsample the taps are the least-travelled path of all.
//!
//! Swedish Embedded AB implements GPU kernels that are gated against an
//! independent oracle at the shapes they actually run, not at whatever shape
//! was convenient to write down. If your team needs that, you can procure our
//! services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;

fn read_f32(p: &str) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn bilinear_upsampling_matches_torch_at_this_models_shapes() {
    let Ok(dir) = std::env::var("WM_RESIZE_GOLDEN") else {
        brain_testutil::skip("set WM_RESIZE_GOLDEN to a dir written by tools/goldens/worldmirror2_dump_resize.py");
        return;
    };
    let man: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/cases.json")).expect("cases.json"))
            .expect("valid cases.json");
    let gpu = Gpu::new_cpu(worldmirror2::model::PIPELINES);
    let dk = worldmirror2::model::dpt_kernels(0);

    // Loose enough for fp32 accumulation order, tight enough that a
    // source-coordinate convention error (which moves samples by a half pixel,
    // not a few ulp) cannot hide under it.
    const TOL: f32 = 1e-3;
    let mut worst_overall = 0.0f32;
    let mut bad: Vec<String> = Vec::new();
    for case in man.as_array().expect("array") {
        let g = |k: &str| case[k].as_u64().unwrap() as usize;
        let (i, c, hin, win, hout, wout) =
            (g("i"), g("c"), g("hin"), g("win"), g("hout"), g("wout"));
        let x = read_f32(&format!("{dir}/in_{i}.f32"));
        let want = read_f32(&format!("{dir}/out_{i}.f32"));
        assert_eq!(x.len(), c * hin * win);
        assert_eq!(want.len(), c * hout * wout);

        let xb = gpu.storage_init("x", &x);
        let yb = gpu.storage((c * hout * wout) as u64);
        let s = gpu.step(
            dk.resize_bilinear,
            &[&xb, &yb],
            &[1, c as u32, hin as u32, win as u32, hout as u32, wout as u32, 1],
            (c * hout * wout) as u32,
        );
        gpu.submit(&[], &[s]);
        let got = gpu.read(&yb, c * hout * wout);

        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale;
        eprintln!("  {hin}x{win} -> {hout}x{wout} (c={c}): worst {worst:.3e} relative");
        worst_overall = worst_overall.max(worst);
        bad.extend((worst >= TOL).then(|| format!("{hin}x{win} -> {hout}x{wout}: {worst:.3e}")));
    }
    eprintln!("worst across all shapes: {worst_overall:.3e}");
    assert!(bad.is_empty(), "differs from torch beyond fp32 rounding at {bad:?}");
}

/// The DPT's convolutions, including the transposed ones that upsample the
/// taps, against torch at the strides, paddings and extents this model runs.
#[test]
fn dpt_convolutions_match_torch_at_this_models_shapes() {
    let Ok(dir) = std::env::var("WM_RESIZE_GOLDEN") else {
        brain_testutil::skip("set WM_RESIZE_GOLDEN to a dir written by tools/goldens/worldmirror2_dump_resize.py");
        return;
    };
    let man: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{dir}/conv_cases.json")).expect("conv_cases.json"),
    )
    .expect("valid conv_cases.json");
    let gpu = Gpu::new_cpu(worldmirror2::model::PIPELINES);
    let dk = worldmirror2::model::dpt_kernels(0);
    const TOL: f32 = 1e-3;
    let mut bad: Vec<String> = Vec::new();

    for case in man.as_array().expect("array") {
        let g = |k: &str| case[k].as_u64().unwrap() as usize;
        let (i, cin, cout) = (g("i"), g("cin"), g("cout"));
        let (hin, win, k, stride, pad) = (g("hin"), g("win"), g("k"), g("stride"), g("pad"));
        let (hout, wout) = (g("hout"), g("wout"));
        let tr = case["transposed"].as_bool().unwrap();
        let name = case["name"].as_str().unwrap();

        let x = gpu.storage_init("x", &read_f32(&format!("{dir}/cin_{i}.f32")));
        let wt = gpu.storage_init("w", &read_f32(&format!("{dir}/cw_{i}.f32")));
        let bias = gpu.storage_init("b", &read_f32(&format!("{dir}/cb_{i}.f32")));
        let want = read_f32(&format!("{dir}/cout_{i}.f32"));
        let n = cout * hout * wout;
        assert_eq!(want.len(), n, "{name}: golden is {} floats, expected {n}", want.len());
        let out = gpu.storage(n as u64);

        let conv = if tr {
            gpu.step(
                dk.conv2d_dx,
                &[&x, &wt, &out],
                &[1, cout as u32, hout as u32, wout as u32, cin as u32, k as u32, k as u32, 0, hin as u32, win as u32],
                n as u32,
            )
        } else {
            gpu.step(
                dk.conv2d,
                &[&x, &wt, &out],
                &[1, cin as u32, hin as u32, win as u32, cout as u32, k as u32, stride as u32, pad as u32, hout as u32, wout as u32],
                n as u32,
            )
        };
        let add = gpu.step(dk.add_chan_inplace, &[&out, &bias], &[n as u32, cout as u32, (hout * wout) as u32], n as u32);
        gpu.submit(&[], &[conv, add]);
        let got = gpu.read(&out, n);

        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale;
        eprintln!("  {name}: {cin}x{hin}x{win} -> {cout}x{hout}x{wout}  worst {worst:.3e} relative");
        bad.extend((worst >= TOL).then(|| format!("{name}: {worst:.3e}")));
    }
    assert!(bad.is_empty(), "differs from torch beyond fp32 rounding at {bad:?}");
}
