// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T1 — PIL-exact preprocessing parity; T2 — DINOv2 patch-token parity.
//!
//! Goldens come from `tools/mirror_dump_reference.py` (run manually against
//! the reference repo + checkpoint); the committed `golden_meta.json` holds
//! shape + rms + 64–256 sampled values per stage. The synthetic input image
//! is regenerated here bit-for-bit (same integer formula as the dump script).
//!
//! T2 needs the real 5 GB checkpoint → gated on MIRROR_CKPT.

use mirror::config::MirrorConfig;
use mirror::preprocess::{resize_bicubic, resize_dims, RgbImage, IMAGENET_MEAN, IMAGENET_STD};

fn synth_image(w: usize, h: usize) -> RgbImage {
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            rgb[i] = ((x * 255) / (w - 1).max(1)) as u8;
            rgb[i + 1] = ((y * 255) / (h - 1).max(1)) as u8;
            rgb[i + 2] = (((x * 7 + y * 13) / 4) % 256) as u8;
        }
    }
    for &(cx, cy, r, col) in &[
        (150i64, 100i64, 60i64, [255u8, 40, 40]),
        (420, 260, 90, [30, 220, 90]),
        (300, 180, 30, [250, 250, 250]),
    ] {
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r {
                    let i = ((y as usize) * w + x as usize) * 3;
                    rgb[i..i + 3].copy_from_slice(&col);
                }
            }
        }
    }
    RgbImage { w, h, rgb }
}

struct Sample {
    shape: Vec<usize>,
    rms: f64,
    indices: Vec<usize>,
    values: Vec<f64>,
}

fn meta() -> serde_json::Value {
    serde_json::from_str(include_str!("golden/golden_meta.json")).unwrap()
}

fn get_sample(m: &serde_json::Value, key: &str) -> Sample {
    let v = &m[key];
    let arr = |k: &str| v[k].as_array().unwrap_or_else(|| panic!("bad golden {key}.{k}"));
    Sample {
        shape: arr("shape").iter().map(|x| x.as_u64().unwrap() as usize).collect(),
        rms: v["rms"].as_f64().unwrap(),
        indices: arr("indices").iter().map(|x| x.as_u64().unwrap() as usize).collect(),
        values: arr("values").iter().map(|x| x.as_f64().unwrap()).collect(),
    }
}

/// Compare a stage against its golden; returns an error description instead
/// of panicking so one run reports every diverging stage.
fn diff_sample(name: &str, got: &[f32], s: &Sample, tol: f32) -> Option<String> {
    let numel: usize = s.shape.iter().product();
    if got.len() != numel {
        return Some(format!("{name}: len {} vs {numel}", got.len()));
    }
    let rms = (got.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / got.len() as f64).sqrt();
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for (&i, &v) in s.indices.iter().zip(&s.values) {
        let d = (got[i] - v as f32).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    if (rms - s.rms).abs() > (0.001 * s.rms.abs()).max(1e-6) || worst > tol {
        return Some(format!(
            "{name}: rms {rms:.6} vs golden {:.6}; worst [{worst_i}] got {} vs {} (tol {tol})",
            s.rms,
            got[worst_i],
            s.values[s.indices.iter().position(|&x| x == worst_i).unwrap()]
        ));
    }
    None
}

fn check_sample(name: &str, got: &[f32], s: &Sample, tol: f32) {
    if let Some(e) = diff_sample(name, got, s, tol) {
        panic!("{e}");
    }
}

#[test]
fn t1_pil_bicubic_exact() {
    let m = meta();
    let img = synth_image(600, 400);
    let dims = m["t1_dims"].as_array().unwrap();
    let (nw, nh) = (dims[0].as_u64().unwrap() as usize, dims[1].as_u64().unwrap() as usize);
    assert_eq!(resize_dims(600, 400, 518, 14), (nw, nh));
    let resized = resize_bicubic(&img, nw, nh);

    // resized u8: golden sampled from HWC u8 — BIT-exact (tol < 1)
    let got: Vec<f32> = resized.rgb.iter().map(|&b| b as f32).collect();
    check_sample("t1_resized_u8", &got, &get_sample(&m, "t1_resized_u8"), 0.5);

    // normalized CHW
    let mut chw = vec![0.0f32; 3 * nw * nh];
    for c in 0..3 {
        for y in 0..nh {
            for x in 0..nw {
                let v = resized.rgb[(y * nw + x) * 3 + c] as f32 / 255.0;
                chw[c * nw * nh + y * nw + x] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }
    check_sample("t1_norm_chw", &chw, &get_sample(&m, "t1_norm_chw"), 1e-5);
}

/// T2: 518×518 square crop path → DINOv2 patch tokens vs reference (CPU
/// backend; needs MIRROR_CKPT).
#[test]
fn t2_dinov2_patch_tokens() {
    let Ok(ckpt) = std::env::var("MIRROR_CKPT") else {
        eprintln!("MIRROR_CKPT not set — skipping");
        return;
    };
    let m = meta();
    // reference: pil.crop((41, 0, 441, 400)) → 400x400 → resize 518x518
    let img = synth_image(600, 400);
    let mut crop = RgbImage { w: 400, h: 400, rgb: vec![0; 400 * 400 * 3] };
    for y in 0..400 {
        let src = (y * 600 + 41) * 3;
        let dst = y * 400 * 3;
        crop.rgb[dst..dst + 1200].copy_from_slice(&img.rgb[src..src + 1200]);
    }
    let sq = resize_bicubic(&crop, 518, 518);
    let got_u8: Vec<f32> = sq.rgb.iter().map(|&b| b as f32).collect();
    check_sample("t2_input_u8", &got_u8, &get_sample(&m, "t2_input_u8"), 0.5);

    // model.forward takes RAW [0,1] CHW (it normalizes internally).
    let mut chw = vec![0.0f32; 3 * 518 * 518];
    for c in 0..3 {
        for y in 0..518 {
            for x in 0..518 {
                chw[c * 518 * 518 + y * 518 + x] = sq.rgb[(y * 518 + x) * 3 + c] as f32 / 255.0;
            }
        }
    }

    let cfg = MirrorConfig::default();
    let init = mirror::import::load(&ckpt, &cfg).expect("import");
    let gpu = gpu_core::Gpu::new_cpu(mirror::model::PIPELINES);
    let mut model = mirror::model::Mirror::new(&gpu, cfg, &init, 0);
    drop(init);
    model.forward(&chw, 1, 37, 37);
    let got = gpu.read(model.patch_tokens(), 1369 * 1024);
    check_sample("t2_patch_tokens", &got, &get_sample(&m, "t2_patch_tokens"), 2e-4);

    // ---- T4 + T5: report every diverging stage in one run ----
    let mut errs: Vec<String> = Vec::new();
    for (i, tap) in model.taps().iter().enumerate() {
        let got = gpu.read(tap, 1376 * 2048);
        if let Some(e) = diff_sample(&format!("t4_tap{i}"), &got, &get_sample(&m, &format!("t4_tap{i}")), 3e-3) {
            errs.push(e);
        }
    }
    use mirror::model::Head;
    let px = 518 * 518;
    for (key, head, ch) in [
        ("t5_depth_head", Head::Depth, 3usize),
        ("t5_pts_head", Head::Points, 4),
        ("t5_norm_head", Head::Normals, 4),
        ("t5_gs_head", Head::GsDepth, 3),
        ("t5_gs_params", Head::GsParams, 12),
    ] {
        let got = gpu.read(model.head_out(head, 0), ch * px);
        if let Some(e) = diff_sample(key, &got, &get_sample(&m, key), 5e-3) {
            errs.push(e);
        }
    }
    let mut cam = model.cam_pred_raw();
    cam[7] = cam[7].max(0.0); // reference activates fov with relu
    cam[8] = cam[8].max(0.0);
    if let Some(e) = diff_sample("t5_cam", &cam, &get_sample(&m, "t5_cam"), 2e-3) {
        errs.push(format!("{e}; full cam vec {cam:?}"));
    }
    assert!(errs.is_empty(), "stage mismatches:\n{}", errs.join("\n"));
}
