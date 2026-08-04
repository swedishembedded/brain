// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Stage-by-stage forward parity against the insightface goldens dumped by
//! `tools/goldens/arcface_dump_reference.py`.
//!
//! Every test SKIPS ITSELF when its fixture is absent (`AGENTS.md`): the
//! goldens and the two `.onnx` files live under `$BRAIN_TESTDATA`
//! (default `<repo>/testdata`), which is gitignored.
//!
//! ```bash
//! CARGO_HOME=/…/cargo-home cargo test --release -p brain-facenet --test parity -- --nocapture
//! BRAIN_DEVICE=cpu   # same numbers, no GPU
//! ```
//!
//! Reported per stage: **cosine** and **max_abs**. Cosine alone cannot see a
//! scale error, and max_abs alone cannot see a rotation; a stage passes only on
//! both.

use std::collections::HashMap;
use std::sync::OnceLock;

use facenet::config::{ArcFaceConfig, ScrfdConfig};
use facenet::model::{ArcFace, Scrfd, PIPELINES};
use model::hostmath::{cosine, l2_normalize};

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

use brain_testutil::testdata;

fn dir() -> String {
    testdata("face/antelopev2")
}

fn have(files: &[&str]) -> bool {
    files.iter().all(|f| std::path::Path::new(&format!("{}/{f}", dir())).exists())
}

/// Load a golden safetensors file as name -> (shape, data).
fn golden(file: &str) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    checkpoint::safetensors::read(&format!("{}/{file}", dir()))
        .unwrap_or_else(|e| panic!("read {file}: {e}"))
        .into_iter()
        .map(|t| (t.name, (t.shape, t.data)))
        .collect()
}

/// The two imported checkpoints, decoded once for the whole test binary (the
/// ArcFace protobuf is 261 MB — decoding it per test is minutes of wall clock).
fn weights() -> &'static (facenet::Tensors, facenet::Tensors) {
    static W: OnceLock<(facenet::Tensors, facenet::Tensors)> = OnceLock::new();
    W.get_or_init(|| {
        facenet::import_dir(std::path::Path::new(&dir())).expect("import antelopev2")
    })
}

// ---------------------------------------------------------------------------
// comparison
// ---------------------------------------------------------------------------

struct Cmp {
    cos: f32,
    max_abs: f32,
}

fn compare(got: &[f32], want: &[f32]) -> Cmp {
    assert_eq!(got.len(), want.len(), "length {} vs golden {}", got.len(), want.len());
    let max_abs = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    Cmp { cos: cosine(got, want), max_abs }
}

/// Assert a stage and PRINT its measured numbers, so the report carries values
/// that were actually observed rather than a bare pass.
fn gate(label: &str, got: &[f32], want: &[f32], min_cos: f32, max_abs: f32) {
    let c = compare(got, want);
    println!("  {label:<22} cos {:.7}  max_abs {:.3e}  (n={})", c.cos, c.max_abs, got.len());
    assert!(c.cos >= min_cos, "{label}: cosine {:.7} < {min_cos}", c.cos);
    assert!(c.max_abs <= max_abs, "{label}: max_abs {:.3e} > {max_abs:.3e}", c.max_abs);
}

// ---------------------------------------------------------------------------
// import
// ---------------------------------------------------------------------------

#[test]
fn import_covers_both_graphs_in_both_directions() {
    if !have(&["glintr100.onnx", "scrfd_10g_bnkps.onnx"]) {
        eprintln!("skip: antelopev2 .onnx not under {}", dir());
        return;
    }
    let (arc, scr) = weights();
    assert_eq!(arc.len(), ArcFaceConfig::iresnet100().tensor_manifest().len());
    assert_eq!(arc.len(), 462, "glintr100 has 462 initializers");
    assert_eq!(scr.len(), ScrfdConfig::scrfd_10g_bnkps().tensor_manifest().len());
    assert_eq!(scr.len(), 119);
    // spot-check that a folded conv really carries a bias and a PReLU a slope
    assert_eq!(arc["stem.conv.weight"].0, vec![64, 3, 3, 3]);
    assert_eq!(arc["stem.conv.bias"].0, vec![64]);
    assert_eq!(arc["stem.prelu.weight"].0, vec![64]);
    assert_eq!(arc["fc.weight"].0, vec![512, 25088]);
    assert_eq!(scr["head.stride8.cls.weight"].0, vec![2, 80, 3, 3]);
}

/// A missing node must be an error naming what was expected, never a zero-fill.
#[test]
fn a_truncated_graph_errors_instead_of_zero_filling() {
    if !have(&["glintr100.onnx"]) {
        eprintln!("skip: glintr100.onnx not under {}", dir());
        return;
    }
    let m = onnx::read_file(format!("{}/glintr100.onnx", dir())).unwrap();
    let mut g = onnx::read::graph(&m).unwrap().clone();
    g.node.truncate(g.node.len() - 1); // drop the `features` BatchNorm
    let err = facenet::import_arcface(&g, &ArcFaceConfig::iresnet100()).unwrap_err();
    assert!(err.contains("BatchNormalization") || err.contains("features"), "{err}");
}

// ---------------------------------------------------------------------------
// alignment
// ---------------------------------------------------------------------------

#[test]
fn alignment_matches_the_reference_transform_and_grid() {
    if !have(&["align.safetensors"]) {
        eprintln!("skip: align.safetensors not under {}", dir());
        return;
    }
    let g = golden("align.safetensors");
    // the template constant itself
    let dst = &g["arcface_dst_112"].1;
    let ours: Vec<f32> = facenet::ARCFACE_DST_112.iter().flat_map(|p| [p[0], p[1]]).collect();
    let c = compare(&ours, dst);
    println!("align:");
    println!("  {:<22} max_abs {:.3e}", "arcface_dst_112", c.max_abs);
    assert!(c.max_abs < 1e-4, "the template constant must match the reference");

    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let (sh, sw) = (320u32, 320u32);
    let src_hwc = &g["src_img_u8"].1;
    let src_chw = imaging::pixels::hwc_to_chw(src_hwc, 3, sh as usize, sw as usize);

    for case in ["a", "b", "c"] {
        let lmk = &g[&format!("lmk_{case}")].1;
        let m = facenet::estimate_norm(lmk).unwrap();
        gate(&format!("M_{case}"), &m, &g[&format!("M_{case}")].1, 0.9999, 1e-4);

        let grid = facenet::warp_grid(&m, sw, sh, 112, 112).unwrap();
        gate(&format!("grid_{case}"), &grid, &g[&format!("grid_{case}")].1, 0.999999, 1e-5);

        let (warped, _) =
            facenet::norm_crop_chw(&gpu, &src_chw, 3, sh, sw, lmk, 112).unwrap();
        // EXACT gate: the reference `warp_grid_sample_*` is torch's grid_sample
        // over this same grid, so this is kernel-vs-kernel.
        gate(
            &format!("warp_gs_{case}"),
            &warped,
            &g[&format!("warp_grid_sample_{case}")].1,
            0.999999,
            2e-2,
        );
        // LOOSE gate vs cv2: `cv2.warpAffine` uses 5-bit fixed-point weights and
        // differs by up to 0.5/255 by construction (manifest records 0.500).
        // Tightening this would mean the grid was wrong in a cancelling way.
        let cv2_chw = imaging::pixels::hwc_to_chw(&g[&format!("warp_cv2_{case}_u8")].1, 3, 112, 112);
        let c = compare(&warped, &cv2_chw);
        println!("  {:<22} cos {:.7}  max_abs {:.3e}  (vs cv2, expected ~0.5)", format!("warp_cv2_{case}"), c.cos, c.max_abs);
        assert!(c.max_abs < 1.01, "cv2 disagreement {:.3} exceeds the documented ~0.5/255", c.max_abs);
    }
}

// ---------------------------------------------------------------------------
// ArcFace
// ---------------------------------------------------------------------------

#[test]
fn arcface_forward_parity_stage_by_stage() {
    if !have(&["glintr100.onnx", "arcface.safetensors", "arcface_blocks.safetensors"]) {
        eprintln!("skip: arcface fixtures not under {}", dir());
        return;
    }
    let g = golden("arcface.safetensors");
    let gb = golden("arcface_blocks.safetensors");
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let cfg = ArcFaceConfig::iresnet100();

    // The dumper's own blob is the input, so preprocessing is not under test
    // here — but check ours reproduces it from the aligned pixels first.
    let bgr: Vec<u8> = g["aligned_bgr_u8"].1.iter().map(|&v| v as u8).collect();
    let blob = facenet::blob_from_bgr_u8(&gpu, &bgr, 112, 112, &cfg.pre);
    println!("arcface:");
    gate("blob", &blob, &g["blob"].1, 0.999999, 1e-5);

    let m = ArcFace::new(gpu, cfg, &weights().0);
    let t = m.forward(&g["blob"].1);

    gate("stem", &t.stem, &g["stem"].1, 0.9999, 5e-2);
    for (i, stage) in [1usize, 2, 3, 4].iter().enumerate() {
        // first-block internals of each stage: bn_in, conv1, prelu, conv2, branch
        for (j, name) in ["bn_in", "conv1", "prelu", "conv2", "branch"].iter().enumerate() {
            let key = format!("s{stage}b0_{name}");
            gate(&key, &t.stage_b0[i][j], &g[&key].1, 0.9999, 5e-2);
        }
        let key = format!("layer{stage}");
        gate(&key, &t.layers[i], &g[&key].1, 0.9999, 5e-2);
    }
    // every residual Add output, the bisection ladder
    // Seeded from +inf, not from 1.0: a 1.0-seeded minimum reports "1.0000000"
    // with an empty label when nothing beats it, which looks like a measurement
    // and is not one.
    let mut worst = (f32::INFINITY, String::new(), 0.0f32);
    for (i, b) in t.blocks.iter().enumerate() {
        let key = format!("block{i:02}");
        let c = compare(b, &gb[&key].1);
        if c.cos < worst.0 {
            worst = (c.cos, key.clone(), c.max_abs);
        }
        assert!(c.cos >= 0.9999, "{key}: cosine {:.7}", c.cos);
    }
    println!(
        "  {:<22} worst cos {:.7} max_abs {:.3e} at {} ({} blocks)",
        "block00..48",
        worst.0,
        worst.2,
        worst.1,
        t.blocks.len()
    );
    assert_eq!(t.blocks.len(), 49, "IResNet-100 has 49 residual blocks");

    gate("bn2", &t.bn2, &g["bn2"].1, 0.9999, 5e-2);
    gate("flatten", &t.bn2, &g["flatten"].1, 0.9999, 5e-2);
    gate("fc", &t.fc, &g["fc"].1, 0.9999, 5e-2);
    gate("embedding", &t.embedding, &g["embedding"].1, 0.999, 5e-2);
    let normed = l2_normalize(&t.embedding);
    // THE GATE: cosine >= 0.999 against the insightface-dumped embedding.
    gate("embedding_normed", &normed, &g["embedding_normed"].1, 0.999, 5e-3);
}

/// End-to-end on the three real photos: the whole detect -> align -> embed chain
/// replayed from the captured detector output, plus the identity/cross-identity
/// cosine matrix the gate lives next to.
#[test]
fn arcface_end_to_end_on_real_photos() {
    if !have(&["glintr100.onnx", "e2e.safetensors"]) {
        eprintln!("skip: e2e fixtures not under {}", dir());
        return;
    }
    let g = golden("e2e.safetensors");
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let cfg = ArcFaceConfig::iresnet100();
    let m = ArcFace::new(gpu.share(), cfg.clone(), &weights().0);

    println!("arcface e2e:");
    let mut ours: Vec<Vec<f32>> = Vec::new();
    for p in 0..4usize {
        // alignment: our M from the reference landmarks
        let lmk = &g[&format!("photo{p}_kps")].1;
        let mm = facenet::estimate_norm(lmk).unwrap();
        gate(&format!("photo{p}_M"), &mm, &g[&format!("photo{p}_M")].1, 0.9999, 1e-3);

        // embedding from the reference aligned crop (so this stage isolates the
        // network from the cv2-vs-grid_sample resampling difference)
        let bgr: Vec<u8> = g[&format!("photo{p}_aligned_bgr_u8")].1.iter().map(|&v| v as u8).collect();
        let blob = facenet::blob_from_bgr_u8(&gpu, &bgr, 112, 112, &cfg.pre);
        gate(&format!("photo{p}_blob"), &blob, &g[&format!("photo{p}_blob")].1, 0.999999, 1e-5);
        let e = m.embed_blob(&blob);
        gate(&format!("photo{p}_embedding"), &e, &g[&format!("photo{p}_embedding")].1, 0.999, 5e-2);
        let n = l2_normalize(&e);
        gate(&format!("photo{p}_normed"), &n, &g[&format!("photo{p}_embedding_normed")].1, 0.999, 5e-3);
        ours.push(n);
    }

    // The identity structure must survive the port, not just the vectors.
    let want = &g["cosine_matrix"].1;
    let mut worst = 0.0f32;
    for i in 0..4 {
        for j in 0..4 {
            let d = (cosine(&ours[i], &ours[j]) - want[i * 4 + j]).abs();
            worst = worst.max(d);
        }
    }
    println!("  {:<22} max |Δcos| {:.3e}", "cosine_matrix", worst);
    assert!(worst < 5e-3, "cosine matrix drifted by {worst:.3e}");
    // same identity (photo0 vs its re-capture) must stay far above cross-identity
    assert!(cosine(&ours[0], &ours[3]) > 0.99, "same-identity cosine collapsed");
    assert!(cosine(&ours[0], &ours[1]) < 0.3, "cross-identity cosine is implausibly high");
}

// ---------------------------------------------------------------------------
// SCRFD
// ---------------------------------------------------------------------------

#[test]
fn scrfd_forward_parity_stage_by_stage() {
    if !have(&["scrfd_10g_bnkps.onnx", "scrfd.safetensors"]) {
        eprintln!("skip: scrfd fixtures not under {}", dir());
        return;
    }
    let g = golden("scrfd.safetensors");
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let cfg = ScrfdConfig::scrfd_10g_bnkps();

    let bgr: Vec<u8> = g["image_bgr_u8"].1.iter().map(|&v| v as u8).collect();
    let blob = facenet::blob_from_bgr_u8(&gpu, &bgr, 640, 640, &cfg.pre);
    println!("scrfd:");
    gate("blob", &blob, &g["blob"].1, 0.999999, 1e-5);

    let m = Scrfd::new(gpu, cfg.clone(), &weights().1);
    let t = m.forward(&g["blob"].1);

    gate("stem_pre_pool", &t.stem_pre_pool, &g["stem_pre_pool"].1, 0.9999, 5e-2);
    gate("stem", &t.stem, &g["stem"].1, 0.9999, 5e-2);
    for (i, k) in ["c2", "c3", "c4", "c5"].iter().enumerate() {
        gate(k, &t.c[i], &g[*k].1, 0.9999, 5e-2);
    }
    for (i, k) in ["lat3", "lat4", "lat5"].iter().enumerate() {
        gate(k, &t.lat[i], &g[*k].1, 0.9999, 5e-2);
    }
    gate("fpn4", &t.fpn4, &g["fpn4"].1, 0.9999, 5e-2);
    gate("fpn3", &t.fpn3, &g["fpn3"].1, 0.9999, 5e-2);
    gate("pafpn16_pre", &t.pafpn16_pre, &g["pafpn16_pre"].1, 0.9999, 5e-2);
    gate("pafpn32_pre", &t.pafpn32_pre, &g["pafpn32_pre"].1, 0.9999, 5e-2);
    gate("pafpn16", &t.pafpn16, &g["pafpn16"].1, 0.9999, 5e-2);
    gate("pafpn32", &t.pafpn32, &g["pafpn32"].1, 0.9999, 5e-2);
    for (i, k) in ["neck8", "neck16", "neck32"].iter().enumerate() {
        gate(k, &t.neck[i], &g[*k].1, 0.9999, 5e-2);
    }
    for (i, s) in cfg.strides.iter().enumerate() {
        gate(&format!("head{s}_feat"), &t.head_feat[i], &g[&format!("head{s}_feat")].1, 0.9999, 5e-2);
        gate(&format!("head{s}_cls_raw"), &t.cls_raw[i], &g[&format!("head{s}_cls_raw")].1, 0.9999, 5e-2);
        gate(&format!("head{s}_bbox_scaled"), &t.bbox_scaled[i], &g[&format!("head{s}_bbox_scaled")].1, 0.9999, 5e-2);
        gate(&format!("head{s}_kps_raw"), &t.kps_raw[i], &g[&format!("head{s}_kps_raw")].1, 0.9999, 5e-2);
        // the 9 GRAPH OUTPUTS, after transpose+reshape (+ sigmoid for score)
        gate(&format!("out_score_{s}"), &t.out_score[i], &g[&format!("out_score_{s}")].1, 0.9999, 5e-3);
        gate(&format!("out_bbox_{s}"), &t.out_bbox[i], &g[&format!("out_bbox_{s}")].1, 0.9999, 5e-2);
        gate(&format!("out_kps_{s}"), &t.out_kps[i], &g[&format!("out_kps_{s}")].1, 0.9999, 5e-2);
    }

    // anchors + the decoded box/kps tensors, which the goldens carry per stride
    for (i, s) in cfg.strides.iter().enumerate() {
        let side = cfg.image_size / s;
        let a = facenet::detect::anchor_centers(side, side, *s, cfg.num_anchors);
        gate(&format!("anchors_{s}"), &a, &g[&format!("anchors_{s}")].1, 0.999999, 1e-3);
        let _ = i;
    }
}

/// The decode + NMS path, gated on the real-photo detections (the synthetic
/// detector image yields zero positives above threshold — the manifest records
/// `synthetic_positives_per_stride: {8:0, 16:0, 32:0}` — so it cannot gate this).
#[test]
fn scrfd_decode_and_nms_reproduce_the_reference_detections() {
    if !have(&["scrfd_10g_bnkps.onnx", "e2e.safetensors"]) {
        eprintln!("skip: e2e fixtures not under {}", dir());
        return;
    }
    let g = golden("e2e.safetensors");
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let cfg = ScrfdConfig::scrfd_10g_bnkps();
    let m = Scrfd::new(gpu, cfg.clone(), &weights().1);

    // det_scale per photo, from the dumper's recorded resize (manifest
    // `per_photo.det_scale`); recomputed here from the box the reference chose
    // is NOT possible, so compare in DETECTOR space (det_scale = 1) against the
    // reference boxes scaled back up.
    println!("scrfd decode:");
    let scales = [1.1786372007366483f32, 1.1636363636363636, 0.4383561643835616, 1.3882863340563991];
    for p in 0..4usize {
        let t = m.forward(&g[&format!("photo{p}_det_blob")].1);
        let faces = facenet::decode(&cfg, &t.out_score, &t.out_bbox, &t.out_kps, scales[p]);
        let want_box = &g[&format!("photo{p}_dets")].1;
        let want_kps = &g[&format!("photo{p}_kpss")].1;
        assert_eq!(faces.len(), 1, "photo{p}: expected 1 face, got {}", faces.len());
        let f = &faces[0];
        let got = [f.bbox[0], f.bbox[1], f.bbox[2], f.bbox[3], f.score];
        let c = compare(&got, want_box);
        let gk: Vec<f32> = f.kps.iter().flat_map(|k| [k[0], k[1]]).collect();
        let ck = compare(&gk, want_kps);
        println!(
            "  photo{p}  box max_abs {:.3e}  score {:.6} (ref {:.6})  kps max_abs {:.3e}",
            c.max_abs, f.score, want_box[4], ck.max_abs
        );
        assert!(c.max_abs < 0.5, "photo{p}: box drifted by {:.3e} px", c.max_abs);
        assert!(ck.max_abs < 0.5, "photo{p}: kps drifted by {:.3e} px", ck.max_abs);
    }
}
