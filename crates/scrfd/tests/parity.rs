// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Stage-by-stage forward parity against the insightface goldens dumped by
//! `tools/goldens/scrfd_dump_reference.py`.
//!
//! `e2e.safetensors` - the real-photo golden the decode/NMS gate replays - comes
//! from the *pipeline* dumper instead (`arcface_dump_reference.py --photos`),
//! because an end-to-end run is detect THEN embed and needs both released
//! graphs. Everything else here is the detector alone.
//!
//! Every test SKIPS ITSELF when its fixture is absent (`AGENTS.md`): the
//! goldens and the `.onnx` file live under `$BRAIN_TESTDATA`
//! (default `<repo>/testdata`), which is gitignored.
//!
//! ```bash
//! cargo test --release -p brain-scrfd --test parity -- --nocapture
//! BRAIN_DEVICE=cpu   # same numbers, no GPU
//! ```
//!
//! Reported per stage: **cosine** and **max_abs**. Cosine alone cannot see a
//! scale error, and max_abs alone cannot see a rotation; a stage passes only on
//! both.

use std::collections::HashMap;
use std::sync::OnceLock;

use model::hostmath::cosine;
use scrfd::config::ScrfdConfig;
use scrfd::model::{Scrfd, PIPELINES};

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

/// The imported checkpoint, decoded once for the whole test binary.
fn weights() -> &'static scrfd::Tensors {
    static W: OnceLock<scrfd::Tensors> = OnceLock::new();
    W.get_or_init(|| scrfd::import_dir(std::path::Path::new(&dir())).expect("import scrfd_10g_bnkps"))
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
fn import_covers_the_graph_in_both_directions() {
    if !have(&["scrfd_10g_bnkps.onnx"]) {
        brain_testutil::skip(&format!("scrfd_10g_bnkps.onnx not under {}", dir()));
        return;
    }
    let scr = weights();
    assert_eq!(scr.len(), ScrfdConfig::scrfd_10g_bnkps().tensor_manifest().len());
    assert_eq!(scr.len(), 119);
    // spot-check that a folded conv really carries a bias
    assert_eq!(scr["head.stride8.cls.weight"].0, vec![2, 80, 3, 3]);
    assert_eq!(scr["head.stride8.cls.bias"].0, vec![2]);
    assert_eq!(scr["backbone.stem.0.weight"].0, vec![28, 3, 3, 3]);
}

/// The walk binds POSITIONALLY, so a config that disagrees with the graph must
/// fail loudly rather than bind 57 of 58 convolutions and zero-fill the rest.
#[test]
fn a_config_that_disagrees_with_the_graph_errors_instead_of_zero_filling() {
    if !have(&["scrfd_10g_bnkps.onnx"]) {
        brain_testutil::skip(&format!("scrfd_10g_bnkps.onnx not under {}", dir()));
        return;
    }
    let m = onnx::read_file(format!("{}/scrfd_10g_bnkps.onnx", dir())).unwrap();
    let g = onnx::read::graph(&m).unwrap();
    let cfg = ScrfdConfig { layers: [3, 4, 2, 2], ..ScrfdConfig::scrfd_10g_bnkps() };
    let err = scrfd::import_scrfd(g, &cfg).unwrap_err();
    assert!(err.contains("Conv nodes"), "{err}");
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

#[test]
fn scrfd_forward_parity_stage_by_stage() {
    if !have(&["scrfd_10g_bnkps.onnx", "scrfd.safetensors"]) {
        brain_testutil::skip(&format!("scrfd fixtures not under {}", dir()));
        return;
    }
    let g = golden("scrfd.safetensors");
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let cfg = ScrfdConfig::scrfd_10g_bnkps();

    let bgr: Vec<u8> = g["image_bgr_u8"].1.iter().map(|&v| v as u8).collect();
    let blob = scrfd::blob_from_bgr_u8(&gpu, &bgr, 640, 640, &cfg.pre);
    println!("scrfd:");
    gate("blob", &blob, &g["blob"].1, 0.999999, 1e-5);

    let m = Scrfd::new(gpu, cfg.clone(), weights());
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

    // anchors, which the goldens carry per stride
    for s in cfg.strides.iter() {
        let side = cfg.image_size / s;
        let a = scrfd::detect::anchor_centers(side, side, *s, cfg.num_anchors);
        gate(&format!("anchors_{s}"), &a, &g[&format!("anchors_{s}")].1, 0.999999, 1e-3);
    }
}

/// The decode + NMS path, gated on the real-photo detections (the synthetic
/// detector image yields zero positives above threshold - the manifest records
/// `synthetic_positives_per_stride: {8:0, 16:0, 32:0}` - so it cannot gate this).
#[test]
fn scrfd_decode_and_nms_reproduce_the_reference_detections() {
    if !have(&["scrfd_10g_bnkps.onnx", "e2e.safetensors"]) {
        brain_testutil::skip(&format!("e2e fixtures not under {}", dir()));
        return;
    }
    let g = golden("e2e.safetensors");
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let cfg = ScrfdConfig::scrfd_10g_bnkps();
    let m = Scrfd::new(gpu, cfg.clone(), weights());

    // det_scale per photo, from the dumper's recorded resize (manifest
    // `per_photo.det_scale`); recomputing it here from the box the reference
    // chose is NOT possible, so replay the recorded value.
    println!("scrfd decode:");
    // The dumper recorded these as f64; kept at full precision so the literal is
    // a verbatim copy of `per_photo.det_scale` rather than a re-rounded one.
    #[allow(clippy::excessive_precision)]
    let scales = [1.1786372007366483f32, 1.1636363636363636, 0.4383561643835616, 1.3882863340563991];
    for p in 0..4usize {
        let t = m.forward(&g[&format!("photo{p}_det_blob")].1);
        let faces = scrfd::decode(&cfg, &t.out_score, &t.out_bbox, &t.out_kps, scales[p]);
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
