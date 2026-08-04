// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Stage-by-stage forward parity against the dumped SAM 2 reference goldens.
//!
//! Fixtures live under `$BRAIN_TESTDATA` (default `<repo>/testdata`):
//!
//! ```text
//! testdata/sam2/hiera-{large,tiny}/
//!   sam2.1_hiera_{large,tiny}.pt      the reference checkpoint (hard-linked in)
//!   input.safetensors                 the normalized model input
//!   trunk.safetensors                 pos_embed / patch_embed / tapped blocks / stage feats
//!   neck.safetensors                  laterals / fpn / possine / high-res / image_embed
//!   case_*.safetensors                prompt + decoder taps, one file per prompt case
//! ```
//!
//! Each test SKIPS ITSELF when its fixture is absent (`make fetch/testdata` /
//! `tools/goldens/sam2_dump_reference.py` populate the tree).

use std::collections::HashMap;
use std::path::Path;

use checkpoint::safetensors::StTensor;
use sam2::{Sam2, Sam2Config};

use brain_testutil::testdata_path as testdata;

fn load(path: &Path) -> HashMap<String, StTensor> {
    checkpoint::safetensors::read(path.to_str().unwrap())
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect()
}

/// Cosine similarity and max absolute difference. Both are reported for every
/// stage: cosine alone hides a scale error, max_abs alone hides a shape error.
fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        mx = mx.max((x - y).abs());
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-30);
    (dot / denom, mx)
}

struct Report {
    rows: Vec<(String, f64, f64)>,
    floor: f64,
}

impl Report {
    fn new(floor: f64) -> Report {
        Report { rows: Vec::new(), floor }
    }
    fn check(&mut self, name: &str, got: &[f32], want: &StTensor) {
        let (c, m) = compare(got, &want.data);
        println!("  {name:<28} cos {c:.10}  max_abs {m:.3e}  n={}", want.data.len());
        self.rows.push((name.to_string(), c, m));
    }
    fn finish(self, label: &str) {
        let worst = self
            .rows
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("no stages compared");
        println!("  [{label}] {} stages, worst cosine {:.10} at {}", self.rows.len(), worst.1, worst.0);
        let bad: Vec<&(String, f64, f64)> = self.rows.iter().filter(|r| r.1 < self.floor).collect();
        assert!(bad.is_empty(), "{label}: {} stage(s) below cosine {}: {bad:?}", bad.len(), self.floor);
    }
}

fn run_model(dir: &str, cfg: Sam2Config, ckpt: &str) {
    let base = testdata(&format!("sam2/{dir}"));
    let pt = base.join(ckpt);
    if !base.join("trunk.safetensors").exists() || !pt.exists() {
        eprintln!("skip sam2/{dir}: fixture missing under {}", base.display());
        return;
    }
    println!("== sam2/{dir}");

    // ---- import (two-way coverage) ----
    let raw = checkpoint::torchpt::read(pt.to_str().unwrap()).expect("read .pt");
    let tensors: Vec<(String, Vec<usize>, Vec<f32>)> = raw
        .into_iter()
        // the archive root is `{"model": state_dict}`; torchpt flattens with '.'
        .filter_map(|t| t.name.strip_prefix("model.").map(|n| (n.to_string(), t.shape, t.data)))
        .collect();
    let (weights, rep) = sam2::import(tensors, &cfg).expect("import");
    println!(
        "  import: {} source, {} image-path, {} video-only skipped",
        rep.source, rep.imported, rep.skipped_video
    );

    let gpu = gpu_core::testgpu::dev(sam2::PIPELINES);
    let m = Sam2::new(gpu, cfg, &weights);

    // ---- image encoder ----
    let input = load(&base.join("input.safetensors"));
    let trunk = load(&base.join("trunk.safetensors"));
    let neck = load(&base.join("neck.safetensors"));
    // Rung 0: brain's own preprocessing must reproduce the reference's
    // normalized input, and the encoder then runs FROM that buffer.
    let img = m.preprocess(&input["image_rgb01"].data);
    let mut r = Report::new(0.9999);
    let rd = |b: &gpu_core::DeviceBuffer, n: usize| m.gpu.read(b, n);
    r.check("image_norm", &rd(&img, input["image_norm"].data.len()), &input["image_norm"]);
    let enc = m.encode(&img);
    r.check("pos_embed_interp", &rd(&enc.pos_embed, trunk["pos_embed_interp"].data.len()), &trunk["pos_embed_interp"]);
    r.check("patch_embed", &rd(&enc.patch_embed, trunk["patch_embed"].data.len()), &trunk["patch_embed"]);
    for (i, b) in enc.blocks.iter().enumerate() {
        let key = format!("blk{i:02}_out");
        if let Some(want) = trunk.get(&key) {
            r.check(&key, &rd(b, want.data.len()), want);
        }
    }
    for (i, t) in enc.trunk_feats.iter().enumerate() {
        let want = &trunk[&format!("trunk_feat{i}")];
        r.check(&format!("trunk_feat{i}"), &rd(t, want.data.len()), want);
    }
    for (i, t) in enc.lateral.iter().enumerate() {
        let want = &neck[&format!("lateral_level{i}")];
        r.check(&format!("lateral_level{i}"), &rd(t, want.data.len()), want);
    }
    for (i, t) in enc.fpn.iter().enumerate() {
        let want = &neck[&format!("fpn_level{i}")];
        r.check(&format!("fpn_level{i}"), &rd(t, want.data.len()), want);
    }
    for (i, t) in enc.pos_sine.iter().enumerate() {
        let want = &neck[&format!("possine_level{i}")];
        r.check(&format!("possine_level{i}"), &rd(t, want.data.len()), want);
    }
    for (i, t) in enc.high_res.iter().enumerate() {
        let want = &neck[&format!("high_res_feat{i}")];
        r.check(&format!("high_res_feat{i}"), &rd(t, want.data.len()), want);
    }
    r.check("image_embed", &rd(&enc.image_embed, neck["image_embed"].data.len()), &neck["image_embed"]);
    r.finish(&format!("sam2/{dir} encoder"));

    // ---- prompt encoder + mask decoder, one rung per case ----
    for case in ["point1", "point1_single", "point2_negpos", "box_bar", "point_and_mask"] {
        let path = base.join(format!("case_{case}.safetensors"));
        if !path.exists() {
            continue;
        }
        println!("-- case {case}");
        let g = load(&path);
        let coords: Vec<(f32, f32)> = g["point_coords"].data.chunks(2).map(|c| (c[0], c[1])).collect();
        let prompt = sam2::Prompt {
            coords,
            labels: g["point_labels"].data.clone(),
            mask_lowres: g.get("mask_input_lowres").map(|t| t.data.clone()),
            multimask_output: g["multimask_output"].data[0] > 0.5,
        };
        let dec = m.decode(&enc, &prompt);
        let mut r = Report::new(0.9999);
        r.check("sparse_embeddings", &rd(&dec.sparse, g["sparse_embeddings"].data.len()), &g["sparse_embeddings"]);
        r.check("dense_embeddings", &rd(&dec.dense, g["dense_embeddings"].data.len()), &g["dense_embeddings"]);
        r.check("dense_pe", &rd(&dec.dense_pe, g["dense_pe"].data.len()), &g["dense_pe"]);
        r.check("tokens", &rd(&dec.tokens, g["tokens"].data.len()), &g["tokens"]);
        r.check("src_in", &rd(&dec.src_in, g["src_in"].data.len()), &g["src_in"]);
        for (i, (q, k)) in dec.twoway.iter().enumerate() {
            let wq = &g[&format!("twoway{i}_queries")];
            let wk = &g[&format!("twoway{i}_keys")];
            r.check(&format!("twoway{i}_queries"), &rd(q, wq.data.len()), wq);
            r.check(&format!("twoway{i}_keys"), &rd(k, wk.data.len()), wk);
        }
        r.check("final_attn_out", &rd(&dec.final_attn_out, g["final_attn_out"].data.len()), &g["final_attn_out"]);
        r.check("hs", &rd(&dec.hs, g["hs"].data.len()), &g["hs"]);
        r.check("src_out", &rd(&dec.src_out, g["src_out"].data.len()), &g["src_out"]);
        r.check("dc1_out", &rd(&dec.dc1_out, g["dc1_out"].data.len()), &g["dc1_out"]);
        r.check("dc2_out", &rd(&dec.dc2_out, g["dc2_out"].data.len()), &g["dc2_out"]);
        r.check("upscaled_embedding", &rd(&dec.upscaled_embedding, g["upscaled_embedding"].data.len()), &g["upscaled_embedding"]);
        r.check("hyper_in", &rd(&dec.hyper_in, g["hyper_in"].data.len()), &g["hyper_in"]);
        r.check("masks_all4", &rd(&dec.masks_all, g["masks_all4"].data.len()), &g["masks_all4"]);
        r.check("iou_all4", &rd(&dec.iou_all, g["iou_all4"].data.len()), &g["iou_all4"]);
        r.check("object_score_logits", &rd(&dec.object_score_logits, 1), &g["object_score_logits"]);
        r.check("low_res_multimasks", &rd(&dec.low_res_multimasks, g["low_res_multimasks"].data.len()), &g["low_res_multimasks"]);
        r.check("high_res_multimasks", &rd(&dec.high_res_multimasks, g["high_res_multimasks"].data.len()), &g["high_res_multimasks"]);
        r.check("obj_ptr", &rd(&dec.obj_ptr, g["obj_ptr"].data.len()), &g["obj_ptr"]);
        assert_eq!(dec.best_iou_index as f32, g["best_iou_index"].data[0], "best mask index");
        r.finish(&format!("sam2/{dir} case {case}"));
    }
}

#[test]
fn hiera_tiny_forward_parity() {
    run_model("hiera-tiny", Sam2Config::hiera_tiny(), "sam2.1_hiera_tiny.pt");
}

#[test]
#[ignore = "hiera-large is a 48-block trunk at 1024x1024; run explicitly"]
fn hiera_large_forward_parity() {
    run_model("hiera-large", Sam2Config::hiera_large(), "sam2.1_hiera_large.pt");
}
