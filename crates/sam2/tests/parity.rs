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
//!   manifest.json                     the reference's own per-block trunk table
//! ```
//!
//! Each test SKIPS ITSELF when its fixture is absent, through
//! [`brain_testutil::skip`] - so `BRAIN_REQUIRE_FIXTURES=1` turns the skip into
//! a failure and a green run under that flag means every comparison below
//! really ran. `tools/goldens/sam2_dump_reference.py` populates the tree from
//! the released checkpoint; `make fetch/testdata` hard-links the checkpoint in.

use std::path::Path;

use sam2::{Sam2, Sam2Config};

use brain_testutil::parity::{load, Report};
use brain_testutil::testdata_path as testdata;

/// Cosine floor every tap is held to. The reference dump is fp32 on CPU and
/// brain's forward is fp32 too, so anything below this is a real disagreement,
/// not accumulated rounding.
const FLOOR: f64 = 0.9999;

/// The DERIVED per-block Hiera table against the reference's own dumped one.
///
/// `Sam2Config::blocks` computes the `dim_mul`/`head_mul` schedule and the
/// "window size lags by a block" rule rather than transcribing them, precisely
/// because a hand-typed table is where that goes wrong - so the table is worth
/// nothing unless something checks it against the reference, and `manifest.json`
/// carries the reference's own version of exactly these fields.
fn check_block_table(base: &Path, cfg: &Sam2Config) {
    let Ok(raw) = std::fs::read(base.join("manifest.json")) else {
        return brain_testutil::skip(&format!("{}/manifest.json absent", base.display()));
    };
    let m: serde_json::Value = serde_json::from_slice(&raw).expect("manifest.json is not JSON");
    let u32s = |v: &serde_json::Value| -> Vec<u32> {
        v.as_array().expect("array").iter().map(|x| x.as_u64().expect("integer") as u32).collect()
    };
    let p = &m["params"];
    assert_eq!(cfg.image_size, p["image_size"].as_u64().unwrap() as u32, "image_size");
    assert_eq!(cfg.stage_ends(), u32s(&p["trunk_stage_ends"]), "stage ends");
    assert_eq!(cfg.q_pool_blocks(), u32s(&p["trunk_q_pool_blocks"]), "q_pool blocks");
    assert_eq!(cfg.global_att_blocks, u32s(&p["trunk_global_att_blocks"]), "global-attention blocks");
    assert_eq!(cfg.window_spec, u32s(&p["trunk_window_spec"]), "window spec");
    assert_eq!(cfg.trunk_channel_list(), u32s(&p["trunk_channel_list"]), "trunk channel list");

    let want = m["trunk_blocks"].as_array().expect("manifest.trunk_blocks");
    let got = cfg.blocks();
    assert_eq!(got.len(), want.len(), "block count");
    for (b, w) in got.iter().zip(want) {
        let n = |k: &str| w[k].as_u64().unwrap_or_else(|| panic!("block {} has no {k}", b.index)) as u32;
        let hw = |k: &str| {
            let a = u32s(&w[k]);
            (a[0], a[1])
        };
        let (i, want_i) = (b.index, n("index"));
        assert_eq!((i, b.dim, b.dim_out, b.num_heads, b.window_size),
                   (want_i, n("dim"), n("dim_out"), n("num_heads"), n("window_size")),
                   "block {i}: (index, dim, dim_out, heads, window)");
        assert_eq!(b.q_pool, w["q_pool"].as_bool().expect("q_pool"), "block {i} q_pool");
        assert_eq!((b.in_hw, b.out_hw), (hw("in_hw"), hw("out_hw")), "block {i} token grid");
        assert_eq!(b.needs_pad(), w["window_pad"].as_bool().expect("window_pad"), "block {i} window padding");
    }
    println!("  block table: {} blocks match the reference manifest", got.len());
}

/// The released `.pt`, from the goldens directory if `make fetch/testdata`
/// hard-linked one in, else straight out of the model store where `brain fetch
/// facebook/sam2.1-<variant>` leaves it. The second is what actually holds it on
/// a box that has the checkpoint but no `sam2/` testdata mirror, and the import
/// half of this test needs the raw archive, not the converted weights.
fn checkpoint_path(base: &Path, dir: &str, ckpt: &str) -> Option<std::path::PathBuf> {
    let local = base.join(ckpt);
    if local.exists() {
        return Some(local);
    }
    let store = std::path::PathBuf::from(brain_testutil::model_dir(&format!("facebook/sam2.1-{dir}"))?).join(ckpt);
    store.exists().then_some(store)
}

fn run_model(dir: &str, cfg: Sam2Config, ckpt: &str) {
    let base = testdata(&format!("sam2/{dir}"));
    let Some(pt) = checkpoint_path(&base, dir, ckpt).filter(|_| base.join("trunk.safetensors").exists()) else {
        return brain_testutil::skip(&format!(
            "sam2/{dir}: need {ckpt} (in {} or the model store as facebook/sam2.1-{dir}) AND the \
             stage goldens next to it - regenerate with tools/goldens/sam2_dump_reference.py",
            base.display()
        ));
    };
    println!("== sam2/{dir}");
    check_block_table(&base, &cfg);

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
    let mut r = Report::new(FLOOR);
    let rd = |b: &gpu_core::DeviceBuffer, n: usize| m.gpu.read(b, n);
    r.against("image_norm", &rd(&img, input["image_norm"].data.len()), &input);
    let enc = m.encode(&img);
    r.against("pos_embed_interp", &rd(&enc.pos_embed, trunk["pos_embed_interp"].data.len()), &trunk);
    r.against("patch_embed", &rd(&enc.patch_embed, trunk["patch_embed"].data.len()), &trunk);
    for (i, b) in enc.blocks.iter().enumerate() {
        let key = format!("blk{i:02}_out");
        if let Some(want) = trunk.get(&key) {
            r.check(&key, &rd(b, want.data.len()), &want.data);
        }
    }
    for (i, t) in enc.trunk_feats.iter().enumerate() {
        let want = &trunk[&format!("trunk_feat{i}")];
        r.check(&format!("trunk_feat{i}"), &rd(t, want.data.len()), &want.data);
    }
    for (i, t) in enc.lateral.iter().enumerate() {
        let want = &neck[&format!("lateral_level{i}")];
        r.check(&format!("lateral_level{i}"), &rd(t, want.data.len()), &want.data);
    }
    for (i, t) in enc.fpn.iter().enumerate() {
        let want = &neck[&format!("fpn_level{i}")];
        r.check(&format!("fpn_level{i}"), &rd(t, want.data.len()), &want.data);
    }
    for (i, t) in enc.pos_sine.iter().enumerate() {
        let want = &neck[&format!("possine_level{i}")];
        r.check(&format!("possine_level{i}"), &rd(t, want.data.len()), &want.data);
    }
    for (i, t) in enc.high_res.iter().enumerate() {
        let want = &neck[&format!("high_res_feat{i}")];
        r.check(&format!("high_res_feat{i}"), &rd(t, want.data.len()), &want.data);
    }
    r.against("image_embed", &rd(&enc.image_embed, neck["image_embed"].data.len()), &neck);
    r.finish(&format!("sam2/{dir} encoder"));

    // ---- prompt encoder + mask decoder, one rung per case ----
    for case in ["point1", "point1_single", "point2_negpos", "box_bar", "point_and_mask"] {
        let path = base.join(format!("case_{case}.safetensors"));
        if !path.exists() {
            // Also a fixture, so also a skip that `BRAIN_REQUIRE_FIXTURES=1`
            // refuses: an encoder-only run must not read as full parity.
            brain_testutil::skip(&format!("sam2/{dir} case {case}: {} absent", path.display()));
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
        let mut r = Report::new(FLOOR);
        r.against("sparse_embeddings", &rd(&dec.sparse, g["sparse_embeddings"].data.len()), &g);
        r.against("dense_embeddings", &rd(&dec.dense, g["dense_embeddings"].data.len()), &g);
        r.against("dense_pe", &rd(&dec.dense_pe, g["dense_pe"].data.len()), &g);
        r.against("tokens", &rd(&dec.tokens, g["tokens"].data.len()), &g);
        r.against("src_in", &rd(&dec.src_in, g["src_in"].data.len()), &g);
        for (i, (q, k)) in dec.twoway.iter().enumerate() {
            let wq = &g[&format!("twoway{i}_queries")];
            let wk = &g[&format!("twoway{i}_keys")];
            r.check(&format!("twoway{i}_queries"), &rd(q, wq.data.len()), &wq.data);
            r.check(&format!("twoway{i}_keys"), &rd(k, wk.data.len()), &wk.data);
        }
        r.against("final_attn_out", &rd(&dec.final_attn_out, g["final_attn_out"].data.len()), &g);
        r.against("hs", &rd(&dec.hs, g["hs"].data.len()), &g);
        r.against("src_out", &rd(&dec.src_out, g["src_out"].data.len()), &g);
        r.against("dc1_out", &rd(&dec.dc1_out, g["dc1_out"].data.len()), &g);
        r.against("dc2_out", &rd(&dec.dc2_out, g["dc2_out"].data.len()), &g);
        r.against("upscaled_embedding", &rd(&dec.upscaled_embedding, g["upscaled_embedding"].data.len()), &g);
        r.against("hyper_in", &rd(&dec.hyper_in, g["hyper_in"].data.len()), &g);
        r.against("masks_all4", &rd(&dec.masks_all, g["masks_all4"].data.len()), &g);
        r.against("iou_all4", &rd(&dec.iou_all, g["iou_all4"].data.len()), &g);
        r.against("object_score_logits", &rd(&dec.object_score_logits, 1), &g);
        r.against("low_res_multimasks", &rd(&dec.low_res_multimasks, g["low_res_multimasks"].data.len()), &g);
        r.against("high_res_multimasks", &rd(&dec.high_res_multimasks, g["high_res_multimasks"].data.len()), &g);
        r.against("obj_ptr", &rd(&dec.obj_ptr, g["obj_ptr"].data.len()), &g);
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
