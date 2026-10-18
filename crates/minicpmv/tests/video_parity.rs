// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Video against the reference: the frame indices `sample_frames` picks for
//! every recorded clip metadata (short clips, long ones thinned to the frame
//! budget, odd frame rates), and the golden clip end to end - token ids,
//! video features, next-token logits and the greedy continuation.

mod common;

use minicpmv::prompt::{Message, Part};
use minicpmv::video::{sample_frames, ClipMeta};
use minicpmv::{GenOptions, MiniCpmV};
use model::serve::PagedDecoder;

#[test]
fn frame_indices_match_the_reference_for_every_recorded_clip() {
    let Some(fx) = common::fixture() else { return };
    let rows: serde_json::Value = serde_json::from_slice(&std::fs::read(fx.golden.join("video/sample_frames.json")).unwrap()).unwrap();
    let rows = rows.as_array().unwrap();
    assert!(rows.len() >= 8);
    for r in rows {
        let meta = ClipMeta { frames: r["total_num_frames"].as_u64().unwrap(), fps: r["fps"].as_f64().unwrap(), duration: r["duration"].as_f64().unwrap() };
        let (max, stack) = (r["max_num_frames"].as_u64().unwrap() as u32, r["stack_frames"].as_u64().unwrap() as u32);
        let got = sample_frames(meta, max, stack);
        if stack > 1 {
            assert!(got.is_err(), "{meta:?}: stack_frames {stack} must be refused");
            continue;
        }
        let want: Vec<u64> = r["indices"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert_eq!(got.unwrap(), want, "{meta:?} max {max}");
    }
}

#[test]
fn the_golden_clip_matches_the_reference_end_to_end() {
    let Some(fx) = common::fixture() else { return };
    let manifest = fx.manifest();
    let entry = &manifest["model"]["video_16x"];
    let golden = fx.tensors("model/video_16x.safetensors");
    let clip = fx.tensors("video/clip.safetensors");
    let (shape, data) = &clip["frames"];
    let (t, h, w) = (shape[0], shape[1] as u32, shape[2] as u32);
    let frames: Vec<imaging::Rgb8> = data.chunks_exact(data.len() / t).map(|f| imaging::Rgb8 { w, h, px: f.iter().map(|&v| v as u8).collect() }).collect();
    let md = &entry["video_meta"];
    let meta = ClipMeta { frames: md["total_num_frames"].as_u64().unwrap(), fps: md["fps"].as_f64().unwrap(), duration: md["duration"].as_f64().unwrap() };
    let picked: Vec<imaging::Rgb8> = sample_frames(meta, minicpmv::video::MAX_NUM_FRAMES, 1).unwrap().into_iter().map(|i| frames[i as usize].clone()).collect();

    let gpu = gpu_core::testgpu::dev(qwen35::model::pipelines());
    let mut m = MiniCpmV::load(&fx.dir, &gpu, 2048, 1, model::kv_tier::KvTier::F32).expect("load MiniCPM-V");
    let want = common::ids(&golden["greedy"].1);
    let opts = GenOptions { max_new: want.len(), ..GenOptions::default() };
    let rendered = m.fmt.render(&[Message::user(vec![Part::Video, Part::Text(entry["question"].as_str().unwrap().into())])], true, false).unwrap();
    assert_eq!(rendered, entry["prompt"].as_str().unwrap(), "rendered template");
    let prepared = m.prepare_text(&rendered, &[], std::slice::from_ref(&picked), &opts).expect("prepare");
    assert_eq!(prepared.prompt.ids, common::ids(&golden["input_ids"].1), "token ids");

    let mut table = brain_testutil::parity::Table::new(0.9999, 2e-3);
    // The video features, as the decoder receives them.
    let d = m.cfg.text.d_model as usize;
    let rows: Vec<f32> = prepared.prompt.runs.iter().flat_map(|&(r0, n)| prepared.embeds[r0 as usize * d..(r0 + n) as usize * d].to_vec()).collect();
    table.check("video features", &rows, &golden["merger"].1);
    let mut tbl = model::paged::BlockTable::new();
    let hidden = m.engine.prefill_embeds(&mut tbl, &prepared.embeds);
    let logits = m.engine.logits(&hidden);
    m.engine.release_table(&mut tbl);
    let v = logits.len();
    let rows_l = golden["logits_last"].0[0];
    table.check("logits", &logits, &golden["logits_last"].1[(rows_l - 1) * v..]);
    table.print();
    assert!(table.failures.is_empty(), "{:#?}", table.failures);

    let got = m.generate_prepared(&prepared, &opts, &mut |_| true).expect("generate");
    let want_text: Vec<u32> = want.iter().copied().take_while(|t| !m.eos.contains(t)).collect();
    assert_eq!(got, want_text, "greedy continuation\n got: {:?}\nwant: {:?}", m.decode(&got), m.decode(&want_text));
}
