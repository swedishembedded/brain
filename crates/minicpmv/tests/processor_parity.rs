// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The image processor and the prompt layout against the reference
//! processor, exactly: resized views byte for byte, slice grids, patch grids,
//! normalized patches, and the token ids (placeholder counts and positions
//! included) for every processor case the golden records - the slicing
//! branches, 4x, slicing off, no image ids, several images - and for the
//! prompts of the model cases.

mod common;

use minicpmv::preprocess::{process_image, ImageOptions};
use minicpmv::prompt::{view_layout, Formatter, Message, Part, IMAGE_PAD};
use minicpmv::Downsample;

fn options(fx: &common::Fixture, opts: &serde_json::Value) -> ImageOptions {
    let mode = opts.get("downsample_mode").and_then(|v| v.as_str()).map(|m| Downsample::parse(m).unwrap()).unwrap_or(fx.cfg.downsample);
    let mut o = ImageOptions::from_dir(&fx.dir, fx.cfg.vision.patch, mode).unwrap();
    if let Some(n) = opts.get("max_slice_nums").and_then(|v| v.as_u64()) {
        o.max_slices = n as u32;
    }
    if let Some(b) = opts.get("slice_mode").and_then(|v| v.as_bool()) {
        o.slice_mode = b;
    }
    if let Some(b) = opts.get("use_image_id").and_then(|v| v.as_bool()) {
        o.use_image_id = b;
    }
    o
}

fn planar_u8(img: &imaging::Rgb8) -> Vec<f32> {
    let n = (img.w * img.h) as usize;
    (0..3).flat_map(|c| (0..n).map(move |i| (c, i))).map(|(c, i)| img.px[i * 3 + c] as f32).collect()
}

/// The reference's NaViT-packed pixels (`[3, p, n*p]`) as patch rows.
fn patches_from_navit(px: &[f32], n: usize, p: usize) -> Vec<f32> {
    let width = n * p;
    (0..n).flat_map(|i| (0..3).flat_map(move |c| (0..p).flat_map(move |ky| (0..p).map(move |kx| px[c * p * width + ky * width + i * p + kx])))).collect()
}

fn image(fx: &common::Fixture, name: &str) -> imaging::Rgb8 {
    imaging::codec::load(fx.golden.join(format!("inputs/{name}.png"))).expect("decode the golden input")
}

#[test]
fn images_and_prompts_match_the_reference_processor_exactly() {
    let Some(fx) = common::fixture() else { return };
    let manifest = fx.manifest();
    let fmt = Formatter::from_dir(&fx.dir).expect("tokenizer and template");
    let cases = manifest["processor"].as_object().expect("processor cases");
    assert!(cases.len() >= 14, "the golden records {} processor cases", cases.len());
    for (case, entry) in cases {
        let golden = fx.tensors(&format!("processor/{case}.safetensors"));
        let opts = options(&fx, &entry["options"]);
        let names: Vec<&str> = entry["images"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let processed: Vec<_> = names.iter().map(|n| process_image(&image(&fx, n), &opts)).collect();

        let resized: Vec<&imaging::Rgb8> = processed.iter().flat_map(|p| p.resized.iter()).collect();
        assert_eq!(resized.len() as u64, entry["n_views"].as_u64().unwrap(), "{case}: resize count");
        for (i, r) in resized.iter().enumerate() {
            assert_eq!(planar_u8(r), golden[&format!("resized.{i}")].1, "{case}: resized view {i} differs");
        }
        let sizes: Vec<f32> = processed.iter().flat_map(|p| p.views.iter().flat_map(|v| [v.h as f32, v.w as f32])).collect();
        assert_eq!(sizes, golden["target_sizes"].1, "{case}: target sizes");
        let grids: Vec<f32> = processed.iter().flat_map(|p| { let (r, c) = p.grid.unwrap_or((0, 0)); [r as f32, c as f32] }).collect();
        assert_eq!(grids, golden["grids"].1, "{case}: slice grids");
        if let Some((_, px)) = golden.get("pixel_values") {
            let n: u32 = processed.iter().flat_map(|p| p.views.iter()).map(|v| v.patches()).sum();
            let want = patches_from_navit(px, n as usize, opts.patch as usize);
            let got: Vec<f32> = processed.iter().flat_map(|p| p.patches.iter().copied()).collect();
            assert_eq!(got, want, "{case}: normalized patches");
        }

        let mut parts: Vec<Part> = names.iter().map(|_| Part::Image).collect();
        parts.push(Part::Text("Describe.".into()));
        let rendered = fmt.render(&[Message::user(parts)], true, false).unwrap();
        assert_eq!(rendered, entry["prompt"].as_str().unwrap(), "{case}: rendered template");
        let layouts: Vec<String> = processed.iter().map(|p| view_layout(p, opts.downsample, IMAGE_PAD)).collect();
        let prompt = fmt.expand(&rendered, &layouts, &[], opts.use_image_id).unwrap();
        assert_eq!(prompt.ids, common::ids(&golden["input_ids"].1), "{case}: token ids");
        let visual: u32 = prompt.runs.iter().map(|r| r.1).sum();
        let want: u32 = processed.iter().flat_map(|p| p.views.iter()).map(|&v| minicpmv::prompt::view_tokens(v, opts.downsample)).sum();
        assert_eq!(visual, want, "{case}: placeholder rows equal visual tokens");
    }
}

#[test]
fn model_case_prompts_match_the_reference_token_for_token() {
    let Some(fx) = common::fixture() else { return };
    let manifest = fx.manifest();
    let fmt = Formatter::from_dir(&fx.dir).expect("tokenizer and template");
    for case in ["text", "room_16x", "room_4x", "doc_16x", "two_images"] {
        let entry = &manifest["model"][case];
        let golden = fx.tensors(&format!("model/{case}.safetensors"));
        let opts = options(&fx, &entry["options"]);
        let names: Vec<&str> = entry["images"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let mut parts: Vec<Part> = names.iter().map(|_| Part::Image).collect();
        parts.push(Part::Text(entry["question"].as_str().unwrap().into()));
        let rendered = fmt.render(&[Message::user(parts)], true, false).unwrap();
        let layouts: Vec<String> = names.iter().map(|n| view_layout(&process_image(&image(&fx, n), &opts), opts.downsample, IMAGE_PAD)).collect();
        let prompt = fmt.expand(&rendered, &layouts, &[], opts.use_image_id).unwrap();
        assert_eq!(prompt.ids, common::ids(&golden["input_ids"].1), "{case}: token ids");
    }
}
