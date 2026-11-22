// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the vision tower reproduces the reference tower tap by tap (embeddings with class tokens and the resampled
//! position table, `ln_pre`, every layer, the merger input and its output), on a small random tower in the published
//! naming and on the published checkpoint; images do not influence each other; and the features land on the prompt's
//! placeholder positions exactly as the reference processor and `get_placeholder_mask` place them.
//!
//! Goldens: `tools/goldens/inkling2_vision_dump.py` (`tiny` and `real`; `preproc` for the PNG inputs).

use brain_testutil::golden::Source;
use brain_testutil::parity::{load, Table};
use checkpoint::packed::{CodecRegistry, PackedReader, PackedSource};
use imaging::RegionImage;
use inkling2::config::VisionConfig;
use inkling2::vision::{expand_placeholders, pipelines, splice_embeds, ImageProcessorSpec, VisionTower, VisionWeights, IMAGE_TOKEN_ID, MAX_SIDE};

const DUMPER: &str = "tools/goldens/inkling2_vision_dump.py";
const REPO: &str = "thinkingmachines/Inkling-2-Mini";
const PROCESSOR_CONFIG: &str = r#"{"image_processor":{"do_convert_rgb":true,"do_normalize":true,"do_rescale":true,"image_mean":[0.5,0.5,0.5],
  "image_processor_type":"Inkling2ImageProcessor","image_std":[0.5,0.5,0.5],"rescale_factor":0.00392156862745098,
  "size":{"height":40,"width":40}},"image_token":"<|unused_200054|>","processor_class":"Inkling2Processor"}"#;

fn vision_config(extra: serde_json::Value) -> VisionConfig {
    let mut v = serde_json::json!({
        "model_type": "inkling2_vision", "text_hidden_size": 2304, "temporal_patch_size": 2, "native_patch_size": 20,
        "adapter_merge_size": 2, "projection_hidden_dim": 4096, "projector_bias": false, "hidden_size": 1024,
        "num_hidden_layers": 23, "num_attention_heads": 16, "mlp_ratio": 4.0, "posemb_grid_size": 32, "layer_norm_eps": 1e-5
    });
    v.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    serde_json::from_value(v).unwrap()
}

/// The golden's `RegionImage`s from its `pixel_values` and `image_grid_thw` (the reference processor's own output, so
/// the tower is tested apart from the preprocessor).
fn golden_images(golden: &std::collections::HashMap<String, checkpoint::safetensors::StTensor>) -> Vec<RegionImage> {
    let cell = 40 * 40 * 3;
    let mut at = 0;
    golden["image_grid_thw"]
        .data
        .chunks(3)
        .map(|g| {
            let (rows, cols) = (g[1] as u32, g[2] as u32);
            let n = (rows * cols) as usize * cell;
            let img = RegionImage { rows, cols, pixels: golden["pixel_values"].data[at..at + n].to_vec() };
            at += n;
            img
        })
        .collect()
}

fn check_taps(table: &mut Table, golden: &std::collections::HashMap<String, checkpoint::safetensors::StTensor>, trace: &inkling2::vision::VisionTrace, features: &[f32]) {
    table.check("embeddings", &trace.embeddings, &golden["embeddings"].data);
    table.check("pre_layernorm", &trace.pre_layernorm, &golden["pre_layernorm"].data);
    for (l, rows) in &trace.layers {
        table.check(&format!("layer.{l}"), rows, &golden[&format!("layer.{l}")].data);
    }
    table.check("last_hidden", &trace.last_hidden, &golden["last_hidden"].data);
    table.check("merger_in", &trace.merger_in, &golden["merger_in"].data);
    table.check("features", features, &golden["features"].data);
}

struct Tiny {
    cfg: VisionConfig,
    src: PackedSource,
    golden: std::collections::HashMap<String, checkpoint::safetensors::StTensor>,
}

fn tiny() -> Option<Tiny> {
    let dir = brain_testutil::testdata_path("golden/inkling2/vision/tiny");
    let source = Source::open(&dir, &format!("{DUMPER} tiny"))?;
    if !source.require(&[("hidden_size", 64), ("num_hidden_layers", 3), ("num_attention_heads", 4), ("posemb_grid_size", 8), ("text_hidden_size", 48)]) {
        return None;
    }
    let cfg = vision_config(serde_json::json!({
        "text_hidden_size": 48, "projection_hidden_dim": 96, "hidden_size": 64, "num_hidden_layers": 3, "num_attention_heads": 4, "posemb_grid_size": 8
    }));
    let reader = PackedReader::open(&dir.join("model.safetensors")).expect("the tiny tower checkpoint opens");
    let src = PackedSource::identity(Box::new(reader), CodecRegistry::reference()).expect("the tiny tower checkpoint is readable");
    Some(Tiny { cfg, src, golden: load(&dir.join("golden.safetensors")) })
}

#[test]
fn the_tiny_tower_reproduces_every_reference_tap() {
    let Some(t) = tiny() else { return };
    let g = gpu_core::testgpu::dev(pipelines());
    let weights = VisionWeights::load(&t.src, &t.cfg).unwrap();
    let tower = VisionTower::new(&g, &t.cfg, &weights).unwrap();
    let images = golden_images(&t.golden);
    let (features, trace) = tower.encode_traced(&g, &images, &[0, 1, 2]).unwrap();
    assert_eq!(features.tokens_per_image(), [4, 6, 1]);
    let mut table = Table::new(0.999_999, 2e-5);
    check_taps(&mut table, &t.golden, &trace, features.rows());
    table.print();
    assert!(table.has("layer.2") && table.has("features"));
    table.assert_clean();

    // Attention is block-diagonal per image: an image alone gives the rows it has inside the batch.
    for (i, img) in images.iter().enumerate() {
        let alone = tower.encode(&g, std::slice::from_ref(img)).unwrap();
        let worst = alone.rows().iter().zip(features.image(i)).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-5, "image {i} differs by {worst} between a batch and alone");
    }
}

#[test]
fn a_tower_refuses_a_wrong_checkpoint_and_a_wrong_image_by_name() {
    let Some(t) = tiny() else { return };
    // A config that makes the shapes differ is refused naming the tensor.
    let wide = vision_config(serde_json::json!({"text_hidden_size": 48, "projection_hidden_dim": 96, "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4, "posemb_grid_size": 8}));
    let e = VisionWeights::load(&t.src, &wide).err().expect("a fourth layer is not in the file");
    assert!(e.contains("layers.3"), "{e}");
    let narrow = vision_config(serde_json::json!({"text_hidden_size": 48, "projection_hidden_dim": 96, "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "posemb_grid_size": 8}));
    let e = VisionWeights::load(&t.src, &narrow).err().expect("a third layer is not in the config");
    assert!(e.contains("layers.2"), "{e}");
    // An image whose pixels do not fill its grid is refused.
    let g = gpu_core::testgpu::dev(pipelines());
    let tower = VisionTower::new(&g, &t.cfg, &VisionWeights::load(&t.src, &t.cfg).unwrap()).unwrap();
    let e = tower.encode(&g, &[RegionImage { rows: 1, cols: 1, pixels: vec![0.0; 10] }]).expect_err("short pixels");
    assert!(e.contains("image 0"), "{e}");
}

#[test]
fn the_published_tower_reproduces_the_reference_on_two_images() {
    let dir = brain_testutil::testdata_path("golden/inkling2/vision/real");
    let Some(source) = Source::open(&dir, &format!("{DUMPER} real {REPO}")) else { return };
    if !source.require(&[("hidden_size", 1024), ("num_hidden_layers", 23), ("num_attention_heads", 16), ("posemb_grid_size", 32), ("text_hidden_size", 2304)]) {
        return;
    }
    let Some(model) = brain_testutil::model_dir(REPO).map(std::path::PathBuf::from).filter(|d| d.join("hf_quant_config.json").is_file()) else {
        brain_testutil::skip_unavailable(&format!("{REPO} is not in the model store"));
        return;
    };
    let cfg = inkling2::Inkling2Config::from_dir(&model).unwrap().vision.expect("the release has a vision tower");
    let reader = PackedReader::open(&model).unwrap();
    let src = PackedSource::identity_lenient(Box::new(reader), CodecRegistry::reference());
    let weights = VisionWeights::load(&src, &cfg).unwrap();
    let g = gpu_core::testgpu::dev(pipelines());
    let tower = VisionTower::new(&g, &cfg, &weights).unwrap();
    let golden = load(&dir.join("golden.safetensors"));
    let (features, trace) = tower.encode_traced(&g, &golden_images(&golden), &[0, 11, 22]).unwrap();
    let mut table = Table::new(0.9999, 1e-3);
    check_taps(&mut table, &golden, &trace, features.rows());
    table.print();
    table.assert_clean();
}

#[test]
fn the_processor_spec_is_held_to_the_recipe_the_tower_was_trained_with() {
    let spec = ImageProcessorSpec::from_json(PROCESSOR_CONFIG).unwrap();
    let tiling = spec.tiling(&vision_config(serde_json::json!({}))).unwrap();
    assert_eq!((tiling.region, tiling.max_side, spec.image_token.as_str()), (40, MAX_SIDE, "<|unused_200054|>"));
    // A region that is not patch size times merge size would silently change what a token means.
    let e = spec.tiling(&vision_config(serde_json::json!({"native_patch_size": 14}))).unwrap_err();
    assert!(e.contains("region 40"), "{e}");
    for (what, from, to) in [
        ("a disabled normalisation", "\"do_normalize\":true", "\"do_normalize\":false"),
        ("another processor", "Inkling2ImageProcessor", "OtherProcessor"),
        ("a non-square region", "\"width\":40", "\"width\":41"),
        ("a zero std", "\"image_std\":[0.5,0.5,0.5]", "\"image_std\":[0.5,0.0,0.5]"),
        ("an unknown field", "\"do_rescale\":true,", "\"do_rescale\":true,\"resample\":3,"),
    ] {
        assert!(ImageProcessorSpec::from_json(&PROCESSOR_CONFIG.replace(from, to)).is_err(), "{what} was accepted");
    }
}

#[test]
fn features_land_on_the_placeholder_positions_in_image_order() {
    let ph = IMAGE_TOKEN_ID;
    // The chat template renders one placeholder per image; the processor makes it one per region.
    let prompt = [1, 2, ph, 3, ph, 4];
    let expanded = expand_placeholders(&prompt, ph, &[3, 2]).unwrap();
    assert_eq!(expanded, [1, 2, ph, ph, ph, 3, ph, ph, 4]);
    // Rows 0..3 are image 0's, rows 3..5 image 1's, whatever the width.
    let rows: Vec<f32> = (0..5).flat_map(|r| [r as f32, 10.0 + r as f32]).collect();
    let feats = inkling2::vision::ImageFeatures::new(2, vec![3, 2], rows.clone()).unwrap();
    let embeds = splice_embeds(&expanded, ph, &feats).unwrap();
    assert_eq!(embeds.positions(), [2, 3, 4, 6, 7]);
    assert_eq!(embeds.rows(), rows.as_slice());
    assert_eq!(embeds.row_at(6).unwrap(), [3.0, 13.0]);
    // Wrong counts are refused naming both numbers.
    assert!(expand_placeholders(&prompt, ph, &[3]).unwrap_err().contains("2 image placeholder"));
    assert!(expand_placeholders(&prompt, ph, &[3, 0]).is_err());
    let e = splice_embeds(&[1, ph, ph, 2], ph, &feats).unwrap_err();
    assert!(e.contains("2 placeholder") && e.contains("5 feature row"), "{e}");
}

#[test]
fn encoded_image_files_become_prompt_embeddings_through_the_whole_chain() {
    let Some(t) = tiny() else { return };
    let pre = brain_testutil::testdata_path("golden/inkling2/vision/preproc");
    let Some(reference) = Some(pre.join("golden.safetensors")).filter(|p| p.exists()).map(|p| load(&p)) else {
        brain_testutil::skip(&format!("{}: run {DUMPER} preproc", pre.display()));
        return;
    };
    let g = gpu_core::testgpu::dev(pipelines());
    let tower = VisionTower::new(&g, &t.cfg, &VisionWeights::load(&t.src, &t.cfg).unwrap()).unwrap();
    let tiling = ImageProcessorSpec::from_json(PROCESSOR_CONFIG).unwrap().tiling(&t.cfg).unwrap();
    let names = ["rgb_odd", "rgba_1x1"];
    let files: Vec<Vec<u8>> = names.iter().map(|n| std::fs::read(pre.join(format!("{n}.png"))).unwrap()).collect();
    let files: Vec<&[u8]> = files.iter().map(Vec::as_slice).collect();
    let ph = IMAGE_TOKEN_ID;
    let (expanded, embeds) = tower.embed_prompt(&g, &tiling, &[5, ph, 6, ph, 7], ph, &files).unwrap();

    // rgb_odd is 83x61 (a 2x3 grid of regions), rgba_1x1 one region: 7 placeholders in all.
    let grids: Vec<[f32; 3]> = names.iter().map(|n| reference[&format!("{n}.grid")].data.clone().try_into().unwrap()).collect();
    let tokens: Vec<usize> = grids.iter().map(|g| (g[1] * g[2]) as usize).collect();
    assert_eq!(tokens, [6, 1]);
    assert_eq!(expanded.len(), 3 + 7);
    assert_eq!(embeds.positions(), [1, 2, 3, 4, 5, 6, 8]);
    assert_eq!(&expanded[..1], [5]);
    assert_eq!(&expanded[7..], [6, ph, 7]);
    // The rows are the tower's output for the reference processor's own pixels.
    let images: Vec<RegionImage> = names
        .iter()
        .zip(&grids)
        .map(|(n, g)| RegionImage { rows: g[1] as u32, cols: g[2] as u32, pixels: reference[&format!("{n}.pixels")].data.clone() })
        .collect();
    let want = tower.encode(&g, &images).unwrap();
    assert_eq!(embeds.rows(), want.rows(), "the file chain and the reference pixels give different features");
    assert_eq!(embeds.width(), 48);
    // An image that is not a supported file is refused naming its index.
    let e = tower.embed_prompt(&g, &tiling, &[ph], ph, &[b"nope".as_slice()]).unwrap_err();
    assert!(e.contains("image 0"), "{e}");
}
