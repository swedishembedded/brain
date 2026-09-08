// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-OCR-2's mmproj, against the real checkpoint.
//!
//! Mirrors `crates/gguf/tests/deepseek_ocr.rs`'s vision-side proof: a config
//! derived purely from the file must reproduce the real tower, and the
//! tensor map must cover the file in **both** directions. The LM half needs
//! no sibling test here - its `general.architecture` and every tensor name
//! and shape are byte-identical to the predecessor's, so `deepseek_ocr.rs`'s
//! own tests already cover it.
//!
//! Lives in the model store (`brain fetch deepseek-ai/DeepSeek-OCR-2`); a box
//! without it skips, loudly, rather than failing.

use checkpoint::gguf::MmapGguf;
use gguf::deepseekocr2_vision as vision;

const REPO: &str = "deepseek-ai/DeepSeek-OCR-2";
const MMPROJ_FILE: &str = "mmproj-deepseek-ocr-2-q8_0.gguf";

/// The mmap'd checkpoint, or `None` (with a loud skip) when it is not fetched.
fn open(file: &str) -> Option<MmapGguf> {
    let Some(dir) = brain_testutil::model_dir(REPO) else {
        brain_testutil::skip(&format!("no model store to resolve {REPO}"));
        return None;
    };
    let path = format!("{dir}/{file}");
    if !std::path::Path::new(&path).exists() {
        brain_testutil::skip(&format!("{path} absent (brain fetch {REPO})"));
        return None;
    }
    Some(MmapGguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}")))
}

#[test]
fn vision_config_comes_out_of_the_real_mmproj() {
    let Some(mg) = open(MMPROJ_FILE) else { return };
    let cfg = vision::config_from_gguf(&mg).expect("config_from_gguf on the real mmproj");

    // SAM ViT-B, unchanged from the predecessor.
    assert_eq!(cfg.sam.d_model, 768);
    assert_eq!(cfg.sam.n_layers, 12);
    assert_eq!(cfg.sam.n_heads, 12);
    assert_eq!(cfg.sam.head_dim(), 64);
    assert_eq!(cfg.sam.ffn_hidden, 3072);
    assert_eq!(cfg.sam.patch_size, 16);
    assert_eq!(cfg.sam.grid, 64);
    assert_eq!(cfg.sam.image_size(), 1024);
    assert_eq!(cfg.sam.window_size, 14);
    assert_eq!(cfg.sam.global_attn_layers, vec![2, 5, 8, 11]);
    assert_eq!(cfg.sam.neck_channels, 256);
    assert_eq!(cfg.sam.compress_mid, 512);
    // The settled downsample_channels fact: 896, not config.json's stale 1024.
    assert_eq!(cfg.sam.compress_out, 896);

    // The new Qwen2 resampler: GQA 14/2, not plain MHA.
    assert_eq!(cfg.encoder.d_model, 896);
    assert_eq!(cfg.encoder.n_layers, 24);
    assert_eq!(cfg.encoder.n_heads, 14);
    assert_eq!(cfg.encoder.n_kv_heads, 2);
    assert_eq!(cfg.encoder.head_dim(), 64);
    assert_eq!(cfg.encoder.kv_dim(), 128);
    assert_eq!(cfg.encoder.ffn_hidden, 4864, "not the file's inert clip.vision.feed_forward_length");
    assert!((cfg.encoder.layer_norm_eps - 1e-6).abs() < 1e-8);
    assert_eq!(cfg.encoder.n_query_local, 144, "the 768x768-tile query bank");
    assert_eq!(cfg.encoder.n_query_global, 256, "the 1024x1024-global-view query bank");

    assert_eq!(cfg.projector_in, 896, "no second tower to concatenate against here");
    assert_eq!(cfg.projection_dim, 1280, "the language model's width");
    assert_eq!(cfg.image_mean, vec![0.5, 0.5, 0.5]);
    assert_eq!(cfg.image_std, vec![0.5, 0.5, 0.5]);
}

#[test]
fn vision_imports_with_full_two_way_coverage() {
    let Some(mg) = open(MMPROJ_FILE) else { return };
    let cfg = vision::config_from_gguf(&mg).unwrap();
    let params = cfg.param_list();

    let out = std::env::temp_dir().join(format!("deepseek-ocr2-mmproj-{}.safetensors", std::process::id()));
    let out = out.to_string_lossy().into_owned();

    let stats = vision::import(&mg, &out, Some("test/deepseek-ocr2-vision")).expect("full mmproj import");
    assert_eq!(stats.source_tensors, 473, "every tensor in the file");
    assert_eq!(stats.written, params.len());
    assert!(stats.dropped.is_empty(), "the mmproj has nothing to drop: {:?}", stats.dropped);
    assert_eq!(stats.written, stats.source_tensors, "no fan-out in this tower");

    let reader = checkpoint::weightio::WeightReader::open(&out).unwrap();
    for name in [
        "vision.sam.patch_embed.weight",
        "vision.sam.blocks.11.attn.rel_pos_h",
        "vision.sam.neck.conv2.weight",
        "vision.sam.compress.conv2.weight",
        "vision.encoder.blocks.23.mlp.down.weight",
        "vision.encoder.norm.weight",
        "vision.query_local.weight",
        "vision.query_global.weight",
        "vision.projector.fc.weight",
        "vision.view_separator",
    ] {
        assert!(reader.tensor(name).is_some(), "{name} missing from the imported checkpoint");
    }
    // A value, not just a name: the learned separator must be verbatim.
    let want = mg.tensor("v.view_seperator").unwrap().unwrap();
    assert_eq!(reader.tensor("vision.view_separator").unwrap(), want);

    std::fs::remove_file(&out).ok();
}

/// The file must be told apart by its own `clip.projector_type`, not by the
/// `clip` architecture string it shares with every other mmproj in this crate
/// (including the predecessor's own).
#[test]
fn the_mmproj_routes_on_its_own_projector_type() {
    let Some(mg) = open(MMPROJ_FILE) else { return };
    assert_eq!(gguf::kv::architecture(&mg), Some(vision::GGUF_ARCHITECTURE));
    let cfg = vision::config_from_gguf(&mg).expect("a file declaring the right projector_type must parse");
    let _ = cfg;
    let root_kv = mg.kv().get("clip.projector_type").and_then(|v| v.as_str());
    assert_eq!(root_kv, Some(vision::PROJECTOR_TYPE));
    assert_ne!(vision::PROJECTOR_TYPE, gguf::deepseek_ocr_vision::PROJECTOR_TYPE, "must not collide with the predecessor's mmproj");
}
