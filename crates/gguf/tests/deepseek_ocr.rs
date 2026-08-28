// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The two DeepSeek-OCR GGUF files, against the real checkpoints.
//!
//! These are the proof cases for the shared loader: a config derived from the
//! file must reproduce the model, and the tensor map must cover the file in
//! **both** directions. Every number asserted below is re-derived from the
//! checkpoint by the code under test - the assertions are the independently
//! decoded header, so a mapping that drifts fails here rather than silently
//! producing a checkpoint with a hole in it.
//!
//! Both files live in the model store (`brain fetch ggml-org/DeepSeek-OCR-GGUF`);
//! a box without them skips, loudly, rather than failing.

use checkpoint::gguf::MmapGguf;
use gguf::deepseek_ocr as lm;
use gguf::deepseek_ocr_vision as vision;
use gguf::import;

const REPO: &str = "ggml-org/DeepSeek-OCR-GGUF";
const LM_FILE: &str = "DeepSeek-OCR-Q8_0.gguf";
const MMPROJ_FILE: &str = "mmproj-DeepSeek-OCR-Q8_0.gguf";

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
fn lm_config_comes_out_of_the_real_checkpoint() {
    let Some(mg) = open(LM_FILE) else { return };
    let cfg = lm::config_from_gguf(&mg).expect("config_from_gguf on the real LM");

    assert_eq!(cfg.vocab, 129280);
    assert_eq!(cfg.n_layers, 12);
    assert_eq!(cfg.d_model, 1280);
    assert_eq!(cfg.max_position_embeddings, 8192);
    assert!((cfg.rms_eps - 1e-6).abs() < 1e-9, "rms_eps {}", cfg.rms_eps);
    assert!(!cfg.tie_embeddings, "output.weight is present -> untied");

    // Plain MHA: head_count == head_count_kv, and head_dim comes off the
    // square attn_q tensor (no attention.key_length key in this file).
    assert_eq!(cfg.n_heads, 10);
    assert_eq!(cfg.n_kv_heads, 10);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.q_dim(), cfg.d_model, "1280 = 10 heads x 128");

    // rope.dimension_count is 0 in this file; the full head is rotated.
    assert_eq!(cfg.rotary_dim, 128);
    assert_eq!(cfg.rope_theta, 10_000.0);

    assert_eq!(cfg.n_dense_layers, 1, "blk.0 is dense, blk.1..11 are MoE");
    assert!(!cfg.is_moe_layer(0));
    assert!(cfg.is_moe_layer(1));
    assert_eq!(cfg.ffn_hidden, 6848);
    assert_eq!(cfg.n_experts, 64);
    assert_eq!(cfg.top_k, 6);
    assert_eq!(cfg.moe_intermediate_size, 896);
    assert_eq!(cfg.n_shared_experts, 2);
    assert_eq!(cfg.shared_intermediate_size(), 1792, "the fused *_shexp width");
    assert_eq!(cfg.n_expert_groups, 1);
    assert_eq!(cfg.n_expert_groups_used, 1);

    // The config's own manifest must agree with the checkpoint's shapes.
    let dense = mg.shape("blk.0.ffn_gate.weight").unwrap();
    assert_eq!(dense, [cfg.ffn_hidden as usize, cfg.d_model as usize]);
    let shexp = mg.shape("blk.1.ffn_gate_shexp.weight").unwrap();
    assert_eq!(shexp, [cfg.shared_intermediate_size() as usize, cfg.d_model as usize]);
    let exps = mg.shape("blk.1.ffn_gate_exps.weight").unwrap();
    assert_eq!(exps, [cfg.n_experts as usize, cfg.moe_intermediate_size as usize, cfg.d_model as usize]);
}

#[test]
fn lm_tensor_map_covers_the_real_checkpoint_both_ways() {
    let Some(mg) = open(LM_FILE) else { return };
    let cfg = lm::config_from_gguf(&mg).unwrap();
    let params = cfg.param_list();

    // Header-only: this checkpoint's fp32 expansion is ~10 GB, and the whole
    // mapping (including the 64-way expert fan-out and every element count)
    // is provable without expanding a byte of it.
    let stats = import::dry_run(&mg, &params, &|n| lm::classify(n, &cfg), "deepseek-ocr")
        .expect("every source tensor classified and every planned tensor produced");

    assert_eq!(stats.source_tensors, mg.names().len());
    assert_eq!(stats.written, params.len(), "the plan and the mapping must be the same set");
    assert!(stats.dropped.is_empty(), "the LM file has nothing to drop: {:?}", stats.dropped);

    // The expert fan-out is where a plan and a classifier drift apart: 11 MoE
    // layers x 64 experts x 3 leaves.
    let experts = params.iter().filter(|(n, _)| n.contains(".mlp.experts.")).count();
    assert_eq!(experts, 11 * 64 * 3);
    assert!(params.iter().any(|(n, _)| n == "blocks.0.mlp.gate.weight"), "blk.0 is the dense block");
    assert!(!params.iter().any(|(n, _)| n == "blocks.0.mlp.router.weight"), "a dense block has no router");
    assert!(params.iter().any(|(n, _)| n == "blocks.11.mlp.shared.down.weight"));
}

#[test]
fn vision_config_comes_out_of_the_real_mmproj() {
    let Some(mg) = open(MMPROJ_FILE) else { return };
    let cfg = vision::config_from_gguf(&mg).expect("config_from_gguf on the real mmproj");

    // SAM ViT-B at 1024x1024 (a 64x64 patch grid) - the image size is NOT the
    // file's clip.vision.image_size (224, which describes CLIP).
    assert_eq!(cfg.sam.d_model, 768);
    assert_eq!(cfg.sam.n_layers, 12);
    assert_eq!(cfg.sam.n_heads, 12);
    assert_eq!(cfg.sam.head_dim(), 64);
    assert_eq!(cfg.sam.ffn_hidden, 3072, "4x width, read off mlp.lin1");
    assert_eq!(cfg.sam.patch_size, 16);
    assert_eq!(cfg.sam.grid, 64);
    assert_eq!(cfg.sam.image_size(), 1024);
    assert_eq!(cfg.sam.window_size, 14);
    // Every 3rd block from 2 is global, decided by its own rel-pos extent.
    assert_eq!(cfg.sam.global_attn_layers, vec![2, 5, 8, 11]);
    assert_eq!(cfg.sam.rel_pos_rows(0), 27, "2 * window - 1");
    assert_eq!(cfg.sam.rel_pos_rows(2), 127, "2 * grid - 1");
    assert_eq!(cfg.sam.neck_channels, 256);
    assert_eq!(cfg.sam.compress_mid, 512);
    assert_eq!(cfg.sam.compress_out, 1024);

    // CLIP-L/14 at 224: 256 patches + a class token.
    assert_eq!(cfg.clip.d_model, 1024);
    assert_eq!(cfg.clip.n_layers, 24);
    assert_eq!(cfg.clip.n_heads, 16);
    assert_eq!(cfg.clip.patch_size, 14);
    assert_eq!(cfg.clip.image_size, 224);
    assert_eq!(cfg.clip.n_positions, 257);
    // The file's clip.vision.feed_forward_length says 64 (a converter bug:
    // heads*4 where width*4 was meant). The tensors say 4096, and the tensors
    // are what the model runs.
    assert_eq!(cfg.clip.ffn_hidden, 4096);
    assert_eq!(
        mg.kv().get("clip.vision.feed_forward_length").and_then(|v| v.as_u64()),
        Some(64),
        "if the file ever stops lying, this test is the place to notice"
    );

    assert_eq!(cfg.projector_in, 2048, "CLIP 1024 ++ compressor 1024");
    assert_eq!(cfg.projection_dim, 1280, "the language model's width");
    assert_eq!(cfg.image_mean, vec![0.5, 0.5, 0.5]);
    assert_eq!(cfg.image_std, vec![0.5, 0.5, 0.5]);
    assert!(cfg.use_gelu);
    assert_eq!(cfg.scale_factor, 1);
}

#[test]
fn vision_imports_with_full_two_way_coverage() {
    let Some(mg) = open(MMPROJ_FILE) else { return };
    let cfg = vision::config_from_gguf(&mg).unwrap();
    let params = cfg.param_list();

    let out = std::env::temp_dir().join(format!("deepseek-ocr-mmproj-{}.safetensors", std::process::id()));
    let out = out.to_string_lossy().into_owned();

    // The real thing: dequantize every tensor, write brain's own file, and
    // fail unless both directions of the coverage check hold.
    let stats = vision::import(&mg, &out, Some("test/deepseek-ocr-vision")).expect("full mmproj import");
    assert_eq!(stats.source_tensors, 476, "every tensor in the file");
    assert_eq!(stats.written, params.len());
    assert!(stats.dropped.is_empty(), "the mmproj has nothing to drop: {:?}", stats.dropped);
    // No fan-out in this tower: one source tensor, one output tensor.
    assert_eq!(stats.written, stats.source_tensors);

    // Spot-check that the bytes landed under the remapped names, one per stage.
    let reader = checkpoint::weightio::WeightReader::open(&out).unwrap();
    for name in [
        "vision.sam.patch_embed.weight",
        "vision.sam.blocks.11.attn.rel_pos_h",
        "vision.sam.neck.conv2.weight",
        "vision.sam.compress.conv2.weight",
        "vision.clip.blocks.23.mlp.fc1.weight",
        "vision.projector.fc.weight",
        "vision.view_separator",
    ] {
        assert!(reader.tensor(name).is_some(), "{name} missing from the imported checkpoint");
    }
    // Values, not just names: the learned newline vector must be verbatim.
    let want = mg.tensor("v.image_newline").unwrap().unwrap();
    assert_eq!(reader.tensor("vision.image_newline").unwrap(), want);

    std::fs::remove_file(&out).ok();
}

/// Both shipped files route by their own metadata, through the one seam every
/// GGUF consumer shares.
///
/// This used to assert against a second architecture table private to this
/// crate. That table's rows now live in the single importer table beside every
/// other architecture (`cli::gguf_import`), which is what this crate cannot
/// see and does not need to: what belongs here is that the two REAL files
/// resolve to the right architecture and the right projector type, which is
/// exactly what a consumer dispatches on.
#[test]
fn both_shipped_files_route_by_their_own_metadata() {
    // The `clip` architecture is shared by every mmproj ever produced, so the
    // vision file must be told apart by its projector_type, not by `clip`
    // alone.
    if let Some(mg) = open(LM_FILE) {
        let r = gguf::route(&mg).expect("the LM file must route");
        assert_eq!(r.tag, gguf::deepseek_ocr::GGUF_ARCHITECTURE);
        assert_eq!(r.id(), "deepseek2ocr");
        assert!(!r.is_projector(), "the decoder half is not a projector");
    }
    if let Some(mg) = open(MMPROJ_FILE) {
        let r = gguf::route(&mg).expect("the mmproj file must route");
        assert_eq!(r.tag, gguf::deepseek_ocr_vision::GGUF_ARCHITECTURE);
        assert!(r.is_projector(), "the mmproj half must be recognized as a projector");
        assert_eq!(r.projector.as_deref(), Some(gguf::deepseek_ocr_vision::PROJECTOR_TYPE));
    }
}
