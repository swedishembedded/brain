// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving of released FP8 LLM checkpoints at
// their stored precision for its clients. If your team needs expertise in
// serving FP8 or GGUF releases without converting them, you can procure our
// services by sending an email to info@swedishembedded.com.

// hf-fixture: qwen35
//! `qwen35::resident_checkpoint::ResidentCheckpoint` over a HuggingFace FP8 directory in the real
//! release's layout (128x128 scale blocks): what the paged resident engine is built from.
//!
//! Spec:
//!   * every linear is held as stored - FP8 e4m3 where the release quantized it, bf16 where it did not
//!     (the GDN gate projections) - and the head is bf16, the embedding rows are the table's rows;
//!   * a model built at those tiers computes what the same weights decoded to fp32 compute (the FP8
//!     tile decode is exact, so only the arithmetic differs);
//!   * a directory without a tokenizer is reported unservable by name, not discovered at the first request.

use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};
use qwen35::resident_checkpoint::ResidentCheckpoint;

mod common;
use common::hf_fixture::write_checkpoint_blocked;

/// The real release's scale grid.
const BLOCK: usize = 128;

/// A model whose linears all have a contraction width of a whole 128-block (the FP8 tier's rule), with the
/// real architecture's mix: GDN and GQA layers, a gated query, grouped value heads.
fn fp8_cfg() -> Qwen35Config {
    Qwen35Config {
        d_model: 128,
        intermediate_size: 256,
        vocab: 64,
        n_layers: 4,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 64,
        mrope_section: [3, 3, 2],
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 32,
        linear_value_head_dim: 32,
        ..Qwen35Config::tiny()
    }
}

fn checkpoint(tag: &str) -> (std::path::PathBuf, ResidentCheckpoint) {
    let dir = write_checkpoint_blocked(&fp8_cfg(), tag, BLOCK);
    let ckpt = ResidentCheckpoint::open(dir.to_str().unwrap()).expect("the fixture opens");
    (dir, ckpt)
}

#[test]
fn linears_are_held_as_stored_and_the_head_is_bf16() {
    let (dir, ckpt) = checkpoint("tiers");
    let cfg = ckpt.config(32).unwrap();
    assert_eq!((cfg.block_size, cfg.mtp), (32, false), "the resident sizes stages for its own context and loads MTP itself");
    let tiers = ckpt.held_tiers(&cfg).unwrap();
    for (l, ty) in cfg.layer_types().into_iter().enumerate() {
        for (suffix, _, _) in cfg.layer_leaves(ty) {
            let want = if suffix.contains("in_proj_a") || suffix.contains("in_proj_b") { Dtype::BF16 } else { Dtype::F8E4M3 };
            assert_eq!(tiers.want(&format!("blocks.{l}.{suffix}")), want, "blocks.{l}.{suffix}");
        }
    }
    assert_eq!(ckpt.head_tier(), Dtype::BF16);
    let (embed, norm, head) = ckpt.endpoint_names(&cfg).unwrap();
    assert_eq!((embed.as_str(), norm.as_str(), head.as_str()), ("tok.weight", "norm.weight", cfg.head_weight()));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn embedding_rows_are_the_tables_rows_in_the_order_asked() {
    let (dir, ckpt) = checkpoint("embed");
    let cfg = ckpt.config(32).unwrap();
    let d = cfg.d_model as usize;
    let mut table = Vec::new();
    assert!(ckpt.endpoint_source().with_tensor("tok.weight", &mut |t| table = t.to_vec()));
    let rows = [5usize, 0, 63, 5];
    let got = ckpt.embed_rows("tok.weight", &rows, d).expect("in range").unwrap();
    let want: Vec<f32> = rows.iter().flat_map(|&r| table[r * d..(r + 1) * d].iter().copied()).collect();
    assert_eq!(got, want);
    assert!(ckpt.embed_rows("tok.weight", &[64], d).is_none(), "a token past the vocabulary is refused");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_model_at_the_stored_tiers_computes_what_the_decoded_weights_compute() {
    let (dir, ckpt) = checkpoint("parity");
    let cfg = ckpt.config(24).unwrap();
    let tiers = ckpt.held_tiers(&cfg).unwrap();
    let shard = model::shard::Shard { start: 0, end: cfg.n_layers as usize, embed: true, head: true, gpu_index: 0 };
    let src_holder = ckpt.shard_source(&cfg, &shard).unwrap();
    let src = src_holder.as_source();

    let held = Qwen35::new_on_dt(Gpu::new_cpu(pipelines()), cfg.clone(), 1, cfg.block_size, src, &tiers);
    let dense = Qwen35::new_on_dt(Gpu::new_cpu(pipelines()), cfg.clone(), 1, cfg.block_size, src, &TierPolicy::uniform(Dtype::F32));
    assert_eq!(held.weight_tier("blocks.0.mlp.gate.weight"), Some(Dtype::F8E4M3), "the linears must stay FP8 on the device");

    let tokens: Vec<u32> = (0..cfg.block_size).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let (a, b) = (held.logits_all(&tokens), dense.logits_all(&tokens));
    let num: f64 = a.iter().zip(&b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    assert!(b.iter().any(|v| v.abs() > 1e-6), "the reference is degenerate");
    assert!((num / den).sqrt() < 1e-4, "FP8-held logits drift from the decoded weights' by rel-L2 {}", (num / den).sqrt());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_directory_without_a_tokenizer_is_unservable_by_name() {
    let (dir, ckpt) = checkpoint("notok");
    assert!(!ckpt.has_tokenizer());
    let err = ckpt.tokenizer().err().expect("no tokenizer.json");
    assert!(err.contains("tokenizer.json"), "{err}");
    std::fs::remove_dir_all(dir).ok();
}
