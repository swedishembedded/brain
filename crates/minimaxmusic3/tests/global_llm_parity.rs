// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-weight parity for the Global LLM (`crates/qwen3::Qwen`, reused
//! verbatim - see `crate::global_llm`'s own doc): a single REAL decoder
//! layer's forward, streamed straight from the checkpoint's
//! `language_model/` directory via `qwen3::import::hf_source` + a
//! 1-layer `model::Shard`, compared against `transformers.Qwen3DecoderLayer`
//! loaded with the SAME real weights
//! (`tools/goldens/minimaxmusic3_global_llm_dump_reference.py`).
//!
//! Never loads the whole 36-layer, ~17 GB model - `Qwen::new_shard` with
//! `Shard{start:L, end:L+1, embed:false, head:false}` allocates only
//! layer `L`'s own weights, streamed one tensor at a time. The layer's
//! raw hidden-state input/output is fed/read directly through
//! `write_in_res`/`read_out_res` (bypassing `embed_tokens`/`lm_head`,
//! neither of which this 1-layer shard owns).
//!
//! Real weights (~17 GB) are not committed - set `BRAIN_MINIMAXMUSIC3_LM`
//! to the checkpoint's `language_model/` directory; skips cleanly when
//! unset or the golden fixture is absent.

use std::path::Path;

use brain_testutil::{golden::Source, parity::compare, testdata_path};
use model::Shard;
use qwen3::Qwen;

const DUMPER: &str = "tools/goldens/minimaxmusic3_global_llm_dump_reference.py";
const COS_FLOOR: f64 = 0.999;
const LAYER: usize = 0;

#[test]
fn real_layer_matches_transformers() {
    let dir = testdata_path(&format!("golden/minimaxmusic3/global_llm_layer_{LAYER}"));
    let meta = dir.join("manifest.json");
    let Some(src) = Source::open_manifest(&meta, DUMPER) else { return };

    let Ok(weights_dir) = std::env::var("BRAIN_MINIMAXMUSIC3_LM") else {
        brain_testutil::skip("BRAIN_MINIMAXMUSIC3_LM unset");
        return;
    };
    if !Path::new(&weights_dir).exists() {
        brain_testutil::skip(&format!("BRAIN_MINIMAXMUSIC3_LM={weights_dir} not found"));
        return;
    }

    let config_json = std::fs::read_to_string(Path::new(&weights_dir).join("config.json")).expect("read language_model/config.json");
    let cfg = qwen3::import::config_from_hf(&config_json).expect("config_from_hf");
    if !src.require(&[
        ("layer", LAYER as i64),
        ("hidden_size", cfg.d_model as i64),
        ("num_attention_heads", cfg.n_heads as i64),
        ("num_key_value_heads", cfg.n_kv_heads as i64),
        ("head_dim", cfg.head_dim as i64),
    ]) {
        return;
    }

    let fixture = checkpoint::safetensors::read(dir.join("layer.safetensors").to_str().unwrap()).expect("read golden fixture");
    let get = |name: &str| fixture.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden fixture missing {name}"));
    let x_in = get("x_in");
    let want_out = &get("out").data;
    let t = x_in.shape[0];
    let d_model = x_in.shape[1];
    assert_eq!(d_model, cfg.d_model as usize, "golden hidden_size does not match the real config");

    let reader = checkpoint::weightio::WeightReader::open_hf_dir(Path::new(&weights_dir)).expect("open_hf_dir");
    let source = qwen3::import::hf_source(&reader, &cfg).expect("hf_source");
    let shard = Shard { start: LAYER, end: LAYER + 1, embed: false, head: false, gpu_index: Shard::ANY_GPU };
    let qwen = Qwen::new_shard(cfg, 1, t as u32, &source, false, shard);

    qwen.write_in_res(&x_in.data);
    qwen.run_forward();
    let got = qwen.read_out_res();

    assert_eq!(got.len(), want_out.len(), "global_llm[layer {LAYER}]: output length mismatch");
    let (cos, max_abs) = compare(&got, want_out);
    println!("global_llm[layer {LAYER}]: cosine={cos:.9} max_abs={max_abs:.6}");
    assert!(cos >= COS_FLOOR, "global_llm[layer {LAYER}]: cosine {cos} below floor {COS_FLOOR}");
}
