// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which dtype tier the REAL Global LLM checkpoint actually lands in.
//!
//! `global_llm::import` asks for int8 (`Qwen::new_shard_i8`), and on a
//! backend reporting `int8_dot` that request is honoured - but nothing
//! checked it against the real checkpoint, and the difference is the whole
//! ballgame: Qwen3-8B's per-layer linears are ~6.95 B parameters, so the
//! tier decides between ~7 GB and ~28 GB for that half of the model. The
//! AR stage loads TWO of these (one per CFG branch), so an unnoticed
//! promotion to fp32 is the difference between fitting two 24 GB cards and
//! exhausting the first one.
//!
//! This gate loads a small SHARD - not the whole 36-layer stack, which is
//! the same "real weights, too big to load whole" discipline
//! `global_llm_parity.rs` uses - and asserts the tier off the resident
//! `Weight` values via `Qwen::linear_dtype`, i.e. what actually landed,
//! never what was requested.

use std::path::Path;

use model::Shard;
use qwen3::{Dtype, Qwen};

/// Layers to load. Two is enough to prove the tier and cheap enough to run
/// beside the parity gates.
const LAYERS: usize = 2;

#[test]
fn the_real_checkpoint_loads_its_linears_as_int8() {
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
    let reader = checkpoint::weightio::WeightReader::open_hf_dir(Path::new(&weights_dir)).expect("open_hf_dir");
    let src = qwen3::import::hf_source(&reader, &cfg).expect("hf_source");

    let shard = Shard { start: 0, end: LAYERS, embed: false, head: false, gpu_index: Shard::ANY_GPU };
    let q = Qwen::new_shard_i8(cfg.clone(), 1, 16, &src, shard);

    let landed = q.linear_dtype();
    let bytes = q.linear_weight_bytes();
    eprintln!("global_llm real checkpoint, {LAYERS} layers: linear_dtype={landed:?} linear_weight_bytes={bytes}");

    // What an fp32 build of the same layers would cost, from the config's
    // own dims - the number this tier exists to avoid.
    let d = cfg.d_model as u64;
    let kv = cfg.kv_dim() as u64;
    let ff = cfg.d_ff as u64;
    let per_layer_params = 2 * d * d + 2 * d * kv + 3 * d * ff;
    let fp32_bytes = per_layer_params * LAYERS as u64 * 4;
    eprintln!("  (fp32 equivalent for these {LAYERS} layers would be {fp32_bytes} bytes)");

    assert_eq!(
        landed,
        Some(Dtype::I8),
        "the real Global LLM checkpoint did NOT land at int8 (got {landed:?}). \
         global_llm::import asks for int8 via Qwen::new_shard_i8; if the device reports \
         int8_dot this must be honoured. A silent promotion to fp32 quadruples the \
         per-layer linears and is what makes two CFG branches exhaust a 24 GB card."
    );
    assert!(
        bytes < fp32_bytes / 2,
        "int8 linears reported {bytes} bytes, not meaningfully smaller than the fp32 \
         equivalent {fp32_bytes} - the tier is nominally int8 but is not saving bytes"
    );
}
