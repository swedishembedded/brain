// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-weight parity on layers 0 (sliding) and 4 (global) of the released
//! checkpoint, with its real embedding, final norm and `lm_head`.
//!
//! Swedish Embedded AB implements solutions for proving that ported large
//! language models reproduce their reference implementations on real weights
//! for its clients. If your team needs expertise in partial-checkpoint parity
//! validation then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The 80-layer checkpoint is far too large to hold, so the test runs the two
//! layers as a 2-layer model with their real types (`[sliding, full]`): the
//! reference side is `tools/goldens/ricursive_dump_reference.py real`, which
//! builds the same truncated model, renames layer 4 to layer 1 and promotes the
//! bf16 weights to f32. The Rust side reads the same file through
//! `open_safetensors` with the layer map `[0, 4]`.
//!
//! Inputs, under the testdata root (see `make testdata/fetch`):
//! `ricursive/ripple-65b-partial/{layers-0-4.safetensors,config.json}` and the
//! golden `golden/ricursive/real/`.

use std::path::Path;

use model::paged::BlockTable;
use model::serve::{PagedDecoder, PrefillProgress};
use ricursive::{open_safetensors, RicursiveConfig, RicursiveEngine, RicursiveModel, ServeConfig};

use crate::taps::{check, scaled_error, Taps};

const DUMPER: &str = "tools/goldens/ricursive_dump_reference.py real";
const SOURCE_LAYERS: [usize; 2] = [0, 4];

/// Real weights are not normalized to unit scale, and the stream passes
/// through a 5120-wide, 40960-wide stack, so the bound is looser than the tiny
/// model's; the measured worst case is recorded in the roadmap ledger.
const TOLERANCE: f32 = 5e-5;

#[test]
fn real_layers_0_and_4_match_the_reference() {
    let checkpoint = brain_testutil::testdata("ricursive/ripple-65b-partial/layers-0-4.safetensors");
    let config = brain_testutil::testdata("ricursive/ripple-65b-partial/config.json");
    let golden = brain_testutil::testdata("golden/ricursive/real");
    for path in [&checkpoint, &config] {
        if !Path::new(path).exists() {
            brain_testutil::skip(&format!("{path} absent: place the partial Ripple-65B checkpoint (embed, norm, lm_head, layers 0 and 4) under testdata/ricursive/ripple-65b-partial/"));
            return;
        }
    }
    let Some(source) = brain_testutil::golden::Source::open(Path::new(&golden), DUMPER) else { return };

    let full = RicursiveConfig::from_file(&config).expect("config");
    let cfg = full.truncated(&SOURCE_LAYERS).expect("truncated config");
    if !source.require(&[
        ("hidden_size", cfg.hidden_size as i64),
        ("intermediate_size", cfg.intermediate_size as i64),
        ("num_hidden_layers", cfg.num_hidden_layers as i64),
        ("num_attention_heads", cfg.num_attention_heads as i64),
        ("sliding_window", cfg.sliding_window as i64),
    ]) {
        return;
    }
    let taps = Taps::read(&format!("{golden}/ricursive_real_reduced.safetensors"));
    let ids = taps.input_ids();

    let started = std::time::Instant::now();
    let weights = open_safetensors(&checkpoint, &cfg, &SOURCE_LAYERS).expect("open checkpoint");
    let out = RicursiveModel::new(cfg, &weights, None).forward(&ids).expect("forward");
    eprintln!("forward over {} tokens in {:.1}s", ids.len(), started.elapsed().as_secs_f64());

    let worst = check(&out, &taps, TOLERANCE);
    eprintln!("real-weight parity: worst scaled error {worst:.3e} (tolerance {TOLERANCE:.0e})");
}

/// The serving engine on the same two real layers: prefill the first 9 tokens of the reference prompt in chunks
/// of 4, then decode the next 4 one at a time (the prompt's own tokens as inputs), against the official
/// reference's logits at those positions. The 13-token prompt is far inside the 1024 window, so this proves the
/// real-width paged global layer and the ring write/read path, not window exclusion (the tiny model does that).
#[test]
fn the_serving_engine_decodes_real_layers_0_and_4_as_the_reference() {
    let checkpoint = brain_testutil::testdata("ricursive/ripple-65b-partial/layers-0-4.safetensors");
    let config = brain_testutil::testdata("ricursive/ripple-65b-partial/config.json");
    let golden = brain_testutil::testdata("golden/ricursive/real");
    for path in [&checkpoint, &config] {
        if !Path::new(path).exists() {
            brain_testutil::skip(&format!("{path} absent: place the partial Ripple-65B checkpoint under testdata/ricursive/ripple-65b-partial/"));
            return;
        }
    }
    let Some(source) = brain_testutil::golden::Source::open(Path::new(&golden), DUMPER) else { return };
    let cfg = RicursiveConfig::from_file(&config).expect("config").truncated(&SOURCE_LAYERS).expect("truncated config");
    if !source.require(&[("hidden_size", cfg.hidden_size as i64), ("num_hidden_layers", cfg.num_hidden_layers as i64), ("sliding_window", cfg.sliding_window as i64)]) {
        return;
    }
    let taps = Taps::read(&format!("{golden}/ricursive_real_reduced.safetensors"));
    let (ids, vocab) = (taps.input_ids(), cfg.vocab_size);
    let want = taps.get("logits");
    let weights = open_safetensors(&checkpoint, &cfg, &SOURCE_LAYERS).expect("open checkpoint");
    let serve = ServeConfig { block_size: 16, num_blocks: 4, max_seq_len: 32, max_batch: 2, prefix_slots: 0, prefill_chunk: 4 };
    let mut engine = RicursiveEngine::from_source(cfg, Box::new(weights), serve, None).expect("engine");

    let (prompt, rest) = ids.split_at(9);
    let mut table = BlockTable::new();
    let hidden = loop {
        if let PrefillProgress::Done(h) = engine.prefill_chunk(&mut table, prompt, 4) {
            break h;
        }
    };
    let mut worst = scaled_error("prefill, position 8", &engine.logits(&hidden), &want[8 * vocab..9 * vocab]);
    for (i, &token) in rest.iter().enumerate() {
        let pos = 9 + i;
        let logits = engine.decode_logits(&mut [&mut table], &[token]);
        worst = worst.max(scaled_error(&format!("decode, position {pos}"), &logits[0], &want[pos * vocab..(pos + 1) * vocab]));
    }
    eprintln!("real-weight serving: worst scaled error {worst:.3e} (tolerance {TOLERANCE:.0e})");
    assert!(worst <= TOLERANCE, "scaled error {worst:.3e} exceeds {TOLERANCE:.0e}");
}
