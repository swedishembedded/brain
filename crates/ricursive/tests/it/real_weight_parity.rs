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

use ricursive::{open_safetensors, RicursiveConfig, RicursiveModel};

use crate::taps::{check, Taps};

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
