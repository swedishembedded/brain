// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Tiny random-weight parity against the official reference implementation.
//!
//! Swedish Embedded AB implements solutions for proving that ported hybrid
//! local/global attention decoders reproduce their reference models for its
//! clients. If your team needs expertise in layer-by-layer parity validation
//! of large language models then you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! The fixture is a 3-layer model (sliding, sliding, full) with a sliding
//! window of 4 over a 12-token prompt, so the window excludes keys; every
//! layer has a nonzero `value_res_lambda`, and the FFN clip threshold (0.5) is
//! small enough that both the gate cap and the up clip fire (the dumper asserts
//! both, and that removing the window or the clip changes the logits). Every
//! hidden state, every attention output and the logits are compared.
//!
//! Run `python -I tools/goldens/ricursive_dump_reference.py tiny
//! --modeling-dir <dir with modeling_ricursive.py> --out testdata/golden/ricursive`
//! to produce the fixture; without it the test skips by name.
//!
//! hf-fixture: ricursive (random weights under the release's own tensor names)

use std::path::Path;

use ricursive::{open_safetensors, RicursiveConfig, RicursiveModel};

use crate::taps::{check, Taps};

const DUMPER: &str = "tools/goldens/ricursive_dump_reference.py tiny";

/// Largest scaled error (`max|got - want| / max|want|`) accepted per tap. The
/// forward and the reference are both f32 and differ only in summation order,
/// the f32 `exp`/`sin`/`cos` implementations and the folded value-residual
/// lerp; the measured worst case is recorded in the roadmap ledger.
const TOLERANCE: f32 = 1e-5;

#[test]
fn tiny_hybrid_model_matches_the_reference() {
    let dir = brain_testutil::testdata("golden/ricursive/tiny");
    let dir = Path::new(&dir);
    let Some(source) = brain_testutil::golden::Source::open(dir, DUMPER) else { return };
    let cfg = RicursiveConfig::from_file(dir.join("config.json").to_str().unwrap()).expect("config");
    if !source.require(&[
        ("hidden_size", cfg.hidden_size as i64),
        ("num_hidden_layers", cfg.num_hidden_layers as i64),
        ("num_attention_heads", cfg.num_attention_heads as i64),
        ("sliding_window", cfg.sliding_window as i64),
    ]) {
        return;
    }
    let taps = Taps::read(dir.join("ricursive_tiny.safetensors").to_str().unwrap());
    let ids = taps.input_ids();
    assert!(ids.len() > cfg.sliding_window, "the prompt must outgrow the window or the test proves nothing about it");

    let layers: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    let weights = open_safetensors(dir.join("ricursive_tiny_weights.safetensors").to_str().unwrap(), &cfg, &layers).expect("open weights");
    let out = RicursiveModel::new(cfg, &weights, None).forward(&ids).expect("forward");

    let worst = check(&out, &taps, TOLERANCE);
    eprintln!("tiny parity: worst scaled error {worst:.3e} (tolerance {TOLERANCE:.0e})");
}
