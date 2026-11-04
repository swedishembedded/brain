// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gemma-4 tiny-config parity against `tools/goldens/
//! gemma4_dump_reference.py`'s fixture (6 layers - 5 `sliding_attention` + 1
//! `full_attention`, the real 5:1 ratio's minimal instance).
//!
//! Replays the golden's OWN `input_ids` through [`gemma4::Gemma4Model::
//! forward`] and asserts every captured tap - both RoPE tables, EVERY
//! `hidden_states` entry (proving the embedding + 5 layers' worth of the
//! 49-hidden-state aggregate convention this milestone exists to pin), the
//! self-attention output of layer 0 (`sliding_attention`, GQA) AND of layer 5
//! (`full_attention`, MQA + `attention_k_eq_v` - the two structurally
//! DIFFERENT attention paths, each getting its own assertion, not just an
//! aggregate pass/fail), the final (post-norm) hidden state, and LTX's own
//! aggregate-embed projection - at cosine >= 0.999999, this repo's
//! established tiny-config parity bar (`ltxv::tests::dit_parity`'s bar).
//!
//! Skips loudly without the fixture (`BRAIN_REQUIRE_FIXTURES=1` upgrades a
//! skip to a failure), matching every other tiny-config parity suite in this
//! repo.

use std::path::Path;

use gemma4::{load_tiny_weights, AggregateEmbed, Gemma4Config, Gemma4Model};

// ------------------------------------------------------------------ metrics

/// Same formula `ltxv::tests::dit_parity`/`model::hostmath::cosine` use (f64
/// accumulation, both norms as separate factors).
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch ({} vs {})", a.len(), b.len());
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    let den = na.sqrt() * nb.sqrt();
    if den <= 0.0 {
        0.0
    } else {
        d / den
    }
}

/// The golden dumps `rotary_emb`'s OWN return value directly - full
/// `head_dim`-wide per row (`emb = cat(freqs, freqs)` in the reference,
/// duplicated, and for the `full_attention`/`proportional` table also
/// front-loaded with the genuinely-rotating frequencies followed by
/// zero-frequency ("nope") padding before that duplication). This crate's
/// own [`gemma4::rope::RopeTable`] intentionally stores only the COMPACT
/// active-frequency prefix each row needs (`half = head_dim/2` for sliding,
/// `half = rope_angles` for full/global - see `gemma4::rope`'s doc for why:
/// that is exactly what `rope2d`/`rope2d_partial` consume, and the remaining
/// columns are a redundant duplicate-plus-identity-padding restatement of the
/// same values). Slicing the first `half` columns of each golden row is
/// therefore the correct comparison, not a shortcut - `emb`'s own
/// construction guarantees columns `[0, half)` of the un-duplicated `freqs`
/// ARE those active frequencies, verbatim.
fn row_prefix(golden_flat: &[f32], t: usize, full_width: usize, half: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(t * half);
    for row in 0..t {
        out.extend_from_slice(&golden_flat[row * full_width..row * full_width + half]);
    }
    out
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn report(label: &str, got: &[f32], want: &[f32], min_cos: f64) {
    assert_eq!(got.len(), want.len(), "{label}: {} values vs {}", got.len(), want.len());
    let (c, m) = (cosine(got, want), max_abs(got, want));
    eprintln!("{label}: cosine={c:.9}  max_abs={m:.3e}  n={}", got.len());
    assert!(c >= min_cos, "{label}: cosine {c:.9} < {min_cos}");
}

const MIN_COS: f64 = 0.999999;

// ---------------------------------------------------------- real fixtures

struct Fixture {
    t: Vec<checkpoint::safetensors::StTensor>,
}

impl Fixture {
    fn get(&self, name: &str) -> &[f32] {
        &self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).data
    }
}

/// `(fixture, weights)` or `None` with a loud skip.
fn setup() -> Option<(Fixture, gemma4::block::Tensors)> {
    let fx_path = brain_testutil::testdata("golden/gemma4/gemma4_tiny.safetensors");
    let w_path = brain_testutil::testdata("golden/gemma4/gemma4_tiny_weights.safetensors");
    if !Path::new(&fx_path).exists() || !Path::new(&w_path).exists() {
        brain_testutil::skip(&format!("fixture {fx_path} absent - run tools/goldens/gemma4_dump_reference.py"));
        return None;
    }
    let t = checkpoint::safetensors::read(&fx_path).expect("read golden");
    let w = load_tiny_weights(&w_path);
    Some((Fixture { t }, w))
}

#[test]
fn gemma4_tiny_matches_reference() {
    let Some((fx, w)) = setup() else { return };

    let cfg = Gemma4Config::tiny();
    let n = cfg.num_hidden_layers as usize;
    let hidden = cfg.hidden_size as usize;

    let input_ids: Vec<u32> = fx.get("input_ids").iter().map(|&v| v.round() as u32).collect();
    let t = input_ids.len();

    let model = Gemma4Model::new(cfg, w.clone(), None);
    let out = model.forward(&input_ids);

    // ---- RoPE tables: both layer types, both structurally different -------
    // The golden captures `rotary_emb`'s own full-duplicated-width return
    // value directly; this crate's tables are the compact active-frequency
    // prefix each row needs - see `row_prefix`'s doc for why slicing the
    // golden's first `half` columns per row is the correct comparison.
    let sliding_half = out.rope_sliding_cos.len() / t;
    let full_half = out.rope_full_cos.len() / t;
    let sliding_full_width = cfg.head_dim as usize;
    let full_full_width = cfg.global_head_dim as usize;
    report("rope_sliding_cos", &out.rope_sliding_cos, &row_prefix(fx.get("rope_sliding_cos"), t, sliding_full_width, sliding_half), MIN_COS);
    report("rope_sliding_sin", &out.rope_sliding_sin, &row_prefix(fx.get("rope_sliding_sin"), t, sliding_full_width, sliding_half), MIN_COS);
    report("rope_full_cos", &out.rope_full_cos, &row_prefix(fx.get("rope_full_cos"), t, full_full_width, full_half), MIN_COS);
    report("rope_full_sin", &out.rope_full_sin, &row_prefix(fx.get("rope_full_sin"), t, full_full_width, full_half), MIN_COS);

    // ---- self-attention output: one SLIDING (GQA) layer, one FULL
    // ---- (MQA + attention_k_eq_v) layer - two structurally different
    // ---- paths, each with its own assertion ---------------------------
    report("layer0_self_attn_out (sliding, GQA)", &out.layer0_self_attn_out, fx.get("layer0_self_attn_out"), MIN_COS);
    report(
        &format!("layer{}_self_attn_out (full, MQA + k_eq_v)", n - 1),
        &out.layer_last_self_attn_out,
        fx.get(&format!("layer{}_self_attn_out", n - 1)),
        MIN_COS,
    );

    // ---- every hidden_states entry (the 49-state convention this
    // ---- milestone exists to pin - see `gemma4::model`'s doc) -------------
    assert_eq!(out.hidden_states.len(), n + 1);
    for (k, hs) in out.hidden_states.iter().enumerate() {
        report(&format!("hidden_states.{k}"), hs, fx.get(&format!("hidden_states.{k}")), MIN_COS);
    }

    report("last_hidden_state", &out.last_hidden_state, fx.get("last_hidden_state"), MIN_COS);

    // ---- LTX's own 49-hidden-state aggregate-embed projection --------------
    let agg = AggregateEmbed::from_weights(&w, hidden, n + 1);
    let agg_out = agg.forward(&out.hidden_states, t, hidden);
    report("aggregate_out", &agg_out, fx.get("aggregate_out"), MIN_COS);
}
