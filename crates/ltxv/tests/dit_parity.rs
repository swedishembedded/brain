// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5 video-only DiT parity against `tools/goldens/
//! ltxv_dit_dump_reference.py`'s tiny (2-layer, `inner_dim` 64) fixture.
//!
//! One test, replaying the golden's OWN inputs (`latent`/`context`/
//! `timesteps`/`positions`/`keyframes_mask` - read straight off the fixture,
//! never hand-reconstructed, per this port's porting playbook's "dump
//! reference goldens first" rule) through
//! [`ltxv::LtxDit::forward`] and asserting every captured tap (`rope_cos`/
//! `rope_sin`, `adaln_table`, `embedded_timestep`, the block-0 internal taps,
//! each block's output, and the final `out`) at cosine >= 0.999999 -
//! `crate::vae_parity`'s bar, and this crate's own M2 tests still pass
//! unchanged alongside these new M3 ones.
//!
//! Skips loudly without the fixture (`BRAIN_REQUIRE_FIXTURES=1` upgrades a
//! skip to a failure), matching `vae_parity.rs`'s convention. Unlike that
//! suite's `OnceLock`-shared weights (needed there because the real VAE
//! checkpoint is ~726M parameters), this milestone's tiny weights are 60
//! tensors / 0.84 MB total - cheap enough to load fresh per test, so there is
//! no shared-static ceremony to get right here.

use std::path::Path;

use ltxv::block::{open_device, EmbeddingsConnector};
use ltxv::{load_tiny_weights, LtxDit, LtxDitConfig};

// ------------------------------------------------------------------ metrics

/// Same formula `vae_parity.rs`/`model::hostmath::cosine` use (f64
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
/// `crate::shard_parity`'s own strict-tap bound - used only by the gated/
/// connector test below (the existing no-gating test above is UNCHANGED,
/// still `report`-only, per this port's own precedent for keeping a landed
/// milestone's assertions untouched when a later one adds new taps).
const MAX_ABS_BOUND: f32 = 1e-3;

fn report_strict(label: &str, got: &[f32], want: &[f32], min_cos: f64) {
    assert_eq!(got.len(), want.len(), "{label}: {} values vs {}", got.len(), want.len());
    let (c, m) = (cosine(got, want), max_abs(got, want));
    eprintln!("{label}: cosine={c:.9}  max_abs={m:.3e}  n={}", got.len());
    assert!(c >= min_cos, "{label}: cosine {c:.9} < {min_cos}");
    assert!(m < MAX_ABS_BOUND, "{label}: max_abs {m:.3e} >= {MAX_ABS_BOUND:.3e}");
}

// ---------------------------------------------------------- real fixtures

struct Fixture {
    t: Vec<checkpoint::safetensors::StTensor>,
}

impl Fixture {
    fn get(&self, name: &str) -> &[f32] {
        &self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).data
    }
    fn shape(&self, name: &str) -> &[usize] {
        &self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).shape
    }
}

/// `(fixture, weights)` or `None` with a loud skip.
fn setup() -> Option<(Fixture, vae::blocks::Tensors)> {
    let fx_path = brain_testutil::testdata("golden/ltxv/dit/dit_tiny.safetensors");
    let w_path = brain_testutil::testdata("golden/ltxv/dit/dit_tiny_weights.safetensors");
    if !Path::new(&fx_path).exists() || !Path::new(&w_path).exists() {
        brain_testutil::skip(&format!("fixture {fx_path} absent - run tools/goldens/ltxv_dit_dump_reference.py"));
        return None;
    }
    let t = checkpoint::safetensors::read(&fx_path).expect("read golden");
    let w = load_tiny_weights(&w_path);
    Some((Fixture { t }, w))
}

#[test]
fn ltxv_dit_tiny_matches_reference() {
    let Some((fx, w)) = setup() else { return };

    let cfg = LtxDitConfig::tiny();
    let dim = cfg.inner_dim as usize;

    let latent = fx.get("latent");
    let t = fx.shape("latent")[0];
    let context = fx.get("context");
    let context_len = fx.shape("context")[0];
    let timesteps = fx.get("timesteps"); // [T, 1] -> already flat as [T]
    let positions = fx.get("positions"); // [3, T, 2]
    let keyframes_mask = fx.get("keyframes_mask"); // [T, 1] -> already flat as [T]

    let model = LtxDit::new(cfg, w, None);
    let context_valid = vec![1.0f32; context_len];
    let taps = model.forward(latent, timesteps, positions, keyframes_mask, context, context_len, t, &context_valid);

    // RoPE tables: [heads, T, half] row-major, matching the golden exactly.
    report("rope_cos", &taps.rope_cos, fx.get("rope_cos"), MIN_COS);
    report("rope_sin", &taps.rope_sin, fx.get("rope_sin"), MIN_COS);

    report("adaln_table", &taps.adaln_table, fx.get("adaln_table"), MIN_COS);
    report("embedded_timestep", &taps.embedded_timestep, fx.get("embedded_timestep"), MIN_COS);

    report("b0_attn1_out", &taps.b0_attn1_out, fx.get("b0_attn1_out"), MIN_COS);
    report("b0_attn2_out", &taps.b0_attn2_out, fx.get("b0_attn2_out"), MIN_COS);
    report("b0_ff_out", &taps.b0_ff_out, fx.get("b0_ff_out"), MIN_COS);

    for (i, out) in taps.block_out.iter().enumerate() {
        report(&format!("block.{i}.out"), out, fx.get(&format!("block.{i}.out")), MIN_COS);
    }

    report("out", &taps.out, fx.get("out"), MIN_COS);

    let _ = dim; // shape sanity only, no further use
}

// -------------------------------------------------- gated + connector (M?)

/// `(fixture, weights)` for the gated/connector golden, or `None` with a
/// loud skip - same convention as [`setup`], separate fixture files (this
/// dumper run does not touch `dit_tiny.safetensors`/`dit_tiny_weights.
/// safetensors`, see `tools/goldens/ltxv_dit_dump_reference.py`'s
/// `dump_gated`'s doc).
fn setup_gated() -> Option<(Fixture, vae::blocks::Tensors)> {
    let fx_path = brain_testutil::testdata("golden/ltxv/dit/dit_tiny_gated.safetensors");
    let w_path = brain_testutil::testdata("golden/ltxv/dit/dit_tiny_gated_weights.safetensors");
    if !Path::new(&fx_path).exists() || !Path::new(&w_path).exists() {
        brain_testutil::skip(&format!("fixture {fx_path} absent - run tools/goldens/ltxv_dit_dump_reference.py"));
        return None;
    }
    let t = checkpoint::safetensors::read(&fx_path).expect("read golden");
    let w = load_tiny_weights(&w_path);
    Some((Fixture { t }, w))
}

/// Gated attention (`apply_gated_attention`/`connector_apply_gated_
/// attention: true`) + the video embeddings connector
/// (`use_embeddings_connector: true`), replayed against `tools/goldens/
/// ltxv_dit_dump_reference.py`'s `dump_gated` fixture -
/// [`LtxDitConfig::tiny_gated`], every axis distinct from every other and
/// from [`LtxDitConfig::tiny`]'s (lesson #4). Two things are checked
/// independently, both at cosine >= 0.999999 AND `max_abs < 1e-4`:
///
/// 1. [`EmbeddingsConnector`] alone, run directly on the golden's RAW
///    pre-connector `raw_context`/`context_valid` - proves the connector's
///    OWN forward (register substitution, gated self-attention, RoPE,
///    output norm) independent of the surrounding DiT.
/// 2. [`LtxDit::forward`] given that SAME raw context - `cfg.use_
///    embeddings_connector` routes it through the connector internally, so
///    every existing tap (RoPE, adaLN, block-0 internals, every block's
///    output, the final `out`) now also proves gated attention is wired
///    correctly through the whole block stack (self-, text-cross-
///    attention all read `w.gate`, see `crate::block::attention`'s doc).
#[test]
fn ltxv_dit_tiny_gated_matches_reference() {
    let Some((fx, w)) = setup_gated() else { return };

    let cfg = LtxDitConfig::tiny_gated();
    assert!(cfg.apply_gated_attention && cfg.connector_apply_gated_attention && cfg.use_embeddings_connector);

    let latent = fx.get("latent");
    let t = fx.shape("latent")[0];
    let raw_context = fx.get("raw_context");
    let context_len = fx.shape("raw_context")[0];
    let context_valid = fx.get("context_valid");
    let timesteps = fx.get("timesteps");
    let positions = fx.get("positions");
    let keyframes_mask = fx.get("keyframes_mask");

    // ---- 1: the connector alone -----------------------------------------
    let gpu = open_device(None);
    let connector = EmbeddingsConnector::on(
        gpu,
        &w,
        "video_embeddings_connector",
        cfg.connector_inner_dim(),
        cfg.connector_num_attention_heads,
        cfg.connector_attention_head_dim,
        cfg.connector_num_layers,
        cfg.connector_num_learnable_registers,
        cfg.connector_apply_gated_attention,
        cfg.connector_norm_output,
        cfg.positional_embedding_theta,
        &cfg.connector_positional_embedding_max_pos,
        cfg.norm_eps,
    );
    let connector_out = connector.forward(raw_context, context_valid, context_len as u32);
    report_strict("connector_out", &connector_out, fx.get("connector_out"), MIN_COS);

    // ---- 2: the whole DiT, routing raw_context through the SAME connector
    // internally ------------------------------------------------------------
    let model = LtxDit::new(cfg, w, None);
    let taps = model.forward(latent, timesteps, positions, keyframes_mask, raw_context, context_len, t, context_valid);

    report_strict("rope_cos", &taps.rope_cos, fx.get("rope_cos"), MIN_COS);
    report_strict("rope_sin", &taps.rope_sin, fx.get("rope_sin"), MIN_COS);
    report_strict("adaln_table", &taps.adaln_table, fx.get("adaln_table"), MIN_COS);
    report_strict("embedded_timestep", &taps.embedded_timestep, fx.get("embedded_timestep"), MIN_COS);
    report_strict("connector_out (via forward)", &taps.connector_out, fx.get("connector_out"), MIN_COS);

    report_strict("b0_attn1_out", &taps.b0_attn1_out, fx.get("b0_attn1_out"), MIN_COS);
    report_strict("b0_attn2_out", &taps.b0_attn2_out, fx.get("b0_attn2_out"), MIN_COS);
    report_strict("b0_ff_out", &taps.b0_ff_out, fx.get("b0_ff_out"), MIN_COS);

    for (i, out) in taps.block_out.iter().enumerate() {
        report_strict(&format!("block.{i}.out"), out, fx.get(&format!("block.{i}.out")), MIN_COS);
    }

    report_strict("out", &taps.out, fx.get("out"), MIN_COS);
}
