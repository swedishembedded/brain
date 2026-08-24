// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5 audio+video DiT parity against `tools/goldens/
//! ltxv_av_dit_dump_reference.py`'s tiny (2-layer, video `inner_dim` 64 /
//! audio `inner_dim` 32) fixture.
//!
//! Sibling of `dit_parity.rs` (video-only) - same structure, same
//! `>= 0.999999` cosine bar, but replaying BOTH streams' inputs through
//! [`ltxv::LtxAvDit::forward`] and asserting every tap the AV extension
//! adds: both streams' own self-attention RoPE tables, the SHARED
//! cross-modal RoPE tables, both streams' adaLN-single tables, the four new
//! per-block AV adaLN raw tables, the A2V/V2A attention outputs, both
//! streams' block-0 internal taps, every block's output, and both final
//! outputs.
//!
//! Skips loudly without the fixture (`BRAIN_REQUIRE_FIXTURES=1` upgrades a
//! skip to a failure), matching `dit_parity.rs`'s convention.

use std::path::Path;

use gpu_core::Gpu;
use ltxv::block::{EmbeddingsConnector, KERNELS};
use ltxv::{load_tiny_weights, LtxAvDit, LtxAvDitConfig};

// ------------------------------------------------------------------ metrics

/// Same formula `dit_parity.rs`/`vae_parity.rs`/`model::hostmath::cosine`
/// use (f64 accumulation, both norms as separate factors).
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
/// Same bound `dit_parity.rs`'s gated test uses - see that file's doc for
/// why cross-implementation (PyTorch CPU vs. this crate's GPU dispatch)
/// fp32 accumulation needs a looser absolute bound than a same-device
/// comparison (`shard_parity.rs`'s `1e-4`).
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
    let fx_path = brain_testutil::testdata("golden/ltxv/av_dit/av_dit_tiny.safetensors");
    let w_path = brain_testutil::testdata("golden/ltxv/av_dit/av_dit_tiny_weights.safetensors");
    if !Path::new(&fx_path).exists() || !Path::new(&w_path).exists() {
        brain_testutil::skip(&format!("fixture {fx_path} absent - run tools/goldens/ltxv_av_dit_dump_reference.py"));
        return None;
    }
    let t = checkpoint::safetensors::read(&fx_path).expect("read golden");
    let w = load_tiny_weights(&w_path);
    Some((Fixture { t }, w))
}

#[test]
fn ltxv_av_dit_tiny_matches_reference() {
    let Some((fx, w)) = setup() else { return };

    let cfg = LtxAvDitConfig::tiny();

    let v_latent = fx.get("video.latent");
    let tv = fx.shape("video.latent")[0];
    let v_context = fx.get("video.context");
    let v_context_len = fx.shape("video.context")[0];
    let v_timesteps = fx.get("video.timesteps"); // [Tv, 1] -> already flat as [Tv]
    let v_positions = fx.get("video.positions"); // [3, Tv, 2]
    let v_keyframes_mask = fx.get("video.keyframes_mask"); // [Tv, 1] -> already flat as [Tv]
    let v_sigma = fx.get("video.sigma")[0];

    let a_latent = fx.get("audio.latent");
    let ta = fx.shape("audio.latent")[0];
    let a_context = fx.get("audio.context");
    let a_context_len = fx.shape("audio.context")[0];
    let a_timesteps = fx.get("audio.timesteps"); // [Ta, 1] -> already flat as [Ta]
    let a_positions = fx.get("audio.positions"); // [1, Ta, 2]
    let a_sigma = fx.get("audio.sigma")[0];

    let model = LtxAvDit::new(cfg, w, None);
    let v_context_valid = vec![1.0f32; v_context_len];
    let a_context_valid = vec![1.0f32; a_context_len];
    #[rustfmt::skip]
    let taps = model.forward(
        v_latent, v_timesteps, v_positions, v_keyframes_mask, v_context, v_context_len, tv, v_sigma, &v_context_valid,
        a_latent, a_timesteps, a_positions, a_context, a_context_len, ta, a_sigma, &a_context_valid,
    );

    // ---- self-attention RoPE tables, both streams ----------------------
    report("video.rope_cos", &taps.video.rope_cos, fx.get("video.rope_cos"), MIN_COS);
    report("video.rope_sin", &taps.video.rope_sin, fx.get("video.rope_sin"), MIN_COS);
    report("audio.rope_cos", &taps.audio.rope_cos, fx.get("audio.rope_cos"), MIN_COS);
    report("audio.rope_sin", &taps.audio.rope_sin, fx.get("audio.rope_sin"), MIN_COS);

    // ---- shared cross-modal RoPE tables ---------------------------------
    report("video.cross_rope_cos", &taps.v_cross_rope_cos, fx.get("video.cross_rope_cos"), MIN_COS);
    report("video.cross_rope_sin", &taps.v_cross_rope_sin, fx.get("video.cross_rope_sin"), MIN_COS);
    report("audio.cross_rope_cos", &taps.a_cross_rope_cos, fx.get("audio.cross_rope_cos"), MIN_COS);
    report("audio.cross_rope_sin", &taps.a_cross_rope_sin, fx.get("audio.cross_rope_sin"), MIN_COS);

    // ---- adaLN-single raw tables, both streams --------------------------
    report("video.adaln_table", &taps.video.adaln_table, fx.get("video.adaln_table"), MIN_COS);
    report("video.embedded_timestep", &taps.video.embedded_timestep, fx.get("video.embedded_timestep"), MIN_COS);
    report("audio.adaln_table", &taps.audio.adaln_table, fx.get("audio.adaln_table"), MIN_COS);
    report("audio.embedded_timestep", &taps.audio.embedded_timestep, fx.get("audio.embedded_timestep"), MIN_COS);

    // ---- the four new per-block AV adaLN tables (model-level raw MLP
    // output shared across every block) ------------------------------------
    report("av.video_ss_table", &taps.av_video_ss_table, fx.get("av.video_ss_table"), MIN_COS);
    report("av.audio_ss_table", &taps.av_audio_ss_table, fx.get("av.audio_ss_table"), MIN_COS);
    report("av.a2v_gate_table", &taps.av_a2v_gate_table, fx.get("av.a2v_gate_table"), MIN_COS);
    report("av.v2a_gate_table", &taps.av_v2a_gate_table, fx.get("av.v2a_gate_table"), MIN_COS);

    // ---- block-0 internal taps, both streams + the A2V/V2A attention
    // outputs (RAW, before *gate - see crate::block's BlockTaps doc) -------
    report("video.b0_attn1_out", &taps.video.b0_attn1_out, fx.get("video.b0_attn1_out"), MIN_COS);
    report("video.b0_attn2_out", &taps.video.b0_attn2_out, fx.get("video.b0_attn2_out"), MIN_COS);
    report("video.b0_ff_out", &taps.video.b0_ff_out, fx.get("video.b0_ff_out"), MIN_COS);
    report("audio.b0_attn1_out", &taps.audio.b0_attn1_out, fx.get("audio.b0_attn1_out"), MIN_COS);
    report("audio.b0_attn2_out", &taps.audio.b0_attn2_out, fx.get("audio.b0_attn2_out"), MIN_COS);
    report("audio.b0_ff_out", &taps.audio.b0_ff_out, fx.get("audio.b0_ff_out"), MIN_COS);
    report("av.b0_a2v_out", &taps.b0_a2v_out, fx.get("av.b0_a2v_out"), MIN_COS);
    report("av.b0_v2a_out", &taps.b0_v2a_out, fx.get("av.b0_v2a_out"), MIN_COS);

    // ---- every block's output, both streams -----------------------------
    for (i, out) in taps.video.block_out.iter().enumerate() {
        report(&format!("video.block.{i}.out"), out, fx.get(&format!("video.block.{i}.out")), MIN_COS);
    }
    for (i, out) in taps.audio.block_out.iter().enumerate() {
        report(&format!("audio.block.{i}.out"), out, fx.get(&format!("audio.block.{i}.out")), MIN_COS);
    }

    // ---- final outputs, both streams -------------------------------------
    report("video.out", &taps.video.out, fx.get("video.out"), MIN_COS);
    report("audio.out", &taps.audio.out, fx.get("audio.out"), MIN_COS);
}

// -------------------------------------------------- gated + connectors (M?)

/// `(fixture, weights)` for the gated/connector AV golden, or `None` with a
/// loud skip - separate fixture files from [`setup`]'s (this dumper run
/// does not touch `av_dit_tiny.safetensors`/`av_dit_tiny_weights.
/// safetensors`, see `tools/goldens/ltxv_av_dit_dump_reference.py`'s
/// `dump_gated`'s doc).
fn setup_gated() -> Option<(Fixture, vae::blocks::Tensors)> {
    let fx_path = brain_testutil::testdata("golden/ltxv/av_dit/av_dit_tiny_gated.safetensors");
    let w_path = brain_testutil::testdata("golden/ltxv/av_dit/av_dit_tiny_gated_weights.safetensors");
    if !Path::new(&fx_path).exists() || !Path::new(&w_path).exists() {
        brain_testutil::skip(&format!("fixture {fx_path} absent - run tools/goldens/ltxv_av_dit_dump_reference.py"));
        return None;
    }
    let t = checkpoint::safetensors::read(&fx_path).expect("read golden");
    let w = load_tiny_weights(&w_path);
    Some((Fixture { t }, w))
}

/// Gated attention + BOTH embeddings connectors on the AV path, replayed
/// against `tools/goldens/ltxv_av_dit_dump_reference.py`'s `dump_gated`
/// fixture - [`LtxAvDitConfig::tiny_gated`]. Same two-part check as
/// `dit_parity.rs`'s `ltxv_dit_tiny_gated_matches_reference`, doubled (one
/// [`EmbeddingsConnector`] per stream, own weight prefix/geometry - video's
/// own `connector_*` fields drive BOTH connectors' shared layer/register/
/// max-pos/gate/norm-output configuration, only per-stream head geometry
/// differs, matching `crate::dit::route_context_through_connector`'s two
/// call sites in `LtxAvDit::forward`).
#[test]
fn ltxv_av_dit_tiny_gated_matches_reference() {
    let Some((fx, w)) = setup_gated() else { return };

    let cfg = LtxAvDitConfig::tiny_gated();
    cfg.assert_supported();
    assert!(cfg.video.apply_gated_attention && cfg.video.connector_apply_gated_attention && cfg.video.use_embeddings_connector);

    let v_latent = fx.get("video.latent");
    let tv = fx.shape("video.latent")[0];
    let v_raw_context = fx.get("video.raw_context");
    let v_context_len = fx.shape("video.raw_context")[0];
    let v_context_valid = fx.get("video.context_valid");
    let v_timesteps = fx.get("video.timesteps");
    let v_positions = fx.get("video.positions");
    let v_keyframes_mask = fx.get("video.keyframes_mask");
    let v_sigma = fx.get("video.sigma")[0];

    let a_latent = fx.get("audio.latent");
    let ta = fx.shape("audio.latent")[0];
    let a_raw_context = fx.get("audio.raw_context");
    let a_context_len = fx.shape("audio.raw_context")[0];
    let a_context_valid = fx.get("audio.context_valid");
    let a_timesteps = fx.get("audio.timesteps");
    let a_positions = fx.get("audio.positions");
    let a_sigma = fx.get("audio.sigma")[0];

    // ---- 1: each connector alone -----------------------------------------
    #[rustfmt::skip]
    let video_connector = EmbeddingsConnector::on(
        Gpu::open(None, &KERNELS), &w, "video_embeddings_connector",
        cfg.video.connector_inner_dim(), cfg.video.connector_num_attention_heads, cfg.video.connector_attention_head_dim,
        cfg.video.connector_num_layers, cfg.video.connector_num_learnable_registers, cfg.video.connector_apply_gated_attention,
        cfg.video.connector_norm_output, cfg.video.positional_embedding_theta, &cfg.video.connector_positional_embedding_max_pos, cfg.video.norm_eps,
    );
    let v_connector_out = video_connector.forward(v_raw_context, v_context_valid, v_context_len as u32);
    report_strict("video.connector_out", &v_connector_out, fx.get("video.connector_out"), MIN_COS);

    #[rustfmt::skip]
    let audio_connector = EmbeddingsConnector::on(
        Gpu::open(None, &KERNELS), &w, "audio_embeddings_connector",
        cfg.audio.connector_inner_dim(), cfg.audio.connector_num_attention_heads, cfg.audio.connector_attention_head_dim,
        cfg.video.connector_num_layers, cfg.video.connector_num_learnable_registers, cfg.video.connector_apply_gated_attention,
        cfg.video.connector_norm_output, cfg.video.positional_embedding_theta, &cfg.video.connector_positional_embedding_max_pos, cfg.video.norm_eps,
    );
    let a_connector_out = audio_connector.forward(a_raw_context, a_context_valid, a_context_len as u32);
    report_strict("audio.connector_out", &a_connector_out, fx.get("audio.connector_out"), MIN_COS);

    // ---- 2: the whole AV DiT, routing both raw contexts through the SAME
    // connectors internally ---------------------------------------------
    let model = LtxAvDit::new(cfg, w, None);
    #[rustfmt::skip]
    let taps = model.forward(
        v_latent, v_timesteps, v_positions, v_keyframes_mask, v_raw_context, v_context_len, tv, v_sigma, v_context_valid,
        a_latent, a_timesteps, a_positions, a_raw_context, a_context_len, ta, a_sigma, a_context_valid,
    );

    report_strict("video.rope_cos", &taps.video.rope_cos, fx.get("video.rope_cos"), MIN_COS);
    report_strict("video.rope_sin", &taps.video.rope_sin, fx.get("video.rope_sin"), MIN_COS);
    report_strict("audio.rope_cos", &taps.audio.rope_cos, fx.get("audio.rope_cos"), MIN_COS);
    report_strict("audio.rope_sin", &taps.audio.rope_sin, fx.get("audio.rope_sin"), MIN_COS);
    report_strict("video.cross_rope_cos", &taps.v_cross_rope_cos, fx.get("video.cross_rope_cos"), MIN_COS);
    report_strict("video.cross_rope_sin", &taps.v_cross_rope_sin, fx.get("video.cross_rope_sin"), MIN_COS);
    report_strict("audio.cross_rope_cos", &taps.a_cross_rope_cos, fx.get("audio.cross_rope_cos"), MIN_COS);
    report_strict("audio.cross_rope_sin", &taps.a_cross_rope_sin, fx.get("audio.cross_rope_sin"), MIN_COS);

    report_strict("video.adaln_table", &taps.video.adaln_table, fx.get("video.adaln_table"), MIN_COS);
    report_strict("video.embedded_timestep", &taps.video.embedded_timestep, fx.get("video.embedded_timestep"), MIN_COS);
    report_strict("audio.adaln_table", &taps.audio.adaln_table, fx.get("audio.adaln_table"), MIN_COS);
    report_strict("audio.embedded_timestep", &taps.audio.embedded_timestep, fx.get("audio.embedded_timestep"), MIN_COS);

    report_strict("video.connector_out (via forward)", &taps.video.connector_out, fx.get("video.connector_out"), MIN_COS);
    report_strict("audio.connector_out (via forward)", &taps.audio.connector_out, fx.get("audio.connector_out"), MIN_COS);

    report_strict("av.video_ss_table", &taps.av_video_ss_table, fx.get("av.video_ss_table"), MIN_COS);
    report_strict("av.audio_ss_table", &taps.av_audio_ss_table, fx.get("av.audio_ss_table"), MIN_COS);
    report_strict("av.a2v_gate_table", &taps.av_a2v_gate_table, fx.get("av.a2v_gate_table"), MIN_COS);
    report_strict("av.v2a_gate_table", &taps.av_v2a_gate_table, fx.get("av.v2a_gate_table"), MIN_COS);

    report_strict("video.b0_attn1_out", &taps.video.b0_attn1_out, fx.get("video.b0_attn1_out"), MIN_COS);
    report_strict("video.b0_attn2_out", &taps.video.b0_attn2_out, fx.get("video.b0_attn2_out"), MIN_COS);
    report_strict("video.b0_ff_out", &taps.video.b0_ff_out, fx.get("video.b0_ff_out"), MIN_COS);
    report_strict("audio.b0_attn1_out", &taps.audio.b0_attn1_out, fx.get("audio.b0_attn1_out"), MIN_COS);
    report_strict("audio.b0_attn2_out", &taps.audio.b0_attn2_out, fx.get("audio.b0_attn2_out"), MIN_COS);
    report_strict("audio.b0_ff_out", &taps.audio.b0_ff_out, fx.get("audio.b0_ff_out"), MIN_COS);
    report_strict("av.b0_a2v_out", &taps.b0_a2v_out, fx.get("av.b0_a2v_out"), MIN_COS);
    report_strict("av.b0_v2a_out", &taps.b0_v2a_out, fx.get("av.b0_v2a_out"), MIN_COS);

    for (i, out) in taps.video.block_out.iter().enumerate() {
        report_strict(&format!("video.block.{i}.out"), out, fx.get(&format!("video.block.{i}.out")), MIN_COS);
    }
    for (i, out) in taps.audio.block_out.iter().enumerate() {
        report_strict(&format!("audio.block.{i}.out"), out, fx.get(&format!("audio.block.{i}.out")), MIN_COS);
    }

    report_strict("video.out", &taps.video.out, fx.get("video.out"), MIN_COS);
    report_strict("audio.out", &taps.audio.out, fx.get("audio.out"), MIN_COS);
}
