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

// ---------------------------------------------------------------------------
// The int8 COMPUTE tier for the audio+video block (`ltxv::block::
// LtxAvBlockQ`), held to the same bar the video-only tier is held to by
// `crates/ltxv/tests/int8_compute.rs`.
//
// int8 is LOSSY, so every comparison below asserts cosine AND relative L2.
// Cosine alone is scale-invariant: a uniform gain on this model measured
// cosine 1.0000000000 with only rel_l2 moving, so a cosine-only gate would
// pass a systematically wrong output.
//
// Three properties, deliberately independent of each other:
//
// 1. the quantized block tracks the fp32 block (a tolerance gate);
// 2. deriving the modulation ON the device is BIT-IDENTICAL to uploading it
//    from the host (an exact gate - this is a routing change, not an
//    arithmetic one, so anything but equality is a bug);
// 3. a CLOSED cross-modal gate makes the gated stream exactly independent of
//    the other one (an analytic gate, no tolerance at all).
//
// (3) is the one this whole change most needs. Every attention here is gated,
// and an always-open or dropped cross-attention gate produces plausible
// output that is wrong - which neither (1) nor (2) can see, because both
// compare two implementations that would carry the same mistake.
// ---------------------------------------------------------------------------

use ltxv::block::{AvRope, LtxAvBlock, LtxAvBlockQ, QTier};
use ltxv::dit::random_av_tiny_weights;

/// Relative L2, `||got - want|| / ||want||` - the metric cosine cannot see.
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let num: f64 = got.iter().zip(want).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|&y| (y as f64).powi(2)).sum();
    if den <= 0.0 {
        0.0
    } else {
        (num / den).sqrt()
    }
}

fn report_lossy(label: &str, got: &[f32], want: &[f32], min_cos: f64, max_rel: f64) {
    let (c, r, m) = (cosine(got, want), rel_l2(got, want), max_abs(got, want));
    eprintln!("{label}: cosine={c:.10}  rel_l2={r:.3e}  max_abs={m:.3e}  n={}", got.len());
    assert!(c >= min_cos, "{label}: cosine {c:.10} < {min_cos}");
    assert!(r <= max_rel, "{label}: rel_l2 {r:.3e} > {max_rel:.3e}");
}

fn upload_rope(gpu: &Gpu, rope: &ltxv::rope::LtxRopeTables) -> (Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>) {
    let (mut cos_bufs, mut sin_bufs) = (Vec::with_capacity(rope.heads), Vec::with_capacity(rope.heads));
    for h in 0..rope.heads {
        let (c, s) = rope.head(h);
        let cb = gpu.storage(c.len() as u64);
        gpu.write_f32(&cb, c);
        let sb = gpu.storage(s.len() as u64);
        gpu.write_f32(&sb, s);
        cos_bufs.push(cb);
        sin_bufs.push(sb);
    }
    (cos_bufs, sin_bufs)
}

/// A deterministic ramp - shape-correct scratch with no RNG dependency, so
/// two runs of this file compare the same numbers.
fn ramp(n: usize, period: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * scale).collect()
}

/// Positions for `ltxv::rope::ltx_rope_tables`: `[axes, t, 2]`, each token a
/// `[start, end)` pair on its own axis.
fn positions(axes: usize, t: usize) -> Vec<f32> {
    let mut p = vec![0f32; axes * t * 2];
    for a in 0..axes {
        for i in 0..t {
            p[(a * t + i) * 2] = (i % 7) as f32 * 0.25;
            p[(a * t + i) * 2 + 1] = (i % 7) as f32 * 0.25 + 0.25;
        }
    }
    p
}

/// Everything one AV block forward needs, at a tiny config - built once and
/// replayed through as many block implementations as a test wants.
struct AvBlockCase {
    cfg: LtxAvDitConfig,
    w: vae::blocks::Tensors,
    vx: Vec<f32>,
    ax: Vec<f32>,
    v_adaln: Vec<f32>,
    a_adaln: Vec<f32>,
    v_context: Vec<f32>,
    a_context: Vec<f32>,
    v_ss: Vec<f32>,
    a_ss: Vec<f32>,
    a2v_gate: Vec<f32>,
    v2a_gate: Vec<f32>,
    tv: u32,
    ta: u32,
    v_ctx_len: u32,
    a_ctx_len: u32,
}

impl AvBlockCase {
    fn tiny(seed: u64) -> AvBlockCase {
        let cfg = LtxAvDitConfig::tiny_gated();
        cfg.assert_supported();
        assert!(cfg.video.apply_gated_attention, "this suite must exercise the GATED attention path");
        let (vdim, adim) = (cfg.video.inner_dim as usize, cfg.audio.inner_dim as usize);
        let (tv, ta) = (11usize, 7usize);
        let (v_ctx_len, a_ctx_len) = (5usize, 3usize);
        AvBlockCase {
            w: random_av_tiny_weights(&cfg, seed),
            vx: ramp(tv * vdim, 23, 1.1),
            ax: ramp(ta * adim, 29, 0.9),
            v_adaln: ramp(tv * 9 * vdim, 13, 0.3),
            a_adaln: ramp(ta * 9 * adim, 17, 0.3),
            v_context: ramp(v_ctx_len * vdim, 7, 1.4),
            a_context: ramp(a_ctx_len * adim, 11, 1.2),
            v_ss: ramp(tv * 4 * vdim, 19, 0.3),
            a_ss: ramp(ta * 4 * adim, 31, 0.3),
            a2v_gate: ramp(vdim, 5, 0.6),
            v2a_gate: ramp(adim, 5, 0.6),
            tv: tv as u32,
            ta: ta as u32,
            v_ctx_len: v_ctx_len as u32,
            a_ctx_len: a_ctx_len as u32,
            cfg,
        }
    }

    /// This case's four RoPE table sets, uploaded to `gpu`.
    #[allow(clippy::type_complexity)]
    fn rope(&self, gpu: &Gpu) -> (Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>) {
        let (v, a) = (&self.cfg.video, &self.cfg.audio);
        let vp = positions(3, self.tv as usize);
        let ap = positions(1, self.ta as usize);
        let cross_max = [self.cfg.cross_pe_max_pos()];
        let vr = ltxv::rope::ltx_rope_tables(v.inner_dim, v.num_heads, v.positional_embedding_theta, &v.positional_embedding_max_pos, &vp, self.tv as usize);
        let ar = ltxv::rope::ltx_rope_tables(a.inner_dim, a.num_heads, v.positional_embedding_theta, &a.positional_embedding_max_pos, &ap, self.ta as usize);
        let vcr = ltxv::rope::ltx_rope_tables(a.cross_attention_dim, a.num_heads, v.positional_embedding_theta, &cross_max, &vp[0..self.tv as usize * 2], self.tv as usize);
        let acr = ltxv::rope::ltx_rope_tables(a.cross_attention_dim, a.num_heads, v.positional_embedding_theta, &cross_max, &ap, self.ta as usize);
        let (vc, vs) = upload_rope(gpu, &vr);
        let (ac, as_) = upload_rope(gpu, &ar);
        let (vcc, vcs) = upload_rope(gpu, &vcr);
        let (acc, acs) = upload_rope(gpu, &acr);
        (vc, vs, ac, as_, vcc, vcs, acc, acs)
    }
}

/// The lossy tier tracks the fp32 one, on BOTH streams and on BOTH raw
/// cross-attention outputs.
///
/// The floors are set with headroom below what a clean run measures, not at
/// it - an int8 tier is not expected to be 1.0 and a gate pinned to a
/// measurement becomes a flake the first time a kernel reassociates a sum.
/// The video-only tier's own real-weight block-0 number is around 0.9963
/// cosine, and this tiny config's i.i.d. weights have far less per-channel
/// dynamic range than a real checkpoint, so it measures much closer to 1: a
/// clean run is cosine 0.99999999996 / rel_l2 3.0e-5 on both stream outputs,
/// and the tightest of the internal taps is `video attn1_out` at 0.99984 /
/// 1.80e-2, comfortably under its own rel_l2 ceiling of 5e-2.
///
/// Mutation-verified: forcing every `to_gate_logits` attention gate OPEN in
/// the int8 path turns this RED on `video attn1_out`'s COSINE (0.99812 <
/// 0.999) while both stream OUTPUTS stay at cosine 1.0 to nine places - which
/// is why the internal taps are compared separately and not just the outputs.
/// A 5% uniform GAIN on the int8 outputs turns it RED on rel_l2 alone
/// (5.0e-2 > 3.0e-2) with cosine unchanged in all ten printed digits - the
/// direct demonstration that a cosine-only gate would have passed a
/// systematically wrong result. Forcing the A2V CROSS-MODAL gate open does
/// NOT turn this red (it moves `video out` rel_l2 from 3.0e-5 to 6.0e-3,
/// still inside a floor sized for a lossy tier); that fault is
/// `a_closed_cross_modal_gate_...`'s job, which catches it exactly.
#[test]
fn av_int8_block_tracks_the_fp32_av_block() {
    let c = AvBlockCase::tiny(0x00A0_D108);
    let (vcfg, acfg) = (&c.cfg.video, &c.cfg.audio);

    let gpu_f32 = Gpu::open(None, &KERNELS);
    let r = c.rope(&gpu_f32);
    let blk = LtxAvBlock::on(gpu_f32.share(), vcfg, acfg, &c.w, "transformer_blocks.0", c.v_ctx_len, c.a_ctx_len);
    #[rustfmt::skip]
    let (vf, af, tf) = blk.forward(&c.vx, &c.ax, &c.v_adaln, &c.a_adaln, &c.v_context, &c.a_context,
        &r.0, &r.1, &r.2, &r.3, &r.4, &r.5, &r.6, &r.7,
        &c.v_ss, &c.a_ss, &c.a2v_gate, &c.v2a_gate, c.tv, c.ta, true);

    let gpu_q = Gpu::open(None, &KERNELS);
    let rq = c.rope(&gpu_q);
    let rope = AvRope { v_cos: &rq.0, v_sin: &rq.1, a_cos: &rq.2, a_sin: &rq.3, v_cross_cos: &rq.4, v_cross_sin: &rq.5, a_cross_cos: &rq.6, a_cross_sin: &rq.7 };
    let blkq = LtxAvBlockQ::on(gpu_q.share(), vcfg, acfg, &c.w, "transformer_blocks.0", c.v_ctx_len, c.a_ctx_len, QTier::Int8);
    #[rustfmt::skip]
    let (vq, aq, tq) = blkq.forward(&c.vx, &c.ax, &c.v_adaln, &c.a_adaln, &c.v_context, &c.a_context, rope,
        &c.v_ss, &c.a_ss, &c.a2v_gate, &c.v2a_gate, c.tv, c.ta);

    report_lossy("av int8 video out", &vq, &vf, 0.9995, 3e-2);
    report_lossy("av int8 audio out", &aq, &af, 0.9995, 3e-2);
    report_lossy("av int8 a2v_out (raw, pre-gate)", &tq.a2v_out, &tf.a2v_out, 0.999, 5e-2);
    report_lossy("av int8 v2a_out (raw, pre-gate)", &tq.v2a_out, &tf.v2a_out, 0.999, 5e-2);
    report_lossy("av int8 video attn1_out", &tq.v_attn1_out, &tf.v_attn1_out, 0.999, 5e-2);
    report_lossy("av int8 audio attn1_out", &tq.a_attn1_out, &tf.a_attn1_out, 0.999, 5e-2);
}

/// Deriving every modulation vector ON the card (the production path) must be
/// BIT-IDENTICAL to combining and uploading it from the host (the reference
/// path). This is a routing change - `adaln_row` reproduces `add_table`'s
/// operand order and `slice_mod`'s `1.0 + x` exactly, and the two cross-modal
/// gates are one `add2` over the same two rows `av_gate` adds - so anything
/// but equality means the routing changed a number, which is the one thing it
/// must not do.
///
/// Asserted on BITS, not on `==`: two NaNs compare unequal and are the same
/// answer, and `0.0 == -0.0` is true for two different results.
#[test]
fn device_derived_av_modulation_is_bit_identical_to_the_host_uploaded_form() {
    let c = AvBlockCase::tiny(0x0DE8_17ED);
    let (vcfg, acfg) = (&c.cfg.video, &c.cfg.audio);
    let (vdim, adim) = (vcfg.inner_dim as usize, acfg.inner_dim as usize);

    let gpu = Gpu::open(None, &KERNELS);
    let r = c.rope(&gpu);
    let rope = AvRope { v_cos: &r.0, v_sin: &r.1, a_cos: &r.2, a_sin: &r.3, v_cross_cos: &r.4, v_cross_sin: &r.5, a_cross_cos: &r.6, a_cross_sin: &r.7 };
    let blk = LtxAvBlockQ::on(gpu.share(), vcfg, acfg, &c.w, "transformer_blocks.0", c.v_ctx_len, c.a_ctx_len, QTier::Int8);

    #[rustfmt::skip]
    let (v_host, a_host, _) = blk.forward(&c.vx, &c.ax, &c.v_adaln, &c.a_adaln, &c.v_context, &c.a_context, rope,
        &c.v_ss, &c.a_ss, &c.a2v_gate, &c.v2a_gate, c.tv, c.ta);

    // The production path's inputs: the same tables, as DENSE `RowTable`s
    // (one row per token, i.e. the dedup finds nothing) plus their row maps.
    // Dense on purpose - the dedup is gated by `dit::adaln`'s own bit-identity
    // test, and this gate is about the DERIVATION, so the harder case (every
    // row distinct, every gather a different index) is the right one here.
    let up = |v: &[f32]| {
        let b = gpu.storage(v.len() as u64);
        gpu.write_f32(&b, v);
        b
    };
    let table = |v: &[f32], t: usize, width: usize| {
        let rt = dit::adaln::RowTable::dense(v.to_vec(), width);
        assert_eq!(rt.len(), t);
        let b = gpu.storage(rt.distinct().len() as u64);
        gpu.write_f32(&b, rt.distinct());
        let m = gpu.storage(rt.row_of().len() as u64);
        gpu.write_at(&m, 0, rt.row_of());
        (b, m)
    };
    let (v_adaln_b, v_adaln_m) = table(&c.v_adaln, c.tv as usize, 9 * vdim);
    let (a_adaln_b, a_adaln_m) = table(&c.a_adaln, c.ta as usize, 9 * adim);
    let (v_ss_b, v_ss_m) = table(&c.v_ss, c.tv as usize, 4 * vdim);
    let (a_ss_b, a_ss_m) = table(&c.a_ss, c.ta as usize, 4 * adim);
    let (a2v_b, v2a_b) = (up(&c.a2v_gate), up(&c.v2a_gate));
    let (v_ctx_b, a_ctx_b) = (up(&c.v_context), up(&c.a_context));
    let av = ltxv::block::AvModelTables { v_ss: &v_ss_b, v_ss_map: &v_ss_m, a_ss: &a_ss_b, a_ss_map: &a_ss_m, a2v_gate: &a2v_b, v2a_gate: &v2a_b };
    let mut timings = ltxv::block::BlockTimings::default();
    #[rustfmt::skip]
    let (v_dev, a_dev) = blk.forward_prod(&gpu, &c.vx, &c.ax, &v_adaln_b, &v_adaln_m, &a_adaln_b, &a_adaln_m, &av,
        &v_ctx_b, &a_ctx_b, rope, c.tv, c.ta, &mut timings);

    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    assert_eq!(bits(&v_dev), bits(&v_host), "device-derived video modulation changed a bit (max_abs {:.3e})", max_abs(&v_dev, &v_host));
    assert_eq!(bits(&a_dev), bits(&a_host), "device-derived audio modulation changed a bit (max_abs {:.3e})", max_abs(&a_dev, &a_host));
    eprintln!("device-derived AV modulation: bit-identical on both streams ({} + {} values)", v_dev.len(), a_dev.len());
}

/// A CLOSED cross-modal gate must make the gated stream EXACTLY independent
/// of the other stream - and an open one must not.
///
/// This is the gate the whole audio-visual change turns on, and it is
/// analytic rather than a tolerance: with `gate_a2v == 0`, `gate_row`
/// computes `vx2 = vx1 + 0 * a2v_out = vx1`, so the video stream's output
/// cannot depend on a single audio value. Feed two DIFFERENT audio latents
/// and the video output must be BIT-IDENTICAL. Then open the gate and it must
/// not be.
///
/// What that catches, which a comparison of two implementations cannot: an
/// always-open gate (both arms differ), a dropped gate multiply (both arms
/// differ), a gate wired to the wrong stream (the closed arm differs), and a
/// cross-attention reading the wrong operand (the OPEN arm stops differing).
/// Run on the quantized block, because that is the new code; the fp32 block
/// is checked the same way in the same test so a divergence between them
/// cannot hide here either.
///
/// Mutation-verified with exactly that fault: replacing the A2V `gate_row`
/// with a plain add - an always-open cross-modal gate - turns this RED on the
/// CLOSED arm's bit equality (the two audio latents move the video output by
/// max_abs 1.9e-3 where the answer must be exactly 0). No tolerance gate in
/// this file catches that mutation at floors sized for a lossy tier; the
/// real-weight block-0 gate does, at cosine 0.595, and nothing else does.
#[test]
fn a_closed_cross_modal_gate_makes_the_gated_stream_independent_of_the_other() {
    let mut c = AvBlockCase::tiny(0x0006_A7E0);
    let (vcfg, acfg) = (c.cfg.video, c.cfg.audio);
    let (vdim, adim) = (vcfg.inner_dim as usize, acfg.inner_dim as usize);
    let ax_other = ramp(c.ta as usize * adim, 37, 1.7);
    assert_ne!(c.ax, ax_other, "the two audio latents must actually differ");

    // Close BOTH gates exactly: the block's own row 4 AND the model-level
    // row. `gate = table5[row4] + model_row`, so both have to be zero for the
    // sum to be, and zeroing only one would leave a test that passes for the
    // wrong reason.
    for (name, dim) in [("scale_shift_table_a2v_ca_video", vdim), ("scale_shift_table_a2v_ca_audio", adim)] {
        let key = format!("transformer_blocks.0.{name}");
        let e = c.w.get_mut(&key).unwrap_or_else(|| panic!("no {key}"));
        for v in e.1[4 * dim..5 * dim].iter_mut() {
            *v = 0.0;
        }
    }
    let zeros_v = vec![0f32; vdim];
    let zeros_a = vec![0f32; adim];

    let run = |w: &vae::blocks::Tensors, ax: &[f32], a2v_gate: &[f32], v2a_gate: &[f32]| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let gpu = Gpu::open(None, &KERNELS);
        let r = c.rope(&gpu);
        let rope = AvRope { v_cos: &r.0, v_sin: &r.1, a_cos: &r.2, a_sin: &r.3, v_cross_cos: &r.4, v_cross_sin: &r.5, a_cross_cos: &r.6, a_cross_sin: &r.7 };
        let q = LtxAvBlockQ::on(gpu.share(), &vcfg, &acfg, w, "transformer_blocks.0", c.v_ctx_len, c.a_ctx_len, QTier::Int8);
        #[rustfmt::skip]
        let (vq, aq, _) = q.forward(&c.vx, ax, &c.v_adaln, &c.a_adaln, &c.v_context, &c.a_context, rope,
            &c.v_ss, &c.a_ss, a2v_gate, v2a_gate, c.tv, c.ta);
        let f = LtxAvBlock::on(gpu.share(), &vcfg, &acfg, w, "transformer_blocks.0", c.v_ctx_len, c.a_ctx_len);
        #[rustfmt::skip]
        let (vf, af, _) = f.forward(&c.vx, ax, &c.v_adaln, &c.a_adaln, &c.v_context, &c.a_context,
            &r.0, &r.1, &r.2, &r.3, &r.4, &r.5, &r.6, &r.7,
            &c.v_ss, &c.a_ss, a2v_gate, v2a_gate, c.tv, c.ta, false);
        (vq, aq, vf, af)
    };

    // ---- closed: the video stream must not see the audio latent at all ----
    let (vq0, _, vf0, _) = run(&c.w, &c.ax, &zeros_v, &zeros_a);
    let (vq1, _, vf1, _) = run(&c.w, &ax_other, &zeros_v, &zeros_a);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    assert_eq!(bits(&vq0), bits(&vq1), "int8: a CLOSED A2V gate still let the audio latent reach the video stream (max_abs {:.3e})", max_abs(&vq0, &vq1));
    assert_eq!(bits(&vf0), bits(&vf1), "fp32: a CLOSED A2V gate still let the audio latent reach the video stream (max_abs {:.3e})", max_abs(&vf0, &vf1));

    // ---- open: it must ------------------------------------------------------
    let (vq2, _, vf2, _) = run(&c.w, &c.ax, &c.a2v_gate, &c.v2a_gate);
    let (vq3, _, vf3, _) = run(&c.w, &ax_other, &c.a2v_gate, &c.v2a_gate);
    let (dq, df) = (max_abs(&vq2, &vq3), max_abs(&vf2, &vf3));
    eprintln!("open A2V gate: video output moves by max_abs {dq:.3e} (int8) / {df:.3e} (fp32) when the audio latent changes");
    assert!(dq > 1e-6, "int8: an OPEN A2V gate left the video stream unmoved by a different audio latent - the cross-attention is not connected");
    assert!(df > 1e-6, "fp32: an OPEN A2V gate left the video stream unmoved by a different audio latent - the cross-attention is not connected");
    // And the closed arm must genuinely differ from the open one, or "closed"
    // was a no-op rather than a gate.
    assert!(max_abs(&vq0, &vq2) > 1e-6, "int8: closing the A2V gate changed nothing at all");
    assert!(max_abs(&vf0, &vf2) > 1e-6, "fp32: closing the A2V gate changed nothing at all");
}

// ------------------------------------------- real-weight AV int8 tier ------

const REPO: &str = "Lightricks/LTX-2.5";

/// The real distilled Q8_0 DiT GGUF, if this box has one. Discriminated on
/// the file's OWN declared architecture rather than on its name - the model
/// store legitimately holds several Q8_0 GGUFs for this repo (the DiT and the
/// Gemma-4 text encoder), and a name glob picks whichever sorts first.
fn real_dit_gguf() -> Option<String> {
    if let Ok(p) = std::env::var("BRAIN_LTXV_DIT") {
        if !p.is_empty() {
            return Some(p);
        }
    }
    let dir = brain_testutil::model_dir(REPO)?;
    let mut found: Vec<String> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains("Q8_0") && n.ends_with(".gguf")))
        .filter(|p| {
            checkpoint::gguf::MmapGguf::open(&p.to_string_lossy()).ok().and_then(|g| g.kv().get("general.architecture").and_then(|v| v.as_str()).map(str::to_string)).as_deref() == Some(ltxv::import::GGUF_ARCHITECTURE)
        })
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    found.sort();
    found.into_iter().next()
}

/// The AV counterpart of `int8_compute.rs::real_q8_0_block0_int8_compute_
/// matches_fp32`, and the number that actually decides whether this tier is
/// shippable: block 0 of the REAL 22B audio+video checkpoint, run once fp32
/// and once int8, on both streams and both cross-modal directions.
///
/// Real weights, not i.i.d. noise, is the whole point - a real checkpoint's
/// per-channel dynamic range is what int8 has to survive, and the video-only
/// tier measures around 0.9963 cosine on exactly this comparison. The AV tier
/// measures cosine 0.99858 / rel_l2 5.63e-2 on video and 0.99887 / 4.83e-2 on
/// audio, with both raw cross-modal outputs above 0.9995. The floors below sit
/// under that with headroom - a cosine floor of 0.99 against a measured
/// 0.99858, and a rel_l2 ceiling of 1.5e-1 against a measured 5.63e-2 - per
/// this file's own doc, and rel_l2 is asserted alongside because cosine
/// cannot see a systematic gain.
///
/// Mutation-verified: an always-OPEN A2V cross-modal gate turns this RED at
/// cosine 0.595, and an always-open `to_gate_logits` attention gate at cosine
/// 0.709 - both on the video output. This is the gate with the most authority
/// over the tier, because it is the only tolerance gate here whose weights are
/// the checkpoint's own.
///
/// Skips loudly without the checkpoint (`BRAIN_REQUIRE_FIXTURES=1` upgrades a
/// skip to a failure).
#[test]
fn real_q8_0_av_block0_int8_compute_matches_fp32() {
    let Some(path) = real_dit_gguf() else {
        brain_testutil::skip(&format!("set BRAIN_LTXV_DIT to a real {REPO} distilled Q8_0 GGUF (none in the model store)"));
        return;
    };
    let cfg = LtxAvDitConfig::ltx25();
    let (vcfg, acfg) = (cfg.video, cfg.audio);
    let (vdim, adim) = (vcfg.inner_dim as usize, acfg.inner_dim as usize);
    let src = ltxv::gguf_src::LtxvGgufSource::open(&path).unwrap_or_else(|e| panic!("opening {path}: {e}"));
    let w = ltxv::block::load_av_block_tensors_from_source(&src, &cfg, "transformer_blocks.0");

    // A small shape budget, matching `int8_compute.rs`'s own: the tier's
    // per-channel error does not depend on the token count, and a real-width
    // fp32 AV block does not fit alongside its int8 twin on one card.
    let (tv, ta) = (8u32, 6u32);
    let (v_ctx_len, a_ctx_len) = (4u32, 4u32);
    let vx = ramp(tv as usize * vdim, 29, 0.2);
    let ax = ramp(ta as usize * adim, 31, 0.2);
    let v_adaln = ramp(tv as usize * 9 * vdim, 13, 0.3);
    let a_adaln = ramp(ta as usize * 9 * adim, 17, 0.3);
    let v_context = ramp(v_ctx_len as usize * vdim, 7, 1.4);
    let a_context = ramp(a_ctx_len as usize * adim, 11, 1.2);
    let v_ss = ramp(tv as usize * 4 * vdim, 19, 0.3);
    let a_ss = ramp(ta as usize * 4 * adim, 23, 0.3);
    let a2v_gate = ramp(vdim, 5, 0.6);
    let v2a_gate = ramp(adim, 5, 0.6);

    let vp = positions(3, tv as usize);
    let ap = positions(1, ta as usize);
    let cross_max = [cfg.cross_pe_max_pos()];
    let vr = ltxv::rope::ltx_rope_tables(vcfg.inner_dim, vcfg.num_heads, vcfg.positional_embedding_theta, &vcfg.positional_embedding_max_pos, &vp, tv as usize);
    let ar = ltxv::rope::ltx_rope_tables(acfg.inner_dim, acfg.num_heads, vcfg.positional_embedding_theta, &acfg.positional_embedding_max_pos, &ap, ta as usize);
    let vcr = ltxv::rope::ltx_rope_tables(acfg.cross_attention_dim, acfg.num_heads, vcfg.positional_embedding_theta, &cross_max, &vp[0..tv as usize * 2], tv as usize);
    let acr = ltxv::rope::ltx_rope_tables(acfg.cross_attention_dim, acfg.num_heads, vcfg.positional_embedding_theta, &cross_max, &ap, ta as usize);

    let gpu_f32 = Gpu::open(None, &KERNELS);
    let (vc, vs) = upload_rope(&gpu_f32, &vr);
    let (ac, as_) = upload_rope(&gpu_f32, &ar);
    let (vcc, vcs) = upload_rope(&gpu_f32, &vcr);
    let (acc, acs) = upload_rope(&gpu_f32, &acr);
    let blk = LtxAvBlock::on(gpu_f32.share(), &vcfg, &acfg, &w, "transformer_blocks.0", v_ctx_len, a_ctx_len);
    #[rustfmt::skip]
    let (vf, af, tf) = blk.forward(&vx, &ax, &v_adaln, &a_adaln, &v_context, &a_context,
        &vc, &vs, &ac, &as_, &vcc, &vcs, &acc, &acs, &v_ss, &a_ss, &a2v_gate, &v2a_gate, tv, ta, true);
    drop(blk);

    let gpu_q = Gpu::open(None, &KERNELS);
    let (vc2, vs2) = upload_rope(&gpu_q, &vr);
    let (ac2, as2) = upload_rope(&gpu_q, &ar);
    let (vcc2, vcs2) = upload_rope(&gpu_q, &vcr);
    let (acc2, acs2) = upload_rope(&gpu_q, &acr);
    let rope = AvRope { v_cos: &vc2, v_sin: &vs2, a_cos: &ac2, a_sin: &as2, v_cross_cos: &vcc2, v_cross_sin: &vcs2, a_cross_cos: &acc2, a_cross_sin: &acs2 };
    let blkq = LtxAvBlockQ::on(gpu_q.share(), &vcfg, &acfg, &w, "transformer_blocks.0", v_ctx_len, a_ctx_len, QTier::Int8);
    #[rustfmt::skip]
    let (vq, aq, tq) = blkq.forward(&vx, &ax, &v_adaln, &a_adaln, &v_context, &a_context, rope,
        &v_ss, &a_ss, &a2v_gate, &v2a_gate, tv, ta);

    report_lossy("real Q8_0 AV block-0 int8 video out", &vq, &vf, 0.99, 1.5e-1);
    report_lossy("real Q8_0 AV block-0 int8 audio out", &aq, &af, 0.99, 1.5e-1);
    report_lossy("real Q8_0 AV block-0 int8 a2v_out (raw, pre-gate)", &tq.a2v_out, &tf.a2v_out, 0.98, 2e-1);
    report_lossy("real Q8_0 AV block-0 int8 v2a_out (raw, pre-gate)", &tq.v2a_out, &tf.v2a_out, 0.98, 2e-1);
}

// ------------------------------------ the real-weight AV FORWARD, streamed --

/// Every non-block tensor of a config's manifest plus blocks `[0, layers)`,
/// dequantized to fp32 - what the eager [`ltxv::LtxAvDit`] wants and what the
/// streamed path deliberately never materializes.
fn load_all_av_weights(path: &str, cfg: &LtxAvDitConfig) -> vae::blocks::Tensors {
    let mg = checkpoint::gguf::MmapGguf::open(path).unwrap_or_else(|e| panic!("opening {path}: {e}"));
    ltxv::dit::av_dit_tensor_manifest(cfg)
        .into_iter()
        .map(|(name, shape)| {
            let want: usize = shape.iter().product();
            let data = mg.tensor(&name).unwrap_or_else(|| panic!("real ltxv AV gguf: missing tensor {name}")).unwrap_or_else(|e| panic!("real ltxv AV gguf: {name}: {e}"));
            assert_eq!(data.len(), want, "real ltxv AV gguf: {name} has {} values, expected {want}", data.len());
            (name, (shape, data))
        })
        .collect()
}

/// The AV path's own `streamed_vs_eager_real.rs`: the REAL production
/// audio-visual forward (`ltxv::dit::av_forward_q_streamed_in` - int8 compute,
/// blocks streamed off the GGUF into a device-resident window, which is
/// exactly what a generation dispatches) against the eager fp32
/// [`ltxv::LtxAvDit::forward`], on the SAME real weights and the SAME inputs.
///
/// Why this exists on top of the block-level gate above, which it does not
/// replace: `LtxAvBlockQ` being right block for block does not make the
/// FORWARD right. Everything BETWEEN the blocks lives only on the streamed
/// path - both streams' patchify, the six model-level adaLN row tables and
/// their row maps, the four RoPE table sets, both embeddings connectors, the
/// resident window's own Belady rotation and its upload/hit bookkeeping, and
/// both output stages. Before this test, none of that had any coverage at all
/// on the audio-visual path: `av_forward_q_streamed_in` was called by one
/// bench binary and by nothing else in the workspace.
///
/// int8 against fp32 is LOSSY, so cosine AND relative L2 are both asserted -
/// cosine is scale-invariant and a systematic gain moves only rel_l2. The
/// floors sit below what a clean run measures, with headroom, and NOT at it:
/// an int8 tier is not expected to reach 1.0 (the video-only tier's own
/// real-weight block-0 comparison measures around 0.9963 cosine), and a floor
/// pinned to a measurement becomes a flake the first time a kernel
/// reassociates a sum. A clean run of this comparison measures cosine
/// 0.99750 / rel_l2 7.78e-2 on video and 0.99814 / 6.15e-2 on audio, so the
/// floors below (cosine 0.99, rel_l2 1.5e-1) sit clear of both without being
/// pinned to either.
///
/// Deliberately RESIDENT (`AvDitSession::resident_with_slots`) with fewer
/// slots than layers, so the rotation - the one part of the window that only
/// runs when the window is narrower than the model - is on the tested path
/// rather than only on a 24 GiB card at production width.
///
/// Mutation-verified, and the interesting result is what this gate does NOT
/// catch:
///
/// | mutation | this gate | which gate did catch it |
/// |---|---|---|
/// | every `to_gate_logits` attention gate forced OPEN in the int8 path | RED, cosine -0.014 | also both block-level tolerance gates |
/// | the block-to-block activation chain not advanced | RED, cosine 0.697 | only this gate on the AV side - the resident-vs-streaming bit-identity gate carries the same fault in both arms and passes |
/// | the A2V CROSS-MODAL gate forced OPEN | green (0.9948 / 1.16e-1, inside the floors) | `a_closed_cross_modal_gate_...`, exactly (bit-identity), and the real-weight block-0 gate at cosine 0.595 |
/// | the A2V video scale row derived without its `1.0 +` | green (0.99746, a 4e-5 move) | `device_derived_av_modulation_...`, on BITS - no tolerance gate can see a change that small |
///
/// The last two rows are the reason this file keeps an exact gate and an
/// analytic gate beside its tolerance gates rather than only widening the
/// tolerance ones.
#[test]
fn real_q8_0_av_streamed_forward_matches_the_eager_fp32_forward() {
    let Some(path) = real_dit_gguf() else {
        brain_testutil::skip(&format!("set BRAIN_LTXV_DIT to a real {REPO} distilled Q8_0 GGUF (none in the model store)"));
        return;
    };
    let mut cfg = LtxAvDitConfig::ltx25();
    cfg.video.num_layers = 2;
    cfg.assert_supported();
    assert!(cfg.video.use_embeddings_connector, "this test's point is the whole real path, connectors included");

    // Small but real-shaped: a `context_len` that is a real multiple of the
    // connector's own register count (128, `EmbeddingsConnector`'s assertion),
    // and per-token timesteps that are NOT all equal, so the adaLN row table
    // has more than one distinct row and its row map is actually exercised.
    let (tv, ta, ctx_len) = (8usize, 6usize, 128usize);
    let (vcfg, acfg) = (cfg.video, cfg.audio);
    let v_latent = ramp(tv * vcfg.in_channels as usize, 23, 1.1);
    let a_latent = ramp(ta * acfg.in_channels as usize, 29, 0.9);
    let v_timesteps: Vec<f32> = (0..tv).map(|i| if i % 3 == 0 { 0.0 } else { 0.7 }).collect();
    let a_timesteps: Vec<f32> = (0..ta).map(|i| if i % 2 == 0 { 0.0 } else { 0.7 }).collect();
    let v_positions = positions(3, tv);
    let a_positions = positions(1, ta);
    let v_keyframes_mask = vec![0f32; tv];
    let v_context = ramp(ctx_len * vcfg.cross_attention_dim as usize, 7, 1.4);
    let a_context = ramp(ctx_len * acfg.connector_inner_dim() as usize, 11, 1.2);
    let mut context_valid = vec![0f32; ctx_len];
    context_valid[..20].fill(1.0);
    let (v_sigma, a_sigma) = (0.7f32, 0.7f32);

    // ---- eager, fp32: the reference this crate's other AV gates replay ----
    let t0 = std::time::Instant::now();
    let w = load_all_av_weights(&path, &cfg);
    eprintln!("eager AV weight subset loaded ({} tensors) in {:.1} s", w.len(), t0.elapsed().as_secs_f64());
    let model = ltxv::LtxAvDit::new(cfg, w, None);
    let t1 = std::time::Instant::now();
    #[rustfmt::skip]
    let taps = model.forward(&v_latent, &v_timesteps, &v_positions, &v_keyframes_mask, &v_context, ctx_len, tv, v_sigma, &context_valid,
        &a_latent, &a_timesteps, &a_positions, &a_context, ctx_len, ta, a_sigma, &context_valid);
    eprintln!("eager AV forward (fp32) ran in {:.1} s", t1.elapsed().as_secs_f64());
    drop(model);

    // ---- streamed, int8, device-resident: the production path ------------
    let src = ltxv::gguf_src::LtxvGgufSource::open(&path).unwrap_or_else(|e| panic!("opening {path}: {e}"));
    let head = ltxv::dit::load_av_head_tensors_from_source(&src, &cfg);
    let step = ltxv::dit::AvStreamedStep {
        v_latent: &v_latent,
        v_timesteps: &v_timesteps,
        v_positions: &v_positions,
        v_keyframes_mask: &v_keyframes_mask,
        v_context: &v_context,
        v_context_len: ctx_len,
        tv,
        v_sigma,
        v_context_valid: &context_valid,
        a_latent: &a_latent,
        a_timesteps: &a_timesteps,
        a_positions: &a_positions,
        a_context: &a_context,
        a_context_len: ctx_len,
        ta,
        a_sigma,
        a_context_valid: &context_valid,
    };
    let session = ltxv::devres::AvDitSession::resident_with_slots(None, 1);
    let cache = ltxv::block::GenerationCache::default();
    let t2 = std::time::Instant::now();
    let (v_streamed, a_streamed) = ltxv::dit::av_forward_q_streamed_in(&session, &cfg, &src, &head, QTier::Int8, &step, &cache);
    eprintln!("streamed AV forward (int8, resident) ran in {:.1} s", t2.elapsed().as_secs_f64());
    let rs = session.stats();
    assert!(rs.slots == 1 && rs.uploads >= u64::from(cfg.video.num_layers), "the narrow resident window must have rotated: {rs:?}");

    report_lossy("real Q8_0 AV streamed-vs-eager video out", &v_streamed, &taps.video.out, 0.99, 1.5e-1);
    report_lossy("real Q8_0 AV streamed-vs-eager audio out", &a_streamed, &taps.audio.out, 0.99, 1.5e-1);

    // A second forward on the SAME session and cache is the shape a denoise
    // loop actually runs (every block resident or re-uploaded from the host
    // cache, nothing re-read from the GGUF) and must be BIT-identical to the
    // first: residency changes WHEN bytes move, never what any kernel reads.
    let (v2, a2) = ltxv::dit::av_forward_q_streamed_in(&session, &cfg, &src, &head, QTier::Int8, &step, &cache);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    assert_eq!(bits(&v2), bits(&v_streamed), "a warm AV forward changed the video output (max_abs {:.3e})", max_abs(&v2, &v_streamed));
    assert_eq!(bits(&a2), bits(&a_streamed), "a warm AV forward changed the audio output (max_abs {:.3e})", max_abs(&a2, &a_streamed));
    eprintln!("warm AV forward: bit-identical on both streams");
}

/// [`positions`] displaced along every axis, so two shapes at the SAME token
/// count still have different RoPE rotations - which is what a long-form
/// window boundary produces and what the `RopeCache`'s key has to see.
fn positions_shifted(axes: usize, t: usize, shift: f32) -> Vec<f32> {
    positions(axes, t).iter().map(|p| p + shift).collect()
}

/// One shape a reused session is asked to run: both streams' own inputs for a
/// single joint forward, owned so several of them can be built up front and
/// replayed in any order.
struct AvShape {
    tv: usize,
    ta: usize,
    v_latent: Vec<f32>,
    a_latent: Vec<f32>,
    v_timesteps: Vec<f32>,
    a_timesteps: Vec<f32>,
    v_positions: Vec<f32>,
    a_positions: Vec<f32>,
    v_keyframes_mask: Vec<f32>,
}

impl AvShape {
    fn step<'a>(&'a self, v_context: &'a [f32], a_context: &'a [f32], context_valid: &'a [f32], ctx_len: usize) -> ltxv::dit::AvStreamedStep<'a> {
        ltxv::dit::AvStreamedStep {
            v_latent: &self.v_latent,
            v_timesteps: &self.v_timesteps,
            v_positions: &self.v_positions,
            v_keyframes_mask: &self.v_keyframes_mask,
            v_context,
            v_context_len: ctx_len,
            tv: self.tv,
            v_sigma: 0.7,
            v_context_valid: context_valid,
            a_latent: &self.a_latent,
            a_timesteps: &self.a_timesteps,
            a_positions: &self.a_positions,
            a_context,
            a_context_len: ctx_len,
            ta: self.ta,
            a_sigma: 0.7,
            a_context_valid: context_valid,
        }
    }
}

/// One audio-visual generation reuses ONE session across shapes it did not
/// build that session at - and the answer must not depend on that.
///
/// The pipeline holds a single `RealAvDit` for a whole clip and hands it three
/// kinds of boundary: stage 1 -> stage 2 (the token count roughly quadruples),
/// window k -> window k+1 (the token count may be unchanged while every
/// position moves, because a continuation window re-bases both streams onto
/// its own time origin), and step -> step (nothing moves). The session is
/// released at the first two, but "released" is a memory decision, not a
/// correctness one: a session that survived a boundary must still be exact, or
/// the correctness of a clip depends on when a card happened to be handed
/// back.
///
/// The comparison is BIT equality against a session built fresh for each
/// shape, not a tolerance - residency and caching change WHEN bytes move,
/// never what any kernel reads, so there is nothing here for a tolerance to
/// absorb.
///
/// Mutation-verified. `rope_key` is the interesting one: the second shape has
/// the same token count as the first and differs only in its POSITIONS, so a
/// key that hashed geometry without positions would serve shape B the tables
/// it built for shape A. That is exactly the "silently reusing another
/// shape's rotation produces plausible video" failure `RopeCache`'s own doc
/// names, and only a same-`t` pair can see it.
#[test]
fn a_reused_av_session_is_exact_across_a_window_or_stage_boundary() {
    let Some(path) = real_dit_gguf() else {
        brain_testutil::skip(&format!("set BRAIN_LTXV_DIT to a real {REPO} distilled Q8_0 GGUF (none in the model store)"));
        return;
    };
    let mut cfg = LtxAvDitConfig::ltx25();
    cfg.video.num_layers = 2;
    cfg.assert_supported();
    let (vcfg, acfg) = (cfg.video, cfg.audio);
    let ctx_len = 128usize;

    let src = ltxv::gguf_src::LtxvGgufSource::open(&path).unwrap_or_else(|e| panic!("opening {path}: {e}"));
    let head = ltxv::dit::load_av_head_tensors_from_source(&src, &cfg);

    // Three shapes: a step, the same token count with re-based positions (a
    // window seam), and a different token count (a stage boundary).
    let built: Vec<AvShape> = [(8usize, 6usize, 0.0f32), (8, 6, 1.5), (12, 9, 0.0)]
        .iter()
        .map(|&(tv, ta, shift)| AvShape {
            tv,
            ta,
            v_latent: ramp(tv * vcfg.in_channels as usize, 23, 1.1),
            a_latent: ramp(ta * acfg.in_channels as usize, 29, 0.9),
            v_timesteps: (0..tv).map(|i| if i % 3 == 0 { 0.0 } else { 0.7 }).collect(),
            a_timesteps: (0..ta).map(|i| if i % 2 == 0 { 0.0 } else { 0.7 }).collect(),
            v_positions: positions_shifted(3, tv, shift),
            a_positions: positions_shifted(1, ta, shift),
            v_keyframes_mask: vec![0f32; tv],
        })
        .collect();
    let v_context = ramp(ctx_len * vcfg.cross_attention_dim as usize, 7, 1.4);
    let a_context = ramp(ctx_len * acfg.connector_inner_dim() as usize, 11, 1.2);
    let mut context_valid = vec![0f32; ctx_len];
    context_valid[..20].fill(1.0);

    // A narrow window, so the slot rotation is on the tested path too.
    let shared = ltxv::devres::AvDitSession::resident_with_slots(None, 1);
    let cache = ltxv::block::GenerationCache::default();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    for (i, b) in built.iter().enumerate() {
        let step = b.step(&v_context, &a_context, &context_valid, ctx_len);
        let (v_shared, a_shared) = ltxv::dit::av_forward_q_streamed_in(&shared, &cfg, &src, &head, QTier::Int8, &step, &cache);
        let fresh = ltxv::devres::AvDitSession::resident_with_slots(None, 1);
        let (v_fresh, a_fresh) = ltxv::dit::av_forward_q_streamed_in(&fresh, &cfg, &src, &head, QTier::Int8, &step, &cache);
        // `assert!` on the comparison, not `assert_eq!` on the vectors: a
        // failure here is thousands of u32s wide, and the number that
        // identifies it is the max_abs, not the dump.
        assert!(
            bits(&v_shared) == bits(&v_fresh),
            "shape {i} (tv={}, ta={}) video output depends on whether the session was reused (max_abs {:.3e})",
            b.tv,
            b.ta,
            max_abs(&v_shared, &v_fresh)
        );
        assert!(
            bits(&a_shared) == bits(&a_fresh),
            "shape {i} (tv={}, ta={}) audio output depends on whether the session was reused (max_abs {:.3e})",
            b.tv,
            b.ta,
            max_abs(&a_shared, &a_fresh)
        );
        eprintln!("shape {i}: tv={} ta={} - reused session bit-identical to a fresh one on both streams", b.tv, b.ta);
    }
    // Distinct shapes must have produced distinct answers, or the comparison
    // above would be satisfied by a forward that ignored its inputs.
    let redo = |i: usize| {
        let s = ltxv::devres::AvDitSession::resident_with_slots(None, 1);
        ltxv::dit::av_forward_q_streamed_in(&s, &cfg, &src, &head, QTier::Int8, &built[i].step(&v_context, &a_context, &context_valid, ctx_len), &cache)
    };
    let (v0, _) = redo(0);
    let (v1, _) = redo(1);
    assert!(bits(&v0) != bits(&v1), "re-basing every position must change the video output, or the RoPE tables are not reaching the forward");
}
