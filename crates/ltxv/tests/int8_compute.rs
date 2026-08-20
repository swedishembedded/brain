// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Compute-time int8/int4 (DP4A) parity - the sibling `crate::int8`'s module
//! doc names as the explicit gap left by the storage-only tier
//! (`crates/ltxv/tests/int8_storage.rs`): "no compute-time DP4A
//! activation-quantization path". This file exercises the NEW dispatch path
//! (`crate::block::{QTier, LtxBlockQ}`, wired through `LtxDit::forward_q`)
//! rather than the storage-only round trip.
//!
//! Two things, matching this crate's own `int8_storage.rs` style:
//!
//! 1. A tiny (2-layer, `inner_dim` 64) random-weight [`LtxDit`] forward run
//!    twice - once plain fp32 ([`LtxDit::forward`]), once with every
//!    eligible weight quantized on load and every linear dispatched through
//!    the DP4A GEMM kernels ([`LtxDit::forward_q`]) - at both the int8 and
//!    int4 tiers. No fixture dependency, always runs.
//! 2. A REAL-weight-gated check: one block (`transformer_blocks.0`) of the
//!    real 22B LTX-2.5 distilled Q8_0 checkpoint, loaded straight off
//!    [`ltxv::gguf_src::LtxvGgufSource`] via
//!    [`ltxv::block::load_block_tensors_from_source`] (bounded to this one
//!    block's own tensors, never the whole 22B model), run once fp32
//!    ([`ltxv::block::LtxBlock`]) and once int8-compute
//!    ([`ltxv::block::LtxBlockQ`]) at a SMALL token count (`t=8`,
//!    `context_len=6` - the same shape budget Phase 4's own
//!    `ltxv_real_dit_dump_reference.py`/`ltxv_bench.rs` use, per this task's
//!    "start small" constraint). Skips loudly without the real checkpoint
//!    (`BRAIN_REQUIRE_FIXTURES=1` upgrades the skip to a failure), matching
//!    `gguf_quant_real.rs`'s convention.

use ltxv::block::{LtxBlockQ, QTier};
use ltxv::dit::random_tiny_weights;
use ltxv::modelgrad::Cfg;
use ltxv::{LtxDit, LtxDitConfig};

// ------------------------------------------------------------------ metrics

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

// ------------------------------------------------------- synthetic inputs

struct Inputs {
    latent: Vec<f32>,
    timesteps: Vec<f32>,
    positions: Vec<f32>,
    keyframes_mask: Vec<f32>,
    context: Vec<f32>,
    context_len: usize,
    t: usize,
}

fn synthetic_inputs(cfg: &LtxDitConfig, t: usize, context_len: usize) -> Inputs {
    let mcfg = Cfg::from_ltx(cfg, t, context_len);
    let positions = mcfg.simple_positions();
    let latent: Vec<f32> = (0..t * cfg.in_channels as usize).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.1).collect();
    let context: Vec<f32> = (0..context_len * cfg.cross_attention_dim as usize).map(|i| ((i % 7) as f32 / 7.0 - 0.5) * 1.4).collect();
    let timesteps: Vec<f32> = (0..t).map(|i| 0.2 + 0.05 * (i % 5) as f32).collect();
    let mut keyframes_mask = vec![0f32; t];
    keyframes_mask[0] = 1.0;
    Inputs { latent, timesteps, positions, keyframes_mask, context, context_len, t }
}

// ------------------------------------------------ 1. tiny synthetic config

#[test]
fn dit_forward_stays_close_with_int8_compute_dispatch() {
    let cfg = LtxDitConfig::tiny();
    let w = random_tiny_weights(&cfg, 0xC0117E);
    let inputs = synthetic_inputs(&cfg, 7, 5);
    let context_valid = vec![1.0f32; inputs.context_len];

    let model = LtxDit::new(cfg, w, None);
    let taps_f32 = model.forward(&inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &context_valid);
    let taps_i8 = model.forward_q(&inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &context_valid, QTier::Int8);
    let taps_i4 = model.forward_q(&inputs.latent, &inputs.timesteps, &inputs.positions, &inputs.keyframes_mask, &inputs.context, inputs.context_len, inputs.t, &context_valid, QTier::Int4);

    let c8 = cosine(&taps_f32.out, &taps_i8.out);
    let c4 = cosine(&taps_f32.out, &taps_i4.out);
    println!("int8-compute forward parity: final output cosine = {c8:.9}");
    println!("int4-compute forward parity: final output cosine = {c4:.9}");
    // Measured on this fixture: int8 cosine 0.999999989, int4 cosine
    // 0.999998446 (both well above FLUX.2's own real-fixture int8 measurement
    // of ~0.9985-0.9989 - this tiny config's per-channel scales have very
    // little dynamic range to begin with). Matching FLUX.2's own documented
    // approach (measure, then pick a floor with a sane margin below the
    // measured number rather than assuming bit-exactness).
    assert!(c8 >= 0.9999, "int8-compute forward diverged too far from fp32: cosine {c8:.9}");
    assert!(c4 >= 0.9999, "int4-compute forward diverged too far from fp32: cosine {c4:.9}");

    for (i, (a, b)) in taps_f32.block_out.iter().zip(&taps_i8.block_out).enumerate() {
        let c = cosine(a, b);
        println!("int8-compute forward parity: block {i} output cosine = {c:.9}");
        assert!(c >= 0.9999, "block {i} output diverged too far under int8 compute: cosine {c:.9}");
    }
}

// --------------------------------------------------- 2. real-weight-gated

const REPO: &str = "Lightricks/LTX-2.5";

fn gguf_path() -> Option<String> {
    if let Ok(p) = std::env::var("BRAIN_LTXV_DIT") {
        if !p.is_empty() {
            return Some(p);
        }
    }
    let dir = brain_testutil::model_dir(REPO)?;
    let mut found: Vec<String> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains("Q8_0") && n.ends_with(".gguf")))
        // Discriminate on the file's OWN declared architecture, not on
        // its name. The model store legitimately holds several Q8_0
        // GGUFs for one repo - the DiT and, since the text encoder was
        // quantized too, Gemma-4 - and a name glob picked whichever
        // sorted first, which surfaced as an architecture mismatch deep
        // inside an importer rather than as "no fixture here".
        .filter(|p| {
            checkpoint::gguf::MmapGguf::open(&p.to_string_lossy())
                .ok()
                .and_then(|g| g.kv().get("general.architecture").and_then(|v| v.as_str()).map(str::to_string))
                .as_deref()
                == Some(ltxv::import::GGUF_ARCHITECTURE)
        })
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    found.sort();
    found.into_iter().next()
}

#[test]
fn real_q8_0_block0_int8_compute_matches_fp32() {
    let Some(path) = gguf_path() else {
        brain_testutil::skip(&format!("set BRAIN_LTXV_DIT to a real {REPO} distilled Q8_0 GGUF (none in the model store)"));
        return;
    };

    let cfg = LtxDitConfig::ltx25_22b();
    let src = ltxv::gguf_src::LtxvGgufSource::open(&path).unwrap_or_else(|e| panic!("opening {path}: {e}"));
    let tensors = ltxv::block::load_block_tensors_from_source(&src, &cfg, "transformer_blocks.0");

    // Small shape budget (this task's "start small" constraint): grid
    // (2,2,2) -> 8 tokens, context_len 6 - the same shape Phase 4's own
    // dumper/bench already use at this checkpoint's real width.
    let (t, context_len) = (8usize, 6usize);
    let mcfg = Cfg::from_ltx(&cfg, t, context_len);
    let positions = mcfg.simple_positions();
    let latent: Vec<f32> = (0..t * cfg.in_channels as usize).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 1.1).collect();
    let context: Vec<f32> = (0..context_len * cfg.cross_attention_dim as usize).map(|i| ((i % 7) as f32 / 7.0 - 0.5) * 1.4).collect();
    let adaln_table: Vec<f32> = (0..t * cfg.adaln_rows() as usize * cfg.inner_dim as usize).map(|i| ((i % 13) as f32 / 13.0 - 0.5) * 0.3).collect();

    let rope = ltxv::rope::ltx_rope_tables(cfg.inner_dim, cfg.num_heads, cfg.positional_embedding_theta, &cfg.positional_embedding_max_pos, &positions, t);

    let gpu_f32 = ltxv::block::open_device(None);
    let (cos_f32, sin_f32) = upload_rope(&gpu_f32, &rope);
    let blk_f32 = ltxv::block::LtxBlock::on(gpu_f32.share(), &cfg, &tensors, "transformer_blocks.0", t as u32, context_len as u32);
    let (out_f32, _) = blk_f32.forward(&latent_padded(&cfg, &latent, t), &adaln_table, &context, &cos_f32, &sin_f32, t as u32);

    let gpu_i8 = ltxv::block::open_device(None);
    let (cos_i8, sin_i8) = upload_rope(&gpu_i8, &rope);
    let blk_i8 = LtxBlockQ::on(gpu_i8.share(), &cfg, &tensors, "transformer_blocks.0", t as u32, context_len as u32, QTier::Int8);
    let (out_i8, _) = blk_i8.forward(&latent_padded(&cfg, &latent, t), &adaln_table, &context, &cos_i8, &sin_i8, t as u32);

    let c = cosine(&out_f32, &out_i8);
    println!("real Q8_0 block-0 int8-compute vs fp32: cosine = {c:.9}, t={t}, context_len={context_len}, dim={}", cfg.inner_dim);
    // Measured on the real checkpoint: 0.996303655 (dim=4096, t=8,
    // context_len=6, block 0). Lower than the synthetic tiny-config number
    // above because these are the checkpoint's own real weight
    // distributions (real per-channel dynamic range), not i.i.d. Gaussian
    // noise - a floor with a sane margin below the measured value, per this
    // file's own doc.
    assert!(c >= 0.99, "real-weight int8-compute block output diverged too far from fp32: cosine {c:.9}");
}

/// [`LtxBlock::forward`]/[`LtxBlockQ::forward`] both take the block's own
/// `[t, dim]` hidden-state input (post-`patchify_proj`), not the raw
/// `[t, in_channels]` latent - reuse the same deterministic formula at the
/// block's own width so this test needs no host-side `patchify_proj`
/// matmul of its own.
fn latent_padded(cfg: &LtxDitConfig, _latent: &[f32], t: usize) -> Vec<f32> {
    let dim = cfg.inner_dim as usize;
    (0..t * dim).map(|i| ((i % 29) as f32 / 29.0 - 0.5) * 0.2).collect()
}

fn upload_rope(gpu: &gpu_core::Gpu, rope: &ltxv::rope::LtxRopeTables) -> (Vec<gpu_core::DeviceBuffer>, Vec<gpu_core::DeviceBuffer>) {
    let mut cos_bufs = Vec::with_capacity(rope.heads);
    let mut sin_bufs = Vec::with_capacity(rope.heads);
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
