// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Stage parity against a **tiny random reference encoder** whose dims break
//! the two coincidences T5-XXL hides.
//!
//! `tests/parity.rs` gates the real 4.762 B model, but at XXL
//! `num_heads == d_kv == 64` and `num_heads * d_kv == d_model == 4096`. Every
//! attention Params list here carries both the head count and the head width,
//! and the context width (`heads * d_kv`) is a different quantity from
//! `d_model` — at XXL all three are the same number, so a swap or a
//! substitution among them is arithmetically invisible and the 19 GB gate
//! passes anyway. This fixture uses `num_heads=2, d_kv=64, d_model=64`
//! (`inner = 128 != d_model`, `heads != d_kv`), so the same forward graph is
//! re-checked where those confusions produce a wrong number.
//!
//! It needs no released checkpoint — the reference weights are seeded random —
//! and the whole run is well under a second, so this is the cheap gate and
//! `parity.rs` is the expensive one. Regenerate the fixture with:
//!
//! ```text
//! python3 tools/goldens/t5encoder_dump_reference.py --out testdata/t5      # no --model needed
//! ```
//!
//! Fixtures resolve from `$BRAIN_TESTDATA` (default `<repo>/testdata`); the
//! test SKIPS itself when they are absent.

use std::path::{Path, PathBuf};

use t5encoder::config::T5Config;
use t5encoder::model::{T5Encoder, Tap};

const GATE: f64 = 0.9999;

fn testdata(rel: &str) -> PathBuf {
    let root = std::env::var("BRAIN_TESTDATA")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    Path::new(&root).join(rel)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).fold(0.0, f64::max)
}

/// The reference config the fixture was generated from — asserted against the
/// golden's own shapes below, so the two cannot drift apart silently.
fn tiny() -> T5Config {
    T5Config { vocab: 256, d_model: 64, d_ff: 128, d_kv: 64, layers: 2, heads: 2, ..T5Config::xxl() }
}

#[test]
fn tiny_reference_stage_parity_with_distinct_dims() {
    let (ckpt, golden) = (testdata("t5/tiny/ckpt"), testdata("t5/tiny/golden.safetensors"));
    if !ckpt.exists() || !golden.exists() {
        brain_testutil::skip(&format!("{} / {} absent", ckpt.display(), golden.display()));
        return;
    }
    let g = checkpoint::safetensors::read(golden.to_str().unwrap()).expect("read golden");
    let find = |n: &str| {
        g.iter().find(|t| t.name == n).unwrap_or_else(|| panic!("golden tensor {n}"))
    };

    let cfg = tiny();
    assert_ne!(cfg.inner(), cfg.d_model, "fixture must break inner == d_model");
    assert_ne!(cfg.heads, cfg.d_kv, "fixture must break heads == d_kv");

    let ids_shape = find("input_ids").shape.clone();
    let (b, t) = (ids_shape[0] as u32, ids_shape[1] as u32);
    let ids: Vec<u32> = find("input_ids").data.iter().map(|&x| x as u32).collect();
    // The golden's own widths pin the config: d_model from a block output and
    // the attention inner width from q, so a fixture regenerated at other dims
    // fails here instead of comparing the wrong tensors.
    assert_eq!(find("block0_out").shape[2], cfg.d_model as usize, "golden d_model");
    assert_eq!(find("b0_q").shape[2], cfg.inner() as usize, "golden heads*d_kv");

    let src = t5encoder::import::read_encoder(&ckpt).expect("read tiny ckpt");
    let imported = t5encoder::import::import_hf(src, &cfg).expect("import_hf");
    assert_eq!(imported.len(), cfg.tensor_manifest().len());

    let m = T5Encoder::new_on(
        gpu_core::testgpu::dev(t5encoder::model::PIPELINES),
        cfg.clone(),
        b,
        t,
        &t5encoder::import::to_init(imported),
    );
    m.set_tokens(&ids);
    m.forward();
    m.poll_wait();

    let n = (b * t) as usize;
    let inner = cfg.inner() as usize;
    let qkv = m.read_block_tap(0, Tap::Qkv);
    let region = |off: usize| -> Vec<f32> {
        (0..n).flat_map(|r| qkv[r * 3 * inner + off..r * 3 * inner + off + inner].to_vec()).collect()
    };

    let mut failures: Vec<String> = Vec::new();
    let mut worst = f64::INFINITY;
    let mut check = |stage: &str, got: &[f32], want: &[f32]| {
        assert_eq!(got.len(), want.len(), "{stage}: len {} != golden {}", got.len(), want.len());
        let (c, e) = (cosine(got, want), max_abs(got, want));
        eprintln!("  {stage:<20} cosine={c:.10}  max_abs={e:.3e}");
        worst = worst.min(c);
        if c.is_nan() || c < GATE {
            failures.push(format!("{stage}: cosine {c:.10} < {GATE}"));
        }
    };
    eprintln!("t5 tiny reference (B={b}, T={t}, heads={} d_kv={} inner={} d_model={}):",
              cfg.heads, cfg.d_kv, cfg.inner(), cfg.d_model);
    check("position_bias", &m.read_position_bias(), &find("b0_position_bias").data);
    check("embed", &m.read_x(0), &find("embed").data);
    check("b0_attn_norm", &m.read_block_tap(0, Tap::AttnNorm), &find("b0_attn_norm").data);
    check("b0_q", &region(0), &find("b0_q").data);
    check("b0_k", &region(inner), &find("b0_k").data);
    check("b0_v", &region(2 * inner), &find("b0_v").data);
    check("b0_attn_ctx", &m.read_block_tap(0, Tap::Ctx), &find("b0_attn_ctx").data);
    check("b0_attn_out", &m.read_block_tap(0, Tap::AttnOut), &find("b0_attn_out").data);
    check("b0_attn_res", &m.read_block_tap(0, Tap::AttnRes), &find("b0_attn_res").data);
    check("b0_ff_norm", &m.read_block_tap(0, Tap::FfNorm), &find("b0_ff_norm").data);
    check("b0_wi0", &m.read_block_tap(0, Tap::Wi0), &find("b0_wi0").data);
    check("b0_wi1", &m.read_block_tap(0, Tap::Wi1), &find("b0_wi1").data);
    check("b0_gated", &m.read_block_tap(0, Tap::Gated), &find("b0_gated").data);
    check("b0_ff_out", &m.read_block_tap(0, Tap::FfOut), &find("b0_ff_out").data);
    for l in 0..cfg.layers as usize {
        check(&format!("block{l}_out"), &m.read_x(l + 1), &find(&format!("block{l}_out")).data);
    }
    check("last_hidden_state", &m.read_hidden(), &find("last_hidden_state").data);
    eprintln!("t5 tiny: {} failed, worst cosine {worst:.10}", failures.len());
    assert!(failures.is_empty(), "tiny reference parity failures:\n  {}", failures.join("\n  "));
}
