// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Stage-by-stage forward parity for the T5 encoder against the goldens dumped
//! by `tools/t5_dump_reference.py`.
//!
//! Fixtures live under `$BRAIN_TESTDATA` (default `<repo>/testdata`) in `t5/`;
//! the reference WEIGHTS are named by env var. The test SKIPS itself (never
//! fails) when either is absent:
//!
//! ```text
//! BRAIN_T5_XXL=/path/to/FLUX.1-Kontext-dev/text_encoder_2
//! ```
//!
//! Gate: cosine >= 0.9999 per stage. `last_hidden_state` is ALSO split by row
//! population (content rows vs right-pad rows). That split matters more here
//! than it does for CLIP: T5 is bidirectional and FLUX passes no attention
//! mask, so pad rows are genuinely attended and a pad-row error cannot hide
//! behind causal isolation — it would corrupt the content rows too.
//!
//! At 19 GB of fp32 weights this is a heavy test. It fits a 24 GB P40 alongside
//! its ~1.9 GB of activations at the golden's B=2 x T=128; `BRAIN_DEVICE=cpu`
//! runs it out of host RAM if the card is busy.

use std::path::{Path, PathBuf};

use t5::config::T5Config;
use t5::model::{T5Encoder, Tap};

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

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&x, &y) in got.iter().zip(want) {
        num += (x as f64 - y as f64).powi(2);
        den += (y as f64).powi(2);
    }
    if den == 0.0 {
        return 0.0;
    }
    (num / den).sqrt()
}

/// Collected results so a run prints one table and fails once, at the end — a
/// single failing stage must not hide the twenty behind it.
#[derive(Default)]
struct Report {
    rows: Vec<(String, f64, f64, f64)>,
    failures: Vec<String>,
}

impl Report {
    fn check(&mut self, stage: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{stage}: len {} != golden {}", got.len(), want.len());
        let (c, m, r) = (cosine(got, want), max_abs(got, want), rel_l2(got, want));
        eprintln!("  {stage:<28} cosine={c:.10}  max_abs={m:.3e}  rel_l2={r:.3e}");
        self.rows.push((stage.to_string(), c, m, r));
        // NaN-safe: a NaN cosine (an all-zero or poisoned stage) must FAIL, so
        // this is an explicit NaN check plus `<`, not `!(>=)`.
        if c.is_nan() || c < GATE {
            self.failures.push(format!("{stage}: cosine {c:.10} < {GATE}"));
        }
    }

    /// Same stage, restricted to the rows `keep` selects.
    fn check_rows(
        &mut self,
        stage: &str,
        got: &[f32],
        want: &[f32],
        width: usize,
        keep: &dyn Fn(usize) -> bool,
    ) {
        let (mut g, mut w) = (Vec::new(), Vec::new());
        for r in 0..got.len() / width {
            if keep(r) {
                g.extend_from_slice(&got[r * width..(r + 1) * width]);
                w.extend_from_slice(&want[r * width..(r + 1) * width]);
            }
        }
        if g.is_empty() {
            return;
        }
        self.check(stage, &g, &w);
    }

    fn finish(self, what: &str) {
        let worst = self.rows.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
        eprintln!(
            "{what}: {} stages checked, {} failed, worst cosine {worst:.10}",
            self.rows.len(),
            self.failures.len()
        );
        assert!(self.failures.is_empty(), "{what} parity failures:\n  {}", self.failures.join("\n  "));
    }
}

struct Golden {
    t: Vec<checkpoint::safetensors::StTensor>,
}

impl Golden {
    fn open(rel: &str) -> Option<Golden> {
        let p = testdata(rel);
        if !p.exists() {
            eprintln!("SKIP: golden {} absent", p.display());
            return None;
        }
        Some(Golden { t: checkpoint::safetensors::read(p.to_str().unwrap()).expect("read golden") })
    }
    fn find(&self, name: &str) -> &checkpoint::safetensors::StTensor {
        self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden tensor {name}"))
    }
    fn get(&self, name: &str) -> &Vec<f32> {
        &self.find(name).data
    }
    fn shape(&self, name: &str) -> &Vec<usize> {
        &self.find(name).shape
    }
    fn ids(&self, name: &str) -> Vec<u32> {
        self.get(name).iter().map(|&x| x as u32).collect()
    }
}

fn env_path(var: &str) -> Option<PathBuf> {
    let v = std::env::var(var).ok().filter(|s| !s.is_empty())?;
    let p = PathBuf::from(v);
    if !p.exists() {
        eprintln!("SKIP: {var}={} not found", p.display());
        return None;
    }
    Some(p)
}

#[test]
fn t5_xxl_encoder_stage_parity() {
    let Some(g) = Golden::open("t5/t5xxl/encoder.safetensors") else { return };
    let Some(weights) = env_path("BRAIN_T5_XXL") else {
        eprintln!("SKIP: set BRAIN_T5_XXL to a FLUX.1 text_encoder_2 directory");
        return;
    };

    let cfg = T5Config::xxl();
    let ids_shape = g.shape("input_ids").clone();
    assert_eq!(ids_shape.len(), 2, "input_ids must be [B, T]");
    let (b, t) = (ids_shape[0] as u32, ids_shape[1] as u32);
    let ids = g.ids("input_ids");
    let mask = g.ids("attention_mask");
    let n = (b * t) as usize;
    let (d, ff, inner) = (cfg.d_model as usize, cfg.d_ff as usize, cfg.inner() as usize);

    // ---- rung 1: mapping units ------------------------------------------
    // The bucket table is host integer math; check it EXACTLY (not by cosine)
    // against the reference's own table before anything reads it.
    let buckets = t5::hostbias::buckets(t, cfg.rel_buckets, cfg.rel_max_distance);
    let want_buckets = g.ids("relative_position_bucket");
    assert_eq!(buckets, want_buckets, "relative_position_bucket table differs from the reference");
    eprintln!("relative_position_bucket: {} entries, exact match", buckets.len());

    let src = t5::import::read_encoder(&weights).expect("read text_encoder_2");
    let src_count = src.len();
    let imported = t5::import::import_hf(src, &cfg).expect("import_hf");
    eprintln!("t5xxl: {src_count} source tensors -> {} parameters", imported.len());
    assert_eq!(imported.len(), cfg.tensor_manifest().len());
    assert_eq!(src_count, 219, "T5-XXL encoder ships 219 tensors");

    // ---- rungs 2-3: stage + single-forward parity ------------------------
    let m = T5Encoder::new_on(
        gpu_core::testgpu::dev(t5::model::PIPELINES),
        cfg.clone(),
        b,
        t,
        &t5::import::to_init(imported),
    );
    m.set_tokens(&ids);
    m.forward();
    m.poll_wait();

    let mut rep = Report::default();
    eprintln!("t5xxl stages (B={b}, T={t}):");
    rep.check("position_bias", &m.read_position_bias(), g.get("b0_position_bias"));
    rep.check("embed", &m.read_x(0), g.get("embed"));

    // block-0 internals
    let qkv = m.read_block_tap(0, Tap::Qkv);
    let region = |off: usize| -> Vec<f32> {
        (0..n).flat_map(|r| qkv[r * 3 * inner + off..r * 3 * inner + off + inner].to_vec()).collect()
    };
    rep.check("b0_attn_norm", &m.read_block_tap(0, Tap::AttnNorm), g.get("b0_attn_norm"));
    rep.check("b0_q", &region(0), g.get("b0_q"));
    rep.check("b0_k", &region(inner), g.get("b0_k"));
    rep.check("b0_v", &region(2 * inner), g.get("b0_v"));
    rep.check("b0_attn_ctx", &m.read_block_tap(0, Tap::Ctx), g.get("b0_attn_ctx"));
    rep.check("b0_attn_out", &m.read_block_tap(0, Tap::AttnOut), g.get("b0_attn_out"));
    rep.check("b0_attn_res", &m.read_block_tap(0, Tap::AttnRes), g.get("b0_attn_res"));
    rep.check("b0_ff_norm", &m.read_block_tap(0, Tap::FfNorm), g.get("b0_ff_norm"));
    rep.check("b0_wi0", &m.read_block_tap(0, Tap::Wi0), g.get("b0_wi0"));
    rep.check("b0_wi1", &m.read_block_tap(0, Tap::Wi1), g.get("b0_wi1"));
    rep.check("b0_gated", &m.read_block_tap(0, Tap::Gated), g.get("b0_gated"));
    rep.check("b0_ff_out", &m.read_block_tap(0, Tap::FfOut), g.get("b0_ff_out"));

    // every block output — this is where a per-layer drift shows up
    for l in 0..cfg.layers as usize {
        rep.check(&format!("block{l}_out"), &m.read_x(l + 1), g.get(&format!("block{l}_out")));
    }

    let hidden = m.read_hidden();
    rep.check("last_hidden_state", &hidden, g.get("last_hidden_state"));
    let want = g.get("last_hidden_state");
    rep.check_rows("last_hidden_state[content]", &hidden, want, d, &|r| mask[r] == 1);
    rep.check_rows("last_hidden_state[pad]", &hidden, want, d, &|r| mask[r] == 0);
    let wi0 = m.read_block_tap(0, Tap::Wi0);
    rep.check_rows("b0_wi0[content]", &wi0, g.get("b0_wi0"), ff, &|r| mask[r] == 1);

    // The masked reference run is dumped for reference only — brain implements
    // the UNMASKED contract FLUX uses. Report the distance so the choice stays
    // visible in the log instead of buried in a doc comment.
    let masked = g.get("last_hidden_state_masked");
    eprintln!(
        "  (masked reference run differs from the unmasked one by max_abs={:.3e} — \
         brain implements the unmasked FLUX contract)",
        max_abs(want, masked)
    );

    rep.finish("t5xxl");
}
