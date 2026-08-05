// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Stage-by-stage forward parity for the SDXL `ControlNetModel` against the
//! goldens dumped by `tools/goldens/controlnet_dump_reference.py`.
//!
//! Fixtures live under `$BRAIN_TESTDATA` (default `<repo>/testdata`) in
//! `controlnet/`; the reference WEIGHTS are named by env var. The test **SKIPS
//! itself** (never fails) when either is absent:
//!
//! ```text
//! BRAIN_CONTROLNET=/path/to/a/diffusers/ControlNetModel
//! python3 tools/goldens/controlnet_dump_reference.py --controlnet $BRAIN_CONTROLNET \
//!     --out testdata/controlnet
//! ```
//!
//! VRAM: the reference ControlNet is 1.251 G parameters, so **fp32 weights are
//! ~5.0 GB** and this needs a card with that much free.
//!
//! Scope, stated up front so no number here is over-read: this gates TWO
//! forward evaluations (a 32x32 latent and a deliberately non-square 24x16
//! one), batch 1, 77 text tokens, fp32, on ONE checkpoint. It does **not** gate
//! the 128x128 latent SDXL generates at, batch > 1, int8, a sampling loop, or
//! anything about speed. It also does not gate the UNet consuming these
//! residuals at full size — `tests/smoke.rs` gates the seam at toy dims.

use std::path::{Path, PathBuf};

use controlnet::adapter::ControlSource;
use controlnet::config::ControlNetConfig;
use controlnet::model::{ControlNet, KERNELS};

/// Per-stage direction gate. The reference is fp32 torch on the CPU; brain's
/// reduction order differs, so exact equality is not the expectation.
const GATE: f64 = 0.9999;

/// Per-stage MAGNITUDE gate, and it is not redundant with [`GATE`]: cosine is
/// scale-invariant, so `got = 1.05 · want` scores a perfect 1.0. A ControlNet
/// has a whole family of whole-tensor scale mistakes available to it —
/// `conditioning_scale` applied twice, applied to the conditioning embedding
/// instead of the residual, or the zero-conv bias dropped — and every one of
/// them is invisible to cosine alone.
const REL_GATE: f64 = 1e-3;

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

struct Golden(Vec<checkpoint::safetensors::StTensor>);

impl Golden {
    fn get(&self, name: &str) -> Option<&[f32]> {
        self.0.iter().find(|t| t.name == name).map(|t| t.data.as_slice())
    }
    fn need(&self, name: &str) -> &[f32] {
        self.get(name).unwrap_or_else(|| panic!("golden missing {name}"))
    }
    /// The dumped SHAPE, which is the only place a non-square tensor's H and W
    /// survive: a flat length cannot be un-multiplied.
    fn shape(&self, name: &str) -> &[usize] {
        self.0
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.shape.as_slice())
            .unwrap_or_else(|| panic!("golden missing {name}"))
    }
}

struct Report {
    rows: Vec<(String, f64, f64, f64)>,
    failures: Vec<String>,
}

impl Report {
    fn check(&mut self, label: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{label}: {} values, golden has {}", got.len(), want.len());
        let (c, m, r) = (cosine(got, want), max_abs(got, want), rel_l2(got, want));
        self.rows.push((label.to_string(), c, m, r));
        if c < GATE {
            self.failures.push(format!("{label}: cosine {c:.10} < {GATE}, max_abs {m:.3e}"));
        }
        if r > REL_GATE {
            self.failures.push(format!("{label}: rel_l2 {r:.3e} > {REL_GATE:.0e}, max_abs {m:.3e}"));
        }
    }
}

#[test]
fn sdxl_controlnet_residuals_match_diffusers() {
    run_parity("controlnet/stages.safetensors");
}

/// The same ladder at a deliberately **non-square** latent (24x16, a 192x128
/// conditioning image).
///
/// Not redundant: at a square latent an H/W transposition is invisible
/// everywhere, and this crate's genuinely new stage — the conditioning
/// embedder, which halves H and W three times on its own bookkeeping — is
/// exactly where one would live. Every other gate in the imaging workstream is
/// square, so this is the only place in the UNet family where that class of bug
/// can fail.
#[test]
fn sdxl_controlnet_residuals_match_diffusers_at_a_non_square_latent() {
    run_parity("controlnet/stages_rect.safetensors");
}

fn run_parity(rel: &str) {
    let g = testdata(rel);
    if !g.exists() {
        eprintln!("SKIP: {} absent (run tools/goldens/controlnet_dump_reference.py)", g.display());
        return;
    }
    let Ok(weights) = std::env::var("BRAIN_CONTROLNET") else {
        eprintln!("SKIP: BRAIN_CONTROLNET unset (point it at a diffusers ControlNetModel dir)");
        return;
    };
    if !Path::new(&weights).exists() {
        eprintln!("SKIP: {weights} absent");
        return;
    }
    let gold =
        Golden(checkpoint::safetensors::read(g.to_str().expect("utf-8 path")).expect("read golden"));

    let cfg = ControlNetConfig::sdxl();
    let bb = cfg.backbone.clone();
    // The golden pins the sizes and the token count; deriving them keeps the
    // test valid if the dumper is re-run at a different `--latent`.
    let s = gold.shape("in.sample");
    let (lh, lw) = (s[2] as u32, s[3] as u32);
    let cs = gold.shape("in.controlnet_cond");
    let ds = cfg.cond_downscale();
    assert_eq!(
        (cs[2] as u32, cs[3] as u32),
        (lh * ds, lw * ds),
        "the golden's conditioning image does not match a {ds}x embedder over a {lh}x{lw} latent"
    );
    let t_enc = gold.need("in.encoder_hidden_states").len() as u32 / bb.cross_attention_dim;
    let timestep = gold.need("in.timestep")[0];
    let time_ids = gold.need("in.time_ids").to_vec();
    let pooled = gold.need("in.text_embeds").to_vec();
    let enc = gold.need("in.encoder_hidden_states").to_vec();
    let sample = gold.need("in.sample").to_vec();
    let cond = gold.need("in.controlnet_cond").to_vec();
    println!("== {rel}: latent {lh}x{lw}, cond {}x{}, t_enc {t_enc}, timestep {timestep}", lh * ds, lw * ds);

    let mut r = Report { rows: Vec::new(), failures: Vec::new() };

    // ---- host conditioning first: it is byte-for-byte the UNet's, so a
    // failure here is a `model::hostmath` / `unet::hostemb` failure and
    // explains every stage below it.
    let te = model::hostmath::timestep_embedding(
        timestep,
        bb.block_out_channels[0] as usize,
        bb.flip_sin_to_cos,
        bb.freq_shift as f64,
        10_000.0,
    );
    r.check("time_proj", &te, gold.need("time_proj"));
    let add = unet::hostemb::added_cond(
        &pooled,
        &time_ids,
        bb.addition_time_embed_dim,
        bb.flip_sin_to_cos,
        bb.freq_shift,
    );
    r.check("add_time_proj", &add[pooled.len()..], gold.need("add_time_proj"));

    println!("importing weights from {weights} ...");
    let tensors = controlnet::import::load(&weights, &cfg).expect("import");
    let params: usize = tensors.values().map(|(_, d)| d.len()).sum();
    println!(
        "imported {} tensors, {params} parameters = {:.2} GB fp32; building the graph ...",
        tensors.len(),
        params as f64 * 4.0 / 1e9
    );

    let m = ControlNet::new(
        gpu_core::testgpu::dev(&KERNELS),
        cfg.clone(),
        &tensors,
        lh,
        lw,
        t_enc,
        true,
    );
    drop(tensors);
    println!("{} steps; running ...", m.steps().len());
    let res = m.run(&sample, timestep, &enc, &pooled, &time_ids, &cond, 1.0);

    // ---- every tap that has a golden, in record order --------------------
    let mut covered = 0usize;
    for name in m.tap_names().iter().map(|s| s.to_string()).collect::<Vec<_>>() {
        let Some(want) = gold.get(&name) else { continue };
        let got = m.read_tap(&name).expect("tap exists");
        r.check(&name, &got, want);
        covered += 1;
    }

    // ---- the deliverable: per-injection-point residual parity -------------
    let points = ControlSource::injection_points(&m);
    assert_eq!(points.len(), 10, "SDXL has 9 down injection points and 1 mid");
    for (k, p) in points.iter().enumerate() {
        let gname = if p.name == "mid" { "out.mid".to_string() } else { format!("out.down{k}") };
        let got = res.get(&p.name).unwrap_or_else(|| panic!("no residual at {}", p.name));
        r.check(&format!("residual[{}]", p.name), got, gold.need(&gname));
    }

    // ---- and that `conditioning_scale` is a pure multiply of exactly these
    // The golden's second forward was run by diffusers at 0.75, so this is not
    // a self-consistency check: it gates that brain applies the scale in the
    // same PLACE the reference does.
    let scaled = m.run(&sample, timestep, &enc, &pooled, &time_ids, &cond, 0.75);
    for (k, p) in points.iter().enumerate() {
        let gname = if p.name == "mid" { "out0.75.mid".to_string() } else { format!("out0.75.down{k}") };
        let got = scaled.get(&p.name).unwrap_or_else(|| panic!("no residual at {}", p.name));
        r.check(&format!("residual0.75[{}]", p.name), got, gold.need(&gname));
    }

    let mut worst = ("".to_string(), 1.0f64);
    let mut worst_rel = ("".to_string(), 0.0f64);
    println!("\n{:<40} {:>14} {:>11} {:>11}", "stage", "cosine", "max_abs", "rel_l2");
    for (k, c, mx, rl) in &r.rows {
        println!("{k:<40} {c:>14.10} {mx:>11.3e} {rl:>11.3e}");
        if *c < worst.1 {
            worst = (k.clone(), *c);
        }
        if *rl > worst_rel.1 {
            worst_rel = (k.clone(), *rl);
        }
    }
    // Both extremes, because at this parity level every cosine rounds to
    // 1.0000000000 and `rel_l2` is the only discriminating column left.
    println!(
        "\n{} comparisons ({covered} device taps + 2 host + 20 residuals)\n  worst cosine  {} 1-cos {:.3e}\n  worst rel_l2  {} {:.3e}\n",
        r.rows.len(),
        worst.0,
        1.0 - worst.1,
        worst_rel.0,
        worst_rel.1
    );

    // A golden the graph never compares is a silent hole in the ladder.
    let untapped: Vec<&str> = gold
        .0
        .iter()
        .map(|t| t.name.as_str())
        .filter(|n| !n.starts_with("in.") && !n.starts_with("out.") && !n.starts_with("out0.75."))
        .filter(|n| *n != "time_proj" && *n != "add_time_proj")
        .filter(|n| !r.rows.iter().any(|(k, ..)| k == n))
        .collect();
    assert!(untapped.is_empty(), "{} goldens have no matching tap: {untapped:?}", untapped.len());
    assert!(r.failures.is_empty(), "{} failed:\n  {}", r.failures.len(), r.failures.join("\n  "));
}
