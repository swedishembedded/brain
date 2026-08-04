// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Stage-by-stage forward parity for the SDXL `UNet2DConditionModel` against
//! the goldens dumped by `tools/sdxl_dump_reference.py`.
//!
//! Fixtures live under `$BRAIN_TESTDATA` (default `<repo>/testdata`) in
//! `sdxl/`; the reference WEIGHTS are named by env var. The test **SKIPS
//! itself** (never fails) when either is absent:
//!
//! ```text
//! BRAIN_SDXL=/path/to/stable-diffusion-xl-base-1.0
//! python3 tools/sdxl_dump_reference.py --sdxl $BRAIN_SDXL --out testdata/sdxl
//! ```
//!
//! VRAM: the UNet is 2.567 G parameters, so **fp32 weights are ~10.3 GB** and
//! this needs a card with that much free. `BRAIN_DEVICE=gpu1` picks the second
//! P40 when the first is busy.
//!
//! Scope, stated up front so no number here is over-read: this gates ONE
//! forward evaluation of the transformer at a 32x32 latent (a 256x256 image),
//! batch 1, 77 text tokens, fp32. It does **not** gate the 128x128 latent SDXL
//! generates at, batch > 1, int8, the VAE or the text encoders, a sampling
//! loop, or anything about speed.

use std::path::{Path, PathBuf};

use unet::config::UNetConfig;
use unet::model::{Unet, KERNELS};

/// Per-stage direction gate. The reference is fp32 torch on the CPU; brain's
/// reduction order differs, so exact equality is not the expectation.
const GATE: f64 = 0.9999;

/// Per-stage MAGNITUDE gate, and it is not redundant with [`GATE`]: cosine is
/// scale-invariant, so `got = 1.05 · want` scores a perfect 1.0. Every whole-
/// tensor scale mistake this graph can make — a dropped `output_scale_factor`,
/// an attention `1/sqrt(head_dim)` applied twice, a GroupNorm gain read from
/// the wrong buffer — is exactly of that shape, and a cosine-only ladder passes
/// all of them. The worst stage measured on a P40 is 1.54e-5, so 1e-3 leaves
/// ~65x of headroom over reduction-order noise while still catching a 0.1%
/// systematic rescale.
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
fn sdxl_unet_forward_matches_diffusers() {
    let g = testdata("sdxl/unet/stages.safetensors");
    if !g.exists() {
        eprintln!("SKIP: {} absent (run tools/sdxl_dump_reference.py)", g.display());
        return;
    }
    let Ok(weights) = std::env::var("BRAIN_SDXL") else {
        eprintln!("SKIP: BRAIN_SDXL unset (point it at stable-diffusion-xl-base-1.0)");
        return;
    };
    let unet_dir = Path::new(&weights).join("unet");
    if !unet_dir.exists() {
        eprintln!("SKIP: {} absent", unet_dir.display());
        return;
    }
    let gold = Golden(
        checkpoint::safetensors::read(g.to_str().expect("utf-8 path")).expect("read golden"),
    );

    let cfg = UNetConfig::sdxl_base();
    // The golden pins the latent size and the token count; deriving them
    // instead of hardcoding keeps the test valid if the dumper is re-run at a
    // different `--latent`.
    let n_sample = gold.need("in.sample").len() as u32;
    let latent = ((n_sample / cfg.in_channels) as f64).sqrt() as u32;
    assert_eq!(latent * latent * cfg.in_channels, n_sample, "in.sample is not square");
    let t_enc = gold.need("in.encoder_hidden_states").len() as u32 / cfg.cross_attention_dim;
    let timestep = gold.need("in.timestep")[0];
    let time_ids = gold.need("in.time_ids").to_vec();
    let pooled = gold.need("in.text_embeds").to_vec();
    let enc = gold.need("in.encoder_hidden_states").to_vec();
    let sample = gold.need("in.sample").to_vec();
    println!("latent {latent}x{latent}, t_enc {t_enc}, timestep {timestep}");

    let mut r = Report { rows: Vec::new(), failures: Vec::new() };

    // ---- host conditioning first: it gates `hostemb` on its own, and every
    // stage below depends on it, so a failure here explains everything.
    let te = model::hostmath::timestep_embedding(
        timestep,
        cfg.block_out_channels[0] as usize,
        cfg.flip_sin_to_cos,
        cfg.freq_shift as f64,
        10_000.0,
    );
    r.check("time_proj", &te, gold.need("time_proj"));
    let add = unet::hostemb::added_cond(
        &pooled,
        &time_ids,
        cfg.addition_time_embed_dim,
        cfg.flip_sin_to_cos,
        cfg.freq_shift,
    );
    // The golden `add_time_proj` is [6, 256] — the six sinusoids alone, before
    // the pooled text is prepended. Checking that slice separately is what
    // pins the CONCAT ORDER rather than just the total width.
    r.check("add_time_proj", &add[pooled.len()..], gold.need("add_time_proj"));

    println!("importing weights from {} ...", unet_dir.display());
    let tensors = unet::import::load(unet_dir.to_str().expect("utf-8 path"), &cfg).expect("import");
    println!("imported {} tensors; building the graph ...", tensors.len());

    let m = Unet::new(gpu_core::testgpu::dev(&KERNELS), cfg.clone(), &tensors, latent, latent, t_enc, true);
    drop(tensors);
    println!("{} steps; running ...", m.steps().len());
    let out = m.run(&sample, timestep, &enc, &pooled, &time_ids);

    // ---- every tap that has a golden, in record order --------------------
    let mut covered = 0usize;
    for name in m.tap_names().iter().map(|s| s.to_string()).collect::<Vec<_>>() {
        let Some(want) = gold.get(&name) else { continue };
        let got = m.read_tap(&name).expect("tap exists");
        r.check(&name, &got, want);
        covered += 1;
    }
    r.check("out.sample", &out, gold.need("out.sample"));

    let mut worst = ("".to_string(), 1.0f64);
    println!("\n{:<40} {:>14} {:>11} {:>11}", "stage", "cosine", "max_abs", "rel_l2");
    for (k, c, mx, rl) in &r.rows {
        println!("{k:<40} {c:>14.10} {mx:>11.3e} {rl:>11.3e}");
        if *c < worst.1 {
            worst = (k.clone(), *c);
        }
    }
    println!(
        "\n{} comparisons ({covered} device taps + 2 host + out), worst {} at cosine {:.10}\n",
        r.rows.len(),
        worst.0,
        worst.1
    );
    // A golden the graph never taps is a silent hole in the ladder.
    let untapped: Vec<&str> = gold
        .0
        .iter()
        .map(|t| t.name.as_str())
        .filter(|n| !n.starts_with("in.") && !n.starts_with("out.") && *n != "time_proj" && *n != "add_time_proj")
        .filter(|n| !r.rows.iter().any(|(k, ..)| k == n))
        .collect();
    assert!(untapped.is_empty(), "{} goldens have no matching tap: {untapped:?}", untapped.len());
    assert!(r.failures.is_empty(), "{} failed:\n  {}", r.failures.len(), r.failures.join("\n  "));
}

/// Does the whole thing fit ONE card at SDXL's native 1024x1024 (a 128x128
/// latent), in fp32, with no INT8 and no sharding?
///
/// `#[ignore]`d: it needs ~11 GB free on the selected device and there is no
/// golden at that size, so it is a RESIDENCY measurement, not a parity gate.
/// Run it explicitly:
/// ```text
/// BRAIN_DEVICE=gpu1 BRAIN_SDXL=... cargo test --release -p brain-unet \
///     --test parity -- --ignored --nocapture native_resolution
/// ```
#[test]
#[ignore = "residency measurement: ~11 GB of weights, no golden at this size"]
fn native_resolution_fits_one_card() {
    let Ok(weights) = std::env::var("BRAIN_SDXL") else {
        eprintln!("SKIP: BRAIN_SDXL unset");
        return;
    };
    let unet_dir = Path::new(&weights).join("unet");
    if !unet_dir.exists() {
        eprintln!("SKIP: {} absent", unet_dir.display());
        return;
    }
    let cfg = UNetConfig::sdxl_base();
    let tensors = unet::import::load(unet_dir.to_str().expect("utf-8 path"), &cfg).expect("import");
    let params: usize = tensors.values().map(|(_, d)| d.len()).sum();
    println!("{} parameters = {:.2} GB fp32", params, params as f64 * 4.0 / 1e9);
    // Production graph: taps off, so the activation pool is live.
    let m = Unet::new(gpu_core::testgpu::dev(&KERNELS), cfg.clone(), &tensors, 128, 128, 77, false);
    drop(tensors);
    println!("{} steps", m.steps().len());
    let sample = vec![0.5f32; (cfg.in_channels * 128 * 128) as usize];
    let enc = vec![0.1f32; (77 * cfg.cross_attention_dim) as usize];
    let pooled = vec![0.2f32; cfg.pooled_dim() as usize];
    let t0 = std::time::Instant::now();
    let out = m.run(&sample, 601.0, &enc, &pooled, &[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]);
    let dt = t0.elapsed();
    println!("one forward at a 128x128 latent: {:.2} s", dt.as_secs_f64());
    if let Some(s) = m.gpu().stats() {
        println!("device stats: {s:?}");
    }
    assert!(out.iter().all(|v| v.is_finite()), "non-finite output at 128x128");
    assert!(out.iter().map(|v| v * v).sum::<f32>() > 0.0, "zero output at 128x128");
}
