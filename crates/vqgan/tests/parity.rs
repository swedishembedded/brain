// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! VQGAN forward parity vs the `basicsr` reference, replayed stage by stage.
//!
//! Goldens (step 1, `tools/goldens/codeformer_dump_reference.py`) live under
//! `testdata/restore/vqgan/{codeformer,vqgan_code1024}/` and are gitignored;
//! each test skips itself when its fixture is absent. The reference weights are
//! not in `testdata/` either — point **`BRAIN_VQGAN_WEIGHTS`** at the directory
//! holding `codeformer.pth` and `vqgan_code1024.pth`, or the weight-gated tests
//! skip.
//!
//! The two checkpoints' VQGAN weights differ (CodeFormer retrains the encoder:
//! max |Δ| = 1.15), so each variant is gated against its own goldens.
//!
//! `BRAIN_VQGAN_DEVICE=cpu` runs everything on the CPU JIT instead of the
//! pooled test device.

use std::collections::HashMap;

use vqgan::model::Codebook;
use vqgan::{Vqgan, VqganConfig};

/// Resolve a fixture under the fetched `testdata/` tree (override the root with
/// `BRAIN_TESTDATA`).
use brain_testutil::testdata;

type Golden = HashMap<String, (Vec<usize>, Vec<f32>)>;

fn load(path: &str) -> Option<Golden> {
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: fixture {path} absent (step-1 goldens are gitignored)");
        return None;
    }
    Some(
        checkpoint::safetensors::read(path)
            .unwrap_or_else(|e| panic!("read {path}: {e}"))
            .into_iter()
            .map(|t| (t.name, (t.shape, t.data)))
            .collect(),
    )
}

/// The reference checkpoint for `variant`, or `None` (test skips).
fn weights(variant: &str) -> Option<String> {
    let dir = match std::env::var("BRAIN_VQGAN_WEIGHTS") {
        Ok(d) if !d.is_empty() => d,
        _ => {
            eprintln!("SKIP: set BRAIN_VQGAN_WEIGHTS to the dir holding codeformer.pth");
            return None;
        }
    };
    let p = format!("{dir}/{variant}.pth");
    if !std::path::Path::new(&p).exists() {
        eprintln!("SKIP: {p} not found");
        return None;
    }
    Some(p)
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).fold(0.0, f64::max)
}

/// Relative L2 error `‖got − want‖ / ‖want‖`.
///
/// Cosine on its own is **scale-invariant**: a stage that is uniformly 2× the
/// reference — a dropped `1/√C`, a doubled residual, a bias applied twice —
/// still reports cosine 1.000000000. Every gate below therefore also carries
/// this, which is scale-sensitive, so a wrong magnitude cannot pass as a right
/// direction.
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&x, &y) in got.iter().zip(want) {
        num += (x as f64 - y as f64).powi(2);
        den += (y as f64).powi(2);
    }
    if den == 0.0 {
        return if num == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (num / den).sqrt()
}

/// One stage comparison; gates on cosine AND relative L2.
struct Report {
    rows: Vec<(String, f64, f64, f64)>,
}

impl Report {
    fn new() -> Report {
        Report { rows: Vec::new() }
    }
    fn add(&mut self, label: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{label}: {} values vs golden {}", got.len(), want.len());
        self.rows.push((
            label.to_string(),
            cosine(got, want),
            max_abs_diff(got, want),
            rel_l2(got, want),
        ));
    }
    /// Print every stage, then assert the worst cosine clears `floor` and the
    /// worst relative L2 clears `rel_floor`.
    fn finish(&self, title: &str, floor: f64, rel_floor: f64) {
        println!("\n=== {title} ===");
        let mut worst = (f64::INFINITY, String::new());
        let mut worst_rel = (0.0f64, String::new());
        for (name, cos, mad, rel) in &self.rows {
            println!(
                "  {name:<34} cosine {cos:.9} (1-cos {:.2e})  max|Δ| {mad:.3e}  relL2 {rel:.3e}",
                1.0 - cos
            );
            if *cos < worst.0 {
                worst = (*cos, name.clone());
            }
            if *rel > worst_rel.0 {
                worst_rel = (*rel, name.clone());
            }
        }
        println!("  worst: {} at cosine {:.9} (1-cos {:.2e})", worst.1, worst.0, 1.0 - worst.0);
        println!("  worst relative L2: {} at {:.3e}", worst_rel.1, worst_rel.0);
        assert!(worst.0 >= floor, "{title}: {} cosine {:.9} < {floor}", worst.1, worst.0);
        assert!(
            worst_rel.0 <= rel_floor,
            "{title}: {} relative L2 {:.3e} > {rel_floor:.0e} — the direction matches but the \
             MAGNITUDE does not",
            worst_rel.1,
            worst_rel.0
        );
    }
}

/// Serializes the 512² cases.
///
/// Each one builds a `taps = true` model, which pins every activation (the
/// builder's pool is off), and a 512² graph with all 50 block outputs live is
/// **6.9 GB measured on a P40**. The default `--test-threads` is the core count
/// (48 on this box), so all three ran at once and the suite peaked at
/// **22.2 GB of the card's 24.5 GB** — 90% occupancy, with a full-suite run
/// observed failing once under concurrent GPU load and not reproducing in 11
/// retries. Serialized, the suite peaks at ~7 GB and leaves the card usable.
fn heavy() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn gpu() -> gpu_core::Gpu {
    match std::env::var("BRAIN_VQGAN_DEVICE").as_deref() {
        Ok("cpu") => gpu_core::Gpu::new_cpu(&vqgan::KERNELS),
        _ => gpu_core::testgpu::dev(&vqgan::KERNELS),
    }
}

fn imported(variant: &str) -> Option<vae::blocks::Tensors> {
    let path = weights(variant)?;
    let cfg = VqganConfig::codeformer();
    let im = vqgan::import::load(&path, &cfg).unwrap_or_else(|e| panic!("import {path}: {e}"));
    println!(
        "{variant}: imported {} tensors, skipped {} CodeFormer-only",
        im.tensors.len(),
        im.skipped.len()
    );
    Some(im.tensors)
}

// ---------------------------------------------------------------------------
// 1. the quantizer alone (no encoder in the loop) — goldens carry a fixed
//    seeded `z`, so this needs no reference weights beyond the codebook, which
//    the fixture itself ships.
// ---------------------------------------------------------------------------

fn quantizer_unit(variant: &str) {
    let Some(g) = load(&testdata(&format!("restore/vqgan/{variant}/quantizer.safetensors"))) else {
        return;
    };
    let (cb_shape, cb) = &g["codebook"];
    let (k, d) = (cb_shape[0] as u32, cb_shape[1] as u32);
    let mut rep = Report::new();
    for unit in ["u4", "u16"] {
        let (fshape, z_flat) = &g[&format!("{unit}.z_flat")];
        let m = fshape[0] as u32;
        let cbk = Codebook::new(gpu(), cb, k, d, m);

        let (idx, dist) = cbk.assign(z_flat);
        let want_idx: Vec<u32> = g[&format!("{unit}.indices")].1.iter().map(|&v| v as u32).collect();
        let flips = idx.iter().zip(&want_idx).filter(|(a, b)| a != b).count();
        println!("{variant}/{unit}: {m} queries, {flips} index mismatches vs reference argmin");
        assert_eq!(flips, 0, "{variant}/{unit}: argmin disagrees with the reference");

        let want_dist = &g[&format!("{unit}.min_dist")].1;
        rep.add(&format!("{unit}.min_dist"), &dist, want_dist);

        // `lookup` returns [m, d] rows; the golden `codebook_feat` is NCHW
        // [d, h, w] — permute the rows back the way the model does.
        let rows = cbk.lookup(&idx);
        let hw = m as usize;
        let mut chw = vec![0.0f32; rows.len()];
        for t in 0..hw {
            for c in 0..d as usize {
                chw[c * hw + t] = rows[t * d as usize + c];
            }
        }
        rep.add(&format!("{unit}.codebook_feat"), &chw, &g[&format!("{unit}.codebook_feat")].1);
        // `quantize.forward`'s straight-through z_q = z + (z_q - z).
        rep.add(&format!("{unit}.z_q"), &chw, &g[&format!("{unit}.z_q")].1);
    }
    rep.finish(&format!("{variant} quantizer unit"), 0.999999, 1e-4);
}

#[test]
fn quantizer_unit_codeformer() {
    quantizer_unit("codeformer");
}

#[test]
fn quantizer_unit_vqgan_code1024() {
    quantizer_unit("vqgan_code1024");
}

// ---------------------------------------------------------------------------
// 2. every block, 128x128 — the full 25+25 ladder plus the sub-block taps.
// ---------------------------------------------------------------------------

fn stages_128(variant: &str) {
    let Some(g) = load(&testdata(&format!("restore/vqgan/{variant}/stages_128.safetensors"))) else {
        return;
    };
    let Some(t) = imported(variant) else { return };
    let cfg = VqganConfig::codeformer();
    let m = Vqgan::new(cfg.clone(), &t, 128, 128, gpu(), true);
    let r = m.reconstruct(&g["input"].1);

    let mut rep = Report::new();
    let tap = |name: &str| m.read_tap(name).unwrap_or_else(|| panic!("no tap {name}"));

    for i in 0..cfg.encoder_blocks().len() {
        let key = format!("enc.{i:02}");
        rep.add(&key, &tap(&format!("encoder.blocks.{i}")), &g[&key].1);
    }
    rep.add("vq.z_flat", &tap("z_flat"), &g["vq.z_flat"].1);
    rep.add("quant.z_q", &tap("z_q"), &g["quant.z_q"].1);
    rep.add("vq.min_dist", &r.min_dist, &g["vq.min_dist"].1);
    for i in 0..cfg.generator_blocks().len() {
        let key = format!("gen.{i:02}");
        rep.add(&key, &tap(&format!("generator.blocks.{i}")), &g[&key].1);
    }
    rep.add("output", &r.image, &g["output"].1);

    // Sub-block taps: the reference dumped a plain ResBlock (enc 1), a
    // shortcut ResBlock (enc 4) and an AttnBlock (enc 17).
    for leaf in ["norm1", "conv1", "norm2", "conv2"] {
        rep.add(
            &format!("sub.res01.{leaf}"),
            &tap(&format!("encoder.blocks.1.{leaf}")),
            &g[&format!("sub.res01.{leaf}")].1,
        );
        rep.add(
            &format!("sub.res_sc04.{leaf}"),
            &tap(&format!("encoder.blocks.4.{leaf}")),
            &g[&format!("sub.res_sc04.{leaf}")].1,
        );
    }
    rep.add(
        "sub.res_sc04.conv_out",
        &tap("encoder.blocks.4.conv_out"),
        &g["sub.res_sc04.conv_out"].1,
    );
    // q/k/v are fused into one 1x1 conv by the shared attention block, so only
    // the pre-norm and the output projection are separately observable.
    rep.add("sub.attn17.norm", &tap("encoder.blocks.17.norm"), &g["sub.attn17.norm"].1);
    rep.add(
        "sub.attn17.proj_out",
        &tap("encoder.blocks.17.proj_out"),
        &g["sub.attn17.proj_out"].1,
    );

    let want_idx: Vec<u32> = g["indices"].1.iter().map(|&v| v as u32).collect();
    let flips = r.indices.iter().zip(&want_idx).filter(|(a, b)| a != b).count();
    println!("{variant} stages_128: {}/{} index mismatches", flips, want_idx.len());
    assert_eq!(flips, 0, "{variant} stages_128: code assignment differs from the reference");

    rep.finish(&format!("{variant} stages 128x128"), 0.9999, 1e-4);
}

#[test]
fn stages_128_codeformer() {
    stages_128("codeformer");
}

#[test]
fn stages_128_vqgan_code1024() {
    stages_128("vqgan_code1024");
}

// ---------------------------------------------------------------------------
// 2b. THE PRODUCTION PATH. Every test above builds with `taps = true`, which
//     pins every activation and therefore DISABLES `vae::blocks::Builder`'s
//     buffer pool. Callers outside the tests pass `taps = false`, so the graph
//     they actually run — the one where activations are ALIASED — was never
//     compared to anything. A `free` issued one step early is invisible in the
//     tapped build and silently corrupts the pooled one.
//
//     Bit-equality (not cosine) is the right gate: pooling only changes WHICH
//     buffer a step writes, never the arithmetic, so any difference at all is a
//     lifetime bug.
// ---------------------------------------------------------------------------

fn pooled_matches_tapped(variant: &str) {
    let Some(g) = load(&testdata(&format!("restore/vqgan/{variant}/stages_128.safetensors"))) else {
        return;
    };
    let Some(t) = imported(variant) else { return };
    let cfg = VqganConfig::codeformer();

    let tapped = Vqgan::new(cfg.clone(), &t, 128, 128, gpu(), true);
    let a = tapped.reconstruct(&g["input"].1);
    let a_latent = tapped.latent();
    let a_zq = tapped.quantized();

    let pooled = Vqgan::new(cfg, &t, 128, 128, gpu(), false);
    let b = pooled.reconstruct(&g["input"].1);
    let b_latent = pooled.latent();
    let b_zq = pooled.quantized();

    assert_eq!(a.indices, b.indices, "{variant}: pooled graph assigns different codes");
    assert_eq!(a.min_dist, b.min_dist, "{variant}: pooled graph has a different min_dist");
    assert_eq!(
        max_abs_diff(&a.image, &b.image),
        0.0,
        "{variant}: pooled reconstruction differs from the tapped one \
         (cosine {:.9}) — a buffer is freed before its last read",
        cosine(&a.image, &b.image)
    );
    // `latent()` / `quantized()` read buffers that are NEVER freed; if the pool
    // ever handed one of them out as scratch these would come back overwritten.
    assert_eq!(max_abs_diff(&a_latent, &b_latent), 0.0, "{variant}: latent() aliased by the pool");
    assert_eq!(max_abs_diff(&a_zq, &b_zq), 0.0, "{variant}: quantized() aliased by the pool");
    // …and the pooled latent is still the reference's encoder output.
    let latent_gap = max_abs_diff(&b_latent, &g["enc.24"].1);
    assert!(latent_gap < 1e-3, "{variant}: latent() vs reference enc.24 max|Δ| {latent_gap:.3e}");

    // `generate` submits a SUFFIX of the pooled graph; its scratch must still
    // be sound when the gather's steps did not run.
    let again = pooled.generate(&b_zq);
    assert_eq!(
        max_abs_diff(&again, &b.image),
        0.0,
        "{variant}: generate(z_q) != decode(indices) on the pooled graph"
    );
    println!("{variant}: pooled (taps=false) graph is bit-identical to the tapped one");
}

#[test]
fn pooled_matches_tapped_codeformer() {
    pooled_matches_tapped("codeformer");
}

#[test]
fn pooled_matches_tapped_vqgan_code1024() {
    pooled_matches_tapped("vqgan_code1024");
}

// ---------------------------------------------------------------------------
// 3. the real 512x512 configuration, synthetic and real-face inputs.
// ---------------------------------------------------------------------------

fn e2e_512(variant: &str, fixture: &str) {
    let Some(g) = load(&testdata(&format!("restore/vqgan/{variant}/{fixture}.safetensors"))) else {
        return;
    };
    let Some(t) = imported(variant) else { return };
    let _heavy = heavy();
    let cfg = VqganConfig::codeformer();
    let m = Vqgan::new(cfg, &t, 512, 512, gpu(), true);
    let r = m.reconstruct(&g["input"].1);

    let mut rep = Report::new();
    let tap = |name: &str| m.read_tap(name).unwrap_or_else(|| panic!("no tap {name}"));
    let mut keys: Vec<&String> = g.keys().filter(|k| k.starts_with("enc.") || k.starts_with("gen.")).collect();
    keys.sort();
    for key in keys {
        let (net, i) = key.split_once('.').expect("enc.NN / gen.NN");
        let net = if net == "enc" { "encoder" } else { "generator" };
        let i: usize = i.parse().expect("block index");
        rep.add(key, &tap(&format!("{net}.blocks.{i}")), &g[key].1);
    }
    rep.add("quant.z_q", &tap("z_q"), &g["quant.z_q"].1);
    rep.add("vq.min_dist", &r.min_dist, &g["vq.min_dist"].1);
    rep.add("output", &r.image, &g["output"].1);

    let want_idx: Vec<u32> = g["indices"].1.iter().map(|&v| v as u32).collect();
    let flips = r.indices.iter().zip(&want_idx).filter(|(a, b)| a != b).count();
    println!("{variant}/{fixture}: {}/{} index mismatches", flips, want_idx.len());
    assert_eq!(flips, 0, "{variant}/{fixture}: code assignment differs from the reference");

    // `generate` (the CodeFormer seam) must reproduce `decode` from the same
    // quantized latent.
    let again = m.generate(&m.quantized());
    rep.add("generate(z_q) vs decode", &again, &r.image);

    // Looser relative-L2 floor than the 128² ladder (1e-4) for one measured
    // reason: at 512² the CPU JIT cannot run `gn_stats_wg` (two workgroup
    // barriers), so every GroupNorm falls back to `gn_stats`, which sums a
    // group's up-to-16 M elements as ONE serial ascending run instead of a
    // 256-way tree. Measured worst relative L2 at 512²: **1.8e-5 on the GPU,
    // 4.3e-4 on the CPU JIT** — a 24x accuracy gap that is pure summation
    // order, not a port defect (indices still match 0/256, cosine still
    // 0.9999999). 3e-3 keeps ~7x headroom over the worst backend while staying
    // ~300x tighter than any magnitude error worth the name.
    rep.finish(&format!("{variant} {fixture}"), 0.9999, 3e-3);
}

#[test]
fn e2e_512_synth_codeformer() {
    e2e_512("codeformer", "e2e_512_synth");
}

#[test]
fn e2e_512_face_codeformer() {
    e2e_512("codeformer", "e2e_512_face");
}

#[test]
fn e2e_512_face_vqgan_code1024() {
    e2e_512("vqgan_code1024", "e2e_512_face");
}
