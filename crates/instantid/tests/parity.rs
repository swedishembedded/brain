// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! InstantID Resampler forward parity, stage by stage, against the reference.
//!
//! Goldens come from `tools/goldens/instantid_dump_reference.py`, which imports the
//! UPSTREAM `ip_adapter.resampler.Resampler` rather than reimplementing it and
//! taps `proj_in`, every layer's attention and feed-forward, `proj_out` and
//! `norm_out`.
//!
//! The input is a deliberately NON-unit ArcFace vector: the released embedding
//! has `‖e‖ ≈ 15-20` and the resampler is not scale-invariant, so gating on a
//! unit vector would gate the wrong operating point (and would hide the
//! raw-vs-L2-normalised confusion `pulid::idcond` documents).
//!
//! Run:
//!   BRAIN_INSTANTID=/path/to/instantid/ip-adapter.bin \
//!     cargo test --release -p brain-instantid --test parity -- --nocapture
//!
//! Fixtures resolve from `$BRAIN_TESTDATA` (default `<repo>/testdata`); the test
//! skips itself when either the checkpoint or the goldens are absent.

use std::collections::HashMap;
use std::path::PathBuf;

use instantid::config::ResamplerConfig;
use instantid::model::{Resampler, KERNELS};

/// Cosine gate. Every stage of a correct fp32 replay of this graph sits at
/// 1.0 to ten digits; anything below this is a defect, not noise.
const GATE: f64 = 0.999_999_9;

fn testdata(rel: &str) -> PathBuf {
    let root = std::env::var("BRAIN_TESTDATA")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    PathBuf::from(root).join(rel)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt())
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn resampler_forward_matches_the_reference() {
    let gp = testdata("instantid/resampler.safetensors");
    if !gp.exists() {
        eprintln!("SKIP: {} absent (run tools/goldens/instantid_dump_reference.py)", gp.display());
        return;
    }
    let Ok(ckpt) = std::env::var("BRAIN_INSTANTID") else {
        eprintln!("SKIP: BRAIN_INSTANTID unset (ip-adapter.bin)");
        return;
    };
    if !std::path::Path::new(&ckpt).exists() {
        eprintln!("SKIP: {ckpt} does not exist");
        return;
    }

    // The released file is a torch archive whose nested dicts flatten with '.'
    // joins, so the Resampler's tensors arrive under an `image_proj.` prefix.
    let tensors = checkpoint::torchpt::read(&ckpt).expect("read ip-adapter.bin");
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
    let mut n_sites = 0usize;
    for t in tensors {
        if let Some(rest) = t.name.strip_prefix("image_proj.") {
            shapes.insert(rest.to_string(), t.shape.clone());
            weights.insert(rest.to_string(), t.data);
        } else if t.name.starts_with("ip_adapter.") && t.name.ends_with(".to_k_ip.weight") {
            n_sites += 1;
        }
    }
    assert!(!weights.is_empty(), "no image_proj.* tensors in {ckpt}");

    // Derive the config from the checkpoint, then check it against the shapes
    // the release actually carries — not against a hardcoded expectation.
    let cfg = ResamplerConfig::from_tensors(&shapes).expect("derive config");
    eprintln!(
        "instantid: dim={} depth={} heads={} queries={} embed={} out={} kv_rows={} ({n_sites} ip sites)",
        cfg.dim, cfg.depth, cfg.heads, cfg.num_queries, cfg.embedding_dim, cfg.output_dim, cfg.kv_rows()
    );
    let weights = instantid::import::validate_resampler(weights, &cfg).expect("two-way coverage");

    let g = checkpoint::safetensors::read(gp.to_str().unwrap()).expect("read goldens");
    let golden = |n: &str| -> &Vec<f32> {
        &g.iter().find(|t| t.name == n).unwrap_or_else(|| panic!("golden has no `{n}`")).data
    };

    let gpu = gpu_core::testgpu::dev(KERNELS);
    let m = Resampler::new_on(gpu, cfg.clone(), KERNELS, &weights);
    m.set_embedding(golden("input"));
    m.forward();

    // Every tap the dumper wrote, in graph order.
    let mut stages: Vec<String> = vec!["latents_init".into(), "proj_in".into()];
    for l in 0..cfg.depth {
        stages.push(format!("layer{l}_attn"));
        stages.push(format!("layer{l}_ff"));
    }
    stages.push("proj_out".into());
    stages.push("id_tokens".into());

    let (mut worst, mut worst_at) = (1.0f64, String::new());
    let mut failed = 0usize;
    for name in &stages {
        let want = golden(name);
        let got = m.read_tap(name);
        assert_eq!(got.len(), want.len(), "{name}: got {} floats, want {}", got.len(), want.len());
        let c = cosine(&got, want);
        let ma = max_abs(&got, want);
        eprintln!("  {name:16} cosine={c:.10}  max_abs={ma:.3e}");
        if c < worst {
            worst = c;
            worst_at = name.clone();
        }
        if c < GATE {
            failed += 1;
        }
    }
    eprintln!("Resampler: {} comparisons, {failed} failed, worst {worst_at} at cosine {worst:.10}", stages.len());
    assert_eq!(failed, 0, "worst {worst_at} at cosine {worst:.10}");
}

/// The decoupled branch at both SDXL cross-attention widths.
///
/// The reference computes `k = to_k_ip(id)`, `v = to_v_ip(id)`, attends the
/// image queries over them, and returns the context WITHOUT a `to_out` — the
/// shared one is applied to `text_attn + scale * ip_out` afterwards. This gates
/// exactly that term.
#[test]
fn decoupled_attention_matches_the_reference() {
    let gp = testdata("instantid/resampler.safetensors");
    if !gp.exists() {
        eprintln!("SKIP: {} absent", gp.display());
        return;
    }
    let Ok(ckpt) = std::env::var("BRAIN_INSTANTID") else {
        eprintln!("SKIP: BRAIN_INSTANTID unset");
        return;
    };
    if !std::path::Path::new(&ckpt).exists() {
        eprintln!("SKIP: {ckpt} does not exist");
        return;
    }

    let tensors = checkpoint::torchpt::read(&ckpt).expect("read ip-adapter.bin");
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut data: HashMap<String, Vec<f32>> = HashMap::new();
    for t in tensors {
        if let Some(rest) = t.name.strip_prefix("ip_adapter.") {
            shapes.insert(rest.to_string(), t.shape.clone());
            data.insert(rest.to_string(), t.data);
        }
    }
    let sites = instantid::import::validate_sites(&shapes, data).expect("site coverage");
    eprintln!("instantid: {} decoupled sites", sites.cfg.len());

    let g = checkpoint::safetensors::read(gp.to_str().unwrap()).expect("read goldens");
    let golden = |n: &str| -> Option<&Vec<f32>> { g.iter().find(|t| t.name == n).map(|t| &t.data) };
    let id_tokens = golden("id_tokens").expect("golden id_tokens");

    // Gate whichever sites the dumper chose — it picks ONE PER DISTINCT WIDTH
    // (640 and 1280 on SDXL), because `heads = hidden / 64` differs between them
    // and one width cannot catch a width-dependent bug. Discovering them from
    // the goldens rather than hardcoding indices means a re-dump that changes
    // the representative sites does not silently stop gating a width.
    let mut idxs: Vec<usize> = g
        .iter()
        .filter_map(|t| t.name.strip_prefix("site").and_then(|r| r.strip_suffix("_out")))
        .filter_map(|n| n.parse().ok())
        .collect();
    idxs.sort_unstable();
    assert!(!idxs.is_empty(), "goldens carry no site*_out");

    let gpu = gpu_core::testgpu::dev(KERNELS);
    let mut checked = 0usize;
    let mut widths: Vec<usize> = Vec::new();
    for idx in idxs {
        let Some(want_out) = golden(&format!("site{idx}_out")) else { continue };
        let sc = sites.cfg.iter().find(|s| s.index == idx).expect("site in checkpoint").clone();
        let q = golden(&format!("site{idx}_q")).expect("golden q");
        let n_img = q.len() / sc.hidden;
        widths.push(sc.hidden);

        let a = instantid::model::SiteAttn::new_on(
            gpu.share(),
            KERNELS,
            sc.clone(),
            id_tokens.len() / sc.token_dim,
            n_img,
            &sites.kv[&idx],
        );
        a.set_id(id_tokens);
        let got = a.run(q, n_img);
        let (gk, gv) = a.read_kv();

        for (tag, got, want) in [
            (format!("site{idx}_k"), &gk, golden(&format!("site{idx}_k")).expect("golden k")),
            (format!("site{idx}_v"), &gv, golden(&format!("site{idx}_v")).expect("golden v")),
            (format!("site{idx}_out"), &got, want_out),
        ] {
            let c = cosine(got, want);
            eprintln!("  {tag:14} (hidden {:4})  cosine={c:.10}  max_abs={:.3e}", sc.hidden, max_abs(got, want));
            assert!(c >= GATE, "{tag} cosine {c:.10}");
            checked += 1;
        }
    }
    assert!(checked > 0, "no site goldens were compared");
    widths.sort_unstable();
    widths.dedup();
    assert!(widths.len() >= 2, "only width(s) {widths:?} gated — SDXL has two, and heads = hidden/64 differs");
    eprintln!("Decoupled attention: {checked} comparisons over widths {widths:?}, 0 failed");
}
