// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Int8 (DP4A) DiT parity vs the fp32 forward AND the diffusers reference,
//! replaying the exact captured transformer inputs (the `dit_parity` fixture,
//! t2i: 512 txt + 1024 img = 1536 joint tokens).
//!
//! int8 is the LOSSY tier — the gates are measured, not aspirational.
//! Measured on the Tesla P40 (gpu0): cosine 0.998950 / max_abs 0.81 vs both
//! fp32 and the golden (fp32 matches the golden to 1e-4, so the two
//! comparisons coincide); the asserts gate at 0.998, tighter than the 0.995 /
//! 0.99 the port plan required — see `docs/models/flux2/status.md` for the
//! per-family bisection that bought the margin. Also prints fp32 vs int8
//! single-forward wall times (informational).
//!
//! Env: `BRAIN_FLUX2_TRANSFORMER` = the klein-4B diffusers `transformer/` dir.
//! Skips without weights, fixtures, or a GPU backend (int8 needs DP4A).

use flux2::{Flux2Config, Flux2Model, Precision};

fn testdata(rel: &str) -> String {
    let root = std::env::var("BRAIN_TESTDATA")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    format!("{root}/{rel}")
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn load_weights() -> Option<flux2::Tensors> {
    let Ok(dir) = std::env::var("BRAIN_FLUX2_TRANSFORMER") else {
        eprintln!("SKIP: BRAIN_FLUX2_TRANSFORMER unset");
        return None;
    };
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("transformer dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|q| q.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    let mut tensors = Vec::new();
    for f in files {
        tensors.extend(checkpoint::safetensors::read(f.to_str().unwrap()).unwrap());
    }
    Some(flux2::import_diffusers(tensors, &Flux2Config::klein_4b()).unwrap())
}

/// Replayed reference inputs from the t2i fixture.
struct Case {
    hs: Vec<f32>,
    ctx: Vec<f32>,
    t: f32,
    ids: Vec<u32>,
    n_img: usize,
    want: Vec<f32>,
}

fn load_case(fixture: &str) -> Case {
    let fx = checkpoint::safetensors::read(fixture).expect("read fixture");
    let get = |name: &str| {
        fx.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden {name}"))
    };
    let hs = get("hs");
    let n_img = hs.shape[1];
    let mut ids: Vec<u32> = Vec::with_capacity((512 + n_img) * 4);
    ids.extend(get("txt_ids").data.iter().map(|&v| v as u32));
    ids.extend(get("img_ids").data.iter().map(|&v| v as u32));
    Case {
        hs: hs.data.clone(),
        ctx: get("ctx").data.clone(),
        t: get("timestep").data[0],
        ids,
        n_img,
        want: get("out").data.clone(),
    }
}

/// Forward once, then time `reps` more (the first forward absorbs pipeline
/// compilation; wall time includes the per-forward modulation/RoPE uploads).
fn run_timed(model: &Flux2Model, c: &Case, reps: usize) -> (Vec<f32>, f64) {
    let out = model.forward(&c.hs, &c.ctx, c.t, &c.ids, c.n_img);
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        let _ = model.forward(&c.hs, &c.ctx, c.t, &c.ids, c.n_img);
    }
    (out, t0.elapsed().as_secs_f64() / reps as f64)
}

/// TEMPORARY bisection harness: which linear families cost the parity.
/// Run explicitly: `cargo test --release --test int8_parity -- --ignored --nocapture`
#[test]
#[ignore = "bisection tool, not a gate"]
fn int8_bisect_keep_f32_families() {
    let f_t2i = testdata("flux2/klein-4b/dit.safetensors");
    if !std::path::Path::new(&f_t2i).exists() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(flux2::KERNELS);
    if !gpu.caps().workgroup_reductions {
        return;
    }
    let Some(map) = load_weights() else { return };
    let cfg = Flux2Config::klein_4b();
    let case = load_case(&f_t2i);
    let n_max = 512 + case.n_img as u32;
    // On top of the always-fp32 boundary linears (img_in/txt_in/final_layer).
    for keep in ["", "_mlp.2", "_mlp.2,linear2"] {
        std::env::set_var("BRAIN_FLUX2_I8_KEEP_F32", keep);
        let (out, dt) = {
            let model = Flux2Model::new_with(&cfg, &map, gpu.share(), n_max, Precision::Int8);
            run_timed(&model, &case, 2)
        };
        eprintln!(
            "keep_f32=[{keep}]: cosine_vs_golden={:.6} max_abs={:.4} fwd={dt:.3}s",
            cosine(&out, &case.want),
            max_abs(&out, &case.want)
        );
    }
    std::env::remove_var("BRAIN_FLUX2_I8_KEEP_F32");
}

#[test]
fn int8_forward_matches_fp32_and_reference() {
    let f_t2i = testdata("flux2/klein-4b/dit.safetensors");
    if !std::path::Path::new(&f_t2i).exists() {
        eprintln!("SKIP: fixture {f_t2i} absent");
        return;
    }
    let gpu = gpu_core::testgpu::dev(flux2::KERNELS);
    if !gpu.caps().workgroup_reductions {
        eprintln!("SKIP: int8 needs a GPU backend (DP4A), current is {}", gpu.kind());
        return;
    }
    let Some(map) = load_weights() else { return };
    let cfg = Flux2Config::klein_4b();
    let case = load_case(&f_t2i);
    let n_max = 512 + case.n_img as u32;

    // fp32 reference forward first, then FREE it before the int8 build — both
    // resident would need ~20 GiB and OOM a 24 GiB card alongside other users.
    let (fp32_out, fp32_dt) = {
        let model = Flux2Model::new(&cfg, &map, gpu.share(), n_max);
        run_timed(&model, &case, 3)
    };
    let (int8_out, int8_dt) = {
        let model = Flux2Model::new_with(&cfg, &map, gpu.share(), n_max, Precision::Int8);
        run_timed(&model, &case, 3)
    };

    let cos_fp32 = cosine(&int8_out, &fp32_out);
    let cos_ref = cosine(&int8_out, &case.want);
    eprintln!(
        "int8 vs fp32:   cosine={cos_fp32:.6} max_abs={:.4}",
        max_abs(&int8_out, &fp32_out)
    );
    eprintln!(
        "int8 vs golden: cosine={cos_ref:.6} max_abs={:.4}",
        max_abs(&int8_out, &case.want)
    );
    eprintln!(
        "single forward @{} joint tokens: fp32 {:.3} s, int8 {:.3} s ({:.2}x)",
        n_max,
        fp32_dt,
        int8_dt,
        fp32_dt / int8_dt
    );
    // Measured 0.998950 on the P40; gated with a small margin. (The port plan's
    // floors were 0.995 / 0.99 — the measured number is tighter, so it gates.)
    assert!(cos_fp32 >= 0.998, "int8 vs fp32 cosine {cos_fp32:.6} < 0.998");
    assert!(cos_ref >= 0.998, "int8 vs golden cosine {cos_ref:.6} < 0.998");
}
