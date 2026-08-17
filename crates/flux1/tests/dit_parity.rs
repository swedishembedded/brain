// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.1 / Kontext forward parity vs the diffusers reference, replaying the
//! exact transformer inputs and EVERY block boundary captured by
//! `tools/goldens/flux1_dump_reference.py` (forward hooks during a real
//! `FluxKontextPipeline` run).
//!
//! Two gates, because the fp32 model does not fit one card:
//!
//! * **reduced depth, fp32** (`dit_small*.safetensors`) — the first
//!   `small_double` double + `small_single` single blocks of the REAL
//!   checkpoint, goldens dumped from a model truncated the same way. ~3.9 GiB
//!   of weights. This is the exact-math gate: it proves the modulation
//!   permutation, the bias placement, the GELU MLPs, the 3-axis RoPE, the joint
//!   attention, the column-split `linear2` and the final head.
//! * **full depth, int8** (`dit*.safetensors`, `BRAIN_FLUX1_FULL=1`) — all
//!   19 + 38 blocks at ~12 GiB. fp32 would be 47.6 GiB and does not fit a
//!   24 GiB card, so this run reports the *quantized* cosine and is honest
//!   about it; the fp32 full-depth number is NOT measured here.
//!
//! Env: `BRAIN_FLUX1_TRANSFORMER` = the `FLUX.1-Kontext-dev/transformer` dir.
//! Skips itself without weights or fixtures.

use flux1::{Flux1Config, Flux1Model, Precision};

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

/// Read the diffusers `transformer/` shards, dropping out-of-depth blocks as
/// each shard lands so a reduced-depth run never materializes 47.6 GiB.
fn load(cfg: &Flux1Config) -> Option<flux1::Tensors> {
    let Ok(dir) = std::env::var("BRAIN_FLUX1_TRANSFORMER") else {
        brain_testutil::skip("BRAIN_FLUX1_TRANSFORMER unset");
        return None;
    };
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("transformer dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|q| q.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no safetensors in {dir}");
    let (mut tensors, mut dropped, mut seen) = (Vec::new(), 0usize, 0usize);
    for fpath in files {
        let shard = checkpoint::safetensors::read(fpath.to_str().unwrap()).unwrap();
        seen += shard.len();
        let (kept, d) = flux1::truncate_to_depth(shard, cfg);
        dropped += d;
        tensors.extend(kept);
    }
    eprintln!("checkpoint: {seen} tensors, {dropped} dropped as out-of-depth");
    if cfg.depth_double == 19 && cfg.depth_single == 38 {
        // the released FLUX.1-dev / Kontext-dev transformer
        assert_eq!(seen, 1160, "unexpected checkpoint tensor count");
        assert_eq!(dropped, 0);
    }
    let ts = flux1::import_diffusers(tensors, cfg).unwrap();
    assert_eq!(ts.len(), cfg.tensor_manifest().len());
    Some(ts)
}

struct Fixture {
    tensors: Vec<checkpoint::safetensors::StTensor>,
}

impl Fixture {
    fn open(path: &str) -> Option<Fixture> {
        if !std::path::Path::new(path).exists() {
            brain_testutil::skip(&format!("fixture {path} absent (run tools/goldens/flux1_dump_reference.py)"));
            return None;
        }
        Some(Fixture { tensors: checkpoint::safetensors::read(path).unwrap() })
    }
    fn get(&self, name: &str) -> &checkpoint::safetensors::StTensor {
        self.tensors.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden {name}"))
    }
    fn opt(&self, name: &str) -> Option<&checkpoint::safetensors::StTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

/// Replay one dumped run stage by stage. `floor` is the per-stage cosine the
/// tier is required to clear.
fn run_case(model: &Flux1Model, cfg: &Flux1Config, path: &str, floor: f64) -> bool {
    let Some(fx) = Fixture::open(path) else { return false };
    let hs = fx.get("hs");
    let ctx = fx.get("ctx");
    let pooled = fx.get("pooled");
    let t = fx.get("timestep").data[0];
    let guidance = fx.get("guidance").data[0];
    assert!((0.0..=1.0).contains(&t), "timestep {t} not in [0,1]");
    let want = &fx.get("out").data;

    let nt = ctx.shape[0];
    let ni = hs.shape[0];
    // the reference's OWN ids, text rows first, f32 -> u32
    let mut ids: Vec<u32> = Vec::with_capacity((nt + ni) * 3);
    ids.extend(fx.get("txt_ids").data.iter().map(|&v| v as u32));
    ids.extend(fx.get("img_ids").data.iter().map(|&v| v as u32));

    // the transformer predicts every image row; the pipeline truncates to the
    // noise span afterwards, so the golden covers all `ni` rows
    let (got, tr) =
        model.forward_traced(&hs.data, &ctx.data, &pooled.data, t, guidance, &ids, ni);

    let d = cfg.hidden;
    let mut worst = (1.0f64, String::new());
    let mut check = |name: &str, got: &[f32], want: &[f32]| {
        assert_eq!(got.len(), want.len(), "{name}: length");
        let c = cosine(got, want);
        // print 1-cosine too: at this accuracy `1.000000000` hides the digits
        // that distinguish "exact" from "nearly exact"
        eprintln!("  {name:<14} cosine {c:.12}  (1-cos {:.3e})  max_abs {:.5}", 1.0 - c, max_abs(got, want));
        if c < worst.0 {
            worst = (c, name.to_string());
        }
    };

    check("temb", &tr.temb, &fx.get("temb").data);
    for (name, slab) in &tr.stages {
        let (txt, img) = slab.split_at(nt * d);
        if let Some(g) = fx.opt(&format!("{name}_txt")) {
            check(&format!("{name}_txt"), txt, &g.data);
        }
        if let Some(g) = fx.opt(&format!("{name}_img")) {
            check(&format!("{name}_img"), img, &g.data);
        }
    }
    check("pre_final", &tr.pre_final, &fx.get("pre_final").data);
    check("out", &got, want);

    eprintln!("{path}: worst stage {} at cosine {:.12}", worst.1, worst.0);
    assert!(worst.0 >= floor, "worst stage {} cosine {:.12} < {floor}", worst.1, worst.0);
    true
}

/// The exact-math gate: reduced depth, fp32, real weights.
#[test]
fn reduced_depth_fp32_parity() {
    let manifest = testdata("flux1/kontext-dev/manifest.json");
    let (sd, ss) = match std::fs::read_to_string(&manifest) {
        Ok(s) => {
            let v: serde_json::Value = serde_json::from_str(&s).unwrap();
            (
                v["params"]["small_double"].as_u64().unwrap() as usize,
                v["params"]["small_single"].as_u64().unwrap() as usize,
            )
        }
        Err(_) => {
            brain_testutil::skip(&format!("{manifest} absent (run tools/goldens/flux1_dump_reference.py)"));
            return;
        }
    };
    let cfg = Flux1Config::kontext_dev().with_depth(sd, ss);
    let Some(ts) = load(&cfg) else { return };
    let gpu = gpu_core::testgpu::dev(flux1::KERNELS);
    // t2i: 256 txt + 256 img; edit: 256 txt + 512 img (256 noise + 256 ref)
    let model = Flux1Model::new(&cfg, &ts, gpu, 1024);
    let mut ran = false;
    for f in ["dit_small.safetensors", "dit_small_edit.safetensors"] {
        eprintln!("--- {f} (fp32, depth {sd}+{ss}) ---");
        ran |= run_case(&model, &cfg, &testdata(&format!("flux1/kontext-dev/{f}")), 0.9999);
    }
    assert!(ran, "no fixture ran");
}

/// The full 12 B model at int8 (~12 GiB). Opt-in: it needs ~60 GiB of host RAM
/// to import and a >=16 GiB card. Reports the measured cosine; the fp32
/// full-depth forward is NOT run here (47.6 GiB does not fit one P40).
#[test]
fn full_depth_int8_parity() {
    if std::env::var("BRAIN_FLUX1_FULL").is_err() {
        brain_testutil::skip_unavailable("set BRAIN_FLUX1_FULL=1 for the full-depth int8 run");
        return;
    }
    let cfg = Flux1Config::kontext_dev();
    let Some(ts) = load(&cfg) else { return };
    let gpu = gpu_core::testgpu::dev(flux1::KERNELS);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable(&format!("int8 needs a GPU backend, current is {}", gpu.kind()));
        return;
    }
    let model = Flux1Model::new_with(&cfg, &ts, gpu, 1024, Precision::Int8);
    drop(ts);
    let mut ran = false;
    for f in ["dit.safetensors", "dit_edit.safetensors"] {
        eprintln!("--- {f} (int8, full depth) ---");
        // int8 is a lossy tier; the floor only catches a BROKEN port, and the
        // per-stage numbers printed above are the deliverable.
        ran |= run_case(&model, &cfg, &testdata(&format!("flux1/kontext-dev/{f}")), 0.95);
    }
    assert!(ran, "no fixture ran");
}
