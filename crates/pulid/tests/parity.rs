// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Stage-by-stage forward parity for PuLID-FLUX v0.9.1 against the goldens
//! dumped by `tools/goldens/pulid_dump_reference.py`.
//!
//! Three gates, in ladder order:
//!
//! 1. `idformer_stage_parity` — the whole ID-embedding pipeline, from a real
//!    ArcFace embedding and real EVA-CLIP hidden states to the 32 projected ID
//!    tokens, tapped at every internal stage.
//! 2. `ca_stage_parity` — one injected `PerceiverAttentionCA`, tapped at every
//!    internal stage, plus the last module's output and the `id_weight` scaling.
//! 3. `flux1_conditioned_parity` — ONE conditioned transformer evaluation:
//!    the reduced-depth FLUX.1 backbone with the ID injected at the reference
//!    sites, per block boundary and at the output.
//!
//! **End-to-end generation is NOT gated and is not claimed** — `crates/flux1`
//! has no sampler loop and no VAE glue, so there is no image to compare.
//! Neither is full-depth conditioning: fp32 FLUX.1 is 47.6 GiB and does not fit
//! one 24 GiB card, which is why `crates/flux1`'s own fp32 gate is also reduced
//! depth. The 20-site full-depth schedule is gated as a schedule, in
//! `pulid::config`'s unit tests.
//!
//! Fixtures resolve from `$BRAIN_TESTDATA` (default `<repo>/testdata`); weights
//! are named by env var. Every test SKIPS itself when a fixture or a weight file
//! is absent:
//!
//! ```text
//! BRAIN_PULID=/path/to/pulid_flux_v0.9.1.safetensors
//! BRAIN_FLUX1_TRANSFORMER=/path/to/FLUX.1-Kontext-dev/transformer   # test 3 only
//! ```

use std::path::{Path, PathBuf};

use pulid::config::PulidConfig;
use pulid::model::{IdFormer, PulidCa};

/// Per-stage floor. These are fp32 replays of an fp32 reference through the
/// same arithmetic, so the bar is high; the printed `1-cos` is the deliverable.
const GATE: f64 = 0.999999;

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

/// One table, one failure at the end — a failing stage must not hide the ones
/// behind it.
#[derive(Default)]
struct Report {
    n: usize,
    failures: Vec<String>,
    worst: Option<(f64, String)>,
}

impl Report {
    fn check(&mut self, stage: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{stage}: len {} != golden {}", got.len(), want.len());
        let (c, m, r) = (cosine(got, want), max_abs(got, want), rel_l2(got, want));
        eprintln!(
            "  {stage:<20} cosine={c:.10} (1-cos {:.2e})  max_abs={m:.3e}  rel_l2={r:.3e}",
            1.0 - c
        );
        self.n += 1;
        // cosine alone is scale-invariant, so rel_l2 is asserted too.
        if c.is_nan() || c < GATE {
            self.failures.push(format!("{stage}: cosine {c:.10} < {GATE}"));
        }
        if r.is_nan() || r > 1e-4 {
            self.failures.push(format!("{stage}: rel_l2 {r:.3e} > 1e-4"));
        }
        if self.worst.as_ref().is_none_or(|(w, _)| c < *w) {
            self.worst = Some((c, stage.to_string()));
        }
    }

    fn finish(self, what: &str) {
        let (c, s) = self.worst.clone().unwrap_or((1.0, "none".into()));
        eprintln!("{what}: {} comparisons, {} failed, worst {s} at cosine {c:.10}", self.n, self.failures.len());
        assert!(self.n > 0, "{what}: nothing was compared");
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
            eprintln!("SKIP: golden {} absent (run tools/goldens/pulid_dump_reference.py)", p.display());
            return None;
        }
        Some(Golden { t: checkpoint::safetensors::read(p.to_str().unwrap()).expect("read golden") })
    }
    fn get(&self, name: &str) -> &Vec<f32> {
        &self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden tensor {name}")).data
    }
}

fn env_path(var: &str) -> Option<PathBuf> {
    let v = std::env::var(var).ok().filter(|s| !s.is_empty())?;
    let p = PathBuf::from(&v);
    if !p.exists() {
        eprintln!("SKIP: {var}={v} not found");
        return None;
    }
    Some(p)
}

fn weights(cfg: &PulidConfig) -> Option<pulid::PulidWeights> {
    let p = match env_path("BRAIN_PULID") {
        Some(p) => p,
        None => {
            eprintln!("SKIP: BRAIN_PULID unset (pulid_flux_v0.9.1.safetensors)");
            return None;
        }
    };
    let w = pulid::read(p.to_str().unwrap(), cfg).expect("pulid import");
    eprintln!(
        "imported {} encoder + {} ca tensors ({} modules)",
        w.encoder.len(),
        w.ca.len(),
        w.num_ca
    );
    Some(w)
}

// ---------------------------------------------------------------------------
// 1. the ID-embedding pipeline
// ---------------------------------------------------------------------------

#[test]
fn idformer_stage_parity() {
    let cfg = PulidConfig::v0_9_1();
    let Some(g) = Golden::open("pulid/idformer.safetensors") else { return };
    let Some(w) = weights(&cfg) else { return };

    let gpu = gpu_core::testgpu::dev(pulid::KERNELS);
    let m = IdFormer::new_on(gpu, cfg.clone(), pulid::KERNELS, w.encoder);

    // The inputs are exactly what brain's OWN parity-gated towers produce:
    // `id_cond = cat(arcface ArcFace 512, clip::EvaVision L2-normed cls 768)`
    // and the 5 `clip::EvaVisionConfig::PULID_TAPS` block outputs. This test
    // replays them from the fixtures rather than re-running those two towers,
    // so a failure here is PuLID's, not theirs.
    let arc = g.get("arcface_embedding");
    let eva = g.get("eva_cls_l2norm");
    assert_eq!(arc.len() + eva.len(), cfg.id_cond_dim);
    let mut id_cond = arc.clone();
    id_cond.extend_from_slice(eva);
    assert_eq!(&id_cond, g.get("id_cond"), "id_cond concat order");
    let hidden: Vec<Vec<f32>> = (0..cfg.scales).map(|j| g.get(&format!("vit_hidden{j}")).clone()).collect();
    m.set_inputs(&id_cond, &hidden);

    let mut r = Report::default();
    eprintln!("--- IDFormer ({} layers over {} scales) ---", cfg.depth, cfg.scales);
    r.check("id_tokens", &m.read_tap("id_tokens"), g.get("id_tokens"));
    r.check("latents_in", &m.read_tap("latents_in"), g.get("latents_in"));
    for i in 0..cfg.scales {
        let n = format!("map{i}_out");
        r.check(&n, &m.read_tap(&n), g.get(&n));
    }
    r.check("ctx0", &m.read_tap("ctx0"), g.get("ctx0"));
    for l in 0..cfg.depth {
        for tap in ["attn", "ff"] {
            let n = format!("layer{l}_{tap}");
            r.check(&n, &m.read_tap(&n), g.get(&n));
        }
    }
    m.forward();
    r.check("id_embedding", &m.read_id_embedding(), g.get("id_embedding"));
    r.finish("IDFormer");
}

// ---------------------------------------------------------------------------
// 2. the injected cross-attention
// ---------------------------------------------------------------------------

#[test]
fn ca_stage_parity() {
    let cfg = PulidConfig::v0_9_1();
    let Some(g) = Golden::open("pulid/ca.safetensors") else { return };
    let Some(w) = weights(&cfg) else { return };
    let n_ca = w.num_ca;

    let img = g.get("img");
    let id = g.get("id");
    let n_img = img.len() / cfg.ca_dim;
    let gpu = gpu_core::testgpu::dev(pulid::KERNELS);
    let ca = PulidCa::new_on(gpu, cfg.clone(), pulid::KERNELS, n_ca, n_img, w.ca);
    ca.set_id(id);

    let mut r = Report::default();
    eprintln!("--- PerceiverAttentionCA (module 0, {n_img} image rows) ---");
    let got0 = ca.apply_host(0, img, 1.0);
    r.check("ca0.norm1_id", &ca.read_stage("norm1_id"), g.get("ca0_norm1_id"));
    r.check("ca0.norm2_img", &ca.read_stage("norm2_img"), g.get("ca0_norm2_img"));
    r.check("ca0.q", &ca.read_stage("q"), g.get("ca0_q"));
    r.check("ca0.kv", &ca.read_stage("kv"), g.get("ca0_kv"));
    r.check("ca0.ctx", &ca.read_stage("ctx"), g.get("ca0_ctx"));
    // the injection is `img + id_weight * ca(id, img)`, so at id_weight = 1 the
    // module output is the difference
    let delta0: Vec<f32> = got0.iter().zip(img).map(|(a, b)| a - b).collect();
    r.check("ca0.out", &delta0, g.get("ca0_out"));

    eprintln!("--- PerceiverAttentionCA (module {}) ---", n_ca - 1);
    let gotl = ca.apply_host(n_ca - 1, img, 1.0);
    let deltal: Vec<f32> = gotl.iter().zip(img).map(|(a, b)| a - b).collect();
    r.check(&format!("ca{}.out", n_ca - 1), &deltal, g.get(&format!("ca{}_out", n_ca - 1)));

    // id_weight is a linear scale on the contribution, nothing else
    let half = ca.apply_host(0, img, 0.5);
    let want: Vec<f32> = img.iter().zip(&delta0).map(|(x, d)| x + 0.5 * d).collect();
    r.check("ca0.out@id_weight=0.5", &half, &want);
    r.finish("PerceiverAttentionCA");
}

// ---------------------------------------------------------------------------
// 3. one conditioned transformer evaluation
// ---------------------------------------------------------------------------

/// Read the diffusers `transformer/` shards, dropping out-of-depth blocks as
/// each shard lands (the `flux1::dit_parity` idiom — a reduced-depth run must
/// never materialize 47.6 GiB).
fn load_flux(cfg: &flux1::Flux1Config) -> Option<flux1::Tensors> {
    let dir = match env_path("BRAIN_FLUX1_TRANSFORMER") {
        Some(d) => d,
        None => {
            eprintln!("SKIP: BRAIN_FLUX1_TRANSFORMER unset");
            return None;
        }
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("transformer dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|q| q.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no safetensors in {}", dir.display());
    let mut tensors = Vec::new();
    for f in files {
        let shard = checkpoint::safetensors::read(f.to_str().unwrap()).unwrap();
        let (kept, _) = flux1::truncate_to_depth(shard, cfg);
        tensors.extend(kept);
    }
    Some(flux1::import_diffusers(tensors, cfg).unwrap())
}

#[test]
fn flux1_conditioned_parity() {
    let cfg = PulidConfig::v0_9_1();
    let Some(g) = Golden::open("pulid/flux_cond.safetensors") else { return };
    let Some(fx) = Golden::open("flux1/kontext-dev/dit_small.safetensors") else { return };
    // depth of the truncation the goldens were dumped from
    let mpath = testdata("pulid/manifest.json");
    let Ok(mtxt) = std::fs::read_to_string(&mpath) else {
        eprintln!("SKIP: {} absent", mpath.display());
        return;
    };
    let mv: serde_json::Value = serde_json::from_str(&mtxt).unwrap();
    let sd = mv["params"]["small_double"].as_u64().unwrap() as usize;
    let ss = mv["params"]["small_single"].as_u64().unwrap() as usize;
    let id_weight = mv["params"]["id_weight"].as_f64().unwrap() as f32;

    // The two interval constants and the module count come from the DUMPER, which
    // derived them independently from `flux/model.py`. Checking them here means a
    // drift in `PulidConfig` fails against the reference rather than against
    // another brain-side copy of the same number.
    assert_eq!(
        mv["params"]["double_interval"].as_u64().unwrap() as usize,
        cfg.double_interval,
        "double_interval disagrees with the golden's dumper"
    );
    assert_eq!(
        mv["params"]["single_interval"].as_u64().unwrap() as usize,
        cfg.single_interval,
        "single_interval disagrees with the golden's dumper"
    );
    assert_eq!(
        mv["params"]["num_ca"].as_u64().unwrap() as usize,
        cfg.num_ca(19, 38),
        "the checkpoint's module count disagrees with num_ca(19, 38)"
    );

    let fcfg = flux1::Flux1Config::kontext_dev().with_depth(sd, ss);
    let Some(ts) = load_flux(&fcfg) else { return };
    let Some(w) = weights(&cfg) else { return };

    // ONE device handle, ONE kernel list: flux1's indices are unmoved and
    // PuLID's are appended, so the adapter's steps join the model's own list.
    let ks = pulid::joint_kernels();
    let gpu = gpu_core::testgpu::dev(ks);
    let nt = fx.get("txt_ids").len() / 3;
    let ni = fx.get("img_ids").len() / 3;
    let model = flux1::Flux1Model::new(&fcfg, &ts, gpu.share(), (nt + ni) as u32);
    drop(ts);
    let n_ca = w.num_ca;
    let ca = PulidCa::new_on(gpu, cfg.clone(), ks, n_ca, ni, w.ca);
    ca.set_id(g.get("id"));
    let adapter = pulid::PulidAdapter::new(ca, &cfg, sd, ss, id_weight);
    eprintln!(
        "--- FLUX.1 depth {sd}+{ss}, {} injection site(s) at id_weight {id_weight} ---",
        adapter.schedule().len()
    );
    assert!(!adapter.schedule().is_empty(), "the truncation has no injection site — the gate would be vacuous");

    // The dumper recorded which block indices it fired at, per stream, computed
    // from its own `range(0, keep, interval)`. Gate `schedule()` against THAT
    // rather than only against brain's own restatement of the rule in
    // `config::tests` — and do it from the manifest, so re-dumping the golden at
    // a deeper truncation tightens this automatically.
    //
    // NOTE this is where the forward gate's schedule coverage ends: at 2 + 2 the
    // site list is {double 0, single 0} for ANY double interval >= 2 and ANY
    // single interval >= 2, so the *values* 2 and 4 are gated by the manifest
    // assertions above and by `config::tests`, not by the injected forward.
    let sites = |stream: &str| -> Vec<usize> {
        mv["flux_cond.safetensors"]["ca_indices"][stream]
            .as_array()
            .unwrap_or_else(|| panic!("manifest has no ca_indices.{stream}"))
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    };
    let (want_d, want_s) = (sites("double"), sites("single"));
    let got_d: Vec<usize> =
        adapter.schedule().iter().filter(|s| s.stream == pulid::Stream::Double).map(|s| s.block).collect();
    let got_s: Vec<usize> =
        adapter.schedule().iter().filter(|s| s.stream == pulid::Stream::Single).map(|s| s.block).collect();
    assert_eq!(got_d, want_d, "double-stream sites disagree with the golden's");
    assert_eq!(got_s, want_s, "single-stream sites disagree with the golden's");
    // `ca_idx` is ONE counter over both loops: the singles continue where the
    // doubles stopped, they do not restart and they do not jump to 10.
    assert_eq!(
        adapter.schedule().iter().map(|s| s.ca).collect::<Vec<_>>(),
        (0..adapter.schedule().len()).collect::<Vec<_>>(),
        "the shared ca_idx counter is not sequential over the two loops"
    );

    let hs = fx.get("hs");
    let ctx = fx.get("ctx");
    let pooled = fx.get("pooled");
    let t = fx.get("timestep")[0];
    let guidance = fx.get("guidance")[0];
    let mut ids: Vec<u32> = fx.get("txt_ids").iter().map(|&v| v as u32).collect();
    ids.extend(fx.get("img_ids").iter().map(|&v| v as u32));

    let d = fcfg.hidden;
    let mut r = Report::default();

    // (a) the SAME model with no adapter must still reproduce the plain golden
    // — an anchor proving the seam is inert when nothing is injected.
    let uncond = model.forward(hs, ctx, pooled, t, guidance, &ids, ni);
    r.check("uncond.out", &uncond, g.get("out_uncond"));

    // (b) the conditioned evaluation, per block boundary and at the output
    let (cond, tr) =
        model.forward_traced_injected(hs, ctx, pooled, t, guidance, &ids, ni, &adapter);
    for (name, slab) in &tr.stages {
        let (txt, img) = slab.split_at(nt * d);
        r.check(&format!("cond.{name}_txt"), txt, g.get(&format!("cond_{name}_txt")));
        r.check(&format!("cond.{name}_img"), img, g.get(&format!("cond_{name}_img")));
    }
    r.check("cond.out", &cond, g.get("out_cond"));

    // (c) the injection must actually have moved the prediction — otherwise
    // (b) would pass on a no-op adapter.
    let moved = max_abs(&cond, &uncond);
    eprintln!("  conditioned vs unconditioned prediction: max_abs {moved:.4}");
    assert!(moved > 1e-3, "the ID injection changed nothing (max_abs {moved:.3e})");
    r.finish("FLUX.1 + PuLID");
}
