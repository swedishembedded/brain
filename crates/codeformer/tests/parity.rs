// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CodeFormer forward parity vs the `basicsr` reference, replayed stage by
//! stage, at several fidelity weights including both endpoints.
//!
//! Goldens come from `tools/goldens/codeformer_restore_dump_reference.py` and live
//! under `testdata/restore/codeformer/` (gitignored); each test skips itself
//! when its fixture is absent. The reference weights are not in `testdata/`
//! either - point **`BRAIN_CODEFORMER_WEIGHTS`** (or `BRAIN_VQGAN_WEIGHTS`, the
//! same directory) at the directory holding `codeformer.pth`, or the
//! weight-gated tests skip.
//!
//! The VQ autoencoder underneath is already gated by `crates/vqgan`'s own
//! parity suite; what is gated here is everything CodeFormer adds - the
//! code-prediction Transformer, the controllable feature transformation, and
//! the `w` dial's direction.
//!
//! `BRAIN_CODEFORMER_DEVICE=cpu` runs everything on the CPU JIT instead of the
//! pooled test device.

use brain_testutil::parity::WorstTable;
use codeformer::{CodeFormer, CodeFormerConfig};

/// Resolve a fixture under the fetched `testdata/` tree (override the root with
/// `BRAIN_TESTDATA`).
fn testdata(rel: &str) -> String {
    let root = std::env::var("BRAIN_TESTDATA")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    format!("{root}/{rel}")
}

type Golden = std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>;

fn load(rel: &str) -> Option<Golden> {
    let path = testdata(rel);
    if !std::path::Path::new(&path).exists() {
        brain_testutil::skip(&format!("fixture {path} absent (goldens are gitignored)"));
        return None;
    }
    Some(
        checkpoint::safetensors::read(&path)
            .unwrap_or_else(|e| panic!("read {path}: {e}"))
            .into_iter()
            .map(|t| (t.name, (t.shape, t.data)))
            .collect(),
    )
}

/// `codeformer.pth`, or `None` (test skips).
fn weights() -> Option<String> {
    let dir = ["BRAIN_CODEFORMER_WEIGHTS", "BRAIN_VQGAN_WEIGHTS"]
        .iter()
        .find_map(|k| std::env::var(k).ok().filter(|d| !d.is_empty()));
    let Some(dir) = dir else {
        brain_testutil::skip("set BRAIN_CODEFORMER_WEIGHTS to the dir holding codeformer.pth");
        return None;
    };
    let p = format!("{dir}/codeformer.pth");
    if !std::path::Path::new(&p).exists() {
        brain_testutil::skip(&format!("{p} not found"));
        return None;
    }
    Some(p)
}

fn gpu() -> gpu_core::Gpu {
    match std::env::var("BRAIN_CODEFORMER_DEVICE").as_deref() {
        Ok("cpu") => gpu_core::Gpu::new_cpu(&codeformer::KERNELS),
        _ => gpu_core::testgpu::dev(&codeformer::KERNELS),
    }
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


/// The w grid the dumper wrote, and the file-name tag it used.
const WS: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

fn wtag(w: f32) -> String {
    format!("{w:.2}").replace('.', "p")
}

fn imported(cfg: &CodeFormerConfig) -> Option<vae::blocks::Tensors> {
    let path = weights()?;
    let im = codeformer::import::load(&path, cfg).unwrap_or_else(|e| panic!("import {path}: {e}"));
    println!(
        "codeformer.pth: {} source tensors -> {} runtime tensors",
        im.source_tensors,
        im.tensors.len()
    );
    assert_eq!(im.source_tensors, 515, "the released checkpoint has 515 tensors");
    Some(im.tensors)
}

/// Serializes the 512² cases: a tapped build pins every activation (the shared
/// builder's pool is off) and this graph holds the encoder, the transformer,
/// four pinned encoder features AND the generator with four fuse blocks live at
/// once. `crates/vqgan`'s smaller tapped 512² model already measured 6.9 GB on a
/// P40; running these concurrently is how a suite fills a card.
fn heavy() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// 1. mapping units - the checkpoint contract, no device needed.
// ---------------------------------------------------------------------------

#[test]
fn import_covers_the_real_checkpoint_both_ways() {
    let cfg = CodeFormerConfig::codeformer();
    let Some(t) = imported(&cfg) else { return };
    assert_eq!(t.len(), cfg.runtime_manifest().len());
    // The VQGAN half must be byte-identical to what `crates/vqgan` imports from
    // the same file - one import, not two.
    let path = weights().expect("weights present");
    let vq = vqgan::import::load(&path, &cfg.vqgan).expect("vqgan import");
    assert_eq!(vq.skipped.len(), 515 - 329, "vqgan skips exactly the CodeFormer tensors");
    for (name, (_, data)) in &vq.tensors {
        assert_eq!(&t[name].1, data, "{name} differs between the two imports");
    }
}

// ---------------------------------------------------------------------------
// 2 + 3. the full ladder for one input: encoder taps -> transformer -> codes,
//        then the generator + CFT at every w on the dumped grid.
// ---------------------------------------------------------------------------

fn ladder(case: &str) {
    let Some(genc) = load(&format!("restore/codeformer/encoder_{case}.safetensors")) else {
        return;
    };
    let Some(gtf) = load(&format!("restore/codeformer/transformer_{case}.safetensors")) else {
        return;
    };
    let Some(t) = imported(&CodeFormerConfig::codeformer()) else { return };
    let _heavy = heavy();
    let cfg = CodeFormerConfig::codeformer();
    let m = CodeFormer::new(cfg.clone(), &t, gpu(), true);
    let tap = |name: &str| m.read_tap(name).unwrap_or_else(|| panic!("no tap {name}"));

    // ---- encoder + transformer -------------------------------------------
    let indices = m.predict_codes(&genc["input"].1);
    let mut rep = WorstTable::new(30);
    for tp in cfg.taps() {
        rep.add(
            &format!("enc.{:02} ({}²)", tp.enc_block, tp.size),
            &tap(&format!("enc.{}", tp.size)),
            &genc[&format!("enc.{:02}", tp.enc_block)].1,
        );
    }
    rep.add("lq_feat", &tap("lq_feat"), &genc["lq_feat"].1);
    rep.add("feat_emb", &tap("feat_emb"), &gtf["feat_emb"].1);
    // Inside layer 0 - the five stage boundaries of one TransformerSALayer.
    for leaf in ["norm1", "attn_out", "norm2", "linear1", "linear2"] {
        rep.add(
            &format!("ft.00.{leaf}"),
            &tap(&format!("ft.00.{leaf}")),
            &gtf[&format!("ft.00.{leaf}")].1,
        );
    }
    for l in 0..cfg.n_layers as usize {
        let k = format!("ft.{l:02}");
        rep.add(&k, &tap(&k), &gtf[&k].1);
    }
    rep.add("logits_norm", &tap("logits_norm"), &gtf["logits_norm"].1);
    rep.add("logits", &m.code_logits(), &gtf["logits"].1);

    // The predicted CODE INDICES are the discrete output; a cosine on the
    // logits can be 0.9999999 while an index flips, so they are gated exactly.
    let want_idx: Vec<u32> = gtf["indices"].1.iter().map(|&v| v as u32).collect();
    let flips = indices.iter().zip(&want_idx).filter(|(a, b)| a != b).count();
    println!(
        "{case}: {flips}/{} predicted code indices differ from the reference ({} distinct)",
        want_idx.len(),
        gtf["n_unique_codes"].1[0] as u32
    );
    assert_eq!(flips, 0, "{case}: the code-prediction transformer picked different codes");

    rep.finish(&format!("codeformer {case}: encoder + code-prediction transformer"), 0.9999, 3e-3);

    // ---- the generator + CFT at every w -----------------------------------
    let mut drift = Vec::new();
    let mut base = Vec::new();
    for w in WS {
        let Some(g) = load(&format!("restore/codeformer/gen_{case}_w{}.safetensors", wtag(w)))
        else {
            continue;
        };
        let image = m.generate(&indices, w);
        assert_eq!(g["w"].1[0], w, "golden w tag mismatch");

        let mut rep = WorstTable::new(30);
        rep.add("quant_feat", &tap("quant_feat"), &gtf["quant_feat"].1);
        for tp in cfg.taps() {
            let key = format!("gen.{:02}", tp.gen_block);
            rep.add(
                &format!("{key} (pre-fuse)"),
                &tap(&format!("generator.blocks.{}", tp.gen_block)),
                &g[&key].1,
            );
            // At w = 0 the reference SKIPS the fuse block, so it dumped no
            // internals; the port still evaluates it and scales by 0.
            if w > 0.0 {
                for leaf in ["encode_enc", "scale", "shift", "out"] {
                    let k = format!("fuse.{}.{leaf}", tp.size);
                    rep.add(&k, &tap(&k), &g[&k].1);
                }
            }
        }
        rep.add("gen.24", &tap("generator.blocks.24"), &g["gen.24"].1);
        rep.add("output", &image, &g["output"].1);
        rep.finish(&format!("codeformer {case}: generator + CFT at w = {w}"), 0.9999, 3e-3);

        if w == 0.0 {
            // The reference's w=0 branch skips the fuse entirely; the port
            // evaluates it and multiplies by zero. That must be EXACT, not
            // close: `0 * finite = 0` and `x + 0 = x`.
            for tp in cfg.taps() {
                let pre = tap(&format!("generator.blocks.{}", tp.gen_block));
                let post = tap(&format!("fuse.{}.out", tp.size));
                assert_eq!(
                    max_abs_diff(&pre, &post),
                    0.0,
                    "{case}: at w=0 the fuse at {}² is not the identity",
                    tp.size
                );
            }
            base = image.clone();
        }
        if !base.is_empty() {
            drift.push((w, max_abs_diff(&image, &base)));
        }
    }

    // ---- the dial's DIRECTION, measured on our own outputs -----------------
    // Getting `w` inverted is invisible in any single-w comparison and visible
    // to a human looking at faces. `w = 0` must be the no-fusion baseline and
    // the CFT contribution must grow monotonically with w.
    println!(
        "{case}: max|out(w) - out(0)| = {}",
        drift.iter().map(|(w, d)| format!("w={w:.2}->{d:.4}")).collect::<Vec<_>>().join(", ")
    );
    assert_eq!(drift.first().map(|(_, d)| *d), Some(0.0), "{case}: w=0 is not the baseline");
    for pair in drift.windows(2) {
        assert!(
            pair[1].1 >= pair[0].1,
            "{case}: the CFT contribution shrank from w={} to w={} - the dial is inverted or \
             the residual is not scaled by w",
            pair[0].0,
            pair[1].0
        );
    }
}

#[test]
fn ladder_face() {
    ladder("face");
}

#[test]
fn ladder_synth() {
    ladder("synth");
}

// ---------------------------------------------------------------------------
// 4. THE PRODUCTION PATH. Every test above builds with `taps = true`, which
//    pins every activation and DISABLES the shared builder's buffer pool.
//    Callers outside the tests pass `taps = false`, so the graph they actually
//    run - the one where activations are ALIASED - is never otherwise compared
//    to anything. A `free` issued one step early is invisible in the tapped
//    build and silently corrupts the pooled one.
//
//    Bit-equality is the right gate: pooling changes WHICH buffer a step
//    writes, never the arithmetic.
// ---------------------------------------------------------------------------

#[test]
fn pooled_matches_tapped() {
    let Some(genc) = load("restore/codeformer/encoder_face.safetensors") else { return };
    let Some(t) = imported(&CodeFormerConfig::codeformer()) else { return };
    let _heavy = heavy();
    let cfg = CodeFormerConfig::codeformer();
    let img = &genc["input"].1;

    let tapped = CodeFormer::new(cfg.clone(), &t, gpu(), true);

    // A freshly built model has never run submit A, so the four encoder
    // features and the logits that `generate`/`code_logits` read have never
    // been written. Returning a plausible image from them would be the exact
    // "silently wrong, not a crash" failure this repo pays for most, so both
    // entry points must refuse.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let codes = vec![0u32; cfg.latent_size as usize];
    let early_gen = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tapped.generate(&codes, 0.5)
    }));
    let early_logits =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tapped.code_logits()));
    std::panic::set_hook(hook);
    assert!(early_gen.is_err(), "generate() before predict_codes() returned an image anyway");
    assert!(early_logits.is_err(), "code_logits() before predict_codes() returned logits anyway");

    let a = tapped.restore(img, 0.5);
    drop(tapped);

    let pooled = CodeFormer::new(cfg, &t, gpu(), false);
    let b = pooled.restore(img, 0.5);

    assert_eq!(a.indices, b.indices, "pooled graph predicts different codes");
    assert_eq!(
        max_abs_diff(&a.image, &b.image),
        0.0,
        "pooled restoration differs from the tapped one (cosine {:.9}) - a buffer is freed \
         before its last read",
        cosine(&a.image, &b.image)
    );
    // The four encoder features live ACROSS the two submits; if the pool ever
    // handed one out as generator scratch, w>0 would silently drift.
    let again = pooled.generate(&b.indices, 0.5);
    assert_eq!(
        max_abs_diff(&again, &b.image),
        0.0,
        "a second generate() at the same w differs - an encoder feature was aliased"
    );
    println!("pooled (taps=false) graph is bit-identical to the tapped one");
}
