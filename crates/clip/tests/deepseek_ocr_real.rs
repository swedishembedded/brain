// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **`ClipVision` against the REAL DeepSeek-OCR mmproj** - import coverage, the
//! config the shipped header actually implies, and a sanity forward on the real
//! patch tokens.
//!
//! ## What this can and cannot claim
//!
//! There is **no reference tap inside the CLIP tower**. llama.cpp's graph-eval
//! debug callback segfaults immediately after the cls-token concat that builds
//! this tower's input (a pre-existing defect in its debug tooling for this
//! model - its production path runs the same graph fine), so the golden capture
//! stops exactly one node before `build_vit` begins. Consequently:
//!
//!  * the import's two-way coverage, the derived config, and the class-token
//!    row are **verified against the real file**;
//!  * the tower's OUTPUT is only checked for being finite and in a sane range.
//!    No cosine floor is invented for it, because there is nothing to compare
//!    it against. This test reports what it observes and asserts only what it
//!    can actually justify.
//!
//! ## What the real header changed
//!
//! Reading the shipped `mmproj-DeepSeek-OCR-Q8_0.gguf`'s KV settled two things
//! that had been carried as beliefs, and both were wrong in
//! `ClipVisionConfig::deepseek_ocr()`:
//!
//!  * `clip.use_gelu = **true**`, not false. The preset said quick-GELU. See
//!    that constructor's doc for the three-way ambiguity this leaves open
//!    (original CLIP-L uses quick-GELU; llama.cpp reads `true` as its tanh
//!    `FFN_GELU`; brain follows the file's flag into the exact-erf form).
//!  * `clip.vision.attention.layer_norm_epsilon = 1e-6`, which is the SAM
//!    tower's number and **is not what the reference runs for CLIP** - it
//!    overwrites `hparams.eps` with 1e-5 in its DeepSeek-OCR branch. So the key
//!    is deliberately not adopted; see `ClipVisionConfig::from_gguf`.
//!
//! Skips itself when the checkpoint or the fixture is absent.

use brain_testutil::parity::compare;
use brain_testutil::testdata_path as testdata;
use checkpoint::gguf::MmapGguf;
use clip::config::ClipVisionConfig;
use clip::model::{ClipVision, PatchSource};

const STORE: &str = "ggml-org/DeepSeek-OCR-GGUF";

fn mmproj() -> Option<std::path::PathBuf> {
    let dir = brain_testutil::model_dir(STORE)?;
    let p = std::path::Path::new(&dir).join("mmproj-DeepSeek-OCR-Q8_0.gguf");
    p.exists().then_some(p)
}

/// See `crates/sam1/tests/parity.rs`'s "Backend" section: the wgpu path
/// corrupts large multi-block graphs on this class of device. A 24-block CLIP-L
/// is squarely in that regime, and an unreliable "is it finite" answer is worse
/// than no answer.
///
/// # Safety
/// Single-threaded at this point; no other test in this binary touches
/// `BRAIN_DEVICE`.
fn pin_cpu_backend() {
    unsafe { std::env::set_var("BRAIN_DEVICE", "cpu") };
}

#[test]
fn real_mmproj_clip_import_covers_the_manifest_both_ways() {
    let Some(path) = mmproj() else {
        return brain_testutil::skip(&format!("{STORE} mmproj not in the model store"));
    };
    let mg = MmapGguf::open(path.to_str().expect("utf-8 path")).expect("open mmproj");
    let (cfg, stats) = clip::import::gguf_mmproj::dry_run(&mg).expect("clip dry run");
    println!("clip mmproj coverage: {stats}");

    assert_eq!(cfg, ClipVisionConfig::deepseek_ocr(), "the shipped file disagrees with the documented preset");
    assert_eq!(stats.source_tensors, 476, "the shipped mmproj tensor count changed");
    assert_eq!(stats.written, cfg.tensor_manifest().len(), "{stats}");
    assert_eq!(cfg.layers(), 24);
    assert_eq!(cfg.n_positions(), 1 + cfg.native_patches());
    assert!(
        stats.dropped.values().sum::<usize>() > 0,
        "the SAM half and the projector must be dropped ON THE RECORD, not silently missing: {stats}"
    );

    // The `pre_ln` / `post_ln` question, settled against the shipped file
    // rather than against a fixture. The manifest declares ONE LayerNorm on the
    // stem (`pre_norm`) and no post-LayerNorm; the two-way check above already
    // proves every one of the 476 source names is accounted for, and this makes
    // the absence explicit so a future reader does not have to re-derive it.
    //
    // Scoping matters here and the first version of this check got it wrong:
    // llama.cpp names the **SAM** blocks' second LayerNorm `post_ln`
    // (`v.sam.blk.N.post_ln.*`, 24 tensors, brain's `blocks.N.norm2.*`), so a
    // whole-file search for "post_ln" finds a post-norm that has nothing to do
    // with the CLIP tower. The CLIP tower is the `v.*` names that are NOT
    // `v.sam.*`.
    let names: Vec<&str> = mg.names().iter().map(|s| s.as_str()).collect();
    let clip_names = || names.iter().filter(|n| n.starts_with("v.") && !n.starts_with("v.sam."));
    assert!(names.iter().any(|n| n.starts_with("v.pre_ln")), "no v.pre_ln in the shipped mmproj");
    let post: Vec<&&str> = clip_names().filter(|n| n.contains("post_ln") || n.contains("post_norm")).collect();
    assert!(post.is_empty(), "the CLIP tower carries a post-LayerNorm after all: {post:?}");
    // ...and the block bodies really are plain pre-LN pairs: exactly two norms
    // per block, no third.
    let block_norms = clip_names().filter(|n| n.contains(".ln") && n.ends_with(".weight")).count();
    assert_eq!(block_norms, cfg.layers() as usize * 2, "CLIP block LayerNorm count is not 2 per block");
    let manifest: Vec<String> = cfg.tensor_manifest().into_iter().map(|(n, _)| n).collect();
    assert!(manifest.iter().any(|n| n == "pre_norm.weight") && manifest.iter().any(|n| n == "pre_norm.bias"));
    assert!(!manifest.iter().any(|n| n.starts_with("post_")), "the manifest declares a post-norm the file lacks");
}

/// The real load (not just the header), plus a forward on the reference's own
/// patch tokens.
#[test]
fn real_mmproj_clip_forward_on_the_reference_tokens_is_sane() {
    let fixture = testdata("deepseek-ocr/real/vision.safetensors");
    let (Some(path), true) = (mmproj(), fixture.exists()) else {
        return brain_testutil::skip(&format!(
            "{STORE} mmproj or {} is missing",
            fixture.display()
        ));
    };
    pin_cpu_backend();
    let golden = brain_testutil::parity::load(&fixture);

    let mg = MmapGguf::open(path.to_str().expect("utf-8 path")).expect("open mmproj");
    let (cfg, weights) = clip::import::gguf_mmproj::weights_from_gguf(&mg).expect("clip weights");
    drop(mg);
    let d = cfg.d_model() as usize;

    // The class token: the reference's assembled input row 0 is `class_embed`
    // verbatim - no norm, no position, nothing. An exact check on real weights.
    let (cos, max_abs) = compare(&weights["class_embed"], &golden["clip_input_cls"].data);
    println!("class_embed vs reference row 0: cos {cos:.10} max_abs {max_abs:.3e}");
    assert_eq!(max_abs, 0.0, "the reference's cls row is not this checkpoint's class_embed");

    // The spatial rows: the reference's own CLIP input, i.e. the compressor
    // output flattened NLC. Feeding these instead of re-running SAM makes this
    // a test of the CLIP tower alone.
    let tokens = &golden["clip_input_tokens"].data[d..];
    let n = tokens.len() / d;
    let side = (n as f64).sqrt() as u32;
    assert_eq!((side * side) as usize, n, "the reference token grid is not square");
    println!("feeding {n} = {side}x{side} reference patch tokens of width {d}");

    let gpu = gpu_core::testgpu::dev(clip::model::CLIP_VISION_PIPELINES);
    let m = ClipVision::new_on(gpu, cfg.clone(), 1, PatchSource::Tokens { grid: (side, side) }, &weights);
    m.set_tokens(tokens);
    m.forward();

    // The position table at this grid: the checkpoint's native grid is 16 and so
    // is the compressor's, so the bicubic resample is the IDENTITY here and the
    // table must come back unchanged. (That the resample is exercised at all is
    // the tiny fixture's job - at real scale a bug in it is invisible, which is
    // exactly why this assertion is worth making explicitly.)
    assert_eq!(side, cfg.native_grid(), "real scale should need no position resample");
    let pos_full = m.read_pos_full();
    let pos_ck = &weights["pos_embed"];
    assert_eq!(pos_full.len(), pos_ck.len());
    let (pcos, pmax) = compare(&pos_full, pos_ck);
    println!("pos_embed at this grid: cos {pcos:.10} max_abs {pmax:.3e} (identity resample expected)");
    assert_eq!(pmax, 0.0, "the identity resample changed the position table");

    // No ground truth for the output - report, do not invent a threshold.
    let stats = |name: &str, v: &[f32]| {
        let finite = v.iter().filter(|x| x.is_finite()).count();
        let (mut lo, mut hi, mut sum, mut sq) = (f32::INFINITY, f32::NEG_INFINITY, 0f64, 0f64);
        for x in v.iter().filter(|x| x.is_finite()) {
            lo = lo.min(*x);
            hi = hi.max(*x);
            sum += *x as f64;
            sq += (*x as f64) * (*x as f64);
        }
        let nf = finite as f64;
        let mean = sum / nf;
        println!(
            "  {name:<16} n={} finite={finite} min={lo:.4} max={hi:.4} mean={mean:.5} rms={:.5}",
            v.len(),
            (sq / nf).sqrt()
        );
        assert_eq!(finite, v.len(), "{name}: {} non-finite values", v.len() - finite);
        (lo, hi)
    };

    println!("CLIP tower on real weights (no reference tap exists past this point):");
    stats("x0", &m.read_x0());
    for l in [0usize, 11, 23] {
        stats(&format!("block{l:02}_out"), &m.read_block_out(l));
    }
    let (lo, hi) = stats("output", &m.read_output());

    // The only defensible bound: a 24-block pre-LN ViT that has not diverged
    // stays within a couple of orders of magnitude of its input scale. This
    // catches a catastrophically wrong import (a transposed qkv blows up); it
    // does NOT certify correctness and is not presented as if it did.
    assert!(lo > -1e4 && hi < 1e4, "tower output range [{lo}, {hi}] is not a plausible hidden state");
}
