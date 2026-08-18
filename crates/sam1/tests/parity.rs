// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Real-weight stage parity for the SAM-1 tower**, against llama.cpp.
//!
//! `tests/gradcheck.rs` proves this crate's backward agrees with its own
//! forward at toy dims. It says nothing about whether the forward matches the
//! reference implementation, or whether the import puts the real checkpoint's
//! tensors where the graph expects them. This test is that rung: the real
//! `mmproj-DeepSeek-OCR-Q8_0.gguf` weights, the real 1024x1024 production shape
//! (patch 16 -> a 64x64 grid, embed 768, 12 heads, window 14, global attention
//! at blocks 2/5/8/11), compared tap for tap against a dump taken from
//! **llama.cpp** - the upstream consumer the GGUF format targets - running the
//! same file.
//!
//! ## Why the floor is where it is
//!
//! Both sides read the *same* Q8_0 blocks, but they do not do the same
//! arithmetic with them: brain dequantizes to fp32 and runs an fp32 GEMM, while
//! ggml keeps the weights quantized and **quantizes the activations to 8 bits
//! per 32-element block** for the dot product. That is a real, one-sided
//! error source of order 1e-2 relative per matmul, and it accumulates through
//! 12 residual blocks. So this is not a "two fp32 implementations" comparison
//! and a 0.9999 floor would be wrong on the physics, not merely strict. The
//! floor below is set from the measured run, and the per-tap numbers are
//! printed so a REGRESSION (a sudden drop at one block) is visible even where
//! the absolute number is not 1.0.
//!
//! Measured (CPU backend, real Q8_0 mmproj): every one of the 14 taps clears
//! **0.9994**, worst `sam_blk07_out` at cosine 0.9994355 / max_abs 1.655e-1,
//! the compressor output `sam_output` at 0.9999586 / 5.232e-3. That the error
//! does not grow monotonically with depth (block 0 is 0.9999663, block 11 is
//! 0.9999253) is itself evidence it is quantization noise being re-normalized
//! by each block's LayerNorms rather than a systematic divergence.
//!
//! The claim that the residual gap is ggml's activation quantization and not
//! brain's is not an assumption: an independent longhand NumPy fp64
//! reimplementation of block 0 - built from the dequantized mmproj tensors and
//! sharing no code with either side - lands at cosine 0.9999663 against the
//! same reference tap, i.e. the **same** number brain gets to seven digits.
//! Two independent fp32/fp64 implementations agreeing with each other more
//! closely than either agrees with the reference is what a reference-side
//! numeric difference looks like. That experiment also settled the MLP
//! activation on the real weights: exact-erf GELU scores 0.99997 where
//! quick-GELU scores 0.9797, so `gelu_erf` (what this crate binds) is right -
//! worth pinning because llama.cpp derives BOTH towers' activation from the
//! single `use_gelu` hparam, which for this file is false and therefore
//! selects `FFN_GELU_QUICK` in its own `hparams`.
//!
//! ## Backend: this test pins the CPU backend, and that is a real defect
//!
//! On the wgpu backend at this production shape the tower's per-block output
//! buffers are corrupted as soon as the graph has **three or more** blocks:
//! `block_out(0)` changes by up to 2.1e2 purely by adding a third block, and
//! the patch-embed tap stops being spatially constant for a spatially constant
//! image (max deviation 6.66, identical for every block count from 3 to 12).
//! The same run on `backend-cpu` is bit-identical between a 2-block and a
//! 3-block build and holds the constancy exactly. It is not weight-dependent
//! (a dense-init build reproduces it), not the import (every parameter
//! round-trips through the ParamStore at max abs error 0.0), and not global
//! attention as such (a 2-block config with one global block is clean). It is
//! a device/allocation-level failure that only appears past a few blocks'
//! worth of ~500 MB-per-block buffers at 1024x1024, and it is out of scope for
//! a parity change to fix - but it does mean the SAM tower cannot presently be
//! trusted on that backend at production shape. `crates/fastvlm/src/parity.rs`
//! pins the CPU backend the same way and for the same class of reason.
//!
//! ## The input, and why a constant one still tests something
//!
//! The reference capture ran `llama-mtmd-debug -p encode --image gray -n 1024`,
//! whose `-p encode` branch (verified in `tools/mtmd/debug/mtmd-debug.cpp` and
//! `mtmd_debug_encode_image` in `tools/mtmd/mtmd.cpp`) builds a constant
//! `0.5f`-filled `[1024][1024*3]` buffer and hands it **straight to
//! `clip_image_encode`** - no `clip_image_preprocess`, no mean/std
//! normalization. So the network's pixel input is exactly 0.5 everywhere and
//! brain must feed the same constant with no normalization step of its own.
//! (This is deliberately NOT DeepSeek-OCR's real preprocessing, which would map
//! a gray image to 0.0 through mean=std=0.5.)
//!
//! A spatially constant input does not make the test trivial: SAM's learned
//! absolute position embedding and its decomposed relative-position bias both
//! break translation invariance, so the tower's output varies strongly across
//! the grid (the golden's own generator asserts that). What it does buy is that
//! the input's HWC-vs-CHW layout cannot be gotten wrong - a uniform tensor is
//! invariant to axis order - so any mismatch found here is in the weights, the
//! import or the graph, never in the test's input construction.
//!
//! ## Fixture
//!
//! `<testdata>/deepseek-ocr/real/vision.safetensors`, produced by
//! `tools/goldens/deepseek_ocr_convert_llamacpp_dump.py` from a patched
//! llama.cpp graph-eval dump; that script's header documents the capture
//! commands and the byte-layout checks behind each tensor's shape. The test
//! SKIPS ITSELF when either the fixture or the real mmproj checkpoint is
//! absent - it never panics for a missing input.

use std::collections::HashMap;

use brain_testutil::parity::{load, Report};
use brain_testutil::testdata_path as testdata;
use checkpoint::gguf::MmapGguf;
use sam1::{SamEncoder, SamViTConfig};

/// The real checkpoint, in the model store rather than under `testdata/`
/// (`testdata/` is fixtures; weights belong to the store - see
/// `brain_testutil::model_dir`).
const MMPROJ: &str = "ggml-org/DeepSeek-OCR-GGUF";

/// See the module header: ggml's 8-bit activation quantization, not fp32
/// round-off, sets this scale. The measured worst tap is 0.99944, so this is a
/// real gate (a wrong window boundary or a transposed weight lands far below
/// it) rather than a rubber stamp.
const FLOOR: f64 = 0.999;

/// Pin the CPU backend - see the module header's "Backend" section. Called
/// before any device is built.
///
/// # Safety
/// Single-threaded at this point in the test binary; no other test in this
/// crate reads or writes `BRAIN_DEVICE`.
fn pin_cpu_backend() {
    unsafe { std::env::set_var("BRAIN_DEVICE", "cpu") };
}

fn mmproj_path() -> Option<std::path::PathBuf> {
    let dir = brain_testutil::model_dir(MMPROJ)?;
    let p = std::path::Path::new(&dir).join("mmproj-DeepSeek-OCR-Q8_0.gguf");
    p.exists().then_some(p)
}

#[test]
fn real_mmproj_sam_tower_matches_llamacpp() {
    let fixture = testdata("deepseek-ocr/real/vision.safetensors");
    if !fixture.exists() {
        brain_testutil::skip(&format!("sam1 parity: fixture missing at {}", fixture.display()));
        return;
    }
    let Some(mmproj) = mmproj_path() else {
        brain_testutil::skip(&format!("sam1 parity: {MMPROJ} mmproj not in the model store"));
        return;
    };
    pin_cpu_backend();
    let golden = load(&fixture);

    // ---- import (two-way coverage over the whole 476-tensor mmproj) ----
    let mg = MmapGguf::open(mmproj.to_str().expect("utf-8 path")).expect("open mmproj");
    let (cfg, stats) = sam1::import::dry_run(&mg).expect("dry run");
    println!("== sam1 real-weight parity\n  import: {stats}");
    assert_eq!(cfg, SamViTConfig::deepseek_ocr(), "the shipped file disagrees with the preset");
    cfg.check_bindable();
    let (_, weights) = sam1::import::weights_from_gguf(&mg).expect("import weights");
    drop(mg);

    // ---- the constant input, exactly as the reference capture built it ----
    let fill = golden["input_fill"].data[0];
    assert_eq!(fill, 0.5, "the fixture's recorded input fill changed");
    let px = vec![fill; 3 * cfg.image_h() as usize * cfg.image_w() as usize];

    let gpu = gpu_core::testgpu::dev(sam1::PIPELINES);
    // Forward parity: an INFERENCE build. Every parameter is `Role::Frozen` and
    // no backward scratch is allocated, which at this 1024x1024 shape is the
    // difference between ~5.8 GiB and ~2.4 GiB of peak RSS. The forward graph
    // is bit-identical either way -- this file's numbers did not move when the
    // build switched.
    let enc = SamEncoder::new_inference(gpu, cfg.clone(), &weights, 0);
    enc.write_image(&px);
    let obj = enc.forward();
    assert!(obj.is_finite(), "forward produced a non-finite objective {obj}");

    // ---- every block, then the neck + compressor ----
    //
    // `block_out` is `[rows, C]` NLC in row-major grid order, which is exactly
    // the golden's `(H, W, C)`; `output` is `[1, c_out, gh/4, gw/4]` NCHW, which
    // is exactly the golden's `(C, H, W)`. Both layout claims were verified from
    // the reference bytes by the converter, not assumed here.
    let rows = cfg.rows() as usize * cfg.d_model as usize;
    let mut r = Report::new(FLOOR);
    for l in 0..cfg.n_layers as usize {
        let name = format!("sam_blk{l:02}_out");
        r.check(&name, &enc.gpu.read(enc.block_out(l), rows), &golden[&name].data);
    }
    let nchw = enc.gpu.read(enc.output(), enc.out_len());
    r.check("sam_output", &nchw, &golden["sam_output"].data);

    // The compressor output IS CLIP's spatial input, flattened NCHW -> NLC and
    // pushed one row down past the class token. Checking it here rather than in
    // `crates/clip` keeps the handoff pinned on the side that produces it; the
    // golden's own generator proved the two halves bit-identical in the
    // reference, so a mismatch is ours.
    let (gh, gw) = cfg.compress_grid();
    let (c, n) = (cfg.compress_out as usize, (gh * gw) as usize);
    let mut nlc = Vec::with_capacity(n * c);
    for p in 0..n {
        nlc.extend((0..c).map(|k| nchw[k * n + p]));
    }
    r.check("clip_input_spatial", &nlc, &golden["clip_input_tokens"].data[c..]);
    r.finish("sam1 real-weight");
}

/// The class-token row of CLIP's assembled input is `v.class_embd` verbatim -
/// the reference applies no transform to it before the concat. That makes it a
/// free, exact check that the mmproj's CLIP branch imports the tensor the
/// reference actually used, on a run that needs no GPU at all.
#[test]
fn clip_class_token_row_is_the_imported_class_embd() {
    let fixture = testdata("deepseek-ocr/real/vision.safetensors");
    let (Some(mmproj), true) = (mmproj_path(), fixture.exists()) else {
        brain_testutil::skip("real mmproj or vision fixture missing");
        return;
    };
    let golden = load(&fixture);
    let mg = MmapGguf::open(mmproj.to_str().expect("utf-8 path")).expect("open mmproj");
    let full = gguf::deepseek_ocr_vision::config_from_gguf(&mg).expect("config");
    // One-entry manifest: everything else is a recorded drop, so the driver's
    // two-way check still runs and this stays a single dequantized tensor.
    const CLS: &str = "vision.clip.class_embed";
    let want_one = vec![(CLS.to_string(), full.clip.d_model as usize)];
    let w: HashMap<String, Vec<f32>> = gguf::import::to_map(
        &mg,
        &want_one,
        &|n| match gguf::deepseek_ocr_vision::classify(n, &full)? {
            gguf::Mapped::Simple(b) if b == CLS => Ok(gguf::Mapped::Simple(b)),
            _ => Ok(gguf::Mapped::Dropped("not the class token")),
        },
        "class-embd probe",
    )
    .unwrap_or_else(|e| panic!("class_embed probe: {e}"));
    let got = &w[CLS];
    let want = &golden["clip_input_cls"].data;
    assert_eq!(got.len(), want.len());
    let (cos, max_abs) = brain_testutil::parity::compare(got, want);
    println!("  clip_input_cls   cos {cos:.10}  max_abs {max_abs:.3e}  n={}", want.len());
    assert_eq!(max_abs, 0.0, "the class token must be the imported tensor verbatim");
}
