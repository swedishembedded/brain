// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **A real image through the real weights** - arbitrary decoded RGB ->
//! `deepseek2ocr::preprocess` -> the real 400 M-parameter DeepEncoder (and, when
//! the decoder's expansion is on disk, on to logits).
//!
//! Every other real-weight test in this crate feeds a **spatially constant**
//! `0.5` fill, and deliberately so: that is the buffer llama.cpp's `-p encode`
//! debug path copies straight into the graph, which is what makes those tests
//! comparable to the reference capture. But a constant image is blind to
//! everything preprocessing can get wrong - a transposed resize, a flipped
//! axis, a channel swap and a missing normalization are all invisible when every
//! pixel is the same number. This test is the other side: a non-uniform,
//! document-shaped source at an extent that is neither square nor 1024, taken
//! through the real preprocessing path.
//!
//! ## What is claimed
//!
//! **No byte-level oracle exists and none is invented.** The reference capture
//! that gates `tests/real_weight.rs` was taken on the constant fill, and
//! llama.cpp's debug callback segfaults inside this model's CLIP graph anyway,
//! so nothing downstream of the SAM compressor was ever captured for any input.
//! What this test asserts is what it can justify:
//!
//!  * the preprocessor's output really is `[3, 1024, 1024]`, finite, and in the
//!    checkpoint's `[-1, 1]` normalized range rather than raw `[0, 1]`;
//!  * the letterbox bars are **exactly** the normalization's zero, and the
//!    content band is not - so an aspect-preserving fit really happened;
//!  * the whole encoder runs on it and every one of its 327 680 output floats is
//!    finite, with a reported (not gated) distribution;
//!  * **the encoder's output actually depends on the picture.** The same tower,
//!    same weights, fed the constant gray page that the parity tests use,
//!    produces a materially different embedding. A pipeline that silently
//!    dropped its input - a zeroed upload, a mis-sized buffer, a resize that
//!    collapsed - would still be finite and still be the right shape, and this
//!    is the assertion that would catch it.
//!
//! ## Two stages, so the cheap half still runs on a small box
//!
//! Stage 1 builds the **encoder alone** (~1.6 GiB of weights): everything the
//! preprocessor feeds. Stage 2 rebuilds as the full composite and runs to
//! logits, but only when `crates/deepseekv2`'s fp32 expansion is already on
//! disk; the encoder is dropped first, so the peak is one model's worth, not
//! two. Backend: CPU, per `tests/common/real_vision.rs`.

use checkpoint::gguf::MmapGguf;
use checkpoint::weightio::WeightReader;
use deepseek2ocr::config::DeepseekOcrConfig;
use deepseek2ocr::model::DeepseekOcr;
use deepseek2ocr::preprocess::{self, Fit};
use deepseek2ocr::DeepEncoder;

use brain_testutil::mem;
use brain_testutil::parity::compare;

/// The model-store lookup, the CPU pin, the mmproj import, the synthetic page,
/// the `MemAvailable` guard and `describe` - shared with the other three
/// real-weight binaries in this crate.
#[path = "common/real_vision.rs"]
mod real_vision;

use real_vision::{mem_available_gib, synthetic_page, DECODER_GIB, SRC_H, SRC_W};

#[ignore = "a real image through the real 400 M-parameter DeepEncoder at 1024x1024 (and on to the 2.9 B decoder when its expansion is on disk). Slow lane only. `make test/slow`, or `cargo test --release -p brain-deepseekocr --test real_weight_image -- --nocapture`."]
#[test]
fn real_weight_real_image_forward() {
    let Some(mmproj) = real_vision::mmproj_path() else { return };
    real_vision::pin_cpu_backend();
    mem("start");

    let cfg = DeepseekOcrConfig::deepseek_ocr(1);
    cfg.check_real_scale_shaped();
    let side = cfg.sam.image_h();
    assert_eq!((side, cfg.sam.image_w()), (1024, 1024), "the square is sam grid_h * patch_size");

    // ---- preprocessing ---------------------------------------------------
    let gpu = gpu_core::testgpu::dev(preprocess::PIPELINES);
    let page = synthetic_page(SRC_W, SRC_H);
    let image = preprocess::preprocess_image(&gpu, &cfg, &page, SRC_W, SRC_H, Fit::Pad);
    println!("== deepseek-ocr real-weight real-image forward ({SRC_W}x{SRC_H} -> {side}x{side})");
    assert_eq!(image.len(), 3 * (side * side) as usize);
    assert_eq!(real_vision::describe("preprocessed", &image), image.len(), "preprocess produced a non-finite pixel");
    let (lo, hi) = image.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
    assert!(lo < -0.5 && hi > 0.5, "range [{lo}, {hi}] is not a [-1,1] normalization");
    // Exactly, not approximately: the clamp inside `preprocess` is what makes
    // the range match the reference's 8-bit-saturated intermediate, and the
    // cubic filter really does ring past it on a page full of hard text edges.
    assert!(lo >= -1.0 && hi <= 1.0, "range [{lo}, {hi}] escaped [-1,1] - did the clamp run?");

    // The letterbox really is a letterbox: the bars are the normalization's
    // exact zero and the content band is not.
    let (_, fit_h, border) = preprocess::placement(Fit::Pad, SRC_W, SRC_H, side, side);
    assert!(border.top > 0 && border.bottom > 0, "a {SRC_W}x{SRC_H} source must letterbox top/bottom");
    println!("  letterbox: content {side}x{fit_h}, border top {} bottom {}", border.top, border.bottom);
    let px = |c: u32, x: u32, y: u32| image[(c * side * side + y * side + x) as usize];
    for c in 0..3 {
        for y in [0, border.top - 1, side - border.bottom, side - 1] {
            assert_eq!(px(c, side / 2, y), 0.0, "row {y} of channel {c} should be the mean-grey border");
        }
        assert!(px(c, side / 2, side / 2).abs() > 1e-3, "the middle of the page looks like border");
    }

    // ---- stage 1: the encoder -------------------------------------------
    let mg = MmapGguf::open(mmproj.to_str().expect("utf-8 path")).expect("open mmproj");
    let vision = real_vision::encoder_weights(&mg);
    drop(mg);
    mem("mmproj imported");

    let dev = |k: &'static [(&'static str, &'static str)]| gpu_core::testgpu::dev(k);
    let enc = DeepEncoder::new(&dev, cfg.clone(), &vision, 0, false);
    mem("encoder built (inference)");
    let embeds = enc.forward(&image);
    mem("encoder forward done");
    assert_eq!(embeds.len(), (cfg.image_tokens() * cfg.projector_out()) as usize);
    assert_eq!(real_vision::describe("projector_out", &embeds), embeds.len(), "the encoder produced a non-finite value");
    assert_eq!(real_vision::describe("clip_spatial", &enc.read_clip_spatial()), (cfg.image_tokens() * cfg.clip_width()) as usize);

    // The one real gate available without an oracle: the embedding must depend
    // on the picture. The constant-gray page is the input every other
    // real-weight test in this crate uses, so this compares against a known,
    // already-characterised point rather than against an arbitrary second image.
    let gray = vec![0.0f32; image.len()]; // 0.5 gray, normalized: exactly zero
    let flat = enc.forward(&gray);
    let (cos, max) = compare(&embeds, &flat);
    println!("  page vs constant-gray embedding: cos {cos:.6} max_abs {max:.3e}");
    assert!(cos < 0.99, "the encoder's output barely moved with the image (cos {cos}) - is the input reaching it?");
    assert!(max > 1e-2, "the encoder's output barely moved with the image (max_abs {max})");
    drop(enc);
    mem("encoder dropped");

    // ---- stage 2: on to the decoder, when its expansion exists ------------
    let Some(dir) = real_vision::store_dir() else { return };
    let expanded = dir.join(real_vision::EXPANDED);
    if !expanded.exists() {
        println!("  (decoder stage skipped: {} absent - run crates/deepseekv2's parity test to build it)", expanded.display());
        return;
    }
    let avail = mem_available_gib();
    if avail < DECODER_GIB {
        println!("  (decoder stage skipped: MemAvailable {avail:.1} GiB < {DECODER_GIB} GiB - the composite peaks near 24 GiB)");
        return;
    }
    println!("  decoder stage: MemAvailable {avail:.1} GiB");
    let seq = cfg.image_tokens() + 2;
    let (row0, n_rows) = (1u32, cfg.image_tokens());
    let decoder = WeightReader::open(expanded.to_str().expect("utf-8 path")).expect("open expansion");
    let m = DeepseekOcr::new_split(&dev, cfg.clone(), &vision, &decoder, 0, seq, row0, false);
    drop(decoder);
    drop(vision);
    assert_eq!(m.image_run(), (row0, n_rows));
    mem("composite built (inference)");

    m.set_tokens_unsupervised(&vec![0u32; seq as usize]);
    let loss = m.forward(&image);
    assert_eq!(loss, 0.0, "every target is IGNORE, so a forward-only run reports no loss");
    mem("composite forward done");

    // The splice placed this image's own embedding, not a stale or zero one.
    let res0 = m.read_decoder_input();
    let dm = cfg.decoder.d_model() as usize;
    let spliced = &res0[row0 as usize * dm..(row0 + n_rows) as usize * dm];
    let (scos, smax) = compare(spliced, &m.encoder().read_projector_out());
    println!("  spliced rows vs projector output: cos {scos:.10} max_abs {smax:.3e}");
    assert_eq!(smax, 0.0, "the splice did not place the projector output verbatim");

    let logits = m.read_logits();
    assert_eq!(real_vision::describe("logits", &logits), logits.len(), "logits have non-finite values");
    let vocab = cfg.decoder.vocab() as usize;
    let last = &logits[logits.len() - vocab..];
    let mut top: Vec<usize> = (0..vocab).collect();
    top.sort_by(|&a, &b| last[b].total_cmp(&last[a]));
    println!("  top-5 next-token ids: {:?} (logits {:?})", &top[..5], top[..5].iter().map(|&i| last[i]).collect::<Vec<f32>>());
    let (lo, hi) = last.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &x| (l.min(x), h.max(x)));
    assert!(hi - lo > 1.0, "the final-position logits are nearly uniform (spread {})", hi - lo);
    assert!(hi < 1e3 && lo > -1e3, "final-position logit range [{lo}, {hi}] is not a plausible distribution");
    mem("end");
}
