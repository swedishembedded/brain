// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The real 273-row image block, on the real weights.**
//!
//! Every other real-weight test in this crate runs the *contiguous* splice -
//! 256 projector rows, no newline and no view separator - because that is the
//! checkpoint-free golden fixture's own scope. This one runs what the reference
//! model actually feeds its decoder: `prompt::build_prompt`'s 273-row block,
//! filled by `deepseekocr::layout::RowGather` from the projector's output plus
//! the mmproj's two learned vectors.
//!
//! ## What is claimed, and why it can be claimed EXACTLY
//!
//! There is still no byte-level oracle for this model's forward (llama.cpp's
//! debug callback segfaults inside its CLIP graph - see `tests/real_weight.rs`),
//! so nothing here compares a computed tensor against a captured one. But the
//! thing this phase adds is **not** a computed tensor: 17 of the 273 rows are a
//! straight COPY of two tensors read off the checkpoint. So they are asserted
//! at exact equality, not at a cosine:
//!
//!  * the 16 newline rows are bit-identically `vision.image_newline`;
//!  * the one separator row is bit-identically `vision.view_separator`;
//!  * the other 256 rows are bit-identically the projector's own rows, each at
//!    the layout position `rows::row_plan` puts it;
//!  * and after the splice, the decoder's residual stream carries that same
//!    block verbatim at `[row0, row0 + 273)`.
//!
//! A wrong layout - a swapped newline/separator, an off-by-one in the run, a
//! block written in projector order and then padded - fails one of those four
//! with an exact mismatch, not with a small number.
//!
//! ## Two stages, so the cheap half still runs on a small box
//!
//! Stage 1 is the **encoder alone** (~1.6 GiB of weights) plus the row gather:
//! all four exact claims above except the splice. Stage 2 rebuilds as the full
//! composite through `DeepseekOcr::new_with_prompt`, runs to logits, and checks
//! the splice - but only when `crates/deepseekv2`'s fp32 expansion is on disk
//! and the box has the headroom (`real_vision::DECODER_GIB`). Backend: CPU, per
//! `tests/common/real_vision.rs`.

use checkpoint::gguf::MmapGguf;
use checkpoint::weightio::WeightReader;
use deepseekocr::config::{DeepseekOcrConfig, IMAGE_NEWLINE, VIEW_SEPARATOR};
use deepseekocr::model::DeepseekOcr;
use deepseekocr::preprocess::{self, Fit};
use deepseekocr::prompt::{build_prompt, tokenizer_from_gguf};
use deepseekocr::rows::Src;
use deepseekocr::DeepEncoder;

use brain_testutil::mem;

/// The model-store lookup, the CPU pin, the mmproj import, the synthetic page
/// and `describe` - shared with the other three real-weight binaries.
#[path = "common/real_vision.rs"]
mod real_vision;

use real_vision::{mem_available_gib, synthetic_page, DECODER_GIB, SRC_H, SRC_W};

/// Assert every row of `block` is exactly the vector its layout entry names.
/// Returns `(projector, newline, separator)` row counts actually checked, so a
/// caller can assert the layout was the one it meant.
fn assert_rows_are_their_sources(block: &[f32], rows: &[Src], proj: &[f32], newline: &[f32], separator: &[f32], dm: usize) -> (u32, u32, u32) {
    assert_eq!(block.len(), rows.len() * dm, "block is not [rows, d_model]");
    let (mut np, mut nn, mut ns) = (0u32, 0u32, 0u32);
    for (r, src) in rows.iter().enumerate() {
        let got = &block[r * dm..(r + 1) * dm];
        let want: &[f32] = match *src {
            Src::Projector(i) => {
                np += 1;
                &proj[i as usize * dm..(i as usize + 1) * dm]
            }
            Src::Newline => {
                nn += 1;
                newline
            }
            Src::Separator => {
                ns += 1;
                separator
            }
        };
        assert_eq!(got, want, "block row {r} ({src:?}) is not its source row");
    }
    (np, nn, ns)
}

#[ignore = "the real 273-row image block: a 400 M-parameter DeepEncoder at 1024x1024 (and on to the 2.9 B decoder when its expansion is on disk). Slow lane only. `make test/slow`, or `cargo test --release -p brain-deepseekocr --test real_weight_layout -- --nocapture`."]
#[test]
fn real_weight_full_row_layout() {
    let Some(mmproj) = real_vision::mmproj_path() else { return };
    let Some(dir) = real_vision::store_dir() else { return };
    let lm = dir.join(real_vision::LM);
    if !lm.exists() {
        eprintln!("skip: {} absent (the tokenizer lives in its KV block)", lm.display());
        return;
    }
    real_vision::pin_cpu_backend();
    mem("start");

    let cfg = DeepseekOcrConfig::deepseek_ocr(1);
    cfg.check_real_scale_shaped();
    let (gh, gw) = cfg.token_grid();
    assert_eq!((gh, gw), (16, 16));
    let dm = cfg.projector_out() as usize;

    // ---- the real prompt: header KV only, no weights ---------------------
    let tok = tokenizer_from_gguf(lm.to_str().expect("utf-8 path")).expect("tokenizer");
    let prompt = build_prompt(&tok, "", "\n<|grounding|>Convert the document to markdown.", gh).expect("prompt");
    let (row0, n_rows) = prompt.image_run();
    println!("== deepseek-ocr real-weight FULL row layout");
    println!("  prompt {} ids, image run [{row0}, {})", prompt.len(), row0 + n_rows);
    assert_eq!(n_rows, 273, "the real global view is 16*(16+1) + 1 rows");
    assert_eq!(prompt.plan.projector_rows(), cfg.image_tokens());
    assert_eq!(prompt.plan.special_rows(), 17);

    // ---- the image -------------------------------------------------------
    let side = cfg.sam.image_h();
    let gpu = gpu_core::testgpu::dev(preprocess::PIPELINES);
    let page = synthetic_page(SRC_W, SRC_H);
    let image = preprocess::preprocess_image(&gpu, &cfg, &page, SRC_W, SRC_H, Fit::Pad);
    assert_eq!(image.len(), 3 * (side * side) as usize);

    // ---- stage 1: the encoder and the row gather --------------------------
    let mg = MmapGguf::open(mmproj.to_str().expect("utf-8 path")).expect("open mmproj");
    let vision = real_vision::encoder_weights(&mg);
    drop(mg);
    mem("mmproj imported");

    // The two learned rows really came out of the checkpoint -- they are the
    // whole point of this test, so their presence and width are asserted before
    // anything is built on them, and they are NOT allowed to be all-zero (a
    // zero-filled tensor would satisfy every equality below and prove nothing).
    let ck_newline = vision.get(IMAGE_NEWLINE).unwrap_or_else(|| panic!("{IMAGE_NEWLINE} missing from the mmproj import")).clone();
    let ck_separator = vision.get(VIEW_SEPARATOR).unwrap_or_else(|| panic!("{VIEW_SEPARATOR} missing from the mmproj import")).clone();
    assert_eq!((ck_newline.len(), ck_separator.len()), (dm, dm), "both learned rows must be [d_model]");
    assert_eq!(real_vision::describe(IMAGE_NEWLINE, &ck_newline), dm);
    assert_eq!(real_vision::describe(VIEW_SEPARATOR, &ck_separator), dm);
    assert!(ck_newline.iter().any(|v| *v != 0.0), "{IMAGE_NEWLINE} is all zero");
    assert!(ck_separator.iter().any(|v| *v != 0.0), "{VIEW_SEPARATOR} is all zero");
    assert_ne!(ck_newline, ck_separator, "the two learned rows are identical -- a swap would be invisible");

    let dev = |k: &'static [(&'static str, &'static str)]| gpu_core::testgpu::dev(k);
    let enc = DeepEncoder::new(&dev, cfg.clone(), &vision, 0, false);
    mem("encoder built (inference)");
    // The ParamStore really holds what the checkpoint had, byte for byte.
    assert_eq!(enc.read_glue_weight(IMAGE_NEWLINE), ck_newline, "the import did not land image_newline verbatim");
    assert_eq!(enc.read_glue_weight(VIEW_SEPARATOR), ck_separator, "the import did not land view_separator verbatim");

    let rg = enc.row_gather(&prompt.plan.rows);
    assert_eq!((rg.rows(), rg.projector_rows()), (n_rows, cfg.image_tokens()));
    assert_eq!(rg.shared_row_counts(), (16, 1), "16 newline rows and one separator");

    let block = enc.forward_rows(&image, &rg);
    mem("encoder forward_rows done");
    assert_eq!(block.len(), n_rows as usize * dm);
    assert_eq!(real_vision::describe("image_block", &block), block.len(), "the block has a non-finite value");
    let proj = enc.read_projector_out();
    let counts = assert_rows_are_their_sources(&block, &prompt.plan.rows, &proj, &ck_newline, &ck_separator, dm);
    assert_eq!(counts, (256, 16, 1));
    println!("  block rows exact: 256 projector, 16 image_newline, 1 view_separator");

    // The contiguous path is unchanged and still produces exactly the 256 rows
    // this block was built from -- the two paths share one encoder forward.
    assert_eq!(enc.forward(&image), proj, "forward and forward_rows disagree about the projector output");
    drop(enc);
    drop(rg);
    mem("encoder dropped");

    // ---- stage 2: the composite, when the decoder's expansion exists ------
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
    let seq = prompt.len() as u32;
    let decoder = WeightReader::open(expanded.to_str().expect("utf-8 path")).expect("open expansion");
    let m = DeepseekOcr::new_with_prompt(&dev, cfg.clone(), &vision, &decoder, 0, seq, &prompt, false);
    drop(decoder);
    drop(vision);
    assert_eq!(m.image_run(), (row0, n_rows), "the splice was not sized from the prompt");
    mem("composite built (inference)");

    m.set_tokens_unsupervised(&prompt.ids);
    let loss = m.forward(&image);
    assert_eq!(loss, 0.0, "every target is IGNORE, so a forward-only run reports no loss");
    mem("composite forward done");

    // The residual stream carries the assembled block VERBATIM over the run,
    // and nothing but it: the same four exact claims, now after the splice.
    let res0 = m.read_decoder_input();
    assert_eq!(res0.len(), seq as usize * dm);
    let spliced = &res0[row0 as usize * dm..(row0 + n_rows) as usize * dm];
    let proj = m.encoder().read_projector_out();
    let counts = assert_rows_are_their_sources(
        spliced,
        &prompt.plan.rows,
        &proj,
        &m.encoder().read_glue_weight(IMAGE_NEWLINE),
        &m.encoder().read_glue_weight(VIEW_SEPARATOR),
        dm,
    );
    assert_eq!(counts, (256, 16, 1));
    println!("  spliced rows exact: 256 projector, 16 image_newline, 1 view_separator at [{row0}, {})", row0 + n_rows);

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
