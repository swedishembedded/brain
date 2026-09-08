// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The real composite, global view only, on the real Q8_0 checkpoint pair.**
//! Real (constant-fill) image, real SAM, real Qwen2 GQA resampler under the
//! real prefix-LM mask, real projector, real splice, real decoder, logits.
//!
//! ## What is claimed, and what is not
//!
//! `tests/tiny_ref.rs` is the byte-level gate: every stage at cosine >=
//! 0.999999 against a checkpoint-free golden. This test runs the same graph
//! at real scale on real weights, where no independent oracle exists for
//! THIS checkpoint's own numbers (unlike v1, whose SAM tower has a captured
//! llama.cpp reference from the SAME file it ships in - `deepseek-ai/DeepSeek-OCR-2`'s
//! own mmproj has not, as of this test, been captured that way; see M6's
//! ledger entry on why). So every stage here is checked for being finite,
//! dimensionally right, and of a plausible distribution - **reported, not
//! gated**. Inventing a cosine floor for a quantity with no reference would
//! be theatre.
//!
//! ## Global view only
//!
//! SAM's `pos_embed` is a fixed-size checkpoint tensor sized for the
//! 1024x1024 global view's 64x64 patch grid. A 768x768 local tile needs that
//! embedding resampled to a 48x48 grid, which `crates/sam1` does not
//! currently implement (a real gap, not a shortcut - recorded in this
//! crate's ledger entry). This test exercises the global view only
//! (`rows::TileGrid::none()`), which needs no resampling at all; the
//! composite's row-gather/splice mechanism for MULTIPLE tiles is already
//! proven correct against the checkpoint-free golden (`tests/tiny_ref.rs`,
//! `tests/composite.rs`) independently of whether SAM itself can produce a
//! real local-tile grid yet.
//!
//! Backend: CPU, for the reason `crates/sam1/tests/parity.rs` documents.
//! Skips itself when the checkpoint is absent.

use brain_testutil::mem;
use deepseek2::DeepseekV2Config;
use deepseekocr2::encoder::sam_tokens_from_nchw;
use deepseekocr2::import;
use deepseekocr2::model::DeepseekOcr2;
use deepseekocr2::rows::{row_plan, TileGrid};
use sam1::SamEncoder;

#[path = "common/real_vision.rs"]
mod real_vision;

use real_vision::{describe, pin_cpu_backend, real_files};

#[ignore = "the whole real composite: SAM + a 24-layer, 400M-parameter resampler, plus the 2.9B-parameter decoder. Slow lane only. `make test/slow`, or `cargo test --release -p brain-deepseekocr2 --test real_weight -- --nocapture`."]
#[test]
fn real_weight_composite_forward_global_view() {
    let Some(files) = real_files() else { return };
    pin_cpu_backend();
    mem("start");

    let decoder_cfg = DeepseekV2Config::deepseek_ocr(1);
    let vision_cfg = import::vision_config(&files.mmproj, decoder_cfg.shape.d_model).expect("derive the real vision config");
    let plan = row_plan(TileGrid::none(), vision_cfg.encoder.n_query_local, vision_cfg.encoder.n_query_global);
    let (row0, n_rows) = (1u32, plan.len() as u32);
    let seq = n_rows + 2; // BOS, the image run, one trailing text row
    println!("== deepseek-ocr-2 real-weight composite (seq {seq}, image rows [{row0}, {}))", row0 + n_rows);
    mem("vision config derived");

    // ---- real SAM, global view, constant fill ----------------------------
    let gpu_sam = gpu_core::testgpu::dev(sam1::PIPELINES);
    let vision_init = import::vision_reader(&files).expect("open the vision expansion");
    let sam = SamEncoder::new_inference(gpu_sam, vision_cfg.sam.clone(), &vision_init, 0);
    let image = vec![0.5f32; (3 * vision_cfg.sam.image_h() * vision_cfg.sam.image_w()) as usize];
    sam.write_image(&image);
    sam.forward();
    let sam_nchw = sam.gpu.read(sam.output(), sam.out_len());
    let sam_tokens = sam_tokens_from_nchw(&sam_nchw, vision_cfg.sam.compress_out as usize, vision_cfg.encoder.n_query_global as usize);
    assert_eq!(describe("sam_output", &sam_tokens), sam_tokens.len(), "SAM output has non-finite values");
    mem("real SAM forward done");
    drop(sam);

    // ---- the composite: encoder + splice + decoder ------------------------
    let decoder_init = import::decoder_reader(&files).expect("open the decoder expansion");
    let gpu_vision = gpu_core::testgpu::dev(deepseekocr2::encoder::PIPELINES);
    let gpu_decoder = gpu_core::testgpu::dev(deepseek2::PIPELINES);
    let m = DeepseekOcr2::new_on(gpu_vision, gpu_decoder, vision_cfg.clone(), decoder_cfg.clone(), &vision_init, &decoder_init, TileGrid::none(), seq, row0, false);
    drop(vision_init);
    drop(decoder_init);
    assert_eq!(m.image_run(), (row0, n_rows));
    mem("composite built (inference)");

    let ids: Vec<u32> = vec![0u32; seq as usize]; // BOS everywhere; image rows are placeholders, their embedding is overwritten
    m.set_tokens_unsupervised(&ids);
    let (loss, _state) = m.forward(&[], &sam_tokens);
    assert_eq!(loss, 0.0, "every target is IGNORE, so a forward-only run reports no loss");
    mem("forward done");

    let projected = m.encoder().resample_view(&sam_tokens, false).projected;
    assert_eq!(describe("projector_out", &projected), projected.len(), "projector output has non-finite values");
    assert_eq!(projected.len(), (vision_cfg.encoder.n_query_global * vision_cfg.decoder_hidden) as usize);

    // ---- the splice really landed --------------------------------------
    let res0 = m.read_decoder_input();
    let dm = decoder_cfg.shape.d_model as usize;
    let spliced = &res0[row0 as usize * dm..(row0 as usize + vision_cfg.encoder.n_query_global as usize) * dm];
    assert_eq!(spliced, &projected[..], "the splice did not place the projector output verbatim");
    let text_row = &res0[..dm];
    assert_ne!(text_row, &projected[..dm], "row 0 is outside the image run and must still be the token embedding");

    let logits = m.decoder().read_logits();
    assert_eq!(describe("logits", &logits), logits.len(), "logits have non-finite values");
    let vocab = decoder_cfg.vocab() as usize;
    let last = &logits[logits.len() - vocab..];
    let (lo, hi) = last.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &x| (l.min(x), h.max(x)));
    println!("  final-position logit range [{lo:.3}, {hi:.3}]");
    assert!(hi - lo > 1.0, "the final-position logits are nearly uniform (spread {})", hi - lo);
    assert!(hi < 1e3 && lo > -1e3, "final-position logit range [{lo}, {hi}] is not a plausible distribution");
    mem("end");
}
