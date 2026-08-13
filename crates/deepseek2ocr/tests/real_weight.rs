// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The whole composite on the real Q8_0 checkpoints** - real image -> real
//! SAM -> real CLIP -> real concat -> real projector -> real splice -> real
//! decoder -> logits.
//!
//! ## What is claimed, and what is not
//!
//! `tests/tiny_ref.rs` is the byte-level gate: 31 taps at cosine 1.0 against a
//! checkpoint-free golden. This test is the other half - it runs the same graph
//! at real scale on real weights, where **no end-to-end reference capture
//! exists**. llama.cpp's debug callback segfaults inside the CLIP block graph
//! for this model, so nothing past the SAM compressor was captured and there is
//! no image+decoder golden to compare against.
//!
//! So this test asserts only what it can justify:
//!  * the SAM stage inside the composite reproduces the SAM-only parity result
//!    (`crates/sam1/tests/parity.rs`) - same weights, same input, same numbers,
//!    which proves the composite wires the tower up the way the parity test
//!    does;
//!  * the compressor -> CLIP handoff matches the reference's own assembled CLIP
//!    input, which is the last node the capture reached;
//!  * everything downstream (CLIP output, concat, projector, spliced residual
//!    stream, logits) is checked for being finite and dimensionally right, and
//!    its distribution is **reported, not gated**. Inventing a cosine floor for
//!    a quantity with no reference would be theatre.
//!
//! ## The composite really is `DeepseekOcr`, and it is built for INFERENCE
//!
//! Phase 5 could not build this model through `DeepseekOcr::new` and said so:
//! that constructor took one eager `&HashMap<String, Vec<f32>>` (so the 2.9 B
//! decoder needed its ~12 GB fp32 expansion resident on the host *and* on the
//! device at once) and hardcoded `train = true` (so every parameter also got a
//! gradient and two AdamW moments, ~47 GB in total). Phase 6a closed both:
//!
//!  * every constructor down the chain now takes a
//!    `&dyn checkpoint::TensorSource`, so an eager map coerces and an
//!    mmap-backed `WeightReader` streams one tensor at a time;
//!  * `DeepseekOcr::new_split` takes the encoder's and the decoder's weights
//!    from **two** sources - which is what the shipped checkpoint is, an mmproj
//!    plus an LM file - and threads an explicit `train` flag into all four
//!    stages.
//!
//! So this test builds the real thing: one `DeepseekOcr`, `train = false`, the
//! vision half from the mmproj and the decoder streamed from the cached fp32
//! expansion, and runs `forward` on it. It prints the process's peak RSS
//! (`VmHWM`) as it goes, because "does this fit in 30 GiB of RAM" is the
//! question Phase 6a exists to answer and an estimate would not answer it.
//!
//! The image here is the **constant `0.5` fill** the SAM capture was taken on,
//! and it must stay that way - it is what makes the SAM stage comparable to the
//! reference. `tests/real_weight_image.rs` is the complementary run: a real,
//! non-uniform image through `deepseek2ocr::preprocess`, where no reference
//! exists but a constant fill would be blind to every preprocessing bug.
//!
//! Backend: CPU, for the reasons `crates/sam1/tests/parity.rs` documents.
//! Skips itself when either checkpoint or the fixture is absent.

use brain_testutil::mem;
use brain_testutil::parity::{compare, load};
use brain_testutil::testdata_path as testdata;
use checkpoint::gguf::MmapGguf;
use checkpoint::weightio::WeightReader;
use deepseek2ocr::config::DeepseekOcrConfig;
use deepseek2ocr::model::DeepseekOcr;

/// The model-store lookup, the CPU-backend pin, the one-pass mmproj import and
/// the sanity reporter - shared with this crate's other real-weight binaries.
#[path = "common/real_vision.rs"]
mod real_vision;

use real_vision::{describe, encoder_weights, pin_cpu_backend, store_dir, EXPANDED};

#[ignore = "the whole real composite: a 400 M-parameter encoder at 1024x1024 plus the 2.9 B-parameter decoder (~12 GB resident), ~2 minutes. Slow lane only. `make test/slow`, or `cargo test --release -p brain-deepseekocr --test real_weight -- --nocapture`."]
#[test]
fn real_weight_composite_forward() {
    let fixture = testdata("deepseek-ocr/real/vision.safetensors");
    let Some(dir) = store_dir() else { return };
    let (mmproj, lm, expanded) = (dir.join(real_vision::MMPROJ), dir.join(real_vision::LM), dir.join(EXPANDED));
    if !fixture.exists() || !mmproj.exists() || !lm.exists() {
        eprintln!("skip: real checkpoints or the vision fixture are missing");
        return;
    }
    pin_cpu_backend();
    let golden = load(&fixture);
    mem("start");

    // ---- config + the encoder's weights ----------------------------------
    let mg = MmapGguf::open(mmproj.to_str().expect("utf-8 path")).expect("open mmproj");
    let vision = encoder_weights(&mg);
    let cfg = DeepseekOcrConfig::deepseek_ocr(1);
    drop(mg);
    mem("mmproj imported (encoder weights)");
    // The real-scale invariants: the compressor output IS CLIP's patch token,
    // there is no fixture bridge, and the projector lands on the decoder width.
    cfg.check_real_scale_shaped();
    let seq = cfg.image_tokens() + 2; // BOS, the image run, one trailing text row
    let (row0, n_rows) = (1u32, cfg.image_tokens());
    println!("== deepseek-ocr real-weight composite (seq {seq}, image rows [{row0}, {}))", row0 + n_rows);

    if !expanded.exists() {
        eprintln!(
            "skip: {} absent (run crates/deepseekv2's parity test first, it builds the expansion)",
            expanded.display()
        );
        return;
    }

    // ---- ONE inference composite, both halves streamed --------------------
    //
    // `new_split` because the shipped checkpoint really is two files. The
    // decoder half never becomes a host map: `WeightReader` hands ParamStore one
    // tensor at a time.
    let decoder = WeightReader::open(expanded.to_str().expect("utf-8 path")).expect("open expansion");
    let dev = |k: &'static [(&'static str, &'static str)]| gpu_core::testgpu::dev(k);
    let m = DeepseekOcr::new_split(&dev, cfg.clone(), &vision, &decoder, 0, seq, row0, false);
    drop(decoder);
    drop(vision);
    assert_eq!(m.image_run(), (row0, n_rows));
    mem("composite built (inference)");

    // The same constant-0.5 image the SAM parity fixture was captured on, so
    // the composite's SAM stage is directly comparable to the SAM-only run.
    let fill = golden["input_fill"].data[0];
    let image = vec![fill; (3 * cfg.sam.image_h() * cfg.sam.image_w()) as usize];
    // Token ids are placeholders wherever the image is spliced (their embedding
    // rows are overwritten), so only the two text rows matter; 0 is this
    // checkpoint's BOS.
    let ids: Vec<u32> = (0..seq).map(|_| 0u32).collect();
    m.set_tokens_unsupervised(&ids);
    let loss = m.forward(&image);
    assert_eq!(loss, 0.0, "every target is IGNORE, so a forward-only run reports no loss");
    mem("forward done");

    let enc = m.encoder();
    let embeds = enc.read_projector_out();

    // Rung 1: the SAM tower inside the composite must reproduce the SAM-only
    // parity numbers. This is a real gate - it is the wiring that is under test,
    // and it has a reference.
    let sam = enc.sam();
    let sam_out = sam.gpu.read(sam.output(), sam.out_len());
    let (scos, smax) = compare(&sam_out, &golden["sam_output"].data);
    println!("  sam_output vs reference: cos {scos:.10} max_abs {smax:.3e}");
    assert!(scos > 0.999, "the composite's SAM stage disagrees with the reference (cos {scos})");

    // Rung 2: the tokens the composite hands CLIP are the reference's own
    // assembled CLIP input rows (its last captured node).
    let d = cfg.clip_width() as usize;
    let (tcos, tmax) = compare(&enc.clip().read_tokens(), &golden["clip_input_tokens"].data[d..]);
    println!("  clip patch tokens vs reference: cos {tcos:.10} max_abs {tmax:.3e}");
    assert!(tcos > 0.999, "the compressor -> CLIP handoff disagrees with the reference (cos {tcos})");

    // Rung 3 onwards: no reference exists. Report.
    println!("  downstream of the last captured reference node (reported, NOT gated):");
    let clip_spatial = enc.read_clip_spatial();
    assert_eq!(describe("clip_spatial", &clip_spatial), clip_spatial.len(), "CLIP output has non-finite values");
    let cat = enc.read_vision_concat();
    assert_eq!(describe("vision_concat", &cat), cat.len(), "concat has non-finite values");
    assert_eq!(cat.len(), (n_rows * cfg.projector_in()) as usize);
    assert_eq!(describe("projector_out", &embeds), embeds.len(), "projector output has non-finite values");
    assert_eq!(embeds.len(), (n_rows * cfg.projector_out()) as usize);

    // ---- the splice, and the decoder ------------------------------------
    let res0 = m.read_decoder_input();
    let dm = cfg.decoder.d_model() as usize;
    // The splice really landed: rows [row0, row0+n_rows) of the residual stream
    // ARE the projector output, and the rows outside it are not.
    let spliced = &res0[row0 as usize * dm..(row0 + n_rows) as usize * dm];
    let (rcos, rmax) = compare(spliced, &embeds);
    println!("  spliced residual rows vs projector output: cos {rcos:.10} max_abs {rmax:.3e}");
    assert_eq!(rmax, 0.0, "the splice did not place the projector output verbatim");
    let text_row = &res0[..dm];
    assert!(
        compare(text_row, &embeds[..dm]).0 < 0.9999,
        "row 0 is outside the image run and must still be the token embedding"
    );

    let logits = m.read_logits();
    assert_eq!(describe("logits", &logits), logits.len(), "logits have non-finite values");
    let vocab = cfg.decoder.vocab() as usize;
    let last = &logits[logits.len() - vocab..];
    let mut top: Vec<usize> = (0..vocab).collect();
    top.sort_by(|&a, &b| last[b].total_cmp(&last[a]));
    println!(
        "  top-5 next-token ids at the final position: {:?} (logits {:?})",
        &top[..5],
        top[..5].iter().map(|&i| last[i]).collect::<Vec<f32>>()
    );
    // The only defensible assertion: a decoder that has not diverged produces
    // logits of language-model magnitude, not 1e30 and not all-equal.
    let (lo, hi) = last.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &x| (l.min(x), h.max(x)));
    assert!(hi - lo > 1.0, "the final-position logits are nearly uniform (spread {})", hi - lo);
    assert!(hi < 1e3 && lo > -1e3, "final-position logit range [{lo}, {hi}] is not a plausible distribution");
    mem("end");
}
