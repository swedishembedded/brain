// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The composed decode loop on the real composite, global view only** -
//! real SAM -> real resampler -> real projector -> real splice -> several
//! real greedy decode steps.
//!
//! Like `crates/deepseek2ocr/tests/real_weight_generate.rs`, this has no
//! external oracle (no captured reference exists for this checkpoint's own
//! multi-step output) and does not pretend to. What it asserts is real
//! nonetheless:
//!
//!  * the loop **completes** at real scale, and the prompt (image
//!    placeholders and all) comes back verbatim;
//!  * every logit of the deciding forward is **finite** and of
//!    language-model magnitude;
//!  * **causal self-consistency across every step, for free.** With a
//!    correct causal mask, position `i`'s logits do not depend on tokens
//!    after `i`, so ONE forward over the length-`L-1` prefix must reproduce,
//!    at every position `i`, exactly the argmax the step-time forward over
//!    `[0..=i]` chose - `deepseek2::DeepseekV2::generate_greedy_cb`'s own
//!    `logits_all` already computes this, so checking it costs nothing and
//!    fails loudly if the causal mask leaks the future or RoPE's position
//!    argument does not advance with the sequence.
//!
//! Its own test binary, not a second `#[test]` in `real_weight.rs`: cargo
//! runs test binaries one at a time, and two composites of this size
//! resident at once would exhaust this box. Backend: CPU. Skips itself when
//! the checkpoint is absent.

use brain_testutil::mem;
use deepseek2::DeepseekV2Config;
use deepseekocr2::encoder::sam_tokens_from_nchw;
use deepseekocr2::import;
use deepseekocr2::model::DeepseekOcr2;
use deepseekocr2::prompt::build_prompt;
use deepseekocr2::rows::TileGrid;
use sam1::SamEncoder;

#[path = "common/real_vision.rs"]
mod real_vision;

use real_vision::{pin_cpu_backend, real_files};

/// Greedy steps. Each is a full recompute over the whole ~260-row sequence
/// through the decoder's 12 MoE layers, so this is minutes, not seconds; the
/// property under test does not get truer with more of them.
const N_NEW: u32 = 3;

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate().skip(1) {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

#[ignore = "the whole real composite driven for several decode steps: SAM + the resampler + the 2.9B-parameter decoder, one full recompute per step. Slow lane only. `make test/slow`, or `cargo test --release -p brain-deepseekocr2 --test real_weight_generate -- --nocapture`."]
#[test]
fn real_weight_composite_greedy_decode_global_view() {
    let Some(files) = real_files() else { return };
    pin_cpu_backend();
    mem("start");

    let decoder_cfg = DeepseekV2Config::deepseek_ocr(1);
    let vision_cfg = import::vision_config(&files.mmproj, decoder_cfg.shape.d_model).expect("derive the real vision config");

    let lm = files.lm.to_string_lossy().into_owned();
    let tok = deepseekocr2::prompt::tokenizer_from_gguf(&lm).expect("build the tokenizer");
    let n_rows = vision_cfg.encoder.n_query_global + 1; // the global view's rows plus the separator
    let after = "\n<|grounding|>Convert the document to markdown.";
    let prompt = build_prompt(&tok, "", after, n_rows).expect("build the prompt");
    let seq = prompt.len() as u32 + N_NEW;
    println!("== deepseek-ocr-2 real-weight composed loop (prompt {}, +{N_NEW} greedy, image rows {:?})", prompt.len(), prompt.image_run());

    // ---- real SAM, global view, constant fill ----------------------------
    let gpu_sam = gpu_core::testgpu::dev(sam1::PIPELINES);
    let vision_init = import::vision_reader(&files).expect("open the vision expansion");
    let sam = SamEncoder::new_inference(gpu_sam, vision_cfg.sam.clone(), &vision_init, 0);
    let image = vec![0.5f32; (3 * vision_cfg.sam.image_h() * vision_cfg.sam.image_w()) as usize];
    sam.write_image(&image);
    sam.forward();
    let sam_nchw = sam.gpu.read(sam.output(), sam.out_len());
    let sam_tokens = sam_tokens_from_nchw(&sam_nchw, vision_cfg.sam.compress_out as usize, vision_cfg.encoder.n_query_global as usize);
    mem("real SAM forward done");
    drop(sam);

    // ---- the composite: prime the vision half, then decode ----------------
    let decoder_init = import::decoder_reader(&files).expect("open the decoder expansion");
    let gpu_vision = gpu_core::testgpu::dev(deepseekocr2::encoder::PIPELINES);
    let gpu_decoder = gpu_core::testgpu::dev(deepseek2::PIPELINES);
    let (row0, rows) = prompt.image_run();
    let m = DeepseekOcr2::new_on(gpu_vision, gpu_decoder, vision_cfg.clone(), decoder_cfg.clone(), &vision_init, &decoder_init, TileGrid::none(), seq, row0, false);
    drop(vision_init);
    drop(decoder_init);
    assert_eq!(m.image_run(), (row0, rows));
    let _ = m.prime_vision(&[], &sam_tokens);
    mem("composite built, vision primed");

    let ids = m.decoder().generate_greedy(&prompt.ids, N_NEW);
    mem("generation done");
    assert_eq!(ids.len(), seq as usize, "generate_greedy returned {} ids, want {seq}", ids.len());
    assert_eq!(&ids[..prompt.len()], &prompt.ids[..], "the prompt must come back verbatim");
    println!("  generated ids: {:?}", &ids[prompt.len()..]);

    let vocab = decoder_cfg.vocab() as usize;
    let logits = m.decoder().read_logits();
    let live = (seq as usize - 1) * vocab;
    assert!(logits.len() >= live, "logits buffer is shorter than the deciding forward");
    let logits = &logits[..live];
    let nonfinite = logits.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(nonfinite, 0, "{nonfinite} of {live} logits are not finite");

    for i in (prompt.len() - 1)..(seq as usize - 1) {
        let row = &logits[i * vocab..(i + 1) * vocab];
        assert_eq!(
            argmax(row),
            ids[i + 1] as usize,
            "position {i} of the length-{} forward picks {} but the step-time forward picked {} -- \
             the causal mask or the RoPE position advance depends on tokens it must not see",
            seq - 1,
            argmax(row),
            ids[i + 1]
        );
    }

    let last = &logits[live - vocab..];
    let (lo, hi) = last.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &x| (l.min(x), h.max(x)));
    println!("  deciding logit range [{lo:.3}, {hi:.3}]");
    assert!(hi - lo > 1.0, "the deciding logits are nearly uniform (spread {})", hi - lo);
    assert!(hi < 1e3 && lo > -1e3, "logit range [{lo}, {hi}] is not a plausible distribution");
    mem("end");
}
