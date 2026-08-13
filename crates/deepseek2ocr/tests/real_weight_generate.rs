// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The composed decode loop on the whole real composite** - image -> SAM ->
//! CLIP -> concat -> projector -> splice -> decoder, then several greedy steps
//! continuing from the spliced sequence.
//!
//! `crates/deepseekv2/tests/generate.rs` is the exact-oracle version of this:
//! eight greedy steps of the text decoder alone, matched token for token
//! against llama.cpp on the same weights. **This test has no such oracle and
//! does not pretend to.** llama.cpp's graph-eval debug callback segfaults inside
//! this model's CLIP graph, so nothing past the SAM compressor was ever
//! captured; and a `llama-mtmd-cli` run on a real image would produce generated
//! TEXT for eyeballing, not token ids, because brain's and llama.cpp's image
//! preprocessing differ at this point. Inventing a reference for the multimodal
//! multi-step case would be theatre.
//!
//! So this asserts only what it can justify, which is more than "it ran":
//!
//!  * the loop **completes** with a real 400 M-parameter encoder in front of the
//!    2.9 B-parameter decoder at the production 1024x1024 / 258-row shape, and
//!    the prompt (image placeholders and all) comes back verbatim;
//!  * every logit of the deciding forward is **finite**, and the deciding
//!    distribution is of language-model magnitude rather than uniform or 1e30;
//!  * **causal self-consistency across every step, for free.** With a correct
//!    causal mask, position `i`'s logits do not depend on tokens after `i` - so
//!    ONE forward over the length-`L-1` prefix must reproduce, at each position
//!    `i`, exactly the argmax that the step-time forward over `[0..=i]` chose.
//!    That is what the last `logits_all` inside the loop already computed, so
//!    checking it costs nothing, and it fails loudly if the mask leaks the
//!    future or if RoPE's position argument does not advance with the sequence.
//!
//! Its own test binary, not a second `#[test]` in `real_weight.rs`: cargo runs
//! test binaries one at a time, and two ~21 GiB composites resident at once
//! would exhaust any machine that can host one. Backend: CPU, for the reasons
//! `crates/sam1/tests/parity.rs` documents. Skips itself when the checkpoints
//! or the fixture are absent.

use brain_testutil::mem;
use brain_testutil::parity::load;
use brain_testutil::testdata_path as testdata;
use checkpoint::gguf::MmapGguf;
use checkpoint::weightio::WeightReader;
use deepseek2ocr::config::DeepseekOcrConfig;
use deepseek2ocr::model::DeepseekOcr;

/// The model-store lookup, the CPU-backend pin and the one-pass mmproj import -
/// shared with this crate's other real-weight binaries.
#[path = "common/real_vision.rs"]
mod real_vision;

use real_vision::{encoder_weights, pin_cpu_backend, store_dir, EXPANDED};

/// Greedy steps. Each is a FULL recompute over the whole ~260-row sequence
/// through 12 layers of a 2.9 B-parameter MoE, so this is minutes, not seconds;
/// the property under test does not get truer with more of them.
const N_NEW: u32 = 3;

/// Index of the greatest element, ties to the lowest index - the same rule
/// `deepseekv2`'s greedy sampler and llama.cpp's both use.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate().skip(1) {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

#[ignore = "the whole real composite driven for several decode steps: a 400 M-parameter encoder at 1024x1024 plus the 2.9 B-parameter decoder (~21 GiB resident), and one FULL ~260-token recompute per step. Slow lane only. `make test/slow`, or `cargo test --release -p brain-deepseekocr --test real_weight_generate -- --nocapture`."]
#[test]
fn real_weight_composite_greedy_decode() {
    let fixture = testdata("deepseek-ocr/real/vision.safetensors");
    let Some(dir) = store_dir() else { return };
    let (mmproj, lm, expanded) = (dir.join(real_vision::MMPROJ), dir.join(real_vision::LM), dir.join(EXPANDED));
    if !fixture.exists() || !mmproj.exists() || !lm.exists() {
        eprintln!("skip: real checkpoints or the vision fixture are missing");
        return;
    }
    if !expanded.exists() {
        eprintln!("skip: {} absent (run crates/deepseekv2's parity test first, it builds the expansion)", expanded.display());
        return;
    }
    pin_cpu_backend();
    let golden = load(&fixture);
    mem("start");

    let mg = MmapGguf::open(mmproj.to_str().expect("utf-8 path")).expect("open mmproj");
    let vision = encoder_weights(&mg);
    drop(mg);
    let cfg = DeepseekOcrConfig::deepseek_ocr(1);
    cfg.check_real_scale_shaped();
    mem("mmproj imported (encoder weights)");

    // BOS, the image run, one trailing text row -- `tests/real_weight.rs`'s
    // exact prompt shape -- plus room for the tokens this test generates.
    let (row0, n_rows) = (1u32, cfg.image_tokens());
    let prompt_len = cfg.image_tokens() + 2;
    let seq = prompt_len + N_NEW;
    println!("== deepseek-ocr real-weight composed loop (prompt {prompt_len}, +{N_NEW} greedy, image rows [{row0}, {}))", row0 + n_rows);

    let decoder = WeightReader::open(expanded.to_str().expect("utf-8 path")).expect("open expansion");
    let dev = |k: &'static [(&'static str, &'static str)]| gpu_core::testgpu::dev(k);
    let m = DeepseekOcr::new_split(&dev, cfg.clone(), &vision, &decoder, 0, seq, row0, false);
    drop(decoder);
    drop(vision);
    mem("composite built (inference)");

    // The same constant-fill image the SAM parity fixture was captured on, and
    // the same all-BOS placeholder ids: the image rows' embeddings are
    // overwritten by the splice, so only the two text rows' ids matter.
    let fill = golden["input_fill"].data[0];
    let image = vec![fill; (3 * cfg.sam.image_h() * cfg.sam.image_w()) as usize];
    let prompt: Vec<u32> = vec![0u32; prompt_len as usize];

    let ids = m.generate_greedy(&image, &prompt, N_NEW);
    mem("generation done");
    assert_eq!(ids.len(), seq as usize, "generate_greedy returned {} ids, want {seq}", ids.len());
    assert_eq!(&ids[..prompt_len as usize], &prompt[..], "the prompt must come back verbatim");
    println!("  generated ids: {:?}", &ids[prompt_len as usize..]);

    // The deciding forward's logits: the loop's LAST `logits_all`, over
    // `ids[..seq-1]`. Every position of it is live.
    let vocab = cfg.decoder.vocab() as usize;
    let logits = m.read_logits();
    let live = (seq as usize - 1) * vocab;
    assert!(logits.len() >= live, "logits buffer is shorter than the deciding forward");
    let logits = &logits[..live];
    let nonfinite = logits.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(nonfinite, 0, "{nonfinite} of {live} logits are not finite");

    // Causal self-consistency: one forward, every step's choice re-derived at
    // its own position. See the module header for why this is free and why it
    // is not vacuous.
    for i in (prompt_len as usize - 1)..(seq as usize - 1) {
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

    // The deciding distribution, reported and sanity-gated (no reference exists).
    let last = &logits[live - vocab..];
    let mut top: Vec<usize> = (0..vocab).collect();
    top.sort_by(|&a, &b| last[b].total_cmp(&last[a]));
    println!("  final-step top-5: {:?}", top[..5].iter().map(|&i| (i, last[i])).collect::<Vec<_>>());
    let (lo, hi) = last.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &x| (l.min(x), h.max(x)));
    assert!(hi - lo > 1.0, "the deciding logits are nearly uniform (spread {})", hi - lo);
    assert!(hi < 1e3 && lo > -1e3, "logit range [{lo}, {hi}] is not a plausible distribution");
    mem("end");
}
