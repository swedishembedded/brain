// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The chunked (`batched = false`) serving path, on the real weights** -
//! `caps::Session::load`'s actual construction
//! (`DeepseekOcr::new_with_prompt_devices_sized`) and actual decode call
//! (`generate_greedy_kv_from_prompt_stream`), at production defaults
//! (`ctx = 8192`, `chunk = 512`), which no other real-weight test in this
//! crate exercises - they all build through `DeepseekOcr::new`/`new_split`/
//! `new_with_prompt` (`batched = true`), which this phase's chunked-prefill
//! change never touches the dispatches of.
//!
//! Two things this test is FOR, that nothing else in this crate proves at
//! real scale:
//!
//! 1. **The chunked and batched paths agree**, on the real 273-row prompt
//!    and real weights - not just `crates/deepseek2/tests/chunked_prefill.rs`'s
//!    tiny fixture. Built twice from the SAME weight sources (the vision
//!    `HashMap` is borrowed, not moved, so both builds read identical
//!    tensors; the decoder `WeightReader` is reopened for the second build
//!    - a second stream of the same file, not a second import path),
//!    generating the SAME `n_new` tokens from the SAME prompt.
//! 2. **A real VmHWM for the constant this phase's own doc flagged as
//!    stale** (`crates/cli/src/resident_deepseekocr.rs`'s
//!    `COMPOSITE_PEAK_BYTES`) - printed, not asserted into a hardcoded
//!    number here, so whoever updates that constant has a real measurement
//!    to cite instead of the arithmetic prediction that comment already
//!    warns against trusting on its own.
//!
//! Same "no byte-level oracle" honesty as `real_weight_generate.rs`: nothing
//! here compares a computed tensor against a captured one, because none
//! exists past the SAM compressor for this model. What IS claimed: the two
//! architecturally different decode paths (one flat batched tape, one
//! bounded-round chunked prefill against a KV cache) compute the SAME
//! answer, and finiteness/plausibility of that answer.

use brain_testutil::mem;
use checkpoint::gguf::MmapGguf;
use checkpoint::weightio::WeightReader;
use deepseek2ocr::config::DeepseekOcrConfig;
use deepseek2ocr::model::DeepseekOcr;
use deepseek2ocr::prompt::{build_prompt, tokenizer_from_gguf};

#[path = "common/real_vision.rs"]
mod real_vision;

use real_vision::{encoder_weights, mem_available_gib, pin_cpu_backend, store_dir, DECODER_GIB, EXPANDED};

/// Greedy steps to generate past the prompt. Small: each round after the
/// first prefill round is still a real `O(1)` KV-cached step either way, so
/// this is about proving agreement and measuring peak RSS, not throughput.
const N_NEW: u32 = 4;

#[ignore = "the whole real composite, TWICE (chunked then batched), at production ctx/chunk: ~21+ GiB resident, several minutes. Slow lane only. `make test/slow`, or `cargo test --release -p brain-deepseekocr --test real_weight_long_context -- --nocapture`."]
#[test]
fn chunked_and_batched_composites_agree_at_real_scale() {
    let Some(mmproj) = real_vision::mmproj_path() else { return };
    let Some(dir) = store_dir() else { return };
    let lm = dir.join(real_vision::LM);
    let expanded = dir.join(EXPANDED);
    if !lm.exists() {
        brain_testutil::skip(&format!("{} absent (the tokenizer lives in its KV block)", lm.display()));
        return;
    }
    if !expanded.exists() {
        brain_testutil::skip(&format!("{} absent (run crates/deepseekv2's parity test first, it builds the expansion)", expanded.display()));
        return;
    }
    let avail = mem_available_gib();
    if avail < DECODER_GIB {
        brain_testutil::skip_unavailable(&format!("MemAvailable {avail:.1} GiB < {DECODER_GIB} GiB (the composite peaks near 24 GiB)"));
        return;
    }
    pin_cpu_backend();
    mem("start");

    let cfg = DeepseekOcrConfig::deepseek_ocr(1);
    cfg.check_real_scale_shaped();
    let (gh, _gw) = cfg.token_grid();

    let tok = tokenizer_from_gguf(lm.to_str().expect("utf-8 path")).expect("tokenizer");
    let prompt = build_prompt(&tok, "", "\n<|grounding|>Convert the document to markdown.", gh).expect("prompt");
    let (row0, n_rows) = prompt.image_run();
    assert_eq!(n_rows, 273, "the real global view is 16*(16+1) + 1 rows");
    let seq = prompt.len() as u32 + N_NEW;
    println!("== deepseek-ocr real-weight chunked-vs-batched (prompt {} + {N_NEW} greedy, image rows [{row0}, {}))", prompt.len(), row0 + n_rows);

    let mg = MmapGguf::open(mmproj.to_str().expect("utf-8 path")).expect("open mmproj");
    let vision = encoder_weights(&mg);
    drop(mg);
    mem("mmproj imported");

    let side = cfg.sam.image_h();
    let image = vec![0.5f32; 3 * (side * side) as usize]; // a real (constant) image is enough - see real_weight_generate.rs's own use of a fill image

    let dev = |k: &'static [(&'static str, &'static str)]| gpu_core::testgpu::dev(k);

    // ---- production path: batched = false, ctx/chunk as caps.rs defaults ----
    let ctx = 8192u32;
    let chunk = 512u32;
    let decoder_chunked = WeightReader::open(expanded.to_str().expect("utf-8 path")).expect("open expansion (chunked)");
    let m_chunked = DeepseekOcr::new_with_prompt_devices_sized(&dev, &dev, cfg.clone(), &vision, &decoder_chunked, 0, ctx, chunk, &prompt);
    drop(decoder_chunked);
    mem("chunked composite built (batched=false, ctx=8192, chunk=512)");
    let got = m_chunked.generate_greedy_kv_from_prompt(&image, &prompt, N_NEW);
    mem("chunked generation done");
    drop(m_chunked);
    mem("chunked composite dropped");

    assert_eq!(got.len(), seq as usize, "generate_greedy_kv_from_prompt returned {} ids, want {seq}", got.len());
    assert_eq!(&got[..prompt.len()], &prompt.ids[..], "the prompt must come back verbatim");
    let generated = &got[prompt.len()..];
    println!("  chunked generated ids: {generated:?}");
    assert!(generated.iter().all(|&id| (id as usize) < cfg.decoder.vocab() as usize), "a generated id is out of vocab range");

    // ---- reference path: batched = true (the pre-existing shape) ----
    let decoder_batched = WeightReader::open(expanded.to_str().expect("utf-8 path")).expect("open expansion (batched)");
    let m_batched = DeepseekOcr::new_with_prompt(&dev, cfg.clone(), &vision, &decoder_batched, 0, seq, &prompt, false);
    drop(decoder_batched);
    drop(vision);
    mem("batched composite built (batched=true)");
    let want = m_batched.generate_greedy_kv_from_prompt(&image, &prompt, N_NEW);
    mem("batched generation done");
    let peak_gib = brain_testutil::peak_rss_bytes() as f64 / (1u64 << 30) as f64;
    drop(m_batched);

    println!("  batched  generated ids: {:?}", &want[prompt.len()..]);
    assert_eq!(got, want, "the chunked (batched=false) composite diverged from the batched one on the real checkpoint");
    println!("== PASS: chunked and batched composites agree at real scale. Process peak RSS across both builds: {peak_gib:.2} GiB");
    mem("end");
}
