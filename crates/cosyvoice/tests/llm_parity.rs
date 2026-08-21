// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forward parity vs the real `Qwen2LM.inference()` reference, dumped by
//! `tools/goldens/cosyvoice_dump_reference.py` (`llm_real_*`).
//!
//! Parity ladder actually reached:
//!   1. mapping units - `llm_import::tests::backbone_name_mapping` +
//!      `import_llm_pt`'s own two-way coverage check (this test's import call
//!      IS that rung: a bad mapping fails loudly before any forward runs).
//!
//!   2/3. stage + single-forward parity - THIS FILE: prefill hidden-state and
//!      `llm_decoder` logits, both cosine >= 0.9999 AND `rel_l2` asserted (never
//!      cosine alone).
//!
//!   5. real run (own RNG) - `real_ar_generation_is_seed_deterministic_and_valid`:
//!      an HONEST best-effort check, not exact-token parity. The reference's
//!      `ras_sampling` draws from torch's global Mersenne-Twister RNG;
//!      `crate::sampling` draws from `data::rng::Rng`. The two streams are
//!      unrelated, so this test does NOT assert token-for-token equality
//!      against `llm_real_ar_tokens.i32` - it reports the incidental match
//!      count (informational) and asserts the properties that DO transfer
//!      across the RNG boundary: every generated id is a valid speech-token id,
//!      no stop id leaks into the output, and the same seed reproduces the
//!      same sequence (mirroring the golden's OWN reseed-determinism
//!      self-validation, just against a different RNG).
//!
//! Skips cleanly when the golden or the checkpoint is absent.

use brain_testutil::{golden::Source, parity::Table, read_f32, read_i32, testdata_path};
use cosyvoice::config::CosyVoiceLmConfig;
use cosyvoice::llm::CosyVoiceLm;
use cosyvoice::llm_import::import_llm_pt;

const DUMPER: &str = "tools/goldens/cosyvoice_dump_reference.py";
const COS_FLOOR: f64 = 0.9999;
const REL_CEIL: f64 = 1e-3;

/// `BRAIN_COSYVOICE_LLM`, else the repo-relative `resources/cosyvoice/weights`.
/// Same `weights_dir` convention every other real-weight test in this port uses.
fn weights_dir() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("BRAIN_COSYVOICE_LLM") {
        let p = std::path::PathBuf::from(p);
        return p.join("llm.pt").is_file().then_some(p);
    }
    let p = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/cosyvoice/weights"));
    p.join("llm.pt").is_file().then_some(p)
}

/// Load the LM plus the three golden id arrays the prompt assembly needs.
/// Returns `None` when either the golden or the checkpoint is absent (the
/// caller skips).
fn load() -> Option<(CosyVoiceLm, Vec<u32>, Vec<u32>, std::path::PathBuf)> {
    let dir = testdata_path("golden/cosyvoice");
    let meta = dir.join("llm_real_meta.json");
    let src = Source::open_manifest(&meta, DUMPER)?;
    let cfg = CosyVoiceLmConfig::cosyvoice2();
    if !src.require(&[
        ("llm_input_size", cfg.llm_input_size as i64),
        ("llm_output_size", cfg.llm_output_size as i64),
        ("speech_token_size", cfg.speech_token_size as i64),
        ("llm_decoder_out_features", cfg.speech_vocab() as i64),
    ]) {
        return None;
    }
    let wdir = weights_dir().or_else(|| {
        brain_testutil::skip("set BRAIN_COSYVOICE_LLM to a directory containing llm.pt");
        None
    })?;

    // text_ids = concat([prompt_text, text]) - the reference's own
    // `torch.concat([prompt_text, text], dim=1)` in `Qwen2LM.inference`.
    let prompt_text = read_i32(dir.join("llm_real_prompt_text.i32"))?;
    let text = read_i32(dir.join("llm_real_text.i32"))?;
    let mut text_ids = prompt_text;
    text_ids.extend(text);

    // The prefill's prompt_speech_token is the SAME reference clip's
    // s3tokenizer output (87 tokens) - verified by shape arithmetic against
    // `llm_real_meta.json`'s prefill_hidden_shape (1 + 30 + 1 + 87 = 119),
    // there is no separate `llm_real_prompt_speech_token.i32` file.
    let prompt_speech_tokens = read_i32(dir.join("s3tokenizer_real_tokens.i32"))?;

    let llm_pt = wdir.join("llm.pt");
    let ctx = 1 + text_ids.len() as u32 + 1 + prompt_speech_tokens.len() as u32 + 64;
    let weights = import_llm_pt(llm_pt.to_str().unwrap(), &cfg).unwrap_or_else(|e| panic!("import_llm_pt: {e}"));
    let lm = CosyVoiceLm::from_weights(cfg, weights, ctx);
    Some((lm, text_ids, prompt_speech_tokens, dir))
}

#[test]
fn real_prefill_hidden_and_logits_match_the_reference() {
    let Some((lm, text_ids, prompt_speech_tokens, dir)) = load() else { return };

    let want_hidden = read_f32(dir.join("llm_real_prefill_hidden.f32")).expect("llm_real_prefill_hidden.f32");
    let want_logits = read_f32(dir.join("llm_real_prefill_logits.f32")).expect("llm_real_prefill_logits.f32");

    let got_hidden = lm.prefill(&text_ids, &prompt_speech_tokens);
    assert_eq!(got_hidden.len(), want_hidden.len(), "prefill hidden length");
    let got_logits = lm.decoder_logits_all(&got_hidden);
    assert_eq!(got_logits.len(), want_logits.len(), "prefill logits length");

    let mut table = Table::new(COS_FLOOR, REL_CEIL);
    table.check("llm_real_prefill_hidden", &got_hidden, &want_hidden);
    table.check("llm_real_prefill_logits", &got_logits, &want_logits);
    table.print();
    table.assert_clean();
}

#[test]
fn real_ar_generation_is_seed_deterministic_and_valid() {
    let Some((lm, text_ids, prompt_speech_tokens, dir)) = load() else { return };

    let want_tokens = read_i32(dir.join("llm_real_ar_tokens.i32")).expect("llm_real_ar_tokens.i32");
    let d = lm.cfg.llm_input_size as usize;

    // The reference's own `min_token_text_ratio=2` gate would compute
    // min_len=30 here (target text_len=15 * 2), but the golden's OWN capture
    // stops at 32 tokens without ever needing the eos-ignore boundary - this
    // test isn't exercising that gate, so 0 keeps it out of the way.
    let min_len = 0;
    let seed = 20240727;

    // `generate()` advances the SAME `qwen` KV cache it reads its starting
    // hidden state from (`step_embed` per decoded token), so two calls in a
    // row would decode the second sequence starting from wherever the first
    // one's cache was left, not from the prefill point again. `prefill()`
    // resets the cache to position 0 before replaying the prompt, so a fresh
    // `prefill()` per `generate()` call is what puts each run at the SAME
    // starting state (mirrors this crate's own
    // `llm::tests::generate_is_deterministic_given_the_same_seed`).
    let h1 = lm.prefill(&text_ids, &prompt_speech_tokens);
    let a = lm.generate(&h1[h1.len() - d..], want_tokens.len(), min_len, seed);
    let h2 = lm.prefill(&text_ids, &prompt_speech_tokens);
    let b = lm.generate(&h2[h2.len() - d..], want_tokens.len(), min_len, seed);
    assert_eq!(a, b, "generate() must be deterministic for a fixed seed given a fresh prefill each time");

    for &t in &a {
        assert!(t < lm.cfg.speech_token_size, "generated id {t} is not a valid FSQ speech-token id");
    }

    let matches = a.iter().zip(&want_tokens).filter(|(x, y)| x == y).count();
    println!(
        "AR generation: {}/{} tokens generated, {matches}/{} incidentally match the torch-RNG golden \
         (informational only - brain's sampler intentionally uses its own RNG stream, see \
         crate::sampling's module doc; this is NOT a parity gate).",
        a.len(),
        want_tokens.len(),
        want_tokens.len().min(a.len()),
    );
}
