// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forward parity vs the real `CosyVoice3LM.inference()` reference, dumped by
//! `tools/goldens/cosyvoice3_dump_reference.py` (`llm_real_*`).
//!
//! Mirrors `crates/cosyvoice/tests/llm_parity.rs` (CosyVoice 2's `Qwen2LM`)
//! rung for rung - both are driven by the SAME `crate::llm::CosyVoiceLm`,
//! parameterized by `CosyVoiceLmConfig::cosyvoice3()`'s
//! `SpecialTokenSource::SpeechEmbedding` (`sos`/`task_id` read from
//! `speech_embedding`, no `llm_embedding` table, a bias-free `llm_decoder`)
//! rather than a second, duplicated LM implementation.
//!
//! Parity ladder actually reached:
//!   1. mapping units - `import_llm_pt`'s own two-way coverage check against
//!      the real CosyVoice 3 `llm.pt` IS this rung: a bad mapping (or a
//!      wrongly-required/forbidden `llm_embedding`/`llm_decoder.bias`) fails
//!      loudly before any forward runs.
//!   2. stage + single-forward parity - THIS FILE: prefill hidden-state and
//!      `llm_decoder` logits, both cosine >= 0.9999 AND `rel_l2` asserted.
//!   5. real run (own RNG) - an honest best-effort check, not exact-token
//!      parity, for the same RNG-crossing reason `llm_parity.rs`'s own test
//!      documents (torch's global RNG vs `data::rng::Rng`).
//!
//! Skips cleanly when the golden or the checkpoint is absent.

use brain_testutil::{golden::Source, parity::Table, read_f32, read_i32, testdata_path};
use cosyvoice::config::CosyVoiceLmConfig;
use cosyvoice::llm::CosyVoiceLm;
use cosyvoice::llm_import::import_llm_pt;

const DUMPER: &str = "tools/goldens/cosyvoice3_dump_reference.py";
const COS_FLOOR: f64 = 0.9999;
const REL_CEIL: f64 = 1e-3;

/// `BRAIN_COSYVOICE3_LLM`, else the repo-relative `resources/cosyvoice/weights3`.
fn weights_dir() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("BRAIN_COSYVOICE3_LLM") {
        let p = std::path::PathBuf::from(p);
        return p.join("llm.pt").is_file().then_some(p);
    }
    let p = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/cosyvoice/weights3"));
    p.join("llm.pt").is_file().then_some(p)
}

fn load() -> Option<(CosyVoiceLm, Vec<u32>, Vec<u32>, std::path::PathBuf)> {
    let dir = testdata_path("golden/cosyvoice3");
    let meta = dir.join("llm_real_meta.json");
    let src = Source::open_manifest(&meta, DUMPER)?;
    let cfg = CosyVoiceLmConfig::cosyvoice3();
    if !src.require(&[
        ("llm_input_size", cfg.llm_input_size as i64),
        ("llm_output_size", cfg.llm_output_size as i64),
        ("speech_token_size", cfg.speech_token_size as i64),
        ("llm_decoder_out_features", cfg.speech_vocab() as i64),
    ]) {
        return None;
    }
    let wdir = weights_dir().or_else(|| {
        brain_testutil::skip("set BRAIN_COSYVOICE3_LLM to a directory containing llm.pt");
        None
    })?;

    // text_ids = concat([prompt_text, text]) - the reference's own
    // `torch.concat([prompt_text, text], dim=1)` in `Qwen2LM.inference`
    // (shared, unmodified, by `CosyVoice3LM`). The golden's own prompt
    // already contains `<|endofprompt|>` (151646) per the dumper's own
    // hard-coded assertion, so the caller-side precondition
    // `crate::llm`'s module doc documents is satisfied without re-tokenizing.
    let prompt_text = read_i32(dir.join("llm_real_prompt_text.i32"))?;
    let text = read_i32(dir.join("llm_real_text.i32"))?;
    let mut text_ids = prompt_text;
    text_ids.extend(text);
    assert!(text_ids.contains(&151646), "golden prompt must contain <|endofprompt|> (151646) - CosyVoice3LM's own precondition");

    let prompt_speech_tokens = read_i32(dir.join("s3tokenizer_real_tokens.i32"))?;

    let llm_pt = wdir.join("llm.pt");
    let ctx = 1 + text_ids.len() as u32 + 1 + prompt_speech_tokens.len() as u32 + 64;
    let weights = import_llm_pt(llm_pt.to_str().unwrap(), &cfg).unwrap_or_else(|e| panic!("import_llm_pt: {e}"));
    assert!(weights.llm_embedding.is_none(), "CosyVoice3LM has no llm_embedding table");
    assert!(weights.llm_decoder_b.is_none(), "CosyVoice3LM's llm_decoder is bias-free");
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

    let min_len = 0;
    let seed = 20240727;

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
