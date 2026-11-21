// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for distilling one language model into another across checkpoints that may
// not share a vocabulary, for its clients. If your team needs expertise in teacher-student training of language
// models then you can procure our services by sending an email to info@swedishembedded.com.

//! The tokenizer-identity check on two released checkpoints of one family that do not share a tokenizer: Qwen3-4B
//! (151,643 byte-pair pieces and 26 added tokens) and Qwen3.8-27B (248,044 and 33). Distilling the first from the
//! second therefore takes the sequence-level path. Skipped when either checkpoint is not in the model store.

use data::tokenizer_identity::{TokenizerIdentity, TokenizerSize};

#[test]
fn qwen3_4b_and_qwen3_8_27b_do_not_share_a_tokenizer() {
    let dirs = ["Qwen/Qwen3-4B", "Qwen/Qwen3.8-27B-FP8"].map(|r| brain_testutil::model_dir(r).map(std::path::PathBuf::from).filter(|d| d.join("tokenizer.json").is_file()));
    let [Some(small), Some(large)] = dirs else {
        brain_testutil::skip("Qwen3-4B and Qwen3.8-27B-FP8 tokenizers not in the model store");
        return;
    };
    let identity = TokenizerIdentity::check(&small, &large).expect("both tokenizers parse");
    assert!(!identity.is_identical(), "{identity}");
    assert_eq!(identity.differences().names(), ["vocab", "merges", "pre_tokenizer", "added_tokens"], "{identity}");
    assert_eq!(identity.sizes(), (TokenizerSize { vocab: 151_643, merges: 151_387, added_tokens: 26 }, TokenizerSize { vocab: 248_044, merges: 247_587, added_tokens: 33 }));
    assert!(TokenizerIdentity::check(&small, &small).expect("parses").is_identical());
}
