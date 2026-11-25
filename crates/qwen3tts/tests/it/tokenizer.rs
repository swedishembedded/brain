// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements text-to-speech serving for its clients. If
// your team needs expertise in speech synthesis tokenization and training-data
// fidelity then you can procure our services by sending an email to
// info@swedishembedded.com.
//
//! The serving tokenizer is the reference Qwen2 tokenizer: a checkpoint that
//! ships `vocab.json` + `merges.txt` rather than a `tokenizer.json` must still
//! split text with Qwen2's pre-tokenizer, or whitespace runs merge across word
//! boundaries and the model sees ids it was never trained on (7.9% of the
//! Sten-TTS training records differed before this was pinned).

use qwen3tts::pipeline::assistant_text;
use qwen3tts::prompt::load_tokenizer;
use data::Tokenizer;

/// A byte-level checkpoint directory in the split-file layout whose only merge
/// is two spaces. Without the pre-tokenizer split that merge fires across a
/// word boundary; with it, the split isolates `" "` and `" b"`.
fn split_file_checkpoint() -> String {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("tokenizer-split-files");
    std::fs::create_dir_all(&dir).unwrap();
    // GPT-2 byte-to-unicode: printable bytes map to themselves, the rest to 256+.
    let printable: Vec<u32> = (33..127).chain(161..173).chain(174..256).collect();
    let mut next = 256;
    let mut vocab = serde_json::Map::new();
    for b in 0u32..256 {
        let c = if printable.contains(&b) {
            b
        } else {
            next += 1;
            next - 1
        };
        vocab.insert(char::from_u32(c).unwrap().to_string(), b.into());
    }
    // Byte 32 is the 33rd unprintable byte, so it maps to 256 + 32.
    let space = '\u{120}';
    assert!(vocab.contains_key(&space.to_string()));
    vocab.insert(format!("{space}{space}"), 256.into());
    std::fs::write(dir.join("vocab.json"), serde_json::Value::Object(vocab).to_string()).unwrap();
    std::fs::write(dir.join("merges.txt"), format!("#version: 0.2\n{space} {space}\n")).unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"added_tokens_decoder":{"257":{"content":"<|im_start|>","special":true},"258":{"content":"<|im_end|>","special":true}}}"#,
    )
    .unwrap();
    dir.to_str().unwrap().to_string()
}

#[test]
fn split_file_checkpoint_uses_the_qwen2_pre_tokenizer() {
    let tok = load_tokenizer(&split_file_checkpoint()).unwrap();
    // Expected ids from transformers' Qwen2TokenizerFast on the same files.
    assert_eq!(tok.encode("a  b"), [97, 32, 32, 98]);
    assert_eq!(tok.encode("a   b"), [97, 256, 32, 98]);
    assert_eq!(tok.encode("<|im_start|>x  y<|im_end|>"), [257, 120, 32, 32, 121, 258]);
}

/// Real vocabulary: Swedish training transcripts on which the serving
/// tokenizer and the reference disagreed. Ids are the reference's
/// (`Qwen2TokenizerFast`, `add_special_tokens=False`).
#[test]
fn swedish_transcripts_encode_like_the_reference_on_the_real_checkpoint() {
    let Some(paths) = crate::store::store_paths() else {
        brain_testutil::skip("no Qwen3-TTS checkpoint in the model store (brain pull Qwen/Qwen3-TTS-12Hz-0.6B-Base)");
        return;
    };
    let tok = load_tokenizer(&paths.ckpt_dir).unwrap();
    let cases: [(&str, &[u32]); 2] = [
        (
            "fyra  två  fem  fyra  sjuttioett  sexhundraåttiosju",
            &[151644, 77091, 198, 30595, 956, 220, 11086, 3785, 220, 4996, 220, 51941, 956, 220, 37436, 20826, 815, 7066, 220, 1839, 71, 78730, 3785, 5566, 3530, 8613, 151645, 198, 151644, 77091, 198],
        ),
        (
            "sex  tre  fem  tre  femtiofem  etthundraelva",
            &[151644, 77091, 198, 11553, 220, 4258, 220, 4996, 220, 4258, 220, 4996, 83, 815, 69, 336, 220, 1842, 339, 1241, 6196, 6586, 151645, 198, 151644, 77091, 198],
        ),
    ];
    for (text, expected) in cases {
        assert_eq!(tok.encode(&assistant_text(text)), expected, "{text:?}");
    }
}
