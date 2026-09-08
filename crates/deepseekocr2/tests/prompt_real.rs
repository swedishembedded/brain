// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The real tokenizer and the real prompt**, against the shipped
//! `deepseek-ocr-2-q8_0.gguf`. No GPU, no weights - only the file's
//! `tokenizer.ggml.*` KV block, so this runs in seconds.
//!
//! The tokenizer is confirmed byte-identical to v1's (M0's ledger: same
//! vocab size, same `tokenizer.ggml.pre`, same reserved-token ids) - so the
//! ground-truth vectors below are the same facts `crates/deepseek2ocr/tests/prompt_real.rs`
//! already pins for the SAME tokenizer, re-checked here against a
//! DIFFERENT file (v2's own GGUF) to confirm the conversion carried the
//! table over intact rather than assumed it did.
//!
//! Self-skips when the checkpoint is not in the model store.

use data::tokenizer::Tokenizer;
use deepseekocr2::prompt::{self, build_prompt};
use deepseekocr2::rows::{row_plan, TileGrid};

#[path = "common/real_vision.rs"]
mod real_vision;

use real_vision::real_files;

/// The complete non-placeholder CONTROL inventory of this vocabulary, in id
/// order - the same table `crates/deepseek2ocr/tests/prompt_real.rs` pins for
/// v1's identical tokenizer.
const CONTROL_TOKENS: &[(u32, &str)] = &[
    (0, "<｜begin▁of▁sentence｜>"),
    (1, "<｜end▁of▁sentence｜>"),
    (2, "<｜▁pad▁｜>"),
    (128800, "<｜fim▁hole｜>"),
    (128801, "<｜fim▁begin｜>"),
    (128802, "<｜fim▁end｜>"),
    (128803, "<｜User｜>"),
    (128804, "<｜Assistant｜>"),
    (128805, "<|EOT|>"),
    (128806, "<｜tool▁calls▁begin｜>"),
    (128807, "<｜tool▁calls▁end｜>"),
    (128808, "<｜tool▁call▁begin｜>"),
    (128809, "<｜tool▁call▁end｜>"),
    (128810, "<｜tool▁outputs▁begin｜>"),
    (128811, "<｜tool▁outputs▁end｜>"),
    (128812, "<｜tool▁output▁begin｜>"),
    (128813, "<｜tool▁output▁end｜>"),
    (128814, "<｜tool▁sep｜>"),
    (128815, "<image>"),
    (128816, "<|ref|>"),
    (128817, "<|/ref|>"),
    (128818, "<|det|>"),
    (128819, "<|/det|>"),
    (128820, "<|grounding|>"),
    (128821, "<td>"),
    (128822, "</td>"),
    (128823, "<tr>"),
    (128824, "</tr>"),
    (128825, "<|User|>"),
    (128826, "<|Assistant|>"),
];

/// Ground truth from HF `tokenizers` on the same vocabulary - digits, CJK,
/// and the actual grounding prompt this model is driven with.
const HF_VECTORS: &[(&str, &[u32])] = &[
    ("Hello", &[19923]),
    ("The capital of France is", &[671, 6102, 294, 8760, 344]),
    ("\n<|grounding|>Convert the document to markdown.", &[201, 128820, 21842, 270, 4940, 304, 2121, 7919, 16]),
    ("\nFree OCR.", &[201, 21431, 126041, 16]),
    ("12345", &[6895, 1883]),
    ("year 2026, price $1299.99", &[24821, 223, 939, 24, 14, 5220, 957, 9603, 27, 16, 1977]),
    ("日本語のテキスト", &[88768, 1576, 17383, 20367, 24552]),
];

#[test]
fn real_tokenizer_reserved_tokens_and_prompt() {
    let Some(files) = real_files() else { return };
    let lm = files.lm.to_string_lossy().into_owned();

    // ---- 1. the reserved-token inventory, straight from the file ----------
    let mg = checkpoint::gguf::MmapGguf::open(&lm).expect("open the LM gguf");
    let gt = mg.tokenizer().expect("the LM gguf declares a tokenizer");
    assert_eq!((gt.model.as_str(), gt.pre.as_deref()), ("gpt2", Some("deepseek-v3")));
    assert_eq!(gt.tokens.len(), 129280);
    assert_eq!((gt.bos, gt.eos, gt.pad), (Some(0), Some(1), Some(2)));

    let control: Vec<(u32, &str)> = gt
        .token_types
        .iter()
        .enumerate()
        .filter(|(_, ty)| **ty == 3 || **ty == 4)
        .map(|(id, _)| (id as u32, gt.tokens[id].as_str()))
        .filter(|(_, t)| !t.starts_with("<｜place▁holder▁no▁"))
        .collect();
    assert_eq!(control, CONTROL_TOKENS, "the checkpoint's control-token inventory moved");
    drop(mg);

    // ---- 2. the tokenizer itself -----------------------------------------
    let tok = prompt::tokenizer_from_gguf(&lm).expect("build the tokenizer from the gguf");
    assert_eq!(tok.vocab_size(), 129280);
    for (id, s) in CONTROL_TOKENS {
        assert_eq!(tok.special_id(s), Some(*id), "special_id({s:?})");
        assert_eq!(tok.encode(s), vec![*id], "encode({s:?})");
    }
    assert_eq!(tok.special_id(prompt::IMAGE), Some(128815));
    assert_eq!(tok.special_id(prompt::BOS), Some(0));
    for (text, want) in HF_VECTORS {
        assert_eq!(&tok.encode(text), want, "encode({text:?})");
        assert_eq!(&tok.decode(want), text, "decode round-trip of {text:?}");
    }

    // ---- 3. the prompt, global view only ----------------------------------
    let plan = row_plan(TileGrid::none(), 144, 256);
    let (before, after) = ("", "\n<|grounding|>Convert the document to markdown.");
    let p = build_prompt(&tok, before, after, plan.len() as u32).expect("build the prompt");

    let (before_ids, after_ids) = (tok.encode(before), tok.encode(after));
    assert_eq!(p.len(), 1 + before_ids.len() + plan.len() + after_ids.len(), "no hidden extras");
    assert_eq!(p.image_run(), (1, plan.len() as u32));
    assert_eq!(p.ids[0], 0, "BOS");
    assert_eq!(&p.ids[(p.row0 + p.n_rows) as usize..], &after_ids[..]);

    let block = &p.ids[p.row0 as usize..(p.row0 + p.n_rows) as usize];
    assert!(block.iter().all(|&i| i == 128815), "every image row, tiles/global/separator alike, is the <image> placeholder");
    println!("prompt: {} ids, image run [{}, {}), {} rows (global {} + separator 1)", p.len(), p.row0, p.row0 + p.n_rows, p.n_rows, 256);
}
