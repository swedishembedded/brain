// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The real tokenizer and the real prompt**, against the shipped
//! `DeepSeek-OCR-Q8_0.gguf`. No GPU, no weights - only the file's
//! `tokenizer.ggml.*` KV block, so this runs in seconds.
//!
//! Three things are pinned here, each with its own oracle:
//!
//! 1. **The reserved-token inventory.** Enumerated from the file's own
//!    `token_type` array, not looked up one string at a time - so this fails if
//!    a re-quantized checkpoint adds, drops or renumbers a control token, and
//!    it is what makes the "there is no newline/separator token" claim in
//!    `deepseek2ocr::prompt` a checked fact rather than a failed search.
//! 2. **Text tokenization, against HF ground truth.** The expected ids were
//!    produced by `tokenizers` on `deepseek-ai/DeepSeek-OCR`'s own
//!    `tokenizer.json` (the vocab and merges in the GGUF are the same table;
//!    what is under test is brain's hand-written pre-tokenizer + merge loop).
//!    `"Hello" -> [19923]` is independently the tokenization
//!    `crates/deepseekv2/tests/generate.rs` anchors its llama.cpp comparison
//!    on, so the two agree on at least one point by construction.
//! 3. **The assembled prompt**: length, the spliced run, and that nothing but
//!    the image placeholder lands inside it.
//!
//! Self-skips when the checkpoint is not in the model store.

use checkpoint::gguf::MmapGguf;
use data::tokenizer::Tokenizer;
use deepseek2ocr::config::DeepseekOcrConfig;
use deepseek2ocr::prompt::{self, build_prompt, ImageTokens};

const STORE: &str = "ggml-org/DeepSeek-OCR-GGUF";
const LM: &str = "DeepSeek-OCR-Q8_0.gguf";

fn lm_path() -> Option<String> {
    let dir = brain_testutil::model_dir(STORE)?;
    let p = std::path::Path::new(&dir).join(LM);
    p.exists().then(|| p.to_string_lossy().into_owned())
}

/// The complete non-placeholder CONTROL inventory of this vocabulary, in id
/// order. `<image>` is in it; an image-newline or view-separator token is not,
/// and that absence is the whole reason `prompt::ImageTokens` points its three
/// row kinds at one id.
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

/// Ground truth from HF `tokenizers` on `deepseek-ai/DeepSeek-OCR`'s
/// `tokenizer.json`, `add_special_tokens=False`. The digit cases are the ones
/// that fail if the `\p{N}{1,3}` cap is not honoured (`"12345"` becomes five
/// ids), the CJK case is the one that would fail if the byte map were wrong,
/// and the two prompt strings are what this model is actually driven with.
const HF_VECTORS: &[(&str, &[u32])] = &[
    ("Hello", &[19923]),
    ("The capital of France is", &[671, 6102, 294, 8760, 344]),
    ("\n<|grounding|>Convert the document to markdown.", &[201, 128820, 21842, 270, 4940, 304, 2121, 7919, 16]),
    ("\nFree OCR.", &[201, 21431, 126041, 16]),
    ("<image>\n<|grounding|>Convert the document to markdown.", &[128815, 201, 128820, 21842, 270, 4940, 304, 2121, 7919, 16]),
    ("12345", &[6895, 1883]),
    ("year 2026, price $1299.99", &[24821, 223, 939, 24, 14, 5220, 957, 9603, 27, 16, 1977]),
    ("日本語のテキスト", &[88768, 1576, 17383, 20367, 24552]),
    ("  spaced", &[223, 48914]),
    ("def main():\n\tpass", &[3465, 1840, 15536, 200, 9762]),
];

#[test]
fn real_tokenizer_reserved_tokens_and_prompt() {
    let Some(lm) = lm_path() else {
        brain_testutil::skip(&format!("{STORE}/{LM} is not in the model store"));
        return;
    };

    // ---- 1. the reserved-token inventory, straight from the file ----------
    let mg = MmapGguf::open(&lm).expect("open the LM gguf");
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
    // The 800 reserved placeholders are the rest of the CONTROL class, and no
    // RESERVED token (control, user-defined or the trailing unused block) names
    // a row separator. The scan is deliberately over the reserved classes only:
    // ordinary vocabulary contains the English words " separator" / "newline",
    // which say nothing about the layout.
    assert_eq!(gt.token_types.iter().filter(|t| **t == 3 || **t == 4).count(), CONTROL_TOKENS.len() + 800);
    for (t, _) in gt.tokens.iter().zip(&gt.token_types).filter(|(_, ty)| matches!(**ty, 3..=5)) {
        let low = t.to_ascii_lowercase();
        assert!(!low.contains("newline") && !low.contains("separator") && !low.contains("seperator"), "unexpected {t:?}");
    }
    drop(mg);

    // ---- 2. the tokenizer itself -----------------------------------------
    let tok = prompt::tokenizer_from_gguf(&lm).expect("build the tokenizer from the gguf");
    assert_eq!(tok.vocab_size(), 129280);
    for (id, s) in CONTROL_TOKENS {
        assert_eq!(tok.special_id(s), Some(*id), "special_id({s:?})");
        // A reserved marker is atomic wherever it appears, not BPE'd apart.
        assert_eq!(tok.encode(s), vec![*id], "encode({s:?})");
    }
    assert_eq!(tok.special_id(prompt::IMAGE), Some(128815));
    assert_eq!(tok.special_id(prompt::BOS), Some(0));
    assert_eq!(tok.special_id(prompt::EOS), Some(1));
    for (text, want) in HF_VECTORS {
        assert_eq!(&tok.encode(text), want, "encode({text:?})");
        assert_eq!(&tok.decode(want), text, "decode round-trip of {text:?}");
    }

    // ---- 3. the prompt ---------------------------------------------------
    let cfg = DeepseekOcrConfig::deepseek_ocr(8192);
    let (g, _) = cfg.token_grid();
    assert_eq!(g, 16, "the real compressor grid");
    // The reference's own prompt, split on `<image>`: nothing before it, the
    // grounding instruction after it.
    let (before, after) = ("", "\n<|grounding|>Convert the document to markdown.");
    let p = build_prompt(&tok, before, after, g).expect("build the prompt");

    let plan = deepseek2ocr::row_plan(g, deepseek2ocr::ViewGrid::global_only());
    let (before_ids, after_ids) = (tok.encode(before), tok.encode(after));
    // Length is exactly BOS + before + the row plan + after - no hidden extras.
    assert_eq!(p.len(), 1 + before_ids.len() + plan.len() + after_ids.len());
    assert_eq!(p.image_run(), (1, 273));
    assert_eq!((p.n_rows as usize, p.plan.projector_rows(), p.plan.special_rows()), (plan.len(), 256, 17));

    let img = ImageTokens::resolve(&tok).unwrap();
    let block = &p.ids[p.row0 as usize..(p.row0 + p.n_rows) as usize];
    assert!(block.iter().all(|&i| img.contains(i)), "a text id leaked into the image block");
    assert!(block.iter().all(|&i| i == 128815), "every image row is <image>");
    assert_eq!(p.ids[0], 0, "BOS");
    assert_eq!(&p.ids[(p.row0 + p.n_rows) as usize..], &after_ids[..]);
    // The instruction really did survive as reserved ids + text, in order.
    assert_eq!(after_ids[1], 128820, "<|grounding|> is one id");

    // One decoder-side run; the newlines break the PROJECTOR side into 16 runs
    // of 16, which is what an embedding-block assembler will iterate.
    let runs = p.plan.runs();
    assert_eq!(runs.len(), 16);
    assert!(runs.iter().all(|(_, _, n)| *n == 16));
    assert_eq!(runs.iter().map(|(_, _, n)| n).sum::<u32>(), 256);
    println!(
        "prompt: {} ids, image run [{}, {}), <image> = {}, plan {} rows ({} projector + {} learned)",
        p.len(),
        p.row0,
        p.row0 + p.n_rows,
        img.image,
        p.n_rows,
        p.plan.projector_rows(),
        p.plan.special_rows()
    );
}
