// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A raw completion request with a `suffix` becomes a fill-in-the-middle
//! prompt framed with the checkpoint's own FIM tokens - DeepSeek-Coder's
//! `<｜fim▁begin｜>`/`<｜fim▁hole｜>`/`<｜fim▁end｜>`, the Qwen vocabulary's
//! `<|fim_prefix|>`/`<|fim_suffix|>`/`<|fim_middle|>` - and a vocabulary with
//! neither refuses it by the fixed reason the API surface reports. A raw
//! completion's output is text as generated: markup a chat scanner would
//! read as reasoning stays in it.
//!
//! Needs the tokenizers in the model store; skips a checkpoint that is not
//! downloaded.

use capability::Invocation;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use qwen3::chat::{parse_request_as, ChatFormat, SeqState, NO_FIM_TOKENS};
use serde_json::json;

fn tokenizer(repo: &str) -> Option<(QwenBpe, ChatFormat)> {
    let dir = brain_testutil::model_dir(repo).filter(|d| std::path::Path::new(d).join("tokenizer.json").exists())?;
    let tok = QwenBpe::from_file(&format!("{dir}/tokenizer.json")).unwrap();
    let format = ChatFormat::for_checkpoint(Some(std::path::Path::new(&dir)), &tok);
    Some((tok, format))
}

fn fim(prefix: &str, suffix: &str) -> Invocation {
    Invocation::new().set("prompt", json!(prefix)).set("suffix", json!(suffix)).set("chat", json!(false))
}

#[test]
fn a_suffix_is_framed_with_the_vocabularys_own_fim_tokens() {
    let cases = [
        ("deepseek-ai/deepseek-coder-1.3b-base", "<｜fim▁begin｜>def add(a, b):\n    return <｜fim▁hole｜>\n<｜fim▁end｜>"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "<|fim_prefix|>def add(a, b):\n    return <|fim_suffix|>\n<|fim_middle|>"),
    ];
    for (repo, framed) in cases {
        let Some((tok, format)) = tokenizer(repo) else {
            brain_testutil::skip(&format!("{repo} not downloaded"));
            continue;
        };
        let req = parse_request_as(&tok, &format, &fim("def add(a, b):\n    return ", "\n")).unwrap();
        assert_eq!(req.ids, tok.encode(framed), "{repo}");
        assert!(req.raw, "{repo}: a completion is raw");
    }
}

#[test]
fn a_suffix_without_fim_tokens_or_on_chat_messages_is_refused() {
    let Some((tok, format)) = tokenizer("deepseek-ai/deepseek-llm-7b-base") else {
        brain_testutil::skip("deepseek-llm-7b-base not downloaded");
        return;
    };
    let err = parse_request_as(&tok, &format, &fim("a", "b")).unwrap_err();
    assert_eq!(err, NO_FIM_TOKENS);

    let Some((tok, format)) = tokenizer("deepseek-ai/deepseek-coder-1.3b-instruct") else {
        brain_testutil::skip("deepseek-coder-1.3b-instruct not downloaded");
        return;
    };
    let chat = Invocation::new().set("messages", json!(r#"[{"role":"user","content":"hi"}]"#)).set("suffix", json!("b"));
    assert!(parse_request_as(&tok, &format, &chat).unwrap_err().contains("suffix"));
}

#[test]
fn a_raw_completion_keeps_its_text_as_generated() {
    let Some((tok, format)) = tokenizer("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") else {
        brain_testutil::skip("DeepSeek-R1-Distill-Qwen-1.5B not downloaded");
        return;
    };
    let inv = Invocation::new().set("prompt", json!("x")).set("chat", json!(false));
    let req = parse_request_as(&tok, &format, &inv).unwrap();
    let generated = "<think>a</think> b";
    let seq = SeqState::new(&req, Default::default());
    let out = seq.finish(&tok, &tok.encode(generated), &mut |_| {});
    assert_eq!(out.outputs.get("reasoning_content").and_then(|v| v.as_str()).unwrap_or_default(), "");
    assert_eq!(String::from_utf8(out.blobs["text"].bytes.clone()).unwrap(), generated);
}
