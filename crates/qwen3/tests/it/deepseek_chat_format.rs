// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The serving path's chat format for the real DeepSeek checkpoints: each is
//! rendered with its own template, exactly as `transformers` renders it
//! (`testdata/deepseek/<repo>/chat.json`), the R1 prompts open a reasoning
//! block, and a base model with no template takes raw prompts only.

use capability::Invocation;
use qwen3::chat::{render_prompt_as, ChatFormat};
use serde_json::json;

fn checkpoint(repo: &str) -> Option<(std::path::PathBuf, data::qwen_tokenizer::QwenBpe, serde_json::Value)> {
    let dir = std::path::PathBuf::from(brain_testutil::model_dir(&format!("deepseek-ai/{repo}"))?);
    let golden = brain_testutil::testdata_path(&format!("deepseek/{repo}/chat.json"));
    let tok = data::qwen_tokenizer::QwenBpe::from_file(dir.join("tokenizer.json").to_str()?).ok()?;
    let g = serde_json::from_str(&std::fs::read_to_string(golden).ok()?).ok()?;
    Some((dir, tok, g))
}

#[test]
fn deepseek_checkpoints_serve_in_their_own_chat_format() {
    for (repo, thinks) in [("DeepSeek-R1-Distill-Qwen-1.5B", true), ("DeepSeek-R1-Distill-Llama-8B", true), ("deepseek-coder-1.3b-instruct", false), ("deepseek-llm-7b-chat", false)] {
        let Some((dir, tok, g)) = checkpoint(repo) else {
            brain_testutil::skip(&format!("{repo}: checkpoint or golden missing"));
            continue;
        };
        let format = ChatFormat::for_checkpoint(Some(&dir), &tok);
        assert!(matches!(format, ChatFormat::Template(_)), "{repo} renders with its own template");
        for (name, r) in g["renders"].as_object().unwrap() {
            let inv = Invocation::new().set("messages", json!(r["messages"].to_string()));
            let p = render_prompt_as(&format, &inv).unwrap_or_else(|e| panic!("{repo}/{name}: {e}"));
            assert_eq!(p.text, r["with_generation_prompt"].as_str().unwrap(), "{repo}/{name}");
            assert_eq!(p.thinking_open, thinks, "{repo}/{name}");
        }
    }
    for repo in ["deepseek-llm-7b-base", "deepseek-coder-1.3b-base"] {
        let Some((dir, tok, _)) = checkpoint(repo) else { continue };
        assert!(matches!(ChatFormat::for_checkpoint(Some(&dir), &tok), ChatFormat::None), "{repo} has no chat template");
    }
}
