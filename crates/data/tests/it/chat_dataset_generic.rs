// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A chat dataset is the checkpoint's own conversations and nothing else,
//! whatever its vocabulary: a DeepSeek-Coder dataset (32256 ids) indexes one
//! example per sample out of band, holds no id its model does not have,
//! opens each example with exactly one BOS and trains the end-of-turn token
//! that ends each answer; an R1 dataset trains the reasoning it is given
//! when asked to, although R1's template drops it from rendered history.
//!
//! Needs the tokenizers in the model store; skips a checkpoint that is not
//! downloaded.

use std::path::{Path, PathBuf};

use data::binio;
use data::chat::{ChatMessage, ChatSample, RenderOpts};
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;

fn checkpoint(repo: &str) -> Option<(QwenBpe, ChatTemplate)> {
    let dir = brain_testutil::model_dir(repo).filter(|d| Path::new(d).join("tokenizer.json").exists())?;
    let tok = QwenBpe::from_file(&format!("{dir}/tokenizer.json")).unwrap();
    let tmpl = ChatTemplate::from_model_dir(Path::new(&dir)).unwrap();
    Some((tok, tmpl))
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-chat-dataset-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn conversation(i: usize) -> ChatSample {
    ChatSample {
        messages: vec![
            ChatMessage::user(format!("Write a function returning {i}.")),
            ChatMessage::assistant(format!("def f():\n    return {i}"), true),
            ChatMessage::user("Now add a docstring."),
            ChatMessage::assistant(format!("def f():\n    \"\"\"Return {i}.\"\"\"\n    return {i}"), true),
        ],
        tools: Vec::new(),
        rendered: None,
        media: Vec::new(),
        rollout: None,
    }
}

#[test]
fn a_32k_vocabulary_dataset_is_one_example_per_sample() {
    let Some((tok, tmpl)) = checkpoint("deepseek-ai/deepseek-coder-1.3b-instruct") else {
        brain_testutil::skip("deepseek-coder-1.3b-instruct not downloaded");
        return;
    };
    const VOCAB: usize = 32256;
    let (bos, eot) = (tok.special_id("<｜begin▁of▁sentence｜>").unwrap(), tok.special_id("<|EOT|>").unwrap());
    let samples: Vec<ChatSample> = (0..5).map(conversation).collect();
    let dir = scratch("coder");
    let prepared = data::chat::prepare_chat_samples(&samples, &samples[..2], &tok, &tmpl, RenderOpts::default(), VOCAB, &dir).unwrap();

    let ids = binio::read_u32_bin(&dir.join("train.u32.bin")).unwrap();
    let mask = binio::read_mask_bin(&dir.join("train.mask.bin")).unwrap();
    let starts = binio::read_u64_bin(&dir.join("train.ex.bin")).unwrap();
    assert_eq!(starts.len(), samples.len(), "one example per sample");
    assert_eq!(binio::read_u64_bin(&dir.join("val.ex.bin")).unwrap().len(), 2);
    assert!(ids.iter().all(|&t| (t as usize) < VOCAB), "an id past the vocabulary");

    for (i, &a) in starts.iter().enumerate() {
        let a = a as usize;
        let b = starts.get(i + 1).map_or(ids.len(), |&s| s as usize);
        let (ex, m) = (&ids[a..b], &mask[a..b]);
        assert_eq!(ex.iter().filter(|&&t| t == bos).count(), 1, "example {i}: exactly one BOS");
        assert_eq!(ex[0], bos, "example {i} opens with its BOS");
        let trained_eots = ex.iter().zip(m).filter(|(&t, &on)| t == eot && on).count();
        assert_eq!(trained_eots, 2, "example {i}: each answer's end-of-turn token is trained");
        assert!(b - a <= prepared.longest_example);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn reasoning_is_trained_only_when_kept() {
    let Some((tok, tmpl)) = checkpoint("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") else {
        brain_testutil::skip("DeepSeek-R1-Distill-Qwen-1.5B not downloaded");
        return;
    };
    let sample = ChatSample {
        messages: vec![ChatMessage::user("What is 17 + 25?"), ChatMessage::assistant("<think>\n17 + 25: seven and five carry one.\n</think>\n\n42", true)],
        tools: Vec::new(),
        rendered: None,
        media: Vec::new(),
        rollout: None,
    };
    let trained = |opts: RenderOpts| {
        let (ids, mask) = sample.encode_with(&tok, &tmpl, opts).unwrap();
        let kept: Vec<u32> = ids.iter().zip(&mask).filter(|(_, &on)| on).map(|(&t, _)| t).collect();
        tok.decode(&kept)
    };
    let kept = trained(RenderOpts { keep_reasoning: true });
    assert!(kept.contains("<think>\n17 + 25: seven and five carry one.\n</think>\n\n42"), "{kept:?}");
    assert!(kept.ends_with("<｜end▁of▁sentence｜>"), "the answer's end token is trained: {kept:?}");
    let dropped = trained(RenderOpts::default());
    assert!(!dropped.contains("carry one") && dropped.contains("42"), "R1's template drops history reasoning: {dropped:?}");
}

/// A preference pair trains on the form its model is asked from: with the
/// reasoning kept, each candidate keeps the closed think block it opens with,
/// which R1's template would otherwise strip from the rendered turn.
#[test]
fn a_preference_pair_keeps_the_think_block_its_candidates_open_with_when_asked() {
    let Some((tok, tmpl)) = checkpoint("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") else {
        brain_testutil::skip("DeepSeek-R1-Distill-Qwen-1.5B not downloaded");
        return;
    };
    let turn = |answer: &str| ChatSample {
        messages: vec![ChatMessage::user("Shall we wait?"), ChatMessage::assistant(format!("<think>\n\n</think>\n\n{answer}"), true)],
        tools: Vec::new(),
        rendered: None,
        media: Vec::new(),
        rollout: None,
    };
    let pair = data::preference::PreferenceSample { chosen: turn("Do not wait."), rejected: turn("Wait."), prompt_len: 1 };
    let supervised = |opts: RenderOpts| {
        let (chosen, rejected) = pair.encode_with(&tok, &tmpl, opts).unwrap();
        (tok.decode(&chosen.supervised()), tok.decode(&rejected.supervised()))
    };
    let (chosen, rejected) = supervised(RenderOpts { keep_reasoning: true });
    assert!(chosen.contains("<think>\n\n</think>\n\nDo not wait."), "{chosen:?}");
    assert!(rejected.contains("<think>\n\n</think>\n\nWait."), "{rejected:?}");
    let (dropped, _) = supervised(RenderOpts::default());
    assert!(!dropped.contains("</think>"), "R1's template strips it: {dropped:?}");
}

/// An agent continuation, as a preference candidate or a feedback completion, trains on each answer the way the
/// model was asked for it, exactly as packed SFT trains a dialogue. Qwen3's template drops the reasoning of every
/// assistant turn before the last user turn, and an agent run carries user turns mid-run (a time-left notice). A
/// candidate rendered whole is therefore not prefix-stable, and it was refused for most real rollouts.
#[test]
fn an_agent_continuation_with_reasoning_trains_each_answer_as_asked_in_preference_and_feedback() {
    let Some((tok, tmpl)) = checkpoint("Qwen/Qwen3-4B") else {
        brain_testutil::skip("Qwen3-4B not downloaded");
        return;
    };
    let run = |last: &str| {
        format!(
            r#"[{{"role":"assistant","content":"<think>\nList first.\n</think>\n\nI will list the files."}},{{"role":"user","content":"Time left: 5 minutes."}},{{"role":"assistant","content":"<think>\nWrite it.\n</think>\n\n{last}"}}]"#
        )
    };
    let prompt = r#"[{"role":"system","content":"You are an agent."},{"role":"user","content":"Fix the report."}]"#;
    let line = format!(r#"{{"prompt":{prompt},"chosen":{},"rejected":{}}}"#, run("Done: answer.txt written."), run("I give up."));
    let pair = data::preference::PreferenceSample::from_json_line(&line).expect("parses");
    let (chosen, rejected) = pair.encode_with(&tok, &tmpl, RenderOpts::default()).expect("a continuation with reasoning renders as packed SFT renders it");
    let packed = |sample: &ChatSample| {
        let packed = sample.answers_packed(&tmpl, true).unwrap().expect("supervises assistant turns");
        let (ids, mask) = packed.encode_with(&tok, &tmpl, RenderOpts::default()).unwrap();
        tok.decode(&ids.iter().zip(&mask).filter(|(_, m)| **m).map(|(i, _)| *i).collect::<Vec<_>>())
    };
    for (side, encoded, sample) in [("chosen", &chosen, &pair.chosen), ("rejected", &rejected, &pair.rejected)] {
        let supervised = tok.decode(&encoded.supervised());
        assert_eq!(supervised, packed(sample), "{side}: the supervised tokens are packed SFT's");
        assert!(supervised.contains("Write it.") && supervised.contains("List first."), "{side}: each answer keeps the reasoning it was asked with: {supervised:?}");
    }

    let feedback = format!(r#"{{"prompt":{prompt},"completion":{},"desirable":true}}"#, run("Done: answer.txt written."));
    let example = data::feedback::FeedbackSample::from_json_line(&feedback).expect("parses");
    let encoded = example.encode_with(&tok, &tmpl, RenderOpts::default()).expect("a feedback continuation renders packed");
    assert_eq!(tok.decode(&encoded.supervised()), packed(&example.sample));
}
