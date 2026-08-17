// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ATIF trajectory -> weighted chat/tool-call training data, the first
//! **feedback-source adapter** on top of [`crate::fit_weighted`]'s generic
//! driver. Everything here is qwen3-shaped (chat/tool-call CE-head
//! training data), not architecture-generic, and deliberately so: the
//! generic weighted-training core lives in `lib.rs` (works for any
//! `Head::TokenClassifier` model), while turning a specific reward SOURCE
//! (here, real sven coding sessions) into training examples is inherently
//! domain-specific and does not try to generalize further.
//!
//! ## Scope (v1)
//!
//! - Text-only step content ([`atif::MessageBody::Text`]); a multimodal
//!   ([`atif::MessageBody::Segments`]) step is a hard error, not a silent
//!   drop or lossy flatten.
//! - Direct tool calls/results only; a subagent-delegated observation
//!   (`ObservationEntry::subagent_trajectory_ref`) is a hard error - walking
//!   into embedded subagent trajectories is not implemented yet.
//! - Trajectory-level reward only (P0's stamp on
//!   `Trajectory.final_metrics.extra.reward`), broadcast across every token
//!   the trajectory contributes - matching lecture 3's finding (cited in
//!   the roadmap doc) that outcome-level reward is the right place to
//!   start, before process-level (per-step) reward.
//! - A trajectory with NO reward stamp is skipped, not defaulted to weight
//!   `1.0` - training on a trajectory whose outcome is unknown would be
//!   silently indistinguishable from training on a known-good one, exactly
//!   what the weighted-loss contract exists to prevent.

use std::path::Path;

use atif::{MessageBody, StepOrigin, TraceStep, Trajectory};
use data::chat::{ChatMessage, ChatSample, ToolCall};
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;

/// The trajectory-level reward sven's task machine stamps at
/// `final_metrics.extra.reward` once a task concludes (self-improve roadmap
/// P0). `None` when absent - see this module's doc comment on why that
/// means "skip", not "default to 1.0".
pub fn trajectory_reward(traj: &Trajectory) -> Option<f32> {
    traj.final_metrics.as_ref()?.extra.as_ref()?.get("reward")?.as_f64().map(|r| r as f32)
}

/// Convert one tool invocation's arguments into the compact string form
/// `data::chat::ToolCall::arguments` expects (the chat template renders it
/// verbatim into the tool-call block).
fn tool_call_args(v: &serde_json::Value) -> String {
    v.to_string()
}

/// Convert one ATIF trajectory's SFT-eligible steps ([`Trajectory::
/// sft_steps`]) into brain's packed [`ChatSample`] format - reusing
/// `data::chat`'s existing message/tool-call model rather than inventing a
/// second one. See this module's doc comment for the v1 scope limits.
pub fn to_chat_sample(traj: &Trajectory) -> Result<ChatSample, String> {
    let mut messages: Vec<ChatMessage> = Vec::new();
    for step in traj.sft_steps() {
        push_step(step, &mut messages)?;
    }
    if messages.is_empty() {
        return Err("trajectory has no SFT-eligible steps".to_string());
    }
    Ok(ChatSample { messages, tools: Vec::new() })
}

fn push_step(step: &TraceStep, out: &mut Vec<ChatMessage>) -> Result<(), String> {
    let text = step
        .message
        .as_text()
        .ok_or_else(|| format!("step {}: multimodal message body not supported by rl::atif ingestion yet", step.step_id))?
        .to_string();
    match step.source {
        StepOrigin::System => out.push(ChatMessage::system(text)),
        StepOrigin::User => out.push(ChatMessage::user(text)),
        StepOrigin::Agent => {
            let tool_calls: Vec<ToolCall> = step
                .tool_calls
                .iter()
                .flatten()
                .map(|tc| ToolCall { id: Some(tc.tool_call_id.clone()), name: tc.function_name.clone(), arguments: tool_call_args(&tc.arguments) })
                .collect();
            out.push(if tool_calls.is_empty() {
                ChatMessage::assistant(text, true)
            } else {
                ChatMessage::assistant_tool_calls(text, tool_calls, true)
            });
            if let Some(obs) = &step.observation {
                for entry in &obs.results {
                    if entry.subagent_trajectory_ref.is_some() {
                        return Err(format!("step {}: subagent-delegated observation not supported by rl::atif ingestion yet", step.step_id));
                    }
                    let content = entry
                        .content
                        .as_ref()
                        .and_then(MessageBody::as_text)
                        .ok_or_else(|| format!("step {}: tool observation has no plain-text content", step.step_id))?
                        .to_string();
                    out.push(ChatMessage { role: "tool".to_string(), content, tool_calls: Vec::new(), tool_call_id: entry.source_call_id.clone(), train: false });
                }
            }
        }
    }
    Ok(())
}

/// One trajectory's chat sample plus the reward that weights every token it
/// contributes.
struct WeightedSample {
    sample: ChatSample,
    reward: f32,
}

/// A weighted training block, one entry per token: `(token ids, loss mask,
/// per-token reward weight)`.
type WeightedTokens = (Vec<u32>, Vec<bool>, Vec<f32>);

/// [`data::chat::encode_sample_split`], but tracking each sample's own
/// token span so its reward can be broadcast across exactly those
/// positions - `data::chat` itself has no notion of a per-sample reward
/// (it is a generic chat-transcript format used by plain SFT too), so this
/// stays here rather than widening that crate's surface for one caller.
fn encode_weighted(
    samples: &[WeightedSample],
    tok: &QwenBpe,
    tmpl: &ChatTemplate,
) -> Result<WeightedTokens, String> {
    let mut ids = Vec::new();
    let mut mask = Vec::new();
    let mut weights = Vec::new();
    for ws in samples {
        let (i, m) = ws.sample.encode(tok, tmpl).map_err(|e| e.to_string())?;
        weights.extend(std::iter::repeat_n(ws.reward, i.len()));
        ids.extend(i);
        mask.extend(m);
    }
    Ok((ids, mask, weights))
}

/// Ingest every `*.json` ATIF trajectory file directly under
/// `trajectories_dir` into a [`crate::fit_weighted`]-consumable weighted
/// dataset directory at `out_dir`: `train.{u32,mask,weight}.bin` +
/// `meta.json`, plus empty `val.*` files (`model::load_dataset` requires
/// them to exist; an empty split is `model::train`'s own deliberate
/// "skip eval" signal, not an error - see that crate's own test
/// `load_dataset_accepts_an_empty_validation_split_as_the_deliberate_skip_eval_signal`).
///
/// A trajectory that fails to parse, has no reward stamp
/// ([`trajectory_reward`]), or hits this module's v1 scope limits
/// ([`to_chat_sample`]) is SKIPPED with a logged reason, not a hard error -
/// one malformed or unstamped trajectory in a batch must not block every
/// other trajectory in it from training. Returns the count of trajectories
/// actually ingested (0 is a valid "nothing usable this cycle" result).
pub fn ingest_dir(trajectories_dir: &Path, tok: &QwenBpe, tmpl: &ChatTemplate, vocab: usize, out_dir: &Path) -> std::io::Result<usize> {
    let mut samples = Vec::new();
    let entries = std::fs::read_dir(trajectories_dir)?;
    for entry in entries {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let text = std::fs::read_to_string(&path)?;
        let traj: Trajectory = match serde_json::from_str(&text) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("rl::atif::ingest_dir: {}: not a parseable ATIF trajectory ({e}), skipping", path.display());
                continue;
            }
        };
        let Some(reward) = trajectory_reward(&traj) else {
            eprintln!("rl::atif::ingest_dir: {}: no reward stamp (final_metrics.extra.reward), skipping", path.display());
            continue;
        };
        match to_chat_sample(&traj) {
            Ok(sample) => samples.push(WeightedSample { sample, reward }),
            Err(e) => eprintln!("rl::atif::ingest_dir: {}: {e}, skipping", path.display()),
        }
    }

    std::fs::create_dir_all(out_dir)?;
    let count = samples.len();
    let (ids, mask, weights) = encode_weighted(&samples, tok, tmpl).map_err(std::io::Error::other)?;
    data::binio::write_u32_bin(&out_dir.join("train.u32.bin"), &ids)?;
    data::binio::write_mask_bin(&out_dir.join("train.mask.bin"), &mask)?;
    data::binio::write_f32_bin(&out_dir.join("train.weight.bin"), &weights)?;
    data::binio::write_u32_bin(&out_dir.join("val.u32.bin"), &[])?;
    std::fs::write(out_dir.join("meta.json"), data::binio::Meta::vocab_only(vocab))?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use atif::{AgentProfile, FinalMetrics, ObservationEntry, StepObservation, ToolInvocation};

    fn tiny_env() -> ChatTemplate {
        ChatTemplate::compile("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}").unwrap()
    }

    fn tiny_tok() -> QwenBpe {
        use checkpoint::gguf::GgufTokenizer;
        let gt = GgufTokenizer {
            model: "gpt2".into(),
            pre: Some("qwen2".into()),
            tokens: vec!["<|endoftext|>".into(), "<|im_start|>".into(), "<|im_end|>".into(), "h".into(), "i".into(), "hi".into()],
            merges: vec!["h i".into()],
            token_types: vec![3, 3, 3, 1, 1, 1],
            bos: Some(0),
            eos: Some(2),
            unk: None,
            pad: None,
        };
        QwenBpe::from_gguf(&gt).unwrap()
    }

    fn trajectory_with_reward(reward: Option<f64>) -> Trajectory {
        let mut traj = Trajectory::new("ATIF-v1.7", AgentProfile::new("test-agent", "0.0.1"));
        traj.steps.push(TraceStep::new(1, StepOrigin::User, "hi"));
        let mut agent_step = TraceStep::new(2, StepOrigin::Agent, "hi");
        agent_step.tool_calls = Some(vec![ToolInvocation::new("call-1", "get_weather")]);
        agent_step.observation = Some(StepObservation::single(ObservationEntry::for_call("call-1", "hi")));
        traj.steps.push(agent_step);
        if let Some(r) = reward {
            traj.final_metrics = Some(FinalMetrics { extra: Some(serde_json::json!({ "reward": r })), ..Default::default() });
        }
        traj
    }

    #[test]
    fn trajectory_reward_reads_the_p0_stamp_and_is_none_without_it() {
        assert_eq!(trajectory_reward(&trajectory_with_reward(Some(0.75))), Some(0.75));
        assert_eq!(trajectory_reward(&trajectory_with_reward(None)), None);
    }

    #[test]
    fn to_chat_sample_turns_a_user_turn_plus_tool_call_and_result_into_messages() {
        let traj = trajectory_with_reward(Some(1.0));
        let sample = to_chat_sample(&traj).expect("to_chat_sample");
        assert_eq!(sample.messages.len(), 3, "user + assistant(tool_call) + tool(result)");
        assert_eq!(sample.messages[0].role, "user");
        assert!(!sample.messages[0].train);
        assert_eq!(sample.messages[1].role, "assistant");
        assert!(sample.messages[1].train);
        assert_eq!(sample.messages[1].tool_calls.len(), 1);
        assert_eq!(sample.messages[1].tool_calls[0].name, "get_weather");
        assert_eq!(sample.messages[2].role, "tool");
        assert!(!sample.messages[2].train);
        assert_eq!(sample.messages[2].tool_call_id.as_deref(), Some("call-1"));
    }

    #[test]
    fn ingest_dir_skips_a_trajectory_with_no_reward_stamp_and_counts_only_stamped_ones() {
        let dir = std::env::temp_dir().join(format!("brain-rl-atif-ingest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("stamped.json"), serde_json::to_string(&trajectory_with_reward(Some(1.0))).unwrap()).unwrap();
        std::fs::write(dir.join("unstamped.json"), serde_json::to_string(&trajectory_with_reward(None)).unwrap()).unwrap();
        std::fs::write(dir.join("not-json.txt"), "ignore me").unwrap();

        let out = dir.join("out");
        let count = ingest_dir(&dir, &tiny_tok(), &tiny_env(), 6, &out).expect("ingest_dir");
        assert_eq!(count, 1, "only the reward-stamped trajectory should be ingested");

        let ids = data::binio::read_tokens_u32(&out.join("train")).expect("train.u32.bin");
        let weights = data::binio::read_f32_bin(&out.join("train.weight.bin")).expect("train.weight.bin");
        assert_eq!(ids.len(), weights.len());
        assert!(!ids.is_empty());
        assert!(weights.iter().all(|&w| w == 1.0), "the stamped trajectory's reward (1.0) must cover every token it contributed");
    }
}
