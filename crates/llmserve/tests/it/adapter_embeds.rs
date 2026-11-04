// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving fine-tuned vision-language models to its clients. If your
// team needs expertise in serving LoRA adapters over multimodal prompts then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a request routed to an adapter is served whatever its prompt carries. A prompt with embedding rows (a
//! vision tower's output) is admitted under the adapter, built inline or on a background encoder, exactly as it is
//! under the base: the adapter's deltas apply to the decoder's layers, not to where the input rows came from.

use std::sync::mpsc;

use capability::Invocation;
use data::qwen_chat::TemplateFlavor;
use data::qwen_tokenizer::QwenBpe;
use llmserve::{run_batch_scheduled, Encoding, Frontend, ParsedRequest};
use model::paged::BlockAllocator;
use model::serve::{AdapterId, PromptEmbeds, SampleParams, Scheduler};
use qwen3::chat::ToolChoice;
use residency::AdapterIdentity;

use crate::common;
use common::{tokenizer, Toy, EMBED_WIDTH};

/// A front end whose requests all carry rows and run under one adapter; `background` builds them on a worker.
struct Vision {
    tok: QwenBpe,
    background: bool,
}

impl Vision {
    fn built() -> Result<(ParsedRequest, Option<PromptEmbeds>), String> {
        let req = ParsedRequest {
            ids: (0..6).collect(),
            max_new: Some(3),
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            sampler: SampleParams::greedy(),
            // seed: greedy sampling never consults it
            seed: 0,
            stops: Vec::new(),
            tool_choice: ToolChoice::Auto,
            flavor: TemplateFlavor::default(),
            thinking_open: false,
            raw: true,
        };
        Ok((req, Some(PromptEmbeds::run(EMBED_WIDTH, 2, vec![0.5; 2 * EMBED_WIDTH]).expect("rows"))))
    }
}

impl Frontend for Vision {
    fn tok(&self) -> &QwenBpe {
        &self.tok
    }
    fn stop(&self) -> &[u32] {
        &[]
    }
    fn label(&self) -> &str {
        "vision"
    }
    fn parse(&self, _inv: &Invocation) -> Result<ParsedRequest, String> {
        Self::built().map(|(req, _)| req)
    }
    fn parse_embedded(&self, _inv: &Invocation) -> Result<(ParsedRequest, Option<PromptEmbeds>), String> {
        Self::built()
    }
    fn parse_in_background(&self, _inv: &Invocation) -> Option<Encoding> {
        let (tx, rx) = mpsc::channel();
        self.background.then(|| {
            tx.send(Self::built()).expect("receiver alive");
            rx
        })
    }
    fn route(&self, _inv: &Invocation) -> Result<Option<(AdapterId, AdapterIdentity)>, String> {
        let identity = AdapterIdentity { id: "candidate".into(), digest: "sha256:0".into() };
        Ok(Some((AdapterId::new(1).expect("namespace 1"), identity)))
    }
}

fn serve(background: bool) -> Vec<capability::ActionResult> {
    let mut sched = Scheduler::new(Toy::new(BlockAllocator::new(128, 8), 0), 4);
    run_batch_scheduled(&mut sched, &Vision { tok: tokenizer(), background }, &[Invocation::new()], &mut |_, _| {})
}

#[test]
fn an_adapter_serves_a_prompt_with_embedding_rows_built_inline() {
    let results = serve(false);
    let out = results[0].as_ref().unwrap_or_else(|e| panic!("refused: {e}"));
    assert_eq!(out.outputs["completion_tokens"].as_u64(), Some(3));
    assert_eq!(out.outputs["adapter"]["id"], "candidate", "the reply names the adapter that answered");
}

#[test]
fn an_adapter_serves_a_prompt_with_embedding_rows_built_in_the_background() {
    let results = serve(true);
    let out = results[0].as_ref().unwrap_or_else(|e| panic!("refused: {e}"));
    assert_eq!(out.outputs["completion_tokens"].as_u64(), Some(3));
    assert_eq!(out.outputs["adapter"]["id"], "candidate", "the reply names the adapter that answered");
}
