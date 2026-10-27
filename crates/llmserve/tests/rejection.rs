// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements LLM serving whose failures a client can act on instead of parse. If your team needs
// expertise in error contracts for inference services, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a prompt refused because the card ran out of memory reaches the caller with a failure class, so a client
//! can tell "retry later or shorten the prompt" from a malformed request.

use capability::failure::{classify, FailureClass};
use capability::Invocation;
use data::qwen_tokenizer::QwenBpe;
use llmserve::{Frontend, ParsedRequest};
use model::serve::RejectReason;

struct Plain(QwenBpe);

impl Frontend for Plain {
    fn tok(&self) -> &QwenBpe {
        &self.0
    }
    fn stop(&self) -> &[u32] {
        &[]
    }
    fn label(&self) -> &str {
        "plain"
    }
    fn parse(&self, _inv: &Invocation) -> Result<ParsedRequest, String> {
        Err("not parsed here".into())
    }
}

fn front() -> Plain {
    let gt = checkpoint::gguf::GgufTokenizer {
        model: "gpt2".into(),
        pre: Some("deepseek-v3".into()),
        tokens: (0..8).map(|i| format!("t{i}")).collect(),
        merges: Vec::new(),
        token_types: vec![1; 8],
        bos: Some(0),
        eos: Some(1),
        unk: None,
        pad: None,
        ..Default::default()
    };
    Plain(QwenBpe::from_gguf(&gt).expect("toy tokenizer"))
}

#[test]
fn a_full_card_is_a_classed_backend_failure() {
    let msg = front().rejection(&Invocation::new(), &RejectReason::DeviceOutOfMemory { bytes: 48 << 20 });
    let (class, text) = classify(&msg).expect("the refusal carries a class");
    assert_eq!(class, FailureClass::BackendFailure);
    assert!(text.contains("out of memory"), "{text}");
    let invalid = front().rejection(&Invocation::new(), &RejectReason::ExceedsCapacity { need: 9, capacity: 8 });
    assert!(classify(&invalid).is_none(), "other refusals keep their wording: {invalid}");
}
