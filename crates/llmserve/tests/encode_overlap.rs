// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for low-latency multi-tenant LLM
// serving for its clients. If your team needs expertise in serving that keeps
// every running conversation generating while new inputs are encoded, then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec of background request building: a request whose encoder (a vision
//! tower) runs on its own queue is admitted to the scheduler when its rows
//! exist, while the requests already running keep decoding.
//!
//! The slow request's "encoder" is a thread that waits on a gate the test opens
//! only after the fast request has produced a token. An inline build would
//! block on the gate before the fast request ever decoded, so the gate's own
//! safety timeout (not the decode) would open it; the test asserts which.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use capability::{Invocation, Progress};
use data::qwen_chat::TemplateFlavor;
use data::qwen_tokenizer::QwenBpe;
use llmserve::{run_batch_scheduled, Encoding, Frontend, ParsedRequest};
use model::paged::BlockAllocator;
use model::serve::{SampleParams, Scheduler};
use qwen3::chat::ToolChoice;

mod common;
use common::{tokenizer, Toy, VOCAB};

fn request(seed: u32, max_new: usize) -> ParsedRequest {
    ParsedRequest {
        ids: (0..4).map(|i| (seed + i) % VOCAB as u32).collect(),
        max_new,
        temp: 0.0,
        top_k: 0,
        top_p: 1.0,
        sampler: SampleParams::greedy(),
        // seed: the request samples greedily, so its seed is never consulted
        seed: 0,
        stops: Vec::new(),
        tool_choice: ToolChoice::Auto,
        flavor: TemplateFlavor::default(),
        thinking_open: false,
        raw: true,
    }
}

/// `slow` requests are built by a thread that waits for the gate; the others inline.
struct Gated {
    tok: QwenBpe,
    gate: Arc<AtomicBool>,
}

impl Frontend for Gated {
    fn tok(&self) -> &QwenBpe {
        &self.tok
    }
    fn stop(&self) -> &[u32] {
        &[]
    }
    fn label(&self) -> &str {
        "gated"
    }
    fn parse(&self, inv: &Invocation) -> Result<ParsedRequest, String> {
        // seed: the test double samples greedily, so the seed it passes on is never consulted
        Ok(request(inv.get_i64("seed").unwrap_or(0) as u32, inv.get_i64("max_new").unwrap_or(8) as usize))
    }
    fn parse_in_background(&self, inv: &Invocation) -> Option<Encoding> {
        inv.get_bool("slow")?;
        let (tx, rx) = mpsc::channel();
        let (gate, req) = (self.gate.clone(), self.parse(inv).map(|r| (r, None)));
        std::thread::spawn(move || {
            // The safety timeout is what an inline build would run into: it would wait here before decoding anything.
            for _ in 0..500 {
                if gate.load(Ordering::SeqCst) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            let _ = tx.send(req);
        });
        Some(rx)
    }
}

#[test]
fn a_request_built_in_the_background_joins_while_the_others_keep_decoding() {
    let gate = Arc::new(AtomicBool::new(false));
    let front = Gated { tok: tokenizer(), gate: gate.clone() };
    let mut sched = Scheduler::new(Toy::new(BlockAllocator::new(512, 8), 0), 8);
    let invs = [
        Invocation::new().set("slow", serde_json::json!(true)).set("seed", serde_json::json!(10)).set("max_new", serde_json::json!(5)),
        Invocation::new().set("seed", serde_json::json!(20)).set("max_new", serde_json::json!(30)),
    ];

    let opened_by_decode = Arc::new(AtomicBool::new(false));
    let (flag, g) = (opened_by_decode.clone(), gate.clone());
    let results = run_batch_scheduled(&mut sched, &front, &invs, &mut |i, p| {
        // The fast request (index 1) streams a token: only now does the slow request's encoder finish.
        if i == 1 && matches!(p, Progress { .. }) && !g.swap(true, Ordering::SeqCst) {
            flag.store(true, Ordering::SeqCst);
        }
    });

    assert!(opened_by_decode.load(Ordering::SeqCst), "the fast request did not stream before the slow one's encoder was released");
    assert_eq!(results.len(), 2, "one result per invocation, in order");
    let tokens = |r: &capability::ActionResult| r.as_ref().expect("served").outputs["completion_tokens"].as_u64().unwrap();
    assert_eq!((tokens(&results[0]), tokens(&results[1])), (5, 30));
}
