// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for measuring whether an adapter as
// trained behaves as the adapter as served, for its clients. If your team needs
// expertise in train-serve agreement of large language models then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Teacher-forced per-token scores from the IN-PROCESS side: the 27B trainer's
//! own forward with an adapter attached and no optimizer step, over the records
//! `emit_held_out_records` wrote.
//!
//! The output is the served bin's own format - one JSONL object per record,
//! `{"record","tokens","positions":[...],"logprob":[...],"argmax":[...]}` - so
//! the two sides of a train-serve comparison are read by one script.
//!
//! ```text
//! cargo run --release -p brain --example score_tokens_qwen35 -- \
//!   --base local:Qwen/Qwen3.8-27B-FP8 --records held-out.jsonl --out inproc-base.jsonl --device gpu0
//! cargo run --release -p brain --example score_tokens_qwen35 -- \
//!   --base local:Qwen/Qwen3.8-27B-FP8 --adapter adapter.brain.safetensors \
//!   --records held-out.jsonl --out inproc-ce8t.jsonl --device gpu0
//! ```
//!
//! `--variant plain|rs` and `--adapter-kind lora|ia3|lokr|dora` must name the
//! tuning the adapter trained with: the card carries its rank and scale, not
//! the variant, and a wrong variant applies the wrong scale silently.

use std::path::PathBuf;

use brain::{score_tokens_qwen35, Device};
use model::adapter::device::AdapterTuning;

fn die(why: &str) -> ! {
    eprintln!("score_tokens_qwen35: {why}");
    std::process::exit(2)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut base, mut adapter, mut records, mut out, mut device) = (None, None, None, None, None);
    let (mut variant, mut kind) = (None, None);
    while let Some(flag) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| die(&format!("{flag} needs a value")));
        match flag.as_str() {
            "--base" => base = Some(value()),
            "--adapter" => adapter = Some(PathBuf::from(value())),
            "--records" => records = Some(PathBuf::from(value())),
            "--out" => out = Some(PathBuf::from(value())),
            "--device" => device = Some(value()),
            "--variant" => variant = Some(value()),
            "--adapter-kind" => kind = Some(value()),
            other => die(&format!("unknown flag {other}")),
        }
    }
    let (Some(base), Some(records), Some(out)) = (base, records, out) else {
        die("usage: score_tokens_qwen35 --base <model> --records <held-out.jsonl> --out <scores.jsonl> [--adapter FILE] [--device D] [--variant plain|rs] [--adapter-kind KIND]")
    };
    let tuning = AdapterTuning::parse(variant.as_deref(), None, kind.as_deref()).unwrap_or_else(|e| die(&e));
    let device = Device::parse(device.as_deref().unwrap_or("gpu0")).unwrap_or_else(|e| die(&e));
    let readout = score_tokens_qwen35(&base, adapter.as_deref(), &records, tuning, &device).unwrap_or_else(|e| die(&e.to_string()));
    let mut lines = String::new();
    for record in &readout.records {
        let positions: Vec<String> = record.positions.iter().map(|s| s.position.to_string()).collect();
        let logprob: Vec<String> = record.positions.iter().map(|s| format!("{}", s.logprob)).collect();
        let argmax: Vec<String> = record.positions.iter().map(|s| s.argmax.to_string()).collect();
        lines.push_str(&format!(
            "{{\"record\":{},\"tokens\":{},\"positions\":[{}],\"logprob\":[{}],\"argmax\":[{}]}}\n",
            record.record,
            record.tokens,
            positions.join(","),
            logprob.join(","),
            argmax.join(",")
        ));
    }
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| die(&format!("{}: {e}", dir.display())));
    }
    std::fs::write(&out, &lines).unwrap_or_else(|e| die(&format!("{}: {e}", out.display())));
    let mean = readout.mean_logprob.unwrap_or(f32::NAN);
    eprintln!(
        "score_tokens_qwen35: {} record(s), {} supervised position(s), mean log-probability {mean:.6}{}",
        readout.records.len(),
        readout.positions,
        if adapter.is_some() { " (adapter)" } else { " (base)" }
    );
}
