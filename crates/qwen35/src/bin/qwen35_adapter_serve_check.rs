// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one large model from a single resident copy of its weights for
// its clients. If your team needs expertise in multi-tenant inference of
// hybrid-attention models then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The released Q8_0 GGUF of Qwen3.8-27B served with an adapter bank, on the
//! real GPU: a check, not a unit test, because it needs the checkpoint and an
//! adapter trained by `brain qwen35 finetune`.
//!
//! * a sequence under an adapter trained by `brain qwen35 finetune` answers
//!   differently from the same prompt on the base, in the same engine;
//! * base and adapter sequences share ONE batched decode step (batch 8), and
//!   the throughput of all-base, mixed and all-adapter batches is printed.
//!
//! Usage: `qwen35_adapter_serve_check <Q8_0.gguf> <adapter.safetensors> [--grouped]`;
//! exits non-zero when the adapter does not change the answer. `--grouped` builds
//! the bank the way a server does, one projection and one epilogue per group of
//! linears that read one activation; without it every linear is corrected on
//! its own.

use std::time::Instant;

use data::tokenizer::Tokenizer;
use model::adapter::bank::{AdapterBankConfig, AdapterServing, AdapterTargets, VerifiedAdapter};
use model::serve::{AdapterId, Request, SampleParams, Scheduler};
use qwen35::int8_gguf_resident::{Qwen35GgufEngine, Qwen35GgufResident};
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

const CAP: u32 = 256;
const BATCHES: [usize; 3] = [8, 16, 32];
const BATCH: usize = 32;
const RESERVE: u64 = 2 << 30;

fn devices() -> Vec<(Device, u64)> {
    qwen35::int8_gguf_resident::budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, RESERVE)
}

/// Greedy tokens per second of `adapters.len()` requests (`None`: base) run as
/// one batch of `BATCH`, prompts of a few tokens so decode dominates.
fn batch_rate(sched: &mut Scheduler<Qwen35GgufEngine>, prompt: &[u32], adapters: &[Option<AdapterId>], max_new: usize) -> f64 {
    let ids: Vec<u64> = adapters
        .iter()
        .map(|a| {
            let req = Request { prompt: prompt.to_vec(), max_new, stop: Vec::new() };
            match a {
                Some(id) => sched.submit_for_adapter(req, *id, SampleParams::greedy(), 0),
                None => sched.submit(req),
            }
        })
        .collect();
    let t = Instant::now();
    let out = sched.run();
    let secs = t.elapsed().as_secs_f64();
    let tokens: usize = ids.iter().map(|id| out[id].len()).sum();
    tokens as f64 / secs
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grouped = args.iter().any(|a| a == "--grouped");
    let (Some(gguf), Some(adapter_path)) = (args.get(1).cloned(), args.get(2).cloned()) else {
        eprintln!("usage: qwen35_adapter_serve_check <Q8_0.gguf> <adapter.safetensors> [--grouped]");
        std::process::exit(2);
    };
    let devices = devices();
    if devices.is_empty() {
        eprintln!("qwen35_adapter_serve_check: no usable GPU");
        std::process::exit(2);
    }
    let tier = qwen35::int8_gguf_resident::held_tiers_at(&gguf).unwrap_or_else(|e| panic!("{e}"));
    let r = Qwen35GgufResident::new(gguf, devices, CAP, tier).with_max_batch(BATCH as u32);
    let key = r.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = r.estimate_multi(&key).devices().collect();
    let (mut engine, tok, _stop) = r.activate_engine(&placed).expect("activate the real checkpoint");

    let bytes = std::fs::read(&adapter_path).expect("read the adapter");
    let (adapter, card) = model::lora::device_adapter::parse_adapter(&bytes, &adapter_path).expect("the trainer's adapter parses and its card agrees with its tensors");
    let targets = AdapterTargets::parse(&card.adapter.as_ref().unwrap().targets.clone().unwrap().join(",")).unwrap();
    let leaves = targets.resolve(engine.instance().config()).expect("the card's targets are sites of the served model");
    let mut bank = AdapterBankConfig::new(2, adapter.rank, &leaves);
    if grouped {
        bank.groups = AdapterServing::input_groups(engine.instance().config()).into_iter().map(|g| g.into_iter().map(str::to_string).collect()).collect();
    }
    engine.enable_adapters(bank).expect("enable the bank");
    println!("  adapter bank ({}): {} MiB over {} leaves, rank {}", if grouped { "grouped" } else { "per linear" }, engine.adapter_bank_bytes() >> 20, leaves.len(), adapter.rank);
    let id = engine.load_adapter(VerifiedAdapter { label: card.id.clone(), digest: "sha256:real".into(), adapter }).expect("register the adapter");

    let prompt = tok.encode("Question: What is the capital of Velmora?\nAnswer:");
    let mut sched = Scheduler::new(engine, BATCH);
    let ask = |sched: &mut Scheduler<Qwen35GgufEngine>, adapter: Option<AdapterId>| {
        let req = Request { prompt: prompt.clone(), max_new: 8, stop: Vec::new() };
        let rid = match adapter {
            Some(id) => sched.submit_for_adapter(req, id, SampleParams::greedy(), 0),
            None => sched.submit(req),
        };
        tok.decode(&sched.run().remove(&rid).unwrap())
    };
    let base = ask(&mut sched, None);
    let adapted = ask(&mut sched, Some(id));
    println!("  base:    {base:?}\n  adapter: {adapted:?}");
    assert_ne!(base, adapted, "the served adapter must change the answer");
    assert_eq!(ask(&mut sched, None), base, "a base request after an adapter one is unchanged");
    println!("  the adapter changes the answer, and a base request after it is the base's");

    // Throughput of each batch size: base only, half adapter, all adapter.
    let short = tok.encode("Question: What is the capital of Velmora?");
    let mix = |batch: usize, n_adapter: usize| -> Vec<Option<AdapterId>> { (0..batch).map(|i| (i < n_adapter).then_some(id)).collect() };
    let _warm = batch_rate(&mut sched, &short, &mix(8, 8), 4);
    for batch in BATCHES {
        for n_adapter in [0, batch / 2, batch] {
            let rate = batch_rate(&mut sched, &short, &mix(batch, n_adapter), 48);
            println!("  batch {batch}: {n_adapter} adapter + {} base rows: {rate:.1} generated tok/s (prefill included)", batch - n_adapter);
        }
    }
}
