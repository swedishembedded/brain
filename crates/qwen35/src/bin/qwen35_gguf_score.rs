// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for measuring the agreement between
// an adapter as trained and the same adapter as served from a quantised
// checkpoint, for its clients. If your team needs expertise in train-serve
// agreement measurement for large language models then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Teacher-forced per-token scores of a record set from the SERVED side: the
//! released Q8_0 GGUF of Qwen3.8-27B, optionally with an adapter in the bank,
//! exactly as `brain serve` runs it.
//!
//! For every supervised position of every record it prints the log-probability
//! the served model gives the recorded token and the token the model itself
//! picks there. A second block of the same size scores the same records from
//! the in-process trainer (`brain::score_tokens_qwen35`); comparing the two is
//! what says whether the adapter that trained is the adapter that serves.
//!
//! Usage: `qwen35_gguf_score <Q8_0.gguf|hf-dir> <records.jsonl> [--adapter FILE]
//!        [--adapter-slots N] [--adapter-rank-cap N] [--cap N] [--kv int8|bf16|f32] [--tier stored|bf16]
//!        [--device D] [--backend B] [--limit N] [--out FILE]`
//!
//! The records are the form [`qwen35::score`] documents: one JSONL object per
//! line, either `{"ids": [...], "mask": [...]}` or a chat record (encoded with
//! the checkpoint's OWN tokenizer and chat template). The output is one JSONL
//! object per record, in input order:
//!
//! ```text
//! {"record":0,"tokens":1024,"positions":[3,4,9],"logprob":[-1.5,...],"argmax":[1234,...]}
//! ```
//!
//! `positions`, `logprob` and `argmax` are parallel arrays over the record's
//! supervised positions. Nothing but these lines goes to stdout, so a caller
//! can pipe them; the summary goes to stderr.

use model::adapter::bank::{AdapterBankConfig, AdapterTargets, VerifiedAdapter};
use model::kv_tier::KvTier;
use model::serve::AdapterId;
use qwen35::int8_gguf_resident::{budgeted_gpus, Qwen35GgufResident};
use qwen35::resident_checkpoint::ResidentCheckpoint;
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

const RESERVE: u64 = 2 << 30;

/// The default adapter bank: two slots of rank 16 or less, correcting every
/// targetable linear - the evaluation server's own configuration
/// (`--adapter-slots 2 --adapter-rank-cap 16 --adapter-targets all`).
const SLOTS: u32 = 2;
const RANK_CAP: u32 = 16;

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(gguf), Some(records_path)) = (args.get(1), args.get(2)) else {
        eprintln!(
            "usage: qwen35_gguf_score <Q8_0.gguf> <records.jsonl> [--adapter FILE] [--adapter-slots N] \
             [--adapter-rank-cap N] [--cap N] [--kv int8|bf16|f32] [--device D] [--backend B] [--limit N] [--out FILE]"
        );
        std::process::exit(2);
    };
    // The checkpoint's tokenizer and chat template, read from the header only.
    let (records, longest) = {
        let ckpt = ResidentCheckpoint::open(gguf).unwrap_or_else(|e| panic!("open the checkpoint: {e}"));
        let (tok, _) = ckpt.tokenizer().unwrap_or_else(|e| panic!("tokenizer: {e}"));
        let tmpl = ckpt.chat_template().unwrap_or_else(|e| panic!("{e}"));
        let mut records = qwen35::score::read_records(std::path::Path::new(records_path), &tok, &tmpl).unwrap_or_else(|e| panic!("{e}"));
        if let Some(limit) = flag(&args, "--limit").map(|v| v.parse::<usize>().expect("--limit is a number")) {
            records.truncate(limit);
        }
        let longest = records.iter().map(|r| r.ids.len()).max().expect("at least one record");
        (records, longest)
    };
    let cap = flag(&args, "--cap").map_or(longest as u32 + 16, |v| v.parse().expect("--cap is a number"));
    assert!(cap as usize >= longest, "--cap {cap} is shorter than the longest record ({longest} tokens)");

    let set = gpu_core::select_compute(flag(&args, "--device").unwrap_or("gpu"), flag(&args, "--backend")).unwrap_or_else(|e| panic!("{e}"));
    let devices = budgeted_gpus(&set.gpus, RESERVE);
    if devices.is_empty() {
        eprintln!("qwen35_gguf_score: no GPU with queryable VRAM - this resident is GPU-only");
        std::process::exit(2);
    }
    let kv = KvTier::parse(flag(&args, "--kv").unwrap_or("int8")).unwrap_or_else(|e| panic!("{e}"));
    // The stored Q8_0 tier, which is what the evaluation server holds its
    // weights at when no `--weights` override is given: a scoring pass must see
    // the arithmetic the served request sees.
    // `--tier bf16` scores the same weights decoded to bf16 instead (an FP8 release's reference arithmetic).
    let tier = match flag(&args, "--tier") {
        None | Some("stored") => qwen35::int8_gguf_resident::held_tiers_at(gguf).unwrap_or_else(|e| panic!("{e}")),
        Some("bf16") => model::ops::TierPolicy::uniform(gpu_core::select::Dtype::BF16),
        Some(other) => panic!("--tier is stored or bf16, not {other:?}"),
    };
    let resident = Qwen35GgufResident::new(gguf.clone(), devices, cap, tier).with_kv_tier(kv);
    let key = resident.instance_key("generate", &capability::Invocation::new());
    let placed: Vec<Device> = resident.estimate_multi(&key).devices().collect();
    let (mut engine, ..) = resident.activate_engine(&placed).expect("activate the checkpoint");

    let adapter: Option<AdapterId> = match flag(&args, "--adapter") {
        None => None,
        Some(path) => {
            let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read the adapter {path}: {e}"));
            let (device_adapter, card) = model::lora::device_adapter::parse_adapter(&bytes, path).expect("the trainer's adapter parses and its card agrees with its tensors");
            let targets = card.adapter.as_ref().and_then(|a| a.targets.clone()).unwrap_or_else(|| vec!["all".to_string()]);
            let leaves = AdapterTargets::parse(&targets.join(",")).and_then(|t| t.resolve(engine.instance().config())).unwrap_or_else(|e| panic!("{path}: {e}"));
            let slots = flag(&args, "--adapter-slots").map_or(SLOTS, |v| v.parse().expect("--adapter-slots is a number"));
            let rank_cap = flag(&args, "--adapter-rank-cap").map_or(RANK_CAP, |v| v.parse().expect("--adapter-rank-cap is a number"));
            engine.enable_adapters(AdapterBankConfig::new(slots, rank_cap, &leaves)).expect("enable the adapter bank");
            let rank = device_adapter.rank;
            // The content digest a served adapter is named by, so two runs over
            // the same file are the same adapter to the bank.
            let digest = brain_modelstore::fetch::file_digest(std::path::Path::new(path)).unwrap_or_else(|e| panic!("digest {path}: {e}"));
            let id = engine.load_adapter(VerifiedAdapter { label: card.id.clone(), digest, adapter: device_adapter }).expect("register the adapter");
            eprintln!("qwen35_gguf_score: adapter {} (rank {rank}) in slot {:?}", card.id, id.kv_namespace());
            Some(id)
        }
    };

    let mut out = String::new();
    let (mut positions, mut sum) = (0usize, 0f64);
    let mut hits = 0usize;
    for (at, record) in records.iter().enumerate() {
        let scores = engine.score_supervised(&record.ids, &record.mask, adapter).unwrap_or_else(|e| panic!("record {}: {e}", at + 1));
        let ids: Vec<String> = scores.iter().map(|s| s.position.to_string()).collect();
        // Full f32 precision (Rust's shortest round-trip form): a comparison
        // between two sides must not be limited by this file's formatting.
        let logprob: Vec<String> = scores.iter().map(|s| format!("{}", s.logprob)).collect();
        let argmax: Vec<String> = scores.iter().map(|s| s.argmax.to_string()).collect();
        out.push_str(&format!(
            "{{\"record\":{at},\"tokens\":{},\"positions\":[{}],\"logprob\":[{}],\"argmax\":[{}]}}\n",
            record.ids.len(),
            ids.join(","),
            logprob.join(","),
            argmax.join(",")
        ));
        positions += scores.len();
        sum += scores.iter().map(|s| s.logprob as f64).sum::<f64>();
        hits += scores.iter().filter(|s| s.argmax == record.ids[s.position as usize]).count();
    }
    match flag(&args, "--out") {
        Some(path) => std::fs::write(path, &out).unwrap_or_else(|e| panic!("write {path}: {e}")),
        None => print!("{out}"),
    }
    eprintln!(
        "qwen35_gguf_score: {} record(s), {positions} supervised position(s), mean log-probability {:.6}, \
         top-1 equals the recorded token at {:.2}%{}",
        records.len(),
        sum / positions.max(1) as f64,
        100.0 * hits as f64 / positions.max(1) as f64,
        if adapter.is_some() { " (adapter)" } else { " (base)" }
    );
}
