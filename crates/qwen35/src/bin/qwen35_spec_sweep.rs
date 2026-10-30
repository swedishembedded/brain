// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding - drafters, verifiers
// and the state rollback a recurrent stack needs - for clients serving large
// models on modest hardware. If your team needs expertise in measuring
// speculative decoding honestly under continuous batching, then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Speculative decoding of the REAL Qwen3.8-27B (released GGUF, INT8) through
//! the generic continuous-batching scheduler: aggregate decode throughput and
//! the acceptance a drafter reaches, per batch size and workload.
//!
//! It drives `model::serve::Scheduler<Qwen35GgufEngine>` exactly as the server
//! does (the same chat template, the same engine, the same scheduler), each
//! request running until its stop token or `--new` tokens. Decode throughput is
//! tokens generated over the time from the moment every request has been
//! admitted to the moment the last one finishes, so the prompt prefill is not in
//! it; sequences that stop early shrink the batch, as they do in service.
//!
//! One process measures one `--speculative` mode (the mode sizes the resident);
//! run it once per mode. `--dump FILE` writes every request's generated token
//! ids, so two modes can be compared token for token.
//!
//! Usage:
//!   qwen35_spec_sweep <Qwen3.8-27B*.gguf> --speculative off|auto|mtp|ngram
//!       [--ctx 4096] [--kv bf16] [--batches 1,8,16,32,64,128]
//!       [--workload code|agent|both|random] [--new 128] [--dump FILE]
//!       [--batch-limit N] [--raw-hidden] [--equivalence] [--temp T]
//!       [--prefill-slice N] [--prefill-batch N] [--mix LEN:NEW,LEN:NEW,...] [--backend cuda|vulkan|wgpu]
//!
//! `--equivalence` replaces the timing: for every workload and batch size it
//! generates the same requests twice on the same scheduler, once with
//! speculation switched off and once on, and exits non-zero if any sequence's
//! tokens differ, naming the position of the first difference.
//!
//! Run it under `scripts/lease/gpu --lock ... --`.

use std::io::Write;
use std::time::Instant;

use data::tokenizer::Tokenizer;
use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::{PagedDecoder, Request, SampleParams, Scheduler, SpecPolicy};
use qwen35::int8_gguf_resident::{PrefillPolicy, Qwen35GgufResident, SpecServing, SpeculativeMode};
use residency::multi::MultiDeviceResidentModel;
use residency::{Device, ResidentModel};

const REAL_PROMPT: &str = include_str!("../../tests/fixtures/real_prompt.txt");

const RESERVE: u64 = 2 << 30;

/// Prompt length of the `random` workload.
const RANDOM_PROMPT_TOKENS: usize = 1024;

struct Args {
    gguf: String,
    ctx: u32,
    kv: KvTier,
    mode: SpeculativeMode,
    batches: Vec<u32>,
    workloads: Vec<&'static str>,
    new_tokens: usize,
    dump: Option<String>,
    batch_limit: usize,
    normed_hidden: bool,
    scratch_rows: u32,
    equivalence: bool,
    profile_verify: bool,
    kernel_table: bool,
    temp: f32,
    prefill_tokens: Vec<usize>,
    /// Prompt tokens one sequence gets per scheduler iteration (default: the whole prompt).
    prefill_slice: u32,
    /// Rows one prefill forward carries across sequences (default: the engine's; 0 = one sequence per forward).
    prefill_batch: Option<usize>,
    mixed: bool,
    /// `--mix LEN:NEW,...`: one request per entry (LEN prompt tokens, NEW new tokens), all submitted at once.
    mix: Vec<(usize, usize)>,
    /// Greedy-decode this many long prompts (`--identity-prompts N,P`: N prompts of about P tokens)
    /// and write their tokens to `--dump`, to compare sliced against whole-prompt prefill.
    identity: Option<(usize, usize)>,
    /// `--backend`: the compute backend to run on (default: the process default).
    backend: Option<String>,
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        gguf: String::new(),
        ctx: 4096,
        kv: KvTier::BF16,
        mode: SpeculativeMode::Off,
        batches: vec![1, 8, 16, 32, 64, 128],
        workloads: vec!["code", "agent"],
        new_tokens: 128,
        dump: None,
        batch_limit: usize::MAX,
        normed_hidden: true,
        scratch_rows: 32,
        equivalence: false,
        profile_verify: false,
        kernel_table: false,
        temp: 0.0,
        prefill_tokens: Vec::new(),
        prefill_slice: u32::MAX,
        prefill_batch: None,
        mixed: false,
        mix: Vec::new(),
        identity: None,
        backend: None,
    };
    while let Some(flag) = it.next() {
        let mut value = |name: &str| it.next().unwrap_or_else(|| panic!("{name} needs a value"));
        match flag.as_str() {
            "--ctx" => a.ctx = value("--ctx").parse().expect("--ctx N"),
            "--kv" => a.kv = KvTier::parse(&value("--kv")).unwrap_or_else(|e| panic!("{e}")),
            "--speculative" => a.mode = SpeculativeMode::parse(&value("--speculative")).unwrap_or_else(|e| panic!("{e}")),
            "--batches" => a.batches = value("--batches").split(',').map(|s| s.trim().parse().expect("--batches a,b,c")).collect(),
            "--workload" => {
                a.workloads = match value("--workload").as_str() {
                    "code" => vec!["code"],
                    "agent" => vec!["agent"],
                    "both" => vec!["code", "agent"],
                    "random" => vec!["random"],
                    other => panic!("--workload code|agent|both|random, got {other}"),
                }
            }
            "--new" => a.new_tokens = value("--new").parse().expect("--new N"),
            "--dump" => a.dump = Some(value("--dump")),
            "--batch-limit" => a.batch_limit = value("--batch-limit").parse().expect("--batch-limit N"),
            "--raw-hidden" => a.normed_hidden = false,
            "--equivalence" => a.equivalence = true,
            "--profile-verify" => a.profile_verify = true,
            "--kernel-table" => a.kernel_table = true,
            "--backend" => a.backend = Some(value("--backend")),
            "--mixed" => a.mixed = true,
            "--mix" => {
                a.mix = value("--mix")
                    .split(',')
                    .map(|e| {
                        let (len, new) = e.split_once(':').expect("--mix LEN:NEW,LEN:NEW");
                        (len.trim().parse().expect("--mix LEN:NEW"), new.trim().parse().expect("--mix LEN:NEW"))
                    })
                    .collect()
            }
            "--identity-prompts" => {
                let v = value("--identity-prompts");
                let (n, p) = v.split_once(',').expect("--identity-prompts N,P");
                a.identity = Some((n.parse().expect("--identity-prompts N,P"), p.parse().expect("--identity-prompts N,P")));
            }
            "--prefill-batch" => a.prefill_batch = Some(value("--prefill-batch").parse().expect("--prefill-batch N")),
            "--prefill-slice" => a.prefill_slice = value("--prefill-slice").parse().expect("--prefill-slice N"),
            "--prefill-tokens" => a.prefill_tokens = value("--prefill-tokens").split(',').map(|s| s.trim().parse().expect("--prefill-tokens a,b,c")).collect(),
            "--temp" => a.temp = value("--temp").parse().expect("--temp T"),
            "--scratch-rows" => a.scratch_rows = value("--scratch-rows").parse().expect("--scratch-rows N"),
            other if !other.starts_with("--") && a.gguf.is_empty() => a.gguf = other.to_string(),
            other => panic!("unknown argument {other}"),
        }
    }
    if a.gguf.is_empty() {
        eprintln!("usage: qwen35_spec_sweep <Qwen3.8-27B*.gguf> --speculative off|auto|mtp|ngram [--ctx N] [--kv f32|bf16|int8|fp8] [--batches a,b,c] [--workload code|agent|both|random] [--new N] [--dump FILE]");
        std::process::exit(2);
    }
    a
}

const CODE_TASKS: [&str; 16] = [
    "Write a Rust function that merges two sorted Vec<i64> into one sorted Vec<i64> without using sort. Include a short doc comment and two unit tests.",
    "Write a Python class LRUCache with get and put in O(1), using only the standard library. Include type hints.",
    "Implement binary search in C that returns the index of the first element not less than the key in a sorted int array, and explain the loop invariant in a comment.",
    "Write a TypeScript function debounce(fn, ms) that returns a debounced function with a cancel method. Include types.",
    "Write a Rust struct RingBuffer<T> with push, pop and len, backed by a fixed-size array, with unit tests.",
    "Write a Python function that parses an ISO 8601 date string without using datetime.fromisoformat and returns (year, month, day), raising ValueError on bad input.",
    "Write a Go function that reads a file line by line and returns the number of lines containing a given substring, handling errors idiomatically.",
    "Write a SQL query and a short Python script using sqlite3 that computes the top 3 customers by total order value from tables customers(id, name) and orders(id, customer_id, amount).",
    "Write a C++ template function that returns the k-th smallest element of a vector using nth_element, with a main that demonstrates it.",
    "Write a bash script that watches a directory and prints the name of each new file that appears, polling once per second.",
    "Write a Rust function that tokenizes arithmetic expressions with + - * / and parentheses into an enum Token, with tests.",
    "Write a Python generator that yields all permutations of a list without using itertools, and a pytest test for it.",
    "Write a JavaScript function that deep-clones objects containing Date, Map, Set and circular references.",
    "Implement a simple thread pool in Rust with std only: new(n), execute(closure), and Drop that joins the workers.",
    "Write a Python dataclass Matrix with __matmul__ and transpose for lists of lists of floats, with a doctest.",
    "Write a C function that reverses a singly linked list in place and a main that builds, prints, reverses and frees a list.",
];

const AGENT_TASKS: [&str; 16] = [
    "What's the weather in Stockholm right now, and should I bring an umbrella?",
    "Find the file that defines the scheduler in this repository and tell me how many lines it has.",
    "Search for the latest release notes of the project and summarize the breaking changes.",
    "Read src/main.rs, then run the tests and report any failures.",
    "Look up the population of Gothenburg and Malmo and tell me the ratio.",
    "List the files changed in the last commit and show me the diff of the largest one.",
    "Check the weather in Paris and in Berlin and tell me which is warmer.",
    "Open config/settings.toml and change the log level to debug, then show me the result.",
    "Search the codebase for TODO comments and group them by file.",
    "Run the linter on the whole project and fix whatever it can fix automatically.",
    "What is the current time in Tokyo and in New York?",
    "Fetch the README of the repository, then list the sections it contains.",
    "Find every place where the function parse_request is called and summarize the call sites.",
    "Read the file build.log and tell me why the build failed.",
    "Look up the exchange rate from SEK to EUR and convert 2500 SEK.",
    "Create a new file notes.md containing a bullet list of today's tasks: write docs, review PRs, run benchmarks.",
];

const TOOLS: &str = r#"[
{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}},
{"type":"function","function":{"name":"read_file","description":"Read a text file from the workspace and return its contents","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path relative to the workspace root"}},"required":["path"]}}},
{"type":"function","function":{"name":"write_file","description":"Write text to a file in the workspace, replacing it","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
{"type":"function","function":{"name":"run_command","description":"Run a shell command in the workspace and return its output","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
{"type":"function","function":{"name":"web_search","description":"Search the web and return the top results","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}
]"#;

/// The rendered, tokenized prompt of request `i` of `workload`.
fn prompt_ids(tok: &data::qwen_tokenizer::QwenBpe, workload: &str, i: usize) -> Vec<u32> {
    if workload == "random" {
        // The serving benchmark's prompts: RANDOM_PROMPT_TOKENS uniformly drawn ordinary tokens decoded to text, which the
        // chat template then wraps like any user message (thinking on, the template's default).
        // seed: a fixed stream per request index, so two runs of the sweep compare token for token
        let mut rng = data::rng::Lcg::new(0x5eed + i as u64);
        let ids: Vec<u32> = (0..RANDOM_PROMPT_TOKENS).map(|_| 1000 + (rng.unit() * 100_000.0) as u32).collect();
        let messages = serde_json::json!([{ "role": "user", "content": tok.decode(&ids) }]).to_string();
        let inv = qwen35::caps::with_template_flavor_default(&capability::Invocation::new().set("messages", serde_json::json!(messages)));
        return qwen3::chat::parse_request(tok, &inv).expect("render the chat prompt").ids;
    }
    let (tasks, tools): (&[&str; 16], bool) = if workload == "code" { (&CODE_TASKS, false) } else { (&AGENT_TASKS, true) };
    // Distinct requests past 16: the same task, a different variant tag.
    let text = if i < 16 { tasks[i % 16].to_string() } else { format!("{} (variant {})", tasks[i % 16], i / 16) };
    let messages = serde_json::json!([{ "role": "user", "content": text }]).to_string();
    let mut inv = capability::Invocation::new().set("messages", serde_json::json!(messages)).set("enable_thinking", serde_json::json!(false));
    if tools {
        inv = inv.set("tools", serde_json::json!(TOOLS)).set("system", serde_json::json!("You are a helpful assistant with access to tools. Call a tool whenever it helps, using the tool-call format."));
    }
    let inv = qwen35::caps::with_template_flavor_default(&inv);
    qwen3::chat::parse_request(tok, &inv).expect("render the chat prompt").ids
}

fn main() {
    let a = parse_args();
    if let Some(backend) = &a.backend {
        gpu_core::select_compute("gpu", Some(backend)).unwrap_or_else(|e| panic!("{e}"));
    }
    let max_batch = *a.batches.iter().max().expect("a batch size");
    let devices = qwen35::int8_gguf_resident::budgeted_gpus(&gpu_core::devices::ambient_compute_set().gpus, RESERVE);
    if devices.is_empty() {
        eprintln!("no GPU with queryable VRAM - this resident is GPU-only");
        std::process::exit(2);
    }
    let tier = qwen35::int8_gguf_resident::held_tiers_at(&a.gguf).unwrap_or_else(|e| panic!("{e}"));
    let spec = SpecServing { mode: a.mode, mtp_normed_hidden: a.normed_hidden, scratch_rows: a.scratch_rows, batch_limit: a.batch_limit };
    // Without --prefill-slice a prompt goes in whole (and, the default being one sequence at a time, alone).
    let sliced = a.prefill_slice != u32::MAX;
    let batch_tokens = a.prefill_batch.unwrap_or(PrefillPolicy::default().batch_tokens);
    let policy = PrefillPolicy { slice_tokens: if sliced { a.prefill_slice as usize } else { usize::MAX }, batch_tokens, ..PrefillPolicy::default() };
    let r = Qwen35GgufResident::new(a.gguf.clone(), devices, a.ctx, tier).with_kv_tier(a.kv).with_max_batch(max_batch).with_speculation(spec).with_prefill_policy(policy);
    let cost = r.estimate_multi(&r.instance_key("generate", &capability::Invocation::new()));
    let placed: Vec<Device> = cost.devices().collect();
    if placed.is_empty() {
        eprintln!("ctx {} x max_batch {max_batch} does not fit (see the placement message above)", a.ctx);
        std::process::exit(1);
    }
    println!("qwen35 spec sweep: mode {}, ctx {}, kv {}, {} new tokens, max_batch {max_batch}", a.mode.as_str(), a.ctx, a.kv.as_str(), a.new_tokens);
    let t0 = Instant::now();
    let (engine, tok, stop) = r.activate_engine(&placed).expect("activate the real checkpoint");
    println!("cold load {:.1} s; MTP head loaded: {}; verify rows {}", t0.elapsed().as_secs_f64(), engine.instance().has_mtp(), engine.instance().speculation_rows());
    let mut sched = Scheduler::new(engine, max_batch as usize);
    r.configure_scheduler(&mut sched);

    if !a.mix.is_empty() {
        // The agent mix through the scheduler without the HTTP layer: every request arrives at
        // once, prompts are distinct rotations of one passage (no shared prefix). Reports the
        // aggregate prompt tokens per second and where the wall time went.
        let mut base = tok.encode(REAL_PROMPT);
        while base.len() < 4096 {
            let more = base.clone();
            base.extend(more);
        }
        let total: usize = a.mix.iter().map(|m| m.0).sum();
        let before = sched.decoder_mut().spec_times().prefill;
        let t = Instant::now();
        let ids: Vec<u64> = a
            .mix
            .iter()
            .enumerate()
            .map(|(i, &(len, new))| {
                let prompt: Vec<u32> = (0..len).map(|j| base[(i * 937 + j) % base.len()]).collect();
                sched.submit(Request { prompt, max_new: new, stop: Vec::new() })
            })
            .collect();
        let mut finish: Vec<f64> = Vec::new();
        let mut generated = 0usize;
        while sched.pending() {
            for (id, toks) in sched.step() {
                assert!(ids.contains(&id));
                generated += toks.len();
                finish.push(t.elapsed().as_secs_f64());
            }
        }
        let wall = t.elapsed().as_secs_f64();
        let prefill = (sched.decoder_mut().spec_times().prefill - before).as_secs_f64();
        println!("mix: {} requests, {total} prompt tokens, {generated} generated; slice {}, batch rows {}", a.mix.len(), if a.prefill_slice == u32::MAX { "whole prompt".to_string() } else { a.prefill_slice.to_string() }, policy.batch_tokens);
        println!("mix: wall {wall:.2} s = {:.0} prompt tok/s aggregate; inside prefill_chunk {prefill:.2} s = {:.0} tok/s; elsewhere (decode, admission) {:.2} s", total as f64 / wall, total as f64 / prefill, wall - prefill);
        println!("mix: finish times (s): {}", finish.iter().map(|f| format!("{f:.1}")).collect::<Vec<_>>().join(" "));
        return;
    }

    if a.mixed {
        // The agent mix that exposed static waves: 16 requests with ~2k-token prompts arrive
        // together, one of them a runaway of 400 new tokens and fifteen of them 48-token
        // answers. Print when each finished.
        let mut base = tok.encode(REAL_PROMPT);
        while base.len() < 4096 {
            let more = base.clone();
            base.extend(more);
        }
        let t = Instant::now();
        let mut ids = Vec::new();
        for i in 0..16usize {
            let prompt: Vec<u32> = (0..2048).map(|j| base[(i * 37 + j) % base.len()]).collect();
            let max_new = if i == 0 { 400 } else { 48 };
            ids.push((sched.submit(Request { prompt, max_new, stop: Vec::new() }), max_new));
        }
        let mut at: Vec<(f64, usize)> = Vec::new();
        while sched.pending() {
            for (id, toks) in sched.step() {
                let want = ids.iter().find(|x| x.0 == id).expect("known id").1;
                at.push((t.elapsed().as_secs_f64(), want));
                assert_eq!(toks.len(), want);
            }
        }
        println!("prefill slice {} tokens: finish times (s) of the 48-token answers: {}", if a.prefill_slice == u32::MAX { "whole prompt".to_string() } else { a.prefill_slice.to_string() }, at.iter().filter(|x| x.1 == 48).map(|x| format!("{:.1}", x.0)).collect::<Vec<_>>().join(" "));
        println!("                the 400-token runaway: {:.1} s", at.iter().find(|x| x.1 == 400).map_or(0.0, |x| x.0));
        return;
    }

    if let Some((n, p)) = a.identity {
        // N distinct long prompts (rotations of the passage, each a different length so the last
        // slice of each ends at a different offset), decoded greedily alongside one another.
        let mut base = tok.encode(REAL_PROMPT);
        while base.len() < 4096 {
            let more = base.clone();
            base.extend(more);
        }
        let ids: Vec<u64> = (0..n)
            .map(|i| {
                let prompt: Vec<u32> = (0..p + i * 131).map(|j| base[(i * 37 + j) % base.len()]).collect();
                assert!(prompt.len() + a.new_tokens + 8 <= a.ctx as usize, "prompt of {} tokens does not fit --ctx", prompt.len());
                sched.submit(Request { prompt, max_new: a.new_tokens, stop: stop.clone() })
            })
            .collect();
        let mut done = sched.run();
        let mut dump = a.dump.as_ref().map(|path| std::fs::File::create(path).expect("create --dump file"));
        for (i, id) in ids.iter().enumerate() {
            let tokens = done.remove(id).expect("every request finishes");
            println!("prompt {i}: {} tokens, first generated {:?}", p + i * 131, &tokens[..tokens.len().min(8)]);
            if let Some(f) = dump.as_mut() {
                writeln!(f, "{}", serde_json::json!({ "workload": "long", "batch": n, "request": i, "tokens": tokens })).expect("write --dump");
            }
        }
        return;
    }

    if !a.prefill_tokens.is_empty() {
        // Prompt-processing throughput through the scheduler's admission, as the server
        // does it: `b` requests of `p` prompt tokens each and one new token, timed from
        // submission to the last first token.
        let mut base = tok.encode(REAL_PROMPT);
        while base.len() < 4096 {
            let more = base.clone();
            base.extend(more);
        }
        println!("{:>6} {:>7} {:>10} {:>10}", "batch", "prompt", "wall s", "prefill tok/s");
        for &p in &a.prefill_tokens {
            for &b in &a.batches {
                let t = Instant::now();
                for i in 0..b as usize {
                    // Distinct prompts of the same length: a rotation of the passage, so no prefix is shared.
                    let prompt: Vec<u32> = (0..p).map(|j| base[(i * 37 + j) % base.len()]).collect();
                    sched.submit(Request { prompt, max_new: 1, stop: Vec::new() });
                }
                sched.run();
                let wall = t.elapsed().as_secs_f64();
                println!("{:>6} {:>7} {:>10.2} {:>10.0}", b, p, wall, (b as usize * p) as f64 / wall);
            }
        }
        return;
    }

    if a.profile_verify {
        // The cost of one verify step against one plain step, per row count and batch size,
        // on a decoder with nothing else going on: prefill `b` sequences, then time the
        // steps directly (each verify is committed keeping only its first row, so every
        // step starts from the same state).
        let ids = prompt_ids(&tok, "code", 0);
        println!("{:>5} {:>6} {:>10} {:>10}", "batch", "rows", "verify ms", "plain ms");
        for &b in &a.batches {
            let engine = sched.decoder_mut();
            let mut tables: Vec<BlockTable> = (0..b).map(|_| BlockTable::new()).collect();
            let mut inputs = Vec::new();
            for t in tables.iter_mut() {
                let h = engine.prefill(t, &ids);
                inputs.push(engine.admit_greedy(&h));
            }
            let time = |f: &mut dyn FnMut()| {
                f();
                let t = Instant::now();
                for _ in 0..8 {
                    f();
                }
                t.elapsed().as_secs_f64() * 1e3 / 8.0
            };
            let plain = {
                let before: Vec<u32> = tables.iter().map(BlockTable::len).collect();
                let ms = time(&mut || {
                    let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
                    engine.forward_batched_greedy(&mut refs, &inputs);
                    for (t, &n) in tables.iter_mut().zip(&before) {
                        t.truncate(n, engine.alloc_mut());
                    }
                });
                ms
            };
            for rows in [1usize, 2, 3, 4, 5, 6, 8] {
                if rows * b as usize > engine.spec_row_budget() {
                    continue;
                }
                let before: Vec<u32> = tables.iter().map(BlockTable::len).collect();
                let rows_tokens: Vec<Vec<u32>> = inputs.iter().map(|&t| vec![t; rows]).collect();
                let ms = time(&mut || {
                    let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
                    engine.forward_batched_verify(&mut refs, &rows_tokens, 1);
                    engine.commit_verified(&mut refs, &vec![1; b as usize]);
                    for (t, &n) in tables.iter_mut().zip(&before) {
                        t.truncate(n, engine.alloc_mut());
                    }
                });
                println!("{:>5} {:>6} {:>10.1} {:>10.1}", b, rows, ms, plain);
                if a.kernel_table && [1usize, 4, 8].contains(&rows) {
                    // Per-kernel device time of the verify step, to see what grows with the rows.
                    let gpu = engine.instance().stage_gpu(0);
                    gpu.set_kernel_timing(true);
                    gpu.reset_kernel_times();
                    for _ in 0..4 {
                        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
                        engine.forward_batched_verify(&mut refs, &rows_tokens, 1);
                        engine.commit_verified(&mut refs, &vec![1; b as usize]);
                        for (t, &n) in tables.iter_mut().zip(&before) {
                            t.truncate(n, engine.alloc_mut());
                        }
                    }
                    let mut times = engine.instance().stage_gpu(0).kernel_times().unwrap_or_default();
                    engine.instance().stage_gpu(0).set_kernel_timing(false);
                    times.sort_by(|x, y| y.1.partial_cmp(&x.1).expect("finite times"));
                    for (name, ms, calls) in times.iter().take(10) {
                        println!("          {:<48} {:>8.3} ms/step {:>6} calls/step", name, ms / 4.0, calls / 4);
                    }
                }
            }
            for t in tables.iter_mut() {
                engine.release_table(t);
            }
        }
        return;
    }

    if a.equivalence {
        // Three comparisons per workload and batch size. The noise floor first: plain decode
        // of a batch against the same requests decoded ALONE, because the engine's attention
        // splits its key range by batch size, so plain decode is itself not batch-invariant to
        // the last bit. Then speculation against plain decode of the same batch, and against
        // each request decoded alone.
        let first_diff = |p: &[u32], s: &[u32]| p.iter().zip(s).position(|(x, y)| x != y).or((p.len() != s.len()).then_some(p.len().min(s.len())));
        let (mut floor_bad, mut batch_bad, mut alone_bad, mut solo_bad, mut total) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for workload in &a.workloads {
            for &b in &a.batches {
                let prompts: Vec<Vec<u32>> = (0..b as usize).map(|i| prompt_ids(&tok, workload, i)).collect();
                let generate = |sched: &mut Scheduler<qwen35::int8_gguf_resident::Qwen35GgufEngine>, which: &[Vec<u32>]| -> Vec<Vec<u32>> {
                    let ids: Vec<u64> = which.iter().map(|p| sched.submit(Request { prompt: p.clone(), max_new: a.new_tokens, stop: stop.clone() })).collect();
                    let mut done = sched.run();
                    ids.iter().map(|id| done.remove(id).expect("every request finishes")).collect()
                };
                sched.set_speculation(SpecPolicy { enabled: false, ..SpecPolicy::default() });
                let plain = generate(&mut sched, &prompts);
                let alone: Vec<Vec<u32>> = prompts.iter().map(|p| generate(&mut sched, std::slice::from_ref(p)).remove(0)).collect();
                sched.set_speculation(SpecPolicy::default());
                let before = sched.spec_stats();
                let spec_alone: Vec<Vec<u32>> = prompts.iter().map(|p| generate(&mut sched, std::slice::from_ref(p)).remove(0)).collect();
                let spec = generate(&mut sched, &prompts);
                let stats = sched.spec_stats();
                let (mut floor, mut vs_batch, mut vs_alone) = (0, 0, 0);
                for i in 0..plain.len() {
                    total += 1;
                    if let Some(at) = first_diff(&plain[i], &alone[i]) {
                        floor += 1;
                        println!("  noise floor: plain batch {b} request {i} differs from alone at token {at}");
                    }
                    if let Some(at) = first_diff(&plain[i], &spec[i]) {
                        vs_batch += 1;
                        println!("  DIVERGES {workload} batch {b} request {i} from plain at token {at} (plain {:?}.. speculative {:?}.., {} / {} tokens)", plain[i].get(at), spec[i].get(at), plain[i].len(), spec[i].len());
                    }
                    if first_diff(&alone[i], &spec[i]).is_some() {
                        vs_alone += 1;
                    }
                    if let Some(at) = first_diff(&alone[i], &spec_alone[i]) {
                        solo_bad += 1;
                        println!("  DIVERGES {workload} request {i} decoded alone, speculative from plain, at token {at}");
                    }
                }
                floor_bad += floor;
                batch_bad += vs_batch;
                alone_bad += vs_alone;
                println!(
                    "{workload:<6} batch {b:>3}: of {} requests, plain-vs-alone differ {floor}, speculative-vs-plain differ {vs_batch}, speculative-vs-alone differ {vs_alone}; {} verify steps, {:.2} tokens per sequence-round",
                    plain.len(),
                    stats.verify_steps - before.verify_steps,
                    stats.tokens_per_round()
                );
            }
        }
        println!("equivalence over {total} requests: plain-vs-alone {floor_bad} differ (the numerical floor), speculative-vs-plain {batch_bad}, speculative-vs-alone {alone_bad}; each request decoded alone, speculative-vs-plain {solo_bad}");
        std::process::exit(if solo_bad == 0 { 0 } else { 1 });
    }

    let seed_base = data::rng::random_seed();
    if a.temp > 0.0 {
        println!("sampling at temperature {} with seed {seed_base} (drawn at random per run)", a.temp);
    }
    let mut dump = a.dump.as_ref().map(|p| std::fs::File::create(p).expect("create --dump file"));
    println!();
    println!("{:<6} {:>5} {:>10} {:>10} {:>11} {:>12} {:>12} {:>9}", "load", "batch", "agg tok/s", "per seq", "tok/round", "accept/prop", "verify/plain", "mtp fwd");
    for workload in &a.workloads {
        for &b in &a.batches {
            let ids: Vec<u64> = (0..b as usize)
                .map(|i| {
                    let prompt = prompt_ids(&tok, workload, i);
                    assert!(prompt.len() + a.new_tokens + 8 <= a.ctx as usize, "prompt of {} tokens does not fit --ctx", prompt.len());
                    sched.submit_sampled(Request { prompt, max_new: a.new_tokens, stop: if workload == &"random" { Vec::new() } else { stop.clone() } }, SampleParams { temp: a.temp, top_k: 20, top_p: 0.95, ..SampleParams::greedy() }, seed_base.wrapping_add(i as u64))
                })
                .collect();
            // Admit everything (prefill), then time the decode alone (the `random` workload times the whole request).
            // The `random` workload is the serving benchmark's shape: prompts and decode share the wall clock.
            while *workload != "random" && sched.running_len() < b as usize {
                sched.step();
            }
            let before: usize = ids.iter().map(|&id| sched.tokens_of(id).map_or(0, <[u32]>::len)).sum();
            let stats0 = sched.spec_stats();
            let fwd0 = sched.decoder_mut().instance().mtp_forwards();
            let times0 = sched.decoder_mut().spec_times();
            let mtp0 = sched.decoder_mut().instance().mtp_times();
            let t = Instant::now();
            let mut done: std::collections::HashMap<u64, Vec<u32>> = std::collections::HashMap::new();
            let mut tokens_done = 0usize;
            while sched.pending() {
                for (id, toks) in sched.step() {
                    tokens_done += toks.len();
                    done.insert(id, toks);
                }
            }
            let wall = t.elapsed().as_secs_f64();
            let generated = tokens_done - before;
            let stats = sched.spec_stats();
            let rounds = stats.rounds - stats0.rounds;
            let tok_per_round = if rounds > 0 { ((stats.accepted - stats0.accepted) + rounds) as f64 / rounds as f64 } else { 1.0 };
            let accept = if stats.proposed > stats0.proposed { (stats.accepted - stats0.accepted) as f64 / (stats.proposed - stats0.proposed) as f64 } else { 0.0 };
            println!(
                "{:<6} {:>5} {:>10.1} {:>10.1} {:>11.2} {:>12.2} {:>6}/{:<5} {:>9}",
                workload,
                b,
                generated as f64 / wall,
                generated as f64 / wall / b as f64,
                tok_per_round,
                accept,
                stats.verify_steps - stats0.verify_steps,
                stats.plain_steps - stats0.plain_steps,
                sched.decoder_mut().instance().mtp_forwards() - fwd0
            );
            println!("         steps the cost gate sent to plain decoding: {}", stats.gated_steps - stats0.gated_steps);
            let times = sched.decoder_mut().spec_times();
            println!(
                "         time in draft {:.0} ms, verify {:.0} ms, commit {:.0} ms of {:.0} ms",
                (times.draft - times0.draft).as_secs_f64() * 1e3,
                (times.verify - times0.verify).as_secs_f64() * 1e3,
                (times.commit - times0.commit).as_secs_f64() * 1e3,
                wall * 1e3
            );
            let mtp = sched.decoder_mut().instance().mtp_times();
            if mtp[0] > mtp0[0] {
                println!(
                    "         MTP head: prep+projection {:.0} ms, layer {:.0} ms, shared head {:.0} ms",
                    (mtp[0] - mtp0[0]).as_secs_f64() * 1e3,
                    (mtp[1] - mtp0[1]).as_secs_f64() * 1e3,
                    (mtp[2] - mtp0[2]).as_secs_f64() * 1e3
                );
            }
            if let Some(f) = dump.as_mut() {
                for (i, id) in ids.iter().enumerate() {
                    writeln!(f, "{}", serde_json::json!({ "workload": workload, "batch": b, "request": i, "tokens": done[id] })).expect("write --dump");
                }
            }
        }
    }
    if let Some(stats) = sched.device_stats() {
        println!("\ndevice stats at exit: {stats:?}");
    }
}
