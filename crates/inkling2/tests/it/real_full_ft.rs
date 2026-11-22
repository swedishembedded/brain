// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements full fine-tuning of released mixture-of-experts models larger than the accelerator
// for its clients. If your team needs expertise in training every weight of such a release and serving the result,
// you can procure our services by sending an email to info@swedishembedded.com.

//! Spec: a full fine-tune of the released 26B (`weights=bf16,window=1`: the NVFP4/MXFP8 expert banks dequantized once
//! into bf16 values under an f32 master, master and AdamW state in a state file, a one-layer window of f32 slots on the
//! card; every other tensor in f32 on the card) on one short supervised row:
//! - a few steps lower the loss;
//! - a run restored from the state committed after step 2 (the store's weights and moments, the streamed state) takes
//!   step 3 exactly as the run that did not stop: the same loss before it and after it, bit for bit;
//! - the trained model exported through the generic `export_checkpoint` is a checkpoint the family's serving engine
//!   opens (bf16 banks held as bf16), and its cached greedy decode is the trainer's argmax at every step.
//!
//! It needs a card with about 60 GB free, about 120 GB of host memory and about 800 GB of disk under the temporary
//! directory, and runs for tens of minutes: it is run by hand (`--ignored`), the command is in the ledger.

use std::io::{Read, Seek, SeekFrom, Write};

use data::tokenizer::Tokenizer;
use gpu_core::provider::library::GemmPrecision;
use inkling2::serve::{Engine, EngineOptions};
use inkling2::train::Inkling2Train;
use model::finetune::ChatTrainer;
use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::visit_state::VisitPoolSizing;
use model::serve::PagedDecoder;
use model::Model;

const REPO: &str = "thinkingmachines/Inkling-2-Mini";
const TEXT: &str = "The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid.";
const LR: f32 = 1e-5;
/// Tokens of the text the greedy comparison starts from, and tokens it generates.
const PROMPT: usize = 8;
const NEW: usize = 4;

/// The process's resident set and the memory its cgroup is charged (page cache included), as Linux reports them.
fn host_memory() -> String {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut out: Vec<String> = status.lines().filter(|l| l.starts_with("VmRSS") || l.starts_with("VmHWM")).map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).collect();
    let group = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    if let Some(path) = group.lines().find_map(|l| l.strip_prefix("0::")) {
        if let Ok(bytes) = std::fs::read_to_string(format!("/sys/fs/cgroup{}/memory.current", path.trim())) {
            out.push(format!("cgroup {:.1} GB", bytes.trim().parse::<f64>().unwrap_or(0.0) / 1e9));
        }
    }
    out.join(", ")
}

fn device_memory(m: &Inkling2Train) -> String {
    match m.gpu().memory_usage() {
        Some(u) => format!("device live {:.1} GB, peak {:.1} GB", u.live as f64 / 1e9, u.peak_live as f64 / 1e9),
        None => "device memory not accounted by this backend".into(),
    }
}

fn step(m: &Inkling2Train, t: u32, x: &[u32], y: &[u32]) -> f32 {
    let at = std::time::Instant::now();
    m.zero_grads();
    m.set_batch(model::Batch::Lm { tokens: x, targets: y });
    let loss = m.forward();
    m.backward();
    m.adamw_step(t, LR, 0.0, model::Adam::default(), None, 1.0);
    m.poll_wait();
    eprintln!("step {t}: loss {loss:.6} ({:.0} s; {}; host {})", at.elapsed().as_secs_f64(), device_memory(m), host_memory());
    loss
}

fn loss_of(m: &Inkling2Train, x: &[u32], y: &[u32]) -> f32 {
    m.set_batch(model::Batch::Lm { tokens: x, targets: y });
    m.forward()
}

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32
}

/// The store's own parameters (weights and moments), written to one scratch file and read back: what the fit loop's
/// resume file holds besides the streamed state.
struct Remainder {
    file: std::fs::File,
    at: Vec<(String, u64, usize)>,
}

impl Remainder {
    fn save(m: &Inkling2Train, path: &std::path::Path) -> Remainder {
        let mut file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path).unwrap();
        let mut at = Vec::new();
        let mut offset = 0u64;
        for name in m.optimized_params().expect("the store's parameters") {
            let w = m.read_weight(&name);
            let (mo, v) = m.read_moments(&name).expect("moments");
            for part in [&w, &mo, &v] {
                let bytes: Vec<u8> = part.iter().flat_map(|x| x.to_le_bytes()).collect();
                file.write_all(&bytes).unwrap();
            }
            at.push((name, offset, w.len()));
            offset += 12 * w.len() as u64;
        }
        Remainder { file, at }
    }

    fn restore(&mut self, m: &Inkling2Train) {
        for (name, offset, n) in &self.at {
            self.file.seek(SeekFrom::Start(*offset)).unwrap();
            let mut read = || {
                let mut bytes = vec![0u8; 4 * n];
                self.file.read_exact(&mut bytes).unwrap();
                bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect::<Vec<f32>>()
            };
            let (w, mo, v) = (read(), read(), read());
            m.write_weight(name, &w);
            m.write_moments(name, &mo, &v).unwrap();
        }
    }
}

#[test]
#[ignore = "needs a GH200-class card, ~120 GB of host memory and ~800 GB of temporary disk; run by hand (inkling2 ledger, I6)"]
fn a_full_fine_tune_of_the_release_learns_resumes_exactly_and_exports_what_it_trained() {
    let Some(dir) = brain_testutil::model_dir(REPO).map(std::path::PathBuf::from).filter(|d| d.join("hf_quant_config.json").is_file()) else {
        brain_testutil::skip_unavailable(&format!("{REPO} is not in the model store"));
        return;
    };
    let bpe = data::qwen_tokenizer::QwenBpe::from_file(&dir.join("tokenizer.json").to_string_lossy()).expect("the release's tokenizer");
    let tokens = bpe.encode(TEXT);
    let t = tokens.len() as u32 - 1;
    let (x, y) = (tokens[..t as usize].to_vec(), tokens[1..].to_vec());
    let scratch = brain_testutil::Scratch::dir("inkling2-real-full-ft");
    let seed = data::rng::random_seed();
    let mut resident = model::TrainPrecision::parse("weights=bf16,window=1").unwrap().resident_config(Some(seed)).unwrap().expect("a resident precision");
    resident.state_dir = Some(scratch.path().join("state"));
    eprintln!("seed {seed}, {t} tokens, {}", host_memory());

    let started = std::time::Instant::now();
    let mut m = Inkling2Train::from_dir_resident(&dir, 1, t, &resident).unwrap();
    // The served engine below runs the exact products; so does the trainer, so the two are one function.
    m.set_gemm_precision(GemmPrecision::Fp32Reference);
    eprintln!("built in {:.0} s; {}; host {}", started.elapsed().as_secs_f64(), device_memory(&m), host_memory());

    let first = step(&m, 1, &x, &y);
    step(&m, 2, &x, &y);
    // The state after step 2, as the fit loop's resume commits it.
    let mut remainder = Remainder::save(&m, &scratch.path().join("remainder.bin"));
    let committed = scratch.path().join("offdevice");
    let at = std::time::Instant::now();
    m.off_device_state().expect("a streamed build keeps off-device state").commit(&committed, 2).unwrap();
    eprintln!("committed in {:.0} s", at.elapsed().as_secs_f64());
    let third = step(&m, 3, &x, &y);
    let after = loss_of(&m, &x, &y);
    eprintln!("loss {first:.6} -> {after:.6} after three steps");
    assert!(after < first, "the loss went from {first} to {after}");

    let at = std::time::Instant::now();
    remainder.restore(&m);
    m.off_device_state().unwrap().restore(&committed, 2).unwrap();
    eprintln!("restored in {:.0} s", at.elapsed().as_secs_f64());
    let again = step(&m, 3, &x, &y);
    let again_after = loss_of(&m, &x, &y);
    assert_eq!(third.to_bits(), again.to_bits(), "the resumed step 3 starts from loss {again}, the uninterrupted one from {third}");
    assert_eq!(after.to_bits(), again_after.to_bits(), "the resumed step 3 ends at loss {again_after}, the uninterrupted one at {after}");
    drop(remainder);
    let _ = std::fs::remove_dir_all(&committed);
    let _ = std::fs::remove_file(scratch.path().join("remainder.bin"));

    // The trainer's greedy continuation, one uncached forward per token.
    let mut seq: Vec<u32> = tokens[..PROMPT].to_vec();
    let valid = m.config().valid_vocab as usize;
    let mut want = Vec::new();
    for _ in 0..NEW {
        let logits = m.logits(&seq).unwrap();
        let next = argmax(&logits[(seq.len() - 1) * valid..seq.len() * valid]);
        want.push(next);
        seq.push(next);
    }
    let trained = scratch.path().join("trained.safetensors");
    let at = std::time::Instant::now();
    m.save(trained.to_str().unwrap());
    drop(m);
    let _ = std::fs::remove_dir_all(scratch.path().join("state"));
    let out = scratch.path().join("export");
    inkling2::finetune::Inkling2Trainer::export_checkpoint(&trained, &dir, &out, None, &mut || true).unwrap();
    std::fs::remove_file(&trained).unwrap();
    eprintln!("saved and exported in {:.0} s", at.elapsed().as_secs_f64());

    let sizing = VisitPoolSizing { block_size: 16, num_blocks: 4, live_rows: 1, snapshot_rows: 0, undo_rows: false };
    let mut ops = model::ops::Ops::new(gpu_core::Gpu::new(Box::leak(inkling2::kernels().into_boxed_slice()))).unwrap();
    ops.set_gemm_precision(GemmPrecision::Fp32Reference);
    let mut e = Engine::open(ops, &out, EngineOptions { kv_tier: KvTier::F32, sizing, max_seq_len: 32, prefill_chunk: 16 }).unwrap();
    let mut table = BlockTable::new();
    let hidden = e.prefill(&mut table, &tokens[..PROMPT]);
    let mut served = vec![e.admit_greedy(&hidden)];
    while served.len() < NEW {
        let next = e.forward_batched_greedy(&mut [&mut table], &[*served.last().unwrap()]);
        served.push(next[0]);
    }
    e.release_table(&mut table);
    eprintln!("trainer greedy {want:?}, served cached greedy {served:?}");
    assert_eq!(served, want, "the exported checkpoint's cached greedy decode against the trainer's argmax");
}
