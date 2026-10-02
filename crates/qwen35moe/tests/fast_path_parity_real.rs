// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The released Qwen3.6-35B-A3B Q8_0 GGUF, full depth, on its fast path against its
//! portable reference path: prefill and greedy decode, one sequence and a batch.
//!
//! Swedish Embedded AB implements fast, verified inference for large sparse-MoE
//! models for its clients. If your team needs expertise in making a model several
//! times faster without changing what it says then you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! The fast path is the native int8 kernels, the grouped sparse-MoE GEMM, the fused
//! decode kernels, the Gated-DeltaNet prefill kernels and replayed decode tapes. The
//! reference path switches every one of them off (`BRAIN_NO_KERNEL_UPGRADE`,
//! `BRAIN_NO_GDN_FAST`, no grouping, no fusion, no tapes): the same weights through
//! the plain kernels. Two of those switches are read once per process, so the
//! reference runs in a child process of this test and hands its numbers back as JSON.
//!
//! The fast side is fed the reference's greedy tokens (teacher forcing), so one
//! router near-tie cannot turn into a different continuation and make every later
//! step incomparable. Per step it must agree on the next token wherever the
//! reference's own top-2 gap is wider than the numerics floor, and its top-k logits
//! must sit within a relative L2 of the reference's.
//!
//! Gated on `BRAIN_QWEN35MOE_GGUF` (skipped without it).

use checkpoint::gguf::MmapGguf;
use data::tokenizer::Tokenizer;
use model::paged::BlockTable;
use qwen35moe::gguf_load;
use qwen35moe::serve::{Engine, EngineOptions};
use serde_json::{json, Value};

const STEPS: usize = 24;
const TOPK: usize = 16;
const ROLE_ENV: &str = "BRAIN_PARITY_ROLE";
const OUT_ENV: &str = "BRAIN_PARITY_OUT";
const TEST_NAME: &str = "the_fast_path_follows_the_portable_reference_on_the_real_checkpoint";

/// Prompts of different lengths: all above the grouped-MoE row threshold, so
/// prefill takes the grouped kernels, and distinct, so a batch routes differently
/// per row.
const PROMPTS: [&str; 4] = [
    "The Gated DeltaNet layers of a hybrid model keep a small recurrent state instead of a key-value cache, so the memory a long conversation needs grows only with the attention layers, and the cost of one more token stays flat. A mixture of experts adds capacity without adding the work per token, because",
    "Write a short story about a lighthouse keeper who discovers that the light he tends is answering a signal from far below the sea. On the first night of the storm he",
    "Explain, step by step and for a first-year student, why the sum of the first n odd numbers is n squared, and then give two different proofs. First,",
    "Quarterly infrastructure report. Summary of GPU utilisation, queue wait times and failed jobs for the cluster, followed by recommendations for the next quarter. Utilisation rose because",
];

/// How a run is built; the reference is everything off.
struct Mode {
    fast: bool,
}

fn engine(mg: &MmapGguf, mode: &Mode) -> Engine {
    let cfg = gguf_load::resident_config(mg, 512).expect("config");
    let src = gguf_load::source(mg, &cfg).expect("source");
    let mut opts = EngineOptions::new(512, PROMPTS.len() as u32)
        .with_tier(gguf_load::tier_from_env())
        .with_kv_tier(gguf_load::kv_tier_from_env().expect("kv tier"));
    if !mode.fast {
        opts = opts
            .with_moe_grouped_min_rows(u32::MAX)
            .with_decode_fusion(false)
            .with_decode_tapes(false);
    }
    Engine::from_source(cfg, &src, opts)
}

/// What one run records for one set of sequences decoded together.
struct Run {
    /// Last-token hidden state of each prompt's prefill.
    hidden: Vec<Vec<f32>>,
    /// `steps[s][row]` = top-k (id, logit) of step `s`.
    steps: Vec<Vec<Vec<(u32, f32)>>>,
}

/// Prefill `prompts`, then decode `STEPS` steps together. `forced`, when given,
/// is the token each row takes at each step; otherwise the row's own argmax.
fn run(e: &mut Engine, prompts: &[Vec<u32>], forced: Option<&[Vec<u32>]>) -> Run {
    let mut tables: Vec<BlockTable> = prompts.iter().map(|_| BlockTable::new()).collect();
    let hidden: Vec<Vec<f32>> = prompts
        .iter()
        .zip(tables.iter_mut())
        .map(|(p, t)| e.prefill(t, p))
        .collect();
    let mut tokens: Vec<u32> = prompts
        .iter()
        .map(|p| *p.last().expect("a prompt"))
        .collect();
    let mut steps = Vec::new();
    for s in 0..STEPS {
        let top = {
            let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
            e.forward_batched_topk(&mut refs, &tokens, TOPK)
        };
        tokens = match forced {
            Some(f) => f[s].clone(),
            None => top.iter().map(|r| r[0].0).collect(),
        };
        steps.push(top);
    }
    for t in &mut tables {
        e.release_table(t);
    }
    Run { hidden, steps }
}

fn to_json(r: &Run) -> Value {
    json!({
        "hidden": r.hidden,
        "steps": r.steps.iter().map(|rows| rows.iter().map(|row| row.iter().map(|&(i, v)| json!([i, v])).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>(),
    })
}

fn from_json(v: &Value) -> Run {
    let hidden = v["hidden"]
        .as_array()
        .expect("hidden")
        .iter()
        .map(|h| {
            h.as_array()
                .expect("row")
                .iter()
                .map(|x| x.as_f64().expect("f") as f32)
                .collect()
        })
        .collect();
    let steps = v["steps"]
        .as_array()
        .expect("steps")
        .iter()
        .map(|rows| {
            rows.as_array()
                .expect("rows")
                .iter()
                .map(|row| {
                    row.as_array()
                        .expect("row")
                        .iter()
                        .map(|c| {
                            (
                                c[0].as_u64().expect("id") as u32,
                                c[1].as_f64().expect("logit") as f32,
                            )
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    Run { hidden, steps }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

/// Both configurations over the same prompts: the single-sequence runs for each
/// prompt in turn, then the whole batch together.
fn runs(e: &mut Engine, prompts: &[Vec<u32>], forced: Option<&Value>) -> Value {
    let mut all = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let f = forced.map(|v| tokens_of(&from_json(&v[i])));
        all.push(to_json(&run(e, std::slice::from_ref(p), f.as_deref())));
    }
    let f = forced.map(|v| tokens_of(&from_json(&v[prompts.len()])));
    all.push(to_json(&run(e, prompts, f.as_deref())));
    Value::Array(all)
}

/// The greedy token of each step and row of a run.
fn tokens_of(r: &Run) -> Vec<Vec<u32>> {
    r.steps
        .iter()
        .map(|rows| rows.iter().map(|row| row[0].0).collect())
        .collect()
}

fn gguf() -> Option<(String, MmapGguf, Vec<Vec<u32>>)> {
    let path = std::env::var(gguf_load::GGUF_ENV).ok()?;
    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let tok = gguf_load::tokenizer(&mg).expect("the GGUF's tokenizer");
    let prompts = PROMPTS.iter().map(|p| tok.encode(p)).collect();
    Some((path, mg, prompts))
}

#[test]
fn the_fast_path_follows_the_portable_reference_on_the_real_checkpoint() {
    let Some((_, mg, prompts)) = gguf() else {
        return brain_testutil::skip_unavailable("BRAIN_QWEN35MOE_GGUF is not set");
    };

    // Child: the reference path. Its switches are read once per process.
    if std::env::var(ROLE_ENV).as_deref() == Ok("reference") {
        let mut e = engine(&mg, &Mode { fast: false });
        let out = std::env::var(OUT_ENV).expect("the reference child needs an output path");
        std::fs::write(out, runs(&mut e, &prompts, None).to_string())
            .expect("write the reference run");
        return;
    }

    let mut e = engine(&mg, &Mode { fast: true });
    if !e.gpu().caps().numeric.int8_dot {
        return brain_testutil::skip_unavailable("no packed int8 dot on this device");
    }

    let out = std::env::temp_dir().join(format!(
        "brain-fast-path-parity-{}.json",
        std::process::id()
    ));
    let status = std::process::Command::new(std::env::current_exe().expect("this test binary"))
        .args([TEST_NAME, "--exact", "--nocapture"])
        .env(ROLE_ENV, "reference")
        .env(OUT_ENV, &out)
        .env("BRAIN_NO_KERNEL_UPGRADE", "1")
        .env("BRAIN_NO_GDN_FAST", "1")
        .status()
        .expect("spawn the reference child");
    assert!(status.success(), "the reference child failed");
    let reference: Value =
        serde_json::from_str(&std::fs::read_to_string(&out).expect("read the reference run"))
            .expect("reference json");
    let _ = std::fs::remove_file(&out);

    let fast = runs(&mut e, &prompts, Some(&reference));
    for (idx, (f, r)) in fast
        .as_array()
        .unwrap()
        .iter()
        .zip(reference.as_array().unwrap())
        .enumerate()
    {
        let (f, r) = (from_json(f), from_json(r));
        let label = if idx < prompts.len() {
            format!("prompt {idx} alone")
        } else {
            format!("batch of {}", prompts.len())
        };
        for (row, (fh, rh)) in f.hidden.iter().zip(&r.hidden).enumerate() {
            let d = rel_l2(fh, rh);
            eprintln!("{label}: prefill row {row} hidden rel-L2 {d:.3e}");
            assert!(
                d <= PREFILL_REL_L2,
                "{label}: prefill hidden of row {row} differs by {d:.3e}"
            );
        }
        let (mut worst, mut flips, mut ambiguous) = (0f64, 0, 0);
        for (s, (fs, rs)) in f.steps.iter().zip(&r.steps).enumerate() {
            for (row, (fr, rr)) in fs.iter().zip(rs).enumerate() {
                // Compare the logits of the ids both sides ranked (the tail of a top-k swaps in near-ties).
                let (mut a, mut b) = (Vec::new(), Vec::new());
                for &(id, v) in rr {
                    if let Some(&(_, w)) = fr.iter().find(|c| c.0 == id) {
                        a.push(w);
                        b.push(v);
                    }
                }
                assert!(
                    a.len() >= TOPK / 2,
                    "{label}: step {s} row {row}: the top-{TOPK} sets share only {} ids",
                    a.len()
                );
                let d = rel_l2(&a, &b);
                worst = worst.max(d);
                let gap = rr[0].1 - rr[1].1;
                let tie_gap = TIE_REL_GAP * rr[0].1.abs().max(1.0);
                if fr[0].0 != rr[0].0 {
                    if gap <= tie_gap {
                        ambiguous += 1;
                    } else {
                        flips += 1;
                        eprintln!("{label}: step {s} row {row}: argmax {} against {} with a reference gap of {gap}", fr[0].0, rr[0].0);
                    }
                }
            }
        }
        eprintln!("{label}: decode worst top-k logit rel-L2 {worst:.3e}, clear flips {flips}, near-tie flips {ambiguous}");
        assert_eq!(flips, 0, "{label}: the fast path picks a different token where the reference is not in a near-tie");
        assert!(
            worst <= DECODE_REL_L2,
            "{label}: decode logits differ by rel-L2 {worst:.3e}"
        );
    }
}

/// The numerics floor, measured on this checkpoint (see the module doc of the
/// commit that set them): rounding differences between int8 accumulation orders
/// and fused exponentials, compounded through forty layers.
const PREFILL_REL_L2: f64 = 5e-2;
const DECODE_REL_L2: f64 = 1e-1;
/// A reference top-2 logit gap, relative to the top logit, below which either token
/// is a legitimate argmax: the logits themselves are only good to `DECODE_REL_L2`.
const TIE_REL_GAP: f32 = 0.05;
