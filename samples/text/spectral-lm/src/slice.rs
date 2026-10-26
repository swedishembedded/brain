// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model compression that keeps a pretrained
// model a standard checkpoint for its clients. If your team needs expertise
// in shrinking large language models for constrained hardware, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Shrinking a pretrained Qwen3 by its residual stream's principal
//! directions, and measuring what the narrower model keeps.
//!
//! `qwen-slice-fit` runs the model over calibration text, pools the second
//! moment of the residual stream at every point (each layer boundary and each
//! layer's stream after attention), every row scaled to unit length because
//! each consumer of the stream sees it through a norm, and writes the
//! eigenvectors of that matrix, largest first, as one basis for the whole
//! stream. `qwen-slice-eval` scores a model, or its slice to the first
//! `--width` directions of a basis ([`qwen3::slice::Slicing`]), by perplexity
//! on held-out text in fixed windows and by zero-shot accuracy on a
//! multiple-choice set, each choice scored by the log-likelihood of its
//! answer text after the question.

use std::time::Instant;

use appopts::Args;
use checkpoint::TensorSource;
use data::tokenizer::Tokenizer;
use model::{Batch, Model, Shard};
use qwen3::model::Qwen;
use qwen3::{Dtype, QwenConfig};

use crate::{die, read_text};

/// The model split over `count` consecutive stages, stage `i` on card `i`
/// when there is more than one, every stage at batch shape `b x t`.
struct Stages {
    stages: Vec<Qwen>,
    /// The layers of each stage.
    bounds: Vec<std::ops::Range<usize>>,
    b: usize,
    t: usize,
}

impl Stages {
    fn new(cfg: &QwenConfig, src: &dyn TensorSource, b: u32, t: u32, count: usize, dt: Dtype) -> Stages {
        let n = cfg.n_layers as usize;
        if count == 0 || count > n {
            die(&format!("--stages {count} must be in 1..={n}"));
        }
        let bounds: Vec<std::ops::Range<usize>> = (0..count).map(|i| i * n / count..(i + 1) * n / count).collect();
        let stages = bounds
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let shard = Shard { start: l.start, end: l.end, embed: i == 0, head: i + 1 == count, gpu_index: if count == 1 { Shard::ANY_GPU } else { i } };
                Qwen::new_shard_dt(cfg.clone(), b, t, src, shard, dt)
            })
            .collect();
        Stages { stages, bounds, b: b as usize, t: t as usize }
    }

    /// Run one batch; then `log p(target)` per position (0 where ignored).
    /// `visit(stage, residual points it holds)` runs after each stage.
    fn forward(&self, x: &[u32], y: &[u32], mut visit: impl FnMut(&Qwen, std::ops::Range<usize>)) -> Vec<f32> {
        let last = self.stages.len() - 1;
        let mut res: Option<Vec<f32>> = None;
        for (i, (s, l)) in self.stages.iter().zip(&self.bounds).enumerate() {
            Model::set_batch(s, Batch::Lm { tokens: x, targets: y });
            if let Some(r) = res.take() {
                s.write_in_res(&r);
            }
            if i == last {
                Model::forward(s);
                visit(s, 2 * l.start..2 * l.end + 1);
                return s.batch_token_logprobs();
            }
            s.run_forward();
            visit(s, 2 * l.start..2 * l.end);
            res = Some(s.read_out_res());
        }
        unreachable!("a model has at least one stage")
    }
}

/// `--dtype`: the storage tier of the linears.
fn dtype(a: &mut Args) -> Dtype {
    match a.str_or("--dtype", "f32").as_str() {
        "f32" => Dtype::F32,
        "bf16" => Dtype::BF16,
        other => die(&format!("--dtype {other:?}: expected f32 or bf16")),
    }
}

/// Inputs and next-token targets of windows of `t` tokens, `b` rows (a short
/// last group is padded with ignored rows).
fn lm_batch(windows: &[&[u32]], b: usize, t: usize) -> (Vec<u32>, Vec<u32>) {
    let (mut x, mut y) = (vec![0u32; b * t], vec![model::IGNORE; b * t]);
    for (r, w) in windows.iter().enumerate() {
        x[r * t..r * t + w.len()].copy_from_slice(w);
        y[r * t..r * t + w.len() - 1].copy_from_slice(&w[1..]);
    }
    (x, y)
}

pub fn fit(a: &mut Args) {
    let weights = a.take_str("--weights").unwrap_or_else(|| die("qwen-slice-fit needs --weights <Qwen3 checkpoint>"));
    let (tok, _) = crate::qwen::tokenizer(a, &weights);
    let calib = a.take_str("--calib").unwrap_or_else(|| die("qwen-slice-fit needs --calib <calibration text>"));
    let out = a.take_str("--out").unwrap_or_else(|| die("qwen-slice-fit needs --out <basis file>"));
    let windows = a.usize_or("--windows", 128);
    let window = a.u32_or("--window", 1024);
    let stages = a.usize_or("--stages", 1);
    let dt = dtype(a);
    a.finish();

    let start = Instant::now();
    let ids = tok.encode(&read_text(&calib));
    let chunks: Vec<&[u32]> = ids.chunks_exact(window as usize).take(windows).collect();
    if chunks.len() < windows {
        die(&format!("{calib} holds {} windows of {window} tokens, --windows asks for {windows}", chunks.len()));
    }
    let (cfg, src) = qwen3::open_checkpoint(&weights).unwrap_or_else(|e| die(&e));
    let d = cfg.d_model as usize;
    let model = Stages::new(&cfg, &*src, 1, window, stages, dt);
    eprintln!("calibrating on {} windows of {window} tokens", chunks.len());

    // Pooled second moment of unit rows, accumulated in f64 from f32 blocks.
    let mut moment = vec![0.0f64; d * d];
    let (mut rows, mut nll, mut predicted) = (0usize, 0.0f64, 0usize);
    let mut pending: Vec<f32> = Vec::with_capacity(BLOCK_ROWS * d);
    for w in &chunks {
        let (x, y) = lm_batch(&[w], 1, window as usize);
        let lp = model.forward(&x, &y, |stage, points| {
            for p in points {
                for row in stage.read_residual(p).chunks(d) {
                    let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(f32::MIN_POSITIVE);
                    pending.extend(row.iter().map(|v| v / norm));
                    rows += 1;
                    if pending.len() == BLOCK_ROWS * d {
                        add_moment(&mut pending, &mut moment, d);
                    }
                }
            }
        });
        nll -= lp.iter().map(|v| *v as f64).sum::<f64>();
        predicted += w.len() - 1;
    }
    if !pending.is_empty() {
        add_moment(&mut pending, &mut moment, d);
    }
    drop(model);
    moment.iter_mut().for_each(|m| *m /= rows as f64);
    let fitted = start.elapsed().as_secs_f64();
    let (values, vectors) = model::eigen::symmetric_eigen(&moment, d);
    let basis: Vec<u8> = vectors.iter().rev().flatten().flat_map(|v| (*v as f32).to_le_bytes()).collect();
    std::fs::write(&out, basis).unwrap_or_else(|e| die(&format!("{out}: {e}")));
    let total: f64 = values.iter().sum();
    let descending: Vec<f64> = values.iter().rev().copied().collect();
    let kept = |k: usize| descending[..k].iter().sum::<f64>() / total;
    let fractions: Vec<serde_json::Value> = (1..8).map(|eighth| d * eighth / 8).map(|k| serde_json::json!({ "width": k, "energy": kept(k) })).collect();
    println!(
        "{}",
        serde_json::json!({
            "event": "slice-fit",
            "checkpoint": weights,
            "calibration": calib,
            "residual_width": d,
            "layers": cfg.n_layers,
            "windows": chunks.len(),
            "window": window,
            "rows": rows,
            "calibration_nats_per_token": nll / predicted as f64,
            "dtype": format!("{dt:?}"),
            "energy": fractions,
            "basis": out,
            "fit_seconds": fitted,
            "seconds": start.elapsed().as_secs_f64(),
        })
    );
}

/// Residual rows per block of the pooled second moment.
const BLOCK_ROWS: usize = 4096;

/// `moment += Xᵀ X` for the rows `X` (`[n, d]`) in `pending`, which is emptied.
fn add_moment(pending: &mut Vec<f32>, moment: &mut [f64], d: usize) {
    let n = pending.len() / d;
    let mut xt = vec![0.0f32; pending.len()];
    for (r, row) in pending.chunks(d).enumerate() {
        for (c, v) in row.iter().enumerate() {
            xt[c * n + r] = *v;
        }
    }
    let block = model::hostmath::linear_rows(&xt, &xt, d, n, d);
    moment.iter_mut().zip(&block).for_each(|(m, v)| *m += *v as f64);
    pending.clear();
}

/// One multiple-choice question: the question text, its choices, the index of the correct one.
struct Question {
    question: String,
    choices: Vec<String>,
    answer: usize,
}

impl Question {
    fn parse(line: &str) -> Result<Question, String> {
        let v: serde_json::Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let question = v["question"].as_str().ok_or("no \"question\" string")?.to_string();
        let choices: Vec<String> = v["choices"].as_array().ok_or("no \"choices\" array")?.iter().map(|c| c.as_str().map(str::to_string).ok_or("a choice is not a string")).collect::<Result<_, _>>()?;
        let answer = v["answer"].as_u64().ok_or("no \"answer\" index")? as usize;
        if answer >= choices.len() {
            return Err(format!("answer {answer} of {} choices", choices.len()));
        }
        Ok(Question { question, choices, answer })
    }
}

pub fn eval(a: &mut Args) {
    let weights = a.take_str("--weights").unwrap_or_else(|| die("qwen-slice-eval needs --weights <Qwen3 checkpoint>"));
    let (tok, _) = crate::qwen::tokenizer(a, &weights);
    let basis_path = a.take_str("--basis");
    let width = a.opt_u32("--width").map(|w| w as usize);
    let text = a.take_str("--text").unwrap_or_else(|| die("qwen-slice-eval needs --text <held-out text>"));
    let choice = a.take_str("--choice").unwrap_or_else(|| die("qwen-slice-eval needs --choice <multiple-choice jsonl>"));
    let window = a.u32_or("--window", 1024);
    let batch = a.u32_or("--batch", 1);
    let choice_batch = a.u32_or("--choice-batch", 16);
    let stages = a.usize_or("--stages", 1);
    let dt = dtype(a);
    a.finish();

    let start = Instant::now();
    let (cfg, src) = qwen3::open_checkpoint(&weights).unwrap_or_else(|e| die(&e));
    let d = cfg.d_model as usize;
    let basis: Option<Vec<f32>> = basis_path.as_ref().map(|p| {
        let bytes = std::fs::read(p).unwrap_or_else(|e| die(&format!("{p}: {e}")));
        if bytes.len() != d * d * 4 {
            die(&format!("{p} holds {} bytes, a {d} x {d} basis is {}", bytes.len(), d * d * 4));
        }
        bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    });
    let slicing = match (&basis, width) {
        (Some(b), Some(k)) => Some(qwen3::slice::Slicing::new(&cfg, &*src, b, k).unwrap_or_else(|e| die(&e))),
        (None, None) => None,
        _ => die("--basis and --width go together"),
    };
    let (cfg_eval, source): (QwenConfig, &dyn TensorSource) = match &slicing {
        Some(s) => (s.config().clone(), s),
        None => (cfg.clone(), &*src),
    };
    let params: usize = cfg_eval.param_list().iter().map(|(_, n)| n).sum();
    let tables = cfg_eval.vocab as usize * cfg_eval.d_model as usize * if cfg_eval.tie_embeddings { 1 } else { 2 };

    // Perplexity over non-overlapping windows of held-out text.
    let ids = tok.encode(&read_text(&text));
    let windows: Vec<&[u32]> = ids.chunks_exact(window as usize).collect();
    let model = Stages::new(&cfg_eval, source, batch, window, stages, dt);
    let (mut nll, mut predicted) = (0.0f64, 0usize);
    for group in windows.chunks(model.b) {
        let (x, y) = lm_batch(group, model.b, model.t);
        nll -= model.forward(&x, &y, |_, _| {}).iter().map(|v| *v as f64).sum::<f64>();
        predicted += group.iter().map(|w| w.len() - 1).sum::<usize>();
    }
    drop(model);
    let lm_seconds = start.elapsed().as_secs_f64();

    // Zero-shot multiple choice: every (question, choice) sequence scored for
    // the log-likelihood of the choice's tokens.
    let questions: Vec<Question> = read_text(&choice)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| Question::parse(l).unwrap_or_else(|e| die(&format!("{choice} line {}: {e}", i + 1))))
        .collect();
    let mut seqs: Vec<(Vec<u32>, usize, usize)> = Vec::new(); // (tokens, answer start, answer bytes)
    for q in &questions {
        let context = tok.encode(&format!("Question: {}\nAnswer:", q.question));
        for c in &q.choices {
            let answer = format!(" {c}");
            let mut t = context.clone();
            t.extend(tok.encode(&answer));
            seqs.push((t, context.len(), answer.len()));
        }
    }
    let longest = seqs.iter().map(|s| s.0.len()).max().unwrap_or(2) as u32;
    let model = Stages::new(&cfg_eval, source, choice_batch, longest, stages, dt);
    let mut scores = vec![0.0f64; seqs.len()];
    for (gi, group) in seqs.chunks(model.b).enumerate() {
        let (mut x, mut y) = (vec![0u32; model.b * model.t], vec![model::IGNORE; model.b * model.t]);
        for (r, (t, at, _)) in group.iter().enumerate() {
            x[r * model.t..r * model.t + t.len()].copy_from_slice(t);
            y[r * model.t + at - 1..r * model.t + t.len() - 1].copy_from_slice(&t[*at..]);
        }
        let lp = model.forward(&x, &y, |_, _| {});
        for (r, _) in group.iter().enumerate() {
            scores[gi * model.b + r] = lp[r * model.t..(r + 1) * model.t].iter().map(|v| *v as f64).sum();
        }
    }
    let (mut correct, mut correct_norm, mut first) = (0usize, 0usize, 0usize);
    for q in &questions {
        let n = q.choices.len();
        let s = &scores[first..first + n];
        let bytes: Vec<usize> = seqs[first..first + n].iter().map(|x| x.2).collect();
        let best = (0..n).max_by(|&i, &j| s[i].total_cmp(&s[j])).unwrap_or(0);
        let best_norm = (0..n).max_by(|&i, &j| (s[i] / bytes[i] as f64).total_cmp(&(s[j] / bytes[j] as f64))).unwrap_or(0);
        correct += (best == q.answer) as usize;
        correct_norm += (best_norm == q.answer) as usize;
        first += n;
    }
    println!(
        "{}",
        serde_json::json!({
            "event": "slice-eval",
            "checkpoint": weights,
            "basis": basis_path,
            "residual_width": cfg_eval.d_model,
            "original_width": d,
            "params": params,
            "params_non_embedding": params - tables,
            "dtype": format!("{dt:?}"),
            "text": text,
            "window": window,
            "windows": windows.len(),
            "tokens": predicted,
            "nats_per_token": nll / predicted as f64,
            "perplexity": (nll / predicted as f64).exp(),
            "choice": choice,
            "questions": questions.len(),
            "accuracy": correct as f64 / questions.len() as f64,
            "accuracy_norm": correct_norm as f64 / questions.len() as f64,
            "lm_seconds": lm_seconds,
            "seconds": start.elapsed().as_secs_f64(),
        })
    );
}
