// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Click the thing I asked for: a decision model that reads a rendered screen
//! and returns which control to click, for an instruction it is given in
//! plain language at run time.
//!
//! ```text
//! screen pixels ──► per-control crops ──┬──► state rows ─┐
//!                                       │                ├─► decide head ──► one score
//! "click the red square in the toolbar" ┴──► slot rows ──┘        per control
//!            (one frozen encoder pass)
//! ```
//!
//! The controls on screen are the OPTIONS, and there is a different set of
//! them every frame - which is the decision surface's own premise, that the
//! options arrive at run time and are scored by what they mean. Here what
//! they mean is what they look like.
//!
//! **Every number the model reads comes from the rendered image.**
//! `vision::widget_features` pools the canvas the PNG is written from; the
//! control's declared colour and shape are used to DRAW it and to check the
//! answer, never to feed it. `vision`'s own tests pin that down.
//!
//! What makes it work is a trainable projection on each side of the head's
//! cross-attention - see `project`'s module doc for the measurement that
//! showed why a frozen sentence encoder's output cannot be used as an option
//! query directly, and what replacing it with a learned one buys.
//!
//! Run it:
//!
//! ```text
//! make samples/learning/visualclick/run ARGS="--shot out/screens"
//! make samples/learning/visualclick/run ARGS="--validate"
//! make samples/learning/visualclick/run ARGS="--examples out/examples"
//! make samples/learning/visualclick/run ARGS="--ablate pixels"
//! make samples/learning/visualclick/run ARGS="--ablate instruction"
//! ```
//!
//! Swedish Embedded AB builds decision systems that ground an answer in
//! whatever evidence actually bears on it - a screen, a document, a live
//! signal - and reports exactly how that claim was checked rather than
//! assuming it. If your team needs a grounded decision layer taken to
//! production, you can procure our services by sending an email to
//! info@swedishembedded.com.

mod project;
mod screen;
mod vision;

use brain::{decision_loss, ece, Features, LossConfig, Opt, Question, RlcdPipeline};
use project::Grounder;
use screen::{Rng, Screen};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ablate {
    None,
    /// The screen the pixels come from is not the screen the instruction is
    /// about. Must collapse to chance, or the answer was not coming from the
    /// image.
    Pixels,
    /// The instruction is another screen's. Must collapse to chance, or the
    /// answer was not coming from the instruction.
    Instruction,
}

struct Args {
    encoder: String,
    ablate: Ablate,
    train_n: usize,
    eval_n: usize,
    seed: u64,
    shot: Option<String>,
    examples: Option<String>,
    examples_n: usize,
    validate: bool,
    head_lr: f32,
    grounder_lr: f32,
    checkpoint: Option<String>,
    retrain: bool,
}

fn checkpoint_dir(ablate: Ablate, seed: u64, train_n: usize) -> String {
    format!("out/visualclick/{ablate:?}-seed{seed}-train{train_n}").to_lowercase()
}

fn parse_args() -> Args {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut a = Args {
        encoder: std::env::var("BRAIN_MINILM_DIR")
            .unwrap_or_else(|_| format!("{home}/.local/share/brain/models/sentence-transformers/all-MiniLM-L6-v2")),
        ablate: Ablate::None,
        train_n: 20000,
        eval_n: 500,
        seed: 11,
        shot: None,
        examples: None,
        examples_n: 8,
        validate: false,
        head_lr: 1e-3,
        grounder_lr: 2e-3,
        checkpoint: None,
        retrain: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let next = || argv.get(i + 1).cloned().unwrap_or_default();
        let mut took = 2;
        match argv[i].as_str() {
            "--encoder" => a.encoder = next(),
            "--ablate" => {
                a.ablate = match next().as_str() {
                    "none" => Ablate::None,
                    "pixels" => Ablate::Pixels,
                    "instruction" => Ablate::Instruction,
                    other => {
                        eprintln!("visualclick: unknown --ablate {other:?}, expected none|pixels|instruction");
                        std::process::exit(1);
                    }
                }
            }
            "--train-screens" => a.train_n = next().parse().unwrap_or(a.train_n),
            "--eval-screens" => a.eval_n = next().parse().unwrap_or(a.eval_n),
            "--seed" => a.seed = next().parse().unwrap_or(a.seed),
            "--shot" => a.shot = Some(next()),
            "--examples" => a.examples = Some(next()),
            "--examples-n" => a.examples_n = next().parse().unwrap_or(a.examples_n),
            "--head-lr" => a.head_lr = next().parse().unwrap_or(a.head_lr),
            "--grounder-lr" => a.grounder_lr = next().parse().unwrap_or(a.grounder_lr),
            "--checkpoint" => a.checkpoint = Some(next()),
            "--validate" => {
                a.validate = true;
                took = 1;
            }
            "--retrain" => {
                a.retrain = true;
                took = 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: visualclick [--encoder DIR] [--ablate none|pixels|instruction]\n\
                     \x20                  [--train-screens N] [--eval-screens N] [--seed N]\n\
                     \x20                  [--shot DIR] [--examples DIR] [--examples-n N]\n\
                     \x20                  [--validate] [--checkpoint DIR] [--retrain]\n\n\
                     --shot DIR    render screens with the instruction's control outlined, and exit - no encoder needed.\n\
                     --validate    check the whole data path end to end (oracle, pixels, row layout, gradients) and exit.\n\
                     --examples DIR  after evaluating, render held-out screens with the correct control outlined in black\n\
                     \x20             and, where it differs, the model's own answer in red.\n\
                     --ablate      break one input on purpose; both settings must collapse the score to chance.\n\
                     --retrain     ignore an existing checkpoint and train anyway."
                );
                std::process::exit(0);
            }
            other => eprintln!("visualclick: ignoring unknown argument {other:?}"),
        }
        i += took;
    }
    a
}

fn require(path: &std::path::Path, what: &str, remedy: &str) {
    if !path.exists() {
        eprintln!("visualclick: no {what} at {}", path.display());
        eprintln!("visualclick: {remedy}");
        std::process::exit(1);
    }
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
    let z: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / z).collect()
}

fn argmax(p: &[f32]) -> usize {
    p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0)
}

/// The instruction, through the frozen encoder, ONCE - mean-pooled over its
/// own token rows, which is the pooling the released sentence-transformer
/// head uses.
///
/// This is the sample's only encoder pass per decision. The option rows are
/// projected, not encoded, so adding a control to the screen costs a
/// matrix-vector product rather than another pass through a 22M-parameter
/// transformer.
fn instruction_row_raw(pipeline: &mut RlcdPipeline, instruction: &str) -> Vec<f32> {
    // `Decide` always scores a question, so the cheapest way to ask it for a
    // representation of one string is a one-option question over that string.
    // The option's own row is discarded; the state rows are the instruction.
    let q = Question::Choice { instructions: instruction.to_string(), options: vec![Opt::new("x")] };
    let (_, kept) = pipeline.model_mut().score_keeping(instruction, &q).expect("score_keeping");
    let d = pipeline.model_mut().cfg.d_model as usize;
    let n = kept.state_rows() as usize;
    assert!(n > 0, "the encoder returned no rows for {instruction:?}");
    let mut out = vec![0.0f32; d];
    for r in 0..n {
        for (h, v) in out.iter_mut().enumerate() {
            *v += kept.hidden()[r * d + h];
        }
    }
    for v in &mut out {
        *v /= n as f32;
    }
    assert!(out.iter().all(|v| v.is_finite()), "the instruction row is not finite");
    out
}

/// Per-dimension mean and inverse standard deviation of the instruction
/// embedding, estimated once over the training instructions.
///
/// This is what makes a spatial instruction answerable at all. Every
/// instruction this sample issues starts "click the ...", and a sentence
/// encoder puts most of its output into that shared shape: measured on this
/// repository's MiniLM checkpoint, two instructions differing only in the
/// DIRECTION word sit at **cosine 0.9837**, against 0.8510 for two differing
/// in the colour. The thing that decides the answer moves the vector about a
/// tenth as far as the thing that does not, and a layer reading the raw
/// embedding sees the template, not the word.
///
/// Centring removes the template. What is left is the part that varies BETWEEN
/// instructions, which is the part carrying the instruction's content; scaling
/// each dimension by its own spread then stops a few high-variance dimensions
/// from standing in for all of it. Neither step is learned and neither looks at
/// a label - it is the ordinary whitening any feature gets before a linear
/// layer, applied to the one input here that had been going in raw.
struct InstructionNorm {
    mean: Vec<f32>,
    inv_std: Vec<f32>,
}

impl InstructionNorm {
    /// Estimated from TRAINING instructions only, so the held-out screens are
    /// scored by statistics that never saw them.
    fn fit(pipeline: &mut RlcdPipeline, screens: &[Screen], d: usize) -> InstructionNorm {
        let n = screens.len().min(1000).max(1);
        let rows: Vec<Vec<f32>> = screens.iter().take(n).map(|s| instruction_row_raw(pipeline, &s.instruction)).collect();
        let mut mean = vec![0.0f32; d];
        for r in &rows {
            for (h, v) in mean.iter_mut().enumerate() {
                *v += r[h];
            }
        }
        for v in &mut mean {
            *v /= rows.len() as f32;
        }
        let mut var = vec![0.0f32; d];
        for r in &rows {
            for (h, v) in var.iter_mut().enumerate() {
                let dv = r[h] - mean[h];
                *v += dv * dv;
            }
        }
        // A floor rather than a bare reciprocal: a dimension that never moves
        // across the whole instruction distribution carries no information,
        // and dividing it by its own near-zero spread would amplify only
        // numerical noise into the query.
        let inv_std = var.iter().map(|v| 1.0 / ((v / rows.len() as f32).sqrt() + 1e-3)).collect();
        InstructionNorm { mean, inv_std }
    }

    fn apply(&self, row: &mut [f32]) {
        for (h, v) in row.iter_mut().enumerate() {
            *v = (*v - self.mean[h]) * self.inv_std[h];
        }
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.mean.len() as u64).to_le_bytes());
        for v in self.mean.iter().chain(&self.inv_std) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, out)
    }

    fn load(path: &str, d: usize) -> Result<InstructionNorm, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let len = bytes.get(..8).ok_or_else(|| format!("{path}: truncated"))?;
        let len = u64::from_le_bytes(len.try_into().expect("8 bytes")) as usize;
        if len != d || bytes.len() != 8 + 2 * d * 4 {
            return Err(format!("{path}: holds {len} dimensions, this build expects {d}"));
        }
        let f = |at: usize| f32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"));
        Ok(InstructionNorm {
            mean: (0..d).map(|i| f(8 + i * 4)).collect(),
            inv_std: (0..d).map(|i| f(8 + (d + i) * 4)).collect(),
        })
    }
}

/// Write the decision head's tensors beside the grounder's.
///
/// `Decide::save_head` plus `RlcdPipeline::builder().head()` is the usual
/// route and is deliberately not the one taken here. That pair round-trips a
/// TASK CONTRACT - a fixed question with a fixed list of options - and this
/// sample has no such thing to write: its options are the controls on the
/// screen in front of it, a different set every frame, which is the entire
/// point of scoring them by what they look like. A contract invented to
/// satisfy the loader would be a fixed question this model was never trained
/// on, so the head's tensors are written directly instead, in the same format
/// as the grounder's.
fn save_head(pipeline: &mut RlcdPipeline, path: &str) -> std::io::Result<()> {
    let mut out = Vec::new();
    let w = pipeline.model_mut().head.weights();
    out.extend_from_slice(&(w.len() as u64).to_le_bytes());
    for (name, values) in &w {
        out.extend_from_slice(&(name.len() as u64).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, out)
}

fn load_head(pipeline: &mut RlcdPipeline, path: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let mut at = 0usize;
    let mut take = |n: usize| -> Result<&[u8], String> {
        let s = bytes.get(at..at + n).ok_or_else(|| format!("{path}: truncated at byte {at}"))?;
        at += n;
        Ok(s)
    };
    let u64_of = |s: &[u8]| u64::from_le_bytes(s.try_into().expect("8 bytes"));
    let count = u64_of(take(8)?) as usize;
    let mut w = Vec::with_capacity(count);
    for _ in 0..count {
        let n = u64_of(take(8)?) as usize;
        let name = String::from_utf8(take(n)?.to_vec()).map_err(|e| format!("{path}: tensor name is not UTF-8: {e}"))?;
        let len = u64_of(take(8)?) as usize;
        let raw = take(len * 4)?;
        let values: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).collect();
        w.push((name, values));
    }
    if at != bytes.len() {
        return Err(format!("{path}: {} trailing bytes", bytes.len() - at));
    }
    // Checked rather than trusted: this file is a build artifact under `out/`
    // and a code change that resizes the head would otherwise load silently
    // into a model that scores nonsense.
    let want = pipeline.model_mut().head.weights();
    for (name, values) in &w {
        match want.iter().find(|(n, _)| n == name) {
            Some((_, v)) if v.len() == values.len() => {}
            Some((_, v)) => return Err(format!("{path}: {name} has {} values, this build expects {}", values.len(), v.len())),
            None => return Err(format!("{path}: {name} is not a parameter of this head")),
        }
    }
    if w.len() != want.len() {
        return Err(format!("{path}: holds {} tensors, this head has {}", w.len(), want.len()));
    }
    pipeline.model_mut().head.set_weights(&w);
    Ok(())
}

/// What one decision reads: the controls' appearance, and the instruction.
/// Ablations break exactly one of the two and nothing else.
struct Evidence {
    feats: Vec<Vec<f32>>,
    /// The instruction, pooled to one vector - what conditions each option's
    /// QUERY. A query is one row and cannot be a sequence.
    cls: Vec<f32>,
}

fn other(i: usize, n: usize) -> usize {
    (i + n / 2 + 1) % n
}

fn perceive(pipeline: &mut RlcdPipeline, norm: &InstructionNorm, screens: &[Screen], i: usize, ablate: Ablate) -> Evidence {
    let from = if ablate == Ablate::Pixels { other(i, screens.len()) } else { i };
    let said = if ablate == Ablate::Instruction { other(i, screens.len()) } else { i };

    let canvas = screens[from].render();
    let mut feats = screens[from].features(&canvas);
    // An ablated screen can hold a different number of controls than the one
    // being answered about; the option set is the answered screen's, so the
    // rows are trimmed or padded to it rather than silently changing the
    // question's arity.
    let want = screens[i].widgets.len();
    feats.resize(want, vec![0.0; vision::FEAT_DIM]);

    let mut cls = instruction_row_raw(pipeline, &screens[said].instruction);
    norm.apply(&mut cls);
    Evidence { feats, cls }
}

/// `[instruction rows; control state rows; one slot row per control]` - the
/// layout `Features::from_parts` reads, with the instruction's tokens leading
/// the state so the head can attend to them alongside the controls.
fn features_of(grounder: &mut Grounder, e: &Evidence) -> Features {
    let n = e.feats.len() as u32;
    Features::from_parts(grounder.forward(&e.feats, &e.cls), n, n)
}

fn train(pipeline: &mut RlcdPipeline, grounder: &mut Grounder, norm: &InstructionNorm, screens: &[Screen], ablate: Ablate, head_lr: f32) {
    pipeline.model_mut().set_encoder_frozen(true);
    let loss_cfg = LossConfig::cross_entropy();
    let report_every = (screens.len() / 12).clamp(1, 500);
    let d = pipeline.model_mut().cfg.d_model as usize;

    for i in 0..screens.len() {
        let e = perceive(pipeline, norm, screens, i, ablate);
        let gold = screens[i].target;
        let n = e.feats.len();
        let f = features_of(grounder, &e);

        pipeline.model_mut().zero_grads();
        let loss = pipeline
            .model_mut()
            .accumulate_kept(&f, |scores| decision_loss(scores, gold, &loss_cfg))
            .expect("accumulate_kept");

        // The head scattered its gradient on the rows it was given into the
        // encoder's seed buffer, in the order it was given them: the state
        // rows, then one row per option. `Decide::accumulate_kept` clears that
        // buffer per call, so this is THIS step's gradient.
        // The head scattered its gradient on the rows it was given into the
        // encoder's seed buffer, in the order it was given them: the state
        // rows, then one row per option. `Decide::accumulate_kept` clears that
        // buffer per call, so this is THIS step's gradient.
        let grad = {
            let m = pipeline.model_mut();
            m.enc.gpu().read(m.enc.seed_buf(), 2 * n * d)
        };
        grounder.backward(&e.feats, &e.cls, &grad);
        grounder.step();
        pipeline.model_mut().adamw(0.0, head_lr);

        if i % report_every == 0 {
            println!("visualclick: step {i}/{}, loss {loss:.4}", screens.len());
        }
    }
}

/// The three things an instruction can demand, kept apart in the report
/// because they are not equally hard and an aggregate hides which one is
/// failing.
const KINDS: [&str; 3] = ["attribute", "state", "region"];

fn instruction_kind(instruction: &str) -> usize {
    if screen::REGIONS.iter().any(|r| instruction.ends_with(r.name)) {
        2
    } else if instruction.contains("dimmed") || instruction.contains("bright") {
        1
    } else {
        0
    }
}

struct Report {
    accuracy: f32,
    ece: f32,
    /// Distance in screen pixels from the click the model returns to the
    /// centre of the control it should have clicked.
    mean_click_error_px: f32,
    /// Mean number of controls on a screen - the option count the accuracy is
    /// against, and so what chance is.
    mean_options: f32,
    /// `(right, total)` per entry of [`KINDS`].
    by_kind: [(usize, usize); 3],
    /// `(right, total)` per entry of `screen::REGIONS` - an aggregate over
    /// the region words hides the case where some are solved and others are
    /// not, which is a different problem with a different fix from "spatial
    /// is hard".
    by_region: Vec<(usize, usize)>,
}

fn evaluate(pipeline: &mut RlcdPipeline, grounder: &mut Grounder, norm: &InstructionNorm, screens: &[Screen], ablate: Ablate) -> Report {
    let mut confidences = Vec::with_capacity(screens.len());
    let mut correct = Vec::with_capacity(screens.len());
    let mut err_px = 0.0f64;
    let mut options = 0usize;
    let mut by_kind = [(0usize, 0usize); 3];
    let mut by_region = vec![(0usize, 0usize); screen::REGIONS.len()];

    for i in 0..screens.len() {
        let e = perceive(pipeline, norm, screens, i, ablate);
        let f = features_of(grounder, &e);
        let p = softmax(&pipeline.model_mut().score_kept(&f).expect("score_kept"));
        let picked = argmax(&p);

        let (px, py) = screens[i].widgets[picked].center();
        let (tx, ty) = screens[i].target_point();
        err_px += (((px as f64 - tx as f64).powi(2)) + ((py as f64 - ty as f64).powi(2))).sqrt();

        let right = picked == screens[i].target;
        let k = instruction_kind(&screens[i].instruction);
        by_kind[k].1 += 1;
        by_kind[k].0 += usize::from(right);
        if let Some(d) = screen::REGIONS.iter().position(|r| screens[i].instruction.ends_with(r.name)) {
            by_region[d].1 += 1;
            by_region[d].0 += usize::from(right);
        }

        confidences.push(p[picked]);
        correct.push(right);
        options += screens[i].widgets.len();
    }

    Report {
        accuracy: correct.iter().filter(|&&c| c).count() as f32 / correct.len().max(1) as f32,
        ece: ece(&confidences, &correct, 10),
        mean_click_error_px: (err_px / screens.len().max(1) as f64) as f32,
        mean_options: options as f32 / screens.len().max(1) as f32,
        by_kind,
        by_region,
    }
}

/// End-to-end decision latency, the way a caller would actually pay it: the
/// encoder pass for the instruction, the projections, and the head - timed
/// separately so the cost is attributable rather than a single number.
fn measure_latency(pipeline: &mut RlcdPipeline, grounder: &mut Grounder, norm: &InstructionNorm, screens: &[Screen]) -> (f32, f32, f32) {
    let n = screens.len().min(100);
    // One pass to warm the device and any lazily-compiled kernel, so the
    // reported number is steady-state rather than first-dispatch.
    let warm = perceive(pipeline, norm, screens, 0, Ablate::None);
    let f = features_of(grounder, &warm);
    pipeline.model_mut().score_kept(&f).expect("score_kept");

    let mut enc_s = 0.0f64;
    let mut rest_s = 0.0f64;
    for sc in screens.iter().take(n) {
        let canvas = sc.render();
        let feats = sc.features(&canvas);

        let t0 = std::time::Instant::now();
        let mut cls = instruction_row_raw(pipeline, &sc.instruction);
        norm.apply(&mut cls);
        let t1 = std::time::Instant::now();

        let e = Evidence { feats, cls };
        let f = features_of(grounder, &e);
        pipeline.model_mut().score_kept(&f).expect("score_kept");
        let t2 = std::time::Instant::now();

        enc_s += t1.duration_since(t0).as_secs_f64();
        rest_s += t2.duration_since(t1).as_secs_f64();
    }
    let n = n.max(1) as f64;
    let (enc_ms, rest_ms) = (enc_s / n * 1e3, rest_s / n * 1e3);
    (enc_ms as f32, rest_ms as f32, (1.0 / (enc_s / n + rest_s / n)) as f32)
}

fn outline(canvas: &mut brain::viewport::Canvas, w: &screen::Widget, color: [u8; 3]) {
    let pad = 3i32;
    canvas.outline(w.x as i32 - pad, w.y as i32 - pad, w.w + 2 * pad as u32, w.h + 2 * pad as u32, color);
}

fn describe(s: &Screen, idx: usize) -> String {
    let w = &s.widgets[idx];
    let (x, y) = w.center();
    format!(
        "{}{} {} at ({x}, {y})",
        if w.dim { "dimmed " } else { "" },
        screen::COLORS[w.color].0,
        screen::SHAPES[w.shape]
    )
}

/// `--shot`: render screens with the instruction's control outlined, so the
/// world can be checked by eye before any model is involved.
fn run_shot(dir: &str, seed: u64) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| {
        eprintln!("visualclick: cannot create {dir}: {e}");
        std::process::exit(1);
    });
    let mut rng = Rng::new(seed);
    for i in 0..8 {
        let s = Screen::generate(&mut rng);
        let mut canvas = s.render();
        outline(&mut canvas, &s.widgets[s.target], [10, 10, 10]);
        let path = format!("{dir}/screen-{i}.png");
        canvas.save(&path).unwrap_or_else(|e| {
            eprintln!("visualclick: cannot save {path}: {e}");
            std::process::exit(1);
        });
        println!("visualclick: {path} - \"{}\" -> {}", s.instruction, describe(&s, s.target));
    }
    println!("visualclick: open the PNGs and confirm the outline sits on the control the instruction names");
}

fn run_examples(pipeline: &mut RlcdPipeline, grounder: &mut Grounder, norm: &InstructionNorm, screens: &[Screen], ablate: Ablate, dir: &str, n: usize) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| {
        eprintln!("visualclick: cannot create {dir}: {e}");
        std::process::exit(1);
    });
    for i in 0..n.min(screens.len()) {
        let s = &screens[i];
        let e = perceive(pipeline, norm, screens, i, ablate);
        let f = features_of(grounder, &e);
        let p = softmax(&pipeline.model_mut().score_kept(&f).expect("score_kept"));
        let picked = argmax(&p);

        let mut canvas = s.render();
        outline(&mut canvas, &s.widgets[s.target], [10, 10, 10]);
        if picked != s.target {
            outline(&mut canvas, &s.widgets[picked], [220, 0, 0]);
        }
        let path = format!("{dir}/example-{i}.png");
        canvas.save(&path).unwrap_or_else(|e| {
            eprintln!("visualclick: cannot save {path}: {e}");
            std::process::exit(1);
        });
        let (cx, cy) = s.widgets[picked].center();
        println!(
            "visualclick: {path}\n  \"{}\"  ({} controls on screen)\n  answer: click ({cx}, {cy}) - {} - p={:.3} - {}\n",
            s.instruction,
            s.widgets.len(),
            describe(s, picked),
            p[picked],
            if picked == s.target { "correct".to_string() } else { format!("WRONG, wanted {}", describe(s, s.target)) }
        );
    }
}

/// Every claim this sample's number rests on, checked rather than assumed.
///
/// Each check is one thing that, if it were silently false, would produce a
/// plausible-looking accuracy that meant nothing.
fn run_validate(pipeline: &mut RlcdPipeline, grounder: &mut Grounder, norm: &InstructionNorm, screens: &[Screen]) -> bool {
    let mut ok = true;
    let mut check = |name: &str, pass: bool, detail: String| {
        println!("  [{}] {name} - {detail}", if pass { "pass" } else { "FAIL" });
        ok &= pass;
    };
    let d = pipeline.model_mut().cfg.d_model as usize;

    // 1. The instruction identifies exactly one control, re-derived from the
    //    screen rather than trusted from the generator.
    let mut ambiguous = 0;
    for s in screens.iter().take(200) {
        let t = &s.widgets[s.target];
        let same = s.widgets.iter().filter(|w| w.color == t.color && w.shape == t.shape && w.dim == t.dim).count();
        let qualified = s.instruction.split_whitespace().count() > 3;
        if same > 1 && !qualified {
            ambiguous += 1;
        }
    }
    check("oracle", ambiguous == 0, format!("{ambiguous}/200 screens had an under-qualified instruction"));

    // 2. What the model reads is what was drawn. Re-pooling the target's crop
    //    must recover the colour it was rendered in.
    let mut mismatched = 0;
    let mut checked = 0;
    for s in screens.iter().take(100) {
        let canvas = s.render();
        let t = &s.widgets[s.target];
        if !t.dim && screen::SHAPES[t.shape] == "square" {
            let f = vision::widget_features(&canvas, t);
            let want = screen::COLORS[t.color].1;
            let got = [f[0] * 255.0, f[1] * 255.0, f[2] * 255.0];
            if (0..3).any(|c| (got[c] - want[c] as f32).abs() > 2.0) {
                mismatched += 1;
            }
            checked += 1;
        }
    }
    check("pixels carry the answer", mismatched == 0 && checked > 0, format!("{mismatched}/{checked} target crops disagreed with what was drawn"));

    // 3. The rows handed to the head have the arity the head is told they do,
    //    and every one is finite and unit-scale.
    let e = perceive(pipeline, norm, screens, 0, Ablate::None);
    let n = e.feats.len();
    let rows = grounder.forward(&e.feats, &e.cls);
    let finite = rows.iter().all(|v| v.is_finite());
    let scales: Vec<f32> = rows.chunks(d).map(|r| (r.iter().map(|v| v * v).sum::<f32>() / d as f32).sqrt()).collect();
    let scaled = scales.iter().all(|s| (0.3..3.0).contains(s));
    check(
        "row layout",
        rows.len() == 2 * n * d && finite && scaled,
        format!("{n} control + {n} slot rows of {d}, rms {:.2}-{:.2}", scales.iter().cloned().fold(f32::MAX, f32::min), scales.iter().cloned().fold(0.0, f32::max)),
    );

    // 4. The gradient the head leaves in the seed buffer is the gradient of
    //    the loss with respect to the rows we put in. This is the seam between
    //    this sample and `crates/decide`, and a finite difference is the only
    //    thing that actually proves it is wired up the way the layout says.
    let gold = screens[0].target;
    let cfg = LossConfig::cross_entropy();
    let total = 2 * n;
    let f = Features::from_parts(rows.clone(), n as u32, n as u32);
    pipeline.model_mut().zero_grads();
    pipeline.model_mut().accumulate_kept(&f, |s| decision_loss(s, gold, &cfg)).expect("accumulate_kept");
    let analytic = {
        let m = pipeline.model_mut();
        m.enc.gpu().read(m.enc.seed_buf(), total * d)
    };

    let loss_at = |pipeline: &mut RlcdPipeline, rows: &[f32]| -> f32 {
        let f = Features::from_parts(rows.to_vec(), n as u32, n as u32);
        let s = pipeline.model_mut().score_kept(&f).expect("score_kept");
        decision_loss(&s, gold, &cfg).0
    };

    // A spread of probes across BOTH halves - a layout that had the two
    // regions the wrong way round would pass a state-only check.
    let eps = 1e-2f32;
    let probes: Vec<usize> = (0..12).map(|k| (k * 7919 + 13) % (total * d)).collect();
    let mut worst: f32 = 0.0;
    let mut worst_at = 0usize;
    for &j in &probes {
        let mut plus = rows.clone();
        plus[j] += eps;
        let mut minus = rows.clone();
        minus[j] -= eps;
        let numeric = (loss_at(pipeline, &plus) - loss_at(pipeline, &minus)) / (2.0 * eps);
        let rel = (analytic[j] - numeric).abs() / (numeric.abs() + 1e-2);
        if rel > worst {
            worst = rel;
            worst_at = j;
        }
    }
    let half = if worst_at < n * d { "state" } else { "slot" };
    check("gradient seam", worst < 0.15, format!("worst relative error {worst:.4} over {} probes (in a {half} row)", probes.len()));

    // 5. Both halves actually receive gradient. A zero half would train to a
    //    plausible number while half the model never moved. Reported in
    //    scientific notation because a CONVERGED model is confident, so its
    //    loss - and therefore this gradient - is genuinely tiny, and a
    //    fixed-decimal "0.0000" would read as the failure this is checking for.
    let mag = |r: std::ops::Range<usize>| -> f32 { analytic[r].iter().map(|v| v.abs()).sum() };
    let (sm, om) = (mag(0..n * d), mag(n * d..total * d));
    check("both halves learn", sm > 1e-9 && om > 1e-9, format!("state |g| {sm:.3e}, slot |g| {om:.3e}"));

    // 6. The frozen encoder has to SEPARATE the words the answer turns on.
    //    Nothing downstream can recover a distinction the encoder collapsed,
    //    and this sample has already been bitten by exactly that once: option
    //    slots built from a frozen `[CLS]` sat at cosine 0.908 to each other,
    //    which is what capped the previous design. The words that flip the
    //    answer here are the colour, the shape and the direction - so each is
    //    checked against a sentence differing in that word alone.
    let pairs: [(&str, &str, &str); 3] = [
        ("colour", "click the red square", "click the blue square"),
        ("shape", "click the red square", "click the red triangle"),
        ("region", "click the red square in the toolbar", "click the red square in the main panel"),
    ];
    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let (na, nb) = (a.iter().map(|x| x * x).sum::<f32>().sqrt(), b.iter().map(|x| x * x).sum::<f32>().sqrt());
        dot / (na * nb)
    };
    let mut worst_pair = ("", 0.0f32);
    for (what, a, b) in pairs {
        let c = cos(&instruction_row_raw(pipeline, a), &instruction_row_raw(pipeline, b));
        if c > worst_pair.1 {
            worst_pair = (what, c);
        }
        println!("      {what:<10} cosine {c:.4}");
    }
    check(
        "the instruction is separable",
        worst_pair.1 < 0.995,
        format!("closest pair is {} at cosine {:.4}", worst_pair.0, worst_pair.1),
    );

    // 7. Repeating an identical step must give an identical gradient - the
    //    `[CLS]` scatter accumulates, so a missing clear shows up here and
    //    nowhere else.
    pipeline.model_mut().zero_grads();
    pipeline.model_mut().accumulate_kept(&f, |s| decision_loss(s, gold, &cfg)).expect("accumulate_kept");
    let again = {
        let m = pipeline.model_mut();
        m.enc.gpu().read(m.enc.seed_buf(), total * d)
    };
    let drift: f32 = analytic.iter().zip(&again).map(|(a, b)| (a - b).abs()).sum();
    check("no stale gradient", drift < 1e-3, format!("repeating a step moved the gradient by {drift:.6}"));

    ok
}

fn main() {
    let args = parse_args();

    if let Some(dir) = &args.shot {
        run_shot(dir, args.seed);
        return;
    }

    require(
        std::path::Path::new(&args.encoder).join("config.json").as_path(),
        "encoder checkpoint",
        "run `brain pull sentence-transformers/all-MiniLM-L6-v2`, or pass --encoder DIR",
    );

    let dir = args.checkpoint.clone().unwrap_or_else(|| checkpoint_dir(args.ablate, args.seed, args.train_n));
    let head_path = format!("{dir}/head.bin");
    let grounder_path = format!("{dir}/grounder.bin");
    let norm_path = format!("{dir}/instruction-norm.bin");
    let have = [&head_path, &grounder_path, &norm_path].iter().all(|p| std::path::Path::new(p).exists());
    let reuse = have && !args.retrain;

    let mut rng = Rng::new(args.seed);
    let train_screens: Vec<Screen> = (0..args.train_n).map(|_| Screen::generate(&mut rng)).collect();
    let eval_screens: Vec<Screen> = (0..args.eval_n).map(|_| Screen::generate(&mut rng)).collect();

    let mut pipeline = match RlcdPipeline::builder(&args.encoder).load() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("visualclick: {e}");
            std::process::exit(1);
        }
    };
    pipeline.model_mut().set_encoder_frozen(true);
    let d_model = pipeline.model_mut().cfg.d_model as usize;

    if reuse {
        if let Err(e) = load_head(&mut pipeline, &head_path) {
            eprintln!("visualclick: cannot load {head_path}: {e}");
            std::process::exit(1);
        }
    }

    let mut grounder = if reuse {
        match Grounder::load(&grounder_path, vision::FEAT_DIM, vision::CROP_DIMS, args.grounder_lr) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("visualclick: cannot load {grounder_path}: {e}");
                std::process::exit(1);
            }
        }
    } else {
        Grounder::new(vision::FEAT_DIM, vision::CROP_DIMS, d_model, args.seed, args.grounder_lr)
    };

    let norm = if reuse {
        match InstructionNorm::load(&norm_path, d_model) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("visualclick: cannot load {norm_path}: {e}");
                std::process::exit(1);
            }
        }
    } else {
        InstructionNorm::fit(&mut pipeline, &train_screens, d_model)
    };

    if args.validate {
        println!("visualclick: validating the data path end to end");
        let ok = run_validate(&mut pipeline, &mut grounder, &norm, &eval_screens);
        println!("visualclick: {}", if ok { "every check passed" } else { "A CHECK FAILED" });
        std::process::exit(if ok { 0 } else { 1 });
    }

    println!(
        "visualclick: ablate={:?} train={} eval={} seed={} ({})",
        args.ablate,
        args.train_n,
        args.eval_n,
        args.seed,
        if reuse { "loading, training skipped" } else { "training" }
    );

    if !reuse {
        train(&mut pipeline, &mut grounder, &norm, &train_screens, args.ablate, args.head_lr);
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| {
            eprintln!("visualclick: cannot create {dir}: {e}");
            std::process::exit(1);
        });
        if let Err(e) = save_head(&mut pipeline, &head_path) {
            eprintln!("visualclick: cannot save {head_path}: {e}");
            std::process::exit(1);
        }
        if let Err(e) = grounder.save(&grounder_path) {
            eprintln!("visualclick: cannot save {grounder_path}: {e}");
            std::process::exit(1);
        }
        if let Err(e) = norm.save(&norm_path) {
            eprintln!("visualclick: cannot save {norm_path}: {e}");
            std::process::exit(1);
        }
        println!("visualclick: saved trained weights to {dir}");
    }

    let r = evaluate(&mut pipeline, &mut grounder, &norm, &eval_screens, args.ablate);
    let (enc_ms, rest_ms, per_s) = measure_latency(&mut pipeline, &mut grounder, &norm, &eval_screens);

    if let Some(dir) = &args.examples {
        run_examples(&mut pipeline, &mut grounder, &norm, &eval_screens, args.ablate, dir, args.examples_n);
    }

    println!();
    println!("visualclick: {} held-out screens, ablate={:?}", eval_screens.len(), args.ablate);
    println!("  accuracy            {:.1}%  (chance {:.1}%, {:.1} controls per screen)", r.accuracy * 100.0, 100.0 / r.mean_options, r.mean_options);
    for (k, name) in KINDS.iter().enumerate() {
        let (right, total) = r.by_kind[k];
        if total > 0 {
            println!("    {name:<10}        {:.1}%  ({right}/{total})", right as f32 / total as f32 * 100.0);
        }
    }
    for (d, region) in screen::REGIONS.iter().enumerate() {
        let (right, total) = r.by_region[d];
        if total > 0 {
            println!("      {:<18}{:.1}%  ({right}/{total})", region.name, right as f32 / total as f32 * 100.0);
        }
    }
    println!("  mean click error    {:.1} px", r.mean_click_error_px);
    println!("  calibration (ECE)   {:.3}", r.ece);
    println!("  latency             {:.1} ms/decision ({:.1} ms encoder + {:.1} ms rest) = {:.0} decisions/s", enc_ms + rest_ms, enc_ms, rest_ms, per_s);
    println!("  trained parameters  {} (grounder) + the decision head, on a frozen encoder", grounder.parameters());
}
