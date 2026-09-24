// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Visual click grounding: does splicing rows the encoder never produced
//! into `Decide`'s cross-attention head let it answer a spatially grounded
//! question - "click the red rectangle" - from a rendered image, rather than
//! from text?
//!
//! Three arms answer the SAME 16-way "which grid cell" question over the
//! SAME synthetic scenes, differing only in what reaches the model:
//!
//! ```text
//! blind   nothing - the instruction alone            -> must read chance
//! text    scene positions serialized to text          -> the EXISTING,
//!         (Decide::train_step_with, unmodified)          unmodified path
//! pixels  16 image-patch rows spliced into the head's  -> the NEW path:
//!         cross-attention state (Decide::accumulate_kept)  Features::from_parts
//! ```
//!
//! **The instruction is the per-example QUESTION, not scene content, and it
//! is never put in `state`.** `Decide`'s head computes an option's query from
//! the option's OWN slot text (`"{instructions} [SEP] {option}"`), never from
//! the state it attends over - see `crates/decide/src/head.rs`. A first
//! version put "click the red rectangle" in `state` and left `instructions`
//! a shared, per-arm-fixed string; `text` then failed to learn EVEN WITH THE
//! ENCODER UNFROZEN, because every option's query was nearly
//! example-invariant and had no channel to compare retrieved state facts
//! against an instruction-stated color it never saw. Building a fresh
//! `Question` per example, with the color IN `instructions`, is the fix, and
//! `blind`/`text`/`pixels` all build the question the same way now - see
//! `question_for`.
//!
//! `blind` at chance rules out label leakage through the option names.
//! `text` well above chance proves the task, the grid framing, the question
//! shape, and the training loop are sound before any new capability is on
//! trial - if `text` fails, nothing else here is worth running. `pixels`
//! decisively above `blind` is the verdict on whether the splice mechanism
//! itself works; two ablations (`--ablate noise`, `--ablate shuffle`) must
//! both collapse `pixels` back to chance, or the measurement is not
//! trustworthy - a result an ablation also produces is not evidence for the
//! thing being tested.
//!
//! Run it:
//!
//! ```text
//! make samples/learning/visualclick/run ARGS="--shot out/scenes"
//! make samples/learning/visualclick/run ARGS="--arm blind"
//! make samples/learning/visualclick/run ARGS="--arm text"
//! make samples/learning/visualclick/run ARGS="--arm pixels"
//! make samples/learning/visualclick/run ARGS="--arm pixels --ablate noise"
//! make samples/learning/visualclick/run ARGS="--arm pixels --ablate shuffle"
//! ```
//!
//! Swedish Embedded AB builds decision systems that ground a probability in
//! whatever evidence actually bears on it, image or text, and reports
//! exactly how that claim was checked rather than assuming it. If your team
//! needs a grounded decision system audited before it ships, you can procure
//! our services by sending an email to info@swedishembedded.com.

mod patches;
mod scene;

use brain::{decision_loss, ece, Features, LossConfig, Opt, Question, RlcdPipeline};
use patches::Projector;
use scene::{Rng, Scene, CELLS};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Blind,
    Text,
    Pixels,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ablate {
    None,
    Noise,
    Shuffle,
}

struct Args {
    encoder: String,
    arm: Arm,
    ablate: Ablate,
    /// `None` resolves per-arm in `main` - see [`default_train_n`].
    train_n: Option<usize>,
    eval_n: usize,
    seed: u64,
    shot: Option<String>,
    /// Render the first `examples_n` held-out scenes with BOTH outlines (the
    /// oracle cell, and the model's own predicted cell when it differs) and
    /// print the exact state/instruction/answer that produced each one - the
    /// "render it and look at it" complement to the held-out accuracy number.
    examples: Option<String>,
    examples_n: usize,
    head_lr: f32,
    projector_lr: f32,
    /// `None` resolves to a path derived from `arm`/`ablate`/`seed`/`train_n`
    /// in `main` - see [`checkpoint_dir`]. A checkpoint is keyed on exactly
    /// the settings that change what gets trained, so two different configs
    /// can never silently reuse each other's weights.
    checkpoint: Option<String>,
    /// Train and overwrite the checkpoint even if one already exists.
    retrain: bool,
}

/// `text`/`blind` converge (a clear loss trend, well above chance) inside
/// 3000 steps. `pixels` does not: measured flat at chance through 3000, and
/// even 6000, before a clear downward trend emerges around step 4000-4400
/// and continues to a comparable held-out accuracy by 15000. That gap is the
/// cold-start cost of bootstrapping a whole new, unaligned input channel
/// (the projector) jointly with the head, against `text`'s head-only
/// bootstrap on an already richly-structured frozen encoder - see
/// `spliced_features`'s doc.
fn default_train_n(arm: Arm) -> usize {
    match arm {
        Arm::Blind | Arm::Text => 3000,
        Arm::Pixels => 15000,
    }
}

/// One directory per distinct (arm, ablate, seed, train_n) combination, so a
/// leftover checkpoint from a different configuration can never be loaded by
/// mistake - training is deterministic in every other argument this sample
/// takes, so those four are exactly what changes the weights that come out.
fn checkpoint_dir(arm: Arm, ablate: Ablate, seed: u64, train_n: usize) -> String {
    format!("out/visualclick-checkpoints/{arm:?}-{ablate:?}-seed{seed}-train{train_n}").to_lowercase()
}

fn parse_args() -> Args {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut a = Args {
        encoder: std::env::var("BRAIN_MINILM_DIR")
            .unwrap_or_else(|_| format!("{home}/.local/share/brain/models/sentence-transformers/all-MiniLM-L6-v2")),
        arm: Arm::Text,
        ablate: Ablate::None,
        train_n: None,
        eval_n: 400,
        seed: 11,
        shot: None,
        examples: None,
        examples_n: 8,
        head_lr: 1e-3,
        projector_lr: 5e-3,
        checkpoint: None,
        retrain: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let next = || argv.get(i + 1).cloned().unwrap_or_default();
        match argv[i].as_str() {
            "--encoder" => a.encoder = next(),
            "--arm" => {
                a.arm = match next().as_str() {
                    "blind" => Arm::Blind,
                    "text" => Arm::Text,
                    "pixels" => Arm::Pixels,
                    other => {
                        eprintln!("visualclick: unknown --arm {other:?}, expected blind|text|pixels");
                        std::process::exit(1);
                    }
                }
            }
            "--ablate" => {
                a.ablate = match next().as_str() {
                    "none" => Ablate::None,
                    "noise" => Ablate::Noise,
                    "shuffle" => Ablate::Shuffle,
                    other => {
                        eprintln!("visualclick: unknown --ablate {other:?}, expected none|noise|shuffle");
                        std::process::exit(1);
                    }
                }
            }
            "--train-scenes" => a.train_n = next().parse().ok(),
            "--eval-scenes" => a.eval_n = next().parse().unwrap_or(a.eval_n),
            "--seed" => a.seed = next().parse().unwrap_or(a.seed),
            "--shot" => a.shot = Some(next()),
            "--examples" => a.examples = Some(next()),
            "--examples-n" => a.examples_n = next().parse().unwrap_or(a.examples_n),
            "--head-lr" => a.head_lr = next().parse().unwrap_or(a.head_lr),
            "--projector-lr" => a.projector_lr = next().parse().unwrap_or(a.projector_lr),
            "--checkpoint" => a.checkpoint = Some(next()),
            "--retrain" => {
                a.retrain = true;
                i += 1;
                continue;
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: visualclick [--encoder DIR] [--arm blind|text|pixels] [--ablate none|noise|shuffle]\n                    [--train-scenes N] [--eval-scenes N] [--seed N] [--shot DIR]\n                    [--examples DIR] [--examples-n N] [--checkpoint DIR] [--retrain]\n\n\
                     --shot DIR: render a handful of scenes with the oracle cell outlined, to out/, and exit - no encoder needed.\n\
                     --examples DIR: after training, render the first N held-out scenes with the oracle cell outlined in black and the model's OWN predicted cell outlined in red when it differs, plus the exact state/instruction/answer printed for each.\n\
                     --checkpoint DIR: where trained weights are cached (default: a path derived from --arm/--ablate/--seed/--train-scenes under out/). If it already holds a checkpoint, training is skipped and those weights are loaded instead.\n\
                     --retrain: ignore an existing checkpoint and train (and overwrite it) anyway."
                );
                std::process::exit(0);
            }
            other => eprintln!("visualclick: ignoring unknown argument {other:?}"),
        }
        i += 2;
    }
    a
}

/// Rule 5 of samples/README.md: say what is missing and leave cleanly.
fn require(path: &std::path::Path, what: &str, remedy: &str) {
    if !path.exists() {
        eprintln!("visualclick: no {what} at {}", path.display());
        eprintln!("visualclick: {remedy}");
        std::process::exit(1);
    }
}

/// The per-example question - see the module doc on why the color lives
/// here and never in `state`.
fn question_for(scene: &Scene) -> Question {
    Question::Choice { instructions: scene.instruction(), options: scene::option_names().into_iter().map(Opt::new).collect() }
}

fn state_for(arm: Arm, scene: &Scene) -> String {
    match arm {
        Arm::Blind | Arm::Pixels => scene::BLIND_STATE.to_string(),
        Arm::Text => scene.text_state(),
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

/// One cell's screen rectangle, outlined in `color` - shared by `--shot`
/// (oracle only) and `--examples` (oracle plus, when it differs, the
/// model's own predicted cell).
fn outline_cell(canvas: &mut brain::viewport::Canvas, cell: usize, color: [u8; 3]) {
    let (row, col) = (cell / scene::GRID, cell % scene::GRID);
    let x = col as i32 * scene::CELL_PX as i32;
    let y = row as i32 * scene::CELL_PX as i32;
    canvas.outline(x + 2, y + 2, scene::CELL_PX - 4, scene::CELL_PX - 4, color);
}

fn cell_label(cell: usize) -> String {
    format!("cell {cell} (row {}, col {})", cell / scene::GRID, cell % scene::GRID)
}

/// `--shot`: render a few scenes with the oracle cell outlined in black, so
/// M1's gate is "open the PNGs and confirm the outline sits on the named
/// color" - a visual check, not a number, per the standing rule that visual
/// bugs pass every automated test.
fn run_shot(dir: &str) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| {
        eprintln!("visualclick: cannot create {dir}: {e}");
        std::process::exit(1);
    });
    let mut rng = Rng::new(0);
    for i in 0..8 {
        let s = Scene::generate(&mut rng);
        let mut canvas = s.render();
        let cell = s.oracle_cell();
        outline_cell(&mut canvas, cell, [0, 0, 0]);
        // `text`'s size is a multiplier on a 5x7 base font, not a pixel
        // height - `px: 12` rendered a banner nine rows tall that blotted
        // out the whole top of a 256px canvas. `px: 1` is a 9px line.
        canvas.text(4, 4, &s.instruction(), 1, [0, 0, 0]);
        let path = format!("{dir}/scene-{i}.png");
        canvas.save(&path).unwrap_or_else(|e| {
            eprintln!("visualclick: cannot save {path}: {e}");
            std::process::exit(1);
        });
        println!("visualclick: {path} - {} - oracle {}", s.instruction(), cell_label(cell));
    }
    println!("visualclick: open the PNGs above and confirm the black outline sits on the named color's rectangle");
}

/// One arm's Features, built by encoding `state` through the frozen text
/// path and splicing sixteen projected image rows in after it - the
/// `pixels` arm's whole new mechanism, in one place. `state` is always
/// [`scene::BLIND_STATE`] in this sample; kept as a parameter because
/// nothing here actually requires that.
/// `state` still has to be a real, non-empty string - `Decide::pack_request`
/// refuses an empty one - but its ENCODED ROWS ARE DROPPED here rather than
/// kept as extra state alongside the image rows. A first version kept them:
/// probing `head.read_grad` on the first few steps showed `wkv`'s gradient at
/// a normal scale (~0.3-0.9) while the slice of it reaching the image rows
/// specifically was two orders of magnitude smaller (~0.004-0.01) - the
/// cross-attention head was spending its budget on the placeholder text
/// (constant, familiar-looking to a frozen encoder, informationally useless)
/// rather than the image rows (novel at init, actually informative), and
/// with little gradient reaching the projector it never got the chance to
/// become useful. Dropping the placeholder rows removes that lazy
/// alternative entirely: the image rows are the ONLY state left to attend to.
fn spliced_features(pipeline: &mut RlcdPipeline, q: &Question, state: &str, colors: &[usize; CELLS], projector: &mut Projector) -> (Features, u32, usize) {
    let (_, kept) = pipeline.model_mut().score_keeping(state, q).expect("score_keeping");
    let d_model = pipeline.model_mut().cfg.d_model as usize;
    let instr_rows = kept.state_rows();
    let image_rows = projector.forward_rows(colors);
    let mut hidden = Vec::with_capacity(image_rows.len() + kept.hidden().len() - instr_rows as usize * d_model);
    hidden.extend_from_slice(&image_rows);
    hidden.extend_from_slice(&kept.hidden()[instr_rows as usize * d_model..]);
    (Features::from_parts(hidden, CELLS as u32, kept.n_slots()), instr_rows, d_model)
}

fn ablated_colors(image_scene: &Scene, ablate: Ablate, noise_seed: u64) -> [usize; CELLS] {
    let mut colors = image_scene.patch_color_index();
    if ablate == Ablate::Noise {
        let mut rng = Rng::new(noise_seed);
        for c in colors.iter_mut() {
            *c = rng.index(patches::N_COLOR_CLASSES);
        }
    }
    colors
}

/// `Ablate::Shuffle`'s image source: a fixed offset into the same slice, so
/// every scene's patches come from a DIFFERENT scene than the one supplying
/// its label - deterministic, and never `i` itself.
fn image_scene(scenes: &[Scene], i: usize, ablate: Ablate) -> &Scene {
    match ablate {
        Ablate::Shuffle => &scenes[(i + scenes.len() / 2 + 1) % scenes.len()],
        _ => &scenes[i],
    }
}

fn run_train(pipeline: &mut RlcdPipeline, mut projector: Option<&mut Projector>, arm: Arm, scenes: &[Scene], ablate: Ablate, seed: u64, head_lr: f32) {
    pipeline.model_mut().set_encoder_frozen(true);
    let loss_cfg = LossConfig::cross_entropy();
    let report_every = (scenes.len() / 10).clamp(1, 200);
    for (i, scene) in scenes.iter().enumerate() {
        let q = question_for(scene);
        let gold = scene.oracle_cell();
        let loss = match arm {
            Arm::Blind | Arm::Text => {
                let state = state_for(arm, scene);
                pipeline.model_mut().train_step_with(&state, &q, 0.0, head_lr, |scores| decision_loss(scores, gold, &loss_cfg)).expect("train_step_with")
            }
            Arm::Pixels => {
                let projector = projector.as_deref_mut().expect("pixels arm needs a projector");
                let colors = ablated_colors(image_scene(scenes, i, ablate), ablate, seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15));
                let (features, _, d_model) = spliced_features(pipeline, &q, scene::BLIND_STATE, &colors, projector);

                pipeline.model_mut().zero_grads();
                let loss = pipeline.model_mut().accumulate_kept(&features, |scores| decision_loss(scores, gold, &loss_cfg)).expect("accumulate_kept");

                // The image rows are the WHOLE state now (see
                // `spliced_features`'s doc), so they sit at `[0, CELLS)` of
                // the seed buffer's state region - no offset to compute.
                let grad_slab = {
                    let d = pipeline.model_mut();
                    let gpu = d.enc.gpu();
                    let seed_buf = d.enc.seed_buf();
                    gpu.read(seed_buf, CELLS * d_model)
                };
                let d_image_rows = &grad_slab[..CELLS * d_model];
                projector.accumulate(&colors, d_image_rows);
                projector.step();
                pipeline.model_mut().adamw(0.0, head_lr);
                loss
            }
        };
        if i % report_every == 0 {
            println!("visualclick: {arm:?} step {i}/{}, loss {loss:.4}", scenes.len());
        }
    }
}

/// One scene's raw per-cell scores, under whichever arm is active - the one
/// place `run_eval` and `run_examples` both read the model from, so a
/// reported accuracy number and a rendered example are guaranteed to come
/// from the identical call.
fn score_scene(pipeline: &mut RlcdPipeline, projector: Option<&mut Projector>, arm: Arm, scenes: &[Scene], i: usize, ablate: Ablate, seed: u64) -> Vec<f32> {
    let scene = &scenes[i];
    let q = question_for(scene);
    match arm {
        Arm::Blind | Arm::Text => {
            let state = state_for(arm, scene);
            pipeline.model_mut().score(&state, std::slice::from_ref(&q)).expect("score")[0].clone()
        }
        Arm::Pixels => {
            let projector = projector.expect("pixels arm needs a projector");
            let colors = ablated_colors(image_scene(scenes, i, ablate), ablate, seed ^ (i as u64).wrapping_mul(0x2545F4914F6CDD1D));
            let (features, _, _) = spliced_features(pipeline, &q, scene::BLIND_STATE, &colors, projector);
            pipeline.model_mut().score_kept(&features).expect("score_kept")
        }
    }
}

fn run_eval(pipeline: &mut RlcdPipeline, mut projector: Option<&mut Projector>, arm: Arm, scenes: &[Scene], ablate: Ablate, seed: u64) -> (f32, f32) {
    let mut confidences = Vec::with_capacity(scenes.len());
    let mut correct = Vec::with_capacity(scenes.len());
    for (i, scene) in scenes.iter().enumerate() {
        let scores = score_scene(pipeline, projector.as_deref_mut(), arm, scenes, i, ablate, seed);
        let p = softmax(&scores);
        let predicted = argmax(&p);
        confidences.push(p[predicted]);
        correct.push(predicted == scene.oracle_cell());
    }
    let accuracy = correct.iter().filter(|&&c| c).count() as f32 / correct.len().max(1) as f32;
    (accuracy, ece(&confidences, &correct, 10))
}

/// `--examples`: render the first `n` held-out scenes with the oracle cell
/// outlined in black and, when the model got it wrong, its own predicted
/// cell outlined in red - and print the exact state/instruction/answer that
/// produced each one, so a claimed accuracy number can be checked by eye
/// against the actual inputs and outputs it came from.
fn run_examples(pipeline: &mut RlcdPipeline, mut projector: Option<&mut Projector>, arm: Arm, scenes: &[Scene], ablate: Ablate, seed: u64, dir: &str, n: usize) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| {
        eprintln!("visualclick: cannot create {dir}: {e}");
        std::process::exit(1);
    });
    for i in 0..n.min(scenes.len()) {
        let scene = &scenes[i];
        let scores = score_scene(pipeline, projector.as_deref_mut(), arm, scenes, i, ablate, seed);
        let p = softmax(&scores);
        let predicted = argmax(&p);
        let oracle = scene.oracle_cell();

        let mut canvas = scene.render();
        outline_cell(&mut canvas, oracle, [0, 0, 0]);
        if predicted != oracle {
            outline_cell(&mut canvas, predicted, [220, 0, 0]);
        }
        canvas.text(4, 4, &scene.instruction(), 1, [0, 0, 0]);
        let path = format!("{dir}/{arm:?}-{i}.png");
        canvas.save(&path).unwrap_or_else(|e| {
            eprintln!("visualclick: cannot save {path}: {e}");
            std::process::exit(1);
        });

        let state = match arm {
            Arm::Blind => scene::BLIND_STATE.to_string(),
            Arm::Text => scene.text_state(),
            Arm::Pixels => format!("({CELLS} image-patch rows spliced into cross-attention, no text state)"),
        };
        println!(
            "visualclick: {path}\n  state: {state}\n  instructions: \"{}\"\n  oracle {}, predicted {} (p={:.3}) - {}\n",
            scene.instruction(),
            cell_label(oracle),
            cell_label(predicted),
            p[predicted],
            if predicted == oracle { "correct" } else { "WRONG" }
        );
    }
}

fn main() {
    let args = parse_args();

    if let Some(dir) = &args.shot {
        run_shot(dir);
        return;
    }

    if args.arm != Arm::Pixels && args.ablate != Ablate::None {
        eprintln!("visualclick: --ablate only applies to --arm pixels (blind/text never see image rows to ablate)");
        std::process::exit(1);
    }

    require(std::path::Path::new(&args.encoder).join("config.json").as_path(), "encoder checkpoint", "run `brain pull sentence-transformers/all-MiniLM-L6-v2`, or pass --encoder DIR");

    let train_n = args.train_n.unwrap_or_else(|| default_train_n(args.arm));

    let dir = args.checkpoint.clone().unwrap_or_else(|| checkpoint_dir(args.arm, args.ablate, args.seed, train_n));
    let head_path = format!("{dir}/head.safetensors");
    let projector_path = format!("{dir}/projector.txt");
    let have_checkpoint = std::path::Path::new(&head_path).exists() && (args.arm != Arm::Pixels || std::path::Path::new(&projector_path).exists());
    let reuse = have_checkpoint && !args.retrain;

    println!(
        "visualclick: arm={:?} ablate={:?} train={train_n} eval={} seed={} checkpoint={dir} ({})",
        args.arm,
        args.ablate,
        args.eval_n,
        args.seed,
        if reuse { "loading, training skipped" } else if have_checkpoint { "retraining, --retrain given" } else { "training, no checkpoint yet" }
    );

    let mut rng = Rng::new(args.seed);
    let train_scenes: Vec<Scene> = (0..train_n).map(|_| Scene::generate(&mut rng)).collect();
    let eval_scenes: Vec<Scene> = (0..args.eval_n).map(|_| Scene::generate(&mut rng)).collect();

    let mut builder = RlcdPipeline::builder(&args.encoder);
    if reuse {
        builder = builder.head(&head_path);
    }
    let mut pipeline = match builder.load() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("visualclick: {e}");
            std::process::exit(1);
        }
    };

    let mut projector = if args.arm == Arm::Pixels {
        let d_model = pipeline.model_mut().cfg.d_model as usize;
        if reuse {
            match Projector::load(&projector_path, args.projector_lr) {
                Ok(p) => Some(p),
                Err(e) => {
                    eprintln!("visualclick: cannot load {projector_path}: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            Some(Projector::new(d_model, args.seed, args.projector_lr))
        }
    } else {
        None
    };

    if reuse {
        pipeline.model_mut().set_encoder_frozen(true);
    } else {
        run_train(&mut pipeline, projector.as_mut(), args.arm, &train_scenes, args.ablate, args.seed, args.head_lr);
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| {
            eprintln!("visualclick: cannot create {dir}: {e}");
            std::process::exit(1);
        });
        if let Err(e) = pipeline.model_mut().save_head(&head_path) {
            eprintln!("visualclick: cannot save {head_path}: {e}");
            std::process::exit(1);
        }
        if let Some(p) = &projector {
            if let Err(e) = p.save(&projector_path) {
                eprintln!("visualclick: cannot save {projector_path}: {e}");
                std::process::exit(1);
            }
        }
        println!("visualclick: saved trained weights to {dir}");
    }

    let (accuracy, e) = run_eval(&mut pipeline, projector.as_mut(), args.arm, &eval_scenes, args.ablate, args.seed ^ 0xFEED);
    if let Some(dir) = &args.examples {
        run_examples(&mut pipeline, projector.as_mut(), args.arm, &eval_scenes, args.ablate, args.seed ^ 0xFEED, dir, args.examples_n);
    }
    println!(
        "visualclick: {:?} (ablate={:?}) held-out accuracy {:.3} (chance {:.3}), ECE {:.3}, {} scenes",
        args.arm,
        args.ablate,
        accuracy,
        1.0 / CELLS as f32,
        e,
        eval_scenes.len()
    );
}
