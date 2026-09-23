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
//! blind   instruction only, no scene information       -> must read chance
//! text    scene positions serialized to text            -> the EXISTING,
//!         (Decide::score, unmodified)                      unmodified path
//! pixels  16 image-patch rows spliced into the head's    -> the NEW path:
//!         cross-attention state (Decide::accumulate_kept)    Features::from_parts
//! ```
//!
//! `blind` at chance rules out label leakage through the option names.
//! `text` well above chance proves the task, the grid framing, and the
//! training loop are sound before any new capability is on trial - if `text`
//! fails, nothing else here is worth running. `pixels` decisively above
//! `blind` is the verdict on whether the splice mechanism itself works; two
//! ablations (`--ablate noise`, `--ablate shuffle`) must both collapse
//! `pixels` back to chance, or the measurement is not trustworthy - a result
//! an ablation also produces is not evidence for the thing being tested.
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

use brain::{decision_loss, ece, Features, Flow, LossConfig, Opt, Question, RlcdExample, RlcdPipeline, RlcdSpec};
use patches::Projector;
use scene::{Rng, Scene, CELLS};

/// Shared across every arm, so `blind`/`text`/`pixels` answer literally the
/// same question over literally the same sixteen option names - only what
/// reaches the model about a given scene differs.
const INSTRUCTIONS: &str = "which grid cell should be clicked";

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
    train_n: usize,
    eval_n: usize,
    seed: u64,
    shot: Option<String>,
    head_lr: f32,
    projector_lr: f32,
}

fn parse_args() -> Args {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut a = Args {
        encoder: std::env::var("BRAIN_MINILM_DIR")
            .unwrap_or_else(|_| format!("{home}/.local/share/brain/models/sentence-transformers/all-MiniLM-L6-v2")),
        arm: Arm::Text,
        ablate: Ablate::None,
        train_n: 2000,
        eval_n: 400,
        seed: 11,
        shot: None,
        head_lr: 1e-3,
        projector_lr: 5e-3,
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
            "--train-scenes" => a.train_n = next().parse().unwrap_or(a.train_n),
            "--eval-scenes" => a.eval_n = next().parse().unwrap_or(a.eval_n),
            "--seed" => a.seed = next().parse().unwrap_or(a.seed),
            "--shot" => a.shot = Some(next()),
            "--head-lr" => a.head_lr = next().parse().unwrap_or(a.head_lr),
            "--projector-lr" => a.projector_lr = next().parse().unwrap_or(a.projector_lr),
            "--help" | "-h" => {
                eprintln!(
                    "usage: visualclick [--encoder DIR] [--arm blind|text|pixels] [--ablate none|noise|shuffle]\n                    [--train-scenes N] [--eval-scenes N] [--seed N] [--shot DIR]\n\n\
                     --shot DIR: render a handful of scenes with the oracle cell outlined, to out/, and exit - no encoder needed."
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

fn option_question() -> Question {
    Question::Choice { instructions: INSTRUCTIONS.into(), options: scene::option_names().into_iter().map(Opt::new).collect() }
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
        let (row, col) = (cell / scene::GRID, cell % scene::GRID);
        let x = col as i32 * scene::CELL_PX as i32;
        let y = row as i32 * scene::CELL_PX as i32;
        canvas.outline(x + 2, y + 2, scene::CELL_PX - 4, scene::CELL_PX - 4, [0, 0, 0]);
        // `text`'s size is a multiplier on a 5x7 base font, not a pixel
        // height - `px: 12` rendered a banner nine rows tall that blotted
        // out the whole top of a 256px canvas. `px: 1` is a 9px line.
        canvas.text(4, 4, &s.instruction(), 1, [0, 0, 0]);
        let path = format!("{dir}/scene-{i}.png");
        canvas.save(&path).unwrap_or_else(|e| {
            eprintln!("visualclick: cannot save {path}: {e}");
            std::process::exit(1);
        });
        println!("visualclick: {path} - {} - oracle cell {cell} (row {row}, col {col})", s.instruction());
    }
    println!("visualclick: open the PNGs above and confirm the black outline sits on the named color's rectangle");
}

fn generate_examples(rng: &mut Rng, arm: Arm, n: usize) -> (Vec<Scene>, Vec<RlcdExample>) {
    let mut scenes = Vec::with_capacity(n);
    let mut examples = Vec::with_capacity(n);
    for _ in 0..n {
        let s = Scene::generate(rng);
        let mut target = vec![0.0f32; CELLS];
        target[s.oracle_cell()] = 1.0;
        let state = match arm {
            Arm::Blind => s.instruction(),
            Arm::Text => s.text_state(),
            Arm::Pixels => unreachable!("pixels builds its own Features, never an RlcdExample"),
        };
        examples.push(RlcdExample::new(state, target));
        scenes.push(s);
    }
    (scenes, examples)
}

/// One arm's Features, built by encoding `instruction` through the frozen
/// text path and splicing sixteen projected image rows in after it - the
/// `pixels` arm's whole new mechanism, in one place.
fn spliced_features(pipeline: &mut RlcdPipeline, q: &Question, instruction: &str, colors: &[[f32; 3]; CELLS], projector: &Projector) -> (Features, u32, usize) {
    let (_, kept) = pipeline.model_mut().score_keeping(instruction, q).expect("score_keeping");
    let d_model = pipeline.model_mut().cfg.d_model as usize;
    let instr_rows = kept.state_rows();
    let image_rows = projector.forward_rows(colors);
    let mut hidden = Vec::with_capacity(kept.hidden().len() + image_rows.len());
    hidden.extend_from_slice(&kept.hidden()[..instr_rows as usize * d_model]);
    hidden.extend_from_slice(&image_rows);
    hidden.extend_from_slice(&kept.hidden()[instr_rows as usize * d_model..]);
    (Features::from_parts(hidden, instr_rows + CELLS as u32, kept.n_slots()), instr_rows, d_model)
}

fn ablated_colors(scene: &Scene, image_scene: &Scene, ablate: Ablate, noise_seed: u64) -> [[f32; 3]; CELLS] {
    let mut colors = image_scene.patch_colors();
    if ablate == Ablate::Noise {
        let mut rng = Rng::new(noise_seed);
        for c in colors.iter_mut() {
            *c = [rng.f32(), rng.f32(), rng.f32()];
        }
    }
    let _ = scene; // the label always comes from `scene`, never `image_scene` - see call sites
    colors
}

fn train_pixels(pipeline: &mut RlcdPipeline, projector: &mut Projector, q: &Question, scenes: &[Scene], ablate: Ablate, seed: u64, head_lr: f32) {
    pipeline.model_mut().set_encoder_frozen(true);
    let loss_cfg = LossConfig::cross_entropy();
    let mut report_every = (scenes.len() / 10).max(1);
    if report_every > 200 {
        report_every = 200;
    }
    for (i, scene) in scenes.iter().enumerate() {
        let image_scene = match ablate {
            Ablate::Shuffle => &scenes[(i + scenes.len() / 2 + 1) % scenes.len()],
            _ => scene,
        };
        let colors = ablated_colors(scene, image_scene, ablate, seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let (features, instr_rows, d_model) = spliced_features(pipeline, q, &scene.instruction(), &colors, projector);
        let gold = scene.oracle_cell();

        pipeline.model_mut().zero_grads();
        let loss = pipeline
            .model_mut()
            .accumulate_kept(&features, |scores| decision_loss(scores, gold, &loss_cfg))
            .expect("accumulate_kept");

        let new_state_rows = instr_rows as usize + CELLS;
        let grad_slab = {
            let d = pipeline.model_mut();
            let gpu = d.enc.gpu();
            let seed_buf = d.enc.seed_buf();
            gpu.read(seed_buf, new_state_rows * d_model)
        };
        let d_image_rows = &grad_slab[instr_rows as usize * d_model..new_state_rows * d_model];
        projector.accumulate(&colors, d_image_rows);
        projector.step();
        pipeline.model_mut().adamw(0.0, head_lr);

        if i % report_every == 0 {
            println!("visualclick: pixels step {i}/{}, loss {loss:.4}", scenes.len());
        }
    }
}

fn eval_pixels(pipeline: &mut RlcdPipeline, projector: &Projector, q: &Question, scenes: &[Scene], ablate: Ablate, seed: u64) -> (f32, f32) {
    let mut confidences = Vec::with_capacity(scenes.len());
    let mut correct = Vec::with_capacity(scenes.len());
    for (i, scene) in scenes.iter().enumerate() {
        let image_scene = match ablate {
            Ablate::Shuffle => &scenes[(i + scenes.len() / 2 + 1) % scenes.len()],
            _ => scene,
        };
        let colors = ablated_colors(scene, image_scene, ablate, seed ^ (i as u64).wrapping_mul(0x2545F4914F6CDD1D));
        let (features, _, _) = spliced_features(pipeline, q, &scene.instruction(), &colors, projector);
        let scores = pipeline.model_mut().score_kept(&features).expect("score_kept");
        let p = softmax(&scores);
        let predicted = argmax(&p);
        confidences.push(p[predicted]);
        correct.push(predicted == scene.oracle_cell());
    }
    let accuracy = correct.iter().filter(|&&c| c).count() as f32 / correct.len().max(1) as f32;
    (accuracy, ece(&confidences, &correct, 10))
}

fn main() {
    let args = parse_args();

    if let Some(dir) = &args.shot {
        run_shot(dir);
        return;
    }

    require(std::path::Path::new(&args.encoder).join("config.json").as_path(), "encoder checkpoint", "run `brain pull sentence-transformers/all-MiniLM-L6-v2`, or pass --encoder DIR");

    println!("visualclick: arm={:?} ablate={:?} train={} eval={} seed={}", args.arm, args.ablate, args.train_n, args.eval_n, args.seed);

    let mut rng = Rng::new(args.seed);

    match args.arm {
        Arm::Blind | Arm::Text => {
            if args.ablate != Ablate::None {
                eprintln!("visualclick: --ablate only applies to --arm pixels (blind/text never see image rows to ablate)");
                std::process::exit(1);
            }
            let (_, train) = generate_examples(&mut rng, args.arm, args.train_n);
            let (_, eval) = generate_examples(&mut rng, args.arm, args.eval_n);
            let spec = RlcdSpec::default()
                .instructions(INSTRUCTIONS.to_string())
                .options(scene::option_names())
                .train(train)
                .eval(eval)
                .loss(LossConfig::cross_entropy())
                .steps(args.train_n)
                .seed(args.seed)
                .freeze_encoder(true);
            let chain = RlcdPipeline::builder(&args.encoder).load();
            let chain = Flow::new(chain).train(spec).evaluate().report();
            if let Err(e) = chain.finish() {
                eprintln!("visualclick: {e}");
                std::process::exit(1);
            }
        }
        Arm::Pixels => {
            let (train_scenes, _) = generate_examples(&mut rng, Arm::Blind, args.train_n);
            let (eval_scenes, _) = generate_examples(&mut rng, Arm::Blind, args.eval_n);

            let mut pipeline = match RlcdPipeline::builder(&args.encoder).load() {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("visualclick: {e}");
                    std::process::exit(1);
                }
            };
            let d_model = pipeline.model_mut().cfg.d_model as usize;
            let mut projector = Projector::new(d_model, args.seed, args.projector_lr);
            let q = option_question();

            train_pixels(&mut pipeline, &mut projector, &q, &train_scenes, args.ablate, args.seed, args.head_lr);
            let (accuracy, e) = eval_pixels(&mut pipeline, &projector, &q, &eval_scenes, args.ablate, args.seed ^ 0xFEED);
            println!(
                "visualclick: pixels (ablate={:?}) held-out accuracy {:.3} (chance {:.3}), ECE {:.3}, {} scenes",
                args.ablate,
                accuracy,
                1.0 / CELLS as f32,
                e,
                eval_scenes.len()
            );
        }
    }
}
