// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain wm …` — world-model subcommands: import (torch .pt -> .safetensors),
//! play (windowed SDL / headless), bench. Models: `diamond` (pretrained
//! Atari-100k EDM world model) and `fake` (deterministic GPU-free test
//! model). See docs/models/world-models/status.md.

use crate::args::Args;
use data::episode::EpisodeDataset;
use wm_core::{FakeWorldModel, WorldModel};
use wm_diamond::DiamondWorldModel;
use wm_display::keymap::{Key, KeyChordMap, KeySet};
use wm_display::pacing::{FixedTimestep, WallClock};
use wm_display::record::RecorderSink;
use wm_display::sink::{fnv1a, FrameSink, HashSink, HeadlessSink, Hud, PpmDirSink, TeeSink};
use wm_display::{play_loop, PlayReport, ScriptInput, SplitIo};

pub fn run_wm(args: &[String]) {
    match args.first().map(String::as_str) {
        Some("play") => run_play(&args[1..]),
        Some("replay") => run_replay(&args[1..]),
        Some("bench") => run_bench(&args[1..]),
        Some("import") => run_import(&args[1..]),
        Some("finetune") => run_finetune(&args[1..]),
        Some("export") => run_export(&args[1..]),
        _ => {
            eprintln!("usage: brain wm <import|export|play|replay|bench> [options]");
            eprintln!("  import --arch diamond --src <agent.pt> --out <F.safetensors> [--actions-count N]");
            eprintln!("  export --arch diamond --weights <F.safetensors> --onnx <F.onnx>");
            eprintln!("  play  --model fake|diamond [--weights F.safetensors] [--device cpu|gpu|npu]");
            eprintln!("        [--onnx F.onnx (npu)] [--fps N] [--scale N] [--seed N] [--adaptive]");
            eprintln!("        [--record DIR (episode dataset)]");
            eprintln!("        [--headless --frames N [--actions FILE | --action-seq a,b,c]");
            eprintln!("         [--dump-ppm DIR] [--hashes]]");
            eprintln!("  replay --episode DIR [--verify --model fake|diamond [--weights F]");
            eprintln!("         [--device cpu|gpu|npu] [--onnx F.onnx] [--seed N] [--denoise-steps N]");
            eprintln!("         [--context N] [--tolerance T]]");
            eprintln!("  bench --model fake|diamond [--weights F] [--onnx F.onnx (npu)] [--frames N] [--seed N]");
            std::process::exit(2);
        }
    }
}

/// A [`FrameSink`] that may be absent: forwards when present, no-op when
/// not. Lets play keep ONE sink pipeline (`hashes -> ppm? -> recorder?`)
/// instead of a combinatorial match.
struct OptSink<S: FrameSink>(Option<S>);

impl<S: FrameSink> FrameSink for OptSink<S> {
    fn frame(&mut self, rgb: &[u8], w: u32, h: u32, hud: &Hud) {
        if let Some(s) = &mut self.0 {
            s.frame(rgb, w, h, hud);
        }
    }
}

/// Finish a `--record` session: atomically publish the dataset directory.
fn finalize_recording(rec: RecorderSink, dir: &str) {
    let n = rec.frames_recorded();
    match rec.finalize() {
        Ok(()) if n > 0 => println!("recorded {n} frames -> {dir}"),
        Ok(()) => eprintln!("record: no frames were recorded; {dir} not created"),
        Err(e) => {
            eprintln!("record: finalize failed: {e}");
            std::process::exit(1);
        }
    }
}

fn run_finetune(rest: &[String]) {
    let mut a = Args::new(rest);
    let weights = a.take_str("--weights").unwrap_or_else(|| {
        eprintln!("wm finetune needs --weights <base.safetensors>");
        std::process::exit(2);
    });
    let data_dir = a.take_str("--data").unwrap_or_else(|| {
        eprintln!("wm finetune needs --data <episode-dir>");
        std::process::exit(2);
    });
    let out = a.take_str("--out").unwrap_or_else(|| {
        eprintln!("wm finetune needs --out <tuned.safetensors>");
        std::process::exit(2);
    });
    let steps = a.u32_or("--steps", 300);
    let lr = a.f32_or("--lr", 1e-4);
    let wd = a.f32_or("--wd", 1e-2);
    let clip = a.f32_or("--clip", 1.0);
    let seed = a.u64_or("--seed", 7);
    let device = a.take_str("--device");

    let (cfg, tensors) = wm_diamond::import::load(&weights).unwrap_or_else(|e| {
        eprintln!("cannot load {weights}: {e}");
        std::process::exit(1);
    });
    let ds = data::episode::EpisodeDataset::open(std::path::Path::new(&data_dir)).unwrap_or_else(|e| {
        eprintln!("cannot open {data_dir}: {e}");
        std::process::exit(1);
    });
    let nsc = cfg.num_steps_conditioning as usize;
    let frame_len = (cfg.img_channels * cfg.h * cfg.w) as usize;

    let tr = wm_diamond::train::DiamondTrainer::from_tensors(cfg.clone(), &tensors, device.as_deref());
    println!(
        "fine-tuning {} trainable conv tensors on {data_dir} ({steps} steps, lr {lr})",
        wm_diamond::train::trainable_list(&cfg).len()
    );
    let mut drng = data::rng::Rng::new(seed ^ 0xD5);
    let sampler = move |_: &mut wm_diamond::play::NormalRng| {
        let w = ds
            .sample_window(&mut drng, nsc + 1)
            .expect("dataset has no episode long enough for nsc+1 frames");
        // Dataset frames are [0,1]; the trainer wants [-1,1]. actions[i] is
        // the action that PRODUCED frame i, so context actions (the action
        // taken AT each context frame) are actions[1..=nsc].
        let to_pm1 = |v: &f32| v * 2.0 - 1.0;
        wm_diamond::train::Transition {
            obs: w.frames_f32[..nsc * frame_len].iter().map(to_pm1).collect(),
            clean: w.frames_f32[nsc * frame_len..].iter().map(to_pm1).collect(),
            actions: w.actions[1..=nsc].to_vec(),
        }
    };
    let t0 = std::time::Instant::now();
    let (first, last) = wm_diamond::train::finetune(
        &tr,
        sampler,
        steps,
        lr,
        wd,
        Some(clip),
        seed,
        |t, loss| println!("  step {t:>5}  loss {loss:.5}"),
    );
    println!(
        "done in {:.1}s: loss {first:.5} -> {last:.5}",
        t0.elapsed().as_secs_f32()
    );
    if !last.is_finite() {
        eprintln!("training diverged (NaN/inf loss) — NOT saving. Lower --lr (batch-1 \
fine-tuning is sensitive; see docs/models/world-models/playbooks.md).");
        std::process::exit(1);
    }
    if let Err(e) = tr.save(&tensors, &out) {
        eprintln!("save failed: {e}");
        std::process::exit(1);
    }
    println!("saved {out}");
}

fn run_import(rest: &[String]) {
    let mut a = Args::new(rest);
    let arch = a.str_or("--arch", "diamond");
    if arch != "diamond" {
        eprintln!("unknown --arch '{arch}' (available: diamond)");
        std::process::exit(2);
    }
    let src = a.take_str("--src").unwrap_or_else(|| {
        eprintln!("wm import needs --src <agent.pt>");
        std::process::exit(2);
    });
    let out = a.take_str("--out").unwrap_or_else(|| {
        eprintln!("wm import needs --out <F.safetensors>");
        std::process::exit(2);
    });
    let actions = a.u32_or("--actions-count", 4);
    match wm_diamond::import::import(&src, &out, actions) {
        Ok(cfg) => println!(
            "imported {} denoiser tensors -> {out} ({}x{} img, {} actions)",
            cfg.param_list().len(),
            cfg.h,
            cfg.w,
            cfg.num_actions
        ),
        Err(e) => {
            eprintln!("import failed: {e}");
            std::process::exit(1);
        }
    }
}

/// `brain wm export`: DIAMOND `.safetensors` -> fp32 ONNX of the UNet inner model
/// (for `--device npu` play/bench via OpenVINO).
fn run_export(rest: &[String]) {
    let mut a = Args::new(rest);
    let arch = a.str_or("--arch", "diamond");
    if arch != "diamond" {
        eprintln!("unknown --arch '{arch}' (available: diamond)");
        std::process::exit(2);
    }
    let weights = a.take_str("--weights").unwrap_or_else(|| {
        eprintln!("wm export needs --weights <F.safetensors> (from `brain wm import`)");
        std::process::exit(2);
    });
    let onnx = a.take_str("--onnx").unwrap_or_else(|| {
        eprintln!("wm export needs --onnx <F.onnx>");
        std::process::exit(2);
    });
    match wm_diamond::npu::export_onnx(&weights, &onnx) {
        Ok(cfg) => println!(
            "exported diamond UNet -> {onnx} ({}x{} img, {} ctx frames, cond {}, fp32)",
            cfg.h, cfg.w, cfg.num_steps_conditioning, cfg.cond_channels
        ),
        Err(e) => {
            eprintln!("export failed: {e}");
            std::process::exit(1);
        }
    }
}

/// Atari keymap (Breakout-style 4 actions): Space=FIRE, D=RIGHT, A=LEFT.
fn atari_keymap(num_actions: u32) -> KeyChordMap {
    let mut chords = vec![];
    if num_actions > 1 {
        chords.push((KeySet::of(&[Key::Space]), 1));
    }
    if num_actions > 2 {
        chords.push((KeySet::of(&[Key::D]), 2));
        chords.push((KeySet::of(&[Key::Right]), 2));
    }
    if num_actions > 3 {
        chords.push((KeySet::of(&[Key::A]), 3));
        chords.push((KeySet::of(&[Key::Left]), 3));
    }
    KeyChordMap::new(chords, 0)
}

/// Load sorted `*.ppm` frames from a directory as trait-convention context
/// (CHW f32 [0,1], oldest first, concatenated).
fn load_seed_context(dir: &str, c: u32, h: u32, w: u32) -> Vec<f32> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("--seed-context {dir}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "ppm").unwrap_or(false))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "--seed-context {dir}: no .ppm files");
    let mut out = vec![];
    for p in paths {
        let bytes = std::fs::read(&p).unwrap();
        let img = imaging::decode(&bytes).unwrap_or_else(|e| panic!("{p:?}: {e}"));
        assert_eq!((img.w, img.h), (w, h), "{p:?}: expected {w}x{h}");
        // HWC u8 -> CHW f32 [0,1]: the cast and the permutation are separate
        // steps in `imaging` precisely so neither hides inside the other.
        let chw = imaging::pixels::hwc_to_chw(&img.to_hwc_unit(), 3, h as usize, w as usize);
        // Frames are RGB; a model configured for more channels leaves the rest zero.
        let mut padded = vec![0.0f32; (c * h * w) as usize];
        let n = chw.len().min(padded.len());
        padded[..n].copy_from_slice(&chw[..n]);
        out.extend(padded);
    }
    out
}

fn build_model(
    name: &str,
    seed: u64,
    weights: Option<&str>,
    device: Option<&str>,
    onnx: Option<&str>,
) -> Box<dyn WorldModel> {
    match name {
        "diamond" => {
            let path = weights.unwrap_or_else(|| {
                eprintln!("--model diamond needs --weights <F.safetensors> (from `brain wm import`)");
                std::process::exit(2);
            });
            // `--device npu` is consumed by main's select_backend (it is a
            // whole-graph OpenVINO path, not a gpu_core backend) and lands in
            // crate::npu_requested(), like the yolo/glm/tts subcommands.
            if crate::npu_requested() || device == Some("npu") {
                // Intel-NPU path: the UNet runs as a compiled OpenVINO graph,
                // the sampler stays host-side (wm_diamond::npu).
                let onnx_path = onnx.unwrap_or_else(|| {
                    eprintln!(
                        "--model diamond --device npu needs --onnx <F.onnx> \
                         (from `brain wm export --weights {path} --onnx F.onnx`)"
                    );
                    std::process::exit(2);
                });
                let mut m = match wm_diamond::npu::DiamondNpuWorldModel::load(path, onnx_path, seed)
                {
                    Ok(m) => Box::new(m),
                    Err(e) => {
                        // e.g. OpenVINO runtime not installed / NPU absent —
                        // NpuError's Display carries the install instructions.
                        eprintln!("cannot start the diamond NPU world model: {e}");
                        std::process::exit(1);
                    }
                };
                println!("diamond npu: compiled {onnx_path} on {}", m.device());
                m.reset(&[], &[]);
                return m;
            }
            let (cfg, tensors) = wm_diamond::import::load(path).unwrap_or_else(|e| {
                eprintln!("cannot load {path}: {e}");
                std::process::exit(1);
            });
            let unet = wm_diamond::DiamondUNet::new(cfg, &tensors, device);
            let mut m = Box::new(DiamondWorldModel::new(unet, seed));
            m.reset(&[], &[]);
            m
        }
        "fake" => {
            let mut m = Box::new(FakeWorldModel::new());
            // Deterministic start position derived from the seed via the
            // context hash contract of FakeWorldModel::reset.
            let ctx: Vec<f32> =
                (0..8).map(|i| (seed >> (i * 8)) as u8 as f32 / 255.0).collect();
            m.reset(&ctx, &[]);
            m
        }
        other => {
            eprintln!("unknown --model '{other}' (available: fake; diamond lands in P2)");
            std::process::exit(2);
        }
    }
}

fn parse_actions(a: &mut Args) -> Option<Vec<u32>> {
    let parse = |t: &str| -> u32 {
        t.trim()
            .parse::<u32>()
            .unwrap_or_else(|_| panic!("bad action id {t:?}"))
    };
    if let Some(path) = a.take_str("--actions") {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read --actions {path}: {e}"));
        Some(text.split_whitespace().map(parse).collect())
    } else {
        a.take_str("--action-seq")
            .map(|s| s.split(',').filter(|t| !t.trim().is_empty()).map(parse).collect())
    }
}

fn print_report(prefix: &str, r: &PlayReport) {
    println!(
        "{prefix}steps={} fps={:.1} work_ms_mean={:.2} work_ms_p95={:.2}",
        r.steps, r.fps, r.work_ms_mean, r.work_ms_p95
    );
}

fn run_play(rest: &[String]) {
    let mut a = Args::new(rest);
    let model_name = a.str_or("--model", "fake");
    let fps = a.u32_or("--fps", 15);
    let scale = a.u32_or("--scale", 10);
    let seed = a.u64_or("--seed", 7);
    let headless = a.take_flag("--headless");
    let frames = a.u64_or("--frames", 0);
    let adaptive = a.take_flag("--adaptive");
    let dump_ppm = a.take_str("--dump-ppm");
    let record = a.take_str("--record");
    let print_hashes = a.take_flag("--hashes");
    let actions = parse_actions(&mut a);

    let weights = a.take_str("--weights");
    let device = a.take_str("--device");
    let onnx = a.take_str("--onnx");
    let seed_ctx = a.take_str("--seed-context");
    let denoise_steps = a.u32_or("--denoise-steps", 0); // 0 = model default
    let mut model =
        build_model(&model_name, seed, weights.as_deref(), device.as_deref(), onnx.as_deref());
    if denoise_steps > 0 {
        // WorldModel::set_nfe quality codes: 0 = default (3 steps), 1 = 2
        // steps, >=2 = 1 step (see wm_diamond::play).
        model.set_nfe(match denoise_steps {
            n if n >= 3 => 0,
            2 => 1,
            _ => 2,
        });
    }
    if let Some(dir) = seed_ctx {
        let (c, h, w) = model.frame_shape();
        let ctx = load_seed_context(&dir, c, h, w);
        model.reset(&ctx, &[]);
    }
    let km = if model_name == "diamond" {
        atari_keymap(model.num_actions())
    } else {
        KeyChordMap::wasd(0)
    };
    let max_steps = if frames > 0 { Some(frames) } else { None };

    if headless {
        let acts = actions.unwrap_or_else(|| vec![0; frames.max(1) as usize]);
        let mut input = ScriptInput::new(acts);
        // Unpaced (huge fps target): CI never sleeps.
        let pacer = FixedTimestep::new(WallClock::new(), 100_000, false);
        let ppm = dump_ppm
            .as_deref()
            .map(|dir| PpmDirSink::new(dir).expect("cannot create --dump-ppm dir"));
        let rec = record
            .as_deref()
            .map(|dir| RecorderSink::new(dir, model.num_actions(), fps));
        // One pipeline: hashes always, PPM dump / recorder when requested.
        let mut sink = TeeSink(HashSink::default(), TeeSink(OptSink(ppm), OptSink(rec)));
        let report = {
            let mut io = SplitIo { input: &mut input, sink: &mut sink };
            play_loop(model.as_mut(), &mut io, &km, pacer, fps, &model_name, max_steps)
        };
        let hashes = sink.0;
        if let Some(rec) = sink.1 .1 .0 {
            finalize_recording(rec, record.as_deref().unwrap());
        }
        print_report("", &report);
        if print_hashes {
            for h in &hashes.hashes {
                println!("{h:016x}");
            }
        } else {
            let mut bytes = Vec::with_capacity(hashes.hashes.len() * 8);
            for h in &hashes.hashes {
                bytes.extend_from_slice(&h.to_le_bytes());
            }
            println!("rollout_hash={:016x}", fnv1a(&bytes));
        }
        return;
    }

    // Windowed play (opens a real SDL window; needs a display).
    {
        let (_c, h, w) = model.frame_shape();
        let mut win = match wm_display::window::SdlWindow::new("brain wm", w, h, scale) {
            Ok(win) => win,
            Err(e) => {
                eprintln!("cannot open SDL window: {e}");
                eprintln!("(headless environment? use --headless, or SDL_VIDEODRIVER=dummy)");
                std::process::exit(1);
            }
        };
        println!("controls: WASD move | Enter reset | . pause | e step | [ ] quality | Esc quit");
        let mut rec = OptSink(
            record.as_deref().map(|dir| RecorderSink::new(dir, model.num_actions(), fps)),
        );
        let mut io = WinRecIo { win: &mut win, rec: &mut rec };
        let report = play_loop(
            model.as_mut(),
            &mut io,
            &km,
            FixedTimestep::new(WallClock::new(), fps, adaptive),
            fps,
            &model_name,
            max_steps,
        );
        if let Some(r) = rec.0 {
            finalize_recording(r, record.as_deref().unwrap());
        }
        print_report("", &report);
    }
}

/// Windowed play + recording: the SDL window serves input and display while
/// an optional [`RecorderSink`] tees the frames off into an episode dataset.
struct WinRecIo<'a> {
    win: &'a mut wm_display::window::SdlWindow,
    rec: &'a mut OptSink<RecorderSink>,
}

impl wm_display::PlayIo for WinRecIo<'_> {
    fn poll(&mut self) -> wm_display::PolledInput {
        wm_display::PlayIo::poll(self.win)
    }
    fn frame(&mut self, rgb: &[u8], w: u32, h: u32, hud: &Hud) {
        FrameSink::frame(self.win, rgb, w, h, hud);
        self.rec.frame(rgb, w, h, hud);
    }
}

/// `brain wm replay`: inspect a recorded episode dataset, and optionally
/// re-generate it through a model and compare frame-by-frame.
///
/// Verification is exact-by-construction for deterministic models (fake) and
/// honest about stochastic ones (diamond): every DIAMOND step draws exactly
/// `frame_len` normals, so with the SAME `--seed` (and `--denoise-steps`)
/// used at record time the noise streams align. `--context 0` (default)
/// rebuilds the model exactly as `wm play` did and replays the FULL action
/// sequence — the noise stream aligns from frame 0. `--context N` burns N
/// steps (consuming N * frame_len normals, discarding the frames) and then
/// resets with the first N recorded frames + actions as context, so
/// verification starts at frame N with the noise stream still aligned.
/// N.B. `--context N` only matches models whose reset is a pure context
/// load (diamond); the fake model derives its state from a context HASH, so
/// verify it with `--context 0`.
fn run_replay(rest: &[String]) {
    let mut a = Args::new(rest);
    let dir = a.take_str("--episode").unwrap_or_else(|| {
        eprintln!("wm replay needs --episode <DIR> (from `brain wm play --record DIR`)");
        std::process::exit(2);
    });
    let verify = a.take_flag("--verify");
    let ds = EpisodeDataset::open(std::path::Path::new(&dir)).unwrap_or_else(|e| {
        eprintln!("cannot open {dir}: {e}");
        std::process::exit(1);
    });
    println!(
        "{dir}: {} frames ({}x{}x{} CHW u8), num_actions={} fps={} episodes={} rewards={}",
        ds.n,
        ds.c,
        ds.h,
        ds.w,
        ds.num_actions,
        ds.fps,
        ds.episodes.len(),
        if ds.rewards().is_some() { "yes" } else { "no" },
    );
    for (i, e) in ds.episodes.iter().enumerate() {
        println!("  episode {i}: frames [{}, {})", e.start, e.start + e.len);
    }
    if !verify {
        return;
    }

    let model_name = a.str_or("--model", "fake");
    let seed = a.u64_or("--seed", 7);
    let weights = a.take_str("--weights");
    let device = a.take_str("--device");
    let onnx = a.take_str("--onnx");
    let denoise_steps = a.u32_or("--denoise-steps", 0);
    let tolerance = a.f32_or("--tolerance", 2.0 / 255.0);
    let context = a.usize_or("--context", 0);
    let mut model =
        build_model(&model_name, seed, weights.as_deref(), device.as_deref(), onnx.as_deref());
    if denoise_steps > 0 {
        // Same quality-code mapping as `wm play` (see run_play).
        model.set_nfe(match denoise_steps {
            n if n >= 3 => 0,
            2 => 1,
            _ => 2,
        });
    }
    let (c, h, w) = model.frame_shape();
    if (c, h, w) != (ds.c, ds.h, ds.w) {
        eprintln!(
            "model frames are {c}x{h}x{w} but the dataset is {}x{}x{}",
            ds.c, ds.h, ds.w
        );
        std::process::exit(1);
    }
    if context >= ds.n {
        eprintln!("--context {context} leaves no frames to verify (n = {})", ds.n);
        std::process::exit(2);
    }
    let die = |e: String| -> ! {
        eprintln!("{e}");
        std::process::exit(1);
    };
    let actions = ds.actions().to_vec();
    if context > 0 {
        // Burn the context steps so a stochastic model's rng lands where it
        // was at record time, then load the recorded frames as context.
        for &act in &actions[..context] {
            let _ = model.step(act);
        }
        let mut ctx = Vec::with_capacity(context * ds.frame_len());
        for i in 0..context {
            ctx.extend(ds.frame_f32(i).unwrap_or_else(|e| die(e)));
        }
        model.reset(&ctx, &actions[..context]);
    }

    let mut max_mad = 0f32;
    let mut worst = context;
    let mut failed = 0usize;
    for (i, &action) in actions.iter().enumerate().take(ds.n).skip(context) {
        let f = model.step(action);
        // The exact conversion the recorder applied: f32 -> RGB8 -> CHW u8.
        let rgb = imaging::pixels::chw_to_rgb8(&f, w, h, c as usize, imaging::ChannelPolicy::RequireRgb)
            .unwrap_or_else(|e| die(e))
            .px;
        let got = imaging::pixels::hwc_to_chw(&rgb, 3, h as usize, w as usize);
        let want = ds.frame(i).unwrap_or_else(|e| die(e));
        let sum: u64 = got
            .iter()
            .zip(&want)
            .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
            .sum();
        let mad = sum as f32 / (got.len() as f32 * 255.0);
        if mad > max_mad {
            max_mad = mad;
            worst = i;
        }
        if mad > tolerance {
            failed += 1;
        }
    }
    let checked = ds.n - context;
    let pass = failed == 0;
    println!(
        "verified {checked} frames: max_mad={max_mad:.6} (frame {worst}), tolerance={tolerance:.6} -> {}",
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        eprintln!("{failed} of {checked} frames exceeded the tolerance");
        std::process::exit(1);
    }
}

fn run_bench(rest: &[String]) {
    let mut a = Args::new(rest);
    let model_name = a.str_or("--model", "fake");
    let frames = a.u64_or("--frames", 200);
    let seed = a.u64_or("--seed", 7);
    let profile = a.take_flag("--profile");

    if profile {
        if model_name != "diamond" {
            eprintln!("--profile is diamond-only");
            std::process::exit(2);
        }
        let weights = a.take_str("--weights").expect("--profile needs --weights");
        let device = a.take_str("--device");
        let (cfg, tensors) = wm_diamond::import::load(&weights).unwrap();
        let unet = wm_diamond::DiamondUNet::new(cfg.clone(), &tensors, device.as_deref());
        let n = (cfg.img_channels * cfg.h * cfg.w) as usize;
        let noisy = vec![0.1f32; n];
        let obs = vec![0.0f32; n * cfg.num_steps_conditioning as usize];
        unet.set_context(&obs);
        let actions = vec![0u32; cfg.num_steps_conditioning as usize];
        // Warm up (JIT compile / pipeline setup), then profile.
        let _ = unet.forward(&noisy, 0.1, &actions);
        let prof = unet.profile_forward(&noisy, 0.1, &actions);
        let total: f64 = prof.iter().map(|(_, ms, _)| ms).sum();
        println!("per-kernel (one UNet forward, ONE SUBMIT PER STEP — ranking only):");
        for (name, ms, count) in &prof {
            println!("  {name:<20} {ms:8.2} ms  {count:4} dispatches  {:5.1}%", ms / total * 100.0);
        }
        println!("  {:<20} {total:8.2} ms  (per-step submit overhead included)", "TOTAL");
        return;
    }

    let weights = a.take_str("--weights");
    let device = a.take_str("--device");
    let onnx = a.take_str("--onnx");
    let mut model =
        build_model(&model_name, seed, weights.as_deref(), device.as_deref(), onnx.as_deref());
    let na = model.num_actions() as u64;
    let mut input = ScriptInput::new((0..frames).map(|i| (i % na) as u32).collect());
    let mut sink = HeadlessSink;
    let mut io = SplitIo { input: &mut input, sink: &mut sink };
    let t0 = std::time::Instant::now();
    let report = play_loop(
        model.as_mut(),
        &mut io,
        &KeyChordMap::wasd(0),
        FixedTimestep::new(WallClock::new(), 100_000, false),
        0,
        &model_name,
        Some(frames),
    );
    let total_s = t0.elapsed().as_secs_f32();
    println!(
        "model={} frames={} total_s={:.3} ms_per_frame_mean={:.3} p95={:.3} fps={:.1}",
        model_name,
        report.steps,
        total_s,
        report.work_ms_mean,
        report.work_ms_p95,
        report.steps as f32 / total_s.max(1e-6),
    );
}
