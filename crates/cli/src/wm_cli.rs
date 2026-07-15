// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain wm …` — world-model subcommands: play (windowed SDL / headless)
//! and bench. Models: `fake` today (deterministic test model); DIAMOND lands
//! next (docs/world-models/STATUS.md).

use crate::args::Args;
use wm_core::{FakeWorldModel, WorldModel};
use wm_display::keymap::KeyChordMap;
use wm_display::pacing::{FixedTimestep, WallClock};
use wm_display::sink::{fnv1a, HashSink, HeadlessSink, PpmDirSink, TeeSink};
use wm_display::{play_loop, PlayReport, ScriptInput, SplitIo};

pub fn run_wm(args: &[String]) {
    match args.first().map(String::as_str) {
        Some("play") => run_play(&args[1..]),
        Some("bench") => run_bench(&args[1..]),
        _ => {
            eprintln!("usage: brain wm <play|bench> [options]");
            eprintln!("  play  --model fake [--fps N] [--scale N] [--seed N] [--adaptive]");
            eprintln!("        [--headless --frames N [--actions FILE | --action-seq a,b,c]");
            eprintln!("         [--dump-ppm DIR] [--hashes]]");
            eprintln!("  bench --model fake [--frames N] [--seed N]");
            std::process::exit(2);
        }
    }
}

fn build_model(name: &str, seed: u64) -> Box<dyn WorldModel> {
    match name {
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
    let print_hashes = a.take_flag("--hashes");
    let actions = parse_actions(&mut a);

    let mut model = build_model(&model_name, seed);
    let km = KeyChordMap::wasd(0);
    let max_steps = if frames > 0 { Some(frames) } else { None };

    if headless {
        let acts = actions.unwrap_or_else(|| vec![0; frames.max(1) as usize]);
        let mut input = ScriptInput::new(acts);
        let mut hashes = HashSink::default();
        // Unpaced (huge fps target): CI never sleeps.
        let pacer = FixedTimestep::new(WallClock::new(), 100_000, false);
        let report = match dump_ppm {
            Some(dir) => {
                let ppm = PpmDirSink::new(&dir).expect("cannot create --dump-ppm dir");
                let mut sink = TeeSink(std::mem::take(&mut hashes), ppm);
                let mut io = SplitIo { input: &mut input, sink: &mut sink };
                let r = play_loop(model.as_mut(), &mut io, &km, pacer, fps, &model_name, max_steps);
                hashes = sink.0;
                r
            }
            None => {
                let mut io = SplitIo { input: &mut input, sink: &mut hashes };
                play_loop(model.as_mut(), &mut io, &km, pacer, fps, &model_name, max_steps)
            }
        };
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

    // Windowed play (needs the wm-sdl build feature and a real display).
    #[cfg(feature = "wm-sdl")]
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
        let report = play_loop(
            model.as_mut(),
            &mut win,
            &km,
            FixedTimestep::new(WallClock::new(), fps, adaptive),
            fps,
            &model_name,
            max_steps,
        );
        print_report("", &report);
    }
    #[cfg(not(feature = "wm-sdl"))]
    {
        let _ = (scale, adaptive);
        eprintln!("windowed play requires building with --features wm-sdl (make build/wm); use --headless");
        std::process::exit(1);
    }
}

fn run_bench(rest: &[String]) {
    let mut a = Args::new(rest);
    let model_name = a.str_or("--model", "fake");
    let frames = a.u64_or("--frames", 200);
    let seed = a.u64_or("--seed", 7);

    let mut model = build_model(&model_name, seed);
    let mut input = ScriptInput::new((0..frames).map(|i| (i % 5) as u32).collect());
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
