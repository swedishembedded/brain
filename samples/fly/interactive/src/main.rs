// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Drive a fruit fly's connectome and watch what the body does.
//!
//! 23,000 neurons and five million synapses of *Drosophila* ventral nerve
//! cord, stepped on the GPU, driving the flybody model through the motor
//! neurons the connectome itself names - with the keyboard on the descending
//! command, which is the only thing a real brain gets to send down.

use std::time::Instant;

use brain::{Creature, Error, View};

/// The body's own control rate: 2 ms per control tick.
const CONTROL_HZ: f64 = 500.0;
const TARGET_FPS: u32 = 30;
const W: u32 = 960;
const H: u32 = 720;

struct Args {
    connectome: String,
    body: String,
    frames: u64,
    drive: f32,
    shot: Option<String>,
    shuffled: bool,
    plastic: bool,
}

fn usage() -> ! {
    eprintln!(
        "usage: sample-fly-interactive --connectome <dir> --body <scene.xml> \\
         [--frames N] [--drive X] [--shot out.ppm] [--shuffled-connectome] [--plastic]

  --connectome           directory holding manc-codex/
  --body                 the flybody MJCF, the one WITH a floor
  --frames N             stop after N frames instead of running until closed
  --drive X              starting descending command (default 1.5)
  --shot FILE            write the last frame as a PPM, for a run with nobody watching
  --shuffled-connectome  run the structural control: same degrees, shuffled wiring
  --plastic              let synapses change while it runs

Keys: W/S drive, A/D turn, C (held) lesions proprioception, R resets, Esc quits."
    );
    std::process::exit(2)
}

fn parse() -> Args {
    let mut a = Args {
        connectome: String::new(),
        body: String::new(),
        frames: u64::MAX,
        drive: 1.5,
        shot: None,
        shuffled: false,
        plastic: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--connectome" => a.connectome = value(),
            "--body" => a.body = value(),
            "--frames" => a.frames = value().parse().unwrap_or_else(|_| usage()),
            "--drive" => a.drive = value().parse().unwrap_or_else(|_| usage()),
            "--shot" => a.shot = Some(value()),
            "--shuffled-connectome" => a.shuffled = true,
            "--plastic" => a.plastic = true,
            "-h" | "--help" => usage(),
            _ => usage(),
        }
    }
    if a.connectome.is_empty() || a.body.is_empty() {
        usage();
    }
    a
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Error> {
    let args = parse();

    let loading = Instant::now();
    let mut builder = Creature::fruit_fly().connectome(&args.connectome).body(&args.body).plasticity(args.plastic);
    if args.shuffled {
        builder = builder.shuffled_connectome(0x5EED);
    }
    let mut fly = builder.build()?;
    println!(
        "{} neurons, {} - loaded in {:.1} s{}",
        fly.neurons(),
        fly.wiring(),
        loading.elapsed().as_secs_f64(),
        if args.shuffled { " [SHUFFLED CONNECTOME]" } else { "" }
    );

    let mut view = View::open(&fly, "brain fly", W, H)?;

    // One frame's worth of simulated time, run regardless of how long it
    // takes. A machine that cannot keep up shows a correct fly in slow motion
    // rather than an incorrect fly at speed, and the title bar reports the
    // ratio rather than hiding it.
    let ticks_per_frame = (CONTROL_HZ / TARGET_FPS as f64).round() as u32;
    let mut drive = args.drive;
    fly.drive(drive);

    let mut frames = 0u64;
    while frames < args.frames {
        frames += 1;
        let steering = view.steering();
        if steering.quit {
            break;
        }
        if steering.reset {
            fly.reset();
            drive = args.drive;
        }
        if steering.faster {
            drive = (drive + 0.05).min(4.0);
        }
        if steering.slower {
            drive = (drive - 0.05).max(0.0);
        }
        fly.drive(drive);
        fly.turn(if steering.left {
            -1.0
        } else if steering.right {
            1.0
        } else {
            0.0
        });
        fly.set_proprioception(!steering.lesion);

        let began = Instant::now();
        let beat = fly.step_for(ticks_per_frame)?;
        let realtime = (ticks_per_frame as f64 / CONTROL_HZ) / began.elapsed().as_secs_f64().max(1e-9);

        let [x, _, z] = fly.position();
        let status = format!(
            "drive {drive:.2} | {realtime:.2}x realtime | {} spikes ({} motor) | x {x:+.3} z {z:+.3}{}",
            beat.spikes,
            beat.motor_spikes,
            if fly.proprioception() { "" } else { " | NO PROPRIOCEPTION" }
        );
        view.show(&fly, &status)?;
        if args.frames != u64::MAX && frames.is_multiple_of(30) {
            println!("frame {frames} tick {}: {status}", beat.tick);
        }
    }

    if let Some(path) = &args.shot {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        write!(f, "P6\n{} {}\n255\n", view.width(), view.height())?;
        f.write_all(view.frame())?;
        println!("{path}: {}x{} after {frames} frames", view.width(), view.height());
    }
    Ok(())
}
