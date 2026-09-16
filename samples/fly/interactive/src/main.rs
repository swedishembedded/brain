// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Watch a fruit fly's connectome drive its body.
//!
//! 23,000 neurons and five million synapses of *Drosophila* ventral nerve
//! cord, stepped on the GPU, driving the flybody model through the motor
//! neurons the connectome itself names - walking on the ground, or beating its
//! wings in the air, with something to go and find.
//!
//! ## What is the animal and what is standing in for the rest of it
//!
//! MANC is the VENTRAL NERVE CORD. A fly's navigation happens in its brain,
//! which this is not, and the only thing a brain sends down is a descending
//! command. So when this sample works out which way the food is and pushes
//! that into the descending population, it is standing in for the missing half
//! of the animal at exactly the point the missing half would have connected -
//! and everything below that point, the premotor circuits and the motor
//! neurons and the muscles they reach, is the published wiring.
//!
//! The wingbeat is the same kind of honesty in the other direction. Wing power
//! muscles are stretch-activated: they contract many times per motor spike and
//! drive a thorax that resonates at a frequency the thorax chooses. The
//! wingbeat is therefore GENERATED and the nervous system modulates it, which
//! is the anatomy rather than a shortcut - and `--cord-wings` hands the wings
//! back to the motor neurons for anyone who wants to see what they currently
//! produce on their own.

use std::time::Instant;

use brain::{Arena, Creature, Error, View};

/// The body's own control rate: 2 ms per control tick.
const CONTROL_HZ: f64 = 500.0;
const TARGET_FPS: u32 = 30;
const W: u32 = 960;
const H: u32 = 720;
/// A fruit fly is about 2.5 mm long, and the model works in centimetres.
const BODY_LENGTH: f64 = 0.25;

struct Args {
    connectome: String,
    body: String,
    arena: Arena,
    food: Option<[f64; 3]>,
    seek: bool,
    cord_wings: bool,
    frames: u64,
    drive: f32,
    shot: Option<String>,
    shuffled: bool,
    plastic: bool,
    throttle: Option<f32>,
    log: Option<String>,
}

fn usage() -> ! {
    eprintln!(
        "usage: sample-fly-interactive --connectome <dir> --body <fruitfly.xml> [options]

  --connectome DIR       where the MANC export is (a dir with neurons.csv.gz,
                         or a parent holding manc/ or manc-codex/)
  --body FILE            the fruit-fly MJCF - the BODY model, not a scene;
                         the floor, sky and food are generated around it
  --air                  fly instead of walk: wing aerodynamics, a finer
                         timestep, and a floor to take off from and land on
  --food X,Y,Z           put something in the world, in centimetres
  --seek                 steer towards it, instead of going straight
  --cord-wings           fly on what the wing motor neurons produce, rather
                         than on a wingbeat the throttle drives
  --drive X              starting descending command (default 1.5)
  --throttle X           hold the wings at X (0 to 1) instead of on the key,
                         so a run with nobody at the keyboard still flies
  --frames N             stop after N frames instead of running until closed
  --shot FILE            write the last frame as a PPM
  --log FILE             write one CSV row per CONTROL TICK: position,
                         velocity, attitude, wing stroke, spikes. A picture
                         shows where the fly ended up; this shows what it did
  --shuffled-connectome  the structural control: same degrees, shuffled wiring
  --plastic              let synapses change while it runs

Keys: W/S descending command, A/D turn, SPACE wing throttle (in the air),
      C (held) lesions proprioception, RETURN resets, ESC quits."
    );
    std::process::exit(2)
}

fn parse() -> Args {
    let mut a = Args {
        connectome: String::new(),
        body: String::new(),
        arena: Arena::Ground,
        food: None,
        seek: false,
        cord_wings: false,
        frames: u64::MAX,
        drive: 1.5,
        shot: None,
        shuffled: false,
        plastic: false,
        throttle: None,
        log: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--connectome" => a.connectome = value(),
            "--body" => a.body = value(),
            "--air" => a.arena = Arena::Air,
            "--food" => {
                let v: Vec<f64> = value().split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if v.len() != 3 {
                    eprintln!("--food takes three comma-separated numbers, e.g. --food 3,0,0");
                    usage();
                }
                a.food = Some([v[0], v[1], v[2]]);
            }
            "--seek" => a.seek = true,
            "--cord-wings" => a.cord_wings = true,
            "--frames" => a.frames = value().parse().unwrap_or_else(|_| usage()),
            "--drive" => a.drive = value().parse().unwrap_or_else(|_| usage()),
            "--throttle" => a.throttle = Some(value().parse().unwrap_or_else(|_| usage())),
            "--shot" => a.shot = Some(value()),
            "--log" => a.log = Some(value()),
            "--shuffled-connectome" => a.shuffled = true,
            "--plastic" => a.plastic = true,
            "-h" | "--help" => usage(),
            _ => usage(),
        }
    }
    if a.connectome.is_empty() || a.body.is_empty() {
        usage();
    }
    // Seeking nothing is a flag that silently does nothing, which is worse
    // than an error because the fly still flies and still looks right.
    if a.seek && a.food.is_none() {
        eprintln!("--seek needs somewhere to go: pass --food X,Y,Z too");
        std::process::exit(2);
    }
    a
}

/// What an RGB8 buffer actually looks like: distinct colours, mean brightness,
/// and how much of it is not nearly black.
///
/// A colour count alone was not enough and said so on the first machine it was
/// used on. Counting stops once the answer is "many", so a frame that is
/// almost entirely black with a faint sky gradient scores exactly the same as
/// a lit scene - and telling those two apart is the whole question when
/// somebody reports a black window.
fn describe(rgb: &[u8]) -> String {
    let mut seen = std::collections::HashSet::new();
    let (mut sum, mut lit) = (0u64, 0usize);
    let n = rgb.len() / 3;
    for p in rgb.chunks_exact(3) {
        if seen.len() <= 64 {
            seen.insert([p[0], p[1], p[2]]);
        }
        let l = p[0] as u64 + p[1] as u64 + p[2] as u64;
        sum += l;
        if l > 90 {
            lit += 1;
        }
    }
    format!(
        "{}{} colours, mean brightness {:.0}/255, {:.1}% of pixels lit",
        if seen.len() > 64 { ">" } else { "" },
        seen.len().min(64),
        sum as f64 / (3.0 * n.max(1) as f64),
        100.0 * lit as f64 / n.max(1) as f64
    )
}

/// An RGB8 buffer as a PPM.
fn write_ppm(path: &str, rgb: &[u8], w: u32, h: u32) -> Result<(), Error> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    write!(f, "P6\n{w} {h}\n255\n")?;
    f.write_all(rgb)?;
    Ok(())
}

/// One trajectory sample.
fn write_row(f: &mut std::fs::File, fly: &Creature, beat: &brain::Beat, throttle: f32) -> Result<(), Error> {
    use std::io::Write;
    let [x, y, z] = fly.position();
    let [vx, vy, vz] = fly.velocity();
    let (yaw, pitch) = fly.attitude();
    let w = fly.wing_angles();
    let (range, bearing) = fly.bearing_to_food().unwrap_or((f64::NAN, f64::NAN));
    writeln!(
        f,
        "{},{:.4},{x:.5},{y:.5},{z:.5},{vx:.4},{vy:.4},{vz:.4},{yaw:.4},{:.2},{:.4},{:.4},{:.4},{throttle:.3},{},{},{range:.4},{bearing:.4},{}",
        beat.tick,
        beat.tick as f64 / CONTROL_HZ,
        pitch.to_degrees(),
        w[0][0],
        w[1][0],
        w[0][2],
        beat.spikes,
        beat.motor_spikes,
        u8::from(fly.grounded()),
    )?;
    Ok(())
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
    let mut builder =
        Creature::fruit_fly().connectome(&args.connectome).body(&args.body).arena(args.arena).plasticity(args.plastic);
    if let Some(at) = args.food {
        builder = builder.food(at);
    }
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
    if fly.flying() {
        println!("{}", fly.wing_wiring());
        // Hand the wings to the cord only when asked. Nothing has trained
        // those motor neurons, so on this path the wings do very little - which
        // is a thing worth being able to SEE rather than a thing to hide.
        if args.cord_wings {
            fly.set_wing_power(None);
            println!("wings driven by the cord's own motor neurons (nothing has trained them yet)");
        } else {
            fly.set_wing_power(Some(0.0));
            println!("wings on the throttle: hold SPACE to spin up and take off");
        }
    }
    if let Some(at) = args.food {
        println!("food at {at:?}{}", if args.seek { ", steering for it" } else { "" });
    }

    let mut view = View::open(&fly, "brain fly", W, H)?;

    // One frame's worth of simulated time, run regardless of how long it
    // takes. A machine that cannot keep up shows a correct fly in slow motion
    // rather than an incorrect fly at speed, and the title bar reports the
    // ratio rather than hiding it.
    let ticks_per_frame = (CONTROL_HZ / TARGET_FPS as f64).round() as u32;
    let mut drive = args.drive;
    let mut throttle = args.throttle.unwrap_or(0.0);
    let mut eaten = false;
    fly.drive(drive);

    // One row per control tick rather than per frame. A trajectory is a thing
    // to measure, not to squint at: climb rate, turn rate, whether the wings
    // are beating at the frequency they were asked for and whether the path is
    // powered flight or a ballistic arc are all arithmetic on this file, and
    // none of them is legible in a screenshot.
    let mut log = match &args.log {
        Some(path) => {
            let mut f = std::fs::File::create(path)?;
            use std::io::Write;
            writeln!(
                f,
                "tick,t,x,y,z,vx,vy,vz,yaw,pitch_deg,stroke_l,stroke_r,aoa_l,wing_power,spikes,motor_spikes,food_range,food_bearing,grounded"
            )?;
            Some(f)
        }
        None => None,
    };

    let mut frames = 0u64;
    let start = fly.position();
    view.capture_next_frame();
    while frames < args.frames {
        frames += 1;
        let steering = view.steering();
        if steering.quit {
            break;
        }
        if steering.reset {
            fly.reset();
            drive = args.drive;
            throttle = args.throttle.unwrap_or(0.0);
            eaten = false;
        }
        if steering.faster {
            drive = (drive + 0.05).min(4.0);
        }
        if steering.slower {
            drive = (drive - 0.05).max(0.0);
        }

        // Wings: the throttle spins the thorax up and lets it spin down, which
        // is what taking off and landing actually are.
        if fly.flying() && !args.cord_wings {
            // A held throttle overrides the key, so a run with nobody at the
            // keyboard still takes off; otherwise the key spins the thorax up
            // and letting go spins it down, which is what taking off and
            // landing actually are.
            if args.throttle.is_none() {
                throttle = if steering.throttle { (throttle + 0.04).min(1.0) } else { (throttle - 0.02).max(0.0) };
            }
            fly.set_wing_power(Some(throttle));
        }

        let range = if args.seek && !eaten {
            let r = fly.seek_food(drive);
            if fly.reached_food() {
                eaten = true;
                println!("frame {frames}: reached the food");
            }
            r
        } else {
            fly.drive(drive);
            fly.turn(if steering.left {
                -1.0
            } else if steering.right {
                1.0
            } else {
                0.0
            });
            fly.bearing_to_food().map(|(r, _)| r)
        };
        fly.set_proprioception(!steering.lesion);

        let began = Instant::now();
        let beat = if let Some(f) = log.as_mut() {
            // Stepped one tick at a time so every row is a real sample rather
            // than an average over a frame.
            let mut total = brain::Beat::default();
            for _ in 0..ticks_per_frame {
                let b = fly.step()?;
                total.tick = b.tick;
                total.spikes += b.spikes;
                total.motor_spikes += b.motor_spikes;
                total.cord += b.cord;
                total.body += b.body;
                write_row(f, &fly, &b, throttle)?;
            }
            total
        } else {
            fly.step_for(ticks_per_frame)?
        };
        let simulated = began.elapsed();
        let realtime = (ticks_per_frame as f64 / CONTROL_HZ) / simulated.as_secs_f64().max(1e-9);

        let [x, y, z] = fly.position();
        let vel = fly.velocity();
        // BOTH speeds, because the instantaneous one alone is misleading and
        // was misread here already. A body jittering in place reports metres
        // per second while going nowhere; net progress from the start is what
        // says whether the animal is travelling. Measured on a walking run,
        // they differ by more than tenfold.
        let speed = (vel[0] * vel[0] + vel[1] * vel[1]).sqrt() / BODY_LENGTH;
        let elapsed = beat.tick as f64 / CONTROL_HZ;
        let net = ((x - start[0]).powi(2) + (y - start[1]).powi(2)).sqrt() / BODY_LENGTH / elapsed.max(1e-9);
        let mut status = format!(
            "drive {drive:.2} | {realtime:.2}x rt | {} spikes ({} motor) | {:+.2},{:+.2},{:+.2} | {speed:.1} BL/s now, {net:.2} net",
            beat.spikes, beat.motor_spikes, x, y, z
        );
        if fly.flying() {
            let stroke = fly.wing_angles().iter().map(|w| w[0].abs()).fold(0.0f64, f64::max);
            status += &format!(
                " | wings {:.0}% stroke {stroke:.2} rad | {}",
                100.0 * throttle,
                if fly.grounded() { "ON GROUND" } else { "AIRBORNE" }
            );
        }
        if let Some(r) = range {
            status += &format!(" | food {r:.2} cm");
        }
        if eaten {
            status += " | FED";
        }
        if !fly.proprioception() {
            status += " | NO PROPRIOCEPTION";
        }
        // Follow the animal rather than the origin. A flying fly leaves a
        // fixed frame in about a second, and an empty floor on screen reads as
        // the fly having failed rather than as the camera having been left.
        view.follow(&fly, 1.0);
        let drawing = Instant::now();
        view.show(&fly, &status)?;
        let drawn = drawing.elapsed();

        // Once, on the first frame: did the blit path carry the frame?
        //
        // This is deliberately NOT called a window check. Nothing inside the
        // process can read the window back off the display server, so a pass
        // here means the texture upload and copy are sound and says nothing
        // whatever about whether the screen is lit. When the screen is black
        // and this passes, the next thing to run is
        // `cargo run -p brain-wm-display --example window_smoke`, which puts
        // an unmistakable pattern up with no simulator behind it.
        if frames == 1 {
            // stderr, with everything else diagnostic: these lines exist to be
            // read off someone else's terminal when their screen is black, and
            // stdout is where the trajectory goes.
            eprintln!("render   -> {}", describe(view.frame()));
            match view.captured() {
                Some(blit) if blit == view.frame() => {
                    eprintln!("blit     -> carried the frame intact (says NOTHING about the screen)")
                }
                Some(blit) => {
                    eprintln!("blit     -> {}", describe(blit));
                    eprintln!("blit     -> THE BLIT PATH CORRUPTED THE FRAME");
                }
                None => eprintln!("blit     -> could not be read back"),
            }
            if let Some(path) = &args.shot {
                let blit = view.captured().map(|b| b.to_vec());
                write_ppm(&format!("{path}.render.ppm"), view.frame(), view.width(), view.height())?;
                if let Some(b) = blit {
                    write_ppm(&format!("{path}.blit.ppm"), &b, view.width(), view.height())?;
                }
                eprintln!("wrote {path}.render.ppm and {path}.blit.ppm");
            }
        }
        // How long a frame takes decides what a black window even means. A
        // window presented once and then stalled for a minute inside the
        // simulator looks exactly like a window that never presented, and the
        // two have nothing in common.
        //
        // Broken down, because "slow" is not an actionable report and the four
        // parts have nothing to do with each other: the CORD is the spiking
        // network and the device it is on, the BODY is MuJoCo's solver on one
        // CPU thread, LOOP is this sample's own sensing and bookkeeping, and
        // DRAW is the offscreen render plus the readback and blit that put it
        // on the screen. Each one is fixed by something different, and the
        // first question anyone asks about a 0.05x frame is which of them it
        // was.
        if frames == 1 || frames == 10 || frames.is_multiple_of(150) {
            let ms = |d: std::time::Duration| 1000.0 * d.as_secs_f64();
            let loop_ms = ms(simulated) - ms(beat.cord) - ms(beat.body);
            eprintln!(
                "frame {frames}: {} presented, {:.0} ms/frame, {realtime:.2}x realtime \
                 | cord {:.1} + body {:.1} + loop {loop_ms:.1} + draw {:.1} ms over {ticks_per_frame} ticks",
                view.presented(),
                ms(simulated) + ms(drawn),
                ms(beat.cord),
                ms(beat.body),
                ms(drawn),
            );
        }
        if args.frames != u64::MAX && frames.is_multiple_of(30) {
            println!("frame {frames} tick {}: {status}", beat.tick);
        }
    }

    if let Some(path) = &args.shot {
        write_ppm(path, view.frame(), view.width(), view.height())?;
        println!("{path}: {}x{} after {frames} frames", view.width(), view.height());
    }
    Ok(())
}
