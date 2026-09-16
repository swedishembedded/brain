// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Watch the connectome drive the fly, in a window.
//!
//! The loop is the same one the experiments run - connectome to motor neurons
//! to torque to proprioception and back - with a renderer hung off the body
//! and a keyboard on the descending command. It exists because every number
//! this crate reports is a summary of something that either looks like an
//! animal or does not, and until now nothing could look.
//!
//! ```text
//! BRAIN_CONNECTOME_DIR=... BRAIN_FLYBODY_XML=.../floor.xml \
//!   cargo run -p brain-fly --release --example watch
//! ```
//!
//! Keys: W/S raise and lower the descending command, A/D bias it left and
//! right, P lesions the proprioceptive channel, R resets, Esc quits.
use fly::{Coupling, Fly, Timing};
use mujoco::{Model, MuJoCo, Renderer};
use neuro::LifParams;
use wm_display::keymap::{Key, KeySet, UxKey};
use wm_display::sink::{FrameSink, Hud};
use wm_display::window::SdlWindow;

const W: u32 = 640;
const H: u32 = 480;
/// The body's own control rate: 2 ms per control tick.
const CONTROL_HZ: f64 = 500.0;
const TARGET_FPS: u32 = 30;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        eprintln!("${name} is not set. This example needs a connectome and a body:");
        eprintln!("  BRAIN_CONNECTOME_DIR   directory holding manc-codex/");
        eprintln!("  BRAIN_FLYBODY_XML      the flybody scene, the one WITH a floor");
        std::process::exit(2)
    })
}

fn main() {
    let mj = MuJoCo::load().expect("MuJoCo loads");
    let dir = std::path::PathBuf::from(env("BRAIN_CONNECTOME_DIR")).join("manc-codex");
    let c = connectome::load("manc", &dir.join("neurons.csv.gz"), &dir.join("connections_princeton.csv.gz"))
        .expect("the connectome loads");
    let model = Model::from_xml(&mj, env("BRAIN_FLYBODY_XML")).expect("the body loads");
    let lif = LifParams { dt_over_tau: 0.2, v_th: 1.0, r: 1.0, refrac_ticks: 1, ..LifParams::default() };
    let gpu = gpu_core::testgpu::dev(&neuro::KERNELS);
    let mut fly = Fly::new(gpu, &c, model, lif, 3e-2, None, Timing::default(), Coupling::default())
        .expect("the connectome attaches to the body");
    println!("{} neurons, {}", c.neurons.len(), fly.motor_map().summary());

    let mut win = match SdlWindow::new("brain fly", W, H, 1) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("no window: {e}");
            eprintln!("On a headless box, render single frames with brain-mujoco's render_ppm example instead.");
            std::process::exit(1)
        }
    };
    let (m, d) = fly.body();
    // Built AFTER the window, because both want the GPU and the renderer's
    // error message is the more useful of the two to hit first.
    let mut r = Renderer::for_model(&mj, m, W, H).expect("an offscreen context");
    let _ = d;

    // One frame's worth of simulated time, in control ticks. The loop runs
    // this many per frame regardless of how long it takes, so a box that
    // cannot keep up shows a correct fly in slow motion rather than an
    // incorrect fly at speed - and the HUD reports the ratio rather than
    // hiding it.
    let ticks_per_frame = (CONTROL_HZ / TARGET_FPS as f64).round() as u32;
    let mut step = 0u64;
    let mut fps = 0.0f32;

    // A bounded run, for a machine with no one sitting at it. `$FRAMES` stops
    // after that many frames and `$SHOT` writes the last one, which is what
    // makes this example checkable rather than merely runnable: the window,
    // the blit and the pixel path are all exercised under
    // `SDL_VIDEODRIVER=dummy` with no display attached.
    let limit: u64 = std::env::var("FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(u64::MAX);
    let shot = std::env::var_os("SHOT");
    let hold: f32 = std::env::var("DRIVE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let mut drive = hold;

    let mut frames = 0u64;
    let mut last_rgb = Vec::new();
    while frames < limit {
        frames += 1;
        let began = std::time::Instant::now();
        let input = win.pump();
        if input.quit {
            break;
        }
        for ux in &input.ux {
            if *ux == UxKey::Reset {
                fly.reset();
                drive = hold;
            }
        }
        if input.pressed.contains(KeySet::of(&[Key::W])) {
            drive = (drive + 0.05).min(4.0);
        }
        if input.pressed.contains(KeySet::of(&[Key::S])) {
            drive = (drive - 0.05).max(0.0);
        }
        let bias = if input.pressed.contains(KeySet::of(&[Key::A])) {
            -1.0
        } else if input.pressed.contains(KeySet::of(&[Key::D])) {
            1.0
        } else {
            0.0
        };
        // Held rather than toggled: a lesion you have to keep your finger on
        // is one you cannot forget you left enabled.
        fly.set_proprioception(!input.pressed.contains(KeySet::of(&[Key::C])));

        // The command is a standing current on the descending neurons. Half
        // are biased one way and half the other, so A/D turn the animal by
        // asymmetric drive rather than by anything steering it directly.
        let n = fly.descending_count();
        let command: Vec<f32> = (0..n)
            .map(|i| drive * if i * 2 < n { 1.0 + bias * 0.5 } else { 1.0 - bias * 0.5 })
            .collect();
        fly.set_descending(&command).expect("the command fits");

        let sim_began = std::time::Instant::now();
        let mut spikes = 0u64;
        for _ in 0..ticks_per_frame {
            let t = fly.step().expect("the loop steps");
            spikes += t.total_spikes as u64;
            step = t.control_tick;
        }
        let sim = sim_began.elapsed().as_secs_f64();
        let realtime = (ticks_per_frame as f64 / CONTROL_HZ) / sim.max(1e-9);

        let (m, d) = fly.body();
        r.render(m, d).expect("a frame");
        last_rgb = r.rgb_top_down();
        let rgb = &last_rgb;
        let z = fly.qpos().get(2).copied().unwrap_or(0.0);
        let hud = Hud {
            model: format!(
                "drive {drive:.2} bias {bias:+.1} | {}x realtime | {spikes} spikes | z {z:+.3} cm{}",
                format_args!("{realtime:.2}"),
                if fly.proprioception() { "" } else { " | NO PROPRIOCEPTION" }
            ),
            fps,
            target_fps: TARGET_FPS,
            step,
            ..Default::default()
        };
        win.frame(rgb, W, H, &hud);
        // The HUD lives in the window title, which a bounded run has nobody to
        // read. Echo it periodically so a headless run still reports what it
        // did rather than only that it finished.
        if limit != u64::MAX && frames % 30 == 0 {
            println!("frame {frames}: {}", hud.model);
        }

        let elapsed = began.elapsed().as_secs_f32();
        fps = if elapsed > 0.0 { 1.0 / elapsed } else { 0.0 };
    }

    if let Some(path) = shot {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).expect("the screenshot opens");
        write!(f, "P6\n{W} {H}\n255\n").expect("the header writes");
        f.write_all(&last_rgb).expect("the pixels write");
        println!("{}: {W}x{H} after {frames} frames", std::path::Path::new(&path).display());
    }
}
