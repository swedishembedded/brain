// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Render one frame of an MJCF model to a PPM file.
//!
//! The smallest thing that proves the offscreen renderer end to end, and the
//! one a human can actually look at:
//!
//! ```text
//! BRAIN_MUJOCO_DIR=... cargo run -p brain-mujoco --release --example render_ppm -- model.xml out.ppm
//! ```

use std::io::Write;

use mujoco::{Data, Model, MuJoCo, Renderer};

fn main() {
    let mut args = std::env::args().skip(1);
    let xml = args.next().unwrap_or_else(|| {
        eprintln!("usage: render_ppm <model.xml> [out.ppm] [settle-seconds]");
        std::process::exit(2)
    });
    let out = args.next().unwrap_or_else(|| "frame.ppm".to_string());
    let settle: f64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.0);

    let mj = MuJoCo::load().expect("MuJoCo loads");
    let model = Model::from_xml(&mj, &xml).expect("the model compiles");
    let mut data = Data::new(&model).expect("mj_makeData");
    let mut r = Renderer::for_model(&mj, &model, 640, 480).expect("an offscreen context");

    // Letting the model settle first is what makes the picture worth looking
    // at: a body rendered at t=0 is in whatever pose the file declares, which
    // for a legged model is usually mid-air.
    while data.time(&model) < settle {
        data.step(&model);
    }
    data.forward(&model);
    r.render(&model, &data).expect("a frame");

    let rgb = r.rgb_top_down();
    let mut f = std::fs::File::create(&out).expect("the output opens");
    write!(f, "P6\n{} {}\n255\n", r.width(), r.height()).expect("the header writes");
    f.write_all(&rgb).expect("the pixels write");
    println!("{out}: {}x{} at t = {:.3} s", r.width(), r.height(), data.time(&model));
}
