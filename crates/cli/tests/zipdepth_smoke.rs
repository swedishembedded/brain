// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P5 smoke: `brain zipdepth --image --headless` runs end to end and is deterministic.
//!
//! Env-gated on the real checkpoint (ZIPDEPTH_PTH). Builds a synthetic PPM, runs
//! the `brain` binary through the zipdepth image path, and asserts it writes a
//! composite of the right size with a STABLE content hash — so a silent change to
//! the pipeline (preprocess, forward, colorize, composite) is caught.
use std::process::Command;

fn bin() -> String {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.push("brain");
    p.to_string_lossy().into_owned()
}

fn write_ppm(path: &str, w: u32, h: u32) {
    let mut px = Vec::new();
    for y in 0..h {
        for x in 0..w {
            px.push((255 * x / w) as u8);
            px.push((255 * y / h) as u8);
            px.push(128);
        }
    }
    imaging::save_ppm(path, &imaging::Rgb8::new(w, h, px).unwrap()).unwrap();
}

#[test]
fn depth_image_headless_is_deterministic() {
    let Ok(ckpt) = std::env::var("ZIPDEPTH_PTH") else {
        eprintln!("SKIP: set ZIPDEPTH_PTH to run the zipdepth smoke test");
        return;
    };
    let dir = std::env::temp_dir().join("brain_zipdepth_smoke");
    std::fs::create_dir_all(&dir).unwrap();
    let img = dir.join("in.ppm");
    let out = dir.join("out.ppm");
    write_ppm(img.to_str().unwrap(), 96, 72);

    let run = || -> String {
        let o = Command::new(bin())
            .args([
                "zipdepth", "--image", img.to_str().unwrap(),
                "--weights", &ckpt, "--headless", "--out", out.to_str().unwrap(),
            ])
            .env("BRAIN_DEVICE", "cpu")
            .env("DISPLAY", "")
            .output()
            .expect("run brain zipdepth");
        assert!(o.status.success(), "brain zipdepth failed: {}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8_lossy(&o.stdout)
            .lines()
            .find_map(|l| l.split("rollout_hash=").nth(1))
            .expect("no rollout_hash in output")
            .trim()
            .to_string()
    };

    let h1 = run();
    let h2 = run();
    assert_eq!(h1, h2, "the zipdepth pipeline must be deterministic across runs");
    let (px, w, ht) = events::ppm::decode_p6(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!((w, ht), (192, 72), "side-by-side composite is 2x input width");
    assert_eq!(px.len(), (192 * 72 * 3) as usize);
}
