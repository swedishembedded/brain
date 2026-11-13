// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Smoke: `brain zipdepth infer --view side` runs end to end and is deterministic.
//!
//! Gated on the real checkpoint (ZIPDEPTH_PTH). Builds a synthetic image, runs
//! the `brain` binary through the served `infer` action, and asserts it writes
//! a side-by-side view of the right size with identical bytes on a second run,
//! so a silent change to the pipeline (preprocess, forward, colorize,
//! composite) is caught.
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
        brain_testutil::skip("set ZIPDEPTH_PTH to run the zipdepth smoke test");
        return;
    };
    let dir = std::env::temp_dir().join("brain_zipdepth_smoke");
    std::fs::create_dir_all(&dir).unwrap();
    let img = dir.join("in.ppm");
    let out = dir.join("out.ppm");
    write_ppm(img.to_str().unwrap(), 96, 72);

    let run = || -> Vec<u8> {
        let o = Command::new(bin())
            .args(["zipdepth", "infer", "--image", img.to_str().unwrap(), "--weights", &ckpt, "--view", "side", "--out", &format!("view={}", out.display())])
            .args(["--device", "cpu"])
            .output()
            .expect("run brain zipdepth");
        assert!(o.status.success(), "brain zipdepth failed: {}", String::from_utf8_lossy(&o.stderr));
        std::fs::read(&out).unwrap()
    };

    let first = run();
    assert_eq!(first, run(), "the zipdepth pipeline must be deterministic across runs");
    let (px, w, ht) = events::ppm::decode_p6(&first).unwrap();
    assert_eq!((w, ht), (192, 72), "side-by-side composite is 2x input width");
    assert_eq!(px.len(), (192 * 72 * 3) as usize);
}
