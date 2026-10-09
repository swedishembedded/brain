// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain lpips distance` end to end: two image files in, JSON out. Needs the
//! LPIPS weights in the model store and skips, by name, without them.

use std::process::Command;

fn write_ppm(path: &std::path::Path, shade: u8) {
    let (w, h) = (48usize, 48usize);
    let mut bytes = format!("P6\n{w} {h}\n255\n").into_bytes();
    bytes.extend((0..w * h).flat_map(|p| [(p % w * 5) as u8 ^ shade, (p / w * 5) as u8, shade]));
    std::fs::write(path, bytes).unwrap();
}

fn distance(a: &std::path::Path, b: &std::path::Path) -> f64 {
    let out = Command::new(env!("CARGO_BIN_EXE_brain"))
        .env("BRAIN_DEVICE", "cpu")
        .args(["lpips", "distance", "--json", "--in"])
        .arg(format!("a={}", a.display()))
        .arg("--in")
        .arg(format!("b={}", b.display()))
        .output()
        .expect("run brain");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    let json: serde_json::Value = serde_json::from_str(stdout.lines().last().expect("a JSON line")).expect("JSON on stdout");
    json["distance"].as_f64().expect("scalar distance")
}

#[test]
fn distance_is_zero_for_the_same_file_and_positive_for_different_ones() {
    if let Err(e) = lpips::spec::resolve() {
        brain_testutil::skip(&format!("LPIPS weights not in the model store ({e})"));
        return;
    }
    let dir = std::env::temp_dir().join(format!("brain-lpips-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (a, b) = (dir.join("a.ppm"), dir.join("b.ppm"));
    write_ppm(&a, 0);
    write_ppm(&b, 90);
    assert_eq!(distance(&a, &a), 0.0);
    assert!(distance(&a, &b) > 0.0);
    std::fs::remove_dir_all(&dir).ok();
}
