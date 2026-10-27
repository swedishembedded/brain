// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

fn main() {
    let pipe = brain::DetectionPipeline::from_pretrained(std::env::args().nth(1).unwrap_or_default());
    println!("load ok: {}", pipe.is_ok());
    if let Err(e) = pipe { println!("{e}"); }
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for l in s.lines().filter(|l| l.starts_with("VmHWM") || l.starts_with("VmRSS")) { println!("{l}"); }
}
