// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generate the two procedural clip sets the `brain-wan` LoRA finetune gates
//! train and evaluate against: a `concept/` set (magenta triangle orbiting a
//! white dot - the LoRA target) and a `distractor/` set (cyan square bouncing
//! between two walls - the held-out control), each a `data::episode` dataset
//! plus `captions.json` under `--out`.
//!
//! ```text
//! cargo run -p brain-data --example gen_wan_clips -- --out scratchpad/wan-lora-demo/clips
//! ```

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut out, mut frames, mut size, mut concept_n, mut distractor_n, mut seed, mut fps) =
        (None, 9usize, 256u32, 30usize, 30usize, 0u64, 8u32);
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> &String { args.get(i + 1).unwrap_or_else(|| panic!("{} needs a value", args[i])) };
        match args[i].as_str() {
            "--out" => out = Some(need(i).clone()),
            "--frames" => frames = need(i).parse().expect("--frames"),
            "--size" => size = need(i).parse().expect("--size"),
            "--concept-n" => concept_n = need(i).parse().expect("--concept-n"),
            "--distractor-n" => distractor_n = need(i).parse().expect("--distractor-n"),
            "--seed" => seed = need(i).parse().expect("--seed"),
            "--fps" => fps = need(i).parse().expect("--fps"),
            other => {
                eprintln!("unknown flag {other} (expected --out --frames --size --concept-n --distractor-n --seed --fps)");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    let out = out.expect("--out <dir> is required");
    let out_dir = std::path::PathBuf::from(&out);
    let concept_dir = out_dir.join("concept");
    let distractor_dir = out_dir.join("distractor");

    let concept = data::gen_clips::generate_concept_set(concept_n, frames, size, size, seed);
    // A different seed stream so the two sets never share a motion/phase draw.
    let distractor = data::gen_clips::generate_distractor_set(distractor_n, frames, size, size, seed ^ 0xD157_7AC7_0721_BEEF);

    for d in [&concept_dir, &distractor_dir] {
        if d.exists() {
            std::fs::remove_dir_all(d).unwrap_or_else(|e| panic!("clearing {}: {e}", d.display()));
        }
    }
    let n1 = data::videoset::write_clipset(&concept_dir, &concept, size, size, fps).expect("write concept set");
    let n2 = data::videoset::write_clipset(&distractor_dir, &distractor, size, size, fps).expect("write distractor set");
    println!("wrote {n1} concept clips ({frames} frames, {size}x{size}) -> {}", concept_dir.display());
    println!("wrote {n2} distractor clips ({frames} frames, {size}x{size}) -> {}", distractor_dir.display());
}
