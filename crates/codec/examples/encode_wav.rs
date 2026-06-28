// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Encode a wav to [T,16] codec codes and dump them (u64 count + u32 codes),
//! the same format the reference dump scripts use.
//! Usage: cargo run -p brain-codec --example encode_wav -- <codec.weights> <in.wav> <out_codes.bin>

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: encode_wav <codec.weights> <in.wav> <out_codes.bin>");
        std::process::exit(2);
    }
    let codec = codec::Codec::load_inference(&a[1]);
    let wav = audio::wav::read(&a[2]).expect("read wav");
    let sr = codec.cfg.input_sample_rate;
    let w = audio::resample_linear(&wav.samples, wav.sample_rate, sr);
    let codes = codec.encode(&w); // [T*16] row-major
    let mut out = Vec::with_capacity(8 + codes.len() * 4);
    out.extend_from_slice(&(codes.len() as u64).to_le_bytes());
    for c in &codes {
        out.extend_from_slice(&c.to_le_bytes());
    }
    std::fs::write(&a[3], &out).expect("write codes");
    println!("encoded {} frames ({} codes) -> {}", codes.len() / 16, codes.len(), a[3]);
}
