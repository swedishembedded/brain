// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Export the real codec decoder to ONNX for OpenVINO/NPU execution.
//! Usage: cargo run --release -p brain-npu --example export_codec -- <codec.weights> <out.onnx> <code_len>

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: export_codec <codec.weights> <out.onnx> <code_len>");
        std::process::exit(2);
    }
    let (w, out, len) = (&a[1], &a[2], a[3].parse::<usize>().expect("code_len"));
    npu::export_codec_fp32(w, out, len).expect("export_codec_fp32");
    println!("wrote {out} (code_len={len})");
}
