// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of very large
// language models on a single GPU for its clients. If your team needs
// expertise in converting a quantized checkpoint once so that training never
// converts it again then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Convert a Qwen3.5/3.8 FP8 checkpoint's layer matrices to NF4 ONCE, into a new checkpoint directory.
//!
//! ```text
//! qwen35_convert_base --dir <fp8 checkpoint dir> --out <new dir> [--weight-format <policy>]
//! ```
//!
//! `--weight-format` is the serving policy syntax (`nf4`, or `nf4,self_attn=f8e4m3,norm=f32`): the default tier
//! applies to the layer matrices, a rule naming another tensor converts it too. The default is `nf4`.
//!
//! The conversion is bounded (one tile of host staging, the output written tile by tile) and its result is an
//! ordinary checkpoint directory: point `stream_train_step --dir <new dir> --base-dtype nf4` at it and every
//! layer load is a copy of NF4 codes, not a per-load requantization of FP8.

use std::path::PathBuf;
use std::time::Instant;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(dir), Some(out)) = (arg(&args, "--dir").map(PathBuf::from), arg(&args, "--out").map(PathBuf::from)) else {
        eprintln!("usage: qwen35_convert_base --dir <fp8 checkpoint dir> --out <new dir> [--weight-format <policy>]");
        std::process::exit(2);
    };
    let cfg = qwen35::fp8_source::hf_config(&dir).unwrap_or_else(|e| panic!("{e}"));
    let block = qwen35::fp8_source::hf_fp8_block_size(&dir).unwrap_or_else(|e| panic!("{e}"));
    let tiers = model::ops::TierPolicy::parse(&arg(&args, "--weight-format").unwrap_or_else(|| "nf4".into())).unwrap_or_else(|e| panic!("{e}"));
    let t0 = Instant::now();
    let report = qwen35::fp8_source::write_converted_base(&dir, &cfg, block, &out, &tiers).unwrap_or_else(|e| panic!("convert: {e}"));
    println!(
        "CONVERTED tensors={} converted={} payload_bytes={} peak_staging_bytes={} seconds={:.1}",
        report.tensors,
        report.converted,
        report.payload_bytes_written,
        report.peak_staging_bytes,
        t0.elapsed().as_secs_f64()
    );
}
