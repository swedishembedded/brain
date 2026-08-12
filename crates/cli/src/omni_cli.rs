// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain omni …` - the Qwen3-Omni conversion step.
//!
//!   brain omni import --hf <HF checkpoint dir> --out omni-int8.safetensors
//!
//! Produces the brain-native, int8-quantized checkpoint the GPU-resident
//! sharded Thinker (`BRAIN_OMNI_INT8_CHECKPOINT`,
//! `omni::int8_thinker_resident`) loads. The raw HF bf16 checkpoint the
//! validation-tier `BRAIN_OMNI_HF_DIR` path reads is NOT that format - it is
//! ~2x the size and stores no packed int8 - so this conversion is the step
//! between "downloaded the weights" and "the model is GPU-resident".
//!
//! The conversion (`omni::import::import_as`) streams one tensor at a time -
//! peak host memory is roughly one tensor's f32 expansion, never the ~70 GB
//! checkpoint. Its other caller is the model store's transparent
//! auto-fetch/convert pipeline (`crate::supply`), which only fires for a
//! checkpoint brain itself downloaded; this verb is how a checkpoint already
//! on disk gets converted, matching every sibling model's `import` verb
//! (`brain qwen import`, `brain glm import`, `brain lfm import`, …).

use crate::args::{canon_verb, Args};

pub fn run_omni(argv: &[String]) {
    match argv.first().map(|s| canon_verb(s)) {
        Some("import") => import(&argv[1..]),
        other => eprintln!("usage: brain omni import --hf <dir> --out <file>  (got {other:?})"),
    }
}

fn import(argv: &[String]) {
    let mut a = Args::new(argv);
    let hf = a.take_str("--hf");
    let out = a.str_or("--out", "omni-int8.safetensors");
    let id = a.take_str("--id");
    a.finish();
    let Some(hf) = hf else {
        eprintln!("usage: brain omni import --hf <HF checkpoint dir> [--out FILE] [--id VENDOR/REPO]");
        std::process::exit(2);
    };
    let t = std::time::Instant::now();
    match omni::import::import_as(&hf, &out, id.as_deref()) {
        Ok(()) => {
            let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
            println!("wrote {out} ({:.2} GiB) in {:.1}s", bytes as f64 / (1u64 << 30) as f64, t.elapsed().as_secs_f64());
            println!("serve it GPU-resident with:  BRAIN_OMNI_INT8_CHECKPOINT={out} brain serve --dbus --openai");
        }
        Err(e) => {
            eprintln!("brain omni import: {e}");
            std::process::exit(1);
        }
    }
}
