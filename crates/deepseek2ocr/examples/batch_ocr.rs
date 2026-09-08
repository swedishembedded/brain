// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Batch-OCR every image in a directory through ONE resident
//! `deepseek2ocr::caps::Session` - the efficient shape a multi-page document
//! actually wants: the ~15-20 GiB weight load and JIT/pipeline compilation
//! happen ONCE, and each page after the first pays only its own encode +
//! prefill + decode. The `brain deepseek2ocr generate` CLI verb instead
//! builds a fresh composite (and re-pays that whole load) on every
//! invocation, which is the wrong way to process more than one page - this
//! example is the in-process equivalent of driving the same `Session` a
//! served `brain serve` process holds resident, without the HTTP/D-Bus
//! plumbing in between.
//!
//! Usage:
//!   cargo run --release -p brain-deepseek2ocr --example batch_ocr -- \
//!     <page-dir> [--prompt "..."] [--max-new N] [--out <dir>]
//!
//! `<page-dir>` is scanned for `*.png`/`*.jpg`/`*.jpeg` files, sorted by
//! name, and each is run through the SAME session in order. Per-page wall
//! time and token counts are printed to stderr as they complete; the decoded
//! text goes to stdout (page-numbered) and, with `--out`, one `.md` file per
//! page.

use std::path::{Path, PathBuf};
use std::time::Instant;

use capability::{Blob, Invocation, Media, Progress};
use deepseek2ocr::caps::Session;

fn read_images(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("png" | "jpg" | "jpeg" | "PNG" | "JPG" | "JPEG")))
        .collect();
    files.sort();
    files
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: batch_ocr <page-dir> [--prompt STR] [--max-new N] [--out DIR] [--weights DIR]");
        std::process::exit(2);
    }
    let page_dir = PathBuf::from(&args[1]);
    let mut prompt = "<|grounding|>Convert the document to markdown.".to_string();
    let mut max_new: i64 = 2048;
    let mut out_dir: Option<PathBuf> = None;
    let mut weights_dir: Option<String> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                prompt = args[i + 1].clone();
                i += 2;
            }
            "--max-new" => {
                max_new = args[i + 1].parse().expect("--max-new wants an integer");
                i += 2;
            }
            "--out" => {
                out_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--weights" => {
                weights_dir = Some(args[i + 1].clone());
                i += 2;
            }
            other => panic!("unknown argument {other:?}"),
        }
    }
    if let Some(d) = &out_dir {
        std::fs::create_dir_all(d).expect("creating --out dir");
    }

    let images = read_images(&page_dir);
    if images.is_empty() {
        eprintln!("no .png/.jpg files found in {}", page_dir.display());
        std::process::exit(1);
    }
    eprintln!("== batch_ocr: {} pages from {}", images.len(), page_dir.display());
    eprintln!("   prompt: {prompt:?}, max_new: {max_new}");

    let dir = weights_dir.or_else(|| brain_modelstore::default_root().map(|r| r.join("ggml-org/DeepSeek-OCR-GGUF").to_string_lossy().into_owned())).expect("no --weights and no default model store root");
    eprintln!("   weights: {dir}");

    let t_load = Instant::now();
    let session = Session::load(&dir).unwrap_or_else(|e| panic!("Session::load({dir}): {e}"));
    eprintln!("== model resident in {:.1}s (one-time cost, amortized over every page below)\n", t_load.elapsed().as_secs_f64());

    let mut total_prompt_tokens = 0i64;
    let mut total_completion_tokens = 0i64;
    let t_all = Instant::now();

    for (idx, path) in images.iter().enumerate() {
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("page").to_string();
        let img = imaging::load(path).unwrap_or_else(|e| panic!("loading {}: {e}", path.display()));
        let (hwc, w, h) = (img.to_hwc_unit(), img.w, img.h);
        let bytes: Vec<u8> = hwc.iter().flat_map(|f| f.to_le_bytes()).collect();
        let blob = Blob::new(Media::Image, bytes).with_meta(serde_json::json!({"w": w, "h": h, "c": 3}));

        let inv = Invocation::new().set("prompt", serde_json::json!(prompt)).set("max_new", serde_json::json!(max_new)).blob("image", blob);

        let t_page = Instant::now();
        let mut last_step = 0u32;
        let mut progress = |p: Progress| {
            last_step = p.step;
        };
        let result = session.generate(&inv, &mut progress);
        let elapsed = t_page.elapsed().as_secs_f64();

        match result {
            Ok(out) => {
                let text = out.blobs.get("text").map(|b| String::from_utf8_lossy(&b.bytes).into_owned()).unwrap_or_default();
                let pt = out.outputs.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
                let ct = out.outputs.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
                let finish = out.outputs.get("finish_reason").and_then(|v| v.as_str()).unwrap_or("?");
                total_prompt_tokens += pt;
                total_completion_tokens += ct;
                eprintln!(
                    "[{:>2}/{}] {name}: {elapsed:6.1}s  prompt={pt} completion={ct} finish={finish:?} (dispatched steps={last_step})",
                    idx + 1,
                    images.len()
                );
                println!("\n===== page {} ({name}) =====\n{text}", idx + 1);
                if let Some(d) = &out_dir {
                    let _ = std::fs::write(d.join(format!("{name}.md")), &text);
                }
            }
            Err(e) => {
                eprintln!("[{:>2}/{}] {name}: FAILED after {elapsed:.1}s: {e}", idx + 1, images.len());
            }
        }
    }

    let total = t_all.elapsed().as_secs_f64();
    eprintln!(
        "\n== done: {} pages in {total:.1}s ({:.1}s/page mean, model load excluded) - {total_prompt_tokens} prompt + {total_completion_tokens} completion tokens total",
        images.len(),
        total / images.len() as f64
    );
}
