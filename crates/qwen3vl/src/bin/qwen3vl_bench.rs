// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-VL profiler: where a caption's time goes, stage by stage, against the
//! machine's own MEASURED roofline.
//!
//! Two modes, because the two questions have different costs to ask:
//!
//! * `vision` -- the ViT tower and the PatchMerger at the real 4B geometry on
//!   **random weights**. Tower cost depends only on shape, so this needs no
//!   checkpoint and runs in seconds, which is what makes it the loop an
//!   optimisation is actually iterated in. Both backends, per-kernel device
//!   table, and the analytic FLOP count next to the measured roof.
//! * `caption` -- the real checkpoint end to end, through
//!   [`qwen3vl::caps::generate_profiled`] so the numbers describe the SHIPPED
//!   path (same resident, same preprocessing, same prompt assembly) rather
//!   than a copy of it. Model build is reported separately from the per-image
//!   cost, because one is paid once per process and the other once per image
//!   and folding them together hides which one is the problem.
//!
//! Measurement discipline (repo lessons, see AGENTS.md): warm-up never enters
//! the statistics, results are best-of-N with N stated, and nothing samples
//! `nvidia-smi` during a timed run.
//!
//! Swedish Embedded AB implements performance engineering for on-device neural
//! networks. If your team needs a vision-language model to fit a latency or
//! power budget on real hardware then you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::collections::HashMap;
use std::time::Instant;

use gpu_core::{roof, Gpu};
use qwen3vl::config::{Qwen3VlConfig, VisionConfig};
use qwen3vl::encoder::{vision_pipelines, PatchMerger, VisionEncoder, BLOCK_LEAVES};
use qwen3vl::preprocess::{image_token_count, patch_grid, smart_resize, DEFAULT_MIN_PIXELS};

const USAGE: &str = "usage: qwen3vl_bench <mode> [options]
  vision  [--pixels N] [--reps N] [--device cpu|gpu|both]
          ViT tower + PatchMerger at the real 4B geometry, random weights.
  caption [--image FILE] [--pixels N] [--max-new N] [--reps N]
          the real checkpoint end to end ($BRAIN_QWEN3VL_WEIGHTS), per stage.";

/// The captioner's own resident pixel budget (`qwen3vl::captioner`), so the
/// default profile is the shape `brain label images` actually runs.
const CAPTION_MAX_PIXELS: u32 = 1280 * 1280;

// ---------------------------------------------------------------------------
// Analytic cost models. FLOP = 2 x MAC throughout.
// ---------------------------------------------------------------------------

/// FLOPs of one ViT forward over `n` patches: patch embed, then per block
/// qkv/proj/mlp linears plus the two attention GEMMs (scores and apply, each
/// `2*n*n*dim` summed over heads).
fn vit_flops(v: &VisionConfig, n: u64) -> u64 {
    let c = v.hidden as u64;
    let mlp = v.intermediate as u64;
    let pv = v.patch_vec_dim() as u64;
    let per_block = 2 * n * c * (3 * c) + 2 * n * c * c + 2 * n * c * mlp + 2 * n * mlp * c + 4 * n * n * c;
    2 * n * pv * c + v.depth as u64 * per_block
}

/// Weight bytes one ViT forward must read at least once (fp32). Activations
/// are excluded: at these shapes they are re-read many times inside the GEMM
/// tiles, so counting them once would understate traffic and counting the
/// re-reads would just re-derive the cache hierarchy. This is the DENOMINATOR
/// for "is the tower bandwidth-bound", and at `n` in the thousands it is not
/// close.
fn vit_weight_bytes(v: &VisionConfig) -> u64 {
    let c = v.hidden as u64;
    let mlp = v.intermediate as u64;
    let per_block = 3 * c * c + c * c + 2 * c * mlp;
    4 * (v.patch_vec_dim() as u64 * c + v.depth as u64 * per_block)
}

/// FLOPs of one PatchMerger over `n` patches (`ln -> fc1 -> gelu -> fc2`).
fn merger_flops(v: &VisionConfig, n: u64) -> u64 {
    let m2 = (v.spatial_merge_size * v.spatial_merge_size) as u64;
    let merged = v.hidden as u64 * m2;
    let rows = n / m2;
    2 * rows * merged * merged + 2 * rows * merged * v.out_hidden_size as u64
}

/// Decoder parameter count (fp32 weights are 4x this), for the per-token
/// bandwidth floor a batch-1 decode step cannot go under.
fn decoder_params(cfg: &qwen3::QwenConfig) -> u64 {
    let d = cfg.d_model as u64;
    let hd = cfg.head_dim as u64;
    let attn = d * hd * cfg.n_heads as u64 + 2 * d * hd * cfg.n_kv_heads as u64 + hd * cfg.n_heads as u64 * d;
    let mlp = 3 * d * cfg.d_ff as u64;
    cfg.n_layers as u64 * (attn + mlp) + cfg.vocab as u64 * d
}

/// The per-token weight-bandwidth ceiling of a batch-1 decode step: every
/// weight is read once per token, so no kernel however good can beat
/// `bandwidth / weight_bytes`. Printed next to the measured rate because a
/// token rate alone says nothing - the same number is excellent on one card
/// and a tenth of roof on another.
fn decode_ceiling_tok_s(params: u64, r: Option<roof::Roofs>) -> Option<(f64, f64)> {
    let bw = r?.gbs as f64 * 1e9;
    Some((bw / (params as f64 * 4.0), bw / params as f64))
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// One stage row: measured seconds against the roof for its own cost model.
fn report(label: &str, secs: f64, flops: u64, bytes: u64, r: Option<roof::Roofs>) {
    let gflops = flops as f64 / secs.max(1e-12) / 1e9;
    let gbs = bytes as f64 / secs.max(1e-12) / 1e9;
    match r {
        Some(r) => {
            let bound = r.bound_of(flops, 0, bytes);
            // `utilisation_of` returns a PERCENT, not a fraction - scaling it
            // again reported 1.6% of roof as 162%, which is the kind of
            // impossible number the meter, not the kernel, is responsible for.
            let pct = r.utilisation_of(flops, 0, bytes, secs).unwrap_or(0.0);
            println!(
                "  {label:<22} {:>9.1} ms  {gflops:>8.1} GFLOP/s  {gbs:>7.1} GB/s  {pct:>5.1}% of roof  [{}]",
                secs * 1e3,
                bound.as_str()
            );
        }
        None => println!("  {label:<22} {:>9.1} ms  {gflops:>8.1} GFLOP/s  {gbs:>7.1} GB/s  (no measured roof)", secs * 1e3),
    }
}

fn print_kernel_table(label: &str, rows: &[(String, f64, u64)]) {
    if rows.is_empty() {
        println!("{label}: this backend does not report per-kernel device time");
        return;
    }
    let total: f64 = rows.iter().map(|r| r.1).sum();
    println!("{label} (sum of per-kernel device time {total:.1} ms):");
    for (name, ms, calls) in rows.iter().take(14) {
        println!("  {name:<26} {ms:>9.1} ms  {calls:>7} calls  ({:>4.1}%)", 100.0 * ms / total);
    }
}

// ---------------------------------------------------------------------------
// vision mode
// ---------------------------------------------------------------------------

/// Random weights at exactly the key set [`VisionEncoder::new`] requires. Not a
/// checkpoint: the tower's cost is a function of shape alone, and a value-free
/// profile is the only one that can be run in seconds.
fn random_vision_weights(v: &VisionConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = data::rng::Rng::new(seed);
    let mut fill = |n: usize| -> Vec<f32> { (0..n).map(|_| (rng.next_f32() - 0.5) * 0.05).collect() };
    let (c, pv, mlp) = (v.hidden as usize, v.patch_vec_dim() as usize, v.intermediate as usize);
    let mut w = HashMap::new();
    w.insert("patch_embed.weight".to_string(), fill(c * pv));
    w.insert("patch_embed.bias".to_string(), fill(c));
    w.insert("pos_embed".to_string(), fill(v.num_position_embeddings as usize * c));
    for b in 0..v.depth {
        for leaf in BLOCK_LEAVES {
            let n = match *leaf {
                "qkv.weight" => 3 * c * c,
                "qkv.bias" => 3 * c,
                "proj.weight" => c * c,
                "fc1.weight" => mlp * c,
                "fc1.bias" => mlp,
                "fc2.weight" => c * mlp,
                _ => c,
            };
            w.insert(format!("blocks.{b}.{leaf}"), fill(n));
        }
    }
    w
}

/// Random main-PatchMerger weights (`merged = hidden * merge^2`).
fn random_merger_weights(v: &VisionConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = data::rng::Rng::new(seed);
    let mut fill = |n: usize| -> Vec<f32> { (0..n).map(|_| (rng.next_f32() - 0.5) * 0.05).collect() };
    let merged = (v.hidden * v.spatial_merge_size * v.spatial_merge_size) as usize;
    let out = v.out_hidden_size as usize;
    HashMap::from([
        ("ln.weight".to_string(), vec![1.0; v.hidden as usize]),
        ("ln.bias".to_string(), fill(v.hidden as usize)),
        ("fc1.weight".to_string(), fill(merged * merged)),
        ("fc1.bias".to_string(), fill(merged)),
        ("fc2.weight".to_string(), fill(out * merged)),
        ("fc2.bias".to_string(), fill(out)),
    ])
}

fn bench_vision(label: &str, gpu: &Gpu, v: &VisionConfig, gh: u32, gw: u32, reps: usize) {
    let n = gh * gw;
    let pv = v.patch_vec_dim();
    let pixels: Vec<f32> = (0..(n * pv) as usize).map(|i| ((i % 251) as f32 / 251.0) - 0.5).collect();
    let vw = random_vision_weights(v, 7);
    let mw = random_merger_weights(v, 11);

    let t_up = Instant::now();
    let enc = VisionEncoder::new(gpu, v.clone(), &vw);
    let merger = PatchMerger::new(gpu, &mw, v.hidden, v.spatial_merge_size, v.out_hidden_size, false);
    let upload = t_up.elapsed().as_secs_f64();

    let r = roof::ensure(gpu);
    println!("\n=== {label}: {gh}x{gw} patch grid ({n} patches, {} visual tokens) ===", n / (v.spatial_merge_size * v.spatial_merge_size));
    println!("  weight upload + pipeline build: {:.1} ms", upload * 1e3);

    // Warm-up never enters the statistics.
    let (mut feats, _) = enc.encode_with_taps(gpu, gh, gw, &pixels, &v.deepstack_indexes);
    let _ = merger.merge(gpu, &feats, n);

    let mut best_vit = f64::INFINITY;
    let mut best_mrg = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let (f, _) = enc.encode_with_taps(gpu, gh, gw, &pixels, &v.deepstack_indexes);
        best_vit = best_vit.min(t.elapsed().as_secs_f64());
        feats = f;
        let t = Instant::now();
        let out = merger.merge(gpu, &feats, n);
        best_mrg = best_mrg.min(t.elapsed().as_secs_f64());
        assert!(out.iter().all(|x| x.is_finite()), "merger produced non-finite output");
    }
    println!("  best of {reps}:");
    report("vision tower", best_vit, vit_flops(v, n as u64), vit_weight_bytes(v), r);
    report("patch merger", best_mrg, merger_flops(v, n as u64), 0, r);

    if gpu.set_kernel_timing(true) {
        gpu.reset_kernel_times();
        let (f, _) = enc.encode_with_taps(gpu, gh, gw, &pixels, &v.deepstack_indexes);
        let _ = merger.merge(gpu, &f, n);
        let mut rows = gpu.kernel_times().unwrap_or_default();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        print_kernel_table(&format!("  {label} kernels"), &rows);
        gpu.set_kernel_timing(false);
    }
}

fn vision_mode(args: &Args) {
    let cfg = Qwen3VlConfig::qwen3_vl_4b();
    let v = cfg.vision;
    let pixels = args.pixels.unwrap_or(CAPTION_MAX_PIXELS);
    let factor = v.patch_size * v.spatial_merge_size;
    let side = (pixels as f64).sqrt() as u32;
    let (h_bar, w_bar) = smart_resize(side, side, factor, DEFAULT_MIN_PIXELS, pixels);
    let (gh, gw) = patch_grid(h_bar, w_bar, v.patch_size);
    println!(
        "qwen3vl vision profile: {pixels} px budget -> {h_bar}x{w_bar} -> {gh}x{gw} patches -> {} visual tokens",
        image_token_count(h_bar, w_bar, v.patch_size, v.spatial_merge_size)
    );
    let f = vit_flops(&v, (gh * gw) as u64) + merger_flops(&v, (gh * gw) as u64);
    println!("analytic cost of one image: {:.2} TFLOP", f as f64 / 1e12);

    if args.device != "cpu" {
        bench_vision("wgpu", &Gpu::new(vision_pipelines()), &v, gh, gw, args.reps);
    }
    if args.device != "gpu" {
        bench_vision("cpu", &Gpu::new_cpu(vision_pipelines()), &v, gh, gw, args.reps);
    }
}

// ---------------------------------------------------------------------------
// caption mode
// ---------------------------------------------------------------------------

fn caption_mode(args: &Args) {
    let dir = std::env::var("BRAIN_QWEN3VL_WEIGHTS").unwrap_or_default();
    if dir.is_empty() {
        eprintln!("caption: set BRAIN_QWEN3VL_WEIGHTS to a Qwen3-VL checkpoint directory");
        std::process::exit(2);
    }
    let Some(path) = &args.image else {
        eprintln!("caption: --image FILE is required");
        std::process::exit(2);
    };
    let img = imaging::codec::load(path).unwrap_or_else(|e| {
        eprintln!("caption: {e}");
        std::process::exit(1);
    });
    let hwc: Vec<f32> = img.px.iter().map(|&b| b as f32 / 255.0).collect();
    let pixels = args.pixels.unwrap_or(CAPTION_MAX_PIXELS);

    let cfg = Qwen3VlConfig::qwen3_vl_4b();
    let params = decoder_params(&cfg.text);
    println!("qwen3vl caption profile: {} ({}x{}), {pixels} px budget, max_new {}", path, img.w, img.h, args.max_new);
    println!("decoder: {:.2} B params, {:.1} GiB fp32 -- every weight is read once per token at batch 1", params as f64 / 1e9, params as f64 * 4.0 / (1 << 30) as f64);

    let load = qwen3vl::caps::load_time(&dir, pixels).unwrap_or_else(|e| {
        eprintln!("caption: {e}");
        std::process::exit(1);
    });
    println!("model build (once per process): {:.1} s", load);

    // AFTER the build, never before, and through the resident's OWN handle:
    // `roof::known` answers only from the in-process cache, so asking it
    // without a device to measure on reported every stage as "no measured
    // roof" - the meter failing quietly rather than the kernels being
    // ungradeable.
    let r = qwen3vl::caps::device_roof(&dir, pixels).unwrap_or(None);
    match decode_ceiling_tok_s(params, r) {
        Some((f32_ceil, i8_ceil)) => println!("decode is weight-bandwidth bound: ceiling {f32_ceil:.1} tok/s fp32, {i8_ceil:.1} tok/s int8"),
        None => println!("decode ceiling: unknown (no measured roof cached for this backend -- run `qwen3vl_bench vision` once to measure it)"),
    }

    let prompt = "Describe this image in detail.";
    // Warm-up never enters the statistics -- but only when there is more than
    // one rep to warm up FOR. `load_time` above already built the resident and
    // compiled every pipeline, which is what a warm-up is for; a second full
    // caption costs as much as the measurement itself, and at one rep it would
    // double a multi-minute run to protect a number nothing is averaged into.
    if args.reps > 1 {
        let _ = qwen3vl::caps::generate_profiled(&dir, pixels, prompt, &hwc, img.w, img.h, args.max_new);
    }

    let mut best: Option<(qwen3vl::model::StageTimes, f64, String)> = None;
    for _ in 0..args.reps {
        let (text, st, pre) = qwen3vl::caps::generate_profiled(&dir, pixels, prompt, &hwc, img.w, img.h, args.max_new).unwrap_or_else(|e| {
            eprintln!("caption: {e}");
            std::process::exit(1);
        });
        let total = st.total_s() + pre;
        if best.as_ref().is_none_or(|(b, bp, _)| total < b.total_s() + bp) {
            best = Some((st, pre, text));
        }
    }
    let (st, pre, text) = best.expect("at least one rep");

    let v = &cfg.vision;
    let n = (st.visual_tokens * v.spatial_merge_size * v.spatial_merge_size) as u64;
    println!("\nbest of {} ({} prompt tokens incl. {} visual, {} generated):", args.reps, st.prompt_tokens, st.visual_tokens, st.new_tokens);
    report("image preprocess", pre, 0, (img.w as u64 * img.h as u64 * 3 + n * v.patch_vec_dim() as u64) * 4, r);
    report("vision tower", st.vision_s, vit_flops(v, n), vit_weight_bytes(v), r);
    report("projector/merger", st.merge_s, merger_flops(v, n) * (1 + cfg.vision.deepstack_indexes.len() as u64), 0, r);
    report("prefill", st.prefill_s, 2 * params * st.prompt_tokens as u64, 4 * params * st.prompt_tokens as u64, r);
    // The generation loop is pipelined, so it is priced as ONE stage - see
    // `StageTimes`' own doc on why splitting it costs more than it tells you.
    // The two halves are still printed, unpriced, so a reader can see the
    // shape of the loop.
    let gen_s = st.decode_s + st.head_s;
    let head_flops = 2 * cfg.text.vocab as u64 * cfg.text.d_model as u64 * st.new_tokens as u64;
    let head_bytes = 4 * cfg.text.vocab as u64 * cfg.text.d_model as u64 * st.new_tokens as u64;
    report("decode + head", gen_s, 2 * params * st.new_tokens as u64 + head_flops, 4 * params * st.new_tokens as u64 + head_bytes, r);
    println!("    of which {:>7.1} ms submitting decode steps, {:>7.1} ms in the head (which drains them)", st.decode_s * 1e3, st.head_s * 1e3);
    let per_image = st.total_s() + pre;
    println!("  {:<22} {:>9.1} ms", "TOTAL (per image)", per_image * 1e3);

    // The four rates, per the brief: a VL model's context is mostly image, so
    // decode tok/s alone describes only the last few percent of the work.
    let decode_tok_s = st.new_tokens as f64 / gen_s.max(1e-12);
    let prefill_tok_s = st.prompt_tokens as f64 / st.prefill_s.max(1e-12);
    println!("\nrates (best of {}, warm-up excluded):", args.reps);
    println!("  vision tokens per image  {:>10}  ({} of {} context tokens are the image)", st.visual_tokens, st.visual_tokens, st.prompt_tokens);
    println!("  prefill                  {prefill_tok_s:>10.1} tok/s");
    println!("  decode                   {decode_tok_s:>10.1} tok/s  (what a streaming caller sees: step + head)");
    if let Some((f32_ceil, i8_ceil)) = decode_ceiling_tok_s(params, r) {
        println!("  ceiling (weight BW)      {f32_ceil:>10.1} tok/s fp32, {i8_ceil:.1} tok/s int8 -- decode is at {:.1}% of it", 100.0 * decode_tok_s / f32_ceil);
    }
    println!("  end-to-end (per image)   {:>10.2} tok/s  ({} tokens / {:.1} s of preprocess+vision+merge+prefill+decode)", st.new_tokens as f64 / per_image, st.new_tokens, per_image);
    println!(
        "  end-to-end incl. load    {:>10.2} tok/s  (one image in a fresh process: {:.1} s build + {:.1} s image)",
        st.new_tokens as f64 / (per_image + load),
        load,
        per_image
    );
    println!("\ncaption: {}", text.trim());
}

// ---------------------------------------------------------------------------

struct Args {
    mode: String,
    device: String,
    reps: usize,
    pixels: Option<u32>,
    max_new: u32,
    image: Option<String>,
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    // caption reps default to 1: one rep is minutes long, and `--reps N`
    // switches on the warm-up as well as the statistics.
    let mut a = Args { mode: String::new(), device: "both".into(), reps: 0, pixels: None, max_new: 90, image: None };
    let mut i = 0;
    while i < argv.len() {
        let next = |i: &mut usize| -> String {
            *i += 1;
            argv.get(*i).cloned().unwrap_or_else(|| {
                eprintln!("{USAGE}");
                std::process::exit(2)
            })
        };
        match argv[i].as_str() {
            "--device" => a.device = next(&mut i),
            "--reps" => a.reps = next(&mut i).parse().unwrap_or(a.reps),
            "--pixels" => a.pixels = next(&mut i).parse().ok(),
            "--max-new" => a.max_new = next(&mut i).parse().unwrap_or(a.max_new),
            "--image" => a.image = Some(next(&mut i)),
            "-h" | "--help" => {
                println!("{USAGE}");
                return;
            }
            other if a.mode.is_empty() => a.mode = other.to_string(),
            other => {
                eprintln!("unknown argument {other}\n{USAGE}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    match a.mode.as_str() {
        "vision" => {
            if a.reps == 0 {
                a.reps = 3;
            }
            vision_mode(&a)
        }
        "caption" => {
            if a.reps == 0 {
                a.reps = 1;
            }
            caption_mode(&a)
        }
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    }
}
