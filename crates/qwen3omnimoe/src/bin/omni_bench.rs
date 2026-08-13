// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Omni profiler: where a Thinker/Talker MoE decoder LAYER, the audio tower,
//! and the vision tower actually spend their time, graded against the
//! device's measured roofline. Shaped after `qwen_bench`
//! (`crates/qwen3/src/bin/qwen_bench.rs`), with one structural difference
//! stated up front: `qwen3::Qwen` builds one flat, un-submitted `Vec<Step>`
//! per pass that `gpu_core::profile::profile` times kernel-group by
//! kernel-group. Omni's `thinker`/`talker`/tower modules submit eagerly
//! (several `g.submit()` calls per layer — attention, then the MoE tail),
//! so there is no single Step list to hand that profiler. Instead each mode
//! here wall-clock-times the real call (the number that decides whether a
//! change worked, same floor `PassProfile::total_secs` measures) and prints
//! the roofline + weight-budget context around it; run with `BRAIN_PROFILE=1`
//! for the engine's own per-kernel-kind breakdown of the SAME submitted work
//! (`gpu_core::Gpu::dump_profile`), which needs no returned Step list either.
//!
//! Random weights throughout: cost depends on shape, not values. Every
//! number this produces is **valid for cost and meaningless for output
//! quality**.
//!
//! **Real production scale, not a toy config**: a single MoE decoder layer
//! (even at the real 128-experts/hidden-2048 Thinker shape) is 2-3 GB of
//! random fp32 weights — trivially synthesizable, unlike a full 48-layer /
//! 30B-parameter forward, which is not attempted here (that needs a real
//! streamed checkpoint, not random weights — `cli::resident_omni`'s scope,
//! not this bench's). `thinker-layer`/`talker-layer` profile ONE layer at
//! the real config (`MoeTextConfig::thinker_defaults`/`talker_defaults`) —
//! the same router/expert-gather/grouped-GEMM kernel set every layer
//! dispatches, so the per-layer numbers here scale linearly (roughly; KV
//! cache footprint and expert-selection skew are the caveats) to the real
//! 48-layer / 20-layer stacks.
//!
//! **Found running this on the CPU backend**: `banner()`'s
//! `gpu_core::roof::ensure` roofline probe (calibrated for GPU throughput)
//! takes so long on CPU/Cranelift JIT that it reads as a hang, not a slow
//! measurement — a real-scale `thinker-layer` run sat at 12 MB RSS / 0.3%
//! CPU for hours of wall-clock before being killed, entirely inside the
//! probe (confirmed by bisecting: the process never got past `banner()`'s
//! first print). `gpu_core::roof` already ships the fix as an env var —
//! `BRAIN_NO_ROOF=1` skips the probe and `banner()` reports "roofline
//! unmeasured" instead of guessing. **Set it for any CPU-backend run of this
//! bench** (`BRAIN_DEVICE=cpu BRAIN_NO_ROOF=1 omni_bench thinker-layer`);
//! on a real GPU backend the probe is fast and safe to leave on.
//!
//! No `perf/omni` Makefile target: none of this bench's direct analogs
//! (`qwen_bench`, `vqgan_bench`, `unet_bench`, `zimage_bench`,
//! `flux2_bench`) are wired into the Makefile's `perf/*` targets either —
//! those go through `brain perf run`'s residency-executor target
//! resolution, a different (bigger) integration that would need omni
//! plugged into production residency first (not yet built). Run
//! this bench directly, same as its siblings.
//!
//! Usage:
//!   omni_bench cost                        # offline param/GFLOP/byte accounting, no device
//!   omni_bench thinker-layer [n] [reps]     # one real-scale Thinker MoE layer, prefill T=n
//!   omni_bench talker-layer  [n] [reps]     # one real-scale Talker MoE layer (+shared expert)
//!   omni_bench encode-audio  [n_audio] [reps]
//!   omni_bench encode-vision [grid_h] [grid_w] [reps]

use std::time::Instant;

use gpu_core::roof::Roofs;
use gpu_core::{DeviceBuffer, Gpu};
use qwen3omnimoe::config::MoeTextConfig;
use qwen3omnimoe::talker::{self, TalkerLayerWeights};
use qwen3omnimoe::thinker::{self, ThinkerLayerWeights};
use qwen3asr::config::AudioEncoderConfig;
use qwen3asr::encoder::{audio_pipelines, AudioEncoder};
use qwen3vl::config::VisionConfig;
use qwen3vl::encoder::{vision_pipelines, PatchMerger, VisionEncoder};

fn banner(gpu: &Gpu) -> Option<Roofs> {
    let roofs = gpu_core::roof::ensure(gpu);
    match roofs {
        Some(r) => println!("measured roofline: {:.0} GFLOP/s, {:.1} GB/s DRAM, {:.1} GB/s cache, ridge {:.1} FLOP/byte", r.gflops, r.gbs, r.cache_gbs, r.ridge()),
        None => println!("roofline unmeasured — utilisation columns print '-' rather than a guess"),
    }
    roofs
}

fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 40) as f32) / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

/// A `[n, head_dim/2]` RoPE table at plain sequential positions — the
/// degenerate diagonal-collapse case `thinker.rs`'s own module doc names
/// (pure text: all three M-RoPE axes share one position). Position
/// SEMANTICS don't matter for a cost profile, only that the shapes are
/// valid and the values are finite.
fn rope_table(n: u32, head_dim: u32, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = (head_dim / 2) as usize;
    let mut cos = vec![0f32; n as usize * half];
    let mut sin = vec![0f32; n as usize * half];
    for p in 0..n as usize {
        for j in 0..half {
            let inv = theta.powf(-2.0 * j as f32 / head_dim as f32);
            let ang = p as f32 * inv;
            cos[p * half + j] = ang.cos();
            sin[p * half + j] = ang.sin();
        }
    }
    (cos, sin)
}

fn report(label: &str, reps: usize, mut f: impl FnMut()) -> f64 {
    f(); // warm up (pipeline/shader compile, allocator warmup)
    let t0 = Instant::now();
    for _ in 0..reps {
        f();
    }
    let secs = t0.elapsed().as_secs_f64() / reps as f64;
    println!("{label}: {:.3} ms/call ({reps} reps)", secs * 1e3);
    secs
}

// ---- cost (offline, no device) ----

/// Total resident params for one MoE decoder layer (every expert), and the
/// ACTIVE params one token's decode step actually reads (router + its
/// top_k experts + the always-on shared expert, if any) — the distinction
/// that matters: MoE decode bandwidth is bounded
/// by ACTIVE bytes, not resident ones, unlike a dense model where they're
/// the same number.
fn moe_layer_params(cfg: &MoeTextConfig) -> (u64, u64) {
    let (d, hd, nh, nkv) = (cfg.hidden as u64, cfg.head_dim as u64, cfg.n_heads as u64, cfg.n_kv_heads as u64);
    let (hq, hkv) = (nh * hd, nkv * hd);
    let attn = d * hq + 2 * (d * hkv) + hq * d; // wq + wk + wv + wo
    let router = d * cfg.n_experts as u64;
    let expert = 3 * d * cfg.moe_intermediate as u64; // gate+up+down
    let shared = if cfg.has_shared_expert() { 3 * d * cfg.shared_expert_intermediate as u64 + d } else { 0 };
    let resident = attn + router + cfg.n_experts as u64 * expert + shared;
    let active = attn + router + cfg.top_k as u64 * expert + shared;
    (resident, active)
}

fn print_moe_cost(name: &str, cfg: &MoeTextConfig, roofs: Option<Roofs>) {
    let (resident_layer, active_layer) = moe_layer_params(cfg);
    let resident = resident_layer * cfg.n_layers as u64;
    let active = active_layer * cfg.n_layers as u64;
    println!(
        "\n{name}: {} layers, {} experts top-{}{}, hidden {}, moe_ff {}",
        cfg.n_layers,
        cfg.n_experts,
        cfg.top_k,
        if cfg.has_shared_expert() { format!("+shared({})", cfg.shared_expert_intermediate) } else { String::new() },
        cfg.hidden,
        cfg.moe_intermediate
    );
    println!(
        "  resident: {:.2}B params = {:.1} GB fp32 / {:.1} GB int8 (VRAM footprint — every expert must be resident somewhere)",
        resident as f64 / 1e9,
        resident as f64 * 4.0 / 1e9,
        resident as f64 / 1e9,
    );
    println!(
        "  active/token: {:.2}B params ({:.1}% of resident) = {:.2} GFLOP/token",
        active as f64 / 1e9,
        100.0 * active as f64 / resident as f64,
        2.0 * active as f64 / 1e9,
    );
    if let Some(r) = roofs {
        let bw = r.gbs as f64 * 1e9;
        println!(
            "  decode ceiling (bandwidth on ACTIVE bytes/token — the honest MoE number, \
             not the resident one a dense-model formula would give): {:.0} tok/s fp32, {:.0} tok/s int8",
            bw / (active as f64 * 4.0),
            bw / active as f64,
        );
    }
}

/// Dense (no MoE) tower cost: `depth` transformer blocks, fused QKV + proj +
/// 2-layer MLP, resident == active (every weight is read every token/patch —
/// no sparse routing).
fn dense_tower_params(depth: u32, hidden: u32, ffn: u32) -> u64 {
    let c = hidden as u64;
    let qkv = 3 * c * c;
    let proj = c * c;
    let mlp = 2 * c * ffn as u64;
    depth as u64 * (qkv + proj + mlp)
}

fn print_dense_cost(name: &str, params: u64, roofs: Option<Roofs>) {
    println!("{name}: {:.2}B params = {:.2} GB fp32 / {:.2} GB int8", params as f64 / 1e9, params as f64 * 4.0 / 1e9, params as f64 / 1e9);
    if let Some(r) = roofs {
        let bw = r.gbs as f64 * 1e9;
        println!("  decode/per-token ceiling (dense — every weight read every call): {:.0} tok/s fp32, {:.0} tok/s int8", bw / (params as f64 * 4.0), bw / params as f64);
    }
}

fn print_cost(roofs: Option<Roofs>) {
    print_moe_cost("Thinker", &MoeTextConfig::thinker_defaults(), roofs);
    print_moe_cost("Talker", &MoeTextConfig::talker_defaults(), roofs);

    let ac = AudioEncoderConfig::qwen3_omni();
    let audio = dense_tower_params(ac.n_layers, ac.d_model, ac.ffn_dim) + ac.d_model as u64 * ac.d_model as u64 * 2; // + projector
    println!("\nAudio tower: {} layers, d_model {}, ffn {} — {:.2}B params = {:.2} GB fp32", ac.n_layers, ac.d_model, ac.ffn_dim, audio as f64 / 1e9, audio as f64 * 4.0 / 1e9);

    let vc = VisionConfig::qwen3_omni();
    let merged = vc.hidden * vc.spatial_merge_size * vc.spatial_merge_size;
    let vision = dense_tower_params(vc.depth, vc.hidden, vc.intermediate) + (vc.hidden * vc.patch_vec_dim()) as u64 + (merged as u64 * merged as u64 + vc.out_hidden_size as u64 * merged as u64);
    println!("Vision tower: {} layers, hidden {}, intermediate {} — {:.2}B params = {:.2} GB fp32", vc.depth, vc.hidden, vc.intermediate, vision as f64 / 1e9, vision as f64 * 4.0 / 1e9);

    // Code2Wav: field-comment real defaults from `qwen3omnimoe::config::Code2WavConfig`
    // (no standalone preset fn exists — these are the exact numbers
    // `Code2WavConfig::from_json`'s own fallbacks carry).
    let (hidden, inter, layers, dec_dim): (u64, u64, u64, u64) = (1024, 3072, 8, 1536);
    let pre_transformer = layers * (3 * hidden * hidden / 4 /* GQA-ish attn, rough */ + 2 * hidden * inter);
    let seanet = 4 * (2 * dec_dim * dec_dim); // 4 upsample stages, rough
    let codec = pre_transformer + seanet;
    println!("Code2Wav: {layers} pre-transformer layers, hidden {hidden}, decoder_dim {dec_dim} — ~{:.2}B params (rough) = ~{:.2} GB fp32", codec as f64 / 1e9, codec as f64 * 4.0 / 1e9);
    println!("  (Code2Wav's SEANet decoder has an irregular per-stage channel schedule — this is a coarse order-of-magnitude estimate, not the precise weight_budget style count the other rows are.)");
}

// ---- thinker-layer / talker-layer (real single-layer scale, random weights) ----

/// Owned random weight buffers for one Thinker MoE decoder layer — a struct
/// rather than a 10-tuple so [`ThinkerLayerWeights`] can borrow from it by
/// field name at the call site (and so clippy doesn't flag a bare tuple this
/// wide as an unreadable type).
struct ThinkerWeightBufs {
    ln1: DeviceBuffer,
    wq: DeviceBuffer,
    wk: DeviceBuffer,
    wv: DeviceBuffer,
    wo: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    ln2: DeviceBuffer,
    router: DeviceBuffer,
    experts: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer)>,
}

fn thinker_weights(g: &Gpu, cfg: &MoeTextConfig, seed: u64) -> ThinkerWeightBufs {
    let (d, hd, nh, nkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let (hq, hkv) = (nh * hd, nkv * hd);
    let mut s = seed;
    let mut next = |n: usize| {
        s += 1;
        fill(s, n, 0.02)
    };
    let experts: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer)> = (0..cfg.n_experts)
        .map(|_| (g.storage_init("gw", &next((d * cfg.moe_intermediate) as usize)), g.storage_init("uw", &next((d * cfg.moe_intermediate) as usize)), g.storage_init("dw", &next((cfg.moe_intermediate * d) as usize))))
        .collect();
    ThinkerWeightBufs {
        ln1: g.storage_init("ln1", &vec![1.0f32; d as usize]),
        wq: g.storage_init("wq", &next((d * hq) as usize)),
        wk: g.storage_init("wk", &next((d * hkv) as usize)),
        wv: g.storage_init("wv", &next((d * hkv) as usize)),
        wo: g.storage_init("wo", &next((hq * d) as usize)),
        q_norm: g.storage_init("qn", &vec![1.0f32; hd as usize]),
        k_norm: g.storage_init("kn", &vec![1.0f32; hd as usize]),
        ln2: g.storage_init("ln2", &vec![1.0f32; d as usize]),
        router: g.storage_init("router", &next((d * cfg.n_experts) as usize)),
        experts,
    }
}

fn thinker_layer_mode(a: &[String]) {
    let n: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let reps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let cfg = MoeTextConfig::thinker_defaults();
    eprintln!("omni_bench thinker-layer: real-scale (128 experts top-8, hidden {}), T={n}, {reps} reps (random weights)", cfg.hidden);
    let gpu = Gpu::new(thinker::thinker_pipelines());
    let roofs = banner(&gpu);
    print_moe_cost("Thinker (1 layer)", &MoeTextConfig { n_layers: 1, ..cfg.clone() }, roofs);

    let t0 = Instant::now();
    let b = thinker_weights(&gpu, &cfg, 7);
    let w = ThinkerLayerWeights { ln1: &b.ln1, wq: &b.wq, wk: &b.wk, wv: &b.wv, wo: &b.wo, q_norm: &b.q_norm, k_norm: &b.k_norm, ln2: &b.ln2, router: &b.router, experts: &b.experts };
    eprintln!("weights built + uploaded in {:.1}s", t0.elapsed().as_secs_f32());

    let x = gpu.storage_init("x", &fill(1, (n * cfg.hidden) as usize, 0.5));
    let (cos, sin) = rope_table(n, cfg.head_dim, cfg.rope_theta);
    let cos_b = gpu.storage_init("cos", &cos);
    let sin_b = gpu.storage_init("sin", &sin);

    let secs = report(&format!("thinker-layer prefill T={n}"), reps, || {
        thinker::layer_fwd(&gpu, &cfg, &w, &x, &cos_b, &sin_b, n, None, None);
    });
    println!("-> {:.0} tok/s (single layer; multiply by ~1/{} for a rough 48-layer estimate)", n as f64 / secs, cfg.n_layers);
}

/// Owned random weight buffers for one Talker MoE decoder layer — a struct
/// for the same reason [`ThinkerWeightBufs`] is (Talker additionally carries
/// the always-active shared expert + its gate).
struct TalkerWeightBufs {
    ln1: DeviceBuffer,
    wq: DeviceBuffer,
    wk: DeviceBuffer,
    wv: DeviceBuffer,
    wo: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    ln2: DeviceBuffer,
    router: DeviceBuffer,
    experts: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer)>,
    shared_expert: (DeviceBuffer, DeviceBuffer, DeviceBuffer),
    shared_expert_gate: DeviceBuffer,
}

fn talker_weights(g: &Gpu, cfg: &MoeTextConfig, seed: u64) -> TalkerWeightBufs {
    let (d, hd, nh, nkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let (hq, hkv) = (nh * hd, nkv * hd);
    let mut s = seed;
    let mut next = |n: usize| {
        s += 1;
        fill(s, n, 0.02)
    };
    let experts: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer)> = (0..cfg.n_experts)
        .map(|_| (g.storage_init("gw", &next((d * cfg.moe_intermediate) as usize)), g.storage_init("uw", &next((d * cfg.moe_intermediate) as usize)), g.storage_init("dw", &next((cfg.moe_intermediate * d) as usize))))
        .collect();
    let shared_expert = (g.storage_init("sgw", &next((d * cfg.shared_expert_intermediate) as usize)), g.storage_init("suw", &next((d * cfg.shared_expert_intermediate) as usize)), g.storage_init("sdw", &next((cfg.shared_expert_intermediate * d) as usize)));
    TalkerWeightBufs {
        ln1: g.storage_init("ln1", &vec![1.0f32; d as usize]),
        wq: g.storage_init("wq", &next((d * hq) as usize)),
        wk: g.storage_init("wk", &next((d * hkv) as usize)),
        wv: g.storage_init("wv", &next((d * hkv) as usize)),
        wo: g.storage_init("wo", &next((hq * d) as usize)),
        q_norm: g.storage_init("qn", &vec![1.0f32; hd as usize]),
        k_norm: g.storage_init("kn", &vec![1.0f32; hd as usize]),
        ln2: g.storage_init("ln2", &vec![1.0f32; d as usize]),
        router: g.storage_init("router", &next((d * cfg.n_experts) as usize)),
        experts,
        shared_expert,
        shared_expert_gate: g.storage_init("sg", &next(d as usize)),
    }
}

fn talker_layer_mode(a: &[String]) {
    let n: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let reps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let cfg = MoeTextConfig::talker_defaults();
    eprintln!("omni_bench talker-layer: real-scale (128 experts top-6 + shared, hidden {}), T={n}, {reps} reps (random weights)", cfg.hidden);
    let gpu = Gpu::new(talker::talker_pipelines());
    let roofs = banner(&gpu);
    print_moe_cost("Talker (1 layer)", &MoeTextConfig { n_layers: 1, ..cfg.clone() }, roofs);

    let t0 = Instant::now();
    let b = talker_weights(&gpu, &cfg, 7);
    let w = TalkerLayerWeights {
        ln1: &b.ln1,
        wq: &b.wq,
        wk: &b.wk,
        wv: &b.wv,
        wo: &b.wo,
        q_norm: &b.q_norm,
        k_norm: &b.k_norm,
        ln2: &b.ln2,
        router: &b.router,
        experts: &b.experts,
        shared_expert: (&b.shared_expert.0, &b.shared_expert.1, &b.shared_expert.2),
        shared_expert_gate: &b.shared_expert_gate,
    };
    eprintln!("weights built + uploaded in {:.1}s", t0.elapsed().as_secs_f32());

    let x = gpu.storage_init("x", &fill(1, (n * cfg.hidden) as usize, 0.5));
    let (cos, sin) = rope_table(n, cfg.head_dim, cfg.rope_theta);
    let cos_b = gpu.storage_init("cos", &cos);
    let sin_b = gpu.storage_init("sin", &sin);

    let secs = report(&format!("talker-layer prefill T={n}"), reps, || {
        talker::layer_fwd(&gpu, &cfg, &w, &x, &cos_b, &sin_b, n, None, None);
    });
    println!("-> {:.0} tok/s (single layer; multiply by ~1/{} for a rough 20-layer estimate)", n as f64 / secs, cfg.n_layers);
}

// ---- encode-audio / encode-vision (real full tower scale, random weights) ----

fn encode_audio_mode(a: &[String]) {
    let n_audio: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let reps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let cfg = AudioEncoderConfig::qwen3_omni();
    eprintln!("omni_bench encode-audio: real-scale ({} layers, d_model {}), n_audio={n_audio}, {reps} reps (random weights)", cfg.n_layers, cfg.d_model);
    let gpu = Gpu::new(audio_pipelines());
    let roofs = banner(&gpu);
    let audio_params = dense_tower_params(cfg.n_layers, cfg.d_model, cfg.ffn_dim) + cfg.d_model as u64 * cfg.d_model as u64 * 2;
    print_dense_cost("Audio tower", audio_params, roofs);

    let (c, ffn, out) = (cfg.d_model as usize, cfg.ffn_dim as usize, cfg.output_dim as usize);
    let mut w = std::collections::HashMap::new();
    let mut s = 1u64;
    let mut next = |n: usize| {
        s += 1;
        fill(s, n, 0.02)
    };
    for b in 0..cfg.n_layers {
        let p = format!("blocks.{b}");
        w.insert(format!("{p}.norm1.weight"), vec![1.0f32; c]);
        w.insert(format!("{p}.norm1.bias"), next(c));
        w.insert(format!("{p}.qkv.weight"), next(3 * c * c));
        w.insert(format!("{p}.qkv.bias"), next(3 * c));
        w.insert(format!("{p}.proj.weight"), next(c * c));
        w.insert(format!("{p}.proj.bias"), next(c));
        w.insert(format!("{p}.norm2.weight"), vec![1.0f32; c]);
        w.insert(format!("{p}.norm2.bias"), next(c));
        w.insert(format!("{p}.fc1.weight"), next(ffn * c));
        w.insert(format!("{p}.fc1.bias"), next(ffn));
        w.insert(format!("{p}.fc2.weight"), next(c * ffn));
        w.insert(format!("{p}.fc2.bias"), next(c));
    }
    w.insert("ln_post.weight".into(), vec![1.0f32; c]);
    w.insert("ln_post.bias".into(), next(c));
    w.insert("multi_modal_projector.linear_1.weight".into(), next(c * c));
    w.insert("multi_modal_projector.linear_1.bias".into(), next(c));
    w.insert("multi_modal_projector.linear_2.weight".into(), next(out * c));
    w.insert("multi_modal_projector.linear_2.bias".into(), next(out));

    let t0 = Instant::now();
    let enc = AudioEncoder::new(&gpu, cfg, &w);
    eprintln!("weights built + uploaded in {:.1}s", t0.elapsed().as_secs_f32());
    let packed = fill(0xBEEF, n_audio as usize * c, 1.0);
    let spans = [(0u32, n_audio)];
    let secs = report(&format!("encode-audio n_audio={n_audio}"), reps, || {
        enc.encode_packed(&packed, n_audio, &spans);
    });
    println!("-> {:.0} tokens/s", n_audio as f64 / secs);
}

fn encode_vision_mode(a: &[String]) {
    let gh: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let gw: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
    let reps: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
    let cfg = VisionConfig::qwen3_omni();
    eprintln!("omni_bench encode-vision: real-scale ({} layers, hidden {}), grid {gh}x{gw}, {reps} reps (random weights)", cfg.depth, cfg.hidden);
    let gpu = Gpu::new(vision_pipelines());
    let roofs = banner(&gpu);
    let merged_dim = cfg.hidden * cfg.spatial_merge_size * cfg.spatial_merge_size;
    let vision_params = dense_tower_params(cfg.depth, cfg.hidden, cfg.intermediate) + (cfg.hidden * cfg.patch_vec_dim()) as u64 + (merged_dim as u64 * merged_dim as u64 + cfg.out_hidden_size as u64 * merged_dim as u64);
    print_dense_cost("Vision tower", vision_params, roofs);

    let (c, pv, inter) = (cfg.hidden as usize, cfg.patch_vec_dim() as usize, cfg.intermediate as usize);
    let mut enc_w = std::collections::HashMap::new();
    let mut s = 1u64;
    let mut next = |n: usize| {
        s += 1;
        fill(s, n, 0.02)
    };
    enc_w.insert("patch_embed.weight".into(), next(c * pv));
    enc_w.insert("patch_embed.bias".into(), next(c));
    enc_w.insert("pos_embed".into(), next(cfg.num_position_embeddings as usize * c));
    for b in 0..cfg.depth {
        let p = format!("blocks.{b}");
        enc_w.insert(format!("{p}.norm1.weight"), vec![1.0f32; c]);
        enc_w.insert(format!("{p}.norm1.bias"), next(c));
        enc_w.insert(format!("{p}.qkv.weight"), next(3 * c * c));
        enc_w.insert(format!("{p}.qkv.bias"), next(3 * c));
        enc_w.insert(format!("{p}.proj.weight"), next(c * c));
        enc_w.insert(format!("{p}.proj.bias"), next(c));
        enc_w.insert(format!("{p}.norm2.weight"), vec![1.0f32; c]);
        enc_w.insert(format!("{p}.norm2.bias"), next(c));
        enc_w.insert(format!("{p}.fc1.weight"), next(inter * c));
        enc_w.insert(format!("{p}.fc1.bias"), next(inter));
        enc_w.insert(format!("{p}.fc2.weight"), next(c * inter));
        enc_w.insert(format!("{p}.fc2.bias"), next(c));
    }
    let merged = (cfg.hidden * cfg.spatial_merge_size * cfg.spatial_merge_size) as usize;
    let mut mrg_w = std::collections::HashMap::new();
    mrg_w.insert("ln.weight".into(), vec![1.0f32; c]);
    mrg_w.insert("ln.bias".into(), next(c));
    mrg_w.insert("fc1.weight".into(), next(merged * merged));
    mrg_w.insert("fc1.bias".into(), next(merged));
    mrg_w.insert("fc2.weight".into(), next(cfg.out_hidden_size as usize * merged));
    mrg_w.insert("fc2.bias".into(), next(cfg.out_hidden_size as usize));

    let t0 = Instant::now();
    let enc = VisionEncoder::new(&gpu, cfg.clone(), &enc_w);
    let merger = PatchMerger::new(&gpu, &mrg_w, cfg.hidden, cfg.spatial_merge_size, cfg.out_hidden_size, false);
    eprintln!("weights built + uploaded in {:.1}s", t0.elapsed().as_secs_f32());
    let n = gh * gw;
    let pixels = fill(0xC0FFEE, n as usize * pv, 0.5);
    let secs = report(&format!("encode-vision grid={gh}x{gw}"), reps, || {
        let features = enc.encode(gh, gw, &pixels);
        merger.merge(&features, n);
    });
    println!("-> {:.0} patches/s", n as f64 / secs);
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).map(|s| s.as_str()).unwrap_or("cost");
    match mode {
        "cost" => print_cost(None),
        "thinker-layer" => thinker_layer_mode(&a),
        "talker-layer" => talker_layer_mode(&a),
        "encode-audio" => encode_audio_mode(&a),
        "encode-vision" => encode_vision_mode(&a),
        other => eprintln!("usage: omni_bench <cost|thinker-layer|talker-layer|encode-audio|encode-vision> ...  (got {other:?})"),
    }
}
