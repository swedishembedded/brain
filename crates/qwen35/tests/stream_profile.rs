// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-checkpoint stage breakdown for one streamed layer of each type
//! (`LayerType::Linear`/`Linear` and `LayerType::Full`), split into: mmap
//! open, FP8-block host dequant (`crate::import::import_layer`), int8
//! quantize + GPU upload (`model::ops::Weight::upload`), and pure GPU
//! forward compute (`StreamState::layer_forward`, weights already resident,
//! repeated). `crate::stream::generate`'s real measured per-decode-step cost
//! is 17-40 minutes for all 64 layers (M16/M17/M18); this test exists to
//! show WHICH of those stages that time actually goes to, since
//! `qwen35_bench.rs`'s own M13 GPU-compute-only numbers (random weights,
//! already resident) account for at most a few seconds across all 64
//! layers - nowhere near 17-40 minutes - so the real cost must live
//! elsewhere in the per-decode-step pipeline, and this test measures where.
//!
//! Self-skips loudly without `BRAIN_QWEN35_DIR`. Run with:
//!
//! ```text
//! BRAIN_QWEN35_DIR=/path/to/Qwen3.8-27B-FP8 \
//!     cargo test -p brain-qwen35 --test stream_profile -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use checkpoint::mmap::MmapSafetensors;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::{Ops, Weight};
use qwen35::config::{LayerType, Qwen35Config};
use qwen35::import::import_layer;
use qwen35::stream::{OwnedGdnLayer, OwnedGqaLayer, OwnedStreamedLayer, StreamState};

fn dir() -> Option<PathBuf> {
    std::env::var("BRAIN_QWEN35_DIR").ok().map(PathBuf::from)
}

/// One layer's stage timings, in milliseconds.
#[derive(Debug, Default)]
struct Stages {
    mmap_open_ms: f64,
    import_dequant_ms: f64,
    quantize_upload_ms: f64,
    forward_compute_ms_per_rep: f64,
}

/// Build one layer's `OwnedStreamedLayer` from an already-imported host
/// weight map, timing the quantize+upload stage separately from import -
/// a close copy of `StreamState::build_layer` (private to `crate::stream`),
/// reusing the same public primitives (`Weight::upload`) so this test needs
/// no new crate-internal visibility.
fn build_layer_timed(ops: &Ops, cfg: &Qwen35Config, l: usize, ty: LayerType, w: &std::collections::HashMap<String, Vec<f32>>, gpu: &Gpu) -> (OwnedStreamedLayer, f64) {
    let p = |s: &str| format!("blocks.{l}.{s}");
    let get = |name: &str| w.get(name).unwrap_or_else(|| panic!("missing {name}")).as_slice();
    let d = cfg.d_model as usize;
    let ff = cfg.intermediate_size as usize;

    let t0 = Instant::now();
    let f32buf = |name: &str| gpu.storage_init(name, get(name));
    let i8w = |name: &str, n: usize, k: usize| Weight::upload(ops, get(name), n, k, Dtype::I8);
    let (mlp_gate, mlp_up, mlp_down) = (i8w(&p("mlp.gate.weight"), ff, d), i8w(&p("mlp.up.weight"), ff, d), i8w(&p("mlp.down.weight"), d, ff));

    let layer = match ty {
        LayerType::Linear => {
            let conv_dim = cfg.linear_conv_dim() as usize;
            let value_dim = cfg.linear_value_dim() as usize;
            let nvh = cfg.linear_num_value_heads as usize;
            OwnedStreamedLayer::Linear(OwnedGdnLayer {
                ln1: f32buf(&p("ln1.weight")),
                ln2: f32buf(&p("ln2.weight")),
                conv1d_weight: f32buf(&p("linear_attn.conv1d.weight")),
                a_log: f32buf(&p("linear_attn.A_log")),
                dt_bias: f32buf(&p("linear_attn.dt_bias")),
                norm_weight: f32buf(&p("linear_attn.norm.weight")),
                in_proj_qkv: i8w(&p("linear_attn.in_proj_qkv.weight"), conv_dim, d),
                in_proj_z: i8w(&p("linear_attn.in_proj_z.weight"), value_dim, d),
                in_proj_b: i8w(&p("linear_attn.in_proj_b.weight"), nvh, d),
                in_proj_a: i8w(&p("linear_attn.in_proj_a.weight"), nvh, d),
                out_proj: i8w(&p("linear_attn.out_proj.weight"), d, value_dim),
                mlp_gate,
                mlp_up,
                mlp_down,
            })
        }
        LayerType::Full => {
            let hqp = cfg.q_proj_dim() as usize;
            let hkv = cfg.kv_dim() as usize;
            let hq = cfg.q_dim() as usize;
            OwnedStreamedLayer::Full(OwnedGqaLayer {
                ln1: f32buf(&p("ln1.weight")),
                ln2: f32buf(&p("ln2.weight")),
                q_norm: f32buf(&p("self_attn.q_norm.weight")),
                k_norm: f32buf(&p("self_attn.k_norm.weight")),
                q_proj: i8w(&p("self_attn.q_proj.weight"), hqp, d),
                k_proj: i8w(&p("self_attn.k_proj.weight"), hkv, d),
                v_proj: i8w(&p("self_attn.v_proj.weight"), hkv, d),
                o_proj: i8w(&p("self_attn.o_proj.weight"), d, hq),
                mlp_gate,
                mlp_up,
                mlp_down,
            })
        }
    };
    (layer, t0.elapsed().as_secs_f64() * 1000.0)
}

fn profile_layer(state: &StreamState, dir: &Path, cfg: &Qwen35Config, l: usize, n: u32) -> Stages {
    let ty = cfg.layer_types()[l];
    let shard = dir.join(format!("layers-{l}.safetensors"));

    let t0 = Instant::now();
    let reader = MmapSafetensors::open(&shard).unwrap_or_else(|e| panic!("open {}: {e}", shard.display()));
    let mmap_open_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let w = import_layer(&reader, cfg, l, 128).unwrap_or_else(|e| panic!("import_layer({l}): {e}"));
    let import_dequant_ms = t1.elapsed().as_secs_f64() * 1000.0;
    drop(reader);

    let (layer, quantize_upload_ms) = build_layer_timed(&state.ops, cfg, l, ty, &w, &state.gpu);

    // Pure GPU forward compute, weights already resident: several reps,
    // averaged, to isolate compute from the one-time upload cost above.
    const REPS: u32 = 5;
    let xres = state.gpu.storage_init("profile.xres", &qwen35::stream::seed_residual(n, cfg.d_model, 7));
    let t2 = Instant::now();
    for _ in 0..REPS {
        let _ = state.layer_forward(cfg, &layer, &xres, n);
    }
    let forward_compute_ms_per_rep = t2.elapsed().as_secs_f64() * 1000.0 / REPS as f64;

    Stages { mmap_open_ms, import_dequant_ms, quantize_upload_ms, forward_compute_ms_per_rep }
}

#[test]
#[ignore]
fn profile_one_gdn_and_one_gqa_layer_real_weights() {
    let Some(dir) = dir() else {
        eprintln!("SKIP profile_one_gdn_and_one_gqa_layer_real_weights: set BRAIN_QWEN35_DIR");
        return;
    };
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    eprintln!("ambient load (/proc/loadavg) at measurement start: {}", load.trim());

    let cfg = Qwen35Config::qwen38_27b();
    let n = 128u32;
    let gpu = Gpu::new(qwen35::model::pipelines());
    let state = StreamState::new(gpu, &cfg, n);

    // Layer 0: Gated DeltaNet (Linear). Layer 3: first GQA (Full) layer -
    // both already used as the real-weight parity reference layers (M10).
    for &l in &[0usize, 3] {
        let s = profile_layer(&state, &dir, &cfg, l, n);
        let ty = cfg.layer_types()[l];
        eprintln!(
            "layer {l} ({ty:?}): mmap_open={:.1}ms import_dequant={:.1}ms quantize_upload={:.1}ms forward_compute={:.2}ms/rep total_load={:.1}ms",
            s.mmap_open_ms,
            s.import_dequant_ms,
            s.quantize_upload_ms,
            s.forward_compute_ms_per_rep,
            s.mmap_open_ms + s.import_dequant_ms + s.quantize_upload_ms,
        );
    }

    let load_end = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    eprintln!("ambient load (/proc/loadavg) at measurement end: {}", load_end.trim());
}
