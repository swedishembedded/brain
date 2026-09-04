// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity gate for the Qwen3-VL/Omni vision-tower **head** (patch-embed +
//! `depth`× ViT block + main `PatchMerger`) as an OpenVINO ONNX graph. Ported
//! op-for-op from `qwen3vl::encoder::VisionEncoder::encode` +
//! `PatchMerger::merge` (single-merger path — see `crate::qwenvl_topology`'s
//! module doc for the DeepStack scope note). Self-contained: a tiny
//! random-weight head is run through both the reference (device) and the
//! ONNX graph on the OpenVINO **CPU** device, and the visual embeds must
//! agree. Patch packing + pos-embed bilinear resample stay on host (like the
//! audio tower's conv stem).
//!
//! Skips cleanly without an OpenVINO runtime. Run:
//!   LD_LIBRARY_PATH=<openvino/libs> cargo test -p brain-npu --test qwenvl_onnx -- --nocapture

use std::collections::HashMap;

use npu::openvino::{available_devices, Feed, NpuConfig, NpuDevice, NpuGraph, PerfHint};
use npu::{build_vit_head, VitTopo};
use qwen3vl::config::VisionConfig;
use qwen3vl::encoder::{vision_pipelines, PatchMerger, VisionEncoder};

fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    // The unified deterministic LCG (audit F39/F40) — one premix keeps
    // distinct seeds decorrelated, as the old local copy did.
    let mut l = data::rng::Lcg::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
    (0..n).map(|_| l.scaled(scale)).collect()
}

fn tiny_cfg() -> VisionConfig {
    VisionConfig {
        depth: 2,
        hidden: 32,
        num_heads: 2,
        intermediate: 64,
        patch_size: 2,
        temporal_patch_size: 1,
        spatial_merge_size: 2,
        num_position_embeddings: 16, // 4x4 table
        out_hidden_size: 40,
        in_channels: 2,
        deepstack_indexes: vec![],
        // Unused by the ViT export (it is a video-timestamp-to-position scale
        // that only `qwen3vl::mrope::get_rope_index_video` reads), but the
        // field is not optional, so it carries the same default every other
        // constructor of this config uses.
        tokens_per_second: 2,
    }
}

fn topo_from(cfg: &VisionConfig) -> VitTopo {
    VitTopo {
        depth: cfg.depth,
        hidden: cfg.hidden,
        num_heads: cfg.num_heads,
        intermediate: cfg.intermediate,
        out_hidden: cfg.out_hidden_size,
        merge: cfg.spatial_merge_size,
        eps: 1e-6,
        rope_theta: 10000.0,
    }
}

/// Encoder weights (`patch_embed`, `pos_embed`, `blocks.N.*`) — same layout
/// `qwen3vl::encoder`'s own `rand_weights` test helper builds.
fn encoder_weights(cfg: &VisionConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let (c, pv, mlp) = (cfg.hidden as usize, cfg.patch_vec_dim() as usize, cfg.intermediate as usize);
    let mut w = HashMap::new();
    let mut s = seed;
    let mut next = |n: usize| {
        s += 1;
        fill(s, n, 0.1)
    };
    w.insert("patch_embed.weight".into(), next(c * pv));
    w.insert("patch_embed.bias".into(), next(c));
    w.insert("pos_embed".into(), next(cfg.num_position_embeddings as usize * c));
    for b in 0..cfg.depth {
        let p = format!("blocks.{b}");
        w.insert(format!("{p}.norm1.weight"), vec![1.0; c]);
        w.insert(format!("{p}.norm1.bias"), next(c));
        w.insert(format!("{p}.qkv.weight"), next(3 * c * c));
        w.insert(format!("{p}.qkv.bias"), next(3 * c));
        w.insert(format!("{p}.proj.weight"), next(c * c));
        w.insert(format!("{p}.proj.bias"), next(c));
        w.insert(format!("{p}.norm2.weight"), vec![1.0; c]);
        w.insert(format!("{p}.norm2.bias"), next(c));
        w.insert(format!("{p}.fc1.weight"), next(mlp * c));
        w.insert(format!("{p}.fc1.bias"), next(mlp));
        w.insert(format!("{p}.fc2.weight"), next(c * mlp));
        w.insert(format!("{p}.fc2.bias"), next(c));
    }
    w
}

/// Main-merger weights (`ln`/`fc1`/`fc2`, pre-shuffle norm).
fn merger_weights(in_dim: u32, merge: u32, out_dim: u32, seed: u64) -> HashMap<String, Vec<f32>> {
    let merged = (in_dim * merge * merge) as usize;
    let mut w = HashMap::new();
    let mut s = seed;
    let mut next = |n: usize| {
        s += 1;
        fill(s, n, 0.1)
    };
    w.insert("ln.weight".into(), vec![1.0; in_dim as usize]);
    w.insert("ln.bias".into(), next(in_dim as usize));
    w.insert("fc1.weight".into(), next(merged * merged));
    w.insert("fc1.bias".into(), next(merged));
    w.insert("fc2.weight".into(), next(out_dim as usize * merged));
    w.insert("fc2.bias".into(), next(out_dim as usize));
    w
}

#[test]
fn vision_head_matches_reference_on_cpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    if available_devices().map(|d| d.is_empty()).unwrap_or(true) {
        brain_testutil::skip_unavailable("no OpenVINO runtime");
        return;
    }
    let cfg = tiny_cfg();
    let topo = topo_from(&cfg);
    let enc_w = encoder_weights(&cfg, 1);
    let mrg_w = merger_weights(cfg.hidden, cfg.spatial_merge_size, cfg.out_hidden_size, 900);

    let (gh, gw) = (4u32, 4u32); // 16 patches, one 2x2-merge grid per block
    let n = (gh * gw) as usize;
    let pv = cfg.patch_vec_dim() as usize;
    let pixels = fill(0xC0FFEE, n * pv, 0.5);

    // reference (device) head: ViT encode -> main PatchMerger.
    let gpu = gpu_core::Gpu::new_cpu(vision_pipelines());
    let vit = VisionEncoder::new(&gpu, cfg.clone(), &enc_w);
    let features = vit.encode(&gpu, gh, gw, &pixels);
    let merger = PatchMerger::new(&gpu, &mrg_w, cfg.hidden, cfg.spatial_merge_size, cfg.out_hidden_size, false);
    let reference = merger.merge(&gpu, &features, gh * gw);
    let mrows = (gh * gw / (cfg.spatial_merge_size * cfg.spatial_merge_size)) as usize;
    assert_eq!(reference.len(), mrows * cfg.out_hidden_size as usize);

    // ONNX head: combine encoder + merger (merger keys under "merger." — see
    // `qwen3omnimoe::npu_export::export_vision_onnx`'s same convention).
    let mut w = enc_w;
    for (k, v) in mrg_w {
        w.insert(format!("merger.{k}"), v);
    }
    let mut g = onnx::GraphBuilder::new("vision_head");
    g.input_f32("pixels", &[n as i64, pv as i64]);
    build_vit_head(&mut g, &topo, &w, gh, gw, cfg.pos_grid(), cfg.patch_vec_dim(), "pixels", "visual_embeds");
    g.output_f32("visual_embeds", &[mrows as i64, cfg.out_hidden_size as i64]);
    let bytes = g.finish_with(onnx::DEFAULT_OPSET, onnx::DEFAULT_IR_VERSION);

    let cfgv = NpuConfig { device: NpuDevice::Cpu, perf_hint: PerfHint::Latency, allow_fallback: true, ..Default::default() };
    // NOT a skip. The test already established above that an OpenVINO runtime
    // is present, and this compiles OUR OWN emitted ONNX onto the always-present
    // CPU plugin with fallback allowed - so a failure here is a malformed graph
    // out of brain's exporter, not an unavailable machine. Swallowing it as a
    // skip is exactly how a broken exporter reports a green suite.
    let mut graph = NpuGraph::compile_bytes(&bytes, &cfgv).unwrap_or_else(|e| {
        panic!("OpenVINO is present but refused brain's emitted Qwen-VL vision head ONNX graph: {e:?}")
    });
    let ovout = graph.run(&[("pixels", Feed::F32(&pixels, vec![n as i64, pv as i64]))]).expect("run head");
    let (_n, shape, data) = &ovout[0];
    eprintln!("visual_embeds out shape {shape:?} ({} elems)", data.len());
    assert_eq!(data.len(), reference.len(), "visual_embeds shape mismatch");

    let dot: f32 = data.iter().zip(&reference).map(|(x, y)| x * y).sum();
    let na: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = reference.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cosine = dot / (na * nb + 1e-12);
    let maxdiff = data.iter().zip(&reference).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    eprintln!("qwenvl vision-head ONNX(cpu) vs reference: cosine {cosine:.6}, maxdiff {maxdiff:.3e} (grid {gh}x{gw})");
    assert!(cosine > 0.999, "vision-head parity cosine {cosine} too low");
    assert!(maxdiff < 5e-2, "vision-head parity maxdiff {maxdiff} too high");
}
