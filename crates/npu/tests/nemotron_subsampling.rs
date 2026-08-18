// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity gate for the Nemotron FastConformer **subsampling** ONNX stage: compile it
//! on the OpenVINO **CPU** device (fp32, tight tolerance) and check the output against
//! the dumped HF golden. Skips without OpenVINO or the testdata checkpoint/golden.
//!
//! Run: `make fetch/testdata` then
//! `LD_LIBRARY_PATH=<openvino/libs> cargo test -p brain-npu --test nemotron_subsampling -- --nocapture`

use std::collections::HashMap;
use std::path::Path;

use npu::openvino::{available_devices, Feed, NpuConfig, NpuDevice, NpuGraph, PerfHint};
use npu::NemotronTopo;

use brain_testutil::{model_dir, testdata};

fn read_f32(p: &str) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|_| panic!("missing {p}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn subsampling_matches_golden_on_cpu() {
    let ckpt = model_dir("nvidia/nemotron-3.5-asr-streaming-0.6b").unwrap_or_default();
    let gold = testdata("asr/golden/nemotron");
    if !Path::new(&format!("{ckpt}/model.safetensors")).exists() || !Path::new(&format!("{gold}/subsampling.f32")).exists() {
        brain_testutil::skip("nemotron checkpoint/golden absent (run `make fetch/testdata`)");
        return;
    }
    if available_devices().map(|d| d.is_empty()).unwrap_or(true) {
        brain_testutil::skip_unavailable("no OpenVINO runtime");
        return;
    }

    let topo = NemotronTopo::default();
    let nmel = topo.num_mel_bins as usize;
    let mel = read_f32(&format!("{gold}/input_features.f32")); // [T, 128]
    let t = (mel.len() / nmel) as u32;
    // valid frames = frames the frontend mask did not zero (masked frames are exactly 0)
    let valid = (0..t as usize).filter(|&i| mel[i * nmel..(i + 1) * nmel].iter().any(|&v| v != 0.0)).count() as u32;
    let ref_sub = read_f32(&format!("{gold}/subsampling.f32")); // [T', 1024]

    // weights → name→f32 map (impls WeightSource)
    let tensors = checkpoint::safetensors::read_model_dir(Path::new(&ckpt)).expect("read checkpoint");
    let weights: HashMap<String, Vec<f32>> = tensors.into_iter().map(|t| (t.name, t.data)).collect();

    // build the subsampling graph: mel [1,1,T,128] -> [T', 1024]
    let tsub = topo.subsampled_len(valid).max(topo.subsampled_len(t)); // graph time = full-length subsample of t
    let tt = topo.subsampled_len(t);
    let mut g = onnx::GraphBuilder::new("nemotron_subsampling");
    g.input_f32("mel", &[1, 1, t as i64, nmel as i64]);
    npu::build_subsampling(&mut g, &topo, &weights, t, valid, "mel", "subsampling");
    g.output_f32("subsampling", &[tt as i64, topo.hidden as i64]);
    let bytes = g.finish_with(onnx::DEFAULT_OPSET, onnx::DEFAULT_IR_VERSION);
    let _ = tsub;

    // compile fp32 on CPU for a tight parity check (device precision-independent).
    let cfg = NpuConfig { device: NpuDevice::Cpu, perf_hint: PerfHint::Latency, allow_fallback: true, ..Default::default() };
    let mut graph = NpuGraph::compile_bytes(&bytes, &cfg).expect("compile subsampling on CPU");
    let out = graph.run(&[("mel", Feed::F32(&mel, vec![1, 1, t as i64, nmel as i64]))]).expect("run");
    let (_n, shape, data) = &out[0];
    eprintln!("subsampling out shape {shape:?}, golden {} elems (T'={tt})", ref_sub.len());
    assert_eq!(data.len(), ref_sub.len(), "shape mismatch");

    let stat = |v: &[f32]| {
        let (mut mn, mut mx, mut s) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f32);
        for &x in v {
            mn = mn.min(x);
            mx = mx.max(x);
            s += x;
        }
        (mn, mx, s / v.len() as f32)
    };
    eprintln!("T={t} valid={valid}");
    eprintln!("ONNX  min/max/mean = {:?}", stat(data));
    eprintln!("golden min/max/mean = {:?}", stat(&ref_sub));
    eprintln!("ONNX[0..6]   = {:?}", &data[..6]);
    eprintln!("golden[0..6] = {:?}", &ref_sub[..6]);
    let maxdiff = data.iter().zip(&ref_sub).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("nemotron subsampling ONNX(cpu) maxdiff vs golden = {maxdiff:.3e}");
    assert!(maxdiff < 3e-3, "subsampling parity maxdiff {maxdiff}");
}
