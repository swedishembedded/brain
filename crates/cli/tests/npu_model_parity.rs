// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! TDD gate: the two forecast sessions (`Chronos2Session`/`FincastSession`)
//! are migrated onto the generic `npu::NpuModel` seam (`Chronos2NpuModel`/
//! `FincastNpuModel` in `resident_forecast.rs`), the same seam
//! `resident_depth.rs`'s `DepthNpuModel` already proves.
//!
//! `brain-cli` is a **bin-only** crate (no `[lib]` target), so an external
//! integration test cannot `use brain_cli::...`. This file pulls
//! `resident_forecast.rs` in directly via `#[path]` - it is compiled a second
//! time as part of THIS test binary's own crate, which is why the migrated
//! types (`Chronos2NpuModel`/`FincastNpuModel`/`Chronos2Resident`/
//! `FincastResident`) are reachable here as `pub(crate)`/`pub` items despite
//! `main.rs` never exporting them anywhere.

#[path = "../src/resident_forecast.rs"]
mod resident_forecast;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use capability::{Blob, Invocation, Media};
use npu::NpuModel as _;
use residency::{Device, MemCost, ResidentModel};
use resident_forecast::{Chronos2NpuModel, Chronos2Resident, FincastNpuModel, FincastResident};
use serde_json::json;

/// Cosine-similarity tolerance for an NPU-graph output vs its host `parity_ref`.
/// The graph is fp32 end to end when compiled on CPU-OpenVINO (no NPU rounding),
/// so an honest reproduction of the same math should be near-exact; anything
/// below this indicates a real graph/wiring bug, not floating-point noise.
/// (The real Intel-NPU fp16 path is a separate, looser tolerance question --
/// see `chronos2_export.rs`/`fincast_export.rs`'s own hardware-gated tests,
/// which use 0.99 for that reason.)
const PARITY_COSINE_MIN: f32 = 0.999;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (na * nb + 1e-9)
}

/// Per-process-unique, per-call-unique id for a scratch checkpoint path. `pid`
/// alone collides: `cargo test` runs every `#[test]` fn concurrently on
/// separate threads *within one process*, and both `tiny_chronos2`/
/// `tiny_fincast` are called from more than one test in this file -- a bare
/// pid-named temp path let one test's cleanup (`remove_file`) delete another,
/// still-running test's checkpoint out from under it.
fn unique_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    (std::process::id() as u64) << 32 | NEXT.fetch_add(1, Ordering::Relaxed)
}

fn write_f32_le(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn context_invocation(context: &[f32], horizon: i64) -> Invocation {
    Invocation::new()
        .set("horizon", json!(horizon))
        .blob("context", Blob::new(Media::Bytes, write_f32_le(context)).with_meta(json!({"shape": [context.len()]})))
}

// ============================== chronos2 ==============================

fn tiny_chronos2() -> (chronos2::Chronos2Config, chronos2::model::Chronos2, String) {
    let cfg = chronos2::Chronos2Config::tiny();
    let weights: HashMap<String, Vec<f32>> = cfg
        .param_list()
        .into_iter()
        .map(|(k, s)| {
            let n: usize = s.iter().product();
            let seed = k.len();
            (k, (0..n).map(|i| (((i + seed) as f32) * 0.1).sin() * 0.05).collect())
        })
        .collect();
    let model = chronos2::model::Chronos2::from_weights(cfg.clone(), &weights).unwrap();

    let tmp = std::env::temp_dir().join(format!("brain_c2_npu_parity_{}.safetensors", unique_id()));
    let tmp = tmp.to_str().unwrap().to_string();
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
        cfg.param_list().into_iter().map(|(k, shp)| (k.clone(), shp.iter().map(|&x| x as u64).collect(), weights[&k].clone())).collect();
    checkpoint::save(&tmp, cfg.to_json(), &tensors);
    (cfg, model, tmp)
}

/// Hardware-independent: `Chronos2NpuModel::build` (the ONNX graph construction)
/// and `parity_ref` need no OpenVINO/NPU at all -- this is the real RED-then-GREEN
/// half of the TDD gate. Before this phase, `Chronos2NpuModel` didn't exist
/// (compile error = RED); afterwards this is a deterministic, always-green check
/// that the trait seam is wired to the exact same math the pre-migration
/// `Chronos2Session` reproduced.
#[test]
fn chronos2_npu_model_builds_and_parity_ref_matches_core_forward() {
    let (cfg, model, path) = tiny_chronos2();
    let d = cfg.d_model;
    let (s, n_out) = (6usize, 2usize);
    let emb: Vec<f32> = (0..s * d).map(|i| ((i as f32) * 0.017).sin() * 0.1).collect();
    let kmask = vec![0.0f32; s];

    let npu_model = Chronos2NpuModel::new(&model, &path, s, n_out);
    let bytes = npu_model.onnx_bytes().expect("build chronos2 ONNX graph");
    assert!(bytes.len() > 1000, "chronos2 ONNX graph suspiciously small: {} bytes", bytes.len());

    let reference = model.core_forward(&emb, &kmask, n_out);
    let inputs: [(&str, Vec<f32>); 2] = [("emb", emb.clone()), ("kmask", kmask.clone())];
    let parity = npu_model.parity_ref(&inputs).expect("chronos2 parity_ref");
    assert_eq!(parity.len(), 1);
    assert_eq!(parity[0], reference, "parity_ref must call the exact same core_forward the graph was exported from");

    let _ = std::fs::remove_file(&path);
}

/// Hardware-gated: also actually compiles (`NpuGraph`, CPU-OpenVINO fallback
/// when no real NPU) and runs the graph, checking its output against
/// `parity_ref` by cosine similarity. Skips cleanly (does not fail) if no
/// OpenVINO runtime is reachable at all -- mirrors
/// `chronos2_export.rs::chronos2_session_matches_core_forward`'s own skip guard.
#[test]
fn chronos2_npu_graph_output_matches_parity_ref_when_openvino_available() {
    let (cfg, model, path) = tiny_chronos2();
    let d = cfg.d_model;
    let (s, n_out) = (6usize, 2usize);
    let emb: Vec<f32> = (0..s * d).map(|i| ((i as f32) * 0.017).sin() * 0.1).collect();
    let kmask = vec![0.0f32; s];

    let npu_model = Chronos2NpuModel::new(&model, &path, s, n_out);
    let ov_cfg = npu::openvino::NpuConfig { device: npu::openvino::NpuDevice::Cpu, allow_fallback: true, ..Default::default() };
    let mut graph = match npu_model.compile(&ov_cfg) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no OpenVINO runtime, skipping chronos2 NPU graph parity: {e}");
            let _ = std::fs::remove_file(&path);
            return;
        }
    };
    let out = graph
        .run(&[
            ("emb", npu::openvino::Feed::F32(&emb, vec![1, s as i64, d as i64])),
            ("kmask", npu::openvino::Feed::F32(&kmask, vec![1, 1, 1, s as i64])),
        ])
        .expect("chronos2 NPU graph infer");
    let (_name, _shape, data) = out.into_iter().next().expect("chronos2 NPU graph: no output");

    let inputs: [(&str, Vec<f32>); 2] = [("emb", emb), ("kmask", kmask)];
    let reference = npu_model.parity_ref(&inputs).expect("chronos2 parity_ref")[0].clone();
    let cos = cosine(&data, &reference);
    eprintln!("chronos2 NPU graph on {}: cosine {cos:.6}", graph.device());
    assert!(cos >= PARITY_COSINE_MIN, "chronos2 NPU graph vs parity_ref cosine {cos} below {PARITY_COSINE_MIN}");

    let _ = std::fs::remove_file(&path);
}

// =============================== fincast ===============================

fn tiny_fincast() -> (fincast::FincastConfig, fincast::model::Fincast, String) {
    let cfg = fincast::FincastConfig::tiny();
    let weights: HashMap<String, Vec<f32>> = cfg
        .param_list()
        .into_iter()
        .map(|(k, s)| {
            let n: usize = s.iter().product();
            let seed = k.len();
            (k, (0..n).map(|i| (((i + seed) as f32) * 0.1).sin() * 0.05).collect())
        })
        .collect();
    let model = fincast::model::Fincast::from_weights(cfg.clone(), &weights).unwrap();

    let tmp = std::env::temp_dir().join(format!("brain_fincast_npu_parity_{}.safetensors", unique_id()));
    let tmp = tmp.to_str().unwrap().to_string();
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
        cfg.param_list().into_iter().map(|(k, shp)| (k.clone(), shp.iter().map(|&x| x as u64).collect(), weights[&k].clone())).collect();
    checkpoint::save(&tmp, cfg.to_json(), &tensors);
    (cfg, model, tmp)
}

/// Hardware-independent: mirrors `chronos2_npu_model_builds_and_parity_ref_matches_core_forward`.
#[test]
fn fincast_npu_model_builds_and_parity_ref_matches_core_forward_amask() {
    let (cfg, model, path) = tiny_fincast();
    let d = cfg.hidden_size;
    let s = 6usize;
    let emb: Vec<f32> = (0..s * d).map(|i| ((i as f32) * 0.017).sin() * 0.1).collect();
    let mut amask = vec![0.0f32; s * s];
    for i in 0..s {
        for j in 0..s {
            if j > i {
                amask[i * s + j] = -1.0e9;
            }
        }
    }

    let npu_model = FincastNpuModel::new(&model, &path, s);
    let bytes = npu_model.onnx_bytes().expect("build fincast ONNX graph");
    assert!(bytes.len() > 1000, "fincast ONNX graph suspiciously small: {} bytes", bytes.len());

    let reference = model.core_forward_amask(&emb, &amask);
    let inputs: [(&str, Vec<f32>); 2] = [("emb", emb.clone()), ("amask", amask.clone())];
    let parity = npu_model.parity_ref(&inputs).expect("fincast parity_ref");
    assert_eq!(parity.len(), 1);
    assert_eq!(parity[0], reference, "parity_ref must call the exact same core_forward_amask the graph was exported from");

    let _ = std::fs::remove_file(&path);
}

/// Hardware-gated: mirrors `chronos2_npu_graph_output_matches_parity_ref_when_openvino_available`.
/// Fincast compiles via the external-data path (`FincastNpuModel::compile`'s
/// override) even at this tiny scale, so this also exercises that code path.
#[test]
fn fincast_npu_graph_output_matches_parity_ref_when_openvino_available() {
    let (cfg, model, path) = tiny_fincast();
    let d = cfg.hidden_size;
    let s = 6usize;
    let emb: Vec<f32> = (0..s * d).map(|i| ((i as f32) * 0.017).sin() * 0.1).collect();
    let mut amask = vec![0.0f32; s * s];
    for i in 0..s {
        for j in 0..s {
            if j > i {
                amask[i * s + j] = -1.0e9;
            }
        }
    }

    let npu_model = FincastNpuModel::new(&model, &path, s);
    let ov_cfg = npu::openvino::NpuConfig { device: npu::openvino::NpuDevice::Cpu, allow_fallback: true, ..Default::default() };
    let mut graph = match npu_model.compile(&ov_cfg) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no OpenVINO runtime, skipping fincast NPU graph parity: {e}");
            let _ = std::fs::remove_file(&path);
            return;
        }
    };
    let out = graph
        .run(&[
            ("emb", npu::openvino::Feed::F32(&emb, vec![1, s as i64, d as i64])),
            ("amask", npu::openvino::Feed::F32(&amask, vec![1, 1, s as i64, s as i64])),
        ])
        .expect("fincast NPU graph infer");
    let (_name, _shape, data) = out.into_iter().next().expect("fincast NPU graph: no output");

    let inputs: [(&str, Vec<f32>); 2] = [("emb", emb), ("amask", amask)];
    let reference = npu_model.parity_ref(&inputs).expect("fincast parity_ref")[0].clone();
    let cos = cosine(&data, &reference);
    eprintln!("fincast NPU graph on {}: cosine {cos:.6}", graph.device());
    assert!(cos >= PARITY_COSINE_MIN, "fincast NPU graph vs parity_ref cosine {cos} below {PARITY_COSINE_MIN}");

    let _ = std::fs::remove_file(&path);
}

// ========================= ResidentModel contract =========================

/// Both migrated foundation-model residents must (a) advertise an NPU
/// footprint via `MemCost::with_npu` (`estimate`) -- this is precisely how a
/// model opts into NPU auto-placement (`residency::place::pick_device`) -- and
/// (b) never let an NPU compile/infer failure escape as a Rust panic: a bad
/// device, a missing OpenVINO runtime, or (in this sandbox, most likely) a
/// present-but-non-functional NPU device node must surface as a clean,
/// catchable `Err`, never take down the shared NPU lane thread (see
/// `resident_forecast.rs`'s own `guard_npu` doc).
///
/// Unlike `resident_depth.rs`'s `DepthNpuModel` (which compiles inside
/// `activate()` itself, so a bad NPU fails `activate` directly), the forecast
/// residents defer compilation to the first `run()` call (the compiled-graph
/// cache lives inside the `Instance`, keyed on the request's actual context
/// length) -- so `activate(Device::Npu)` succeeding here is the correct,
/// unchanged contract, not a regression; the "never panics, always a typed
/// Err on failure" guarantee is checked at `run()`, where the real NPU work
/// happens.
///
/// This sandbox has a `/dev/accel` node (so `Inventory::probe().npus == 1`)
/// but, per prior investigation, no working NPU firmware -- `NpuConfig`'s
/// `allow_fallback` means OpenVINO transparently retargets the compile to its
/// GPU plugin (this box also has a usable Intel iGPU) instead of erroring, so
/// the `Ok` branch below -- not the `Err` one -- is what actually runs here.
/// Either branch is accepted, but when `Ok`, this additionally cross-checks
/// the NPU-resident output against the same resident activated on
/// `Device::Cpu` (the plain `gpu_core` path) for the identical input --
/// the "pluggable core, bit-comparable" contract this module's own doc
/// comment promises, exercised end to end through the public
/// `ResidentModel`/`Instance` surface rather than the lower-level
/// `NpuModel::parity_ref` the tests above already cover directly.
#[test]
fn forecast_residents_advertise_npu_and_never_panic_on_npu_activation() {
    let (_c2cfg, _c2model, c2path) = tiny_chronos2();
    let (_fccfg, _fcmodel, fcpath) = tiny_fincast();

    let npus = gpu_core::devices::Inventory::probe().npus;
    eprintln!("Inventory::probe().npus = {npus} (device-node count, not a firmware/functionality guarantee)");

    check_one("chronos2", Chronos2Resident::new(&c2path), &[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    check_one("fincast", FincastResident::new(&fcpath), &[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);

    let _ = std::fs::remove_file(&c2path);
    let _ = std::fs::remove_file(&fcpath);
}

fn check_one(name: &str, resident: impl ResidentModel, context: &[f32]) {
    let inv = Invocation::new();
    let key = resident.instance_key("forecast", &inv);

    let cost: MemCost = resident.estimate(&key);
    assert!(cost.npu > 0, "{name}: estimate() must advertise a nonzero NPU footprint (MemCost::with_npu)");

    let mut instance = resident.activate(&key, Device::Npu(0)).unwrap_or_else(|e| panic!("{name}: activate(Npu(0)) failed: {e}"));

    let run_inv = context_invocation(context, 4);
    let mut progress = |_p: capability::Progress| {};
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| instance.run("forecast", &run_inv, &mut progress)));
    let result = result.unwrap_or_else(|_| panic!("{name}: run() on Device::Npu(0) panicked instead of returning a typed Err"));

    match result {
        Err(e) => {
            eprintln!("{name}: NPU run() cleanly errored (expected without working NPU hardware): {e}");
        }
        Ok(outcome) => {
            eprintln!("{name}: NPU run() succeeded -- real NPU/OpenVINO hardware is usable in this sandbox");
            let npu_blob =
                outcome.blobs.get("forecast").unwrap_or_else(|| panic!("{name}: NPU forecast outcome missing its 'forecast' blob"));
            let npu_data = read_f32_le(&npu_blob.bytes);

            // Cross-check against the SAME resident on Device::Cpu for the identical
            // input -- the "pluggable core, bit-comparable" contract this module's
            // doc promises, exercised through the public ResidentModel/Instance
            // surface (NpuModel::parity_ref is already covered directly above).
            let mut cpu_instance = resident.activate(&key, Device::Cpu).unwrap_or_else(|e| panic!("{name}: activate(Cpu) failed: {e}"));
            let cpu_outcome =
                cpu_instance.run("forecast", &run_inv, &mut progress).unwrap_or_else(|e| panic!("{name}: CPU run() failed: {e}"));
            let cpu_blob = cpu_outcome.blobs.get("forecast").unwrap_or_else(|| panic!("{name}: CPU outcome missing its 'forecast' blob"));
            let cpu_data = read_f32_le(&cpu_blob.bytes);

            let cos = cosine(&npu_data, &cpu_data);
            eprintln!("{name}: NPU-resident vs CPU/GPU-resident forecast cosine {cos:.6}");
            assert!(cos >= PARITY_COSINE_MIN, "{name}: NPU-resident output vs CPU/GPU reference cosine {cos} below {PARITY_COSINE_MIN}");
        }
    }
}

fn read_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect()
}
