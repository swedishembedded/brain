// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Time-series forecasting (chronos2 / fincast / kronos) behind the residency
//! scheduler + the Brain1 D-Bus surface.
//!
//! Each foundation model is a [`ResidentModel`] exposing one `forecast` action:
//! a context series in (f32-LE blob + `{shape}` meta), a forecast tensor out
//! (f32-LE blob + `{shape,kind,levels}` meta). The dispatch is fully generic, so
//! this is all that is needed to reach these models over `Run` — see
//! `docs/serving-contract.md`.
//!
//! **NPU placement.** chronos2 and fincast advertise an NPU footprint
//! (`MemCost::with_npu`), so `place::pick_device` auto-schedules them on the NPU
//! when one is budgeted. `activate(Device::Npu)` wraps the model's pluggable-core
//! seam (`forecast_quantiles_with_core` / `forecast_full_with_core`) onto the
//! bespoke OpenVINO session (`Chronos2Session` / `FincastSession`), caching a
//! compiled session per context-length bucket. Every other device runs the exact
//! same math on `gpu_core` (`forecast_quantiles` / `forecast_full`), so the NPU
//! and CPU/GPU paths are bit-comparable. kronos (autoregressive OHLCV rollout)
//! is served on CPU/GPU here; its two-graph NPU rollout is a follow-up.

use std::cell::RefCell;
use std::collections::HashMap;

use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use npu::openvino::{Chronos2Session, FincastSession, NpuConfig, NpuDevice};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;

// ============================ shared wire codec ============================

/// Decode a numeric input blob: raw f32 little-endian + meta `{"shape":[...]}`.
/// A missing shape is treated as a 1-D `[len]` series.
fn decode_f32(inv: &Invocation, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let blob = inv.get_blob(name).ok_or_else(|| format!("forecast: missing input '{name}'"))?;
    if blob.bytes.len() % 4 != 0 {
        return Err(format!("forecast: input '{name}' is not a whole number of f32"));
    }
    let data: Vec<f32> = blob.bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
    let shape = blob
        .meta
        .get("shape")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect::<Vec<_>>())
        .filter(|s: &Vec<usize>| !s.is_empty() && s.iter().product::<usize>() == data.len())
        .unwrap_or_else(|| vec![data.len()]);
    Ok((data, shape))
}

/// Decode an optional u32-LE blob (e.g. kronos calendar stamps).
fn decode_u32_opt(inv: &Invocation, name: &str) -> Option<Vec<u32>> {
    let blob = inv.get_blob(name)?;
    Some(blob.bytes.chunks_exact(4).map(|q| u32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect())
}

/// Encode a forecast tensor as raw f32-LE + meta `{shape,dtype,kind,levels}`.
fn encode_forecast(data: &[f32], shape: Vec<usize>, kind: &str, levels: &[f32]) -> Blob {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    Blob::new(Media::Bytes, bytes).with_meta(json!({
        "shape": shape, "dtype": "f32le", "kind": kind, "levels": levels,
    }))
}

fn horizon_of(inv: &Invocation, default: i64) -> usize {
    inv.get_i64("horizon").unwrap_or(default).max(1) as usize
}

/// The base `forecast` action schema shared by every foundation model: a context
/// series in, a horizon, a forecast tensor out. Callers append model-specific
/// params (fincast `freq`, kronos calendar stamps).
fn base_forecast_spec(summary: &str) -> ActionSpec {
    ActionSpec::new("forecast", summary)
        .param(ParamSpec::new("horizon", ParamType::Int, "number of steps to forecast").default(json!(64)))
        .input(BlobSpec::new("context", Media::Bytes, "context series as raw f32-LE; meta {shape}").required())
        .output(BlobSpec::new("forecast", Media::Bytes, "forecast as raw f32-LE; meta {shape,kind,levels}"))
}

/// Hot-footprint RAM estimate for a weights file (~1.3× the on-disk size for the
/// f32 unpack + index overhead), mirroring the depth resident.
fn file_ram(path: &str) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0).saturating_mul(13) / 10
}

fn npu_cfg() -> NpuConfig {
    NpuConfig { device: NpuDevice::Npu, allow_fallback: true, ..Default::default() }
}

// ================================ chronos2 =================================

/// Chronos-2 universal forecaster behind the scheduler. `BRAIN_CHRONOS2` = the
/// brain-format weights. Emits the model's native 21 quantile levels.
pub struct Chronos2Resident {
    path: String,
}

impl Chronos2Resident {
    pub fn from_env() -> Option<Chronos2Resident> {
        std::env::var("BRAIN_CHRONOS2").ok().filter(|p| !p.is_empty()).map(|path| Chronos2Resident { path })
    }
}

impl ResidentModel for Chronos2Resident {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            "chronos2",
            "probabilistic time-series forecasting (Chronos-2); 21 quantile levels",
            vec![base_forecast_spec("probabilistic forecast; forecast blob is [levels, horizon] quantile-major")],
        )
    }
    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        // One hot instance per horizon; the NPU session cache inside the instance
        // keys the compiled graph on the context-length bucket.
        InstanceKey::new("chronos2", format!("h{}", horizon_of(inv, 64)))
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // Weights in RAM for the gpu_core path; a compiled fp16 core graph on the
        // NPU (npu > 0 => NPU-eligible; place::pick_device schedules it there).
        MemCost::new(0, file_ram(&self.path)).with_npu(512 << 20)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if let Device::Gpu(i) = device {
            std::env::set_var("BRAIN_GPU_INDEX", i.to_string());
        }
        let model = chronos2::model::Chronos2::load(&self.path)?;
        if let Device::Npu(_) = device {
            return Ok(Box::new(Chronos2NpuInstance {
                model,
                path: self.path.clone(),
                cfg: npu_cfg(),
                cache: RefCell::new(HashMap::new()),
                device: RefCell::new("npu".into()),
            }));
        }
        Ok(Box::new(Chronos2CpuInstance { model }))
    }
}

/// chronos2 on gpu_core (CPU or GPU per the process backend).
struct Chronos2CpuInstance {
    model: chronos2::model::Chronos2,
}

impl Instance for Chronos2CpuInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (context, _shape) = decode_f32(inv, "context")?;
        let horizon = horizon_of(inv, 64);
        let q = self.model.forecast_quantiles(&context, horizon); // [21, horizon] quantile-major
        Ok(chronos2_outcome(q, horizon, "gpu_core"))
    }
}

/// chronos2 on the NPU: the transformer core runs on OpenVINO via the pluggable
/// seam; the host does scaler/patch/embed + head/denorm. Sessions are cached per
/// `(context_len, n_out)` so repeated same-shape requests skip recompilation.
struct Chronos2NpuInstance {
    model: chronos2::model::Chronos2,
    path: String,
    cfg: NpuConfig,
    cache: RefCell<HashMap<(usize, usize), Chronos2Session>>,
    device: RefCell<String>,
}

impl Instance for Chronos2NpuInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (context, _shape) = decode_f32(inv, "context")?;
        let horizon = horizon_of(inv, 64);
        let d = self.model.config().d_model;
        let (path, cfg, cache, devcell) = (&self.path, &self.cfg, &self.cache, &self.device);
        let q = self.model.forecast_quantiles_with_core(&context, horizon, |emb, mask, n_out| {
            let s = emb.len() / d;
            let mut c = cache.borrow_mut();
            let sess = c.entry((s, n_out)).or_insert_with(|| {
                let bytes = npu::chronos2_export::export_onnx(path, s, n_out, npu::qwen_topology::Quant::F32)
                    .expect("chronos2 NPU export");
                Chronos2Session::load_bytes(&bytes, cfg).expect("chronos2 NPU compile")
            });
            *devcell.borrow_mut() = sess.device().to_string();
            sess.run(emb, mask).expect("chronos2 NPU infer")
        });
        let dev = self.device.borrow().clone();
        Ok(chronos2_outcome(q, horizon, &dev))
    }
}

fn chronos2_outcome(q: Vec<f32>, horizon: usize, device: &str) -> Outcome {
    let levels = chronos2::QUANTILES;
    Outcome::new()
        .set("model", json!("chronos2"))
        .set("horizon", json!(horizon))
        .set("device", json!(device))
        .blob("forecast", encode_forecast(&q, vec![levels.len(), horizon], "quantiles", &levels))
}

// ================================= fincast =================================

/// FinCast financial forecaster behind the scheduler. `BRAIN_FINCAST` = the
/// brain-format weights. `freq` selects the frequency bucket (0/1/2). Emits the
/// full head: `[horizon, num_outputs]` (col 0 = mean, cols 1.. = 9 quantiles).
pub struct FincastResident {
    path: String,
}

impl FincastResident {
    pub fn from_env() -> Option<FincastResident> {
        std::env::var("BRAIN_FINCAST").ok().filter(|p| !p.is_empty()).map(|path| FincastResident { path })
    }
    fn spec() -> ActionSpec {
        base_forecast_spec("financial forecast; forecast blob is [horizon, 1+levels] (col 0 mean)")
            .param(ParamSpec::new("freq", ParamType::Int, "frequency bucket: 0 daily, 1 weekly, 2 monthly").default(json!(0)))
    }
}

impl ResidentModel for FincastResident {
    fn manifest(&self) -> Manifest {
        Manifest::new("fincast", "financial time-series forecasting (FinCast); mean + 9 quantiles", vec![Self::spec()])
    }
    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        let freq = inv.get_i64("freq").unwrap_or(0);
        InstanceKey::new("fincast", format!("h{}f{freq}", horizon_of(inv, 64)))
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::new(0, file_ram(&self.path)).with_npu(512 << 20)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if let Device::Gpu(i) = device {
            std::env::set_var("BRAIN_GPU_INDEX", i.to_string());
        }
        let model = fincast::model::Fincast::load(&self.path)?;
        if let Device::Npu(_) = device {
            return Ok(Box::new(FincastNpuInstance {
                model,
                path: self.path.clone(),
                cfg: npu_cfg(),
                cache: RefCell::new(HashMap::new()),
                device: RefCell::new("npu".into()),
            }));
        }
        Ok(Box::new(FincastCpuInstance { model }))
    }
}

struct FincastCpuInstance {
    model: fincast::model::Fincast,
}

impl Instance for FincastCpuInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (context, _shape) = decode_f32(inv, "context")?;
        let horizon = horizon_of(inv, 64);
        let freq = inv.get_i64("freq").unwrap_or(0).max(0) as usize;
        let out = self.model.forecast_full(&context, freq, horizon); // [horizon, num_outputs]
        Ok(fincast_outcome(&self.model, out, horizon, "gpu_core"))
    }
}

struct FincastNpuInstance {
    model: fincast::model::Fincast,
    path: String,
    cfg: NpuConfig,
    cache: RefCell<HashMap<usize, FincastSession>>,
    device: RefCell<String>,
}

impl Instance for FincastNpuInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (context, _shape) = decode_f32(inv, "context")?;
        let horizon = horizon_of(inv, 64);
        let freq = inv.get_i64("freq").unwrap_or(0).max(0) as usize;
        let (path, cfg, cache, devcell) = (&self.path, &self.cfg, &self.cache, &self.device);
        let out = self.model.forecast_full_with_core(&context, freq, horizon, |emb, amask| {
            let s = (amask.len() as f64).sqrt() as usize;
            let mut c = cache.borrow_mut();
            let sess = c.entry(s).or_insert_with(|| {
                let bytes = npu::fincast_export::export_onnx(path, s, npu::qwen_topology::Quant::F32)
                    .expect("fincast NPU export");
                FincastSession::load_bytes(&bytes, cfg).expect("fincast NPU compile")
            });
            *devcell.borrow_mut() = sess.device().to_string();
            sess.run(emb, amask).expect("fincast NPU infer")
        });
        let dev = self.device.borrow().clone();
        Ok(fincast_outcome(&self.model, out, horizon, &dev))
    }
}

fn fincast_outcome(model: &fincast::model::Fincast, out: Vec<f32>, horizon: usize, device: &str) -> Outcome {
    let no = model.config().num_outputs();
    Outcome::new()
        .set("model", json!("fincast"))
        .set("horizon", json!(horizon))
        .set("device", json!(device))
        .blob("forecast", encode_forecast(&out, vec![horizon, no], "mean+quantiles", &fincast::QUANTILES))
}

// ================================== kronos =================================

/// Kronos autoregressive OHLCV forecaster behind the scheduler.
/// `BRAIN_KRONOS_TOKENIZER` + `BRAIN_KRONOS_DECODER` = the two checkpoint dirs.
/// Input `context` is the OHLCV bar matrix `[T, feat]`; optional `ctx_stamp` /
/// `fut_stamp` are calendar stamps `[·, 5]` u32 (zeros if absent). Output is the
/// generated bars `[horizon, feat]`. CPU/GPU only for now (the two-graph NPU
/// rollout is a follow-up).
pub struct KronosResident {
    tokenizer: String,
    decoder: String,
}

impl KronosResident {
    pub fn from_env() -> Option<KronosResident> {
        let tokenizer = std::env::var("BRAIN_KRONOS_TOKENIZER").ok().filter(|p| !p.is_empty())?;
        let decoder = std::env::var("BRAIN_KRONOS_DECODER").ok().filter(|p| !p.is_empty())?;
        Some(KronosResident { tokenizer, decoder })
    }
    fn spec() -> ActionSpec {
        base_forecast_spec("OHLCV forecast; context is [T, feat] bars, forecast is [horizon, feat] samples")
            .param(ParamSpec::new("temperature", ParamType::Float, "sampling temperature (0 or argmax=true => deterministic)").default(json!(1.0)))
            .param(ParamSpec::new("argmax", ParamType::Bool, "deterministic argmax decode").default(json!(true)))
            .param(ParamSpec::new("seed", ParamType::Int, "RNG seed when sampling").default(json!(0)))
            .input(BlobSpec::new("ctx_stamp", Media::Bytes, "optional context calendar stamps [T,5] u32-LE"))
            .input(BlobSpec::new("fut_stamp", Media::Bytes, "optional future calendar stamps [horizon,5] u32-LE"))
    }
}

impl ResidentModel for KronosResident {
    fn manifest(&self) -> Manifest {
        Manifest::new("kronos", "autoregressive OHLCV forecasting (Kronos)", vec![Self::spec()])
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // The AR decoder handles any horizon on one hot instance.
        InstanceKey::new("kronos", "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // tokenizer + decoder weight dirs, resident in RAM. NPU rollout is a
        // follow-up, so no NPU footprint yet (stays on CPU/GPU).
        let ram = dir_ram(&self.tokenizer) + dir_ram(&self.decoder);
        MemCost::new(0, ram)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if let Device::Gpu(i) = device {
            std::env::set_var("BRAIN_GPU_INDEX", i.to_string());
        }
        let model = kronos::import::load_model(&self.tokenizer, &self.decoder)?;
        Ok(Box::new(KronosInstance { model }))
    }
}

fn dir_ram(dir: &str) -> u64 {
    std::fs::read_dir(dir)
        .map(|rd| rd.flatten().filter_map(|e| e.metadata().ok().map(|m| m.len())).sum::<u64>())
        .unwrap_or(0)
        .saturating_mul(13)
        / 10
}

struct KronosInstance {
    model: kronos::generate::KronosModel,
}

impl Instance for KronosInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let feat = self.model.feat();
        let (raw, shape) = decode_f32(inv, "context")?;
        // Accept either full OHLCV bars `[T, feat]`, or a univariate close series
        // `[T]` / `[T,1]` which we expand to bars (o=h=l=c=close; the remaining
        // features — volume, amount — held at 1.0) so a caller can forecast from a
        // single series exactly like chronos2/fincast.
        let (bars, t) = if shape.len() == 2 && shape[1] == feat {
            (raw, shape[0])
        } else if shape.len() <= 1 || (shape.len() == 2 && shape[1] == 1) {
            let t = raw.len();
            let mut bars = Vec::with_capacity(t * feat);
            for &c in &raw {
                let mut row = vec![c; 4.min(feat)]; // o,h,l,c
                row.resize(feat, 1.0);
                bars.extend(row);
            }
            (bars, t)
        } else {
            return Err(format!("kronos: context must be [T,{feat}] OHLCV bars or a univariate [T] series; got {shape:?}"));
        };
        let horizon = horizon_of(inv, 64);
        // Calendar stamps: use the client's if provided, else zeros (calendar-agnostic).
        let ctx_stamp = decode_u32_opt(inv, "ctx_stamp").unwrap_or_else(|| vec![0u32; t * 5]);
        let fut_stamp = decode_u32_opt(inv, "fut_stamp").unwrap_or_else(|| vec![0u32; horizon * 5]);
        if ctx_stamp.len() != t * 5 {
            return Err(format!("kronos: ctx_stamp must be [{t},5] u32, got {}", ctx_stamp.len()));
        }
        if fut_stamp.len() != horizon * 5 {
            return Err(format!("kronos: fut_stamp must be [{horizon},5] u32, got {}", fut_stamp.len()));
        }
        let opts = kronos::generate::GenOpts {
            temperature: inv.get_f64("temperature").unwrap_or(1.0) as f32,
            argmax: inv.get_bool("argmax").unwrap_or(true),
            seed: inv.get_i64("seed").unwrap_or(0) as u64,
            ..Default::default()
        };
        let out = self.model.forecast(&bars, &ctx_stamp, &fut_stamp, horizon, &opts); // [horizon, feat]
        Ok(Outcome::new()
            .set("model", json!("kronos"))
            .set("horizon", json!(horizon))
            .set("device", json!("gpu_core"))
            .blob("forecast", encode_forecast(&out, vec![horizon, feat], "samples", &[])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Weight-free schema checks (the serving contract's cheap capability test).
    #[test]
    fn forecast_manifests_are_well_formed() {
        let specs = [
            Chronos2Resident { path: String::new() }.manifest(),
            FincastResident { path: String::new() }.manifest(),
            KronosResident { tokenizer: String::new(), decoder: String::new() }.manifest(),
        ];
        for m in &specs {
            let a = m.actions.iter().find(|a| a.name == "forecast").expect("has a forecast action");
            assert!(a.inputs.iter().any(|b| b.name == "context" && b.required), "{}: required context input", m.model);
            assert!(a.outputs.iter().any(|b| b.name == "forecast"), "{}: forecast output", m.model);
            assert!(a.params.iter().any(|p| p.name == "horizon"), "{}: horizon param", m.model);
        }
    }

    #[test]
    fn f32_codec_roundtrips_with_shape() {
        let inv = Invocation::new().blob(
            "context",
            Blob::new(Media::Bytes, [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect())
                .with_meta(json!({"shape": [2, 2]})),
        );
        let (data, shape) = decode_f32(&inv, "context").unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, vec![2, 2]);
    }
}
