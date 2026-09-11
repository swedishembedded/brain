// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Time-series forecasting (chronos2 / fincast / kronos) behind the residency
//! scheduler + the Brain1 D-Bus surface.
//!
//! Each foundation model is a [`ResidentModel`] exposing one `forecast` action:
//! a context series in (f32-LE blob + `{shape}` meta), a forecast tensor out
//! (f32-LE blob + `{shape,kind,levels}` meta). The dispatch is fully generic, so
//! this is all that is needed to reach these models over `Run`.
//!
//! **NPU placement.** chronos2 and fincast advertise an NPU footprint
//! (`MemCost::with_npu`), so `place::pick_device` auto-schedules them on the NPU
//! when one is budgeted. `activate(Device::Npu)` wraps the model's pluggable-core
//! seam (`forecast_quantiles_with_core` / `forecast_full_with_core`) onto the
//! generic `npu::NpuModel` seam (`Chronos2NpuModel` / `FincastNpuModel`, below -
//! the same seam `resident_depth.rs`'s `DepthNpuModel` proves), compiling through
//! `npu::openvino::NpuGraph` and caching the compiled graph per context-length
//! bucket. Every other device runs the exact same math on `gpu_core`
//! (`forecast_quantiles` / `forecast_full`), so the NPU and CPU/GPU paths are
//! bit-comparable. kronos (autoregressive OHLCV rollout) is served on CPU/GPU
//! here; its two-graph NPU rollout is a follow-up.

use std::cell::RefCell;
use std::collections::HashMap;

use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use npu::openvino::{Feed, NpuConfig, NpuDevice, NpuGraph};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;

/// Catalog ids for the three forecasters. No dedicated `caps.rs` exists for
/// these crates (unlike yolo/depth/tts/…) -- these three consts are the
/// analogous single source of truth here (also referenced by `perf_cli.rs`'s
/// `ExecutorTarget` construction, which must route to the same catalog id).
pub(crate) const CHRONOS2_MODEL: &str = "brain/chronos2";
pub(crate) const FINCAST_MODEL: &str = "brain/fincast";
pub(crate) const KRONOS_MODEL: &str = "brain/kronos";
pub(crate) const TIMESFM3_MODEL: &str = "brain/timesfm3";

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

/// `decode_f32`, but `Ok(None)` when the named blob is simply absent - an
/// optional wire input, not an error.
fn decode_f32_opt(inv: &Invocation, name: &str) -> Result<Option<(Vec<f32>, Vec<usize>)>, String> {
    if inv.get_blob(name).is_none() {
        return Ok(None);
    }
    decode_f32(inv, name).map(Some)
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

/// [`encode_forecast`] plus a `names` meta entry - one name per leading-axis
/// slice, for a multi-target tensor where position alone does not say which
/// target is which.
fn encode_named_forecast(data: &[f32], shape: Vec<usize>, kind: &str, levels: &[f32], names: &[String]) -> Blob {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    Blob::new(Media::Bytes, bytes).with_meta(json!({
        "shape": shape, "dtype": "f32le", "kind": kind, "levels": levels, "names": names,
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

/// Run an NPU forecast, converting a panic (e.g. an OpenVINO compile/infer
/// failure surfaced through `.expect` in the pluggable-core closure) into a clean
/// error. Without this, one model's NPU failure would unwind and kill the shared
/// NPU lane thread — taking every other NPU-scheduled model down with it. A
/// RefCell borrow held at panic time is released during unwind, so the cached
/// sessions stay usable for the next request.
fn guard_npu<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .map_err(|_| "NPU forecast failed (compile/infer error — see stderr)".to_string())
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
    /// Explicit `.safetensors` path (the `brain perf` target and non-env callers).
    pub fn new(path: &str) -> Chronos2Resident {
        Chronos2Resident { path: path.to_string() }
    }
}

/// The static (weights-free) Chronos-2 manifest — the catalog's discovery entry
/// (`brain caps`), shared with [`ResidentModel::manifest`] so the two cannot drift.
pub(crate) fn chronos2_manifest() -> Manifest {
    Manifest::new(
        CHRONOS2_MODEL,
        "probabilistic time-series forecasting (Chronos-2); 21 quantile levels",
        vec![base_forecast_spec("probabilistic forecast; forecast blob is [levels, horizon] quantile-major")],
    )
}

impl ResidentModel for Chronos2Resident {
    fn manifest(&self) -> Manifest {
        chronos2_manifest()
    }
    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        // One hot instance per horizon; the NPU session cache inside the instance
        // keys the compiled graph on the context-length bucket.
        InstanceKey::new(CHRONOS2_MODEL, format!("h{}", horizon_of(inv, 64)))
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // Weights in RAM for the gpu_core path; a compiled fp16 core graph on the
        // NPU (npu > 0 => NPU-eligible; place::pick_device schedules it there).
        // vram == ram: the transformer core runs on gpu_core, so it is placeable
        // on a GPU (incl. the integrated GPU) as well as CPU; NPU stays preferred
        // (place::pick_device tries NPU, then GPU, then CPU).
        let r = file_ram(&self.path);
        MemCost::new(r, r).with_npu(512 << 20)
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

/// The Chronos-2 transformer core as a generic [`npu::NpuModel`]: the exact same
/// ONNX topology `npu::chronos2_export::export_onnx` builds
/// (`npu::chronos2_topology::build_chronos2_graph_quant`), reached through the
/// trait seam instead of a bespoke session - mirrors `resident_depth.rs`'s
/// `DepthNpuModel`. `parity_ref` is the model's own device `core_forward`, the
/// exact function the ONNX graph was built to reproduce.
///
/// `pub(crate)`: `crates/cli/tests/npu_model_parity.rs` pulls this file in via
/// `#[path]` and constructs this type directly - `brain-cli` is a bin-only
/// crate (no `[lib]` target an external integration test could depend on
/// instead), so `#[path]` inclusion is the only way to unit-test a private
/// `main.rs` module from `tests/`.
pub(crate) struct Chronos2NpuModel<'a> {
    model: &'a chronos2::model::Chronos2,
    path: &'a str,
    s: usize,
    n_out: usize,
}

impl<'a> Chronos2NpuModel<'a> {
    /// `pub(crate)`, not just used internally, so
    /// `crates/cli/tests/npu_model_parity.rs` (see the struct doc) can construct
    /// one without the private fields being visible at its call site.
    pub(crate) fn new(model: &'a chronos2::model::Chronos2, path: &'a str, s: usize, n_out: usize) -> Chronos2NpuModel<'a> {
        Chronos2NpuModel { model, path, s, n_out }
    }
}

impl npu::NpuModel for Chronos2NpuModel<'_> {
    fn build(&self, g: &mut onnx::GraphBuilder) -> Result<(), String> {
        let reader = checkpoint::weightio::WeightReader::open(self.path).map_err(|e| format!("open {}: {e}", self.path))?;
        let cfg = chronos2::Chronos2Config::from_hf(&reader.config())?;
        npu::chronos2_topology::build_chronos2_graph_quant(&cfg, &reader, self.s, self.n_out, g, npu::qwen_topology::Quant::F32);
        Ok(())
    }
    fn cache_key(&self) -> String {
        format!("chronos2-s{}-n{}", self.s, self.n_out)
    }
    /// `inputs` are named exactly like [`build`](npu::NpuModel::build)'s graph
    /// inputs (`emb`, `kmask`); the reference is `Chronos2::core_forward` with
    /// this model's fixed `n_out` - the same forward the graph was exported from.
    fn parity_ref(&self, inputs: &[(&str, Vec<f32>)]) -> Option<Vec<Vec<f32>>> {
        let emb = inputs.iter().find(|(n, _)| *n == "emb")?.1.as_slice();
        let kmask = inputs.iter().find(|(n, _)| *n == "kmask")?.1.as_slice();
        Some(vec![self.model.core_forward(emb, kmask, self.n_out)])
    }
}

/// chronos2 on the NPU: the transformer core runs on OpenVINO via the generic
/// `npu::NpuModel` seam (`Chronos2NpuModel`); the host does scaler/patch/embed +
/// head/denorm. Compiled graphs are cached per `(context_len, n_out)` so repeated
/// same-shape requests skip recompilation.
struct Chronos2NpuInstance {
    model: chronos2::model::Chronos2,
    path: String,
    cfg: NpuConfig,
    cache: RefCell<HashMap<(usize, usize), NpuGraph>>,
    device: RefCell<String>,
}

impl Instance for Chronos2NpuInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (context, _shape) = decode_f32(inv, "context")?;
        let horizon = horizon_of(inv, 64);
        let d = self.model.config().d_model;
        let (model, path, cfg, cache, devcell) = (&self.model, self.path.as_str(), &self.cfg, &self.cache, &self.device);
        let q = guard_npu(|| {
            model.forecast_quantiles_with_core(&context, horizon, |emb, mask, n_out| {
                let s = emb.len() / d;
                let mut c = cache.borrow_mut();
                let graph = c.entry((s, n_out)).or_insert_with(|| {
                    let m = Chronos2NpuModel::new(model, path, s, n_out);
                    <Chronos2NpuModel as npu::NpuModel>::compile(&m, cfg).expect("chronos2 NPU compile")
                });
                *devcell.borrow_mut() = graph.device().to_string();
                let (si, di) = (s as i64, d as i64);
                let out = graph
                    .run(&[("emb", Feed::F32(emb, vec![1, si, di])), ("kmask", Feed::F32(mask, vec![1, 1, 1, si]))])
                    .expect("chronos2 NPU infer");
                out.into_iter().next().map(|(_, _, data)| data).expect("chronos2 NPU: no output")
            })
        })?;
        let dev = self.device.borrow().clone();
        Ok(chronos2_outcome(q, horizon, &dev))
    }
}

fn chronos2_outcome(q: Vec<f32>, horizon: usize, device: &str) -> Outcome {
    let levels = chronos2::QUANTILES;
    Outcome::new()
        .set("model", json!(CHRONOS2_MODEL))
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
    /// Explicit `.safetensors` path (the `brain perf` target and non-env callers).
    pub fn new(path: &str) -> FincastResident {
        FincastResident { path: path.to_string() }
    }
    fn spec() -> ActionSpec {
        base_forecast_spec("financial forecast; forecast blob is [horizon, 1+levels] (col 0 mean)")
            .param(ParamSpec::new("freq", ParamType::Int, "frequency bucket: 0 daily, 1 weekly, 2 monthly").default(json!(0)))
    }
}

/// The static (weights-free) FinCast manifest — see [`chronos2_manifest`].
pub(crate) fn fincast_manifest() -> Manifest {
    Manifest::new(FINCAST_MODEL, "financial time-series forecasting (FinCast); mean + 9 quantiles", vec![FincastResident::spec()])
}

impl ResidentModel for FincastResident {
    fn manifest(&self) -> Manifest {
        fincast_manifest()
    }
    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        let freq = inv.get_i64("freq").unwrap_or(0);
        InstanceKey::new(FINCAST_MODEL, format!("h{}f{freq}", horizon_of(inv, 64)))
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // NPU-eligible: the ~1 B-param core is exported with an external-data
        // sidecar and compiled via FincastSession::load_path (the in-memory buffer
        // path would exceed protobuf's 2 GB limit). ~1.5 GB is a generous NPU
        // footprint bound for the compiled fp16 blob.
        let r = file_ram(&self.path);
        MemCost::new(r, r).with_npu(1536 << 20)
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

/// The FinCast transformer core as a generic [`npu::NpuModel`]: the exact same
/// ONNX topology `npu::fincast_export::export_external` builds
/// (`npu::fincast_topology::build_fincast_graph_quant`), reached through the
/// trait seam. `parity_ref` is `Fincast::core_forward_amask`, which takes the
/// explicit `[s,s]` additive mask the graph itself consumes (unlike
/// `core_forward`, which derives it from a padding mask) - the exact function
/// the ONNX graph was built to reproduce.
///
/// `pub(crate)` for the same reason as [`Chronos2NpuModel`] (see its doc):
/// `crates/cli/tests/npu_model_parity.rs` reaches it via `#[path]` inclusion.
pub(crate) struct FincastNpuModel<'a> {
    model: &'a fincast::model::Fincast,
    path: &'a str,
    s: usize,
}

impl<'a> FincastNpuModel<'a> {
    /// `pub(crate)` for the same reason as [`Chronos2NpuModel::new`].
    pub(crate) fn new(model: &'a fincast::model::Fincast, path: &'a str, s: usize) -> FincastNpuModel<'a> {
        FincastNpuModel { model, path, s }
    }
}

impl npu::NpuModel for FincastNpuModel<'_> {
    fn build(&self, g: &mut onnx::GraphBuilder) -> Result<(), String> {
        let reader = checkpoint::weightio::WeightReader::open(self.path).map_err(|e| format!("open {}: {e}", self.path))?;
        let cfg = fincast::FincastConfig::from_json(&reader.config())?;
        npu::fincast_topology::build_fincast_graph_quant(&cfg, &reader, self.s, g, npu::qwen_topology::Quant::F32);
        Ok(())
    }
    fn cache_key(&self) -> String {
        format!("fincast-s{}", self.s)
    }
    fn parity_ref(&self, inputs: &[(&str, Vec<f32>)]) -> Option<Vec<Vec<f32>>> {
        let emb = inputs.iter().find(|(n, _)| *n == "emb")?.1.as_slice();
        let amask = inputs.iter().find(|(n, _)| *n == "amask")?.1.as_slice();
        Some(vec![self.model.core_forward_amask(emb, amask)])
    }
    /// FinCast's ~1 B-param core's single-protobuf ONNX exceeds protobuf's 2 GB
    /// read-from-buffer limit, so - unlike the trait's default (`onnx_bytes` +
    /// `NpuGraph::compile_bytes`) - this writes an external-data sidecar
    /// (`GraphBuilder::finish_external`) and compiles from the file, then drops
    /// the sidecar; the compiled graph owns the weights (mirrors the
    /// pre-migration `fincast_export::export_external` +
    /// `FincastSession::load_path` path).
    fn compile(&self, cfg: &npu::openvino::NpuConfig) -> Result<npu::openvino::NpuGraph, String> {
        let dir = std::env::temp_dir().join(format!("brain-fincast-npumodel-{}-{}", std::process::id(), self.cache_key()));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let onnx_path = dir.join("model.onnx");
        let op = onnx_path.to_str().ok_or("non-utf8 temp path")?;
        let mut g = onnx::GraphBuilder::new(&self.cache_key());
        self.build(&mut g)?;
        g.finish_external(op, 1 << 20).map_err(|e| e.to_string())?;
        let graph = npu::openvino::NpuGraph::compile_path(&onnx_path, cfg).map_err(|e| e.to_string());
        std::fs::remove_dir_all(&dir).ok();
        graph
    }
}

/// fincast on the NPU: the transformer core runs on OpenVINO via the generic
/// `npu::NpuModel` seam (`FincastNpuModel`). Compiled graphs are cached per
/// context length `s` so repeated same-shape requests skip recompilation.
struct FincastNpuInstance {
    model: fincast::model::Fincast,
    path: String,
    cfg: NpuConfig,
    cache: RefCell<HashMap<usize, NpuGraph>>,
    device: RefCell<String>,
}

impl Instance for FincastNpuInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (context, _shape) = decode_f32(inv, "context")?;
        let horizon = horizon_of(inv, 64);
        let freq = inv.get_i64("freq").unwrap_or(0).max(0) as usize;
        let (model, path, cfg, cache, devcell) = (&self.model, self.path.as_str(), &self.cfg, &self.cache, &self.device);
        let out = guard_npu(|| {
            model.forecast_full_with_core(&context, freq, horizon, |emb, amask| {
                let s = (amask.len() as f64).sqrt() as usize;
                let mut c = cache.borrow_mut();
                let graph = c.entry(s).or_insert_with(|| {
                    let m = FincastNpuModel::new(model, path, s);
                    <FincastNpuModel as npu::NpuModel>::compile(&m, cfg).expect("fincast NPU compile")
                });
                *devcell.borrow_mut() = graph.device().to_string();
                let (si, di) = (s as i64, (emb.len() / s) as i64);
                let out = graph
                    .run(&[("emb", Feed::F32(emb, vec![1, si, di])), ("amask", Feed::F32(amask, vec![1, 1, si, si]))])
                    .expect("fincast NPU infer");
                out.into_iter().next().map(|(_, _, data)| data).expect("fincast NPU: no output")
            })
        })?;
        let dev = self.device.borrow().clone();
        Ok(fincast_outcome(&self.model, out, horizon, &dev))
    }
}

fn fincast_outcome(model: &fincast::model::Fincast, out: Vec<f32>, horizon: usize, device: &str) -> Outcome {
    let no = model.config().num_outputs();
    Outcome::new()
        .set("model", json!(FINCAST_MODEL))
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
    /// Explicit tokenizer + decoder checkpoint dirs (the `brain perf` target and
    /// any caller that isn't env-driven).
    pub fn new(tokenizer: &str, decoder: &str) -> KronosResident {
        KronosResident { tokenizer: tokenizer.to_string(), decoder: decoder.to_string() }
    }
    fn spec() -> ActionSpec {
        base_forecast_spec("OHLCV forecast; context is [T, feat] bars, forecast is [horizon, feat] samples")
            .param(ParamSpec::new("temperature", ParamType::Float, "sampling temperature (0 or argmax=true => deterministic)").default(json!(1.0)))
            .param(ParamSpec::new("argmax", ParamType::Bool, "deterministic argmax decode").default(json!(true)))
            .param(ParamSpec::new("seed", ParamType::Int, "RNG seed when sampling").default(json!(0)))
            .param(ParamSpec::new("samples", ParamType::Int, "sampled paths sharing one prefill (out [N,horizon,feat])").default(json!(1)))
            .param(ParamSpec::new("checkpoint", ParamType::Str,
                "decoder checkpoint path override (.safetensors file or HF dir); \
                 empty = the boot decoder. Instances are keyed on (path, mtime, \
                 size), so per-request checkpoints stay warm side by side and an \
                 overwritten file hot-reloads — checkpoint selection is request \
                 state, not server state.").default(json!("")))
            .input(BlobSpec::new("ctx_stamp", Media::Bytes, "optional context calendar stamps [T,5] u32-LE"))
            .input(BlobSpec::new("fut_stamp", Media::Bytes, "optional future calendar stamps [horizon,5] u32-LE"))
    }

    /// The decoder a request selects: the `checkpoint` param when non-empty,
    /// else the boot decoder from the env.
    fn decoder_for(&self, inv: &Invocation) -> String {
        inv.get_str("checkpoint").filter(|p| !p.is_empty()).unwrap_or_else(|| self.decoder.clone())
    }
}

/// `"path|mtime|size"` — the identity the residency cache keys a decoder
/// instance on. mtime+size in the key means overwriting a checkpoint file
/// (a new fine-tune) transparently activates a fresh instance; the stale one
/// ages out via normal eviction. `|` cannot appear in the sortable fields, so
/// the path (which may contain `|` only pathologically) parses back out with
/// rsplitn.
fn decoder_key(path: &str) -> String {
    match std::fs::metadata(path) {
        Ok(m) => {
            let mtime = m.modified().ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs()).unwrap_or(0);
            format!("{path}|{mtime}|{}", m.len())
        }
        Err(_) => format!("{path}|missing|0"),
    }
}

fn decoder_from_key(config: &str) -> &str {
    // strip the two identity fields appended by decoder_key
    config.rsplitn(3, '|').nth(2).unwrap_or(config)
}

/// The static (weights-free) Kronos manifest — see [`chronos2_manifest`].
pub(crate) fn kronos_manifest() -> Manifest {
    Manifest::new(KRONOS_MODEL, "autoregressive OHLCV forecasting (Kronos)", vec![KronosResident::spec()])
}

impl ResidentModel for KronosResident {
    fn manifest(&self) -> Manifest {
        kronos_manifest()
    }
    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        // One hot instance per decoder checkpoint (any horizon); see decoder_key.
        InstanceKey::new(KRONOS_MODEL, decoder_key(&self.decoder_for(inv)))
    }
    fn estimate(&self, key: &InstanceKey) -> MemCost {
        // tokenizer + the requested decoder resident in RAM (host embed/sample
        // on gpu_core). NPU-eligible: activate(Npu) compiles the two decoder
        // graphs (s1 + dep-s2) and runs the rollout on the accelerator.
        let ram = path_ram(&self.tokenizer) + path_ram(decoder_from_key(&key.config));
        MemCost::new(0, ram).with_npu(512 << 20)
    }
    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if let Device::Gpu(i) = device {
            std::env::set_var("BRAIN_GPU_INDEX", i.to_string());
        }
        let decoder = decoder_from_key(&key.config).to_string();
        if !std::path::Path::new(&decoder).exists() {
            return Err(format!("kronos checkpoint not found: {decoder}"));
        }
        let model = kronos::import::load_model(&self.tokenizer, &decoder)?;
        if let Device::Npu(_) = device {
            return Ok(Box::new(KronosNpuInstance {
                model,
                dec_dir: decoder,
                cfg: npu_cfg(),
                cached: RefCell::new(HashMap::new()),
                device: RefCell::new("npu".into()),
            }));
        }
        Ok(Box::new(KronosInstance { model }))
    }
}

/// Resident bytes for a checkpoint path: a `.safetensors` file's own size, or the
/// summed size of an HF checkpoint dir. (+30% working overhead.)
fn path_ram(path: &str) -> u64 {
    let meta = std::fs::metadata(path);
    let raw = match meta {
        Ok(m) if m.is_file() => m.len(),
        _ => std::fs::read_dir(path)
            .map(|rd| rd.flatten().filter_map(|e| e.metadata().ok().map(|m| m.len())).sum::<u64>())
            .unwrap_or(0),
    };
    raw.saturating_mul(13) / 10
}

struct KronosInstance {
    model: kronos::generate::KronosModel,
}

/// Decode the kronos context: either full OHLCV bars `[T, feat]`, or a univariate
/// close series `[T]` / `[T,1]` expanded to bars (o=h=l=c=close; the remaining
/// features — volume, amount — held at 1.0), so a caller can forecast from a
/// single series exactly like chronos2/fincast. Returns `(bars, T)`.
fn kronos_bars(inv: &Invocation, feat: usize) -> Result<(Vec<f32>, usize), String> {
    let (raw, shape) = decode_f32(inv, "context")?;
    if shape.len() == 2 && shape[1] == feat {
        Ok((raw, shape[0]))
    } else if shape.len() <= 1 || (shape.len() == 2 && shape[1] == 1) {
        let t = raw.len();
        let mut bars = Vec::with_capacity(t * feat);
        for &c in &raw {
            let mut row = vec![c; 4.min(feat)]; // o,h,l,c
            row.resize(feat, 1.0);
            bars.extend(row);
        }
        Ok((bars, t))
    } else {
        Err(format!("kronos: context must be [T,{feat}] OHLCV bars or a univariate [T] series; got {shape:?}"))
    }
}

/// Calendar stamps `[T,5]` / `[horizon,5]` u32 from the client, or zeros
/// (calendar-agnostic) when absent.
fn kronos_stamps(inv: &Invocation, t: usize, horizon: usize) -> Result<(Vec<u32>, Vec<u32>), String> {
    let ctx_stamp = decode_u32_opt(inv, "ctx_stamp").unwrap_or_else(|| vec![0u32; t * 5]);
    let fut_stamp = decode_u32_opt(inv, "fut_stamp").unwrap_or_else(|| vec![0u32; horizon * 5]);
    if ctx_stamp.len() != t * 5 {
        return Err(format!("kronos: ctx_stamp must be [{t},5] u32, got {}", ctx_stamp.len()));
    }
    if fut_stamp.len() != horizon * 5 {
        return Err(format!("kronos: fut_stamp must be [{horizon},5] u32, got {}", fut_stamp.len()));
    }
    Ok((ctx_stamp, fut_stamp))
}

fn kronos_opts(inv: &Invocation) -> kronos::generate::GenOpts {
    kronos::generate::GenOpts {
        temperature: inv.get_f64("temperature").unwrap_or(1.0) as f32,
        argmax: inv.get_bool("argmax").unwrap_or(true),
        seed: inv.get_i64("seed").unwrap_or(0) as u64,
        ..Default::default()
    }
}

fn kronos_outcome(out: Vec<f32>, horizon: usize, feat: usize, device: &str) -> Outcome {
    Outcome::new()
        .set("model", json!(KRONOS_MODEL))
        .set("horizon", json!(horizon))
        .set("device", json!(device))
        .blob("forecast", encode_forecast(&out, vec![horizon, feat], "samples", &[]))
}

impl Instance for KronosInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let feat = self.model.feat();
        let (bars, t) = kronos_bars(inv, feat)?;
        let horizon = horizon_of(inv, 64);
        let (ctx_stamp, fut_stamp) = kronos_stamps(inv, t, horizon)?;
        // Fast path: the KV-cached rollout (dep-KV cache + AVX matvec) — identical
        // result to `forecast` (cosine >0.999, tests/kvcache_parity) but O(T²)
        // prefill + O(T)/step instead of O(T²)/step. `--samples N` shares one
        // prefill across N sampled paths (returned as [N, horizon, feat]).
        let opts = kronos_opts(inv);
        let samples = inv.get_i64("samples").unwrap_or(1).max(1) as usize;
        if samples > 1 {
            let outs = self.model.forecast_cached_samples(&bars, &ctx_stamp, &fut_stamp, horizon, samples, &opts);
            let flat: Vec<f32> = outs.into_iter().flatten().collect();
            return Ok(Outcome::new()
                .set("model", json!(KRONOS_MODEL))
                .set("horizon", json!(horizon))
                .set("samples", json!(samples))
                .set("device", json!("gpu_core"))
                .blob("forecast", encode_forecast(&flat, vec![samples, horizon, feat], "samples", &[])));
        }
        let out = self.model.forecast_cached(&bars, &ctx_stamp, &fut_stamp, horizon, &opts); // [horizon, feat]
        Ok(kronos_outcome(out, horizon, feat, "gpu_core"))
    }
}

/// Kronos on the NPU: both decoder graphs (s1 + dep-s2) compiled on OpenVINO and
/// driven by the model's `forecast_with_cores` seam — the host does normalize /
/// tokenize / embed / sample / denormalize on gpu_core, the two transformer cores
/// run on the accelerator. Sessions are cached per context-length `T` (the graph
/// is fixed-shape, so the rollout uses a fixed sliding window of `T`).
struct KronosNpuInstance {
    model: kronos::generate::KronosModel,
    dec_dir: String,
    cfg: NpuConfig,
    /// One compiled KV-cache backend per (context length `t`, cache capacity `cap`).
    cached: RefCell<HashMap<(usize, usize), KronosCachedNpu>>,
    device: RefCell<String>,
}

impl KronosNpuInstance {
    /// Compile (once, cached) the four KV-cache graphs for `(t, cap)`.
    fn ensure(&self, t: usize, cap: usize) -> Result<(), String> {
        if self.cached.borrow().contains_key(&(t, cap)) {
            return Ok(());
        }
        let q = npu::qwen_topology::Quant::F32;
        let (s1p, s1d, depp, depd) = npu::kronos_export::export_cached_onnx(&self.dec_dir, t, cap, q)?;
        let mk = |b: &[u8]| NpuGraph::compile_bytes(b, &self.cfg).map_err(|e| e.to_string());
        let core = KronosCachedNpu::new(self.model.decoder_config(), cap, mk(&s1p)?, mk(&s1d)?, mk(&depp)?, mk(&depd)?);
        *self.device.borrow_mut() = core.s1_decode.device().to_string();
        self.cached.borrow_mut().insert((t, cap), core);
        Ok(())
    }
}

impl Instance for KronosNpuInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let feat = self.model.feat();
        let (bars, t) = kronos_bars(inv, feat)?;
        let horizon = horizon_of(inv, 64);
        let (ctx_stamp, fut_stamp) = kronos_stamps(inv, t, horizon)?;
        let opts = kronos_opts(inv);
        let samples = inv.get_i64("samples").unwrap_or(1).max(1) as usize;
        let cap = t + horizon;
        self.ensure(t, cap)?; // compile errors propagate cleanly (no panic)
        let model = &self.model;
        let cached = &self.cached;
        let dev = self.device.borrow().clone();
        // KV-cached rollout: one prefill fills the cache, then O(cap)/step decode
        // (the same optimization as the host `forecast_cached`), not the old
        // O(T²)/step full-window re-run. `--samples N` shares one prefill.
        if samples > 1 {
            let outs = guard_npu(|| {
                let mut map = cached.borrow_mut();
                let core = map.get_mut(&(t, cap)).expect("kronos cached graphs");
                model.forecast_cached_samples_with_cores(&bars, &ctx_stamp, &fut_stamp, horizon, samples, &opts, core)
            })?;
            let flat: Vec<f32> = outs.into_iter().flatten().collect();
            return Ok(Outcome::new()
                .set("model", json!(KRONOS_MODEL))
                .set("horizon", json!(horizon))
                .set("samples", json!(samples))
                .set("device", json!(dev))
                .blob("forecast", encode_forecast(&flat, vec![samples, horizon, feat], "samples", &[])));
        }
        let out = guard_npu(|| {
            let mut map = cached.borrow_mut();
            let core = map.get_mut(&(t, cap)).expect("kronos cached graphs");
            model.forecast_cached_with_cores(&bars, &ctx_stamp, &fut_stamp, horizon, &opts, core)
        })?;
        Ok(kronos_outcome(out, horizon, feat, &dev))
    }
}

/// The NPU KV-cache backend for one `(t, cap)`: the four compiled graphs plus the
/// host-side K/V cache buffers. Implements [`kronos::generate::CachedCores`] — the
/// driver stays in kronos, this owns the graph runs + cache. Buffer slots beyond
/// the written prefix are masked out each step, so `prefill` fully re-initialises
/// the logical state (positions/valid counts) and the backend is reused across
/// forecasts of the same shape.
struct KronosCachedNpu {
    s1_prefill: NpuGraph,
    s1_decode: NpuGraph,
    dep_prefill: NpuGraph,
    dep_decode: NpuGraph,
    d: usize,
    nl: usize,
    heads: usize,
    hd: usize,
    dep_heads: usize,
    dep_hd: usize,
    cap: usize,
    s1v: usize,
    // per-rollout state (reset by `prefill`)
    pk: Vec<Vec<f32>>, // [nl][heads*cap*hd] RoPE'd keys
    pv: Vec<Vec<f32>>,
    dk: Vec<f32>, // [dep_heads*cap*dep_hd] RoPE'd dep keys
    dv: Vec<f32>,
    ctx_last: Vec<f32>, // [d] most recent s1 context row (dep residual + self)
    s1_pos: usize,      // next s1 absolute position to write
    dep_valid: usize,   // number of dep positions in the cache (past extent)
    snap: Option<KronosCacheSnap>, // post-prefill state, for shared-prefill sampling
}

/// A post-prefill cache snapshot (the buffers the decode loop mutates), so a
/// samples=N forecast forks from one prefill (mirrors `Cache::clone()`).
#[derive(Clone)]
struct KronosCacheSnap {
    pk: Vec<Vec<f32>>,
    pv: Vec<Vec<f32>>,
    dk: Vec<f32>,
    dv: Vec<f32>,
    ctx_last: Vec<f32>,
    s1_pos: usize,
    dep_valid: usize,
}

/// Look up a named output tensor from an [`NpuGraph::run`] result.
fn named<'a>(out: &'a [(String, Vec<usize>, Vec<f32>)], name: &str) -> &'a [f32] {
    &out.iter().find(|(n, _, _)| n == name).unwrap_or_else(|| panic!("missing NPU output {name}")).2
}

/// Half-width RoPE cos/sin tables for one absolute position (θ=10000, NeoX split).
fn rope_tables(pos: usize, half: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for j in 0..half {
        let ang = pos as f32 * 10000f32.powf(-(2.0 * j as f32) / hd as f32);
        cos[j] = ang.cos();
        sin[j] = ang.sin();
    }
    (cos, sin)
}

impl KronosCachedNpu {
    fn new(cfg: &kronos::config::KronosConfig, cap: usize, s1_prefill: NpuGraph, s1_decode: NpuGraph, dep_prefill: NpuGraph, dep_decode: NpuGraph) -> KronosCachedNpu {
        let d = cfg.d_model;
        let (heads, dep_heads) = (cfg.n_heads, cfg.dep_n_heads);
        let (hd, dep_hd) = (d / heads, d / dep_heads);
        let nl = cfg.n_layers;
        KronosCachedNpu {
            s1_prefill,
            s1_decode,
            dep_prefill,
            dep_decode,
            d,
            nl,
            heads,
            hd,
            dep_heads,
            dep_hd,
            cap,
            s1v: cfg.s1_vocab(),
            pk: vec![vec![0.0; heads * cap * hd]; nl],
            pv: vec![vec![0.0; heads * cap * hd]; nl],
            dk: vec![0.0; dep_heads * cap * dep_hd],
            dv: vec![0.0; dep_heads * cap * dep_hd],
            ctx_last: vec![0.0; d],
            s1_pos: 0,
            dep_valid: 0,
            snap: None,
        }
    }
}

impl kronos::generate::CachedCores for KronosCachedNpu {
    fn prefill(&mut self, x_ctx: &[f32], t: usize) -> Vec<f32> {
        let (d, nl, heads, hd, cap) = (self.d, self.nl, self.heads, self.hd, self.cap);
        // s1 prefill: x[1,t,d] → ctx[1,t,d], s1_logits, k_l/v_l[heads,t,hd].
        let out = self.s1_prefill.run(&[("x", Feed::F32(x_ctx, vec![1, t as i64, d as i64]))]).expect("kronos s1 prefill");
        let ctx = named(&out, "ctx").to_vec();
        let s1_logits = named(&out, "s1_logits").to_vec();
        for l in 0..nl {
            let kl = named(&out, &format!("k_{l}"));
            let vl = named(&out, &format!("v_{l}"));
            for h in 0..heads {
                for p in 0..t {
                    for j in 0..hd {
                        self.pk[l][(h * cap + p) * hd + j] = kl[(h * t + p) * hd + j];
                        self.pv[l][(h * cap + p) * hd + j] = vl[(h * t + p) * hd + j];
                    }
                }
            }
        }
        // dep prefill over ctx[0..t-1] fills dep positions 0..t-2 (the last, t-1,
        // is self-projected by the first dep_step). t==1 → no dep prefill.
        let (dep_heads, dep_hd) = (self.dep_heads, self.dep_hd);
        if t >= 2 {
            let tp = t - 1;
            let dout = self.dep_prefill.run(&[("ctx", Feed::F32(&ctx[..tp * d], vec![1, tp as i64, d as i64]))]).expect("kronos dep prefill");
            let dk = named(&dout, "dep_k").to_vec();
            let dv = named(&dout, "dep_v").to_vec();
            for h in 0..dep_heads {
                for p in 0..tp {
                    for j in 0..dep_hd {
                        self.dk[(h * cap + p) * dep_hd + j] = dk[(h * tp + p) * dep_hd + j];
                        self.dv[(h * cap + p) * dep_hd + j] = dv[(h * tp + p) * dep_hd + j];
                    }
                }
            }
            self.dep_valid = tp;
        } else {
            self.dep_valid = 0;
        }
        self.ctx_last = ctx[(t - 1) * d..t * d].to_vec();
        self.s1_pos = t;
        let s1v = self.s1v;
        s1_logits[(t - 1) * s1v..t * s1v].to_vec()
    }

    fn dep_step(&mut self, sib: &[f32]) -> Vec<f32> {
        let (d, cap, dep_heads, dep_hd) = (self.d, self.cap, self.dep_heads, self.dep_hd);
        let mask: Vec<f32> = (0..cap).map(|j| if j < self.dep_valid { 0.0 } else { -1e9 }).collect();
        let ctx_last = std::mem::take(&mut self.ctx_last);
        let out = {
            let feeds: Vec<(&str, Feed)> = vec![
                ("sib", Feed::F32(sib, vec![1, 1, d as i64])),
                ("ctx_last", Feed::F32(&ctx_last, vec![1, 1, d as i64])),
                ("dep_mask", Feed::F32(&mask, vec![1, 1, 1, cap as i64])),
                ("past_dep_k", Feed::F32(&self.dk, vec![1, dep_heads as i64, cap as i64, dep_hd as i64])),
                ("past_dep_v", Feed::F32(&self.dv, vec![1, dep_heads as i64, cap as i64, dep_hd as i64])),
            ];
            self.dep_decode.run(&feeds).expect("kronos dep decode")
        };
        self.ctx_last = ctx_last;
        let p = self.dep_valid;
        let nk = named(&out, "new_dep_k").to_vec();
        let nv = named(&out, "new_dep_v").to_vec();
        for h in 0..dep_heads {
            for j in 0..dep_hd {
                self.dk[(h * cap + p) * dep_hd + j] = nk[h * dep_hd + j];
                self.dv[(h * cap + p) * dep_hd + j] = nv[h * dep_hd + j];
            }
        }
        self.dep_valid += 1;
        named(&out, "s2_logits").to_vec()
    }

    fn s1_step(&mut self, x: &[f32]) -> Vec<f32> {
        let (d, nl, heads, hd, cap) = (self.d, self.nl, self.heads, self.hd, self.cap);
        let half = hd / 2;
        let pos = self.s1_pos;
        let (cos, sin) = rope_tables(pos, half, hd);
        let mask: Vec<f32> = (0..cap).map(|j| if j < pos { 0.0 } else { -1e9 }).collect();
        let keys: Vec<(String, String)> = (0..nl).map(|l| (format!("past_k_{l}"), format!("past_v_{l}"))).collect();
        let out = {
            let mut feeds: Vec<(&str, Feed)> = vec![
                ("x", Feed::F32(x, vec![1, 1, d as i64])),
                ("rope_cos", Feed::F32(&cos, vec![1, 1, 1, half as i64])),
                ("rope_sin", Feed::F32(&sin, vec![1, 1, 1, half as i64])),
                ("past_mask", Feed::F32(&mask, vec![1, 1, 1, cap as i64])),
            ];
            for (l, (kname, vname)) in keys.iter().enumerate().take(nl) {
                feeds.push((kname.as_str(), Feed::F32(&self.pk[l], vec![1, heads as i64, cap as i64, hd as i64])));
                feeds.push((vname.as_str(), Feed::F32(&self.pv[l], vec![1, heads as i64, cap as i64, hd as i64])));
            }
            self.s1_decode.run(&feeds).expect("kronos s1 decode")
        };
        for l in 0..nl {
            let nk = named(&out, &format!("new_k_{l}")).to_vec();
            let nv = named(&out, &format!("new_v_{l}")).to_vec();
            for h in 0..heads {
                for j in 0..hd {
                    self.pk[l][(h * cap + pos) * hd + j] = nk[h * hd + j];
                    self.pv[l][(h * cap + pos) * hd + j] = nv[h * hd + j];
                }
            }
        }
        self.ctx_last = named(&out, "ctx").to_vec();
        self.s1_pos += 1;
        named(&out, "s1_logits").to_vec()
    }

    fn snapshot(&mut self) {
        self.snap = Some(KronosCacheSnap {
            pk: self.pk.clone(),
            pv: self.pv.clone(),
            dk: self.dk.clone(),
            dv: self.dv.clone(),
            ctx_last: self.ctx_last.clone(),
            s1_pos: self.s1_pos,
            dep_valid: self.dep_valid,
        });
    }

    fn restore(&mut self) {
        let s = self.snap.as_ref().expect("restore before snapshot");
        self.pk.clone_from(&s.pk);
        self.pv.clone_from(&s.pv);
        self.dk.clone_from(&s.dk);
        self.dv.clone_from(&s.dv);
        self.ctx_last.clone_from(&s.ctx_last);
        self.s1_pos = s.s1_pos;
        self.dep_valid = s.dep_valid;
    }
}

// ================================= timesfm3 =================================

/// TimesFM-3 behind the scheduler. `BRAIN_TIMESFM3` = the brain-format
/// weights. Unlike the other three foundation models here, this wire DOES
/// carry the model's native multivariate/covariate capability: `context` is
/// `[T]` (one target, byte-identical to the original single-series contract)
/// or `[num_target, T]`; optional `past_covariates` (`[T]`/`[P,T]`) and
/// `known_future` (`[T+horizon]`/`[K,T+horizon]`) add covariate variates; an
/// optional `observed` mask (same shape as `context`) marks missing steps,
/// honored end to end since `Timesfm3Forecaster::forecast` started reading
/// `forecast::Variate::observed`. `forecast` is `[horizon, 9]` for one target
/// (`kind:"quantiles_hq"`, unchanged) or `[num_target, horizon, 9]` for more
/// than one (`kind:"quantiles_thq"`, plus a `names` meta array - position
/// alone does not say which target is which). CPU/GPU only, like kronos
/// above - no NPU export for this model yet.
pub struct Timesfm3Resident {
    path: String,
}

impl Timesfm3Resident {
    pub fn from_env() -> Option<Timesfm3Resident> {
        std::env::var("BRAIN_TIMESFM3").ok().filter(|p| !p.is_empty()).map(|path| Timesfm3Resident { path })
    }
    /// Explicit `.safetensors` path (the `brain perf` target and non-env callers).
    pub fn new(path: &str) -> Timesfm3Resident {
        Timesfm3Resident { path: path.to_string() }
    }
    fn spec() -> ActionSpec {
        base_forecast_spec(
            "multivariate forecast (TimesFM-3, 9 native quantiles); context is [T] or [num_target,T], \
             optional past_covariates/known_future/observed add covariates/masking",
        )
        .input(BlobSpec::new("past_covariates", Media::Bytes, "past-only covariate series as raw f32-LE; meta {shape: [T] or [P,T]}"))
        .input(BlobSpec::new("known_future", Media::Bytes, "known-future covariate series as raw f32-LE; meta {shape: [T+horizon] or [K,T+horizon]}"))
        .input(BlobSpec::new("observed", Media::Bytes, "1.0/0.0 per-step observed mask, same shape as context, as raw f32-LE"))
    }
}

/// The static (weights-free) TimesFM-3 manifest - see [`chronos2_manifest`].
pub(crate) fn timesfm3_manifest() -> Manifest {
    Manifest::new(TIMESFM3_MODEL, "time-series forecasting (TimesFM-3); 9 native quantiles, native multivariate + covariates over this wire", vec![Timesfm3Resident::spec()])
}

impl ResidentModel for Timesfm3Resident {
    fn manifest(&self) -> Manifest {
        timesfm3_manifest()
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        // NOT keyed on horizon: the key controls which weights are resident,
        // and horizon does not change them - keying on it used to force N
        // redundant 330M-parameter copies of the SAME weights and fragment
        // the scheduler's same-key batching so two requests that only differ
        // in horizon could never share a `run_batch` call. `run_batch` itself
        // groups by horizon internally, where a shape difference actually is
        // one.
        InstanceKey::new(TIMESFM3_MODEL, String::new())
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        let r = file_ram(&self.path);
        MemCost::new(r, r)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if let Device::Gpu(i) = device {
            std::env::set_var("BRAIN_GPU_INDEX", i.to_string());
        }
        let forecaster = timesfm3::Timesfm3Forecaster::load(&self.path)?;
        Ok(Box::new(Timesfm3Instance { forecaster }))
    }
}

/// `[T]` -> `(1, T)`, `[N, T]` -> `(N, T)`; anything else is a wire error
/// naming the offending field.
fn parse_series_shape(shape: &[usize], field: &str) -> Result<(usize, usize), String> {
    match shape {
        [t] => Ok((1, *t)),
        [n, t] => Ok((*n, *t)),
        _ => Err(format!("timesfm3: {field} shape {shape:?} must be [T] or [N, T]")),
    }
}

/// Decode one invocation's forecasting inputs into an `Item`, native
/// multivariate/covariate shapes and all - shared by `run` and `run_batch` so
/// a served request reaches the same capability either way. `context` is
/// `[T]` (one target) or `[num_target, T]`; `past_covariates`/`known_future`
/// are optional and follow the same `[T]`/`[N,T]` convention (`known_future`
/// carries `horizon` extra steps); `observed`, if present, must share
/// `context`'s exact shape and becomes each target's `Variate::observed`.
fn item_from_invocation(inv: &Invocation, item_id: String, horizon: usize) -> Result<forecast::Item, String> {
    let (context, shape) = decode_f32(inv, "context")?;
    let (num_target, t) = parse_series_shape(&shape, "context")?;
    let observed = decode_f32_opt(inv, "observed")?;
    if let Some((_, oshape)) = &observed {
        if *oshape != shape {
            return Err(format!("timesfm3: observed shape {oshape:?} must match context shape {shape:?}"));
        }
    }

    let mut variates = Vec::with_capacity(num_target);
    for i in 0..num_target {
        let mut v = forecast::Variate::target(format!("t{i}"), context[i * t..(i + 1) * t].to_vec());
        if let Some((data, _)) = &observed {
            v.observed = Some(data[i * t..(i + 1) * t].to_vec());
        }
        variates.push(v);
    }
    if let Some((data, pshape)) = decode_f32_opt(inv, "past_covariates")? {
        let (np, pt) = parse_series_shape(&pshape, "past_covariates")?;
        if pt != t {
            return Err(format!("timesfm3: past_covariates length {pt} must equal context length {t}"));
        }
        for i in 0..np {
            let mut v = forecast::Variate::target(format!("p{i}"), data[i * pt..(i + 1) * pt].to_vec());
            v.role = forecast::Role::PastCovariate;
            variates.push(v);
        }
    }
    if let Some((data, fshape)) = decode_f32_opt(inv, "known_future")? {
        let (nf, ft) = parse_series_shape(&fshape, "known_future")?;
        if ft != t + horizon {
            return Err(format!("timesfm3: known_future length {ft} must equal context length {t} + horizon {horizon}"));
        }
        for i in 0..nf {
            let mut v = forecast::Variate::target(format!("f{i}"), data[i * ft..i * ft + t].to_vec());
            v.role = forecast::Role::KnownFuture;
            v.future = Some(data[i * ft + t..(i + 1) * ft].to_vec());
            variates.push(v);
        }
    }
    Ok(forecast::Item::new(item_id, variates))
}

struct Timesfm3Instance {
    forecaster: timesfm3::Timesfm3Forecaster,
}

impl Timesfm3Instance {
    /// Encode a request's target(s) into the wire `Outcome` - shared by `run`
    /// and `run_batch`'s success path so the two cannot diverge. One target
    /// is `kind:"quantiles_hq"` `[horizon, levels]`, byte-identical to the
    /// original single-series contract; more than one is
    /// `kind:"quantiles_thq"` `[num_target, horizon, levels]` plus a `names`
    /// meta array, since position alone would not say which target is which.
    fn outcome_from_targets(horizon: usize, targets: &[forecast::TargetForecast], levels: &[f32]) -> ActionResult {
        let mut data = Vec::with_capacity(targets.len() * horizon * levels.len());
        let mut names = Vec::with_capacity(targets.len());
        for tf in targets {
            let q = tf.quantiles.as_ref().ok_or_else(|| "timesfm3: model returned no quantiles".to_string())?;
            data.extend_from_slice(&q.data);
            names.push(tf.name.clone());
        }
        let blob = if targets.len() == 1 {
            encode_forecast(&data, vec![horizon, levels.len()], "quantiles_hq", levels)
        } else {
            encode_named_forecast(&data, vec![targets.len(), horizon, levels.len()], "quantiles_thq", levels, &names)
        };
        Ok(Outcome::new().set("model", json!(TIMESFM3_MODEL)).set("horizon", json!(horizon)).set("device", json!("gpu_core")).blob("forecast", blob))
    }
}

impl Instance for Timesfm3Instance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let horizon = horizon_of(inv, 64);
        let levels = self.forecaster.config().quantile_levels.clone();

        // `item_from_invocation` reads native multivariate/covariate shapes;
        // `Timesfm3Forecaster::forecast` itself left-pads any context to the
        // checkpoint's patch boundary, so the FULL series is used, no
        // truncation to a patch-aligned tail here.
        let item = item_from_invocation(inv, "series".to_string(), horizon)?;
        let panel = forecast::Panel::single("", "series", item.variates);
        let spec = forecast::ForecastSpec { horizon, quantile_levels: levels.clone(), ..forecast::ForecastSpec::default() };
        let out = forecast::ForecastModel::forecast(&self.forecaster, &panel, &spec).map_err(|e| format!("timesfm3: {}", e.message))?;
        if out.targets.is_empty() {
            return Err("timesfm3: no forecast produced".to_string());
        }
        Self::outcome_from_targets(horizon, &out.targets, &levels)
    }

    /// Group by horizon (the one dimension `Timesfm3Forecaster::forecast`
    /// needs uniform across a `Panel` - it batches everything else, including
    /// mixed context lengths, variate counts, and multi-target items,
    /// internally) and issue one forecast() per group, each item keyed by its
    /// own batch index so results scatter back positionally (an item may
    /// itself carry several targets, grouped back together by that index
    /// before encoding). A malformed invocation gets its own `Err` before
    /// ever entering a group. If a whole group's shared call fails - one bad
    /// item can fail a Panel outright, e.g. an all-missing target - that
    /// group is re-run one invocation at a time so a single bad request never
    /// fails its batch-mates, mirroring `resident_asr`'s offline_batch
    /// per-job isolation without needing forecast() itself to return partial
    /// results.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], _progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        let levels = self.forecaster.config().quantile_levels.clone();

        struct Job {
            item: forecast::Item,
            horizon: usize,
        }
        let mut jobs: Vec<Option<Job>> = Vec::with_capacity(invs.len());
        let mut results: Vec<Option<ActionResult>> = vec![None; invs.len()];
        for (i, inv) in invs.iter().enumerate() {
            let horizon = horizon_of(inv, 64);
            match item_from_invocation(inv, i.to_string(), horizon) {
                Ok(item) => jobs.push(Some(Job { item, horizon })),
                Err(e) => {
                    jobs.push(None);
                    results[i] = Some(Err(e));
                }
            }
        }

        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
        for (i, j) in jobs.iter().enumerate() {
            if let Some(j) = j {
                groups.entry(j.horizon).or_default().push(i);
            }
        }

        for (&horizon, idxs) in &groups {
            let items: Vec<forecast::Item> = idxs.iter().map(|&i| jobs[i].as_ref().unwrap().item.clone()).collect();
            let panel = forecast::Panel { freq: String::new(), start: None, items };
            let spec = forecast::ForecastSpec { horizon, quantile_levels: levels.clone(), ..forecast::ForecastSpec::default() };
            match forecast::ForecastModel::forecast(&self.forecaster, &panel, &spec) {
                Ok(out) => {
                    let mut by_item: std::collections::BTreeMap<usize, Vec<forecast::TargetForecast>> = std::collections::BTreeMap::new();
                    for tf in out.targets {
                        let i: usize = tf.item_id.parse().expect("item_id is this batch's own index, set just above");
                        by_item.entry(i).or_default().push(tf);
                    }
                    for (i, targets) in by_item {
                        results[i] = Some(Self::outcome_from_targets(horizon, &targets, &levels));
                    }
                }
                Err(_) => {
                    for &i in idxs {
                        results[i] = Some(self.run(action, &invs[i], &mut |_| {}));
                    }
                }
            }
        }

        results.into_iter().map(|r| r.expect("every invocation was either decoded into a job or given a decode error")).collect()
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
            Timesfm3Resident { path: String::new() }.manifest(),
        ];
        for m in &specs {
            let a = m.actions.iter().find(|a| a.name == "forecast").expect("has a forecast action");
            assert!(a.inputs.iter().any(|b| b.name == "context" && b.required), "{}: required context input", m.model);
            assert!(a.outputs.iter().any(|b| b.name == "forecast"), "{}: forecast output", m.model);
            assert!(a.params.iter().any(|p| p.name == "horizon"), "{}: horizon param", m.model);
        }
    }

    /// A tiny checkpoint-free `Timesfm3Instance`, deterministic synthetic
    /// weights - mirrors `crates/timesfm3/tests/forecaster.rs`'s own helper,
    /// needing neither the golden manifest nor a real checkpoint. Built on
    /// the POOLED test device (`gpu_core::testgpu::dev`): this helper is
    /// called once per test in this file, and `Timesfm3::from_weights`'s own
    /// plain `Gpu::new` creates a brand-new real device every time, which is
    /// exactly the per-test-binary sharing `testgpu` exists to avoid.
    fn synthetic_timesfm3_instance() -> Timesfm3Instance {
        let cfg = timesfm3::Timesfm3Config::tiny();
        let weights: HashMap<String, Vec<f32>> = cfg
            .param_list()
            .into_iter()
            .enumerate()
            .map(|(i, (k, s))| {
                let n: usize = s.iter().product();
                let data: Vec<f32> = (0..n).map(|j| (((i * 131 + j * 17) % 23) as f32 - 11.0) * 0.01).collect();
                (k, data)
            })
            .collect();
        let model = timesfm3::Timesfm3::from_weights_on(gpu_core::testgpu::dev(timesfm3::model::PIPELINES), cfg, &weights).unwrap();
        Timesfm3Instance { forecaster: timesfm3::Timesfm3Forecaster::new(model) }
    }

    fn timesfm3_context_invocation(seed: u32) -> Invocation {
        let data: Vec<f32> = (0..8).map(|i| ((seed * 131 + i * 17) % 23) as f32 - 11.0).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Invocation::new().set("horizon", json!(4)).blob("context", Blob::new(Media::Bytes, bytes))
    }

    #[test]
    fn timesfm3_run_batch_matches_run_called_individually() {
        let mut inst = synthetic_timesfm3_instance();
        let invs: Vec<Invocation> = (1..=3).map(timesfm3_context_invocation).collect();

        let batched = inst.run_batch("forecast", &invs, &mut |_, _| {});
        assert_eq!(batched.len(), invs.len());
        for (i, inv) in invs.iter().enumerate() {
            let single = inst.run("forecast", inv, &mut |_| {}).expect("run");
            let got = batched[i].as_ref().expect("run_batch result");
            assert_eq!(got.blobs["forecast"].bytes, single.blobs["forecast"].bytes, "invocation {i}: run_batch must match run() bit-for-bit");
        }
    }

    #[test]
    fn timesfm3_run_batch_isolates_a_malformed_invocation() {
        let mut inst = synthetic_timesfm3_instance();
        let bad = Invocation::new().set("horizon", json!(4)); // no "context" blob
        let invs = vec![timesfm3_context_invocation(1), bad, timesfm3_context_invocation(3)];

        let out = inst.run_batch("forecast", &invs, &mut |_, _| {});
        assert!(out[0].is_ok(), "invocation 0 is well-formed and must succeed");
        assert!(out[1].is_err(), "invocation 1 is missing its context blob");
        assert!(out[2].is_ok(), "invocation 2 is well-formed and must succeed despite invocation 1's failure");
    }

    fn f32_blob(data: &[f32], shape: &[usize]) -> Blob {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Blob::new(Media::Bytes, bytes).with_meta(json!({"shape": shape}))
    }

    #[test]
    fn timesfm3_single_target_output_is_unchanged_by_the_multivariate_refactor() {
        let mut inst = synthetic_timesfm3_instance();
        let out = inst.run("forecast", &timesfm3_context_invocation(1), &mut |_| {}).expect("run");
        let blob = &out.blobs["forecast"];
        assert_eq!(blob.meta["kind"], "quantiles_hq");
        assert_eq!(blob.meta["shape"], json!([4, 5])); // tiny() has 5 native quantile levels
        assert!(blob.meta.get("names").is_none(), "a single target must not grow a names array");
    }

    #[test]
    fn timesfm3_multivariate_context_produces_named_quantiles_thq() {
        let mut inst = synthetic_timesfm3_instance();
        let data: Vec<f32> = (0..16).map(|i| ((131 * i) % 23) as f32 - 11.0).collect(); // [2 targets, 8 steps]
        let inv = Invocation::new().set("horizon", json!(4)).blob("context", f32_blob(&data, &[2, 8]));
        let out = inst.run("forecast", &inv, &mut |_| {}).expect("run");
        let blob = &out.blobs["forecast"];
        assert_eq!(blob.meta["kind"], "quantiles_thq");
        assert_eq!(blob.meta["shape"], json!([2, 4, 5]));
        assert_eq!(blob.meta["names"], json!(["t0", "t1"]));
        assert_eq!(blob.bytes.len(), 2 * 4 * 5 * 4);
    }

    #[test]
    fn timesfm3_observed_mask_over_the_wire_is_invariant_to_the_masked_value() {
        let mut inst = synthetic_timesfm3_instance();
        let observed = [1.0f32, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]; // step 3 unobserved
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 5.0, 5.0, 6.0, 7.0, 8.0];
        let b: Vec<f32> = vec![1.0, 2.0, 3.0, 999_999.0, 5.0, 6.0, 7.0, 8.0];
        let inv_of = |data: &[f32]| {
            Invocation::new()
                .set("horizon", json!(4))
                .blob("context", f32_blob(data, &[8]))
                .blob("observed", f32_blob(&observed, &[8]))
        };
        let out_a = inst.run("forecast", &inv_of(&a), &mut |_| {}).expect("run a");
        let out_b = inst.run("forecast", &inv_of(&b), &mut |_| {}).expect("run b");
        assert_eq!(out_a.blobs["forecast"].bytes, out_b.blobs["forecast"].bytes, "the value at an unobserved wire step must never affect the forecast");
    }

    #[test]
    fn timesfm3_rejects_an_observed_mask_whose_shape_does_not_match_context() {
        let mut inst = synthetic_timesfm3_instance();
        let inv = Invocation::new()
            .set("horizon", json!(4))
            .blob("context", f32_blob(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]))
            .blob("observed", f32_blob(&[1.0, 1.0, 1.0, 1.0], &[4])); // wrong length
        let err = inst.run("forecast", &inv, &mut |_| {}).expect_err("mismatched observed shape must be rejected");
        assert!(err.contains("observed"), "error should name the offending field: {err}");
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
