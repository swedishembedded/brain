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
use npu::openvino::{Chronos2Session, Feed, FincastSession, NpuConfig, NpuDevice, NpuGraph};
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
        let (model, path, cfg, cache, devcell) = (&self.model, &self.path, &self.cfg, &self.cache, &self.device);
        let q = guard_npu(|| {
            model.forecast_quantiles_with_core(&context, horizon, |emb, mask, n_out| {
                let s = emb.len() / d;
                let mut c = cache.borrow_mut();
                let sess = c.entry((s, n_out)).or_insert_with(|| {
                    let bytes = npu::chronos2_export::export_onnx(path, s, n_out, npu::qwen_topology::Quant::F32)
                        .expect("chronos2 NPU export");
                    Chronos2Session::load_bytes(&bytes, cfg).expect("chronos2 NPU compile")
                });
                *devcell.borrow_mut() = sess.device().to_string();
                sess.run(emb, mask).expect("chronos2 NPU infer")
            })
        })?;
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
    /// Explicit `.safetensors` path (the `brain perf` target and non-env callers).
    pub fn new(path: &str) -> FincastResident {
        FincastResident { path: path.to_string() }
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
        let (model, path, cfg, cache, devcell) = (&self.model, &self.path, &self.cfg, &self.cache, &self.device);
        let out = guard_npu(|| {
            model.forecast_full_with_core(&context, freq, horizon, |emb, amask| {
                let s = (amask.len() as f64).sqrt() as usize;
                let mut c = cache.borrow_mut();
                let sess = c.entry(s).or_insert_with(|| {
                    // External-data export (large model) → compile from file, then
                    // drop the sidecar; the compiled blob owns the weights.
                    let dir = std::env::temp_dir().join(format!("brain-fincast-{}-{s}", std::process::id()));
                    std::fs::create_dir_all(&dir).ok();
                    let onnx = dir.join("model.onnx");
                    let op = onnx.to_str().expect("utf8 temp path");
                    npu::fincast_export::export_external(path, s, npu::qwen_topology::Quant::F32, op)
                        .expect("fincast NPU export");
                    let sess = FincastSession::load_path(op, cfg).expect("fincast NPU compile");
                    std::fs::remove_dir_all(&dir).ok();
                    sess
                });
                *devcell.borrow_mut() = sess.device().to_string();
                sess.run(emb, amask).expect("fincast NPU infer")
            })
        })?;
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

impl ResidentModel for KronosResident {
    fn manifest(&self) -> Manifest {
        Manifest::new("kronos", "autoregressive OHLCV forecasting (Kronos)", vec![Self::spec()])
    }
    fn instance_key(&self, _action: &str, inv: &Invocation) -> InstanceKey {
        // One hot instance per decoder checkpoint (any horizon); see decoder_key.
        InstanceKey::new("kronos", decoder_key(&self.decoder_for(inv)))
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
        .set("model", json!("kronos"))
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
                .set("model", json!("kronos"))
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
                .set("model", json!("kronos"))
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
        let half = dep_hd / 2;
        let pos = self.s1_pos - 1; // the ctx_last (self) absolute position
        let (cos, sin) = rope_tables(pos, half, dep_hd);
        let mask: Vec<f32> = (0..cap).map(|j| if j < self.dep_valid { 0.0 } else { -1e9 }).collect();
        let ctx_last = std::mem::take(&mut self.ctx_last);
        let out = {
            let feeds: Vec<(&str, Feed)> = vec![
                ("sib", Feed::F32(sib, vec![1, 1, d as i64])),
                ("ctx_last", Feed::F32(&ctx_last, vec![1, 1, d as i64])),
                ("rope_cos", Feed::F32(&cos, vec![1, 1, 1, half as i64])),
                ("rope_sin", Feed::F32(&sin, vec![1, 1, 1, half as i64])),
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
            for l in 0..nl {
                feeds.push((keys[l].0.as_str(), Feed::F32(&self.pk[l], vec![1, heads as i64, cap as i64, hd as i64])));
                feeds.push((keys[l].1.as_str(), Feed::F32(&self.pv[l], vec![1, heads as i64, cap as i64, hd as i64])));
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
