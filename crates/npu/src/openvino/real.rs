// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real OpenVINO runtime (x86_64 linux/windows). Loads an ONNX graph (fp32 or
//! INT8-QDQ), compiles it to the chosen device — `NPU` by default — and runs
//! inference, handing the raw head tensors back to brain's host DFL-decode+NMS.
//!
//! The OpenVINO shared library is loaded at run time (`runtime-linking`): if it
//! is not installed, [`available_devices`]/[`NpuSession::load`] return
//! [`NpuError::RuntimeNotFound`] rather than failing the build.

use super::{BenchResult, HeadOutputs, NpuConfig, NpuDevice, NpuError, PerfHint};
use openvino::{Core, DeviceType, ElementType, RwPropertyKey, Shape, Tensor};
use std::path::Path;
use std::time::Instant;

fn dev_to_ov(d: NpuDevice) -> DeviceType<'static> {
    match d {
        NpuDevice::Npu => DeviceType::NPU,
        NpuDevice::Cpu => DeviceType::CPU,
        NpuDevice::Gpu => DeviceType::GPU,
        NpuDevice::Auto => DeviceType::Other("AUTO".into()),
    }
}

fn dev_str(d: &DeviceType<'_>) -> String {
    d.as_ref().to_string()
}

/// Is device id `id` (e.g. "NPU.0") an instance of base device `base` (e.g. "NPU")?
fn matches_base(id: &str, base: &str) -> bool {
    id == base || id.starts_with(&format!("{base}."))
}

/// Best-effort: make the OpenVINO runtime discoverable when it was installed via
/// the `openvino` pip wheel (the common case here). The wheel puts the libraries
/// in `<site-packages>/openvino/libs`, which isn't a path the openvino-finder
/// checks by default — but it DOES scan `LD_LIBRARY_PATH`, and the wheel's
/// `libopenvino_c.so` has `RPATH=$ORIGIN` so its dependencies resolve from the
/// same dir. So we just locate that dir (active virtualenv first, then `python3`)
/// and prepend it to `LD_LIBRARY_PATH` in-process before `Core::new`. Respects an
/// already-configured OpenVINO env and is a no-op if nothing is found (the caller
/// then reports `RuntimeNotFound`). This removes the manual env dance for
/// `--device npu` inside a venv with `make requirements` installed.
fn ensure_openvino_on_path() {
    use std::path::{Path, PathBuf};
    // Respect an explicit OpenVINO install env (e.g. a real setupvars.sh).
    if ["OPENVINO_INSTALL_DIR", "INTEL_OPENVINO_DIR", "OPENVINO_BUILD_DIR"]
        .iter()
        .any(|k| std::env::var_os(k).is_some())
    {
        return;
    }
    let has_c = |dir: &Path| {
        std::fs::read_dir(dir)
            .map(|rd| rd.flatten().any(|e| e.file_name().to_string_lossy().starts_with("libopenvino_c.so")))
            .unwrap_or(false)
    };
    // Already reachable on LD_LIBRARY_PATH? Then nothing to do.
    if let Some(p) = std::env::var_os("LD_LIBRARY_PATH") {
        if std::env::split_paths(&p).any(|d| has_c(&d)) {
            return;
        }
    }
    // Find the pip wheel's libs dir: the active venv, then ask python3.
    let mut found: Option<PathBuf> = None;
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        if let Ok(rd) = std::fs::read_dir(Path::new(&venv).join("lib")) {
            for e in rd.flatten() {
                let cand = e.path().join("site-packages/openvino/libs");
                if has_c(&cand) {
                    found = Some(cand);
                    break;
                }
            }
        }
    }
    if found.is_none() {
        if let Ok(out) = std::process::Command::new("python3")
            .args(["-c", "import openvino,os;print(os.path.join(os.path.dirname(openvino.__file__),'libs'))"])
            .output()
        {
            if out.status.success() {
                let dir = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
                if has_c(&dir) {
                    found = Some(dir);
                }
            }
        }
    }
    if let Some(dir) = found {
        let mut paths = vec![dir];
        if let Some(p) = std::env::var_os("LD_LIBRARY_PATH") {
            paths.extend(std::env::split_paths(&p));
        }
        if let Ok(joined) = std::env::join_paths(paths) {
            std::env::set_var("LD_LIBRARY_PATH", joined);
        }
    }
}

fn new_core() -> Result<Core, NpuError> {
    ensure_openvino_on_path();
    Core::new().map_err(|e| NpuError::RuntimeNotFound(format!("{e:?}")))
}

/// OpenVINO devices visible on this machine (e.g. `["CPU", "GPU", "NPU"]`).
/// Returns [`NpuError::RuntimeNotFound`] if OpenVINO is not installed.
pub fn available_devices() -> Result<Vec<String>, NpuError> {
    let core = new_core()?;
    let devs = core.available_devices().map_err(|e| NpuError::Other(format!("{e:?}")))?;
    Ok(devs.iter().map(dev_str).collect())
}

/// Whether a supported Intel NPU is present (per OpenVINO's device list).
pub fn npu_present() -> bool {
    available_devices()
        .map(|ds| ds.iter().any(|d| matches_base(d, "NPU")))
        .unwrap_or(false)
}

/// Resolve the requested device against what's actually present, honouring
/// `allow_fallback` (NPU → GPU → CPU). Returns the device to compile for.
fn resolve_device(
    want: NpuDevice,
    avail: &[String],
    allow_fallback: bool,
) -> Result<DeviceType<'static>, NpuError> {
    if matches!(want, NpuDevice::Auto) {
        return Ok(dev_to_ov(want));
    }
    let want_str = want.ov_str();
    if avail.iter().any(|d| matches_base(d, want_str)) {
        return Ok(dev_to_ov(want));
    }
    if allow_fallback {
        for cand in ["GPU", "CPU"] {
            if avail.iter().any(|d| matches_base(d, cand)) {
                eprintln!(
                    "brain npu: requested device {want_str} not available; falling back to {cand}"
                );
                return Ok(DeviceType::from(cand).to_owned());
            }
        }
    }
    Err(NpuError::DeviceUnavailable(format!(
        "{want_str} not in available OpenVINO devices {avail:?}{}",
        if allow_fallback { " (and no CPU/GPU fallback found)" } else { "" }
    )))
}

/// Resolve the configured device against what's present and apply the config's
/// OpenVINO properties — the shared prologue of every `*Session::load_*`.
fn pick_device(core: &mut Core, cfg: &NpuConfig) -> Result<DeviceType<'static>, NpuError> {
    let avail: Vec<String> = core
        .available_devices()
        .map_err(|e| NpuError::Other(format!("{e:?}")))?
        .iter()
        .map(dev_str)
        .collect();
    let device = resolve_device(cfg.device, &avail, cfg.allow_fallback)?;
    apply_properties(core, &device, cfg);
    Ok(device)
}

/// Apply the `NpuConfig` knobs to the device as OpenVINO properties (best-effort:
/// a property a device rejects is logged, not fatal). NPU-only keys are skipped
/// on non-NPU devices.
fn apply_properties(core: &mut Core, device: &DeviceType<'static>, cfg: &NpuConfig) {
    let dname = dev_str(device);
    let is_npu = matches_base(&dname, "NPU");
    let hint = match cfg.perf_hint {
        PerfHint::Latency => "LATENCY",
        PerfHint::Throughput => "THROUGHPUT",
    };
    let mut set = |key: RwPropertyKey, val: &str| {
        if let Err(e) = core.set_property(device, &key, val) {
            eprintln!("brain npu: set_property {} = {val} on {dname} failed ({e:?}); ignoring", key.as_ref());
        }
    };
    set(RwPropertyKey::HintPerformanceMode, hint);
    if let Some(dir) = &cfg.cache_dir {
        if let Some(s) = dir.to_str() {
            set(RwPropertyKey::CacheDir, s);
        }
    }
    if cfg.profiling {
        set(RwPropertyKey::EnableProfiling, "YES");
    }
    if is_npu {
        if cfg.qdq_opt {
            set(RwPropertyKey::Other("NPU_QDQ_OPTIMIZATION".into()), "YES");
        }
        if cfg.turbo {
            set(RwPropertyKey::Other("NPU_TURBO".into()), "YES");
        }
        if let Some(t) = cfg.tiles {
            set(RwPropertyKey::Other("NPU_TILES".into()), &t.to_string());
        }
        if let Some(p) = &cfg.compilation_params {
            set(RwPropertyKey::Other("NPU_COMPILATION_MODE_PARAMS".into()), p);
        }
    }
}

/// A model compiled to a device, ready to run.
pub struct NpuSession {
    // `core` must outlive the compiled model / request (owns the plugin).
    _core: Core,
    request: openvino::InferRequest,
    input_shape: [usize; 4],
    output_names: Vec<String>,
    device: String,
}

impl NpuSession {
    /// Read an ONNX file (fp32 or INT8-QDQ) and compile it for the configured
    /// device. The ONNX must have a single static `[1,3,S,S]` input.
    pub fn load(onnx_path: &Path, cfg: &NpuConfig) -> Result<Self, NpuError> {
        let bytes = std::fs::read(onnx_path)
            .map_err(|e| NpuError::Other(format!("read {}: {e}", onnx_path.display())))?;
        Self::load_bytes(&bytes, cfg)
    }

    /// Compile ONNX bytes directly (no temp file), e.g. an in-memory fp32 export.
    pub fn load_bytes(bytes: &[u8], cfg: &NpuConfig) -> Result<Self, NpuError> {
        let mut core = new_core()?;

        let avail: Vec<String> = core
            .available_devices()
            .map_err(|e| NpuError::Other(format!("{e:?}")))?
            .iter()
            .map(dev_str)
            .collect();
        let device = resolve_device(cfg.device, &avail, cfg.allow_fallback)?;
        apply_properties(&mut core, &device, cfg);

        let model = core
            .read_model_from_buffer(bytes, None)
            .map_err(|e| NpuError::Other(format!("read_model (ONNX): {e:?}")))?;
        let compiled = core
            .compile_model(&model, device.to_owned())
            .map_err(|e| NpuError::Other(format!("compile_model on {}: {e:?}", dev_str(&device))))?;

        // Static input shape (sanity for the caller + run() validation).
        let in_node = compiled
            .get_input_by_index(0)
            .map_err(|e| NpuError::Other(format!("get_input: {e:?}")))?;
        let in_dims = in_node
            .get_shape()
            .map_err(|e| NpuError::Other(format!("input shape (is it static?): {e:?}")))?;
        let d = in_dims.get_dimensions();
        if d.len() != 4 {
            return Err(NpuError::Other(format!("expected 4-D input, got shape {d:?}")));
        }
        let input_shape = [d[0] as usize, d[1] as usize, d[2] as usize, d[3] as usize];

        let nout = compiled.get_output_size().map_err(|e| NpuError::Other(format!("{e:?}")))?;
        let mut output_names = Vec::with_capacity(nout);
        for i in 0..nout {
            let name = compiled
                .get_output_by_index(i)
                .and_then(|n| n.get_name())
                .unwrap_or_else(|_| format!("output_{i}"));
            output_names.push(name);
        }

        let mut compiled = compiled;
        let request = compiled
            .create_infer_request()
            .map_err(|e| NpuError::Other(format!("create_infer_request: {e:?}")))?;

        Ok(NpuSession { _core: core, request, input_shape, output_names, device: dev_str(&device) })
    }

    /// The static input shape `[N,C,H,W]` the compiled model expects.
    pub fn input_shape(&self) -> [usize; 4] {
        self.input_shape
    }

    /// The OpenVINO device the model was compiled for (e.g. "NPU", or a fallback).
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Run one inference on a preprocessed, letterboxed CHW f32 input and return
    /// the raw head tensors (NCHW), in the graph's output order.
    pub fn run(&mut self, input_chw: &[f32], shape: [usize; 4]) -> Result<HeadOutputs, NpuError> {
        let want: usize = self.input_shape.iter().product();
        if input_chw.len() != want {
            return Err(NpuError::Other(format!(
                "input has {} elems but model expects {} (shape {:?})",
                input_chw.len(),
                want,
                self.input_shape
            )));
        }
        let dims: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        let ov_shape = Shape::new(&dims).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        let mut tensor =
            Tensor::new(ElementType::F32, &ov_shape).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        {
            let dst = tensor.get_data_mut::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?;
            dst.copy_from_slice(input_chw);
        }
        self.request.set_input_tensor(&tensor).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        self.request.infer().map_err(|e| NpuError::Other(format!("infer: {e:?}")))?;

        let mut tensors = Vec::with_capacity(self.output_names.len());
        for (i, name) in self.output_names.iter().enumerate() {
            let t = self
                .request
                .get_output_tensor_by_index(i)
                .map_err(|e| NpuError::Other(format!("get_output {i}: {e:?}")))?;
            let sh: Vec<usize> =
                t.get_shape().map_err(|e| NpuError::Other(format!("{e:?}")))?.get_dimensions().iter().map(|&x| x as usize).collect();
            let data = t.get_data::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?.to_vec();
            tensors.push((name.clone(), sh, data));
        }
        Ok(HeadOutputs { tensors })
    }
}

/// A compiled decoder graph: `input_ids:[1,T]` (int64) -> `logits:[1,T,vocab]`
/// (f32). Separate from [`NpuSession`] (which is the YOLO 4-D-f32 shape).
pub struct DecoderSession {
    _core: Core,
    request: openvino::InferRequest,
    seq_len: usize,
    vocab: usize,
    device: String,
}

impl DecoderSession {
    /// Compile a decoder ONNX from a file path. Required for models with ONNX
    /// external data (the reader resolves the sidecar relative to the model dir).
    pub fn load_path(onnx_path: &Path, cfg: &NpuConfig) -> Result<Self, NpuError> {
        let mut core = new_core()?;
        let avail: Vec<String> = core
            .available_devices()
            .map_err(|e| NpuError::Other(format!("{e:?}")))?
            .iter()
            .map(dev_str)
            .collect();
        let device = resolve_device(cfg.device, &avail, cfg.allow_fallback)?;
        apply_properties(&mut core, &device, cfg);
        // ONNX external-data is resolved relative to the model file's directory;
        // the IR weights_path is unused for ONNX, so pass "".
        let path_str = onnx_path.to_str().ok_or_else(|| NpuError::Other("non-utf8 path".into()))?;
        let model = core
            .read_model_from_file(path_str, "")
            .map_err(|e| NpuError::Other(format!("read_model {}: {e:?}", onnx_path.display())))?;
        Self::compile(core, model, device)
    }

    /// Compile ONNX decoder bytes for the configured device (no external data).
    pub fn load_bytes(bytes: &[u8], cfg: &NpuConfig) -> Result<Self, NpuError> {
        let mut core = new_core()?;
        let avail: Vec<String> = core
            .available_devices()
            .map_err(|e| NpuError::Other(format!("{e:?}")))?
            .iter()
            .map(dev_str)
            .collect();
        let device = resolve_device(cfg.device, &avail, cfg.allow_fallback)?;
        apply_properties(&mut core, &device, cfg);

        let model = core
            .read_model_from_buffer(bytes, None)
            .map_err(|e| NpuError::Other(format!("read_model (ONNX): {e:?}")))?;
        Self::compile(core, model, device)
    }

    fn compile(mut core: Core, model: openvino::Model, device: DeviceType<'static>) -> Result<Self, NpuError> {
        let compiled = core
            .compile_model(&model, device.to_owned())
            .map_err(|e| NpuError::Other(format!("compile_model on {}: {e:?}", dev_str(&device))))?;

        let in_dims = compiled
            .get_input_by_index(0)
            .and_then(|n| n.get_shape())
            .map_err(|e| NpuError::Other(format!("input shape: {e:?}")))?;
        let id = in_dims.get_dimensions();
        let seq_len = *id.last().unwrap() as usize;

        let out_dims = compiled
            .get_output_by_index(0)
            .and_then(|n| n.get_shape())
            .map_err(|e| NpuError::Other(format!("output shape: {e:?}")))?;
        let od = out_dims.get_dimensions();
        let vocab = *od.last().unwrap() as usize;

        let mut compiled = compiled;
        let request = compiled
            .create_infer_request()
            .map_err(|e| NpuError::Other(format!("create_infer_request: {e:?}")))?;
        Ok(DecoderSession { _core: core, request, seq_len, vocab, device: dev_str(&device) })
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Run the prefill over `ids` (length must equal the compiled `seq_len`;
    /// shorter contexts are caller-padded). Returns the full `[T*vocab]` logits.
    pub fn run_ids(&mut self, ids: &[i64]) -> Result<Vec<f32>, NpuError> {
        if ids.len() != self.seq_len {
            return Err(NpuError::Other(format!(
                "decoder expects {} ids, got {}",
                self.seq_len,
                ids.len()
            )));
        }
        let shape = Shape::new(&[1, self.seq_len as i64]).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        let mut tensor =
            Tensor::new(ElementType::I64, &shape).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        {
            let dst = tensor.get_data_mut::<i64>().map_err(|e| NpuError::Other(format!("{e:?}")))?;
            dst.copy_from_slice(ids);
        }
        self.request.set_input_tensor(&tensor).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        self.request.infer().map_err(|e| NpuError::Other(format!("infer: {e:?}")))?;
        let out = self
            .request
            .get_output_tensor_by_index(0)
            .map_err(|e| NpuError::Other(format!("get_output: {e:?}")))?;
        let data = out.get_data::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?.to_vec();
        Ok(data)
    }
}

/// A compiled graph with a single f32 input `inputs_embeds:[1,T,d_in]` and a
/// single f32 output `hidden:[1,T,d_out]` — the Qwen3-TTS Talker hidden-state
/// graph. Like [`DecoderSession`] it is a cache-free fixed-length prefill, but
/// driven by an input-embedding stream rather than token ids: the autoregressive
/// loop pads the real context to `T` and reads the hidden row at the last real
/// position (causal masking makes it independent of the zero padding).
pub struct EmbedSession {
    _core: Core,
    request: openvino::InferRequest,
    seq_len: usize,
    d_in: usize,
    d_out: usize,
    device: String,
}

impl EmbedSession {
    /// Compile from a file path (required for ONNX external data — the sidecar is
    /// resolved relative to the model dir).
    pub fn load_path(onnx_path: &Path, cfg: &NpuConfig) -> Result<Self, NpuError> {
        let mut core = new_core()?;
        let device = pick_device(&mut core, cfg)?;
        let path_str = onnx_path.to_str().ok_or_else(|| NpuError::Other("non-utf8 path".into()))?;
        let model = core
            .read_model_from_file(path_str, "")
            .map_err(|e| NpuError::Other(format!("read_model {}: {e:?}", onnx_path.display())))?;
        Self::compile(core, model, device)
    }

    /// Compile from ONNX bytes (no external data).
    pub fn load_bytes(bytes: &[u8], cfg: &NpuConfig) -> Result<Self, NpuError> {
        let mut core = new_core()?;
        let device = pick_device(&mut core, cfg)?;
        let model = core
            .read_model_from_buffer(bytes, None)
            .map_err(|e| NpuError::Other(format!("read_model (ONNX): {e:?}")))?;
        Self::compile(core, model, device)
    }

    fn compile(mut core: Core, model: openvino::Model, device: DeviceType<'static>) -> Result<Self, NpuError> {
        let compiled = core
            .compile_model(&model, device.to_owned())
            .map_err(|e| NpuError::Other(format!("compile_model on {}: {e:?}", dev_str(&device))))?;
        let id = compiled
            .get_input_by_index(0)
            .and_then(|n| n.get_shape())
            .map_err(|e| NpuError::Other(format!("input shape: {e:?}")))?;
        let id = id.get_dimensions();
        if id.len() != 3 {
            return Err(NpuError::Other(format!("expected 3-D inputs_embeds [1,T,d], got {id:?}")));
        }
        let seq_len = id[1] as usize;
        let d_in = id[2] as usize;
        let od = compiled
            .get_output_by_index(0)
            .and_then(|n| n.get_shape())
            .map_err(|e| NpuError::Other(format!("output shape: {e:?}")))?;
        let d_out = *od.get_dimensions().last().unwrap() as usize;
        let mut compiled = compiled;
        let request = compiled
            .create_infer_request()
            .map_err(|e| NpuError::Other(format!("create_infer_request: {e:?}")))?;
        Ok(EmbedSession { _core: core, request, seq_len, d_in, d_out, device: dev_str(&device) })
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
    pub fn d_in(&self) -> usize {
        self.d_in
    }
    pub fn d_out(&self) -> usize {
        self.d_out
    }
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Run the prefill over `embeds` (length must equal `seq_len * d_in`; shorter
    /// contexts are caller-padded with zeros). Returns the full `[seq_len*d_out]`
    /// hidden states.
    pub fn run_embeds(&mut self, embeds: &[f32]) -> Result<Vec<f32>, NpuError> {
        let want = self.seq_len * self.d_in;
        if embeds.len() != want {
            return Err(NpuError::Other(format!(
                "embed session expects {want} f32 ({}x{}), got {}",
                self.seq_len,
                self.d_in,
                embeds.len()
            )));
        }
        let shape = Shape::new(&[1, self.seq_len as i64, self.d_in as i64])
            .map_err(|e| NpuError::Other(format!("{e:?}")))?;
        let mut tensor =
            Tensor::new(ElementType::F32, &shape).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        {
            let dst = tensor.get_data_mut::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?;
            dst.copy_from_slice(embeds);
        }
        self.request.set_input_tensor(&tensor).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        self.request.infer().map_err(|e| NpuError::Other(format!("infer: {e:?}")))?;
        let out = self
            .request
            .get_output_tensor_by_index(0)
            .map_err(|e| NpuError::Other(format!("get_output: {e:?}")))?;
        let data = out.get_data::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?.to_vec();
        Ok(data)
    }
}

/// A compiled codec-decoder graph: int64 `codes:[nq,T]` (codebook-major) ->
/// f32 `waveform:[1,1,L]`. Single whole-graph inference (no autoregression).
pub struct CodecSession {
    _core: Core,
    request: openvino::InferRequest,
    nq: usize,
    code_len: usize,
    out_len: usize,
    device: String,
}

impl CodecSession {
    pub fn load_path(onnx_path: &Path, cfg: &NpuConfig) -> Result<Self, NpuError> {
        let mut core = new_core()?;
        let device = pick_device(&mut core, cfg)?;
        let path_str = onnx_path.to_str().ok_or_else(|| NpuError::Other("non-utf8 path".into()))?;
        let model = core
            .read_model_from_file(path_str, "")
            .map_err(|e| NpuError::Other(format!("read_model {}: {e:?}", onnx_path.display())))?;
        Self::compile(core, model, device)
    }

    pub fn load_bytes(bytes: &[u8], cfg: &NpuConfig) -> Result<Self, NpuError> {
        let mut core = new_core()?;
        let device = pick_device(&mut core, cfg)?;
        let model = core
            .read_model_from_buffer(bytes, None)
            .map_err(|e| NpuError::Other(format!("read_model (ONNX): {e:?}")))?;
        Self::compile(core, model, device)
    }

    fn compile(mut core: Core, model: openvino::Model, device: DeviceType<'static>) -> Result<Self, NpuError> {
        let compiled = core
            .compile_model(&model, device.to_owned())
            .map_err(|e| NpuError::Other(format!("compile_model on {}: {e:?}", dev_str(&device))))?;
        let id = compiled
            .get_input_by_index(0)
            .and_then(|n| n.get_shape())
            .map_err(|e| NpuError::Other(format!("input shape: {e:?}")))?;
        let id = id.get_dimensions();
        if id.len() != 2 {
            return Err(NpuError::Other(format!("expected 2-D codes [nq,T], got {id:?}")));
        }
        let nq = id[0] as usize;
        let code_len = id[1] as usize;
        let od = compiled
            .get_output_by_index(0)
            .and_then(|n| n.get_shape())
            .map_err(|e| NpuError::Other(format!("output shape: {e:?}")))?;
        let out_len: usize = od.get_dimensions().iter().map(|&x| x as usize).product();
        let mut compiled = compiled;
        let request = compiled
            .create_infer_request()
            .map_err(|e| NpuError::Other(format!("create_infer_request: {e:?}")))?;
        Ok(CodecSession { _core: core, request, nq, code_len, out_len, device: dev_str(&device) })
    }

    pub fn nq(&self) -> usize {
        self.nq
    }
    pub fn code_len(&self) -> usize {
        self.code_len
    }
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Decode `codes` (length `nq * code_len`, codebook-major) to the waveform.
    pub fn run_codes(&mut self, codes: &[i64]) -> Result<Vec<f32>, NpuError> {
        let want = self.nq * self.code_len;
        if codes.len() != want {
            return Err(NpuError::Other(format!(
                "codec session expects {want} codes ({}x{}), got {}",
                self.nq,
                self.code_len,
                codes.len()
            )));
        }
        let shape = Shape::new(&[self.nq as i64, self.code_len as i64])
            .map_err(|e| NpuError::Other(format!("{e:?}")))?;
        let mut tensor =
            Tensor::new(ElementType::I64, &shape).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        {
            let dst = tensor.get_data_mut::<i64>().map_err(|e| NpuError::Other(format!("{e:?}")))?;
            dst.copy_from_slice(codes);
        }
        self.request.set_input_tensor(&tensor).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        self.request.infer().map_err(|e| NpuError::Other(format!("infer: {e:?}")))?;
        let out = self
            .request
            .get_output_tensor_by_index(0)
            .map_err(|e| NpuError::Other(format!("get_output: {e:?}")))?;
        let data = out.get_data::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?.to_vec();
        Ok(data)
    }

    pub fn out_len(&self) -> usize {
        self.out_len
    }
}

/// A compiled **KV-cache decode-step** Talker graph (see
/// [`crate::qwen_topology::build_talker_decode_graph`]): one token + per-layer
/// past K/V in, hidden + per-layer new K/V out. Dimensions are supplied by the
/// caller (known from the Talker config) rather than introspected.
pub struct KvSession {
    _core: Core,
    request: openvino::InferRequest,
    n_layers: usize,
    d: usize,
    nkv: usize,
    hd: usize,
    cap: usize,
    device: String,
}

impl KvSession {
    #[allow(clippy::too_many_arguments)]
    pub fn load_path(
        onnx_path: &Path,
        cfg: &NpuConfig,
        n_layers: usize,
        d: usize,
        nkv: usize,
        hd: usize,
        cap: usize,
    ) -> Result<Self, NpuError> {
        let mut core = new_core()?;
        let device = pick_device(&mut core, cfg)?;
        let path_str = onnx_path.to_str().ok_or_else(|| NpuError::Other("non-utf8 path".into()))?;
        let model = core
            .read_model_from_file(path_str, "")
            .map_err(|e| NpuError::Other(format!("read_model {}: {e:?}", onnx_path.display())))?;
        let compiled = core
            .compile_model(&model, device.to_owned())
            .map_err(|e| NpuError::Other(format!("compile_model on {}: {e:?}", dev_str(&device))))?;
        let mut compiled = compiled;
        let request = compiled
            .create_infer_request()
            .map_err(|e| NpuError::Other(format!("create_infer_request: {e:?}")))?;
        Ok(KvSession {
            _core: core,
            request,
            n_layers,
            d,
            nkv,
            hd,
            cap,
            device: dev_str(&device),
        })
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    fn set_f32(&mut self, holders: &mut Vec<Tensor>, name: &str, dims: &[i64], data: &[f32]) -> Result<(), NpuError> {
        let shape = Shape::new(dims).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        let mut t = Tensor::new(ElementType::F32, &shape).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        t.get_data_mut::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?.copy_from_slice(data);
        self.request.set_tensor(name, &t).map_err(|e| NpuError::Other(format!("set {name}: {e:?}")))?;
        holders.push(t);
        Ok(())
    }

    /// Run one decode step. `past_k`/`past_v` are `n_layers` buffers of
    /// `nkv*cap*hd` f32 (layout `[nkv,cap,hd]`); `mask` is `[cap]` additive (0 for
    /// filled slots, -inf otherwise); `cos`/`sin` are `[hd]` for this position.
    /// Returns `(hidden[d], new_k[L][nkv*hd], new_v[L][nkv*hd])`.
    #[allow(clippy::type_complexity)]
    pub fn run_step(
        &mut self,
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        mask: &[f32],
        past_k: &[Vec<f32>],
        past_v: &[Vec<f32>],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>), NpuError> {
        let (d, nkv, hd, cap, nl) = (self.d as i64, self.nkv as i64, self.hd as i64, self.cap as i64, self.n_layers);
        let mut holders: Vec<Tensor> = Vec::with_capacity(4 + 2 * nl);
        self.set_f32(&mut holders, "x", &[1, 1, d], x)?;
        self.set_f32(&mut holders, "rope_cos", &[1, 1, 1, hd], cos)?;
        self.set_f32(&mut holders, "rope_sin", &[1, 1, 1, hd], sin)?;
        self.set_f32(&mut holders, "past_mask", &[1, 1, 1, cap], mask)?;
        for l in 0..nl {
            self.set_f32(&mut holders, &format!("past_k_{l}"), &[1, nkv, cap, hd], &past_k[l])?;
            self.set_f32(&mut holders, &format!("past_v_{l}"), &[1, nkv, cap, hd], &past_v[l])?;
        }
        self.request.infer().map_err(|e| NpuError::Other(format!("infer: {e:?}")))?;

        let get = |req: &openvino::InferRequest, name: &str| -> Result<Vec<f32>, NpuError> {
            let t = req.get_tensor(name).map_err(|e| NpuError::Other(format!("get {name}: {e:?}")))?;
            Ok(t.get_data::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?.to_vec())
        };
        let hidden = get(&self.request, "hidden")?;
        let mut new_k = Vec::with_capacity(nl);
        let mut new_v = Vec::with_capacity(nl);
        for l in 0..nl {
            new_k.push(get(&self.request, &format!("new_k_{l}"))?);
            new_v.push(get(&self.request, &format!("new_v_{l}"))?);
        }
        drop(holders);
        Ok((hidden, new_k, new_v))
    }
}

/// A compiled **prefill** Talker graph (full context -> hidden + per-layer K/V):
/// seeds the decode KV cache for the whole prompt prefix in one inference.
pub struct PrefillSession {
    _core: Core,
    request: openvino::InferRequest,
    n_layers: usize,
    d: usize,
    nkv: usize,
    hd: usize,
    cap: usize,
    device: String,
}

impl PrefillSession {
    #[allow(clippy::too_many_arguments)]
    pub fn load_path(
        onnx_path: &Path,
        cfg: &NpuConfig,
        n_layers: usize,
        d: usize,
        nkv: usize,
        hd: usize,
        cap: usize,
    ) -> Result<Self, NpuError> {
        let mut core = new_core()?;
        let device = pick_device(&mut core, cfg)?;
        let path_str = onnx_path.to_str().ok_or_else(|| NpuError::Other("non-utf8 path".into()))?;
        let model = core
            .read_model_from_file(path_str, "")
            .map_err(|e| NpuError::Other(format!("read_model {}: {e:?}", onnx_path.display())))?;
        let compiled = core
            .compile_model(&model, device.to_owned())
            .map_err(|e| NpuError::Other(format!("compile_model on {}: {e:?}", dev_str(&device))))?;
        let mut compiled = compiled;
        let request = compiled
            .create_infer_request()
            .map_err(|e| NpuError::Other(format!("create_infer_request: {e:?}")))?;
        Ok(PrefillSession { _core: core, request, n_layers, d, nkv, hd, cap, device: dev_str(&device) })
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    /// Run the prefill over `embeds` (length `cap*d`, zero-padded by the caller).
    /// Returns `(hidden[cap*d], k[L][nkv*cap*hd], v[L][nkv*cap*hd])`.
    #[allow(clippy::type_complexity)]
    pub fn run(&mut self, embeds: &[f32]) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>), NpuError> {
        let shape = Shape::new(&[1, self.cap as i64, self.d as i64]).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        let mut t = Tensor::new(ElementType::F32, &shape).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        t.get_data_mut::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?.copy_from_slice(embeds);
        self.request.set_input_tensor(&t).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        self.request.infer().map_err(|e| NpuError::Other(format!("infer: {e:?}")))?;
        let get = |req: &openvino::InferRequest, name: &str| -> Result<Vec<f32>, NpuError> {
            let t = req.get_tensor(name).map_err(|e| NpuError::Other(format!("get {name}: {e:?}")))?;
            Ok(t.get_data::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?.to_vec())
        };
        let hidden = get(&self.request, "hidden")?;
        let mut k = Vec::with_capacity(self.n_layers);
        let mut v = Vec::with_capacity(self.n_layers);
        let _ = (self.nkv, self.hd);
        for l in 0..self.n_layers {
            k.push(get(&self.request, &format!("k_{l}"))?);
            v.push(get(&self.request, &format!("v_{l}"))?);
        }
        Ok((hidden, k, v))
    }
}

/// Warm-up then time `iters` inferences; report p50/p99/mean latency + throughput.
pub fn bench(
    session: &mut NpuSession,
    input_chw: &[f32],
    shape: [usize; 4],
    warmup: usize,
    iters: usize,
) -> Result<BenchResult, NpuError> {
    for _ in 0..warmup.max(1) {
        session.run(input_chw, shape)?;
    }
    let mut samples = Vec::with_capacity(iters);
    let t0 = Instant::now();
    for _ in 0..iters.max(1) {
        let t = Instant::now();
        session.run(input_chw, shape)?;
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let wall = t0.elapsed().as_secs_f64();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| samples[((samples.len() as f64 * p) as usize).min(samples.len() - 1)];
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    Ok(BenchResult {
        device: session.device().to_string(),
        iters: samples.len(),
        p50_ms: pct(0.50),
        p99_ms: pct(0.99),
        mean_ms: mean,
        throughput_fps: samples.len() as f64 / wall,
    })
}
