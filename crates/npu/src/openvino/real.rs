// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real OpenVINO runtime (x86_64 linux/windows). Loads an ONNX graph (fp32 or
//! INT8-QDQ), compiles it to the chosen device — `NPU` by default — and runs
//! inference, handing the raw head tensors back to brain's host DFL-decode+NMS.
//!
//! The OpenVINO shared library is loaded at run time (`runtime-linking`): if it
//! is not installed, [`available_devices`]/[`NpuSession::load`] return
//! [`NpuError::RuntimeNotFound`] rather than failing the build.

use super::{BenchResult, DeviceInfo, HeadOutputs, NpuConfig, NpuDevice, NpuError, PerfHint};
use openvino::{Core, DeviceType, ElementType, PropertyKey, RwPropertyKey, Shape, Tensor};
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
    // Already reachable on LD_LIBRARY_PATH? A prefix match isn't enough here — the
    // pip wheel ships only the versioned file, so a dir containing just
    // `libopenvino_c.so.2630` would (wrongly) satisfy a `starts_with` check and skip
    // the symlink creation below, leaving the unversioned dlopen to fail later.
    let has_unversioned_c = |dir: &Path| dir.join("libopenvino_c.so").is_file();
    if let Some(p) = std::env::var_os("LD_LIBRARY_PATH") {
        if std::env::split_paths(&p).any(|d| has_unversioned_c(&d)) {
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
        // The openvino-sys loader dlopens the UNVERSIONED `libopenvino_c.so`, but the
        // pip wheel ships only versioned files (e.g. `libopenvino_c.so.2620` for
        // 2026.2). Create the missing unversioned symlinks so the runtime resolves —
        // best-effort (a read-only libs dir just falls through to `RuntimeNotFound`,
        // whose message tells the user to set LD_LIBRARY_PATH / symlink manually).
        #[cfg(unix)]
        ensure_unversioned_solinks(&dir);
        let mut paths = vec![dir];
        if let Some(p) = std::env::var_os("LD_LIBRARY_PATH") {
            paths.extend(std::env::split_paths(&p));
        }
        if let Ok(joined) = std::env::join_paths(paths) {
            std::env::set_var("LD_LIBRARY_PATH", joined);
        }
    }
}

/// In an OpenVINO libs dir, create `lib<name>.so → lib<name>.so.<ver>` for each core
/// library when the unversioned link is missing (the pip wheel omits it). Idempotent,
/// best-effort — errors are ignored (the caller reports `RuntimeNotFound` if the
/// runtime still cannot load).
#[cfg(unix)]
fn ensure_unversioned_solinks(dir: &std::path::Path) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let files: Vec<String> = rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    for base in ["libopenvino_c", "libopenvino", "libopenvino_onnx_frontend", "libopenvino_ir_frontend"] {
        let unversioned = format!("{base}.so");
        if files.iter().any(|f| f == &unversioned) {
            continue; // already linked / present
        }
        // pick the first versioned match `libX.so.<...>`
        if let Some(target) = files.iter().find(|f| f.starts_with(&format!("{base}.so."))) {
            let _ = std::os::unix::fs::symlink(target, dir.join(&unversioned));
        }
    }
}

/// Build an OpenVINO `Core` under the shared NPU device-init lock.
///
/// Same defect class as the GPU backends' device creation, on physically
/// different silicon, so it takes the NPU key rather than the GPU one (see
/// [`backend_api::hardware`]). Two things here are unsafe to run
/// concurrently:
///
/// * [`ensure_openvino_on_path`] mutates the process-global
///   `LD_LIBRARY_PATH`. Two threads opening sessions at once race that write
///   against every other thread's read of the environment.
/// * `Core::new` makes the runtime `dlopen` its plugin set, and the device
///   plugin then opens the accelerator - the loader path this workspace has
///   already been bitten by on the graphics side, and one that another
///   THREAD of this same process contends for identically.
///
/// Held only across construction: the compiles that follow are long, and
/// serialising those in-process would cost far more than the race is worth.
/// Bounding the driver-side compile itself is a separate, larger change: the
/// compile borrows the `Core` and the model, so it cannot cross into a worker
/// thread without restructuring session construction.
fn new_core() -> Result<Core, NpuError> {
    let _init = backend_api::hardware::device_class_lock(backend_api::hardware::NPU);
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

/// Resolve `device` (honouring `allow_fallback`) and report what it actually is:
/// the resolved device string, its `FULL_DEVICE_NAME`, and its
/// `OPTIMIZATION_CAPABILITIES`. Lets the caller print, at startup, the real
/// hardware path and whether a requested weight precision is natively supported.
pub fn device_info(device: NpuDevice, allow_fallback: bool) -> Result<DeviceInfo, NpuError> {
    let core = new_core()?;
    let avail: Vec<String> = core
        .available_devices()
        .map_err(|e| NpuError::Other(format!("{e:?}")))?
        .iter()
        .map(dev_str)
        .collect();
    let dev = resolve_device(device, &avail, allow_fallback)?;
    let full_name = core
        .get_property(&dev, &PropertyKey::DeviceFullName)
        .unwrap_or_else(|_| "unknown".to_string());
    let caps_raw = core
        .get_property(&dev, &PropertyKey::DeviceCapabilities)
        .unwrap_or_default();
    // OV returns the list as a string (space- or comma-separated, sometimes
    // bracketed) — split into tokens.
    let capabilities = caps_raw
        .split([' ', ',', '[', ']', '\'', '"'])
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    Ok(DeviceInfo { device: dev_str(&dev), full_name, capabilities })
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

/// A model compiled to a device, ready to run. Thin wrapper over [`NpuGraph`]:
/// keeps the vision-specific `[N,C,H,W]` shape validation and the `HeadOutputs`
/// return shape callers depend on, delegates the actual compile/set/infer/read
/// to the generic runner.
pub struct NpuSession {
    graph: NpuGraph,
    input_shape: [usize; 4],
}

/// One named output tensor: `(name, shape, row-major f32 data)`.
pub type NamedTensor = (String, Vec<usize>, Vec<f32>);

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
        let (mut core, device) = open_device(cfg)?;
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

        let graph = NpuGraph::from_compiled(core, compiled, device)?;
        Ok(NpuSession { graph, input_shape })
    }

    /// The static input shape `[N,C,H,W]` the compiled model expects.
    pub fn input_shape(&self) -> [usize; 4] {
        self.input_shape
    }

    /// The OpenVINO device the model was compiled for (e.g. "NPU", or a fallback).
    pub fn device(&self) -> &str {
        self.graph.device()
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
        let tensors = self.graph.run(&[("input", Feed::F32(input_chw, dims))])?;
        Ok(HeadOutputs { tensors })
    }
}

/// A compiled decoder graph: `input_ids:[1,T]` (int64) -> `logits:[1,T,vocab]`
/// (f32). Separate from [`NpuSession`] (which is the YOLO 4-D-f32 shape). Thin
/// wrapper over [`NpuGraph`]: keeps the typed `seq_len`/`vocab` metadata and
/// `run_ids` signature callers depend on, delegates to the generic runner.
pub struct DecoderSession {
    graph: NpuGraph,
    seq_len: usize,
    vocab: usize,
}

impl DecoderSession {
    /// Compile a decoder ONNX from a file path. Required for models with ONNX
    /// external data (the reader resolves the sidecar relative to the model dir).
    pub fn load_path(onnx_path: &Path, cfg: &NpuConfig) -> Result<Self, NpuError> {
        let (mut core, device) = open_device(cfg)?;
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
        let (mut core, device) = open_device(cfg)?;
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

        let graph = NpuGraph::from_compiled(core, compiled, device)?;
        Ok(DecoderSession { graph, seq_len, vocab })
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    pub fn device(&self) -> &str {
        self.graph.device()
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
        let dims = vec![1, self.seq_len as i64];
        let out = self.graph.run(&[("input_ids", Feed::I64(ids, dims))])?;
        Ok(out.into_iter().next().map(|(_, _, data)| data).unwrap_or_default())
    }
}

/// One named input tensor for [`NpuGraph::run`].
pub enum Feed<'a> {
    F32(&'a [f32], Vec<i64>),
    I64(&'a [i64], Vec<i64>),
}

impl Feed<'_> {
    fn dims(&self) -> &[i64] {
        match self {
            Feed::F32(_, d) | Feed::I64(_, d) => d,
        }
    }
    fn elem(&self) -> ElementType {
        match self {
            Feed::F32(..) => ElementType::F32,
            Feed::I64(..) => ElementType::I64,
        }
    }
}

/// A **generic** compiled OpenVINO graph with named multi-tensor I/O — the one
/// reuse seam every model's NPU export shares. Any model's exported ONNX (fp16 on the NPU
/// by default; INT8/INT4 orthogonal) compiles here once and runs via [`run`], feeding
/// f32/i64 inputs by name and reading f32 outputs by name. This generalises the
/// per-model bespoke sessions so a residency `NpuInstance` is model-agnostic: a model
/// only supplies its `crate::NpuModel` (graph bytes); `NpuGraph` does compile / cache
/// / infer / evict.
///
/// [`run`]: NpuGraph::run
pub struct NpuGraph {
    _core: Core,
    request: openvino::InferRequest,
    input_names: Vec<String>,
    output_names: Vec<String>,
    device: String,
}

impl NpuGraph {
    /// Compile in-memory ONNX bytes (no external data).
    pub fn compile_bytes(bytes: &[u8], cfg: &NpuConfig) -> Result<NpuGraph, NpuError> {
        let (mut core, device) = open_device(cfg)?;
        let model = core
            .read_model_from_buffer(bytes, None)
            .map_err(|e| NpuError::Other(format!("read_model (ONNX): {e:?}")))?;
        Self::finish(core, model, device)
    }

    /// Compile an ONNX file (required for graphs with external-data sidecars).
    pub fn compile_path(path: &Path, cfg: &NpuConfig) -> Result<NpuGraph, NpuError> {
        let (mut core, device) = open_device(cfg)?;
        let ps = path.to_str().ok_or_else(|| NpuError::Other("non-utf8 path".into()))?;
        let model = core
            .read_model_from_file(ps, "")
            .map_err(|e| NpuError::Other(format!("read_model {}: {e:?}", path.display())))?;
        Self::finish(core, model, device)
    }

    fn finish(mut core: Core, model: openvino::Model, device: DeviceType<'static>) -> Result<NpuGraph, NpuError> {
        let compiled = core
            .compile_model(&model, device.to_owned())
            .map_err(|e| NpuError::Other(format!("compile_model on {}: {e:?}", dev_str(&device))))?;
        Self::from_compiled(core, compiled, device)
    }

    /// Wrap an already-compiled model (its shape/topology metadata was already
    /// pulled out of `compiled` by the caller) as a generic named-I/O graph - the
    /// shared tail every bespoke `*Session::compile` in this module also uses, so
    /// a session that needs typed metadata (`seq_len`, `vocab`, ...) can extract it
    /// from `compiled` itself and then hand the same compiled model here instead
    /// of hand-rolling its own `create_infer_request`/name-introspection.
    fn from_compiled(core: Core, compiled: openvino::CompiledModel, device: DeviceType<'static>) -> Result<NpuGraph, NpuError> {
        let nin = compiled.get_input_size().map_err(|e| NpuError::Other(format!("get_input_size: {e:?}")))?;
        let input_names = (0..nin)
            .map(|i| compiled.get_input_by_index(i).and_then(|n| n.get_name()).unwrap_or_else(|_| format!("input_{i}")))
            .collect();
        let nout = compiled.get_output_size().map_err(|e| NpuError::Other(format!("get_output_size: {e:?}")))?;
        let output_names = (0..nout)
            .map(|i| compiled.get_output_by_index(i).and_then(|n| n.get_name()).unwrap_or_else(|_| format!("output_{i}")))
            .collect();
        let mut compiled = compiled;
        let request = compiled
            .create_infer_request()
            .map_err(|e| NpuError::Other(format!("create_infer_request: {e:?}")))?;
        Ok(NpuGraph { _core: core, request, input_names, output_names, device: dev_str(&device) })
    }

    pub fn device(&self) -> &str {
        &self.device
    }
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Run one inference. `feeds` maps input names to tensors (order-independent);
    /// returns each output as `(name, shape, f32 data)` in graph order. A single-input
    /// graph accepts the feed regardless of name.
    pub fn run(&mut self, feeds: &[(&str, Feed)]) -> Result<Vec<NamedTensor>, NpuError> {
        for (name, feed) in feeds {
            let shape = Shape::new(feed.dims()).map_err(|e| NpuError::Other(format!("{e:?}")))?;
            let mut t = Tensor::new(feed.elem(), &shape).map_err(|e| NpuError::Other(format!("{e:?}")))?;
            match feed {
                Feed::F32(data, _) => t.get_data_mut::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?.copy_from_slice(data),
                Feed::I64(data, _) => t.get_data_mut::<i64>().map_err(|e| NpuError::Other(format!("{e:?}")))?.copy_from_slice(data),
            };
            if self.input_names.len() == 1 {
                self.request.set_input_tensor(&t).map_err(|e| NpuError::Other(format!("set input: {e:?}")))?;
            } else {
                self.request.set_tensor(name, &t).map_err(|e| NpuError::Other(format!("set {name}: {e:?}")))?;
            }
        }
        self.request.infer().map_err(|e| NpuError::Other(format!("infer: {e:?}")))?;
        let mut out = Vec::with_capacity(self.output_names.len());
        for (i, name) in self.output_names.iter().enumerate() {
            let t = self.request.get_output_tensor_by_index(i).map_err(|e| NpuError::Other(format!("get_output {i}: {e:?}")))?;
            let sh = t.get_shape().map_err(|e| NpuError::Other(format!("{e:?}")))?.get_dimensions().iter().map(|&x| x as usize).collect();
            let data = t.get_data::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?.to_vec();
            out.push((name.clone(), sh, data));
        }
        Ok(out)
    }
}

/// Shared prologue: new core + resolve/apply the configured device.
fn open_device(cfg: &NpuConfig) -> Result<(Core, DeviceType<'static>), NpuError> {
    let mut core = new_core()?;
    let avail: Vec<String> = core.available_devices().map_err(|e| NpuError::Other(format!("{e:?}")))?.iter().map(dev_str).collect();
    let device = resolve_device(cfg.device, &avail, cfg.allow_fallback)?;
    apply_properties(&mut core, &device, cfg);
    Ok((core, device))
}

/// A compiled graph with a single f32 input `inputs_embeds:[1,T,d_in]` and a
/// single f32 output `hidden:[1,T,d_out]` — the Qwen3-TTS Talker hidden-state
/// graph. Like [`DecoderSession`] it is a cache-free fixed-length prefill, but
/// driven by an input-embedding stream rather than token ids: the autoregressive
/// loop pads the real context to `T` and reads the hidden row at the last real
/// position (causal masking makes it independent of the zero padding).
pub struct EmbedSession {
    graph: NpuGraph,
    seq_len: usize,
    d_in: usize,
    d_out: usize,
}

impl EmbedSession {
    /// Compile from a file path (required for ONNX external data — the sidecar is
    /// resolved relative to the model dir).
    pub fn load_path(onnx_path: &Path, cfg: &NpuConfig) -> Result<Self, NpuError> {
        let (mut core, device) = open_device(cfg)?;
        let path_str = onnx_path.to_str().ok_or_else(|| NpuError::Other("non-utf8 path".into()))?;
        let model = core
            .read_model_from_file(path_str, "")
            .map_err(|e| NpuError::Other(format!("read_model {}: {e:?}", onnx_path.display())))?;
        Self::compile(core, model, device)
    }

    /// Compile from ONNX bytes (no external data).
    pub fn load_bytes(bytes: &[u8], cfg: &NpuConfig) -> Result<Self, NpuError> {
        let (mut core, device) = open_device(cfg)?;
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
        let graph = NpuGraph::from_compiled(core, compiled, device)?;
        Ok(EmbedSession { graph, seq_len, d_in, d_out })
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
        self.graph.device()
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
        let dims = vec![1, self.seq_len as i64, self.d_in as i64];
        let out = self.graph.run(&[("inputs_embeds", Feed::F32(embeds, dims))])?;
        Ok(out.into_iter().next().map(|(_, _, data)| data).unwrap_or_default())
    }
}

/// The Chronos-2 transformer core as a compiled graph: two f32 inputs
/// `emb:[1,S,D]` + `kmask:[1,1,1,S]` (additive) → f32 `qhead:[1,n_out,head_out]`.
/// Single whole-graph inference (not autoregressive); the host does the
/// scaler/patch/embed/REG assembly and the head rearrange/denorm around it. A
/// graph is compiled per `(S, n_out)`.
pub struct Chronos2Session {
    graph: NpuGraph,
    s: usize,
    d: usize,
    n_out: usize,
    head_out: usize,
}

impl Chronos2Session {
    /// Compile the core ONNX (e.g. from [`crate::chronos2_export::export_onnx`]).
    pub fn load_bytes(bytes: &[u8], cfg: &NpuConfig) -> Result<Self, NpuError> {
        let (mut core, device) = open_device(cfg)?;
        let model = core
            .read_model_from_buffer(bytes, None)
            .map_err(|e| NpuError::Other(format!("read_model (ONNX): {e:?}")))?;
        Self::compile(core, model, device)
    }

    fn compile(mut core: Core, model: openvino::Model, device: DeviceType<'static>) -> Result<Self, NpuError> {
        let compiled = core
            .compile_model(&model, device.to_owned())
            .map_err(|e| NpuError::Other(format!("compile_model on {}: {e:?}", dev_str(&device))))?;
        // input 0 = emb [1, S, D]
        let ie = compiled
            .get_input_by_index(0)
            .and_then(|n| n.get_shape())
            .map_err(|e| NpuError::Other(format!("emb input shape: {e:?}")))?;
        let ed = ie.get_dimensions();
        if ed.len() != 3 {
            return Err(NpuError::Other(format!("expected emb [1,S,D], got {ed:?}")));
        }
        let (s, d) = (ed[1] as usize, ed[2] as usize);
        // output 0 = qhead [1, n_out, head_out]
        let od = compiled
            .get_output_by_index(0)
            .and_then(|n| n.get_shape())
            .map_err(|e| NpuError::Other(format!("qhead output shape: {e:?}")))?;
        let odd = od.get_dimensions();
        let (n_out, head_out) = (odd[1] as usize, odd[2] as usize);
        let graph = NpuGraph::from_compiled(core, compiled, device)?;
        Ok(Chronos2Session { graph, s, d, n_out, head_out })
    }

    pub fn device(&self) -> &str {
        self.graph.device()
    }
    pub fn seq_len(&self) -> usize {
        self.s
    }
    pub fn n_out(&self) -> usize {
        self.n_out
    }
    /// Width of one head row, i.e. `run`'s output is `n_out * head_out` f32.
    pub fn head_out(&self) -> usize {
        self.head_out
    }

    /// Run the core: `emb` is `S*D` f32, `kmask` is `S` f32 (additive per key).
    /// Returns the raw head `[n_out*head_out]`.
    pub fn run(&mut self, emb: &[f32], kmask: &[f32]) -> Result<Vec<f32>, NpuError> {
        if emb.len() != self.s * self.d {
            return Err(NpuError::Other(format!("emb: expected {} f32, got {}", self.s * self.d, emb.len())));
        }
        if kmask.len() != self.s {
            return Err(NpuError::Other(format!("kmask: expected {} f32, got {}", self.s, kmask.len())));
        }
        let emb_dims = vec![1, self.s as i64, self.d as i64];
        let km_dims = vec![1, 1, 1, self.s as i64];
        let out = self.graph.run(&[("emb", Feed::F32(emb, emb_dims)), ("kmask", Feed::F32(kmask, km_dims))])?;
        Ok(out.into_iter().next().map(|(_, _, data)| data).unwrap_or_default())
    }
}

/// The LFM2.5-Encoder graph: inputs `ids:[1,S]` (i64 token ids) +
/// `kmask:[1,1,1,S]` (additive key-padding mask; zeros for no padding) →
/// output `hidden:[1,S,D]` (post embedding_norm). The tied MLM head runs on
/// host over probe rows. Mirrors [`Chronos2Session`] with an i64 ids input.
pub struct LfmSession {
    graph: NpuGraph,
    s: usize,
    d: usize,
}

impl LfmSession {
    /// Compile in-memory ONNX bytes (small/test graphs; real checkpoints use
    /// [`Self::load_path`] — external-data models must load from a file).
    pub fn load_bytes(bytes: &[u8], cfg: &NpuConfig) -> Result<Self, NpuError> {
        let (mut core, device) = open_device(cfg)?;
        let model = core
            .read_model_from_buffer(bytes, None)
            .map_err(|e| NpuError::Other(format!("read_model (ONNX): {e:?}")))?;
        Self::compile(core, model, device)
    }

    /// Compile from a model file (required for `finish_external` exports).
    pub fn load_path(path: &str, cfg: &NpuConfig) -> Result<Self, NpuError> {
        let (mut core, device) = open_device(cfg)?;
        let model = core
            .read_model_from_file(path, "")
            .map_err(|e| NpuError::Other(format!("read_model {path}: {e:?}")))?;
        Self::compile(core, model, device)
    }

    fn compile(mut core: Core, model: openvino::Model, device: DeviceType<'static>) -> Result<Self, NpuError> {
        let compiled = core
            .compile_model(&model, device.to_owned())
            .map_err(|e| NpuError::Other(format!("compile_model on {}: {e:?}", dev_str(&device))))?;
        let od = compiled
            .get_output_by_index(0)
            .and_then(|n| n.get_shape())
            .map_err(|e| NpuError::Other(format!("hidden output shape: {e:?}")))?;
        let odd = od.get_dimensions();
        if odd.len() != 3 {
            return Err(NpuError::Other(format!("expected hidden [1,S,D], got {odd:?}")));
        }
        let (s, d) = (odd[1] as usize, odd[2] as usize);
        let graph = NpuGraph::from_compiled(core, compiled, device)?;
        Ok(LfmSession { graph, s, d })
    }

    pub fn device(&self) -> &str {
        self.graph.device()
    }
    pub fn seq_len(&self) -> usize {
        self.s
    }
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Run the encoder: `ids` are exactly `S` token ids; `kmask` is `S` additive
    /// per-key floats (0 keeps, large-negative masks). Returns `[S*D]` hidden.
    pub fn run(&mut self, ids: &[i64], kmask: &[f32]) -> Result<Vec<f32>, NpuError> {
        if ids.len() != self.s {
            return Err(NpuError::Other(format!("ids: expected {} tokens, got {}", self.s, ids.len())));
        }
        if kmask.len() != self.s {
            return Err(NpuError::Other(format!("kmask: expected {} f32, got {}", self.s, kmask.len())));
        }
        let id_dims = vec![1, self.s as i64];
        let km_dims = vec![1, 1, 1, self.s as i64];
        let out = self.graph.run(&[("ids", Feed::I64(ids, id_dims)), ("kmask", Feed::F32(kmask, km_dims))])?;
        Ok(out.into_iter().next().map(|(_, _, data)| data).unwrap_or_default())
    }
}

/// The FinCast transformer core graph: inputs `emb:[1,S,D]` (assembled patch
/// tokens) + `amask:[1,1,S,S]` (additive causal + padding mask) → output
/// `qhead:[1,S,head_out]`. Mirrors [`Chronos2Session`] with a full `[S,S]` mask
/// (FinCast is causal) and an in-graph top-2 MoE. The host does the patch
/// embed/freq and the head rearrange/denorm.
pub struct FincastSession {
    graph: NpuGraph,
    s: usize,
    d: usize,
    head_out: usize,
}

impl FincastSession {
    /// Compile the core ONNX (from [`crate::fincast_export::export_onnx`]).
    pub fn load_bytes(bytes: &[u8], cfg: &NpuConfig) -> Result<Self, NpuError> {
        let (mut core, device) = open_device(cfg)?;
        let model = core
            .read_model_from_buffer(bytes, None)
            .map_err(|e| NpuError::Other(format!("read_model (ONNX): {e:?}")))?;
        Self::compile(core, model, device)
    }

    /// Compile from a model file — required for external-data
    /// (`finish_external`) exports. The full ~1B-param FinCast core's single-
    /// protobuf ONNX exceeds protobuf's 2 GB read-from-buffer limit, so the NPU
    /// path exports with a `.data` sidecar and loads it here (mirrors
    /// [`LfmSession::load_path`]).
    pub fn load_path(path: &str, cfg: &NpuConfig) -> Result<Self, NpuError> {
        let (mut core, device) = open_device(cfg)?;
        let model = core
            .read_model_from_file(path, "")
            .map_err(|e| NpuError::Other(format!("read_model {path}: {e:?}")))?;
        Self::compile(core, model, device)
    }

    fn compile(mut core: Core, model: openvino::Model, device: DeviceType<'static>) -> Result<Self, NpuError> {
        let compiled = core
            .compile_model(&model, device.to_owned())
            .map_err(|e| NpuError::Other(format!("compile_model on {}: {e:?}", dev_str(&device))))?;
        let ie = compiled
            .get_input_by_index(0)
            .and_then(|n| n.get_shape())
            .map_err(|e| NpuError::Other(format!("emb input shape: {e:?}")))?;
        let ed = ie.get_dimensions();
        if ed.len() != 3 {
            return Err(NpuError::Other(format!("expected emb [1,S,D], got {ed:?}")));
        }
        let (s, d) = (ed[1] as usize, ed[2] as usize);
        let od = compiled
            .get_output_by_index(0)
            .and_then(|n| n.get_shape())
            .map_err(|e| NpuError::Other(format!("qhead output shape: {e:?}")))?;
        let odd = od.get_dimensions();
        let head_out = odd[2] as usize;
        let graph = NpuGraph::from_compiled(core, compiled, device)?;
        Ok(FincastSession { graph, s, d, head_out })
    }

    pub fn device(&self) -> &str {
        self.graph.device()
    }
    pub fn seq_len(&self) -> usize {
        self.s
    }
    pub fn head_out(&self) -> usize {
        self.head_out
    }

    /// Run the core: `emb` is `S*D` f32, `amask` is `S*S` f32 (additive). Returns
    /// the raw head `[S*head_out]`.
    pub fn run(&mut self, emb: &[f32], amask: &[f32]) -> Result<Vec<f32>, NpuError> {
        if emb.len() != self.s * self.d {
            return Err(NpuError::Other(format!("emb: expected {} f32, got {}", self.s * self.d, emb.len())));
        }
        if amask.len() != self.s * self.s {
            return Err(NpuError::Other(format!("amask: expected {} f32, got {}", self.s * self.s, amask.len())));
        }
        let emb_dims = vec![1, self.s as i64, self.d as i64];
        let am_dims = vec![1, 1, self.s as i64, self.s as i64];
        let out = self.graph.run(&[("emb", Feed::F32(emb, emb_dims)), ("amask", Feed::F32(amask, am_dims))])?;
        Ok(out.into_iter().next().map(|(_, _, data)| data).unwrap_or_default())
    }
}

/// The Kronos `decode_s1` core graph: input `x:[1,T,D]` (host token-embedding) →
/// two outputs `ctx:[1,T,D]` + `s1_logits:[1,T,s1_vocab]`. One AR step of the
/// s1 head; the host embeds tokens, samples the last position, and slides.
pub struct KronosS1Session {
    graph: NpuGraph,
    t: usize,
    d: usize,
    s1v: usize,
    ctx_idx: usize,
    s1_idx: usize,
}

impl KronosS1Session {
    pub fn load_bytes(bytes: &[u8], cfg: &NpuConfig) -> Result<Self, NpuError> {
        let (mut core, device) = open_device(cfg)?;
        let model = core
            .read_model_from_buffer(bytes, None)
            .map_err(|e| NpuError::Other(format!("read_model (ONNX): {e:?}")))?;
        Self::compile(core, model, device)
    }

    fn compile(mut core: Core, model: openvino::Model, device: DeviceType<'static>) -> Result<Self, NpuError> {
        let compiled = core
            .compile_model(&model, device.to_owned())
            .map_err(|e| NpuError::Other(format!("compile_model on {}: {e:?}", dev_str(&device))))?;
        let ix = compiled
            .get_input_by_index(0)
            .and_then(|n| n.get_shape())
            .map_err(|e| NpuError::Other(format!("x input shape: {e:?}")))?;
        let ixd = ix.get_dimensions();
        let (t, d) = (ixd[1] as usize, ixd[2] as usize);
        // two outputs, order ctx/s1_logits: disambiguate by last dim (== d → ctx).
        let last = |i| -> Result<usize, NpuError> {
            let s = compiled
                .get_output_by_index(i)
                .and_then(|n| n.get_shape())
                .map_err(|e| NpuError::Other(format!("output {i} shape: {e:?}")))?;
            Ok(*s.get_dimensions().last().unwrap() as usize)
        };
        let (l0, l1) = (last(0)?, last(1)?);
        let (ctx_idx, s1_idx, s1v) = if l0 == d { (0, 1, l1) } else { (1, 0, l0) };
        let graph = NpuGraph::from_compiled(core, compiled, device)?;
        Ok(KronosS1Session { graph, t, d, s1v, ctx_idx, s1_idx })
    }

    pub fn device(&self) -> &str {
        self.graph.device()
    }
    pub fn seq_len(&self) -> usize {
        self.t
    }
    pub fn s1_vocab(&self) -> usize {
        self.s1v
    }

    /// Run one s1 step over the host embedding `x` (`T*D` f32). Returns
    /// `(ctx [T*D], s1_logits [T*s1_vocab])`.
    pub fn run(&mut self, x: &[f32]) -> Result<(Vec<f32>, Vec<f32>), NpuError> {
        if x.len() != self.t * self.d {
            return Err(NpuError::Other(format!("x: expected {} f32, got {}", self.t * self.d, x.len())));
        }
        let dims = vec![1, self.t as i64, self.d as i64];
        let mut out = self.graph.run(&[("x", Feed::F32(x, dims))])?;
        // ctx_idx/s1_idx index into the same graph-output order NpuGraph::run
        // returns (computed from the compiled model at load time above).
        let s1 = std::mem::take(&mut out[self.s1_idx].2);
        let ctx = std::mem::take(&mut out[self.ctx_idx].2);
        Ok((ctx, s1))
    }
}

/// The Kronos `decode_s2` dependency graph: inputs `ctx:[1,T,D]` + `sib:[1,T,D]`
/// (RAW `emb_s1` of the sampled s1) → `s2_logits:[1,T,s2_vocab]`.
pub struct KronosS2Session {
    graph: NpuGraph,
    t: usize,
    d: usize,
    s2v: usize,
}

impl KronosS2Session {
    pub fn load_bytes(bytes: &[u8], cfg: &NpuConfig) -> Result<Self, NpuError> {
        let (mut core, device) = open_device(cfg)?;
        let model = core
            .read_model_from_buffer(bytes, None)
            .map_err(|e| NpuError::Other(format!("read_model (ONNX): {e:?}")))?;
        Self::compile(core, model, device)
    }

    fn compile(mut core: Core, model: openvino::Model, device: DeviceType<'static>) -> Result<Self, NpuError> {
        let compiled = core
            .compile_model(&model, device.to_owned())
            .map_err(|e| NpuError::Other(format!("compile_model on {}: {e:?}", dev_str(&device))))?;
        let ic = compiled
            .get_input_by_index(0)
            .and_then(|n| n.get_shape())
            .map_err(|e| NpuError::Other(format!("ctx input shape: {e:?}")))?;
        let icd = ic.get_dimensions();
        let (t, d) = (icd[1] as usize, icd[2] as usize);
        let od = compiled
            .get_output_by_index(0)
            .and_then(|n| n.get_shape())
            .map_err(|e| NpuError::Other(format!("s2_logits output shape: {e:?}")))?;
        let s2v = *od.get_dimensions().last().unwrap() as usize;
        let graph = NpuGraph::from_compiled(core, compiled, device)?;
        Ok(KronosS2Session { graph, t, d, s2v })
    }

    pub fn device(&self) -> &str {
        self.graph.device()
    }
    pub fn seq_len(&self) -> usize {
        self.t
    }
    pub fn s2_vocab(&self) -> usize {
        self.s2v
    }

    /// Run one s2 step: `ctx` (`T*D`) from the s1 session + `sib` (`T*D`) the RAW
    /// s1-sibling embedding. Returns `s2_logits [T*s2_vocab]`.
    pub fn run(&mut self, ctx: &[f32], sib: &[f32]) -> Result<Vec<f32>, NpuError> {
        if ctx.len() != self.t * self.d || sib.len() != self.t * self.d {
            return Err(NpuError::Other(format!("ctx/sib: expected {} f32 each", self.t * self.d)));
        }
        let dims = vec![1, self.t as i64, self.d as i64];
        let out = self.graph.run(&[("ctx", Feed::F32(ctx, dims.clone())), ("sib", Feed::F32(sib, dims))])?;
        Ok(out.into_iter().next().map(|(_, _, data)| data).unwrap_or_default())
    }
}

/// A compiled codec-decoder graph: int64 `codes:[nq,T]` (codebook-major) ->
/// f32 `waveform:[1,1,L]`. Single whole-graph inference (no autoregression).
pub struct CodecSession {
    graph: NpuGraph,
    nq: usize,
    code_len: usize,
    out_len: usize,
}

impl CodecSession {
    pub fn load_path(onnx_path: &Path, cfg: &NpuConfig) -> Result<Self, NpuError> {
        let (mut core, device) = open_device(cfg)?;
        let path_str = onnx_path.to_str().ok_or_else(|| NpuError::Other("non-utf8 path".into()))?;
        let model = core
            .read_model_from_file(path_str, "")
            .map_err(|e| NpuError::Other(format!("read_model {}: {e:?}", onnx_path.display())))?;
        Self::compile(core, model, device)
    }

    pub fn load_bytes(bytes: &[u8], cfg: &NpuConfig) -> Result<Self, NpuError> {
        let (mut core, device) = open_device(cfg)?;
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
        let graph = NpuGraph::from_compiled(core, compiled, device)?;
        Ok(CodecSession { graph, nq, code_len, out_len })
    }

    pub fn nq(&self) -> usize {
        self.nq
    }
    pub fn code_len(&self) -> usize {
        self.code_len
    }
    pub fn device(&self) -> &str {
        self.graph.device()
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
        let dims = vec![self.nq as i64, self.code_len as i64];
        let out = self.graph.run(&[("codes", Feed::I64(codes, dims))])?;
        Ok(out.into_iter().next().map(|(_, _, data)| data).unwrap_or_default())
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
    /// Timing of the most recent [`run_step`](Self::run_step), split so a
    /// caller can tell host<->device marshalling apart from device compute -
    /// both scale with `cap` (the past K/V upload is `O(n_layers*nkv*cap*hd)`
    /// every step) and a wall-clock total alone cannot distinguish "the step
    /// got slower" from "marshalling now dominates the step".
    last_marshal_ms: f64,
    last_infer_ms: f64,
    last_readback_ms: f64,
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
            last_marshal_ms: 0.0,
            last_infer_ms: 0.0,
            last_readback_ms: 0.0,
        })
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    /// Host->device tensor upload time (`set_tensor` for x/rope/mask/past-K/V)
    /// of the most recent [`run_step`](Self::run_step), in ms.
    pub fn last_marshal_ms(&self) -> f64 {
        self.last_marshal_ms
    }

    /// Device compute time (`InferRequest::infer`) of the most recent
    /// [`run_step`](Self::run_step), in ms.
    pub fn last_infer_ms(&self) -> f64 {
        self.last_infer_ms
    }

    /// Device->host tensor download time (`get_tensor` for hidden/new-K/V) of
    /// the most recent [`run_step`](Self::run_step), in ms.
    pub fn last_readback_ms(&self) -> f64 {
        self.last_readback_ms
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
        let t_marshal = Instant::now();
        let mut holders: Vec<Tensor> = Vec::with_capacity(4 + 2 * nl);
        self.set_f32(&mut holders, "x", &[1, 1, d], x)?;
        self.set_f32(&mut holders, "rope_cos", &[1, 1, 1, hd], cos)?;
        self.set_f32(&mut holders, "rope_sin", &[1, 1, 1, hd], sin)?;
        self.set_f32(&mut holders, "past_mask", &[1, 1, 1, cap], mask)?;
        for l in 0..nl {
            self.set_f32(&mut holders, &format!("past_k_{l}"), &[1, nkv, cap, hd], &past_k[l])?;
            self.set_f32(&mut holders, &format!("past_v_{l}"), &[1, nkv, cap, hd], &past_v[l])?;
        }
        self.last_marshal_ms = t_marshal.elapsed().as_secs_f64() * 1e3;

        let t_infer = Instant::now();
        self.request.infer().map_err(|e| NpuError::Other(format!("infer: {e:?}")))?;
        self.last_infer_ms = t_infer.elapsed().as_secs_f64() * 1e3;

        let t_readback = Instant::now();
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
        self.last_readback_ms = t_readback.elapsed().as_secs_f64() * 1e3;
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

/// A compiled **fused MTP** graph (see [`crate::qwen_topology::build_mtp_fused_graph`]):
/// `talker_hidden:[1,1,emb]` + `cb0_embed:[1,1,emb]` -> `codes:[1,1,nres]` (f32,
/// argmax ids cast to float) + `res_sum:[1,1,emb]`. One inference does all 15
/// residual substeps on-device (no per-substep host round-trip).
pub struct FusedMtpSession {
    _core: Core,
    request: openvino::InferRequest,
    emb: usize,
    nres: usize,
    device: String,
}

impl FusedMtpSession {
    pub fn load_path(onnx_path: &Path, cfg: &NpuConfig, emb: usize, nres: usize) -> Result<Self, NpuError> {
        let mut core = new_core()?;
        let device = pick_device(&mut core, cfg)?;
        let path_str = onnx_path.to_str().ok_or_else(|| NpuError::Other("non-utf8 path".into()))?;
        let model = core
            .read_model_from_file(path_str, "")
            .map_err(|e| NpuError::Other(format!("read_model {}: {e:?}", onnx_path.display())))?;
        let mut compiled = core
            .compile_model(&model, device.to_owned())
            .map_err(|e| NpuError::Other(format!("compile_model on {}: {e:?}", dev_str(&device))))?;
        let request = compiled
            .create_infer_request()
            .map_err(|e| NpuError::Other(format!("create_infer_request: {e:?}")))?;
        Ok(FusedMtpSession { _core: core, request, emb, nres, device: dev_str(&device) })
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    /// Run the whole per-frame residual prediction in one inference. Returns
    /// `(codes[nres] as u32, res_sum[emb])`.
    pub fn run(&mut self, talker_hidden: &[f32], cb0_embed: &[f32]) -> Result<(Vec<u32>, Vec<f32>), NpuError> {
        let e = self.emb as i64;
        let set = |req: &mut openvino::InferRequest, name: &str, data: &[f32]| -> Result<(), NpuError> {
            let shape = Shape::new(&[1, 1, e]).map_err(|er| NpuError::Other(format!("{er:?}")))?;
            let mut t = Tensor::new(ElementType::F32, &shape).map_err(|er| NpuError::Other(format!("{er:?}")))?;
            t.get_data_mut::<f32>().map_err(|er| NpuError::Other(format!("{er:?}")))?.copy_from_slice(data);
            req.set_tensor(name, &t).map_err(|er| NpuError::Other(format!("set {name}: {er:?}")))
        };
        set(&mut self.request, "talker_hidden", talker_hidden)?;
        set(&mut self.request, "cb0_embed", cb0_embed)?;
        self.request.infer().map_err(|e| NpuError::Other(format!("infer: {e:?}")))?;
        let get = |req: &openvino::InferRequest, name: &str| -> Result<Vec<f32>, NpuError> {
            let t = req.get_tensor(name).map_err(|e| NpuError::Other(format!("get {name}: {e:?}")))?;
            Ok(t.get_data::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?.to_vec())
        };
        let codes_f = get(&self.request, "codes")?;
        let res_sum = get(&self.request, "res_sum")?;
        let codes = codes_f.iter().take(self.nres).map(|&x| x.round().max(0.0) as u32).collect();
        Ok((codes, res_sum))
    }
}

/// A compiled **streaming-back** codec graph (see
/// [`crate::codec_topology::build_codec_back_stream_graph`]): `latent` chunk +
/// per-conv `bufin.{prefix}` in, `waveform` chunk + per-conv `bufout.{prefix}`
/// out. The host carries the buffers across chunks so each chunk decodes only its
/// new frames. Buffer specs `(prefix, channels, width)` are supplied by the
/// exporter.
pub struct BackStreamSession {
    _core: Core,
    request: openvino::InferRequest,
    bufs: Vec<(String, i64, i64)>,
    latent_dim: usize,
    chunk: usize,
    device: String,
}

impl BackStreamSession {
    pub fn load_path(
        onnx_path: &Path,
        cfg: &NpuConfig,
        bufs: Vec<(String, i64, i64)>,
        latent_dim: usize,
        chunk: usize,
    ) -> Result<Self, NpuError> {
        let mut core = new_core()?;
        let device = pick_device(&mut core, cfg)?;
        let path_str = onnx_path.to_str().ok_or_else(|| NpuError::Other("non-utf8 path".into()))?;
        let model = core
            .read_model_from_file(path_str, "")
            .map_err(|e| NpuError::Other(format!("read_model {}: {e:?}", onnx_path.display())))?;
        let mut compiled = core
            .compile_model(&model, device.to_owned())
            .map_err(|e| NpuError::Other(format!("compile_model on {}: {e:?}", dev_str(&device))))?;
        let request = compiled
            .create_infer_request()
            .map_err(|e| NpuError::Other(format!("create_infer_request: {e:?}")))?;
        Ok(BackStreamSession { _core: core, request, bufs, latent_dim, chunk, device: dev_str(&device) })
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    /// Zeroed initial buffer state, one Vec per conv (`channels*width` f32).
    pub fn zero_buffers(&self) -> Vec<Vec<f32>> {
        self.bufs.iter().map(|&(_, c, w)| vec![0.0f32; (c * w) as usize]).collect()
    }

    fn set_f32(&mut self, holders: &mut Vec<Tensor>, name: &str, dims: &[i64], data: &[f32]) -> Result<(), NpuError> {
        let shape = Shape::new(dims).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        let mut t = Tensor::new(ElementType::F32, &shape).map_err(|e| NpuError::Other(format!("{e:?}")))?;
        t.get_data_mut::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?.copy_from_slice(data);
        self.request.set_tensor(name, &t).map_err(|e| NpuError::Other(format!("set {name}: {e:?}")))?;
        holders.push(t);
        Ok(())
    }

    /// Decode one chunk: `latent` is `[latent_dim*chunk]` (NCL), `bufins` are the
    /// current per-conv buffers. Returns `(waveform, updated buffers)`.
    pub fn run(&mut self, latent: &[f32], bufins: &[Vec<f32>]) -> Result<(Vec<f32>, Vec<Vec<f32>>), NpuError> {
        let (ld, ch) = (self.latent_dim as i64, self.chunk as i64);
        let mut holders: Vec<Tensor> = Vec::with_capacity(1 + self.bufs.len());
        self.set_f32(&mut holders, "latent", &[1, ld, ch], latent)?;
        let specs: Vec<(String, i64, i64)> = self.bufs.clone();
        for (i, (prefix, c, w)) in specs.iter().enumerate() {
            self.set_f32(&mut holders, &format!("bufin.{prefix}"), &[1, *c, *w], &bufins[i])?;
        }
        self.request.infer().map_err(|e| NpuError::Other(format!("infer: {e:?}")))?;
        let get = |req: &openvino::InferRequest, name: &str| -> Result<Vec<f32>, NpuError> {
            let t = req.get_tensor(name).map_err(|e| NpuError::Other(format!("get {name}: {e:?}")))?;
            Ok(t.get_data::<f32>().map_err(|e| NpuError::Other(format!("{e:?}")))?.to_vec())
        };
        let wav = get(&self.request, "waveform")?;
        let mut bufouts = Vec::with_capacity(specs.len());
        for (prefix, _, _) in &specs {
            bufouts.push(get(&self.request, &format!("bufout.{prefix}"))?);
        }
        drop(holders);
        Ok((wav, bufouts))
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
    // Shared latency stats (`perf::stats::Dist`) instead of a locally
    // reimplemented sort+index percentile.
    let mut d = perf::stats::Dist::from_millis(samples);
    let n = d.len();
    Ok(BenchResult {
        device: session.device().to_string(),
        iters: n,
        p50_ms: d.percentile(0.50).unwrap_or(0.0),
        p99_ms: d.percentile(0.99).unwrap_or(0.0),
        mean_ms: d.mean().unwrap_or(0.0),
        throughput_fps: n as f64 / wall,
    })
}
