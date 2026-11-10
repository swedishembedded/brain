// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Strict import and export of the released `mirage.safetensors`.
//!
//! The header is inspected before a single tensor byte is read. The embedded
//! configuration is parsed, the canonical [`Manifest`] derived from it, and
//! every expected tensor must be present with its exact shape, dtype and byte
//! length; any missing or unexpected tensor is an error naming it. Nothing is
//! ever zero-filled or reshaped to fit. The file's metadata (`source_sha256`,
//! `step`, `format`, `config`) is kept and written back on export, so a
//! round trip reproduces the checkpoint exactly.
//!
//! Swedish Embedded AB implements solutions for strict, auditable model
//! checkpoint ingestion for its clients. If your team needs expertise in
//! supply-chain-safe model loading then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use checkpoint::mmap::MmapSafetensors;
use sha2::{Digest, Sha256};

use crate::config::{CheckpointConfig, ConfigError, MirageConfig};
use crate::spec::Manifest;

/// The `format` the metadata of a Mirage safetensors file declares.
pub const FORMAT: &str = "mirage";

/// File name of the weights inside a model directory.
pub const WEIGHTS_FILE: &str = "mirage.safetensors";

/// File name of the configuration inside a model directory.
pub const CONFIG_FILE: &str = "config.json";

/// Metadata keys a Mirage file may carry; any other key is refused.
const METADATA_KEYS: [&str; 4] = ["config", "source_sha256", "step", "format"];

/// A `config.json` larger than this is not a configuration.
const MAX_CONFIG_BYTES: u64 = 1 << 20;

/// The provenance a checkpoint carries in its safetensors header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Metadata {
    /// SHA-256 of the training checkpoint the file was exported from.
    pub source_sha256: String,
    /// Optimiser step of the weights.
    pub step: u64,
    /// Always [`FORMAT`].
    pub format: String,
}

/// One tensor in host memory, row-major F32.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Shape, matching the manifest.
    pub shape: Vec<usize>,
    /// Values.
    pub data: Vec<f32>,
}

/// The CQ matrices of a deployment archive in their stored (packed) form, by canonical tensor name and decoder layer
/// (`None` for a tensor that is not stacked). Each is `[out, in]`, the orientation of [`crate::vision::ops::Linear`];
/// the dequantised values are also in [`Checkpoint::tensors`], so the host paths read them unchanged and only a
/// device that was asked for the packed form keeps these.
pub type PackedMatrices = BTreeMap<(String, Option<usize>), Arc<model::cq::Cq>>;

/// A fully loaded checkpoint: configuration, provenance and every tensor.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// The geometry the tensors were validated against.
    pub config: MirageConfig,
    /// The file's provenance metadata.
    pub metadata: Metadata,
    /// Tensors by Flax name.
    pub tensors: BTreeMap<String, Tensor>,
    /// The quantised matrices as stored; empty for a checkpoint read from safetensors.
    pub packed: PackedMatrices,
}

/// What the loader can be asked to enforce.
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    /// Hex SHA-256 the file must hash to (pin the release being loaded).
    pub expected_sha256: Option<String>,
}

/// One way a tensor disagrees with the manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorProblem {
    /// The manifest requires it, the file lacks it.
    Missing { name: String },
    /// The file holds it, the manifest does not know it.
    Unexpected { name: String },
    /// The shape differs.
    Shape { name: String, expected: Vec<usize>, found: Vec<usize> },
    /// The dtype is not F32.
    Dtype { name: String, found: String },
    /// The byte length does not match the shape.
    Size { name: String, expected_bytes: usize, found_bytes: usize },
}

impl TensorProblem {
    /// The tensor concerned.
    pub fn name(&self) -> &str {
        match self {
            TensorProblem::Missing { name }
            | TensorProblem::Unexpected { name }
            | TensorProblem::Shape { name, .. }
            | TensorProblem::Dtype { name, .. }
            | TensorProblem::Size { name, .. } => name,
        }
    }
}

impl fmt::Display for TensorProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorProblem::Missing { name } => write!(f, "missing tensor `{name}`"),
            TensorProblem::Unexpected { name } => write!(f, "unexpected tensor `{name}`"),
            TensorProblem::Shape { name, expected, found } => write!(f, "tensor `{name}` has shape {found:?}, expected {expected:?}"),
            TensorProblem::Dtype { name, found } => write!(f, "tensor `{name}` has dtype {found}, expected F32"),
            TensorProblem::Size { name, expected_bytes, found_bytes } => {
                write!(f, "tensor `{name}` spans {found_bytes} bytes, its shape needs {expected_bytes}")
            }
        }
    }
}

/// Why an import or export failed.
#[derive(Debug)]
pub enum ImportError {
    /// A file could not be read or written.
    Io { path: PathBuf, message: String },
    /// The safetensors container is malformed (truncated, bad offsets, bad JSON).
    Header(String),
    /// A metadata entry is missing, unknown or invalid.
    Metadata { key: String, reason: String },
    /// A configuration was refused.
    Config(ConfigError),
    /// Tensors disagree with the manifest.
    Tensors(Vec<TensorProblem>),
    /// The file does not hash to the pinned digest.
    Digest { expected: String, found: String },
    /// `config.json` and the weights file describe different checkpoints.
    Sidecar { field: String, reason: String },
}

/// How many tensor problems a message lists before summarising.
const LISTED_PROBLEMS: usize = 8;

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImportError::Io { path, message } => write!(f, "mirage import: {}: {message}", path.display()),
            ImportError::Header(m) => write!(f, "mirage import: safetensors header: {m}"),
            ImportError::Metadata { key, reason } => write!(f, "mirage import: metadata `{key}`: {reason}"),
            ImportError::Config(e) => write!(f, "mirage import: embedded {e}"),
            ImportError::Tensors(problems) => {
                write!(f, "mirage import: {} tensor problem(s): ", problems.len())?;
                for (i, p) in problems.iter().take(LISTED_PROBLEMS).enumerate() {
                    write!(f, "{}{p}", if i == 0 { "" } else { "; " })?;
                }
                if problems.len() > LISTED_PROBLEMS {
                    write!(f, "; and {} more", problems.len() - LISTED_PROBLEMS)?;
                }
                Ok(())
            }
            ImportError::Digest { expected, found } => write!(f, "mirage import: sha256 is {found}, pinned {expected}"),
            ImportError::Sidecar { field, reason } => write!(f, "mirage import: {CONFIG_FILE} disagrees with {WEIGHTS_FILE} on `{field}`: {reason}"),
        }
    }
}

impl std::error::Error for ImportError {}

impl From<ConfigError> for ImportError {
    fn from(e: ConfigError) -> Self {
        ImportError::Config(e)
    }
}

fn io_error(path: &Path, e: impl fmt::Display) -> ImportError {
    ImportError::Io { path: path.to_path_buf(), message: e.to_string() }
}

/// A tensor as the file header describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderTensor {
    /// Flax name.
    pub name: String,
    /// Shape from the header.
    pub shape: Vec<usize>,
    /// Safetensors dtype string.
    pub dtype: String,
}

/// What the header alone establishes, after full validation against the manifest.
#[derive(Debug, Clone)]
pub struct Header {
    /// The embedded configuration.
    pub config: MirageConfig,
    /// The provenance metadata.
    pub metadata: Metadata,
    /// Every tensor, by name.
    pub tensors: Vec<HeaderTensor>,
}

/// Lower-case hex SHA-256 of a file, streamed.
pub fn file_sha256(path: &Path) -> Result<String, ImportError> {
    let mut file = std::fs::File::open(path).map_err(|e| io_error(path, e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).map_err(|e| io_error(path, e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn parse_metadata(raw: &BTreeMap<String, String>) -> Result<(MirageConfig, Metadata), ImportError> {
    for key in raw.keys() {
        if !METADATA_KEYS.contains(&key.as_str()) {
            return Err(ImportError::Metadata { key: key.clone(), reason: "unknown key".to_string() });
        }
    }
    let get = |key: &str| raw.get(key).ok_or_else(|| ImportError::Metadata { key: key.to_string(), reason: "missing".to_string() });
    let config = MirageConfig::from_json(get("config")?)?;
    let source_sha256 = get("source_sha256")?.clone();
    if !is_sha256_hex(&source_sha256) {
        return Err(ImportError::Metadata { key: "source_sha256".to_string(), reason: format!("`{source_sha256}` is not a 64-digit hexadecimal SHA-256") });
    }
    let step_text = get("step")?;
    let step = step_text
        .parse::<u64>()
        .map_err(|e| ImportError::Metadata { key: "step".to_string(), reason: format!("`{step_text}` is not a step count: {e}") })?;
    let format = get("format")?.clone();
    if format != FORMAT {
        return Err(ImportError::Metadata { key: "format".to_string(), reason: format!("`{format}` is not `{FORMAT}`") });
    }
    Ok((config, Metadata { source_sha256, step, format }))
}

/// Open the file and validate its header and metadata against the manifest
/// the embedded configuration implies. Reads no tensor data.
fn open_validated(path: &Path) -> Result<(MmapSafetensors, Header), ImportError> {
    let file = MmapSafetensors::open(path).map_err(ImportError::Header)?;
    let (config, metadata) = parse_metadata(file.metadata())?;
    let manifest = Manifest::new(&config);

    let mut problems = Vec::new();
    let mut tensors = Vec::with_capacity(file.names().len());
    let present: BTreeSet<&str> = file.names().iter().map(String::as_str).collect();
    for spec in manifest.iter() {
        if !present.contains(spec.name.as_str()) {
            problems.push(TensorProblem::Missing { name: spec.name.clone() });
        }
    }
    for name in file.names() {
        let (Some(shape), Some(dtype)) = (file.shape(name), file.dtype(name)) else {
            return Err(ImportError::Header(format!("tensor `{name}` vanished between names() and shape()")));
        };
        tensors.push(HeaderTensor { name: name.clone(), shape: shape.to_vec(), dtype: dtype.to_string() });
        let Some(spec) = manifest.get(name) else {
            problems.push(TensorProblem::Unexpected { name: name.clone() });
            continue;
        };
        if dtype != "F32" {
            problems.push(TensorProblem::Dtype { name: name.clone(), found: dtype.to_string() });
        } else if shape != spec.shape.as_slice() {
            problems.push(TensorProblem::Shape { name: name.clone(), expected: spec.shape.clone(), found: shape.to_vec() });
        } else {
            let found_bytes = file.nbytes(name).unwrap_or(0) as usize;
            if found_bytes != spec.numel() * 4 {
                problems.push(TensorProblem::Size { name: name.clone(), expected_bytes: spec.numel() * 4, found_bytes });
            }
        }
    }
    if !problems.is_empty() {
        return Err(ImportError::Tensors(problems));
    }
    Ok((file, Header { config, metadata, tensors }))
}

/// Validate a file's header, metadata and tensor inventory without reading
/// any tensor data.
pub fn inspect(path: &Path) -> Result<Header, ImportError> {
    open_validated(path).map(|(_, header)| header)
}

/// Load a checkpoint, validating everything the header can tell first.
pub fn load(path: &Path, options: &ImportOptions) -> Result<Checkpoint, ImportError> {
    let (file, header) = open_validated(path)?;
    if let Some(expected) = &options.expected_sha256 {
        let found = file_sha256(path)?;
        if !found.eq_ignore_ascii_case(expected) {
            return Err(ImportError::Digest { expected: expected.clone(), found });
        }
    }
    let mut tensors = BTreeMap::new();
    for t in &header.tensors {
        let mut data = Vec::with_capacity(t.shape.iter().product());
        // Chunked decode keeps the mapping's resident pages bounded while the
        // owned copy is built.
        let found = file.with_tensor_chunks(&t.name, 1 << 20, &mut |_, chunk| data.extend_from_slice(chunk));
        if !found {
            return Err(ImportError::Header(format!("tensor `{}` vanished between inspection and reading", t.name)));
        }
        tensors.insert(t.name.clone(), Tensor { shape: t.shape.clone(), data });
    }
    Ok(Checkpoint { config: header.config, metadata: header.metadata, tensors, packed: PackedMatrices::new() })
}

/// Load `mirage.safetensors` from a model directory and require that
/// `config.json` describes the same checkpoint.
pub fn load_dir(dir: &Path, options: &ImportOptions) -> Result<Checkpoint, ImportError> {
    let config_path = dir.join(CONFIG_FILE);
    let size = std::fs::metadata(&config_path).map_err(|e| io_error(&config_path, e))?.len();
    if size > MAX_CONFIG_BYTES {
        return Err(io_error(&config_path, format!("{size} bytes is too large for a configuration")));
    }
    let text = std::fs::read_to_string(&config_path).map_err(|e| io_error(&config_path, e))?;
    let sidecar = CheckpointConfig::from_json(&text)?;

    let ck = load(&dir.join(WEIGHTS_FILE), options)?;
    let disagree = |field: &str, reason: String| Err(ImportError::Sidecar { field: field.to_string(), reason });
    if sidecar.architecture != ck.config {
        return disagree("architecture", "differs from the configuration embedded in the weights".to_string());
    }
    if sidecar.training_step != ck.metadata.step {
        return disagree("training_step", format!("{} but the weights record step {}", sidecar.training_step, ck.metadata.step));
    }
    if !sidecar.source_checkpoint_sha256.eq_ignore_ascii_case(&ck.metadata.source_sha256) {
        return disagree("source_checkpoint_sha256", "differs from the weights' source_sha256".to_string());
    }
    Ok(ck)
}

impl Checkpoint {
    /// Check the tensors against the manifest the configuration implies.
    pub fn validate(&self) -> Result<(), ImportError> {
        self.config.validate()?;
        if self.metadata.format != FORMAT {
            return Err(ImportError::Metadata { key: "format".to_string(), reason: format!("`{}` is not `{FORMAT}`", self.metadata.format) });
        }
        if !is_sha256_hex(&self.metadata.source_sha256) {
            return Err(ImportError::Metadata { key: "source_sha256".to_string(), reason: "is not a 64-digit hexadecimal SHA-256".to_string() });
        }
        let manifest = Manifest::new(&self.config);
        let mut problems = Vec::new();
        for spec in manifest.iter() {
            match self.tensors.get(&spec.name) {
                None => problems.push(TensorProblem::Missing { name: spec.name.clone() }),
                Some(t) if t.shape != spec.shape => {
                    problems.push(TensorProblem::Shape { name: spec.name.clone(), expected: spec.shape.clone(), found: t.shape.clone() })
                }
                Some(t) if t.data.len() != spec.numel() => problems.push(TensorProblem::Size {
                    name: spec.name.clone(),
                    expected_bytes: spec.numel() * 4,
                    found_bytes: t.data.len() * 4,
                }),
                Some(_) => {}
            }
        }
        problems.extend(self.tensors.keys().filter(|n| manifest.get(n).is_none()).map(|n| TensorProblem::Unexpected { name: n.clone() }));
        if problems.is_empty() {
            Ok(())
        } else {
            Err(ImportError::Tensors(problems))
        }
    }
}

/// Write a checkpoint as a canonical safetensors file: the tensors are checked
/// against the manifest first, the metadata is written back verbatim, and the
/// header is key-sorted so equal checkpoints produce equal bytes.
pub fn export(ck: &Checkpoint, path: &Path) -> Result<(), ImportError> {
    ck.validate()?;
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
        ck.tensors.iter().map(|(name, t)| (name.clone(), t.shape.iter().map(|&d| d as u64).collect(), t.data.clone())).collect();
    let metadata = BTreeMap::from([
        ("config".to_string(), ck.config.to_json()),
        ("source_sha256".to_string(), ck.metadata.source_sha256.clone()),
        ("step".to_string(), ck.metadata.step.to_string()),
        ("format".to_string(), ck.metadata.format.clone()),
    ]);
    let path_str = path.to_str().ok_or_else(|| io_error(path, "path is not valid UTF-8"))?;
    checkpoint::st::save_safetensors_with_metadata(path_str, &tensors, metadata).map_err(|e| io_error(path, e))
}
