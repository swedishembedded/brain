// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for getting trained detectors onto
// edge hardware with verified numerics for its clients. If your team needs
// expertise in model conversion and checkpoint validation then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Checkpoint loading on the host: a brain-format `.safetensors` file becomes a
//! [`Checkpoint`] (config, [`Graph`], fp32 [`Weights`]) with no device involved.
//! This is the entry point a board runtime uses.
//!
//! ## Format
//!
//! A plain safetensors file whose `brain.config` metadata is a
//! [`Yolov5Config`] and whose tensors are exactly [`Graph::param_list`]: for
//! each convolution `<name>.weight` `[cout, cin, k, k]` and, if it has one,
//! `<name>.bias` `[cout]`, fp32, batch norm already folded into the convolution.
//! `tools/yolov5_export/export_yolov5.py` produces it from an ONNX export.
//!
//! Loading validates both directions and reports every problem at once: a
//! tensor the graph needs but the file lacks, a tensor the file carries that the
//! graph does not use, and any shape that differs. Nothing is zero-filled or
//! skipped.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use checkpoint::mmap::MmapSafetensors;

use crate::config::Yolov5Config;
use crate::error::Error;
use crate::graph::{ConvOp, Graph};

/// Elements per chunk when streaming a tensor out of the mapped file.
const READ_CHUNK_ELEMS: usize = 1 << 20;

/// fp32 weights and biases, validated against a [`Graph`]'s manifest.
#[derive(Clone, Debug)]
pub struct Weights {
    tensors: HashMap<String, Vec<f32>>,
}

impl Weights {
    /// Validate `tensors` (name to flat row-major data) against `graph`'s
    /// manifest by name and element count.
    pub fn new(graph: &Graph, tensors: HashMap<String, Vec<f32>>) -> Result<Weights, Error> {
        let mut problems = Vec::new();
        let expected = graph.param_list();
        for (name, shape) in &expected {
            let want: usize = shape.iter().product();
            match tensors.get(name) {
                None => problems.push(format!("missing {name} {shape:?}")),
                Some(t) if t.len() != want => problems.push(format!("{name}: {} elements, expected {want} {shape:?}", t.len())),
                Some(_) => {}
            }
        }
        let known: BTreeSet<&str> = expected.iter().map(|(n, _)| n.as_str()).collect();
        let mut extra: Vec<&String> = tensors.keys().filter(|k| !known.contains(k.as_str())).collect();
        extra.sort();
        problems.extend(extra.into_iter().map(|n| format!("unexpected tensor {n}")));
        if problems.is_empty() {
            Ok(Weights { tensors })
        } else {
            Err(Error::Checkpoint(problems))
        }
    }

    /// `conv`'s weight `[cout, cin, k, k]` and, when it has one, its bias.
    pub fn conv(&self, conv: &ConvOp) -> (&[f32], Option<&[f32]>) {
        let w = &self.tensors[&conv.weight_name()];
        let b = conv.bias.then(|| self.tensors[&conv.bias_name()].as_slice());
        (w, b)
    }

    /// Every tensor by name.
    pub fn tensors(&self) -> &HashMap<String, Vec<f32>> {
        &self.tensors
    }
}

/// A loaded checkpoint.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub config: Yolov5Config,
    pub graph: Graph,
    pub weights: Weights,
}

/// Read `path`'s config and check its tensor manifest (names, element counts,
/// shapes) against the graph that config implies, decoding no tensor data.
/// This is what classifies a file as a YOLOv5 checkpoint.
pub fn read_config(path: &Path) -> Result<Yolov5Config, Error> {
    let mm = open_mapped(path)?;
    let (cfg, _) = config_and_graph(&mm)?;
    Ok(cfg)
}

fn open_mapped(path: &Path) -> Result<MmapSafetensors, Error> {
    MmapSafetensors::open(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))
}

fn config_and_graph(mm: &MmapSafetensors) -> Result<(Yolov5Config, Graph), Error> {
    let cfg = Yolov5Config::from_json(&mm.config())?;
    let graph = Graph::build(&cfg);
    let mut problems = Vec::new();
    let expected = graph.param_list();
    for (name, shape) in &expected {
        match mm.shape(name) {
            None => problems.push(format!("missing {name} {shape:?}")),
            Some(got) if got != shape.as_slice() => problems.push(format!("{name}: shape {got:?}, expected {shape:?}")),
            Some(_) => {}
        }
    }
    let known: BTreeSet<&str> = expected.iter().map(|(n, _)| n.as_str()).collect();
    problems.extend(mm.names().iter().filter(|n| !known.contains(n.as_str())).map(|n| format!("unexpected tensor {n}")));
    if problems.is_empty() {
        Ok((cfg, graph))
    } else {
        Err(Error::Checkpoint(problems))
    }
}

/// Load a brain-format YOLOv5 checkpoint entirely on the host.
pub fn open(path: &Path) -> Result<Checkpoint, Error> {
    let mm = open_mapped(path)?;
    let (config, graph) = config_and_graph(&mm)?;
    let mut tensors = HashMap::new();
    for (name, shape) in graph.param_list() {
        let numel: usize = shape.iter().product();
        let mut data = Vec::with_capacity(numel);
        let found = mm.with_tensor_chunks(&name, READ_CHUNK_ELEMS, &mut |_, chunk| data.extend_from_slice(chunk));
        if !found || data.len() != numel {
            return Err(Error::Io(format!("{}: tensor {name} is unreadable as fp32 ({} of {numel} elements)", path.display(), data.len())));
        }
        tensors.insert(name, data);
    }
    let weights = Weights::new(&graph, tensors)?;
    Ok(Checkpoint { config, graph, weights })
}

/// A validated checkpoint whose tensors stay in the mapped file until asked
/// for: the form a board with a few MB of RAM can load (the NPU path reads
/// one layer at a time).
pub struct MappedCheckpoint {
    pub config: Yolov5Config,
    pub graph: Graph,
    pub file: Arc<MmapSafetensors>,
}

/// Open and validate a checkpoint (same checks as [`open`]) without reading
/// its tensors.
pub fn open_mapped_checkpoint(path: &Path) -> Result<MappedCheckpoint, Error> {
    let mm = open_mapped(path)?;
    let (config, graph) = config_and_graph(&mm)?;
    Ok(MappedCheckpoint { config, graph, file: Arc::new(mm) })
}

/// Write `weights` and `config` as a brain-format checkpoint (tests, and tools
/// that build a checkpoint without Python).
pub fn save(path: &Path, config: &Yolov5Config, graph: &Graph, weights: &Weights) -> Result<(), Error> {
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = graph
        .param_list()
        .into_iter()
        .map(|(name, shape)| {
            let data = weights.tensors[&name].clone();
            (name, shape.into_iter().map(|d| d as u64).collect(), data)
        })
        .collect();
    let path_str = path.to_str().ok_or_else(|| Error::Io(format!("{}: not valid UTF-8", path.display())))?;
    checkpoint::st::save_safetensors(path_str, &tensors, &config.to_json(), None).map_err(|e| Error::Io(format!("{}: {e}", path.display())))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::config::Activation;

    /// Small deterministic weights for the full manifest of `graph`.
    pub(crate) fn synthetic(graph: &Graph, seed: u64) -> Weights {
        let mut rng = data::rng::Lcg::new(seed);
        let tensors = graph
            .param_list()
            .into_iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                let fan_in = shape.iter().skip(1).product::<usize>().max(1) as f32;
                let scale = (2.0 / fan_in).sqrt();
                (name, (0..n).map(|_| rng.scaled(scale)).collect())
            })
            .collect();
        Weights::new(graph, tensors).unwrap()
    }

    fn tiny() -> (Yolov5Config, Graph) {
        let mut cfg = Yolov5Config::yolov5s(Activation::Relu);
        cfg.nc = 3;
        cfg.width_multiple = 0.125;
        let graph = Graph::build(&cfg);
        (cfg, graph)
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("brain-yolov5-weights-{tag}-{}.safetensors", std::process::id()))
    }

    #[test]
    fn a_saved_checkpoint_loads_back_bit_exact() {
        let (cfg, graph) = tiny();
        let w = synthetic(&graph, 5);
        let path = scratch("roundtrip");
        save(&path, &cfg, &graph, &w).unwrap();
        assert_eq!(read_config(&path).unwrap(), cfg);
        let loaded = open(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.config, cfg);
        assert_eq!(loaded.graph, graph);
        assert_eq!(loaded.weights.tensors(), w.tensors());
    }

    #[test]
    fn missing_extra_and_misshapen_tensors_are_all_reported_by_name() {
        let (_, graph) = tiny();
        let mut t = synthetic(&graph, 6).tensors().clone();
        let victim = graph.param_list()[3].0.clone();
        t.remove(&victim);
        let wrong = graph.param_list()[4].0.clone();
        t.get_mut(&wrong).unwrap().push(0.0);
        t.insert("model.99.extra.weight".to_string(), vec![0.0]);
        let Err(Error::Checkpoint(problems)) = Weights::new(&graph, t) else { panic!("must be rejected") };
        assert_eq!(problems.len(), 3, "{problems:?}");
        assert!(problems.iter().any(|p| p.contains(&victim) && p.starts_with("missing")));
        assert!(problems.iter().any(|p| p.contains(&wrong)));
        assert!(problems.iter().any(|p| p.contains("model.99.extra.weight")));
    }

    #[test]
    fn a_file_for_another_architecture_is_not_a_yolov5_checkpoint() {
        let path = scratch("foreign");
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &[("embedding.weight".to_string(), vec![4], vec![0.0; 4])], &serde_json::json!({"some_other_model": true}), None).unwrap();
        let err = read_config(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(matches!(err, Error::Config(_)), "{err:?}");
    }
}
