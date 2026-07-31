// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ONNX **import** front-end: pull weights out of a serialized `ModelProto`.
//!
//! The crate was export-only. Every brain model imported so far arrived as
//! safetensors or a torch archive, so nothing needed to read an ONNX file —
//! until `crates/facenet`, whose reference release (insightface `antelopev2`)
//! ships ONNX and *only* ONNX.
//!
//! This module is deliberately small and lives HERE rather than in the model
//! crate, for the AGENTS.md "one implementation" reason: a private protobuf
//! reader inside `facenet` would be a second decoder of the same wire format
//! that nothing compares against the first. The vendored [`crate::onnx`]
//! bindings already carry `dims`, `raw_data`, `float_data`, `data_location` and
//! the external-data key/value pairs; all that was missing was the extraction.
//!
//! What it does NOT do: build a [`crate::graph::Graph`] from a proto. Importing
//! a model into brain means *rewriting its topology as brain blocks*, and the
//! per-model graph walk that does that belongs to the model crate (a
//! `Graph::from_proto` would be a third representation nobody consumes). What is
//! shared — and therefore here — is decoding tensors and the node/attribute
//! accessors a graph walk needs.

use std::collections::HashMap;
use std::path::Path;

use crate::onnx::{tensor_proto::DataType, AttributeProto, GraphProto, ModelProto, TensorProto};

/// A decoded initializer: shape + values, always widened to f32.
///
/// f32 is not a lossy convenience here — it is brain's storage format for
/// imported weights everywhere (`checkpoint::safetensors::StTensor` is f32 too),
/// and every dtype this decodes (`FLOAT`, `INT32`, `INT64`, `INT8`, `UINT8`) is
/// exact in f32 for the magnitudes an ONNX initializer holds. A value that is
/// *not* exact (an int64 past 2^24, e.g. a shape sentinel) would be a silent
/// change, so [`f32_tensor`] rejects it rather than rounding.
#[derive(Clone, Debug, PartialEq)]
pub struct OnnxTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl OnnxTensor {
    /// Element count implied by `shape` (1 for a scalar, i.e. empty dims).
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(if self.shape.is_empty() { 1 } else { 0 })
    }
}

/// Read and decode a `.onnx` file.
///
/// Single-file models only: a tensor whose `data_location` is EXTERNAL errors by
/// name rather than yielding zeros. Both antelopev2 models are single-file, and
/// the sidecar resolution is a bounded follow-up (`external_data` carries
/// location/offset/length) that should not be guessed at now.
pub fn read_file(path: impl AsRef<Path>) -> Result<ModelProto, String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    crate::decode_model(&bytes).map_err(|e| format!("decode {}: {e}", path.display()))
}

/// The model's graph, or an error — `ModelProto::graph` is optional in the wire
/// format and a `None` here means the file is not a model.
pub fn graph(m: &ModelProto) -> Result<&GraphProto, String> {
    m.graph.as_ref().ok_or_else(|| "onnx: ModelProto has no graph".to_string())
}

/// Decode one `TensorProto` to f32.
///
/// Handles both storage forms the format allows: inline `raw_data` (the packed
/// little-endian bytes an exporter writes) and the typed repeated fields
/// (`float_data` / `int32_data` / `int64_data`), which is what a hand-built or
/// text-format proto uses. ONNX permits either for the same dtype, so a reader
/// that only checks `raw_data` returns an empty tensor for a perfectly valid
/// file — silently, since the shape still parses.
pub fn f32_tensor(t: &TensorProto) -> Result<OnnxTensor, String> {
    let name = t.name.clone();
    if t.data_location != 0 {
        return Err(format!(
            "onnx: tensor {name} is stored externally (data_location={}); external-data sidecars are not supported",
            t.data_location
        ));
    }
    let shape: Vec<usize> = t
        .dims
        .iter()
        .map(|&d| {
            usize::try_from(d).map_err(|_| format!("onnx: tensor {name} has negative dim {d}"))
        })
        .collect::<Result<_, _>>()?;
    let want: usize = shape.iter().product();

    let dt = DataType::try_from(t.data_type)
        .map_err(|_| format!("onnx: tensor {name} has unknown data_type {}", t.data_type))?;

    let data: Vec<f32> = if !t.raw_data.is_empty() {
        let raw = &t.raw_data;
        match dt {
            DataType::Float => decode_raw(&name, raw, 4, |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))?,
            DataType::Int32 => decode_raw(&name, raw, 4, |b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)?,
            DataType::Int64 => {
                let v: Vec<i64> = decode_raw_t(&name, raw, 8, |b| {
                    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                })?;
                i64_to_f32(&name, &v)?
            }
            DataType::Int8 => raw.iter().map(|&b| b as i8 as f32).collect(),
            DataType::Uint8 => raw.iter().map(|&b| b as f32).collect(),
            other => return Err(format!("onnx: tensor {name} has unsupported dtype {other:?}")),
        }
    } else if !t.float_data.is_empty() {
        t.float_data.clone()
    } else if !t.int64_data.is_empty() {
        i64_to_f32(&name, &t.int64_data)?
    } else if !t.int32_data.is_empty() {
        t.int32_data.iter().map(|&v| v as f32).collect()
    } else if want == 0 {
        Vec::new()
    } else {
        return Err(format!("onnx: tensor {name} carries no data (raw_data and *_data all empty)"));
    };

    if data.len() != want {
        return Err(format!("onnx: tensor {name} shape {shape:?} wants {want} values, decoded {}", data.len()));
    }
    Ok(OnnxTensor { name, shape, data })
}

fn decode_raw(
    name: &str,
    raw: &[u8],
    width: usize,
    f: impl Fn(&[u8]) -> f32,
) -> Result<Vec<f32>, String> {
    if !raw.len().is_multiple_of(width) {
        return Err(format!("onnx: tensor {name} raw_data length {} is not a multiple of {width}", raw.len()));
    }
    Ok(raw.chunks_exact(width).map(f).collect())
}

fn decode_raw_t<T>(
    name: &str,
    raw: &[u8],
    width: usize,
    f: impl Fn(&[u8]) -> T,
) -> Result<Vec<T>, String> {
    if !raw.len().is_multiple_of(width) {
        return Err(format!("onnx: tensor {name} raw_data length {} is not a multiple of {width}", raw.len()));
    }
    Ok(raw.chunks_exact(width).map(f).collect())
}

/// int64 -> f32, refusing anything not exactly representable.
///
/// Rounding here would be silent: an ONNX `Reshape` shape or a `Gather` index
/// that landed past 2^24 would come back as a *different, plausible* integer.
fn i64_to_f32(name: &str, v: &[i64]) -> Result<Vec<f32>, String> {
    v.iter()
        .map(|&x| {
            let y = x as f32;
            if y as i64 == x {
                Ok(y)
            } else {
                Err(format!("onnx: tensor {name} int64 value {x} is not exact in f32"))
            }
        })
        .collect()
}

/// Every initializer of `g`, decoded, keyed by name.
///
/// Duplicate names are an error, not a last-one-wins: an importer's coverage
/// check counts source tensors, and a silently dropped duplicate makes that
/// count lie.
pub fn initializers(g: &GraphProto) -> Result<HashMap<String, OnnxTensor>, String> {
    let mut out: HashMap<String, OnnxTensor> = HashMap::with_capacity(g.initializer.len());
    for t in &g.initializer {
        let d = f32_tensor(t)?;
        if let Some(prev) = out.insert(d.name.clone(), d) {
            return Err(format!("onnx: duplicate initializer {}", prev.name));
        }
    }
    Ok(out)
}

/// An attribute of `n` by name, or `None`.
pub fn attr<'a>(n: &'a crate::onnx::NodeProto, name: &str) -> Option<&'a AttributeProto> {
    n.attribute.iter().find(|a| a.name == name)
}

/// An `ints` attribute, or `default` when absent.
pub fn attr_ints(n: &crate::onnx::NodeProto, name: &str, default: &[i64]) -> Vec<i64> {
    attr(n, name).map(|a| a.ints.clone()).unwrap_or_else(|| default.to_vec())
}

/// An `int` attribute, or `default` when absent.
pub fn attr_int(n: &crate::onnx::NodeProto, name: &str, default: i64) -> i64 {
    attr(n, name).map(|a| a.i).unwrap_or(default)
}

/// A `float` attribute, or `default` when absent (`epsilon`, `alpha`, …).
pub fn attr_f32(n: &crate::onnx::NodeProto, name: &str, default: f32) -> f32 {
    attr(n, name).map(|a| a.f).unwrap_or(default)
}

/// A `string` attribute as UTF-8, or `default` when absent/not UTF-8.
pub fn attr_str(n: &crate::onnx::NodeProto, name: &str, default: &str) -> String {
    attr(n, name)
        .and_then(|a| std::str::from_utf8(&a.s).ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| default.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphBuilder, Node};

    /// Round-trip through this crate's own serializer: build a graph with an
    /// initializer, encode, decode, and read the values back.
    #[test]
    fn reads_back_an_initializer_this_crate_wrote() {
        let mut g = GraphBuilder::new("rt");
        g.input_f32("x", &[1, 2]);
        g.output_f32("y", &[1, 2]);
        g.init_f32("w", &[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        g.add(Node::new("Add", &["x", "w"], &["y"]).name("add"));
        let bytes = g.finish();

        let m = crate::decode_model(&bytes).unwrap();
        let gr = graph(&m).unwrap();
        let inits = initializers(gr).unwrap();
        assert_eq!(inits.len(), 1);
        let w = &inits["w"];
        assert_eq!(w.shape, vec![2, 2]);
        assert_eq!(w.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(gr.node.len(), 1);
        assert_eq!(gr.node[0].op_type, "Add");
    }

    #[test]
    fn int64_initializers_decode_exactly_and_reject_the_inexact() {
        let mut t = TensorProto { name: "shape".into(), data_type: DataType::Int64 as i32, ..Default::default() };
        t.dims = vec![2];
        t.int64_data = vec![1, -1];
        let d = f32_tensor(&t).unwrap();
        assert_eq!(d.data, vec![1.0, -1.0]);

        // 2^24 + 1 is the first integer f32 cannot represent.
        t.int64_data = vec![16_777_217, 0];
        let err = f32_tensor(&t).unwrap_err();
        assert!(err.contains("not exact in f32"), "{err}");
    }

    #[test]
    fn external_data_errors_by_name_instead_of_returning_zeros() {
        let t = TensorProto {
            name: "big.weight".into(),
            data_type: DataType::Float as i32,
            dims: vec![4],
            data_location: 1,
            ..Default::default()
        };
        let err = f32_tensor(&t).unwrap_err();
        assert!(err.contains("big.weight"), "{err}");
        assert!(err.contains("externally"), "{err}");
    }

    #[test]
    fn a_shape_data_length_mismatch_is_an_error() {
        let t = TensorProto {
            name: "w".into(),
            data_type: DataType::Float as i32,
            dims: vec![3],
            float_data: vec![1.0, 2.0],
            ..Default::default()
        };
        assert!(f32_tensor(&t).unwrap_err().contains("wants 3 values"));
    }

    #[test]
    fn attribute_accessors_read_conv_hyperparameters() {
        let mut g = GraphBuilder::new("conv");
        g.input_f32("x", &[1, 1, 4, 4]);
        g.output_f32("y", &[1, 1, 2, 2]);
        g.init_f32("w", &[1, 1, 3, 3], vec![0.0; 9]);
        g.add(
            Node::new("Conv", &["x", "w"], &["y"])
                .name("conv")
                .attr_ints("kernel_shape", &[3, 3])
                .attr_ints("strides", &[2, 2])
                .attr_int("group", 1)
                .attr_str("mode", "nearest"),
        );
        let m = crate::decode_model(&g.finish()).unwrap();
        let n = &graph(&m).unwrap().node[0];
        assert_eq!(attr_ints(n, "kernel_shape", &[]), vec![3, 3]);
        assert_eq!(attr_ints(n, "strides", &[]), vec![2, 2]);
        assert_eq!(attr_int(n, "group", 7), 1);
        assert_eq!(attr_int(n, "dilations_missing", 5), 5);
        assert_eq!(attr_ints(n, "pads", &[0, 0, 0, 0]), vec![0, 0, 0, 0]);
        assert_eq!(attr_str(n, "mode", "bilinear"), "nearest");
        // an absent float attribute falls back, which is how `epsilon` reads on
        // an exporter that leaves the ONNX default implicit
        assert_eq!(attr_f32(n, "epsilon", 1e-5), 1e-5);
    }
}
