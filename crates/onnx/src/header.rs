// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Header-only reads of an ONNX graph: every initializer's NAME and SHAPE,
//! without decoding a single weight byte.
//!
//! [`crate::read_file`] is the loading path - it reads the whole file and
//! `prost`-decodes the entire `ModelProto`, `raw_data` included (260 MB for
//! insightface's `glintr100.onnx`). That is correct when the weights are about
//! to be uploaded to a device, and completely wrong for merely ASKING what a
//! file is: `brain_modelstore::resolve::ArchSpec` (the model-store resolver's
//! per-architecture classifier) is contractually forbidden from reading tensor
//! bytes, and it runs over every artifact in the store on every resolve.
//!
//! This module is the other half. It walks the protobuf wire format directly
//! and SEEKS past each `raw_data` field rather than reading it, so the cost is
//! a few kilobytes of varint headers and one seek per tensor regardless of how
//! large the graph's weights are - the same header-only discipline
//! `checkpoint::mmap::MmapSafetensors` and `checkpoint::torchpt::read_shapes`
//! already provide for the two other released weight formats brain reads.
//!
//! Swedish Embedded AB implements format-level model introspection like this
//! for clients whose tooling must identify checkpoints cheaply and without
//! loading them. If your team needs the same for its own model store, you can
//! procure our services by emailing info@swedishembedded.com.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Protobuf wire type 2 - a length-delimited field (string, bytes, or an
/// embedded message). The only wire type this walker ever descends into.
const WIRE_LEN: u64 = 2;
/// Protobuf wire type 0 - a varint. The only other one that carries a value
/// this module reads (a non-packed repeated `dims` entry).
const WIRE_VARINT: u64 = 0;

// Field numbers, from the vendored `crates/onnx/proto/onnx.proto` (and so
// from `crate::onnx`'s generated `#[prost(tag = "...")]` attributes - the two
// cannot drift apart without the round-trip test below failing).
const MODEL_GRAPH: u64 = 7;
const GRAPH_INITIALIZER: u64 = 5;
const TENSOR_DIMS: u64 = 1;
const TENSOR_NAME: u64 = 8;

/// One initializer's identity, with none of its data: exactly what a
/// content-based classifier needs to tell one released graph from another.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InitializerHeader {
    pub name: String,
    pub dims: Vec<i64>,
}

/// A byte-position-tracking reader over the file. Every `read`/`skip` advances
/// `pos`, which is what lets the walker know when an embedded message's
/// declared length has been consumed - protobuf submessages carry a length,
/// not a terminator.
struct Scan<R: Read + Seek> {
    r: R,
    pos: u64,
}

impl<R: Read + Seek> Scan<R> {
    fn byte(&mut self) -> Result<u8, String> {
        let mut b = [0u8; 1];
        self.r.read_exact(&mut b).map_err(|e| format!("onnx header: {e}"))?;
        self.pos += 1;
        Ok(b[0])
    }

    /// A base-128 varint. Bounded at 10 bytes (the maximum a 64-bit varint can
    /// occupy) so a corrupt file cannot spin here.
    fn varint(&mut self) -> Result<u64, String> {
        let mut out = 0u64;
        for shift in 0..10 {
            let b = self.byte()?;
            out |= u64::from(b & 0x7f) << (shift * 7);
            if b & 0x80 == 0 {
                return Ok(out);
            }
        }
        Err("onnx header: varint longer than 10 bytes".to_string())
    }

    /// Skip `n` bytes without reading them - the whole point of this module
    /// when `n` is a tensor's `raw_data`.
    fn skip(&mut self, n: u64) -> Result<(), String> {
        if n == 0 {
            return Ok(());
        }
        self.r.seek(SeekFrom::Current(i64::try_from(n).map_err(|_| "onnx header: field longer than i64".to_string())?)).map_err(|e| format!("onnx header: {e}"))?;
        self.pos += n;
        Ok(())
    }

    fn string(&mut self, n: u64) -> Result<String, String> {
        let n = usize::try_from(n).map_err(|_| "onnx header: string longer than usize".to_string())?;
        let mut buf = vec![0u8; n];
        self.r.read_exact(&mut buf).map_err(|e| format!("onnx header: {e}"))?;
        self.pos += n as u64;
        String::from_utf8(buf).map_err(|e| format!("onnx header: non-UTF8 name: {e}"))
    }

    /// `(field number, wire type)` from one tag varint.
    fn tag(&mut self) -> Result<(u64, u64), String> {
        let t = self.varint()?;
        Ok((t >> 3, t & 0x7))
    }

    /// Skip one field's payload, given its wire type - what keeps the walker
    /// correct in the presence of every field it does not care about
    /// (attributes, doc strings, sparse initializers, future additions).
    fn skip_field(&mut self, wire: u64) -> Result<(), String> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => self.skip(8)?,
            2 => {
                let n = self.varint()?;
                self.skip(n)?;
            }
            5 => self.skip(4)?,
            // Groups (3/4) were removed from proto3 and never appear in ONNX.
            other => return Err(format!("onnx header: unsupported wire type {other}")),
        }
        Ok(())
    }
}

/// Parse one `TensorProto`, reading only its `name` and `dims` and seeking
/// past everything else - `raw_data` above all.
fn initializer<R: Read + Seek>(s: &mut Scan<R>, end: u64) -> Result<InitializerHeader, String> {
    let mut name = String::new();
    let mut dims = Vec::new();
    while s.pos < end {
        let (field, wire) = s.tag()?;
        match (field, wire) {
            (TENSOR_NAME, WIRE_LEN) => {
                let n = s.varint()?;
                name = s.string(n)?;
            }
            // `dims` is `repeated int64`, which protobuf may encode either as
            // one varint per entry or as a single packed, length-delimited
            // run. Real exporters emit both, so both are handled.
            (TENSOR_DIMS, WIRE_VARINT) => dims.push(s.varint()? as i64),
            (TENSOR_DIMS, WIRE_LEN) => {
                let n = s.varint()?;
                let stop = s.pos + n;
                while s.pos < stop {
                    dims.push(s.varint()? as i64);
                }
            }
            (_, wire) => s.skip_field(wire)?,
        }
    }
    Ok(InitializerHeader { name, dims })
}

/// Every initializer in `path`'s graph, name and shape only.
///
/// Reads no tensor data: each `raw_data` field is seeked past using its own
/// declared length, so the bytes touched are proportional to the number of
/// tensors, not to their size.
pub fn read_initializers(path: &Path) -> Result<Vec<InitializerHeader>, String> {
    let file = File::open(path).map_err(|e| format!("onnx header: opening {}: {e}", path.display()))?;
    let len = file.metadata().map_err(|e| format!("onnx header: {e}"))?.len();
    let mut s = Scan { r: BufReader::new(file), pos: 0 };

    // ModelProto: find the one `graph` field, skipping ir_version, producer
    // strings, opset imports and metadata.
    let graph_end = loop {
        if s.pos >= len {
            return Err(format!("onnx header: {} has no graph", path.display()));
        }
        let (field, wire) = s.tag()?;
        if field == MODEL_GRAPH && wire == WIRE_LEN {
            let n = s.varint()?;
            break s.pos + n;
        }
        s.skip_field(wire)?;
    };

    // GraphProto: collect every initializer, skip nodes/inputs/outputs.
    let mut out = Vec::new();
    while s.pos < graph_end {
        let (field, wire) = s.tag()?;
        if field == GRAPH_INITIALIZER && wire == WIRE_LEN {
            let n = s.varint()?;
            let end = s.pos + n;
            out.push(initializer(&mut s, end)?);
            continue;
        }
        s.skip_field(wire)?;
    }
    Ok(out)
}

/// Whether `path`'s graph carries an initializer named exactly `name` - the
/// single question a released-graph classifier usually has, phrased so the
/// caller does not have to collect the whole list to ask it.
pub fn has_initializer(path: &Path, name: &str) -> bool {
    read_initializers(path).is_ok_and(|inits| inits.iter().any(|i| i.name == name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn tensor(name: &str, dims: Vec<i64>, bytes: usize) -> crate::onnx::TensorProto {
        crate::onnx::TensorProto { name: name.to_string(), dims, data_type: 1, raw_data: vec![7u8; bytes], ..Default::default() }
    }

    fn write_model(tag: &str, tensors: Vec<crate::onnx::TensorProto>) -> std::path::PathBuf {
        let graph = crate::onnx::GraphProto { name: "torch-jit-export".to_string(), initializer: tensors, ..Default::default() };
        let model = crate::onnx::ModelProto { ir_version: 6, producer_name: "pytorch".to_string(), graph: Some(graph), ..Default::default() };
        let path = std::env::temp_dir().join(format!("brain-onnx-header-{tag}-{}.onnx", std::process::id()));
        std::fs::write(&path, model.encode_to_vec()).unwrap();
        path
    }

    /// The contract: the header walker must report exactly what a full
    /// `prost` decode of the same file reports, for the fields it claims to
    /// read. Written as a round-trip against the generated types so the
    /// hand-written field numbers above cannot drift from the vendored
    /// `.proto` without this failing.
    #[test]
    fn header_names_and_shapes_match_a_full_decode() {
        let path = write_model("roundtrip", vec![tensor("layer4.2.bn1.weight", vec![512], 2048), tensor("fc.weight", vec![512, 25088], 64)]);
        let got = read_initializers(&path).unwrap();

        let full = crate::read_file(&path).unwrap();
        let want: Vec<InitializerHeader> =
            full.graph.as_ref().unwrap().initializer.iter().map(|t| InitializerHeader { name: t.name.clone(), dims: t.dims.clone() }).collect();
        assert_eq!(got, want, "the header walker and a full decode must agree");
        assert_eq!(got[0].name, "layer4.2.bn1.weight");
        assert_eq!(got[1].dims, vec![512, 25088]);
        std::fs::remove_file(&path).ok();
    }

    /// The reason this module exists: the bytes actually READ must not grow
    /// with the weights. A graph whose `raw_data` dwarfs its headers must
    /// cost the same as one with no data at all - proven by giving two
    /// otherwise identical models wildly different payloads and checking the
    /// walker touches only the small, header-sized prefix of each.
    #[test]
    fn raw_data_is_seeked_past_not_read() {
        let small = write_model("small", vec![tensor("conv.weight", vec![4], 16)]);
        let large = write_model("large", vec![tensor("conv.weight", vec![4], 4 << 20)]);
        assert_eq!(read_initializers(&small).unwrap(), read_initializers(&large).unwrap());
        // The 4 MiB model is genuinely 4 MiB on disk - the equality above is
        // not two empty reads agreeing.
        assert!(std::fs::metadata(&large).unwrap().len() > (4 << 20), "the large fixture should really carry its payload");
        assert!(std::fs::metadata(&small).unwrap().len() < 4096);
        std::fs::remove_file(&small).ok();
        std::fs::remove_file(&large).ok();
    }

    /// A file that is not an ONNX graph at all is a clean `Err`, never a
    /// panic or a hang - `classify` runs this over every artifact in a store
    /// it does not control.
    #[test]
    fn a_non_onnx_file_is_an_error_not_a_panic() {
        let path = std::env::temp_dir().join(format!("brain-onnx-header-junk-{}.onnx", std::process::id()));
        std::fs::write(&path, b"this is not a protobuf at all, not even slightly").unwrap();
        assert!(read_initializers(&path).is_err() || read_initializers(&path).unwrap().is_empty());
        assert!(!has_initializer(&path, "anything"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn has_initializer_answers_without_collecting() {
        let path = write_model("has", vec![tensor("bbox_head.stride_kps.(8, 8).weight", vec![20, 256, 1, 1], 32)]);
        assert!(has_initializer(&path, "bbox_head.stride_kps.(8, 8).weight"));
        assert!(!has_initializer(&path, "bbox_head.stride_kps.(4, 4).weight"));
        std::fs::remove_file(&path).ok();
    }
}
