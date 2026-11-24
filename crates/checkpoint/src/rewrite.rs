// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for publishing fine-tuned models in
// the exact layout their upstream tooling loads for its clients. If your team
// needs expertise in checkpoint formats and model release engineering then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Rewrite a safetensors file in its OWN layout: the same tensor names,
//! dtypes, shapes and `__metadata__`, with chosen tensors given new values
//! and chosen tensors left out. Every other tensor is copied byte for byte.
//!
//! This is the writer a trained checkpoint is published through when its
//! consumers are the upstream tools that made the base: they expect the base's
//! names and dtypes (a BF16 release stays BF16), not brain's F32 containers
//! ([`crate::st::save_safetensors`] writes F32 under `brain.config`). New
//! values are f32 and are narrowed to the stored dtype, BF16 and F16 rounding
//! to nearest even.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io;
use std::path::Path;

use serde_json::{Map, Value};

use crate::pio::PositionedIo;

/// Bytes copied per read when a tensor is carried over unchanged.
const COPY_CHUNK: usize = 64 << 20;

/// One tensor as the source header declares it.
struct Entry {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    start: u64,
    end: u64,
}

fn inval(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// The source header: its tensors in file order and its `__metadata__`.
fn read_entries(file: &File, path: &Path) -> io::Result<(Vec<Entry>, Option<Value>, u64)> {
    let file_len = file.metadata()?.len();
    let mut len = [0u8; 8];
    file.read_exact_at(&mut len, 0)?;
    let shown = path.display().to_string();
    let hlen = crate::safetensors::validate_header_len(&shown, u64::from_le_bytes(len), file_len)?;
    let mut hbytes = vec![0u8; hlen];
    file.read_exact_at(&mut hbytes, 8)?;
    let header: Map<String, Value> = serde_json::from_slice(&hbytes).map_err(|e| inval(format!("{shown}: header: {e}")))?;
    let blob = file_len - 8 - hlen as u64;
    let mut meta = None;
    let mut entries = Vec::with_capacity(header.len());
    for (name, v) in header {
        if name == "__metadata__" {
            meta = Some(v);
            continue;
        }
        let bad = |what: &str| inval(format!("{shown}: tensor {name:?}: {what}"));
        let dtype = v["dtype"].as_str().ok_or_else(|| bad("no dtype"))?.to_string();
        let shape: Vec<u64> = v["shape"].as_array().ok_or_else(|| bad("no shape"))?.iter().map(|d| d.as_u64().ok_or_else(|| bad("a shape entry is not an unsigned integer"))).collect::<io::Result<_>>()?;
        let offs = v["data_offsets"].as_array().filter(|o| o.len() == 2).ok_or_else(|| bad("data_offsets is not a [start, end] pair"))?;
        let (start, end) = (offs[0].as_u64().ok_or_else(|| bad("bad start offset"))?, offs[1].as_u64().ok_or_else(|| bad("bad end offset"))?);
        if start > end || end > blob {
            return Err(bad("data_offsets lie outside the file"));
        }
        entries.push(Entry { name, dtype, shape, start, end });
    }
    entries.sort_by_key(|e| e.start);
    Ok((entries, meta, 8 + hlen as u64))
}

/// `values` in `dtype`'s little-endian bytes, or `None` for a dtype that does
/// not hold real numbers.
fn encode(dtype: &str, values: &[f32]) -> Option<Vec<u8>> {
    Some(match dtype {
        "F32" => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        "BF16" => values.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect(),
        "F16" => values.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect(),
        _ => return None,
    })
}

/// Write `dst` as `src` with each of `replace`'s tensors (name, shape, f32
/// values) stored in place of the source's under the source's dtype, and every
/// tensor `leave_out` names left out; every other tensor's bytes and the
/// `__metadata__` are the source's.
///
/// Refused before anything is written: a replaced name the source does not
/// hold or that `leave_out` also names, a shape other than the source's, a value
/// count the shape does not have, a replaced tensor stored as anything but
/// F32/BF16/F16, a name replaced twice. The write goes to a temporary beside
/// `dst` that is synced and renamed into place; `keep_going` is asked before
/// each tensor, and once it says no the temporary is removed and the answer
/// is `Ok(false)`.
pub fn rewrite_safetensors(src: &Path, dst: &Path, replace: &[(String, Vec<u64>, Vec<f32>)], leave_out: &dyn Fn(&str) -> bool, keep_going: &mut dyn FnMut() -> bool) -> io::Result<bool> {
    let source = File::open(src).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", src.display())))?;
    let (entries, meta, blob_start) = read_entries(&source, src)?;
    let by_name: HashMap<&str, &Entry> = entries.iter().map(|e| (e.name.as_str(), e)).collect();
    let mut new_values: HashMap<&str, &[f32]> = HashMap::with_capacity(replace.len());
    for (name, shape, values) in replace {
        let entry = by_name.get(name.as_str()).ok_or_else(|| inval(format!("{name}: the base checkpoint has no such tensor")))?;
        if leave_out(name) {
            return Err(inval(format!("{name}: a replaced tensor is also one the rewrite leaves out")));
        }
        if *shape != entry.shape {
            return Err(inval(format!("{name}: shape {shape:?}, the base stores {:?}", entry.shape)));
        }
        let numel: u64 = shape.iter().product();
        if values.len() as u64 != numel {
            return Err(inval(format!("{name}: {} values for shape {shape:?} ({numel})", values.len())));
        }
        if encode(&entry.dtype, &[]).is_none() {
            return Err(inval(format!("{name}: stored as {}, which a trained value cannot be written in", entry.dtype)));
        }
        if new_values.insert(name.as_str(), values).is_some() {
            return Err(inval(format!("{name}: replaced twice")));
        }
    }

    let kept: Vec<&Entry> = entries.iter().filter(|e| !leave_out(&e.name)).collect();
    let mut header: BTreeMap<String, Value> = BTreeMap::new();
    let mut offset = 0u64;
    for e in &kept {
        let n = e.end - e.start;
        header.insert(e.name.clone(), serde_json::json!({ "dtype": e.dtype, "shape": e.shape, "data_offsets": [offset, offset + n] }));
        offset += n;
    }
    if let Some(m) = meta {
        header.insert("__metadata__".to_string(), m);
    }
    let mut hbytes = serde_json::to_vec(&header)?;
    // Eight-byte aligned blob, as the reference serializer pads it.
    while hbytes.len() % 8 != 0 {
        hbytes.push(b' ');
    }

    let mut tmp = dst.as_os_str().to_owned();
    tmp.push(format!(".{}.tmp", std::process::id()));
    let tmp = std::path::PathBuf::from(tmp);
    let out = File::create(&tmp)?;
    let written = (|| -> io::Result<bool> {
        out.write_all_at(&(hbytes.len() as u64).to_le_bytes(), 0)?;
        out.write_all_at(&hbytes, 8)?;
        let mut at = 8 + hbytes.len() as u64;
        let mut buf = Vec::new();
        for e in &kept {
            if !keep_going() {
                return Ok(false);
            }
            if let Some(values) = new_values.get(e.name.as_str()) {
                let bytes = encode(&e.dtype, values).expect("dtype checked above");
                out.write_all_at(&bytes, at)?;
            } else {
                let mut from = blob_start + e.start;
                let mut to = at;
                let mut left = e.end - e.start;
                while left > 0 {
                    let n = left.min(COPY_CHUNK as u64) as usize;
                    buf.resize(n, 0);
                    source.read_exact_at(&mut buf, from)?;
                    out.write_all_at(&buf, to)?;
                    (from, to, left) = (from + n as u64, to + n as u64, left - n as u64);
                }
            }
            at += e.end - e.start;
        }
        out.sync_all()?;
        Ok(true)
    })();
    drop(out);
    match written {
        Ok(true) => crate::st::durable_rename(&tmp, dst).map(|()| true),
        other => {
            let _ = std::fs::remove_file(&tmp);
            other
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weightio::WeightReader;

    fn scratch() -> brain_testutil::Scratch {
        brain_testutil::Scratch::dir("rewrite")
    }

    /// A small mixed-dtype file the way an upstream release stores one.
    fn base(dir: &Path) -> std::path::PathBuf {
        let bf16: Vec<u8> = [1.0f32, -2.0, 0.5, 3.0, 0.25, -1.5].iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect();
        let f32b: Vec<u8> = [7.0f32, 8.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let ids: Vec<u8> = [5i64, 6].iter().flat_map(|v| v.to_le_bytes()).collect();
        let views = vec![
            ("a.weight", safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2, 3], &bf16).unwrap()),
            ("b.bias", safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], &f32b).unwrap()),
            ("c.ids", safetensors::tensor::TensorView::new(safetensors::Dtype::I64, vec![2], &ids).unwrap()),
            ("spk.w", safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], &f32b).unwrap()),
        ];
        let meta: HashMap<String, String> = [("format".to_string(), "pt".to_string())].into();
        let path = dir.join("model.safetensors");
        std::fs::write(&path, safetensors::serialize(views, Some(meta)).unwrap()).unwrap();
        path
    }

    fn raw(path: &Path, name: &str) -> (String, Vec<usize>, Vec<u8>) {
        let bytes = std::fs::read(path).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let t = st.tensor(name).unwrap();
        (format!("{:?}", t.dtype()), t.shape().to_vec(), t.data().to_vec())
    }

    /// Named f32 tensors, as a rewrite takes them.
    type Tensors = Vec<(String, Vec<u64>, Vec<f32>)>;

    fn none(_: &str) -> bool {
        false
    }

    #[test]
    fn an_unchanged_rewrite_keeps_every_tensor_and_the_metadata_byte_for_byte() {
        let dir = scratch();
        let src = base(dir.path());
        let dst = dir.path().join("out.safetensors");
        assert!(rewrite_safetensors(&src, &dst, &[], &none, &mut || true).unwrap());
        for name in ["a.weight", "b.bias", "c.ids", "spk.w"] {
            assert_eq!(raw(&src, name), raw(&dst, name), "{name}");
        }
        let (_, meta) = safetensors::SafeTensors::read_metadata(&std::fs::read(&dst).unwrap()).unwrap();
        assert_eq!(meta.metadata().as_ref().and_then(|m| m.get("format")).map(String::as_str), Some("pt"));
    }

    #[test]
    fn a_replaced_tensor_is_stored_in_the_base_dtype_rounded_to_nearest_even_and_a_dropped_one_is_gone() {
        let dir = scratch();
        let src = base(dir.path());
        let dst = dir.path().join("out.safetensors");
        // 1 + 2^-8 is a tie between two bf16 neighbours: even rounds down to 1.0.
        let new = vec![1.0 + 1.0 / 256.0, 0.1, -0.3, 1e-3, 2.5, 100.7];
        let replace = vec![("a.weight".to_string(), vec![2, 3], new.clone()), ("b.bias".to_string(), vec![2], vec![0.1, 0.2])];
        assert!(rewrite_safetensors(&src, &dst, &replace, &|n| n.starts_with("spk."), &mut || true).unwrap());
        let (dtype, shape, bytes) = raw(&dst, "a.weight");
        assert_eq!((dtype.as_str(), shape), ("BF16", vec![2, 3]));
        let want: Vec<u8> = new.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect();
        assert_eq!(bytes, want);
        assert_eq!(half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32(), 1.0);
        let r = WeightReader::open(dst.to_str().unwrap()).unwrap();
        assert_eq!(r.tensor("b.bias").unwrap(), vec![0.1, 0.2], "an F32 tensor takes the values exactly");
        assert_eq!(raw(&src, "c.ids"), raw(&dst, "c.ids"));
        assert!(r.shape("spk.w").is_none(), "a dropped tensor is not in the file");
    }

    #[test]
    fn a_rewrite_that_names_the_wrong_tensor_shape_or_dtype_is_refused_and_writes_nothing() {
        let dir = scratch();
        let src = base(dir.path());
        let dst = dir.path().join("out.safetensors");
        let cases: Vec<(Tensors, &str)> = vec![
            (vec![("nope".to_string(), vec![2], vec![0.0; 2])], "no such tensor"),
            (vec![("a.weight".to_string(), vec![3, 2], vec![0.0; 6])], "shape"),
            (vec![("a.weight".to_string(), vec![2, 3], vec![0.0; 5])], "values"),
            (vec![("c.ids".to_string(), vec![2], vec![0.0; 2])], "I64"),
            (vec![("spk.w".to_string(), vec![2], vec![0.0; 2])], "leaves out"),
            (vec![("b.bias".to_string(), vec![2], vec![0.0; 2]), ("b.bias".to_string(), vec![2], vec![1.0; 2])], "twice"),
        ];
        for (replace, why) in cases {
            let e = rewrite_safetensors(&src, &dst, &replace, &|n| n.starts_with("spk."), &mut || true).unwrap_err();
            assert!(e.to_string().contains(why), "{why}: {e}");
            assert!(!dst.exists());
        }
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1, "no temporary is left behind");
    }

    #[test]
    fn a_stopped_rewrite_leaves_no_file() {
        let dir = scratch();
        let src = base(dir.path());
        let dst = dir.path().join("out.safetensors");
        assert!(!rewrite_safetensors(&src, &dst, &[], &none, &mut || false).unwrap());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
