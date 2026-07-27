// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared weight container: `[u64 LE json_len][json header][f32 LE blob]`,
//! tensor offsets in f32 units. Used for inference weights, training
//! checkpoints, and the PyTorch golden-reference / batch-stream files.

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;

use serde_json::Value;

pub mod safetensors;
#[cfg(not(target_arch = "wasm32"))]
pub mod mmap;
pub mod torchpt;
pub mod zipread;

/// One tensor read from a container (role is "" if the header omits it).
pub struct LoadedTensor {
    pub name: String,
    pub role: String,
    pub data: Vec<f32>,
}

pub struct Container {
    pub header: Value,
    pub tensors: Vec<LoadedTensor>,
}

impl Container {
    /// Tensors whose role matches `role`, keyed by name.
    pub fn by_role(&self, role: &str) -> HashMap<String, Vec<f32>> {
        self.tensors
            .iter()
            .filter(|t| t.role == role)
            .map(|t| (t.name.clone(), t.data.clone()))
            .collect()
    }
    pub fn find(&self, name: &str, role: &str) -> Option<&Vec<f32>> {
        self.tensors
            .iter()
            .find(|t| t.name == name && t.role == role)
            .map(|t| &t.data)
    }
}

/// Parse a weight container from an in-memory byte slice. This is the portable
/// core: native `load` reads the file then calls this; the browser entry point
/// (`web::run_inference`) passes the fetched bytes directly, since there is no
/// `std::fs` in a browser.
pub fn parse(bytes: &[u8]) -> Container {
    let jlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header: Value =
        serde_json::from_str(std::str::from_utf8(&bytes[8..8 + jlen]).expect("bad header utf8"))
            .expect("bad header json");
    let data = &bytes[8 + jlen..];
    let read = |offset: usize, numel: usize| -> Vec<f32> {
        data[offset * 4..(offset + numel) * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };
    let mut tensors = Vec::new();
    for t in header["tensors"].as_array().expect("tensors array") {
        let name = t["name"].as_str().unwrap().to_string();
        let role = t["role"].as_str().unwrap_or("").to_string();
        let vals = read(
            t["offset"].as_u64().unwrap() as usize,
            t["numel"].as_u64().unwrap() as usize,
        );
        tensors.push(LoadedTensor { name, role, data: vals });
    }
    Container { header, tensors }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load(path: &str) -> Container {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("cannot read {path}"));
    parse(&bytes)
}

/// Write a checkpoint: `config` is the model config object, `tensors` is an
/// ordered list of (name, shape, data). Offsets/numel are filled in here.
///
/// The write is atomic: bytes go to a sibling `<path>.tmp` which is then renamed
/// over `path`. A crash (or the GPU device loss that periodic checkpointing
/// guards against) mid-write therefore never truncates or corrupts an existing
/// good checkpoint — readers see either the old file or the complete new one.
#[cfg(not(target_arch = "wasm32"))]
pub fn save(path: &str, config: Value, tensors: &[(String, Vec<u64>, Vec<f32>)]) {
    let mut entries = Vec::new();
    let mut blob: Vec<f32> = Vec::new();
    for (name, shape, data) in tensors {
        entries.push(serde_json::json!({
            "name": name, "shape": shape, "offset": blob.len(), "numel": data.len()
        }));
        blob.extend_from_slice(data);
    }
    let header = serde_json::json!({ "config": config, "tensors": entries });
    let hbytes = serde_json::to_vec(&header).unwrap();

    // Create the parent directory if needed so `--out some/dir/x.weights` works
    // without a manual mkdir.
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .unwrap_or_else(|e| panic!("cannot create directory {}: {e}", parent.display()));
        }
    }

    let tmp = format!("{path}.tmp");
    {
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(&tmp).unwrap_or_else(|_| panic!("cannot write {tmp}")),
        );
        file.write_all(&(hbytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&hbytes).unwrap();
        let mut bytes: Vec<u8> = Vec::with_capacity(blob.len() * 4);
        for v in &blob {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();
    }
    std::fs::rename(&tmp, path).unwrap_or_else(|_| panic!("cannot finalise {path}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_load_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("moe_rs_ckpt_test_{}.bin", std::process::id()));
        let p = path.to_str().unwrap();
        let cfg = serde_json::json!({"d_model": 8, "n_layers": 1});
        let tensors = vec![
            ("a".to_string(), vec![2u64, 2], vec![1.0f32, -2.5, 3.25, 4.0]),
            ("b".to_string(), vec![3u64], vec![0.1f32, 0.2, 0.3]),
        ];
        save(p, cfg, &tensors);
        let c = load(p);
        assert_eq!(c.header["config"]["d_model"].as_u64().unwrap(), 8);
        let map = c.by_role(""); // saved tensors have no role
        assert_eq!(map["a"], vec![1.0, -2.5, 3.25, 4.0]);
        assert_eq!(map["b"], vec![0.1, 0.2, 0.3]);
        assert_eq!(c.find("a", "").unwrap().len(), 4);
        assert!(c.find("a", "init").is_none()); // role filter works
        std::fs::remove_file(p).ok();
    }
}
