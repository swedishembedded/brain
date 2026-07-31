// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared weight container, backed by safetensors (see [`st`]). Used for
//! inference weights and training checkpoints. The public `Container` API is
//! preserved: `header` carries `{"config": ...}` and every tensor has role `""`
//! (safetensors has no role concept). Reads/writes delegate to [`st`].

use std::collections::HashMap;

use serde_json::Value;

pub mod safetensors;
pub mod st;
pub mod gguf;
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
/// `std::fs` in a browser. Backed by safetensors; every tensor gets role `""`.
pub fn parse(bytes: &[u8]) -> Container {
    let m = st::parse_safetensors(bytes).expect("parse safetensors");
    let header = serde_json::json!({ "config": m.config() });
    let tensors = m
        .tensors
        .into_iter()
        .map(|(name, data)| LoadedTensor { name, role: String::new(), data })
        .collect();
    Container { header, tensors }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load(path: &str) -> Container {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("cannot read {path}"));
    parse(&bytes)
}

/// Write a checkpoint: `config` is the model config object, `tensors` is an
/// ordered list of (name, shape, data). Delegates to [`st::save_safetensors`],
/// which writes atomically (tmp + rename).
#[cfg(not(target_arch = "wasm32"))]
pub fn save(path: &str, config: Value, tensors: &[(String, Vec<u64>, Vec<f32>)]) {
    st::save_safetensors(path, tensors, &config, None).expect("save safetensors");
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
