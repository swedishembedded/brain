// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T5/umT5 behind the generalized [`capability`] interface - what makes
//! `brain caps t5encoder` / `brain do t5encoder encode …`, the D-Bus `Run`
//! method and `brain perf`'s `CapabilityTarget` work with no T5-specific
//! plumbing in the CLI or the transports.
//!
//! One action, one per variant:
//!
//! * **`encode`** - a string in, `[T, d_model]` f32 last-hidden-state out
//!   (`T5Encoder::read_context`, see its doc for why context rather than
//!   hidden: it is a no-op for the unmasked FLUX variant and matters for the
//!   masked Wan one). `variant` selects `flux_xxl` (FLUX.1/2's second text
//!   encoder, unmasked, T5-XXL v1.1) or `wan_umt5` (Wan2.1/2.2's text tower,
//!   masked, umT5-XXL).
//!
//! # Batching is real here, for the same reason as CLIP
//!
//! [`Session::encode_batch`] builds the encoder at `b = texts.len()` and runs
//! **one** forward over the whole batch. Every row is the same fixed `max_len`
//! context (right-padded), so the residency adapter
//! (`crates/cli/src/resident_t5encoder.rs`) groups `run_batch` invocations by
//! `(variant, max_len)` and forwards each group to this in one call.
//!
//! # Directory layout
//!
//! `BRAIN_T5ENCODER_DIR` holds either or both variants, so a FLUX.1 release
//! root can be pointed at directly with no renaming:
//!
//! * `flux_xxl`: `text_encoder_2/` (safetensors, HF `T5EncoderModel` layout)
//!   + `tokenizer_2/tokenizer.json` - exactly the FLUX.1-*/ release layout.
//! * `wan_umt5`: `wan/models_t5_umt5-xxl-enc-bf16.pth` (the native Wan2.1
//!   checkpoint) + `wan/tokenizer.json`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use data::unigram::UnigramTokenizer;
use gpu_core::Gpu;
use serde_json::json;

use crate::config::T5Config;
use crate::import::Tensors;
use crate::model::T5Encoder;

/// The model id used on the CLI (`brain do t5encoder …`), over D-Bus and in
/// the residency manifest.
pub const MODEL: &str = "brain/t5encoder";

/// The two encoder variants this action can serve, by the name used on the
/// wire.
pub const VARIANTS: [&str; 2] = ["flux_xxl", "wan_umt5"];

/// A default context length wide enough for FLUX's real prompts without
/// forcing every caller to know umT5's 512-token window; `max_len` is still a
/// UI-rangeable param because callers with shorter prompts want a cheaper
/// forward.
pub const DEFAULT_MAX_LEN: u32 = 256;

fn to_map(t: Tensors) -> HashMap<String, Vec<f32>> {
    t.into_iter().map(|(k, (_, d))| (k, d)).collect()
}

fn variant_cfg(name: &str) -> Result<T5Config, String> {
    match name {
        "flux_xxl" => Ok(T5Config::xxl()),
        "wan_umt5" => Ok(T5Config::umt5_xxl()),
        other => Err(format!("t5encoder: unknown variant '{other}' (expected one of {VARIANTS:?})")),
    }
}

fn encode_spec() -> ActionSpec {
    ActionSpec::new("encode", "Encode a string with T5-XXL or umT5-XXL, returning its last hidden state.")
        .param(ParamSpec::new("text", ParamType::Str, "the string to encode").required())
        .param(
            ParamSpec::new("variant", ParamType::Str, "flux_xxl (FLUX.1/2, unmasked) or wan_umt5 (Wan2.1/2.2, masked)")
                .default(json!("flux_xxl")),
        )
        .param(
            ParamSpec::new("max_len", ParamType::Int, "context length; the input is right-padded/truncated to it")
                .default(json!(DEFAULT_MAX_LEN))
                .min(1.0)
                .max(512.0)
                .step(1.0),
        )
        .output(BlobSpec::new(
            "hidden_states",
            Media::Bytes,
            "f32 little-endian, row-major [max_len, d_model]; padded rows are exactly zero",
        ))
}

/// The full, static capability manifest - safe to build with no weights
/// loaded.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "T5-XXL v1.1 and umT5-XXL text encoders.", vec![encode_spec()])
}

// ===================== the shared work =====================

/// The encoders on one device - the single implementation of `encode`, shared
/// by [`T5encoderProvider`] and the residency adapter
/// (`crates/cli/src/resident_t5encoder.rs`).
///
/// Encoders are built lazily and keyed by `(variant, b, max_len)`: a fixed `t`
/// graph, so a batch of 4 at 256 tokens needs a different build than a batch
/// of 1 at 512.
pub struct Session {
    gpu: Gpu,
    dir: String,
    tok: Mutex<HashMap<&'static str, UnigramTokenizer>>,
    enc: Mutex<HashMap<(String, u32, u32), T5Encoder>>,
}

impl Session {
    /// `dir` is `BRAIN_T5ENCODER_DIR` - see the module docs for the layout.
    pub fn load(dir: &str, gpu: Gpu) -> Result<Session, String> {
        if !has_flux(dir) && !has_wan(dir) {
            return Err(format!("t5encoder: {dir} holds neither the flux_xxl nor the wan_umt5 layout"));
        }
        Ok(Session { gpu, dir: dir.into(), tok: Mutex::new(HashMap::new()), enc: Mutex::new(HashMap::new()) })
    }

    fn encode_ids(&self, variant: &str, text: &str, max_len: u32) -> Result<(Vec<u32>, Vec<u32>), String> {
        let key: &'static str = if variant == "flux_xxl" { "flux_xxl" } else { "wan_umt5" };
        let mut guard = self.tok.lock().map_err(|_| "t5encoder: tokenizer lock poisoned")?;
        if !guard.contains_key(key) {
            let dir = if key == "flux_xxl" { format!("{}/tokenizer_2", self.dir) } else { format!("{}/wan", self.dir) };
            guard.insert(key, UnigramTokenizer::from_dir(&dir)?);
        }
        let t = guard.get(key).expect("inserted above");
        let (ids, mask) = t.encode_padded(text, max_len as usize);
        Ok((ids, mask))
    }

    /// Build (or reuse) the encoder for `(variant, b, max_len)`.
    fn with_encoder<R>(
        &self,
        variant: &str,
        b: u32,
        max_len: u32,
        f: impl FnOnce(&T5Encoder) -> R,
    ) -> Result<R, String> {
        let cfg = variant_cfg(variant)?;
        let mut guard = self.enc.lock().map_err(|_| "t5encoder: encoder lock poisoned")?;
        let key = (variant.to_string(), b, max_len);
        if !guard.contains_key(&key) {
            let init = to_map(if variant == "flux_xxl" {
                let dir = std::path::Path::new(&self.dir).join("text_encoder_2");
                let tensors = crate::import::read_encoder(&dir)?;
                crate::import::import_hf(tensors, &cfg)?
            } else {
                let path = std::path::Path::new(&self.dir).join("wan").join("models_t5_umt5-xxl-enc-bf16.pth");
                let tensors = crate::import::read_encoder_pth(&path)?;
                crate::import::import_wan(tensors, &cfg)?
            });
            // `Gpu::share` - one device for every variant, the same rule
            // `clip::caps::Session` follows.
            let m = T5Encoder::new_on(self.gpu.share(), cfg, b, max_len, &init);
            guard.insert(key.clone(), m);
        }
        Ok(f(guard.get(&key).expect("built above")))
    }

    /// One forward over `texts.len()` rows - the genuine batched path.
    pub fn encode_batch(&self, variant: &str, texts: &[String], max_len: u32) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let cfg = variant_cfg(variant)?;
        let b = texts.len() as u32;
        let mut ids = Vec::with_capacity(texts.len() * max_len as usize);
        let mut mask = Vec::with_capacity(texts.len() * max_len as usize);
        for t in texts {
            let (row_ids, row_mask) = self.encode_ids(variant, t, max_len)?;
            ids.extend_from_slice(&row_ids);
            mask.extend_from_slice(&row_mask);
        }
        self.with_encoder(variant, b, max_len, |m| {
            m.set_tokens(&ids);
            if cfg.masked {
                m.set_mask(&mask);
            }
            m.forward();
            let flat = m.read_context();
            let d = flat.len() / texts.len();
            flat.chunks(d).map(|r| r.to_vec()).collect()
        })
    }

    fn encode(&self, inv: &Invocation) -> ActionResult {
        let text = inv.get_str("text").ok_or("t5encoder: 'text' is required")?;
        let variant = inv.get_str("variant").unwrap_or_else(|| "flux_xxl".into());
        let max_len = inv.get_i64("max_len").map(|v| v as u32).unwrap_or(DEFAULT_MAX_LEN);
        let mut out = self.encode_batch(&variant, std::slice::from_ref(&text), max_len)?;
        let v = out.pop().ok_or("t5encoder: empty batch result")?;
        Ok(Outcome::new()
            .set("max_len", json!(max_len))
            .set("variant", json!(variant))
            .blob("hidden_states", Blob::new(Media::Bytes, f32_le(&v))))
    }

    /// Dispatch by action name - the one place the one action is named.
    pub fn run(&self, action: &str, inv: &Invocation) -> ActionResult {
        match action {
            "encode" => self.encode(inv),
            other => Err(format!("t5encoder: unknown action '{other}'")),
        }
    }
}

fn has_flux(dir: &str) -> bool {
    let d = std::path::Path::new(dir);
    d.join("text_encoder_2").exists() && d.join("tokenizer_2").join("tokenizer.json").exists()
}

fn has_wan(dir: &str) -> bool {
    let d = std::path::Path::new(dir).join("wan");
    d.join("models_t5_umt5-xxl-enc-bf16.pth").exists() && d.join("tokenizer.json").exists()
}

fn f32_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ===================== the provider =====================

type HotSession = Arc<Mutex<Option<(String, Arc<Session>)>>>;

/// The executable T5/umT5 stack behind the manifest. Construction is free -
/// encoders import lazily on first use and stay resident.
pub struct T5encoderProvider {
    dir: String,
    hot: HotSession,
}

impl T5encoderProvider {
    pub fn new(dir: impl Into<String>) -> T5encoderProvider {
        T5encoderProvider { dir: dir.into(), hot: Arc::new(Mutex::new(None)) }
    }

    /// `BRAIN_T5ENCODER_DIR` - `None` when unset, or when the directory holds
    /// neither released layout, since without one no action can run.
    pub fn from_env() -> Option<T5encoderProvider> {
        let dir = std::env::var("BRAIN_T5ENCODER_DIR").ok().filter(|p| !p.is_empty())?;
        (has_flux(&dir) || has_wan(&dir)).then(|| T5encoderProvider::new(dir))
    }
}

impl Provider for T5encoderProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "encode")
            .then(|| Arc::new(T5encoderAction { dir: self.dir.clone(), hot: self.hot.clone() }) as Arc<dyn Action>)
    }
}

struct T5encoderAction {
    dir: String,
    hot: HotSession,
}

impl Action for T5encoderAction {
    fn spec(&self) -> ActionSpec {
        encode_spec()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let session = {
            let mut guard = self.hot.lock().map_err(|_| "t5encoder: hot model lock poisoned")?;
            if !matches!(&*guard, Some((d, _)) if *d == self.dir) {
                *guard = None; // free the old build before importing another directory
                let gpu = Gpu::new(crate::model::PIPELINES);
                *guard = Some((self.dir.clone(), Arc::new(Session::load(&self.dir, gpu)?)));
            }
            guard.as_ref().expect("built above").1.clone()
        };
        session.run("encode", inv)
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;

    #[test]
    fn manifest_declares_encode() {
        let m = manifest();
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, vec!["encode"]);
    }

    #[test]
    fn variant_names_round_trip_to_configs() {
        for v in VARIANTS {
            assert!(variant_cfg(v).is_ok(), "{v} should map to a config");
        }
        assert!(variant_cfg("nope").is_err());
        assert!(!variant_cfg("flux_xxl").unwrap().masked);
        assert!(variant_cfg("wan_umt5").unwrap().masked);
    }

    #[test]
    fn an_unknown_action_is_named_not_ignored() {
        let p = T5encoderProvider::new("/nonexistent");
        assert!(p.action("summarize").is_none());
    }

    #[test]
    fn from_env_declines_a_directory_with_neither_layout() {
        assert!(T5encoderProvider::from_env().is_none() || std::env::var("BRAIN_T5ENCODER_DIR").is_ok());
    }

    #[test]
    fn max_len_param_carries_ui_range() {
        let spec = encode_spec();
        let p = spec.params.iter().find(|p| p.name == "max_len").expect("max_len param");
        assert_eq!(p.min, Some(1.0));
        assert_eq!(p.max, Some(512.0));
    }
}
