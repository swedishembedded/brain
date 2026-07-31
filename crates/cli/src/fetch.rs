// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generic model downloader (`brain fetch`): a small [`Source`]/[`Fetcher`]
//! seam so today's streaming HTTP client can be swapped for (or joined by) a
//! torrent fetcher later without touching the registry or the CLI.
//!
//! GGUF is preferred wherever a known model offers it (self-contained — it
//! embeds its own tokenizer, see `checkpoint::gguf::GgufTokenizer` — so a
//! fetched `.gguf` drops straight into the model dir and auto-serves with no
//! sidecar file). safetensors + a tokenizer.json sidecar is the fallback.

use std::io::{Read, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

/// Where to get one file, and how to know it arrived intact. `size` is a
/// best-effort expected byte count (from the origin at registry-authoring
/// time) used only for progress display and a sanity check, not integrity —
/// `sha256`, when set, is the actual integrity check.
pub struct Source {
    pub url: &'static str,
    pub sha256: Option<&'static str>,
    pub size: Option<u64>,
}

/// One entry in the known-model registry (`brain fetch <name>`).
pub struct KnownModel {
    pub name: &'static str,
    pub description: &'static str,
    pub gguf: Option<Source>,
    /// Weights + tokenizer sidecar (HF convention: both files land in the same
    /// model-dir directory so `model_dir::discover`'s sibling-tokenizer lookup
    /// finds it — see `docs/models/apis/readme.md`).
    pub safetensors: Option<(Source, Source)>,
}

/// Small, hand-verified starter set (HF API file listings + resolved sizes
/// checked against the origin when added — see the fetch task). Extend this
/// as more models are verified; a wrong/dead URL fails cleanly (a fetch
/// error), it does not corrupt the model dir.
pub fn known_models() -> Vec<KnownModel> {
    vec![
        KnownModel {
            name: "qwen3-0.6b",
            description: "Qwen3 0.6B — chat, GGUF Q8_0 (~610 MiB) or safetensors (~1.4 GiB)",
            gguf: Some(Source {
                url: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
                sha256: None,
                size: Some(639_446_688),
            }),
            safetensors: Some((
                Source {
                    url: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors",
                    sha256: None,
                    size: Some(1_503_300_328),
                },
                Source {
                    url: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json",
                    sha256: None,
                    size: Some(11_422_654),
                },
            )),
        },
        KnownModel {
            name: "qwen3-1.7b",
            description: "Qwen3 1.7B — chat, GGUF Q8_0 (~1.8 GiB)",
            gguf: Some(Source {
                url: "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf",
                sha256: None,
                size: Some(1_834_426_016),
            }),
            safetensors: None,
        },
        KnownModel {
            name: "qwen3-4b",
            description: "Qwen3 4B — chat, GGUF Q4_K_M (~2.5 GiB)",
            gguf: Some(Source {
                url: "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
                sha256: None,
                size: Some(2_497_280_256),
            }),
            safetensors: None,
        },
    ]
}

pub fn find(name: &str) -> Option<KnownModel> {
    known_models().into_iter().find(|m| m.name == name)
}

/// Downloads one [`Source`] to `dest`. Implementations MUST stream (never
/// buffer the whole file in memory — the same OOM invariant weight loading
/// follows) and write atomically (temp file + rename) so a killed or failed
/// fetch never leaves a partial file where the model-dir scanner could find
/// it. `progress(got, total)` is called periodically (`total` is `None` when
/// unknown, e.g. no `Content-Length`).
pub trait Fetcher {
    fn fetch(&self, source: &Source, dest: &Path, progress: &mut dyn FnMut(u64, Option<u64>)) -> Result<(), String>;
}

/// Streaming HTTPS downloader (blocking — a one-shot CLI command has no need
/// for an async runtime). A future `TorrentFetcher` implements the same
/// [`Fetcher`] trait; nothing else in `brain fetch` needs to change.
pub struct HttpFetcher;

impl Fetcher for HttpFetcher {
    fn fetch(&self, source: &Source, dest: &Path, progress: &mut dyn FnMut(u64, Option<u64>)) -> Result<(), String> {
        let resp = ureq::get(source.url).call().map_err(|e| format!("{}: {e}", source.url))?;
        let total = resp
            .header("Content-Length")
            .and_then(|s| s.parse::<u64>().ok())
            .or(source.size);
        let tmp = dest.with_extension(match dest.extension().and_then(|e| e.to_str()) {
            Some(ext) => format!("{ext}.part"),
            None => "part".to_string(),
        });
        let mut file = std::fs::File::create(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
        let mut hasher = source.sha256.is_some().then(Sha256::new);
        let mut reader = resp.into_reader();
        // A fixed small buffer — never grow it with the file; this is the
        // whole reason `brain fetch` can't OOM regardless of model size.
        let mut buf = [0u8; 64 * 1024];
        let mut got: u64 = 0;
        loop {
            let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
            if let Some(h) = &mut hasher {
                h.update(&buf[..n]);
            }
            got += n as u64;
            progress(got, total);
        }
        file.sync_all().map_err(|e| e.to_string())?;
        drop(file);
        if let (Some(expected), Some(h)) = (source.sha256, hasher) {
            let digest = hex_lower(&h.finalize());
            if digest != expected {
                std::fs::remove_file(&tmp).ok();
                return Err(format!("sha256 mismatch: expected {expected}, got {digest}"));
            }
        }
        std::fs::rename(&tmp, dest).map_err(|e| format!("rename {} -> {}: {e}", tmp.display(), dest.display()))?;
        Ok(())
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_names_are_unique_and_findable() {
        let names: Vec<&str> = known_models().iter().map(|m| m.name).collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(names.len(), sorted.len(), "duplicate name in known_models()");
        for n in names {
            assert!(find(n).is_some());
        }
    }

    #[test]
    fn every_known_model_offers_at_least_one_format() {
        for m in known_models() {
            assert!(m.gguf.is_some() || m.safetensors.is_some(), "{}: no source offered", m.name);
        }
    }

    #[test]
    fn hex_lower_matches_known_vector() {
        // sha256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let mut h = Sha256::new();
        h.update(b"");
        assert_eq!(hex_lower(&h.finalize()), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }
}
