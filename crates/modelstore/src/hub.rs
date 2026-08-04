// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The remote side of the resolution ladder: listing and downloading files
//! from a model host. [`Hub`] is the seam [`crate::plan`] is tested against —
//! every test in this crate uses [`FakeHub`], never the network. [`HfHub`] is
//! the one real implementation, restricted to a fixed host allowlist so a
//! redirect can never be used to exfiltrate a fetch to an arbitrary origin.

use std::io::Read;
use std::path::Path;

use crate::fetch::stream_to_file;

/// Hosts a fetch is ever allowed to touch, including redirect targets. HF
/// serves large files from a CDN subdomain reached via a 3xx from the API
/// host, so the allowlist must cover both -- but nothing else.
const ALLOWED_HOSTS: &[&str] = &["huggingface.co", "cdn-lfs.huggingface.co", "cdn-lfs-us-1.huggingface.co"];

#[derive(Debug)]
pub enum HubError {
    NotFound(String),
    Network(String),
    BadResponse(String),
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HubError::NotFound(s) => write!(f, "not found: {s}"),
            HubError::Network(s) => write!(f, "network error: {s}"),
            HubError::BadResponse(s) => write!(f, "bad response: {s}"),
        }
    }
}

impl std::error::Error for HubError {}

/// A model host: list what a repo offers, read a small file whole, or stream
/// a large one to disk. `revision` is always a branch/tag/sha the caller
/// supplies; [`resolve_revision`](Hub::resolve_revision) pins it to a sha for
/// provenance recording, but callers are free to pass a sha directly and skip
/// that call.
pub trait Hub {
    fn resolve_revision(&self, vendor: &str, repo: &str, revision: &str) -> Result<String, HubError>;
    fn list_files(&self, vendor: &str, repo: &str, revision: &str) -> Result<Vec<String>, HubError>;
    /// Reads one file fully into memory -- only for small files (`config.json`,
    /// index manifests). Weight files use [`download`](Hub::download).
    fn read_file(&self, vendor: &str, repo: &str, revision: &str, file: &str) -> Result<Vec<u8>, HubError>;
    fn download(
        &self,
        vendor: &str,
        repo: &str,
        revision: &str,
        file: &str,
        dest: &Path,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(), HubError>;
}

/// The real Hub, talking to Hugging Face's model hub over HTTPS.
pub struct HfHub {
    base_url: String,
}

impl HfHub {
    pub fn new() -> HfHub {
        HfHub { base_url: "https://huggingface.co".to_string() }
    }

    fn api_url(&self, vendor: &str, repo: &str, revision: &str) -> String {
        format!("{}/api/models/{vendor}/{repo}/revision/{revision}", self.base_url)
    }

    fn resolve_url(&self, vendor: &str, repo: &str, revision: &str, file: &str) -> String {
        format!("{}/{vendor}/{repo}/resolve/{revision}/{file}", self.base_url)
    }

    fn revision_info(&self, vendor: &str, repo: &str, revision: &str) -> Result<serde_json::Value, HubError> {
        let url = self.api_url(vendor, repo, revision);
        let resp = get_following_allowed_redirects(&url)?;
        if resp.status() == 404 {
            return Err(HubError::NotFound(format!("{vendor}/{repo}@{revision}")));
        }
        resp.into_json().map_err(|e| HubError::BadResponse(e.to_string()))
    }
}

impl Default for HfHub {
    fn default() -> Self {
        Self::new()
    }
}

impl Hub for HfHub {
    fn resolve_revision(&self, vendor: &str, repo: &str, revision: &str) -> Result<String, HubError> {
        let info = self.revision_info(vendor, repo, revision)?;
        info.get("sha")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .ok_or_else(|| HubError::BadResponse("revision info missing \"sha\"".into()))
    }

    fn list_files(&self, vendor: &str, repo: &str, revision: &str) -> Result<Vec<String>, HubError> {
        let info = self.revision_info(vendor, repo, revision)?;
        let siblings = info
            .get("siblings")
            .and_then(|v| v.as_array())
            .ok_or_else(|| HubError::BadResponse("revision info missing \"siblings\"".into()))?;
        Ok(siblings.iter().filter_map(|s| s.get("rfilename").and_then(|v| v.as_str()).map(str::to_string)).collect())
    }

    fn read_file(&self, vendor: &str, repo: &str, revision: &str, file: &str) -> Result<Vec<u8>, HubError> {
        let url = self.resolve_url(vendor, repo, revision, file);
        let resp = get_following_allowed_redirects(&url)?;
        if resp.status() == 404 {
            return Err(HubError::NotFound(format!("{vendor}/{repo}@{revision}/{file}")));
        }
        let mut buf = Vec::new();
        resp.into_reader().read_to_end(&mut buf).map_err(|e| HubError::Network(e.to_string()))?;
        Ok(buf)
    }

    fn download(
        &self,
        vendor: &str,
        repo: &str,
        revision: &str,
        file: &str,
        dest: &Path,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(), HubError> {
        let url = self.resolve_url(vendor, repo, revision, file);
        let resp = get_following_allowed_redirects(&url)?;
        if resp.status() == 404 {
            return Err(HubError::NotFound(format!("{vendor}/{repo}@{revision}/{file}")));
        }
        let total = resp.header("Content-Length").and_then(|s| s.parse::<u64>().ok());
        stream_to_file(resp.into_reader(), dest, total, None, progress).map_err(|e| HubError::Network(e.to_string()))
    }
}

/// GETs `url`, manually following redirects one hop at a time and refusing
/// any whose `Location` host is not in [`ALLOWED_HOSTS`]. `ureq`'s built-in
/// redirect handling has no host-inspection hook, so this loop exists
/// specifically to make an off-host redirect a hard error instead of a
/// silent follow.
fn get_following_allowed_redirects(url: &str) -> Result<ureq::Response, HubError> {
    let agent = ureq::AgentBuilder::new().redirects(0).build();
    let mut current = url.to_string();
    for _ in 0..10 {
        match agent.get(&current).call() {
            Ok(resp) => return Ok(resp),
            Err(ureq::Error::Status(code, resp)) if (300..400).contains(&code) => {
                let location = resp.header("Location").ok_or_else(|| HubError::BadResponse("redirect with no Location header".into()))?;
                check_redirect_allowed(location)?;
                current = location.to_string();
            }
            Err(ureq::Error::Status(404, resp)) => return Ok(resp),
            Err(e) => return Err(HubError::Network(e.to_string())),
        }
    }
    Err(HubError::Network(format!("too many redirects following {url}")))
}

/// The host-allowlist check, pulled out as a pure function so it is testable
/// without a live server or a mock HTTP stack.
fn check_redirect_allowed(location: &str) -> Result<(), HubError> {
    let host = url_host(location)?;
    if !ALLOWED_HOSTS.contains(&host.as_str()) {
        return Err(HubError::Network(format!("refused off-host redirect to {host}")));
    }
    Ok(())
}

fn url_host(url: &str) -> Result<String, HubError> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .ok_or_else(|| HubError::BadResponse(format!("redirect to a non-http(s) url: {url}")))?;
    let host = rest.split(['/', ':']).next().unwrap_or("");
    if host.is_empty() {
        return Err(HubError::BadResponse(format!("redirect url has no host: {url}")));
    }
    Ok(host.to_string())
}

/// An in-memory [`Hub`] for tests: no network, no filesystem beyond writing
/// the destination `download` is asked for. Every [`crate::plan`] test uses
/// this, never [`HfHub`].
#[derive(Default)]
pub struct FakeHub {
    // (vendor, repo, revision) -> file name -> bytes.
    files: std::collections::HashMap<(String, String, String), std::collections::HashMap<String, Vec<u8>>>,
}

impl FakeHub {
    pub fn new() -> FakeHub {
        FakeHub::default()
    }

    /// Registers one file's content under `vendor/repo@revision`. Repeated
    /// calls for the same repo/revision accumulate into one file listing.
    pub fn add_file(&mut self, vendor: &str, repo: &str, revision: &str, file: &str, bytes: impl Into<Vec<u8>>) -> &mut Self {
        self.files
            .entry((vendor.to_string(), repo.to_string(), revision.to_string()))
            .or_default()
            .insert(file.to_string(), bytes.into());
        self
    }

    fn repo(&self, vendor: &str, repo: &str, revision: &str) -> Result<&std::collections::HashMap<String, Vec<u8>>, HubError> {
        self.files
            .get(&(vendor.to_string(), repo.to_string(), revision.to_string()))
            .ok_or_else(|| HubError::NotFound(format!("{vendor}/{repo}@{revision}")))
    }
}

impl Hub for FakeHub {
    fn resolve_revision(&self, vendor: &str, repo: &str, revision: &str) -> Result<String, HubError> {
        self.repo(vendor, repo, revision)?;
        Ok(format!("fake-sha-{revision}"))
    }

    fn list_files(&self, vendor: &str, repo: &str, revision: &str) -> Result<Vec<String>, HubError> {
        Ok(self.repo(vendor, repo, revision)?.keys().cloned().collect())
    }

    fn read_file(&self, vendor: &str, repo: &str, revision: &str, file: &str) -> Result<Vec<u8>, HubError> {
        self.repo(vendor, repo, revision)?
            .get(file)
            .cloned()
            .ok_or_else(|| HubError::NotFound(format!("{vendor}/{repo}@{revision}/{file}")))
    }

    fn download(
        &self,
        vendor: &str,
        repo: &str,
        revision: &str,
        file: &str,
        dest: &Path,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(), HubError> {
        let bytes = self.read_file(vendor, repo, revision, file)?;
        stream_to_file(&bytes[..], dest, Some(bytes.len() as u64), None, progress).map_err(|e| HubError::Network(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_hub_lists_and_reads_registered_files() {
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", b"{}".to_vec());
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", vec![1, 2, 3]);

        let mut files = hub.list_files("Qwen", "Qwen3-0.6B", "main").unwrap();
        files.sort();
        assert_eq!(files, vec!["config.json".to_string(), "model.safetensors".to_string()]);
        assert_eq!(hub.read_file("Qwen", "Qwen3-0.6B", "main", "config.json").unwrap(), b"{}");
    }

    #[test]
    fn fake_hub_unknown_repo_is_not_found() {
        let hub = FakeHub::new();
        assert!(matches!(hub.list_files("nobody", "nothing", "main"), Err(HubError::NotFound(_))));
    }

    #[test]
    fn fake_hub_download_streams_to_dest() {
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B-GGUF", "main", "Qwen3-0.6B-Q8_0.gguf", vec![9, 9, 9]);
        let dir = std::env::temp_dir().join("modelstore-hub-test-download");
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("Q8_0.gguf");
        hub.download("Qwen", "Qwen3-0.6B-GGUF", "main", "Qwen3-0.6B-Q8_0.gguf", &dest, &mut |_, _| {}).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), vec![9, 9, 9]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn url_host_extracts_host_from_https_and_rejects_non_http() {
        assert_eq!(url_host("https://cdn-lfs.huggingface.co/repos/abc").unwrap(), "cdn-lfs.huggingface.co");
        assert_eq!(url_host("http://huggingface.co:443/x").unwrap(), "huggingface.co");
        assert!(url_host("ftp://evil.example/x").is_err());
        assert!(url_host("https://").is_err());
    }

    #[test]
    fn redirect_allowlist_accepts_known_hosts_and_refuses_others() {
        assert!(check_redirect_allowed("https://huggingface.co/x").is_ok());
        assert!(check_redirect_allowed("https://cdn-lfs.huggingface.co/x").is_ok());
        assert!(check_redirect_allowed("https://cdn-lfs-us-1.huggingface.co/x").is_ok());
        assert!(check_redirect_allowed("https://evil.example/steal").is_err());
        assert!(check_redirect_allowed("https://huggingface.co.evil.example/x").is_err());
    }
}
