// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The remote side of the resolution ladder: listing and downloading files
//! from a model host. [`Hub`] is the seam [`crate::plan`] is tested against -
//! every test in this crate uses [`FakeHub`], never the network. [`HfHub`] is
//! the one real implementation, restricted to a fixed host allowlist so a
//! redirect can never be used to exfiltrate a fetch to an arbitrary origin.

use std::io::Read;
use std::path::Path;

use crate::fetch::stream_to_file;

/// Domains a fetch is ever allowed to touch, including redirect targets,
/// checked as an exact match OR a `.`-delimited suffix (`host == domain ||
/// host.ends_with(".{domain}")`) -- never a bare substring, so
/// `evilhuggingface.co`/`huggingface.co.evil.example` are still refused (see
/// [`host_is_allowed`]'s tests). Both domains are needed: `huggingface.co`
/// itself for the API + the legacy Git-LFS CDN
/// (`cdn-lfs[-us-1].huggingface.co`); `hf.co` for the newer Xet content
/// storage backend large files are served from today (verified live: `GET
/// .../resolve/main/<file>` 302s to `<region>.aws.cdn.hf.co/xet-bridge-
/// <region>/...`, and that response's own `Link`/CORS headers point back at
/// `xethub.hf.co`/`huggingface.co`, confirming common ownership). A per-
/// region subdomain list would need updating every time HF adds a region;
/// the suffix check is what makes that not this crate's problem, while still
/// refusing anything NOT under HF's own domains.
const ALLOWED_HOST_SUFFIXES: &[&str] = &["huggingface.co", "hf.co"];

/// Is `host` exactly one of [`ALLOWED_HOST_SUFFIXES`], or a subdomain of one?
fn host_is_allowed(host: &str) -> bool {
    ALLOWED_HOST_SUFFIXES.iter().any(|domain| host == *domain || host.ends_with(&format!(".{domain}")))
}

/// The caller's HF auth token, if any: `HF_TOKEN` env var first (the
/// standard override every HF client honors), else `~/.cache/huggingface/
/// token` -- written by `huggingface-cli login` / `hf auth login`, so a box
/// where the user has already logged in via the `hf` CLI gets faster,
/// higher-rate-limit fetches with no extra brain-side setup. `None` means
/// anonymous requests, same as before this existed: unauthenticated HF
/// traffic works for public repos, just throttled hard (HF's own `resolve/`
/// response carries an `x-hf-warning` saying so).
///
/// Read fresh on every call rather than cached once -- a fetch can span
/// hours for a large checkpoint, and a login performed after brain started
/// (or a token that gets revoked) should take effect/fail on the NEXT
/// request rather than being frozen at process start.
fn hf_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        let t = t.trim();
        if !t.is_empty() {
            return Some(t.to_string());
        }
    }
    let home = std::env::var("HOME").ok()?;
    let bytes = std::fs::read_to_string(std::path::Path::new(&home).join(".cache/huggingface/token")).ok()?;
    let t = bytes.trim();
    (!t.is_empty()).then(|| t.to_string())
}

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
pub trait Hub: Send + Sync {
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
/// any whose `Location` host is not [`host_is_allowed`]. `ureq`'s built-in
/// redirect handling has no host-inspection hook, so this loop exists
/// specifically to make an off-host redirect a hard error instead of a
/// silent follow.
///
/// With `redirects(0)`, `ureq` hands back a 3xx as `Ok(resp)` (its own docs:
/// "get a response object with the 3xx status code") -- NOT as
/// `Err(ureq::Error::Status(..))`, which only ever fires for a genuinely
/// unsuccessful (4xx/5xx) response. An earlier version of this function
/// checked for the redirect ONLY in the `Err` arm, so it never actually
/// triggered: every redirect fell straight through `Ok(resp) => return
/// Ok(resp)` and handed the caller the 3xx response's OWN body (HF's
/// `resolve/<rev>/<file>` endpoint replies with a plain-text "Temporary
/// Redirect..." page) instead of ever fetching the real file. Caught by
/// actually running the auto-fetch acceptance test end to end against
/// `huggingface.co`, not by the (host-only) unit tests below.
fn get_following_allowed_redirects(url: &str) -> Result<ureq::Response, HubError> {
    let agent = ureq::AgentBuilder::new().redirects(0).build();
    let token = hf_token();
    let mut current = url.to_string();
    for _ in 0..10 {
        let mut req = agent.get(&current);
        // Every hop by this point has already passed `check_redirect_allowed`
        // (the very first request is to `self.base_url`, always
        // huggingface.co) -- the token never reaches a host outside
        // `ALLOWED_HOST_SUFFIXES`, so attaching it unconditionally here is
        // safe, and matches how `huggingface_hub` itself forwards auth
        // across the API -> CDN redirect (needed for a gated/private repo's
        // signed CDN URL to authorize at all).
        if let Some(t) = &token {
            req = req.set("Authorization", &format!("Bearer {t}"));
        }
        match req.call() {
            Ok(resp) if (300..400).contains(&resp.status()) => {
                let location = resp.header("Location").ok_or_else(|| HubError::BadResponse("redirect with no Location header".into()))?.to_string();
                let absolute = resolve_redirect_location(&current, &location)?;
                check_redirect_allowed(&absolute)?;
                current = absolute;
            }
            Ok(resp) => return Ok(resp),
            Err(ureq::Error::Status(404, resp)) => return Ok(resp),
            Err(e) => return Err(HubError::Network(e.to_string())),
        }
    }
    Err(HubError::Network(format!("too many redirects following {url}")))
}

/// Resolve a `Location` header against the URL that produced it, into an
/// absolute URL. HF's own `resolve/<rev>/<file>` -> CDN redirect is
/// ABSOLUTE-PATH relative (`/api/resolve-cache/...`, no scheme/host) per RFC
/// 7231 §7.1.2 -- with automatic redirect-following disabled, nothing
/// resolves that for us the way a browser or `curl -L` would, so this does:
/// an already-absolute `http(s)://` location is used as-is; an absolute-path
/// location (`/...`) resolves against `base`'s origin. Anything else
/// (relative-to-current-path, protocol-relative `//host/...`, etc.) is a
/// hard error rather than a guess -- HF has never sent brain anything else,
/// and guessing wrong here is exactly the kind of thing
/// [`check_redirect_allowed`] exists to not have to reason about.
fn resolve_redirect_location(base: &str, location: &str) -> Result<String, HubError> {
    if location.starts_with("http://") || location.starts_with("https://") {
        return Ok(location.to_string());
    }
    let Some(path) = location.strip_prefix('/') else {
        return Err(HubError::BadResponse(format!("unsupported relative redirect (not absolute-path): {location}")));
    };
    Ok(format!("{}/{path}", url_origin(base)?))
}

/// The host-allowlist check, pulled out as a pure function so it is testable
/// without a live server or a mock HTTP stack.
fn check_redirect_allowed(location: &str) -> Result<(), HubError> {
    let host = url_host(location)?;
    if !host_is_allowed(&host) {
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

/// The `scheme://host` prefix of an absolute `http(s)://` URL.
fn url_origin(url: &str) -> Result<String, HubError> {
    let scheme = if url.starts_with("https://") {
        "https"
    } else if url.starts_with("http://") {
        "http"
    } else {
        return Err(HubError::BadResponse(format!("not an http(s) url: {url}")));
    };
    Ok(format!("{scheme}://{}", url_host(url)?))
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
    fn hf_token_prefers_the_env_var_over_the_token_file() {
        // Same save/mutate/assert/restore discipline as `default_root`'s own
        // test in `lib.rs` -- env vars are process-global, so a test touching
        // them must never leave a real value clobbered for whatever runs
        // next in this process.
        let orig = std::env::var_os("HF_TOKEN");
        unsafe {
            std::env::set_var("HF_TOKEN", "hf_from_env_var");
        }
        assert_eq!(hf_token().as_deref(), Some("hf_from_env_var"));

        unsafe {
            std::env::set_var("HF_TOKEN", "  ");
        }
        // A whitespace-only override must not be mistaken for "set" -- falls
        // through to the token-file lookup (which may itself be `None` on a
        // box with no `hf auth login`, and that's fine to assert either way
        // here since this test only pins the env-var branch's own emptiness
        // check, not the file fallback).
        assert_ne!(hf_token().as_deref(), Some("  "));

        unsafe {
            match orig {
                Some(v) => std::env::set_var("HF_TOKEN", v),
                None => std::env::remove_var("HF_TOKEN"),
            }
        }
    }

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
        // The newer Xet CDN backend (verified live -- see `host_is_allowed`'s
        // doc comment): a `hf.co` subdomain, not `huggingface.co`.
        assert!(check_redirect_allowed("https://us.aws.cdn.hf.co/xet-bridge-us/x").is_ok());
        assert!(check_redirect_allowed("https://cas-server.xethub.hf.co/v1/x").is_ok());
        assert!(check_redirect_allowed("https://evil.example/steal").is_err());
        assert!(check_redirect_allowed("https://huggingface.co.evil.example/x").is_err());
    }

    #[test]
    fn host_is_allowed_matches_the_bare_domain_and_any_subdomain_never_a_substring() {
        for domain in ["huggingface.co", "hf.co"] {
            assert!(host_is_allowed(domain), "{domain}");
            assert!(host_is_allowed(&format!("cdn.{domain}")), "cdn.{domain}");
            assert!(host_is_allowed(&format!("a.b.{domain}")), "a.b.{domain}");
        }
        // A substring match (no '.' separator) must NOT pass -- these are
        // attacker-registerable domains that merely CONTAIN the allowed name.
        assert!(!host_is_allowed("evilhuggingface.co"));
        assert!(!host_is_allowed("notahf.co"));
        assert!(!host_is_allowed("huggingface.co.evil.example"));
        assert!(!host_is_allowed("hf.co.evil.example"));
        assert!(!host_is_allowed("evil.example"));
    }

    #[test]
    fn resolve_redirect_location_keeps_an_already_absolute_url_as_is() {
        assert_eq!(
            resolve_redirect_location("https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json", "https://cdn-lfs.huggingface.co/repos/x").unwrap(),
            "https://cdn-lfs.huggingface.co/repos/x"
        );
    }

    /// Regression for the real bug: HF's `resolve/<rev>/<file>` redirect is
    /// ABSOLUTE-PATH relative (no scheme/host), e.g.
    /// `/api/resolve-cache/models/Qwen/Qwen3-0.6B/<sha>/config.json?...` -
    /// verified against the actual live response. This must resolve onto the
    /// SAME origin as the request that produced it, not be rejected or
    /// mis-treated as a host.
    #[test]
    fn resolve_redirect_location_resolves_an_absolute_path_onto_the_request_origin() {
        let resolved = resolve_redirect_location(
            "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json",
            "/api/resolve-cache/models/Qwen/Qwen3-0.6B/deadbeef/config.json?etag=%22abc%22",
        )
        .unwrap();
        assert_eq!(resolved, "https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3-0.6B/deadbeef/config.json?etag=%22abc%22");
    }

    #[test]
    fn resolve_redirect_location_rejects_a_relative_to_current_path_location() {
        // Not absolute-path (no leading '/') and not an absolute URL -- HF
        // never sends this shape; refuse rather than guess a resolution.
        assert!(resolve_redirect_location("https://huggingface.co/a/b/resolve/main/config.json", "config.json").is_err());
    }

    #[test]
    fn resolve_redirect_location_still_enforces_the_allowlist_after_resolving() {
        // An absolute-path redirect resolves onto the CURRENT request's own
        // origin -- so if that origin were ever something other than an
        // allowed host, resolution must not be what launders it past
        // `check_redirect_allowed`, which runs on the resolved URL right
        // after this in `get_following_allowed_redirects`.
        let resolved = resolve_redirect_location("https://evil.example/x", "/y").unwrap();
        assert!(check_redirect_allowed(&resolved).is_err());
    }
}
