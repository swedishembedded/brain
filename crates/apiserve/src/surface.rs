// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One HTTP surface = one provider dialect bound to one socket with its own key.
//!
//! A `brain serve --api` run stands up several surfaces at once (e.g. an Anthropic
//! surface on :8790, an OpenAI surface on :8791). Each has an independent random
//! key so an agent pointed at one dialect can't drive another. Keys are surfaced
//! two ways for test/automation harnesses: a machine-parseable `APIKEY <provider>
//! <key>` line on stdout, and an optional `{provider: key}` JSON file (with which
//! the stdout line names the file instead of repeating the key).

use std::net::SocketAddr;
use std::path::Path;

use serde_json::{json, Value};

/// The API dialect a surface speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    OpenAI,
    Anthropic,
    OpenRouter,
}

impl Provider {
    /// Stable lowercase tag used in the `APIKEY` line, the keys JSON, and logs.
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::OpenAI => "openai",
            Provider::Anthropic => "anthropic",
            Provider::OpenRouter => "openrouter",
        }
    }
    /// Parse the tag back (the inverse of [`Provider::as_str`]).
    pub fn parse(s: &str) -> Option<Provider> {
        Some(match s {
            "openai" => Provider::OpenAI,
            "anthropic" => Provider::Anthropic,
            "openrouter" => Provider::OpenRouter,
            _ => return None,
        })
    }
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One bound provider surface: dialect + listen address + the key that guards it.
#[derive(Clone, Debug)]
pub struct Surface {
    pub provider: Provider,
    pub addr: SocketAddr,
    pub api_key: String,
}

impl Surface {
    /// A surface with a caller-supplied key.
    pub fn new(provider: Provider, addr: SocketAddr, api_key: impl Into<String>) -> Surface {
        Surface { provider, addr, api_key: api_key.into() }
    }

    /// A surface with a fresh random `sk-brain-<hex>` key (access control is always
    /// on — a key is required, never blank).
    pub fn generate(provider: Provider, addr: SocketAddr) -> Surface {
        Surface { provider, addr, api_key: random_key() }
    }

    /// The stdout line announcing this surface's key. Without a keys file it is the
    /// machine-parseable `APIKEY <provider> <key>` a harness can scrape (P16). With
    /// one (`--api-keys-out`, owner-only) the key is already in the file, so the line
    /// names the provider and the file instead: stdout ends up in server logs that
    /// other users of a shared node can read, and a secret must not.
    pub fn announcement(&self, keys_file: Option<&Path>) -> String {
        match keys_file {
            Some(file) => format!("APIKEY {} written to {}", self.provider, file.display()),
            None => format!("APIKEY {} {}", self.provider, self.api_key),
        }
    }

    /// Print [`Surface::announcement`] on stdout.
    pub fn announce(&self, keys_file: Option<&Path>) {
        println!("{}", self.announcement(keys_file));
    }
}

/// A fresh `sk-brain-<32 hex>` key.
pub fn random_key() -> String {
    let hi: u64 = rand::random();
    let lo: u64 = rand::random();
    format!("sk-brain-{hi:016x}{lo:016x}")
}

/// The key a dialect is served with: the operator's fixed key when one was
/// given (`brain serve --api-key-file`, so every dialect shares it), else a
/// fresh [`random_key`] per dialect per launch.
pub fn resolved_key(fixed: Option<&str>) -> String {
    fixed.map_or_else(random_key, str::to_string)
}

/// Read a fixed key from `path`: its contents with surrounding whitespace
/// trimmed. An empty file or one whose key holds whitespace is refused - a
/// blank key would be a server nobody can reach, and a key with a space in it
/// cannot travel in an `Authorization` header.
pub fn read_key_file(path: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let key = text.trim();
    if key.is_empty() {
        return Err(format!("{} holds no key", path.display()));
    }
    if key.chars().any(char::is_whitespace) {
        return Err(format!("{} holds more than one word; a key is one", path.display()));
    }
    Ok(key.to_string())
}

/// The URL a client on this host reaches a listener on: an unspecified bind
/// (`0.0.0.0`, `::`) is reached through the loopback of its family.
pub fn local_url(addr: SocketAddr) -> String {
    let ip = match addr.ip() {
        std::net::IpAddr::V4(ip) if ip.is_unspecified() => std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
        std::net::IpAddr::V6(ip) if ip.is_unspecified() => std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST),
        ip => ip,
    };
    format!("http://{}", SocketAddr::new(ip, addr.port()))
}

/// The connection record of a running server: what a client on this host
/// needs to use it without being told anything. `{"pid": N, "surfaces":
/// [{"provider": "openai", "url": "http://127.0.0.1:8788", "key": "sk-brain-…"}]}`,
/// one entry per bound address, `url` the address the listener actually holds
/// (never a requested `:0`). The OpenAI and OpenRouter routes answer under
/// `url` and under `url/v1`; the Anthropic ones under `url/v1` only.
pub fn record_json(pid: u32, bound: &[(Provider, SocketAddr, String)]) -> Value {
    let surfaces: Vec<Value> = bound.iter().map(|(provider, addr, key)| json!({"provider": provider.as_str(), "url": local_url(*addr), "key": key})).collect();
    json!({"pid": pid, "surfaces": surfaces})
}

/// The `{provider: key}` map for a set of surfaces.
pub fn keys_json(surfaces: &[Surface]) -> Value {
    let mut m = serde_json::Map::new();
    for s in surfaces {
        m.insert(s.provider.as_str().to_string(), json!(s.api_key));
    }
    Value::Object(m)
}

/// Write the `{provider: key}` JSON for `surfaces` to `path` (the `--api-keys-out`
/// file). Pretty-printed so it is greppable. The file holds live API keys, so on Unix
/// it is created (and, if it pre-existed, re-tightened) with owner-only `0600`
/// permissions — never world-readable.
pub fn write_keys(surfaces: &[Surface], path: &Path) -> std::io::Result<()> {
    let body = serde_json::to_string_pretty(&keys_json(surfaces)).unwrap_or_else(|_| "{}".into());
    write_private(path, body.as_bytes())
}

/// Make `dir` a directory only this user can write into: created owner-only
/// when missing, refused when it exists but belongs to another user or others
/// may write to it. A client trusts the URL a connection record names, so a
/// record must never sit where someone else could plant or replace one (the
/// per-user temp-dir fallback lives in a shared `/tmp`).
pub fn private_dir(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, MetadataExt};
        std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir)?;
        let meta = std::fs::symlink_metadata(dir)?;
        // SAFETY: geteuid has no preconditions and cannot fail.
        let me = unsafe { libc::geteuid() };
        if !meta.is_dir() || meta.uid() != me || meta.mode() & 0o022 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("{} is not a directory owned by this user and closed to others", dir.display()),
            ));
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(dir)
    }
}

/// Write `body` to `path` readable by its owner only. A secret is never
/// visible to anyone else, not even for the moment between create and chmod:
/// on Unix the file is written aside with mode `0600` and renamed over `path`,
/// so a reader sees the old file or the whole new one.
pub fn write_private(path: &Path, body: &[u8]) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let mut aside = path.as_os_str().to_os_string();
        aside.push(format!(".tmp-{}", std::process::id()));
        let aside = std::path::PathBuf::from(aside);
        let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&aside)?;
        // `mode` applies on creation only; a leftover aside file keeps its
        // own bits until they are tightened here.
        f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        // No fsync: these files live as long as the server, in a runtime
        // directory a reboot clears, so the rename's atomicity is all a
        // reader needs; an fsync per server start stalls a loaded overlay
        // filesystem for everything else writing to it.
        f.write_all(body)?;
        std::fs::rename(&aside, path)
    }
    #[cfg(not(unix))]
    {
        std::fs::write(path, body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed key is every dialect's key; with none, each call draws a
    /// fresh one.
    #[test]
    fn resolved_key_is_the_fixed_key_else_a_fresh_random_one() {
        assert_eq!(resolved_key(Some("sk-brain-fixed")), "sk-brain-fixed");
        assert_ne!(resolved_key(None), resolved_key(None), "with no fixed key, two calls draw two keys");
    }

    /// With a keys file the stdout line must not carry the key (logs are group
    /// readable on a shared node); without one it still does, so a quickstart
    /// can copy it.
    #[test]
    fn the_announcement_omits_the_key_once_it_is_in_a_file() {
        let s = Surface::new(Provider::OpenAI, "127.0.0.1:1".parse().unwrap(), "sk-brain-secret");
        let line = s.announcement(Some(Path::new("/run/keys.json")));
        assert!(!line.contains("sk-brain-secret"), "{line}");
        assert!(line.contains(s.provider.as_str()) && line.contains("/run/keys.json"), "{line}");
        assert_eq!(s.announcement(None), format!("APIKEY {} sk-brain-secret", s.provider));
    }

    /// A key file yields its one word; an empty or multi-word file is
    /// refused, since access control is always on and a blank key is no key.
    #[test]
    fn a_key_file_holds_exactly_one_word() {
        let dir = std::env::temp_dir().join(format!("brain-key-file-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("key");
        std::fs::write(&file, "  sk-brain-fixed\n").unwrap();
        assert_eq!(read_key_file(&file).as_deref(), Ok("sk-brain-fixed"));
        for bad in ["", " \n", "sk-brain-a sk-brain-b"] {
            std::fs::write(&file, bad).unwrap();
            assert!(read_key_file(&file).is_err(), "{bad:?} must be refused");
        }
        assert!(read_key_file(&dir.join("absent")).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The record names where a local client connects: an unspecified bind is
    /// reached on loopback, any other address as bound.
    #[test]
    fn the_record_names_a_local_url_per_bound_address() {
        let bound = vec![
            (Provider::OpenAI, "0.0.0.0:8788".parse().unwrap(), "sk-a".to_string()),
            (Provider::Anthropic, "[::]:8787".parse().unwrap(), "sk-b".to_string()),
            (Provider::OpenRouter, "192.168.1.5:8789".parse().unwrap(), "sk-c".to_string()),
        ];
        let record = record_json(42, &bound);
        assert_eq!(record["pid"], 42);
        let urls: Vec<&str> = record["surfaces"].as_array().unwrap().iter().map(|s| s["url"].as_str().unwrap()).collect();
        assert_eq!(urls, ["http://127.0.0.1:8788", "http://[::1]:8787", "http://192.168.1.5:8789"]);
        assert_eq!(record["surfaces"][1]["provider"], "anthropic");
        assert_eq!(record["surfaces"][2]["key"], "sk-c");
    }

    /// A record directory is created closed to others, and one others may
    /// write into is refused.
    #[cfg(unix)]
    #[test]
    fn a_record_directory_is_one_only_this_user_writes() {
        use std::os::unix::fs::PermissionsExt;
        let root = std::env::temp_dir().join(format!("brain-private-dir-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        let dir = root.join("run");
        private_dir(&dir).unwrap();
        assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(private_dir(&dir).is_err(), "a world-writable directory is refused");
        std::fs::remove_dir_all(&root).ok();
    }
}
