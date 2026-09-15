// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One HTTP surface = one provider dialect bound to one socket with its own key.
//!
//! A `brain serve --api` run stands up several surfaces at once (e.g. an Anthropic
//! surface on :8790, an OpenAI surface on :8791). Each has an independent random
//! key so an agent pointed at one dialect can't drive another. Keys are surfaced
//! two ways for test/automation harnesses: a machine-parseable `APIKEY <provider>
//! <key>` line on stdout, and an optional `{provider: key}` JSON file.

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

    /// Print the key on a machine-parseable line (`APIKEY <provider> <key>`) so a
    /// harness that launched brain can scrape it from stdout (P16).
    pub fn announce(&self) {
        println!("APIKEY {} {}", self.provider, self.api_key);
    }
}

/// A fresh `sk-brain-<32 hex>` key.
pub fn random_key() -> String {
    let hi: u64 = rand::random();
    let lo: u64 = rand::random();
    format!("sk-brain-{hi:016x}{lo:016x}")
}

/// `$BRAIN_API_KEY` if the operator set one (a fixed key, reused across
/// every surface/dialect `brain serve` binds - set via a shell export, a
/// `systemd`/container env var, or the CLI's own optional YAML config
/// file), else a fresh [`random_key`] exactly as before this existed.
/// Every dialect that binds gets the SAME key when this is set, unlike
/// the default (each dialect independently random) - that is the whole
/// point of naming a fixed one.
pub fn resolved_key() -> String {
    std::env::var("BRAIN_API_KEY").ok().filter(|s| !s.is_empty()).unwrap_or_else(random_key)
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
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        // `mode(0o600)` applies on creation; `set_permissions` re-tightens a file that
        // already existed with looser bits before we write the secret into it.
        let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(path)?;
        f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        f.write_all(body.as_bytes())
    }
    #[cfg(not(unix))]
    {
        std::fs::write(path, body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BRAIN_API_KEY` set: every call returns that SAME fixed key, not a
    /// fresh random one. `BRAIN_API_KEY` unset: falls back to `random_key`'s
    /// own randomness (two calls differ).
    #[test]
    fn resolved_key_prefers_a_fixed_env_key_over_a_fresh_random_one() {
        let _serial = brain_testutil::env_lock();
        std::env::set_var("BRAIN_API_KEY", "sk-brain-fixed-for-test");
        assert_eq!(resolved_key(), "sk-brain-fixed-for-test");
        assert_eq!(resolved_key(), "sk-brain-fixed-for-test", "must be the same key every call, not regenerated");
        std::env::remove_var("BRAIN_API_KEY");

        assert_ne!(resolved_key(), resolved_key(), "with no fixed key, two calls must fall back to two fresh random ones");
    }

    /// An empty `BRAIN_API_KEY` (e.g. `BRAIN_API_KEY=` inherited from a
    /// wrapper script) must not be treated as "the key is the empty
    /// string" - access control is always on, so an empty key would be a
    /// real hazard, not a harmless default.
    #[test]
    fn resolved_key_treats_an_empty_env_value_as_unset() {
        let _serial = brain_testutil::env_lock();
        std::env::set_var("BRAIN_API_KEY", "");
        assert!(resolved_key().starts_with("sk-brain-"), "an empty BRAIN_API_KEY must fall back to a real random key");
        std::env::remove_var("BRAIN_API_KEY");
    }
}
