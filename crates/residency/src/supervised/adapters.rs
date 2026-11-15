// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving fine-tuned model
// variants from isolated worker processes for its clients. If your team needs
// expertise in hot-swapping adapters across process boundaries then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The adapters a hosted base offers, both ends: what the serving process
//! offers ([`AdapterSource`]), the registry a worker rebuilds from it
//! ([`AdapterSink`]), and the parent's record of what its worker was told
//! ([`AdapterFeed`]), from which the `attach` and `detach` requests and frames
//! are made.
//!
//! The serving process owns the adapter registry (what an operator offers,
//! the views registered with the executor, their leases). A worker hosting
//! the base rebuilds its own registry from what it is told: every change
//! reaches it before the next group starts (as `attach`/`detach` requests),
//! and while a group runs as frames its reader applies at once, sent before
//! the parent's reply to the worker's next `take`. A job that joins a group
//! after its adapter was offered therefore never reaches the worker before
//! the adapter does, and the worker's group refreshes its routes as the
//! in-process group does (when it next goes idle).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{json, Value};

/// One adapter a hosted base offers: its serving id, the digest of its file's
/// bytes, the file a worker reads it from (the release the serving process
/// verified), and the version the serving process numbered the release under
/// its id, which a reply names (a worker started after a release was replaced
/// has seen only the new one, and must still call it by the serving process's
/// number). The worker checks the digest of what it reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostedAdapter {
    pub id: String,
    pub digest: String,
    pub path: PathBuf,
    pub version: u64,
}

/// The serving process's side: the adapters a hosted base offers now.
pub trait AdapterSource: Send + Sync {
    /// A number that changes whenever the offered set does: one atomic load
    /// tells a caller nothing changed.
    fn generation(&self) -> u64;

    /// The offered set, with a generation at least as new as it.
    fn offered(&self) -> (u64, Vec<HostedAdapter>);
}

/// The worker's side: the hosted model's own adapter registry, rebuilt from
/// what its parent offers. Called from the worker's reader thread while a
/// group runs, so it is shared state the model synchronises.
pub trait AdapterSink: Send + Sync {
    /// Offer `adapter` under its id, replacing what was offered under it. An
    /// error (a file whose digest is not the one named) leaves the id
    /// unoffered: a request naming it is answered as one naming no model.
    fn attach(&self, adapter: &HostedAdapter) -> Result<(), String>;

    /// Stop offering `id`; nothing when it is not offered.
    fn detach(&self, id: &str);
}

/// One change a worker is told about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Change {
    Attach(HostedAdapter),
    Detach(String),
}

impl Change {
    /// The request or frame that carries it: `attach <json>` or `detach <id>`.
    pub(super) fn line(&self) -> String {
        match self {
            Change::Attach(a) => format!("attach {}", json!({ "id": a.id, "digest": a.digest, "path": a.path.to_string_lossy(), "version": a.version })),
            Change::Detach(id) => format!("detach {id}"),
        }
    }

    /// What [`Self::line`] wrote, without its verb: `verb` is `attach` or `detach`.
    pub(super) fn parse(verb: &str, rest: &str) -> Result<Change, String> {
        match verb {
            "attach" => {
                let v: Value = serde_json::from_str(rest).map_err(|e| format!("attach: not JSON ({e})"))?;
                let map = v.as_object().ok_or("an adapter is not an object")?;
                if let Some(k) = map.keys().find(|k| !["id", "digest", "path", "version"].contains(&k.as_str())) {
                    return Err(format!("an adapter has an unknown field {k:?}"));
                }
                let field = |k: &str| map.get(k).and_then(Value::as_str).map(str::to_string).ok_or_else(|| format!("an adapter without its {k}"));
                let version = map.get("version").and_then(Value::as_u64).ok_or("an adapter without its version")?;
                Ok(Change::Attach(HostedAdapter { id: field("id")?, digest: field("digest")?, path: PathBuf::from(field("path")?), version }))
            }
            "detach" if !rest.is_empty() && !rest.contains(char::is_whitespace) => Ok(Change::Detach(rest.to_string())),
            _ => Err(format!("{verb} {rest:?} is not an adapter change")),
        }
    }

    /// Apply it to the worker's registry.
    pub(super) fn apply(&self, sink: &dyn AdapterSink) -> Result<(), String> {
        match self {
            Change::Attach(a) => sink.attach(a).map_err(|e| format!("adapter {}: {e}", a.id)),
            Change::Detach(id) => {
                sink.detach(id);
                Ok(())
            }
        }
    }
}

/// What the parent told one worker, against what its base offers: the
/// changes that bring the worker in line, each handed out once.
pub(super) struct AdapterFeed {
    source: Arc<dyn AdapterSource>,
    /// The generation the worker was brought in line with; `None` for a
    /// worker told nothing yet.
    seen: Option<u64>,
    /// What the worker was told is offered, by id.
    told: BTreeMap<String, HostedAdapter>,
}

impl AdapterFeed {
    pub(super) fn new(source: Arc<dyn AdapterSource>) -> AdapterFeed {
        AdapterFeed { source, seen: None, told: BTreeMap::new() }
    }

    /// A new worker knows nothing: everything offered is told again.
    pub(super) fn reset(&mut self) {
        self.seen = None;
        self.told.clear();
    }

    /// The changes since the last call, detaches first; empty (one atomic
    /// load) when the source did not change. Once handed out they count as
    /// told.
    pub(super) fn changes(&mut self) -> Vec<Change> {
        if self.seen == Some(self.source.generation()) {
            return Vec::new();
        }
        let (generation, offered) = self.source.offered();
        let offered: BTreeMap<String, HostedAdapter> = offered.into_iter().map(|a| (a.id.clone(), a)).collect();
        let mut changes: Vec<Change> = self.told.keys().filter(|id| !offered.contains_key(*id)).map(|id| Change::Detach(id.clone())).collect();
        changes.extend(offered.values().filter(|a| self.told.get(&a.id) != Some(a)).map(|a| Change::Attach(a.clone())));
        self.told = offered;
        self.seen = Some(generation);
        changes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Source(Mutex<(u64, Vec<HostedAdapter>)>);

    impl AdapterSource for Source {
        fn generation(&self) -> u64 {
            self.0.lock().unwrap().0
        }
        fn offered(&self) -> (u64, Vec<HostedAdapter>) {
            self.0.lock().unwrap().clone()
        }
    }

    fn adapter(id: &str, digest: &str) -> HostedAdapter {
        HostedAdapter { id: id.into(), digest: digest.into(), path: PathBuf::from(format!("/adapters/{id}.safetensors")), version: 1 }
    }

    /// The feed tells a worker each change once, a replaced release as a new
    /// attach and a withdrawn one as a detach, and everything again after a
    /// reset; each change survives its line.
    #[test]
    fn a_worker_is_told_each_adapter_change_once() {
        let source = Arc::new(Source(Mutex::new((1, vec![adapter("a", "d1"), adapter("b", "d2")]))));
        let mut feed = AdapterFeed::new(source.clone());
        assert_eq!(feed.changes(), [Change::Attach(adapter("a", "d1")), Change::Attach(adapter("b", "d2"))]);
        assert!(feed.changes().is_empty(), "nothing changed");
        *source.0.lock().unwrap() = (2, vec![adapter("b", "d3")]);
        assert_eq!(feed.changes(), [Change::Detach("a".into()), Change::Attach(adapter("b", "d3"))]);
        feed.reset();
        assert_eq!(feed.changes(), [Change::Attach(adapter("b", "d3"))]);
        for change in [Change::Attach(adapter("x:y", "sha256:00")), Change::Attach(HostedAdapter { version: 7, ..adapter("x:y", "sha256:01") }), Change::Detach("x:y".into())] {
            let line = change.line();
            let (verb, rest) = line.split_once(' ').unwrap();
            assert_eq!(Change::parse(verb, rest).unwrap(), change);
        }
        assert!(Change::parse("detach", "a b").is_err(), "an id is one word");
        assert!(Change::parse("attach", r#"{"id": "a"}"#).is_err(), "an adapter names its digest and file");
        assert!(Change::parse("attach", r#"{"id": "a", "digest": "d", "path": "/a"}"#).is_err(), "an adapter names the version its release is numbered");
    }
}
