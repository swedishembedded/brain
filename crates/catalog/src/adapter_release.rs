// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verified release of fine-tuned model
// adapters into live inference servers for its clients. If your team needs
// expertise in serving and rolling out fine-tuned language models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Which LoRA adapter release `brain serve` puts in front of its Qwen3 base,
//! and the checks that make "this is what is served" a verified statement
//! rather than a file name.
//!
//! An [`AdapterRelease`] is an adapter file that has been read as a Qwen3
//! LoRA, bound to the served base, and named by its content digest -
//! `brain_modelstore::fetch::file_digest`, the same string a fine-tune
//! reports as its adapter digest and a chat pipeline reports as its
//! identity. [`verify_adapter`] produces one; nothing else does.
//!
//! **Binding to the base.** An adapter is only meaningful folded into the
//! exact base it was trained against. When its card records that base's
//! digest (`TrainingProvenance::base_digest`), the served base must hash to
//! it. An adapter whose card records no base digest (one written before the
//! field existed, or by a trainer that does not record it) is bound by its
//! card's base id instead, which is weaker: two different files can carry
//! the same id.
//!
//! **Following a release manifest.** [`ManifestFollower`] re-reads a small
//! JSON file, `{"adapter": "<path>", "digest": "sha256:<hex>"}`, that a
//! release step replaces atomically (write beside it, then rename over it).
//! A new manifest yields a release only after the named file hashes to the
//! digest the manifest states AND the file verifies against the base; a
//! manifest that fails any check is reported once and yields nothing, so the
//! caller keeps serving what it had. The adapter file must not change after
//! its manifest is published: it is verified when the manifest changes and
//! folded when the model is next loaded, so a file rewritten in between is
//! served unverified. Publish each release under a new file name.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The base checkpoint a server folds adapters into: its file, the id it
/// was resolved under (when it came from `BRAIN_QWEN_WEIGHTS`), and its
/// digest, hashed at most once and only when an adapter asks for it (the
/// base is gigabytes; an adapter bound by id never needs it hashed).
pub struct ServedBase {
    path: PathBuf,
    id: Option<String>,
    digest: OnceLock<Result<String, String>>,
}

impl ServedBase {
    pub fn new(path: impl Into<PathBuf>, id: Option<String>) -> ServedBase {
        ServedBase { path: path.into(), id, digest: OnceLock::new() }
    }

    fn digest(&self) -> Result<&str, String> {
        self.digest
            .get_or_init(|| brain_modelstore::fetch::file_digest(&self.path).map_err(|e| format!("the served base {}: {e}", self.path.display())))
            .as_deref()
            .map_err(Clone::clone)
    }
}

/// A verified adapter: what `brain serve` reports it serves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterRelease {
    /// The id on the adapter's own card.
    pub id: String,
    pub path: PathBuf,
    /// `sha256:<hex>` of the file.
    pub digest: String,
}

impl AdapterRelease {
    /// The line `brain serve` prints when `model` starts serving this
    /// release, the one a caller greps to confirm what is served.
    pub fn serving_line(&self, model: &str) -> String {
        format!("brain serve: {model} adapter={} digest={}", self.id, self.digest)
    }
}

/// Read `path` as a Qwen3 LoRA adapter, bind it to `base` and digest it.
/// Every refusal names the file and what was wrong with it; a base mismatch
/// names both digests.
pub fn verify_adapter(path: &Path, base: &ServedBase) -> Result<AdapterRelease, String> {
    let shown = path.display();
    let path_str = path.to_str().ok_or_else(|| format!("adapter {shown}: not a UTF-8 path"))?;
    let digest = brain_modelstore::fetch::file_digest(path).map_err(|e| format!("adapter {shown}: {e}"))?;
    let st = checkpoint::st::load_safetensors(path_str).map_err(|e| format!("adapter {shown}: {e}"))?;
    let card = st.card().ok_or_else(|| format!("adapter {shown}: no model card, so nothing says which base it belongs to"))?;
    let adapter = card.adapter.as_ref().ok_or_else(|| format!("adapter {shown}: its card ({}) describes a model, not an adapter", card.id))?;
    // The serving fold implements exactly this family and kind; anything
    // else would only fail later, at the first request.
    if card.family != "qwen" || adapter.kind != "lora" {
        return Err(format!("adapter {shown}: a {} {:?} adapter; the served base is a Qwen3 that folds LoRA adapters", card.family, adapter.kind));
    }
    if !st.tensors.keys().any(|name| model::adapter::device::is_adapter_param(name)) {
        return Err(format!("adapter {shown}: holds no .lora_a/.lora_b tensors"));
    }

    let recorded_base_digest = card.training.as_ref().and_then(|t| t.base_digest.as_deref());
    match (recorded_base_digest, adapter.base.as_deref()) {
        (Some(trained_on), _) => {
            let served = base.digest()?;
            if trained_on != served {
                return Err(format!(
                    "adapter {shown} was trained against base {trained_on}, but the served base {} is {served}; refusing to fold it into a different base",
                    base.path.display()
                ));
            }
        }
        (None, Some(base_id)) => {
            if base.id.as_deref() != Some(base_id) {
                let served = base.digest()?;
                return Err(format!(
                    "adapter {shown} names base {base_id:?} and records no base digest, but the served base {} is {:?} ({served}); refusing to fold it into a different base",
                    base.path.display(),
                    base.id.as_deref().unwrap_or("unnamed")
                ));
            }
        }
        (None, None) => return Err(format!("adapter {shown}: its card names no base (neither a base digest nor a base id)")),
    }
    Ok(AdapterRelease { id: card.id, path: path.to_path_buf(), digest })
}

/// The release manifest's shape. Unknown fields are refused: a manifest
/// carrying a field this reader does not know was written for a different
/// reader, and guessing what it meant would serve the wrong thing.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    adapter: String,
    digest: String,
}

/// What one look at the manifest found.
#[derive(Debug, PartialEq, Eq)]
pub enum ManifestPoll {
    /// Nothing new since the last look, or a new manifest naming the release
    /// already served.
    Unchanged,
    /// No manifest file (yet).
    Missing,
    /// A new manifest that failed a check; the reason says which. The
    /// previous release, if any, stays served.
    Rejected(String),
    /// A new, verified release to serve.
    Release(AdapterRelease),
}

/// What the manifest path held at the last look, so an unchanged file costs
/// one read and a bad one is reported once rather than every poll.
#[derive(PartialEq, Eq)]
enum Seen {
    Missing,
    Unreadable(String),
    Bytes(Vec<u8>),
}

/// Follows one release manifest. See this module's doc.
pub struct ManifestFollower {
    path: PathBuf,
    base: ServedBase,
    seen: Option<Seen>,
    serving: Option<String>,
}

impl ManifestFollower {
    pub fn new(path: impl Into<PathBuf>, base: ServedBase) -> ManifestFollower {
        ManifestFollower { path: path.into(), base, seen: None, serving: None }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Look at the manifest once.
    pub fn poll(&mut self) -> ManifestPoll {
        let seen = match std::fs::read(&self.path) {
            Ok(bytes) => Seen::Bytes(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Seen::Missing,
            Err(e) => Seen::Unreadable(e.to_string()),
        };
        if self.seen.as_ref() == Some(&seen) {
            return ManifestPoll::Unchanged;
        }
        let outcome = match &seen {
            Seen::Missing => ManifestPoll::Missing,
            Seen::Unreadable(e) => ManifestPoll::Rejected(format!("manifest {}: {e}", self.path.display())),
            Seen::Bytes(bytes) => match self.release_of(bytes) {
                Ok(release) if self.serving.as_deref() == Some(release.digest.as_str()) => ManifestPoll::Unchanged,
                Ok(release) => {
                    self.serving = Some(release.digest.clone());
                    ManifestPoll::Release(release)
                }
                Err(why) => ManifestPoll::Rejected(why),
            },
        };
        self.seen = Some(seen);
        outcome
    }

    fn release_of(&self, bytes: &[u8]) -> Result<AdapterRelease, String> {
        let shown = self.path.display();
        let manifest: Manifest = serde_json::from_slice(bytes).map_err(|e| format!("manifest {shown}: {e}"))?;
        let hex = manifest.digest.strip_prefix("sha256:").unwrap_or_default();
        if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            return Err(format!("manifest {shown}: digest {:?} is not sha256:<64 lowercase hex digits>", manifest.digest));
        }
        // A relative adapter path is relative to the manifest, so a release
        // directory can be moved as a whole.
        let named = PathBuf::from(&manifest.adapter);
        let adapter = if named.is_relative() { self.path.parent().unwrap_or_else(|| Path::new(".")).join(named) } else { named };
        let release = verify_adapter(&adapter, &self.base).map_err(|e| format!("manifest {shown}: {e}"))?;
        if release.digest != manifest.digest {
            return Err(format!("manifest {shown}: names {} as {}, but that file is {}", adapter.display(), manifest.digest, release.digest));
        }
        Ok(release)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_testutil::adapters::{file_digest as digest, scratch_dir as tmp, write_adapter, write_base};

    /// The pinned case: the release is named by the same digest a fine-tune
    /// reports for the file, and a base other than the one the card records
    /// is refused with both digests in the message.
    #[test]
    fn an_adapter_verifies_only_against_the_base_it_was_trained_on() {
        let dir = tmp("pin");
        let (base, other) = (dir.join("base.safetensors"), dir.join("other.safetensors"));
        write_base(&base, 1.0);
        write_base(&other, 2.0);
        let adapter = dir.join("adapter.safetensors");
        write_adapter(&adapter, "local/base:local:chat:latest", "local/base", Some(digest(&base)), 0.5);

        let release = verify_adapter(&adapter, &ServedBase::new(&base, None)).expect("the adapter's own base verifies");
        assert_eq!(release, AdapterRelease { id: "local/base:local:chat:latest".to_string(), path: adapter.clone(), digest: digest(&adapter) });
        assert_eq!(release.serving_line("brain/qwen3"), format!("brain serve: brain/qwen3 adapter=local/base:local:chat:latest digest={}", digest(&adapter)));

        // Same id, different bytes: the digest decides, not the name.
        let err = verify_adapter(&adapter, &ServedBase::new(&other, Some("local/base".to_string()))).unwrap_err();
        assert!(err.contains(&digest(&base)) && err.contains(&digest(&other)), "a mismatch names both digests: {err}");
    }

    /// An adapter whose card records no base digest falls back to its base
    /// id, and is refused under any other.
    #[test]
    fn an_adapter_without_a_base_digest_is_bound_by_its_base_id() {
        let dir = tmp("by-id");
        let base = dir.join("base.safetensors");
        write_base(&base, 1.0);
        let adapter = dir.join("adapter.safetensors");
        write_adapter(&adapter, "a", "Qwen/Qwen3-0.6B", None, 0.5);

        assert!(verify_adapter(&adapter, &ServedBase::new(&base, Some("Qwen/Qwen3-0.6B".to_string()))).is_ok());
        let err = verify_adapter(&adapter, &ServedBase::new(&base, Some("Qwen/Qwen3-8B".to_string()))).unwrap_err();
        assert!(err.contains("Qwen/Qwen3-0.6B") && err.contains("Qwen/Qwen3-8B"), "{err}");
        assert!(verify_adapter(&base, &ServedBase::new(&base, None)).unwrap_err().contains("no model card"), "a bare checkpoint is not an adapter");
    }

    /// A manifest yields a release once, only when the file matches the
    /// digest it states; a bad manifest is reported once and yields nothing.
    #[test]
    fn a_manifest_yields_a_release_only_when_its_digest_verifies() {
        let dir = tmp("manifest");
        let base = dir.join("base.safetensors");
        write_base(&base, 1.0);
        let adapter = dir.join("adapter-1.safetensors");
        write_adapter(&adapter, "one", "local/base", Some(digest(&base)), 0.5);
        let manifest = dir.join("release.json");
        let mut follower = ManifestFollower::new(&manifest, ServedBase::new(&base, None));

        assert_eq!(follower.poll(), ManifestPoll::Missing);
        assert_eq!(follower.poll(), ManifestPoll::Unchanged, "a manifest still missing is not news");

        let wrong = format!("sha256:{}", "0".repeat(64));
        std::fs::write(&manifest, serde_json::json!({"adapter": "adapter-1.safetensors", "digest": wrong}).to_string()).unwrap();
        match follower.poll() {
            ManifestPoll::Rejected(why) => assert!(why.contains(&wrong) && why.contains(&digest(&adapter)), "{why}"),
            other => panic!("a digest that does not match the file must be refused, got {other:?}"),
        }
        assert_eq!(follower.poll(), ManifestPoll::Unchanged, "the same bad manifest is reported once");

        // Relative to the manifest's own directory.
        std::fs::write(&manifest, serde_json::json!({"adapter": "adapter-1.safetensors", "digest": digest(&adapter)}).to_string()).unwrap();
        assert_eq!(follower.poll(), ManifestPoll::Release(AdapterRelease { id: "one".to_string(), path: adapter.clone(), digest: digest(&adapter) }));

        std::fs::write(&manifest, "{\"adapter\": ").unwrap();
        assert!(matches!(follower.poll(), ManifestPoll::Rejected(_)), "a torn manifest is refused");
        std::fs::write(&manifest, serde_json::json!({"adapter": adapter, "digest": digest(&adapter)}).to_string()).unwrap();
        assert_eq!(follower.poll(), ManifestPoll::Unchanged, "a new manifest naming the release already served is not a new release");
    }
}
