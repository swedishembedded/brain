// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verified release of fine-tuned model
// adapters into live inference servers for its clients. If your team needs
// expertise in serving and rolling out fine-tuned language models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Which LoRA adapter release `brain serve` puts beside its Qwen3 base, and
//! the checks that make "this is what is served" a verified statement rather
//! than a file name.
//!
//! A [`LoadedAdapter`] is an adapter file that has been read as a Qwen3 LoRA,
//! (for [`verify_adapter`]) bound to the served base, and named by its content
//! digest - `brain_modelstore::fetch::bytes_digest` over the file's bytes, the
//! same string `brain_modelstore::fetch::file_digest` gives for the file, a
//! fine-tune reports as its adapter digest and a chat pipeline reports as its
//! identity. [`verify_adapter`] and [`load_adapter`] produce one; nothing else
//! does.
//!
//! **One read.** The file is read into memory once. The digest is taken of
//! those bytes and the factors are parsed from those same bytes, so there is
//! no window between "verified" and "served" in which the file can be
//! rewritten: what the digest names is what the engine is given. The engine
//! cannot do this check itself (the catalog depends on the engine crate, not
//! the reverse), so a [`LoadedAdapter`] is what crosses that boundary
//! ([`LoadedAdapter::into_engine`]).
//!
//! **Binding to the base.** An adapter is only meaningful on the exact base it
//! was trained against. When its card records that base's digest
//! (`TrainingProvenance::base_digest`), the served base must hash to it. An
//! adapter whose card records no base digest (one written before the field
//! existed, or by a trainer that does not record it) is bound by its card's
//! base id instead, which is weaker: two different files can carry the same
//! id.
//!
//! **Following a release manifest.** [`ManifestFollower`] re-reads a small
//! JSON file, `{"adapter": "<path>", "digest": "sha256:<hex>"}`
//! ([`AdapterManifest`]), that a release step replaces atomically
//! ([`AdapterManifest::write_atomically`]: write beside it, then rename over it).
//! A new manifest yields a release only after the named file hashes to the
//! digest the manifest states AND the file verifies against the base; a
//! manifest that fails any check is reported once and yields nothing, so the
//! caller keeps serving what it had. Publish each release under a new file
//! name: a release is read when its manifest changes, and a manifest that
//! still names the digest already served is not a new release.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub use brain_modelstore::adapter_manifest::AdapterManifest;
use checkpoint::st::ModelCard;
use qwen3::lora::DeviceAdapter;
use qwen3::serve::VerifiedAdapter;
use residency::AdapterIdentity;

/// The largest adapter file read into memory. An adapter is megabytes (a
/// rank-16 attention adapter on an 8B base is tens of MiB); a file past this
/// is not one, and reading it whole would only exhaust the host.
const MAX_ADAPTER_BYTES: u64 = 2 << 30;

/// The base checkpoint a server puts adapters on: its file (or a
/// `transformers` directory), the id it was resolved under, and its digest,
/// hashed at most once and only when an adapter asks for it (the base is
/// gigabytes; an adapter bound by id never needs it hashed).
#[derive(Clone)]
pub struct ServedBase {
    path: PathBuf,
    id: Option<String>,
    digest: OnceLock<Result<String, String>>,
}

impl ServedBase {
    /// The base at `path`, served under `id` when it has one (a catalog id, or
    /// the id `BRAIN_QWEN_WEIGHTS` resolved to).
    pub fn new(path: impl Into<PathBuf>, id: Option<String>) -> ServedBase {
        ServedBase { path: path.into(), id, digest: OnceLock::new() }
    }

    fn digest(&self) -> Result<&str, String> {
        self.digest
            .get_or_init(|| brain_modelstore::fetch::base_digest(&self.path).map_err(|e| format!("the served base {}: {e}", self.path.display())))
            .as_deref()
            .map_err(Clone::clone)
    }
}

/// A named adapter release: what `brain serve` reports it serves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterRelease {
    /// The id on the adapter's own card.
    pub id: String,
    /// Where the file was read from. For a log line; never part of a model
    /// card or any response.
    pub path: PathBuf,
    /// `sha256:<hex>` of the file.
    pub digest: String,
}

impl AdapterRelease {
    /// What the serving surface calls this release: id and digest, no path.
    pub fn identity(&self) -> AdapterIdentity {
        AdapterIdentity { id: self.id.clone(), digest: self.digest.clone() }
    }

    /// The line `brain serve` prints when `model` starts serving this
    /// release, the one a caller greps to confirm what is served.
    pub fn serving_line(&self, model: &str) -> String {
        format!("brain serve: {model} adapter={} digest={}", self.id, self.digest)
    }
}

/// An adapter read once: its name and its factors, both from the same bytes.
#[derive(Clone)]
pub struct LoadedAdapter {
    /// The release this file is: card id, digest, where it was read.
    pub release: AdapterRelease,
    /// The factors, parsed from the bytes `release.digest` names.
    pub adapter: DeviceAdapter,
}

impl LoadedAdapter {
    /// The form the serving engine registers an adapter in
    /// ([`qwen3::serve::Engine::load_adapter`]).
    pub fn into_engine(self) -> VerifiedAdapter {
        VerifiedAdapter { label: self.release.id, digest: self.release.digest, adapter: self.adapter }
    }
}

/// Read the adapter file at `path` ONCE: its digest, card and factors, all
/// from the same bytes. Checks the file is a Qwen3 LoRA with at least one
/// factor pair; nothing about the base.
fn read_once(path: &Path) -> Result<(LoadedAdapter, ModelCard), String> {
    let shown = path.display();
    let len = std::fs::metadata(path).map_err(|e| format!("adapter {shown}: {e}"))?.len();
    if len > MAX_ADAPTER_BYTES {
        return Err(format!("adapter {shown}: {len} bytes is larger than any adapter file ({MAX_ADAPTER_BYTES} at most)"));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("adapter {shown}: {e}"))?;
    let digest = format!("sha256:{}", brain_modelstore::fetch::bytes_digest(&bytes));
    let (adapter, card) = qwen3::lora::parse_adapter(&bytes, &shown.to_string()).map_err(|e| match e.kind() {
        // The model crate says "has no ModelCard"; say what that means for a release.
        std::io::ErrorKind::InvalidData if e.to_string().contains("has no ModelCard") => format!("adapter {shown}: no model card, so nothing says which base it belongs to"),
        _ => e.to_string(),
    })?;
    let descriptor = card.adapter.as_ref().ok_or_else(|| format!("adapter {shown}: its card ({}) describes a model, not an adapter", card.id))?;
    // The engine's bank implements exactly this family and kind; anything
    // else would only fail later, at the first request.
    if card.family != "qwen" || descriptor.kind != "lora" {
        return Err(format!("adapter {shown}: a {} {:?} adapter; the served base is a Qwen3 that takes LoRA adapters", card.family, descriptor.kind));
    }
    let release = AdapterRelease { id: card.id.clone(), path: path.to_path_buf(), digest };
    Ok((LoadedAdapter { release, adapter }, card))
}

/// Read the adapter at `path` as a release, WITHOUT binding it to a base:
/// for a directory a gated promotion fills, whose files were checked where
/// they were made. Every other refusal [`verify_adapter`] has still applies,
/// and the engine still checks the adapter's shapes against the model it is
/// given to.
pub fn load_adapter(path: &Path) -> Result<LoadedAdapter, String> {
    read_once(path).map(|(loaded, _)| loaded)
}

/// Read `path` as a Qwen3 LoRA adapter and bind it to `base`. Every refusal
/// names the file and what was wrong with it; a base mismatch names both
/// digests.
pub fn verify_adapter(path: &Path, base: &ServedBase) -> Result<LoadedAdapter, String> {
    let shown = path.display();
    let (loaded, card) = read_once(path)?;
    let adapter = card.adapter.as_ref().expect("read_once refused a card without an adapter descriptor");
    let recorded_base_digest = card.training.as_ref().and_then(|t| t.base_digest.as_deref());
    match (recorded_base_digest, adapter.base.as_deref()) {
        (Some(trained_on), _) => {
            let served = base.digest()?;
            if trained_on != served {
                return Err(format!(
                    "adapter {shown} was trained against base {trained_on}, but the served base {} is {served}; refusing to put it on a different base",
                    base.path.display()
                ));
            }
        }
        (None, Some(base_id)) => {
            if base.id.as_deref() != Some(base_id) {
                let served = base.digest()?;
                return Err(format!(
                    "adapter {shown} names base {base_id:?} and records no base digest, but the served base {} is {:?} ({served}); refusing to put it on a different base",
                    base.path.display(),
                    base.id.as_deref().unwrap_or("unnamed")
                ));
            }
        }
        (None, None) => return Err(format!("adapter {shown}: its card names no base (neither a base digest nor a base id)")),
    }
    Ok(loaded)
}

/// What one look at the manifest found.
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
    Release(Box<LoadedAdapter>),
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
    /// Follow the manifest at `path`, binding each release to `base`.
    pub fn new(path: impl Into<PathBuf>, base: ServedBase) -> ManifestFollower {
        ManifestFollower { path: path.into(), base, seen: None, serving: None }
    }

    /// The manifest being followed.
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
                Ok(loaded) if self.serving.as_deref() == Some(loaded.release.digest.as_str()) => ManifestPoll::Unchanged,
                Ok(loaded) => {
                    self.serving = Some(loaded.release.digest.clone());
                    ManifestPoll::Release(Box::new(loaded))
                }
                Err(why) => ManifestPoll::Rejected(why),
            },
        };
        self.seen = Some(seen);
        outcome
    }

    fn release_of(&self, bytes: &[u8]) -> Result<LoadedAdapter, String> {
        let shown = self.path.display();
        let manifest: AdapterManifest = serde_json::from_slice(bytes).map_err(|e| format!("manifest {shown}: {e}"))?;
        let hex = manifest.digest.strip_prefix("sha256:").unwrap_or_default();
        if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            return Err(format!("manifest {shown}: digest {:?} is not sha256:<64 lowercase hex digits>", manifest.digest));
        }
        // A relative adapter path is relative to the manifest, so a release
        // directory can be moved as a whole.
        let named = manifest.adapter.clone();
        let adapter = if named.is_relative() { self.path.parent().unwrap_or_else(|| Path::new(".")).join(named) } else { named };
        let loaded = verify_adapter(&adapter, &self.base).map_err(|e| format!("manifest {shown}: {e}"))?;
        if loaded.release.digest != manifest.digest {
            return Err(format!("manifest {shown}: names {} as {}, but that file is {}", adapter.display(), manifest.digest, loaded.release.digest));
        }
        Ok(loaded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_testutil::adapters::{file_digest as digest, scratch_dir as tmp, write_adapter, write_base};

    /// The release a poll produced; any other outcome fails the test.
    fn released(poll: ManifestPoll) -> AdapterRelease {
        match poll {
            ManifestPoll::Release(loaded) => loaded.release,
            ManifestPoll::Unchanged => panic!("expected a release, the manifest was unchanged"),
            ManifestPoll::Missing => panic!("expected a release, the manifest is missing"),
            ManifestPoll::Rejected(why) => panic!("expected a release, the manifest was rejected: {why}"),
        }
    }

    fn rejected(poll: ManifestPoll) -> String {
        match poll {
            ManifestPoll::Rejected(why) => why,
            ManifestPoll::Release(loaded) => panic!("expected a rejection, got release {:?}", loaded.release),
            ManifestPoll::Unchanged => panic!("expected a rejection, the manifest was unchanged"),
            ManifestPoll::Missing => panic!("expected a rejection, the manifest is missing"),
        }
    }

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

        let loaded = verify_adapter(&adapter, &ServedBase::new(&base, None)).expect("the adapter's own base verifies");
        assert_eq!(loaded.release, AdapterRelease { id: "local/base:local:chat:latest".to_string(), path: adapter.clone(), digest: digest(&adapter) });
        assert_eq!(loaded.release.serving_line("brain/qwen3"), format!("brain serve: brain/qwen3 adapter=local/base:local:chat:latest digest={}", digest(&adapter)));
        assert_eq!((loaded.adapter.rank, loaded.adapter.sites.len()), (2, 1), "the factors come with the release");

        // Same id, different bytes: the digest decides, not the name.
        let err = verify_adapter(&adapter, &ServedBase::new(&other, Some("local/base".to_string()))).err().expect("another base refuses it");
        assert!(err.contains(&digest(&base)) && err.contains(&digest(&other)), "a mismatch names both digests: {err}");
    }

    /// The digest and the factors come from the same bytes: the release a read
    /// produced is the file as it was read, whatever is written to the path
    /// afterwards.
    #[test]
    fn the_digest_names_the_bytes_the_factors_were_parsed_from() {
        let dir = tmp("one-read");
        let base = dir.join("base.safetensors");
        write_base(&base, 1.0);
        let adapter = dir.join("adapter.safetensors");
        write_adapter(&adapter, "x", "local/base", Some(digest(&base)), 0.5);

        let loaded = verify_adapter(&adapter, &ServedBase::new(&base, None)).unwrap();
        let before = digest(&adapter);
        write_adapter(&adapter, "x", "local/base", Some(digest(&base)), 9.0);
        assert_ne!(digest(&adapter), before, "the file changed after it was read");
        assert_eq!(loaded.release.digest, before, "the release still names what was read");
        assert!(loaded.adapter.sites[0].a.iter().all(|&v| v == 0.5), "and the factors are the ones that digest names");
        // Parsing the bytes and reading the path agree.
        let same = qwen3::lora::read_adapter(adapter.to_str().unwrap()).unwrap();
        assert!(same.sites[0].a.iter().all(|&v| v == 9.0));
    }

    /// A Hugging Face directory served as the base is named by its config and
    /// weights together, the same digest the fine-tune recorded; changing a
    /// weight file is another base and refuses the adapter.
    #[test]
    fn an_adapter_verifies_against_a_served_hugging_face_directory() {
        let dir = tmp("hf-dir");
        let base = dir.join("DeepSeek-R1-Distill-Qwen-1.5B");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("config.json"), b"{}").unwrap();
        write_base(&base.join("model.safetensors"), 1.0);
        let adapter = dir.join("adapter.safetensors");
        let trained_on = brain_modelstore::fetch::base_digest(&base).unwrap();
        write_adapter(&adapter, "local/DeepSeek-R1-Distill-Qwen-1.5B:local:chat:latest", "local/DeepSeek-R1-Distill-Qwen-1.5B", Some(trained_on), 0.5);

        verify_adapter(&adapter, &ServedBase::new(&base, None)).expect("the directory the adapter was trained on verifies");

        write_base(&base.join("model.safetensors"), 2.0);
        let err = verify_adapter(&adapter, &ServedBase::new(&base, None)).err().unwrap();
        assert!(err.contains("refusing to put it on a different base"), "{err}");
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
        let err = verify_adapter(&adapter, &ServedBase::new(&base, Some("Qwen/Qwen3-8B".to_string()))).err().unwrap();
        assert!(err.contains("Qwen/Qwen3-0.6B") && err.contains("Qwen/Qwen3-8B"), "{err}");
        assert!(verify_adapter(&base, &ServedBase::new(&base, None)).err().unwrap().contains("no model card"), "a bare checkpoint is not an adapter");
    }

    /// An unbound read (a directory a gated promotion fills) names the file
    /// by its card and digest without hashing any base, and still refuses a
    /// file that is not a Qwen3 LoRA.
    #[test]
    fn an_unbound_load_names_the_file_and_still_refuses_what_is_not_a_qwen3_lora() {
        let dir = tmp("unbound");
        let adapter = dir.join("adapter.safetensors");
        write_adapter(&adapter, "promoted", "base", None, 0.5);
        let loaded = load_adapter(&adapter).expect("an unbound adapter loads");
        assert_eq!(loaded.release, AdapterRelease { id: "promoted".to_string(), path: adapter.clone(), digest: digest(&adapter) });

        let base = dir.join("base.safetensors");
        write_base(&base, 1.0);
        assert!(load_adapter(&base).err().unwrap().contains("no model card"));
        assert!(load_adapter(&dir.join("absent.safetensors")).is_err());
    }

    /// The manifest a release step writes is the one the follower reads: a
    /// manifest written by [`AdapterManifest::write_atomically`] yields its
    /// release, and one whose digest is not the file's is rejected.
    #[test]
    fn a_written_manifest_is_accepted_and_a_digest_mismatch_is_rejected() {
        let dir = tmp("written-manifest");
        let base = dir.join("base.safetensors");
        write_base(&base, 1.0);
        let adapter = dir.join("adapter-1.safetensors");
        write_adapter(&adapter, "one", "local/base", Some(digest(&base)), 0.5);
        let manifest = dir.join("release.json");
        let mut follower = ManifestFollower::new(&manifest, ServedBase::new(&base, None));

        AdapterManifest { adapter: adapter.clone(), digest: format!("sha256:{}", "0".repeat(64)) }.write_atomically(&manifest).unwrap();
        assert!(rejected(follower.poll()).contains(&digest(&adapter)));

        AdapterManifest { adapter: adapter.clone(), digest: digest(&adapter) }.write_atomically(&manifest).unwrap();
        assert_eq!(released(follower.poll()), AdapterRelease { id: "one".to_string(), path: adapter.clone(), digest: digest(&adapter) });
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

        assert!(matches!(follower.poll(), ManifestPoll::Missing));
        assert!(matches!(follower.poll(), ManifestPoll::Unchanged), "a manifest still missing is not news");

        let wrong = format!("sha256:{}", "0".repeat(64));
        std::fs::write(&manifest, serde_json::json!({"adapter": "adapter-1.safetensors", "digest": wrong}).to_string()).unwrap();
        let why = rejected(follower.poll());
        assert!(why.contains(&wrong) && why.contains(&digest(&adapter)), "{why}");
        assert!(matches!(follower.poll(), ManifestPoll::Unchanged), "the same bad manifest is reported once");

        // Relative to the manifest's own directory.
        std::fs::write(&manifest, serde_json::json!({"adapter": "adapter-1.safetensors", "digest": digest(&adapter)}).to_string()).unwrap();
        assert_eq!(released(follower.poll()), AdapterRelease { id: "one".to_string(), path: adapter.clone(), digest: digest(&adapter) });

        std::fs::write(&manifest, "{\"adapter\": ").unwrap();
        rejected(follower.poll());
        std::fs::write(&manifest, serde_json::json!({"adapter": adapter, "digest": digest(&adapter)}).to_string()).unwrap();
        assert!(matches!(follower.poll(), ManifestPoll::Unchanged), "a new manifest naming the release already served is not a new release");
    }
}
