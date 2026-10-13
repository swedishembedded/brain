// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verified release of fine-tuned model
// adapters into live inference servers for its clients. If your team needs
// expertise in serving and rolling out fine-tuned language models, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The release manifest a server follows to learn which adapter to serve:
//! `{"adapter": "<path>", "digest": "sha256:<hex>"}`.
//!
//! The release step that publishes an adapter writes it with
//! [`AdapterManifest::write_atomically`]; the server reads it with the same
//! type, so the two cannot disagree about the shape. The server's checks (the
//! named file hashes to `digest`, binds to the served base) live with the
//! server; this module owns only the file.

use std::io::Write;
use std::path::{Path, PathBuf};

/// One release: the adapter file and the digest the server must find it at.
/// Unknown fields are refused on read: a manifest carrying a field this reader
/// does not know was written for a different reader, and guessing what it
/// meant would serve the wrong thing.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterManifest {
    /// The adapter file. A relative path is relative to the manifest's
    /// directory.
    pub adapter: PathBuf,
    /// `sha256:<hex>` of the adapter file, as `fetch::file_digest` states it.
    pub digest: String,
}

impl AdapterManifest {
    /// Publish this manifest at `path`: written to a file beside it, then
    /// renamed over it, so a server polling `path` sees the old manifest or
    /// the new one, never a torn one. The adapter file it names must already
    /// be in place and must not change afterwards.
    ///
    /// # Errors
    /// `path` has no file name, the directory is not writable, or the
    /// manifest cannot be serialised (a non-UTF-8 adapter path). Nothing is
    /// left beside `path` on failure.
    pub fn write_atomically(&self, path: &Path) -> std::io::Result<()> {
        let name = path.file_name().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("manifest path {} has no file name", path.display())))?;
        let mut body = serde_json::to_vec(self).map_err(std::io::Error::other)?;
        body.push(b'\n');
        let mut beside = std::ffi::OsString::from(".");
        beside.push(name);
        beside.push(format!(".{}.tmp", std::process::id()));
        let beside = path.with_file_name(beside);
        let written = std::fs::File::create(&beside).and_then(|mut file| {
            file.write_all(&body)?;
            file.sync_all()
        });
        written.and_then(|()| std::fs::rename(&beside, path)).inspect_err(|_| {
            let _ = std::fs::remove_file(&beside);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Publishing replaces the previous manifest whole, reads back as written,
    /// and leaves nothing else in the directory.
    #[test]
    fn publishing_replaces_the_manifest_and_leaves_nothing_beside_it() {
        let dir = std::env::temp_dir().join(format!("brain-adapter-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("release.json");
        let first = AdapterManifest { adapter: "releases/one.safetensors".into(), digest: format!("sha256:{}", "a".repeat(64)) };
        let second = AdapterManifest { adapter: "releases/two.safetensors".into(), digest: format!("sha256:{}", "b".repeat(64)) };
        first.write_atomically(&path).unwrap();
        second.write_atomically(&path).unwrap();
        assert_eq!(serde_json::from_slice::<AdapterManifest>(&std::fs::read(&path).unwrap()).unwrap(), second);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        assert!(AdapterManifest::write_atomically(&first, Path::new("/")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
