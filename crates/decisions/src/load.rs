// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Loading a decision checkpoint into a [`Session`], with every way it can go
//! wrong named.
//!
//! The classes a client can branch on ([`capability::FailureClass`]):
//!
//! * `not-installed` - the directory or the files a load reads are absent.
//! * `failed-to-load` - the files are there and are not a servable model: a
//!   manifest that does not parse, a declared hash the file does not match, a
//!   declared context the checkpoint does not hold, an encoder with no trained
//!   head, weights that fail the architecture's own coverage checks, or a
//!   device that could not be built.
//!
//! Error text reaches remote callers, so it never carries the checkpoint's
//! path; the full error goes to the server log.
//!
//! Swedish Embedded AB implements load paths whose failures say what is wrong
//! with the model, not just that something is. If your team needs expertise in
//! checkpoint validation, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::Path;

use capability::{tag, FailureClass};
use gpu_core::Gpu;

use crate::caps::{Served, Session};
use crate::engine::{Backend, DecideWeights, Engine, LayaWeights};
use crate::manifest::{verify_identity, DecisionManifest};
use crate::store::{inspect, Readiness};

/// The kernel set an architecture builds its device handle with.
pub fn pipelines(backend: Backend) -> &'static [(&'static str, &'static str)] {
    match backend {
        Backend::Decide => decide::kern::PIPELINES,
        Backend::Laya => modernbert::kern::PIPELINES,
    }
}

/// A loaded model and the manifest it was loaded under.
pub struct Loaded {
    pub session: Session,
    pub manifest: DecisionManifest,
}

fn failed(id: &str, dir: &Path, detail: &str) -> String {
    eprintln!("decisions: model '{id}' failed to load: {detail}");
    let shown = detail.replace(&dir.to_string_lossy().into_owned(), "<model dir>");
    tag(FailureClass::FailedToLoad, format!("model '{id}' could not be loaded: {shown}"))
}

/// Load the model `id` from `dir`. `make_gpu` builds the device handle for the
/// architecture on the device the caller already selected.
///
/// Reads the whole checkpoint and may take seconds; callers run it on the
/// thread that owns the device, never on a request thread that must answer
/// quickly.
pub fn load(id: &str, dir: &Path, make_gpu: impl FnOnce(Backend) -> Result<Gpu, String>) -> Result<Loaded, String> {
    let record = inspect(id, dir, false);
    match &record.readiness {
        Readiness::Ready => {}
        Readiness::NotInstalled { missing } => {
            return Err(tag(FailureClass::NotInstalled, format!("model '{id}' is not installed (missing {}): run `brain pull {id}`", missing.join(", "))))
        }
        Readiness::Incompatible { reason } => return Err(failed(id, dir, reason)),
    }
    let (backend, manifest) = match (record.backend, record.manifest) {
        (Some(b), Some(m)) => (b, m),
        _ => return Err(failed(id, dir, "the directory is not a decision checkpoint")),
    };
    verify_identity(&manifest, dir, backend).map_err(|e| failed(id, dir, &e))?;

    let dir_str = dir.to_str().ok_or_else(|| failed(id, dir, "the checkpoint path is not UTF-8"))?;
    let engine = match backend {
        Backend::Decide => {
            let weights = DecideWeights::read(dir_str, None, 0).map_err(|e| failed(id, dir, &e))?;
            if !weights.head_is_trained() {
                return Err(failed(id, dir, "the encoder carries no trained head"));
            }
            let gpu = make_gpu(backend).map_err(|e| failed(id, dir, &e))?;
            Engine::Decide(Box::new(weights.build(gpu, decide::decide::Limits::default(), false)))
        }
        Backend::Laya => {
            let weights = LayaWeights::read(dir_str, None).map_err(|e| failed(id, dir, &e))?;
            let gpu = make_gpu(backend).map_err(|e| failed(id, dir, &e))?;
            Engine::Laya(Box::new(weights.build(gpu, modernbert::Training::Off)))
        }
    };
    if let Some(declared) = manifest.max_input_tokens {
        let actual = engine.max_input_tokens();
        if declared != actual {
            return Err(failed(id, dir, &format!("the manifest declares max_input_tokens {declared} but the checkpoint holds {actual}")));
        }
    }
    let served = Served { model: id.to_string(), question_types: manifest.question_types.clone() };
    Ok(Loaded { session: Session::new(served, engine), manifest })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{write_decide, write_laya, write_laya_head_only, Scratch};
    use crate::laya_head::{BASE_FILE, HEAD_FILE};
    use crate::manifest::{weights_identity, FILE};
    use capability::classify;

    fn dev(b: Backend) -> Result<Gpu, String> {
        Ok(gpu_core::testgpu::dev(pipelines(b)))
    }

    fn class(e: &str) -> FailureClass {
        classify(e).unwrap_or_else(|| panic!("untagged: {e}")).0
    }

    #[test]
    fn an_absent_model_is_not_installed_and_the_message_names_no_path() {
        let s = Scratch::new("load-absent");
        let dir = s.join("acme/gone");
        let e = load("acme/gone", &dir, dev).err().expect("must fail");
        assert_eq!(class(&e), FailureClass::NotInstalled);
        assert!(e.contains("brain pull acme/gone"), "{e}");
        assert!(!e.contains(s.to_str().unwrap()), "no path may reach a remote caller: {e}");
    }

    #[test]
    fn a_checkpoint_missing_a_file_is_not_installed() {
        let s = Scratch::new("load-partial");
        write_laya(&s, 48);
        std::fs::remove_file(s.join("model.safetensors")).unwrap();
        let e = load("a/b", &s, dev).err().expect("must fail");
        assert_eq!(class(&e), FailureClass::NotInstalled);
        assert!(e.contains("model.safetensors"), "{e}");
    }

    #[test]
    fn corrupt_weights_fail_to_load_without_leaking_the_path() {
        let s = Scratch::new("load-corrupt");
        write_laya(&s, 48);
        std::fs::write(s.join("model.safetensors"), b"not a safetensors file").unwrap();
        let e = load("a/b", &s, dev).err().expect("must fail");
        assert_eq!(class(&e), FailureClass::FailedToLoad);
        assert!(!e.contains(s.to_str().unwrap()), "{e}");
    }

    #[test]
    fn a_hash_that_does_not_match_the_weights_fails_to_load() {
        let s = Scratch::new("load-hash");
        write_laya(&s, 48);
        let wrong = format!("sha256:{}", "0".repeat(64));
        std::fs::write(s.join(FILE), format!(r#"{{"schema": 1, "weights_sha256": "{wrong}"}}"#)).unwrap();
        let e = load("a/b", &s, dev).err().expect("must fail");
        assert_eq!(class(&e), FailureClass::FailedToLoad);
        assert!(e.contains("does not match the manifest"), "{e}");
    }

    #[test]
    fn a_declared_context_the_checkpoint_does_not_hold_fails_to_load() {
        let s = Scratch::new("load-ctx");
        write_laya(&s, 48);
        std::fs::write(s.join(FILE), r#"{"schema": 1, "max_input_tokens": 8192}"#).unwrap();
        let e = load("a/b", &s, dev).err().expect("must fail");
        assert_eq!(class(&e), FailureClass::FailedToLoad);
        assert!(e.contains("8192") && e.contains("48"), "{e}");
    }

    #[test]
    fn a_device_that_cannot_be_built_fails_to_load() {
        let s = Scratch::new("load-device");
        write_laya(&s, 48);
        let e = load("a/b", &s, |_| Err("no adapter".to_string())).err().expect("must fail");
        assert_eq!(class(&e), FailureClass::FailedToLoad);
        assert!(e.contains("no adapter"), "{e}");
    }

    #[test]
    fn both_architectures_load_and_report_their_own_backend_and_context() {
        let l = Scratch::new("load-laya");
        write_laya(&l, 48);
        let loaded = load("a/laya", &l, dev).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(loaded.session.engine().backend(), Backend::Laya);
        assert_eq!(loaded.session.engine().max_input_tokens(), 48);

        let d = Scratch::new("load-decide");
        write_decide(&d, 2);
        let loaded = load("a/decide", &d, dev).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(loaded.session.engine().backend(), Backend::Decide);
        assert_eq!(loaded.session.served().model, "a/decide");
    }

    /// A base and a head-only directory over it whose manifest declares the
    /// weights identity, the way a training run writes it.
    fn head_only(tag: &str) -> Scratch {
        let s = Scratch::new(tag);
        write_laya(&s.join("base"), 48);
        write_laya_head_only(&s.join("out"), &s.join("base"), "acme/laya", 99);
        let identity = weights_identity(&s.join("out"), Backend::Laya).unwrap();
        std::fs::write(s.join("out").join(FILE), format!(r#"{{"schema": 1, "weights_sha256": "{identity}"}}"#)).unwrap();
        s
    }

    #[test]
    fn a_head_only_directory_loads_as_a_laya_model_with_the_bases_context() {
        let s = head_only("load-head-only");
        let loaded = load("acme/trained", &s.join("out"), dev).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(loaded.session.engine().backend(), Backend::Laya);
        assert_eq!(loaded.session.engine().max_input_tokens(), 48);
    }

    #[test]
    fn a_corrupt_head_or_base_reference_or_base_fails_to_load_without_leaking_a_path() {
        type Damage = (&'static str, fn(&Scratch));
        let damage: [Damage; 5] = [
            ("head bytes", |s| {
                let head = s.join("out").join(HEAD_FILE);
                let mut bytes = std::fs::read(&head).unwrap();
                let last = bytes.len() - 1;
                bytes[last] ^= 1;
                std::fs::write(head, bytes).unwrap();
            }),
            ("head not a safetensors file", |s| std::fs::write(s.join("out").join(HEAD_FILE), b"not a safetensors file").unwrap()),
            ("reference unreadable", |s| std::fs::write(s.join("out").join(BASE_FILE), b"{").unwrap()),
            ("reference names another base", |s| {
                let text = std::fs::read_to_string(s.join("out").join(BASE_FILE)).unwrap();
                let old = text.split("sha256:").nth(1).unwrap()[..64].to_string();
                std::fs::write(s.join("out").join(BASE_FILE), text.replace(&old, &"0".repeat(64))).unwrap();
            }),
            ("base weights changed", |s| {
                let weights = s.join("base").join("model.safetensors");
                let mut bytes = std::fs::read(&weights).unwrap();
                let last = bytes.len() - 1;
                bytes[last] ^= 1;
                std::fs::write(weights, bytes).unwrap();
            }),
        ];
        for (what, apply) in damage {
            let s = head_only("load-head-only-damage");
            apply(&s);
            let e = load("acme/trained", &s.join("out"), dev).err().unwrap_or_else(|| panic!("{what}: must fail"));
            assert_eq!(class(&e), FailureClass::FailedToLoad, "{what}: {e}");
            assert!(!e.contains(s.to_str().unwrap()), "{what}: {e}");
        }
    }

    #[test]
    fn a_head_only_directory_missing_its_head_is_not_installed() {
        let s = head_only("load-head-only-missing");
        std::fs::remove_file(s.join("out").join(HEAD_FILE)).unwrap();
        let e = load("acme/trained", &s.join("out"), dev).err().expect("must fail");
        assert_eq!(class(&e), FailureClass::NotInstalled);
        assert!(e.contains(HEAD_FILE), "{e}");
    }
}
