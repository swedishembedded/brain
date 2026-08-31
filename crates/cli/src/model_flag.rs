// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The generic `--model <ARG>` flag: one resolver every per-architecture CLI
//! shares, so each model's command grows the same weights argument instead of
//! a private one.
//!
//! `ARG` is tried against the same ladder whether the caller typed a path or
//! a model id, and the ladder is deliberate about which of the two it is:
//!
//! 1. An explicit `.gguf`/`.safetensors` extension (case-insensitive) names
//!    the file OUTRIGHT. It is taken literally -- the file must exist, or the
//!    error says so. Nothing is probed and the model store is never consulted.
//! 2. Without an extension, ARG is a path STEM: `<ARG>.gguf` then
//!    `<ARG>.safetensors` are probed beside it, in that order -- brain's own
//!    format first.
//! 3. Still nothing: ARG is read as a `<vendor>/<repo>[-<QUANT>]` model id.
//!    A local copy in the model store wins (a compound checkpoint hands back
//!    the `role` the caller names -- `"dit"`, `"vae"`, ...); otherwise the
//!    model is announced and DOWNLOADED through the store's resolution
//!    ladder, printing per-file progress -- but only when fetching is enabled
//!    (`--autofetch` / `BRAIN_AUTO_FETCH=1`); with it off the error names
//!    `brain pull <model>` instead.
//!
//! The caller learns the resolved weights path plus a display name (the
//! canonical reference for a store id, the argument as typed for a path), so
//! the command can print which model it is actually running before any load
//! begins.
//!
//! Swedish Embedded AB implements command-line model resolution and
//! distribution for its clients. If your team needs expertise in ML tooling
//! ergonomics then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::io::IsTerminal;
use std::path::Path;

use brain_modelref::ModelRef;

/// The weights a `--model` argument resolved to, ready to hand to the
/// architecture's own loader.
#[derive(Debug)]
pub struct ResolvedModel {
    /// What to call this model in messages: the canonical `<vendor>/<repo>`
    /// reference when the argument named one, else the argument as typed.
    pub name: String,
    /// The weights file (or, for a compound checkpoint's role, the file or
    /// directory that role's loader accepts).
    pub path: String,
}

/// The weights extensions `--model` recognises as an explicit file name.
/// Anything ending in one of these is never probed or fetched -- see the
/// module ladder.
const WEIGHTS_EXTENSIONS: &[&str] = &[".gguf", ".safetensors"];

/// Resolve a `--model` value against the real model store and hub.
///
/// `role` selects one file/directory of a COMPOUND checkpoint (flux2's DiT is
/// `"dit"`); it is ignored for single-file models and plain paths.
pub fn resolve(arg: &str, role: &str) -> Result<ResolvedModel, String> {
    let root = crate::model_dir::resolve(None);
    let hub = brain_modelstore::HfHub::new();
    resolve_with(arg, role, root.as_deref(), &hub)
}

fn resolve_with(arg: &str, role: &str, store_root: Option<&Path>, hub: &dyn brain_modelstore::Hub) -> Result<ResolvedModel, String> {
    // An explicit extension names the file outright. Taken literally -- the
    // model store is never consulted for a path the caller already pointed
    // at, so a miss here is the caller's typo, and the error says which.
    if has_weights_extension(arg) {
        if !Path::new(arg).is_file() {
            return Err(format!(
                "--model {arg}: no such file (an explicit extension is taken literally; drop it to probe .gguf/.safetensors beside the name, or give a <vendor>/<repo> id to fetch from the model store)"
            ));
        }
        return Ok(ResolvedModel { name: arg.to_string(), path: arg.to_string() });
    }

    // Extensionless: probe brain's two weights formats beside the stem, GGUF
    // first -- the format the flag exists for, and the one a directory of
    // checkpoints usually ships.
    for ext in [".gguf", ".safetensors"] {
        let candidate = format!("{arg}{ext}");
        if Path::new(&candidate).is_file() {
            return Ok(ResolvedModel { name: arg.to_string(), path: candidate });
        }
    }

    // Nothing on disk: read the name as a store id. A local copy serves with
    // no hub touch at all -- the same fast path [`brain_modelstore::plan`]
    // itself takes -- and anything else is where fetching is opt-in
    // ([`crate::supply::auto_fetch_enabled`]): a store id that is not pulled
    // errors naming both remedies rather than downloading.
    let reference = ModelRef::parse(arg).map_err(|e| {
        format!("--model {arg}: no weights file found (tried {arg}.gguf, {arg}.safetensors) and the name is not a <vendor>/<repo> model id either ({e})")
    })?;
    let root = store_root
        .ok_or_else(|| "--model: no models directory (set --models-dir, BRAIN_MODELS_DIR, or $HOME)".to_string())?;
    let store = brain_modelstore::Store::new(root);
    let Some(local) = store.local(&reference) else {
        if !crate::supply::auto_fetch_enabled() {
            return Err(format!(
                "--model {arg}: not pulled. Fetch with `brain pull {arg}`, or rerun with --autofetch (BRAIN_AUTO_FETCH=1)."
            ));
        }
        eprintln!("brain: {arg}: no local copy - downloading ...");
        let plan = brain_modelstore::plan(&reference, &store, hub).map_err(|e| format!("--model {arg}: {e}"))?;
        // Progress on stderr: the command's stdout carries its own output.
        let mut err = std::io::stderr();
        let mode = crate::pull_cli::Mode::of(err.is_terminal());
        let (local, moved, secs) = crate::supply::execute_plan_reported(&store, hub, &plan, arg, mode, &mut err)?;
        eprintln!(
            "brain: {arg}: fetched {} in {}",
            crate::pull_cli::human_bytes(moved),
            crate::pull_cli::human_secs(secs)
        );
        // The plan's reference, not the argument's: a recipe that CHOSE
        // between interchangeable artifacts (a GGUF release resolving to one
        // quant) records the choice there, and that choice is the canonical
        // name.
        return stored(&local, &plan.reference, role, arg);
    };
    stored(&local, &reference, role, arg)
}

/// Pick the weights file out of a store model: the role the caller named for
/// a compound checkpoint (its roles ARE the per-component layout), the single
/// weights file otherwise.
fn stored(local: &brain_modelstore::LocalModel, reference: &ModelRef, role: &str, arg: &str) -> Result<ResolvedModel, String> {
    let path = match &local.roles {
        Some(roles) => roles.get(role).ok_or_else(|| {
            format!(
                "--model {arg}: the checkpoint's roles are [{}], not {role:?}",
                roles.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        })?,
        None => &local.weights,
    };
    Ok(ResolvedModel { name: reference.to_string(), path: path.to_string_lossy().into_owned() })
}

fn has_weights_extension(arg: &str) -> bool {
    let lower = arg.to_ascii_lowercase();
    WEIGHTS_EXTENSIONS.iter().any(|e| lower.ends_with(e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::FakeHub;
    use brain_testutil::env_lock;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("cli-model-flag-{name}"));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// An explicit extension names the file outright -- used exactly as
    /// typed, with no probing and no store involvement (the store root is
    /// `None` here precisely to prove that).
    #[test]
    fn an_explicit_extension_is_used_exactly_as_typed() {
        let dir = scratch("explicit");
        let p = dir.join("dit.gguf");
        std::fs::write(&p, b"x").unwrap();
        let m = resolve_with(p.to_str().unwrap(), "dit", None, &FakeHub::new()).unwrap();
        assert_eq!(m.path, p.to_str().unwrap());
        assert_eq!(m.name, p.to_str().unwrap());

        let p = dir.join("dit.SAFETENSORS");
        std::fs::write(&p, b"x").unwrap();
        let m = resolve_with(p.to_str().unwrap(), "dit", None, &FakeHub::new()).unwrap();
        assert_eq!(m.path, p.to_str().unwrap(), "the extension case must not matter for recognition");
    }

    /// A named extension is a promise, not a hint: a missing file behind one
    /// is an error that names the file, never a silent fallthrough to
    /// probing or fetching.
    #[test]
    fn an_explicit_extension_with_no_file_behind_it_is_a_named_error() {
        let dir = scratch("explicit-missing");
        let p = dir.join("missing.gguf");
        let err = resolve_with(p.to_str().unwrap(), "dit", None, &FakeHub::new()).unwrap_err();
        assert!(err.contains("missing.gguf"), "{err}");
    }

    /// Without an extension both formats are probed, `.gguf` first (brain's
    /// own streaming format) -- so a stem with both files resolves to the
    /// GGUF, deterministically.
    #[test]
    fn without_an_extension_gguf_wins_over_safetensors() {
        let dir = scratch("probe-order");
        std::fs::write(dir.join("dit.gguf"), b"x").unwrap();
        std::fs::write(dir.join("dit.safetensors"), b"x").unwrap();
        let arg = dir.join("dit").to_str().unwrap().to_string();
        let m = resolve_with(&arg, "dit", None, &FakeHub::new()).unwrap();
        assert_eq!(m.path, format!("{arg}.gguf"));
        assert_eq!(m.name, arg);
    }

    #[test]
    fn without_an_extension_safetensors_is_found_when_there_is_no_gguf() {
        let dir = scratch("probe-st");
        std::fs::write(dir.join("dit.safetensors"), b"x").unwrap();
        let arg = dir.join("dit").to_str().unwrap().to_string();
        let m = resolve_with(&arg, "dit", None, &FakeHub::new()).unwrap();
        assert_eq!(m.path, format!("{arg}.safetensors"));
    }

    /// No file anywhere and the name is not a model id either: the error must
    /// say what was probed, so "I meant the path" and "I meant the repo" are
    /// both diagnosable.
    #[test]
    fn nothing_on_disk_and_not_a_model_id_names_what_was_tried() {
        let err = resolve_with("just some words", "dit", None, &FakeHub::new()).unwrap_err();
        assert!(err.contains("just some words.gguf") && err.contains("just some words.safetensors"), "{err}");
    }

    /// A `<vendor>/<repo>` id already in the model store resolves locally --
    /// with a hub that holds nothing, which would error on any call.
    #[test]
    fn a_model_id_already_in_the_store_resolves_without_the_hub() {
        let dir = scratch("store-hit");
        let repo = dir.join("Qwen/Qwen3-0.6B");
        std::fs::create_dir_all(&repo).unwrap();
        checkpoint::st::save_safetensors(
            repo.join("model.brain.safetensors").to_str().unwrap(),
            &[("weight".to_string(), vec![2], vec![1.0, 2.0])],
            &serde_json::json!({"hidden_size": 8}),
            None,
        )
        .unwrap();
        let m = resolve_with("Qwen/Qwen3-0.6B", "dit", Some(&dir), &FakeHub::new()).unwrap();
        assert_eq!(m.name, "Qwen/Qwen3-0.6B");
        assert!(m.path.ends_with("model.brain.safetensors"), "{}", m.path);
    }

    /// A compound (diffusers-pipeline) id not on disk is downloaded through
    /// the store's own ladder and hands back the role the caller named; a
    /// second resolve hits the store with an empty hub.
    #[test]
    fn a_compound_model_id_is_downloaded_and_resolves_to_the_named_role() {
        let _serial = env_lock(); // the download path is opt-in
        std::env::set_var("BRAIN_AUTO_FETCH", "1");
        let dir = scratch("compound");
        let mut hub = FakeHub::new();
        for f in [
            "model_index.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "tokenizer/tokenizer.json",
        ] {
            hub.add_file("Tongyi-MAI", "Z-Image-Turbo", "main", f, b"stub".to_vec());
        }
        let m = resolve_with("Tongyi-MAI/Z-Image-Turbo", "dit", Some(&dir), &hub).unwrap();
        assert_eq!(m.name, "Tongyi-MAI/Z-Image-Turbo");
        assert!(m.path.ends_with("transformer"), "{}", m.path);

        let again = resolve_with("Tongyi-MAI/Z-Image-Turbo", "dit", Some(&dir), &FakeHub::new()).unwrap();
        assert_eq!(again.path, m.path, "the second resolve must be a store hit, not a re-download");
        std::env::remove_var("BRAIN_AUTO_FETCH");
    }

    /// A GGUF-release id resolves to the one quant file itself (no roles),
    /// which is what `--model` on a quantized DiT looks like.
    #[test]
    fn a_gguf_release_id_downloads_and_resolves_to_the_gguf_itself() {
        let _serial = env_lock(); // the download path is opt-in
        std::env::set_var("BRAIN_AUTO_FETCH", "1");
        let dir = scratch("gguf-release");
        let mut hub = FakeHub::new();
        hub.add_file("unsloth", "Toy-GGUF", "main", "toy-Q8_0.gguf", crate::supply::tests::tiny_qwen3_gguf());
        let m = resolve_with("unsloth/Toy-GGUF", "dit", Some(&dir), &hub).unwrap();
        assert_eq!(m.name, "unsloth/Toy-GGUF-Q8_0");
        assert!(m.path.ends_with("Q8_0.gguf"), "{}", m.path);
        std::env::remove_var("BRAIN_AUTO_FETCH");
    }

    /// A model id that exists nowhere fails with the store's own fetch error,
    /// not a silent success or a panic.
    #[test]
    fn a_model_id_that_nowhere_exists_fails_with_the_fetch_error() {
        let _serial = env_lock(); // the download path is opt-in
        std::env::set_var("BRAIN_AUTO_FETCH", "1");
        let dir = scratch("fetch-miss");
        let err = resolve_with("someone/absent-model", "dit", Some(&dir), &FakeHub::new()).unwrap_err();
        assert!(err.contains("someone/absent-model"), "{err}");
        std::env::remove_var("BRAIN_AUTO_FETCH");
    }

    /// Fetching is opt-in: a store id with no local copy must fail naming
    /// both remedies (`brain pull`, `--autofetch`) rather than download --
    /// the hub here holds nothing, so any fetch attempt would be a different
    /// error and fail the assertion.
    #[test]
    fn a_store_id_is_not_fetched_unless_autofetch_is_enabled() {
        let _serial = env_lock();
        std::env::remove_var("BRAIN_AUTO_FETCH");
        let dir = scratch(&format!("gate-miss-{}", std::process::id()));
        let err = resolve_with("someone/absent-model", "dit", Some(&dir), &FakeHub::new()).unwrap_err();
        assert!(err.contains("brain pull someone/absent-model"), "{err}");
        assert!(err.contains("--autofetch"), "{err}");
        // A local store hit is NOT a fetch: it must resolve with the gate off.
        let repo = dir.join("Qwen/Qwen3-0.6B");
        std::fs::create_dir_all(&repo).unwrap();
        checkpoint::st::save_safetensors(
            repo.join("model.brain.safetensors").to_str().unwrap(),
            &[("weight".to_string(), vec![2], vec![1.0, 2.0])],
            &serde_json::json!({"hidden_size": 8}),
            None,
        )
        .unwrap();
        let m = resolve_with("Qwen/Qwen3-0.6B", "dit", Some(&dir), &FakeHub::new()).unwrap();
        assert!(m.path.ends_with("model.brain.safetensors"), "{}", m.path);
    }
}
