// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain pull <model>` accepts what people actually paste: a bare
//! `<vendor>/<repo>` id, or a HuggingFace page URL in any of the shapes a
//! browser address bar produces. Both spellings must reach the SAME
//! [`brain_modelref::ModelRef`], and anything that is not a recognisable model
//! reference must be a named error -- never a silent fallback that treats a
//! whole URL as a repo id and then asks the hub for `https:/huggingface.co`.
//!
//! Swedish Embedded AB implements robust command-line model-reference parsing
//! for its clients. If your team needs expertise in CLI ergonomics for
//! machine-learning tooling then you can procure our services by sending an
//! email to info@swedishembedded.com.

use brain_modelstore::refurl::{parse_model_arg, RefArgError};

/// Every accepted spelling of the same model resolves to the same ref.
#[test]
fn every_url_shape_reaches_the_same_model_ref() {
    let want = "Qwen/Qwen3-8B";
    for arg in [
        // The bare id -- the canonical spelling.
        "Qwen/Qwen3-8B",
        // Full page URL, the copy-from-address-bar case.
        "https://huggingface.co/Qwen/Qwen3-8B",
        // ... with a trailing slash.
        "https://huggingface.co/Qwen/Qwen3-8B/",
        // ... without the scheme (what a copy from a link label gives).
        "huggingface.co/Qwen/Qwen3-8B",
        // ... with the www host.
        "https://www.huggingface.co/Qwen/Qwen3-8B",
        // ... over the short domain HF also serves pages on.
        "https://hf.co/Qwen/Qwen3-8B",
        // ... plain http.
        "http://huggingface.co/Qwen/Qwen3-8B",
        // Deep links inside the repo: the branch view, a file view, and the
        // raw-download link. All three name the same repo.
        "https://huggingface.co/Qwen/Qwen3-8B/tree/main",
        "https://huggingface.co/Qwen/Qwen3-8B/tree/refs%2Fpr%2F1",
        "https://huggingface.co/Qwen/Qwen3-8B/blob/main/model.safetensors",
        "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json",
        // ... and with the query string / fragment a share link carries.
        "https://huggingface.co/Qwen/Qwen3-8B?library=transformers",
        "https://huggingface.co/Qwen/Qwen3-8B#usage",
    ] {
        let got = parse_model_arg(arg).unwrap_or_else(|e| panic!("{arg:?} should parse: {e}"));
        assert_eq!(got.to_string(), want, "{arg:?} parsed to the wrong model");
    }
}

/// The quant suffix is part of the reference grammar and must survive the
/// bare-id path (a URL never carries one -- HF has no such page).
#[test]
fn a_quant_suffix_survives_the_bare_id_path() {
    assert_eq!(parse_model_arg("Qwen/Qwen3-0.6B-Q8_0").unwrap().to_string(), "Qwen/Qwen3-0.6B-Q8_0");
}

/// Nothing that is not a model reference may quietly become one. Asserted on
/// the error's REASON, not on its rendered text: every variant's `Display`
/// echoes the offending argument and appends the same "expected ..."
/// sentence, so a `contains("datasets")` check against the whole message is
/// satisfied by the echoed URL alone and would pass with the section list
/// deleted. Match the variant and read the reason field.
#[test]
fn a_non_model_reference_is_a_named_error_not_a_silent_fallback() {
    // A URL under a host that does not serve model pages at all.
    match parse_model_arg("https://github.com/Qwen/Qwen3-8B") {
        Err(RefArgError::ForeignHost { host, .. }) => assert_eq!(host, "github.com"),
        other => panic!("a github URL must be a foreign host, got {other:?}"),
    }

    // HuggingFace's own non-model sections. The three-component ones
    // (`/spaces/<org>/<name>`) would otherwise resolve to `spaces/<org>`;
    // the two-component `/datasets/squad` would resolve to `datasets/squad`,
    // which is a perfectly well-formed model reference for a repo that does
    // not exist -- exactly the silent fallback this must not do.
    for (arg, section) in [
        ("https://huggingface.co/datasets/openai/gsm8k", "datasets"),
        ("https://huggingface.co/datasets/squad", "datasets"),
        ("https://huggingface.co/spaces/Qwen/Qwen3-Demo", "spaces"),
        ("https://huggingface.co/collections/Qwen/qwen3-abc", "collections"),
    ] {
        match parse_model_arg(arg) {
            Err(RefArgError::NotAModelPath { why, .. }) => {
                assert!(why.contains(section), "{arg:?} should be refused as the {section} section, reason was {why:?}")
            }
            other => panic!("{arg:?} must be refused as a non-model section, got {other:?}"),
        }
    }

    // A model-host URL that names no repo.
    for arg in ["https://huggingface.co/Qwen", "https://huggingface.co/"] {
        match parse_model_arg(arg) {
            Err(RefArgError::NotAModelPath { why, .. }) => assert!(why.contains("<vendor>/<repo>"), "{arg:?} reason was {why:?}"),
            other => panic!("{arg:?} names no repo and must be refused, got {other:?}"),
        }
    }

    // Not a URL and not a reference either -- the reference grammar's own
    // refusal, surfaced with the offending argument attached.
    for arg in ["Qwen3-8B", ""] {
        match parse_model_arg(arg) {
            Err(RefArgError::Grammar { arg: echoed, .. }) => assert_eq!(echoed, arg),
            other => panic!("{arg:?} is not a model reference and must be refused, got {other:?}"),
        }
    }

    // Whichever way it was wrong, the rendered message states the shape that
    // WOULD have worked -- this is what a user reads.
    for arg in ["https://huggingface.co/Qwen", "https://github.com/a/b", "Qwen3-8B", "https://huggingface.co/datasets/squad"] {
        let err = parse_model_arg(arg).unwrap_err().to_string();
        assert!(err.contains("<vendor>/<repo>"), "{arg:?}: {err}");
    }
}

/// A URL under an HF-owned CONTENT host is not a model page. The download
/// allowlist accepts those hosts by SUFFIX (a redirect legitimately lands on
/// `cdn-lfs.huggingface.co` / `<region>.aws.cdn.hf.co`); the argument parser
/// must match the host EXACTLY, or this pasted link would become the model
/// reference `Qwen/Qwen3-8B` fetched from the wrong idea of a page.
#[test]
fn an_hf_content_cdn_url_is_not_a_model_page() {
    match parse_model_arg("https://cdn-lfs.huggingface.co/Qwen/Qwen3-8B") {
        Err(RefArgError::ForeignHost { host, .. }) => assert_eq!(host, "cdn-lfs.huggingface.co"),
        other => panic!("a CDN blob host is not a model page, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// A pull argument names a repo, and MAY name a revision and one artifact
// inside it. `parse_pull_arg` is the structured form of the same single
// parser `parse_model_arg` above is the ref-only view of.
// ---------------------------------------------------------------------------

use brain_modelref::Quant;
use brain_modelstore::refurl::parse_pull_arg;

/// The repo every row below names, whichever way it is spelled.
const REPO: &str = "unsloth/FLUX.2-klein-9B-GGUF";
/// One artifact inside it, at the repo root.
const FILE: &str = "flux-2-klein-9b-Q8_0.gguf";

/// One row per URL shape a user can produce, each asserting all three parts
/// of the result: the repo, the revision, and the artifact. The revision and
/// the artifact are exactly what used to be dropped on the floor, so a table
/// that only checked the repo (the one above) passed while `brain pull` of a
/// file URL fetched something else entirely.
#[test]
fn every_pull_url_shape_resolves_to_the_same_repo_revision_and_artifact() {
    let rows: &[(&str, Option<&str>, Option<&str>)] = &[
        // Bare id, and the URL shapes that name the whole repo.
        ("unsloth/FLUX.2-klein-9B-GGUF", None, None),
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF", None, None),
        ("http://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF", None, None),
        ("https://www.huggingface.co/unsloth/FLUX.2-klein-9B-GGUF", None, None),
        ("huggingface.co/unsloth/FLUX.2-klein-9B-GGUF", None, None),
        ("https://hf.co/unsloth/FLUX.2-klein-9B-GGUF", None, None),
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/", None, None),
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF?library=diffusers", None, None),
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF#usage", None, None),
        // The branch view names a revision and no artifact.
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/tree/main", Some("main"), None),
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/tree/9a3b0c1d", Some("9a3b0c1d"), None),
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/tree/refs%2Fpr%2F1", Some("refs/pr/1"), None),
        // The file views: `/blob/` is what the address bar shows, `/resolve/`
        // is the direct-download link. Same artifact, two spellings.
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/blob/main/flux-2-klein-9b-Q8_0.gguf", Some("main"), Some(FILE)),
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/resolve/main/flux-2-klein-9b-Q8_0.gguf", Some("main"), Some(FILE)),
        // HF's own download button appends a query string.
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/resolve/main/flux-2-klein-9b-Q8_0.gguf?download=true", Some("main"), Some(FILE)),
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/blob/main/flux-2-klein-9b-Q8_0.gguf#L1", Some("main"), Some(FILE)),
        // A non-default revision in the file forms, including the
        // percent-encoded one the branch view already accepts.
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/blob/refs%2Fpr%2F1/flux-2-klein-9b-Q8_0.gguf", Some("refs/pr/1"), Some(FILE)),
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/resolve/9a3b0c1d/flux-2-klein-9b-Q8_0.gguf", Some("9a3b0c1d"), Some(FILE)),
        // A NESTED artifact path survives whole -- keeping only the last
        // segment would fetch the wrong URL and land the wrong name.
        (
            "https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/blob/main/text_encoder/model.safetensors",
            Some("main"),
            Some("text_encoder/model.safetensors"),
        ),
        // The raw view is a file view too.
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/raw/main/README.md", Some("main"), Some("README.md")),
    ];
    for (arg, revision, artifact) in rows {
        let got = parse_pull_arg(arg).unwrap_or_else(|e| panic!("{arg:?} should parse: {e}"));
        assert_eq!(got.reference.to_string(), REPO, "{arg:?}: wrong repo");
        assert_eq!(got.revision.as_deref(), *revision, "{arg:?}: wrong revision");
        assert_eq!(got.artifact.as_deref(), *artifact, "{arg:?}: wrong artifact");
    }
}

/// A URL that names neither the whole repo nor one artifact must be refused
/// by name. Truncating it back to the repo and pulling that instead is doing
/// something adjacent to what was asked, which is worse than an error.
#[test]
fn a_url_that_names_neither_a_repo_nor_one_artifact_is_refused_by_name() {
    let rows: &[(&str, &str)] = &[
        // A file view with no file.
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/blob/main", "names no file"),
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/resolve/main/", "names no file"),
        // A view with no revision at all.
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/blob", "names no revision"),
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/tree", "names no revision"),
        // Views that name a conversation about the repo, not its contents.
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/commits/main", "commits"),
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/discussions/3", "discussions"),
        // A subdirectory is not one artifact.
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/tree/main/text_encoder", "directory"),
        // Malformed and unsafe artifact paths.
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/blob/main/%zz.gguf", "percent"),
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/blob/main/../../etc/passwd", ".."),
        ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/blob/%2e%2e/x.gguf", ".."),
    ];
    for (arg, needle) in rows {
        match parse_pull_arg(arg) {
            Err(e) => {
                let why = e.to_string();
                assert!(why.contains(needle), "{arg:?}: expected the refusal to mention {needle:?}, got {why:?}");
            }
            Ok(t) => panic!("{arg:?} must be refused, got {t:?}"),
        }
    }
}

/// Which quantization to pull is spelled with the reference grammar's OWN
/// `-<QUANT>` suffix, not a second `:QUANT` spelling invented for pull: `:`
/// is already the adapter separator (`vendor/repo:owner:name:tag`), so a
/// colon form would collide with a grammar that exists.
#[test]
fn a_quantization_is_named_by_the_reference_grammars_own_suffix() {
    let t = parse_pull_arg("unsloth/FLUX.2-klein-9B-GGUF-Q8_0").unwrap();
    assert_eq!(t.reference.repo(), "FLUX.2-klein-9B-GGUF");
    assert_eq!(t.reference.quant(), Some(Quant::Q8_0));
    assert_eq!(t.artifact, None, "a quant suffix names a quantization, not a file path");
    assert!(parse_pull_arg("unsloth/FLUX.2-klein-9B-GGUF:Q8_0").is_err(), "the colon form is the adapter grammar and must not be reinterpreted");
}
