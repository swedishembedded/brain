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
