// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a user is allowed to TYPE for a model, as opposed to what the store
//! calls it. [`parse_model_arg`] takes the two spellings that reach a command
//! line -- the canonical `<vendor>/<repo>` id, and a HuggingFace page URL
//! pasted out of a browser -- and produces the one [`ModelRef`] both name.
//!
//! Pure: no network, no filesystem, no environment. The whole point is that
//! `brain pull https://huggingface.co/Qwen/Qwen3-8B` is decided before any
//! I/O happens, and that anything which is NOT a model reference is a named
//! error rather than a repo id that gets sent to the hub and 404s there.
//!
//! Strict beats guessing, the same rule [`brain_modelref`]'s quant-suffix
//! grammar follows: a URL shape this module does not recognise is refused
//! with what was expected, never silently reinterpreted.
//!
//! Swedish Embedded AB implements robust command-line interfaces for
//! machine-learning tooling for its clients. If your team needs expertise in
//! model distribution and CLI ergonomics then you can procure our services by
//! sending an email to info@swedishembedded.com.

use brain_modelref::ModelRef;

/// Hosts whose PAGES name a model repo, matched exactly (after dropping a
/// leading `www.`). Deliberately NOT [`crate::hub`]'s `ALLOWED_HOST_SUFFIXES`,
/// which answers a different question: that list decides where a download is
/// allowed to be redirected TO, so it accepts subdomains
/// (`cdn-lfs.huggingface.co`, `<region>.aws.cdn.hf.co`) because HF really
/// does serve blobs from them. A content host is not a model page, so
/// reusing the suffix rule here would turn a pasted CDN blob link into a
/// bogus repo id.
const MODEL_PAGE_HOSTS: &[&str] = &["huggingface.co", "hf.co"];

/// Path prefixes on a model-page host that are known NOT to be model repos.
/// Listed so a pasted dataset/space/collection link says what it actually is
/// instead of resolving to `datasets/openai`. Only sections that genuinely
/// collide with the `<vendor>/<repo>` shape need listing -- a one-component
/// path (`/models`, `/pricing`) already fails the "needs a vendor and a repo"
/// check below.
const NON_MODEL_SECTIONS: &[&str] = &["datasets", "spaces", "collections", "docs", "blog", "papers", "organizations", "settings", "posts"];

/// Path components that may legally FOLLOW `<vendor>/<repo>` on a model page:
/// the deep links a browser produces while you are looking at a repo. The
/// rest of the path is dropped -- `brain pull` fetches the repo, not the one
/// file you happened to be viewing.
const REPO_SUBPATHS: &[&str] = &["tree", "blob", "resolve", "raw", "commit", "commits", "discussions", "edit", "blame"];

/// Why an argument is not a model reference. Every variant's [`Display`]
/// names the offending input AND what was expected, because this error is
/// read by a human who just mistyped or pasted the wrong link.
///
/// [`Display`]: std::fmt::Display
#[derive(Debug)]
pub enum RefArgError {
    /// A URL whose host does not serve HuggingFace model pages.
    ForeignHost { arg: String, host: String },
    /// A model-host URL pointing at something that is not a model repo.
    NotAModelPath { arg: String, why: String },
    /// Not a URL at all, and not `<vendor>/<repo>` either.
    Grammar { arg: String, why: String },
}

/// The one sentence every variant appends, so the expected shape is stated
/// no matter which way the input was wrong.
const EXPECTED: &str = "expected a model reference <vendor>/<repo> (e.g. Qwen/Qwen3-8B) or a huggingface.co model URL (e.g. https://huggingface.co/Qwen/Qwen3-8B)";

impl std::fmt::Display for RefArgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefArgError::ForeignHost { arg, host } => {
                write!(f, "{arg:?}: {host} does not serve HuggingFace model pages -- {EXPECTED}")
            }
            RefArgError::NotAModelPath { arg, why } => write!(f, "{arg:?}: {why} -- {EXPECTED}"),
            RefArgError::Grammar { arg, why } => write!(f, "{arg:?}: {why} -- {EXPECTED}"),
        }
    }
}

impl std::error::Error for RefArgError {}

/// Parse one command-line model argument into the [`ModelRef`] it names.
///
/// Accepts the canonical id (`Qwen/Qwen3-8B`, quant suffix and all) and a
/// HuggingFace model URL with or without a scheme, with or without a trailing
/// slash, with or without a `www.` host, with or without a `/tree/<branch>`,
/// `/blob/...` or `/resolve/...` tail, and with or without a query string or
/// fragment. A URL's deep-link tail is DROPPED: the unit `brain pull` works
/// in is a repo, so `.../blob/main/model.safetensors` pulls the repo that
/// file belongs to.
pub fn parse_model_arg(arg: &str) -> Result<ModelRef, RefArgError> {
    let trimmed = arg.trim();
    match split_url(trimmed)? {
        Some((host, path)) => parse_url_path(arg, &host, &path),
        None => as_bare_ref(arg, trimmed),
    }
}

/// `Some((host, path))` when `s` is a URL, `None` when it is a bare id.
///
/// A scheme makes it a URL unconditionally. Without one, only a first
/// component that IS a known model-page host counts -- `deepseek.ai/model`
/// stays a bare `vendor/repo` (a vendor name may contain a dot), while
/// `huggingface.co/Qwen/Qwen3-8B` does not.
fn split_url(s: &str) -> Result<Option<(String, String)>, RefArgError> {
    let rest = match s.split_once("://") {
        Some((scheme, rest)) => {
            let scheme = scheme.to_ascii_lowercase();
            if scheme != "http" && scheme != "https" {
                return Err(RefArgError::NotAModelPath { arg: s.to_string(), why: format!("{scheme:?} is not an http(s) URL") });
            }
            rest
        }
        None => {
            let head = s.split('/').next().unwrap_or("").to_ascii_lowercase();
            if !is_model_page_host(&head) {
                return Ok(None);
            }
            s
        }
    };
    let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
    Ok(Some((host.to_ascii_lowercase(), path.to_string())))
}

/// Exactly a model-page host, tolerating a `www.` prefix. Never a suffix
/// match -- see [`MODEL_PAGE_HOSTS`].
fn is_model_page_host(host: &str) -> bool {
    let host = host.strip_prefix("www.").unwrap_or(host);
    MODEL_PAGE_HOSTS.contains(&host)
}

/// The `<vendor>/<repo>[/<subpath>...]` half of a model-page URL.
fn parse_url_path(arg: &str, host: &str, path: &str) -> Result<ModelRef, RefArgError> {
    if !is_model_page_host(host) {
        return Err(RefArgError::ForeignHost { arg: arg.to_string(), host: host.to_string() });
    }
    // A share link's `?...` / `#...` tail names a view, not a repo.
    let path = path.split(['?', '#']).next().unwrap_or("");
    let parts: Vec<&str> = path.split('/').filter(|c| !c.is_empty()).collect();

    if let Some(first) = parts.first() {
        if NON_MODEL_SECTIONS.contains(&first.to_ascii_lowercase().as_str()) {
            return Err(RefArgError::NotAModelPath { arg: arg.to_string(), why: format!("/{first}/ is HuggingFace's {first} section, not a model repo") });
        }
    }
    let (vendor, repo) = match parts.as_slice() {
        [vendor, repo, ..] => (*vendor, *repo),
        _ => return Err(RefArgError::NotAModelPath { arg: arg.to_string(), why: "the URL names no <vendor>/<repo>".to_string() }),
    };
    // Anything after the repo must be a repo view brain knows how to ignore.
    // An unrecognised tail is refused rather than guessed at, so a future HF
    // URL shape that means something else does not get pulled as a repo.
    if let Some(tail) = parts.get(2) {
        if !REPO_SUBPATHS.contains(&tail.to_ascii_lowercase().as_str()) {
            return Err(RefArgError::NotAModelPath { arg: arg.to_string(), why: format!("{tail:?} is not a repo view brain recognises") });
        }
    }
    as_bare_ref(arg, &format!("{vendor}/{repo}"))
}

/// The final gate for both paths: the [`ModelRef`] grammar itself (reserved
/// characters, quant suffix, `.`/`..` rejection). One grammar, one place.
fn as_bare_ref(arg: &str, candidate: &str) -> Result<ModelRef, RefArgError> {
    ModelRef::parse(candidate).map_err(|e| RefArgError::Grammar { arg: arg.to_string(), why: e.to_string() })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host rule is an EXACT match, not a suffix match: the lookalike
    /// domains `hub::host_is_allowed` refuses must be refused here too, and
    /// so must HF's own content subdomains (which that other list accepts on
    /// purpose).
    #[test]
    fn only_exact_model_page_hosts_are_accepted() {
        for host in ["huggingface.co", "www.huggingface.co", "hf.co", "HuggingFace.co"] {
            assert!(is_model_page_host(&host.to_ascii_lowercase()), "{host} should be a model page host");
        }
        for host in ["evilhuggingface.co", "huggingface.co.evil.example", "cdn-lfs.huggingface.co", "xethub.hf.co", "github.com"] {
            assert!(!is_model_page_host(host), "{host} must not be a model page host");
        }
    }

    /// A dotted VENDOR name is not a host: without a scheme, only a real
    /// model-page host turns the argument into a URL.
    #[test]
    fn a_dotted_vendor_is_still_a_bare_reference() {
        assert_eq!(parse_model_arg("deepseek.ai/DeepSeek-V3").unwrap().to_string(), "deepseek.ai/DeepSeek-V3");
    }

    /// A non-http scheme is refused by name rather than being stripped.
    #[test]
    fn a_non_http_scheme_is_refused() {
        let err = parse_model_arg("ftp://huggingface.co/Qwen/Qwen3-8B").unwrap_err().to_string();
        assert!(err.contains("ftp"), "{err}");
    }
}
