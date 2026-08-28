// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a user is allowed to TYPE for a model, as opposed to what the store
//! calls it. [`parse_pull_arg`] takes every spelling that reaches a command
//! line -- the canonical `<vendor>/<repo>[-<QUANT>]` id, and a HuggingFace
//! page URL pasted out of a browser -- and produces the one [`PullTarget`]
//! they all name: a repo, optionally a revision of it, optionally ONE
//! artifact inside it.
//!
//! ONE parser, in one place. [`parse_model_arg`] is a view of
//! [`parse_pull_arg`]'s result for the callers that only want the
//! [`ModelRef`], not a second thing that understands URLs -- two of those is
//! how a pasted `/blob/main/<file>` link came to silently lose its filename
//! and pull the whole repo instead.
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

/// What a path component following `<vendor>/<repo>` names.
#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    /// `/tree/<rev>` -- the repo itself, at a revision.
    Repo,
    /// `/blob|resolve|raw/<rev>/<path>` -- ONE artifact, at a revision.
    /// `/blob/` is what the address bar shows, `/resolve/` is the
    /// direct-download link, `/raw/` is the un-rendered view. Three spellings
    /// of the same file, so all three name the same artifact.
    File,
}

/// The repo views brain can act on, and what each names. A component that is
/// not in this table is refused BY NAME rather than dropped: `/commits/main`
/// and `/discussions/3` name a conversation about a repo, not its contents,
/// and quietly pulling the whole repo instead is doing something adjacent to
/// what was asked.
const VIEWS: &[(&str, View)] = &[("tree", View::Repo), ("blob", View::File), ("resolve", View::File), ("raw", View::File)];

/// [`VIEWS`]' keys, for an error that has to state the closed set.
fn view_names() -> String {
    VIEWS.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
}

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

/// What one `brain pull` argument names: always a repo, sometimes a specific
/// revision of it, sometimes exactly ONE artifact inside it.
///
/// The artifact is the whole point of the type. A file URL already answers
/// "which of this repo's 15 quantizations" with nothing left to infer, so it
/// deliberately bypasses quantization parsing entirely: the file is named, so
/// nothing needs guessing, and a name outside the closed [`Quant`] set
/// (`...-BF16.gguf`) is pullable this way and only this way.
///
/// [`Quant`]: brain_modelref::Quant
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PullTarget {
    /// The repo, quant suffix and adapter suffix included when the argument
    /// was a bare id (a URL never carries either -- HF has no such page).
    pub reference: ModelRef,
    /// The revision the argument named, percent-decoded (`refs%2Fpr%2F1` ->
    /// `refs/pr/1`). `None` means the repo's default branch.
    pub revision: Option<String>,
    /// The one artifact the argument named, as a repo-relative path with its
    /// directories intact (`text_encoder/model.safetensors`). `None` means
    /// the whole repo.
    pub artifact: Option<String>,
}

/// Parse one command-line model argument into the [`PullTarget`] it names.
///
/// Accepts the canonical id (`Qwen/Qwen3-8B`, quant suffix and all) and a
/// HuggingFace model URL with or without a scheme, with or without a trailing
/// slash, with or without a `www.` host, with or without a query string or
/// fragment, and with any of the repo views in [`VIEWS`]. A file view carries
/// its artifact through; a view brain does not recognise, or one that names
/// neither the repo nor a single file, is a named error.
pub fn parse_pull_arg(arg: &str) -> Result<PullTarget, RefArgError> {
    let trimmed = arg.trim();
    match split_url(trimmed)? {
        Some((host, path)) => parse_url_path(arg, &host, &path),
        None => Ok(PullTarget { reference: as_bare_ref(arg, trimmed)?, revision: None, artifact: None }),
    }
}

/// The [`ModelRef`]-only view of [`parse_pull_arg`], for callers that want
/// the repo and nothing else. Not a second parser: it drops fields, it does
/// not re-derive them.
pub fn parse_model_arg(arg: &str) -> Result<ModelRef, RefArgError> {
    parse_pull_arg(arg).map(|t| t.reference)
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

/// The `<vendor>/<repo>[/<view>/<rev>[/<path>]]` half of a model-page URL.
fn parse_url_path(arg: &str, host: &str, path: &str) -> Result<PullTarget, RefArgError> {
    if !is_model_page_host(host) {
        return Err(RefArgError::ForeignHost { arg: arg.to_string(), host: host.to_string() });
    }
    // A share link's `?...` / `#...` tail names a view of the page, not part
    // of the path -- `?download=true` is what HF's own download button emits.
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
    let reference = as_bare_ref(arg, &format!("{vendor}/{repo}"))?;

    let Some(view_name) = parts.get(2) else {
        return Ok(PullTarget { reference, revision: None, artifact: None });
    };
    let lowered = view_name.to_ascii_lowercase();
    let Some((_, view)) = VIEWS.iter().find(|(n, _)| *n == lowered) else {
        return Err(RefArgError::NotAModelPath {
            arg: arg.to_string(),
            why: format!("{view_name:?} is not a repo view brain can pull (expected one of: {})", view_names()),
        });
    };
    let Some(rev) = parts.get(3) else {
        return Err(RefArgError::NotAModelPath { arg: arg.to_string(), why: format!("/{view_name}/ names no revision") });
    };
    let revision = Some(safe_component(arg, &percent_decode(arg, rev)?, "revision")?);
    let rest = &parts[4..];
    match view {
        View::Repo if rest.is_empty() => Ok(PullTarget { reference, revision, artifact: None }),
        // A subdirectory is neither the repo nor one artifact, and pulling
        // the whole repo because a directory was named is exactly the silent
        // near-miss this parser exists to refuse.
        View::Repo => Err(RefArgError::NotAModelPath {
            arg: arg.to_string(),
            why: "the URL names a directory inside the repo; brain pulls a whole repo or one file".to_string(),
        }),
        View::File if rest.is_empty() => Err(RefArgError::NotAModelPath { arg: arg.to_string(), why: format!("/{view_name}/{rev}/ names no file") }),
        View::File => {
            // Decoded per component and rejoined, so a nested path survives
            // whole and a `%2F` inside one segment becomes a real separator.
            let mut segs = Vec::with_capacity(rest.len());
            for seg in rest {
                segs.push(percent_decode(arg, seg)?);
            }
            let joined = segs.join("/");
            // Re-split AFTER decoding: `%2e%2e` is a `..` that was not one
            // before, and this path becomes both a URL and a filename under
            // the store root.
            for c in joined.split('/') {
                safe_component(arg, c, "file path")?;
            }
            Ok(PullTarget { reference, revision, artifact: Some(joined) })
        }
    }
}

/// One decoded path component that is safe to use as both a URL segment and
/// a filename under the store root: non-empty, not a traversal primitive, no
/// control characters.
fn safe_component(arg: &str, c: &str, what: &str) -> Result<String, RefArgError> {
    if c.is_empty() || c == "." || c == ".." {
        return Err(RefArgError::NotAModelPath { arg: arg.to_string(), why: format!("the {what} contains an empty or \".\"/\"..\" component") });
    }
    if c.chars().any(char::is_control) {
        return Err(RefArgError::NotAModelPath { arg: arg.to_string(), why: format!("the {what} contains a control character") });
    }
    Ok(c.to_string())
}

/// `%XX` percent-decoding, strict: a malformed escape or a decode that is not
/// UTF-8 is refused rather than passed through as literal `%`. Strict beats
/// guessing here for the same reason it does in the quant grammar -- a
/// half-decoded path would be sent to the hub and 404 there instead.
fn percent_decode(arg: &str, s: &str) -> Result<String, RefArgError> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hi = bytes.get(i + 1).and_then(|c| (*c as char).to_digit(16));
            let lo = bytes.get(i + 2).and_then(|c| (*c as char).to_digit(16));
            match (hi, lo) {
                (Some(hi), Some(lo)) => {
                    out.push((hi * 16 + lo) as u8);
                    i += 3;
                }
                _ => return Err(RefArgError::NotAModelPath { arg: arg.to_string(), why: format!("{s:?} has a malformed percent-escape") }),
            }
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).map_err(|_| RefArgError::NotAModelPath { arg: arg.to_string(), why: format!("{s:?} percent-decodes to invalid UTF-8") })
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
