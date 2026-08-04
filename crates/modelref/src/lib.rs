// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The fully-qualified model reference grammar: `<vendor>/<repo>[-<QUANT>]`,
//! matching the HuggingFace URL a model came from exactly (case included).
//!
//! This is deliberately a leaf crate — no filesystem access, no network, no
//! dependency on any other brain crate — so it can be depended on from
//! `checkpoint`, `modelstore`, `apiserve`, `dbus`, `cli`, and any model crate's
//! `caps.rs` without creating a dependency cycle, and it stays usable from wasm.
//!
//! # Grammar
//!
//! ```text
//! ModelRef  ::= vendor "/" repo ("-" QUANT)?
//! vendor    ::= non-empty, no "/"
//! repo      ::= non-empty, no "/", does not itself end in "-" QUANT
//! QUANT     ::= one of the closed set in [`Quant`] (see below)
//! ```
//!
//! Three rules make this unambiguous, each earning its keep against a real
//! upstream name:
//!
//! 1. **Byte-exact case.** `qwen/qwen3-0.6b` is not `Qwen/Qwen3-0.6B` — the
//!    reference must match the HuggingFace URL exactly, or resolution against
//!    the real upstream repo fails.
//! 2. **Quant tokens are a closed, uppercase-only set.** `-Q8_0` is a quant
//!    suffix; `-q8_0` is just part of the repo name. Strict beats guessing.
//! 3. **`BF16`/`F16`/`F32`/`FP8` are NOT quant tokens** — they are base-repo
//!    dtypes, and treating them as quantizations would make
//!    `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-Q8_0` unparseable (or parsed
//!    wrong). Excluding them from the quant set is what makes that name parse
//!    as repo `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` + quant `Q8_0`.
//!
//! Exactly one quant suffix is accepted: `Qwen/Qwen3-0.6B-Q8_0-Q4_0` is a
//! [`RefError::TwoQuants`], not a silently-truncated parse.
//!
//! # Reserved vendors
//!
//! `brain`, `local`, and `test` never resolve to a network fetch (see
//! [`is_reserved`]) — the grammar itself is the security gate: a name with no
//! `/`, or under a reserved vendor, never triggers I/O.
//!
//! * `brain/…` — built-in and env-gated dev residents (`brain/demo`,
//!   `brain/mock`, `brain/imageops`, `brain/fastvlm`, …).
//! * `local/…` — a file a user dropped into the model store with no upstream
//!   provenance.
//! * `test/…` — reserved, unused today; for `--testmode` mocks later.

#![forbid(unsafe_code)]

use std::fmt;

use serde::{Deserialize, Serialize};

pub mod alias;

/// A fully-qualified model reference: `<vendor>/<repo>[-<QUANT>]`.
///
/// `Display` round-trips: `ModelRef::parse(&r.to_string()).unwrap() == r` for
/// every `r` (see the `roundtrips` proptest-style unit test).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ModelRef {
    vendor: String,
    /// The repo name WITHOUT any quant suffix — `base_repo` in store-layout
    /// terms (see `modelstore`'s directory-keying doc).
    repo: String,
    quant: Option<Quant>,
}

impl ModelRef {
    /// Parse `s` as `<vendor>/<repo>[-<QUANT>]`. See the module docs for the
    /// exact grammar and [`RefError`] for what can go wrong.
    pub fn parse(s: &str) -> Result<ModelRef, RefError> {
        if s.is_empty() {
            return Err(RefError::NoVendor);
        }
        let mut parts = s.splitn(2, '/');
        let vendor = parts.next().unwrap_or("");
        let rest = match parts.next() {
            Some(r) => r,
            None => return Err(RefError::NoVendor),
        };
        if vendor.is_empty() {
            return Err(RefError::EmptySegment);
        }
        if rest.is_empty() {
            return Err(RefError::EmptySegment);
        }
        if rest.contains('/') {
            return Err(RefError::TooManySegments);
        }
        if vendor.contains(char::is_whitespace) || rest.contains(char::is_whitespace) {
            return Err(RefError::BadChar);
        }

        // `repo` is non-empty on both arms below: `strip_quant_suffix` only
        // returns `Some` when a non-empty base remains (see its doc comment),
        // and `rest` was already checked non-empty above.
        let (repo, quant) = match strip_quant_suffix(rest) {
            Some((repo, q)) => {
                // A second quant suffix on the already-stripped repo is an error,
                // not a second silently-accepted level of stripping.
                if strip_quant_suffix(repo).is_some() {
                    return Err(RefError::TwoQuants);
                }
                (repo, Some(q))
            }
            None => (rest, None),
        };

        Ok(ModelRef { vendor: vendor.to_string(), repo: repo.to_string(), quant })
    }

    /// Build a ref directly (no parsing) — for callers that already have the
    /// three fields (e.g. `modelstore` composing a ref from a store directory).
    pub fn new(vendor: impl Into<String>, repo: impl Into<String>, quant: Option<Quant>) -> ModelRef {
        ModelRef { vendor: vendor.into(), repo: repo.into(), quant }
    }

    pub fn vendor(&self) -> &str {
        &self.vendor
    }

    /// The repo name, WITHOUT the quant suffix.
    pub fn repo(&self) -> &str {
        &self.repo
    }

    pub fn quant(&self) -> Option<Quant> {
        self.quant
    }

    /// This ref with its quant suffix removed — the base model. A no-op if
    /// already unquantized. This is the store-directory key (see `modelstore`).
    pub fn base(&self) -> ModelRef {
        ModelRef { vendor: self.vendor.clone(), repo: self.repo.clone(), quant: None }
    }

    /// This ref with `quant` applied (replacing any existing one).
    pub fn with_quant(&self, quant: Quant) -> ModelRef {
        ModelRef { vendor: self.vendor.clone(), repo: self.repo.clone(), quant: Some(quant) }
    }

    /// Is this vendor one of the reserved, never-fetched namespaces
    /// (`brain`/`local`/`test`)? See the module docs.
    pub fn is_reserved(&self) -> bool {
        is_reserved(&self.vendor)
    }
}

/// Is `vendor` one of the reserved, never-fetched namespaces? Free function so
/// a caller checking a raw vendor string (before constructing a [`ModelRef`])
/// doesn't need one.
pub fn is_reserved(vendor: &str) -> bool {
    matches!(vendor, "brain" | "local" | "test")
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.vendor, self.repo)?;
        if let Some(q) = self.quant {
            write!(f, "-{}", q.as_str())?;
        }
        Ok(())
    }
}

impl std::str::FromStr for ModelRef {
    type Err = RefError;
    fn from_str(s: &str) -> Result<ModelRef, RefError> {
        ModelRef::parse(s)
    }
}

/// Why a string failed to parse as a [`ModelRef`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefError {
    /// No `/` at all (or the string was empty).
    NoVendor,
    /// The vendor segment, or the repo segment, is empty (`"/repo"` or
    /// `"vendor/"`).
    EmptySegment,
    /// More than one `/` — `<vendor>/<repo>` allows exactly one.
    TooManySegments,
    /// Whitespace in the vendor or repo segment.
    BadChar,
    /// The repo segment ends in two quant suffixes, e.g. `...-Q8_0-Q4_0`.
    /// Specifying more than one quantization is not allowed.
    TwoQuants,
}

impl fmt::Display for RefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            RefError::NoVendor => "model ref must be \"<vendor>/<repo>\" (no '/' found)",
            RefError::EmptySegment => "model ref has an empty vendor or repo segment",
            RefError::TooManySegments => "model ref must have exactly one '/'",
            RefError::BadChar => "model ref must not contain whitespace",
            RefError::TwoQuants => "model ref names two quantizations (only one is allowed)",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for RefError {}

/// If `repo` ends with `-<QUANT>` for exactly one token in the closed
/// [`Quant`] set, return `(repo_without_suffix, quant)`. Longest-token-first so
/// no token can be shadowed by a shorter one that happens to also match a
/// trailing substring (in practice none of the 15 tokens are suffixes of each
/// other, but checking longest-first keeps that an invariant, not a lucky
/// accident of the current table).
fn strip_quant_suffix(repo: &str) -> Option<(&str, Quant)> {
    let mut candidates: Vec<Quant> = Quant::ALL.to_vec();
    candidates.sort_by_key(|q| std::cmp::Reverse(q.as_str().len()));
    for q in candidates {
        let suffix = q.as_str();
        if let Some(stripped) = repo.strip_suffix(suffix) {
            if let Some(base) = stripped.strip_suffix('-') {
                if !base.is_empty() {
                    return Some((base, q));
                }
            }
        }
    }
    None
}

/// The closed set of quantization tokens brain's GGUF reader
/// (`crates/checkpoint/src/gguf.rs`) can dequantize — legacy blocks-of-32 and
/// k-quant superblocks-of-256. Every variant's [`Quant::as_str`] is its exact,
/// case-sensitive wire token (matching the `general.file_type` convention
/// GGUF/llama.cpp-ecosystem files use, so a name parsed from a HuggingFace
/// `*-GGUF` repo's file list round-trips unchanged).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Quant {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3KS,
    Q3KM,
    Q3KL,
    Q4KS,
    Q4KM,
    Q5KS,
    Q5KM,
    Q6K,
    Q8K,
}

impl Quant {
    /// Every variant, in declaration order (not sorted).
    pub const ALL: [Quant; 15] = [
        Quant::Q4_0,
        Quant::Q4_1,
        Quant::Q5_0,
        Quant::Q5_1,
        Quant::Q8_0,
        Quant::Q2K,
        Quant::Q3KS,
        Quant::Q3KM,
        Quant::Q3KL,
        Quant::Q4KS,
        Quant::Q4KM,
        Quant::Q5KS,
        Quant::Q5KM,
        Quant::Q6K,
        Quant::Q8K,
    ];

    /// The exact, case-sensitive wire token — what appears after the `-` in a
    /// [`ModelRef`], and in a GGUF `general.file_type` string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Quant::Q4_0 => "Q4_0",
            Quant::Q4_1 => "Q4_1",
            Quant::Q5_0 => "Q5_0",
            Quant::Q5_1 => "Q5_1",
            Quant::Q8_0 => "Q8_0",
            Quant::Q2K => "Q2_K",
            Quant::Q3KS => "Q3_K_S",
            Quant::Q3KM => "Q3_K_M",
            Quant::Q3KL => "Q3_K_L",
            Quant::Q4KS => "Q4_K_S",
            Quant::Q4KM => "Q4_K_M",
            Quant::Q5KS => "Q5_K_S",
            Quant::Q5KM => "Q5_K_M",
            Quant::Q6K => "Q6_K",
            Quant::Q8K => "Q8_K",
        }
    }

    /// Parse the exact, case-sensitive wire token (no leading `-`). `None` for
    /// anything outside the closed set — including a lowercase spelling.
    pub fn parse(s: &str) -> Option<Quant> {
        Quant::ALL.into_iter().find(|q| q.as_str() == s)
    }

    /// Is this a k-quant super-block-of-256 type (as opposed to a legacy
    /// block-of-32 type)? Drives which quantizer algorithm applies (see
    /// `checkpoint::quant`).
    pub const fn is_k_quant(self) -> bool {
        !matches!(self, Quant::Q4_0 | Quant::Q4_1 | Quant::Q5_0 | Quant::Q5_1 | Quant::Q8_0)
    }

    /// A rough fidelity ordering, highest first — for the monotonicity
    /// assertion in the quantizer's correctness gate (`checkpoint::quant`'s
    /// tests), NOT for choosing a quantization: `Q8_0 > Q6_K > Q5_K_M >
    /// Q5_K_S > Q4_K_M > Q4_K_S > Q3_K_L > Q3_K_M > Q3_K_S > Q2_K`. `Q4_0`/
    /// `Q4_1`/`Q5_0`/`Q5_1`/`Q8_K` are legacy/internal types with no fixed
    /// place in this ladder and rank last (never chosen by the ladder logic).
    pub const fn fidelity_rank(self) -> u8 {
        match self {
            Quant::Q8_0 => 0,
            Quant::Q6K => 1,
            Quant::Q5KM => 2,
            Quant::Q5KS => 3,
            Quant::Q4KM => 4,
            Quant::Q4KS => 5,
            Quant::Q3KL => 6,
            Quant::Q3KM => 7,
            Quant::Q3KS => 8,
            Quant::Q2K => 9,
            Quant::Q4_0 | Quant::Q4_1 | Quant::Q5_0 | Quant::Q5_1 | Quant::Q8K => 255,
        }
    }
}

impl fmt::Display for Quant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_vendor_repo() {
        let r = ModelRef::parse("Qwen/Qwen3-0.6B").unwrap();
        assert_eq!(r.vendor(), "Qwen");
        assert_eq!(r.repo(), "Qwen3-0.6B");
        assert_eq!(r.quant(), None);
    }

    #[test]
    fn parses_vendor_repo_quant() {
        let r = ModelRef::parse("Qwen/Qwen3-0.6B-Q8_0").unwrap();
        assert_eq!(r.vendor(), "Qwen");
        assert_eq!(r.repo(), "Qwen3-0.6B");
        assert_eq!(r.quant(), Some(Quant::Q8_0));
    }

    #[test]
    fn bf16_is_part_of_the_repo_name_not_a_quant() {
        // The exact example from .todo/cleanup-examples.md.
        let r = ModelRef::parse("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-Q8_0").unwrap();
        assert_eq!(r.vendor(), "nvidia");
        assert_eq!(r.repo(), "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16");
        assert_eq!(r.quant(), Some(Quant::Q8_0));
    }

    #[test]
    fn a_bare_bf16_repo_has_no_quant() {
        let r = ModelRef::parse("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16").unwrap();
        assert_eq!(r.repo(), "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16");
        assert_eq!(r.quant(), None);
    }

    #[test]
    fn case_sensitive_vendor_and_repo() {
        assert_ne!(ModelRef::parse("Qwen/Qwen3-0.6B").unwrap(), ModelRef::parse("qwen/qwen3-0.6b").unwrap());
        // Lowercase "vendor" segment is syntactically valid (case sensitivity is
        // about matching the REAL upstream name, not a charset restriction) --
        // resolution against the real repo is what enforces the exact case, and
        // that lives in `modelstore`, not here.
        assert!(ModelRef::parse("qwen/Qwen3-0.6B").is_ok());
    }

    #[test]
    fn lowercase_quant_suffix_is_not_a_quant() {
        let r = ModelRef::parse("Qwen/Qwen3-0.6B-q8_0").unwrap();
        assert_eq!(r.repo(), "Qwen3-0.6B-q8_0");
        assert_eq!(r.quant(), None);
    }

    #[test]
    fn two_quant_suffixes_is_rejected() {
        let err = ModelRef::parse("Qwen/Qwen3-0.6B-Q8_0-Q4_0").unwrap_err();
        assert_eq!(err, RefError::TwoQuants);
    }

    #[test]
    fn every_quant_token_round_trips_as_a_suffix() {
        for q in Quant::ALL {
            let s = format!("V/R-{}", q.as_str());
            let r = ModelRef::parse(&s).unwrap_or_else(|e| panic!("{s}: {e}"));
            assert_eq!(r.quant(), Some(q), "{s}");
            assert_eq!(r.repo(), "R", "{s}");
        }
    }

    #[test]
    fn no_slash_is_novendor() {
        assert_eq!(ModelRef::parse("mock").unwrap_err(), RefError::NoVendor);
        assert_eq!(ModelRef::parse("").unwrap_err(), RefError::NoVendor);
    }

    #[test]
    fn too_many_slashes_is_rejected() {
        assert_eq!(ModelRef::parse("a/b/c").unwrap_err(), RefError::TooManySegments);
    }

    #[test]
    fn empty_vendor_or_repo_is_rejected() {
        assert_eq!(ModelRef::parse("/repo").unwrap_err(), RefError::EmptySegment);
        assert_eq!(ModelRef::parse("vendor/").unwrap_err(), RefError::EmptySegment);
        // "Q8_0" alone (no leading '-') has no quant delimiter, so it is just a
        // literal (unusual, but valid) repo name, not a stripped-to-empty repo.
        assert_eq!(ModelRef::parse("vendor/Q8_0").unwrap().repo(), "Q8_0");
        // "-Q8_0" DOES have the delimiter, but stripping it would leave an
        // EMPTY base -- strip_quant_suffix refuses that (it only strips when a
        // non-empty repo remains), so this falls through to a literal
        // (unusual, but valid) repo name "-Q8_0", not an error. There is no
        // input shaped like "<vendor>/-<QUANT>" that gets rejected as
        // EmptySegment via quant-stripping specifically, by construction.
        assert_eq!(ModelRef::parse("vendor/-Q8_0").unwrap().repo(), "-Q8_0");
    }

    #[test]
    fn whitespace_is_rejected() {
        assert_eq!(ModelRef::parse("Qwen/Qwen3 0.6B").unwrap_err(), RefError::BadChar);
    }

    #[test]
    fn display_round_trips() {
        let cases = [
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-0.6B-Q8_0",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-Q4_K_M",
            "LiquidAI/LFM2.5-350M",
            "brain/mock",
            "local/my-checkpoint",
        ];
        for s in cases {
            let r = ModelRef::parse(s).unwrap_or_else(|e| panic!("{s}: {e}"));
            assert_eq!(r.to_string(), s, "round-trip for {s}");
            assert_eq!(ModelRef::parse(&r.to_string()).unwrap(), r);
        }
    }

    #[test]
    fn reserved_vendors() {
        assert!(ModelRef::parse("brain/mock").unwrap().is_reserved());
        assert!(ModelRef::parse("local/x").unwrap().is_reserved());
        assert!(ModelRef::parse("test/x").unwrap().is_reserved());
        assert!(!ModelRef::parse("Qwen/Qwen3-0.6B").unwrap().is_reserved());
    }

    #[test]
    fn base_strips_quant_and_with_quant_applies() {
        let q = ModelRef::parse("Qwen/Qwen3-0.6B-Q8_0").unwrap();
        let base = q.base();
        assert_eq!(base.to_string(), "Qwen/Qwen3-0.6B");
        assert_eq!(base.quant(), None);
        assert_eq!(base.with_quant(Quant::Q4KM).quant(), Some(Quant::Q4KM));
        assert_eq!(base.with_quant(Quant::Q4KM).to_string(), "Qwen/Qwen3-0.6B-Q4_K_M");
    }

    #[test]
    fn fidelity_rank_is_monotonic_in_the_documented_order() {
        let order = [
            Quant::Q8_0,
            Quant::Q6K,
            Quant::Q5KM,
            Quant::Q5KS,
            Quant::Q4KM,
            Quant::Q4KS,
            Quant::Q3KL,
            Quant::Q3KM,
            Quant::Q3KS,
            Quant::Q2K,
        ];
        for w in order.windows(2) {
            assert!(w[0].fidelity_rank() < w[1].fidelity_rank(), "{:?} should outrank {:?}", w[0], w[1]);
        }
    }

    #[test]
    fn quant_parse_is_case_sensitive_and_closed() {
        assert_eq!(Quant::parse("Q8_0"), Some(Quant::Q8_0));
        assert_eq!(Quant::parse("q8_0"), None);
        assert_eq!(Quant::parse("Q9_0"), None);
        assert_eq!(Quant::parse("BF16"), None);
        assert_eq!(Quant::parse("F16"), None);
    }
}
