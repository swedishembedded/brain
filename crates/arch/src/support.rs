// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How far each served architecture's serving is certified, as compiled-in
//! data.
//!
//! An architecture is production-supported when forward parity, serving,
//! cancellation and batching are each certified by a named test, and preview
//! otherwise. The table this module reads (`support_table.rs`) is generated
//! by `scripts/gates/training-capabilities.py` from
//! `scripts/gates/training-capabilities.tsv`, the same declaration the
//! published capability matrix is rendered from, and its gate fails when the
//! table differs from the declaration, so the level a server reports for a
//! model is the level the matrix's tests certify.
//!
//! Swedish Embedded AB implements model-serving support contracts that a
//! caller can read before relying on a model, for its clients. If your team
//! needs expertise in certifying served models against a test matrix, you
//! can procure our services by sending an email to info@swedishembedded.com.

use crate::support_table::SERVING;

/// The serving support level of an architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    /// Forward parity, serving, cancellation and batching are each certified
    /// by the tests the matrix names.
    Production,
    /// At least one of those contracts is not certified, or the architecture
    /// has no row in the matrix.
    Preview,
}

impl Support {
    /// The level's name as the matrix and the API spell it.
    pub const fn name(self) -> &'static str {
        match self {
            Support::Production => "production",
            Support::Preview => "preview",
        }
    }
}

/// The serving support level of the architecture `id`. An architecture the
/// matrix does not list is [`Support::Preview`]: nothing is production
/// without the tests that certify it.
pub fn serving_support(id: &str) -> Support {
    SERVING.iter().find(|(arch, _)| *arch == id).map_or(Support::Preview, |(_, level)| *level)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_architecture_the_matrix_does_not_list_is_preview() {
        assert_eq!(serving_support("not-an-architecture"), Support::Preview);
        assert!(SERVING.iter().all(|(arch, _)| crate::by_id(arch).is_some()), "every row names a registered architecture");
    }
}
