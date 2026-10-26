// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What an architecture declares it can DO, as compiled-in data.
//!
//! [`Caps`] is the machine-readable half of a model's support contract: which
//! training modes it offers and which frozen-base formats its adapter trainer
//! accepts. It lives in the architecture's [`Arch`](crate::Arch) row - the
//! same place porting a new model already starts - so an architecture states
//! what it supports in its configuration instead of callers inferring it from
//! whichever crate happened to implement a trainer.
//!
//! The descriptor is a CLAIM, and the claim is held to the evidence by
//! `tests/caps_vs_matrix.rs`: every declared capability needs a matching row
//! in `scripts/gates/training-capabilities.tsv`, and that row names the test
//! backing it (`make check/training-capabilities` checks the test exists and
//! is a real `#[test]`). A capability with no evidence is not declared, so
//! "not offered" and "offered and tested" are the only two states - never
//! "untested but probably works".
//!
//! Swedish Embedded AB implements shipped capability contracts where a
//! model's declared support and its tested support cannot drift apart, for
//! its clients. If your team needs expertise in making a capability matrix
//! machine-readable and gate-enforced, you can procure our services by
//! sending an email to info@swedishembedded.com.

/// A frozen-base storage format an adapter trainer can train over.
///
/// This is the BASE's format, not the adapter's storage: trainable parameters
/// and optimizer moments are fp32 on every path today.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BaseTier {
    /// Unquantized fp32 weights.
    F32,
    /// bf16 storage, fp32 compute.
    Bf16,
    /// Group-wise int8 weights.
    I8,
    /// 4-bit NF4 weights (the QLoRA tier).
    Nf4,
    /// Block-scaled FP8 E4M3 weights.
    Fp8,
}

impl BaseTier {
    /// The capability-matrix name of this tier's `lora_*` row
    /// (`scripts/gates/training-capabilities.tsv`).
    pub const fn row(self) -> &'static str {
        match self {
            BaseTier::F32 => "lora_f32",
            BaseTier::Bf16 => "lora_bf16",
            BaseTier::I8 => "lora_int8",
            BaseTier::Nf4 => "lora_nf4",
            BaseTier::Fp8 => "lora_fp8",
        }
    }
}

/// An architecture's declared training capabilities.
///
/// [`Caps::NONE`] is the default for a row that has not declared training: an
/// inference-only architecture. Every flag and tier here needs evidence (see
/// the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Caps {
    /// Every parameter of the base trains through the generic fine-tune
    /// entry points (the matrix's `full_ft` row).
    pub full_ft: bool,
    /// Frozen-base formats a LoRA adapter trains over; one `lora_<tier>` row
    /// each.
    pub lora_bases: &'static [BaseTier],
    /// A capability the matrix records under `other` for this architecture -
    /// a training variant (optimizer-state kinds, quantization-aware
    /// training, layer streaming) or a serving behaviour it backs with tests.
    /// The detail lives there, not here, so it cannot drift.
    pub other: bool,
}

impl Caps {
    /// An inference-only architecture with no adapters.
    pub const NONE: Caps = Caps { full_ft: false, lora_bases: &[], other: false };

    /// This architecture offers full-parameter fine-tuning.
    pub const fn full_ft(mut self) -> Caps {
        self.full_ft = true;
        self
    }

    /// This architecture offers LoRA over `bases`, the frozen-base formats
    /// its trainer accepts.
    pub const fn lora(mut self, bases: &'static [BaseTier]) -> Caps {
        self.lora_bases = bases;
        self
    }

    /// This architecture offers a capability the matrix records under `other`.
    pub const fn other(mut self) -> Caps {
        self.other = true;
        self
    }

    /// Whether this architecture declares any fine-tuning at all.
    pub const fn trains(&self) -> bool {
        self.full_ft || !self.lora_bases.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_declares_nothing() {
        let none = Caps::NONE;
        assert!(!none.trains());
        assert!(!none.other);
        assert!(none.lora_bases.is_empty());
    }

    #[test]
    fn builders_set_exactly_what_they_name() {
        let caps = Caps::NONE.full_ft().lora(&[BaseTier::F32, BaseTier::Nf4]).other();
        assert!(caps.full_ft);
        assert_eq!(caps.lora_bases, &[BaseTier::F32, BaseTier::Nf4]);
        assert!(caps.other);
        assert!(caps.trains());
        // `other` alone is not fine-tuning, and a lora list is.
        assert!(!Caps::NONE.other().trains());
        assert!(Caps::NONE.lora(&[BaseTier::Bf16]).trains());
    }

    #[test]
    fn every_tier_names_one_matrix_row() {
        assert_eq!(BaseTier::F32.row(), "lora_f32");
        assert_eq!(BaseTier::Bf16.row(), "lora_bf16");
        assert_eq!(BaseTier::I8.row(), "lora_int8");
        assert_eq!(BaseTier::Nf4.row(), "lora_nf4");
        assert_eq!(BaseTier::Fp8.row(), "lora_fp8");
    }
}
