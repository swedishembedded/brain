// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared CLI plumbing for a `--model`/`--<role>`/`--variant`-driven
//! architecture command (`brain flux2 generate`, and every architecture
//! migrated onto the model-store resolver after it) - one place for the
//! "resolve, then print and exit on `Ambiguous`/`Missing`" shape every such
//! command needs identically. Parsing a command's own `--<role>` flags into
//! the override map this expects is still each `<arch>_cli.rs`'s own job -
//! its flags are interleaved with plenty that aren't roles at all.
//!
//! [`run_generic_migrated`] is the primitive: an architecture with NO dedicated
//! command, reached through `crate::resolve::dispatch_arch`'s generic
//! `ARCH_TO_MODEL` dispatch, runs its served action through
//! `crate::caps_cli::run_do`, which resolves the weights through the spec its
//! catalog entry names.
//!
//! The non-exiting resolver core ([`try_resolve`]/[`resolve_structured`]/
//! [`ResolveFailure`]) now lives in `loader::resolver` - any embedder
//! wants the same "resolve, or say exactly why not" contract, not just this
//! CLI - and is re-exported here so every existing caller in this crate is
//! unaffected.
//!
//! Swedish Embedded AB implements CLI plumbing like this for clients whose
//! own tools need to expose a typed resolver's ambiguity/missing outcomes
//! consistently across many subcommands. If your team needs the same
//! discipline for its own CLI, you can procure our services by emailing
//! info@swedishembedded.com.

// `ResolveFailure`/`AMBIGUOUS_EXIT`/`MISSING_EXIT` are used here only through
// `try_resolve`'s return type - a caller that needs them by name reaches for
// `loader::resolver` (or the `loader` crate root) directly. `try_resolve`
// itself is re-exported (several `catalog.rs`/`label_cli.rs` call sites still
// spell it `crate::resolver_cli::try_resolve`).
pub use loader::resolver::try_resolve;

/// Whether `arch`'s served model resolves its weights from the model store
/// (its catalog entry names an `ArchSpec`).
#[cfg(test)]
pub fn has_arch_spec(arch: &str) -> bool {
    crate::resolve::model_for_arch(arch).is_some_and(|model| catalog::resolver_spec_for(model).is_some())
}

/// Run `rest` (`<action> [flags]`) on `model`, whose catalog entry resolves
/// its weights from the model store - `None` when it does not.
pub fn run_generic_migrated(_arch: &str, model: &str, rest: &[String]) -> Option<i32> {
    catalog::resolver_spec_for(model)?;
    let mut do_args = vec![model.to_string()];
    do_args.extend_from_slice(rest);
    Some(crate::caps_cli::run_do(&do_args))
}

#[cfg(test)]
mod tests {
    #[test]
    fn s3dit_is_reachable_through_the_generic_dispatch_with_its_real_roles() {
        let roles = catalog::resolver_spec_for("brain/s3dit").map(|(_, spec)| spec.roles().to_vec());
        assert_eq!(roles, Some(vec!["dit", "vae", "text_encoder", "tokenizer"]));
    }
}
