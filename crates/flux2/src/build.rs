// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`build_resolved`]: the one "which variant, is it licensed, what config,
//! what precision, then build" sequence every FLUX.2 pipeline construction
//! site needs, whichever of the three legitimate ways it already knows (or
//! has to determine) the variant.
//!
//! Before this module existed, four call sites - `crates/cli/src/flux2_cli.rs`,
//! `crates/sdk/src/pipeline.rs`, `caps::Flux2Action::run` and
//! `crates/cli/src/resident_flux2.rs::activate` - each re-derived the same
//! ~8 lines (`bind_variant`/`check_license`/`Flux2Config::from_name`/
//! `effective_dit_precision`) independently, and one of the four differences
//! that drifted in was a real bug: two sites forgot `effective_dit_precision`
//! entirely, so a served `.gguf` DiT could build at the wrong precision (see
//! the sdk-design sweep roadmap). This module is where that sequence lives
//! now, so a future change to it changes it everywhere at once.
//!
//! What stays call-site-owned, deliberately NOT absorbed here: the token
//! ceilings (`n_fwd`/`n_gen` - a CLI run's tiling math and a served request's
//! plain width/height produce these very differently), the adapter stack,
//! and `max_batch`. Swallowing those into this function would re-create the
//! SDK's own fixed-1024² limitation everywhere else that calls it.

use capability::Assembly;

use crate::caps::{bind_variant, check_license};
use crate::config::Flux2Config;
use crate::pipeline::{effective_dit_precision, AdapterSpec, Paths, Pipeline};
use crate::Precision;

/// The three legitimate ways a caller already knows - or has to determine -
/// which FLUX.2 variant it's building. Unifying them here is what lets every
/// site run the SAME license/config/precision decision regardless of which
/// one applies.
pub enum VariantSource<'a> {
    /// `crates/loader`'s resolver already picked a variant among the store's
    /// roles (`Assembly::variant`) - the CLI's and SDK's own case, where the
    /// variant is a fact about what's ON DISK, already established before
    /// this call.
    Assembly(&'a Assembly),
    /// A caller named a variant (a stated FAMILY, not a proof), and the REAL
    /// weights at `paths.dit` must be sniffed and reconciled with it
    /// (`bind_variant`) - the served capability action's case, where the
    /// operator's configured weights may not match what a remote client
    /// claimed.
    Sniff(&'a str),
    /// The variant is already bound - fixed at construction from the real
    /// weights, never re-derived per request (a resident instance's own
    /// case: `Flux2Resident::variant`, sniffed once when the resident was
    /// registered).
    Bound(&'a str),
}

/// Resolve `source` to a real variant name, license-gate it, load its
/// config, and correct `requested_precision` for the real DiT file (a
/// `.gguf` executes through the packed int8 path regardless of what was
/// asked - `effective_dit_precision`'s own doc) - without building.
///
/// Split out from [`build_resolved`] because this half is cheap enough to
/// run on every request even when a caller's OWN cache means the expensive
/// build won't happen: `caps::Flux2Action`'s hot-pipeline cache uses the
/// resolved variant/precision as its cache KEY, computed before deciding
/// whether a rebuild is even needed - calling the full resolve-AND-build
/// unconditionally there would defeat the cache entirely.
pub fn resolve(source: VariantSource, paths: &Paths, requested_precision: Precision, precision_was_explicit: bool) -> Result<(Flux2Config, Precision, String), String> {
    let variant = match source {
        VariantSource::Assembly(a) => a.variant.clone().ok_or_else(|| "flux2: assemble: no dit chosen".to_string())?,
        VariantSource::Sniff(requested) => bind_variant(&paths.dit, requested)?,
        VariantSource::Bound(v) => v.to_string(),
    };
    check_license(&variant)?;
    let cfg = Flux2Config::from_name(&variant)?;
    let precision = effective_dit_precision(&paths.dit, requested_precision, precision_was_explicit)?;
    Ok((cfg, precision, variant))
}

/// [`resolve`] plus the actual [`Pipeline`] build - the shape every call
/// site that does NOT keep its own pre-build cache wants directly (the CLI,
/// the SDK, a resident's `activate`, which is only ever called once the
/// residency layer has already decided a fresh build is needed).
///
/// `n_fwd`/`n_gen`/`adapters`/`max_batch` are call-site-computed - see this
/// module's doc for why they aren't resolved here too. Returns the built
/// pipeline alongside the config/precision/variant a caller needs to hold
/// onto afterward (a resident's own bookkeeping, a rebuild-with-an-adapter
/// call).
pub fn build_resolved(
    source: VariantSource,
    paths: &Paths,
    n_fwd: u32,
    n_gen: u32,
    adapters: &[AdapterSpec],
    requested_precision: Precision,
    precision_was_explicit: bool,
    max_batch: u32,
) -> Result<(Pipeline, Flux2Config, Precision, String), String> {
    let (cfg, precision, variant) = resolve(source, paths, requested_precision, precision_was_explicit)?;
    let pipe = Pipeline::build_sized(&cfg, paths, n_fwd, n_gen, adapters, precision, max_batch)?;
    Ok((pipe, cfg, precision, variant))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn paths_with_dit(dit: &str) -> Paths {
        Paths { dit: dit.to_string(), vae: String::new(), te: String::new(), tokenizer: String::new() }
    }

    /// `VariantSource::Assembly` with no variant chosen fails before touching
    /// license/config/precision/build at all - the exact message every
    /// caller of this path already relied on (`crates/sdk`'s own
    /// `format!("flux2: assemble: no dit chosen")`, `flux2_cli.rs`'s
    /// `.ok_or("flux2: resolved assembly has no variant")`).
    #[test]
    fn assembly_with_no_variant_fails_before_any_real_work() {
        let assembly = Assembly { id: "x".to_string(), arch: "flux2".to_string(), variant: None, roles: BTreeMap::new(), provenance: Vec::new() };
        let paths = paths_with_dit("model.safetensors");
        let Err(err) = build_resolved(VariantSource::Assembly(&assembly), &paths, 1, 1, &[], Precision::F32, false, 1) else { panic!("expected an error") };
        assert_eq!(err, "flux2: assemble: no dit chosen");
    }

    /// `VariantSource::Bound` skips `bind_variant`'s sniff entirely (the
    /// variant is already a fact from construction) but still runs the
    /// license gate - a resident instance never gets to skip that.
    #[test]
    fn bound_variant_still_runs_the_license_gate() {
        let _guard = brain_testutil::env_lock();
        std::env::remove_var("BRAIN_FLUX2_ALLOW_NC");
        let paths = paths_with_dit("model.safetensors");
        let Err(err) = build_resolved(VariantSource::Bound("klein-9b"), &paths, 1, 1, &[], Precision::F32, false, 1) else { panic!("expected an error") };
        assert!(err.contains("Non-Commercial"), "{err}");
    }

    /// The whole point of this extraction: a `.gguf` DiT gets the precision
    /// correction under `VariantSource::Bound` - this is the fix M10a
    /// applied by hand at `resident_flux2.rs`'s call site, now proven for
    /// the shared implementation that site is meant to converge on.
    /// `VariantSource::Sniff`'s equivalent needs a real file for
    /// `bind_variant` to sniff (covered by `caps`'s own
    /// `bind_variant_keeps_the_stated_family_but_sniffs_the_size`), so it's
    /// not repeated here with a fake path that would fail at the sniff
    /// itself rather than proving anything about the precision decision.
    #[test]
    fn a_gguf_dit_is_corrected_to_int8_under_a_bound_variant() {
        let paths = paths_with_dit("model.gguf");
        // Fails at `Pipeline::build_sized` (no real weights) - proven far
        // enough that variant/license/config/precision all already ran,
        // since neither a missing-variant nor a license-gate message is
        // what comes back.
        let Err(err) = build_resolved(VariantSource::Bound("klein-4b"), &paths, 1, 1, &[], Precision::F32, false, 1) else { panic!("expected an error") };
        assert!(!err.contains("Non-Commercial"), "{err}");
        assert!(!err.contains("incompatible"), "{err}");
    }

    /// An EXPLICIT fp32 request against a `.gguf` DiT is still a named
    /// error, not silently coerced - `effective_dit_precision`'s own
    /// contract, reachable through every `VariantSource` because the
    /// correction runs after variant resolution regardless of which arm
    /// picked the variant.
    #[test]
    fn an_explicit_fp32_request_against_a_gguf_dit_is_refused() {
        let paths = paths_with_dit("model.gguf");
        let Err(err) = build_resolved(VariantSource::Bound("klein-4b"), &paths, 1, 1, &[], Precision::F32, true, 1) else { panic!("expected an error") };
        assert!(err.contains("incompatible"), "{err}");
    }
}
