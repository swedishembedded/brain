// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`resolve_with_policy`]: `loader::resolve_structured` plus
//! [`loader::DownloadPolicy`], shared by every pipeline builder that
//! resolves a `model_id` against ONE `brain_modelstore::resolve::ArchSpec`
//! this way (`depth`/`embedding`/`restore`/`ground`/`music`/`video`/`vlm`/
//! `text`/`detect`/`segment`/`upscale`/`asr`) - the resolve-first-then-
//! fetch-on-`Missing` sequence a real, previously confirmed bug required:
//! `Store::local` only recognizes a compound `brain.manifest.json` or a bare
//! `model.brain.safetensors`, so checking it BEFORE resolving forces an
//! unnecessary (and for several architectures, failing) network round trip
//! even when a real, already-downloaded checkpoint is fully local in its
//! native upstream shape. Factored out here rather than repeated a dozen
//! times with the policy branching newly folded into each copy, which is
//! exactly the kind of drift that let that original bug survive uncaught in
//! some copies long after others were fixed.
//!
//! [`resolve_two_with_policy`] is the same idea for a builder that resolves
//! against TWO specs with a tie-break (`tts`/`forecast`): try `a`, then `b`
//! only when `a` did not resolve, preferring whichever side found real
//! (if ambiguous) evidence over one that found none, same as before this
//! existed. `crate::pipeline::ImagePipelineBuilder` (Phase 6.3) still keeps
//! its own inline version rather than this one: it also reports build/
//! download PROGRESS through caller-supplied closures, a capability neither
//! of these two functions carries, so folding it in here would mean adding
//! an unused parameter to every OTHER caller just to satisfy the one that
//! needs it - not a simplification.

// Every surface that selects the `resolve` tier calls one of the two
// resolvers, but no surface calls both, so each item below compiles only
// under the surfaces whose modules call it: a surface built alone (the SDK
// feature sweep builds every one alone) must not carry the other's resolver
// as dead code. Adding a caller in a new module means adding its surface to
// that item's `cfg`; the sweep names the item if one is missed.

use std::collections::BTreeMap;

use brain_modelstore::resolve::{ArchSpec, Resolution};

use crate::{Error, Result};

fn fetch(reference: &brain_modelref::ModelRef, model_id: &str) -> Result<()> {
    let root = loader::model_dir::resolve(None).ok_or_else(|| Error::Backend("no models directory configured (set BRAIN_MODELS_DIR, or $HOME)".to_string()))?;
    let store = brain_modelstore::Store::new(root);
    let hub = brain_modelstore::HfHub::new();
    let plan = brain_modelstore::plan(reference, &store, &hub)?;
    loader::supply::execute_plan(&store, &hub, &plan, model_id, &mut |_name, _got, _total| {}).map_err(Error::Download)?;
    Ok(())
}

/// Resolve `model_id` against `spec` (named `arch`), over what `model_id`'s
/// own repo directory holds (`loader::resolver::resolve_reference`: another
/// repo of the same architecture is never a candidate), honoring
/// `download_policy`:
///
/// - `Offline`: one resolve only - a `Missing` result returns
///   immediately as [`Error::Missing`], never a fetch attempt.
/// - `IfMissing` (the default every existing caller already had): the same
///   resolve-first order, and only on `Missing` does this fetch (skipped if
///   `Store::local` already recognizes the reference - the cheap
///   pre-existing check each caller had) and retry the resolve once.
/// - `AlwaysCheck`: fetches FIRST, unconditionally, then resolves - the same
///   "always go through plan/execute" reading
///   `loader::supply::ensure_default_weights_with` already established for
///   this variant, not a second one invented here.
#[cfg(any(feature = "audio", feature = "image", feature = "multimodal", feature = "text", feature = "video", feature = "vision"))]
pub(crate) fn resolve_with_policy(arch: &str, spec: &dyn ArchSpec, model_id: &str, overrides: &BTreeMap<String, String>, download_policy: loader::DownloadPolicy) -> Result<capability::Assembly> {
    let reference = brain_modelref::ModelRef::parse(model_id).map_err(|e| Error::ModelNotFound(format!("{model_id}: {e}")))?;

    if download_policy == loader::DownloadPolicy::AlwaysCheck {
        fetch(&reference, model_id)?;
    }

    // The SDK takes no models-directory argument here, so the store is the
    // resolver's documented default (`BRAIN_MODELS_DIR`, then XDG/HOME).
    let models_dir = loader::model_dir::resolve(None);
    match loader::resolver::resolve_reference(models_dir.as_deref(), arch, spec, &reference, overrides).map_err(Error::Backend)? {
        Resolution::Resolved(a) => Ok(*a),
        Resolution::Ambiguous(a) => Err(Error::Ambiguous(a)),
        Resolution::Missing(m) => {
            if download_policy == loader::DownloadPolicy::Offline {
                return Err(Error::Missing(m));
            }
            let root = models_dir.clone().ok_or_else(|| Error::Backend("no models directory configured (set BRAIN_MODELS_DIR, or $HOME)".to_string()))?;
            let store = brain_modelstore::Store::new(root);
            if store.local(&reference).is_none() {
                fetch(&reference, model_id)?;
            }
            match loader::resolver::resolve_reference(models_dir.as_deref(), arch, spec, &reference, overrides).map_err(Error::Backend)? {
                Resolution::Resolved(a) => Ok(*a),
                Resolution::Ambiguous(a) => Err(Error::Ambiguous(a)),
                Resolution::Missing(m) => Err(Error::Missing(m)),
            }
        }
    }
}

/// Which of the two architectures [`resolve_two_with_policy`] resolved -
/// which side won is the caller's own dispatch, not this function's; this
/// only carries the resolved [`capability::Assembly`] back with that
/// decision attached.
#[cfg(any(feature = "audio", feature = "forecast", feature = "image"))]
pub(crate) enum Resolved2 {
    A(capability::Assembly),
    B(capability::Assembly),
}

/// `a_arch`/`a_spec` tried first, `b_arch`/`b_spec` only when `a` did not
/// resolve - both `Resolved` cases win outright; a store that resolves
/// NEITHER reports whichever side found real (if ambiguous) evidence over
/// one that found none, and falls back to `a`'s own outcome when both are
/// plain `Missing` (an arbitrary tie-break, not a claim that `a` is the more
/// likely answer).
///
/// `pub(crate)` (not just this module's own [`resolve_two_with_policy`]):
/// `crate::pipeline::resolve_arch` calls this directly too, since flux2/s3dit
/// dispatch is the exact same two-way tie-break with no policy/fetch wrapper
/// around it (`ImagePipelineBuilder` keeps its own progress-reporting fetch
/// path, per this module's own doc - only the tie-break itself was ever
/// duplicated).
#[cfg(any(feature = "audio", feature = "forecast", feature = "image"))]
pub(crate) fn try_two(a_arch: &str, a_spec: &dyn ArchSpec, b_arch: &str, b_spec: &dyn ArchSpec, overrides: &BTreeMap<String, String>) -> Result<Resolved2> {
    // Same default store as `resolve_with_policy`: no models-directory argument here.
    let models_dir = loader::model_dir::resolve(None);
    let a_outcome = loader::resolve_structured(models_dir.as_deref(), a_arch, a_spec, overrides).map_err(Error::Backend)?;
    if matches!(a_outcome, Resolution::Resolved(_)) {
        let Resolution::Resolved(a) = a_outcome else { unreachable!("just matched") };
        return Ok(Resolved2::A(*a));
    }

    let b_outcome = loader::resolve_structured(models_dir.as_deref(), b_arch, b_spec, overrides).map_err(Error::Backend)?;
    if matches!(b_outcome, Resolution::Resolved(_)) {
        let Resolution::Resolved(b) = b_outcome else { unreachable!("just matched") };
        return Ok(Resolved2::B(*b));
    }

    // Both `Resolved` cases already returned above; only `Ambiguous`/
    // `Missing` combinations can reach here.
    match (a_outcome, b_outcome) {
        (Resolution::Ambiguous(a), _) => Err(Error::Ambiguous(a)),
        (_, Resolution::Ambiguous(b)) => Err(Error::Ambiguous(b)),
        (a, _) => match a {
            Resolution::Missing(m) => Err(Error::Missing(m)),
            Resolution::Resolved(_) => unreachable!("Resolved handled above"),
            Resolution::Ambiguous(_) => unreachable!("Ambiguous handled above"),
        },
    }
}

/// [`resolve_with_policy`], but against TWO architectures with [`try_two`]'s
/// own tie-break instead of one `ArchSpec` - see this module's own doc for
/// why `tts`/`forecast` need this shape and `ImagePipelineBuilder` does not
/// use it.
#[cfg(any(feature = "audio", feature = "forecast"))]
pub(crate) fn resolve_two_with_policy(
    a_arch: &str,
    a_spec: &dyn ArchSpec,
    b_arch: &str,
    b_spec: &dyn ArchSpec,
    model_id: &str,
    overrides: &BTreeMap<String, String>,
    download_policy: loader::DownloadPolicy,
) -> Result<Resolved2> {
    let reference = brain_modelref::ModelRef::parse(model_id).map_err(|e| Error::ModelNotFound(format!("{model_id}: {e}")))?;

    if download_policy == loader::DownloadPolicy::AlwaysCheck {
        fetch(&reference, model_id)?;
    }

    match try_two(a_arch, a_spec, b_arch, b_spec, overrides) {
        Err(Error::Missing(m)) => {
            if download_policy == loader::DownloadPolicy::Offline {
                return Err(Error::Missing(m));
            }
            let root = loader::model_dir::resolve(None).ok_or_else(|| Error::Backend("no models directory configured (set BRAIN_MODELS_DIR, or $HOME)".to_string()))?;
            let store = brain_modelstore::Store::new(root);
            if store.local(&reference).is_none() {
                fetch(&reference, model_id)?;
            }
            try_two(a_arch, a_spec, b_arch, b_spec, overrides)
        }
        other => other,
    }
}

/// [`resolve_with_policy`] against TWO architectures, directed by `model_id`
/// itself: each side resolves over what `model_id`'s own repo directory holds,
/// so the model asked for decides the architecture. [`try_two`] and
/// [`resolve_two_with_policy`] cannot do this: they resolve over the whole
/// store, so an architecture tried first wins whenever any of its checkpoints
/// is present, whichever one was asked for. That is right for a task with one
/// canonical checkpoint per architecture and wrong where several
/// architectures hold different models of one task.
///
/// `a` is tried first, then `b`. Only when both are `Missing` is the model
/// fetched (under `download_policy`, as in [`resolve_with_policy`]) and both
/// tried again. A side that finds the directory but fails (ambiguous roles,
/// a read error) ends the search with that error rather than falling through
/// to the other side.
#[cfg(feature = "audio")]
pub(crate) fn resolve_either_for_reference(
    a_arch: &str,
    a_spec: &dyn ArchSpec,
    b_arch: &str,
    b_spec: &dyn ArchSpec,
    model_id: &str,
    overrides: &BTreeMap<String, String>,
    download_policy: loader::DownloadPolicy,
) -> Result<Resolved2> {
    let reference = brain_modelref::ModelRef::parse(model_id).map_err(|e| Error::ModelNotFound(format!("{model_id}: {e}")))?;
    let models_dir = loader::model_dir::resolve(None);

    // Resolved, or the first side's `Missing` report when neither side holds it.
    let try_both = || -> Result<std::result::Result<Resolved2, Box<brain_modelstore::resolve::Missing>>> {
        let a = loader::resolver::resolve_reference(models_dir.as_deref(), a_arch, a_spec, &reference, overrides).map_err(Error::Backend)?;
        let a_missing = match a {
            Resolution::Resolved(assembly) => return Ok(Ok(Resolved2::A(*assembly))),
            Resolution::Ambiguous(found) => return Err(Error::Ambiguous(found)),
            Resolution::Missing(missing) => missing,
        };
        match loader::resolver::resolve_reference(models_dir.as_deref(), b_arch, b_spec, &reference, overrides).map_err(Error::Backend)? {
            Resolution::Resolved(assembly) => Ok(Ok(Resolved2::B(*assembly))),
            Resolution::Ambiguous(found) => Err(Error::Ambiguous(found)),
            Resolution::Missing(_) => Ok(Err(a_missing)),
        }
    };

    if download_policy == loader::DownloadPolicy::AlwaysCheck {
        fetch(&reference, model_id)?;
    }
    let missing = match try_both()? {
        Ok(resolved) => return Ok(resolved),
        Err(missing) => missing,
    };
    if download_policy == loader::DownloadPolicy::Offline {
        return Err(Error::Missing(missing));
    }
    let root = models_dir.clone().ok_or_else(|| Error::Backend("no models directory configured (set BRAIN_MODELS_DIR, or $HOME)".to_string()))?;
    if brain_modelstore::Store::new(root).local(&reference).is_none() {
        fetch(&reference, model_id)?;
    }
    try_both()?.map_err(Error::Missing)
}
