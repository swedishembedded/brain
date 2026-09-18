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
//! `crate::pipeline::ImagePipelineBuilder` (Phase 6.3) does not use this: it
//! resolves against TWO specs (flux2, s3dit) with its own tie-break, not
//! one, so the same three-line match would not simplify to a single call
//! here without a second generalization this milestone does not need yet.

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

/// Resolve `model_id` against `spec` (named `arch`, the same string every
/// caller already passes `resolve_structured` today), honoring
/// `download_policy`:
///
/// - `Offline`: `resolve_structured` only, ONCE - a `Missing` result returns
///   immediately as [`Error::Missing`], never a fetch attempt.
/// - `IfMissing` (the default every existing caller already had): the same
///   resolve-first order, and only on `Missing` does this fetch (skipped if
///   `Store::local` already recognizes the reference - the cheap
///   pre-existing check each caller had) and retry the resolve once.
/// - `AlwaysCheck`: fetches FIRST, unconditionally, then resolves - the same
///   "always go through plan/execute" reading
///   `loader::supply::ensure_default_weights_with` already established for
///   this variant, not a second one invented here.
pub(crate) fn resolve_with_policy(arch: &str, spec: &dyn ArchSpec, model_id: &str, overrides: &BTreeMap<String, String>, download_policy: loader::DownloadPolicy) -> Result<capability::Assembly> {
    let reference = brain_modelref::ModelRef::parse(model_id).map_err(|e| Error::ModelNotFound(format!("{model_id}: {e}")))?;

    if download_policy == loader::DownloadPolicy::AlwaysCheck {
        fetch(&reference, model_id)?;
    }

    match loader::resolve_structured(arch, spec, overrides).map_err(Error::Backend)? {
        Resolution::Resolved(a) => Ok(*a),
        Resolution::Ambiguous(a) => Err(Error::Ambiguous(a)),
        Resolution::Missing(m) => {
            if download_policy == loader::DownloadPolicy::Offline {
                return Err(Error::Missing(m));
            }
            let root = loader::model_dir::resolve(None).ok_or_else(|| Error::Backend("no models directory configured (set BRAIN_MODELS_DIR, or $HOME)".to_string()))?;
            let store = brain_modelstore::Store::new(root);
            if store.local(&reference).is_none() {
                fetch(&reference, model_id)?;
            }
            match loader::resolve_structured(arch, spec, overrides).map_err(Error::Backend)? {
                Resolution::Resolved(a) => Ok(*a),
                Resolution::Ambiguous(a) => Err(Error::Ambiguous(a)),
                Resolution::Missing(m) => Err(Error::Missing(m)),
            }
        }
    }
}
