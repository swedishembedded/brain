// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The model-store resolver's non-exiting core -- moved out of
//! `crates/cli/src/resolver_cli.rs`. [`try_resolve`]/[`resolve_structured`]
//! resolve an architecture's roles against the model store and report every
//! non-`Resolved` outcome as a value, never by exiting the process, so any
//! embedder can call them directly.
//!
//! `crates/cli::resolver_cli` keeps everything that genuinely IS
//! CLI-specific on top of these: `resolve_or_exit` (calls
//! `std::process::exit` -- a process-lifetime decision that has no meaning
//! for an in-process embedder), `extract_role_overrides` (parses `--<role>`
//! flags out of `argv`), and the served/resident path (`served_assembly`,
//! which answers an ambiguity by suggesting an environment variable to set,
//! a daemon-operator idiom, not a library concern).

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::resolve::{describe_ambiguity, describe_missing, ArchSpec, Resolution};
use capability::Assembly;

/// Exit code for an unresolvable `Resolution::Ambiguous` - distinct from
/// [`MISSING_EXIT`] so a script can tell "name one more thing" apart from
/// "nothing here at all" without parsing the message.
pub const AMBIGUOUS_EXIT: i32 = 3;
/// Exit code for `Resolution::Missing`.
pub const MISSING_EXIT: i32 = 4;

/// [`try_resolve`]'s failure shapes, each carrying its own already-rendered
/// message - kept distinct (rather than one `String`) so a caller that must
/// exit the process (`crate::cli`'s `resolve_or_exit`, in `crates/cli`) can
/// still pick the right exit code, while a caller that cannot (an in-process
/// embedder) can collapse it to one message via [`ResolveFailure::message`].
pub enum ResolveFailure {
    /// No models directory is configured at all - not a resolver outcome,
    /// a precondition the resolver was never reached to evaluate.
    NoModelsDir(String),
    Ambiguous(String),
    Missing(String),
}

impl ResolveFailure {
    /// The rendered message, regardless of which shape this is.
    pub fn message(&self) -> &str {
        match self {
            ResolveFailure::NoModelsDir(m) | ResolveFailure::Ambiguous(m) | ResolveFailure::Missing(m) => m,
        }
    }
    /// The process exit code a CLI caller should use for this outcome.
    pub fn exit_code(&self) -> i32 {
        match self {
            ResolveFailure::NoModelsDir(_) => 1,
            ResolveFailure::Ambiguous(_) => AMBIGUOUS_EXIT,
            ResolveFailure::Missing(_) => MISSING_EXIT,
        }
    }
}

impl std::fmt::Display for ResolveFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

/// Resolve `arch`'s roles through the model-store resolver: scans
/// `models_dir` and reports every non-`Resolved` outcome as a
/// [`ResolveFailure`] instead of exiting - for a caller that cannot exit the
/// process on failure.
///
/// `models_dir` is the ALREADY-resolved store root - what
/// [`crate::model_dir::resolve`] answered for the caller's own `--models-dir`
/// flag (or its absence) - never looked up again here, so the one flag a
/// caller parsed decides the store for every resolver it reaches. `None`
/// means no store is configured at all ([`ResolveFailure::NoModelsDir`]).
pub fn try_resolve(models_dir: Option<&Path>, arch: &str, spec: &dyn ArchSpec, overrides: &BTreeMap<String, String>) -> Result<Assembly, ResolveFailure> {
    match resolve_structured(models_dir, arch, spec, overrides).map_err(ResolveFailure::NoModelsDir)? {
        Resolution::Resolved(assembly) => Ok(*assembly),
        Resolution::Ambiguous(a) => Err(ResolveFailure::Ambiguous(describe_ambiguity(&a))),
        Resolution::Missing(m) => Err(ResolveFailure::Missing(describe_missing(&m))),
    }
}

/// [`try_resolve`]'s models-directory lookup, inventory scan and resolve, with
/// the rendering left to the caller - for a caller that must report an
/// `Ambiguous`/`Missing` outcome in its OWN vocabulary rather than in
/// [`describe_ambiguity`]'s CLI-flag one.
///
/// `models_dir` is the already-resolved store root, exactly as for
/// [`try_resolve`]. `Err` is the one outcome that is not a resolution at
/// all: no models directory is configured, so the resolver was never
/// reached.
pub fn resolve_structured(models_dir: Option<&Path>, arch: &str, spec: &dyn ArchSpec, overrides: &BTreeMap<String, String>) -> Result<Resolution, String> {
    let root = models_dir.ok_or_else(|| format!("{arch}: no models directory (set --models-dir, BRAIN_MODELS_DIR, or $HOME)"))?;
    let records = brain_modelstore::inventory::scan(root);
    let specs: [&dyn ArchSpec; 1] = [spec];
    Ok(brain_modelstore::resolve::resolve(arch, &records, &specs, overrides))
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::ArtifactRecord;
    use brain_modelstore::resolve::{AssembleOutcome, AssembledVariant, Confidence};

    struct OneRoleSpec;
    impl ArchSpec for OneRoleSpec {
        fn arch(&self) -> &'static str {
            "loadertest"
        }
        fn roles(&self) -> &'static [&'static str] {
            &["weights"]
        }
        fn classify(&self, _records: &[ArtifactRecord], _root: &Path) -> Vec<(usize, String, Confidence)> {
            Vec::new()
        }
        fn assemble(&self, _chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
            Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/loadertest".to_string(), variant: None }))
        }
        fn validate(&self, _assembly: &Assembly) -> Result<(), String> {
            Ok(())
        }
    }

    /// No models directory configured is the one outcome that is not a
    /// `Resolution` at all - [`try_resolve`] must report it as
    /// [`ResolveFailure::NoModelsDir`], exit code 1, distinct from an
    /// [`ResolveFailure::Missing`] the resolver itself could have produced.
    #[test]
    fn try_resolve_reports_no_models_dir_when_none_is_configured() {
        let err = try_resolve(None, "loadertest", &OneRoleSpec, &BTreeMap::new()).expect_err("no models dir must fail");
        assert!(matches!(err, ResolveFailure::NoModelsDir(_)));
        assert_eq!(err.exit_code(), 1);
    }

    /// A configured but empty store resolves to `Missing` (nothing for
    /// `OneRoleSpec` to classify), which `try_resolve` must surface as
    /// [`ResolveFailure::Missing`] carrying [`MISSING_EXIT`].
    #[test]
    fn try_resolve_reports_missing_with_its_own_exit_code() {
        let root = std::env::temp_dir().join(format!("brain-loader-resolver-missing-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();

        let err = try_resolve(Some(&root), "loadertest", &OneRoleSpec, &BTreeMap::new()).expect_err("an empty store must not resolve");
        assert!(matches!(err, ResolveFailure::Missing(_)));
        assert_eq!(err.exit_code(), MISSING_EXIT);

        std::fs::remove_dir_all(&root).ok();
    }
}
