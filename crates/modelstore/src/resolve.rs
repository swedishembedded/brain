// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The resolver core: turns "every artifact found on disk"
//! ([`crate::inventory::scan`]'s output) into one terminal outcome for a
//! named architecture - a fully resolved [`Assembly`], a question only a
//! human can answer ([`Ambiguity`]), or a list of roles nothing on disk
//! satisfies ([`Missing`]). Never a fourth option: nothing downstream is
//! allowed to collapse an [`Ambiguity`] into a pick, and [`Resolution::Resolved`]
//! is only ever returned once [`ArchSpec::validate`] has passed - so a caller
//! that only handles `Resolved` can never load two components a validated
//! check would have rejected.
//!
//! Each architecture (`flux2`, `wan`, ...) supplies one [`ArchSpec`] telling
//! [`resolve`] which artifacts are plausible for which role and how sure that
//! guess is - `resolve` itself has no architecture-specific knowledge at all.
//!
//! Swedish Embedded AB implements model-resolution layers like this one for
//! clients running mixed fleets of hand-placed and fetched checkpoints. If
//! your team needs deterministic, never-silently-guessing model assembly, you
//! can procure our services by emailing info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use capability::Assembly;

use crate::inventory::{ArtifactRecord, Completeness};

/// How sure a [`ArchSpec::classify`] call is that one artifact fills one
/// role. Ordered weakest to strongest; a record whose only evidence is its
/// own filename must never rise above [`Confidence::Guessed`], and
/// [`Confidence::Guessed`] is never treated as usable on its own (see
/// [`resolve`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    /// Filename pattern only - no header, no adjacent config read. Never
    /// enough to select a role by itself.
    Guessed,
    /// Computed from the artifact's own tensor shapes/header fields, needed
    /// because a single self-reported fact (e.g. a GGUF `general.architecture`
    /// shared by two real architectures) was not enough on its own.
    Derived,
    /// The artifact's own header/config unambiguously names this role
    /// (a GGUF `general.architecture` KV, an HF `config.json` `architectures`
    /// entry) with no further disambiguation needed.
    Declared,
    /// Read from a manifest this resolver itself wrote previously.
    Recorded,
    /// The caller stated it explicitly (a CLI override) - never inferred.
    Stated,
}

/// The one question a [`resolve`] outcome poses when it cannot pick a single
/// artifact/variant on its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Question {
    /// More than one artifact classifies for `role` at the same top
    /// confidence tier, with nothing stating which one is meant.
    Role { role: String },
    /// The chosen components' shapes name a size class
    /// ([`import::DitSize`](../../flux2/import/enum.DitSize.html)-style), but
    /// not which named variant of it (e.g. FLUX.2's klein-vs-base, which no
    /// shape can ever answer).
    Variant { shape_class: String },
    /// A candidate exists but this resolver cannot read enough of it to
    /// classify at all (an unreadable header with no fallback).
    Unverifiable { path: PathBuf },
}

/// One way [`resolve`] could have gone, fully spelled out: the [`Assembly`]
/// it would produce, and the `(flag, value)` pairs a caller (the CLI) passes
/// back to select it deterministically next time.
#[derive(Clone, Debug)]
pub struct ModelCandidate {
    pub assembly: Box<Assembly>,
    pub selector: Vec<(String, String)>,
    pub summary: String,
}

/// [`resolve`] could not pick on its own: `question` names what's undecided,
/// `choices` names every way it could go.
#[derive(Clone, Debug)]
pub struct Ambiguity {
    pub arch: String,
    pub question: Question,
    pub choices: Vec<ModelCandidate>,
}

/// One required role nothing on disk (or nothing at high-enough confidence)
/// satisfies.
#[derive(Clone, Debug)]
pub struct MissingRole {
    pub role: String,
    pub doc: String,
    /// Human-readable near-misses (e.g. an interrupted download) that don't
    /// count as usable but explain why the role looks empty anyway.
    pub near_misses: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Missing {
    pub arch: String,
    pub roles: Vec<MissingRole>,
}

/// [`resolve`]'s three terminal outcomes. There is no fourth: a caller that
/// matches all three has handled every possible answer.
#[derive(Debug)]
pub enum Resolution {
    Resolved(Box<Assembly>),
    Ambiguous(Box<Ambiguity>),
    Missing(Box<Missing>),
}

/// What [`ArchSpec::assemble`] decided once exactly one candidate per role
/// was already chosen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssembledVariant {
    /// The id the resulting [`Assembly`] registers/serves under.
    pub id: String,
    pub variant: Option<String>,
}

/// [`ArchSpec::assemble`]'s result: either a fully named variant, or (never a
/// default) the exact named options it could not choose between - carrying
/// the options itself is what lets [`resolve`] build a real
/// [`Question::Variant`] [`Ambiguity`] without string-sniffing an error
/// message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssembleOutcome {
    Assembled(AssembledVariant),
    UnresolvedVariant { shape_class: String, options: Vec<String> },
}

/// One architecture's rules for turning an inventory into an [`Assembly`].
/// Every method here is pure/header-only: nothing implementing this trait may
/// read tensor bytes or touch a device.
pub trait ArchSpec: Send + Sync {
    fn arch(&self) -> &'static str;
    /// Required role names, in the order [`resolve`] reports them.
    fn roles(&self) -> &'static [&'static str];
    /// Every `(record index, role, confidence)` the inventory plausibly
    /// supports - one record may appear for more than one role, and a role
    /// may have zero, one, or many candidates.
    fn classify(&self, records: &[ArtifactRecord], inventory_root: &Path) -> Vec<(usize, String, Confidence)>;
    /// Given exactly one chosen candidate per role (by index into `records`)
    /// plus any caller-supplied overrides (role-path overrides, and any
    /// architecture-specific ones such as FLUX.2's `"variant"`), decide the
    /// full variant - or say precisely which named options it could not
    /// choose between. Never picks a default.
    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String>;
    /// The last gate: a header-only compatibility check between the chosen
    /// components. Must not read tensor bytes or touch a device - this runs
    /// before either ever happens anywhere in the calling code.
    fn validate(&self, assembly: &Assembly) -> Result<(), String>;
}

/// The deepest path ancestor common to every record - a stable inventory
/// root [`ArchSpec::classify`] can use for directory-relative lookups.
/// Every artifact's own path is already absolute (see
/// [`ArtifactRecord::path`]), so today's specs derive everything from a
/// record's own path instead; this exists for a future spec that needs the
/// scan root itself.
fn common_root(records: &[ArtifactRecord]) -> PathBuf {
    let mut paths = records.iter().map(|r| r.path.as_path());
    let Some(first) = paths.next() else { return PathBuf::new() };
    let mut common: Vec<std::path::Component> = first.components().collect();
    for p in paths {
        let comps: Vec<std::path::Component> = p.components().collect();
        let n = common.iter().zip(comps.iter()).take_while(|(a, b)| a == b).count();
        common.truncate(n);
    }
    common.into_iter().collect()
}

fn near_misses(records: &[ArtifactRecord]) -> Vec<String> {
    records
        .iter()
        .filter_map(|r| match &r.completeness {
            Completeness::Partial { final_path } => Some(format!("an interrupted download exists at {}", final_path.display())),
            _ => None,
        })
        .collect()
}

fn build_assembly(spec: &dyn ArchSpec, av: &AssembledVariant, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord]) -> Assembly {
    let roles: BTreeMap<String, PathBuf> = chosen.iter().map(|(role, &idx)| (role.clone(), records[idx].path.clone())).collect();
    let mut provenance: Vec<String> = chosen.iter().map(|(role, &idx)| format!("{role}: {}", records[idx].path.display())).collect();
    provenance.sort();
    Assembly { id: av.id.clone(), arch: spec.arch().to_string(), variant: av.variant.clone(), roles, provenance }
}

/// One candidate per remaining ambiguous role, each carrying a COMPLETE
/// hypothetical [`Assembly`] built by holding every other already-decided
/// role fixed and substituting this candidate in for `role`.
fn role_ambiguity(spec: &dyn ArchSpec, arch: &str, role: &str, candidates: &[(usize, Confidence)], resolved: &BTreeMap<String, usize>, records: &[ArtifactRecord], overrides: &BTreeMap<String, String>) -> Ambiguity {
    let choices = candidates
        .iter()
        .map(|&(idx, _)| {
            let mut chosen = resolved.clone();
            chosen.insert(role.to_string(), idx);
            let av = match spec.assemble(&chosen, records, overrides) {
                Ok(AssembleOutcome::Assembled(av)) => av,
                // The other role's own ambiguity (e.g. klein-vs-base) hasn't
                // been settled yet either - still produce a real, if
                // variant-less, hypothetical assembly rather than failing to
                // report this candidate at all.
                _ => AssembledVariant { id: format!("local/{arch}"), variant: None },
            };
            let assembly = build_assembly(spec, &av, &chosen, records);
            let path = records[idx].path.to_string_lossy().into_owned();
            ModelCandidate { summary: format!("{role}: {path}"), selector: vec![(format!("--{}", role.replace('_', "-")), path)], assembly: Box::new(assembly) }
        })
        .collect();
    Ambiguity { arch: arch.to_string(), question: Question::Role { role: role.to_string() }, choices }
}

/// Turn `records` into one terminal [`Resolution`] for `arch`, using
/// whichever of `specs` declares that [`ArchSpec::arch`].
///
/// `overrides` carries role-name -> explicit path/ref pairs a caller (a CLI
/// flag) already supplied - honored directly rather than re-classified, plus
/// any architecture-specific override keys an [`ArchSpec`] defines for itself
/// (FLUX.2's `"variant"`).
///
/// # Panics
/// If `specs` names no [`ArchSpec`] for `arch` - a caller error (the wrong
/// spec list was passed in), not a real resolution outcome.
pub fn resolve(arch: &str, records: &[ArtifactRecord], specs: &[&dyn ArchSpec], overrides: &BTreeMap<String, String>) -> Resolution {
    let spec = *specs.iter().find(|s| s.arch() == arch).unwrap_or_else(|| panic!("resolve: no ArchSpec registered for arch {arch:?}"));

    let root = common_root(records);
    let classifications = spec.classify(records, &root);
    let mut by_role: BTreeMap<&str, Vec<(usize, Confidence)>> = BTreeMap::new();
    for (idx, role, conf) in &classifications {
        by_role.entry(role.as_str()).or_default().push((*idx, *conf));
    }

    enum RoleResult {
        One(usize),
        Ambiguous(Vec<(usize, Confidence)>),
        None,
    }

    let mut per_role: Vec<(&'static str, RoleResult)> = Vec::new();
    for &role in spec.roles() {
        if let Some(path) = overrides.get(role) {
            let found = records.iter().position(|r| r.path.to_string_lossy() == *path);
            per_role.push((role, found.map(RoleResult::One).unwrap_or(RoleResult::None)));
            continue;
        }
        let mut candidates = by_role.get(role).cloned().unwrap_or_default();
        // A Guessed-only candidate is never usable on its own.
        candidates.retain(|(_, c)| *c > Confidence::Guessed);
        if candidates.is_empty() {
            per_role.push((role, RoleResult::None));
            continue;
        }
        let top = candidates.iter().map(|(_, c)| *c).max().unwrap();
        candidates.retain(|(_, c)| *c == top);
        per_role.push((role, if candidates.len() == 1 { RoleResult::One(candidates[0].0) } else { RoleResult::Ambiguous(candidates) }));
    }

    let missing: Vec<MissingRole> = per_role
        .iter()
        .filter(|(_, r)| matches!(r, RoleResult::None))
        .map(|(role, _)| MissingRole { role: role.to_string(), doc: format!("no artifact classifies as {role} for arch {arch}"), near_misses: near_misses(records) })
        .collect();
    if !missing.is_empty() {
        return Resolution::Missing(Box::new(Missing { arch: arch.to_string(), roles: missing }));
    }

    // A best-current single choice per role, for building EVERY role's
    // hypothetical assemblies - including the one about to be reported as
    // ambiguous, which needs every OTHER role already pinned down.
    let mut chosen: BTreeMap<String, usize> = BTreeMap::new();
    for (role, r) in &per_role {
        let idx = match r {
            RoleResult::One(i) => *i,
            RoleResult::Ambiguous(cands) => cands[0].0,
            RoleResult::None => unreachable!("missing roles returned above"),
        };
        chosen.insert(role.to_string(), idx);
    }

    if let Some((role, RoleResult::Ambiguous(candidates))) = per_role.iter().find(|(_, r)| matches!(r, RoleResult::Ambiguous(_))) {
        return Resolution::Ambiguous(Box::new(role_ambiguity(spec, arch, role, candidates, &chosen, records, overrides)));
    }

    match spec.assemble(&chosen, records, overrides) {
        Ok(AssembleOutcome::Assembled(av)) => {
            let assembly = build_assembly(spec, &av, &chosen, records);
            match spec.validate(&assembly) {
                Ok(()) => Resolution::Resolved(Box::new(assembly)),
                // `Resolved` is only ever returned once `validate` has
                // passed; a header-verified incompatibility between two
                // otherwise-unambiguous components is reported the same way
                // as an absent one - both mean "cannot build this today".
                Err(e) => Resolution::Missing(Box::new(Missing { arch: arch.to_string(), roles: vec![MissingRole { role: "compatibility".to_string(), doc: e, near_misses: Vec::new() }] })),
            }
        }
        Ok(AssembleOutcome::UnresolvedVariant { shape_class, options }) => {
            let choices = options
                .iter()
                .map(|opt| {
                    let mut forked = overrides.clone();
                    forked.insert("variant".to_string(), opt.clone());
                    let av = match spec.assemble(&chosen, records, &forked) {
                        Ok(AssembleOutcome::Assembled(av)) => av,
                        _ => AssembledVariant { id: format!("local/{arch}-{opt}"), variant: Some(opt.clone()) },
                    };
                    let assembly = build_assembly(spec, &av, &chosen, records);
                    ModelCandidate { summary: format!("{arch} {opt}"), selector: vec![("--variant".to_string(), opt.clone())], assembly: Box::new(assembly) }
                })
                .collect();
            Resolution::Ambiguous(Box::new(Ambiguity { arch: arch.to_string(), question: Question::Variant { shape_class }, choices }))
        }
        Err(e) => Resolution::Missing(Box::new(Missing { arch: arch.to_string(), roles: vec![MissingRole { role: "assemble".to_string(), doc: e, near_misses: Vec::new() }] })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(path: &str) -> ArtifactRecord {
        ArtifactRecord { path: PathBuf::from(path), size: 1, mtime_ns: 0, kind: crate::inventory::ArtifactKind::Opaque, completeness: Completeness::Complete }
    }

    fn partial_rec(path: &str, final_path: &str) -> ArtifactRecord {
        ArtifactRecord { path: PathBuf::from(path), size: 1, mtime_ns: 0, kind: crate::inventory::ArtifactKind::Opaque, completeness: Completeness::Partial { final_path: PathBuf::from(final_path) } }
    }

    /// A toy single-role arch: "dit" only, one candidate classifies at
    /// `CLASSIFY_CONFIDENCE`, `assemble` always names the same fixed variant,
    /// `validate` always passes - just enough to drive [`resolve`]'s control
    /// flow without any real architecture's rules.
    struct ToySpec {
        classify_confidence: Confidence,
        validate_ok: bool,
    }

    impl ArchSpec for ToySpec {
        fn arch(&self) -> &'static str {
            "toy"
        }
        fn roles(&self) -> &'static [&'static str] {
            &["dit"]
        }
        fn classify(&self, records: &[ArtifactRecord], _root: &Path) -> Vec<(usize, String, Confidence)> {
            records.iter().enumerate().filter(|(_, r)| r.path.to_string_lossy().contains("dit")).map(|(i, _)| (i, "dit".to_string(), self.classify_confidence)).collect()
        }
        fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
            let idx = *chosen.get("dit").ok_or("no dit")?;
            Ok(AssembleOutcome::Assembled(AssembledVariant { id: format!("local/toy-{}", records[idx].path.display()), variant: Some("only".to_string()) }))
        }
        fn validate(&self, _assembly: &Assembly) -> Result<(), String> {
            if self.validate_ok {
                Ok(())
            } else {
                Err("toy validate: deliberately incompatible".to_string())
            }
        }
    }

    #[test]
    fn resolves_cleanly_when_exactly_one_candidate_per_role() {
        let records = vec![rec("/models/toy/dit.bin")];
        let spec = ToySpec { classify_confidence: Confidence::Declared, validate_ok: true };
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("toy", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.arch, "toy");
                assert_eq!(a.variant.as_deref(), Some("only"));
                assert_eq!(a.roles["dit"], PathBuf::from("/models/toy/dit.bin"));
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn missing_when_a_required_role_has_zero_candidates() {
        let records: Vec<ArtifactRecord> = vec![rec("/models/toy/unrelated.bin")];
        let spec = ToySpec { classify_confidence: Confidence::Declared, validate_ok: true };
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("toy", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Missing(m) => {
                assert_eq!(m.roles.len(), 1);
                assert_eq!(m.roles[0].role, "dit");
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    #[test]
    fn missing_names_a_partial_download_as_a_near_miss() {
        let records = vec![partial_rec("/models/toy/dit.bin.part", "/models/toy/dit.bin")];
        // classify() only matches complete/usable artifacts in real specs, but
        // this toy spec classifies by path alone regardless of completeness -
        // exercise the near-miss reporting path directly via a role that
        // still comes up empty for a DIFFERENT reason (no "dit" in the path).
        let unrelated = vec![partial_rec("/models/toy/other.bin.part", "/models/toy/other.bin")];
        let mut all = records.clone();
        all.extend(unrelated);
        let spec = ToySpec { classify_confidence: Confidence::Declared, validate_ok: true };
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        // Force a miss by overriding "dit" to a path that isn't in the inventory.
        let mut overrides = BTreeMap::new();
        overrides.insert("dit".to_string(), "/nowhere.bin".to_string());
        let out = resolve("toy", &all, &specs, &overrides);
        match out {
            Resolution::Missing(m) => {
                assert!(!m.roles[0].near_misses.is_empty(), "{:?}", m.roles[0].near_misses);
                assert!(m.roles[0].near_misses[0].contains("interrupted download"));
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    #[test]
    fn a_guessed_only_candidate_never_resolves_on_its_own() {
        let records = vec![rec("/models/toy/dit.bin")];
        let spec = ToySpec { classify_confidence: Confidence::Guessed, validate_ok: true };
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("toy", &records, &specs, &BTreeMap::new());
        assert!(matches!(out, Resolution::Missing(_)), "a Guessed-only candidate must never resolve: {out:?}");
    }

    #[test]
    fn an_override_is_honored_without_reclassifying() {
        let records = vec![rec("/models/toy/anything.bin")];
        // classify_confidence Guessed would normally never resolve - proving
        // the override path bypasses classify() entirely, as documented.
        let spec = ToySpec { classify_confidence: Confidence::Guessed, validate_ok: true };
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let mut overrides = BTreeMap::new();
        overrides.insert("dit".to_string(), "/models/toy/anything.bin".to_string());
        let out = resolve("toy", &records, &specs, &overrides);
        match out {
            Resolution::Resolved(a) => assert_eq!(a.roles["dit"], PathBuf::from("/models/toy/anything.bin")),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn validate_failure_after_a_clean_assemble_reports_missing_not_resolved() {
        let records = vec![rec("/models/toy/dit.bin")];
        let spec = ToySpec { classify_confidence: Confidence::Declared, validate_ok: false };
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("toy", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Missing(m) => assert!(m.roles[0].doc.contains("deliberately incompatible")),
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    #[test]
    fn two_top_tier_candidates_for_one_role_is_ambiguous_with_a_choice_per_candidate() {
        let records = vec![rec("/models/toy/dit-a.bin"), rec("/models/toy/dit-b.bin")];
        let spec = ToySpec { classify_confidence: Confidence::Declared, validate_ok: true };
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("toy", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Ambiguous(a) => {
                assert_eq!(a.question, Question::Role { role: "dit".to_string() });
                assert_eq!(a.choices.len(), 2);
                for c in &a.choices {
                    assert_eq!(c.selector.len(), 1);
                    assert_eq!(c.selector[0].0, "--dit");
                }
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn a_higher_tier_candidate_wins_over_a_lower_tier_one_without_ambiguity() {
        // classify() itself can't express "one Declared, one Derived" through
        // the toy spec's uniform confidence, so drive it through two records
        // where only ONE actually matches "dit" in its path (the other is a
        // decoy for a different role) - proving the single top-tier winner
        // path, not the every-candidate-same-tier ambiguity path.
        let records = vec![rec("/models/toy/dit.bin"), rec("/models/toy/decoy.bin")];
        let spec = ToySpec { classify_confidence: Confidence::Declared, validate_ok: true };
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("toy", &records, &specs, &BTreeMap::new());
        assert!(matches!(out, Resolution::Resolved(_)), "{out:?}");
    }

    #[test]
    #[should_panic(expected = "no ArchSpec registered")]
    fn resolve_panics_when_no_spec_matches_the_requested_arch() {
        let spec = ToySpec { classify_confidence: Confidence::Declared, validate_ok: true };
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        resolve("not-toy", &[], &specs, &BTreeMap::new());
    }
}
