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

/// The question plus every real candidate's own selector flags - what a
/// caller types next to pick one, straight from the resolver's own
/// [`ModelCandidate::selector`]. Every architecture's own CLI needs this
/// identical rendering (only the architecture name in the message differs,
/// and that already lives on `a.arch`), so it lives here once instead of
/// being hand-copied per architecture.
pub fn describe_ambiguity(a: &Ambiguity) -> String {
    let mut s = match &a.question {
        Question::Role { role } => format!("{}: more than one candidate for '{role}' - nothing picked automatically. Choose one:\n", a.arch),
        Question::Variant { shape_class } => format!("{}: which {shape_class} variant? (not recoverable from the weights' own shape). Choose one:\n", a.arch),
        Question::Unverifiable { path } => format!("{}: {} could not be read enough to classify\n", a.arch, path.display()),
    };
    for c in &a.choices {
        for (flag, value) in &c.selector {
            s.push_str(&format!("  {flag} {value}\n"));
        }
    }
    s
}

/// Every missing role's own doc string plus any near-misses (an interrupted
/// download, say) that explain why it looks empty anyway - the same
/// rendering every architecture's CLI needs, shared for the same reason as
/// [`describe_ambiguity`].
pub fn describe_missing(m: &Missing) -> String {
    let mut s = String::new();
    for r in &m.roles {
        s.push_str(&format!("{}: {}: {}\n", m.arch, r.role, r.doc));
        for near in &r.near_misses {
            s.push_str(&format!("  {near}\n"));
        }
    }
    s
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
    /// Every role name this architecture declares, in the order [`resolve`]
    /// reports them - both required and optional (see [`Self::optional_roles`]).
    fn roles(&self) -> &'static [&'static str];
    /// The subset of [`Self::roles`] that MAY resolve to zero candidates
    /// without producing [`Resolution::Missing`] - a role real production
    /// code treats as genuinely opt-in (present, it's used; absent, the
    /// caller's own documented fallback runs), not a placeholder for "not
    /// implemented yet". A role in this list that DOES have a candidate is
    /// still picked exactly like a required one - this only changes what
    /// happens at zero.
    ///
    /// Default: empty, so every declared role is required - the behavior
    /// every `ArchSpec` had before this existed.
    fn optional_roles(&self) -> &'static [&'static str] {
        &[]
    }
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

    /// [`MissingRole::doc`] for a role [`Self::classify`] found no candidate
    /// for, with no override supplied either. The default reproduces
    /// `resolve`'s own historical generic wording; an architecture whose role
    /// has NO on-disk acquisition path at all (no `default_ref`, and nothing
    /// content-based `classify` could ever recognize as a fetchable default -
    /// `llava`/`campplus`/`s3tokenizer`'s shared case) overrides this with
    /// [`no_default_checkpoint_doc`] to name the real escape hatch (the
    /// override flag) instead of a bare "no artifact classifies" - the
    /// resolver-side equivalent of `crate::supply::ensure_env_weights_with`'s
    /// identically-worded env-based "no default checkpoint known" error.
    fn missing_doc(&self, role: &str) -> String {
        format!("no artifact classifies as {role} for arch {}", self.arch())
    }
}

/// Shared wording for [`ArchSpec::missing_doc`] on a role with no on-disk
/// acquisition path at all - names the exact `--<dashed-role>` override flag
/// the CLI's per-role flag stripper (`crates/cli/src/resolver_cli.rs`'s
/// `extract_role_overrides`, every resolver-migrated architecture's own)
/// derives from `role`, so the message tells the caller precisely what to
/// type next instead of leaving them to guess at a flag spelling from a
/// generic "nothing classifies" message.
pub fn no_default_checkpoint_doc(arch: &str, role: &str) -> String {
    format!("{arch}: no default checkpoint known for role {role:?} -- pass --{} to name one explicitly", role.replace('_', "-"))
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
            // Two ways to name a candidate: the exact absolute path an
            // advanced/scripted caller already has, or (matched on trailing
            // path COMPONENTS, not a substring, and against every ANCESTOR
            // directory too - not just the record's own path) a
            // `<vendor>/<repo>` reference the same way `--model`/`brain
            // pull` already accept one elsewhere in this codebase -
            // `--tokenizer Qwen/Qwen3-8B` must resolve to the tokenizer.json
            // FILE that role actually needs, not the repo DIRECTORY the
            // reference literally names, and must work without the caller
            // ever typing this store's absolute root.
            let want = Path::new(path.as_str());
            let matches = |p: &Path| p.to_string_lossy() == *path || p.ancestors().any(|a| a.ends_with(want));
            // Prefer a candidate `classify` already picked out for THIS
            // role - it is what makes an ancestor-directory reference land
            // on the right FILE within that directory, when more than one
            // real artifact happens to live under it. Only an override
            // naming something `classify` found no candidate for at all
            // falls back to a bare whole-inventory search (the advanced
            // escape hatch: point at an exact path by hand).
            let found = by_role
                .get(role)
                .and_then(|cands| cands.iter().map(|(idx, _)| *idx).find(|&idx| matches(&records[idx].path)))
                .or_else(|| records.iter().position(|r| matches(&r.path)));
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

    let optional = spec.optional_roles();
    let missing: Vec<MissingRole> = per_role
        .iter()
        .filter(|(role, r)| matches!(r, RoleResult::None) && !optional.contains(role))
        .map(|(role, _)| MissingRole { role: role.to_string(), doc: spec.missing_doc(role), near_misses: near_misses(records) })
        .collect();
    if !missing.is_empty() {
        return Resolution::Missing(Box::new(Missing { arch: arch.to_string(), roles: missing }));
    }

    // A best-current single choice per role, for building EVERY role's
    // hypothetical assemblies - including the one about to be reported as
    // ambiguous, which needs every OTHER role already pinned down. An
    // optional role with zero candidates contributes no entry at all - it
    // already passed the missing-role filter above precisely because
    // nothing on disk answers for it, so `chosen`/the resulting `Assembly`
    // must not claim a path for it either.
    let mut chosen: BTreeMap<String, usize> = BTreeMap::new();
    for (role, r) in &per_role {
        match r {
            RoleResult::One(i) => {
                chosen.insert(role.to_string(), *i);
            }
            RoleResult::Ambiguous(cands) => {
                chosen.insert(role.to_string(), cands[0].0);
            }
            RoleResult::None => {}
        }
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

// ===================== shared tokenizer-role classification =====================
//
// A tokenizer.json carries no `architecture`/`_class_name` field of its own
// to check against, so "is this valid JSON" is true of every tokenizer.json
// in an entire store, from every unrelated architecture. Every architecture
// with a tokenizer-shaped role (flux2's `tokenizer`, and every future
// architecture with one - wan, s3dit, cosyvoice, minimaxmusic3, qwen3tts all
// have the identical shape) needs the same two real signals to narrow that
// down: the file was published by the same vendor as a component this same
// `classify` call already identified as one of that architecture's real
// roles, AND its own vocabulary size is actually compatible with that
// component's declared embedding-table size - one vendor commonly publishes
// several differently-sized, differently-vocabbed models under the same
// top-level directory, so vendor alone is not enough either.

/// The path component directly under `root` that `path` falls within (the
/// vendor directory, in store terms) - `None` if `path` is not under `root`
/// at all. Two paths sharing this are two artifacts published by the same
/// vendor, regardless of how deeply either one is nested below that point:
/// a plain HF checkpoint's own directory, a diffusers pipeline's per-role
/// subdirectories, and a hand-placed loose file plus its own small
/// tokenizer-only sub-repo are all real, differently-shaped layouts a real
/// vendor's release takes - and all share this one property.
pub fn vendor_dir(path: &Path, root: &Path) -> Option<PathBuf> {
    let rel = path.strip_prefix(root).ok()?;
    rel.components().next().map(|c| root.join(c))
}

/// A tokenizer's own vocabulary size: the BPE `vocab` table plus
/// `added_tokens` - real content, not the file's name.
pub fn tokenizer_vocab_count(bytes: &[u8]) -> Option<usize> {
    let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let base = v.get("model")?.get("vocab")?.as_object()?.len();
    let added = v.get("added_tokens").and_then(|a| a.as_array()).map_or(0, Vec::len);
    Some(base + added)
}

/// The embedding table row count a checkpoint declares: an HF directory's
/// `config.json` `vocab_size`, or a GGUF's own `tokenizer.ggml.tokens` array
/// length (the real embedded vocab, present on every real release regardless
/// of whether a separate `vocab_size` KV is).
pub fn checkpoint_vocab_size(path: &Path) -> Option<usize> {
    if path.extension().is_some_and(|e| e == "gguf") {
        let g = checkpoint::gguf::MmapGguf::open(&path.to_string_lossy()).ok()?;
        g.kv().get("tokenizer.ggml.tokens").and_then(|v| if let checkpoint::gguf::GgufValue::Array(a) = v { Some(a.len()) } else { None })
    } else {
        let bytes = std::fs::read(path.join("config.json")).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
        v.get("vocab_size").and_then(serde_json::Value::as_u64).map(|n| n as usize)
    }
}

/// A tokenizer whose own vocabulary is within 5% of a candidate's declared
/// embedding-table size: real checkpoints pad the embedding table a little
/// past the tokenizer's literal entry count (reserved/unused slots), so
/// exact equality is too strict, but a genuinely different model's
/// tokenizer (a different vocabulary entirely, not a padding difference)
/// misses by a wide margin, not a few hundred tokens.
pub fn vocab_is_compatible(tokenizer_count: usize, checkpoint_vocab: usize) -> bool {
    tokenizer_count <= checkpoint_vocab && checkpoint_vocab - tokenizer_count <= checkpoint_vocab / 20
}

/// Classify every usable [`crate::inventory::ArtifactKind::TokenizerJson`]
/// record in `records` as `role`, using [`vendor_dir`] and
/// [`vocab_is_compatible`] as the real signal a bare "is this valid JSON"
/// check cannot provide. `dependency_candidates` is the set of already
/// classified paths this tokenizer must pair with (an architecture's own
/// text-encoder/LLM-shaped role candidates, from earlier in the same
/// `classify` call) - a tokenizer is a candidate only if it shares a vendor
/// with at least one of them AND its own vocabulary is compatible with that
/// same candidate's declared embedding size.
///
/// Appends to `out` in place, matching every other `classify_*` helper in
/// this module.
pub fn classify_tokenizer_role(records: &[ArtifactRecord], root: &Path, role: &str, dependency_candidates: &[&Path], out: &mut Vec<(usize, String, Confidence)>) {
    let vendor_dirs: std::collections::BTreeSet<PathBuf> = dependency_candidates.iter().filter_map(|p| vendor_dir(p, root)).collect();
    for (idx, rec) in records.iter().enumerate() {
        if !rec.usable() || rec.kind != crate::inventory::ArtifactKind::TokenizerJson {
            continue;
        }
        let Ok(bytes) = std::fs::read(&rec.path) else { continue };
        let Some(tok_count) = tokenizer_vocab_count(&bytes) else { continue };
        let Some(vendor) = vendor_dir(&rec.path, root) else { continue };
        if !vendor_dirs.contains(&vendor) {
            continue;
        }
        let compatible = dependency_candidates.iter().filter(|p| vendor_dir(p, root).as_deref() == Some(vendor.as_path())).any(|p| checkpoint_vocab_size(p).is_some_and(|v| vocab_is_compatible(tok_count, v)));
        if compatible {
            out.push((idx, role.to_string(), Confidence::Declared));
        }
    }
}

/// Read every `brain.manifest.json` (`crate::MANIFEST_FILE`) under
/// `root/<vendor>/<repo>/` that declares `family`, mapping its own recorded
/// roles straight onto whichever already-scanned [`ArtifactRecord`] lives at
/// each role's resolved path - [`Confidence::Recorded`], never re-derived
/// from that role's own raw file content. This is the shared short-circuit
/// every architecture with a real multi-file conversion step
/// (`convert_files`'s own `brain.manifest.json` writer, in
/// `crates/cli/src/supply.rs`) needs identically, so a compound-converted
/// checkpoint's roles are recognized once here rather than re-derived per
/// architecture.
///
/// Matches by PATH against `records`, not by [`crate::inventory::
/// ArtifactKind::Compound`] alone: a role name is a property of the
/// manifest, not of the record, so this is what ties the two together.
/// Appends to `out` in place, matching every other `classify_*` helper.
pub fn classify_compound_manifest(records: &[ArtifactRecord], root: &Path, family: &str, out: &mut Vec<(usize, String, Confidence)>) {
    let Ok(vendors) = std::fs::read_dir(root) else { return };
    for vendor in vendors.flatten() {
        let vendor_path = vendor.path();
        if !vendor_path.is_dir() {
            continue;
        }
        let Ok(repos) = std::fs::read_dir(&vendor_path) else { continue };
        for repo in repos.flatten() {
            let repo_path = repo.path();
            if !repo_path.is_dir() {
                continue;
            }
            let Ok(bytes) = std::fs::read(repo_path.join(crate::MANIFEST_FILE)) else { continue };
            let Ok(manifest) = serde_json::from_slice::<crate::CompoundManifest>(&bytes) else { continue };
            if manifest.family != family {
                continue;
            }
            for (role, rel) in &manifest.roles {
                let rel_path = Path::new(rel);
                if rel_path.is_absolute() || rel_path.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
                    continue;
                }
                let want = repo_path.join(rel_path);
                if let Some(idx) = records.iter().position(|r| r.usable() && r.path == want) {
                    out.push((idx, role.clone(), Confidence::Recorded));
                }
            }
        }
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

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-modelstore-resolve-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: crate::inventory::ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    fn write_tokenizer_json(path: &Path, vocab_count: usize) {
        let vocab: serde_json::Map<String, serde_json::Value> = (0..vocab_count).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
        std::fs::write(path, serde_json::to_vec(&serde_json::json!({"model": {"vocab": vocab}, "added_tokens": []})).unwrap()).unwrap();
    }

    fn write_hf_checkpoint(dir: &Path, vocab_size: usize) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"vocab_size": vocab_size})).unwrap()).unwrap();
    }

    /// Two different architectures' tokenizer-classification needs, at
    /// once: a role whose real candidate lives in the SAME vendor directory
    /// as the checkpoint it must pair with, at a matching vocab size, and a
    /// same-vendor-but-wrong-model tokenizer (a real store commonly has
    /// exactly this - one vendor publishing several differently-sized
    /// checkpoints) that must not classify despite sharing the vendor.
    #[test]
    fn classify_tokenizer_role_requires_both_vendor_and_vocab_match() {
        let dir = tmp("tokenizer-role");
        let checkpoint_dir = dir.join("vendor").join("model-a");
        write_hf_checkpoint(&checkpoint_dir, 100);
        let real_tok = dir.join("vendor").join("model-a").join("tokenizer.json");
        write_tokenizer_json(&real_tok, 100);
        // Same vendor, a DIFFERENT (larger) checkpoint - real signal a
        // vendor-only check would miss.
        let other_checkpoint_dir = dir.join("vendor").join("model-b");
        write_hf_checkpoint(&other_checkpoint_dir, 400);
        let mismatched_tok = dir.join("vendor").join("model-b-tokenizer.json");
        write_tokenizer_json(&mismatched_tok, 400);
        // A different vendor entirely, with a vocab size that WOULD match
        // model-a if vendor weren't checked at all.
        let unrelated_tok = dir.join("other-vendor").join("tokenizer.json");
        std::fs::create_dir_all(unrelated_tok.parent().unwrap()).unwrap();
        write_tokenizer_json(&unrelated_tok, 100);

        let records = vec![complete(real_tok.clone(), crate::inventory::ArtifactKind::TokenizerJson), complete(mismatched_tok, crate::inventory::ArtifactKind::TokenizerJson), complete(unrelated_tok, crate::inventory::ArtifactKind::TokenizerJson)];
        let dependency_candidates: Vec<&Path> = vec![checkpoint_dir.as_path()];
        let mut out = Vec::new();
        classify_tokenizer_role(&records, &dir, "tokenizer", &dependency_candidates, &mut out);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(records[out[0].0].path, real_tok);
    }

    /// Padding tolerance: a real checkpoint's declared vocab_size is
    /// slightly larger than the tokenizer's literal entry count (reserved
    /// slots) - within 5% must still match.
    #[test]
    fn vocab_is_compatible_tolerates_real_world_padding_but_not_a_different_model() {
        assert!(vocab_is_compatible(151_669, 151_936), "real Qwen3-8B numbers must be compatible");
        assert!(!vocab_is_compatible(151_669, 248_320), "a genuinely different model's vocab must not be compatible");
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

    /// A toy TWO-role arch: "component" (any path containing "component")
    /// and "sidecar" (any path containing "sidecar") - enough to test an
    /// override reference that names a DIRECTORY containing several real
    /// candidates for DIFFERENT roles, the shape a plain HF checkpoint
    /// directory (weights + a co-located tokenizer file) actually takes.
    struct TwoRoleSpec;
    impl ArchSpec for TwoRoleSpec {
        fn arch(&self) -> &'static str {
            "two"
        }
        fn roles(&self) -> &'static [&'static str] {
            &["component", "sidecar"]
        }
        fn classify(&self, records: &[ArtifactRecord], _root: &Path) -> Vec<(usize, String, Confidence)> {
            records
                .iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let s = r.path.to_string_lossy();
                    if s.contains("sidecar") {
                        Some((i, "sidecar".to_string(), Confidence::Declared))
                    } else if s.contains("component") {
                        Some((i, "component".to_string(), Confidence::Declared))
                    } else {
                        None
                    }
                })
                .collect()
        }
        fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
            if chosen.contains_key("component") && chosen.contains_key("sidecar") {
                Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/two".to_string(), variant: None }))
            } else {
                Err("missing a role".to_string())
            }
        }
        fn validate(&self, _assembly: &Assembly) -> Result<(), String> {
            Ok(())
        }
    }

    /// A `<vendor>/<repo>`-shaped override for one role must resolve to a
    /// real candidate ALREADY classified for THAT role, not to whatever
    /// record the reference's trailing path components happen to match
    /// first - a bare directory reference for a role whose real artifact is
    /// a FILE living inside that directory (a plain HF checkpoint's own
    /// tokenizer.json, say) must land on the file, never the directory.
    #[test]
    fn an_override_directory_reference_resolves_to_the_classified_file_inside_it_not_the_directory() {
        let records = vec![rec("/models/vendor/repo-component"), rec("/models/vendor/repo-component/sidecar.json")];
        let spec = TwoRoleSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let mut overrides = BTreeMap::new();
        overrides.insert("sidecar".to_string(), "vendor/repo-component".to_string());
        let found = match resolve("two", &records, &specs, &overrides) {
            Resolution::Missing(m) => panic!("expected the sidecar role to resolve, got Missing: {m:?}"),
            Resolution::Ambiguous(a) => panic!("expected the sidecar role to resolve, got Ambiguous: {a:?}"),
            Resolution::Resolved(a) => a,
        };
        assert_eq!(found.roles["sidecar"], PathBuf::from("/models/vendor/repo-component/sidecar.json"));
    }

    fn empty_assembly(arch: &str) -> Assembly {
        Assembly { id: format!("local/{arch}"), arch: arch.to_string(), variant: None, roles: BTreeMap::new(), provenance: Vec::new() }
    }

    /// The printed message must name the actual question and every real
    /// candidate's own selector flags - what a caller types next - straight
    /// from the resolver's own output, never a hand-summarized guess. Every
    /// architecture's CLI needs this identical rendering, only the
    /// architecture name (already on `Ambiguity::arch`) differs.
    #[test]
    fn describe_ambiguity_prints_the_question_and_every_selector() {
        let choices = vec![
            ModelCandidate { assembly: Box::new(empty_assembly("wan")), selector: vec![("--variant".to_string(), "t2v-14b".to_string())], summary: "wan t2v-14b".to_string() },
            ModelCandidate { assembly: Box::new(empty_assembly("wan")), selector: vec![("--variant".to_string(), "t2v-1.3b".to_string())], summary: "wan t2v-1.3b".to_string() },
        ];
        let a = Ambiguity { arch: "wan".to_string(), question: Question::Variant { shape_class: "14b".to_string() }, choices };
        let out = describe_ambiguity(&a);
        assert!(out.contains("wan"), "{out}");
        assert!(out.contains("14b"), "{out}");
        assert!(out.contains("--variant t2v-14b"), "{out}");
        assert!(out.contains("--variant t2v-1.3b"), "{out}");
    }

    /// A `Role` ambiguity names the role in the question line, not only in
    /// the selectors below it.
    #[test]
    fn describe_ambiguity_names_the_role_for_a_role_question() {
        let choices = vec![ModelCandidate {
            assembly: Box::new(empty_assembly("wan")),
            selector: vec![("--text-encoder".to_string(), "/models/google/umt5-xxl".to_string())],
            summary: "text_encoder: /models/google/umt5-xxl".to_string(),
        }];
        let a = Ambiguity { arch: "wan".to_string(), question: Question::Role { role: "text_encoder".to_string() }, choices };
        let out = describe_ambiguity(&a);
        assert!(out.contains("text_encoder"), "{out}");
        assert!(out.contains("--text-encoder /models/google/umt5-xxl"), "{out}");
    }

    /// Every missing role's own doc string and near-misses (an interrupted
    /// download, say) must survive into the printed message, per role.
    #[test]
    fn describe_missing_prints_every_roles_doc_and_near_misses() {
        let m = Missing {
            arch: "wan".to_string(),
            roles: vec![
                MissingRole { role: "dit".to_string(), doc: "no artifact classifies as dit for arch wan".to_string(), near_misses: vec!["an interrupted download exists at /models/wan/dit.gguf".to_string()] },
                MissingRole { role: "vae".to_string(), doc: "no artifact classifies as vae for arch wan".to_string(), near_misses: Vec::new() },
            ],
        };
        let out = describe_missing(&m);
        assert!(out.contains("dit") && out.contains("no artifact classifies as dit"), "{out}");
        assert!(out.contains("interrupted download exists at /models/wan/dit.gguf"), "{out}");
        assert!(out.contains("vae") && out.contains("no artifact classifies as vae"), "{out}");
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

    /// A caller's override (a CLI flag like `--text-encoder`) may name a
    /// `<vendor>/<repo>` reference the same way every other part of this
    /// codebase does, not only a raw absolute path -- matching on the
    /// trailing path components (`Path::ends_with`) rather than requiring
    /// exact string equality is what makes `--text-encoder Qwen/Qwen3-8B`
    /// work the same way `--model Qwen/Qwen3-8B` and `brain pull
    /// Qwen/Qwen3-8B` already do elsewhere.
    #[test]
    fn an_override_accepts_a_vendor_repo_reference_not_only_a_raw_path() {
        let records = vec![rec("/models/Qwen/Qwen3-8B")];
        let spec = ToySpec { classify_confidence: Confidence::Guessed, validate_ok: true };
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let mut overrides = BTreeMap::new();
        overrides.insert("dit".to_string(), "Qwen/Qwen3-8B".to_string());
        let out = resolve("toy", &records, &specs, &overrides);
        match out {
            Resolution::Resolved(a) => assert_eq!(a.roles["dit"], PathBuf::from("/models/Qwen/Qwen3-8B")),
            other => panic!("expected Resolved, got {out:?}", out = other),
        }
    }

    /// A `<vendor>/<repo>`-shaped override must still match nothing when no
    /// record's path actually ends with it -- this is name-based matching
    /// against real scanned artifacts, never a filesystem probe of its own.
    #[test]
    fn a_vendor_repo_override_matching_no_record_is_missing_not_a_panic() {
        let records = vec![rec("/models/toy/dit.bin")];
        let spec = ToySpec { classify_confidence: Confidence::Guessed, validate_ok: true };
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let mut overrides = BTreeMap::new();
        overrides.insert("dit".to_string(), "Nobody/Nothing".to_string());
        let out = resolve("toy", &records, &specs, &overrides);
        assert!(matches!(out, Resolution::Missing(_)), "{out:?}");
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

    /// A spec that never classifies anything, standing in for
    /// `llava`/`campplus`/`s3tokenizer` - no `default_ref`, and nothing
    /// on-disk content could ever satisfy the role, so [`ArchSpec::missing_doc`]
    /// is overridden with [`no_default_checkpoint_doc`] instead of the default
    /// generic wording.
    struct NoAcquisitionSpec;
    impl ArchSpec for NoAcquisitionSpec {
        fn arch(&self) -> &'static str {
            "noacq"
        }
        fn roles(&self) -> &'static [&'static str] {
            &["weights"]
        }
        fn classify(&self, _records: &[ArtifactRecord], _root: &Path) -> Vec<(usize, String, Confidence)> {
            Vec::new()
        }
        fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
            chosen.get("weights").ok_or("no weights chosen")?;
            Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/noacq".to_string(), variant: None }))
        }
        fn validate(&self, _assembly: &Assembly) -> Result<(), String> {
            Ok(())
        }
        fn missing_doc(&self, role: &str) -> String {
            no_default_checkpoint_doc(self.arch(), role)
        }
    }

    /// The default [`ArchSpec::missing_doc`] every OTHER spec in this module
    /// (never overriding it) still gets - `resolve`'s prior generic wording,
    /// unchanged by this mechanism existing.
    #[test]
    fn missing_doc_defaults_to_the_generic_wording_when_a_spec_does_not_override_it() {
        let records: Vec<ArtifactRecord> = vec![rec("/models/toy/unrelated.bin")];
        let spec = ToySpec { classify_confidence: Confidence::Declared, validate_ok: true };
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        match resolve("toy", &records, &specs, &BTreeMap::new()) {
            Resolution::Missing(m) => assert_eq!(m.roles[0].doc, "no artifact classifies as dit for arch toy"),
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    /// A role with no on-disk acquisition path at all: with no override and
    /// nothing to classify, `resolve` reports Missing with a doc naming BOTH
    /// that no default checkpoint is known AND the exact override flag to
    /// pass - never the bare "no artifact classifies" wording, which leaves
    /// the caller to guess how to actually run this architecture.
    #[test]
    fn a_role_with_no_acquisition_path_names_the_override_flag_instead_of_the_generic_wording() {
        let spec = NoAcquisitionSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        match resolve("noacq", &[], &specs, &BTreeMap::new()) {
            Resolution::Missing(m) => {
                assert_eq!(m.roles.len(), 1);
                let doc = &m.roles[0].doc;
                assert!(doc.contains("no default checkpoint known"), "{doc}");
                assert!(doc.contains("--weights"), "{doc}");
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    /// The override escape hatch still resolves cleanly for a no-acquisition-
    /// path role, exactly as it does for every other architecture - the
    /// override bypasses `classify` entirely (see
    /// `an_override_is_honored_without_reclassifying` above), so an empty
    /// `classify` never blocks it.
    #[test]
    fn a_role_with_no_acquisition_path_still_resolves_via_an_explicit_override() {
        let records = vec![rec("/models/noacq/hand-placed.bin")];
        let spec = NoAcquisitionSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let mut overrides = BTreeMap::new();
        overrides.insert("weights".to_string(), "/models/noacq/hand-placed.bin".to_string());
        match resolve("noacq", &records, &specs, &overrides) {
            Resolution::Resolved(a) => assert_eq!(a.roles["weights"], PathBuf::from("/models/noacq/hand-placed.bin")),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// One required role ("dit"), one optional role ("sidecar") - the toy
    /// arch a real architecture with an opt-in, absent-today extra weight
    /// (LTX-2.5's real DiT/text-encoder/tokenizer roles) needs to prove
    /// against.
    struct RequiredAndOptionalSpec;
    impl ArchSpec for RequiredAndOptionalSpec {
        fn arch(&self) -> &'static str {
            "reqopt"
        }
        fn roles(&self) -> &'static [&'static str] {
            &["dit", "sidecar"]
        }
        fn optional_roles(&self) -> &'static [&'static str] {
            &["sidecar"]
        }
        fn classify(&self, records: &[ArtifactRecord], _root: &Path) -> Vec<(usize, String, Confidence)> {
            records
                .iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let s = r.path.to_string_lossy();
                    if s.contains("sidecar") {
                        Some((i, "sidecar".to_string(), Confidence::Declared))
                    } else if s.contains("dit") {
                        Some((i, "dit".to_string(), Confidence::Declared))
                    } else {
                        None
                    }
                })
                .collect()
        }
        fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
            if chosen.contains_key("dit") {
                Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/reqopt".to_string(), variant: None }))
            } else {
                Err("missing dit".to_string())
            }
        }
        fn validate(&self, _assembly: &Assembly) -> Result<(), String> {
            Ok(())
        }
    }

    /// The gap this whole extension exists to close: an optional role with
    /// zero candidates on disk must resolve cleanly (not `Missing`), and the
    /// resulting `Assembly` must carry no path for it at all - there is
    /// nothing to claim one from.
    #[test]
    fn an_optional_role_with_zero_candidates_resolves_instead_of_going_missing() {
        let records = vec![rec("/models/reqopt/dit.bin")];
        let spec = RequiredAndOptionalSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("reqopt", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.roles["dit"], PathBuf::from("/models/reqopt/dit.bin"));
                assert!(!a.roles.contains_key("sidecar"), "an absent optional role must carry no path: {a:?}");
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// The required role in the SAME arch still goes `Missing` at zero
    /// candidates - `optional_roles` narrows the exemption to exactly the
    /// roles it names, not every role on the spec.
    #[test]
    fn a_required_role_with_zero_candidates_still_goes_missing_even_when_the_arch_has_an_optional_one_too() {
        let records = vec![rec("/models/reqopt/sidecar.bin")];
        let spec = RequiredAndOptionalSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("reqopt", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Missing(m) => {
                assert_eq!(m.roles.len(), 1, "{m:?}");
                assert_eq!(m.roles[0].role, "dit");
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    /// An optional role that DOES have a candidate is picked exactly like a
    /// required one - `optional_roles` only changes what happens at zero.
    #[test]
    fn an_optional_role_with_a_real_candidate_is_still_picked() {
        let records = vec![rec("/models/reqopt/dit.bin"), rec("/models/reqopt/sidecar.bin")];
        let spec = RequiredAndOptionalSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("reqopt", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => assert_eq!(a.roles["sidecar"], PathBuf::from("/models/reqopt/sidecar.bin")),
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// A directory carrying a `brain.manifest.json` for the SAME family the
    /// caller asks about maps each declared role onto whichever already-
    /// scanned record sits at that role's resolved path - the two-role
    /// (`ckpt`/`weights_dir`) shape a real `convert_files`-written manifest
    /// takes.
    #[test]
    fn classify_compound_manifest_maps_declared_roles_onto_matching_records_by_path() {
        let dir = tmp("compound-manifest-classify");
        let repo = dir.join("Qwen").join("Qwen3-TTS-12Hz-0.6B-Base");
        std::fs::create_dir_all(repo.join("brain_tts")).unwrap();
        let manifest = crate::CompoundManifest {
            id: "Qwen/Qwen3-TTS-12Hz-0.6B-Base".to_string(),
            family: "qwen3tts".to_string(),
            roles: BTreeMap::from([("ckpt".to_string(), ".".to_string()), ("weights_dir".to_string(), "brain_tts".to_string())]),
        };
        std::fs::write(repo.join(crate::MANIFEST_FILE), serde_json::to_vec(&manifest).unwrap()).unwrap();
        let records = vec![
            complete(repo.clone(), crate::inventory::ArtifactKind::Compound),
            complete(repo.join("brain_tts"), crate::inventory::ArtifactKind::Compound),
        ];

        let mut out = Vec::new();
        classify_compound_manifest(&records, &dir, "qwen3tts", &mut out);
        assert_eq!(out.len(), 2, "{out:?}");
        assert!(out.contains(&(0, "ckpt".to_string(), Confidence::Recorded)), "{out:?}");
        assert!(out.contains(&(1, "weights_dir".to_string(), Confidence::Recorded)), "{out:?}");
    }

    /// A manifest declaring a DIFFERENT family must contribute nothing - a
    /// compound-converted checkpoint for one architecture must never
    /// classify as another's roles just because both happen to use the
    /// same on-disk manifest mechanism.
    #[test]
    fn classify_compound_manifest_ignores_a_manifest_for_a_different_family() {
        let dir = tmp("compound-manifest-wrong-family");
        let repo = dir.join("Qwen").join("Other-Model");
        std::fs::create_dir_all(&repo).unwrap();
        let manifest = crate::CompoundManifest { id: "Qwen/Other-Model".to_string(), family: "wan".to_string(), roles: BTreeMap::from([("ckpt".to_string(), ".".to_string())]) };
        std::fs::write(repo.join(crate::MANIFEST_FILE), serde_json::to_vec(&manifest).unwrap()).unwrap();
        let records = vec![complete(repo.clone(), crate::inventory::ArtifactKind::Compound)];

        let mut out = Vec::new();
        classify_compound_manifest(&records, &dir, "qwen3tts", &mut out);
        assert!(out.is_empty(), "{out:?}");
    }
}
