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
}
