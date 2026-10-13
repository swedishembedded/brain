// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which decision model answers when the caller says `"auto"`.
//!
//! Pure: the candidates in, a [`Selection`] and the reasons out. No file is
//! read and no device touched, so the policy is unit-testable and the same
//! function answers for every transport.
//!
//! The order is:
//!
//! 1. **explicitly configured** - the operator named a model; it is used, as
//!    named, whether or not it is validated. A configured model that is not
//!    among the candidates selects nothing and says so: an explicit choice is
//!    never silently replaced by another.
//! 2. **a validated Laya** model,
//! 3. **a validated `decide`** model,
//! 4. **none**.
//!
//! "Validated" means a validation is recorded in the model's manifest. An
//! unvalidated model is never picked automatically, however good it might
//! turn out to be: a caller that wants it names it. A model that is not
//! installed is never picked automatically either, since the call would only
//! fail. Within a rule the lexicographically smallest id wins, so the answer
//! does not depend on the order a directory happened to be listed in.
//!
//! Swedish Embedded AB implements model governance for its clients: a
//! deployment that picks the model it can justify, and says why. If your team
//! needs expertise in policy-driven model selection, you can procure our
//! services by sending an email to info@swedishembedded.com.

use crate::engine::Backend;

/// What the selector knows about one model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub id: String,
    pub backend: Backend,
    /// Its weights are on this host.
    pub installed: bool,
    /// A validation is recorded in its manifest.
    pub validated: bool,
}

/// The rule that decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    Configured,
    ValidatedLaya,
    ValidatedDecide,
    None,
}

impl Rule {
    pub fn slug(self) -> &'static str {
        match self {
            Rule::Configured => "configured",
            Rule::ValidatedLaya => "validated-laya",
            Rule::ValidatedDecide => "validated-decide",
            Rule::None => "none",
        }
    }
}

/// The outcome of a selection and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    pub chosen: Option<String>,
    pub rule: Rule,
    /// One sentence a human can read in a log or an error.
    pub reason: String,
}

/// Choose among `candidates`. `configured` is the operator's explicit choice.
pub fn select(candidates: &[Candidate], configured: Option<&str>) -> Selection {
    if let Some(id) = configured {
        return match candidates.iter().find(|c| c.id == id) {
            Some(c) => Selection {
                chosen: Some(c.id.clone()),
                rule: Rule::Configured,
                reason: format!("'{id}' is the explicitly configured model ({})", describe(c)),
            },
            None => Selection { chosen: None, rule: Rule::None, reason: format!("the configured model '{id}' is not available on this host") },
        };
    }
    for (backend, rule) in [(Backend::Laya, Rule::ValidatedLaya), (Backend::Decide, Rule::ValidatedDecide)] {
        let best = candidates.iter().filter(|c| c.backend == backend && c.installed && c.validated).min_by(|a, b| a.id.cmp(&b.id));
        if let Some(c) = best {
            return Selection { chosen: Some(c.id.clone()), rule, reason: format!("'{}' is the smallest-id installed {} model with a recorded validation", c.id, backend.slug()) };
        }
    }
    Selection { chosen: None, rule: Rule::None, reason: none_reason(candidates) }
}

fn describe(c: &Candidate) -> String {
    let install = if c.installed { "installed" } else { "not installed" };
    let validation = if c.validated { "validated" } else { "unvalidated" };
    format!("{}, {install}, {validation}", c.backend.slug())
}

fn none_reason(candidates: &[Candidate]) -> String {
    if candidates.is_empty() {
        return "no decision model is installed".to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    for c in candidates {
        let why = if !c.installed {
            "not installed"
        } else {
            "installed but has no recorded validation"
        };
        parts.push(format!("'{}' ({why})", c.id));
    }
    format!("no installed decision model has a recorded validation: {}", parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(id: &str, backend: Backend, installed: bool, validated: bool) -> Candidate {
        Candidate { id: id.into(), backend, installed, validated }
    }

    #[test]
    fn an_explicitly_configured_model_beats_everything_even_unvalidated() {
        let cs = [c("a/laya", Backend::Laya, true, true), c("b/mine", Backend::Decide, true, false)];
        let s = select(&cs, Some("b/mine"));
        assert_eq!((s.chosen.as_deref(), s.rule), (Some("b/mine"), Rule::Configured));
        assert!(s.reason.contains("unvalidated") && s.reason.contains("explicitly configured"), "{}", s.reason);
    }

    #[test]
    fn a_configured_model_that_is_missing_selects_nothing_rather_than_something_else() {
        let cs = [c("a/laya", Backend::Laya, true, true)];
        let s = select(&cs, Some("gone/model"));
        assert_eq!((s.chosen, s.rule), (None, Rule::None));
        assert!(s.reason.contains("gone/model"), "{}", s.reason);
    }

    #[test]
    fn validated_laya_beats_validated_decide() {
        let cs = [c("d/decide", Backend::Decide, true, true), c("l/laya", Backend::Laya, true, true)];
        let s = select(&cs, None);
        assert_eq!((s.chosen.as_deref(), s.rule), (Some("l/laya"), Rule::ValidatedLaya));
    }

    #[test]
    fn validated_decide_is_chosen_when_no_laya_qualifies() {
        let cs = [c("l/laya", Backend::Laya, true, false), c("d/decide", Backend::Decide, true, true)];
        let s = select(&cs, None);
        assert_eq!((s.chosen.as_deref(), s.rule), (Some("d/decide"), Rule::ValidatedDecide));
    }

    #[test]
    fn an_unvalidated_or_uninstalled_model_is_never_chosen_automatically() {
        let cs = [c("l/laya", Backend::Laya, true, false), c("l/other", Backend::Laya, false, true), c("d/decide", Backend::Decide, true, false)];
        let s = select(&cs, None);
        assert_eq!((s.chosen.as_deref(), s.rule), (None, Rule::None));
        for id in ["l/laya", "l/other", "d/decide"] {
            assert!(s.reason.contains(id), "the reason must account for '{id}': {}", s.reason);
        }
        assert!(s.reason.contains("not installed") && s.reason.contains("no recorded validation"), "{}", s.reason);
    }

    #[test]
    fn nothing_installed_says_so() {
        let s = select(&[], None);
        assert_eq!(s.rule, Rule::None);
        assert!(s.reason.contains("no decision model"), "{}", s.reason);
    }

    #[test]
    fn ties_resolve_to_the_smallest_id_whatever_the_listing_order() {
        let a = [c("z/laya", Backend::Laya, true, true), c("a/laya", Backend::Laya, true, true)];
        let b = [c("a/laya", Backend::Laya, true, true), c("z/laya", Backend::Laya, true, true)];
        assert_eq!(select(&a, None).chosen.as_deref(), Some("a/laya"));
        assert_eq!(select(&b, None), select(&a, None));
    }

    #[test]
    fn rule_slugs_are_stable() {
        let s: Vec<&str> = [Rule::Configured, Rule::ValidatedLaya, Rule::ValidatedDecide, Rule::None].iter().map(|r| r.slug()).collect();
        assert_eq!(s, ["configured", "validated-laya", "validated-decide", "none"]);
    }
}
