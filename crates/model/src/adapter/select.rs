// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generic target selection over a model's [`super::LinearSite`]s, plus a
//! dependency-free glob matcher. No `regex`: the crate is layer 3
//! (`brain-model`), reachable from every model crate AND `brain-rl`'s
//! leaf-crate closure (`scripts/gates/check-crate-layers.sh`) AND the wasm
//! build, and `regex` sits in `Cargo.lock` only transitively today - adding
//! it as a direct dependency here would put a full regex engine in all
//! three. The glob below covers every pattern a PEFT config actually needs
//! (`blocks.*.attn.q`, `**.ffn.*`, `blocks.{0,1,2}.*`).

use super::LinearSite;

/// Inclusive-optional layer bounds plus a stride, for `Layers { .. }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct LayerRange {
    pub first: Option<usize>,
    pub last: Option<usize>,
    /// Keep one layer out of every `every` (1 = keep all).
    pub every: usize,
}

impl LayerRange {
    pub fn all() -> LayerRange {
        LayerRange { first: None, last: None, every: 1 }
    }

    fn contains(&self, layer: Option<usize>) -> bool {
        let Some(l) = layer else { return false };
        if let Some(f) = self.first {
            if l < f {
                return false;
            }
        }
        if let Some(last) = self.last {
            if l > last {
                return false;
            }
        }
        let every = self.every.max(1);
        (l - self.first.unwrap_or(0)) % every == 0
    }
}

/// Which of a model's [`LinearSite`]s a plan targets.
#[derive(Clone, Debug)]
pub enum TargetSelector {
    All,
    Names(Vec<String>),
    Leaves(Vec<String>),
    Suffixes(Vec<String>),
    Glob(Vec<String>),
    Layers { inner: Box<TargetSelector>, range: LayerRange },
    Any(Vec<TargetSelector>),
    Not(Box<TargetSelector>),
}

impl TargetSelector {
    pub fn matches(&self, site: &LinearSite) -> bool {
        match self {
            TargetSelector::All => true,
            TargetSelector::Names(names) => names.iter().any(|n| n == &site.name),
            TargetSelector::Leaves(leaves) => leaves.iter().any(|l| l == site.leaf),
            TargetSelector::Suffixes(suffixes) => suffixes.iter().any(|s| site.name.ends_with(s.as_str())),
            TargetSelector::Glob(patterns) => patterns.iter().any(|p| glob_match(p, &site.name)),
            TargetSelector::Layers { inner, range } => range.contains(site.layer) && inner.matches(site),
            TargetSelector::Any(inner) => inner.iter().any(|s| s.matches(site)),
            TargetSelector::Not(inner) => !inner.matches(site),
        }
    }

    pub fn attention() -> TargetSelector {
        TargetSelector::Glob(vec!["**.attn*".to_string(), "**.self_attn.*".to_string(), "**.cross_attn.*".to_string()])
    }

    pub fn mlp() -> TargetSelector {
        TargetSelector::Glob(vec!["**.ffn.*".to_string(), "**.mlp.*".to_string(), "**.feed_forward.*".to_string()])
    }

    pub fn experts() -> TargetSelector {
        TargetSelector::Glob(vec!["**.expert*.*".to_string(), "**.experts.*".to_string()])
    }

    pub fn all_linear() -> TargetSelector {
        TargetSelector::All
    }
}

/// `*` matches any run of characters not containing `.`; `**` matches any
/// run including `.`; `?` matches exactly one character; `{a,b,c}` matches
/// any one literal alternative. Anything else must match literally.
/// Iterative (no backtracking blowup) - see the property test in
/// `crates/model/tests/adapter_basics.rs` for a brute-force cross-check.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let pat = expand_braces(pattern);
    pat.iter().any(|p| glob_match_one(p, text))
}

fn expand_braces(pattern: &str) -> Vec<String> {
    if let Some(open) = pattern.find('{') {
        if let Some(close_rel) = pattern[open..].find('}') {
            let close = open + close_rel;
            let prefix = &pattern[..open];
            let suffix = &pattern[close + 1..];
            let mut out = Vec::new();
            for alt in pattern[open + 1..close].split(',') {
                for rest in expand_braces(&format!("{prefix}{alt}{suffix}")) {
                    out.push(rest);
                }
            }
            return out;
        }
    }
    vec![pattern.to_string()]
}

fn glob_match_one(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    is_match(&p, &t)
}

/// Plain recursive matcher: `**` consumes zero or more of ANY character
/// (including `.`), a lone `*` consumes zero or more non-`.` characters,
/// `?` consumes exactly one character. Tensor names and patterns are short
/// (tens of characters, few wildcards) and both come from trusted
/// config/CLI input, so the branching factor here never matters in
/// practice; a DP table would only add complexity for no measurable gain.
fn is_match(p: &[char], t: &[char]) -> bool {
    if p.is_empty() {
        return t.is_empty();
    }
    if p[0] == '*' && p.get(1) == Some(&'*') {
        for k in 0..=t.len() {
            if is_match(&p[2..], &t[k..]) {
                return true;
            }
        }
        return false;
    }
    if p[0] == '*' {
        let run = t.iter().take_while(|&&c| c != '.').count();
        for k in 0..=run {
            if is_match(&p[1..], &t[k..]) {
                return true;
            }
        }
        return false;
    }
    if p[0] == '?' {
        return !t.is_empty() && is_match(&p[1..], &t[1..]);
    }
    !t.is_empty() && p[0] == t[0] && is_match(&p[1..], &t[1..])
}
