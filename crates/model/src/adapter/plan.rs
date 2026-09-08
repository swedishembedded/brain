// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-target rank/alpha (and every other [`super::TargetHp`] field):
//! resolving a default plus a list of selector-scoped overrides into one
//! [`super::TargetHp`] per [`super::LinearSite`], and the one place this
//! workspace parses an encoded `"pattern=value,..."` override string (the
//! CLI/capability surface has no list/map parameter type, so a per-layer
//! rank map has to travel as a string).

use super::select::TargetSelector;
use super::{LinearSite, TargetHp};

/// A sparse override: only the fields present replace the base
/// [`TargetHp`]'s. `None` means "leave whatever the default/earlier
/// override already set."
#[derive(Clone, Copy, Debug, Default)]
pub struct TargetHpPatch {
    pub rank: Option<usize>,
    pub alpha: Option<f32>,
    pub rank_stabilized: Option<bool>,
    pub dropout: Option<f32>,
    pub lr_ratio: Option<f32>,
    pub freeze_a: Option<bool>,
}

impl TargetHpPatch {
    pub fn apply(&self, base: TargetHp) -> TargetHp {
        TargetHp {
            rank: self.rank.unwrap_or(base.rank),
            alpha: self.alpha.unwrap_or(base.alpha),
            rank_stabilized: self.rank_stabilized.unwrap_or(base.rank_stabilized),
            dropout: self.dropout.unwrap_or(base.dropout),
            lr_ratio: self.lr_ratio.unwrap_or(base.lr_ratio),
            freeze_a: self.freeze_a.unwrap_or(base.freeze_a),
        }
    }
}

/// Which linears an adapter targets, plus their hyperparameters: `default`
/// for everything `select` matches, then `overrides` applied in order
/// (later entries win on a field they both touch) to the subset each
/// override's own selector matches.
pub struct AdapterPlan {
    pub select: TargetSelector,
    pub default: TargetHp,
    pub overrides: Vec<(TargetSelector, TargetHpPatch)>,
}

impl AdapterPlan {
    pub fn uniform(select: TargetSelector, default: TargetHp) -> AdapterPlan {
        AdapterPlan { select, default, overrides: Vec::new() }
    }

    /// Resolve every site `select` matches to its final [`TargetHp`]. A
    /// selector (the plan's own, or any override's) that matches NOTHING is
    /// a hard error naming it - the same policy
    /// `model::lora::read_external_adapter` applies to an unrecognised
    /// tensor key: a rank override that silently matches zero targets is a
    /// config a caller believes took effect and didn't.
    pub fn resolve(&self, sites: &[LinearSite]) -> Result<Vec<(LinearSite, TargetHp)>, String> {
        let mut override_hit = vec![false; self.overrides.len()];
        let mut selected_any = false;
        let mut out = Vec::new();
        for site in sites {
            if !self.select.matches(site) {
                continue;
            }
            selected_any = true;
            let mut hp = self.default;
            for (i, (sel, patch)) in self.overrides.iter().enumerate() {
                if sel.matches(site) {
                    hp = patch.apply(hp);
                    override_hit[i] = true;
                }
            }
            out.push((site.clone(), hp));
        }
        if !selected_any {
            return Err("adapter plan: the target selector matched no sites".to_string());
        }
        if let Some(i) = override_hit.iter().position(|hit| !hit) {
            return Err(format!("adapter plan: override #{i} matched no sites"));
        }
        Ok(out)
    }

    /// The largest rank any resolved target uses - what a device trainer
    /// sizes its shared per-site scratch buffers from, since a mixed-rank
    /// plan still shares one buffer sized for the worst case.
    pub fn max_rank(&self, sites: &[LinearSite]) -> Result<usize, String> {
        Ok(self.resolve(sites)?.iter().map(|(_, hp)| hp.rank).max().unwrap_or(0))
    }
}

/// Parse `"pattern=value,pattern=value,..."` into glob-selector/value pairs,
/// in the order given - later entries win where patterns overlap, matching
/// [`AdapterPlan::resolve`]'s own last-match-wins rule. An empty string
/// parses to no overrides (the uniform case). The one place a per-layer
/// rank or alpha map is parsed from a string, since `capability::ParamType`
/// has no list/map variant for it to travel as anything else.
pub fn parse_map(s: &str) -> Result<Vec<(TargetSelector, f32)>, String> {
    if s.trim().is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|entry| {
            let (pat, val) = entry.split_once('=').ok_or_else(|| format!("adapter plan: {entry:?} is not \"pattern=value\""))?;
            let val: f32 = val.trim().parse().map_err(|_| format!("adapter plan: {entry:?} has a non-numeric value"))?;
            Ok((TargetSelector::Glob(vec![pat.trim().to_string()]), val))
        })
        .collect()
}
