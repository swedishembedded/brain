// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that teach speech
// synthesizers new languages and voices for its clients. If your team needs
// expertise in speech-model fine-tuning, you can procure our services by
// sending an email to info@swedishembedded.com.

//! New language and voice rows of the Talker's codec table.
//!
//! A language or a named voice is a row of `talker.model.codec_embedding`
//! that the checkpoint's `config.json` names (`codec_language_id`, `spk_id`).
//! A run names, for each language and voice its shard holds, whether the base
//! checkpoint already has it or it is new. A new one gets a row the base never
//! trained, found by its norm: an untrained row keeps its initial norm (0.022
//! in the 1.7B Base, against 1.0-1.3 for a trained language row), so a row is
//! free when its norm is below [`RowsConfig::free_row_fraction`] of the median
//! norm of the rows the base names. Free rows are taken from the special band
//! only (never a codebook code), never one the configuration names, new
//! languages upward from the base's language rows and new voices downward from
//! the top of the table, where the released checkpoints keep their speakers.
//!
//! A new voice row starts at the robust centroid of that voice's x-vectors
//! (the mean after dropping the share least aligned with their mean direction), so
//! it starts where the x-vector it replaces in the speaker slot would put it.
//! A new language row starts at a weighted mix of base language rows (by
//! default the mean of every language the base names). Training then updates
//! them like any other codec row.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use super::objective::RowMap;
use crate::prompt::TtsSpecials;
use crate::shard::Shard;

/// Whether a shard's language or voice is one the base checkpoint names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowSource {
    /// The base checkpoint names it (`codec_language_id` / `spk_id`).
    Base,
    /// A row this run allocates and trains.
    New,
}

/// Which rows a run's languages and voices take, and how new ones start.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RowsConfig {
    /// Every language of the shard, by its name there. `auto` (base) trains the
    /// no-language prefix.
    pub languages: BTreeMap<String, RowSource>,
    /// Every named voice of the shard.
    pub voices: BTreeMap<String, RowSource>,
    /// A new language row's start as weights over base language names; `None`
    /// is the mean of every language the base names.
    pub language_init: Option<BTreeMap<String, f32>>,
    /// The share of a voice's x-vectors, least aligned with their mean
    /// direction, left out of its centroid.
    pub voice_outlier_fraction: f32,
    /// A row is free when its norm is below this share of the median norm of
    /// the rows the base names.
    pub free_row_fraction: f32,
}

impl Default for RowsConfig {
    fn default() -> Self {
        RowsConfig { languages: BTreeMap::new(), voices: BTreeMap::new(), language_init: None, voice_outlier_fraction: 0.1, free_row_fraction: 0.1 }
    }
}

/// The rows a run resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct Rows {
    /// Per shard language and voice index, its row.
    pub map: RowMap,
    /// Every language and voice by name, base and new: what an exported
    /// checkpoint's `codec_language_id` and `spk_id` name.
    pub languages: BTreeMap<String, u32>,
    pub voices: BTreeMap<String, u32>,
    /// The rows allocated here, which train at their own rate.
    pub new_rows: Vec<u32>,
}

fn norm(row: &[f32]) -> f32 {
    row.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// The centroid of `rows` (each `d` wide) after dropping the `outliers` share
/// least aligned (by cosine) with their mean direction (the mean of the unit
/// vectors, which one long outlier cannot swing as it swings the plain mean).
pub fn robust_centroid(rows: &[&[f32]], d: usize, outliers: f32) -> Vec<f32> {
    let mean = |set: &mut dyn Iterator<Item = Vec<f32>>| -> Vec<f32> {
        let mut m = vec![0.0f32; d];
        let mut n = 0usize;
        for r in set {
            for (a, b) in m.iter_mut().zip(&r) {
                *a += b;
            }
            n += 1;
        }
        m.iter_mut().for_each(|a| *a /= n.max(1) as f32);
        m
    };
    let unit = |r: &[f32]| -> Vec<f32> {
        let n = norm(r).max(f32::MIN_POSITIVE);
        r.iter().map(|x| x / n).collect()
    };
    let direction = mean(&mut rows.iter().map(|r| unit(r)));
    let cosine = |r: &[f32]| unit(r).iter().zip(&direction).map(|(a, b)| a * b).sum::<f32>();
    let mut ranked: Vec<(f32, usize)> = rows.iter().enumerate().map(|(i, r)| (cosine(r), i)).collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let drop = ((rows.len() as f32 * outliers).floor() as usize).min(rows.len() - 1);
    mean(&mut ranked[..rows.len() - drop].iter().map(|&(_, i)| rows[i].to_vec()))
}

/// Resolve every language and voice of `train` to a row of the Talker table
/// `table` (`[vocab, d]`), allocating and initialising the new ones in place.
/// `codebook` is the size of the first codebook: rows below it are codes.
pub fn allocate(cfg: &RowsConfig, sp: &TtsSpecials, codebook: u32, table: &mut [f32], d: usize, train: &Shard) -> Result<Rows, String> {
    if !(0.0..1.0).contains(&cfg.voice_outlier_fraction) || !(cfg.free_row_fraction > 0.0 && cfg.free_row_fraction < 1.0) {
        return Err(format!("rows: voice_outlier_fraction must be in [0, 1) and free_row_fraction in (0, 1), got {} and {}", cfg.voice_outlier_fraction, cfg.free_row_fraction));
    }
    let vocab = (table.len() / d) as u32;
    let meta = &train.meta;
    for (what, named, held) in [("language", &cfg.languages, &meta.languages), ("voice", &cfg.voices, &meta.voices)] {
        if let Some(missing) = held.iter().find(|n| !named.contains_key(*n)) {
            return Err(format!("rows: the shard's {what} {missing:?} is not named in the configuration (base or new)"));
        }
        if let Some(extra) = named.keys().find(|n| !held.contains(n)) {
            return Err(format!("rows: the configuration names the {what} {extra:?}, which the shard does not hold"));
        }
    }
    let codec_specials = [sp.codec_nothink, sp.codec_think, sp.codec_think_bos, sp.codec_think_eos, sp.codec_pad, sp.codec_bos, sp.codec_eos];
    let mut taken: HashSet<u32> = codec_specials.iter().chain(sp.lang.values()).chain(sp.spk_id.values()).copied().collect();
    if let Some(&id) = taken.iter().find(|&&id| id >= vocab) {
        return Err(format!("rows: the checkpoint names row {id}, past its {vocab}-row table"));
    }
    let row = |t: &[f32], id: u32| t[id as usize * d..(id as usize + 1) * d].to_vec();
    let mut named_norms: Vec<f32> = taken.iter().map(|&id| norm(&row(table, id))).collect();
    named_norms.sort_by(f32::total_cmp);
    let reference = named_norms[named_norms.len() / 2];
    let free_below = cfg.free_row_fraction * reference;
    let free = |t: &[f32], id: u32, taken: &HashSet<u32>| !taken.contains(&id) && norm(&row(t, id)) < free_below;

    let mut languages = BTreeMap::new();
    let mut voices = BTreeMap::new();
    let mut new_rows = Vec::new();
    // New languages upward from the base's language band, new voices downward from the top.
    let language_start = sp.lang.values().copied().min().unwrap_or(codebook).max(codebook);
    let mut up = language_start..vocab;
    let mut down = (codebook..vocab).rev();
    for (name, source) in &cfg.languages {
        let id = match source {
            RowSource::Base if name == "auto" => continue,
            RowSource::Base => sp.lang.get(&name.to_lowercase()).copied().ok_or_else(|| format!("rows: the base checkpoint names no language {name:?}"))?,
            RowSource::New => {
                let id = up.find(|&id| free(table, id, &taken)).ok_or_else(|| format!("rows: no free row is left for the new language {name:?}"))?;
                taken.insert(id);
                new_rows.push(id);
                let init = language_row(cfg, sp, table, d)?;
                table[id as usize * d..(id as usize + 1) * d].copy_from_slice(&init);
                id
            }
        };
        languages.insert(name.clone(), id);
    }
    for (name, source) in &cfg.voices {
        let id = match source {
            RowSource::Base => sp.spk_id.get(&name.to_lowercase()).copied().ok_or_else(|| format!("rows: the base checkpoint names no voice {name:?}"))?,
            RowSource::New => {
                let id = down.find(|&id| free(table, id, &taken)).ok_or_else(|| format!("rows: no free row is left for the new voice {name:?}"))?;
                taken.insert(id);
                new_rows.push(id);
                let index = meta.voices.iter().position(|v| v == name).expect("checked against the shard") as u32;
                let xvectors: Vec<&[f32]> = (0..train.len()).map(|r| train.record(r)).filter(|r| r.voice == Some(index)).map(|r| r.xvector).collect();
                if xvectors.is_empty() {
                    return Err(format!("rows: the new voice {name:?} has no record to start its row from"));
                }
                if xvectors[0].len() != d {
                    return Err(format!("rows: x-vectors of {} values, the Talker table is {d} wide", xvectors[0].len()));
                }
                table[id as usize * d..(id as usize + 1) * d].copy_from_slice(&robust_centroid(&xvectors, d, cfg.voice_outlier_fraction));
                id
            }
        };
        voices.insert(name.clone(), id);
    }
    let map = RowMap {
        languages: meta.languages.iter().map(|n| languages.get(n).copied()).collect(),
        voices: meta.voices.iter().map(|n| voices[n]).collect(),
    };
    Ok(Rows { map, languages, voices, new_rows })
}

/// A new language row's start: [`RowsConfig::language_init`]'s weighted mix of
/// base language rows, or their plain mean.
fn language_row(cfg: &RowsConfig, sp: &TtsSpecials, table: &[f32], d: usize) -> Result<Vec<f32>, String> {
    let weights: Vec<(u32, f32)> = match &cfg.language_init {
        Some(mix) => mix.iter().map(|(n, &w)| sp.lang.get(&n.to_lowercase()).map(|&id| (id, w)).ok_or_else(|| format!("rows: language_init names {n:?}, which the base checkpoint does not"))).collect::<Result<_, _>>()?,
        None => sp.lang.values().map(|&id| (id, 1.0)).collect(),
    };
    let total: f32 = weights.iter().map(|(_, w)| w).sum();
    if weights.is_empty() || weights.iter().any(|(_, w)| !(w.is_finite() && *w >= 0.0)) || total <= 0.0 {
        return Err("rows: a new language starts from a mix of base language rows; the base names none, or the weights are not positive".to_string());
    }
    let mut out = vec![0.0f32; d];
    for (id, w) in weights {
        for (o, x) in out.iter_mut().zip(&table[id as usize * d..(id as usize + 1) * d]) {
            *o += w / total * x;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speech_sft::tests::{shard, RECORDS};

    const D: usize = 4;
    const VOCAB: u32 = 48;

    /// A table whose rows are trained (norm ~1) but for 33, 35 and 44..48 (norm 0.02).
    fn table() -> Vec<f32> {
        let mut t = vec![0.0f32; VOCAB as usize * D];
        let untrained: Vec<u32> = [33, 35].into_iter().chain(44..48).collect();
        for id in 0..VOCAB {
            let v = if untrained.contains(&id) { 0.01 } else { 0.5 + 0.01 * id as f32 };
            t[id as usize * D..(id as usize + 1) * D].fill(v);
        }
        t
    }

    fn sp() -> TtsSpecials {
        let mut sp = crate::speech_sft::tests::specials();
        sp.lang.insert("alpha".into(), 30);
        sp.lang.insert("gamma".into(), 32);
        // A named row that never trained is still never taken.
        sp.spk_id.insert("v0".into(), 47);
        sp
    }

    fn cfg(languages: &[(&str, RowSource)], voices: &[(&str, RowSource)]) -> RowsConfig {
        RowsConfig { languages: languages.iter().map(|(n, s)| (n.to_string(), *s)).collect(), voices: voices.iter().map(|(n, s)| (n.to_string(), *s)).collect(), ..RowsConfig::default() }
    }

    /// The shard of the objective's tests, with `D`-wide x-vectors.
    fn train() -> Shard {
        shard(D as u32, &RECORDS, 0x61) // seed: fixed fixture
    }

    #[test]
    fn new_rows_take_only_untrained_rows_of_the_special_band_that_no_one_names() {
        let sp = sp();
        let mut t = table();
        let rows = allocate(&cfg(&[("alpha", RowSource::Base), ("beta", RowSource::New)], &[("v1", RowSource::New)]), &sp, 16, &mut t, D, &train()).unwrap();
        assert_eq!(rows.languages["alpha"], 30, "a base language keeps its row");
        assert_eq!(rows.languages["beta"], 33, "the first free row above the base's language band");
        assert_eq!(rows.voices["v1"], 46, "the first free row from the top that no one names");
        assert_eq!(rows.new_rows, vec![33, 46]);
        assert_eq!(rows.map, RowMap { languages: vec![Some(30), Some(33)], voices: vec![46] });
    }

    #[test]
    fn a_new_language_starts_at_the_mean_of_the_base_languages_or_the_mix_asked_for() {
        let sp = sp();
        let mut t = table();
        allocate(&cfg(&[("alpha", RowSource::Base), ("beta", RowSource::New)], &[("v1", RowSource::New)]), &sp, 16, &mut t, D, &train()).unwrap();
        let mean = (t[30 * D] + t[32 * D]) / 2.0;
        assert!((t[33 * D] - mean).abs() < 1e-6, "{} vs the mean {mean}", t[33 * D]);
        let mut t = table();
        let mix = RowsConfig { language_init: Some([("gamma".to_string(), 3.0), ("alpha".to_string(), 1.0)].into()), ..cfg(&[("alpha", RowSource::Base), ("beta", RowSource::New)], &[("v1", RowSource::New)]) };
        allocate(&mix, &sp, 16, &mut t, D, &train()).unwrap();
        let want = 0.75 * (0.5 + 0.32) + 0.25 * (0.5 + 0.30);
        assert!((t[33 * D] - want).abs() < 1e-6, "{} vs {want}", t[33 * D]);
    }

    #[test]
    fn a_voice_starts_at_the_centroid_of_its_x_vectors_without_the_outlier() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.9f32, 0.1, 0.0, 0.0];
        let c = [1.1f32, -0.1, 0.0, 0.0];
        let outlier = [-5.0f32, 0.0, 5.0, 0.0];
        let rows: Vec<&[f32]> = vec![&a, &b, &outlier, &c];
        let got = robust_centroid(&rows, 4, 0.25);
        assert!((got[0] - 1.0).abs() < 1e-6 && got[1].abs() < 1e-6 && got[2].abs() < 1e-6, "{got:?}");
        let all = robust_centroid(&rows, 4, 0.0);
        assert!((all[2] - 1.25).abs() < 1e-6, "no outlier dropped at a share of 0: {all:?}");
    }

    #[test]
    fn a_configuration_that_does_not_match_the_shard_or_the_checkpoint_is_refused() {
        let sp = sp();
        let refused = |c: RowsConfig| allocate(&c, &sp, 16, &mut table(), D, &train()).unwrap_err();
        assert!(refused(cfg(&[("alpha", RowSource::Base)], &[("v1", RowSource::New)])).contains("\"beta\" is not named"));
        assert!(refused(cfg(&[("alpha", RowSource::Base), ("beta", RowSource::New), ("delta", RowSource::New)], &[("v1", RowSource::New)])).contains("does not hold"));
        assert!(refused(cfg(&[("alpha", RowSource::Base), ("beta", RowSource::Base)], &[("v1", RowSource::New)])).contains("no language \"beta\""));
        assert!(refused(cfg(&[("alpha", RowSource::Base), ("beta", RowSource::New)], &[("v1", RowSource::Base)])).contains("no voice \"v1\""));
        let mut few = table();
        for id in [33, 35, 44, 45, 46] {
            few[id * D..(id + 1) * D].fill(1.0);
        }
        let e = allocate(&cfg(&[("alpha", RowSource::Base), ("beta", RowSource::New)], &[("v1", RowSource::New)]), &sp, 16, &mut few, D, &train()).unwrap_err();
        assert!(e.contains("no free row"), "{e}");
    }
}
