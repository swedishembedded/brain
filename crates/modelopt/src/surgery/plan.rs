// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A surgery plan: edits named by what they act on and how much they keep,
//! chosen by the calibration of the checkpoint they start from.
//!
//! ```json
//! [{"trim_vocabulary": {"vocabulary": "text"}},
//!  {"fold_embedding": {"fold": "talker.text", "keep": 1024, "score": "wanda"}},
//!  {"select_channels": {"dims": "talker.layers.*.mlp", "keep": 4608, "score": "wanda"}},
//!  {"select_channels": {"dims": "talker.layers.*.kv_heads", "keep": 6, "score": "taylor"}},
//!  {"rotate_stream": {"stream": "talker", "width": 1536, "basis": "pca"}},
//!  {"remove_layers": {"stack": "talker.layers", "count": 8, "by": "kl"}}]
//! ```
//!
//! Every edit is chosen from one calibration, so a plan is applied in a
//! fixed order whatever order it lists its edits in: vocabulary trims,
//! folds, channel selections, stream rotations, then layer removals (the
//! removal renumbers the layers the other edits name).

use std::collections::BTreeMap;

use model::surgery::{Calibration, SurgeryGraph};
use serde::{Deserialize, Serialize};

use super::scores::{act_rms, wanda, Score};
use super::{basis_from_covariance, basis_within_groups, gain_groups, top_k, Basis, BasisKind, Change, Checkpoint, SharedNormFit};

/// What a layer removal ranks layers by (the least costly go).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerScore {
    /// The KL divergence from the unedited predictions without the layer.
    Kl,
    /// The loss without the layer.
    Loss,
    /// How much the layer changes the stream (`1 - cosine` of its input and output).
    Cosine,
}

/// One edit of a plan. Unknown fields are refused.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Edit {
    /// Keep the token ids a corpus needs (worked out by the caller, [`super::bpe`]).
    TrimVocabulary { vocabulary: String },
    /// Narrow a fold's hidden channels to `keep`, ranked by `score`.
    FoldEmbedding { fold: String, keep: usize, score: Score },
    /// Keep `keep` channels of every dimension matching `dims`, ranked by `score`.
    SelectChannels { dims: String, keep: usize, score: Score },
    /// Rotate a stream into `basis` and keep `width` directions. On a stream
    /// with a shared norm a PCA basis mixes only channels whose shared gain
    /// is equal to within `gain_tolerance` (relative; default 0, which keeps
    /// the rotation exact at full width): a wider tolerance mixes more and
    /// approximates the shared norm.
    RotateStream {
        stream: String,
        width: usize,
        basis: BasisKind,
        #[serde(default)]
        gain_tolerance: Option<f64>,
    },
    /// Remove `count` layers of a stack ranked by `by`, or exactly `layers`.
    /// With `greedy` (`by` kl or loss) the layers are chosen one at a time,
    /// each measured with those already chosen removed
    /// ([`super::greedy_removal`]); the caller, which can measure,
    /// resolves such an edit into its `layers` before [`apply`].
    RemoveLayers {
        stack: String,
        #[serde(default)]
        count: Option<usize>,
        #[serde(default)]
        by: Option<LayerScore>,
        #[serde(default)]
        layers: Option<Vec<usize>>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        greedy: bool,
    },
}

impl Edit {
    /// The order a plan applies its edits in (lowest first).
    pub fn rank(&self) -> u8 {
        match self {
            Edit::TrimVocabulary { .. } => 0,
            Edit::FoldEmbedding { .. } => 1,
            Edit::SelectChannels { .. } => 2,
            Edit::RotateStream { .. } => 3,
            Edit::RemoveLayers { .. } => 4,
        }
    }
}

/// How the inputs a checkpoint is fed from outside change with an edit: the
/// token ids a vocabulary keeps (old id at the new id's index), the basis a
/// stream was rotated into and its kept width.
#[derive(Clone, Debug, Default)]
pub struct Inputs {
    pub vocabularies: BTreeMap<String, Vec<usize>>,
    pub streams: BTreeMap<String, (Basis, usize)>,
}

/// What a plan did: the edited checkpoint's changes in order, each with the
/// method that chose what it kept, and how its inputs moved.
pub struct Applied {
    pub changes: Vec<(Change, String)>,
    pub inputs: Inputs,
}

/// The model's side of a plan: its graph for a checkpoint, and its check that
/// a config is one its upstream code builds as the tensors say.
pub struct Model<'a> {
    pub graph: &'a dyn Fn(&Checkpoint) -> Result<SurgeryGraph, String>,
    pub check: &'a dyn Fn(&Checkpoint) -> Result<(), String>,
}

/// The scores of `dim` by `score`, from the calibration.
fn channel_scores(ck: &Checkpoint, cal: &Calibration, dim: &model::surgery::ChannelDim, score: Score) -> Result<Vec<f64>, String> {
    let missing = |what: &str| format!("{}: the calibration has no {what} for it", dim.name);
    let scores = match score {
        Score::ActRms | Score::Wanda => {
            let stats = cal.channels.get(&dim.name).ok_or_else(|| missing("activation statistics"))?;
            if score == Score::ActRms {
                act_rms(&stats.sum_sq, stats.rows)
            } else {
                let reader = dim.masks.first().ok_or_else(|| format!("{}: no tensor reads its channels", dim.name))?;
                let lines = dim.lines.iter().find(|l| &l.tensor == reader).ok_or_else(|| format!("{}: its mask is not one of its lines", dim.name))?;
                wanda(ck, &stats.sum_sq, reader, lines.axis, lines.block)?
            }
        }
        Score::Fisher => cal.curvature.get(&dim.name).ok_or_else(|| missing("curvature"))?.fisher.clone(),
        Score::Taylor => cal.curvature.get(&dim.name).ok_or_else(|| missing("curvature"))?.taylor.clone(),
    };
    if scores.len() != dim.width {
        return Err(format!("{}: the calibration scores {} channels, the dimension has {}", dim.name, scores.len(), dim.width));
    }
    Ok(scores)
}

/// The `count` layers of `stack` whose removal costs least by `by`.
fn cheapest_layers(cal: &Calibration, stack: &str, count: usize, by: LayerScore) -> Result<Vec<usize>, String> {
    let layers = cal.layers.get(stack).ok_or_else(|| format!("{stack}: the calibration ablated no layer of it"))?;
    let cost: Vec<f64> = layers
        .iter()
        .enumerate()
        .map(|(i, a)| match by {
            LayerScore::Kl => a.kl.ok_or_else(|| format!("{stack}: layer {i} was not ablated")),
            LayerScore::Loss => a.loss.ok_or_else(|| format!("{stack}: layer {i} was not ablated")),
            LayerScore::Cosine => Ok(1.0 - a.cosine),
        })
        .collect::<Result<_, _>>()?;
    // The cheapest are the top scores of the negated cost.
    let negated: Vec<f64> = cost.iter().map(|c| -c).collect();
    Ok(top_k(&negated, count))
}

/// Apply `plan` to `ck`, every choice made from `cal` (the calibration of
/// `ck` as it is now) and `vocab_keep` (per vocabulary, the ids a corpus
/// needs). The model's check runs after every edit; a refused edit leaves
/// the edits before it applied and answers the error.
pub fn apply(ck: &mut Checkpoint, plan: &[Edit], cal: &Calibration, vocab_keep: &BTreeMap<String, Vec<usize>>, m: &Model<'_>) -> Result<Applied, String> {
    let mut order: Vec<&Edit> = plan.iter().collect();
    order.sort_by_key(|e| e.rank());
    let mut applied = Applied { changes: Vec::new(), inputs: Inputs::default() };
    for edit in order {
        let g = (m.graph)(ck)?;
        let (change, method) = match edit {
            Edit::TrimVocabulary { vocabulary } => {
                let keep = vocab_keep.get(vocabulary).ok_or_else(|| format!("{vocabulary}: no corpus was given to trim it to"))?;
                let change = super::trim_vocabulary(ck, g.vocabulary(vocabulary)?, keep)?;
                applied.inputs.vocabularies.insert(vocabulary.clone(), keep.clone());
                (change, "corpus".to_string())
            }
            Edit::FoldEmbedding { fold, keep, score } => {
                let f = g.fold(fold)?;
                let scores = channel_scores(ck, cal, &f.hidden, *score)?;
                (super::fold_embedding(ck, f, &top_k(&scores, *keep))?, format!("{score:?}").to_lowercase())
            }
            Edit::SelectChannels { dims, keep, score } => {
                let matched = g.dims_matching(dims);
                if matched.is_empty() {
                    return Err(format!("no dimension matches {dims}"));
                }
                let kept: Vec<Vec<usize>> = matched.iter().map(|d| channel_scores(ck, cal, d, *score).map(|s| top_k(&s, *keep))).collect::<Result<_, _>>()?;
                (super::select_channels(ck, &matched, &kept)?, format!("{score:?}").to_lowercase())
            }
            Edit::RotateStream { stream, width, basis, gain_tolerance } => {
                let s = g.stream(stream)?;
                let stats = cal.streams.get(stream).ok_or_else(|| format!("{stream}: the calibration has no statistics of it"))?;
                if stats.width != s.width {
                    return Err(format!("{stream}: calibrated {} wide, the checkpoint is {}", stats.width, s.width));
                }
                let shared: Vec<&str> = s.norms.iter().filter(|n| n.shared).map(|n| n.tensor.as_str()).collect();
                let b = match (basis, shared.as_slice()) {
                    (BasisKind::Pca, [norm]) => basis_within_groups(&stats.covariance, s.width, *basis, &gain_groups(&ck.get(norm)?.data, gain_tolerance.unwrap_or(0.0)))?,
                    (BasisKind::Pca, [_, _, ..]) => return Err(format!("{stream}: a PCA basis keeps one shared norm exact, the stream has {}", shared.len())),
                    _ => basis_from_covariance(&stats.covariance, s.width, *basis)?,
                };
                let fits: Vec<SharedNormFit> = stats.shared.iter().map(|(norm, c)| SharedNormFit { norm: norm.clone(), covariance: c.clone() }).collect();
                let change = super::rotate_stream(ck, s, &b, *width, &fits)?;
                applied.inputs.streams.insert(stream.clone(), (b, *width));
                (change, format!("{basis:?}").to_lowercase())
            }
            Edit::RemoveLayers { stack, count, by, layers, greedy } => {
                let st = g.stack(stack)?;
                if *greedy {
                    return Err(format!("remove_layers {stack}: a greedy choice is measured on the model; resolve it into its layers first"));
                }
                let (remove, method) = match (count, by, layers) {
                    (None, None, Some(l)) => (l.clone(), "given".to_string()),
                    (Some(n), Some(by), None) => (cheapest_layers(cal, stack, *n, *by)?, format!("ablation_{by:?}").to_lowercase()),
                    _ => return Err(format!("remove_layers {stack}: give either count and by, or layers")),
                };
                (super::remove_layers(ck, st, &remove)?, method)
            }
        };
        (m.check)(ck)?;
        applied.changes.push((change, method));
    }
    Ok(applied)
}
