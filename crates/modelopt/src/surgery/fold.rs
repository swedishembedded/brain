// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! An embedding folded into the linear map it feeds.

use model::hostmath::linear_rows;
use model::surgery::{Axis, EmbeddingFold};

use super::{check_keep, keep_lines, line_count, Change, Checkpoint, Tensor};

/// Narrow `fold` to output channels `keep` (ascending) of its linear map.
///
/// The map is linear up to its output channels (the nonlinearity comes
/// after them), so the pre-activations of the kept channels are
/// `E W_keepᵀ + b_keep` for every token: the table becomes `E W_keepᵀ`
/// (`[vocab, k]`), the map the `k x k` identity, its bias the kept entries,
/// and every other tensor indexing the channels keeps their lines. The kept
/// channels compute exactly what they did; the dropped ones are gone.
/// Refused, touching nothing: a map that is not square at the table's width,
/// and a dimension whose tensors are not the declared width.
pub fn fold_embedding(ck: &mut Checkpoint, fold: &EmbeddingFold, keep: &[usize]) -> Result<Change, String> {
    let h = fold.hidden.width;
    let what = |e: String| format!("{}: {e}", fold.name);
    check_keep(&fold.name, keep, h)?;
    let e = ck.get(&fold.embedding).map_err(what)?;
    let w = ck.get(&fold.linear).map_err(what)?;
    let (vocab, width) = e.rows_cols();
    if w.shape != [h, h] || width != h {
        return Err(what(format!("the table is {:?} and the map {:?}: the fold needs [_, {h}] and [{h}, {h}]", e.shape, w.shape)));
    }
    for l in &fold.hidden.lines {
        let t = ck.get(&l.tensor).map_err(what)?;
        if line_count(t, l.axis) != h * l.block {
            return Err(what(format!("{} has {} lines along {:?}, not {h} x {}", l.tensor, line_count(t, l.axis), l.axis, l.block)));
        }
    }
    let w_keep: Vec<f32> = keep.iter().flat_map(|&r| w.data[r * h..(r + 1) * h].iter().copied()).collect();
    let r = keep.len();
    let table = Tensor { shape: vec![vocab, r], data: linear_rows(&e.data, &w_keep, vocab, h, r) };
    let mut identity = vec![0f32; r * r];
    (0..r).for_each(|i| identity[i * r + i] = 1.0);
    let mut cut = vec![(fold.embedding.clone(), table), (fold.linear.clone(), Tensor { shape: vec![r, r], data: identity })];
    for l in fold.hidden.lines.iter().filter(|l| l.tensor != fold.linear && l.tensor != fold.embedding) {
        let lines: Vec<usize> = keep.iter().flat_map(|&c| c * l.block..(c + 1) * l.block).collect();
        cut.push((l.tensor.clone(), keep_lines(ck.get(&l.tensor)?, l.axis, &lines).map_err(what)?));
    }
    if fold.bias.as_ref().is_some_and(|b| !fold.hidden.lines.iter().any(|l| &l.tensor == b && l.axis == Axis::Rows)) {
        return Err(what("its bias is not one of the hidden dimension's row lines".into()));
    }
    let (mut change, before) = Change::begin("fold_embedding", ck);
    for (name, t) in cut {
        *ck.get_mut(&name)? = t;
    }
    for key in &fold.hidden.keys {
        change.set(ck, &key.key, (key.per_channel * r).into())?;
    }
    change.targets.push(fold.name.clone());
    change.kept.insert(fold.name.clone(), keep.to_vec());
    Ok(change.finish(ck, before))
}
