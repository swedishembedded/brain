// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Vocabulary trimming: the rows of kept token ids, renumbered densely.

use model::surgery::{Axis, Vocabulary};

use super::{check_keep, keep_lines, Change, Checkpoint};

/// Keep token ids `keep` (ascending) of `vocab`: id `keep[i]` becomes `i`.
/// Every table keeps those rows, the size key becomes the kept count and
/// every config key holding a token id is renumbered. Refused, touching
/// nothing: a table that is not `size` rows, and a config id that is not kept
/// (a special token the tokenizer would lose).
pub fn trim_vocabulary(ck: &mut Checkpoint, vocab: &Vocabulary, keep: &[usize]) -> Result<Change, String> {
    let what = |e: String| format!("{}: {e}", vocab.name);
    check_keep(&vocab.name, keep, vocab.size)?;
    let mut cut = Vec::new();
    for name in &vocab.tables {
        let t = ck.get(name).map_err(what)?;
        if t.rows_cols().0 != vocab.size {
            return Err(what(format!("{name} has {} rows, the vocabulary {}", t.rows_cols().0, vocab.size)));
        }
        cut.push((name.clone(), keep_lines(t, Axis::Rows, keep)?));
    }
    let mut ids = Vec::new();
    for key in &vocab.ids {
        let old = ck.config_u64(key).map_err(what)? as usize;
        let new = keep.binary_search(&old).map_err(|_| what(format!("{key} is token {old}, which the trimmed vocabulary does not keep")))?;
        ids.push((key, new));
    }
    let (mut change, before) = Change::begin("trim_vocabulary", ck);
    for (name, t) in cut {
        *ck.get_mut(&name)? = t;
    }
    change.set(ck, &vocab.key, keep.len().into())?;
    for (key, new) in ids {
        change.set(ck, key, new.into())?;
    }
    change.targets.push(vocab.name.clone());
    change.kept.insert(vocab.name.clone(), keep.to_vec());
    Ok(change.finish(ck, before))
}
