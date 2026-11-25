// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which ids a byte-level BPE vocabulary keeps for a corpus, and its files
//! rewritten for the trimmed vocabulary (`vocab.json`, `merges.txt` and the
//! added tokens of `tokenizer_config.json`).
//!
//! Kept: the 256 byte tokens (every text still encodes), every added token,
//! every token the corpus encodes to, and the merge closure of those (both
//! parts of the merge that made a kept token, recursively), so every merge
//! the corpus applied is still in `merges.txt` and the corpus encodes to the
//! same tokens under the new ids. A merge survives when its parts and its
//! result are all kept, in its original rank order. Ids are renumbered
//! densely in their original order: regular tokens first, then the added
//! ones, as the original numbers them.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use serde::Serialize;
use serde_json::Value;

/// A byte-level BPE tokenizer's split files, as text and JSON.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenizerFiles {
    pub vocab_json: String,
    pub merges_txt: String,
    pub tokenizer_config: Value,
}

/// What a trim kept and what it changed.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BpeReport {
    /// Ids kept, of the ids the vocabulary had.
    pub kept: usize,
    pub of: usize,
    /// Kept for being a byte token, an added token, a corpus token, and only for the merge closure.
    pub bytes: usize,
    pub added: usize,
    pub corpus: usize,
    pub closure: usize,
    pub merges_kept: usize,
    pub merges_of: usize,
    /// Corpus texts, and how many encode to other tokens after the trim (0 by construction, measured).
    pub corpus_texts: usize,
    pub corpus_changed: usize,
    /// Texts outside the corpus, and how many encode to other tokens after the trim.
    pub probe_texts: usize,
    pub probe_changed: usize,
}

/// A trimmed vocabulary: the kept original ids (ascending; new id = index),
/// the rewritten files and the report.
#[derive(Clone, Debug)]
pub struct BpeTrim {
    pub keep: Vec<usize>,
    pub files: TokenizerFiles,
    pub report: BpeReport,
}

/// The added tokens of a `tokenizer_config.json`, as `id -> record`.
fn added_tokens(config: &Value) -> Result<BTreeMap<usize, Value>, String> {
    let Some(decoder) = config.get("added_tokens_decoder") else { return Ok(BTreeMap::new()) };
    let decoder = decoder.as_object().ok_or("tokenizer_config.json: added_tokens_decoder is not an object")?;
    decoder.iter().map(|(id, rec)| Ok((id.parse::<usize>().map_err(|_| format!("tokenizer_config.json: added token id {id:?} is not a number"))?, rec.clone()))).collect()
}

fn records(added: &BTreeMap<usize, Value>) -> Result<Value, String> {
    let mut out = Vec::with_capacity(added.len());
    for (id, rec) in added {
        let content = rec.get("content").and_then(Value::as_str).ok_or_else(|| format!("tokenizer_config.json: added token {id} has no content"))?;
        out.push(serde_json::json!({"content": content, "id": id}));
    }
    Ok(Value::Array(out))
}

fn tokenizer(vocab: &Value, merges: &str, added: &BTreeMap<usize, Value>) -> Result<QwenBpe, String> {
    QwenBpe::from_split_files(vocab, merges, records(added)?)
}

/// Trim `files` to what `corpus` needs; `probe` are texts outside the corpus
/// whose encodings the report compares before and after. Refused: files that
/// do not parse, a vocabulary whose ids are not dense, a missing byte token.
pub fn trim(files: &TokenizerFiles, corpus: &[&str], probe: &[&str]) -> Result<BpeTrim, String> {
    let vocab: BTreeMap<String, usize> = serde_json::from_str(&files.vocab_json).map_err(|e| format!("vocab.json: {e}"))?;
    let added = added_tokens(&files.tokenizer_config)?;
    let mut by_id: BTreeMap<usize, &str> = BTreeMap::new();
    for (tok, &id) in &vocab {
        if by_id.insert(id, tok).is_some() {
            return Err(format!("vocab.json: id {id} is given twice"));
        }
    }
    let regular = vocab.len();
    if by_id.keys().next_back().is_some_and(|&m| m + 1 != regular) {
        return Err("vocab.json: its ids are not 0..n".into());
    }
    if let Some(id) = added.keys().find(|&&id| id < regular) {
        return Err(format!("tokenizer_config.json: added token {id} collides with a vocab.json id"));
    }
    let total = regular + added.len();
    let original = tokenizer(&serde_json::to_value(&vocab).map_err(|e| e.to_string())?, &files.merges_txt, &added)?;

    // The merges, by rank: (left, right, result).
    let header: Vec<&str> = files.merges_txt.lines().take_while(|l| l.starts_with("#version")).collect();
    let merges: Vec<(&str, &str)> = files.merges_txt.lines().skip(header.len()).filter(|l| !l.is_empty()).map(|l| l.split_once(' ').ok_or_else(|| format!("merges.txt: {l:?} is not a pair"))).collect::<Result<_, _>>()?;
    let mut made_by: HashMap<String, (&str, &str)> = HashMap::with_capacity(merges.len());
    for &(a, b) in &merges {
        made_by.entry(format!("{a}{b}")).or_insert((a, b));
    }

    let mut keep: BTreeSet<usize> = BTreeSet::new();
    let mut bytes = 0;
    for c in data::bpe::bytes_to_unicode() {
        let id = *vocab.get(c.to_string().as_str()).ok_or_else(|| format!("vocab.json has no byte token {c:?}"))?;
        bytes += usize::from(keep.insert(id));
    }
    let n_added = added.keys().filter(|&&id| keep.insert(id)).count();
    let encoded: Vec<Vec<u32>> = corpus.iter().map(|t| original.encode(t)).collect();
    let mut from_corpus = 0;
    for &id in encoded.iter().flatten() {
        if id as usize >= total {
            return Err(format!("the tokenizer encoded id {id}, past the vocabulary of {total}"));
        }
        from_corpus += usize::from(keep.insert(id as usize));
    }
    // The merge closure of every kept regular token.
    let mut stack: Vec<usize> = keep.iter().copied().filter(|&id| id < regular).collect();
    let mut closure = 0;
    while let Some(id) = stack.pop() {
        if let Some(&(a, b)) = made_by.get(by_id[&id]) {
            for part in [a, b] {
                let pid = *vocab.get(part).ok_or_else(|| format!("merges.txt: part {part:?} is not in vocab.json"))?;
                if keep.insert(pid) {
                    closure += 1;
                    stack.push(pid);
                }
            }
        }
    }

    let keep: Vec<usize> = keep.into_iter().collect();
    let new_id: HashMap<usize, usize> = keep.iter().enumerate().map(|(new, &old)| (old, new)).collect();
    let new_vocab: BTreeMap<&str, usize> = keep.iter().filter(|&&id| id < regular).map(|&id| (by_id[&id], new_id[&id])).collect();
    let kept_tok = |t: &str| vocab.get(t).is_some_and(|id| new_id.contains_key(id));
    let kept_merges: Vec<String> = merges.iter().filter(|(a, b)| kept_tok(a) && kept_tok(b) && kept_tok(&format!("{a}{b}"))).map(|(a, b)| format!("{a} {b}")).collect();
    let mut merges_txt: String = header.iter().map(|h| format!("{h}\n")).collect();
    for m in &kept_merges {
        merges_txt.push_str(m);
        merges_txt.push('\n');
    }
    let new_added: BTreeMap<usize, Value> = added.iter().map(|(id, rec)| (new_id[id], rec.clone())).collect();
    let mut tokenizer_config = files.tokenizer_config.clone();
    if tokenizer_config.get("added_tokens_decoder").is_some() {
        tokenizer_config["added_tokens_decoder"] = Value::Object(new_added.iter().map(|(id, rec)| (id.to_string(), rec.clone())).collect());
    }
    let vocab_json = serde_json::to_string(&new_vocab).map_err(|e| e.to_string())?;

    let trimmed = tokenizer(&serde_json::to_value(&new_vocab).map_err(|e| e.to_string())?, &merges_txt, &new_added)?;
    let old_ids = |ids: &[u32]| -> Vec<Option<usize>> { ids.iter().map(|&i| new_id.get(&(i as usize)).copied()).collect() };
    let changed = |text: &str, before: &[u32]| old_ids(before) != trimmed.encode(text).iter().map(|&i| Some(i as usize)).collect::<Vec<_>>();
    let corpus_changed = corpus.iter().zip(&encoded).filter(|(t, e)| changed(t, e)).count();
    let probe_changed = probe.iter().filter(|t| changed(t, &original.encode(t))).count();
    let report = BpeReport {
        kept: keep.len(),
        of: total,
        bytes,
        added: n_added,
        corpus: from_corpus,
        closure,
        merges_kept: kept_merges.len(),
        merges_of: merges.len(),
        corpus_texts: corpus.len(),
        corpus_changed,
        probe_texts: probe.len(),
        probe_changed,
    };
    Ok(BpeTrim { keep, files: TokenizerFiles { vocab_json, merges_txt, tokenizer_config }, report })
}
