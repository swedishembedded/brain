// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! BANKING77: 13,083 customer-service utterances over 77 fine-grained banking
//! intents, the decision model's first training and evaluation set.
//!
//! Casanueva, Temcinas, Gerz, Henderson and Vulic, *Efficient Intent Detection
//! with Dual Sentence Encoders*, NLP4ConvAI @ ACL 2020. CC-BY-4.0, taken from
//! the authors' own repository; `make fetch/testdata` puts it in place.
//!
//! **Options are sampled, not fixed.** A model trained against all 77 intents
//! every time learns a 77-way head wearing a costume: nothing forces it to
//! read the option text, because position alone identifies the answer. So each
//! example sees a random SUBSET of the intents, always including the correct
//! one. That is what makes the option space genuinely runtime-defined, and it
//! is what makes the held-out-intent evaluation mean anything.
//!
//! It also costs something, and the cost is worth stating plainly: a model
//! trained this way will usually score below one trained on the fixed 77-way
//! task, because the fixed-task model never has to score an option it has not
//! met. The held-out number is the one that measures what this model is for.

use std::collections::BTreeSet;
use std::path::Path;

use data::rng::Rng;

/// One labelled utterance.
#[derive(Clone, Debug)]
pub struct Row {
    pub text: String,
    /// Index into [`Banking77::categories`].
    pub label: usize,
}

pub struct Banking77 {
    /// The 77 intent names, in the dataset's own order.
    pub categories: Vec<String>,
    pub train: Vec<Row>,
    pub test: Vec<Row>,
}

/// `card_arrival` -> `card arrival`.
///
/// The released category names are the only option text there is: this dataset
/// ships no descriptions, so what the model reads for an option is its
/// humanized name and nothing else.
pub fn humanize(category: &str) -> String {
    category.replace('_', " ")
}

/// Split one RFC4180 line into fields.
///
/// Worth the 30 lines rather than a `split(',')`: 1,341 of the training
/// utterances contain a comma inside a quoted field ("How do I know if I will
/// get my card, or if it is lost?"), and splitting naively would truncate
/// every one of them into a shorter sentence with a bogus label.
fn csv_fields(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted => {
                // A doubled quote inside a quoted field is one literal quote.
                if chars.peek() == Some(&'"') {
                    cur.push('"');
                    chars.next();
                } else {
                    quoted = false;
                }
            }
            '"' => quoted = true,
            ',' if !quoted => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

/// Read one `text,category` CSV, skipping its header.
///
/// A quoted field may contain newlines in RFC4180. This reader joins physical
/// lines until the quotes balance rather than assuming one record per line,
/// because a file that ever gains such a row would otherwise load with a
/// silently wrong row count.
fn read_csv(path: &Path, categories: &[String]) -> Result<Vec<Row>, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut rows = Vec::new();
    let mut pending = String::new();
    for (n, line) in raw.lines().enumerate() {
        if n == 0 && line.starts_with("text,") {
            continue;
        }
        if pending.is_empty() {
            pending.push_str(line);
        } else {
            pending.push('\n');
            pending.push_str(line);
        }
        if pending.chars().filter(|&c| c == '"').count() % 2 != 0 {
            continue; // an unbalanced quote means the record continues
        }
        let record = std::mem::take(&mut pending);
        if record.trim().is_empty() {
            continue;
        }
        let f = csv_fields(&record);
        if f.len() < 2 {
            return Err(format!("{}:{}: expected `text,category`, got {f:?}", path.display(), n + 1));
        }
        let cat = f[f.len() - 1].trim();
        let label = categories
            .iter()
            .position(|c| c == cat)
            .ok_or_else(|| format!("{}:{}: unknown category {cat:?}", path.display(), n + 1))?;
        rows.push(Row { text: f[..f.len() - 1].join(","), label });
    }
    Ok(rows)
}

impl Banking77 {
    /// Load from a directory holding `categories.json`, `train.csv` and
    /// `test.csv`.
    pub fn load(dir: &Path) -> Result<Banking77, String> {
        let cats = std::fs::read_to_string(dir.join("categories.json"))
            .map_err(|e| format!("read categories.json: {e}"))?;
        let categories: Vec<String> =
            serde_json::from_str(&cats).map_err(|e| format!("categories.json: {e}"))?;
        let train = read_csv(&dir.join("train.csv"), &categories)?;
        let test = read_csv(&dir.join("test.csv"), &categories)?;
        Ok(Banking77 { categories, train, test })
    }

    /// The option text an intent is scored as.
    pub fn option_text(&self, label: usize) -> String {
        humanize(&self.categories[label])
    }
}

/// Which intents a run is allowed to train on, and which are kept back.
#[derive(Clone, Debug)]
pub struct IntentSplit {
    pub seen: Vec<usize>,
    pub unseen: Vec<usize>,
}

impl IntentSplit {
    /// Hold `n_unseen` intents back, chosen by a fixed seed so the split is
    /// reproducible and reportable rather than whatever the run happened to
    /// draw.
    pub fn holdout(n_categories: usize, n_unseen: usize, seed: u64) -> IntentSplit {
        assert!(n_unseen < n_categories, "cannot hold back every intent");
        let mut rng = Rng::new(seed);
        let mut unseen = BTreeSet::new();
        while unseen.len() < n_unseen {
            unseen.insert((rng.next_u64() % n_categories as u64) as usize);
        }
        let seen = (0..n_categories).filter(|i| !unseen.contains(i)).collect();
        IntentSplit { seen, unseen: unseen.into_iter().collect() }
    }

    /// Every intent, i.e. the classic fixed-label task.
    pub fn all(n_categories: usize) -> IntentSplit {
        IntentSplit { seen: (0..n_categories).collect(), unseen: Vec::new() }
    }

    pub fn contains_seen(&self, label: usize) -> bool {
        self.seen.binary_search(&label).is_ok()
    }
}

/// Draws the option set one example is scored against.
#[derive(Clone, Copy, Debug)]
pub struct OptionSampler {
    /// Fewest options in a drawn set, including the correct one.
    pub min: usize,
    /// Most options in a drawn set.
    pub max: usize,
}

impl Default for OptionSampler {
    fn default() -> OptionSampler {
        OptionSampler { min: 2, max: 32 }
    }
}

impl OptionSampler {
    /// Draw an option set containing `gold` plus distractors from `pool`.
    ///
    /// Returns the intent ids in the order they will be presented, and where
    /// the correct one landed. The gold option is placed at a RANDOM position,
    /// not first: a model scored against a set whose answer is always at index
    /// zero can learn the position instead of the text, which is the exact
    /// failure this sampling exists to prevent.
    pub fn draw(&self, gold: usize, pool: &[usize], rng: &mut Rng) -> (Vec<usize>, usize) {
        let available: Vec<usize> = pool.iter().copied().filter(|&l| l != gold).collect();
        let hi = self.max.min(available.len() + 1);
        let lo = self.min.min(hi);
        let k = if hi > lo { lo + (rng.next_u64() % (hi - lo + 1) as u64) as usize } else { lo };
        let n_distract = k.saturating_sub(1).min(available.len());

        // Partial Fisher-Yates over a copy: sampling WITHOUT replacement, so a
        // set never scores the same intent twice.
        let mut shuffled = available;
        for i in 0..n_distract {
            let j = i + (rng.next_u64() % (shuffled.len() - i) as u64) as usize;
            shuffled.swap(i, j);
        }
        let mut options: Vec<usize> = shuffled[..n_distract].to_vec();
        let at = (rng.next_u64() % (options.len() + 1) as u64) as usize;
        options.insert(at, gold);
        (options, at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_fields_with_commas_survive() {
        let f = csv_fields("\"How do I know if I will get my card, or if it is lost?\",card_arrival");
        assert_eq!(f.len(), 2);
        assert_eq!(f[0], "How do I know if I will get my card, or if it is lost?");
        assert_eq!(f[1], "card_arrival");
    }

    #[test]
    fn a_doubled_quote_is_one_literal_quote() {
        let f = csv_fields("\"he said \"\"hi\"\" once\",x");
        assert_eq!(f[0], "he said \"hi\" once");
    }

    #[test]
    fn an_unquoted_row_still_parses() {
        let f = csv_fields("I am still waiting on my card?,card_arrival");
        assert_eq!(f, vec!["I am still waiting on my card?", "card_arrival"]);
    }

    #[test]
    fn holdout_is_disjoint_complete_and_reproducible() {
        let a = IntentSplit::holdout(77, 20, 7);
        let b = IntentSplit::holdout(77, 20, 7);
        assert_eq!(a.unseen, b.unseen, "the split must be reproducible from its seed");
        assert_eq!(a.unseen.len(), 20);
        assert_eq!(a.seen.len(), 57);
        assert!(a.seen.iter().all(|s| !a.unseen.contains(s)));
        let mut all: Vec<usize> = a.seen.iter().chain(&a.unseen).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..77).collect::<Vec<_>>());
    }

    #[test]
    fn a_drawn_option_set_holds_the_gold_once_and_no_duplicates() {
        let pool: Vec<usize> = (0..77).collect();
        let mut rng = Rng::new(3);
        let s = OptionSampler::default();
        let mut positions = BTreeSet::new();
        for _ in 0..200 {
            let (opts, at) = s.draw(11, &pool, &mut rng);
            assert_eq!(opts[at], 11, "`at` must point at the gold option");
            assert_eq!(opts.iter().filter(|&&o| o == 11).count(), 1, "gold appears once");
            let uniq: BTreeSet<usize> = opts.iter().copied().collect();
            assert_eq!(uniq.len(), opts.len(), "an option set must not repeat an intent");
            assert!((s.min..=s.max).contains(&opts.len()), "drew {} options", opts.len());
            positions.insert(at);
        }
        // The gold lands in many different positions, which is the property
        // that stops the model learning an index instead of the text.
        assert!(positions.len() > 5, "gold only ever appeared at {positions:?}");
    }

    /// A pool smaller than `min` must still produce a legal set rather than
    /// asking for distractors that do not exist.
    #[test]
    fn a_tiny_pool_is_handled() {
        let mut rng = Rng::new(1);
        let (opts, at) = OptionSampler::default().draw(4, &[4, 9], &mut rng);
        assert_eq!(opts.len(), 2);
        assert_eq!(opts[at], 4);
    }
}
