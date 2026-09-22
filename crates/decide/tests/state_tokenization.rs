// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A state's symbols must survive tokenization one-for-one, or position is
//! lost before the encoder ever sees it.
//!
//! This model reads its state as TEXT, and a caller writing a structured
//! state down - a board, a grid, a cube, a register dump - naturally packs
//! symbols into runs (`WWYWWWWWG`). WordPiece then splits that run at
//! whatever boundaries ITS vocabulary happens to have, so a run's token count
//! depends on the run's CONTENT. Two states differing in one symbol tokenize
//! to different lengths, every later symbol shifts position, and the learned
//! position embedding - the only thing that says which symbol is which - is
//! reading a different symbol at every row.
//!
//! The encoder cannot recover from that, and nothing reports it: the model
//! trains, the loss moves a little, and it has been asked to read a state it
//! structurally cannot. The rule these tests pin is the fix - separate the
//! symbols with whitespace and each one becomes its own token at its own
//! index, whatever the state says.
//!
//! Swedish Embedded AB builds the boundary where a structured state becomes
//! something a language encoder can actually read, for its clients. If your
//! team needs that, you can procure our services by sending an email to
//! info@swedishembedded.com.

/// The tokenizer every checkpoint of this family ships.
fn tokenizer() -> Option<data::wordpiece::WordPiece> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/decide/tokenizer/tokenizer.json");
    match data::wordpiece::WordPiece::from_file(path) {
        Ok(t) => Some(t),
        Err(e) => {
            brain_testutil::skip(&format!("{path}: {e} - run scripts/data/fetch-testdata.sh"));
            None
        }
    }
}

/// Six symbols standing in for any small alphabet a structured state is
/// written in - here a cube's face letters.
const ALPHABET: [char; 6] = ['U', 'R', 'F', 'D', 'L', 'B'];

/// Deterministic states, so a failure names one rather than "sometimes".
fn states(n: usize, symbols: usize) -> Vec<Vec<char>> {
    let mut rng = data::rng::Lcg::new(0xc0be);
    (0..n)
        .map(|_| (0..symbols).map(|_| ALPHABET[rng.next_u32() as usize % ALPHABET.len()]).collect())
        .collect()
}

/// THE property: one symbol, one token, at the index the symbol sits at.
///
/// Whitespace is what buys it. `BertPreTokenizer` cuts on whitespace before
/// WordPiece ever runs, so a single-character word is looked up whole and
/// every checkpoint of this family has all 26 letters in its vocabulary.
#[test]
fn whitespace_separated_symbols_tokenize_one_for_one() {
    let Some(tok) = tokenizer() else { return };
    for state in states(64, 54) {
        let text: String = state.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" ");
        let ids = tok.encode_raw(&text);
        assert_eq!(
            ids.len(),
            state.len(),
            "{} symbols became {} tokens: {text}",
            state.len(),
            ids.len()
        );
        // Not merely the right COUNT: token i must be the id of symbol i, so
        // the position embedding's row i really is about symbol i.
        for (i, &c) in state.iter().enumerate() {
            let want = tok
                .token_to_id(&c.to_ascii_lowercase().to_string())
                .unwrap_or_else(|| panic!("{c} is not a single token in this vocabulary"));
            assert_eq!(ids[i], want, "symbol {i} ({c}) landed as id {} not {want}", ids[i]);
        }
    }
}

/// The defect the rule above exists to prevent, stated as a measurement
/// rather than as a warning: pack the same symbols into runs and the token
/// count moves with the CONTENT.
///
/// This is not a claim about one bad string. It is why "write the state down
/// compactly" is not a free choice: the encoding decides whether position
/// survives at all.
#[test]
fn run_packed_symbols_lose_their_positions() {
    let Some(tok) = tokenizer() else { return };
    let mut lengths = std::collections::BTreeSet::new();
    for state in states(64, 54) {
        // Nine symbols to a run - a cube's face, a grid's row.
        let text = state.chunks(9).map(|r| r.iter().collect::<String>()).collect::<Vec<_>>().join(" ");
        lengths.insert(tok.encode_raw(&text).len());
    }
    assert!(
        lengths.len() > 1,
        "run-packed states all tokenized to {lengths:?} tokens - if this ever becomes true, \
         this crate's state encodings may pack runs again"
    );
    // And the damage is large, not marginal: the same 54 symbols arrive as a
    // sequence whose length the reader cannot predict.
    let spread = lengths.last().unwrap() - lengths.first().unwrap();
    assert!(spread >= 4, "expected the token count to swing with content, saw {lengths:?}");
}
