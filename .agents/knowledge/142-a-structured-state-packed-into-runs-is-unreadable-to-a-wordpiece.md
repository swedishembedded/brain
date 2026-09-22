<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 142. A structured state packed into runs is unreadable to a WordPiece encoder

A caller writing a board, grid, cube or register dump into a decision model's
state naturally packs symbols into runs (`WWYWWWWWG`). WordPiece cuts that run
at whatever boundaries its vocabulary has, so the token COUNT moves with the
CONTENT: `WWWWWWWWW` is one token and `WWYWWWWWG` is seven. Every symbol after
the first difference shifts row, and the learned position embedding - the only
thing that says which symbol is which - is then reading a different symbol at
every row for every state.

Separating the symbols with whitespace fixes it structurally: the
pre-tokenizer cuts on whitespace before WordPiece runs and a lone letter is in
every vocabulary of this family, so the stream is one token per symbol at a
fixed index for every state. `crates/decide/tests/state_tokenization.rs` holds
both halves.

HONEST SCOPE, because this is the more useful half of the lesson. Fixing this
did NOT move the rubiks sample's accuracy: at equal budget the run-packed and
whitespace-separated encodings scored 84% and 85%, inside the noise of a
200-item eval. The run-packed encoding's merged tokens are themselves informative
local-run features, and this model was limited by #141, not by its input. The
encoding was diagnosed first BECAUSE it was a real defect with a clean
mechanism, and a real defect with a clean mechanism is exactly what a plausible
root cause looks like when it is not the one that matters.

THE RULE. A defect you can prove from the code is still a hypothesis about
what is LIMITING the system, and the two are different claims needing
different evidence. Prove the mechanism with a test; prove it was the cause
with an A/B at a fixed budget, before writing it down as the fix.
