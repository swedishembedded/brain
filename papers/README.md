<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Papers

`make papers/<name>/pdf` (or `make papers/pdf` for all) builds a paper into
`build/papers/<name>.pdf`.

Each paper has one question, stated as its scope in the introduction. A new
verified finding goes into the paper whose scope it falls in, in the same change
that produces it (`.agents/rules/papers.md`). A paper whose `status.txt` says
`draft` may hold registered measurements whose results are not yet in; a
finding that fits no scope below starts a new paper directory with its own
scope.

| Directory | Scope: a finding belongs here if it changes what is claimed about, or how one would measure... |
|---|---|
| `spectral-lm/` | language models whose hidden state is a truncated real Fourier spectrum and whose feed-forward mixing is done by per-mode filters around a signal-domain nonlinearity: the representation, the operators and their gradients, the training and decoding path, the spectral structure of a conventional model's hidden states, and the matched comparisons against conventional transformers |
