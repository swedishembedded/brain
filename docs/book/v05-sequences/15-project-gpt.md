<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Project: Your Own Small Language Model

*Can you train a small GPT from raw text and generate from it with a KV cache that provably matches uncached generation?*

A language model that writes sentences has proved less than it seems to. A sampled text reads like text whether the cache that produced it was right or subtly wrong, and a loss that falls is as happy with a leak in the data as with learning. Fourteen chapters have taken a string of bytes to a model that talks, and each chapter checked its own part. What none of them could check is that the parts fit: that the tokenizer's ids are the ids the model's table expects, that the batch the trainer reads is the batch the pipeline cut, that the cache the server fills is the cache the trainer's forward pass would have computed. This chapter asks you to build the whole thing once, in a language of your choice, and to check every joint. You will write a small language model in seven parts:

1. **A tokenizer.** A byte-level byte-pair encoder (Chapter 2) trained on a corpus that a printed rule and a printed seed generate, with a decoder that gives the original text back for every input, including text the corpus never contained.
2. **A data pipeline.** The token stream with a separator before every document, a split by document, rows packed from the stream (Chapter 10) and batches taken in a seeded order, so that the first batch can be printed and compared.
3. **A GPT.** Pre-norm blocks (Chapter 9), causal attention (Chapter 7), learned absolute positions (Chapter 8) and a head tied to the embedding table, with its forward pass compared with printed logits for a printed initialization and its backward pass checked against central differences (Volume III, Chapter 6).
4. **Training.** AdamW, a warmup followed by a cosine, a clip on the global gradient norm, a validation loss, and checkpoints that can be written to bytes and resumed exactly (Chapter 10, Volume III, Chapters 7 and 8).
5. **Generation.** Greedy decoding, temperature, top-k and top-p with a sampler that is seeded and consumes its random numbers in a stated order (Chapter 11).
6. **A KV cache.** Prefill and decode (Chapter 12), tested against uncached generation over every split of a text, with a tolerance that is counted from the shapes, together with the off-by-one in the position of the new token that the test must catch.
7. **A work report.** FLOPs per token for training and for decoding, cache bytes per token and the arithmetic intensity of a decode step on the roofline of Volume IV.

Nothing in the list is new; every part was derived and worked by hand in an earlier chapter, and the project asks whether the parts compose. Two words are used from here on. A **digest** is the first 16 hexadecimal digits of the SHA-256 of the bytes of a result, with the layout of those bytes stated beside it; one different byte changes it, so it compares a table too long to print. A **mutation** is a defect that you plant on purpose in a correct program, to find out whether the test that is supposed to catch it does.

The two halves of the title are held to different standards, and the difference organizes the chapter. Training is not reproducible digit for digit in another language: a run of three hundred steps amplifies rounding differences in the library's `exp` and `cos`, so a trained loss is printed only with the digits that survive when the run is repeated from slightly different starting points (the stable-digits rule of this book, applied in milestone 4). The cache is held to an exact standard: for every way of cutting a text into a prompt and a continuation, the cached logits equal the uncached ones, bit for bit in one summation order and within a tolerance that has a derivation in another. The project is built so that the exact parts are checked exactly and the statistical part is checked statistically, and never the other way round.

The model is small, 6,288 parameters, so that a plain loop in any language trains it in minutes. Its structure is that of GPT; nothing has been simplified away except the width.

![Seven boxes labelled with the milestones in build order. Milestones 1 (tokenizer), 2 (data pipeline) and 3 (the GPT) are in a row across the top from left to right. Milestone 4 (training) is below 3, and milestones 5 (generation) and 6 (KV cache, highlighted) continue to the left along the second row. Milestone 7 (work report) is below 6. Each arrow points from a milestone to the next one that uses its result.](img/project-gpt-milestones.svg)

The figure is a valid build order. The tokenizer is used by everything after it; the model of milestone 3 is the thing the cache of milestone 6 must reproduce and the thing milestone 7 counts. Build and check each milestone before starting the next, because a disagreement in milestone 6 that traces back to the tokenizer costs an afternoon and the same disagreement caught by the digest of milestone 1 costs a minute. The **acceptance tests** at the end collect every expected output in one place. Every expected value was computed by a second, independent implementation, `scripts/book/worked/project_gpt.py`, which imports the worked modules of the earlier chapters instead of copying them and is regenerated whenever the book is built. Use only your language's standard library. You need the book's generator (Volume I, Chapter 17), SHA-256, double-precision arithmetic with a correctly rounded `exp`, `log`, `sqrt`, `sin` and `cos` where the book uses them, and the libraries you wrote for the earlier projects.

Before the first milestone, six questions bring back what it leans on. They reach across the book, and the fifth and the sixth reach back to Volumes III and IV. Answer each from memory; the answers are at the end.

1. (Chapter 2, [Tokenization](02-tokenization.md)) A corpus has the words `abab` twice, `ab` once and `abb` three times. Which pair does byte-pair encoding merge first, which second, and how is `abb` written after the second merge?
2. (Chapter 10, [Training a GPT from Scratch](10-training-gpt.md)) A stream of 100 token ids is packed into rows of $T = 16$ positions with the target of each position the next token. How many rows are there, and how many positions of the last row are padding?
3. (Chapter 11, [Decoding](11-decoding.md)) A model gives five tokens, numbered 0 to 4, the probabilities $0.10, 0.50, 0.04, 0.30, 0.06$. With a temperature of 1, top-p 0.85 and the uniform number $u = 0.5$, which token is drawn when the kept tokens are walked in increasing id order?
4. (Chapter 12, [The KV Cache](12-kv-cache.md)) A prompt of three tokens is prefilled and the first generated token is fed back. What position does that token get, how many keys does its attention read, and how many bytes does one token of context occupy in the cache of a model with $L = 2$ layers and width $D = 16$ when each stored number takes 8 bytes?
5. (Volume III, Chapter 6, [Proving Gradients Correct](../v03-neural-networks/06-gradient-checking.md)) A central difference with step $h = 10^{-5}$ is used to check an analytic gradient in double precision. What size of error does the formula itself make, what size does rounding make, and why is the central form used instead of the one-sided one?
6. (Volume IV, Chapter 3, [Memory Bandwidth, Compute and the Roofline](../v04-computation/03-roofline.md)) A $128 \times 128$ weight matrix in float32 multiplies one vector. How many FLOPs is that, how many bytes must be read, and what fraction of its peak can a device with a ridge of 50 FLOP per byte reach on it?

## Building the model in seven steps

The pseudocode follows [Pseudocode Conventions](../pseudocode.md): 0-indexed, row-major, explicit broadcasts, explicit randomness, `require` for every input assumption. It fixes interfaces and the order of the operations, and it reuses by name the routines of earlier chapters (`train_bpe` and `encode_piece` of Chapter 2, `pack`, `batch_order` and `fit` of Chapter 10, `choose_token` of Chapter 11, `extend` of Chapter 12) instead of repeating them. Every random input comes from the book's SplitMix64 (Volume I, Chapter 17) at a seed printed beside its table, derived by the rule `seed_of(name)` of the pseudocode page, and the draws are taken in the order the table states. Write the checks of each milestone as a test file that you rerun after every change.

Results are compared by digest wherever a table would be too long to print, and the layout of the bytes is always stated. A digest of integers (token ids, a mask) uses each integer as a little-endian unsigned 32-bit number; a digest of the 16-bit ids of a merge list uses little-endian unsigned 16-bit numbers. A digest of real numbers is taken after scaling: each value is multiplied by a stated power of ten, rounded to the nearest integer (ties away from zero) and written as a little-endian signed 64-bit integer. The scaling is what makes the digest portable. The weights are drawn with `normal()`, which calls `log`, `sqrt`, `sin` and `cos`, and two correctly working libraries can differ in the last bit of those; rounding to $10^{-12}$ (weights) or $10^{-8}$ (logits) removes a difference of $10^{-16}$ without hiding a mistake that matters. A trained number is never printed as a digest and never to more digits than survive the nudges of milestone 4.

### Milestone 1: a tokenizer that cannot fail

A byte-level encoder has one property that a word-level one lacks: it has an answer for every input. The first milestone builds one on a corpus small enough to print the rule of, and the corpus is built first because every later number depends on it.

The corpus is a family of three-sentence documents, drawn from four short lists, in which the second and third sentences repeat things from the first. The rule, in the book's pseudocode, with the lists written out in the table below it:

```text
function make_documents(rng: inout Generator, n_train: int, n_val: int)
        -> (list<text>, list<text>):
    names = ["ana", "ben", "cy", "dee", "eli", "fay"]
    colours = ["red", "blue", "green", "gold", "grey"]
    things = ["ball", "cup", "hat", "box", "pen", "key"]
    places = ["barn", "shed", "attic", "cellar"]
    train = []
    seen = map of text to bool, empty
    val = []
    while len(val) < n_val:
        n = names[rng.integer(len(names))]             # four draws per candidate, in this order
        c = colours[rng.integer(len(colours))]
        t = things[rng.integer(len(things))]
        p = places[rng.integer(len(places))]
        doc = concat(n, " has a ", c, " ", t, ". the ", t, " is ", c, ". ", n,
                     " keeps it in the ", p, ".")
        if len(train) < n_train:
            append doc to train
            seen[doc] = true
        else if not (doc in seen):
            append doc to val
            seen[doc] = true
    return (train, val)
```

The first 240 candidates are the training documents, repeats included, as a sample from a source would be. The validation documents are the next distinct candidates that occur neither in training nor among the validation documents already kept. That rule makes the split by document and the leak zero by construction; the validation loss of milestone 4 then measures what a model does on documents it has not seen, and a model that merely memorized the training documents cannot score well on it. The tokenizer is learned on the training documents only: count the pre-tokens of each training document with the pre-tokenizer of Chapter 2 (a word carries the space before it) and learn merges by `train_bpe`, taking the pair with the largest count each round and giving ties to the smaller (left id, right id), with ids 0 to 255 the bytes and merge $k$ the id $256 + k$. Training stops after the requested number of merges or when no pair is left.

<!-- worked: project_gpt.corpus_and_tokenizer -->
Corpus rule. Slots: name ['ana', 'ben', 'cy', 'dee', 'eli', 'fay'], colour ['red', 'blue', 'green', 'gold', 'grey'], thing ['ball', 'cup', 'hat', 'box', 'pen', 'key'], place ['barn', 'shed', 'attic', 'cellar']. Generator: the book's SplitMix64 seeded with 0x22442373fa631536 (`seed_of` of the name `project_gpt.corpus`). One draw is four integers: `integer(6)` for the name, `integer(5)` for the colour, `integer(6)` for the thing, `integer(4)` for the place (an index into the list above). The document is `<name> has a <colour> <thing>. the <thing> is <colour>. <name> keeps it in the <place>.` with single spaces. The first 240 draws are the training split (200 of them are different documents). Validation draws follow in the same stream; a draw is kept only if its document differs from every training document and from every validation document kept before it, until 60 are kept, which takes 94 draws. There are 720 possible documents, so one document has entropy ln 720 = 6.5793 nats; 0 validation documents occur in the training split.

| document | text | tokens (raw ids) |
|---:|---|---:|
| 0 | `dee has a gold pen. the pen is gold. dee keeps it in the barn.` | 18 |
| 1 | `eli has a green hat. the hat is green. eli keeps it in the barn.` | 18 |
| 2 | `cy has a gold pen. the pen is gold. cy keeps it in the cellar.` | 18 |

Tokenizer. Pre-tokenize each training document with the pre-tokenizer of Chapter 2 (a word carries the space before it), count the pre-tokens over the 240 training documents, and learn merges as Chapter 2 specifies: ids 0 to 255 are the bytes, merge $k$ is id $256 + k$; each round takes the pair with the largest weighted count, ties to the smaller (left id, right id), and stops after 110 merges or when no pair is left. After **73 merges** every pre-token of the corpus is a single symbol, and training stops there. The first twelve merges, as the bytes they produce (a space is written `_`): `_i`, `he`, `_t`, `_the`, `ee`, `_b`, `_h`, `_ha`, `_a`, `_k`, `_g`, `ps`. Merge-list digest: SHA-256 over the merges in order, each merge as the left id then the right id, each a little-endian uint16 (292 bytes): **3d33ce4d8c46fc84**. Every training document encodes to 18 raw tokens.

| sentence | bytes | tokens | raw ids | decode(encode(s)) == s |
|---|---:|---:|---|---|
| `ana has a gold hat.` | 19 | 6 | 327 272 264 295 290 46 | yes |
| `the hat is gold.` | 16 | 6 | 116 257 290 269 295 46 | yes |
| `fay keeps it in the cellar.` | 27 | 7 | 302 273 270 268 259 309 46 | yes |
| `ben has a blue cup. the cup is blue.` | 36 | 11 | 321 272 264 284 299 46 259 299 269 284 46 | yes |
| `café été` | 11 | 11 | 99 97 102 195 169 32 195 169 116 195 169 | yes |
<!-- /worked -->

Three things in the table are worth reading before you move on. The training stops after 73 merges although 110 were allowed: the corpus has only so many distinct words, and by the end of the merge list every word of the corpus has become one token. That is why every document has the same length, 18 tokens. The sentence `café été` was never in the corpus, and each of its three `é` is the two bytes 195 169; the encoder has no merge for them and leaves the raw bytes, and decoding gives the text back. That is the sense in which the encoder cannot fail, and the round trip `decode(encode(s)) == s` is the test to run on every input you can think of, not only on corpus text. And the digest of the merge list pins the tie rule: two correct implementations that break the same tie differently differ in the digest and in every later table, so when your digest disagrees, look for the tie rule before the counts.

### Milestone 2: the stream, the rows and the first batch

A model reads rows of a fixed length, and the milestone between the tokenizer and the model is the one place where a lost token or a shifted target is invisible to every later check except the loss. The pipeline has three steps that Chapter 10 derived: cut each document into ids and put the separator in front of it, cut the stream of all the documents into rows of $T$ positions with the target of each position the next token, and take the rows in a seeded order.

One decision is new here. A vocabulary with a row for each of the 256 bytes would add rows that no training token ever selects: only 22 byte values occur in the corpus, so 234 rows of 16 numbers, 3,744 parameters, which is 60 percent more than the model has, and it would widen the softmax of every position from 96 logits to 330. The model's vocabulary is therefore the set of ids that occur: id 0 is the document separator `EOT`, ids $1, 2, \ldots$ are the bytes that occur in the training documents in increasing order of byte value, and the merges follow in the order learned. The tokenizer of milestone 1 keeps its raw ids and a table maps a raw id to its compact id. A byte that is not in the training text has no compact id; the encoder raises an error for it (`require`), which is a stated limit of this model and not of the tokenizer.

<!-- worked: project_gpt.pipeline -->
Vocabulary. Id 0 is the document separator `EOT`; ids 1 to 22 are the bytes that occur in the 240 training documents in increasing byte value (`_` `.` `a` `b` `c` `d` `e` `f` `g` `h` `i` `k` `l` `n` `o` `p` `r` `s` `t` `u` `x` `y`); ids 23 to 95 are the 73 merges in the order learned. V = 1 + 22 + 73 = **96**. The encoder of milestone 1 gives raw ids; a raw id is mapped to its compact id, and a byte that is not in the training text has no compact id (the encoder refuses it).

Stream. Each document contributes its separator and then its tokens; the stream of the 240 training documents has 4,560 ids and that of the 60 validation documents 1,140. Rows. A row has T = 20 positions: row $r$ takes inputs `ids[20 r .. 20 r + 19]` and targets `ids[20 r + 1 .. 20 r + 20]`; the last row is padded with `EOT` and a loss mask of 0. That gives 228 training rows (the last has 1 padded position) and 57 validation rows. The generator's entropy is 0.3464 nats per target on the training stream and 0.3466 on the validation stream.

Batches. Seed 0x2f3c50d7dcdf7d22 (`project_gpt.batches`). One `shuffle` of 0 .. 227 per epoch, consecutive groups of 8 rows are the batches, an incomplete last group is dropped, and a new epoch is shuffled when the batches run out. The first batch holds rows **225 180 142 51 43 209 145 66**. Digest of the first batch: SHA-256 over the x ids of its 8 rows, then the y ids, then the masks, row by row, every number a little-endian uint32: **0f12396243159750**. Its first row (row 225), position by position:

| position | input token | input id | target token | target id | mask | document |
|---:|---|---:|---|---:|---:|---:|
| 0 | `_the` | 26 | `_shed` | 82 | 1 | 236 |
| 1 | `_shed` | 82 | `.` | 2 | 1 | 236 |
| 2 | `.` | 2 | `EOT` | 0 | 1 | 236 |
| 3 | `EOT` | 0 | `eli` | 58 | 1 | 237 |
| 4 | `eli` | 58 | `_has` | 39 | 1 | 237 |
| 5 | `_has` | 39 | `_a` | 31 | 1 | 237 |
| 6 | `_a` | 31 | `_gold` | 62 | 1 | 237 |
| 7 | `_gold` | 62 | `_pen` | 64 | 1 | 237 |
| 8 | `_pen` | 64 | `.` | 2 | 1 | 237 |
| 9 | `.` | 2 | `_the` | 26 | 1 | 237 |
| 10 | `_the` | 26 | `_pen` | 64 | 1 | 237 |
| 11 | `_pen` | 64 | `_is` | 36 | 1 | 237 |
| 12 | `_is` | 36 | `_gold` | 62 | 1 | 237 |
| 13 | `_gold` | 62 | `.` | 2 | 1 | 237 |
| 14 | `.` | 2 | `_eli` | 87 | 1 | 237 |
| 15 | `_eli` | 87 | `_keeps` | 40 | 1 | 237 |
| 16 | `_keeps` | 40 | `_it` | 37 | 1 | 237 |
| 17 | `_it` | 37 | `_in` | 35 | 1 | 237 |
| 18 | `_in` | 35 | `_the` | 26 | 1 | 237 |
| 19 | `_the` | 26 | `_cellar` | 76 | 1 | 237 |
<!-- /worked -->

The first batch is the check. Its rows come from the shuffled order, its first row shows a document boundary inside a row (the `EOT` at position 3 is the separator of the next document, and its target is the name that opens it), and the digest fixes all 480 numbers of the batch at once. When the digest disagrees, print the first row and compare it with the table: a target that is the input shifted by one position in the wrong direction, a separator missing at the head of a document, and a mask that is 1 on padding all show in the first row. Check also that the target of the last position of a row is the first input of the next row, since a packer that drops that token loses one target per row.

### Milestone 3: the model, and a gradient you can believe

The model is the one of Chapter 10: a token embedding $\mathbf{E}$ of shape $[V, D]$ plus a learned position table of shape $[T, D]$, $L$ pre-norm blocks of RMS norm, multi-head causal attention, RMS norm and a SwiGLU network with two residual additions (Chapter 9), a final RMS norm and a head that is the embedding table used a second time, $\text{logits} = \mathbf{X}_n \mathbf{E}^\top$. There are no biases, the epsilon of every RMS norm is $10^{-6}$, and the scores of a head are divided by $\sqrt{d_h}$ with $d_h = D/H$. The learned positions are the scheme chosen for this project because the cache of milestone 6 then has a position to get wrong and the position of the new token is a number you can print; the rotary scheme of Chapter 8 is an extension.

```text
function forward(theta, ids: list<int>) -> array<real>[T_in, V]:
    # theta holds the 21 tensors of the table below; D, H, F, L, V are their sizes
    require 1 <= len(ids) and len(ids) <= T
    X = zeros(real, [len(ids), D])
    for i in 0 .. len(ids):
        X[i, :] = E[ids[i], :] + P[i, :]               # token plus position
    for l in 0 .. L:                                   # block_forward of Chapter 9 (causal)
        X = block_forward(X, Wq[l], Wk[l], Wv[l], Wo[l], g1[l], g2[l],
                          Wgate[l], Wup[l], Wdown[l], H, 1e-6)
    (Xn, _, _) = rms_norm(X, gf, 1e-6)
    return matmul(Xn, transpose(E))                     # the tied head
```

The initialization is the one that Chapter 10 recommends for a model of this depth: every matrix entry is $0.02$ times a draw of `normal()`, taken row by row, and the two output matrices of each block ($\mathbf{W}_O$ and $\mathbf{W}_{\mathrm{down}}$) are further multiplied by $1/\sqrt{2L}$, so that the variance of the residual stream does not grow with the number of blocks; every gain starts at 1. The draw order differs from the storage order, so both are stated below.

The forward pass is checked against logits printed for a given initialization and input. The check is a digest, with the scaling of the milestones' preface, and it fixes all $8 \times V$ logits at once; the last position's first six logits are printed so that a mistake can be located (a wrong position table moves every row, a wrong mask moves only the rows after the first, a head that is not tied moves all of them).

<!-- worked: project_gpt.model_forward -->
Configuration: V = 96, D = 16, H = 2 heads of width 8, SwiGLU width F = 24, L = 2 pre-norm blocks (RMS norm, Chapter 9), T = 20 learned absolute positions, a final RMS norm and a head tied to E, no biases: 6,288 parameters. Initialization. Seed 0xc516b3eb6b4d3acb (`project_gpt.init`). Every matrix entry is `0.02 * normal()`, drawn row by row in the order E, P, then for each block W_Q, W_K, W_V, W_O, W_gate, W_up, W_down, with W_O and W_down also multiplied by 1/sqrt(2 L); all gains are 1. Digest of the initial weights: every entry of the 21 tensors of the table below, in the order of the table (which is not the order of the draws), each tensor row by row, scaled by $10^{12}$: **54053371f8a402ad**.

Forward check. Input: the first 8 input ids of validation row 0, `0 91 39 31 59 66 2 26`. The logits are the 8 by 96 matrix from the causal forward pass. Digest of all 768 logits row-major, scaled by $10^{8}$: **445d1657c033fe25**. The logits of the last position for token ids 0 to 5: 0.04241541 0.07403079 0.09798034 0.01703784 0.13506890 0.01562750. The validation loss of this initialization, the pooled mean cross-entropy per supervised position over the 57 validation rows (the sum over all their supervised positions divided by their number, so a padded position counts for nothing), is 4.5747, against $\ln V$ = 4.5643 for a model that guesses uniformly.

| tensor index | name | shape | what it is |
|---|---|---|---|
| 0 | E | [96, 16] | the token embedding and the tied head |
| 1 | P | [20, 16] | the learned positions |
| 2 to 10, 11 to 19 | W_Q, W_K, W_V, W_O, g1, g2, W_gate, W_up, W_down | 4 x [16, 16], 2 x [1, 16], 2 x [16, 24], [24, 16] | block 0, then block 1; g1 and g2 are the gains of its two norms |
| 20 | g_f | [1, 16] | the gain of the final norm |
<!-- /worked -->

The loss measured before any step is the other number to look at: Chapter 10 showed that this initialization gives a loss within a small excess of $\ln V$, and the block above shows 4.5747 against 4.5643, an excess of 0.010. A first loss of 9 would say the weights were drawn too large, and a first loss of 0.2 would say that the model can see its targets.

A digest cannot tell you whether the backward pass is right, only whether the forward pass is the same as ours. The backward pass is checked as Volume III taught: against central differences in double precision, at coordinates drawn at random. Checking all 6,288 coordinates would cost 12,576 evaluations of the loss; sixty coordinates drawn from a seeded stream cost 120 evaluations, touch every group of tensors in the table below, and are the same sixty in every language.

```text
function check_gradients(theta, row, rng: inout Generator, n_coords: int, h: real)
        -> real:
    # masked_loss_and_gradients of Chapter 10: the mean loss over the supervised
    # positions of the rows of one batch, and its gradient for every tensor
    (loss, g) = masked_loss_and_gradients(theta, [row], [0])
    worst = 0.0
    for c in 0 .. n_coords:
        k = rng.integer(21)                        # the tensor, by the index of the table
        r = rng.integer(theta[k].shape[0])
        j = rng.integer(theta[k].shape[1])
        keep = theta[k][r, j]
        theta[k][r, j] = keep + h
        (hi, _) = masked_loss_and_gradients(theta, [row], [0])
        theta[k][r, j] = keep - h
        (lo, _) = masked_loss_and_gradients(theta, [row], [0])
        theta[k][r, j] = keep                       # restore before the next coordinate
        worst = max(worst, abs(g[k][r, j] - (hi - lo) / (2 * h)))
    return worst
```

<!-- worked: project_gpt.gradient_check -->
The model of the forward check at its initialization, the loss of one training row (row 225, the first row of the first batch) as the mean cross-entropy over its supervised positions. 60 coordinates, drawn from seed 0x8c85ab6be65283f0 (`project_gpt.gradcheck`): for each, `integer(n)` for the tensor index of the table of the forward check ($n$ = 21), then `integer(rows)` and `integer(columns)` of that tensor. Numerical derivative: central difference with step 1e-05 in float64. For example the first coordinate is tensor index 13, row 15, column 4: analytic 0.005081573, numeric 0.005081573. Largest absolute difference over the 60 coordinates: 2.2e-10, against a largest analytic gradient of 2.4e-2; criterion: below $10^{-8}$ on every coordinate (the last column shows the power of ten that the group stayed below in this run).

| parameters | coordinates drawn | largest |analytic| | largest |analytic - numeric| |
|---|---:|---:|---:|
| W_emb (token embedding and head) | 2 | 2.5e-3 | below 1e-10 |
| W_pos (positions) | 3 | 2.4e-2 | below 1e-9 |
| W_Q | 11 | 3.4e-5 | below 1e-9 |
| W_K | 3 | 2.1e-5 | below 1e-10 |
| W_V | 5 | 5.1e-3 | below 1e-10 |
| W_O | 9 | 1.4e-2 | below 1e-10 |
| W_gate | 8 | 6.2e-4 | below 1e-10 |
| W_up | 5 | 4.5e-4 | below 1e-10 |
| W_down | 5 | 1.2e-3 | below 1e-10 |
| block gains | 6 | 2.5e-4 | below 1e-10 |
| final gain | 3 | 1.0e-2 | below 1e-10 |
<!-- /worked -->

The criterion is a difference below $10^{-8}$ on every coordinate. That is a statement with a derivation: the central difference makes an error of order $h^2$ times a third derivative, and with $h = 10^{-5}$ that is about $10^{-10}$ for these gradients, while the rounding of the loss ($10^{-16}$ relative to a loss near 4.6) divided by $2h$ is about $10^{-11}$. A gradient that is wrong has an error of the size of the gradient itself, and the largest gradients of the groups in the table lie between $2 \times 10^{-5}$ and $2 \times 10^{-2}$. Hence the gap of about a decade between what a correct gradient does here and the criterion, and of three decades or more between the criterion and a wrong gradient of these sizes. The restore before the next coordinate matters more than it looks: a checker that leaves a perturbed weight in place checks every later coordinate against a different model.

### Milestone 4: training, and a checkpoint that resumes exactly

Training is `fit` of Chapter 10 with the settings below, a plausible choice for a short run and not a tuned one. Batches of 8 rows in the order of milestone 2; 300 steps of AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$ and a weight decay of 0.1 on every matrix except the gains; a rate that warms up linearly over 20 steps to a peak of 0.01 and follows half a cosine to 0.001 at step 300 (the rate of the update numbered $s$ is that of the 0-based index $s - 1$); and a clip on the global norm of the gradient at 1.0 with the coefficient $\min(1, \tau_{\mathrm{clip}}/(\lVert g \rVert + 10^{-6}))$. The loss is the mean cross-entropy over the supervised positions of the batch. Every 10 steps the validation loss is recorded: the pooled mean cross-entropy per supervised position over all validation rows (the sum of their cross-entropies at supervised positions divided by the number of those positions).

What is new is the way the numbers are reported. A run of 300 steps is a long chain of operations, and the library's `exp` and `cos` differ from one language to the next in the last bit; the chain can amplify that difference until the digits of a late loss are not the same. The book's rule (`scripts/book/worked/stable.py`) is to repeat the run from four starting points that differ from the original by adding $\pm 10^{-12}$ and $\pm 3 \times 10^{-12}$ to the first entry of $\mathbf{E}$, and to print only the digits on which all five runs agree. This run is stable to the four digits shown: the five agree on every number in the table, so an implementation that follows the specification should reproduce them. If yours does not agree in the third decimal, do not tune your arithmetic: look first for a difference in the specification (the decay applied to the gains, the bias correction using the step index in place of the count of updates, the clip coefficient, an epoch boundary in the batch order).

<!-- worked: project_gpt.training_curve -->
Model of milestone 3 (V = 96, T = 20, D = 16, L = 2, 2 heads, SwiGLU width 24), initialization `project_gpt.init`. Batches of 8 rows in the order of milestone 2, 300 steps of AdamW (beta1 0.9, beta2 0.95, epsilon 1e-08, decay 0.1 on every matrix except the gains), the rate warming up linearly over 20 steps to 0.01 and following half a cosine to 0.001 at step 300 (`schedules.cosine`; the table shows the rate of the update that ends at the step), global-norm clip at 1.0 with the coefficient $\min(1, \tau_{\mathrm{clip}} / (\lVert g \rVert + 10^{-6}))$. The training loss is the mean of the batch losses of the 10 steps that end at the step; the gradient norm is that of the step itself, before the clip. Trained numbers are printed with the digits that survive four nudged restarts (a weight changed by $\pm 10^{-12}$ and $\pm 3 \times 10^{-12}$, `stable.py`).

| step | learning rate | gradient norm | training loss | validation loss |
|---|---:|---:|---:|---:|
| 0 | - | - | - | 4.5747 |
| 10 | 0.00500 | 0.852 | 4.4453 | 4.1538 |
| 20 | 0.01000 | 0.584 | 3.7871 | 3.3602 |
| 30 | 0.00998 | 0.920 | 3.1489 | 2.8810 |
| 50 | 0.00976 | 0.919 | 1.9652 | 1.6656 |
| 100 | 0.00835 | 0.988 | 0.7541 | 0.7352 |
| 150 | 0.00605 | 0.548 | 0.6762 | 0.6776 |
| 200 | 0.00359 | 0.509 | 0.6549 | 0.6678 |
| 250 | 0.00172 | 0.511 | 0.6489 | 0.6529 |
| 300 | 0.00100 | 0.663 | 0.6443 | 0.6480 |

The clip engaged in 37 of the 300 steps. A model that guesses uniformly has loss $\ln V$ = 4.5643; the entropy of the generator is 0.3466 nats per target on the validation stream. A model that learned the grammar and the vocabulary but copied nothing from the first sentence into the second would pay, in addition, $\ln 6 + \ln 5 + \ln 6$ = 5.193 nats in each document, which is 0.620 per target on the validation stream.
<!-- /worked -->

![Training and validation loss against the optimizer step, from step 50 to 300. Both curves fall steeply from about 2.0 and 1.7 at step 50 and flatten near 0.65 after step 130, the validation curve just above the training curve. A dashed horizontal line near 0.62 is labelled "grammar learned, nothing copied" and another near 0.35 is labelled "entropy of the generator"; the curves approach the first and stay well above the second.](img/project-gpt-curves.svg)

The loss does not reach the entropy of the generator, and the table says why in numbers that are not guesses. Every document repeats its thing and its colour in the second sentence and its name in the third. A model that has learned the vocabulary and the grammar but not to copy pays for those three choices as if they were new draws: $\ln 6 + \ln 5 + \ln 6$ nats in each document, which is the level the figure marks. The validation loss ends within a few hundredths of it. The model has learned the language and has not yet learned the one thing that needs attention to look back at a specific earlier token, which this model has not learned in these 300 steps. The sampled text of milestone 5 shows the same thing from the other side.

A checkpoint is the second thing that has to be exact, because a run that is stopped and resumed must be the run that was not stopped. What a checkpoint holds is the parameters, both moments of Adam, and the number of applied optimizer steps. The batch order need not be stored, because it is a function of its seed and the step. The bytes are laid out so that a reader in any language writes the same file: the step count as one little-endian unsigned 64-bit integer, then all parameters, then all first moments, then all second moments, each in the tensor order of milestone 3 and row by row, every number the little-endian bytes of its IEEE 754 double.

```text
function resume(file: bytes, theta, steps: int, ...) -> bytes:
    (theta, m, v, t0) = load(file)                  # the 8-byte count, then the three groups
    order = batch_order(n_rows, B, steps, rng)     # regenerated from the seed, never stored
    for t in t0 .. steps:                           # the same loop body as `fit`, from step t0
        batch = order[t]
        (loss, g) = masked_loss_and_gradients(theta, rows, batch)
        norm = global_norm_clip(g, tau_clip)
        eta = schedule_rate(t, eta_max, eta_min, warmup, 0, steps)
        for i in 0 .. len(theta):
            adam_step(theta[i], g[i], m[i], v[i], t + 1, eta, 0.9, 0.95, 1e-8,
                      lam if decay[i] else 0.0, true)
    return save(theta, m, v, steps)
```

The test compares two files. Run steps 0 to 299 without stopping and keep the file written after step 300. Run steps 0 to 149, write the file, load it into fresh variables, continue to step 300 and write the file. The files must be equal byte for byte. The table adds the two mistakes that readers make, so that you can see what a failed resume looks like: moments that are reset to zero, and a step counter that restarts, which sets the rate back to the warmup and the bias correction of Adam back to its first step.

<!-- worked: project_gpt.checkpoint_and_resume -->
The run of the training table, stopped after step 150 and continued from a checkpoint. Checkpoint layout: the number of applied optimizer steps as a little-endian uint64, then all 6,288 parameters, then all first moments, then all second moments, each tensor row by row in the order of the weight digest, every number binary64 little-endian: 8 + 3 x 8 x 6,288 = **150,920** bytes (the file written at step 150 has 150,920). The batch order is not stored: it is regenerated from its seed, and the step counter picks the batch. The resumed run trains steps 150 to 299 of the same batch order; the comparison is of the checkpoint bytes written after step 300 (parameters and moments) with those of the run that never stopped.

| what the checkpoint restores | bytes equal to the uninterrupted run | largest parameter difference |
|---|---:|---:|
| parameters, both moments and the step counter | yes | 0 |
| parameters and step counter; both moments set to zero | no | above 1e-6 |
| parameters and moments; step counter restarted at 0 | no | above 1e-6 |
<!-- /worked -->

Neither broken resume crashes, and each trains a different model: the parameters at step 300 differ from the uninterrupted run's by more than $10^{-6}$. Nothing in the loop reports it, and the only defence is the comparison of the bytes, which is why the test is stated as one. The comparison is made on a determined run: a resumed run and an uninterrupted run are bit-identical only because both ran on one implementation with one order of additions. That is the sense in which the exact standard holds for this milestone: it compares your implementation with itself, and no digest of the trained weights is printed because no other language could match it.

### Milestone 5: generation with a seeded sampler

The trained model gives, for a context, the logits of the next token, and Chapter 11's sampler turns them into a choice. The project's sampler is `choose_token` with no penalties (the repetition factor 1, both additive penalties 0, no min-p): the logits are divided by the temperature, softmaxed, cut to the $k$ most probable ids and then to the smallest set of the most probable whose mass reaches $p_{\mathrm{top}}$, renormalized, and one uniform number $u$ walks the kept ids in increasing id order and selects the first whose cumulative probability exceeds $u$. A temperature of 0 is greedy decoding, takes the largest logit (the lowest id on a tie) and draws no random number. A generation continues from a prompt until it draws the separator or the context holds $T$ tokens, and the context is never cropped, because a cache holds positions $0, \dots, T - 1$ and nothing slides.

```text
function generate(theta, prompt: list<int>, tau: real, k: int, p_top: real,
                  rng: inout Generator) -> list<int>:
    ctx = copy(prompt)
    out = []
    seen = zeros(bool, [V])                              # no penalties: nothing is seen,
    count = zeros(int, [V])                              # nothing is counted
    while len(ctx) < T:
        z = forward(theta, ctx)[len(ctx) - 1, :]         # the last position's logits
        t = choose_token(z, seen, count, 1.0, 0.0, 0.0, tau, k, p_top, 0.0, rng)
        append t to ctx
        append t to out
        if t == 0:
            break                                        # the separator
    return out
```

The sampler is checked first on its own, with no model, because a mistake in the order of its steps or in the walk is easier to find on ten numbers than in the text of a trained model. The table has four draws for each setting from one fixed distribution.

<!-- worked: project_gpt.sampler_fixture -->
Ten logits z = [2.0, 1.0, 0.5, 0.0, -0.5, 1.5, 0.2, -1.0, 0.8, 0.3] (ids 0 to 9) and four uniform numbers u = [0.3, 0.6, 0.85, 0.98], each used for one draw from the same distribution. The sampler divides z by the temperature, takes the softmax, keeps the top-k ids and then the smallest top-p set of the survivors (the most probable first, ties to the lower id), renormalizes, and walks the kept ids in increasing id order, returning the first whose cumulative probability exceeds u. Greedy takes the largest logit and ignores u.

| setting | ids kept | ids drawn for the four u |
|---|---|---|
| greedy | 0 | 0 0 0 0 |
| temperature 0.7 | 0 1 2 3 4 5 6 7 8 9 | 0 3 6 9 |
| temperature 1, top-k 4 | 0 1 5 8 | 0 1 5 8 |
| temperature 1, top-p 0.9 | 0 1 2 5 6 8 9 | 0 5 8 9 |
<!-- /worked -->

The numbers to compare are the ids drawn, and the setting that shows the order of the steps is the last: the top-p set is cut from the distribution after the temperature, so changing the order of the two changes the set (Chapter 11). Then the model. A prompt is the separator followed by the encoding of the prompt text; each row of the table below has a generator of its own, seeded from the name of its setting and prompt, so that the rows do not depend on one another or on the order in which you compute them.

<!-- worked: project_gpt.generation -->
The trained model of milestone 4 (step 300). A prompt is the separator followed by the encoding of the prompt text; the model continues until it produces `EOT` (id 0) or the context holds T = 20 tokens. The context is never cropped. Each row has a generator of its own, seeded with `seed_of` of `project_gpt.sample.<setting>.<prompt>` (printed in the table); one `uniform()` is drawn per new token for the sampling settings, none for greedy. Settings: temperature divides the logits before the softmax; top-k keeps the 4 most probable ids; top-p keeps the smallest set of most probable ids whose mass reaches 0.9 (applied after top-k). The generated ids are identical in the base run and in all four nudged runs: yes. The ids shown are the compact ids of milestone 2.

| setting | prompt | seed | generated ids | text |
|---|---|---:|---|---|
| greedy | `ana has a gold` | 0xa630a002f9317c43 | 56 2 26 64 36 51 2 87 40 37 35 26 73 2 0 | `ana has a gold box. the pen is blue. eli keeps it in the barn.` |
| greedy | `fay has a` | 0xba773092e42a8c1c | 48 56 2 26 64 36 51 2 87 40 37 35 26 73 2 0 | `fay has a green box. the pen is blue. eli keeps it in the barn.` |
| greedy | `ben has a blue cup. the` | 0x4be971178df827db | 64 36 48 2 87 40 37 35 26 73 2 0 | `ben has a blue cup. the pen is green. eli keeps it in the barn.` |
| temperature 0.7 | `ana has a gold` | 0x7b41e30c73f7c166 | 66 2 26 66 36 48 2 89 40 37 35 26 73 2 0 | `ana has a gold cup. the cup is green. ben keeps it in the barn.` |
| temperature 0.7 | `fay has a` | 0xc7e2b3a9c054f7c9 | 48 64 2 26 64 36 59 2 95 40 37 35 26 86 2 0 | `fay has a green pen. the pen is grey. ana keeps it in the attic.` |
| temperature 0.7 | `ben has a blue cup. the` | 0x3d2ea8ad571fdef8 | 56 36 51 2 89 40 37 35 26 73 2 0 | `ben has a blue cup. the box is blue. ben keeps it in the barn.` |
| temperature 1, top-k 4 | `ana has a gold` | 0x8adeae5c3b09fcee | 56 2 26 57 36 51 2 87 40 37 35 26 76 2 0 | `ana has a gold box. the hat is blue. eli keeps it in the cellar.` |
| temperature 1, top-k 4 | `fay has a` | 0x5be8067a67a7fc3b | 59 56 2 26 71 36 54 2 89 40 37 35 26 76 2 0 | `fay has a grey box. the key is red. ben keeps it in the cellar.` |
| temperature 1, top-k 4 | `ben has a blue cup. the` | 0xe70fcffc397eec92 | 64 36 54 2 90 40 37 35 26 76 2 0 | `ben has a blue cup. the pen is red. dee keeps it in the cellar.` |
| temperature 1, top-p 0.9 | `ana has a gold` | 0x67cc77a7e9ce924d | 79 2 26 56 36 48 2 87 40 37 35 26 82 2 0 | `ana has a gold ball. the box is green. eli keeps it in the shed.` |
| temperature 1, top-p 0.9 | `fay has a` | 0x2758712e7bc2adc0 | 54 57 2 26 57 36 51 2 95 40 37 35 26 73 2 0 | `fay has a red hat. the hat is blue. ana keeps it in the barn.` |
| temperature 1, top-p 0.9 | `ben has a blue cup. the` | 0x8e3901a23976e9f2 | 57 36 48 2 92 40 37 35 26 86 2 0 | `ben has a blue cup. the hat is green. cy keeps it in the attic.` |
<!-- /worked -->

Every generated id of the table is the same in the five runs of milestone 4, so these ids are as reproducible as the trained weights are, and your implementation should print them exactly. The text shows the claim of the previous milestone from the other side: every document in the table has the form of the corpus, each slot filled from its own list, and the second sentence names a thing and a colour that need not be the first sentence's. The next table puts a number on it.

<!-- worked: project_gpt.what_it_learned -->
200 documents sampled from the trained model of milestone 4 with temperature 1 and no cut, starting from the separator, one generator for all of them (seed 0x453c8c021d031d4d, `project_gpt.samples`, one `uniform()` per token). A document is well formed if it is exactly `<name> has a <colour> <thing>. the <thing> is <colour>. <name> keeps it in the <place>.` with every slot filled from its list. Well formed: **180** of 200. The counts below are among the well-formed documents; 'chance' is the rate of a model that fills the slot independently of the first sentence. The counts are identical in the four nudged runs: yes.

| what the grammar does not force | documents | rate | chance |
|---|---:|---:|---:|
| thing repeated in the second sentence | 18 | 0.100 | 0.167 |
| colour repeated in the second sentence | 37 | 0.206 | 0.200 |
| name repeated in the third sentence | 22 | 0.122 | 0.167 |
<!-- /worked -->

The rates of the thing and the name are at or below chance and the colour's is at chance: the model's choices in the later sentences are not tied to the first sentence's. The model has learned what the grammar forces, and only that. The loss of milestone 4 said so before the samples did, and from a number that was derived on paper before the run; a check on a trained model should be of that kind.

### Milestone 6: the cache, and a test that fails when it should

The cache of Chapter 12 stores, for each layer, the keys and the values of every token of the context so far. Prefill runs the prompt through the model once and fills the cache; each decode step runs one token at the position equal to the cache length, appends its keys and values, and attends to the whole cache. The routine is `extend` of Chapter 12; the project's version differs in nothing but its model, the one of milestone 3, and the point of the milestone is the test.

The test is the property named in Chapter 12, cache equivalence: for every text and every way of cutting it into a prefill and decode steps, the logits of the cached run equal those of one forward pass over the whole text. The reference is `forward` of milestone 3, the code that training ran, with no cache. The test cuts one text at every position.

```text
function cache_equivalence(theta, ids: list<int>, tol: real) -> bool:
    ref = forward(theta, ids)                          # the uncached reference, all rows
    for c in 1 .. len(ids):                            # every split: prefill c, decode the rest
        cache = KvCache(zeros(real, [L, T, D]), zeros(real, [L, T, D]), 0)
        rows = []
        z = extend(theta, cache, ids[0:c])             # the prefill: logits of position c - 1
        append z to rows
        for i in c .. len(ids):
            z = extend(theta, cache, [ids[i]])         # one decode step, position cache.length
            append z to rows
        for i in c - 1 .. len(ids):
            if max(abs(rows[i - c + 1] - ref[i, :])) > tol:
                return false
    return true
```

Two things decide whether such a test means anything: what is compared, and the tolerance. The text is the separator and the tokens of validation document 0, so every split cuts a real document. The tolerance is not chosen. The cached run and the reference add the same numbers, and in the same order they agree exactly (the first row of the table). A decode kernel of a real device adds in another order, which `kv_cache.extend` models with a second summation order: each dot product of a query and a key as two partial sums, each weighted sum of values in blocks of four keys. Volume IV, Chapter 9 gives the size of what that can do: each order of summing $m$ numbers is within about $m u S$ of the exact sum, where $u = 2^{-53}$ is the unit round-off of double precision and $S$ is the sum of the absolute values of the terms, so two orders differ by at most about $2 m u S$. The attention of one layer has two such sums, one over the $d_h$ entries of a query and a key and one over at most $T$ cached values, and no other sum differs between the two orders. Each layer's attention output is therefore perturbed by a relative error of at most $2(d_h + T)u$; the perturbation enters the residual stream by an addition, and the head is linear in the final state, so over $L$ layers and on a scale of the largest logit $z_{\max}$ the tolerance is
$$\mathrm{tol} = 2\,L\,(d_h + T)\,u\,z_{\max}.$$
This is a first-order count and not a proof: it takes the layers after the perturbed one to carry it with a gain of at most one. The check that it is a good tolerance has two sides, and the table shows both. It must be above what the reordering does, which the table's largest difference is, and far below any gap between the best and second-best logit, so that a difference within the tolerance cannot change a greedy choice.

<!-- worked: project_gpt.cache_equivalence -->
Text: the separator and the tokens of validation document 0 (`cy has a grey cup. the cup is grey. cy keeps it in the attic.`), n = 19 ids `0 91 39 31 59 66 2 26 66 36 59 2 92 40 37 35 26 86 2`, run through the trained model of milestone 4. Reference: the full causal forward pass over all n ids (the training code, no cache). Test: for every split $c$ = 1 .. n - 1, prefill the first $c$ ids with the cache, then decode the remaining ids one at a time, and compare the logits of the positions $c - 1, \dots, n - 1$ (the last row of the prefill and one row per decode step) with the reference. Orders of summing: 'serial' adds left to right as the reference does; 'split' sums each query-key dot product as two partial sums (even and odd entries) and each weighted sum of values in blocks of 4 keys (`kv_cache.extend`). Tolerance: $2 L (d_h + T)\, u\, z_{\max}$ with $L$ = 2, $d_h$ = 8, $T$ = 20, $u = 2^{-53}$ and $z_{\max}$ = 15.27, the largest absolute logit of the reference, which is **1.9e-13**. The smallest gap between the best and second-best logit at any position is 0.0015.

| order of the sums | splits tested | splits with logits bit-identical | largest logit difference | within tolerance |
|---|---:|---:|---:|---|
| serial | 18 | all | 0 | yes |
| split | 18 | not required | below 1e-14 | yes |

The tolerance is at least 19 times the difference of the 'split' order, and the smallest gap between the best and second-best logit is more than $10^{9}$ times the tolerance, so a difference within it cannot change a greedy choice.
<!-- /worked -->

In the order of the reference, the cached logits equal the reference's in all 18 splits to the last bit. In the other order they differ by rounding, far below the tolerance, and the tolerance is far below the gap between the best and second-best logit. Together these give the claim of the project's title, in the form it can be tested: the cached generation produces the same greedy tokens as the uncached one because the logits differ by far less than a billionth of the smallest gap that an argmax has to cross. The claim needs the gap: a test in brain, described below, had to pick a weight draw "with no near-tie argmax", which is the same condition stated the other way round, as a seed chosen after looking. Here the condition is a number computed from the model.

A test that passes the correct cache has shown nothing until it is shown to fail on a wrong one. The wrong one the milestone requires is the off-by-one in the position of the new token. A decode step feeds one token, and nothing in that call says where it sits, so the position has to be computed: it is the cache length, and a routine that numbers positions by the index of the row within the call is right for every prefill and wrong for every decode step. The mutations below change only the position that a decode step adds to the embedding of the fed token. The prefill is always right and the keys and values are always appended at the end of the cache.

<!-- worked: project_gpt.position_mutations -->
The text of the equivalence test, split after the first 4 ids (`0 91 39 31`): the prefill is correct, and each decode step feeds the next id of the text and is given the position the variant computes from the cache length $c$ and the prompt length. Three tests, each against the reference forward pass: 'first step' compares the logits of the first decode step only; 'all steps' compares the logits of all 15 decode steps with the tolerance 1.9e-13 of the equivalence test; 'greedy tokens' generates 15 tokens by greedy decoding with the cache and without and compares the two token lists. The verdicts are the same in the four nudged runs.

| position given to the fed token | first step | all steps | greedy tokens | largest logit difference |
|---|---|---|---|---|
| correct: the cache length | pass | pass | pass | 0 |
| one too small: the last position already in the cache | FAIL | FAIL | FAIL | 2.8e0 |
| one too large | FAIL | FAIL | FAIL | 2.9e0 |
| never advanced: the length of the prompt | pass | FAIL | FAIL | 4.8e0 |
| restarted: position 0, the row index inside the call | FAIL | FAIL | FAIL | 1.1e1 |
<!-- /worked -->

All four mutations are caught by the test that compares the logits of every decode step, by more than twelve orders of magnitude over the tolerance, and none of them crashes or produces anything that looks wrong in the text. The table also shows why the test has to look at more than one step. The mutation that never advances the position is exactly right at the first decode step, where the cache length equals the prompt length, so a test that checks one decode step passes it. And a test that compares only the generated tokens is a weaker test than one that compares logits, because a mutation can move the logits by a large amount and leave an argmax where it was; here it happens that every mutation also changes the tokens, but nothing in the design of the model promised that. Compare the logits, at every step, at several splits.

Last, the generations of milestone 5 are repeated with the cache. The generators draw the same uniform numbers in the same order, so a sampled generation is a stronger test than a greedy one: any difference of logits that is large enough to move a cumulative probability past a draw changes a token.

<!-- worked: project_gpt.generation_equivalence -->
The twelve generations of milestone 5 repeated with the cache: the prompt is prefilled once, each new token is decoded with the keys and values of the tokens before it, and the same generators draw the same uniform numbers in the same order.

| setting | prompt | prompt ids | ids generated | same ids as without the cache |
|---|---|---:|---:|---|
| greedy | `ana has a gold` | 5 | 15 | yes |
| greedy | `fay has a` | 4 | 16 | yes |
| greedy | `ben has a blue cup. the` | 8 | 12 | yes |
| temperature 0.7 | `ana has a gold` | 5 | 15 | yes |
| temperature 0.7 | `fay has a` | 4 | 16 | yes |
| temperature 0.7 | `ben has a blue cup. the` | 8 | 12 | yes |
| temperature 1, top-k 4 | `ana has a gold` | 5 | 15 | yes |
| temperature 1, top-k 4 | `fay has a` | 4 | 16 | yes |
| temperature 1, top-k 4 | `ben has a blue cup. the` | 8 | 12 | yes |
| temperature 1, top-p 0.9 | `ana has a gold` | 5 | 15 | yes |
| temperature 1, top-p 0.9 | `fay has a` | 4 | 16 | yes |
| temperature 1, top-p 0.9 | `ben has a blue cup. the` | 8 | 12 | yes |
<!-- /worked -->

### Milestone 7: the work report

The last milestone counts what the model costs, in FLOPs and in bytes, as Volume IV taught: a count that can be derived on paper and that no machine can contradict.

The training count is Chapter 10's. A token costs, in a forward pass, twice the parameters that take part in a matrix product plus four times $LTD$ for the two products of attention, with all $T$ scores counted; the backward pass costs twice that. The decode count is Chapter 12's: one new token on a cache of $c$ tokens reads $k = c + 1$ keys. Write your count as a formula and also count by running: increment a counter at every multiply-add of the model's products and compare. The table has both and they agree to the unit.

<!-- worked: project_gpt.work_report -->
The model of milestone 3: V = 96, D = 16, L = 2, H = 2, F = 24, T = 20, SwiGLU (three matrices). A multiply-add is two FLOPs; only matrix products and attention products are counted (norms, softmax, the SwiGLU gate and residual adds are not). Parameters in matrix products: $N_{\mathrm{p}} = L(4D^2 + 3DF) + VD$ = 2 x (4 x 256 + 3 x 384) + 1536 = 5,888 (the position table and the gains take no part). Forward FLOPs per token on a row of $T$ positions, all $T$ scores counted: $2N_{\mathrm{p}} + 4LTD$ = 11,776 + 2,560 = **14,336**. Training costs three times the forward: **43,008** per token, 6,881,280 per step of 160 tokens, 2.064e9 for the 300 steps.

Decoding one token on a cache of $c$ tokens (formula `kv_cache.flops_new`: $2(N_{\mathrm{p}} - VD)$ for the blocks, $4LDk$ for $k = c + 1$ visible keys, $2DV$ for the head) against the multiply-adds counted by running `kv_cache.extend`:

| cache length c | keys read k | FLOPs counted by running | FLOPs by the formula |
|---|---:|---:|---:|
| 0 | 1 | 11,904 | 11,904 |
| 1 | 2 | 12,032 | 12,032 |
| 8 | 9 | 12,928 | 12,928 |
| 16 | 17 | 13,952 | 13,952 |
| 19 | 20 | 14,336 | 14,336 |
<!-- /worked -->

The two counts agree for every cache length, which is the check on the formula, and the decode cost grows only with the $4LDk$ of attention: at a cache length of 19, which reads $k = 20$ keys, $4 L D k = 2{,}560$ of the 14,336 FLOPs are attention, and the other 11,776, which is $2N_{\mathrm{p}}$, are the same at every length. The cache has a size in bytes that depends only on the shapes and the format of the stored numbers, and the roofline of Volume IV gives the arithmetic intensity of the step.

<!-- worked: project_gpt.cache_and_intensity -->
Cache bytes per token = 2 (keys and values) x L x D x bytes per number = 2 x 2 x 16 x b:

| numbers stored as | b | cache bytes per token | cache bytes at T = 20 |
|---|---:|---:|---:|
| binary64 (this module) | 8 | 512 | 10,240 |
| float32 | 4 | 256 | 5,120 |
| bfloat16 | 2 | 128 | 2,560 |

Decode arithmetic intensity, one sequence, numbers stored in float32 (b = 4): the FLOPs of the step over the bytes it must read, which are the weights once (4 $N_{\mathrm{p}}$ bytes; the embedding lookup reads one row and is left out) and the cache rows of the $c$ tokens before the new one. The devices are the two illustrative ones of Volume IV (`roofline.MACHINES`): the GPU, whose ridge is the peak over the bandwidth, 50 FLOP per byte, and, in the last column, the CPU, whose ridge is 10. 'Attainable' is the intensity over the ridge, capped at 1.

| cache length c | FLOPs | weight bytes | cache bytes read | FLOP per byte | attainable fraction, GPU | attainable fraction, CPU |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 12,032 | 23,552 | 256 | 0.5054 | 0.0101 | 0.0505 |
| 8 | 12,928 | 23,552 | 2,048 | 0.5050 | 0.0101 | 0.0505 |
| 16 | 13,952 | 23,552 | 4,096 | 0.5046 | 0.0101 | 0.0505 |
| 19 | 14,336 | 23,552 | 4,864 | 0.5045 | 0.0101 | 0.0505 |

Prefill of n tokens into an empty cache (the weights read once, the cache rows written):

| n | FLOPs | weight bytes | cache bytes written | FLOP per byte | attainable fraction, GPU |
|---|---:|---:|---:|---:|---:|
| 4 | 39,168 | 23,552 | 1,024 | 1.594 | 0.0319 |
| 10 | 97,152 | 23,552 | 2,560 | 3.721 | 0.0744 |
| 20 | 204,032 | 23,552 | 5,120 | 7.116 | 0.1423 |

With many sequences sharing one read of the weights, the intensity at a cache length of 16 tends to the FLOPs of one step over its own cache bytes, 13,952 / 4,096 = 3.41 FLOP per byte, which is below both ridges: no batch size makes this model's decode step compute-bound. Generating 16 tokens after a prompt of 4 takes 1,819,648 FLOPs when every step recomputes the whole text and 238,848 with the cache, a factor of 7.62.
<!-- /worked -->

The decode step of this model has an intensity of 0.5 FLOP per byte, about a hundredth of the GPU's ridge and a twentieth of the CPU's. The reason is that it reads every weight to do two FLOPs with it, and the weights are 23,552 bytes against at most 4,864 bytes of cache. Prefill reads the same weights once for $n$ tokens, so its intensity grows with $n$, and a prompt of 20 tokens reaches 7.1. Neither phase reaches the ridge in this model, and no batch size would bring a decode step there, because the cache a sequence adds to the step scales with the batch exactly as its FLOPs do. All the numbers of this milestone are properties of the shapes, which is why the whole report can be reproduced on a machine that has never run the model. The one thing it does not contain is a time, and none could be stated without naming a machine and a date.

## What your run must reproduce

The tests are the blocks printed in the milestones above, collected as a checklist. An implementation passes when every statement holds. "Exact" means integers, digests and text equal; a number marked "digits shown" is compared only to the digits printed, which are the ones that survive the nudges of milestone 4.

| milestone | test | expected | tolerance |
|---|---|---|---|
| 1 | the corpus and the merge list: digest, number of merges, first merges | as printed | exact |
| 1 | raw ids of the five sentences, and `decode(encode(s)) == s` for each | as printed, yes | exact |
| 2 | vocabulary size, training and validation rows, leak of validation into training (zero) | as printed | exact |
| 2 | the first batch: row numbers, digest, and the 20 positions of its first row | as printed | exact |
| 3 | parameter count, digest of the initial weights, digest of the logits, the six printed logits | as printed | digests exact; logits to 8 decimals |
| 3 | gradient check at 60 drawn coordinates | every difference below $10^{-8}$ | criterion |
| 4 | learning rate, gradient norm, training and validation loss at the printed steps; clips | as printed | the digits shown |
| 4 | checkpoint size; resume with the full state equals the uninterrupted run, byte for byte; each broken resume differs | as printed | exact |
| 5 | the sampler fixture: ids drawn for the four settings | as printed | exact |
| 5 | the twelve generations: ids and text | as printed | exact |
| 5 | the 200 sampled documents: well-formed, and copy counts | as printed | exact |
| 6 | cache equivalence over all splits in both orders: bit-identical in the first, within the tolerance in the second | as printed | tolerance to 2 digits |
| 6 | the five position variants: verdicts of the three tests | as printed | exact |
| 6 | the twelve cached generations equal the uncached ones | yes | exact |
| 7 | FLOPs per token, counted and by formula; cache bytes per token in three formats; intensity and fractions of the ridge | as printed | exact; fractions to 4 decimals |

Three conventions apply to the whole list. Test in the order of the table: the digests of milestones 1 to 3 do not depend on training, so a failure there is a failure of a specification you can read; the trained numbers of milestones 4 to 6 depend on everything before them, so a disagreement there that follows agreement in 1 to 3 points at the optimizer, the batch order or the clip. The second convention is the one on which milestone 4 rests: no trained number is compared beyond the digits printed. The third is for the planted defect. Where a table shows a mutation, the verdict (FAIL by at least twelve orders of magnitude) is the test; the exact size of the logit difference it produces depends on the trained weights and is a qualitative outcome. A compact fingerprint of a correct implementation, to check at the end:

<!-- worked: project_gpt.acceptance_fingerprint -->
| milestone | quantity | value | tolerance |
|---|---|---|---|
| 1 | merge-list digest; number of merges | 3d33ce4d8c46fc84; 73 | exact |
| 1 | raw ids of `ana has a gold hat.` | 327 272 264 295 290 46 | exact |
| 2 | vocabulary size V; training rows; validation rows | 96; 228; 57 | exact |
| 2 | first batch: rows; digest | 225 180 142 51 43 209 145 66; 0f12396243159750 | exact |
| 3 | parameters; initial-weight digest | 6,288; 54053371f8a402ad | exact |
| 3 | logit digest of the forward check | 445d1657c033fe25 | exact |
| 3 | gradient check: largest difference over 60 coordinates | below 1e-8 | criterion |
| 4 | validation loss at steps 100 and 300 | 0.7352 and 0.6480 | digits shown |
| 4 | checkpoint size in bytes; resume with the full state gives equal bytes | 150,920; yes | exact |
| 5 | sampler fixture: ids for greedy; for top-p 0.9 | 0 0 0 0; 0 5 8 9 | exact |
| 5 | well-formed documents of 200 samples; thing, colour, name repeated | 180; 18, 37, 22 | exact |
| 6 | tolerance on the logits | 1.9e-13 | 2 digits |
| 7 | forward FLOPs per token; training FLOPs per token | 14,336; 43,008 | exact |
<!-- /worked -->

## The counterpart in brain's GPT and its serving cache

brain's GPT is `crates/gpt2`, and the project is its small sibling with a different norm and activation. `crates/gpt2/src/model.rs` documents its architecture in its header: pre-norm blocks, learned positions, a final norm and a head tied to the embedding table, with a LayerNorm and a GELU where this project's model has RMS norm and SwiGLU. Training is `crates/gpt2/src/train.rs`, a thin wrapper over the generic loop in `crates/model/src/train.rs` that Chapter 10 described, and the byte-level BPE trained on a corpus is `crates/data/src/bytebpe.rs`, whose ids $0$ to $255$ are the raw bytes, merges follow, and the separator takes the last id, with ties broken to the smallest pair as in milestone 1.

The cache is `Gpt::step` in `crates/gpt2/src/model.rs` and its caller `generate_kv` in `crates/gpt2/src/sample.rs`: a prompt is fed token by token through `step` to fill the cache, and each new token is one more `step`. The commit that introduced it is `14cb7d95f` (July 2026), "gpt: incremental GPU/CPU KV-cache decode step" (the crate was then called `gpt`). Its message records what the step needed that the other models' did not (learned positions added by a lookup, no rotary table, LayerNorm with a bias) and the comparison that tests it: feed ten tokens one at a time and compare every prefix's logits with `logits_all` on the same prefix. That test is `kv_step_matches_full_recompute_within_the_roundoff_bound` in `crates/gpt2/tests/it/kv_decode.rs`, and its tolerance is the one this chapter derives. In float32 the unit round-off is $2^{-24}$, and the formula of milestone 6, with the model's own shapes and weights, gives a bound for each logit (`logit_bounds`, built on `brain_testutil::rounding` and shared with the backend comparison of `cuda_backend_parity.rs`). For the test's model the bound is $1.4 \times 10^{-3}$ and the largest difference between the cached step and the recompute is $2.4 \times 10^{-7}$, so the bound is a ceiling thousands of times above what happens. That is the price of a bound that holds for every draw of the weights and does not have to be re-fitted to one; the commit's own run had reported 4.17e-7 on a GPU and 3.58e-7 on the CPU backend, the same order as the test's.

The second test shows the condition under which a greedy comparison is a fair one. `greedy_kv_generation_equals_recompute_with_a_margin_the_bound_needs` generates 16 tokens with and without the cache and asserts that they are equal, and then asserts that it is entitled to: at every one of the 16 steps the best logit leads the runner-up by more than twice the bound, since each of the two logits may differ between the paths by the bound. The weights are drawn from a fixed seed so that the report reproduces, but the draw is not trusted to have no near tie; the margin is computed, and on the fixed draw the smallest lead is $5.6 \times 10^{-3}$ against a requirement of $2.4 \times 10^{-3}$. That is milestone 6's comparison of the tolerance with the smallest gap, done instead of assumed. The serving cache of the larger models is `gqa_decode_step` in `crates/model/src/block.rs`, introduced by commit `4d61547a3` (August 2026) with a test that calls it once per position and compares with the batched causal attention, the same equivalence at the level of one layer; Chapter 12 tells that story and the position contract of `seq_lens` that is the other form of milestone 6's mutation. What differs from this project is scale and execution, never the contract: the same prefill, the same decode step, the same equality to be proved.

## Summary

If every acceptance test passes, you have built and checked a language model from the bytes up. A byte-level tokenizer that returns every input, trained on a corpus that a rule and a seed regenerate, with a digest that pins its tie rule. A pipeline whose first batch is printed, in which a split by document leaves no validation document in training. A GPT whose forward pass is pinned by a digest and whose backward pass agrees with central differences to below $10^{-8}$. A training run whose numbers are stated to the digits that survive its own perturbations, and a checkpoint that resumes to the byte. A sampler that is checked on ten numbers before it is trusted on a model and that draws one uniform number per token. A cache that matches the uncached model bit for bit in one order of summing and within a derived tolerance in another, over every split of a text, with a test that is shown to fail on the position error that breaks real caches. And a work report in FLOPs, bytes and fractions of a roofline.

What the project established, in order. A tokenizer is a function with a specification, and the specification includes its ties. A pipeline is correct when its first batch can be read. A gradient is believed when it agrees with the definition of a derivative at coordinates nobody chose after looking. A trained number is printed with the digits that its own noise leaves, and a resume is tested by bytes. A sampler's order of steps is part of its definition. A cache is correct when it agrees with the reference at every split and the tolerance of "agrees" has a derivation, and a test of a cache is evidence only if it is shown to fail on a plausible defect. And what a model costs is a count of FLOPs and bytes that a pencil reproduces.

Where this stops. The model is 6,288 parameters trained for 300 steps on a corpus of 240 documents, so it learns a grammar and not the dependency between sentences, and its generations show it. The corpus is a rule, not language; its tokenizer ends with every word a single token, which no tokenizer of real text does. The cache is checked in double precision, where a tolerance is a few units of $u$; in float32, with quantized weights or with several sequences in one step, the floor is higher, and Chapter 12 showed what the floor of a real stack looks like. Nothing here is faster than the reference: the project measures work and never time. The next volume starts from this model. Its first chapter lists what separates it from the decoders of today (grouped-query attention, normalized queries and keys, an untied head and rotary positions), each a change of the shapes that milestone 7 counts, and the cache-equivalence test of milestone 6 is the one that every such change must keep passing.

## Where to take the model next

These are optional. Each is a bigger test of the same ideas.

1. **Rotary positions.** Replace the position table by the rotary embedding of Chapter 8, applied to queries and keys inside every layer. Decide whether the cache stores keys before or after the rotation (Chapter 12), plant each wrong choice as a mutation, and show that the test of milestone 6 catches the ones that are wrong and that its tolerance is unchanged.
2. **A model that copies.** The loss of milestone 4 stops at the level of a model that copies nothing. Find, by experiment, a change of the training run, the row length or the model that makes the copy rates of milestone 5 rise well above chance, and report the step at which the validation loss leaves the plateau. The plateau level is computable in advance; say what you predicted before you ran it.
3. **A larger corpus.** Generate 10 times more documents from the same rule and a larger set of lists, retrain the tokenizer with a vocabulary that no longer makes every word one token, and repeat milestones 1 and 2 with their checks. Which of the digests of the milestones depend on the corpus, and which only on the specification?
4. **An encoder.** Use the model's blocks without the causal mask and train it to predict masked tokens (Chapter 14) on the same stream. State which parts of the pipeline you reuse unchanged and which of the milestone checks no longer applies.
5. **Flash-style attention.** Replace the attention of the model by the tiled computation of Chapter 13, and show that the forward logits digest of milestone 3 is unchanged to the scaling of the digest and that the cache-equivalence tolerance is, if it must be, a different one. Derive the new tolerance from the new order of summation.
6. **A float32 cache.** Store the keys and values in float32 and run the cache test against a float64 reference. Derive the tolerance from the unit round-off of float32 by the formula of milestone 6, find the smallest gap between best and second-best logit on your text, and say whether the exact token match still follows.

## Answers

**Recall questions.**

1. The pair $(a, b)$ occurs twice in each `abab` (counted with the word's count 2: 4), once in `ab` (1) and once in each of the three `abb` (3): 8 in all, against 2 for $(b, a)$ and 3 for $(b, b)$. It is merged first and `abab`, `ab`, `abb` become $[ab, ab]$, $[ab]$, $[ab, b]$. The pair $(ab, b)$ now has a count of 3 and $(ab, ab)$ a count of 2, so $(ab, b)$ is merged second and `abb` is written as one symbol.
2. The 100 ids have 99 targets, so there are $\lceil 99 / 16 \rceil = 7$ rows. Six rows hold 96 targets and the seventh holds the remaining 3, so 13 positions of the last row are padding, with a loss mask of 0.
3. Sorted by probability the tokens are 1 (0.50), 3 (0.30), 0 (0.10), 4 (0.06), 2 (0.04). The cumulative sums 0.50, 0.80, 0.90 first reach 0.85 at the third token, so the kept set is $\{1, 3, 0\}$ with total mass 0.90 and renormalized probabilities $0.556, 0.333, 0.111$. Walking the ids in increasing order, token 0 has cumulative probability 0.111, which does not exceed $u = 0.5$, and token 1 has 0.667, which does: the draw is token 1.
4. The cache holds three tokens, so the fed token gets position 3, the cache length. After its own key and value are appended its attention reads 4 keys. One token of context needs $2 \cdot 2 \cdot 16 \cdot 8 = 512$ bytes: two arrays (keys and values), two layers, 16 numbers, 8 bytes each.
5. The formula's error is of order $h^2$ times a third derivative, here about $10^{-10}$ for $h = 10^{-5}$; rounding in the two loss evaluations contributes about $\epsilon / h$ with $\epsilon \approx 10^{-16}$, here about $10^{-11}$. The one-sided difference makes an error of order $h$ times a second derivative, about $10^{-5}$, five orders worse at the same step, and its rounding is the same; so the central form is used.
6. $128 \cdot 128 = 16{,}384$ weights, two FLOPs each: 32,768 FLOPs. The weights are read once at 4 bytes: 65,536 bytes. The intensity is 0.5 FLOP per byte, and the attainable fraction of the peak is $0.5 / 50 = 0.01$: the device can reach one percent of its peak on this product, which is bandwidth bound.
