<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Capstone: From Raw Data to a Served Model

*Can you build the whole path yourself, in the language of your choice?*

Eight requests reach a small language-model server over the first eight iterations of its loop. The server has three slots, so it runs at most three conversations at once, and it fills a slot the moment one is free, which is continuous batching. Every request is the first sentence of a short document, such as `ben has a red box.`, and asks for the next sentence. The same trained model, asked the same eight questions offline, one at a time and with the whole context run through the network for every token, gives the answers in the fourth column below. The server in the fifth column gives the same eight. The server in the sixth column differs from it in one line, and three of its eight answers are not the model's:

<!-- worked: capstone.opening -->
The adapted model after both stages of training, eight requests (the first sentences of the first eight held-out fine-tuning documents), 3 slots, greedy decoding. 'Offline' is the reference answer, the whole context through the network at every token; the two servers run the cached forward pass with continuous batching, the first giving each new token the position equal to its own cache length and the second giving every decode row of an iteration (the step of one already running sequence) the position of the row in the lowest-numbered slot. A space is written `_`. The five runs of the stable-digits rule (a trained number is printed only to the digits that survive a nudge of the first weight, explained below) give the same tokens in every cell.

| request | arrives at iteration | at most tokens | offline answer | correct server | one position per batch |
|---:|---:|---:|---|---|---|
| 0 | 0 | 6 | `_a_cup_is_grey.` | same | same |
| 1 | 0 | 3 | `_a_hat_is` | same | same |
| 2 | 0 | 6 | `_a_key_is_grey.` | same | same |
| 3 | 1 | 6 | `_a_key_is_grey.` | same | `_a_a_ball_a_ball_is` |
| 4 | 3 | 4 | `_a_hat_is_red` | same | same |
| 5 | 4 | 6 | `_a_pen_is_red.` | same | `_a_pen_is_red_pen_pen` |
| 6 | 4 | 3 | `_a_hat_is` | same | `_a_hat_a` |
| 7 | 7 | 6 | `_a_box_is_red.` | same | same |

Agreement with the offline answers: 8 of 8 for the correct server, 5 of 8 for the other. The same faulty server agrees on 8 of 8 when each request is sent alone, and on 8 of 8 when all eight arrive at iteration 0 and it has 8 slots.
<!-- /worked -->

Nothing failed loudly. The faulty server produced text from the model's own vocabulary, finished every request within its limit, and passed the two tests such a server is usually given: each request sent alone, and all of them sent together. Its line gives every decode row of an iteration the position of the row in the lowest-numbered slot, a shortcut that is exact as long as all the rows started together, as they do in both tests. What found it was a third comparison: the answers the server gave under the arrivals it actually saw, set against the answers of the simplest program that computes them. Made at every seam and not only at the end, that comparison is why the build below has two halves: a system of stages, each fast or durable in its own way, and a reference that computes the same final numbers by the simplest path, against which every stage is compared where it hands its result to the next.

An **end-to-end machine learning system** is the whole path from raw data to answers served to clients, built as stages that pass each other only artifacts whose bytes are named and checked (the cleaned data, the tokenizer, the trained weights, the adapter, a manifest), together with a reference that computes the same final answers by the simplest path, so that a fault at any stage, or at any seam between two stages, shows as a difference that can be traced to where it entered. Each stage of this one is something an earlier volume built and tested on its own. What the build adds is the joining, and the joining is where every planted mistake below sits.

Five ideas from earlier volumes carry the build, and each is worth bringing back before the stage that needs it. In the gradient check of Volume III, Chapter 6 ("Proving Gradients Correct"), what is compared with what, why does its criterion have an absolute part and a relative part, and why is a transposed product found by its shape on one matrix and only by its numbers on another? A low-rank adapter of Volume XIII, Chapter 4 ("Parameter-Efficient Fine-Tuning") adds $s\,\mathbf{B}\mathbf{A}$ to a frozen weight: which factor starts at zero, and what does the adapted network compute before the first update? A transactional checkpoint of Volume XIII, Chapter 7 ("Checkpoints, Determinism and Recovery") is published in one atomic step: which part is written last, what does a reader check before it accepts a generation, and what besides the weights must a resumed Adam read? Continuous batching, in Volume XIV, Chapter 4 ("Batching and Scheduling Requests"), lets a sequence join and leave the batch between iterations: what does a static batch do with a slot whose sequence has ended, and what does that cost in reads of the weights? And in the layered architecture of Chapter 1 ("The Architecture of a Machine Learning Engine"), what may a part on one layer depend on, and why does the rule make a part replaceable?

## Three kinds of agreement, and the rules behind each

A comparison is only as sharp as the rules it is made under, and this build uses three kinds. Where two computations perform the same operations in the same order, they must give the same bits: a run that is killed and resumed on the same machine is compared with the run that was never killed, 64-bit pattern by 64-bit pattern. Where they perform different operations that should agree in exact arithmetic, they are compared by a criterion stated in advance: a gradient against central differences (below $10^{-7}$ after division by $\max(1, |\text{analytic}|)$), a changed run against its reference (below $10^{-9}$ at the largest entry). And where the output is a choice, the comparison is of the choices: the served tokens of a request against the tokens of the reference, all or nothing. The last is the one a user sees, and it is the coarsest: one of the planted mistakes below moves every trained number and changes no answer.

Everything is computed in 64-bit floating point (IEEE 754 binary64), each operation rounded on its own, with no fused multiply-add, and every sum in the order the text gives it, from `0.0`. The functions the digits depend on are the library's `exp` (in the softmax and the SwiGLU gate), `log` (in the cross-entropy and in `normal()`), `sqrt` (in the norms, in Adam and in `normal()`), and `sin` and `cos` (in `normal()`, the Box-Muller rule of Volume I, Chapter 17); an implementation whose `exp` differs in the last bit trains a network that differs in the last bits, which is why every trained number below is printed only to the digits that survive the stable-digits rule of the book (the run and four copies whose first weight is moved by $\pm 10^{-12}$ and $\pm 3 \times 10^{-12}$ must agree on a digit before it is printed). Every random input comes from the book's SplitMix64 generator (Volume I, Chapter 17), each generator started once at the seed `seed_of(name)` printed beside its table and never restarted; the order of the training rows is not a stream at all but `epoch_order` of Volume XIII, Chapter 1, a function of the seed and the index of the draw. Indices count from 0 (documents, rows, slots, requests, iterations of the server); updates count from 1.

You need, in your language: 64-bit floats with those library functions, the SplitMix64 generator with `uniform`, `integer` and `normal` exactly as Volume I specifies them, the 64-bit FNV-1a (Fowler-Noll-Vo) hash of Volume XIII, Chapter 7, a byte-level text encoding (UTF-8; every byte here is ASCII), and a way to write 64-bit floats and 32-bit unsigned integers as little-endian bytes. Nothing else is assumed: the tokenizer, the network, the optimizer, the checkpoint, the cache and the scheduler are all yours to write, and the earlier volumes say how.

![A three-row diagram. The top row is training: 200 raw documents with dirty case, spaces and duplicates, cleaned, deduplicated and split by digest, the tokenizer learned on the training text, the base run of 120 updates killed after 107 and resumed, and the adapter run of 60 updates killed after 47 and resumed. Each of the last three stages writes one file of the middle row (tokenizer.bin, base.bin, adapter.bin), beside manifest.json, which records their lengths, digests and the base the adapter was trained on. The bottom row is serving: the loader checks the files and rebuilds the network, the server with 3 slots runs continuous batching over a cached forward pass on 8 requests arriving over iterations 0 to 7, and its 8 answers are compared, token by token, with the answers of a reference that trains straight and computes its answers offline.](img/capstone-path.svg)

The module that produced every number below, `scripts/book/worked/capstone.py`, imports what the earlier volumes already wrote rather than copying it: the cleaning rule and the epoch order of Volume XIII, Chapter 1 (`training_data`), the tokenizer, the network with its forward and backward passes and the one-document rows of Volume V, Chapter 10 (`training_gpt`), the cached forward pass of Volume V, Chapter 12 (`kv_cache.extend`), the adapter's merge and gradient of Volume XIII, Chapter 4 (`peft`), the digest of Volume XIII, Chapter 7 and the Adam step, generations (checkpoint directories published whole) and reader of the Volume XIII project (`project_scaling_training`). It writes the cleaning of a raw corpus, the artifacts, the loader, the server and the comparisons.

## Layers that meet only through files

The build is divided before any of it is written, because the division decides which comparisons are possible. The layered architecture of Chapter 1 applies to a program of a few hundred lines as it does to an engine, and here it has four layers. At the bottom are the arithmetic and the network: the forward pass, the backward pass and the cached forward pass, which know nothing of where data comes from or where answers go. Above them sit two parts that never import each other. Training reads text, learns the tokenizer, orders the rows, runs the optimizer and writes generations; serving loads files, keeps caches and schedules requests. At the top a driver runs the two and the comparisons.

| part | what it holds | may use |
|---|---|---|
| arithmetic and network | matrix products, the norm, attention, the forward and backward passes, `extend` | nothing above it |
| training | cleaning, the tokenizer learner, rows, epoch order, Adam, generations | arithmetic and network |
| serving | the loader, the caches, the scheduler, decoding | arithmetic and network |
| driver | the stages in order, the reference, every comparison | all of the above |

Training and serving share one network implementation and nothing else. In particular the server never sees the tokenizer object or the weights the trainer held in memory: it receives four files and rebuilds everything from their bytes. That is a constraint the build accepts on purpose. A server that borrows the trainer's tokenizer object cannot be shown to work on a machine where only the files arrive, and the check that the tokenizer it uses is the one the base was trained with has nothing to check. Every seam that crosses from training to serving is therefore a file with a length and a digest, and every stage below ends with the bytes it hands on.

## From two hundred dirty lines to two clean splits

Raw text arrives the way people write it: with capital letters where they felt like them, two spaces after a full stop, stray spaces at the ends, and the same document many times. The corpus here is generated by a rule short enough to print, so that the reader can generate it too, and its dirt is drawn with it.

<!-- worked: capstone.raw_and_clean -->
The generator `capstone.raw` (seed 14314482791355729068) writes 200 raw documents; for each, in this order, integer(6) picks a name of ['ana', 'ben', 'cy', 'dee', 'eli', 'fay'], integer(5) a colour of ['red', 'blue', 'green', 'gold', 'grey'], integer(6) a thing of ['ball', 'cup', 'hat', 'box', 'pen', 'key'], integer(3) the case (0 as written, 1 the first letter upper case, 2 every letter upper case) and integer(3) the spacing (0 single spaces, 1 two spaces after the first full stop, 2 one space before the text and two after it). The text before the dirt is '<name> has a <colour> <thing>. the <thing> is <colour>.'. Cleaning lower-cases the text, splits it at runs of white space and joins the pieces with single spaces; a cleaned text equal to an earlier one is dropped; a cleaned text is held out when the 64-bit FNV-1a digest of its UTF-8 bytes is 0 modulo 5. A space is written `_` in the table.

| raw document | as drawn | cleaned | split |
|---:|---|---|---|
| 0 | `_Fay_has_a_green_ball._the_ball_is_green.__` | `fay_has_a_green_ball._the_ball_is_green.` | training |
| 1 | `cy_has_a_red_cup.__the_cup_is_red.` | `cy_has_a_red_cup._the_cup_is_red.` | training |
| 2 | `_Fay_has_a_gold_ball._the_ball_is_gold.__` | `fay_has_a_gold_ball._the_ball_is_gold.` | training |
| 3 | `Ana_has_a_green_pen._the_pen_is_green.` | `ana_has_a_green_pen._the_pen_is_green.` | training |
| 4 | `_ana_has_a_grey_cup._the_cup_is_grey.__` | `ana_has_a_grey_cup._the_cup_is_grey.` | training |
| 5 | `_Ben_has_a_green_ball._the_ball_is_green.__` | `ben_has_a_green_ball._the_ball_is_green.` | held out |

| corpus | raw documents | duplicates dropped | training | held out |
|---|---:|---:|---:|---:|
| base (`capstone.raw`, seed 14314482791355729068) | 200 | 80 | 97 | 23 |
| fine-tuning (`capstone.raw_ft`, seed 4230107496116625141, second sentence 'a <thing> is <colour>.') | 200 | 77 | 91 | 32 |

The 97 cleaned training documents of the base corpus, joined by one newline byte (10) in their order, have the FNV-1a digest f08a9f5abcc8df0a. No held-out document is also a training document, in either corpus, since a document's split is a function of its text. The planted mistake holds out every fifth raw document (raw index modulo 5 equal to 4) first and cleans and deduplicates each part after: 38 distinct held-out texts, of which 28 are also training texts.
<!-- /worked -->

The cleaning rule is the `normalise` of Volume XIII, Chapter 1: lower-case, cut at runs of white space, join with single spaces. It maps all the dirty spellings of one document to one text, and the 200 raw documents become 120 distinct ones: 80 were copies of an earlier document under different dirt. The split is the decision that matters most. A held-out document must not occur among the training documents, or every held-out number measures memory and not generalization. The rule here decides a document's split by a function of its cleaned text alone, the digest modulo 5, so that two copies of one document always land on the same side, and it is applied after the copies are removed. The planted mistake does what a quick script does: it holds out every fifth raw line first and cleans each part after. Of its 38 distinct held-out texts, 28 are also training texts, so 74 percent of its held-out set is memorised data.

```text
function clean_and_split(raw: list<text>, k: int) -> (list<text>, list<text>):
    seen = map of text to bool, empty
    train = []
    held = []
    for r in raw:
        d = normalise(r)
        if not (d in seen):
            seen[d] = true
            if fnv1a64(bytes of d) mod k == 0:
                append d to held
            else:
                append d to train
    return (train, held)
```

The cleaning stage is right when your 200 raw documents reproduce the six printed rows, your counts are those of the second table for both corpora, your training list has the printed digest, and the planted index split gives 38 and 28.

## A tokenizer that belongs to the base

The byte-pair tokenizer of Volume V, Chapter 2 is learned on the training documents of the base corpus and nothing else, so that no held-out text shapes the vocabulary. With the compact id space of Volume V, Chapter 10 it gives every word of this corpus one token, and every document twelve tokens with its separator.

<!-- worked: capstone.tokenizer_stage -->
The byte-pair learner of Volume V, Chapter 2, asked for 80 merges on the 97 cleaned training documents of the base corpus, stops after 48 (no adjacent pair is left in any word). The compact vocabulary of Volume V, Chapter 10 has id 0 for the separator `EOT`, then the 22 bytes that occur in the training text in increasing order, then the merges in the order learned: V = 1 + 22 + 48 = 71. Ids 23 to 70 are the merges (`_h`, `_ha`, `_g`, `re`, `_b`, `_a`, `_i`, `_t`, `he`, `_has`, `_is`, `_the`, `en`, `_gre`, `_grey`, `_green`, `al`, `lu`, `_bal`, `_blu`, `_ball`, `_blue`, `_c`, `up`, `_cup`, `ld`, `old`, `_gold`, `_re`, `_red`, `_p`, `ox`, `_box`, `_pen`, `_hat`, `_k`, `ey`, `_key`, `an`, `ana`, `ben`, `de`, `dee`, `ay`, `fay`, `cy`, `el`, `eli`; a space is written `_`).

| document | text | tokens | ids |
|---|---|---|---|
| base, training document 0 | `fay_has_a_green_ball._the_ball_is_green.` | `fay` `_has` `_a` `_green` `_ball` `.` `_the` `_ball` `_is` `_green` `.` | 67 32 28 38 43 2 34 43 33 38 2 |
| fine-tuning, training document 0 | `cy_has_a_red_ball._a_ball_is_red.` | `cy` `_has` `_a` `_red` `_ball` `.` `_a` `_ball` `_is` `_red` `.` | 68 32 28 52 43 2 28 43 33 52 2 |

Every base document is 12 tokens with its separator and every fine-tuning document 12, so one row of 12 positions holds any of them. The artifact `tokenizer.bin` (a u32 count of bytes, the bytes one u8 each, a u32 count of merges, each merge as two u32 byte-pair ids (a byte is its value 0 to 255, merge number r is 256 + r), all little-endian) is 414 bytes with the FNV-1a digest fdd07d442f56898b. The tokenizer rebuilt from those bytes alone gives the same ids as the one that learned them, and decodes them back to the text, for 243 of the 243 cleaned documents of both corpora.
<!-- /worked -->

The tokenizer is not a preprocessing step that can be redone at will. The ids it assigns are the rows of the embedding table the network learns, so the tokenizer and the base are one artifact in two files: a server that rebuilds a tokenizer with the same algorithm on slightly different text gets other ids for the same words and feeds the network rows it never trained. The fine-tuning corpus is encoded with the base's tokenizer for the same reason, and it was chosen so that every one of its words is already a token. A word the base never saw would still encode, as bytes and partial merges, but it would arrive as tokens whose embedding rows were never trained to stand for it, and an adapter on two matrices could not repair that.

The tokenizer is handed on as `tokenizer.bin`, a byte layout simple enough to write in any language, and the round trip of the block above, over every document of both corpora, is the test of the artifact.

## The network, its adapters and two gradient checks

The network is the small GPT (generative pre-trained transformer) of Volume V, Chapter 10, two pre-norm blocks wide enough for this grammar, and its initial weights are drawn ten times larger than that chapter's, because the copy that the second sentence needs (the thing and the colour of the first) is learned much sooner from that start. The adapters of the second stage sit on the two matrices that write into the residual stream, $\mathbf{W}_O$ and $\mathbf{W}_{\mathrm{down}}$ of both blocks.

<!-- worked: capstone.model_and_check -->
The GPT of Volume V, Chapter 10 with V = 71, D = 16, H = 2, F = 24, L = 2 and T = 12: token embedding plus learned position, 2 pre-norm blocks (RMS norm with epsilon 1e-6, causal attention, SwiGLU), a final RMS norm and a head tied to E; 5,760 numbers. `init_gpt` of Volume V, Chapter 14 with every 0.02 replaced by 0.2: the generator `capstone.init` (seed 3515344130392400139) draws normal() for every entry, row by row, in the order E, P, then for each block W_Q, W_K, W_V, W_O, W_gate, W_up, W_down, each entry times 0.2 and W_O and W_down also times 1/sqrt(2 L) = 0.5; the gains start at 1.0 and take no draw. The flat vector holds the tensors in the order E, P, then for each block W_Q, W_K, W_V, W_O, g1, g2, W_gate, W_up, W_down, then g_f, each row by row; written as 64-bit floats it has the FNV-1a digest 8ea9d2ddb72c09b0, and its first three numbers are 0.07742699, 0.06172680, 0.37532556.

| tensor | shape (as stored, [in, out]) | numbers | what it is |
|---|---|---:|---|
| E | [71, 16] | 1,136 | token embedding, also the head (logits = x E^T) |
| P | [12, 16] | 192 | learned positions 0 to 11 |
| W_Q, W_K, W_V, W_O | 4 x [16, 16] | 1,024 per block | attention of one block, 2 heads of width 8 |
| g1, g2 | 2 x [1, 16] | 32 per block | gains of the block's two RMS norms (start at 1) |
| W_gate, W_up, W_down | 2 x [16, 24], [24, 16] | 1,152 per block | SwiGLU network of one block |
| g_f | [1, 16] | 16 | gain of the final RMS norm (starts at 1) |

The adapters (rank r = 2, alpha = 2, scale s = alpha / r = 1) sit on W_O and W_down of both blocks. A has shape [r, in] and B [out, r] in the orientation of Volume XIII, Chapter 4, so the stored matrix becomes W0 + s (B A)^T. The generator `capstone.adapters` (seed 5940399326676008139) draws, site by site in the order of the table, normal() / sqrt(in) for every entry of A, row by row; B starts at 0.0. The adapter vector is A then B of each site, row-major: 288 numbers.

| site | A | B | numbers |
|---|---|---|---:|
| block 0, W_O | [2, 16] | [16, 2] | 64 |
| block 0, W_down | [2, 24] | [16, 2] | 80 |
| block 1, W_O | [2, 16] | [16, 2] | 64 |
| block 1, W_down | [2, 24] | [16, 2] | 80 |

Gradient checks, by central differences of step 1e-05 on the loss of one row (the first draw of the first epoch of the order `capstone.order`), the deviation being |numeric - analytic| / max(1, |analytic|), criterion below 1e-7. The base: 40 coordinates of the initial network, drawn from `capstone.gradcheck` (seed 3402741513478348829) as integer(21) for the tensor (in the order of the flat vector), then integer(rows) and integer(columns) of that tensor (a tensor of one row, a gain, still draws integer(1)): passes (below 1e-7). The adapters: all 288 adapter numbers on the initial network, with A drawn as above and B drawn, site by site and row by row, as normal() / 10 from `capstone.gradcheck_b` (seed 7136145745017463899) (a B of zero would make the gradient of A zero and test nothing): passes (below 1e-7). The planted mistake forms (dA, dB) from G instead of G^T: block 0 W_O: largest deviation 0.29, FAILS; block 0 W_down: the shapes do not agree, so it cannot be formed; block 1 W_O: largest deviation 0.25, FAILS; block 1 W_down: the shapes do not agree, so it cannot be formed.
<!-- /worked -->

The second check is the one that the joining needs. The network stores each matrix as $[\text{in}, \text{out}]$ and multiplies row vectors by it, $\mathbf{y} = \mathbf{x}\mathbf{W}$; the adapter of Volume XIII, Chapter 4 is written for the orientation of Volume III, $[\text{out}, \text{in}]$, acting on column vectors. Both are correct, and the seam between them is a transpose. The merged weight the network uses is

$$\mathbf{W} = \mathbf{W}_0 + s\,(\mathbf{B}\mathbf{A})^{\top},$$

and if $\mathbf{G}$ is the gradient of the loss with respect to the stored $\mathbf{W}$, the gradient with respect to the weight in the adapter's orientation is $\mathbf{G}^{\top}$, so the adapter's gradients are

$$\frac{\partial \mathcal{L}}{\partial \mathbf{A}} = s\,\mathbf{B}^{\top}\mathbf{G}^{\top}, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{B}} = s\,\mathbf{G}^{\top}\mathbf{A}^{\top},$$

the formulas of Volume XIII, Chapter 4 with $\mathbf{G}^{\top}$ in place of its $d\mathbf{W}$. Forgetting the transpose is the planted mistake, and it shows what a gradient check can and cannot see. On $\mathbf{W}_{\mathrm{down}}$, stored $[24, 16]$, the shapes do not agree and the mistake cannot even be computed; on $\mathbf{W}_O$, $[16, 16]$, the product exists, has the right shape, and is wrong, by 0.29 and 0.25 at the largest entry of the two blocks. A test that only checks shapes passes the square matrix, which is why the check compares numbers.

```text
function adapter_gradient(G: array<real>[n_in, n_out], A: array<real>[r, n_in],
                          B: array<real>[n_out, r], s: real)
        -> (array<real>[r, n_in], array<real>[n_out, r]):
    return lora_gradients(transpose(G), A, B, s)

function merged_weight(W0: array<real>[n_in, n_out], A: array<real>[r, n_in],
                       B: array<real>[n_out, r], s: real) -> array<real>[n_in, n_out]:
    D = transpose(matmul(B, A))
    return [[W0[i, o] + s * D[i, o] for o in 0 .. n_out] for i in 0 .. n_in]
```

The routine `lora_gradients` is that of Volume XIII, Chapter 4. The network's stage is right when your initial vector has the printed digest and first three numbers, and both checks pass with the coordinates and the B drawn as stated (and the transposed mistake fails on the two square sites).

## Two training runs that can be killed

The base run teaches the grammar: 120 updates of 8 rows, one document to a row, so that the positions of a document in training are the positions it has when it is served (a row that packed several documents would teach the network to read a document that starts in the middle of a row, which no request ever is). The adapter run then teaches the new form of the second sentence, `a <thing> is <colour>.` in place of `the <thing> is <colour>.`, with 60 updates of the 288 adapter numbers on the frozen base. Each run writes a generation every few updates, with the commit of the Volume XIII project: values and state into a temporary directory, the meta file last, and the directory renamed to `gen-<update>`. The meta records the digest of what the run continues, the tokenizer for the base run and the base for the adapter run, and a reader refuses a generation that belongs to another.

```text
function run_stage(st: inout Run, until: int, rows: list<Row>, seed: u64, lr: real,
                   grad: function(list<real>, list<Row>) -> list<real>,
                   every: int, owner: text, disk: inout map<text, bytes>) -> none:
    while st.t < until:
        batch = [rows[draw(seed, len(rows), st.d + b)] for b in 0 .. 8]
        g = grad(st.p, batch)
        st.t = st.t + 1
        st.d = st.d + 8
        adam_update(st.p, g, st.m, st.v, st.t, lr)
        if st.t mod every == 0:
            commit(st, owner, disk)
```

`Run` is the record `(p, m, v, t, d)` of the Volume XIII project's `AdapterRun`, here holding either the 5,760 numbers of the network or the 288 of the adapters; `draw` is the routine of Volume XIII, Chapter 7, `adam_update` that of Volume XIII, Chapter 4 and `commit` that of the Volume XIII project with the owner's digest in place of the base's. For the base run `grad` is the gradient of the network's mean loss over the supervised positions of the batch, flattened in the order of the vector; for the adapter run it is that gradient taken at the merged network and passed through `adapter_gradient` site by site.

<!-- worked: capstone.training_stages -->
The base run: 120 updates, each of 8 draws, draw d being entry d mod n of `training_data.epoch_order` of epoch d // n over the n = 97 training rows, with the seed 10808073266906533035 (seed_of 'capstone.order'). A row is one document (the separator, then its tokens; targets shifted by one, the last target the closing separator, padding beyond with loss mask 0). The loss of an update is the mean cross-entropy (natural log) over the supervised positions of its 8 rows together, its gradient that of `training_gpt.batch_loss`, and the update is `adam_update` of Volume XIII, Chapter 4 at the rate 0.01 (bias corrections by repeated multiplication from 1.0, no decay, no clip, constant rate). The held-out loss is the same pooled mean over the 23 held-out rows; an answer is right when the greedy continuation of the first sentence and its full stop (the reference rule: the whole context through the network at every token, the largest logit, the lowest id on a tie; EOT ends the answer and is not kept, and so does a full row of 12, whose logits still choose a last token that is kept) decodes to the held-out second sentence. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation. All five runs agree on every count of answers (the resume without moments included: each run is also resumed without its moments from its own generation 100).

| updates | held-out loss (nats per target token) | held-out answers right |
|---:|---:|---:|
| 0 | 4.311 | 0 of 23 |
| 20 | 1.266 | 3 of 23 |
| 40 | 0.8481 | 1 of 23 |
| 60 | 0.7393 | 8 of 23 |
| 80 | 0.6508 | 11 of 23 |
| 100 | 0.5512 | 14 of 23 |
| 120 | 0.4801 | 23 of 23 |

The adapter run on the frozen base after 120 updates: 60 updates of 8 draws from the 91 fine-tuning rows (seed 15192820369151416723 (seed_of 'capstone.order_ft')), the forward pass with every adapted matrix merged as W0 + s (B A)^T, the adapter gradient (dA, dB) = (s B^T G^T, s G^T A^T) from the gradient G of the stored matrix, Adam at the rate 0.03 on the 288 adapter numbers only. 'New form' is the held-out fine-tuning corpus ('... a <thing> is <colour>.'), 'old form' the held-out base corpus ('... the <thing> is <colour>.').

| updates | held-out loss on the new form | new-form answers right | old-form answers right |
|---:|---:|---:|---:|
| 0 | 2.123 | 0 of 32 | 23 of 23 |
| 10 | 0.7879 | 11 of 32 | 0 of 23 |
| 20 | 0.6635 | 16 of 32 | 0 of 23 |
| 30 | 0.5885 | 20 of 32 | 0 of 23 |
| 40 | 0.5585 | 23 of 32 | 0 of 23 |
| 50 | 0.5477 | 27 of 32 | 0 of 23 |
| 60 | 0.5339 | 31 of 32 | 0 of 23 |

After the adapter run the base's digest is unchanged (b28ea43b5debc50b). The base run with a generation every 20 updates (the commit of the Volume XIII project: `values.bin`, `state.bin`, `meta.json` last, then the rename; two newest generations kept; the meta's owner field is the digest of `tokenizer.bin`), killed after 107 updates, leaves gen-0100, gen-0080; the reader accepts gen-0100, and the second process repeats the updates from there to 120: its 5,760 numbers are the same 64-bit patterns as the run that was never killed. Resumed with the moments set to zero, it ends 0.36 from it at the largest entry (FAILS against 1e-9) and answers 23 of 23 held-out prompts right. The adapter run with a generation every 10 updates (owner field: the digest of the base), killed after 47 updates, leaves gen-0040, gen-0030; the reader accepts gen-0040, and the resumed adapter numbers are the same 64-bit patterns as the straight run's.
<!-- /worked -->

The base run's held-out loss falls from 4.311 nats per token, close to $\ln 71 = 4.263$ for a network that has not yet learned which tokens occur, to 0.4801, and the count of held-out documents whose second sentence it writes correctly rises unevenly, 3 of 23 after 20 updates, 1 after 40, then 8, 11, 14 and all 23 at the end. The floor of the loss is set by the grammar: the name, the colour and the thing are drawn uniformly, $\ln(6 \times 5 \times 6) = 5.19$ nats a document, which over the 12 targets of a document (the eleven tokens after the separator and the closing separator) is 0.433 nats per token, and the other targets are certain; the loss of 0.4801 is 0.047 above it. The adapter run moves the other way on the base form, as it should: the old form is right for 23 of 23 before the first update (the adapters start at $\mathbf{B} = \mathbf{0}$, so the network is the base) and for none after it, while the new form rises to 31 of its 32 held-out documents.

The two kills restore what they should, and the bits say so: the base run resumes from generation 100 and the adapter run from generation 40, each repeats seven updates, and both end with the same 64-bit patterns as the runs that were never stopped. The planted mistake of the base run is the instructive row. Resumed with its moments set to zero, the network ends 0.36 from the right one at its largest entry, and it still answers 23 of 23 held-out prompts correctly. An evaluation of answers cannot see this mistake; only the comparison of the trained numbers can, and the next stage is where that comparison is made into a rule.

## Four files and a loader that trusts none of them

Training ends by writing what serving needs, and nothing more: the tokenizer, the base, the adapter and a manifest that names the other three by length and digest. The manifest also records `trained_on`, the digest of the base that the adapter run's generations named as their owner, because an adapter is a change to one particular base and means nothing on another.

<!-- worked: capstone.artifacts -->
The four files that cross from training to serving, written from the resumed runs of both stages: `tokenizer.bin` (stage 2), `base.bin` (the 64-bit floats of the trained base in the order of the flat vector, little-endian), `adapter.bin` (the adapter vector, likewise) and `manifest.json`, whose text, keys sorted and no spaces, is one line; it is printed here cut after commas, and the file is these lines joined with nothing between them (no line break):

```text
{"adapter":{"alpha":2.0,"digest":"c97b4935cfd1d553","length":2304,
"rank":2,"sites":["Wo0","Wd0","Wo1","Wd1"],
"trained_on":"b28ea43b5debc50b"},"base":{"digest":"b28ea43b5debc50b",
"length":46080},"config":{"D":16,"FF":24,"H":2,"L":2,"T":12,"V":71},
"tokenizer":{"digest":"fdd07d442f56898b","length":414}}
```

| file | bytes | FNV-1a digest |
|---|---:|---|
| tokenizer.bin | 414 | fdd07d442f56898b |
| base.bin | 46,080 | b28ea43b5debc50b |
| adapter.bin | 2,304 | c97b4935cfd1d553 |
| manifest.json | 301 | bc1cbdf9b0199265 |

The digest of `tokenizer.bin` depends on no floating-point operation, and the lengths depend on none of the trained numbers. Those of `base.bin`, `adapter.bin` and `manifest.json` depend on the last bit of every trained number, which depends on the order of the sums of the backward pass and on how the library functions round, and neither is fixed by this book: the digests printed for them are those of the module's run, and an implementation checks its own files against its own manifest, built from its own three digests in the form above.

The loader reads the manifest, then accepts the files only if each of the three has the recorded length and digest, the adapter's `trained_on` equals the base's digest and the tokenizer's vocabulary equals the configuration's V; it then rebuilds the tokenizer from its bytes and the network by merging the adapter into the base.

| what the server is given | loader |
|---|---|
| the four files as written | accepted |
| tokenizer.bin rebuilt from the training and held-out documents of the base corpus | refused: tokenizer.bin is not the file the manifest names |
| base.bin from the run resumed without its moments, manifest rewritten for it | refused: the adapter was trained on another base |
| one bit of adapter.bin flipped (byte 100, bit 0) | refused: adapter.bin is not the file the manifest names |
<!-- /worked -->

The loader is written as if every file might be the wrong one, because each of the refused cases is a way a real deployment goes wrong: a tokenizer regenerated on a build machine, a base retrained while the adapter stayed, a disk or a copy that changed one bit. The check is one pass of the hash over the 48,798 bytes of the three named files, and it refuses all three substitutions before a request is answered.

```text
record Entry:
    length: int
    digest: u64

record Manifest:
    tokenizer: Entry
    base: Entry
    adapter: Entry
    trained_on: u64
    V: int

function accept(files: map<text, bytes>, meta: Manifest) -> bool:
    for (name, e) in [("tokenizer.bin", meta.tokenizer), ("base.bin", meta.base),
                      ("adapter.bin", meta.adapter)]:
        if not (name in files):
            return false
        if len(files[name]) != e.length or fnv1a64(files[name]) != e.digest:
            return false
    if meta.trained_on != meta.base.digest:
        return false
    return vocabulary(files["tokenizer.bin"]) == meta.V
```

The function `vocabulary` reads the two counts of the tokenizer's bytes and returns $1 + n_{\mathrm{bytes}} + n_{\mathrm{merges}}$. The digests are written in the manifest as sixteen lower-case hexadecimal digits and compared as 64-bit numbers. The loader stage is right when your four files have the printed lengths, your `tokenizer.bin` has the printed digest, your manifest has the printed form with the digests of your own three files (the manifest has the printed length 301 whatever the digests are, and equals the printed text wherever those digests equal the printed ones), and your loader gives the four verdicts of the second table.

## Eight requests on three slots

The server has one network, three slots and a queue. Each iteration of its loop is one pass in which every running sequence advances by one token: a request that has just been admitted is prefilled (its whole prompt runs through the network, filling its cache, and the last logits give its first token), and a sequence admitted earlier decodes one token on top of its cache, with `extend` of Volume V, Chapter 12. The only difference between the two policies is when a slot may be refilled. Under continuous batching, a slot whose sequence has ended takes the next request in the queue at the start of the next iteration. Under static batching, the server waits until all three sequences of a batch have ended and then admits the next three.

```text
record Request:
    id: int
    arrival: int
    prompt: list<int>
    limit: int

record Seq:
    id: int
    cache: KvCache
    out: list<int>
    limit: int
    admitted: int
    ended: bool

function live(x: Seq) -> bool:
    return (not x.ended) and len(x.out) < x.limit and x.cache.length < T

function serve(model, reqs: list<Request>, S: int, continuous: bool) -> map<int, list<int>>:
    queue = []
    slot = [nothing for s in 0 .. S]
    done = map of int to list<int>, empty
    k = 0
    while len(done) < len(reqs):
        for q in reqs:
            if q.arrival == k:
                append q to queue
        if continuous or every slot[s] is absent:
            for s in 0 .. S:
                if slot[s] is absent and len(queue) > 0:
                    q = queue[0]
                    queue = queue[1 : len(queue)]
                    c = KvCache(zeros(real, [L, T, D]), zeros(real, [L, T, D]), 0)
                    t = argmax(extend(model, c, q.prompt))
                    slot[s] = Seq(q.id, c, [] if t == EOT else [t], q.limit, k, t == EOT)
        for s in 0 .. S:
            if slot[s] is present and slot[s].admitted < k and live(slot[s]):
                t = argmax(extend(model, slot[s].cache, [slot[s].out[len(slot[s].out) - 1]]))
                if t == EOT:
                    slot[s].ended = true
                else:
                    append t to slot[s].out
        for s in 0 .. S:
            if slot[s] is present and not live(slot[s]) and not (slot[s].id in done):
                done[slot[s].id] = slot[s].out
            if continuous and slot[s] is present and slot[s].id in done:
                slot[s] = nothing
        if (not continuous) and every present slot[s] has slot[s].id in done:
            slot = [nothing for s in 0 .. S]
        k = k + 1
    return done
```

Every request here is a prompt of 7 tokens (the separator and the six tokens of a first sentence), so a sequence holds at most $12 - 7 = 5$ generated tokens in its cache and can choose a sixth, which is the separator that ends a full answer. The limits of the requests differ, as the limits clients send do: five of them allow 6 tokens, two allow 3 and one allows 4.

<!-- worked: capstone.schedules -->
The eight requests of the opening on the same server under both policies. Continuous: in each iteration the free slots are filled from the queue (first come, first served; the lowest free slot first) and each admitted request is prefilled in that iteration, which gives its first token; every sequence admitted earlier then decodes one token. Static: the server admits only when all 3 slots are free, and a slot whose sequence has ended stays empty until the whole batch has ended. A sequence ends at EOT (not counted as a token) or at its limit; its slot is free from the next iteration. Iterations count from 0. The five runs of the stable-digits rule give the same numbers in both tables.

| request | arrives | at most tokens | continuous: admitted | continuous: last token | tokens | static: admitted | static: last token |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 6 | 0 | 5 | 5 | 0 | 5 |
| 1 | 0 | 3 | 0 | 2 | 3 | 0 | 2 |
| 2 | 0 | 6 | 0 | 5 | 5 | 0 | 5 |
| 3 | 1 | 6 | 3 | 8 | 5 | 6 | 11 |
| 4 | 3 | 4 | 6 | 9 | 4 | 6 | 9 |
| 5 | 4 | 6 | 6 | 11 | 5 | 6 | 11 |
| 6 | 4 | 3 | 9 | 11 | 3 | 12 | 14 |
| 7 | 7 | 6 | 10 | 15 | 5 | 12 | 17 |

Work, counted by these rules: an iteration is one in which at least one row runs; a row is one token through the network (a prefill of 7 tokens is 7 rows); FLOPs (floating-point operations) are 2 for each multiply-add of the cached forward pass `extend` of Volume V, Chapter 12 (every matrix product, every attention score and every weighted sum of values); every iteration reads all 5,760 served weights once, 8 bytes each; a decode row at cache length c reads the c + 1 key rows and value rows of every layer (the new token's own included), 8 bytes a number. A request's wait is its admission iteration minus its arrival iteration, and its span is the iteration of its last token minus its arrival iteration, plus 1; the means are over the 8 requests, exact multiples of 1/8 printed to 3 decimals.

| policy | iterations | rows | tokens | FLOPs | weight bytes read | weight bytes per token | cache bytes read | mean wait | mean span | same as offline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| continuous | 16 | 88 | 35 | 925,312 | 737,280 | 21,065.1 | 159,232 | 1.875 | 6.875 | 8 of 8 |
| static | 18 | 88 | 35 | 925,312 | 829,440 | 23,698.3 | 159,232 | 2.875 | 7.875 | 8 of 8 |
<!-- /worked -->

![Two grids of three rows (slots) and one column per iteration of the server, each cell marked P j where request j is prefilled, D j where it decodes one token and (j) where it holds a slot without running. Under continuous batching requests 0, 1 and 2 are prefilled in iteration 0; request 1 ends in iteration 2 and request 3 takes its slot in iteration 3; requests 4 and 5 take the slots of requests 0 and 2 in iteration 6, request 6 is prefilled in iteration 9 and request 7 in iteration 10, whose last token comes in iteration 15: 16 iterations. Under static batching the slot of request 1 is held empty in iterations 3 to 5, requests 3, 4 and 5 are admitted together in iteration 6, the slot of request 4 is held in iterations 10 and 11, requests 6 and 7 are admitted in iteration 12, the slot of request 6 is held in iterations 15 to 17 and slot 2 stays empty: 18 iterations.](img/capstone-schedules.svg)

Both policies produce the same 35 tokens with the same 88 rows and the same 925,312 FLOPs, and both give eight answers equal to the reference's. They differ in what a FLOP cannot see. A decode iteration reads every weight of the network once, whatever the number of rows it carries, so the bytes of weights that the server reads are proportional to its iterations: 737,280 bytes in 16 iterations under continuous batching, 829,440 in 18 under static batching, 21,065.1 and 23,698.3 bytes for each generated token. At this size the 46,080 bytes of weights are read once an iteration and the cache reads are 159,232 bytes in all, so the weights are most of the traffic, as they are for a real model at small batch. Waiting is counted in iterations: a request waits a mean of 1.875 iterations for a slot under continuous batching and 2.875 under static batching, and its answer is complete a mean of 6.875 and 7.875 iterations after it arrived. Neither number is a time; each is a count of passes of the network, which is what the device does in either case.

The server is right when your schedule reproduces the admission and last-token iteration of every request under both policies, your counts of rows, tokens and FLOPs are those of the table (with FLOPs counted as `extend` counts them: 2 for each multiply-add of every matrix product, attention score and weighted sum), and your eight answers equal your reference's.

## The position of a token that joins late

The opening's faulty server is now one line of `serve`: the decode loop passes to `extend` not the cache length of its own sequence, $c$, but the cache length of the decode row in the lowest-numbered slot. With learned absolute positions (the table $\mathbf{P}$ of the network), position is part of every token's input, and a token at the wrong position is a different input. The two tests the server passed were exactly the two cases in which the shortcut is exact. A request sent alone is the only row, so its own length is the lowest slot's length. Eight requests sent together into eight slots all have prompts of 7 tokens and all advance one token an iteration, so every row has the same length at every iteration. In the opening's run the shortcut breaks as soon as a request joins a batch that started earlier: request 3 is admitted at iteration 3 into the slot that request 1 has left, beside requests 0 and 2, whose caches are three tokens longer, and its first decoded token is given position 10 where its own is 7.

The general lesson is about the tests and not about positions. A batching scheduler is correct when the answer of a request does not depend on what else the server was doing, and a test of that property must vary what else the server is doing: arrivals at different iterations, prompts of different lengths, limits that end sequences early, a slot refilled while others are in the middle of their answers. The reference that such a test compares with is the request alone, computed by the simplest program, and the comparison is token by token, so that a scheduler that is right for every arrival pattern but one fails on that one.

## The whole path against its reference

The reference is the shortest program that computes the answers: both stages trained straight in one process, each answer computed offline by running the whole context through the network for every token. The system is everything above: both stages killed and resumed from their generations, the four files written and checked, the network rebuilt from bytes and served under continuous batching with the opening's arrivals. Each planted mistake changes one seam.

<!-- worked: capstone.whole_path -->
The reference trains both stages straight in one process and computes each answer offline; the system kills each stage once (after 107 and 47 updates), resumes from the newest generation that verifies, writes the four files, and answers from what its loader rebuilt from them, under continuous batching with the arrivals of the opening. Each planted mistake changes one seam of the system. 'Trained numbers' compares the base and adapter numbers the server is given with the reference's (the bits, or the criterion 1e-9 at the largest entry); 'loader' is what the loader with its checks says; the third row's tokenizer is learned, with the same 80 merges, on the training and held-out documents of the base corpus (120 of them) and its ids are compared with the reference's ids; the last two columns are what the server answers when the files are loaded without the checks: answers equal to the reference's offline answers, and answers that are the held-out second sentence, among the 5 requests allowed 6 tokens. The five runs of the stable-digits rule give the same counts in every row (the resume without moments is taken from each run's own generation 100).

| build | trained numbers | loader | same as the reference | right answers |
|---|---|---|---:|---:|
| the reference: straight runs, the answers computed offline one request at a time | - | - | 8 of 8 | 5 of 5 |
| the system: both stages killed and resumed, four files, the loader, continuous batching | same bits | accepted | 8 of 8 | 5 of 5 |
| tokenizer rebuilt by the server from training and held-out documents | same bits | refused (tokenizer.bin) | 3 of 8 | 0 of 5 |
| base resumed without its moments, served with the adapter of the right base | FAILS (base) | refused (trained on another base) | 2 of 8 | 2 of 5 |
| the adapter not merged by the server | same bits | accepted | 0 of 8 | 0 of 5 |
| the system serving 8-bit weights (a tolerance, not a mistake) | same bits | accepted | 8 of 8 | 5 of 5 |
| one position for every decode row of an iteration | same bits | accepted | 5 of 8 | - |
<!-- /worked -->

The system's trained numbers are the reference's to the bit, its loader accepts the files, and its eight answers are the reference's. Every planted mistake is caught, and the table says by what. One is caught only by the loader: the tokenizer rebuilt on more text, whose files carry the same trained numbers, would agree with the reference on only 3 of 8 answers. The base resumed without its moments is caught twice, by the comparison of trained numbers and by the loader; served with an adapter trained on the right base without those checks it would agree on only 2 of 8, although that base on its own answers every held-out prompt correctly. One is caught only by the comparison of answers: a server that loads the adapter and does not merge it answers in the base's form, 0 of 8, and every check before serving passes, since the files are right and only the server's use of them is wrong. And the position shortcut of the opening is caught only by the comparison under arrivals that vary. No single check finds every mistake, and no mistake escapes all of them: the trained-number comparison, the loader and the comparison of answers each cover seams the others cannot see.

The quantized row is not a mistake. A server that stores its weights in 8 bits serves a different network, and its answers cannot be held to the reference's bits, only to its tokens, under a stated reason why they should agree:

<!-- worked: capstone.quantized_serving -->
The served network of the opening, and the same network with E, P and every matrix of both blocks stored as 8-bit codes: for each row of a stored matrix the scale is its largest magnitude divided by 127, the code is the entry divided by the scale and rounded half away from zero, and the network computes with code times scale; the gains stay in 64 bits. Both serve the eight requests of the opening under continuous batching. The margin is the largest logit minus the second largest of the 64-bit network at each of the 40 steps of the offline answers (the steps whose logits choose a token, the final EOT included); the change is the largest absolute difference between the two networks' logits at the same steps.

| served weights | bytes | same as offline | largest logit change |
|---|---:|---:|---:|
| 64-bit weights | 46,080 | 8 of 8 | - |
| 8-bit codes, one scale per row | 8,904 | 8 of 8 | 0.264 |

The smallest margin over those steps is 0.51 and the largest change 0.264, so the bound taken over all steps at once (the largest change below half of the smallest margin) does not hold. Step by step, the change is below half of that step's own margin at 39 of the 40 steps, and the largest logit of the 8-bit network is the reference's token at 40 of the 40 steps. The margin and the change are trained numbers, printed by the stable-digits rule; the counts are the same in all five runs.
<!-- /worked -->

The reason offered for such agreement is a margin. If, at every step of the reference answers, the largest logit exceeds the second by at least $m_{\mathrm{top}}$, and quantization moves no logit by as much as $m_{\mathrm{top}}/2$, then the largest logit stays the largest and greedy decoding chooses the same token, step after step. Here the bound fails when it is taken over all steps at once: the largest change, 0.264, is a little more than half of the smallest margin, 0.51. Taken step by step, each step's change against that step's own margin, it holds at 39 of the 40 steps, and at the remaining step the 8-bit network's largest logit is still the reference's token, which only the comparison of tokens shows. The bound is sufficient and not necessary, and the table reports both what it proves and what was only observed. That is the form every tolerant comparison of this build takes: a stated quantity, a stated bound, an argument that the bound implies the property that matters, and an honest count of the cases where the argument does not reach.

## What the whole path costs in arithmetic and bytes

Speed here is counted as work: the FLOPs a stage computes and the bytes it moves, never the seconds a machine took.

<!-- worked: capstone.work_bill -->
FLOPs are 2 for every multiply-add of a matrix product, the products as the code runs them. A forward pass of one row is 75,072 multiply-adds (for each block the 4 projections, T x D x D each, the scores and weighted sums of the 2 heads over all T x T pairs, masked ones included, and the 3 SwiGLU products; then the head, T x D x V), and the backward pass is counted as twice the forward; norms, softmax and the optimizer are not counted. An update of 8 rows is then 3 x 2 x 75,072 x 8 FLOPs, in both stages: the adapter run here forms the gradient of every stored matrix and keeps the part that reaches the adapters. A generation's values and state are 8 bytes for every trained number, its first and its second moment, and 16 bytes of counts.

| stage | updates or iterations | FLOPs per update | FLOPs | bytes moved |
|---|---:|---:|---:|---|
| base run | 120 | 3,603,456 | 432,414,720 | 6 generations of 138,256 bytes of values and state |
| adapter run | 60 | 3,603,456 | 216,207,360 | 6 generations of 6,928 bytes of values and state |
| serving the 8 requests | 16 | - | 925,312 | 737,280 bytes of weights read |
<!-- /worked -->

Training is almost all of the arithmetic: the base run's 432 million FLOPs and the adapter run's 216 million are about 700 times the 0.93 million of serving eight answers. An adapter update costs the same as a base update here because this implementation forms the gradient of every stored matrix and keeps only what reaches the adapters; a trainer that skips the weight gradients of frozen matrices saves most of the half of the backward pass that forms them (a third of an update, by the counting rule above, less the products the four adapted sites still need), but not the half that carries the gradient through them to the adapters below. The adapter's saving is in bytes: a generation of the adapter run is 6,928 bytes against the base run's 138,256, twenty times smaller, because the state that a resumed Adam reads is three numbers for every trained number and the adapter run trains 288 numbers instead of 5,760.

## The same path through brain's crates

The brain engine is far larger than this build, and its path from raw data to a served answer crosses the same seams, each guarded in the code by a check of the kind above.

Data enters training through `load_dataset` in `crates/model/src/train.rs`, which validates what it is given at the point of entry: a token split too short to draw a row from is refused with a message that names the split, its token count and the row length (a chat dataset that indexes its examples needs no window and takes the one-example-per-row path described below). That check came from an end-to-end run. Commit `d82f39ccb` (5 August 2026) records that a small dataset used to load without error and then fail deep in a later batch draw, with an index near $2^{64}$ produced by an unsigned subtraction that underflowed, and that it was found by running the whole fine-tuning command against a small dataset; the commit that follows, `3ced153d7`, adds that command and the test that runs it end to end against a real checkpoint (an ignored test, run only where the checkpoint's directory is given). The current code still refuses such a split, and its test (`load_dataset_rejects_a_train_split_shorter_than_block_size_instead_of_panicking_later`) builds exactly that dataset. The one-document rows of this build have their counterpart too. Commit `a9f38b586` (24 September 2026) made a row of a chat dataset one example, padded, after a measurement in which a held-out loss of 0.0353 nats per supervised token (mean cross-entropy) sat beside 25 percent of the same questions answered correctly when generated: packed rows let an example copy its neighbours' answers, which no request at serving time can do. Those two numbers could not both describe the same working model, and setting them side by side was a comparison between training's measure and serving's, the kind this build makes at every stage. In the current `load_dataset` an example longer than a row is refused rather than cut.

The crossing from training to serving is a content-addressed store in brain's served training jobs (`crates/trainjob/src/store.rs`, added in commit `3cc9eda11`, 6 November 2026): a dataset or an adapter is named by the digest of its bytes (BLAKE3, a cryptographic hash, where this build uses FNV-1a, which detects accidents but not an adversary), a name parses only from 64 hexadecimal digits so that it cannot spell a path, and `fetch` copies the file into the job's own directory while hashing it and deletes the copy and refuses it if the digest differs, so that what the job reads is the bytes that were checked. That is the loader of this build, with the check and the use made one operation.

Continuous batching in brain is one scheduler, `model::serve::Scheduler` in `crates/model/src/serve.rs`, generic over the `PagedDecoder` interface a model's engine implements; each iteration (`iterate`, which `step_inner` runs and wraps with its bookkeeping of the card's memory pressure) admits waiting requests while there is room, runs one batched decode step over the running sequences and then reaps the finished ones, the loop of `serve` above. Its contract is a conformance suite, `crates/model/src/serve/conformance.rs`, added in commit `8a708e7e7` (19 October 2026) with a gate (`scripts/gates/check-serve-conformance.sh`) that fails any engine implementing the interface that neither runs the suite nor states why it is exempt. Its first property, `batched_output_equals_single_request_output`, submits four prompts of 5, 9, 17 and 25 tokens and requires each to decode exactly as it does alone, the comparison of this build's opening, with prompts whose different lengths put the rows of one batch at different positions from the first iteration; another, `adapters_never_share_kv_with_the_base`, requires that keys and values computed under an adapter are never served to the base or to another adapter. The comparisons are exact (`assert_eq!` on the token streams), and the suite states why: a tolerant comparison would hide a step that really diverges, so a decoder whose speculative verification step adds in another order than plain decoding builds its test model with a clear margin between its first and second token instead, the margin argument of the quantized row above.

The layering that keeps training and serving apart in this build is enforced in brain by the layer map of Chapter 1 (`scripts/gates/crate-layers.map`, checked by `scripts/gates/check-crate-layers.sh`): the training substrate (`model`, `optim`, `data`, `checkpoint`) is on layer 3, the served training job and the server on layer 6, and no dependency may point up.

## Larger versions of the same path

The build is small so that every number fits on a page, and each of its seams survives a change of scale. The serving cache can be paged, in blocks of positions, as Volume XIV pages it, and the comparison with the reference must still be exact, which is the test that a block table is right. The scheduler can take prompts of different lengths, cut long prompts into pieces between decode steps, and run several adapters beside the base, each with its own caches; the opening's comparison, varied over arrivals, is the test of each. The training can move to several ranks as in the Volume XIII project, with the tolerance of $10^{-9}$ in place of the bits. The base can be stored in 4 bits, where the margin argument may fail at some steps, and then the honest statement is the number of answers that differ, each traced to a step whose margin was smaller than the change. And the corpus can be real text, where no reference answer is known in advance and the reference is the slow program itself.

## Check your understanding

1. The faulty server of the opening passes the test that sends each request alone and the test that sends all eight at once into eight slots. Explain why the shortcut is exact in both, and design the smallest test that fails it.
2. The base run resumed without its moments ends 0.36 from the reference at its largest entry and answers every held-out prompt correctly. A colleague concludes that the moments do not matter for this model. Find the error.
3. The planted split holds out every fifth raw line and cleans each part afterwards. Why does cleaning before splitting not repair it if the split is still by raw index, and why does the digest rule make the order of cleaning and splitting irrelevant for copies?
4. A colleague's forward pass applies the adapter's scale twice, $s^2\,\mathbf{B}\mathbf{A}$, while the backward pass uses $s$ once. Predict what the gradient check of the adapters reports with the $s = 1$ of this build, and with $s = 0.5$, and say what the prediction means for the choice of test values.
5. A server loads the four files with every check passing and serves answers in the old form. Which seam is wrong, and why can no check before serving detect it?
6. Static batching produces the same 35 tokens with the same 925,312 FLOPs as continuous batching. In what sense is it slower, and which quantity in the table shows it?
7. The 8-bit server's answers equal the reference's. Is that a proof that quantization is harmless for this model? State what the margin argument proves and what it does not.

## Summary

If every stage passes, you have built the whole path: a raw corpus cleaned and split so that no held-out document is also a training document; a tokenizer that travels with its base as bytes; a network whose two gradients, of the weights and of the adapters in another orientation, agree with central differences; a base run and an adapter run that a kill turns into a pause, resumed to the bit from generations that name what they continue; four files and a loader that refuses each of three realistic substitutions; and a server whose answers under continuous batching, with requests arriving and ending at different iterations, are the reference's, token for token, while reading fewer bytes of weights per token than static batching. The planted mistakes show why the comparisons are three: the trained numbers see a lost optimizer state that no answer shows, the loader sees a wrong tokenizer or base before a request is served, and only the comparison of answers under varied arrivals sees a server that uses correct files wrongly.

The build is limited in the ways its size allows. The corpus is a grammar whose answers are known, so the reference answer of every request exists; with real text the reference is the slow program, and the comparison is of the system with it, not with the truth. The network has 5,760 numbers, so the work of a step is far too small to say anything about a device; the counts of FLOPs and bytes show the structure of the costs, not a speed. And the faults are planted ones, one at a time; real systems fail at several seams at once, which is why each check here is independent of the others. The next chapter reads the history of one real engine, brain, in which these seams were found one failure at a time.

## Exercises

1. **[Calculate]** The base corpus has 180 possible documents. Using the probability that a given document is never drawn in 200 uniform draws, estimate the expected number of distinct documents and compare it with the 120 of the cleaning table.
2. **[Calculate]** Compute the bytes of `tokenizer.bin` from its layout and the counts of the tokenizer table (22 bytes, 48 merges), and check the 414 of the artifact table.
3. **[Calculate]** For the network of the model table, count the multiply-adds of one forward pass of a row of 12 positions by the rule of the work table, and check the 75,072 printed there. Which term dominates, and how does the count change if the attention scores of masked pairs are not computed?
4. **[Calculate]** A decode row at cache length $c$ reads $2 L (c + 1) D$ numbers of the cache. Using the schedule table, add by hand the cache reads of the decode rows of request 0, and then of all eight requests, and check the 159,232 bytes.
5. **[Implement]** Reproduce `capstone.raw_and_clean` and `capstone.tokenizer_stage` in a language other than Python, including the digest of the cleaned training list and the bytes of `tokenizer.bin`, and the round trip of your tokenizer through its own bytes.
6. **[Implement]** Reproduce `capstone.model_and_check`: the initial vector's digest and first numbers, both gradient checks, and the failure of the transposed adapter gradient on the square sites.
7. **[Implement]** Reproduce `capstone.training_stages`: both runs, the kills after 107 and 47 updates, the resumed runs to the bit, and the run resumed without moments. Print your losses only to the digits your own nudged copies agree on, and compare them with the table's.
8. **[Implement]** Reproduce `capstone.schedules` and `capstone.opening` with your own `serve`, both policies and the faulty position rule, and then write a test that sends random arrivals and limits and compares every answer with the reference's; confirm that it finds the faulty rule.
9. **[Implement]** Reproduce `capstone.artifacts` and `capstone.whole_path`: write the four files with the printed lengths and the printed digest of `tokenizer.bin`, a manifest built from the digests of your own files, your loader and its four verdicts, and every row of the whole-path table.
10. **[Investigate]** Replace the learned position table by rotary positions (Volume V, Chapter 8), retrain, and run the faulty position rule again. Does the mistake still change answers, and why, given that rotary positions enter only through the relative offsets of query and key?
11. **[Investigate]** Store the served network in 4 bits (the symmetric 4-bit format of Volume IV, Chapter 12, one scale per row) and compute the margin argument at every step of the reference answers. At which steps does it fail, and do those steps change the served tokens?

## Key terms

- **end-to-end machine learning system**: the whole path from raw data to answers served to clients, built as stages that hand each other only named and checked artifacts, together with a reference that computes the same final answers by the simplest path, so that a fault at any stage or seam shows as a difference that can be traced to where it entered. <!-- concept: end-to-end-ml-system -->

## Answers

Answers to the questions of the opening.

- The gradient check compares the gradient of the backward pass with central differences of the forward pass, $(\mathcal{L}(w + h) - \mathcal{L}(w - h)) / 2h$, entry by entry. Its gate, $|a - n| \le \text{atol} + \text{rtol}\,\max(|a|, |n|)$, is relative for large entries and absolute for small ones, so that an entry near zero is not failed for a difference that is tiny in absolute terms; this build's division by $\max(1, |\text{analytic}|)$ is the same idea with the switch at 1. A product that confuses rows with columns is rejected by its shape on a matrix whose two sizes differ, and on a square matrix it has a legal shape, so only the numbers can betray it.
- $\mathbf{B}$ starts at zero and $\mathbf{A}$ random, so $s\,\mathbf{B}\mathbf{A} = \mathbf{0}$ and the adapted network computes exactly what the base computes before the first update; the gradient of $\mathbf{A}$ is zero at that start and that of $\mathbf{B}$ is not.
- The meta file is written last and the directory is published by one rename; a reader checks that the meta parses, that every file has the recorded length and digest, and that the generation belongs to this run, and it falls back to an older generation otherwise. A resumed Adam needs the first and second moments, the update count (for the bias corrections) and, to continue the data, the number of examples drawn.
- A static batch keeps a slot whose sequence has ended empty until the whole batch ends, so later requests wait; every iteration reads all the weights, so the iterations spent waiting for the longest sequence are reads of the weights that produce fewer tokens.
- A part may depend only on parts of its own or a lower layer. A part that nothing below depends on can be replaced by another with the same interface without changing anything beneath it, and a part whose dependencies all point down can be built and tested with only the layers beneath it present.

Answers to Check your understanding.

1. Alone, the request is the only decode row, so the lowest slot's length is its own. Together in eight slots, all prompts have 7 tokens and every sequence advances one token per iteration, so all lengths are equal at every iteration (sequences that end leave, but the others are still in step). The smallest failing test sends two requests into two slots, the second arriving one iteration after the first, the first allowed at least three tokens and the second at least two: at the second's first decode the first is still decoding one position further on, and the second is given that position, one more than its own.
2. Answers are a coarse measure: the 23 prompts are decided by the largest logit, which a different network can share. The moments changed the trajectory (0.36 at the largest entry), so the run is another run that happens to answer these prompts the same way; nothing guarantees it on other prompts, and nothing in the answers would show a mistake that mattered. The comparison of the trained numbers is what detects that the resume was wrong.
3. If the split is by raw index, two copies of a document at indices on different sides of the split still go to different sides, cleaned or not; cleaning only makes the copies visibly equal. The digest rule assigns a side from the cleaned text, so all copies of a document go to the same side whatever their order, and removing copies before or after cannot change which side a text is on.
4. With $s = 1$, $s^2 = s$ and the mistake changes no number: the check passes and the training is unchanged, so the build cannot see it. With $s = 0.5$ the forward pass uses $0.25$ and the backward pass $0.5$, so every analytic adapter gradient is twice the numeric one and the check fails on every site, square or not, since the error is in the value and not in the shape. A test value of 1 (or 0) hides every mistake that multiplies or raises it to a power; checks should run at values that are not fixed points of the operations they test.
5. The seam between the files and the server's use of them: the server did not merge the adapter. Every check before serving examines the files, which are right; only a comparison of what the server answers with what the reference answers sees what the server did with them.
6. It reads the weights more often: 18 iterations against 16, so 829,440 bytes of weights against 737,280, and 23,698.3 bytes per generated token against 21,065.1. Its requests also wait longer for a slot (2.875 iterations against 1.875 on average). The FLOPs are the same because the same rows are computed.
7. No. The margin argument, taken step by step, proves that greedy decoding keeps the reference's token at 39 of the 40 steps of these eight answers, because at each of them no logit moved by half of that step's margin; at the 40th the token was kept, but that is an observation, not a consequence of the bound. Neither says anything about other prompts, about sampling with a temperature (where every logit matters, not only the largest), or about the probabilities a client might read.

Answers to the exercises.

1. The chance that a given document is never drawn is $(1 - 1/180)^{200} = 0.328$, so the expected number of distinct documents is $180 \times (1 - 0.328) = 120.9$; the drawn corpus has 120.
2. $4 + 22 + 4 + 8 \times 48 = 414$ bytes.
3. Per block $4 \times 12 \times 16 \times 16 = 12{,}288$ for the projections, $2 \times 2 \times 12 \times 12 \times 8 = 4{,}608$ for scores and weighted sums, $3 \times 12 \times 16 \times 24 = 13{,}824$ for SwiGLU: $30{,}720$; two blocks $61{,}440$, plus the head $12 \times 16 \times 71 = 13{,}632$: $75{,}072$. The SwiGLU products and the projections dominate; computing only the 78 unmasked pairs of a row of 12 instead of 144 removes $2 \times 2 \times (144 - 78) \times 8 = 2{,}112$ multiply-adds per block.
4. Request 0 has 5 tokens; its first comes from the prefill, and it decodes at cache lengths 7 to 11 (the decode at length 11 chooses the separator), reading $2 \times 2 \times 16 \times (8 + 9 + 10 + 11 + 12) = 3{,}200$ numbers, 25,600 bytes. A request that stops at its limit after $n$ tokens decodes $n - 1$ times, at lengths 7 to $n + 5$, and one that ends at the separator after $n$ tokens decodes $n$ times; over the eight requests the sums of $c + 1$ are 50 for each of the five that end at the separator, 17 for each of the two that stop at 3 tokens and 27 for the one that stops at 4, 311 in all, and $64 \times 311 = 19{,}904$ numbers are 159,232 bytes.
5. to 9. The worked examples named in each exercise are the reference.
10. With rotary positions the score of a query and a key depends on the difference of their positions. The faulty rule shifts the position of the new token only, not those of the cached keys, so the offsets change and the answers can still change; a rule that shifted every position of a sequence by the same amount would be harmless under rotary positions and not under learned absolute ones.
11. Compute, as `capstone.quantized_serving` does, the smallest margin and the largest change at each step; a step whose change exceeds half its margin may change its token, and only the comparison of the served tokens says whether it did.
