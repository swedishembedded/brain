<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Porting a Model from Its Reference

*What is the disciplined way to reimplement a published model and prove it correct?*

A first port of a published model almost never has exactly one mistake in it. It has a convention read the wrong way in one place, a fused matrix cut in the wrong order in another, and perhaps a third mistake that nobody has made yet because the code that would make it has not run. The standard tool for finding a mistake, a comparison against a reference golden at every hidden state, looks for the first state that leaves its rounding budget, and it finds exactly that: the first. Everything after the first mistake is computed from a wrong input, so every later state is wrong too, and the comparison says nothing about whether a later layer would have been right on a right input. A port with two mistakes looks, under that comparison, like a port with one mistake and a long shadow.

The six-layer tiny decoder below has two. Its import exchanges the keys and the values of layer 1 (both are 6 by 8 matrices, so nothing refuses the exchange) and the gate and up matrices of layer 4 (both 6 by 10). The same port is compared with its golden twice. The first comparison, here called chained, runs the port end to end, as a user would, and compares each of its states with the golden's. The second, called isolated, runs each stage of the port alone, fed the golden's own stored input to that stage, and compares its output with the golden's output of the same stage.

<!-- worked: porting.two_mistakes_table -->
The 6-layer tiny decoder (V = 11, D = 6, H = 4, 2 key-value heads of width 4, F = 10, L = 6, rotary base 100, eps 1e-6; weights from the generator `importing_models.deep`, seed 0x6b38c609eb7b8278, drawn in the order of Volume VI, Chapter 1 and rounded to bf16) on the 8-token text 3, 9, 1, 7, 4, 4, 10, 2. The golden is the 64-bit forward pass with every hidden state and the logits stored in 32 bits; each stored state is a tap, a state at which two runs can be compared. The implementation under test is the 32-bit forward pass (every operation rounded, the gated product of the feed-forward block evaluated as fl(fl(g / fl(1 + fl(exp(-g)))) * up), that is silu(g) = g / (1 + exp(-g)) with exp, add, divide and the product with up rounded in that order, the four roundings of Volume VI, Chapter 2; sums left to right) of an import with keys and values exchanged in layer 1 and gate and up exchanged in layer 4; the correct import is the same pass without the two mistakes. Chained: the implementation runs end to end and each of its states is compared with the golden's. Isolated: each stage is run alone on the golden's stored input to that stage (layer k on the stored output of layer k - 1, the final norm on the stored output of layer 5, the head on the stored final norm). Every number is rel_l2 against the golden state. A layer counts 106 roundings, the final norm 12 and the head 6 (the counting rule of Volume VI, Chapter 2: a sum of n products costs n, each further operation 1). Chained bound after k stages: (roundings of all stages so far + 1) x 2^-24; isolated bound: (roundings of the stage + 2) x 2^-24, one rounding for the stored input and one for the stored output (the embedding lookup has no arithmetic and its bound is 2^-24).

| stage | chained, correct | chained, mistaken | chained bound |  | isolated, correct | isolated, mistaken | isolated bound |  |
|---|---:|---:|---:|---|---:|---:|---:|---|
| embeddings | 0 | 0 | 5.96e-8 | within | 0 | 0 | 5.96e-8 | within |
| layer 0 | 1.15e-7 | 1.15e-7 | 6.38e-6 | within | 1.15e-7 | 1.15e-7 | 6.44e-6 | within |
| layer 1 | 1.57e-7 | 4.50e-1 | 1.27e-5 | above | 1.15e-7 | 4.50e-1 | 6.44e-6 | above |
| layer 2 | 1.43e-7 | 3.92e-1 | 1.90e-5 | above | 8.24e-8 | 8.24e-8 | 6.44e-6 | within |
| layer 3 | 1.55e-7 | 3.93e-1 | 2.53e-5 | above | 8.64e-8 | 8.64e-8 | 6.44e-6 | within |
| layer 4 | 2.29e-7 | 5.01e-1 | 3.17e-5 | above | 8.84e-8 | 3.01e-1 | 6.44e-6 | above |
| layer 5 | 2.20e-7 | 4.81e-1 | 3.80e-5 | above | 7.09e-8 | 7.09e-8 | 6.44e-6 | within |
| final norm | 3.23e-7 | 4.69e-1 | 3.87e-5 | above | 6.64e-8 | 6.64e-8 | 8.34e-7 | within |
| logits | 3.06e-7 | 4.84e-1 | 3.90e-5 | above | 6.35e-8 | 6.35e-8 | 4.77e-7 | within |

Chained: 7 of the 9 states are above their bound, the first at layer 1. Isolated: above their bound are layer 1, layer 4, and nothing else.
<!-- /worked -->

In the chained columns, from layer 1 on, every state of the mistaken port is between $3.92 \times 10^{-1}$ and $5.01 \times 10^{-1}$ away from the golden in relative L2 (rel_l2) and far above its bound, 7 of the 9 states in all. Layer 4 is where the second mistake is made, and nothing in the chained column marks it: the error goes from $3.93 \times 10^{-1}$ after layer 3 to $5.01 \times 10^{-1}$ after layer 4, a change no reader could tell from the drift of a wrong input through a right layer. The isolated columns name both mistakes in one run: layer 1 at $4.50 \times 10^{-1}$, layer 4 at $3.01 \times 10^{-1}$, and every other stage of the mistaken port at the level of rounding, between $6.35 \times 10^{-8}$ and $1.15 \times 10^{-7}$, the same numbers as the correct port's. That difference between the two ways of comparing is where a disciplined port begins; what remains is what has to be true of the golden, of the gates and of the order of work for comparisons like these to prove a port correct.

Four ideas from earlier chapters carry the argument, and each is worth bringing back before reading on.

1. (Volume VI, Chapter 2, "Importing Pretrained Models and Proving Parity".) What does a reference golden hold besides the final output, and why is it computed from the published 16-bit weights and not from the weights before the publisher rounded them?
2. (The same chapter.) How is the parity tolerance $\tau_{\mathrm{par}}$ of a forward pass derived, and what can be concluded about a mistake whose effect lies below it?
3. (The same chapter.) Which mistakes of a weight-name mapping load and run without an error on a model whose matrices are all rectangular, and why do the shapes not catch them?
4. (Volume III, Chapter 6, "Proving Gradients Correct".) What does a gradient check compare a backward pass with, and why can it not say whether the forward pass computes the right function?

## Feeding every stage the reference's own input

A forward pass is a composition of stages. Write them $f_0, f_1, \dots, f_{S-1}$: the layers in order, then the final norm, then the head, so that a decoder of $L$ layers has $S = L + 2$ stages. The golden stores the state before the first stage and after every stage, $\mathbf{h}^{\mathrm{g}}_0, \dots, \mathbf{h}^{\mathrm{g}}_S$, where $\mathbf{h}^{\mathrm{g}}_0$ is the embeddings of the text and $\mathbf{h}^{\mathrm{g}}_{s+1} = f_s(\mathbf{h}^{\mathrm{g}}_s)$ up to the rounding of storing it. The port has its own stages $\tilde f_s$. A chained comparison computes $\tilde{\mathbf{h}}_{s+1} = \tilde f_s(\tilde{\mathbf{h}}_s)$ from the port's own embeddings and compares $\tilde{\mathbf{h}}_{s+1}$ with $\mathbf{h}^{\mathrm{g}}_{s+1}$. An isolated comparison computes $\tilde f_s(\mathbf{h}^{\mathrm{g}}_s)$ and compares it with $\mathbf{h}^{\mathrm{g}}_{s+1}$. Comparing a port with its golden at every stage boundary, in either form, is **layer-by-layer parity**, and the two forms answer different questions: isolated parity asks whether each stage computes its function, and chained parity asks whether the stages put together compute the model.

The bounds of the two forms follow from the rounding model of Volume VI, Chapter 2. There, a stage $s$ adds at most $n_s$ roundings along its longest chain of dependent operations, a 32-bit result carries a relative error of at most $u = 2^{-24}$ per rounding, and a stage is assumed to pass the relative error of its input through with a factor of at most 1. The chained state after stage $s$ has passed through every stage before it, so its bound is the sum of their roundings, plus one for storing the golden:

$$
\tau^{\mathrm{chain}}_s = \Big(\sum_{j \le s} n_j + 1\Big)\,u .
$$

The isolated output of stage $s$ has a different history. Its input is the stored golden state, which differs from the reference's exact state by one rounding, at most $u$ relative in every entry and so at most $u$ in norm; the stage passes that through with a factor of at most 1 and adds its own $n_s$; and the stored golden output it is compared with adds one more:

$$
\tau^{\mathrm{iso}}_s = (n_s + 2)\,u .
$$

For the tiny decoder each layer counts $n_s = 106$ roundings on the 8-token text (the count of Volume VI, Chapter 2), so every isolated layer is held to $108 \times 2^{-24} = 6.44 \times 10^{-6}$, whatever its depth, while the chained bound climbs from $6.38 \times 10^{-6}$ after layer 0 to $3.90 \times 10^{-5}$ at the logits. The isolated bound is constant because the comparison starts afresh at every stage: no error the port made before stage $s$ reaches stage $s$, since stage $s$ is not given the port's own input. The same fact is what lets the isolated comparison name every mistake at once. A mistake in stage $s$ changes $\tilde f_s(\mathbf{h}^{\mathrm{g}}_s)$ and nothing else, because no other isolated comparison reads the output of $\tilde f_s$.

![Two lines of rel_l2 on a logarithmic axis over the eight stages after the embeddings: the chained mistaken port is at the rounding level of about 1e-7 at layer 0, jumps to about 0.45 at layer 1 and stays between 0.39 and 0.51 for every later stage, far above a dashed chained bound that rises from 6e-6 to 4e-5; the isolated mistaken port is about 0.45 at layer 1 and 0.30 at layer 4 and below 2e-7 at every other stage, under its own dashed bound, flat near 6e-6 over the layers and lower at the final norm and the logits.](img/chained-and-isolated.svg)

In the book's pseudocode, with the stages passed as functions and the golden as the list of its stored states:

```text
function isolated_parity(stage: list<function(tensor) -> tensor>, want: list<tensor>,
                         bound: array<real>[S]) -> list<int>:
    # want[s] is the golden's stored input to stage s and want[s + 1] its stored output
    require len(stage) == S and len(want) == S + 1, "one golden state before and after every stage"
    above = []
    for s in 0 .. S:
        got = stage[s](want[s])
        require got.shape == want[s + 1].shape, "a stage changed the shape of the golden state"
        if rel_l2(got, want[s + 1]) > bound[s]:
            append s to above
    return above
```

`rel_l2` is the routine of Volume VI, Chapter 2, applied to the tensors flattened in row-major order, and `bound[s]` is $\tau^{\mathrm{iso}}_s$. The routine returns every stage above its bound, not the first, which is the point. The embedding lookup is not one of the stages: it has no arithmetic, so the worked table compares the port's embeddings with the golden's exactly, under a bound of a single rounding for storage, and both ports pass with a difference of 0.

The arithmetic of an isolated run is the arithmetic of one forward pass, since every stage is evaluated once, and the comparisons add one subtraction and one norm per state. What it needs that a chained run does not is a golden that stores the input of every stage, which a chained comparison layer by layer already requires. Isolation is therefore free once the golden has its taps (the stored states of every stage), and it should be the first comparison run on a new port.

It is not the only one, because it bypasses something. An isolated stage is handed its input by the test, and the test, not the port, decides which tensor that is. A port whose stages are each right and whose plumbing is wrong passes every isolated comparison: a layer that reads the residual from the wrong tensor, a cache of keys reused from another layer, two layers called in the wrong order with each other's weights, a stage that is right on the golden's input and amplifies the small errors of the port's own input more than the reference does. Chained parity is the test of the composition, and its bound is the honest one for what a user runs. A port is proved stage by stage by isolation and as a whole by the chain, in that order.

The Kronos parity test in brain is written this way. Its rungs, documented at the top of `crates/kronos/tests/parity.rs`, are the normalization of the raw input, the tokenizer's encode, its decode, the decoder's logits with and without the calendar embedding, the dependency layer, and two composed rollouts. Each stage before the rollouts is fed the reference's own stored input: the encode rung reads the reference's normalized context ("so this rung is independent of N", in the test's words), the decode rung reads the reference's tokens, and the dependency layer is conditioned on the token the reference sampled. The rollouts then compose the stages on brain's own outputs and must reproduce the reference's generated token streams exactly. The file's header states why those rollout rungs exist: two defects, a dependency layer that rotated its cross-attention and a detokenization that dropped the context window, had passed every gate that compared one brain path with another.

A second question sits inside "every stage": where the boundary between stages is drawn. A comparison can only see a mistake if no operation between the mistake and the tap removes its effect, and normalizations remove scale. Take a vector $\mathbf{x}$ of length $D$ and a factor $c > 0$. The RMS (root mean square) norm of Volume III divides by $\sqrt{\mathrm{mean}(\mathbf{x}^2) + \epsilon_{\mathrm{norm}}}$, and

$$
\frac{c\,\mathbf{x}}{\sqrt{c^2\,\mathrm{mean}(\mathbf{x}^2) + \epsilon_{\mathrm{norm}}}} \;=\; \frac{\mathbf{x}}{\sqrt{\mathrm{mean}(\mathbf{x}^2) + \epsilon_{\mathrm{norm}}/c^2}},
$$

which equals the norm of $\mathbf{x}$ itself except for the size of the epsilon. A scale mistake made before a norm is invisible to every tap after it, up to that change of epsilon, and a tap after a residual addition sees it diluted by the residual. Knowledge #47 records the case in brain: a gate on an 8-bit tier of a text encoder, which held the layer output to a cosine and a relative L2 and the self-attention output to a cosine, passed a mutation that halved every per-channel weight scale. The cosine of the attention output was bit-identical with and without the mutation, and the layer output stayed inside every tolerance (relative L2 from 0.067 to 0.082), because the norm after the attention branch divides the factor out and the residual dominates what survives. Only a relative L2 check on the self-attention output, before that norm, failed it (0.061 to 0.503), which is the assertion `crates/gemma4/tests/it/int8_compute_real.rs` now makes. Stage boundaries for isolation belong where the sublayers end, before their norms, as well as where the layers end.

## What a golden can see depends on the text it was made from

Every comparison above is only as strong as the golden it compares with, and a golden is made, not found. **Golden generation** is the recorded, rerunnable procedure that produces a reference golden: it chooses inputs that exercise every mechanism of the model, runs the reference on them at a precision well above the one the comparison is held to, captures the state at every stage boundary from the reference's real pipeline, and stores the result with a record of the checkpoint it came from.

The obvious golden is a short one, and the shortest is a single token. It is also the weakest, for reasons that can be read off the decoder's equations. At position 0 the rotation of the rotary embedding turns each pair by an angle of $0 \cdot \omega = 0$, and a rotation by 0 is the identity whatever the rates $\omega$ are. With one position, the causal softmax of each head runs over a single score $s$, and $e^{s - s} / e^{s - s} = 1$ exactly, whatever $s$ is. So at one token, the output of every attention head is the value of position 0 itself, and nothing that changes only the angles or only the scores can reach the logits. Five mistakes, each made in every layer of the published two-layer tiny decoder, show which mistakes that leaves invisible:

<!-- worked: porting.golden_text -->
The published 2-layer tiny decoder of Volume VI, Chapter 2 (V = 11, D = 6, H = 4, 2 key-value heads of width 4, F = 10, rotary base 100, eps 1e-6; weights from the generator `modern_decoder.tiny`, seed 0xa39c85ac33ea1d52, drawn in the order of Volume VI, Chapter 1 and rounded to bf16). Three goldens are generated, one for each text in the columns (prefixes of the text 3, 9, 1, 7, 4, 4, 10, 2). Each row makes one mistake in every layer of the port and reports rel_l2 of its logits against the golden's, both in 64-bit arithmetic, so the number is the mistake alone. The query-norm gain is halved to divide every score by d_h = 4 instead of sqrt(d_h) = 2, which multiplies each score by 0.5. The last row is the parity tolerance of Volume VI, Chapter 2 for that text (the rounding count of a 32-bit pass times 2^-24; it grows with the number of positions T through the softmax, which costs T + 3 roundings, and the weighted sum of values, which costs T, so each layer counts 90 + 2T; the counts N of the three columns are 203, 207, 231).

| mistake in the port | 1 token: 3 | 2 tokens: 3, 9 | 8 tokens: the text |
|---|---:|---:|---:|
| rotary base 10000 for the published 100 | exactly 0 | 4.03e-3 | 2.47e-2 |
| query-norm and key-norm gains exchanged | exactly 0 | 7.64e-3 | 3.70e-2 |
| scores divided by d_h, not sqrt(d_h) | exactly 0 | 2.45e-2 | 5.17e-2 |
| keys and values exchanged | 4.54e-1 | 4.92e-1 | 5.91e-1 |
| gate and up exchanged | 8.66e-2 | 1.35e-1 | 4.15e-1 |
| parity tolerance of the logits | 1.21e-5 | 1.23e-5 | 1.38e-5 |
<!-- /worked -->

On the one-token golden, three of the five mistakes give logits that are exactly equal to the golden's, not close to it: a rotary base 100 times too large, the gains of the query and key norms exchanged, and the scores divided by $d_h$ where the reference divides by $\sqrt{d_h}$. Each changes only angles or scores, and the derivation said that neither matters at one position. The other two change what the model computes from the value of position 0, the exchange of keys with values (the value is then a projection that was trained as a key) and the exchange of gate with up, and they show at any length. Two tokens are enough to bring the first three into view, at $4.03 \times 10^{-3}$, $7.64 \times 10^{-3}$ and $2.45 \times 10^{-2}$, every one more than two hundred times the tolerance of $1.23 \times 10^{-5}$. The 8-token text sees all five more strongly. The same reasoning gives the ones the table does not list: a missing causal mask is invisible at one position, where there is nothing to mask, and every mistake in how a cache of keys and values is appended is invisible to a golden that never decodes a second token.

That is the first rule of golden generation: an input is chosen for the mechanisms it exercises, and the list of mechanisms is read from the model, not from what is convenient to feed it. More than one position, so that angles and softmax weights matter; a repeated token (the text has 4 twice), so that two positions with the same embedding must be told apart by position alone; a batch of different lengths, if the model pads, so that the masked rows are checked against the reference's handling of them; and, for a generator, a golden of several decode steps after the prompt. Knowledge #71 records the same blindness in a recurrent layer of brain: in a delta-rule recurrence the first token's output is $(\mathbf{q} \cdot \mathbf{k})\,\mathbf{v}\,\beta$, symmetric in the query and the key, so an exchange of the two is invisible at position 0, and so is every mistake of the rotation, of the cache and of the decay, and the list of what position 0 cannot see is worth working out before a per-position comparison is read as evidence.

The second rule is that the inputs a golden feeds a stage must be the ones the reference's real pipeline builds, captured from it, and not a stand-in the generator assembles by hand. The porting playbook of brain (`.agents/rules/porting.md`) puts it as a hook on the module during a real pipeline run, so that the test becomes a pure replay of tensors the reference made. The reason is the story of commit `091c3f9ef` (August 2026) and knowledge #48, which Volume X's video-generation chapter tells from the side of the positions. The generator of the real-weight goldens of a video transformer built its rotary positions from a plain grid of latent indices, a scope cut its own documentation stated (prove the sequence of operations given arbitrary positions), and the port used the same grid to match; both sides agreed to a cosine of 0.999999 or higher, while the reference's pipeline built its positions in pixel units, 32 times larger along each spatial axis. A golden whose inputs were built by the same simplification as the port cannot see the simplification. The function `real_pixel_positions` in `crates/ltxv/src/pipeline.rs` now builds the reference's positions, and its comment names the goldens that could not have caught the difference.

## A golden made at a precision above the comparison

The second part of golden generation is the arithmetic the reference runs in. A golden is stored in 32 bits, because the comparison is about a 32-bit port and because a reader that decodes nothing wider can read it. Whether the reference itself computed in 32 or 64 bits looks like a detail, and the stored bytes say it is not.

<!-- worked: porting.golden_precision -->
Six reference runs of the published 2-layer tiny decoder (the model of the previous table) on the 8-token text, each stored as a golden of 280 32-bit numbers: the 8 x 6 embeddings, the outputs of the two layers, the final norm (each 8 x 6) and the 8 x 11 logits, in that order, little-endian. A run is either 64-bit or 32-bit arithmetic (every operation rounded to 32 bits, as in Volume VI, Chapter 2) with one of the three summation orders of that chapter, applied to every sum. Each stored golden is compared, number by number, with the first one (64-bit, left to right): how many stored 32-bit patterns differ, and the largest relative difference |a - b| / |b| over the numbers. The digest is the first 16 hex digits of the SHA-256 (Secure Hash Algorithm, 256 bits) of the stored bytes.

| reference arithmetic | summation order | stored numbers that differ | largest relative difference | digest |
|---|---|---:|---:|---|
| 64-bit | left to right | 0 of 280 | 0 | `6ac9151cd2a3d989` |
| 64-bit | right to left | 0 of 280 | 0 | `6ac9151cd2a3d989` |
| 64-bit | pairwise | 0 of 280 | 0 | `6ac9151cd2a3d989` |
| 32-bit | left to right | 190 of 280 | 5.20e-6 | `f23209db4a7c9bb2` |
| 32-bit | right to left | 176 of 280 | 4.14e-6 | `ac032910c487c9cf` |
| 32-bit | pairwise | 168 of 280 | 8.37e-6 | `711e2aab33afa586` |
<!-- /worked -->

The three 64-bit references, each adding every sum in a different order, store the same 280 numbers to the last bit, and so the same digest. The three 32-bit references store golden files that differ from the first in 190, 176 and 168 of the 280 numbers, and from each other too, each with its own digest. The cause is the size of a reference's own error beside the spacing of the numbers it is stored as. A 64-bit result carries, by the bound of Volume VI, Chapter 2 with the 64-bit round-off $2^{-53}$, a relative error of at most $N_{\mathrm{rnd}} \cdot 2^{-53}$, where $N_{\mathrm{rnd}} = 231$ for this model: about $2.6 \times 10^{-14}$. Neighbouring 32-bit numbers are between $2^{-24}$ and $2^{-23}$ apart, relative to their size. Two 64-bit references can store different 32-bit numbers only if the exact result lies within their difference, at most $2 N_{\mathrm{rnd}} \cdot 2^{-53}$, of a point halfway between two neighbours, and the chance of that, for a value that falls anywhere in the gap, is at most that difference divided by the smallest gap:

$$
\frac{2 N_{\mathrm{rnd}} \cdot 2^{-53}}{2^{-24}} \;=\; N_{\mathrm{rnd}} \cdot 2^{-28} \;\approx\; 8.6 \times 10^{-7} \text{ per number.}
$$

Over 280 numbers that is an expected $2.4 \times 10^{-4}$ numbers, and the table's 0 is what such a small expectation predicts. A 32-bit reference, by contrast, has an error of the same order as the gap itself, so its stored bits are a property of its order of summation.

Three consequences follow, and each changes how a port is tested. The tolerance is simpler: a 64-bit golden contributes one rounding (its storage) to the bound, so the bounds of the previous section count only the port's own roundings. A 32-bit golden is itself a second implementation within its own rounding bound of the exact function, and the tolerance of the comparison must cover both: twice the bound, for two evaluations of the same quantity that are each within the bound of the exact value. That is the rule `agreement_bound` in `crates/testutil/src/rounding.rs` implements: $2 k u \Sigma_{\mathrm{abs}} / (1 - k u)$ for a chain of $k$ roundings over terms whose absolute values sum to $\Sigma_{\mathrm{abs}}$. This is the form of Higham's bound, which keeps what the first-order $k u$ drops: $\gamma_k = k u/(1 - k u)$ (Lemma 3.1 of his Chapter 3), with a recursive sum of $n$ terms off by at most $\gamma_{n-1}$ times the sum of the absolute values of the terms (Chapter 4). `accumulation_bound` takes $k$ as the count of roundings on the longest chain, which for a sum is at least $n - 1$, and $\gamma_k$ grows with $k$, so the bound it returns holds. The golden is reproducible: a 64-bit reference rerun on another machine, in another order, writes the same bytes, so the digest of the golden is a fingerprint of the function on those inputs, and a regenerated golden that does not match its recorded digest is evidence that something other than arithmetic changed. And a digest is not, in general, a test of a port's correctness: a 32-bit computation has a correct digest for each order of summation, as the three 32-bit rows of the table show.

A reference framework does not always offer 64 bits on every operation, and some reference code is written for a graphics processor and 16-bit numbers. For that case the playbook asks for the next best thing, a run on the central processing unit (CPU) in 32-bit arithmetic with fixed seeds, and the bound then has to count the reference's chain as well as the port's. A golden generated in 16 bits on a device is not a reference for a 32-bit port at all: its own error is $2^{-8}$ per operation in bf16 and $2^{-11}$ in fp16, larger than every mistake of the fine kind the tolerance exists to catch.

## A golden names the checkpoint it came from

The third part of golden generation is the record that ties a golden to its weights. A golden is a set of tensors and a claim, "this is what the reference computed", and the claim is about a particular checkpoint. Offered to another checkpoint, a golden can only mislead. The table pairs the golden of the six-layer model with three checkpoints:

<!-- worked: porting.pairing -->
The golden of the 6-layer model of the two-mistakes table records, beside its tensors, the identity of the checkpoint it was generated from (the integers that fix every tensor shape: V 11, D 6, L 6, H 4, HKV 2, DH 4, F 10) and the SHA-256 of that checkpoint's file (the 51 tensors of the published layout of Volume VI, Chapter 2 in bf16, layer after layer in the order of that chapter's tables (input norm, fused q-k-v projection, query norm, key norm, output projection, post-attention norm, fused gate-up projection, down projection; each matrix stored [out, in]), then the final norm and the head, written as one safetensors file exactly as that chapter writes one: the compact JSON (JavaScript Object Notation) header with `__metadata__` `{"format":"pt"}` first, padded with spaces to a multiple of 8, then the data; digest beginning `577b1417a089d771`). Three checkpoints are offered to it. The identity is compared field by field and a mismatch refuses the pairing by name; the digest is compared only to say which file it is. When the identity agrees, the 32-bit port of that checkpoint is run on the text and compared with the golden (rel_l2). The redrawn weights come from the generator `porting.redrawn`, seed 0x384ae9c085d7bbbb, in the same order and with the same scales, then rounded to bf16.

| checkpoint offered | identity check | file digest | comparison |
|---|---|---|---|
| the 6-layer model the golden was generated from | agrees | same | embeddings 0, logits 3.06e-7 |
| the 2-layer tiny model (same widths, other depth and weights) | refused: L: golden 6, checkpoint 2 | different | not run |
| 6 layers, the same widths, weights drawn again | agrees | different | embeddings 1.41e+0, logits 1.36e+0 |
<!-- /worked -->

The identity, the integers that fix every tensor shape, refuses the two-layer model by name: the golden says 6 layers and the checkpoint says 2. Without that check the comparison would run, since the per-layer tensors have the same shapes in both models and nothing refuses the load, and the failure that followed would read as a broken port instead of a wrong pairing. The redrawn weights have the same identity, so the comparison runs, and it fails at the embeddings with a relative L2 of 1.41, at the first tensor, where no arithmetic has happened. A failure at the embeddings is never a bug of the forward pass; it says the weights are not the golden's. The file digest names which file the golden was made from and makes that diagnosis immediate, but it is not compared as a condition, because the same weights legitimately arrive as several files (sharded or not, under several names).

In brain this record arrived with commit `c6d9b4f9b` (August 2026), whose message tells why it was needed: a time-series model's goldens had been generated from the tier of the model with a width of 512, the checkpoint the project shipped had a width of 832, and the parity test, finding a mismatch, printed to its error stream and returned, which the test runner counts as a pass. The commit adds three legs that are all still in place. `tools/goldens/golden_source.py` writes a `source` block into a golden's manifest with three fields: `identity`, the shape-fixing integers, which must be non-empty and integers only ("a float tolerance would put the tier check back into judgement", the commit says); `checkpoint`, the reference it was pointed at; and `files`, each weight file by its base name with its SHA-256 digest, for forensics (a dumper over weights too large to hash may record the names alone). On the reading side, `brain_testutil::golden::Source` in `crates/testutil/src/golden.rs` compares every field the test names with the recorded identity, and treats a mismatch as a missing fixture, not as a parity failure: the checkpoint and the golden are each fine, they do not belong together, and the test prints a named skip that quotes the recorded and the actual integers and names the generator to rerun (under `BRAIN_REQUIRE_FIXTURES`, the flag a run sets when it means to prove parity, the skip is a failure). The block is read only from the manifest's `source` key, whatever the manifest file is called (the Kronos parity test points `Source::open_manifest` at `t_meta.json`). A field the test asks for that the golden never recorded is reported as only partly proven, not counted as a match. And `scripts/gates/check-golden-source.sh` refuses a new generator that writes a manifest without the block; the generators that predate the convention are listed by name in the gate, and the list may only shrink. A golden without the block prints `UNVERIFIED GOLDEN SOURCE` and still runs, unless `BRAIN_REQUIRE_GOLDEN_SOURCE` is set, so that each suite can be made strict as its generator is rerun.

In the book's pseudocode the identity check is short, and its one subtlety is the field that was never recorded:

```text
function identity_mismatches(recorded: list<(text, int)>, actual: list<(text, int)>) -> list<text>:
    # recorded: the identity the golden was generated from; actual: the same fields read from the checkpoint under test
    bad = []
    for a in actual:
        found = false
        for r in recorded:
            if r[0] == a[0]:
                found = true
                if r[1] != a[1]:
                    append a[0] to bad
        require found, "the golden does not record a field the test needs"
    return bad
```

An empty result means the pairing is proved for the fields asked, and the comparison may run. The pseudocode takes the strict choice for an unrecorded field; brain's `Source::require` reports it as unverified and lets the comparison run, failing only when `BRAIN_REQUIRE_GOLDEN_SOURCE` is set. The policy of where a golden lives completes the picture, and Volume VI, Chapter 2 described it: brain commits the generator and not the large tensor files it writes, and a test whose golden is absent says so and skips. That choice has its own danger, since a skipped test reports no failure.

## Two gates that guard different things

A port that will be trained has a backward pass, and the backward pass has its own proof, the gradient check of Volume III, Chapter 6. It is tempting to treat the two proofs as one, since both compare numbers with numbers; the table below shows why they cannot be. The network is the two-layer network of that chapter, with a reference that uses tanh; each port runs the reference's forward parity gate and a per-entry gradient check.

<!-- worked: porting.two_gates -->
The two-layer network of Volume III, Chapter 6 (B = 3 examples, D = 5 inputs, H = 7 hidden units, O = 2 outputs, 58 parameters, mean squared error), as the reference with tanh. Its inputs, targets and weights come from the generator `porting.two_gates`, seed 0x25df5e5464a358ec, drawn in the order X (row by row), T (row by row), W1 (row by row), b1, W2 (row by row), b2, each entry `normal()`, and the four tensors W1, b1, W2, b2 (the last four of the six) multiplied by 0.5; W1 has 7 rows of 5 entries and W2 has 2 rows of 7, so the hidden layer is tanh(W1 x + b1) and the outputs are W2 a + b2, and the loss is the sum of the squared errors divided by B x O. Every port runs in 64-bit arithmetic in the same order of operations as the reference, so a correct port's outputs are equal to the reference's and the forward gate demands rel_l2 = 0 exactly; a real port, with its own order of operations, would use the rounding bound instead. The gradient check compares the port's backward pass with the central difference of the port's own loss, step h = 1e-05 on every parameter, under the gate |a - n| <= 1e-07 + 1e-05 max(|a|, |n|) of Volume III, Chapter 6.

| mistake in the port | forward: rel_l2 of the outputs | forward gate | gradient check: entries failing | gradient gate |
|---|---:|---|---:|---|
| none (a correct port) | 0 | pass | 0 of 58 | pass |
| ReLU in place of tanh, in the forward and the backward pass | 1.21e+0 | FAIL | 0 of 58 | pass |
| backward: tanh derivative taken from the pre-activation | 0 | pass | 42 of 58 | FAIL |
<!-- /worked -->

The port that uses a rectified linear unit (ReLU) where the reference uses tanh, in its forward pass and consistently in its backward pass, passes the gradient check on all 58 parameters and fails forward parity with a relative L2 of 1.21. The port whose backward pass takes the derivative of tanh from the pre-activation instead of the activation passes forward parity exactly, since its forward pass is the reference's, and fails the gradient check on 42 of 58 entries. Neither gate implies the other, and the reason is what each compares. A gradient check compares the backward pass with the finite differences of the port's own forward pass, so it proves that the two are consistent and says nothing about whether the forward pass is the reference's function. Forward parity compares the port's outputs with the reference's and says nothing about any derivative. Knowledge #3 states the first half from a real case: a finite-difference check of an image tokenizer's loss could not see which of two terms a weight multiplied, because a wrongly weighted objective is self-consistent, and the fix was to read the weight's place from the reference's source.

So a port runs both, and in a fixed order. Forward parity comes first, because a gradient check of a wrong forward pass certifies the derivative of the wrong function, and a gradient check comes after it for every path the port will train. The two are also not equal in cost, which is worth counting before deciding how often to run them. With a multiply and an add each one floating-point operation (FLOP), and the bias additions, the tanh and the loss not counted, one forward pass of this network costs $2B(DH + HO) = 2 \cdot 3 \cdot (35 + 14) = 294$ FLOPs. A per-entry central difference needs two forward passes for each of the 58 parameters, 116 forward passes or 34,104 FLOPs, against one forward pass for the parity gate. A directional check, which compares the gradient projected on a random direction of $\pm 1$ entries with the central difference along that direction, needs three forward passes per direction (the loss at the point and one step either side, to detect a kink) and so costs the same whatever the number of parameters.

That is the check `Directional::check` in `crates/testutil/src/fd.rs` makes (commit `b8cb733c0`, October 2026), the one the backward-pass tests of brain's depth-estimation and audio-codec crates use. It takes the loss in 64 bits from the caller, so that the difference divided by twice the step is not swamped by the round-off of a 32-bit sum over the outputs. It compares the two one-sided differences of each direction: a direction whose two sides disagree by more than the tolerance has a kink of a ReLU inside its step and is drawn again, up to 8 directions, so that the gradient is checked where the loss is differentiable. Its own test shows that a correct gradient passes, a gradient 5 percent too large fails, and a loss with a kink in every step is reported as such. The commit replaced three private copies of a 32-bit central difference that failed on some random seeds.

## The order in which a port is climbed

The parts are now on the table, and what remains is the order in which they are used, which is most of what makes a port disciplined instead of lucky. The **porting workflow** is that order: an ordered sequence of gated steps that takes a published model to a proven reimplementation, in which each step is begun only when the one before it passes, so that a failure is looked for in what the newest step added.

![Eight boxes in two rows connected by arrows: facts from the references, goldens, mapping, smoke run at toy size in the top row from left to right, then isolated stages, chained run, gradient check and optimize in the bottom row from right to left, with a note that a rung is entered only when the one before it passes.](img/porting-ladder.svg)

The first step is gathering facts, before any code. The authority on the mathematics is the publisher's own implementation; a widely used reimplementation is evidence about names and pipeline glue, and a third tool a third opinion. Where they disagree, the disagreement is written down and settled by an experiment in the reference, not by a guess, because a convention two references disagree about is exactly the kind a port gets wrong. The porting playbook adds two observations that cost nothing: the header of a checkpoint is a complete list of its tensor names and shapes, and so of the model's widths, depth, fused layouts and, by what is absent, its missing features, without a download of the weights; and secondhand write-ups get details wrong and are for orientation only.

The second step is the goldens, before the port's first line, generated by the rules above: inputs that exercise every mechanism, a reference run well above the comparison's precision, and a record of the checkpoint. They come first because every later step needs them, and because a golden generated after the port tends to be shaped by the port's assumptions. The third step is the mapping of Volume VI, Chapter 2, checked in both directions for coverage (every tensor of the checkpoint used, every parameter of the model produced) and for shape, on synthetic tensors, without a single forward pass. The fourth is a smoke run at toy size: random weights in the shapes of the manifest, every dimension different so that a confusion of two sizes changes a shape (the test-dimension choice of Volume III, Chapter 6), every kind of step exercised, which catches a buffer sized wrong or a binding misplaced before a run on real weights hits it at the end of a long wait.

The fifth and sixth steps are the two forms of layer-by-layer parity, isolated first and chained second, on the real weights and the real goldens. The seventh is the gradient check of every path that will be trained, after the forward pass is proved. The last is optimization: only once the forward pass is frozen by its parity gate is any change made for speed, and every such change reruns the gate, because without it a speedup and a bug cannot be told apart.

The order is set by localization. Each step adds one kind of thing that can be wrong, and a step that fails points at that kind: a failure of the mapping is about names and shapes, a failure of the smoke run about buffers and bindings, an isolated failure about one stage's arithmetic, a chained failure with clean isolation about composition, a gradient failure with clean parity about the backward pass. A port that skips to the end, a whole model on real weights compared at the logits, has all of those kinds of mistake behind one number, and the work of finding them is spent re-running the most expensive step. The cost of each step grows down the ladder, from a mapping check that reads only names and shapes, through a toy-size run, to forward passes of the real model and, for the gradient check, two forward passes per checked parameter; climbing in this order means the expensive steps run on a port whose cheap mistakes are already gone.

There is one situation the ladder does not resolve by itself, and brain met it. When a port and the file it reads agree with each other and the output is still wrong, no comparison of the port against itself can say which of them is at fault. Commit `450ef61ae` (August 2026), with knowledge #71, is the response: a second, independent implementation of the same architecture in plain Python, `tools/goldens/qwen35_gguf_reference_forward.py`, which parses the quantized single file itself, decodes its blocks itself and runs the decoder over the same real bytes, transcribed from the publisher's reference. The 32-bit stack of brain agreed with it to a cosine of 1.000000 and a relative L2 of 0.0000 at 1, 4, 8 and 12 real layers over 4 positions, which ruled the loader out as the cause, and the four-layer case of that agreement is now a standing gate, `crates/qwen35/tests/it/gguf_reference_parity_real.rs`. The lesson is general. A reference for a layer is a few matrix products, a norm, a softmax and a recurrence, and when no outside oracle is at hand one can be written from the reference's source in a few hundred lines, which is cheaper than another wrong hypothesis about a model of many layers.

## The shared test kit behind brain's ports

The machinery of these comparisons lives in one crate, `crates/testutil` (package `brain-testutil`), a dev-dependency of 111 crates (the manifests that name it) and a dependency of no production code. It began as the cure for a copied function. Commit `7de4f8a80` (August 2026) found 36 byte-identical copies of a four-line function that resolved the path of a test fixture, one per parity test file, and replaced them with `testdata` and `testdata_path` in `crates/testutil/src/lib.rs`: the directory named by `BRAIN_TESTDATA` if it is set and not empty, and otherwise the repository's own `testdata/` directory, computed from the crate's own location (safe because every crate sits at the same depth). `model_dir` resolves the directory of an upstream checkpoint in the model store instead, since `testdata/` holds goldens and input media only and never a checkpoint, and `read_f32` and `read_i32` read the raw little-endian dumps that several older generators write.

`crates/testutil/src/parity.rs` holds the comparison itself; commit `d2a345388` (August 2026) folded into it the private copies of its report that other crates' parity tests had kept. `compare` returns the cosine and the largest absolute difference, accumulated in 64 bits, and `rel_l2` the relative L2 difference, which for an all-zero reference is 0 when the result is also zero and infinite otherwise, so that a nonzero output against an empty reference reads as maximally wrong. Three report types differ in when they print and what they gate on. `Report` prints a line per tap as it goes (cosine, relative L2, largest difference and element count), refuses a reference tap that is all close to zero, since such a tap compares perfectly against anything, and at the end asserts that every tap's cosine clears one floor, a not-a-number cosine counting as a failure. `Table` gates every tap on a cosine floor and a relative L2 ceiling together and collects every violation instead of stopping at the first, with a `has` method that checks a golden tap was actually compared. `WorstTable` prints its rows and asserts both extremes. Both numbers are kept because each is blind where the other sees: cosine cannot see a uniform scale, and the largest difference alone can miss a permutation of values on a smooth field.

The other files hold the remaining pieces of the method. `golden.rs` is the identity check above. `rounding.rs` (commit `65c09d5df`, November 2026) gives the bounds of the precision section, `accumulation_bound` and `agreement_bound`, which replaced a fixed tolerance of $10^{-6}$ in two backend parity tests with a bound computed from each comparison's own count of roundings and sum of absolute terms. `fd.rs` is the directional gradient check. `hfquant.rs` writes random weights as checkpoint shards in the layouts real releases ship, plain floats, 8-bit floats with a per-tensor scale and a 4-bit block format with two levels of scale, through encoders written in the test kit that share no code with brain's own decoders, so that a loader's result can be compared with values the fixture decoded independently: the idea of the independent Python reference, applied to loading. `adapters.rs` writes adapter files for the tests of serving, and `lib.rs` also holds the named skips that let a test without its fixture say so, a lock for tests that change the process environment, and readers of the process's resident and peak memory for tests that bound it.

## Check your understanding

1. A port passes isolated parity at every stage, and its chained comparison is within its bound up to layer 2 and above it from layer 3 on. Where would you look first, and why not inside layer 3's arithmetic?
2. A colleague argues that the isolated bound of layer 5 should be larger than that of layer 0, because layer 5 is deeper and errors accumulate with depth. Find the error in the argument.
3. A port passes a parity test against a golden made from the single token `3`. Name three mistakes that could still be in it, and the shortest golden that would expose all three.
4. Two people regenerate a 32-bit golden from the same reference on different machines and get different digests. One concludes that the golden is useless. What is right about the observation and wrong about the conclusion, and what would make the digests agree?
5. A port's gradient check passes on every entry. Its activation is the tanh approximation of GELU (the Gaussian error linear unit) where the reference uses the exact one. Predict what forward parity reports, and explain why the gradient check is silent.
6. A golden's identity agrees with the checkpoint, and the comparison fails at the embeddings with a relative L2 of 1.4. What happened, and where is the bug?
7. A scale factor of 2 is applied by mistake to the output of a sublayer that is followed by an RMS norm, and the only tap is after the norm. What does the tap report, and what tap would catch it?

## Summary

A port of a published model is proved correct by comparisons against a reference golden, and the strength of the proof is set by three choices: how the comparison is made, how the golden is made, and the order in which the steps are taken. Layer-by-layer parity has two forms. Isolated parity feeds each stage the golden's stored input and holds it to its own rounding bound, $(n_s + 2)u$, constant with depth, and so names every mistake in one run, two in the worked port where the chained comparison named one. Chained parity runs the port on its own outputs under a bound that grows with depth and is the only test of how the stages are put together. Taps belong before normalizations, because a norm removes a scale mistake from everything after it.

Golden generation decides what the comparisons can see. A golden's inputs must exercise every mechanism: at one token, any mistake in the angles or the scores is not small but exactly zero, and two tokens expose it. Its inputs to each stage must be captured from the reference's real pipeline, or a simplification shared with the port cannot be seen. Made in 64 bits and stored in 32, a golden is independent of the reference's order of summation (0 of 280 stored numbers differed across three orders, against 168 to 190 for 32-bit references), its digest is reproducible and the tolerance counts only the port; made in 32 bits, the tolerance must count both implementations. And a golden records the identity of its checkpoint, so that a wrong pairing is refused by name instead of passing or failing for the wrong reason. Forward parity and the gradient check prove different things, the function and its derivative, and a port needs both, parity first. The porting workflow puts all of it in an order in which every failure points at the newest step.

The limits are those of the reference. Parity proves that a port computes what the reference computes on the inputs the golden covers; it does not prove the reference is right, and it does not prove anything about inputs no golden exercised. The tests that carry these proofs can themselves fail to prove anything, by passing when they should fail or by not running at all, and the next chapter turns to those failures of numerical tests.

## References

- Nicholas J. Higham, *Accuracy and Stability of Numerical Algorithms*, second edition, 2002: the rounding-error analysis of sums and inner products (Chapter 3), the source of the bound $k u/(1 - k u)$ on a chain of $k$ roundings.

## Exercises

1. **[Calculate]** A decoder has $L = 24$ layers whose stage count, by the rule of Volume VI, Chapter 2, is $n_s = 10{,}528$ roundings per layer on its text. Give the isolated bound of a layer and the chained bound after the last layer (before the final norm), both as multiples of $u$ and as numbers. By what factor do they differ?
2. **[Calculate]** For that decoder, with $N_{\mathrm{rnd}} = 252{,}672$ roundings on the longest chain, use the estimate of the precision section to bound the chance that two 64-bit references store a different 32-bit number at one position, and the expected number of differing numbers in a golden of $10^7$ numbers.
3. **[Calculate]** For each of these mistakes, say whether a golden made from one token can see it: a missing causal mask; the subtraction of the maximum score omitted from the softmax (in exact arithmetic); a wrong epsilon in the RMS norms; the output projection $\mathbf{W}_O$ of attention transposed in a model where it is square; the rotary embedding applied to the values as well as the queries and keys.
4. **[Calculate]** A network has $B = 4$, $D = 16$, $H = 32$, $O = 8$. Count the FLOPs of one forward pass by the rule of the gradient section, then of a per-entry central-difference check over every parameter (weights and biases), and of a directional check that needs 2 directions.
5. **[Implement]** Reproduce `porting.two_mistakes_table`: build the six-layer model from its seed, generate its golden in 64 bits stored in 32, make the two mistakes in the import, and run both the chained comparison and `isolated_parity` in emulated 32-bit arithmetic, to the digits printed.
6. **[Implement]** Reproduce `porting.golden_text`: make each of the five mistakes in the published two-layer model and compare with goldens made from one, two and eight tokens, reporting the exact zeros as exact.
7. **[Implement]** Reproduce `porting.golden_precision`: run the reference in 64 and in emulated 32-bit arithmetic with the three orders of summation, store each golden in 32 bits, and count the differing bit patterns and compute the digests.
8. **[Implement]** Reproduce `porting.two_gates`: from the seed and draw order given, run forward parity and the per-entry gradient check on the three ports and report the counts.
9. **[Investigate]** Make a mistake in the port's composition that every isolated comparison passes and the chained comparison catches: take the six-layer port of the first table with every stage correct, and let layer 3 read the output of layer 1 instead of the output of layer 2. Which chained state is the first above its bound, and what would a reader of the isolated table alone conclude?
10. **[Investigate]** On the published two-layer tiny decoder in 64-bit arithmetic, multiply $\mathbf{W}_O$ of layer 0 by 2, which doubles the attention output of that layer, and compare the rel_l2 at the attention output before the residual, at the input of the second RMS norm, at the output of the layer and at the logits with their bounds. Repeat with $\mathbf{W}_Q$ of layer 0 times 2, which doubles the queries before the QK-norm (the RMS norm of each head's queries and keys), at the queries, the layer output and the logits, and explain why that mistake changes nothing but the effect of the epsilon.

## Key terms

- **porting workflow**: the ordered sequence of gated steps that takes a published model to a proven reimplementation (facts from the references, goldens generated first, the mapping checked for coverage and shape, a smoke run at toy size, isolated and then chained parity, a gradient check of every trained path, and optimization only under the frozen parity gate), each begun only when the one before it passes. <!-- concept: porting-workflow -->
- **layer-by-layer parity**: comparison of a port with its golden at every stage boundary, either isolated (each stage fed the golden's own input and held to its own constant rounding bound, which locates every mistake in one run) or chained (the port running on its own outputs, under a bound that grows with depth, which tests how the stages are composed). <!-- concept: layer-by-layer-parity -->
- **golden generation**: the recorded, rerunnable procedure that makes a reference golden: inputs chosen to exercise every mechanism of the model, the reference run at a precision well above the comparison's, the state at every stage captured from the reference's real pipeline, and a record of the identity and digest of the checkpoint it came from. <!-- concept: golden-generation -->

## Answers

**Recall questions.**

1. It holds the input and the intermediate states, the hidden state after the embedding and after every layer, so that a failure can be located; it is computed from the published 16-bit weights because those are the numbers a port reads, and the rounding of the publisher's weights changes the logits by far more than the parity tolerance, so a golden from the unrounded weights would fail a correct port.
2. It is the number of roundings along the longest chain of dependent operations, $N_{\mathrm{rnd}}$, times the unit round-off $u = 2^{-24}$, assuming that no stage amplifies the relative error of its input. A mistake whose effect is below it cannot be told from rounding by that comparison, so it must be found by another test, such as comparing the configuration's values or choosing an input that enlarges the mistake's effect.
3. Those whose result has the same shapes as the correct one: keys exchanged with values when their widths are equal, gate exchanged with up, and a head dropped because a tied configuration was assumed. The shapes cannot catch them because the exchanged tensors have identical shapes and the model's shape checks are satisfied.
4. It compares the analytic gradient with finite-difference estimates of the derivative of the same forward pass the port computes. Since both come from the port's forward pass, a wrong forward pass with a consistent backward pass passes the check.

**Check your understanding.**

1. In the composition: what layer 3 is given. Isolated parity says that layer 3 computes the right function of the golden's input, so the chained failure must come from layer 3 receiving something other than layer 2's output (a wrong tensor for the residual, a cache from another layer, a stage called out of order), or from a layer whose amplification of small errors is much larger than the reference's, which a look at the chained errors before and after layer 3 can tell apart.
2. The isolated comparison of layer 5 starts from the golden's stored state, not from the port's state after five layers, so none of the port's earlier errors reaches it. Its bound is its own roundings plus one for the stored input and one for the stored output, $(n_s + 2)u$, the same as layer 0's when the layers have the same count.
3. Any mistake that changes only the rotation angles or only the attention scores, for example a wrong rotary base, exchanged query-norm and key-norm gains, a wrong score scale, a missing causal mask. A golden of two tokens exposes all of them, since position 1 has a nonzero angle and its softmax runs over two scores.
4. The observation is right: a 32-bit reference's stored bits depend on its order of summation and its library functions, so two correct runs can store different numbers. The conclusion is wrong, since both goldens are within the rounding bound of the exact function and either serves for parity if the tolerance counts the reference's roundings as well as the port's. A reference run in 64 bits and stored in 32 would make the digests agree, except with a chance of about $N_{\mathrm{rnd}} \cdot 2^{-28}$ per number.
5. Forward parity reports a small but nonzero difference, above the parity tolerance wherever the inputs of the activation are not tiny, since the two functions differ. The gradient check is silent because the port's backward pass is the derivative of the port's own activation, which is what the finite differences measure.
6. The checkpoint is not the one the golden was made from: the same architecture with different weights (another release, a fine-tune, a redrawn set). The embedding lookup has no arithmetic, so a disagreement there is about the weights. There is no bug in the port; the file digest in the golden's record says which file it was made from.
7. The tap reports agreement, up to the change of the effective epsilon, because the RMS norm divides the factor out. A tap on the sublayer's own output, before the norm, sees a relative L2 of 1.0.

**Exercises.**

1. Isolated: $(10{,}528 + 2)u = 10{,}530\,u = 6.28 \times 10^{-4}$. Chained after 24 layers: $(24 \times 10{,}528 + 1)u = 252{,}673\,u = 1.51 \times 10^{-2}$. They differ by a factor of about 24, the number of layers.
2. $N_{\mathrm{rnd}} \cdot 2^{-28} = 252{,}672 / 268{,}435{,}456 = 9.4 \times 10^{-4}$ per number, so at most about $9{,}400$ of $10^7$ numbers would be expected to differ; real errors are usually far below the bound, so fewer would.
3. A missing causal mask: invisible (one key, nothing to mask). The maximum subtraction: invisible at any length in exact arithmetic, since the softmax is unchanged by a shift. A wrong epsilon: visible, but only by its small effect on the norms, possibly below the tolerance. A transposed square $\mathbf{W}_O$: visible, since the value of position 0 still passes through it. The rotation applied to the values: invisible at position 0, where the rotation is the identity.
4. One forward pass: $2 \cdot 4 \cdot (16 \cdot 32 + 32 \cdot 8) = 8 \cdot 768 = 6{,}144$ FLOPs. Parameters: $16 \cdot 32 + 32 + 32 \cdot 8 + 8 = 808$, so the per-entry check takes $2 \times 808 = 1{,}616$ forward passes, $9{,}928{,}704$ FLOPs. The directional check with 2 directions takes $2 \times 3 = 6$ forward passes, $36{,}864$ FLOPs.

**Implement and Investigate.** The Implement exercises must reproduce the printed tables to the digits printed.

Exercise 9. The wire is a mistake of the plumbing, so every stage computes its own function correctly on whatever it is given:

<!-- worked: porting.composition -->
The 6-layer decoder, text, golden and bounds of the two-mistakes table, ported with every stage correct and one wire wrong: layer 3 reads the output of layer 1 instead of the output of layer 2 (layer 2 is still computed and its output is discarded). Each stage is the 32-bit stage function of the two-mistakes table, so the same code run with every wire right is the chained run of that table to the last bit. Chained: the wired port end to end against the golden's states. Isolated: every stage fed the golden's own input, which the test supplies and so bypasses the wire. Every number is rel_l2 against the golden state.

| stage | chained | chained bound |  | isolated | isolated bound |  |
|---|---:|---:|---|---:|---:|---|
| embeddings | 0 | 5.96e-8 | within | 0 | 5.96e-8 | within |
| layer 0 | 1.15e-7 | 6.38e-6 | within | 1.15e-7 | 6.44e-6 | within |
| layer 1 | 1.57e-7 | 1.27e-5 | within | 1.15e-7 | 6.44e-6 | within |
| layer 2 | 1.43e-7 | 1.90e-5 | within | 8.24e-8 | 6.44e-6 | within |
| layer 3 | 5.01e-1 | 2.53e-5 | above | 8.64e-8 | 6.44e-6 | within |
| layer 4 | 4.55e-1 | 3.17e-5 | above | 8.84e-8 | 6.44e-6 | within |
| layer 5 | 4.52e-1 | 3.80e-5 | above | 7.09e-8 | 6.44e-6 | within |
| final norm | 6.70e-1 | 3.87e-5 | above | 6.64e-8 | 8.34e-7 | within |
| logits | 7.42e-1 | 3.90e-5 | above | 6.35e-8 | 4.77e-7 | within |

Chained: 5 of the 9 states are above their bound, the first at layer 3. Isolated: 0 stages are above their bound.
<!-- /worked -->

The first chained state above its bound is the output of layer 3, the first layer that reads the wrong state, at a relative L2 of $5.01 \times 10^{-1}$, and every state after it is above its bound too, up to $7.42 \times 10^{-1}$ at the logits. The isolated table has no stage above its bound, so a reader of it alone would conclude that the port has no mistake. Only the chained comparison sees a wire.

Exercise 10. The decoder normalizes before each sublayer, so the residual stream carries the attention output past the norm that follows it, and only the effect of the queries is divided out:

<!-- worked: porting.scale -->
The published 2-layer tiny decoder of the golden-text table (weights from `modern_decoder.tiny`, seed 0xa39c85ac33ea1d52, text of 8 tokens), reference and mistaken port both in 64-bit arithmetic, so each number is the mistake alone. The mistake is a factor 2 on the matrix named, in layer 0 only; the product with a matrix is linear, so a factor on W_O scales the attention output and a factor on W_Q scales the queries, exactly. Taps: the queries before the QK-norm (the input of the norm of Volume VI, Chapter 1, 8 x 4 per head); the attention output before the residual add (read as the layer's output minus its input when W_down is zero, so that the MLP adds nothing); the input of RMS norm 2, which is the residual sum; the output of the layer; the logits. Every number is rel_l2 against the reference's tap. The bound of a tap is (the roundings of the stages of the layer up to it, by the count of the two-mistakes table, plus 1) x 2^-24: 19 after the projection to queries, 73 after W_O, 74 after the residual add, and for the layer output and the logits as in the golden-text table.

| mistake | tap | rel_l2 | bound |  |
|---|---|---:|---:|---|
| attention output (W_O) times 2 | attention output before the residual | 1.00e+0 | 4.35e-6 | above |
| attention output (W_O) times 2 | input of RMS norm 2 (after the residual) | 4.38e-1 | 4.41e-6 | above |
| attention output (W_O) times 2 | output of layer 0 | 3.88e-1 | 6.38e-6 | above |
| attention output (W_O) times 2 | logits | 1.77e-1 | 1.38e-5 | above |
| queries before the QK-norm (W_Q) times 2 | queries before the QK-norm | 1.00e+0 | 1.13e-6 | above |
| queries before the QK-norm (W_Q) times 2 | output of layer 0 | 3.37e-8 | 6.38e-6 | within |
| queries before the QK-norm (W_Q) times 2 | logits | 3.19e-8 | 1.38e-5 | within |
<!-- /worked -->

A doubled attention output is seen exactly at the tap on the sublayer's own output (a relative L2 of 1, against a bound of $4.35 \times 10^{-6}$) and is diluted by the residual at every later tap, to $0.438$ at the input of the second norm, $0.388$ at the layer output and $0.177$ at the logits, each far above its bound; the residual sum is a tap after the sublayer, not a tap on it. A doubled query is also seen exactly before the QK-norm, and then the norm divides the factor out: $c\,\mathbf{q}$ normalizes to $\mathbf{q}\big/\sqrt{\mathrm{mean}(\mathbf{q}^2) + \epsilon_{\mathrm{norm}}/c^2}$, which differs from the reference only through the epsilon, divided by $c^2 = 4$. The effect, $3.37 \times 10^{-8}$ at the layer output and $3.19 \times 10^{-8}$ at the logits, is below the bounds of $6.38 \times 10^{-6}$ and $1.38 \times 10^{-5}$, so no comparison after the norm can see that mistake.
