<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Project: Comparing Language Model Blocks

*Can you implement several alternative language-model blocks and compare their quality, cost and memory fairly?*

Suppose a table says that block A beats block B by 0.03 nats. Everything that makes the sentence true or false is missing from it. A may hold twelve percent more parameters than B, or spend a third more FLOPs on every token. The learning rate may have been tuned for A and left at a default for B. There may have been one seed, or the seed may have been picked after looking. The test sequences may be short enough that B's one real weakness never shows. A reader who meets such a table has no way to say what it measured, and neither does its author.

Ten chapters of this volume have each offered a way to change the block of Volume V, and each came with a count and a claim. Grouped keys and values shrink the cache. A window shrinks it further. Routing sends a token through two small networks instead of one large one. A recurrence with a matrix state replaces the cache by a memory that does not grow. None of those chapters could put two blocks side by side on equal terms, because equal terms need the whole volume at once: the blocks themselves with gradients that are right, a cost model that agrees with a counted run, a way to prove that an implementation computes what it should, and the method of the research chapter for deciding whether a difference is more than noise. This chapter asks you to build all of that once, in a language of your choice, and to use it on a small task that was designed so that the blocks must differ. You will write five parts:

1. **Four blocks behind one interface.** A pre-norm layer whose mixer and whose network can be exchanged: dense grouped-query attention with a SwiGLU network (Chapter 1), the same attention with a mixture-of-experts network, a top-2 router and a balancing loss (Chapter 4), a gated delta network layer in its recurrent and its chunked forms (Chapter 5), and a sliding-window attention layer (Chapter 3). Each has a hand-written backward pass checked against central differences, and the recurrent layer's two forms are checked against each other within a tolerance you derive.
2. **A cost model.** Parameters, active parameters, FLOPs per token at prefill and at decode, and bytes held, for every block at stated sizes (Chapter 10), each count checked against what the running code did.
3. **A parity harness.** Goldens of every block's outputs written to a file, three measures of difference and a tolerance counted from the roundings (Chapter 2), and mistakes planted on purpose that the harness has to catch.
4. **A controlled comparison.** One synthetic task that needs local and long-range information, every block in the same model skeleton at matched parameters, six seeds, predictions registered before the runs, paired intervals and verdicts (Chapter 9), numbers printed only to the digits that survive a nudge.
5. **A hybrid.** A stack that interleaves a recurrent and an attention layer, measured for cost and for quality against the two pure stacks.

Nothing in the list is new. Every part was derived and worked by hand in a chapter of this volume or of Volume V, and `scripts/book/worked/project_architecture_comparison.py`, the module that produced every expected value below, imports the worked modules of those chapters instead of copying them. The standard of the chapter is the one the volume has been teaching: report what comes out. A prediction that the data refute is a result, an interval that straddles its margin is a result, and a comparison in which two blocks cannot be told apart at this size is a result that the next section should say plainly instead of dressing it up.

![Five boxes labelled with the milestones in build order. Milestone 1 (four blocks) is at the left. Milestone 2 (cost model) is above and to the right of it and milestone 3 (parity harness) below and to the right; both receive an arrow from milestone 1. Milestone 4 (comparison, highlighted) is to the right of both and receives arrows from milestones 1, 2 and 3. Milestone 5 (hybrid) is at the far right and receives an arrow from milestone 4.](img/project-architecture-milestones.svg)

The figure is a valid build order. The comparison reads everything before it: the blocks, the costs that turn a loss into a point on a plane of quality against FLOPs, and a harness that says the implementations compute what they claim. Build and check each milestone before the next, because a comparison of a block that has a wrong backward pass is a comparison of the bug. The **acceptance tests** at the end collect the expected values in one place. You need the book's generator (Volume I, Chapter 17), double-precision arithmetic with a correctly rounded `exp`, `log`, `sqrt`, `sin` and `cos`, and the libraries of Volumes III and V. The 32-bit rounding of milestone 3 is a conversion of a 64-bit number to the nearest 32-bit float.

Six questions bring back what the milestones lean on. They reach across the volume and back into Volume V; answer each from memory before reading on, since the answers are at the end.

1. (Volume VI, [Chapter 3](03-attention-variants.md), cache of grouped and windowed attention) A layer has 4 query heads and 2 key-value heads of width 4, and numbers are stored in bfloat16. How many bytes does one token add to that layer's cache, and how many bytes does a sliding window of 4 keys hold at most?
2. (Volume VI, [Chapter 4](04-mixture-of-experts.md), the router and its balancing loss) A router gives four experts the probabilities 0.5, 0.3, 0.15 and 0.05 and the token uses $k = 2$. Which experts does it use, and what are the combine weights? What does the balancing loss $\alpha_{\mathrm{bal}} E \sum_e f_e \bar p_e$ equal when every expert receives an equal share of the slots and of the probability?
3. (Volume VI, [Chapter 5](05-linear-attention.md), the gated delta network) A gated delta layer has 2 heads with $d_k = 4$ and $d_v = 7$. How many numbers does its state hold after 10 tokens and after 10,000? If the decay gate is $\alpha = 0.95$ at every token, after how many tokens has a value that was written lost half of its weight?
4. (Volume VI, [Chapter 2](02-importing-models.md), parity) An output $\mathbf{a}$ equals $2\mathbf{b}$ for the golden $\mathbf{b}$. Which of max_abs, rel_l2 and the cosine similarity see the difference? What two quantities does the parity tolerance multiply?
5. (Volume VI, [Chapter 10](10-model-cost.md), FLOPs per token) A decoder has $N_{\mathrm{act}} = 4{,}032$ parameters in the matrix products of a token, $L = 2$ layers, $H = 4$ heads of width $d_h = 4$, and a token sees $T = 13$ keys. How many FLOPs does the token cost?
6. (Volume VI, [Chapter 9](09-spectral-lm.md), a registered claim and its interval) Six paired differences have mean $-0.20$ and standard deviation $0.10$. Give the standard error and the 95 percent interval (multiplier 2.5706 for 5 degrees of freedom), and say whether a claim registered as "the difference is at most $-0.1$" is supported, refuted or undecided.

## Four blocks and a verdict, in five steps

The pseudocode follows [Pseudocode Conventions](../pseudocode.md). It fixes interfaces and the order of operations, and it reuses by name the routines of earlier chapters (`rms_norm` and `rms_norm_backward` of Volume V, Chapter 9, `route` and `balance_loss` of Chapter 4, `gated_delta_tokenwise` and `gated_delta_chunked` of Chapter 5, `ledger` of Chapter 10, `max_abs`, `rel_l2` and `cosine` of Chapter 2). Every random input comes from the book's SplitMix64 at a seed printed beside its table and derived by the rule `seed_of(name)`; the draws are taken in the order the table states. A dot product may be accumulated in any order that is at least as accurate as left to right: every printed digit stays clear of the last bit of a sum. Trained numbers are printed only to the digits that survive the nudge rule of this book (`scripts/book/worked/stable.py`): the run is repeated from four starting points whose first weight differs by $\pm 10^{-12}$ and $\pm 3 \times 10^{-12}$, and a digit is printed only if it is the same in all five runs.

### Milestone 1: four blocks behind one interface

A block, in this chapter, is the pair of sublayers that Volume V called a layer: a pre-norm mixer with its residual addition, then a pre-norm network with its residual addition. What differs between the four blocks is the mixer (how a token reads other tokens) and the network (what is done to one token), and everything outside that stays fixed so that the blocks are compared in the same skeleton: a token table of shape $[12, 16]$, drawn as `normal() / sqrt(D)` and used again as the output head (Volume V, Chapter 9, tied embeddings), a stack of layers, a final RMS norm with a gain. The interface is two functions and a state:

```text
function layer_forward(mixer, ffn, lay, X: array<real>[n, D])
        -> (array<real>[n, D], trace, real):
    (N1, xhat1, inv1) = rms_norm(X, lay.g1, 1e-6)
    (A, mixer_trace) = mixer.forward(lay, N1)           # tokens read other tokens
    X1 = X + A
    (N2, xhat2, inv2) = rms_norm(X1, lay.g2, 1e-6)
    (F, ffn_trace, aux) = ffn.forward(lay, N2)          # each token alone; aux is a loss term
    return (X1 + F, trace of all of the above, aux)

function layer_backward(mixer, ffn, lay, trace, dY: array<real>[n, D])
        -> (array<real>[n, D], gradients of lay):
    (dN2, g_ffn) = ffn.backward(lay, trace.ffn_trace, dY)
    (dX1n, dg2) = rms_norm_backward(dN2, xhat2, inv2, lay.g2)
    dX1 = dY + dX1n                                     # the residual carries dY through
    (dN1, g_mixer) = mixer.backward(lay, trace.mixer_trace, dX1)
    (dXn, dg1) = rms_norm_backward(dN1, xhat1, inv1, lay.g1)
    return (dX1 + dXn, gradients of lay assembled from dg1, g_mixer, dg2, g_ffn)
```

Because the two sublayers meet only in these two functions, exchanging a mixer or a network touches nothing else, and the model's forward pass, loss and backward pass are written once. The shapes are those of one small configuration, chosen so that no two dimensions coincide (a transposed matrix is then a shape error, not a quiet bug): width $D = 16$; four query heads and two key-value heads of width 4 (so the query projection is $[16, 16]$ and the key and value projections $[16, 8]$); a SwiGLU hidden width of 24; rotary pairs split in halves with base 100; a window of 4 keys; four experts of hidden width 6 with the top two used; two heads of the gated delta layer with key width $d_k = 4$ and value width $d_v = 7$. The value width is the odd one: it is the width at which the gated delta layer holds one parameter more than the attention layer, so that the arms are matched on parameters without padding either.

The four blocks:

- **Dense.** The recipe of [Chapter 1](01-modern-decoder.md): grouped-query attention (query head $h$ reads key-value head $\lfloor h / 2 \rfloor$), a QK-norm with a gain of width 4 for queries and one for keys, the rotary rotation after the norm, causal softmax attention, $\mathbf{W}_O$, then SwiGLU. Weights are drawn in the order $g_1$, $\mathbf{W}_Q$, $\mathbf{W}_K$, $\mathbf{W}_V$, QK-gains, $\mathbf{W}_O$, $g_2$, $\mathbf{W}_{\mathrm{gate}}$, $\mathbf{W}_{\mathrm{up}}$, $\mathbf{W}_{\mathrm{down}}$; gains as `1 + 0.1 * normal()`, matrices as `normal() / sqrt(fan_in)`, row by row. This is the order of the decoder of Chapter 1, which is what lets milestone 3 compare the two.
- **Sliding window.** The same tensors, drawn in the same order, and a mask that lets position $t$ read positions $t - 3 .. t$ only (a window of 4 keys including its own). Nothing else changes, so any difference between this block and the dense one is the window.
- **Mixture.** The attention of the dense block, and a network of $E = 4$ experts that are SwiGLU networks of hidden width 6: a router $\mathbf{W}_{\mathrm{r}}$ of shape $[16, 4]$ gives each token probabilities by softmax, the two largest are kept (a tie to the lower index), their probabilities are divided by their sum to give the combine weights, and the output is the weighted sum of the two experts' outputs. The balancing loss $\alpha_{\mathrm{bal}} E \sum_e f_e \bar p_e$ with $\alpha_{\mathrm{bal}} = 0.01$ is added to the training loss; its gradient flows through the mean probabilities $\bar p_e$ and holds the loads $f_e$ fixed (Chapter 4). Draws: the router, then each expert's $\mathbf{W}_{\mathrm{gate}}$, $\mathbf{W}_{\mathrm{up}}$, $\mathbf{W}_{\mathrm{down}}$.
- **Gated delta.** Per token: projections to $\mathbf{q}$ and $\mathbf{k}$ (each $[16, 8]$), $\mathbf{v}$ ($[16, 14]$), and two scalars per head, $a$ and $b$ ($[16, 2]$ each). Per head, $\mathbf{q}$ and $\mathbf{k}$ are L2-normalised (with the $10^{-6}$ of Chapter 5) and $\mathbf{q}$ scaled by $1/\sqrt{d_k}$; the decay is $g_t = -\operatorname{softplus}(a_t + c)$ with a learned bias $c$ per head and $\alpha_t = e^{g_t}$ (so $A = 1$ in the notation of Chapter 5), the write strength is $\beta_t = \sigma(b_t)$; the recurrence of Chapter 5 runs per head; the output of each head is RMS-normalised with a gain of width $d_v$, the heads are concatenated, and $\mathbf{W}_O$ ($[14, 16]$) projects back. The bias $c$ starts at $-3$ with no draw, so that a fresh layer has $\alpha \approx 0.95$ and a half-life near 14 tokens, longer than the 13 tokens of the task: a gate initialised as in Chapter 5's table would forget everything between one token and the next and the layer would have nothing to learn from. The short convolution and the output gate of the full layer are left out; they change what the layer can express, not what the comparison measures. Draws: $g_1$, $\mathbf{W}_Q$, $\mathbf{W}_K$, $\mathbf{W}_V$, $\mathbf{W}_a$, $\mathbf{W}_b$, the output-norm gain, $\mathbf{W}_O$, $g_2$, then the SwiGLU network.

The stacks are two layers deep: `dense`, `moe`, `gdn` and `swa` repeat one block twice, and `hybrid` (milestone 5) puts the gated delta layer first and the dense layer second. Each arm's weights are drawn from `bookrng.generator("project_architecture_comparison.blocks.<arm>")`, the token table first, then the layers in order, then the final gain.

<!-- worked: project_architecture_comparison.blocks -->
The five stacks of the project, two layers each, at width D = 16 and vocabulary 12 (the token table is tied to the head). Weights come from `bookrng.generator("project_architecture_comparison.blocks.<arm>")` in the order of the initialization routine (seeds `dense` 0x20fcf1785840a9ec, `moe` 0x56fb054b744ccea1, `gdn` 0x802a814c4aae7d05, `swa` 0x43190857da7735d5, `hybrid` 0x8a23be6e4dc7d9c1). Parameters are counted by walking the tensors that were drawn.

| arm | layers | parameters in each layer | parameters of the model | against dense |
|---|---|---:|---:|---:|
| `dense` | dense: grouped-query attention, SwiGLU | 1,960 + 1,960 | 4,128 | +0.00% |
| `moe` | mixture: grouped-query attention, 4 experts, top 2 | 2,024 + 2,024 | 4,256 | +3.10% |
| `gdn` | gated delta network, SwiGLU | 1,961 + 1,961 | 4,130 | +0.05% |
| `swa` | sliding window of 4 keys, SwiGLU | 1,960 + 1,960 | 4,128 | +0.00% |
| `hybrid` | hybrid: gated delta network then dense | 1,961 + 1,960 | 4,129 | +0.02% |
<!-- /worked -->

The parameter counts of the first four arms differ by at most 3.10 percent, and that is the stated matching of the project: **matched parameters**, within the percentage in the last column, with the mixture the one arm that is not exact because experts of hidden width 6 are the nearest whole width to the dense network's 1,152 weights per layer (4 experts of 288 weights plus the router's 64 give 1,216). The mixture therefore has the most parameters and, as milestone 2 shows, the fewest active ones. Both facts will matter when its numbers are read.

Every part of the four blocks already has a backward pass in an earlier chapter except one: the recurrence of the gated delta layer. Its backward pass is the reverse of the forward steps. With $\mathbf{S}'_t = \alpha_t \mathbf{S}_{t-1}$, $\hat{\mathbf{v}}_t = \mathbf{S}'^\top_t \mathbf{k}_t$, $\mathbf{e}_t = \beta_t(\mathbf{v}_t - \hat{\mathbf{v}}_t)$, $\mathbf{S}_t = \mathbf{S}'_t + \mathbf{k}_t \mathbf{e}_t^\top$ and $\mathbf{o}_t = \mathbf{S}_t^\top \mathbf{q}_t$, walk $t$ from the last token to the first, carrying $\mathrm{d}\mathbf{S}$, the gradient of the loss with respect to the state after token $t$ that comes from the tokens after it:

```text
function gdn_head_backward(Q, K, V, g, beta, states, dO) -> (dQ, dK, dV, dg, dbeta):
    dS = zeros(real, [d_k, d_v])                      # gradient of the state after the last token
    for t in n - 1 down to 0:
        alpha = exp(g[t])
        Sp = alpha * (states[t - 1] if t > 0 else zeros(real, [d_k, d_v]))
        e = beta[t] * (V[t, :] - matmul(Sp^T, K[t, :]))
        dQ[t, :] = matmul(states[t], dO[t, :])        # o = S^T q
        dS = dS + outer(Q[t, :], dO[t, :])            # the read adds q do^T
        dK[t, :] = matmul(dS, e)                      # S_t = Sp + k e^T
        de = matmul(dS^T, K[t, :])
        dbeta[t] = dot(de, V[t, :] - matmul(Sp^T, K[t, :]))
        dV[t, :] = beta[t] * de
        dpred = -beta[t] * de                         # e = beta (v - pred)
        dS = dS + outer(K[t, :], dpred)               # pred = Sp^T k: this is the gradient of Sp
        dK[t, :] = dK[t, :] + matmul(Sp, dpred)
        dg[t] = alpha * sum(states[t - 1] * dS)       # Sp = alpha S_{t-1}
        dS = alpha * dS                               # now the gradient of the state after token t - 1
    return (dQ, dK, dV, dg, dbeta)
```

(In the line that sets `dg[t]`, `states[t - 1]` is zero for $t = 0$.) The remaining pieces are small. The L2 normalisation $\mathbf{y} = \mathbf{x} r$ with $r = (\mathbf{x}\cdot\mathbf{x} + 10^{-6})^{-1/2}$ has the backward rule $\mathrm{d}\mathbf{x} = r\,\mathrm{d}\mathbf{y} - r^3 \mathbf{x}(\mathbf{x}\cdot\mathrm{d}\mathbf{y})$, the scale $1/\sqrt{d_k}$ multiplies the gradient of $\mathbf{q}$ before it, $\mathrm{d}g_t$ passes through $-\operatorname{softplus}$ (whose derivative is the sigmoid, so $\mathrm{d}(a_t + c) = -\sigma(a_t + c)\,\mathrm{d}g_t$, which also gives $\mathrm{d}c$ by summing over $t$) and $\mathrm{d}b_t = \beta_t(1 - \beta_t)\,\mathrm{d}\beta_t$.

The check is the one of Volume III: central differences with a step of $10^{-5}$ in double precision, against the criterion that every difference is below $10^{-6}$. Checking every coordinate of every tensor is not needed to catch a wrong rule and is costly in a plain loop; check instead two coordinates of every tensor, drawn from a generator, so that the router, the gate bias, the output-norm gain and every other tensor are touched at least twice and none is skipped because nobody thought to look at it.

<!-- worked: project_architecture_comparison.gradient_checks -->
Central differences with step 1e-05 against the hand-written backward pass, in double precision, for each stack on the reference sequence (ids 9 4 1 2 5 4 1 2 5 1 4 4 1, drawn from `project_architecture_comparison.reference`, seed 0x28bfd63b41452eb4). The loss is the mean cross-entropy of the six predictions plus the balancing loss. Two coordinates of every tensor are checked, drawn from the generator `project_architecture_comparison.gradcheck.<arm>`: for each tensor in storage order, a row `integer(rows)` then a column `integer(columns)`, twice. Storage order is the order of the draws of milestone 1 with the token table first and the final gain last, layer by layer (the dense layer: g1, W_Q, W_K, W_V, the two QK gains, W_O, g2, W_gate, W_up, W_down; the gated delta layer: g1, W_Q, W_K, W_V, W_a, W_b, the decay bias c, the output-norm gain, W_O, g2, then the network; the mixture's network: the router, then each expert's W_gate, W_up, W_down); the bias c, which has no draws, sits between W_b and the output-norm gain. A vector (a gain or the bias) is a matrix of one row: its row draw is `integer(1)`, which consumes a draw and gives 0, and then comes the column draw. Criterion: every difference below 1e-6.

| stack | loss | tensors | coordinates | largest gradient checked | largest difference |
|---|---:|---:|---:|---:|---:|
| `dense` | 2.7771 | 24 | 48 | 0.376 | below 1e-6 |
| `moe` | 3.1142 | 44 | 88 | 1.126 | below 1e-6 |
| `gdn` | 3.7583 | 28 | 56 | 1.431 | below 1e-6 |
| `swa` | 3.1992 | 24 | 48 | 0.671 | below 1e-6 |
| `hybrid` | 2.7769 | 26 | 52 | 0.530 | below 1e-6 |
<!-- /worked -->

The column of largest gradients shows that no tensor was checked against a gradient of zero. The mixture's check also tests the one place where the function is not smooth: the choice of the top two experts is a step, and a perturbation of $10^{-5}$ in a weight does not move any token across a tie on this input. That is a property of the input and not a guarantee, and it is why the check of a router is run on a drawn input and not on a hand-picked one.

The recurrent layer has a second form, the chunked one of Chapter 5, which processes $C$ tokens at a time with a triangular solve and matrix products and carries the state across chunks. It is the form a device would run at prefill, and it must compute the function the recurrent form defines. The two differ only in the order of the additions and in the algebra of the solve, so the difference to expect is rounding, and the rounding can be counted. In the recurrent form every step applies a map to the state whose norm is at most 1: the decay $\alpha_t$ is in $(0, 1]$ and $\mathbf{I} - \beta_t\mathbf{k}_t\mathbf{k}_t^\top$ has eigenvalues in $(0, 1]$ for a unit key, so the rounding of a step ($d_k + d_v + 2$ products and sums, each at most $u = 2^{-53}$ relative) is added to the state's error and never amplified, and $n$ steps give at most $n$ times that. The chunked form does the same $n$ updates through $n/C$ chunk updates and adds the $C$ steps of its triangular solve. Taking the logits of an $L$-layer model as the quantity compared, with the same first-order argument as the cache-equivalence tolerance of Volume V, Chapter 15 ($\mathrm{tol} = 2L(d_h + T)\,u\,z_{\max}$), the tolerance is

$$\mathrm{tol} = 2L\,(n + C)\,(d_k + d_v + 2)\,u\,z_{\max},$$

where $n$ is the number of tokens, $C$ the chunk size and $z_{\max}$ the largest absolute logit. It is a count and not a proof: it takes the layers after the first to pass the error on with a gain of at most one. The check that it is a good tolerance has two sides, as before: above what the reordering does, and far below the error of a mistake. The mistake to plant is the one a chunked implementation makes: forgetting to carry the state from one chunk to the next.

<!-- worked: project_architecture_comparison.chunked_recurrent -->
The `gdn` stack on the first 12 ids of the reference sequence: the logits (12 x 12) of the recurrent form and of the chunked form of Chapter 5 (`linear_attention.gated_delta_chunked`) at six chunk sizes, with the largest absolute difference and the tolerance $2L(n + C)(d_k + d_v + 2)\,u\,z_{\max}$ with $L = 2$, $n = 12$, $d_k = 4$, $d_v = 7$, $u = 2^{-53}$ and the largest logit $z_{\max}$ = 3.053. The last row starts every chunk of the chunked form from a zero state.

| chunk size C | chunks | largest difference of logits | tolerance | within tolerance |
|---|---:|---:|---:|---:|
| 1 | 12 | below 1e-13 | 2.29e-13 | yes |
| 2 | 6 | below 1e-13 | 2.47e-13 | yes |
| 3 | 4 | below 1e-13 | 2.64e-13 | yes |
| 4 | 3 | below 1e-13 | 2.82e-13 | yes |
| 6 | 2 | below 1e-13 | 3.17e-13 | yes |
| 12 | 1 | below 1e-13 | 4.23e-13 | yes |
| 4, state dropped at chunk boundaries (planted) | 3 | 3.490 | 2.82e-13 | NO |
<!-- /worked -->

The measured differences are at the level of the last bit, below $10^{-13}$, and the tolerance is above them, as a first-order count that ignores cancellations should be; the planted mistake is about thirteen orders of magnitude above the tolerance. A tolerance that is only twice what the correct code does would be fragile, and one that is a thousand times what it does would not catch a small mistake, and the table shows which side this one is on.

### Milestone 2: what each block costs

The cost model is the ledger of [Chapter 10](10-model-cost.md): a layer is a record of its mixer parameters, the network parameters it stores and the ones a token uses, its gain parameters, the numbers it caches per token, its window and the width of its scores, and the model's cost is a sum over its layers. The routine `ledger` of that chapter returns the parameters $P$, the FLOPs of a token that sees $T$ keys, and the cache bytes of a sequence of $T$ tokens; this milestone uses it unchanged for the three attention blocks and adds what a recurrence needs, which the ledger has no column for: a state that does not grow, and FLOPs that do not depend on the context. For the gated delta layer, the cached numbers per token are 0, the window is 0, and the recurrence adds $8 d_k d_v$ FLOPs per head per token (the decay, the read, the write and the read-out of a state of $d_k d_v$ numbers are $4 d_k d_v$ multiply-adds, the count of `linear_attention.tokenwise_macs`) and $d_k d_v$ numbers of state per head, whatever the length.

The definitions are those of the chapter: $N_{\mathrm{act}}$ counts the parameters that take part in the matrix products of one token (the table lookup is not a product, the tied head counts once, a mixture counts the router and the two experts it uses), and FLOPs count two per multiply-add in the matrix products, in the two attention products over the keys a token sees, and in the recurrence; norms, softmax, gates and additions are not counted. A count is only worth having if a run can contradict it. The module therefore counts the multiply-adds while the forward pass runs, in every matrix product, every attention product and every recurrence step, and the table is printed only after the counts agree: the total for a prompt of 13 tokens, the cost of the 13th token as the difference between the counts for 13 and 12 tokens, the parameters against the tensors that were drawn, and the bytes against the numbers that the forward pass holds after it (the keys and values of the last four positions for the window, the final state of each head for the recurrence).

<!-- worked: project_architecture_comparison.cost_model -->
Costs at the sizes of the project (D = 16, 2 layers, vocabulary 12, tied head) for a sequence of T = 13 tokens, numbers kept in bfloat16 (2 bytes). Parameters are the tensors that exist; active parameters are the ones in the matrix products of a token (the table lookup is not a product; the tied head counts once; the mixture counts the router and the 2 experts used). FLOPs count 2 per multiply-add in the matrix products, in the products of attention (scores and weighted values, over the keys a token sees) and in the recurrence (8 times the heads times $d_k d_v$ per token and layer); norms, softmax, gates and additions are not counted. 'Prefill' is the mean per token over a prompt of 13 tokens (token $i$ sees $i$ keys; the total for the prompt divided by 13 and rounded down to a whole number, so the window's 8,516.92 is printed 8,516), 'decode' the cost of the 13th token alone. Each FLOP count was checked against the multiply-adds counted while the forward pass ran (the total for the prompt, and the 13th token as the difference between the counts for 13 and 12 tokens), each parameter count against the tensors, and each cache against the numbers the forward pass holds.

| stack | parameters | active parameters | prefill FLOPs per token | decode FLOPs | cache bytes | state bytes | bytes held |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dense` | 4,128 | 4,032 | 8,960 | 9,728 | 832 | 0 | 832 |
| `moe` | 4,256 | 3,008 | 6,912 | 7,680 | 832 | 0 | 832 |
| `gdn` | 4,130 | 4,032 | 8,960 | 8,960 | 0 | 224 | 224 |
| `swa` | 4,128 | 4,032 | 8,516 | 8,576 | 256 | 0 | 256 |
| `hybrid` | 4,129 | 4,032 | 8,960 | 9,344 | 416 | 112 | 528 |
<!-- /worked -->

Four things in the table are worth reading. The mixture holds the most parameters (4,256) and uses the fewest (3,008 active): of its 1,216 network parameters per layer, only the router and two of the four experts take part in the products of a token, and the saving in decode FLOPs against the dense stack is 21 percent (7,680 against 9,728), not 50, because the attention, the head and the router are still there. The sliding window costs fewer FLOPs than the dense block at 13 tokens and holds under a third of its bytes (256 against 832), but a window is not a fixed saving: it bites only when the context exceeds it. The gated delta stack is the only one whose bytes do not depend on the number of tokens, and it holds 224 of them at this size against the dense stack's 832. And the gated delta stack's prefill and decode FLOPs per token are the same, because the recurrence does the same work at every token, while the dense stack's decode token costs more than its mean prefill token, because it sees all 13 keys and the mean token sees 7. The coincidence that the dense and gated delta stacks have the same mean prefill FLOPs, 8,960, is arithmetic and not a principle: the mean number of keys over 13 tokens is 7, and $4 H d_h \cdot 7 = 448$ per layer equals $8 \cdot 2 \cdot 4 \cdot 7 = 448$, the recurrence of this layer.

What a decode step costs depends on the context, and the ledger gives it at any context from the same formulas. Counted runs cannot reach 4,096 tokens in a plain loop, but the formulas that agreed with the count at 13 tokens extend, and the table shows them.

<!-- worked: project_architecture_comparison.cost_scaling -->
The same ledger at four contexts (tokens a decode step sees; sizes of the project; bfloat16). The attention stacks' decode FLOPs and cache grow with the context; the sliding window stops growing at its 4 keys; the gated delta stack and the state of a hybrid do not grow at all. The gated delta stack needs fewer decode FLOPs than the dense one from a context of 8 tokens, and holds fewer bytes from a context of 4 tokens.

| stack | FLOPs at 13 | bytes at 13 | FLOPs at 256 | bytes at 256 | FLOPs at 4,096 | bytes at 4,096 | FLOPs at 65,536 | bytes at 65,536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense` | 9,728 | 832 | 40,832 | 16,384 | 532,352 | 262,144 | 8,396,672 | 4,194,304 |
| `moe` | 7,680 | 832 | 38,784 | 16,384 | 530,304 | 262,144 | 8,394,624 | 4,194,304 |
| `gdn` | 8,960 | 224 | 8,960 | 224 | 8,960 | 224 | 8,960 | 224 |
| `swa` | 8,576 | 256 | 8,576 | 256 | 8,576 | 256 | 8,576 | 256 |
| `hybrid` | 9,344 | 528 | 24,896 | 8,304 | 270,656 | 131,184 | 4,202,816 | 2,097,264 |
<!-- /worked -->

At the size of the project the crossovers come within a handful of tokens, because the attention blocks cache so little (32 bytes per token and layer) and the state is small too (112 bytes per layer); the point of the table is the shape, not the numbers. The dense and mixture stacks grow linearly in both FLOPs and bytes, the window flattens at its fourth key, the gated delta stack is flat from the start, and the hybrid grows at half the dense slope because one of its two layers keeps a cache. Sizes at which those differences are the whole story need a bigger model, and the same ledger prices one:

<!-- worked: project_architecture_comparison.big_shape -->
The ledger at a size where the blocks differ in what they cost: width 1,024, 24 layers, vocabulary 32,000, 16 query heads and 4 key-value heads of width 64, SwiGLU hidden width 2,816, a window of 1,024 keys, a mixture of 16 experts of hidden width 704 with 2 used, and a gated delta layer of 16 heads with $d_k = d_v = 64$; the hybrid has a gated delta layer in three of every four positions and a dense layer in the fourth. The sizes are not matched between the stacks (the gated delta layer holds more parameters than the attention layer here); the table reports what each costs. Millions of parameters and of FLOPs, bfloat16, MiB for one sequence.

| stack | parameters (M) | active (M) | decode MFLOPs at 4,096 | decode MFLOPs at 131,072 | MiB held at 131,072 |
|---|---:|---:|---:|---:|---:|
| `dense` | 303.4 | 303.3 | 1009.3 | 13491.5 | 3072.0 |
| `moe` | 926.6 | 199.9 | 802.4 | 13284.7 | 3072.0 |
| `gdn` | 341.9 | 341.8 | 696.3 | 696.3 | 3.0 |
| `swa` | 303.4 | 303.3 | 707.3 | 707.3 | 24.0 |
| `hybrid` | 332.3 | 332.2 | 774.5 | 3895.1 | 770.2 |
<!-- /worked -->

At a width of 1,024 and 24 layers the dense stack holds 3,072 MiB of cache for one sequence of 131,072 tokens and the gated delta stack holds 3.0 MiB of state; the hybrid, with one attention layer in four, holds a quarter of the dense cache plus the small state, which is the 3:1 pattern of Chapter 5 priced at this size. The sizes are not matched between the stacks here, and the table is for reading costs and not quality.

### Milestone 3: a parity harness that fails when it should

A harness is a function from a candidate to a verdict, and it is worth only as much as the mistakes it has been shown to catch. [Chapter 2](02-importing-models.md) built one for a decoder loaded from someone else's checkpoint; this one serves a different purpose, which is to say, for each block of the engine, whether an implementation of the block computes what the reference implementation computes. The ingredients are the chapter's: a **golden** (the reference outputs stored in a file), three measures of the difference, and a tolerance derived from a count of roundings and not chosen.

The golden is a safetensors file written byte by byte by `write_safetensors` of Chapter 2. For each arm it holds the ids of the reference sequence (13 ids drawn from `project_architecture_comparison.reference`), the hidden state after the embeddings and after each layer (the output of each block), the output of the final norm, and the logits, all stored as 32-bit floats.

<!-- worked: project_architecture_comparison.golden_files -->
Goldens of the five stacks on the reference sequence (ids 9 4 1 2 5 4 1 2 5 1 4 4 1), each a safetensors file written by `importing_models.write_safetensors` (Chapter 2): the ids as 32-bit integers, then the hidden state after the embeddings and after each layer, the output of the final norm and the logits as 32-bit floats (the 64-bit result rounded to the nearest 32-bit float), little-endian, back to back in this order. The `dense` file holds `ids`, `hidden_00`, `hidden_01`, `hidden_02`, `final_norm`, `logits`. The file is an 8-byte little-endian length prefix, the header text, and the 4,004 bytes of data. The header is compact JSON (separators `,` and `:`, no other spaces), in the key order below, with the metadata first, padded with spaces on the right to a multiple of 8 bytes: 568 bytes for every stack (the text is 566 bytes for `dense`, and the padding absorbs the different lengths of the other arms' names). The header text of `dense` is these lines joined with no separator (the line breaks are for the page); the other stacks differ only in the last word of the generator name, which is the arm:

```text
{"__metadata__":{"generator":"project_architecture_comparison.blocks.dense",
"arithmetic":"64-bit, stored as 32-bit",
"tolerance_basis":"rounding count x 2^-24"},
"ids":{"dtype":"I32","shape":[13],"data_offsets":[0,52]},
"hidden_00":{"dtype":"F32","shape":[13,16],"data_offsets":[52,884]},
"hidden_01":{"dtype":"F32","shape":[13,16],"data_offsets":[884,1716]},
"hidden_02":{"dtype":"F32","shape":[13,16],"data_offsets":[1716,2548]},
"final_norm":{"dtype":"F32","shape":[13,16],"data_offsets":[2548,3380]},
"logits":{"dtype":"F32","shape":[13,12],"data_offsets":[3380,4004]}}
```

Digest: the first 16 hexadecimal digits of the SHA-256 of the whole file. Stack, tensors, bytes and digest:

| stack | tensors | file bytes | digest |
|---|---:|---:|---:|
| `dense` | 6 | 4,580 | 1f6196ce6dffd1fd |
| `moe` | 6 | 4,580 | 9c01d6aaecd160ce |
| `gdn` | 6 | 4,580 | 1f02617b8ff996ef |
| `swa` | 6 | 4,580 | 24958349d5b5f684 |
| `hybrid` | 6 | 4,580 | 405fae79d414c53d |
<!-- /worked -->

The golden is a 64-bit result stored in 32 bits, so it carries a relative error of at most $2^{-24}$ per number. The candidate for the correct-code test is the stack with every weight rounded to 32 bits, run in 64-bit arithmetic: what a published checkpoint in 32 bits would give a loader that computes in higher precision. The tolerance for the logits is the parity tolerance of Chapter 2, $\tau_{\mathrm{par}} = (N_{\mathrm{rnd}} + 1)\,2^{-24}$, where $N_{\mathrm{rnd}}$ is the number of roundings along the longest chain of dependent operations (the rule: a sum of $n$ products costs $n$ roundings, each further operation on its result one) and the one added is the golden's own storage. The count is made per block and differs between them: the attention blocks reuse the stage list of Chapter 2 (with the window's four keys in place of the context for the softmax), the mixture replaces the dense network's stages by the router's dot product and softmax, the renormalisation, the expert's stages and the weighted sum of the two experts, and the gated delta layer adds the $d_k + 5$ roundings of one recurrence step for each of the $n$ tokens along the chain of the state, because the state at the last token depends on every step before it. The first table below lists every stage of the four kinds of layer with its count, and the second gives the bound of each tap (the embeddings, the output of each layer, the logits) as one more than the roundings accumulated up to it. The three tests that follow from the one bound $\varepsilon = \tau_{\mathrm{par}}$ are: rel_l2 at most $\varepsilon$; max_abs at most $\varepsilon$ times the length of the golden; and $1 - $ cosine at most $\varepsilon^2/2$, which is what a relative error of $\varepsilon$ can change in the direction of a vector.

<!-- worked: project_architecture_comparison.rounding_stages -->
The roundings counted along the longest chain of one layer, stage by stage, for a token that sees 13 keys (the window of `swa` keeps 4 of them in the softmax and the weighted sum; the recurrence runs along the 13 tokens of the sequence). The rule is that of Chapter 2: a dot product of length $n$ costs $n$, an RMS norm over $n$ numbers $n + 6$, and each further operation on a result one. The L2 normalisation is the sum of $d_k$ squares, the addition of $\varepsilon$, the square root, the reciprocal and the product, $d_k + 4$; a step of the recurrence is the decay product, the dot of length $d_k$, the subtraction, the product with $\beta$, the product with the key and the addition to the state, $d_k + 5$; the QK-norm is $d_h + 6$ as in Chapter 2; the router's renormalisation is a sum and a division, and the weighted sum of the $K$ experts is $K$. A dash means the layer has no such stage. The softplus of the decay and the sigmoid of the write strength are on branches shorter than the chain through the L2-normalised key and do not enter. The hybrid has the gated delta layer first and the dense layer second, so its layers cost those of the `gdn` and `dense` columns.

| stage | dense layer | window layer | mixture layer | gated delta layer |
|---|---:|---:|---:|---:|
| RMS norm 1 | 22 | 22 | 22 | 22 |
| projection (dot of length D) | 16 | 16 | 16 | 16 |
| L2 normalisation of q and k | - | - | - | 8 |
| QK-norm | 10 | 10 | 10 | - |
| rotation | 4 | 4 | 4 | - |
| scores (dot of length d_h, scale) | 5 | 5 | 5 | - |
| softmax over the keys | 16 | 7 | 16 | - |
| weighted sum of values | 13 | 4 | 13 | - |
| recurrence, 13 tokens of d_k + 5 | - | - | - | 117 |
| read-out (dot of length d_k) | - | - | - | 4 |
| output norm over d_v | - | - | - | 13 |
| W_O of the attention layer (dot of length 4 d_h) | 16 | 16 | 16 | - |
| W_O of the recurrent layer (dot of length 2 d_v) | - | - | - | 14 |
| residual add (mixer) | 1 | 1 | 1 | 1 |
| RMS norm 2 | 22 | 22 | 22 | 22 |
| gate or up projection (dot of length D) | 16 | 16 | - | 16 |
| router (dot of length D) | - | - | 16 | - |
| router softmax over E experts | - | - | 7 | - |
| renormalisation of the kept pair | - | - | 2 | - |
| expert gate or up (dot of length D) | - | - | 16 | - |
| silu and product with up | 4 | 4 | 4 | 4 |
| W_down (dot of length F) | 24 | 24 | - | 24 |
| expert W_down (dot of length FE) | - | - | 6 | - |
| weighted sum of the K experts | - | - | 2 | - |
| residual add (network) | 1 | 1 | 1 | 1 |
| **layer total** | **170** | **152** | **179** | **262** |

The head adds a final RMS norm (16 + 6) and the output projection (16), 38 in all, and the golden's own storage adds 1. The bound on a tap is $(N + 1)\,2^{-24}$ with $N$ the sum of the counts of the layers up to the tap (and of the head for the logits); the embeddings tap, before any layer, has $N = 0$. Per stack, in units of $2^{-24}$:

| stack | layer 0 | layer 1 | head | embeddings tap | layer 0 tap | layer 1 tap | logits tap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dense` | 170 | 170 | 38 | 1 | 171 | 341 | 379 |
| `moe` | 179 | 179 | 38 | 1 | 180 | 359 | 397 |
| `gdn` | 262 | 262 | 38 | 1 | 263 | 525 | 563 |
| `swa` | 152 | 152 | 38 | 1 | 153 | 305 | 343 |
| `hybrid` | 262 | 170 | 38 | 1 | 263 | 433 | 471 |
<!-- /worked -->

<!-- worked: project_architecture_comparison.parity_table -->
The harness on correct candidates. The candidate is the stack with every weight rounded to 32 bits and the arithmetic in 64 bits; the golden is the file above, so the two differ by the rounded weights and by the golden's own rounding. The bound is $\tau_{\mathrm{par}} = (N_{\mathrm{rnd}} + 1)\,2^{-24}$ on the relative error of the logits, where $N_{\mathrm{rnd}}$ counts the roundings along the longest chain of dependent operations by the rule of Chapter 2 (a sum of $n$ products costs $n$, each further operation one; the stage counts are in the table above, and the bound of each tap in the second table), and the one that is added is the golden's storage. The tests derived from that one bound are rel_l2 at most the bound, max_abs at most the bound times the length of the golden, and 1 - cosine at most half the bound squared.

| stack | $N_{\mathrm{rnd}}$ | bound $\tau_{\mathrm{par}}$ | rel_l2 of logits | max_abs of logits | all three tests pass | bound over rel_l2 |
|---|---:|---:|---:|---:|---:|---:|
| `dense` | 378 | 2.26e-5 | 2.17e-7 | below 1e-6 | yes | 104 |
| `moe` | 396 | 2.37e-5 | below 1e-7 | below 1e-6 | yes | 367 |
| `gdn` | 562 | 3.36e-5 | 2.52e-7 | 1.62e-6 | yes | 133 |
| `swa` | 342 | 2.04e-5 | 1.66e-7 | below 1e-6 | yes | 123 |
| `hybrid` | 470 | 2.81e-5 | 1.13e-7 | below 1e-6 | yes | 247 |

The dense stack against the decoder of Chapter 1 (`modern_decoder.forward`, same weights, 64 bits both): rel_l2 below 1e-14.
<!-- /worked -->

All five stacks pass all three tests, with a bound between 104 and 367 times above the measured error (the last column), a ratio of the order that Chapter 2 found for the 32-bit pass of the decoder. The dense stack also agrees with the decoder of Chapter 1 to rounding when both run in 64 bits, which is the last row's claim: the engine's dense block and the reference decoder are two independent implementations of one function.

Now the other direction. A harness that passes correct code has shown nothing until it has been shown to fail on wrong code, so each row of the next table plants one mistake in a stack that passes, runs the harness, and records which of the three tests fired and which tap, going forward through the network, was the first to leave its bound. The mistakes are the ones that the blocks make easy: a key-value head assigned by $h \bmod 2$ in place of $\lfloor h/2 \rfloor$; queries left unrotated; QK-norm gains dropped; a window of 5 keys; combine weights that are the raw probabilities; a decay gate with the wrong sign. Each is planted in layer 1 only, so the taps before it must stay inside their bounds and the first tap beyond its bound must be the output of layer 1. The last three rows are not mistakes of the blocks but of the harness's resolution.

<!-- worked: project_architecture_comparison.planted_bugs -->
Mistakes planted in a stack that otherwise passes (the candidate of the previous table), compared with its golden. Each row lists the relative error of the logits, 1 - cosine, which of the three tests (max_abs, rel_l2, 1 - cosine) failed, and the first tap whose rel_l2 exceeded its bound: the embeddings, then layer 0, layer 1, then the logits. A planted mistake in layer 1 leaves the taps before it inside their bounds.

- `group_map`: query head h reads key-value head h mod 2, not h // 2
- `rotate_keys_only`: queries are not rotated
- `no_qk_norm_gain`: the QK-norm gains are not applied
- `window_off_by_one`: the window holds 5 keys, not 4
- `no_renorm`: the combine weights are the raw probabilities
- `gate_sign`: the decay gate is +softplus, so the state grows
- `group_map`: as above, in the dense layer of the hybrid
- `final_gain_doubled`: the final norm gain is doubled: every logit is twice as large
- `weight_1e-9`: the weight in row 0, column 0 of W_Q ($[16, 16]$) of layer 0, after the rounding to 32 bits, is increased by 1e-9
- `weight_1e-3`: the same weight is increased by 1e-3

| stack | planted mistake | rel_l2 | 1 - cosine | tests failed (max_abs, rel_l2, cosine) | first tap beyond bound |
|---|---|---:|---:|---|---|
| `dense` | `group_map` in layer 1 | 5.24e-1 | 1.47e-1 | caught caught caught | layer 1 |
| `dense` | `rotate_keys_only` in layer 1 | 2.17e-1 | 2.35e-2 | caught caught caught | layer 1 |
| `dense` | `no_qk_norm_gain` in layer 1 | 2.74e-2 | 3.66e-4 | caught caught caught | layer 1 |
| `swa` | `window_off_by_one` in layer 1 | 1.85e-1 | 1.70e-2 | caught caught caught | layer 1 |
| `moe` | `no_renorm` in layer 1 | 4.27e-2 | 9.10e-4 | caught caught caught | layer 1 |
| `gdn` | `gate_sign` in layer 1 | 1.40e-1 | 9.52e-3 | caught caught caught | layer 1 |
| `hybrid` | `group_map` in layer 1 | 8.07e-1 | 3.54e-1 | caught caught caught | layer 1 |
| `dense` | `final_gain_doubled` | 1.00e0 | 2.36e-14 | caught caught - | logits |
| `dense` | `weight_1e-9` | 2.17e-7 | 2.36e-14 | - - - | none |
| `dense` | `weight_1e-3` | 8.77e-5 | 3.84e-9 | caught caught caught | layer 0 |
<!-- /worked -->

All seven mistakes of the blocks are caught by all three tests, and the first tap to leave its bound is layer 1 each time, the layer that holds the mistake. That is the use of storing the output of every block and not only the logits: the harness points at a layer, not just at a model. The last three rows show the edges. Doubling the final norm's gain doubles every logit: the direction does not change, so $1 - $ cosine is zero up to rounding and the cosine test does not fire, while the other two do, and a harness that used the cosine alone, as the loose gate of some model-conversion scripts does, would pass a model whose every logit is wrong by a factor of two. And a weight changed by $10^{-9}$ is not seen by any test: the tolerance is the precision of the golden, about $10^{-5}$ relative here, and a mistake smaller than that cannot be told from storing the golden in 32 bits. The same weight changed by $10^{-3}$ is caught, in layer 0, where it was planted. A tolerance is a resolution, and the table is the way to know what it is.

### Milestone 4: a comparison that can say "undecided"

The comparison is useful only if the task makes the blocks differ in a way that can be predicted from what each block is, and only if the way it is run does not decide the outcome. Both are decided in advance and written down.

**The task.** It must need two kinds of information, a local one and a long-range one, and the long-range one must not be recoverable from the local one. A task in which each token follows from the previous two would be solved by a window of 2. The task here puts the secret at the start and makes the answer depend on it at every pair thereafter.

<!-- worked: project_architecture_comparison.task_statement -->
The task needs two kinds of information. A sequence has 13 tokens: a header token $6 + h$ at position 0, with $h$ drawn from $\{0, \dots, 5\}$, then six pairs $(x_i, y_i)$ in which $x_i$ is a symbol drawn from $\{0, \dots, 5\}$ and $y_i = (x_i + h) \bmod 6$. Draws from `bookrng.generator("project_architecture_comparison.data.<seed>")` (the example below: `project_architecture_comparison.example`, seed 0x4d462974ecb77778), seven per sequence in this order: `integer(6)` for $h$, then `integer(6)` for each of $x_1, \dots, x_6$. The model reads the sequence and at the position of $x_i$ (positions 1, 3, 5, 7, 9, 11) predicts $y_i$; only these six predictions are trained and scored. The symbol $x_i$ is the previous token, the header is up to eleven positions back. A model that knows $x_i$ but not $h$ can do no better than $\ln 6 = 1.7918$ nats; one that finds the header reaches 0.

| sequence (ids) | h | x_1 .. x_6 | y_1 .. y_6 |
|---|---:|---|---|
| 9 2 5 0 3 2 5 2 5 1 4 1 4 | 3 | 2 0 2 2 1 1 | 5 3 5 5 4 4 |
| 7 2 3 2 3 1 2 0 1 0 1 2 3 | 1 | 2 2 1 0 0 2 | 3 3 2 1 1 3 |
| 8 2 4 1 3 4 0 2 4 3 5 4 0 | 2 | 2 1 4 2 3 4 | 4 3 0 4 5 0 |

A window of 4 keys cannot see the header from the position of $x_i$ once $2i - 1 - 3 > 0$. After 2 layers a position reaches back $2 \cdot 3 = 6$ positions, so 3 of the six predictions (those for $i = 4, 5, 6$) cannot see the header at all, and a windowed stack's mean loss is at least 3 / 6 of $\ln 6$ = 0.8959 nats, even if it passes the header through the first layer to the second (one layer alone: 4 of 6, 1.1945).
<!-- /worked -->

The loss of a model that learned only the local half is $\ln 6$, and the windowed stack has a floor of its own that can be derived from the mask before anything is trained: the number of the six predictions whose positions cannot reach the header through two windows of 4 keys, times $\ln 6$, over six. That floor is an expected result of the architecture and not of any training run.

**The predictions.** Before the reported runs, write down what you expect and why, as claims about paired differences with a margin and a side, and fix the rule that turns an interval into a verdict. A prediction written after the runs is a description. The five below are derived from the mechanisms: the window cannot see the header, a state of fixed size can carry it, narrower experts hold less of the table the network has to learn, and a single attention layer is enough for the lookup.

<!-- worked: project_architecture_comparison.registered -->
Predictions registered before the six runs of each arm were made, from the mechanisms and not from their outcome, each as a claim about a paired difference of the mean loss over the six scored predictions (nats), the first arm minus the second, and a margin. The verdict rule: 'supported' when the 95 percent interval lies wholly on the predicted side of the margin, 'refuted' when wholly on the other, 'undecided' otherwise.

| claim | statement |
|---|---|
| H1 | the window cannot see the header: swa - dense >= +0.5 |
| H2 | the state can carry the header: gdn - dense <= +0.1 |
| H3 | experts of width 6 are narrower than one of 24: moe - dense >= +0.05 |
| H4 | one attention layer is enough: hybrid - dense <= +0.1 |
| H5 | the hybrid is no worse than the pure recurrent stack: hybrid - gdn <= +0.05 |
<!-- /worked -->

**The way it is run.** The arms have matched parameters (milestone 1) and the same skeleton, depth and width. They are trained on the same sequences in the same order for a given seed, so the paired differences remove the variation of the data; they are scored on the same held-out sequences. Each arm chooses its peak learning rate from a short grid by one run on a selection seed that none of the reported seeds uses, so no arm is handicapped by a rate chosen for another, and the choice is made on a seed that is not reported, so the reported intervals are not conditional on the selection. The recipe, its draws and its seeds are in the table.

<!-- worked: project_architecture_comparison.selection -->
Recipe: Adam with the moments of Volume III (0.9, 0.95, epsilon 1e-8, 1-based step in the bias corrections), no weight decay, 120 steps numbered $t = 0, \dots, 119$ with the rate $\eta_t = \eta_{\max}\,(t + 1) / 10$ for $t < 10$ (so step 9 is the first at the peak) and $\eta_t = \eta_{\max}\,(1 - 0.9\,(t - 10) / 110)$ for $t \ge 10$ (a linear fall to a tenth of the peak at step 120, which is never reached; step 119 uses $1 - 0.9 \cdot 109 / 110$ of it), a clip of the batch gradient's global norm (the square root of the sum of squares of every entry) at 1: when the norm exceeds it the gradient is multiplied by clip over norm, then 120 steps of 8 sequences, each sequence the next seven draws of the data generator, the gradient being the mean over the batch. The loss that is differentiated is the mean over the six scored predictions plus, for a mixture, the balancing loss with weight 0.01; the 'training loss' columns of this chapter report the cross-entropy part alone (the balancing term is a regulariser of the router and is not a quality). Each arm chose its peak rate from 0.01, 0.02, 0.04 by the held-out cross-entropy on 120 sequences after one run of 120 steps on the selection seed (`s0`, data from `project_architecture_comparison.data.s0`, held-out set from `project_architecture_comparison.eval.s0`, initial weights from `project_architecture_comparison.init.<arm>.s0`, no nudge), a seed none of the reported runs uses; a tie goes to the lower rate. The ratio column is the runner-up's loss over the best loss, to one digit (these are single runs that end at a few hundredths, whose further digits are not reproducible); the last column says whether it exceeds 1.1, so that the choice does not hinge on a small difference.

| arm | peak rate chosen | runner-up loss over best loss | ratio above 1.1 |
|---|---:|---:|---:|
| `dense` | 0.04 | 1.2 | yes |
| `moe` | 0.04 | 8.4 | yes |
| `gdn` | 0.02 | 4.2 | yes |
| `swa` | 0.04 | 1.0 | no |
| `hybrid` | 0.02 | 6.5 | yes |
<!-- /worked -->

The table shows that the choice of rate was not a coin toss for four of the five arms (the runner-up's loss is at least 1.2 times the best's), and that for the window it was: the window's best rate is not clearly better than its runner-up, which matters little, because every rate leaves it near the floor of the task. The rest of the recipe is the same for every arm.

Now run every arm on the six reported seeds, each with its four nudged copies. The weights of arm `a` and seed `s` come from `project_architecture_comparison.init.<a>.<s>`, drawn in the order of milestone 1, the data from `project_architecture_comparison.data.<s>` and the held-out set from `project_architecture_comparison.eval.<s>`. A run with the same seed but a different arm sees the same training sequences in the same order.

<!-- worked: project_architecture_comparison.comparison -->
Held-out loss (mean cross-entropy of the six scored predictions, nats) after 120 steps, on the 6 reported seeds (`r1`, `r2`, `r3`, `r4`, `r5`, `r6`; data `project_architecture_comparison.data.<seed>`, held-out `project_architecture_comparison.eval.<seed>`, weights `project_architecture_comparison.init.<arm>.<seed>`, the seed of a stream is a function of its full name by the rule of Chapter 17: `project_architecture_comparison.data.r1` has the seed 0x341a22768f472695, `project_architecture_comparison.data.r2` 0xeec0a40fd2891cd0, and every other stream named here has the seed of its own name). All arms see the same sequences in the same order for a seed. Mean over seeds, sample standard deviation (divisor $n - 1$), the 95 percent interval of the mean (Student $t$ with 5 degrees of freedom, multiplier 2.5706), and the training loss: the batch cross-entropy of the last 10 steps (the balancing term excluded), averaged over the steps and the seeds. A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation.

| arm | peak rate | parameters | mean | sd | 95 percent interval of the mean | training loss, last 10 steps |
|---|---:|---:|---:|---:|---:|---:|
| `dense` | 0.04 | 4,128 | 0.566 | 0.307 | 0.244 to 0.889 | 0.573 |
| `moe` | 0.04 | 4,256 | 0.578 | 0.556 | -0.005 to 1.161 | 0.608 |
| `gdn` | 0.02 | 4,130 | below 0.1 | below 0.05 | digits of the ends do not survive | below 0.1 |
| `swa` | 0.04 | 4,128 | 1.552 | 0.143 | 1.401 to 1.703 | 1.566 |
| `hybrid` | 0.02 | 4,129 | below 0.05 | below 0.1 | digits of the ends do not survive | below 0.1 |
<!-- /worked -->

Read the table before the contrasts. The window sits where the derivation put it: its mean is 1.552, between the 0.896 of the best a windowed stack could do and the $\ln 6 = 1.792$ of a model that has not found the header at all, and nearer the second, because passing a header through the first layer to the second is something the stack has to learn and these runs mostly did not. The dense and mixture stacks sit at about 0.57 with enormous spread (standard deviations of 0.31 and 0.56 on a mean of 0.57): at 120 steps most of their runs are still partly on the plateau at which the header is not used, and one dense run has found it (below 0.02). The recurrent stacks are far lower, with a mean below 0.1, and their digits are printed only as bounds, because a run that has converged to a loss of a few hundredths moves by about that much when its first weight moves by $10^{-12}$: the end of training is chaotic at that scale, and the rule of this chapter does not let a digit through that another implementation could not reproduce. The dense, mixture and window numbers did not move in any digit shown.

That is the reading of the table that the nudge rule forces: what the recurrent stacks have is a loss that is small, not a loss of a particular small value, and the comparison with the others rests on the difference being large against the spread, not on a digit. The next table is the same data seen through the predictions that were written down in advance.

<!-- worked: project_architecture_comparison.contrasts -->
The registered contrasts, paired by seed (every arm of a seed shares the sequences and the held-out set): the mean of the six differences first arm minus second, and the 95 percent t interval of that mean (5 degrees of freedom, multiplier 2.5706). A negative difference favours the first arm. 'ge' claims the difference is at least the margin, 'le' at most. A verdict that differs between a run and its four nudged copies would print 'unstable'.

| claim | contrast | predicted | mean difference | 95 percent interval | verdict |
|---|---|---|---:|---:|---|
| H1 | `swa` - `dense` | ge +0.5 | +0.986 | +0.601 to +1.371 | supported |
| H2 | `gdn` - `dense` | le +0.1 | about -0.5 | about -0.9 to about -0.2 | supported |
| H3 | `moe` - `dense` | ge +0.05 | +0.012 | -0.721 to +0.744 | undecided |
| H4 | `hybrid` - `dense` | le +0.1 | about -0.5 | about -0.86 to about -0.2 | supported |
| H5 | `hybrid` - `gdn` | le +0.05 | within 0.05 of zero | digits of the ends do not survive | undecided |
<!-- /worked -->

Three predictions are supported and two are undecided, and the manner of each is worth stating. H1 holds with room to spare: a window of 4 keys loses about a full nat to the dense stack, and the lower end of the interval is still above the margin of 0.5. H2 and H4 are supported by a large margin and for a reason the prediction did not have: it claimed that the gated delta stack and the hybrid would be at most 0.1 behind the dense stack, and they are about half a nat ahead. The prediction was one-sided and weak, the data went well past it, and the right thing to report is that the dense stack at this budget is the slow learner, not that the recurrent layer is better at the task in general: the dense stack must discover that one head should attend to the header token, and 120 steps of 8 sequences do not always suffice, while the recurrent layer's state carries the header from the first token whether or not the model has learned to look for it. Trained for several times as long, the dense stack would be expected to catch up; this project does not run that, and an extension asks for it. H3 is undecided in the strict sense: the mixture's mean difference from the dense stack is a hundredth of a nat but its interval spans almost a nat and a half, since its runs split between solved and unsolved (two stay above one nat, the other four lie between 0.16 and 0.30) and six runs cannot say whether the narrower experts cost anything. H5, whether the hybrid is no worse than the pure recurrent stack, is undecided because both are so near zero that the interval of the difference does not fit inside a margin of 0.05 once the chaos of the last digits is counted; an experiment that wanted to settle it would need more seeds, as Chapter 9's table of seeds needed predicts.

Where the loss is lost is visible in the position the stack could not reach.

<!-- worked: project_architecture_comparison.by_position -->
Held-out loss of each of the six predictions, mean over the six seeds (nats). A prediction at position $t = 2i - 1$ has the header $t$ positions back; the bound for a window of 4 keys in two layers is $t \le 6$.

| prediction | `dense` | `swa` | `gdn` | `moe` | `hybrid` |
|---|---:|---:|---:|---:|---:|
| $y_1$ (position 1, header 1 back) | 0.59 | 1.00 | below 0.1 | 0.55 | below 0.1 |
| $y_2$ (position 3, header 3 back) | 0.55 | 1.12 | below 0.1 | 0.58 | below 0.1 |
| $y_3$ (position 5, header 5 back) | 0.54 | 1.81 | below 0.05 | 0.58 | below 0.1 |
| $y_4$ (position 7, header 7 back) | 0.56 | 1.80 | below 0.05 | 0.57 | below 0.05 |
| $y_5$ (position 9, header 9 back) | 0.56 | 1.80 | below 0.05 | 0.57 | below 0.05 |
| $y_6$ (position 11, header 11 back) | 0.59 | 1.79 | 0.05 | 0.61 | below 0.05 |
<!-- /worked -->

The window's column is the derivation, drawn from the data: for $y_1$ and $y_2$, whose positions are within one window of the header, its loss is 1.00 and 1.12, well under $\ln 6$ though not zero, and for $y_3$ onward, whose positions are beyond it, the loss is at or just above $\ln 6$, 1.81 to 1.79, because the stack has nothing that carries the header across (it would have had to relay it through the first layer, and it did not). The dense, mixture and recurrent columns are flat in the position, which is what they should be: nothing in their mixers depends on how far back the header is.

<!-- worked: project_architecture_comparison.moe_balance -->
Share of the expert slots taken by the busiest of the 4 experts on the first 30 sequences of the held-out stream `project_architecture_comparison.eval.<seed>` (the first 30 draws of sequences, all 13 positions of each, so 390 tokens and 780 slots per layer), mean over the six seeds, for the two mixture layers of the `moe` arm; a slot is one (token, chosen expert) pair and the share is the largest of the four counts over their sum: layer 0 0.36, layer 1 0.32 (1 / 4 = 0.25 when the router is perfectly balanced, 0.5 when two experts take everything). The balancing loss has weight 0.01.
<!-- /worked -->

The balancing loss did what it is for. The busiest of four experts takes about a third of the slots and not a half; it is not perfectly balanced, and with a weight of 0.01 over 120 steps it should not be. Milestone 1's gradient check covered the router and the balancing term, and this number is the check that the term does something in a run.

The figure puts the blocks on the plane that a decision is made on: what a token costs and what the model achieves. The cost axis is the decode FLOPs of the 13th token, from the ledger of milestone 2; the quality axis is the held-out loss with its 95 percent interval over the six seeds.

![Held-out loss against the decode FLOPs of the 13th token for five stacks. The dense stack is at 9,728 FLOPs with a mean near 0.57 and a wide bar; the mixture is at 7,680 FLOPs with a mean near 0.58 and the widest bar; the gated delta stack is at 8,960 and the hybrid at 9,344, both near zero with very short bars; the window is at 8,576 near 1.55. A dashed horizontal line marks ln 6.](img/project-architecture-quality.svg)

At this size the FLOPs axis barely separates the blocks: all five lie within 2,048 FLOPs of one another, a fifth of the dense stack's cost, while the quality axis spans more than a nat and a half. That is the honest content of the plane. The cost model says the blocks are almost equal in arithmetic at 13 tokens and diverge in memory and in arithmetic only with the context (milestone 2); the comparison says they are very unequal in what they can learn from 13 tokens. A reader who looked at only one of the two would conclude the opposite of a reader who looked at both.

### Milestone 5: a hybrid

A hybrid interleaves the two kinds of mixer so that each does what it is good at: the recurrent layers carry a summary of the past at a fixed cost, and the attention layers, a fraction of the depth, retrieve exactly. The stack here is the smallest that has both: the gated delta layer first, the dense layer second, drawn in that order from `project_architecture_comparison.blocks.hybrid` and, for the comparison, `project_architecture_comparison.init.hybrid.<seed>`. It needed nothing new in the engine, since the stack is a list of (mixer, network) pairs, which is what the interface of milestone 1 is for; the ledger of milestone 2 priced it as one gated delta layer and one dense layer, and the parity harness of milestone 3 passed it and caught a planted mistake in its attention layer.

<!-- worked: project_architecture_comparison.hybrid_tradeoff -->
The hybrid beside the two pure stacks it is made of (two layers each; the hybrid has the gated delta layer first and the dense layer second). Quality is the held-out loss of the previous table; costs are the ledger's, at the task's 13 tokens and at 4,096 tokens by the same formulas.

| stack | held-out loss | decode FLOPs at 13 | bytes held at 13 | decode FLOPs at 4,096 | bytes held at 4,096 |
|---|---:|---:|---:|---:|---:|
| `dense` | 0.566 | 9,728 | 832 | 532,352 | 262,144 |
| `gdn` | below 0.1 | 8,960 | 224 | 8,960 | 224 |
| `hybrid` | below 0.05 | 9,344 | 528 | 270,656 | 131,184 |
<!-- /worked -->

Against the pure dense stack, the hybrid learned the task in these runs where the dense stack mostly had not (H4, supported), at a decode cost of 9,344 FLOPs at 13 tokens against 9,728 and half the memory, 528 bytes against 832; at 4,096 tokens it costs 270,656 FLOPs against 532,352 and holds 131,184 bytes against 262,144, which is the half-slope of milestone 2 priced at the length where it matters. Against the pure recurrent stack the quality is not distinguishable here (H5, undecided), and the cost is higher: at 13 tokens a few hundred FLOPs and 304 more bytes, and at 4,096 tokens the attention layer's linear growth, 270,656 FLOPs against 8,960. The trade-off the table shows is therefore the one Chapter 5 predicted from counts, now with a quality column beside it: the hybrid buys, over the pure recurrent stack, the exact lookup that attention has, and pays for it in a cache that grows; and it buys, over the pure attention stack, half of the memory and half of the growth. Whether the exact lookup is worth its price depends on a task that needs it, and this one, in which the recurrent state was enough, is one on which it did not show. A task that asks the model to copy a long passage would be one on which it would, and the second extension below builds that task.

## What the comparison must report

The tests are the values the worked module prints, collected as a checklist; an implementation passes when every statement holds. "Exact" means integers, text and digests equal; "criterion" means the stated inequality; the trained numbers are not in the list because the nudge rule leaves them as bounds, but the verdict of each registered claim is.

<!-- worked: project_architecture_comparison.acceptance_fingerprint -->
| milestone | quantity | value | tolerance |
|---|---|---|---:|
| 1 | parameters of dense, moe, gdn, swa, hybrid | 4,128 / 4,256 / 4,130 / 4,128 / 4,129 | exact |
| 1 | gradient check, largest difference over every tensor of the five stacks | below 1e-6 | criterion |
| 1 | chunked against recurrent, chunk sizes 1 to 12; with the state dropped | all within tolerance; 3.490 | tolerance; 3 digits |
| 2 | `dense`: active parameters; prefill FLOPs per token (rounded down); decode FLOPs; bytes held | 4,032; 8,960; 9,728; 832 | exact |
| 2 | `moe`: active parameters; prefill FLOPs per token (rounded down); decode FLOPs; bytes held | 3,008; 6,912; 7,680; 832 | exact |
| 2 | `gdn`: active parameters; prefill FLOPs per token (rounded down); decode FLOPs; bytes held | 4,032; 8,960; 8,960; 224 | exact |
| 2 | `swa`: active parameters; prefill FLOPs per token (rounded down); decode FLOPs; bytes held | 4,032; 8,516; 8,576; 256 | exact |
| 2 | `hybrid`: active parameters; prefill FLOPs per token (rounded down); decode FLOPs; bytes held | 4,032; 8,960; 9,344; 528 | exact |
| 3 | golden digest, `dense` | 1f6196ce6dffd1fd | exact |
| 3 | golden digest, `gdn` | 1f02617b8ff996ef | exact |
<!-- /worked -->

Further tests with expected values in the tables above: the parameter counts of milestone 1 (4,128, 4,256, 4,130, 4,128, 4,129); the counted FLOPs of milestone 2 against the formula for every arm; every stack passing all three tests of the parity table and every planted mistake of the next table failing the tests it failed there, with the first tap beyond the bound where it is printed; and, for milestone 4, the verdicts H1 supported, H2 supported, H3 undecided, H4 supported and H5 undecided, with H1 and H3 reproduced to the digits printed. Test in the order of the milestones: a disagreement in milestone 4 that follows agreement in 1 to 3 points at the optimizer, the order of the batches or the learning rate, never at the blocks.

## Where each block lives in brain

brain's counterpart of the interface of milestone 1 is `crates/model/src/block.rs`: reusable step-builders for the pieces that blocks share (normalisation, rotary positions, causal grouped-query attention, the decode and cache steps, SwiGLU), with linear projections left in each model because they carry adapters and biases that differ between models. The dense block is `crates/qwen3/src/model.rs`. The window is a per-layer setting of `crates/gemma4`: `sliding_window` in `crates/gemma4/src/config.rs`, with the layer types alternating by a pattern. The mixture is `crates/model/src/moe.rs` (a router and a top-k dispatch that does the work only for the rows routed to each expert) with a trainable version, including the auxiliary balancing coefficient `aux_coef`, in `crates/toymoe/src/train.rs`, and its backward pass is checked against central differences in `crates/model/tests/moe_block_gradcheck.rs`, which runs at both $k < E$ and $k = E$ because a check at $k = E$ alone never takes the sparse branch. The gated delta layer is `crates/model/src/gdn.rs`, with `gdn_chunk_fwd` and `gdn_chunk_bwd` for the chunked form, `gdn_recurrent_step` for decode, and the mixer around them in `crates/model/src/gdn_mixer.rs`; `crates/model/tests/it/gdn_chunk_fwd.rs` is the chunked-against-oracle test of milestone 1. The hybrid pattern is `layer_types` in `crates/qwen35/src/config.rs`. The cost model is `crates/modelcost/src/lib.rs`, whose exact tier builds a real model at its own configuration and reports its forward cost, the counterpart of checking a formula against a counted run; the numerical gradient checker is `crates/gradcheck/src/lib.rs`, and `crates/model/src/conformance.rs` is a suite of contracts that every model registering with it must satisfy, a standing harness in the sense of milestone 3.

Two differences from the engine of this chapter are worth knowing. brain's gated delta layer is the full layer of Chapter 5, with the short convolution and the output gate this chapter left out, so its parameter count is not matched to its attention layer and its comparisons are made between models and not between layers. And brain runs its blocks on devices in 32-bit and lower precisions, where the tolerance of milestone 3 is the count of roundings of that precision and not the $2^{-24}$ of a stored golden.

## Summary

If every acceptance test passes you have built four blocks behind one interface with backward passes that agree with the definition of a derivative, a recurrent layer whose chunked and recurrent forms agree within a tolerance that was counted and not chosen, a cost model whose every FLOP and byte was checked against a run, a parity harness that passes correct code, catches seven planted mistakes in the layer that holds them, and has a resolution you can state, and a comparison that was registered before it was run.

What the project established, in order. A block interface is two functions and a trace, and exchanging a mixer or a network touches nothing else. A gradient is believed when two coordinates of every tensor agree with central differences, because that covers the router, the gate bias and the output gain that nobody would have thought to look at. A cost is believed when a count of the multiply-adds of a run equals the formula. A tolerance is a resolution: it catches every mistake larger than the precision of the golden and none smaller, and the cosine alone misses a mistake of scale. A comparison is fair when the parameters are matched, the data are shared, the rate is chosen per arm on a seed that is not reported, the predictions are registered, the intervals are paired and the digits are the ones that survive. And what the comparison said: at 13 tokens the blocks cost nearly the same and differ greatly in what they can learn; a window of 4 keys cannot use a header it cannot reach, as derived; the recurrent state carried the header at this budget where attention often had not found it; a mixture at matched parameters could not be told from the dense stack with six seeds; and a hybrid was as good as the best arm here at the cost of a cache.

Where this stops. The task is synthetic and short, so it favours the block that carries a fixed secret and says nothing about language. The budget is 120 steps, so the quality column is sample efficiency and not quality at convergence. The models have 4,128 parameters, the matching of the mixture is to 3.1 percent and not exact, and six seeds leave two of five predictions undecided. The gated delta layer lacks its convolution and output gate. The costs at sizes where the blocks differ in arithmetic were counted by formula and not by a run. These limits are the reason the extensions exist, and the reason the volume closes here: every later model in the book is a combination of choices that this chapter priced, tested and compared one at a time.

## Further blocks and larger budgets

These are optional. Each is a larger test of the same ideas.

1. **Multi-head latent attention.** Add the latent attention of Chapter 3 as a fifth mixer with a latent width you choose. Extend the ledger with its cache of $d_c$ numbers per token, check the counted bytes against it, plant the mistake that applies the rotary rotation to the latent before it is absorbed (Chapter 3's absorption fails with a rotated key), and add a registered claim about its loss against the dense stack at matched parameters.
2. **Sparse attention and a task that needs an exact lookup.** Add a top-$k$ selected attention (Chapter 6) as a mixer, and design a second task in which the answer is a token copied from a position that varies and has to be found by content. Register before you run which of dense, window, recurrent, sparse and hybrid should fail and why, and report the verdicts. The hybrid's attention layer should matter here.
3. **Looped depth.** Replace the two distinct layers by one layer applied twice (Chapter 7), at the same FLOPs, so that the parameters are halved. Decide in advance what the comparison with the two-layer stack should show, and note what the ledger must change to count active parameters of a looped stack.
4. **Train the dense stack longer.** Repeat the comparison at 4 times the steps for the dense and the mixture arms only, keeping the registered claims, and report at what step their median run leaves the plateau. State what you predicted before you ran it.
5. **A larger corpus of seeds.** Use the table of seeds of Chapter 9 to compute how many seeds would decide H3 and H5 at their margins from the spread you observed, then run that many, and report whether the verdicts move.
6. **A float32 candidate.** Run the candidate of milestone 3 in 32-bit arithmetic instead of with 32-bit weights in 64-bit arithmetic, count the roundings per block with the rule of Chapter 2, and compare the bound with the measured error.

## Answers

**Recall questions.**

1. A token adds keys and values: $2 \cdot 2 \cdot 4 = 16$ numbers, 32 bytes in bfloat16, to each layer's cache. A window of 4 keys holds at most $4 \cdot 32 = 128$ bytes per layer, however long the sequence.
2. The two largest probabilities are 0.5 and 0.3, experts 0 and 1. Divided by their sum 0.8 the combine weights are 0.625 and 0.375. With $f_e = \bar p_e = 1/E$ the sum $\sum_e f_e \bar p_e$ is $E \cdot (1/E)^2 = 1/E$, so the loss is $\alpha_{\mathrm{bal}} E \cdot (1/E) = \alpha_{\mathrm{bal}}$, its minimum.
3. The state is $2 \cdot 4 \cdot 7 = 56$ numbers after 10 tokens and after 10,000: it does not grow. A written value is multiplied by 0.95 at each token, and it is half after $\ln 2 / (-\ln 0.95) \approx 13.5$ tokens.
4. Max_abs is $\max |b_i|$ and rel_l2 is 1, so both see it; the cosine of $2\mathbf{b}$ and $\mathbf{b}$ is 1, so it does not. The parity tolerance is the number of roundings along the longest chain of dependent operations times the unit round-off.
5. $2 N_{\mathrm{act}} + 4 L H d_h T = 8{,}064 + 4 \cdot 2 \cdot 4 \cdot 4 \cdot 13 = 8{,}064 + 1{,}664 = 9{,}728$ FLOPs, the decode figure of the dense stack in milestone 2.
6. The standard error is $0.10/\sqrt{6} = 0.0408$, the half-width $2.5706 \cdot 0.0408 = 0.105$, and the interval $-0.305$ to $-0.095$. The claim "at most $-0.1$" is supported only if the whole interval is below $-0.1$; its upper end $-0.095$ is above it, and the lower end is below it, so the claim is undecided.
