<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# The KV Cache

*Why recompute the past at every step when we could remember it, and what does remembering cost?*

Every loop of the last chapter begins the same way. Give the model the text so far, take the distribution at the last position, choose a token, put it on the end, and go round again. "Give the model the text so far" hides a decision. At step 100 the text has 100 tokens, and 99 of them went through every layer a moment ago, one step earlier, to produce the previous token. If the model is run the way it was trained, as one forward pass over the whole text, then all of that work is done a second time, and at step 101 a third time. Nothing is wrong with the answer. What is open is how much of the arithmetic was needed.

The answer of this chapter is: almost none of it, and the reason is one fact about the mask of Chapter 7 that can be proved in a paragraph. The fact lets a model remember a small part of its past, two vectors per token and layer, and never compute that part again. Remembering has a bill of its own, in bytes instead of FLOPs, and the bill decides how long a context a device can hold and how many people it can talk to at once. Both halves, the saving and the bill, have to be counted before either is believed.

To follow the count, six ideas from earlier chapters need to be in hand. Try them first; the answers are at the end of the chapter.

1. (Volume V, [Chapter 7](07-self-attention.md), queries, keys and values.) In one attention head, position $i$ forms a query, a key and a value. Which of the three does the output at position $i$ use from the other positions $j$, and which does it use from position $i$ itself?
2. (Volume V, [Chapter 7](07-self-attention.md), the causal mask.) A text has 5 tokens. How many of the 25 pairs (query position, key position) does the causal mask allow, counting a position with itself?
3. (Volume V, [Chapter 8](08-positions.md), rotary embeddings.) The score of a query at position $i$ and a key at position $j$ under rotary embeddings depends on the two positions in what way, and what does the rotation applied to the key depend on?
4. (Volume V, [Chapter 11](11-decoding.md), greedy decoding.) The logits of four tokens, numbered from 0, are $(1.2, 3.4, 3.4, 0.1)$. Which token does greedy decoding choose?
5. (Volume I, [Chapter 16](../v01-mathematics/16-algorithms.md), memory complexity and the time-memory tradeoff.) A loop computes the same $n$-element result $n$ times over. What does storing the result once cost in memory and save in time, in the chapter's terms?
6. (Volume IV, [Chapter 3](../v04-computation/03-roofline.md), the roofline.) A device has a peak arithmetic rate of $4 \times 10^{13}$ FLOP per second and a memory bandwidth of $8 \times 10^{11}$ bytes per second. Is a step that does 2 FLOPs per byte moved limited by the arithmetic or by the bytes, and at what intensity does the answer change?

## Eighteen rows for four tokens

The model is the small GPT of Chapter 10 with a smaller vocabulary, and the count is easiest to see on it. To keep the count honest, every product the model runs is added to a tally as it runs.

<!-- worked: kv_cache.tiny_model -->
Generator: the book's SplitMix64, seed 0xe354ebf228c667c0 (`seed_of` of the name `kv_cache.model`). The model is the GPT of Chapter 10 with V = 12 token ids, D = 16, 2 heads of width 8, L = 2 pre-norm blocks with a SwiGLU network of hidden width F = 24, at most T = 24 positions, learned absolute positions, RMS norms, a head tied to the embedding table and no biases. Every weight is 1.0 x `normal()`, drawn row by row in the order E [V, D], P [T, D], then for each block W_Q, W_K, W_V, W_O (D by D), W_gate, W_up (D by F), W_down (F by D), with W_O and W_down multiplied by 1/sqrt(2 L); all gains are 1. That is 5,008 parameters. The weights are random: nothing here is trained, and the prompt is the three token ids [5, 2, 9]. Greedy decoding takes the largest logit, the lowest id on a tie.
<!-- /worked -->

The standard deviation is 1.0, fifty times the 0.02 of Chapter 10's initialization, on purpose: weights that small leave an untrained model's logits almost equal at every position, and a test of a cache on logits that hardly depend on the text would prove little. Positions are numbered from 0 in this chapter, as they are in the pseudocode and in brain's cache rows (Chapter 7 counted from 1), and layers from 0 as well. The naive loop runs the whole text through the model at every step:

<!-- worked: kv_cache.naive_generation -->
The naive loop: at every step the whole text so far is run through the model from scratch (`extend` on an empty cache), and the logits of its last position give the next token. Prompt [5, 2, 9], 4 new tokens. FLOPs are counted by the model run: a multiply-add is two FLOPs; the matrix products of every layer and the head (for the last position only) are counted, the norms, the softmax and the SwiGLU gate are not (the rule of Chapter 10).

| step | rows computed | text fed in | next token | FLOPs |
|---|---:|---:|---:|---:|
| 1 | 3 | 5 2 9 | 4 | 27,264 |
| 2 | 4 | 5 2 9 4 | 6 | 36,480 |
| 3 | 5 | 5 2 9 4 6 | 6 | 45,824 |
| 4 | 6 | 5 2 9 4 6 6 | 4 | 55,296 |

In all 18 rows are computed and 164,864 FLOPs are spent, to produce 4 tokens. The logits of the last row agree with Chapter 10's masked forward pass (`training_gpt.forward`, no cache) to a largest difference of 0 (identical) at every step.
<!-- /worked -->

Four tokens cost 18 rows, and the text they end in has 6 positions. The rows of step 4 are the rows of step 3 and one more; the rows of step 3 are those of step 2 and one more. Writing the prompt as $p$ tokens and the number of generated tokens as $N$, step $t$ (counting from 0) runs $p + t$ rows, so the loop computes

$$
\sum_{t=0}^{N-1} (p + t) = pN + \frac{N(N-1)}{2}
$$

rows in all, which is 18 for $p = 3$ and $N = 4$. That is quadratic in $N$: once the reply is much longer than the prompt, ten times as many generated tokens computes about a hundred times as many rows. Each row costs the same in the matrices of the blocks (the weights of Chapter 9 are applied to every row of the text, whatever its position), and a row at position $i$ costs more in the attention products, because it looks at $i + 1$ keys. So the total is quadratic in the weights' arithmetic and cubic in the attention's, since the number of rows is quadratic and each attends to a number of keys that grows with the text. A model that must answer a prompt of a thousand tokens with a thousand more would spend most of its life repeating itself.

The question is whether the repetition is real. A row of step 3 at position 1 is the same token at the same position as at step 2. If the model is a function of the text it is given, the row may still come out differently, because the new token at the end could change what position 1 computes. The mask of Chapter 7 is designed to forbid that, and the next section turns "designed to" into a statement that can be tested.

## What the mask promises about the past

Take one layer. The key and the value of position $j$ are linear maps of the normalized stream at position $j$ alone: $\mathbf{k}_j = \mathbf{n}_j\mathbf{W}_K$ and $\mathbf{v}_j = \mathbf{n}_j\mathbf{W}_V$, where $\mathbf{n}_j$ is row $j$ of the normalized input of the layer and the normalization acts on each row on its own (Chapter 9). Whether they change when tokens are appended depends on whether $\mathbf{n}_j$ does, and $\mathbf{n}_j$ is the output of the layers below. So the question is about the whole stack, and the answer is an induction on the layers. (With rotary embeddings the key is also rotated by an angle that depends on its own position $j$ alone, so the argument is unchanged.)

The statement to prove is this. For every layer $l$ and every position $j$, the stream $\mathbf{x}^{(l)}_j$ that enters layer $l$ at position $j$, and therefore the key $\mathbf{k}^{(l)}_j$ and the value $\mathbf{v}^{(l)}_j$ of that layer, are functions of the tokens at positions $0, \dots, j$ only.

*Proof.* For $l = 0$ the stream at $j$ is the embedding of token $j$ plus the position vector of $j$ (Chapter 10), which depends on token $j$ alone. Suppose the statement holds for the streams entering layer $l$. The key and value at $j$ are rows of products of $\mathbf{x}^{(l)}_j$, so they depend on tokens $0, \dots, j$. The attention output at $j$ is $\sum_{i \le j} p_{ji}\mathbf{v}_i$ with weights from the scores $\mathbf{q}_j \cdot \mathbf{k}_i$, $i \le j$: the mask has set every other weight to exactly 0. Each term uses only the query at $j$, which comes from $\mathbf{x}^{(l)}_j$, and keys and values at positions $i \le j$, which depend on tokens $0, \dots, i \subseteq 0, \dots, j$. So the attention output at $j$ depends on tokens $0, \dots, j$. The residual add, the second normalization and the feed-forward network act on each position on its own, so $\mathbf{x}^{(l+1)}_j$ depends on tokens $0, \dots, j$ and the statement holds for the next layer. $\blacksquare$

The consequence is the one the section was after. When a token is appended at position $c$, the tokens at $0, \dots, c - 1$ have not changed, so none of the keys and values at those positions, in any layer, changes. They never need to be computed again. The query of position $j$ is not on the list: it is used once, to produce the output at $j$, and after that nothing reads it. The logits of the earlier positions are not needed either, because the next token is chosen from the last position. What the future needs from the past is exactly the keys and values, and that set is called the **KV cache**: for each layer, one key row and one value row for every token seen so far, kept so that later tokens attend to them without recomputing them.

The statement can be tested on the model instead of trusted. Run a text of six tokens twice, once as its first three tokens alone and once whole, and compare the keys and values of the first three positions, layer by layer. For contrast, run it also with no mask at all.

<!-- worked: kv_cache.mask_invariance -->
The text of six tokens [5, 2, 9, 4, 6, 6] (the prompt and the first three tokens the naive loop chose), run twice: once as its first three tokens alone and once whole. For each layer l and each of the first three positions, the largest difference over the key and value entries (all heads) between the two runs, for the causal mask and for no mask at all.

| mask | layer | position | largest difference of key and value |
|---|---:|---:|---:|
| causal, j <= i | 0 | 0 | 0 (identical) |
|  | 0 | 1 | 0 (identical) |
|  | 0 | 2 | 0 (identical) |
|  | 1 | 0 | 0 (identical) |
|  | 1 | 1 | 0 (identical) |
|  | 1 | 2 | 0 (identical) |
| no mask, every j | 0 | 0 | 0 (identical) |
|  | 0 | 1 | 0 (identical) |
|  | 0 | 2 | 0 (identical) |
|  | 1 | 0 | 10.8 |
|  | 1 | 1 | 11.6 |
|  | 1 | 2 | 0.00411 |
<!-- /worked -->

Under the causal mask every difference is exactly 0: not small, zero, because the arithmetic that produces those rows is the same sequence of operations on the same numbers. Without the mask, the keys and values of layer 1 at positions 0 and 1 change by 10.8 and 11.6 when the three later tokens arrive, and the layer-1 key and value at position 2 change by 0.00411 (this position attends strongly to itself in this model, so the others move it little). Layer 0 shows no difference even without the mask, and it should not: the key of layer 0 is computed from the embedding of its own token before any attention has run. The mask matters from layer 1 on, where each stream has already mixed in the others. That is a reminder of what the proof needed from the mask, and of why an engine with a leaky mask would pass any test that looked only at layer 0.

![Two grids of positions against steps for a prompt of three tokens and four generated ones. Left, recomputing the whole text: step 1 has three dark cells, step 2 four, step 3 five and step 4 six, 18 in all. Right, with a cache: step 1 has three dark cells and every later step one dark cell, the new token, with the cells to its left light and marked cache, 6 dark cells in all.](img/cache-grid.svg)

## One routine for the prompt and for every token after it

With the statement in hand the loop can be rewritten so that no row is computed twice. The only new object is the cache, an array with a row of keys and a row of values for each layer and position, and a count of how many positions are filled. One routine, `extend`, runs $n$ new tokens through the model on top of a cache that holds $c$ earlier ones. The new tokens sit at positions $c, c + 1, \dots$; their keys and values are written into the cache at those rows; and new token $i$ attends to cache rows $0, \dots, c + i$, which is the causal mask without a mask matrix, since later rows do not exist yet.

```text
record KvCache:
    K: array<real>[L, T, D]      # K[l, j, :] is the key of position j in layer l, all heads side by side
    V: array<real>[L, T, D]
    length: int                  # positions 0 .. length - 1 are filled

function extend(model, cache: inout KvCache, ids: list<int>) -> array<real>[V]
    # E, P, Wq[l], ..., gf and the sizes L, T, D, H, d_k, V, eps are the weights and sizes of `model`
    n = len(ids)
    c = cache.length
    require n >= 1 and c + n <= T
    X = zeros(real, [n, D])
    for i in 0 .. n:
        X[i, :] = E[ids[i], :] + P[c + i, :]           # the position of new token i is c + i
    for l in 0 .. L:
        (A, _, _) = rms_norm(X, g1[l], eps)            # the block of Chapter 9, pre-norm, on the rows of X
        Q = matmul(A, Wq[l])
        cache.K[l, c:c + n, :] = matmul(A, Wk[l])
        cache.V[l, c:c + n, :] = matmul(A, Wv[l])
        O = zeros(real, [n, D])
        for i in 0 .. n:
            for h in 0 .. H:
                lo = h * d_k
                S = [dot(Q[i, lo:lo + d_k], cache.K[l, j, lo:lo + d_k]) / sqrt(d_k) for j in 0 .. c + i + 1]
                p = softmax(S)
                for j in 0 .. c + i + 1:
                    O[i, lo:lo + d_k] = O[i, lo:lo + d_k] + p[j] * cache.V[l, j, lo:lo + d_k]
        X = X + matmul(O, Wo[l])
        (B, _, _) = rms_norm(X, g2[l], eps)
        Hd = map(silu, matmul(B, Wgate[l])) * matmul(B, Wup[l])
        X = X + matmul(Hd, Wdown[l])
    cache.length = c + n
    (z, _, _) = rms_norm(X[n - 1:n, :], gf, eps)
    return matmul(z, transpose(E))[0, :]               # logits of the last new token

function generate(model, prompt: list<int>, n_new: int) -> list<int>
    require n_new >= 1
    cache = KvCache(zeros(real, [L, T, D]), zeros(real, [L, T, D]), 0)
    z = extend(model, cache, prompt)                   # the prompt in one call
    out = []
    for step in 0 .. n_new:
        t = argmax(z)
        append t to out
        if step < n_new - 1:
            z = extend(model, cache, [t])              # one token, at position cache.length
    return out
```

The naive loop of the last section is the same routine called on a fresh cache with the whole text at every step. The call with $n$ equal to the prompt length and an empty cache is the **prefill phase**: the whole prompt goes through every matrix at once, so the matrices see $n$ rows (matrix products of two matrices), the keys and values of every prompt position are written into the cache in one go, and the logits of the last position give the first generated token. Every call after it, with $n = 1$ and a cache of $c$ rows, is the **decode phase**: one token goes through the model, each matrix sees one row (a matrix-vector product), one key row and one value row per layer are appended, and the attention is a single query against all $c + 1$ cached rows. The two phases are the same arithmetic on different shapes, and the rest of this chapter is largely about why that difference of shape matters so much.

<!-- worked: kv_cache.cached_generation -->
The same model, prompt and 4 tokens with a cache: step 1 runs the prompt through the model once (the prefill phase) and stores the keys and values of its 3 tokens in every layer; each later step feeds only the token chosen before, at position equal to the number of tokens already in the cache, and appends one key and one value row per layer (the decode phase). 'Largest difference' is over the 12 logits against the naive loop's logits at the same step.

| step | phase | rows computed | cache length after | next token | FLOPs | largest difference from naive logits |
|---|---:|---:|---:|---:|---:|---:|
| 1 | prefill | 3 | 3 | 4 | 27,264 | 0 |
| 2 | decode | 1 | 4 | 6 | 9,600 | 0 |
| 3 | decode | 1 | 5 | 6 | 9,728 | 0 |
| 4 | decode | 1 | 6 | 4 | 9,856 | 0 |

The tokens are [4, 6, 6, 4], the same as the naive loop's [4, 6, 6, 4]. 6 rows are computed instead of 18 and 56,448 FLOPs are spent instead of 164,864: 2.92 times fewer. Each cache holds 2 layers of 6 rows of key and of value (the last token chosen is never fed back).
<!-- /worked -->

The tokens are the naive loop's, and so are the logits, to the last bit: the largest difference is 0 at every step. That is no accident of this model; it follows from using the same sums in the same order, and the section on equivalence says what happens when the order changes. The cached run computes 6 rows where the naive run computed 18, and spends 56,448 FLOPs where it spent 164,864. Step 1 is the same in both, since both run the whole prompt. Steps 2 to 4 cost 9,600, 9,728 and 9,856 FLOPs: each is the work of one row through the weights, 8,704 FLOPs, the head at 384, and an attention term of 128 FLOPs for every key the new token reads, which is why the three differ by 128 each.

The cache has a layout, and brain's choice is the plain one. For each layer there are two arrays of shape [capacity, $h_{\mathrm{kv}} d_h$], where $h_{\mathrm{kv}}$ is the number of key-value heads (here equal to the number of heads, $H$) and $d_h$ the head width, with one row per position: row $j$ is the key (or value) of position $j$ with its heads side by side. `crates/model/src/block.rs` documents the contract of its decode step, `gqa_decode_step` (the `gqa` is grouped-query attention, in which several query heads share one key-value head; a later section uses it, and for the model of this chapter the two numbers of heads are equal), in these words: it "appends the new token's (already QK-normed + RoPE'd) `k_new`/`v_new` into the persistent per-layer `kcache`/`vcache` at row `pos`, then attends `q` against all `pos+1` cached positions", and, since it only ever reads rows `0..=pos`, it is "implicitly causal". It is the O(cached length) twin of the O(T squared) attention of training, and its introduction in commit `4d61547a3` (August 2026) says so in its title, "KV-cache decode -- O(cached length), not O(cached length)^2". The prefill side is a bulk copy: the keys and values that a batched forward pass produced for the whole prompt have the same row layout as the cache, so `kv_cache_fill` writes them into rows `0..n` of the cache with a single flat copy.

## What the saving is worth in FLOPs

Count a step's work with the rule of Chapter 10: a multiply-add is two FLOPs, only matrix products count, and the head is applied to the last position only. For a model with $N_{\mathrm{blk}}$ parameters in the matrices of its blocks, $L$ layers, $H$ query heads of width $d_h$ and a head of $V D$ parameters, running $n$ new tokens on a cache of $c$ costs

$$
F(c, n) = 2\,n\,N_{\mathrm{blk}} + 4\,L\,H\,d_h \Bigl(n c + \frac{n(n+1)}{2}\Bigr) + 2\,D\,V.
$$

The first term is every weight matrix applied once per new token. The second is the attention products: the scores of one query against $m$ keys cost $2 H d_h m$ per layer, and the weighted sum of the values costs the same again, and new token $i$ sees $c + i + 1$ keys, which sums to $nc + n(n+1)/2$ over the $n$ new tokens. The third is the head once. The formula is not trusted: the run that produced the tables above also counted its own multiply-adds, and the two agree at every step, for the tiny model and for the 12-token run of the next sections.

The naive loop for a prompt of $p$ tokens and $N$ new ones costs $\sum_{t=0}^{N-1} F(0, p + t)$ and the cached loop costs $F(0, p) + \sum_{t=0}^{N-2} F(p + t, 1)$. For GPT-2 small they come out as follows.

<!-- worked: kv_cache.flops_growth -->
GPT-2 small (V = 50,257, D = 768, L = 12, 12 heads of width 64, F = 3,072, a network of two matrices) with a prompt of 64 tokens, FLOPs of the matrix products by the formula of the text: 2 x 84,934,656 per token for the weights of the blocks, 4 L H d_h = 36,864 per token per visible key, and the head (2 D V = 77,194,752) once per step. The formulas are asserted equal to the counted runs of the tiny model for 4 and for 12 new tokens.

| new tokens N | naive FLOPs | cached FLOPs | naive / cached |
|---|---:|---:|---:|
| 4 | 45,135,771,648 | 11,773,999,104 | 3.8 |
| 16 | 197,100,625,920 | 14,771,281,920 | 13.3 |
| 64 | 1,054,455,595,008 | 26,813,497,344 | 39.3 |
| 256 | 8,547,149,414,400 | 75,831,705,600 | 112.7 |
| 512 | 28,999,948,173,312 | 143,303,245,824 | 202.4 |
<!-- /worked -->

At four tokens the cache is worth a factor of 3.8 (2.92 for the tiny model); at 512 tokens it is worth 202. The ratio keeps growing with the number of tokens generated, because the naive loop's weight work grows with the square of $N$ and the cached loop's with $N$ itself. It grows more slowly than $N$, since the cached loop still pays for the head at every step and for the attention over a growing cache. The cached total is still not linear: its attention term, $4LHd_h$ per key read, adds up to a quadratic in the length of the text. The cache removes the quadratic that came from recomputing the matrices and leaves the one that is intrinsic, a new token reading every earlier token. At a context of 1,024 the weights still cost far more per decoded token than the attention does, 2 x 84,934,656 FLOPs against 4 x 12 x 12 x 64 x 1,024 = 37,748,736; at a context of 32,768, which GPT-2 could not use but a larger model can, the attention is the larger term. FLOPs are not the whole account, though, and the other half of it is the reason the cache is the central object of inference engineering.

## Why one token reads the whole model

Volume IV taught a way of asking whether a computation is limited by arithmetic or by bytes: divide the FLOPs by the bytes that must cross from memory, and compare with the ridge of the device. In its section on decoding, Chapter 3 counted a token as $2P$ FLOPs against $P b$ bytes of weights and put the memory of earlier tokens aside with the remark that an exercise would add it. This is that exercise. A decode step does the arithmetic of one row, but it needs every weight, because every weight multiplies the row; and it needs the whole cache of every layer, because the new query attends to all of it. So the bytes of a decode step are the weights once plus the cache of the sequence, and the FLOPs are $F(c, 1)$.

A prefill step does the same reading and a great deal more arithmetic: all $n$ prompt rows meet the same weights, read once, so the FLOPs are $F(0, n)$ for the bytes of the weights and the $n$ new cache rows that are written.

The numbers need a model large enough for the weights to matter. The one used here is a decoder of about 7.5 billion parameters in its matrix products: $D = 4096$, 32 layers, 32 query heads and 8 key-value heads of width 128 (grouped-query attention, which a later volume derives; here it only means that the cache holds $h_{\mathrm{kv}} = 8$ heads' worth of keys and values), a feed-forward width of 14,336 (SwiGLU, three matrices) and a vocabulary of 128,256, an output head of its own ($D$ by $V$, not tied to the embedding table, so its $VD$ parameters are counted once as a product), with every value stored in bf16. The device is the illustrative GPU of Volume IV.

<!-- worked: kv_cache.decode_and_prefill -->
A decoder with D = 4096, L = 32, 32 query heads and 8 key-value heads of width 128, F = 14,336 (SwiGLU), V = 128,256, values stored in bf16 (2 bytes). Its 7,504,658,432 parameters in matrix products are 15,009,316,864 bytes. FLOPs: the formula of the text, 13,958,643,712 per token for the blocks, 4 L H d_h = 524,288 per visible key, and the head once. Bytes: the weights once (the embedding lookup reads one row and is left out) plus the cache rows read (decode) or written (prefill) at 131,072 bytes per token; activations are not counted. A decode step at context c counts the c stored rows as bytes read, while its attention FLOPs count c + 1 keys: the new token attends to its own key and value too, which it has just computed, so they are not read from the cache and their write is left out of the bytes. The decoder's head is a matrix of its own (D by V), not tied to the embedding table, and counted once per step. The device is the illustrative GPU of Volume IV (F_pk = 40,000 (GFLOP/s), BW = 800 (GB/s)), whose ridge is 50 FLOP per byte.

| phase | size | FLOPs | weight bytes | cache bytes | FLOP per byte | limited by |
|---|---|---:|---:|---:|---:|---|
| decode, one token | context 1,024 | 15,546,712,064 | 15,009,316,864 | 134,217,728 | 1.03 | bandwidth |
| decode, one token | context 8,192 | 19,304,808,448 | 15,009,316,864 | 1,073,741,824 | 1.20 | bandwidth |
| decode, one token | context 32,768 | 32,189,710,336 | 15,009,316,864 | 4,294,967,296 | 1.67 | bandwidth |
| decode, one token | context 131,072 | 83,729,317,888 | 15,009,316,864 | 17,179,869,184 | 2.60 | bandwidth |
| prefill | 1 token | 15,009,841,152 | 15,009,316,864 | 131,072 | 1.00 | bandwidth |
| prefill | 16 tokens | 224,460,275,712 | 15,009,316,864 | 2,097,152 | 14.95 | bandwidth |
| prefill | 128 tokens | 1,792,085,590,016 | 15,009,316,864 | 16,777,216 | 119.26 | compute |
| prefill | 1,024 tokens | 14,569,848,176,640 | 15,009,316,864 | 134,217,728 | 962.12 | compute |
| prefill | 8,192 tokens | 131,944,593,489,920 | 15,009,316,864 | 1,073,741,824 | 8203.95 | compute |

Decoding at context 4,096 for several sequences at once: the weights are read once for the step and every sequence reads its own cache.

| sequences | FLOPs | weight bytes | cache bytes | FLOP per byte | fraction of peak attainable |
|---|---:|---:|---:|---:|---:|
| 1 | 17,157,324,800 | 15,009,316,864 | 536,870,912 | 1.10 | 0.022 |
| 8 | 137,258,598,400 | 15,009,316,864 | 4,294,967,296 | 7.11 | 0.142 |
| 64 | 1,098,068,787,200 | 15,009,316,864 | 34,359,738,368 | 22.24 | 0.445 |
| 256 | 4,392,275,148,800 | 15,009,316,864 | 137,438,953,472 | 28.81 | 0.576 |

The shortest prompt whose prefill reaches the ridge (50 FLOP per byte) is 54 tokens. With ever more sequences at context 4,096 the weights are shared by all of them and the intensity tends to the FLOPs of one sequence's step over its own cache bytes, 17,157,324,800 / 536,870,912 = 31.96 FLOP per byte.
<!-- /worked -->

Read the decode rows first. At a context of 1,024 tokens a decode step does 15,546,712,064 FLOPs and moves 15,009,316,864 bytes of weights and 134,217,728 of cache: 1.03 FLOPs for every byte, against a ridge of 50. The device could do 50 FLOPs in the time it takes to deliver one byte and the step gives it one. Every row of the decode part of the table is limited by the bytes, and lengthening the context hardly changes that: the cache adds to the arithmetic and to the bytes in nearly the same proportion, so the intensity rises from 1.03 only to 2.60 at 131,072 tokens. The roofline then bounds the time of a decode step from below by the bytes divided by the bandwidth, and the bytes grow with the context.

![Gigabytes read to decode one token against the tokens already in the cache, for the decoder of the chapter. The weights alone are a horizontal dashed line at 15.0 GB. With 8 key-value heads the total rises from 15.0 GB to 32.2 GB at 131,072 tokens and crosses twice the weights at 114,512 tokens; with 32 key-value heads it starts at 16.6 GB, reaches 85.3 GB at 131,072 tokens and crosses twice the weights at 31,700 tokens.](img/bytes-per-decoded-token.svg)

The figure shows what grows. The weights are a constant bill for every token; the cache is a bill proportional to the text. With 8 key-value heads the cache read equals the weights read at 114,512 tokens, and with one key-value head per query head, where each token's cache is four times larger, at 31,700. The second figure uses the weights of its own variant: widening $\mathbf{W}_K$ and $\mathbf{W}_V$ from 1,024 to 4,096 columns in each of the 32 layers adds 805,306,368 parameters, so its weights are 8,309,964,800 parameters, 16,619,929,600 bytes, and the crossing is 16,619,929,600 / 524,288 = 31,700 tokens. With the 7.5-billion decoder's weights of 15,009,316,864 bytes the same arithmetic would give 28,628, which is the crossing of a model with four times the cache per token and no more weights. Past that point a decoded token is mostly a read of the past. Pope and colleagues (2023) count the same two memories, the weights and the cache, to see how large a batch and how long a context a given set of devices can serve.

The prefill rows tell the other story. A prompt of one token is a decode step and has intensity 1.00. At 16 tokens the intensity is 14.95, still below the ridge; at 128 it is 119.26, above it. The shortest prompt whose prefill reaches the ridge is 54 tokens. The reason is the one Volume IV gave for batching: the weights are read once and used for $n$ rows, so the intensity grows in proportion to $n$, and a long prompt turns the same bytes into far more arithmetic. Prefill is limited by the arithmetic and decode by the bytes, and since a generation is one prefill and then hundreds of decode steps, the long middle of a generation is the part the cache made necessary and the roofline makes expensive.

![The roofline of the illustrative GPU on logarithmic axes, a sloped line that becomes flat at 50 FLOP per byte. Decode steps at context 4,096 for 1, 8 and 64 sequences lie on the slope at intensities 1.10, 7.11 and 22.24; prefill passes of 16 tokens lie on the slope at 14.95, and of 128, 1,024 and 8,192 tokens lie on the flat part at 119.26, 962.12 and 8,203.95.](img/prefill-decode-roofline.svg)

Batching the decode steps of several sequences is the lever, and the lower table of the block counts how far it goes. The weights are read once for the step however many sequences share it, and every sequence reads its own cache. At context 4,096 the intensity is 1.10 for one sequence, 7.11 for eight and 22.24 for 64. It does not reach the ridge for any batch, because every added sequence brings its own cache read: with ever more sequences at that context the intensity tends to the FLOPs of one step over the cache bytes of one sequence, 31.96, below 50. On this device and for this model, at this context, batching can lift a decode step to at most 31.96 / 50, 0.64 of the attainable arithmetic, and no further. That limit is a statement about the cache, not about the weights, and it is the first sign of the bill the chapter's second half describes.

## The same logits, and what "the same" means

Everything so far assumed that the cached run produces what the naive run produces. The property has a name, and it is the one any implementation of the cache must be held to. A cache is correct when, for every text and every point at which the text is split into a prefill and a sequence of decode steps, the logits of the cached run equal those of one forward pass over the whole text. This is **cache equivalence**, and a test of it needs a tolerance, because "equal" can mean two things.

In exact arithmetic the equality is an identity, by the statement proved in the section on the mask. On a machine it is not automatically exact, because floating-point addition is not associative and the two runs do not have to add in the same order. The prefill multiplies matrices with tiles whose partial sums are accumulated in a fixed pattern. The decode step multiplies a matrix by a single vector, usually with a different kernel: its dot products are split among threads, its sums over cached keys are cut into blocks and combined, and, in brain's fused decode kernels, the number of blocks depends on the context and on how many sequences share the step. Volume IV, Chapter 7 showed that a tree and a loop give different sums of the same numbers. Here is that observation on 64 numbers, added in five orders in 32-bit arithmetic.

<!-- worked: kv_cache.summation_order -->
Generator: the book's SplitMix64, seed 0x421102b8518fb6cf (`seed_of` of the name `kv_cache.sums`). Draws: 64 values `normal()`, each rounded to float32. They are added in float32, every addition rounded to float32, in five orders: left to right; right to left; the even-numbered entries and the odd-numbered entries apart, then the two sums added; eight blocks of eight consecutive entries, each summed left to right, the block sums added left to right; and a pairwise tree (entries 0 and 1, 2 and 3, ... then the pair sums in pairs). The exact sum of the 64 float32 values is 4.44071633 (`math.fsum`).

| order | float32 sum | sum minus exact |
|---|---:|---:|
| left to right | 4.44071579 | -5.44e-07 |
| right to left | 4.44071484 | -1.50e-06 |
| two lanes | 4.44071531 | -1.02e-06 |
| eight blocks of eight | 4.44071674 | 4.10e-07 |
| pairwise tree | 4.44071722 | 8.87e-07 |

The five sums take 5 different values; the largest gap between two of them is 2.38e-06, about 5.4e-07 of the sum. The largest error against the exact sum is 1.50e-06, which is 3.4e-07 of the sum (relative error), 5.7 unit round-offs of 2^-24.
<!-- /worked -->

Five orders give five different sums, 4.44071484 to 4.44071722, a spread of 2.38e-06 which is about 5.4e-07 of the sum: a few units in the seventh digit, which is the round-off of the 32-bit format (24 significant bits, a unit round-off of $2^{-24} = 6 \times 10^{-8}$). None of the five is wrong. Each is within six unit round-offs of the exact sum, 4.44071633, in relative error: the worst, right to left, is off by 1.50e-06, which is 3.4e-07 of the sum and 5.7 unit round-offs, and which one is closest is luck.

So the cached run, with a decode kernel that adds in another order, should agree with the naive run to rounding and not exactly. The model of the chapter shows both cases. In the first the cached run uses the naive loop's order, and in the second the decode steps compute each dot product of query and key as two partial sums and each weighted sum of the values in blocks of four keys, both in the order a parallel kernel might use.

<!-- worked: kv_cache.cache_equivalence -->
The same model and prompt as before, 12 greedy tokens. 'Largest difference' is over all 12 steps and all 12 logits of a step against the naive loop's logits at the same step. In the third row the prefill keeps the serial order but every decode step computes each dot product of a query and a key as two partial sums (even and odd entries, then added) and each weighted sum of values as blocks of 4 keys summed apart and then added in order: the same mathematics in the order a parallel kernel might use. Tolerance of the check: 1e-12 on every logit (the unit round-off of double precision is 1.1e-16).

| generation | largest difference of logits | same tokens | tokens |
|---|---:|---:|---|
| naive loop (recompute) | - | - | 4 6 6 4 8 0 6 2 6 2 4 3 |
| cached, summation order of the naive loop | 0 (identical) | yes | 4 6 6 4 8 0 6 2 6 2 4 3 |
| cached, decode steps summed in lanes of 2 and blocks of 4 | below 1e-13 | yes | 4 6 6 4 8 0 6 2 6 2 4 3 |

The largest absolute logit along the naive run is 14.83, and the smallest gap between the best and the second-best logit of a step is 0.0853.
<!-- /worked -->

With the same order the logits agree exactly. With the other order the largest difference over 12 steps and 12 logits is below 1e-13, which is under a thousand times the unit round-off of double precision, in a model whose logits reach about 15, and the generated tokens are the same. The tolerance this chapter holds the second row to is $10^{-12}$ on every logit, which is a bound with room above the measurement and far below the gap between the best and second-best logit (the smallest gap between the best and second-best logit along this run is 0.0853, so 1e-12 cannot change a greedy choice). The right tolerance is a decision with two sides: loose enough to admit the rounding of a legitimate reordering, tight enough that a real error (a wrong position, a stale row, a mask that leaks one token) shows far above it. The bugs of the next section move the logits by 15 to 19, more than thirteen orders of magnitude above this tolerance.

Brain's first test of the property shows the discipline. When commit `4d61547a3` (August 2026) hoisted the decode attention into `crates/model/src/block.rs`, it added to that file the test `decode_step_matches_causal_batched_attention_at_every_position`: with 5 positions, 4 query heads, 2 key-value heads of width 8, fixed sine-formula inputs and no random numbers, it runs the full masked attention of training once and then calls `gqa_decode_step` once per position, and compares every output row of the second with the same row of the first. The bound is $10^{-4}$ on every element, in 32-bit arithmetic, three orders of magnitude above the unit round-off of that format. A second test of the same commit, `bulk_fill_matches_per_row_append`, checks that filling the cache by one flat copy gives exactly the cache that appending row by row would, and it compares them with equality. A tolerance is for the arithmetic that is allowed to differ; a copy is not arithmetic, so it gets none.

How exact the equality has to be depends on what is sitting on top of the logits. A speculative decoder (a later volume) verifies a guess by running several tokens through the model in one pass and accepting those that plain decoding would have produced; if the pass that verifies is not bit-for-bit the pass that decodes, near ties between two tokens flip. The test `crates/qwen35/tests/it/chunk_tape_decode_parity.rs` states the reason in its own header for an int8 model: the weights and the activations are quantized, "so any difference between two tapes in a late layer moves a rounding decision", and every later layer amplifies it to the size of the quantization noise itself. Commit `26f792ea4` (October 2026) accordingly made a short chunk round, up to 32 rows, run through the decode tape's own kernels, and the test compares the hidden state after rounds of 1, 2, 3, 8, 9 and 32 rows with the decode tape's by bits, and the next two decode steps too, "which cannot agree unless the round left the KV cache, the Gated-DeltaNet state and the conv window exactly as token-by-token decode would have". The commit message gives the symptom that motivated it: speculative decoding diverging from plain decoding at token 7 on the int8 stack. The code keeps the other side available. `set_exact_rounds(false)` in `crates/qwen35/src/model.rs` withdraws the exact form so that the chunk tape can still be measured, and its documentation observes that "the distance between two correct renderings of the same model" is the floor against which a numerical change elsewhere is to be judged.

The floor is not small. Knowledge #328 records that on the 27B model of that stack, with quantized weights and activations, the same cache scored through its two dispatch shapes, the decode tape and the chunk tape, differs in logits by a relative L2 distance of the order of 0.1, and that a kernel change which moved the logits by that much was not a defect: it was the model's own numerics. The test `crates/qwen35/tests/it/gguf_kv_tier_real.rs` is built on that. It measures the floor in the same run and accepts a compact cache format only if its logits are within a small multiple of the floor (1.5 in the test), and not within an absolute bound that no honest change could meet. The lesson for a tolerance follows from the model's arithmetic. When a pipeline rounds its activations to integers, a reordering of sums is amplified to the size of the rounding noise, so "agrees to a float32 reordering" is not available as a standard, and the choice is between bit-for-bit agreement, which has to be engineered, as in the last paragraph, and agreement to a floor measured on the model itself.

A last form of the same effect is the co-batch. Commit `678a77956` (October 2026) and knowledge #358 report that a served model gave two distinct token sequences for identical greedy requests when six or eight of them ran together, and one sequence when they ran one after the other, with the first difference at token 7. One of the three causes it names belongs to this chapter: the fused decode attention sizes the number of key blocks it splits a row's keys into from the number of sequences in the batch and from the longest of them. The grouping of one sequence's keys, and so the summation order of its attention, depended on who else was in the step even though the neighbours' keys are never read. `FlashDecodeShape::pin_splits` in `crates/model/src/kv_tier.rs` is the remedy: it pins the split count to what a batch of one would get, so that "every row sees the same tile boundaries and the same number of partial records however many sequences share the batch". The mode is off by default, and the commit records a cost for it: the projections that must stay bit-identical run in smaller groups than the fastest kernel would take, so reproducibility is paid for in the efficiency of the weight reads.

## The position of the token that is not there yet

A decode step feeds one token, and the model must be told where it sits. In this chapter's model the position is added to the embedding before the first layer, so the new token must be given position $c$, the number of tokens already in the cache. The rule is as short as that, and it is where a cache is most often broken, because in a decode step the position cannot be read off from anything the step contains. The tokens of a prefill have positions $0, \dots, n - 1$ given by the index of the row within the call; the single row of a decode step has index 0 in its own call. A routine that numbers positions by the row index inside the call is right for every prefill and wrong for every decode step.

The table below changes only the position that the decode steps give the token, and measures what the logits do. The prefill is always right, and the keys and values are always written at the end of the cache.

<!-- worked: kv_cache.position_bugs -->
The cached generation of the last block (12 tokens), with the position given to the token fed at each decode step changed. The prefill is always right (positions 0, 1, 2) and the new keys and values are always appended at the end of the cache; only the number added to the token's embedding differs. Two measurements. Teacher forced: every variant is fed the naive loop's own tokens (4 6 6 4 8 0 6 2 6 2 4) at decode steps 2 to 12, so any difference in the logits comes from the position alone; the table gives the largest difference over those 11 steps and the number of steps at which the largest logit moves to another token. Free running: the step at which the variant's own generated token first differs from the naive loop's.

| position given to the fed token | largest logit difference | steps with another top token, of 11 | first wrong token when run freely |
|---|---:|---:|---:|
| correct: the position of the new token is the cache length | 0 (identical) | 0 | none |
| one too small: the last position that is already in the cache | 14.86 | 9 | 2 |
| never advanced: the length of the prompt for every decode step | 17.84 | 9 | 3 |
| restarted: position 0, the index of the row inside the call | 18.53 | 7 | 4 |
<!-- /worked -->

None of the three wrong variants crashes, produces a non-number or leaves a trace in any shape. All three give plausible logits, of the right size and in range, and the text that comes out is a text. The first row is correct and the generated tokens match. The off-by-one, a position one lower, moves the logits by up to 14.86 and changes the top token at 9 of the 11 decode steps; run freely it goes wrong at its second generated token. A position that is never advanced, the prompt length for every step, is right at the first decode step, where the cache length equals the prompt length, and wrong from the next one, so a test of a single step would pass it; it differs from the reference at 9 of 11 steps and, run freely, at the third token. The restart at 0 differs at 7 of 11 and, run freely, goes wrong at the fourth token. The symptom of this family is a text that begins as it should and then goes subtly wrong, or goes wrong at once but in a way that looks like the model's own failings; no error is ever raised. It shows only against a reference computed by a path that has no cache, and the most useful reference is the one this chapter uses, the cached run against the full forward pass at every length.

A second place for the same off-by-one sits in the kernel contract, where it takes a different form. `gqa_decode_batched_step` in `crates/model/src/block.rs` takes, for each sequence in a batch, an array `seq_lens` and documents that it "must COUNT THE NEW TOKEN (`seq_lens[b] == offsets[b] + block_index*block_size + 1`): it is both the causal bound and the softmax's own normalisation width". The new token is written into the cache before the attention reads it, so the number of keys the attention must read is the old length plus 1. An array that holds the old length makes the new token blind to itself, the diagonal entry of the mask of Chapter 7 gone, and for the first token of a sequence leaves the softmax with no key to normalise over, the failure that Chapter 7 showed for a mask one position too tight. An array that holds one more than the new length reads a slot that nothing has written for this sequence; in a pool whose blocks are recycled between sequences such a slot can hold another sequence's keys and values, and the average then contains them.

The model of the chapter uses learned positions, which are added to the stream once. With rotary embeddings (Chapter 8) the position acts inside every layer, on the queries and keys, and a second question arises: when is the key rotated? A key cached before rotation would have to be rotated on every read, by a number that depends on the reading position; a key cached after rotation is final. The score of a query at position $i$ and a key at position $j$ depends on the two positions only through the offset $j - i$ (Chapter 8), and the rotation applied to the key depends only on $j$, so a key rotated when it is stored and never touched again gives every future query exactly the score that recomputation would. That is the property Chapter 8 required of a position signal, that it must not depend on the length of the sequence, put to use. Brain stores keys already rotated: the contract of `gqa_decode_step` quoted above says its `k_new` is "already QK-normed + RoPE'd" when it arrives.

<!-- worked: kv_cache.rope_cache -->
Generator: the book's SplitMix64, seed 0xa8f8f44367e72b5c (`seed_of` of the name `kv_cache.rope`). Draws: the 8 entries of the query q, then the 8 entries of each of 5 keys k_0 .. k_4, each `round(normal(), 1)`. One head of width 8, adjacent pairs, base 10000 (Chapter 8, `positions.rotate`). The query is the token at position 5 and attends to the five cached keys; each score is the dot product of the rotated query and the key as stored, without the scale 1/sqrt(8).

| how the keys are cached and read | scores for keys 0 .. 4 | largest difference from recomputed | key with most weight |
|---|---|---:|---:|
| recomputed: every key rotated by its own position | 3.026 -7.238 -1.246 0.157 -2.428 | 0.000 | 0 |
| cached after rotation, read later | 3.026 -7.238 -1.246 0.157 -2.428 | 0.000 | 0 |
| cached before rotation, only the query rotated | 3.026 -6.955 -3.090 -1.081 -0.708 | 1.844 | 0 |
| cached after rotation, rotated again when read | 3.026 -5.952 -0.106 -0.071 -1.836 | 1.285 | 0 |
| query rotated by position 4 instead of 5 | 2.111 -5.952 0.005 1.876 -2.450 | 1.719 | 0 |
<!-- /worked -->

The first row is recomputation and the others are four ways of caching and reading. Caching after the rotation and reading later gives the scores of recomputation exactly. Caching before the rotation and rotating only the query changes every score but the first (key 0 sits at position 0, and its rotation is the identity), by up to 1.844; rotating an already rotated key a second time at read changes them by up to 1.285; and a query rotated for position 4 instead of 5 by up to 1.719. In this example the key with the most weight is key 0 in all five rows, so a check of which key wins would pass every one of them, while the output, a different average of the values, is wrong.

The same failure was found in brain in a form that leaves no doubt about how it arises. Chapter 8 told the story of commit `d63b3f031` (September 2026) from the side of the rotation table: a hybrid decoder had three call sites that built the rotation table, one for whole-sequence prefill, one for single-step decode and one for the chunked rounds that a prompt replay and a speculative verification run on, and the third had been left on the unscaled table when the other two were given a long-context scaling. From the side of the cache the story is simpler. The chunked prefill wrote keys into the cache rotated one way; the decode steps that followed read them with queries rotated another way. The message calls it a model that "wrote its K/V rows at one rotation and read them back with queries at another", silent because it needs the scaling on, a round boundary crossed and a position past the original context at once. The regression test `yarn_scaled_chunked_prefill_matches_token_by_token_replay` in `crates/qwen35/tests/it/yarn_rope_scaling.rs` is cache equivalence in miniature: it prefills a 14-token prompt in rounds of 4 and then decodes three more tokens, and asserts that the last hidden state and each later step agree with a token-by-token replay to $10^{-5}$. Before the fix, the message reports, the test failed with a difference of 5.86e-4 against its bound of 1e-5, 58 times the bound.

One more consequence of the same rule is worth stating because it breaks the short version of it. "The position of the new token is the cache length" holds for text. It does not hold for a prompt that carries images, where a block of image rows spans fewer rotary positions than rows. `PromptRope` in `crates/qwen3vl/src/mrope.rs` documents it: "A prompt token's KV row is its index; its rotary position is not", and every generated token sits at its KV row shifted by a constant. The cache then needs two counters, where the row to write and where the token sits are different numbers. Commit `05f7b182a` (November 2026) wires that into the paged engine's decode steps ("decode past the image at the shifted position"). A sequence without images has no `PromptRope`, and its rotary position is its row.

## The bill: bytes per token of context

The saving came from storing two vectors per token and layer. How much is that? Each layer stores a key row and a value row of $h_{\mathrm{kv}} d_h$ numbers for every token, so with $b$ bytes per value the cache of $c$ tokens is

$$
\mathrm{cache}(c) = 2\,L\,h_{\mathrm{kv}}\,d_h\,b\,c \quad\text{bytes}.
$$

The 2 counts keys and values, and everything else is a count of what the model is made of: layers, key-value heads, the width of a head and the size of a stored number. None of them depends on the vocabulary, on the width of the feed-forward network or on the batch. The cost per token of context, $2Lh_{\mathrm{kv}}d_h b$, is a fixed property of the architecture and the format.

<!-- worked: kv_cache.cache_memory -->
The four architectures, with every number the weights need. The weights counted are the parameters that take part in a matrix product per token, as in the text: per block W_Q and W_O (D by H d_h each), W_K and W_V (D by h_kv d_h each) and the feed-forward network (m matrices D by F, m = 2 for GPT-2 small's GELU network and 3 for the SwiGLU network of the others), times L, plus one head D by V (a matrix of its own, not tied to the embedding table, whose lookup is not a product); norm gains and position tables are left out. 'Same, 32 key-value heads' is the 7.5-billion decoder with W_K and W_V widened from D by 1,024 to D by 4,096, which is why its weights are larger.

| model | V | D | L | query heads H | key-value heads h_kv | head width d_h | F | m | parameters in matrix products |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 small | 50,257 | 768 | 12 | 12 | 12 | 64 | 3,072 | 2 | 123,532,032 |
| 7.5-billion decoder, 8 key-value heads | 128,256 | 4,096 | 32 | 32 | 8 | 128 | 14,336 | 3 | 7,504,658,432 |
| same, 32 key-value heads | 128,256 | 4,096 | 32 | 32 | 32 | 128 | 14,336 | 3 | 8,309,964,800 |
| 70-billion decoder, 8 key-value heads | 128,256 | 8,192 | 80 | 64 | 8 | 128 | 28,672 | 3 | 69,501,714,432 |

Cache bytes per token = 2 (keys and values) x layers x key-value heads x head width x bytes per value, in bf16 (2 bytes). The context columns are GiB of cache for ONE sequence of that many tokens; the last column is the GiB of the weights in bf16 (2 bytes per parameter of the table above), for comparison.

| model | 2 x L x kv heads x d_h x bytes | bytes per token | GiB at 1,024 | GiB at 4,096 | GiB at 32,768 | GiB at 131,072 | weights, GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-2 small | 2 x 12 x 12 x 64 x 2 | 36,864 | 0.04 | 0.14 | 1.12 | 4.50 | 0.2 |
| 7.5-billion decoder, 8 key-value heads | 2 x 32 x 8 x 128 x 2 | 131,072 | 0.12 | 0.50 | 4.00 | 16.00 | 14.0 |
| same, 32 key-value heads | 2 x 32 x 32 x 128 x 2 | 524,288 | 0.50 | 2.00 | 16.00 | 64.00 | 15.5 |
| 70-billion decoder, 8 key-value heads | 2 x 80 x 8 x 128 x 2 | 327,680 | 0.31 | 1.25 | 10.00 | 40.00 | 129.5 |

Sessions of 32,768 tokens that fit beside the weights on a device of 80 GiB (the cache is the only other memory counted):

| model | GiB free after the weights | GiB per session | sessions that fit |
|---|---:|---:|---:|
| 7.5-billion decoder, 8 key-value heads | 66.0 | 4.00 | 16 |
| same, 32 key-value heads | 64.5 | 16.00 | 4 |
<!-- /worked -->

GPT-2 small, with 12 layers and 12 heads of width 64, needs 36,864 bytes per token in bf16, 36 MiB for its full context of 1,024. The 7.5-billion decoder of the last sections needs 131,072 bytes per token, 128 KiB: 4.00 GiB for 32,768 tokens and 16.00 GiB for 131,072, which is more than its own weights, 14.0 GiB. If it had one key-value head for every query head the bytes per token would be four times as many, 524,288, and a context of 131,072 would need 64 GiB. Grouped key-value heads exist to shrink this factor (a later volume derives them; here they appear only as $h_{\mathrm{kv}}$ in the formula). The 70-billion decoder with the same eight key-value heads needs 327,680 bytes per token and 40 GiB at the long context, on top of weights of 129.5 GiB (its 69,501,714,432 parameters in matrix products, from the architecture in the first table, at 2 bytes each).

The consequence the second table draws is the one that matters for a service. On a device of 80 GiB that holds the 7.5-billion decoder, 66.0 GiB remain for caches. Each session with 32,768 tokens takes 4.00 GiB, so 16 sessions fit; with 32 key-value heads each takes 16.00 GiB and 4 fit. The memory of the device is divided between the weights, which are paid once, and the caches, which are paid per conversation, and with a long context or many sessions the second is the larger. A model that is cheap to run for one user at a short context is not cheap to run for sixteen at a long one. That is why the cache is the dominant memory cost of serving long contexts, and why later volumes spend so much of their length on what to do about it: fewer key-value heads, caches stored in fewer bits, caches moved to host memory or dropped and recomputed, and a mix of all three.

The 4 to 1 ratio between the two decoder variants is the reason grouped key-value heads exist, and whether a model keeps its quality with fewer of them is a question for the chapter that derives them. The other factor in the formula, $b$, is the lever of `crates/model/src/kv_tier.rs`, which defines four element formats for the cache, 32-bit, bf16, 8-bit integer and 8-bit float, the last two with one scale for every token and key-value head, and holds each to a bound against the full-precision cache (the real-checkpoint test quoted above is one of those bounds). Both belong to later chapters; the formula above is what they reduce.

The bill has a time side too, which the first figure showed: every decoded token reads the whole cache, $c$ rows of it, so a long context makes every token a long read however many sessions share the device. Chapter 13 will show how to avoid holding the $T \times T$ score matrix, which is a different memory from this one and leaves the cache as it is.

## Many sequences, none of them the same length

A server rarely holds one sequence. Several requests are in flight, each with its own prompt length and its own point in its reply, and a decode step wants to advance them all at once so that the weights are read once for all (the lower table of the roofline section). Sequences of lengths 7, 3 and 12 form a ragged set. A rectangular cache, an array of shape [3, 12] sized for the longest, reserves $3 \times 12 = 36$ rows to hold $7 + 3 + 12 = 22$, and its attention must be told per sequence how many of its rows are live, or the dead rows join the average. The next volumes go further: a cache that is allocated per sequence in fixed-size blocks and mapped through a table instead of one reserved rectangle, which is how Kwon and colleagues (2023) built the memory manager of a serving system, on the model of the paging of virtual memory. Brain has it as the paged cache of `crates/model/src/paged.rs`.

The core of it is small enough to state here. A `BlockTable` is a map from a sequence's logical positions to physical blocks of a shared pool; `locate(tok, block_size)` returns the block and the offset of token `tok`, which are `blocks[tok / block_size]` and `tok mod block_size`. Its `append` reserves room for one more token, allocating a block when the current one is full, and returns the block and offset the caller writes the new key and value into, and in doing so it takes the offset from the length before it advances the length. The position of the new token is the length before the append, which is the rule of the last section in the language of the allocator. The first commit of the module, `dde2dc0fb` (July 2026), describes its purpose in one line: a sequence's KV "grows one block at a time instead of reserving its worst-case length".

Two more properties of the cache matter once several sequences share a pool, and both come from the same sentence of the statement proved above: a stored row is a function of the weights that computed it and of the tokens at positions up to its own. The first is that a row that has been written is not wrong for a later request that begins with the same tokens, and that is the basis of a prefix cache, which keeps the blocks of a finished prompt so that the next request with the same beginning skips its prefill (a later chapter). The second is the converse. A row cached under one set of weights is wrong under another. Commit `53bd99ecd` (October 2026) states it for adapters: "A block's K/V depend on the weights that computed it, so a prefix cached under the base model or one runtime adapter is wrong under any other". The `BlockTable` now carries a namespace, which is the adapter its keys and values were computed under; the prefix cache is keyed by it, and the identifiers of adapter namespaces are never reused, "so a stale id cannot reach a later adapter's blocks". The documentation of `gqa_decode_batched_step` closes the other kind of staleness at the kernel: the append overwrites its slot, so a slot recycled from a finished sequence "carries nothing stale", provided the attention never reads past the sequence's own length, which is the contract of `seq_lens` above.

## How the pieces run together in a server

The two phases do not just differ in cost, they differ in what a scheduler should do with them. Prefill is arithmetic-heavy and finishes; decode is byte-heavy and goes on. A server that decides to run a whole new prompt before returning to the sequences that are decoding makes every one of them wait for it, and a long prompt is a long wait. `crates/model/src/serve.rs` records the reason in the documentation of its prefill budget: without a bound on the prompt tokens prefilled per iteration, "a burst of N arrivals performs N whole prompt forwards back-to-back while every running sequence stalls". One iteration of the scheduler resumes sequences that were moved out of the device's memory, admits waiting requests within the budget and prefills them in slices (`prefill_chunk` continues a prompt where the previous call for it stopped, and the chunked form of the decode step, `gqa_chunk_step`, generalizes `gqa_decode_step` from one query row to $n$), runs one batched decode step over the running sequences and removes the ones that have finished.

The chunked step restates the position rule once more. Its row $i$ sits at the absolute position $\mathrm{start} + i$, not at $i$, so the causal bound is carried per row, by `seq_lens[i] == start + i + 1`, rather than by a comparison $j \le i$ that assumes the chunk begins at 0. Its cache fill needed a new kernel. `kv_cache_fill` writes rows `0..n` only: its destination is `row*width`, which, in the words of its documentation, "pins the destination to a multiple of the chunk length", and a later round of a chunked prefill starts wherever the previous one ended. Commit `85255a05b` (September 2026), which introduced the chunked primitives, added `kv_cache_fill_at` with an independent destination offset. It is the same family as the table above: a number that is right only when the cache is empty.

The scheduler can also run decode rows and prompt slices as one forward. Commit `f74979a12` (October 2026) added a mixed iteration in which a slice of the waiting prompts rides along with the decode rows, so that "each weight is read once for both", as the documentation of `crates/model/src/serve/budget.rs` puts it. In this chapter's terms, the decode rows bring the intensity down and the prompt rows bring it up, and a mixed forward is a way of filling the arithmetic that the decode rows leave idle. How large the slice may be is set by the time one iteration is allowed to take, which every decoding sequence experiences as the gap between its tokens. The later volume that treats scheduling picks this up.

## Check your understanding

1. A colleague caches the queries as well as the keys and values, "to be safe". Which of the three is never read again after the step that computes it, and what does the extra storage cost per token for the 7.5-billion decoder if queries are stored at the same width as the 32 query heads?
2. In the tiny model of this chapter a layer-0 key is identical with and without the mask, but a layer-1 key is not. Explain why in one sentence, and say what a leaky mask would not show in a test that compared only layer-0 keys.
3. A cached run and a full run agree to $10^{-14}$ on the logits. Someone concludes that the cache is correct. What else must the test cover before the conclusion is safe, using the position bugs of this chapter?
4. The decode step of a model reads $15 \times 10^9$ bytes of weights and $10^9$ of cache and does $16 \times 10^9$ FLOPs. On a device with a ridge of 50 FLOP per byte, is it limited by the arithmetic or by the bytes? What does batching 16 sequences of the same length do to the intensity?
5. "Cache the key before the rotary rotation, because then it is the same for every query." Find the error.
6. A prefix cache keeps the blocks of an earlier request so that a new request with the same first 500 tokens skips its prefill. Under what change to the system is that unsafe, and which line of the proof says so?
7. A test compares a cached decode with a full forward pass for a single prompt of 12 tokens split at 3. List two other splits it should also try and what each one exercises.

## Summary

Generating a token by running the whole text through the model repeats, at every step, the work of the steps before it: for a prompt of $p$ and $N$ new tokens the loop computes $pN + N(N-1)/2$ rows and the arithmetic of the weights grows with the square of $N$. The causal mask makes the key and value of a position, in every layer, a function of the tokens up to it, so they can be stored once, in a cache of two rows per token and layer, and read by every later token. Running the prompt in one call fills the cache (prefill) and running one token at a time appends to it and attends to it (decode); the cached run does 6 rows of work where the naive one did 18, and its saving grows roughly in proportion to the number of tokens generated. The cost moves to memory: a decode step reads all weights and all the cache for one row of arithmetic, at about 1 FLOP per byte, which is limited by the bytes on any device whose ridge is above 1; prefill gives the same bytes many rows and is limited by the arithmetic beyond about 54 tokens in the example. A cache equivalent to the full computation is a property to be tested, to a tolerance chosen against the reordering of sums and the noise of quantization, and the usual ways to break it are numbers that are right only when the cache is empty: the position of the new token, the count of live rows and the rotation of the stored key. The memory is $2Lh_{\mathrm{kv}}d_hb$ bytes per token, 128 KiB for the 7.5-billion decoder, 16 GiB at 131,072 tokens, more than the weights, and every session pays it.

The limits are those of the model: it is small, its weights are random, and the byte counts are arithmetic and not measurements of any device. What the chapter did not do is touch the other large memory of attention, the $T \times T$ matrix of scores that training and prefill would hold if they computed it as the text of Chapter 7 does. The next chapter computes attention exactly without ever holding it, which removes that memory from the long prefill of this chapter.

## References

- R. Pope, S. Douglas, A. Chowdhery, J. Devlin, J. Bradbury, A. Levskaya, J. Heek, K. Xiao, S. Agrawal and J. Dean, "Efficiently scaling transformer inference", 2023.
- W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang and I. Stoica, "Efficient memory management for large language model serving with PagedAttention", 2023.

## Exercises

1. **[Calculate]** For the tiny model of the chapter ($D = 16$, $L = 2$, $H = 2$, $F = 24$, $V = 12$, three weight matrices in the feed-forward network) give $N_{\mathrm{blk}}$ and evaluate $F(c, n)$ for $(c, n) = (0, 3)$ and $(3, 1)$. Check the first against the table of the naive loop and the second against the cached one.
2. **[Calculate]** A decoder has 24 layers, 16 key-value heads of width 64 and stores the cache in bf16. How many bytes does one token of context cost, and how many GiB does a context of 8,192 tokens need? How many such sequences fit in 24 GiB if the weights take 8 GiB?
3. **[Calculate]** In bf16, with $H$ query heads and $h_{\mathrm{kv}}$ key-value heads of width $d_h$ per layer, show that the attention products of a decode step have an intensity of $H / h_{\mathrm{kv}}$ FLOPs per byte of cache read, whatever the context. Evaluate it for 32 and 8 heads and say what it implies about the limit of batching.
4. **[Calculate]** A cache holds 7 tokens. At which row does the next token's key go, which position does the model add to its embedding, and what is the cache length afterwards? In a paged pool with blocks of 4 rows, give the block index and the offset.
5. **[Calculate]** In 32-bit arithmetic with 24-bit significands, the numbers 16,777,216 ($2^{24}$), 1 and 1 are added in the two orders $(2^{24} + 1) + 1$ and $2^{24} + (1 + 1)$. Give both results and say which is closer to the exact sum.
6. **[Calculate]** A cache stores, instead of the key and value of every layer, the stream $\mathbf{x}^{(l)}$ at its input, from which the keys and values can be recomputed when needed. How many numbers per token and layer does that store, against the cache of the chapter for $h_{\mathrm{kv}} d_h = D$? What does it cost in FLOPs per cached token on every step?
7. **[Implement]** Reproduce `kv_cache.naive_generation` and `kv_cache.cached_generation`: write `extend` of the chapter's pseudocode on the tiny model (its weights are fixed by the seed, the draw order and the standard deviation printed in `kv_cache.tiny_model`), count the multiply-adds as the text says, and check that the two runs give the same tokens and the logits agree.
8. **[Implement]** Reproduce `kv_cache.mask_invariance`: run the six-token text on its first three tokens and whole, under the causal mask and under no mask, and compare the keys and values of the first three positions in every layer.
9. **[Implement]** Reproduce `kv_cache.summation_order` and `kv_cache.cache_equivalence`: the five orders of the float32 sum of the 64 numbers of the stated generator, and the decode steps whose dot products and weighted sums are added in the orders described in the text.
10. **[Implement]** Reproduce `kv_cache.position_bugs` and `kv_cache.rope_cache`: the four position rules, with teacher forcing and run freely, and the five ways of caching and reading the rotated keys.
11. **[Investigate]** Test cache equivalence at every split of the 12-token text of `kv_cache.cache_equivalence` into a prefill of $c$ tokens and a sequence of single-token decode steps, for $c = 1, \dots, 11$, and then with the part after the prefill run in chunks of 2 and of 5 tokens. Does any split give a larger difference than the one reported? The chapter ran one split.
12. **[Investigate]** Replace the learned positions of the model by the rotary embeddings of Chapter 8 (rotating queries and keys of every layer by the position of their token) and repeat `kv_cache.position_bugs` with the key cached before and after the rotation. Does the table keep its shape? The chapter did not run this.

## Key terms

- **KV cache**: for every layer, the key and the value of every token already seen, stored once and read by every later token, so that the past is never run through the model again; its size is two rows of $h_{\mathrm{kv}} d_h$ numbers per token and layer. <!-- concept: kv-cache -->
- **prefill phase**: processing the whole prompt in one call, with every matrix seeing all the prompt's rows at once and the keys and values of every prompt position written into the cache; limited by the arithmetic once the prompt is long. <!-- concept: prefill-phase -->
- **decode phase**: generating one token at a time, each step running one row through the model, appending one key and one value per layer to the cache and attending to everything in it; limited by the bytes of the weights and the cache. <!-- concept: decode-phase -->
- **cache equivalence**: the property that running a text as a prefill followed by cached decode steps gives the same logits as one forward pass over the whole text, exactly when the arithmetic is the same and to a stated tolerance when only the order of the sums differs. <!-- concept: cache-equivalence -->

## Answers

**Recall questions.**

1. The output at position $i$ uses the query of position $i$ and the keys and values of the positions it attends to; from its own position it uses the query, and also the key and value, because the diagonal of the mask is open. From other positions $j \le i$ it uses only keys (to score) and values (to average), never their queries.
2. Position $i$ (from 1) sees $i$ positions, so $1 + 2 + 3 + 4 + 5 = 15$ of the 25 pairs are allowed, and 10 are hidden.
3. The score depends on the offset $j - i$ only, not on either position alone. The rotation applied to the key depends on the key's own position $j$ and on nothing else.
4. Token 1: the largest logit is 3.4 and on a tie the lowest index wins.
5. Storing the result once costs $n$ cells of memory, $\Theta(n)$, and saves the time of every recomputation after the first: the time-memory tradeoff of the chapter, spending memory to avoid recomputing.
6. Intensity 2 is below the ridge $40{,}000 / 800 = 50$, so the step is limited by the bytes (bandwidth-bound); above 50 FLOPs per byte it would be limited by the arithmetic.

**Check your understanding.**

1. The query. It is used to produce the output of its own position and then never again, so storing it costs memory and saves nothing. At the width of 32 heads of 128 it would add $32 \times 128 \times 32\ \text{layers} \times 2\ \text{bytes} = 262{,}144$ bytes per token, twice the 131,072 bytes of the cache itself.
2. The key of layer 0 is a function of the token's own embedding and position, before any attention has mixed in other tokens, so the mask cannot change it; from layer 1 on, the stream at a position contains what attention took from the others. A test of layer-0 keys alone would pass a model with no mask or a leaky one.
3. It must cover the positions and the counts: a run that agrees at one split with a correct position can still be wrong at another, because a position that is never advanced agrees with the correct one at the first decode step and differs from the second on. The test should compare every decode step against the full forward pass, at several prefill lengths, and with prefill rounds of different sizes.
4. By the bytes: the intensity is $16 / 16 = 1$ FLOP per byte, below 50. Batching 16 sequences reads the $15 \times 10^9$ bytes of weights once and 16 caches of $10^9$ bytes, for $16 \times 16 \times 10^9 = 256 \times 10^9$ FLOPs: $256 / 31 = 8.3$ FLOP per byte, higher but still below 50.
5. A key cached before rotation would have to be rotated at every read by the position it was stored at, which is not a function of the reader, and rotating it by the reader's position instead gives the wrong offset. The score depends on $j - i$, so the key must carry its own rotation by $j$, applied once at storage; then every later query's rotation by $i$ completes the offset.
6. If the weights change, for example an adapter is loaded or swapped: the statement says the keys and values are functions of the tokens through the weights, and in the proof the matrices $\mathbf{W}_K$, $\mathbf{W}_V$ and everything below are fixed. A different set of weights is a different function.
7. A split at 1 (a prefill of a single token, which is a decode step, and the shortest case) and a split at 11 (a single decode step after a long prefill, which exercises the cache of the longest length). A split at a multiple of a block size, where a paged cache moves to a new block, is the third.

**Exercises.**

1. $N_{\mathrm{blk}} = L(4D^2 + 3DF) = 2 (1024 + 1152) = 4352$. $F(0, 3) = 2 \cdot 3 \cdot 4352 + 4 \cdot 2 \cdot 2 \cdot 8 \cdot (0 + 6) + 2 \cdot 16 \cdot 12 = 26112 + 768 + 384 = 27264$, the first row of the naive table. $F(3, 1) = 8704 + 128 \cdot 4 + 384 = 9600$, the first decode step. (Here $4 L H d_h = 4 \cdot 2 \cdot 2 \cdot 8 = 128$.)
2. $2 \times 24 \times 16 \times 64 \times 2 = 98{,}304$ bytes per token. For 8,192 tokens, $805{,}306{,}368$ bytes, which is 0.75 GiB. In $24 - 8 = 16$ GiB, $16 / 0.75 = 21.3$, so 21 sequences.
3. FLOPs of the attention products per layer per cached key: $4 H d_h$ (scores and weighted sum, 2 each per head dimension). Bytes: $2 h_{\mathrm{kv}} d_h b$ with $b = 2$, which is $4 h_{\mathrm{kv}} d_h$. The ratio is $H / h_{\mathrm{kv}}$, the number of query heads that share each key-value head, whatever the context. For 32 and 8 heads, 4 FLOPs per byte, far below the ridge of 50: attention over the cache alone is always bandwidth-bound on such a device, and no batch can lift it.
4. Row 7 (0-indexed), position 7, cache length 8 afterwards. In a pool with blocks of 4: logical block $7 \div 4 = 1$, offset $7 \bmod 4 = 3$.
5. $2^{24} + 1$ is exactly halfway between $2^{24}$ and $2^{24} + 2$, and ties go to the even significand, which is $2^{24}$; adding 1 again gives $2^{24}$. So $(2^{24} + 1) + 1 = 16{,}777{,}216$. The other order adds $1 + 1 = 2$ first and then $2^{24} + 2 = 16{,}777{,}218$, exactly representable. The exact sum is 16,777,218, so the second order is exact and the first is off by 2.
6. It stores $D$ numbers per token and layer against $2 h_{\mathrm{kv}} d_h = 2D$ for the cache of the chapter, half as many. Recomputing the key and the value of one cached token in one layer is two products of a vector of $D$ with a matrix of $D \times h_{\mathrm{kv}} d_h = D \times D$, $4D^2$ FLOPs, at every step; with $D = 4096$ and 32 layers that is $4 \cdot 4096^2 \cdot 32 = 2.1 \times 10^9$ FLOPs per cached token per step. The saving of bytes is bought with an arithmetic bill that grows with the context.
7. The tokens agree, and the logits agree exactly under the same summation order; the counts of FLOPs are those of the tables (27,264 for the prompt, then 9,600, 9,728 and 9,856, and a total of 56,448 against 164,864).
8. Every difference is exactly 0 under the causal mask. Without the mask, layer 0 is still 0 and layer 1 differs by 10.8, 11.6 and 0.00411 at positions 0, 1 and 2.
9. The five float32 sums of the table (4.44071579, 4.44071484, 4.44071531, 4.44071674 and 4.44071722 in the order listed), and for the equivalence table: 0 for the same order and below 1e-13 for the other, with the same 12 tokens.
10. For the position rules: 0 difference for the correct one, and largest differences of 14.86, 17.84 and 18.53 for the three wrong ones, with another top token at 9, 9 and 7 of 11 steps. For the rotated keys: zero difference for the stored rotated key, 1.844, 1.285 and 1.719 for the three wrong ones.
11. An investigation. The chapter ran one split. The expected result is that no split exceeds the tolerance with any order of sums, and that the exact split $c$ where the number of keys crosses a multiple of 4 (the block size of the lane order) is where an error in a block-sum implementation would show.
12. An investigation. The chapter did not run it. The expected result is that the caching-before-rotation variant keeps the same large differences and the position variants keep theirs, since the rotary position acts in every layer and not once.
