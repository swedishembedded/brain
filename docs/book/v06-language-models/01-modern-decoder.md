<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Anatomy of a Modern Decoder

*What changed between the first GPT and today's open-weight language models, and why?*

Print the parameter list of one block of the GPT-2 that Volume V trained (`crates/gpt2`) next to the parameter list of one block of Qwen3 (`crates/qwen3`), and read the names.

| what the tensor is for | GPT-2 block, 12 tensors | Qwen3 block, 11 tensors |
|---|---|---|
| first normalization | `ln1.weight`, `ln1.bias` | `ln1.weight` |
| queries, keys, values | `attn.qkv.weight`, `attn.qkv.bias` | `attn.wq.weight`, `attn.wk.weight`, `attn.wv.weight` |
| per-head normalization of queries and keys | none | `attn.q_norm.weight`, `attn.k_norm.weight` |
| attention output | `attn.out.weight`, `attn.out.bias` | `attn.wo.weight` |
| second normalization | `ln2.weight`, `ln2.bias` | `ln2.weight` |
| feed-forward network | `mlp.fc.weight`, `mlp.fc.bias`, `mlp.proj.weight`, `mlp.proj.bias` | `mlp.gate.weight`, `mlp.up.weight`, `mlp.down.weight` |

Outside the blocks the lists differ as well: GPT-2 has a table of positions (`pos.weight`) and a final normalization with a shift (`ln.bias`) that Qwen3 does not, and Qwen3 has a separate output matrix (`lm_head.weight`) only in its larger sizes (brain's `crates/gpt2` keeps its head untied, as Volume V, Chapter 10 noted; the tied head of the published model is what the counts below use). Not one name on the left is a name on the right except the two normalization gains, and even those lost a partner. The fused matrix for queries, keys and values has come apart into three; the biases are gone; the feed-forward network has a third matrix; two small vectors have appeared that GPT-2 had no use for. Six years of work on language models sit in the difference between these lists, and the lists do not say why any of it happened.

This chapter gives each difference a reason, and the reasons are counted: parameters, FLOPs, bytes of cache, and in one case a bound on a number that can otherwise grow without limit. Some of the changes are nearly free and are made because they remove things (a position table, a bias, half of a normalization). Three of them change the economics of the decoder and get their own sections: sharing keys and values between query heads, normalizing queries and keys before they meet, and giving the output its own matrix. The chapter closes by building a decoder small enough to check, with every choice at once, and by reading how brain's Qwen3 carries the same choices in its configuration.

Six ideas from earlier chapters are needed first. Try them; the answers are at the end of the chapter.

1. (Volume V, [Chapter 7](../v05-sequences/07-self-attention.md), multi-head attention.) In one layer of 12-head attention at width 768, with heads of width 64, how many key vectors does a token produce, and how many numbers do they hold in all?
2. (Volume V, [Chapter 8](../v05-sequences/08-positions.md), rotary embeddings.) At which point of attention is the rotation applied, to which of the three of query, key and value, and what does the score of a pair of positions depend on afterwards?
3. (Volume III, [Chapter 10](../v03-neural-networks/10-normalization.md), RMS normalization.) Write the RMS normalization of a row $\mathbf{x}$ of $D$ numbers with gain $\boldsymbol{\gamma}$ and floor $\epsilon$. How many learned numbers does it have, against LayerNorm?
4. (Volume V, [Chapter 9](../v05-sequences/09-transformer-block.md), SwiGLU.) Write the SwiGLU network from its three matrices and their shapes, and count its parameters at $D = 768$ and $F = 2{,}048$.
5. (Volume V, [Chapter 12](../v05-sequences/12-kv-cache.md), the KV cache.) How many bytes does one token of context cost in the cache of GPT-2 small (12 layers, 12 heads of width 64) in bf16?
6. (Volume III, [Chapter 6](../v03-neural-networks/06-gradient-checking.md), gradient checking.) What does a gradient check compare for one entry of a weight matrix, and why is the difference central?

## Two recipes, row by row

The table below puts GPT-2 small beside Qwen3 0.6B and Qwen3 8B, with the shapes of brain's presets (`QwenConfig::qwen3_0_6b` and `qwen3_8b` in `crates/qwen3/src/config.rs`). The first is the model of Volume V, Chapter 10; the second is the small end of an open-weight family, and the third the first size of that family, in brain's presets, with an untied head. Every number in the last two rows is counted from the shapes above it.

<!-- worked: modern_decoder.side_by_side -->
The choices of three decoders. GPT-2 small is the published model of Volume V, Chapter 10. Qwen3 0.6B and 8B are the shapes of brain's presets `QwenConfig::qwen3_0_6b` and `qwen3_8b` (`crates/qwen3/src/config.rs`), with the published context of the 0.6B checkpoint (`max_position_embeddings`) in the context row (the 8B's is not pinned by brain's sources and is left blank). Parameters are counted by `parameter_groups` from the shapes (every tensor listed in the next table); the cache row is 2 x layers x key-value heads x head width x 2 bytes.

| choice | GPT-2 small | Qwen3 0.6B | Qwen3 8B |
|---|---|---|---|
| positions | learned table, 1,024 rows | rotary, base 1,000,000, no table | rotary, base 1,000,000, no table |
| normalization | LayerNorm (gain and shift) | RMS norm (gain only) | RMS norm (gain only) |
| feed-forward network | GELU network, 768 to 3,072 to 768 | SwiGLU, 1,024 to 3,072 to 1,024, three matrices | SwiGLU, 4,096 to 12,288 to 4,096, three matrices |
| biases on linear maps | all of them | none | none |
| attention | 12 heads, one key-value head each | 16 query heads, 8 key-value heads (groups of 2) | 32 query heads, 8 key-value heads (groups of 4) |
| head width | 64 | 128 (H x head width = 2,048, width D = 1,024) | 128 (H x head width = 4,096, width D = 4,096) |
| QK-norm | none | per head, before rotation | per head, before rotation |
| output head | tied to the embedding | tied to the embedding | separate matrix |
| vocabulary | 50,257 | 151,936 | 151,936 |
| context | 1,024 (trained) | 40,960 (published) |  |
| layers | 12 | 28 | 36 |
| parameters | 124,439,808 | 596,049,920 | 8,190,735,360 |
| cache bytes per token (bf16) | 36,864 | 114,688 | 147,456 |
<!-- /worked -->

The right-hand columns are not a quirk of one family. Pre-norm with RMS normalization, SwiGLU, rotary positions and no biases were assembled into one block by LLaMA (Touvron and colleagues, 2023), and the open-weight decoders since have mostly kept them. Two choices go beyond that block: grouped key-value heads, which brain's Qwen2 presets already have (14 query heads over 2 key-value heads at 0.5 billion parameters), and the normalization of queries and keys, which is what Qwen3 adds to Qwen2 (`qk_norm` is false in `QwenConfig::qwen2`). A separate output matrix is chosen per size. Call this set of choices the **modern decoder recipe**: a pre-norm decoder with RMS norms and no biases, rotary positions applied to normalized queries and keys, attention whose key-value heads are shared by groups of query heads, a SwiGLU network, and an output matrix that is tied or untied according to size. Its pieces are the ones of Volume V with three of them reshaped, which the figure shows block against block.

![Two columns of eight stacked boxes, the decoder of GPT-2 small on the left and of Qwen3 0.6B on the right, with a dashed frame around the repeated layers: embedding, normalization, projections, a step between projection and scores, attention, normalization, feed-forward network, final normalization and head. The left column has LayerNorm, projections with biases and 12 heads, nothing between projection and scores, a GELU network, and a tied head, with 124,439,808 parameters. The right column has no position table, RMS norm, 16 query heads over 8 key-value heads, a highlighted QK-norm and rotation step, a SwiGLU network, and a head tied at this size, with 596,049,920 parameters.](img/gpt2-vs-modern-block.svg)

The parameters of the two models are counted tensor group by tensor group in the next table. It is the place to look first when a change is said to save something.

<!-- worked: modern_decoder.parameter_table -->
Every tensor of the three decoders, counted from the shapes of the previous table (zero where the decoder has none). A layer's attention holds W_Q (D by H x head width), W_K and W_V (D by key-value heads x head width) and W_O (H x head width by D); a layer's network holds 2 (GELU) or 3 (SwiGLU) matrices of D by F; the QK-norm of Qwen3 has a gain of head-width numbers for queries and again for keys in every layer.

| group | GPT-2 small | Qwen3 0.6B | Qwen3 8B |
|---|---:|---:|---:|
| token embedding table | 38,597,376 | 155,582,464 | 622,329,856 |
| learned position table | 786,432 | 0 | 0 |
| attention matrices W_Q, W_K, W_V, W_O | 28,311,552 | 176,160,768 | 1,509,949,440 |
| attention biases | 36,864 | 0 | 0 |
| QK-norm gains | 0 | 7,168 | 9,216 |
| feed-forward matrices | 56,623,104 | 264,241,152 | 5,435,817,984 |
| feed-forward biases | 46,080 | 0 | 0 |
| normalizations (two per layer and the last) | 38,400 | 58,368 | 299,008 |
| separate output matrix | 0 | 0 | 622,329,856 |
| **total** | 124,439,808 | 596,049,920 | 8,190,735,360 |
| embedding table as a share of the total | 31.0% | 26.1% | 7.6% |
<!-- /worked -->

The Qwen3 0.6B column holds 596,049,920 parameters, nearly five times the 124,439,808 of GPT-2, and the shares have moved: the embedding table is 26.1 percent of the smaller model's parameters and 7.6 percent of the 8B's. The feed-forward matrices are 44 percent of the parameters of Qwen3 0.6B, the attention matrices 30. Nothing in the table says which of the differences earns its place, because a count of parameters cannot tell a useful change from a harmless one. For that, the changes have to be applied one at a time.

## One change at a time, counted

Start from GPT-2 small at its own width of 768, its 12 layers and its vocabulary, apply the changes in the order below, and recount after each. Three things are counted: the parameters; the bytes that one token adds to the cache, the price of Chapter 12 of Volume V; and the FLOPs of the matrix products of one decoded token that sees 1,024 keys, by the formula of the same chapter. The last is a count of products only: norms, rotations and gates are small beside them, as Volume III showed for the norms.

<!-- worked: modern_decoder.ladder -->
Changes applied one after another to GPT-2 small at its own width (768), depth (12) and vocabulary. The first row is GPT-2; each later row keeps all earlier changes. SwiGLU takes F = 2,048 so that its three matrices (3 x 768 x 2,048) hold exactly as many numbers as the two matrices of the GELU network (2 x 768 x 3,072). Cache bytes are per token in bf16. FLOPs are those of the matrix products of one decoded token that sees 1,024 keys (cache of 1,023): 2 FLOPs for every weight of the blocks, 4 x layers x H x head width for every visible key, and 2 x D x V for the head.

| step | parameters | change | cache bytes | change | FLOPs per decoded token | change |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 small | 124,439,808 |  | 36,864 |  | 284,812,800 |  |
| rotary embeddings, no position table | 123,653,376 | -786,432 | 36,864 | +0 | 284,812,800 | +0 |
| RMS norm instead of LayerNorm | 123,634,176 | -19,200 | 36,864 | +0 | 284,812,800 | +0 |
| SwiGLU, hidden width 2,048 | 123,646,464 | +12,288 | 36,864 | +0 | 284,812,800 | +0 |
| no biases | 123,551,232 | -95,232 | 36,864 | +0 | 284,812,800 | +0 |
| 4 key-value heads for 12 query heads | 114,114,048 | -9,437,184 | 12,288 | -24,576 | 265,938,432 | -18,874,368 |
| QK-norm | 114,115,584 | +1,536 | 12,288 | +0 | 265,938,432 | +0 |
| separate output matrix | 152,712,960 | +38,597,376 | 12,288 | +0 | 265,938,432 | +0 |
<!-- /worked -->

Six of the seven changes leave the cache and the FLOPs of the matrix products where they were. Four of them change what the block is made of, and cost or save only parameters.

The learned table of positions is 786,432 parameters that rotary embeddings (Volume V, Chapter 8) replace with arithmetic: queries and keys are rotated after their projections, which is three FLOPs for each of their numbers (four multiplications and two additions for a pair), and nothing is stored. What the change buys beyond the parameters is the property of Chapter 8, that a score depends on the offset between two positions; a rotation has no last row, so it is defined at every position, though how well a model reads positions it was not trained on is the subject of the chapter on long context. RMS normalization (Volume III, Chapter 10) is 19,200 parameters smaller because it has a gain and no shift, and its forward pass is $4D + 3$ FLOPs per row against $7D + 3$ for LayerNorm. SwiGLU needs a hidden width of $\tfrac{2}{3}$ of the old one to hold as many numbers, 2,048 instead of 3,072, because it has three matrices where the GELU network had two; that is why SwiGLU networks have a hidden width near $\tfrac{8}{3} D$ where a GELU network has $4D$ (Qwen3 0.6B uses $3D$, 3,072 at $D = 1{,}024$). The row grows by 12,288 parameters only because the gate and up matrices each have a bias in this count, which the next row removes: 95,232 parameters, less than a tenth of a percent of the model. Dropping biases buys no arithmetic worth the name. It buys a simpler tensor list, as the opening table showed, and a linear map that is nothing but a matrix product.

The fifth change is different in kind. Giving the 12 query heads 4 key-value heads instead of 12 removes 9,437,184 parameters from $\mathbf{W}_K$ and $\mathbf{W}_V$, divides the cache by three, and takes 18,874,368 FLOPs from the 284,812,800 of the step: 6.6 percent. It is also the only change in the table that moves the cache. That is the first of the three sections that follow.

## Many query heads, few key-value heads

Chapter 12 of Volume V ended with a warning. The cache costs $2 L h_{\mathrm{kv}} d_h b$ bytes per token, so at a long context it outgrows the weights of the model, and the number to attack is $h_{\mathrm{kv}}$, the number of key-value heads a layer stores. In ordinary multi-head attention $h_{\mathrm{kv}} = H$: every query head has its own key and value head. Nothing in the algebra of attention demands it. A query head needs some keys to score and some values to average, and two query heads may use the same ones.

**Grouped-query attention** gives a layer $H$ query heads and $h_{\mathrm{kv}}$ key-value heads, with $H$ a multiple of $h_{\mathrm{kv}}$, and lets each key-value head serve a group of $n_{\mathrm{grp}} = H / h_{\mathrm{kv}}$ neighbouring query heads. The projections are $\mathbf{W}_Q$ of shape $[D, H d_h]$ and $\mathbf{W}_O$ of shape $[H d_h, D]$ as before, but $\mathbf{W}_K$ and $\mathbf{W}_V$ are $[D, h_{\mathrm{kv}} d_h]$. Query head $h$ reads key-value head $u(h) = \lfloor h / n_{\mathrm{grp}} \rfloor$:

$$
\mathbf{o}_h = \mathrm{softmax}\!\left(\frac{\mathbf{Q}_h \mathbf{K}_{u(h)}^\top}{\sqrt{d_h}} + \mathbf{M}\right) \mathbf{V}_{u(h)}, \qquad h = 0, \dots, H - 1,
$$

with the causal mask $\mathbf{M}$ of Chapter 7 of Volume V, and the head outputs concatenated and multiplied by $\mathbf{W}_O$ as ever. Multi-head attention is $h_{\mathrm{kv}} = H$, groups of one. The other extreme, $h_{\mathrm{kv}} = 1$, one key-value head read by all the query heads, is multi-query attention in the proposal of Shazeer (2019); Ainslie and colleagues (2023) introduced the groups in between. Chapter 3 of this volume returns to the multi-query end and to compressing the keys and values themselves.

### What the sharing is, shown on a count of four heads

The claim is that this is multi-head attention with some of its heads tied, and a small case makes it checkable. With $H = 4$ query heads and $h_{\mathrm{kv}} = 2$, query heads 0 and 1 read key-value head 0 and heads 2 and 3 read head 1. Run grouped-query attention as written, and run the multi-head attention of Chapter 7 of Volume V on matrices $\mathbf{W}_K$ and $\mathbf{W}_V$ widened to four heads by repeating each key-value head's columns for the query heads of its group. The two must agree.

<!-- worked: modern_decoder.gqa_equals_mha -->
Generator: the book's SplitMix64, seed 0xd17aaf8ef3ca32a2 (`seed_of` of the name `modern_decoder.gqa`). Shapes: T = 3 tokens, D = 8, H = 4 query heads of width 2, 2 key-value heads. Draws, in this order, each as `round(normal(), 1)` and row by row: X [3, 8], W_Q [8, 8], W_K [8, 4], W_V [8, 4], W_O [8, 8]. The mask is causal; scores are divided by sqrt(2).

Which key-value head each query head reads, and the attention weights of the last token over positions 0, 1, 2 (grouped-query attention written directly; multi-head attention of Volume V, Chapter 7 run on W_K and W_V with each key-value head's columns repeated for the query heads of its group):

| query head | key-value head | weights, grouped-query | weights, multi-head with repeated heads |
|---:|---|---|---|
| 0 | 0 | (0.391, 0.031, 0.579) | (0.391, 0.031, 0.579) |
| 1 | 0 | (0.079, 0.815, 0.106) | (0.079, 0.815, 0.106) |
| 2 | 1 | (0.077, 0.614, 0.309) | (0.077, 0.614, 0.309) |
| 3 | 1 | (0.503, 0.135, 0.362) | (0.503, 0.135, 0.362) |

Output rows (one per token, 8 numbers; first two shown):

| token | grouped-query, columns 0-1 | multi-head with repeated heads, columns 0-1 |
|---:|---|---|
| 0 | (-0.102, -0.036) | (-0.102, -0.036) |
| 1 | (0.587, 0.296) | (0.587, 0.296) |
| 2 | (-1.732, -2.726) | (-1.732, -2.726) |

Largest absolute difference over all 24 output numbers and all weights: below 1e-12. With the layout that cycles through the key-value heads instead (query head i reads head i mod 2, neighbours do not share) the multi-head run differs from grouped-query attention by 5.622 at most. Multi-query attention, one key-value head (head 0 of W_K and W_V), equals multi-head attention with that head repeated four times (difference below 1e-12) and differs from the grouped-query output by 7.074 at most.

Numbers cached for one token in one layer (a key and a value row per key-value head):

| attention | key-value heads | numbers cached per token and layer |
|---|---:|---:|
| multi-head, 4 key-value heads | 4 | 16 |
| grouped-query, 2 key-value heads | 2 | 8 |
| multi-query, 1 key-value head | 1 | 4 |
<!-- /worked -->

They agree to round-off, weights and outputs alike, and the table at the bottom shows the saving in one layer for one token: 16 numbers cached for multi-head, 8 for the groups, 4 for multi-query. The layout is not a detail. Heads that are repeated in the other order, so that query head $i$ reads key-value head $i \bmod 2$, give a model of the same shape, load without any error from the same file and produce an output that differs by as much as 5.622 here. A checkpoint trained with one layout and run with the other is silently wrong, which is the failure that parity tests against a reference (the next chapter) exist to catch. In brain the kernels compute the key-value head of query head `h` as `h / group` (`crates/kernels/wgsl/gqa_scores.wgsl`), neighbours sharing, and the expansion kernel `kv_expand` that turns a grouped key-value buffer into a multi-head one writes `src[ho / group]` and documents the backward pass as the adjoint pair, a group-sum (commit `79025ad09`). The reason for a sum is plain from the construction. A key-value head enters the loss through all $n_{\mathrm{grp}}$ query heads that read it, so its gradient is the sum of the gradients that the repeated copies would receive:

$$
\frac{\partial L}{\partial \mathbf{K}_u} = \sum_{h:\, u(h) = u} \frac{\partial L}{\partial \mathbf{K}^{(h)}},
$$

and the same for $\mathbf{V}_u$. A backward pass that keeps only the first query head's contribution is a plausible mistake and a wrong one; a later section shows the gradient check catching it.

### What it saves, counted

The cache is the obvious saving. A token stores $2 L h_{\mathrm{kv}} d_h b$ bytes, so the cache is $h_{\mathrm{kv}} / H = 1 / n_{\mathrm{grp}}$ of the multi-head cache. For the layer counts and head widths of two Qwen3 sizes, with 32,768 tokens of context in bf16:

<!-- worked: modern_decoder.cache_vs_kv_heads -->
Cache bytes per token (bf16) and GiB for a context of 32,768 tokens as the key-value heads of one layer fall from one per query head to one in all, for the layer counts and head widths of Qwen3 0.6B (16 query heads, 28 layers, width 128) and Qwen3 8B (32 query heads, 36 layers, width 128). The row of the published choice, 8 key-value heads, is in both.

| h_kv | 0.6B bytes per token | 0.6B GiB | 8B bytes per token | 8B GiB |
|---|---:|---:|---:|---:|
| 1 | 14,336 | 0.44 | 18,432 | 0.56 |
| 2 | 28,672 | 0.88 | 36,864 | 1.12 |
| 4 | 57,344 | 1.75 | 73,728 | 2.25 |
| 8 | 114,688 | 3.50 | 147,456 | 4.50 |
| 16 | 229,376 | 7.00 | 294,912 | 9.00 |
| 32 |  |  | 589,824 | 18.00 |
<!-- /worked -->

![Cache size in GiB for 32,768 tokens against the number of key-value heads per layer, two straight lines through the origin. For the 8B shape the line rises from 0.56 GiB at 1 key-value head through 4.50 GiB at 8 to 18.00 GiB at 32. For the 0.6B shape it rises from 0.44 GiB through 3.50 GiB at 8 to 7.00 GiB at 16.](img/cache-vs-kv-heads.svg)

The straight line says the whole story: the cache is proportional to $h_{\mathrm{kv}}$, and the published choice of 8 key-value heads is a quarter of the multi-head cache for the 8B shape (4.50 GiB against 18.00) and half for the 0.6B (3.50 against 7.00).

The arithmetic is another matter, and here the answer is that most of it does not change. A decoded token scores every cached key with every one of its $H$ query heads and averages every value with every one of those heads, whatever $h_{\mathrm{kv}}$ is, so the attention products still cost $4 L H d_h$ FLOPs per visible key. Only the projections shrink: $\mathbf{W}_K$ and $\mathbf{W}_V$ are narrower, by $2 \cdot 2 L D (H - h_{\mathrm{kv}}) d_h$ FLOPs per token, which is the 18,874,368 of the ladder. What changes is the ratio of the two prices. For one cached key, a layer reads $2 h_{\mathrm{kv}} d_h b$ bytes and does $4 H d_h$ FLOPs on them, an intensity of $2 n_{\mathrm{grp}} / b$ FLOPs per byte of cache, which is $n_{\mathrm{grp}}$ at $b = 2$. A byte of cache is read once and used by $n_{\mathrm{grp}}$ query heads. Decoding is limited by bytes, not by arithmetic (Volume V, Chapter 12), so this is the right trade: the same arithmetic on fewer bytes.

The family that brain implements shows the choices that were actually made, with the cost of each:

<!-- worked: modern_decoder.family -->
Five decoders from brain's presets (`crates/qwen3/src/config.rs`). Parameters are counted by `parameter_groups`; the embedding share is V x D over the total, the table that an untied head repeats. Cache bytes per token are in bf16 for the key-value heads the model has and for the same model with one key-value head per query head.

| decoder | layers | D | H / h_kv | head width | F | head | parameters | embedding share | cache bytes per token | same with h_kv = H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B | 28 | 1,024 | 16 / 8 | 128 | 3,072 | tied | 596,049,920 | 26.1% | 114,688 | 229,376 |
| Qwen3 1.7B | 28 | 2,048 | 16 / 8 | 128 | 6,144 | tied | 1,720,574,976 | 18.1% | 114,688 | 229,376 |
| Qwen3 4B | 36 | 2,560 | 32 / 8 | 128 | 9,728 | tied | 4,022,468,096 | 9.7% | 147,456 | 589,824 |
| Qwen3 8B | 36 | 4,096 | 32 / 8 | 128 | 12,288 | untied | 8,190,735,360 | 7.6% | 147,456 | 589,824 |
| Llama-2 13B | 40 | 5,120 | 40 / 40 | 128 | 13,824 | untied | 13,015,864,320 | 1.3% | 819,200 | 819,200 |
<!-- /worked -->

All four Qwen3 sizes keep 8 key-value heads, so the group grows with the model from 2 to 4 and the cache per token grows only with the depth: 114,688 bytes at 28 layers and 147,456 at 36, against 229,376 and 589,824 for the same models with a key-value head per query head. The last row is the other end, the Llama-2 13B that brain also carries as a preset: 40 heads and 40 key-value heads, 819,200 bytes per token, 5.6 times the cache per token of the 8B Qwen3. What is paid for the sharing is capacity: $h_{\mathrm{kv}}$ distinct keys and values are stored where $H$ were, so the heads of a group cannot look for different things in the same positions. How much that costs in quality is an empirical matter, which Ainslie and colleagues measured; this chapter counts only what is saved.

## Normalizing queries and keys

A score is $s_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j / \sqrt{d_h}$, with $\mathbf{q}_i = \mathbf{x}_i \mathbf{W}_Q$ and $\mathbf{k}_j = \mathbf{x}_j \mathbf{W}_K$. It is bilinear in the two weight matrices: multiply both by a factor and the score is multiplied by the factor squared. Nothing in the architecture limits that factor. Training is free to grow the weights, and the scaling by $1/\sqrt{d_h}$ of Chapter 7 only fixes the size of the scores at initialization. If the weights grow the scores grow quadratically, the softmax saturates (Chapter 7 showed its rows turning one-hot), and a saturated row passes almost no gradient back to its scores, so nothing in training pulls the scale down again. The next table makes the quadratic growth concrete on one head and six tokens, with the weights multiplied by a factor, and with an RMS normalization of every query and key row in front of the product.

<!-- worked: modern_decoder.logit_growth -->
Generator: the book's SplitMix64, seed 0x0ab25e02c40917d2 (`seed_of` of the name `modern_decoder.logits`). Six tokens (T = 6) of width D = 8, one head of width 8. Draws, in this order: X [6, 8] as `round(normal(), 1)` row by row, then W_Q and W_K [8, 8] as `normal() / sqrt(8)` (unrounded), row by row. The weights of a row are the factor in its first column times these, for queries and keys alike; the mask is causal and the logit of a pair is the query-key dot product divided by sqrt(8). The normalized columns apply the RMS norm of Volume V, Chapter 9 (epsilon 1e-6, gain 1) to every query row and every key row before the dot product. Largest absolute logit over the 21 allowed pairs, and the largest attention weight in the last row (the weight of the sharpest position out of 6; a uniform row has 0.1667):

| factor | largest absolute logit, no norm | largest absolute logit, QK-norm | largest weight, no norm | largest weight, QK-norm |
|---|---:|---:|---:|---:|
| 1 | 1.449 | 2.0975 | 0.3509 | 0.5323 |
| 2 | 5.794 | 2.0975 | 0.8357 | 0.5323 |
| 4 | 23.177 | 2.0975 | 0.9998 | 0.5323 |
| 8 | 92.710 | 2.0975 | 1.0000 | 0.5323 |
| 16 | 370.839 | 2.0975 | 1.0000 | 0.5323 |

The bound for unit gains is sqrt(8) = 2.8284: a normalized row has Euclidean length sqrt(8 x mean square / (mean square + epsilon)), below sqrt(8), so a dot product of two of them is below 8 and the scaled logit below sqrt(8).
<!-- /worked -->

![Largest absolute attention logit against the factor on the weights from 0.5 to 16. Without normalization it rises as a parabola, from 1.4 at factor 1 through 23.2 at factor 4 to 370.8 at factor 16. With an RMS norm of queries and keys it is a flat line near 2.1, below the dashed bound sqrt(8) = 2.83.](img/logit-growth.svg)

Without normalization the logit grows 256-fold from a factor of 1 to a factor of 16, and by a factor of 4 the last row of attention has put 0.9998 of its weight on one position. With normalization the largest logit is 2.0975 at every factor, because normalizing a vector removes its scale: the weights $\mathbf{W}_Q$ and $\mathbf{W}_K$ can no longer move the logit by scaling.

**QK-normalization** is this: every query and every key of every head is RMS-normalized over its own $d_h$ numbers, with a learned gain per channel, before the two meet,

$$
\hat{\mathbf{q}} = \boldsymbol{\gamma}_{\mathrm{q}} \odot \frac{\mathbf{q}}{\sqrt{\frac{1}{d_h}\sum_c q_c^2 + \epsilon}}, \qquad \hat{\mathbf{k}} = \boldsymbol{\gamma}_{\mathrm{k}} \odot \frac{\mathbf{k}}{\sqrt{\frac{1}{d_h}\sum_c k_c^2 + \epsilon}},
$$

and the score is $\hat{\mathbf{q}} \cdot \hat{\mathbf{k}} / \sqrt{d_h}$. The two gains are vectors of $d_h$ numbers shared by all heads of a layer (in brain, `attn.q_norm.weight` and `attn.k_norm.weight`, each of `head_dim` numbers). The bound follows in two lines. With both gains equal to 1, a normalized vector has squared length $d_h \cdot m / (m + \epsilon)$ where $m$ is the mean square of the unnormalized one, which is below $d_h$; by the Cauchy-Schwarz inequality $\lvert \hat{\mathbf{q}} \cdot \hat{\mathbf{k}} \rvert < d_h$, and the scaled score is below $\sqrt{d_h}$, which is $\sqrt{8} = 2.83$ in the table and 11.3 at the head width of 128 that Qwen3 uses. With learned gains the bound is $\max_c \lvert \gamma_{\mathrm{q},c} \rvert \, \max_c \lvert \gamma_{\mathrm{k},c} \rvert \sqrt{d_h}$ (the same argument, on vectors whose entries have been multiplied by the gains): the scale of the logits is now in $2 d_h$ numbers per layer, not in the matrices of $D \cdot H d_h$ and $D \cdot h_{\mathrm{kv}} d_h$ entries. The idea is older than the decoders that now use it: Henry and colleagues (2020) proposed normalizing queries and keys for translation models, and Dehghani and colleagues (2023) used it to stabilize a vision transformer of 22 billion parameters.

### Before the rotation, not after

Qwen3 puts the normalization before the rotary embedding: $\mathbf{q} \to \mathrm{norm} \to \mathrm{gain} \to \mathrm{rotate}$. The order matters when the gains are not all equal, for a reason that the geometry of Chapter 8 makes plain. A rotation turns each pair of channels by an angle that grows with position, and the relative-position property of the score depends on the rotation being exactly that, an orthogonal map of each pair. A gain applied afterwards scales the two channels of a pair by different numbers, and a scaling of the two coordinates of a pair is no longer a rotation. The table checks both orders on one query and one key, with random gains, at three pairs of positions with the same offset:

<!-- worked: modern_decoder.norm_rope_order -->
Generator: the book's SplitMix64, seed 0xf23a4a33f45fe8cd (`seed_of` of the name `modern_decoder.order`). One query and one key of width 8. Draws, in this order: q and k, each as 8 times `round(normal(), 1)`; then the gain vectors for queries and keys, each 8 times `1 + 0.5 * normal()` (unrounded). The rotation is the half-split rotary embedding of Volume V, Chapter 8 with base 100 (rates 1.0000, 0.3162, 0.1000, 0.0316), the norm the RMS norm of Volume V, Chapter 9 with epsilon 1e-6. The score is the dot product of the processed query at position i and key at position j, for three pairs of positions with the same offset 3:

| order | i = 2, j = 5 | i = 10, j = 13 | i = 40, j = 43 | largest difference |
|---|---:|---:|---:|---:|
| norm (with gains), then rotate | 5.433743 | 5.433743 | 5.433743 | below 1e-12 |
| rotate, then norm (with gains) | 2.328619 | 2.558739 | 3.808789 | 1.480 |
| norm with gains all 1, then rotate | 3.891038 | 3.891038 | 3.891038 | below 1e-12 |
| rotate, then norm with gains all 1 | 3.891038 | 3.891038 | 3.891038 | below 1e-12 |

With the gains applied before the rotation the score is the same for every pair of positions with that offset; applied after it, the gains act on rotated channels and the score depends on where the pair sits.
<!-- /worked -->

Normalizing and applying the gain before the rotation gives the same score at all three pairs, to round-off; rotating first and normalizing afterwards makes the score depend on where the pair sits, by as much as 1.480 here. With gains of 1 the order is immaterial, since a rotation does not change the root mean square of a vector, and that is why the property is easy to miss in a test with default gains. The rotation also preserves the length of a vector, so the bound above survives it.

Brain's kernel for this step normalizes a row, applies the gain and rotates the pair, in that order, in a single workgroup per head row (`crates/kernels/wgsl/qknorm_rope_base_fused.wgsl`); fusing the two was done for the serving path in commit `d661ef04c`, whose tests (`qk_norm_rope_fused_is_bit_identical_to_the_unfused_pair` in `crates/qwen3/src/serve.rs`) compare the fused kernel against the separate normalization and rotation bit for bit. The cost of the step is small beside what it protects: $2 d_h$ parameters per layer (7,168 in all for Qwen3 0.6B), and a normalization of $4 d_h + 3$ FLOPs for each of the $H + h_{\mathrm{kv}}$ query and key vectors of a token, which at 24 vectors of width 128 is 12,360 FLOPs per layer, against 31,457,280 for the matrix products of the same layer: about 0.04 percent.

## One table or two

The last of the three changes concerns the ends of the network. The embedding table $\mathbf{W}_{\mathrm{emb}}$ of shape $[V, D]$ turns a token id into a row, and the head turns the final normalized row back into $V$ logits. Chapter 10 of Volume V tied them, after Press and Wolf (2017): the logits of the head are $\mathbf{z} = \mathbf{h} \mathbf{W}_{\mathrm{emb}}^\top$ and there is one table for both. An **untied output head** is the other design, a second matrix $\mathbf{W}_{\mathrm{out}}$ of the same shape $[V, D]$, with its own parameters and its own gradient.

Tying saves $V D$ parameters, and the question is how much that is. The family table above answers it: the embedding is 26.1 percent of Qwen3 0.6B (155,582,464 of 596,049,920 parameters), 18.1 percent of the 1.7B, 9.7 percent of the 4B, 7.6 percent of the 8B, and 31.0 percent of GPT-2 small. Brain's presets tie the table in the first three Qwen3 sizes and untie it in the 8B (`tie_embeddings: false` in `QwenConfig::qwen3_8b`), and an untied head costs the 8B 622,329,856 parameters, which is 1.16 GiB in bf16. The two uses of the table differ in kind. In the input, a token reads one row; in the head, every position multiplies the final row by all $V$ rows, $2 V D$ FLOPs per token, which is 311,164,928 at Qwen3 0.6B and is counted once in every decode step whether the head is tied or not. Tying makes one matrix serve a lookup and a readout. Whether a model does better with two matrices is something only an experiment shows, and the accounting says only that the price of giving each use its own matrix falls as the model grows, from 31 percent of the parameters of GPT-2 small to 7.6 percent of those of the 8B.

Tying has a cost in the engine that does not appear in the parameter count. A tied table is touched at both ends of the network, so its gradient has two sources: the lookup at the bottom adds the gradient of the stream into the rows of the tokens that were read, and the head at the top adds the product of the logit gradient with the normalized stream. When the network is cut into pipeline stages, the first stage holds the lookup and the last stage the head, and a tied table is then a replicated parameter whose gradient is summed across the two. Brain's Qwen3 adapter states exactly that: `replicated_params` in `crates/qwen3/src/shard.rs` returns `tok.weight` for a tied head and nothing for an untied one. An untied head makes every parameter belong to one stage. The gradient check later in this chapter covers both designs, and the tied one has a row of its own for $\mathbf{W}_{\mathrm{emb}}$, whose gradient is the sum of the two contributions.

## The whole recipe in a decoder small enough to check

Each of the three changes was shown on its own. The recipe is all of them at once, together with the pieces of Volume V, and what makes it a recipe and not a list is that the pieces have to compose: the rotation acts on normalized queries and keys, the cache stores them after both, the groups read the cache, and the backward pass of each has to find its way through the others. To check that, the chapter builds the whole decoder in plain Python, reusing the RMS normalization, the SwiGLU network and their backward passes of Volume V (`transformer_block`), the multi-head attention (`self_attention`) and the rotation (`positions`). Its dimensions are chosen so that no two of the widths are equal ($D = 6$, $H d_h = 16$, $h_{\mathrm{kv}} d_h = 8$, $F = 10$), which matters for a reason brain learned the expensive way and recorded as knowledge #4: a toy size where two dimensions coincide lets a transposed or swapped matrix pass every test.

<!-- worked: modern_decoder.tiny_setup -->
Generator: the book's SplitMix64, seed 0xa39c85ac33ea1d52 (`seed_of` of the name `modern_decoder.tiny`). The tiny decoder has V = 11 token ids, width D = 6, L = 2 layers, H = 4 query heads and 2 key-value heads of width 4 (so groups of 2, and the query projection is 16 wide while the width is 6), a SwiGLU network of hidden width F = 10, rotary base 100 on the half-split pairs, RMS norms with epsilon 1e-6, no biases, QK-norm before the rotation and an untied output matrix. Draws, in this order, each matrix row by row: W_emb as `normal()`; then for each layer the gain of norm 1 (each `1 + 0.1 * normal()`), W_Q, W_K, W_V (each `normal() / sqrt(D)`), the QK-norm gains of queries then keys (each `1 + 0.1 * normal()`), W_O (`normal() / sqrt(H x width)`), the gain of norm 2, W_gate and W_up (`normal() / sqrt(D)`), W_down (`normal() / sqrt(F)`); then the last gain and W_out (`normal() / sqrt(D)`). That is 1,114 parameters (1,048 with the head tied to W_emb, which is not drawn again). The text is the ids 3, 9, 1, 7, 4, 4, 10, 2; the loss predicts each id from the ones before it.

| tensor | shape | how often |
|---|---|---|
| token embedding W_emb | [11, 6] | once |
| norm 1, norm 2 gains | [6] | per layer |
| W_Q | [6, 16] | per layer |
| W_K, W_V | [6, 8] | per layer |
| QK-norm gains (queries, keys) | [4] each | per layer |
| W_O | [16, 6] | per layer |
| W_gate, W_up | [6, 10] | per layer |
| W_down | [10, 6] | per layer |
| last gain | [6] | once |
| output matrix W_out | [11, 6] | once, untied |
<!-- /worked -->

The heart of the decoder is the attention of one layer on $n$ new tokens that follow $c$ cached positions. It is the `extend` of Chapter 12 of Volume V with three changes: the key and value rows are $h_{\mathrm{kv}}$ wide, the queries and keys are normalized and rotated before use, and a query head reads the key-value head of its group. In the book's pseudocode, with `rms_norm` of Volume III and `rotary` of Chapter 8 of Volume V (the half-split pairing, `adjacent = false`), and the sizes $H$, $h_{\mathrm{kv}}$, $d_h$ and $T$ those of the model as in Chapter 12:

```text
function modern_attention(A: array<real>[n, D], K: array<real>[T, h_kv, d_h],
                          V: array<real>[T, h_kv, d_h], c: int,
                          Wq: array<real>[D, H * d_h], Wk: array<real>[D, h_kv * d_h],
                          Wv: array<real>[D, h_kv * d_h], Wo: array<real>[H * d_h, D],
                          gq: array<real>[d_h], gk: array<real>[d_h],
                          base: real, eps: real) -> array<real>[n, D]
    # A holds the normalized stream of n new tokens at positions c .. c + n - 1;
    # K and V hold the c earlier positions and receive the new ones
    require H mod h_kv == 0 and c + n <= T
    n_grp = H // h_kv
    Q = matmul(A, Wq)
    Kn = matmul(A, Wk)
    Vn = matmul(A, Wv)
    for i in 0 .. n:
        for u in 0 .. h_kv:
            (k, _, _) = rms_norm(Kn[i:i + 1, u * d_h:(u + 1) * d_h], gk, eps)
            K[c + i, u, :] = rotary(k[0, :], c + i, base, false)     # stored after the norm and the rotation
            V[c + i, u, :] = Vn[i, u * d_h:(u + 1) * d_h]
    O = zeros(real, [n, H * d_h])
    for i in 0 .. n:
        for h in 0 .. H:
            u = h // n_grp                                           # the key-value head of query head h
            (q, _, _) = rms_norm(Q[i:i + 1, h * d_h:(h + 1) * d_h], gq, eps)
            q = rotary(q[0, :], c + i, base, false)
            S = [dot(q, K[j, u, :]) / sqrt(d_h) for j in 0 .. c + i + 1]
            p = softmax(S)
            for j in 0 .. c + i + 1:
                O[i, h * d_h:(h + 1) * d_h] = O[i, h * d_h:(h + 1) * d_h] + p[j] * V[j, u, :]
    return matmul(O, Wo)
```

Around it sit the pre-norm residual block, the SwiGLU network and a final RMS norm, as in Chapter 12, and an output matrix of its own. Run as a forward pass on an empty cache, this routine is the training-time computation; run on a cache and one token, it is a decode step. The test of Chapter 12, that the two agree, now has to hold with groups, normalization and rotation in the way, and it is repeated at three numbers of key-value heads: the multi-head case $h_{\mathrm{kv}} = 4$, the grouped case 2 and the multi-query case 1.

<!-- worked: modern_decoder.decode_equals_forward -->
The decoder of the previous table at three numbers of key-value heads (each drawn with the same generator and the same order, so a different number of draws for W_K and W_V). The 8 tokens are run once as a forward pass with an empty cache; and again as a prefill of the first 3 tokens followed by 5 decode steps of one token each, on the cache the prefill left. Largest absolute difference between the two sets of 8 rows of logits (11 numbers each):

| key-value heads | group size | parameters | numbers in the cache after 8 tokens | largest difference of logits |
|---|---:|---:|---:|---:|
| 4 | 1 | 1,306 | 512 | below 1e-12 |
| 2 | 2 | 1,114 | 256 | below 1e-12 |
| 1 | 4 | 1,018 | 128 | below 1e-12 |
<!-- /worked -->

The cache holds the normalized and rotated keys, which is what makes the later steps cheap and what the test confirms: a key stored before its normalization or rotation would have to be redone on every read. The count of numbers in the cache after 8 tokens is $2 L \cdot 8 \cdot h_{\mathrm{kv}} d_h$, which is 512, 256 and 128: the saving of the previous section, counted in storage that the run actually used.

The backward pass is the larger test. It is written by hand for the whole decoder: the cross-entropy, the head, the last norm, then for each layer in reverse the SwiGLU network, its norm, the output matrix, the attention with its group-sum, the rotation run backwards, the QK-norms, the projections and the first norm. The check of Volume III compares it with central differences, entry by entry, in double precision, against the criterion that every difference stays below $10^{-6}$. It is run for both designs of the head.

<!-- worked: modern_decoder.gradient_check -->
Gradient check of the hand-written backward pass against central differences, entry by entry, with step 1e-05, in double precision: for every entry w of every tensor, (loss(w + h) - loss(w - h)) / 2h. The loss is the mean cross-entropy of the text of the previous table. The column of largest gradients is there to show that no tensor was checked against a gradient of zero.

untied head: loss 3.2691 on the text; 1,114 entries checked, largest difference over all tensors below 1e-6 (criterion: below 1e-6).

| tensor (both layers merged) | entries | largest absolute gradient | largest difference from central difference |
|---|---:|---:|---:|
| W_emb | 66 | 0.091 | below 1e-6 |
| gain 1 | 12 | 0.058 | below 1e-6 |
| W_Q | 192 | 0.078 | below 1e-6 |
| W_K | 96 | 0.102 | below 1e-6 |
| W_V | 96 | 0.129 | below 1e-6 |
| QK-norm gain, queries | 8 | 0.035 | below 1e-6 |
| QK-norm gain, keys | 8 | 0.031 | below 1e-6 |
| W_O | 192 | 0.229 | below 1e-6 |
| gain 2 | 12 | 0.073 | below 1e-6 |
| W_gate | 120 | 0.081 | below 1e-6 |
| W_up | 120 | 0.127 | below 1e-6 |
| W_down | 120 | 0.207 | below 1e-6 |
| last gain | 6 | 0.498 | below 1e-6 |
| W_out | 66 | 0.731 | below 1e-6 |

head tied to W_emb: loss 2.8931 on the text; 1,048 entries checked, largest difference over all tensors below 1e-6 (criterion: below 1e-6).

| tensor (both layers merged) | entries | largest absolute gradient | largest difference from central difference |
|---|---:|---:|---:|
| W_emb | 66 | 0.434 | below 1e-6 |
| gain 1 | 12 | 0.246 | below 1e-6 |
| W_Q | 192 | 0.246 | below 1e-6 |
| W_K | 96 | 0.236 | below 1e-6 |
| W_V | 96 | 0.369 | below 1e-6 |
| QK-norm gain, queries | 8 | 0.098 | below 1e-6 |
| QK-norm gain, keys | 8 | 0.105 | below 1e-6 |
| W_O | 192 | 0.600 | below 1e-6 |
| gain 2 | 12 | 0.341 | below 1e-6 |
| W_gate | 120 | 0.295 | below 1e-6 |
| W_up | 120 | 0.266 | below 1e-6 |
| W_down | 120 | 0.540 | below 1e-6 |
| last gain | 6 | 0.687 | below 1e-6 |
<!-- /worked -->

Every tensor passes, in both designs. The column of largest gradients is the guard of knowledge #40, which records a check that passed while a whole family of parameters reported a numeric gradient of exactly zero, because their effect on the loss had been numerically starved: a tensor compared against a gradient of nothing proves nothing, so the table shows that each one carries gradient. In the tied design the table $\mathbf{W}_{\mathrm{emb}}$ is differentiated through the lookup and through the head at once. A passing check proves little if it cannot fail, so the same check is run on backward passes with one deliberate mistake each: the three that this chapter's changes invite, and, for the tied head, a table that keeps the gradient of its lookup and loses that of the head.

<!-- worked: modern_decoder.wrong_backward_passes -->
The same check on backward passes with one deliberate mistake each, on the weights and the text of the previous tables. A check passes when every tensor is below the criterion of 1e-6; the table lists how many kinds of tensor fail (a kind merges the two layers) and the one that fails most.

| backward pass | head | kinds of tensor failing | worst tensor | largest difference |
|---|---|---:|---|---|
| the correct backward pass | untied | 0 of 14 | none | all below 1e-6 |
| a key-value head takes the gradient of the first query head of its group only | untied | 12 of 14 | gain 1 | 0.103 |
| the rotation of the gradient turns by +angle instead of -angle | untied | 12 of 14 | W_Q | 0.100 |
| the QK-norm's scale factor is treated as a constant | untied | 12 of 14 | W_K | 0.047 |
| the correct backward pass | tied | 0 of 13 | none | all below 1e-6 |
| the tied table receives the gradient of the lookup but not that of the head | tied | 1 of 13 | W_emb | 0.243 |
<!-- /worked -->

The correct pass has no failing tensor, and each of the three mistakes fails twelve of the fourteen kinds of tensor, with largest differences of 0.103, 0.100 and 0.047, more than four orders of magnitude over the criterion. The mistakes sit in the places where the recipe composes: the group-sum, the sign of the rotation's angle and the scale factor of the QK-norm. The two tensors that do not fail are the output matrix and the last gain, which sit after the mistake: their gradients are computed before the wrong step is reached. The tied table loses 0.243 of gradient when it forgets the head, and no other tensor notices, which is why the check has a row of its own for it.

## The recipe as configuration in brain's Qwen3

Brain carries the recipe in one record, `QwenConfig` in `crates/qwen3/src/config.rs`, and every choice of the opening table is a field of it. For the 0.6B preset:

| field | value | choice of the recipe |
|---|---|---|
| `vocab` | 151,936 | token table `tok.weight`, $V \times D$ |
| `d_model`, `n_layers` | 1,024 and 28 | width and depth |
| `n_heads`, `n_kv_heads` | 16 and 8 | query heads and key-value heads, `group()` = 2 |
| `head_dim` | 128 | decoupled from the width: `q_dim()` = 2,048 is not `d_model` |
| `d_ff` | 3,072 | hidden width of the SwiGLU network |
| `rope_theta` | $10^6$ | base of the half-split rotary embedding |
| `rms_eps` | $10^{-6}$ | floor of every RMS norm, QK-norms included |
| `qk_norm` | true | per-head normalization before the rotation |
| `attn_bias` | false | no bias on the query, key and value maps |
| `tie_embeddings` | true (false at 8B) | `head_weight()` returns `tok.weight` or `lm_head.weight` |
| `rope_scaling` | none | plain rotary embedding; the chapter on long context returns to the rest |

The module comment of `config.rs` reads like the opening of this chapter: RMSNorm without a bias, rotary embeddings with a base of $10^6$ in the half-split pairing, grouped-query attention, a per-head QK-norm applied to `q` and `k` before rotary embeddings, a SwiGLU network and a tied head, with no attention or MLP biases, and a `head_dim` decoupled from `d_model`. The shapes of the presets are those of the published checkpoints' `config.json` files (Qwen Team, 2025), as their comments say, except for the context: a preset sets `max_position_embeddings` to the 1,024 positions it is built for, where the published 0.6B declares 40,960, which a test in `crates/qwen3/src/hf.rs` reads from a verbatim copy of that file. The decoupled head width matters for shapes: sixteen heads of 128 are 2,048 channels in a model whose width is 1,024, so $\mathbf{W}_Q$ expands and $\mathbf{W}_O$ contracts, which is why the tiny decoder above has $H d_h = 16$ with $D = 6$. The parameter list `param_list` is the list of the opening table in code: for each layer `ln1.weight`, the three projections, `attn.q_norm.weight` and `attn.k_norm.weight` when `qk_norm` is set, `attn.wo.weight`, `ln2.weight` and the three `mlp` matrices; then `norm.weight`, and `lm_head.weight` only when the head is untied. Initialization is the rule of Chapter 10 of Volume V: gains 1, every matrix a normal with standard deviation 0.02, the two matrices that write into the residual stream scaled by a further $1/\sqrt{2L}$ (`crates/qwen3/src/init.rs`).

The forward pass of one layer follows the order the module header of `crates/qwen3/src/model.rs` gives: normalize, project to $\mathbf{q}$, $\mathbf{k}$ and $\mathbf{v}$, normalize $\mathbf{q}$ and $\mathbf{k}$ per head with the same row-normalization kernel as the block's own norms (here over `head_dim` instead of `d_model`), rotate, attend with the groups, project out, add, normalize, SwiGLU, add. The KV-cache decode of the same model, `Qwen::step`, projects the new token's key and value, normalizes and rotates the key at its absolute position, appends both to a per-layer buffer of `kv_dim()` columns and attends with a single query. The plain cache is counted in `crates/qwen3/src/footprint.rs` as $2 \cdot L \cdot t \cdot$ `kv_dim()` $\cdot 4$ bytes, since it holds 32-bit numbers: for Qwen3 0.6B 229,376 bytes per token, the same number as the bf16 cache of the multi-head version of the same model in the family table, by coincidence of a factor of two in each direction.

How is it tested? Three tests stand behind the choices. The decode test `kv_step_matches_full_recompute` feeds a random token sequence through the cache one token at a time and compares each step's logits with a full recompute of the prefix, to a tolerance of $2 \times 10^{-3}$ in 32-bit arithmetic; it is the test of the previous section on the real engine. The gradient check `check_qwen` in `crates/gradcheck/src/lib.rs` differentiates a tiny Qwen3 (4 query and 2 key-value heads, a decoupled head width, QK-norm, rotary base, SwiGLU and a tied head) and its doc comment names the same list of things that the table above checked, with the group accumulation and the tied gradient among them. And a forward-parity test against a reference implementation (`crates/qwen3/tests/it/parity.rs`) requires, when it is pointed at a real checkpoint and its reference logits, that the top prediction agree at every position and that the largest logit error stay below 1.0. That bound is loose; the message of the commit that introduced the first Qwen3 crate (`2a79b9ecb`) reports an error near $2 \times 10^{-4}$ against the reference, a number this chapter could not reproduce because the test needs a downloaded checkpoint. Making parity a gate with a tolerance one can defend is the subject of the next chapter.

### What the history of the configuration shows

Two stories in brain's history concern this one record and the checks that guard it.

The first Qwen3 crate (commit `2a79b9ecb`, June 2026) had QK-norm built in, and a `tie_embeddings` switch for the head. Qwen2 was then added as a variant of the same decoder: commit `af6cb4cd3` (July 2026) gave `QwenConfig` the two fields `qk_norm` and `attn_bias`, which are the differences between a Qwen3 and a Qwen2 (QK-norm off, bias on the query, key and value maps), with the parameter list and the backward pass made conditional on them, and a Llama preset followed. Variants multiplied, and in September commit `8dedcb408` found that the switches were not written down: `to_json` never wrote `qk_norm` or `attention_bias`, and the reader defaults both to the values of Qwen3, so a Llama decoder saved as a brain checkpoint reloaded with a QK-norm it never had, and a Qwen2 lost its biases. The fix is the one the code now shows: `to_json` writes both, the test `json_roundtrip_is_identity` in `crates/qwen3/src/config.rs` pins that reading what was written gives back the same config for every preset, and loaders call `from_json_checked`, which refuses a config that lacks a shape key instead of filling in a default. The general lesson is recorded as knowledge #176: a default in a config reader is a guess about which model wrote the file, and an architecture switch is part of the model, not of the loader.

The second story is a gradient check that did not fail. Commit `04023a6ff` (September 2026) added finite-difference checks for the variants the decoder serves: Llama with full multi-head attention and with grouped-query attention, both untied and without QK-norm, and Qwen2 with biases, plus element-by-element checks of the biases, the norm gains, an untied head and a tied table. Its message states that nothing in the decoder needed fixing. One check had missed at first, on the LoRA twin of the Qwen2 variant: the difference was 0.150 at a step of $5 \times 10^{-3}$, 0.039 at $2.5 \times 10^{-3}$ and 0.010 at $1.25 \times 10^{-3}$, with the analytic value unchanged. A difference that falls by about four when the step is halved is truncation error of the central difference on a sharply curved loss, which goes as the square of the step. A wrong gradient does not move with the step. That is how to read a failing check before concluding the backward pass is wrong, and the reason the check of this chapter reports its step (here $10^{-5}$, in double precision) with its criterion. The brain gate, run in 32-bit arithmetic, is the looser pair $(4 \times 10^{-3}, 8 \times 10^{-2})$ of Volume III, Chapter 6, and the stricter $10^{-6}$ is a property of the double-precision reference.

## Check your understanding

1. A colleague says that grouped-query attention with 8 key-value heads for 16 query heads halves the cost of attention. Which of the three prices (bytes of cache, FLOPs of the attention products, FLOPs of the key and value projections) is that right for, and which is it not?
2. A paper states that QK-normalization bounds attention logits by $\sqrt{d_h}$. Under what condition is that true, and what can training still do to the logits?
3. A new implementation of rotary embeddings with QK-norm passes a test whose gains are all 1 and disagrees with a real checkpoint. What is the likeliest error, and why did the test not see it?
4. The weights of a grouped-query checkpoint are loaded with the key-value heads in a cycling order instead of the grouped one. Every shape matches and the file loads. What does the output look like, and what kind of check finds it?
5. "Tying saves a third of the parameters, so every model should tie." Find the error.
6. A gradient check reports a difference below the criterion for every entry of a tensor, and the analytic gradient of that tensor is zero at every entry. Is the backward pass of that tensor correct?
7. A gradient check fails with a difference of 0.04 at a step of $10^{-3}$, and with 0.01 at a step of $5 \times 10^{-4}$. Is the backward pass wrong?

## Summary

A modern open-weight decoder is the decoder of Volume V with seven choices changed, and the changes divide into two kinds. Four change what the block is made of: rotary positions in place of the position table, RMS norms in place of LayerNorm, a SwiGLU network of three matrices in place of two, and no biases, which together take a net 888,576 parameters from GPT-2 small and change neither the cache nor the FLOPs of the matrix products. Three change the economics. Grouped-query attention shares key-value heads among groups of query heads, so the cache is $h_{\mathrm{kv}} / H$ of the multi-head cache and each cached byte is used by $H / h_{\mathrm{kv}}$ query heads, while the attention FLOPs stay as they were. QK-normalization bounds the attention logits by $\sqrt{d_h}$ (times the gains), at a cost near 0.04 percent of the layer's FLOPs, and has to act before the rotation to leave the relative-position property intact. An untied head gives the output its own $V D$ parameters, a price that is 31 percent of GPT-2 small and 7.6 percent of Qwen3 8B. All of them were assembled into one decoder, checked as a forward pass, as a cached decode and against a hand-written backward pass, and found in brain as the fields of one config record whose switches have to be saved with the model.

The limits are those of counting. The ladder says what each change costs and saves; whether the model is better for it is shown only by training, and this chapter trained nothing. The cache is the bill that grouped key-value heads attack, and they shrink it only by a factor of $n_{\mathrm{grp}}$. The next chapter turns to the practical question behind every checkpoint of this family, how to load weights trained elsewhere into this recipe and prove that the result computes the same function; the chapter after it asks how much further the cache can be shrunk by sharing and compressing the keys themselves.

## References

- J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebron and S. Sanghai, "GQA: Training generalized multi-query transformer models from multi-head checkpoints", 2023.
- N. Shazeer, "Fast transformer decoding: one write-head is all you need", 2019.
- A. Henry, P. R. Dachapally, S. Pawar and Y. Chen, "Query-key normalization for transformers", 2020.
- M. Dehghani and colleagues, "Scaling vision transformers to 22 billion parameters", 2023.
- H. Touvron and colleagues, "LLaMA: open and efficient foundation language models", 2023.
- O. Press and L. Wolf, "Using the output embedding to improve language models", 2017.
- Qwen Team, "Qwen3 technical report", 2025.

## Exercises

1. **[Calculate]** A decoder has 24 layers, 20 query heads and 4 key-value heads of width 64, and stores its cache in bf16. Give the group size, the bytes of cache per token, and the bytes for a context of 16,384 tokens. How many bytes would it need with one key-value head per query head?
2. **[Calculate]** For the decoder of Exercise 1, give the FLOPs per decoded token at a cache of $c$ tokens that depend on $h_{\mathrm{kv}}$, and the ones that do not. At $D = 1{,}280$ what do the key and value projections of one layer cost per token, and what would they cost with 20 key-value heads?
3. **[Calculate]** A head of width $d_h = 64$ has queries and keys that are normalized with gains of 1. Give the bound on the absolute value of a score. What is the bound if the largest absolute gain is 3 for the queries and 3 for the keys?
4. **[Calculate]** Using the formulas of the chapter, compute the parameters of a decoder with $V = 32{,}000$, $D = 2{,}048$, $L = 22$, $H = 32$, $h_{\mathrm{kv}} = 4$, $d_h = 64$ and a SwiGLU network of hidden width 5,632, with no biases, no QK-norm and one RMS-norm gain per norm (two per layer and a last one), first with a tied head and then with an untied one. What share of the total is the embedding table in each case?
5. **[Calculate]** A norm applied after the rotation scales the two channels of a pair by $a$ and $b$. For which values of $a$ and $b$ is the result still a rotation of the pair, and what does that say about gains that are learned per channel?
6. **[Calculate]** Show that the gradient of $\mathbf{K}_u$ in grouped-query attention is the sum of the gradients of the repeated copies, starting from $\mathbf{K}^{(h)} = \mathbf{K}_{u(h)}$ for every query head $h$ and the chain rule.
7. **[Implement]** Reproduce `modern_decoder.gqa_equals_mha`: grouped-query attention written directly with $H = 4$, $h_{\mathrm{kv}} = 2$, $d_h = 2$, and multi-head attention on matrices with each key-value head repeated for its group, from the seed, the draw order and the rounding printed in the text; then the layout that cycles through the key-value heads, and multi-query attention.
8. **[Implement]** Reproduce `modern_decoder.logit_growth` and `modern_decoder.norm_rope_order`: the largest logit with and without the normalization at the five factors, and the three scores in the four orders of norm and rotation.
9. **[Implement]** Reproduce `modern_decoder.decode_equals_forward`: the tiny decoder of `modern_decoder.tiny_setup` from its seed and draw order, its forward pass, and the same eight tokens as a prefill of three and five decode steps on a cache, at $h_{\mathrm{kv}} = 4$, 2 and 1.
10. **[Implement]** Reproduce `modern_decoder.gradient_check` and `modern_decoder.wrong_backward_passes`: write the backward pass of the tiny decoder by hand, check it entry by entry against central differences with step $10^{-5}$, for the untied and the tied head, and then the four wrong variants.
11. **[Investigate]** The chapter counted the effect of each change and measured nothing about quality. Train the tiny decoder of the chapter on a task of your choosing (for instance the lookup task of Chapter 7 of Volume V) with $h_{\mathrm{kv}} = 4$, 2 and 1 and compare the final losses. What does a task this small show about the cost of sharing, and what does it not?
12. **[Investigate]** Repeat `modern_decoder.logit_growth` with gains in the normalization that are learned in the sense of being increased by the same factor as the weights. At which factor does the bound of the chapter stop being informative, and what does that say about where the stability comes from?

## Key terms

- **grouped-query attention**: attention in which each key-value head is read by a group of neighbouring query heads, so a layer stores $h_{\mathrm{kv}}$ key-value heads for $H$ query heads, the cache is $h_{\mathrm{kv}} / H$ of the multi-head cache, and the attention FLOPs are unchanged. <!-- concept: grouped-query-attention -->
- **QK-normalization**: the RMS normalization of every query and every key over its own head width, with a learned gain per channel and applied before the rotary rotation, which bounds attention logits by $\sqrt{d_h}$ times the gains. <!-- concept: qk-normalization -->
- **untied output head**: an output matrix $\mathbf{W}_{\mathrm{out}}$ of shape $[V, D]$ that is separate from the embedding table, so each has its own parameters and its own gradient. <!-- concept: untied-output-head -->
- **modern decoder recipe**: the combination of choices common to present-day open-weight decoders: pre-norm blocks with RMS normalization and no biases, rotary positions applied to normalized queries and keys, grouped key-value heads, a SwiGLU network, and an output matrix that is tied or untied by size. <!-- concept: modern-decoder-recipe -->

## Answers

**Recall questions.**

1. A token produces 12 key vectors of 64 numbers each, 768 numbers in all, as many as the width of the model.
2. After the projections, on queries and keys only, never on values. The score of a pair of positions then depends only on the offset between them, $j - i$.
3. $y_d = \gamma_d x_d / \sqrt{\frac{1}{D}\sum_k x_k^2 + \epsilon}$. It has $D$ learned numbers, the gain; LayerNorm has $2D$, a gain and a shift.
4. $\mathbf{Y} = \bigl(\mathrm{silu}(\mathbf{X}\mathbf{W}_{\mathrm{gate}}) \odot \mathbf{X}\mathbf{W}_{\mathrm{up}}\bigr)\mathbf{W}_{\mathrm{down}}$, with $\mathbf{W}_{\mathrm{gate}}$ and $\mathbf{W}_{\mathrm{up}}$ of shape $[D, F]$ and $\mathbf{W}_{\mathrm{down}}$ of shape $[F, D]$. At $D = 768$ and $F = 2{,}048$: $3 \times 768 \times 2{,}048 = 4{,}718{,}592$, the same as the two matrices $2 \times 768 \times 3{,}072$ of the GELU network.
5. $2 L h_{\mathrm{kv}} d_h b = 2 \times 12 \times 12 \times 64 \times 2 = 36{,}864$ bytes.
6. It compares the gradient of the backward pass with $(L(w + h) - L(w - h)) / 2h$, computed by running the forward pass twice with the entry nudged. The central difference is used because its truncation error goes as $h^2$, where that of the one-sided difference goes as $h$.

**Check your understanding.**

1. At 8 key-value heads and 16 query heads, the group size is 2. The cache is half of the multi-head cache (114,688 against 229,376 bytes per token at 28 layers), but the FLOPs of the attention products, $4 L H d_h$ per visible key, are the same, since every query head still scores every key. The projections of $\mathbf{W}_K$ and $\mathbf{W}_V$ are half as wide.
2. The bound is $\sqrt{d_h}$ for gains of 1, and gains larger than 1 raise it by their product. Training can still increase the logits by increasing the gains, which enter the logit as a product like the matrices did. What the normalization changes is where the scale lives: no longer in the $D \cdot H d_h$ and $D \cdot h_{\mathrm{kv}} d_h$ entries of the projections, which it cancels, but in $2 d_h$ numbers per layer. A claim of "bounded by $\sqrt{d_h}$" is a claim about unit gains.
3. The order of norm and rotation. If the gain is applied after the rotation the score is no longer a function of the offset alone. The test passes with all gains at 1, since the two orders agree there, and a model trained with such a test would carry the error into real checkpoints, where the gains are not 1.
4. It computes the right shapes and loads without error, and every shape agrees ($H d_h$ is the same however the key-value heads are numbered). Only the pairing of query heads with key-value heads differs, so the output is wrong by as much as the table's 5.622 and no shape or load check can see it. A reference output, compared to a tolerance, can.
5. The embedding is a smaller share of the larger model (26.1 percent at 0.6B, 7.6 percent at 8B), so the same $V D$ parameters are a smaller price. The saving is $V D$ parameters in either case: 155,582,464 at 0.6B and 622,329,856 at 8B. The account is not a measurement of quality, and a model may do better with the saving spent elsewhere.
6. Not established. A gradient check that passes with every gradient equal to zero proves nothing: the true gradient could be zero as well, or the tensor could be disconnected from the loss. The column of largest gradients shows that each tensor was compared with a nonzero gradient, as knowledge #40 requires.
7. It does not follow. A difference that falls by four when the step is halved is the truncation error of the central difference, which goes as $h^2$ (commit `04023a6ff`: 0.150, 0.039, 0.010). A wrong gradient stays where it is as the step shrinks. The check to run is the one with a smaller step.

**Exercises.**

1. The group size is $20 / 4 = 5$. Bytes per token: $2 \times 24 \times 4 \times 64 \times 2 = 24{,}576$. For 16,384 tokens, $24{,}576 \times 16{,}384 = 402{,}653{,}184$ bytes, which is 0.375 GiB. With 20 key-value heads the cache is five times as big: 122,880 bytes per token and 1.875 GiB.
2. The attention products cost $4 L H d_h = 4 \times 24 \times 20 \times 64 = 122{,}880$ FLOPs per visible key and do not depend on $h_{\mathrm{kv}}$; the key and value projections do, $4 L D h_{\mathrm{kv}} d_h = 31{,}457{,}280$ FLOPs per token at $D = 1{,}280$. At $D = 1{,}280$ the key and value projections of one layer cost $2 \times 2 \times 1{,}280 \times (4 \times 64) = 1{,}310{,}720$ FLOPs per token, and with 20 key-value heads $2 \times 2 \times 1{,}280 \times (20 \times 64) = 6{,}553{,}600$, five times as many.
3. $\sqrt{64} = 8$. With the largest gains 3 and 3 the bound is $3 \times 3 \times 8 = 72$.
4. Per layer: attention $D(Hd_h) + 2D(h_{\mathrm{kv}}d_h) + (Hd_h)D = 2{,}048 \times 2{,}048 \times 2 + 2 \times 2{,}048 \times 256 = 8{,}388{,}608 + 1{,}048{,}576 = 9{,}437{,}184$; feed-forward $3 \times 2{,}048 \times 5{,}632 = 34{,}603{,}008$; total per layer $44{,}040{,}192$, over 22 layers $968{,}884{,}224$. Embedding $32{,}000 \times 2{,}048 = 65{,}536{,}000$. Norms: $(2 \times 22 + 1) \times 2{,}048 = 92{,}160$. Tied: $968{,}884{,}224 + 65{,}536{,}000 + 92{,}160 = 1{,}034{,}512{,}384$, embedding share 6.3 percent. Untied: $1{,}100{,}048{,}384$, and each of the two tables is 6.0 percent.
5. It is a rotation only if $a = b$ (and then it is a rotation combined with a scaling by $a$); with $a \neq b$ it stretches one axis of the pair and the angle of the vector changes. A gain learned per channel will in general differ between the two channels of a pair, so it has to be applied before the rotation, where it acts on the pair's content and the rotation then turns the result.
6. $\mathbf{K}^{(h)} = \mathbf{K}_{u(h)}$ for every $h$, so $L$ depends on $\mathbf{K}_u$ through all copies with $u(h) = u$. By the chain rule $\partial L / \partial \mathbf{K}_u = \sum_h (\partial L / \partial \mathbf{K}^{(h)})(\partial \mathbf{K}^{(h)} / \partial \mathbf{K}_u)$ and the second factor is the identity if $u(h) = u$ and zero otherwise, which leaves the sum over the group.
7. The table has the attention weights of the last token for each of the four heads in both computations (equal to 3 decimals), the outputs equal below $10^{-12}$, the cycled layout differing by 5.622 and multi-query attention differing from the grouped output by 7.074.
8. The largest logits without normalization are 1.449, 5.794, 23.177, 92.710 and 370.839 at factors 1, 2, 4, 8 and 16, and 2.0975 with it at every factor; the scores of the four orders are in the table (5.433743 at all three pairs for norm with gains then rotate, and 2.328619, 2.558739 and 3.808789 for rotate then norm).
9. The tiny decoder has 1,306, 1,114 and 1,018 parameters at $h_{\mathrm{kv}} = 4$, 2 and 1, and caches of 512, 256 and 128 numbers after 8 tokens; the cached and the uncached logits agree below $10^{-12}$.
10. The largest difference between the analytic gradient and the central difference stays below $10^{-6}$ for every tensor of both designs (1,114 and 1,048 entries); each of the three wrong variants fails twelve of the fourteen kinds of tensor with largest differences of 0.103, 0.100 and 0.047, and the tied table that forgets the head fails one of thirteen (W_emb, 0.243).
11. An investigation. The chapter trained nothing. On a task as small as the lookup of Chapter 7 of Volume V a single head suffices, so the three runs are expected to look alike; that shows the sharing does not break learning, and says nothing about the quality cost at scale, which depends on tasks that need several different retrievals from the same positions.
12. An investigation. The chapter did not run it. Expected: the bound $\max\lvert\gamma_{\mathrm{q}}\gamma_{\mathrm{k}}\rvert\sqrt{d_h}$ grows as the square of the gain factor, so once the gains scale with the weights the normalization has moved the quadratic growth from the matrices into the gains and bounds nothing; the stability of QK-norm rests on the gains staying small, which is a property of training and not of the architecture.
