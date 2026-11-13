<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Vision-Language Models

*How do we give a language model eyes?*

Every decoder of Volumes V and VI has exactly one door. A token id comes in, the embedding table turns it into a row of $D$ numbers, and from there on the model sees rows and nothing else. A 336 by 336 colour picture is 338,688 numbers and has no id. Two ways round the door suggest themselves. The first is to describe the picture in words with another model and hand the description to the decoder, which works until someone asks about a detail the description left out: the decoder cannot look again. The second is to go through the wall. Skip the table at some positions and put rows there that were computed from the picture.

The second way is open because of a fact about the decoder that Volumes V and VI established without needing it. By the time attention reads a row, the row carries no label of where it came from. The table does not stamp its rows, and position is not added to them either: a rotary decoder applies it inside attention, by turning queries and keys. A row computed from a picture, of the right width, is a row. What remains is a list of engineering questions. How are the picture's rows made to have the decoder's width (the projector), and put in the right places of the prompt (the splice)? How many rows does a picture cost, and how can the number follow the picture instead of being fixed (the budget, and a grid that follows the picture)? What do the rows' positions say about where they sit in the picture? And what changes when the answer is not a sentence but a box?

A vision-language model joins parts built earlier, and six of them are recalled first.

1. ([Volume V, Chapter 12](../v05-sequences/12-kv-cache.md), the cache.) A decoder has 36 layers and 8 key-value heads of width 128 and stores bf16 numbers. How many bytes does one token occupy in its cache?
2. ([Chapter 6](06-vision-transformers.md), patches.) A 448 by 448 picture is cut into patches of 16 pixels. How many tokens does the tower attend over, and how many does it hand on if each 2 by 2 block of patches is merged into one?
3. ([Volume V, Chapter 8](../v05-sequences/08-positions.md), multi-axis rotary embedding.) The counter stands at $c = 5$ before a picture of one frame, 2 rows and 3 columns. What triple does the patch in row 1, column 2 get?
4. ([Chapter 8](08-clip.md), contrastive pretraining.) On which vector does the loss of a contrastive tower act, and what does it say about the other rows the tower computes?
5. ([Volume V, Chapter 2](../v05-sequences/02-tokenization.md), special tokens.) What makes a special token different from a string of ordinary characters that happens to spell the same thing?
6. ([Volume V, Chapter 8](../v05-sequences/08-positions.md) and [Volume VI, Chapter 1](../v06-language-models/01-modern-decoder.md).) Does a rotary decoder add position to its embedding rows? And if it has 32 query heads over 8 key-value heads, how many query heads share one key-value head?

## A prompt with a hole in it

The smallest possible case shows the whole mechanism, and every number in it can be checked by hand. The decoder has a vocabulary of eight tokens and a width of three. The picture has sixteen pixels. The tower is reduced to its patch embedding, the first step of Chapter 6, because the blocks that follow would only make the numbers longer; the projector is one linear map.

<!-- worked: vlms.splice_setup -->
A decoder with the vocabulary of eight tokens below and width D = 3. Its embedding table E has one row of three numbers for each token.

| id | token | row of E |
|---:|---|---|
| 0 | <bos> | (0.0, 0.1, 0.0) |
| 1 | <image> | (0.5, 0.5, 0.5) |
| 2 | what | (1.0, 0.0, 0.5) |
| 3 | is | (0.0, 1.0, 0.5) |
| 4 | this | (0.5, 0.5, 1.0) |
| 5 | ? | (0.2, 0.2, 0.2) |
| 6 | cat | (1.0, 1.0, 0.0) |
| 7 | <eos> | (0.0, 0.0, 0.1) |

The picture is 4 by 4 pixels, one channel, brightness 0 to 1.

|  | column 0 | column 1 | column 2 | column 3 |
|---|---:|---:|---:|---:|
| row 0 | 0.0 | 0.0 | 1.0 | 1.0 |
| row 1 | 0.0 | 0.5 | 1.0 | 0.5 |
| row 2 | 1.0 | 0.0 | 0.0 | 0.5 |
| row 3 | 1.0 | 1.0 | 0.5 | 1.0 |

It is cut into four patches of 2 by 2 pixels, each flattened row by row. The tower is reduced to its patch embedding (the blocks of Chapter 6 are left out so that the numbers stay printable): a 2 by 4 matrix W_v = [[0.5, 0.5, 0.5, 0.5], [1.0, -1.0, 1.0, -1.0]] with no bias gives each patch a vector of width D_v = 2. The projector is one linear map from 2 to 3 numbers, W_p = [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]] and b_p = [0.1, -0.1, 0.0], so that p = W_p f + b_p.

| patch | flattened pixels | tower output f | projector output p |
|---|---|---|---|
| patch 0 (rows 0-1, columns 0-1) | (0.0, 0.0, 0.0, 0.5) | (0.250, -0.500) | (0.600, -1.100, -0.250) |
| patch 1 (rows 0-1, columns 2-3) | (1.0, 1.0, 1.0, 0.5) | (1.750, 0.500) | (3.600, 0.900, 2.250) |
| patch 2 (rows 2-3, columns 0-1) | (1.0, 0.0, 1.0, 1.0) | (1.500, 1.000) | (3.100, 1.900, 2.500) |
| patch 3 (rows 2-3, columns 2-3) | (0.0, 0.5, 0.5, 1.0) | (1.000, -1.000) | (2.100, -2.100, 0.000) |
<!-- /worked -->

The user wrote `<image> what is this ?`. The tokenizer gives five ids, with a `<bos>` in front of them. But the picture is not going to be one row: it is four, one per patch, and the decoder's sequence has to have room for them before the computation starts. So the single `<image>` is **expanded** to as many copies of its id as there will be rows, and the sequence is built as if the picture were four words.

<!-- worked: vlms.splice_sequence -->
The prompt `<image> what is this ?` is tokenized to ids (1, 2, 3, 4, 5), a `<bos>` (id 0) goes in front, and the single `<image>` is expanded to four copies, one per patch. The embedding gather reads row 1 of E four times; the splice then overwrites those four rows with the four projector outputs, in order. The decoder reads the right-hand column.

| position | id | token | where the row comes from | row after the splice |
|---:|---:|---|---|---:|
| 0 | 0 | <bos> | table row 0 | (0.000, 0.100, 0.000) |
| 1 | 1 | <image> | projector row 0 | (0.600, -1.100, -0.250) |
| 2 | 1 | <image> | projector row 1 | (3.600, 0.900, 2.250) |
| 3 | 1 | <image> | projector row 2 | (3.100, 1.900, 2.500) |
| 4 | 1 | <image> | projector row 3 | (2.100, -2.100, 0.000) |
| 5 | 2 | what | table row 2 | (1.000, 0.000, 0.500) |
| 6 | 3 | is | table row 3 | (0.000, 1.000, 0.500) |
| 7 | 4 | this | table row 4 | (0.500, 0.500, 1.000) |
| 8 | 5 | ? | table row 5 | (0.200, 0.200, 0.200) |

The sequence has 9 rows: 5 from the table and 4 from the picture. The values `(0.5, 0.5, 0.5)` of E's row 1 are gathered and then overwritten, so nothing in the output depends on them.
<!-- /worked -->

That is **image token splicing**: the embedding rows at the positions of the placeholder ids are replaced by the rows the projector made from the picture, in order, and the decoder reads the result exactly as it would a sequence of words. As an algorithm it is a gather in which one id is special:

```text
function splice(ids: array<int>[T], E: array<real>[V, D], image_rows: array<real>[N, D],
                image_id: int) -> array<real>[T, D]:
    count = 0
    for t in 0 .. T:
        if ids[t] == image_id:
            count = count + 1
    require count == N, "one picture row per placeholder"
    X = zeros(real, [T, D])
    k = 0
    for t in 0 .. T:
        if ids[t] == image_id:
            X[t, :] = image_rows[k, :]            # the k-th picture row, in order
            k = k + 1
        else:
            X[t, :] = E[ids[t], :]
    return X
```

![The flow of the prompt of the example. The text goes through the tokenizer, which expands the placeholder to four ids, and the embedding gather, which gives 9 rows of width 3. The picture goes through patchify and the tower, giving 4 patches of width 2, and the projector, giving 4 rows of width 3. Both streams meet in the splice, which overwrites rows 1 to 4 with the projector rows, and the 9 rows go to the decoder. A dashed arrow returns the gradient of the picture rows to the projector.](img/splice-flow.svg)

The row of the table that belongs to `<image>` is never used: in the sequence above its value $(0.5, 0.5, 0.5)$ appears nowhere. That is why the backward pass needs a rule of its own.

<!-- worked: vlms.splice_backward -->
Suppose the gradient of the loss with respect to the residual-stream row at position t is g_t = (0.1 t, -0.1 t, 0.05), for t = 0 to 8. The splice routes the gradient of the four picture rows (positions 1 to 4) to the projector output, and zeroes those rows so that the gather's backward pass does not add them to E.

Gradient of E, row by row, if the picture rows were not zeroed and if they are:

| id | token | without the zeroing | with it |
|---:|---|---|---|
| 0 | <bos> | (0.00, 0.00, 0.05) | (0.00, 0.00, 0.05) |
| 1 | <image> | (1.00, -1.00, 0.20) | (0.00, 0.00, 0.00) |
| 2 | what | (0.50, -0.50, 0.05) | (0.50, -0.50, 0.05) |
| 3 | is | (0.60, -0.60, 0.05) | (0.60, -0.60, 0.05) |
| 4 | this | (0.70, -0.70, 0.05) | (0.70, -0.70, 0.05) |
| 5 | ? | (0.80, -0.80, 0.05) | (0.80, -0.80, 0.05) |
| 6 | cat | (0.00, 0.00, 0.00) | (0.00, 0.00, 0.00) |
| 7 | <eos> | (0.00, 0.00, 0.00) | (0.00, 0.00, 0.00) |

The gradient that reaches the projector is the four rows (0.10, -0.10, 0.05), (0.20, -0.20, 0.05), (0.30, -0.30, 0.05), (0.40, -0.40, 0.05), and with the tower outputs f of the previous table it gives dW_p = [[1.225, -0.05], [-1.225, 0.05], [0.225, 0.0]] and db_p = [1.0, -1.0, 0.2] (the sums over the four picture rows of g f^T and of g).
<!-- /worked -->

The gradient of a loss with respect to a row of $\mathbf{E}$ that was overwritten is zero, because the row never reached the output. A gather's backward does not know that. It adds the gradient of every position to the row of the id found there, so without the zeroing the table's row 1 would receive $(1.00, -1.00, 0.20)$, four gradients that belong to the projector. In the toy that is a harmless wrong number. In Qwen3-VL-4B it is not, because the 4B configuration ties the output head to the embedding table (`tie_embeddings: true` in `crates/qwen3vl/src/config.rs`): the row of the placeholder id is also the row that scores that id as the next token, and a gradient from the picture would be training the head. The splice's backward therefore does two things, it moves the picture rows' gradient out to the projector and zeroes them where they were.

### Placeholders in the crates

The crates fill in the same hole in three ways, and what differs is where the count of rows is decided. LLaVA keeps one sentinel. The string `<image>` is not in the Llama vocabulary, so `crates/llava/src/prompt.rs` splits the prompt on it before tokenizing, so that the tokenizer never sees it, and puts the id $-200$ in its place, a value outside the vocabulary on purpose so that it cannot collide with any real id; the splice code then expands that one sentinel into 576 rows, and refuses a prompt with any other number of splice points than one. Qwen3-VL, MiniCPM-V, DeepSeek-VL and Janus-Pro expand the placeholder into as many ids as rows before the sequence is built, wrapped in marker tokens (`vision_start` and `vision_end` for Qwen3-VL, `image` and `slice` markers for MiniCPM-V, `begin_of_image` and `end_of_image` for Janus-Pro), so that the length of the prompt and every position are known before the tower has run. The placeholder is a special token in the sense of Volume V, Chapter 2: an id the tokenizer matches whole, never one that ordinary text can spell by accident. MiniCPM-V performs the splice on the host, copying the picture rows over the pad rows of the embeddings of the prompt before prefill, and fails with a message that names both counts when the placeholders and the rows disagree (`MiniCpmV::splice` in `crates/minicpmv/src/model.rs`).

The kernels are shared. A decoder that splices on the device uses `splice.wgsl` and `splice_bwd.wgsl`, wrapped by `crates/model/src/vlm.rs`, and the story of that file is the usual one of a seam. Commit d0ebba798 (July 2026) added the two kernels and their step-builders as the one seam every vision-language model would compose: a forward that overwrites a contiguous run of residual rows and a backward that extracts the gradient and zeroes the run. By October the seam had been copied around it: commit 4bf2ad787 says that the decoders of qwen3, qwen35, qwen35moe and deepseek2 each carried their own run list, per-run buffers and step lists, and that only qwen3 took more than one run. A prompt with a sliced image has several runs, one per view, so the one-run copies could not serve it. The commit moved the bookkeeping into `SpliceRuns`, which validates that the runs are non-empty and do not overlap, owns one input and one gradient buffer per run, and gives qwen35 the several runs a sliced image needs. The test that belongs to it is a finite-difference check of the gradient through the splice (`check_vlm_splice` and `check_qwen35_splices` in `crates/gradcheck/src/lib.rs`), the kind of check [Volume III, Chapter 6](../v03-neural-networks/06-gradient-checking.md) taught, here applied to a hand-written backward that no framework would have produced for free. The splice is tiny and everything model-specific is on one side or the other of it; four copies of it were four chances to get the zeroing wrong.

## Two widths and a map between them

The tower writes rows of width $D_{\mathrm{v}}$ (1024 for the CLIP-L tower of Chapter 8, 1152 for SigLIP), and the decoder wants rows of width $D$ (5120 for Vicuna-1.5-13B, 2560 for the 4B model of Qwen3-VL). A **vision projector** is the small trained map from the first to the second. The toy's was one linear map, $\mathbf{p} = \mathbf{W}\mathbf{f} + \mathbf{b}$. The common upgrade, which LLaVA-1.5 calls `mlp2x_gelu`, puts a GELU between two linear layers,

$$
\mathbf{p} = \mathbf{W}_2\,\mathrm{gelu}(\mathbf{W}_1\mathbf{f} + \mathbf{b}_1) + \mathbf{b}_2,
\tag{1}
$$

with the exact (erf) GELU. A single linear map can turn the tower's space and rescale it, but it cannot bend it; the second layer is the cheapest way to let the map be nonlinear, at the price of a second matrix of $D \times D$. Brain has one implementation of this stack, `crates/model/src/projector.rs`, with a device path that runs forward and backward (the aligner is what a vision-language fine-tune trains) and a host path that serves as the reference for it. Commit 3aa2673ab (September 2026) moved LLaVA and FastVLM onto it and removed their private copies of the host projector and of the erf GELU.

Which rows feed it is a decision as well. LLaVA does not take the last block of its CLIP tower. Its configuration, `mm_vision_select_layer` of $-2$, taps the penultimate block (index 22 of 24, counting from 0), and `mm_vision_select_feature` of `patch` drops the class token, so that the projector reads the 576 patch rows (`LlavaConfig` in `crates/llava/src/config.rs`, which rejects any other selector with an error naming it instead of picking a near one). Chapter 8's loss acts on one pooled vector. Nothing in it asks the final block's patch rows to be anything in particular, which is a reason to read the rows one block earlier.

### Projectors that also shrink the sequence

Most of the cost of a picture is the number of rows, not the width of each, and the projector is the place to cut it. Qwen3-VL's `PatchMerger` (`crates/qwen3vl/src/encoder.rs`) takes each 2 by 2 block of patch rows, normalizes, concatenates the four rows into one of width $4D_{\mathrm{v}}$ and applies the two layers of (1) to that. The gather is a free reshape because the tower emits its patches in "merge-block order". It is the inverse of the `pixel_shuffle` of Chapter 5, which spreads channels over space: here space is folded into channels, and the same MLP then mixes them.

<!-- worked: vlms.merge_ratio_rows -->
A square picture of the given side through a tower with patches of 16 pixels. Merging each 2 x 2 block of patches into one token divides the count by 4; merging 4 x 4 blocks divides it by 16.

| side | patches | tokens, 2 x 2 merge | tokens, 4 x 4 merge |
|---|---:|---:|---:|
| 448 | 784 | 196 | 49 |
| 1024 | 4,096 | 1,024 | 256 |
| 1920 | 14,400 | 3,600 | 900 |
<!-- /worked -->

MiniCPM-V does the folding in two places. Its final merger is the same 2 by 2 block and MLP, but before it, after block 6 of its 27, a window merger (`crates/minicpmv/src/vision.rs`) runs one attention block inside each 2 by 2 window of patches, concatenates the four rows to 4,608 numbers and maps them down to 1,152 through a 17,216-wide layer, adding the window's mean. In its default 16x mode that is one token per 16 patches. Merging inside the tower, not after it, is what makes the later blocks cheap:

<!-- worked: vlms.window_merger_saving -->
FLOPs of the MiniCPM-V 4.6 tower for a 448 x 448 picture (1,024 patches of 14 pixels), width 1152, feed-forward 4304. A block costs 2 x rows x (4 x 1152^2 + 2 x 1152 x 4304) FLOPs for its matrix products and 4 x rows^2 x 1152 for the score and average products. The window merger runs one attention block over each 2 x 2 window (projections on 1,024 rows, scores over 4 keys), concatenates the four rows to 4,608 numbers and applies 4608 -> 17216 -> 1152 on 256 rows. Billions of FLOPs:

| tower | matrix products | score and average | total |
|---|---:|---:|---:|
| no merger: 27 blocks on 1,024 rows | 841.87 | 130.46 | 972.33 |
| window merger after block 6: 7 blocks on 1,024 rows, 20 on 256 | 374.17 | 39.86 | 414.03 |
| the window merger itself | 61.64 | 0.02 | 61.66 |
| merged tower plus its window merger | 435.81 | 39.88 | 475.69 |

The merged tower, window merger included, needs 48.9 percent of the FLOPs of the tower without it, a saving of 51.1 percent, and its later blocks attend over 256 rows instead of 1,024.
<!-- /worked -->

The table counts the same work the same way as Chapter 6 did. What it adds is a reason to put a merge early: the score products scale with the square of the rows, so the 20 later blocks, on a quarter of the rows, pay a sixteenth of their attention.

Now the projectors themselves, counted from the tensor shapes of the crates' configurations:

<!-- worked: vlms.projector_table -->
Parameters of the vision-to-language maps of the crates, counted from the tensor shapes of their configurations (weights and biases, norm gains and offsets included; `matmul` is the weights that multiply a row, the number that sets the FLOPs per row: 2 per weight). A DeepStack merger differs from the main merger only in its norm, over the 4096 concatenated numbers instead of the 1024 of one patch. The MiniCPM-V window merger is an attention block (a layer norm of width 1152, the query, key and value projection 1152 -> 3 x 1152 and the output projection 1152 -> 1152, both with biases) followed by a layer norm of width 4608 and the MLP 4608 -> 17216 -> 1152 with biases; its 11,520 norm numbers are 2 x 1152 + 2 x 4608 (gain and offset), its biases 3 x 1152 + 1152 + 17216 + 1152 = 22,976, and the two together are the difference between its two count columns.

| projector | widths | parameters | matmul weights | where |
|---|---|---:|---:|---|
| LLaVA-1.5-13B, two layers | 1024 -> 5120 -> 5120 | 31,467,520 | 31,457,280 | crates/llava |
| Qwen3-VL-4B, main merger | 1024 x 4 -> 4096 -> 2560 | 27,271,680 | 27,262,976 | crates/qwen3vl |
| Qwen3-VL-4B, one DeepStack merger | 1024 x 4 -> 4096 -> 2560 | 27,277,824 | 27,262,976 | norm over the 4096 |
| DeepSeek-VL-7B, split aligner | (1024, 1024) -> (2048 + 2048) -> 4096 | 20,979,712 | 20,971,520 | crates/deepseekvl |
| Janus-Pro-7B, understanding aligner | 1024 -> 4096 -> 4096 | 20,979,712 | 20,971,520 | crates/januspro |
| Janus-Pro-7B, generation aligner | 8 -> 4096 -> 4096 | 16,818,176 | 16,809,984 | Volume VIII |
| Moondream 3 preview, connector | 2304 -> 8192 -> 2048 | 35,661,824 | 35,651,584 | crates/moondream3 |
| MiniCPM-V 4.6, window merger (inside the tower) | 1152 x 4 -> 4608 -> 17216 -> 1152 | 104,507,072 | 104,472,576 | attention block over the 4 patches, then the MLP |

For comparison, one cross-attention layer of a resampler with 64 learned queries of width 2048 reading towers of width 1152 (queries 64 x 2048, W_q and W_o 2048 x 2048, W_k and W_v 1152 x 2048, no biases) has 13,238,272 parameters and always returns 64 tokens, whatever the number of patches it reads.
<!-- /worked -->

Three things stand out. The projectors are small, 17 to 36 million parameters against a tower of some hundreds of millions and a decoder of several billions, and the split aligner of DeepSeek-VL and the single-stream aligner of Janus-Pro have exactly the same count because the two halves of 2,048 the split aligner joins add up to the 4,096 the other has. The window merger of MiniCPM-V is nearly three times the largest of them, because it is a block of attention plus an MLP of its own and not a map, which is what the previous table's saving pays for. And the DeepStack mergers of Qwen3-VL are whole projectors of their own: the tower is tapped after blocks 5, 11 and 17, counting from 0, each tap is merged by a copy of the merger, and the merged rows are added to the image rows of the residual stream after decoder layers 0, 1 and 2 (the convention checked in `crates/qwen3vl/src/text_tower.rs`: DeepStack adds follow layers 0 up to the number of taps), so the decoder receives the picture three more times at shallow depth. That is three more projectors, 82 million parameters, and the FLOPs below.

<!-- worked: vlms.projector_flops -->
FLOPs of the projector for one picture: 2 FLOPs per matmul weight per row it maps (the merger maps one row per merged block of patches). The pictures are the usual size of each design, see the next section.

| projector | patches from the tower | rows mapped | GFLOPs |
|---|---:|---:|---:|
| LLaVA-1.5-13B, two layers | 576 | 576 | 36.24 |
| Qwen3-VL-4B, main merger | 4,096 | 1,024 | 55.83 |
| Qwen3-VL-4B, one DeepStack merger | 4,096 | 1,024 | 55.83 |
| DeepSeek-VL-7B, split aligner | 576 | 576 | 24.16 |
| Moondream 3 preview, connector | 729 | 729 | 51.98 |
<!-- /worked -->

A DeepStack picture costs the main merger and three more, $4 \times 55.83 = 223.32$ billion FLOPs for a 1024 by 1024 picture, against 36.24 billion for LLaVA's projector on its 576 rows. The reason is the merged width: a merger multiplies four times the tower's width, and each of the 1,024 rows it maps costs 2 FLOPs per weight of $27.3$ million.

A third family produces a fixed number of tokens whatever the tower's output, the perceiver resampler of Flamingo (Alayrac et al., 2022): $n_q$ learned query vectors cross-attend to all of the tower's rows and the $n_q$ results are the tokens. One layer of it with 64 queries of width 2,048 over a tower of width 1,152 has 13,238,272 parameters, and the number of tokens it hands on does not depend on the picture. It is the design to choose when the budget is fixed and the pictures vary, and none of brain's vision-language crates uses one: MiniCPM-V 4.6 reads its patches through the window merger and the merger instead.

## What a picture costs the decoder

A row of the decoder's sequence costs the same whether it came from a word or from a patch: every matrix of every layer is applied to it, its keys and values are stored, and every later row attends to it. Volume V, Chapter 12 counted both. The image tokens of one prompt add the following, for a prompt with 24 text tokens:

<!-- worked: vlms.decoder_cost -->
What the image tokens add to a prompt of 24 text tokens, with the counting rules of Volume V (Chapter 12): the prefill FLOPs of the whole prompt minus those of the 24 text tokens alone (weights at 2 FLOPs each, plus the score and average products over the keys each row sees; the output head counted once in both, so it cancels), and the cache the image rows occupy, at 2 bytes per stored number. Shapes: LLaVA-1.5's decoder has 40 layers, width 5120, 40 query and 40 key-value heads of width 128, feed-forward width 13824, gated; Qwen3-VL-4B's has 36 layers, width 2560, 32 query and 8 key-value heads of width 128, feed-forward width 9728, gated.

| decoder | picture | image tokens | prefill TFLOPs added | cache bytes per token | cache of the image (MiB) |
|---|---:|---:|---:|---:|---:|
| LLaVA-1.5-13B | 336 x 336, fixed | 576 | 14.76 | 819,200 | 450.00 |
| Qwen3-VL-4B | 448 x 448 | 196 | 1.44 | 147,456 | 27.56 |
| Qwen3-VL-4B | 1080 x 1920 | 2,040 | 16.08 | 147,456 | 286.88 |
| Qwen3-VL-4B | 1080 x 1920, served limit | 1,008 | 7.64 | 147,456 | 141.75 |
| Qwen3-VL-4B | 3000 x 2000 | 5,828 | 52.45 | 147,456 | 819.56 |
<!-- /worked -->

LLaVA-1.5's 576 tokens cost the 13-billion-parameter decoder 14.76 trillion FLOPs of prefill and 450 MiB of cache, because its decoder has 40 key-value heads and every token stores 819,200 bytes. Qwen3-VL's 8 key-value heads make a token 147,456 bytes, 0.18 of that, which is why the 4B model can afford 2,040 tokens for a 1080 by 1920 picture in 287 MiB. The count of tokens is what differs between designs, and there are three ways of fixing it.

**A fixed square.** LLaVA-1.5 resizes the shortest edge to 336 and takes the central 336 by 336 crop (`crates/llava/src/caps.rs`), 576 tokens for every picture. A 16 by 9 photograph loses the side parts: 336 of 597 columns survive, 56 percent. DeepSeek-VL and Janus-Pro pad instead of cropping, so that nothing is lost and the tower sees a square of which, for the same photograph, 44 percent is padding: of the 576 tokens, 252 describe grey. Moondream 3 and its pooling are a fixed budget of another kind, taken up in the section on slices and crops. A fixed budget is predictable, which is what a serving system needs, and for a large photograph it is both the cheapest and the poorest in detail.

**Slices.** A large picture is cut into pieces of the tower's native size and each piece is encoded as if it were a picture of its own, together with a downscaled view of the whole. The count grows in steps.

**Dynamic resolution.** The picture keeps its shape, within bounds, and the grid, and with it the number of tokens, is whatever the picture's size makes it, instead of being fixed in advance. The count grows in proportion to the pixels, and that is both its strength and its danger.

<!-- worked: vlms.token_budget -->
Image tokens that reach the decoder for one picture of the given height x width, under the rule of each design: a fixed square (LLaVA-1.5, and DeepSeek-VL and Janus-Pro, which also give 576), a fixed count after pooling (Moondream 3), slices with a source view (MiniCPM-V 4.6 at 16x), and dynamic resolution with the released limits and with brain's served limit of 1024 x 1024 pixels (Qwen3-VL).

| picture h x w | fixed square (LLaVA-1.5) | fixed after pooling (Moondream 3) | sliced (MiniCPM-V) | dynamic (Qwen3-VL), released limits | dynamic, served limit |
|---|---:|---:|---:|---:|---:|
| 336 x 336 | 576 | 729 | 64 | 100 | 100 |
| 448 x 448 | 576 | 729 | 64 | 196 | 196 |
| 480 x 640 | 576 | 729 | 203 | 300 | 300 |
| 1080 x 1920 | 576 | 729 | 578 | 2,040 | 1,008 |
| 3000 x 2000 | 576 | 729 | 574 | 5,828 | 1,014 |
<!-- /worked -->

![Image tokens against the side of a square picture, on logarithmic axes, for four designs. A fixed square gives 576 at every size and a pooled design 729. The sliced design is flat at 64 up to a side of 448, steps to 320 and then to 640 as slices are added. The dynamic design grows as the square of the side, from 49 tokens at 224 pixels to 7,056 at 2,688 pixels, and passes the fixed 576 at a side of 768 pixels.](img/image-tokens-resolution.svg)

The figure and the table say that no design wins everywhere. The dynamic design is cheaper than the fixed square below a side of 768 pixels, where its merged grid is 24 by 24, because it spends tokens only on the pixels there are, and dearer above, quadratically; the sliced design is the cheapest at every size of the figure from 336 pixels up (at 224 the dynamic grid has 49 tokens to its 64) and pays for it by showing the decoder 640 tokens for a picture 2,688 pixels on a side, where the dynamic design shows 7,056. The costs of the encoder run the same way, and more steeply, because the tower attends over every patch, before the merger has reduced them to a quarter:

<!-- worked: vlms.tower_attention -->
The tower attends over every patch of the picture at once (no windows), 24 blocks of width 1024; one block's score and average products cost 4 T^2 x 1024 FLOPs for T patches (Chapter 6). The decoder sees T / 4 tokens.

| picture h x w | patches T | tokens T / 4 | tower attention TFLOPs |
|---|---:|---:|---:|
| 448 x 448 | 784 | 196 | 0.06 |
| 1080 x 1920 | 8,160 | 2,040 | 6.55 |
| 3000 x 2000 | 23,312 | 5,828 | 53.42 |
<!-- /worked -->

At 3000 by 2000 the score and average products of the tower alone, 53.4 trillion FLOPs, are as large as the whole decoder's prefill of the 5,828 tokens (52.45 in the previous table), which is the first reason dynamic resolution always comes with a bound on the area. The second is memory. The tower of Qwen3-VL attends over the whole picture as one span, so the matrix of scores grows with the square of the patch count, and `crates/qwen3vl/src/encoder.rs` cuts the query rows into chunks so that one slab of scores stays under 256 MiB (the cuts split the queries only, never the keys, so every row still sums over the whole picture in the same order and the result is bit-identical to one unchunked pass):

<!-- worked: vlms.tower_slab -->
A square picture at each pixel budget through the Qwen3-VL-4B tower: the resized side, the patches T the tower attends over, the tokens T / 4 the decoder reads, one slab of scores `[16 heads, T queries, T keys]` in 32-bit numbers (GiB), and the query rows per chunk that keeps one slab within 256 MiB (the chunk is 256 MiB / (16 x T x 4 bytes), at least 64 and at most 4,096 rows) with the number of chunks that makes.

| budget | resized | patches T | decoder tokens | score slab (GiB) | query rows per chunk | chunks |
|---|---|---:|---:|---:|---:|---:|
| served default 1024 x 1024 | 1024 x 1024 | 4,096 | 1,024 | 1.00 | 1,024 | 4 |
| captioning default 1280 x 1280 | 1280 x 1280 | 6,400 | 1,600 | 2.44 | 655 | 10 |
| released limit 3584 x 3584 | 3584 x 3584 | 50,176 | 12,544 | 150.06 | 83 | 605 |
<!-- /worked -->

## Letting the picture choose its size

Dynamic resolution needs a rule that turns the size of the picture into the size of its grid. The rule that Qwen2-VL introduced (Wang et al., 2024), `smart_resize`, is short enough to derive.

Three constraints decide it. The tower cuts patches of $P$ pixels, and the merger groups $m \times m$ of them, so each side of the grid must be a multiple of the factor $P\,m$, which is $16 \times 2 = 32$ pixels in Qwen3-VL. The picture should keep its shape as far as that allows. And the area must lie between $N_{\min}$ and $N_{\max}$, so that no picture costs less than a few tokens or more than the decoder and the tower can hold. Round each side to the nearest multiple, at least one factor:

$$
\bar h = \max\!\bigl(Pm,\ Pm \cdot \mathrm{round}(h / Pm)\bigr), \qquad \bar w = \max\!\bigl(Pm,\ Pm \cdot \mathrm{round}(w / Pm)\bigr),
\tag{2}
$$

with ties to the even multiple. Each side moves by at most half a factor, 16 pixels, so the shape changes by at most the fraction $16/h$ in one side and $16/w$ in the other, which is a few percent for a photograph and large only for a very thin picture. If the rounded area $\bar h \bar w$ is above $N_{\max}$, shrink both original sides by the same factor $\beta = \sqrt{hw/N_{\max}}$, which brings the area to exactly $N_{\max}$ with the shape unchanged, and round **down**, but never below one factor. Rounding to nearest at that point could carry the area back over the bound; rounding down cannot. If the area is below $N_{\min}$, grow by $\beta = \sqrt{N_{\min}/(hw)}$ and round **up**, for the same reason mirrored.

```text
function smart_resize(h: int, w: int, f: int, lo: int, hi: int) -> (int, int):
    require h >= 1 and w >= 1 and f >= 1 and lo <= hi, "a picture and a valid range of areas"
    hb = max(f, round_even(h / f) * f)
    wb = max(f, round_even(w / f) * f)
    if hb * wb > hi:
        beta = sqrt(h * w / hi)
        hb = max(f, floor(h / beta / f) * f)
        wb = max(f, floor(w / beta / f) * f)
    else if hb * wb < lo:
        beta = sqrt(lo / (h * w))
        hb = ceil(h * beta / f) * f
        wb = ceil(w * beta / f) * f
    return (hb, wb)
```

The tokens the picture then costs are $\bar h \bar w / (Pm)^2$. The three branches on three pictures, at the released limits $N_{\min} = 56^2 = 3{,}136$ and $N_{\max} = 3584^2 = 12{,}845{,}056$:

<!-- worked: vlms.smart_resize_branches -->
The three branches of the rule at factor 32 and the released limits: first both sides are rounded to the nearest multiple of 32 (halves to the even multiple) and at least 32; if the area is then over the maximum both original sides are divided by beta and rounded down to a multiple of 32, and if it is under the minimum both are multiplied by beta and rounded up.

| picture h x w | h / 32, w / 32 | rounded | area test | result | area |
|---|---:|---:|---|---:|---:|
| 1080 x 1920 | 33.750, 60.000 | 1088 x 1920 | 3,136 <= 2,088,960 <= 12,845,056: keep | 1088 x 1920 | 2,088,960 |
| 4000 x 6000 | 125.000, 187.500 | 4000 x 6016 | 24,064,000 > 12,845,056: shrink by beta = sqrt(4000 6000 / 12,845,056) = 1.3669 | 2912 x 4384 | 12,766,208 |
| 40 x 40 | 1.250, 1.250 | 32 x 32 | 1,024 < 3,136: grow by beta = sqrt(3,136 / (40 40)) = 1.4000 | 64 x 64 | 4,096 |
<!-- /worked -->

And on a range of shapes, at the released limit and at the 1024 by 1024 that brain's served model uses by default (`DEFAULT_SERVE_MAX_PIXELS` in `crates/qwen3vl/src/caps.rs`, a practical capacity for the splice and DeepStack buffers of a resident model, not the released ceiling):

<!-- worked: vlms.smart_resize_table -->
`smart_resize` with factor 32 (patch 16 times merge 2), minimum area 56 x 56 = 3,136 pixels and maximum area 3584 x 3584 = 12,845,056 pixels (the released limits), and with the maximum 1024 x 1024 = 1,048,576 of brain's served default. Sizes are height x width in pixels; a token is one 32 x 32 block of the resized picture.

| picture h x w | w / h | resized (released limits) | w / h after | tokens | resized (served limit) | tokens |
|---|---:|---:|---:|---:|---:|---:|
| 448 x 448 | 1.000 | 448 x 448 | 1.000 | 196 | 448 x 448 | 196 |
| 480 x 640 | 1.333 | 480 x 640 | 1.333 | 300 | 480 x 640 | 300 |
| 1080 x 1920 | 1.778 | 1088 x 1920 | 1.765 | 2,040 | 768 x 1344 | 1,008 |
| 3000 x 2000 | 0.667 | 3008 x 1984 | 0.660 | 5,828 | 1248 x 832 | 1,014 |
| 100 x 1500 | 15.000 | 96 x 1504 | 15.667 | 141 | 96 x 1504 | 141 |
| 40 x 40 | 1.000 | 64 x 64 | 1.000 | 4 | 64 x 64 | 4 |
| 4000 x 6000 | 1.500 | 2912 x 4384 | 1.505 | 12,467 | 832 x 1248 | 1,014 |
<!-- /worked -->

Read the table by columns. Between the bounds, `smart_resize` changes nothing but the rounding: 448 by 448 and 480 by 640 are already multiples of 32, and 1080 becomes 1088 with an aspect ratio of 1.765 for 1.778. Over the bound, the shape survives and the area is cut, to 1,008 tokens for a 1080 by 1920 picture at the served limit and 1,014 for the others, all just under the 1,024 that $N_{\max}/32^2$ allows. Under the lower bound, the 40 by 40 picture is blown up to 64 by 64, four tokens. And the thin banner of 100 by 1500 shows where the rule is least kind: its short side rounds from 100 to 96, so the shape moves from 15 to 15.67, 4.4 percent.

The rounding rule has one trap that is worth seeing once, because the language it was ported from and the language it was ported to have different defaults. Python's `round` goes to the even neighbour on a tie, and Rust's `round` goes away from zero; the port uses `round_ties_even` to match. The two differ on every side that is an odd multiple of 16, which is far from rare for a real photograph (1,584 pixels is 49.5 factors):

<!-- worked: vlms.smart_resize_halves -->
Sides that are exactly half way between two multiples of 32 go to the even multiple (Python's `round`, and the Rust `round_ties_even` that ports it):

| side | side / 32 | round half to even | floor | ceil |
|---|---:|---:|---:|---:|
| 48 | 1.5 | 64 | 32 | 64 |
| 80 | 2.5 | 64 | 64 | 96 |
| 112 | 3.5 | 128 | 96 | 128 |
| 144 | 4.5 | 128 | 128 | 160 |
| 176 | 5.5 | 192 | 160 | 192 |

And a picture 31 pixels high and 600,000 wide: rounded it is 32 x 600,000 = 19,200,000 pixels, over the maximum, so beta = sqrt(31 * 600,000 / 12,845,056) = 1.2033 and floor(31 / 1.2033 / 32) * 32 = 0; the clamp to one factor makes the height 32 (the result is 32 x 498,592).
<!-- /worked -->

The last line of that block is a different matter: a picture 31 pixels high and 600,000 wide is rounded to 32 by 600,000 and shrunk, and the floor alone would take its height to zero. The rule therefore clamps each side to at least one factor after shrinking, as the reference does. Brain's port once dropped that clamp, so such a banner reached the tower with a side of zero; commit 8b5f1310b restored it, and `smart_resize` in `crates/qwen3vl/src/preprocess.rs` now clamps, with a test for exactly these shapes. The reference also refuses aspect ratios beyond 200 to 1, and `smart_resize_video` asserts the same; the image function sizes such a picture instead, so that a served request is answered rather than aborted.

### A constant that agreed on every picture that was tried

The limits were once wrong, and the way they were wrong is the lesson. Commit 70665529a (August 2026) found that `DEFAULT_MIN_PIXELS` and `DEFAULT_MAX_PIXELS` were $256^2$ and $4096^2$, and that a comment claimed they came from the released `preprocessor_config.json`. Fetched directly (the file of the Qwen3-Omni release, whose image processor this crate also serves; the values are not claimed here for every checkpoint), it said 3,136 and 12,845,056: the values of an earlier Qwen-VL generation had been carried over. The commit message adds that for the bench fixture of 697 by 503 both pairs of bounds give the same 512 by 704, so the error was not the cause of the inaccuracy being chased and could sit unnoticed. The constants in `crates/qwen3vl/src/preprocess.rs` now carry the corrected values and a comment saying where they were checked.

<!-- worked: vlms.bounds_story -->
The same rule with two pairs of limits: the constants the crate carried before commit `70665529a` (minimum area 256 x 256, maximum 4096 x 4096) and the released ones (56 x 56 and 3584 x 3584). Pictures in the wide middle of the range cannot tell them apart.

| picture h x w | resized, old limits | tokens | resized, released limits | tokens |
|---|---:|---:|---:|---:|
| 503 x 697 | 512 x 704 | 352 | 512 x 704 | 352 |
| 100 x 100 | 256 x 256 | 64 | 96 x 96 | 9 |
| 200 x 200 | 256 x 256 | 64 | 192 x 192 | 36 |
| 3500 x 4000 | 3488 x 4000 | 13,625 | 3328 x 3808 | 12,376 |
| 4096 x 4096 | 4096 x 4096 | 16,384 | 3584 x 3584 | 12,544 |
<!-- /worked -->

A picture in the wide middle of the range cannot tell the two pairs apart, and a picture at either end gets a different token count, 64 against 9 for a 100 by 100 image, or 16,384 against 12,544 for a 4096 by 4096 one. A preprocessing constant that is right on typical inputs and wrong at the edges is invisible to any test that uses typical inputs, which is the argument of [Chapter 1](01-images.md) about the constants a model was trained with, from the opposite side: the model's weights never forgot the numbers it was trained under, and a copy of the numbers from the previous release is the quiet way to lose them.

## Slices and crops

A tower trained at one size has a native size, and cutting a large picture into native-size pieces is the oldest route to detail. MiniCPM-V's processor (`crates/minicpmv/src/preprocess.rs`) does it with a rule that is easy to state and a little odd in its details, all reproduced from the reference with its Python rounding and truncation. A picture of $h \times w$ pixels is first compared with the slice scale of 448: $\lceil hw / 448^2 \rceil$, at most 9, is the target number of slices, and if it is 1 the picture stays whole. Otherwise the candidates are the grids of $r$ rows and $c$ columns with $rc$ equal to the target minus one, the target, or the target plus one, and the winner is the one whose aspect ratio $c/r$ is nearest to the picture's in log space, with a tie going to more rows.

```text
function slice_grid(h: int, w: int, scale: int, max_slices: int) -> (int, int):
    require h >= 1 and w >= 1 and scale >= 1 and max_slices >= 1, "a picture and a slice scale"
    multiple = min(ceil(w * h / (scale * scale)), max_slices)
    if multiple <= 1:
        return (0, 0)                          # (0, 0): the picture is not sliced
    best = (1, 1)
    best_err = infinity
    for n in multiple - 1 ..= multiple + 1:
        if n == 1 or n > max_slices:
            continue
        for r in 1 ..= n:
            if n mod r == 0:
                c = n // r
                err = abs(log(w / h) - log(c / r))
                if err < best_err or (err == best_err and r > best[0]):
                    best = (r, c)
                    best_err = err
    return best
```

<!-- worked: vlms.slicing_candidates -->
A 1080 x 1920 picture: area / 448^2 = 10.332, so multiple = ceil(10.332) = 9 and the candidates are the slice counts 8, 9 and 10, of which the counts above the maximum of 9 slices are skipped (here 10, which therefore has no rows in the table). The aspect ratio of the picture is 1.778, ln = 0.5754. The error of a grid is |ln(w / h) - ln(cols / rows)|; the smallest wins and a tie goes to more rows.

| slices | rows x cols | cols / rows | error |
|---|---:|---:|---:|
| 8 | 1 x 8 | 8.000 | 1.5041 |
| 8 | 2 x 4 | 2.000 | 0.1178 |
| 8 | 4 x 2 | 0.500 | 1.2685 |
| 8 | 8 x 1 | 0.125 | 2.6548 |
| 9 | 1 x 9 | 9.000 | 1.6219 |
| 9 | 3 x 3 | 1.000 | 0.5754 |
| 9 | 9 x 1 | 0.111 | 2.7726 |

The winner is 2 x 4.
<!-- /worked -->

The log is what makes a wide grid and a tall one symmetric: 2 by 4 and 4 by 2 are equally far from a square in the sense of log, $\pm\ln 2$, where the plain ratios 2 and 0.5 would not be. The picture is then resized so that the grid cuts it into whole slices and each slice is cut from it, and the whole picture is also kept as a **source view**. Both sizes come from one function. Given a region of $h \times w$ pixels it shrinks the region to the slice scale when it is larger (or always, when `upscale` is true, which also enlarges a smaller one), keeping its aspect ratio, truncating as the reference's `int` does, and then rounds each side to a multiple of 56 pixels, the patch of 14 times the 4 patches on a side of one token of the 16x mode:

```text
function view_size(h: real, w: real, scale: int, patch: int, upscale: bool) -> (int, int):
    require h > 0.0 and w > 0.0, "a region of positive size"
    hh = h
    ww = w
    if h * w > scale * scale or upscale:
        a = w / h
        hh = floor(scale / sqrt(a))              # the reference's int(): truncation of a positive number
        ww = floor(hh * a)
    q = 4 * patch
    return (max(round_even(hh / q) * q, q), max(round_even(ww / q) * q, q))

function slice_size(h: int, w: int, rows: int, cols: int, scale: int, patch: int) -> (int, int):
    rh = max(round_even(h / rows) * rows, rows)  # the picture is resized to (rh, rw), a whole number
    rw = max(round_even(w / cols) * cols, cols)  # of pixels per slice, and cut into rows x cols slices
    return view_size(rh / rows, rw / cols, scale, patch, true)
```

The source view is `view_size(h, w, 448, 14, upscale)` with `upscale` true only when the picture is not sliced; one slice is `slice_size`. A view of $a \times b$ pixels has $(a/14)(b/14)$ patches and a sixteenth of that many tokens in the 16x mode, a quarter in the 4x mode. Every view goes through the tower at its own size, with attention only inside the view, and the prompt lays them out with markers, one `image` marker for the source view and one `slice` marker for each slice, rows of slices separated by a newline (`view_layout` in `crates/minicpmv/src/prompt.rs`).

<!-- worked: vlms.slicing_table -->
The image processor of MiniCPM-V 4.6 as `preprocess.rs` states it (patch 14, slice scale 448, at most 9 slices; Python rounding and truncation): whether the picture is sliced and into how many rows x columns, the source view (the whole picture resized to a multiple of 56 on each side), the size of one slice, and the visual tokens at the default 16x downsampling (one token per 16 patches), in total and at 4x (one token per 4 patches).

| picture h x w | grid rows x cols | source view | source tokens | one slice | tokens per slice | visual tokens (16x) | visual tokens (4x) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 448 x 448 | none | 448 x 448 | 64 | - | - | 64 | 256 |
| 768 x 1024 | 2 x 2 | 392 x 504 | 63 | 392 x 504 | 63 | 315 | 1260 |
| 1080 x 1920 | 2 x 4 | 336 x 616 | 66 | 448 x 448 | 64 | 578 | 2312 |
| 3000 x 2000 | 4 x 2 | 560 x 392 | 70 | 392 x 504 | 63 | 574 | 2296 |
| 200 x 1600 | 1 x 3 | 168 x 1288 | 69 | 280 x 728 | 65 | 264 | 1056 |
| 4000 x 6000 | 2 x 4 | 392 x 560 | 70 | 504 x 392 | 63 | 574 | 2296 |
<!-- /worked -->

Two things follow. The count is bounded by the number of views, a source view and at most nine slices, and it moves in steps (203, 315, 578 in the table), not smoothly. And the slices are not squares of 448: those of a 4000 by 6000 photograph are 504 by 392, portrait pieces of a 2 by 4 grid whose whole is 1,008 by 1,568, an aspect ratio of 1.56 for the photograph's 1.5, and its source view is 392 by 560. That is how a 2 by 4 grid serves a picture that is not 2 to 1: each slice is taller than it is wide.

Moondream 3 takes the opposite choice for what the decoder sees. Its tower is a SigLIP with a crop size of 378, a 27 by 27 grid of 14-pixel patches. A large picture is cut into overlapping crops, every crop's feature map is trimmed of the margin it shares with its neighbours and stitched into one big map of $(27 - 8)\,t + 8$ patches per side for $t$ tiles, and the stitched map is average-pooled back to 27 by 27 and concatenated channel-wise with the global crop's map, 2,304 numbers per cell:

<!-- worked: vlms.overlap_crop_table -->
The overlap multi-crop of Moondream 3 (crop 378 pixels = 27 x 27 patches of 14, margin 4 patches, at most 12 crops): a crop origin every (27 - 8) x 14 = 266 pixels, a resized picture of 266 x tiles + 112 pixels per side, local feature maps stitched into (27 - 8) x tiles + 8 patches per side, and one more pass for the global crop. Whatever the tiling, the stitched map is pooled to 27 x 27, so the decoder reads 729 rows.

| picture h x w | tiles rows x cols | resized to | stitched patches | pooling ratio (area) | tower passes | patches through the tower | image tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| 378 x 378 | 1 x 1 | 378 x 378 | 27 x 27 | 1.00 | 2 | 1,458 | 729 |
| 600 x 800 | 2 x 4 | 644 x 1176 | 46 x 84 | 5.30 | 9 | 6,561 | 729 |
| 768 x 1024 | 3 x 4 | 910 x 1176 | 65 x 84 | 7.49 | 13 | 9,477 | 729 |
| 500 x 2000 | 1 x 6 | 378 x 1708 | 27 x 122 | 4.52 | 7 | 5,103 | 729 |
| 3000 x 4000 | 2 x 4 | 644 x 1176 | 46 x 84 | 5.30 | 9 | 6,561 | 729 |
<!-- /worked -->

The tiling comes from a rule of its own, a function of the picture's size after the margin of every crop edge (4 patches of 14 pixels, 56 pixels on each side) is taken off, against the stride between crop origins, $(27 - 8) \times 14 = 266$ pixels, and the largest number of crops, 12; the picture is then resized to $266\,t + 112$ pixels per side for $t$ tiles on it:

```text
function crop_tiling(h: int, w: int, crop: int, max_crops: int) -> (int, int):
    require h >= 1 and w >= 1 and crop >= 1 and max_crops >= 1, "a picture and a crop"
    if h <= crop or w <= crop:
        return (1, 1)
    min_h = ceil(h / crop)
    min_w = ceil(w / crop)
    if min_h * min_w > max_crops:
        ratio = sqrt(max_crops / (min_h * min_w))
        return (max(1, floor(min_h * ratio)), max(1, floor(min_w * ratio)))
    ht = max(floor(sqrt(max_crops * h / w)), min_h)
    wt = max(floor(sqrt(max_crops * w / h)), min_w)
    if ht * wt > max_crops:
        if wt > ht:
            wt = floor(max_crops / ht)
        else:
            ht = floor(max_crops / wt)
    return (max(1, ht), max(1, wt))
```

It is called as `crop_tiling(h - 112, w - 112, 266, 12)`. The decoder reads 729 rows at every size: all the added resolution is paid in tower passes and lost in the pool, which is a choice of a fixed decoder cost over a growing one, and a coarser view of a large picture at high resolution than the others give. The crop policy is a named function of `crates/imaging/src/tiling.rs` and the module's convention keeps it apart from the discrete policy of another model: the two agree on many sizes and disagree on others, and a pinned test of that crate holds a counterexample so that nobody merges them in the name of tidiness. The pixel-space crop geometry of `crates/moondream3/src/preprocess.rs` is checked against the feature-space stitch it must agree with, and that test is what ties the two halves together.

## Where an image token sits

The splice decides which rows are the picture. A second decision is what position each row carries, because a rotary decoder rotates every query and key by an angle that depends on where the token is. Most of the crates give the picture rows what any row gets, its index in the sequence (MiniCPM-V, for one, keeps one-dimensional positions). Chapter 6 made the argument that this loses less than it seems: the tower has already put the two-dimensional layout into every row, and the decoder is not told that rows 1 to 576 were a 24 by 24 grid.

Qwen3-VL tells it, with the multi-axis rotary embedding of Volume V, Chapter 8. Positions for the toy prompt, in both schemes:

<!-- worked: vlms.toy_positions -->
The spliced sequence of the first example: `<bos>`, four picture rows in a grid of 1 frame by 2 rows by 2 columns (sizes (T, H, W) = (1, 2, 2)), and four words. One-dimensional RoPE gives each row its index. The multi-axis rule of Volume V, Chapter 8 gives a word (c, c, c) and advances the counter c by 1, and gives a picture row (c + t, c + h, c + w) for its frame t, row h and column w, each counted from 0 (so the four rows have (t, h, w) = (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)), then advances c by max(T, H, W) = max(1, 2, 2) = 2. The last column is the position triple the rule assigns to the token.

| index | token | one-dimensional position | position triple |
|---|---|---:|---:|
| 0 | <bos> | 0 | (0, 0, 0) |
| 1 | image patch | 1 | (1, 1, 1) |
| 2 | image patch | 2 | (1, 1, 2) |
| 3 | image patch | 3 | (1, 2, 1) |
| 4 | image patch | 4 | (1, 2, 2) |
| 5 | what | 5 | (3, 3, 3) |
| 6 | is | 6 | (4, 4, 4) |
| 7 | this | 7 | (5, 5, 5) |
| 8 | ? | 8 | (6, 6, 6) |
<!-- /worked -->

This is **multimodal position encoding**: the rule that gives the tokens of a sequence with more than one kind of token their positions, here a triple for a patch row, equal on all three axes for a word, and a counter that moves past the picture by its largest extent, not by its token count. It is `get_rope_index_multi` in `crates/qwen3vl/src/mrope.rs`: a run of text advances all three axes together, like plain RoPE, and an image is placed on a meshgrid anchored at the running counter, which then advances by $\max(t, h, w)$ so that the next word starts clear of the whole block. A patch and the one under it are now $(0, 1, 0)$ apart whatever the width, and the first word after the picture is at position 3 and not 5.

How much the counter saves depends on the grid:

<!-- worked: vlms.position_advance -->
How far a picture moves the position counter. A grid of h x w merged tokens takes h w rows of the sequence; one-dimensional RoPE advances the counter by h w, the multi-axis rule by max(h, w). The text that follows is that many positions closer to the text before the picture (and the cached rows are still one per token).

| picture | grid h x w | rows in the sequence | advance, one-dimensional | advance, multi-axis | positions saved |
|---|---|---:|---:|---:|---:|
| toy picture | 2 x 2 | 4 | 4 | 2 | 2 |
| LLaVA-1.5 / DeepSeek-VL grid | 24 x 24 | 576 | 576 | 24 | 552 |
| 448 x 448 in Qwen3-VL | 14 x 14 | 196 | 196 | 14 | 182 |
| 1080 x 1920 in Qwen3-VL | 34 x 60 | 2,040 | 2,040 | 60 | 1,980 |
| 3000 x 2000 in Qwen3-VL | 94 x 62 | 5,828 | 5,828 | 94 | 5,734 |
<!-- /worked -->

A 1080 by 1920 picture is 2,040 rows of the sequence and 60 positions of the counter. That does not make the cache smaller (the keys of the 2,040 rows are all stored, rotated at their triples), but it means that a prompt with several large pictures stays within the range of positions the decoder was trained on, since a long context of pictures needs only $\sum \max(h, w)$ of it. The consequence for the generated text is the part that is easy to get wrong. A token's position is no longer its KV row. Brain keeps this in one place, `PromptRope` in `crates/qwen3vl/src/mrope.rs`: the position of a prompt token is the one `get_rope_index` gave it, and every generated token sits at its KV row plus a constant, the largest position of any axis plus one, minus the prompt length. In the toy, the last prompt token `?` is at $(6, 6, 6)$ in KV row 8, so the next token, KV row 9, is at position 7 and not 9. A decoder that used the KV row would be correct for text-only prompts and wrong, silently, for exactly the prompts that carry pictures.

The channels of a head are shared among the axes in the interleaved layout of Volume V, Chapter 8:

<!-- worked: vlms.axis_ownership -->
Qwen3-VL-4B: head width 128, so 64 pairs, sections (t, h, w) = (24, 20, 20) of the interleaved layout. The pairs owned by each axis (counts 24, 20, 20):

| axis | pairs | first, last |
|---|---:|---|
| t | 24 | 0 to 63 (every third, then 60 to 63) |
| h | 20 | 1, 4, ... 58 |
| w | 20 | 2, 5, ... 59 |

The pairs of a pattern `thwthwthwthw...` rotate through all three axes, so each axis keeps both the fast and the slow end of the frequency range.
<!-- /worked -->

The history told in Volume V, Chapter 8 applies here unchanged: commit 160ff7cda (September 2026) found Qwen3-VL's text-only encoding path rotating through a table that had never been uploaded, and a zero table does not mean no rotation but zero scores. The tower has positions of its own, of a different kind. Its patches carry two-axis rotary embedding (the pairs of a head are split between the row and the column) and a learned $48 \times 48$ table that is bilinearly resampled onto the grid at hand, which is the step that lets one set of weights read a $14 \times 14$ grid and a $94 \times 62$ one.

Two designs do something else with the same rows. Moondream 3 gives its 729 image rows ordinary positions but changes the mask: the `bos` and the 729 rows form a prefix of 730 rows that attend to each other in both directions, and everything after is causal, $\mathrm{allow}(i, j) = (i < 730 \wedge j < 730) \vee j \le i$. An image row is never a prediction target, so there is no reason to forbid it to see the rest of the picture; the cost is that the prefix's keys and values must come from one masked batched pass, because an incremental decode step, which attends causally, would be wrong for the prefix (`generate_kv` in `crates/moondream3/src/model.rs` is that pass). And Florence-2 has no decoder positions to speak of, since the vision rows enter an encoder: its image rows get a learned two-axis position, a row embedding and a column embedding of 512 channels each concatenated to 1,024, and then, together with the prompt's rows, a learned absolute position of the encoder (`crates/florence2/src/vision/project.rs`, `crates/florence2/src/text/encoder.rs`).

## How the pieces learn, and how they are scored

The general recipe for a model of this kind comes from the LLaVA papers (Liu et al., 2023) and is in two stages, stated here as the papers' recipe and not as anything brain does. In the first, the tower and the decoder are frozen and only the projector learns, on picture-caption pairs: its job is to map the tower's rows into the region of the decoder's input space where the caption's words live, the alignment of the two spaces. In the second, the projector and the decoder (sometimes the tower, too) are trained on instructions and answers about pictures, with a language-model loss on the answer tokens only, the picture rows being inputs and never targets. That is why the splice has a backward at all: the first stage needs the gradient of the loss to reach the projector through rows the decoder treats as ordinary.

Where brain trains these models, it follows the same split. `crates/deepseekvl/src/train.rs` fine-tunes the composite with the two towers frozen, their features extracted once per image and the towers released; the aligner trains from the checkpoint's own weights, the decoder through a low-rank adapter (LoRA, a small trained correction added to frozen weights), and the gradient of the spliced rows is carried back through the aligner image by image (commit bf8d27097, September 2026, whose test checks the aligner's gradient through the split aligner, the splice and the decoder against finite differences). MiniCPM-V trains more of itself: commit 2d4b550d3 (October 2026) gives it a backward through the tower with checkpointing, so that a full-resolution request does not hold 27 layers of attention probabilities, and LoRA on the tower's linears, each of the decoder, the connector and the tower being trained or frozen on its own (`crates/minicpmv/src/train.rs`). The choice of what is trained is the choice of how much of the first stage to redo.

Scoring is of two kinds. A captioning model is scored against reference captions by the overlap of their n-grams (BLEU, the bilingual evaluation understudy, and CIDEr, consensus-based image description evaluation, are the standard scores). A visual question answering model is scored by agreement with human answers: the standard accuracy of Antol et al. (2015) gives an answer the credit $\min(1, n/3)$, where $n$ is how many of the ten human annotators gave it. Both measure the model in the end, and neither says whether a port of the model is correct; for a port the gate is parity with the reference, and brain's gates are of that kind. The DeepSeek-VL gate is a requirement of 16 greedy tokens identical to the fp32 reference's, checked stage by stage on the real checkpoint (`crates/deepseekvl/tests/it/composite_parity.rs`), and commit edf32fb7d (October 2026) states the same for MiniCPM-V: the next-token logits after the prompt match the reference at a cosine of at least 0.9999999999 and the whole greedy continuation is identical, for a sliced image at 16x and at 4x, a document and two images in one prompt. What such a gate cannot see is a preprocessing step that is not in the golden: the commit that checked the MiniCPM-V image processor (f52516aea) compared resized views byte for byte, target sizes, grids, normalized patches, the rendered template and the token ids against the reference's own, which is the reproduction of a resizer to the byte that Chapter 1 demanded.

## Florence-2 answers with boxes

Every model so far reads a picture and writes a sentence. Florence-2 (Xiao et al., 2024) has a different shape, and its crate `crates/florence2` holds the smallest model of the eight crates, about 0.23 billion parameters. The tower is a DaViT (dual attention vision transformer), four stages of convolutional downsampling and window and channel attention, that turns a 768 by 768 picture (resized directly, not padded) into a 24 by 24 grid of 1,024 numbers. A learned two-axis position and a projection to 768 give 576 vision rows, plus one more that is their average:

<!-- worked: vlms.florence_tokens -->
The tokens of Florence-2-base: the picture is resized to 768 x 768 and cut by the tower into a 24 x 24 grid (24 per side, stride 32), so 576 tokens of width 1024; one more token is their average; a projection maps each to width 768. The 577 vision tokens come before the prompt tokens in the encoder input. The vocabulary is 50,265 + 1,024 = 51,289 ids: the base byte-level BPE, 24 task markers and the 1,000 location tokens. Parameters of the projection: 1024 x 768 + 2 x 768 = 787,968.
<!-- /worked -->

There is no placeholder and no splice. The 577 vision rows are concatenated in front of the embedded text of the prompt and the whole is the input of an encoder in the style of BART (bidirectional and auto-regressive transformers); the decoder is an ordinary autoregressive one with cross-attention to the encoder's output, six layers each, and it starts from `decoder_start_token_id` and writes the answer (`crates/florence2/src/text/lm.rs`). The picture and the instruction are read together, with full attention between them, which a decoder-only model with a causal mask cannot do for the rows that come before the question. What the model does is chosen by the prompt text: the processor maps a task name to a template, and `ground` in brain builds "Locate {target} in the image." from the template of open-vocabulary detection (the task marker itself never reaches the tokenizer; it only selects the template).

The answer to a grounding prompt is a box, and the decoder can only write tokens. So the vocabulary is extended with 1,000 **location tokens**, `<loc_0>` to `<loc_999>`, added by the processor at load time and not present in the checkpoint's `tokenizer.json`, 24 task markers and 1,000 location tokens on top of the base vocabulary's 50,265 ids (`crates/florence2/src/tokenizer.rs`). A coordinate is divided by the side it belongs to, multiplied by 1,000 and rounded down:

```text
function box_tokens(box: array<real>[4], w: int, h: int, bins: int) -> array<int>[4]:
    require w >= 1 and h >= 1 and bins >= 2, "a picture and at least two bins"
    out = zeros(int, [4])
    for k in 0 .. 4:
        s = w if k mod 2 == 0 else h                   # x1 and x2 use the width, y1 and y2 the height
        out[k] = min(bins - 1, max(0, floor(bins * box[k] / s)))
    return out
```

<!-- worked: vlms.location_table -->
A box in a 640 x 480 picture: corners (x1, y1) = (100, 60) and (x2, y2) = (340, 420) pixels. Each coordinate is divided by the side it belongs to, multiplied by 1000 and rounded down (at most 999); the four bins become four tokens after the phrase. Decoded as bin / 1000 of the side (the lower edge of the bin) or as (bin + 0.5) / 1000 (its centre):

| coordinate | pixels | 1000 x pixels / side | bin | token | decoded, lower edge | error | decoded, centre | error |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| x1 | 100 | 1000 x 100 / 640 = 156.250 | 156 | <loc_156> | 99.84 | 0.16 | 100.16 | 0.16 |
| y1 | 60 | 1000 x 60 / 480 = 125.000 | 125 | <loc_125> | 60.00 | 0.00 | 60.24 | 0.24 |
| x2 | 340 | 1000 x 340 / 640 = 531.250 | 531 | <loc_531> | 339.84 | 0.16 | 340.16 | 0.16 |
| y2 | 420 | 1000 x 420 / 480 = 875.000 | 875 | <loc_875> | 420.00 | 0.00 | 420.24 | 0.24 |

The text the decoder writes is `cat<loc_156><loc_125><loc_531><loc_875>`.
<!-- /worked -->

The error of a bin is the bin's width, and it depends on the picture and not on the model:

<!-- worked: vlms.bin_width_table -->
The width of one bin in pixels on a side of the given length, and the largest error of the centre decoding (half a bin; the lower-edge decoding errs by up to a whole bin).

| side (pixels) | pixels per bin | centre decoding, largest error |
|---|---:|---:|
| 384 | 0.384 | 0.192 |
| 640 | 0.640 | 0.320 |
| 768 | 0.768 | 0.384 |
| 1920 | 1.920 | 0.960 |
| 4000 | 4.000 | 2.000 |
<!-- /worked -->

On a 640-pixel side a bin is 0.64 pixels; on a 4,000-pixel side, 4. The scale is a decision about vocabulary: 1,000 bins cost 1,000 rows of the tied embedding table, $1000 \times 768 = 768{,}000$ parameters, and they put the resolution of the box in the representation, not in a regression head, which is also why a box is four tokens like any others in the output. Brain only reads them. `parse_boxes` in `crates/florence2/src/grounding.rs` scans the decoded text for runs of four tags, takes the text between the end of the previous run and the start of this one (trimmed) as its phrase, divides each bin by 1,000 (the lower edge of the bin, not its centre; the table shows the difference) and drops a trailing run of fewer than four, since a generation cut off in the middle of a box is a real case and not an error:

<!-- worked: vlms.parse_example -->
The text `a cat<loc_156><loc_125><loc_531><loc_875>the dog<loc_10><loc_20><loc_30><loc_40><loc_7><loc_8>` has two complete groups of four tags and a trailing pair, which is dropped (a generation cut short in the middle of a box).

| phrase | [x1, y1, x2, y2] as fractions of the sides |
|---|---|
| a cat | (0.156, 0.125, 0.531, 0.875) |
| the dog | (0.010, 0.020, 0.030, 0.040) |
<!-- /worked -->

## Two towers, and one decoder that draws

DeepSeek-VL's picture has to be seen at two scales at once, as a whole and in fine print, and one tower is a poor compromise. Its tower is hybrid (`crates/deepseekvl/src/tower.rs`): SAM-B, the segmentation encoder of Chapter 7, reads the padded square at 1024 pixels and its map is read as 576 rows of 1,024 numbers; SigLIP-L reads the same square resized to 384 and gives 576 rows of its own. The split aligner of the table above projects each stream to half of the decoder's width and concatenates the halves, high resolution first, so that the picture is 576 tokens each made of one row from each tower. Preprocessing has an unexpected rule, found in the checkpoint's own files: the processor's configuration lists the mean and standard deviation of CLIP and also says `do_normalize: false`, so the processor does not normalize at all. Each branch normalizes inside the tower with its own statistics, and the listed mean serves for one thing only, the colour of the padding, $\lfloor 255 \times \mathrm{mean} \rfloor$ per channel (the case of Chapter 1's section on constants).

Janus-Pro shares the decoder and the processor class with DeepSeek-VL, and `brain_arch::by_hf_config` tells them apart by the generation heads only Janus-Pro configures. Its understanding path is the same composite with one tower, SigLIP-L at 384 and a plain two-layer aligner, and each picture's 576 pad ids wrapped in `begin_of_image` and `end_of_image`. The other half of the model runs the picture through the door the other way: the decoder predicts image tokens, 576 of them (a 24 by 24 grid of codes from a vector-quantized (VQ) codebook of 16,384 codes), with its own head, each of which is embedded by a second aligner from an 8-wide code table back to the decoder's width, and the finished grid is decoded to pixels by the VQ decoder. That is generation, and it needs the machinery of Volume VIII: discrete image tokens, classifier-free guidance (the crate runs a conditional and an unconditional sequence together) and a decoder from codes to pixels. The point to take from it here is what it says about the splice: the same decoder reads pictures through one aligner and writes them through another, with the splice and its zeroing on the reading side only.

## Captioning behind a seam

Labelling a folder of pictures is not a property of one model. The work (walk the folder, ask a model to describe each picture, write a caption file a trainer can read, resume after an interruption) is the same whichever model does the describing, so `crates/captioner` has no model code and no model dependency, and exports a contract first. A `Clip` is an ordered run of frames with an optional frame rate, of which a still image is the one-frame case; a `Captioner` turns a clip and an instruction into text, and declares `Capabilities`, among them `max_frames`, which is the one fact `validate` uses to reject a 48-frame clip sent to an image captioner by name instead of silently describing the first frame. Commit c1c3e113a (August 2026) explains the unit: a video model captions a clip, not a frame, and a seam that passed one image would have forced it to reshape the trait or caption a thumbnail. Only the image path is built. The implementors are in the model crates, and in this repository they are LLaVA, Qwen3-VL and FastVLM (`captioner.rs` in each).

The folder labeller, `label_dir`, is resumable and idempotent: it re-reads the caption file before the run and rewrites it after each picture, so an interrupted run starts again where it stopped; a picture that already has a caption is left exactly as it is, so a caption corrected by hand survives a later run unless `overwrite` is set. The instruction is the whole of the prompt engineering: what the caption covers, how long it is, a trigger phrase for an adapter all live there and not in the model wrapper. The Qwen3-VL implementor lets the caller cap the pixel area, `with_max_pixels`, and its default of $1280 \times 1280$ gives a picture 1,600 tokens and the tower a score slab of 2.44 GiB, in the table above, because a labelling pass has one picture at a time and no other tenant to budget against, and detail in the caption is the point.

## The eight crates side by side

What each crate holds, read off its configuration and constructors:

<!-- worked: vlms.crate_towers -->
The tower, the projector and the token count at each model's native input, from the configurations and constructors of the crates. `Rows into the projector` is what the tower hands on (DeepSeek-VL's two towers each give 576, which the split aligner joins row by row; MiniCPM-V's window merger has already cut its 1,024 patches to 256); the last but one column is what the decoder reads for one picture (the encoder, for Florence-2, whose 577 is the 576 grid cells and one pooled token).

| crate | tower | native input | rows into the projector | projector | image tokens | decoder |
|---|---|---|---:|---|---:|---|
| llava | CLIP-L/14, 24 x 1024, tapped at the penultimate block, class token dropped | 336 x 336 | 576 | two layers 1024 -> 5120 -> 5120 | 576 | Vicuna-1.5-13B, 40 layers, width 5120, no grouping |
| qwen3vl | ViT 24 x 1024, two-axis RoPE, DeepStack taps after blocks 5, 11, 17 (from 0) | 448 x 448 | 784 | merge 2 x 2, then 4096 -> 4096 -> 2560 | 196 | Qwen3 4B preset, 36 layers, width 2560, 32 / 8 heads |
| minicpmv | NaViT SigLIP 27 x 1152, window merger after block 6 | 448 x 448 | 256 (of 1,024 patches) | merge 2 x 2 (MLP), after the window merger | 64 | Qwen3.5-0.8B, hybrid layers |
| moondream3 | SigLIP 27 x 1152 on 378-pixel crops | 378 x 378 | 729 | global and pooled local maps joined, 2304 -> 8192 -> 2048 | 729 | 24 layers, width 2048, sparse from layer 4 |
| deepseekvl | SAM-B at 1024 and SigLIP-L at 384, 24 x 24 each | 1024 and 384 | 2 x 576 | split: (1024, 1024) -> 2 x 2048 -> 4096 | 576 | Llama, 30 layers, width 4096 |
| januspro | SigLIP-L/16 at 384 | 384 x 384 | 576 | two layers 1024 -> 4096 -> 4096 | 576 | the same Llama decoder |
| florence2 | DaViT, four stages, 24 x 24 grid at 768 | 768 x 768 | 576 | one linear 1024 -> 768 and a norm | 577 | BART encoder-decoder, 6 + 6 layers, width 768 |
<!-- /worked -->

And how each fits a picture to its tower, enters its rows into the sequence and positions them:

<!-- worked: vlms.crate_entry -->
How each crate fits a picture to its tower, how the rows enter the sequence and what position they carry.

| crate | resolution policy | how the rows enter | positions of the rows |
|---|---|---|---|
| llava | shortest edge to 336, centre crop | id `-200`, expanded to 576 rows | one-dimensional |
| qwen3vl | dynamic: multiples of 32, area 56^2 to 3584^2 | `vision_start`, N pad ids, `vision_end` | multi-axis, sections 24, 20, 20 |
| minicpmv | source view and up to 9 slices of about 448 | `image` marker, pad ids, then a `slice` marker and pad run per slice | one-dimensional |
| moondream3 | global crop and up to 12 overlapping crops, pooled to 27 x 27 | `bos`, then 729 rows as a prefix attended bidirectionally | one-dimensional, 32 of 64 channels rotated |
| deepseekvl | pad to a square, 1024 and 384 | `image_placeholder` expanded to 576 pad ids | one-dimensional |
| januspro | pad to a square, 384 | `begin_of_image`, 576 pad ids, `end_of_image` | one-dimensional |
| florence2 | direct resize to 768 x 768 | no placeholder: the rows are the first of the encoder input | two learned axes on the rows, learned absolute positions in the encoder |
<!-- /worked -->

Two facts about the crates complete the table. `crates/captioner`, the eighth, holds no model, as the section above said. And the understanding tower of `crates/januspro` is `crates/deepseekvl`'s `SiglipTower`, not a tower of its own.

## Check your understanding

1. In the toy, row 1 of the embedding table holds $(0.5, 0.5, 0.5)$. Change it to $(100, 100, 100)$. What changes in the sequence the decoder reads, in the forward pass and in the gradient of the table? Then say what the zeroing of the picture rows in the backward pass prevents in a model with a tied head.
2. Find the error. "Dynamic resolution means the model accepts a picture of any size, so the preprocessing is the identity."
3. A 3000 by 2000 photograph is sent to LLaVA-1.5 and to Qwen3-VL at the released limits. Predict which model answers a question about a small sign at the edge of the picture and why, using the token counts of the tables, and say what each costs the decoder in prefill and cache.
4. A prompt has one picture of 20 by 30 merged tokens, preceded by 10 words and followed by 6. Give the position of the first word after the picture and of the third generated token under one-dimensional RoPE and under the multi-axis rule.
5. Find the error. "Florence-2 decodes a box coordinate as bin / 1000, so it is never off by more than half a bin."
6. DeepSeek-VL's processor lists CLIP's mean and standard deviation. Someone sets `do_normalize` to true to follow the list. What is wrong with the pixels each branch then sees?
7. A window merger after block 6 saves 51 percent of the tower's FLOPs at 448 by 448. Why does merging earlier save more of the attention products than of the matrix products, and what does the model give up?

## Summary

A language model reads rows, not ids, and a row computed from a picture is as good as one from the embedding table. Giving it eyes is therefore a splice: a projector maps the tower's rows to the decoder's width, the placeholder ids of the prompt are expanded to as many as there are rows, and the rows overwrite them, with a backward pass that carries the gradient to the projector and zeroes it where it came from, so that the overwritten table row learns nothing. The projector is small, seventeen to thirty-six million parameters against billions, and the interesting ones also shrink the sequence, by folding a 2 by 2 block of patches into one row, in the tower or after it. The number of rows is the cost: every row is a row of prefill and of cache. A fixed square makes it predictable and loses detail, slices bound it in steps, and dynamic resolution lets it follow the area through a rounding rule, derived above, that preserves the shape and bounds the area from both sides, at the price of costs that grow with the pixels, in the tower faster than in the decoder. The positions of the rows are one-dimensional in most crates and a triple in Qwen3-VL, where the counter advances by the largest extent of the picture and every generated token must be given the position that follows, not its row. Florence-2 changes the output side instead: an encoder-decoder with task prompts, whose boxes are four of a thousand location tokens.

Some limits remain. The toy has a tower that is one matrix, and the real ones were counted, not trained; which design gives the better answers at a fixed budget is not decided by any number here, and the training recipe is the papers', not a result of this repository. What carries forward is the token budget. Chapter 10 asks how few tokens a page of text can be squeezed into, and its models are the ones above pushed to the point where the decoder reads a page through a few hundred rows.

## References

- H. Liu, C. Li, Q. Wu and Y. J. Lee, *Visual instruction tuning*, NeurIPS 2023. LLaVA: a vision tower, a projector and a language model, trained in two stages.
- P. Wang et al., *Qwen2-VL: enhancing vision-language model's perception of the world at any resolution*, 2024. Dynamic resolution with a variable number of tokens, and the multi-axis rotary embedding of Volume V, Chapter 8.
- B. Xiao et al., *Florence-2: advancing a unified representation for a variety of vision tasks*, CVPR 2024. The encoder-decoder with task prompts and location tokens.
- X. Chen et al., *Janus-Pro: unified multimodal understanding and generation with data and model scaling*, 2025. Separate understanding and generation paths over one decoder.
- J.-B. Alayrac et al., *Flamingo: a visual language model for few-shot learning*, NeurIPS 2022. The perceiver resampler.
- S. Antol et al., *VQA: visual question answering*, ICCV 2015. The question answering task and its human-agreement accuracy.

## Exercises

1. **[Calculate]** In the toy, compute by hand the row at position 3 (the picture's third patch, whose pixels are $(1.0, 0.0, 1.0, 1.0)$) from $\mathbf{W}_v$, $\mathbf{W}_p$ and $\mathbf{b}_p$, and check it against the table of the sequence.
2. **[Calculate]** Give `smart_resize` of a 720 by 1280 picture at factor 32 and the released limits, and its token count. Then do the same for 1000 by 1000 at the limit of 1,000,000 pixels and say how many tokens that is.
3. **[Calculate]** Run the slicing rule by hand for a 1500 by 1500 picture: the target number of slices, the candidate grids and the winner.
4. **[Calculate]** A picture 1000 pixels wide and 750 high holds a box with corners $(50, 75)$ and $(951, 700)$. Give its four location bins, the decoded corners at the lower edge of the bin and at its centre, and the largest error of each.
5. **[Calculate]** A picture becomes a grid of 56 by 40 merged tokens. By how many positions does one-dimensional RoPE push the following text further than the multi-axis rule does, and how many rows of cache does the picture occupy?
6. **[Implement]** Reproduce `vlms.splice_setup`, `vlms.splice_sequence` and `vlms.splice_backward`: the patches, the tower's embedding, the projector, the spliced sequence and the gradient of the table with and without the zeroing.
7. **[Implement]** Reproduce `vlms.smart_resize_branches`, `vlms.smart_resize_table` and `vlms.smart_resize_halves` from the pseudocode, with the rounding of ties to the even multiple.
8. **[Implement]** Reproduce `vlms.slicing_candidates` and `vlms.slicing_table` (the grid rule, the source view, the refined size and the token counts at 16x and 4x) and `vlms.overlap_crop_table`.
9. **[Implement]** Reproduce `vlms.location_table` and `vlms.parse_example`: quantize a box, decode it both ways and parse a text with a trailing partial box.
10. **[Investigate]** Until commit `1183350f8`, brain sized a resident's per-image visual capacity as the token count of a square at the pixel budget, on the assumption that the square is the most token-hungry shape. For a budget of 1,000,000 pixels the rule of the chapter gives the square 961 tokens. Find a picture shape that gets more, say what the capacity check of `generate` did with eight such pictures, and give the bound the capacity uses now.
11. **[Investigate]** Remove the clamp to one factor from the shrinking branch of your `smart_resize`. Find the smallest aspect ratio, at the released limits, for which it then returns a side of zero, and compare it with the ratio of 200 that the reference refuses beyond. What should a guard at the entry of the image function do instead?
12. **[Investigate]** Move the window merger of the MiniCPM-V tower to every block from 0 to 26 and plot the saving of `vlms.window_merger_saving` against the position. Is it linear, where does the merger stop paying for itself, and what does a merger placed at block 0 give up that one placed at block 6 does not?

## Key terms

- **vision projector**: the small trained map, a linear layer or a pair of them around a GELU, sometimes folding a block of neighbouring patches into one row, that takes the rows of a vision tower to the width of a language model's embedding rows so that they can be read like the rows of words. <!-- concept: vision-projector -->
- **image token splicing**: replacing, in the embedding rows of a prompt, the rows at the placeholder ids by the projector's rows computed from a picture, in order, with a backward pass that carries their gradient to the projector and zeroes them where they were. <!-- concept: image-token-splicing -->
- **dynamic resolution**: letting the grid of a picture, and so its number of tokens, follow its size and shape instead of resizing every picture to one fixed square: each side is rounded to a multiple of the patch times the merge, and the area is bounded from both ends. <!-- concept: dynamic-resolution -->
- **multimodal position encoding**: the rule that gives the tokens of a sequence of words and picture rows their positions, such as the index of each row, or a triple of frame, row and column for a patch with a counter that advances past the picture by its largest extent. <!-- concept: multimodal-position-encoding -->

## Answers

**Recall questions.**

1. $2 \times 36 \times 8 \times 128 \times 2 = 147{,}456$ bytes: keys and values, 36 layers, 8 heads, 128 numbers, 2 bytes.
2. $28 \times 28 = 784$ tokens in the tower and $784 / 4 = 196$ after the merge.
3. $(c + t, c + h, c + w) = (5 + 0, 5 + 1, 5 + 2) = (5, 6, 7)$.
4. On the one pooled vector (the class token or the mean of the rows, after the final norm and projection), which is pushed towards the caption's vector. The loss says nothing about the patch rows as such, which is a reason to read an earlier block.
5. It is an id the tokenizer matches whole, added to the vocabulary, that no merging of bytes produces, so ordinary text cannot make it by accident, and a user cannot spell it unless the text is allowed to.
6. No: it turns queries and keys inside attention, and the embedding rows carry no position. 32 / 8 = 4 query heads share one key-value head.

**Check your understanding.**

1. Nothing changes in the sequence the decoder reads: the four positions of the placeholder are overwritten, and no other position holds id 1. Without the zeroing the table's row 1 would receive the four picture rows' gradients whatever its value; with it the row's gradient is zero. In a model whose output head is tied to the table, that row is also the head's row for the placeholder id, so a gradient meant for the projector would change the score of generating that token.
2. The grid must be a multiple of the patch times the merge, so both sides are rounded (to 32 in Qwen3-VL) and the picture is resampled to the result; and the area is bounded, so a large picture is shrunk and a tiny one enlarged. A model that took the picture unchanged would have a grid that does not tile, and a token count without an upper limit.
3. Qwen3-VL gets 5,828 tokens from a 3008 by 1984 grid and LLaVA 576 from a centre crop of a picture resized so that its short side is 336, about a tenth as many tokens, and the sign at the edge may not even be in the crop. Qwen3-VL's prefill added is 52.45 trillion FLOPs and the cache 819.6 MiB against LLaVA's 14.76 trillion and 450 MiB, whose decoder is three times the size and has five times the key-value heads, so the dynamic design is not 10 times dearer but 3.6 times in FLOPs and 1.8 in cache; and the tower's attention adds another 53.4 trillion on top.
4. One-dimensional: the first word after the picture is at position $10 + 600 = 610$ (positions 0 to 9 for the words, 10 to 609 for the 600 rows), and the third generated token is at $610 + 6 + 2 = 618$. Multi-axis: the counter is 10 before the picture and advances by $\max(1, 20, 30) = 30$, so the first word after is at 40, and the third generated token at $40 + 6 + 2 = 48$.
5. The lower-edge decoding reads the bin's lower end, while the coordinate can be anywhere in the bin, so its largest error is a whole bin (0.64 pixels at a width of 640); half a bin is the largest error of decoding at the centre of the bin.
6. Both branches would be fed pixels normalized with CLIP's statistics, so SigLIP, which was trained under mean and standard deviation of 0.5, would see wrongly scaled numbers, and SAM would see a differently coloured border. Shapes would agree and every number would be off, the silent kind of error of Chapter 1.
7. The score products scale with the square of the rows, so quartering the rows of the later blocks cuts their attention sixteen-fold and their matrix products fourfold; the saving is larger for attention. What the model gives up is detail: the later blocks see only the merged windows, and the window merger is a component of 104 million parameters that must itself learn to keep what the later blocks need.

**Exercises.** 1. Pixels $(1.0, 0.0, 1.0, 1.0)$ give $\mathbf{f} = (0.5(1 + 0 + 1 + 1), 1 - 0 + 1 - 1) = (1.5, 1.0)$ and $\mathbf{p} = (2 \cdot 1.5 + 0.1, 2 \cdot 1.0 - 0.1, 1.5 + 1.0) = (3.1, 1.9, 2.5)$, the row at position 3. 2. $720 / 32 = 22.5$ goes to 22, so 704 by 1280, area 901,120, inside the bounds, $22 \times 40 = 880$ tokens. For 1000 by 1000 at 1,000,000: $1000/32 = 31.25$ gives 992 by 992, area 984,064, inside, $31 \times 31 = 961$ tokens. 3. $1500^2 / 448^2 = 11.2$ so the target is $\min(12, 9) = 9$; candidates are 8, 9 and 10 slices, and 10 is above the maximum; the grid 3 by 3 has aspect error 0 against a square and wins. 4. Bins: $x_1 = \lfloor 1000 \times 50 / 1000 \rfloor = 50$, $y_1 = \lfloor 1000 \times 75 / 750 \rfloor = 100$, $x_2 = 951$, $y_2 = \lfloor 933.33 \rfloor = 933$. Lower-edge decoding: $50$, $75$, $951$, $699.75$ pixels, errors $0, 0, 0, 0.25$ (the largest possible is a bin, 1 pixel in x and 0.75 in y); centre decoding: $50.5$, $75.375$, $951.5$, $700.125$, errors $0.5, 0.375, 0.5, 0.125$ (at most 0.5 and 0.375). 5. One-dimensional RoPE advances by $56 \times 40 = 2{,}240$ and the multi-axis rule by 56, so the text is $2{,}184$ positions further on; the cache holds 2,240 rows. 10. At 1,000,000 pixels 800 by 1248 rounds to 800 by 1248, area 998,400, inside the bound, and $25 \times 39 = 975$ tokens, more than the square's 961; `generate` compared the request's total with the resident's capacity (961 times the eight images allowed, 7,688) and refused eight such pictures (7,800 tokens) with the message that names the capacity, although each was within the budget: a refusal at the edge, not a corrupted buffer. Every picture sized within the budget is a grid of 32-pixel cells, one token each, so the capacity is now $\lfloor 1{,}000{,}000 / 32^2 \rfloor = 976$ tokens an image. 11. Without the clamp the height becomes zero when the rounded area is over the maximum and $h / \beta < 32$, that is $\sqrt{N_{\max} h / w} < 32$, or $w / h > N_{\max} / 32^2 = 12{,}544$; and for the rounded area to pass the maximum with a height of 32 the width must round to at least 401,440. The first example is 32 by 401,440 (ratio 12,545), which returns $(0, 401{,}408)$, and 31 by 600,000 does the same, $(0, 498{,}592)$. The reference's limit of 200 is far below that, so the image function accepts shapes the reference would refuse. Either guard closes the hole: the clamp keeps every side at one factor and the request is still answered, and a check at the entry refuses ratios over 200 with an error that names the ratio, as `require` does for video. Neither may let a zero reach the tower, where it becomes an empty patch grid. 12. The saving is nearly linear in the position, falling by about 2.9 points per block: 68.3 percent for a merger after block 0, 51.1 after block 6, 10.9 after block 20, and $-6.3$ after block 26, where the merger costs more than the nothing it saves. A merger at block 0 merges rows that have not yet exchanged any information, while one at block 6 merges rows that have attended to the whole picture seven times.
