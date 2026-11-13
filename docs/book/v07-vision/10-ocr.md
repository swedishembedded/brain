<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Reading Documents

*How does a model read a page of text from pixels, and how few tokens can a page be squeezed into?*

A page of paper scanned at 200 pixels to the inch is 1,654 pixels wide and 2,339 high. Set 930 words on it in letters 15 pixels tall, a dense page, and every tool of the last chapter is ready to read it: a tower that turns the picture into rows, a projector that gives them the decoder's width, a splice that puts them in the prompt. The tower wants a square of 1,024 pixels. Fitting the page into that square scales it by $1024/2339 = 0.4378$, and the letters come out 6.6 pixels tall. Nothing downstream can repair what the resize threw away, and a decoder that is asked to read the result writes fluent text that is not on the page.

The other way round is no better. Give the tower the page at its own pixels and it sees a grid of 103 by 146 patches, 15,038 tokens, and each global attention block of the tower scores their $15{,}038^2$, about 226 million pairs, 13.5 times what a block on the 64 by 64 grid of a 1,024-pixel square scores, and the decoder that follows would have to hold those 15,038 rows in its cache for every page.

<!-- worked: ocr.native_patches -->
The patch grid of a page at its own pixels (whole patches of 16 pixels), the query-key pairs of one global block over it, and that count against the 4,096 x 4,096 = 16,777,216 pairs of a global block of the 64 x 64 grid of the global view.

| page | patch grid | patches | pairs, one global block | against the 64 x 64 grid |
|---|---:|---:|---:|---:|
| 1654 x 2339 | 103 x 146 | 15,038 | 226,141,444 | 13.5 |
| 2480 x 3508 | 155 x 219 | 33,945 | 1,152,263,025 | 68.7 |
<!-- /worked -->

A reader of documents has to square two demands that pull in opposite directions: the letters must reach the model large enough to be told apart, and the pages must reach the decoder in few enough rows to be affordable. This chapter is about the arithmetic of that bargain and about the three crates in brain that strike it: `crates/deepseek2ocr`, a document reader whose vision half is SAM followed by a 16-fold compressor and CLIP (the first generation, v1 in the tables); `crates/deepseekocr2`, its successor, which replaces CLIP by a stack of learned queries (the second generation, v2); and `crates/ocrserve`, which turns either of them into a server that decodes many pages at once.

Reading text from an image reuses six earlier ideas, recalled first.

1. ([Volume VII, Chapter 9](09-vlms.md), projector.) A tower writes rows of width 1,024 and the decoder's rows are 1,280 wide. How many parameters has the smallest map that joins them, with a bias?
2. ([Volume VII, Chapter 9](09-vlms.md), splicing.) A prompt has a placeholder that was expanded to 273 ids, and the picture's tower produces 256 rows. What does the splice require, and what does that leave unexplained?
3. ([Volume VI, Chapter 3](../v06-language-models/03-attention-variants.md), sliding windows.) In a causal window of $w$ tokens, how many keys does a query read, and how far back can information travel through $L$ such layers?
4. ([Volume VI, Chapter 3](../v06-language-models/03-attention-variants.md), latent attention.) A decoder layer has 10 heads of width 128. How many numbers does it cache per token with ordinary multi-head attention, and how many with a latent of 512 and a rotary key of 64?
5. ([Volume VII, Chapter 2](02-convolution.md), convolution.) A convolution of kernel 3, stride 2 and padding 1 reads a map that is 64 wide. How wide is its output?
6. ([Volume VII, Chapter 1](01-images.md), shrinking.) A row of stems that repeat every 4 pixels is shrunk by a factor of 4 with an area average. What does the output look like, and what does shrinking by 2.3 do instead?

Everything computed in this chapter is arithmetic on printed constants: no random number is drawn, so no seed is involved, and the numbers that come from trained models are quoted from commits with their setup.

## Text that does not survive the resize

**Optical character recognition** is the task of turning the pixels of a page into the characters that were printed on it. A classical engine does it in stages, finding lines, cutting them into characters or short runs and classifying each piece. The models of this chapter skip the stages. A vision tower makes rows from the page, and a decoder, the same kind of decoder as in Volume VI, writes the page out as text while it attends to those rows. It is the vision-language model of Chapter 9 with one change of workload: the answer is long, it is all in the picture, and every character of it must be right.

That workload decides what the picture has to be. A stem of a letter is a thin stroke, and whether it is still there after a resize is a question of sampling. Take a row of stems that repeat with period $p$ pixels, ink for half of each period. Shrinking the page by a scale $s < 1$ with an area average (the second method of Chapter 1's table) makes each output pixel the mean of the $1/s$ source pixels it covers. The output repeats every $ps$ pixels, and a pattern can only be told apart if that period is more than 2 output pixels, the Nyquist limit of Volume I, Chapter 14. Below it the stems alias into a slower pattern that was never on the page, as the stripes of Chapter 1 did. Above it the stems survive, with a contrast that the width of the average takes away.

<!-- worked: ocr.legibility -->
A row of stems (ink for half of every period, ink = 1) is averaged over the 1 / scale input pixels that each output pixel covers (an exact area average, not the antialiased bicubic of the crates), and 200 output pixels are taken, output pixel k covering the input interval from k / scale to (k + 1) / scale. 'Period out' is the period times the scale; the stems can only be told apart when it exceeds 2 output pixels (the Nyquist limit of Volume I, Chapter 14). 'Swing' is the largest output minus the smallest.

| text seen at | scale | box width | period out, 4 px | swing | above 2 | period out, 7.5 px | swing | above 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native pixels | 1.0000 | 1.000 | 4.00 | 1.000 | yes | 7.50 | 1.000 | yes |
| v2 tile mosaic, across | 0.9287 | 1.077 | 3.71 | 1.000 | yes | 6.96 | 1.000 | yes |
| v2 tile mosaic, down | 0.9850 | 1.015 | 3.94 | 1.000 | yes | 7.39 | 1.000 | yes |
| v1 tile mosaic, across | 0.7739 | 1.292 | 3.10 | 1.000 | yes | 5.80 | 1.000 | yes |
| v1 tile mosaic, down | 0.8209 | 1.218 | 3.28 | 1.000 | yes | 6.16 | 1.000 | yes |
| global view, 1654 x 2339 | 0.4378 | 2.284 | 1.75 | 0.751 | no | 3.28 | 1.000 | yes |
| global view, 2480 x 3508 | 0.2919 | 3.426 | 1.17 | 0.168 | no | 2.19 | 1.000 | yes |
<!-- /worked -->

The page of the opening, seen through the global view (the sixth row of the table, scale 0.4378), has stems 4 pixels apart arriving every 1.75 output pixels. That is below 2, and the output swing, 0.751, says only that something remains, not that it is the right thing. Letters that are 7.5 pixels apart survive the same shrinking, with a period of 3.28. The two scans show how little room there is: at 300 pixels to the inch the same page is 3,508 pixels high and the global view scales it by 0.2919, and even letters 7.5 pixels apart are down to a period of 2.19, with the denser stems far gone. The tile readings of the next sections keep the page at 0.77 to 0.82 across and down for the first generation and at 0.93 and 0.985 for the second, and in every tile row of the table both periods are well above 2.

The sampling limit is the necessary part of legibility, not the sufficient one. A period of 3 output pixels is enough to tell the stems apart and is not enough for a model that was trained on text of 10 or 12 pixels to read them. Where the encoder stops reading is an empirical fact about its training and was found by running it: a commit of October 2026, `24d2e1f44`, reports the word error rate of the first generation on a dense synthetic page falling from 0.884 to 0.520 when the page is read as tiles instead of as the one 1,024-pixel view, and the second generation's from 1.000 to 0.174. The word error rate is an edit distance, and a later section defines it and counts it. First the machinery that makes the tiles possible.

## A page in 256 rows

The vision half of the first generation, which brain's crates call the DeepEncoder, has four stages, and the previous chapters have supplied most of them.

The first is the tower of Chapter 7. A 1,024-pixel view is cut into patches of 16 and embedded, which gives a grid of 64 by 64, 4,096 tokens of width 768, and twelve blocks of attention run on them: eight over windows of 14 by 14 and four over the whole grid (blocks 2, 5, 8 and 11). The crate is `crates/sam1`, the same one that Chapter 7 used for segmentation, and the document reader takes its output after the neck and the compressor, which the crate holds as well.

The second stage is new: a **compressor** of two convolutions of kernel 3, stride 2 and padding 1, from 256 channels to 512 and then to 1,024. By the output-size formula of Chapter 2 each convolution halves the side, rounding up, so the grid of 64 by 64 becomes 32 by 32 and then 16 by 16.

<!-- worked: ocr.grids -->
Patches of 16 pixels (a convolution of kernel 16 and stride 16), then two convolutions of kernel 3, stride 2, padding 1, whose output side is `out_size` of Volume VII, Chapter 2: (n + 2 - 3) // 2 + 1. The last column is the window partition of the SAM blocks, which pads the patch grid on the bottom and the right to a multiple of 14.

| view | pixels | patch grid | after conv 1 | after conv 2 (tokens) | windows of a windowed block |
|---|---|---|---|---|---|
| global view | 1024 x 1024 | 64 x 64 = 4,096 | 32 x 32 | 16 x 16 = 256 | 5 x 5 of 14 x 14 (padded to 70) |
| v1 tile | 640 x 640 | 40 x 40 = 1,600 | 20 x 20 | 10 x 10 = 100 | 3 x 3 of 14 x 14 (padded to 42) |
| v2 tile | 768 x 768 | 48 x 48 = 2,304 | 24 x 24 | 12 x 12 = 144 | 4 x 4 of 14 x 14 (padded to 56) |
<!-- /worked -->

Two strides of 2 in each of two directions reduce 4,096 positions to 256. That is the 16-fold compression of the crates' own description of the encoder, and it happens before any stage whose cost grows with the number of tokens. The 256 tokens have width 1,024.

The third stage is CLIP's large vision tower, the 24-block tower of Chapter 8, with its patch embedding bypassed: the 256 compressor tokens stand where patch embeddings would, a class row is put in front, the learned position table is resampled to the grid (the identity at 16 by 16, which is the native grid of that tower at its own resolution), and the blocks run with full attention over the 257 rows. The class row is dropped on the way out. The fourth is the projector, a linear map from 2,048 numbers to 1,280. The 2,048 are the 1,024 of CLIP's output for a token concatenated with the 1,024 of the compressor's own output for the same token, in that order, so the projector sees each token both before and after global attention. The crate's tests do not say why the checkpoint is built that way, and this chapter does not claim to know; what they do show is that the order is not a matter of taste, as the next subsection shows.

![The vision half of the two generations. Top, the first generation: a 1,024-pixel view goes into SAM ViT-B (4,096 patches, 8 windowed and 4 global blocks), then a compressor of two strided convolutions (256 tokens of width 1,024), then CLIP-L over 257 rows; the compressor's tokens also go straight to the concat, which is projected from 2,048 to 1,280 numbers, and the image block is the 256 rows plus 16 newlines and a separator, 273 rows. Bottom, the second generation: the same SAM and compressor (width 896), then 24 blocks over the 256 tokens and 256 learned queries under a prefix mask, the query rows kept and projected from 896 to 1,280 numbers, and an image block of 256 rows and a separator, 257 rows.](img/deepencoder.svg)

### A test that could not see the order

At the real scale both halves of the concatenation are 1,024 wide, so swapping them is a change that no shape check can detect and that changes every number. A reference written from scratch for the crate in August 2026 (commit `246a9d189`) therefore ran at dimensions that coincide nowhere: the compressor emits 11 channels where CLIP is 14 wide, and the token grid is 4 by 2 so that a transposed flatten cannot cancel. The crate's `DeepseekOcrConfig::tiny` (`crates/deepseek2ocr/src/config.rs`) is that fixture, and the real-scale check of the same file refuses it as not real-scale shaped. The lesson is general and cheap: when a model's real shapes make two mistakes look alike, test at shapes that tell them apart.

### What the compressor buys

Count what the compression is worth with and without it, on the global view, at the counting rules of the previous chapters.

<!-- worked: ocr.attention_pairs -->
Pairs of one query and one key in the attention of a SAM block. A windowed block's pairs are the windows of the padded grid times `14^2 x 14^2` (padding rows included, as in Chapter 7); a global block's are `tokens x tokens`. The tower has 8 windowed blocks and 4 global ones (blocks 2, 5, 8 and 11), and the last two columns compare its total with a tower of 12 global blocks.

| view | patches | pairs, windowed | pairs, global | global / windowed | pairs, SAM tower | pairs, 12 global | share (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| global view | 4,096 | 960,400 | 16,777,216 | 17.5 | 74,792,064 | 201,326,592 | 37.1 |
| v1 tile | 1,600 | 345,744 | 2,560,000 | 7.4 | 13,005,952 | 30,720,000 | 42.3 |
| v2 tile | 2,304 | 614,656 | 5,308,416 | 8.6 | 26,150,912 | 63,700,992 | 41.1 |
<!-- /worked -->

Windowed blocks are cheap for the reason Volume VI gave for windows: a query reads a fixed number of keys. The difference from the sliding window of that chapter is that these windows do not slide. The grid is cut into windows of 14 by 14 and each window attends only to itself, so within the twelve blocks a token has no path to its neighbour across a window boundary except through the four global blocks, and the $L(w - 1) + 1$ reach of $L$ stacked sliding windows does not apply: eight windowed blocks in a row would still leave a token blind to the next window over. The four global blocks are what make the tower see the page, and they are most of the cost: 37.1 percent of the pairs of a tower of twelve global blocks, spent on a third of the blocks.

<!-- worked: ocr.encoder_flops -->
GFLOPs of the vision half of one view (2 FLOPs per multiply-add; layer norms, activations, biases and the SAM relative-position bias left out). SAM, counted as in Chapter 7: patch embedding of 768 outputs per 16 x 16 x 3 patch; the query, key, value and output products of a windowed block over the padded grid and of a global block over the real tokens, the feed-forward over the real tokens (d = 768, f = 3072), and `4 d` per attention pair (the pairs of the previous table); the neck is a 1 x 1 convolution 768 to 256 and a 3 x 3 convolution 256 to 256; the compressor is a 3 x 3 convolution 256 to 512 and a 3 x 3 convolution 512 to 1024 (896 in v2) at stride 2. CLIP-L: 24 blocks of width 1024 and 4096 over the tokens and a class row. Projector v1: 2048 to 1280. Resampler: 24 blocks of width 896, 14 heads of 64, 2 key-value heads, feed-forward 4864, over twice the tokens (the tokens and as many queries), attention pairs of the prefix mask, projector 896 to 1280. The width of the first convolution of the compressor in v2 (512) is assumed equal to v1's.

| stage | v1, 1024 px | v1, 640 px | v2, 1024 px | v2, 768 px |
|---|---:|---:|---:|---:|
| patch embedding | 4.8 | 1.9 | 4.8 | 2.7 |
| SAM blocks, products | 726.1 | 278.0 | 726.1 | 422.8 |
| SAM blocks, attention | 229.8 | 40.0 | 229.8 | 80.3 |
| neck | 6.4 | 2.5 | 6.4 | 3.6 |
| compressor | 4.8 | 1.9 | 4.5 | 2.5 |
| CLIP-L | 161.7 | 62.0 | - | - |
| resampler | - | - | 380.5 | 210.6 |
| projector | 1.3 | 0.5 | 0.6 | 0.3 |
| total | 1,135.1 | 386.8 | 1,352.8 | 722.9 |
<!-- /worked -->

SAM with its neck and compressor is 971.9 of the 1,135.1 billion FLOPs of one global view, and CLIP, which does the global reading of the compressed tokens, is 161.7. The compressor itself, 4.8 billion, is among the cheapest stages of the encoder, and its payoff is in what follows it.

<!-- worked: ocr.compression_payoff -->
The global view of the first generation, with its 4,096 patch tokens given to CLIP-L and then to the decoder as they are (a count, not a model that exists: its position table and weights would have to be trained for it) and as the compressor gives them. CLIP's attention pairs are those of one block over the tokens and the class row; the decoder's rows are the prompt: 1 BOS row, the image rows and the 9 tokens of the instruction. The image rows of the compressed view are its 256 tokens plus the 16 learned newline rows and the learned separator row, 273, so its prompt is 1 + 273 + 9 = 283 tokens while the tokens column says 256; the hypothetical view has no newline or separator rows, 4,096 image rows and a prompt of 1 + 4,096 + 9 = 4,106.

| variant | tokens | pairs, one CLIP block | CLIP TFLOPs | prompt tokens | decoder prefill TFLOPs | cache (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| patches straight on (hypothetical) | 4,096 | 16,785,409 | 4.12 | 4,106 | 3.87 | 240.6 |
| after the 16x compressor | 256 | 66,049 | 0.16 | 283 | 0.23 | 16.6 |
<!-- /worked -->

This is **vision token compression**: reducing the number of tokens that stand for a picture before the stages whose cost grows with that number, which are the quadratic attention of a global tower, the prefill of the decoder and the cache that the decoder keeps for the rest of the page's life. The compressor does it by a fixed convolution of 16 positions into 1, and the choice of the reading, tiles or no tiles, decides how many compressed tokens a page gets. The arithmetic above says the compression is worth a factor of about 26 in the global tower, about 17 in the decoder's prefill and about 14.5 in the cache. It does not say what is lost, because the arithmetic of a count cannot: the compressor is trained, and what its 1,024 channels keep of a 64-pixel block of the page is a property of its weights. That is what the tiles of the next sections are for. A 640-pixel tile is read at a scale near 0.8, so that a token that stands for 64 pixels of the view stands for about 80 pixels of the page, not 146.

## Rows that remember a grid

The splice of Chapter 9 requires one picture row for every placeholder, and a picture of 16 by 16 tokens has 256. The global view takes 273. The 17 extra rows are not tokens of the page; they are what lets a decoder that reads one long sequence recover a grid of 16 columns.

The problem is the one that Chapter 9 solved for Qwen3-VL by changing the positions: flatten a grid of tokens row by row into a sequence and the sequence no longer says where one row ends. The decoder of this chapter's crates does not carry a triple of positions. It is a standard decoder with a one-dimensional rotary embedding, so the width of the picture has to be written into the sequence itself. The checkpoint ships two learned vectors for this: a **newline row**, one vector of the decoder's width appended after each row of 16 tokens, and a **separator row**, one vector that ends the image block. The block of the global view is therefore $g(g + 1) + 1$ rows for $g = 16$ tokens per side: $16 \times 17 + 1 = 273$.

The grid of the page is an instance of what this chapter calls a **document layout**. By document layout it means the two-dimensional arrangement of a page's content, lines of text in columns, tables and their cells, headings above bodies, and with it the order in which a reader takes those parts. A page's layout has to survive twice: once on the way in, from a grid of pixels to a sequence of rows, and again on the way out, from a sequence of text tokens to a transcript whose order and structure are right. The newline and separator rows are the way in. The reserved tokens of the transcript, which a later section shows, are the way out.

Nothing in the prompt marks those rows as different. In the token ids every row of the block is the same placeholder, `<image>`, and what differs is the vector written over it (`crates/deepseek2ocr/src/prompt.rs` says so, and checks the real vocabulary for a newline or separator id that is not there). A row table is built from the encoder's output, the newline vector and the separator vector, and the block is a gather from it, so the same machinery of Chapter 9 fills it. Its adjoint is the one the chapter taught for a broadcast parameter: each projector row is read once, so its gradient is a gather, and the newline vector, read 16 times, receives the sum of 16 gradients (`crates/deepseek2ocr/src/layout.rs`).

### Tiles in the sequence

A page read as a mosaic of tiles needs more layout, because the tiles are read side by side. The rule, written out as a routine, has three parts: the tiles of one row of the mosaic are interleaved by token row, so that the first row of the block is row 0 of tile 0, then row 0 of tile 1, and so on across the page, followed by one newline; then the global view, a newline after each of its rows; then one separator.

```text
function row_plan(g: int, gt: int, tc: int, tr: int) -> list:
    # g tokens per side of the global view, gt per side of a tile,
    # tc columns and tr rows of tiles (tc = tr = 0: global view only).
    # View 0 is the global view; view 1 + k is the k-th tile, row-major.
    rows = empty list
    for ty in 0 .. tr:
        for y in 0 .. gt:
            for tx in 0 .. tc:
                view = 1 + ty * tc + tx
                for x in 0 .. gt:
                    append ("P", view, y * gt + x) to rows
            append "N" to rows
    for y in 0 .. g:
        for x in 0 .. g:
            append ("P", 0, y * g + x) to rows
        append "N" to rows
    append "S" to rows
    return rows
```

Its length is $g(g + 1) + 1 + T_{\mathrm{h}}(g_{\mathrm{t}} T_{\mathrm{w}} + 1)g_{\mathrm{t}}$ for $T_{\mathrm{w}}$ tiles across and $T_{\mathrm{h}}$ down. A toy with a global view of 3 by 3 tokens and two tiles of 2 by 2 in one row of the mosaic has every kind of row in 23 positions.

<!-- worked: ocr.toy_layout -->
A page read as a mosaic of 2 tiles in one row (2 columns, 1 row), each tile 2 x 2 tokens, plus a global view of 3 x 3 tokens. The projector rows are numbered as the encoder produces them: the 9 rows of the global view first (0 to 8), then 4 rows for tile 0 (9 to 12) and 4 for tile 1 (13 to 16). The image block has 3 x (3 + 1) + 1 = 13 rows for the global view and (2 x 2 + 1) x 2 = 10 for the mosaic, 23 in all.

| row of the block | what stands there | projector row | from | token of that view |
|---|---:|---:|---:|---:|
| 0 | projector row | 9 | tile 0 | 0 |
| 1 | projector row | 10 | tile 0 | 1 |
| 2 | projector row | 13 | tile 1 | 0 |
| 3 | projector row | 14 | tile 1 | 1 |
| 4 | newline | - | - | - |
| 5 | projector row | 11 | tile 0 | 2 |
| 6 | projector row | 12 | tile 0 | 3 |
| 7 | projector row | 15 | tile 1 | 2 |
| 8 | projector row | 16 | tile 1 | 3 |
| 9 | newline | - | - | - |
| 10 | projector row | 0 | global view | 0 |
| 11 | projector row | 1 | global view | 1 |
| 12 | projector row | 2 | global view | 2 |
| 13 | newline | - | - | - |
| 14 | projector row | 3 | global view | 3 |
| 15 | projector row | 4 | global view | 4 |
| 16 | projector row | 5 | global view | 5 |
| 17 | newline | - | - | - |
| 18 | projector row | 6 | global view | 6 |
| 19 | projector row | 7 | global view | 7 |
| 20 | projector row | 8 | global view | 8 |
| 21 | newline | - | - | - |
| 22 | separator | - | - | - |
<!-- /worked -->

<!-- worked: ocr.toy_runs_text -->
Contiguous runs of the toy block, written (row of the block, first projector row, length): (0, 9, 2), (2, 13, 2), (5, 11, 2), (7, 15, 2), (10, 0, 3), (14, 3, 3), (18, 6, 3). There are 7 of them: a newline breaks a run on the block's side and a change of tile breaks it on the encoder's.
<!-- /worked -->

![The 23 rows of the image block of the toy page, in order: the first mosaic row reads tiles 0 and 1 side by side (P9, P10, P13, P14) and ends in a newline, the second mosaic row (P11, P12, P15, P16) likewise, then the global view row by row with a newline after each row, then the separator. Tile 0 is blue, tile 1 green, the global view yellow, newlines grey and the separator red.](img/image-block-order.svg)

The newline rows break the contiguity of the encoder's output: the toy has 7 runs and a global view of 16 by 16 has 16. The crate therefore builds the block in one pass, a gather from a table of rows (`assemble_block` in `crates/deepseek2ocr/src/layout.rs`), and hands the decoder one run of positions. In the real geometry the rows of the block depend on the shape of the mosaic and not only on how many tiles it has.

<!-- worked: ocr.block_rows_table -->
Rows of the image block of the first generation (global view 16 x 16 tokens, tile 10 x 10) for several tile grids, written columns x rows. The rows are `16 x 17 + 1 = 273` for the global view plus `rows x (10 x columns + 1) x 10` for the mosaic: a mosaic with more tile rows has more newlines. The second column is the number of tiles and the third the projector rows the encoder produces.

| tile grid | tiles | projector rows | rows of the block |
|---|---:|---:|---:|
| none (global view only) | 0 | 256 | 273 |
| 1 x 2 | 2 | 456 | 493 |
| 2 x 1 | 2 | 456 | 483 |
| 2 x 2 | 4 | 656 | 693 |
| 2 x 3 | 6 | 856 | 903 |
| 3 x 2 | 6 | 856 | 893 |
| 3 x 3 | 9 | 1,156 | 1,203 |
| 9 x 1 | 9 | 1,156 | 1,183 |
| 1 x 9 | 9 | 1,156 | 1,263 |
<!-- /worked -->

A mosaic of one column and nine rows costs 1,263 rows against the 1,183 of nine columns and one row, because a mosaic row ends in a newline and a tall mosaic has more of them. The tile tokens are the same 900.

### The order that was assumed

When the first generation's row layout was written in August 2026, its header said what it did not know. The newline-and-separator layout was taken from the reference's graph, and the order of the two blocks, tiles before the global view or after, was an assumption: the preprocessor that orders them was outside the file the author had read, and a swap changes no row count, so "only a real forward can catch it". The layout put the global view first. The second generation's layout had said the same thing in September, calling the multi-tile order a design assumption (commit `3160f2922`). On 26 October 2026 commit `24d2e1f44` read the reference model's own code, which concatenates its embeddings as `[local tiles, global view, view separator]`, found the first generation's assumption backwards and reversed it, in the same change that made the tiled reading work at all; the second generation's layout had already put the tiles first, and its tiled reading was built on that. The header of `crates/deepseek2ocr/src/rows.rs` now carries the reference's line and says that the graph builder did not settle the question and the reference's own code did.

The commit message gives the word error rates above for the whole change, tiles, views and order together, and cannot say how much of the gain is the order. What the history supports is the method: an order that no shape can check is a fact to be read from the reference, never inferred, and the layout was written so that, in the words of the original header, reversing the two blocks is a local change.

## Cutting a page into tiles

The first generation reads a page in one of three ways. A page whose sides are both at most 640 pixels is one aspect-fitted view of 640 by 640, through the encoder at a grid of 40 by 40, which gives 10 by 10 tokens and a block of 111 rows. A page the caller asks to read without tiles is the one global view of 1,024, 273 rows. Any other page is cut into a **mosaic**: the page is resized (stretched, not aspect-fitted) to $640\,T_{\mathrm{w}}$ by $640\,T_{\mathrm{h}}$ and cut into tiles of 640, each read at a grid of 40 by 40, and the page is read once more as an aspect-fitted global view in the 1,024 square, padded with the grey 127. The tile tower is a second encoder over the same weights: SAM's absolute position table is resampled to the tile's grid when the model is loaded, and the relative-position tables are resampled inside the graph, so no second checkpoint exists (`crates/deepseek2ocr/src/serve.rs`, `PageEncoder`, built on `GridSource` of `crates/sam1`).

The grid $(T_{\mathrm{w}}, T_{\mathrm{h}})$ is chosen by a policy, written in `crates/imaging/src/tiling.rs` as the reference model defines it: among the grids of at least 2 and at most 9 tiles, take the one whose ratio of columns to rows is closest to the page's ratio of width to height.

```text
function tile_grid(w: int, h: int, tile: int, max_tiles: int) -> optional (int, int):
    if w <= tile and h <= tile:
        return none
    aspect = w / h
    best = (1, 1)
    best_diff = infinity
    for n in 2 ..= max(max_tiles, 2):
        for i in 1 ..= n:
            if n mod i == 0:
                j = n // i
                diff = abs(aspect - i / j)
                if diff < best_diff:
                    best = (i, j)
                    best_diff = diff
                else if diff == best_diff and w * h > 0.5 * tile ^ 2 * n:
                    best = (i, j)
    return best
```

The loop visits the candidates in order of tile count, and within a count by columns. On an exact tie a later candidate, with more tiles, replaces an earlier one only if the page has more than half the pixels of the tiles it would be cut into: more tiles are spent only when the page has pixels to fill them. The reference breaks ties through the order of a Python set, which is not a specification, and the crate pins the order down (the header of the grid function in that file says which rule is the reference's and which only fixes the order). The page of the opening picks its grid like this.

<!-- worked: ocr.candidates_table -->
The page of 1654 x 2339 pixels has aspect ratio 1654/2339 = 0.7071. The five grids of 2 to 9 tiles whose columns / rows is closest to it:

| grid (columns x rows) | tiles | columns / rows | difference |  |
|---|---:|---:|---:|---:|
| 2 x 3 | 6 | 0.6667 | 0.0405 | chosen |
| 1 x 2 | 2 | 0.5000 | 0.2071 |  |
| 2 x 4 | 8 | 0.5000 | 0.2071 |  |
| 2 x 2 | 4 | 1.0000 | 0.2929 |  |
| 3 x 3 | 9 | 1.0000 | 0.2929 |  |
<!-- /worked -->

A 16:9 screenshot shows the tie rule at work.

<!-- worked: ocr.tie_example -->
A 1920 x 1080 page (aspect ratio 1.7778), tiles of 640 pixels, at most 9. The grids 2 x 1 and 4 x 2 are both 0.2222 away from it (columns / rows = 2). A later candidate of the same error replaces an earlier one only if the page has more than half of `640^2 x tiles` pixels; the page has 2,073,600.

| grid | tiles | half the pixels of the tiles | page has more |
|---|---:|---:|---:|
| 2 x 1 | 2 | 409,600 | yes |
| 4 x 2 | 8 | 1,638,400 | yes |

The grid chosen is 4 x 2.
<!-- /worked -->

The second generation uses the same policy with tiles of 768 pixels and at most 6 of them, and a block that is simpler to count: 144 rows for each tile, 256 for the global view and one separator, because it has no newline rows. Both generations read a page like this.

<!-- worked: ocr.readings -->
The reading each generation gives a page of the given width x height, and the rows of the image block it puts in the sequence. First generation: tiles of 640 pixels, at most 9, 273 rows for the global view, 111 for one small view. Second generation: tiles of 768 pixels, at most 6, 257 rows for the global view, 145 for one small view, 144 rows per tile.

| page (w x h) | v1 reading | v1 rows | v2 reading | v2 rows |
|---|---|---:|---|---:|
| 300 x 40 | one small view | 111 | one small view | 145 |
| 640 x 640 | one small view | 111 | one small view | 145 |
| 1024 x 1024 | 2 x 2 tiles and global | 693 | 2 x 2 tiles and global | 833 |
| 1240 x 1754 | 2 x 3 tiles and global | 903 | 2 x 3 tiles and global | 1,121 |
| 1654 x 2339 | 2 x 3 tiles and global | 903 | 2 x 3 tiles and global | 1,121 |
| 2480 x 3508 | 2 x 3 tiles and global | 903 | 2 x 3 tiles and global | 1,121 |
| 1920 x 1080 | 4 x 2 tiles and global | 1,093 | 2 x 1 tiles and global | 545 |
<!-- /worked -->

The scales at which the page reaches the encoders are the numbers that decide legibility, and they are the reason the policy is built on a mosaic of the page instead of a mosaic of crops.

<!-- worked: ocr.reading_scales -->
Scale of the page in the pixels the encoder sees. 'Global' is the aspect-fitted view in a 1024 x 1024 square, 1024 / max(width, height); 'tiles' is the mosaic of the chosen grid, tile x columns / width across and tile x rows / height down.

| page | v1 global | v1 tiles (across / down) | v2 global | v2 tiles (across / down) |
|---|---:|---:|---:|---:|
| 1654 x 2339 | 0.4378 | 0.7739 / 0.8209 | 0.4378 | 0.9287 / 0.9850 |
| 2480 x 3508 | 0.2919 | 0.5161 / 0.5473 | 0.2919 | 0.6194 / 0.6568 |
| 1920 x 1080 | 0.5333 | 1.3333 / 1.1852 | 0.5333 | 0.8000 / 0.7111 |
<!-- /worked -->

A scan at 300 pixels to the inch gets the same 903 and 1,121 rows as one at 200, and shows the page at a smaller scale: 15-pixel text at 200 pixels to the inch is 22.5-pixel text at 300, and in the first generation's mosaic both arrive at 11.6 pixels. Pixels above what a mosaic of at most 9 tiles can show are thrown away by the resize, which is why the number of rows saturates.

![Rows of the image block against the height of a portrait page. The first generation is flat at 111 rows until the page is taller than 640 pixels, jumps to 903 and stays; the second is flat at 145 until 768 pixels, jumps to 1,121 and stays; the dynamic resolution of the previous chapter, at its served limit of one million pixels, rises smoothly to 988 and stays.](img/image-rows-page-size.svg)

The figure shows what a policy of tiles does to the cost of a page that a policy of dynamic resolution (Chapter 9) does not. Both end on a plateau, because each has a ceiling of pixels or tiles, but the tiles have steps, because a page that has just crossed 640 pixels gets a mosaic and a global view at once, and a small crop is about an eighth of the cost of a page: 111 rows against 903.

The path from pixels to the tower was not always this. The crate's first image path resized the page on the device with a cubic filter, clamped to the unit interval because a cubic filter rings past the range at the hard edges of text, and put the result in the one global view. The filter is not antialiased when it shrinks a page, as `crates/deepseek2ocr/src/preprocess.rs` says in the comment on its constant. Commit `11d3cdf7a` added views prepared as the reference prepares them (`crates/imaging/src/views.rs`): an antialiased bicubic resize on eight-bit pixels, an aspect-preserving pad, a mosaic cut row-major, and the half-range normalization, with the pad and the mosaic pinned to the reference byte for byte; the tiled reading of commit `24d2e1f44` is built on them. Chapter 1's stripes explain why the change mattered for text: a filter that reads two neighbours of an output pixel aliases a page that is shrunk by more than 2.

## What a page costs, and what squeezing it means

The number of rows of a page is fixed by its geometry. What a row is worth is not. The question of this chapter, how few tokens a page can be squeezed into, depends on how many tokens the same page would have been as text, and the answer is a ratio, written $\zeta$ here: the number of text tokens of the page divided by the rows of its image block.

<!-- worked: ocr.squeeze -->
A page of 930 words (the dense synthetic page of the commit cited below), at 6 characters a word including the space and 4 characters a text token, is 1,395 text tokens. Divided by the rows of the image block:

| reading of the 1654 x 2339 page | rows | text tokens / rows |
|---|---:|---:|
| global view | 273 | 5.11 |
| v1, 2 x 3 tiles and global | 903 | 1.54 |
| v2, 2 x 3 tiles and global | 1,121 | 1.24 |

What one projector token can carry. A token stands for a block of 64 x 64 pixels of the view (a 16-pixel patch times the 4 of the compressor), which is `64 / s` by `64 / s'` pixels of the page when the view shows the page at scales s across and s' down (those of the previous tables). Text of font size f is taken to need `0.5 f` of width per character and `1.2 f` per line, so `0.6 f^2` pixels of page, and the table assumes the block is full of it.

| font size (px) | seen as | page pixels per token | characters | text tokens |
|---|---:|---:|---:|---:|
| 15 | global view | 21,371 | 158.3 | 39.6 |
| 15 | v1 tiles | 6,448 | 47.8 | 11.9 |
| 15 | v2 tiles | 4,478 | 33.2 | 8.3 |
| 30 | global view | 21,371 | 39.6 | 9.9 |
| 30 | v1 tiles | 6,448 | 11.9 | 3.0 |
| 30 | v2 tiles | 4,478 | 8.3 | 2.1 |

A row of the block is 1,280 numbers of 16 bits, 20,480 bits. The 158.3 characters of the first row, at log2(70) = 6.13 bits for a character drawn from 70 equally likely symbols (an upper bound), are 970 bits.
<!-- /worked -->

The dense page of the commit, 930 words, is about 1,395 text tokens under the stated assumptions, and read as tiles it costs 903 rows in the first generation: a ratio of 1.54, and 1.24 in the second. Read as the one global view the ratio is 5.11, and the page is not read, because it is the same 273 rows asked to carry 1,395 tokens' worth of letters 6.6 pixels tall. The lower table says why. A projector token stands for a block of $64 \times 64$ view pixels, and what that block carries depends on the scale at which the page was shown: 47.8 characters for text of 15 pixels read as the first generation's tiles, 158.3 in the global view. A row of the decoder's stream is 1,280 numbers, and what limits the global view is not the information a row can carry, which at 16 bits a number is 20,480 bits against the 970 that those 158 characters could need at most, but how much of the page's detail is still in the pixels after the shrink, and whether the trained encoder can read it.

A ratio above 1 is what the word squeezing means: the page as pixels takes fewer rows than the page as text would. A page of 150 words is 225 text tokens against the same 903 rows, a ratio of 0.25: the pixels cost four times what the text would. That is no argument against reading them, since the pixels are what there is when the text is not, and the ratio is for comparing pages and readings with each other and for asking how much of the decoder's window a page of a given density takes.

### The word error rate

The commit's evidence is a word error rate, and it needs a definition that a reader can compute. The **edit distance** between a reference and a hypothesis, each a list of words, is the fewest insertions, deletions and substitutions that turn one into the other, found by filling a table of prefixes; the word error rate is that distance divided by the number of words of the reference.

```text
function edit_distance(ref: list<text>, hyp: list<text>) -> int:
    d = zeros(int, [len(ref) + 1, len(hyp) + 1])
    for i in 0 ..= len(ref):
        d[i, 0] = i
    for j in 0 ..= len(hyp):
        d[0, j] = j
    for i in 1 ..= len(ref):
        for j in 1 ..= len(hyp):
            substitute = d[i - 1, j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
            d[i, j] = min(substitute, d[i - 1, j] + 1, d[i, j - 1] + 1)
    return d[len(ref), len(hyp)]
```

<!-- worked: ocr.wer -->
Reference: the page reads left to right (6 words). Hypothesis: the page read left right now. Cell (i, j) is the fewest insertions, deletions and substitutions turning the first i reference words into the first j hypothesis words; the columns are the empty hypothesis and then the words of the hypothesis in order.

| reference word | - | the | page | read | left | right | now |
|---|---:|---:|---:|---:|---:|---:|---:|
| start | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| the | 1 | 0 | 1 | 2 | 3 | 4 | 5 |
| page | 2 | 1 | 0 | 1 | 2 | 3 | 4 |
| reads | 3 | 2 | 1 | 1 | 2 | 3 | 4 |
| left | 4 | 3 | 2 | 2 | 1 | 2 | 3 |
| to | 5 | 4 | 3 | 3 | 2 | 2 | 3 |
| right | 6 | 5 | 4 | 4 | 3 | 2 | 3 |

The edit distance is 3, and the word error rate is 3 / 6 = 0.500. A hypothesis of 12 unrelated words against the same reference has distance 12 and a rate of 2.000, above 1.
<!-- /worked -->

A rate can exceed 1, because insertions count, and a hypothesis that invents more words than the page holds is worse than one that writes nothing. The ledger of the second generation describes its output on the dense page, read as the global view alone, as 427 hallucinated words, in the sense of the opening: text that is fluent and is not on the page. The numbers of the commit are one synthetic page, one decode, greedy, with a budget of 1,500 tokens; they are a dated observation of a stated setup, the kind the book's rules call history, and not a score that another page should be expected to reach.

## Letting queries order the tokens

The second generation, `crates/deepseekocr2`, keeps SAM and the compressor and the decoder, and replaces CLIP and the concatenation. After the compressor the view is a grid of tokens, 256 for the 1,024-pixel global view and 144 for a 768-pixel tile (the grid of 48 by 48 patches becomes 12 by 12), with a width of 896 instead of 1,024. A bank of learned vectors, **queries**, as many as there are tokens, is appended to them, and a tower of 24 blocks of a decoder-style stack runs over the $2n$ rows: width 896, 14 query heads and 2 key-value heads of width 64, a gated feed-forward of 4,864, rotary positions and no learned table. Only the query half of its output is kept and passed through one linear map to the decoder's width, 1,280.

What makes it a different design is the mask. The first $n$ rows are the image tokens, and they attend to each other in both directions, as the tokens of a vision tower do. The last $n$ rows are the queries, and each attends to every image row and to the queries up to and including itself, as a decoder's token does. With $L_{\mathrm{p}}$ leading rows that see each other, the rule is

$$
\mathrm{allow}(i, j) = (i < L_{\mathrm{p}} \text{ and } j < L_{\mathrm{p}}) \text{ or } j \le i,
\tag{1}
$$

which is the prefix mask that Chapter 9's Moondream 3 decoder uses for its image tokens, and the crate runs it with the kernel that decoder uses, `crates/kernels/wgsl/attn_prefix_mask.wgsl`, with $L_{\mathrm{p}} = n$ (`crates/deepseekocr2/src/encoder.rs`). The two heads' worth of keys and values are widened to 14 before the scores, so each key-value head serves 7 query heads.

<!-- worked: ocr.mask_toy -->
The mask of the resampler for 3 image rows and 3 queries (6 rows in all), `allow(i, j) = (i < 3 and j < 3) or j <= i`: row i may read column j when the entry is 1. The last column counts the keys of the row.

| row | 0 | 1 | 2 | 3 | 4 | 5 | keys |
|---|---:|---:|---:|---:|---:|---:|---:|
| image 0 | 1 | 1 | 1 | 0 | 0 | 0 | 3 |
| image 1 | 1 | 1 | 1 | 0 | 0 | 0 | 3 |
| image 2 | 1 | 1 | 1 | 0 | 0 | 0 | 3 |
| query 0 | 1 | 1 | 1 | 1 | 0 | 0 | 4 |
| query 1 | 1 | 1 | 1 | 1 | 1 | 0 | 5 |
| query 2 | 1 | 1 | 1 | 1 | 1 | 1 | 6 |

The mask has 24 entries equal to 1 of 36, which is 3^2 + (4 + 5 + 6) = 24.
<!-- /worked -->

A toy of 3 image rows and 3 queries shows the shape: the image rows read 3 keys each, the queries 4, 5 and 6. Every query can see the whole page and the queries before it, so the $i$-th query's output is a function of the whole image and of the queries before it. That is a causal structure laid over a picture, and it is an opening for the model to put the tokens in an order that suits a decoder that reads left to right. What the training makes of that opening is a claim about a model's weights, and the crate, whose tests check the layout, the mask and the gradient of the encoder, cannot show it; what the code verifies is the structure.

What it costs follows from the shapes.

<!-- worked: ocr.v2_tower -->
Parameters of the second tower of each generation, counted from the shapes: v1's CLIP block is `4 d^2 + 4 d` (attention with biases) `+ 2 d f + f + d` (feed-forward with biases) `+ 4 d` (two layer norms with gain and offset); v2's block has a query matrix `d^2 + d`, key and value matrices `128 d + 128` each (2 key-value heads of 64), an output matrix `d^2`, three feed-forward matrices `3 d f` of f = 4864, and two gains `2 d`; v2's tower also holds the final gain, and the two query banks (144 and 256 rows of 896). The patch embedding and class row of CLIP, which v1 bypasses, are not counted.

| part | per block | parameters |
|---|---:|---:|
| CLIP-L tower of v1 (24 blocks, width 1024) | 12,596,224 | 302,309,376 |
| learned-query tower of v2 (24 blocks, width 896) | 14,912,384 | 358,256,512 |
| projector of v1: Linear(2048, 1280) | - | 2,622,720 |
| projector and separator of v2: Linear(896, 1280), 1280 numbers | - | 1,149,440 |
<!-- /worked -->

Its 358 million parameters are 19 percent more than the 302 million of the CLIP tower it replaces. In FLOPs the resampler is the largest stage of the second generation's global view after the products of SAM's blocks, 380.5 billion against CLIP's 161.7, because it runs over twice as many rows with a mask that lets the queries read every key, and a whole global view is 1,352.8 against 1,135.1 (the table of the encoder above). The rows the decoder sees are fewer for a global view alone, 257 against 273, because there are no newlines. For a page with tiles they are more, 1,121 against 903, because the tiles are 768 pixels and have 144 tokens each, and the page is read at a scale much nearer to its own pixels.

The layout has changed with the encoder. The checkpoint of the second generation carries one separator and no newline, so every view's rows are one uninterrupted run, and the row plan of `crates/deepseekocr2/src/rows.rs` is a concatenation: the rows of every tile, row-major, the global view's, one separator. The order of the tiles is the same one the first generation's history gave: tiles, global, separator. Nothing in the second generation tells the decoder the width of a row; the queries are a sequence from the start, and whatever the tower learned about layout has been put into their order.

## What the decoder reads, and what it writes

Both generations feed the same decoder, `crates/deepseek2`: twelve blocks of width 1,280 that Volume VI's chapter on attention variants priced for its cache. It is a mixture of experts with ordinary multi-head attention, 10 heads and 10 key-value heads of width 128, block 0 dense and blocks 1 to 11 holding 64 routed experts of which 6 are chosen for each token, plus 2 shared ones, a vocabulary of 129,280 and an output head that is not tied to the embedding table.

<!-- worked: ocr.decoder_table -->
The decoder of both generations: 12 blocks of width 1280, 10 heads of 128 with 10 key-value heads (multi-head attention), block 0 dense with feed-forward width 6848, blocks 1 to 11 mixing 64 routed experts of width 896 (6 chosen per token) and 2 shared experts, vocabulary 129,280, output head not tied to the embedding table. Counts are products of the shapes in the second column.

| part | shape | parameters |
|---|---:|---:|
| attention, per block (q, k, v, o) | 4 x 1280^2 | 6,553,600 |
| dense feed-forward, block 0 | 3 x 1280 x 6848 | 26,296,320 |
| 64 routed experts, per MoE block | 64 x 3 x 1280 x 896 | 220,200,960 |
| 2 shared experts, per MoE block | 3 x 1280 x 2 x 896 | 6,881,280 |
| router, per MoE block | 64 x 1280 | 81,920 |
| embedding table and output head (untied), each | 129,280 x 1280 | 165,478,400 |
| all parameters (and 25 gains of width 1280) |  | 2,934,734,080 |
| applied to every prompt token (6 of 64 experts) |  | 408,616,960 |
| output head, applied once per scored row |  | 165,478,400 |

Cache per token: 2 x 12 x 10 x 128 x 2 bytes = 61,440 bytes in bf16. A latent cache of the kind of Volume VI, Chapter 3 (a latent of 512 and a rotary key of 64 per layer) would store 12 x 576 x 2 = 13,824 bytes per token, 4.44 times fewer; this decoder does not have that layer.
<!-- /worked -->

The decoder has 2.93 billion parameters and applies 408.6 million of them, plus the output head, to each token: the mixture is what lets a decoder of nearly 3 billion parameters cost, per token, about what one of 0.57 billion would. It is not a latent-attention decoder, in spite of the family name, as Chapter 3 of Volume VI showed from the checkpoint's tensors; what that chapter's counting does give here is a price for the cache of a page. Under the multi-head layer a token costs 61,440 bytes in bf16, and the layer of Chapter 3 would have cost 13,824, 4.44 times fewer. The decoder as shipped keeps the larger cache, and every row of the image block is paid for in it.

<!-- worked: ocr.page_costs -->
What one page costs the first generation before the first word is written: the vision half (the encoder over every view, at the FLOPs of the table above), the decoder's prefill of the prompt (BOS, the image block and the 9 tokens of the default instruction) at 2 FLOPs per active parameter per token, `4 L H d_h` per attention pair and the output head once, and the cache the prompt occupies at 61,440 bytes per token.

| reading | rows | prompt tokens | vision TFLOPs | decoder prefill TFLOPs | cache (MiB) |
|---|---:|---:|---:|---:|---:|
| one small view | 111 | 121 | 0.39 | 0.10 | 7.1 |
| global view | 273 | 283 | 1.14 | 0.23 | 16.6 |
| 2 x 3 tiles and global | 903 | 913 | 3.46 | 0.77 | 53.5 |
| 3 x 3 tiles and global | 1,203 | 1,213 | 4.62 | 1.04 | 71.1 |
<!-- /worked -->

A page of 903 rows costs 3.46 trillion FLOPs in the vision towers, 0.77 trillion in the decoder's prefill, and 53.5 MiB of cache that stays for as long as the page is being written out. The prompt that carries it is BOS, the block, and an instruction of 9 tokens: by default `<|grounding|>Convert the document to markdown.` (`crates/ocrserve/src/lib.rs`), preceded by a newline (`crates/deepseek2ocr/src/serve.rs`). The decoder does not mind that 903 of its positions were put there by an encoder. It reads them through one-dimensional rotary positions like any others.

### Writing the page out

The answer is text, and the layout survives in it as markup. The vocabulary of the checkpoint holds reserved tokens that ordinary text cannot spell: the grounding marker, a reference marker pair `<|ref|>` and `<|/ref|>`, a detection marker pair `<|det|>` and `<|/det|>`, and the table cell tags `<td>` and `</td>`, each a single id (`crates/deepseek2ocr/src/prompt.rs`). The instruction's leading `<|grounding|>` is a switch, not decoration: with it the decoded text pairs each piece of recognised text with its bounding box on the page, and the documentation of the crate shows a real run on a scanned page that opens `<|ref|>image<|/ref|><|det|>[[48, 0,` before its token budget of 10 cuts it off mid-box. A request that wants plain markdown deletes the marker. Brain streams this text and does not interpret the boxes; reading them back, as Chapter 9 did for the location tokens of a different model, is the caller's job.

A document is a long output, and a long output of a greedy decoder has a failure that a short caption does not. A table whose rows look alike, or a repeated header, can lead the decoder back into a span it has just written, and nothing in greedy decoding breaks the loop. The recipe that goes with the model is a filter on the logits: no run of $n$ tokens may repeat inside a window. At each step the decoder takes the last $n - 1$ generated tokens, looks through the last `window` generated tokens for earlier occurrences of the same $n - 1$ tokens, and bans the token that followed each one, except tokens on an allow list.

```text
function ngram_banned(gen: list<int>, n: int, window: int,
                      allow: list<int>) -> list<int>:
    out = empty list
    if len(gen) < n:
        return out
    prefix = gen[len(gen) - (n - 1) : len(gen)]
    for i in max(len(gen) - window, 0) .. len(gen) - n + 1:
        nxt = gen[i + n - 1]
        if gen[i : i + n - 1] == prefix and not (nxt in out) and not (nxt in allow):
            append nxt to out
    return out
```

<!-- worked: ocr.ngram_filter -->
Generated so far: [7, 8, 9, 3, 5, 7, 8, 9, 11, 2, 7, 8, 9] (13 tokens). n-gram size n = 4, window 13, allowed token [11]. The trailing 3 tokens are [7, 8, 9]. Start of the scan = max(13 - 13, 0) = 0; end (exclusive) = 13 - 4 + 1 = 10.

| start i | tokens i .. i + 2 | token i + 3 | equals the trailing three |
|---|---:|---:|---:|
| 0 | [7, 8, 9] | 3 | yes |
| 1 | [8, 9, 3] | 5 | no |
| 2 | [9, 3, 5] | 7 | no |
| 3 | [3, 5, 7] | 8 | no |
| 4 | [5, 7, 8] | 9 | no |
| 5 | [7, 8, 9] | 11 | yes |
| 6 | [8, 9, 11] | 2 | no |
| 7 | [9, 11, 2] | 7 | no |
| 8 | [11, 2, 7] | 8 | no |
| 9 | [2, 7, 8] | 9 | no |

Tokens that would complete a repeated 4-gram: [3, 11]. After removing the allowed token [11]: [3]. The same history under three windows:

| window | scan starts at | followers | banned |
|---|---:|---|---|
| 13 | 0 | [3, 11] | [3] |
| 12 | 1 | [11] | [] |
| 7 | 6 | [] | [] |
<!-- /worked -->

The values for these models are 30 and 90 (`NGRAM_SIZE` and `NGRAM_WINDOW` in `crates/ocrserve/src/frontend.rs`), and the allowed tokens are the two table cell tags, because a real table repeats them far more often than any limit on repeated spans would tolerate. Commit `ae78b39a3` (September 2026) added it, and says in its message that without it a long document's greedy decode can lock onto a repeated span such as a table row; the code is the filter above, and the scheduler's form of it asks the decoder for one candidate more than the number banned and takes the best that is not banned, so a step with nothing banned is a plain greedy step. The filter only looks at generated tokens, never at the prompt.

## Serving pages, not images

Two models with different encoders, different tile sizes and different checkpoints share everything after the encoder, and `crates/ocrserve` is where the sharing is written. A model implements one trait, `OcrModel`, which names it, describes it, says how many bytes it holds while hot and how to find and load a checkpoint; the crate supplies the one `generate` action, the scheduler front end, the placement of the two halves on the offered cards and the adapter that the residency scheduler holds. Commit `04c85a38f` (October 2026) made the split: the first generation's resident and caps shrank to an encoder and a loader, and the second generation, whose serving had been a CPU session that recomputed the whole prefix at every step, was served through the same batched decoder engine.

A request is a page and an instruction. The page's pixels arrive as a blob of floating-point numbers, tens of megabytes for a page, and unpacking them is the encoder thread's job and not the scheduler's. The encoder stage is a pair of threads (`ENCODE_WORKERS`), each owning a complete set of towers, because the scratch of a tower is not shared between pages, and the scheduler decodes other pages while a page encodes. The queue that feeds them does not take pages in arrival order. Commit `e01cf6972` (October 2026) made it take the cheapest page, discounted by how long it has waited, so that a small crop is not held up behind a full page's encode and a full page cannot be passed over for ever.

<!-- worked: ocr.encode_order -->
The encoder takes the waiting page of the smallest `pixels / (1 + seconds waited)`. Three pages that arrived together (wait 0) are taken in this order:

| taken | page | cost when taken |
|---|---:|---:|
| 1 | 600 x 400 | 240,000 |
| 2 | 1200 x 1700 | 2,040,000 |
| 3 | 2000 x 3000 | 6,000,000 |

A 2000 x 3000 page that has waited 120 seconds costs 49,587 against the 240,000 of a fresh 600 x 400 crop, and goes first. The wait at which it passes a fresh crop is `6,000,000 / 240,000 - 1` = 24 seconds.
<!-- /worked -->

The cost is read from the blob's metadata, the pixels `w x h`, without unpacking the page; a page whose metadata is missing costs 0. A page of $P$ pixels that has waited $t$ seconds is taken before a fresh crop of $C$ pixels when $P/(1 + t) < C$, so the bound on how long a page can be passed over by crops is $P/C - 1$ seconds, whatever the stream of crops is.

The decoder is the batched engine of `crates/deepseek2`, with a paged cache pool. A page occupies blocks of 64 tokens, the pool holds 131,072 tokens in bf16 or half as many in f32 (the pool is sized by bytes, so a wider element means fewer tokens), by default 64 pages decode together (a flag raises it to 128), the prompt is prefilled in chunks of 512 rows and one chunk is prefilled between two decode steps. A request with no budget gets 2,048 new tokens, and no sequence may pass 8,192. The pool caps how many full-budget pages can be in flight.

<!-- worked: ocr.pool_capacity -->
The cache pool holds 131,072 tokens in blocks of 64 (2,048 blocks, 7.50 GiB at 61,440 bytes a token), and at most 64 pages are decoded together. A page that generates its whole budget of new tokens holds `ceil((prompt + new) / 64)` blocks; the table gives that and the number of such pages the pool holds (capped at 64).

| reading | prompt tokens | blocks, 512 new | pages | blocks, 2048 new | pages |
|---|---:|---:|---:|---:|---:|
| one small view | 121 | 10 | 64 | 34 | 60 |
| global view | 283 | 13 | 64 | 37 | 55 |
| 2 x 3 tiles and global | 913 | 23 | 64 | 47 | 43 |
| 3 x 3 tiles and global | 1,213 | 27 | 64 | 51 | 40 |
<!-- /worked -->

When a page ends early, at its end-of-sequence token, its blocks are freed that step, so these are the worst cases. A page that needs more blocks than the pool has free is preempted and its work recomputed, as the documentation of the serving policy says; with more pages in the batch, each gets fewer tokens before that happens. The same crate decides which flags the models honour (`--max-batch`, `--ctx`, `--kv-cache`, `--gemm`), and a flag the model cannot honour stops it from being served at all, rather than served differently from what was asked (commit `e01cf6972`).

The last piece is placement. The vision towers of a replica need 15 GiB, and the decoder 18, as budgeted by the constants of the two model crates, and the question of where they go depends on the cards.

<!-- worked: ocr.placement -->
The encoders of one replica need 15 GiB and the decoder 18 GiB (the figures the crates budget). Cards are written (usable GiB). The encoder goes on the card with the most room that holds it (the last of equals); the decoder on another card that holds it, else on the encoder's card if it holds both, else on the host. That is the placement of one replica, and it reads the free memory of the cards; it alone decides the five rows that do not ask for several replicas. The other three ask for as many replicas as fit. A card qualifies for a whole replica when max(total capacity, free memory) is at least 33 GiB, the encoders and the decoder together; the total capacity is the card's free memory unless the row gives it ("96 and 96 in all"). If two or more cards qualify each runs one replica with both parts on it, otherwise the single placement above is used. So the count of replicas follows the total capacity and the placement of a lone replica follows the free memory.

| offered | replicas | placement |
|---|---:|---|
| one card of 24 | 1 | encoder on card 0, decoder on host |
| cards of 24 and 30 | 1 | encoder on card 1, decoder on card 0 |
| two cards of 90, a single replica asked for | 1 | encoder on card 1, decoder on card 0 |
| one card of 90 | 1 | encoder on card 0, decoder on same card |
| two cards of 90 | 2 | encoder on card 0, decoder on same card; encoder on card 1, decoder on same card |
| cards of 90 and 12 | 1 | encoder on card 0, decoder on same card |
| one card of 10 | 1 | no card, decoder on host |
| cards 90 and 12 free, 96 and 96 in all | 2 | encoder on card 0, decoder on same card; encoder on card 1, decoder on same card |
<!-- /worked -->

A card that holds both gets the whole model; two such cards each get a replica, judged by total capacity and not by what is free at the moment, so that a card that a neighbouring process is holding at start gets its replica as soon as it is freed (commit `20ec3b5d1`, October 2026). A decoder that fits no card is put on the host, as it is the smaller loss to be slow than to fail to allocate. The encoder is placed first, on the card with the most room that holds it, and the last of equals wins a tie, as the iterator of the implementation does.

## Check your understanding

1. A colleague says that because the compressor cuts 4,096 tokens to 256, the first generation reads a 1,024-pixel view at one sixteenth of the cost of a model with no compressor. Which parts of the table of FLOPs does the claim fit, and which does it not?
2. A page of 1,500 by 600 pixels is read as a 3 by 1 mosaic. Why does the grid 4 by 2, which is as close to the page's aspect ratio, lose, and why does 2 by 1 lose too?
3. A first version of the row layout put the global view before the tiles. Why could no check of the shapes or of the number of rows find the mistake, and what kind of check could?
4. In the toy mask of the chapter, which rows' outputs change when the input vector of query 0 (row 3) changes, after one block, and after any number of blocks?
5. A user sets `max_tiles` to 1 for a page of 1,654 by 2,339. What does the page get, and what does the question tell about the meaning of the parameter?
6. A scan arrives at 600 pixels to the inch. Argue that it costs the same rows as a scan at 200 pixels to the inch, and say what it costs that the 200 does not.

## Summary

A model reads a page by turning it into rows, 16 positions to one after a convolutional compressor, and handing the rows to a decoder that writes the text. What survives in those rows is decided before the model sees a pixel, by how the page is shrunk: below two output pixels to the period of a stroke the stroke aliases, and the tiled reading exists to keep the page at 0.77 to 0.99 of its own scale. The compressor, a fixed convolution, is what makes this affordable, a factor of about 26 in the global tower, 17 in the prefill and 14.5 in the cache; the layout of the page, a 2-dimensional fact, is carried into the 1-dimensional decoder by newline and separator rows in the first generation and by the order of learned queries in the second, and out of it by reserved tokens of the transcript. The rows of a page are set by its geometry, 111 for a crop, 273 for the global view alone and 903 for a page of ordinary shape in the first generation, 145, 257 and 1,121 in the second; what a row is worth depends on the size and density of the text. The serving crate shares the rest: a queue that encodes cheap pages first, a batched decoder with a paged cache, a placement of two halves on whatever cards there are, and a filter against repetition.

The chapter's compression is a count of tokens and the evidence for its quality is a word error rate on one page; a model that was never trained on a document cannot be given this behaviour by a counting argument. The next chapter turns from a page to a face, where the output is not text but a box and a vector, and the quality of a model is measured by whether the same person lands in the same place.

## References

- Haoran Wei, Yaofeng Sun, Yukun Li, "DeepSeek-OCR: Contexts Optical Compression", 2025.
- Yanghao Li, Hanzi Mao, Ross Girshick, Kaiming He, "Exploring Plain Vision Transformer Backbones for Object Detection", 2022.
- Alexander Kirillov et al., "Segment Anything", 2023.
- DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", 2024.

## Exercises

1. **[Calculate]** A page is 1,000 by 1,500 pixels. Give its reading, the rows of its image block, and its prompt tokens (BOS, the block and the 9 tokens of the default instruction) for both generations.
2. **[Calculate]** For the first generation (global view 16 by 16 tokens, tiles 10 by 10), give the rows of the image blocks of a 3 by 1 mosaic and of a 1 by 3 mosaic, and say which tokens differ.
3. **[Calculate]** A page 2,000 by 2,800 pixels carries stems with a period of 5 pixels. Give the scale of the global view, the period of the stems in it, and whether they can be told apart.
4. **[Calculate]** Find by hand the grid that the tile policy picks for a page of 1,500 by 600 pixels with tiles of 640 and at most 9 tiles, including the tie, and the rows of the block it gives.
5. **[Calculate]** For a page read as a 3 by 3 mosaic with 2,048 new tokens, give the blocks it holds in the pool, and the number of such pages that the pool holds if its elements are stored as f32 (half the tokens).
6. **[Calculate]** One block of the resampler has 2 key-value heads of width 64 at width 896. Give the parameters of its key and value projections, with biases, and what they would be if every query head had its own key-value head.
7. **[Implement]** Reproduce `ocr.legibility` from the description of the area average: the cumulative ink of a square wave, the average over $1/s$ input pixels, and the swing of 200 outputs.
8. **[Implement]** Reproduce `ocr.toy_layout`, `ocr.toy_runs_text` and `ocr.block_rows_table` from the `row_plan` routine, and `ocr.candidates_table`, `ocr.tie_example` and `ocr.readings` from `tile_grid`.
9. **[Implement]** Reproduce `ocr.mask_toy`, `ocr.ngram_filter` and `ocr.wer`.
10. **[Investigate]** Which grid of at most 9 tiles gives the most rows in the first generation's block, and what aspect ratio of page picks it? Compare it with the second generation's rows for the same page.
11. **[Investigate]** The encoder queue takes the page of the smallest $P/(1 + t)$. Derive the longest a page of 6,000,000 pixels can wait behind a steady stream of 300 by 200 crops, and say what an operator who wants a bound of 60 seconds would change.

## Key terms

- **optical character recognition**: turning the pixels of a page into the characters that were printed on it; in this chapter a decoder writes the page as text while it attends to rows computed from the picture. <!-- concept: optical-character-recognition -->
- **vision token compression**: reducing the number of tokens that stand for a picture before the stages whose cost grows with that number, the quadratic attention of a global tower and the decoder's prefill and cache; here by a fixed convolution that folds 16 positions into 1, and by the choice of how many views a page gets. <!-- concept: vision-token-compression -->
- **document layout**: the two-dimensional arrangement of a page's content, columns, lines, tables and their order of reading, which a model must carry from a grid of pixels into a sequence of rows and back out as an ordered transcript. <!-- concept: document-layout -->

## Answers

**Recall questions.**

1. $1024 \times 1280 + 1280 = 1{,}312{,}000$ parameters: a weight matrix and a bias.
2. The splice requires one picture row for every placeholder, so the block must have exactly 273 rows. The tower produces 256, so 17 rows stand for something that is not a patch of the page; the chapter says they are the 16 newline rows and the separator.
3. A query reads $w$ keys, itself included, and $L$ layers reach $L(w - 1) + 1$ positions.
4. $2 \times 10 \times 128 = 2{,}560$ numbers with multi-head attention, and $512 + 64 = 576$ with the latent.
5. $(64 + 2 - 3) // 2 + 1 = 32$.
6. With the area average of 4 pixels every output pixel is the mean of one full period, so the stems vanish into the constant 127.5 (half of 255). Shrinking by 2.3 averages 2.3 pixels, which is not a whole period, so some contrast remains but the output period is under 2 pixels and the pattern aliases.

**Check your understanding.**

1. It fits the stages that come after the compressor: CLIP's FLOPs are 4.12 trillion without it against 0.16 with it, the decoder's prefill 3.87 against 0.23 and the cache 240.6 MiB against 16.6. It does not fit SAM, which runs on all 4,096 patches in both cases and is 971.9 of the 1,135.1 billion FLOPs, so the encoder as a whole is nowhere near a sixteenth.
2. The aspect ratio of the page is 2.5, and the three grids 2 by 1, 3 by 1 and 4 by 2 are all 0.5 away. The rule prefers the later candidate only if the page has more than half the pixels of its tiles: 3 tiles need 614,400 and the page has 900,000, so 3 by 1 replaces 2 by 1; 8 tiles need 1,638,400, which the page does not have, so 4 by 2 does not replace it.
3. Both orders contain the same rows and the same number of them, only in a different sequence, so every shape and every count agrees. Only a forward pass compared with the reference's can tell, which is why commit `24d2e1f44` went to the reference's own code.
4. Rows 3, 4 and 5, the three queries, and no others. Each query reads the queries before it, so query 0's vector reaches all of them in the first block; the image rows (0 to 2) never read a query, so they are the same after one block and after any number of blocks.
5. The code raises a limit of 1 to 2, since a grid has at least 2 tiles, and the page of 1,654 by 2,339 gets the closest grid of at most 2 tiles: 1 by 2 (aspect 0.5 against 0.7071) rather than 2 by 1. Its block is $273 + 2 \times (10 \times 1 + 1) \times 10 = 493$ rows. A request for at most one tile therefore reads two, and only 0 turns tiling off.
6. The mosaic of at most 9 tiles has the same size whatever the scan, so the page gets the grid 2 by 3, 903 rows, exactly as at 200 pixels to the inch. What the 600 costs is the resize of a page of 4,961 by 7,016 pixels, 34.8 million of them, and the transfer of a blob of 398 MiB against 44.3 MiB for 1,654 by 2,339 (three floats of 4 bytes a pixel), and what it gives the model is nothing.

**Exercises.** 1. Both pages have aspect 0.667, so the grid is 2 by 3 for each: the first generation has 903 rows and 913 prompt tokens, the second 1,121 rows and 1,131 prompt tokens. 2. A 3 by 1 mosaic has $273 + 1 \times 31 \times 10 = 583$ rows and a 1 by 3 mosaic $273 + 3 \times 11 \times 10 = 603$; the 900 tile tokens are the same and the 20 extra rows of the second are newlines. 3. $1024/2800 = 0.3657$, the period is $5 \times 0.3657 = 1.83$, which is under 2, so they cannot be told apart. 4. The aspect ratio is 2.5. The grids 2 by 1, 3 by 1 and 4 by 2 are 0.5 away; 2 by 1 is first, 3 by 1 replaces it because the page has 900,000 pixels against the 614,400 that 3 tiles need, and 4 by 2 does not replace it, because 8 tiles need 1,638,400. The grid is 3 by 1, with 583 rows. 5. $1 + 1203 + 9 + 2048 = 3{,}261$ tokens, 51 blocks of 64; the f32 pool holds 65,536 tokens, 1,024 blocks, so $\lfloor 1024/51 \rfloor = 20$ pages. 6. $2 \times (128 \times 896 + 128) = 229{,}632$; with 14 key-value heads it would be $2 \times (896 \times 896 + 896) = 1{,}607{,}424$, seven times as many. 10. The grid 1 by 9 with 1,263 rows, for a page whose aspect ratio is 1/9 or thinner (a page 100 by 900 pixels); the second generation reads the same page as 1 by 6 with 1,121 rows, so the first spends more rows on this page (1,263 against 1,121) though its tiles are smaller, because of the newline per mosaic row. 11. $6{,}000{,}000/60{,}000 - 1 = 99$ seconds, so an operator who wants a bound of 60 seconds can divide by $1 + kt$ with $k > 99/60 = 1.65$ instead, which bounds the wait at $99/k$ seconds, or put a deadline on the wait.
