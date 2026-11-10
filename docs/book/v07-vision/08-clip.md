<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Contrastive Vision-Language Learning

*How can images and texts be placed in one space where matching pairs land close together?*

A classifier ends in a matrix. The network of [Chapter 3](03-cnns.md) finished in a layer with three rows, one per class, and asking it about a fourth thing was impossible in the plainest sense: there was no row to read. A network with a thousand rows is no better off about the thousand-and-first. Every row was learned from labelled pictures of its class, and the row for "a red square" does not exist until someone collects the pictures, names them and trains again.

Suppose the row came from somewhere else. Suppose a second network read the words "a red square" and wrote out a vector, and suppose it had been trained so that this vector sits where the first network puts pictures of red squares. Then the matrix at the end of the classifier would not be a parameter at all. It would be built at the moment of the question, one row for each sentence the user chooses to supply, and the classes could be anything that a sentence can say. Nothing about the first network would have to change.

That is the design of this chapter, and it asks for one thing that none of the earlier chapters had to provide: a **joint embedding space**, a single vector space into which two different networks, one that reads pictures and one that reads text, write their outputs, so that the vector of a picture and the vector of a sentence that describes it are close. The tool for building one is already in hand. [Chapter 14 of Volume V](../v05-sequences/14-encoders.md) trained an encoder so that two sentences of one meaning had a high cosine and the sentences of other meanings in the batch had a low one. Here the two members of a pair are no longer the same kind of thing, and one network has to be trained for each kind so that matching pairs end up close and the other pairings of a batch far apart. That is **contrastive pretraining**. The rest of the chapter works out what the change from sentences to pictures and captions brings, what a batch of pairs buys, how the learned sharpness of the softmax is fenced in, what a pairwise sigmoid loss does without a softmax at all, how the trained towers classify with nothing but a list of sentences, and what the same idea of "compare deep features" does for a different job, measuring how far apart two pictures look.

Six things from earlier chapters are needed first; the answers are at the end.

1. ([Volume V, Chapter 14](../v05-sequences/14-encoders.md), the contrastive loss.) A batch of 3 pairs has every cosine equal to 0.2. At any temperature, what is the loss of one direction?
2. ([Volume V, Chapter 11](../v05-sequences/11-decoding.md), temperature.) The logits $(2, 0)$ are divided by a temperature $\tau = 0.5$. What happens to their softmax, and what happens as $\tau$ goes to 0?
3. ([Volume V, Chapter 14](../v05-sequences/14-encoders.md), sentence embeddings.) Name two ways to turn the token rows of an encoder into one vector for the whole text. Why must a padded position not enter a mean?
4. ([Volume III, Chapter 3](../v03-neural-networks/03-softmax.md), softmax cross-entropy.) The logits are $(2, 0)$ and the correct class is the first. What is the gradient of the loss with respect to the first logit?
5. ([Volume VII, Chapter 6](06-vision-transformers.md), vision transformers.) A 224 by 224 picture is cut into patches of 14. How many tokens enter the encoder when a class token is added, and which row is read as the summary of the picture?
6. ([Volume VII, Chapter 2](02-convolution.md), output sizes.) A 224 by 224 picture meets a convolution with kernel 11, stride 4 and padding 2. What is the side of the output?

## Four pairs and a similarity matrix

Take a batch of four pictures and their four captions, already passed through the two networks, so that each is a vector of the joint space. Here the space has width 3, which is too small to be useful and large enough to print. The raw vectors are on the left of the first table below; what the loss sees is the cosine of every picture with every caption, a $4 \times 4$ matrix $\mathrm{cs}_{ij}$ whose diagonal is the four true pairs.

<!-- worked: clip.hand_batch -->
Four images and four captions, raw vectors of a joint space of width 3 (row i of each list is the pair i):

| pair | image vector | caption vector | image norm | caption norm |
|---|---:|---:|---:|---:|
| 0 | (3.0, 1.0, 0.0) | (2.0, 1.0, 0.5) | 3.1623 | 2.2913 |
| 1 | (0.0, 2.0, 1.0) | (-0.5, 3.0, 1.0) | 2.2361 | 3.2016 |
| 2 | (1.0, -1.0, 3.0) | (0.5, -2.0, 2.0) | 3.3166 | 2.8723 |
| 3 | (1.5, -1.0, 2.0) | (1.0, -1.5, 3.0) | 2.6926 | 3.5000 |

The cosines cs_ij = (image i . caption j) / (norm * norm), image i down the side and caption j across the top:

|  | j = 0 | j = 1 | j = 2 | j = 3 |
|---|---:|---:|---:|---:|
| i = 0 | 0.9661 | 0.1482 | -0.0550 | 0.1355 |
| i = 1 | 0.4880 | 0.9778 | -0.3114 | 0.0000 |
| i = 2 | 0.3290 | -0.0471 | 0.8923 | 0.9907 |
| i = 3 | 0.4863 | -0.2030 | 0.8728 | 0.9550 |

The logits S_ij = 10 * cs_ij (logit scale a = 10, temperature 0.1):

|  | j = 0 | j = 1 | j = 2 | j = 3 |
|---|---:|---:|---:|---:|
| i = 0 | 9.661 | 1.482 | -0.550 | 1.355 |
| i = 1 | 4.880 | 9.778 | -3.114 | 0.000 |
| i = 2 | 3.290 | -0.471 | 8.923 | 9.907 |
| i = 3 | 4.863 | -2.030 | 8.728 | 9.550 |

The softmax along each row of S, P_ij = exp(S_ij) / sum_k exp(S_ik): for image i, which caption belongs to it (image to text):

|  | j = 0 | j = 1 | j = 2 | j = 3 |
|---|---:|---:|---:|---:|
| i = 0 | 0.9994 | 0.0003 | 0.0000 | 0.0002 |
| i = 1 | 0.0074 | 0.9925 | 0.0000 | 0.0001 |
| i = 2 | 0.0010 | 0.0000 | 0.2718 | 0.7272 |
| i = 3 | 0.0064 | 0.0000 | 0.3034 | 0.6903 |

The softmax down each column of S, Q_ij = exp(S_ij) / sum_k exp(S_kj): for caption j, which image belongs to it (text to image):

|  | j = 0 | j = 1 | j = 2 | j = 3 |
|---|---:|---:|---:|---:|
| i = 0 | 0.9820 | 0.0002 | 0.0000 | 0.0001 |
| i = 1 | 0.0082 | 0.9997 | 0.0000 | 0.0000 |
| i = 2 | 0.0017 | 0.0000 | 0.5485 | 0.5882 |
| i = 3 | 0.0081 | 0.0000 | 0.4514 | 0.4117 |

The loss of each direction is -ln of the diagonal entry of its softmax:

| pair i | image to text: -ln P_ii | text to image: -ln Q_ii | mean of the two |
|---|---:|---:|---:|
| 0 | 0.0006 | 0.0182 | 0.0094 |
| 1 | 0.0075 | 0.0003 | 0.0039 |
| 2 | 1.3027 | 0.6005 | 0.9516 |
| 3 | 0.3707 | 0.8875 | 0.6291 |
| mean | 0.4203 | 0.3766 | 0.3985 |

With every cosine equal the loss would be ln 4 = 1.3863. The symmetric loss of the batch is 0.3985.

The derivative of the symmetric loss with respect to a logit is (P_ij + Q_ij - 2 [i = j]) / (2B): for the true pair of image 2 it is -0.1475 (the loss falls if S_22 rises), and for the rival caption 3 of image 2 it is 0.1644.
<!-- /worked -->

The notation is that of Chapter 14. The unit vector of picture $i$ is $\hat{\mathbf{a}}_i$ and that of caption $j$ is $\hat{\mathbf{b}}_j$, so that $\mathrm{cs}_{ij} = \hat{\mathbf{a}}_i \cdot \hat{\mathbf{b}}_j$. The logits are the cosines times a **logit scale** $a > 0$, the inverse of the temperature, $S_{ij} = a\,\mathrm{cs}_{ij}$. Here $a = 10$, which is a temperature of 0.1. Row $i$ of $S$ is a classification problem with four classes, "which of these four captions is mine?", and its cross-entropy is the loss of Chapter 14. Column $j$ of $S$ is the other problem: "which of these four pictures is the one this caption describes?" Their softmaxes are the matrices $P$ (along each row) and $Q$ (down each column) of the tables, and the loss of a batch of $B$ pairs is the mean of all $2B$ cross-entropies:

$$
\mathcal{L} = \frac{1}{2B}\sum_{i=1}^{B}\Bigl(\mathrm{lse}_j(S_{ij}) - S_{ii}\Bigr) + \frac{1}{2B}\sum_{j=1}^{B}\Bigl(\mathrm{lse}_i(S_{ij}) - S_{jj}\Bigr).
\tag{1}
$$

The first sum is the one Chapter 14 derived, written with log-sum-exp as there. This is the **InfoNCE** loss in the symmetric form that the CLIP paper trains with (Radford et al., 2021; the name and the idea of scoring a positive against in-batch negatives are from Oord, Li and Vinyals, 2018). What Chapter 14 showed about the first sum holds unchanged for each of the two, and this chapter does not repeat it: the gradient of a cross-entropy with respect to a logit is the softmax minus the one-hot label, so the derivative of (1) with respect to a logit is $(P_{ij} + Q_{ij} - 2[i = j])/(2B)$, which the last line of the table evaluates for the troubled pair.

Row 0 of the table is the easy case and it can be done by hand. Its logits are $(9.661, 1.482, -0.550, 1.355)$, their exponentials are $(15{,}692.18,\ 4.40,\ 0.58,\ 3.88)$ and sum to $15{,}701.04$, so the share of the true caption is $15{,}692.18/15{,}701.04 = 0.9994$ and the loss of the row is $-\ln 0.9994 = 0.0006$. Row 2 is the hard one. Picture 2 is nearly as close to caption 3 as to its own caption 2 (cosines 0.9907 and 0.8923, which the figure shows as two dark cells in one row), so its exponentials are $26.84,\ 0.62,\ 7{,}500.21,\ 20{,}066.39$ with sum $27{,}594.05$, the share of the true caption is 0.2718 and the row costs 1.3027 nats. Almost all of the loss of the batch is in rows and columns 2 and 3, which is where the learning signal should be.

![Three 4 by 4 panels of the printed batch, with the diagonal outlined: the cosines of each picture with each caption, the softmax along each row, and the softmax down each column. Rows 0 and 1 are almost one-hot on the diagonal; rows 2 and 3 split their weight between captions 2 and 3.](img/similarity-matrix.svg)

Why both directions? A loss on the rows alone would train every picture to find its caption among the four, and would not mind a caption that was the best match for every picture. The columns charge exactly for that. The two problems are different, too, in a way that Chapter 14 did not have to face: its two members were both sentences, so which was the anchor was arbitrary, and here picture and caption are different kinds of objects and neither deserves to be the anchor. Averaging the two keeps the loss symmetric in them. The mean over both directions is 0.3985 (0.4203 for picture to caption and 0.3766 for caption to picture), and the two directions disagree about which pair is hard: picture to caption it is pair 2 (1.3027, against 0.6005 the other way) and caption to picture it is pair 3 (0.8875, against 0.3707).

## What the batch size sets

Every pair in the batch is a negative for every other, and nothing else is: no extra picture or caption is encoded for the purpose. A batch of $B$ pairs therefore gives each picture $B - 1$ negative captions and each caption $B - 1$ negative pictures, and the batch size, which looked like a detail of the hardware, is the number of classes of the classification problem the loss poses. That has a floor under it that is easy to compute. A model that scores every candidate alike gives each a share $1/B$ and pays $\ln B$ per direction, so $\ln B$ is where training starts if the towers know nothing and the most a batch of $B$ can be said to teach. Oord and colleagues made the second half of that precise: the loss bounds from below the mutual information between the two sides, $I \ge \ln B - \mathcal{L}$, so a batch of $B$ can never certify more than $\ln B$ nats of shared information however good the model is.

<!-- worked: clip.batch_size_floor -->
A batch of B pairs gives every image B - 1 negative captions. A model that cannot tell the candidates apart pays ln B per direction, and since the information the loss can certify is at most ln B minus the loss itself, ln B is also the ceiling on what a batch of B can measure.

| B | negatives per image | ln B (nats) | log2 B (bits) |
|---|---:|---:|---:|
| 2 | 1 | 0.6931 | 1.000 |
| 4 | 3 | 1.3863 | 2.000 |
| 12 | 11 | 2.4849 | 3.585 |
| 256 | 255 | 5.5452 | 8.000 |
| 4,096 | 4,095 | 8.3178 | 12.000 |
| 32,768 | 32,767 | 10.3972 | 15.000 |
<!-- /worked -->

The batch of the CLIP paper had $B = 32{,}768$, a ceiling of 10.40 nats or 15 bits per pair, and a similarity matrix of $2^{30}$ entries, which is 4 GiB at four bytes each if it is stored in one piece. The loss is cheap in FLOPs compared with the towers (the product that forms $S$ costs $2B^2 d$ FLOPs, and so does each of the two that carry its gradient back to the vectors), and expensive in memory and in communication when the batch is spread over devices, a point the section on the sigmoid loss takes up.

There is a second, quieter cost: the negatives are only negatives if they differ in meaning. Chapter 14 showed in a toy language that a batch holding two pairs of one meaning forces a loss of at least $\ln 2$ on both anchors. The batches of the experiment below are drawn without replacement from the training classes for that reason; in web-scale data the same thing happens at a low rate and is simply tolerated.

## A scale that is learned and then held in

Equation (1) has one number that no pair of the batch supplies, the logit scale $a$. In Chapter 14 the temperature was a fixed constant chosen by the author. CLIP learns it. The parameter is the logarithm $\psi$, with $a = e^{\psi}$, so that $a$ stays positive and a step of the optimizer changes it by a fixed fraction rather than a fixed amount; it starts at $\ln(1/0.07)$, a temperature of 0.07, and after every update it is clamped so that $a \le 100$ (Radford et al., 2021). The gradient with respect to $\psi$ costs nothing extra, because $\partial S_{ij}/\partial\psi = S_{ij}$:

$$
\frac{\partial \mathcal{L}}{\partial \psi} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial S_{ij}}\,S_{ij}.
\tag{2}
$$

Why would a scale have to be fenced in? The next table says what the loss itself wants. It evaluates the printed batch and a second one, in which every diagonal cosine is 0.9 and every other cosine 0.1, at growing scales.

<!-- worked: clip.scale_sweep -->
The symmetric loss as the logit scale a grows, for the printed batch (pair 2 and pair 3 are confused) and for a batch whose four diagonal cosines are 0.9 and all others 0.1.

| logit scale a | printed batch | diagonal 0.9, others 0.1 |
|---|---:|---:|
| 1 | 0.948405 | 0.853558 |
| 5 | 0.420698 | 0.053490 |
| 10 | 0.398482 | 0.001006 |
| 1/0.07 = 14.285714 | 0.430287 | 0.000033 |
| 30 | 0.611605 | 0.000000 |
| 50 | 0.900433 | 0.000000 |
| 100 | 1.696263 | 0.000000 |
| 1000 | 16.760884 | 0.000000 |

The printed batch has an interior minimum: a = 7.888, loss 0.3921. Past it the loss grows linearly in a, with slope 0.01676: the sum of the 2 gaps by which a wrong cosine beats the diagonal of its row or column (0.0984, 0.0357), divided by 2B = 8. The second batch has no minimum: its loss falls towards 0 for ever.
<!-- /worked -->

The second batch is the one that matters. Every row and every column is ranked correctly, so every cross-entropy shrinks as $a$ grows, the derivative (2) is negative at every scale, and an optimizer that is allowed to move $\psi$ will do so for as long as the training lasts. Sharper is always better as far as the loss can tell. But a sharper softmax is not better for the towers. The gradient that reaches the vectors is $a(P_{ij} + Q_{ij} - 2[i = j])/(2B)$, so for any pair the model still gets wrong, the same factor $a$ multiplies the correction, and the first batch shows where this ends: with one rival that beats the diagonal the loss has a minimum at a finite scale (7.888) and above it grows in proportion to $a$. In a real batch of tens of thousands, some pairs are always wrong or ambiguous, the loss on them is paid at the full scale, and noise in the labels is amplified in the same proportion. The clamp at 100 is a guard on a quantity that has no natural stopping point.

The experiment below can only partly show this. In its 300 steps the learned scale moved from 14.29 to between 15 and 17 in the three runs, nowhere near the fence, so the clamp is never active. What it does show is the other side of the table: a scale of 100 held fixed trains slower and ends worse than the others (the scale table below). The clamp matters for the long, large runs it was designed for, not for a toy that finishes in seconds.

A decision never needs the scale. The class a picture is assigned to is the one with the largest cosine, and $\arg\max_c\, a\,\mathrm{cs}_c = \arg\max_c\, \mathrm{cs}_c$ for every $a > 0$. The scale is a property of training and of the probabilities it produces, not of the embedding space, which is one reason it is not one of the tensors in the checkpoint manifests of brain's CLIP crate (the manifest of `ClipTextConfig` in `crates/clip/src/config.rs` binds token and position tables, the blocks, a final norm and the optional text projection, and nothing else).

## Two towers into one space

The two networks that produce $\hat{\mathbf{a}}_i$ and $\hat{\mathbf{b}}_j$ are ordinary encoders ending in an ordinary linear map. The picture tower is the vision transformer of [Chapter 6](06-vision-transformers.md): patches, positions, unmasked blocks, and the final row of the class token as the readout, after a layer normalization and then a linear projection to the width of the joint space. The text tower is a transformer too, with two differences from the encoders of Chapter 14. Its attention is causal, as in a language model, and its readout is the hidden state at the end-of-text token, the one position that has seen every word, after a final layer normalization and a linear projection. In the original CLIP both projections are matrices without bias (the head of brain's EVA02 tower has one), and both outputs are scaled to length 1, which is the step that makes the dot product of (1) a cosine.

The text tower's input is where a lot of the care goes, because a tokenizer that differs from the training one is a silent change of the model. The CLIP tokenizer of `crates/data/src/clip_bpe.rs` lowercases and collapses whitespace, splits into words, runs byte-level BPE with an explicit end-of-word marker on the last piece of each word, wraps the ids in a start and an end token, truncates to 77 positions (so at most 75 content tokens) and pads. Which token pads is a property of the encoder, not of the vocabulary: SDXL's first text encoder pads with the end token and its second pads with `!`, which also splits the text at every literal `!` there, as the crate's header documents. The readout position is found by taking the first position that holds the largest id. The id of the end token is the largest in the vocabulary, and when the pad is the end token as well, every pad ties with the real end; the code, which asserts that the maximum it finds is the end token, takes the first maximum for that reason, and a comment beside it records that a last-wins maximum would silently read the last padded position instead. The towers entered brain in commit 8034b57a1, with that note already in the first version of `crates/clip/src/model.rs`.

Where the pictures are concerned, the preprocessing constants matter in the same way as in Chapter 1: the EVA02-CLIP configuration carries the per-channel mean $(0.48145466, 0.4578275, 0.40821073)$ and standard deviation $(0.26862954, 0.26130258, 0.27577711)$ of OpenAI's CLIP, the constants the tower was trained under.

<!-- worked: clip.tower_cost -->
Per forward pass of one text (or one picture): tokens, parameters read off the tensor manifests of `crates/clip/src/config.rs`, and FLOPs by the rule of Volume V, Chapter 9 (the four projections 8 T D^2, the feed-forward network 4 T D F or, gated, 6 T D F, the score and average products 4 T^2 D, the softmax 5 H T^2) summed over the blocks; the patch embedding, norms and activations are under one percent and left out. Parameters, with D the width and F the feed-forward width: a block has the query, key, value and output projections with biases (4 D^2 + 4 D), two LayerNorms (4 D) and a feed-forward network of two matrices with biases (2 D F + F + D); the EVA block has three such matrices (3 D F + 3 F + D) and two more LayerNorms, one of width D and one of width F (2 D + 2 F). The CLIP-L text tower is the 49,408-row token table, 77 position rows, 12 blocks (F = 3,072) and a final LayerNorm, without the text projection. The bigG text tower is the same with 1,280-wide rows, 32 blocks (F = 5,120) and the 1,280 by 1,280 projection (no bias), 20 heads. The CLIP-L/14 picture tower is the class token, a 14 by 14 patch convolution from 3 channels without bias, 257 position rows, the LayerNorm before the blocks and 24 blocks (F = 4,096), without the final LayerNorm and the projection. The EVA02 tower is the class token, the patch convolution with a bias, 577 position rows, 24 gated blocks (F = 2,730), the final LayerNorm and the projection to 768 with a bias.

| tower | tokens | parameters | GFLOPs per pass |
|---|---:|---:|---:|
| CLIP-L text tower (12 x 768) | 77 | 123,060,480 | 13.30 |
| OpenCLIP-bigG text tower (32 x 1280, with projection) | 77 | 694,659,840 | 97.88 |
| CLIP-L/14 image tower at 224 (24 x 1024) | 257 | 303,177,728 | 161.84 |
| EVA02-CLIP-L/14 image tower at 336 (24 x 1024, gated) | 577 | 304,522,928 | 381.81 |

One 224 by 224 picture through the CLIP-L/14 image tower costs 12.2 times one caption through the CLIP-L text tower. A zero-shot classifier pays the caption side once per class, then the picture side once per picture.
<!-- /worked -->

The cost table says what the design implies. A caption through the small text tower is more than an order of magnitude cheaper than a picture through the image tower, and they differ in kind: a caption is 77 tokens at width 768, a picture is 257 tokens at width 1024. That is why zero-shot classification is cheap on the text side. The class sentences are embedded once, however many pictures follow.

## Training the pair on shapes and colours

To see the machinery work, and to see where it stops working, the chapter trains a miniature of the whole thing. The world is a set of $6 \times 6$ pictures with three colour channels, each showing one of four shapes in one of four colours on a dark background. A caption is three words: a template, a colour and a shape.

<!-- worked: clip.shape_grids -->
The four shapes on the 6 by 6 grid (`#` is a pixel of the shape, `.` background); a picture of class (colour, shape) puts the shape's pixels at colour * brightness in the three channels, the background at 0, and adds noise to every number:

| row | square | frame | cross | diagonal |
|---|---|---|---|---|
| 0 | `......` | `......` | `......` | `#.....` |
| 1 | `.###..` | `.####.` | `..##..` | `##....` |
| 2 | `.###..` | `.#..#.` | `.####.` | `.##...` |
| 3 | `.###..` | `.#..#.` | `.####.` | `..##..` |
| 4 | `......` | `.####.` | `..##..` | `...##.` |
| 5 | `......` | `......` | `......` | `....##` |

Colours (R, G, B): red (1, 0, 0), green (0, 1, 0), blue (0, 0, 1), yellow (1, 1, 0). Pixels per shape: square 9, frame 12, cross 12, diagonal 11. Noise: normal with standard deviation 0.1; brightness uniform in (0.7, 1.0). Held-out classes (never in a training batch): red square, green frame, blue cross, yellow diagonal. Class index k = 4 * colour + shape.
<!-- /worked -->

There are 16 classes. Four of them, the "diagonal" of the colour-by-shape grid, are held out: no training batch ever contains a red square, a green frame, a blue cross or a yellow diagonal, although every colour and every shape appears in three other training classes. The captions of the held-out classes are made of words the towers have seen, in a combination they have not. That is the situation of a real zero-shot classifier, which meets a caption made of familiar words that no training pair contained.

The towers are as small as the two parts of the design allow, and the loss does not care what is inside them.

<!-- worked: clip.training_setup -->
The towers. Picture tower: the 108 numbers (channel-major) go through a linear layer to 16 units with a bias, tanh, and a linear projection without bias to 8 numbers, scaled to length 1. Caption tower: the three words of a caption (a template, a colour, a shape; template = `a`, `photo of a` or `drawing of a`, each one token) are looked up in an embedding table of 11 rows of 4 numbers, the three rows are concatenated (12 numbers; the slot of a word is its position), a linear layer to 12 units with a bias, tanh, and a linear projection without bias to 8 numbers, scaled to length 1. Parameters: 1,872 in the picture tower and 296 in the caption tower. Initialization of run r from `clip.init.r` (run 0: seed 0xa7524adef9184365 (12056781473946551141); runs 1 and 2 use the names with 1 and 2), `normal() * scale` for every entry, matrix by matrix and row by row, in this order: W1 [16, 108] with scale 1/sqrt(108), Wp [8, 16] with 1/sqrt(16), E [11, 4] with 0.5, V1 [12, 12] with 1/sqrt(12), Vp [8, 12] with 1/sqrt(12); both biases 0. The logit scale is a = exp(psi) with psi starting at ln(1/0.07) = 2.6593 (a = 14.286) and clamped to [0, ln 100] after every update. The classes are the 16 (colour, shape) pairs; the 4 held-out classes never appear in a training batch. Each step takes one batch of 12 pairs, one for each of 12 of the 12 training classes (a fresh `shuffle` of them, the first 12 kept), so that no two captions of a batch are equal. The training classes, before each shuffle, are [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14] in ascending order (the held-out classes [0, 5, 10, 15] removed); the rows of the embedding table E are the templates (rows 0 to 2), the colours (3 to 6) and the shapes (7 to 10). Pictures and templates come from `bookrng.generator("clip.stream.r.main")` (run 0: seed 0x2544d20a03ea71df (2685502218276860383)), the same stream that does the shuffle: each step first shuffles, then, for each pair in the shuffled order, draws the template `integer(3)`, the brightness `0.7 + 0.3 * uniform()` and the 108 noise values (channel-major, one `normal()` times 0.1 each). Adam (beta_1 0.9, beta_2 0.999, epsilon 1e-8, bias correction, learning rate 0.01, no weight decay) on all tower weights and, in the settings with a learned scale, on psi = ln a itself (the gradient is taken with respect to psi), 300 steps. The 128 test pictures (8 per class) come from `bookrng.generator("clip.test")` (seed 0xe8a1ca49eb169442 (16762901706875245634)): for class 0 to 15 in order and 8 pictures each, the brightness and then the 108 noise values. A picture is classified by the class whose prompt vector has the largest dot product with its vector (the first on a tie). A number that comes from a chaotic run is printed with only the digits that are the same in the run and in four copies of it whose first weight is changed by +1e-12, -1e-12, +3e-12 and -3e-12: in full if all agree on it, as 'about x' with the leading digits on which all agree, or as 'unstable' when not even one digit survives; the other digits cannot be reproduced in another implementation. The nudge is added to Wp[0][0], the first weight of the picture projection, which every picture uses.
<!-- /worked -->

A hand-derived backward pass through two towers and a loss is the kind of chain that is plausibly wrong in a way no review would see, and the repository's rule for such a thing is a gradient check against the number it can be compared with ([Volume III, Chapter 6](../v03-neural-networks/06-gradient-checking.md)), here for the softmax loss (1) and the sigmoid loss of a later section, including the derivative (2) with respect to $\psi$.

<!-- worked: clip.gradient_check -->
The initialized towers (run 0) on a batch of four pairs (classes 1, 6, 11, 12 with templates drawn from `bookrng.generator("clip.gradcheck")`, seed 0x98d9c14f2d67739c (11014046909519590300): for each pair in turn the template `integer(3)`, the brightness `uniform()` and the 108 noise values), the logit scale a = exp(1.3) and, for the sigmoid loss, the bias -0.4. The analytic gradient of the loss with respect to every tower weight, to the log scale psi and (sigmoid) to the bias against the central difference (L(w + h) - L(w - h)) / (2h), h = 1e-6 in 64-bit arithmetic, with the gate |analytic - numeric| <= 1e-7 + 1e-5 max(|analytic|, |numeric|).

| loss | parameters checked | largest absolute difference | pass the gate |
|---|---:|---:|---:|
| softmax | 2,169 | below 1e-8 | 2169 |
| sigmoid | 2,170 | below 1e-8 | 2170 |
<!-- /worked -->

Both pass, with differences below $10^{-8}$ at every one of more than two thousand parameters. Now the run. At each recorded step the table gives the learned scale, the loss on a fixed batch, and the accuracy of zero-shot classification, which is defined by the algorithm below.

## Zero-shot classification

Once the towers are trained, a class is a sentence. Given $C$ class names and a list of $T$ prompt templates, the class vector of a class is the unit-length mean of the caption vectors of its $T$ filled-in templates, and a picture goes to the class whose vector has the largest dot product with its own:

```text
function class_vectors(names: list<text>, templates: list<text>) -> array<real>[C, D]
    require len(names) >= 1 and len(templates) >= 1, "at least one class and one template"
    V = zeros(real, [len(names), D])
    for c in 0 .. len(names):
        z = zeros(real, [D])
        for t in 0 .. len(templates):
            u = encode_caption(templates[t], names[c])    # the text tower, projected and scaled to length 1
            for k in 0 .. D:
                z[k] = z[k] + u[k]
        n = norm(z)
        for k in 0 .. D:
            V[c, k] = z[k] / n
    return V

function classify(u: array<real>[D], V: array<real>[C, D]) -> int
    scores = zeros(real, [C])
    for c in 0 .. C:
        scores[c] = dot(V[c, :], u)
    return argmax(scores)                                 # the first class on a tie
```

`encode_caption` is the text tower followed by its projection and the scaling to length 1, and `u` is the picture tower's output scaled to length 1. **Zero-shot classification** is this procedure: classifying with class vectors computed at the time of the question from the classes' names, for classes that need not have had any labelled picture, with no weight of the classifier trained for them. The averaging over templates is **prompt ensembling**: a class name alone ("square") is ambiguous and a template ("a photo of a square") supplies context, and since no one template is best, averaging the embeddings of several is cheap, because the class vectors are computed once.

<!-- worked: clip.training_curve -->
Run 0, the main configuration. The evaluation loss is the symmetric loss of the 12 training classes (the first test picture of each, template `a`) at the current scale; accuracies are over the 16 class prompts with the ensemble of the three templates, on the 96 test pictures of the 12 training classes and the 32 of the 4 held-out classes (chance 1/16 = 0.0625). Numbers by the nudge rule above.

| step | logit scale a | evaluation loss | accuracy, training classes | accuracy, held-out classes |
|---|---:|---:|---:|---:|
| 0 | 14.29 | 6.5957 | 0.062 | 0.062 |
| 10 | 13.18 | 1.0330 | 0.594 | 0.094 |
| 20 | 12.80 | 0.6465 | 0.677 | 0.031 |
| 40 | 12.93 | 0.2109 | 0.917 | 0.312 |
| 60 | 13.38 | 0.0492 | 1.000 | 0.312 |
| 100 | 13.96 | 0.0155 | 1.000 | 0.219 |
| 200 | 14.81 | 0.0082 | 1.000 | 0.125 |
| 300 | 15.46 | 0.0046 | 1.000 | 0.094 |
<!-- /worked -->

![Zero-shot accuracy against the training step. The accuracy on the 12 training classes of run 0 rises to 1 by step 60. The accuracy on the 4 held-out classes of the three runs rises early to between 0.3 and 0.4, then moves apart: run 1 climbs to about 0.6, runs 0 and 2 fall back towards the chance line at 1/16.](img/zero-shot-by-step.svg)

The loss falls to nearly zero within a hundred steps and the accuracy on the training classes reaches 1. What the held-out classes do is the interesting part, and it is not what a reader hoping for zero-shot generalization wants to see. Their accuracy rises with the others at first, to about 0.31 in this run at step 60, and then falls back as the training classes are fit more tightly: 0.094 at step 300, close to chance (1/16 = 0.0625). The three initializations of the next table disagree with one another widely.

<!-- worked: clip.three_runs -->
The main configuration for the three initializations (the init generator of run r is `clip.init.r`, the stream generator `clip.stream.r.main`; seeds below are the init seeds), after 300 steps, ensemble accuracy; the last column is the best held-out accuracy at any of the 31 recorded steps (every 10th). Chance on 16 classes: 0.0625.

| run | init seed | training classes | held-out classes | best held-out |
|---|---:|---:|---:|---:|
| 0 | 0xa7524adef9184365 | 1.000 | 0.094 | 0.344 |
| 1 | 0x73a36061b7864127 | 1.000 | 0.625 | 0.656 |
| 2 | 0xf393b920c02494ff | 1.000 | 0.188 | 0.406 |
<!-- /worked -->

Run 1 ends at 0.625 on the held-out classes and run 0 at 0.094. So the honest statement is that a pair of towers trained on twelve of the sixteen combinations transfers to the other four only by luck of the initialization: the objective asked the towers to agree on twelve pairs, and it said nothing about the four others. There are many ways for two towers to place twelve points, and which of them extends sensibly to a thirteenth is decided by the inductive bias of the architecture and by how much of the combination space the training pairs cover. A real CLIP is trained on hundreds of millions of captions, so that the covered fraction of what a user will ask is enormous, and even there the compositions it never saw together are where it fails first. The toy runs the argument in miniature, and it supports the other half of the zero-shot story too: for the classes whose pairs were seen, the class vectors from sentences work at once and without a trained classifier row.

How much does the choice of template matter?

<!-- worked: clip.template_table -->
Zero-shot accuracy over all 128 test pictures and all 16 classes after the last step, with each template alone and with the ensemble of the three (the mean of the three prompt vectors per class, scaled back to length 1).

| run | `a` | `photo of a` | `drawing of a` | ensemble |
|---|---:|---:|---:|---:|
| 0 | 0.781 | 0.773 | 0.773 | 0.773 |
| 1 | 0.914 | 0.906 | 0.898 | 0.906 |
| 2 | 0.781 | 0.797 | 0.797 | 0.797 |
| mean | 0.826 | 0.826 | 0.823 | 0.826 |

Where the templates differ, on the same towers: the mean over the 16 classes of the length of the mean of the three unit prompt vectors (1 if they coincide), the number of the 128 pictures on which the three single-template answers are not all the same, the number the ensemble gets right while some single template gets it wrong, and the number it gets wrong while some single template gets it right.

| run | length of the mean prompt vector | pictures with disagreeing templates | ensemble gains | ensemble losses |
|---|---:|---:|---:|---:|
| 0 | 0.9949 | 2 | 0 | 1 |
| 1 | 0.9984 | 2 | 1 | 1 |
| 2 | 0.9986 | 4 | 2 | 0 |
<!-- /worked -->

In this world, not at all: the three templates were trained on the same pictures, so all three give nearly the same class vectors, the mean of three prompt vectors is almost as long as one, and the ensemble changes the answer on a handful of the 128 pictures in either direction. The mechanism is not wrong, only idle here. It pays when different templates carry different context that the pictures do depend on, which a real caption collection has and this toy does not.

## Scale, batch and loss on the same data

The three runs also give a way to ask whether the choices of the previous sections matter here. Every setting below trains the same towers from the same three initializations on the same number of pairs, 3,600.

<!-- worked: clip.scale_table -->
The same towers, data and 3,600 pairs for each setting, mean over the three runs (0, 1, 2; the stream generators are `clip.stream.r.<setting>` with the settings fixed_1, fixed_14, fixed_100, main and sigmoid, in the order of the rows), ensemble accuracy after the last step. The scale column is the final a; the next column is the training-class accuracy after 1,200 pairs.

| setting | final a | training accuracy after 1,200 pairs | accuracy, training classes | accuracy, held-out classes |
|---|---:|---:|---:|---:|
| fixed a = 1 (temperature 1) | 1.00 | 1.000 | 1.000 | 0.375 |
| fixed a = 14.286 (temperature 0.07) | 14.29 | 0.986 | 0.997 | 0.344 |
| fixed a = 100 (temperature 0.01) | 100.00 | 0.920 | 0.951 | 0.333 |
| learned, start 14.286, clamped at 100 | 15.86 | 0.986 | 1.000 | 0.302 |
| pairwise sigmoid loss, learned scale (start 10) and bias (start -10) | 15.86 | 0.997 | 1.000 | 0.427 |
<!-- /worked -->

On the training classes every setting ends at or near accuracy 1, and the early column separates them: a scale of 100 held fixed is the one that learns more slowly and does not get there. The held-out column does not separate anything. Its values, 0.30 to 0.43, lie inside the spread between runs of one setting (0.09 to 0.63 above), so the table cannot rank the scale settings or the sigmoid loss on held-out accuracy, and it would be wrong to say it does.

<!-- worked: clip.batch_table -->
Batch size with the number of pairs seen held at 3,600 (so 1,800, 900 and 300 steps), learned scale, mean over the three runs (stream generators `clip.stream.r.batch_2`, `clip.stream.r.batch_4` and `clip.stream.r.main`). The evaluation loss is that of the 12 training classes as one batch of 12 (its chance level is ln 12 = 2.4849), whatever the batch of training.

| batch size | negatives per picture | steps | ln B | evaluation loss | training accuracy after 1,200 pairs | accuracy, training classes | accuracy, held-out classes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 1800 | 0.6931 | 0.0107 | 0.951 | 0.993 | 0.448 |
| 4 | 3 | 900 | 1.3863 | 0.0027 | 1.000 | 1.000 | 0.375 |
| 12 | 11 | 300 | 2.4849 | 0.0043 | 0.986 | 1.000 | 0.302 |
<!-- /worked -->

The batch-size table has the same limit. The number of pairs seen is the same in every row, so a batch of 2 takes six times as many steps as the batch of 12. The held-out column is again inside the run-to-run spread, and the other columns are means over only three runs that do not order the rows monotonically (the evaluation loss is lowest at a batch of 4, the early accuracy highest there too). A table like this has nothing to say about batch size at this scale, and the lower bound of the section above is the part of the theory that does: a batch of 2 can never certify more than $\ln 2 = 0.693$ nats however long it trains. The claim that the quality of a pretrained model improves with batch size over a very wide range is one for runs of a scale that this chapter cannot reproduce.

## A loss with no softmax

The softmax in (1) is the coupling. Every entry of a row is divided by a sum over the whole row, so the gradient on one logit depends on all the others in its row and its column, and a loss on a batch of $B$ cannot be computed without all $B$ captions for every picture. The sigmoid loss of Zhai et al. (2023), called SigLIP, removes it by changing the question. Instead of "which of the $B$ captions is mine?", each (picture $i$, caption $j$) pair is its own binary question, "do these two belong together?", with the label $y_{ij} = +1$ on the diagonal and $-1$ elsewhere, and the binary cross-entropy of [Volume II, Chapter 3](../v02-learning/03-logistic-regression.md) on each of the $B^2$ pairs:

$$
\mathcal{L}_{\mathrm{sig}} = \frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{B} -\ln\sigma\bigl(y_{ij}\,(a\,\mathrm{cs}_{ij} + b_{\mathrm{sg}})\bigr) = \frac{1}{B}\sum_{i,j}\mathrm{softplus}\bigl(-y_{ij}\,(a\,\mathrm{cs}_{ij} + b_{\mathrm{sg}})\bigr).
\tag{3}
$$

The second form is the stable one. The logit scale $a$ is learned as before, and a learned bias $b_{\mathrm{sg}}$ is new: a batch has $B$ positives and $B^2 - B$ negatives, and with no bias the initial loss is dominated by the many negatives that the unit-length vectors cannot say "no" to, so the bias starts far below zero (here $-10$ in the experiment, $-5$ in the printed batch below) and the model starts out saying "no match" about everything. The derivative of one cell with respect to its logit is $-y_{ij}\bigl(1 - \sigma(y_{ij}(S_{ij} + b_{\mathrm{sg}}))\bigr)/B$, which depends on that cell alone.

```text
function sigmoid_loss(Uhat: array<real>[B, D], That: array<real>[B, D], a: real, b: real) -> real
    require a > 0.0, "the logit scale is positive"
    total = 0.0
    for i in 0 .. B:
        for j in 0 .. B:
            y = 1.0 if i == j else -1.0
            S = a * dot(Uhat[i, :], That[j, :])           # S_ij = a cs_ij
            total = total + softplus(-y * (S + b))
    return total / B
```

<!-- worked: clip.sigmoid_batch -->
The batch of the previous table, logit scale a = 10 and bias b = -5. First sigma(a cs_ij + b), the probability the pair (image i, caption j) is a match:

|  | j = 0 | j = 1 | j = 2 | j = 3 |
|---|---:|---:|---:|---:|
| i = 0 | 0.9906 | 0.0288 | 0.0039 | 0.0255 |
| i = 1 | 0.4699 | 0.9917 | 0.0003 | 0.0067 |
| i = 2 | 0.1531 | 0.0042 | 0.9806 | 0.9927 |
| i = 3 | 0.4657 | 0.0009 | 0.9765 | 0.9895 |

Then the cell losses softplus(-y_ij (a cs_ij + b)) = -ln sigma(y_ij (a cs_ij + b)), y_ij = +1 on the diagonal and -1 elsewhere:

|  | j = 0 | j = 1 | j = 2 | j = 3 |
|---|---:|---:|---:|---:|
| i = 0 | 0.0094 | 0.0292 | 0.0039 | 0.0258 |
| i = 1 | 0.6347 | 0.0084 | 0.0003 | 0.0067 |
| i = 2 | 0.1662 | 0.0042 | 0.0196 | 4.9142 |
| i = 3 | 0.6268 | 0.0009 | 3.7516 | 0.0105 |

The sum of the 16 cells is 10.2124: the 4 positive cells contribute 0.0479 and the 12 negative cells 10.1645. Divided by B = 4, the loss is 2.5531.

Locality. Raise the logit S_01 by 1 (cs_01 by 0.1). The softmax loss of row 0 moves from 0.000564 to 0.001046, since S_01 is in its normalizer. In the sigmoid loss only the cell (0, 1) moves, from 0.029216 to 0.077505; the cell (0, 0) stays at 0.009413 and so does every other cell.
<!-- /worked -->

The locality is the point. A softmax row is a single decision about the whole row, so moving one logit moves the whole row's loss, including the true pair's term. In the sigmoid loss a cell moves only when its own logit moves. That has a consequence for the cost, and it is worth being exact about which. The matrix of logits is $B^2$ entries either way, and forming it costs the same $2B^2 d$ FLOPs in both losses, so the saving is not in arithmetic. It is in what must be present at once. A softmax loss on a batch spread over several devices needs, for each picture, the normalizer of its row over every caption of the batch (and the same for each caption's column): either each device holds all the captions, or partial log-sum-exps are exchanged between devices, as the running maximum and sum of [Chapter 13 of Volume V](../v05-sequences/13-flash-attention.md) are. A sigmoid loss needs neither. A device computes a block of logits against whichever captions it holds, turns them into cell losses and discards them, and the next block of captions arrives from a neighbour.

<!-- worked: clip.sigmoid_scale -->
A batch of B = 32,768 pairs split over a number of devices, each holding B / devices pictures and their captions. A softmax loss normalizes each picture's row over all B captions, so a device holds (or streams with a running log-sum-exp) its rows against every caption. A sigmoid loss has no normalizer: a block of (B / devices) by (B / devices) logits is turned into its cell losses and discarded before the next block arrives.

| devices | pictures per device | logits against all B captions | logits in one block | bytes (32-bit), all B | bytes, one block | ratio |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 4,096 | 134,217,728 | 16,777,216 | 536,870,912 | 67,108,864 | 8 |
| 64 | 512 | 16,777,216 | 262,144 | 67,108,864 | 1,048,576 | 64 |
| 256 | 128 | 4,194,304 | 16,384 | 16,777,216 | 65,536 | 256 |
<!-- /worked -->

For the batch of the CLIP paper on 8 devices that is a block of 16.8 million logits (64 MiB) where holding the rows against every caption is 134 million (512 MiB), and the ratio grows with the number of devices. What the sigmoid loss gives up is the softmax's built-in competition among the captions of one picture, which the learned bias has to replace. How the two compare in accuracy at scale is an empirical question that the SigLIP paper reports on and that the toy above cannot settle: its row for the sigmoid loss ended with the same accuracy on the training classes as the softmax and a held-out figure inside the spread of the others.

## What the crates hold, and what they leave to the checkpoint

`crates/clip` holds three graphs behind one configuration: the CLIP-L and OpenCLIP-bigG text towers, the EVA02-CLIP-L image tower, and a vanilla CLIP-L image tower with two stems (OpenAI CLIP's, with a class token and a pre-norm, and SigLIP's, with no class token, a biased patch convolution and a final norm; knowledge #186 records why SigLIP is a stem around the same blocks and not a second tower, and commit f9fe812bc added it). The loss of this chapter is not among them, and neither is a logit scale: the towers are imported from released checkpoints and run forward, as the tensor manifests show. The text tower carries a backward pass, built so that it is bit-identical in its forward to the inference build and checked by a descent step on a proxy objective (`crates/clip/tests/it/train.rs`; commit 3b15554a1), and the CLIP and SigLIP image towers have one verified against finite differences (`crates/clip/tests/it/vision.rs`). Brain has no CLIP-style loss that trains the two together. Its InfoNCE objective is in `crates/sdk/src/embed_train.rs` (commit 92482da2b): the symmetric loss (1), computed on the host over unit vectors with in-batch negatives, used to train a linear projection head on frozen text embeddings, with a fixed temperature of 0.05 and no learned scale, and checked against finite differences over every parameter; `crates/sdk/src/embed_finetune.rs` feeds the same loss back through a live text encoder. The name `LogitScale` in `crates/model/src/logitscale.rs` is a different thing, the factor $1/T$ with which a sampler's temperature scales a language model's logits.

Two facts about the towers belong to the story of how a pretrained configuration is carried over, and not to the theory. The first is the activation. CLIP-L's configuration says `quick_gelu`, $x\sigma(1.702x)$, and OpenCLIP-bigG's says the exact-erf GELU; the two differ by up to two hundredths in one activation, which is not a rounding difference.

<!-- worked: clip.gelu_forms -->
CLIP-L's configuration says `quick_gelu`, x * sigmoid(1.702 x), and OpenCLIP-bigG's says `gelu_erf`, 0.5 x (1 + erf(x / sqrt 2)). Values at nine inputs (the library `erf`), and the largest absolute difference over x = -6, -5.999, ..., 6.

| x | gelu (erf) | quick gelu | quick minus erf |
|---|---:|---:|---:|
| -3 | -0.00405 | -0.01807 | -0.01402 |
| -2 | -0.04550 | -0.06434 | -0.01884 |
| -1 | -0.15866 | -0.15420 | 0.00445 |
| -0.5 | -0.15427 | -0.14961 | 0.00466 |
| 0 | 0.00000 | 0.00000 | 0.00000 |
| 0.5 | 0.34573 | 0.35039 | 0.00466 |
| 1 | 0.84134 | 0.84580 | 0.00445 |
| 2 | 1.95450 | 1.93566 | -0.01884 |
| 3 | 2.99595 | 2.98193 | -0.01402 |

The largest difference is 0.02033, at x = -2.27: a systematic deviation of about two hundredths in a single activation, applied by every hidden unit of every layer, which is why a tower must run the form its weights were trained with.
<!-- /worked -->

The configuration of the CLIP-L/14 body behind DeepSeek-OCR has a comment, added in commit 481bd84f8, saying that the upstream source applies quick GELU there while the configuration selects the erf form, and recording the finding in place without changing the value, because switching an already-shipped path needs its own re-gate against real-weight parity tests that skip when their weights are absent. At the time of writing the value and the comment still disagree, which is worth knowing as an example of a documented discrepancy rather than a fixed one. The second is the readout layer. For SDXL the text conditioning is not the final normalized output: it is the output of the second-to-last layer, before any normalization (`penultimate_layer` in the configuration: the layer with 0-based index 10 of CLIP-L's 12, and 30 of bigG's 32), and the pooled and projected EOS vector is a second conditioning input. FLUX.1 uses the pooled CLIP-L vector. PuLID reads the EVA02 tower's class embedding, scaled to length 1, taps five of its blocks, and its head is the one projection with a bias. In the vision-language models of the next chapter the image tower is used with no contrastive head at all: the SigLIP towers of DeepSeek-VL, Janus-Pro and Moondream 3 are read at the output of their final normalization, and the importer for the first two skips by name the 13 tensors of the attention-pooling head that the contrastive SigLIP uses to make its one picture vector, because those models never run it; the LLaVA tower is read at the output of its second-to-last block. These are the uses that Chapter 9 builds on. Throughout the crates the parity gate is a cosine of at least 0.9999 at every stage against dumped reference activations (`crates/clip/tests/it/parity.rs`), the ladder of 148 stage checks that commit 8034b57a1 reports.

## When pixels disagree with eyes

The same idea, comparing the activations of a pretrained network instead of the pixels themselves, answers a question that has nothing to do with text: how far apart do two pictures look? The pixel mean squared error of [Chapter 1](01-images.md) is the first thing anyone computes and it is a poor judge. Take a picture of fine texture and move it one pixel to the right. To a viewer nothing has happened, and in pixels everything has: each bright pixel now sits where a dark one was. Blur the same picture and the viewer sees a loss at once, while the pixel error is smaller. The toy below makes that ranking exact.

<!-- worked: clip.toy_picture_text -->
The reference picture, 12 by 12, `#` = 0.8 and `.` = 0.2: the top six rows are a vertical grating (even columns bright), the bottom six a horizontal grating (even rows bright).

```text
#.#.#.#.#.#.
#.#.#.#.#.#.
#.#.#.#.#.#.
#.#.#.#.#.#.
#.#.#.#.#.#.
#.#.#.#.#.#.
############
............
############
............
############
............
```
<!-- /worked -->

The feature extractor is fixed, chosen by hand, and minimal: three channels at every position of a $3 \times 3$ window that lies wholly inside the $12 \times 12$ picture (so $10 \times 10$ positions), the local mean of the nine values, and the absolute value of the second difference $(-1, 2, -1)$ taken along the middle row of the window (the three values of that row, weighted $-1, 2, -1$), and the same along the middle column. (The absolute value makes the sign of an alternation, which is the phase of a grating, irrelevant.) A second level is the same three maps, each averaged over non-overlapping $2 \times 2$ blocks (so $5 \times 5$ positions). What the construction does with them is the whole of the method. At each position the vector of channel values is divided by its length, so that the feature records which kinds of structure are present and not how strongly; the two pictures' unit vectors are subtracted, the differences squared, weighted per channel and summed; and the result is averaged over the positions of the level and summed over the levels:

$$
d_{\mathrm{lp}}(\mathbf{x}, \mathbf{x}_0) = \sum_{\ell}\frac{1}{n_\ell}\sum_{p}\sum_{c} w_{\ell c}\,\bigl(\hat f^{\ell}_{c}(p) - \hat f^{\ell,0}_{c}(p)\bigr)^2, \qquad \hat{\mathbf{f}}(p) = \frac{\mathbf{f}(p)}{\lVert\mathbf{f}(p)\rVert + 10^{-10}}.
\tag{4}
$$

Here $\mathbf{f}^{\ell}(p)$ is the vector of the $C_\ell$ channels of level $\ell$ at position $p$, $n_\ell$ is the number of positions of the level, and $w_{\ell c} \ge 0$ is a weight per channel, 1 in this toy.

```text
function perceptual_distance(F: list<array<real>[C, H, W]>, F0: list<array<real>[C, H, W]>, w: list<array<real>[C]>) -> real
    require len(F) == len(F0) and len(F) == len(w), "one weight vector for each level"
    total = 0.0
    for l in 0 .. len(F):
        acc = 0.0
        for r in 0 .. H:
            for c in 0 .. W:
                n = norm(F[l][:, r, c]) + 0.0000000001
                n0 = norm(F0[l][:, r, c]) + 0.0000000001
                for k in 0 .. C:
                    u = F[l][k, r, c] / n
                    u0 = F0[l][k, r, c] / n0
                    acc = acc + w[l][k] * (u - u0) ^ 2
        total = total + acc / (H * W)
    return total
```

`F` and `F0` are the feature maps of the two pictures at each level; `C`, `H` and `W` are those of the level `l` (the sizes of the levels differ), and `F[l][:, r, c]` is the vector of its channels at one position. The construction at a single position, for the reference and the picture blurred once:

<!-- worked: clip.one_position -->
Tap 1 at the top-left valid position (the 3 by 3 window of rows 0 to 2, columns 0 to 2), the reference against the picture blurred once: the three channel values, the same divided by the vector's length (plus 1e-10), and the squared difference of the unit vectors.

| channel | reference | blurred | reference, unit | blurred, unit | squared difference |
|---|---:|---:|---:|---:|---:|
| local mean | 0.6000 | 0.4750 | 0.4472 | 0.9878 | 0.2922 |
| horizontal texture | 1.2000 | 0.0000 | 0.8944 | 0.0000 | 0.8000 |
| vertical texture | 0.0000 | 0.0750 | 0.0000 | 0.1560 | 0.0243 |
| length / sum | 1.3416 | 0.4809 |  |  | 1.1165 |
<!-- /worked -->

The reference has fine horizontal texture here (the second difference is 1.2) and no vertical texture; the blur removes the texture, and what is left is mostly the local mean, so the unit vector has turned from $(0.447, 0.894, 0)$ to $(0.988, 0, 0.156)$, and the squared difference of the two unit vectors is 1.1165. Now all the distortions, with the pixel error beside the feature distance:

<!-- worked: clip.pixel_vs_feature -->
Each distortion of the reference picture, scored by the pixel mean squared error (mean of the 144 squared differences) and by the feature distance of the construction below, with weights 1 on each channel, tap by tap and summed. Noise is sd * z for one field z of 144 `normal()` values drawn row by row from `bookrng.generator("clip.perceptual.noise")` (seed 0xb018b2695c7e04db (12689088115709445339)), added without clipping. A shift wraps around the border (a shift by 2 returns the reference, a shift by 3 equals a shift by 1); a blur is the separable kernel [1, 2, 1] / 4 applied to rows then columns, wrapping; a contrast factor f maps v to 0.5 + f (v - 0.5).

| distortion | pixel MSE | feature distance, tap 1 | tap 2 | sum |
|---|---:|---:|---:|---:|
| shifted right by 1 | 0.1800 | 0.0314 | 0.0000 | 0.0314 |
| blurred 1 time | 0.0844 | 1.1239 | 1.0834 | 2.2073 |
| blurred 2 times | 0.0874 | 1.1608 | 1.1240 | 2.2848 |
| blurred 3 times | 0.0881 | 1.1829 | 1.1466 | 2.3295 |
| noise, sd 0.05 | 0.0024 | 0.0114 | 0.0065 | 0.0178 |
| noise, sd 0.1 | 0.0095 | 0.0462 | 0.0248 | 0.0710 |
| noise, sd 0.2 | 0.0378 | 0.1705 | 0.0895 | 0.2600 |
| noise, sd 0.3 | 0.0852 | 0.2867 | 0.1668 | 0.4535 |
| contrast times 0.75 | 0.0056 | 0.0129 | 0.0133 | 0.0262 |
| contrast times 0.5 | 0.0225 | 0.0898 | 0.0912 | 0.1810 |
| contrast times 0.25 | 0.0506 | 0.3856 | 0.3830 | 0.7687 |

By pixel error the picture shifted by 1 (0.1800) is further from the original than the picture blurred once (0.0844); by feature distance the order reverses (0.0314 against 2.2073).
<!-- /worked -->

![Pixel mean squared error against the feature distance, both on logarithmic axes. The picture shifted by one pixel is at the right of the plot (error 0.18) and near the bottom (distance 0.03). The three blurs are a tight group near error 0.085 and distance 2.2 to 2.3. Noise from sd 0.05 to 0.3 and the contrast reductions each rise along their own line below the blurs.](img/pixel-vs-perceptual.svg)

The picture shifted by one pixel has the largest pixel error of the table, 0.18, and a feature distance of 0.0314: the absolute value of the second difference of a period-2 grating is the same at either phase, and the local mean changes only a little. The blurred picture has less than half the pixel error and a feature distance 70 times larger (2.2073), because blur removes the fine texture that two of the three channels measure. A single blur and noise of standard deviation 0.3 have the same pixel error (0.0844 and 0.0852), and the blur is about five times further away by the feature distance (2.2073 against 0.4535). The toy is built so that the two scales disagree, and the claim it makes is only that: a distance computed on features can be insensitive to what pixels are sensitive to (a shift of a texture) and sensitive to what they underrate (the loss of local structure), and which it is depends on what the features measure. That the features of a deep network, calibrated on human judgments, measure what people see is the claim of Zhang et al. (2018), for whom classical measures such as PSNR and SSIM fell well behind; this toy does not test it.

The method that the toy copies in structure is called LPIPS, and a trained network supplies what was chosen by hand here. The levels are the activations at several depths of a pretrained classifier, the channel normalization and the squared difference are exactly those of (4), and the weights $w_{\ell c}$ are learned: they are fit so that the distance agrees with which of two distorted versions of a picture human judges said was closer to the original. The result is a **perceptual distance**: a distance between two pictures computed on the deep features of a pretrained network, normalized per channel and weighted by a learned vector, so that it measures how different the pictures look and not how different their pixels are.

## The metric in brain

`crates/lpips` implements version 0.1 of the metric with the AlexNet trunk (commit 346c856f1, then exposed as a capability in commit 0394036b5). The architecture is not a configuration; it is a table in `crates/lpips/src/config.rs` read off the reference sources: five convolutions, each followed by a ReLU, at which the features are tapped, with a $3 \times 3$ max-pool of stride 2 in front of the second and the third:

<!-- worked: clip.trunk_cost -->
The trunk of the LPIPS metric in brain (layer table of `crates/lpips/src/config.rs`) on a 224 by 224 picture: the five taps, the multiply-adds of each convolution (output positions * output channels * input channels * kernel^2), of its 1 by 1 head (positions * channels), and its parameters (1152 head weights besides, one per tap channel). The pooling in front of taps 2 and 3 is a 3 by 3 maximum with stride 2 and no padding.

| convolution | tap grid | multiply-adds | head multiply-adds | parameters |
|---|---:|---:|---:|---:|
| 3 -> 64, kernel 11, stride 4 | 55 x 55 | 70,276,800 | 193,600 | 23,296 |
| 64 -> 192, kernel 5, stride 1, max-pool in front | 27 x 27 | 223,948,800 | 139,968 | 307,392 |
| 192 -> 384, kernel 3, stride 1, max-pool in front | 13 x 13 | 112,140,288 | 64,896 | 663,936 |
| 384 -> 256, kernel 3, stride 1 | 13 x 13 | 149,520,384 | 43,264 | 884,992 |
| 256 -> 256, kernel 3, stride 1 | 13 x 13 | 99,680,256 | 43,264 | 590,080 |
| total, one picture |  | 655,566,528 | 484,992 | 2,469,696 |

A distance runs the trunk on both pictures: 1,311,133,056 multiply-adds (2,622,266,112 FLOPs) for the trunk, and 484,992 multiply-adds for the heads (one picture's worth: the squared differences of the two pictures are one tensor). The normalization, the squaring and the averaging are a few operations per tap value (484,992 tap values) and are not counted.
<!-- /worked -->

The tap grids are the 55, 27, 13, 13, 13 that the crate's own test asserts for a 224 by 224 input. The input in $[0, 1]$ is mapped to $[-1, 1]$ and then through the reference's scaling layer, $(x - \mathrm{shift})/\mathrm{scale}$ with shift $(-0.030, -0.088, -0.188)$ and scale $(0.458, 0.448, 0.450)$, folded into one per-channel multiply-add. Each head is a $1 \times 1$ convolution to one channel without bias, which is the per-channel weight vector $w_\ell$ of (4) applied to the squared differences and summed over channels in one step, and the distance is the sum over the five taps of each tap's spatial mean, which the type `Distance` returns together with the share of each tap.

The implementation is built from kernels brain already had, and the module comment says where each step of (4) goes. Both pictures run through the trunk as a batch of two, so that the features of the pair sit in one buffer and their difference is a single subtraction. The channel normalization uses a kernel that adds its $\epsilon$ to the sum of squares before the root, where the reference adds $10^{-10}$ to the norm; with $\epsilon = 10^{-20}$ the two agree to within $10^{-10}/|x|$ relative, and both give exactly zero where a ReLU map is zero in every channel. An optional mask weights the spatial average: the mask is area-averaged onto each tap's grid and the tap's distance is $\sum(d\,m)/\sum m$, the unmasked metric exactly when the mask is all ones, with the caveat that a convolutional feature's receptive field reaches past the edge of the mask.

How is it tested? Not against another run of itself. `crates/lpips/tests/it/reference.rs` holds the device to an independent f64 transcription of the reference formula, tap by tap, with and without a mask on a non-square pair, and compares with the distances that the official package gives on its own example pictures; it also checks the two properties that a distance must have to be one, that identical pictures are at exactly zero and that the distance grows with more blur and with more noise, and that pictures too small for the trunk or masks that cover nothing are refused rather than scored. The tests skip, by name, when the weights are not in the model store. In the next volume the same distance becomes a training signal: Volume VIII's chapter on discrete latents uses it as the perceptual loss of an autoencoder, and 3D reconstruction reports it beside PSNR and SSIM, as the crate's header says.

## Check your understanding

1. In the symmetric loss (1), suppose a model had a caption that was the nearest to every picture in the batch. Which direction of the loss notices, which does not, and what happens to the derivative with respect to the logits of that caption's column?
2. A batch of 8 pairs is used to train a model whose loss reaches 0.1. Using $I \ge \ln B - \mathcal{L}$, what can be certified about the information shared between pictures and captions, and why would a loss of 0.1 on a batch of 32,768 say more?
3. Someone proposes to clamp the logit scale at 1 "to keep the softmax honest". What happens to the loss of a batch with every diagonal cosine at 0.9 and every other at 0.1, and what happens to a zero-shot decision?
4. A zero-shot classifier scores 100 percent on classes whose (picture, caption) pairs were in the training set and near chance on a combination of two familiar words that never appeared together. Is that a bug? What would change it?
5. Find the error. "The sigmoid loss needs no softmax, so it needs no similarity matrix and its cost no longer depends on the batch size."
6. In (4), why is each position's channel vector divided by its length before the squared difference? What would a distance without it be dominated by, and what does it throw away?
7. A shifted picture has a larger pixel error than a blurred one in the toy, and a smaller feature distance. Name what in the feature extractor makes it nearly blind to the shift and what makes it see the blur.

## Summary

Two networks that write into one unit-length space, trained so that the cosine of a picture and its caption is high and that of every other pairing in the batch is low, are the whole of contrastive pretraining; its loss is the symmetric InfoNCE of (1), a softmax cross-entropy along the rows and along the columns of the matrix of cosines times a logit scale. The batch is the number of negatives and sets a ceiling of $\ln B$ nats on what it can certify; the scale is learned as a logarithm and clamped, because the loss alone would sharpen it for ever. A class becomes a sentence, and zero-shot classification is an argmax over the cosines with a list of caption vectors computed at question time, averaged over templates; the experiment shows it working on the classes whose pairs were seen and failing, run by run, on the compositions that were not. The pairwise sigmoid loss (3) makes every cell of the matrix its own binary decision and removes the normalizer, which changes what must be held at once and not the FLOPs. The same recipe of unit-normalized features from a network, differenced and weighted, gives a distance between pictures that can disagree with the pixel error in either direction, and brain's LPIPS reproduces the AlexNet form of it, tested against an independent transcription and the official package.

The limits are the toy's: shapes of six pixels, towers of two layers and a world of sixteen classes cannot say anything about batch sizes of tens of thousands, and the perceptual toy's features were chosen by hand and are shown to disagree with pixels, not to agree with people. What carries forward is the tower. Chapter 9 puts a language model behind a vision tower of this chapter's kind and has to decide how its tokens reach the decoder, and the contrastive pretraining that produced the tower is what makes its features something a language model can read.

## References

- A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger and I. Sutskever, *Learning transferable visual models from natural language supervision*, ICML 2021. The two towers, the symmetric loss on the matrix of cosines, the learned temperature clipped at 100, and zero-shot classification with prompt templates.
- X. Zhai, B. Mustafa, A. Kolesnikov and L. Beyer, *Sigmoid loss for language image pre-training*, ICCV 2023. The pairwise sigmoid loss with a learned scale and bias.
- A. van den Oord, Y. Li and O. Vinyals, *Representation learning with contrastive predictive coding*, 2018. The InfoNCE loss and the bound of mutual information by $\ln B$ minus the loss.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman and O. Wang, *The unreasonable effectiveness of deep features as a perceptual metric*, CVPR 2018. Deep features, channel-normalized, differenced, weighted by learned per-channel vectors and averaged, as a perceptual distance.

## Exercises

1. **[Calculate]** For the printed batch, compute by hand the loss of column 3 (text to image) from the cosines $(0.1355, 0, 0.9907, 0.9550)$ down that column at $a = 10$ and check it against the table.
2. **[Calculate]** For a batch of $B = 8$ pairs with every diagonal cosine 1 and every other cosine $-1$, give the one-direction loss at $a = 1$, $a = 5$ and $a = 100$. At what $a$ is it below $10^{-4}$?
3. **[Calculate]** A softmax row has its true logit $s$ and $B - 1$ other logits equal to 0. Find the scale at which the loss is one half of $\ln B$ for $B = 4$, assuming the true cosine is 0.9.
4. **[Calculate]** With $a = 10$, $b_{\mathrm{sg}} = -5$, what is the cell loss of a positive pair with cosine 0.9 and of a negative pair with cosine 0.9? What does the second say about the bias at this scale?
5. **[Calculate]** For the first convolution of the LPIPS trunk, check the output side 55 for a 224 by 224 input, and give its multiply-adds from the layer table.
6. **[Implement]** Reproduce `clip.hand_batch`: the cosine matrix, the logits, the row and column softmaxes and the loss of each direction for the four pairs.
7. **[Implement]** Reproduce `clip.gradient_check`: implement the symmetric loss, the sigmoid loss and their gradients for the two towers and the scale, and show agreement with central differences below the stated gate.
8. **[Implement]** Reproduce `clip.training_curve` and `clip.three_runs` from the setup of `clip.training_setup`: the generators, the draw orders, the towers and Adam, and the zero-shot accuracy at the recorded steps (up to the digits that the nudge rule permits).
9. **[Implement]** Reproduce `clip.pixel_vs_feature` and `clip.one_position` with the fixed feature extractor of the chapter.
10. **[Investigate]** Hold out two classes instead of four, then eight, and report the zero-shot accuracy on them over the three initializations. How does the share of combinations covered by training change what the toy can do on the rest?
11. **[Investigate]** Make the templates informative: render the pictures of one template with noise 0.3 and of another with noise 0, and test pictures of both kinds. Does prompt ensembling now help over a single template?
12. **[Investigate]** Replace the weights $w_c = 1$ of the toy distance with $(0, 1, 1)$ and then $(1, 0, 0)$. Which distortions change rank, and what does each choice say about what the metric measures?

## Key terms

- **contrastive pretraining**: training two networks, here one for pictures and one for captions, so that the vectors they write into a shared space are close for the two members of a pair and far for the members of different pairs, with the other pairs of the batch as the negatives. <!-- concept: contrastive-pretraining -->
- **InfoNCE**: the loss that scores each positive pair by a softmax cross-entropy against the negatives of the batch, written here in its symmetric form (rows and columns of the matrix of cosines times a logit scale), which also lower-bounds the shared information by the logarithm of the batch size minus the loss. <!-- concept: info-nce -->
- **zero-shot classification**: assigning an input to the class whose caption vector is nearest, with the class vectors computed at the time of the question from sentences naming the classes (averaged over several templates), so that classes need no labelled examples and no trained weight of their own. <!-- concept: zero-shot-classification -->
- **prompt ensembling**: averaging, for each class, the text vectors of several sentence templates that name it (for example "a photo of a square" and "a drawing of a square") and renormalizing, so that no single template's wording decides the class vector. <!-- concept: prompt-ensembling -->
- **perceptual distance**: a distance (LPIPS, learned perceptual image patch similarity, when the features are learned) between two pictures computed on the features of a pretrained network (channel-normalized at each position, differenced, squared, weighted per channel and averaged over positions and levels), which measures how different they look and need not agree with the pixel error. <!-- concept: perceptual-distance -->

## Answers

**Recall questions.**

1. $\ln 3 = 1.0986$: all candidates score alike, each gets a share of 1/3 whatever the temperature.
2. The logits become $(4, 0)$ and the softmax moves from $(0.8808, 0.1192)$ to $(0.9820, 0.0180)$, sharper. As $\tau \to 0$ it becomes one-hot on the largest logit, the argmax.
3. The row of the first token (the class or start token) and the mean of the rows of the real tokens; a padded position is not a word, and in a mean it would add a vector that depends on how much padding the batch happened to need, so that the same text would get different vectors in different batches.
4. $p_0 - 1$ with $p_0 = 0.8808$: $-0.1192$.
5. $16 \times 16 = 256$ patches plus one class token is 257 tokens; the final row of the class token is read.
6. $\lfloor (224 + 2\cdot 2 - 11)/4 \rfloor + 1 = 54 + 1 = 55$.

**Check your understanding.**

1. The row direction does not: each picture still finds its own caption if its own cosine is higher than that caption's, and a caption that is merely close to many pictures costs the rows only where it outranks their true caption. The column direction does: down the column of that caption every picture competes with the true one, the softmax $Q$ of the column spreads over many pictures and the loss of the column grows, and the derivative with respect to the logits of the column pushes the rival pictures' logits down in proportion to the share each has.
2. At most $\ln 8 - 0.1 = 1.98$ nats, out of a ceiling of 2.08. A loss of 0.1 on a batch of 32,768 certifies $10.40 - 0.1 = 10.30$ nats: the same loss was earned against 32,767 negatives instead of 7.
3. At $a = 1$ the diagonal-0.9 batch has loss about 0.85 (the first table of the scale section) and no scale above 1 is allowed, so the loss cannot fall; the gradient reaching the vectors is multiplied by a, so it is ten to fifteen times smaller than at the scales of the training. The zero-shot decision does not change, since the argmax of cosines does not depend on the scale.
4. It is not a bug: the objective asked only that the towers agree on the seen pairs, and nothing ties the extrapolation of the two towers to unseen combinations to each other. Training on more combinations of the same words, a larger and more varied collection, or an architecture whose text and image sides both represent colour and shape separately would change it.
5. The sigmoid loss has $B^2$ cells, one for every picture-caption pair, so it needs all of the cosines of the batch, and its arithmetic is the same $2B^2 d$ for the products as the softmax's. What it removes is the normalizer over the row, which changes what has to be present on one device at a time, not the number of logits.
6. Without it the distance would be dominated by the positions and channels with the largest activations, so that a high-contrast region counted more than a low-contrast one however similar their structure; with it, only the direction of the vector, which kinds of structure are present at the position, enters. What is thrown away is the strength at each position.
7. The absolute value of a second difference does not see the phase of a period-2 grating (the sign alternates, the size does not), and the local mean hardly depends on it, so a shift by one pixel leaves the vector at each position almost unchanged. A blur removes the texture: the two second-difference channels fall toward zero while the local mean stays, so the unit vector turns toward the mean channel at every position of the textured half.

**Exercises.** 1. The logits down column 3 are $(1.355, 0, 9.907, 9.550)$; the exponentials are $3.878, 1, 20{,}066, 14{,}045$ (to the digits shown), their sum is $34{,}116$, the share of the true entry is $14{,}045/34{,}116 = 0.4117$ and the loss is $0.8875$, as in the table. 2. One direction with the diagonal at 1 and the others at $-1$ is $\ln(1 + 7e^{-2a})$: at $a = 1$ it is $\ln(1 + 7 e^{-2}) = 0.6665$, at $a = 5$ it is $\ln(1 + 7e^{-10}) = 3.18 \times 10^{-4}$, at $a = 100$ it is $7e^{-200}$, below $10^{-85}$; it is below $10^{-4}$ when $7e^{-2a} < 10^{-4}$, $a > 5.6$. 3. The loss of a row with the true cosine 0.9 and three zeros is $\ln(1 + 3e^{-0.9a})$; it equals $\tfrac12\ln 4 = 0.6931$ where $1 + 3e^{-0.9a} = 2$, $e^{-0.9a} = 1/3$, $a = \ln 3/0.9 = 1.221$. 4. A positive at 0.9: $S + b = 4$, $\mathrm{softplus}(-4) = 0.0181$. A negative at 0.9: $S + b = 4$, $y = -1$, $\mathrm{softplus}(4) = 4.0181$. Two pictures that are similar and not a pair are punished by 4 nats with this bias, so at this scale the bias says a negative must have a cosine below 0.5 to be safe ($a\,\mathrm{cs} + b < 0$). 5. $\lfloor(224 + 4 - 11)/4\rfloor + 1 = 55$; $55 \cdot 55 \cdot 64 \cdot 3 \cdot 11 \cdot 11 = 70{,}276{,}800$ multiply-adds. 6 to 9. Each reproduces its named worked example. 10 to 12. Each is an experiment on the worked module; state the generators and the draw order you used. Expect the held-out accuracy to depend strongly on the initialization at every setting, as in the chapter.
