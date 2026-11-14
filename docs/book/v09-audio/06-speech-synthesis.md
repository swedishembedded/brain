<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Speech Synthesis

*How does a model speak a sentence in a given voice?*

Recognition was handed its input and had to find the words. Synthesis is handed the words and has to find the sound, and the first thing it has to decide is something the words do not say: how long the sound lasts. Nothing in "The quick brown fox jumps over the lazy dog." tells a model that speaking it takes 3.04 seconds. A synthesiser that writes speech one frame at a time settles the question as it goes: at every step it may write an end-of-speech token, and the clip is exactly as long as the steps before it. The Qwen3-TTS crate of brain, `crates/qwen3tts`, works this way, and when it got its first end-to-end quality check on 3 September 2026, a round trip through a speech recogniser, the first real run found a clip of this sentence that was 200 frames long, 16 seconds at 80 milliseconds a frame, and silent from its first frame (commit `fee25c3f0`). A healthy run of the same sentence ends after 38 frames (commit `375758271` gives the figure). The model had not misjudged a length. It had lost the ability to end.

That failure holds the whole problem of synthesis in its smallest form. A synthesiser writes a long sequence one step at a time, nothing says how long the sequence must be, and what it writes is not yet sound. **Text-to-speech** is the task of turning a text, and usually a description of a voice, into the waveform of someone saying it. Two of brain's crates do it by two different routes, and Chapter 3 and Chapter 5 supplied one half of each: the codec of Chapter 3 that turns audio into tokens a language model can write, and the speaker vector of Chapter 5 that says whose voice it is. What is left is the part in the middle, taken here in the order the routes meet it: how a language model is made to write sound, how its choices are drawn, how a voice reaches it, and then the second route, which cuts the sentence in a different place and needs an iterative decoder and a vocoder to get back to samples.

Six earlier ideas come into play as soon as the first sound is synthesised.

1. ([Chapter 3](03-audio-codecs.md), audio tokens.) How many integers a second does brain's codec write, and which of the 16 codes of a column comes from a separate codebook group?
2. ([Volume V, Chapter 4](../v05-sequences/04-language-modeling.md), the causal language model.) How does a causal language model score a sequence, and what does teacher forcing give it as input during training?
3. ([Volume V, Chapter 11](../v05-sequences/11-decoding.md), repetition penalty and top-p.) Under a repetition penalty of 1.25, what happens to a seen token's logit of $3.0$ and to one of $-2.0$? Which tokens does a nucleus of 0.8 keep from the probabilities $(0.5, 0.3, 0.1, 0.1)$?
4. ([Volume VIII, Chapter 6](../v08-image-generation/06-flow-matching.md), flow matching.) In the book's convention which end of the path is noise, and what is one Euler step of the velocity field?
5. ([Volume VIII, Chapter 7](../v08-image-generation/07-samplers-and-guidance.md), classifier-free guidance.) Write the guided velocity $u_g$ in terms of $u_c$ and $u_{\mathrm{un}}$, and say what $g = 1$ and $g > 1$ give.
6. ([Chapter 2](02-spectrograms.md), front ends.) A clip of 24,000 samples goes through a log-mel front end with a window of 1920 samples, a hop of 480 and a reflect pad of 720 on each side. How many frames does it make?

## Two places to cut the sentence

A sentence of ten words is a few hundred bits. One second of its speech at 24 kHz in 16-bit samples is 384,000 bits, and between the two lies everything that makes a voice a voice: the pitch contour, the length of each vowel, the breath before a clause, the room. A synthesiser has to put an interface somewhere between the text and the waveform, a representation that a first model can write from the text and a second can turn into samples, and the choice of that interface decides almost everything else about the system. The table counts what each candidate carries in a second.

<!-- worked: speech_synthesis.interface_rates -->
The representations a text-to-speech system passes through, per second of speech: steps a second, numbers per step, bits per number (a codebook of 2048 entries is 11 bits, 6561 speech tokens are log2 6561 bits), and the product of each.

| representation | steps per second | numbers per step | bits per number | numbers per second | bits per second |
|---|---:|---:|---:|---:|---:|
| waveform, 16-bit samples | 24000 | 1 | 16 | 24,000 | 384,000 |
| log-mel spectrogram, 32-bit floats | 50 | 80 | 32 | 4,000 | 128,000 |
| codec codes of the 16-codebook synthesiser | 12.5 | 16 | 11 | 200 | 2,200 |
| only the first codebook of those | 12.5 | 1 | 11 | 12.5 | 137.5 |
| speech tokens of the single-token synthesiser | 25 | 1 | 12.68 | 25 | 317 |
<!-- /worked -->

The two crates choose different rows. Qwen3-TTS puts the interface at the codec's codes, 200 integers a second, and lets one large model write them all; the decoder that follows is the codec decoder of Chapter 3. A language model that is trained on such codes, reads text (and a voice) and continues a stream of codes is a **codec language model**. The CosyVoice crate, `crates/cosyvoice`, puts the interface higher and thinner: a stream of 25 speech tokens a second from a vocabulary of 6561, 317 bits a second, which says what is being said and not much about how. A language model writes those, a second model (a flow-matching decoder, the generator of Volume VIII, Chapter 6, given a new target) turns them into a mel spectrogram, and a third model, a vocoder, turns the spectrogram into samples. The figure draws both on the same scale.

![Two rows of boxes. The codec synthesiser takes text tokens and a language tag, plus a 1024-number speaker vector entering the Talker, through a Talker that writes code 0 at 12.5 frames a second, a code predictor that writes codes 1 to 15, and a codec decoder that turns 16 codes into 1920 samples, ending in a 24,000 Hz waveform. The speech-token synthesiser takes text tokens and reference text into a 24-layer speech-token language model at 25 tokens a second, helped by reference tokens, then a flow decoder of 10 Euler steps producing 50 by 80 mel values a second, helped by a speaker vector that goes from 192 to 80 numbers, then a vocoder, ending in a 24,000 Hz waveform.](img/synthesis-pipelines.svg)

Neither cut is free. The narrow interface of the second route carries so little that the speaker, the pitch and the room must be supplied by the models after it, which is why they are each given a vector or a reference spectrogram. The wide interface of the first carries nearly everything, so one model must write 16 numbers per step and get them mutually consistent. What each choice costs can be counted in parameters and multiply-adds from the crates' configurations, and the two routes' histories show where each is fragile.

## A language model whose words are columns of sound

The Talker is a Qwen3-style decoder of 28 layers and width 1024, grouped-query attention with 16 query heads and 8 key-value heads of width 128 (the decoder family of Volume VI, Chapter 1), and its vocabulary is the codec's first codebook plus a window of special ids. It writes one code of the first codebook per frame; a second network, the code predictor, writes the other fifteen. That makes the Talker an ordinary causal language model, with one change that matters: what it reads back is not the code it wrote but the whole column.

When frame $s$ has been written, its 16 codes are looked up, each in a table of its own (the Talker's table for the first, the code predictor's fifteen tables for the others), and the 16 rows are **added** into one vector of width 1024, to which the next row of the text stream is added too (the stream is described just below). The Talker therefore advances one position per column, not one per code. The saving is the same quadratic argument as in Chapter 3. For ten seconds of speech, 125 columns are 125 positions and $125 \cdot 126 / 2 = 7{,}875$ query-key pairs per head and layer; the 2000 codes written out one by one would be 2000 positions and $2000 \cdot 2001 / 2 = 2{,}001{,}000$ pairs, 254 times as many. The price is that a sum of 16 rows is a lossy way to present a column, and that the 16 codes of a column are not drawn together: the first comes from the Talker's output and the other fifteen one after another from that second network.

What does the model read before the first frame, and how does the text get in? The answer is a stream laid alongside the sound, and its layout is worth seeing exactly, because it fixes what the Talker can and cannot know. The prompt builder, `crates/qwen3tts/src/prompt.rs`, assembles a row for every position, and each row is a sum of a codec side and a text side. The codec side starts with a tag that says whether the model should think about a language (a think tag, a begin tag, a language row, an end tag; or a no-think form without the language row), then the speaker vector as one row of width 1024 (the vector of Chapter 5, placed unchanged, which is why the prompt builder asserts that its length equals the model's width), then a padding row and a begin-of-speech row. The text side is a three-row role header and then the text tokens, projected by a small two-layer network from the width of the text embeddings, 2048, to 1024. The text is not all put before the sound. Its first token is summed onto the begin-of-speech row, and every later token is held back and added, one per frame, onto the row that frame's codes make.

<!-- worked: speech_synthesis.text_channel -->
A prompt in the order the code builds it. The codec side is m rows: the think or no-think tag (3 rows, or 4 with a language row), an optional speaker row, then the padding row and the begin-of-speech row. The text side is the role header of 3 rows, then the first m - 1 codec rows each summed with a text-side row (padding, and the begin row of the text on the last of them), then the first text token summed with the codec side's begin-of-speech row. That is 3 + (m - 1) + 1 = m + 3 rows before frame 0. The in-context form puts 3 + (m - 1) rows first and then the reference block, one row for each reference frame and one for the begin row. 'Trailing rows' are the text rows still to be fed, one per frame. The target text has 6 tokens, the reference text 20.

| form | tag rows | speaker rows | m | rows before frame 0 | trailing rows | text read by |
|---|---:|---:|---:|---:|---:|---|
| no voice, language 'auto' | 3 | 0 | 5 | 8 | 6 | frames 0 .. 5 |
| x-vector voice, a named language | 4 | 1 | 7 | 10 | 6 | frames 0 .. 5 |
| in-context clone, 125 reference frames | 4 | 1 | 7 | 135 | 1 | text rows (27) are summed onto the 126 reference rows, the rest is padding |

The row the Talker reads after frame s is generated, for the x-vector form above (target text of 6 tokens, so 6 trailing rows):

| frame fed back | codec part of the row | text part of the row |
|---|---|---|
| 0 | the sum of the 16 codec rows of frame 0 | text token 2 |
| 1 | the sum of the 16 codec rows of frame 1 | text token 3 |
| 2 | the sum of the 16 codec rows of frame 2 | text token 4 |
| 3 | the sum of the 16 codec rows of frame 3 | text token 5 |
| 4 | the sum of the 16 codec rows of frame 4 | text token 6 |
| 5 | the sum of the 16 codec rows of frame 5 | end-of-text row |
| 6 | the sum of the 16 codec rows of frame 6 | padding row |
| 7 | the sum of the 16 codec rows of frame 7 | padding row |
| 8 | the sum of the 16 codec rows of frame 8 | padding row |
<!-- /worked -->

The consequence is a fixed text rate: the model is shown at most 12.5 text tokens a second, one beside each column, and a sentence of $n$ tokens fills the first $n$ slots of the stream and then padding rows follow. A clip of 38 frames has 38 slots, so unless the speech is very short the text ends before the sound does, and from then on the model has only its own past to go on, which includes the end-of-text row it saw once. The in-context form, which comes below, instead lines the text up against the reference clip's frames and feeds the rest later. Nothing outside the prompt tells the model how long to speak. The length is a decision it makes by drawing the end-of-speech id.

## Teaching it: one shift, not two

The training of a codec language model is teacher forcing (Volume V, Chapter 4) on a stream of frames, with the loss on the code each position must write. The one thing to get right is the alignment of inputs and targets, because a loss function that shifts its labels internally, applied to labels that were already shifted, scores every position against the wrong frame without any error; `crates/qwen3tts/src/sft.rs` shifts exactly once, explicitly, in the data. A tiny case shows the rule.

<!-- worked: speech_synthesis.label_alignment -->
Three frames of four codes, codes[f][q] with the first code q = 0: frame 0 is (5, 1, 2, 3), frame 1 (7, 4, 0, 6), frame 2 (9, 8, 1, 2). The first-code model reads one summed row per frame and is scored once against the next frame's first code (one shift). The residual model of a frame reads a short sequence whose position k holds code k - 1 of the frame (position 0 holds the Talker's hidden state) and is scored against code k of the same frame (no shift).

| row read | its input | its target | target value |
|---|---|---|---|
| Talker position 0 | sum of the codes of frame 0 | first code of frame 1 | 7 |
| Talker position 1 | sum of the codes of frame 1 | first code of frame 2 | 9 |
| Talker position 2 | sum of the codes of frame 2 | nothing (the end row has no label here) | - |

| row read | input at this position | its target | target value |
|---|---|---|---|
| frame 0, position 1 | code 0 of frame 0 (after the Talker's hidden state at position 0) | code 1 of frame 0 | 1 |
| frame 0, position 2 | code 1 of frame 0 | code 2 of frame 0 | 2 |
| frame 0, position 3 | code 2 of frame 0 | code 3 of frame 0 | 3 |
| frame 1, position 1 | code 0 of frame 1 (after the Talker's hidden state at position 0) | code 1 of frame 1 | 4 |
| frame 1, position 2 | code 1 of frame 1 | code 2 of frame 1 | 0 |
| frame 1, position 3 | code 2 of frame 1 | code 3 of frame 1 | 6 |
| frame 2, position 1 | code 0 of frame 2 (after the Talker's hidden state at position 0) | code 1 of frame 2 | 8 |
| frame 2, position 2 | code 1 of frame 2 | code 2 of frame 2 | 1 |
| frame 2, position 3 | code 2 of frame 2 | code 3 of frame 2 | 2 |

A second shift in the loss would score Talker position 0 against frame 2's first code, 9, and leave position 1 with no target; the first-code targets would be a list of length 1 instead of 2.
<!-- /worked -->

The Talker's position $p$ has read the frames up to $p$, so its output is scored against the first code of frame $p+1$: one shift. The code predictor works inside a frame, in a short sequence that starts from the Talker's hidden state, and its position $k$ has read the code $k - 1$ of the same frame and is scored against code $k$ of that same frame: no shift. If the loss shifted again, the first-code targets would skip a frame and every residual target would be taken from the wrong column, and nothing would crash, because every shape is still right. The crate's unit tests pin the indices on a tensor small enough to check by hand for that reason: when the wrong answer has the right shape, the test has to know the right one.

## Fifteen more codes per column, and where the work goes

The code predictor, the multi-token-prediction network of the crate (`crates/qwen3tts/src/mtp.rs`), is a 5-layer decoder of the same layer shape as the Talker. For each frame it reads a sequence of 16 rows: the Talker's hidden state at the position that wrote the first code, then the first code's embedding, then the embedding of each residual code as it is drawn, and its position $k$ reads the head that predicts code $k$. Writing a frame therefore costs one Talker step and sixteen decoder steps of a smaller network. Whether that is a good division of the work is a matter of counting.

<!-- worked: speech_synthesis.talker_counts -->
Parameters from the tensors of the configuration (a 1024-wide, 28-layer Talker with 16 query heads and 8 key-value heads of 128, feed-forward 3072; a 5-layer code predictor of the same layer shape; layer = q, k, v and output projections, gate, up and down matrices, two norms and the two query-key norm vectors). Multiply-adds per frame count each matrix product once per row it is applied to; attention scores and norms are left out. A layer holds q (1024 x 2048), k and v (1024 x 1024 each), o (2048 x 1024), gate and up (1024 x 3072 each) and down (3072 x 1024), with no bias on any, two RMSNorm weights of 1024 and a query norm and a key norm of 128: 15,730,944 parameters (15,728,640 of them in the matrices, which is also the layer's multiply-adds a row). The Talker's embedding row is the codec table (3072 x 1024), the Talker's final norm (1024), the text table (151936 x 2048) and the text projection (2048 -> 2048 -> 1024, both with bias); the head row is the 1024 x 3072 codebook-0 head, counted separately from the codec table. The code predictor has 5 such layers, a final norm, and for each of the 15 residual codes a head and an embedding table of 2048 x 1024.

| component | parameters | decoder steps per frame | multiply-adds per frame (millions) |
|---|---:|---:|---:|
| Talker: 28 layers | 440,466,432 | 1 | 440.4 |
| Talker: codebook-0 head (1024 -> 3072) | 3,145,728 | 1 | 3.1 |
| Talker: embedding tables and text projection (not matrix products per frame) | 320,606,208 | 0 | 0.0 |
| code predictor: 5 layers and final norm | 78,655,744 | 16 | 1,258.3 |
| code predictor: 15 heads (1024 -> 2048) and 15 embedding tables | 62,914,560 | 15 | 31.5 |

| total | parameters | multiply-adds (millions) |
|---|---:|---:|
| one frame (80 milliseconds of speech) | 905,788,672 | 1,733.3 |
| one second of speech (12.5 frames) |  | 21,666.2 |
| one second of speech, billions |  | 21.67 |

The Talker holds 764,218,368 parameters and the code predictor 141,570,304. The code predictor runs 16 decoder steps of 5 layers, 80 layer-steps, per frame; the Talker runs 28. Re-running the whole 16-row sequence for each of the 15 codes, as the first version did, is 1200 layer-rows per frame, 15 times as many.
<!-- /worked -->

The code predictor is small in parameters, 141,570,304 of the 905,788,672, and is not small in work: it runs 80 layer-steps a frame against the Talker's 28, so the code predictor does 2.9 times the layer arithmetic of the Talker, 1,258.3 million of the 1,733.3 million multiply-adds of a frame. A second of speech is 12.5 frames, 21.67 billion multiply-adds, before the codec decoder's 31.0 billion from Chapter 3.

That arithmetic explains an episode in the crate's history. The first version re-ran the whole fixed 16-row sequence for each of the 15 residual codes. The attention is causal, so row $k$'s hidden state depends only on rows up to $k$ and the other rows were wasted: $15 \cdot 16 \cdot 5 = 1200$ layer-rows a frame against the 80 that a key-value cache needs, 15 times as many.

<!-- perf-number: historical measurement, commit d5a0c4113 -->
The message of commit `d5a0c4113` reports the profile that found it, on one integrated GPU on 4 September 2026: the code predictor took 5771.6 milliseconds a frame against the Talker's 233.9, and after the key-value cache, a better kernel for single-row products and a head read once instead of fifteen times, 300.5 against 120.8. What the code lets a reader check is the structure: `generate_residuals_with` in `mtp.rs` fills a cache one position at a time, the slow version survives as `generate_residuals_recompute`, and the test `kv_cached_residuals_match_the_full_recompute` requires the two to give the same codes at both of the checkpoint's shapes. The recompute is kept as an oracle, the pattern that Volume V, Chapter 12, called cache equivalence: a speed-up is accepted only beside the slow answer it must reproduce.

The same type holds a second lesson of that family. The 1.7-billion variant of the model has a Talker of width 2048 and a code predictor of width 1024, joined by a projection the checkpoint ships, and for a while the model code assumed the two widths always agreed, as they do in the variant the checkpoint names 0.6B: every run of the larger model stopped at an assertion until the projection was loaded and applied (commit `5f61b9068`). A configuration value that exists, is parsed and is read by nothing is knowledge #43 again, in a new place.

## Drawing one token from the Talker's 3072

The Talker's output head has 3072 entries, not 2048. Its first codebook has 2048 codes, and the remaining 1024 ids are specials: the tags the prompt uses, the padding, begin and end ids, and the preset speakers' ids. A draw from the raw logits could pick any of them. The sampler of `crates/qwen3tts/src/sampling.rs` is a chain of six stages that every decoding loop of the crate goes through, and Volume V, Chapter 11, taught each of the last five separately; what is new here is the first, which needs the structure of this vocabulary.

1. **Mask.** The last 1024 ids are set to $-\infty$, so that no special can be drawn as a speech code; the end-of-speech id, which lives in that window, is restored only once a minimum number of frames has been written.
2. **Repetition penalty.** Once for each distinct id that has been generated.
3. **Temperature.** The logits are divided by it.
4. **Top-k.** Everything below the $k$-th largest scaled logit is removed.
5. **Top-p.** The shortest prefix of the ranked probabilities that reaches the cut is kept.
6. **Draw.** A softmax over what is left and a walk along its cumulative distribution until it reaches one uniform number.

A toy with 12 ids, the last 4 of them the suppressed window, shows each stage's effect, with a penalty of 1.25 so that it is visible.

<!-- worked: speech_synthesis.sampling_chain -->
A toy chain: 12 ids, the last 4 (ids 8 .. 11) are the suppressed window and id 10 is the end-of-speech id; the generated history is [3, 3, 5, 3]; penalty 1.25, temperature 0.9, top-k 5, top-p 0.9. The real default penalty is 1.05, and the real window is the last 1024 of 3072 ids. The order is the code's: mask, penalty (once per distinct id, positive logits divided and negative ones multiplied), division by the temperature, top-k (entries below the k-th largest are removed, a tie at the threshold stays), top-p (the shortest prefix of the ranked probabilities whose sum reaches the cut), then a softmax over what is left and an inverse-CDF draw with one uniform u (scan the ids in order, stop at the first whose running sum reaches u times the total).

End-of-speech not yet allowed (fewer than the minimum number of frames so far):

| id | logit | after the mask | after the penalty | after temperature | after top-k | after top-p | probability |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.0 | 2.000 | 2.000 | 2.222 | 2.222 | 2.222 | 0.1384 |
| 1 | 1.0 | 1.000 | 1.000 | 1.111 | 1.111 | -inf | 0.0000 |
| 2 | 0.5 | 0.500 | 0.500 | 0.556 | -inf | -inf | 0.0000 |
| 3 | 4.0 | 4.000 | 3.200 | 3.556 | 3.556 | 3.556 | 0.5250 |
| 4 | 0.0 | 0.000 | 0.000 | 0.000 | -inf | -inf | 0.0000 |
| 5 | 3.5 | 3.500 | 2.800 | 3.111 | 3.111 | 3.111 | 0.3366 |
| 6 | -1.0 | -1.000 | -1.000 | -1.111 | -inf | -inf | 0.0000 |
| 7 | 0.8 | 0.800 | 0.800 | 0.889 | 0.889 | -inf | 0.0000 |
| 8 | 1.5 | -inf | -inf | -inf | -inf | -inf | 0.0000 |
| 9 | 0.3 | -inf | -inf | -inf | -inf | -inf | 0.0000 |
| 10 | 5.0 | -inf | -inf | -inf | -inf | -inf | 0.0000 |
| 11 | 2.5 | -inf | -inf | -inf | -inf | -inf | 0.0000 |

With u = 0.40 the draw is id 3, with u = 0.97 it is id 5.

End-of-speech allowed:

| id | logit | after the mask | after the penalty | after temperature | after top-k | after top-p | probability |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.0 | 2.000 | 2.000 | 2.222 | 2.222 | -inf | 0.0000 |
| 1 | 1.0 | 1.000 | 1.000 | 1.111 | 1.111 | -inf | 0.0000 |
| 2 | 0.5 | 0.500 | 0.500 | 0.556 | -inf | -inf | 0.0000 |
| 3 | 4.0 | 4.000 | 3.200 | 3.556 | 3.556 | 3.556 | 0.1107 |
| 4 | 0.0 | 0.000 | 0.000 | 0.000 | -inf | -inf | 0.0000 |
| 5 | 3.5 | 3.500 | 2.800 | 3.111 | 3.111 | 3.111 | 0.0710 |
| 6 | -1.0 | -1.000 | -1.000 | -1.111 | -inf | -inf | 0.0000 |
| 7 | 0.8 | 0.800 | 0.800 | 0.889 | -inf | -inf | 0.0000 |
| 8 | 1.5 | -inf | -inf | -inf | -inf | -inf | 0.0000 |
| 9 | 0.3 | -inf | -inf | -inf | -inf | -inf | 0.0000 |
| 10 | 5.0 | 5.000 | 5.000 | 5.556 | 5.556 | 5.556 | 0.8183 |
| 11 | 2.5 | -inf | -inf | -inf | -inf | -inf | 0.0000 |

With u = 0.40 the draw is id 10, with u = 0.97 it is id 10.
<!-- /worked -->

The first table is the interesting one. Id 10 has the largest raw logit by a margin (5.0 against 4.0), and it is the end-of-speech id; while the minimum length has not been reached the mask removes it, and the two draws land on id 3 and id 5, the two ids the penalty had lowered. In the second table it is allowed, and the draw is id 10 for both uniform numbers. Whether a clip can end at a given frame is thus a number in this chain, and the length of the speech is the first time the model's end-of-speech draw is allowed and succeeds.

In pseudocode the chain is short. `argsort` and `softmax` are those of the pseudocode page, with $-\infty$ entries receiving probability 0.

```text
function chain_draw(logits: array<real>[V], history: list<int>, eos: int, allow_eos: bool,
                    window: int, theta: real, tau: real, k: int, p: real, u: real) -> int
    require 0 <= u and u < 1, "u is a uniform draw"
    require 0 < p and p < 1, "an active nucleus cut"    # the crate reads 0 and 1 as 'off'
    z = copy(logits)
    eos_logit = z[eos]
    for i in V - window .. V
        z[i] = -infinity                        # no special can be drawn
    if allow_eos
        z[eos] = eos_logit
    seen = zeros(bool, [V])
    for j in history
        if not seen[j] and is_finite(z[j])
            z[j] = z[j] / theta if z[j] > 0 else z[j] * theta     # once for each distinct id
        seen[j] = true
    z = z / tau
    order = argsort(z, descending)
    threshold = z[order[k - 1]]
    for i in 0 .. V
        if z[i] < threshold
            z[i] = -infinity                    # top-k: a tie at the threshold stays
    q = softmax(z)
    order = argsort(q, descending)
    cum = 0.0
    keep = V
    for r in 0 .. V
        cum = cum + q[order[r]]
        if q[order[r]] > 0 and cum >= p
            keep = r + 1
            break
    for r in keep .. V
        z[order[r]] = -infinity                 # top-p: the shortest prefix whose mass reaches p
    q = softmax(z)
    acc = 0.0
    last = argmax(q)
    for i in 0 .. V
        if q[i] > 0
            acc = acc + q[i]
            last = i
            if acc >= u * sum(q)
                return i
    return last                                 # only a surviving id can be returned
```

Two details of the last loop are not decoration. An id whose probability is 0 is skipped, so a masked id can never be returned: with $u = 0$ the first id of the table would otherwise qualify, and with a rounding that made $u \cdot \mathrm{sum}$ slightly larger than the running sum the walk would fall off the end onto the last id of the vocabulary, and both of those are masked specials far more often than they are real candidates. Commit `375758271` found exactly this in the crate's first sampler, and the loop above is the fixed one. The penalty is also applied once per distinct id, which is the other thing that commit corrected, and the table below shows why it is not a detail.

<!-- worked: speech_synthesis.penalty_compounding -->
Repetition penalty 1.05 on a positive logit of 5.0 for a token seen n times: once per distinct token (divide by 1.05 however many times it was said) against once per occurrence (divide by 1.05 n times, that is by 1.05^n).

| n occurrences | 1.05^n | logit, once per token | logit, once per occurrence |
|---|---:|---:|---:|
| 1 | 1.0500 | 4.7619 | 4.7619 |
| 2 | 1.1025 | 4.7619 | 4.5351 |
| 5 | 1.2763 | 4.7619 | 3.9176 |
| 10 | 1.6289 | 4.7619 | 3.0696 |
| 50 | 11.4674 | 4.7619 | 0.4360 |
| 100 | 131.5013 | 4.7619 | 0.0380 |

The handicap of the once-per-token rule on a positive logit z is z (1 - 1/1.05), in nats:

| logit z | logit lost |
|---|---:|
| 1 | 0.048 |
| 2 | 0.095 |
| 5 | 0.238 |
| 10 | 0.476 |
| 20 | 0.952 |
<!-- /worked -->

Applied once for each occurrence, a penalty of 1.05 divides a logit of 5.0 by $1.05^{50} = 11.47$ after fifty repeats and by $131.5$ after a hundred, which is a different processor from the one its value was chosen for; applied once for each distinct id, it divides by 1.05 however often the id was said. The second table points at why the same small factor is nevertheless strong where it is needed. A multiplicative penalty $\theta_{\mathrm{rep}}$ takes away $z(1 - 1/\theta_{\mathrm{rep}})$ from a positive logit $z$, so the largest handicaps go to the tokens the model is surest of: 0.238 nats at a logit of 5, 0.952 at 20. Those are exactly the tokens a repetition loop is made of.

## The sentence that never ended

That brings back the clip of 200 silent frames. The story, as the commits tell it and as the code lets it be checked, has four steps.

The first is the gate. `crates/qwen3tts/tests/it/asr_roundtrip.rs` synthesises the nine-word sentence, resamples the waveform to 16 kHz (through the anti-aliased resampler of Chapter 1, since commit `49bca711f`), transcribes it with the recogniser of Chapter 4, and asserts that the word error rate against the input is below 0.5 and that the clip's root-mean-square level is above 0.01. The bound is loose on purpose: two independently trained models each carry their own error, so the test is a net for garbled, silent or wrong-language output, not a benchmark. It needed no reference implementation at all, only two models already in the tree. On its first real run it produced a silent clip, and commit `fee25c3f0` recorded it as an open, reproducible defect without explaining it.

The second is the diagnosis, in commit `375758271`. Dumping the distribution that the chain actually drew from, frame by frame, showed 50 live candidates at every step (the top-k) and a top-1 probability that rose each time a token repeated: the commit reports 0.92, then 0.97, then 0.9998. At 0.9998 no draw at temperature 0.9 leaves the loop, so the clip ran to the cap, 13 distinct first codes in 200 frames, decoded to near silence. The cause was a default: the checkpoint's own `generation_config.json` ships a repetition penalty of 1.05 and the crate's options had it at 1.0, switched off. With 1.05, the commit reports, the same request ends on the end-of-speech id after 38 frames, and the test `sampled_decode.rs` now asserts a root-mean-square level above 0.01 for this request, which had measured $1.1 \times 10^{-5}$ before the fix.

The third is the architecture lesson in commit `7d31ef313`, and it is the one that generalises. One number had been fixed by changing the number; the defect was that a set of values transcribed into Rust from a file the checkpoint already shipped had become a second source of truth. The crate now resolves every sampling setting in the order the reference uses, `crates/qwen3tts/src/genconfig.rs`: an explicit request from the caller, then the checkpoint's `generation_config.json`, then the reference's hard default. The request type holds each setting as an optional value, because "the caller said nothing" and "the caller asked for 1.0" are different requests that one plain number cannot tell apart, and a user who disables the penalty means the second. Commit `e02fa3119` closed the same gap for the other fifteen codes: the checkpoint says the code predictor samples, the crate drew an argmax, and the fix routes the residual codes through the same draw function as the first code, so that there is one implementation of the arithmetic and no pair of implementations to keep in parity.

The fourth is a detector that watches without acting. `DegenerationWatch` in `sampling.rs` reports once, on standard error, when a first code repeats more than 20 times in a row with a post-filter probability above 0.99, thresholds read off the measured collapse (in a healthy clip of the same sentence, a first code repeats for two or three frames at most). It never changes what is sampled; the sanctioned countermeasure is the penalty. The reference implementation reports that a small fraction of runs can still fail to end with correct settings, and a collapsed clip that explains itself in the log is a different thing to debug from one that does not.

## Borrowing a voice

Everything so far has made a voice that nobody chose. **Voice cloning** is making a synthesiser speak in the voice of a person it has only a short recording of, with no training on that person. The Talker offers two ways in, and the difference between them is a difference in how much of the recording the model is allowed to see.

The first is the speaker vector alone. The reference wav goes through the speaker encoder of Chapter 5, and the vector sits in one row of the prompt: the model knows a timbre and nothing about what the person said. The prompt is ten rows long in the form above and independent of the length of the recording. The second is **in-context cloning**: the reference wav is also run through the codec's encoder, which gives its codes, and the prompt carries the reference's transcript and those codes as if the reference were the first part of the utterance, so that the Talker continues speech it has been shown. The reference's frames each contribute a row, the text side of the stream is lined up with the reference's frames, and the prompt grows by one row per reference frame, 135 rows for a ten-second reference against 10 for the vector form. The model can condition on pitch, rate, room and prosody as well as timbre, at the price of attending to every reference row at every step it writes.

An engine that serves one voice many times need not pay the price of the reference more than once. `ResidentEngine` in `crates/qwen3tts/src/engine.rs` keeps the reference's vector and codes keyed by the pair of recording and transcript, so only the first request pays the encoder. What the reference rows cost is counted below, with the work that comes before the first sound.

One detail of how the reference enters is a callback of Chapter 1. The codec expects 24 kHz, and the reference arrives at whatever rate it was recorded, so it is resampled. Until commit `49bca711f` that was done by `audio::resample_linear` (`crates/qwen3tts/src/pipeline.rs` for the one-shot paths, `engine.rs` for the resident one, and the speaker encoder's own front end, `crates/ecapatdnn/src/model.rs`), the interpolating resampler that Chapter 1 showed has no anti-aliasing filter: for a 48 kHz recording the ratio is exactly one half, so it kept every second sample and folded whatever the recording held between 12 and 24 kHz down into the band below. The CosyVoice pipeline, which came after the Kaiser design of Chapter 1 was added, already resampled its reference with `audio::resample::rational`, and commit `49bca711f` moved the others to it. Speech holds little energy above 12 kHz; the lasting change is that one job now has one resampler.

## Tokens made to carry what was said

The codec of Chapter 3 was trained to reconstruct audio, and so its codes carry everything a decoder needs to rebuild the waveform: what is said, who says it, in what room. The second route wants a different token, one that carries what is said and little else, and the model that makes it is a **speech tokenizer**: a network trained so that its discrete outputs are informative about the text (and so are the same for two speakers saying the same words) rather than about the waveform. A speech tokenizer is trained with a supervised target, such as the transcript, and not by reconstruction; the crate behind this route, `crates/s3tokenizer`, has no decoder at all, because the released model has none: tokens go one way, into a speech-token language model.

How a continuous vector becomes one of 6561 symbols without a codebook is the interesting part, because the tokenizer does not use the residual vector quantization of Chapter 3. It uses **finite scalar quantization**: project the encoder's 1280-wide state to 8 numbers, squash each with $\tanh$, and round each independently to one of three levels. The digit of coordinate $i$ is

$$
\mathsf{d}_i = \operatorname{round}\bigl(0.999\,\tanh z_i\bigr) + 1 \in \{0, 1, 2\},
\qquad
y = \sum_{i=0}^{7} \mathsf{d}_i\, 3^{i},
$$

with the rounding to the nearest integer and ties to the even one, so that the token $y$ is a base-3 numeral with 8 digits and the vocabulary is $3^8 = 6561$. There is no table to train or search, no codebook to collapse, and the "nearest code" step is eight roundings.

<!-- worked: speech_synthesis.fsq_toy -->
One coordinate: h = round(tanh(z) x 0.999) + 1, the rounding to the nearest integer with ties to even (the library tanh in double precision). The digit changes where 0.999 tanh z crosses 1/2, at |z| = atanh(0.5/0.999) = 0.549974.

| z | tanh z | 0.999 tanh z | rounded | digit h |
|---|---:|---:|---:|---:|
| -2.0 | -0.96403 | -0.96306 | -1 | 0 |
| -0.6 | -0.53705 | -0.53651 | -1 | 0 |
| -0.54 | -0.49299 | -0.49249 | 0 | 1 |
| 0.0 | 0.00000 | 0.00000 | 0 | 1 |
| 0.54 | 0.49299 | 0.49249 | 0 | 1 |
| 0.56 | 0.50798 | 0.50747 | 1 | 2 |
| 3.0 | 0.99505 | 0.99406 | 1 | 2 |

Eight coordinates make one token, the digit of coordinate i weighted by 3^i (so the first coordinate is the least significant):

| z (8 coordinates) | digits h | sum | token |
|---|---|---|---|
| 0.8, -0.3, 1.9, -1.2, 0.1, 0.6, -0.7, 2.4 | 2 1 2 0 1 2 0 2 | 2x1 + 1x3 + 2x9 + 0x27 + 1x81 + 2x243 + 0x729 + 2x2187 | 4964 |
| -0.2, -0.2, 0.3, 0.45, -3, 0.7, 1, -0.56 | 1 1 1 1 0 2 2 0 | 1x1 + 1x3 + 1x9 + 1x27 + 0x81 + 2x243 + 2x729 + 0x2187 | 1984 |

The token space has 3^8 = 6,561 entries, log2 of that is 12.6797 bits, no table is stored and a nearest-entry search is replaced by eight roundings.
<!-- /worked -->

The first table shows the thresholds. A digit changes where $0.999\tanh|z|$ crosses one half, at $|z| = \operatorname{atanh}(0.5/0.999) = 0.549974$; a coordinate smaller than that in magnitude is the middle level, 1. The integer value of a token carries no similarity (two tokens that differ only in the last coordinate differ by 2187 as integers and by one level in one digit), which is why the speech-token model reads it through an embedding table and never as a number. In pseudocode (`round_even` is defined in the pseudocode page's note on this chapter's routines):

```text
function fsq_token(z: array<real>[8]) -> int
    y = 0
    for i in 0 .. 8
        y = y + (round_even(tanh(z[i]) * 0.999) + 1) * 3^i
    return y
```

The rest of the tokenizer is a stack of the familiar parts. A 128-bin log-mel of the 16 kHz clip (window 400, hop 160, 100 frames a second) goes through two convolutions of kernel 3 and stride 2, which leave 25 positions a second, and six transformer blocks of width 1280 with 20 heads, rotary positions and, beside each attention, a depthwise filter of 31 taps along time added to the value path; the head projects to 8 numbers. It is not causal, since the attention looks in both directions over the clip, so it needs the whole reference before it writes a token, and that costs nothing here because it is only ever run on the reference recording.

<!-- worked: speech_synthesis.tokenizer_counts -->
Counted from the tensors of the released tokenizer (102 of them): weights and biases and the layer-norm gains, in the layer shapes above; one second of audio is 100 mel frames, 50 after the first convolution and 25 after the second. Both convolutions have k 3, stride 2 and a bias. A block has width 1280 and 20 heads of 64: attention q, v and output matrices with a bias, the k matrix without (each 1280 x 1280), a depthwise filter of 1280 x 31 along time without bias (added to the values, counted as 1280 x 31 multiply-adds a position), two LayerNorms with gain and shift, and an MLP 1280 -> 5120 -> 1280 with both biases; the projection to 8 digits has a bias.

| part | parameters | positions per second | multiply-adds per second of audio (millions) |
|---|---:|---:|---:|
| two convolutions, 128 -> 1280 -> 1280 (k 3, stride 2) | 5,409,280 | 50 and 25 | 147.5 |
| 6 blocks (attention, depthwise filter k 31, MLP 1280 -> 5120 -> 1280) | 118,295,040 | 25 | 2,955.1 |
| projection to 8 digits | 10,248 | 25 | 0.3 |

In all 123,714,568 parameters and 3,102.8 million multiply-adds per second of reference audio before the attention scores. The scores cost 2 x heads x head width per query and key pair of each block (the encoder attends in both directions over the whole clip):

| seconds | tokens | attention multiply-adds of the clip (millions) |
|---:|---:|---:|
| 3 | 75 | 86.4 |
| 10 | 250 | 960.0 |
| 30 | 750 | 8,640.0 |
<!-- /worked -->

The tokenizer has 123,714,568 parameters and takes 3.1 billion multiply-adds for each second of reference audio, and at ten seconds the attention scores add 960 million. It is also the place where two front ends have to agree. The reference clip is turned into speech tokens by the 16 kHz mel and into a prompt mel by a 24 kHz one (window 1920, hop 480, 50 frames a second), and the prompt must use the same stretch of time in both, so it keeps the smaller of the number of tokens and half the prompt mel's frames (rounded down), and twice as many mel frames as tokens.

<!-- worked: speech_synthesis.tokenizer_geometry -->
A reference clip of the given length is resampled twice. At 16 kHz a 400-sample window at hop 160 (reflect padding of 200, the last frame dropped) gives floor(n / 160) frames, and two convolutions of kernel 3, stride 2 and padding 1 (output (l - 1) // 2 + 1) give the tokens. At 24 kHz a 1920-sample window at hop 480 with reflect padding of 720 on each side gives floor(n / 480) frames. The prompt keeps min(mel frames // 2, tokens) tokens and twice as many mel frames.

| seconds | samples at 16 kHz | mel frames | tokens | samples at 24 kHz | mel frames | mel // 2 | tokens kept | mel frames kept | tokens dropped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16,000 | 100 | 25 | 24,000 | 50 | 25 | 25 | 50 | 0 |
| 2.5 | 40,000 | 250 | 63 | 60,000 | 125 | 62 | 62 | 124 | 1 |
| 3.3 | 52,800 | 330 | 83 | 79,200 | 165 | 82 | 82 | 164 | 1 |
| 7 | 112,000 | 700 | 175 | 168,000 | 350 | 175 | 175 | 350 | 0 |
| 10 | 160,000 | 1000 | 250 | 240,000 | 500 | 250 | 250 | 500 | 0 |
<!-- /worked -->

For a clip of 2.5 seconds the tokenizer writes 63 tokens and the mel only supports 62, so one is dropped; for 1, 7 and 10 seconds the two agree. The rule is in the pipeline in `crates/cosyvoice/src/pipeline.rs`, and a test in the crate pins the tokenizer's own output: the gate `crates/s3tokenizer/tests/s3tokenizer_parity.rs` requires the integer tokens to equal a reference run exactly, since an integer sequence cannot be "close", and the commit that added the model (`6be2a2d4e`) reports that all 87 tokens of the real reference clip matched. Exactness also explains why the whole forward pass is plain host arithmetic with no device dispatch: a second source of floating-point reassociation could flip a rounding in the last coordinate and change a token, and an exact-equality gate would then fail for a reason no one could use.

Against the codec's 2200 bits a second, these tokens carry 317 (the table of interface rates), about seven times fewer. That is the point of the design, and it is also the reason that the next model's output is not a waveform: the tokens do not say who is speaking or how.

## A language model over 6561 symbols

The model that writes speech tokens is a stock Qwen2.5-0.5B decoder, hosted by `crates/qwen3` as the Talker's backbone is, with three tables of its own added: an embedding table for the 6561 speech tokens and a few specials (6564 rows of 896), an output head over the same rows, and for the second generation a two-row table for the start and task markers. Its prompt is simpler than the Talker's.

$$
\text{sos} \;\Vert\; \text{text}_{\mathrm{ref}} \;\Vert\; \text{text}_{\mathrm{target}} \;\Vert\; \text{task} \;\Vert\; y_1, \dots, y_m
$$

where the text tokens are the reference's transcript followed by the target text, the $y$ are the reference clip's speech tokens, and the model then writes speech tokens autoregressively until it draws a stop id. Two facts about this prompt are consequences of the route. The reference's recording enters the language model as speech tokens, so that the model is shown how the sentence it transcribes was said, in the only terms the model reads; and the 192-number speaker vector does not enter it at all (the crate's `crates/cosyvoice/src/llm.rs` documents that the reference implementation accepts the vector and never reads it, and `prefill` has no argument for it). The voice reaches the pipeline later, at the flow decoder.

<!-- worked: speech_synthesis.speech_lm_counts -->
Counted from the tensors of the released speech-token model (295 in the file; the 151,936-wide output matrix of the text backbone is a copy of the embedding table and is not read). A layer has width 896, 14 query heads and 2 key-value heads of 64: q (896 x 896), k and v (896 x 128) each with a bias, o without, gate, up and down of 896 x 4864 without bias, two RMSNorm weights of 896, no query or key norm; the 'start and task rows' are two rows of 896.

| part | parameters |
|---|---:|
| 24 layers (grouped-query attention 14 query heads, 2 of 64, feed-forward 4864) | 357,897,216 |
| text embedding table, tied (151,936 x 896) and final norm | 136,135,552 |
| speech embedding (6564 x 896), speech head (896 -> 6564 with bias), start and task rows | 11,771,044 |
| total | 505,803,812 |

One step of the model is 363.7 million multiply-adds (24 layers and the 6564-way head), so a second of speech at 25 tokens costs 9.09 billion, attention scores aside.

A prompt: reference of 3.3 seconds (82 speech tokens after the truncation of the last table), 10 reference-text tokens and 8 target-text tokens. The rows it feeds before the first draw are 1 + 18 + 1 + 82 = 102; the model may draw at most 160 tokens (6.4 seconds) and is held from stopping before 16; the cache is sized for 102 + 160 + 8 = 270 positions (the 8 is a margin that the pipeline adds to the cache; the rows actually written never exceed 102 + 160).

The budget as a function of the number of target-text tokens n (minimum round(2 n), maximum round(20 n), ties upward):

| target text tokens n | minimum tokens | seconds | maximum tokens | seconds |
|---:|---:|---:|---:|---:|
| 1 | 2 | 0.08 | 20 | 0.80 |
| 4 | 8 | 0.32 | 80 | 3.20 |
| 8 | 16 | 0.64 | 160 | 6.40 |
| 20 | 40 | 1.60 | 400 | 16.00 |
| 60 | 120 | 4.80 | 1200 | 48.00 |
<!-- /worked -->

The model has 505,803,812 parameters and each step is 363.7 million multiply-adds, so a second of speech at 25 tokens costs 9.09 billion, less than half of the Talker and its code predictor (21.67 billion for 12.5 frames), because the model writes one symbol and not sixteen. Its attention cache grows more slowly too:

<!-- worked: speech_synthesis.kv_growth -->
Bytes of attention keys and values kept per position, 32-bit floats (2 tensors x layers x key-value heads x head width x 4 bytes), per second of speech, and for 10 seconds.

| model | positions per second | bytes per position | bytes per second of speech | megabytes for 10 seconds |
|---|---:|---:|---:|---:|
| 16-codebook synthesiser's Talker | 12.5 | 229,376 | 2,867,200 | 28.67 |
| single-token synthesiser's language model | 25 | 24,576 | 614,400 | 6.14 |

A cache of 1024 Talker positions is 234,881,024 bytes and holds 81.92 seconds of speech, the prompt included.
<!-- /worked -->

The speech-token model has 2 key-value heads of width 64 and the Talker has 8 of width 128, so a position of the second costs 24,576 bytes of cache against 229,376, and even at twice the positions a second the speech-token model's cache grows at 614,400 bytes a second of speech and the Talker's at 2,867,200.

The model is told how long it may speak by the text, not by itself. The budget is a ratio of speech tokens to target-text tokens: the model may not stop before twice the number of target-text tokens, and may not run past twenty times it (`token_budget` in `pipeline.rs`). In seconds that is 80 milliseconds and 0.8 seconds per text token, so a 60-token text is held between 4.8 and 48 seconds. The cap is derived from the text, where the Talker's is a frame count the caller sets. The minimum is enforced by removing only the end-of-speech id from the distribution.

The sampler is not the Talker's. `ras_sampling` (`crates/cosyvoice/src/sampling.rs`) is repetition-aware: it draws from the nucleus, and if the drawn token appears at least $10 \times 0.1 = 1$ time among the last 10 tokens it discards that draw, removes the token, and draws again from the whole vocabulary. The first stage is the usual nucleus (top-p 0.8 over at most 25 candidates); the second is the repetition guard.

<!-- worked: speech_synthesis.ras_demo -->
Eight speech tokens with the logits below (a softmax of the log-softmax is the softmax, so the logits are used directly), the last 10 generated tokens [4, 1, 6, 1, 2, 0, 5, 1, 7, 3], nucleus 0.8 over at most 25 candidates, and a repetition window of 10 with threshold 10 x 0.1 = 1 occurrence. The nucleus takes ids in order of probability while the sum so far is below 0.8; a draw is one uniform u, scanned over the kept probabilities in that order (stop at the first whose running sum reaches u times the total).

| id | logit | probability | in the nucleus | times in the last 10 |
|---|---:|---:|---|---|
| 0 | 3.0 | 0.5644 | yes | 1 |
| 1 | 2.2 | 0.2536 | yes | 3 |
| 2 | 1.0 | 0.0764 | no | 1 |
| 3 | 0.5 | 0.0463 | no | 1 |
| 4 | 0.0 | 0.0281 | no | 1 |
| 5 | -0.5 | 0.0170 | no | 1 |
| 6 | -1.0 | 0.0103 | no | 1 |
| 7 | -2.0 | 0.0038 | no | 1 |

The nucleus is [0, 1] with probabilities [0.5644, 0.2536].

| u1 | first draw | its count in the window | redrawn | u2 | second draw (all ids, first one removed) | token emitted |
|---:|---:|---:|---|---:|---:|---:|
| 0.05 | 0 | 1 | yes | 0.60 | 2 | 2 |
| 0.70 | 1 | 3 | yes | 0.60 | 0 | 0 |
| 0.90 | 1 | 3 | yes | 0.20 | 0 | 0 |
<!-- /worked -->

In all three rows the first draw lands on a token that is already in the window, so the second draw is taken, from everything except that token: even at 0.05, where the nucleus draw is token 0 with probability 0.56, the sampler moves on. With a window of 10 and a threshold of one occurrence, a nucleus draw that repeats any of the last ten tokens is always thrown away (the redraw is not checked again), which is the speech-token model's answer to the repetition loop that silenced the Talker, built into the sampler.

## Filling in a spectrogram from noise

The tokens say what is said at 25 a second, and the waveform needs 24,000 numbers. The step between is the flow decoder, and the reason it is a flow decoder is in the tokenizer: there is no decoder, so nothing can undo the tokens, and the information the tokens left out (the pitch contour, the exact timbre, the room) has to be generated by something that is told the tokens and the voice and is allowed to choose. A conditional distribution to sample is exactly what Volume VIII's generators are for. The flow decoder of this crate generates the mel spectrogram of Chapter 2, 80 bins at 50 frames a second, by the flow-matching ordinary differential equation (ODE) of Volume VIII, Chapter 6, starting from noise.

Its conditions are four tensors of the same length as the mel, and the network reads them side by side, as extra channels:

* $x$, the current noisy mel, 80 channels;
* $\mu$, the "mean" the tokens imply: each token's embedding goes through an encoder (a stack of conformer layers in the second generation, a single look-ahead layer in the third) and is repeated twice, so that 25 tokens a second become 50 frames a second;
* the speaker vector, normalized to unit length and mapped by a learned layer from 192 to 80 numbers, repeated at every frame;
* the condition, the reference's mel in the frames that stand for the reference and zeros elsewhere.

That makes 320 channels per frame. A toy with one number in place of each 80-vector shows the assembly.

<!-- worked: speech_synthesis.cfm_condition -->
The conditions of the flow decoder for 3 toy tokens (5, 2, 7), the first of them the reference's: the level of token id i is [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9][i], and each token becomes 2 frames. The 3-number speaker vector (3.0, 4.0, 0.0) is scaled to unit length, [0.6, 0.8, 0.0], and mapped by the weights (0.5, 0.25, -0.5) and the bias 0.1 to the one number 0.60, which is repeated at every frame. The two prompt frames carry their mel values [0.55, 0.45] in the condition channel and the rest carries 0. The starting values are 6 draws of normal() from `speech_synthesis.condition_noise` (seed 0x385843144e71a63e), in frame order. At t = 0 the network reads one row of four channels per frame:

| frame | role | x (noise) | mu (from the tokens) | speaker | condition |
|---|---|---:|---:|---:|---:|
| 0 | prompt | -2.6052 | 0.50 | 0.60 | 0.55 |
| 1 | prompt | -1.0534 | 0.50 | 0.60 | 0.45 |
| 2 | generated | 0.5870 | -0.10 | 0.60 | 0.00 |
| 3 | generated | -0.7783 | -0.10 | 0.60 | 0.00 |
| 4 | generated | -1.5933 | 0.90 | 0.60 | 0.00 |
| 5 | generated | 0.3739 | 0.90 | 0.60 | 0.00 |

Only the last 4 frames are kept as the mel the vocoder will read.
<!-- /worked -->

The reference's tokens are placed in front of the generated ones and the decoder produces a mel for the whole span, reference included; the reference's frames are then dropped, and the rest is the mel to vocode. Putting the reference in the same sequence is how the decoder is made to continue in the same voice, and the condition channel is how it is told what the reference sounded like exactly. The starting noise is the first $T$ columns of one fixed buffer of 80 by 15,000 values, drawn once from a seed of zero (the crate reproduces the framework's generator bit for bit so as not to ship a 4.8-megabyte file), so the flow decoder is a deterministic function of the tokens and the voice, and 15,000 frames at 50 a second is 300 seconds, the longest span it accepts. All the randomness of a CosyVoice utterance is in the language model's sampler and the vocoder's noise.

Time runs the other way from Volume VIII here. The decoder's reference code puts noise at $t = 0$ and data at $t = 1$, so $t = 1 - \sigma$ in the notation of Volume VIII, Chapter 6, and its velocity $u$ is the derivative with respect to $t$, the negative of the earlier one; the guided blend is unchanged by the sign. The grid is not uniform:

$$
t_i = 1 - \cos\!\Bigl(\frac{\pi}{2}\cdot\frac{i}{n}\Bigr), \qquad i = 0, \dots, n,
$$

with $n = 10$ steps, and each Euler step moves $x \leftarrow x + (t_{i+1} - t_i)\,u_g(x, t_i)$.

<!-- worked: speech_synthesis.cfm_grid -->
The grid t_i = 1 - cos(pi (i / 10) / 2), from t = 0 (noise) to t = 1 (data), against the uniform step 0.1.

| i | i / 10 | t_i | step t_(i+1) - t_i | uniform step |
|---|---:|---:|---:|---:|
| 0 | 0.0 | 0.000000 | 0.012312 | 0.100000 |
| 1 | 0.1 | 0.012312 | 0.036632 | 0.100000 |
| 2 | 0.2 | 0.048943 | 0.060050 | 0.100000 |
| 3 | 0.3 | 0.108993 | 0.081990 | 0.100000 |
| 4 | 0.4 | 0.190983 | 0.101910 | 0.100000 |
| 5 | 0.5 | 0.292893 | 0.119322 | 0.100000 |
| 6 | 0.6 | 0.412215 | 0.133795 | 0.100000 |
| 7 | 0.7 | 0.546010 | 0.144974 | 0.100000 |
| 8 | 0.8 | 0.690983 | 0.152583 | 0.100000 |
| 9 | 0.9 | 0.843566 | 0.156434 | 0.100000 |
| 10 | 1.0 | 1.000000 |  |  |

The last step is 12.71 times the first: 0.012312 against 0.156434.
<!-- /worked -->

The first step is 0.0123 and the last is 0.1564, 12.7 times as long: the grid takes small steps at the noise end and large ones at the data end. Whether that is the right place for the steps is a statement about a trained network, and the toy below, whose field changes most near the data end, will not agree with it.

Each step needs the guided velocity, and the guidance is the one of Volume VIII, Chapter 7, with the unconditioned branch obtained by zeroing every condition ($\mu$, the speaker vector and the condition tensor), and the code's blend $(1 + 0.7)\,u_c - 0.7\,u_{\mathrm{un}}$, which is $u_g = u_{\mathrm{un}} + g(u_c - u_{\mathrm{un}})$ with $g = 1.7$. Each step is therefore two evaluations of the network.

```text
function cfg_euler(u_cond: function(array<real>[C, T], real) -> array<real>[C, T],
                   u_uncond: function(array<real>[C, T], real) -> array<real>[C, T],
                   x0: array<real>[C, T], grid: array<real>[n + 1], g: real) -> array<real>[C, T]
    require grid[0] == 0 and grid[n] == 1, "the grid runs from noise to data"
    x = copy(x0)
    for i in 0 .. n
        uc = u_cond(x, grid[i])
        un = u_uncond(x, grid[i])
        x = x + (grid[i + 1] - grid[i]) * (un + g * (uc - un))
    return x
```

To see what a decoder like this does with the guidance, a toy has to be exact. Take one mel value. Given the tokens and the speaker its distribution is $\mathcal{N}(1.5, 0.3^2)$, and a model told nothing knows only $\mathcal{N}(0, 1)$. For a Gaussian target of mean $m$ and standard deviation $\varsigma$ on the straight path $x_t = (1 - t)z + t y$, the exact velocity field has the closed form below, so both fields can be written down and no network is needed:

$$
u(x, t) = m + \frac{t\varsigma^2 - (1 - t)}{(1 - t)^2 + t^2\varsigma^2}\,(x - t m).
$$

<!-- worked: speech_synthesis.cfm_toy -->
One mel value, one frame. The target given the tokens and the speaker is N(1.5, 0.3^2) and the model without conditions knows only N(0, 1). Both fields are the exact marginal velocity of the straight path x_t = (1 - t) z + t y, u(x, t) = m + (t sd^2 - (1 - t)) / ((1 - t)^2 + t^2 sd^2) (x - t m) for a target of mean m and standard deviation sd, which makes the example free of any trained network. Guidance is u_un + 1.7 (u_c - u_un) (rate 0.7 in the form (1 + 0.7) u_c - 0.7 u_un). Starting values: 6 draws of normal() from `speech_synthesis.flow_start` (seed 0xdb48f279a4e7f2f8), in order; each row is one start, run through 10 Euler steps on the cosine grid.

The first start, step by step (conditional only, then guided):

| step | t | dt | x, conditional | u_c | x, guided | u_un at the guided x | u guided |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0123 | 0.9651 | 0.5349 | 0.9651 | -0.9651 | 1.5849 |
| 1 | 0.0123 | 0.0366 | 0.9717 | 0.5360 | 0.9846 | -0.9843 | 1.5780 |
| 2 | 0.0489 | 0.0600 | 0.9913 | 0.5396 | 1.0424 | -1.0369 | 1.5522 |
| 3 | 0.1090 | 0.0820 | 1.0237 | 0.5465 | 1.1356 | -1.1021 | 1.4896 |
| 4 | 0.1910 | 0.1019 | 1.0685 | 0.5586 | 1.2577 | -1.1250 | 1.3499 |
| 5 | 0.2929 | 0.1193 | 1.1254 | 0.5801 | 1.3953 | -0.9866 | 1.0617 |
| 6 | 0.4122 | 0.1338 | 1.1947 | 0.6203 | 1.5220 | -0.5185 | 0.5681 |
| 7 | 0.5460 | 0.1450 | 1.2777 | 0.7029 | 1.5980 | 0.2916 | 0.0443 |
| 8 | 0.6910 | 0.1526 | 1.3796 | 0.8884 | 1.6044 | 1.0696 | 0.0801 |
| 9 | 0.8436 | 0.1564 | 1.5151 | 1.2728 | 1.6166 | 1.5091 | 0.9504 |

End points of all six starts:

| start z | exact map m + sd z | 10 steps, cosine grid | 10 steps, uniform grid | guided, 10 steps, cosine grid | guided, 2000 uniform steps |
|---:|---:|---:|---:|---:|---:|
| 0.9651 | 1.7895 | 1.7142 | 1.7382 | 1.7653 | 1.8580 |
| -0.4186 | 1.3744 | 1.4071 | 1.3967 | 1.6789 | 1.6796 |
| 0.6131 | 1.6839 | 1.6361 | 1.6514 | 1.7433 | 1.8127 |
| 0.2958 | 1.5887 | 1.5657 | 1.5730 | 1.7235 | 1.7717 |
| 0.3590 | 1.6077 | 1.5797 | 1.5886 | 1.7274 | 1.7799 |
| 1.0636 | 1.8191 | 1.7361 | 1.7626 | 1.7715 | 1.8707 |

The largest distance of the 10-step conditional run from the exact map is 0.0830 on the cosine grid and 0.0565 on the uniform grid. The field is linear in x, so the guided map from start to end point is a straight line, found here from the starts 0 and 1 with 2000 steps: end = 1.7336 + 0.1289 z, against 1.5 + 0.3 z without guidance. Guided runs cost 20 evaluations of the field, conditional ones 10.
<!-- /worked -->

![Three flow trajectories from t = 0 to t = 1 in one colour each, solid under the conditional field and dashed under the guided field. The solid paths rise steadily to between 1.37 and 1.82; the dashed paths rise faster, overshoot the conditional mean of 1.5 and finish close together between 1.68 and 1.87.](img/flow-trajectories.svg)

Three things are visible in the tables and the figure. Ten Euler steps on the cosine grid are visibly off the exact map $1.5 + 0.3z$ (by up to 0.083, against 0.057 on the uniform grid, so in this toy the cosine grid is the worse choice; a trained network's field is not this field, and the grid was chosen for it). The guided map is a straight line of slope 0.129, where the conditional one has slope 0.3: guidance has pulled the end points together, so that a given set of starts gives less variety in the mel, and moved the image of $z = 0$ from 1.5 to 1.73, past the conditional mean. That is the trade of Volume VIII, Chapter 7 in two numbers: the guided output follows its conditions harder and is less diverse. Last, the guided 10-step run is further from the guided limit (1.7653 against 1.8580 for the first start, a distance of 0.093) than the conditional one is from its own (1.7142 against 1.7895, 0.075), so ten steps are a coarser grid for the guided field.

Counting this decoder's work is the point where the two routes' costs diverge most.

<!-- worked: speech_synthesis.flow_layout -->
The layers behind the counts below: widths, heads, kernels, groups and which tensors carry a bias. A LayerNorm has a weight and a bias of its width unless said otherwise.

Second generation

| part | layout | biases and norms |
|---|---|---|
| token embedding, speaker map, output projection | embedding 6561 x 512; speaker vector 192 -> 80; encoder output 512 -> 80 | the two linear maps have a bias |
| conformer encoder | a linear 512 -> 512 with a LayerNorm; a look-ahead pair of convolutions 512 -> 512 with k 4 (reading 3 later rows) and k 3, added to its input; 6 layers at 25 Hz; every row repeated twice (50 Hz) and a convolution 512 -> 512 with k 5, a linear 512 -> 512 with a LayerNorm; 4 layers at 50 Hz; a final LayerNorm. A layer, with its LayerNorm before the attention and before the feed-forward (no convolution module): 8 heads of 64, q, k, v and o of 512 x 512 with bias, a relative-position matrix 512 x 512 without bias, two vectors u and v of 512, one feed-forward 512 -> 2048 -> 512 with both biases, two LayerNorms (weight and bias) | every convolution and linear map biased except the position matrix |
| estimator | input 320 channels (the noisy mel, the token mel, the speaker vector repeated and the reference mel, 80 each); a time embedding of a sinusoid of width 320 through 320 -> 1024 -> 1024; 14 residual blocks (one going down, 12 in the middle, one going up, whose input is 512 channels, the middle output and the skip from the down stage), each followed by 4 transformer blocks, so 56 in all; a convolution 256 -> 256 (k 3) after the down stage and one after the up stage, a final convolution 256 -> 256 (k 3), a LayerNorm, Mish and a convolution 256 -> 80 (k 1) | all convolutions and linear maps biased |
| residual block (ci -> co) | convolution ci -> co (k 3), LayerNorm, Mish; the time embedding through a projection 1024 -> co added to every position; convolution co -> co (k 3), LayerNorm, Mish; shortcut convolution ci -> co (k 1) added; 320 -> 256 once, 256 -> 256 twelve times, 512 -> 256 once | every convolution, LayerNorm and the time projection with bias |
| transformer block of the estimator | LayerNorm 256; attention with 8 heads of 64, q, k, v 256 -> 512 without bias, o 512 -> 256 with bias; LayerNorm 256; feed-forward 256 -> 1024 -> 256 | LayerNorm weight and bias, the feed-forward and o biased |

Third generation

| part | layout | biases and norms |
|---|---|---|
| token embedding, speaker map, look-ahead layer | embedding 6561 x 80; speaker vector 192 -> 80; convolution 80 -> 1024 (k 4) and convolution 1024 -> 80 (k 3) | biased |
| time embedding | a sinusoid of width 256 through 256 -> 1024 -> 1024 | biased |
| input projection and position convolutions | 320 -> 1024 (the same four 80-channel pieces); then two convolutions of 1024 channels with k 31 in 16 groups (each channel group of 64 reads 64 channels) | all three biased |
| 22 blocks | width 1024, 16 heads of 64 with rotary positions, a LayerNorm without gain or shift before attention and before the feed-forward, whose scale, shift and gate come from the time embedding by one matrix 1024 -> 6144 per block; attention q, k, v, o (1024 x 1024); feed-forward 1024 -> 2048 -> 1024 | all matrices biased; the LayerNorms have no parameters |
| final scale-and-shift and output | a matrix 1024 -> 2048 from the time embedding, then 1024 -> 80 | biased |
<!-- /worked -->

<!-- worked: speech_synthesis.flow_counts -->
Parameters counted from the tensor shapes of the two released flow files (the first matches the total of the released file's 1121 tensors, 112,549,360; the second matches its 330 tensors apart from a 32-entry rotary-frequency buffer that is recomputed, 332,257,088 + 32 = 332,257,120).

| part of the second-generation decoder (convolutional estimator) | parameters |
|---|---:|
| token embedding (6561 x 512), speaker map (192 -> 80), output projection (512 -> 80) | 3,415,712 |
| conformer encoder: 6 layers at 25 Hz, upsampling by 2, 4 layers at 50 Hz | 37,831,168 |
| estimator: 56 transformer blocks of width 256 | 58,863,616 |
| estimator: 14 residual blocks | 10,448,896 |
| estimator: time embedding and final convolutions | 1,989,968 |
| total | 112,549,360 |

| part of the third-generation decoder (transformer estimator) | parameters |
|---|---:|
| token embedding (6561 x 80), speaker map, look-ahead layer (80 -> 1024 -> 80) | 1,114,864 |
| time embedding (256 -> 1024 -> 1024) | 1,312,768 |
| input projection (320 -> 1024) and two grouped position convolutions | 4,393,984 |
| 22 blocks: scale-and-shift matrices (1024 -> 6144) from the time embedding | 138,547,200 |
| 22 blocks: attention (4 matrices of 1024 x 1024) and feed-forward (1024 -> 2048 -> 1024) | 184,707,072 |
| final scale-and-shift and output projection (1024 -> 80) | 2,181,200 |
| total | 332,257,088 |

Work per second of speech: 50 mel frames a second, 10 Euler steps, and two evaluations of the estimator per step (the conditional one and the unconditioned one), so 20 evaluations of every frame. Matrix products are counted once per row they are applied to; attention costs 2 x width per pair of frames and block, where width is heads x head width: 8 x 64 = 512 inside each of the 56 blocks of the second generation's estimator (the block's own width is 256) and 16 x 64 = 1024 in each of the 22 blocks of the third generation's, so a pair costs 2 x 512 or 2 x 1024 multiply-adds, scores and weighted values together, and the attention column is blocks x 2 x width x T; norms and activations are left out. The conformer's relative-position projection is counted over 2 rows per frame. The 'independent of T' column per evaluation is, for the second generation, the time embedding (320 -> 1024 -> 1024) and the 14 time projections (1024 -> 256), and for the third the 22 scale-and-shift matrices (1024 -> 6144), the final one (1024 -> 2048) and the time embedding (256 -> 1024 -> 1024); the third's 'matrix products per frame' are its 22 blocks (4 attention matrices of 1024 x 1024 and the feed-forward 1024 -> 2048 -> 1024), the input projection with the two grouped position convolutions and the output projection.

| decoder | matrix products per frame per evaluation (millions) | attention scores and values per frame at T = 650 (millions) | per evaluation, independent of T (millions) | evaluations | billions per second of speech | attention, billions per second at T = 650 | encoder, once (billions per second) |
|---|---:|---:|---:|---:|---:|---:|---:|
| second generation (convolutional estimator) | 66.1 | 37.3 | 5.0 | 20 | 66.1 | 37.3 | 1.42 |
| third generation (transformer estimator) | 189.0 | 29.3 | 141.8 | 20 | 189.0 | 29.3 | 0 |
<!-- /worked -->

The second-generation decoder has 112,549,360 parameters and the third 332,257,088, and what each costs for a second of speech is 20 evaluations of every one of 50 frames: 66.1 billion multiply-adds for the first, 189.0 billion for the second, before attention. The parameter counts match the totals of the tensor shapes in the two released files (1121 and 330 tensors, the third's file holding a further 32-entry frequency buffer that the crate recomputes), which is how a count like this is checked when the weights are available. The architecture of the second generation never changes resolution: its channel list has one entry, so the "down" and "up" convolutions of the estimator are of stride 1, and every one of its 56 transformer blocks runs at the full mel length. The third replaces the encoder and the U-Net estimator by a 22-layer transformer whose blocks are modulated by the time embedding (scale, shift and gate from the timestep, the adaptive layer normalization of the diffusion transformer of Volume VIII, Chapter 8).

## From mel to samples

The mel spectrogram has 80 numbers for each of 50 frames a second, 4000 a second, and discards the phase. The step that turns it into 24,000 samples a second is the **vocoder**: a network that maps a spectrogram back to a waveform. The vocoder of this crate generates the waveform through a short-time Fourier transform rather than sample by sample. Its trunk is a convolutional upsampler that lengthens the 50 frames by 8, 5 and 3, to 6000 frames a second, and its last layer writes, for each of those frames, a magnitude and a phase for the 9 bins of a 16-point transform; an inverse transform with hop 4 turns them, with one mirrored frame in front, into 24,000 samples. The arithmetic of the lengths is exact.

<!-- worked: speech_synthesis.vocoder_geometry -->
One second of mel (50 frames) through the vocoder's trunk. Each upsampling stage is a transposed convolution whose kernel (16, 11, 7) minus its stride (8, 5, 3) is even, padded by half of that difference, so it multiplies the length by the stride exactly; the last stage mirrors one sample in front, and an inverse short-time transform with a 16-point window and hop 4 turns the 6001 frames of 9 complex bins into (6001 - 1) x 4 samples.

| stage | length | rate |
|---|---:|---|
| mel frames in | 50 | 50 per second |
| after upsampling by 8 | 400 | 400 per second |
| after upsampling by 5 | 2000 | 2000 per second |
| after upsampling by 3 and one mirrored sample | 6001 | 6001 frames of the inverse transform |
| inverse transform, window 16, hop 4 | 24000 | 24000 samples per second |

The harmonic source for a pitch of 200 Hz: 9 sinusoids at multiples of the pitch, each advancing its phase by (frequency / 24000) cycles per sample, and the excitation rule of the source: a frame is voiced when the pitch is above 10 Hz, a voiced sinusoid has amplitude 0.1 and noise of standard deviation 0.003, an unvoiced sample is noise of standard deviation 0.1 / 3 = 0.0333.

| harmonic | frequency (Hz) | cycles per sample |
|---:|---:|---:|
| 1 | 200 | 0.00833 |
| 2 | 400 | 0.01667 |
| 3 | 600 | 0.02500 |
| 4 | 800 | 0.03333 |
| 5 | 1,000 | 0.04167 |
| 6 | 1,200 | 0.05000 |
| 7 | 1,400 | 0.05833 |
| 8 | 1,600 | 0.06667 |
| 9 | 1,800 | 0.07500 |
<!-- /worked -->

The trunk is given a periodic signal to start from. A small network predicts a pitch contour, in hertz, at 50 frames a second from the mel, and a source module turns the contour into an excitation: nine sinusoids at the multiples of the pitch, each with phase accumulated from its frequency, plus noise, where a frame is voiced when the pitch exceeds 10 Hz and unvoiced otherwise. This is the harmonic-plus-noise source of the neural source-filter family, and it is why the trunk, which would otherwise have to find periodicity in noise, has a periodic signal handed to it at every upsampling stage. (The reference's name for the pitch network mentions a recurrent layer, but the network has none: it is five convolutions and a linear read-out.) The magnitude is the exponential of the network's output, capped at 100, and the final samples are clamped to $\pm 0.99$.

<!-- worked: speech_synthesis.vocoder_layout -->
The layers behind the counts below: widths, kernels, strides and which tensors carry a bias or a gain.

| part | layout | biases and gains |
|---|---|---|
| pitch predictor | convolution 80 -> 512 (k 3), four convolutions 512 -> 512 (k 3), each followed by an ELU, then a linear 512 -> 1 whose absolute value is the pitch in Hz | every convolution and the linear map biased |
| first convolution | 80 -> 512, k 7 | biased |
| upsampling stages | before each, a leaky ReLU; transposed convolutions 512 -> 256 (k 16, stride 8), 256 -> 128 (k 11, stride 5), 128 -> 64 (k 7, stride 3), taking 50, 400 and 2000 positions a second in | biased |
| source branch | the nine harmonic sinusoids at 24,000 samples a second are merged into one excitation by the linear map 9 -> 1, and a 16-point transform with hop 4 makes of it 9 complex bins, 18 real channels, at 6000 frames a second; one strided convolution for each stage, 18 -> 256 (k 30, stride 15), 18 -> 128 (k 6, stride 3) and 18 -> 64 (k 1, stride 1); a residual block on its output with k 7, 7 and 11; the result is added to the trunk after the stage's transposed convolution | biased |
| residual block (c channels, kernel k) | three branches in sequence on the same signal, each Snake, convolution c -> c (kernel k, dilation 1, 3 or 5), Snake, convolution c -> c (kernel k, dilation 1), the branch added to its input; a Snake multiplies by a learned per-channel alpha | six convolutions with bias and six alpha vectors of c: 6 (c x c x k + c) + 6 c |
| trunk | after each stage three residual blocks with k 3, 7 and 11 on the same input, their outputs averaged: 9 blocks in all |  |
| output convolution | after a leaky ReLU, 64 -> 18, k 7: for each of the 9 bins a logarithm of the magnitude and a phase argument, at 6000 frames a second, then the inverse transform | biased |
| harmonic merge | linear map 9 -> 1 at each of the 24,000 samples a second | biased (10 numbers) |
<!-- /worked -->

<!-- worked: speech_synthesis.vocoder_counts -->
Counted from the tensors of the released second-generation vocoder, after the weight-normalization gains are folded into the kernels (the file stores one extra gain per convolution, 14,738 numbers, which are not parameters of the folded model). A convolution costs c_out c_in k multiply-adds per output position and a transposed convolution c_in c_out k per input position, at the rates 50, 400, 2000 and 6000 positions a second; the inverse transform and the activations are not counted.

| part | parameters | multiply-adds per second of speech (millions) |
|---|---:|---:|
| pitch predictor: 5 convolutions (k 3) at 50 Hz, then a linear read-out | 3,271,681 | 163.5 |
| first convolution (80 -> 512, k 7) at 50 Hz | 287,232 | 14.3 |
| 3 transposed convolutions (512 -> 256, 256 -> 128, 128 -> 64) | 2,515,392 | 363.7 |
| source branch: 3 strided convolutions of the excitation's 18 channels | 153,664 | 89.9 |
| source branch: 3 residual blocks (k 7, 7, 11), 6 convolutions and 6 gains each | 3,716,352 | 4,099.3 |
| trunk: 9 residual blocks (k 3, 7, 11 at each of 3 stages) | 10,854,144 | 10,528.4 |
| output convolution (64 -> 18, k 7) at 6000 Hz | 8,082 | 48.4 |
| harmonic merge (9 -> 1) | 10 | 0.2 |
| total | 20,806,557 | 15,307.6 |

That is 15.3 billion multiply-adds for each second of speech.
<!-- /worked -->

The second-generation vocoder has 20.8 million parameters and costs 15.3 billion multiply-adds a second of speech, nearly all in the nine residual blocks and the three source blocks, which run at rates of 400 to 6000 positions a second with 256 to 64 channels. The third generation's vocoder is a different network (causal throughout, with nearest-neighbour upsampling and left-padded convolutions where the second has transposed convolutions), and the counts above are not its counts. One of the differences is subtle and deserves its story. The source module accumulates the phase at the frame rate and then raises it back to the sample rate; the second generation interpolates linearly there, and the third repeats each value. A first port used the linear form for both, matched the second generation's reference and broke the third's badly partway through the signal (the commit reports a cosine of about 0.28, and the code now selects the mode from the generation: commit `aa0332ad1`). What passed was a test of one generation, and what failed was the other, through a piece that the two share, which is why that module's tests are written against both generations' references.

## Where a second of speech spends its work

Putting the stages of the three pipelines on one scale needs only the tables above.

<!-- worked: speech_synthesis.work_per_second -->
Multiply-adds for one second of speech in billions, from the tables of this chapter (attention scores, norms and activations not counted; the third generation's vocoder is not counted here).

| system | stage | billions per second of speech |
|---|---|---:|
| 16-codebook synthesiser | Talker and code predictor, 12.5 frames | 21.7 |
|  | codec decoder | 31.0 |
|  | total | 52.7 |
| single-token synthesiser, second generation | speech-token model, 25 tokens | 9.1 |
|  | flow decoder, 20 evaluations of 50 frames, with its encoder | 67.5 |
|  | vocoder | 15.3 |
|  | total | 91.9 |
| single-token synthesiser, third generation | flow decoder, 20 evaluations of 50 frames | 189.0 |
<!-- /worked -->

Three observations follow. The second route's weight is in the flow decoder, not in the language model: 67.5 billion of the second generation's 91.9 billion, and 189.0 billion for the third's decoder alone, against 9.1 billion for the language model that chooses the words. The first route is cheaper in total (52.7 billion) and its work is split between generation (21.7 billion) and the codec decoder (31.0 billion). And a flow decoder's cost is a multiple of its step count and its guidance: with 10 steps and two evaluations each, 20 passes over every frame, the setting that controls quality in a flow decoder is the same setting that controls the cost. These are multiply-adds of the matrix products and convolutions only; attention scores are left out, and they are not small for the flow decoders (a 13-second span adds 37 billion a second to the second generation and 29 billion to the third, in the flow table). What these counts say is where the work is, not how long a given machine takes, which depends on how close each part gets to the machine's limits.

## Sound before the sentence is finished

A listener wants the first word while the third is still being written. What must exist before the first sample can come out? For each route the answer is a count of work.

The Qwen3-TTS engine that keeps its weights on a device (`ResidentEngine` in `crates/qwen3tts/src/engine.rs`) hands out its first sound once 8 frames exist, 0.64 seconds of speech. It uses a fact about the codec: the decoder is causal throughout, so the samples of a prefix of the codes are the first samples of the whole. The engine decodes the codes so far at 8 frames, at 24, at 72, and at every time the count reaches three times the last trigger, then once more on the finished clip, and each decode starts at frame zero and sends only the samples not yet sent. The repeated work is the price of not having a streaming decoder on this path, and it is bounded by the geometric spacing.

<!-- worked: speech_synthesis.stream_schedule -->
The streaming schedule of the device engine: the codec decodes the codes so far when the first 8 frames exist, then each time the frame count reaches 3 times the last trigger (8, 24, 72, 216), and once more on the finished clip unless the last trigger already decoded all of it; every decode restarts at frame 0 and only the samples not yet sent are handed out. A clip shorter than 8 frames is decoded once. One decode of n frames costs n x 2,479.4 million multiply-adds (the decoder of Chapter 3).

| frames n | seconds | frames decoded in each call | calls | frames decoded in all | ratio to n | seconds of speech before the first sound |
|---:|---:|---|---:|---:|---:|---:|
| 6 | 0.48 | 6 | 1 | 6 | 1.00 | 0.48 |
| 20 | 1.60 | 8, 20 | 2 | 28 | 1.40 | 0.64 |
| 24 | 1.92 | 8, 24 | 2 | 32 | 1.33 | 0.64 |
| 38 | 3.04 | 8, 24, 38 | 3 | 70 | 1.84 | 0.64 |
| 125 | 10.00 | 8, 24, 72, 125 | 4 | 229 | 1.83 | 0.64 |
| 250 | 20.00 | 8, 24, 72, 216, 250 | 5 | 570 | 2.28 | 0.64 |
<!-- /worked -->

A clip of 38 frames, the healthy length of the nine-word sentence, is decoded in three calls of 8, 24 and 38 frames, 70 frames in all, 1.84 times a single decode, and its first sound goes out when 0.64 seconds of speech exist. The row for 24 frames can be run by hand: a clip of 24 frames is decoded at 8 and at 24, and the trigger at 24 has already decoded the finished clip, so no third call is made. Until commit `c9b4af7e1` the finishing decode ran whatever the last trigger was, and a clip whose length equalled a trigger was decoded twice at that length, the second time sending nothing. The commit that introduced the scheme (`a1da5a2d6`) says the streamed audio equals the whole decode; its test (`crates/qwen3tts/tests/it/resident_device.rs`) asserts that the relative root-mean-square difference is below $10^{-3}$, which is a bound of closeness and not equality.

The host placement of the same engine streams only its codec. It generates every code first, then decodes them 16 frames, 1.28 seconds, at a time with the convolution state carried between chunks, so nothing is heard before the whole sentence is written. What the first sound costs on the device path is a sum that the table below makes explicit.

<!-- worked: speech_synthesis.first_sound -->
Multiply-adds before the first sound of the device engine, in billions: the reference's codes and speaker vector (a 10-second reference through the first speaker encoder, and for the in-context form also the codec's encoder), the prefix rows through the 28 layers of the Talker, 8 frames of generation (one Talker step with its head and the code predictor's 16 steps and 15 heads each), and the first decode of 8 frames. Attention scores and the text projections of the prompt are not counted.

| case | prefix rows | reference analysis | prefix | 8 frames generated | first decode | total |
|---|---:|---:|---:|---:|---:|---:|
| x-vector voice, vector computed now | 10 | 5.0 | 4.4 | 13.9 | 19.8 | 43.1 |
| x-vector voice, vector kept from an earlier request | 10 | 0.0 | 4.4 | 13.9 | 19.8 | 38.1 |
| in-context clone, reference seen for the first time | 135 | 32.6 | 59.5 | 13.9 | 19.8 | 125.8 |
| in-context clone, reference conditioning kept | 135 | 0.0 | 59.5 | 13.9 | 19.8 | 93.2 |
<!-- /worked -->

For a clone with its reference seen for the first time, the prompt's 135 rows and the reference's analysis cost more than the 8 frames that are generated, and keeping the reference's conditioning removes 32.6 billion of the 125.8. Even then 59.5 billion of the 93.2 are the reference rows going through the Talker's 28 layers once. For a voice used many times, the cost per request is the prefix, and a longer reference buys fidelity at that price.

The second route streams differently. `crates/cosyvoice/src/pipeline.rs` runs the whole utterance as a single pass of the language model, the flow decoder and the vocoder, and its flow decoder attends over the whole sequence in both directions. The released reference streams by a chunk rule, and its arithmetic shows that the flow decoder dominates before the first sound as it does over the whole utterance.

<!-- worked: speech_synthesis.stream_reference -->
The chunk rule of the released reference's streaming loop (the third generation's class), set against the single pass of brain's pipeline. A chunk is cut when the tokens generated so far exceed the tokens already sent by the hop plus the 3 tokens of look-ahead the flow decoder reads; the first hop is 25 tokens plus the padding that rounds the prompt up to a multiple of 25; after each chunk the hop doubles, up to 100. Each chunk runs the flow decoder again on the prompt tokens and every token up to the end of the chunk, and keeps only its new frames.

| prompt tokens | padding | first hop | tokens needed before the first chunk | seconds of speech generated |
|---:|---:|---:|---:|---:|
| 25 | 0 | 25 | 28 | 1.12 |
| 75 | 0 | 25 | 28 | 1.12 |
| 82 | 18 | 43 | 46 | 1.84 |
| 150 | 0 | 25 | 28 | 1.12 |

The chunks of a 250-token utterance (10 seconds) after a prompt of 75 tokens. A flow evaluation of T frames costs 20 x (T x matrix products + attention + the fixed part) multiply-adds, with the third-generation numbers of the flow table: the matrix products per frame, the attention 22 blocks x 2 x 1024 x T x T (each of T frames scores T keys, 2 x 1024 multiply-adds a pair), and the fixed part per evaluation; 'billions' are billions of multiply-adds. The cumulative column adds the unrounded flows, so it can differ by one from the sum of the printed ones.

| chunk | tokens already sent | hop | tokens through the flow | frames T | flow billions | cumulative billions |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 25 | 103 | 206 | 820 | 820 |
| 2 | 25 | 50 | 153 | 306 | 1,244 | 2,064 |
| 3 | 75 | 100 | 253 | 506 | 2,146 | 4,210 |
| 4 (final) | 175 | 75 | 325 | 650 | 2,841 | 7,051 |

The same utterance in one pass is T = 650 frames and 2,841 billion; the streamed run runs the flow on 1668 frames in all, 2.57 times as many. Before the first sound the language model has run its 95-row prompt and 28 tokens, 44.7 billion, and the first chunk's flow is 820 billion.
<!-- /worked -->

Before the first second of speech, the language model of the example has written 28 tokens after a 95-row prompt (44.7 billion multiply-adds), and the flow decoder has run 20 evaluations over the 206 frames of a 3-second reference and the new tokens (820 billion). Every later chunk runs the decoder again on the reference and all that has gone before, so the streamed utterance costs 7,051 billion against 2,841 for one pass, 2.5 times. The reference's design pays that for a first sound after 28 tokens, which is the exchange to compare with the 8 frames and 38 billion of the Talker's device path, and the price of that exchange is that every chunk pays for the reference again.

## What the three crates and their histories show

The three crates were built in different ways, and each carries one lesson that outlives its model.

The Talker came first, and fast: the whole path from text and a reference clip to a waveform, with greedy, temperature and top-k sampling, was in one commit on 28 June 2026 (`79f30e24c`), the same day as the codec and the speaker encoder it composes. What it lacked showed up later: more than two months on, `GenOpts` still carried only the options its author had needed, while the checkpoint carried a whole file of sampling defaults nobody read. The silent clip was the first symptom, and the cure was not a number but a rule about where numbers come from.

The CosyVoice crate was built the opposite way, from the inside out. Each part (the language model, the flow decoder, the vocoder) was ported against a real checkpoint and gated at its own interface before any two were joined, and the pipeline that composes the second generation's three came after them (`5139936e9`); the third generation's parts followed. The crate's documentation records the places where the port cannot be bit-exact: its sampler draws from its own generator, not the framework's, so a seed gives the same tokens on this implementation and not the reference's, and the flow decoder's fixed noise is the one random buffer that is reproduced exactly. The third generation was wired into the pipeline late, in commit `f6ea9255e`, whose message is explicit that every component already had its own real-weight parity, that what was missing was "twenty lines" of composition, and that no real third-generation run end to end had been made, because no checkpoint was available. The crate's top-level documentation, brought up to date by commit `280f1db3e` (11 November 2026), says that one pipeline runs both generations.

The s3tokenizer crate shows the value of a gate that cannot be approximated. The tokenizer's output is integers; a test that required closeness would pass on tokens that differ, and a test that requires equality fails on a one-bit rounding flip, and the crate was designed around the second.

## Check your understanding

1. A reader says that the Talker must run 16 positions for each frame because a frame has 16 codes. What is wrong, and what does the real arrangement save?
2. The code predictor has 5 layers and the Talker 28, so the code predictor is the cheap half of a frame. Find the error in the reasoning and give the correct ratio of layer-steps.
3. A penalty of 1.05 changes a logit of 5.0 by only 0.238. Why can it still stop a repetition loop whose top-1 probability is 0.9998, and what feature of the rule makes it act hardest on the tokens that need it?
4. A reimplementation applies the penalty once for each occurrence. After a token has been seen 50 times, by what factor is its positive logit divided, and why would a test of sampled output at small sample sizes miss this?
5. Two tokens of the speech tokenizer differ only in their last coordinate. By how much do they differ as integers, and why does the model that reads them never use that difference?
6. In the toy, guidance moved the centre of the end points from 1.5 to 1.73 and shrank their spread from 0.3 to 0.129. What would you expect the guided mel of a real decoder to sound like compared with an unguided one, and what is the cost of getting more of it?
7. A clip is exactly 72 frames. How many times is the codec decoder called by the device engine, on how many frames, and how many frames of decoding did the rule before commit `c9b4af7e1` add?
8. The flow decoder of the third generation needs 28 tokens before it can emit the first chunk of a stream, but the first chunk costs about 820 billion multiply-adds. Name two things that make that number so much larger than the language model's share.

## Summary

Speech synthesis puts an interface between text and waveform and builds a model on each side of it. Qwen3-TTS puts it at the codec's 16 codes per frame: a causal language model of 28 layers, the Talker, reads one summed row per 80-millisecond frame, with the text laid alongside at one row per frame and the speaker vector as one prompt row, and writes the first code, and a 5-layer code predictor writes the other 15 in 16 small steps that cost 2.9 times the Talker's layer arithmetic. The first code is drawn through a chain that masks the specials, penalises repeats once for each distinct id, scales, truncates and draws, and the length of a clip is the frame at which the end-of-speech id is allowed and drawn. A silent clip of 200 frames was a repetition loop under a penalty that a default had left at 1.0, and the repair was a precedence of sources for every sampling setting. CosyVoice puts the interface at 25 speech tokens a second from a tokenizer trained to preserve what is said, quantized by rounding eight coordinates to three levels each, written by a 0.5-billion language model with a repetition-aware sampler; a flow-matching decoder then generates the 80-bin mel from noise in 10 guided Euler steps, 20 network evaluations of every frame, conditioned on the tokens, the speaker vector and the reference's mel, and a vocoder with a harmonic source and an inverse short-time transform turns the mel into 24,000 samples a second. Counted in multiply-adds, the first route costs 52.7 billion a second of speech, the second 91.9 billion with the second generation's decoder and 189.0 billion for the third's decoder alone, nearly all of it in the flow decoder. Streaming is a question of what must exist before the first sample: 8 frames and about 38 billion multiply-adds for the Talker's device path, and 28 tokens and about 820 billion for the reference's flow decoder alone, because it runs again on the reference at every chunk.

The next chapter keeps the codec language model and changes the target: minutes of structured music, where the length of the sequence and not its voice is the difficulty.

## References

- Wang, Chen, Wu, Zhang, Zhou, Liu, Chen, Liu, Wang, Li, He, Zhao and Wei, *Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers*, 2023.
- Mentzer, Minnen, Agustsson and Tschannen, *Finite Scalar Quantization: VQ-VAE Made Simple*, 2023.
- Du, Chen, Zhang, Gao, Zhang and others, *CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens*, 2024.
- Du, Wang, Chen, Zhang and others, *CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models*, 2024.
- Mehta, Tu, Henter and Beskow, *Matcha-TTS: A Fast TTS Architecture with Conditional Flow Matching*, 2024.
- Kong, Kim and Bae, *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis*, 2020.
- Kaneko, Tanaka, Kameoka and Kanagawa, *iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform*, 2022.
- Wang, Takaki and Yamagishi, *Neural Source-Filter Waveform Models for Statistical Parametric Speech Synthesis*, 2020.
- Peebles and Xie, *Scalable Diffusion Models with Transformers*, 2023.
- Ho and Salimans, *Classifier-Free Diffusion Guidance*, 2021.

## Exercises

1. **[Calculate]** How many rows come before frame 0 of the Talker's prompt for the x-vector form with a language tag and no speaker row? How many for an in-context clone with a speaker row, a language tag and 50 reference frames?
2. **[Calculate]** A token with logit 8.4 has been generated 20 times. Under a penalty of 1.05 applied once for each distinct id, and under the same penalty applied once for each occurrence, what is its logit?
3. **[Calculate]** Compute the speech token of the vector $(0.1, 0.7, -0.9, 0.6, -0.5, 0.0, 1.2, -0.45)$.
4. **[Calculate]** A reference clip lasts 5 seconds. How many speech tokens does the tokenizer write, how many mel frames does the 24 kHz front end make, and how many tokens and mel frames does the prompt keep?
5. **[Calculate]** For a clip of 100 codec frames, list the frames decoded by each call of the device engine and the total.
6. **[Calculate]** How many bytes of attention cache do the Talker and the speech-token model hold for 60 seconds of speech, at 32-bit floats?
7. **[Calculate]** In the toy flow decoder, compute $u_c$, $u_{\mathrm{un}}$ and the guided $u_g$ (for $g = 1.7$) at $x = 1.0$ and $t = 0.5$.
8. **[Implement]** Reproduce `speech_synthesis.sampling_chain`: the chain of the pseudocode on the toy logits and history, for both settings of the end-of-speech guard, and the two draws.
9. **[Implement]** Reproduce `speech_synthesis.fsq_toy` and `speech_synthesis.tokenizer_geometry`: the digits and tokens of the two vectors and the table of clip lengths.
10. **[Implement]** Reproduce `speech_synthesis.cfm_toy`: the stream named there, the six starts in order, the cosine grid, the exact conditional map and the guided Euler runs, to the digits printed.
11. **[Investigate]** Run the toy flow decoder at guidance scales $g = 1, 1.5, 1.7, 2, 2.5$ and $3$ and tabulate where $z = 0$ ends and by how much the end points spread. Does more guidance always push the centre further from the unconditioned mean 0? What does the spread do, and what would that mean for the variety of a decoder's mels?
12. **[Investigate]** Resample a tone at 15 kHz sampled at 48 kHz to 24 kHz with the linear function of Chapter 1 and with the Kaiser design. Where does the tone appear in each? What fraction of a speech recording's energy would have to lie above 12 kHz for the fold to matter to a voice you can measure?

## Key terms

- **text-to-speech**: the task of turning a text, and usually a description of a voice, into the waveform of someone saying it. <!-- concept: text-to-speech -->
- **codec language model**: a causal language model whose vocabulary is the codes of a neural audio codec and whose context carries the text and the voice, so that it writes audio as a sequence of codes that the codec's decoder turns into a waveform. <!-- concept: codec-language-model -->
- **speech tokenizer**: a network trained, with a supervised target such as the transcript and not by reconstruction, to turn speech into discrete tokens that carry what is said and little of who says it or how. <!-- concept: speech-tokenizer -->
- **vocoder**: a network that turns a spectrogram back into a waveform, supplying the phase and the fine periodic structure that the spectrogram discards. <!-- concept: vocoder -->
- **voice cloning**: making a synthesiser speak in the voice of a person from whom it has only a short recording, without training on that person. <!-- concept: voice-cloning -->

## Answers

**Recall questions.**

1. 12.5 columns a second of 16 codes is 200 integers a second. The first code comes from a separate group of one codebook, while the other fifteen are a residual chain.
2. It multiplies the probabilities of each token given the tokens before it. Teacher forcing gives the model the true earlier tokens, shifted by one position, whatever the model itself would have written.
3. The positive logit becomes $3.0 / 1.25 = 2.4$ and the negative one $-2.0 \times 1.25 = -2.5$; both move toward making the token less likely. The nucleus keeps the shortest prefix of the sorted probabilities whose sum reaches 0.8: the first two, $0.5$ and $0.3$.
4. Noise is at $\sigma = 1$ and data at $\sigma = 0$; a step is $x_{\sigma'} = x_\sigma + (\sigma' - \sigma)\,u(x_\sigma, \sigma)$, with a negative factor on the way from noise to data.
5. $u_g = u_{\mathrm{un}} + g\,(u_c - u_{\mathrm{un}})$. With $g = 1$ it is the conditional field, with $g = 0$ the unconditional one, and with $g > 1$ it extrapolates beyond the conditional one.
6. $(24000 + 2 \cdot 720 - 1920) // 480 + 1 = 50$ frames.

**Check your understanding.**

1. The sixteen codes of a frame are looked up in their own tables and added into one row, so the Talker has one position per frame. For ten seconds that is 125 positions and 7,875 pairs per head and layer against 2000 positions and 2,001,000 pairs, 254 times as many.
2. It confuses the number of layers with the number of steps. The code predictor runs 16 decoder steps of 5 layers, 80 layer-steps, against the Talker's 28, so 80 / 28 = 2.9 times the layer arithmetic.
3. The factor acts on the logit, and the reduction is $z(1 - 1/1.05)$, which grows with $z$: 0.238 at 5 and 0.952 at 20. A token with probability 0.9998 has a logit far above the rest, so it receives the largest handicap, applied at every step it is in the history, and the loop's self-reinforcement has to beat it each time.
4. By $1.05^{50} = 11.47$, so 5.0 becomes 0.436 (and 131.5 after a hundred repeats). A short clip repeats a token only a few times, where the two rules give factors of 1.05 and $1.05^3 = 1.16$, so a test on short clips cannot tell them apart; only long runs do.
5. The last coordinate has the place value $3^7 = 2187$, so the integers differ by 2187 or 4374 while the vectors differ in one digit. The model reads a token through an embedding table, one row per id, so the integer's size is never used and no ordering of the ids carries meaning.
6. Guided speech follows its conditions harder (the words, the voice) and varies less from run to run, and with too much guidance sounds over-articulated or monotone, because the end points are pushed past what the data contain. The cost is a second evaluation of the network at every step, twice the work.
7. The calls decode 8, 24 and 72 frames, three calls and 104 frames in all; the trigger at 72 has already decoded the finished clip. The earlier rule decoded the finished clip again, a fourth call of 72 frames (176 in all) that sent nothing.
8. The flow decoder runs 20 evaluations (10 steps with guidance), each over every frame, and the reference's frames are part of the sequence: the first chunk's T is 206 frames, the reference's 150 and the tokens' 56, so most of the work is spent on the reference, which a language model step pays only once in its prefix.

**Exercises.**

1. $m = 4 + 0 + 2 = 6$ and the rows before frame 0 are $m + 3 = 9$. For the clone, $m = 7$ and the rows are $3 + (7 - 1) + (1 + 50) = 60$.
2. Once for each distinct id: $8.4 / 1.05 = 8.0$. Once for each occurrence: $8.4 / 1.05^{20} = 8.4 / 2.6533 = 3.166$.
3. The digits are $(1, 2, 0, 2, 1, 1, 2, 1)$ and the token is $1 + 6 + 0 + 54 + 81 + 243 + 1458 + 2187 = 4030$.
4. At 16 kHz, 80,000 samples give 500 mel frames and, after two stride-2 convolutions, $(500 - 1)//2 + 1 = 250$ and then $(250 - 1)//2 + 1 = 125$ tokens. At 24 kHz, 120,000 samples give $120000 // 480 = 250$ frames, so $250 // 2 = 125$. The prompt keeps $\min(125, 125) = 125$ tokens and 250 mel frames.
5. The calls decode 8, 24, 72 and 100 frames, 204 frames in all, 2.04 times the clip.
6. Talker: 750 positions at 229,376 bytes, 172,032,000 bytes. Speech-token model: 1500 positions at 24,576 bytes, 36,864,000 bytes.
7. With $m = 1.5$ and $\varsigma = 0.3$ at $t = 0.5$: the variance is $0.25 + 0.25 \cdot 0.09 = 0.2725$, the numerator $0.5 \cdot 0.09 - 0.5 = -0.455$ and $x - tm = 0.25$, so $u_c = 1.5 - 0.455 \cdot 0.25 / 0.2725 = 1.0826$. The unconditioned field has $m = 0$ and $\varsigma = 1$: variance 0.5, numerator $0.5 - 0.5 = 0$, so $u_{\mathrm{un}} = 0$. Then $u_g = 0 + 1.7 \cdot 1.0826 = 1.8404$.
8. The chain's tables of the chapter: the masked ids are 8 to 11, the history 3, 3, 5, 3 gives the distinct ids 3 and 5, and the draws are 3 and 5 with the guard closed and 10 and 10 with it open.
9. The tokens 4964 and 1984 and the rows of the geometry table, including the dropped token at 2.5 and 3.3 seconds.
10. The starts are the six draws of the stream `speech_synthesis.flow_start` named in the table, and the end points are those of the last two columns.
11. The field is linear in $x$, so each guided map is a line, found by integrating from two starts.

<!-- worked: speech_synthesis.guidance_sweep -->
The toy flow decoder with the guidance scale g of the first column, 2000 uniform Euler steps. The field is linear in x, so the map from the start z to the end point is a line: 'centre' is its value at z = 0 and 'slope' its change per unit of z, which is the standard deviation of the end points for a standard normal start.

| g | centre | slope |
|---|---:|---:|
| 1 | 1.5000 | 0.2997 |
| 1.5 | 1.6959 | 0.1641 |
| 1.7 | 1.7336 | 0.1289 |
| 2 | 1.7651 | 0.0898 |
| 2.5 | 1.7785 | 0.0491 |
| 3 | 1.7697 | 0.0269 |
<!-- /worked -->

    The centre does not keep moving away from 0: it rises from 1.5 to a largest value near 1.78 (at $g$ near 2.5) and then falls. The spread shrinks all the way, from 0.30 at $g = 1$ to 0.027 at $g = 3$, so for a decoder more guidance means less variety in the mels it can produce from different starting noises.
12. A 15 kHz tone at 48 kHz, kept at every second sample by the linear function, has folded frequency $|15000 - 24000| = 9000$ Hz: it appears at 9 kHz. The Kaiser design suppresses it, as the tables of Chapter 1 show for a 10 kHz tone. The fraction depends on the recording; measure it from a spectrogram of the kind built in Chapter 2.
