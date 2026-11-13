<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Models That Listen, See and Speak

*How is one model built to take in audio, images and text and answer in speech?*

Give the model a minute of video with its sound, a request of 50 tokens, and ask what is in it. Before the first word of the answer exists, the model has done 89 trillion multiply-adds, and the question worth asking first is not how it understands the clip but where those operations went. Not into the sound: the audio tower reads the whole minute for 0.63 trillion, a little over 0.7 percent. Not into the answer: the first token of it is the last thing computed. They went into rows. The sound became 780 of them, the video 11,760, the request 50 and the markers 4, and a decoder of 48 layers read all 12,594 as one sequence, each row attending to every row before it. The sound is the cheapest part of what a model that listens has to carry, and the picture is the most expensive, by a factor of fifteen in rows.

That is the shape of the chapter. brain's crate for the model, `crates/qwen3omnimoe`, holds one network that takes in sound, pictures, video and text and gives back text and speech, and what it adds to the earlier chapters of this volume is not a new kind of layer. The parts are all ones the reader has met: a speech encoder from Chapter 4, a vision tower and projector from Volume VII, a mixture-of-experts decoder from Volume VI, the codec language model and the codec decoder of Chapters 3 and 6. What is new is how they are joined, and the joining raises four questions that the sections take in order: how the rows of three kinds of input are laid in one sequence and given positions; what a model of that size holds and what it uses; why the voice is a second decoder and what crosses to it from the first; and when, after the question is asked, the first sample of the answer can exist. A model that does all of this I will call an **omni model**: one network whose decoder reads rows from the encoders of several media in a single sequence, and that answers in more than one medium, here in text and in speech.

An omni model combines many earlier parts, and six of them are recalled here.

1. ([Volume IX, Chapter 4](04-speech-recognition.md), the speech encoder.) A clip of 2 seconds at 16 kHz goes through a log-mel front end with a hop of 160 samples and an encoder that cuts the frames into chunks of 100 and takes each chunk through three stride-2 convolutions with padding 1. How many rows does the clip become?
2. ([Volume VII, Chapter 9](../v07-vision/09-vlms.md), the vision projector.) A 448 by 448 picture is cut into patches of 16 by 16 pixels, and a projector merges each 2 by 2 block of patches into one row. How many rows does it hand to the decoder?
3. ([Volume VII, Chapter 9](../v07-vision/09-vlms.md), multi-axis positions.) A block of picture rows has the grid $(t, h, w) = (1, 2, 2)$ and starts where the position counter reads $c$. Which triples do its rows get, and where does the counter stand after it?
4. ([Volume IX, Chapter 6](06-speech-synthesis.md), the codec language model.) Why does the Talker advance one position for each frame of 16 codes and not one for each code, and which network writes the codes 1 to 15?
5. ([Volume VI, Chapter 4](../v06-language-models/04-mixture-of-experts.md), the router.) A layer has 128 experts and a token is sent to 8. What share of the expert parameters does the token take part in, and how are the 8 weights of the combination formed?
6. ([Volume IX, Chapter 3](03-audio-codecs.md), audio tokens.) A codec writes 12.5 columns of 16 codes a second, each code from 2048 entries. How many integers and how many bits is that in a second?

## Rows from three kinds of input

The decoder of this model has one input width, 2,048 numbers, and everything that is to be read must arrive as rows of that width. Text gets there by a table lookup: the embedding table of 152,064 rows, one row for each token. Sound gets there through the audio tower, which is the speech encoder of Chapter 4 at a larger scale: the same three strided convolutions over the 128-bin log-mel picture (480 channels, chunks of 100 frames), the same windows of 800 frames for attention, and then 32 blocks of width 1,280 with 20 heads and a feed-forward of 5,120 where the recogniser has 24 of 1,024, and a two-layer projector from 1,280 to 2,048. A picture gets there through the vision tower of Volume VII: 27 blocks of width 1,152, patches of 16 pixels, a merger that joins each 2 by 2 block of patches and maps the four rows of 1,152 to one of 2,048. The crate does not carry a second encoder implementation for either: `crates/qwen3omnimoe/src/mm.rs` builds `qwen3asr`'s audio encoder and `qwen3vl`'s vision encoder with the Omni presets (`AudioEncoderConfig::qwen3_omni`, `VisionConfig::qwen3_omni`), which differ from the recogniser's and the 4-billion model's only in width and depth, and the test files `audio_parity.rs` and `vision_parity.rs` compare them with the released checkpoint's own outputs. How many rows each medium becomes is the first thing to count.


![Two panels. The first: 16 kHz sound goes through an audio tower of 32 blocks of width 1,280 giving 13 rows a second, frames of video and pictures through a vision tower of 27 blocks of width 1,152 giving 196 rows for a 448 by 448 picture, and text tokens through an embedding table of 152,064 by 2,048, and all three arrive at the Thinker, which has 48 layers of width 2,048 and 8 of 128 experts used. The second: the reply tokens of the Thinker, plus the layer-24 states of the media rows, go into the Talker of 20 layers with 6 of 128 experts, then a code predictor of 5 layers writing codes 1 to 15, then Code2Wav with 8 layers and an upsampling by 1,920, ending in speech at 24,000 samples a second.](img/omni-pipeline.svg)

<!-- worked: omni_models.rows_per_medium -->
Rows that each medium hands to the decoder. Sound: the audio tower cuts the 100-frames-a-second log-mel picture (hop 160 samples) into chunks of 100 frames, each of which gives 13 rows (a partial chunk fewer, by three halvings that round up). Picture: both sides are resized by the rounding rule of Volume VII, Chapter 9 (called `smart_resize` there): each side goes to the nearest multiple of 32, a half way value to the even multiple, and at least 32; if the area is then over 3584 x 3584 both original sides are divided by sqrt(h w / 3584^2) and rounded down to a multiple of 32 (at least 32), and if it is under 56 x 56 both are multiplied by sqrt(56^2 / (h w)) and rounded up; none of the printed pictures reaches a tie or a limit. Then a patch of 16 x 16 pixels, and 2 x 2 patches merged into one row. A pair of video frames is one picture's worth of rows.

| medium | input | frames or patches | rows | rows a second |
|---|---|---:|---:|---:|
| sound | 0.5 seconds at 16 kHz | 50 | 7 | 14.0 |
| sound | 1 second at 16 kHz | 100 | 13 | 13.0 |
| sound | 2 seconds at 16 kHz | 200 | 26 | 13.0 |
| sound | 10 seconds at 16 kHz | 1,000 | 130 | 13.0 |
| sound | 60 seconds at 16 kHz | 6,000 | 780 | 13.0 |
| picture | 64 x 64, resized to 64 x 64 | 4 x 4 patches | 4 | - |
| picture | 448 x 448, resized to 448 x 448 | 28 x 28 patches | 196 | - |
| picture | 480 x 640, resized to 480 x 640 | 30 x 40 patches | 300 | - |
| picture | 1080 x 1920, resized to 1088 x 1920 | 68 x 120 patches | 2,040 | - |
| video, 2 frames a second | 448 x 448, one pair of frames a second | 28 x 28 patches | 196 | 196.0 |

Per second, what goes in and what comes out (rows and integers are counted, not timed).

| stream | per second | each |
|---|---|---|
| sound in, one second | 13 rows | of 2,048 numbers |
| picture in, 448 x 448, one pair of frames a second | 196 rows | of 2,048 numbers |
| speech out, codes | 12.5 frames x 16 codes = 200 integers | of 11 bits: 2,200 bits |
| speech out, waveform | 24,000 samples | 16 bits: 384,000 bits |
<!-- /worked -->

A second of sound is 13 rows, a pair of video frames at 448 by 448 is 196. The sound of the recogniser's chapter keeps its slightly awkward arithmetic (a clip of half a second is 7 rows, not 6.5, because a partial chunk rounds up three times), and the picture keeps the rounding rule of Volume VII: the table's 1080 by 1920 picture is resized to 1088 by 1920 and becomes 2,040 rows. What leaves the model is a different order of quantity again. Speech is written as 200 integers a second, 2,200 bits, against the 384,000 bits of the samples they stand for, so the model's answer is the thinnest stream of the second table in bits and its video, in rows, the widest.

Now the minute. The video of the example is sampled at 2 frames a second, so a pair of frames covers one second and the vision tower runs once for each pair; the sound is 16 kHz mono; the request is 50 tokens; and each medium is bracketed by a start and an end marker, which is 4 more.

<!-- worked: omni_models.minute_of_video -->
A clip of 60 seconds with its sound, the video sampled at 2 frames a second at 448 x 448 pixels (resized to 448 x 448, a grid of 14 x 14 merged rows), preceded by a request of 50 tokens.

| part | rows |
|---|---:|
| text of the request (an input of the example) | 50 |
| start and end markers, one pair for the sound and one for the video | 4 |
| sound: 60 chunks of 13 rows | 780 |
| video: 120 frames, 60 pairs of 196 rows | 11,760 |
| sequence the Thinker prefills | 12,594 |

Multiply-adds, in trillions, that the clip costs before the first token of the reply (the vision tower's three DeepStack mergers are extra copies of the final merger that the checkpoint carries; their outputs are added to the visual rows after the Thinker's first three layers, so each costs the main merger's two linear maps; norms, activations, the additions and the position-table resampling are left out).

| stage | trillions of multiply-adds | percent |
|---|---:|---:|
| audio tower: convolution stem | 0.134 | 0.1 |
| audio tower: 32 blocks, weights | 0.491 | 0.5 |
| audio tower: attention in windows of 104 rows | 0.006 | 0.0 |
| audio tower: projector | 0.003 | 0.0 |
| vision tower: patch embedding, 60 pairs | 0.083 | 0.1 |
| vision tower: 27 blocks, weights, 60 pairs | 19.337 | 21.6 |
| vision tower: attention inside each pair's grid | 2.294 | 2.6 |
| vision tower: merger | 0.361 | 0.4 |
| vision tower: three DeepStack mergers | 1.082 | 1.2 |
| Thinker: weights of 48 layers, every row, and the head once | 34.388 | 38.5 |
| Thinker: attention scores and weighted sums | 31.186 | 34.9 |
| everything before the first reply token | 89.366 | 100.0 |

The cache that the prefill leaves behind: 48 layers x 2 x 4 key-value heads x 128 x 4 bytes is 196,608 bytes a position (32-bit floats, as brain's caches are), so 12,594 positions hold 2,476,081,152 bytes, 2.31 GiB.
<!-- /worked -->

93.4 percent of the sequence is video, 6.2 percent is sound, and the share of the work is not far from the share of the rows: of the 89.37 trillion multiply-adds, the vision tower takes 23.16 and the audio tower 0.63. The Thinker's weights applied to 12,594 rows are 34.39 trillion, and the Thinker's attention is 31.19, 34.9 percent of the total, because every row reads every earlier row and the sequence is long: the 12,594 rows make $12{,}594 \cdot 12{,}595 / 2$ query-key pairs in each of 48 layers. That is the prefill in the sense of Volume V, Chapter 12, and what it leaves behind is its cache, 196,608 bytes for each position (48 layers, two tensors, 4 key-value heads of 128, 32-bit values, as brain's caches are) and 2.31 GiB for the clip. These are the numbers a serving system has to plan around, and they are driven by the one choice of the example that a user makes without thinking about it, the number of frames a second. At 1 frame a second the video would be 30 pairs and 5,880 rows, and the vision tower would do half of the work the table counts.

The towers also tell what a mixture of experts is not. The Thinker is the sparse part of the model: a row uses 8 of 128 experts. The two towers are dense, every parameter works on every row, and the 23.8 trillion multiply-adds they spend on this clip are close to the 34.39 trillion the sparse Thinker spends on every row of it, though the towers are about three percent of the model's parameters. A budget that is read as "3 billion active parameters" is true of one token of the reply and says nothing of the clip.

## Two clocks in one sequence

The rows are ready; they have to be put in order. A decoder that is causal reads each row with the rows before it and no row after it, so the order of the splice is also a decision about who may see whom. Put the whole sound first and the whole video after it, and every video row can attend to all the sound, and no sound row ever attends to a picture: the sound of second 3 is read without knowing what was on the screen at second 3. Put them in the order of time, and the sound of second 3 follows the pictures of seconds 0 to 3 and precedes those of seconds 4 onward, which is what listening to a video is. The order has a second half, the position that each row carries into the rotary embedding, and for a sequence of two media that move on the same clock the position has a job that it did not have in Volume VII's picture of a still image: it has to say that the voice at second 3 and the frame at second 3 are at the same moment.

The checkpoint's configuration names the two numbers that this needs: `thinker_config` has `position_id_per_seconds`, 13, and `seconds_per_chunk`, 2, the fields by which the reference scales the time axis of sound and video by the wall clock. What follows is the rule these numbers define, written as one definition. A clip is cut into chunks of 2 seconds. A chunk lists first the rows of its video groups, then the rows of its sound. A group of video is the rows of one pair of frames, laid on a grid of $h$ by $w$ merged rows, and the sound has 13 rows a second. With $p_0$ the position at which the clip begins, a sound row $k$ gets the same position on all three axes, $(p_0 + k, p_0 + k, p_0 + k)$, as a word would, and the row $(i, j)$ of video group $g$ gets $(p_0 + 13 g,\ p_0 + i,\ p_0 + j)$ when each group covers one second. The first axis is the clock, 13 positions a second for both media, and the other two are the grid, anchored at $p_0$. After the clip the counter stands one past the largest position that any row has on any axis. As a procedure:

```text
function clip_positions(p0: int, seconds: int, gh: int, gw: int, P: int, chunk: int)
                        -> (list<(int, int, int)>, int)
    require seconds mod chunk == 0, "whole chunks"
    out = []
    top = p0
    for c in 0 .. seconds // chunk
        for g in c * chunk .. (c + 1) * chunk          # one group of video a second
            for i in 0 .. gh
                for j in 0 .. gw
                    append (p0 + P * g, p0 + i, p0 + j) to out
                    top = max(top, p0 + P * g, p0 + i, p0 + j)
        for k in 13 * c * chunk .. 13 * (c + 1) * chunk  # 13 rows of sound a second
            append (p0 + k, p0 + k, p0 + k) to out
            top = max(top, p0 + k)
    return (out, top + 1)
```

On a clip of 4 seconds, small enough to list, with $P = 13$, chunks of 2 seconds, a grid of 2 by 2 merged rows and a text prefix of 6 tokens so that $p_0 = 6$:

<!-- worked: omni_models.interleave_toy -->
A clip of 4 seconds, video at 2 frames a second (one pair, one group, a second) with a grid of 2 x 2 merged rows, sound at 13 rows a second, after a text prefix of 6 tokens (positions 0 to 5). Positions per the time-aligned rule with 13 positions a second and chunks of 2 seconds; within a chunk the video groups come first, then the sound. The counter after the clip is one more than the largest position on any axis: 58.

| segment | sequence index | rows | T | H | W |
|---|---|---:|---:|---:|---:|
| video group 0 | 6 to 9 | 4 | 6 | 6 to 7 | 6 to 7 |
| video group 1 | 10 to 13 | 4 | 19 | 6 to 7 | 6 to 7 |
| sound rows 0 to 25 | 14 to 39 | 26 | 6 to 31 | 6 to 31 | 6 to 31 |
| video group 2 | 40 to 43 | 4 | 32 | 6 to 7 | 6 to 7 |
| video group 3 | 44 to 47 | 4 | 45 | 6 to 7 | 6 to 7 |
| sound rows 26 to 51 | 48 to 73 | 26 | 32 to 57 | 32 to 57 | 32 to 57 |

The same clip under brain's rule: the sound as one block, then the video as one block, a pair at T equal to the block's start plus its ordinal (markers and the text after the clip are left out, as above). The counter after both blocks is 62.

| segment | sequence index | rows | T | H | W |
|---|---|---:|---:|---:|---:|
| sound block | 6 to 57 | 52 | 6 to 57 | 6 to 57 | 6 to 57 |
| video block | 58 to 73 | 16 | 58 to 61 | 58 to 59 | 58 to 59 |

The sound row that starts the third second is row 39 and the video group of that second is group 3. Time-aligned: the sound row sits at 45 on every axis and the group at T = 45. Frame-ordinal: the sound row sits at 45 and the group at T = 61, with the whole sound block of 52 rows between them in the sequence.
<!-- /worked -->

The first table is the rule. Group 3 of the video and sound row 39, which is where the third second of sound begins, are both at $T = 45$: the picture and the voice of the same moment sit on the same tick of the first axis, and an attention head whose rotary angles come from that axis sees them at distance 0. The sound rows run through the diagonal of all three axes, so a stretch of sound alone is positioned exactly as a stretch of text would be, and a clip without video reduces to the plain rotary embedding that the earlier volumes used. The picture grid, on the other hand, stays pinned to its anchor: the 4 rows of every group are on the same two small ranges of $H$ and $W$ whatever the second, which says where in the frame a row is and nothing about when, as in Volume VII.

The second table is the crate's own frame-ordinal rule, and it is a different design. The crate's multimodal prompt builder, `build_multimodal_prompt` in `crates/qwen3omnimoe/src/mm.rs`, encodes each medium to a block and splices the blocks, as whole blocks, at one point of the prompt, the sound first, the picture next and the video last. Positions come from `get_rope_index_multi` in `crates/qwen3vl/src/mrope.rs`, which is the rule of Volume VII, Chapter 9 with one case added: a run of placeholder rows whose grid has no spatial extent is a diagonal run, like text, and any other block is a meshgrid whose counter advances afterwards by $\max(t, h, w)$. Given a sequence that holds the sound block and the video block, it produces the second table: the video's $T$ axis counts pairs, 0, 1, 2, 3, and begins where the sound block ended. A video blob carries frames and no frame rate, so there is no seconds-per-group to scale the video's axis by, and the crate's configuration keeps neither `position_id_per_seconds` nor `seconds_per_chunk`. For the same reason the serving path takes sound or video in a prompt and refuses the two together with a message that says why (`check_media_combination`, commit `c1f2cc0b8`): the reference positions of a pair interleave them by the time axis that needs the frame rate. Sound with a picture, and either alone, are positioned as above.

A figure makes the difference visible for the whole minute.

![A line chart of position on the T axis against the second of a clip of 60 seconds. The sound rows rise from 0 to 767, and the time-aligned video groups lie on the same line. The frame-ordinal video groups lie on a nearly flat line from 780 to 839, entirely above the sound.](img/position-clocks.svg)

<!-- worked: omni_models.position_clocks -->
The minute of the chapter (60 seconds, video groups of 14 x 14 merged rows), no text prefix: the position on the T axis of the first sound row of second s and of the video group of second s, under each rule. Frame-ordinal: the video block starts after the 780 sound rows, at position 780; the counter after the clip is 840 against 780 for the time-aligned rule.

| second s | time-aligned: sound | time-aligned: video | frame-ordinal: sound | frame-ordinal: video | video minus sound |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 780 | 780 |
| 1 | 13 | 13 | 13 | 781 | 768 |
| 10 | 130 | 130 | 130 | 790 | 660 |
| 30 | 390 | 390 | 390 | 810 | 420 |
| 59 | 767 | 767 | 767 | 839 | 72 |
<!-- /worked -->

In the aligned scheme the picture of second 59 is 767 positions after the picture of second 0, as far from it as the sound is, and in brain's it is 59. After the whole clip the counter is 780 in the first and 840 in the second: the words of the reply come 60 positions later in the second, and, more to the point, every video row sits after all 780 sound rows. Rotary attention reads differences of positions, so what the decoder can learn about a word said at second 3 and an object on screen at second 3 is not what it could learn if they were at the same position, and a model that was trained with the first rule and is given the second reads a picture of time that is not the one it was trained on.

### Two splices that went wrong

The history of this part of the crate is the history of two positions that the first versions got wrong, and both are worth telling because both fail without an error.

The first is the position of a sound row. When audio input was first wired in, `get_rope_index_multi` treated a run of rows with a grid of $(n, 1, 1)$ like a picture and pinned $H$ and $W$ at the block's anchor for every row, so that every sound row had the same position on two of the three axes. Commit `e39f5fb08` (13 August 2026) says the symptom: with sound, and much worse with sound and a picture together, the output was broken and the chat template's own control tokens leaked into it, while a picture alone worked. It says the cause too, that the reference gives every sound row a plain diagonal position, $\mathrm{arange}(n)$ on all three axes shifted to the block's start; the code now takes that branch whenever `h == 1 && w == 1`, and the function's tests were rewritten, two of which had encoded the old expectation. The lesson is about positions in general: a rotary channel that every row of a block shares puts all those rows on one point of that channel, which is a position, a wrong one, and no shape or error message says so.

The second is the position of a block in the sequence. Commit `36aa4c697` (13 August 2026) begins with a request that carried a long system prompt, an image and a sound, and made the model write exactly one token, a control token, and stop. The prompt builder had put every medium at index 0, before the system turn, which for a short prompt is harmless and for a long one moves the role tags, which the model expects near the start, far from where it was trained to find them. The fix is `media_splice_point` in `mm.rs`: after the system turn's closing tag, and past the next turn's opening tag when that is known. What the code and its comments add to the message is the reason that the rest of the design is not time-aligned. A true inline placement, each medium expanded where the chat template put its single placeholder, was built, run on a real request on a quantised checkpoint, and measured to be worse than the whole-block splice (the model looped on an opening tag three times at the same length and with positions that had been checked), so the crate keeps the block and strips the inline placeholder run before tokenizing (`strip_media_placeholder_text`, whose own comment records that stripping at the token level instead changes how the neighbouring text is split by the tokenizer). It is one request on one checkpoint, and the comment says so. The general point stands without more: where a medium is spliced is part of what the model was trained to read, and the position of a block, like its rows, is a contract between the splice and the weights.

## One decoder that holds thirty billion parameters and uses three

The decoder behind the rows is the mixture of experts of Volume VI at a size that makes the earlier chapter's arithmetic concrete: 48 layers of width 2,048, 32 query heads and 4 key-value heads of 128, and in every layer a router that scores 128 experts of width 768 and sends the token to 8. The crate's `thinker.rs` builds this layer from the shared pieces of `model::block` and `model::moe`, with attention and normalisation exactly those of the dense decoder of Volume V and one new operation, the expert dispatch. Counting what the whole model holds is the first thing to do with its configuration, and a count has a check available that most such counts do not: the crate's import test records that the released checkpoint has 28,010 tensors.

The tensors come first, each with its shape, so that every count below can be derived and the tensor total checked group by group.

<!-- worked: omni_models.architecture_layouts -->
The tensors behind the next table, so that every parameter count and every tensor count in it can be re-derived. A shape lists outputs before inputs; a convolution is outputs x inputs x kernel; each copy is one tensor stored under its own name, so the experts of a layer are 128 gate, 128 up and 128 down tensors, and a weight and its bias are two tensors. A group's parameters are the product of its shape and its tensors. Where a bias is not listed the map has none.

Audio tower (the stem takes the 128-bin log-mel picture of a chunk of 100 frames to 13 rows of 7,680 numbers; 32 blocks of width 1,280, 20 heads, a feed-forward of 5,120; every linear map of the blocks and of the projector has a bias):

| tensor group | shape (outputs x inputs) | tensors | parameters |
|---|---|---:|---:|
| stem convolution 1 weight (stride 2, padding 1) | 480 x 1 x 3 x 3 | 1 | 4,320 |
| stem convolution 1 bias | 480 | 1 | 480 |
| stem convolution 2 weight | 480 x 480 x 3 x 3 | 1 | 2,073,600 |
| stem convolution 2 bias | 480 | 1 | 480 |
| stem convolution 3 weight | 480 x 480 x 3 x 3 | 1 | 2,073,600 |
| stem convolution 3 bias | 480 | 1 | 480 |
| stem output linear, 480 channels x 16 mel bins = 7,680 to 1,280 (no bias) | 1,280 x 7,680 | 1 | 9,830,400 |
| per block: attention q, k, v, out weights | 1,280 x 1,280 | 128 | 209,715,200 |
| per block: their four biases | 1,280 | 128 | 163,840 |
| per block: feed-forward up weight | 5,120 x 1,280 | 32 | 209,715,200 |
| per block: its bias | 5,120 | 32 | 163,840 |
| per block: feed-forward down weight | 1,280 x 5,120 | 32 | 209,715,200 |
| per block: its bias | 1,280 | 32 | 40,960 |
| per block: two LayerNorms, gain and bias each | 1,280 | 128 | 163,840 |
| final LayerNorm gain and bias | 1,280 | 2 | 2,560 |
| projector, first linear 1,280 to 1,280, weight | 1,280 x 1,280 | 1 | 1,638,400 |
| its bias | 1,280 | 1 | 1,280 |
| projector, second linear 1,280 to 2,048, weight | 2,048 x 1,280 | 1 | 2,621,440 |
| its bias | 2,048 | 1 | 2,048 |
| total |  | 525 | 647,927,168 |

Vision tower (27 blocks of width 1,152, 16 heads, a feed-forward of 4,304, fused query-key-value maps with bias):

| tensor group | shape (outputs x inputs) | tensors | parameters |
|---|---|---:|---:|
| patch embedding, a 3-D convolution over 3 channels, 2 frames, 16 x 16 pixels: weight | 1,152 x 3 x 2 x 16 x 16 | 1 | 1,769,472 |
| its bias | 1,152 | 1 | 1,152 |
| position table, a 48 x 48 grid of learned positions (resampled to the picture's grid) | 2,304 x 1,152 | 1 | 2,654,208 |
| per block: two LayerNorms, gain and bias each | 1,152 | 108 | 124,416 |
| per block: fused q, k, v weight | 3,456 x 1,152 | 27 | 107,495,424 |
| per block: its bias | 3,456 | 27 | 93,312 |
| per block: attention output weight | 1,152 x 1,152 | 27 | 35,831,808 |
| per block: its bias | 1,152 | 27 | 31,104 |
| per block: feed-forward up weight | 4,304 x 1,152 | 27 | 133,871,616 |
| per block: its bias | 4,304 | 27 | 116,208 |
| per block: feed-forward down weight | 1,152 x 4,304 | 27 | 133,871,616 |
| per block: its bias | 1,152 | 27 | 31,104 |
| main merger: LayerNorm over 1,152, gain and bias | 1,152 | 2 | 2,304 |
| main merger: first linear 4,608 to 4,608, weight | 4,608 x 4,608 | 1 | 21,233,664 |
| its bias | 4,608 | 1 | 4,608 |
| main merger: second linear 4,608 to 2,048, weight | 2,048 x 4,608 | 1 | 9,437,184 |
| its bias | 2,048 | 1 | 2,048 |
| three DeepStack mergers: LayerNorm over 4,608 (the four rows merged), gain and bias | 4,608 | 6 | 27,648 |
| three DeepStack mergers: first linear weight | 4,608 x 4,608 | 3 | 63,700,992 |
| their bias | 4,608 | 3 | 13,824 |
| three DeepStack mergers: second linear weight | 2,048 x 4,608 | 3 | 28,311,552 |
| their bias | 2,048 | 3 | 6,144 |
| total |  | 351 | 538,631,408 |

Thinker decoder (48 layers of width 2,048, 32 query and 4 key-value heads of 128, 128 experts of width 768; no biases):

| tensor group | shape (outputs x inputs) | tensors | parameters |
|---|---|---:|---:|
| token embedding table | 152,064 x 2,048 | 1 | 311,427,072 |
| output matrix | 152,064 x 2,048 | 1 | 311,427,072 |
| final RMSNorm gain | 2,048 | 1 | 2,048 |
| per layer: W_Q | 4,096 x 2,048 | 48 | 402,653,184 |
| per layer: W_K | 512 x 2,048 | 48 | 50,331,648 |
| per layer: W_V | 512 x 2,048 | 48 | 50,331,648 |
| per layer: W_O | 2,048 x 4,096 | 48 | 402,653,184 |
| per layer: query-norm and key-norm gains | 128 | 96 | 12,288 |
| per layer: input and post-attention RMSNorm gains | 2,048 | 96 | 196,608 |
| per layer: router | 128 x 2,048 | 48 | 12,582,912 |
| per layer: expert gate and up weights | 768 x 2,048 | 12,288 | 19,327,352,832 |
| per layer: expert down weights | 2,048 x 768 | 6,144 | 9,663,676,416 |
| total |  | 18,867 | 30,532,646,912 |

Talker (20 layers of width 1,024, 16 query and 2 key-value heads of 128, 128 experts of width 384 and a shared expert of width 768):

| tensor group | shape (outputs x inputs) | tensors | parameters |
|---|---|---:|---:|
| codec embedding table | 3,072 x 1,024 | 1 | 3,145,728 |
| codec head | 3,072 x 1,024 | 1 | 3,145,728 |
| final RMSNorm gain | 1,024 | 1 | 1,024 |
| per layer: W_Q | 2,048 x 1,024 | 20 | 41,943,040 |
| per layer: W_K | 256 x 1,024 | 20 | 5,242,880 |
| per layer: W_V | 256 x 1,024 | 20 | 5,242,880 |
| per layer: W_O | 1,024 x 2,048 | 20 | 41,943,040 |
| per layer: query-norm and key-norm gains | 128 | 40 | 5,120 |
| per layer: input and post-attention RMSNorm gains | 1,024 | 40 | 40,960 |
| per layer: router | 128 x 1,024 | 20 | 2,621,440 |
| per layer: expert gate and up weights | 384 x 1,024 | 5,120 | 2,013,265,920 |
| per layer: expert down weights | 1,024 x 384 | 2,560 | 1,006,632,960 |
| per layer: shared expert gate and up weights | 768 x 1,024 | 40 | 31,457,280 |
| per layer: shared expert down weight | 1,024 x 768 | 20 | 15,728,640 |
| per layer: shared expert's gate, one row | 1,024 | 20 | 20,480 |
| text_projection and hidden_projection: first linear 2,048 to 2,048, weights | 2,048 x 2,048 | 2 | 8,388,608 |
| their biases | 2,048 | 2 | 4,096 |
| the same two: second linear 2,048 to 1,024, weights | 1,024 x 2,048 | 2 | 4,194,304 |
| their biases | 1,024 | 2 | 2,048 |
| total |  | 7,951 | 3,183,026,176 |

Code predictor (5 layers of width 1,024, 16 query and 8 key-value heads of 128, a feed-forward of 3,072):

| tensor group | shape (outputs x inputs) | tensors | parameters |
|---|---|---:|---:|
| final RMSNorm gain | 1,024 | 1 | 1,024 |
| per layer: W_Q | 2,048 x 1,024 | 5 | 10,485,760 |
| per layer: W_K | 1,024 x 1,024 | 5 | 5,242,880 |
| per layer: W_V | 1,024 x 1,024 | 5 | 5,242,880 |
| per layer: W_O | 1,024 x 2,048 | 5 | 10,485,760 |
| per layer: query-norm and key-norm gains | 128 | 10 | 1,280 |
| per layer: input and post-attention RMSNorm gains | 1,024 | 10 | 10,240 |
| per layer: gate and up weights | 3,072 x 1,024 | 10 | 31,457,280 |
| per layer: down weight | 1,024 x 3,072 | 5 | 15,728,640 |
| codec embedding tables, one per code 1 to 15 | 2,048 x 1,024 | 15 | 31,457,280 |
| output heads, one per code 1 to 15 | 2,048 x 1,024 | 15 | 31,457,280 |
| total |  | 86 | 141,570,304 |

Code2Wav, first the code table and the transformer (8 layers of width 1,024, 16 heads of 64, a feed-forward of 3,072, no biases, two layer scales a layer):

| tensor group | shape (outputs x inputs) | tensors | parameters |
|---|---|---:|---:|
| code table, 16 codes x 2,048 values | 32,768 x 1,024 | 1 | 33,554,432 |
| per layer: q, k, v, out weights | 1,024 x 1,024 | 32 | 33,554,432 |
| per layer: gate and up weights | 3,072 x 1,024 | 16 | 50,331,648 |
| per layer: down weight | 1,024 x 3,072 | 8 | 25,165,824 |
| per layer: two RMSNorm gains | 1,024 | 16 | 16,384 |
| per layer: two layer-scale vectors | 1,024 | 16 | 16,384 |
| final RMSNorm gain | 1,024 | 1 | 1,024 |
| total |  | 90 | 142,640,128 |

and then the two upsampling stages and the SEANet decoder, which have the widths of Chapter 3's codec decoder. Every convolution has a bias, and a Snake activation has two vectors, alpha and beta, of one number per channel. The multiply-adds are per position of the layer's axis (a transposed convolution at its input positions), and the last column is how many such positions one code frame has there:

| layer | parameters, as a sum of tensors | parameters | tensors | multiply-adds a position | positions a frame |
|---|---|---:|---:|---:|---:|
| upsampling stage 1: transposed convolution (k = 2, stride 2) | 1,024 x 1,024 x 2 + 1,024 | 2,098,176 | 2 | 2,097,152 | 1 |
| upsampling stage 1: ConvNeXt block | depthwise 1,024 x 7 + 1,024, norm 2 x 1,024, 1,024 x 4,096 + 4,096, 4,096 x 1,024 + 1,024, scale 1,024 | 8,404,992 | 9 | 8,395,776 | 2 |
| upsampling stage 2: transposed convolution (k = 2, stride 2) | 1,024 x 1,024 x 2 + 1,024 | 2,098,176 | 2 | 2,097,152 | 2 |
| upsampling stage 2: ConvNeXt block | depthwise 1,024 x 7 + 1,024, norm 2 x 1,024, 1,024 x 4,096 + 4,096, 4,096 x 1,024 + 1,024, scale 1,024 | 8,404,992 | 9 | 8,395,776 | 4 |
| SEANet first convolution (k = 7) | 1,024 x 1,536 x 7 + 1,536 | 11,011,584 | 2 | 11,010,048 | 4 |
| SEANet block 1: Snake (alpha and beta) and transposed convolution (k = 16, stride 8) | 2 x 1,536 + 1,536 x 768 x 16 + 768 | 18,878,208 | 4 | 18,874,368 | 4 |
| SEANet block 1: three residual units (7-tap and 1-tap convolutions, two Snakes each) | 3 x (2 x 2 x 768 + 768^2 x 7 + 768 + 768^2 + 768) | 14,169,600 | 24 | 14,155,776 | 32 |
| SEANet block 2: Snake (alpha and beta) and transposed convolution (k = 10, stride 5) | 2 x 768 + 768 x 384 x 10 + 384 | 2,951,040 | 4 | 2,949,120 | 32 |
| SEANet block 2: three residual units (7-tap and 1-tap convolutions, two Snakes each) | 3 x (2 x 2 x 384 + 384^2 x 7 + 384 + 384^2 + 384) | 3,545,856 | 24 | 3,538,944 | 160 |
| SEANet block 3: Snake (alpha and beta) and transposed convolution (k = 8, stride 4) | 2 x 384 + 384 x 192 x 8 + 192 | 590,784 | 4 | 589,824 | 160 |
| SEANet block 3: three residual units (7-tap and 1-tap convolutions, two Snakes each) | 3 x (2 x 2 x 192 + 192^2 x 7 + 192 + 192^2 + 192) | 888,192 | 24 | 884,736 | 640 |
| SEANet block 4: Snake (alpha and beta) and transposed convolution (k = 6, stride 3) | 2 x 192 + 192 x 96 x 6 + 96 | 111,072 | 4 | 110,592 | 640 |
| SEANet block 4: three residual units (7-tap and 1-tap convolutions, two Snakes each) | 3 x (2 x 2 x 96 + 96^2 x 7 + 96 + 96^2 + 96) | 222,912 | 24 | 221,184 | 1,920 |
| SEANet tail: Snake and convolution (k = 7, 96 to 1) | 2 x 96 + 96 x 7 + 1 | 865 | 4 | 672 | 1,920 |
| total, after the transformer |  | 73,376,449 | 140 | 2,447,136,768 |  |
<!-- /worked -->

<!-- worked: omni_models.parameter_counts -->
Parameters counted from the tensors that the configuration and the code of the crate imply (weights and the biases that exist, norm gains, layer scales, the Snake parameters of the vocoder), with the number of tensors beside them. The Talker's two projections are the one assumption: their middle width is not a field the crate parses, and 2,048 is taken.

| component | tensors | parameters |
|---|---:|---:|
| Thinker: audio tower (conv stem, 32 blocks, projector) | 525 | 647,927,168 |
| Thinker: vision tower (patch embedding, 27 blocks, main merger) | 333 | 446,571,248 |
| Thinker: three DeepStack mergers | 18 | 92,060,160 |
| Thinker: embedding table and output matrix (152,064 x 2,048 each) | 2 | 622,854,144 |
| Thinker: 48 decoder layers and the final norm | 18,865 | 29,909,792,768 |
| Talker: 20 decoder layers and the final norm | 7,941 | 3,164,145,664 |
| Talker: codec embedding and codec head (3,072 x 1,024 each) | 2 | 6,291,456 |
| Talker: text and hidden projections (middle width assumed 2,048) | 8 | 12,589,056 |
| code predictor: 5 layers, 15 heads, 15 tables | 86 | 141,570,304 |
| Code2Wav: code embedding, 8-layer transformer | 90 | 142,640,128 |
| Code2Wav: upsampling stages and SEANet decoder | 140 | 73,376,449 |
| **the whole model** | 28,010 | 35,259,818,545 |

The tensors sum to 28,010, the number of names that the crate's import test records for the released checkpoint (28,010). The Thinker with its towers holds 31,719,205,488 parameters, the speech side (Talker, code predictor and Code2Wav) 3,540,613,057.
<!-- /worked -->

The tensors add to 28,010 exactly, which says that the list of what the model contains is right, and does not say that every shape is: a count of tensors cannot distinguish a matrix of 2,048 by 768 from one of 768 by 2,048, and the one number of the table that is an assumption, the middle width of the Talker's two projections, is in the row that marks it. No checkpoint is in this repository to compare the totals against. With that said, the model has 35.26 billion parameters; the Thinker's decoder is 29.9 billion of them, in its layers, and the whole Thinker with its two towers is 31.7. The speech side, the Talker, the code predictor and the code-to-wave decoder, is 3.54 billion, a tenth. And the three DeepStack mergers of the vision tower, 92 million parameters in three more copies of the merger that Volume VII's chapter described, are in the checkpoint and are not used: `vision_weights` in `mm.rs` skips every tensor whose name begins `deepstack_merger.` with a comment that the plain splice path does not need them. The checkpoint carries the taps because the model was built to inject them into the residual stream after its first layers, as Volume VII's chapter described for the Qwen3-VL decoder, and brain's Thinker has no injection point: a picture reaches it once, at the input.

What a token uses is a smaller number, and the router is what makes it so.

<!-- worked: omni_models.moe_layers -->
One decoder layer of the Thinker (width 2,048, experts of 768) and of the Talker (width 1,024, experts of 384, a shared expert of 768): parameters held, and parameters whose products one token takes part in.

| part | Thinker: held | Thinker: used | Talker: held | Talker: used |
|---|---:|---:|---:|---:|
| attention (q, k, v, o) | 18,874,368 | 18,874,368 | 4,718,592 | 4,718,592 |
| norms (two gains, two query-key norms) | 4,352 | 4,352 | 2,304 | 2,304 |
| router | 262,144 | 262,144 | 131,072 | 131,072 |
| routed experts | 128 x 4,718,592 = 603,979,776 | 8 x 4,718,592 = 37,748,736 | 128 x 1,179,648 = 150,994,944 | 6 x 1,179,648 = 7,077,888 |
| shared expert and its gate | 0 | 0 | 2,360,320 | 2,360,320 |
| **a layer** | 623,120,640 | 56,889,600 | 158,207,232 | 14,290,176 |
| held / used |  | 10.95 |  | 11.07 |
<!-- /worked -->

A layer of the Thinker holds 623,120,640 parameters and a token uses 56,889,600 of them, one eleventh; the attention is the same 18.9 million in both columns, and the 128 experts are all of the difference, of which a token meets one sixteenth. The Talker's layer has the shape of Volume VI's Qwen 3.5, one shared expert that every token uses and a sigmoid gate that scales it, plus 128 routed experts of which 6 are chosen: 158 million held, 14.3 million used, one eleventh again. The shared expert is the one place where the two decoders differ, and the code is the formula of that chapter, the routed experts' combination plus the shared expert's output times a sigmoid of a one-row gate (`crates/qwen3omnimoe/src/talker.rs`). Both routers are plain softmax over the experts with the chosen weights renormalised, as `config.rs` records with `norm_topk_prob`.

<!-- worked: omni_models.active_counts -->
Multiply-adds of one unit of each role, weights only, no attention scores except where the row says: the audio tower's one second and the vision tower's pair of frames are complete counts, with their windowed attention.

| unit | multiply-adds |
|---|---:|
| Thinker, one prompt row (attention, router, 8 of 128 experts; no head) | 2,730,491,904 |
| Thinker, one generated token (the same and the 152,064-way head) | 3,041,918,976 |
| audio tower, one second (13 rows) | 10,479,923,200 |
| vision tower, one pair of 448 x 448 frames (196 rows) | 367,914,516,480 |
| Talker, one frame (20 layers with 6 of 128 experts and the shared expert, and the codec head) | 288,903,168 |
| code predictor, one frame (16 steps of 5 layers, 15 heads) | 1,289,748,480 |
| Code2Wav, one frame | 2,556,188,672 |

A frame of speech is 4,134,840,320 multiply-adds in all, 1.36 times the generated token of the Thinker, and a second of speech is 12.5 of them, 51.69 billion. The Thinker's decoder holds 29,909,792,768 parameters in its layers, 10.95 times what a row uses; with the matrices of the table it holds 30,532,646,912, and a generated token uses 3,041,918,976 of them, 10.04 times fewer.
<!-- /worked -->

The first rows are the 'A3B' of the name: a generated token passes through 3,041,918,976 multiply-adds of weights (the 48 layers' attention, routers and 8 experts, then the 152,064-way head), 2.73 billion before the head is applied, and the decoder's layers hold 10.95 times that. The last rows are the point that a reader who knows only the first would miss. A frame of speech is not cheap. It is 4.13 billion multiply-adds, 1.36 times the Thinker's generated token, because the Talker's 289 million are a small part of it: the code predictor's 16 steps are 1.29 billion, as Chapter 6 found for the Qwen3-TTS pair, and the code-to-wave decoder is 2.56 billion, the largest single part. A second of speech is 12.5 frames and 51.7 billion, as much as 17 generated tokens of the Thinker, and the sparse decoder is not where speech output spends its work.

## Why the voice is a second model

The Thinker could, in principle, write the speech codes itself: give its vocabulary 2,048 more entries and let it write frames after its words. The model does not do that, and the **thinker-talker split** is the choice of two decoders with different jobs and a defined interface: the Thinker reads everything and writes the reply as text; the Talker reads what the Thinker made of the request and what it wrote, and writes the speech as codec frames. The two have their own weights, their own experts, their own vocabularies (152,064 text tokens against 3,072 codec ids), their own caches and their own clocks, one token at a time in the first and one frame, 80 milliseconds, at a time in the second. A reason one can give for the separation, that speech training does not disturb what the Thinker knows about the world, is a statement about training, and brain only runs the released checkpoint and cannot show it. What the code shows is how the interface is built, and that is the substance of the split.

What crosses is mostly words. The Talker's prompt is made, row by row, from two kinds of source, and the crate's `talker_prompt.rs` assembles it for a user turn and the reply. A row of the user turn that is plain text is the Thinker's embedding of the token, passed through a small two-layer network, `text_projection`, from 2,048 numbers to the Talker's 1,024. A row of the user turn that is a spliced medium, a sound or a picture, is not an embedding at all: it is the Thinker's hidden state at that row after its layer 24, counting from 0, passed through a second such network, `hidden_projection`. That is the only place where something the Thinker computed, and not something it was given or wrote, goes to the Talker, and the reply itself crosses as tokens, which are projected from their embeddings and not from any state.

<!-- worked: omni_models.talker_prompt_rows -->
The prefill of the Talker for a user turn of 6 tokens (positions 2 to 4 are the spliced media rows) and a reply of 7 text tokens a0 to a6. Each row is the sum of a text side and a codec side; `text_projection` and `hidden_projection` each map 2,048 numbers to the Talker's 1,024. `pad`, `bos` and `eos` are the special text ids whose embeddings go through `text_projection`.

| row | segment | text side | codec side |
|---:|---|---|---|
| 0 | user turn | text_projection(thinker embedding of user token 0) | - |
| 1 | user turn | text_projection(thinker embedding of user token 1) | - |
| 2 | user turn | hidden_projection(thinker hidden at layer 24, position 2) | - |
| 3 | user turn | hidden_projection(thinker hidden at layer 24, position 3) | - |
| 4 | user turn | hidden_projection(thinker hidden at layer 24, position 4) | - |
| 5 | user turn | text_projection(thinker embedding of user token 5) | - |
| 6 | assistant segment | text_projection(embedding of a0) | 0 |
| 7 | assistant segment | text_projection(embedding of a1) | 0 |
| 8 | assistant segment | text_projection(embedding of a2) | 0 |
| 9 | assistant segment | text_projection(embedding of tts_pad) | codec_embedding(nothink) |
| 10 | assistant segment | text_projection(embedding of tts_pad) | codec_embedding(think_bos) |
| 11 | assistant segment | text_projection(embedding of tts_pad) | codec_embedding(think_eos) |
| 12 | assistant segment | text_projection(embedding of tts_pad) | codec_embedding(speaker) |
| 13 | assistant segment | text_projection(embedding of tts_bos) | codec_embedding(codec_pad) |
| 14 | assistant segment | text_projection(embedding of a3) | codec_embedding(codec_bos) |

After the prefill (15 rows), every frame s that the Talker writes is fed back as one row: the sum of the 16 codec embeddings of the frame and one more row, the one in the table (the text of the reply reaches the Talker at one token a frame, then the end marker, then padding).

| frame s | text row added to the frame's codes |
|---:|---|
| 0 | text_projection(embedding of a4) |
| 1 | text_projection(embedding of a5) |
| 2 | text_projection(embedding of a6) |
| 3 | text_projection(embedding of tts_eos) |
| 4 | text_projection(embedding of tts_pad) |
| 5 | text_projection(embedding of tts_pad) |
| 6 | text_projection(embedding of tts_pad) |
| 7 | text_projection(embedding of tts_pad) |

The reply has 7 tokens, so 4 rows trail the prefill (3 tokens and the end marker); a reply of fewer than 4 tokens cannot fill the template.
<!-- /worked -->

The assistant segment is a fixed template, which the source comments call copied from the reference bit for bit: nine rows. The first three carry only the projected text of the reply's first three tokens. The last six add a text row to a codec row: four rows of the padding text id over three tags (no language is being thought about) and the voice, then the begin-of-speech text id over a codec padding row, and the reply's fourth token over the codec's begin-of-speech row. The voice is one of three speaker ids, 2301, 2302 and 2303, the names the configuration records. The shape explains the rest. After the prefill the Talker writes one frame at a time, and each frame it has written is fed back as the sum of its 16 codec embeddings plus one more row, which for the first frames is the next token of the reply, then the end marker, then padding: the reply's words arrive at the Talker at one for each frame of speech, 12.5 a second. The Talker decides when to stop by writing the codec end token, as in Chapter 6, and nothing forces the speech to be as long as the text calls for. A reply of fewer than four tokens cannot fill the template, and the function asserts it.

Two facts follow from the interface, and each has a cost that can be counted. First, a Talker that needs only the first four tokens of the reply, and one token a frame after that, could start long before the Thinker is done; the interface supports streaming, which is the subject of a later section. Second, a user turn that has media costs the Thinker a second pass. The hidden states at layer 24 of the user rows are not kept when the Thinker generates, because decoding computes one new row at a time, and so `OmniInner::converse` in `crates/qwen3omnimoe/src/caps.rs` runs the prompt through the first 25 layers again, teacher-forced, and reads them off: 25 of the 48 layers' products, 52.1 percent of the prefill's weights, again. The `speak` action, which takes text only, does not pay it and takes the text branch for every row of the user turn; commit `62ad38f9d` (11 August 2026) added `converse` because, as its message says, a spoken reply to a spoken question had the Talker's conditioning ignoring the question's sound and picture (the Thinker's text was shaped by the media, the Talker's voice was not), and the new action routes media positions through `hidden_projection`, with a test that an all-text mask reproduces the old output exactly.

A smaller commit shows how the Talker was built. `5121cfbcb` (7 August 2026) changed one default: the Talker's configuration has no key for query-key normalisation, the crate had defaulted it to off, and the reference's attention class, which the Talker reuses, applies the two norms with no flag at all, and the checkpoint carries their weights for every layer. A configuration that does not mention a feature is not a configuration that disables it, and the commit message says it was found by reading the attention class before any test ran.

## From sixteen codes to a waveform

Frame $s$ of the speech is 16 integers: code 0 from the Talker, written from a vocabulary of 3,072 of which the top 1,024 are masked except the end token (the suppression window of Chapter 6, which leaves 2,048 usable ids), and codes 1 to 15 from the code predictor, a 5-layer decoder with the shape and the key-value cache of Chapter 6's, which `crates/qwen3omnimoe` borrows from `qwen3tts` unchanged and fills from `talker.code_predictor.*`. Speech at 12.5 frames a second is 200 integers, 2,200 bits.

The decoder that turns frames into samples is called Code2Wav in the configuration, and it is the codec decoder of Chapter 3 with one part replaced. Chapter 3's decoder dequantised each of the 16 codes through its own table of 256 numbers and two output projections, and ran a pre-convolution and a pre-transformer of width 512. Here the 16 codes of a frame index one table of 32,768 rows of 1,024 numbers (code $q$ looks up the row numbered $q \cdot 2048$ plus its value; the configuration gives the semantic codebook a size of 4,096, but the Talker's masking keeps code 0 below 2,048 for every frame it writes, so its lookups stay inside its own slice of 2,048 rows and the table has exactly $16 \cdot 2048 = 32{,}768$ rows) and the 16 rows are averaged, a mean and not a sum, straight into the working width of 1,024, with no projections. An 8-layer transformer of width 1,024 and a feed-forward of 3,072 follows, causal with a sliding window of 72 frames, which is 5.76 seconds; commit `0893002c3` found the window parsed and never applied, the story of Chapter 3. Two upsampling stages by 2 and the SEANet decoder of Chapter 3, with rates 8, 5, 4 and 3 and the same widths, complete it (`crates/mimi/src/model.rs`, `decode_omni`). The total factor is $2 \cdot 2 \cdot 8 \cdot 5 \cdot 4 \cdot 3 = 1920$, 24,000 samples over 12.5 frames, and the arithmetic of the lengths has one detail that surprised the crate's author.

<!-- worked: omni_models.code_to_wave -->
Positions along the time axis for 8 code frames: the two upsampling stages (each by 2) turn 8 into 32, and each of the four SEANet blocks upsamples by its rate $\mathsf{u}_j$ in 8, 5, 4, 3 with a transposed convolution of kernel $2\mathsf{u}_j$ whose output is cropped by $\mathsf{u}_j$ on both sides, $L \mapsto (L - 1)\,\mathsf{u}_j$.

| stage | length |
|---|---:|
| codes [8, 16] -> mean of 16 embeddings -> transformer -> after the two upsampling stages | 32 |
| after block 1 (x8, crop) | 248 |
| after block 2 (x5, crop) | 1,235 |
| after block 3 (x4, crop) | 4,936 |
| after block 4 (x3, crop) | 14,805 |

Samples for T frames: the product 1,920 T that the total upsampling factor suggests, and the length that the stages give.

| frames T | 1,920 T | samples | missing | seconds at 24,000 Hz |
|---|---:|---:|---:|---:|
| 1 | 1,920 | 1,365 | 555 | 0.057 |
| 2 | 3,840 | 3,285 | 555 | 0.137 |
| 8 | 15,360 | 14,805 | 555 | 0.617 |
| 38 | 72,960 | 72,405 | 555 | 3.017 |
| 155 | 297,600 | 297,045 | 555 | 12.377 |
| 256 | 491,520 | 490,965 | 555 | 20.457 |
| 300 | 576,000 | 575,445 | 555 | 23.977 |
<!-- /worked -->

The decoder does not write $1920 T$ samples but $1920 T - 555$. A transposed convolution of kernel $2\mathsf{u}_j$ and stride $\mathsf{u}_j$ in the reference crops its padding from both sides, so a stage maps a length $L$ to $(L-1)\,\mathsf{u}_j$, not to $L\,\mathsf{u}_j$, where Chapter 3's codec decoder crops only the right. The length follows by composing the four stages after the two doublings:

```text
function wave_length(T: int, ups: list<int>, rates: list<int>) -> int
    n = T
    for u in ups
        n = n * u
    for r in rates
        n = (n - 1) * r
    return n
```

With `ups` = [2, 2] and `rates` = [8, 5, 4, 3] this is $1920 T - 555$ for every $T$: 555 samples, 23 milliseconds, fewer than the total factor suggests, however long the speech. The crate found this the usual way, from a reference output of a different length: the `decode_omni` documentation says that a golden of 14,805 samples for 8 frames disagreed with the 15,360 that $L \cdot \mathrm{rate}$ gave, and the mismatch is what exposed the second convention. The table's row for 155 frames, 297,045 samples, is the length that commit `528cbde5f` (8 August 2026) reports for the first end-to-end run of the whole speech chain that completed, on a reply to "Say hello.": 12.38 seconds at 24 kHz.

<!-- worked: omni_models.code2wav_work -->
Multiply-adds of the code-to-wave decoder per code frame: the multiply-adds a position of each layer, as in the layout above (the transformer's weights are counted per frame, with no attention scores), the positions that one frame has on that layer's axis, and their product in millions. A frame is 1 position at the transformer, 2 and then 4 at the upsampling stages, and 1,920 at the last SEANet block, so the SEANet decoder works at the sample rate. Chapter 3 counts the same layers per second, at 12.5 frames a second; the figures here are those divided by 12.5 and are derived above from the shapes.

| layer | multiply-adds a position | positions a frame | millions a frame |
|---|---:|---:|---:|
| mean of 16 embeddings (additions, not counted) |  |  | 0 |
| 8-layer transformer, weights: 8 x (4 x 1,024^2 + 3 x 1,024 x 3,072) | 109,051,904 | 1 | 109.1 |
| upsampling stage 1: transposed convolution (k = 2, stride 2) | 2,097,152 | 1 | 2.1 |
| upsampling stage 1: ConvNeXt block | 8,395,776 | 2 | 16.8 |
| upsampling stage 2: transposed convolution (k = 2, stride 2) | 2,097,152 | 2 | 4.2 |
| upsampling stage 2: ConvNeXt block | 8,395,776 | 4 | 33.6 |
| SEANet first convolution (k = 7) | 11,010,048 | 4 | 44.0 |
| SEANet block 1: Snake (alpha and beta) and transposed convolution (k = 16, stride 8) | 18,874,368 | 4 | 75.5 |
| SEANet block 1: three residual units (7-tap and 1-tap convolutions, two Snakes each) | 14,155,776 | 32 | 453.0 |
| SEANet block 2: Snake (alpha and beta) and transposed convolution (k = 10, stride 5) | 2,949,120 | 32 | 94.4 |
| SEANet block 2: three residual units (7-tap and 1-tap convolutions, two Snakes each) | 3,538,944 | 160 | 566.2 |
| SEANet block 3: Snake (alpha and beta) and transposed convolution (k = 8, stride 4) | 589,824 | 160 | 94.4 |
| SEANet block 3: three residual units (7-tap and 1-tap convolutions, two Snakes each) | 884,736 | 640 | 566.2 |
| SEANet block 4: Snake (alpha and beta) and transposed convolution (k = 6, stride 3) | 110,592 | 640 | 70.8 |
| SEANet block 4: three residual units (7-tap and 1-tap convolutions, two Snakes each) | 221,184 | 1,920 | 424.7 |
| SEANet tail: Snake and convolution (k = 7, 96 to 1) | 672 | 1,920 | 1.3 |
| **one frame** |  |  | 2,556.2 |
| one second (12.5 frames) |  |  | 31,952.4 |
<!-- /worked -->

A frame costs the code-to-wave decoder 2.56 billion multiply-adds, the four SEANet blocks 2.35 billion of them, which is why the speech side of the previous section is not cheap: the sparse Talker that was the point of the architecture is 289 million of a frame's 4,135. The hardware-independent statement of what makes speech expensive is that the output waveform is 1,920 samples for every frame and the decoder works at the sample rate in its last blocks, in channels of 192 and 96.

## When the first sound can come out

A listener wants the reply to begin while it is still being written. Whether a model can do that depends on what each stage needs from the one before, and the interfaces of the previous section say a lot. The Talker needs the first four tokens of the reply and then one token a frame. The code predictor needs only the Talker's frame. The code-to-wave transformer is causal with a window of 72 frames, so the samples of a prefix of the frames are the first samples of the whole, and each block after it keeps a small state between chunks. Nothing in the chain needs the end of the reply before it can start. The structure therefore allows **streaming speech output**: speech that begins to leave the model while the text it says is still being written and while its own later frames are still being drawn, so that the work before the first sound is the work of a prefix and not of the whole reply.

Brain's crate runs the stages in sequence, and the commit that added streaming says so. `OmniInner::speak` and `converse` call `generate` or the multimodal generator to the end of the reply, then `generate_codes` to the end of the Talker's frames (capped at 256, which is 20.48 seconds), then `decode_omni_chunked_until`, which runs the transformer over every frame once and then the upsampling and SEANet back in chunks of 300 frames, 24 seconds, calling the callback with each and stopping when the request is cancelled. A reply shorter than 300 frames is therefore one chunk, and the first chunk arrives when the frames are all written. `generate_codes` is a thin wrapper over `generate_codes_streaming`, whose per-frame callback is the seam at which a vocoder could start early. Commit `8a5424b4e` (11 August 2026) and the commit that made the callback (`149ca7466`, 11 August 2026) both describe this order: speak and converse generate all codes first, so time to first audio is bounded by the whole generation. The message of the second also says that the end-to-end test was extended to assert that the callback "fires more than once"; the test the commit contains asserts `n_chunks > 0`, which is the strongest assertion that can hold for a reply of 155 frames, the length of the first real run, and a chunk of 300. The message claimed more than the code checked, and the code is right about what it can check.

What this costs is a count of work, not of time.

<!-- worked: omni_models.first_audio_work -->
Multiply-adds, in billions, that must be done before the first sample of the reply exists, for a prompt of 40 rows, a user turn of 20 tokens for the Talker, a reply of n text tokens and F code frames. Brain's order runs the Thinker to the end of the reply, the Talker to the end of its frames and the code-to-wave front over all frames, and hands out audio in chunks of up to 300 frames, so for F up to 300 the first chunk is the whole reply. The order the interfaces allow starts the Talker when the reply has 4 tokens and hands out the first chunk after 8 frames (weights only; attention scores and the 25-layer second pass of a conversation are left out). The count is by these rules. The Thinker's prefill of the prompt gives the reply's first token, so it costs the 40 rows through the 48 layers and the head once, and each of the n - 1 further tokens one row through the layers and the head; '4 tokens' is the prefill and 3 further steps. The Talker's prefill is the user turn's 20 rows and the template's 9, 29 rows, each through its 20 layers and the codec head (the crate computes the logits of every prefill row and reads the last); then each of the F frames costs a Talker step and the code predictor's 16 steps. The code-to-wave transformer costs its weights for each of the F frames, and the upsampling and SEANet back its per-frame count for each frame of the first chunk, the smaller of 300 and F frames.

| reply tokens n | code frames F | brain's order | first chunk of 8 frames | ratio |
|---|---:|---:|---:|---:|
| 8 | 20 | 221.9 | 160.1 | 1.4 |
| 40 | 100 | 650.0 | 160.1 | 4.1 |
| 80 | 200 | 1,185.2 | 160.1 | 7.4 |
| 160 | 256 | 1,660.1 | 160.1 | 10.4 |

Where those multiply-adds of brain's order go for n = 40 and F = 100 (in billions).

| stage | billions | percent |
|---|---:|---:|
| Thinker: prefill of the prompt | 109.5 | 16.9 |
| Thinker: the rest of 40 reply tokens | 118.6 | 18.3 |
| Talker: prefill (29 rows) | 8.4 | 1.3 |
| Talker and code predictor: 100 frames | 157.9 | 24.3 |
| code-to-wave: transformer over 100 frames | 10.9 | 1.7 |
| code-to-wave: upsampling and SEANet over the first chunk (100 frames) | 244.7 | 37.6 |
| **total** | 650.0 | 100.0 |
<!-- /worked -->

In brain's order the work before the first sample is the whole reply's: for a reply of 40 tokens spoken in 100 frames it is 650 billion multiply-adds, and the largest part is not the Thinker but the vocoder's back, 37.6 percent, because the first chunk is all 100 frames. A path that began when the reply had four tokens and handed out its first 8 frames would need 160 billion, of which 109.5 are the prompt through the Thinker, which no streaming removes: the ratio is 4.1 at this reply length and 10.4 at the longest the Talker can write, and it grows with the reply because brain's order has the whole reply in front of the first sound and the streaming order has a prefix. That is why the prefill of the first section, 88 trillion multiply-adds for the clip, is the floor of an omni model on video: no ordering of the speech side can start before it.

There is a second cost in the same path that has nothing to do with the order of the stages. The Talker, in this crate, is run in the way the crate's documentation calls validation tier: for every prefill and every decoded frame `talker_generate.rs` reads all of the Talker's weights again from the checkpoint, uploads them as 32-bit numbers and runs the step.

<!-- worked: omni_models.talker_stream_bytes -->
Bytes that the Talker's weights move in brain's validation-tier path, where every step re-reads all 20 layers of its 128 experts each from the checkpoint (32-bit values), against the bytes that the step's products use. Both count what the `moe_layers` table counts, norms included: a layer holds 158,207,232 parameters and uses 14,290,176.

| quantity | bytes |
|---|---:|
| every weight of the 20 layers uploaded for one step (32-bit) | 12,656,578,560 |
| the weights that step's products use (32-bit) | 1,143,214,080 |
| held / used | 11.07 |
| prefill and 100 frames: 101 steps | 1,278,314,434,560 |
| the same in gibibytes | 1,190.5 |
<!-- /worked -->

A step uses 1.14 billion bytes of weights and reads 12.66 billion, 11.07 times as many, because the 128 experts are all uploaded and six are used: the mixture-of-experts ratio of the earlier table, paid in bytes moved instead of operations. A reply of 100 frames reads 1,190 GiB of weights for the Talker alone. The Thinker is better placed: `generate.rs` and `thinker_plan.rs` keep as many layers resident across the cards as fit and stream only the rest, an int8 resident Thinker exists in `int8_thinker_resident.rs`, and int8 kernels for the Talker's experts exist with a parity test; but the int8 Thinker serves text and does not speak, and `speak` calls the Talker with no int8 store. In the terms of Volume IV, Chapter 3, the step moves the bytes of all 128 experts for the products of 6, so its arithmetic intensity, already low for a matrix-vector product, is eleven times lower again.

## What the crate runs

The crate runs the whole chain: sound, pictures and video in, text and speech out, behind three actions, `generate` (text, with optional media), `speak` (text in, text and speech out) and `converse` (media in, text and speech out). The first is exposed on the chat endpoints; the second and third are not, because those endpoints key on the action's name. Its tests compare each piece with the released checkpoint's outputs where the checkpoint is available, among them the layer-by-layer parity of the Thinker and the Talker, the code predictor, Code2Wav and the two towers, and an end-to-end text and speech run with an assertion that the waveform is finite and not silent, a check that says the chain runs and nothing about whether the speech is good.

The Thinker's text is generated by one decode loop (`crates/qwen3omnimoe/src/decode.rs`) that serves every stack, bf16 or int8, through the sampler that every chat engine of brain shares, over the whole row of logits, so that `temp`, `top_k`, `top_p` and the penalties are exact: a temperature of 0 is the argmax, and a seed reproduces a sampled reply. `crates/qwen3omnimoe/src/chat.rs` serves a request the way the other Qwen engines do (commit `c1f2cc0b8`, 13 November 2026): the checkpoint's chat template with tools and thinking, the shared stop-string matching, a scanner that returns structured tool calls and separated reasoning, `tool_choice` enforced, streaming deltas, and cancellation. The Talker's frames are sampled under a seed that the request supplies and the outcome reports; two commits of October 2026 made that true: `9ad7099dd` found that the sampled frames had always run on seed 0, and `d2634ffb2` that only code 0 of each frame used the seed, the other fifteen being drawn greedily. The vision tower's three DeepStack taps are added to the visual rows after the Thinker's first three layers during prefill (decode steps are text only), as in the reference, and the work table of the minute counts their mergers. `speak` and `converse` honour cancellation in the Thinker's decode, in the Talker's frame loop and between the chunks of Code2Wav, and a cancelled reply keeps the audio already produced. A prompt carries one clip of sound, one picture and one video at most, each spliced as one block at one point, and the Talker's prefill and decode use plain sequential positions. The request's earlier turns are not kept: all three actions are single-turn.

The crate's documentation states the same scope: `thinker.rs` is forward-only on a single device, with the int8 expert path and the KV-cached `layer_decode_step`; `caps.rs` names the `generate` action a "KV-cached decode" in its specification string; and the talker's multi-token predictor is `qwen3tts::mtp`, configured from the checkpoint's `code_predictor_config`.

## Check your understanding

1. A reader says that the sound is a third of the work of the omni model because it is one of three inputs. Using the minute of the chapter, say what the shares are and what they depend on.
2. In brain's frame-ordinal positions, a sound row and a video group of the same second are at different positions, and in the time-aligned positions they are at the same position. Does the decoder in the first case lose the information that they are simultaneous? Say what it can and cannot recover.
3. A commit message says that a test asserts that a callback "fires more than once", and the test asserts `n_chunks > 0`. Give the reply length at which the stronger assertion could be true and say why it was not true in the real run the commit cites.
4. The 28,010 tensors of the count are matched exactly. A reader concludes that every parameter count in the table is right. Find the gap and give an example of a count that would still match if it were wrong.
5. The Talker writes the speech at 12.5 frames a second and is fed one token of the reply a frame. A reply has 6 tokens, a0 to a5. What text row is added to the codes of frames 0, 1, 2, 3 and 7, and what stops the speech?
6. A frame of speech costs 1.36 times a generated token of the Thinker although the Talker uses 289 million multiply-adds and the Thinker 3.04 billion. Explain.
7. Someone proposes to remove the second Thinker pass of `converse` by caching the layer-24 states while generating. Which rows' states would be in the cache, which would be missing, and does the Talker need the missing ones?

## Summary

An omni model joins parts that earlier chapters built, by choosing how the rows of the media are ordered and positioned and what crosses from the decoder that reads to the decoder that speaks. In brain's crate, sound becomes 13 rows a second through a wider copy of the recogniser's tower, a 448 by 448 picture or pair of video frames becomes 196 through the vision tower and its merger, text one row a token, and a decoder of 48 layers, 128 experts and 8 used reads them as one sequence: a minute of video with sound is 12,594 rows and 89 trillion multiply-adds before the first token of the reply, 93 percent of the rows and 26 percent of the work from the video, 35 percent from attention. The positions that the model's configuration points to put the sound and the picture of the same second at the same place on a clock of 13 positions a second; brain's crate counts frames and places the video after the sound in a sequence that holds both. The whole model is 35.26 billion parameters, 28,010 tensors, of which a token uses 3.04 billion; the three vision taps of the design are in the checkpoint and are added to the visual rows after the Thinker's first three layers. A second decoder, the Talker, with 20 layers and 6 of 128 experts used, writes one of the 16 codes of each frame of speech from a prompt of projected tokens and, at media rows, the Thinker's states after layer 24; a 5-layer code predictor writes the other 15, and the code-to-wave decoder, a transformer, two upsampling stages and four SEANet blocks, turns frames into $1920 T - 555$ samples at 2.56 billion multiply-adds a frame. Brain runs these in sequence, so the first sound comes when the reply is written, and moves 11 times the Talker's used weights through the device on every frame; the structure of the model allows the first sound after four tokens and a handful of frames. The next chapter is the project that closes the volume: a pipeline from a waveform to a learned representation and back to generated audio, built from the parts of the eight chapters that came before.

## References

- Qwen Team, *Qwen3-Omni Technical Report*, 2025.
- Qwen Team, *Qwen2.5-Omni Technical Report*, 2025 (the thinker and talker design and the time-aligned positions).
- Qwen Team, *Qwen3-VL Technical Report*, 2025 (the vision tower and the DeepStack taps).
- Su, Lu, Pan, Murtadha, Wen and Liu, *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021.
- Défossez, Copet, Synnaeve and Adi, *High Fidelity Neural Audio Compression*, 2022.
- Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, 2017.

## Exercises

1. **[Calculate]** A clip of sound lasts 37.5 seconds at 16 kHz. How many mel frames, how many chunks (full and partial) and how many rows does the audio tower produce?
2. **[Calculate]** A clip of 10 seconds with its sound, video at 2 frames a second at 1080 by 1920, with no text and the four markers. How many positions does the prefill have and how many bytes does its cache hold at 196,608 bytes a position?
3. **[Calculate]** For a clip of 6 seconds with a grid of 3 by 2 merged rows, a text prefix of 10 tokens, sound at 13 rows a second and video at one group a second, list the segments of the time-aligned rule (first index, rows, $T$, $H$, $W$) and the counter after the clip. Then give the counter after the clip under brain's rule.
4. **[Calculate]** The Talker's router is changed to send each token to 8 experts instead of 6. How many parameters does a layer use, and how many multiply-adds does one frame of the Talker (not the code predictor) cost?
5. **[Calculate]** How many samples and how many seconds at 24,000 Hz does Code2Wav write for 100 frames, and how many are missing against $1920 T$?
6. **[Calculate]** A user turn of 11 tokens, none of them media, and a reply of 25 tokens. How many rows does the Talker's prefill have, how many rows trail it, and at which frame does the padding row start?
7. **[Implement]** Reproduce `omni_models.rows_per_medium` and `omni_models.minute_of_video` from the first section: the rows of a clip of sound and of a picture, and the stages of the minute's multiply-adds.
8. **[Implement]** Reproduce `omni_models.interleave_toy` and `omni_models.position_clocks` from `clip_positions` and from the description of brain's rule.
9. **[Implement]** Reproduce `omni_models.parameter_counts`, `omni_models.moe_layers` and `omni_models.active_counts` from the configuration numbers in the chapter, and check the tensor total against 28,010.
10. **[Implement]** Reproduce `omni_models.talker_prompt_rows` and `omni_models.code_to_wave` from the descriptions in the sections on the Talker and the code-to-wave decoder.
11. **[Investigate]** In `omni_models.first_audio_work`, take a reply of 8 tokens and a user turn of 20 tokens. For which number of frames $F$ does the upsampling and SEANet back become the largest single part of brain's order, and which part was the largest before? What does that say about where a first change to the order of the stages should go?
12. **[Investigate]** For a clip with sound only, show that the time-aligned rule and brain's rule give the same positions. For a clip with video only, say how the two rules differ on the $T$ axis and what that does to the angle that a rotary channel of a given frequency turns between two groups.

## Key terms

- **omni model**: one network whose decoder reads rows from the encoders of several media (sound, pictures, video and text) in a single sequence, and that answers in more than one medium, here in text and in speech. <!-- concept: omni-model -->
- **thinker-talker split**: the division of an omni model's decoding into a Thinker, which reads the whole request and writes the reply as text, and a Talker, which reads the reply's words and the Thinker's states at the media rows and writes the speech as codec frames, with its own weights, vocabulary and clock. <!-- concept: thinker-talker-split -->
- **streaming speech output**: speech that begins to leave the model while the text it says is still being written and its own later frames are still being drawn, so that the work before the first sound is that of a prefix of the reply and not of the whole reply. <!-- concept: streaming-speech-output -->

## Answers

**Recall questions.**

1. 200 mel frames, two chunks of 100, each 13 rows after three halvings that round up (100, 50, 25, 13): 26 rows.
2. The picture is 28 by 28 patches, 784, and merging 2 by 2 blocks gives 14 by 14 = 196 rows.
3. The row at frame $t$, grid row $h$ and column $w$ gets $(c + t, c + h, c + w)$, so the four rows get $(c, c, c)$, $(c, c, c+1)$, $(c, c+1, c)$ and $(c, c+1, c+1)$, and the counter after the block is $c + \max(1, 2, 2) = c + 2$.
4. One position per code would make the sequence 16 times longer and the attention cost grow with its square; the Talker adds the 16 rows into one. The code predictor, a 5-layer decoder, writes codes 1 to 15.
5. $8/128 = 1/16$. The router softmaxes the scores over the experts, keeps the 8 largest and renormalises them to add to 1; those are the combination weights.
6. 200 integers, and 2,200 bits at 11 bits a code.

**Check your understanding.**

1. The audio tower and the sound's rows are 6.2 percent of the sequence and 0.7 percent of the multiply-adds, the video 93.4 percent of the rows and about a quarter of the work in its tower, with the attention of the Thinker, which grows with the square of the whole sequence, another 35 percent. The shares depend on the frame rate and the picture size, not on the number of media.
2. The difference between a sound row's position and the video group's $T$ of the same second is $780 - 12 \cdot s$ for second $s$ of the minute, not 0 and not constant, and it depends on the length of the sound block, so the decoder cannot read 'the same moment' off a distance. It could learn a rule that maps the one to the other for clips of one length, but that would be a rule the position scheme does not give it, and the configuration points to the other scheme.
3. Only a reply of more than 300 frames, 24 seconds, can give two chunks; the real run was 155 frames, 12.4 seconds, one chunk.
4. A tensor count cannot see shapes. A matrix of 2,048 by 768 and one of 768 by 2,048 have the same count; so does the Talker's projection whatever its middle width, which is why that row is marked as an assumption.
5. The prefill used a0 to a3, so the trailing rows are a4, a5 and the end marker: frame 0 adds a4, frame 1 adds a5, frame 2 adds the end marker, and frames 3 and 7 add the padding row. The speech stops when the Talker writes the codec end token (allowed once at least 2 frames exist), or at the 256-frame cap.
6. Because the code predictor runs 16 steps of 5 layers and the code-to-wave decoder works at the sample rate: 1.29 and 2.56 billion against the Talker's 0.29.
7. The cache holds keys and values, not hidden states, so it has no layer-24 states at all; what would be cached is a copy of the prefill's layer-24 output, a state for each row of the prompt. The reply's rows are computed one at a time during decoding and are missing, but the Talker takes states only for the media rows of the user turn, and those are all in the prefill. A copy made during the prefill would be enough, and the second pass could go.

**Exercises.**

1. 3,750 frames; 37 full chunks of 100 and a partial one of 50; $37 \cdot 13 + 7 = 488$ rows (a chunk of 50 frames goes 25, 13, 7).
2. A 1080 by 1920 picture is 2,040 rows, 10 pairs make 20,400 rows, the sound is 130, the markers 4: 20,534 positions, 4,037,148,672 bytes, 3.76 GiB.
3. Segments (first index, rows, $T$, $H$, $W$): group 0: 10, 6, 10, 10 to 12, 10 to 11; group 1: 16, 6, 23, 10 to 12, 10 to 11; sound rows 0 to 25: 22, 26, 10 to 35 on every axis; group 2: 48, 6, 36, 10 to 12, 10 to 11; group 3: 54, 6, 49, 10 to 12, 10 to 11; sound rows 26 to 51: 60, 26, 36 to 61 on every axis; group 4: 86, 6, 62, 10 to 12, 10 to 11; group 5: 92, 6, 75, 10 to 12, 10 to 11; sound rows 52 to 77: 98, 26, 62 to 87 on every axis. The counter after the clip is 88. Under brain's rule the sound block takes positions 10 to 87 and the video block starts at 88 with $T$ from 88 to 93 and $H$, $W$ from 88: the counter is 94.
4. A layer uses $16{,}649{,}472$ parameters including norms (2 more experts of 1,179,648 each than the 14,290,176 of the table), and one frame of the Talker is $288{,}903{,}168 + 20 \cdot 2 \cdot 1{,}179{,}648 = 336{,}089{,}088$ multiply-adds.
5. $1920 \cdot 100 - 555 = 191{,}445$ samples, 7.977 seconds; 555 are missing.
6. $11 + 9 = 20$ rows; $25 - 4 + 1 = 22$ rows trail (21 tokens and the end marker); the padding row starts at frame 22.
7. to 10. See the worked blocks named.
11. The Thinker's prefill of 40 prompt rows is 109.53 billion and does not depend on $F$; the back is $2.447\,F$ billion, 107.67 at $F = 44$ and 110.12 at $F = 45$, so from $F = 45$ the back is the largest part and before that the prefill was. The back's share grows with $F$ and the prefill's does not, so for any reply that is more than a few seconds of speech the first change belongs to the vocoder's back (start it on the first frames), and the prefill remains as the floor.
12. Sound only: the diagonal run $(p_0 + k)$ on every axis in both rules; the same. Video only: the time-aligned $T$ of group $g$ is $p_0 + 13 g$ and brain's is $p_0 + g$; the angle turned on the $T$ axis between two neighbouring groups by a channel of frequency $\omega$ is $13\omega$ against $\omega$, so the time axis rotates 13 times as fast per group.
