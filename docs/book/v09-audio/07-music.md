<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Music Generation

*What changes when the target is minutes of structured music rather than a sentence of speech?*

The sentence of the last chapter lasted 3.04 seconds, and the Talker wrote it in 38 frames. A song of five minutes is about 99 times as long, and the first thing that breaks is not the sound but the arithmetic of attention. The model that brain carries for this task, MiniMax Music 3 in `crates/minimaxmusic3`, takes a caption of the style wanted and the lyrics, and writes a stereo song at 44,100 samples a second, up to five minutes long. At 25 frames a second that is 7,500 frames, and 25,839 of the latent steps that the decoder works in. A bidirectional denoiser that attended over all of them would pay for every pair of steps: $(25{,}840/690)^2$ is about 1,400 times the pairs of the 690 rows that the model's denoiser actually sees at once. The model avoids the cost by refusing to look at the whole song at once, except in one place. A language model reads and writes the whole song, one position per frame, because a causal model over 7,500 positions is only a long document. Everything that makes the sound is done in windows of eight seconds.

That division is the subject of this chapter. **Music generation**, as brain's crate does it, is the production of a long stereo signal from a description by a chain of models that each see a different length of the song: a causal language model that sees all of it as codes, a denoiser that sees eight seconds of latents, and a decoder that turns the same eight seconds into samples. The way a piece of that length is made from windows that each see a part of it is what the chapter calls **long-form audio generation**, and the part that needs the most care is the joint between one window and the next. Chapter 6 supplied the codec language model and the guidance, Volume VIII the latent denoiser and the idea of keeping a region fixed; the chapter follows the song through the five components in the order in which it is made.

Five things from earlier chapters are needed first. The answers are at the end.

1. ([Volume IX, Chapter 3](03-audio-codecs.md), the neural audio codec.) A codec writes 12.5 columns of 16 codes a second, each code from a codebook of 2048 entries. How many integers and how many bits is that a second?
2. ([Volume IX, Chapter 6](06-speech-synthesis.md), the codec language model.) Why does the Talker run one position per frame and not one per code, and what is the price of adding the codes of a frame into one row?
3. ([Volume VIII, Chapter 10](../v08-image-generation/10-latent-diffusion-systems.md), latent diffusion.) In a masked run, what is done after every Euler step to the region that must stay as it was?
4. ([Volume VIII, Chapter 7](../v08-image-generation/07-samplers-and-guidance.md), classifier-free guidance.) Write the guided velocity in terms of the conditional and the unconditional one, and say how many evaluations of the network a step costs.
5. ([Volume VIII, Chapter 6](../v08-image-generation/06-flow-matching.md), flow matching.) For a model that treats time as 0 at noise and 1 at data, what is one Euler step, and what is the velocity of a Gaussian target in closed form?

## Counting a song

A song is long in three units at once, and the first job is to keep them apart. The language model writes **frames**, one every 40 milliseconds: a frame is 960 samples of a signal at 24,000 samples a second, which the crate's configuration states as an input rate and an input hop. The decoder works in **latent steps**, each 512 samples of the output at 44,100 samples a second. The two clocks are not commensurable: a frame is $960 \cdot 44100/24000 = 1764$ output samples, and 1764 is not a multiple of 512. A frame is therefore $1764/512 = 441/128 = 3.4453125$ latent steps, and the first table shows what that does to the lengths.

<!-- worked: music.frame_arithmetic -->
Frames $k$ of 40 milliseconds, in samples of the 24,000 Hz rate that defines the hop, in samples at 44,100 Hz, in latent steps before truncation, and the latent length after it (the product is formed left to right in 64-bit floating point, $k \cdot 44100 / 24000 \cdot 960 / 512$, and truncated once).

| frames | samples at 24,000 Hz | samples at 44,100 Hz | latent steps, exact | latent length |
|---:|---:|---:|---:|---:|
| 1 | 960 | 1,764 | 3.4453 | 3 |
| 2 | 1,920 | 3,528 | 6.8906 | 6 |
| 3 | 2,880 | 5,292 | 10.3359 | 10 |
| 4 | 3,840 | 7,056 | 13.7812 | 13 |
| 5 | 4,800 | 8,820 | 17.2266 | 17 |
| 200 | 192,000 | 352,800 | 689.0625 | 689 |

The nearest-neighbour map of 5 frames onto their 17 latent steps, $\lfloor i \cdot 5 / 17 \rfloor$ for $i = 0 \ldots 16$: 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4. Frame 0 is read 4 times, frame 1 3, frame 2 4, frame 3 3, frame 4 3.
<!-- /worked -->

The conditioning of the denoiser lives on the frame clock and has to be laid on the latent clock, and the code does it in the plainest way there is, by repeating: each latent step reads the frame whose index is $\lfloor i \cdot l_{\mathrm{in}} / l_{\mathrm{out}} \rfloor$ (the nearest-neighbour rule of `crates/minimaxmusic3/src/condition_encoder.rs`), so a frame is read three times or four. The number of latent steps itself has a trap that the crate's own comment points at. The expression is a product and two quotients, and what matters is the order in which they truncate.

<!-- worked: music.integer_division_trap -->
The latent length of $k$ frames with every division truncated at once (`(((k * 44100) // 24000) * 960) // 512`), against the one truncation of the left-to-right float product.

| frames | truncated at each division | one truncation |
|---:|---:|---:|
| 1 | 1 | 3 |
| 2 | 5 | 6 |
| 3 | 9 | 10 |
| 4 | 13 | 13 |
| 5 | 16 | 17 |

Of the frame counts from 1 to 7,500, 5,216 give a different length, by at most 2 steps (first at 1 frames: 1 against 3).
<!-- /worked -->

The condition encoder forms the product in 64-bit floating point from left to right and truncates once, as the reference does; a first draft that truncated at every division wrote 16 latent steps for 5 frames where the right count is 17, and the table shows that the two orders disagree for most frame counts, never by more than two steps. Two steps are 23 milliseconds, but a length that is wrong by a step or two is not an error anywhere downstream, because every later stage takes its length from this one.

Now the lengths themselves. A song's length in frames fixes everything downstream: the integers the language model must write, the latent steps, the number of windows the denoiser needs, and so how many times it runs.

<!-- worked: music.song_lengths -->
A song of a given length, counted as the model sees it: frames at 25 a second, integers the autoregressive stage writes (8 a frame), samples per channel at 44,100 Hz, latent steps, denoiser chunks, and denoiser evaluations (30 steps of 2 evaluations per chunk).

| seconds | frames | integers written | samples per channel | latent steps | chunks | evaluations |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 250 | 2,000 | 441,000 | 861 | 2 | 120 |
| 30 | 750 | 6,000 | 1,323,000 | 2,583 | 7 | 420 |
| 60 | 1,500 | 12,000 | 2,646,000 | 5,167 | 14 | 840 |
| 180 | 4,500 | 36,000 | 7,938,000 | 15,503 | 44 | 2,640 |
| 240 | 6,000 | 48,000 | 10,584,000 | 20,671 | 59 | 3,540 |
| 300 | 7,500 | 60,000 | 13,230,000 | 25,839 | 74 | 4,440 |
<!-- /worked -->

The last two columns are the ones to look at. A song of 10 seconds needs 2 windows and 120 evaluations of a network of 2.4 billion parameters; one of 5 minutes needs 74 windows and 4,440. The integers the language model writes, 60,000 for the longest song, are a small number next to what happens to each frame afterwards, when it becomes 3.45 latent steps of 128 numbers.

## Two interfaces, and the one the denoiser reads

Chapter 6 compared the representations a speech synthesiser passes through and asked at which of them the first model should stop writing. The same table for a song has one new row, and it changes the answer.

<!-- worked: music.interface_rates -->
Per second of music: steps a second, numbers per step, bits per number, and the products. A code of a codebook of $K$ entries is $\log_2 K$ bits; the latent rate is exactly $44100/512$ steps a second.

| representation | steps per second | numbers per step | bits per number | numbers per second | bits per second |
|---|---:|---:|---:|---:|---:|
| waveform, stereo, 16-bit samples | 44,100 | 2 | 16 | 88,200 | 1,411,200 |
| Flow-VAE latents, 32-bit floats | 86.1328125 | 128 | 32 | 11,025 | 352,800 |
| semantic code (one of 16,384) | 25 | 1 | 14 | 25 | 350 |
| residual codes (seven of 1,024 each) | 25 | 7 | 10 | 175 | 1,750 |
| hidden states of the language model and the depth steps, 32-bit floats | 25 | 32,768 | 32 | 819,200 | 26,214,400 |
| condition for the denoiser, 32-bit floats | 86.1328125 | 2,048 | 32 | 176,400 | 5,644,800 |

The eight codes of a frame together are 2,100 bits a second, 200 integers a second.
<!-- /worked -->

Read the codes first. The language model (the crate calls it the Global LLM) writes one semantic code a frame from a vocabulary of 16,384, 14 bits, and a second, small model writes seven residual codes from vocabularies of 1,024, 10 bits each: 2,100 bits a second, 200 integers a second. This is the split of Chapter 6 again, a large model for the first code and a small decoder for the others, a Talker and a code predictor, and the total is about what the 16-codebook synthesiser of that chapter writes (2,200 bits). A codec language model for speech and one for music are the same kind of object at the same rate.

Then read the fifth row. The denoiser does not read the codes. The loop that writes them keeps, for every frame, the hidden vector of the language model that produced the semantic code and the hidden vector of the depth decoder at each of its seven steps, eight vectors of 4,096 numbers, 32,768 numbers a frame and 819,200 numbers a second. These, mixed and projected, are what the denoiser is conditioned on. In bits the interface between the two halves is 26,214,400 a second when the numbers are 32-bit floats, 12,483 times the bits of the codes. The codes are fed back into the language model so that it can write the next frame, and then forgotten: `crates/minimaxmusic3/src/pipeline.rs` returns only the hidden vectors. The latents that the denoiser writes are those of the model's own autoencoder, which the crate calls the Flow-VAE; brain holds its decoder, as the vocoder, and not its encoder.

![Two panels. The first, writing the whole song one frame at a time at 25 frames a second: caption and lyrics as tokens enter a Global LLM of 36 layers that writes one semantic code, with a second branch whose prompt is masked, then a depth decoder of 4 layers that writes 7 residual codes, then 8 hidden vectors of 4096 per frame that are kept and not the codes, then an end token or the frame cap. The second, painting it in chunks of 689 latent steps of 200 frames each: a condition encoder from 8 times 4096 to 2048, a flow DiT of 36 blocks and 30 steps with 2 guidance passes and 172 steps carried from the last chunk, a vocoder upsampling by 512 with two streams, a crop of 86 steps off the left and 258 off the right, and a stereo waveform at 44,100 Hz.](img/music-pipeline.svg)

Why a vector and not the code? The crate does not say. A plausible reading is that a code of 16,384 values is a coarse label and the hidden state that chose it carries what the model was weighing between close candidates, but the source is silent on it and I have found no way to test it from the repository, so it stays a reading. What the code does show is the cost: the hidden vectors of a five-minute song are 983 megabytes of 32-bit floats in host memory until the denoiser has read them (the table that counts them is near the end of the chapter), against 60,000 integers, under half a megabyte even at eight bytes each, for the codes that were thrown away.

## What the prompt says and what it cannot say

The prompt is text, and its shape is a contract with the checkpoint. `crates/minimaxmusic3/src/global_llm.rs` builds it as a caption and a lyrics body between structure tokens: `<|im_start|><|caption_start|>`, the caption, `<|caption_end|><|lyrics_start|>`, the lyrics, `<|lyrics_end|><|im_end|><|audio_start|>`. After the last token the language model writes audio, each frame a token whose id is 151,675 plus the semantic code, so that audio codes are plain extra rows of the embedding table and the output head that the text tokens use, and a special id, 151,670, ends the song. The code comment says that even whitespace changes the sound, and so the caption and the lyrics are not passed through raw but rewritten by two functions that copy the reference's text cleaning. The caption has its markdown stripped (headings, bullets, bold, italics) and any span of the form `<|key value|>` turned into the prose "key is value". The lyrics are rewritten as follows.

<!-- worked: music.prompt_contract -->
The lyrics `[Verse 1] Quiet morning light\nfading into you\n[Chorus]\nHold on to this feeling` after the rewrite (a line that opens with a tag keeps only its tags, so the words after `[Verse 1]` on that line are dropped; tags are lowercased; the text opens with `[start]`):

```text
[start]
[verse 1]
fading into you
[chorus]
Hold on to this feeling
```

The conditional prompt is `<|im_start|><|caption_start|>`caption`<|caption_end|><|lyrics_start|>`lyrics`<|lyrics_end|><|im_end|><|audio_start|>`, and the unconditional prompt has the same length with every token except the first and the last two replaced by one placeholder id. With each part of the toy standing for one token (a real prompt has as many tokens as its caption and lyrics tokenise to), the 9 positions are:

| position | conditional prompt | unconditional prompt |
|---:|---|---|
| 0 | <|im_start|> | <|im_start|> |
| 1 | <|caption_start|> | <audio-cfg id 151654> |
| 2 | [caption text] | <audio-cfg id 151654> |
| 3 | <|caption_end|> | <audio-cfg id 151654> |
| 4 | <|lyrics_start|> | <audio-cfg id 151654> |
| 5 | [lyrics text] | <audio-cfg id 151654> |
| 6 | <|lyrics_end|> | <audio-cfg id 151654> |
| 7 | <|im_end|> | <|im_end|> |
| 8 | <|audio_start|> | <|audio_start|> |
<!-- /worked -->

The rewrite has a consequence for anyone who writes lyrics: the words that follow a tag on the same line are discarded, so `[Verse 1] Quiet morning light` keeps only `[verse 1]`. The tags are the only structure the model is given. Nothing else in the prompt says where in the song a line is sung, and nothing in the code forces it: there is no alignment between the lines of the lyrics and the frames, and the length of the song is not an input but a cap. The option `duration_seconds` becomes a cap of $\mathrm{round}(25 \cdot \mathrm{seconds})$ frames (`max_frames_for_duration` in `crates/minimaxmusic3/src/generate.rs`), the loop stops earlier if the model writes the end token, and a request for ten seconds of a song whose structure needs three minutes is cut where the cap falls, mid-line if it must. The Talker of Chapter 6 had lost the ability to end; this model is able to end and is trusted to do so, and the cap is what catches it when it does not.

The unconditional prompt, the second branch of the figure, is built from the first by replacing tokens. Every token except the first and the last two, which are `<|im_start|>` and the closing structure tokens `<|im_end|>` and `<|audio_start|>`, becomes one placeholder id, 151,654. The two prompts therefore have the same length, which matters because the two branches are stepped in lockstep and the code asserts that the lengths are equal. This is classifier-free guidance in the form that Volume VIII's chapter on guidance gave it, on logits: the guided logits of the next code are $\mathbf{z}_u + g(\mathbf{z}_c - \mathbf{z}_u)$ with $g = 1.5$, a constant of the crate that no option changes. The denoiser has a second, independent guidance with its own scale, 1.7, whose unconditional branch is not a masked prompt at all but a condition tensor of zeros. The two are ported separately, and each doubles the work of its own stage: the language model runs two branches, the denoiser two evaluations a step.

## One row per frame

The language model of this crate is not new. It is the Qwen3 8B decoder of Volume VI with a vocabulary of 200,000 ids, reused from `crates/qwen3` without a change (`global_llm::import` streams the checkpoint into 8-bit linears so that it is never resident in full), and Volume VI's chapter on cost counted its parameters at the usual vocabulary. What is new is how a frame is read back. When the model has written a frame's eight codes, the semantic one is looked up in the language model's own table and the seven residual ones in tables of the depth decoder, the eight rows are added, and the sum is multiplied by $8^{-1/2}$ and fed back as one row (`pipeline::embed_audio_frame`). That is the trick of the Talker in Chapter 6, and it pays for the same reason, shown here with the counts of this song.

<!-- worked: music.column_vs_flat -->
Positions of the causal language model for a song, with one position per frame (the eight codes of a frame summed into one row) and with one position per code, and the number of query-key pairs per head and layer, $n(n+1)/2$ for $n$ positions.

| seconds | positions, one per frame | pairs | positions, one per code | pairs | ratio of pairs |
|---:|---:|---:|---:|---:|---:|
| 10 | 250 | 31,375 | 2,000 | 2,001,000 | 63.8 |
| 60 | 1,500 | 1,125,750 | 12,000 | 72,006,000 | 64.0 |
| 300 | 7,500 | 28,128,750 | 60,000 | 1,800,030,000 | 64.0 |
<!-- /worked -->

One position per frame makes the causal attention of a five-minute song 64 times cheaper than one position per code, which is $8^2$ and no accident. Each position of the language model's attention cache holds $2 \cdot 36 \cdot 8 \cdot 128 = 73{,}728$ numbers, so the cache is a few gigabytes only because there are 7,500 positions and not 60,000.

<!-- worked: music.kv_cache -->
The attention cache of one branch of the Global LLM: 2 tensors x 36 layers x 8 key-value heads x 128 = 73,728 numbers per position. One position per frame, after a prompt of 200 tokens (an assumed length: the real one is the length of the caption and the lyrics). Bytes are for 32-bit numbers; the last column is the two guidance branches.

| seconds | positions | numbers per branch | GB per branch | GB, two branches |
|---:|---:|---:|---:|---:|
| 10 | 450 | 33,177,600 | 0.133 | 0.265 |
| 60 | 1,700 | 125,337,600 | 0.501 | 1.003 |
| 300 | 7,700 | 567,705,600 | 2.271 | 4.542 |
<!-- /worked -->

The semantic code comes from the language model's output head at the last position. The seven residual codes need a second network, the **depth decoder**, a causal transformer of 4 layers of width 4,096 with 16 heads and a feed-forward width of 6,144, a table of position embeddings for at most 16 positions, and no rotary embedding, no grouped heads and no bias. For one frame it is run on a sequence of 8 positions, each the output of one and the same bias-free 4,096 by 4,096 projection: the language model's hidden vector, the embedding of the semantic code, and then the embedding of each of the first six residual codes as it is drawn (the seventh is the last thing the frame needs, so it is not fed back). Each of the last seven positions has an output head of its own (1,024 outputs) giving the logits of one residual codebook, and each residual code is embedded by a table of its own, seven tables of 1,024 rows (the seventh serves only the sum that becomes the language model's next row). The tensors are listed with their counts in the section on what a song costs. It runs for the two guidance branches, which share the sampled code, so one code is drawn and fed to both. Its inference path keeps the keys and values of the earlier positions of a frame, which turns 35 position-forwards a branch into 8. The commit that did it, `f111e6cb3`, claims that nothing observable moves, and the file supports the claim in the strong form: the cached step is compared with the full forward by an equality of the 32-bit vectors, not a tolerance. The depth decoder is the memory-bound part of the loop. One position costs 570,425,344 multiply-adds per branch and reads 4 layers of weights, 2.28 gigabytes at 4 bytes a weight: a quarter of a multiply-add, half an operation, for each byte. That is why the two branches are batched into one pass over the weights (`depth_decoder::step_batch`) and not run one after the other.

## Drawing a code under two prompts

The sampler is short enough to write out. For the semantic code it takes the logits of each branch at the last position, sets to $-\infty$ every id that is neither a semantic code nor the end token, blends the two with $g = 1.5$, keeps only the ids that are among the 50 largest logits of the *conditional* branch, takes the 50 largest of what remains, applies a softmax with no temperature, and draws one id by the inverse CDF of a uniform number from the run's generator. The residual codes use the same blend and then only the top 50 of the blended logits.

The restriction by the conditional branch is the one step that is not in Chapter 6's chain or in Volume VIII's account of guidance, and a small vocabulary shows what it does.

<!-- worked: music.ar_sampling_toy -->
One draw of the autoregressive sampler on a vocabulary of 8 codes, with the top-$k$ of the real sampler shrunk to $k = 3$ (the code uses 50) and the guidance scale 1.5 of the code. Logits of the conditional and of the unconditional prompt; guided = unconditional + 1.5 (conditional - unconditional); the conditional top-3 threshold is 3.1. The $k$ largest of the *unrestricted* guided logits would have been codes 3, 4, 6; after the conditional restriction the final top-$k$ keeps 1, 3, 4. The softmax of the survivors has no temperature. The uniform draw is the first of `music.ar_sampling_toy` (seed 0x13fbc4de27f5e5da), $u = 0.026356$, and the inverse CDF returns code 1.

| code | conditional | unconditional | guided | in the conditional top-3 | kept after the second top-3 | probability |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.00 | 2.40 | 1.80 | no | -inf | 0.0000 |
| 1 | 3.10 | 2.00 | 3.65 | yes | 3.65 | 0.2101 |
| 2 | 0.20 | 1.00 | -0.20 | no | -inf | 0.0000 |
| 3 | 4.00 | 3.00 | 4.50 | yes | 4.50 | 0.4917 |
| 4 | 3.90 | 3.70 | 4.00 | yes | 4.00 | 0.2982 |
| 5 | 1.00 | 1.20 | 0.90 | no | -inf | 0.0000 |
| 6 | 2.60 | 0.40 | 3.70 | no | -inf | 0.0000 |
| 7 | 3.00 | 3.40 | 2.80 | no | -inf | 0.0000 |
<!-- /worked -->

Code 6 has a conditional logit of 2.60, below the third largest, and an unconditional logit of only 0.40; the blend lifts it to 3.70, above code 1. Without the restriction the guided top 3 would be 3, 4 and 6. With it, a code that the conditional prompt itself does not rank among the leading candidates cannot enter just because the unconditional prompt disliked it more, and the survivors are 1, 3 and 4. Guidance here can reorder the candidates that the prompt proposes and cannot add one.

One detail of the loop is easy to miss and the code does not explain it. The loop runs for one more iteration than the cap, and the frame sampled in the first iteration, from the hidden state at the end of the prompt, is fed back into the language model but is not kept for the denoiser (`if frame_index > 0` in `pipeline::generate_frames`). I could not tell from the crate whether that is the reference's convention or a choice. The arithmetic of this chapter takes the kept frames as the song's frames and does not depend on it.

## What the denoiser reads, and how it is given the time

The hidden vectors reach the denoiser through the **condition encoder**, which is almost nothing: a learned softmax over the eight vectors of a frame (eight scalar logits and one scale) that mixes them into one vector of 4,096 numbers, a convolution of three taps from 4,096 to 2,048 channels along the frame axis, and the nearest-neighbour map of the first table. It runs on the host, once per chunk, because a device round trip would be pure overhead for a few megabytes.

The denoiser itself, `crates/minimaxmusic3/src/dit.rs`, is a transformer of 36 blocks of width 2,048 (32 heads of 64) with full bidirectional attention over a chunk and no causal mask, because it denoises the whole chunk at once. Its input is built by concatenating three blocks of channels at every latent step: the 128 channels of the current latents, 128 channels of zeros, and the 2,048 channels of the condition, 2,304 channels in all, to which a convolution of one tap adds a residual and a projection takes the width to 2,048. The 128 channels of zeros are zero in this crate; what the checkpoint was trained to receive there the code does not say. Two things in the block stack set it apart from the image models of Volume VIII. The first is that the time is not injected into the blocks at all. The norms are plain LayerNorms with a gain and a bias, and no block is modulated by the time. The time $t$ becomes a vector (a random-Fourier embedding, $\cos(2\pi t w_i)$ and $\sin(2\pi t w_i)$ for 128 weights $w_i$, passed through two linear layers with a SiLU between them) that is placed as row 0 of the sequence, in front of the latent steps. The sequence therefore has one row more than the chunk, 690 for a chunk of 689, and the time is read the way any token is read, by attention. The second is the rotary embedding, which rotates only the first 32 of the 64 numbers of each head, at a base of 10,000 (`dit::rope_tables`), so that half of each head sees position and half does not. The feed-forward network is a gated one with a SiLU gate, a SwiGLU, whose up and gate projections are one stored matrix of 16,384 rows split into two halves.

<!-- worked: music.dit_breakdown -->
The multiply-adds of one DiT evaluation over a full chunk (689 latent steps, 690 rows with the timestep token), in giga, split by kind. The blocks see all rows; the glue is applied to the latent steps only, as 1-tap convolutions 2,304 to 2,304, a projection 2,304 to 2,048, a projection 2,048 to 128 and a convolution 128 to 128, and the time embedding (256 to 2,048 to 2,048) is applied once.

| part | multiply-adds (G) | share (%) |
|---|---:|---:|
| attention projections (q, k, v, out), 36 blocks | 416.7 | 23.89 |
| feed-forward (up, gate, down), 36 blocks | 1,250.2 | 71.68 |
| attention scores and weighted sums | 70.2 | 4.02 |
| input and output glue (1x1 convolutions, projections, timestep embedding) | 7.1 | 0.41 |
| one evaluation | 1,744.3 | 100.00 |
<!-- /worked -->

Almost three quarters of the arithmetic is the feed-forward network and a further quarter the attention projections. The attention proper, the scores and the weighted sums, is 4 percent of a chunk's multiply-adds. That number matters in the section on history: a term that is 4 percent of the arithmetic was once three quarters of the time.

The clock of the denoiser is the other convention that Volume VIII prepared. The model runs with time 0 at noise and 1 at data, the opposite of the scheduler of the image models, and the scheduler took a flag for it, `invert_sigmas`, in commit `c0468958f`, whose message says that it was added for this model and that the update step needed no change because it is written in differences. With 30 steps the schedule works out to something very simple.

<!-- worked: music.denoise_clock -->
The clock of the denoiser for 30 steps: the input sigmas are linspace(1, 1/30, 30), the shift is 1, every sigma is replaced by 1 - sigma and the terminal value 1 is appended. The closed form is $t_i = i/30$; the largest difference between the two over the 30 steps is below 1e-15 (floating-point rounding only). The last three columns are, at each step, the step $\Delta t = t_{i+1} - t_i$, and the weights of the noise and of the carried data in the blend $(1 - (1 - 10^{-6}) t)\,\varepsilon + t\,x_{\mathrm{prev}}$ that holds a carried prefix.

| step i | t_i | i / 30 | Delta t | weight of the noise | weight of the carried data |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000000 | 0.000000 | 0.033333 | 1.000000 | 0.000000 |
| 1 | 0.033333 | 0.033333 | 0.033333 | 0.966667 | 0.033333 |
| 2 | 0.066667 | 0.066667 | 0.033333 | 0.933333 | 0.066667 |
| 3 | 0.100000 | 0.100000 | 0.033333 | 0.900000 | 0.100000 |
| 28 | 0.933333 | 0.933333 | 0.033333 | 0.066668 | 0.933333 |
| 29 | 0.966667 | 0.966667 | 0.033333 | 0.033334 | 0.966667 |

The 30 steps all have $\Delta t = 1/30 = 0.033333$ (the largest deviation is below 1e-15).
<!-- /worked -->

The inverted, unshifted schedule is $t_i = i/30$, an even grid of 30 equal steps of 1/30. The network is evaluated at $t = 0, 1/30, \ldots, 29/30$ and never at 1, and the last Euler step lands on the data. The same table prints the weights of the blend that holds a carried prefix, which the next section needs.

## Chunks of eight seconds

A window cannot be made by cutting the song into eight-second pieces and denoising each alone, and a toy shows why before the code does. Suppose that the denoiser, given its condition, knows the melody of a stereo clip but not its texture, and that the texture of one chunk is a sample of a smooth random process: neighbouring steps are correlated, and nothing connects the texture of one chunk to the texture of the next. Whatever the network, its chunks will agree about the melody and disagree about the texture, and the stitched clip will jump wherever one chunk's kept stretch ends and the next begins. The remedy comes from Volume VIII, where it was introduced for another purpose. In the masked run of Chapter 10 a region that must stay as it was is, after every step, replaced by the source renoised to the level the step has reached, so that it is always a legal point on the source's own path from noise to data. Here the region to keep is the beginning of a new chunk, and the source is the end of the previous one.

The constants of the crate say how the chunks are laid. A chunk is 200 frames. The next chunk starts 100 frames later. The first 172 latent steps of every chunk after the first are not generated afresh but taken from the previous chunk, and 172 is exactly $(689 - 345)/2$: the slice that the previous chunk hands over is its steps 345 to 517, which starts in the middle of the chunk and ends a quarter of its length before the right edge, and that is where the next chunk, which starts 344.53 steps later, begins.

<!-- worked: music.chunk_constants -->
The constants of the chunked denoiser in the units of the frame, the latent step and the clock. A latent step is $512/44100$ of a second; a frame is $1764/512 = 441/128 = 3.4453125$ latent steps. The chunk and the hop are defined in frames, and their seconds and samples are the frame clock's, nominal; the other four rows are defined in latent steps, and their seconds and samples are the latent clock's. The 689 steps of a chunk are 352,768 samples, 7.9993 seconds, 32 samples short of the nominal 200 frames, because the latent length is truncated; the hop's 344.5312 steps are fractional, and the carried slice is placed at a whole step.

| constant | in frames | in latent steps | seconds | samples at 44,100 Hz |
|---|---:|---:|---:|---:|
| chunk (on the frame clock) | 200 frames | 689 | 8.0000 | 352,800 |
| hop between chunk starts (on the frame clock) | 100 frames | 344.5312 | 4.0000 | 176,400 |
| carried prefix | 49.923 frames | 172 | 1.9969 | 88,064 |
| crop off the left of a later chunk | 24.961 frames | 86 | 0.9985 | 44,032 |
| crop off the right of an earlier chunk | 74.884 frames | 258 | 2.9954 | 132,096 |
| kept from a chunk with a neighbour on each side | 100.136 frames | 345 | 4.0054 | 176,640 |
<!-- /worked -->

The rule that turns a number of frames into chunks is short, and the edges of it are where a reimplementation goes wrong.

<!-- worked: music.chunk_rule -->
A song of $F$ frames is one chunk when $F \le 200$, and otherwise $\lceil (F - 200) / 100 \rceil + 1$ chunks, starting at the frames $0, 100, 200, \ldots$ up to the last start that is below $F - 100$ (the starts are `range(0, F - 100, 100)`). Chunk $i$ has $\min(200, F - s_i)$ frames, so the last chunk has more than 100 and at most 200 frames, and its latent length is the truncated product of the first table. Every chunk but the first drops its first 86 latent steps and every chunk but the last drops its last 258; the last chunk therefore keeps steps 86 up to its own length. A song of 200 frames or fewer is a single chunk with no crop at all.

| frames $F$ | chunks | first starts | frames of the last chunk | latents of the last chunk | kept of the last chunk |
|---:|---:|---|---:|---:|---:|
| 1 | 1 | 0 | 1 | 3 | 0 to 3 |
| 200 | 1 | 0 | 200 | 689 | 0 to 689 |
| 201 | 2 | 0, 100 | 101 | 347 | 86 to 347 |
| 300 | 2 | 0, 100 | 200 | 689 | 86 to 689 |
| 301 | 3 | 0, 100, 200 | 101 | 347 | 86 to 347 |
| 450 | 4 | 0, 100, 200, ... | 150 | 516 | 86 to 516 |
| 7,500 | 74 | 0, 100, 200, ... | 200 | 689 | 86 to 689 |
<!-- /worked -->

In `crates/minimaxmusic3/src/denoise.rs` the hold is written like the masked run of Volume VIII, applied at the top of every step and not at its end. With $\varepsilon_p$ the noise that the chunk was started from in its prefix of $P$ steps and $\mathbf{x}_{\mathrm{prev}}$ the slice carried over, the prefix is set before the network is called to

$$
(1 - (1 - 10^{-6})\,t)\,\varepsilon_p + t\,\mathbf{x}_{\mathrm{prev}},
$$

which is pure noise at $t = 0$ and, up to the $10^{-6}$ that the code keeps, the carried data at $t = 1$; the blend weights are in the last two columns of the clock table. After the last step the prefix is written back as exactly the carried values. The condition of the prefix is carried in the same way, by copying the previous chunk's slice, and not recomputed: the new chunk's frames start 344.53 latent steps after the old one's, the carried latents were placed at step 345, and the copy keeps each carried latent with the conditioning it was denoised under. (That is my reading of why; the code states only what it does.) The loop, with the guidance of Chapter 6's `cfg_euler` and one addition, is short.

```text
function hold_euler(u_cond: function(array<real>[C, L], real) -> array<real>[C, L],
                    u_uncond: function(array<real>[C, L], real) -> array<real>[C, L],
                    noise: array<real>[C, L], prev: optional array<real>[C, P], n: int, g: real)
        -> array<real>[C, L]
    require (not (prev is present)) or prev.shape[1] <= L, "the carried prefix must fit in the chunk"
    x = copy(noise)
    for i in 0 .. n
        t = i / n
        if prev is present
            P = prev.shape[1]
            x[:, 0:P] = (1 - (1 - 0.000001) * t) * noise[:, 0:P] + t * prev
        uc = u_cond(x, t)
        un = u_uncond(x, t)
        x = x + (1 / n) * (un + g * (uc - un))
    if prev is present
        x[:, 0:prev.shape[1]] = prev
    return x
```

A toy makes the loop exact. A song of 40 latent steps in two channels is cut into chunks of 16 steps with a hop of 8, so that the carried prefix is $(16 - 8)/2 = 4$ steps and the crops are 2 on the left and 6 on the right, the real rules ($86 = 172/2$, $258 = 2 \cdot 172 - 86$) at a smaller scale. Given the condition, which is the melody of each channel, the texture of a chunk is a Gaussian process with covariance $\Sigma_{ij} = \sigma^2\rho^{|i-j|}$, with $\sigma = 0.5$ and $\rho = 0.9$, so $0.5^2\,0.9^{|i-j|}$. For a Gaussian target of mean $\mathbf{m}$ and covariance $\Sigma$ on the straight path $\mathbf{x}_t = (1 - t)\boldsymbol\varepsilon + t\mathbf{x}_1$, the posterior mean of the data is $\mathbf{m} + t\Sigma(t^2\Sigma + (1 - t)^2\mathbf{I})^{-1}(\mathbf{x}_t - t\mathbf{m})$, and the exact velocity is the difference between it and $\mathbf{x}_t$ divided by $1 - t$. This is the formula of Chapter 6's one-number toy with a variance replaced by a covariance, and it needs no network.

<!-- worked: music.chunk_flow_toy -->
The chunked flow of the chapter's toy: a song of 40 latent steps in 2 channels, chunks of 16 steps, a hop of 8 (chunks start at 0, 8, 16 and 24), a carried prefix of 4, crops of 2 on the left and 6 on the right, and 30 Euler steps with $t_i = i/30$. Given the melody $m_c(g) = 0.6 \sin(2\pi g / 10 + 0.5c)$ of channel $c$, the texture $z = x_1 - m$ of each channel of a chunk is a Gaussian process with covariance $\Sigma_{ij} = 0.5^2\, 0.9^{|i-j|}$ (neighbouring steps are correlated), the two channels independent, and the velocity is the exact one of the posterior. The noise is `music.chunk_flow_toy` (seed 0xd9987001b06c2d28): four chunks in order, each 2 x 16 draws of `normal()`, channel 0's 16 first; both runs use the same draws. The table prints the texture of channel 0 at the six song positions 8 to 13 around the first seam (the seam is at position 10: the last kept step of chunk 0 is position 9, and the held prefix of chunk 1 ends, its join, at position 12), and the root mean square step of the texture (both channels) at the seams and at the joins against all other positions. The texture of the data itself has a step of 0.224 (rms).

| run | z at 8 | z at 9 | z at 10 | z at 11 | z at 12 | z at 13 | rms step at seams and joins | rms step elsewhere |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| independent chunks | 0.745 | 0.489 | 0.753 | 0.596 | 0.535 | 0.742 | 0.387 | 0.180 |
| hold the carried prefix | 0.745 | 0.489 | 0.343 | 0.065 | 0.411 | 0.696 | 0.208 | 0.181 |

The absolute step of channel 0 across the three seams, and across the three joins (the step into the first position that the chunk generated for itself), for each run:

| run | seam 1 | seam 2 | seam 3 | join 1 | join 2 | join 3 |
|---|---:|---:|---:|---:|---:|---:|
| independent chunks | 0.264 | 0.921 | 0.409 | 0.061 | 0.058 | 0.362 |
| hold the carried prefix | 0.146 | 0.152 | 0.138 | 0.346 | 0.188 | 0.114 |
<!-- /worked -->

![The texture of channel 0 of the toy, stitched from four chunks, against position in the song. The curve of independent chunks jumps at the three seams, marked by green dashed lines at positions 10, 18 and 26; the curve with the prefix held follows a smooth continuation across the seams and shows a kink at the first of the three joins, orange dashed lines at positions 12, 20 and 28.](img/chunk-seam-toy.svg)

The independent chunks jump at the seams, by 0.26, 0.92 and 0.41 against a typical step of 0.18. With the prefix held the seam is not a place at all: the first chunk's own latents continue straight through it, because the new chunk's first four steps are the old chunk's steps 8 to 11, and the steps at the three seams are 0.15, 0.15 and 0.14, ordinary. The discontinuity has moved to where the held prefix ends and the chunk's own denoising takes over, the join of the figure. At the first join it is 0.35, about twice the ordinary step; at the other two it is 0.19 and 0.11, not distinguishable from the rest. It is smaller than the jump it replaced, and for this draw it is not entirely gone. One song proves little. The next table repeats the experiment over 100 songs for three hops, so for prefixes of 2, 4 and 6 steps, and divides the mean squared step at the seams and joins by the mean squared step elsewhere, so that a ratio of 1 is a boundary no one could find.

<!-- worked: music.chunk_flow_trials -->
The same toy over 100 songs for three hops (and so three prefix lengths, $(16 - \mathrm{hop})/2$). For each song the mean squared step of the texture at the seams and joins is set against the mean squared step elsewhere; the table gives the ratio of the square roots of their averages over the songs (1 would be a boundary no one could find). The third run holds the carried prefix at its carried values at every step instead of renoising it. The noise of each hop is its own stream, `music.chunk_flow_toy.trials.hop<hop>` (seeds 0xa56bb12e1f372b0b, 0xd5aa80908bae089e, 0xec9fdc7e5b61906d, for hops 12, 8 and 4), songs one after the other, chunks in order, draws as above; the three runs of a hop replay the same draws. Cost counts denoiser steps for the whole song (one evaluation per step).

| hop | prefix | chunks | steps | independent chunks | hold, renoised | hold, clean |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 2 | 3 | 90 | 2.41 | 1.68 | 1.50 |
| 8 | 4 | 4 | 120 | 2.46 | 1.58 | 1.40 |
| 4 | 6 | 7 | 210 | 2.42 | 1.59 | 1.42 |
<!-- /worked -->

The hold takes the boundary excess down from about 2.4 times the ordinary step to about 1.6 and leaves it there whatever the length of the prefix, and a longer prefix costs more steps, because the hop shrinks. Holding the carried values unchanged at every step, which is not a legal input of the exact velocity, which expects at time $t$ a point that is partly noise, lowers the ratio only to between 1.4 and 1.5. So the renoising is not what keeps the join from vanishing. What does is a question the toy leaves open: it has a straight path, an exact velocity and 30 steps, and I do not claim its numbers of the real denoiser, whose behaviour at a join the repository does not measure.

## Where a chunk's samples go

A chunk of 689 latent steps goes through the vocoder whole, and then the stitcher crops its samples and concatenates them. The crops are written in latent steps, 86 off the left of every chunk but the first and 258 off the right of every chunk but the last, and each step is 512 samples (`stitch::Stitcher::push_chunk`). The kept stretch of a chunk with a neighbour on each side is $689 - 86 - 258 = 345$ steps, which is the hop of 344.53 rounded up, and the figure lays the chunks of an 18-second song on the song's clock.

![Four chunks of an 18-second song as horizontal bars on a time axis from 0 to 18 seconds. Each bar is 8 seconds long (the last, 6 seconds) and starts 4 seconds after the one before it. Blue marks the stretch of a chunk that reaches the song, grey the stretch cropped away, orange the first 86 steps of a carried prefix, which are cropped, and green the last 86 steps of the carried prefix, which are kept. Dashed green lines mark the seams at 5.0, 9.0 and 13.0 seconds and dashed orange lines the joins at 6.0, 10.0 and 14.0 seconds.](img/chunk-windows.svg)

<!-- worked: music.chunk_geometry -->
The chunks of a song of 450 frames (18 seconds). `carried` is the latent prefix the chunk takes over from the previous one, `kept` the span of its own latents that reaches the song, and the last two columns where that span starts in the song's samples and seconds.

| chunk | start frame | frames | latents | carried | first kept | end kept | kept | first sample | second |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 200 | 689 | - | 0 | 431 | 431 | 0 | 0.0000 |
| 1 | 100 | 200 | 689 | 172 | 86 | 431 | 345 | 220,672 | 5.0039 |
| 2 | 200 | 200 | 689 | 172 | 86 | 431 | 345 | 397,312 | 9.0093 |
| 3 | 300 | 150 | 516 | 172 | 86 | 516 | 430 | 573,952 | 13.0148 |

The kept spans add up to 1,551 latent steps, 794,112 samples per channel; 450 frames of 1,764 samples are 793,800. The seams, where one chunk's kept stretch ends and the next begins, are at 5.0039, 9.0093, 13.0148 seconds; the joins, where a chunk's prefix ends and its own generation begins, 86 steps after the seams, are at 6.0024, 10.0078, 14.0132 seconds.
<!-- /worked -->

The arithmetic has one visible consequence. The stretch kept from each middle chunk is 345 steps, 176,640 samples, and the stride between chunks is 100 frames, 176,400 samples, so every middle chunk adds 240 samples of song that the frame clock does not account for: the carried slice is placed at an integer step while the hop is a fraction of one.

<!-- worked: music.length_drift -->
The stitched length against the frame count's nominal length, in samples per channel. A chunk that is neither first nor last keeps 689 - 86 - 258 = 345 latent steps, 176,640 samples, while its frame stride of 100 is 176,400.

| seconds | chunks | samples stitched | samples nominal | difference | difference in seconds |
|---:|---:|---:|---:|---:|---:|
| 10 | 2 | 440,832 | 441,000 | -168 | -0.0038 |
| 60 | 14 | 2,649,088 | 2,646,000 | +3,088 | 0.0700 |
| 240 | 59 | 10,597,888 | 10,584,000 | +13,888 | 0.3149 |
| 300 | 74 | 13,247,488 | 13,230,000 | +17,488 | 0.3966 |
<!-- /worked -->

A song of four minutes is 0.31 seconds longer than its frames say, one of five minutes 0.40, a tempo error of about 0.13 percent. It is inaudible, and it is the sort of thing a length computation has to know. The crate says that it ports the constants of the reference, and I cannot say what the reference does about it.

The geometry also says where a boundary lives. The first chunk's kept stretch ends at step 431, 5.0039 seconds into the song, and that is the seam: the next chunk's kept stretch begins at its step 86, and that step is the same latent, the previous chunk's step 431, by the rule that its first 172 steps are the previous chunk's 345 to 517. The join, where the new chunk's own generation begins, is at its step 172, 6.0024 seconds. Commit `7525154b5` measured a seam. Its message reports the largest sample-to-sample step at a boundary at four seconds into the song, 0.0031, against a 99.9th percentile of 0.0635 elsewhere, and concludes that a crossfade is not needed because the overlap is replaced in the latent domain before the vocoder sees it. The mechanism is right, and the code confirms it. The place is not the seam: four seconds is where the second chunk starts, and its first 86 steps never reach the song. I cannot tell from the message where the samples were taken, so what the commit supports is a conclusion about the seam, which is exact by construction, and not about the join, which the toy suggests is where a hold is least perfect.

Why are 86 and 258 so large? The vocoder is a stack of convolutions and its reach in time is finite, and a walk backward through it says how finite.

<!-- worked: music.vocoder_reach -->
The interval of positions that can change one block of 512 output samples (those of latent step $q$ = 1000), found by walking the decoder backward from the output: each convolution widens the interval by its dilation times $(k-1)/2$ on each side, each transposed convolution maps it to the input positions that a window of its kernel touches. Positions are in the layer's own resolution. A residual unit is walked backward as its 1-tap convolution and then its 7-tap convolution at the unit's dilation (a convolution pads each side by dilation x 3 so that the length is kept), and the three units of a block with dilations 9, 3 and 1 in that order (forward they are 1, 3 and 9, after the upsampling). A transposed convolution of stride $s$ has a kernel of $k$ taps, twice $s$, and pads $s/2$ on each side: its output position $o$ takes input $i$ through tap $j$ when $o = i s - s/2 + j$, so an output interval $[a, b]$ needs the inputs from $\lceil (a + s/2 - k + 1)/s \rceil$ to $\lfloor (b + s/2)/s \rfloor$.

| after walking back through | first position | last position | positions |
|---|---:|---:|---:|
| final convolution (k=7) | 511,997 | 512,514 | 518 |
| block 3: three units and the upsampling by 2 | 255,978 | 256,277 | 300 |
| block 2: three units and the upsampling by 4 | 63,984 | 64,079 | 96 |
| block 1: three units and the upsampling by 8 | 7,992 | 8,015 | 24 |
| block 0: three units and the upsampling by 8 | 993 | 1,007 | 15 |
| first convolution (k=7) | 990 | 1,010 | 21 |

At the latent rate the 512 samples of step 1000 depend on steps 990 to 1010: 10 before and 10 after. The crops of the stitcher are 86 and 258 steps.
<!-- /worked -->

Ten latent steps on each side. A crop of 10 would remove all of the vocoder's own edge effect, so 86 and 258 are not about the vocoder. The comment in the stitcher says that the vocoder needs extra context to decode cleanly at the edges; that is true up to 10 steps, and the rest of the margin must be for something else, plausibly the bidirectional attention of the denoiser, whose steps at the edge of a chunk have less to attend to than steps in the middle. That is a reading, with no experiment in the repository behind it.

## From latents to two ears

The decoder is a convolutional stack of the family of Chapter 3's codec decoder and Chapter 6's vocoder, with the changes that music asks. The 128 latent channels at each step are not 128 channels of one signal but two halves of 64, one for each ear: the decoder is a mono decoder applied to a batch of two, so both ears share every weight and the stereo image lives entirely in how the denoiser filled the two halves of the latent. The stack is a one-tap projection from 64 channels to 1,024, a convolution of 7 taps to 1,536, four blocks that upsample by 8, 8, 4 and 2 (512 in all) and halve the channels each time, from 1,536 to 96, and a final Snake activation, a convolution of 7 taps down to one channel and a tanh. Each block is a Snake, a transposed convolution with a kernel twice the stride, and three residual units with dilations 1, 3 and 9. The **Snake** activation of the crate (`crates/audio/src/snake.rs`) is $y = x + \sin^2(\alpha x)/(\alpha + 10^{-9})$ with a learned $\alpha$ for every channel: it adds a periodic term to the signal, which suits tonal sound, and the reasons are in the paper of the references.

The decoder has 54,157,665 parameters, counted from the shapes of its tensors. What decides what a card must hold to run it is not the weights, 207 mebibytes, but the activations, because the whole decoder is recorded as one tape and submitted once, so every intermediate lives until the end.

<!-- worked: music.vocoder_memory -->
The buffers of the vocoder for one chunk of 689 latent steps: the decoder is recorded as one tape and submitted once, so every intermediate is alive until the end. Floats per stream, and mebibytes ($2^{20}$ bytes) for the two streams at 4 bytes. A block of $c$ channels in and $c/2$ out, from $L$ to $sL$ steps, holds the input Snake ($c L$ floats, at the block's input resolution), the transposed convolution's output ($c s L / 2$), and three residual units of five buffers each (Snake, dilated convolution, Snake, 1-tap convolution and the sum, each $c s L / 2$), 17 buffers in all. The first row is the projection ($1{,}024 L$) and the first convolution ($1{,}536 L$); the last is the Snake over 96 channels and the one-channel output.

| part | buffers | floats per stream | MiB, two streams |
|---|---:|---:|---:|
| input projection and first convolution | 2 | 1,763,840 | 13.5 |
| block 0: 768 channels at 5,512 steps | 17 | 68,789,760 | 524.8 |
| block 1: 384 channels at 44,096 steps | 17 | 275,159,040 | 2,099.3 |
| block 2: 192 channels at 176,384 steps | 17 | 558,784,512 | 4,263.2 |
| block 3: 96 channels at 352,768 steps | 17 | 575,717,376 | 4,392.4 |
| final Snake and convolution | 2 | 34,218,496 | 261.1 |
| all of them |  | 1,514,433,024 | 11,554.2 |

The crate's ledger records a measured peak of 12,264 MiB for such a chunk (`VOCODER_CHUNK_DEVICE_PEAK_BYTES`); the activations account for 94.2 percent of it.
<!-- /worked -->

The count says that the intermediates of one chunk are 11,554 mebibytes at 4 bytes, 94.2 percent of the 12,264 that the crate's ledger measured as the stage's peak. The statement in commit `7525154b5` that two chunks in a row peak at the level of one is the consequence: the buffers are freed when a chunk has been decoded, and what two chunks in a row cost is one chunk's worth. The same tape is where the vocoder's other memory defect hid. The decoder passed `elements * 4` where the allocator takes words, so each buffer was four times as big as it needed to be (commit `070631735`), and the defect was invisible to every parity test, which decode a chunk of 6 latent steps, because a buffer four times too large that is never read changes no output. It showed only at a real chunk length, as an out-of-memory in a two-chunk run. A gate that cannot see the failure it guards against is kin to the gates of knowledge #1, which never ran: both report green.

## What a song costs

The five components have these sizes, counted from the crate's configuration and the shapes of its tensors. The tensors come first, so that every count that follows can be derived from them.

<!-- worked: music.architecture_layouts -->
The tensors of the five components, the Global LLM in the layout of Volume VI, with the counting conventions. A shape lists outputs before inputs, so a linear map from 4,096 to 2,048 numbers is 2,048 x 4,096; a convolution is outputs x inputs x taps. A count is the product of the shape and the copies, and the total of each table is the count of the next one. Where a bias is not listed the map has none.

Global LLM (the layout of Volume VI's Qwen3 8B, a vocabulary of 200,000, 36 layers, 32 query and 8 key-value heads of 128, a feed-forward width of 12,288):

| tensor | shape (outputs x inputs) | copies | parameters |
|---|---|---:|---:|
| token embedding table | 200,000 x 4,096 | 1 | 819,200,000 |
| output head (untied) | 200,000 x 4,096 | 1 | 819,200,000 |
| final RMSNorm gain | 4,096 | 1 | 4,096 |
| W_Q, per layer | 4,096 x 4,096 | 36 | 603,979,776 |
| W_K, per layer | 1,024 x 4,096 | 36 | 150,994,944 |
| W_V, per layer | 1,024 x 4,096 | 36 | 150,994,944 |
| W_O, per layer | 4,096 x 4,096 | 36 | 603,979,776 |
| query and key norm gains, per layer | 2 x 128 | 36 | 9,216 |
| W_gate, per layer | 12,288 x 4,096 | 36 | 1,811,939,328 |
| W_up, per layer | 12,288 x 4,096 | 36 | 1,811,939,328 |
| W_down, per layer | 4,096 x 12,288 | 36 | 1,811,939,328 |
| two RMSNorm gains, per layer | 2 x 4,096 | 36 | 294,912 |
| total |  |  | 8,584,475,648 |

RVQ depth decoder (4 layers of width 4,096, 16 heads, a feed-forward width of 6,144; no rotary embedding, no grouped heads, no bias; the table has 1,024 rows, the codebook size):

| tensor | shape (outputs x inputs) | copies | parameters |
|---|---|---:|---:|
| audio embedding tables, one per residual codebook | 1,024 x 4,096 | 7 | 29,360,128 |
| output heads, one per residual codebook | 1,024 x 4,096 | 7 | 29,360,128 |
| input projection, one matrix applied to every input row | 4,096 x 4,096 | 1 | 16,777,216 |
| position table | 16 x 4,096 | 1 | 65,536 |
| final RMSNorm gain | 4,096 | 1 | 4,096 |
| attention W_q, W_k, W_v, W_out, per layer | 4,096 x 4,096 | 16 | 268,435,456 |
| W_gate and W_up, per layer | 6,144 x 4,096 | 8 | 201,326,592 |
| W_down, per layer | 4,096 x 6,144 | 4 | 100,663,296 |
| two RMSNorm gains, per layer | 2 x 4,096 | 4 | 32,768 |
| total |  |  | 646,025,216 |

Condition encoder (the 3-tap convolution runs along the frame axis and has a bias):

| tensor | shape (outputs x inputs) | copies | parameters |
|---|---|---:|---:|
| softmax logits over the 8 hidden vectors of a frame | 8 | 1 | 8 |
| scale | 1 | 1 | 1 |
| convolution weight, 3 taps along the frame axis | 2,048 x 4,096 x 3 | 1 | 25,165,824 |
| convolution bias | 2,048 | 1 | 2,048 |
| total |  |  | 25,167,881 |

Flow-matching DiT, one block listed once and then repeated 36 times in the full table:

| tensor | shape (outputs x inputs) | copies | parameters |
|---|---|---:|---:|
| attention W_q, W_k, W_v, W_out (no bias) | 2,048 x 2,048 | 4 | 16,777,216 |
| two LayerNorms, each a gain and a bias | 2 x 2 x 2,048 | 1 | 8,192 |
| up and gate projections, one stored matrix | 16,384 x 2,048 | 1 | 33,554,432 |
| their bias | 16,384 | 1 | 16,384 |
| down projection | 2,048 x 8,192 | 1 | 16,777,216 |
| its bias | 2,048 | 1 | 2,048 |
| total |  |  | 67,135,488 |

The whole denoiser, the block rows multiplied by 36:

| tensor | shape (outputs x inputs) | copies | parameters |
|---|---|---:|---:|
| random-Fourier weights of the time embedding | 128 | 1 | 128 |
| time MLP, first layer weight (256 sin and cos inputs) | 2,048 x 256 | 1 | 524,288 |
| its bias | 2,048 | 1 | 2,048 |
| time MLP, second layer weight | 2,048 x 2,048 | 1 | 4,194,304 |
| its bias | 2,048 | 1 | 2,048 |
| input convolution, 1 tap, 2,304 to 2,304 (no bias) | 2,304 x 2,304 | 1 | 5,308,416 |
| input projection, 2,304 to 2,048 (no bias) | 2,048 x 2,304 | 1 | 4,718,592 |
| per block: attention W_q, W_k, W_v, W_out (no bias) | 2,048 x 2,048 | 144 | 603,979,776 |
| per block: two LayerNorms, each a gain and a bias | 2 x 2 x 2,048 | 36 | 294,912 |
| per block: up and gate projections, one stored matrix | 16,384 x 2,048 | 36 | 1,207,959,552 |
| per block: their bias | 16,384 | 36 | 589,824 |
| per block: down projection | 2,048 x 8,192 | 36 | 603,979,776 |
| per block: its bias | 2,048 | 36 | 73,728 |
| output projection, 2,048 to 128 (no bias) | 128 x 2,048 | 1 | 262,144 |
| output convolution, 1 tap, 128 to 128 (no bias) | 128 x 128 | 1 | 16,384 |
| total |  |  | 2,431,905,920 |

Vocoder, one 64-channel stream (the two ears share the weights). The multiply-adds are per position of the layer's own resolution, and the last column is that resolution as a multiple of the latent rate; a transposed convolution is counted at its input positions:

| layer | parameters, as a sum of tensors | parameters | multiply-adds a position | positions per latent step |
|---|---|---:|---:|---:|
| input projection (1x1) | 1,024 x 64 + 1,024 | 66,560 | 65,536 | 1 |
| first convolution (k=7) | 1,024 x 1,536 x 7 + 1,536 | 11,011,584 | 11,010,048 | 1 |
| block 0: Snake | 1,536 alphas | 1,536 | 0 | 1 |
| block 0: transposed convolution (k=16, stride 8) | 1,536 x 768 x 16 + 768 | 18,875,136 | 18,874,368 | 1 |
| block 0, unit 0 (dilation 1) | 2 x 768 + (768^2 x 7 + 768) + (768^2 + 768) | 4,721,664 | 4,718,592 | 8 |
| block 0, unit 1 (dilation 3) | 2 x 768 + (768^2 x 7 + 768) + (768^2 + 768) | 4,721,664 | 4,718,592 | 8 |
| block 0, unit 2 (dilation 9) | 2 x 768 + (768^2 x 7 + 768) + (768^2 + 768) | 4,721,664 | 4,718,592 | 8 |
| block 1: Snake | 768 alphas | 768 | 0 | 8 |
| block 1: transposed convolution (k=16, stride 8) | 768 x 384 x 16 + 384 | 4,718,976 | 4,718,592 | 8 |
| block 1, unit 0 (dilation 1) | 2 x 384 + (384^2 x 7 + 384) + (384^2 + 384) | 1,181,184 | 1,179,648 | 64 |
| block 1, unit 1 (dilation 3) | 2 x 384 + (384^2 x 7 + 384) + (384^2 + 384) | 1,181,184 | 1,179,648 | 64 |
| block 1, unit 2 (dilation 9) | 2 x 384 + (384^2 x 7 + 384) + (384^2 + 384) | 1,181,184 | 1,179,648 | 64 |
| block 2: Snake | 384 alphas | 384 | 0 | 64 |
| block 2: transposed convolution (k=8, stride 4) | 384 x 192 x 8 + 192 | 590,016 | 589,824 | 64 |
| block 2, unit 0 (dilation 1) | 2 x 192 + (192^2 x 7 + 192) + (192^2 + 192) | 295,680 | 294,912 | 256 |
| block 2, unit 1 (dilation 3) | 2 x 192 + (192^2 x 7 + 192) + (192^2 + 192) | 295,680 | 294,912 | 256 |
| block 2, unit 2 (dilation 9) | 2 x 192 + (192^2 x 7 + 192) + (192^2 + 192) | 295,680 | 294,912 | 256 |
| block 3: Snake | 192 alphas | 192 | 0 | 256 |
| block 3: transposed convolution (k=4, stride 2) | 192 x 96 x 4 + 96 | 73,824 | 73,728 | 256 |
| block 3, unit 0 (dilation 1) | 2 x 96 + (96^2 x 7 + 96) + (96^2 + 96) | 74,112 | 73,728 | 512 |
| block 3, unit 1 (dilation 3) | 2 x 96 + (96^2 x 7 + 96) + (96^2 + 96) | 74,112 | 73,728 | 512 |
| block 3, unit 2 (dilation 9) | 2 x 96 + (96^2 x 7 + 96) + (96^2 + 96) | 74,112 | 73,728 | 512 |
| final Snake and convolution (k=7) | 96 alphas + 96 x 7 + 1 | 769 | 672 | 512 |
| total, one stream |  | 54,157,665 |  |  |
<!-- /worked -->

The totals of those tables are the rows of the next one.

<!-- worked: music.parameter_counts -->
Parameters of the five components, the totals of the tensor tables above (norm gains, biases and the DiT's time embedding included), and the bytes of a 32-bit copy in gigabytes ($10^9$ bytes). The Global LLM is Volume VI's count of Qwen3 8B, 8,190,735,360 for a vocabulary of 151,936, with the two tables enlarged to 200,000 ids: 8,190,735,360 + 2 x 4096 x (200,000 - 151,936) = 8,584,475,648.

| component | parameters | GB at 32 bits | share (%) |
|---|---:|---:|---:|
| Global LLM (Qwen3 8B decoder, 200,000-id vocabulary, untied head) | 8,584,475,648 | 34.34 | 73.1 |
| RVQ depth decoder (4 layers, width 4096) | 646,025,216 | 2.58 | 5.5 |
| condition encoder (8 layer logits, one 3-tap convolution 4096 to 2048) | 25,167,881 | 0.10 | 0.2 |
| flow-matching DiT (36 blocks of width 2048) | 2,431,905,920 | 9.73 | 20.7 |
| vocoder (convolutional decoder with Snake activations, one 64-channel stream per ear) | 54,157,665 | 0.22 | 0.5 |
| all five | 11,741,732,330 | 46.97 | 100.0 |
<!-- /worked -->

Three quarters of the parameters are the language model, and a fifth is the denoiser, which is where almost all of the arithmetic goes. Two figures in the crate's own ledger, 19 billion parameters in all and 123 million for the vocoder, do not survive the count, which gives 11.7 billion and 54.2 million, and the sizes of the files agree with the count. The language model's 8.58 billion parameters are 17.2 gigabytes at 16 bits, the size that the ledger records for the language-model directory, and the vocoder's 54.2 million at 32 bits are 206.6 mebibytes, the 207 that `crates/minimaxmusic3/src/memory.rs` gives for the vocoder directory. A figure that no code checks is a claim, and a count that reproduces the size of the files is evidence.

The arithmetic of making a song is the second ledger. It counts as Chapter 6 did, with every matrix product once for each row it is applied to, and takes a prompt of 200 tokens and a song of 4 minutes. The terms come first, as formulas in the shapes above.

<!-- worked: music.work_layout -->
The terms of the count, as formulas in the shapes of the tables above. A product is counted once for every row it is applied to; norms, activations, embedding lookups and the sampling are left out. The output head is applied once per frame, to the frame's row, and so the prompt's rows and the loop's extra first iteration (the frame that the loop samples from the end of the prompt and does not keep) are counted as matrices and attention only, not as one more head and one more depth-decoder frame.

| stage | term | multiply-adds |
|---|---|---:|
| Global LLM | matrices of one row, one branch: 36 x (2 x 4,096 x 4,096 + 2 x 4,096 x 1,024 + 3 x 4,096 x 12,288) | 6,945,767,424 |
| Global LLM | output head, one row, one branch: 200,000 x 4,096 | 819,200,000 |
| Global LLM | one frame, both branches: 2 x (matrices + head) | 15,529,934,848 |
| Global LLM | attention of the 6,000 frames, both branches: 2 x 2 x 36 x 4,096 x the sum over t = 1 to 6,000 of the 200 + t positions that frame t reads | 11,326,390,272,000 |
| Global LLM | the 200 prompt rows, both branches: matrices and attention (row p reads p keys), no head | 2,790,162,432,000 |
| depth decoder | one position, one branch: 4 x (4 x 4,096^2 + 3 x 4,096 x 6,144) | 570,425,344 |
| depth decoder | the 8 positions of a frame, both branches | 9,126,805,504 |
| depth decoder | seven output heads, both branches: 2 x 7 x 4,096 x 1,024 | 58,720,256 |
| depth decoder | input projection, applied to 9 rows (each branch's hidden vector, the semantic code's embedding and six residual ones, the last seven shared by the branches): 9 x 4,096^2 | 150,994,944 |
| depth decoder | attention over the 8 positions, both branches: 2 x (4 x 2 x 4,096 x (1 + 2 + ... + 8)) | 2,359,296 |
| depth decoder | one frame, the four terms above | 9,338,880,000 |
| DiT | attention projections, one evaluation: 36 x 690 x 4 x 2,048^2 | 416,746,045,440 |
| DiT | feed-forward, one evaluation: 36 x 690 x 3 x 8,192 x 2,048 | 1,250,238,136,320 |
| DiT | attention scores and weighted sums: 36 x 2 x 690^2 x 2,048 | 70,203,801,600 |
| DiT | glue, on the 689 latent steps: 689 x (2,304^2 + 2,304 x 2,048 + 2,048 x 128 + 128^2), plus the time embedding 256 x 2,048 + 2,048^2 once | 7,105,232,896 |
| vocoder | one latent step, both streams: 2 x the sum over the layers of (multiply-adds a position x positions per step) | 1,608,286,208 |
<!-- /worked -->

The stages add up as follows.

<!-- worked: music.work_per_stage -->
Multiply-adds to make a song of 240 seconds (6,000 frames, 59 chunks), counted as in the speech chapter and by the terms of the table above: every matrix product once per row, attention scores and weighted sums included, the 200 rows of the assumed prompt included, norms and elementwise functions left out. One DiT evaluation over a full chunk (length 689, 690 rows with the timestep token) is 1.744 tera-multiply-adds; one autoregressive frame costs 15.53 giga for the language model's two branches (without attention) and 9.34 for the depth decoder's.

| stage | multiply-adds (T) | per second of music (T) | share (%) |
|---|---:|---:|---:|
| Global LLM, two branches (matrices, head and attention) | 107.3 | 0.447 | 1.68 |
| depth decoder, two branches | 56.0 | 0.233 | 0.88 |
| DiT, 30 steps of 2 evaluations per chunk | 6,174.8 | 25.728 | 96.43 |
| vocoder, every chunk decoded in full | 65.4 | 0.272 | 1.02 |
| total | 6,403.5 | 26.681 | 100.00 |

One second of the 16-codebook speech synthesiser of Chapter 6, its codec decoder included, costs 52.7 giga multiply-adds; 26.681 tera is 507 times that. The 6,404 tera multiply-adds of the song are 12.81 peta floating-point operations, a multiply-add being two.
<!-- /worked -->

The denoiser is 96 percent of it. One second of finished music costs about 26.7 tera multiply-adds, 507 times as much as a second of speech from the 16-codebook synthesiser of the last chapter, and almost all of the difference is one network evaluated 60 times a chunk. The next table builds the denoiser's number from the factors, in order.

<!-- worked: music.work_factors -->
From one DiT evaluation to the denoiser's work per second of finished music (multiply-adds, in tera). The chunk of 8 seconds is paid once for every 4 seconds of song, so each second is denoised in two chunks, and the second factor of 2 of the guidance is paid on top of the 30 steps.

| step of the count | factor | running total (T) |
|---|---:|---:|
| one evaluation of one chunk | 1 | 1.744 |
| 30 steps | 30 | 52.329 |
| 2 evaluations a step (guidance) | 2 | 104.658 |
| the chunk's 8 seconds cover 4 new ones, so per second of music: / 4 | 0.25 | 26.164 |
<!-- /worked -->

The factor of 30 is the number of Euler steps and the factor of 2 the guidance. The last row divides by the 4 seconds of new song that each chunk adds; since the chunk is 8 seconds long, two chunks cover every second, which is the doubling that the overlap costs. There is no way to run the chunks at once, as a tiling of an image can run its windows at once: chunk $k+1$ needs the final latents of chunk $k$, so the 74 chunks of a five-minute song are denoised one after another. The only parallelism the crate finds is that of the two guidance branches of a stage, which run on two cards when the machine has two (`crates/minimaxmusic3/src/devplan.rs`) and give bit-identical results, a property that a test of the crate checks by comparing the bits.

The number of operations also gives a floor for the time, and only a floor.

<!-- worked: music.roof_floor -->
<!-- perf-number: historical measurement, commit 3c5f30d77 -->
Divided by the roof of the card of the crate's bench, 10.517 tera operations a second, the 12.81 peta operations of a four-minute song take at least 1,218 seconds, 20.3 minutes, and a real run is slower than its roof.
<!-- /worked -->

What a card must hold is a different question, and it has a different answer.

<!-- worked: music.stage_memory -->
What one generation asks of a card, stage by stage. The stages run one after the other and drop their device state between them, so the need is the largest of the three, not the sum. A stage's weights are the closed form of its shapes where one exists; the three figures marked measured are measurements of the crate's ledger and cannot be derived from a configuration.

| stage | bytes | GB | source |
|---|---:|---:|---|
| autoregressive: one Global LLM instance (int8 linears, fp32 tables) and the depth decoder | 16,087,553,408 | 16.09 | 6.95 GB measured for the linears |
| denoise: the DiT's block stack in fp32 plus a margin | 10,066,329,600 | 10.07 | margin of 384 MiB measured |
| vocoder: one full 689-step chunk | 12,859,736,064 | 12.86 | 12264 MiB measured |

The first row, term by term:

| part | bytes | how |
|---|---:|---|
| Global LLM linears, int8 | 6,950,000,000 | measured, 6.95 GB (three digits) |
| Global LLM, two tables of 200,000 x 4,096 in fp32 | 6,553,600,000 | 2 x 200,000 x 4,096 x 4 bytes |
| depth decoder, fp32, matrices and tables only | 2,583,953,408 | 4 x (4 x 4,096^2 + 3 x 4,096 x 6,144) + 2 x 7 x 1,024 x 4,096 + 4,096^2 + 16 x 4,096, times 4 bytes |
| autoregressive stage | 16,087,553,408 | the three rows above |

The closed form of the language model's int8 matrices, the 36 layers of 192,937,984 weights at one byte each, is 6,945,767,424 bytes; the measured 6.95 GB exceeds it by 4,232,576 bytes, 0.06 percent, which is inside the rounding of a figure given to three digits and so leaves no room to say where per-row scales or norm gains live. The depth decoder's closed form omits its 36,864 norm gains. The denoise row is the block stack without biases, 36 x (4 x 2,048^2 + 3 x 8,192 x 2,048) = 2,415,919,104 weights at 4 bytes, plus the margin of 384 MiB = 402,653,184 bytes.

The autoregressive stage runs two instances of the language model, one for each guidance branch; 32,175,106,816 bytes (32.18 GB) is what one card would have to hold if both lived on it, against the 24 GiB (25.77 GB) of the cards of the validation machine.
<!-- /worked -->

The stages run in sequence and drop their device state between them (`crates/minimaxmusic3/src/generate.rs`), so the need is the largest of the three, the language-model stage, and not the sum. One instance of the language model keeps its linears in 8 bits and its two tables of 200,000 by 4,096 in 32 bits, 3.28 gigabytes each, which is why two instances do not fit on a card of 24 gibibytes and why the two guidance branches of the language model are placed on two different cards when there are two. What each stage leaves for the next is small except for one thing.

<!-- worked: music.handoff_sizes -->
What the stages hand to one another for a song of 300 seconds (7,500 frames, 74 chunks): the count of numbers, bytes per number as the host holds them, and the total. The codes themselves are not on the list: the loop writes them, feeds them back and forgets them, and returns only the hidden vectors.

| what | numbers | bytes each | bytes | megabytes |
|---|---:|---:|---:|---:|
| hidden vectors kept for the condition encoder | 245,760,000 | 4 | 983,040,000 | 983.0 |
| latents of every chunk, held until the vocoder runs | 6,526,208 | 4 | 26,104,832 | 26.1 |
| the finished waveform, 32-bit floats, two channels | 26,460,000 | 4 | 105,840,000 | 105.8 |
<!-- /worked -->

## What can be trained and what cannot

Nearly every component of the crate has a backward pass, and no component is trained on music. Both statements are true, and the difference is the subject of this section. The vocoder has a trainer (`crates/minimaxmusic3/src/train.rs`) whose gradients are checked against finite differences for every named parameter, and low-rank adapters on its convolutions; the depth decoder has a hand-derived backward and adapters on its seven linear maps per layer; the denoiser has a trainer in `crates/minimaxmusic3/src/dit_train.rs`, device-resident and built from the same block builders as its forward, with adapters on the six linear maps per block; and the language model has an objective that is ordinary cross-entropy restricted by position to the audio codes that follow the prompt (`global_llm::audio_code_batch` gives weight 0 to every position whose target lies inside the prompt and 1 from the first audio code on). The adapters are small next to the components.

<!-- worked: music.adapter_sizes -->
Parameters of the low-rank adapters of the three components that have them: the number of adapted matrices, and for a rank $r$ each matrix of $m \times n$ weights gets $r(m + n)$ trainable numbers. A convolution is adapted as the matrix of its outputs by its inputs times taps ($m = c_{\mathrm{out}}$, $n = c_{\mathrm{in}} k$), and a transposed convolution as $m = c_{\mathrm{in}}$, $n = c_{\mathrm{out}} k$; $r(m + n)$ is the same either way. The DiT adapts the 6 maps of each of its 36 blocks (4 attention maps, the up-and-gate matrix and the down matrix: 216), the depth decoder the 7 maps of each of its 4 layers (28), and the vocoder 31 convolutions: the input projection, the first convolution, per block one transposed convolution and three units of two, and the final convolution. The percentage is against the component's own parameters.

| component | adapted matrices | rank | adapter parameters | % of the component |
|---|---:|---:|---:|---:|
| DiT | 216 | 4 | 6,488,064 | 0.27 |
| DiT | 216 | 16 | 25,952,256 | 1.07 |
| depth decoder | 28 | 4 | 1,015,808 | 0.16 |
| depth decoder | 28 | 16 | 4,063,232 | 0.63 |
| vocoder | 31 | 4 | 307,588 | 0.57 |
| vocoder | 31 | 16 | 1,230,352 | 2.27 |
<!-- /worked -->

A rank-16 adapter of the denoiser is 26 million numbers, 1.07 percent of the component. The three adapter trainers were the same code three times until commit `25705e4dc` made them one pair type of `model::lora`, and the same commit pinned the batch that their convergence tests use, because a test that drew a new batch each run reached its bar or not by luck.

What is shown, and how, is also what is missing. Each trainer is gradient-checked and shown to drive its loss down on a fixed batch. The vocoder's reconstruction loss is a mean squared error against a fixed target, which the crate's own comments call what the gradient check and the overfit test need; a real run adds an adversarial term. The denoiser's trainer takes a fixed latent, a condition, a time and a target and has no sampler of times and noises and no velocity target built from a clean latent. The crate has no code that maps audio to the latents that the denoiser denoises or to the semantic codes that the language model writes, so a caller who wants to train on music has to bring both. The discriminator of Volume VIII, Chapter 4 (commit `205d2c487`) is a single-resolution one over a magnitude spectrogram, with its adversarial and feature-matching losses and a gradient that is checked through the Fourier transform to the waveform, and since commit `40fc9a6b1` a joint loop trains the vocoder against a multi-resolution version of it: one network per STFT size at 512, 1024 and 2048 with a hop of a quarter of the size, stepped on real against generated audio with its own optimiser once `disc_start_step` is reached, while the vocoder trains on the reconstruction plus the least-squares adversarial and feature-matching terms, whose gradient on the waveform enters the vocoder's backward at its output (`crates/minimaxmusic3/src/adversarial.rs`). The defaults are the published GAN-vocoder recipe's, feature matching weighted 2 and adversarial and reconstruction 1, because the model's authors publish no vocoder training code. The joint generator gradient is compared with central differences under a bound derived from the rounding of the arithmetic, and a check that drops the adversarial terms fails that comparison. The header of `train.rs`, which commit `498b722b3` rewrote after an earlier version described a return value that the code did not have, states where the adversarial terms attach. The language model is imported for inference at 8 bits and has no backward at real scale; its objective was shown on a tiny configuration. The training surface is a library, with no command that trains. The served model has one action, `generate`, with five parameters (the lyrics, the caption, a duration, a number of steps and a seed), and the two guidance scales are not among them. I found nothing in the repository that measures whether a generated song is good, or whether its words are the lyrics.

The seed is worth a sentence. A run that gives no seed draws one, a seed that is not a non-negative integer is refused and not clamped, and the outcome reports the seed it used so that a song can be made again (commit `2f9f89cac`); the language model's sampler and the denoiser's noise (one seed per chunk, the run's seed plus the chunk's index plus 1) both follow from it.

## A first song, and three claims about it

The crate was written component by component between 20 and 24 August 2026, from the diffusers reference, starting from the skeleton of commit `dedb7e6d4`, and the autoregressive loop (commit `6401676f5`), the chunked denoiser (`a766c26a2`) and the stitcher (`e45a52a6e`) arrived in one day. The messages of the first two say what their tests were: wiring tests at toy size, because no numerical reference exists for the composition. What was checked against the reference was each component alone, as the ledger records, and the end-to-end gate, `crates/minimaxmusic3/tests/it/e2e_short_generation.rs`, asserts only that a clip of about a second is non-empty, finite, stereo and more than half a second long.

On 24 August the port made its first song, and the message of commit `594a3db3c` is a list of why it had not made one before. The end-to-end gate had composed the five stages inline and so tested a path that no caller used. The language model had been built in the shape of a training step, with room for activations it never used, and the allocation went past the card. The decode path bound a whole table of 3.28 gigabytes where the batched path split it. And the denoiser dispatched every matrix product through the one-thread-per-output reference kernel. The first and the last generalise, and the last came with a measurement:

<!-- perf-number: historical measurement, commit 594a3db3c -->
the commit's message records that a denoising step fell from 37.4 to 1.21 seconds on its test card, one Tesla P40, at the chunk length of the short end-to-end test. The next commit that measured, `3c5f30d77`, is the better story, because it corrected that one.

<!-- perf-number: historical measurement, commit 3c5f30d77 -->
It profiled the denoiser on a card whose roof it measured at 10,517 giga-operations a second and 287.5 gigabytes a second, at the real chunk length of 689 steps, and found that the three attention kernels, the scores, the softmax and the weighted sum, were 75.3 percent of the device time at 0.5 to 4.9 percent of their memory roof, while the matrix products beside them ran at 42.1 percent of the 32-bit roof. The ledger had said that the denoiser was compute-bound and that its matrix products should be optimised first. The table above says the same from the other side: the attention proper is 4 percent of the multiply-adds, so what takes three quarters of the time must be doing so by running badly and not by doing a lot. The fix was already in the repository, as a family of fused attention kernels that the denoiser had never registered (commit `57d482653`; the pipeline list in `crates/minimaxmusic3/src/dit.rs` now names all four). The message of `3c5f30d77` also says that the figure of 37.4 seconds a step had been measured at the short chunk length of the end-to-end test and not at 689, where 1.21 seconds would have been below the floor of the matrix products, so the figure that had been repeated could not have been the figure it was taken for.

A second story is about a claim and its retraction. When a two-chunk run died in the vocoder for lack of memory, commit `7525154b5` explained it by a statement about the graphics library that it asserted as if measured: that a device's memory is not returned when the handle is dropped. It was not measured. Commit `52b76be6b` retracted it, citing the library's own source and the session's own numbers against it, two chunks in a row peaking at the level of one and the first card falling to 15 mebibytes while the vocoder ran on the second, and it recorded the actual cause of the same-card failure as unknown, with the probe that would settle it. The fix, placing the two stages on two cards, still stands, and the comment in `generate.rs` now says that the mechanism is not established. The lesson, in the commit's own words, is that a wrong durable finding is worse than none.

A third concerns a gate. The parity of the vocoder had been asserted by cosine, which is blind to a uniform scale, and commit `8835c9410` adds a ceiling on the relative error and shows by mutation that a gain of 0.1 percent on the output leaves the cosine at 1.000000000 and is caught only by the new bound. That is knowledge #2, and the message says that this was the last stage that still had the hole.

## Check your understanding

1. A reader says that the denoiser reads the language model's codes. What does it read instead, how many numbers a second is that, and what is lost if the interface is described as 2,100 bits a second?
2. A request for 30 seconds of a song asks for 750 frames. The model writes the end token at frame 410. How many chunks does the denoiser run, and what is the song's length in samples per channel?
3. The nearest-neighbour map from 5 frames to 17 latent steps reads frame 0 four times and frame 1 three times. Why are the repeats not equal?
4. A friend suggests running the 74 chunks of a five-minute song at once on 74 cards. Why can the crate not do that, and what does its two-card arrangement split instead?
5. In the toy, the independent run jumps at the seam and the held run does not. Does that show that the held run is better at every position? Where does its remaining defect sit?
6. The crop of a chunk is 86 steps and the vocoder's reach is 10. Name one reason that the crop could still be 86 and say what experiment would test it.
7. A commit reports a smooth waveform at a boundary four seconds into a song and concludes that no crossfade is needed. Using the geometry of the chunks, say where the seam is and what such a measurement could and could not show.
8. The attention of the denoiser is 4 percent of its multiply-adds and was three quarters of its time. What does that say about reading a count of operations as a prediction of time?

## Summary

Music generation in brain's crate separates the three lengths of a song. A language model of 8.6 billion parameters writes the whole song one position per frame, 25 frames a second, drawing a semantic code from 16,384 and, through a small depth decoder, seven residual codes from 1,024 each, 2,100 bits a second, under two prompts, the second a copy of the first with its text replaced by a placeholder, blended with a guidance of 1.5 and restricted to the conditional branch's top 50. What it hands on is not the codes but eight hidden vectors of 4,096 per frame, 819,200 numbers a second, which a condition encoder mixes and lays onto a latent clock of $44100/512$ steps a second by repeating. A flow-matching transformer of 2.4 billion parameters denoises the latents in chunks of 200 frames, 689 steps, at 30 steps of two evaluations each, with the time as the first row of its sequence, and each chunk after the first begins with 172 steps taken from the one before it and held at the level of noise of the step, while the chunk's hop is 100 frames. A vocoder of 54 million parameters turns each chunk into two ears of samples, and the stitcher crops 86 steps off the left and 258 off the right of each, which leaves 345 steps for every chunk with a neighbour on each side. Counted, a song of four minutes is 6.4 peta multiply-adds, of which the denoiser is 96 percent, because every second of song is denoised twice (overlap), twice again (guidance) and 30 times (steps), and the chunks cannot be run at once. The crate trains nothing on music: every component has a gradient-checked backward and a low-rank adapter, and no encoder, tokenizer or flow-matching objective to use them with; the vocoder's trainer includes a joint loop against multi-resolution spectrogram discriminators. The next chapter takes the same ingredients, a language model that speaks and a decoder that listens, and puts them in one model whose input is sound, pictures and text.

## References

- Défossez, Copet, Synnaeve and Adi, *High Fidelity Neural Audio Compression*, 2022.
- Kumar, Seetharaman, Luebs, Kumar and Kumar, *High-Fidelity Audio Compression with Improved RVQGAN*, 2023.
- Lipman, Chen, Ben-Hamu, Nickel and Le, *Flow Matching for Generative Modeling*, 2022.
- Peebles and Xie, *Scalable Diffusion Models with Transformers*, 2023.
- Ho and Salimans, *Classifier-Free Diffusion Guidance*, 2021.
- Bar-Tal, Yariv, Lipman and Dekel, *MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation*, 2023.
- Avrahami, Lischinski and Fried, *Blended Diffusion for Text-driven Editing of Natural Images*, 2022.
- Ziyin, Hartwig and Ueda, *Neural Networks Fail to Learn Periodic Functions and How to Fix It* (the Snake activation), 2020.
- Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang and Chen, *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.

## Exercises

1. **[Calculate]** A song is 137 seconds long. How many frames is that, how many latent steps does the condition encoder produce for all of them, how many chunks does the denoiser run, and how many latent steps and samples per channel does the stitched song have? By how many samples does it differ from the nominal length?
2. **[Calculate]** For a song of 350 frames list the start, the number of frames, the latent length and the kept stretch of each chunk, and the stitched length in samples.
3. **[Calculate]** With 12 Euler steps instead of 30, write $t_i$ for $i = 0, \ldots, 11$ and the two blend weights at step 5.
4. **[Calculate]** A song of four minutes with a prompt of 200 tokens: how many numbers and how many gigabytes (32-bit) does the attention cache of one branch of the language model hold at the end, and how many denoiser evaluations does the song need?
5. **[Calculate]** How many trainable numbers does a rank-8 adapter on all six linear maps of every block of the denoiser have, and what is that as a percentage of the denoiser?
6. **[Calculate]** A residual unit of the vocoder with dilation 9 and 7 taps widens the receptive field at its own resolution by 27 positions on each side. Convert that to latent steps for the unit at the last block (whose resolution is 512 times the latent rate) and for the first block (8 times).
7. **[Implement]** Reproduce `music.frame_arithmetic`, `music.integer_division_trap` and `music.song_lengths` from the description in the first two sections.
8. **[Implement]** Reproduce `music.chunk_geometry` and `music.length_drift`: the starts, the crops, the kept stretches and the stitched length of a song of any number of frames.
9. **[Implement]** Reproduce `music.chunk_flow_toy`: the stream named there, the draws in order, the covariance, the exact velocity, and the two runs, to the digits printed.
10. **[Implement]** Reproduce `music.ar_sampling_toy` and `music.prompt_contract`: the two-stage restriction and the lyrics rewrite, including the line whose words are dropped.
11. **[Investigate]** Run the toy of `music.chunk_flow_trials` at hop 8 with the correlation 0.5, 0.9 and 0.99. How does the boundary excess of independent chunks depend on it, and what do the two holds remove? What does that say about the joins of a real denoiser, whose latents are strongly correlated from one step to the next?
12. **[Investigate]** In `music.chunk_flow_trials` the clean hold gives a smaller ratio than the renoised one. Explain why the clean hold is not a legal input of the exact velocity, and say what in the numbers still shows a join after it.

## Key terms

- **music generation**: the production of a long stereo signal from a description of its style and its lyrics by a chain of models that each see a different length of the song, a language model that writes it as codes, a denoiser that paints it as latents and a decoder that turns the latents into samples. <!-- concept: music-generation -->
- **long-form audio generation**: the production of audio longer than any one component can attend to at once, by generating it in overlapping windows, each started from the end of the last, and cropping and joining the results. <!-- concept: long-form-audio-generation -->

## Answers

**Recall questions.**

1. 12.5 columns of 16 codes is 200 integers a second; with 11 bits for a code of 2048 entries, 2,200 bits a second.
2. A position per code would make the sequence 16 times longer and the attention quadratically more expensive, so the Talker adds the 16 rows into one. The price is that a sum of rows is a lossy way to present a column, and that the codes of a column are not drawn together.
3. The region to keep is replaced by the source renoised to the noise fraction the step has reached, $(1 - \sigma_{i+1})\mathbf{x}_0 + \sigma_{i+1}\boldsymbol\varepsilon$, and at the last step by the source itself.
4. $u_g = u_{\mathrm{un}} + g(u_c - u_{\mathrm{un}})$, and a step costs two evaluations, one conditional and one unconditional, unless the two are folded into one pass.
5. One step is $x_{t'} = x_t + (t' - t)\,u(x_t, t)$ with $t$ running from noise at 0 to data at 1; for a Gaussian of mean $m$ and standard deviation $\varsigma$ the velocity is $m + \frac{t\varsigma^2 - (1 - t)}{(1 - t)^2 + t^2\varsigma^2}(x - tm)$.

**Check your understanding.**

1. It reads eight vectors of 4,096 numbers a frame, mixed into one by a learned softmax: 819,200 numbers a second, 26,214,400 bits a second at 32 bits, 12,483 times the bits of the codes. The codes are a 16,384-way label and seven 1,024-way labels, and the hidden vectors are what the model computed before choosing them, so describing the interface as 2,100 bits leaves out nearly all of what is handed on.
2. The song is 410 frames, so the denoiser runs the chunks that start at 0, 100, 200 and 300, which is 4 chunks (the last has 110 frames). The kept latents are 431 + 345 + 345 + (378 - 86 = 292) = 1,413 steps, 723,456 samples per channel; the nominal length of 410 frames is 723,240.
3. 17 latent steps over 5 frames is 3.4 steps a frame, and a number of repeats is an integer, so the rule $\lfloor i \cdot 5/17 \rfloor$ gives the extra repeat to some frames and not to others: frames 0 and 2 are read four times, the others three.
4. Chunk $k+1$ starts from the final latents of chunk $k$, so each chunk waits for the one before it. What the two cards split is the two guidance branches of each evaluation, which do not depend on each other.
5. No. It removes the jump at the seam, where the two chunks carry the same latents, and moves the discontinuity to the join at the end of the held prefix, where the new chunk's own latents begin; in the toy it is smaller there and not zero.
6. The attention of the denoiser at the edge of a chunk has fewer steps to look at, so the edge steps may be worse than the middle ones whatever the vocoder does. To test it, generate the same song with crops of 10 and of 86 and compare the stretch between them, by ear or by a spectral distance; the repository does not do this.
7. The seam of the first two chunks is at step 431 of the first chunk, 5.0039 seconds into the song. Four seconds is where the second chunk starts, and its first 86 steps are cropped away, so nothing at four seconds in the stitched song is a seam; a measurement there says nothing about one. The mechanism that the commit names, replacement in the latent domain, is supported; the place is not.
8. A count of operations says how much arithmetic there is and nothing about how well a kernel runs it. A term that is small in the count can be large in time if its kernel runs at a small fraction of the device's roof: the count predicts a floor, and the roof says how far a run is from it.

**Exercises.**

1. 3,425 frames; the latent length is 11,800; 34 chunks (starts 0 to 3,300 in steps of 100); the stitched song has 11,815 steps, 6,049,280 samples per channel, against a nominal 3,425 x 1,764 = 6,041,700, which is 7,580 more.
2. Starts 0, 100 and 200. Chunk 0: 200 frames, 689 steps, kept 0 to 431. Chunk 1: 200 frames, 689 steps, kept 86 to 431. Chunk 2: 150 frames, 516 steps, kept 86 to 516. The stitched length is $431 + 345 + 430 = 1{,}206$ steps, 617,472 samples, against a nominal 617,400.
3. $t_i = i/12$: 0, 0.0833, 0.1667, 0.25, 0.3333, 0.4167, 0.5, 0.5833, 0.6667, 0.75, 0.8333, 0.9167. At step 5, $t = 0.4167$, the noise weight is $1 - (1 - 10^{-6})\,5/12 = 0.583334$ and the data weight $0.416667$.
4. 6,000 frames and 200 prompt tokens give 6,200 positions, 457,113,600 numbers, 1.828 gigabytes at 4 bytes a number; 59 chunks times 60 evaluations is 3,540.
5. 12,976,128 numbers, 0.53 percent of 2,431,905,920.
6. At the last block, $27/512 = 0.05$ step; at the first block, $27/8 = 3.4$ steps. The reach of the whole stack, 10 steps, is dominated by the layers of low resolution.
7. to 10. See the worked blocks named.
11. For independent chunks the ratio is about 1.2, 2.5 and 8.8 at the correlations 0.5, 0.9 and 0.99, because a strongly correlated texture has a small ordinary step against which any jump is large. The renoised hold lowers them to about 1.1, 1.6 and 3.1, and the clean hold to about 1.1, 1.4 and 2.6 (hop 8, 100 songs of the hop-8 stream). Both remove a part of the excess and not all of it. A real latent is strongly correlated, so the join of a real denoiser is a place to look, and the repository does not.
12. At time $t$ the exact velocity is derived for a point $\mathbf{x}_t = (1 - t)\boldsymbol\varepsilon + t\mathbf{x}_1$, which is partly noise; a prefix held at its clean values is a point of the path at $t = 1$ inside a chunk at time $t < 1$, so the posterior is evaluated outside its domain, and the crate renoises it to stay on the path. The clean hold still has a ratio above 1 at every correlation (1.4 at 0.9, 2.6 at 0.99), so a join remains even when the prefix is known exactly from the first step, and the free part's own noise is the other source.
