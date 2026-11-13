<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Who Is Speaking

*How does a model capture the identity of a voice in a single vector?*

A voice-cloning request arrives with a reference recording and nothing else. Nobody says how long it is, what is said in it, or what room it was made in. The synthesiser that has to imitate the voice has one slot for the speaker, and the slot has a fixed width: brain's Qwen3-TTS crate, `crates/qwen3tts`, places the speaker in its prompt as one position of 1,024 numbers, and its CosyVoice crate, `crates/cosyvoice`, takes 192. Whatever arrives, a fifth of a second or a minute, must leave as exactly that many numbers. The first thing to see is how little that is the same from one clip to the next.

<!-- worked: speaker_embeddings.clip_geometry -->
Frames of the two front ends for clips of five lengths. Front end 1 (the 24 kHz log-mel): 1024-sample frames at a hop of 256 after a reflect pad of 384 on each side, 128 values per frame. Front end 2 (the 16 kHz filterbank): 400-sample frames at a hop of 160, no padding, 80 values per frame, then one convolution of kernel 5, stride 2 and padding 2 makes (T + 4 - 5) // 2 + 1 rows. 'Readable' applies the rules of the text: encoder 1 needs T >= 5 frames, encoder 2 needs at least 2 rows.

| seconds | samples at 24 kHz | frames T | readable | samples at 16 kHz | frames T | rows | readable |
|---|---:|---:|---|---:|---:|---:|---|
| 0.05 | 1,200 | 4 | no | 800 | 3 | 2 | yes |
| 0.1 | 2,400 | 9 | yes | 1,600 | 8 | 4 | yes |
| 1 | 24,000 | 93 | yes | 16,000 | 98 | 49 | yes |
| 2.5 | 60,000 | 234 | yes | 40,000 | 248 | 124 | yes |
| 7 | 168,000 | 656 | yes | 112,000 | 698 | 349 | yes |

The shortest readable clip is 1,280 samples (53.33 milliseconds) for encoder 1 and 720 samples (45.0 milliseconds) for encoder 2.
<!-- /worked -->

The two front ends of the table are two of those of the previous chapter, the 24 kHz log-mel of `crates/ecapatdnn` and the 16 kHz filterbank that `crates/cosyvoice` computes before `crates/campplus`. A 0.05 second clip makes 4 frames in one and 3 in the other, and a 7 second clip makes 656 and 698. Everything that reads the frames therefore has to work on any $T$, and has to end in one vector whose length does not depend on it. The last two columns hold a second warning: the clips have to be long enough for the encoder to read, and the first row of the table is shorter than that for one of the two. Why, the next section shows; for now it is enough that what follows is a design for a sequence of unknown length that must end in something of fixed size, and that its first job is to say who is speaking and not what is being said. Until this chapter was written, a clip below that length made `crates/ecapatdnn` index outside its frames and stop; commit `269794eb4` derives the length from the kernels and dilations (`SpeakerConfig::min_samples`) and zero-pads a shorter clip up to it, which gives a vector that is finite but says nothing about the speaker.

That second demand is the odd one. A log-mel spectrogram changes mostly with the words: change the sentence and the picture changes completely, change the speaker and it changes a little, in the spacing of the horizontal bands that vowels leave, the pitch, the tilt of the spectrum. A vector for a speaker has to ignore the large change and keep the small one. The recipe that works is the one of the faces chapter, an identity embedding trained so that two clips of one person point the same way and clips of two people do not; what differs here is the input, which is a sequence, and so the encoder must first look at the frames in small groups and then summarise them across the whole clip. The vector it produces is a **speaker embedding**: a vector of fixed length computed from a recording of any length so that recordings of one voice are close in cosine and recordings of different voices are not.

Speaker embeddings draw on five earlier ideas.

1. ([Volume IX, Chapter 2](02-spectrograms.md), front ends.) A clip of 16,000 samples goes through a filterbank with 400-sample frames at a hop of 160 and no padding. How many frames does it make? A clip of 24,000 samples goes through a log-mel front end with 1024-sample frames at a hop of 256 after a reflect pad of 384 on each side. How many?
2. ([Volume VII, Chapter 2](../v07-vision/02-convolution.md), two-dimensional convolution.) A convolution with 32 input channels, 32 output channels and a $3 \times 3$ kernel holds how many weights, and how many multiply-adds does it make over an output of $40 \times 100$ positions?
3. ([Volume VII, Chapter 2](../v07-vision/02-convolution.md), receptive field.) Three stride-1 layers of kernel 3 have dilations 1, 2 and 4. How many input positions does one output of the last layer depend on?
4. ([Volume VII, Chapter 11](../v07-vision/11-faces.md), identity embedding.) Two embeddings have cosine 0.7 and the threshold $\tau_{\mathrm{id}}$ is 0.5: a pair is called the same person when its cosine is at or above $\tau_{\mathrm{id}}$. Are these called the same person? When the threshold is lowered, which of the two kinds of error, a false rejection (a pair of one person called two) or a false acceptance (a pair of two people called one), becomes more frequent? Where the two rates are equal, as the last section defines, the common value is the equal error rate.
5. ([Volume III, Chapter 3](../v03-neural-networks/03-softmax.md), softmax.) What does a softmax over the four scores $(0, \ln 2, \ln 2, 0)$ return?

## A convolution that only moves in time

The frames of a front end make an array with a row for each of $C$ channels (the mel filters) and a column for each of $T$ frames. A picture would be convolved in both directions, but here the two axes are not alike: the rows are frequencies, which no neighbouring-position argument holds for in the same way, and the columns are moments. The layers of a speaker encoder therefore treat the rows as channels and slide only along time. One layer looks at $k$ frames at a time, spaced $d$ frames apart, and for every output channel $o$ computes

$$
y_{o,t} = b_o + \sum_{c=0}^{C-1}\sum_{m=0}^{k-1} W_{o,c,m}\; x_{c,\; t + d\,(m - (k-1)/2)} .
\tag{1}
$$

This is a convolution of kernel size $k$ and dilation $d$ in the sense of Volume VII, Chapter 2, with one spatial axis instead of two, and it has a name that comes from speech recognition: a **time-delay network** is a network of such layers, each unit seeing the layer below at several delayed moments, so that a stack of them sees an ever longer stretch of the signal. The name is older than any of the models in this chapter; the idea of weights shared across time, with delays, goes back to 1989, to a network that told phonemes apart. What a layer holds is $C \cdot k$ weights for each output channel, whatever $T$ is, and that is what lets one set of weights read a clip of any length.

A layer that gives an output for every input frame must read a few frames beyond each end. The encoders of this chapter pad the signal by $p = d(k-1)/2$ frames on each side, so that $T$ frames in give $T$ frames out, and they pad by reflection, the rule of the front-end chapter, in which frame $-i$ is a copy of frame $i$ and frame $T - 1 + i$ a copy of frame $T - 1 - i$. Reflection needs something to reflect: the frames $1, \dots, p$ must exist, so $p < T$. The next table works one layer by hand on two channels and six frames, with the padding of both kinds so that the difference shows.

<!-- worked: speaker_embeddings.tdnn_by_hand -->
One output channel from two input channels over T = 6 frames, kernel 3, dilation 2, so the same padding is 2 frames on each side. Weights of channel 0: (1.0, 0.0, -1.0); of channel 1: (0.5, 0.5, 0.5); bias 0.25. Output t is the bias plus the sum over both channels and the three taps (frames t - 2, t, t + 2 of the padded signal's own positions) of weight times value, accumulated in the order channel 0 tap 0 to 2, then channel 1.

| row | -2 | -1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| input, channel 0 |  |  | 1.00 | 3.00 | 2.00 | 5.00 | 4.00 | 2.00 |  |  |
| input, channel 1 |  |  | 0.50 | -1.00 | 1.00 | 0.00 | 2.00 | 1.00 |  |  |
| channel 0 reflect-padded | 2.00 | 3.00 | 1.00 | 3.00 | 2.00 | 5.00 | 4.00 | 2.00 | 4.00 | 5.00 |
| channel 1 reflect-padded | 1.00 | -1.00 | 0.50 | -1.00 | 1.00 | 0.00 | 2.00 | 1.00 | 2.00 | 0.00 |

Outputs for t = 0 to 5 (a pad outside the signal changes only the frames within 2 of an end):

| padding | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 |
|---|---:|---:|---:|---:|---:|---:|
| reflect padding | 1.500 | -2.750 | -1.000 | 1.250 | 0.750 | 0.750 |
| zero padding | -1.000 | -5.250 | -1.000 | 1.250 | 3.750 | 5.750 |
<!-- /worked -->

Only the outputs within two frames of an end differ, and they differ by a lot: the first output is $1.5$ with reflection and $-1.0$ with zeros, because zeros are not what the signal does at its edge and a layer with a negative tap makes the difference of an invented silence. Interior outputs are the same under both, so padding is a question of the edges, and edges matter more for a short clip, which is almost all edge. The rule $p < T$ gives the limit in the first table. The first convolution of encoder 1 has $k = 5$ and $d = 1$, so $p = 2$; the three blocks after it use dilations 2, 3 and 4 with kernel 3, so $p = 2, 3, 4$. The largest of them, $p = 4$, needs $T \ge 5$ frames, and with 1024-sample frames at a hop of 256 and a reflect pad of 384 on each side that is 1,280 samples, 53.33 milliseconds, the smaller of the two limits printed in the first table. Encoder 2 pads the time axis of its convolutions with zeros, and its limit comes from the pooling: the statistics divide by one less than the number of rows, which must therefore be at least 2.

This is the layer, written for the book's pseudocode.

```text
function time_delay_layer(X: array<real>[C, T], W: array<real>[O, C, k], b: array<real>[O], d: int)
        -> array<real>[O, T]:
    require k mod 2 == 1
    p = d * (k - 1) // 2
    require p < T
    Xp = zeros(real, [C, T + 2 * p])
    for c in 0 .. C:
        row = reflect_pad(X[c, :], p)
        for j in 0 .. T + 2 * p:
            Xp[c, j] = row[j]
    Y = zeros(real, [O, T])
    for o in 0 .. O:
        for t in 0 .. T:
            acc = b[o]
            for c in 0 .. C:
                for m in 0 .. k:
                    acc = acc + W[o, c, m] * Xp[c, t + d * m]
            Y[o, t] = acc
    return Y
```

The tap $m$ of output $t$ reads the padded position $t + dm$, which is the signal position $t + d(m - (k-1)/2)$ of equation (1). The first layer of encoder 1 has 128 input channels and 512 output channels at $k = 5$, which makes $128 \times 512 \times 5 = 327{,}680$ weights and 512 biases, and at 93.75 frames a second it makes 655,360 operations per frame. Count that against the table of costs below, where it is one of the cheaper stages.

## How far one output sees

One frame of the front end covers 1024 samples at 24 kHz, 43 milliseconds, and a vowel lasts about as long as three of them. A speaker's identity lives in longer-range structure than that, in the way vowels follow one another and in how a pitch moves, so what an encoder's output at one frame can see matters. That is the receptive field of Volume VII, Chapter 2: $r$ grows at each layer by $d(k-1)$ times the jump $j$, the input distance between neighbouring units, and the jump multiplies by every stride. The table applies it to both encoders and converts frames to samples with $(r-1)H + N$, the hop and frame length of the front end.

<!-- worked: speaker_embeddings.receptive_fields -->
Receptive field r in frames of the front end along time, by the recursion r = r + d (k - 1) j with the jump j (1 until the stride-2 layer of encoder 2, then 2); a convolution of kernel 1 adds nothing. Samples are (r - 1) hop + frame length, seconds at the clip's own rate. Encoder 1 follows the longest chain of a Res2Net block (7 convolutions), encoder 2 follows the 52 dense layers one after another, since each reads the output of the one before it.

| stage of encoder 1 | r (frames) | samples | seconds |
|---|---:|---:|---:|
| first convolution (k 5, d 1) | 5 | 2,048 | 0.085 |
| block 1 (d 2, chain of 7) | 33 | 9,216 | 0.384 |
| block 2 (d 3, chain of 7) | 75 | 19,968 | 0.832 |
| block 3 (d 4, chain of 7) | 131 | 34,304 | 1.429 |

| stage of encoder 2 | r (frames) | samples | seconds |
|---|---:|---:|---:|
| ten stem convolutions (k 3) | 21 | 3,600 | 0.225 |
| stride-2 convolution (k 5) | 25 | 4,240 | 0.265 |
| block 1: 12 layers, d 1 | 73 | 11,920 | 0.745 |
| block 2: 24 layers, d 2 | 265 | 42,640 | 2.665 |
| block 3: 16 layers, d 2 | 393 | 63,120 | 3.945 |

Tracing the frames one output reads through the real wiring (every chunk, the residual connection, the stride) gives, for the three blocks of encoder 1, spans of 33, 75 and 131 frames made of 33, 75 and 131 distinct frames (every frame of the span is read, because the chunks of a block fill the gaps one chain would leave), and for encoder 2 a span of 393 frames made of 393 distinct frames. Both equal the recursion.
<!-- /worked -->

Encoder 1 reaches 131 frames, 1.43 seconds, with a stack of only four named stages, because each of its three blocks follows a dilated chain, and each of those chains is up to seven convolutions long; the next section says why. Encoder 2 gets to 393 frames of the filterbank, 3.9 seconds, by brute depth: its stem alone is ten convolutions of kernel 3, and its 52 dense layers each add a few frames, 4 in the first block (kernel 3, dilation 1, jump 2) and 8 in the others. The recursion assumes that a chain is followed to its end, and in encoder 2 that is right because each dense layer reads, among the channels it receives, the output of the layer just before it. The trace in the table does not rely on that assumption: it starts from one output, walks back through every chunk, residual connection and stride of the real wiring, and counts the distinct frames that it touches, and for both encoders it ends on the same number as the recursion.

Both receptive fields are shorter than a clip of 7 seconds and longer than a clip of 1 second. An output of the last layer is therefore a local summary of the second or so around it, and the vector for the whole clip has to come from combining these summaries over time, which is what the pooling of the last section but two does. Each encoder also has a gate that is not local at all, one that looks at the whole clip and, in encoder 2, one that looks at two-second segments of it. The next two sections open the ladder of chunks and then those gates.

## Eight chunks and a ladder

In the middle of encoder 1 the stream has 512 channels and a dilated convolution would have to mix all of them: $512 \times 512 \times 3 = 786{,}432$ weights for one layer. The block cuts that. It splits the 512 channels into 8 chunks of 64, passes chunk 0 through untouched, and for the others adds the output of the previous chunk to the chunk before convolving it, so that the chunks form a ladder. This is the idea of the Res2Net backbone, brought here to one dimension. The convolutions have 64 channels in and out, kernel 3 and the dilation of the block.

<!-- worked: speaker_embeddings.ladder -->
The Res2Net block of encoder 1 splits its 512 channels into 8 chunks of 64. Chunk 0 is passed on as it is; chunk i is added to the output of chunk i - 1 (for i >= 2) and goes through one convolution of 64 channels, kernel 3, dilation d. Here d = 2.

| chunk i | input of its convolution | convolutions on its path | frames it reads |
|---|---|---:|---:|
| 0 | none | 0 | 1 |
| 1 | the chunk alone | 1 | 5 |
| 2 | the chunk plus the output of 1 | 2 | 9 |
| 3 | the chunk plus the output of 2 | 3 | 13 |
| 4 | the chunk plus the output of 3 | 4 | 17 |
| 5 | the chunk plus the output of 4 | 5 | 21 |
| 6 | the chunk plus the output of 5 | 6 | 25 |
| 7 | the chunk plus the output of 6 | 7 | 29 |

One dense convolution of 512 channels would hold 512 x 512 x 3 = 786,432 weights; the seven convolutions of the ladder hold 7 x 64 x 64 x 3 = 86,016, which is 9.14 times fewer.
<!-- /worked -->

There are two gains and the table shows both. The first is a count: seven convolutions of $64 \times 64 \times 3$ hold 86,016 weights where one dense convolution holds 786,432, which is 9.14 times fewer, and costs the same factor fewer operations per frame. The second is a spread of scales inside one layer. Chunk 1 reads 5 frames, chunk 7 reads 29, and they are then recombined by the 1-by-1 convolution that follows, so a single block gives the next layer short-range and long-range evidence side by side and lets it choose. The longest path, 7 convolutions at dilation $d$, is the number that entered the receptive-field table: each of the three blocks adds $7 \times 2d$ frames, 28, 42 and 56 for $d = 2, 3, 4$.

Each block is wrapped as the residual stages of Volume VII are: a convolution of kernel 1, the ladder, another convolution of kernel 1, a gate that the next section opens up, and the block's input added on at the end. The residual connection means that the shortest path through all of encoder 1 is the first convolution and nothing else, 5 frames, and the longest is the 131 of the table. The outputs of the three blocks are not only passed on in a chain. They are also concatenated, 3 times 512 making 1,536 channels, and one more convolution of kernel 1 mixes them, so that the pooling that follows can use what was found at three different depths. The architecture calls this multi-layer feature aggregation, and it is why the trace of the receptive-field table reports one span for each block output.

## A gate from the whole clip, and one from a window

The ladder looks at a window. The speaker of a recording, though, is a property of the whole of it, and a gate that tells every position which channels are informative for this clip should be computed from the whole clip. **Squeeze and excitation** does this in two steps: squeeze each channel to one number by its mean over time, pass the vector of means through a small bottleneck to get one gate in $(0, 1)$ for each channel, and multiply every frame of the channel by its gate,

$$
m_c = \frac1T\sum_{t} x_{c,t}, \qquad g = \mathrm{sigmoid}\bigl(W_2\,\mathrm{relu}(W_1 m + b_1) + b_2\bigr), \qquad y_{c,t} = g_c\,x_{c,t}.
\tag{2}
$$

In encoder 1 the bottleneck is 512 channels down to 128 and back. The table is a block with 4 channels and 3 frames and a bottleneck of 2, small enough to check.

<!-- worked: speaker_embeddings.squeeze_excitation -->
Squeeze and excitation on 4 channels and 3 frames. Squeeze: the mean over the 3 frames of each channel. Excite: h = relu(W1 m + b1) with W1 = (0.50, 0.25, -0.50, 0.00) and (-0.25, 0.50, 0.25, 0.25), b1 = (0.00, -0.25); z = W2 h + b2 with the rows of W2 (1.0, 0.0), (0.0, -1.0), (0.5, 0.5), (-1.0, 1.0) and b2 = (0.00, 0.50, 0.00, 0.25); g = sigmoid(z) (the stable form of the book's pseudocode). Output: each channel times its gate. The hidden vector is h = (0.5833, 0.5833).

| channel | frames | mean m | z | gate g | output |
|---|---|---|---:|---:|---|
| 0 | (2.0, 4.0, 0.0) | 2.0000 | 0.5833 | 0.6418 | (1.2837, 2.5673, 0.0000) |
| 1 | (1.0, 1.0, 1.0) | 1.0000 | -0.0833 | 0.4792 | (0.4792, 0.4792, 0.4792) |
| 2 | (-1.0, 3.0, 2.0) | 1.3333 | 0.5833 | 0.6418 | (-0.6418, 1.9255, 1.2837) |
| 3 | (0.0, 0.0, 6.0) | 2.0000 | 0.2500 | 0.5622 | (0.0000, 0.0000, 3.3731) |
<!-- /worked -->

Channels 0 and 2 receive the same gate, 0.6418, though they differ frame by frame, because the weights of this small example happen to give them the same $z$; and the gate is the same for every frame of a channel: it chooses channels, never moments, and a clip that gets longer changes it only through the means. The gate of equation (2) is a number per channel for the whole clip.

Encoder 2 asks for more. A gate for the whole clip is too coarse when a clip holds two speakers, or one speaker whose voice changes as the recording goes on, or silence and speech in turn. Its layers use **context-aware masking**: the gate has a value at every frame, computed from a context that adds two pooled views of the same channel, the mean over the whole clip and the mean over a segment of 100 rows that the frame belongs to (at 50 rows a second, two seconds). Segments are cut from the start, and the last may be shorter, in which case its mean is taken over the rows it has. The gate multiplies the output of a local convolution at that frame. The table sets it on 7 frames with segments of 3 so that the short tail shows.

<!-- worked: speaker_embeddings.context_gate -->
A context gate on 2 channels and 7 frames with segments of 3 frames (so 3, 3 and 1 frames). The context of frame t is the mean of the channel over all 7 frames plus the mean over the segment that holds t (the last segment has one frame and is divided by 1). Channel means (4.2857, 1.5714); segment means of channel 0 (2.0000, 5.0000, 9.0000) and of channel 1 (0.3333, 3.0000, 1.0000). Gate of frame t: sigmoid(W2 relu(W1 context + b1) + b2) with W1 = (0.25, 0.00) and (0.00, 0.50), b1 = (-0.50, 0.00), W2 = (0.50, -1.00), b2 = 0.25.

| row | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| channel 0 | 1.00 | 2.00 | 3.00 | 6.00 | 5.00 | 4.00 | 9.00 |
| channel 1 | 0.00 | 1.00 | 0.00 | 2.00 | 4.00 | 3.00 | 1.00 |
| context, channel 0 | 6.2857 | 6.2857 | 6.2857 | 9.2857 | 9.2857 | 9.2857 | 13.2857 |
| context, channel 1 | 1.9048 | 1.9048 | 1.9048 | 4.5714 | 4.5714 | 4.5714 | 2.5714 |
| gate | 0.4584 | 0.4584 | 0.4584 | 0.2451 | 0.2451 | 0.2451 | 0.5927 |
| local output | 2.00 | -1.00 | 1.00 | 0.50 | 3.00 | 2.00 | -2.00 |
| gated output | 0.9169 | -0.4584 | 0.4584 | 0.1225 | 0.7353 | 0.4902 | -1.1853 |
<!-- /worked -->

The frames of one segment share a context and therefore a gate, and the last frame, alone in its segment of one, has the segment mean equal to its own value, 9 in channel 0, which makes its context the largest and its gate $0.5927$ against $0.2451$ for the segment before it. In brain this is computed on the host, once for each of the 52 dense layers of a forward pass, and the model module explains why: the pooling is a fixed window of 100 rows that divides the tail by the rows it covers, and none of the pooling kernels of the engine has that rule. For a computation that runs once per reference clip, a round trip to the host that is not noticed is a better choice than a kernel with one caller (`crates/campplus/src/model.rs`, `cam_context`).

## From many frames to one vector

Everything so far leaves a sequence: encoder 1 has 1,536 channels and encoder 2 has 512, each at $T$ or $T/2$ frames. To reach a vector, the time axis has to be removed, and the cheapest way is a mean over time. Take the clips $(1, 3, 1, 3)$ and $(2, 2, 2, 2)$ in one channel. Their means are both 2. One clip swings between 1 and 3 and the other is steady, and a mean cannot tell them apart. A voice is often recognised in how the features vary, so the first improvement is to add the standard deviation over time: the means and deviations of all channels, side by side, make a vector of $2C$ numbers, the **statistics pooling** of the x-vector systems that this line of models descends from. Here the two clips differ, $1$ against $0$ in the second half of the vector.

A mean treats every frame alike, and frames are not alike. A silence at the end of a clip carries no speaker at all and still pulls the mean towards its value, and a frame in the middle of a vowel is rich in what makes the voice. The weights ought to be learned and to depend on the frame. In **attentive statistics pooling** a small network reads the frames and writes a score for each, a softmax over time turns the scores of a channel into weights $a_{c,t}$ that add to 1, and the mean and the deviation are taken with those weights:

$$
\mu_c = \sum_t a_{c,t}\,x_{c,t}, \qquad \varsigma_c = \sqrt{\max\Bigl(\sum_t a_{c,t}\,(x_{c,t} - \mu_c)^2,\ \epsilon\Bigr)}, \qquad a_{c,\cdot} = \mathrm{softmax}\bigl(e_{c,\cdot}\bigr).
\tag{3}
$$

The floor $\epsilon$ keeps the square root away from zero, where its derivative is not defined, for a channel that is constant under its weights. The scores $e_{c,t}$ come from a network that is given the frames and also the plain statistics of the whole clip, so that it can tell what is unusual in a frame relative to the clip. Encoder 1's reads, at every frame, the vector of the 1,536 channels, the 1,536 clip means and the 1,536 clip deviations, which is 4,608 numbers, through a convolution of kernel 1 down to 128 and a relu, then a $\tanh$, then a convolution of kernel 1 up to 1,536 scores, one for each channel at each frame. The attention is per channel, so each channel decides its own frames. The table follows all of this on 2 channels and 4 frames, with a silence as the last frame and the attention network cut to 3 hidden units.

<!-- worked: speaker_embeddings.attentive_pooling -->
Attentive statistics pooling of 2 channels over T = 4 frames; the last frame is a silence. Step 1: the plain mean and population standard deviation of each channel (weights 1/4, divisor 4): means (1.5000, 0.5000), deviations (1.1180, 1.1180). Step 2: the 6 x 4 array of x with its channel means and deviations repeated at every frame. Step 3: a convolution of kernel 1 with weights W_a (3 x 6) = (0.50, 0.00, 0.00, 0.00, 0.50, 0.00), (0.00, 0.50, 0.00, 0.00, 0.00, 0.50), (0.25, 0.25, -0.25, -0.25, 0.00, 0.00), no bias, then relu, then tanh (the library function), then a kernel-1 convolution W_c (2 x 3) = (1.0, 0.0, 1.0), (0.0, 1.0, -1.0), no bias, the score. Step 4: a softmax over the 4 frames of each channel. Step 5: weighted mean and weighted standard deviation with those weights, the variance floored at 1e-12.

| row | t=0 | t=1 | t=2 | t=3 |
|---|---:|---:|---:|---:|
| x, channel 0 | 1.00 | 3.00 | 2.00 | 0.00 |
| x, channel 1 | 2.00 | -1.00 | 1.00 | 0.00 |

Scores and attention weights (each weight row adds to 1):

| channel | score 0 | score 1 | score 2 | score 3 | weight 0 | weight 1 | weight 2 | weight 3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| channel 0 | 1.0302 | 0.9680 | 1.1602 | 0.5072 | 0.2724 | 0.2560 | 0.3102 | 0.1615 |
| channel 1 | 0.6703 | 0.0589 | 0.5404 | 0.5072 | 0.3058 | 0.1659 | 0.2685 | 0.2598 |

The pooled vector of 4 numbers is the mean of channel 0, the mean of channel 1, then the two deviations:

| pooling | mean 0 | mean 1 | std 0 | std 1 |
|---|---:|---:|---:|---:|
| plain mean and standard deviation (weights 1/4) | 1.5000 | 0.5000 | 1.1180 | 1.1180 |
| the same with the divisor 3 (unbiased) | 1.5000 | 0.5000 | 1.2910 | 1.2910 |
| attentive mean and standard deviation | 1.6606 | 0.7142 | 1.0291 | 1.0712 |
<!-- /worked -->

The last frame of channel 0 has the lowest weight, $0.1615$ against $0.2724$ for the first, so the silence counts for less, and the attentive mean $1.6606$ is higher than the plain mean $1.5$ for that reason, as the speech frames are brought forward. In channel 1 the weights are nearly even for the silence, $0.2598$ against $0.25$: nothing forces an attention network to discount every silence in every channel, and what it learns is whatever helps the loss. The third row of the second table also shows a detail of the other encoder. Encoder 2 pools without attention and with the unbiased divisor, $T - 1$ instead of $T$: its deviation is larger by $\sqrt{T/(T-1)}$, $1.1547$ for the 4 frames here, which falls under 1% only from 51 rows on, and is the reason the pool needs two rows at least. The brain port confirmed that divisor against the released graph's own arithmetic, a multiplication by $N$ followed by a division by $N - 1$ (the module documentation of `crates/campplus/src/model.rs`), and not by assumption: a wrong divisor would have changed every embedding by a few percent with nothing to say so.

The routine, in the book's pseudocode, takes the two attention matrices as given.

```text
function weighted_stats(X: array<real>[C, T], a: array<real>[C, T], eps: real)
        -> (array<real>[C], array<real>[C]):
    mu = zeros(real, [C])
    sd = zeros(real, [C])
    for c in 0 .. C:
        for t in 0 .. T:
            mu[c] = mu[c] + a[c, t] * X[c, t]
        var = 0.0
        for t in 0 .. T:
            var = var + a[c, t] * (X[c, t] - mu[c])^2
        sd[c] = sqrt(max(var, eps))
    return (mu, sd)

function attentive_pool(X: array<real>[C, T], Wa: array<real>[A, 3 * C], ba: array<real>[A],
                        Wc: array<real>[C, A], bc: array<real>[C], eps: real) -> array<real>[2 * C]:
    require T >= 1
    uniform = zeros(real, [C, T])
    for c in 0 .. C:
        for t in 0 .. T:
            uniform[c, t] = 1.0 / T
    (mean0, sd0) = weighted_stats(X, uniform, eps)
    S = zeros(real, [3 * C, T])
    for t in 0 .. T:
        for c in 0 .. C:
            S[c, t] = X[c, t]
            S[C + c, t] = mean0[c]
            S[2 * C + c, t] = sd0[c]
    H = zeros(real, [A, T])
    for a_ in 0 .. A:
        for t in 0 .. T:
            acc = ba[a_]
            for i in 0 .. 3 * C:
                acc = acc + Wa[a_, i] * S[i, t]
            H[a_, t] = tanh(relu(acc))
    a = zeros(real, [C, T])
    for c in 0 .. C:
        scores = zeros(real, [T])
        for t in 0 .. T:
            acc = bc[c]
            for a_ in 0 .. A:
                acc = acc + Wc[c, a_] * H[a_, t]
            scores[t] = acc
        w = softmax(scores)
        for t in 0 .. T:
            a[c, t] = w[t]
    (mu, sd) = weighted_stats(X, a, eps)
    out = zeros(real, [2 * C])
    for c in 0 .. C:
        out[c] = mu[c]
        out[C + c] = sd[c]
    return out
```

The pooled vector has $2C$ entries, 3,072 for encoder 1 and 1,024 for encoder 2, and a last linear map brings it to the width of the embedding: 3,072 to 1,024 in encoder 1, 1,024 to 192 in encoder 2, where it is followed by a normalisation that has a running mean and variance and no learned gain. The dimension of the embedding is a choice made by whoever designed the consumer: 1,024 is the width of the Qwen3-TTS talker, and 192 is the width that CosyVoice's flow model expects.

Nothing in the pooling or the convolutions knows what a speaker is. That comes from the loss: the published recipes of both architectures train the encoder, followed by a classifier, to name the speaker of each clip in a corpus of many thousands of speakers, with the angular margin of Volume VII, Chapter 11 so that the embedding just before the classifier is spread by angle, which is what cosine scoring then reads. The crates of brain run these encoders and do not train them: they read the published weights, and what their tests establish is that the port reproduces the reference's vector, not that the vector is a good one.

## What the two encoders hold and cost

Both encoders are now described, and they can be counted from their configurations, the way the previous chapter counted the two recognisers.

<!-- worked: speaker_embeddings.layout -->
The layers behind the counts below: widths, kernels, chunking and which tensors exist. 'k 1' is a 1 x 1 convolution, a matrix applied to every frame.

Encoder 1

| part | layout | biases and norms |
|---|---|---|
| first convolution | 128 -> 512, k 5 | weight and bias |
| SE-Res2Net block (three, dilations 2, 3, 4) | 1 x 1 convolution 512 -> 512; the 512 channels split into 8 chunks of 64, chunk 0 passed on, chunks 1 to 7 each a convolution 64 -> 64, k 3, at the block's dilation; 1 x 1 convolution 512 -> 512; squeeze-excitation 512 -> 128 -> 512 (1 x 1, run once on the clip's mean) | weight and bias on every convolution; no normalisation tensor is stored |
| aggregation convolution | 1536 -> 1536, k 1 (the three block outputs side by side) | weight and bias |
| attention of the pooling | 4608 -> 128 -> 1536, k 1 (the channels, their mean and their deviation side by side) | weight and bias on both |
| projection | 3072 -> 1024 (the mean and the deviation of the 1536 channels) | weight and bias |

Encoder 2

| part | layout | biases and norms |
|---|---|---|
| stem | 32 channels, 3 x 3 over frequency and time: a convolution 1 -> 32; two groups of two residual blocks, each block two convolutions 32 -> 32, the first block of each group striding by 2 over frequency (80 -> 40 -> 20) and carrying a 1 x 1 shortcut 32 -> 32; a last convolution 32 -> 32, stride 2 over frequency (-> 10): 1 + 8 + 1 = 10 convolutions and 2 shortcuts, and 32 x 10 = 320 channels leave it | weight and bias on all 12 |
| time-delay convolution | 320 -> 128, k 5, stride 2 over time | weight and bias |
| dense layer | input of c_in channels (c_in is 128 at the first layer of the first block and grows by 32 with each layer): normalisation of the input (gain, shift, running mean, running variance), 1 x 1 convolution c_in -> 128, local convolution 128 -> 32, k 3 at the block's dilation, and the context gate, two 1 x 1 convolutions 128 -> 64 -> 32; the 32 output channels are appended to the input; 11 tensors | bias on the 1 x 1 convolutions and the gate, none on the local convolution |
| dense blocks | 12, 24 and 16 layers at dilations 1, 2 and 2 |  |
| transit | normalisation of the block's c channels (gain, shift, running mean and variance), then 1 x 1 convolution c -> c / 2; c is 512, 1024 and 1024 for the three, and the channels left are 256, 512 and 512 | no bias on the first two; the third has a bias of 512 (the normalisation after it is folded into the convolution in the released graph, which is why it has 6 tensors and the others 5) |
| projection | statistics pooling of the 512 channels (mean and deviation) -> 1024, 1 x 1 convolution 1024 -> 192, then a normalisation with running mean and variance only (no gain or shift) | no bias; 2 x 192 buffers |
<!-- /worked -->

<!-- worked: speaker_embeddings.parameter_counts -->
Stored values and operations of the two encoders, counted from their configurations. Weights are the entries of kernels and matrices (and, for encoder 2, the gain and shift of its normalisations); biases are the entries of bias vectors; buffers are the running means and variances of encoder 2's normalisations, which are stored but not trained. Operations count 2 per multiply-add of the convolutions and projections only.

Encoder 1 (24 kHz log-mel, 93.75 frames a second). 'Operations per frame' is for the convolutions that run at every frame; 'operations per clip' for the vector convolutions that run once, whatever the length.

| stage | weights | biases | tensors | operations per frame | operations per clip |
|---|---:|---:|---:|---:|---:|
| first convolution, 128 to 512, k 5 | 327,680 | 512 | 2 | 655,360 | 0 |
| SE-Res2Net block 1 | 741,376 | 2,112 | 22 | 1,220,608 | 262,144 |
| SE-Res2Net block 2 | 741,376 | 2,112 | 22 | 1,220,608 | 262,144 |
| SE-Res2Net block 3 | 741,376 | 2,112 | 22 | 1,220,608 | 262,144 |
| aggregation convolution, 1536 to 1536 | 2,359,296 | 1,536 | 2 | 4,718,592 | 0 |
| attention of the pooling (4608 to 128 to 1536) | 786,432 | 1,664 | 4 | 1,572,864 | 0 |
| projection, 3072 to 1024 | 3,145,728 | 1,024 | 2 | 0 | 6,291,456 |
| total | 8,843,264 | 11,072 | 76 | 10,608,640 | 7,077,888 |

One second of audio costs encoder 1 994.6 million operations in the per-frame part (93.75 frames), and the once-per-clip part adds 7.1 million whatever the length. Total stored values: 8,854,336.

Encoder 2 (16 kHz filterbank at 100 frames a second; the stem at that rate, the rest at 50 rows a second):

| stage | weights | biases | buffers | tensors | operations per second of audio |
|---|---:|---:|---:|---:|---:|
| stem: 10 2-D convolutions and 2 shortcuts | 85,280 | 384 | 0 | 24 | 477,696,000 |
| time-delay convolution, 320 to 128, k 5, stride 2 | 204,800 | 128 | 0 | 2 | 20,480,000 |
| dense block 1: 12 layers, d 1 | 744,576 | 2,688 | 7,296 | 132 | 73,728,000 |
| transit 1: 512 to 256 | 132,096 | 0 | 1,024 | 5 | 13,107,200 |
| dense block 2: 24 layers, d 2 | 2,487,552 | 5,376 | 29,952 | 264 | 245,760,000 |
| transit 2: 1024 to 512 | 526,336 | 0 | 2,048 | 5 | 52,428,800 |
| dense block 3: 16 layers, d 2 | 1,924,608 | 3,584 | 24,064 | 176 | 190,054,400 |
| transit 3: 1024 to 512 | 526,336 | 512 | 2,048 | 6 | 52,428,800 |
| projection 1024 to 192, final normalisation | 196,608 | 0 | 384 | 3 | 0 |
| total | 6,828,192 | 12,672 | 66,816 | 617 | 1,125,683,200 |

Total stored values: 6,907,680, of which 66,816 are buffers. The tensor counts, 76 and 617, are the numbers of tensors in the two released checkpoints that the engine verifies.
<!-- /worked -->

The two counts, 76 tensors and 617, are not invented here. The second equals the number of initializers in the released graph of CAM++, and the 6,907,680 stored values of the table equal the sum of their sizes, as a direct count over the initializers of the released graph file gives 617 tensors and 6,907,680 values; the first count matches the number that the import test of encoder 1 pins, 76 `speaker_encoder.*` tensors. Both are the same tensors brain reads, which is the content of a parameter count: not that the model is small, but that the list of its tensors is the list of what must be fetched, converted and held.

The tables hold two observations. In encoder 1 the weights are in the wide 1-by-1 convolutions: the projection from 3,072 to 1,024 holds 3,145,728 of the 8,843,264 weights, 35.6%, and the aggregation convolution 2,359,296, 26.7%; the three blocks, where the dilated convolutions are, hold a quarter between them. The projection, though, is paid for once per clip, 6.3 million operations whatever the length, while the aggregation is paid for at every frame. A clip of 7 seconds costs about 7 billion operations in encoder 1. In encoder 2 the order of things is the opposite of the weights: the 2-D stem has 85,280 weights, 1.2% of the total, and 42.4% of the work, 477.7 million of the 1,125.7 million operations a second. A weight in a convolution is used once for every position of its output, and the stem's positions are the $100 \times$ (up to 80) cells of a second of frequency-time, where the dense layers work on 50 rows with no frequency axis. Where the work is does not follow from where the weights are, and the cure for a cost is to find it in the right table. That is also why the brain port needed a 3-dimensional convolution kernel for the stem: it downsamples the frequency axis only, with a stride of $(2, 1)$, and no 2-D kernel of the engine accepts a stride that differs by axis, so the stem runs as a 3-D convolution with a time axis of 1 (`crates/campplus/src/model.rs`, the module documentation).

![Two pipelines on one grid. Encoder 1: a 24,000 per second waveform becomes 93.75 log-mel frames per second of 128 values, passes a time-delay convolution to width 512, three SE-Res2Net blocks with dilations 2, 3 and 4, an aggregation to width 1536, attentive statistics of width 3072 and a projection to 1024 numbers. Encoder 2: a 16,000 per second waveform becomes 100 filterbank frames per second of 80 values, a 2-D stem to width 320, a strided time-delay convolution to 50 rows per second of width 128, 52 context-gated dense layers, statistics of width 1024 and a projection to 192 numbers.](img/speaker-encoders.svg)

## Closeness in cosine

A vector is only useful once it can be compared. Two recordings are scored by the cosine of their embeddings, and the pair is called the same speaker when the cosine reaches a threshold $\tau_{\mathrm{id}}$, as in Volume VII, Chapter 11. Two errors are possible and each has a rate: the **false rejection rate** is the share of pairs of one speaker that fall below the threshold, and the **false acceptance rate** the share of pairs of two speakers that reach it. A low threshold gives few false rejections and many false acceptances, a high one the reverse, and the two curves cross at one threshold; the common value there is the **equal error rate**. It is a single number for a model and a set of recordings, and it hides a good deal: it says nothing of the cost of either error, which a real application has to weigh.

What the encoders of this chapter do with a longer clip is the thing worth seeing, and a trained model is not needed to see it. The next block simulates a population. It does not train anything: 40 speakers are drawn as random unit vectors of width 16, and an utterance of $s$ seconds is the speaker's vector plus noise of size $\rho = 2$ divided by $\sqrt{s}$, because an embedding made of $s$ independent seconds of evidence has the noise of their mean. Four lengths, 1, 2, 4 and 8 seconds, use the same noise draws.

<!-- worked: speaker_embeddings.scoring -->
Simulated embeddings, not a trained network: 40 speakers, 4 utterances each, vectors of width 16. The generator is `speaker_embeddings.scoring` (seed 0x7d90a51ea1a8bbd3 (9047913202367118291)). Draw order: the 40 centres (speaker 0 to 39, 16 standard normals each, each divided by its length), then the noise (speaker, utterance, 16 standard normals). An utterance of s seconds of speaker i is its centre plus 2 times its noise vector divided by sqrt(16 * s), the whole divided by its length; the same noise is used for all four lengths. Cosines are dot products of the unit vectors summed over the 16 components in order. Genuine pairs are two utterances of one speaker (240 pairs), impostor pairs of two speakers (12,480). A pair is accepted when its cosine is at or above the threshold. The equal error rate is found by trying every observed cosine as the threshold and keeping the one where the false rejection rate and the false acceptance rate are closest (the lowest on a tie); the error rate printed is the mean of the two there.

| seconds | mean genuine cosine | mean impostor cosine | threshold at the equal error rate | equal error rate |
|---:|---:|---:|---:|---:|
| 1 | 0.202 | -0.004 | 0.112 | 33.0% |
| 2 | 0.338 | -0.006 | 0.187 | 23.3% |
| 4 | 0.508 | -0.008 | 0.297 | 12.1% |
| 8 | 0.676 | -0.011 | 0.439 | 4.1% |

One threshold, that of the 8 second clips (0.439), applied to all four lengths:

| seconds | threshold | genuine rejected | impostors accepted |
|---:|---:|---:|---:|
| 1 | 0.439 | 82.9% | 3.7% |
| 2 | 0.439 | 65.8% | 3.7% |
| 4 | 0.439 | 30.4% | 3.8% |
| 8 | 0.439 | 4.2% | 4.0% |
<!-- /worked -->

The error falls from 33.0% at 1 second to 4.1% at 8, and the table's means show where it comes from: the impostor cosines stay near 0 and the genuine cosines rise from 0.202 to 0.676, so the two groups move apart. A short calculation explains the numbers. The speaker's unit vector $\mathbf{z}$ (the centre of the simulation, not the bias $u$ of Chapter 4) and the noise vectors $\boldsymbol\nu_1$, $\boldsymbol\nu_2$ of two utterances are nearly orthogonal in width 16, and $\lVert\boldsymbol\nu\rVert^2$ is about $\rho^2/s$, so the cosine of $\mathbf{z} + \boldsymbol\nu_1$ with $\mathbf{z} + \boldsymbol\nu_2$ is about $1/(1 + \rho^2/s)$: $0.2$, $0.333$, $0.5$, $0.667$ for $s = 1, 2, 4, 8$, against the table's $0.202$, $0.338$, $0.508$, $0.676$.

![Two kinds of error against the cosine threshold for the simulated population, for clips of 1 and 8 seconds: the false acceptance rate falls and the false rejection rate rises with the threshold, and the two cross at the equal error rate, at a much lower threshold and a much higher error for the 1 second clips.](img/speaker-scoring.svg)

The second table of the block is the one that costs people their evenings. A threshold tuned for 8-second clips, 0.439, applied to 1-second clips rejects 82.9% of the genuine pairs, and it accepts 3.7% of impostor pairs, about the same as for 8 seconds, because the impostor side did not move. A threshold is a property of the model together with the length of what it is shown, and a system that accepts clips of any length needs either a threshold for each length or a way to make the cosines comparable. The routine of the rates is short.

```text
function equal_error_rate(gen: list<real>, imp: list<real>) -> (real, real):
    require len(gen) >= 1 and len(imp) >= 1
    scores = sort(gen followed by imp)
    best_gap = infinity
    best_tau = 0.0
    best_rate = 0.0
    for tau in scores:
        rejected = 0
        for c in gen:
            if c < tau:
                rejected = rejected + 1
        accepted = 0
        for c in imp:
            if c >= tau:
                accepted = accepted + 1
        frr = rejected / len(gen)
        far = accepted / len(imp)
        if abs(far - frr) < best_gap:
            best_gap = abs(far - frr)
            best_tau = tau
            best_rate = (far + frr) / 2
    return (best_tau, best_rate)
```

Because the loop keeps a candidate only if its gap is strictly smaller, a tie goes to the lowest threshold, the rule that the table states.

## What the two synthesisers do with the vector

The vector leaves the encoder and enters a synthesiser, and the two crates of brain that carry it do very different things with it. The Qwen3-TTS pipeline computes the vector from the reference wav by `ecapatdnn::SpeakerEncoder::embed_wav`, which resamples the audio to 24 kHz and runs the front end of the first section, and places it unchanged as one row of the Talker's input, between the language tags and the padding token, in a position of width $d_{\mathrm{model}}$: the prompt builder asserts that the vector's length equals the Talker's width (`crates/qwen3tts/src/prompt.rs`, `codec_prefix`), 1,024 for the 0.6B Base checkpoint. The CosyVoice pipeline works the other way round. It takes the filterbank of the reference audio, removes the mean of every mel bin over the clip (a step that its reference performs and that the crate reproduces), runs encoder 2, and in the flow model normalises the result to unit length and maps it with a learned affine layer from 192 to 80 numbers, which is passed to the decoder as a condition in its sampling loop (`crates/cosyvoice/src/flow.rs`, `speaker_embedding`). One consumer takes the vector raw and the other takes only its direction.

That difference decides what a test of the encoder has to check, and the history of the crate shows it. Commit `d16050463` (25 August 2026) records that three parity gates, two for the audio codec and one for this encoder, all skipped, and that a skipped test counts as a pass here, so that the speaker encoder had no executable numerical check at all until the goldens were produced. Once it ran, the commit also noted that the encoder's test asserted a floor on the cosine and nothing else, and added a ceiling on the relative L2 distance, `1e-4` against a clean value of about `7e-7`, because cosine is scale invariant: a vector twice as large scores 1.0. That is the content of knowledge #2, whose lesson is to gate the relative L2 distance alongside the cosine. The table shows it on four numbers.

<!-- worked: speaker_embeddings.parity_metrics -->
The reference vector is (3.0, 4.0, 0.0, 12.0) (length 13); 'one entry off' adds the stated amount to its first entry (the 3.0), with a plus sign. 'Cosine' and 'relative L2' compare a candidate with it (relative L2 is the length of the difference over the length of the reference). The last column passes both vectors through the same stand-in for a consumer that normalises first, divides each by its length and multiplies by (1.0, 0.0, -1.0, 0.5), (0.0, 1.0, 0.5, -0.5), and reports the relative L2 of the two results.

| candidate | cosine | relative L2 | relative L2 after normalising and multiplying |
|---|---:|---:|---:|
| the reference vector | 1.000000 | 0.000000 | 0.000000 |
| scaled by 1.05 | 1.000000 | 0.050000 | 0.000000 |
| scaled by 2 | 1.000000 | 1.000000 | 0.000000 |
| one entry off by 0.001 | 1.000000 | 0.000077 | 0.000091 |
| one entry off by 1 | 0.997307 | 0.076923 | 0.086786 |
<!-- /worked -->

The cosine of $2\mathbf{v}$ with $\mathbf{v}$ is 1.000000, and its relative L2 is 1: the candidate has the right direction and is double the size. After the normalise-then-multiply of the CosyVoice route the two are identical, as the last column shows, and the cosine check would have been adequate for that consumer. For the Qwen3-TTS route, where the vector goes in as it is, a doubled vector is a different input to the Talker, and only the second metric sees it. A check that is enough for one consumer is not enough for the other, and an encoder shared between consumers has to be checked against the stricter. This is the same reason that a scoring system and a synthesiser can use the same embedding and need different tests of it.

## Where the two crates came from

Encoder 1 arrived with the first speech-synthesis stack. Commit `26faef787` (28 June 2026) added the crate under its first name, `speaker`, together with the audio codec decoder. Its message lists the architecture, a time-delay block of convolution and ReLU, three SE-Res2Net blocks of scale 8, feature aggregation and attentive statistics pooling, ending in a projection to 1,024, and adds that this variant has no normalisation layers, that the reflect "same" padding is reproduced exactly, and that the pooling reductions run on the host. The code of the crate agrees on each point: no tensor of the checkpoint is a normalisation (the 76 of the table are all convolutions), the padding is the reflect pad of the first section, and the softmax over time and the pooled statistics are computed on the host, for which the module documentation gives the reason that no time-axis softmax kernel existed. Commit `5a8140cd8` (13 August 2026) renamed the crate to `crates/ecapatdnn` when the audio-stack crates were renamed after their architectures.

Encoder 2 was added on 21 August 2026 in commit `c3e369bbd`, as the speaker input of the CosyVoice port. Its message says that the graph was verified node by node from the released ONNX file and not assumed from a summary, and that which normalisation layers the exporter folds into a convolution and which it leaves standalone depends on how many consumers the convolution's output has, not on the type of the module. The code bears this out, and it is a lesson that applies beyond this model: in the released file every batch normalisation in the stem and after each first convolution has been folded into a biased convolution, while the one that precedes each convolution in a dense layer and each transition survives with its four tensors, because its input is a concatenation and folding goes only into the layer before. The manifest of 617 tensors is the checklist of this fact, and the import checks it in both directions. The CAM++ crate resolves its directory through the model-store resolver as of commit `7b05f11ca` (7 September 2026), with a single role: nothing on a disk identifies the released file, so the explicit directory option is the only way in.

Two caveats the commits and modules themselves record. The filterbank front end of encoder 2 (`crates/audio/src/kaldi_fbank.rs`) is, in its own module documentation, checked structurally and by the end-to-end smoke test of the pipeline, but not against a captured reference output; the parity test of encoder 2 feeds the captured filterbank of a real clip straight in. And commit `3240b07c7` (11 November 2026) found that its front end indexed with a negative offset for any clip no longer than the reflect pad of 384 samples, so a short upload made `embed_wav` panic; the pad positions that have nothing to mirror are now zero, as in the audio crate's own front end. That repair is about the front end, and the same rule returns one level up, as the first section derived: the layers of the encoder have their own reflect pad, of up to 4 frames, and the function that applies it states that it requires the pad to be smaller than the signal. The shortest clip of the first table is the clip for which every pad in the encoder is smaller than the signal.

## Check your understanding

1. A reader computes the receptive field of a Res2Net block of dilation 2 by adding up the frames read by its eight chunks, $1 + 5 + 9 + 13 + 17 + 21 + 25 + 29$. What is wrong with the sum, and what is the right number?
2. In the attentive pooling of the chapter, every score of a channel happens to be equal. What does the pooled vector of that channel become, and how does it differ from the statistics pooling of encoder 2?
3. The simulated embeddings of 8-second clips need a threshold of 0.439. Explain, using the noise $\rho^2/s$, why that threshold rejects most genuine pairs of 1-second clips and why it accepts about the same share of impostors.
4. A port of encoder 1 reports cosine 1.000000 against the reference and is declared correct. Describe a bug that this check would miss, and say for which of the two synthesisers of the chapter it would matter.
5. Encoder 2's stem holds 1.2% of its weights and 42.4% of its work. Where does the mismatch come from, and what would reducing the time resolution of the stem by a factor of two do to each number?
6. Why does encoder 1 need at least 5 frames, and what would a largest dilation of 6 change in the shortest readable clip?

## Summary

A speaker embedding turns a recording of any length into a vector of fixed length that is close in cosine for one voice and far for two. Both encoders do it in the same three moves: convolutions that slide along time only (time-delay layers) and give each frame a wider and wider view, gates that let the whole clip or a segment of it say which channels and frames matter, and a pooling that removes the time axis. The pooling that works keeps the mean and the standard deviation and learns which frames to weight, which is attentive statistics pooling. Their sizes and costs are counted from their configurations: 8.85 million stored values and about 1,000 million operations per second of audio in encoder 1, 6.91 million and about 1,126 million in encoder 2, where the stem has 1.2% of the weights and 42.4% of the work. A simulated population showed the equal error rate falling from 33.0% to 4.1% as clips go from 1 to 8 seconds, and a threshold that is tuned for one length failing at another. The two synthesisers use the vector in two ways, raw and normalised, which is why their encoder needs a check on the size of the vector as well as its direction.

The vector is the speaker half of the instruction that a voice-cloning synthesiser needs; the other half is the text. How a model turns the text and this vector into a waveform is the next chapter.

## References

- Waibel, Hanazawa, Hinton, Shikano and Lang, *Phoneme Recognition Using Time-Delay Neural Networks*, 1989.
- Snyder, Garcia-Romero, Sell, Povey and Khudanpur, *X-Vectors: Robust DNN Embeddings for Speaker Recognition*, 2018.
- Okabe, Koshinaka and Shinoda, *Attentive Statistics Pooling for Deep Speaker Embedding*, 2018.
- Hu, Shen and Sun, *Squeeze-and-Excitation Networks*, 2018.
- Gao, Cheng, Zhao, Zhang, Yang and Torr, *Res2Net: A New Multi-scale Backbone Architecture*, 2019.
- Desplanques, Thienpondt and Demuynck, *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification*, 2020.
- Yu and Li, *Densely Connected Time Delay Neural Network for Speaker Verification*, 2020.
- Wang, Zheng, Chen, Cheng and Chen, *CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking*, 2023.

## Exercises

1. **[Calculate]** For a clip of 3.2 seconds, how many frames does each front end of the first table make, and how many rows does the strided convolution of encoder 2 leave?
2. **[Calculate]** A Res2Net block has dilation 5 and the chain of 7 convolutions of kernel 3. By how many frames does it widen the receptive field?
3. **[Calculate]** A Res2Net block has 256 channels split into 4 chunks of 64. How many weights do its convolutions hold, and how many does one dense convolution of kernel 3 on 256 channels hold? What is the ratio?
4. **[Calculate]** A squeeze-and-excitation block on 2 channels and 2 frames has $x = ((2, 4), (1, 1))$, $W_1$ the identity, $b_1 = 0$, $W_2 = \mathrm{diag}(0.5, -1)$ and $b_2 = 0$. Compute the gates and the output.
5. **[Calculate]** From which number of rows on does the unbiased divisor of encoder 2's pooling change the standard deviation by less than 1%?
6. **[Calculate]** Encoder 2's total of operations per second of audio is the last row of its table. How many does it need for a clip of 5 seconds, and how many of those fall in the stem?
7. **[Implement]** Reproduce `speaker_embeddings.tdnn_by_hand`: equation (1) with reflect padding and with zero padding, on the input and weights printed there.
8. **[Implement]** Reproduce `speaker_embeddings.attentive_pooling` with the pseudocode of this chapter, and check that its attention weights add to 1 in each channel.
9. **[Implement]** Reproduce `speaker_embeddings.scoring`: the generator and the draw order printed there, the four embeddings of each utterance and the equal error rate for each of the four lengths.
10. **[Investigate]** In the simulation, vary the noise $\rho$ and the width of the vectors. For which values does the approximation $1/(1 + \rho^2/s)$ for the genuine cosine stop being close, and why?

## Key terms

- **speaker embedding**: a vector of fixed length computed from a recording of any length, such that recordings of one voice are close in cosine and recordings of different voices are not. <!-- concept: speaker-embedding -->
- **attentive statistics pooling**: a pooling over time that learns a weight for each frame of each channel, by a softmax over a score from a small network, and returns the weighted mean and weighted standard deviation of every channel. <!-- concept: attentive-statistics-pooling -->
- **time-delay network**: a network of one-dimensional convolutions along time, whose weights are shared across time and whose units read a layer at several delayed moments, so that a deeper unit sees a longer stretch of the signal. <!-- concept: time-delay-network -->
- **equal error rate**: the common value of the false acceptance rate and the false rejection rate at the threshold where the two cross, one number for a model and a set of recordings that says nothing of the cost of either error. <!-- concept: equal-error-rate -->

## Answers

**Recall questions.**

1. $1 + \lfloor (16000 - 400)/160 \rfloor = 98$ frames. The padded length of the second clip is $24000 + 768 = 24768$, so $1 + \lfloor (24768 - 1024)/256 \rfloor = 93$ frames.
2. $32 \times 32 \times 9 = 9{,}216$ weights, and $9216 \times 40 \times 100 = 36{,}864{,}000$ multiply-adds.
3. $r = 1 + 2 + 4 + 8 = 15$.
4. Yes: 0.7 is at or above 0.5. Lowering the threshold makes false acceptances more frequent and false rejections rarer.
5. The weights are $(1, 2, 2, 1)/6$, that is $0.1667$, $0.3333$, $0.3333$, $0.1667$.

**Check your understanding.**

1. The chunks are not in series: each is read from its own path, and the field of the block is the largest of them, not the sum. Chunk $i$ passes through $i$ convolutions, so it reads $1 + 2 \cdot 2 \cdot i$ frames, and the longest is chunk 7 with 29 frames. The sum is 120.
2. Equal scores give equal weights $1/T$, so the pooled vector is the plain mean and the population standard deviation of the channel. Encoder 2 divides by $T - 1$, so its deviation is larger by $\sqrt{T/(T-1)}$, and the means are equal.
3. The noise of a 1-second utterance is about $\rho^2/s = 4$ against 0.5 for 8 seconds, so the genuine cosine is near 0.2 and not 0.667, and most of them fall under 0.439: 82.9% of the pairs. The impostor cosines come from the speakers' own unit vectors, which the noise barely moves, so the same threshold accepts about the same share, 3.7% against 4.0%.
4. A uniform scale error, such as a doubled output or a dropped scaling factor, gives cosine 1 and relative L2 of 1 or a few percent. For the Qwen3-TTS route, where the vector is put into the Talker's input as it is, it matters; the CosyVoice route normalises first and does not see it.
5. A convolution's work is its weights times the positions of its output. The stem's positions are 100 time columns times up to 80 frequency cells a second; the dense layers work at 50 rows with no frequency axis. Halving the time resolution of the stem would leave the weights as they are and halve its work.
6. The largest pad is 4 frames (kernel 3, dilation 4), and a reflect pad of $p$ needs $p < T$, so $T \ge 5$. With a largest dilation of 6, $p = 6$ and $T \ge 7$, which is $L \ge 1{,}792$ samples at 24 kHz.

**Exercises.**

1. Encoder 1: 76,800 samples and 300 frames. Encoder 2: 51,200 samples, 318 frames and $(318 + 4 - 5)//2 + 1 = 159$ rows.
2. $7 \times 2 \times 5 = 70$ frames; the field of the block is 71.
3. $3 \times 64 \times 64 \times 3 = 36{,}864$ against $256 \times 256 \times 3 = 196{,}608$, a ratio of $5.33$.
4. The means are $(3, 1)$, $h = (3, 1)$, $z = (1.5, -1)$, the gates $\mathrm{sigmoid}(1.5) = 0.8176$ and $\mathrm{sigmoid}(-1) = 0.2689$, and the output $((1.6351, 3.2703), (0.2689, 0.2689))$.
5. The factor $\sqrt{T/(T-1)}$ is below 1.01 from $T = 51$ ($1.00995$); at 50 it is $1.0102$.
6. $5 \times 1{,}125{,}683{,}200 = 5{,}628{,}416{,}000$ operations, of which $5 \times 477{,}696{,}000 = 2{,}388{,}480{,}000$ are in the stem.
7. to 9. Match the printed tables: in 7, the first output is $1.500$ with reflection and $-1.000$ with zeros; in 8, the attentive means are $1.6606$ and $0.7142$ and the deviations $1.0291$ and $1.0712$; in 9, the equal error rates are 33.0%, 23.3%, 12.1% and 4.1%.
10. The approximation assumes that the noise vectors are nearly orthogonal to each other and to the centre, which holds when the width is large. At a width of 16 it holds roughly (the table is within 0.01), and it gets worse for a smaller width, where the cosines of random vectors have a spread of about $1/\sqrt{\text{width}}$.
