<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Project: From Waveform to Speech and Back

*Can you build a pipeline from a waveform to a learned representation and back to generated audio?*

The letter `d` of the language this chapter builds is two sinusoids, one at 700 Hz and one at 5200 Hz, and the recogniser that will read it listens at 8000 samples a second. A signal sampled at 8000 Hz can carry nothing above 4000 Hz, so the second sinusoid has to go. What happens to it depends on the resampler that brings the 12 kHz audio down to 8 kHz. The Kaiser design of [Chapter 1](01-sound.md) removes it: a second of the 5200 Hz tone leaves with an amplitude of $1.9 \times 10^{-5}$. Linear interpolation does not remove it, and does worse than keeping it: the tone arrives at 2800 Hz, where no letter of a speaker with $\mathsf{g} = 1$ has any energy, with an amplitude of 0.604 of what went in. Nothing in the pipeline crashes. The recogniser, which had never made an error on the audio it was trained on, writes the wrong words for almost half of the words it is given.

Here is that pipeline fed in seven ways. The first row is the pipeline as specified; each of the other six differs from it in one thing, and the thing is small enough that a person reading the code would sign it off.

<!-- worked: project_speech_pipeline.opening -->
The test set is the 180 sentences `project_speech_pipeline.test.<n>`, n = 0 .. 179, of milestone 1. 'Word error' is the total of word edits over the reference words of all 180 sentences, 'character error' the total of character edits (spaces included) over their reference characters. Each row is one recogniser trained from `project_speech_pipeline.init` and fed in the way the row says; every number carries only the digits on which the run and its four nudged copies agree.

| how the recogniser was fed | word error | character error |
|---|---:|---:|
| the pipeline as specified | 0.004 | 0.001 |
| test audio resampled by linear interpolation | 0.458 | 0.168 |
| test frames from an HTK-scale filterbank | 0.493 | 0.185 |
| test frames from a filterbank without area normalisation | 0.195 | 0.032 |
| test frames with log10 in place of ln | 0.803 | 0.535 |
| trained with ids not offset past the blank | 1.000 | 0.196 |
| the four-stage quantizer between front end and recogniser | 0.009 | 0.009 |
<!-- /worked -->

Seven pipelines, and the spread between their numbers is the subject of the chapter. The word error is $0.004$ when everything agrees and $0.458$ with the wrong resampler. A filterbank on the wrong frequency scale, one whose triangles are not normalised by their width, a logarithm taken in the wrong base: each costs between a fifth and four fifths of the words. A label table one id off writes no spaces at all, so that every sentence comes out as one long word and the word error is exactly 1, while the character error of that recogniser is $0.196$, because its letters are mostly right; and a quantizer that reduces each frame to 16 bits barely moves the number. The audio chapters taught the parts one at a time, each with its own small world. This project joins them in one world and one loop: a language is spoken by a synthesiser, resampled, turned into log-mel frames, read by a recogniser that has been trained on such audio, and spoken again in a voice estimated from a reference clip, and every joint of the loop is a place where a number can be wrong without anything failing.

The world is small enough to specify on a page, and the project asks you to build it in a language of your choice and to make every number in the acceptance tests come out. There are eight milestones:

1. **A language and a synthesiser.** Four letters, each two sinusoids whose frequencies a speaker parameter scales, a space that is silence, six words, sentences of two or three words; a fully specified renderer at 12 kHz with a random gain and noise, and a stream of sentences each from a generator of its own.
2. **From 12 kHz to a standardized log-mel.** The Kaiser-tapered resampler of Chapter 1 to 8 kHz, the 16-filter Slaney log-mel front end of [Chapter 2](02-spectrograms.md) at 50 frames a second, the standardization that a network needs, and a stacked input of five frames.
3. **A recogniser and its loss.** A two-layer perceptron with six outputs (the blank, the space and four letters), the scaled forward-backward recursion of the loss of [Chapter 4](04-speech-recognition.md), a hand-written backward pass and the check that it is right.
4. **Training, and text to waveform to text.** AdamW, the minibatch loop, greedy decoding, and the word and character error of the round trip, by speaker, with a speaker the recogniser has never heard.
5. **A discrete representation between the front end and the recogniser.** The residual quantizer of [Chapter 3](03-audio-codecs.md) on the standardized frames, and what each stage buys in bits, in error and in words.
6. **A voice, and speaking the transcript again in it.** An embedding by pooled statistics ([Chapter 5](05-speaker-embeddings.md)), a ridge map from it to the speaker parameter, and a loop that recognises a clip and speaks its transcript in the clip's voice ([Chapter 6](06-speech-synthesis.md)).
7. **Three planted mistakes.** An unfiltered resampler, a mel convention mismatch and a label shift, each measured by a number that needs no recogniser and by what it does to the words.
8. **The bill.** The operations and bytes of every stage for a second of audio, and of the training run.

What is real in this list and what is not needs saying plainly. The interfaces are the real ones: a sampling rate at every joint, a frame rate, a standardized log-mel, a blank symbol and a collapse rule, a quantizer that sums codewords, a speaker vector that conditions a synthesiser, an error rate that counts total edits over total reference words. The networks are not. The synthesiser is a specified program of sinusoids where brain's is a language model and a flow decoder; the recogniser has 2,094 weights; and the speaker embedding is a mean of frames. A model that has to be reproduced on a page cannot be a trained network that nobody can specify, so every learned part here is the smallest one that still has the interface and the failure modes of the real one, and the last section says which crate stands behind which piece.

![Two lanes of boxes. Listening: a waveform at 12,000 samples a second goes through a Kaiser sinc resampler to 8,000 a second, a log-mel front end of 50 frames a second of 16 standardized values, an optional residual quantizer of four stages of 16 codewords, and a CTC recogniser of 80, 24 and 6 values with greedy decoding, ending in text. Speaking: a reference clip enrols a speaker embedding of 16 values, a ridge map turns it into a speaker parameter, and a synthesiser of two sinusoids a letter scaled by that parameter speaks the transcript from the recogniser, producing generated audio at 12,000 a second.](img/speech-pipeline.svg)

The figure is a valid build order. The synthesiser comes first because every test of every later stage is made on its audio, and its text is the answer key: unlike a recording, this audio comes with the words that were said in it. The resampler and the front end come next because the recogniser, the quantizer and the voice all read their output. The recogniser must be checked before it is trained, and trained before the representation and the voice are measured through it. Build and check each milestone before the next; a front end with the wrong logarithm and a recogniser with a wrong gradient both train to something, and the only protection is a number that each stage has to meet, which the milestones end with and the **acceptance tests** collect. You need the book's generator (Volume I, Chapter 17), double-precision arithmetic with a correctly rounded `sin`, `cos`, `exp`, `log` and `tanh`, and a radix-2 FFT (the transform of Volume I; a transform by the definition gives the same powers at $129 \times 256$ complex multiply-adds a frame in place of the $N \log_2 N$ butterflies).

Six questions bring back what the milestones lean on. Answer each before reading on; the answers are at the end.

1. (Volume IX, [Chapter 1](01-sound.md), aliasing.) A cosine of 5200 Hz is kept at one sample in 1.5 of a signal sampled at 12,000 Hz, with no filtering first. At what frequency does it appear, and why does a Kaiser-tapered sinc remove it where linear interpolation does not?
2. (Volume IX, [Chapter 2](02-spectrograms.md), frames and gain.) A signal of 5280 samples is padded by $N/2$ on each side by reflection, with $N = 256$ and a hop of 160. How many frames are there? If the signal is made three times louder, what happens to a log-mel value computed as $\ln(E + 10^{-6})$, and what does subtracting the largest value of the clip do about it?
3. (Volume IX, [Chapter 3](03-audio-codecs.md), the residual.) A quantizer has two stages. What does the second stage quantize, what is the reconstruction, and what does the squared error of the reconstruction do as stages are added?
4. (Volume IX, [Chapter 4](04-speech-recognition.md), the loss.) For a recogniser with a blank, a softmax in every frame and a loss $-\ln P(\text{label})$ summed over all alignments, what is the gradient of the loss with respect to the logits of frame $t$, and what is the rule that turns a path of frame labels into a transcript?
5. (Volume IX, [Chapter 6](06-speech-synthesis.md), conditioning.) A synthesiser is told the voice of a reference clip. Which vector does it receive, and how does the reference clip become that vector?
6. (Volume III, [Chapter 6](../v03-neural-networks/06-gradient-checking.md), gradient checks.) What is compared with what, at which step size, and what number says that a hand-written backward pass is right?

## Milestones

The pseudocode follows [Pseudocode Conventions](../pseudocode.md). Every random input comes from the book's SplitMix64 at a seed that is the rule `seed_of(name)` (the first 8 bytes of the SHA-256 digest of the UTF-8 name, read as a big-endian 64-bit integer, as [Pseudocode Conventions](../pseudocode.md) defines it) applied to a name printed beside the table, and the draws are taken in the order the text states. A sum may be accumulated in any order at least as accurate as left to right: no digit of an untrained number printed here depends on the last bit of a sum. Every trained number is printed only to the digits that survive the nudge rule of this book (`scripts/book/worked/stable.py`, taught with Volume VIII's first trained tables): the run is repeated from four starting points whose first weight, here the first weight of the first layer, differs by $+10^{-12}$, $-10^{-12}$, $+3 \times 10^{-12}$ and $-3 \times 10^{-12}$, and a digit is printed only if it is the same in all five runs. Here all five agree on every digit shown. The rule is what lets the tables say so, and what lets a reimplementation in another language, whose `tanh` differs in the last bit, check itself against them.

### Milestone 1: a language and a synthesiser

The alphabet has six symbols with ids chosen for the loss that will read them: `0` is the blank, `1` the space, `2` to `5` the letters `a`, `b`, `c`, `d`. The vocabulary is six words, in this order, `ab`, `cad`, `bd`, `dca`, `bca`, `adb`, and a sentence is two or three of them separated by single spaces. A letter is two sinusoids and a space is silence; the table gives their frequencies and amplitudes. The letter `d` is the one with a component above 4000 Hz.

A **speaker** is one number, the speaker parameter $\mathsf{g}$ that multiplies every frequency. A longer vocal tract has lower formants, and the project's one-parameter version of that fact is $f \mapsto \mathsf{g} f$. Six speakers are fixed: the first five are heard in training and the sixth never is, with a parameter between two of the others, so that the question "does the pipeline work for a voice it was not trained on" has an answer.

```text
function synthesise(text: text, g: real, rng: Generator) -> array<real>[n]:
    require every character of text is one of the letters or the space, "the language"
    durs = [rng.integer(3) for each symbol of text, in order]         # one draw per symbol
    gain = 0.5 + rng.uniform()
    sizes = [12 * (40 + 20 * d if symbol is the space else 60 + 20 * d) for each (symbol, d)]   # samples at 12 kHz
    n = 2 * 1200 + sum(sizes)
    x = zeros(real, [n])
    pos = 1200                                                         # 100 milliseconds of silence before and after
    for each symbol of text with its size m:
        if symbol is a letter:
            for i in 0 .. m:
                u = min(1.0, (i + 0.5) / 120, (m - i - 0.5) / 120)     # a ramp of 10 milliseconds at each end
                env = 0.5 - 0.5 * cos(pi * u)
                s = sum(amp * sin(2 * pi * g * f * i / 12000) for (f, amp) in the letter's two sinusoids)
                x[pos + i] = gain * env * s
        pos = pos + m
    for i in 0 .. n:                                                   # in order of i
        x[i] = x[i] + 0.02 * rng.normal()
    return x
```

A sentence is drawn from a generator of its own so that any one of them can be reproduced alone: sentence $n$ of the training set uses the generator `Generator(seed_of("project_speech_pipeline.train." + n))`, and of the test set `... .test.` $n$. Its draws are, in order: the number of words $2 + $`integer(2)`, one `integer(6)` for each word, and then the draws of `synthesise` (one `integer(3)` for each character, one `uniform()`, one `normal()` for each sample). The speaker of training sentence $n$ is $n \bmod 5$ and of test sentence $n$ is $n \bmod 6$. There are 120 training and 180 test sentences.

<!-- worked: project_speech_pipeline.world -->
| letter | id | first sinusoid | second sinusoid |
|---|---:|---|---|
| `a` | 2 | 350 Hz x 0.5 | 1000 Hz x 0.4 |
| `b` | 3 | 350 Hz x 0.5 | 2200 Hz x 0.4 |
| `c` | 4 | 700 Hz x 0.5 | 1400 Hz x 0.4 |
| `d` | 5 | 700 Hz x 0.5 | 5200 Hz x 0.4 |

Every sinusoid of a letter is amplitude times sin(2 pi g f i / 12000) for sample i of the letter, g the speaker parameter. The ids of the other two symbols are 0 for the blank and 1 for the space. The six words, in the order of the index `integer(6)` draws: `ab`, `cad`, `bd`, `dca`, `bca`, `adb`.

| speaker | g | role | letter `a` first / second frequency (Hz) |
|---:|---:|---|---|
| 0 | 0.88 | heard in training | 308.0 / 880.0 |
| 1 | 0.94 | heard in training | 329.0 / 940.0 |
| 2 | 1 | heard in training | 350.0 / 1000.0 |
| 3 | 1.06 | heard in training | 371.0 / 1060.0 |
| 4 | 1.12 | heard in training | 392.0 / 1120.0 |
| 5 | 1.03 | never heard | 360.5 / 1030.0 |

The first four training sentences, each from its own stream `project_speech_pipeline.train.<n>` (`seed_of` of the name, as defined in Pseudocode Conventions: the first 8 bytes of the SHA-256 digest of its UTF-8 bytes, big-endian; the seeds of n = 0 to 3 are 0xe35b1d7f6ff21bfa, 0xd92273c77f406b92, 0xcdbe9fa26dbaa9b5, 0x7e5aed80686c7ff2): the sentence, the speaker (n mod 5 for the training sentences), the samples at 12 kHz, the samples after the resampler of milestone 2, the frames of milestone 2, the sum of the samples and the sum of their squares (6 decimals, tolerance 1e-6).

| n | sentence | speaker | samples at 12 kHz | samples at 8 kHz | frames | sum | sum of squares |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `ab adb` | 0 | 7920 | 5280 | 34 | 0.198549 | 346.579756 |
| 1 | `bd adb bca` | 1 | 11040 | 7360 | 47 | -0.831011 | 2515.427604 |
| 2 | `dca cad` | 2 | 8880 | 5920 | 38 | -2.394460 | 1928.254957 |
| 3 | `bd ab` | 3 | 6960 | 4640 | 30 | 0.833567 | 938.733665 |
<!-- /worked -->

The sums of samples and of their squares are the first tests of the generator and of the order of draws: a duration drawn before the words instead of after them, or a sine written as a cosine, moves them at once. **Milestone 1 ends** when your first four training sentences agree with the table to the digits printed (the text and the lengths exactly; the sums to $10^{-6}$).

### Milestone 2: from 12 kHz to a standardized log-mel

The recogniser listens at 8000 samples a second, so the audio is resampled by the ratio $8000/12000 = 2/3$. Use the Kaiser-tapered sinc of Chapter 1 with its stated design (80 dB of attenuation, 32 zero crossings, a cutoff at the new Nyquist rate); the signal is zero outside its samples, and the output has $\lceil 2n/3 \rceil$ samples for $n$ input samples. As a reproduction check, the table gives the amplitude at the output of five one-second tones.

<!-- worked: project_speech_pipeline.resampler -->
The reduced ratio of 12000 to 8000 is 2 / 3; the Kaiser design of Chapter 1 has beta = 7.8573, a reach of 96 units (so 97 input samples at most under one output) and a cutoff at 0.6667 of the input's Nyquist rate, which is the output's. One second of cos(2 pi f i / 12000), amplitude 1; the amplitude of the output at the frequency it should have after resampling (the frequency itself below 4000 Hz, 8000 minus it above), measured over output samples 500 to 2499, a whole number of cycles in each case.

| input tone (Hz) | output tone (Hz) | the tone should be | Kaiser sinc | linear interpolation |
|---:|---:|---|---:|---:|
| 1000 | 1000 | kept | 1.0000 | 0.9830 |
| 3000 | 3000 | kept | 1.0000 | 0.8536 |
| 3900 | 3900 | kept, in the transition band | 0.8141 | 0.7612 |
| 4400 | 3600 | must go | 8.66e-05 | 0.7034 |
| 5200 | 2800 | must go | 1.94e-05 | 0.6040 |
<!-- /worked -->

The two tones above 4000 Hz show what the resampler is for. The tone at 5200 Hz, the one the letter `d` carries, leaves the Kaiser design at $2 \times 10^{-5}$ and leaves linear interpolation at 0.604, now at 2800 Hz. The first two rows show the other side of the bargain: linear interpolation shrinks the passband itself (to 0.854 at 3000 Hz, by a filter that nobody specified) where the sinc design keeps it flat to four decimals. The tone at 3900 Hz lies in the transition band, where the Kaiser design is falling towards the cutoff; the project's letters keep out of it, the highest component they keep being 2464 Hz (the second sinusoid of `b`, 2200 Hz times the largest $\mathsf{g}$, 1.12) and the 5200 Hz component being at 4576 Hz or more for every speaker, above the band.

The front end is Chapter 2's: reflect-pad by $N/2$, frames of $N = 256$ samples at a hop of 160 (32 and 20 milliseconds, 50 frames a second), the periodic Hann window, the power of bins $0 \ldots 128$, 16 Slaney triangles from 0 to 4000 Hz, each linear in hertz between its edges and multiplied by $2/(\text{right edge} - \text{left edge})$, and the logarithm with a guard. A clip of $n$ samples gives $1 + \lfloor n / 160 \rfloor$ frames.

```text
function log_mel_frames(y: array<real>[n], bank: array<real>[16, 129]) -> array<real>[T, 16]:
    p = reflect_pad(y, 128)                                  # Chapter 2: 128 samples mirrored at each end
    T = 1 + n // 160
    for t in 0 .. T:
        X = fft(p[160 * t .. 160 * t + 256] * hann_periodic(256))        # elementwise product, complex bins
        for k in 0 .. 129:
            P[t, k] = real(X[k])^2 + imag(X[k])^2
        for m in 0 .. 16:
            E[t, m] = sum(bank[m, k] * P[t, k] for k in 0 .. 129)        # k increasing
            L[t, m] = log(E[t, m] + 0.000001)
    return (L - max(L)) / 4.0                                # the clip's largest value subtracted, then divided by 4
```

Two steps follow that the log-mel of Chapter 2 does not need and a network does. Subtracting the largest value of the clip makes a gain a non-event, as the second recall question said: a clip made three times louder has the same frames. And the network must not be handed numbers whose means are all near $-2$ and whose spreads run from 0.17 to 1.0 by band (the table below), so each of the 16 bands is **standardized**: the mean and the standard deviation (divisor $n$) of that band over every frame of the 120 training sentences, fixed once and applied to every clip afterwards, including the test clips and the clips of any later stage. This is not a detail: the initial weights of milestone 3 are scaled for inputs of about unit size, and a band whose values sit near $-2$ with a spread of 0.17 would give the first layer inputs that carry a constant and little else. The recogniser reads **stacked** frames: the input of frame $t$ is the concatenation of the standardized frames $t - 2, \ldots, t + 2$ (80 values), the indices clamped to $0 \ldots T - 1$ so that the first and last frames repeat themselves.

<!-- worked: project_speech_pipeline.front_end -->
Filterbank: 16 Slaney triangles (Chapter 2) over 0 to 4000 Hz at 8000 Hz and N = 256 (bins at 31.25 Hz), area normalised (each row multiplied by 2 / (right edge - left edge)); columns are the three edges in hertz, the number of nonzero weights in the row and the row's sum.

| filter | left | centre | right | nonzero weights | row sum |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 | 137.90 | 275.79 | 8 | 0.03172 |
| 1 | 137.90 | 275.79 | 413.69 | 9 | 0.03211 |
| 2 | 275.79 | 413.69 | 551.59 | 9 | 0.03201 |
| 3 | 413.69 | 551.59 | 689.49 | 9 | 0.03182 |
| 4 | 551.59 | 689.49 | 827.38 | 9 | 0.03229 |
| 5 | 689.49 | 827.38 | 965.28 | 8 | 0.03172 |
| 6 | 827.38 | 965.28 | 1112.27 | 9 | 0.03223 |
| 7 | 965.28 | 1112.27 | 1282.25 | 11 | 0.03178 |
| 8 | 1112.27 | 1282.25 | 1478.20 | 12 | 0.03218 |
| 9 | 1282.25 | 1478.20 | 1704.09 | 13 | 0.03194 |
| 10 | 1478.20 | 1704.09 | 1964.51 | 15 | 0.03196 |
| 11 | 1704.09 | 1964.51 | 2264.73 | 18 | 0.03205 |
| 12 | 1964.51 | 2264.73 | 2610.82 | 21 | 0.03198 |
| 13 | 2264.73 | 2610.82 | 3009.80 | 24 | 0.03200 |
| 14 | 2610.82 | 3009.80 | 3469.76 | 28 | 0.03199 |
| 15 | 3009.80 | 3469.76 | 4000.00 | 31 | 0.03201 |

The radix-2 transform of the first frame of training sentence 0, against the transform by definition (Chapter 2's `power_bins`): largest relative difference of the 129 powers 1.0e-14, below 1e-9. The log of Chapter 2's recipe A with the guard 1e-06, the sentence's largest value subtracted and a division by 4, then per-band standardisation with the mean and standard deviation (divisor n) over every frame of the 120 training sentences:

| band | mean | standard deviation |
|---:|---:|---:|
| 0 | -2.2094 | 0.2858 |
| 1 | -1.4577 | 0.9971 |
| 2 | -1.4331 | 1.0036 |
| 3 | -1.8357 | 0.6973 |
| 4 | -1.6352 | 0.9696 |
| 5 | -1.7137 | 0.8753 |
| 6 | -1.8475 | 0.7956 |
| 7 | -1.9353 | 0.7268 |
| 8 | -2.0492 | 0.6182 |
| 9 | -2.0943 | 0.6045 |
| 10 | -2.1480 | 0.4838 |
| 11 | -2.0843 | 0.5901 |
| 12 | -2.0146 | 0.6344 |
| 13 | -2.1789 | 0.4399 |
| 14 | -2.2933 | 0.1765 |
| 15 | -2.2973 | 0.1728 |

Training sentence 0 (`ab adb`) has 34 frames of 16 values; the sum of its 544 standardized values is 200.399012 and the first four entries of the stacked vector of frame 0 are 0.948845, -0.692724, -0.689101, -0.620758 (frame 0's own frames -2 and -1 are copies of frame 0, tolerance 1e-6).
<!-- /worked -->

The transform in the code of this project is a radix-2 FFT, and it agrees with the transform by definition (Chapter 2's `power_bins`) to a relative $10^{-14}$; the rows of the table show the Slaney bank is what Chapter 2 derived, with row sums that differ by about 2% across the bank (the width normalization at work). **Milestone 2 ends** when your five tone amplitudes agree with the table to $10^{-4}$, your filter edges to $0.01$ Hz, and your standardized frames of training sentence 0 sum to the number printed.

### Milestone 3: a recogniser and its loss

The network is a perceptron with one hidden layer and a head of six outputs. For the stacked input $\mathbf{x}_t$ of 80 values, a hidden layer of 24 units with $\tanh$, and logits $\mathbf{z}_t = \mathbf{W}_2 \mathbf{h}_t + \mathbf{b}_2$ with a softmax in every frame:

$$
\mathbf{h}_t = \tanh(\mathbf{W}_1 \mathbf{x}_t + \mathbf{b}_1), \qquad \mathbf{p}_t = \mathrm{softmax}(\mathbf{W}_2 \mathbf{h}_t + \mathbf{b}_2).
$$

`tanh` is the correctly rounded library function. The parameters are initialized from the generator `Generator(seed_of("project_speech_pipeline.init"))` by one `normal()` per value in the order: the 1,920 entries of $\mathbf{W}_1$ (24 rows of 80) in row-major order, each times $\sqrt{1/80}$; then the 144 entries of $\mathbf{W}_2$ (6 rows of 24) in row-major order, each times $\sqrt{1/24}$. Both biases start at 0. The flat parameter vector, in the order the book's gradient checks use, is $\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2$: 2,094 values.

The loss of one sentence is the CTC loss of Chapter 4: the negative logarithm of the summed probability of every alignment that collapses to the sentence's label sequence. Chapter 4 wrote the recursion in probabilities, and said that a trainer computes it in logarithms. A product of one probability for every frame underflows a double for a long enough sentence; the recursion here avoids that with a different device that is easier to write, scaling. Every frame of $\alpha$ is divided by its own sum before the next frame is computed, the logarithms of those sums are added up and added back into the loss at the end, and the same is done to $\beta$ (whose own scales cancel when the soft alignment is normalized).

```text
function ctc_scaled(P: array<real>[T, K], label: list<int>) -> (real, array<real>[T, K]):
    require T >= 1 and len(label) >= 1, "a sentence has at least one symbol"
    e = [0]                                            # the extended label: blank, l1, blank, l2, ..., blank
    for l in label: e = e + [l, 0]
    S = len(e)
    skip[s] = (s >= 2 and e[s] != 0 and e[s] != e[s - 2])
    alpha[0, 0] = P[0, e[0]];  alpha[0, 1] = P[0, e[1]]; the other states of frame 0 are 0
    c = sum(alpha[0, :]);  alpha[0, :] = alpha[0, :] / c;  lnscale = log(c)
    for t in 1 .. T:
        for s in 0 .. S:
            a = alpha[t - 1, s] + (alpha[t - 1, s - 1] if s >= 1 else 0.0) + (alpha[t - 1, s - 2] if skip[s] else 0.0)
            alpha[t, s] = a * P[t, e[s]]
        c = sum(alpha[t, :]);  alpha[t, :] = alpha[t, :] / c;  lnscale = lnscale + log(c)
    loss = -(lnscale + log(alpha[T - 1, S - 1] + alpha[T - 1, S - 2]))
    beta[T - 1, S - 1] = P[T - 1, e[S - 1]];  beta[T - 1, S - 2] = P[T - 1, e[S - 2]]; the others 0; normalise the frame by its sum
    for t from T - 2 down to 0:
        for s in 0 .. S:
            b = beta[t + 1, s] + (beta[t + 1, s + 1] if s + 1 < S else 0.0) + (beta[t + 1, s + 2] if s + 2 < S and skip[s + 2] else 0.0)
            beta[t, s] = b * P[t, e[s]]
        normalise the frame by its sum
    for t in 0 .. T:                                   # the soft alignment: share of the paths with symbol k in frame t
        gamma[t, k] = sum(alpha[t, s] * beta[t, s] / P[t, e[s]] for the s with e[s] == k)
        gamma[t, :] = gamma[t, :] / sum(gamma[t, :])
    return (loss, gamma)
```

The soft alignment normalized per frame is the quantity Chapter 4 proved equal to $\gamma_t(k)$, and its use is the gradient: for a softmax output the derivative of the loss with respect to the logits of frame $t$ is $\mathbf{p}_t - \boldsymbol{\gamma}_t$. The backward pass through the network is then the ordinary one, written by hand for the two layers, with the arriving gradient $\mathbf{d}\mathbf{z}_t = \mathbf{p}_t - \boldsymbol{\gamma}_t$ in every frame: $\partial \mathbf{W}_2 \mathrel{+}= \mathbf{d}\mathbf{z}_t \mathbf{h}_t^{\top}$, $\partial \mathbf{b}_2 \mathrel{+}= \mathbf{d}\mathbf{z}_t$, $\mathbf{d}\mathbf{h}_t = \mathbf{W}_2^{\top} \mathbf{d}\mathbf{z}_t$, $\mathbf{d}\mathbf{a}_t = \mathbf{d}\mathbf{h}_t \odot (1 - \mathbf{h}_t^2)$, $\partial \mathbf{W}_1 \mathrel{+}= \mathbf{d}\mathbf{a}_t \mathbf{x}_t^{\top}$ and $\partial \mathbf{b}_1 \mathrel{+}= \mathbf{d}\mathbf{a}_t$, summed over the frames of the sentence. No gradient is computed for the input, which is data.

<!-- worked: project_speech_pipeline.recogniser -->
| parameter | shape | values | initial value |
|---|---:|---:|---|
| W1 | 24 x 80 | 1920 | N(0, 1 / 80) |
| b1 | 24 | 24 | 0 |
| W2 | 6 x 24 | 144 | N(0, 1 / 24) |
| b2 | 6 | 6 | 0 |
| all |  | 2094 |  |

Initialised from `project_speech_pipeline.init` (0x345403b1ebb2269f): the 1920 values of W1 in row-major order, each scale times one normal(), then the 144 of W2. The first three of W1 are 0.094268, 0.100876, 0.062690, the first three of W2 are -0.033022, 0.091489, -0.048564. The softmax row of frame 0 of training sentence 0 at this start is 0.142341, 0.200727, 0.056612, 0.131327, 0.254092, 0.214901 (tolerance 1e-6).

The scaled recursion of the CTC loss on the 14 frames 3 .. 16 of that sentence with the label `ab` gives 20.3549071164; Chapter 4's recursion in probabilities, summed over the two last states, gives 20.3549071164; they agree to within 1e-12, and every row of the soft alignment sums to 1 to within 1e-12. The hand-written backward pass against central differences with step 1e-5 on each of the 2094 parameters: the largest absolute difference is below 1e-6.
<!-- /worked -->

Three checks come before any training, in this order, because each stage of the code is trusted only after the stage under it is. The first is the softmax row of frame 0: if it does not agree with the table, the initialization or the standardization is wrong, and no later number can be. The second is the recursion: your scaled loss must agree with the unscaled recursion of Chapter 4 on a short segment, and the table gives both (they agree to within $10^{-12}$, the scaling being exact in real arithmetic). The third is the gradient: every one of the 2,094 parameters is moved by $\pm 10^{-5}$ and the central difference of the loss compared with the hand-written gradient, on a segment of 14 frames with the two-symbol label `ab`. The criterion is the one Volume III set: the largest difference must be below $10^{-6}$, and the table reports it against that criterion because a difference at a finite step of $10^{-5}$ is truncation and rounding noise, not a property of the code. **Milestone 3 ends** when all three checks pass.

### Milestone 4: training, and text to waveform to text

Training is the loop of Volume III with the loss of this chapter. The 120 training sentences are drawn in batches of four. The list $0, 1, \ldots, 119$ is made once, before the first epoch, and every epoch starts by shuffling that same list in place (the `shuffle` of the pseudocode conventions permutes its argument) from the generator `Generator(seed_of("project_speech_pipeline.order"))`, one continuing generator for the whole run, so that the order of epoch $e$ is a shuffle of the order of epoch $e - 1$ and not of the identity; four consecutive sentences are taken per step, 30 steps an epoch. The gradient of a step is the mean over its four sentences, with the loss of each sentence a sum over its frames, not a mean. The optimizer is AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, a rate of $0.01$ held constant, no weight decay and no clipping. Twenty epochs are 600 steps.

```text
function train_recogniser(sentences: list<(array<real>[T, 80], list<int>)>, theta: array<real>[2094], rng: Generator):
    require len(sentences) == 120, "30 batches of 4"
    m = zeros(real, [2094]);  v = zeros(real, [2094]);  step = 0
    order = list(0 .. 120)                                               # made once, kept across epochs
    for epoch in 0 .. 20:
        shuffle(order, rng)                                              # in place: the permutation of the last epoch's order
        for b in 0 .. 30:
            g = zeros(real, [2094])
            for i in order[4 * b .. 4 * b + 4]:
                loss_i = ctc_loss_and_gradient(theta, sentences[i], g)      # g += the gradient of the sentence
            g = g / 4.0
            step = step + 1
            adamw_step(theta, g, m, v, step, 0.01, 0.0)                     # Volume III; decay 0 here
```

The nudge moves $\mathbf{W}_1[0][0]$, the weight from the first value of the leftmost stacked frame to the first hidden unit, by the stated amount before the first step. It is a weight the computation uses on every frame, so the nudge changes everything downstream and the stable digits mean something. The tables of the chapter that come from trained networks are for the run and its four nudged copies.

<!-- worked: project_speech_pipeline.training -->
20 epochs of 30 steps at batch 4; one list `range(120)` is made once and `shuffle`d in place at the start of every epoch (so epoch e's order is the shuffle of epoch e - 1's, not of the identity) with the one continuing stream `project_speech_pipeline.order` (0x50b5b1ea0353c424), and four consecutive sentences are taken per step. AdamW (0.9, 0.999, 1e-8), rate 0.01, no decay, no clipping, the gradient the mean over the batch. The first weight W1[0][0] is the one the nudges move. The table is the mean over the epoch's sentences of the CTC loss each had when its batch was formed (so the weights moved during the epoch). 'Clean' uses the ids of the alphabet; 'shifted' is the planted mistake of milestone 7.

| epoch | steps so far | clean: mean loss | shifted: mean loss |
|---:|---:|---:|---:|
| 1 | 30 | 14.686 | 22.948 |
| 2 | 60 | 2.967 | 5.495 |
| 3 | 90 | 1.411 | -0.803 |
| 5 | 150 | 0.385 | -3.310 |
| 10 | 300 | 0.076 | -4.252 |
| 20 | 600 | 0.020 | -4.679 |
<!-- /worked -->

The loss of the correct run falls from 14.686 to 0.020 in twenty epochs, and is 0.385 by the fifth. The second column, the planted mistake of milestone 7, goes below zero. The loss $-\ln P$ of a probability is never negative, so a CTC loss below zero means the lattice counts more than all the paths: a label that equals the blank. It is a symptom to remember for the milestone that plants it.

Decoding is greedy, as in Chapter 4: in every frame take the first largest entry of the softmax row, merge neighbouring equal symbols, drop the blanks. The **word error rate** is the corpus rate of Chapter 4: the total of word edits (insertions, deletions and substitutions, by the alignment of Chapter 4) over the total number of reference words, so that a long sentence weighs more than a short one. The **character error rate** is the same count over the characters of the transcripts, spaces included, edits over the total number of reference characters. The test is the round trip the title asks for. A sentence of the test set is synthesised (milestone 1), resampled, framed, recognised, and the transcript compared with the text the synthesiser was given.

<!-- worked: project_speech_pipeline.round_trip -->
The test set holds 456 reference words and 1479 reference characters. Text to waveform to text: each sentence is synthesised (milestone 1), resampled, framed, recognised greedily (the first largest entry of each frame, repeats collapsed, blanks dropped) and compared with the text it was made from. The test sentences are `project_speech_pipeline.test.<n>` with speaker n mod 6.

| speaker | g | in training | sentences | word error | character error |
|---:|---:|---|---:|---:|---:|
| 0 | 0.88 | heard | 30 | 0.000 | 0.000 |
| 1 | 0.94 | heard | 30 | 0.000 | 0.000 |
| 2 | 1 | heard | 30 | 0.000 | 0.000 |
| 3 | 1.06 | heard | 30 | 0.000 | 0.000 |
| 4 | 1.12 | heard | 30 | 0.025 | 0.004 |
| 5 | 1.03 | never heard | 30 | 0.000 | 0.000 |
| training set |  | heard | 120 | 0.000 | 0.000 |
| test set |  |  | 180 | 0.004 | 0.001 |
<!-- /worked -->

The recogniser has no error on four of the five speakers it heard in training and none on the training sentences themselves. The one nonzero row is the highest heard voice, $\mathsf{g} = 1.12$, the edge of what the training set covers (the table cannot say whether that is the edge or chance). The never-heard speaker, whose parameter lies between two heard ones, is read without error: the recogniser has interpolated between voices. What the table does not show, and what the next milestones will, is how narrow the margin is. **Milestone 4 ends** when your loss after epoch 1 is $14.686$, after epoch 20 is $0.020$, and your word and character error on the test set are those of the last row.

### Milestone 5: a discrete representation between the front end and the recogniser

A recogniser reads continuous frames; a language model, as Chapter 3 told, reads integers. The codec of Chapter 3 is a **residual vector quantizer**: stage 1 replaces a vector by the nearest codeword of its codebook, stage 2 quantizes what stage 1 left, and so on, and the reconstruction is the sum of the chosen codewords. The project builds a miniature of it on the thing the recogniser reads, a standardized log-mel frame of 16 values, with $Q = 4$ stages of $K = 16$ codewords, 4 bits a stage.

The codebooks are trained stage by stage with the Lloyd rounds of Chapter 3: stage 1 on every third standardized frame of the 120 training sentences (1,688 frames: the frames of all the sentences, in sentence and frame order, make one list of 5,064, and its entries 0, 3, 6, $\ldots$ are taken, the count not restarted at a sentence); the starting codewords are the first 16 rows of the frames taken in the order of one `shuffle` of the row indices, the generator `Generator(seed_of("project_speech_pipeline.rvq"))` continuing across stages; ten rounds of assigning each row to its nearest codeword (the lower index on a tie) and replacing each codeword that has members by their mean (a codeword with none stays); and stage $q + 1$ is trained on the residuals that the $q$ trained stages leave. The nudge moves the first coordinate of the first starting codeword of every stage. The test frames are quantized, the codewords of the first $q$ stages summed, and the recogniser of milestone 4 reads the result unchanged. It was not retrained on quantized frames, and this is a measurement of what the representation keeps, not of what a model trained on it could do.

<!-- worked: project_speech_pipeline.rvq_table -->
A residual quantizer of the standardized 16-value frames: 4 stages of 16 codewords, trained by 10 Lloyd rounds on every 3rd frame of the training sentences: the standardized frames of all 120 sentences are concatenated in sentence order and the entries 0, 3, 6, ... of that one list are taken, the count never restarted per sentence (1688 frames), starting centres from one `shuffle` of the rows per stage with `project_speech_pipeline.rvq` (0xa5e29ddad470e1f7). The test frames are quantized and summed back, stacked, and read by the recogniser of milestone 4, unchanged. Bits are per frame; the mean squared error is the sum over the 16 values of the squared difference, averaged over the test frames (q = 0 returns zeros).

| stages | bits per frame | bits per second | squared error per frame | word error | character error |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 15.91 | 1.000 | 1.000 |
| 1 | 4 | 200 | 4.12 | 0.110 | 0.042 |
| 2 | 8 | 400 | 2.04 | 0.018 | 0.012 |
| 3 | 12 | 600 | 1.39 | 0.004 | 0.006 |
| 4 | 16 | 800 | 1.03 | 0.009 | 0.009 |
<!-- /worked -->

Two things in the table deserve a pause. The squared error per frame falls at each stage, by a factor of 3.9, then 2.0, 1.5 and 1.3, the shrinking returns that Chapter 3 derived for a residual chain. The word error does not follow it at the end: 0.004 at three stages and 0.009 at four, a difference of two words on 456, at an error per frame that is smaller at four. A frame's error is not a transcript's: the recogniser needs the frame to be on the right side of a decision between letters, and by three stages it is, so that a better frame cannot help any more and a small shift in a codeword can move a boundary frame either way. The table is honest about the other end as well: one stage, at 4 bits a frame and 200 bits a second, already gives a word error of 0.110. The bit rate is the point of the stage: eight thousand 16-bit samples a second are 128,000 bits, and four stages carry the same transcript in 800, a factor of 160.

**Milestone 5 ends** when your squared errors after each stage agree with the table to two decimals and your word errors to the digits shown; the first is a test of the quantizer alone, and the second a test that it was trained on the standardized frames and not on something else.

### Milestone 6: a voice, and speaking the transcript again in it

The pipeline has so far gone from audio to text. The way back, from text to audio, is the synthesiser of milestone 1, and what it lacks is a voice taken from a clip. In brain the voice is a vector from a speaker encoder that conditions the synthesiser (Chapter 5 built the encoder, Chapter 6 the conditioning). The project's pair is as small as the parts allow.

The **embedding** of a clip is the mean of its standardized log-mel frames over the frames that carry a letter, the **active** frames, those whose largest value before standardization is above $-0.5$ (the peak-normalized scale of milestone 2, so 8.7 decibels below the clip's peak), minus the mean of that quantity over the 120 training sentences, made a unit vector of 16 values. It is the pooled-statistics idea of Chapter 5 with only the mean kept. Two clips are compared by the dot product of their unit vectors, and a set of trials is scored with the equal error rate of Chapter 5.

The **voice map** is a ridge regression from the embedding to $\ln \mathsf{g}$, fitted on the 120 training sentences (five speakers, 24 each): with $\mathbf{E}$ the matrix of the training embeddings centred on their mean, $\mathbf{y}$ the centred targets $\ln \mathsf{g}$, the weights solve $(\mathbf{E}^{\top}\mathbf{E} + 10^{-3}\mathbf{I})\mathbf{w} = \mathbf{E}^{\top}\mathbf{y}$, by Gaussian elimination with partial pivoting, and the intercept makes the fit pass through the means. The estimate of a set of clips is $\hat{\mathsf{g}} = \exp(\text{mean over the clips of } \mathbf{w} \cdot \mathbf{e} + b)$.

<!-- worked: project_speech_pipeline.speaker_block -->
The embedding of a clip: the mean of its standardized frames whose largest value before standardizing is above -0.5, minus the mean of that quantity over the training sentences, made a unit vector. Over the 180 test clips, 2,610 same-speaker and 13,500 different-speaker pairs scored by the dot product of the unit vectors (summed over the 16 values in order): the mean same-speaker cosine is 0.250, the mean different-speaker cosine -0.045, and the equal error rate (Chapter 5's rule) is 39.1% at the threshold 0.115.

| speaker | true g | role | clips | mean g, one clip | standard deviation | g, eight clips |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 0.88 | heard | 30 | 0.8997 | 0.0276 | 0.8866 |
| 1 | 0.94 | heard | 30 | 0.9436 | 0.0247 | 0.9333 |
| 2 | 1 | heard | 30 | 0.9994 | 0.0178 | 0.9950 |
| 3 | 1.06 | heard | 30 | 1.0654 | 0.0241 | 1.0633 |
| 4 | 1.12 | heard | 30 | 1.0998 | 0.0297 | 1.0992 |
| 5 | 1.03 | never heard | 30 | 1.0463 | 0.0248 | 1.0531 |

The map from embedding to speaker parameter is ridge regression of ln g on the embedding over the 120 training sentences (the five heard speakers), penalty 0.001, solved by Gaussian elimination with partial pivoting on the 16 normal equations; its intercept is -0.0036 and its largest weight in magnitude 0.275. 'One clip' averages g over each clip estimated alone and gives the spread (the standard deviation with divisor n, n the 30 clips of the speaker); 'eight clips' estimates g from the first 8 clips of the speaker together (the mean of ln g over them).
<!-- /worked -->

The embedding is crude, and the table is candid about it. Over single test clips the equal error rate is 39.1%, because six speakers whose parameters differ by 6% must be separated by a mean of frames that also depends on which letters the sentence happened to use. One clip gives the speaker parameter to a standard deviation of 0.018 to 0.030, and the estimates at the two ends are shrunk towards the middle (1.0998 for a true 1.12, 0.8997 for a true 0.88). Averaging eight clips narrows it: all six estimates are within 0.023 of the truth. The pattern of Chapter 5 returns, that an embedding is as good as the seconds that went into it, and the loop below uses eight clips to enrol.

The loop is the pipeline of the title. Speaker 5, the one the recogniser and the map have never heard, enrols with its first eight test clips: their embeddings give $\hat{\mathsf{g}}$ and a reference embedding. For each of the next sixteen clips, the recogniser writes the transcript; the synthesiser speaks that transcript, from the generator `Generator(seed_of("project_speech_pipeline.voice." + n))` where $n$ is the clip's index among the speaker's clips (so that the durations, the gain and the noise are the same for every voice and only $\mathsf{g}$ differs), once in the estimated voice and once in the voice of each heard speaker; each re-spoken clip is resampled and its embedding compared with the reference. And the estimated version is recognised again, which closes the loop: waveform, text, waveform, text.

<!-- worked: project_speech_pipeline.voice_transfer -->
The unseen speaker 5 (g = 1.03). Its first 8 test clips fix a reference embedding (the unit vector of the sum of their embeddings) and the estimate g_hat. For each of the next 16 clips: the recogniser writes its transcript, and the transcript is spoken again from the stream `project_speech_pipeline.voice.<n>` (n the clip's index among the speaker's clips, 8 .. 23), with the estimated g and with each trained speaker's g, so that only g differs between the voices. The table is the mean over those clips of the cosine between the reference and the embedding of the re-spoken clip, in the order of g.

| voice | g | mean cosine to the reference |
|---|---:|---:|
| trained speaker 0 | 0.88 | -0.519 |
| trained speaker 1 | 0.94 | -0.087 |
| trained speaker 2 | 1 | 0.169 |
| trained speaker 3 | 1.06 | 0.222 |
| trained speaker 4 | 1.12 | 0.165 |
| estimated | 1.053 | 0.211 |

Word error of the transcripts of the 16 clips against their text: 0.000. Word error of the recogniser on the re-spoken audio (estimated voice) against the transcript it was made from: 0.000.
<!-- /worked -->

Read the table as a curve in $\mathsf{g}$. The cosine to the reference rises from $-0.519$ at $\mathsf{g} = 0.88$ to a peak around the voices nearest the true $1.03$ and falls again at 1.12. The estimated voice, at $\hat{\mathsf{g}} = 1.053$, scores $0.211$: far above the voices at the low end, and within $0.011$ of the best of the voices that were heard, at 1.06, which scores a little higher (the estimate is 0.023 above the truth, and the map reads 0.016 high for this speaker even from single clips; the check is that the estimate lands on the peak, not that it beats its neighbours). Both word errors are zero: the transcripts of the reference clips are exact, and the recogniser reads the re-spoken audio back without a mistake, so the loop reproduces the sentence in a new voice and a new noise. **Milestone 6 ends** when your estimates for the six speakers agree with the table to $10^{-3}$ and your cosines agree to the digits shown.

### Milestone 7: three planted mistakes

A pipeline that works is not yet tested. The acceptance of the project requires that each of the mistakes below, planted alone in an otherwise correct pipeline, be caught by a number, and caught twice: once by a measurement that needs no recogniser, which is how a defect is found in practice, and once by the recogniser's error, which is how it is felt.

1. **An unfiltered resampler.** Replace the Kaiser design by linear interpolation in the audio that the trained recogniser reads (the test audio only). The direct measurement is the one the first section opened with, a tone of 5200 Hz reaching 2800 Hz with an amplitude of 0.604.
2. **A mel convention mismatch.** The recogniser was trained on Slaney triangles normalized by their width and a natural logarithm. Three ways to differ at test time, each of which would pass for the same function in a code review: the HTK scale for the Slaney one (the triangles stay linear in hertz and keep their width normalisation, so that only the scale of the edges changes); triangles of height 1 and not normalized by width; log10 for $\ln$. Each of the three changes one thing. The direct measurements are the largest move of a filter centre, the spread of the row sums of the bank, and the factor by which every value is scaled.
3. **A label shift.** Train with the label ids not offset past the blank: the space is `0`, the letters `1` to `4`, and the network still has six outputs with `0` the blank. The loss and the data are unchanged; only the integers that name the symbols have moved. The direct measurement is the loss itself, which should never be below zero, and the number of spaces written.

![Two panels of the log-mel frames of the test sentence adb adb, 16 bands (centre frequencies 138 to 3470 Hz) against 37 frames. In the left panel, the Kaiser sinc, the letters are dark patches in the low bands and, for the letter b, near 2265 Hz. The right panel, linear interpolation, has the same patches and in addition a dark mark in the top band near 3470 Hz during each letter d: the 5200 Hz component folded down.](img/alias-mel.svg)

The first mistake is visible before any recogniser reads it. In the picture the sentence `adb adb`, spoken by speaker 0 ($\mathsf{g} = 0.88$), is the same in both panels except for the top band, where the linear resampler has folded the 5200 Hz component of each `d` to $8000 - 5200 \times 0.88 = 3424$ Hz; the Kaiser design leaves that band at the noise floor.

<!-- worked: project_speech_pipeline.mistakes -->
Each mistake is planted alone in an otherwise correct pipeline (the first four at test time only, the recogniser being the one trained correctly; the last in training) and measured twice: by a number that needs no recogniser, and by the recogniser's error on the 180 test sentences. One thing changes at a time: the HTK-scale bank keeps the triangles linear in hertz and the area normalisation of the Slaney bank and moves only the scale of the edges, the bank without area normalisation keeps the Slaney edges, and log10 changes only the base of the logarithm.

| mistake | what it does, measured directly | word error | character error |
|---|---|---:|---:|
| resampling by linear interpolation | a 5200 Hz tone arrives at 2800 Hz with amplitude 0.604, not 1.9e-05 | 0.458 | 0.168 |
| the HTK scale for the Slaney one | the filter centres move by up to 164 Hz | 0.493 | 0.185 |
| no area normalisation of the triangles | the row sums span a factor 3.6, not 1.0 | 0.195 | 0.032 |
| log10 in place of ln | every value is multiplied by 0.4343 | 0.803 | 0.535 |
| label ids not offset past the blank | the mean training loss reaches -4.679 (a loss of -ln p is never below 0); spaces written: 0, where the references hold 276 | 1.000 | 0.196 |
<!-- /worked -->

Each row is a different kind of failure. The resampler's is physical: the 5200 Hz component of every `d` lands in the band below 4000 Hz, where the recogniser has never heard a `d` carry energy, and nearly half the words are lost. The mel conventions fail by distribution shift: the recogniser's weights are tuned to the statistics of the bands it was trained on, and a filterbank that moves the filter centres by up to 164 Hz changes which band a formant falls in. The unnormalized triangles scale the bands unevenly (the widest has 3.6 times the area of the narrowest), and the loss is the smallest of the three, 0.195 in words and 0.032 in characters. Log10 multiplies every value by 0.4343 before the standardization with the training statistics, so the input arrives with 0.43 of the spread the weights expect, and it costs 0.803 of the words. And the label shift is invisible to the loss, which keeps falling, to below zero, while the transcripts are one word long. A reader who looked only at the loss of that run would have shipped it.

There is a fourth thing in the table that is easy to miss: no mistake fails loudly. The error that a mismatched front end makes is not an exception and not a NaN. The pipeline produces, from an input of the right shape and a loss that looks reasonable, an output of the right type; the only signal is the error rate against an answer key, which this world has by construction and a deployment may not. This is the reason that the project's acceptance tests are numbers at each joint, and why a recording of speech with a transcript that no one has checked is not a test of anything.

**Milestone 7 ends** when each of the five rows is reproduced to the digits shown, and when your code makes each mistake possible by changing one line of it, which is a test that your tests can fail.

### Milestone 8: the bill

The last milestone counts what the pipeline costs, in operations (a multiply-add is 2) and in bytes of 32-bit values, from the shapes alone, for one second of audio. The recogniser's forward pass is $2(80 \cdot 24 + 24 \cdot 6) = 4{,}128$ operations a frame and 206,400 a second; one frame of the window is 256 multiplies and a radix-2 transform of $5 N \log_2 N = 10{,}240$ operations; the mel step is two operations for each of the 234 nonzero weights of the bank; the resampler is one multiply-add for each tap under each output, which for the 2/3 ratio is 96.3 taps on average, counted exactly.

<!-- worked: project_speech_pipeline.bill -->
Operations (a multiply-add is 2) and bytes (32-bit values) for one second of audio through each stage, counted from the shapes. The 'bytes in' and 'bytes out' columns are the arrays read and written.

| stage | operations per second | bytes in | bytes out | count |
|---|---:|---:|---:|---|
| resampler, 12 kHz to 8 kHz | 1,540,896 | 48,000 | 32,000 | 96.3 taps per output, each a multiply-add |
| window and transform | 524,800 | 32,000 | 25,800 | 256 multiplies and 10,240 for the radix-2 transform per frame |
| power and mel | 42,750 | 25,800 | 3,200 | 3 per bin, then 2 per nonzero weight (234 of 2064) |
| recogniser, forward | 206,400 | 3,200 | 1,200 | 8,376 bytes of weights read once per frame |
| quantizer, four stages | 153,600 | 3,200 | 100 | 3 per value per codeword, all stages |

The whole forward chain is 2,314,846 operations per second of audio, of which the resampler is 67%. One frame of the recogniser reads 8,376 bytes of weights for 4,128 operations: 0.49 operations per byte, the 2 operations per 4-byte weight of any vector-matrix product. The speech of the world is 9.7 characters a second: the 982 characters of the training transcripts (spaces included) over their 5,064 frames at 50 a second, 101.28 seconds.

Training: 20 epochs over 5,064 frames is 101,280 frame passes of 4,128 forward operations, 2(80 x 24 + 24 x 6), and 4,416 backward ones: the outer product for the gradient of W1 (80 x 24 multiply-adds), W2 transposed times the arriving gradient (24 x 6) and the outer product for the gradient of W2 (24 x 6), 2(1920 + 144 + 144) = 4,416 (the elementwise products and the sums over frames are left out), plus 600 AdamW steps of 16 operations on each of 2,094 parameters: 885,438,720 operations in all, before the lattice (about 8 S per frame for a label of S states, a few percent of the network's).
<!-- /worked -->

The surprise of the table is where the work is. The resampler, the one stage that no network touches and that Chapter 1 spent a section on, is two thirds of the forward chain: 1.54 million operations a second against 0.77 million for the transform, the mel step and the recogniser together. The recogniser itself is 206,400, about 9% of the chain, and it is the stage whose weights decide the transcript. The cost of the chain is set by the arithmetic of the signal path and the size of the filter, not by the model, and a shorter filter, with fewer zero crossings, cuts the work and widens the transition band: whether the project's letters tolerate that is the second extension below, and it is not an improvement until it is counted and tested.

The second observation is the one that the bill of [Volume VIII's image generator](../v08-image-generation/13-project-image-generator.md) made too. A frame of the recogniser reads 8,376 bytes of weights for 4,128 operations: 0.49 operations per byte, the 2 operations per 4-byte weight of any vector-matrix product with a fresh vector. At this size the weights sit in cache, so the bytes are cheap, but the ratio is the one that decides a real model's speed on real hardware: a model that reads each weight once per frame is limited by the memory, and a larger batch of frames, which reads each weight once for many frames, is the cure. The same arithmetic sets the rates down the chain: 48,000 bytes a second of 12 kHz audio at 32 bits, 3,200 bytes of features, 100 bytes of codes (800 bits), and 9.7 characters of text.

## Acceptance tests

The tests are the values the worked module prints, collected as a checklist. Test in the order of the milestones: a disagreement at milestone 4 that follows agreement at 1 to 3 points at training and not at the front end. "Exact" means equal integers and strings; a numeric tolerance is absolute, and "the digits shown" means the digits on which the run and its four nudged copies agree.

<!-- worked: project_speech_pipeline.acceptance_tests -->
Expected values of the project, computed by this module. 'Exact' means equal; a number is compared with the tolerance in the last column, absolute; 'the digits shown' means the digits on which the run and its four nudged copies agree, and no more can be asked of another implementation.

| milestone | quantity | value | tolerance |
|---:|---|---|---:|
| 1 | training sentence 0: text; samples at 12 kHz; sum; sum of squares | `ab adb`; 7920; 0.198549; 346.579756 | exact; exact; 1e-6; 1e-6 |
| 2 | tones 1000 and 3000 Hz through Kaiser sinc / linear interpolation | 1.0000, 1.0000 / 0.9830, 0.8536 | 0.0001 |
| 2 | tones 4400 and 5200 Hz, amplitude at 3600 and 2800 Hz: Kaiser sinc / linear | 8.7e-05, 1.9e-05 / 0.7034, 0.6040 | below 1e-3; 0.0001 |
| 2 | frames of sentence 0; sum of its 544 standardized values | 34; 200.399012 | exact; 1e-6 |
| 2 | first four entries of the stacked vector of frame 0 | 0.948845, -0.692724, -0.689101, -0.620758 | 1e-6 |
| 3 | parameters; first W1 value; softmax row 0 at the start | 2094; 0.094268; 0.142341, 0.200727, 0.056612, 0.131327, 0.254092, 0.214901 | exact; 1e-6; 1e-6 |
| 3 | CTC loss of frames 3 .. 16 with label `ab` at the start | 20.35490712 | 1e-8 |
| 3 | largest gradient difference against central differences | below 1e-6 | the criterion |
| 4 | mean training loss after epochs 1, 5 and 20 | 14.686, 0.385, 0.020 | the digits shown |
| 4 | word and character error: test set, training set | 0.004, 0.001; 0.000, 0.000 | the digits shown |
| 4 | word error on the unseen speaker 5 | 0.000 | the digits shown |
| 5 | word error after 1, 2, 3 and 4 quantizer stages | 0.110, 0.018, 0.004, 0.009 | the digits shown |
| 5 | bits per second after 4 stages | 800 | exact |
| 6 | equal error rate of the embedding; same / different cosine | 39.1%; 0.250 / -0.045 | 0.1; 0.001 |
| 6 | g estimated for the unseen speaker (true 1.03) from 8 clips | 1.053 | the digits shown |
| 6 | mean cosine to the reference: estimated voice, g = 0.88, 1.12 | 0.211, -0.519, 0.165 | the digits shown |
| 7 | word error: linear resampler; HTK scale; no area norm; log10 | 0.458; 0.493; 0.195; 0.803 | the digits shown |
| 7 | label shift: word error; character error; spaces written | 1.000; 0.196; 0 | the digits shown |
| 7 | operations per second of audio through the resampler | 1,540,896 | exact |
<!-- /worked -->

Beyond the table, an implementation passes when it also meets these, each of which is in the tables above: the five tones of the resampler table; the 16 rows of the filterbank; the gradient check against $10^{-6}$ at every one of the 2,094 parameters; a training loss that falls at every logged epoch of the clean run and goes below zero in the shifted one; the word error of every speaker; the quantizer's four stages; the estimated parameter of each speaker and the eight-clip estimate of the one never heard; and the five planted mistakes, each producing the number of its row. A pipeline that passes the clean rows and cannot be made to fail the planted ones has tests that cannot fail.

## Where each piece of the loop lives in brain

The project's loop crosses the crates below, and the correspondence is closer than the difference in scale suggests, because the interfaces are the same and the failures the project plants are failures that brain has had.

**The resampler and the front end.** `crates/audio` holds the pieces of milestone 2. `crates/audio/src/resample.rs` has `rational`, the Kaiser design of this project (commit `e532b63da`, which records that `resample_linear` has no anti-aliasing filter and so cannot reproduce a reference for a rate change such as 24 kHz to 16 kHz), and `sinc_hann`, the cos-squared design (commit `7d380d66d`). The upload path for speech recognition reaches the recogniser's 16 kHz through `to_speech_rate`, one function for every WAV surface, after commit `676a4f41d` found that a 10 kHz whistle at 48 kHz had been heard as 6 kHz: the first planted mistake of milestone 7, in a deployment. The front ends are `crates/audio/src/mel.rs` and `crates/audio/src/asr_frontend.rs`, the second in 64-bit floating point and parity-gated (commit `91a20b963`).

The unfiltered resampler was not only history when this chapter was written. `audio::resample_linear` was still what the speech synthesiser's round-trip gate used to bring a 24 kHz clip to the recogniser's 16 kHz (`crates/qwen3tts/tests/it/asr_roundtrip.rs`), and what `SpeakerEncoder::embed_wav` in `crates/ecapatdnn/src/model.rs` and the path that hands a reference clip to the codec's encoder in `crates/qwen3tts/src/pipeline.rs` used to reach 24 kHz. Where the source rate is above the target, as for a 48 kHz reference clip, that is the first planted mistake with a real input; where the energy above the new half-rate is small, the error is small, and a bound of 0.5 on the word error would not see it. Commit `49bca711f` moved all of them to `rational`. Both resamplers return their input unchanged when the rates are equal, which is why no parity gate, each of which feeds audio at the model's own rate, moved.

**The recogniser and its loss.** The project's head is a CTC head, and brain's recognisers are two other members of the family, which Chapter 4 told apart: `crates/nemotronasr`, a transducer in which each row emits a letter or the blank and whose trainable core (`crates/nemotronasr/src/train.rs`) is validated by `directional_check`, a full-parameter finite-difference check on tiny configurations, the project's milestone 3 at brain's scale; and `crates/qwen3asr`, whose decoder reads the audio rows as part of its prompt. What carries over is everything around the head: the blank id filtered out of the output before it is turned into text (as `crates/qwen3tts/tests/it/asr_roundtrip.rs` does with `blank_token_id`), the standardized log-mel frames, and the discipline of checking the loss against finite differences before training.

**The error rate.** `crates/eval/src/asr.rs` defines the corpus rate of milestone 4 exactly as the project does: `corpus_wer` is total edits over total reference words, "unlike a mean of per-utterance rates", and `None` when the references hold no words, since an unmeasured rate is not zero (commit `f8d8ec872`, which moved a private copy of the metric out of the round-trip gate so that every caller scores transcripts identically).

**The closed loop.** The project's loop is the shape of one of brain's own tests. Commit `fee25c3f0` (3 September 2026) added `crates/qwen3tts/tests/it/asr_roundtrip.rs` as a quality gate: synthesize a known sentence with the speech synthesiser, transcribe it with `nemotronasr`, and assert that the word error rate against the input is below 0.5. Its message says what it was for: no end-to-end audio-quality signal existed that did not need a reference from another framework, and on its first real run it found a near-silent output, 200 frames of it, that no per-tensor comparison had seen. A loop of the project's shape cannot be fooled by a stage that is individually plausible. Its bound of $0.5$ is loose on purpose, a smoke signal and not a claim about either model's accuracy, as its comment says; the project's acceptance tests are the tight version, possible because the world is small enough for the correct answer to be zero.

**The representation.** `crates/mimi` is the quantizer of milestone 5 at scale: its encoder's `rvq_encode` and its decode path sum codewords of 16 codebooks of 2,048 rows (Chapter 3), against the project's four of 16. The synthesiser that predicts those codes, `crates/qwen3tts`, has its large Talker predict codebook 0 of each frame and a small code predictor fill the other fifteen (Chapter 6), so the stages of the residual chain are not treated alike downstream; the project's quantizer is trained and read stage by stage in the same spirit.

**The voice and the synthesiser.** `crates/ecapatdnn` is the embedding of milestone 6: a 1,024-wide vector from attentive statistics pooling, where the project has the mean of 16 values, and `SpeakerEncoder::embed_wav` is the path of a clip to it. The cosine of two embeddings is the similarity check, and `crates/eval/src/tts.rs` has it as `speaker_similarity` alongside a mel-cepstral distortion and a log-mel distance, a three-part evaluation of a synthesised clip against its reference. The conditioning of the synthesiser is in `crates/qwen3tts` (the 1,024-value x-vector enters the Talker's prompt, `build_xvector_prompt` in `crates/qwen3tts/src/prompt.rs`) and `crates/cosyvoice`, whose reference path resamples the same clip three times for three consumers, the vector for its speaker encoder, the prompt tokens and the prompt mel (its header draws the diagram), with the Kaiser resampler. The project's synthesiser, two sinusoids scaled by a parameter, is what is left of those when a language model and a flow decoder are replaced by an equation, and its one-line conditioning is the shape of theirs: a voice is a vector computed from a reference clip and given to the part that makes sound.

**Where the project's choices differ from brain's.** The project trains and tests its recogniser on audio from the same synthesiser, where brain's recognisers are trained on recordings and tested on others; the world is therefore a closed one, where a model that is wrong is wrong because of the pipeline and not because of the data, and every number can be zero. The sampling rates (12 kHz, 8 kHz) are chosen so that the arithmetic of the resampler fits a page, where brain's are 24 kHz and 16 kHz with the same ratio, 2 to 3. And the speaker map is fitted on five voices and tested on one, where the encoder of `crates/ecapatdnn` is a network and not a regression.

## Check your understanding

1. The project's recogniser reads stacked frames $t-2 \ldots t+2$, and training sentence 0 has 34 frames. How many stacked vectors does it have, which frames does the stack of frame 0 hold, and what would change if the stack were not clamped at the ends?
2. A colleague reports that the label-shift run "trained fine, loss 22.948 in the first epoch, then down to -4.679, and the transcripts look right". Find the error in the reasoning, and say which two numbers of the milestone 7 table to look at.
3. After three stages of the quantizer the word error is 0.004 and after four 0.009. A reader concludes that the fourth codebook is harmful. Is that justified? What measurement would settle it?
4. The estimate of the speaker parameter from eight clips of the unseen speaker is 1.053 and the truth is 1.03. A reader proposes to correct the bias by subtracting 0.023 in the map. Why is that not a fix, and what would be?
5. The resampler is two thirds of the forward chain's operations. Someone proposes to halve the number of zero crossings of the Kaiser design to save a third of the chain. What does the table of tones suggest it will cost, and which row of milestone 7 should be run again to find out?
6. A test set made only of sentences spoken by speakers 1 to 3 gives a word error of 0.000. Which claim of the chapter would it fail to support, and which would it hide?

## Summary

If every acceptance test passes you have a complete speech loop on a world whose answers are known: a synthesiser that speaks a six-word language in a speaker's voice, a Kaiser resampler and a log-mel front end that bring it to a recogniser's rate and shape, a small recogniser trained by a loss that sums over alignments, with a backward pass that agrees with central differences to below $10^{-6}$ at every parameter, a round trip from text to audio to text that returns the words, a residual quantizer that carries the same transcript in 800 bits a second, a voice estimated from eight clips and a transcript spoken again in it, and the cost of all of it counted.

What the project established, in order. A front end is a set of conventions, and each of them is a place where an almost-right pipeline loses a fifth to four fifths of its words while every shape matches. A loss that goes below zero is a bug and a loss that falls smoothly is not evidence of a pipeline that works: the label shift trained as well as the correct run by the loss alone and returned one word per sentence. Error per frame and error per word are different measures, and a quantizer can keep falling in the first while the second has stopped. A voice is recoverable from audio by a map as crude as a mean and a regression, to about two parts in a hundred from eight clips, and the check that the loop works is a curve over voices and not a single comparison. And the cost of a second of speech is not where the model is: the resampler, a filter of 97 taps, is twice the work of everything after it.

Where this stops. The world has four letters and six words, and its answer key is the text that generated it; a real recording has neither. The recogniser is a perceptron over five frames, whose receptive field is 112 milliseconds (four hops of 20 and one window of 32), where the recognisers of Chapter 4 attend over seconds; the synthesiser is an equation, and the speaker's whole identity is one number. Nothing here trains on a second speaker's data or tests a language model's contribution, and the quantizer is not trained jointly with what reads it. The limits are the reasons for the extensions.

## References

- A. Graves, S. Fernandez, F. Gomez and J. Schmidhuber, *Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks*, 2006. The loss, its lattice and its gradient.
- N. Zeghidour, A. Luebs, A. Omran, J. Skoglund and M. Tagliasacchi, *SoundStream: an end-to-end neural audio codec*, 2021. The residual quantizer.
- B. Desplanques, J. Thienpondt and K. Demuynck, *ECAPA-TDNN: emphasized channel attention, propagation and aggregation in TDNN based speaker verification*, 2020. The pooled statistics and the cosine scoring.
- D. P. Kingma and J. Ba, *Adam: a method for stochastic optimization*, 2014; I. Loshchilov and F. Hutter, *Decoupled weight decay regularization*, 2017. The optimizer.
- J. F. Kaiser, *Nonrecursive digital filter design using the I0-sinh window function*, 1974. The taper of the resampler.

## Extensions

These are optional. Each changes one thing of the project and asks you to predict the outcome before you run it.

1. **The same mistake in training and at test time.** Train a second recogniser on audio resampled by linear interpolation and test it on the same. Predict whether its word error is nearer to 0.004 or to 0.458, and what that says about a pipeline that is consistently wrong.
2. **A transition band that is shorter.** Halve the zero crossings of the Kaiser design (16 in place of 32). Predict the amplitude of the 4400 Hz tone at 3600 Hz and the change of the bill, and test the word error of the speakers.
3. **A voice that is not one number.** Give each speaker a second parameter that scales only the second sinusoid, so that formants move separately. Predict how the one-parameter ridge map and the embedding's equal error rate change, and what a map with two outputs needs.
4. **Retrain on the quantizer's output.** Train the recogniser on the frames reconstructed from one stage and compare its word error with the 0.110 of the table. Predict whether a recogniser that has seen the quantizer's error does better, and by how much.
5. **A word the recogniser has never heard.** Add a seventh word that spells a new order of the same letters, and test a recogniser trained on the original six. Predict whether the character error or the word error moves more, and what that says about what a CTC head learns.
6. **Doubled letters.** Add the word `cc` and render it as two tones with a gap of one frame. Predict how many frames the loss needs at least (Chapter 4's minimum), and what a decoder that forgets the blank between repeated letters writes.

## Answers

1. A 5200 Hz cosine at one sample in 1.5 is a signal at 8000 Hz, whose half-rate is 4000 Hz; the tone is above it, so it folds to $8000 - 5200 = 2800$ Hz. A sinc with a cutoff at 4000 Hz and a Kaiser taper (80 dB of stopband attenuation) removes it before the samples are taken, where linear interpolation, whose response is only a mild low-pass, cannot, and the folded tone arrives at 0.604 of its amplitude. The cost of the filter is the 97 taps, about 96.3 multiply-adds per output on average.
2. $1 + \lfloor 5280 / 160 \rfloor = 1 + 33 = 34$ frames. A gain $a$ multiplies every energy by $a^2$, so $\ln(E + 10^{-6}) \approx \ln E + 2 \ln a$ for energies well above the guard: a shift of $2 \ln 3 \approx 2.2$ of every value of the clip equally. Subtracting the clip's largest value removes the same shift from every value, so frames of a louder clip are the same as those of a quieter one (except where the guard matters).
3. Stage 2 quantizes the residual that stage 1 leaves, $\mathbf{r}^{(1)} = \mathbf{x} - \mathbf{c}^{(1)}$, and the reconstruction is $\mathbf{c}^{(1)} + \mathbf{c}^{(2)}$, the sum of the chosen codewords. The squared error falls with each stage, by a smaller factor each time as the residuals become less structured.
4. For frame $t$ the gradient with respect to the logits is $\mathbf{p}_t - \boldsymbol{\gamma}_t$: the softmax row minus the soft alignment (the share of all label-compatible paths that have symbol $k$ in frame $t$). A path of frame labels becomes a transcript by collapse: merge neighbouring equal symbols, then delete the blanks.
5. It receives a speaker vector (an x-vector or an embedding) computed by an encoder from a reference clip, pooled over time; the clip is run through a front end and a network that turns the frames into one vector, and the synthesiser's text prefix or flow decoder is conditioned on it. The project's vector is the unit mean of the standardized frames, through a ridge map to the one parameter $\mathsf{g}$ that the synthesiser takes.
6. The analytic gradient is compared with the central difference $(L(\theta + h) - L(\theta - h)) / 2h$ for each parameter in turn, with $h = 10^{-5}$ here, and the backward pass is accepted if the largest difference is below $10^{-6}$.

Check your understanding.

1. One stacked vector for every frame: 34. With clamping, the stack of frame 0 holds frames 0, 0, 0, 1, 2, so frame 0 supplies 48 of its 80 values. Without the clamp the stack would have no value for $t-2$ and $t-1$ at the first frame and would need a padding rule: zeros are not what the front end would see in a longer clip, and the first frames of every sentence, where the leading silence is, would differ from the rest.
2. The loss of a CTC run that has a label equal to the blank is not a probability: it goes below zero because the lattice counts more than all the paths, and a smoothly falling loss is not evidence that the run is right. The two numbers to look at are the loss itself, which should never be below 0 (the run reaches $-4.679$), and the number of spaces written, which is 0 where the references hold 276. The transcripts do not look right: they are one word per sentence, so the word error is 1.000, and the characters hide that, with a character error of 0.196.
3. Not justified. The word error is a count of 456 words, 0.004 is 2 of them and 0.009 is 4, and a difference of two words is inside what a small shift in a codeword can do to boundary frames; meanwhile the squared error of the frames falls from 1.39 to 1.03. Settling it needs more test sentences (a corpus that gives a standard error below the difference) or the character error and the posteriors of the boundary frames, not the point estimate.
4. A bias of 0.023 is measured on one speaker and eight clips: it may be a property of that speaker's letters (the embedding depends on which letters were said), and the eight-clip estimates of the heard speakers are off by $+0.007$, $-0.007$, $-0.005$, $+0.003$ and $-0.021$, of different signs. Subtracting a constant would move most of the heard speakers' estimates off their truth. The fix is more speakers in the fit, so that the map interpolates better, or a richer embedding (the deviations as well as the means), and a test set of several unseen speakers.
5. The tone table shows the passband flat to four decimals at 1000 and 3000 Hz and the stopband at $9 \times 10^{-5}$ or below at 4400 Hz: a design with half the zero crossings has a wider transition band and a shallower stopband, so the 4400 Hz tone would come through larger than that, perhaps by orders of magnitude (a prediction to test); the first row of milestone 7 should be run again, with the amplitude of the 5200 Hz tone at 2800 Hz and the word error of the recogniser as the numbers.
6. It would not support the claim that the pipeline works for a voice it was not trained on, since all three speakers were heard, and it would hide the edge of the training set: the only errors of the test set belong to the highest voice, speaker 4 at 1.12, which the test set would not contain.
