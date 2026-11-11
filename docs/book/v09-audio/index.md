<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Volume IX: Audio, Speech and Music

A second of speech is sixteen thousand numbers, and none of them means anything alone. An image gives a model its content in neighbouring pixels; a waveform gives it a pressure that rises and falls so fast that a single word spreads over tens of thousands of samples, while the thing a listener hears, a vowel or a pitch or a speaker, lives in the pattern of how the pressure repeats over a few hundredths of a second. Nothing in a raw sample says "this is an a". The first job of every model in this volume is therefore to change what a signal looks like: from a line of numbers to a picture of which frequencies are present at each moment, or to a short sequence of discrete codes, or to a vector that stands for a voice. Once that is done the machinery of the earlier volumes applies almost unchanged, and the volume is largely the story of how each of those machines was pointed at sound.

## Why it comes here

The volume leans on nearly everything before it. Volume I supplied the Fourier transform that turns a window of samples into a spectrum, and Volume IV the integer formats in which samples are stored and the arithmetic cost that decides how much audio a model can afford to read. Volume V and Volume VI supplied the recurrent networks, the attention and the language models that read a token stream, Volume VII the convolutions that treat a spectrogram as an image and the faces chapter's idea of an identity embedding, and Volume VIII the vector quantisation, the diffusion and flow-matching generators and the conditioning that turn a description into a signal. Audio is where those parts are asked to work on a one-dimensional signal with a hard physical constraint, the rate at which it is sampled, and with a human listener who notices errors in places an image viewer would not.

The volume therefore starts with the signal itself, what sampling keeps and what it destroys, because every later chapter inherits a sampling rate and the mistakes that come with a wrong one. It then follows the order in which the representations were found: the spectrogram and the mel scale that give an ear-like picture of a sound, the neural codec that turns the waveform into discrete tokens a language model can predict, recognition that maps speech to text without knowing where each word begins, the embedding that captures who is speaking, synthesis that speaks a sentence in a chosen voice, music generation at the scale of minutes, and the models that listen, look and speak at once. It ends with a project that carries a waveform to a learned representation and back to generated audio.

## What you can do after it

You can say exactly which frequencies a sampled signal can carry, resample it between the rates models expect without folding the rest into the band you keep, and read a recording's format, rate and channels without trusting its header. You can turn a waveform into a log-mel spectrogram and say what each of its parameters costs, explain how a codec's residual quantisers produce audio tokens and what a token stream of a given rate can and cannot carry, and describe the three ways a model aligns speech with text when it does not know where the words are. You can describe how a voice is reduced to one vector and how that vector steers a synthesiser, what a codec language model and a flow-matching decoder each contribute to speech, why music needs a different treatment of length, and how one model takes in sound, pictures and text and answers in speech. You can read the cost of a second of audio as samples, frames and tokens before running anything.

## Chapters

1. [Sound and Sampling](01-sound.md): how is a sound wave turned into numbers without losing what we hear?
2. [Spectrograms and the Mel Scale](02-spectrograms.md): how do we see which frequencies are present at each moment, the way an ear does?
3. [Neural Audio Codecs](03-audio-codecs.md): how can audio become a stream of discrete tokens a language model can predict?
4. [Speech Recognition](04-speech-recognition.md): how does a model turn speech into text when it does not know where each word starts?
5. [Who Is Speaking](05-speaker-embeddings.md): how does a model capture the identity of a voice in a single vector?
6. [Speech Synthesis](06-speech-synthesis.md): how does a model speak a sentence in a given voice?
7. [Music Generation](07-music.md): what changes when the target is minutes of structured music rather than a sentence of speech?
8. Models That Listen, See and Speak: how is one model built to take in audio, images and text and answer in speech?
9. Project: From Waveform to Speech and Back: can you build a pipeline from a waveform to a learned representation and back to generated audio?
