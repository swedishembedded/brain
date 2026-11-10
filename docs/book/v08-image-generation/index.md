<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Volume VIII: Generating Images

A model that classifies an image answers a question with a handful of possible answers. A model that generates one answers a question with more possible answers than there are atoms to write them on: a $512 \times 512$ colour image has $256^{786432}$ values, and almost none of them is a picture of anything. Making one of the few that are, on request, and different each time, asks for something that no earlier volume built: a model of what images are like, in the sense of a probability for every possible image, and a way to draw from it. The first half of this volume is the history of how that was achieved, from a table of 16 numbers to a network that removes noise; the second half is the machinery that turns a sentence, a pose or a photograph of a face into the signal that steers it.

## Why it comes here

The volume stands on the ones before it. Volume I and Volume II supplied the probability: distributions, likelihood, entropy and the divergences, and the generator that draws from them. Volume III supplied the networks and their training, Volume VII the convolutional and transformer networks that read an image and the preprocessing that gives them their input, and Volume V and Volume VI the transformer blocks and the text encoders that let a sentence condition a model. Every generator in this volume is those ingredients in a different arrangement: an encoder and decoder trained under a bound on likelihood, a codebook that turns pixels into tokens, a pair of networks that train each other, a denoiser that is applied many times, and the sampler that decides how many.

The volume therefore starts with the question every one of those arrangements answers, what it means to learn a distribution over images and to sample from it, and works through the families in the order in which each solved a problem the previous one left: the autoencoder made probabilistic, the discrete codes that language models can predict, the adversarial game that dispenses with a likelihood, then diffusion and flow matching, the samplers and guidance that make them fast and obedient, the denoiser architectures, the conditioning on text, the systems that join the parts, the adapters that add control and identity, and restoration. It ends with a project that trains a small generator and grows it into a conditioned one.

## What you can do after it

You can say what a generative model of images offers (a sampler, a density, or both) and what it gives up, and you can score one on images it has not seen. You can derive the training objective of a variational autoencoder, of a diffusion model and of a flow-matching model from the likelihood they approximate, and write the sampler that turns a trained denoiser into an image, with the number of steps and the guidance scale as choices you can reason about. You can describe what a U-Net and a diffusion transformer take in and why, how a prompt becomes the conditioning signal, how an encoder and a decoder move a model from pixels to a latent space where it is cheap to work, and how a pose, an edge map or a face is fed in without retraining the generator. You can read the cost of generating an image as work, in operations and bytes, before running it.

## Chapters

1. [What It Means to Generate](01-generative-modeling.md): what does it mean for a model to learn a distribution over images, and how can we sample from it?
2. [Variational Autoencoders](02-vae.md): how an autoencoder can be made into a generator, and why that needs a bound on likelihood.
3. [Discrete Latents and Vector Quantization](03-vector-quantization.md): how an image can be turned into a short sequence of discrete codes.
4. Adversarial Training: what happens when the loss itself is a network trained to catch the generator.
5. Diffusion Models: how a model can learn to create an image by learning to remove noise.
6. Flow Matching and Rectified Flow: whether generation can be a straight path from noise to data, followed by an ordinary differential equation.
7. Samplers and Guidance: how many steps sampling needs, and how to make a model follow its prompt more closely.
8. Denoiser Architectures: U-Nets and Diffusion Transformers: what network shapes are good at predicting noise, and how they take in the prompt.
9. Conditioning on Text: how a sentence becomes the signal that steers image generation.
10. Latent Diffusion Systems End to End: how the encoder, denoiser, sampler and decoder combine into a working image generator.
11. Control, Identity and Adapters: how to steer a generator with a pose, an edge map or a particular person's face.
12. Restoration and Super-Resolution: how a network can recover detail that is not in the input.
13. Project: A Small Image Generator: whether you can train a small generator and grow it into a conditioned latent generation system.
