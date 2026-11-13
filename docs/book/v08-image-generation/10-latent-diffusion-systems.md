<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Latent Diffusion Systems End to End

*How do the encoder, denoiser, sampler and decoder combine into a working image generator?*

A run of FLUX.2 chose the empty card for its denoiser, completed all fourteen denoising steps, and died in the very last stage, while it was decoding, on a card with 22.9 GiB free. No bug in the arithmetic was involved. The plan had reserved a flat 2 GiB for the decoder, and when the engineers measured the decode of a 1024 by 768 picture on an idle card it took 8,384 MiB: the estimate was short by a factor of 4.1 (commit 7da4f7188). The same shape of failure had been paid for once before, with SDXL, whose first version ran out of memory at the decode after all 24 steps had run. In both stories every step that mattered had completed. The stage that nobody had thought to budget, because it runs once and looks small next to fifty passes of a large network, was the one that took the image away.

The previous eight chapters built the parts one at a time: the autoencoder that makes a latent space, the schedules and solvers that walk through it, the denoisers that predict a velocity, and the encoders that turn a sentence into rows. This chapter puts them in one process and asks what the assembly is. It is not a single network. It is a plan, with a ledger of operations (how much arithmetic each stage does) and a ledger of bytes (what each stage needs on a device, and when). The questions come in the order in which a request meets the plan. Where does the work of one image go, stage by stage? What do the stages have to agree on in order to hand each other arrays? What happens to a request that carries a picture, or a picture and a mask, in place of nothing? And, last, who is on the card at each moment, and what has to be moved to make that true?

A latent diffusion system joins six earlier ideas.

1. (Volume VIII, [Chapter 2](02-vae.md), the variational autoencoder.) A $1024 \times 1024$ picture goes through an autoencoder with 32 latent channels at one eighth of the side, and the latent is then packed in $2 \times 2$ patches. How many numbers does the latent hold, and how many tokens of how many numbers does the denoiser see?
2. (Volume VIII, [Chapter 6](06-flow-matching.md), flow matching.) A distilled sampler takes four Euler steps with noise fractions $1, 0.9674, 0.9081, 0.7672, 0$. What is the size of the last step, and what does it do to the latent?
3. (Volume VIII, [Chapter 7](07-samplers-and-guidance.md), classifier-free guidance.) Write the guided field from the field with the prompt and the field without it, and say how many passes of the denoiser one step needs for guidance scales $g = 1$ and $g = 3$.
4. (Volume VIII, [Chapter 8](08-denoiser-architectures.md), joint attention.) With 512 text rows and a $1024 \times 1024$ image at 16 pixels a token, how long is the joint sequence, and which rows does the final layer predict?
5. (Volume VIII, [Chapter 9](09-text-conditioning.md), text-encoder conditioning.) What does brain's FLUX.2 pipeline do with a prompt that is longer than the text window, and why?
6. (Volume IV, Chapter 3, the roofline.) A device does $P$ operations a second and moves $B$ bytes a second. At what operations per byte does a kernel stop being limited by traffic and start being limited by arithmetic?

## Four stages, and where the work goes

Every pipeline of this volume does the same four things in the same order, and the order is the whole idea. A text encoder turns the prompt into rows once. An autoencoder's encoder turns a reference picture, if there is one, into a latent once. A denoiser runs many times on the latent, with a sampler stepping it from noise toward an image, and the autoencoder's decoder turns the result into pixels once. The arrangement is called **latent diffusion**: the many-times network works in the compressed space of an autoencoder that was trained earlier and is frozen at generation time, so that the expensive full-resolution layers run once at the end and not at every step. The word covers the whole family of this volume, U-Net or transformer, noise-prediction or flow-matching, because what it fixes is where the iteration happens, not what is iterated.

The size of that choice can be counted. The decoder of the 32-channel autoencoder spends most of its operations in its last three levels, at $256 \times 256$ and above, and the last two alone (the levels at $512 \times 512$ and $1024 \times 1024$) are $3.470 + 2.233 = 5.703$ TFLOP in the table below. Fifty passes of just those two levels would be $285$ TFLOP, before a single attention layer, for a network that has not yet looked at the prompt. Running the iteration on the latent, where a pass is a transformer over 4,096 tokens, and the decoder once, is what makes the arithmetic affordable. The whole sequence of stages, with the contracts between them (shapes, scales, positions, schedule, who owns which weights), is an **image generation pipeline**; the rest of the chapter is about reading one.

<!-- worked: latent_diffusion_systems.vae_work -->
The two autoencoders of brain from their channel schedule (128, 256, 512, 512), two residual blocks per level in the encoder and three in the decoder, a mid block with one attention layer, and a multiply-add counted as two operations; norms, activations and the nearest-neighbour doubling are left out. The FLUX.1 autoencoder has 16 latent channels and no 1 x 1 convolutions on the latent; FLUX.2's has 32 and both. The last column is the share of the decode spent in the mid-block attention over the (side/8)^2 positions of the latent.

| autoencoder | direction and image | parameters | TFLOP | mid-block attention, share (%) |
|---|---|---:|---:|---:|
| FLUX.1 (16 channels) | decode 512 x 512 | 49,545,475 | 2.515 | 1.7 |
| FLUX.1 (16 channels) | decode 1024 x 1024 | 49,545,475 | 10.472 | 5.6 |
| FLUX.1 (16 channels) | encode 512 x 512 | 34,274,208 | 1.118 |  |
| FLUX.1 (16 channels) | encode 1024 x 1024 | 34,274,208 | 4.883 |  |
| FLUX.2 (32 channels) | decode 512 x 512 | 49,620,259 | 2.516 | 1.7 |
| FLUX.2 (32 channels) | decode 1024 x 1024 | 49,620,259 | 10.475 | 5.6 |
| FLUX.2 (32 channels) | decode 2048 x 2048 | 49,620,259 | 48.496 | 18.4 |
| FLUX.2 (32 channels) | encode 512 x 512 | 34,425,856 | 1.119 |  |
| FLUX.2 (32 channels) | encode 1024 x 1024 | 34,425,856 | 4.888 |  |

Where the 1024 x 1024 decode of FLUX.2's autoencoder spends its operations (up block i runs three resnets at the size shown; blocks 0 to 2 end with a doubling and a 3 x 3 convolution at the next size, counted in the block that ends with them; the rest of the total is the convolution in, the mid block and the convolution out).

| part of the 1024 x 1024 decode | channels | resnets at | TFLOP | share of the decode (%) |
|---|---:|---:|---:|---:|
| up block 0 | 512 | 128 x 128 | 0.773 | 7.4 |
| up block 1 | 512 | 256 x 256 | 3.092 | 29.5 |
| up block 2 | 256 | 512 x 512 | 3.470 | 33.1 |
| up block 3 | 128 | 1024 x 1024 | 2.233 | 21.3 |
<!-- /worked -->

![The stages of one FLUX.2 klein-4b image at 1024 by 1024: prompt to text encoder to 512 rows, reference image to the VAE encoder, the denoising loop with its start, its mask blend and its four DiT passes, tokens to the decoder and the image, each stage with its work in TFLOP and its bytes on the device.](img/latent-pipeline.svg)

The autoencoder's side of the ledger has two features that the stage-by-stage view hides. A decode of $2048 \times 2048$ costs $48.5$ TFLOP, $4.63$ times the cost at $1024 \times 1024$ and not $4$ times, because the mid-block attention is quadratic in the number of latent positions (it is 5.6 percent of the smaller decode and 18.4 percent of the larger). And the encoder, which only runs when a reference picture arrives, costs $4.89$ TFLOP at $1024 \times 1024$, about 47 percent of a decode: it has two residual blocks a level where the decoder has three.

Now put the stages side by side for one picture at each pipeline's default step count.

<!-- worked: latent_diffusion_systems.stage_work -->
The work of one 1024 x 1024 image, stage by stage, at each pipeline's default pass count: the text encoders (the table of Chapter 9), all denoising passes (Chapter 8's pass counts; the base model runs two passes a step for guidance) and the decode of the autoencoder of this pipeline (`latent_diffusion_systems.vae_work`). TFLOP is 10^12 operations, a multiply-add counted as two.

| pipeline | text, TFLOP | denoise, TFLOP | decode, TFLOP | total, TFLOP | text (%) | denoise (%) | decode (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FLUX.1 dev, 50 steps | 4.86 | 3718.9 | 10.47 | 3734.2 | 0.13 | 99.59 | 0.28 |
| FLUX.2 klein-4b, 4 steps | 2.91 | 139.3 | 10.47 | 152.7 | 1.90 | 91.23 | 6.86 |
| FLUX.2 klein-9b, 4 steps | 5.45 | 302.1 | 10.47 | 318.0 | 1.71 | 94.99 | 3.29 |
| FLUX.2 base-4b, 50 steps with guidance | 2.91 | 3482.0 | 10.47 | 3495.4 | 0.08 | 99.62 | 0.30 |
<!-- /worked -->

The denoiser is 91 percent to 99.6 percent of the operations of every pipeline, and the share of everything else is a statement about steps. FLUX.1 dev and the undistilled FLUX.2 take 50 steps (the second with two passes a step), and for them the text encoders and the decoder together are less than half a percent of the work. FLUX.2 klein takes four, and the same decoder is 6.86 percent of the work of the 4-billion model and 3.29 percent of the 9-billion one. This is a consequence of distillation that is easy to miss: when a method removes most of the passes it does not touch the stages that were run once, and so the ones that were rounding errors become a visible share. Whoever optimizes a four-step pipeline has to look at the decoder; whoever optimizes a fifty-step one can ignore it and should not look anywhere but at the denoiser.

## What one stage hands the next

Chapters 2, 8 and 9 each described one boundary. Seen together they form a list of facts that a pipeline has to get right, and that no stage can check for the others, because every one of them produces arrays of the right shape when it is wrong.

The first boundary is the scale of the latent. FLUX.1 uses the autoencoder's own scalar pair, so that the denoiser sees $(z - c_{\mathrm{shift}})\,c_{\mathrm{scale}}$ and the decoder is handed $z_{\mathrm{den}}/c_{\mathrm{scale}} + c_{\mathrm{shift}}$; FLUX.2 uses a $2 \times 2$ pixel-unshuffle and a per-channel standardization with the statistics stored in the checkpoint. The module documentation of `crates/flux1/src/pipeline.rs` says that reusing FLUX.2's packing for FLUX.1 "would silently apply the wrong normalization", and what it implements is the inverse map in one place, `unpack_tokens`: the latent of token $(y, x)$ in channel $4c + 2p_i + p_j$ goes to pixel $(2\,y + p_i,\; 2\,x + p_j)$ of latent channel $c$, divided by the scale and shifted. The packed channel order is the same as FLUX.2's; only the normalization differs. A pipeline that borrowed the wrong one would make a decoder-ready latent with the right shape and a wrong mean and spread.

The second boundary is the order of the rows. In FLUX.1's joint sequence the text rows come first, with position ids $(0, 0, 0)$; the generated image follows with $(0, h, w)$; a Kontext reference, in the model's position-id builder, carries $(1, h, w)$ for every reference. FLUX.2 has four axes, and reference $i$ takes $10(i + 1)$ on the first one. The prediction is read from the first rows of the image part, and `crates/flux2/src/refcond.rs` says why the order is not a detail: a layout that put references before the generated image would predict a velocity for the photograph. That file is also the one place where the layout is written. Before it existed, generation and training each spelled out their own token layout, and an adapter trained on a reference workflow was fitted against an input that the deployed path never presented (commit c6ce6355b); today both call the same type.

The third is the schedule. FLUX.1 steps a linear ramp, shifted by an exponential map whose strength $\mu$ is a straight line in the number of image tokens, from $0.5$ at 256 tokens to $1.15$ at 4,096; schnell is not shifted at all. FLUX.2's $\mu$ is the empirical fit of Chapter 6 and depends on the step count too. Both loops step the list directly, `dt = sigmas[i + 1] - sigmas[i]`, and neither passes it through the scheduler class, which would append a second terminal zero.

<!-- worked: latent_diffusion_systems.schedule_table -->
Noise fractions at 4,096 image tokens (a 1024 x 1024 canvas). FLUX.1: linspace(1, 0, steps + 1), shifted by e^mu / (e^mu + 1/sigma - 1) with mu the line through (256, 0.5) and (4096, 1.15), so mu = 1.15 here; schnell is not shifted. FLUX.2: `flow_matching.klein_sigmas`. An image-to-image run with strength s takes the FLUX.2 schedule times s.

| schedule at 4,096 tokens | sigma 0 | sigma 1 | sigma 2 | sigma 3 | sigma 4 |
|---|---:|---:|---:|---:|---:|
| FLUX.1 schnell, 4 steps (no shift) | 1.0000 | 0.7500 | 0.5000 | 0.2500 | 0.0000 |
| FLUX.1 dev, 4 steps (mu = 1.15) | 1.0000 | 0.9045 | 0.7595 | 0.5128 | 0.0000 |
| FLUX.2 klein, 4 steps (mu = 2.2912) | 1.0000 | 0.9674 | 0.9081 | 0.7672 | 0.0000 |
| klein, strength 0.6 | 0.6000 | 0.5804 | 0.5449 | 0.4603 | 0.0000 |
| klein, strength 0.3 | 0.3000 | 0.2902 | 0.2724 | 0.2302 | 0.0000 |

The 51 values of FLUX.1 dev's default schedule, at the steps listed.

| step i | 0 | 1 | 5 | 10 | 20 | 30 | 40 | 45 | 49 | 50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLUX.1 dev, 50 steps: sigma | 1.0000 | 0.9936 | 0.9660 | 0.9266 | 0.8257 | 0.6780 | 0.4412 | 0.2598 | 0.0606 | 0.0000 |
<!-- /worked -->

At 4,096 tokens FLUX.1's line gives $\mu = 1.15$; its 50-step schedule is still at $\sigma = 0.6780$ at step 30 and at $0.0606$ at step 49, and a 4-step dev schedule would be $1, 0.9045, 0.7595, 0.5128$. FLUX.2's four steps are $1, 0.9674, 0.9081, 0.7672$, all in the upper quarter of the range for the reason Chapter 6 gave. These are not interchangeable: the distilled checkpoint was trained to be evaluated at those four values and nowhere else.

The fourth is the end of the pipeline. FLUX.1 decodes, maps $[-1, 1]$ to $[0, 1]$ and clamps; FLUX.2 clamps to $[-1, 1]$ first, as its reference does, and then writes eight-bit values as $127.5 (v + 1)$ truncated, so that the image leaves the pipeline as bytes.

The fifth and last is the randomness, and it is worth stating exactly because a "seed" means something different in the two loops. The only random object of a generation is the starting noise (the sampler is Euler, with no fresh noise at any step), so a seed fixes the whole picture. But FLUX.2 draws that noise with `randn` (xorshift64\* in 64-bit floats and a Box-Muller transform that keeps both halves of each pair) and FLUX.1 with `gaussian` (SplitMix64 in 32-bit floats), and the documentation of `crates/model/src/hostmath.rs` says in as many words that the constructions are deliberately different, because unifying them would change what an existing seed reproduces. Neither is the stream of a framework that other tools use, so a picture made here is reproducible here and statistically equivalent elsewhere, never bit-identical. The option defaults of both now draw a fresh seed when none is given, and several of the command-line tools print the seed they drew so that a run can be repeated (commit 7e62ce809); before that, a library caller that took the defaults generated the same picture every time and never knew a seed was involved.

## One loop, and three ways of being guided

The loop itself is short, and the three pipelines differ in it only in how the prompt's strength reaches the denoiser. FLUX.1 dev and Kontext fold a guidance scalar into the conditioning vector: `guidance_in` embeds the scalar, in the way the clock is embedded (the scalar times 1000 through the sinusoid), and adds it to the embedded timestep and the pooled CLIP vector, so one pass a step gives a guided prediction whatever the scale, and schnell, which has no `guidance_in`, ignores it. The default scale is 3.5, the default step counts 50 for dev and 4 for schnell. A second, optional mechanism exists on top: a true guidance branch, which runs one further forward on a negative prompt's own conditioning from a chosen step onward and combines $\mathbf{u}_{\mathrm{neg}} + g\,(\mathbf{u}_{\mathrm{pos}} - \mathbf{u}_{\mathrm{neg}})$, at the price of doubling those steps. FLUX.2 klein has the guidance and the steps distilled into the weights: its step count is fixed at 4 (a requested count is ignored unless an experimental flag opts in, because the checkpoint ships as a fixed-parameter sampler), and it takes no guidance. The undistilled base variants run Chapter 7's guidance, with the unconditional pass as a second slot of the same batched forward when the scale exceeds 1, and FLUX.2 refuses a step count above 150 before the loop starts. Counted in passes of the denoiser an image costs 50 (dev), 4 (schnell and klein), and 100 (the base models), which are the pass counts of Chapter 8's table.

All three loops are one routine, with the optional parts switched by arguments. The routine takes the prompt's rows, the guidance scale, the step count, a strength, a source latent and a mask of one weight per token, and a noise draw. A text-to-image request has strength $1$ and a mask of ones, for which the source is never read.

```text
function generate(rows: array<real>[R, D], g: real, steps: int, strength: real, source: array<real>[n, c], mask: array<real>[n], eps: array<real>[n, c]) -> array<real>[n, c]:
    require steps >= 1, "at least one step"
    require 0 <= strength and strength <= 1, "the strength is a noise fraction"
    if strength == 0:
        return copy(source)
    sigmas = [strength * s for s in schedule(steps, n)]
    x = zeros(real, [n, c])
    for t in 0 .. n:
        for k in 0 .. c:
            x[t, k] = (1 - sigmas[0]) * source[t, k] + sigmas[0] * eps[t, k]
    for i in 0 .. steps:
        u = velocity(x, sigmas[i], rows, g)
        for t in 0 .. n:
            for k in 0 .. c:
                x[t, k] = x[t, k] + (sigmas[i + 1] - sigmas[i]) * u[t, k]
                if mask[t] < 1:
                    r = (1 - sigmas[i + 1]) * source[t, k] + sigmas[i + 1] * eps[t, k]
                    x[t, k] = mask[t] * x[t, k] + (1 - mask[t]) * r
    return x
```

`schedule(steps, n)` is the list of `steps + 1` noise fractions of Chapter 6 for `n` image tokens, ending in 0, and `velocity` is the guided field: the denoiser of Chapter 8 on the prompt's rows and, when `g` is not 1, on the empty prompt's rows, combined as in Chapter 7. The stages around the routine are the ones the earlier chapters built: the prompt encoder gives `rows`, the autoencoder's encoder and the packing give `source`, and the unpacking and the decoder turn the result into pixels. Two lines carry the rest of the chapter. The first line of the body that is not a `require` makes strength $0$ an exact copy, with no evaluation of the denoiser. The `if mask[t] < 1` block is the masked blend of a later section, which is a no-op for a mask of ones.

## A pipeline small enough to check

All of this can be exercised on a pipeline small enough to write out in full. The toy has every stage of the real one, in a size that can be checked by hand, and nothing in it is trained: its denoiser is the exact field of the data it was built for, as in Chapters 6 to 8. The point is not the picture but the interfaces, because the way to find the stage where a pipeline goes wrong is to have a number that every stage must satisfy.

<!-- worked: latent_diffusion_systems.toy_setup -->
The toy pipeline. Images: 8 x 8 greys in [-1, 1], in which the two rows of every pair are equal for the three prototypes (# is +1, . is -1, each line standing for a row pair):

| prototype | rows 0-1 | rows 2-3 | rows 4-5 | rows 6-7 |
|---|---|---|---|---|
| bar | `...##...` | `...##...` | `...##...` | `...##...` |
| box | `........` | `..####..` | `..####..` | `........` |
| slash | `.##.....` | `...##...` | `.....##.` | `.......#` |

Autoencoder: for each 2 x 2 patch with pixels p00, p01 (top row) and p10, p11 the code is two channels, (p00 + p01 + p10 + p11)/4 and (p00 + p10 - p01 - p11)/4; the latent is 2 x 4 x 4. Decoder: the left pixels of a patch are a + d and the right a - d. The latent is packed by the 2 x 2 pixel-unshuffle of `vae.latent_packing` into 8 channels of 2 x 2 tokens (packed channel 4c + 2 pi + pj), and each packed channel is standardized by its own statistics, the mean and the variance (divisor 12) over the 4 tokens of the three prototypes, with epsilon 1e-4. The 32 values of a packed latent are ordered packed channel slowest and then token in raster order, so value k = 4 (packed channel) + 2 (token row) + (token column):

| packed channel | mean | variance | sqrt(variance + 1e-4) |
|---|---:|---:|---:|
| 0 | -0.5000 | 0.4167 | 0.6456 |
| 1 | -0.5000 | 0.4167 | 0.6456 |
| 2 | -0.5833 | 0.4097 | 0.6402 |
| 3 | -0.5000 | 0.4167 | 0.6456 |
| 4 | 0.0000 | 0.3333 | 0.5774 |
| 5 | 0.0000 | 0.3333 | 0.5774 |
| 6 | 0.2500 | 0.1875 | 0.4331 |
| 7 | -0.3333 | 0.2222 | 0.4715 |

Data: x0 given the class c is the standardized packed latent of prototype c plus noise of standard deviation 0.2 in each of the 32 values; the three classes are equally likely. Distances between the class means (32 values):

| distance between class means | bar | box | slash |
|---|---:|---:|---:|
| bar | 0.0000 | 7.1365 | 8.1946 |
| box | 7.1365 | 0.0000 | 7.2634 |
| slash | 8.1946 | 7.2634 | 0.0000 |

Text: the table of 6 tokens by 4 numbers from `latent_diffusion_systems.text` (seed 0x941ea2b1f6255b29), `normal()` in row-major order:

| token | e0 | e1 | e2 | e3 |
|---|---:|---:|---:|---:|
| `<pad>` | -1.3405 | -0.3714 | -0.7403 | -0.4579 |
| `<s>` | -0.4426 | 0.6456 | -1.3873 | -1.0972 |
| `</s>` | 0.8684 | -0.2182 | -0.6384 | 0.9573 |
| `bar` | -0.2398 | 0.7826 | 0.4078 | 0.1892 |
| `box` | 0.4978 | -0.9749 | -0.2334 | -1.1195 |
| `slash` | 1.0752 | -1.0486 | -0.1958 | -0.1721 |

A prompt is [`<s>`] + words + [`</s>`] padded with `<pad>` to 6 rows; the pooled vector is the mean of the rows of the words; the class weights are the softmax over the classes of -|pooled - row of the class word|^2/(2 x 0.25^2); the empty prompt has weights 1/3 (what a model trained with the prompt dropped has learnt).
<!-- /worked -->

The text stage is a table of rows and a rule that turns the pooled vector into weights on the three classes. It stands in for an encoder and a reader: the real denoiser reads the rows through attention, and what the toy keeps is the contract that the prompt reaches the denoiser as an array and as nothing else. The empty prompt gives every class weight $\tfrac13$, which is the denoiser's unconditional case, the model that a conditional-dropout training run learns to be.

The autoencoder is linear, and on purpose it has a null space: for each $2 \times 2$ patch it keeps the mean and the left-right difference and throws away the top-bottom difference and the diagonal. The three prototypes lie in its range and decode exactly. A fourth image, the bar with a fine checkerboard of amplitude $0.25$ laid over it, does not: the checkerboard lies entirely in the part of the space that the latent does not carry. That one image will carry the chapter's second half, because a real autoencoder is also lossy in exactly this sense (the fine texture of a photograph is the part of it that the latent does not hold), and what a pipeline promises about the pixels it was told to keep depends on where it keeps them.

Let $\mathbf{x}_0$ be the standardized packed latent of an image, $\mathbf{m}_c$ the one of prototype $c$ (the class mean), and $\varsigma = 0.2$ the standard deviation of the clusters. The data are the mixture $\mathbf{x}_0 \sim \sum_c \pi_c\,\mathcal{N}(\mathbf{m}_c, \varsigma^2 \mathbf{I})$ with the class weights $\pi_c$ that the prompt gives. For the path $\mathbf{x}_\sigma = (1 - \sigma)\mathbf{x}_0 + \sigma\boldsymbol{\varepsilon}$ the velocity field of Chapter 6 is a sum over classes, each a Gaussian formula, weighted by the posterior of the class at $\mathbf{x}_\sigma$:

$$
\mathbf{u}(\mathbf{x}, \sigma) = \sum_c \mathrm{resp}_c(\mathbf{x}, \sigma)\,\mathbf{f}_c(\mathbf{x}, \sigma),
$$

$$
\mathbf{f}_c(\mathbf{x}, \sigma) = \frac{\sigma - (1 - \sigma)\varsigma^2}{\varrho}\bigl(\mathbf{x} - (1 - \sigma)\mathbf{m}_c\bigr) - \mathbf{m}_c, \qquad \varrho = (1 - \sigma)^2\varsigma^2 + \sigma^2,
$$

where $\mathbf{f}_c$ is the field of class $c$ alone and $\mathrm{resp}_c \propto \pi_c\,\mathcal{N}(\mathbf{x};\, (1 - \sigma)\mathbf{m}_c,\, \varrho\,\mathbf{I})$ normalized to sum to one. The guided field is Chapter 7's, $\mathbf{u}_{\mathrm{un}} + g(\mathbf{u}_c - \mathbf{u}_{\mathrm{un}})$, with $\mathbf{u}_c$ the field $\mathbf{u}$ at the prompt's class weights and $\mathbf{u}_{\mathrm{un}}$ the field at class weights $\tfrac13$. The sampler is Euler on the schedule rule of FLUX.1 at the toy's four image tokens. The stages are then checked one at a time.

<!-- worked: latent_diffusion_systems.toy_stages -->
Stage by stage. The text stage for five prompts:

| prompt | ids | w bar | w box | w slash |
|---|---|---:|---:|---:|
| `bar` | 1 3 2 0 0 0 | 1.0000 | 0.0000 | 0.0000 |
| `box` | 1 4 2 0 0 0 | 0.0000 | 0.9999 | 0.0001 |
| `slash` | 1 5 2 0 0 0 | 0.0000 | 0.0001 | 0.9999 |
| `bar box` | 1 3 4 2 0 0 | 0.4964 | 0.4964 | 0.0072 |
| (empty) | 1 2 0 0 0 0 | 0.3333 | 0.3333 | 0.3333 |

The codec, and the codec with the packing and the standardization after it: the largest pixel error of the round trip of each image. The prototypes are in the codec's range, and the textured source is not; the source is the bar plus a texture that adds +0.25 to the pixel at row y and column x where x + y is even and -0.25 where it is odd (a checkerboard, amplitude 0.25 on both diagonals of every patch), which is exactly what the codec cannot keep, so the error is the texture's amplitude:

| image | codec alone | with packing and statistics |
|---|---:|---:|
| bar | 0.0e+00 | 1.1e-16 |
| box | 0.0e+00 | 0.0e+00 |
| slash | 0.0e+00 | 1.1e-16 |
| bar plus texture (0.25) | 0.2500 | 0.2500 |

The denoiser at sigma = 1 for the prompt `bar box` and the first noise draw of `latent_diffusion_systems.noise` (seed 0x51e5c8a881704c6b): the field must be x minus the mean of the class means weighted by the prompt, and the largest difference is 4.4e-16. The posterior weights there are the prompt's: 0.4964, 0.4964, 0.0072.

The schedule of 8 steps on FLUX.1's rule at 4 tokens (mu = 0.4573437 unrounded: 0.5 + (4 - 256) x 0.65 / 3840, and all these digits are used):

| step i | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sigma_i | 1.0000 | 0.9171 | 0.8258 | 0.7248 | 0.6124 | 0.4866 | 0.3450 | 0.1841 | 0.0000 |
<!-- /worked -->

Four numbers carry the check. The text stage puts at least $0.9999$ on the right class for each single word and splits the weight evenly between `bar` and `box` for the prompt `bar box`. The codec returns the prototypes with no error and the textured source with an error of exactly the texture's amplitude. The packing and the standardization add rounding of $10^{-16}$ and nothing else, which is what an invertible per-channel affine map should do. And the denoiser at $\sigma = 1$ is $\mathbf{x} - \bar{\mathbf{m}}$ to $4 \times 10^{-16}$: with no signal left in $\mathbf{x}$ the posterior is the prior, and the conditional expectation of $\boldsymbol{\varepsilon} - \mathbf{x}_0$ is $\mathbf{x} - \bar{\mathbf{m}}$, with $\bar{\mathbf{m}} = \sum_c \pi_c\mathbf{m}_c$ the mean of the data. A pipeline that failed one of these would fail it in that stage and in no other, and that is the use of such a table: a wrong end-to-end picture has many possible causes, and a wrong row has one.

### Guidance and steps, counted

The whole pipeline can now be run many times. The table draws 100 starting noises, once, from a named stream, and runs the same noises through every setting.

<!-- worked: latent_diffusion_systems.sampling_table -->
100 starting noises (the first 100 draws of 32 numbers from `latent_diffusion_systems.noise` (seed 0x51e5c8a881704c6b), in order), the same for every row, sampled with 8 Euler steps of `toy_sigmas` and the guided field u_un + g (u_c - u_un). The columns count the samples whose nearest class mean is each class; the last is the mean distance of a sample to its nearest class mean (data have 0.2 sqrt(32) = 1.13 on average).

| prompt | g | bar | box | slash | mean distance to the nearest class mean |
|---|---:|---:|---:|---:|---:|
| `box` | 0 | 36 | 35 | 29 | 0.7115 |
| `box` | 1 | 0 | 100 | 0 | 0.6886 |
| `box` | 2 | 0 | 100 | 0 | 0.6936 |
| `box` | 4 | 0 | 100 | 0 | 0.7371 |
| `box` | 8 | 0 | 100 | 0 | 0.8464 |
| `bar box` | 0 | 36 | 35 | 29 | 0.7115 |
| `bar box` | 1 | 50 | 50 | 0 | 0.6948 |
| `bar box` | 2 | 51 | 49 | 0 | 0.6947 |
| `bar box` | 4 | 54 | 46 | 0 | 0.7144 |
| `bar box` | 8 | 61 | 39 | 0 | 0.7542 |

Convergence of the steps for the prompt `box` at g = 2 on the first 20 noises, against the same sampler at 512 steps.

| Euler steps | mean distance to the 512-step result | largest | mean distance to the box mean |
|---|---:|---:|---:|
| 1 | 3.9064 | 4.2237 | 3.9466 |
| 2 | 0.9758 | 1.1714 | 0.1566 |
| 4 | 0.7316 | 0.8765 | 0.3985 |
| 8 | 0.4310 | 0.5172 | 0.6963 |
| 16 | 0.2253 | 0.2704 | 0.9011 |
| 64 | 0.0543 | 0.0652 | 1.0712 |
| 512 | 0.0000 | 0.0000 | 1.1254 |
<!-- /worked -->

At $g = 0$ the sampler uses only the unconditional field, and a hundred noises fall into the three classes as 36, 35 and 29: the prompt does nothing. At $g = 1$ the guided field is the conditional one and all 100 land on `box`; the prompt `bar box` splits 50 and 50, the mixture the prompt asked for. Larger scales change things in the way Chapter 7 predicted: the mean distance to the nearest class mean, the spread of the sample around its cluster, grows from $0.69$ at $g = 1$ to $0.85$ at $g = 8$ for `box`, and the split for `bar box` drifts from 50 and 50 to 61 and 39 because guidance amplifies whatever imbalance the early steps happened to carry. Every number is exact in the sense of this volume: the only thing missing from the toy that the real pipelines have is a trained approximation of the field.

The steps table shows the other axis. One step from noise lands on the guided mean of the data, $\bar{\mathbf{m}} + g(\mathbf{m}_{\text{box}} - \bar{\mathbf{m}})$, which is $(g - 1)$ times the distance between the `box` mean and the mean of the data away from the `box` mean: $3.95$ at $g = 2$. Two steps land near the cluster with a spread of $0.16$ around its mean, and the spread grows with the steps toward the $1.13$ that the data have, reaching $1.1254$ at 512 steps. A few-step sampler returns a picture that is sharper at the level of the class and too uniform at the level of the cluster, and that is not a defect of the toy: it is what the straight path's average does when you cross it in a few steps, the behaviour of a distilled model that is trained to be good at exactly that.

## Starting from a picture

Everything so far began at noise. Most of what a person does with a generator begins at a picture instead, and the pipeline has to decide what a picture is allowed to mean. The simplest meaning is a starting point: instead of noise, start at the reference's latent noised to a chosen level. This is image-to-image generation, and its single dial is the **strength** $\upsilon \in [0, 1]$, the noise fraction at which the run starts. The initial latent is

$$
\mathbf{x}_{\upsilon} = (1 - \upsilon)\,\mathbf{x}_0 + \upsilon\,\boldsymbol{\varepsilon},
$$

the forward process of Chapter 6 evaluated at $\upsilon$ with the source latent $\mathbf{x}_0$ and the same kind of noise draw as before. At $\upsilon = 1$ this is noise, and the run is free generation; at $\upsilon = 0$ it is the source and nothing has to be computed.

What the sampler does between those two ends is a choice that went wrong once in brain's history, and the way it went wrong is worth knowing. The distilled schedule is shifted so hard that its lowest nonzero noise fraction is $0.7672$ at 4 steps (and $0.56$ at 8 steps for a canvas of 3,072 tokens): there is no low-noise entry in the list at which to start. The first implementation integrated a uniform ramp from $\upsilon$ to 0 for any strength below $1 - 10^{-3}$ and the klein schedule above it, which made two different samplers one dial position apart, and rendered, in the words of commit 62e5886db, a bedroom staged at strength $1.00$ and the duvet of the existing bedroom repainted at $0.995$. The fix is `img2img_sigmas`: the klein schedule multiplied by the strength,

$$
\sigma_i^{(\upsilon)} = \upsilon\,\sigma_i, \qquad i = 0, \dots, n.
$$

The starting noise fraction is $\upsilon$ because the model must be asked to denoise the distribution that the initial latent was drawn from, and the shape of the descent from there is the distilled sampler's own at every strength. At $\upsilon = 1$ this is the klein schedule bit for bit, since $1.0 \cdot x$ is exact, and every entry is linear in $\upsilon$, so lowering the dial lowers every noise fraction and moves none by more than the change in the dial. The schedule table above has two such rows, for $\upsilon = 0.6$ and $0.3$: the first step of the second is at $0.3$, the last at $0.2302$ before the terminal zero. The commit chose scaling over the slicing that the reference image-to-image pipelines do (cut the list of timesteps at the strength), because slicing a few-step shifted schedule at $\upsilon$ leaves no entries below it.

Strength $0$ deserves its own sentence. An earlier version clamped the strength to $10^{-3}$ and documented zero as "returns the source through the codec"; the clamp made the promise false, because the sampler ran at almost zero noise and returned a barely different sampling of the free target. Zero now skips the schedule and the sampler entirely, with zero evaluations of the denoiser, and the decoder receives $\mathbf{x}_0$ untouched, which is the exact round trip of the codec (commit 59e8b8cca). It is the right edge of the dial, and a gate asserts its bytes and the empty loop, not a tolerance.

<!-- worked: latent_diffusion_systems.strength_table -->
Image to image. The source is the bar with the texture; x0 is its standardized packed latent (the texture is already gone from it). Each of 40 noises starts the run at x = (1 - s) x0 + s eps with the schedule s times `toy_sigmas(8)` (so the first sigma is s), the prompt is `box` and g = 2. Strength 0 skips the sampler. The columns count the final latents nearest to each class mean, and the last is the mean root-mean-square pixel distance of the decoded result from the decoded source x0.

| strength s | bar | box | slash | mean pixel distance to the codec's source |
|---|---:|---:|---:|---:|
| 0 | 40 | 0 | 0 | 0.0000 |
| 0.25 | 40 | 0 | 0 | 0.1312 |
| 0.3 | 40 | 0 | 0 | 0.1352 |
| 0.35 | 39 | 1 | 0 | 0.1542 |
| 0.4 | 27 | 13 | 0 | 0.3758 |
| 0.45 | 6 | 34 | 0 | 0.7880 |
| 0.5 | 0 | 40 | 0 | 0.9261 |
| 0.65 | 0 | 40 | 0 | 0.9686 |
| 1 | 0 | 40 | 0 | 1.0219 |
<!-- /worked -->

The toy's dial has the shape the construction predicts, and the shape is sharper than the dial's name suggests. At $\upsilon = 0.3$ and below every one of the 40 runs stays with the source's class; at $0.5$ and above every one moves to the prompt's; and in between, where the noise at the first step is comparable to the distance between the class means, the sampler's own decision is made from the noise draw: 27 of 40 stay at $0.40$ and only 6 at $0.45$. The last column is the distance of the decoded result from the decoded source, and it is the dial's other reading: the strength is the noise fraction, and so a small one means that the run barely moves ($0.13$ of root-mean-square pixel distance at $0.25$, the decoded spread of a sample around its cluster) while a large one forgets the source's class.

### What a reference costs

A picture does a second thing in FLUX.2, and brain's pipeline found out the hard way that it has to do both. The model also takes reference images as conditioning: their latent tokens join the image part of the joint sequence, with a position id of their own, and the model attends to them. The first implementation used the init image of an image-to-image run as the starting latent and also as a conditioning token, and the strength appeared to do nothing: with the photograph attended at every step, the edit conditioning pinned the output at every noise level, and strengths of $0.1$, $0.5$ and $0.9$ returned the source. Commit a78665ac1 fixed that by skipping the init image as a reference token. The consequence, found a month later, was the opposite failure: the strength was a switch, not a dial, because at exactly $\upsilon = 1$ the same picture was attended as a reference and at any strength below it the model was blind to it. In the whole open interval the commit that found it says the model ran "blind to the user's photograph", with the reference reaching it only as leftover signal in a partly noised latent, which is why $0.95$ and $1.00$ looked like different models (commit acef10735). The rule today is a single function, `cond_sizes`, that the sizing, the position-id builder and the denoising loop all read: a supplied reference always conditions the model, and the strength decides only how much of the denoising starts from the init latent.

That rule has a price, and it can be counted.

<!-- worked: latent_diffusion_systems.reference_cost -->
The denoising and autoencoding work of one 1024 x 1024 FLUX.2 image at four steps, with 512 text rows and no cost counted for the text encoder. A reference joins the image stream, so a 1024 x 1024 reference doubles the image rows from 4,096 to 8,192 in every pass. The encoder runs once per encoded image, at 1024 x 1024 (the table of `latent_diffusion_systems.vae_work`). The pass rule for FLUX.2 (the one that reproduces Chapter 8's $34{,}819.8$ GFLOP for klein-4b), for n image rows, t text rows and N = n + t: each double-stream block costs 2 N (4 d^2 + 3 m d) and each single-stream block 2 N ((3 d + 2 m) d + d (d + m)) with m = 3 d the feed-forward width, attention costs 4 N^2 d in every block of both kinds, and the three embedders cost 2 n 128 d (image rows in), 2 t c d (text rows in, c = 7,680 for klein-4b and 12,288 for klein-9b) and 2 n d 128 (out). The widths and block counts are Chapter 8's: d = 3,072 with 5 double and 20 single blocks (klein-4b), d = 4,096 with 8 and 24 (klein-9b).

| variant | request | VAE encode, TFLOP | denoise, TFLOP | VAE decode, TFLOP | total, TFLOP |
|---|---|---:|---:|---:|---:|
| klein-4b | text to image | 0.00 | 139.3 | 10.47 | 149.8 |
| klein-4b | edit: one 1024 x 1024 reference, encoded once | 4.89 | 306.8 | 10.47 | 322.2 |
| klein-4b | strength below 1: the same image encoded twice | 9.78 | 306.8 | 10.47 | 327.1 |
| klein-9b | text to image | 0.00 | 302.1 | 10.47 | 312.5 |
| klein-9b | edit: one 1024 x 1024 reference, encoded once | 4.89 | 645.1 | 10.47 | 660.5 |
| klein-9b | strength below 1: the same image encoded twice | 9.78 | 645.1 | 10.47 | 665.4 |
<!-- /worked -->

A $1024 \times 1024$ reference doubles the image rows of every pass from 4,096 to 8,192, and attention grows with the square of the whole sequence, so the denoising work of klein-4b goes from 139.3 to $306.8$ TFLOP, a factor of 2.2 for twice the rows, and the 9-billion model's from 302.1 to 645.1. The dial `ref_resolution_scale` exists to buy this back (a reference at scale $s$ contributes $(s\,\text{side}/16)^2$ rows), and its default is $1.0$, the fidelity end, for the reason the code gives: the conditioning sequence must be a continuous function of the options, and nothing that the model sees may jump where the pipeline changes branch.

The table also holds a small observation about the cost of the rule. For $\upsilon < 1$ the first reference does double duty, as the init latent and as the conditioning tokens, and the code encodes it once for each use, the conditioning copy in the loop over the references and the init latent in the block that follows, both in `denoise_group_on` of `crates/flux2/src/pipeline.rs`. At the default scale of $1.0$ the conditioning copy is the same image at the same size, so the two encodes compute the same tokens from the same pixels. The last row of each group is what that costs: one additional encoder pass, $4.89$ TFLOP at $1024 \times 1024$, which is 1.5 percent of an image-to-image run of klein-4b and the whole of the encoder's bill again. The one-entry cache of the pipeline saves the second graph build, not the second forward. The tokens are a repeat of one deterministic computation on the same pixels, so sharing them when the sizes agree is a change of a few lines whose result should be bit-identical; it is the sort of saving that the ledger of operations finds and that a test of the output never will.

## Keeping a region

A global dial answers "how much of the source survives", and not "where". A person who wants the furniture of a room replaced and its walls and windows kept wants a map, not a number. In latent diffusion the map has the shape of the latent: a weight $\vartheta \in [0, 1]$ for every token, one for white (regenerate) and zero for black (preserve), obtained by averaging the mask over the pixels that each token covers. After every Euler step the sampler recombines the latent with the source renoised to the noise fraction it has just reached:

$$
\mathbf{x} \leftarrow \vartheta\,\mathbf{x} + (1 - \vartheta)\bigl((1 - \sigma_{i+1})\,\mathbf{x}_0 + \sigma_{i+1}\,\boldsymbol{\varepsilon}\bigr),
$$

where $\boldsymbol{\varepsilon}$ is the very noise draw that started the run. This is the rule of `crates/flux2/src/mask.rs`, which names its origin as blended latent diffusion, and it differs from the strength in the way the commit that introduced it said it did: the preserved region is not guided toward the source, it is the source, renoised to the level of the step and so a legal point on the source's own forward path, re-anchored at every step. At the last step $\sigma = 0$ it is $\mathbf{x}_0$ itself, and the code copies it rather than computing $x + 0\cdot\varepsilon$, which would differ from $x$ for a negative zero (commit 6813f0036). Two properties are exact on purpose: an all-white mask is bit for bit the unmasked run, and an all-black one reproduces the source latent bit for bit, which needs both the short circuits in `blend` and an integer area average in the resampler, so that a mask constant over a token is resampled to exactly that constant.

<!-- worked: latent_diffusion_systems.mask_table -->
A masked run of the toy: the source with the texture, the prompt `box`, g = 2, 8 steps, the first noise of `latent_diffusion_systems.noise` (seed 0x51e5c8a881704c6b), and the mask 0 on the two left tokens, at (row, column) = (0, 0) and (1, 0), raster indices 0 and 2, and 1 on the two right ones, raster indices 1 and 3, for all 8 packed channels of each. After every Euler step the kept tokens are replaced by (1 - sigma) x0 + sigma eps at the sigma just reached (x0 itself when it is 0). The largest difference between the final kept tokens and x0 is 0.0e+00. Decoded, the left half of the picture differs from the *source* by up to 0.2500, the amplitude of the texture, because the latent never held it; the left half equals the codec's round trip of the source to 0.0e+00; the right half is within 0.0911 (root mean square per pixel) of the box prototype's right half. A composite in pixel space, source + m (generated - source) per pixel column, keeps the source:

| pixel mask | largest change where the mask is 0 | largest change where the mask is between 0 and 1 |
|---|---:|---:|
| hard (columns 4 to 7) | 0.0e+00 | - |
| soft (0, 0, 0, 0.25, 0.75, 1, 1, 1) | 0.0e+00 | 1.7954 |
<!-- /worked -->

![Eight by eight pictures of the toy pipeline: the prototypes bar, box and slash; the bar with a fine texture and its round trip through the codec, which has lost the texture; image-to-image results at strengths 0.25, 0.5 and 1; the masked run, in which the left half is the source's round trip and the right half a box; and the pixel composite, which keeps the left half exactly, texture and all.](img/toy-pictures.svg)

The toy shows both what the latent blend promises and what it cannot. The kept tokens end equal to the source's tokens, with a largest difference of exactly $0$. The left half of the decoded picture, however, differs from the source by up to $0.25$: it is the round trip of the source through the codec, which has no room for the texture. A latent mask can keep the latent exactly and cannot keep the pixels, because the pixels were never in the latent. The right half is within $0.0911$ per pixel of the box. Everything about the left half of the figure follows from one sentence: a latent is a description, and a preserved description is not a preserved picture.

The way to promise the picture is to leave the latent. The crate `crates/imgpipe` composes an editing as stages on pixels: segment, grow or shrink or soften the mask, restore, and a size-changing tail, in one call. Its central choice is that "change only X, keep the rest" is an exact contract and not a perceptual one, and it is kept by arithmetic. The composite is `old + mask * (new - old)` (`crates/imaging/src/mask.rs`), so where the mask is exactly $0$ the new term is $0.0$ and the sum is $old$ in 32-bit floating point; the test asserts the bits (commit f76d43975). The toy's table has the same two rows. With a hard mask the pixels where the mask is $0$ are unchanged to $0$, texture included. With a soft one the unchanged region is still exact, and the band where the mask is neither $0$ nor $1$ moves, by up to $1.80$: the feather is that band, deliberately. This is why the crate returns the mask it actually composited with, as the authoritative record of which pixels were allowed to move.

The contract has the edges one would expect of a lerp, and a reader should know them. It needs the new image to be finite: $0 \cdot \infty$ and $0 \cdot \mathrm{NaN}$ are NaN, so a generator that overflowed in one pixel contaminates a pixel the mask says is protected. It cannot see a stored negative zero, for the reason `blend` takes care of in the latent (a pixel decoded from bytes is never negative zero, so the exposure is theoretical). And it has no stage that generates: its model stages are segmentation and face restoration and a super-resolution or restoration tail, dispatched through a registry so that the crate links no model at all, and a FLUX.2 edit reaches it, if at all, as an image blob, outside the crate. What `imgpipe` contributes to a generator is its guarantee, and the point that the guarantee lives at the resolution of the pixels.

## Windows over a canvas too large for one pass

A last request goes beyond what one forward fits. FLUX.2's attention is fused (the code's own comment: the scores are never materialized), but the per-token activations (queries, keys, values, the feed-forward hidden) are not, and they grow with the number of tokens. A canvas of four times the area of the native size is four times those buffers, and a resolution the arithmetic would serve fails to fit a card. FLUX.2's answer spends passes instead of memory: the canvas is denoised in overlapping windows, each a forward sized to a resolution known to fit, and the velocities (not finished images) are averaged over the overlap with weights that ramp across the part two windows share (MultiDiffusion). The averaging is on the velocity at every step, which at one fixed noise fraction is the same as averaging the clean-latent estimates, since the latent is common to the windows.

<!-- worked: latent_diffusion_systems.tiling_table -->
Windows of 1,024 x 1,024 pixels (64 x 64 tokens) with an overlap of 256 pixels (16 tokens), `Tiling::new(1024)`, over a larger canvas of FLUX.2 klein-4b, with the canvas-sized first reference that a windowed run always has (cropped to each window), so that a pass over n generated tokens has 2 n image rows by the pass rule of the reference-cost table above. Windows per axis come from `tile_starts`; the shared width is the number of pixels two neighbours have in common. TFLOP is one denoising step over all windows, and for comparison one step of a single forward over the whole canvas, with its generated and reference rows.

| canvas | tokens | windows | shared pixels | rows in one window's forward | TFLOP, all windows | TFLOP, one forward | one forward / windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 x 1024 | 4,096 | 1 x 1 | - | 8,704 | 76.7 | 76.7 | 1.00 |
| 1536 x 1536 | 9,216 | 2 x 2 | 512 | 8,704 | 306.8 | 226.5 | 0.74 |
| 2048 x 2048 | 16,384 | 3 x 3 | 512 | 8,704 | 690.3 | 544.5 | 0.79 |
| 3072 x 3072 | 36,864 | 4 x 4 | 336; 352 | 8,704 | 1227.2 | 2148.7 | 1.75 |
<!-- /worked -->

The table is the honest accounting of what the windows buy, and it says that the work does not go down. Up to a canvas of $2048 \times 2048$ the windows cost more operations than one forward over the whole canvas (the ratio in the last column is below $1$: $306.8$ against $226.5$ TFLOP at $1536 \times 1536$), because each window repeats the 512 text rows and every window carries its own crop of the canvas-sized reference, and only at $3072 \times 3072$ does the quadratic term win, with the single forward costing $1.75$ times the windows. What the windows buy is the size of the largest buffer: every window forward sees 8,704 rows, however large the canvas, and so the device footprint is that of the native size. Windows are chosen to share the same width: `tile_starts` derives the number of windows from the overlap it was asked for and then spaces them evenly, because pinning the last window flush against the edge gave one pair of windows a shared strip of 432 pixels where the others had 128 (commit 9ffa96a8f), and a wide strip is a wide cross-fade between two independent denoisings.

The failure that taught the pipeline what a window is not came from a window that could not see its neighbours. Handed only a prompt, each window of a $2048 \times 1152$ canvas composed its own whole scene, and the run drew the same canal-and-campanile vista three times, in three windows (commit 01faaaebc). Four distilled steps leave no rounds for the windows to reconcile a disagreement of that size through the shared latent. So a windowed run is never started without a full-canvas picture at its own grid: the first reference when it is already the canvas; that reference resampled when it is not; and, with no reference at all, a draft made in one forward at the largest size one window allows, enlarged and refined at strength $0.4$. Each window is conditioned on its own crop of that picture. The guarantee is a type: the denoising functions accept only an `Anchored` request, and the one function that builds it is the one that drafts or resamples, so tiled denoising against nothing is not a state that a later edit can reach.

## Who is on the card, and when

Everything so far has been about operations. The failure at the opening was about bytes, and it is the other half of what a pipeline is: three or four models, each with weights, and between them a rule for which are on which device at which moment. The FLUX.2 pipeline does not name a card. It declares its parts, with what each costs and when, and an engine (`gpu_core::devices::place`, answering from `crates/residency/src/plan.rs`) places them (commit 36c063914). The denoiser and the text encoder are declared apart, so that a two-card machine never stacks them; the decode graph is declared with the denoiser, because it decodes the denoiser's own latents and a split across cards would cost a transfer for them; and each part carries a phase. Parts in different phases never share a card at the same time, and a card is charged its permanent bytes plus its heaviest single phase, not their sum. The denoiser (with the reference encoder, if a reference can occur) is phase 1, the decode is phase 2, and the text encoder is permanent unless it cannot fit beside the others, in which case it moves to a phase 0 of its own.

The planner is only as good as the bytes it is told. They can be derived from the configuration, and the derivation is worth having because it gives a number to check the planner against.

<!-- worked: latent_diffusion_systems.placement_table -->
Bytes (GiB = 2^30) that the parts of a FLUX.2 run hold on a device at 1024 x 1024: the denoiser's weights and its scratch for the joint sequence of 512 text rows, 4,096 generated rows and the reference's 4,096 if there is one (DiT, priced before: what `flux2::pipeline::dit_bytes` summed before commit 38e8b2b7a; DiT, uploaded: the layout the constructor writes, `flux2_weight_bytes`, which `dit_bytes` has summed since), the truncated text encoder's weights at the same tier, and the autoencoder's decode and encode graphs. The rules. Uploaded layout of the denoiser's weights: the six tensors of the clock and the modulations (the two layers of the clock, the two double-stream modulations, the single-stream modulation and the final layer's modulation) stay on the host and are not counted; at the f32 tier every other weight is 4 bytes; at the int8 tier the image and text input matrices, the final linear, every vector (the query and key gains) and the second feed-forward matrix of each double block stay at 4 bytes a weight, and every other matrix costs 36 bytes per 32 weights (32 int8 bytes and one 32-bit scale). Priced before: every two-dimensional tensor at 1 byte under int8 and everything else, and everything at f32, at 4 bytes, the host tensors included. Scratch for a joint sequence of J rows (4 bytes a word): J (16 d + 3 m + 2 x 128 + 128) words for the activations (m = 3 d, 128 the latent channels and the head width), t c words for the text context, 17 d for the clock, and for attention 2 words under int8 (fused) and otherwise 2 H J^2 (H heads); under int8 also J (d + m) bytes of packed activations and 4 J floor(max(d, m) / 32) bytes of scales (priced before: 4 J). Text encoder: the token table and 27 blocks (as deep as the deepest tap of Chapter 9's table), with norms and the table at 4 bytes and each block's seven matrices at 4 bytes, or at 36 bytes per 32 weights under int8. Autoencoder graphs: the weights at 4 bytes, plus 928 bytes per output pixel times 11 (decode) or 8 (encode) live buffers, plus 768 MiB (928 = the 4-byte words of the channel ladder 128, 256, 512, 512 at 1, 1/4, 1/16, 1/64 of the pixels).

| request (1024 x 1024) | DiT, priced before | DiT, uploaded | encoder | decode | encode |
|---|---:|---:|---:|---:|---:|
| klein-9b f32, no reference | 40.67 | 39.54 | 21.73 | 10.90 | - |
| klein-9b f32, one reference | 55.24 | 54.11 | 21.73 | 10.90 | 8.13 |
| klein-9b int8, no reference | 10.31 | 13.35 | 7.78 | 10.90 | - |
| klein-9b int8, one reference | 11.94 | 14.99 | 7.78 | 10.90 | 8.13 |
| klein-4b int8, no reference | 5.00 | 6.10 | 4.30 | 10.90 | - |
| klein-4b int8, one reference | 6.23 | 7.33 | 4.30 | 10.90 | 8.13 |

The four arrangements on cards of 24 GiB, charged as the planner charges them: a card holds its permanent parts and the heaviest single phase (the denoiser, with the reference encoder if there is one, is phase 1; the decode is phase 2).

| request | arrangement | priced before, GiB per card (24 GiB) | uploaded layout, GiB per card |
|---|---|---|---:|
| klein-9b int8 | all resident on one card | 28.99, does not fit | 32.03, does not fit |
| klein-9b int8 | one card, encoder permanent, denoiser and decoder in turn | 18.68, fits | 21.13, fits |
| klein-9b int8 | one card, encoder, denoiser and decoder each in turn | 10.90, fits | 13.35, fits |
| klein-9b int8 | two cards: denoiser and decoder in turn | encoder | 10.90 and 7.78, fits | 13.35 and 7.78, fits |
| klein-9b int8, one reference | all resident on one card | 38.75, does not fit | 41.80, does not fit |
| klein-9b int8, one reference | one card, encoder permanent, denoiser and decoder in turn | 27.85, does not fit | 30.90, does not fit |
| klein-9b int8, one reference | one card, encoder, denoiser and decoder each in turn | 20.07, fits | 23.12, fits |
| klein-9b int8, one reference | two cards: denoiser and decoder in turn | encoder | 20.07 and 7.78, fits | 23.12 and 7.78, fits |
| klein-4b int8, one reference | all resident on one card | 29.56, does not fit | 30.67, does not fit |
| klein-4b int8, one reference | one card, encoder permanent, denoiser and decoder in turn | 18.66, fits | 19.76, fits |
| klein-4b int8, one reference | one card, encoder, denoiser and decoder each in turn | 14.35, fits | 15.46, fits |
| klein-4b int8, one reference | two cards: denoiser and decoder in turn | encoder | 14.35 and 4.30, fits | 15.46 and 4.30, fits |
| klein-9b f32 | all resident on one card | 73.30, does not fit | 72.17, does not fit |
| klein-9b f32 | one card, encoder permanent, denoiser and decoder in turn | 62.40, does not fit | 61.27, does not fit |
| klein-9b f32 | one card, encoder, denoiser and decoder each in turn | 40.67, does not fit | 39.54, does not fit |
| klein-9b f32 | two cards: denoiser and decoder in turn | encoder | 40.67 and 21.73, does not fit | 39.54 and 21.73, does not fit |
<!-- /worked -->

Three readings of the first table matter. The decode graph is 10.90 GiB at $1024 \times 1024$, more than the whole 4-billion denoiser at int8 (6.10 GiB with its scratch), and 8.13 GiB more is wanted for the reference encoder: the codec is not a small part. The layout the constructor uploads for the 9-billion model at int8 is 13.35 GiB with its scratch, the same figure as the commit that introduced phased placement gave for a 9B denoiser beside a full-frame decode ("13.35 + 10.90 GiB", commit fa81ef938), which agrees with the derivation to the digits shown. And the planner's own figure for the same part, `dit_bytes`, was 10.31 GiB until this chapter compared the two. It counted every two-dimensional tensor at one byte, but the layout carries a 32-bit scale for every 32 weights (36 bytes per 32 weights, the group-wise layout of `crates/model/src/int8.rs`) and keeps the boundary linears, the double blocks' second feed-forward matrix and the vectors at 32 bits. The first adds 12.5 percent to the quantized part; the second is 3.19 GiB at 32 bits, which the planner priced at a quarter of that. A checkpoint that arrives as a Q8_0 file was already priced from its own header (`gguf_dit_device_bytes`); a safetensors one at int8 was priced 3.04 GiB short for the 9B model, on the side of the failure that opened the chapter, and the f32 tier erred the other way by 1.13 GiB, because it counted the six tensors that stay on the host. Commit 38e8b2b7a makes `dit_bytes` price each tensor by the same rule the constructor and the Q8_0 route use, and a test now holds the two routes equal on a fully-Q8_0 file; the column of the table headed "priced before" is the old figure, kept for the comparison.

The second table places the arrangements on cards of 24 GiB, the size of the cards that the placement commits were developed on. Keeping the encoder resident beside the phases of the others is preferred, because a server that sees many prompts would otherwise rebuild models for each; the table shows when that stops being possible.

![Bytes on a card through three phases for a FLUX.2 klein-9b run with one reference at int8: everything resident at 41.80 GiB; one card with the encoder resident, peaking at 30.90 GiB in the denoising phase; the encoder in a phase of its own, peaking at 23.12 GiB; and the two-card layout, whose denoiser card peaks at the same 23.12 GiB and whose second card holds the 7.78 GiB encoder.](img/card-phases.svg)

For the same request without a reference, one card holds the encoder, the denoiser and the decode in turn at 21.13 GiB, and everything resident would be 32.03. With a reference the resident encoder no longer fits (30.90 GiB), the plan puts it in a phase of its own before the denoiser, which fits at 23.12 GiB of the uploaded layout, with about 0.9 GiB to spare on the 24 GiB card, and a second card takes the encoder off the first and leaves the same 23.12 GiB there. Each of these is a different pipeline from the standpoint of memory, and the planner picks among them by cost and by what is free at that moment.

### What the plan costs to carry out

A phase is only a plan until something acts on it, and the part of the code that acts is a handful of lines. The denoiser shares a card with the decode graph, so before the decode allocates, `decode_tokens` releases the reference encoder's cache and then the denoiser itself, and the next forward rebuilds it (commit fa81ef938). The text encoder takes a card in a phase before the denoiser's, is released after it encodes, and is rebuilt for the next prompt. Both are **weight streaming** in its simplest form: moving a stage's weights onto a device when its turn comes and off when it ends, so that the card holds the largest stage and not the sum of all of them. Its finer form is a window of blocks rotating through a fixed pool of device buffers inside one network, which Chapter 8 met in the Z-Image family. What the schemes cost can be counted in bytes.

<!-- worked: latent_diffusion_systems.traffic_table -->
Bytes that cross onto the device per image, from the uploaded layouts of the placement table above and the parameter counts of Chapters 8 and 9, and the operations a pass does for each byte it would have to upload (a pass of 1,024 x 1,024 with 512 text rows, by the pass rule of the reference-cost table for FLUX.2 and by Chapter 9's rule for FLUX.1). The columns are the GiB uploaded per image and per pass, and the operations per uploaded byte. FLUX.1's weights are counted from its manifest in 32-bit floats; its encoders are T5-XXL and CLIP-L in 32-bit floats.

| pipeline | scheme | GiB per image | GiB per pass | FLOP per byte |
|---|---|---:|---:|---:|
| FLUX.2 klein-9b, int8 | resident across images | 0 | 0 | - |
| FLUX.2 klein-9b, int8 | evicted for the decode, rebuilt for the next image | 11.49 | 11.49 | 24,487 |
| FLUX.2 klein-9b, int8 | every weight uploaded again for every pass | 45.95 | 11.49 | 6,122 |
| FLUX.2 klein-9b, f32 | every weight uploaded again for every pass | 130.77 | 32.69 | 2,151 |
| FLUX.1 dev, f32 | the encoders built and dropped for each prompt | 18.20 | - | - |
| FLUX.1 dev, f32 | resident across images | 0 | 0 | - |
| FLUX.1 dev, f32 | every weight uploaded again for every pass | 2216.81 | 44.34 | 1,562 |
<!-- /worked -->

The last column is the operations a pass does for each byte it would have to upload, and it is the number to hold against the roofline of Volume IV: a stream is hidden behind the arithmetic when that ratio is above the ridge $P/B$ of the link, the operations a device does for each byte its link can bring. For FLUX.2 klein-9b at int8 a pass does $6{,}122$ operations per uploaded byte if every pass re-uploads the weights, and $24{,}487$ if the weights are uploaded once per image and used by the four passes; at 32 bits the first figure is $2{,}151$ and for FLUX.1 dev it is $1{,}562$, so that for a device whose ridge for its host link is higher than those, streaming blocks every pass is traffic-bound, whatever the device's peak. The block-level form is therefore a tool for the models and links where the ratio is high, and the stage-level form, which uploads once an image, for the rest.

One row of the table is a cost that the code pays without asking whether it must. When the denoiser shares a card with the decode, the plan charges them as phases and `evict_denoiser_before_decode` evicts the denoiser unconditionally on every image whenever the two are on one accelerator. On a card with room for both, a single upload is wasted per image, 11.49 GiB for the 9-billion model at int8 (the second row of the table). Resident across images is the better scheme on a card that has the room, and the code does not ask whether it has.

FLUX.1 takes the other end of the same trade. Its pipeline is the older of the two and its documentation says the debt: only the denoiser stays resident, and the T5-XXL and CLIP-L towers and the decoder are "built for one encode/decode and dropped", the tiering of the SDXL pipeline. Every prompt (and every negative prompt, when the second guidance branch is on) therefore builds 18.20 GiB of encoder weights, and the decode runs, by default, on the host: the `vae_device` param defaults to the CPU, which is the sound choice while the resident model owns the card, though the code does not say that is why. The cost of that choice is the same $10.5$ TFLOP of arithmetic done on a processor with less of it, and a graph of about 10.9 GiB of ordinary memory by the decoder's own estimate; the benefit is the end of the failure that began the chapter. The device for the encoders deserves one more note. The first implementation of the host tier did not run on the host at all: a part "placed on the host" built on whichever card was ambient, so an operator's explicit `cpu` was a label and not a placement, which is why the lesson is to assert which backend was built and not what the planner returned.

Two sizes in the FLUX.1 pipeline went wrong in ways the byte ledger would have caught. The estimate for T5-XXL's weights counted two feed-forward matrices a block where the model has three, which is one matrix of $10240 \times 4096$ in each of 24 blocks and about 3.75 GiB at 32 bits; the plan found a card that fit the estimate, and the real allocation did not (commit dec49aaec). The fix reads the parameter count that the encoder's own manifest gives, instead of re-deriving it by hand. And the buffer of the denoiser was sized for the image tokens and not for the joint sequence that includes the 512 text rows, which panicked on the first real T5 context; the size is now one named constant that the parameter bound and the buffer share. These were not errors of arithmetic in the model but errors of bookkeeping about the model, which is the subject of this section: the planner is a second implementation of the pipeline's memory behaviour, and it is checked only by the real runs and by the derivations of the kind in this chapter.

## Check your understanding

1. Find the error: "A pipeline whose denoiser is a distilled four-step model can ignore the decoder, since the decoder runs once and the denoiser four times."
2. A request with a mask is run with $\upsilon = 1$ so that the whole canvas starts at noise. A colleague proposes dropping the renoising of the preserved tokens, and copying the source tokens in at the end only. What does the denoiser see in the preserved region during the run in each case, and which one keeps the model on the distribution it was trained on?
3. The toy's textured source is passed through a masked latent run, and the left half of the decoded picture differs from the source by $0.25$. Give two changes to the pipeline, one in the codec and one outside the latent, that would make the left half equal the source, and say what each costs.
4. A strength of $0.8$ is requested of a distilled four-step model, and the implementation slices the distilled schedule at $0.8$ instead of scaling it. Using the schedule table, say how many steps remain, and what they cost the request.
5. A canvas of $1536 \times 1536$ is generated in windows. A reviewer says the windowed run must cost fewer operations than the one-forward run, because each window's forward is shorter. Use the tiling table to say what is wrong with the claim.
6. Before commit 38e8b2b7a, the planner could say that the 9-billion model at int8, loaded from a safetensors checkpoint, fits a 24 GiB card with one reference, and the build would run out of memory. Give the two figures of this chapter that explain it, and say which direction each moves the plan.
7. Two pipelines are given the same seed. Why does a picture from FLUX.1 not tell you what FLUX.2 would draw for it?

## Summary

An image generation pipeline is four stages with two ledgers. The ledger of operations says that the denoiser's passes are 91 percent to 99.6 percent of an image's arithmetic and that the stages run once, the text encoders and the decoder, are visible only when the step count is small: for klein-4b the decode is 6.86 percent of the work. The ledger of bytes says that the decode graph, which runs last, is larger than the whole 4-billion denoiser at int8, that a planner's figure for a part is a second implementation of its memory behaviour and can be short by almost a quarter, and that the plan has to be acted on: phases are charged as a maximum, which is true only if something evicts.

Between the stages are contracts that no stage checks for the others (the scale and packing of the latent, the order and the ids of the rows, the schedule, the end of the pipeline, the stream of the noise), and a pipeline small enough to be run by hand, such as the toy of this chapter, checks them one stage at a time. A picture as input has two meanings, which brain's pipeline learned to keep apart: a starting point, with the dial $\upsilon$ that scales the distilled schedule into $[0, \upsilon]$, and a conditioning input that the model sees at every strength. A mask keeps a region of the latent exactly and cannot keep its pixels; the pixels are kept, exactly, by a composite outside the latent. A canvas past one forward is windowed, which spends passes for memory, and needs a full-canvas anchor so that the windows do not each compose a scene.

What has not been done yet is steering a generator by more than a prompt and a picture: a pose, an edge map, a particular face. The next chapter adds those as adapters on the same pipeline, and the seams (the per-block injection of `crates/flux1/src/inject.rs`, the reference tokens) are already in the code that this chapter described.

## References

- Robin Rombach and others, *High-Resolution Image Synthesis with Latent Diffusion Models*, 2022. Diffusion in the latent space of a frozen autoencoder, and the weight of the KL term that makes the scale a stored factor.
- Patrick Esser and others, *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*, 2024. The rectified-flow transformer over a packed latent, with several text encoders, and the resolution-dependent shift of the schedule.
- Omri Avrahami, Ohad Fried and Dani Lischinski, *Blended Latent Diffusion*, 2022. Replacing the region outside a mask by the source at the noise level of each step.
- Omer Bar-Tal and others, *MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation*, 2023. A canvas denoised in overlapping windows with the predictions averaged at every step.
- Chenlin Meng and others, *SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations*, 2022. Starting a sampler from a noised picture, and the strength as a dial.
- Jonathan Ho and Tim Salimans, *Classifier-Free Diffusion Guidance*, 2021. The guided field and its cost of a second pass.

## Exercises

1. **[Calculate]** From the up-block table of `latent_diffusion_systems.vae_work`, compute the share of the 1024 by 1024 decode spent at 256 by 256 and above, and the operations that 50 passes of the levels at 512 and 1024 would add if the decoder's last two levels were iterated at every step.
2. **[Calculate]** Reproduce one row of `latent_diffusion_systems.stage_work`: for klein-9b, take the 4 passes and the text and decode figures and recompute the total and the three shares. Then change the step count to 8 and say what the decode's share becomes.
3. **[Calculate]** Using the strength rows of `latent_diffusion_systems.schedule_table`, give the four noise fractions that the denoiser is evaluated at for $\upsilon = 0.45$, and the distance the latent travels in the last step as a fraction of the distance travelled in the first.
4. **[Calculate]** A reference of $512 \times 512$ is used as a conditioning image beside a $1024 \times 1024$ generation. How many rows join the image stream, and what is the denoising work of klein-4b at four steps (use the pass rule printed under the reference-cost table)?
5. **[Calculate]** Reproduce the DiT, uploaded column of `latent_diffusion_systems.placement_table` for klein-4b at int8 with no reference: the weights by the rules printed under the placement table (which tensors are on the host, which stay at 32 bits, the 36 bytes per 32 weights; the scratch rule) for 4,608 joint rows.
6. **[Implement]** Build the toy of `latent_diffusion_systems.toy_setup` and reproduce `latent_diffusion_systems.toy_stages`: the text table from the stream `latent_diffusion_systems.text`, the packed statistics, the codec checks and the field at $\sigma = 1$ to $10^{-12}$.
7. **[Implement]** Reproduce `latent_diffusion_systems.sampling_table`: the 100 noises from the stream `latent_diffusion_systems.noise` in order, 8 Euler steps on `toy_sigmas`, the counts of the nearest class and the mean distance for each $g$.
8. **[Implement]** Reproduce `latent_diffusion_systems.strength_table` and `latent_diffusion_systems.mask_table`, including the exact zero in the pixel composite where the mask is $0$ and the largest change in the soft band.
9. **[Investigate]** In the toy, replace the scaling of the schedule by $\upsilon$ with slicing the 8-step schedule at the first entry below $\upsilon$ (and starting at that entry's noise fraction). Compare the table of final classes with `strength_table` and say where the two disagree and why.
10. **[Investigate]** Make the toy's mask blend skip the renoising (copy the source tokens in only after the last step) and compare the classes and the distance to the source's tokens at the boundary between the kept and the regenerated tokens with the blended run.
11. **[Investigate]** Extend `latent_diffusion_systems.placement_table` to a $2048 \times 2048$ canvas without windows and with a window of 1024: what does the DiT's scratch become, and does the arrangement of the second table change?

## Key terms

- **latent diffusion**: running the iterated denoiser in the compressed latent space of an autoencoder that was trained earlier and is frozen at generation time, so that the full-resolution decoder runs once at the end and not at every step. <!-- concept: latent-diffusion -->
- **image generation pipeline**: the sequence of stages that turns a request into an image (text encoding, optional reference encoding, the sampled denoising loop and the decode) together with the contracts between them: shapes, latent scale and packing, row order and position ids, schedule, randomness, and which weights are on which device when. <!-- concept: image-generation-pipeline -->
- **weight streaming**: moving the weights of a stage, or of a window of blocks, onto a device when their turn comes and off when it ends, so that the device holds the largest part and not the sum of all of them, at the price of the bytes uploaded each time. <!-- concept: weight-streaming -->

## Answers

**Recall questions.**

1. The latent holds $32 \cdot 128 \cdot 128 = 524{,}288$ numbers; packed in $2 \times 2$ patches it is 4,096 tokens of 128 numbers (and $4096 \cdot 128 = 524{,}288$).
2. The last step is from $0.7672$ to $0$, of size $-0.7672$. It lands exactly on the clean-latent estimate, $x_0 = x_\sigma - \sigma u$: the last step leaves no noise.
3. $\mathbf{u}_g = \mathbf{u}_{\mathrm{un}} + g(\mathbf{u}_c - \mathbf{u}_{\mathrm{un}})$. At $g = 1$ it is $\mathbf{u}_c$ and one pass is enough; at $g = 3$ it needs both, two passes a step.
4. $4096 + 512 = 4608$ rows. The final layer predicts the first rows of the image part, the generated image's 4,096, and not the text rows (and not the reference rows, which follow them).
5. It refuses it, with the token count and the window in the message, rather than conditioning on a prefix of what was asked, because a truncated prompt is a well-formed input and no later stage could detect it.
6. The ridge point $P/B$: above $P/B$ operations per byte the kernel is limited by arithmetic, below it by traffic.

**Check your understanding.**

1. The decoder runs once but is not small: $10.5$ TFLOP, against 139.3 for the four passes of klein-4b, so 6.86 percent of the picture's arithmetic and 10.90 GiB of device memory at $1024 \times 1024$, the largest single part of the run. The share of a once-run stage rises as the steps fall, which is why the first failure of the chapter happened at the decode.
2. Without the renoising the denoiser sees, in the preserved region, whatever it has itself generated there, which was never the source's forward path, and so the regenerated region is conditioned on a latent of its own making beside a region that is replaced only at the end: a seam. With it, every step the preserved tokens are at $(1 - \sigma)\mathbf{x}_0 + \sigma\boldsymbol{\varepsilon}$ at the noise fraction the step has reached, a distribution the model was trained on.
3. In the codec: a latent with room for the texture (more channels, or a codec whose range holds it), which costs a larger latent and so more rows in every pass. Outside the latent: the composite of the pixel contract, `old + mask * (new - old)`, which costs nothing in the generator but needs the mask at pixel resolution and a finite generated image.
4. Of the four values $1, 0.9674, 0.9081, 0.7672$ only $0.7672$ lies below $0.8$, so slicing leaves a single step: the request keeps one of its four steps. Scaling keeps all four, at $0.8\,\sigma_i$.
5. Each window carries its own 512 text rows and its own crop of the canvas-sized reference, so a window's forward is 8,704 rows, and four of them cost $306.8$ TFLOP against $226.5$ for the whole canvas: shorter forwards, but more of them, with the text and reference rows repeated in each. The windows save memory (the buffers are those of one window) and not operations.
6. The planner's old figure for the denoiser, 10.31 GiB without a reference, is 3.04 GiB below the uploaded layout of 13.35 GiB (the scales, and the tensors that stay at 32 bits); and a request with a reference adds the encoder's graph, 8.13 GiB, to the denoiser's phase, which an estimate made for a request without one does not contain. Both make the plan optimistic.
7. The two loops draw the starting noise with different generators (SplitMix64 in 32-bit floats and xorshift64\* with a Box-Muller transform in 64-bit floats), so equal seeds are different noises, and in any case the two models were trained on different latents.

**Exercises (calculation answers).**

1. $29.5\% + 33.1\% + 21.3\% = 83.9\%$ at 256 by 256 and above. Fifty passes of the last two levels: $50 \cdot (3.470 + 2.233) = 285$ TFLOP.
2. $5.45 + 302.1 + 10.47 = 318.0$ TFLOP; shares $1.71\%$, $94.99\%$, $3.29\%$. At 8 steps the denoise is $604.2$ TFLOP, the total $620.1$ and the decode $1.69\%$.
3. $0.45 \cdot (1, 0.9674, 0.9081, 0.7672) = 0.45, 0.4353, 0.4087, 0.3452$, then $0$. The first step is $0.0147$ long and the last $0.3452$, 23.5 times the first.
4. $(512/16)^2 = 1{,}024$ reference rows, so 5,120 image rows and an image stream of $4096 + 1024$. The work is four passes of the pass rule at 5,120 image rows, between the $139.3$ TFLOP of no reference and the 306.8 of a full-size one.
5. The weights follow the rule given in the text for the 4B manifest (about $4.7$ GiB of layout out of $14.4$ GiB at 32 bits) and the scratch is $4608 \cdot (16 \cdot 3072 + 3 \cdot 9216 + 2 \cdot 128 + 128)$ words of 4 bytes plus $512 \cdot 7680$ words of context plus 17 times 3072 plus the int8 packed rows; the sum reproduces the table's $6.10$ GiB.

**Investigate.** Exercise 9: slicing at 0.45 gives a start at the first entry below it and fewer steps, so the final classes should flip at a lower strength and the distance column should rise faster; the scaled schedule keeps all 8 steps. Exercise 10: without the renoising the kept tokens are never anything but the source in the output, but the regenerated tokens are generated without having seen a consistent neighbour, and the distance across the seam should be larger. Exercise 11 is open; the scratch grows with the rows of one forward, and a window of 1024 holds it at the native size.
