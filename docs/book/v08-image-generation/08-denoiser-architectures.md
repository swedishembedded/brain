<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Denoiser Architectures: U-Nets and Diffusion Transformers

*What network shapes are good at predicting noise, and how do they take in the prompt?*

The denoiser of the last three chapters was a function of two numbers. It was handed a point on a line and a noise fraction, it returned a guess of where the clean point had been, and everything the chapters proved about it (that it is a posterior mean, that it converts into a velocity, that a solver may step with it) held because the function was small enough to write down exactly. The denoiser of a real image generator answers the same question about 65,536 numbers at once, the $4 \times 128 \times 128$ latent of a $1024 \times 1024$ picture. It is also handed the prompt, which in SDXL is 77 vectors of 2048 numbers, 157,696 further inputs, and it is handed the noise fraction, which is still one number, now outnumbered a quarter of a million to one.

Three questions follow from that count. The noise fraction is a single scalar among all those inputs, and a network built from convolutions and attention has no obvious place to put it: where must it enter, and in what form? The prompt is a sequence and the picture is a grid, two different kinds of object: how do they meet? And what shape of network is good at the main job, which is to return an array as large as its input when the answer at each position depends on both the fine detail around it and the picture as a whole? Two families answer, and brain carries both. The **U-Net** of SDXL keeps the encoder-decoder shape of Volume VII and adds a clock and a way to read text. The **diffusion transformer** of Z-Image and FLUX.2 cuts the latent into tokens and lets the transformer block of Volume V do everything, with the clock fed through the normalisation layers. This chapter takes the prompt as given, a sequence of vectors that some encoder produced; where that sequence comes from is the next chapter's question.

Six earlier ideas return within a few pages, so try them first from memory (the answers are at the end, under "Recall questions").

1. (Volume VII, [Chapter 5](../v07-vision/05-dense-prediction.md), skip connections.) In the encoder-decoder segmenter a decoder map with 8 channels is concatenated with the encoder map of the same size, which has 4. How many channels does the next $3 \times 3$ convolution read, and how many parameters does it have if it returns 4 channels?
2. (Volume V, [Chapter 6](../v05-sequences/06-seq2seq.md), cross-attention.) In cross-attention from 5 positions to 3, which sequence supplies the queries, which the keys and values, and what is the shape of the matrix of attention weights?
3. (Volume V, [Chapter 9](../v05-sequences/09-transformer-block.md), the transformer block.) A pre-norm block of width $D$ has four $D \times D$ attention matrices and a SwiGLU network of hidden width $8D/3$ with three matrices. How many parameters is that, as a multiple of $D^2$?
4. (Volume V, [Chapter 8](../v05-sequences/08-positions.md), sinusoidal encoding.) For a sinusoidal encoding of dimension 8 and base 10000, what are the four rates, and which pair of coordinates changes fastest as the position grows?
5. (Volume III, [Chapter 10](../v03-neural-networks/10-normalization.md), group normalization.) A group norm with 32 groups is applied to a map of 320 channels and $H \times W$ positions. Over how many numbers is each mean and variance taken?
6. (Volume VIII, [Chapter 6](06-flow-matching.md), the velocity.) On the straight path $x_\sigma = (1 - \sigma)x_0 + \sigma\varepsilon$, what is the velocity $dx_\sigma/d\sigma$, and what does the exact network return for it?

## A denoiser that is not told how noisy its input is

Before building anything, find out what is lost if the noise fraction is simply left out. The question has an exact answer on the three clusters of the last chapters, because the best possible denoiser, with or without knowledge of $\sigma$, is a posterior mean that can be written down. A denoiser that is told $\sigma$ computes $\mathbb{E}[x_0 \mid x, \sigma]$. One that is not told computes $\mathbb{E}[x_0 \mid x]$ with $\sigma$ drawn from the distribution of noise levels it was trained on, which is the same formula with 18 mixture components (six noise levels by three clusters) in place of three. Neither needs training.

<!-- worked: denoiser_architectures.blind_denoiser -->
The data are the three clusters (centres -3, 0, 3, standard deviation 0.25, equal weights) and x = (1 - sigma) x_0 + sigma noise. Two exact denoisers, no training: the posterior mean E[x_0 | x, sigma] that is told sigma, and the posterior mean E[x_0 | x] when sigma is one of 0.02, 0.1, 0.3, 0.5, 0.7, 0.9 with equal probability and is not told. Squared error E (guess - x_0)^2 at each sigma, by Simpson's rule with 16000 panels over x in [-8, 8] (the integrand is a sum of three Gaussians of width at least 0.24, so the rule is converged far below the digits printed).

| sigma | told sigma | not told | ratio |
|---|---:|---:|---:|
| 0.02 | 0.0004 | 0.0148 | 35.8 |
| 0.1 | 0.0103 | 0.0246 | 2.4 |
| 0.3 | 0.0531 | 0.1140 | 2.1 |
| 0.5 | 0.6295 | 0.7867 | 1.2 |
| 0.7 | 2.7320 | 2.8862 | 1.1 |
| 0.9 | 5.6397 | 6.2654 | 1.1 |

Averaged over the six levels the error is 1.5108 when told and 1.6820 when not told. The guesses at three values of x:

| x | told, sigma = 0.02 | told, sigma = 0.1 | told, sigma = 0.3 | told, sigma = 0.5 | told, sigma = 0.7 | told, sigma = 0.9 | not told |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.5068 | 0.4639 | 0.1815 | 0.6098 | 1.2152 | 0.3648 | 0.6404 |
| 1.5 | 1.5366 | 1.8860 | 2.7815 | 2.9597 | 2.6183 | 1.0497 | 2.4868 |
| 2.0 | 2.0472 | 2.3505 | 2.9637 | 3.0564 | 2.8701 | 1.3525 | 2.8606 |
<!-- /worked -->

Averaged over the six levels the two errors are $1.5108$ and $1.6820$, an 11% difference that is dominated by the noisiest levels, where both errors are large because little of the clean value is left in $x$. The ratio column is the more telling one. At $\sigma = 0.02$ a denoiser that knows the level makes an error of $0.0004$, because the input is nearly the answer, and one that does not know makes $0.0148$, thirty-six times more: it has to allow for the possibility that this $x$ came from a much noisier level, and so it pulls every input toward the nearest cluster centre. At $x = 2.0$, which is four standard deviations from the cluster at 3 and so a point the data seldom produce, the guess for $\sigma = 0.02$ is $2.0472$ and for $\sigma = 0.9$ it is $1.3525$, while the blind guess is $2.8606$, the answer for a middling level. A sampler built from such a network would treat an almost finished picture as if it were still noisy and smooth away detail that was already there, and the detail is what the small levels decide. The noise level has to reach the network.

## Where the noise level has to enter

Suppose the network sees $\sigma$ as one extra input. A single layer that adds a term computed from $\sigma$ to its pre-activations, $w\,x + \delta(\sigma)$, can shift its output as $\sigma$ changes but keeps one gain $w$ on $x$ for every $\sigma$. The optimal denoiser asks for more than that, even for one cluster. For $x_0 \sim \mathcal{N}(m, \varsigma^2)$ and $x = (1 - \sigma)x_0 + \sigma\varepsilon$ the best guess is $m + k(\sigma)\,(x - (1 - \sigma)m)$ with a gain $k(\sigma) = (1 - \sigma)\varsigma^2/\varrho$, where $\varrho = (1 - \sigma)^2 \varsigma^2 + \sigma^2$ is the variance of $x$. The gain is close to $1$ at a small $\sigma$ (the input is nearly the answer) and close to $0$ at a large one (the input says almost nothing).

<!-- worked: denoiser_architectures.gain_table -->
One cluster, N(3, 0.25^2), and x = (1 - sigma) x_0 + sigma noise. The best guess of x_0 is m + k(sigma) (x - (1 - sigma) m) with the gain k = (1 - sigma) SD^2 / r and r = (1 - sigma)^2 SD^2 + sigma^2, the variance of x; its squared error is the posterior variance SD^2 sigma^2 / r. A layer of the form w x + delta(sigma) (the noise level enters as an added term) keeps one gain w for every sigma; the best w for the six levels together, weighting each by the variance of x, is w = 0.1199. A layer of the form (1 + s(sigma)) x + delta(sigma) can set the gain at every sigma. The last two columns are the squared error above the best possible one, E (guess - best guess)^2 = (k - w)^2 r for the fixed gain, and 0 when the gain may depend on sigma.

| sigma | gain k | variance of x | best possible error | excess, added term | excess, scale and shift |
|---|---:|---:|---:|---:|---:|
| 0.02 | 1.0137 | 0.0604 | 0.0004 | 0.0483 | 0.0000 |
| 0.1 | 0.9278 | 0.0606 | 0.0103 | 0.0396 | 0.0000 |
| 0.3 | 0.3627 | 0.1206 | 0.0466 | 0.0071 | 0.0000 |
| 0.5 | 0.1176 | 0.2656 | 0.0588 | 0.0000 | 0.0000 |
| 0.7 | 0.0378 | 0.4956 | 0.0618 | 0.0033 | 0.0000 |
| 0.9 | 0.0077 | 0.8106 | 0.0625 | 0.0102 | 0.0000 |
<!-- /worked -->

A fixed gain cannot be right at both ends: the best single $w$ is $0.1199$, which is about right at $\sigma = 0.5$ and, at $\sigma = 0.02$, adds an excess of $0.0483$ to a best possible error of $0.0004$, more than a hundred times as much. A layer that multiplies by a factor computed from $\sigma$ has no excess at all, since it can set $k(\sigma)$ at every level. This is the argument for an **adaptive layer normalization**, a normalization layer whose scale and shift, instead of being learned constants, are computed from the conditioning vector $c$: $\mathrm{norm}(x)\,(1 + s(c)) + b(c)$, where $s$ and $b$ are linear functions of $c$ and the $1 +$ makes zero the neutral value. A transformer block that uses it also puts a **gate** $g(c)$ on what each sublayer writes into the residual stream, $x \leftarrow x + g(c) \odot f(\mathrm{norm}(x)\,(1 + s(c)) + b(c))$. With $g = 0$ the block is the identity, which is the way the diffusion transformer of Peebles and Xie starts its blocks (they call it adaLN-zero): a stack of many blocks begins as the identity map of the residual stream of Volume V, and training moves each gate off zero only as far as it helps.

Three designs of conditioning appear in brain's three crates, and they differ in exactly this respect. SDXL's resnets add a projection of the time embedding to every channel, between two convolutions: the form $w\,x + \delta(\sigma)$ above, in each of its 17 resnets (the first of the five conventions that the header of `crates/sdxlunet/src/model.rs` pins: the checkpoint's setting is `default`, a per-channel add before the second group norm, and not the scale-shift variant that the same library also offers, which would be a different model). The remedy for the missing gain is depth: after the add comes a group normalization and an activation, and 17 resnets in a row can compose an effective gain that no single one has. Z-Image applies scale and gate (no shift) in every block, and FLUX.2 applies shift, scale and gate. In Z-Image the vector $c$ that feeds them is projected separately in every block; in FLUX.2 three projections of one vector serve every block of their kind. The second half of the chapter counts what each choice costs.

## Turning a noise fraction into a vector

Every design starts by turning the scalar into a vector of some width, because a linear layer cannot make a rich function of one number. The standard embedding is the sinusoidal encoding of Volume V, Chapter 8, with the timestep in the place of the position. The **timestep embedding** of dimension $d$ and base $B$ (10000 in all three models) is, for $\mathrm{half} = d/2$ and $k = 0, \dots, \mathrm{half} - 1$,

$$
e_k(t) = \cos(t\,\omega_k), \qquad e_{\mathrm{half}+k}(t) = \sin(t\,\omega_k), \qquad \omega_k = B^{-k/\mathrm{half}},
$$

the cosines first and the sines after. In the position chapter the pairs were interleaved; here the two halves are separate blocks, and which block comes first matters, as the tables below show.

<!-- worked: denoiser_architectures.embedding_table -->
The embedding of dimension 8 and base 10000: half = 4 rates w_k = 10000^(-k/4) = 1, 0.1, 0.01, 0.001, and the vector [cos(t w_0), .., cos(t w_3), sin(t w_0), .., sin(t w_3)] (library cos and sin, radians).

| t | cos(t 1) | cos(t 0.1) | cos(t 0.01) | cos(t 0.001) | sin(t 1) | sin(t 0.1) | sin(t 0.01) | sin(t 0.001) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 0.5403 | 0.9950 | 1.0000 | 1.0000 | 0.8415 | 0.0998 | 0.0100 | 0.0010 |
| 2 | -0.4161 | 0.9801 | 0.9998 | 1.0000 | 0.9093 | 0.1987 | 0.0200 | 0.0020 |
| 10 | -0.8391 | 0.5403 | 0.9950 | 1.0000 | -0.5440 | 0.8415 | 0.0998 | 0.0100 |
| 100 | 0.8623 | -0.8391 | 0.5403 | 0.9950 | -0.5064 | -0.5440 | 0.8415 | 0.0998 |
| 1, halves exchanged | 0.8415 | 0.0998 | 0.0100 | 0.0010 | 0.5403 | 0.9950 | 1.0000 | 1.0000 |
<!-- /worked -->

The first coordinate pair turns once every $2\pi \approx 6.3$ steps of $t$, the second every $63$, the third every $628$, the fourth every $6283$: a long ladder of rates, so that two timesteps that are close are close in the slow pairs and told apart by the fast ones. What the fast pair needs is a range of $t$ in which it turns. The noise fraction lives in $[0, 1]$, and an angle between $0$ and $1$ radian does not tell $\sigma = 0.50$ from $0.51$ in any coordinate.

<!-- worked: denoiser_architectures.scale_table -->
Dimension 256, base 10000. The Euclidean distance between the embeddings of two noise fractions, when the embedding is fed the fraction itself (scale 1) and when it is fed 1000 times the fraction.

| sigma and sigma' | distance, scale 1 | distance, scale 1000 |
|---|---:|---:|
| 0.5 and 0.51 | 0.0273 | 9.1149 |
| 0.5 and 0.501 | 0.0027 | 2.6712 |
| 0.1 and 0.11 | 0.0273 | 9.1149 |

The inner product of two embeddings of dimension 256 with the cosine half first and with the halves exchanged, for arguments a and b (the arguments here are timesteps, already scaled).

| a, b | b - a | cos first | sin first |
|---|---:|---:|---:|
| 200, 500 | 300 | 42.244914 | 42.244914 |
| 600, 900 | 300 | 42.244914 | 42.244914 |
| 0, 300 | 300 | 42.244914 | 42.244914 |

The two orders give inner products that differ by less than 1e-12. Pairs with the same difference b - a = 300 give inner products that differ by less than 1e-09.
<!-- /worked -->

So the models do not hand the embedding the fraction. SDXL's clock is the scheduler's timestep, a number from $0$ to $999$ on the 1000-step chain it was trained on. Z-Image's and FLUX.2's transformers receive a time value in $[0, 1]$ (for Z-Image $1 - \sigma$) multiplied by 1000, the `t_scale` field of `crates/s3dit/src/model.rs` and the `t * 1000.0` of `crates/flux2/src/model.rs`; with the scale, two noise fractions $0.01$ apart are $9.1149$ apart in the embedding, against $0.0273$ without it. The second table above shows a property that explains the next paragraph. Exchanging the halves of the embedding is a permutation of the coordinates, so every inner product between two embeddings is unchanged, to less than $10^{-12}$, and the inner product of two embeddings depends only on the difference of their arguments. No statistic computed from the embeddings alone can tell the two orders apart.

That makes the order a trap. What follows the embedding is a learned layer, whose weight columns are tied to coordinates: a checkpoint trained against $[\cos \,\Vert\, \sin]$ and run against $[\sin \,\Vert\, \cos]$ feeds every column the wrong half. The result does not crash and is still a smooth, plausible vector, and commit `ca12c1f91` records what that looked like in Z-Image: text-to-image output that was clean at $256 \times 256$, had a coherent centre with speckle around it at $512 \times 512$ and was uniform speckle at $768 \times 768$, on the same weights, prompt and seed, in both precisions. The existing parity test ran at a 64-token toy scale where the error was too small to fail its cosine gate; the corruption showed up at the first main block of the 512-pixel run (cosine $0.80$ against $0.997$ for the refiners before it) and was fixed by calling the shared sinusoid with the cosine half first, $[\cos \,\Vert\, \sin]$, which is the order of the reference implementation's embedder and the one FLUX.1 and FLUX.2 already used (the faulty call had the sine half first), with a regression test at the failing scale. The shared `model::hostmath::timestep_embedding` takes the order as an argument, and its documentation, extended by the same commit, warns that a sibling model's value is a fact about that model and not a default.

The embedding is the input of a small network, two linear layers with an activation between them, that turns it into the conditioning vector $c$. The shapes differ: SDXL's goes $320 \to 1280 \to 1280$ (the sinusoid's width is the width of the first level), Z-Image's $256 \to 1024 \to 256$ (`t_embedder.mlp` in `crates/s3dit/src/import.rs`; the second width is $\min(\mathrm{dim}, 256)$) and FLUX.2's $256 \to d \to d$. A third input joins SDXL's: the pooled text vector (1280 numbers) concatenated with six size values (the original size, the crop corner and the target size), each given its own 256-wide sinusoid, which makes $1280 + 6 \times 256 = 2816$ numbers, goes through a second two-layer network to the same width of 1280 and is added to the first one's output. The pooled text goes first in that concatenation, and `crates/sdxlunet/src/hostemb.rs` calls the reverse order the classic mistake: it type-checks, trains nothing and silently changes the framing of the image.

## The U-Net, read as a shape

Volume VII's segmenter was a U-Net of one level: convolve, halve, convolve, double, join the skip, convolve. The image generator's U-Net is the same shape with three changes. Its output has the shape of its input (a $4 \times 128 \times 128$ prediction for a $4 \times 128 \times 128$ latent), so there is no class count or depth at the end, only $4$ channels. The levels are many and wide: widths of $320$, $640$ and $1280$ channels at resolutions of $128$, $64$ and $32$. And every block can read the clock and, at the coarser levels, the prompt. The first table lists what the SDXL U-Net does at a $128 \times 128$ latent, walking down, across the middle and up.

<!-- worked: denoiser_architectures.unet_walk -->
The SDXL U-Net (320, 640, 1280 channels; resnets per down block 2; the configuration lists transformer blocks per level as 1, 2, 10 but attention is on the two coarser levels only, so the 1 of the first level is unused: 2 blocks per spatial transformer at 64 by 64, 10 at 32 by 32 and in the middle block) on the 128 by 128 latent of a 1024 by 1024 image. A row is a stage: its resolution, channels in and out, resnets, spatial-transformer blocks, tokens for the self-attention of those blocks, and what it does to the stack of skips. The skip stack is pushed in the order conv_in, down 0 (two resnets, then its stride-2 convolution), down 1 (the same), down 2 (two resnets, no convolution): nine entries of widths 320, 320, 320, 320, 640, 640, 640, 1280, 1280.

| stage | resolution | channels | resnets | transformer blocks | tokens T | skips |
|---|---|---|---:|---:|---:|---|
| conv_in | 128x128 | 4 -> 320 | 0 | 0 | - | push 1 |
| down 0 | 128x128 | 320 -> 320 | 2 | 0 | - | push 3 |
| down 1 | 64x64 | 320 -> 640 | 2 | 4 | 4096 | push 3 |
| down 2 | 32x32 | 640 -> 1280 | 2 | 20 | 1024 | push 2 |
| middle | 32x32 | 1280 -> 1280 | 2 | 10 | 1024 | - |
| up 0 | 32x32 | 1280 -> 1280 | 3 | 30 | 1024 | pop 3 (resnet inputs 2560, 2560, 1920) |
| up 1 | 64x64 | 1280 -> 640 | 3 | 6 | 4096 | pop 3 (resnet inputs 1920, 1280, 960) |
| up 2 | 128x128 | 640 -> 320 | 3 | 0 | - | pop 3 (resnet inputs 960, 640, 640) |
| conv_out | 128x128 | 320 -> 4 | 0 | 0 | - | - |
<!-- /worked -->

![The skeleton of SDXL's U-Net on a 128 by 128 latent: three levels down with 320, 640 and 1280 channels, a middle block, three levels up, stride-2 convolutions between levels on the way down and doubling with a convolution on the way up, and three horizontal arrows for the skips that cross at each resolution, each carrying three feature maps whose widths are listed.](img/unet-skeleton.svg)

The bookkeeping of the skips is the part of a U-Net that is easy to get wrong while every shape still matches. The down path pushes a feature map on a stack after the input convolution, after every resnet and after every stride-2 convolution, which for two resnets per level and three levels is $1 + 3 \times 2 + 2 = 9$ entries, of widths $320, 320, 320, 320, 640, 640, 640, 1280, 1280$. The up path has one resnet more per level than the down path has, and each one pops one entry and concatenates it to the hidden state before it convolves, so it consumes three entries a level, nine in all, and the stack is empty at the end. The inputs of those resnets are in the last column of the table. At the first up level the hidden state is $1280$ wide and the skips are $1280$, $1280$ and $640$ wide, so the three resnets read $2560$, $2560$ and $1920$ channels. A rule applied to the wrong operand is invisible there, because the hidden width and the level's own width are both $1280$; commit `c7d97a708`, which introduced the crate, says that an inverted choice of input width for the up-path resnets was caught at the second up level (where the two are $1280$ and $640$) by the importer's shape check, and `crates/sdxlunet/src/config.rs` carries a test, `sdxl_up_resnet_input_widths_match_the_checkpoint`, with the nine widths written down. The skip stack is also where Chapter 11 will inject a control signal: the network can be built to take one extra input per entry of the stack and one for the middle block, $9 + 1 = 10$ in all, added to the skip copies and not to the running hidden state (adding it to both would count it twice, with every shape still correct).

Inside the table's rows there are two kinds of block. A resnet of the U-Net is the residual block of Volume III with the clock added in: group normalization, activation, a $3 \times 3$ convolution, a per-channel addition of the projected time embedding, group normalization, activation, a second $3 \times 3$ convolution, and the sum with the input (through a $1 \times 1$ convolution when the widths differ). The group norm uses 32 groups and an $\varepsilon$ of $10^{-5}$. At an attention level, each resnet is followed by a **spatial transformer**: a group norm (with $\varepsilon = 10^{-6}$, a second constant in the same graph, as the model's header notes), a flattening of the $H \times W$ map into $T = HW$ tokens, a linear layer, one or more transformer blocks, a linear layer back and a sum with the input. Each block has three sublayers in the pre-norm arrangement of Volume V: self-attention among the $T$ tokens, cross-attention to the $77$ text tokens (the queries come from the picture tokens and the keys and values from the prompt, whose $2048$-wide vectors are projected by weights of the block, so that the picture reads the prompt and the prompt is never rewritten), and a feed-forward network gated by the GEGLU rule, $\mathrm{hidden} \odot \mathrm{gelu}(\mathrm{gate})$. The transformer counts per level are $1$, $2$ and $10$ and are the largest lever on the size of the network.

<!-- worked: denoiser_architectures.unet_params -->
The SDXL U-Net: 3 levels, resnets per down block 2, per up block 3, 4 latent channels in and 4 out, 32 groups in every group normalisation, time dimension 1280, cross-attention (text) dimension 2048. The middle block is a resnet, a spatial transformer of 10 blocks at 1280 channels with 20 heads, and a resnet. Per level:

| level | resolution | channels | spatial transformers | blocks per transformer | heads | head width |
|---|---|---:|---|---:|---:|---:|
| 0 | 1/1 | 320 | no | - | - | - |
| 1 | 1/2 | 640 | yes | 2 | 10 | 64 |
| 2 | 1/4 | 1280 | yes | 10 | 20 | 64 |

The conditioning vector comes from two networks, each a linear layer, an activation and a second linear layer, all with biases: the timestep network 320 -> 1280 -> 1280 (the sinusoid is 320 wide, the width of the first level) and the added-conditioning network 2816 -> 1280 -> 1280, whose input is the pooled text vector followed by six size values of 256 numbers each (1280 + 6 x 256 = 2816); their outputs are added.

Counting rules. A convolution from ci to co channels with a k by k kernel has co ci k^2 weights and co biases. A group normalisation of c channels has a gain and a shift (2c). A linear layer from i to o numbers has i o weights and, where stated, o biases. A resnet from ci to co channels is: group normalisation of ci, 3 x 3 convolution ci -> co, a linear time projection (time dimension -> co, with bias) added to every pixel, group normalisation of co, 3 x 3 convolution co -> co, and a 1 x 1 shortcut convolution ci -> co only when ci differs from co. A spatial transformer at c channels is: group normalisation, a linear c -> c with bias (proj_in), then for each block a layer normalisation (2c), self-attention (query, key and value c -> c each without bias, 3c^2 in all; output c -> c with bias), a layer normalisation, cross-attention (query c -> c without bias, key and value text dimension -> c each without bias; output c -> c with bias), a layer normalisation, and the feed-forward (a gated projection c -> 4c computed twice, a gate and a value, each a weight and a bias, which is a single c -> 8c layer in the original checkpoint layout; the product of the activated gate and the value goes through 4c -> c with bias), and after the last block a linear c -> c with bias (proj_out). Between levels the down path has a stride-2 3 x 3 convolution at the channels of the level it leaves; the up path doubles the grid by nearest neighbour and applies a 3 x 3 convolution at the channels of the level it enters. The up resnets read the hidden state concatenated with the popped skip. The output is a group normalisation of the first level's channels, then a 3 x 3 convolution to the output channels. A tensor is one weight or one bias (a normalisation's gain and shift are two).

Every parameter of the SDXL U-Net derived from that configuration, by what the weights do (a convolution's bias, a normalisation's gain and shift and a linear layer's bias are counted with their layer).

| kind | parameters | share |
|---|---:|---:|
| convolutions | 333,272,004 | 13.0% |
| time projections | 17,626,560 | 0.7% |
| norms | 588,160 | 0.0% |
| self-attention | 409,683,200 | 16.0% |
| cross-attention | 545,670,400 | 21.3% |
| feed-forward | 1,229,548,800 | 47.9% |
| token projections | 23,778,560 | 0.9% |
| time and size embeddings | 7,296,000 | 0.3% |
| total | 2,567,463,684 | 100.0% |

The network has 17 resnets and 70 spatial-transformer blocks, and 2,567,463,684 parameters: 10.27 GB in 32-bit floats and 5.13 GB in 16-bit floats. Counting one tensor for each weight or bias, with a fused query-key-value matrix and a fused key-value matrix of the cross-attention each as one tensor and the feed-forward's gated projection as two (the way brain stores them), the network has 1610 tensors; with the three separate attention matrices and the one gated projection of the original layout it has 1680.

The same walk, with the same rules, on the variant that brain builds for its smoke tests gives 1,322,852 parameters in 284 tensors. Its configuration:

The smoke-test U-Net: 2 levels, resnets per down block 1, per up block 2, 4 latent channels in and 4 out, 8 groups in every group normalisation, time dimension 128, cross-attention (text) dimension 24. The middle block is a resnet, a spatial transformer of 2 blocks at 64 channels with 4 heads, and a resnet. Per level:

| level | resolution | channels | spatial transformers | blocks per transformer | heads | head width |
|---|---|---:|---|---:|---:|---:|
| 0 | 1/1 | 32 | no | - | - | - |
| 1 | 1/2 | 64 | yes | 2 | 4 | 16 |

The conditioning vector comes from two networks, each a linear layer, an activation and a second linear layer, all with biases: the timestep network 32 -> 128 -> 128 (the sinusoid is 32 wide, the width of the first level) and the added-conditioning network 104 -> 128 -> 128, whose input is the pooled text vector followed by six size values of 16 numbers each (8 + 6 x 16 = 104); their outputs are added.

and its walk on a 16 by 16 latent:

| stage | resolution | channels | resnets | transformer blocks | tokens T | skips |
|---|---|---|---:|---:|---:|---|
| conv_in | 16x16 | 4 -> 32 | 0 | 0 | - | push 1 |
| down 0 | 16x16 | 32 -> 32 | 1 | 0 | - | push 2 |
| down 1 | 8x8 | 32 -> 64 | 1 | 2 | 64 | push 1 |
| middle | 8x8 | 64 -> 64 | 2 | 2 | 64 | - |
| up 0 | 8x8 | 64 -> 64 | 2 | 4 | 64 | pop 2 (resnet inputs 128, 96) |
| up 1 | 16x16 | 64 -> 32 | 2 | 0 | - | pop 2 (resnet inputs 96, 64) |
| conv_out | 16x16 | 32 -> 4 | 0 | 0 | - | - |
<!-- /worked -->

Two things in the table are surprising for a network drawn as a convolutional U. Only $13.0\%$ of the weights are convolutions: the U-Net is a transformer by weight, mostly feed-forward ($47.9\%$) and attention ($16.0\% + 21.3\%$), with a thin convolutional frame around it. And the clock is nearly free in parameters: the two embedding networks and the 17 per-resnet projections together are $1.0\%$ of the total. The total itself is the figure that commit `c7d97a708` states as a measured residency, $2\,567\,463\,684$ parameters; the derivation from the configuration alone reproduces it exactly, and reproduces the commit's other count too, $1610$ tensors in brain's layout against $1680$ in the original one, the difference being the $70$ transformer blocks' three host-side fusions and splits. A configuration trap sits in the same file, and its header records it. The checkpoint's `attention_head_dim` field is `[5, 10, 20]`, and the library that defined it reads it as the number of heads, not the width of a head; read literally it gives 5 heads of width 5, a forward pass that runs, produces the right shapes and is wrong everywhere. The real head width is the level's channels over its heads, $64$ at every level.

<!-- worked: denoiser_architectures.unet_flops -->
One forward pass of the SDXL U-Net on the 128 by 128 grid of a 1024 by 1024 image with 77 text tokens, in GFLOP (a multiply-add is two operations; additions, normalisations, softmax and activations are not counted), by kind and by the resolution at which it is spent. Every layer of the counting rules above is counted, including the time projections of the resnets and the two conditioning networks (a matrix applied to a single vector, 50 million floating-point operations together). A convolution costs 2 ci co k^2 per output pixel, a linear layer 2 i o per token; the cross-attention key and value projections are applied to the 77 text tokens, 2 x 77 x 2048 x c each for a block of c channels; the scores q k^T and the weighted sum of the values cost 2 T^2 c each, 4 T^2 c together (T the tokens of the grid), and in cross-attention 4 T x 77 x c.

| kind | GFLOP | share |
|---|---:|---:|
| convolutions | 1,623.1 | 24.0% |
| time projections | 0.0 | 0.0% |
| self-attention | 939.5 | 13.9% |
| cross-attention | 522.2 | 7.7% |
| feed-forward | 2,818.6 | 41.7% |
| token projections | 73.8 | 1.1% |
| time and size embeddings | 0.0 | 0.0% |
| self-attention scores | 751.6 | 11.1% |
| cross-attention scores | 32.3 | 0.5% |
| total | 6,761.2 | 100.0% |

| resolution | GFLOP | share |
|---|---:|---:|
| 128x128 | 567.8 | 8.4% |
| 64x64 | 1,623.6 | 24.0% |
| 32x32 | 4,569.8 | 67.6% |
| - | 0.0 | 0.0% |

The score matrix that a self-attention layer would build at each resolution (heads x T x T entries, 4 bytes each, if it were materialised), and whether the network has attention there.

| resolution | channels | heads | tokens T | score bytes (32-bit) | attention |
|---|---:|---:|---:|---:|---:|
| 128x128 | 320 | 5 | 16,384 | 5,368,709,120 | no |
| 64x64 | 640 | 10 | 4,096 | 671,088,640 | yes |
| 32x32 | 1280 | 20 | 1,024 | 83,886,080 | yes |
<!-- /worked -->

The cost shows why attention sits only where it does. A convolution costs in proportion to the number of positions, so a level that is four times as large costs four times as much per channel pair; self-attention costs in proportion to $T^2$, sixteen times as much. The $128 \times 128$ level has $16384$ tokens, and a self-attention layer there, with the level's $5$ heads, would build $5 \cdot 16384^2$ scores, $5.4$ GB in 32-bit floats, for one layer; the level has none, and `crates/sdxlunet/src/model.rs` sizes its materialised attention buffers over the levels that have attention only, because including the finest one over-allocates eightfold (its comment counts $10.7$ GB of never-bound buffers, scores and probabilities together, at a $128 \times 128$ latent). At the $64 \times 64$ level one layer's scores are $671$ MB, a number the model's own header gives for the native $1024^2$ size, and at $32 \times 32$ only $84$ MB. So most of the arithmetic ends up at the coarsest level ($67.6\%$ of the pass is spent at $32 \times 32$) where it is cheap in memory and the wide $1280$-channel layers can work on a small grid. The score terms themselves, $11.1\%$ of the pass at this size, are the part that grows fastest with the image side; the last section follows them.

One more property of this graph was learned the hard way. The convolutional half of the U-Net was recorded in brain's differentiable block builder from the start; the transformer half was added by pushing its steps onto the forward list directly, an escape hatch that records nothing on the reverse-mode tape. The forward matched its reference in all 165 stage comparisons, and the network was not differentiable: every parameter upstream of a pushed step received a zero gradient, silently. Commit `7537e04ca` (knowledge #55) moved those operations onto real recorders in the shared builder, and made two changes that train mode needs and that run fine and are wrong if forgotten: a flash-attention kernel never stores the softmax that the adjoint reads, so a recording builder takes the materialised path however capable the device, and each attention site needs its own slab of scores, since two sites that share one would differentiate against each other's softmax. The lesson is not specific to U-Nets. A graph is correct when its forward matches and its backward does, and a finite-difference check gates a backward against whatever forward was emitted.

## Cutting the latent into tokens

The transformer needs a sequence. A latent is a grid of $C$ channels, and Volume VII's vision transformer showed the cut: split the grid into $p \times p$ patches and treat each as a token whose features are its $p^2 C$ values. Z-Image's latent has $16$ channels and its patches are $2 \times 2$, so a token is $64$ numbers, one linear layer takes it to the model width of $3840$, and a $128 \times 128$ latent (a $1024 \times 1024$ image through the VAE of Volume VIII, Chapter 2, which reduces each side eight-fold) becomes $64 \times 64 = 4096$ tokens. The order inside a patch is channel-innermost, $[p_h, p_w, C]$, and tokens are numbered row by row; the two functions in `crates/dit/src/patchify.rs` are inverses of each other (their tests say so, with and without a temporal patch), and the last layer of the network ends with the inverse: a linear layer from the model width back to $64$ values per token and an unpatchify. FLUX.2 does the same with $32$ channels, $128$ numbers a token, and a $16 \times 16$ pixel patch in both. A token is therefore 256 pixels' worth of picture, and this single number, the patch's side, decides how long the sequence is.

With the cut, the picture has lost its grid, and the transformer needs the positions back. This is the multi-axis rotary embedding of Volume V, Chapter 8. Z-Image gives every token an id triple and divides a head of $128$ channels into three axes of $32$, $48$ and $48$, so that pair $k$ of an axis of width $d$ turns by (the token's id on the axis) $\times\, 256^{-2k/d}$.

<!-- worked: denoiser_architectures.token_ids -->
A caption of 2 tokens and an image latent of 4 rows by 6 columns in 2 by 2 patches, which is a grid of 1 frame, 2 rows and 3 columns of patches: the position triples (axis 0, axis 1, axis 2) of the joint sequence, image first.

| index | token | ids |
|---|---|---:|
| 0 | image patch | (3, 0, 0) |
| 1 | image patch | (3, 0, 1) |
| 2 | image patch | (3, 0, 2) |
| 3 | image patch | (3, 1, 0) |
| 4 | image patch | (3, 1, 1) |
| 5 | image patch | (3, 1, 2) |
| 6 | caption token | (1, 0, 0) |
| 7 | caption token | (2, 0, 0) |
<!-- /worked -->

The first axis counts the sequence: a caption token $i$ is at $(1 + i, 0, 0)$, and the image comes after the caption on that axis, at frame $f$ with the id $L + 1 + f$ for a caption of $L$ tokens, so a picture is a constant on the first axis and a grid on the other two. The scheme differs from the one counter of Volume V's example (there the picture advanced the counter by its largest extent); here the text positions are lines and the picture is a plane at a fixed offset, which keeps the image's own row and column offsets independent of how long the caption is.

<!-- worked: denoiser_architectures.rope_axes -->
The three axes of Z-Image's rotary position in a head of width 128: pair k of an axis of width d turns by (id on the axis) times 256^(-2k/d). The fastest rate of every axis is 1; the wavelength is 2 pi over the slowest rate.

| axis | width d | pairs | fastest rate | slowest rate | slowest wavelength | table length |
|---|---:|---:|---:|---:|---:|---:|
| sequence and frame | 32 | 16 | 1 | 0.00552 | 1137 | 1536 |
| patch row | 48 | 24 | 1 | 0.00492 | 1277 | 512 |
| patch column | 48 | 24 | 1 | 0.00492 | 1277 | 512 |
<!-- /worked -->

Each axis has its own scale, starting at rate $1$, as in the Volume V example's remedy, and the slowest wavelength of the row and column axes (about $1277$) is more than twice their table length of $512$, so no pair wraps around within the table. The table length of the first axis is $1536$, the number of ids that a caption plus its frames may use; commit `ca12c1f91` changed it from $1024$ to $1536$ to match the shipped configuration, and says that the change was functionally inert, since no tested resolution came near either bound.

A **diffusion transformer** is then simply this: the patch tokens, a stack of transformer blocks whose normalizations are adaptive, and the inverse cut at the end. Here is the block of Z-Image, as brain implements it (`crates/s3dit/src/block.rs`), written in the book's pseudocode with the routines it uses defined first.

In the routine `x` holds `N` token rows of width `D`, `H` is the number of heads and `HD = D // H` the width of a head, `eps` is the epsilon of the RMS norms, `cvec` is the conditioning vector of width `C` (256 for Z-Image), and `prm` is a record of the block's tensors: `Wmod: [4 * D, C]` with `bmod: [4 * D]`, the matrices `Wq`, `Wk`, `Wv`, `Wo`, `W1`, `W2`, `W3` stored as `[out, in]`, the gains `an1`, `an2`, `fn1`, `fn2` of width `D` and `nq`, `nk` of width `HD`. The tables `cos_t` and `sin_t` have one row per token and one column per pair of a head. A scalar combined with an array (`1 + v`, `scores / sqrt(HD)`) acts on every element.

```text
function rotate_pairs(v: array<real>[2 * P], c: array<real>[P], s: array<real>[P]) -> array<real>[2 * P]:
    out = zeros(real, [2 * P])
    for j in 0 .. P:
        out[2 * j] = v[2 * j] * c[j] - v[2 * j + 1] * s[j]
        out[2 * j + 1] = v[2 * j] * s[j] + v[2 * j + 1] * c[j]
    return out

function rms_rows(x: array<real>[R, D], gamma: array<real>[D], eps: real) -> array<real>[R, D]:
    (y, _, _) = rms_norm(x, gamma, eps)        # rms_norm of the pseudocode reference returns (y, xhat, inv_std); only y is wanted
    return y

function dit_block(x: array<real>[N, D], cvec: array<real>[C], prm, cos_t: array<real>[N, HD // 2],
                   sin_t: array<real>[N, HD // 2], H: int, eps: real) -> array<real>[N, D]:
    require D mod H == 0 and (D // H) mod 2 == 0, "heads of an even width, so that channels pair up"
    HD = D // H
    m = matmul(prm.Wmod, cvec) + prm.bmod
    s_a = m[0 : D]
    g_a = map(tanh, m[D : 2 * D])
    s_m = m[2 * D : 3 * D]
    g_m = map(tanh, m[3 * D : 4 * D])
    n1 = rms_rows(x, prm.an1 * (1 + s_a), eps)
    q = matmul(n1, transpose(prm.Wq))
    k = matmul(n1, transpose(prm.Wk))
    v = matmul(n1, transpose(prm.Wv))
    ctx = zeros(real, [N, D])
    for h in 0 .. H:
        c0 = h * HD
        qh = rms_rows(q[:, c0 : c0 + HD], prm.nq, eps)
        kh = rms_rows(k[:, c0 : c0 + HD], prm.nk, eps)
        for i in 0 .. N:
            qh[i, :] = rotate_pairs(qh[i, :], cos_t[i, :], sin_t[i, :])
            kh[i, :] = rotate_pairs(kh[i, :], cos_t[i, :], sin_t[i, :])
        scores = matmul(qh, transpose(kh)) / sqrt(HD)
        weights = zeros(real, [N, N])
        for i in 0 .. N:
            weights[i, :] = softmax(scores[i, :])
        ctx[:, c0 : c0 + HD] = matmul(weights, v[:, c0 : c0 + HD])
    a = matmul(ctx, transpose(prm.Wo))
    x1 = x + rms_rows(a, prm.an2 * g_a, eps)
    f1 = rms_rows(x1, prm.fn1 * (1 + s_m), eps)
    hid = map(silu, matmul(f1, transpose(prm.W1))) * matmul(f1, transpose(prm.W3))
    return x1 + rms_rows(matmul(hid, transpose(prm.W2)), prm.fn2 * g_m, eps)
```

The routine is the pre-norm block of Volume V with four additions, each visible in the figure that follows. The conditioning vector is projected to four vectors of width $D$ (the scale and the gate of the attention sublayer, the scale and the gate of the feed-forward). The norms come in pairs, one before each sublayer and one after, so that what a sublayer writes to the stream is normalized to unit mean square before the gate sets its size; the gate is $\tanh$ of the projection, so it lies in $(-1, 1)$ and is $0$ when the projection is $0$, which is what makes the zero-gate block an exact identity. The queries and the keys are normalized per head before the rotation, with a learned gain of $\mathrm{HD}$ values each. And there is no shift: Z-Image modulates by scale and gate only.

![One Z-Image block: the token rows pass through an RMS norm whose gain is multiplied by one plus a scale, the attention with query-key norm and rotary rotation, an RMS norm whose gain is multiplied by tanh of a gate, and are added to the input; the same pattern for the feed-forward. A column on the right takes the timestep vector through one linear layer and splits it into the scale and gate for each of the four norms.](img/modulated-block.svg)

The scale and the gate act on the gains of the norms, and that is how brain uses them. Since $\mathrm{rms\_norm}(x, \gamma) = \hat x \odot \gamma$ for the normalized row $\hat x$, multiplying the output by $(1 + s)$ or by $\tanh g$ channel by channel is the same as multiplying the gain $\gamma$ by it before the norm. So the host computes the four modified gains once per block and per pass from the timestep vector (`fold_adaln` in `crates/s3dit/src/block.rs`) and the device runs the plain norm kernel with the new gains: no kernel for scale or gate is needed at all. The check on the small denoiser below confirms that the folded and the unfolded forms agree to rounding.

<!-- worked: denoiser_architectures.zimage_params -->
Z-Image Turbo from its configuration: width 3840, 30 heads (head width 128), feed-forward width 3840*8//3 = 10240, conditioning width min(3840, 256) = 256, 30 main blocks and 2 blocks in each of the noise and context refiners, 16 latent channels in 2 by 2 patches (64 values per token), caption features of width 2560. Outside the blocks: the caption embedder (a gain of 2560, a 2560 -> 3840 matrix and its bias), two learned pad tokens of width 3840, one for the image stream and one for the caption stream (2 x 3840 parameters), the patch embedder (64 -> 3840 with bias), the timestep network (256 -> 1024 -> 256, both with biases) and the final layer (a 256 -> 3840 modulation with bias, then 3840 -> 64 with bias).

| inside one block | parameters |
|---|---:|
| attention matrices | 58,982,400 |
| query and key gains | 256 |
| feed-forward matrices | 117,964,800 |
| norm gains | 15,360 |
| modulation projection | 3,947,520 |
| one block with modulation | 180,910,336 |
| one block without (context refiner) | 176,962,816 |

| part of the network | parameters |
|---|---:|
| 30 main blocks | 5,427,310,080 |
| 2 noise-refiner blocks | 361,820,672 |
| 2 context-refiner blocks | 353,925,632 |
| caption embedder | 9,836,800 |
| pad tokens | 7,680 |
| patch embedder | 249,600 |
| timestep MLP | 525,568 |
| final layer | 1,232,704 |
| total | 6,154,908,736 |

The 32 modulated blocks spend 126,320,640 parameters (2.1% of the network) on turning the 256-wide timestep vector into scales and gates; the tensors are 521 in all (15 outside the blocks, 15 in a modulated block, 13 in a context-refiner block).
<!-- /worked -->

The block's weights are $4D^2$ in the attention and $3 \times D \times (8D/3) = 8D^2$ in the feed-forward, $12D^2 = 176{,}947{,}200$, as in Recall question 3; norms, query-key gains and the modulation projection add $3{,}963{,}136$ more, and that is the $180{,}910{,}336$ of a modulated block. The modulation projection is the only part that depends on the clock, $2.1\%$ of the network; a pass uses each of its $126{,}320{,}640$ weights once, to make four vectors from a 256-wide input. The total is $6{,}154{,}908{,}736$ parameters, the figure that the importer's documentation in `crates/s3dit/src/import.rs` states for the released checkpoint, and the $521$ tensors are the checkpoint's own (two of which, the pad tokens used only to batch sequences of different lengths, brain never reads and keeps in the manifest so that two import routes agree).

Two details of the table need explaining. The 30 main blocks act on the joint sequence, but two blocks per stream come first. The noise refiner is two modulated blocks applied to the image tokens alone, before they meet the caption; the context refiner is two blocks applied to the caption tokens alone and has no modulation, because the caption carries no timestep and so has nothing for a clock to modulate (`import.rs` says so in the comment of its block builder). And the clock enters the model once more at the end: a final layer applies a parameter-free layer norm to every token, multiplies by one plus a scale computed from the SiLU of the same vector, and projects to the $64$ values of a patch, of which the image tokens are kept.

## One sequence for the picture and the prompt

Z-Image's main blocks see the caption and the picture in one sequence of $4096 + L$ tokens, image first, and the attention of every token spans all of them. That is **joint attention**: one softmax per query over the keys of both kinds. The U-Net's text path is different in two respects, which the next table shows on five tokens, three of the picture and two of a caption.

<!-- worked: denoiser_architectures.joint_attention -->
Generator: the book's SplitMix64, seed 0x0ba4c666442e67bf (`seed_of` of the name `denoiser_architectures.joint`). Draws, in this order: the 20 entries of X (5 tokens of 4 numbers, image tokens first), each `round(normal(), 1)`; then W_Q, W_K, W_V (4 by 4 each, row by row), each `round(0.5 * normal(), 1)` (`round` to one decimal, ties away from zero). One head of width 4, scores divided by 2, no mask, no position information.

| token | x (4 numbers) |
|---|---|
| image 0 | (1.2, -1.3, -1.5, 1.7) |
| image 1 | (-1.7, 1, -0.4, 1.2) |
| image 2 | (1.2, -2.3, -1.8, -1.1) |
| caption 0 | (-0.5, -1.9, 0.5, 0.6) |
| caption 1 | (-0.2, -2.1, 0.8, 0.6) |

| W_Q | W_K | W_V |
|---|---|---|
| (-1.2, 0.8, 0, -0.1) | (0.1, -0.1, 0.2, 0.5) | (-0.3, 0.2, -0.8, 0) |
| (-0.1, 0.6, 0, -0.5) | (0.4, 0.4, -0.6, 0.1) | (-0.3, 0, 0.1, -0.4) |
| (1.1, 0, 0.6, -0.5) | (-0.3, -0.1, -0.8, -0.3) | (0, 0.5, -0.9, 0.9) |
| (0.3, 0, 0.2, -0.4) | (-0.2, 0.4, -0.2, 0.3) | (-0.1, -0.7, 0.5, 0.4) |

The attention weights of each token's query. Joint: one softmax over all five keys (image keys 0 to 2, caption keys 3 and 4). Cross: the image queries only, one softmax over the two caption keys; the caption tokens are not updated.

| token | joint weights over keys 0 .. 4 | mass on image keys | mass on caption keys | cross weights over keys 3, 4 |
|---|---|---|---|---|
| image 0 | 0.147 0.127 0.050 0.307 0.370 | 0.324 | 0.676 | 0.453 0.547 |
| image 1 | 0.130 0.224 0.373 0.143 0.130 | 0.727 | 0.273 | 0.522 0.478 |
| image 2 | 0.219 0.071 0.028 0.280 0.402 | 0.319 | 0.681 | 0.410 0.590 |
| caption 0 | 0.160 0.040 0.658 0.071 0.071 | 0.858 | 0.142 | (not updated) |
| caption 1 | 0.155 0.034 0.685 0.063 0.062 | 0.874 | 0.126 | (not updated) |

Add 0.5 to all four entries of image token 0 and look at the outputs of the two caption tokens: they move by at most 0.1099 under joint attention and by 0.0000 under cross-attention. Add 0.5 to caption token 0 and look at the image tokens: they move by at most 0.2991 under joint attention and by 0.2862 under cross-attention.
<!-- /worked -->

In the cross-attention design an image token's weights are a softmax over the two caption keys only, so its weight on the prompt is $1$ by construction and the caption tokens are never rewritten: perturb the picture and the caption outputs move by $0.0000$. In the joint design the weights of image token 0 are $0.307 + 0.370 = 0.676$ on the caption and $0.324$ on the picture, a split that depends on the content, and the caption tokens are queries too: they put $0.858$ and $0.874$ of their weight on the image tokens and so change when the picture does ($0.1099$ for the same perturbation). The prompt, in this design, is not a fixed instruction that the picture reads but one half of a conversation: at every one of the 30 blocks the caption tokens are rewritten in the light of the current picture, and the picture in the light of the caption.

The price is the sequence. A caption slot of $512$ tokens makes the sequence $4608$ instead of $4096$ at $1024 \times 1024$, and attention costs in proportion to its square. FLUX.2 pays the same for the same reason and differs in how it spends its weights.

<!-- worked: denoiser_architectures.flux2_table -->
FLUX.2 from the configurations of its two sizes (hidden width d, SwiGLU width 3d, 128 latent channels per token, caption features of width 7680 and 12288). The total also holds what is outside the blocks, all of it without biases: the patch embedder (128 -> d), the caption embedder (caption width -> d), the timestep network (256 -> d -> d) and the final projection (d -> 128). A double-stream block holds two sets of attention, feed-forward and query-key gains, one for the image stream and one for the caption stream; a single-stream block holds one set with the attention and the feed-forward fused into two matrices. The shared modulation projections (6 d^2 for each double-stream stream, 3 d^2 for the single blocks, 2 d^2 for the final layer) serve every block of their kind.

| variant | d | double + single | double blocks | single blocks | modulation | total | tensors |
|---|---:|---:|---:|---:|---:|---:|---:|
| klein-4b | 3072 | 5 + 20 | 1,226,836,480 | 2,453,672,960 | 160,432,128 | 3,875,544,576 | 149 |
| klein-9b | 4096 | 8 + 24 | 3,489,665,024 | 5,234,497,536 | 285,212,672 | 9,078,581,248 | 201 |
<!-- /worked -->

The first blocks of FLUX.2 are double-stream: the image tokens and the caption tokens each have their own query, key, value, output and feed-forward weights, and their own query-key gains, but their queries and keys meet in one attention, over the concatenated sequence. This is the design that the source code of `crates/flux2/src/model.rs` calls "separate img/txt weights, joint attention". The remaining blocks are single-stream, one set of weights over the whole sequence, the arrangement that Z-Image uses throughout. A double-stream block holds exactly twice the weights of a single-stream block of the same width ($245{,}367{,}296$ against $122{,}683{,}648$ for the 4B variant, which is $13d^2$ per set of weights plus the query-key gains), and every token still meets one set only, so the arithmetic per token is the same in both. The choice is between the capacity of separate weights for the two kinds of token and the economy of one set; Z-Image's refiners play the role of the separate weights, as short specialised stacks before the streams join.

The modulation differs too. FLUX.2 computes the shift, scale and gate vectors once per pass, from one projection of the timestep vector for each of: the image stream of the double blocks (6 vectors, a shift, scale and gate for each of the two sublayers), the caption stream (6), the single blocks (3: one shift, one scale and one gate, because attention and feed-forward share one input norm and run in parallel there) and the final layer (2). The same vectors serve every block of their kind. The module header of `crates/flux2/src/model.rs` states the consequence: because the modulation depends on the timestep alone, $(1 + \mathrm{scale}) \cdot \mathrm{LN}(x) + \mathrm{shift}$ is exactly a layer norm with gain $1 + \mathrm{scale}$ and bias $\mathrm{shift}$, and the model carries six such pairs of vectors per pass for all of its blocks, "no per-block modulation work at all". This is the shared-table idea in a stronger form than Z-Image's per-block projections; the helper `add_table` in `crates/dit/src/adaln.rs` is the weaker form that adds a small per-block table to one shared vector, used by Wan's video model.

That vector is a function of the timestep, and the timestep need not be one number. The video generator's tokens can carry different timesteps, the frozen ones $0$ and the generated ones $\sigma$, and its modulation then has one row per token. Knowledge #49 records the cost of getting that wrong: the sampler gave every token $\sigma$, a frozen token carrying a clean image was modulated as if it were pure noise, and joint attention mixed it into every other token, so that the pinned frame decoded perfectly and every other frame was a fine mesh texture; the symptom pointed at the model's capacity, not at its input. Commit `b13cda561` then noted that the per-token table is a function of the token's scalar and nothing else, so a row costs $2 \times 4096 \times 36864 = 301{,}989{,}888$ FLOPs at that model's shapes and the table only needs one row for each distinct timestep, which is one for plain text-to-video and two once anything is frozen; the helper `distinct_rows` in `adaln.rs` keys the rows by the raw bits of the scalar so that the compact table is bit-identical to the expanded one.

## The clock and the sign of Z-Image

Chapter 6 fixed the straight path $x_\sigma = (1 - \sigma)x_0 + \sigma\varepsilon$ and a network that returns the velocity $\varepsilon - x_0$. Z-Image's reference implementation runs in the other orientation, and the sampler of `crates/s3dit/src/pipeline.rs` adapts to it in two places: the clock the network is given is $t = 1 - \sigma$ (the pipeline's `(1000.0 - ts[i]) / 1000.0`, with the scheduler's timestep $1000\sigma$), which the sinusoid then sees as $1000t$; and the network's output is negated before the Euler step, with the comment that the reference does so (`noise_pred = -noise_pred`) and that brain's scheduler is the bare $x + (\sigma_{\text{next}} - \sigma)v$. The network therefore returns $x_0 - \varepsilon$, the velocity of the path run from noise to data, and the sampler's $v$ is its negative.

<!-- worked: denoiser_architectures.sign_convention -->
Three Euler steps of the straight path with the exact denoiser of the three clusters, written in the clock and the sign of Z-Image: velocity u = E[noise - x_0 | x] = (x - guess)/sigma; the transformer is given t = 1 - sigma, the sinusoid sees 1000 t, and the transformer's output is -u = E[x_0 - noise | x]; the sampler negates the output and steps x + (sigma' - sigma) u.

| x | sigma | sigma' | u | t | 1000 t | output (-u) | step from -output | Euler step with u |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6 | 0.4 | -2.256330 | 0.4 | 400 | 2.256330 | 1.451266 | 1.451266 |
| -2.2 | 0.3 | 0.2 | 2.787565 | 0.7 | 700 | -2.787565 | -2.478756 | -2.478756 |
| 0.4 | 0.9 | 0.7 | 0.119574 | 0.1 | 100 | -0.119574 | 0.376085 | 0.376085 |
<!-- /worked -->

The last two columns of every row agree, which is the only claim: the convention is a change of variables, not of sampler. It is a hazard for the same reason as the half order of the embedding: a sampler that passed $1000\sigma$ instead of $1000(1 - \sigma)$ would hand the network the embedding of a different noise level (the two coincide only at $\sigma = 0.5$) and the network would answer for it, with every tensor still the right shape.

## A small denoiser to build and check

Everything above can be checked on a network small enough to hold in a few pages of code. The small denoiser is Z-Image with every dimension cut: width $24$, two heads of width $12$, a feed-forward of width $64$, two blocks, a $4 \times 4$ latent of two channels in $2 \times 2$ patches (four image tokens of $8$ values), a caption of two tokens of $5$ features, a timestep embedding of $16$ values and a conditioning vector of $24$. It omits the two refiners and the pad tokens, keeping everything that decides the answer: the clock through the embedding and the two-layer network, the caption's norm and projection, the joint sequence with image tokens first, the multi-axis rotation on three axes of width $4$ (two pairs each, base $256$), the double norms with scale and gate folded into the gains, and the final modulated layer.

<!-- worked: denoiser_architectures.small_setup -->
Weights: the book's SplitMix64, seed 0x8a68e1a97d55d4a3 (`seed_of` of the name `denoiser_architectures.small`), one stream for all tensors, drawn tensor by tensor in the order below (the two blocks l = 0 and l = 1 in turn, each in the order listed; a matrix row by row, so that its `[out, in]` entry (o, i) is draw number o*in + i of the matrix; `normal()` is the generator's Box-Muller draw). Inputs: seed 0x0a10f22755053473 (`denoiser_architectures.small_input`): the 32 entries of the latent (2 channels of 4 rows of 4 columns, channel slowest), then the 10 caption features (2 tokens of 5), each `round(normal(), 1)`. Width 24, 2 heads of width 12, feed-forward width 64, conditioning width 24, timestep embedding of dimension 16, 2 blocks, patches of 2 by 2 (8 values), RMS-norm epsilon 1e-5 (final layer norm 1e-6). 43 tensors, 21,365 parameters.

| tensor | recipe | shape |
|---|---|---|
| cap_embedder.0.weight | gain: 1 + 0.1 normal() | [5] |
| cap_embedder.1.weight, .bias | weight: normal()/sqrt(5); bias: 0.1 normal() | [24, 5], [24] |
| x_embedder.weight, .bias | weight: normal()/sqrt(8); bias: 0.1 normal() | [24, 8], [24] |
| t_embedder.mlp.0.weight, .bias | weight: normal()/sqrt(16); bias: 0.1 normal() | [32, 16], [32] |
| t_embedder.mlp.2.weight, .bias | weight: normal()/sqrt(32); bias: 0.1 normal() | [24, 32], [24] |
| layers.l.attention.to_q, to_k, to_v, to_out.0 (weights) | normal()/sqrt(24), in this order | 4 x [24, 24] |
| layers.l.attention.norm_q, norm_k | gain: 1 + 0.1 normal() | 2 x [12] |
| layers.l.feed_forward.w1, w2, w3 (weights) | normal()/sqrt(fan_in): 24, 64, 24 | [64, 24], [24, 64], [64, 24] |
| layers.l.attention_norm1, attention_norm2, ffn_norm1, ffn_norm2 | gain: 1 + 0.1 normal(), in this order | 4 x [24] |
| layers.l.adaLN_modulation.0.weight, .bias | weight: normal()/sqrt(24); bias: 0.1 normal() | [96, 24], [96] |
| final_layer.adaLN_modulation.1.weight, .bias | weight: normal()/sqrt(24); bias: 0.1 normal() | [24, 24], [24] |
| final_layer.linear.weight, .bias | weight: normal()/sqrt(24); bias: 0.1 normal() | [8, 24], [8] |

Latent, channel 0 then channel 1, rows top to bottom: (0.6, 0.5, -0.5, 0.6); (1.1, -0.7, 1.3, -0.1); (-0.5, 0.4, -1.5, -0.6); (0.8, 2.1, -0.4, -0.9) | (-0.9, -0.8, 1.6, 0.5); (1.5, 1.4, 1.8, 0.2); (0, -0.5, -0.9, -0.1); (0.7, -0.4, 0.5, -1). Caption features: (-2, -1.2, -0.3, -1, -1.2); (-0.8, -0.9, -0.1, 1.3, 0.6).
<!-- /worked -->

The routine of the earlier pseudocode is the block; the rest is a few lines that follow the description above. In the order of execution: the clock is `embed(1000 * (1 - sigma), 16)` through `Linear(16, 32)`, `silu`, `Linear(32, 24)`; the image tokens are the patchified latent through `Linear(8, 24)`; the caption tokens are `rms_norm` with the gain of `cap_embedder.0` then `Linear(5, 24)`; the joint sequence is the four image tokens and the two caption tokens in that order, with the ids $(3, h, w)$ for the image patches and $(1, 0, 0)$, $(2, 0, 0)$ for the caption; two blocks are applied; the final layer applies the parameter-free layer norm (variance with divisor $24$, $\varepsilon = 10^{-6}$) to all six tokens, multiplies every row by $1 + \mathrm{Linear}(\mathrm{silu}(c))$, applies `Linear(24, 8)` and keeps the four image rows, which unpatchify (channel innermost, rows of patches in row-major order) to the predicted latent. The sums of every matrix product run over the input index in increasing order.

<!-- worked: denoiser_architectures.small_outputs -->
The output of the small denoiser at sigma = 0.8 and at sigma = 0.2 (time argument 1000 (1 - sigma) = 200 and 800), for the inputs above: the predicted latent, channel and row, four columns. Library cos, sin, exp, tanh; the values are those of double precision and are printed to 6 decimals.

| sigma | channel | row | col 0 | col 1 | col 2 | col 3 |
|---|---|---|---:|---:|---:|---:|
| 0.8 | 0 | 0 | 2.578161 | 0.475865 | -2.770127 | 0.834668 |
| 0.8 | 0 | 1 | -1.330344 | 0.438077 | -0.476083 | 0.144010 |
| 0.8 | 0 | 2 | 2.363874 | 1.396669 | -0.510410 | -0.719178 |
| 0.8 | 0 | 3 | 0.927757 | -0.297448 | 0.100399 | 0.572481 |
| 0.8 | 1 | 0 | -0.051475 | -0.776654 | -0.077868 | -2.271438 |
| 0.8 | 1 | 1 | 0.458400 | 0.900821 | 1.720616 | -0.774639 |
| 0.8 | 1 | 2 | 0.613070 | 1.704113 | -0.388910 | -1.749212 |
| 0.8 | 1 | 3 | -0.176607 | -2.320838 | -0.852303 | 2.494663 |
| 0.2 | 0 | 0 | 1.421319 | -0.222295 | -1.412528 | -0.692966 |
| 0.2 | 0 | 1 | 0.074420 | 0.915238 | -1.098335 | 0.028816 |
| 0.2 | 0 | 2 | 1.810193 | 0.707200 | 1.009897 | -1.001193 |
| 0.2 | 0 | 3 | 0.291272 | -0.173199 | 0.751728 | 1.620283 |
| 0.2 | 1 | 0 | -0.555637 | 0.311597 | 0.505566 | -1.703627 |
| 0.2 | 1 | 1 | 2.940184 | -0.281757 | 1.002423 | -1.029118 |
| 0.2 | 1 | 2 | 0.750289 | 1.566654 | -1.298291 | 1.516545 |
| 0.2 | 1 | 3 | 0.042000 | -1.961018 | 0.506477 | 0.713254 |
<!-- /worked -->

The values are those of a function that was evaluated and not trained, in double precision, so another implementation should agree to the printed digits up to the differences of the library's cosine, sine, exponential and $\tanh$ (a few units in the last of the 16 digits, far below the six printed). The properties the chapter has claimed can be tested on it.

<!-- worked: denoiser_architectures.small_checks -->
Checks on the small denoiser (inputs and weights as above). The sequence is the 6 tokens of the joint stream, the 4 image tokens and then the 2 caption tokens. Every comparison of tokens ('after the blocks', the two exchange rows, the caption-token row) is on that 6-token sequence at sigma = 0.8 after the two blocks, before the final layer, not on the predicted latent; the exchange of the two patch columns permutes the image tokens 0 <-> 1 and 2 <-> 3 (the caption tokens stay), and the outputs are compared after permuting them the same way. The rows that say 'predicted latent' or 'output' compare the final layer's output.

| change | measured | result |
|---|---|---|
| both gates of every block set to 0 | largest change of the joint sequence over 2 blocks | 0 |
| trained-like gates (the weights above) | rms change of the sequence / rms of the sequence | 0.6835 |
| scale and gate folded into the norm gains vs applied after the norms | largest difference of the output | below 1e-12 |
| sigma = 0.8 vs sigma = 0.2 | largest difference of the output | 3.2658 |
| no rotary position, the two patch columns exchanged | largest difference between the permuted outputs | below 1e-12 |
| with rotary position, the same exchange | largest difference between the permuted outputs | 0.7160 |
| 0.5 added to caption token 0 | largest change of the predicted latent | 0.3734 |
| 0.5 added to latent entry (0, 0, 0) | largest change of the two caption tokens after the blocks | 0.2700 |
<!-- /worked -->

The first row is the adaLN-zero property: with both gates of every block at zero, the six tokens leave two blocks exactly as they entered (the change is $0$, not small, since the gate multiplies the norm's output by an exact zero and the sum adds an exact zero). The second row shows the gates off zero doing real work, a change of $0.6835$ of the stream's size. The third is the fold: applying the scale and the gate to the norm's gain gives the result of applying them after the norm, to below $10^{-12}$. The fourth shows that the clock changes the answer, by $3.2658$ between $\sigma = 0.8$ and $0.2$. The fifth and sixth are the reason for the rotary embedding: without it exchanging the two patch columns of the input exchanges the corresponding output tokens (below $10^{-12}$), so the network cannot tell where a patch is, and with it the permuted outputs differ by $0.7160$. The last two rows are joint attention in a whole network: a change of $0.5$ in a caption feature changes the predicted latent by up to $0.3734$, and a change of $0.5$ in one latent entry changes the caption tokens after the blocks by up to $0.2700$.

## What a pass costs

The architecture decides the work of a pass, and the work can be counted from the configuration alone. The next table puts the four networks side by side at $1024 \times 1024$.

<!-- worked: denoiser_architectures.architecture_table -->
Four denoisers at 1024 by 1024 pixels (SDXL: a 128 by 128 grid and 77 text tokens; the transformers: 4096 image tokens and 512 caption tokens, one token per 16 by 16 pixels). 'Time parameters' are the weights that turn the timestep (and for SDXL the size and pooled-text vector) into per-layer conditioning: SDXL's two embedding MLPs and the 17 per-resnet projections; Z-Image's timestep MLP, the per-block modulation projections and the final layer's; FLUX.2's timestep MLP, three shared modulation projections and the final layer's. GFLOP counts multiply-adds as two operations and leaves out norms, softmax and activations. The last column is the share spent in the T x T terms of attention. What is counted differs by what a pass applies to: for SDXL, every layer of the counting rules of the U-Net table, including the matrices that act on the one conditioning vector (about 50 million floating-point operations); for the transformers, the attention and feed-forward matrices applied to every token of every block (refiner blocks to their own tokens only), the patch embedder, the caption embedder and the final projection, and 4 T^2 d per block for the scores and the weighted sum (T the tokens that attend), but not the matrices that act on the one conditioning vector: 255 million floating-point operations for Z-Image (the 32 modulation projections, the timestep network and the final layer's modulation) and 341 and 606 million floating-point operations for the two FLUX.2 sizes (the shared modulation projections and the timestep network), which would move the last printed digit.

| network | stream | parameters | time parameters | share | GFLOP per pass | attention share |
|---|---|---:|---:|---:|---:|---:|
| SDXL U-Net | - | 2,567,463,684 | 24,922,560 | 1.0% | 6,761.2 | - |
| Z-Image Turbo | single | 6,154,908,736 | 127,833,088 | 2.1% | 62,506.1 | 16.5% |
| FLUX.2 klein-4b | double + single | 3,875,544,576 | 170,655,744 | 4.4% | 34,819.8 | 18.7% |
| FLUX.2 klein-9b | double + single | 9,078,581,248 | 303,038,464 | 3.3% | 75,514.1 | 14.7% |
<!-- /worked -->

The transformers have more parameters than the U-Net and spend far more arithmetic: a Z-Image pass is $62{,}506$ GFLOP against SDXL's $6{,}761$, $9.2$ times as much, for $2.4$ times the weights. The ratio is one of tokens: SDXL's widest layers, the $1280$-channel ones, see $1024$ positions, and only its thin $320$-channel layers see $16384$; every Z-Image block, $12D^2$ weights for each of $4608$ tokens, works at full width on the whole sequence. The time parameters are between $1.0\%$ and $4.4\%$ of the weights in all four, so the clock is cheap in every design; FLUX.2 klein-4b spends the largest share, $4.4\%$, on its modulation projections and the timestep network.

The attention column grows with the image, because attention's cost in $T$ is quadratic and everything else is linear.

<!-- worked: denoiser_architectures.cost_vs_side -->
Z-Image with a 512-token caption at four image sides: the image tokens (one per 16 by 16 pixels), the joint sequence, the GFLOP of one pass and the share of them in attention's T x T terms, the bytes (in GB, 32-bit) of the score matrices of the 30 heads of ONE main block if they were stored, and, for comparison, the GFLOP of one SDXL U-Net pass at the same image side (grid side = image side / 8, 77 text tokens; counted with the rules of the U-Net table, so with the conditioning matrices, while the Z-Image column leaves out the matrices that act on the one conditioning vector, as in the table of the four denoisers). The score elements of the 30 heads, 30 T^2, still fit the 32-bit count of a dispatch for T up to 11,965.

| image side | image tokens | joint tokens N | GFLOP, Z-Image | attention share | scores of one block (GB) | GFLOP, SDXL |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 1,024 | 1,536 | 18,533.4 | 6.1% | 0.3 | 1,588.8 |
| 1024 | 4,096 | 4,608 | 62,506.1 | 16.5% | 2.5 | 6,761.2 |
| 1536 | 9,216 | 9,728 | 156,409.9 | 29.6% | 11.4 | 17,261.0 |
| 2048 | 16,384 | 16,896 | 331,168.4 | 42.2% | 34.3 | 35,906.8 |
<!-- /worked -->

![The share of the multiply-adds of a forward pass spent in the T by T terms of self-attention against the image side from 256 to 2048 pixels, for the SDXL U-Net, Z-Image and FLUX.2 klein-4b: all three rise, the two transformers from about 3 and 4 percent at 256 pixels to about 42 and 46 percent at 2048, the U-Net more slowly from under 1 percent to 33 percent.](img/attention-share.svg)

Doubling the image side from $1024$ to $2048$ quadruples the number of image tokens but multiplies the pass by $5.3$, and the attention share goes from $16.5\%$ to $42.2\%$. The score matrices of one block of 30 heads, if they were stored, grow from $2.5$ GB to $34.3$ GB, which is why brain's blocks have a fused path that never stores them (Volume V, Chapter 13): the gate in `crates/s3dit/src/block.rs` calls flash attention a memory escape hatch and turns it on when the score buffer would approach the device's binding limit. From $11{,}966$ tokens on the count of score elements, $30\,T^2$, no longer fits the 32-bit thread count of a dispatch, and the comment on `dispatch_threads` records the failure that the code now refuses by name: a release build would wrap the count silently to a small number, cover a sliver of the scores and corrupt the output with nothing raised.

The U-Net is cheaper at every size in the table, but its attention share also rises, to $33.5\%$ at $2048$ pixels, since its coarse levels' token count grows with the image as well. The curves are not an argument for either design. They say where the cost goes as the picture grows, and what a designer buys with a patch of $16$ pixels, or a coarser autoencoder, or a window on the attention (Volume VII, Chapter 6).

<!-- worked: denoiser_architectures.work_per_image -->
The work of one 1024 by 1024 image at each pipeline's default pass count (the passes of Chapter 7's opening table). A pass reads every weight once; at 16 bits that is 2 bytes a parameter, so the operations per weight byte of a pass are its GFLOP over twice the parameters. TFLOP per image is passes times GFLOP per pass over 1000.

| pipeline | passes | GFLOP per pass | operations per weight byte | TFLOP per image | note |
|---|---:|---:|---:|---:|---|
| SDXL, 30 steps with guidance | 60 | 6,761.2 | 1,317 | 405.7 | two passes a step |
| Z-Image Turbo, 8 steps | 8 | 62,506.1 | 5,078 | 500.0 | no guidance |
| FLUX.2 klein-4b, 4 steps | 4 | 34,819.8 | 4,492 | 139.3 | distilled, no guidance |
| FLUX.2 klein-9b, 4 steps | 4 | 75,514.1 | 4,159 | 302.1 | distilled, no guidance |
| FLUX.2 base 4b, 50 steps with guidance | 100 | 34,819.8 | 4,492 | 3,482.0 | two passes a step |
<!-- /worked -->

The table gives the passes of Chapter 7's opening table and what they come to. SDXL's default request costs $405.7$ TFLOP, Z-Image Turbo's $500.0$, FLUX.2 klein-4b's $139.3$, and the undistilled base model's $3{,}482.0$: the same FLUX.2 weights cost $25$ times as much undistilled as distilled, a factor that comes from the sampler and the distillation of Chapter 7 and not from the architecture. The fourth column is the other cost: a pass reads every weight once, and the operations per weight byte of a pass are above one thousand in every row. Compare the ridge point of any device against it with the roofline of Volume IV, Chapter 3: a pass of any of these networks is limited by arithmetic and not by weight traffic, at the sizes of the table. The picture is the opposite of the decode step of a language model, which reads all its weights to produce one token.

## Where the three crates divide the work

The three crates are a case study in what is shared and what is not. `crates/sdxlunet` adds no kernel and no block of its own; its header says that everything convolutional is the shared block builder of the autoencoder crate and everything transformer is the shared model blocks. The commit that introduced it (`c7d97a708`) calls this "the measured prediction" that put the U-Net family ahead of a bigger diffusion transformer when the imaging plan was drawn up: the U-Net could be assembled from what the repository already had. The same graph carries the adapters of Chapters 11 and 12 as seams (the control residuals, the cross-attention injection, the skip replacement of the restoration network) (commit `cff35b745` added the seam that replaces the up path's skip concatenation).

`crates/dit` is the other direction. Z-Image and Wan, written separately, each carried their own copy of the timestep sinusoid and MLP, of the patchify and of the adaLN table; commit `7b78e8b2d` hoisted them into one crate and migrated both consumers, behaviour-preserving to the bit in their parity and gradient tests, and left Wan's channel-outermost forward patchify where it was, because it is a different order and not a duplicate. What it holds today is the rotary tables, the patchify pair, the timestep network and the table helpers: the modulation itself lives in `crates/s3dit/src/block.rs` for Z-Image and in the model file of `crates/flux2` for FLUX.2, because the transformers do not compute their modulation in the same way, and unifying something deliberately different would change what a checkpoint computes (the module header of `crates/dit/src/adaln.rs` makes the same argument for the shared-table trick).

`crates/s3dit` is the Z-Image assembly: the host orchestration of `model.rs` (patchify, embedders, the three stages, final layer), the recorded device block of `block.rs`, and a family of ways to run it (a resident engine, an 8-bit one, one split across two devices, a windowed one that rotates blocks through a fixed pool of device buffers). The 8-bit path quantizes the seven linear layers of each block and keeps the norms in 32-bit floats; it needs no change to the architecture, which is a point of the design: the architecture is a list of matrix products, and a way to run one is a way to run all.

## Check your understanding

1. Find the error: "A Z-Image checkpoint trained with the cosine half of the timestep embedding first can be run with the sine half first, because both are smooth 256-vectors with the same inner products."
2. Predict: in the gain table, suppose the added-term layer is fitted using only the levels $\sigma \ge 0.5$. Which level has the largest excess error, and why?
3. At the first up level of the SDXL U-Net the three resnets read $2560$, $2560$ and $1920$ channels. Reconstruct the numbers from the skip stack, and say what would have been wrong with a rule that gave every resnet of the level $1280 + 1280$ channels.
4. Why does the context refiner have no modulation projection, and what would a modulated context refiner have as the input of its projection?
5. A caption slot grows from $512$ to $1024$ tokens at a fixed $1024 \times 1024$ image. By what factor do the image tokens, the joint sequence and the attention's T-by-T terms change? Which changes the most?
6. In the five-token example, the caption outputs under cross-attention do not move when the picture changes. What does that say about what the text path of the U-Net can express in one layer, and where does the picture's effect on the prompt show up there?
7. Find the error: "Setting the gate to zero makes the attention weights of the block zero, which is why the block is the identity."
8. A pipeline passes $1000\,\sigma$ to a Z-Image transformer. Every output has the right shape and the picture is noise-like and over-smoothed. Which two parts of the mechanism of this chapter does the failure involve?

## Summary

A denoiser that is not told the noise level pulls every input toward the data as if the level were unknown, and the loss is largest at the smallest levels, where the detail is. Telling it needs the level to enter as a gain and not only as a shift, which a normalization with a scale and a shift computed from the level gives, and a gate on what each sublayer writes lets the block start as the identity. The level reaches the network as a sinusoidal embedding of a scaled timestep, through a two-layer network; the order of the embedding's halves and the scale of the clock are conventions of the checkpoint that no statistic of the embedding can verify.

The U-Net of SDXL is the encoder-decoder of Volume VII at three resolutions and $2.57$ billion parameters, with a stack of nine skips that the up path pops three at a time, a time embedding added in each of 17 resnets, and 70 transformer blocks at the two coarser levels whose cross-attention reads the prompt without rewriting it. Putting attention only where the token count is small is its answer to the $T^2$ cost. The diffusion transformers of Z-Image and FLUX.2 cut the latent into 256-pixel tokens, give each a position on three axes, and let the prompt and the picture attend to one another in one softmax; they spend several times the arithmetic of the U-Net, with attention taking a growing share as the picture grows, and differ in whether the modulation is projected per block (Z-Image, four vectors) or shared across them (FLUX.2). A small version of the Z-Image block fixes all of this in numbers that can be re-computed.

The prompt has been a given in this chapter: $77 \times 2048$ numbers for the U-Net, a $512$-token slot for the transformers. How a sentence becomes those vectors, and what the pooled vector that joins SDXL's clock is, is the next chapter's subject; how the encoder, the denoiser, the sampler and the decoder fit together is the one after.

## References

- Olaf Ronneberger, Philipp Fischer and Thomas Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, 2015. The encoder-decoder with skip connections by concatenation.
- Jonathan Ho, Ajay Jain and Pieter Abbeel, *Denoising Diffusion Probabilistic Models*, 2020. A U-Net with a sinusoidal timestep embedding as the noise predictor.
- Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin and Aaron Courville, *FiLM: Visual Reasoning with a General Conditioning Layer*, 2018. A scale and a shift computed from a conditioning vector.
- Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser and Bjorn Ommer, *High-Resolution Image Synthesis with Latent Diffusion Models*, 2022. The U-Net with cross-attention to a prompt in the latent space.
- Dustin Podell and others, *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*, 2023. The configuration of the SDXL U-Net and its added size and pooled-text conditioning.
- William Peebles and Saining Xie, *Scalable Diffusion Models with Transformers*, 2022. The diffusion transformer, adaptive layer normalization and the zero-initialised gate.
- Patrick Esser and others, *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*, 2024. Separate weights for the image and text streams with a joint attention.

## Exercises

1. **[Calculate]** Count the parameters of the first resnet of the first up level of the SDXL U-Net: input $2560$ channels, output $1280$, time dimension $1280$, two $3 \times 3$ convolutions, a $1 \times 1$ shortcut, two group norms and the time projection. Then give its multiply-adds at $32 \times 32$ (two FLOPs each, convolutions and the projection only).
2. **[Calculate]** A $768 \times 1024$ image is generated by Z-Image with a $512$-token caption slot. How many image tokens and joint tokens are there, and how many score elements does one block's attention build (30 heads)? Is the materialised path still allowed by the 32-bit thread count?
3. **[Calculate]** Compute the embedding of dimension 8 and base 10000 at $t = 3$, to four decimals, in both half orders.
4. **[Calculate]** Normalize the row $(3, 4)$ with an RMS norm of gain $(1, 2)$ and $\varepsilon = 0$, then apply a scale $s = 0.5$ after the norm; then fold the scale into the gain before the norm. Show the two results are the same.
5. **[Calculate]** Derive the parameter count of one FLUX.2 klein-4b single-stream block from $d = 3072$, 24 heads, SwiGLU width $3d$: the fused input matrix has $3d + 2 \cdot 3d$ outputs and the fused output matrix reads $d + 3d$ inputs. Check the product by $20$ against the table.
6. **[Calculate]** For a U-Net with channels $(64, 128, 256)$ and two resnets per down block, write the skip stack and the input width of each up-path resnet. Compare with the SDXL pattern $2560, 2560, 1920 \mid 1920, 1280, 960 \mid 960, 640, 640$.
7. **[Implement]** Reproduce `denoiser_architectures.embedding_table` and `denoiser_architectures.scale_table`, including the two half orders and the Gram check, from the formula of the embedding.
8. **[Implement]** Reproduce `denoiser_architectures.unet_walk` and `denoiser_architectures.unet_params`: the stage table, the stack of skips and the totals ($2{,}567{,}463{,}684$ parameters; $1610$ and $1680$ tensors) from the configuration alone.
9. **[Implement]** Reproduce `denoiser_architectures.zimage_params` and `denoiser_architectures.flux2_table`, including the per-block counts and the tensor totals.
10. **[Implement]** Build the small denoiser and reproduce `denoiser_architectures.small_outputs` and the eight rows of `denoiser_architectures.small_checks` to the printed digits, from the recipes of `denoiser_architectures.small_setup`.
11. **[Investigate]** Give the SDXL U-Net a transformer at its finest level (change its down and up block types to attention at level 0, with 5 heads and one transformer block per resnet) and count the extra parameters, the extra GFLOP at $128 \times 128$ and the extra bytes of the score matrix. What does the table say about the choice not to?
12. **[Investigate]** Extend `denoiser_architectures.blind_denoiser` with a network told $\sigma$ only through an added term, using the fixed-gain error of `denoiser_architectures.gain_table` as a model of what it can do, and find the noise levels at which it is worse than the blind denoiser.

## Key terms

- **U-Net**: an encoder-decoder network that halves the resolution of a map level by level, then doubles it back, joining each level of the way up to the map of the same size from the way down by a skip connection, so that its output has the size of its input; the denoiser of SDXL has three levels and a stack of nine skips. <!-- concept: u-net -->
- **timestep embedding**: the vector made from the scalar noise level by a sinusoidal encoding (cosines of the scaled level at a ladder of rates, then the sines) and a small network, so that a model built from vectors can read how noisy its input is. <!-- concept: timestep-embedding -->
- **adaptive layer normalization**: a normalization layer whose scale and shift (and, around a sublayer, whose gate) are computed from a conditioning vector such as the timestep embedding instead of being learned constants; a zero gate makes the block the identity. <!-- concept: adaptive-layer-norm -->
- **diffusion transformer**: a denoiser made of transformer blocks over the patch tokens of a latent, with the noise level entering through adaptive layer normalization, and a last layer that maps the tokens back to patches. <!-- concept: diffusion-transformer -->
- **joint attention**: one attention in which the prompt tokens and the image tokens are queries and keys of the same softmax, so that each kind of token is updated by the other; a diffusion transformer built on it keeps the prompt as part of its sequence. <!-- concept: joint-attention-dit -->

## Answers

**Recall questions.**

1. $8 + 4 = 12$ channels; $12 \cdot 4 \cdot 9 + 4 = 436$ parameters.
2. The 5 positions supply the queries, the 3 positions the keys and values; the weight matrix is $5 \times 3$ and each row sums to $1$.
3. $4D^2$ for the attention and $3 \cdot D \cdot (8D/3) = 8D^2$ for the feed-forward, $12D^2$ in all.
4. $\omega_k = 10000^{-k/4} = 1, 0.1, 0.01, 0.001$; the pair with rate $1$ changes fastest.
5. 10 channels per group, so each mean and variance is over $10 \cdot H \cdot W$ numbers.
6. $dx_\sigma/d\sigma = \varepsilon - x_0$; the exact network returns its conditional mean $\mathbb{E}[\varepsilon - x_0 \mid x_\sigma]$, which is $(x - \hat x_0)/\sigma$.

**Check your understanding.**

1. Exchanging the halves is a permutation of the coordinates. The inner products are unchanged, but the first linear layer's weights are attached to coordinates and read the wrong half; it works only if the weight columns are permuted the same way, which retraining or editing the checkpoint would have to do (commit `ca12c1f91` is the story of getting this wrong).
2. $\sigma = 0.02$: it is the level furthest from the fitted gain (a fixed $w$ fitted to $\sigma \ge 0.5$ is near $k = 0.1$, while $k(0.02) = 1.01$), and the variance of $x$ there is small, but the squared gain error $(k - w)^2$ is about $0.8$ against a best possible error of $0.0004$.
3. Hidden state $1280$ wide for the first resnet plus the top skip ($1280$) is $2560$; the second resnet's hidden state is the block's own output ($1280$) plus the next skip ($1280$): $2560$; the third is $1280$ plus the $640$ of the last skip pushed by the first level's downsampler: $1920$. A rule that gave all three $1280 + 1280$ would size the third resnet $640$ channels too wide, and the same kind of mistake at the next level, where the hidden state is $1280$ wide and the level $640$, is the one that the importer's shape check caught (commit `c7d97a708`).
4. The caption has no timestep, so there is nothing to project; a modulated block's projection would take the 256-wide timestep vector, which is the same for every caption token and would only add a constant per channel.
5. The image tokens do not change ($4096$); the joint sequence goes from $4608$ to $5120$ ($1.11$ times); the T-by-T terms go as $N^2$: $(5120/4608)^2 = 1.23$ times. The attention terms change the most.
6. Within one cross-attention layer the prompt cannot be adapted to the picture; the picture's effect on the prompt appears only through the picture tokens that read it, so every layer reads the same prompt vectors and the projection to keys and values is recomputed from the same $77 \times 2048$ numbers at each of the 70 sites.
7. The gate multiplies the output of the sublayer after its norm; the attention weights are untouched, and what is zero is the term added to the stream.
8. The clock: the embedding of $1000\sigma$ is the embedding of a different level ($1 - \sigma$ vs $\sigma$ coincide only at $0.5$), so the network denoises for the wrong level; and the convention of the sampler (the clock is $1 - \sigma$), which is a property of the checkpoint and not of the architecture.

**Exercises (calculation answers).**

1. Convolution 1: $2560 \cdot 1280 \cdot 9 + 1280 = 29{,}492{,}480$; convolution 2: $1280 \cdot 1280 \cdot 9 + 1280 = 14{,}746{,}880$; shortcut: $2560 \cdot 1280 + 1280 = 3{,}278{,}080$; time projection: $1280 \cdot 1280 + 1280 = 1{,}639{,}680$; norms: $2 \cdot 2560 + 2 \cdot 1280 = 7{,}680$. Total $49{,}164{,}800$. Multiply-adds at $32 \times 32 = 1024$ positions: $2 \cdot 1024 \cdot (2560\cdot1280\cdot9 + 1280\cdot1280\cdot9 + 2560\cdot1280) + 2 \cdot 1280 \cdot 1280 = 2 \cdot 1024 \cdot 47{,}513{,}600 + 3{,}276{,}800 = 97{,}311{,}129{,}600$ operations, about $97.3$ GFLOP.
2. $48 \cdot 64 = 3072$ image tokens, $3584$ joint; $30 \cdot 3584^2 = 385{,}351{,}680$ score elements, well below $2^{32}$, so the materialised path is allowed (the memory is $1.5$ GB in 32-bit floats).
3. Cosine first: $(\cos 3, \cos 0.3, \cos 0.03, \cos 0.003, \sin 3, \sin 0.3, \sin 0.03, \sin 0.003) = (-0.9900, 0.9553, 0.9996, 1.0000, 0.1411, 0.2955, 0.0300, 0.0030)$; sine first exchanges the halves.
4. The normalized row is $(3, 4)/\sqrt{12.5} = (0.848528, 1.131371)$; times the gain $(1, 2)$ it is $(0.848528, 2.262742)$; after the scale $1.5$ it is $(1.272792, 3.394113)$. Folding: the gain is $(1.5, 3)$ and the result $(1.5 \cdot 0.848528, 3 \cdot 1.131371) = (1.272792, 3.394113)$.
5. The fused input matrix is $(3d + 2 \cdot 3d) \times d = 9d \times d = 9 \cdot 3072 \cdot 3072 = 84{,}934{,}656$; the fused output is $d \times (d + 3d) = 4d^2 = 37{,}748{,}736$; plus two query-key gains of width $128$: $256$; total $122{,}683{,}648$; times $20$ is $2{,}453{,}672{,}960$, the table's single-stream column.
6. The stack is $64, 64, 64, 64, 128, 128, 128, 256, 256$; the up resnets read $512, 512, 384 \mid 384, 256, 192 \mid 192, 128, 128$ (first resnet of a level: the previous level's output plus the top skip).

**Investigate.** Exercise 11: the extra work is about $2{,}063$ GFLOP and $16.8$ million parameters for the transformers at the finest level (the walk gives $8{,}824$ GFLOP in all against $6{,}761$), and the extra scores are $5.4$ GB per layer, $10.7$ GB with the softmax; the table says that the arithmetic is affordable and the memory is not without a fused kernel. Exercise 12 is open; the added-term layer should lose most at the smallest levels, where its fixed gain is far from $k(\sigma) \approx 1$.
