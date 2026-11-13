<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Project: A Small Image Generator

*Can you train a small generator and grow it into a conditioned latent generation system?*

Here are four ways to produce three hundred pictures, one hundred for each of three classes, and the numbers an evaluator might compute for them. The first row is the data. The second is a program that never learns anything: it returns the noise-free template of the requested class, alternating between the class's two positions. Its average log-density under the true distribution is higher than the data's own by more than eighteen nats, because a picture that sits exactly on the peak of a Gaussian is more probable than a typical draw from it. The third row always returns the same position, and the density does not notice. The fourth returns the average of the class's two templates, and it is the one that fails every measure except the class.

<!-- worked: project_image_generator.opening -->
Four ways to produce 300 pictures (100 requests for each of the three classes; the data are drawn class by class) and the five measures of this chapter. 'Accuracy' is the share whose nearest template is of the requested class; 'on data' the share whose distance to that template, per pixel, is below 0.375; 'distance' the mean of that distance (the data's own is about 0.25); the 'mean ln p' column the mean of ln p(x given class) under the exact mixture; 'TV' the total variation distance of the positions among each class's pictures from the even split (0.5 sum |share - 1/2|, averaged over the classes). The data row is drawn from `project_image_generator.data` (seed 0x8d60cb533611bc47) (for each class in turn, 100 times p = integer(2) and then 36 normal()). The expected value of the 'mean ln p' column for the data is -(D/2)(1 + ln(2 pi sigma_x^2)) - ln 2 = -1.87, and for a picture that sits exactly on a template -(D/2) ln(2 pi sigma_x^2) - ln 2 = 16.13.

| generator | accuracy | on data | distance | mean ln p | TV |
|---|---:|---:|---:|---:|---:|
| the data, 300 pictures | 1.000 | 1.000 | 0.249 | -2.02 | 0.050 |
| templates, positions alternating | 1.000 | 1.000 | 0.000 | 16.13 | 0.000 |
| template of position 0 only | 1.000 | 1.000 | 0.000 | 16.13 | 0.500 |
| mean of the class's two templates | 1.000 | 0.000 | 0.542 | -68.51 | 0.500 |
<!-- /worked -->

A generator that is scored by a likelihood alone will be told that the copy machine is the best one ever built. Nothing in the first two columns separates the data from a program that has never seen a noisy picture; the third column does, and only because the true distribution is known well enough to say what the noise of a real picture looks like (a distance of about $0.25$ from the template, not $0$). The last column catches a collapse that every other column forgives. So the first decision of this project is made before any network exists: the world is built so that its distribution is known exactly, in closed form, and every generator in the chapter is judged by five measures of that distribution at once.

The world is small. A picture has $6 \times 6$ pixels. It is a vertical line, a horizontal line or a $2 \times 2$ block, in one of two positions, with Gaussian noise of standard deviation $0.25$ on every pixel, so the data are six Gaussians in 36 dimensions with a label on each. Six chapters of this volume built the parts of a generator for such data, one at a time and each on a problem small enough to solve exactly. This chapter asks you to build all of them at once, in a language of your choice, and to make them agree. You will write six pieces, and the milestones below take them in this order:

1. **A world with a known distribution**: the six templates, the draw of a picture, the five measures, and their values for the data.
2. **The exact field**: the velocity field of that mixture, given the class or not, as the yardstick for every network.
3. **A denoiser with a class and a clock**: a two-layer network that takes the picture, a class (or the null class) and the noise fraction, with its backward pass written by hand and checked.
4. **Training, sampling and guidance**: the flow-matching loss with the class withheld on a fraction of the items, Euler and Heun samplers, classifier-free guidance, and a comparison with the exact field.
5. **An autoencoder, and the same generator in its latent space**: a variational autoencoder, a standardized latent, the denoiser retrained on the codes, and the decoder that returns pictures.
6. **The bill**: the cost of every stage in FLOPs and bytes.

Nothing in the list is new. Chapter 2 derived the autoencoder and its bound, Chapters 5 and 6 the objective, Chapter 7 the samplers and the guidance, Chapter 8 the way a denoiser is told the noise level, and Chapter 10 the way a latent is standardized, handed to the denoiser and decoded. `scripts/book/worked/project_image_generator.py`, the module that produced every number below, imports the worked modules of the volume instead of copying them: the book's generator and the stable-digits rule, the Adam update of Volume III, the timestep embedding of Chapter 8, and the scalar Euler and Heun steps of Chapter 7, against which the vector samplers of this chapter are checked. What is written here is what makes those pieces one generator.

![Two lanes of boxes. Training: a picture of 36 values goes through the encoder and the standardization to a code of 4 values, the straight path mixes it with noise, the denoiser takes the mixed code with a class or the null class, and a squared error compares its output with the target. Sampling: a noise vector of 4 values goes through 16 guided Euler steps with a class and a scale, the standardization is undone, the decoder gives the mean picture, and pixel noise is added.](img/generator-pipeline.svg)

The figure is the end state, the latent system of milestone 5. The pixel system of milestones 3 and 4 is the same picture without the encoder, the standardization and the decoder: a denoiser whose input and output are the 36 pixels. Build and check each milestone before the next. A generator whose denoiser has a wrong gradient trains to something, and the only way to find out that it is the wrong something is a number that every stage must satisfy; the milestones end in such numbers, and the **acceptance tests** at the end collect them in one place. You need the book's generator (Volume I, Chapter 17) and double-precision arithmetic with a correctly rounded `exp`, `log`, `sqrt`, `sin`, `cos` and `tanh`.

Seven questions bring back what the milestones lean on. Answer each before reading on; the answers are at the end.

1. (Volume VIII, [Chapter 6](06-flow-matching.md), the straight path.) For $x_0 = 1$, noise $\varepsilon = -1$ and $\sigma = 0.25$, what are $x_\sigma$ and the velocity target? Where does one Euler step with the exact field from noise fraction $\sigma$ to $0$ land?
2. (Volume VIII, [Chapter 7](07-samplers-and-guidance.md), guidance.) Write the guided field $u_g$ in terms of the fields with and without the prompt. What are $u_0$ and $u_1$, how many passes of the network does one step cost, and if the conditional branch is off by $\delta$, by how much is $u_g$ off?
3. (Volume VIII, [Chapter 2](02-vae.md), the bound.) For a decoder $\mathcal{N}(g(\mathbf{z}), \sigma_{\mathrm{x}}^2\mathbf{I})$ and an encoder $\mathcal{N}(\mathbf{m}, e^{\mathbf{v}})$, what is the negative bound of one picture, and how is $\mathbf{z}$ drawn so that the gradient reaches the encoder?
4. (Volume VIII, [Chapter 10](10-latent-diffusion-systems.md) and [Chapter 2](02-vae.md), the latent's scale.) Why is the encoder's mean not handed to the denoiser as it is, and what is done to it before the decoder sees it again?
5. (Volume VIII, [Chapter 8](08-denoiser-architectures.md), the noise level.) A layer computes $w\,x + \delta(\sigma)$ with a fixed gain $w$. What does the best denoiser of one cluster ask for that this layer cannot give, and what form of layer gives it?
6. (Volume III, [Chapter 6](../v03-neural-networks/06-gradient-checking.md), gradient checks.) What is compared with what, with which step, and what number says that a hand-written backward pass is right?
7. (Volume VIII, [Chapter 1](01-generative-modeling.md), what a model is for.) A model of images offers a sampler, a density, or both. What did the first chapter say a density is needed for, and why is a sample not enough to judge a model?

## Building the generator in six steps

The pseudocode follows [Pseudocode Conventions](../pseudocode.md). Every random input comes from the book's SplitMix64 at a seed printed beside its table and derived by the rule `seed_of(name)`; the draws are taken in the order the text states. A dot product may be accumulated in any order that is at least as accurate as left to right: no printed digit of an untrained number depends on the last bit of a sum, and the trained numbers are printed only to the digits that survive the nudge rule of this book (a trained table prints the digits that a run and four copies of it, nudged at their first weight, share; Chapter 2 of this volume states it with its first trained tables, and `scripts/book/worked/stable.py` is the code that applies it). The run is repeated from four starting points whose first weight differs by $+10^{-12}$, $-10^{-12}$, $+3 \times 10^{-12}$ and $-3 \times 10^{-12}$, and a digit is printed only if it is the same in all five runs. (Here all five agree on every digit shown; the runs are stable, and the rule is what lets the table say so.)

### Milestone 1: a world with a known distribution

A picture is a vector $\mathbf{x}$ of $D = 36$ pixels in row-major order. The noise-free **template** of class $c$ at position $p$ is $+1$ on the shape and $-1$ elsewhere: for the class column, every pixel of column $1 + 3p$; for the class row, every pixel of row $1 + 3p$; for the class block, the $2 \times 2$ square whose top-left pixel is $(1 + 2p, 1 + 2p)$. Index the six templates by $m = 2c + p$. The two lines of one class are three pixels apart, the two blocks do not overlap, and a block overlaps one line of each of the other classes in two pixels, so that classes are close but not confusable at the noise of the data.

```text
function template(c: int, p: int) -> array<real>[36]:
    require 0 <= c and c < 3 and 0 <= p and p < 2, "class and position in range"
    t = zeros(real, [36])
    for r in 0 .. 6:
        for q in 0 .. 6:
            on = (q == 1 + 3 * p) if c == 0 else
                 (r == 1 + 3 * p) if c == 1 else
                 (1 + 2 * p <= r and r < 3 + 2 * p and 1 + 2 * p <= q and q < 3 + 2 * p)
            t[6 * r + q] = 1.0 if on else -1.0
    return t

function draw_picture(rng: Generator) -> (int, int, array<real>[36]):
    c = rng.integer(3)
    p = rng.integer(2)
    t = template(c, p)
    x = zeros(real, [36])
    for i in 0 .. 36:                       # row-major, one normal() per pixel, in order
        x[i] = t[i] + 0.25 * rng.normal()
    return (c, p, x)
```

Given the class, the picture is drawn from $\tfrac12\sum_{p}\mathcal{N}(\mathbf{t}_{2c+p},\ \sigma_{\mathrm{x}}^2\mathbf{I})$ with $\sigma_{\mathrm{x}} = 0.25$, and without it from the average of all six. That is the whole distribution, and the project never needs more than the six vectors, the noise and the class prior of $\tfrac13$, which are exact and easy to check against a reader's own draws.

<!-- worked: project_image_generator.world -->
The six templates of the world, 6 x 6 pixels, `#` is +1 and `.` is -1 (row 0 at the top). Class c is 0 column, 1 row, 2 block; position p is 0 or 1; template index 2 c + p.

| column 0 | column 1 | row 0 | row 1 | block 0 | block 1 |
|---|---|---|---|---|---|
| `.#....` | `....#.` | `......` | `......` | `......` | `......` |
| `.#....` | `....#.` | `######` | `......` | `.##...` | `......` |
| `.#....` | `....#.` | `......` | `......` | `.##...` | `......` |
| `.#....` | `....#.` | `......` | `......` | `......` | `...##.` |
| `.#....` | `....#.` | `......` | `######` | `......` | `...##.` |
| `.#....` | `....#.` | `......` | `......` | `......` | `......` |

A picture is drawn from `project_image_generator.data` (seed 0x8d60cb533611bc47): c = integer(3), p = integer(2), then 36 values normal() in row-major order, pixel = template + 0.25 normal(). The first four pictures of the stream, with the sum of their 36 pixels, their first pixel and their root-mean-square distance to the nearest template per pixel:

| draw | class, position | sum of pixels | first pixel | distance |
|---|---:|---:|---:|---:|
| 0 | block 1 | -29.451198 | -0.678722 | 0.269474 |
| 1 | row 0 | -22.168839 | -0.935950 | 0.241363 |
| 2 | block 0 | -28.870138 | -1.001433 | 0.245452 |
| 3 | row 0 | -23.610533 | -0.935053 | 0.203274 |
<!-- /worked -->

The reader's own draws should agree with the first four pictures of the stream to six digits. They are the first test of the generator of Volume I, Chapter 17, and of the order of the draws: the class, then the position, then the pixels in row-major order. A transposed image or a position drawn after the pixels moves the sum of pixels at once.

Five measures judge a set of pictures against this distribution, and the opening table shows why there are five. *Accuracy* is the share of the requested class: the class of the template nearest to a sample, in squared distance (the lowest index on a tie), against the class that was asked for. **On data** is the share of samples whose root-mean-square distance per pixel to that nearest template is below $1.5\,\sigma_{\mathrm{x}} = 0.375$; the data have a mean distance of about $0.25$ with a standard deviation near $0.03$, and the distance crosses $0.375$ for about one picture in 30,000. **Distance** is the mean of that distance. **Mean $\ln p$** is the mean over samples of the log of the true density of the sample given its requested class,

$$
\ln p(\mathbf{x} \mid c) = \ln\!\Bigl(\tfrac12\sum_{p}\mathcal{N}\bigl(\mathbf{x};\ \mathbf{t}_{2c+p},\ \sigma_{\mathrm{x}}^2\mathbf{I}\bigr)\Bigr),
$$

whose expected value for a picture drawn from the class is

$$
\mathbb{E}\,\ln p(\mathbf{x}\mid c) = -\tfrac{D}{2}\bigl(1 + \ln 2\pi\sigma_{\mathrm{x}}^2\bigr) - \ln 2,
$$

and this value, $-1.87$, is what the data should give (a sample of 300 gives $-2.02$: the density of one picture has a standard deviation of $\sqrt{D/2} = 4.2$ about its mean, so the mean of 300 has a standard error of $0.24$). **TV** is the total variation distance of the positions of the samples requested in one class from the even split, $\tfrac12\sum_p |\mathrm{share}_p - \tfrac12|$, averaged over the three classes; it is $0$ for a generator that covers both positions equally and $0.5$ for one that covers only one. With 100 samples per class the data themselves have a TV near $0.04$, because a fair coin tossed 100 times is not split 50 to 50, and the table gives $0.050$.

One distinction is worth keeping. The density in the 'mean $\ln p$' column is a function of the true distribution that the evaluator computes, and never of the network's own opinion of its samples; the network has none. This is the difference between sampling and density that Chapter 1 drew: a flow-matching network provides a sampler only, and the evaluator brings the density.

**Milestone 1 ends** when your draws agree with the stream (the first picture is a block at position 1 with the sum of its pixels $-29.451198$) and your five measures of the 300 pictures of the data row agree with the first row of the opening table. The data row draws class by class from the same stream, with the class given and not drawn: for each class in turn, 100 times a position `integer(2)` and then the 36 pixels in order.

### Milestone 2: the exact field, held up as the yardstick

For this mixture everything that Chapters 5 to 7 computed for three clusters on a line can be computed for six points in 36 dimensions, and it is the reason the project is built on them. Take the path of Chapter 6, $\mathbf{x}_\sigma = (1-\sigma)\mathbf{x}_0 + \sigma\boldsymbol{\varepsilon}$, and write $a = 1 - \sigma$ and $\varrho = a^2\sigma_{\mathrm{x}}^2 + \sigma^2$. Given the template, $\mathbf{x}_\sigma$ is Gaussian with mean $a\,\mathbf{t}_m$ and covariance $\varrho\,\mathbf{I}$, so the posterior of the template at the point $\mathbf{x}$ is a softmax over the members of the class (all six for the null class),

$$
\mathrm{resp}_m(\mathbf{x}, \sigma) = \frac{\exp\bigl(-\lVert\mathbf{x} - a\,\mathbf{t}_m\rVert^2 / 2\varrho\bigr)}{\sum_{m'}\exp\bigl(-\lVert\mathbf{x} - a\,\mathbf{t}_{m'}\rVert^2 / 2\varrho\bigr)},
$$

and the posterior mean of the clean picture, within one template, is the Gaussian formula of Chapter 8 with a gain $a\,\sigma_{\mathrm{x}}^2/\varrho$:

$$
\hat{\mathbf{x}}_0(\mathbf{x}, \sigma, c) = \sum_m \mathrm{resp}_m\Bigl(\mathbf{t}_m + \frac{a\,\sigma_{\mathrm{x}}^2}{\varrho}\,(\mathbf{x} - a\,\mathbf{t}_m)\Bigr), \qquad
\mathbf{u}(\mathbf{x}, \sigma, c) = \frac{\mathbf{x} - \hat{\mathbf{x}}_0}{\sigma}.
$$

This is exactly the field of the clusters in Chapter 10, with the class weights now one-hot on the class's templates, and it is the best any network can do: it is the conditional expectation that squared error converges to. The null class is the uniform mixture of all six, which is also what a network trained with the class dropped on a fraction of its items learns to be (Chapter 7).

```text
function exact_velocity(x: array<real>[36], s: real, c: int) -> array<real>[36]:
    require 0.0 < s and s <= 1.0, "the velocity is a quotient by sigma"
    require 0 <= c and c <= 3, "class 3 is the null class"
    a = 1.0 - s
    v = a * a * 0.0625 + s * s
    members = [0, 1, 2, 3, 4, 5] if c == 3 else [2 * c, 2 * c + 1]
    resp = softmax([-sum((x[i] - a * template(m // 2, m mod 2)[i])^2 for i in 0 .. 36) / (2.0 * v)
                    for m in members])
    guess = zeros(real, [36])
    for k in 0 .. len(members):
        t = template(members[k] // 2, members[k] mod 2)
        for i in 0 .. 36:
            guess[i] = guess[i] + resp[k] * (t[i] + (a * 0.0625 / v) * (x[i] - a * t[i]))
    return [(x[i] - guess[i]) / s for i in 0 .. 36]
```

Five checks tie it to what the chapters proved, and the module runs every one before any network exists:

<!-- worked: project_image_generator.exact_checks -->
Checks of the exact field on points drawn from `project_image_generator.reference` (seed 0xeb522f8dc288c023) (each point is 36 values by normal() in order; four points, then the five-value start of the sampler check):

| identity | largest difference |
|---|---:|
| at sigma = 1 the field of class c is x minus the mean of the class's templates (and of all six for the null class), 16 cases | below 1e-12 |
| one Euler step from sigma = 0.3 or 0.7 to 0 lands on the guess E[x_0 given x_sigma, class 1], 8 cases | below 1e-12 |
| unconditional field = sum over classes of p(class given x) times the class field, sigma = 0.5, 4 cases | below 1e-12 |
| class-1 field = -(x + sigma s)/(1 - sigma), s the central difference (step 1e-5) of the log density, sigma 0.4 and 0.8, 16 cases | below 1e-06 |
| vector Euler and Heun against the scalar ones of the sampler chapter on a separable guess, 4 and 16 steps | below 1e-12 |
<!-- /worked -->

The first row is the end of the path that has no signal left: at $\sigma = 1$ the field is $\mathbf{x}$ minus the mean of the members. The second is Chapter 6's remark that one Euler step with the exact field lands on the posterior mean. The third is the identity of Chapter 7 by which the field of the unconditional model is the class fields weighted by the probability of each class at the point. The fourth is the identity of Chapter 6 between velocity and score, with the score taken by central differences of the log of the noised mixture. The last row is the one that ties this chapter's vector samplers to the book: your `euler` and `heun` on a coordinate-wise separable guess must equal the scalar ones that the sampler chapter's table was computed with.

With the field in hand, the first generator of the project is a sampler of the exact field, and it can be scored on the five measures. This is a reference and a calibration at once. The reference: no trained network can do better than the field it approximates. The calibration: it shows what a solver alone costs.

<!-- worked: project_image_generator.exact_reference -->
The exact field as a generator: 100 starting vectors in all (the first 100 draws of 36 values each by normal() from `project_image_generator.sample` (seed 0x79bf556f967e3017), the same for every row and for the networks later), each carried to a picture of each of the three classes, so that a row has 300 samples, class 0's first. They are carried from sigma = 1 to 0 by the exact class field (g = 1), by the guided field u_un + g (u_c - u_un) and by 100 Runge-Kutta steps (the classical rule, with the field evaluated at max(sigma, 1e-9) in the last stage of the last step, where it is 0/0 at sigma = 0). For g = 0 the three classes get the same pictures.

| sampler | steps | accuracy | on data | distance | mean ln p | TV |
|---|---:|---:|---:|---:|---:|---:|
| the data, 300 pictures | - | 1.000 | 1.000 | 0.249 | -2.02 | 0.050 |
| exact field, RK4 | 100, g = 1 | 1.000 | 1.000 | 0.247 | -1.68 | 0.040 |
| exact field, Euler | 16, g = 0 | 0.333 | 1.000 | 0.219 | -147.13 | 0.040 |
| exact field, Euler | 16, g = 1 | 1.000 | 1.000 | 0.217 | 2.43 | 0.040 |
| exact field, Euler | 16, g = 2 | 1.000 | 1.000 | 0.215 | 2.63 | 0.047 |
| exact field, Euler | 16, g = 4 | 1.000 | 1.000 | 0.222 | 1.72 | 0.040 |
<!-- /worked -->

Three things stand out. A hundred Runge-Kutta steps of the exact field reproduce the data: the distance $0.247$ and the density $-1.68$ are the data's own, to within what 300 pictures can say. Sixteen Euler steps do not: the distance falls to $0.217$ and the density rises to $2.43$, and the cause is the Euler step and not the field. Each step lands on a point a little nearer to a template than the true path, so that the samples are cleaner than the data, and the density, which pays for cleanliness with a premium (the first table of the chapter), shows it. A sampler can fail by being too good. And the null rows show what the class column is for: at $g = 0$ the unconditional field gives pictures that are on the data, with accuracy $0.333$ and a very poor density *given the requested class* (the pictures are of other classes two times out of three); at $g = 1, 2, 4$ the accuracy is $1.000$, and the exact field does not oversaturate at $g = 4$ (the distance is $0.222$, the density $1.72$). That last fact will matter in a few pages.

**Milestone 2 ends** when the five identities pass and the exact-field rows of your own table agree to the stated digits: the 100-step row to the data's, and the Euler rows to $0.001$ in the distance and $0.01$ in the density.

### Milestone 3: a denoiser with a class and a clock

The generator's network takes a picture $\mathbf{x}_\sigma$ of $d$ values, the noise fraction $\sigma$ and a class $c \in \{0, 1, 2, 3\}$ (3 is the null class), and returns a velocity of $d$ values. It is a small instance of what Chapter 8 argued for. Two hidden layers of $H = 48$ units with $\tanh$; the class enters as a learned embedding of four values, concatenated to the picture; the noise level enters by scale and shift (adaptive layer normalization in its simplest form), because Chapter 8 showed that a layer which only adds a function of $\sigma$ to its pre-activations keeps one gain for every noise level, and the best gain of the best denoiser falls from near $1$ at a small $\sigma$ to near $0$ at a large one. The features of $\sigma$ are four numbers, the sinusoidal embedding of Chapter 8 with two rates (base 4) applied to $\pi\sigma$:

$$
\mathrm{tf}(\sigma) = \bigl[\cos\pi\sigma,\ \cos\tfrac{\pi\sigma}{2},\ \sin\pi\sigma,\ \sin\tfrac{\pi\sigma}{2}\bigr].
$$

For each hidden layer $\ell$ with pre-activation $\mathbf{z}_\ell$ (a matrix product plus a bias), the layer's gain is $\mathbf{k}_\ell = \mathbf{1} + \mathbf{s}_\ell$ and its shift $\mathbf{r}_\ell$, both linear in $\mathrm{tf}(\sigma)$ with biases of their own: $\mathbf{h}_\ell = \tanh\bigl(\mathbf{k}_\ell \odot \mathbf{z}_\ell + \mathbf{r}_\ell\bigr)$. The weights that compute $\mathbf{s}_\ell$ and $\mathbf{r}_\ell$ start at zero, so that a new network is exactly the plain network of Volume III and learns what the noise level does to it, in the spirit of the zero-initialized gate of Chapter 8.

```text
function denoise(P: record, x: array<real>[d], s: real, c: int) -> array<real>[d]:
    tf = [cos(pi * s), cos(pi * s / 2), sin(pi * s), sin(pi * s / 2)]
    a0 = zeros(real, [d + 4])
    a0[0:d] = x                                      # the picture first
    a0[d:d + 4] = P.E[c, :]                          # then the class embedding (row 3: null)
    z1 = [dot(P.W1[j, :], a0) + P.b1[j] for j in 0 .. 48]
    h1 = [tanh((1.0 + dot(P.G1[j, :], tf) + P.c1[j]) * z1[j] + dot(P.B1[j, :], tf) + P.e1[j])
          for j in 0 .. 48]
    z2 = [dot(P.W2[j, :], h1) + P.b2[j] for j in 0 .. 48]
    h2 = [tanh((1.0 + dot(P.G2[j, :], tf) + P.c2[j]) * z2[j] + dot(P.B2[j, :], tf) + P.e2[j])
          for j in 0 .. 48]
    return [dot(P.W3[k, :], h2) + P.b3[k] for k in 0 .. d]
```

The class table is $[4, 4]$, the first weight matrix $[48, d + 4]$, the second $[48, 48]$, the third $[d, 48]$, and each modulation matrix $[48, 4]$ with its bias vector of 48. The pixel network has $d = 36$.

<!-- worked: project_image_generator.network -->
The pixel denoiser's tensors in their flat order (D = 36, H = 48, cond width 4, 4 timestep features).

| tensor | shape | values |
|---|---:|---:|
| `E` | 4 x 4 | 16 |
| `W1` | 48 x 40 | 1,920 |
| `b1` | 48 | 48 |
| `G1` | 48 x 4 | 192 |
| `c1` | 48 | 48 |
| `B1` | 48 x 4 | 192 |
| `e1` | 48 | 48 |
| `W2` | 48 x 48 | 2,304 |
| `b2` | 48 | 48 |
| `G2` | 48 x 4 | 192 |
| `c2` | 48 | 48 |
| `B2` | 48 x 4 | 192 |
| `e2` | 48 | 48 |
| `W3` | 36 x 48 | 1,728 |
| `b3` | 36 | 36 |

Total: 7,060 values. Drawn from `project_image_generator.run` (seed 0xa58a7dbfeedcbe87) in the order E, W1, W2, W3 (E by normal(), W1 by normal() / sqrt(40), W2 and W3 by normal() / sqrt(48), each row-major); the biases and the modulation weights are 0. A forward pass of the initial network at x = the first starting vector of `project_image_generator.sample` (seed 0x79bf556f967e3017), sigma = 0.5, class 1 gives the outputs 0.501486, -0.325876, 0.646063, -0.335603 for the first four pixels and a root-mean-square of 0.460927 over all 36. One forward pass costs 14,340 FLOPs.

Gradient check (`project_image_generator.gradcheck` (seed 0x04db3727cdb1d417): the parameters of the initialization plus 0.1 normal() each in flat order, then 3 items by `draw_training_item`): every one of the 7,060 partial derivatives of the batch loss against the central difference of step 1e-5, the largest difference divided by the largest partial derivative is below 1e-06. The autoencoder's 2,108 partial derivatives, on one picture and one draw of xi (`project_image_generator.ae_gradcheck` (seed 0x00a2c73c40798adc)), the same: below 1e-06.
<!-- /worked -->

A backward pass has to be written by hand, in the same sense as in Volume III, and the check against central differences has to pass before any training run is believed. The chain rule through $\mathbf{h} = \tanh(\mathbf{k}\odot\mathbf{z} + \mathbf{r})$ is the one worth writing out because it is the new line: with $\mathrm{d}\mathbf{h}$ the gradient that arrives at the output and $\mathrm{d}\mathbf{u} = \mathrm{d}\mathbf{h}\odot(1 - \mathbf{h}^2)$, the gradient of the pre-activation is $\mathrm{d}\mathbf{z} = \mathrm{d}\mathbf{u}\odot\mathbf{k}$, that of the gain is $\mathrm{d}\mathbf{k} = \mathrm{d}\mathbf{u}\odot\mathbf{z}$ and that of the shift is $\mathrm{d}\mathbf{u}$ itself. The gain's gradient then goes to the weights of the modulation through $\mathrm{tf}$, and $\mathrm{d}\mathbf{z}$ goes through the matrix product to the layer below, down to the class embedding, of which only the row of the class that was used receives a gradient. The check is done at perturbed weights, not at the initialization, because at the initialization every gain is exactly 1, so that a backward pass which forgot to multiply by the gain would still agree with the finite differences. At the perturbed point every one of the 7,060 derivatives of a three-item loss agrees with its central difference to better than $10^{-6}$ of the largest, and so do the autoencoder's 2,108 in milestone 5.

**Milestone 3 ends** when the check passes for every parameter (not for a sample of them), the count is $7{,}060$ values and the initial forward pass reproduces the four printed outputs to $10^{-6}$. A wrong number here is usually a draw-order mistake: the class table is drawn first, then the first matrix, the second and the third, each in row-major order; the matrices of modulation and every bias are not drawn at all.

### Milestone 4: training with the class sometimes withheld, then sampling and guiding

The objective is the one of Chapter 6, in the vector form and averaged over the pixels, and the one change that makes guidance possible is the one of Chapter 7: on a fraction of the items the class is replaced by the null class, so that one network is both the conditional and the unconditional model.

$$
\mathcal{L}(\theta) = \mathbb{E}\;\frac{1}{D}\,\bigl\lVert\mathbf{u}_\theta(\mathbf{x}_\sigma, \sigma, \tilde c) - (\boldsymbol{\varepsilon} - \mathbf{x}_0)\bigr\rVert^2, \qquad \tilde c = \begin{cases} 3 & \text{with probability } p_{\mathrm{drop}} = 0.15,\\ c & \text{otherwise,}\end{cases}
$$

with $\sigma$ uniform on $[0, 1]$. One item is drawn in a fixed order from one stream, so that a reader's loop consumes the generator exactly as the module does.

```text
function training_item(rng: Generator, encode: function(array<real>[36]) -> array<real>[d])
        -> (real, array<real>[d], array<real>[d], int):
    s = rng.uniform()
    c = rng.integer(3)
    p = rng.integer(2)
    drop = rng.uniform()                              # drawn for every item, used or not
    t = template(c, p)
    pic = zeros(real, [36])
    for i in 0 .. 36:
        pic[i] = t[i] + 0.25 * rng.normal()
    x0 = encode(pic)                                  # the identity for the pixel system
    eps = [rng.normal() for i in 0 .. d]
    x = [(1.0 - s) * x0[i] + s * eps[i] for i in 0 .. d]
    target = [eps[i] - x0[i] for i in 0 .. d]
    return (s, x, target, 3 if drop < 0.15 else c)
```

A step draws 32 items one after another, forms the mean over items and pixels of the squared error, and takes an Adam step (the update of Volume III, Chapter 7: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, no weight decay, bias correction) at rate $10^{-2}(1 - (i-1)/2000)$ at step $i = 1, \dots, 2000$. The first parameter, $\mathbf{E}[0, 0]$, is moved by the nudge before step 1; it is read whenever class 0 appears, so that a nudge changes the computation. The network is trained from `project_image_generator.run` after its initialization, and the evaluation is not on training items but on 600 fresh ones, scored against the exact field that Milestone 2 built: three numbers per row, the loss of the network and of the exact field on the same items, and the mean of $(\text{network} - \text{exact})^2$, which is the gap that matters.

<!-- worked: project_image_generator.training_table -->
The pixel denoiser trained 2000 steps of 32 items from `project_image_generator.run` (seed 0xa58a7dbfeedcbe87) after its initialization, Adam (beta1 0.9, beta2 0.999, epsilon 1e-8, no weight decay) at rate 0.01 (1 - (i - 1)/2000) at step i. After each listed step the network and the exact field are scored on 600 fresh items (`project_image_generator.evaluation` (seed 0xf048a17861f336ee), drawn as in training, so about 15% have the null label): the mean over items of the mean over pixels of (output - target)^2 for the network and for the exact field, and of (network - exact)^2. Digits are those shared by the run and four copies of it with the first weight changed by +1e-12, -1e-12, +3e-12 and -3e-12 (the rule of `stable.py`); here all five agree on every digit shown.

| step | loss of the network | loss of the exact field | mean (network - exact)^2 |
|---|---:|---:|---:|
| 250 | 0.7717 | 0.5036 | 0.2594 |
| 500 | 0.6765 | 0.5036 | 0.1667 |
| 1000 | 0.6245 | 0.5036 | 0.1072 |
| 1500 | 0.5898 | 0.5036 | 0.0768 |
| 2000 | 0.5686 | 0.5036 | 0.0569 |

The same items after the last step, split by the noise fraction sigma of the item (lower end included, upper end excluded).

| sigma | items | loss of the exact field | loss of the network | mean (network - exact)^2 |
|---|---:|---:|---:|---:|
| 0 .. 0.2 | 109 | 1.0351 | 1.0854 | 0.0444 |
| 0.2 .. 0.4 | 123 | 0.5167 | 0.5920 | 0.0664 |
| 0.4 .. 0.6 | 111 | 0.2472 | 0.3213 | 0.0606 |
| 0.6 .. 0.8 | 122 | 0.3278 | 0.4160 | 0.0736 |
| 0.8 .. 1 | 135 | 0.4320 | 0.4712 | 0.0403 |
<!-- /worked -->

The loss of the exact field is $0.5036$ and does not move: it is the least loss any network can reach (Chapter 6), the noise in the target that no amount of training removes. The network's loss falls from $0.7717$ at step 250 to $0.5686$ at step 2,000, and its gap to the exact field falls at every logged step ($0.2594$, $0.1667$, $0.1072$, $0.0768$, $0.0569$), still falling when the rate reaches zero. The gap by noise level shows where the work is left: between $0.040$ and $0.074$ in every band, so that no band is neglected. The loss itself is highest, for the network and for the exact field alike, at the smallest $\sigma$ ($1.0854$ and $1.0351$), where the noise is a small part of the point and $\boldsymbol{\varepsilon}$ can hardly be read off it.

The sampler is the pseudocode of Chapter 7 in vector form: one function that evaluates the guided field and one loop.

```text
function guided_velocity(field: function(array<real>[D], real, int) -> array<real>[D],
                         x: array<real>[D], s: real, c: int, g: real) -> array<real>[D]:
    if g == 1.0:
        return field(x, s, c)                         # the blend is the conditional branch itself
    un = field(x, s, 3)
    if g == 0.0:
        return un
    cond = field(x, s, c)
    return [un[i] + g * (cond[i] - un[i]) for i in 0 .. D]

function euler_sample(field, start: array<real>[D], c: int, g: real, n: int) -> array<real>[D]:
    x = copy(start)
    for i in 0 .. n:
        s = 1.0 - i / n                               # a uniform grid from 1 down to 0
        t = 1.0 - (i + 1) / n
        u = guided_velocity(field, x, s, c, g)
        x = [x[k] + (t - s) * u[k] for k in 0 .. D]
    return x
```

Heun's method replaces the update by the average of the velocity at the start and at the end of an Euler step (the end evaluated at $\max(t, 10^{-9})$, because the exact field is $0/0$ at $\sigma = 0$), at twice the evaluations per step. The reference that every sampler is scored against is the exact class field, integrated by 100 steps of the classical Runge-Kutta rule from the same starting vector, whose last stage at the end of the last step is evaluated at $\max(0, 10^{-9})$ for the same reason.

The first sampling table asks which part of the error is the solver's and which the network's. It varies the number of evaluations of the field, for Euler and for Heun, with the exact field and with the trained network; the score is the mean distance per pixel to the 100-step reference.

<!-- worked: project_image_generator.solver_table -->
Class-conditional sampling (g = 1) without the null pass. For each sampler and each count of evaluations of the field, the mean over 150 samples (the first 50 of the 100 starting vectors of `project_image_generator.sample` (seed 0x79bf556f967e3017), used for each of the three classes) of the root-mean-square distance per pixel between the sampler's end point and the 100-step Runge-Kutta end point of the exact field from the same start. Euler takes n steps, Heun n/2 (two evaluations each). The exact-field columns are plain arithmetic; the network columns share digits across the five runs.

| evaluations | Euler, exact field | Euler, network | Heun, exact field | Heun, network |
|---|---:|---:|---:|---:|
| 4 | 0.1174 | 0.2110 | 0.0727 | 0.2245 |
| 8 | 0.0623 | 0.1766 | 0.0145 | 0.1760 |
| 16 | 0.0326 | 0.1675 | 0.0057 | 0.1678 |
| 32 | 0.0168 | 0.1660 | 0.0018 | 0.1669 |
<!-- /worked -->

The exact field is the table of Chapter 7 again. Euler's distance halves each time the steps double ($0.1174$, $0.0623$, $0.0326$, $0.0168$), and Heun's falls much faster, to $0.0018$ at 32 evaluations. The network's column is a different story. Its distance stops falling at $0.17$: $0.1766$ at 8 evaluations, $0.1675$ at 16, $0.1660$ at 32. Beyond about 8 evaluations the sampler is not the limit and the network is, because the network's field differs from the exact one by the gap of the training table, and an end point follows the field it is given. At 4 evaluations Heun is *worse* than Euler for the network ($0.2245$ against $0.2110$), although it is better for the exact field ($0.0727$ against $0.1174$). A more accurate solver is a better solver of the network's field, and it is not a better generator when the field is wrong in a way the extra accuracy does not repair. The lesson for choosing a sampler is that it is chosen against the trained network and its error, and never against the exact solution of a clean toy.

<!-- worked: project_image_generator.guidance_table -->
Guided Euler sampling in pixel space, 16 steps, the 100 starting vectors of `project_image_generator.sample` (seed 0x79bf556f967e3017) used for each of the three classes, the same for the network and the exact field. Digits are those shared by the run and four copies of it with the first weight changed by +1e-12, -1e-12, +3e-12 and -3e-12 (the rule of `stable.py`); here all five agree on every digit shown.

| sampler | accuracy | on data | distance | mean ln p | TV |
|---|---:|---:|---:|---:|---:|
| network, g = 0 | 0.333 | 0.680 | 0.335 | -127.45 | 0.080 |
| exact field, g = 0 | 0.333 | 1.000 | 0.219 | -147.13 | 0.040 |
| network, g = 1 | 0.990 | 0.930 | 0.286 | -10.70 | 0.023 |
| exact field, g = 1 | 1.000 | 1.000 | 0.217 | 2.43 | 0.040 |
| network, g = 2 | 1.000 | 0.890 | 0.317 | -16.10 | 0.023 |
| exact field, g = 2 | 1.000 | 1.000 | 0.215 | 2.63 | 0.047 |
| network, g = 4 | 1.000 | 0.083 | 0.505 | -65.67 | 0.107 |
| exact field, g = 4 | 1.000 | 1.000 | 0.222 | 1.72 | 0.040 |
<!-- /worked -->

The guidance table is the five measures at $g = 0, 1, 2, 4$ for the network and for the exact field, from the same starting vectors. At $g = 0$ the network's accuracy is $0.333$, which is the right answer: it is the unconditional model, and it gives each class a third of the pictures whatever was requested. At $g = 1$ the network is correct on $0.990$ of the pictures, a share $0.930$ of them is on the data, and the density is $-10.70$ against the data's $-2.02$ (about a tenth of the pictures are off the data, and a picture that is off costs the density a lot). At $g = 2$ accuracy is $1.000$ and the share on the data is $0.890$. At $g = 4$ the generator has collapsed: accuracy $1.000$, but $0.083$ on the data, a distance of $0.505$ and a density of $-65.67$. The exact field, for the same $g = 4$, is on the data in $1.000$ of the cases.

The pictures show what the numbers mean.

![A grid of 6 x 6 grey pictures. Columns are two pictures each of the classes column, row and block; rows are the data, the exact field with 100 Runge-Kutta steps, the pixel network at guidance 1 and 4, and the latent system at guidance 1 and 4. Every row shows a white line or block on a dark noisy background; the pixel network at guidance 4 has the brightest shapes and grey specks in the background.](img/pictures-compared.svg)

At $g = 1$ the network's pictures are close to the exact field's. At $g = 4$ the shapes of the pixel network are brighter than any picture of the data and its background carries grey specks; the figure clips the values to $[-1, 1]$, so the brightness it shows is a lower bound. The numbers behind the picture:

<!-- worked: project_image_generator.saturation_table -->
Saturation of the pictures of the guidance tables at g = 1 and g = 4 (the same 300 samples as there). For the pixels on the shape of the nearest template: their mean and the share above 1 + sigma_x = 1.25; for the background pixels (the template's -1): the share above 0. The data's own shares are those of a Gaussian: 0.159 above one standard deviation of the shape's value, and about 3e-5 of the background above 0. Digits are those shared by the run and four copies of it with the first weight changed by +1e-12, -1e-12, +3e-12 and -3e-12 (the rule of `stable.py`); here all five agree on every digit shown.

| system | g | mean on shape | share on shape above 1.25 | share of background above 0 |
|---|---:|---:|---:|---:|
| the data, 300 pictures | - | 0.994 | 0.153 | 0.000 |
| pixel space | 1 | 0.912 | 0.140 | 0.002 |
| pixel space | 4 | 1.381 | 0.626 | 0.026 |
| latent space | 1 | 0.980 | 0.158 | 0.000 |
| latent space | 4 | 1.034 | 0.210 | 0.000 |
<!-- /worked -->

The pixels on the shape average $0.994$ in the data and $0.912$ at $g = 1$, and $1.381$ at $g = 4$; $0.626$ of them lie more than one pixel-noise deviation above the template's value (the data: $0.153$), and $0.026$ of the background pixels are above $0$ (the data: about $3 \times 10^{-5}$). It is the oversaturation of Chapter 7, which there was a number on a line and is here a picture. The latent system shows a milder form of it ($1.034$ and $0.210$ at $g = 4$). The chapter's claim was that a guided field multiplies an error of one branch by $g$; here is the check on the trained network, on the evaluation items that kept their class.

<!-- worked: project_image_generator.amplification_table -->
The error of the guided field of the trained pixel network against the guided exact field, on the 600 evaluation items of `project_image_generator.evaluation` (seed 0xf048a17861f336ee) that kept their class (the items with the null label are left out), each at its own class: the mean over items of the mean over pixels of (guided network - guided exact)^2. The last column divides by the row of g = 1 (digits as in the other trained tables, the ratio from the first run).

| g | mean squared difference | relative to g = 1 |
|---|---:|---:|
| 1 | 0.0492 | 1.0 |
| 2 | 0.1601 | 3.3 |
| 4 | 0.8895 | 18.1 |
<!-- /worked -->

The squared difference from the exact guided field grows from $0.0492$ at $g = 1$ to $0.1601$ at $g = 2$ and $0.8895$ at $g = 4$, which is $3.3$ and $18.1$ times as much, against the factors $4$ and $16$ of a squared multiplier. The unconditional branch, whose error enters with weight $1 - g$, contributes too, so the match is a rough one; the table says that the growth is of the order of $g^2$, and that the exact field's immunity at $g = 4$ is the immunity of a field with no error to amplify. The pixel generator at this size is usable at $g = 1$ and at $g = 2$, and a scale of $4$ is beyond the network's accuracy, not beyond the method's.

**Milestone 4 ends** when your training table agrees to its printed digits (the table prints only the digits that all five runs share, so a run in another language agrees when it matches them), the solver table shows the network's distance flat from 8 evaluations, and the guidance table reproduces the rows for $g = 1$ and $g = 4$.

### Milestone 5: an autoencoder, and the same generator in its latent space

The pixel generator works, but it is working at the dimension of the picture: 36 numbers enter and 36 leave every one of the 32 passes of a guided sample. Chapter 10's whole argument is that a picture is mostly redundant, and that the denoiser should work on a short code from an autoencoder. In this world the redundancy is exact, since there are six pictures and noise, and a latent of four numbers is plenty.

The autoencoder is the variational one of Chapter 2, with the decoder's standard deviation fixed at the true noise $\sigma_{\mathrm{x}}$. The encoder maps a picture to $\mathbf{m}$ and the log-variance $\mathbf{v}$ of a four-value latent through one hidden layer of 24 $\tanh$ units, and the decoder maps a code through one hidden layer of 24 $\tanh$ units back to 36 values. The loss of one picture is the negative bound,

$$
\mathcal{L}(\mathbf{x}) = \frac{\lVert\mathbf{x} - g(\mathbf{z})\rVert^2}{2\sigma_{\mathrm{x}}^2} + \frac12\sum_{k=1}^{4}\bigl(m_k^2 + e^{v_k} - 1 - v_k\bigr), \qquad \mathbf{z} = \mathbf{m} + e^{\mathbf{v}/2}\odot\boldsymbol{\xi},
$$

without the constant $\tfrac{D}{2}\ln(2\pi\sigma_{\mathrm{x}}^2)$. The draws are those of the pixel network's items without the denoiser's: a picture as in `draw_picture`, then the four values $\boldsymbol{\xi}$ by `normal()`. The weights are drawn in the order encoder hidden, encoder head, decoder hidden, decoder output (each `normal() / sqrt(fan_in)`, row-major, biases zero), the training is 2,000 Adam steps of 32 pictures at rate $3 \times 10^{-3}(1 - (i-1)/2000)$, and the check of the hand-written gradient is the one printed under milestone 3.

<!-- worked: project_image_generator.autoencoder_table -->
The autoencoder's layers (flat order We1, be1, We2, be2, Wd1, bd1, Wd2, bd2; 2,108 values).

| layer | map | values |
|---|---:|---:|
| encoder hidden | 36 -> 24, tanh | 888 |
| encoder head (mean, log-variance) | 24 -> 8 | 200 |
| decoder hidden | 4 -> 24, tanh | 120 |
| decoder output | 24 -> 36 | 900 |

Trained 2000 steps of 32 pictures from `project_image_generator.ae_run` (seed 0x1a383ff3791c023d) (after the initialization; per picture c, p and 36 pixel noises as for `draw_picture`, then 4 values xi by normal()), Adam at rate 0.003 (1 - (i - 1)/2000). On 600 fresh pictures (`project_image_generator.ae_eval` (seed 0xcb667270d708874d)) decoded from the mean code: mean squared error per pixel 0.0590 (the pixel noise alone has 0.0625), mean KL 7.23 nats per picture, and the six noise-free templates come back, each with its own root-mean-square error per pixel over its 36 pixels, the mean of the six being 0.021. The mean KL of a picture is 0.5 the sum over the four channels of m^2 + exp(v) - 1 - v for the mean m and log-variance v of its code. The 600 pictures and the 1200 pictures of the statistics are drawn as `draw_picture` and nothing else (c, p and the 36 pixel noises per picture, no xi). The mean and the variance (divisor 1200) of the four channels of the mean code over the 1200 pictures of `project_image_generator.ae_stat` (seed 0xee81d38b01540daf):

| channel | mean | variance |
|---|---:|---:|
| 0 | 0.896 | 0.473 |
| 1 | -0.327 | 1.032 |
| 2 | -0.036 | 1.073 |
| 3 | 0.460 | 1.088 |

Digits are those shared by the run and four copies of it with the first weight changed by +1e-12, -1e-12, +3e-12 and -3e-12 (the rule of `stable.py`); here all five agree on every digit shown.
<!-- /worked -->

The mean squared error per pixel is $0.0590$ and the noise alone has $0.0625$. A decoder that returned exactly the template would score $0.0625$ in expectation, so $0.0590$ says that the autoencoder reproduces a little of the individual noise of the pictures it sees (the difference is about six standard errors over 600 pictures), and the mean KL of $7.23$ nats per picture is the price. The six noise-free templates, which the autoencoder never saw without noise, come back with a root-mean-square error of $0.021$ per pixel. It is a good autoencoder for this world and a lossy one, as Chapter 2 and Chapter 10 said every autoencoder is: its decoder returns the mean picture and the texture of a particular draw is gone.

That last fact sets the rule for turning the latent system into a generator of the data's distribution. The decoder of the variational model is $\mathcal{N}(g(\mathbf{z}), \sigma_{\mathrm{x}}^2\mathbf{I})$ by construction (Chapter 2), so a *sample* of that model is the decoder's mean plus $\sigma_{\mathrm{x}}$ times a normal draw on every pixel. The generator of this milestone does that last step explicitly, from its own stream. The latent that the denoiser sees is the encoder's mean, standardized per channel by the mean and variance of the codes of 1,200 pictures (the table gives them), with the $10^{-4}$ of Chapter 10 added to the variance: $\tilde{\mathbf{z}} = (\mathbf{m} - \mathbf{c}_{\mathrm{shift}})/\sqrt{\mathbf{c}_{\mathrm{var}} + 10^{-4}}$, undone before the decoder by the inverse. The denoiser is the network of milestone 3 with $d = 4$, trained exactly as in milestone 4 on items whose picture is encoded before it is noised (the same draw order; the encoder runs on every item here because the pictures are drawn fresh, where a fixed dataset would be encoded once). Its sampler is the same function, started from four normal values, and its output goes through the inverse standardization, the decoder and the pixel noise.

<!-- worked: project_image_generator.systems_table -->
The two systems on the same measures. Each row: 100 starting vectors used for each class (so 300 samples), 16 guided Euler steps. Pixel space starts from `project_image_generator.sample` (seed 0x79bf556f967e3017). Latent space starts from `project_image_generator.latent_start` (seed 0xfb205b0c1f4b3468) (100 vectors of 4 values by normal() in order, used for each class), with a denoiser initialized and trained from `project_image_generator.latent_run` (seed 0xfa302f68fd4a5c00) like the pixel one from `project_image_generator.run` (seed 0xa58a7dbfeedcbe87) (the same items, with the picture of each item replaced by its standardized code, and the same 2,000 steps); decodes the standardized code with the autoencoder, and adds 0.25 normal() to every pixel from `project_image_generator.decoder_noise` (seed 0x8c2e2a3a1a80b680) (a fresh generator for every row, 36 draws per sample in order). Digits are those shared by the run and four copies of it with the first weight changed by +1e-12, -1e-12, +3e-12 and -3e-12 (the rule of `stable.py`); here all five agree on every digit shown.

| system | g | accuracy | on data | distance | mean ln p | TV |
|---|---:|---:|---:|---:|---:|---:|
| the data, 300 pictures | - | 1.000 | 1.000 | 0.249 | -2.02 | 0.050 |
| pixel space | 0 | 0.333 | 0.680 | 0.335 | -127.45 | 0.080 |
| pixel space | 1 | 0.990 | 0.930 | 0.286 | -10.70 | 0.023 |
| pixel space | 2 | 1.000 | 0.890 | 0.317 | -16.10 | 0.023 |
| pixel space | 4 | 1.000 | 0.083 | 0.505 | -65.67 | 0.107 |
| latent space | 0 | 0.333 | 0.877 | 0.293 | -147.87 | 0.017 |
| latent space | 1 | 1.000 | 0.993 | 0.257 | -3.30 | 0.037 |
| latent space | 2 | 1.000 | 0.997 | 0.254 | -2.88 | 0.043 |
| latent space | 4 | 1.000 | 0.990 | 0.258 | -3.40 | 0.093 |
<!-- /worked -->

At $g = 1$ the latent generator is nearer to the data than the pixel generator on four of the five measures: its accuracy is $1.000$ (the pixel network's $0.990$), $0.993$ of its pictures are on the data ($0.930$), the distance is $0.257$ ($0.286$; the data's $0.249$) and the density is $-3.30$ ($-10.70$; the data's $-2.02$). The exception is the TV, $0.037$ against $0.023$, both within what 100 draws per class can say. At $g = 2$ it is nearer still, with a density of $-2.88$, and at $g = 4$ it keeps $0.990$ on the data where the pixel network has $0.083$. Two caveats go with that last row. The decoder was trained on six kinds of pictures, and a code near the clusters is decoded to something like one of them, so the oversaturation that damages the pixel network is partly hidden from these measures (the saturation table shows a milder form of it); and the TV rises from $0.037$ at $g = 1$ to $0.093$ at $g = 4$ (against a floor near $0.04$ for 100 draws), which suggests that at $g = 4$ the positions of a class are no longer covered evenly. And the comparison is between two networks of the same recipe on 2,000 steps. It shows that the latent problem, in four dimensions, is easier for this network to learn than the pixel problem in 36; it does not show that a latent system always beats a pixel system at equal cost, and the ledger in milestone 6 counts the cost.

The standard ways a pipeline like this fails are all silent, in the sense that the program runs and produces pictures. Four are planted here, each measured as a generator; each fails a different measure and the other measures stay good, which is why the table has five columns.

<!-- worked: project_image_generator.mistakes_table -->
Four mistakes planted in the systems of the last tables, each measured as a generator (g = 1 and 16 steps unless the row says otherwise, the same starting values). Digits are those shared by the run and four copies of it with the first weight changed by +1e-12, -1e-12, +3e-12 and -3e-12 (the rule of `stable.py`); here all five agree on every digit shown.

| mistake | accuracy | on data | distance | mean ln p | TV |
|---|---:|---:|---:|---:|---:|
| pixel, one Euler step | 0.957 | 0.003 | 0.543 | -70.28 | 0.210 |
| pixel, guidance g = -1 | 0.037 | 0.567 | 0.380 | -315.09 | 0.093 |
| latent, decoded without undoing the standardization | 1.000 | 0.343 | 0.537 | -77.88 | 0.037 |
| latent, decoded without the pixel noise | 1.000 | 0.997 | 0.049 | 14.90 | 0.037 |
<!-- /worked -->

A pixel sampler that takes one step lands on the posterior mean of Chapter 6, a blur of the class: accuracy is still $0.957$, but $0.003$ of the pictures are on the data and the TV is $0.210$ (the blur is equally far from the class's two templates, so which position a sample is nearest to is decided by small differences). A guidance scale of $-1$ is the wrong sign of the blend, and the accuracy is $0.037$, a number near zero and not near a third: the sampler is steered away from the class. A latent decoded without undoing the standardization has full accuracy and $0.343$ on the data, because the decoder reads a code whose scale is not its own; it is the same kind of silent hazard that the module documentation of `crates/flux1/src/pipeline.rs` records for the packing of the wrong autoencoder (Chapter 2). And a latent decoded without the pixel noise has the highest density of the table, $14.90$ against the data's $-2.02$, with a distance of $0.049$ against $0.249$: the copy machine of the first table, reached through a decoder. All four mistakes are visible in a number that the correct generator also reports, and that is the point of having five.

**Milestone 5 ends** when your autoencoder's gradient check passes and its table agrees to the printed digits, the latent system's $g = 1$ and $g = 4$ rows reproduce, and each of the four planted mistakes gives the number of its row to the digits shown.

### Milestone 6: the bill

A speed claim in this chapter is a count: the FLOPs the arithmetic needs and the bytes that have to move. Both come from the shapes. A multiply and an add count as 2 FLOPs, a $\tanh$ as 3, each bias addition as 1, and the modulation costs 5 per hidden unit and layer (add the bias of the gain, add 1, multiply by the pre-activation, add the bias of the shift, add the shift) on top of two four-value dot products. The timestep features and the class lookup are not counted. A pass at batch size one reads every weight once, as a 4-byte float, so a pass moves four bytes per value. A training item costs a forward pass and a backward pass, which this chapter counts as two forward passes, so three.

<!-- worked: project_image_generator.cost_ledger -->
FLOPs and bytes, counted from the shapes (a multiply and an add are 2 FLOPs, a tanh is 3 FLOPs, a pass reads every weight once as a 4-byte float, so the bytes of a pass are four times its values). The last column is FLOPs per byte read.

| stage | values | bytes of weights | FLOPs per pass | FLOPs per byte |
|---|---:|---:|---:|---:|
| pixel denoiser | 7,060 | 28,240 | 14,340 | 0.508 |
| latent denoiser | 3,956 | 15,824 | 8,164 | 0.516 |
| encoder (mean only) | 988 | 3,952 | 2,020 | 0.511 |
| decoder | 1,020 | 4,080 | 2,052 | 0.503 |

A backward pass is counted as two forward passes, so a training item costs three.

| work | count | FLOPs |
|---|---:|---:|
| 16-step guided sample, pixel space | 16 x 2 x 14,340 | 458,880 |
| 16-step guided sample, latent space | 16 x 2 x 8,164 + 2,052 | 263,300 |
| training the pixel denoiser | 64,000 items x 3 x 14,340 | 2,753,280,000 |
| training the autoencoder | 64,000 pictures x 3 x 4,268 | 819,456,000 |
| training the latent denoiser (encoder in the loop) | 64,000 items x (3 x 8,164 + 2,020) | 1,696,768,000 |
| training the latent system in all | autoencoder + latent denoiser | 2,516,224,000 |

A guided pixel sample costs 1.74 times the FLOPs of a guided latent sample; training the whole latent system (autoencoder and denoiser) costs 0.91 times the FLOPs of training the pixel denoiser.
<!-- /worked -->

Every pass is close to $0.5$ FLOPs per byte, which is what a weight that is read once and used for one multiply and one add comes to: $2$ FLOPs for $4$ bytes. The passes of this chapter are memory-bound by construction on any device whose roofline ridge lies above $0.5$ FLOPs per byte, as those of GPUs do by a wide margin, and the way to a cheaper pass is fewer weights and not a faster multiply. The latent denoiser has $3{,}956$ values against the pixel denoiser's $7{,}060$, because the layers that touch the picture shrink from $36$ to $4$ values wide while the hidden core stays the same; a guided pixel sample costs $1.74$ times the FLOPs of a guided latent sample ($458{,}880$ against $263{,}300$). The latent system's saving is the picture-sized layers, and at this size the hidden core, which is shared, dominates: the table understates what a latent buys when the picture is a megapixel and the core is a transformer, where Chapter 2's compression table counts $48$ numbers of picture for every number of latent. Training the whole latent system (the autoencoder and the denoiser) costs $0.91$ times the FLOPs of training the pixel denoiser alone: the autoencoder is paid for once, and the pixel system buys nothing with its extra cost.

**Milestone 6 ends** when your counts equal the table's. They are integer counts of the shapes and must agree exactly; a count that differs by a small number of FLOPs is a missed bias or a missed modulation term.

## The yardstick numbers the generator must hit

The tests are the values the worked module prints, collected as a checklist. Test in the order of the milestones: a disagreement at milestone 4 that follows agreement at 1 to 3 points at training and not at the data. "Exact" means equal integers and strings; a numeric tolerance is absolute.

<!-- worked: project_image_generator.acceptance_tests -->
Expected values of the project, computed by this module. 'Exact' means equal; a number is compared with the tolerance in the last column, absolute. A cell of five numbers is the five measures of the chapter in the order accuracy, share on data, distance to the template, mean ln p, TV.

| milestone | quantity | value | tolerance |
|---:|---|---|---|
| 1 | pixels on in the column, row and block templates | 6 / 6 / 4 | exact |
| 1 | first picture of the stream `data`: class, position; sum of its pixels | block, 1; -29.451198 | exact; 1e-6 |
| 1 | the data, 300 pictures: accuracy / on the data / distance / ln p / TV | 1.000 / 1.000 / 0.249 / -2.02 / 0.050 | exact; exact; 0.001; 0.01; 0.001 |
| 2 | 100-step Runge-Kutta samples of the exact field: the same five measures | 1.000 / 1.000 / 0.247 / -1.68 / 0.040 | as above |
| 2 | exact field, Euler 16 steps, g = 1 and g = 4: distance / ln p | 0.217 / 2.43; 0.222 / 1.72 | 0.001; 0.01 |
| 2 | exact-field Euler against the reference at 4 and 32 steps; Heun at 4 and 32 evaluations | 0.1174 / 0.0168; 0.0727 / 0.0018 | 1e-4 |
| 3 | values of the pixel denoiser, of the autoencoder; FLOPs of one pass | 7,060; 2,108; 14,340 | exact |
| 3 | initial network at the first starting vector, sigma 0.5, class 1: first four outputs | 0.501486, -0.325876, 0.646063, -0.335603 | 1e-6 |
| 4 | after 2,000 steps: loss of the network / of the exact field / (network - exact)^2 | 0.5686 / 0.5036 / 0.0569 | the digits shown |
| 4 | distance to the reference, 16 evaluations: Euler / Heun, network | 0.1675 / 0.1678 | 0.001 |
| 4 | pixel network, g = 1, 16 steps: the five measures | 0.990 / 0.930 / 0.286 / -10.70 / 0.023 | exact; exact; 0.001; 0.01; 0.001 |
| 4 | pixel network, g = 4, 16 steps: the five measures | 1.000 / 0.083 / 0.505 / -65.67 / 0.107 | as above |
| 5 | autoencoder on 600 pictures: squared error per pixel; KL; template round trip | 0.0590 / 7.23 / 0.021 | 1e-4; 0.01; 0.001 |
| 5 | latent system, g = 1, 16 steps: the five measures | 1.000 / 0.993 / 0.257 / -3.30 / 0.037 | exact; exact; 0.001; 0.01; 0.001 |
| 5 | latent system, g = 4, 16 steps: the five measures | 1.000 / 0.990 / 0.258 / -3.40 / 0.093 | as above |
| 5 | latent system decoded without undoing the standardization: accuracy / on the data | 1.000 / 0.343 | exact |
| 6 | FLOPs of a 16-step guided sample: pixel space / latent space | 458,880 / 263,300 | exact |
<!-- /worked -->

Beyond the table, an implementation passes when it also meets these, each of which is in the tables above: the identities of milestone 2 below their criteria; the gradient checks of the pixel denoiser and of the autoencoder below $10^{-6}$ at every parameter; the training table's last row, with the network's gap to the exact field falling at every logged step; the solver table's network columns flat to within $0.02$ from 8 evaluations while the exact-field columns keep falling; the oversaturation of the pixel network at $g = 4$ against the exact field's immunity; and the four planted mistakes, each producing the number of its row. A generator that passes the clean rows and cannot be made to fail the planted ones has tests that cannot fail.

## Where each piece of the generator lives in brain

brain has no class-conditional generator of $6 \times 6$ pictures, and every one of its image pipelines is this project grown. The parts line up piece for piece, and in brain the line-up is explicit in the structure of the crates.

**The objective and its conventions.** `crates/flux2/src/modelgrad.rs` builds a flow-matching batch in `make_flow_batch_paired`: $x_\sigma = (1-\sigma)x_0 + \sigma\varepsilon$, the target $\varepsilon - x_0$, and the model's time input equal to $\sigma$. Its documentation says that this is the convention which the Euler integrator of `crates/flux2/src/pipeline.rs` inverts (`x += dt * v`, with $\sigma$ stepping from 1 to 0). That is `training_item` and `euler_sample` of this chapter, and the sentence states the property that has to hold between them: the batch and the sampler agree on the sign of the velocity and on the clock. Z-Image's fine-tuner makes the same point about its own convention in the module header of `crates/s3dit/src/finetune.rs`, which verifies the convention against its Euler integrator and states that its time input is $1 - \sigma$ (the sign and the clock of Chapter 8); there the clean latent is called $x_1$ and the noise $x_0$, the opposite of this book's names. The loop that the flow-matching fine-tuners share is `crates/model/src/flow.rs`. Its header says that it owns only the lifecycle of a run (cancelling, the cadence of checkpoints, the final save), and commit e43132823 moved Z-Image and FLUX.2 onto it while each family kept its own batch and loss: the split this chapter has between `training_item` and the Adam loop.

**The latent boundary.** Both fine-tuners encode every image and caption once, up front, and drop the encoders before training starts (the headers of `crates/flux2/src/finetune.rs` and `crates/s3dit/src/finetune.rs` say so; the second gives the reason, to free device memory), so that the denoiser is trained on codes of a frozen autoencoder, exactly the arrangement of milestone 5. The standardization is per model: `crates/vae/src/latent.rs` packs the FLUX.2 latent and applies a per-channel statistic (the table of milestone 5 is its small counterpart), and `crates/s3dit/src/pipeline.rs` declares `VAE_SCALE` and `VAE_SHIFT` and applies the inverse, `x / VAE_SCALE + VAE_SHIFT`, before the decoder. Skipping that inverse is the third planted mistake.

**The null class and the guidance blend.** The null class has its counterpart in `crates/flux2/src/pipeline.rs`: the unconditional branch is the encoding of the empty prompt (`encode_prompt("")`), taken only for an undistilled variant with a guidance scale above one; the step then evaluates the conditional and the unconditional passes as two slots of one batch and blends them as `un + guidance * (c - un)`, which is `guided_velocity`. `crates/sdxlunet/src/sampler.rs` makes the same two passes as two calls, because its graph is recorded for one sample, and skips the unconditional pass when guidance is at most 1, as `guided_velocity` does at exactly 1. Chapter 7 told the rest: the training side of the null class is a dropout of the conditioning (`ref_dropout` in the FLUX.2 fine-tuner, commit 97704014f), and a distilled model has no unconditional branch at all.

**The sampler core and what it leaves out.** `crates/diffusion/src/scheduler.rs` is the Euler integrator on the straight path. The header of `crates/diffusion/src/lib.rs` says of the flow-matching scheduler that "the denoiser predicts a velocity, and Euler integrates it", and draws the line that this chapter drew without being told to: the crate holds the schedule and the samplers, which it calls "pure host math", and the training objective and the combine of the guidance live in the model crates, "next to the denoiser they drive". The grid of this chapter is uniform; brain's schedules are the shifted ones of Chapter 6.

**The denoiser.** The project's network is a two-layer perceptron with a clock and a class, and the three denoisers of brain are three answers to what a perceptron could not do at scale. `crates/flux2` and `crates/s3dit` are diffusion transformers whose blocks are modulated by the timestep embedding, the scale-and-shift of milestone 3 grown to every block, and `crates/sdxlunet` is the U-Net that adds the embedding to the channels of every resnet, the form Chapter 8 showed to need depth to recover the gain. The class table with its null row is the text encoder and its empty prompt, grown from four values to hundreds of tokens, and the Chapter 9 interfaces are the ways that sequence reaches the network.

**Where the project's choices differ from brain's.** The project draws $\sigma$ uniformly and independently for every item. The FLUX.2 fine-tuner does not: `step_sigma` in `crates/flux2/src/finetune.rs` draws from the sampler's own schedule (the distilled variant evaluates the network at a handful of noise levels), and stratifies the draw, each block of steps being a permutation of the whole schedule. Its documentation gives the reason, and it is a lesson of this project: the loss at the noisiest of those levels is, by its own account, close to twice the loss at the least noisy, so that the mixture of levels in any short window is the biggest term in what a window of losses averages to, and an independent draw makes that mixture swing. The marginal is the same and the variance is smaller (commit 97704014f). The documentation's other claim, that at 512 pixels the four levels are all above $0.71$, can be checked against the schedule of Chapter 6: with $\mu$ for 1,024 tokens and 4 steps the levels are $1$, $0.958$, $0.884$ and $0.717$. A reader who finds the loss of this chapter's training noisy from one batch to the next can do the same, with the same table to say whether it helped.

## Summary

If every acceptance test passes you have a complete small generator and its evaluation. A world whose distribution is known, so that a generator is judged by five measures of that distribution and a density is not mistaken for quality. An exact field that is the yardstick for every network and gives a sampler that can be wrong on its own (Euler at 16 steps is too clean). A denoiser with a class table and a null row, a scale-and-shift clock and a hand-written backward pass that agrees with central differences. Flow-matching training with the class dropped with probability 0.15, an Euler and a Heun sampler, and classifier-free guidance that works at $g = 1$ and $2$ and breaks down at $4$, where the network's error has grown by a factor of the order of $g^2$. An autoencoder, a standardized latent, and the same denoiser on the codes, which at $g = 1$ is nearer to the data on four of the five measures and costs $263{,}300$ FLOPs per guided sample against $458{,}880$.

What the project established, in order. A likelihood is not a quality score; a set of measures that fail differently is. The best sampler for a trained network is not the best sampler for its exact field, since once a solver is accurate enough the network's own error is the floor. A guidance scale multiplies the error of the network, and the useful range ends where that error is visible. A latent changes what the denoiser has to learn and what the pipeline can get wrong: its standardization must be undone, and a decoder returns means, so a sample of the model adds the noise back. And the cost of a pass at this size is its weights: every pass is $0.5$ FLOPs per byte, and the work to count is the weights read, the number that falls when the picture-sized layers shrink.

Where this stops. The world has six modes and a Gaussian noise, so its exact field is the best the network could do and the generator is judged against it, which no real dataset allows; a real distribution has a density the evaluator cannot compute and the measures would be a classifier, a feature distance and a human. The class is one of three labels, where brain's conditioning is a sequence of text embeddings read by attention. The denoiser is a perceptron, where the denoisers of Chapter 8 mix positions. The autoencoder's decoder returns means, and the images are $6 \times 6$ and noise; the hard parts of a real latent (texture, a decoder with fifty million parameters, a tiling over a canvas) are not here. The limits are the reasons for the extensions.

## References

- Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel and M. Le, *Flow matching for generative modeling*, 2022. The objective and the straight path.
- J. Ho and T. Salimans, *Classifier-free diffusion guidance*, 2021. The null class and the blend.
- D. P. Kingma and M. Welling, *Auto-encoding variational Bayes*, 2013. The bound and the reparameterization.
- R. Rombach, A. Blattmann, D. Lorenz, P. Esser and B. Ommer, *High-resolution image synthesis with latent diffusion models*, 2022. The frozen autoencoder and the denoiser on its codes.
- W. Peebles and S. Xie, *Scalable diffusion models with transformers*, 2023. The conditioning by scale, shift and gate with a zero start.

## Beyond six modes and Gaussian noise

These are optional. Each changes one thing of the project and asks you to predict the outcome before you run it.

1. **A second-order sampler with a shifted grid.** Replace the uniform grid by the shifted one of Chapter 6 with a factor of $2$ and run the solver table again. Predict whether the exact-field columns or the network's change more, and why.
2. **A guidance ramp.** Make the scale a function of $\sigma$ as in Chapter 7: $1$ at the noise end and $4$ at the data end, and the reverse. Predict, before running it, which of the two keeps the pixel network on the data at the average scale of $2.5$.
3. **Guidance distilled.** Train a student with the pixel network's architecture on the teacher's guided field at $g = 2$ (Chapter 7) and compare its five measures at 8 evaluations with the teacher's at 16.
4. **A third position, then a fourth.** Add positions so that the data have nine and then twelve templates. Predict how the pixel network's accuracy and share on the data at $g = 1$ move for the same 2,000 steps and the same width, and whether the latent system's do.
5. **Image to image.** Encode a picture of the data, standardize it, noise it to $\sigma = 0.6$ by the straight path and sample from there with the class changed (Chapter 10). Predict the strength at which the class of the result follows the request and not the picture.

## Answers

1. $x_\sigma = (1 - 0.25)\cdot 1 + 0.25\cdot(-1) = 0.5$ and the target is $\varepsilon - x_0 = -2$. One Euler step with the exact field from $\sigma$ to $0$ lands on the posterior mean $\mathbb{E}[x_0 \mid x_\sigma]$, since $x - \sigma u$ is that mean by the definition of the field.
2. $u_g = u_{\mathrm{un}} + g\,(u_c - u_{\mathrm{un}})$. At $g = 0$ it is the unconditional field and at $g = 1$ the conditional one, with $g > 1$ an extrapolation beyond the conditional field. One step takes two passes (one at $g = 1$ or $g = 0$). An error $\delta$ in the conditional branch becomes $g\delta$.
3. $\lVert\mathbf{x} - g(\mathbf{z})\rVert^2/(2\sigma_{\mathrm{x}}^2) + \tfrac12\sum_k(m_k^2 + e^{v_k} - 1 - v_k)$ plus the constant $\tfrac{D}{2}\ln(2\pi\sigma_{\mathrm{x}}^2)$, with $\mathbf{z} = \mathbf{m} + e^{\mathbf{v}/2}\odot\boldsymbol{\xi}$ for a draw $\boldsymbol{\xi} \sim \mathcal{N}(0, \mathbf{I})$, so that $\mathbf{m}$ and $\mathbf{v}$ are differentiable functions of the weights and the randomness is outside them.
4. Nothing in training pins the scale of the code (the KL is small next to the reconstruction), so the denoiser, whose noise has unit variance, would meet data of an arbitrary scale. The mean is standardized per channel by $(\mathbf{m} - \mathbf{c}_{\mathrm{shift}})/\sqrt{\mathbf{c}_{\mathrm{var}} + 10^{-4}}$, and the denoiser's output is multiplied by $\sqrt{\mathbf{c}_{\mathrm{var}} + 10^{-4}}$ and shifted by $\mathbf{c}_{\mathrm{shift}}$ before the decoder.
5. The best gain on the deviation from the template is $k(\sigma) = (1 - \sigma)\sigma_{\mathrm{x}}^2/\varrho$, which is near $1$ at a small $\sigma$ and near $0$ at a large one; a fixed gain $w$ is right at one noise level and wrong at the others. A layer with a gain computed from the noise level, $(1 + s(\sigma))\,z + b(\sigma)$, which is adaptive layer normalization in its smallest form.
6. The analytic gradient with the central difference $(L(\theta_i + h) - L(\theta_i - h))/2h$ at every parameter, with $h$ near $10^{-5}$ in double precision; the largest difference divided by the largest gradient entry below $10^{-6}$ says the backward pass is the derivative of the forward pass. It is done at weights that are not the initialization, since at the initialization some parameters have gradients that vanish by construction.
7. A density is what a likelihood-trained model gives and what is needed to score a model on data it has not seen and to say by how much two models differ; a sample alone says that a model produces plausible things and not that it produces all of them in the right proportions, and a model that memorizes its training set samples perfectly and generalizes nowhere.
