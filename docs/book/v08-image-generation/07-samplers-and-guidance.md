<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Samplers and Guidance

*How many steps does sampling need, and how do we make a model follow its prompt more closely?*

Count the work in a request before it has produced a pixel. A denoiser is a network, a step of any sampler needs its output, and a step may need it twice. The table counts the passes that each of brain's image and video pipelines makes under its own default settings.

<!-- worked: samplers_and_guidance.forward_counts -->
Network evaluations (forward passes of the denoiser) per image, from each pipeline's default step count and the rule that decides whether a second, unconditional pass is made: evaluations = steps x passes per step.

| model | steps | guidance scale | passes per step | evaluations | source |
|---|---:|---:|---:|---:|---|
| SDXL text2image | 30 | 5 | 2 | 60 | `crates/sdxlunet/src/caps.rs`: steps 30, guidance 5.0 (1.0 disables guidance) |
| Wan text-to-video | 50 | 5 | 2 | 100 | `crates/wan/src/config.rs`: sample_steps 50, guide_scale 5.0 |
| FLUX.2 base | 50 | 4 | 2 | 100 | `crates/flux2/src/pipeline.rs`: 50 steps, guidance 4.0 |
| FLUX.2 Klein (distilled) | 4 | 1 | 1 | 4 | `resolved_steps`: fixed 4, no guidance branch |
| FLUX.1 dev | 50 | 3.5 | 1 | 50 | one forward; the guidance scalar is an input of the network |
| Z-Image Turbo | 8 | 0 | 1 | 8 | `crates/s3dit/src/caps.rs`: steps 8, and no guidance scale: one DiT pass a step |
<!-- /worked -->

The default SDXL request runs the U-Net $60$ times, the default Wan request $100$, and the FLUX.2 architecture takes $100$ passes as the undistilled base model and $4$ as the distilled Klein. Two numbers multiply in every row, the steps and the passes per step, and the two halves of this chapter are about each of them. The first half asks how few steps a sampler may take before its answer is wrong, which is the question of how good a solver of the ordinary differential equation of the last chapter can be made. The second asks why a pipeline would pay a second pass at all, what that pass buys, and how a network can be taught to give its answer without being asked twice.

Six earlier ideas return within a few pages, so try them first from memory (the answers are at the end, under "Recall questions").

1. (Volume VIII, [Chapter 6](06-flow-matching.md), the velocity field.) With the exact field, one Euler step from noise fraction $\sigma$ to $0$ lands on a quantity the network could be asked for directly. What is it?
2. (Volume I, [Chapter 9](../v01-mathematics/09-integrals-and-odes.md), Heun's method.) How many evaluations of the slope does a step of Heun's method need, and how does its global error behave when the step is halved, against Euler's?
3. (Volume I, [Chapter 10](../v01-mathematics/10-probability.md), conditional probability.) Two classes are equally likely. A point $x$ has density $0.9$ under class $1$ and $0.3$ under class $0$. What is $P(\text{class } 1 \mid x)$?
4. (Volume VIII, [Chapter 5](05-diffusion.md), the reverse step.) The sampler of Chapter 5 added a fresh normal draw at every step except the last. What set the variance of that draw?
5. (Volume I, [Chapter 11](../v01-mathematics/11-distributions.md), the Gaussian distribution.) If $X \sim \mathcal{N}(0, \alpha^2)$ and $Z \sim \mathcal{N}(0, \beta^2)$ are independent, what is the variance of $X + Z$?
6. (Volume VIII, [Chapter 6](06-flow-matching.md), the timestep shift.) Where does $\sigma = 0.5$ go under a shift with $k = 3$?

Everything below is written for one real number, with the data of the last two chapters (three clusters at $-3$, $0$ and $3$, standard deviation $0.25$, equal weights), so that each sampler can be scored against the exact transport map $T$ of Chapter 6, and the same $1000$ starting values are used throughout. A second, smaller problem with two overlapping classes is introduced where the prompt enters.

## The same step, written twice

Chapter 6 stepped from noise fraction $\sigma$ to a smaller $\sigma'$ with Euler's method. There is another way to arrive at a point on the same line, which does not mention a velocity at all. A denoiser that has seen $x$ at noise level $\sigma$ can give a guess $\hat x_0$ of the clean value, and the guess implies the noise that must have been added to reach $x$: $\hat\varepsilon = (x - (1 - \sigma)\hat x_0)/\sigma$. The point on the path at the lower level $\sigma'$ that has the same guess and the same implied noise is

$$
x' = (1 - \sigma')\,\hat x_0 + \sigma'\,\hat\varepsilon .
$$

In words: re-noise the guess to the new level, using the noise that the old level implies. The first pair of tables checks that this is the Euler step. On the straight path, substitute $\hat\varepsilon$ and collect the terms in $\hat x_0$: $x' = \frac{\sigma'}{\sigma}x + \bigl(1 - \frac{\sigma'}{\sigma}\bigr)\hat x_0$, which is the weighted average of the last chapter, so Euler's method and "re-noise the guess" are one rule.

<!-- worked: samplers_and_guidance.ddim_is_euler -->
The deterministic step on the straight path (a = 1 - sigma, b = sigma), for the three clusters: from x at noise fraction sigma to sigma', with the denoiser's guess x0 = E[x_0 | x], the noise it implies e = (x - (1 - sigma) x0) / sigma, the step (1 - sigma') x0 + sigma' e, and the Euler step x + (sigma' - sigma)(x - x0)/sigma of the last chapter.

| x | sigma | sigma' | guess x0 | implied e | step from the guess | Euler step |
|---|---:|---:|---:|---:|---:|---:|
| 2.50 | 0.7 | 0.5 | 2.989877 | 2.290053 | 2.63996492 | 2.63996492 |
| -1.00 | 0.4 | 0.3 | -2.121157 | 0.681736 | -1.28028937 | -1.28028937 |
| 0.25 | 0.9 | 0.6 | 0.183126 | 0.257430 | 0.22770867 | 0.22770867 |
| 1.00 | 0.5 | 0.1 | 2.387838 | -0.387838 | 2.11027068 | 2.11027068 |
| 2.90 | 0.2 | 0.0 | 3.312500 | 1.250000 | 3.31250000 | 3.31250000 |

The two steps differ by less than 1e-12 over the five rows (a rounding error).

The same on the variance-preserving chain of the diffusion chapter (T = 50, `diffusion.ab_coeffs` and `diffusion.exact_denoiser`): the step a' x0 + b' e, against one Euler step of the variable y = x / a in the noise level sigma_t = b / a, y' = y + (sigma_t' - sigma_t) e, and then x' = a' y'.

| t | t' | x_t | sigma_t = b/a | sigma_t' | step from the guess | Euler in (y, sigma_t), times a' |
|---|---:|---:|---:|---:|---:|---:|
| 50 | 40 | 1.30 | 31.2032 | 8.5429 | 1.31309371 | 1.31309371 |
| 40 | 30 | -0.80 | 8.5429 | 3.1013 | -0.87449456 | -0.87449456 |
| 30 | 20 | 2.10 | 3.1013 | 1.3317 | 2.61459644 | 2.61459644 |
| 20 | 10 | 0.40 | 1.3317 | 0.5241 | 0.51935450 | 0.51935450 |
| 10 | 0 | 2.90 | 0.5241 | 0.0000 | 3.05081536 | 3.05081536 |

The two differ by less than 1e-12 over the five rows.
<!-- /worked -->

The second table is the same statement for the variance-preserving chain of Chapter 5, where the step is $x_{t'} = a_{t'}\hat x_0 + b_{t'}\hat\varepsilon$. Divide through by $a_{t'}$ and write $y = x/a$ and $\sigma_t = b/a$ for the rescaled variable and its noise level (the $\sigma_t$ of Chapter 5, with its subscript; it is not the noise fraction $\sigma$ of the straight path). The step becomes $y_{t'} = \hat x_0 + \sigma_{t'}\hat\varepsilon$ against $y_t = \hat x_0 + \sigma_t\hat\varepsilon$, so $y_{t'} = y_t + (\sigma_{t'} - \sigma_t)\hat\varepsilon$: one Euler step, in the variable $y$ and the "time" $\sigma_t$, of the equation $dy/d\sigma_t = \hat\varepsilon(y, \sigma_t)$. The chain and the straight path differ in the clock and in the scale of the variable, and the rule that connects a guess to a step is the same on both. It is the rule of **DDIM**, the denoising diffusion implicit model, which was found as a way of sampling a trained chain of Chapter 5 in far fewer than $T$ steps by skipping from $t$ to a distant $t'$ (the table steps from $50$ to $40$, $40$ to $30$ and so on, five steps instead of fifty).

What the step has that the chain of Chapter 5 lacked is the absence of a random draw. Once the starting value is chosen, every later value is a function of it, and the whole sampler is a **deterministic sampler**: the same noise always produces the same image, two noises that are close produce close images, and the map from noise to data can be analysed, interpolated and inverted like any other function. Chapter 6's Euler sampler was one, and the observation in that chapter's opening that once the starting noise is drawn the end point is determined is the property named here. Everything in the rest of this chapter that is called a solver is a deterministic sampler, except where noise is put back on purpose.

## Putting some of the noise back

Why would one put it back? The deterministic step uses the implied noise $\hat\varepsilon$ in full, so any mistake in the guess is carried forward by the next step. A step can instead keep only part of the implied noise and make up the difference with a fresh draw. The requirement that fixes how much is simple. Let the path be $x_t = a_t x_0 + b_t\varepsilon$ and consider a step from level $t$ to a lower level $t'$ of the form

$$
x_{t'} = a_{t'}\,x_0 + \kappa\,(x_t - a_t x_0) + \nu z ,\qquad z \sim \mathcal{N}(0, 1),
$$

that is, the new value is the clean value, scaled to the new level, plus a multiple $\kappa$ of the noise present in $x_t$, plus fresh noise of standard deviation $\nu$. If $x_t$ is a draw from the path, $x_t - a_t x_0 = b_t\varepsilon$ has variance $b_t^2$, so $x_{t'}$ has the right mean $a_{t'}x_0$ and the right variance $b_{t'}^2$ exactly when $\kappa^2 b_t^2 + \nu^2 = b_{t'}^2$ (Recall question 5). Every $\nu$ from $0$ up to $b_{t'}$ gives a step that keeps the marginals of the path, with $\kappa = \sqrt{b_{t'}^2 - \nu^2}/b_t$. Two choices stand out. At $\nu = 0$ the step is the deterministic one above ($\kappa = b_{t'}/b_t$). At one particular value, the one for which the step is the exact reverse step,

$$
\nu_1^2 = b_{t'}^2\Bigl(1 - \frac{a_t^2\,b_{t'}^2}{a_{t'}^2\,b_t^2}\Bigr),
$$

the step is the exact reverse step of the forward process given $x_0$, the one Chapter 5 derived as a product of two Gaussians. The **eta family** interpolates: $\nu = \eta\,\nu_1$ for $\eta \in [0, 1]$. In the sampler the clean value is not known, so $x_0$ is replaced by the guess and $x_t - a_t x_0$ by the implied noise $\hat\varepsilon$ (times $b_t$):

```text
function sample_eta(guess: function(real, real) -> real, grid: array<real>[n + 1], eta: real, x1: real, rng: Generator) -> real:
    require grid[0] == 1 and grid[n] == 0, "the grid runs from noise to data"
    require 0 <= eta and eta <= 1, "eta interpolates between the deterministic and the exact reverse step"
    x = x1
    for i in 0 .. n:
        s = grid[i]
        t = grid[i + 1]
        m = guess(x, s)
        if t == 0:
            return m
        e = (x - (1 - s) * m) / s
        nu = eta * t * sqrt(1 - ((1 - s) * t / ((1 - t) * s)) ^ 2)
        x = (1 - t) * m + sqrt(t ^ 2 - nu ^ 2) * e
        if nu > 0:
            x = x + nu * rng.normal()
    return x
```

Here the grid is of the straight path ($a = 1 - \sigma$, $b = \sigma$), `s` is the current level and `t` the next, and the last step lands on the guess with no noise. At $\eta = 0$ the routine draws nothing and is Euler's method; the loop bound and the early return mean that at most $n - 1$ draws are made.

<!-- worked: samplers_and_guidance.noise_split -->
The noise split of one step of the straight path from sigma = 0.6 to 0.4 (b = sigma): the standard deviation nu = eta nu_1 of the fresh noise, with nu_1 = 0.358323; the weight kappa = sqrt(b'^2 - nu^2)/b on the noise the denoiser implies; and the variance kappa^2 b^2 + nu^2 of the new value given x_0, which must equal b'^2.

| eta | nu | kappa | kappa^2 b^2 + nu^2 | b'^2 |
|---|---:|---:|---:|---:|
| 0 | 0.000000 | 0.666667 | 0.1600000000 | 0.1600000000 |
| 0.5 | 0.179161 | 0.596055 | 0.1600000000 | 0.1600000000 |
| 1 | 0.358323 | 0.296296 | 0.1600000000 | 0.1600000000 |

On the chain of the diffusion chapter: the noise of the eta = 1 step, nu_1 = b' sqrt(1 - (a b' / (a' b))^2), against a' sigma_up with sigma_up = sigma_t' sqrt((sigma_t^2 - sigma_t'^2)/sigma_t^2) the 'up' of the ancestral Euler step in the noise level sigma_t = b/a (and sigma_down = sqrt(sigma_t'^2 - sigma_up^2), the level it lands on before the noise is added).

| t | t' | nu_1 | a' sigma_up | sigma_down | sigma_t' |
|---|---:|---:|---:|---:|---:|
| 50 | 40 | 0.95526934 | 0.95526934 | 2.338890 | 8.542887 |
| 40 | 30 | 0.88681691 | 0.88681691 | 1.125848 | 3.101288 |
| 30 | 20 | 0.72217502 | 0.72217502 | 0.571874 | 1.331745 |
| 20 | 10 | 0.42674414 | 0.42674414 | 0.206247 | 0.524088 |
| 10 | 1 | 0.03156510 | 0.03156510 | 0.001910 | 0.031639 |

The two noise levels differ by less than 1e-12 over the five rows. For single steps (t' = t - 1, t = 2 .. 50) nu_1^2 differs from the variance tilde-beta_t = (1 - abar_(t-1)) beta_t / (1 - abar_t) of the exact reverse step of the diffusion chapter by less than 1e-12.
<!-- /worked -->

The first table splits one step of the straight path, from $\sigma = 0.6$ to $0.4$, between the implied noise and the fresh noise: at $\eta = 1$ the weight $\kappa$ on the implied noise falls from $0.667$ to $0.296$ and the fresh noise takes over, and the variance of the result given $x_0$ is $b_{t'}^2$ throughout. The second table shows that the step with $\eta = 1$ is a sampler that brain already carries under another name, and that it contains the sampler of Chapter 5. For single steps the variance $\nu_1^2$ equals the $\tilde\beta_t$ of Chapter 5, so at $\eta = 1$ and stride $1$ this is the ancestral sampler of that chapter, with the guess in place of the unknown clean value. For the strides of the table, $\nu_1$ equals $a_{t'}\sigma_{\text{up}}$ with $\sigma_{\text{up}} = \sigma_{t'}\sqrt{(\sigma_t^2 - \sigma_{t'}^2)/\sigma_t^2}$, which is the noise that the **Euler ancestral** sampler of `crates/diffusion/src/discrete.rs` adds: it steps deterministically down to $\sigma_{\text{down}} = \sqrt{\sigma_{t'}^2 - \sigma_{\text{up}}^2}$ and then adds $\sigma_{\text{up}}$ times a normal draw. Ancestral Euler is DDIM with $\eta = 1$, written in the rescaled variable.

Does the noise help? The table scores the sampler by how close the *distribution* of its end points is to the data, since a stochastic sampler has no paired end point to compare with: the mean absolute difference between the sorted samples and the quantiles of the data. A sample of $4000$ end points drawn exactly from the data would score $0.0239$ by this measure, which is the floor of the score.

<!-- worked: samplers_and_guidance.eta_table -->
The eta family on the straight path with the exact denoiser: 4000 samples, the starting values one normal() each from `samplers_and_guidance.start` (seed 0xc2e66f3e4e5af5f7), n uniform steps from sigma = 1 to 0 (n evaluations). The fresh noise is one normal() per sample and per step except the last, in sample order and step order, from `samplers_and_guidance.fresh` (seed 0x6023b4d9dab8246d) (restarted for every cell). The score is the mean of |sorted sample - quantile of the data| at the probabilities (k + 1/2)/N; the 4000 exact end points T(noise) themselves score 0.0239, which is the noise of the score, not an error of the sampler.

| steps n | eta = 0 | eta = 0.5 | eta = 1 |
|---|---:|---:|---:|
| 4 | 0.66 | 0.653 | 0.703 |
| 8 | 0.302 | 0.345 | 0.377 |
| 16 | 0.142 | 0.159 | 0.142 |
| 32 | 0.0681 | 0.0906 | 0.117 |
| 64 | 0.0295 | 0.0414 | 0.0564 |
| 128 | 0.018 | 0.048 | 0.0521 |

The same score with a wrong denoiser: the exact guess plus amp sigma sin(2 x + 3 sigma) (library sin, radians), amp = 0.3 and 0.6, for eta = 0 and 1. The noise stream and its order are those of the table above.

| steps n | amp 0.3, eta = 0 | amp 0.3, eta = 1 | amp 0.6, eta = 0 | amp 0.6, eta = 1 |
|---|---:|---:|---:|---:|
| 4 | 0.892 | 0.788 | 1.11 | 0.879 |
| 8 | 0.528 | 0.462 | 0.767 | 0.589 |
| 16 | 0.351 | 0.254 | 0.586 | 0.41 |
| 32 | 0.282 | 0.221 | 0.507 | 0.383 |
| 64 | 0.242 | 0.205 | 0.473 | 0.379 |
| 128 | 0.221 | 0.19 | 0.453 | 0.364 |
<!-- /worked -->

With the exact denoiser the deterministic sampler is the better one at every step count from $8$ up except at $16$, where $\eta = 0$ and $\eta = 1$ tie: at $64$ steps it scores $0.0295$ against $0.0564$ for $\eta = 1$, and at $128$ steps $0.018$, which is below the floor of the score, against $0.0521$. Noise that the denoiser does not need costs accuracy per step. With a denoiser that is wrong the order reverses. The wrong denoiser adds $\mathrm{amp}\cdot\sigma\sin(2\,x + 3\sigma)$ to the exact guess, an error that is smooth, vanishes at $\sigma = 0$ and is of the order of the guess itself at $\mathrm{amp} = 0.6$. At $32$ steps and $\mathrm{amp} = 0.6$ the sampler with $\eta = 1$ scores $0.383$ against $0.507$, and at every row of both columns of the second table it is ahead. This is a measurement of this denoiser and this error and not a theorem, but it reproduces the folklore of the field: stochastic samplers forgive imperfect networks, deterministic ones are faster when the network is good, and neither is right in general. brain carries both families, and the caller chooses (the ancestral scheduler even leaves the random draws to the caller, as the last section explains).

## Integrating what changes slowly

Whether deterministic or not, a sampler that uses the guess at the current point to move to the next one is an **ODE solver**: a rule that turns evaluations of a velocity field into a sequence of points, one step per grid interval, whose quality is measured by how its error falls with the number of steps (its order) and by how many evaluations of the field each step costs. Euler's method is a first-order solver with one evaluation per step, and Heun's method of Volume I a second-order one with two. The last chapter's table said that Euler's error falls by a factor of about $2$ when the steps double; the question for a sampler is whether the structure of this particular equation allows a better rule than the generic ones.

It does, and the structure is that $dx/d\sigma = (x - \hat x_0(x, \sigma))/\sigma$ is *linear* in $x$ with a forcing term, the guess. A linear equation can be integrated exactly if its forcing is known. Multiply by the integrating factor $1/\sigma$ and the equation becomes $d(x/\sigma)/d\sigma = -\hat x_0/\sigma^2$. The natural variable for the right side is the log signal-to-noise ratio

$$
\lambda(\sigma) = \ln\frac{1 - \sigma}{\sigma},
$$

which falls from $+\infty$ at $\sigma = 0$ to $-\infty$ at $\sigma = 1$ and satisfies $d\lambda = -d\sigma/(\sigma(1 - \sigma))$. Changing the variable and integrating from the level $\sigma_i$ of one grid point to the next level $\sigma_{i+1}$ gives the exact solution of the step,

$$
x_{i+1} = \frac{\sigma_{i+1}}{\sigma_i}\,x_i + (1 - \sigma_{i+1})\int_{\lambda_i}^{\lambda_{i+1}} e^{-(\lambda_{i+1} - \lambda)}\,\hat x_0(\lambda)\,d\lambda ,
$$

where $\hat x_0(\lambda)$ is the guess along the true trajectory. All of the stiffness of the equation (the term $x/\sigma$ that blows up near the data end) is in the first coefficient and in the weight $e^{-(\lambda_{i+1} - \lambda)}$, and both are exact. What is left to approximate is the guess, which varies slowly: it is the network's estimate of an image that does not change much from one step to the next. If the guess is replaced by a constant, its value $m_i$ at the start of the step, the integral is $(1 - e^{-h})\,m_i$ with $h = \lambda_{i+1} - \lambda_i$, and since $(1 - \sigma_{i+1})(1 - e^{-h}) = 1 - \sigma_{i+1}/\sigma_i$ the step is exactly

$$
x_{i+1} = \frac{\sigma_{i+1}}{\sigma_i}\,x_i + \Bigl(1 - \frac{\sigma_{i+1}}{\sigma_i}\Bigr) m_i ,
$$

Euler's step again. So Euler's method is the first-order member of a family, **DPM-Solver++**, in which the guess is approximated by a polynomial in $\lambda$ instead of a constant, and the integral of the polynomial against $e^{-(\lambda_{i+1} - \lambda)}$ is done exactly. The word "multistep" says where the polynomial gets its points: from the guesses that the sampler has already computed at earlier steps, so that the extra accuracy costs no extra evaluation.

For the second-order version, fit a straight line through the current guess $m_i$ at $\lambda_i$ and the previous guess $m_{i-1}$ at $\lambda_{i-1}$, with $h_0 = \lambda_i - \lambda_{i-1}$ and the ratio $r_0 = h_0/h$. The step adds a correction proportional to the slope, $D_1 = (m_i - m_{i-1})/r_0$, scaled to the step: $x_{i+1} = \frac{\sigma_{i+1}}{\sigma_i}x_i + (1 - \sigma_{i+1})(1 - e^{-h})\bigl(m_i + b_1(h)\,D_1\bigr)$. Integrating the line exactly gives the weight $b_1(h)$ of $D_1$, with $b_j$ written through the integrals $I_j(h) = \int_0^h e^{-(h-u)}u^j\,du$ (so that $I_0 = 1 - e^{-h}$, and $I_j = h^j - jI_{j-1}$ by integration by parts) as $b_j(h) = I_j/(h^j I_0)$. As $h \to 0$ the weight $b_1$ tends to $\tfrac12$, and brain's multistep solvers simply use $\tfrac12$ in its place (the midpoint form of the method). The table gives the first three of these weights, and then tests the claims on a guess for which the exact integral is known.

<!-- worked: samplers_and_guidance.step_coefficients -->
The right-hand sides b_j = I_j / (h^j (1 - e^-h)) of the exactness conditions, with I_j the integral over [0, h] of e^-(h - u) u^j du computed by I_0 = 1 - e^-h, I_j = h^j - j I_(j-1); b_1 tends to 1/2 as h tends to 0.

| h | b_1 | b_2 | b_3 |
|---|---:|---:|---:|
| 0.05 | 0.504166 | 0.337507 | 0.253760 |
| 0.30 | 0.524963 | 0.358545 | 0.272842 |
| 1.20 | 0.597679 | 0.434880 | 0.343812 |

The I_j agree with Simpson's rule (4000 panels) for j = 0 .. 3 and these three h to less than 1e-10.

One step of each kind from sigma = 0.6 (x = 2) to 0.45 (h = 0.6061), with the history point at sigma = 0.8 and a guess that depends only on lambda (so the exact step is (sigma'/sigma) x + (1 - sigma') I and the integral I is Simpson's rule with 4000 panels): the error of the step against it. 'first order' keeps the guess; 'predictor 1/2' adds the difference with weight 1/2; 'predictor b_1' with weight b_1; 'corrector 1' uses the new guess with weight 1/2; 'corrector 2' solves the 2 x 2 system.

| guess | first order | predictor 1/2 | predictor b_1 | corrector 1 | corrector 2 |
|---|---:|---:|---:|---:|---:|
| linear | -5.84e-02 | -5.33e-03 | below 1e-13 | -5.33e-03 | below 1e-13 |
| quadratic | -6.48e-02 | +1.54e-02 | +2.34e-02 | -8.68e-03 | below 1e-13 |
<!-- /worked -->

The weight $b_1$ makes the step exact for any guess that is a straight line in $\lambda$ (first row, third column: below $10^{-13}$), while the simplified $\tfrac12$ does not (the error is $5.3\times 10^{-3}$ at this $h$) though it has the same order. On the quadratic guess neither is exact, and the last two columns, the correctors, are the subject of the next section. The midpoint form (weight $\tfrac12$) and the "Heun" form (weight $b_1(h)$) are the two `SolverType` options of brain's DPM-Solver++ implementations; the exact form is the more accurate of them only where $h$ is not small.

In the book's pseudocode the second-order multistep solver in its midpoint form is the following, with the guess $m$ of the previous step carried along. The first step has no earlier guess, the step that reaches $\sigma = 0$ returns the guess, and a gap $h_0$ that is infinite (a first grid point at $\sigma = 1$) contributes no correction, since $\log(0) = -\infty$ and $D_1 = 0$ there.

```text
function dpm2m(guess: function(real, real) -> real, grid: array<real>[n + 1], x1: real) -> real:
    require grid[0] == 1 and grid[n] == 0, "the grid runs from noise to data"
    x = x1
    m_prev = 0
    for i in 0 .. n:
        s = grid[i]
        t = grid[i + 1]
        m = guess(x, s)
        if t == 0:
            return m
        h = log((1 - t) / t) - log((1 - s) / s)
        a = (1 - t) * (1 - exp(-h))
        x_next = (t / s) * x + a * m
        if i > 0 and i < n - 1:
            h0 = log((1 - s) / s) - log((1 - grid[i - 1]) / grid[i - 1])
            if is_finite(h0):
                x_next = x_next + a * 0.5 * (m - m_prev) * h / h0
        m_prev = m
        x = x_next
    return x
```

Chapter 6 introduced the timestep shift as a map that moves the grid toward noise. In the new variable the map has a clean meaning, and it is worth seeing now because the solvers will be tried on shifted grids below.

<!-- worked: samplers_and_guidance.log_snr_shift -->
The timestep shift with k = 3, sigma' = 3 sigma / (1 + 2 sigma), seen in the log signal-to-noise ratio lambda = ln((1 - sigma)/sigma) (natural logarithm).

| sigma | sigma' | lambda(sigma) | lambda(sigma') | lambda(sigma) - lambda(sigma') |
|---|---:|---:|---:|---:|
| 0.1 | 0.250000 | 2.197225 | 1.098612 | 1.098612 |
| 0.3 | 0.562500 | 0.847298 | -0.251314 | 1.098612 |
| 0.5 | 0.750000 | 0.000000 | -1.098612 | 1.098612 |
| 0.7 | 0.875000 | -0.847298 | -1.945910 | 1.098612 |
| 0.9 | 0.964286 | -2.197225 | -3.295837 | 1.098612 |

The last column is ln 3 = 1.098612 in every row (lambda(sigma') differs from lambda(sigma) - ln 3 by less than 1e-12). Shifting twice by k = 5 is shifting once by 25 (the two differ by less than 1e-12 at the five sigma). At sigma = 1, lambda is -infinity and at sigma = 0 it is +infinity. The value 1 - 1/1000 rounded to a 32-bit float is 0.999000013, which a timestep rule trunc(1000 sigma) turns into 999, where sigma = 1 gives 1000.
<!-- /worked -->

A shift by $k$ is a translation of $\lambda$ by $-\ln k$: since $(1 - \sigma')/\sigma' = (1 - \sigma)/(k\sigma)$, every finite point of any grid has its log signal-to-noise ratio lowered by the same amount $\ln k$, and the gaps $h$ between consecutive points are unchanged (the two ends, at $\pm\infty$, stay where they are). This also explains two facts of the last chapter. Shifting twice multiplies the factors (the table verifies $5 \cdot 5 = 25$), which is why a scheduler constructed with a shift and then given another one would apply the product, and why `crates/diffusion/src/flowsolvers.rs` constructs its solvers with a shift of $1$ and gives the real shift at `set_timesteps`: commit 303c4d4ff records exactly this trap in its message about Wan's solvers. And the exponential form of FLUX.2's shift, $e^\mu/(e^\mu + 1/\sigma - 1)$, is the translation of $\lambda$ by $-\mu$ itself.

## Reusing the last evaluation, and correcting with the next

A multistep solver has two awkward ends. At the first step there is no earlier guess to fit a line through, and at the last the grid ends at $\sigma = 0$ where $\lambda$ is $+\infty$. The solver handles both by lowering its order, and the table shows how often that happens for a short run.

<!-- worked: samplers_and_guidance.step_orders -->
The orders that two multistep solvers really run at on 8 steps. Left: DPM-Solver++(2M) on the grid sigma_i = 1 - i/8 (lambda = ln((1 - sigma)/sigma), h = lambda_(i+1) - lambda_i); the order is 2 only where an earlier guess exists, the last step is first order, and a guess at sigma = 1 sits at lambda = -infinity. Right: UniPC on sigma_i = (1 - 1/1000 in 32-bit) (1 - i/8), with the order of the predictor, min(2, n - i, i + 1), and of the corrector, the order the previous predictor ran at.

| i | sigma_i | lambda_i | h_i | effective order of DPM++(2M) | UniPC sigma_i | predictor order | corrector order |
|---|---:|---:|---:|---|---:|---:|---:|
| 0 | 1.0000 | -inf | inf | 1 (no earlier guess) | 0.999000 | 1 | none |
| 1 | 0.8750 | -1.946 | 0.847 | 1 (difference is 0: r0 infinite) | 0.874125 | 2 | 1 |
| 2 | 0.7500 | -1.099 | 0.588 | 2 | 0.749250 | 2 | 2 |
| 3 | 0.6250 | -0.511 | 0.511 | 2 | 0.624375 | 2 | 2 |
| 4 | 0.5000 | 0.000 | 0.511 | 2 | 0.499500 | 2 | 2 |
| 5 | 0.3750 | 0.511 | 0.588 | 2 | 0.374625 | 2 | 2 |
| 6 | 0.2500 | 1.099 | 0.847 | 2 | 0.249750 | 2 | 2 |
| 7 | 0.1250 | 1.946 | inf | 1 (lands on the guess) | 0.124875 | 1 | 2 |
<!-- /worked -->

On eight steps the second-order solver is first order at the first step (no history) and at the last, where $h$ is infinite: $e^{-h} = 0$, so the coefficient of the guess is $1 - \sigma_{i+1} = 1$ and the coefficient of $x$ is $\sigma_{i+1}/\sigma_i = 0$, and the step lands on the guess itself. That is the one-step landing of Chapter 6 (a step to $\sigma = 0$ takes the guess whole), now seen as the limit $h \to \infty$ of the solver's own formula; brain's test `dpmpp_last_step_lands_on_x0` in `crates/diffusion/src/discrete.rs` checks that a schedule ending at $\sigma = 0$ gives the guess $\hat x_0$ in the last step (to $10^{-4}$, in 32-bit arithmetic). The table also shows something that only appears on the straight path: when the grid starts at $\sigma = 1$, $\lambda_0 = -\infty$, the gap $h_0$ between the first two guesses is infinite, $r_0$ is infinite and $D_1$ is $0$, so the *second* step of the solver is first order too, although it asks for second order. This is the arithmetic of brain's `FlowDpmSolverPlusPlusScheduler` in `crates/diffusion/src/flowsolvers.rs`, whose sampling grid starts at exactly $1.0$ (its parity test against reference trajectories, `crates/diffusion/tests/it/wan_schedule_parity.rs`, covers both flow solvers when the dumped values are present). Nothing is wrong, since the formula degrades gracefully, but one step of the second-order method is spent at first order, and that is a consequence worth knowing. The same module gives its other solver, UniPC, a grid that starts at $0.99900001$ (the value $1 - 1/1000$ rounded to a 32-bit float, so that its first timestep is $999$ and the other solver's is $1000$, as the table of the shift section prints), and a finite $\lambda_0$.

The idea of UniPC is to use the next evaluation twice. The predictor is the second-order multistep step with weight $\tfrac12$, which is the midpoint step of the last section, and it moves from $x_i$ to a predicted $\tilde x_{i+1}$. The next thing the sampler does is to evaluate the guess at $\tilde x_{i+1}$, which it needs anyway to take the following step. That evaluation is also a new data point for the *previous* step, at $\lambda_{i+1}$: with it, the polynomial for the guess along the step has one more point, and a **corrector** re-does the previous step with the extra point and gets a more accurate $x_{i+1}$. The corrected point is the one the next prediction starts from; the guess at the uncorrected point is kept, and so the correction costs no evaluation at all. Let the step use $q$ points besides the current one, with weights $\rho_k$ on the differences $D_k = (m_{i-k} - m_i)/r_k$, where $r_k = (\lambda_{i-k} - \lambda_i)/h$ for an earlier guess and the new point has $r = 1$. The requirement that the step be exact for a guess that is a polynomial of degree $q$ in $\lambda$ is the linear system

$$
\sum_k r_k^{\,j-1}\rho_k = b_j(h), \qquad j = 1, \dots, q ,
$$

with the same right-hand sides $b_j$ as above. A predictor of order $p$ has $q = p - 1$ earlier points. A corrector of order $p$ has $q = p$ (the $p - 1$ earlier points and the new one) and so is exact one degree higher than the predictor of the same order; brain solves the system with its `solve` function, Gaussian elimination with partial pivoting. For $q = 1$ the system has one equation and one unknown, and the weight is $b_1(h) \approx \tfrac12$; the implementation uses $\tfrac12$ in the order-2 predictor and the order-1 corrector, where it is not solved. The last column of the coefficient table is the check of that claim: with a guess that is quadratic in $\lambda$, the two-by-two corrector is exact (below $10^{-13}$) where the predictors and the order-1 corrector have errors of the order of $10^{-2}$.

At the end of the table the last step is first order again, and the order of the corrector at each step is that of the previous predictor, which is why the last corrector in the table is of order $2$ while the last predictor is of order $1$. When the grid starts at $\sigma = 1$ there is one more case, handled here as the limit: a history point at $\lambda = -\infty$ has weight $0$ and the corrector keeps only the term of the new guess, with weight $b_1(h)$. That the predictor is the midpoint step is a statement that can be checked:

<!-- worked: samplers_and_guidance.predictor_check -->
UniPC with its corrector switched off against DPM-Solver++(2M) on the same grid and the same field, for the 1000 starting values of `flow_matching.sample` (seed 0x40611b1c8371ae36): the largest difference between end points is less than 1e-12 at 8, 16 and 32 evaluations.
<!-- /worked -->

The next section runs the four solvers with this rule.

## Four solvers against the exact map

Here is the experiment of Chapter 6, with more solvers. The exact field of the three clusters is used, so that the only error is the solver's. Every solver is charged for its evaluations of the field (the quantity that costs a network pass): Euler, DPM-Solver++(2M) and UniPC one per step, Heun two. The score is the mean of $|x_0 - T(\varepsilon)|$ over the $1000$ starting values of Chapter 6.

<!-- worked: samplers_and_guidance.solver_table -->
The 1000 starting values of `flow_matching.sample` (seed 0x40611b1c8371ae36) carried from sigma = 1 to 0 with the exact field by four solvers, each given the stated number of evaluations of the field. Euler and DPM-Solver++(2M) take one step per evaluation on sigma_i = 1 - i/n; Heun takes n/2 steps of two evaluations (the second at 1e-9 on the last step); UniPC takes one step per evaluation, its history point at sigma = 1 (lambda = -infinity) given weight 0. Distance is the mean of |x_0 - T(noise)|, to three significant digits.

| evaluations | Euler | Heun | DPM-Solver++(2M) | UniPC |
|---|---:|---:|---:|---:|
| 4 | 0.694 | 0.931 | 0.624 | 0.402 |
| 8 | 0.349 | 0.16 | 0.166 | 0.0488 |
| 16 | 0.164 | 0.0222 | 0.0396 | 0.00305 |
| 32 | 0.0798 | 0.00431 | 0.00773 | 0.00162 |
| 64 | 0.0352 | 0.000853 | 0.00203 | 0.00057 |

For comparison, 200 steps of classical Runge-Kutta (800 evaluations) score 4.0e-09. The factor by which the distance falls when the evaluations double (4 to 8, 8 to 16, 16 to 32, 32 to 64): Euler 2.0, 2.1, 2.1, 2.3; Heun 5.8, 7.2, 5.2, 5.1; DPM-Solver++(2M) 3.8, 4.2, 5.1, 3.8; UniPC 8.2, 16.0, 1.9, 2.8.
<!-- /worked -->

![Mean distance to the exact transport map against the number of evaluations of the field, on logarithmic axes, for Euler, Heun, DPM-Solver++(2M) and UniPC, with dashed lines of slope minus one and minus two: Euler follows the line of slope minus one, Heun and DPM-Solver++ fall about as the line of slope minus two, and UniPC is lowest, falling faster until sixteen evaluations and then levelling off.](img/solver-error.svg)

Read the table by what it takes to reach a level of error. Euler needs $64$ evaluations to get to $0.0352$. Heun is there at $16$ ($0.0222$), DPM-Solver++(2M) at about $16$ ($0.0396$), and UniPC at $16$ is $0.00305$, about $11$ times closer than Euler at $64$, with a quarter of the evaluations. The factors per doubling tell the same story: Euler's error falls by a factor between $2.0$ and $2.3$ (first order), Heun's by $5.1$ to $7.2$ and DPM-Solver++(2M)'s by $3.8$ to $5.1$ (second order, whose asymptotic factor is $4$), while UniPC falls by $8.2$ and then by $16$ over the first two doublings. Equally informative are the entries that go the wrong way. At $4$ evaluations Heun, with two steps, scores $0.931$, worse than Euler's four ($0.694$): a second-order method with too few steps is not a better first-order one. The first row is where the grid is too coarse for any local expansion to hold, and where the sampler must cross the point at which the three bumps merge ($\sigma \approx 0.59$, Chapter 6) in one or two steps.

UniPC's last rows call for a caution: after $16$ evaluations it stops improving at the rate of the first rows ($1.9$ and $2.8$ per doubling). The mean of a thousand errors is a statistic that a few trajectories can dominate, and here they do.

<!-- worked: samplers_and_guidance.tail_table -->
Where UniPC's error sits: for the evaluations n of the table above, the mean over the 1000 starting values, the median (the 501st smallest of the 1000 errors), the share of the total carried by the 50 worst starting values, and the largest error.

| evaluations | mean | median | share of the 50 worst | largest |
|---|---:|---:|---:|---:|
| 16 | 0.00305 | 0.00106 | 0.55 | 1.07 |
| 32 | 0.00162 | 0.0011 | 0.25 | 0.0572 |
| 64 | 0.00057 | 0.000447 | 0.19 | 0.0121 |
<!-- /worked -->

At $16$ evaluations the $50$ worst starting values carry $55\%$ of the total error and one of them is off by $1.07$, a third of the distance between clusters, the sign of a step that was too coarse where the field turns. By $32$ the worst is $0.057$ and the median, $0.0011$, is the main part of the story. A solver table should print the median or the tail with its mean, because a single number hides which regime it is in.

The solvers also depend on the grid, and not always in the way one hopes. The last chapter shifted the grid toward noise because the sampler must commit to a cluster there. Here is that shift of $k = 3$ applied to all four solvers.

<!-- worked: samplers_and_guidance.grid_table -->
The same four solvers, exact field, on the uniform grid and on the grid shifted by k = 3 (sigma' = 3 sigma / (1 + 2 sigma) applied to every grid point), at 8 and 16 evaluations. Mean distance to T(noise).

| solver | 8, uniform | 8, shifted | 16, uniform | 16, shifted |
|---|---:|---:|---:|---:|
| Euler | 0.349 | 0.265 | 0.164 | 0.128 |
| Heun | 0.16 | 0.242 | 0.0222 | 0.0285 |
| DPM-Solver++(2M) | 0.166 | 0.0877 | 0.0396 | 0.0699 |
| UniPC | 0.0488 | 0.111 | 0.00305 | 0.0681 |
<!-- /worked -->

Euler gains from the shift ($0.349$ to $0.265$ at $8$ evaluations, $0.164$ to $0.128$ at $16$), as Chapter 6 found for its own problem. DPM-Solver++(2M) gains at $8$ and loses at $16$; Heun and UniPC lose at both. A shifted grid squeezes the steps toward $\sigma = 1$ and widens the last ones; the table shows only that the best grid depends on the solver, not why. In the toy no ranking of grids holds across solvers, and in brain the grid is a property of the model as much as of the solver: Z-Image and FLUX.2 are trained and tuned for a given shift (Chapter 6), and a solver is chosen to suit it.

The last question of the section is how much of this a better solver can buy when the field is not exact. The wrong denoiser of the stochastic section is used again, with $\mathrm{amp} = 0.3$ and $0.6$. For a field with an error there are two distances. The first, "solver", is the distance of the solver's end points from the accurate end points of the *wrong* field, which is what a better solver can reduce. The second, "total", is the distance to the exact map $T$, which is what the user sees. The difference between them is the floor.

<!-- worked: samplers_and_guidance.model_error_table -->
Solver error against model error. The denoiser is the exact guess plus amp sigma sin(2 x + 3 sigma) with amp 0.3 and 0.6. 'Floor' is the mean distance between T(noise) and the accurate end point of the wrong field (Runge-Kutta, 200 steps): what no solver can remove. For Euler and UniPC (grids of the last table, uniform), the mean distance to that accurate end point ('solver') and to T(noise) ('total').

| amp | evaluations | floor | Euler: solver | Euler: total | UniPC: solver | UniPC: total |
|---|---:|---:|---:|---:|---:|---:|
| 0.3 | 8 | 0.23 | 0.354 | 0.563 | 0.0692 | 0.238 |
| 0.3 | 16 | 0.23 | 0.186 | 0.402 | 0.00678 | 0.229 |
| 0.3 | 32 | 0.23 | 0.102 | 0.323 | 0.00287 | 0.229 |
| 0.3 | 64 | 0.23 | 0.0464 | 0.272 | 0.000719 | 0.23 |
| 0.6 | 8 | 0.465 | 0.384 | 0.812 | 0.0871 | 0.454 |
| 0.6 | 16 | 0.465 | 0.191 | 0.633 | 0.01 | 0.462 |
| 0.6 | 32 | 0.465 | 0.0954 | 0.548 | 0.00189 | 0.465 |
| 0.6 | 64 | 0.465 | 0.0502 | 0.508 | 0.000773 | 0.465 |
<!-- /worked -->

At $\mathrm{amp} = 0.3$ the floor is $0.23$. Euler's total error is $0.563$ at $8$ evaluations and still $0.272$ at $64$; UniPC's total at $8$ evaluations, $0.238$, is already below Euler's at $64$, and from $16$ on it does not move ($0.229$, $0.229$, $0.23$), because the whole of its remaining error is the field's. The same happens at $\mathrm{amp} = 0.6$, where UniPC reaches the floor of $0.465$ by $32$ evaluations. Beyond that, further evaluations do nothing. A sampler's step count is bounded below by the solver and above by the model, and the right place to stop is where the two meet; the toy puts the meeting at $16$ to $32$ evaluations for a good solver. A faster solver is a way to *reach* the floor cheaply, and it is a different problem from lowering it, which is the job of training.

## A second pass for the prompt

The other factor in the opening table is the second pass. Everything so far has sampled from the data without being told what to draw. A model that is given a prompt is trained on pairs, and its denoiser takes the condition $c$ as one more argument, with the guess $\hat x_0(x, \sigma, c)$ and the field $u_c(x, \sigma)$. Sampling with it as it stands is sampling from $p(x \mid c)$ (the exact conditional field carries the standard normal onto the conditional distribution by the argument of Chapter 6), and the result follows the prompt only as closely as the model's training allowed. A model of the world is also a model of what the average caption looks like, and the conditional distribution contains many images that are only loosely related to the prompt. Can the sampler be made to follow the prompt harder than the conditional distribution does?

A problem small enough to be exact is needed, in which the classes overlap. Two classes, equally likely: class $0$ is $\mathcal{N}(-1, 0.6^2)$ and class $1$ is $\mathcal{N}(+1, 0.6^2)$. The noised class $c$ at noise fraction $\sigma$ is Gaussian with mean $(1 - \sigma)\mu_c$ and variance $(1 - \sigma)^2 0.36 + \sigma^2$, so the class field $u_c$ has a closed form, and the field of the mixture, the *unconditional* field $u_{\mathrm{un}}$ (the model asked with no prompt), is the class fields weighted by the probability of each class at the point. Class $1$ is the "prompt" below; $1000$ starting values are drawn from the stream `samplers_and_guidance.guidance`, one `normal()` each. The two classes share a region around $x = 0$, in which a sample from class $1$ lands on the wrong side of $0$ with probability $\Phi(-1/0.6) = 0.048$.

Bayes' rule (Volume I, conditional probability) writes the class density as $p(x \mid c) = p(c \mid x)p(x)/p(c)$, and so the score of the conditional distribution is the score of the unconditional one plus the gradient of a classifier:

$$
\nabla_x \ln p_\sigma(x \mid c) = \nabla_x \ln p_\sigma(x) + \nabla_x \ln p_\sigma(c \mid x).
$$

The last term points in the direction in which the point looks more like class $c$. Taking more of it than the conditional distribution contains is **classifier guidance**: use the score $s_{\mathrm{un}} + g\,\nabla\ln p_\sigma(c\mid x)$ with $g > 1$ ($g = 1$ is the conditional score). It needs a classifier $p(c \mid x_\sigma)$ that works on noisy inputs, which is a second network to train. The remark that makes the second network unnecessary is that the classifier's gradient is already the difference of two scores that the generator can give: $\nabla\ln p_\sigma(c \mid x) = s_c - s_{\mathrm{un}}$, the conditional score minus the unconditional one. The generator gives both if it is trained with the condition replaced by a "null" condition on a small fraction of the examples, so that the same network can be asked with and without the prompt. Using the difference with weight $g$,

$$
s_g = s_{\mathrm{un}} + g\,(s_c - s_{\mathrm{un}}) ,
$$

is **classifier-free guidance**: the prediction with the prompt, pushed further from the prediction without it. Because $u = -(x + \sigma s)/(1 - \sigma)$ (Chapter 6) is affine in the score, and the weights $1 - g$ and $g$ of the blend add to one, the $x$ terms cancel and the same blend can be taken on the velocity,

$$
u_g = u_{\mathrm{un}} + g\,(u_c - u_{\mathrm{un}}) ,
$$

or on the guess $\hat x_0$ (since $u = (x - \hat x_0)/\sigma$ and the $x$ cancels in a blend whose weights add to one). With $g = 1$ this is the conditional field, with $g = 0$ the unconditional one, and with $g > 1$ it extrapolates beyond the conditional one. The three identities on which this rests are checked on the toy:

<!-- worked: samplers_and_guidance.guidance_identities -->
Identities of the two-class problem (classes N(-1, 0.6^2) and N(+1, 0.6^2), equal weights; noised classes N((1 - sigma) mu_c, (1 - sigma)^2 0.36 + sigma^2)), at the 24 points sigma in {0.2, 0.5, 0.8}, x = 0.5 (k - 4)(1 - sigma) + 0.37 (k mod 3) for k = 0 .. 7, with scores taken as central differences of the log densities (step 1e-5):

| identity | largest difference |
|---|---:|
| unconditional velocity = sum over c of p(c given x) times the class velocity | below 1e-12 |
| guided velocity (class 1, g = 3) = -(x + sigma s_g)/(1 - sigma) with s_g = s + g (s_1 - s) | below 1e-06 |
| s_1 - s = d/dx ln p(class 1 given x) | below 1e-06 |
<!-- /worked -->

In pseudocode, one guided evaluation of the velocity is two evaluations of the denoiser and a blend:

```text
function guided_velocity(guess_class: function(real, real, int) -> real, guess_plain: function(real, real) -> real, c: int, g: real, x: real, sigma: real) -> real:
    require sigma > 0, "the velocity is a quotient by sigma"
    plain = guess_plain(x, sigma)
    cond = guess_class(x, sigma, c)
    return (x - (plain + g * (cond - plain))) / sigma
```

For `g == 1` the blend is `cond` itself, so the unconditional pass can be skipped without error; brain's Wan and SDXL pipelines skip it for any scale of $1$ or less.

What does the sampler produce? Carry the $1000$ starting values from $\sigma = 1$ to $0$ with the guided field (Runge-Kutta with $100$ steps, a reference whose error is below $10^{-7}$) and describe the end points. There is also a prediction to test. The score $s_g$ is the score of the density proportional to $p_\sigma(x \mid c)^g\,p_\sigma(x)^{1-g}$ at each noise level, which is the class density multiplied by the classifier's probability raised to $g - 1$; one may expect the samples to follow that tilted density at $\sigma = 0$. The right-hand columns of the table are the tilted density's statistics, computed by a sum on a fine grid.

<!-- worked: samplers_and_guidance.guidance_table -->
Class 1 (N(+1, 0.6^2)) under the guided field u_g = u + g (u_1 - u) with the exact fields of both branches: 1000 starting values (one normal() each, from `samplers_and_guidance.guidance` (seed 0xc9647ffa2fbb1077)) carried from sigma = 1 to 0 by 100 steps of classical Runge-Kutta. 'Wrong side' is the share of end points below 0; 'class' is the mean of p(class 1 | x_0), the probability a classifier with the exact densities gives to the right class. The last three columns are the same statistics for the density proportional to p(x | class 1)^g p(x)^(1 - g), computed by a Riemann sum over 4001 points of [-5, 5]. The class itself has mean 1, standard deviation 0.6 and wrong-side share 0.048.

| g | mean | std | wrong side | class | density: mean | density: std | density: wrong side |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.025 | 1.172 | 0.500 | 0.4988 | 0.000 | 1.166 | 0.500 |
| 1 | 0.982 | 0.603 | 0.058 | 0.9200 | 1.000 | 0.600 | 0.048 |
| 1.5 | 1.215 | 0.484 | 0.000 | 0.9860 | 1.049 | 0.562 | 0.024 |
| 2 | 1.377 | 0.430 | 0.000 | 0.9966 | 1.077 | 0.543 | 0.013 |
| 3 | 1.619 | 0.375 | 0.000 | 0.9994 | 1.111 | 0.522 | 0.005 |
| 5 | 1.968 | 0.328 | 0.000 | 0.9999 | 1.150 | 0.501 | 0.001 |
| 8 | 2.350 | 0.297 | 0.000 | 1.0000 | 1.186 | 0.484 | 0.000 |

Doubling the steps of the reference (200) moves no end point of g = 8 by more than 4e-08.
<!-- /worked -->

![Histograms of the end points of class 1 under guidance scales 0, 1, 3 and 8 together with the class's own density: g = 0 is two bumps around minus one and plus one, g = 1 follows the class around plus one, and g = 3 and g = 8 are narrower and sit further right, around 1.6 and 2.4.](img/guidance-samples.svg)

Three things stand out. First, guidance does what it is for: the share of samples on the wrong side of $0$ falls from $0.058$ at $g = 1$ (a sample of $1000$ around the class's $0.048$) to zero for $g \geq 1.5$, and a classifier with the exact densities gives the right class a mean probability of $0.920$ at $g = 1$ and $0.9994$ at $g = 3$. Second, it costs diversity: the standard deviation of the end points falls from $0.603$ at $g = 1$ to $0.375$ at $g = 3$ and $0.297$ at $g = 8$, against the class's own $0.6$. The samples are more typical of the prompt and less varied, which is why a pipeline chooses a scale and does not take the largest. Third, the samples move. At $g = 8$ the mean of the end points is $2.35$, more than two class standard deviations beyond the class's mean of $1$, in a region where the class has almost no density. This is the one-dimensional picture of what practitioners call oversaturation: at a high scale the generator is told to be more like the prompt than any image of the prompt is.

The right-hand columns say that the claim "guidance samples from $p(x \mid c)^g\,p(x)^{1-g}$" is not what happens. At $g = 5$ the tilted density has mean $1.150$ and standard deviation $0.501$; the samples have $1.968$ and $0.328$. The tilted density is a perfectly good density with a large part of its mass near the class; the samples are pushed further. The reason is that the guided score is the score of the tilted density at *each noise level separately*, and the family of those densities is not the probability path of any distribution: it does not satisfy the continuity equation of Chapter 6, so following its scores does not carry the noise onto the tilted density at $\sigma = 0$. The guided field is a recipe for steering, and its effect is measured as above, not read off a formula.

When the scale acts matters, and the toy shows it. A guidance **ramp** makes the scale a function of the noise level; brain's SUPIR sampler uses a linear one.

<!-- worked: samplers_and_guidance.ramp_table -->
Guidance on a schedule, class 1, the starting values and the Runge-Kutta reference of the last table. The scale at noise fraction sigma is given in the first column; a scale of 1 is the plain class field.

| scale as a function of sigma | mean | std | wrong side | class |
|---|---:|---:|---:|---:|
| constant 5 | 1.968 | 0.328 | 0.000 | 0.9999 |
| 1 at noise, 5 at data: 1 + 4 (1 - sigma) | 1.410 | 0.365 | 0.000 | 0.9985 |
| 5 at noise, 1 at data: 1 + 4 sigma | 1.823 | 0.372 | 0.000 | 0.9998 |
| 5 only below sigma = 0.5 | 1.249 | 0.363 | 0.000 | 0.9970 |
| 5 only above sigma = 0.5 | 1.951 | 0.341 | 0.000 | 0.9999 |
<!-- /worked -->

Guidance applied only above $\sigma = 0.5$ ($1.951$, $0.341$) does almost all of the work of constant guidance ($1.968$, $0.328$), and guidance applied only below $0.5$ does little ($1.249$): the prompt is decided while the noise is high, and the late steps only polish. The ramp from $1$ at the noise end to $5$ at the data end, which is the shape of the ramp of `crates/diffusion/src/restore.rs` (from $1$ to $4$ at its defaults), keeps most of the adherence ($0.9985$ against $0.9999$) with a mean of $1.410$ against $1.968$, much closer to the class's $1$. Choosing a ramp is choosing how much of the push to take where it does least damage to the look of the result, and the toy does not say which choice is best for pictures; it says that the choice moves the output.

There is a cost in accuracy as well as in passes. A model with a small error in its conditional prediction, and an exact unconditional one, passes the error on to the guided field multiplied by $g$, because $u_g = u_{\mathrm{un}} + g(u_c + \delta - u_{\mathrm{un}}) = u_g^{\text{exact}} + g\delta$. The table measures how far the end points move when $\delta = 0.1\,\sigma\sin(3\,x + 2\sigma)$ is added to the class-$1$ guess.

<!-- worked: samplers_and_guidance.error_amplification -->
An error in one branch. The class-1 guess is replaced by the exact one plus 0.1 sigma sin(3 x + 2 sigma) (library sin); the plain guess stays exact. For each g, the mean distance between the end points of the wrong field and the right one (the same starting values and reference as the guidance table), and that distance relative to g = 1.

| g | mean displacement | relative to g = 1 |
|---|---:|---:|
| 1 | 0.0523 | 1.0 |
| 2 | 0.0853 | 1.6 |
| 3 | 0.138 | 2.6 |
| 5 | 0.297 | 5.7 |
| 8 | 0.418 | 8.0 |
<!-- /worked -->

The displacement is $0.0523$ at $g = 1$ and $0.418$ at $g = 8$, exactly $8.0$ times as much. This is the arithmetic behind knowledge #192, which reports the same effect at the scale of a real model: Janus-Pro's two branches, run with $8$-bit linear layers, each agreed with a full-precision reference to a cosine of $0.996$ to $0.9998$, while the guided blend at the first step agreed to $0.848$, because the blend is dominated by $g$ times a small difference (the numbers are the entry's own measurement and are not reproduced here). The entry's remedy was a higher-precision weight format for the model, and its rule, to gate a guided sampler on the guided quantity itself, follows from the table: whatever tolerance a branch is held to, the guided prediction is held to $g$ times less.

## Folding the second pass into the network

If the guided field is a function, a network can be trained to be that function. This is **guidance distillation**: the student network is trained, by the regression of Chapter 6, on the guided prediction of a teacher instead of the plain one, so that one pass of the student does the work of the teacher's two. The scale is either fixed at distillation time or given to the student as an input, in which case the student is also asked to learn how its answer changes with the scale.

The toy distills the guided field of class $1$ at $g = 3$ into the network of Chapter 6 (two inputs, two hidden layers of $32$ units with $\tanh$, one output; its initialisation is the one of Chapter 6, from the stream named in the table), with the input $(x_\sigma, \sigma)$ and no class input, since there is one class. A training item is drawn as $\sigma = \mathtt{uniform}()$, $x_0 = 1 + 0.6\,\mathtt{normal}()$, $\varepsilon = \mathtt{normal}()$ in this order, the point $x_\sigma = (1 - \sigma)x_0 + \sigma\varepsilon$ and the target $u_g(x_\sigma, \sigma)$. The network is trained for $3000$ steps of $64$ items by Adam with the settings of Chapter 5. The teacher is closed-form, so its two "passes" are two evaluations of a formula; the table counts them as passes, to show what the saving is when they are network passes.

<!-- worked: samplers_and_guidance.student_table -->
Guidance distilled into one network. Teacher: the guided field of class 1 with g = 3 (two evaluations per step). Student: the network of `diffusion.init_net` from the stream `samplers_and_guidance.student` (seed 0x45149b3e40b85b1f), input (x_sigma, sigma), trained 3000 steps of 64 items (sigma = uniform(), x_0 = 1 + 0.6 normal(), noise = normal(), per item in this order, then x_sigma = (1 - sigma) x_0 + sigma noise) on the teacher's velocity at x_sigma, Adam with beta1 0.9, beta2 0.999, epsilon 1e-8 and rate 0.003 (1 - (i - 1)/3000). Digits are those shared by the run and four copies with the first weight changed by +1e-12, -1e-12, +3e-12 and -3e-12. The score is the mean of |x_0 - reference| over the 1000 starting values of the guidance tables, the reference being the teacher with 100 Runge-Kutta steps; Euler steps on the uniform grid for both.

| evaluations | teacher: steps | teacher: distance | student: steps | student: distance |
|---|---:|---:|---:|---:|
| 4 | 2 | 0.166 | 4 | 0.107 |
| 8 | 4 | 0.123 | 8 | 0.054 |
| 16 | 8 | 0.0657 | 16 | 0.025 |
| 32 | 16 | 0.0335 | 32 | 0.012 |

After 32 evaluations the teacher's end points have mean 1.619, standard deviation 0.375 and the student's 1.626 and 0.368. The student's mean squared error against the teacher's velocity is 0.0027 on 1000 fresh items of the training distribution (`samplers_and_guidance.student_eval` (seed 0xc5925f503052090e)), and on the states the teacher's own 20-step trajectories visit it is 0.0008 at sigma = 0.9, 0.0038 at 0.5 and 0.0018 at 0.1.
<!-- /worked -->

At every evaluation count the student is closer to the teacher's exact end points than the teacher is, by a factor of $1.6$ to $2.8$ ($0.107$ against $0.166$ at $4$ evaluations, $0.012$ against $0.0335$ at $32$), because the student spends each evaluation on a step of its own and the teacher spends two on each. The student's end points match the teacher's statistics ($1.626$ and $0.368$ against $1.619$ and $0.375$), and its error against the teacher's velocity is $0.0027$ on items of the training distribution and between $0.0008$ and $0.0038$ on the states that the teacher's own trajectories visit, which are not distributed like the training items: the guided teacher drives samples to a region (the mean of $1.6$) that the class path visits seldom. The student was trained on the class path only and is tested on states the teacher visits, and the two errors are of the same size, so for this problem that mismatch is not what limits it; with a student that has to be right at all scales it is the thing to measure.

What the student gives up is the scale: in this toy it is baked in at $g = 3$. brain's models show the two designs side by side. FLUX.1-dev takes the scale as an input: `crates/flux1/src/model.rs` builds its conditioning vector as `time_in(t) + guidance_in(g) + vector_in(pooled)`, with the guidance scalar embedded by the same kind of sinusoidal features as the timestep and sent through its own two-layer network (`guidance_in.in_layer`, `guidance_in.out_layer` in `crates/flux1/src/config.rs`), and the config documents the checkpoint as "guidance-distilled". FLUX.2 Klein bakes the scale in and drops the input: `crates/flux2/src/config.rs` says that the distilled variants run four Euler steps, no guidance, and carry no guidance embedding, and the undistilled base variants of the same shapes run fifty steps with real guidance. Commit 00140b6fc made the pipeline refuse to behave as though that were a choice: `resolved_steps` fixes the count at four for the distilled variants unless the caller opts in with an experimental flag, because the model was distilled for that schedule and any other count integrates "a schedule the model was never distilled for", and a guidance above one is warned about as a no-op. The third and fourth rows of the table at the top of this chapter are the result: $100$ passes against $4$.

Distillation does not remove the null branch from the training of the undistilled model, and the code has a trace of that. The fine-tuner of FLUX.2 has an option, `ref_dropout`, that blanks the reference image of an editing pair on a fraction of the steps (commit 97704014f); its documentation says that an editing method with guidance blanks the conditioning image on a few percent of its steps to train the null branch that guidance evaluates at inference, and that for the distilled Klein, which has no such branch, the dropped steps "spend part of a step budget on a mode the sampler never enters". This is the training side of the "null condition" of the previous section.

## Where the samplers and the guidance live in brain

The solvers are in `crates/diffusion`, and none of them calls a network: each is host arithmetic on the vectors that a pipeline passes in and out. `crates/diffusion/src/discrete.rs` holds the family of the variance-preserving chain: `DdimScheduler`, the Euler and Euler-ancestral schedulers in the rescaled variable, and the DPM-Solver++ multistep scheduler of order $1$ or $2$ with the midpoint or the Heun weight. The header of the file names the classic bug, a mix of two coordinate systems: DDIM works with $\bar\alpha$ directly and needs no input scaling, the Euler family works in the rescaled variable and must divide the latent by $\sqrt{\sigma^2 + 1}$ before each network call, and DPM-Solver++ keeps the variance-preserving latent and converts inside `x0_from_vp`, so that its `scale_model_input` is the identity. Commit cb0b83f62 reports that feeding the Euler family's conversion to DPM-Solver++ gave a different trajectory, off by a relative $1.1$ from its reference, and that the Euler family's input scaling alone was off by $0.81$ on the very first step; the reference values are not in the repository, so the numbers are the commit's own claim, but the mechanism is in the code and is the identity of the first section: the clean value is $(x_t - b_t\hat\varepsilon)/a_t$ in the variance-preserving latent and $y - \sigma_t\hat\varepsilon$ in the rescaled one, and which of the two a function receives is not visible in its type.

The ancestral scheduler leaves the random draw to the caller. `step_with_noise` takes the normal vector as an argument, and the module header gives the reason: an embedded generator would make the scheduler impossible to test against a dumped reference trajectory, and `step`, the noise-free case, is documented as "not the same sampler", only an isolation of its deterministic half for a parity test. The test `ancestral_split_is_a_variance_split` pins the property that this chapter's table shows: $\sigma_{\text{up}}^2 + \sigma_{\text{down}}^2 = \sigma_{t'}^2$ at every step.

`crates/diffusion/src/flowsolvers.rs` holds the same two solver ideas for the straight path, ported for Wan: `FlowDpmSolverPlusPlusScheduler` and `FlowUniPcScheduler`. The module header explains why they are separate from the discrete family (there is no $\bar\alpha$ to build $\lambda$ from; $\alpha = 1 - \sigma$), and `lambda_of` is the log signal-to-noise ratio of this chapter. The predictor of order $2$ uses the hard-coded weight $\tfrac12$ and the corrector of order $1$ the same, and everything else solves the small system with `solve`; the comments say that the $\tfrac12$ is a simplification equal to the exact weight only as $h \to 0$, which the first rows of the coefficient table above confirm ($b_1 = 0.504$ at $h = 0.05$, $0.598$ at $h = 1.2$). The commit that added them, 303c4d4ff, reports step trajectories within $5.2\times 10^{-6}$ of a reference and that forcing order $1$ or skipping the corrector moved them by $0.28$ and $0.075$, which is a claim about dumped goldens that cannot be checked here but fits the table: the predictor is the midpoint step, so dropping the corrector leaves DPM-Solver++(2M), and UniPC is ahead of it by a factor of $3.4$ at $8$ evaluations, $13$ at $16$ and $4.8$ at $32$.

Churn, the other way of putting noise back, is in `crates/diffusion/src/restore.rs`, the scalar math of SUPIR's sampler: a first-order Euler sampler on SDXL's own chain with churn, an optional pull toward a clean reference and a guidance ramp (the module header notes that, despite its name, it is not the Karras sampler). Before the step at noise level $\sigma_t$ it raises the level to $\hat\sigma = \sigma_t(1 + \gamma)$ by adding noise of standard deviation $s_{\text{noise}}\sqrt{\hat\sigma^2 - \sigma_t^2}$, and then denoises from $\hat\sigma$:

<!-- worked: samplers_and_guidance.churn_table -->
Churn, the other way to put noise back: before the step at noise level sigma the sampler raises it to sigma_hat = sigma (1 + gamma), with gamma = min(s_churn / (n - 1), sqrt(2) - 1) for n steps, by adding noise of standard deviation s_noise sqrt(sigma_hat^2 - sigma^2) to the variance-exploding latent, and then denoises from sigma_hat. For s_noise = 1.01:

| s_churn | steps n | gamma | sigma_hat / sigma | added noise / sigma |
|---|---:|---:|---:|---:|
| 5 | 50 | 0.10204 | 1.10204 | 0.46777 |
| 5 | 10 | 0.41421 | 1.41421 | 1.01000 |
| 20 | 10 | 0.41421 | 1.41421 | 1.01000 |
<!-- /worked -->

With the defaults of 50 steps and `s_churn = 5` each step first adds about half of its own level in noise ($0.468\,\sigma_t$) and then takes a step from the higher level. The cap $\gamma \leq \sqrt{2} - 1$ keeps the added noise at $s_{\text{noise}}\sigma_t$. In the same file, `linear_cfg_scale` is the ramp of the previous section as a function of its two end values, taken as plain arguments on purpose: commit a96c52ec2 says that the reference's constructor names them the opposite way from the ramp's shape, so that getting them wrong silently inverts the ramp. The call site in `crates/supir/src/pipeline.rs` passes `spt_linear_cfg` (the scale at the noise end, default $1$) and `cfg_scale` (at the data end, default $4$): guidance rises from none to four as the noise falls, the "$1$ at noise, $5$ at data" row of the ramp table.

Guidance itself is spread over the model crates, each at the point where its pipeline holds both predictions. `crates/sdxlunet/src/sampler.rs` calls its `Denoiser` twice per step because its graph is recorded for one sample, and says so; the loop was factored out of two near-copies in commit 0f1809375 so that a restoration pipeline would not be a third. `crates/wan/src/pipeline.rs` skips the unconditional pass when the scale is at most $1$, which the doc comment calls exact. `crates/flux2/src/pipeline.rs` puts the conditional and the unconditional passes of a step into one batch as two slots (the work is still two passes), and its unconditional prompt is the empty string. `crates/flux1/src/pipeline.rs` has an optional "true" CFG on top of the distilled scalar, a second pass per step on a negative prompt (commit 7c9006d54), so that FLUX.1-dev costs $50$ passes by default and $100$ with it. And `crates/ltxv/src/guidance.rs` is the richest: one step may run four passes, and the guided prediction is formed on the clean-value estimate, where a further "rescale" term pulls the guided prediction's spread over the tokens back toward the conditional one's, an example of a correction that is not linear in the passes and so cannot be done on the velocity, as the file says.

The blend `u + g (c - u)` itself was once written inline in four places. Commit 4f5870376 collected four copies of it (a private helper of the music model, and inline blends in that model's denoiser, in Wan and in the LTX pipeline of the time, which has since grown its own guidance module) into `model::hostmath::cfg_blend`, which also checks that the two branches have the same length, which the inline `zip`s silently truncated; its test pins `g = 1` to the conditional branch, `g = 0` to the unconditional one, and `g = 3` to an extrapolation. The helper's own documentation observes that the autoregressive samplers apply the same blend to token logits, which is what `crates/januspro/src/t2i.rs` does: guidance is a property of conditional generative models, and diffusion is the case this chapter works in. Writing this chapter turned up the opposite failure in `crates/s3dit`: its served actions offered a guidance scale and a negative prompt, and its sampler made one conditional pass a step and read neither, so a request for the non-distilled base model with a scale of 5 sampled unguided and said nothing. Commit `d9145fbc2` withdrew both parameters, so such a request is now refused as naming an unknown parameter, until a second pass and the blend are built.

## Check your understanding

1. Find the error: "DDIM at $\eta = 0$ is a different sampler from Euler's method on the straight path, because it takes the guess and re-noises it instead of following the velocity."
2. Predict: a deterministic sampler is run from the same starting value with $10$ steps and with $20$. Must the two end points agree, differ by about the solver's error, or differ at random? What is different about a stochastic sampler run twice with the same step count?
3. A step goes from $\sigma_i = 0.7$ to $\sigma_{i+1} = 0.5$. Compute $h$, the coefficient $(1 - \sigma_{i+1})(1 - e^{-h})$ of the guess in the first-order step and check that it equals $1 - \sigma_{i+1}/\sigma_i$, and the weight $b_1(h)$ of the second-order correction.
4. Why is the second step of a DPM-Solver++(2M) run that starts at $\sigma = 1$ first order, and why does UniPC's grid avoid the situation?
5. At $4$ evaluations Heun scores worse than Euler. Why does that not contradict Heun being second order?
6. A pipeline lets the user pass `guidance = 0.5`. Which field does the blend give, and what does `guidance <= 1.0` mean for the number of passes?
7. The toy's samples at $g = 5$ have a standard deviation of $0.328$ and the tilted density's is $0.501$. A colleague says "so the sampler is wrong". What would you reply?
8. A conditional branch is wrong by $0.01$ at some point and the unconditional one is exact. By how much is the guided prediction wrong at $g = 7.5$, and what does that say about the tolerance of a parity test on the branches?

## Summary

A step of a sampler turns a denoiser's guess into the next point. The deterministic step re-noises the guess to the new level with the noise the old level implies; it is Euler's method on the straight path and Euler on a rescaled variable on the variance-preserving chain, and a sampler made of it is deterministic, one output for one noise. Keeping only part of the implied noise and adding fresh noise for the rest gives a family, indexed by $\eta$, that keeps the marginals of the path; $\eta = 1$ is the exact reverse step of Chapter 5 and the Euler ancestral sampler. With an exact denoiser noise only costs accuracy; with a wrong one it helps.

Writing the equation in the log signal-to-noise ratio integrates its stiff part exactly and leaves only the slowly varying guess to approximate. A constant guess gives Euler's step; a line through the last two guesses gives the second-order multistep method DPM-Solver++(2M) at one evaluation per step; a corrector that reuses the next evaluation as one more point gives UniPC. On the exact field of three clusters the higher-order solvers reach in $16$ evaluations what Euler needs $64$ for, and a model error sets a floor that no solver goes below, around $16$ to $32$ evaluations for a good solver in the toy. The first and last steps are first order, and a grid that starts at $\sigma = 1$ loses the second step as well.

Classifier-free guidance blends the prediction with the prompt and the prediction without it, $u_g = u_{\mathrm{un}} + g(u_c - u_{\mathrm{un}})$. It follows the prompt harder and costs a second pass per step; its samples are less varied, drift past the class, and are not distributed as the tilted density whose score it uses; the late steps matter little for what it decides; and an error in a branch is multiplied by $g$. Distilling the guided prediction into one network removes the second pass, at the price of fixing the scale or of giving the network the scale as an input, and brain's FLUX.2 Klein and FLUX.1-dev are the two designs.

The chapter used a closed-form denoiser throughout, which isolates the sampler and the guidance from the network. The next chapter opens the network: how a denoiser is built so that it can take the noise level, the prompt and the image together, the U-Net of SDXL and the diffusion transformers of FLUX and Z-Image.

## References

- Jiaming Song, Chenlin Meng and Stefano Ermon, *Denoising Diffusion Implicit Models*, 2020. The deterministic step and the eta family.
- Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li and Jun Zhu, *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models*, 2022. The data-prediction exponential integrators and the multistep second-order method.
- Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou and Jiwen Lu, *UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models*, 2023. The predictor and the corrector of the same method.
- Jonathan Ho and Tim Salimans, *Classifier-Free Diffusion Guidance*, 2022. Training with the condition dropped and the blend of the two predictions.
- Prafulla Dhariwal and Alex Nichol, *Diffusion Models Beat GANs on Image Synthesis*, 2021. Guidance by the gradient of a classifier.
- Chenlin Meng and others, *On Distillation of Guided Diffusion Models*, 2022. Distilling the guided prediction into one network.
- Tero Karras, Miika Aittala, Timo Aila and Samuli Laine, *Elucidating the Design Space of Diffusion-Based Generative Models*, 2022. Churn, and the separation of the solver from the network.

## Exercises

1. **[Calculate]** On the straight path with the guess $\hat x_0 = 2$ at $x = 1$, $\sigma = 0.5$, compute the implied noise $\hat\varepsilon$ and the deterministic step to $\sigma' = 0.25$ by "re-noise the guess"; then compute the Euler step $x + (\sigma' - \sigma)(x - \hat x_0)/\sigma$ and check that they agree.
2. **[Calculate]** For the step $\sigma = 0.6 \to 0.4$ of the noise-split table, compute $\nu_1$ from the formula and the weight $\kappa$ of the implied noise at $\eta = 0.5$, and check $\kappa^2 b^2 + \nu^2 = b'^2$.
3. **[Calculate]** Show that $\lambda(\sigma') = \lambda(\sigma) - \ln k$ for $\sigma' = k\sigma/(1 + (k - 1)\sigma)$, and that two shifts $k_1$ and $k_2$ compose to $k_1 k_2$.
4. **[Calculate]** Derive $I_1(h) = h - 1 + e^{-h}$ and $I_2(h) = h^2 - 2h + 2 - 2e^{-h}$ from $I_j = h^j - jI_{j-1}$, and evaluate $b_1$ and $b_2$ for $h = 0.3$ (compare with the coefficient table).
5. **[Calculate]** Take the second-order step of DPM-Solver++(2M) in the midpoint form from $\sigma_i = 0.5$ to $\sigma_{i+1} = 0.35$, with the previous level $\sigma_{i-1} = 0.65$, the guesses $m_i = 1.2$ and $m_{i-1} = 1.0$ and the current point $x_i = 0.8$. Give $h$, $h_0$, $r_0$ and the new point.
6. **[Calculate]** In the guidance table the class has mean $1$ and standard deviation $0.6$. How many class standard deviations beyond the class's mean is the mean of the samples at $g = 8$, and what is the ratio of the standard deviation of the samples at $g = 3$ to the class's?
7. **[Implement]** Reproduce `samplers_and_guidance.ddim_is_euler` and `samplers_and_guidance.noise_split` to the printed digits, using the chain of Chapter 5.
8. **[Implement]** Reproduce the Euler, Heun and DPM-Solver++(2M) columns of `samplers_and_guidance.solver_table` from the stream `flow_matching.sample`, and `samplers_and_guidance.step_orders`.
9. **[Implement]** Reproduce the UniPC column of `samplers_and_guidance.solver_table`, with the corrector of order $1$ and $2$ and the rule for a history point at $\sigma = 1$, and `samplers_and_guidance.step_coefficients`.
10. **[Implement]** Reproduce the rows $g = 1, 3, 8$ of `samplers_and_guidance.guidance_table` from the stream `samplers_and_guidance.guidance`, and the rows of `samplers_and_guidance.ramp_table`.
11. **[Investigate]** Replace the guess-space blend of the toy by the blend of the guided velocity $u_g$ (the two are the same on the straight path) and then by a blend of the noise estimates in the variance-preserving chain of Chapter 5. Does the blend commute with the change of parametrization, and where does the equivalence fail?
12. **[Investigate]** Train the student of `samplers_and_guidance.student_table` with the scale as a third input, $g$ drawn uniformly from $[1, 6]$ per item. Compare its error at $g = 3$ with the fixed-scale student's and say what the extra input costs.

## Key terms

- **deterministic sampler**: a sampler whose output is a function of its starting noise alone, with no random draw after the first, so that the same noise always gives the same image; the step that re-noises the denoiser's guess to the next level with the noise the old level implies, and Euler's method on the velocity field, are such samplers. <!-- concept: deterministic-sampler -->
- **ODE solver**: a rule that turns evaluations of a velocity field into the next point of a trajectory, judged by how fast its error falls as the steps shrink (its order) and by how many evaluations of the field each step costs; Euler, Heun and the multistep methods of this chapter are examples. <!-- concept: ode-solver -->
- **classifier-free guidance**: sampling with the prediction $u_{\mathrm{un}} + g(u_c - u_{\mathrm{un}})$, the prediction made with the prompt pushed away from the prediction made without it by a scale $g$; $g = 1$ is the plain conditional model, and a larger $g$ follows the prompt harder at the cost of variety and of a second network pass per step. <!-- concept: classifier-free-guidance -->
- **guidance distillation**: training a student network by regression on the guided prediction of a teacher, optionally with the guidance scale as an input, so that one pass of the student gives what two passes of the teacher gave. <!-- concept: guidance-distillation -->

## Answers

**Recall questions.**

1. The posterior mean of the clean value, $x - \sigma u(x, \sigma) = \mathbb{E}[x_0 \mid x_\sigma = x]$, the guess of the denoiser.
2. Two evaluations per step. Its global error falls by a factor of about $4$ when the step is halved (second order), against about $2$ for Euler.
3. $P(1 \mid x) = 0.9 \cdot \tfrac12 / (0.9\cdot\tfrac12 + 0.3\cdot\tfrac12) = 0.75$.
4. The variance $\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\beta_t$ of the exact reverse step given the clean value.
5. $\alpha^2 + \beta^2$: the variances of independent Gaussians add.
6. To $3 \cdot 0.5/(1 + 2 \cdot 0.5) = 0.75$.

**Check your understanding.**

1. The two are the same rule, as the first table of the chapter shows to rounding error: substituting the implied noise into the re-noising step gives $(\sigma'/\sigma)x + (1 - \sigma'/\sigma)\hat x_0$, which is Euler's step.
2. They differ by about the solver's error: the rule and the starting value are fixed, so there is no randomness, and the difference falls as the step count grows. A stochastic sampler run twice at the same step count differs unless it is given the same fresh noise; in the eta table its score at $\eta = 1$ stays above $0.05$ at $64$ and $128$ steps, while $\eta = 0$ reaches $0.0295$ and $0.018$.
3. $h = \lambda(0.5) - \lambda(0.7) = 0 + 0.847298 = 0.847298$, $e^{-h} = 3/7$, so $0.5\,(1 - 3/7) = 0.285714$, which is $1 - 0.5/0.7$. With $I_0 = 4/7$ and $I_1 = h - 1 + e^{-h} = 0.275870$, $b_1 = I_1/(hI_0) = 0.5698$.
4. Because $\lambda(1) = -\infty$, the gap between the first two guesses is infinite, $r_0 = h_0/h$ is infinite and the difference $D_1 = (m_i - m_{i-1})/r_0$ is $0$. UniPC's grid starts at $0.99900001$, where $\lambda$ is finite.
5. Order says how the error falls as steps shrink, not that a second-order method is better at any step count: with two steps the expansion that gives the order does not hold across the merge of the clusters, and the constant in front of the error may be large.
6. $g = 0.5$ gives $u_{\mathrm{un}} + 0.5(u_c - u_{\mathrm{un}})$, halfway between the unconditional field and the conditional one, so a weaker prompt than plain conditioning. `guidance <= 1.0` skips the unconditional pass in brain's Wan and SDXL pipelines; for $g < 1$ that is not exact, since the blend is not the conditional prediction.
7. That the tilted density's score is the guided score at each noise level but the family of tilted densities is not a probability path, so the samples need not follow it; the statistic that matters is whether the samples follow the prompt and how varied they are, which the table measures directly.
8. $7.5 \times 0.01 = 0.075$. A parity test on the branches at tolerance $\epsilon$ guarantees only $g\epsilon$ on the guided prediction, so the tolerance for a guided sampler must be set on the guided quantity (knowledge #192).

**Exercises (calculation answers).**

1. $\hat\varepsilon = (1 - 0.5\cdot 2)/0.5 = 0$; the step is $(1 - 0.25)\cdot 2 + 0.25\cdot 0 = 1.5$; the Euler step is $1 + (0.25 - 0.5)(1 - 2)/0.5 = 1.5$.
2. $\nu_1 = 0.358323$; at $\eta = 0.5$, $\nu = 0.179161$, $\kappa = \sqrt{0.16 - 0.032099}/0.6 = 0.5961$, and $\kappa^2 b^2 + \nu^2 = 0.1279 + 0.0321 = 0.16$.
3. $(1 - \sigma')/\sigma' = (1 + (k-1)\sigma - k\sigma)/(k\sigma) = (1 - \sigma)/(k\sigma)$, so $\lambda(\sigma') = \lambda(\sigma) - \ln k$; two shifts subtract $\ln k_1 + \ln k_2 = \ln(k_1 k_2)$.
4. $I_1 = h^1 - 1\cdot I_0 = h - 1 + e^{-h}$ and $I_2 = h^2 - 2I_1 = h^2 - 2h + 2 - 2e^{-h}$. At $h = 0.3$: $b_1 = 0.524963$ and $b_2 = 0.358545$.
5. $\lambda(0.65) = -0.619039$, $\lambda(0.5) = 0$, $\lambda(0.35) = 0.619039$, so $h_0 = h = 0.619039$ and $r_0 = 1$; $D_1 = (1.2 - 1.0)/1 = 0.2$; $(1 - \sigma_{i+1})(1 - e^{-h}) = 1 - 0.35/0.5 = 0.3$; $x_{i+1} = 0.7\cdot 0.8 + 0.3\,(1.2 + 0.5\cdot 0.2) = 0.56 + 0.39 = 0.95$.
6. $(2.350 - 1)/0.6 = 2.25$ class standard deviations; the ratio at $g = 3$ is $0.375/0.6 = 0.625$.

**Investigate.** Exercises 11 and 12 are open; the first should find that the blend commutes with any parametrization in which the prediction is affine in the guess with coefficients summing to one (guess, velocity, noise for a fixed $\sigma$), and fails where a nonlinear step follows, such as a rescale of the guided prediction.
