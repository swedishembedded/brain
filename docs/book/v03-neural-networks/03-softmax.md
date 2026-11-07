<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Softmax, Logits and Cross-Entropy

*How does a network choose among many classes, and why is its gradient so simple?*

## The problem

A machine part comes off a line with a defect, and the inspection station measures three numbers on it: how rough the surface is, how deep the mark goes and how long it is. The defect is one of four kinds, a scratch, a dent, a crack or a burr, and each kind sends the part to a different repair bench. A classifier has to say which. Its last layer is the linear map of Volume II with one row per class: it takes the three measurements $\mathbf{x}$ and produces four numbers, one logit per class, $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$ (Volume II, chapter 3 used the word for the score whose sigmoid is a probability; here every class has one).

<!-- worked: softmax.defect_inputs -->
Classes: 0 = scratch, 1 = dent, 2 = crack, 3 = burr. Measurements: 0 = roughness, 1 = depth, 2 = length. The part has x = (1.0, 0.5, 0.8) and its true defect is 1 (dent). The classifier is z = W x + b with W of shape [4, 3] and b of shape [4].

| class | W[c,0] | W[c,1] | W[c,2] | b[c] | z[c] = W[c] . x + b[c] |
|---|---:|---:|---:|---:|---:|
| 0 (scratch) | 2.0 | -1.0 | 1.0 | 0.1 | 2.4000 |
| 1 (dent) | -1.0 | 2.0 | 0.5 | 0.0 | 0.4000 |
| 2 (crack) | 0.5 | 1.0 | -1.5 | -0.2 | -0.4000 |
| 3 (burr) | -0.5 | -1.0 | 1.5 | 0.3 | 0.5000 |
<!-- /worked -->

The part has logits $(2.4, 0.4, -0.4, 0.5)$: the classifier leans towards a scratch. Four such numbers are not yet an answer we can use, for three reasons.

- **To train it we need a likelihood.** Volume II, chapter 4 turned a classifier's training into maximum likelihood: the loss of an example is minus the logarithm of the probability the model gave to the label that actually occurred. That needs a probability for each class, positive and adding to one. A logit of $-0.4$ is not one.
- **To act on it we need confidence.** Whether a part goes to the scratch bench or to a second inspection depends on how sure the classifier is, and that is a statement about probabilities, not about the largest of four unconstrained numbers.
- **To learn from it we need a smooth connection.** The decision "the largest logit wins" changes abruptly when two logits cross, so it has no gradient to follow; whatever replaces it must change smoothly as the logits move.

The function that turns the four numbers into four probabilities is the **softmax**. This chapter builds it from the requirement that it must satisfy, shows that for two classes it is the sigmoid you already know, and then solves four problems that appear the moment one writes it in code: it overflows, its logarithm underflows, its loss has a gradient that looks complicated and turns out to be one subtraction, and a loss that demands certainty pushes the logits to infinity. Along the way every number is computed, including the count of operations that a loss over a vocabulary of fifty thousand classes costs per example.

The chapter has eight steps.

1. **From scores to a probability vector.** Why dividing by the sum fails, what exponentiating repairs, and the definition.
2. **The sigmoid is the two-class softmax.** And what a logit means with many classes.
3. **Adding a constant, and scaling.** What the softmax ignores, and what a common factor on the logits does to it.
4. **Computing it safely.** Overflow, the stable softmax, underflow, and the log-softmax.
5. **One-hot targets and the loss.** Cross-entropy as the negative log-likelihood of the categorical.
6. **The Jacobian and the gradient.** The matrix $\mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$, and why the gradient of the loss is $\mathbf{p} - \mathbf{t}$.
7. **Label smoothing, one class or several, and the cost.** Targets that are not certain, the sigmoid per class, and the work per row.
8. **Where this appears in brain.**

You may use from earlier chapters: the logit and the sigmoid (Volume II, chapter 3), the categorical distribution (Volume I, chapter 11), the cross-entropy (Volume I, chapter 13), the log-sum-exp (Volume I, chapter 15) and the Jacobian (Volume I, chapter 8). The weights of the classifier are taken as given; how they are found by repeating the gradient step through many layers is the subject of the next chapters.

## Before you begin

Five questions on ideas from earlier chapters, to answer from memory. The answers are at the end of the chapter. If one is hard, go back to the chapter named before reading on.

1. (Volume II, chapter 3, "Classification and Logistic Regression", the logit and the sigmoid.) A classifier gives a part the logit $z = 1.5$ for the class "rejected". What probability does it assign, and what is the probability of "accepted"?
2. (Volume I, chapter 11, "Distributions", the categorical distribution.) What two conditions must a list of four numbers satisfy to be the probabilities of a categorical distribution on four outcomes? If the third outcome occurred, what is the list of indicators of the outcome?
3. (Volume I, chapter 13, "Information Theory", cross-entropy, three chapters back and in another volume.) What is the cross-entropy of the target $(0, 1, 0)$ against the probabilities $(0.2, 0.5, 0.3)$, and what does it equal when the target is a single certain outcome?
4. (Volume I, chapter 15, "Floating Point and Numerical Error", log-sum-exp.) What is $\mathrm{lse}(0, 0)$, and why does the stable computation of the log-sum-exp subtract the largest score before exponentiating?
5. (Volume I, chapter 8, "Gradients, Jacobians and Many Variables", the Jacobian.) What is the shape of the Jacobian of a function from four inputs to four outputs, and what is its entry in row $i$, column $j$?

## From scores to a probability vector

*What it is for: turning any list of real numbers into a list of probabilities without changing which entry is largest.*

Take the four logits of the part and ask what a conversion into probabilities has to do. Three properties are needed: every output positive (a probability cannot be negative, and a zero would make the loss of an example whose label is that class infinite), the outputs adding to one, and the largest logit staying the largest probability, so that the conversion does not change the classifier's decision.

The first attempt is the obvious one: divide each logit by their sum, which makes the outputs add to one. The second column below does it. The logit $-0.4$ becomes $-0.1379$, a negative probability, and nothing in the method prevents this. Dividing is not the missing step; the numbers have to be made positive first.

The function that makes every real number positive and keeps the order is the exponential, $e^z$. It is positive for every $z$, it increases with $z$, and it turns a difference of logits into a ratio of positive numbers: $e^{z_i}/e^{z_j} = e^{z_i - z_j}$. After exponentiating, dividing by the sum is safe and makes the outputs add to one:

<!-- worked: softmax.scores_to_probabilities -->
| class | z | z / sum(z) | e^z | p = e^z / sum(e^z) |
|---|---:|---:|---:|---:|
| 0 (scratch) | 2.4000 | 0.8276 | 11.0232 | 0.7431 |
| 1 (dent) | 0.4000 | 0.1379 | 1.4918 | 0.1006 |
| 2 (crack) | -0.4000 | -0.1379 | 0.6703 | 0.0452 |
| 3 (burr) | 0.5000 | 0.1724 | 1.6487 | 0.1111 |
| total | 2.9000 | 1.0000 | 14.8340 | 1.0000 |
<!-- /worked -->

That recipe is the **softmax**:

$$
p_i = \mathrm{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=0}^{C-1} e^{z_j}}, \qquad i = 0, \dots, C-1.
$$

Here $C$ is the number of classes. Each $p_i$ is strictly between $0$ and $1$, the $p_i$ add to $1$, and because $e^z$ increases, $p_i > p_j$ exactly when $z_i > z_j$: the class with the largest logit also has the largest probability. For the part, the classifier gives the scratch $0.7431$, the burr $0.1111$, the dent $0.1006$ and the crack $0.0452$. The name says what it is: a soft version of "pick the largest". Picking the largest gives a one-hot answer, all or nothing; the softmax gives every class a share that grows smoothly with its logit. A later section makes the sense in which the first is the limit of the second precise.

Three names will be used throughout. The logits $\mathbf{z}$ are the scores before the softmax; the probability vector $\mathbf{p}$ is the softmax of them, a categorical distribution (Volume I, chapter 11); and the class index $c$ is the number, from $0$ to $C-1$, of the true class of an example.

## The sigmoid is the two-class softmax

*What it is for: knowing that nothing new has been added to Volume II, and what a logit means when there are many classes.*

Take $C = 2$ and write the softmax for class $1$:

$$
p_1 = \frac{e^{z_1}}{e^{z_0} + e^{z_1}} = \frac{1}{1 + e^{z_0 - z_1}} = \sigma(z_1 - z_0),
$$

where the middle step divides the numerator and denominator by $e^{z_1}$. The probability of class $1$ is the sigmoid of the difference of the two logits, and $p_0 = 1 - p_1 = \sigma(z_0 - z_1)$. Volume II's single logit was this difference all along: it measured class $1$ against class $0$. A table over a few pairs shows it, including the pair $(40, 41)$, whose exponentials are huge but whose difference is not:

<!-- worked: softmax.two_classes_are_a_sigmoid -->
| (z0, z1) | z1 - z0 | softmax[0] | softmax[1] | sigmoid(z1 - z0) |
|---|---:|---:|---:|---:|
| (0, 0) | 0 | 0.500000 | 0.500000 | 0.500000 |
| (1, 2) | 1 | 0.268941 | 0.731059 | 0.731059 |
| (3, -1) | -4 | 0.982014 | 0.017986 | 0.017986 |
| (-2, 2.5) | 4.5 | 0.010987 | 0.989013 | 0.989013 |
| (40, 41) | 1 | 0.268941 | 0.731059 | 0.731059 |

Largest absolute difference between the last two columns: 0.
<!-- /worked -->

The reduction suggests the way to read logits when there are more than two classes. Volume II defined the logit of a probability as the log of the odds, $\ln\frac{p}{1-p}$: the log of the ratio of the probability of the class to that of the alternative. With $C$ classes there is no single alternative, but there are ratios of every pair. Suppose we ask that the log-ratio of any two probabilities be the difference of their logits:

$$
\ln\frac{p_i}{p_j} = z_i - z_j \quad\text{for every pair } i, j.
$$

Then $p_i = p_j\,e^{z_i - z_j}$, so $p_i / e^{z_i}$ is the same constant $k$ for every class: $p_i = k\,e^{z_i}$. The probabilities must add to one, which fixes $k = 1/\sum_j e^{z_j}$, and the result is exactly the softmax. So the softmax is not one choice among many: it is the only probability vector whose log-ratios are the logit differences. In that sense a logit is a log-probability known up to a constant shared by all classes. For the part:

<!-- worked: softmax.log_ratio_and_shift -->
Log of the ratio of two probabilities against the difference of their logits:

| pair (i / j) | p_i / p_j | ln(p_i / p_j) | z_i - z_j |
|---|---:|---:|---:|
| scratch / dent | 0.7431 / 0.1006 | 2.0000 | 2.0000 |
| scratch / crack | 0.7431 / 0.0452 | 2.8000 | 2.8000 |
| burr / dent | 0.1111 / 0.1006 | 0.1000 | 0.1000 |

The softmax of z + c for the logits of the part, computed with the stable form:

| shift c | p[scratch] | p[dent] | p[crack] | p[burr] |
|---|---:|---:|---:|---:|
| 0 | 0.743100 | 0.100568 | 0.045188 | 0.111144 |
| 10 | 0.743100 | 0.100568 | 0.045188 | 0.111144 |
| -10 | 0.743100 | 0.100568 | 0.045188 | 0.111144 |
| 1000 | 0.743100 | 0.100568 | 0.045188 | 0.111144 |
<!-- /worked -->

The first table confirms it: the scratch is $e^{2} \approx 7.39$ times as probable as the dent because its logit is $2.0$ higher, and the burr and the dent are almost equally probable because their logits differ by $0.1$. This is also why a classifier with a linear last layer is called a model of log-probabilities: the weights decide the logit differences, and the softmax converts them into a distribution.

## Adding a constant, and scaling

*What it is for: knowing what the softmax cannot see, which is the tool of the next section, and what a common factor on the logits does to it.*

The second table above adds the same number $c$ to every logit of the part, including $c = 1000$, and the probabilities do not move. The reason is one line:

$$
\mathrm{softmax}(\mathbf{z} + c\mathbf{1})_i = \frac{e^{z_i + c}}{\sum_j e^{z_j + c}} = \frac{e^{c}\,e^{z_i}}{e^{c}\sum_j e^{z_j}} = \mathrm{softmax}(\mathbf{z})_i,
$$

because the factor $e^c$ is common to numerator and denominator. This shift invariance has two consequences. It says the logits carry only $C - 1$ numbers of information: only their differences matter, and for $C = 2$ only the single difference $z_1 - z_0$, as the previous section found. And it hands us a freedom to use: whatever $c$ we choose, the result is unchanged, so we may choose the $c$ that makes the arithmetic safe. The next section does that.

A different change is multiplying every logit by the same positive number $a$, which is what changing the sharpness of a model's confidence does. Volume II, chapter 6 used it to repair overconfident probabilities; here it is seen through the softmax:

<!-- worked: softmax.scale_the_logits -->
| scale a | p[scratch] | p[dent] | p[crack] | p[burr] | largest p | entropy (nats) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 1.3863 |
| 0.25 | 0.3670 | 0.2226 | 0.1822 | 0.2282 | 0.3670 | 1.3497 |
| 0.5 | 0.4997 | 0.1838 | 0.1232 | 0.1933 | 0.4997 | 1.2337 |
| 1 | 0.7431 | 0.1006 | 0.0452 | 0.1111 | 0.7431 | 0.8358 |
| 2 | 0.9575 | 0.0175 | 0.0035 | 0.0214 | 0.9575 | 0.2148 |
| 4 | 0.9992 | 0.0003 | 0.0000 | 0.0005 | 0.9992 | 0.0075 |
| 10 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |

The entropy of the uniform distribution on 4 classes is ln 4 = 1.3863.
<!-- /worked -->

Read the table from the middle outwards. At $a = 1$ the logits of the part give the scratch $0.7431$. Doubling every logit moves it to $0.9575$; at $a = 10$ it is $1.0000$ to four decimals, and the entropy of the distribution (Volume I, chapter 13), a measure of how undecided it is, has fallen from $0.8358$ to $0.0000$. Going the other way, $a = 0.25$ shrinks the gaps and the scratch has only $0.3670$; at $a = 0$ all logits are $0$ and the distribution is uniform, with the largest possible entropy $\ln 4 = 1.3863$. In plain words: a larger scale sharpens the distribution towards "the largest logit takes everything", a smaller one flattens it towards "every class is equally likely". The order of the classes never changes, so the decision does not either; only the confidence does. This is the sense in which picking the largest is the limit of the softmax.

The same knob has a second name when a language model chooses its next word: the reciprocal of the scale is what a later volume, on decoding, calls the temperature of sampling. Nothing about it is needed here. What matters is that it is a multiplication of the logits before the softmax, so every formula of this chapter holds for $a\mathbf{z}$ in place of $\mathbf{z}$ (an exercise derives the gradient of the loss under such a scale).

![The four softmax outputs of the part as the logit of the scratch class varies from -6 to 8 while the other three logits stay at 0.4, -0.4 and 0.5. The scratch probability rises from near 0 to near 1 along an S-shaped curve; the other three fall together towards 0 and keep their ratios, with the burr and dent curves close together. A dashed vertical line marks the part's logit 2.4.](img/softmax-vs-logit.svg)

The figure plots the four outputs as one logit moves. The scratch curve is an S, the sigmoid's shape, because against the other three classes taken together it is a two-class problem. The other three curves fall together, and their ratios never change: raising one logit takes probability from the others in proportion to what they have, which is the log-ratio property again.

## Computing it safely

*What it is for: getting the right answer when the logits are large or small, and when a probability is tiny.*

The definition says: exponentiate, add, divide. For logits of a few units that works. Logits are not always a few units. A language model's logits are sometimes in the tens, and in the middle of a training run that has gone wrong they can be in the hundreds. The exponential of a 32-bit number overflows above about $88.7$ and underflows to zero below about $-104$ (Volume I, chapter 15 has the table); in 64 bits the limits are about $709.8$ and $-745$. The table runs the definition in each format, one operation at a time, against the shifted version of the next paragraph and against exact arithmetic:

<!-- worked: softmax.overflow_and_underflow -->
Every operation is rounded to the stated format one at a time (`exp`, `+`, `-`, `/`), an `exp` above the largest finite number gives `inf`, and `inf / inf` and `0 / 0` are `nan`. "naive" is `exp(z[i]) / sum(exp(z))`; "stable" subtracts the largest logit first; "exact" is computed in 60-digit decimal arithmetic.

| logits | format | naive | stable | exact |
|---|---|---|---|---:|
| z = (2.4, 0.4, -0.4, 0.5) | 32-bit | (0.7431, 0.10057, 0.045188, 0.11114) | (0.7431, 0.10057, 0.045188, 0.11114) | (0.7431, 0.10057, 0.045188, 0.11114) |
| z = (100, 98, 90, 0) | 32-bit | (nan, nan, nan, 0) | (0.88076, 0.1192, 3.9987 x 10^-5, 3.3631 x 10^-44) | (0.88076, 0.1192, 3.9987 x 10^-5, 3.2765 x 10^-44) |
| z = (-120, -121, -125, -130) | 32-bit | (nan, nan, nan, nan) | (0.72745, 0.26761, 0.0049015, 3.3026 x 10^-5) | (0.72745, 0.26761, 0.0049015, 3.3026 x 10^-5) |
| z = (1000, 998, 990, 0) | 64-bit | (nan, nan, nan, 0) | (0.88076, 0.1192, 3.9987 x 10^-5, 0) | (0.88076, 0.1192, 3.9987 x 10^-5, 0) |
| z = (-1000, -1001, -1005, -1010) | 64-bit | (nan, nan, nan, nan) | (0.72745, 0.26761, 0.0049015, 3.3026 x 10^-5) | (0.72745, 0.26761, 0.0049015, 3.3026 x 10^-5) |
<!-- /worked -->

The first row is the part: all three columns agree. In the second row, $e^{100}$ is larger than any 32-bit number, so the sum is $\infty$ and every probability is $\infty/\infty$, which is not a number; the same happens at logits of $1000$ in 64 bits. The third and fifth rows fail the other way: every exponential underflows to $0$, the sum is $0$, and each probability is $0/0$. In all four failures the right answer is an ordinary probability vector, which the stable column and the exact column agree on.

The repair is the freedom of the previous section. Choose $c = -m$ with $m = \max_i z_i$. Then every exponent $z_i - m$ is at most $0$, so nothing overflows; the largest term is $e^0 = 1$ exactly, so the sum $s$ lies in $[1, C]$ and cannot be zero; and the result is the same by shift invariance. This is the **stable softmax**:

$$
p_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}, \qquad m = \max_j z_j.
$$

```text
function softmax(z: array<real>[C]) -> array<real>[C]:
    require C > 0, "softmax of nothing is undefined"
    m = max(z)                             # shift so the largest exponent is 0
    p = zeros(real, [C])
    s = 0.0
    for i in 0 .. C:
        p[i] = exp(z[i] - m)               # in (0, 1], and exactly 1 at the maximum
        s = s + p[i]                       # s is in [1, C]
    for i in 0 .. C:
        p[i] = p[i] / s
    return p
```

It reads the row twice (once for the maximum, once for the sum). One more detail of the table: in the second row the last stable entry is $3.3631 \times 10^{-44}$ against the exact $3.2765 \times 10^{-44}$. That value is smaller than the smallest normal 32-bit number, so it is stored with fewer significant digits (a subnormal number, Volume I, chapter 15); the absolute error is about $10^{-45}$ and does not matter for the probabilities, but it explains the third digit.

A tiny probability does matter in one place: its logarithm. The loss of the next section is a log-probability, and a probability that underflows to $0$ has the logarithm $-\infty$ although the true value is an ordinary number. Take the logits $(30, 0, -60, -130)$, whose softmax is dominated by the first class:

<!-- worked: softmax.log_softmax_underflow -->
Logits z = (30, 0, -60, -130), 32-bit arithmetic one operation at a time. The stable log-sum-exp is 30 in 32 bits.

| z[i] | stable softmax p[i] | log(p[i]) | z[i] - lse(z) | exact log p[i] |
|---|---:|---:|---:|---:|
| 30 | 1 | 0 | 0 | -9.35762 x 10^-14 |
| 0 | 9.3576 x 10^-14 | -30 | -30 | -30 |
| -60 | 8.194 x 10^-40 | -90 | -90 | -90 |
| -130 | 0 | -inf | -160 | -160 |
<!-- /worked -->

In 32 bits the stable softmax gives the last class the probability $0$ (the exact value $e^{-160}$ is far below the smallest subnormal number), and the logarithm of that is $-\infty$. The way out is not to compute the probability at all. Take the logarithm of the softmax algebraically:

$$
\ln p_i = \ln\frac{e^{z_i}}{\sum_j e^{z_j}} = z_i - \ln\sum_j e^{z_j} = z_i - \mathrm{lse}(\mathbf{z}),
$$

the **log-softmax**: a logit minus the log-sum-exp of Volume I, chapter 15, computed there as $m + \ln\sum_j e^{z_j - m}$. The column $z_i - \mathrm{lse}(\mathbf{z})$ of the table returns $-160$ for the last class, the exact value. For the first class it returns $0$ where the exact value is $-9.4 \times 10^{-14}$: that is not an error of the method, the 32-bit numbers near $30$ are spaced $1.9 \times 10^{-6}$ apart and $-9 \times 10^{-14}$ is invisible at that scale. The rule to take away is the one the log-sum-exp chapter ended on: never form a probability only to take its logarithm.

## One-hot targets and the loss

*What it is for: a number that says how wrong the probability vector is about the class that occurred.*

The label of an example is a class index $c$. To compare it with a probability vector, write it as a vector of the same length. The **one-hot encoding** of the class $c$ is the vector $\mathbf{t} = \mathbf{e}_c$ with $t_c = 1$ and every other entry $0$ (the indicator list of Volume I, chapter 11): for the dent, $\mathbf{t} = (0, 1, 0, 0)$. It is itself a probability vector, one that puts all its weight on one class.

Volume II, chapter 4 gave the loss of a classifier: the negative logarithm of the probability the model gave to the label that occurred. Here that probability is $p_c$, so the loss of one example is

$$
L = -\ln p_c = -\sum_{i} t_i \ln p_i = H(\mathbf{t}, \mathbf{p}),
$$

the cross-entropy of the model's distribution relative to the one-hot target (Volume I, chapter 13); the sum has a single non-zero term. The entropy of a one-hot vector is zero, so the cross-entropy equals the divergence $D_{\mathrm{KL}}(\mathbf{t}\,\Vert\,\mathbf{p})$ here: the loss is the extra surprise of the model about what happened. Written in the logits with the log-softmax of the previous section,

$$
L = -\bigl(z_c - \mathrm{lse}(\mathbf{z})\bigr) = \mathrm{lse}(\mathbf{z}) - z_c .
$$

The **softmax cross-entropy** is this loss as a function of the logits and the class index: the softmax and the cross-entropy fused into one expression that never forms $p_c$. A table of the loss for each of the four possible true classes at the logits of the part:

<!-- worked: softmax.cross_entropy_by_class -->
Logits z = (2.4000, 0.4000, -0.4000, 0.5000); probabilities p = (0.7431, 0.1006, 0.0452, 0.1111); lse(z) = 2.6969; a uniform guess costs ln 4 = 1.3863.

| true class c | one-hot t | p[c] | -sum t log p | lse(z) - z[c] |
|---|---:|---:|---:|---:|
| 0 (scratch) | (1, 0, 0, 0) | 0.7431 | 0.2969 | 0.2969 |
| 1 (dent) | (0, 1, 0, 0) | 0.1006 | 2.2969 | 2.2969 |
| 2 (crack) | (0, 0, 1, 0) | 0.0452 | 3.0969 | 3.0969 |
| 3 (burr) | (0, 0, 0, 1) | 0.1111 | 2.1969 | 2.1969 |
<!-- /worked -->

If the part really is a scratch, the loss is $0.2969$: the classifier is right and confident. If it is a dent, $2.2969$: the model gave the true class a probability of $0.1006$. The two columns on the right are equal in every row, which is the identity above. A classifier with no information, the uniform guess, costs $\ln 4 = 1.3863$ on every example, a useful reference: a loss above it means the model is worse than not looking at the part. The loss is never negative, and it is zero only if $p_c = 1$.

A training batch has $B$ examples; the loss to minimise is the mean of their individual losses. A batch of three parts, the first being the part above, shows the arithmetic and anticipates the gradient of the next section (the last table divides by $B = 3$ because the loss is a mean):

<!-- worked: softmax.a_batch -->
| row | x | true | z | p | loss |
|---:|---|---|---|---|---:|
| 0 | (1.0, 0.5, 0.8) | dent | (2.4000, 0.4000, -0.4000, 0.5000) | (0.7431, 0.1006, 0.0452, 0.1111) | 2.2969 |
| 1 | (0.2, 1.2, 0.1) | dent | (-0.6000, 2.2500, 0.9500, -0.8500) | (0.0421, 0.7270, 0.1981, 0.0328) | 0.3188 |
| 2 | (0.1, 0.1, 1.5) | burr | (1.7000, 0.8500, -2.3000, 2.4000) | (0.2891, 0.1235, 0.0053, 0.5821) | 0.5411 |

Mean loss over the 3 rows: 1.0523. Gradient of the mean loss with respect to the logits, (p - t) / 3 per row:

| row | dZ[k, scratch] | dZ[k, dent] | dZ[k, crack] | dZ[k, burr] |
|---|---:|---:|---:|---:|
| 0 | 0.2477 | -0.2998 | 0.0151 | 0.0370 |
| 1 | 0.0140 | -0.0910 | 0.0660 | 0.0109 |
| 2 | 0.0964 | 0.0412 | 0.0018 | -0.1393 |
<!-- /worked -->

## The Jacobian and the gradient

*What it is for: the one formula that every classifier's training loop depends on.*

Gradient descent needs $\partial L/\partial z_j$ for every logit. The path from the logits to the loss passes through the softmax, so the chain rule of Volume I (chapter 8) needs the derivative of the softmax: how every probability $p_i$ changes when any logit $z_j$ changes. That is a $C \times C$ matrix, the Jacobian of the softmax, and we derive it first.

Write $s = \sum_j e^{z_j}$, so $p_i = e^{z_i}/s$. Changing $z_j$ changes the numerator of $p_i$ only when $j = i$, and changes the denominator for every $i$, with $\partial s/\partial z_j = e^{z_j}$. The quotient rule gives, for $j = i$,

$$
\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i}s - e^{z_i}e^{z_i}}{s^2} = p_i - p_i^2 = p_i(1 - p_i),
$$

and for $j \neq i$

$$
\frac{\partial p_i}{\partial z_j} = \frac{0\cdot s - e^{z_i}e^{z_j}}{s^2} = -p_i p_j .
$$

Both cases are the single formula $\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$, where $\delta_{ij}$ is $1$ if $i = j$ and $0$ otherwise. As a matrix,

$$
\mathbf{J} = \mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top .
$$

<!-- worked: softmax.softmax_jacobian -->
p = (0.7431, 0.1006, 0.0452, 0.1111).

| J[i, j] | j=0 | j=1 | j=2 | j=3 |
|---|---:|---:|---:|---:|
| 0 (scratch) | 0.1909 | -0.0747 | -0.0336 | -0.0826 |
| 1 (dent) | -0.0747 | 0.0905 | -0.0045 | -0.0112 |
| 2 (crack) | -0.0336 | -0.0045 | 0.0431 | -0.0050 |
| 3 (burr) | -0.0826 | -0.0112 | -0.0050 | 0.0988 |

Largest absolute row sum: below 1e-15. Largest asymmetry |J[i, j] - J[j, i]|: below 1e-15. Largest difference from central differences with step 1e-06: below 1e-09.
<!-- /worked -->

![The Jacobian of the softmax at the part, as a four by four heat map with each entry printed. The diagonal cells are orange and positive (0.1909, 0.0905, 0.0431, 0.0988); every off-diagonal cell is blue and negative, the largest in magnitude between scratch and dent (-0.0747) and scratch and burr (-0.0826), and the matrix is symmetric.](img/softmax-jacobian.svg)

Four facts can be read off the matrix and the heat map, and each has a reason.

- The diagonal is positive and the off-diagonal negative: raising a class's logit raises its own probability and lowers every other class's.
- The matrix is symmetric, since $p_i p_j = p_j p_i$ (the table prints the largest asymmetry, which is rounding noise).
- Every row, and so every column, sums to zero. That is shift invariance seen through derivatives: adding $c$ to every logit changes nothing, so the derivative along the direction $\mathbf{1}$ is zero, $\mathbf{J}\mathbf{1} = \mathbf{p} - \mathbf{p}(\mathbf{p}^\top\mathbf{1}) = \mathbf{0}$ because $\sum_j p_j = 1$.
- For $C = 2$ the entry $\partial p_1/\partial z_1 = p_1(1 - p_1)$ is the slope of the sigmoid of Volume II.

Now the loss. For a true class $c$, $L = -\ln p_c$, so $\partial L/\partial p_i = -t_i/p_i$: zero for every class but the true one, where it is $-1/p_c$. Call this vector $\mathbf{u}$. The chain rule says $\partial L/\partial z_j = \sum_i u_i\, \partial p_i/\partial z_j$, which is the row vector $\mathbf{u}$ multiplied by the Jacobian (the next chapter calls such a product a vector-Jacobian product). Using the formula for $\mathbf{J}$,

$$
\frac{\partial L}{\partial z_j} = \sum_i \left(-\frac{t_i}{p_i}\right) p_i(\delta_{ij} - p_j) = -\sum_i t_i\,\delta_{ij} + p_j\sum_i t_i = -t_j + p_j,
$$

because $\sum_i t_i = 1$. The division by $p_i$ cancels the factor $p_i$ of the Jacobian, and the whole matrix collapses to

$$
\boxed{\;\frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{t}\;}
$$

The gradient of the softmax cross-entropy with respect to the logits is the probability vector minus the one-hot target. The table shows it three ways for the part, whose true class is the dent: through the Jacobian, by the closed form and by a central difference of the loss itself:

<!-- worked: softmax.gradient_through_the_jacobian -->
True class 1 (dent), p = (0.7431, 0.1006, 0.0452, 0.1111). The loss is -ln p[1]; its derivative with respect to p is u = -t / p.

| class | p | t | u = -t / p | u J | p - t | central difference |
|---|---:|---:|---:|---:|---:|---:|
| 0 (scratch) | 0.7431 | 0 | 0.0000 | 0.7431 | 0.7431 | 0.7431 |
| 1 (dent) | 0.1006 | 1 | -9.9436 | -0.8994 | -0.8994 | -0.8994 |
| 2 (crack) | 0.0452 | 0 | 0.0000 | 0.0452 | 0.0452 | 0.0452 |
| 3 (burr) | 0.1111 | 0 | 0.0000 | 0.1111 | 0.1111 | 0.1111 |

An arbitrary upstream vector u = (1.0, -2.0, 0.5, 3.0) sent through the same Jacobian (u . p = 0.8980):

| class | upstream u | u J (matrix product) | p * (u - u . p) |
|---|---:|---:|---:|
| 0 (scratch) | 1 | 0.0758 | 0.0758 |
| 1 (dent) | -2 | -0.2914 | -0.2914 |
| 2 (crack) | 0.5 | -0.0180 | -0.0180 |
| 3 (burr) | 3 | 0.2336 | 0.2336 |
<!-- /worked -->

The columns of the first table agree. Look at the column $\mathbf{u}$: it holds $-9.9436$ for the dent and zeros elsewhere, a large number, and the Jacobian times it gives entries between $-0.9$ and $0.74$. The factor $1/p_c$ and the factor $p_c$ in the Jacobian cancelled exactly, which is why the final gradient never exceeds $1$ in magnitude.

**A shortcut.** The result can also be reached without the Jacobian, from the fused form $L = \mathrm{lse}(\mathbf{z}) - z_c$. The derivative of the log-sum-exp is $\partial\,\mathrm{lse}/\partial z_j = e^{z_j}/\sum_k e^{z_k} = p_j$, and the derivative of $-z_c$ is $-t_j$, so $\partial L/\partial z_j = p_j - t_j$ directly. This is the real reason for the fused form: the softmax and the logarithm are not two operations to be differentiated and multiplied, but one function whose derivative is short.

**The general rule for a softmax in the middle of a network.** The loss is not always the next operation. Attention (Volume V) applies a softmax and then does more arithmetic with it, so the gradient $\mathbf{u}$ arriving at $\mathbf{p}$ can be any vector. The product $\mathbf{u}\mathbf{J}$ is then, using $\mathbf{J} = \mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$,

$$
\left(\mathbf{u}\mathbf{J}\right)_j = p_j u_j - p_j \sum_i p_i u_i = p_j\bigl(u_j - \mathbf{u}\cdot\mathbf{p}\bigr),
$$

one dot product and one elementwise product: about $3C$ operations. The second table above sends an arbitrary $\mathbf{u}$ through the matrix and through this formula and gets the same four numbers. The matrix itself has $C^2$ entries, and for the vocabulary of fifty thousand words used below that is $2.5 \times 10^9$ numbers per example, which nobody would build. The next chapter makes it a principle that a backward pass multiplies a row vector by a Jacobian and never forms the matrix; for the softmax, this one line is that product.

**Why the fused form is simpler and safer.** Compose the two operations in code: compute $\mathbf{p}$, take its logarithm, and differentiate through both. The backward pass divides by $p_c$ and then multiplies by $p_c(1 - p_c)$ and $p_c p_j$; the two factors cancel in exact arithmetic, but a computer first evaluates the quotient. Take a confidently wrong example, logits $(30, -100, 5, 0)$ whose true class is the second:

<!-- worked: softmax.fused_against_composed -->
Logits z = (30, -100, 5, 0), true class 1 (dent), 32-bit arithmetic one operation at a time for the first two columns. Stable softmax p = (1, 0, 1.3888 x 10^-11, 9.3576 x 10^-14). The last column is a squared error on the probabilities, sum (p - t)^2, with its gradient 2 (p - t) J, in 64-bit arithmetic.

|  | composed: -log(p[c]), then back through log and softmax | fused: lse - z[c], gradient p - t | squared error on p |
|---|---:|---:|---:|
| loss | inf | 130 | 2 |
| gradient[0] (scratch) | nan | 1 | 2.8 x 10^-11 |
| gradient[1] (dent) | nan | -1 | -1.39 x 10^-56 |
| gradient[2] (crack) | nan | 1.38879 x 10^-11 | -2.78 x 10^-11 |
| gradient[3] (burr) | nan | 9.35762 x 10^-14 | -1.87 x 10^-13 |
<!-- /worked -->

The composed route underflows: $p_1$ is $e^{-130}$, below the smallest 32-bit number, so the loss is $-\ln 0 = \infty$; the backward pass starts from $u_1 = -1/0 = -\infty$ and multiplies it by a Jacobian entry that is $0$, and $\infty \cdot 0$ is not a number, which then spreads to every logit. The fused route gives the loss $130$, the exact value to the precision of the format, and the gradient $(1, -1, 1.4 \times 10^{-11}, 9.4 \times 10^{-14})$: the model is told exactly what to do, push the first class down and the second up, at full strength. Fusing buys three things at once: fewer operations (there is no division, no Jacobian), a gradient bounded by $1$ in magnitude, and a computation in which nothing that cancels is ever evaluated.

The last column of the table is a warning of what the alternative loss does. A squared error on the probabilities, $\sum_i (p_i - t_i)^2$, has the gradient $2(\mathbf{p} - \mathbf{t})\mathbf{J}$, whose factor $\mathbf{J}$ is almost zero when the softmax is saturated. For this example the squared error is $2$, as wrong as it can be, and the gradient is $3 \times 10^{-11}$ or smaller (the entry of the true class itself is $10^{-56}$): the model cannot learn from its worst mistake. This is the many-class version of the result of Volume II, chapter 3, where the squared error trained poorly on a sigmoid for the same reason; the cross-entropy's logarithm exactly undoes the saturation.

**What the gradient says.** In words, $\mathbf{p} - \mathbf{t}$ says: for every wrong class, push its logit down by its own probability; for the true class, push its logit up by the probability that it is missing, $1 - p_c$, which equals the sum of the wrong classes' probabilities. For the part, the gradient is $(0.7431, -0.8994, 0.0452, 0.1111)$: the scratch, which the model believed in, is pushed down hard, the dent pulled up, and the unlikely crack barely touched. Three properties follow. The entries add to zero, so a step changes no logit's average, in agreement with shift invariance. The gradient is large when the model is wrong and small when it is right, and it vanishes only when $\mathbf{p} = \mathbf{t}$, which for a one-hot target needs $p_c = 1$, a logit gap that is infinite. And its size is bounded: no logit can be told to move by more than $1$ per example, however confidently wrong the model is.

The loss and the gradient together, from the logits of one example and its class index, in the book's pseudocode:

```text
function cross_entropy_row(z: array<real>[C], c: int) -> (real, array<real>[C]):
    require C > 0 and 0 <= c < C, "class index out of range"
    m = max(z)
    s = 0.0
    for i in 0 .. C:
        s = s + exp(z[i] - m)
    loss = (m + log(s)) - z[c]             # lse(z) - z[c]
    dz = zeros(real, [C])
    for i in 0 .. C:
        dz[i] = exp(z[i] - m) / s          # the probability p[i]
    dz[c] = dz[c] - 1.0                    # minus the one-hot target
    return (loss, dz)
```

For a batch of $B$ rows each gradient is divided by $B$ (the mean). The last table of the previous section did exactly that.

**Training the classifier.** The gradient with respect to the logits is the start of the backward pass. For the last layer $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$ a logit is $z_i = \sum_j W_{ij}x_j + b_i$, so $\partial L/\partial W_{ij} = (p_i - t_i)\,x_j$ and $\partial L/\partial b_i = p_i - t_i$: with $d\mathbf{W}$ the matrix of all $\partial L/\partial W_{ij}$, $d\mathbf{W} = (\mathbf{p} - \mathbf{t})\mathbf{x}^\top$ and $d\mathbf{b} = \mathbf{p} - \mathbf{t}$, an outer product of the error and the input. The table computes both for the part and then follows plain gradient descent on the single example:

<!-- worked: softmax.gradient_descent_on_weights -->
Gradient at the starting weights for the part x = (1.0, 0.5, 0.8) and true class 1 (dent):

| dW[i, j] = (p - t)[i] x[j] | j=0 | j=1 | j=2 | db[i] = (p - t)[i] |
|---|---:|---:|---:|---:|
| 0 (scratch) | 0.7431 | 0.3715 | 0.5945 | 0.7431 |
| 1 (dent) | -0.8994 | -0.4497 | -0.7195 | -0.8994 |
| 2 (crack) | 0.0452 | 0.0226 | 0.0362 | 0.0452 |
| 3 (burr) | 0.1111 | 0.0556 | 0.0889 | 0.1111 |

Plain gradient descent, w <- w - eta dW and b <- b - eta db with eta = 0.5, from the weights of the first table of the chapter:

| step k | p | loss | predicted |
|---:|---|---:|---|
| 0 | (0.7431, 0.1006, 0.0452, 0.1111) | 2.2969 | scratch |
| 1 | (0.3342, 0.4855, 0.0557, 0.1246) | 0.7226 | dent |
| 2 | (0.1491, 0.7385, 0.0372, 0.0753) | 0.3032 | dent |
| 3 | (0.0924, 0.8286, 0.0271, 0.0519) | 0.1880 | dent |
| 4 | (0.0665, 0.8725, 0.0214, 0.0396) | 0.1364 | dent |
| 5 | (0.0517, 0.8985, 0.0178, 0.0320) | 0.1071 | dent |
| 6 | (0.0422, 0.9156, 0.0153, 0.0269) | 0.0882 | dent |
<!-- /worked -->

The first step already changes the answer: the dent overtakes the scratch ($0.4855$ against $0.3342$) and the loss drops from $2.2969$ to $0.7226$. After that the loss keeps falling, but more slowly, because the gradient $\mathbf{p} - \mathbf{t}$ shrinks as $p_c$ approaches $1$: after six steps the dent has $0.9156$ and the loss is $0.0882$. It never reaches zero, and the next section is about what to do with that.

## Label smoothing

*What it is for: keeping a classifier from being driven towards a certainty that no finite logit can express.*

The one-hot target asks for $p_c = 1$ and $p_i = 0$ for every other class. The softmax can only reach that in the limit: $p_c \to 1$ needs the gap between the true logit and the others to grow without bound. The loss $\ln(1 + \sum_{i\neq c} e^{z_i - z_c})$ falls towards zero as the gap grows, but never reaches it, and every step of gradient descent makes the gap larger. (This is the same behaviour as logistic regression on separable data in Volume II, chapter 3: the weights grew like $\ln k$ without limit.) The consequences are the ones seen there: large logits, probabilities pinned at the extremes whatever the evidence, and a model that keeps growing its weights on examples it already gets right.

The remedy is to ask for a less extreme target. **Label smoothing** replaces the one-hot target by a mixture of it with the uniform distribution:

$$
\mathbf{t}' = (1 - \epsilon_{\mathrm{ls}})\,\mathbf{t} + \frac{\epsilon_{\mathrm{ls}}}{C}\,\mathbf{1},
$$

where $\epsilon_{\mathrm{ls}} \in [0, 1)$ is the smoothing weight, a small number such as $0.1$. The true class has the target $1 - \epsilon_{\mathrm{ls}} + \epsilon_{\mathrm{ls}}/C$, every other class $\epsilon_{\mathrm{ls}}/C$, and the entries still add to one. With $C = 4$ and $\epsilon_{\mathrm{ls}} = 0.1$:

<!-- worked: softmax.label_smoothing_target -->
| class | t | t' = (1 - eps) t + eps / C | p | gradient p - t | gradient p - t' |
|---|---:|---:|---:|---:|---:|
| 0 (scratch) | 0 | 0.0250 | 0.7431 | 0.7431 | 0.7181 |
| 1 (dent) | 1 | 0.9250 | 0.1006 | -0.8994 | -0.8244 |
| 2 (crack) | 0 | 0.0250 | 0.0452 | 0.0452 | 0.0202 |
| 3 (burr) | 0 | 0.0250 | 0.1111 | 0.1111 | 0.0861 |

With eps = 0.1: loss against the one-hot target 2.2969, against the smoothed target 2.2644; the definition -sum t' ln p gives 2.2644. The smoothed target sums to 1.0000.
<!-- /worked -->

Nothing else in the machinery changes. The loss is the cross-entropy against $\mathbf{t}'$, $L = -\sum_i t'_i \ln p_i = \mathrm{lse}(\mathbf{z}) - \sum_i t'_i z_i$ (using $\sum_i t'_i = 1$), and the derivation of the gradient never used that $\mathbf{t}$ was one-hot, only that its entries add to one, so the gradient is $\mathbf{p} - \mathbf{t}'$. The last two columns of the table compare the gradients: for the true class the pull is $-0.8244$ instead of $-0.8994$, and each wrong class is pushed down less because the target now asks for a small positive probability there.

The loss is also a mixture: $L = (1 - \epsilon_{\mathrm{ls}})\,(-\ln p_c) + \frac{\epsilon_{\mathrm{ls}}}{C}\sum_i(-\ln p_i)$, the usual loss plus a penalty $-\ln p_i$ on a class having a tiny probability, which is what keeps the logits from running away.

The effect is on where the loss is smallest. The gradient vanishes when $\mathbf{p} = \mathbf{t}'$, and by the log-ratio property of the second section that means the logit gap between the true class and any other is

$$
z_c - z_j = \ln\frac{1 - \epsilon_{\mathrm{ls}} + \epsilon_{\mathrm{ls}}/C}{\epsilon_{\mathrm{ls}}/C} = \ln\!\Bigl(1 + \frac{C(1 - \epsilon_{\mathrm{ls}})}{\epsilon_{\mathrm{ls}}}\Bigr),
$$

a finite number. With $C = 4$ and $\epsilon_{\mathrm{ls}} = 0.1$ it is $\ln 37 = 3.6109$. The table gives the best gap and the smallest loss for several smoothing weights, then the loss as the true logit grows with the other three held at zero:

<!-- worked: softmax.label_smoothing_optimum -->
4 classes, true class 0, logits z = (g, 0, 0, 0):

| eps | t'[true] | t'[other] | best gap z_true - z_other | smallest loss = entropy of t' |
|---|---:|---:|---:|---:|
| 0 | 1.0000 | 0.0000 | none (the gap grows without bound) | 0.0000 |
| 0.05 | 0.9625 | 0.0125 | 4.3438 | 0.2011 |
| 0.1 | 0.9250 | 0.0250 | 3.6109 | 0.3488 |
| 0.2 | 0.8500 | 0.0500 | 2.8332 | 0.5875 |

The loss as the true logit grows (the other three logits stay at 0):

| gap g | loss, eps = 0 | loss, eps = 0.1 | loss, eps = 0.2 |
|---|---:|---:|---:|
| 0 | 1.3863 | 1.3863 | 1.3863 |
| 2 | 0.3408 | 0.4908 | 0.6408 |
| 3 | 0.1392 | 0.3642 | 0.5892 |
| 3.6109 | 0.0780 | 0.3488 | 0.6196 |
| 4 | 0.0535 | 0.3535 | 0.6535 |
| 6 | 0.0074 | 0.4574 | 0.9074 |
| 8 | 0.0010 | 0.6010 | 1.2010 |
| 12 | 0.0000 | 0.9000 | 1.8000 |
<!-- /worked -->

![Cross-entropy against the logit of the true class for four classes, the other three logits being 0. The unsmoothed loss falls towards 0 and never rises. With smoothing weight 0.1 and 0.2 the loss falls to a minimum, marked by a dot at 3.61 and 2.83 respectively, with loss 0.35 and 0.59, and then rises linearly as the logit keeps growing.](img/label-smoothing-loss.svg)

The unsmoothed loss keeps falling: at a gap of $12$ it is $0.0000$ to four decimals, and the optimum is nowhere. The smoothed loss has a minimum, at $3.6109$ for $\epsilon_{\mathrm{ls}} = 0.1$ (and at $2.8332$ for $0.2$), and beyond it the loss rises: the model is now penalised for being too sure. The rise is linear in the gap, with slope $\epsilon_{\mathrm{ls}}(C-1)/C$ (the weight of the wrong classes in $\mathbf{t}'$), so the pressure is mild but it never stops. The price is a floor: the loss can no longer reach zero, since the smallest value is the entropy of $\mathbf{t}'$, $0.3488$ at $\epsilon_{\mathrm{ls}} = 0.1$. A reported loss is therefore comparable only between runs with the same smoothing weight, and it is no longer the negative log-likelihood of the labels; to report that likelihood on validation data, evaluate the ordinary cross-entropy ($\epsilon_{\mathrm{ls}} = 0$) there.

Label smoothing is a design choice, not a law: it limits confidence by construction, which can help a model that is overconfident on noisy labels, and it costs the information that some wrong classes are more plausible than others (all wrong classes get the same share). The tools of Volume II, chapter 6 (calibration, the expected calibration error) measure whether it helped.

## One class or several

*What it is for: choosing between two output layers that look alike and answer different questions.*

The softmax assumes that exactly one class is true: the probabilities compete, and raising one lowers the others. A part with one defect is that case. A photograph may contain a dog and a ball; a part may have a scratch and a dent. Then each class is its own yes-or-no question and the model gives each its own sigmoid and its own binary cross-entropy (Volume II, chapter 3), with no constraint that they add to one. The same four logits through both heads:

<!-- worked: softmax.multi_label_against_multi_class -->
| class | z | softmax (one class) | sigmoid per class (any subset) |
|---|---:|---:|---:|
| 0 (scratch) | 2.4000 | 0.7431 | 0.9168 |
| 1 (dent) | 0.4000 | 0.1006 | 0.5987 |
| 2 (crack) | -0.4000 | 0.0452 | 0.4013 |
| 3 (burr) | 0.5000 | 0.1111 | 0.6225 |
| sum |  | 1.0000 | 2.5393 |
<!-- /worked -->

The softmax column adds to $1$: it has to say how its one unit of belief is shared. The sigmoid column adds to $2.5393$: it says that a scratch is very likely ($0.9168$) and a burr more likely than not ($0.6225$), both at once. A softmax trained on a part labelled "scratch and dent" cannot represent it: the best target it can be given is half on each ($0.5, 0.5, 0, 0$), which says "one of the two", not "both". The rule is to match the head to the world: one label from a list, softmax and cross-entropy; any subset of labels, a sigmoid and a binary cross-entropy per class. A consequence of the softmax's competition is worth knowing: it sees only the differences of the scores (shift invariance), so options that produce nearly identical scores give nearly no signal, however different the quantity underneath is; knowledge #133, described below, is a model that failed this way.

## What it costs

*What it is for: seeing that the loss over a large vocabulary is cheap in arithmetic and expensive in data, and counting both.*

A language model's last layer has one logit per word of its vocabulary, so $C$ is not $4$ but $50\,000$ or more. Count the operations of one row, counting each compare, add, subtract, divide, exponential and logarithm as one operation and listing the exponentials separately, since an exponential costs several multiply-adds in hardware. The softmax needs $C - 1$ compares for the maximum, then a subtraction, an exponential, an addition and a division per class: $5C - 1$ operations, $C$ of them exponentials. The fused loss value $\mathrm{lse}(\mathbf{z}) - z_c$ needs no division: $4C + 2$ operations, $C$ exponentials. The gradient has a choice of two designs. One thread per logit, each recomputing the row's maximum and sum before its own entry, costs $C(4C + 4)$ operations per row, with $C(C + 1)$ exponentials: quadratic in $C$. Computing the two row statistics, the maximum and the sum, once per row and letting each entry read them costs $(4C - 1) + 5C = 9C - 1$ operations, $2C$ exponentials: linear. The table runs each algorithm with a counter to confirm the formulas at three small sizes, then evaluates them at the large ones:

<!-- worked: softmax.work_counts -->
Operations per row, counted by running each algorithm with a counter (each compare, add, subtract, divide, `exp` and `log` is one operation); the number of `exp` calls is in brackets.

| C | softmax | loss value | gradient, two passes | gradient, one thread per element | formulas agree |
|---|---:|---:|---:|---:|---:|
| 4 | 19 (4) | 18 (4) | 35 (8) | 80 (20) | yes |
| 16 | 79 (16) | 66 (16) | 143 (32) | 1088 (272) | yes |
| 64 | 319 (64) | 258 (64) | 575 (128) | 16640 (4160) | yes |

The formulas (5C - 1, 4C + 2, 9C - 1 and 4C^2 + 4C operations; C, C, 2C and C(C + 1) exps), evaluated for larger C:

| C | softmax | loss value | gradient, two passes | gradient, one thread per element | ratio naive / two passes |
|---|---:|---:|---:|---:|---:|
| 4 | 19 | 18 | 35 | 80 | 2 |
| 50,000 | 249,999 | 200,002 | 449,999 | 10,000,200,000 | 22,223 |
| 151,936 | 759,679 | 607,746 | 1,367,423 | 92,338,800,128 | 67,528 |
<!-- /worked -->

For $C = 4$ the two gradient designs differ by a factor of $2$ and nobody would care. At $C = 50\,000$ the quadratic design needs $1.0 \times 10^{10}$ operations per row against $4.5 \times 10^{5}$, a ratio of $22\,223$; this is the difference between a design that is natural for a classifier with a few classes and one that cannot be used for a vocabulary. The count is the work; the second cost is data. The logits of a batch are a matrix of $n \times C$ numbers that the final linear layer writes and the loss reads:

<!-- worked: softmax.logits_matrix_size -->
f32 logits (4 bytes each), a batch of 4,096 rows, and a final linear layer with D = 1024 inputs (2 D C operations per row to make the logits).

| C | logit bytes per row | bytes for 4,096 rows | operations of the layer, per row | operations of the loss value, per row | loss / layer |
|---|---:|---:|---:|---:|---:|
| 4 | 16 | 65,536 | 8,192 | 18 | 0.0022 |
| 50,000 | 200,000 | 819,200,000 | 102,400,000 | 200,002 | 0.0020 |
| 248,320 | 993,280 | 4,068,474,880 | 508,559,360 | 993,282 | 0.0020 |
<!-- /worked -->

Two readings. The arithmetic of the loss is about $0.2\%$ of the arithmetic of the layer that makes the logits ($4C + 2$ against $2DC$ operations with $D = 1024$): the loss is not where the operations are. But the logits of $4096$ rows at $C = 50\,000$ are $819\,200\,000$ bytes, and the loss reads each one at least twice (the maximum pass and the sum pass), about $4C + 2$ operations for $8C$ bytes, a half operation per byte. A computation that does so little arithmetic per byte moved is limited by memory, not by arithmetic, so what shortens it is moving fewer bytes: reading each logit fewer times, or not materialising all the logits at all (the next section shows brain doing that). A variant of the stable softmax that reads the row once, with a running maximum and a rescaled running sum, is the subject of the flash-attention chapter of Volume V.

## Where this appears in brain

The softmax cross-entropy is the loss of every language model in brain, the loss of the classifier heads, and, through the same formula, the softmax inside attention. Everything above is visible in a handful of kernels.

**The fused loss and its gradient.** `crates/kernels/wgsl/ce_value.wgsl` computes, for each row of logits, `(mx + log(sum)) - logits[base + targets[n]]`: the maximum, then the sum of `exp(logit - mx)`, then the stable $\mathrm{lse}(\mathbf{z}) - z_c$ of this chapter, one thread per row, with the host summing the per-row values and dividing by the row count. Its sibling `crates/kernels/wgsl/ce_grad.wgsl` writes `(prob - onehot) / f32(p.n_rows)` for every row and class, with `prob` the stable softmax and `onehot` computed as `select(0.0, 1.0, v == targets[n])`: the form $(\mathbf{p} - \mathbf{t})/n$ of the previous section, verified by reading the kernel. Note the last detail: the one-hot vector is never stored. The targets are an array of class indices, and the comparison `v == targets[n]` is the one-hot encoding, evaluated where it is used. The masked siblings `ce_value_masked.wgsl` and `ce_grad_masked.wgsl` add an ignore index: an ignored row gets the gradient $0$, and the divisor is the number of non-ignored rows, passed in as a float, so that padding positions neither contribute nor dilute the mean.

**The story of the quadratic gradient.** `ce_grad.wgsl` has the quadratic design of the last section: one invocation per (row, class), each recomputing the row softmax, and its header says "vocab is small". The two kernels were part of the first import of shaders into the repository (commit `1fa53524c`, 23 June 2026), when the vocabularies were small. The design stopped being acceptable at the vocabulary of a modern language model, $151\,936$ in the header of the later kernel: commit `68db48597` (23 July 2026) added `ce_stats.wgsl`, one invocation per row, which writes the row maximum and the sum of exponentials, and `ce_grad_stats.wgsl`, which reads them and computes $(\mathrm{exp}(z_v - m)/s - t_v)/\text{count}$ in constant work per entry. The header of `ce_stats.wgsl` says the replacement turns $O(\text{rows}\cdot\text{vocab}^2)$ into $O(\text{rows}\cdot\text{vocab})$. The counts of the last section agree with it: for $C = 151\,936$ they give $9.2 \times 10^{10}$ operations per row against $1.4 \times 10^{6}$, a ratio of $67\,528$, which is between four and five orders of magnitude (the header says four). The commit message reports the end result of that and other changes on its own setup:

<!-- perf-number: historical measurement, commit 68db48597 -->
In July 2026, for a full fine-tune of a 0.6-billion-parameter model on two GPUs of one type, the commit message reports a step falling from 89 seconds to 7 after the two-pass gradient, and says profiling had shown the matrix products to be under 1% of the step. That is one measurement of one setup, recorded by the author of the change; the work count above is the part that does not depend on the machine.

The old kernel was not removed. A comment in `crates/arcface/src/model.rs` explains that `ce_grad` stays because its contract (unconditional division by the row count, no ignore mask) is the simplest correct kernel for a caller with few classes, and the face-recognition model there registers both: the right design depends on $C$.

**Not materialising the logits.** `crates/model/src/head_ce.rs` states the second cost of the last section in its header: the logits of a language-model head take `n * V` floats, "993 KB per token at `V = 248320`", and $4 \times 248\,320 = 993\,280$ bytes is the table's number. The module keeps only a chunk of rows of logits alive: the forward computes a chunk's logits and per-token loss and moves on, and the backward recomputes each chunk's logits from the hidden states (the same matrix product), runs the two-pass gradient and applies the head. It was introduced in commit `eaec36bd0` (16 October 2026), whose message says the `qwen3` model now runs its head through it instead of its own copy of the chunking, windowing and gradient logic, so that every token-classifier model shares one. Its conformance check in `crates/model/src/conformance_head.rs` runs the same batch through a head that holds all the rows and a head that chunks, with every fifth target ignored so that the mask crosses chunk boundaries, and requires the losses, the log-probabilities and every gradient to agree to a tolerance: a chunked softmax must be the same function as the unchunked one.

**Log-softmax on the host.** `crates/model/src/logprobs.rs` holds `row_logprob`, which is `row[idx] - max - sum_exp.ln()`, the log-softmax of this chapter at one index, with the sum taken over `exp(v - max)`; `row_top_logprobs` computes the log-sum-exp once and subtracts it from the $k$ largest logits. Its three tests check the formula against the definition and that a log-probability is never positive. `token_logprobs_scaled` multiplies the logits by a scale before the softmax: the scale $a$ of the third section, set to the reciprocal of a sampling temperature when a trainer must score a token under the distribution a sampler drew it from. Knowledge #270 records how the scale is applied to the logits buffer and why the gradient of the loss under the scale is $a(\mathrm{softmax}(a\mathbf{z}) - \mathbf{t})/\text{count}$ (the chain rule on $a\mathbf{z}$; an exercise derives it), and knowledge #290 records the mistake that made it necessary: a reinforcement-learning objective stored the sampler's log-probabilities under the tempered softmax but scored the training forward under the untempered one, so the first update's probability ratio was not $1$. The objective in `crates/rl/src/objective/grpo.rs` now calls `set_logit_scale` with its scale. What the code confirms is that call, the scale in `logprobs.rs` and the chain rule; the size of that gap and the rest of the entry's measurements are its own claims on its own setup.

**The softmax kernels, and the same gradient formula.** Every softmax kernel subtracts the row maximum: `softmax_rows.wgsl` (one 64-thread workgroup per row, for long attention rows), `attn_softmax.wgsl` and its masked, cross and bidirectional variants, `decode_softmax.wgsl` (one thread per attention head in a decode step) and `softmax_k.wgsl` (a softmax over a strided axis of length $K = 9$ for the upsampling mask of a depth-estimation network). The backward of the last, `crates/kernels/wgsl/softmax_k_dx.wgsl`, is the general rule of the sixth section written as two loops: a loop computing `dot = sum dy[k] * y[k]`, and `dx[k] = y[k] * (dy[k] - dot)`. Its header says it takes the forward output $\mathbf{y}$, "the Jacobian is expressed purely in the probabilities", so it need not recompute the exponentials, and that recomputing them would add "a second place for the two to disagree". Commit `81908ab4f` (16 July 2026) added the pair among the kernels for that network, and its message says the forward softmax over a strided axis was a different reduction from the one over a contiguous map that brain already had.

**Common mode: knowledge #133.** The entry records a decision model that scored a yes-or-no proposition by giving two options (the texts "the deal closes" and "the deal is lost") each a score and taking a softmax across the two. The two option representations were nearly identical (cosine $0.989$), so the two scores carried almost the same state information, and the softmax saw only their difference: the entry's probe shows the raw scores spreading by $0.2895$ across three conversations and their difference by $0.0066$, and the model evaluated at chance. The shift invariance of the third section is exactly this: the softmax discards everything the scores share. The fix, which the code documents at `PROPOSITION` in `crates/decide/src/policy.rs`, scores one slot and reads a sigmoid off it, which for two classes is the same softmax with one of the logits fixed at zero, so that the shared part cannot be subtracted away. The lesson generalises to any head over near-identical options: look at the raw scores, not at the probability, which is the quantity that has already done the subtracting.

**Label smoothing is not in brain.** A search of `crates/`, `docs/` and `.agents/` for "label smoothing" in any spelling (spaces, underscores, hyphens) finds nothing outside this book's curriculum file. The cross-entropy kernels cannot express it either: their targets are class indices, from which the one-hot vector is derived, so a smoothed target would need a kernel that takes a target distribution. The one place brain trains against a distribution instead of an index is distillation (`crates/rl/src/objective/distill.rs`, discussed in Volume I, chapter 13), which uses a teacher's probabilities as the target; by the derivation above its logit gradient has the same form $\mathbf{p} - \mathbf{t}$ with $\mathbf{t}$ the teacher.

## Check your understanding

Answers are at the end of the chapter.

1. **Explain.** The softmax of $(1, 2, 3)$ equals the softmax of $(11, 12, 13)$. Why? How many independent numbers does the output of a softmax over $C$ classes contain, and how many for $C = 2$?
2. **Predict.** The logits are $(1000, 1000)$. What does the textbook formula $e^{z_i}/\sum_j e^{z_j}$ give in 64-bit arithmetic, what does the stable softmax give, and what is the loss for the true class $0$?
3. **Find the error.** "To get the loss in code, compute `p = softmax(z)` and then `loss = -log(p[c])`. It is the same formula, so it is as accurate." Give logits for which this breaks in 32 bits and the formula that does not.
4. **Find the error.** "For a one-hot target the gradient $\mathbf{p} - \mathbf{t}$ is zero once the model classifies the example correctly." Correct it, using $p_c = 0.99$ for the true class and the rest of the probability spread equally over three other classes.
5. **Predict.** With $C = 5$ classes and $\epsilon_{\mathrm{ls}} = 0.15$, give the smoothed target and the logit gap at which the loss is smallest.
6. **Explain.** The entries of $\mathbf{p} - \mathbf{t}$ add to zero. What property of the softmax does that reflect, and what does it say about a gradient step on the logits?
7. **Find the error.** "A photograph can contain a dog and a ball, so we label it $(1, 1, 0)$ and train a softmax cross-entropy on it." What goes wrong, and what is the right output layer?
8. **Explain.** Why does a gradient kernel in which every logit's thread recomputes the row's maximum and sum do $C(4C + 4)$ operations per row, and what two numbers per row does the two-pass version store between its passes?

## Exercises

1. **[Calculate]** For $\mathbf{z} = (1, 0, -1)$ compute the softmax by hand to four decimals. Take the true class to be the last, compute the loss as $\mathrm{lse}(\mathbf{z}) - z_c$ and the gradient $\mathbf{p} - \mathbf{t}$, and check that the gradient adds to zero. Then repeat with $\mathbf{z} + 7$ and explain what changed and what did not.

2. **[Calculate]** For two classes with logits $(0.3, 1.1)$ and the true class $1$, compute the loss as a cross-entropy over the softmax and as the binary cross-entropy of Volume II on the logit difference. Compute $\partial L/\partial z_1$ and $\partial L/\partial z_0$ and show that the first equals $\sigma(z_1 - z_0) - 1$, the gradient of Volume II, and the second is its negative.

3. **[Calculate]** For $\mathbf{z} = (2, 0)$, compute $\mathrm{softmax}(a\mathbf{z})$ for $a = 1$ and $a = 2$. At what scale $a$ does the probability of the first class reach $0.99$? Repeat for the four logits $(2, 0, 0, 0)$ and explain why the scale needed is larger.

4. **[Calculate]** Smoothed targets for $C = 10$ and $\epsilon_{\mathrm{ls}} = 0.1$: the target of the true class and of each other class, the logit gap at which the loss is smallest, and the smallest loss (the entropy of the target). Compare the gap with the four-class value $3.6109$ of the chapter and explain the direction of the difference.

5. **[Calculate]** Take $\mathbf{p}$ from the chapter's part and the upstream vector $\mathbf{u} = (0, -1/p_1, 0, 0)$. Compute $\mathbf{u}\mathbf{J}$ by the formula $p_j(u_j - \mathbf{u}\cdot\mathbf{p})$ and show that it is $\mathbf{p} - \mathbf{t}$ for the true class $1$.

6. **[Implement]** Write the stable softmax and the log-softmax in your language, with a mode that rounds every operation to 32 bits. Reproduce the tables of `softmax.overflow_and_underflow` and `softmax.log_softmax_underflow`.

7. **[Implement]** Build the Jacobian $\mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ and the vector-Jacobian product, and the central-difference check of the loss. Reproduce `softmax.softmax_jacobian` and `softmax.gradient_through_the_jacobian`.

8. **[Implement]** Train the linear classifier $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$ on the part with $d\mathbf{W} = (\mathbf{p} - \mathbf{t})\mathbf{x}^\top$ and $\eta = 0.5$. Reproduce the table of `softmax.gradient_descent_on_weights`, and the batch table of `softmax.a_batch`.

9. **[Implement]** Implement the smoothed target, the loss $\mathrm{lse}(\mathbf{z}) - \sum_i t'_i z_i$ and the closed-form best gap. Reproduce `softmax.label_smoothing_target` and `softmax.label_smoothing_optimum`.

10. **[Investigate]** The cross-entropy under a scale $a$ is $L(a\mathbf{z}, c)$. Use the chain rule to derive $\partial L/\partial \mathbf{z}$ and $\partial L/\partial a$, and check both against central differences on the part's logits. Where would the factor $a$ need to be applied in a kernel that computes $\mathbf{p} - \mathbf{t}$, and what happens to the gradient's bound of $1$ when $a > 1$?

11. **[Investigate]** Instrument the three algorithms (softmax, loss value, two-pass gradient) with an operation counter and reproduce `softmax.work_counts`. Then count the operations of a one-pass softmax that keeps a running maximum $m$ and a running sum $s$, rescaling $s \leftarrow s\,e^{m_{\text{old}} - m_{\text{new}}}$ whenever the maximum changes, and say when it does more exponentials than the two-pass version.

## Summary

A classifier with $C$ classes ends in $C$ logits $\mathbf{z}$, and the **softmax** turns them into a probability vector, $p_i = e^{z_i}/\sum_j e^{z_j}$: positive, adding to one, in the order of the logits. For two classes it is the sigmoid of the logit difference, and in general it is the only probability vector whose log-ratios are the differences of the logits. It is unchanged by adding a constant to every logit (so only the differences count) and sharpened or flattened by multiplying them. The **stable softmax** subtracts the largest logit first, which is allowed by that invariance and keeps every exponent at most zero; the log-softmax $z_i - \mathrm{lse}(\mathbf{z})$ never forms a probability whose logarithm would underflow. With a **one-hot encoding** $\mathbf{t}$ of the true class, the loss is the negative log-likelihood of the categorical, the **softmax cross-entropy** $L = \mathrm{lse}(\mathbf{z}) - z_c$. The Jacobian of the softmax is $\mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$, symmetric, with rows adding to zero; composed with the derivative $-t_i/p_i$ of the logarithm it collapses, because the division cancels the factor of the Jacobian, to the gradient $\mathbf{p} - \mathbf{t}$: push every wrong logit down by its probability and the true one up by what is missing, a gradient bounded by $1$ that vanishes only when the prediction matches the target. Computing it fused is shorter and safer than composing the two functions, as the example with logits $(30, -100, 5, 0)$ shows (loss $130$ and a finite gradient against $\infty$ and not-a-number). **Label smoothing** mixes the target with the uniform distribution, $\mathbf{t}' = (1 - \epsilon_{\mathrm{ls}})\mathbf{t} + \epsilon_{\mathrm{ls}}\mathbf{1}/C$, which changes the gradient to $\mathbf{p} - \mathbf{t}'$ and gives the loss a minimum at a finite logit gap, $\ln 37 = 3.6109$ for four classes at $0.1$, instead of an infimum approached only at infinity. A softmax is for one true class; independent yes-or-no questions get a sigmoid each. For a row of $C$ logits the loss needs $4C + 2$ operations and the gradient $9C - 1$ in the two-pass form against $4C^2 + 4C$ when every entry recomputes the row, a ratio of $22\,223$ at $C = 50\,000$; the loss is a fraction of a percent of the arithmetic of the layer that produces the logits but reads its whole output, which is why brain keeps only a chunk of the logits alive.

In brain this is `ce_value.wgsl` and `ce_grad.wgsl` with their masked and two-pass variants, the chunked head `crates/model/src/head_ce.rs`, the host log-softmax in `crates/model/src/logprobs.rs`, and a family of softmax kernels whose backward is the one-line rule of this chapter. Label smoothing is not implemented in brain.

Limits: we have treated the loss of one example and its gradient with respect to the logits, and have taken the logits as given. How the gradient travels through the layers below, with the cost of one more pass, is the next chapter, which derives backpropagation from the chain rule; the one after it writes out the backward pass of every layer, including the softmax in the middle of a network.

## Key terms

- **softmax**: the function that turns a vector of real scores into a probability vector by exponentiating each score and dividing by the sum of the exponentials, so that the outputs are positive, add to one and keep the order of the scores. <!-- concept: softmax -->
- **stable-softmax**: the softmax computed after subtracting the largest score from every score, which gives the same probabilities but keeps every exponent at most zero, so nothing overflows and the sum is at least one. <!-- concept: stable-softmax -->
- **softmax-cross-entropy**: the loss of a classifier computed directly from its scores and the true class as the log-sum-exp of the scores minus the score of the true class; it is the negative log-likelihood of the true class under the softmax, and its gradient with respect to the scores is the probability vector minus the one-hot target. <!-- concept: softmax-cross-entropy -->
- **one-hot-encoding**: writing a class index as a vector with a one in the position of that class and zeros elsewhere, the probability vector that puts all its weight on that class. <!-- concept: one-hot-encoding -->
- **label-smoothing**: replacing the one-hot target by a mixture of it with the uniform distribution, so that the loss is smallest at a finite gap between the scores instead of only in the limit of infinite confidence. <!-- concept: label-smoothing -->

## Answers

**Before you begin.**

1. $\sigma(1.5) = 1/(1 + e^{-1.5}) = 0.8176$ for "rejected", so $0.1824$ for "accepted".
2. All four numbers non-negative and adding to $1$. The indicator list is $(0, 0, 1, 0)$: a one in the position of the outcome that occurred.
3. $-\ln 0.5 = 0.6931$ (only the term of the certain outcome survives). For a single certain outcome the cross-entropy is $-\ln q$ of that outcome, the surprise of the model at it.
4. $\mathrm{lse}(0, 0) = \ln 2 = 0.6931$. The largest score is subtracted so that every exponent is at most $0$, which cannot overflow, and the sum contains a term equal to $1$, which cannot underflow to $0$.
5. $[4, 4]$ (outputs by inputs). The entry $\partial f_i/\partial x_j$ says how output $i$ changes when input $j$ changes.

**Check your understanding.**

1. Adding $c$ to every logit multiplies every exponential by $e^c$, which cancels between numerator and denominator. So the output depends only on the differences of the logits: $C - 1$ independent numbers (the $C$ probabilities add to one, which is the same count), and one number for $C = 2$, the gap $z_1 - z_0$.
2. The textbook formula overflows: $e^{1000} = \infty$ in 64 bits, and $\infty/\infty$ is not a number. The stable softmax subtracts $1000$ and gives $(0.5, 0.5)$. The loss is $\mathrm{lse}(\mathbf{z}) - z_0 = (1000 + \ln 2) - 1000 = \ln 2 = 0.6931$.
3. For $\mathbf{z} = (30, -100, 5, 0)$ with true class $1$, the probability $p_1 = e^{-130}$ underflows to $0$ in 32 bits and the loss is $-\ln 0 = \infty$. The formula that does not break is $\mathrm{lse}(\mathbf{z}) - z_c$ with the stable log-sum-exp, which gives $130$.
4. The gradient is small, not zero. With $p_c = 0.99$ and $0.01/3$ on each other class, the true entry is $-0.01$ and each other entry $+0.00333$. It vanishes only if $p_c = 1$ exactly, which needs an infinite logit gap; so training with a one-hot target keeps pushing the logits apart, and label smoothing is what stops that.
5. $\mathbf{t}'$ has $0.85 + 0.15/5 = 0.88$ for the true class and $0.15/5 = 0.03$ for each of the four others, adding to $0.88 + 0.12 = 1$. The best gap is $\ln(0.88/0.03) = \ln 29.33 = 3.3787$.
6. It reflects shift invariance: adding a constant to all logits changes nothing, so the derivative along $\mathbf{1}$ is zero, and the gradient is orthogonal to $\mathbf{1}$ (the entries add to zero). A gradient step raises and lowers logits by equal total amounts, so it never changes the mean logit.
7. The softmax says exactly one class is true, so its probabilities compete and the best it can fit to $(1, 1, 0)$ is a target of $(0.5, 0.5, 0)$ after normalising, which says "one of the two". The right output layer is one sigmoid per class with a binary cross-entropy each, so that "dog" and "ball" can both be near $1$.
8. Each of the $C$ threads computes the maximum ($C - 1$ compares) and the sum ($3C$ operations: subtract, exponential, add per class), which is $4C - 1$ operations, plus its own $5$ operations; $C$ threads give $C(4C + 4)$, quadratic in $C$. The two-pass version stores the row maximum and the row sum, two numbers per row, and every thread of the second pass reads them.

## References

- Bridle, J. S., *Probabilistic interpretation of feedforward classification network outputs, with relationships to statistical pattern recognition*, 1990 (the name softmax and its use as the output of a classifier).
- Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. and Wojna, Z., *Rethinking the Inception architecture for computer vision*, 2016 (label smoothing).
- Müller, R., Kornblith, S. and Hinton, G., *When does label smoothing help?*, 2019.
- Blanchard, P., Higham, D. J. and Higham, N. J., *Accurately computing the log-sum-exp and softmax functions*, 2021.
- Milakov, M. and Gimelshein, N., *Online normalizer calculation for softmax*, 2018.
