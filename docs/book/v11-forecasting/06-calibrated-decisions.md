<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Scoring Rules and Calibrated Decisions

*How do we reward a model for honest probabilities rather than confident guesses?*

Suppose you hire a forecaster and tell them how they will be paid. The event is rain tomorrow, and the forecaster knows, because the forecaster has studied the sky for years, that the chance is exactly $0.7$. They may report any number $q$ between $0$ and $1$, and when the day is over they pay a penalty that depends on $q$ and on whether it rained. Which penalty makes them report $0.7$?

The table tries five. Each cell is the penalty the forecaster expects to pay, $0.7$ times the penalty if it rains plus $0.3$ times the penalty if it does not. A rule that rewards honesty has its smallest cell in the row $q = 0.7$.

<!-- worked: calibrated_decisions.report_game -->
The event has probability p = 0.7. The forecaster's report q is scored by each loss with outcome y = 1 (probability 0.7) or y = 0; the cells are the expected loss 0.7 L(q, 1) + 0.3 L(q, 0). 'side of 1/2' is 0 when q >= 0.5 and y = 1 or q < 0.5 and y = 0, else 1; 'absolute' is |y - q|; 'Brier' is (y - q)^2; 'log' is -ln of the probability the report gave to the outcome; 'focal 2' is (1 - r)^2 (-ln r) for the probability r the report gave to the outcome. Four decimals; inf is an outcome the report called impossible. The last row scans q = k/1000 for k = 0 .. 1000 and keeps the first q whose expected loss is smaller than every earlier one by more than 1e-12.

| report q | side of 1/2 | absolute | Brier | log | focal 2 |
|---|---:|---:|---:|---:|---:|
| 0.0 | 0.7000 | 0.7000 | 0.7000 | inf | inf |
| 0.1 | 0.7000 | 0.6600 | 0.5700 | 1.6434 | 1.3059 |
| 0.2 | 0.7000 | 0.6200 | 0.4600 | 1.1935 | 0.7237 |
| 0.3 | 0.7000 | 0.5800 | 0.3700 | 0.9498 | 0.4226 |
| 0.4 | 0.7000 | 0.5400 | 0.3000 | 0.7947 | 0.2554 |
| 0.5 | 0.3000 | 0.5000 | 0.2500 | 0.6931 | 0.1733 |
| 0.6 | 0.3000 | 0.4600 | 0.2200 | 0.6325 | 0.1562 |
| 0.7 | 0.3000 | 0.4200 | 0.2100 | 0.6109 | 0.1995 |
| 0.8 | 0.3000 | 0.3800 | 0.2200 | 0.6390 | 0.3153 |
| 0.9 | 0.3000 | 0.3400 | 0.2500 | 0.7645 | 0.5603 |
| 1.0 | 0.3000 | 0.3000 | 0.3000 | inf | inf |
| best report on the 0.001 grid | 0.500 | 1.000 | 0.700 | 0.700 | 0.578 |
<!-- /worked -->

Read the columns one at a time. Paying by the side of $1/2$ (a penalty of $1$ for a report on the wrong side of one half) gives the same expected penalty to every report from $0.5$ to $1$: the forecaster may say $0.5$, $0.7$ or $1.0$ and pay $0.3$, and the rule cannot tell a guess from knowledge. Paying by the absolute error is worse: the expected penalty falls in a straight line as $q$ rises, so the forecaster who knows the chance is $0.7$ should report $1$, a certainty they do not believe. This is what "rewarding confident guesses" means in the chapter's question. The squared error and the negative logarithm of the probability given to what happened both have their minimum at $0.7$, and a fifth rule, the focal penalty $(1 - r)^2(-\ln r)$ for the probability $r$ given to the outcome, which some classifiers are trained with, has its minimum at $0.578$: it rewards a forecaster for understating a chance of $0.7$.

The rest of the volume has used scores like these without asking why they should be trusted. Chapter 2 scored a distribution by the pinball loss and the CRPS, Chapter 4 compared forecasters by them, and Chapter 5 scored a survival probability at a horizon by the mean of $(y - \hat p)^2$ and promised to say where that number comes from. This chapter says. It asks what a score must satisfy so that a model trained or ranked by it has no reason to lie, shows that two familiar scores satisfy it and takes one of them apart into a part about calibration and a part about discrimination, repairs a model's probabilities without fooling oneself about the repair, and then does what probabilities are for: it combines them with costs to choose an action, including the action of not answering.

Before the machinery, six things from earlier are needed. Answer them from memory; the answers are at the end of the chapter.

1. (Volume I, Chapter 10) A random variable takes the value $1$ with probability $p$ and $0$ otherwise. What is the expected value of $(Y - q)^2$ for a fixed number $q$, and how does it change if you write it as $E[Y^2] - 2qE[Y] + q^2$?
2. (Volume I, Chapter 13) For distributions $p$ and $q$, how is the cross-entropy of $q$ relative to $p$ related to the entropy of $p$, and when is the difference zero?
3. (Volume II, Chapter 3) A logistic model gives $p = \sigma(z)$. What is the derivative of the cross-entropy loss for a label $y$ with respect to the logit $z$?
4. (Volume II, Chapter 6) Define calibration, say how a reliability diagram bins predictions, and give the threshold on a calibrated probability that minimises expected cost when a false alarm costs $c_{\mathrm{FP}}$ and a miss costs $c_{\mathrm{FN}}$.
5. (Chapter 2) The CRPS of a forecast distribution $F$ at an observation $y$ is an integral over $x$. Of what?
6. (Chapter 5) What does the squared-error score of a survival probability at a horizon $t^\star$ average, and what does a forecast of $0.5$ for everyone score?

## Why the minimum has to sit at the truth

Fix the vocabulary. An event has a probability $p$; a forecaster reports a number $q \in [0, 1]$; afterwards the outcome $y \in \{0, 1\}$ is revealed and a loss $\ell(q, y)$ is charged. The forecaster who believes $p$ and wants the smallest average loss will report the $q$ that minimises the expected loss

$$\bar\ell(q; p) = p\,\ell(q, 1) + (1 - p)\,\ell(q, 0).$$

A loss is a **proper scoring rule** when reporting what you believe is never beaten: $\bar\ell(q; p) \ge \bar\ell(p; p)$ for every $p$ and every $q$. It is strictly proper when equality holds only at $q = p$, so that the truth is the unique best report. The word score is used for the loss or for its negative, whichever the context makes plainer; here the losses are minimised.

The squared error is the first one to check, and the algebra is the one of the first recall question. With $\ell(q, y) = (y - q)^2$,

$$\bar\ell(q; p) = p(1 - q)^2 + (1 - p)q^2 = q^2 - 2pq + p = (q - p)^2 + p(1 - p).$$

The expected loss is a parabola in $q$ with its vertex at $p$, and its lowest value, $p(1 - p)$, is the variance of the outcome, the uncertainty nothing can remove. The mean of $(y - q)^2$ over many events is the **Brier score**. Its honest value for the forecaster of the opening is $0.7 \cdot 0.3 = 0.21$, the Brier column of the table at $q = 0.7$, and any other report adds $(q - p)^2$: $0.01$ at $0.8$, $0.04$ at $0.5$.

The logarithmic penalty $\ell(q, 1) = -\ln q$, $\ell(q, 0) = -\ln(1 - q)$ is the **log score**, the negative logarithm of the probability the forecast gave to what happened. It is the cross-entropy of Volume I and the loss of the logistic model of Volume II, and Chapter 2 used it under the name of the negative log-likelihood. Its expected value is

$$\bar\ell(q; p) = -p\ln q - (1 - p)\ln(1 - q) = H(p) + D_{\mathrm{KL}}(p \,\Vert\, q),$$

the entropy of the truth plus the divergence of the report from it. Gibbs's inequality, proved in Volume I, says the divergence is zero only at $q = p$. So the log score is strictly proper too, with honest value $H(0.7) = 0.6109$ in the table, and with an excess $D_{\mathrm{KL}}$ that grows without bound as the report approaches a certainty the truth denies: the cells for $q = 0$ and $q = 1$ are infinite because a report of zero was charged $-\ln 0$ on a day it rained.

A third rule completes the picture, and it will return in the brain section. The spherical score rewards the report $\mathbf{v} = (1 - q, q)$ for the probability it gives to the outcome, divided by the length of $\mathbf{v}$: for outcome $y$ it is $v_y / \lVert \mathbf{v} \rVert$. With $\mathbf{u} = (1 - p, p)$ and $\mathbf{v} = (1 - q, q)$, its expectation is $\mathbf{u} \cdot \mathbf{v} / \lVert \mathbf{v} \rVert$, and the Cauchy-Schwarz inequality bounds it by $\lVert \mathbf{u} \rVert$, with equality only when $\mathbf{v}$ points the way $\mathbf{u}$ does; as both sum to one, only when $\mathbf{v} = \mathbf{u}$. It is strictly proper, a score to be maximised, and unlike the logarithm it stays bounded.

Now the failures. The absolute error has $\bar\ell(q; p) = p(1 - q) + (1 - p)q = p + (1 - 2p)q$, which is linear in $q$, so its minimum is at an end point, $q = 1$ when $p > 1/2$ and $q = 0$ when $p < 1/2$. It is not proper, and training a model on it yields a model that rounds. The side-of-one-half rule has $\bar\ell = 1 - p$ for every $q \ge 1/2$ and $p$ for every $q < 1/2$, a constant on each side: honesty is never beaten, so the rule is proper, but not strictly, and a model trained on it receives no gradient that moves a report from $0.51$ to $0.7$. Accuracy is this rule, and that is the first sense in which accuracy is a poor guide to probabilities. The focal penalty is neither: its minimiser for $p = 0.7$ on a grid of step $0.001$ is $0.578$, and brain's calibration objective says so in its module header, where the focal term is described as improper and kept because it often lands better calibrated on held-out data. That is a claim about generalisation, which this chapter does not test; what it can confirm is the improperness.

![Expected loss above the best report for an event of probability 0.7 under five rules, as a function of the report. The absolute error is lowest at the right end. The side-of-one-half rule is a step with a flat floor from 0.5 to 1.0. The focal penalty is lowest at 0.578. The squared error and the log score are lowest at 0.7, the vertical line.](img/propriety-curves.svg)

The picture gives the same verdict: five curves above the best report, and only two of them touch zero at the vertical line.

## A score as an average over the decisions a probability serves

Why these two? One answer is the algebra above. A better answer says what the score is made of, and it ties this chapter to its second half.

A user of the forecast will act on it, and the user's costs set a threshold. Volume II derived it: with a false alarm costing $c_{\mathrm{FP}}$ and a miss costing $c_{\mathrm{FN}}$, a calibrated probability should trigger the action when it reaches $t^\star = c_{\mathrm{FP}} / (c_{\mathrm{FP}} + c_{\mathrm{FN}})$. Write $t$ for that threshold and scale the costs so that $c_{\mathrm{FP}} = t$ and $c_{\mathrm{FN}} = 1 - t$. A user with threshold $t$ who receives the report $q$ acts when $q \ge t$ and pays

$$L_t(q, y) = t\,\mathbf{1}[q \ge t,\ y = 0] + (1 - t)\,\mathbf{1}[q < t,\ y = 1].$$

Average this loss over all users, with every threshold equally represented. When $y = 1$ the loss is $1 - t$ for the users with $t > q$, so the integral is $\int_q^1 (1 - t)\,dt = (1 - q)^2/2$. When $y = 0$ it is $t$ for the users with $t \le q$, so the integral is $\int_0^q t\,dt = q^2/2$. In both cases the average over thresholds is half the squared error:

$$(y - q)^2 = 2\int_0^1 L_t(q, y)\,dt.$$

The Brier score is the average cost to a population of users whose cost ratios are spread evenly over thresholds. Change the weighting and another score appears: weighting the threshold $t$ by $1/(t(1 - t))$ gives $\int_q^1 dt/t = -\ln q$ for $y = 1$ and $\int_0^q dt/(1 - t) = -\ln(1 - q)$ for $y = 0$, exactly the log score. The log score counts users with extreme thresholds, those for whom a miss costs a thousand times a false alarm, much more heavily, and for the user whose threshold is $0.001$ a report of exactly zero on a day the event happened is a catastrophe without a bound.

<!-- worked: calibrated_decisions.scores_as_threshold_averages -->
A report q is acted on at a threshold t as 'act if q >= t'. The loss is t when it acts and the label is 0 and 1 - t when it does not act and the label is 1, else 0. The midpoint rule at t = (j + 0.5)/1000, j = 0 .. 999, averages the loss over t (weight 1) and the loss divided by t (1 - t) (weight 1/(t(1-t))). Five decimals for the first pair, four for the second.

| q | y | mean loss over t | (y - q)^2 / 2 | weighted mean loss | -ln(prob. of the label) |
|---|---:|---:|---:|---:|---:|
| 0.30 | 1 | 0.24500 | 0.24500 | 1.2040 | 1.2040 |
| 0.30 | 0 | 0.04500 | 0.04500 | 0.3567 | 0.3567 |
| 0.90 | 1 | 0.00500 | 0.00500 | 0.1054 | 0.1054 |
| 0.90 | 0 | 0.40500 | 0.40500 | 2.3026 | 2.3026 |
| 0.02 | 1 | 0.48020 | 0.48020 | 3.9119 | 3.9120 |
<!-- /worked -->

The midpoint rule at $1000$ thresholds agrees with the closed forms to the digits shown (the last row differs by one unit in the fourth decimal because the weight $1/(t(1-t))$ is steep near $0$).

Propriety follows. For one threshold $t$, acting costs $t(1 - p)$ in expectation and not acting costs $(1 - t)p$, so the cheaper choice is to act exactly when $p \ge t$. Any report on the same side of $t$ as $p$ makes the same choice, so each $L_t$ is proper (but only weakly). A report $q \ne p$ falls on the wrong side of every threshold between $q$ and $p$, and a rule that gives positive weight to all of them pays for that on a whole interval of users. Averaging with positive weights is what turns weak propriety into strict propriety. It is also, in outline, the converse: every proper rule for a binary event can be written as such an average, a theorem of Savage and of Schervish that this book uses without proof.

The same decomposition accounts for scores met earlier in the volume. Chapter 2 defined the CRPS as $\int (F(x) - \mathbf{1}_{x \ge y})^2 dx$. For each $x$ the integrand is the Brier score of the forecast probability $F(x)$ of the event "the outcome is at most $x$", whose value is $\mathbf{1}_{x \ge y}$, so the CRPS is the Brier score integrated over the thresholds of the outcome. Chapter 5 scored a survival probability at a horizon $t^\star$: that is the Brier score of the event "failed by $t^\star$". A small integer forecast shows the identity with the integral written as a sum, since a step distribution function is constant between integers.

<!-- worked: calibrated_decisions.crps_from_brier -->
A forecast of the integer outcome 0, 1, 2, 3 with probabilities 0.1, 0.4, 0.3, 0.2; the outcome is 2. The event 'outcome <= k' has probability F(k) (the running total of the probabilities) and the value 1 when y <= k. Its Brier score is (F(k) - value)^2; the thresholds k = 0, 1, 2 are the ones between outcomes (k = 3 contributes 0).

| event | forecast F(k) | value | Brier score |
|---|---:|---:|---:|
| outcome <= 0 | 0.1 | 0 | 0.0100 |
| outcome <= 1 | 0.5 | 0 | 0.2500 |
| outcome <= 2 | 0.8 | 1 | 0.0400 |
| sum |  |  | 0.3000 |

The CRPS from the definition E|X - y| - E|X - X'|/2 (X and X' independent draws from the forecast) is 0.3000.
<!-- /worked -->

The three Brier scores add to the CRPS computed from its definition as an expected distance, $E|X - y| - \tfrac12 E|X - X'|$. There was one score all along, applied to different events.

## Taking a Brier score apart

A Brier score of $0.1875$ is hard to read by itself. Is it large because the forecaster says the wrong things, or because the events are hard? The decomposition of Murphy answers, and it needs only the binning of the reliability diagram of Volume II.

Take $n$ forecasts $q_i$ with outcomes $y_i$ and sort them into $K$ equal-width bins: bin $k$ holds $k/K \le q < (k+1)/K$, and $q = 1$ goes into the last bin. Let $n_k$ be the number of forecasts in bin $k$, $\bar q_k$ their mean forecast, $\bar y_k$ the share of them with outcome $1$, and $\bar y$ the overall share. If every forecast in a bin were exactly $\bar q_k$, then

$$\mathrm{BS} = \underbrace{\sum_k \tfrac{n_k}{n}(\bar q_k - \bar y_k)^2}_{\mathrm{REL}} - \underbrace{\sum_k \tfrac{n_k}{n}(\bar y_k - \bar y)^2}_{\mathrm{RES}} + \underbrace{\bar y(1 - \bar y)}_{\mathrm{UNC}}.$$

The first term, the **reliability**, is the calibration gap squared and averaged over the examples: zero when every bin's claim matches its frequency. The second, the **resolution**, measures how far the bins' frequencies lie from the overall frequency; it is large when the forecasts separate cases that behave differently, and a forecaster who always says the same thing has none. The third, the **uncertainty**, is the Brier score of the forecaster who always says $\bar y$. It belongs to the data, not to the model. Good forecasts have small reliability and large resolution, and the decomposition says that the two are independent virtues, which Volume II said in words: the base-rate forecast is perfectly calibrated and ranks nothing.

To derive it, split $q_i - y_i$ in bin $k$ into three pieces, $(q_i - \bar q_k) + (\bar q_k - \bar y_k) + (\bar y_k - y_i)$, and square. The pieces sum to zero over the bin in the first and third positions, so the cross terms with the middle piece vanish. Of the rest, $\sum_i (\bar y_k - y_i)^2 = n_k \bar y_k(1 - \bar y_k)$ for labels that are $0$ or $1$, and summing over bins turns $\bar y_k(1 - \bar y_k)$ into $\bar y(1 - \bar y)$ minus the resolution. What is left over is the cross term between the first and third pieces and the within-bin spread of the forecasts:

$$\mathrm{BS} - (\mathrm{REL} - \mathrm{RES} + \mathrm{UNC}) = \frac1n\sum_i (q_i - \bar q_{k(i)})^2 - \frac2n\sum_i (q_i - \bar q_{k(i)})(y_i - \bar y_{k(i)}),$$

which is zero exactly when the forecasts do not vary inside a bin. The identity is exact for forecasters that use a few distinct values and nearly exact for the rest, and the remainder is worth printing, to see how nearly.

Ten forecasts make the arithmetic visible before the larger tables.

<!-- worked: calibrated_decisions.decomposition_by_hand -->
Ten forecasts: four of probability 0.2 and six of probability 0.9, with the outcomes listed (1 = the event happened).

| forecast | count | outcomes | share of labels 1 |
|---:|---:|---|---|
| 0.2 | 4 | 1000 | 0.2500 |
| 0.9 | 6 | 111101 | 0.8333 |

Brier score 0.1620; share of labels 1 0.6, uncertainty 0.2400; reliability 0.003667; resolution 0.081667; reliability - resolution + uncertainty = 0.1620.
<!-- /worked -->

The four forecasts of $0.2$ came true $25\%$ of the time, a gap of $0.05$, and the six of $0.9$ came true $83\%$, a gap of $0.067$; the weighted squares of the gaps are the reliability $0.0037$. The two bins separate the data well, $0.25$ against $0.83$ around an overall $0.6$, which is the resolution $0.0817$. The data's own uncertainty is $0.6 \cdot 0.4 = 0.24$, and $0.0037 - 0.0817 + 0.24 = 0.162$ is the Brier score computed directly from the ten pairs.

Now the generated forecaster that this chapter keeps using. Its draws and its recipe are printed with the table. The truth is a logistic model of a latent $s$; the model under test knows the truth but reports it through a distortion, a logit twice as steep as it should be with a bend and a shift, so that its probabilities are too extreme and too high, in the manner of a classifier fitted to near-separable data. The first four rows are forecasters of the fresh set, and the fifth is the base rate.

<!-- worked: calibrated_decisions.decomposition -->
Generator: the book's SplitMix64, seed 0x461acfe70792e064 (5051578523220893796). A normal() call that finds a held value returns it without drawing, and a held value left at the end of a series is carried into the next one (the generator is never reset). Example i of the fit set (2000 examples), then of the test set (300), then of the fresh set (20000), in that order from one stream, draws s = normal(), then u = uniform(); eta = -0.8 + 1.5 * s, p = sigmoid(eta), and the label is y = 1 when u < p. The model's logit is z = 2 eta + 0.5 eta |eta| + 0.6. The table is on the fresh set (share of labels 1: 0.3620). Equal-width bins: bin k holds k/10 <= q < (k+1)/10, and q = 1 goes to bin 9. REL = sum over nonempty bins of (n_k/n)(mean q - mean y)^2, RES = sum of (n_k/n)(mean y in the bin - mean y)^2, UNC = ybar (1 - ybar); 'remainder' is BS - (REL - RES + UNC). 'Scale and shift' is sigmoid(a z + b) with (a, b) fitted on the fit set by Newton's method on the cross-entropy: start at (a, b) = (1, 0); each iteration forms the mean gradient and mean Hessian of the cross-entropy over the fit set, computes the Newton step, halves it until the cross-entropy at the stepped point is no larger than at the current one (or the step factor reaches 1e-6), and moves; at most 100 iterations, stopping when the move has |da| + |db| < 1e-12. Undamped Newton from (1, 0) diverges on this data (a = -1.2554 after one step, 26.9 after two), so the halving is part of the recipe. The base rate is the share of labels 1 in the fit set. ECE is the 10-bin expected calibration error of the probability of the event; log loss is the mean -ln of the probability given to the label.

| forecaster | BS | REL | RES | UNC | remainder | ECE | log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| true probability | 0.1672 | 0.0000 | 0.0629 | 0.2310 | -0.00091 | 0.0050 | 0.5032 |
| raw model | 0.1875 | 0.0191 | 0.0609 | 0.2310 | -0.00165 | 0.1286 | 0.6853 |
| model after scale and shift | 0.1680 | 0.0008 | 0.0630 | 0.2310 | -0.00077 | 0.0246 | 0.5054 |
| true probability, bin centre | 0.1681 | 0.0001 | 0.0629 | 0.2310 | 0.00000 | 0.0064 | 0.5072 |
| base rate of the fit set | 0.2310 | 0.0000 | 0.0000 | 0.2310 | 0.00000 | 0.0025 | 0.6546 |
<!-- /worked -->

Take the rows in turn. The true probability is the best any forecaster can do with this information. Its reliability is zero to four decimals, its resolution $0.0629$ is large, and its Brier score $0.1672$ is the uncertainty $0.2310$ minus that resolution, plus the remainder $-0.0009$ that the within-bin spread leaves. The raw model has almost the same resolution, $0.0609$, so it separates the cases nearly as well, but a reliability of $0.0191$ where the truth has none lifts its Brier score to $0.1875$, $0.0203$ above the truth's. The next row anticipates the following section: after a scale and a shift fitted on separate data, the reliability falls to $0.0008$ and the Brier score to $0.1680$, within $0.0008$ of the truth. The row of bin centres rounds the true probability to the middle of its bin, so that forecasts take ten values and the remainder is exactly zero to five decimals, the identity holding as an identity; the rounding costs $0.0009$ of Brier score, the price of ten levels. The base rate has no resolution at all, a Brier score equal to the uncertainty, and an expected calibration error of $0.0025$.

That last number is the point about the expected calibration error. It is the weighted mean of the absolute gaps of the bins, and a forecaster who reports the base rate fits its single bin almost perfectly; the raw model, whose Brier score is much lower, has an error of $0.1286$. A ranking of the two by calibration error alone would put the useless forecaster first. The error is not a proper score, it measures only the reliability half and ignores resolution, and it has a second weakness that Volume II measured: its value depends on the binning, so that the number of bins is part of the metric. The convention here is the one of the earlier chapter, spelled out: equal-width bins, bin $k$ holds $k/10 \le q < (k+1)/10$, a forecast of exactly $1$ lands in bin $9$, empty bins are skipped, and each bin is weighted by its count.

The bins of the raw model, which are the dots of the reliability diagram, show where the error sits.

<!-- worked: calibrated_decisions.reliability_table -->
The raw model on the fresh set of `calibrated_decisions.decomposition`: the 10 equal-width bins of its probability (bin k holds k/10 <= q < (k+1)/10), the number of examples, their mean forecast, the share of them with label 1, and the absolute gap. Four decimals.

| bin | count | mean forecast | share of labels 1 | gap |
|---|---:|---:|---:|---:|
| 0.0 to 0.1 | 8420 | 0.0184 | 0.1240 | 0.1056 |
| 0.1 to 0.2 | 1441 | 0.1460 | 0.2887 | 0.1427 |
| 0.2 to 0.3 | 1048 | 0.2470 | 0.3292 | 0.0822 |
| 0.3 to 0.4 | 894 | 0.3505 | 0.3826 | 0.0321 |
| 0.4 to 0.5 | 867 | 0.4501 | 0.3829 | 0.0672 |
| 0.5 to 0.6 | 892 | 0.5510 | 0.4641 | 0.0869 |
| 0.6 to 0.7 | 982 | 0.6500 | 0.5081 | 0.1418 |
| 0.7 to 0.8 | 991 | 0.7505 | 0.5499 | 0.2006 |
| 0.8 to 0.9 | 1220 | 0.8519 | 0.6098 | 0.2420 |
| 0.9 to 1.0 | 3245 | 0.9718 | 0.7886 | 0.1832 |

ECE = sum of (count / 20000) x gap = 0.1286.
<!-- /worked -->

The lowest bin holds $8420$ of the $20000$ examples, and the model said $1.8\%$ for them on average; $12.4\%$ of them had the label $1$. The highest bin holds $3245$ with a mean forecast of $0.972$ and a frequency of $0.789$. The model is too sure at both ends, and since the lowest bin is also the largest, it contributes $0.421 \cdot 0.1056 = 0.0445$ of the $0.1286$, more than any other, and it is the bin that a threshold near zero will meet in the decision section.

## Repairing probabilities without grading the repair on its own exam

A model that ranks well but speaks with too much confidence can be repaired by a map $g$ from its output to a new probability, fitted on labelled examples that were not used to train the model, the "fit set" here. The simplest map is the one of Volume II: scale the logit and shift it, $g(z) = \sigma(a z + b)$, with $a$ and $b$ chosen to minimise the cross-entropy on the fit set. This is the form attributed to Platt, applied to the logit of a model that already produces probabilities; it is a convex two-parameter fit, and the condition that the derivative with respect to $b$ vanish says that the mean repaired forecast equals the mean label on the fit set. On the generated data the fit gives $a = 0.3043$ and $b = -0.2120$: shrink the logit to three tenths and move it down.

When the distortion is not of that shape, a nonparametric map is the alternative. **Isotonic regression** fits a nondecreasing step function to the labels, by least squares: among all functions that never decrease with the model's score, it picks the one with the smallest sum of squared errors, which is to say the smallest Brier score on the fit set. The algorithm is pool adjacent violators. Sort the examples by score; walk through them, putting each into a block of its own; whenever the block just formed has a smaller mean label than the block before it, which violates monotonicity, merge the two and repeat. Seven labels $1, 0, 1, 1, 0, 0, 1$ in score order become the blocks $(1, 0) \to 0.5$ after the first merge; $1$; $1$; then the incoming $0$ merges with the previous $1$ to give $0.5$, which is below the $1$ before it, so these merge into $(1, 1, 0)$ with mean $2/3$; the incoming $0$ merges again into $(1, 1, 0, 0)$ with mean $0.5$, which is not below the first block's $0.5$, so the process stops, and the last $1$ stands alone. The fitted values are $0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1$.

```text
function pav(score: array<real>[n], y: array<int>[n])
        -> (xs: array<real>[n], fitted: array<real>[n]):
    require n > 0, "nothing to fit"
    order = argsort(score)                  # ties keep input order
    sums = zeros(real, [n])                 # blocks 0 .. m-1: label sum and count
    counts = zeros(int, [n])
    m = 0
    for r in 0 .. n:
        i = order[r]
        sums[m] = y[i]                      # a block of its own
        counts[m] = 1
        m = m + 1
        while m > 1 and sums[m - 2] / counts[m - 2] > sums[m - 1] / counts[m - 1]:
            sums[m - 2] = sums[m - 2] + sums[m - 1]      # merge the last two blocks
            counts[m - 2] = counts[m - 2] + counts[m - 1]
            m = m - 1
    xs = zeros(real, [n])
    fitted = zeros(real, [n])
    r = 0
    for b in 0 .. m:
        for j in 0 .. counts[b]:
            xs[r] = score[order[r]]
            fitted[r] = sums[b] / counts[b]
            r = r + 1
    return (xs, fitted)

function isotonic_at(xs: array<real>[n], fitted: array<real>[n], z: real) -> real:
    j = 0                                   # the first block's value below every training score
    for r in 0 .. n:
        if xs[r] <= z:
            j = r                           # the largest training score not above z
    return fitted[j]
```

A new score receives the value of the block that holds the largest training score not above it, and the first block's value if it is below all of them. The comparison with a strict inequality means that two neighbouring blocks with equal means stay separate, which changes no value of the fitted function.

Now fit the three maps on the fit set and score them where it matters. The test set of $300$ examples is small, as test sets often are; the fresh set of $20000$ stands for the future. The last two recalibrators were fitted on the test examples themselves, which is the mistake the table is built to show.

<!-- worked: calibrated_decisions.recalibration -->
Data as in `calibrated_decisions.decomposition`; every scale-and-shift fit here (fit set or test set) uses the damped Newton recipe given there, started at (a, b) = (1, 0). The scale-and-shift fit on the fit set is a = 0.3043, b = -0.2120; the isotonic fit on the fit set has 80 blocks holding 30 distinct values, on the test set 25 blocks holding 14 (adjacent blocks of equal mean are not merged, so blocks can outnumber values). Isotonic regression pools adjacent violators of the labels sorted by ascending model logit z; the fitted function at a logit z is the value of the block that holds the largest training logit not above z, and the first block's value below all of them; the fit-set values run from 0.0000 to 1.0000. The model logits of the fit and test sets are all different, so no tie rule was used. Brier score and 10-bin ECE on the 300 test examples and on the 20000 fresh examples; four decimals.

| recalibrator | fitted on | test BS | test ECE | fresh BS | fresh ECE |
|---|---:|---:|---:|---:|---:|
| raw model | - | 0.1836 | 0.1332 | 0.1875 | 0.1286 |
| scale and shift, fitted on the fit set | fit | 0.1701 | 0.0523 | 0.1680 | 0.0246 |
| isotonic, fitted on the fit set | fit | 0.1700 | 0.0267 | 0.1685 | 0.0140 |
| scale and shift, fitted on the test set | test | 0.1686 | 0.0532 | 0.1675 | 0.0157 |
| isotonic, fitted on the test set | test | 0.1586 | 0.0000 | 0.1704 | 0.0409 |
| true probability | - | 0.1684 | 0.0582 | 0.1672 | 0.0050 |

The isotonic fit on the fit set gives exactly 0 or exactly 1 to 502 of the 20000 fresh examples, and 9 of those carry the other label.
<!-- /worked -->

![Reliability diagrams on 20000 fresh examples. The diagonal is perfect calibration. The raw model's curve lies far below the diagonal over most of its range and its first point sits above it. The scale-and-shift curve and the isotonic curve fitted on the fit set both follow the diagonal closely.](img/reliability-recalibrated.svg)

The raw model scores $0.1875$ on the fresh set. Both honest repairs, which see only the fit set, bring it to about $0.168$ (scale and shift $0.1680$, isotonic $0.1685$), within $0.0013$ of the true probability's $0.1672$, and the calibration error falls from $0.1286$ to $0.0246$ and $0.0140$. The figure shows the same.

Now read the rows fitted on the test set, which are the reason for the section's title. The isotonic map fitted on the $300$ test examples scores $0.1586$ on them, lower than the true probability's $0.1684$ on the same $300$. No forecast can beat the true probability on average, so the number is not skill: the step function has $25$ free blocks holding $14$ distinct values and has bent itself around the noise of those $300$ labels. On the fresh examples the same map scores $0.1704$, worse than either honest repair, and the calibration error on the test set, $0.0000$, is zero by construction, because each block of the fitted map is a set of examples with a common forecast equal to their frequency, and every block lies inside one bin. The scale-and-shift map fitted on the test set, with two free parameters, does not overfit visibly: $0.1686$ on the test set and $0.1675$ on fresh data. How much peeking costs is a function of the freedom of the map, which is the lesson of Chapter 4 on a test set that has been looked at, in a form where the number exposes itself: a score that beats the truth is a score that has seen the answers.

Two more cautions about the isotonic map. The fit-set values run all the way from $0.0000$ to $1.0000$, because a block at either end can consist only of zeros or only of ones, and the map gives exactly $0$ or $1$ to $502$ of the $20000$ fresh examples; $9$ of those carry the other label, and each of them costs an infinite log score. The Brier score charges such a mistake at most $1$; the logarithm charges without bound, which is why isotonic outputs are usually clipped away from $0$ and $1$ before they are scored by it. And a step function has ties: several different scores map to one value, which spoils a ranking that distinguished them, whereas the scale-and-shift map is strictly increasing and keeps every ranking. It does not keep a decision at a fixed threshold: the shift $b$ moves the point where the repaired probability crosses one half away from where the raw one did (the accuracy cuts of the next section, $0.68$ raw against $0.51$ recalibrated, are that move), so a rule at one half on the raw probability and the same rule on the repaired one disagree on every case between the two crossings.

## From a probability to an action

A probability is not an action. The same $0.2$ means block the shipment to one business and release it to another, and the difference is what each mistake costs. A calibrated probability makes the choice a computation.

Let there be actions $a$ and outcomes $y$, and a cost matrix $c(a, y)$ that says what taking $a$ costs if $y$ turns out true. For a belief $p$ over outcomes, the **expected cost** of $a$ is $\bar c(a; p) = \sum_y p_y\,c(a, y)$, and the Bayes action is the one with the smallest expected cost, whose value is the Bayes risk. The **regret** of taking some other action under the true belief is the excess of its expected cost over the Bayes risk. With two outcomes and two actions, block (cost $c_{\mathrm{FP}}$ if the device is healthy) and release (cost $c_{\mathrm{FN}}$ if it is faulty), blocking is cheaper when $(1 - p)c_{\mathrm{FP}} \le p\,c_{\mathrm{FN}}$, that is, when $p \ge c_{\mathrm{FP}}/(c_{\mathrm{FP}} + c_{\mathrm{FN}})$: Volume II's $t^\star$ again, now seen as the point where two expected costs cross. The **decision threshold** is this crossing point on the probability. It is not a property of the model; it is a property of the costs, and a model's job is to supply a $p$ that the threshold can be applied to.

The device world of brain's decision crate is a small instance, and one worth doing by hand. A device is faulty with probability $0.2$. A test comes back positive with probability $0.8$ if the device is faulty and $0.1$ if it is healthy. Blocking a healthy device costs $1$ and releasing a faulty one costs $10$, so the threshold is $1/11 = 0.0909$.

<!-- worked: calibrated_decisions.device_world -->
Costs: blocking a healthy device 1, releasing a faulty one 10, a correct action 0. Expected cost of an action = sum over outcomes of probability x cost. Prior P(fault) = 0.2, P(+ | fault) = 0.8, P(+ | healthy) = 0.1, so P(+) = 0.24. Four decimals.

| belief | P(fault) | block | release | action | risk |
|---|---:|---:|---:|---|---:|
| before the test | 0.2000 | 0.8000 | 2.0000 | block | 0.8000 |
| test positive | 0.6667 | 0.3333 | 6.6667 | block | 0.3333 |
| test negative | 0.0526 | 0.9474 | 0.5263 | release | 0.5263 |

The risk before the test is 0.80; after it, in expectation, 0.24 x 0.3333 + 0.76 x 0.5263 = 0.48. With a query cost of 0.05 the value of information is 0.80 - 0.48 - 0.05 = 0.27. If blocking a healthy device cost 4 instead of 1, the prior 0.2 would give block 3.2 against release 2.0: the action changes from block to release while the probability stays 0.2; the boundary moves from 1/11 = 0.0909 to 4/14 = 0.2857.
<!-- /worked -->

Bayes's rule gives the posterior $2/3$ after a positive test and $1/19$ after a negative one. The prior $0.2$ is above the threshold, so with no test the action is to block, at a risk of $0.80$. After a positive test it is still to block, at the lower risk of $0.3333$; after a negative one, release, at $0.5263$. Averaged over the two results the risk is $0.48$, and the test, which costs $0.05$, is worth $0.80 - 0.48 - 0.05 = 0.27$: the value of information, the drop in Bayes risk that evidence buys, less its price. The last sentence of the block moves the cost of blocking a healthy device from $1$ to $4$ and the same $0.2$ leads to the other action, because the threshold has moved from $1/11$ to $4/14$. The belief did not change, only what it was to be used for. That is the argument for keeping the probability and the action as separate functions, with the action a plain computation on a probability that has been checked, rather than a policy trained on outcomes.

The Bayes action need not be one of two. Add a third, deferring to a manual inspection at a fixed cost of $0.3$ whatever the truth. The rule is the same, the cheapest of three expected costs.

<!-- worked: calibrated_decisions.deferral_regions -->
Blocking a healthy device costs 1, releasing a faulty one 10, deferring (a manual inspection) 0.3 whatever the truth. Expected cost of each action at a probability p of a fault; four decimals; ties do not occur at these values.

| P(fault) | block | release | defer | cheapest |
|---|---:|---:|---:|---|
| 0.0100 | 0.9900 | 0.1000 | 0.3000 | release |
| 0.0200 | 0.9800 | 0.2000 | 0.3000 | release |
| 0.0500 | 0.9500 | 0.5000 | 0.3000 | defer |
| 0.0909 | 0.9091 | 0.9090 | 0.3000 | defer |
| 0.2000 | 0.8000 | 2.0000 | 0.3000 | defer |
| 0.5000 | 0.5000 | 5.0000 | 0.3000 | defer |
| 0.6500 | 0.3500 | 6.5000 | 0.3000 | defer |
| 0.7500 | 0.2500 | 7.5000 | 0.3000 | block |
| 0.9000 | 0.1000 | 9.0000 | 0.3000 | block |

Release is cheapest for p <= 0.3 / 10 = 0.03, block for p >= 1 - 0.3 / 1 = 0.70, deferring in between; the plain boundary 1/11 = 0.0909 lies inside the deferral band.
<!-- /worked -->

Deferring is cheapest in a band: below $p = 0.3/10 = 0.03$ release is cheaper than inspecting, above $1 - 0.3/1 = 0.7$ blocking is, and in between inspecting wins. The plain threshold $1/11$ lies inside the band, so a model that reports $0.0909$ gets the instruction to inspect, the honest answer when the two mistakes cost nearly the same in expectation and a cheap third option exists. Not answering is an action with a cost, and the section after next uses that.

### Choosing the threshold by accuracy, or by cost

It is common to choose the cut on a probability by maximising accuracy on the fit set. The accuracy-maximising cut belongs to the case where both mistakes cost the same: it is $0.5$ for a calibrated model, whatever the costs actually are. The next table compares the two ways of choosing, on the fresh set, with a miss costing $r$ times a false alarm for five values of $r$. The accuracy cut is searched on a grid of $0.01$ on the fit set; the cost threshold is $t^\star = 1/(1 + r)$ applied to the raw and to the recalibrated probabilities, and, as a floor no rule on this information can beat in expectation, to the true ones.

<!-- worked: calibrated_decisions.threshold_choice -->
Fresh set of `calibrated_decisions.decomposition` (20000 examples). A false alarm (block, label 0) costs 1 and a miss (release, label 1) costs r; block when the probability is at least the threshold. 'accuracy cut' is the cut c = k/100, k = 0 .. 100, that maximises the accuracy of the raw model on the fit set (the first of equals): 0.68 on the raw probabilities and 0.51 on the probabilities recalibrated by scale and shift on the fit set (the same fit as in `calibrated_decisions.decomposition`). t* = 1/(1 + r) is applied to the raw probabilities, to the recalibrated ones, and to the true probabilities (the least cost any rule on the true probabilities can have, in expectation). Cells are the mean cost per example, four decimals.

| r | t* | raw, accuracy cut | raw, t* | recalibrated, accuracy cut | recalibrated, t* | true probability, t* |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.5000 | 0.2494 | 0.2524 | 0.2498 | 0.2489 | 0.2493 |
| 2 | 0.3333 | 0.4139 | 0.3644 | 0.4185 | 0.3650 | 0.3615 |
| 5 | 0.1667 | 0.9076 | 0.5656 | 0.9244 | 0.5029 | 0.4986 |
| 10 | 0.0909 | 1.7304 | 0.7728 | 1.7677 | 0.5749 | 0.5754 |
| 20 | 0.0476 | 3.3758 | 1.0762 | 3.4542 | 0.6195 | 0.6081 |

For reference only, since it looks at the fresh labels: at r = 10 the cheapest cut on the grid 0.01 .. 0.99 is 0.01 for the raw probabilities (cost 0.5976) and 0.10 for the recalibrated ones (cost 0.5740).
<!-- /worked -->

At $r = 1$ the two ways coincide and every column lies between $0.249$ and $0.253$: the cost threshold is $0.5$. As the miss becomes dearer the accuracy cut stays where it is and its cost grows from $0.25$ to $3.45$ per example at $r = 20$, where the recalibrated probabilities at $t^\star = 0.0476$ pay $0.62$. The right-hand columns are the point of the chapter in numbers. At $r = 10$ the recalibrated probabilities with their cost threshold pay $0.5749$, the true probabilities $0.5754$ (the two agree to what $20000$ examples can resolve), and the raw probabilities with the same threshold pay $0.7728$, more than a third above. The raw model does worse at this threshold because its probabilities are wrong exactly where it looks: its lowest bin, which the reliability table showed has a frequency of $0.124$, straddles $1/11$, and a rule that releases what the model scores below $0.0909$ releases examples whose true chance of being positive averages above it. The reference line below the table shows the other route, which Volume II compared: searching the cut on labelled data. For the raw probabilities the cheapest cut on the grid is $0.01$, at $0.5976$, close to what the recalibrated route pays; it is found by looking at the labels again for every cost ratio, where the calibrated route looks at them once, to fit a map, and then needs only the costs.

![Mean cost per example on 20000 fresh examples as a function of the threshold, with a miss costing 10 and a false alarm 1. The curve for the recalibrated probabilities has its minimum, near 0.57, at a threshold close to the vertical line at 1/11. The curve for the raw probabilities is higher than it at 1/11 and keeps falling toward the smallest thresholds at the left edge.](img/cost-by-threshold.svg)

A calibrated probability also lets a forecaster quote the cost of its own decisions before they are made. The table gives, for each set of probabilities at $r = 10$, the cost the probabilities promise, the mean of $\min(1 - p,\ 10p)$, next to the cost the rule actually paid.

<!-- worked: calibrated_decisions.quoted_cost -->
Fresh set, cost of a miss 10 and of a false alarm 1, block when p >= 1/11. 'quoted' is the mean over the examples of min((1 - p), 10 p), the expected cost the forecaster's own probability promises; 'paid' is the mean cost actually incurred. Four decimals.

| probabilities used | quoted | paid | paid - quoted |
|---|---:|---:|---:|
| raw model | 0.2941 | 0.7728 | 0.4787 |
| recalibrated | 0.5557 | 0.5749 | 0.0192 |
| true probability | 0.5694 | 0.5754 | 0.0060 |
<!-- /worked -->

The raw model promises $0.2941$ and pays $0.7728$, a gap of $0.4787$; the recalibrated probabilities promise $0.5557$ and pay $0.5749$, and the true ones promise $0.5694$ and pay $0.5754$. The small gaps are what a finite sample and a slightly imperfect fit leave; the large one is a calibration error, and it costs money. This is the practical reason for caring about calibration. A ranker needs none, but whoever multiplies a probability by a price is quoting a number that is wrong by as much as the probability is.

## Answering only when sure

The deferral action is an instance of a wider idea. A classifier can decline to answer the cases it is least sure of, at a cost, so that its answers are right more often. The price is paid in coverage, the share of cases it still answers.

Take the classifier that answers $1$ when $p \ge 1/2$ and $0$ otherwise, and call its confidence $\max(p, 1 - p)$. Sort the examples by descending confidence, keeping the input order among equal confidences. The **risk-coverage curve** gives, for each $k = 1, \dots, n$, the coverage $k/n$ and the risk, the error rate of the $k$ most confident answers.

```text
function risk_coverage(conf: array<real>[n], err: array<int>[n])
        -> (coverage: array<real>[n], risk: array<real>[n]):
    require n > 0, "nothing to rank"
    order = argsort(conf, descending)       # ties keep input order
    coverage = zeros(real, [n])
    risk = zeros(real, [n])
    wrong = 0
    for k in 0 .. n:
        wrong = wrong + err[order[k]]
        coverage[k] = (k + 1) / n
        risk[k] = wrong / (k + 1)
    return (coverage, risk)
```

The mean of `risk` over $k$ is the area under the curve in the discrete form used here, a number to be minimised. A second summary asks directly whether confidence separates the wrong answers from the right ones: the share of (error, correct answer) pairs in which the error has the lower confidence, with a tie counting one half, the AUROC of Volume II for the task of detecting failures. Ties are a real hazard. Eight answers with confidences $0.9, 0.8, 0.8, 0.7, 0.6, 0.6, 0.55, 0.5$ and errors $0, 0, 1, 0, 1, 0, 1, 1$ have an area of $0.2807$ in the order given, and $0.3182$ if the two tied pairs are swapped; the failure AUROC of $0.8125$ counts the tied pairs one half and does not move. A published convention for the order is needed, and the one here is the order of the data.

<!-- worked: calibrated_decisions.selective_prediction -->
Fresh set (20000 examples). The answer is 1 (the event) when p >= 0.5, an error is an answer that differs from the label, and the confidence is max(p, 1 - p). Examples are sorted by descending confidence (equal confidences keep their order in the data) and the risk at coverage c is the error rate of the first 20000 c; four decimals.

| coverage | true probability | recalibrated | raw model |
|---|---:|---:|---:|
| 0.1 | 0.0390 | 0.0385 | 0.0370 |
| 0.2 | 0.0602 | 0.0592 | 0.0638 |
| 0.3 | 0.0832 | 0.0818 | 0.0852 |
| 0.4 | 0.1057 | 0.1061 | 0.1079 |
| 0.5 | 0.1265 | 0.1260 | 0.1299 |
| 0.6 | 0.1504 | 0.1500 | 0.1524 |
| 0.7 | 0.1734 | 0.1743 | 0.1784 |
| 0.8 | 0.1993 | 0.1991 | 0.2034 |
| 0.9 | 0.2240 | 0.2227 | 0.2297 |
| 1.0 | 0.2493 | 0.2489 | 0.2524 |

The area under the curve is the mean over k = 1 .. 20000 of the error rate of the k most confident examples; the failure AUROC is the share of (error, correct answer) pairs in which the error has the lower confidence, a tie counting one half. true probability: area under the curve 0.1283, failure AUROC 0.7180; recalibrated: area under the curve 0.1283, failure AUROC 0.7171; raw model: area under the curve 0.1312, failure AUROC 0.7146.
<!-- /worked -->

![Risk-coverage curves of the classifier that answers by the side of one half, ranking its answers by max(p, 1 - p), for the true probability, the recalibrated probability and the raw model, with two dashed references: a perfect ranking, which has no errors until coverage about 0.75, and no ranking, the flat overall error rate. The three model curves nearly coincide, rising from about 0.04 at a tenth of the examples to 0.25 at full coverage, far below the flat line and far above the perfect one.](img/risk-coverage.svg)

With the true probability the classifier is wrong $4\%$ of the time on its most confident tenth and $25\%$ of the time overall; answering only the top half of the cases cuts the error rate to $0.1265$. The recalibrated model is indistinguishable from the truth, and the raw model, with its areas $0.1312$ against $0.1283$, is slightly behind. Recalibration moves the ranking little, because a monotone map changes nothing about the order of $p$; it changes $\max(p, 1 - p)$ only through the shift, which moves the point where the two classes swap. What it changes a great deal is the *rule*.

The curve says how risk falls as cases are declined, not where to stop. For that a price is needed. Let a wrong answer cost $1$, a right one $0$, and abstaining $0.2$. Answering a case with probability $p$ of the event costs $\min(p, 1 - p)$ in expectation, so the Bayes action answers when $\min(p, 1 - p) \le 0.2$ and abstains otherwise. This is Chow's rule, published in 1970 for exactly this trade, and it is the three-action computation of the last section with the cost of deferral set to the cost of abstaining.

<!-- worked: calibrated_decisions.abstention_cost -->
Fresh set, recalibrated probabilities. A wrong answer costs 1, a right one 0, abstaining costs 0.2. Cost per example when the first 20000 c examples by descending confidence are answered and the rest are not. Four decimals.

| coverage | cost per example |
|---|---:|
| 0.1 | 0.1839 |
| 0.2 | 0.1719 |
| 0.3 | 0.1646 |
| 0.4 | 0.1625 |
| 0.5 | 0.1630 |
| 0.6 | 0.1700 |
| 0.7 | 0.1820 |
| 0.8 | 0.1993 |
| 0.9 | 0.2205 |
| 1.0 | 0.2489 |

The cheapest coverage on this grid is 0.4 at 0.1625. The rule that answers when min(p, 1 - p) <= 0.2 needs no ranking at all; applied to each set of probabilities it gives:

| probabilities | share answered | quoted cost | paid cost |
|---|---:|---:|---:|
| true probability | 0.4209 | 0.1608 | 0.1615 |
| recalibrated | 0.3476 | 0.1647 | 0.1626 |
| raw model | 0.7163 | 0.0886 | 0.1878 |

'Quoted' is the mean of min(min(p, 1 - p), 0.2), the cost the probabilities promise; 'paid' is the cost incurred.
<!-- /worked -->

On the recalibrated probabilities the cost is lowest, $0.1625$ per example, at a coverage of $0.4$ on the grid of tenths, and the probability-based rule, which needs no ranking and no grid, answers $34.8\%$ of the cases and pays $0.1626$. The true probabilities answer $42.1\%$ and pay $0.1615$. The raw model's probabilities make the rule answer $71.6\%$ of the cases: it believes itself much surer than it is, answers cases it should have declined and pays $0.1878$ where it quoted $0.0886$. The ranking of the raw model was fine; its probabilities were not, and the abstention rule runs on probabilities.

## Training against a proper score

A score that is proper for evaluation need not be a good loss for training, and the difference is visible in a single number.

Take one example with label $1$ and a model whose logit is $z$, so that $p = \sigma(z)$. The gradient of the cross-entropy $-\ln p$ with respect to $z$ is $p - 1$, the expression of the third recall question. The gradient of the one-class Brier score $(p - 1)^2$ is $2(p - 1)\,p(1 - p)$, by the chain rule through the sigmoid. Where the model is confidently wrong both are required to push hard.

<!-- worked: calibrated_decisions.saturated_gradients -->
One example with label 1 and a model whose logit is z, so that p = sigmoid(z). Loss and derivative with respect to z of the cross-entropy -ln p, which is p - 1, and of the one-class Brier score (p - 1)^2, which is 2 (p - 1) p (1 - p). The table prints p and the Brier derivative in scientific notation, the rest to four decimals.

| z | p | cross-entropy | Brier | d cross-entropy / dz | d Brier / dz |
|---|---:|---:|---:|---:|---:|
| 0 | 5.000e-01 | 0.6931 | 0.2500 | -0.5000 | -2.500e-01 |
| -2 | 1.192e-01 | 2.1269 | 0.7758 | -0.8808 | -1.850e-01 |
| -4 | 1.799e-02 | 4.0181 | 0.9644 | -0.9820 | -3.469e-02 |
| -6 | 2.473e-03 | 6.0025 | 0.9951 | -0.9975 | -4.921e-03 |
| -10 | 4.540e-05 | 10.0000 | 0.9999 | -1.0000 | -9.079e-05 |
| -15 | 3.059e-07 | 15.0000 | 1.0000 | -1.0000 | -6.118e-07 |
<!-- /worked -->

At a logit of $-6$, where the model gave the event probability $0.0025$, the cross-entropy pushes with a force of $0.9975$ and the Brier score with $0.0049$; at $-10$ the cross-entropy still pushes with $1$ and the Brier score with $0.00009$. The squared error is proper but flattens exactly where a confident mistake needs the largest correction, since the factor $p(1 - p)$ goes to zero. Cross-entropy, whose logarithm cancels that factor, keeps a constant force on the logit. This is the reason for the weighting in brain's calibration objective $L = (1 - \lambda)\,\mathrm{focal}(\gamma) + \lambda\,\mathrm{Brier}$: with $\gamma = 0$ and $\lambda = 0$ it is plain cross-entropy, the control, and the Brier term is added as a mixture with a weight that a sweep chooses. It is also the story of a bug found in the same file (see the next section), where a clamp inside the logarithm turned the cross-entropy's constant force into a decaying one.

A training target need not be a single label. If the world supplies an exact distribution $q$ over outcomes, as the device world does, the cross-entropy against it, $-\sum_i q_i \ln p_i$, is still minimised by $p = q$ and the squared distance $\sum_i (p_i - q_i)^2$ is the Brier score for a soft target. Three reports for the three posteriors of the device world show how the evaluation of such a model should look.

<!-- worked: calibrated_decisions.soft_scores -->
Three oracle distributions over (healthy, faulty): (0.8, 0.2), (1/3, 2/3), (18/19, 1/19), one report for each. Cells are the mean over the three of sum_i q_i (-ln p_i) (soft cross-entropy), sum_i q_i ln(q_i / p_i) (KL) and sum_i (p_i - q_i)^2, four decimals; the mean entropy of the oracle is 0.4477. 'Likelier label' is the mean -ln of the probability the report gives to the likelier outcome of the oracle (healthy, faulty, healthy), the score of a collapsed target. The reports given, one per oracle distribution in the order above: the oracle itself; hedged toward the middle (0.7, 0.3), (0.5, 0.5), (0.8, 0.2); collapsed to the likelier label (0.98, 0.02), (0.02, 0.98), (0.98, 0.02).

| report | likelier label | soft cross-entropy | KL | squared distance |
|---|---:|---:|---:|---:|
| the oracle itself | 0.2276 | 0.4477 | 0.0000 | 0.0000 |
| hedged toward the middle | 0.4243 | 0.5051 | 0.0574 | 0.0397 |
| collapsed to the likelier label | 0.0202 | 0.7804 | 0.3327 | 0.0878 |
<!-- /worked -->

The oracle's soft cross-entropy is $0.4477$, not zero, because the world has that much entropy and nothing can remove it; the divergence, which subtracts the entropy, is the quantity with a floor of zero, and a report hedged toward the middle is $0.0574$ above it and one collapsed to the likelier label $0.3327$. The first column shows what is lost by scoring against the likelier label of the world instead of its distribution: the collapsed report, the worst, scores $0.0202$ there and the oracle $0.2276$, so a model is rewarded for moving away from the truth.

## What brain's calibration crate computes, and where it falls short of what it says

The case study is `crates/rlcd`. It is a leaf crate (commit `18a8f541b`, 22 September 2026), created by lifting the focal-plus-squared-error objective of a decision crate out of the model layer so that any decision model can train against it, and generalising it from a single gold label to a full target distribution. Its modules are the chapter's sections in code. `crates/rlcd/src/scoring.rs` holds the objective of the previous section, `decision_loss_soft`, with a softmax computed in double precision together with its logarithm; the gradient is checked against finite differences and, where a finite-difference check cannot see (it cannot, at a saturated prediction), against the closed form $p - q$. `crates/rlcd/src/cost.rs` holds `CostMatrix`, `expected_cost`, `bayes_action`, `regret` and `voi`, and its tests are the device world of this chapter: the boundary $1/11$, the risks $0.8$ and $0.48$, the value of information $0.27$ and the flip when the false-alarm cost becomes $4$. `bayes_action` reports ties instead of hiding them behind the index order, with a tolerance of $10^{-6}$ (the module of this chapter takes the first of exactly equal costs, the convention that matters only on a tie). `crates/rlcd/src/metrics.rs` holds the reliability bins, ECE, an adaptive ECE over equal-mass bins, NLL, the Brier score, and for selective prediction `coverage_accuracy` and `failure_auroc`; the first returns the accuracy of the kept examples, whose complement is the risk of this chapter, and it breaks ties by input order with a stable sort, the convention above. Its Brier score sums over all the options, so that for two options it is twice the one-class form $(y - p)^2$ of this chapter and of `survival`'s binary score; a comparison of the two numbers has to divide by two.

`crates/rlcd/src/witness.rs` turns the decision machinery on a model. Its `search` takes a world whose exact posteriors are known and a learner, and finds every observation where the action implied by the learner's belief differs from the action implied by the oracle's, with the regret of the learner's action. A commit of 24 September 2026 (`49e518520`) records why this needs an error channel. The first caller had a prediction path that could fail, and with nowhere to put a failure it returned the vector $(1, 1)$ as a placeholder. A placeholder is not a distribution, but the Bayes action accepts any non-empty vector: under costs $(1, 10)$ the expected costs of $(1, 1)$ are $1$ and $10$, the action is to block, and that agrees with the oracle at two of the three observations of the world. A model that had never been asked anything passed two thirds of the audit. The fix makes the prediction fallible and the search propagate the failure, and the test keeps the placeholder as evidence that it clears two of three. The lesson is the one of the cost section run backwards: because actions depend on beliefs only through a comparison, agreement on actions is not evidence of agreement on probabilities, and a decision audit has to say what it could not ask.

On 24 September 2026 two more commits tightened the evaluation. Commit `07191ec88` is the story of Volume II, Chapter 6, from the other side: a calibration number that measured peakedness instead of reliability, and that scored an exact oracle worse than a trained model. Its repair added `soft_nll`, `posterior_kl`, `posterior_error` and `soft_ece`, which score against the example's own distribution. The soft-score table of this chapter is the arithmetic behind the repair: the oracle's score on the likelier label is the $0.228$ that the commit reports for an exact model, its soft cross-entropy is the world's own entropy of $0.448$, and its divergence is zero. The test `an_exact_oracle_scores_zero_on_every_soft_metric` makes that a property of the code. Commit `1bd342f7b` repaired the clamp described in Volume II, Chapter 3: `cross_entropy_keeps_its_gradient_when_a_prediction_saturates` compares the gradient with $p - q$ at logit gaps up to $400$.

`crates/rlcd/src/proper.rs` is the part that bears most directly on this chapter, and the part where the code says more than it can. Its module header reproduces the reward of a published decision model, made of a log score, a spherical score and an ordinal rank-probability term, and said, when this chapter was written: "All three terms are strictly proper, so the only way to maximize `R` is to report honest probabilities". The log term is floored at $\ln 10^{-4} = -9.21$ per option, and the header calls this floor "load-bearing": a bounded penalty, so that one hopeless sample does not dominate a group of samples. Both statements are true, and together they are not. A floored log score is flat below the floor, and a flat region is the failure that the side-of-one-half rule showed: a report inside it is not penalised for being worse. The table computes this with an independent implementation of the formula in the header, for an event whose true probability is $5 \cdot 10^{-5}$, below the floor.

<!-- worked: calibrated_decisions.floored_log -->
A binary event of probability p = 5e-5 (the honest report) and a reward that is the log score ln(max(r, 1e-12)) floored below at ln(1e-4) = -9.2103, r the probability the report gave the outcome; its expected value for the report (1 - q, q) is in the second column. The third column adds 0.5 times the spherical score (the dot product of the report with the one-hot outcome, divided by the length of the report). Six decimals; the last row scans q = k 1e-6 for k = 0 .. 2000 and keeps the first q whose reward exceeds every earlier one by more than 1e-15.

| report q | floored log reward | floored log + 0.5 spherical |
|---|---:|---:|
| 0 | -0.000461 | 0.499514 |
| 1e-05 | -0.000471 | 0.499504 |
| 5e-05 | -0.000511 | 0.499464 |
| 1e-04 | -0.000561 | 0.499414 |
| 2e-04 | -0.000626 | 0.499349 |
| 1e-03 | -0.001346 | 0.498629 |
| best q on the 1e-6 grid | 0.000000 | 0.000000 |

For the floored log reward alone, reporting 0 beats reporting the true probability p for every p below 2.72e-04 (found by 80 bisections of [1e-4, 1e-3], comparing the two rewards).
<!-- /worked -->

Reporting $0$ earns a higher expected reward than reporting the truth, with the log term alone and with the spherical term added at the weight $0.5$ of the header's default. The reversal holds for any true probability below about $2.7 \cdot 10^{-4}$ for the log term alone. The test in the crate, `reporting_the_true_distribution_maximizes_the_reward`, draws reports around the distribution $(0.5, 0.3, 0.15, 0.05)$, all of whose entries lie far above the floor, so it passes and cannot see this; the claim of the header is true for such targets and false for targets with options rarer than a few in ten thousand. The floor is a deliberate trade of exact propriety for a bounded penalty, and that is a reasonable trade as long as the header says so. (The check here is a computation in Python of the formula as written; the Rust function was not run on these inputs.)

There is a second floor in the same crate: `nll` and `soft_nll` clamp a probability at $10^{-12}$ before taking its logarithm, which bounds a log score at $27.6$ rather than at $9.21$. Two floors in one crate, for the same score, is the kind of detail that only matters when someone compares numbers across them.

Commit `cf5a4bbdd` now states the trade in that header: the terms are strictly proper above the floor, and below it a true probability under about $2.7 \cdot 10^{-4}$ earns more by a report of zero. The same commit settles three smaller points found in the source. In `scoring.rs` a sentence of the header began "squared-error probability score is a proper scoring rule" in lower case in the middle of a paragraph, a trace of a mechanical renaming (commit `d99086e24` replaced the technique's name with a description). In `cost.rs` the comment on `P_POSITIVE_RESULT` read "P(+) = 0.24*P(F|+) ... = 0.2*0.8 + 0.8*0.1", which was garbled: $P(+)$ is $0.24$ itself, and $0.24 \cdot P(F \mid +) = 0.16$ is the joint probability of a fault and a positive result. And the test `classwise_ece_catches_a_class_top1_ece_is_blind_to` in `metrics.rs` asserted only that the class-wise error is below $10^{-6}$ on an input where all classes happen to be calibrated; it now builds the case its name claims, a top label right seven times in ten at confidence $0.7$ with the rejected mass on a class that never occurs, and asserts that the class-wise error sees it. One point remains as it was: the message of commit `18a8f541b` calls "focal+Brier" a proper scoring rule, while the module header says the focal term is improper; for $\gamma > 0$ and $\lambda < 1$ the mixture is not proper either, since the focal part has a nonzero slope at the truth and the Brier part has none.

The crate contains no recalibrator; the scale-and-shift and isotonic maps of this chapter exist in its worked module and, for survival probabilities, in `crates/survival/src/recalibration.rs`, which Volume II, Chapter 6 described. What `rlcd` supplies is the part of the pipeline that comes after: the proper objective, the cost computation and the metrics against which a calibrated probability is judged.

## Check your understanding

1. A forecaster is paid the negative absolute error $-|y - q|$ and believes the chance of the event is $0.3$. Which report maximises the expected payment, and what is the payment for the honest report?
2. Find the error: "The model's expected calibration error is $0.01$ and the base-rate forecast's is $0.02$, so the model is better calibrated and therefore more useful."
3. A forecaster's reliability is $0.01$ and resolution $0.05$ on data with an event rate of $0.3$. What is the Brier score if forecasts take a few distinct values, and what changes if they vary inside the bins?
4. An isotonic map fitted on a test set gives a lower Brier score on it than the true probabilities would. Why is this evidence of a mistake and not of skill?
5. Costs: a false alarm costs $2$ and a miss costs $18$. At what probability does the Bayes action change from release to block, and what happens to that value if the cost of a miss doubles?
6. A model with a better risk-coverage curve than another is deployed with the abstention rule $\min(p, 1 - p) \le 0.2$ and does worse. Name two reasons that do not involve the ranking.
7. Why can the log score of a forecaster be infinite while its Brier score cannot exceed $1$, and what does the floor in the reward of `crates/rlcd/src/proper.rs` give up to avoid it?

## Summary

A proper scoring rule is one under which the best report is the belief. The squared error (the Brier score) and the logarithm are strictly proper: their expected losses are $(q - p)^2 + p(1 - p)$ and $H(p) + D_{\mathrm{KL}}(p \Vert q)$. The absolute error is not proper, the side-of-one-half rule that underlies accuracy is proper but not strictly, and the focal penalty is improper. Each proper rule is an average of the losses of users with different cost ratios, uniform for the Brier score and weighted by $1/(t(1 - t))$ for the log score, which is the link to decisions and to the CRPS and survival scores of the earlier chapters. The Brier score splits into reliability, resolution and uncertainty, and the expected calibration error is the unsquared reliability half only, so it can prefer a forecaster who says nothing. A model's probabilities can be repaired by a scale and a shift or by an isotonic map, fitted on data that the score does not use; fitted on the test set the isotonic map beat the true probability, which is impossible, and on fresh data it was the worst of the repairs. With calibrated probabilities an action is a computation: the Bayes action under a cost matrix, a threshold set by the costs, an extra option to defer, and a rule for when to abstain. On the generated data the cost threshold on recalibrated probabilities paid $0.5749$ per example against $1.7677$ for the accuracy cut at a miss ten times a false alarm, and the raw model that promised $0.2941$ paid $0.7728$.

What this leaves open is what the probabilities were of. Everything here took the probabilities of a model fitted to data that look like the test data. When the world shifts, the calibration shifts with it, and nothing in these tables warns in advance. The next chapter, the project of this volume, builds a forecaster whose uncertainty is checked by every tool of the volume, the quantile loss, the walk-forward evaluation and the scoring rules of this chapter, against exactly that.

## References

- Glenn W. Brier, "Verification of Forecasts Expressed in Terms of Probability", 1950.
- Irving J. Good, "Rational Decisions", 1952.
- Leonard J. Savage, "Elicitation of Personal Probabilities and Expectations", 1971.
- Allan H. Murphy, "A New Vector Partition of the Probability Score", 1973.
- Mark J. Schervish, "A General Method for Comparing Probability Assessors", 1989.
- Tilmann Gneiting and Adrian E. Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation", 2007.
- Miriam Ayer, H. D. Brunk, George M. Ewing, W. T. Reid and Edward Silverman, "An Empirical Distribution Function for Sampling with Incomplete Information", 1955.
- John C. Platt, "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods", 1999.
- Alexandru Niculescu-Mizil and Rich Caruana, "Predicting Good Probabilities with Supervised Learning", 2005.
- C. K. Chow, "On Optimum Recognition Error and Reject Tradeoff", 1970.
- Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger, "On Calibration of Modern Neural Networks", 2017.
- Charles Elkan, "The Foundations of Cost-Sensitive Learning", 2001.

## Exercises

1. **[Calculate]** An event has probability $p = 0.3$. Give the expected Brier loss and the expected log loss of the reports $q = 0.3$ and $q = 0.5$, and the report that minimises the expected absolute error with its value.
2. **[Calculate]** Eight forecasts: four of $0.25$ with outcomes $0, 0, 1, 0$ and four of $0.75$ with outcomes $1, 1, 0, 1$. Give the Brier score, the reliability, the resolution and the uncertainty. Then replace $0.25$ and $0.75$ by $0.2$ and $0.8$ and give the new Brier score.
3. **[Calculate]** Labels in score order are $1, 0, 1, 1, 0, 0, 1$. Run pool adjacent violators by hand and list the fitted values.
4. **[Calculate]** Three actions, ship, scrap and rework, with costs $c(\text{ship}, \text{bad}) = 40$, $c(\text{ship}, \text{good}) = 0$, $c(\text{scrap}, \text{bad}) = 0$, $c(\text{scrap}, \text{good}) = 8$, and $c(\text{rework}, \cdot) = 3$. Give the Bayes action at $P(\text{bad}) = 0.05, 0.1, 0.5, 0.9$ and the two probabilities where the action changes.
5. **[Calculate]** The eight answers of the selective-prediction paragraph (confidences $0.9, 0.8, 0.8, 0.7, 0.6, 0.6, 0.55, 0.5$, errors $0, 0, 1, 0, 1, 0, 1, 1$): give the risk after each prefix, the area under the curve and the failure AUROC.
6. **[Calculate]** A forecast of the outcomes $0, 1, 2$ with probabilities $0.2, 0.5, 0.3$ and an observed outcome $0$. Give the Brier score of each event "outcome at most $k$" and the CRPS as their sum.
7. **[Implement]** Reproduce `calibrated_decisions.report_game` and `calibrated_decisions.floored_log`: the expected loss of each rule at $q = 0, 0.1, \dots, 1$, the first-of-equals scan for the best report, and the floored reward at the six reports.
8. **[Implement]** Reproduce `calibrated_decisions.decomposition_by_hand`, `calibrated_decisions.decomposition` and `calibrated_decisions.reliability_table`: the generator and the draw order of the fit, test and fresh sets, the bins and their edge rule, and the four terms with the remainder.
9. **[Implement]** Reproduce `calibrated_decisions.recalibration`: the scale-and-shift fit by Newton's method, pool adjacent violators and `isotonic_at`, and the five fitted recalibrators scored on the test and fresh sets.
10. **[Implement]** Reproduce `calibrated_decisions.device_world`, `calibrated_decisions.deferral_regions`, `calibrated_decisions.threshold_choice` and `calibrated_decisions.quoted_cost`.
11. **[Implement]** Reproduce `calibrated_decisions.selective_prediction` and `calibrated_decisions.abstention_cost`, including the area under the curve, the failure AUROC and the Chow rule on the three sets of probabilities.
12. **[Investigate]** Repeat the recalibration table with a fit set of $200$ examples in place of $2000$ (the first $200$ of the same draws). Which recalibrator degrades most on the fresh set, and how many fresh examples does the isotonic map now give exactly $0$ or $1$? Explain what you see by the freedom of each map.

## Key terms

- **proper scoring rule**: a loss for probability reports under which the expected loss is smallest when the report equals the true probability, so that a forecaster minimising it has no reason to state anything but their belief; it is strictly proper when the truth is the only best report. <!-- concept: proper-scoring-rule -->
- **Brier score**: the mean squared difference between a forecast probability and the outcome coded as $0$ or $1$; strictly proper, with honest value $p(1 - p)$, and a sum of reliability, minus resolution, plus uncertainty. <!-- concept: brier-score -->
- **log score**: the negative logarithm of the probability the forecast gave to what happened, the cross-entropy loss; strictly proper, with honest value the entropy of the truth, and unbounded for a forecast that called the outcome impossible. <!-- concept: log-score -->
- **decision threshold**: the probability above which one action has a lower expected cost than another, the point where the two expected costs cross; for a false alarm costing $c_{\mathrm{FP}}$ and a miss $c_{\mathrm{FN}}$ it is $c_{\mathrm{FP}}/(c_{\mathrm{FP}} + c_{\mathrm{FN}})$ and it belongs to the costs, not to the model. <!-- concept: decision-threshold -->

## Answers

**Recall questions.**

1. $E[(Y - q)^2] = E[Y^2] - 2qE[Y] + q^2 = p - 2pq + q^2$, because $Y^2 = Y$ for a $0/1$ variable. Completing the square gives $(q - p)^2 + p(1 - p)$.
2. The cross-entropy of $q$ relative to $p$ is $H(p) + D_{\mathrm{KL}}(p \Vert q)$. The difference is the divergence, which is zero exactly when $q = p$.
3. $p - y$.
4. A model is calibrated when, among the cases given probability $p$, about a share $p$ are positive. A reliability diagram sorts the predictions into equal-width bins and plots each bin's mean prediction against its observed share. The threshold is $c_{\mathrm{FP}}/(c_{\mathrm{FP}} + c_{\mathrm{FN}})$.
5. The integral is over the observation axis $x$ of $(F(x) - \mathbf{1}_{x \ge y})^2$.
6. The mean of $(y - \hat p)^2$ over the units, where $y$ is the indicator that the unit failed by $t^\star$ (with weights when some are censored); a forecast of $0.5$ for everyone scores $0.25$.

**Check your understanding.**

1. The expected payment is $-(p(1 - q) + (1 - p)q) = -(0.3 + 0.4q)$, which is largest at $q = 0$. The honest report $0.3$ pays $-0.42$ and the report $0$ pays $-0.3$.
2. A calibration error says nothing about resolution. The base-rate forecast has none, so it can have a small error and a Brier score equal to the uncertainty. Compare the Brier scores, or the reliability and resolution separately; a model that is slightly less calibrated and far better at separating the cases is more useful.
3. $\mathrm{BS} = 0.01 - 0.05 + 0.3 \cdot 0.7 = 0.17$. If forecasts vary inside the bins the identity picks up the remainder, $\frac1n\sum(q_i - \bar q_k)^2 - \frac2n\sum(q_i - \bar q_k)(y_i - \bar y_k)$, and the terms alone no longer add to the score.
4. No forecast can beat the true probabilities on average. The map has as many free values as the number of runs in the sorted labels and bent itself around the noise of these labels; on new data it is worse.
5. The threshold is $2/20 = 0.1$. Doubling the cost of a miss makes it $2/38 = 0.0526$: the cheaper the false alarm is relative to the miss, the earlier the action.
6. The probabilities are miscalibrated, so the threshold $0.2$ on $\min(p, 1 - p)$ answers the wrong share of the cases (the raw model above answers $71.6\%$ of the cases where $42.1\%$ is right, the share the true probabilities answer; the recalibrated ones answer $34.8\%$), and the model is judged on a population whose error rate differs from the one on which it was fitted.
7. The log score is $-\ln$ of the probability given to what happened, which is unbounded as that probability goes to zero; the Brier score is bounded by $1$. The floor bounds the penalty at $9.21$ per option and gives up strict propriety for targets rarer than a few in ten thousand.

**Exercises.**

1. At $q = 0.3$: Brier $0.21$, log $0.6109$. At $q = 0.5$: Brier $0.25$ ($0.21 + 0.2^2$), log $\ln 2 = 0.6931$. The absolute error is $0.3 + 0.4q$, minimised at $q = 0$ with the value $0.3$.
2. Brier $0.1875$, reliability $0$, resolution $0.0625$, uncertainty $0.25$. With $0.2$ and $0.8$ the reliability is $0.05^2 = 0.0025$ and the Brier score $0.19$.
3. $0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1$. The blocks after the fourth label are $(1,0) = 0.5$, $1$, $1$; the fifth label merges with the third block to give $0.5$, then with the second to give $2/3$; the sixth merges again to $0.5$, which equals the first block, and the strict comparison leaves them apart; the seventh stands alone.
4. Ship costs $40p$, scrap $8(1 - p)$, rework $3$. At $0.05$: $2, 7.6, 3$, ship. At $0.1$: $4, 7.2, 3$, rework. At $0.5$: $20, 4, 3$, rework. At $0.9$: $36, 0.8, 3$, scrap. The action changes at $p = 3/40 = 0.075$ (ship to rework) and $p = 1 - 3/8 = 0.625$ (rework to scrap).
5. Risks after the prefixes in the order given: $0, 0, 0.3333, 0.25, 0.4, 0.3333, 0.4286, 0.5$; the area is $0.2807$. The failure AUROC is $13/16 = 0.8125$: the four errors at $0.8, 0.6, 0.55, 0.5$ against the four correct answers at $0.9, 0.8, 0.7, 0.6$, with two ties counted one half. With the tied pairs swapped the area is $0.3182$.
6. The cumulative probabilities are $0.2$ and $0.7$. The events "at most $0$" and "at most $1$" both happened, so the Brier scores are $(0.2 - 1)^2 = 0.64$ and $(0.7 - 1)^2 = 0.09$, and the CRPS is $0.73$.
7. to 11. The tables of the blocks, to the printed digits. Values to check: the expected Brier loss at $q = 0.5$ for $p = 0.7$ is $0.25$ and the focal minimiser is $0.578$; the bins of the raw model and its error $0.1286$; the isotonic fit on the test set scores $0.1586$ on it and $0.1704$ on fresh data; the risk $0.80$ before the test and $0.48$ after it, with the value of information $0.27$; the cost at $r = 10$ of the recalibrated probabilities at $t^\star$, $0.5749$.
12. The fit-set size changes every number of the recalibration table, so the figures come from your run; the pattern to find is that the two-parameter map moves little and the isotonic map, with its many free values, degrades the most and gives exact $0$ or $1$ to many more of the fresh examples.
