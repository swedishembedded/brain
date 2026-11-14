<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Decision Models in Practice

*How is a small learned model used to make fast, auditable decisions inside an application?*

Three weeks after a payments desk starts routing its requests automatically, a customer writes to ask why her message about a stolen card sat in the accounts queue for two days. The honest answer to "was the router wrong?" is yes, and it is the least useful part of the reply. What the desk needs to say is what the router saw (which words, which numbers), what it believed (how sure it was of each of the four teams), what rule turned that belief into "send it to accounts" rather than "ask a person", which version of the model said so, and what it cost to ask. A decision model inside an application is judged by those five answers before it is judged by its accuracy, and a model that cannot give them is a model nobody can fix.

The smallest model that can give them is small enough to build in full and to count to the last operation. The shop of Chapter 1 had a state, an action and a reward that came back later; here the state is a request, the action is one of a fixed handful of routes, and the right answer is known soon enough to be a label. That makes the policy of Chapter 1 a classifier, and the question is what has to be built around a classifier before it may act on its own: a way to turn a request into numbers, a head that turns the numbers into probabilities, a calibration of those probabilities, a rule that prices them, a log that lets a decision be replayed, and a count of the arithmetic.

Each piece of that list rests on an idea met earlier, and six of them are worth bringing back first.

1. (Volume V, Chapter 14, "Bidirectional Encoders and Embedding Models", sentence embedding.) Two word vectors are $(3, 0)$ and $(0, 1)$. What is the sentence embedding by mean pooling, scaled to length 1?
2. (Chapter 1, "Markov Decision Processes", policy.) A policy gives the preferences $(2, 0)$ to two actions through a softmax. What probabilities does it assign, and which action does the greedy policy take?
3. (Volume II, Chapter 6, "Measuring Models: Metrics and Calibration", calibration.) Forty requests have a largest probability between $0.8$ and $0.9$ with mean $0.85$, and $30$ of the forty are routed right. By how much is the model over- or under-confident in that bin?
4. (Volume II, Chapter 6, temperature scaling.) The logits $(2, 0)$ are divided by $T = 2$. What are the probabilities before and after?
5. (Volume XI, Chapter 6, "Scoring Rules and Calibrated Decisions", abstention.) A wrong answer costs $1$, a right one $0$ and abstaining $0.2$. For which largest probability does answering become the cheaper action?
6. (Volume I, Chapter 13, "Information, Entropy and Surprise", entropy.) What is the entropy in nats of the distribution $(0.5, 0.25, 0.25)$, and what fraction of $\ln 3$ is it?

## Ten numbers that stand for a request

A decision model needs a request as numbers, and the part that makes them is the **state encoder**: a fixed function that turns the application's situation into a vector the rest of the model can read, built from whatever the application knows, free text and structured fields alike. The word "state" is the one of Chapter 1: it is what the decision is made from. The encoder is frozen while the decision is learned, so everything it computes can be computed once and kept.

The desk of this chapter has four teams to route to (card, payment, account, dispute) and a request made of five words and four fields: the amount (as its natural logarithm), the age of the account in months, whether the request came from a new device and the number of contacts in the last thirty days. The world is synthetic, so that every number below can be reproduced and the true route of every request is known. The words of a team's family are synonyms of one another, a request mentions its own team's words about two times in three, now and then the words of a neighbouring team (card with account, payment with dispute) and two fillers always. The block that follows gives the rule that makes it, with the seed, the draw order and the splits.

<!-- worked: decision_models.world_rule -->
The generator is `decision_models.world` (seed 0xd7d40b3dcb4483b2, 15552067773248340914). A normal() call that finds a held value returns it without drawing, and a held value left at the end of a series is carried into the next one (the generator is never reset). One generator runs across the whole world, in this order. First the four concept vectors, 6 `normal()` calls each, in route order. Then the word vectors: for each family in route order and each of its four words in the listed order, coordinate j is the concept coordinate plus 0.5 times one `normal()`; then the four filler words, each coordinate 0.5 times one `normal()`. Then the splits, train, validation, test and log, one request after another. One request draws, in this order: `u = uniform()` for the route (the first route whose running prior sum 0.35, 0.65, 0.85, 1.0 exceeds `u`); for each of three word slots `v = uniform()` (own family if `v < 0.65`, the partner family 2, 3, 0, 1 of the routes 0, 1, 2, 3 if `v < 0.85`, a filler otherwise) and then `integer(4)` for the word of that family; for each of two filler slots `integer(4)`; then `normal()` for the amount, `integer(60)` for the age, `uniform()` for the new-device flag (1 if below 0.1, 0.2, 0.15, 0.6 for the routes 0 to 3), `integer(3)` for the contacts, and for route 3 a second `integer(3)` added to them. ln amount is 3.0, 3.6, 2.6, 4.4 for the routes 0 to 3 plus 0.8 times the `normal()`; the age is `1 + integer(60)` months.

| route | family words | concept vector (3 decimals) |
|---|---|---|
| card | card, lost, stolen, replacement | 0.380, -0.623, -0.973, 0.503, -1.257, 0.074 |
| payment | payment, declined, failed, charge | -0.141, -1.704, 0.734, 0.362, 2.003, 0.518 |
| account | account, address, password, login | 0.687, -1.135, 0.069, -0.225, 1.338, 1.647 |
| dispute | refund, dispute, chargeback, unauthorised | -0.559, 0.215, 1.980, -1.264, 0.247, 0.863 |

The filler words are please, help, my, today. The first five training requests (the numbers are ln amount, age, new device, contacts):

| request | route | words | numbers |
|---:|---|---|---|
| 0 | card | account lost replacement please please | 3.64, 52.00, 0.00, 0.00 |
| 1 | account | replacement account stolen my help | 3.88, 37.00, 0.00, 0.00 |
| 2 | card | address replacement card my my | 3.23, 60.00, 1.00, 2.00 |
| 3 | payment | failed payment charge help please | 3.09, 52.00, 1.00, 1.00 |
| 4 | payment | declined failed please please please | 3.23, 2.00, 0.00, 1.00 |

Requests per route in each split:

| split | requests | card | payment | account | dispute |
|---|---:|---:|---:|---:|---:|
| train | 120 | 29 | 42 | 33 | 16 |
| validation | 500 | 166 | 173 | 106 | 55 |
| test | 1000 | 339 | 303 | 194 | 164 |
| log | 2000 | 707 | 600 | 399 | 294 |
<!-- /worked -->

A request's words become one vector in the way Volume V, Chapter 14 taught: a vector per word, their mean, scaled to length 1. The word vectors here are a frozen stand-in for a trained encoder, with the one property that matters for the argument, namely that synonyms are close because they share a concept vector, so that a request that says "stolen" and one that says "replacement" point the same way. The numbers are scaled by the mean and standard deviation of the training requests, so that an amount in the hundreds and a flag that is 0 or 1 reach the head on the same scale. Every convention is in the block: the sums run left to right, the length is the Euclidean one, the standard deviation has the divisor $n = 120$, and the state vector is the six numbers of the sentence followed by the four scaled fields.

<!-- worked: decision_models.state_vector -->
Request 0 of the training split: `account lost replacement please please`.

| word | v0 | v1 | v2 | v3 | v4 | v5 |
|---|---:|---:|---:|---:|---:|---:|
| account | 0.123 | -1.523 | -0.190 | -0.143 | 0.904 | 2.121 |
| lost | 0.363 | -1.268 | -0.442 | 0.663 | -1.145 | -0.405 |
| replacement | 0.451 | -1.491 | -1.312 | 0.565 | -1.346 | -0.817 |
| please | -0.162 | -0.380 | 0.079 | 0.353 | -0.403 | 0.011 |
| please | -0.162 | -0.380 | 0.079 | 0.353 | -0.403 | 0.011 |
| mean of the five | 0.1223 | -1.0083 | -0.3574 | 0.3583 | -0.4786 | 0.1842 |
| divided by its length 1.2453 | 0.0982 | -0.8097 | -0.2870 | 0.2877 | -0.3843 | 0.1479 |

The four numbers are scaled by the mean and the standard deviation (divisor n = 120) of the training requests:

| field | value | training mean | training sd | scaled |
|---|---:|---:|---:|---:|
| ln amount | 3.6424 | 3.3783 | 0.9051 | 0.2918 |
| account age (months) | 52.0000 | 28.7833 | 18.5374 | 1.2524 |
| new device | 0.0000 | 0.2167 | 0.4120 | -0.5259 |
| contacts in 30 days | 0.0000 | 0.9917 | 0.9354 | -1.0602 |

The state vector is the six numbers of the last line of the first table followed by the four scaled numbers: (0.0982, -0.8097, -0.2870, 0.2877, -0.3843, 0.1479, 0.2918, 1.2524, -0.5259, -1.0602).
<!-- /worked -->

The routine below is the whole encoder. `vec` is the table of word vectors, `mean` and `sd` the training statistics of the four fields.

```text
function encode_state(words: list<text>, raw: array<real>[4], vec: map<text, array<real>[6]>,
                      mean: array<real>[4], sd: array<real>[4]) -> array<real>[10]:
    require len(words) > 0, "a request has at least one word"
    s = zeros(real, [6])
    for w in words:                           # left to right
        for j in 0 .. 6:
            s[j] = s[j] + vec[w][j]
    for j in 0 .. 6:
        s[j] = s[j] / len(words)
    n = norm(s)
    require n > 0.0, "a sentence vector of length zero has no direction"
    x = zeros(real, [10])
    for j in 0 .. 6:
        x[j] = s[j] / n
    for j in 0 .. 4:
        x[6 + j] = (raw[j] - mean[j]) / sd[j]
    return x
```

Two conventions in it are choices a later reader could silently reverse, so they are stated here: the embedding is not centred, and it is scaled to length 1 before the fields are appended, which makes the six text coordinates small (their root mean square is $1/\sqrt{6} \approx 0.41$) against fields of unit spread. The head will compensate with larger weights, and the weights printed below show it. Scaling the text up by $\sqrt{6}$ would make the two halves comparable at the cost of one more convention; the encoder keeps the simpler rule.

## Four routes and a softmax

The part that decides is the **decision head**: a small trained map from the state vector to a score for each action of a fixed set, whose softmax is the probability of each. Here it is as small as it can be, one weight per input per route. With the state vector $\mathbf{x}(s)$ of ten numbers, a weight matrix $W \in \mathbb{R}^{4 \times 10}$ and a bias $\mathbf{b} \in \mathbb{R}^4$, the logits and the probabilities are

$$\mathbf{z} = W\mathbf{x} + \mathbf{b}, \qquad \pi_\theta(k \mid s) = \mathrm{softmax}(\mathbf{z})_k,$$

with $\theta$ the 44 numbers of $W$ and $\mathbf{b}$. This is the tabular softmax policy of Chapter 3 with the table replaced by a linear function of the state, and the word "policy" is meant: the head is a policy over four actions, and the greedy action is its argmax (the first index on a tie). What makes it a classifier rather than a reinforcement learner is where the training signal comes from. Every training request carries the route that was right, so the loss is the cross-entropy of Volume III against that route, and its gradient with respect to the logits is $\mathbf{p} - \mathrm{onehot}(y)$: the probability vector minus a one in the place of the right route. The gradient of the weights is that vector times the input, $(\mathbf{p} - \mathrm{onehot}(y))\,\mathbf{x}^\top$, and of the bias the vector itself. Every training routine that follows uses these two lines and nothing else.

```text
function train_head(X: array<real>[n, d], y: array<int>[n], W0: array<real>[4, d],
                    steps: int, rate: real) -> (array<real>[4, d], array<real>[4]):
    require n > 0, "no training requests"
    W = W0
    b = zeros(real, [4])
    for step in 0 .. steps:
        gW = zeros(real, [4, d])
        gb = zeros(real, [4])
        for i in 0 .. n:                      # the requests in order
            z = zeros(real, [4])
            for k in 0 .. 4:
                s = 0.0
                for j in 0 .. d:              # increasing j from 0.0, then the bias
                    s = s + W[k, j] * X[i, j]
                z[k] = s + b[k]
            p = softmax(z)
            for k in 0 .. 4:
                e = p[k] - (1.0 if k == y[i] else 0.0)
                gb[k] = gb[k] + e
                for j in 0 .. d:
                    gW[k, j] = gW[k, j] + e * X[i, j]
        for k in 0 .. 4:
            b[k] = b[k] - rate * gb[k] / n
            for j in 0 .. d:
                W[k, j] = W[k, j] - rate * gW[k, j] / n
    return (W, b)
```

The pseudocode and the block below agree line for line: full-batch descent with no momentum and no penalty on the weights, the loss being the mean of the cross-entropies, and 1500 steps at the step size $0.5$. Three heads are trained, one on the text half of the state, one on the fields and one on both, each from its own initial draw; the block prints the generator and seed of each, the 44 initial numbers of the head on both halves and the 44 it ends with.

<!-- worked: decision_models.head_parameters -->
Three heads, three initial draws (`W` row by row, route by route, `0.1 * normal()` per entry, bias 0):

| head | generator | inputs |
|---|---|---:|
| text | `decision_models.init.text` (seed 0xaa5c2e85c269facf, 12275737836332382927) | 6 |
| numbers | `decision_models.init.numbers` (seed 0x34f58b00ff490029, 3816109095665598505) | 4 |
| both | `decision_models.init.both` (seed 0x7e6873a64f0ea6dd, 9108657404484953821) | 10 |

Initial parameters of the head on the whole state (44 numbers; the draws of the first row are the first six of the first table, then the four of the second):

| route | u0 | u1 | u2 | u3 | u4 | u5 | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| card | -0.0607 | 0.0635 | 0.1604 | -0.0151 | 0.1562 | -0.0217 | 0 |
| payment | 0.0021 | 0.0239 | 0.0789 | 0.0083 | -0.0380 | 0.0082 | 0 |
| account | -0.2807 | 0.3030 | -0.0059 | 0.0285 | -0.0008 | -0.0195 | 0 |
| dispute | 0.0586 | -0.0293 | 0.1477 | -0.0163 | 0.0546 | 0.0341 | 0 |

| route | amount | age | device | contacts |
|---|---:|---:|---:|---:|
| card | -0.1481 | 0.1038 | 0.1113 | 0.0908 |
| payment | -0.0296 | 0.2343 | 0.0687 | 0.0086 |
| account | -0.0966 | -0.0479 | -0.0098 | -0.0270 |
| dispute | 0.1428 | -0.0574 | 0.0337 | 0.0238 |

Training: 1500 steps of full-batch gradient descent with step size 0.5 on the mean cross-entropy of the 120 training requests (the gradient of a row is `p - onehot(y)` times the input, summed over the rows in order and divided by 120, the bias gradient is the same without the input). Mean training cross-entropy at some steps (step 0 is the initial head):

| step | loss |
|---:|---:|
| 0 | 1.4169 |
| 10 | 0.9142 |
| 100 | 0.4156 |
| 500 | 0.2895 |
| 1500 | 0.2553 |

Parameters after step 1500. Digits are printed only where the run and its four copies with the first initial weight changed by +1e-12, -1e-12, +3e-12 and -3e-12 agree; the objective is convex, so a copy is pulled back to the same path and all digits agree:

| route | u0 | u1 | u2 | u3 | u4 | u5 | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| card | 1.0816 | 0.3669 | -4.6301 | 2.1314 | -2.9378 | 0.1845 | 0.8657 |
| payment | -1.9334 | -3.0134 | 3.0601 | -0.7039 | 4.0236 | -4.0998 | -0.4897 |
| account | 1.8470 | 2.5343 | -1.8934 | 1.9166 | -0.0224 | 5.3772 | 0.0744 |
| dispute | -1.2759 | 0.4732 | 3.8445 | -3.3388 | -0.8913 | -1.4608 | -0.4504 |

| route | amount | age | device | contacts |
|---|---:|---:|---:|---:|
| card | -0.0248 | 0.0251 | -0.3042 | -0.0455 |
| payment | -0.3849 | 0.1835 | -0.1915 | -0.5715 |
| account | -0.8843 | 0.0690 | 0.2341 | -0.2003 |
| dispute | 1.1624 | -0.0448 | 0.4656 | 0.9136 |
<!-- /worked -->

Two facts about the table deserve a sentence. The objective is convex in $(W, \mathbf{b})$ (the cross-entropy of a softmax of a linear function is convex, the multiclass form of what Volume II, Chapter 3 showed for the logistic), so descent from a start that differs by $10^{-12}$ in one weight ends at the same numbers to every digit shown, and the digits of the table are exactly the ones another implementation will reproduce. And the large weights on the text coordinates are the scale remark above made visible: a unit-length vector in six dimensions moves a logit by about its weight times $0.4$, so the head needs weights near 4 to separate routes that differ by a few units of logit.

## Intent classification, and what each half of the state buys

Deciding which of a fixed list of intents a message expresses is **intent classification**, and it is the canonical use of a decision model: a request in natural language, a closed set of things the application can do about it, and a label that someone, soon after, can supply. The desk's four routes are four intents. What the head does with them is measured on requests it never trained on, in the way Volume II demanded: accuracy against a baseline that needs no model, and a loss that scores the probabilities and not only the argmax.

<!-- worked: decision_models.intent_heads -->
Accuracy is the share of requests whose argmax (the first of equal probabilities) is the true route; cross-entropy is the mean of -ln of the probability of the true route; four decimals, raw probabilities (temperature 1).

| head | accuracy, train | accuracy, validation | accuracy, test | cross-entropy, train | cross-entropy, test |
|---|---:|---:|---:|---:|---:|
| text | 0.8500 | 0.8300 | 0.8060 | 0.3622 | 0.4374 |
| numbers | 0.4667 | 0.4440 | 0.4560 | 1.0759 | 1.1458 |
| both | 0.9083 | 0.8500 | 0.8380 | 0.2553 | 0.4089 |

The majority route of the training split is payment; always answering it gives 0.3030 on the test split, and a uniform guess is right with probability 0.25.

Confusion matrix of the head on the whole state, test split (rows true route, columns predicted route, then the share of the row that is on the diagonal):

| true \ predicted | card | payment | account | dispute | row recall |
|---|---:|---:|---:|---:|---:|
| card | 253 | 0 | 86 | 0 | 0.7463 |
| payment | 0 | 274 | 0 | 29 | 0.9043 |
| account | 23 | 0 | 170 | 1 | 0.8763 |
| dispute | 0 | 22 | 1 | 141 | 0.8598 |

The counts are the same in all five runs.
<!-- /worked -->

The head on both halves is right on $0.8380$ of the 1000 test requests, against $0.3030$ for always answering the training split's most common route and $0.25$ for a uniform guess. The training accuracy is $0.9083$ and the validation accuracy $0.8500$: with 120 training requests and 44 parameters the head has fitted a little of the noise, and the cross-entropy shows it ($0.2553$ on the training requests, $0.4089$ on the test requests). The ablation says what the state encoder is for. The head on the text alone reaches $0.8060$, the head on the four fields alone only $0.4560$, but the combination gains $0.032$ over the text, and the confusion matrix says where: the fields are what tell a dispute from a payment, since a dispute comes with a large amount, a new device and repeated contacts, which no word says. The matrix also shows the rest of the error to be the confusable pairs by construction. Of 339 card requests 86 go to accounts, and of 164 dispute requests 22 go to payment; no card request is sent to payment or dispute, because those families never leak into card requests.

A state encoder that includes structured fields is the usual shape of a decision model inside an application: the text is one input among the facts the application already holds, and the head is where they are weighed against one another.

## A probability that is a little too sure

A router that only needs the argmax would stop here. A router that is allowed to say "I am not sure, ask a person" needs the probabilities to mean what they say, and Volume II, Chapter 6 gave the test: among the requests given a largest probability near $0.9$, about nine in ten should be routed right. The head was fitted by descent on 120 requests until its training loss was $0.2553$; on requests it has not seen it should be a little less sure than that, and it is not. The largest probability averages $0.8849$ on the test requests while the accuracy is $0.8380$.

The remedy is the single number of Volume II, Chapter 6, now applied to a vector of four logits. Divide every logit by a temperature $T > 0$ before the softmax. That cannot change which logit is largest, so the argmax and the accuracy are untouched; it can only change how peaked the distribution is. The temperature is fitted on the validation requests, not the training ones, for the reason that chapter gave: on the training set the head is overconfident by construction. With $a = 1/T$ the mean cross-entropy of the validation requests is $f(a) = \frac{1}{N}\sum_i [\mathrm{lse}(a\,\mathbf{z}_i) - a z_{i,y_i}]$, a convex function of $a$ whose slope is

$$f'(a) = \frac{1}{N}\sum_i \Bigl(\sum_k \mathrm{softmax}(a\,\mathbf{z}_i)_k\, z_{i,k} - z_{i,y_i}\Bigr),$$

the average over requests of the probability-weighted mean logit minus the logit of the right route. The minimum is where the slope crosses zero, and the bisection of Volume II finds it.

```text
function fit_temperature(Z: array<real>[N, 4], y: array<int>[N]) -> real:
    require N > 0, "no validation requests"
    require slope(0.0, Z, y) < 0.0, "the logits do not favour the right routes on average"
    lo = 0.0
    hi = 1.0
    found = false
    for k in 0 .. 60:                         # double the upper end until the slope is positive
        if slope(hi, Z, y) > 0.0:
            found = true
            break
        lo = hi
        hi = 2.0 * hi
    require found, "no finite scale: the logits separate the labels"
    for k in 0 .. 80:                         # halve the bracket
        mid = (lo + hi) / 2.0
        if slope(mid, Z, y) < 0.0:
            lo = mid
        else:
            hi = mid
    return 1.0 / ((lo + hi) / 2.0)            # T = 1 / a
```

Here `slope(a, Z, y)` is $f'(a)$ above, the requests summed in order and divided by $N$.

<!-- worked: decision_models.calibration -->
The temperature is fitted on the 500 validation requests by minimizing the mean cross-entropy of softmax(z / T): T = 1.1762; on the validation split the mean cross-entropy goes from 0.4068 to 0.3998. The scores below are on the 1000 test requests (natural logarithms; Brier is the mean over requests of the sum over the four routes of the squared difference to the one-hot label; ECE takes the pair (largest probability, whether the argmax is right) through 10 equal-width bins, bin i holding i/10 <= p < (i+1)/10 and 1.0 in the last, the count-weighted mean of the absolute gap, as in Volume XI, Chapter 6; 'mean p_max' is the mean of the largest probability, to be compared with the accuracy).

| probabilities | cross-entropy | Brier | ECE | mean p_max | accuracy |
|---|---:|---:|---:|---:|---:|
| raw, T = 1 | 0.4089 | 0.2381 | 0.0484 | 0.8849 | 0.8380 |
| scaled, T fitted | 0.4000 | 0.2335 | 0.0237 | 0.8586 | 0.8380 |

Dividing every logit by one number cannot change which logit is largest, so the accuracy is the same in both rows.
<!-- /worked -->

The fitted temperature is above 1, which is the sign of an overconfident head, and the effect is the one Volume II, Chapter 6 predicted. The expected calibration error falls from $0.0484$ to $0.0237$, the mean largest probability from $0.8849$ to $0.8586$, which is now close to the accuracy $0.8380$, and the cross-entropy and the Brier score both improve slightly ($0.4089$ to $0.4000$ and $0.2381$ to $0.2335$). The gain is real and small. One number fitted on 500 requests removes about half the miscalibration of 44 parameters fitted on 120, and it leaves the other half, which the bins show.

<!-- worked: decision_models.reliability_bins -->
Test split, unnudged run (counts and means of a bin are what the figure plots); bin = range of the largest probability; 'accuracy' is the share of the bin's requests whose argmax is right; three decimals.

| bin | raw count | raw mean p_max | raw accuracy | scaled count | scaled mean p_max | scaled accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 to 0.1 | 0 | none | none | 0 | none | none |
| 0.1 to 0.2 | 0 | none | none | 0 | none | none |
| 0.2 to 0.3 | 0 | none | none | 0 | none | none |
| 0.3 to 0.4 | 1 | 0.386 | 1.000 | 1 | 0.367 | 1.000 |
| 0.4 to 0.5 | 10 | 0.485 | 0.500 | 21 | 0.481 | 0.524 |
| 0.5 to 0.6 | 62 | 0.545 | 0.532 | 63 | 0.544 | 0.540 |
| 0.6 to 0.7 | 53 | 0.652 | 0.528 | 81 | 0.656 | 0.593 |
| 0.7 to 0.8 | 89 | 0.748 | 0.719 | 94 | 0.751 | 0.691 |
| 0.8 to 0.9 | 136 | 0.857 | 0.713 | 189 | 0.856 | 0.815 |
| 0.9 to 1.0 | 649 | 0.968 | 0.940 | 551 | 0.959 | 0.953 |
<!-- /worked -->

![Reliability diagram of the decision head on the test requests: for each bin of the largest probability with at least 20 requests, accuracy against mean largest probability, for the raw and the temperature-scaled logits, with the dashed diagonal of perfect calibration. The raw curve lies below the diagonal; the scaled curve lies closer to it in every bin but the one near 0.75.](img/head-reliability.svg)

The raw head is worst in the middle. Its bin from $0.6$ to $0.7$ has a mean largest probability of $0.652$ and an accuracy of $0.528$, and its bin from $0.8$ to $0.9$ has $0.857$ and $0.713$. The scaled head's bin from $0.8$ to $0.9$ has $0.856$ and $0.815$. The bin from $0.7$ to $0.8$ is the one place where scaling moves the curve away from the diagonal (a gap of $0.029$ raw against $0.060$ scaled), which a test split of 1000 requests cannot be expected to smooth out. In the top bin, which holds $649$ raw requests and $551$ scaled ones, both are close ($0.968$ against $0.940$, and $0.959$ against $0.953$). A request that moved from the top bin to the one below it is exactly a request the temperature has made the rule less willing to act on, and that is the use of the calibration.

## Pricing the probability

A calibrated probability is not yet a decision. Volume XI, Chapter 6 separated the two: the probability says how likely each route is to be the right one, a cost matrix says what each mistake costs, and the action that minimizes the expected cost is a computation on the two. The desk has five actions, the four routes and an escalation to a person, and the simplest cost matrix with that shape has three numbers. A right route costs $0$, a wrong route costs $c_{\mathrm{w}}$, and escalating costs $c_{\mathrm{e}}$ whatever the truth, because a person reads the request either way. The expected cost of routing to team $k$ is $(1 - p_k)\,c_{\mathrm{w}}$, which is smallest for the most probable team, so the only question is whether to route at all:

$$\text{route to } \arg\max_k p_k \iff (1 - p_{\max})\,c_{\mathrm{w}} \le c_{\mathrm{e}} \iff p_{\max} \ge 1 - \frac{c_{\mathrm{e}}}{c_{\mathrm{w}}}.$$

This is Chow's rule of Volume XI with the cost of abstaining set to the cost of a person. The inequality is not strict: when the two expected costs are equal the rule routes, since a routed request does not occupy a person, and the convention is part of the rule because a threshold computed from round costs such as $c_{\mathrm{w}} = 4$ and $c_{\mathrm{e}} = 1$ ($0.75$) will be hit exactly by some probability in a long enough log.

```text
function decide_action(p: array<real>[4], c_wrong: real, c_escalate: real) -> int:
    require c_wrong > 0.0 and c_escalate > 0.0, "costs must be positive"
    if max(p) >= 1.0 - c_escalate / c_wrong:
        return argmax(p)                      # a route, 0 .. 3; a tie takes the first
    return -1                                 # -1 means: hand it to a person
```

Three prices make the table: a wrong route costs 2, 4 or 10 against a person at 1, so that the thresholds are $0.5$, $0.75$ and $0.9$. Every number is computed on the test requests, once with the raw probabilities and once with the temperature applied. Counting conventions that a reader might guess wrong are stated in the block: an escalated request is in neither the numerator nor the denominator of the selective accuracy, it costs $c_{\mathrm{e}}$ in the cost, and the cost per request is the mean over all 1000.

<!-- worked: decision_models.decision_rule -->
A wrong route costs c, a right route 0 and handing the request to a person costs 1 whatever the truth. The rule acts when p_max >= 1 - 1/c (a tie acts) and otherwise escalates. 'coverage' is the share of requests the rule acts on; 'selective accuracy' the share of those whose route is right (an escalated request is in neither the numerator nor the denominator); 'quoted' the mean over all requests of min((1 - p_max) c, 1), the cost the probabilities promise; 'paid' the mean cost actually incurred, an escalation counted as 1. Test split, four decimals.

| c | threshold | probabilities | coverage | selective accuracy | quoted | paid |
|---:|---:|---|---:|---:|---:|---:|
| 2 | 0.5000 | raw | 0.9890 | 0.8413 | 0.2297 | 0.3250 |
| 2 | 0.5000 | scaled | 0.9780 | 0.8446 | 0.2817 | 0.3260 |
| 4 | 0.7500 | raw | 0.8280 | 0.8937 | 0.3722 | 0.5240 |
| 4 | 0.7500 | scaled | 0.7880 | 0.9010 | 0.4544 | 0.5240 |
| 10 | 0.9000 | raw | 0.6490 | 0.9399 | 0.5598 | 0.7410 |
| 10 | 0.9000 | scaled | 0.5510 | 0.9528 | 0.6746 | 0.7090 |

The two constant policies on the same requests: always act on the argmax costs (1 - accuracy) c, always escalate costs 1:

| c | always act | always escalate |
|---|---:|---:|
| 2 | 0.3240 | 1.0000 |
| 4 | 0.6480 | 1.0000 |
| 10 | 1.6200 | 1.0000 |

The requests the temperature takes out of the routed set (the raw probabilities act, the scaled ones escalate; no request moves the other way, since T > 1 only lowers p_max), unnudged run: how many, how many of them the raw argmax routes right, what routing them costs (the wrong ones at c) and what escalating them costs (1 each).

| c | requests | routed right | cost if routed | cost if escalated |
|---:|---:|---:|---:|---:|
| 2 | 11 | 6 | 10 | 11 |
| 4 | 40 | 30 | 40 | 40 |
| 10 | 98 | 85 | 130 | 98 |

The tie convention in numbers: the scaled probabilities rounded to two decimals put p_max exactly on the threshold for this many of the 1000 test requests. All of them act under the rule above and would escalate under a strict inequality, so the coverage of that variant is lower by this count over 1000 (for rounded probabilities).

| c | threshold | requests with p_max on it |
|---:|---:|---:|
| 2 | 0.50 | 10 |
| 4 | 0.75 | 7 |
| 10 | 0.90 | 27 |
<!-- /worked -->

Reading the table from the top, the rule's behaviour is what Volume XI's was. At $c_{\mathrm{w}} = 2$ it hardly escalates ($0.9780$ of the requests are routed with the scaled probabilities) and the cost, $0.3260$, is within $0.002$ of always routing, $0.3240$. At $c_{\mathrm{w}} = 4$ it routes $0.7880$ of the requests, those it routes are right $0.9010$ of the time, and the cost is $0.5240$, against $0.6480$ for always routing and $1.0000$ for always escalating. At $c_{\mathrm{w}} = 10$, where a mistake is dear, routing every request would cost $1.6200$ and the rule pays $0.7090$ by sending $0.4490$ of the requests to a person, the other $0.5510$ being routed right $0.9528$ of the time. The price of a wrong route moves the rule from a router that rarely asks to one that asks nearly half the time, and nothing in the model changed.

Two rows of the table show what calibration is worth and what it is not. The columns "quoted" and "paid" are the cost the probabilities promise and the cost the rule incurs. With the raw probabilities at $c_{\mathrm{w}} = 4$ the head quotes $0.3722$ and pays $0.5240$, a gap of $0.1518$; with the scaled probabilities it quotes $0.4544$ and pays the same $0.5240$, a gap of $0.0696$. The paid cost did not move at this price, and the last table says why. The temperature takes $40$ requests out of the routed set, and the raw argmax is right on $30$ of them, which is exactly the $0.75$ at which routing and escalating cost the same ($40$ against $40$). At $c_{\mathrm{w}} = 10$ it takes out $98$, of which the raw argmax is right on $85$, routing them would cost $130$ against $98$, and the saving is $0.0320$ per request ($0.7410$ against $0.7090$). The gain of calibration is the honesty of the quote first and the cost second, and it is largest where the threshold is extreme. A gap of $0.07$ remains, which says that one temperature on 500 requests is not the last word and that the quote should be checked against the paid cost on fresh requests, as the table does.

The cost matrix need not be a threshold on one number. Suppose that routing a request to the dispute team costs $10$ if the request was not a dispute (the team is expensive to disturb), routing to another team that is wrong costs $2$, and a person costs $1.5$. The cheapest action is again the minimum over actions of the expected cost, and it is no longer the argmax's route.

<!-- worked: decision_models.cost_matrix -->
Five actions against four truths; the entry in row a and column y is the cost of action a when the request is truly of route y (the same rule as a threshold, argmin over actions of the sum over truths of probability times cost; a tie goes to the first action in the order below).

| action | card | payment | account | dispute |
|---|---:|---:|---:|---:|
| card | 0 | 2 | 2 | 2 |
| payment | 2 | 0 | 2 | 2 |
| account | 2 | 2 | 0 | 2 |
| dispute | 10 | 10 | 10 | 0 |
| escalate | 1.5 | 1.5 | 1.5 | 1.5 |

On the 1000 test requests with the scaled probabilities, the cheapest action differs from the argmax on 46 requests, 16 of them escalations, and the other 30 route to a team that is not the most probable one.

Test request 39 (`failed refund refund help help`, true route payment) has the scaled probabilities card 0.0007, payment 0.4710, account 0.0107, dispute 0.5176. The expected cost of each action:

| action | expected cost |
|---|---:|
| card | 1.9986 |
| payment | 1.0580 |
| account | 1.9787 |
| dispute | 4.8238 |
| escalate | 1.5000 |

The argmax is dispute; the cheapest action is payment.
<!-- /worked -->

The probability is not the action, and request 39 is the case: the most probable route is dispute at $0.5176$, but acting on it risks the expensive mistake, and the cheaper action is payment, at $0.4710$, whose mistake costs $2$. Of the 1000 test requests the cheapest action differs from the argmax on 46, of which 16 are escalations and 30 are a route that is not the most probable one. This is the reason a decision model returns the whole distribution and not only a label, and brain's `crates/decide`, described below, returns it.

The figure shows the whole path of one request, from words to action; the record at its bottom is what an auditor reads.

![The path of one request through the small decision model: the request text goes through the sentence encoder into six numbers and the request fields through scaling into four, the ten numbers are the state vector, the decision head produces four logits, the temperature turns them into probabilities, and the decision rule routes to a team or escalates; a record of the state vector, logits, temperature, probabilities, threshold, action and model identity is written beside the action.](img/decision-pipeline.svg)

## Learning from what the old router did

Everything so far learned from requests whose right route is known. A desk that already has a router has something else: a log of which route the old router chose and whether the request was then resolved. That is a record of outcomes of one policy, not labels. If the old router sent a request to payment and it was resolved, the right route was payment; if it was not, the log says only that the right route was not payment. The reader in `crates/decisions` records this distinction in the label of each example: every example carries its label, who decided it (`oracle`, `counterfactual_replay` or `scorer`, from `LabelSource` in `crates/decisions/src/examples.rs`) and an outcome, so that a label obtained by replaying the log counterfactually is not confused with one a person assigned.

Training on such a log is the subject of a larger literature, but one piece of it fits here, because it is what the decision to replace a router rests on: estimating the value of a new policy from the log of an old one, without deploying it. Suppose the old policy chose the route $a$ in the state $s$ with probability $\mu(a \mid s)$ (its **propensity**), and the outcome $r$ is 1 if the route was right. A new policy $\pi$ that routes in state $s$ to $\pi(s)$ has the value $V(\pi) = \mathbb{E}_s\,\mathbb{P}(r = 1 \mid s, \pi(s))$. The logged request tells us the outcome of the logged route, which is the new policy's route only some of the time. Weighting by the ratio of the probability the new policy gives the action to the probability the old one gave it repairs exactly that. For a deterministic $\pi$ the result is the **inverse propensity** estimate (IPW, for inverse propensity weighting),

$$\hat V_{\mathrm{IPW}} = \frac{1}{n}\sum_{i=1}^{n} \frac{r_i\,\mathbb{1}[a_i = \pi(s_i)]}{\mu(a_i \mid s_i)}, \qquad \mathbb{E}\,\hat V_{\mathrm{IPW}} = V(\pi),$$

because the expectation over the logged action of $r\,\mathbb{1}[a = \pi(s)]/\mu(a \mid s)$ is $\sum_a \mu(a \mid s)\,\mathbb{P}(r = 1 \mid s, a)\,\mathbb{1}[a = \pi(s)]/\mu(a \mid s) = \mathbb{P}(r = 1 \mid s, \pi(s))$: the propensity cancels. This is the importance ratio of Chapter 3 with the roles named differently, and it needs the same thing: the old policy must have given every route the new one might choose a positive probability, which is why the old router below explores at random 30 percent of the time. The inverse propensity form divides by the logged probability; its self-normalized variant divides the same sum by the sum of the weights $1/\mu$ instead of by $n$, which trades a small bias for a much smaller variance.

```text
function ipw_value(route: array<int>[n], logged: array<int>[n], outcome: array<real>[n],
                   propensity: array<real>[n], self_normalize: bool) -> real:
    require n > 0, "an empty log"
    num = 0.0
    den = 0.0
    for i in 0 .. n:                          # requests in order
        if route[i] == logged[i]:
            require propensity[i] > 0.0, "the logged route had probability zero"
            num = num + outcome[i] / propensity[i]
            den = den + 1.0 / propensity[i]
    if self_normalize:
        require den > 0.0, "no logged route agrees with the new policy"
        return num / den
    return num / n
```

The old router in the log follows the argmax of a head trained on the four fields alone (the weakest of the three) with probability $0.7$ and picks a route uniformly at random otherwise, so that every route has propensity at least $0.075$. It is applied to 2000 fresh requests. Two new policies are valued from the log: the head on both halves, and the constant policy that sends everything to dispute. In a simulation the true values are known, which is the only reason the table can print them.

<!-- worked: decision_models.logged_outcomes -->
The log is the 2000 log requests, each routed by the old policy (generator `decision_models.logging` (seed 0xadfdcd08a0d77546, 12537402374611236166); for each request in order one `uniform()`, below 0.7 the numbers-only head's argmax, otherwise one `integer(4)` for a uniformly random route), with the outcome 1 when the route was right and 0 otherwise and the propensity P(a | s) = 0.7 [a = old argmax] + 0.075 of the route it chose. Two target policies are valued: the argmax of the head on the whole state, and the constant policy that routes everything to dispute. The true value of a policy is the share of the requests it would route right, which only a simulation that keeps the labels can print. With w = outcome / propensity on the requests where the new policy's route is the logged one (and nothing elsewhere), 'inverse propensity' is the sum of w over all n requests divided by n, the self-normalized form divides the sum of w by the sum of 1 / propensity over the same requests, and the last row is the mean outcome over those requests alone. The first column uses all 2000 requests; the other two split the log into 20 blocks of 100 consecutive requests, compute each estimate on every block, and give the mean and the standard deviation (divisor 20) of the 20 values. Four decimals.

| target policy | estimate of its value | all 2000 | mean, blocks of 100 | sd, blocks of 100 |
|---|---|---:|---:|---:|
| new policy = the head's argmax | true value | 0.8385 | 0.8385 | 0.0389 |
| new policy = the head's argmax | inverse propensity | 0.9209 | 0.9209 | 0.2460 |
| new policy = the head's argmax | self-normalized inverse propensity | 0.8585 | 0.8616 | 0.0751 |
| new policy = the head's argmax | logged agreements only, no weights | 0.8582 | 0.8593 | 0.0492 |
| new policy = always route to dispute | true value | 0.1470 | 0.1470 | 0.0479 |
| new policy = always route to dispute | inverse propensity | 0.1639 | 0.1639 | 0.1001 |
| new policy = always route to dispute | self-normalized inverse propensity | 0.1610 | 0.1683 | 0.0967 |
| new policy = always route to dispute | logged agreements only, no weights | 0.4789 | 0.4674 | 0.1322 |

For reference the old policy's own mean logged outcome over all 2000 requests is 0.3840.
<!-- /worked -->

The old router itself is right on $0.3840$ of the log's requests. The head on both halves is truly right on $0.8385$ of them, and the inverse propensity estimate from all 2000 requests is $0.9209$, which is high by $0.08$: unbiased does not mean close, and the standard deviation of the estimate over blocks of 100 requests is $0.2460$, about six times that of the truth over the same blocks. The self-normalized estimate, $0.8585$, is much closer and $0.0751$ wide per block. The unweighted mean over the requests where the two policies agree, $0.8582$, is nearly as close here, and that is luck of the policy, not a property of the method. The constant policy shows it: always routing to dispute is right on $0.1470$ of the requests, the inverse propensity estimate is $0.1639$, and the unweighted mean over the requests the old router sent to dispute is $0.4789$, three times too large, because the old router sends a request to dispute when its fields look like a dispute, which is exactly when dispute is right. Agreement between two policies is correlated with the outcome, and without the weights the correlation is mistaken for skill.

The use of the estimate is a decision rule of the same shape as before. A new model replaces the old one only if the estimate of its value, with an honest width, beats the old one; Chapter 7 makes that comparison a test with a stated procedure and a gate. The part that belongs here is the log, and what has to be in it for the estimate to be possible: the route that was taken and the probability it was taken with.

## What an auditor can ask

Four of the five answers the desk needed are a matter of the record (what the router saw, what it believed, why it acted and which version said so), and a small model can give them because it has four properties that a prompt to a large language model has to be engineered to approximate.

The inputs are logged. The state vector of ten numbers is all the head reads, so logging it (with the raw words for a human to read) fixes the input exactly. The inference is deterministic: no sampling, no hidden state, and a fixed order of every sum, so that the same ten numbers give the same four logits, and the replay of a request reproduces the decision. A decision is traceable: the head is linear in its input, so the logit difference between the chosen route and its runner-up splits exactly into one contribution per input, and a reader can see which words and which fields decided. And the weights are versioned: the model that decided is identified by a digest of its parameters, so that "which version said so" has an answer that cannot drift.

The block records two decisions, one acted on and one escalated, replays the first from its logged state vector, scores the whole test split again in a shuffled order, and decomposes the first decision's logits. The model identity here is the SHA-256 of the 44 parameters, each rounded to four decimals as an integer so that the identity is a function of the printed table; brain's identity is the digest of the weights file itself, as the section on `crates/decisions` below shows.

<!-- worked: decision_models.audit_record -->
The decision rule is the one of the previous block at c = 4 (act when p_max >= 0.75), the probabilities are those of the head on the whole state divided by the fitted temperature T = 1.1762. The model identity is the SHA-256 of the 44 parameters (W row by row, then b), each written as the little-endian 32-bit integer round(1e4 w) (ties away from zero); first 16 hex digits 7afd87e7fc97d9d6, the same in all five runs.

Two test requests, the first acted on and the first escalated:

| request | words | numbers | logits z | probabilities | p_max | action |
|---:|---|---|---|---|---:|---|
| 0 | failed failed payment please my | 3.15, 11.00, 0.00, 0.00 | -2.1653, 4.9839, -1.6369, -1.6608 | 0.0023, 0.9907, 0.0036, 0.0035 | 0.9907 | route to payment |
| 1 | stolen today password help please | 3.29, 4.00, 0.00, 2.00 | 1.0535, -1.9864, 2.1555, -1.8078 | 0.2692, 0.0203, 0.6869, 0.0236 | 0.6869 | escalate |

Replay. The probabilities of request 0 recomputed from its logged state vector alone differ from the first computation by 0.0 in every entry. The 1000 test requests scored again in the order of one `shuffle` of the indices 0 to 999 (generator `decision_models.replay`, seed 0x3e1f6820310a7d46) give probabilities that differ from the first pass on 0 requests (a request's answer does not depend on the others).

Why payment rather than account for request 0: the logit difference z[payment] - z[account] splits exactly into one contribution per input group (the weight difference times the input, summed in input order) and the bias difference:

| input group | contribution to the logit difference |
|---|---:|
| sentence vector (6 numbers) | 6.8035 |
| ln amount | -0.1262 |
| account age (months) | -0.1098 |
| new device | 0.2238 |
| contacts in 30 days | 0.3935 |
| bias | -0.5641 |
| sum | 6.6208 |
<!-- /worked -->

Request 0 is routed to payment with probability $0.9907$, and the decomposition says why in terms a reader can check: the sentence vector contributes $6.8035$ to the logit difference of $6.6208$ between payment and account, the four fields together $0.3813$ (the sum of their four lines) and the bias $-0.5641$. The second request, `stolen today password help please` with two contacts, is the one the desk would want a person to read: it mentions a card word and an account word, and the head gives account $0.6869$ and card $0.2692$, below the threshold of $0.75$. The record has everything the customer's question needs except the sentence "the model was wrong", which only the label can supply and which the log should receive once the person has resolved the request. Appending that label to the record is what turns a deployed router into a source of the labelled data that the next version is trained on, and the leakage rule of the section on `crates/decisions` is what keeps the next evaluation honest.

Determinism has a boundary worth stating. The replay above compares two computations by the same code on the same machine and finds no difference at all. A replay on different hardware, or after a change in the order of a reduction (Volume IV), will differ in the last bits of the logits, and the decision can differ only for a request whose probability sits within those bits of the threshold. That is why the record stores the logits and the threshold and not only the action: a replay is a comparison to a stated tolerance, and an audit can tell a real change of behaviour from a rounding one.

## What one decision costs, counted in arithmetic

The case for a small model over a large one is a count, not an impression. A decision costs the multiplications and additions it needs (FLOPs) and the bytes of weights it must read to do them, and both can be derived from the shapes. The encoder that brain's `crates/decide` reads from a released checkpoint has 6 layers of width 384 with 12 heads and a feed-forward width of 1536, a vocabulary of 30522 wordpieces and 512 positions. A span of $n$ tokens costs per layer $n \cdot 2(3d^2 + d^2 + 2df)$ FLOPs for the four attention projections and the two feed-forward matrices, with $d = 384$ and $f = 1536$, and $4n^2 d$ for the attention scores and the weighted sum of values. The head then reads the encoded state: the keys and values of the $R$ state tokens are computed once, $2R \cdot d \cdot 2d$, and each option adds a query projection $2d^2$, attention $4Rd$ and a score $2d$.

The word-vector table of the desk is a stand-in and costs nothing to count, so the count is for the encoder brain actually reads, and the comparison is with the way a large language model would be asked the same question. That model is an assumption of the count, not a measured or a named one: a decoder with seven billion parameters, 2 FLOPs per parameter per prompt token and 2 bytes per weight, reading a prompt of 200 tokens (160 of state and 40 of instruction and options) and answering from one step of its output distribution, the cheapest way of using it. The count is for one request and for ten questions about the same state, where the small model encodes the state once and the large one is shown in two ways: reading the whole prompt for every question, and, kinder to it, reading the shared 160 tokens once.

<!-- worked: decision_models.cost_per_decision -->
The encoder is the 6-layer, 384-wide, 12-head, 1536-wide-feed-forward, 30522-word encoder that `crates/decide` reads from a released checkpoint (token table 30522 x 384, position table 512 x 384, type table 2 x 384, one layer norm; per layer a fused query-key-value matrix 384 -> 1152, an output matrix 384 -> 384, two layer norms and the feed-forward pair 384 -> 1536 -> 384, each with a bias). The cross-attention head is a query matrix 384 x 384 with a bias (147,840), a fused key-value matrix 384 -> 768 with a bias (295,680), one layer norm of 384 (gain and bias, 768) and a score vector of 384 with a bias (385). Parameter counts:

| part | numbers |
|---|---:|
| token, position, type embeddings and their layer norm | 11,918,592 |
| one layer | 1,774,464 |
| 6 layers | 10,646,784 |
| encoder total | 22,565,376 |
| cross-attention head | 444,673 |

In 32-bit floats the weights are 92,040,196 bytes. A span of n tokens costs, per layer, n x 2(3 d^2 + d^2 + 2 d f) FLOPs for the matrix products and 4 n^2 d for the attention, with d = 384 and f = 1536; the head costs 2 R d (2d) for the keys and values of the R state rows once, and per option 2 d^2 for the query, 4 R d for the attention and 2 d for the score. Assumptions: a state of R = 32 tokens, 4 options of 8 tokens each (every option is its own span), Q questions that share the state. The large model is a decoder with 7e+09 parameters at 2 FLOPs per parameter per prompt token and 2 bytes per weight; its prompt is 200 tokens (160 of state, 40 of instruction and options); 'shared' reads the state once for all Q questions, 'separate' reads the whole prompt for each. Attention of the large model is not counted, which favours it.

| Q | small encoder | small head | small total | large, separate | large, shared | large shared / small |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.371e+09 | 2.025e+07 | 1.391e+09 | 2.800e+12 | 2.800e+12 | 2,013 |
| 10 | 7.507e+09 | 3.267e+07 | 7.540e+09 | 2.800e+13 | 7.840e+12 | 1,040 |

Bytes of weights read once per forward pass at batch 1: small 92.0 MB, large 14 GB, a ratio of 152.
<!-- /worked -->

For one question the small model needs $1.391 \times 10^9$ FLOPs, of which the encoder is $1.371 \times 10^9$ and the head $2.025 \times 10^7$, against $2.800 \times 10^{12}$ for the large model: about 2000 times as many. For ten questions about one state the small model needs $7.540 \times 10^9$ and the large model, even with the state read once, $7.840 \times 10^{12}$, so the ratio falls to about 1000 because the large model's cost per question is the 40 tokens of instruction while the small model's is the encoding of four option texts. The weights to be read are $92.0$ MB against $14$ GB, a ratio of 152. These are counts of arithmetic and of bytes at batch 1 and take no machine into account; what time they take is a property of a device and the roofline of Volume IV, and a statement about speed that did not start from these counts would not say what was reduced. The count also says what the small model gives up: it has no generation, so it can only choose among options it is given, and its 22.6 million parameters (of which 11.9 million are embedding tables) have read far less text than the large one's.

## Where brain's decide crate puts the head

`crates/decide` is the desk's encoder and head at the scale of a released model, with one change in the shape that follows from a requirement the desk did not have: the options are supplied with each request. The crate's own statement of its contract is `(state, question, allowed answers)` in and a calibrated distribution over those answers out (`crates/decide/src/lib.rs`); nothing is generated, and an answer outside the supplied set cannot be produced. A head with four fixed outputs, like the desk's, learns which output is which by position, and cannot be pointed at a fifth team tomorrow. The head in brain scores each option as text instead.

The encoder is the 6-layer one counted above, imported one to one from the released all-MiniLM-L6-v2 checkpoint (`crates/decide/src/import.rs`; commit `05ac10e0f` records the parity against the reference implementation, with the worst disagreement below the tolerance of $5 \times 10^{-5}$). Sequences are packed end to end with a span for each, not padded, which, as the commit message of the encoder explains, removes a trap of bidirectional encoders: an unmasked pad position contributes to the attention of every real position, so a padded encoder is wrong in a way that still produces plausible numbers. The state is encoded once per request. Its text is cut into windows of at most `max_span` tokens that overlap by `overlap` tokens, so that a fact straddling a cut is seen whole by one window (`crates/decide/src/pack.rs`), and each option is its own short span, which carries the question's instructions beside the option's text. That is why instructions travel with the options and not with the state (`Question::slots` in `crates/decide/src/primitives.rs`): one state encoding then serves every question of the request, a request with ten questions costs one encode and ten cheap scorings, and the independence of the questions is structural and not a promise.

The head (`crates/decide/src/head.rs`) is a cross-attention from the options to the state. The query of an option is its `[CLS]` row of the encoder, the keys and values come from the state rows, and the score is a dot product with a learned vector over the layer-normalized sum of the attended context and the query:

$$\mathbf{q} = W_q\,\mathbf{h}_{\mathrm{opt}} + \mathbf{b}_q,\quad \mathrm{ctx} = \mathrm{crossattn}(\mathbf{q} \to \text{state rows}),\quad \mathrm{score} = \mathbf{w}^\top \mathrm{LN}(\mathrm{ctx} + \mathbf{q}) + b.$$

It has $444{,}673$ parameters, which the cost block above counts and the crate's documentation rounds to 445 thousand. Its softmax is not on the device: grouping options into questions, normalizing within a question, the loss and its gradient are host arithmetic over at most a few hundred numbers, which is what keeps the number of options defined by the request and not by a kernel. The two halves are trained with two optimizers, because the encoder arrives trained and the head starts from noise (`Decide` in `crates/decide/src/decide.rs`): one learning rate would be too large for one half or too small for the other, and two optimizers say so without a per-tensor multiplier to keep in sync. The head can be saved alone, with a record of what it is an adapter to; `Decide::save_head` refuses once the encoder has moved from the published one, because 445 thousand floats mean nothing without the encoder they were trained against. That refusal is the same idea as the model identity of the audit record.

Three stories from the crate's history are about the state encoder, the head's output and the confidence the model reports, and each is a lesson about a decision model in particular. Each is recorded in a commit message, and a number quoted from a message is that message's measurement.

The first is a state that was only half the state. `Decide::state_embedding` is the encoder's mean-pooled vector of the state, used by whatever sits beside the head (a critic, a router). Commit `0be22ef22` (18 September 2026) records that it returned the pooled vector of the first window alone, so that on a sales conversation, which averages 1.89 windows, the vector covered about the first half of the text and omitted the closing turns. The code today is a length-weighted mean over every window, and the comment in `crates/decide/src/decide.rs` states the one bias that remains, that the overlap between windows is counted twice. The commit message adds a measurement: a linear probe on the embedding went from $0.569$ to $0.601$ accuracy when the whole state was used. What made it hard to see, as the commit puts it, is that a partial embedding is a perfectly ordinary-looking vector of 384 floats. The lesson for the desk's state encoder is that its coverage of the request is a thing to test and not to assume: the desk's encoder reads all five words by construction, but an encoder that cuts a long request does not announce it.

The second is a yes-or-no question that cannot be a softmax over two options. The head answers three kinds of question: a choice among options, a score along ordered levels, and a proposition's probability (`Question` in `crates/decide/src/primitives.rs`). The proposition was first implemented as two options, "yes" and "no", scored against each other by a softmax. Commit `57e982923` (17 September 2026) records the consequence: trained for 14000 steps the model sat at chance, every probability pinned at $0.50$. The cause is arithmetic: the head uses an option's representation as its attention query, the two queries differ by a word and lie at cosine $0.989$, so both attend to the state almost identically and carry the same state signal, and a softmax sees only the difference of its inputs, which subtracts that signal back out. The block shows the mechanism on three states and two unit queries at that cosine.

<!-- worked: decision_models.common_mode -->
Two option queries of length 1 at cosine 0.989: q_yes = (1, 0) and q_no = (0.989, sqrt(1 - 0.989^2)). The context vector c of a state is what the attention hands back (chosen by hand, three states); an option's score is the dot product of its query with c. Four decimals.

| state | context c | score yes | score no | yes - no |
|---|---|---:|---:|---:|
| obvious close | (0.6, 0.1) | 0.6000 | 0.6082 | -0.0082 |
| obvious loss | (0.35, 0.1) | 0.3500 | 0.3609 | -0.0109 |
| undecided | (0.1, 0.05) | 0.1000 | 0.1063 | -0.0063 |

Spread (largest minus smallest) of the yes score over the three states: 0.5000; of the difference: 0.0046. The difference of two unit queries at cosine 0.989 is (q_yes - q_no) . c, and |q_yes - q_no| = sqrt(2 - 2 x 0.989) = 0.1483, so the difference can never be more than 0.1483 times the length of c, while one score alone can be as large as c itself.
<!-- /worked -->

The yes score moves by $0.5000$ across the three states and the difference by $0.0046$. The bound is the last sentence of the block: two unit queries at cosine $0.989$ differ by a vector of length $\sqrt{2 - 2 \times 0.989} = 0.1483$, so their score difference can never exceed $0.1483$ times the length of the context. The fix is in `Question::slots` and `Question::answer`: a proposition scores one slot and reads a logistic off its absolute score, so that the state signal survives. The commit's message reports the measured scores on three extreme conversations, a spread of $0.2895$ in the raw scores against $0.0066$ in their difference. A choice among options with genuinely different text is not affected.

The third is a confidence that means the same thing at any number of options. The distribution is always returned, but a caller who wants one number to threshold would take the largest probability, and its floor is $1/k$: it reads $0.5$ on a two-option question and $0.0039$ on a 255-option question for the same completely undecided answer. The brain engine publishes instead $1 - H/\ln k$ (`confidence` in `crates/decide/src/primitives.rs`), the entropy of the distribution against the largest entropy a $k$-option distribution can have, so that it is $1$ on a one-hot and $0$ on a uniform distribution whatever $k$ is. The test that pins it checks $k = 2, 5, 77, 255$ (commit `fcabe836a` explains the choice).

<!-- worked: decision_models.confidence_measure -->
confidence(p) = 1 - H(p) / ln k with H the entropy in nats and 0 ln 0 = 0, clamped to [0, 1]; four decimals. Three distributions over k options: uniform, one-hot, and one option at 0.5 with the rest sharing 0.5 equally.

| k | uniform p_max | uniform confidence | one-hot p_max | one-hot confidence | half p_max | half confidence |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 0.5000 | 0.0000 |
| 5 | 0.2000 | 0.0000 | 1.0000 | 1.0000 | 0.5000 | 0.1386 |
| 77 | 0.0130 | 0.0000 | 1.0000 | 1.0000 | 0.5000 | 0.3419 |
| 255 | 0.0039 | 0.0000 | 1.0000 | 1.0000 | 0.5000 | 0.3753 |
<!-- /worked -->

The first three columns are the property the commit claims. The last two are a caution: for a distribution with one option at $0.5$ and the rest sharing the other half, the largest probability is $0.5$ at every $k$, yet the confidence is $0$ at $k = 2$ and $0.3753$ at $k = 255$, since the rest of the distribution is more spread at a larger $k$. The measure says how concentrated the whole distribution is relative to the uniform one over the same options, which is not the same as how sure the model is of its top answer. That is the reason the full distribution is always returned, and it is why the desk's rule above thresholds $p_{\max}$, the quantity a cost is multiplied by, and not a normalized concentration.

The head's objective in `crates/decide` is the calibration objective of Volume XI, Chapter 6 (`crates/rlcd/src/scoring.rs`, re-exported through `crates/decide/src/loss.rs`): a mixture $(1 - \lambda_{\mathrm{B}})\,\mathrm{focal}(\gamma_{\mathrm{f}}) + \lambda_{\mathrm{B}}\,\mathrm{Brier}$ in which $\gamma_{\mathrm{f}} = 0$ is exactly cross-entropy, so that a sweep of the two dials compares four settings of one thing. It accepts a full target distribution as well as a gold index, so a model can be trained against an exact posterior (a two-in-three chance of a fault) and not only against "option 2". The training set that exercised it first was BANKING77, 13,083 customer utterances in 77 banking intents (`crates/decide/src/banking77.rs`), with the one change that turns a fixed classifier into a runtime-option model: each training example sees a random subset of the intents, always including the correct one, at a random position, because a model trained against all 77 every time can learn the position of the answer and ignore the option text. Its comment states the price plainly, that a model trained this way will usually score below one trained on the fixed 77-way task, and that the held-out-intent number is the one that measures what the model is for.

## A reward that is a probability, and a router with fixed bands

Not every decision has a label. The sequential-decision half of the crate treats a prediction as an action: `crates/decide/src/policy.rs` takes conversations whose outcome (converted or not) arrives only at the end, and trains the model to commit a probability of conversion at every turn. Chapter 3 described the PPO machinery it uses, a clipped surrogate and an entropy term and a measure of how far the policy has travelled from the one that collected the batch (the k3 estimate of that chapter, `drift` in the same file, and the stopping rule `target_kl` of commit `3a1a1cb57`). What is particular to a decision model is the choice of reward, and the module's documentation argues it with the algebra below.

The reading of "reward the accuracy of the estimate" that fails is the hit rate: sample a call, pay 1 if the call equals the outcome. A state whose outcomes are 1 with probability $q$ has the expected reward $pq + (1 - p)(1 - q)$ for the committed probability $p = \mathrm{sigmoid}(z)$, and its slope in $z$ is $(2q - 1)\,p(1 - p)$, which has the sign of $q - \tfrac{1}{2}$ whatever $p$ is. So the objective is maximized at the mode: a state that converts 70 percent of the time trains towards $p = 1$, accurate and useless. The reading that works is a proper scoring rule, $r = 1 - (p - y)^2$, whose expectation over $y$ has the slope $2(q - p)\,p(1 - p)$ and so vanishes exactly at $p = q$.

<!-- worked: decision_models.reward_direction -->
A state whose outcome has probability q of 1; the policy commits the probability p = sigmoid(z) and, in the first reading, samples a call (1 with probability p) and is paid 1 if the call equals the outcome. Its expected reward is p q + (1 - p)(1 - q), whose slope with respect to z is (2q - 1) p (1 - p). In the second reading it is paid 1 - (p - y)^2, whose expectation over y has the slope 2 (q - p) p (1 - p). The ascent direction in z (four decimals):

| q | p | hit-rate reward | proper-score reward |
|---|---:|---:|---:|
| 0.7 | 0.30 | 0.0840 | 0.1680 |
| 0.7 | 0.50 | 0.1000 | 0.1000 |
| 0.7 | 0.70 | 0.0840 | 0.0000 |
| 0.7 | 0.90 | 0.0360 | -0.0360 |
| 0.7 | 0.99 | 0.0040 | -0.0057 |
| 0.2 | 0.30 | -0.1260 | -0.0420 |
| 0.2 | 0.50 | -0.1500 | -0.1500 |
| 0.2 | 0.70 | -0.1260 | -0.2100 |
| 0.2 | 0.90 | -0.0540 | -0.1260 |
| 0.2 | 0.99 | -0.0059 | -0.0156 |

The value of p after 20000 steps of z <- z + 1.0 x direction from z = 0 (p = 0.5):

| q | reward | p after 20000 steps |
|---:|---|---:|
| 0.7 | hit | 0.9999 |
| 0.7 | proper | 0.7000 |
| 0.2 | hit | 0.0001 |
| 0.2 | proper | 0.2000 |
<!-- /worked -->

At $q = 0.7$ and $p = 0.9$ the hit-rate reward still pushes up by $0.0360$ and the proper score pushes down by the same amount, and after 20000 steps of ascent from $p = 0.5$ the two readings end at $0.9999$ and $0.7000$. The test `the_policy_optimum_is_the_conditional_rate` in `crates/decide/src/policy.rs` is the sampled version of this table, a stochastic fixed point on 60000 draws within $0.03$ of the true rates; its neighbour `a_hit_rate_reward_would_have_no_interior_optimum` pins the sign argument. The module's header says that this test is what caught the first reading when it was the first implementation (commit `57e982923` introduced the module). The distinction is general: a control policy is graded by return, because its optimum should be the best action, and `choice_loss` in the same file does exactly that; a probability is graded by a proper score, because its optimum should be the rate. The same module also notes what is left of PPO when the conversation is replayed and the action cannot change what the customer says next: a trust-region-regularized proper scoring rule, the consequence of the environment not responding and not a simplification chosen.

The fixed bands of `crates/decide/src/routing.rs` are the other end of the spectrum from this chapter's rule. The module combines three signals about a decision (the agreement of the decision representation with the encoder's own sentence embedding, a measure of how much the layer stack's variance falls, and a small learned confidence) into one score, and sends the decision to one of four pathways: act if the score is at least $0.75$, retrieve comparable cases if at least $0.55$, escalate to a larger model if at least $0.35$, otherwise hand it to a person. The three thresholds are a published method's own constants (`the_routing_bands_are_the_published_ones` in the module's tests says so), and the weights of the combination are fitted against correctness by ridge regression. Compare the rule above. Its threshold is derived from two prices and is a different number whenever the prices are; a fixed band of $0.75$ is the threshold the costs give when a wrong route costs four times an escalation, and for any other ratio it is not the threshold that minimizes the expected cost of Volume XI. `crates/rlcd/src/cost.rs` says as much in its header, that the router is a confidence-threshold cascade fitted against correctness, "with no notion that being wrong in different ways can cost different amounts", and `bayes_action` there is the closed-form $\arg\min$ over a cost matrix that the cost-matrix block above computed. The cascade is a reasonable shape when the pathways are several and the prices are not known; where the prices are known, the derivation replaces the constants.

The critic of Chapter 3 sits beside these in `crates/decide/src/value.rs`, and it makes the contrast of this section in one line of architecture: it is a two-layer network over the encoder's frozen state embedding that ends in a linear unit, because a value is a return and a squashed output would cap what it can ever predict (the test `a_value_is_unbounded` fits the targets $-2.5$ and $2.5$ and requires predictions beyond $-1.5$ and $1.5$), where the confidence network of `crates/decide/src/routing.rs` ends in a logistic because a confidence is a probability. It is the batch-independent baseline of Chapter 3 for a control policy whose encoder is frozen, so the fit is a small regression on the host.

## From a trained head to a model that may be chosen

A decision head that is good in a notebook is not yet a model an application may call. `crates/decisions` is the layer that makes it one, and what it adds is the evaluation, calibration and identity above, put behind a contract.

It starts with the data. The reader of outcome-labelled examples (`crates/decisions/src/examples.rs`, commit `8b29b3e9e`) is the only reader of the JSONL format; every downstream use, training, evaluation and promotion, sees its examples. It refuses a duplicate identity and a label that is not one of the options, and it refuses a leak: after the last line is read, three groupings are rejected if they span more than one split, the task, the family and seed together, and an identical state, instruction and option set. A held-out score means nothing if the held-out example, or the situation it came from, was seen in training, and one task asked by several runs is still one situation. This is the leakage rule of Volume II, Chapter 5 turned into a check on the file.

Evaluation (`crates/decisions/src/evaluation.rs`, commit `b3da5ef07`) reports on the test split what the desk's blocks computed, and each is defined once in the module. The accuracy is set against two baselines that need no model, the majority label of the training split (of the evaluated split itself, flatteringly, when no training split exists, and reported as such) and the chance baseline, the mean of $1/k$ over the examples; their larger is the baseline and the difference is the lift. It reports the cross-entropy, the squared-error probability score, ECE and AdaECE over 15 bins with the bins themselves (and notes that AdaECE cannot separate tied confidences, so that on a model confident to the same figure everywhere it measures the order of the file, not calibration), and the accuracy of the most confident tenth, two tenths and so on of the examples, the risk-coverage curve of Volume XI. Two further checks are specific to a model whose options arrive with the request. Every example is asked again with its options in a seeded random order and the share whose winning option changes is reported, the order instability, which is $0$ for a model that reads the options. And with a cost file the evaluation prices the model's own trust-or-defer policy: the action is to trust the top option (a wrong one excludes the right one, costing `false_exclusion`) or to defer and keep every option (a right one was wasted context, costing `wasted_tokens`), the model trusts when its $p_{\max}$ makes trusting the cheaper expected action by `bayes_action`, and the mean realized cost is reported against the two constant policies. That is the desk's pricing rule with two actions, and the module's text states the standard in a sentence: a model that cannot beat the cheaper constant policy has not learned when to trust itself.

The evaluation has a negative control, which is the better part of the commit. The test in `crates/sdk/tests/it/decision_eval.rs` trains a model on a synthetic task with a known answer and the same model on the same data with the training labels shuffled; the first must beat the baseline by more than $0.25$, the second by less than $0.15$. An evaluator that reports a lift on shuffled labels is measuring its own data, and this is the cheapest test that it is not.

Identity and promotion complete the contract. Every checkpoint may carry a manifest (`crates/decisions/src/manifest.rs`, commit `560b4d53e`) with the architecture, the SHA-256 of the weights and of the tokenizer, the lineage (what it was trained from and on) and a validation status, and the status is a claim someone recorded and never a default: no manifest, or no validation block, is unvalidated. A validated claim cannot be parsed without the profile it was measured on and the hash of the weights it applies to, and at load a declared hash that does not match the file is a failure. A validation that did not name its weights would stay true after the weights changed; here it cannot. Promotion (`crates/decisions/src/promotion.rs`, commit `cfe7ff1ac`) re-evaluates a checkpoint on the test split against a policy file of gates (a minimum test size, a minimum lift, a maximum ECE, a maximum order instability, a maximum regret under stated costs, required families) and writes the validation only if every gate passes, trusting no number stored with the checkpoint. At most one checkpoint per profile is validated, the one it supersedes is recorded as last-known-good in an atomically written pointer, and `rollback` swaps them back. The selector for `"auto"` (`crates/decisions/src/select.rs`) is a pure function: an explicitly configured model, else a validated Laya model (the option head of `crates/modernbert/src/laya.rs`, over the encoder of Volume V, Chapter 14), else a validated `decide` model, else nothing, and an unvalidated model is never chosen automatically, however good it might be. A caller that wants it names it. The gates here are a first version of the question Chapter 7 treats in general: when may a new model replace the one in use, and on what evidence.

## Check your understanding

1. Find the error: "Temperature scaling cut the expected calibration error from $0.0484$ to $0.0237$, so the router now routes more requests correctly."
2. A team sets the escalation threshold at $0.75$ because a published router uses it. Wrong routes cost them 10 and a person costs 1. What does the rule of this chapter say the threshold is, and in which direction is the team's router wrong at $0.75$?
3. The first dispute-heavy week shows the old router's logged outcomes for "route to dispute" at $0.48$ right. A colleague proposes replacing every route with dispute because the log says it works. What is wrong, and what in the log would let you value that policy?
4. Why does the audit record keep the state vector and the logits, and not only the text and the action?
5. A yes-or-no question is implemented as a softmax over the options "yes" and "no", and after training every answer is $0.5$ while the loss falls. What does the computation of this chapter say to look at first?
6. A promoted checkpoint's weights file is retrained for one more step and saved under the same name. What happens to its validation in brain's manifest, and why is that the right behaviour?
7. Two heads have the same accuracy on a test split; one has ECE $0.02$ and the other $0.08$. At a wrong-route cost ten times the escalation cost, which one does the rule's cost depend on more, and why?
8. The confidence $1 - H/\ln k$ of an answer is $0.34$ on a 77-option question. Can you conclude that the model's largest probability is below $0.5$?

## Summary

A small decision model is three parts and a rule. The state encoder turns the application's situation, text through a sentence embedding and structured fields through scaling, into a vector. The decision head is a policy over a fixed or a supplied set of actions, trained on labels by the cross-entropy whose logit gradient is $\mathbf{p} - \mathrm{onehot}(y)$. A temperature fitted on a held-out split calibrates its probabilities, and a rule derived from two prices acts when $p_{\max} \ge 1 - c_{\mathrm{e}}/c_{\mathrm{w}}$ and otherwise hands the request to a person; with a full cost matrix the cheapest action is not always the argmax. A log of the old policy's routes and their propensities values a new policy before it is deployed, with inverse propensity weights that are unbiased and noisy, and without them an agreement between policies is mistaken for skill. A decision is auditable when its inputs are logged, its inference is deterministic, its logit difference decomposes over its inputs and its weights have a digest. And one decision costs about $1.4 \times 10^9$ FLOPs and 92 MB of weights against about $2.8 \times 10^{12}$ and 14 GB for a large model asked the same question, a ratio that is counted and not measured.

In brain, `crates/decide` puts the options in the request and scores them as text against a state encoded once, so that it is a policy over a set that changes every call, and its history is a list of ways a decision model can look right and be wrong: a state encoder that saw half the request, a softmax that subtracted the answer, a reward that trained to the mode. `crates/decisions` wraps it in the checks above: a leakage-checked reader, baselines and a negative control, a manifest that binds a validation to its weights, and a promotion gate. The limit of all of it is the label. A head learns what the labels say, a calibration is only as honest as the held-out split, and a decision model that is right on its log can still be wrong on the requests the log never held. The next chapter takes up the model that keeps learning from new experience: what it forgets as it learns, how to keep what it knew, and how a new version is shown better than the one in use before it replaces it.

## References

- C. K. Chow, "On optimum recognition error and reject tradeoff", 1970.
- I. Casanueva, T. Temcinas, D. Gerz, M. Henderson and I. Vulic, "Efficient intent detection with dual sentence encoders", 2020.
- N. Reimers and I. Gurevych, "Sentence-BERT: sentence embeddings using Siamese BERT-networks", 2019.
- C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On calibration of modern neural networks", 2017.
- D. G. Horvitz and D. J. Thompson, "A generalization of sampling without replacement from a finite universe", 1952.
- A. Swaminathan and T. Joachims, "The self-normalized estimator for counterfactual learning", 2015.
- Nandakishor M, "Confidence-aware routing for large language model reliability enhancement: a multi-signal approach to pre-generation hallucination mitigation", 2025.

## Exercises

1. **[Calculate]** A request has three words, with the vectors $(1, 0)$, $(0, 1)$ and $(1, 1)$. Give the mean, its length and the sentence embedding of length 1, and say what the embedding of the same three words in a different order is.
2. **[Calculate]** A head with $K = 3$ routes gives the logits $\mathbf{z} = (1, 0, -1)$ to a request whose right route is 1. Give $\mathbf{p}$, the cross-entropy and the gradient with respect to $\mathbf{z}$, and check that its entries add to zero.
3. **[Calculate]** For the single validation request with $\mathbf{z} = (2, 0)$ and right route 1, give the slope $f'(a)$ of the cross-entropy at $a = 1$ and say in which direction the fitted $a$ lies. What does a slope of exactly zero at $a = 1$ mean?
4. **[Calculate]** The cost matrix of the chapter (wrong route $2$, wrong dispute route $10$, person $1.5$) and the probabilities $(0.10, 0.40, 0.05, 0.45)$ over (card, payment, account, dispute). Give the expected cost of each of the five actions, the argmax and the cheapest action.
5. **[Calculate]** Five logged requests agree with a new policy, with outcomes $1, 1, 0, 1, 1$, and propensities $0.775, 0.075, 0.075, 0.775, 0.775$, in a log of $n = 5$ requests. Give the inverse propensity estimate and the self-normalized one, and say which is larger and why.
6. **[Calculate]** A request has $p_{\max} = 0.75$ at $c_{\mathrm{w}} = 4$ and $c_{\mathrm{e}} = 1$. Give the expected costs of routing and of escalating, and the action. What does the action become if the rule used a strict inequality, and what does that do to the coverage of a model whose probabilities are rounded to two decimals?
7. **[Implement]** Reproduce `decision_models.world_rule`, `decision_models.state_vector` and `decision_models.head_parameters`: the generator and its draw order, the sentence vectors, the scaling with the divisor $n$, the initial weights and the 1500 steps of descent at the step size $0.5$, to the digits printed.
8. **[Implement]** Reproduce `decision_models.intent_heads`, `decision_models.calibration` and `decision_models.reliability_bins`, including the bisection for the temperature and the ECE of the largest probability over ten equal-width bins.
9. **[Implement]** Reproduce `decision_models.decision_rule` and `decision_models.cost_matrix`, and add a column for the price $c_{\mathrm{w}} = 6$ with its threshold. State what the strict inequality would change on the test requests.
10. **[Implement]** Reproduce `decision_models.logged_outcomes`: the old policy's draws, the propensities, the three estimators and the blocks of 100 requests. Add the policy "always route to payment" and say whether the unweighted agreement estimate is biased for it.
11. **[Implement]** Reproduce `decision_models.audit_record`, including the SHA-256 identity of the rounded parameters and the decomposition of a logit difference, and `decision_models.cost_per_decision`.
12. **[Investigate]** Train the head on 30, 60, 120, 240 and 480 requests drawn from the same generator (with names of your choosing, stated) and report, for each size, the test accuracy, the fitted temperature and the ECE before and after it. Where does the temperature stop being worth fitting, and what does it say about the 120 requests of the chapter?

## Key terms

- **state encoder**: the frozen function that turns an application's situation, free text through a sentence embedding and structured fields through scaling, into the vector a decision head reads. <!-- concept: state-encoder -->
- **decision head**: a small trained map from the state vector to a score for each action of a set, whose softmax is the probability of each, so that it is a policy over a fixed (or, in brain, a supplied) set of actions. <!-- concept: decision-head -->
- **intent classification**: choosing which of a closed list of intents a message expresses, the canonical use of a decision model, trained on messages whose intent has been supplied. <!-- concept: intent-classification -->

## Answers

**Recall questions.**

1. The mean is $(1.5, 0.5)$ with length $\sqrt{2.5} = 1.5811$, so the embedding is $(0.9487, 0.3162)$.
2. The probabilities are $e^2/(e^2 + 1) = 0.8808$ and $0.1192$; the greedy policy takes the first action, the one with the larger preference.
3. The mean probability $0.85$ exceeds the accuracy $30/40 = 0.75$, so the model is overconfident in that bin by $0.10$.
4. Before, $\mathrm{softmax}(2, 0) = (0.8808, 0.1192)$; after, $\mathrm{softmax}(1, 0) = (0.7311, 0.2689)$. The argmax is unchanged.
5. Answering costs $\min(p, 1 - p)$ in expectation for the better call, and abstaining $0.2$, so answering is cheaper when $1 - p_{\max} \le 0.2$, that is $p_{\max} \ge 0.8$.
6. $H = 0.5 \ln 2 + 2 \times 0.25 \ln 4 = 1.0397$ nats, which is $1.0397/1.0986 = 0.9464$ of $\ln 3$; one minus that, $0.0536$, is the confidence of this chapter's measure for $k = 3$.

**Check your understanding.**

1. Temperature scaling divides the logits by a number and so cannot change which logit is largest: the argmax, and with it the accuracy and the number of requests routed right, are exactly the same. What it improved is the probabilities, which is a matter of how much the rule can trust them.
2. $1 - 1/10 = 0.9$. The team's $0.75$ is too low: a request with $p_{\max} = 0.8$ is routed, at an expected cost of $0.2 \times 10 = 2$ against $1$ for a person, so the router makes mistakes it should have paid to avoid.
3. Of the requests the old router sent to dispute, those are the ones its fields made look like a dispute, so agreement is correlated with being right, and the unweighted mean is a biased estimate of how dispute does on requests it was not chosen for (the chapter's table: $0.4789$ against a true $0.1470$). The log needs the propensity of the logged route; with it the inverse propensity estimate is unbiased, given that every route had positive probability.
4. The logits and the threshold let a replay be compared to a tolerance when the hardware or the order of a sum changes, and the state vector fixes the head's input exactly; the text alone would also depend on the encoder's version. The action alone cannot distinguish a real change of behaviour from a rounding one.
5. The queries of "yes" and "no" are nearly parallel, so both options attend to the state alike and a softmax, which sees only the difference of two scores, subtracts the state signal out. The cure is to score one proposition and read a logistic off its absolute score.
6. The manifest's validation names the hash of the weights it was measured on, and at load a declared hash that does not match the file is a failure. The retrained file is therefore refused as the validated model until it is promoted again, which is right because the measured evidence is about different numbers.
7. The one with the larger ECE. The rule multiplies a probability by a price; at a ratio of ten the threshold is $0.9$, in the steep part of the curve, where a miscalibration of $0.08$ moves many requests across it, and the quoted cost of the less calibrated head is wrong by correspondingly more.
8. No. For one option at $0.5$ and the rest sharing the other half, the confidence is $0.3419$ at $k = 77$ with a largest probability of exactly $0.5$; the measure is the concentration of the whole distribution against the uniform one, and the same value can come from different largest probabilities.

**Exercises.**

1. The mean of the three vectors is $(2/3, 2/3)$, of length $\sqrt{8}/3 = 0.9428$, so the embedding is $(0.7071, 0.7071)$. A mean does not depend on the order of the words, so any order gives the same embedding (floating-point sums in different orders can differ in the last digits, which is why the chapter fixes the order).
2. $\mathbf{p} = (0.6652, 0.2447, 0.0900)$, the cross-entropy is $-\ln 0.2447 = 1.4076$, and the gradient is $\mathbf{p} - \mathrm{onehot}(1) = (0.6652, -0.7553, 0.0900)$, which adds to zero.
3. At $a = 1$ the probabilities are $(0.8808, 0.1192)$, the probability-weighted mean logit is $0.8808 \times 2 = 1.7616$ and the logit of the right route is $0$, so the slope is $1.7616 > 0$: the cross-entropy rises with $a$ and the fitted $a$ is below $1$, because the one request is confidently wrong. Here the slope is positive for every $a \ge 0$ (at $a = 0$ it is $(2 + 0)/2 - 0 = 1$), so the routine's first requirement fails: a single wrong request would push the temperature to infinity, and that is why a temperature is fitted on hundreds of requests. A slope of exactly zero at $a = 1$ means the logits are already calibrated on that set in the sense of the cross-entropy.
4. The expected costs are card $2 \times 0.90 = 1.8$, payment $2 \times 0.60 = 1.2$, account $2 \times 0.95 = 1.9$, dispute $10 \times 0.55 = 5.5$ and a person $1.5$. The argmax is dispute ($0.45$) and the cheapest action is payment, at $1.2$.
5. The sum of $r/\mu$ is $1/0.775 + 1/0.075 + 0 + 1/0.775 + 1/0.775 = 1.2903 \times 3 + 13.3333 = 17.2043$, so the inverse propensity estimate is $17.2043/5 = 3.4409$ (an estimate above 1 is a sign of how noisy the weights are). The self-normalized one divides the same sum by $3 \times 1.2903 + 2 \times 13.3333 = 30.5376$ (the sum of $1/\mu$ over the five): $0.5634$, much smaller, because it is a weighted mean of outcomes and so lies between 0 and 1.
6. Routing costs $(1 - 0.75) \times 4 = 1$ and escalating $1$: a tie, and the rule routes. With a strict inequality it would escalate, and a model whose probabilities are rounded to two decimals would send every request that lands exactly on $0.75$ to a person, lowering the coverage by the share of such requests: in the table of the rule the scaled probabilities rounded to two decimals sit exactly on the threshold for 7 of the 1000 test requests at $c_{\mathrm{w}} = 4$ (a coverage lower by $0.007$), 10 at $c_{\mathrm{w}} = 2$ and 27 at $c_{\mathrm{w}} = 10$.
7. The block of the chapter, to the digits printed; the draw order is in the block.
8. The block of the chapter; the temperature is $T = 1.1762$ on the validation requests.
9. The block of the chapter; at $c_{\mathrm{w}} = 6$ the threshold is $1 - 1/6 = 0.8333$.
10. The block of the chapter; the unweighted agreement estimate is biased for any policy whose route correlates with the old router's, which "always payment" does through the amount and the contacts.
11. The block of the chapter; the identity is the first 16 hexadecimal digits of the SHA-256 of the 44 integers.
12. The expected shape: the test accuracy rises with the training size, the fitted temperature is largest where the head overfits most (the fewest requests) and tends to $1$ as the size grows, and the ECE before scaling falls towards the ECE after it; the size at which the two meet is the size at which fitting a temperature stops being worth a held-out split.
