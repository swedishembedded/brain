<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Decoding: Turning Probabilities into Text

*Given next-token probabilities, how do we choose what to say?*

A language model never says anything. Ask the counting model of [Chapter 4](04-language-modeling.md) what follows the word `the` and it answers with eighteen numbers, one for each token it knows, and the numbers add up to 1. Turning that answer into a word is a separate act, done by a few lines of code outside the model, and it matters about as much as the model does. Two choices suggest themselves at once. Take the largest of the eighteen, every time. Or draw from all eighteen, each in proportion to its number. Each one fails, in opposite ways, and both failures can be measured on this page before the chapter has said anything clever.

The first choice repeats itself: the model's favourite continuation of every word is fixed, so the text walks a short circle and stays on it. The second is faithful to the model and therefore to its mistakes: a model that gives 9 percent of its probability to twelve tokens that cannot follow `the` will put one of them there about one time in eleven. Everything in this chapter sits between those two, and everything is a rule applied to the same eighteen numbers: reshape them, cut them, penalize some of them, or give up choosing one token at a time.

Some ideas from earlier chapters are needed at once, so bring them back first. The answers are at the end of the chapter.

1. (Volume III, [Chapter 3](../v03-neural-networks/03-softmax.md), softmax and logits.) Three logits are $(2, 0, 0)$. What probabilities does the softmax give them, and what happens to the probabilities when every logit is increased by 5?
2. (Volume I, [Chapter 17](../v01-mathematics/17-randomness.md), sampling from a distribution.) The probabilities of three tokens are $(0.5, 0.3, 0.2)$, listed in that order, and the uniform number drawn is $u = 0.83$. Which token does the cumulative walk return?
3. (Volume V, [Chapter 4](04-language-modeling.md), the autoregressive factorization.) A text of three tokens has conditional probabilities $0.5$, $0.4$ and $0.1$ for its tokens in turn. What is the probability of the whole text?
4. (Volume I, [Chapter 13](../v01-mathematics/13-information-theory.md), entropy.) What is the entropy, in nats, of the distribution $(0.5, 0.25, 0.25)$?
5. (Volume I, [Chapter 15](../v01-mathematics/15-floating-point.md), underflow.) The smallest positive normal number of 32-bit floating point is about $1.18 \times 10^{-38}$. A text of $n$ tokens, each given probability 0.1, has probability $0.1^n$. From which $n$ on is that product below the smallest normal number?
6. (Volume I, [Chapter 17](../v01-mathematics/17-randomness.md), reproducibility.) Two runs of a program use the same seed and print different random numbers. Name two things other than the seed that could differ.

## Eighteen numbers and a choice

The model is the interpolated bigram model of Chapter 4: counts from its 40 printed training sentences, which reads the one word before the word it predicts. Its answer is a probability for each of its 18 tokens, and its logits are the logarithms of those probabilities plus a constant. The constant is chosen as 4 here, and it is arbitrary, because the softmax cannot see it: add any number to every logit and the probabilities are unchanged. A later section shows one place where the constant does matter, which is a reason to write it down.

<!-- worked: decoding.toy_distribution -->
The model: the interpolated bigram model of Chapter 4 (counts from the 40 printed training sentences, interpolation ratio 8, a uniform floor of 0.01 shared over the 18 tokens), which reads the one token before the one it predicts. Its logits are $z_i = \ln p_i + 4$ (the 4 is arbitrary: the softmax of $\mathbf{z}$ is $p$ for any constant). The context is the single word `the`. Ties in probability keep the lower id first (`cats` has id 4, `dogs` id 5).

| rank | token | logit $z$ | $p$ | cumulative $p$ |
|---:|---|---:|---:|---:|
| 1 | `fox` | 2.413 | 0.2045 | 0.2045 |
| 2 | `dog` | 2.339 | 0.1900 | 0.3945 |
| 3 | `foxes` | 2.259 | 0.1754 | 0.5699 |
| 4 | `cat` | 2.173 | 0.1608 | 0.7307 |
| 5 | `cats` | 1.569 | 0.0880 | 0.8187 |
| 6 | `dogs` | 1.569 | 0.0880 | 0.9067 |
| 7 | `the` | 0.267 | 0.0239 | 0.9306 |
| 8 | `.` | -0.159 | 0.0156 | 0.9462 |
| 9 | `near` | -0.728 | 0.0088 | 0.9551 |
| 10 | `ball` | -0.915 | 0.0073 | 0.9624 |
| 11 | `fish` | -1.082 | 0.0062 | 0.9686 |
| 12 | `sees` | -1.145 | 0.0058 | 0.9744 |
| 13 | `big` | -1.212 | 0.0055 | 0.9799 |
| 14 | `small` | -1.212 | 0.0055 | 0.9853 |
| 15 | `chase` | -1.360 | 0.0047 | 0.9900 |
| 16 | `see` | -1.635 | 0.0036 | 0.9936 |
| 17 | `chases` | -1.747 | 0.0032 | 0.9968 |
| 18 | `bone` | -1.747 | 0.0032 | 1.0000 |

The six nouns hold 0.9067 of the probability and the other twelve tokens 0.0933. After the word `near` the three largest are `the` 0.9039, `.` 0.0156, `near` 0.0088.
<!-- /worked -->

Read the table as a reader of the language would. After `the` comes a noun, and there are six of them, which hold 0.9067 of the probability between them. The other twelve tokens, none of which can follow `the` in the language, hold 0.0933: the model has not forgotten the grammar, but its smoothing has mixed word frequencies into every distribution, and a token such as `the` or the full stop keeps a probability of a few percent in a place where it never occurs. Nothing about this is particular to counting models. A softmax gives every token a positive probability (Volume III, Chapter 3), so every model has a tail, and what decides whether the tail matters is how much probability it holds in all.

Draw from it the way Volume I taught, one uniform number and a walk along the cumulative probabilities.

<!-- worked: decoding.sampling_junk -->
Generator: the book's SplitMix64, seed 0xee8aaa94d40768a0 (`seed_of` of the name `decoding.junk`). Each of the 2,000 draws is one `uniform()`; the token is the first id, in increasing id order, whose cumulative probability exceeds it (the walk of Volume I, Chapter 17). The distribution is the one after `the` of the table above.

| draw | $u$ | token |
|---:|---:|---|
| 1 | 0.3933 | `fox` |
| 2 | 0.8477 | `foxes` |
| 3 | 0.8962 | `foxes` |
| 4 | 0.4117 | `fox` |
| 5 | 0.6953 | `dogs` |
| 6 | 0.7303 | `dogs` |
| 7 | 0.0220 | `the` |
| 8 | 0.3789 | `fox` |

Over 2,000 draws: `fox` 397, `dog` 390, `foxes` 364, `cat` 317, `cats` 196, `dogs` 155; the other twelve tokens together 181 times, 9.05 percent against the 9.33 percent the distribution assigns them. If twenty tokens were drawn from this distribution one after another, the probability that every one is a noun would be $0.9067^{20} = 0.1410$.
<!-- /worked -->

The count of 181 out of 2,000 is the tail made visible, and the last sentence of the block is the way it hurts. A text is a chain of such draws, so the chance that all of them avoid the tail shrinks with the length: if twenty tokens each faced this distribution, only 14 percent of the texts would have no impossible token in them. The tail is not a nuisance to be tolerated in a long text, it is certain to appear.

## Always taking the largest

The opposite rule has a name. **Greedy decoding** picks, at every step, the token of largest probability, and the first of them when two are tied. It needs no random number, it is a function of the context alone, and it costs one argmax per token (the reduction of Volume IV, [Chapter 7](../v04-computation/07-reductions.md)). Here it is on the same model, started from the start marker and run for 30 tokens without stopping at the full stop.

<!-- worked: decoding.greedy_run -->
Greedy decoding of the bigram model of the previous block from the start marker, 30 tokens, no stop at the full stop (the model has never seen a word after a full stop, so its counts for that context are empty and it falls back to word frequencies; the most frequent word is `the`).

`the fox sees small fish . the fox sees small fish . the fox sees small fish . the fox sees small fish . the fox sees small fish .`

| step | previous token | largest | $p$ | second | $p$ |
|---:|---|---:|---:|---|---:|
| 1 | `<s>` | `the` | 0.9039 | `.` | 0.0156 |
| 2 | `the` | `fox` | 0.2045 | `dog` | 0.1900 |
| 3 | `fox` | `sees` | 0.3201 | `near` | 0.1974 |
| 4 | `sees` | `small` | 0.3197 | `ball` | 0.2588 |
| 5 | `small` | `fish` | 0.4124 | `ball` | 0.2781 |
| 6 | `fish` | `.` | 0.8956 | `the` | 0.0239 |
| 7 | `.` | `the` | 0.2108 | `.` | 0.1362 |

The text repeats with period 6 from the first token. Mean log-probability per token of the greedy text: -1.0214. The smallest gap between the two largest probabilities over the 30 steps is 0.0146, so no step is a near tie.

For contrast, three 30-token texts sampled from the full distribution at every step. Generator: seed 0xf907437323a5004d (`seed_of` of `decoding.docs`), 30 draws per text, one `uniform()` each, the three texts one after another from the same generator.

| text | tokens | mean log-probability per token |
|---:|---|---:|
| 1 | the fox near the fox near the fox big ball . the fox near the cat near the dog chase ball fish . . near the foxes chases big bone | -1.4687 |
| 2 | chases small bone . ball . near the fox see small fish . big ball . the foxes near the cat near the cat sees small fish . the fox | -1.3528 |
| 3 | the foxes see big ball . ball . . fish . sees big fish . chase ball . foxes see big bone . the fox sees see big bone . | -1.4651 |
<!-- /worked -->

The text is one sentence, `the fox sees small fish .`, six tokens long, said five times. Every step of the table is a clear win, so the loop is not a coincidence of near ties. It is forced, and the argument is short. The model reads one token. The token chosen after a given token is the largest probability for that context, and it is the same every time the context returns, because greedy decoding is a deterministic function of what the model reads. The model's next choice depends on the one token it read, and 18 different tokens can be read, so it can be in at most 18 different states; a text that has made 20 steps has therefore been in one of them twice (the start marker is a 19th, visited once), and from the second visit on it does exactly what it did from the first. A model that reads the last $k$ tokens has at most $18^k$ states, which is a bound of 324 for two tokens and no bound at all in practice for a window of a thousand. The pigeonhole argument stops being useful there, but the mechanism does not: a passage that has been written becomes part of the context, and a context that contains the passage makes the passage's own continuation likely.

The sampled texts in the second table show the other side. Their mean log-probability per token, about $-1.4$, is lower than the $-1.02$ of the greedy text, and they contain things such as `fish . . near the foxes`, which no one would write. Greedy decoding is not producing better text; it is producing the most probable token at each place, which is a different thing and the subject of the section on beam search. In brain's history the two failures were one commit's message. Commit `0e77db2b8` (October 2026) replaced the sampling a chat model used when its checkpoint declared nothing, which had been "greedy or temperature 0.8 with an untruncated distribution", with a fixed fallback that both cools the distribution and cuts its tail (three numbers, a temperature, a top-p and a top-k, which the next sections define and the last section of the chapter gives), and its message gives the reason in the chapter's own words: the old setting "loops and admits the tail". The comment on the fallback in `crates/data/src/generation.rs` says the same, greedy decoding "which loops" or an untruncated distribution "which admits the tail". The rest of this chapter is how such a fallback is built from its three numbers.

## Sharpening and flattening the distribution

The cheapest change to the distribution leaves its order alone. Divide every logit by a number $\tau > 0$ before the softmax:

$$
p_i(\tau) = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}.
$$

The number $\tau$ is the **temperature**, borrowed from statistical physics where the same formula gives the probability of a state of energy $-z_i$; with $\tau = 1$ nothing changes. Volume II, Chapter 6 divided logits by a temperature for another purpose, to recalibrate probabilities that were too confident, and the operation is the same. What differs is who chooses the number: there it was fitted to held-out data so that probabilities would be honest, and here it is chosen by the user of the model to trade variety against safety.

Because $z_i = \ln p_i + c$, the result has a form that needs no logits at all: $p_i(\tau)$ is proportional to $p_i^{1/\tau}$, the original probability raised to the power $1/\tau$ and renormalized. Two tokens with probabilities $p_i > p_j$ have the ratio $(p_i/p_j)^{1/\tau}$ afterwards. As $\tau \to 0$ the exponent grows without bound and the ratio goes to infinity: all the probability goes to the largest token, the greedy rule is the limit, and a tie between two largest tokens stays a tie. As $\tau \to \infty$ the exponent goes to 0, every ratio goes to 1, and the distribution is uniform.

<!-- worked: decoding.temperature_table -->
Logits of the distribution after `the` (the table above) divided by $\tau$, then the softmax. 'Non-nouns' is the probability of the twelve tokens the language does not allow after `the`; 'effective tokens' is $e^{H}$ with $H$ the entropy in nats. The uniform distribution has $H = \ln 18 = 2.8904$ and 18 effective tokens. Tolerance of every number: half a unit in the last digit.

| $\tau$ | largest $p$ | six nouns | non-nouns | entropy $H$ | effective tokens |
|---|---:|---:|---:|---:|---:|
| 0.01 | 0.9994 | 1.0000 | 0.0000 | 0.0052 | 1.01 |
| 0.25 | 0.3655 | 0.9999 | 0.0001 | 1.4279 | 4.17 |
| 0.50 | 0.2767 | 0.9925 | 0.0075 | 1.6768 | 5.35 |
| 0.75 | 0.2344 | 0.9603 | 0.0397 | 1.8897 | 6.62 |
| 1.00 | 0.2045 | 0.9067 | 0.0933 | 2.0995 | 8.16 |
| 1.50 | 0.1629 | 0.7894 | 0.2106 | 2.4137 | 11.18 |
| 2.00 | 0.1369 | 0.6964 | 0.3036 | 2.5910 | 13.34 |
| 4.00 | 0.0943 | 0.5192 | 0.4808 | 2.8099 | 16.61 |
| 100.00 | 0.0569 | 0.3403 | 0.6597 | 2.8903 | 18.00 |

The derivative of the entropy with respect to $\tau$ by a central difference with step $10^{-5}$ against the formula $\mathrm{Var}_{q}(z)/\tau^3$ of the text:

| $\tau$ | central difference | $\mathrm{Var}_q(z)/\tau^3$ |
|---|---:|---:|
| 1.0 | 0.787851 | 0.787851 |
| 2.0 | 0.259461 | 0.259461 |
<!-- /worked -->

![Probability of the 18 tokens after the word the, in order of their probability at temperature 1, at temperatures 0.5, 1 and 2. A low temperature steepens the head and empties the tail; a high one flattens the head and lifts the tail. The two tokens at ranks 5 and 6 have equal probability at every temperature.](img/temperature-distribution.svg)

Two things in the table are the point. The first is the one-way movement of the entropy: 0.0052 nats at $\tau = 0.01$, 2.0995 at $\tau = 1$, 2.8903 at $\tau = 100$ against the ceiling of $\ln 18 = 2.8904$. It is not a coincidence of this distribution. Write $\beta = 1/\tau$, $q_i = e^{\beta z_i}/Z$ and $Z = \sum_j e^{\beta z_j}$. Then $\ln q_i = \beta z_i - \ln Z$, and the entropy is $H = -\sum_i q_i \ln q_i = \ln Z - \beta\,\mathbb{E}_q[z]$. The derivative of $\ln Z$ with respect to $\beta$ is $\mathbb{E}_q[z]$, and the derivative of $\mathbb{E}_q[z]$ is the variance $\mathrm{Var}_q(z)$, so

$$
\frac{dH}{d\beta} = \mathbb{E}_q[z] - \mathbb{E}_q[z] - \beta\,\mathrm{Var}_q(z) = -\beta\,\mathrm{Var}_q(z),
\qquad
\frac{dH}{d\tau} = \frac{dH}{d\beta}\cdot\frac{d\beta}{d\tau} = \frac{\mathrm{Var}_q(z)}{\tau^3}.
$$

A variance is never negative, so the entropy never falls as the temperature rises, and it is flat only when all logits are equal. The second table checks the formula against a central difference at two temperatures, and the two columns agree to the six digits printed.

The second thing is what temperature does to the tail. Lowering $\tau$ from 1 to 0.5 cuts the probability of the twelve impossible tokens from 0.0933 to 0.0075, and to 0.25 cuts it to 0.0001. That is the effect wanted, and its price is in the same rows: the effective number of tokens $e^H$ falls from 8.16 to 5.35 to 4.17, and the largest token rises from 0.2045 to 0.3655, although six nouns are equally good continuations. Temperature reshapes the whole distribution, head and tail together, and it cannot remove the tail without also shrinking the head. The next section cuts the tail and leaves the head alone.

The limit $\tau = 0$ is a division by zero and is defined as the greedy rule. Implementations take the limit explicitly: in brain a temperature at or below zero means the argmax of the logits and draws no random number (`SampleParams::is_greedy` in `crates/model/src/serve.rs`), and a sampler that wants to stay on one code path clamps the temperature to a small positive value instead, as `crates/kronos/src/generate.rs` does with $10^{-6}$. The clamp is safe because the stable softmax of Volume III subtracts the largest logit first: the largest becomes $e^0 = 1$, every other term underflows to 0, and the answer is the greedy one.

## Cutting off the tail

If the trouble is twelve tokens that carry 9 percent of the probability, the direct remedy is to forbid them and renormalize what is left. Two rules do it, and they differ in how they decide where the tail starts.

**Top-k sampling** keeps the $k$ most probable tokens and sets the rest to zero. With $S$ the set kept, the new probabilities are $p_i / \sum_{j \in S} p_j$ for $i \in S$ (Fan, Lewis and Dauphin 2018). A tie at the edge has to be broken by a rule. The book's rule is the lower id first, which is what a stable sort gives; after `the` the fifth and sixth tokens, `cats` and `dogs`, both have 0.0880, so a top-k of 5 keeps `cats` and drops `dogs` for no reason in the probabilities. Some implementations keep every token whose logit is at least the $k$-th largest, and so keep more than $k$ when there is a tie. Brain's two samplers do not: the full-vocabulary sampler of `crates/qwen3vl/src/sample.rs` and the candidate-list path of `crates/model/src/serve.rs` both keep exactly $k$ tokens and, among equal logits, the lowest ids first.

**Top-p sampling**, also called nucleus sampling, keeps the smallest set of most probable tokens whose probability adds up to at least $p_{\mathrm{top}}$ (Holtzman et al. 2020), and renormalizes it. The loop is a cumulative sum over the tokens in order of decreasing probability that stops at the first partial sum reaching $p_{\mathrm{top}}$ (it is written out in the next section); the set it stops on is the nucleus. A third rule, **min-p**, keeps the tokens whose probability is at least a fraction $r_{\min}$ of the largest probability. All three cut the same two distributions below: after `the` and after `near`, the word that in this language can only be followed by `the`.

<!-- worked: decoding.truncation_table -->
Top-k, top-p and min-p cut the distributions after `the` and after `near`. Settings are applied alone. 'kept' is the number of tokens left, 'mass' the probability they held before renormalizing, and 'bad' how many of them the language does not allow (after `the` only the six nouns are allowed, after `near` only `the`). The nucleus rule is: the shortest prefix of the tokens in order of decreasing probability whose cumulative probability is at least $p_{\mathrm{top}}$, with the cumulative sum of the renormalized probabilities; min-p keeps the tokens with probability at least $r_{\min}$ times the largest.

| setting | `the`: kept | mass | bad | `near`: kept | mass | bad |
|---|---:|---:|---:|---:|---:|---:|
| top-k 1 | 1 | 0.2045 | 0 | 1 | 0.9039 | 0 |
| top-k 2 | 2 | 0.3945 | 0 | 2 | 0.9195 | 1 |
| top-k 4 | 4 | 0.7307 | 0 | 4 | 0.9357 | 3 |
| top-k 6 | 6 | 0.9067 | 0 | 6 | 0.9478 | 5 |
| top-k 8 | 8 | 0.9462 | 2 | 8 | 0.9590 | 7 |
| top-p 0.5 | 3 | 0.5699 | 0 | 1 | 0.9039 | 0 |
| top-p 0.8 | 5 | 0.8187 | 0 | 1 | 0.9039 | 0 |
| top-p 0.9 | 6 | 0.9067 | 0 | 1 | 0.9039 | 0 |
| top-p 0.95 | 9 | 0.9551 | 3 | 7 | 0.9536 | 6 |
| min-p 0.1 | 7 | 0.9306 | 1 | 1 | 0.9039 | 0 |
| min-p 0.3 | 6 | 0.9067 | 0 | 1 | 0.9039 | 0 |

After `the` the fifth and sixth places are tied (`cats` and `dogs`, both 0.0880), so a top-k of 5 keeps `cats` and drops `dogs` by the lower-id rule. Top-p 0.95 over the full distribution keeps 9 tokens; over a list of only the 8 best candidates (renormalized from a mass of 0.9462) it keeps 6.
<!-- /worked -->

![Cumulative probability of the n most probable tokens after the and after near. After near the first token already holds 0.9039, so a top-p cut at 0.9 keeps one token; after the it takes six. A top-k cut at four tokens, the vertical line, keeps 0.7307 of the probability in one place and 0.9357 in the other.](img/nucleus-cut.svg)

A fixed count cannot be right in both places. Top-k with $k = 4$ cuts the two good nouns `cats` and `dogs` after `the`, losing 0.1760 of the probability that belongs to correct answers, and keeps three impossible tokens after `near`, where one token holds nearly all of it. A $k$ of 6 is exactly right after `the`, and a $k$ of 8 admits two impossible tokens there. Top-p adapts: 0.9 keeps six tokens after `the`, which are exactly the six nouns, and one token after `near`. Min-p adapts the same way, and needs no sorting, because its threshold depends on the largest probability alone.

None of the three is safe at every setting. Top-p at 0.95 keeps nine tokens after `the`, three of them impossible, and seven after `near`, six of them impossible: the nucleus is as wide as the doubt of the model, and when the tail is long, a high $p_{\mathrm{top}}$ includes some of it. Min-p at 0.1 keeps the token `the` after `the`, whose probability 0.0239 is just above one tenth of 0.2045. The settings are a dial, not a cure, and a later table measures what turning it does to whole sentences.

A further limit concerns how a decoder gets the sorted probabilities. A server cannot afford to bring all of a vocabulary's logits back from the card each step, and brain's serving engines read back only a short list of the best candidates of each row, whose length is a constant of the engine named `TOPK_CAPACITY`: 32 in `crates/qwen35moe/src/serve.rs` and 64 in `crates/qwen3/src/serve.rs`. Everything after that, temperature, top-p, the normalization, works on the candidate list, so the cut is also a top-k of the list's length whatever else is asked. The last sentence of the block shows what that costs a nucleus of 0.95: over the exact distribution it has 9 tokens, over a list of the 8 best it has 6, because the tail the list has dropped is no longer counted when the cumulative sum is taken. The comment on `sample_from_topk` in `crates/model/src/serve.rs` calls this "a documented ceiling on `top_p`'s width, not a bug", which is a fair description of a trade chosen with its price known.

## The order of the steps

The pieces so far are applied one after another to the same vector, and the order in which they are applied is part of the definition of the sampler, not a matter of taste. The order used here, and by brain, is: penalties on the raw logits (the next section), then temperature, top-k, top-p and min-p, then renormalization and the draw. In the book's pseudocode:

```text
function choose_token(z: array<real>[V], seen: array<bool>[V], count: array<int>[V],
        theta: real, beta_pres: real, beta_frq: real, tau: real, k: int,
        p_top: real, r_min: real, rng: inout Generator) -> int:
    require theta > 0 and tau >= 0 and k >= 0, "penalty and temperature ranges"
    require 0 < p_top and p_top <= 1 and 0 <= r_min and r_min <= 1, "cut ranges"
    w = copy(z)
    for i in 0 .. V:                                  # 1. penalties, on the raw logits
        if seen[i]:
            if w[i] > 0:
                w[i] = w[i] / theta
            else:
                w[i] = w[i] * theta
        if count[i] > 0:
            w[i] = w[i] - (beta_pres + beta_frq * count[i])
    if tau == 0:                                      # 2. temperature 0 is greedy: no draw
        return argmax(w)
    q = softmax([w[i] / tau for i in 0 .. V])         # 3. temperature
    order = argsort(q, descending)                    #    ties keep the lower id first
    n = V                                             # 4. top-k
    if k > 0:
        n = min(k, V)
    total = sum([q[order[j]] for j in 0 .. n])
    keep = n                                          # 5. top-p over the renormalized survivors
    acc = 0.0
    for j in 0 .. n:
        acc = acc + q[order[j]] / total
        if acc >= p_top:
            keep = j + 1
            break
    final = zeros(real, [V])                          # 6. min-p, then renormalize
    for j in 0 .. keep:
        if q[order[j]] >= r_min * q[order[0]]:
            final[order[j]] = q[order[j]]
    norm = sum(final)
    for i in 0 .. V:
        final[i] = final[i] / norm
    return draw(final, rng)

function draw(q: array<real>[V], rng: inout Generator) -> int:
    u = rng.uniform()                                 # one draw per token, none at tau = 0
    acc = 0.0
    last = 0
    for i in 0 .. V:
        if q[i] > 0:
            acc = acc + q[i]
            last = i
            if u < acc:
                return i
    return last
```

Each step is a decision that changes the result, and the next table makes four of them visible on the distribution after `the`, with a temperature of 1.5, a top-k of 10 and a top-p of 0.9.

<!-- worked: decoding.filter_order -->
Distribution after `the` (the table of the first block), temperature $\tau = 1.5$, top-k $k = 10$, top-p $p_{\mathrm{top}} = 0.9$ (cumulative rule of the previous block). 'tokens' is how many remain, 'largest' the probability of the most likely, and 'non-nouns' the total probability on tokens the language does not allow after `the`. The line without a temperature is the same cuts on the model's own distribution.

| order of the steps | tokens | largest $p$ | non-nouns |
|---|---:|---:|---:|
| temperature, then top-k, then top-p | 7 | 0.1967 | 0.0470 |
| top-k, then top-p, then temperature | 6 | 0.2064 | 0.0000 |
| temperature, then top-p, then top-k | 10 | 0.1819 | 0.1184 |
| top-k, then top-p (no temperature) | 6 | 0.2256 | 0.0000 |

The softmax of the logits of the six best tokens and the probabilities of those six renormalized agree to below 1e-16 (largest absolute difference; the digits of a difference at the last bit of a probability are rounding noise).

The probability of `fox` after `the` when `fox` and `dog` have been seen, with the penalty applied to the logits before or after they are divided by $\tau = 0.5$ (the logits $z$ above, penalties on `fox` and `dog`):

| penalty and place | $p$(`fox`) |
|---|---:|
| repetition penalty 1.5, before the division | 0.0939 |
| repetition penalty 1.5, after the division | 0.0939 |
| presence penalty 1.0, before the division | 0.0676 |
| presence penalty 1.0, after the division | 0.1510 |
<!-- /worked -->

The first row is the order above: the temperature widens the distribution, the top-k of 10 keeps ten tokens, and top-p of 0.9 over their renormalized probabilities trims them to seven, one of which is impossible (`the`), for 0.047 of the probability. The second row applies the cuts to the model's own distribution and spends the temperature last, so the temperature cannot widen the set: six tokens, none of them wrong. The last row has no temperature at all and keeps the same six, with a larger largest probability, 0.2256. The third row cuts by top-p before top-k: at a temperature of 1.5 the nucleus of 0.9 already holds ten or more tokens, a top-k of 10 is no limit, and 0.1184 of the probability sits on impossible tokens. The same three settings give seven, six and ten tokens.

The reason for the differences is a single fact. The order of the tokens by probability does not depend on the temperature (division by a positive number keeps the order), so top-k, which uses the order alone, can be done before or after it. Top-p and min-p use the probabilities themselves, and the probabilities depend on the temperature and on how much mass the earlier cuts removed, so where they sit in the pipeline matters. The convention above makes the temperature mean what a user expects: temperature 1.5 flattens the distribution the cuts then see.

The line below the table is a convenience that follows from the way a softmax is built. The softmax of the logits of the six best tokens and the probabilities of those six, renormalized, agree to below $10^{-16}$, a few units in the last bit of a probability. This is not luck. If $S$ is the set kept, $p_i/\sum_{j\in S}p_j = (e^{z_i}/Z)/(\sum_{j \in S}e^{z_j}/Z) = e^{z_i}/\sum_{j\in S}e^{z_j}$, the softmax of the kept logits alone, because the common factor $1/Z$ cancels. So a cut can be done on logits by setting the dropped ones to $-\infty$ and taking one softmax afterwards, which is what `crates/data/src/sample.rs` and the other full-vocabulary samplers do, or on probabilities by dropping terms and dividing by what is left, which is what the candidate-list sampler does. They give the same distribution.

The two rows at the bottom are about the penalties of the next section, and they show an order that is not a matter of taste for one kind of penalty and is for the other. Dividing the logits by $\tau$ and then dividing a seen token's positive logit by $\theta$ is the same as the other way round, since $(z/\theta)/\tau = (z/\tau)/\theta$ and a division by a positive number keeps the sign: the repetition penalty commutes with the temperature, and the table gives 0.0939 both ways. A penalty that subtracts a fixed amount does not: a presence penalty of 1.0 taken from the logits before they are divided by $\tau = 0.5$ acts like 2.0 in the units the softmax sees, and the same subtraction after the division acts like 1.0, so the probability of `fox` is 0.0676 in one order and 0.1510 in the other. A penalty written as "1.0" has no meaning until the order is stated; brain's samplers apply all penalties to the raw logits, before the temperature, and their source says so (`crates/model/src/serve/sampler.rs`, introduced by commit `f2eab2a43`, October 2026).

What the settings do to complete sentences is measured next. Each setting draws 200 sentences from the order-4 model of Chapter 4, the model that reads three words, and the grammar of that chapter judges them.

<!-- worked: decoding.decoding_quality -->
Each setting draws 200 sentences from the order-4 interpolated model of Chapter 4 (interpolation ratio 2), starting from `<s>`, ending at the full stop or after 12 tokens; one `uniform()` per token, the token chosen by the id-order walk over the tokens left after the cuts (cuts applied in the order temperature, top-k, top-p, min-p, each renormalizing). One generator per setting, seeds: `decoding.quality.full` 0x793a5d9d84d3d736; `decoding.quality.t07` 0x14329edf85f51d01; `decoding.quality.t15` 0xe48a3ace95507e38; `decoding.quality.k4` 0x0afe1de104f05b50; `decoding.quality.p09` 0x938f2a2ef86a0292; `decoding.quality.m01` 0x38f7545031f6704c; `decoding.quality.t07p09` 0xb1e005a8219f411f; `decoding.quality.t15p09` 0x32a9a59bdc532a8f. 'accepted' is the share the grammar of Chapter 4 accepts (right shape, verb agreeing with the subject); 'distinct' counts different sentences; 'surprisal' is the mean of $-\ln p$ per token under the model itself (no cut, temperature 1) over all tokens of the sentences.

| setting | accepted | percent | distinct | surprisal |
|---|---:|---:|---:|---:|
| full distribution | 78/200 | 39.0 | 161 | 1.2315 |
| temperature 0.7 | 128/200 | 64.0 | 117 | 0.8374 |
| temperature 1.5 | 23/200 | 11.5 | 184 | 2.0838 |
| top-k 4 | 136/200 | 68.0 | 114 | 0.8863 |
| top-p 0.9 | 138/200 | 69.0 | 120 | 0.8227 |
| min-p 0.1 | 129/200 | 64.5 | 115 | 0.8183 |
| temperature 0.7, top-p 0.9 | 159/200 | 79.5 | 79 | 0.7590 |
| temperature 1.5, top-p 0.9 | 39/200 | 19.5 | 170 | 1.8297 |
| greedy (one sentence) | 0/1 | 0.0 | 1 | 0.6851 |

The greedy sentence is `the foxes near the fox sees ball .`.
<!-- /worked -->

Three results stand out. Cutting the tail helps: top-k 4, top-p 0.9 and min-p 0.1 take the share of accepted sentences from 39.0 to between 64.5 and 69.0 percent, and the mean surprisal of the sentences under the model, which is a measure of how unlikely the sampler's own choices were, falls from 1.2315 to between 0.8183 and 0.8863. Combining a temperature of 0.7 with top-p 0.9 does best at 79.5 percent. A temperature above 1 hurts, as expected: 11.5 percent at 1.5, and even with top-p 0.9 on top, 19.5 percent, because the cut then retains a flatter distribution. And the price of the improvement is variety: the number of distinct sentences among the 200 falls from 161 to 79 for the best row. A sampler is a point on a curve between a text that is right and a text that is varied, and there is no setting that is simply good.

The last row is a warning against reading the first row of the pipeline as the goal. The greedy sentence, `the foxes near the fox sees ball .`, has the lowest surprisal of the table, 0.6851, and the grammar rejects it: the verb agrees with the nearest noun, `fox`, and not with the subject, `foxes`. The most probable token at each place, under this model, builds a sentence with an error, because the error is the model's and greedy decoding cannot be better than what it is given.

## Punishing words already said

Loops are what repetition looks like when the model is at fault. An alternative to flattening or cutting is to change the logits of tokens that have already appeared. The **repetition penalty** of Keskar et al. (2019), introduced for a controllable language model, takes a factor $\theta_{\mathrm{rep}} > 1$ and applies it to every token already in the context: a positive logit is divided by $\theta_{\mathrm{rep}}$ and a negative logit is multiplied by it. The sign rule is not a detail. The aim is to make a seen token less probable, which means making its logit smaller, and dividing a negative logit by a number above 1 moves it toward zero, that is, up.

<!-- worked: decoding.repetition_rule -->
Logits $(3, 1, -1, -3)$ of four tokens that have all been seen, penalty $\theta_{\mathrm{rep}} = 2$. 'divide all' is the first rule one might write; 'sign rule' divides a positive logit and multiplies a negative one.

| logit $z$ | divide all | sign rule |
|---|---:|---:|
| 3 | 1.5 | 1.5 |
| 1 | 0.5 | 0.5 |
| -1 | -0.5 | -2 |
| -3 | -1.5 | -6 |

Greedy decoding of the bigram model of the first block for 30 tokens, no stop at the full stop, with the repetition penalty on every token seen so far (the context starts empty; the first token has nothing to be penalised for). The logits are $\ln p + c$ for the offset $c$ in the first column, which gives the same probabilities for every $c$. 'accepted' counts the complete sentences the grammar accepts out of the complete sentences in the text; 'distinct' is the number of different tokens used. At every step the two largest penalized logits are either exactly equal (`cats` and `dogs`, which the lower-id rule settles) or more than $10^{-9}$ apart.

| offset $c$ | $\theta_{\mathrm{rep}}$ | accepted | distinct tokens |
|---:|---:|---:|---:|
| 0 | 1 | 5/5 | 6 |
| 0 | 1.3 | 2/3 | 11 |
| 0 | 1.5 | 2/3 | 13 |
| 0 | 2 | 2/5 | 16 |
| 4 | 1 | 5/5 | 6 |
| 4 | 1.3 | 3/4 | 15 |
| 4 | 1.5 | 3/4 | 15 |
| 4 | 2 | 2/4 | 17 |
| 8 | 1 | 5/5 | 6 |
| 8 | 1.3 | 4/5 | 17 |
| 8 | 1.5 | 1/5 | 17 |
| 8 | 2 | 1/6 | 17 |

Two of these texts in full.

- offset 0, $\theta_{\mathrm{rep}} = 1.5$: `the fox sees small fish . the dog near the foxes near the cat near the fox chases big ball . the fox sees small fish . the fox sees`
- offset 8, $\theta_{\mathrm{rep}} = 1.5$: `the fox sees small fish . near the dog chase ball . big bone . foxes see big ball . cat chases big ball . the cats see big ball`
<!-- /worked -->

The first table shows the rule on four logits. Dividing all of them gives 1.5, 0.5, $-0.5$ and $-1.5$, and the last two have risen: the penalty has rewarded tokens that were already unlikely. The sign rule gives 1.5, 0.5, $-2$ and $-6$.

The second table shows something less comfortable, and it is the reason the constant 4 was written down at the start. The logits of the model are $\ln p + c$ for an offset $c$, the probabilities are the same for every $c$, and the rule's effect is not: at $c = 0$ every logit is negative, so every seen token is multiplied and a rare token loses more than a common one; at $c = 8$ every logit is positive, so every seen token is divided and a token with a large logit loses most. Nothing in the loss of Volume III fixes the offset, since the softmax cross-entropy ignores a shift of all logits, so which of these a trained model is in is an accident of its training, and the penalty is a different penalty for each. In this experiment the damage is largest at the largest offset: at $c = 8$ and $\theta_{\mathrm{rep}} = 1.5$ the text breaks into `near the dog chase ball . big bone . foxes see big ball .`, and one sentence in five is accepted, because `the` is a seen token after the first sentence and is penalized away, and a sentence without its article is not a sentence. At $c = 0$ the penalty stops the loop and keeps two sentences of three, with a long chain of `near the` phrases that the bigram model cannot know it has already used.

A seen-token penalty cannot tell a loop from grammar. Words such as `the` have to repeat; the way to tell them apart is to look at repeated *sequences* and not repeated tokens, which is what two other processors of brain's sampler do. The n-gram filter bans a token that would complete a sequence of $n$ tokens already present in a recent window (`apply_no_repeat_ngram` in `crates/model/src/serve.rs`, which keeps a whitelist for tokens that legitimately recur, such as the table tags of a document being read), and the DRY penalty (a name for "do not repeat yourself") subtracts $m \cdot b^{\,\ell - \ell_0}$ from the logit of a token that would extend a repeat of length $\ell$ beyond an allowed length $\ell_0$, with a multiplier $m$ and a base $b > 1$ that grow the penalty geometrically. With the test's own numbers, $m = 2$, $b = 2$ and $\ell_0 = 2$, a token that would extend a repeat of six tokens loses $2 \cdot 2^4 = 32$ from its logit (`dry_penalises_the_token_that_would_extend_a_repeat` in `crates/model/src/serve/sampler.rs`), a penalty that grows geometrically with the length of the repeat and is zero on the short repeats that language needs.

The other two penalties are the ones of the OpenAI interface. A **presence penalty** $\beta_{\mathrm{pres}}$ subtracts a fixed amount from the logit of every token that has appeared in the generated text, and a **frequency penalty** $\beta_{\mathrm{frq}}$ subtracts the amount for every occurrence, so a token generated $c$ times loses $\beta_{\mathrm{pres}} + \beta_{\mathrm{frq}}\,c$.

<!-- worked: decoding.additive_penalties -->
Four tokens with logits $(2.4, 2.0, 0.5, -1.0)$; token 0 has been generated 3 times, token 1 once, tokens 2 and 3 never. The penalty subtracted from a token generated $c > 0$ times is $\beta_{\mathrm{pres}} + \beta_{\mathrm{frq}}\, c$.

| $\beta_{\mathrm{pres}}$ | $\beta_{\mathrm{frq}}$ | $z_0$ | $z_1$ | $z_2$ | $z_3$ | $p_0$ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 2.40 | 2.00 | 0.50 | -1.00 | 0.5396 |
| 1 | 0 | 1.40 | 1.00 | 0.50 | -1.00 | 0.4613 |
| 0 | 0.5 | 0.90 | 1.50 | 0.50 | -1.00 | 0.2746 |
| 0.5 | 0.5 | 0.40 | 1.00 | 0.50 | -1.00 | 0.2396 |
<!-- /worked -->

Unlike the multiplicative rule, these need no sign rule and have no dependence on the offset, but they are linear in the count, so a token said twenty times under $\beta_{\mathrm{frq}} = 0.5$ loses 10, more than the entire spread of any logit row above. The brain sampler counts them over the generated text only, and the repetition penalty over the whole context, prompt included, the semantics of the two interfaces it follows; the test `presence_and_frequency_penalties_count_generated_tokens_only` and its neighbour pin the difference.

A last point concerns the work. Done naively, a penalty rebuilds the set of seen tokens from the whole context at each generated token, which visits $n + j$ tokens for the new token numbered $j$, counting from 0 (the prompt has $n$ tokens and no new one yet when $j = 0$). For a prompt of 8,000 tokens and 1,000 generated tokens, $j = 0, \dots, 999$, that is $1000 \cdot 8000 + 999 \cdot 1000/2 = 8{,}499{,}500$ token visits against 1,000 if each new token is folded into a set that is kept. Commit `5786b1f74` (October 2026) made that change: each sequence now carries a `ProcessorState` that is updated by the tokens appended since the previous step, and `apply_processors` stays as the one-shot form that rebuilds from scratch. Its tests keep the old whole-context implementation as an oracle and compare the two on every prefix of random contexts.

## Log-probabilities, and which distribution they belong to

A decoder deals in products of probabilities and does not want to. The probability of a text is the product of the conditional probabilities of its tokens (Chapter 4), and a product of many numbers below 1 is small enough to leave the range of the arithmetic.

<!-- worked: decoding.logprob_underflow -->
A text of $n$ tokens each given probability 0.1 by the model: the product $0.1^n$ against the sum $n \ln 0.1$, with each product rounded to the stated format (`float32` is IEEE binary32 with subnormals; `float64` is the double of the rest of the book).

| quantity | value |
|---|---:|
| float32 product after 40 tokens | 1.000e-40 (below the smallest normal float32, 1.18e-38: a subnormal) |
| float32 product is exactly 0 from token | 46 |
| float64 product is exactly 0 from token | 324 |
| sum of 400 log-probabilities, float64 | -921.034 |
<!-- /worked -->

A text of 40 tokens at 0.1 each is already a subnormal number in 32 bits, and from token 46 the product is exactly zero, a probability that looks impossible for a text that is merely long; the 64-bit product survives until token 324. The sum of the logarithms has no such limit: 400 tokens at $\ln 0.1 = -2.3026$ give $-921.034$. The **log-probability** of a token is $\log p(x_t \mid x_{<t})$, a number at most 0, that of a text is the sum of its tokens' (the form in which Chapter 4 wrote the loss), and with it beam search can add, a client can compare two texts of the same length, and a mean over tokens turns into the cross-entropy and the perplexity of Volume I. This is why a decoder and an interface report log-probabilities of what they chose, and often of the runners-up.

The numbers a decoder reports have to say which distribution they are of, because there are two.

<!-- worked: decoding.logprob_report -->
One sentence sampled from the order-4 model with $\tau = 0.7$ and top-p 0.9 (cuts in the order temperature, top-p). Generator: seed 0x769c128532f440a7 (`seed_of` of `decoding.report`), one `uniform()` per token, id-order walk. 'model $\ln p$' is the log-probability the model itself gives the sampled token (temperature 1, no cut); 'sampler $\ln q$' is the log-probability under the distribution the token was actually drawn from; 'next best' lists the two most probable other tokens by model $\ln p$.

| position | token | model $\ln p$ | sampler $\ln q$ | next best, model $\ln p$ |
|---:|---|---:|---:|---|
| 1 | `the` | -0.0634 | 0.0000 | `.` -4.646, `near` -5.198 |
| 2 | `cats` | -2.0138 | -1.9577 | `foxes` -1.575, `cat` -1.688 |
| 3 | `see` | -0.7671 | -0.5783 | `near` -1.160, `chase` -1.851 |
| 4 | `small` | -0.5430 | -0.3733 | `big` -1.098, `ball` -3.857 |
| 5 | `ball` | -0.9232 | -0.7932 | `bone` -0.956, `fish` -1.878 |
| 6 | `.` | -0.0687 | 0.0000 | `the` -4.229, `near` -5.198 |

`the cats see small ball .`: sum of model $\ln p$ = -4.3792 (probability 0.0125, mean -0.7299 per token over 6 tokens); sum of sampler $\ln q$ = -3.7025.
<!-- /worked -->

The third column is the model's own log-probability of the sampled token, its softmax at temperature 1 with nothing cut. The fourth is the log-probability under the distribution the token was actually drawn from, after temperature 0.7 and top-p 0.9 and renormalization. For `see` the two are $-0.7671$ and $-0.5783$; for the first token, `the`, the second is 0, because the nucleus held that token alone. The sums are $-4.3792$ and $-3.7025$. Both are correct, and they answer different questions: the first says how likely the model finds the text, and the second how likely this sampler was to write it.

Brain reports the second. The scheduler records with each generated token the log-probability under the temperature, top-k, top-p and min-p truncated softmax it was drawn from, and 0 for a greedy token, a rule fixed by commit `d3740711b` (October 2026) so that the plain sampler and the verification step of speculative decoding (a way to draft several tokens cheaply and check them in one pass of the model; Chapter 12 returns to it) agree on one definition (`sample_from_topk_with_logprob` in `crates/model/src/serve.rs`). Since the draw works on the candidate list, the normalizer covers the 32 or 64 candidates and not the whole vocabulary. The server returns it for the sampled token when a request asks for `logprobs`, and refuses `top_logprobs`, the alternatives, by name, because only the sampled token's value exists on that path (`crates/apiserve/src/sampling.rs`). The model's own log-probabilities and its top alternatives are computed elsewhere, on the scoring path, by `row_logprob` and `row_top_logprobs` in `crates/model/src/logprobs.rs`: a stable log-softmax at one index, and a selection of the $k$ largest with a heap, the cost of one pass over the row plus $V \log_2 k$ comparisons.

The two distributions stay apart in a place where mixing them up costs a training run. Reinforcement learning from sampled answers needs the probability, under the model being trained, of the tokens the sampler wrote, and compares it with the probability the sampler had when it wrote them. Knowledge #290 records the first mistake: a rollout (a text sampled from the model being trained, kept with the probabilities it was drawn with so that it can be scored later) stored the log-probability under the softmax of the logits divided by the temperature, and the trainer scored the plain softmax, so that at a temperature of 1.5 the trainer's probability of the sampled token differed from the sampler's already at the first update (by up to 12.3 percent in that entry's measurement on a tiny model). The fix of commit `d1807c4ed` (October 2026) is to scale the trainer's logits by $1/\tau$, so that it scores the distribution the sampler used, and `token_logprobs_scaled` in `crates/model/src/logprobs.rs` is the oracle that computes the same thing on the host. The second mistake is the subject of commit `de2e91432` (November 2026). A sampler that cuts the distribution draws token $a$ with probability $q(a) = p(a)/\sum_{a' \in S}p(a')$, where $p(a)$ is the model's probability of $a$ and $S$ is the set of tokens the cuts kept ($q(a) = 0$ outside $S$, and $\ln q$ is the sampler log-probability of the table above), and the trainer's ratio basis does not score that, since exact truncated likelihoods are not implemented; the validation of the reinforcement-learning trainer's configuration (`GrpoConfig::validate`) therefore refuses top-k, top-p, min-p, penalties and a thinking budget on the trainer's ratio basis, rather than train against a distribution that never produced the samples, and a test pins top-k, top-p, min-p and penalties as refused.

## Several candidates at once: beam search

Greedy decoding maximizes each factor of the product. The product is what measures the text, and the largest factor now can leave only small ones later. Here is a language with three tokens $a$, $b$ and $c$, small enough to do by hand, in which that happens.

<!-- worked: decoding.beam_hand -->
A language with three tokens $a, b, c$ and sequences of exactly three. First-token probabilities $(0.5, 0.4, 0.1)$; the probability of the next token after each token (rows are the previous token, columns the next one):

| after | $a$ | $b$ | $c$ |
|---|---:|---:|---:|
| a | 0.1 | 0.4 | 0.5 |
| b | 0.1 | 0.7 | 0.2 |
| c | 0.6 | 0.1 | 0.3 |

The joint probability of a prefix is the product of the probabilities along it. Beam search with width 2 keeps, after each step, the two prefixes of largest joint probability (a tie would keep the candidate generated first, beam by beam and token by token; there is none here). Candidates of each step, best first:

| step | candidates (joint probability) |
|---:|---|
| 1 | `a` 0.5000, `b` 0.4000, `c` 0.1000 (dropped) |
| 2 | `bb` 0.2800, `ac` 0.2500, `ab` 0.2000 (dropped), `bc` 0.0800 (dropped), `aa` 0.0500 (dropped), `ba` 0.0400 (dropped) |
| 3 | `bbb` 0.1960, `aca` 0.1500, `acc` 0.0750 (dropped), `bbc` 0.0560 (dropped), `bba` 0.0280 (dropped), `acb` 0.0250 (dropped) |

Greedy decoding gives `aca` with probability 0.1500. Beam search of width 2 returns `bbb` with probability 0.1960. The best of all 27 sequences is `bbb` with probability 0.1960, and the second best `aca` with 0.1500.
<!-- /worked -->

**Beam search** keeps $B$ partial texts alive instead of one, the beam, and at each step extends every one of them by every token, scores the extensions by their total log-probability and keeps the $B$ best. With $B = 1$ it is greedy decoding. Here $B = 2$: after the first token it keeps `a` (0.5) and `b` (0.4) and drops `c`. After the second it has six candidates, keeps `bb` (0.28) and `ac` (0.25), and drops `ab` (0.2) and the other three. After the third it keeps `bbb` (0.196) and `aca` (0.15). The greedy text is the second of these, `aca`, because greedy committed to `a` at the first step and never saw `b`. The beam found `bbb`, the most probable of all 27 sequences.

![Beam search of width 2 on the three-token table. Each node is a prefix with its joint probability. Prefixes that stay in the beam are solid, those dropped are dashed grey, and the path of the winning sequence, b, bb, bbb, is red. The greedy path a, ac, aca is the other solid branch.](img/beam-tree.svg)

The cost is $B$ times the work at every step, $B$ copies of whatever the model keeps for each text, and $B \cdot V$ extensions to rank; and the search is still not exact. A prefix that scores badly early can be the start of the best text, and a beam of width 2 will have dropped it. In pseudocode, with the search ending at the end of the first step that leaves at least $B$ texts finished (one step can finish several of the $B$ it keeps, so more than $B$ can be returned), or when no text is left that has not finished, or after `max_len` steps:

```text
function beam_search(next_logp: function(list<int>) -> array<real>[V], B: int, end: int,
        max_len: int) -> list<(real, list<int>)>:
    require B >= 1 and max_len >= 1, "a beam of at least one and one step"
    active = [(0.0, [])]                      # (total log-probability, the tokens so far)
    finished = []
    for step in 0 .. max_len:
        cand = []
        for h in active:                      # in beam order
            lp = next_logp(h[1])
            for y in 0 .. V:                  # in increasing token id
                append (h[0] + lp[y], concat(h[1], [y])) to cand
        order = argsort([c[0] for c in cand], descending)   # stable: ties keep generation order
        active = []
        for j in 0 .. min(B, len(cand)):
            c = cand[order[j]]
            if c[1][len(c[1]) - 1] == end:
                append c to finished
            else:
                append c to active
        if len(finished) >= B or len(active) == 0:
            break
    return finished
```

Texts that end at different lengths cannot be compared by their total log-probability, because every further token multiplies by a number below 1, and the shorter text has fewer such factors. **Length normalization** divides the total by $n^{\alpha_{\mathrm{len}}}$ for a text of $n$ tokens: with $\alpha_{\mathrm{len}} = 1$ it is the mean log-probability per token, with $\alpha_{\mathrm{len}} = 0$ no correction. The next block runs the search on the order-4 model of Chapter 4, with the full stop as the end token, and ranks the finished texts both ways.

<!-- worked: decoding.beam_language -->
Beam search on the order-4 model of Chapter 4 (interpolation ratio 2), started from `<s>`, at most 12 tokens; the rule is the one in the text: the `width` best extensions by total log-probability, those ending in the full stop leave the beam, and the search ends after the first step that leaves at least `width` hypotheses finished (one step can finish several, so more than `width` may be returned), when no beam is left, or after 12 tokens. For each width, the finished hypothesis with the largest total log-probability and the one with the largest mean log-probability per token (length normalization with exponent 1).

| width | finished | best by total | total $\ln p$ | best by mean | mean $\ln p$ |
|---:|---:|---|---:|---|---:|
| 1 | 1 | `the foxes near the fox sees ball .` | -5.481 | `the foxes near the fox sees ball .` | -0.685 |
| 2 | 2 | `.` | -4.646 | `the cat near the fox sees ball .` | -0.651 |
| 4 | 5 | `.` | -4.646 | `the cat near the fox sees big fish .` | -0.614 |
| 8 | 11 | `the cat sees small ball .` | -4.299 | `the cat near the fox sees big fish .` | -0.614 |

The four best finished hypotheses at width 8 by mean log-probability:

- `the cat near the fox sees big fish .` (9 tokens, total -5.529, mean -0.614)
- `the foxes near the fox sees big fish .` (9 tokens, total -5.800, mean -0.644)
- `the cat near the fox chase small fish .` (9 tokens, total -5.807, mean -0.645)
- `the cat near the fox sees ball .` (8 tokens, total -5.210, mean -0.651)

The order-2 model of the first block with width 4 and the same rule finishes 1 hypothesis (`.`) and still has 4 unfinished beams after 12 tokens; the best of them is `the foxes near the dog near the dog near the fox sees` at total -11.721.

Mean surprisal per token under the model: the best sentence by mean of width 8, 0.6144; the greedy sentence, 0.6851; the 200 sentences sampled from the full distribution in the earlier table, 1.2315.
<!-- /worked -->

The ranking by total log-probability is a lesson in what it rewards. With a width of 2 or 4 the best text is the empty sentence, `.` alone, at $-4.646$: the model gives the full stop at the start of a text a probability of about 0.0096, and the eight-token sentences those widths find pay eight factors each and score below it. Normalized by length the empty sentence is the worst. At width 8 the best by total is a five-word sentence, and the best by mean, at every width from 4 up, is `the cat near the fox sees big fish .` with $-0.614$ per token. The best by mean is one sentence at width 1, another at width 2 and a third from width 4 on, a reminder that a wider beam is a different search and not just more of a narrower one.

The four best finished texts at width 8 are the same sentence with a word or two changed. This is the second property of beam search that matters in text generation: the hypotheses share their prefix, and a wider beam does not give four different answers, it gives one answer four times. And the last paragraph of the block is the third: the best sentence by mean has a surprisal of 0.6144 nats per token under the model, the greedy sentence 0.6851, and the 200 sentences actually sampled from the model 1.2315. The search has found a text that is far more probable per token than the texts the model writes when it is left to sample. That is what a search for the highest probability does in a task with no single right answer, and Holtzman et al. (2020) argued the same of human language: people do not choose the most probable word, so text that is the most probable is not text that reads as human. The order-2 model, with a width of 4, shows the worst case: within twelve tokens no sentence finishes except the empty one, and the four unfinished beams chain `near the` phrases, as in `the foxes near the dog near the dog near the fox sees`, since `the` follows `near` with probability 0.9039 and the model has no way of knowing it has already said it.

Why, then, is beam search standard in translation, where Sutskever, Vinyals and Le (2014) used it and Graves (2012) before them in sequence transduction? A translation has one right answer up to paraphrase. The distribution of each step is sharp, the most probable sequence is a sequence the model is confident of, and the mode is what is wanted. In an open-ended generation the distribution is wide for many steps and the mode is a blandness. Brain's text decoders do not implement beam search. The word `beam` in `crates/model/src/lsa/tree.rs` is another search, down a tree of index nodes, and the serving scheduler chooses one token at a time by the sampler of this chapter.

## The same seed, the same text?

A seeded sampler is a deterministic function, and "the same seed gives the same text" is true only if everything else the function depends on is also the same. The things it depends on are more than the seed. They are the generator itself, the number of random draws taken for each token, the order in which a draw walks along the tokens, the order of the cuts and the way ties are settled, the logits down to the last bit, and, in a program that serves several requests at once, which of them uses which generator. Volume I made the point about the generator (a seed names a stream, not a program); the next block makes it about the rest.

<!-- worked: decoding.determinism -->
Three requests A, B, C each sample 8 tokens from the bigram model of the first block, started from `<s>`, one `uniform()` per token, id-order walk, temperature 1 and no cut. The three advance in lock step (a token for the first request of the list, then for the second, then for the third, and again). Shared: one generator of seed 0x2b2465ad3c6bb03a (`decoding.shared`). Own: one generator each, seeds 0xc0f4b881ff6a0788 (`decoding.request.A`), 0x22552e5278d9a82f (`decoding.request.B`), 0x1c86f9bd835259b2 (`decoding.request.C`). The requests are served in the order A, B, C, and in the order B, C, A (every request then has a different place in the round).

| generators | request | order A, B, C | order B, C, A |  |
|---|---|---|---|---|
| one shared generator | A | `the cat near the cat near the fox` | `the dog near the fox sees ball .` | different |
| one shared generator | B | `the foxes chase big fish the cat the` | `the cat near the cat near the fox` | different |
| one shared generator | C | `the dog near the fox sees ball .` | `the foxes chase big fish the cat the` | different |
| one generator per request | A | `the dogs sees fish . near the cat` | `the dogs sees fish . near the cat` | same |
| one generator per request | B | `the foxes chase ball . foxes see big` | `the foxes chase ball . foxes see big` | same |
| one generator per request | C | `the dogs sees big ball . fox sees` | `the dogs sees big ball . fox sees` | same |

The same uniform numbers (seed 0x998eaf02e83500de, `decoding.walk`) applied to the distribution after `the`, walking the tokens in increasing id order or from the most probable to the least:

| draw | $u$ | id-order walk | walk by probability |
|---:|---:|---|---|
| 1 | 0.0949 | `cat` | `fox` |
| 2 | 0.7743 | `foxes` | `cats` |
| 3 | 0.0171 | `the` | `fox` |
| 4 | 0.8724 | `foxes` | `dogs` |
| 5 | 0.3221 | `dog` | `dog` |
| 6 | 0.3569 | `dog` | `dog` |
| 7 | 0.7444 | `dogs` | `cats` |
| 8 | 0.1148 | `cat` | `fox` |

The two walks pick different tokens for 6 of the 8 numbers.

Three requests that all use the seed of `decoding.constant` (0x04559d39a4d2b52c) produce: `the foxes see ball . big bone .`; `the foxes see ball . big bone .`; `the foxes see ball . big bone .`.
<!-- /worked -->

The first table is the one that matters for a server. When three requests share one generator and advance round by round, a request's draws come from wherever the stream happens to stand, which depends on who was served before it, so serving the same three in a different order changes the text of every one (all three differ). With a generator of its own for each request, the order is irrelevant and every request gives the same text both times. The second table is the draw itself. The same eight uniform numbers choose different tokens when the walk goes from the most probable token downward than when it goes by token id, for six of the eight; two correct samplers with the same seed can write different texts, and a seed is a promise made to one particular sampler. The last line shows what a constant seed does: three requests that all use the seed of `decoding.constant` write the same text.

Brain's scheduler is built on this reasoning. Each sequence is admitted with its own generator, `data::rng::Rng::new(seed)`, held in the record of the running sequence; the comment on the field says why: it is kept "per-sequence (not per-batch) so a `seed` reproduces exactly regardless of what else is in the batch alongside it" (`crates/model/src/serve.rs`). A sequence that is preempted to make room is requeued with its generator and resumes where it stopped, so its seed still reproduces it. A sampled token consumes exactly one draw, a greedy token none, and the walk goes from the most probable candidate down with the draw scaled by the sum of the kept weights, which is a different walk from the book's, as the second table shows is allowed.

Two stories from the history are about the seed and not the generator. Commit `ad35f7931` (September 2026) found that both chat surfaces filled in `seed: 0` and `top_k: 40` when a request left them out, so that every unseeded request decoded the same sequence; the surfaces now forward a sampling parameter only when the client sent it, and the actions draw a fresh seed. The mirror image is commit `83e2f44a3` (October 2026), recorded as knowledge #321: the server echoes the seed it drew, so that a decode can be reproduced by sending it back, and the HTTP layer read that seed as a signed 64-bit integer, so that a seed above $2^{63} - 1$, about half of all those drawn, was dropped from the reply and refused when sent back. A reproducibility feature that fails for half the seeds without an error is worse than none, since it gives a reason to believe in it; the fix carries the seed as unsigned 64-bit at both ends, and a test sends the largest one. A third story is about a seed that was accepted and then not used. The talker of an audio model draws sixteen codes for each frame, and commit `d2634ffb2` (October 2026) found that it chose fifteen of them through a greedy convenience wrapper, so only the first code of each frame ever used the caller's seed and temperature; the rest now come from the same generator under the same temperature and top-k.

One limit remains that a seed cannot repair. A seed fixes the uniform numbers and not the logits. If the logits are computed with a different order of additions (a different batch size, a different kernel, a different card), they differ in their last bits (Volume IV, Chapter 7), and a draw that falls close to a boundary between two tokens can then land on the other side. The text then agrees up to that token and not after it. A claim of exact reproducibility is a claim about the whole path from the weights to the draw.

## What choosing a token costs

None of the rules above is expensive, and the arithmetic says how little.

<!-- worked: decoding.sampling_cost -->
Operations to choose one token, against the operations of the forward pass that produced its logits. Counts: argmax is $V - 1$ comparisons; softmax and the walk are about $4V$ (an exponential, a sum, a division and a cumulative addition per token); a heap that keeps the $k$ best costs $V \log_2 k$ comparisons; a full comparison sort $V \log_2 V$. The forward pass is $2N_{\mathrm{p}} + 4LTD$ FLOPs per token at context $T$ ($N_{\mathrm{p}} = L(4D^2 + 2DF) + VD$, Chapter 10; the larger decoder has $D = 4096$, $L = 32$, $F = 4D$, $V = 128{,}000$ and $T = 4096$; GPT-2 small has $T = 1024$). Tolerance: counts are rounded down to whole operations.

| model | step | operations | percent of the forward pass |
|---|---|---:|---:|
| GPT-2 small | argmax | 50,256 | 0.018 |
| GPT-2 small | softmax and draw | 201,028 | 0.071 |
| GPT-2 small | top-k by heap, k = 40 (up to) | 267,464 | 0.094 |
| GPT-2 small | top-p by full sort | 784,865 | 0.276 |
| GPT-2 small | forward pass (2 N_p + 4 L T D) | 284,812,800 | 100 |
| a larger decoder | argmax | 127,999 | 0.001 |
| a larger decoder | softmax and draw | 512,000 | 0.003 |
| a larger decoder | top-k by heap, k = 40 (up to) | 681,206 | 0.004 |
| a larger decoder | top-p by full sort | 2,171,620 | 0.014 |
| a larger decoder | forward pass (2 N_p + 4 L T D) | 16,080,961,536 | 100 |

Bytes that cross between the card and the host per decoding step of 64 sequences, against the bytes of weights the forward pass reads (32-bit values): the whole row of logits of every sequence, or a list of the 32 best (id, logit) pairs of eight bytes each, or one 4-byte id.

| model | weights read, bytes | one logits row | 64 logits rows | 64 lists of 32 pairs | 64 ids |
|---|---:|---:|---:|---:|---:|
| GPT-2 small | 494,128,128 | 201,028 | 12,865,792 | 16,384 | 256 |
| a larger decoder | 27,866,955,776 | 512,000 | 32,768,000 | 16,384 | 256 |

Rebuilding a seen-token set from the whole context at every generated token touches $n$ tokens per step: for a prompt of 8,000 tokens and 1,000 generated tokens, 8,499,500 token visits; folding in each new token touches 1,000.
<!-- /worked -->

For GPT-2 small, whose forward pass needs 284,812,800 FLOPs per token at a context of 1,024, the worst row of the first table is a full sort of the vocabulary for top-p, 784,865 comparisons, which is 0.276 percent of the forward pass; an argmax is 0.018 percent. For a decoder with a vocabulary of 128,000 and a context of 4,096 the same sort is 0.014 percent. The count of bytes agrees: the weights the forward pass reads are $4N_{\mathrm{p}}$ bytes, 494 million for GPT-2 small, against 201,028 for one row of logits. The reason is plain. The forward pass multiplies the token's vector by every one of the $N_{\mathrm{p}}$ weights; choosing the token touches a vector of $V$ numbers a handful of times, and $V$ is a small fraction of $N_{\mathrm{p}}$.

So why is token selection engineered in brain at all? Because work and time are different things, which Volume IV made precise. A single thread scanning the logits of a row is $V$ dependent steps long however few operations they are, and a server that decodes 64 sequences at once and brings every row of logits back to the host moves $64 \times 4V$ bytes, 12,865,792 for GPT-2's vocabulary, across a link and waits for it, in front of every step, when the answer is 64 integers of 4 bytes. `crates/model/src/readout.rs` keeps the choice where the logits are: the greedy pick is the two-stage reduction of Chapter 7 of Volume IV (`argmax_part` and `argmax_final` over 256 chunks of each row, ties to the lowest index), and the top-$k$ list is `cap` rounds of an argmax followed by masking the winner, so that only `rows * cap` pairs come back, 16,384 bytes for 64 rows and a cap of 32. That chapter counted when the rounds beat a sort: for $V = 2^{17}$ they move fewer words than a radix sort of the whole row for every $k$ up to 43, so a cap of 32 is below that line and a cap of 64 above it. The count says where the rounds stop being the cheaper way to get a top-$k$, and it is a count of words moved, not of launches. A later change, commit `96d55ffc9` (October 2026), replaced the three launches per candidate, 96 launches for a cap of 32, by one launch for all candidates where the device offers the kernel (`native_topk` in the same file), keeping the old chain as the fallback with the same tie rule.

The cost of the pieces that need sorting also shows why the candidate list is attractive. A full sort for top-p is $V \log_2 V$ comparisons, 784,865 for GPT-2's vocabulary; sorting the 32 candidates already extracted is $32 \log_2 32 = 160$. The saving is real, and its price is the ceiling on the nucleus that the earlier section measured.

## Where brain draws its lines

Most of what the chapter described has a place in the tree. The settings travel in `SampleParams` and are applied by `sample_in_context_with` in `crates/model/src/serve.rs`: the processors of `crates/model/src/serve/sampler.rs` first, then `truncated_weights`, which divides by the temperature, keeps the first $k$ candidates, takes the softmax, cuts at the nucleus and then at min-p; the draw is made last. A request names its settings through `crates/apiserve/src/sampling.rs`, which refuses an out-of-range value by name (a temperature outside 0 to 2, a top-p outside 0 to 1, a top-k above 1000, a repetition penalty outside 0.01 to 4) and drops the neutral penalty values every client sends by default, so that a request that asks for nothing is decoded exactly as it always was. When a request leaves a setting out, the one the checkpoint declares is used (`crates/catalog/src/sampling_defaults.rs` reads it from `generation_config.json`), and when the checkpoint declares none, a chat model gets the fallback of commit `0e77db2b8`: temperature 0.7, top-p 0.95 and top-k 20, in the order of the sampler above (temperature, then top-k, then top-p).

The temperature of 0 is greedy and there is no draw, but greedy is not always the plain argmax of the logits: with a penalty asked for it is the argmax of the penalized logits, so such a request leaves the on-device argmax for the candidate path (`SampleParams::needs_candidates` in `crates/model/src/serve.rs`). The first step of a sampled sequence is sampled from the candidates of the prefill in the same way as every later one. Full-vocabulary samplers survive in the per-model crates, and they differ in small ways that are part of their contracts. The sampler of `crates/kronos/src/generate.rs`, which draws the tokens of a price-forecasting model, applies top-k and top-p exclusively, the first when both are asked for, and says so: "top-k short-circuits top-p (reference quirk)", with a test that pins it, because that is what the model it ports does and a port that did otherwise would not reproduce it. It cuts the nucleus where the running sum exceeds $p$, and the candidate-list sampler where it reaches $p$; the two differ only when a partial sum equals the threshold exactly.

A sampler also meets the numbers a model should never produce. Commit `072148692` (August 2026) found a sampler that panicked as soon as one logit was NaN, because the top-k sort called `partial_cmp(...).unwrap()` and a NaN has no order. The fix is an ordering in which every NaN is the least preferred value, applied in the sorts and in the argmax. The argmax shows the point in two lines: a loop that keeps the index of the largest value and replaces it when `s[i] > s[best]`, started at index 0, returns 0 for `[NaN, 1, 5]`, since every comparison with the NaN is false, and the sampler would pick the one value that is not a number. The commit's tests, `nan_logit_does_not_panic_and_is_never_preferred` and `argmax_skips_nan_regardless_of_position`, put the NaN at the front, the middle and the end.

The processors run on the candidate list, and the module comment of `crates/model/src/serve/sampler.rs` states when that is exact and when it is not. It is exact while the best processed candidate is still at or above the lowest unprocessed candidate logit, because every token outside the list started at or below that. It is not exact when the penalties push every candidate below a token outside the list, as a heavily repeated context does, nor when a request asks for a negative presence or frequency penalty or a repetition penalty below 1, which raise logits: `crates/apiserve/src/sampling.rs` accepts presence and frequency penalties between $-2$ and 2 and a repetition penalty from 0.01, and a raised token outside the list is never seen. A small case shows the second kind: with logits 10, 9.9 and 9.8, a list of the best two and a repetition penalty of 2, the third token is the best after the penalty and is not in the list. It is the same trade as the nucleus ceiling, a candidate list standing in for the vocabulary. The chapter did not measure how often it changes a token, and with 32 or 64 candidates it is likely to be seldom.

## Check your understanding

1. A sampler applies temperature 2.0 and then top-p 0.9; a second applies top-p 0.9 to the unscaled distribution and then temperature 2.0. Which of them can put more tokens in the set it draws from, and why? Use the table of the order of the steps.
2. "To make a model repeat itself less, divide the logits of the tokens already seen by 1.2." Find the error, and say for which tokens the effect is the opposite of the intended one.
3. A model's distribution is $(0.4, 0.4, 0.1, 0.1)$. What do top-k with $k = 1$, top-p with $p_{\mathrm{top}} = 0.5$ and greedy decoding each give, and which of the three consumes a random number?
4. A request logged `seed = 17` and its replay with seed 17 produced a different text. Give three causes from this chapter, one of them in the sampler and one outside it.
5. A server returns a log-probability of $-0.02$ for a sampled token to which the model's own softmax gives 0.3. How can both be right?
6. A width-5 beam search returns a text whose total log-probability is lower than the greedy text's. Is the implementation wrong?
7. A sampler is written to set the dropped logits to $-\infty$ and take one softmax, and another to drop terms and divide by what is left. Why do they give the same distribution, and what must be true of the probabilities the second one starts from?
8. A team removes "unneeded" random draws by skipping the draw whenever only one token survives the cuts, which gives the same texts for a single request. What does it change for a request that shares a generator with others?

## Summary

A model hands over a distribution and a decoder chooses a token from it, and the choice is a pipeline whose parts are individually simple and whose order is part of its meaning. Taking the largest probability at each step is greedy decoding: deterministic, cheap, and forced into a cycle by the pigeonhole argument whenever the model reads only a bounded window. Drawing from the whole distribution is faithful, and a tail of individually small probabilities makes a long text certain to contain impossible tokens. Dividing the logits by a temperature moves the probability along the ranking and raises the entropy monotonically, with the derivative $\mathrm{Var}_q(z)/\tau^3$, but changes head and tail together; top-k, top-p and min-p cut the tail and leave the head, with a count, a cumulative mass and a ratio to the largest as three different notions of where the tail begins. Penalties on seen tokens fight loops and fight grammar, the multiplicative one with a sign rule and a dependence on the zero of the logits, the additive ones in the units of the logits and therefore on the temperature. Log-probabilities turn products into sums, and each belongs to a stated distribution. Beam search keeps $B$ texts and finds more probable ones than greedy decoding, in tasks with one right answer, and in open generation finds text far more predictable than the model's own samples. Reproducing a seeded text needs the whole procedure to be the same, a generator of its own for each request included. And the cost of choosing is negligible in operations and bytes against the forward pass and not negligible in depth or in round trips, which is why brain does it where the logits are.

The limits are those of the toy. The probabilities come from a counting model and a grammar judges its sentences, so the percentages of the quality table are for that model and that language; the loop, the tail, the order effects and the failure of the repetition penalty are mechanisms, and their sizes would be different elsewhere. The counts of the cost section are operations and bytes and say nothing about the time on a machine. The chapter has taken the model's next-token distribution as given, computed once per token. The next chapter asks what it costs to compute it, and what the model can be made to remember so that the past is not recomputed at every step.

## References

- A. Fan, M. Lewis and Y. Dauphin, "Hierarchical neural story generation", 2018.
- A. Holtzman, J. Buys, L. Du, M. Forbes and Y. Choi, "The curious case of neural text degeneration", 2020.
- N. S. Keskar, B. McCann, L. R. Varshney, C. Xiong and R. Socher, "CTRL: a conditional transformer language model for controllable generation", 2019.
- A. Graves, "Sequence transduction with recurrent neural networks", 2012.
- I. Sutskever, O. Vinyals and Q. V. Le, "Sequence to sequence learning with neural networks", 2014.

## Exercises

1. **[Calculate]** Four tokens have logits $(3, 1, 0, -2)$. Give the probabilities at $\tau = 1$ and at $\tau = 0.5$ and their entropies. Then give the temperature at which the largest token has probability 0.9.
2. **[Calculate]** A distribution is $(0.40, 0.30, 0.15, 0.10, 0.05)$. Give the renormalized distribution after top-k with $k = 3$, after top-p with $p_{\mathrm{top}} = 0.8$ and after min-p with $r_{\min} = 0.5$, and say which tokens each keeps.
3. **[Calculate]** Logits are $(2.0, -1.0, 0.5, 0.0)$ and tokens 0 and 1 have been seen. Apply a repetition penalty of 1.25, then a presence penalty of 0.5 to token 0 only (generated once), and give the final logits. Repeat with the presence penalty first, and say whether the order changes the answer.
4. **[Calculate]** Two finished texts have total log-probabilities $-6.0$ over 4 tokens and $-7.2$ over 6 tokens. Which wins under length normalization with $\alpha_{\mathrm{len}} = 0$, 0.5 and 1?
5. **[Calculate]** For a vocabulary of 32,000 tokens give the number of comparisons of a full sort, of a heap that keeps the best 40, and of an argmax, and each as a share of a forward pass of $2 \cdot 3 \times 10^9$ FLOPs.
6. **[Calculate]** For the three-token table of the beam section, run beam search of width 3 by hand, listing the beam after each step, and say whether it returns the best of the 27 sequences.
7. **[Implement]** Reproduce `decoding.toy_distribution` from the 40 sentences of Chapter 4, `decoding.sampling_junk` (the generator, the one `uniform()` per draw and the id-order walk are stated there) and `decoding.temperature_table`, including the central difference of the entropy.
8. **[Implement]** Reproduce `decoding.truncation_table` and `decoding.filter_order`: the three cuts on both contexts, the four orders of the steps and the table of penalties before and after the temperature.
9. **[Implement]** Reproduce `decoding.decoding_quality`: for each setting one generator named in the text, 200 sentences, one draw per token and the id-order walk, then the share accepted, the distinct sentences and the mean surprisal.
10. **[Implement]** Reproduce `decoding.beam_hand` and `decoding.beam_language` with the pseudocode of the beam section, and `decoding.repetition_rule` for the greedy runs with the penalty.
11. **[Investigate]** Apply the repetition penalty not to the whole context but to the last 8 tokens only, with the offsets and factors of `decoding.repetition_rule`. Does the penalty still break the grammar at an offset of 8? The chapter did not run this.
12. **[Investigate]** Search random tables of 4 tokens and 5 steps for one where a beam of width 2 returns a text less probable than the greedy one. Is it rare? The chapter did not run this.

## Key terms

- **greedy decoding**: choosing, at every step, the token of largest probability (the first on a tie), so that the text is a deterministic function of the model and the prompt and, for a model that reads a bounded window, eventually repeats. <!-- concept: greedy-decoding -->
- **temperature**: a positive number $\tau$ by which the logits are divided before the softmax; low values sharpen the distribution toward the largest token, high values flatten it toward uniform, and the entropy rises with it. <!-- concept: temperature -->
- **top-k sampling**: keeping only the $k$ most probable tokens, setting the rest to zero and renormalizing before the draw. <!-- concept: top-k-sampling -->
- **top-p sampling**: keeping the smallest set of most probable tokens whose cumulative probability reaches a threshold, also called nucleus sampling, so that the number of tokens kept follows how spread out the distribution is. <!-- concept: top-p-sampling -->
- **repetition penalty**: a factor applied to the logits of tokens already in the text, dividing a positive logit and multiplying a negative one, to make repeats less likely; its additive relatives subtract an amount per presence or per occurrence. <!-- concept: repetition-penalty -->
- **beam search**: keeping the $B$ partial texts of highest total log-probability at every step instead of one, to find a more probable whole text than greedy decoding does, at $B$ times the work. <!-- concept: beam-search -->
- **log-probabilities**: the logarithms of probabilities, at most 0, which a decoder reports and adds instead of multiplying probabilities, each stated for the distribution it belongs to: the model's or the sampler's. <!-- concept: log-probabilities -->

## Answers

**Recall questions.**

1. $e^2 = 7.389$ and $e^0 = 1$, so the probabilities are $7.389/9.389 = 0.787$, $0.106$ and $0.106$. Adding 5 to every logit multiplies numerator and denominator by $e^5$, so nothing changes.
2. The cumulative sums are 0.5, 0.8 and 1.0; 0.83 is at least 0.8 and below 1.0, so the walk returns the third token.
3. $0.5 \times 0.4 \times 0.1 = 0.02$.
4. $-(0.5 \ln 0.5 + 2 \cdot 0.25 \ln 0.25) = 0.3466 + 0.6931 = 1.0397$ nats, which is 1.5 bits.
5. $0.1^n < 1.18 \times 10^{-38}$ when $n > 37.9$, so from $n = 38$ on. The product is then subnormal; it is exactly zero a few tokens later.
6. A different number or order of draws before the one that is printed (an extra call, a different loop order), and a different generator or conversion from the raw bits to a number (an algorithm or a rounding that differs). A different seed derivation or a different platform arithmetic would also do.

**Check your understanding.**

1. The first can. Temperature 2.0 first flattens the distribution and the nucleus of 0.9 of the flattened distribution is wide; top-p first takes the nucleus of the unscaled distribution (six tokens) and the temperature only reshapes that set, which cannot gain a token. The table of the order of the steps shows the same effect at a temperature of 1.5: 7 tokens when the temperature comes first against 6 when it comes last, and the third order, with top-p before top-k, keeps all 10.
2. Dividing a negative logit by 1.2 moves it toward zero, so it rises. A seen token with a negative logit becomes more likely, the opposite of the intent, and the rule has to divide positive logits and multiply negative ones. The effect also depends on the zero of the logits, which the softmax does not fix.
3. Greedy decoding chooses the first of the two tied tokens by the lower-id rule, with no draw. Top-k with $k = 1$ keeps the same token and chooses it; brain's sampler still consumes one random number for it, because only the temperature-0 case returns before drawing. Top-p with 0.5 keeps the smallest set that reaches 0.5, which is both tied tokens (0.4 is not enough), and chooses between them with one draw.
4. In the sampler: a different walk order, a different number of draws per token, or a different order of the cuts. Outside it: a generator shared among requests, so that the stream depends on who was served before, or logits that differ in their last bits between runs because the batch or the kernel changed. (A server that read the seed back as a signed integer and refused or dropped it would also fail, as commit `83e2f44a3` found.)
5. The server reports the log-probability under the distribution the token was drawn from, after temperature and cuts. If the cuts left only a few tokens, or a low temperature sharpened the distribution, the token holds 0.98 of the sampled distribution, $\ln 0.98 = -0.02$, while the model's own softmax gives it 0.3.
6. Not necessarily. Beam search is not an exact search: a prefix that the beam drops can be the start of the greedy text, and a width of 5 does not contain the greedy beam of width 1. It can return a text less probable than greedy decoding's, though it usually does not.
7. With $S$ the set kept, $p_i/\sum_{j\in S}p_j = e^{z_i}/\sum_{j\in S}e^{z_j}$, because the common normalizer $Z$ of the softmax cancels. The second sampler must start from probabilities that are the full softmax of the same logits (all of them, not a list that is already a part of it), or the divisor it uses is not the sum over $S$ of the true probabilities.
8. With one generator per request nothing changes for the others. With a shared generator, skipping a draw shifts every later draw of every request, so the texts of the other requests change with a change in the cuts of one of them.

**Exercises.**

1. At $\tau = 1$: $e^3, e^1, e^0, e^{-2} = 20.086, 2.718, 1, 0.135$; sum 23.939; probabilities 0.8390, 0.1135, 0.0418, 0.0057, entropy 0.5562 nats. At $\tau = 0.5$: $e^6, e^2, e^0, e^{-4} = 403.43, 7.389, 1, 0.0183$; sum 411.84; probabilities 0.9796, 0.0179, 0.0024, 0.00004, entropy 0.1074 nats. For the largest token to have probability 0.9: $e^{3/\tau}/(e^{3/\tau} + e^{1/\tau} + 1 + e^{-2/\tau}) = 0.9$; solved numerically, $\tau \approx 0.81$ (at $\tau = 1$ it is 0.839, so the temperature must fall).
2. Top-k with $k = 3$ keeps the first three with mass 0.85: $(0.4706, 0.3529, 0.1765)$. Top-p 0.8: cumulative sums 0.40, 0.70, 0.85, so three tokens are kept and the distribution is the same as for $k = 3$. Min-p 0.5 keeps tokens with probability at least 0.20, the first two, with mass 0.70: $(0.5714, 0.4286)$.
3. Repetition penalty 1.25: token 0 has $2.0/1.25 = 1.6$ and token 1 has $-1.0 \cdot 1.25 = -1.25$; the others stay. Then the presence penalty on token 0 gives $1.1$. The logits are $(1.1, -1.25, 0.5, 0.0)$. With the presence penalty first, token 0 is $1.5$ and the repetition penalty then gives $1.2$, so $(1.2, -1.25, 0.5, 0.0)$: the order changes the answer, because the additive penalty is not scaled by the multiplicative one.
4. $\alpha_{\mathrm{len}} = 0$: $-6.0$ against $-7.2$, so the first. $\alpha_{\mathrm{len}} = 0.5$: $-6.0/2 = -3.0$ and $-7.2/\sqrt{6} = -2.94$, so the second. $\alpha_{\mathrm{len}} = 1$: $-1.5$ against $-1.2$, so the second.
5. A full sort: $32{,}000 \log_2 32{,}000 \approx 478{,}900$ comparisons. A heap of 40: $32{,}000 \log_2 40 \approx 170{,}300$. An argmax: 31,999. Against $6 \times 10^9$ the shares are $8.0 \times 10^{-5}$, $2.8 \times 10^{-5}$ and $5.3 \times 10^{-6}$, that is 0.008, 0.003 and 0.0005 percent.
6. Width 3 keeps all three one-token prefixes, `a` 0.5, `b` 0.4 and `c` 0.1. At step 2 the nine candidates are `bb` 0.28, `ac` 0.25, `ab` 0.20, `bc` 0.08, `ca` 0.06, `aa` 0.05, `ba` 0.04, `cc` 0.03 and `cb` 0.01, and it keeps `bb`, `ac` and `ab`. At step 3 the best three of the extensions are `bbb` 0.196, `aca` 0.15 and `abb` 0.14. It returns `bbb`, the best of the 27 sequences, which the width-2 beam also found.
7. The distribution after `the` of the table, the draws of the table (0.3933 is `fox` and so on) and counts 397, 390, 364, 317, 196, 155 for the six nouns, 181 for the rest; the entropy of the temperature table, and the central difference 0.787851 at $\tau = 1$ and 0.259461 at $\tau = 2$.
8. The kept counts of the table of cuts after `the` and `near`, the three orders giving 7, 6 and 10 tokens with the largest probabilities 0.1967, 0.2064 and 0.1819, the agreement to $10^{-16}$ of the renormalized cut and the softmax of the kept logits, and the pairs 0.0939 and 0.0939 against 0.0676 and 0.1510.
9. The percentages of the table: 39.0, 64.0, 11.5, 68.0, 69.0, 64.5, 79.5 and 19.5, and the greedy sentence `the foxes near the fox sees ball .`, rejected by the grammar.
10. The table of the three-token language (the candidates of each step), and the finished lists of the table of widths (for width 1: `the foxes near the fox sees ball .` at $-5.481$; for width 8 by mean: `the cat near the fox sees big fish .` at $-0.614$). For the repetition table, the counts 5/5 for every offset at a factor of 1 and 1/5 at offset 8 with factor 1.5.
11. An investigation. The chapter did not run it. The expectation to test is that a window of eight tokens still catches a loop of period six, and that `the` is no longer penalized once its last occurrence is more than eight tokens back.
12. An investigation. The chapter did not run it. A width-2 beam can discard the greedy text's prefix when two other prefixes beat it at some step, so the answer is expected to be yes, and rare.
