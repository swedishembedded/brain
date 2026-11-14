<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# From Hypothesis to Paper

*How does an idea become a result others can trust, check and build on?*

Here is a test that a claim false by a hair passes almost half the time, written in the form that appears in real protocols. The claim: a new feed-forward layer is no worse than a standard one by more than 0.02 bits per byte of held-out text (the model's mean code length per byte of text, in bits, from Volume V's tokenization chapter; lower is better, so "worse" means a larger number). The rule: train three models of each kind, and reject the claim if the mean of the new layer's three scores exceeds the mean of the standard layer's by more than 0.02. Suppose the truth sits exactly on the line, the new layer worse by 0.02. Then the difference of the two means is centred on 0.02, it falls on either side with equal chance, and the rule accepts the claim in half of all experiments. Move the truth a little past the line, to 0.025, where the claim is false, and the rule still accepts it often. How often depends on how much a model's score moves from one training run to the next, so give that a size: 0.005 bits per byte, about the spread of the real comparison recomputed further on. The table puts the rule beside a second one, which accepts the claim only when the whole 95 percent interval of the difference lies at or below 0.02.

<!-- worked: research.rule_pass_rates -->
The claim: the candidate's bits per byte exceed the baseline's by at most 0.02 (the margin). A simulated experiment trains n runs of each arm; run i of the candidate scores delta + 0.005 z and run i of the baseline 0.005 z, z a standard normal draw. Generator `research.rule_pass_rates` (0xf1ba86779ee39801), one generator for the whole table, `rng.normal()` (Box-Muller, the sine partner held for the next call) in this order: n = 3 before n = 8, the true differences top to bottom, 10,000 experiments per cell, and in each experiment the n candidate draws before the n baseline draws. The mean rule passes when the difference of the two means is at most 0.02; in brackets, its exact chance Phi((0.02 - delta) / (0.005 sqrt(2 / n))). The interval rule passes when the upper end of the Welch 95% interval of the difference is at most 0.02, evaluated as t_cdf_real((0.02 - d) / se, nu) >= 0.975 with d <= 0.02, which is the same condition. Shares are counts divided by 10,000, printed exactly; their Monte Carlo standard error is at most 0.005.

| true difference delta | claim is | mean rule, n = 3 | interval rule, n = 3 | mean rule, n = 8 | interval rule, n = 8 |
|---|---:|---:|---:|---:|---:|
| -0.010 | true | 1.0000 (1.0000) | 0.9933 | 1.0000 (1.0000) | 1.0000 |
| -0.005 | true | 1.0000 (1.0000) | 0.9729 | 1.0000 (1.0000) | 1.0000 |
| 0.000 | true | 1.0000 (1.0000) | 0.8823 | 1.0000 (1.0000) | 1.0000 |
| +0.005 | true | 1.0000 (0.9999) | 0.6693 | 1.0000 (1.0000) | 0.9996 |
| +0.010 | true | 0.9927 (0.9928) | 0.3644 | 1.0000 (1.0000) | 0.9593 |
| +0.015 | true | 0.8927 (0.8897) | 0.1171 | 0.9777 (0.9772) | 0.4595 |
| +0.020 | true | 0.4987 (0.5000) | 0.0171 | 0.4963 (0.5000) | 0.0218 |
| +0.025 | false | 0.1086 (0.1103) | 0.0011 | 0.0228 (0.0228) | 0.0001 |
| +0.030 | false | 0.0079 (0.0072) | 0.0000 | 0.0000 (0.0000) | 0.0000 |
<!-- /worked -->

Read down the column of the mean rule for three runs. Where the claim is true it passes nearly always, which is pleasant, and where the claim is false by a quarter of the margin (true difference $+0.025$) it still passes in about 11 percent of experiments. With eight runs the mean rule is sharper but its value at the line does not move: at $\delta = +0.02$ it passes in half the experiments for any number of runs, because the rule compares a noisy estimate with the very boundary of the claim. The interval rule behaves differently. It passes a claim false by $0.005$ in 11 of 10,000 experiments with three runs, and it pays for that caution with its power: a true difference of zero, where the new layer is as good as the old, passes only 88 percent of the time with three runs, and a true $+0.01$ passes only 36 percent. Both rules are honest arithmetic. They are different statements about what result would count against the claim, and only one of them says how often a false claim survives.

An idea becomes a result others can build on in three steps: it is turned into a statement that a result can contradict, the result is recorded so that someone else can recompute it, rerun it and repeat it, and the whole is written up so that every number in the write-up leads back to the run that produced it. Four ideas from earlier volumes carry the weight, and each is worth recalling before it is needed.

1. (Volume VI, [Chapter 9](../v06-language-models/09-spectral-lm.md), architecture hypothesis testing.) The spectral mixer was compared with two baselines, one with its parameter count and one with its arithmetic. Why two, and what does a baseline that differs from the candidate in exactly one thing allow a reader to conclude?
2. (Volume II, [Chapter 8](../v02-learning/08-experiments.md), ablations.) Removing component A from a full system costs 5 points and removing B costs 1, yet removing both costs 3. What does that say about the components, and how should the result be reported?
3. (Volume II, Chapter 8, reporting uncertainty.) Name five things a result must be reported with besides its mean.
4. (Volume I, [Chapter 12](../v01-mathematics/12-statistics.md), Student's t.) A 95 percent interval from the mean of three numbers uses the multiplier 4.303 where one from many numbers uses 1.96. Where does the larger multiplier come from, and how many degrees of freedom does it have?

## A claim that names what would refute it

A hope ("the spectral mixer is a good feed-forward layer") cannot be shown wrong, because no outcome of any run is excluded by it. A claim that can be tested has to say, before the run, which outcomes would count against it, and the opening table shows that "which outcomes" is not enough: it also matters how often those outcomes occur when the claim is false. Popper made the ability to be refuted the mark of a scientific statement (Popper, 1959); for an experiment it has to be made concrete. A **falsifiable hypothesis** is a claim stated before the data together with five things that make it decidable. It names the quantity measured (bits per byte on the validation documents, not "quality"), the comparison (the candidate minus a named baseline, trained identically), the margin that matters ($0.02$ bits per byte, chosen for what the reader cares about, not for what the data turn out to show), the decision rule (which interval or statistic is compared with the margin, and how), and the sample (how many runs of each arm, with which seeds). Given all five, every possible set of results is assigned in advance to one of the verdicts "supported", "refuted" or, for rules that allow it, "undecided", and so some possible results refute the claim.

The fifth part is what turns the rule into something whose behaviour can be computed. Let $\delta$ be the true difference between the arms (the mean of the candidate's score over all possible seeds, minus the baseline's), $\Delta$ the margin, $\sigma$ the standard deviation of one run's score about its arm's mean, and $n$ the number of runs per arm. The observed difference of the two means, $d = \bar a - \bar b$, has standard deviation $\sigma\sqrt{2/n}$ when the runs are independent and the spreads equal. The mean rule supports the claim when $d \le \Delta$, and if $d$ is Gaussian the chance of that is

$$
P_{\text{mean}}(\delta) = \Phi\!\left(\frac{\Delta - \delta}{\sigma\sqrt{2/n}}\right),
$$

with $\Phi$ the standard Gaussian distribution function of Volume I. At $\delta = \Delta$ the argument is zero and $\Phi(0) = \tfrac12$ for every $n$, which is the one-half share of the opening table. With $\Delta = 0.02$, $\sigma = 0.005$, $n = 3$ and $\delta = 0.025$ the argument is $-0.005/0.004082 = -1.2247$ and $\Phi(-1.2247) = 0.1103$, the bracketed value in the table, beside the simulated share 0.1086 that agrees within its Monte Carlo error. The function $P(\delta)$ that gives a rule's chance of saying "supported" for each true difference is the rule's **operating characteristic**, a name borrowed from quality control; the figure draws it for both rules.

![Four curves of the chance of a verdict of supported against the true difference, for the mean rule and the interval rule at three and at eight runs per arm, with a vertical line at the margin 0.02; both mean-rule curves cross one half at the line, the interval-rule curves are already near zero there.](img/research-rule-pass.svg)

Two features of a good rule show in the picture. Its curve should be low to the right of the margin, where the claim is false, so that a false claim rarely survives; the interval rule's curve is at most 0.022 at the margin itself and near zero beyond it, and the mean rule's is one half at the margin. And its curve should be high to the left, so that a true claim is usually confirmed; that is the power, and it grows with $n$. A rule that is careful at the margin pays for it with power, and only more runs, or a smaller spread, buy both. Choosing $n$ is therefore part of the hypothesis: a protocol that registers three runs per arm and the interval rule is registering a test that confirms an equally good candidate 88 times in 100, and a reader who knows the rule can work that out before any model is trained.

The interval rule needs an interval, and when the two arms have different spreads (the real comparison below has spreads that differ by a factor of nine) an interval built on one common spread for both arms is not the right one. Welch's interval treats each arm's variance separately. With $s_a^2$ and $s_b^2$ the two sample variances (divisor $n - 1$) of $n_a$ and $n_b$ runs, the standard error of $d$ is $\mathrm{SE} = \sqrt{s_a^2/n_a + s_b^2/n_b}$, and the difference divided by it follows approximately Student's t with

$$
\nu = \frac{\left(s_a^2/n_a + s_b^2/n_b\right)^2}{\dfrac{(s_a^2/n_a)^2}{n_a - 1} + \dfrac{(s_b^2/n_b)^2}{n_b - 1}}
$$

degrees of freedom (Welch, 1947; the same formula is Satterthwaite's earlier approximation for a combination of variance estimates, 1946), a number that is generally not an integer and lies between the smaller of $n_a - 1$ and $n_b - 1$ and their sum. The 95 percent interval is $d \pm t_{0.975}(\nu)\,\mathrm{SE}$, with $t_{0.975}(\nu)$ the point below which Student's t with $\nu$ degrees of freedom puts 97.5 percent of its mass. In the book's pseudocode the interval and three decision rules read as follows.

```text
function welch(a: array<real>[na], b: array<real>[nb]) -> (real, real, real, real):
    require na >= 2 and nb >= 2
    ma = mean(a)
    mb = mean(b)
    va = 0.0
    for i in 0 .. na:
        va = va + (a[i] - ma) * (a[i] - ma)
    va = va / (na - 1) / na                       # s_a^2 / n_a
    vb = 0.0
    for i in 0 .. nb:
        vb = vb + (b[i] - mb) * (b[i] - mb)
    vb = vb / (nb - 1) / nb                       # s_b^2 / n_b
    require va + vb > 0.0
    d = ma - mb
    se = sqrt(va + vb)
    nu = (va + vb) ^ 2 / (va ^ 2 / (na - 1) + vb ^ 2 / (nb - 1))
    k = t_quantile_real(0.975, nu)
    return (d, d - k * se, d + k * se, nu)

function supported(rule: text, a: array<real>[na], b: array<real>[nb], margin: real) -> bool:
    (d, lo, hi, nu) = welch(a, b)
    if rule == "mean":
        return d <= margin                        # the mean rule
    else if rule == "interval":
        return hi <= margin                       # the interval rule
    else:
        require rule == "superiority"
        return hi < 0.0                           # the whole interval favours a
```

Here `mean` is the function defined in the arrays section of the pseudocode page, and the sums run in increasing `i`; `t_quantile_real` and the distribution function `t_cdf_real` it inverts are defined, for a real number of degrees of freedom, in the pseudocode page. The simulation of the opening table evaluates the interval rule through the equivalent condition `d <= margin and t_cdf_real((margin - d) / se, nu) >= 0.975`, which needs no search for a quantile in each of its 180,000 experiments.

## Four registered hypotheses and how each could fail

The paper in `papers/spectral-lm` registers four hypotheses before its comparison runs, in its protocol section, and they make a useful exhibit because they fail in four different ways. The comparison trains decoders of six blocks and width 256 on a corpus of children's stories, with a spectral mixer of $C = 24$ channels, a SwiGLU feed-forward network of hidden width 24 with exactly its parameter count, and a SwiGLU of hidden width 1024 with about its arithmetic (Volume VI, Chapter 9 built the architecture; here only the claims matter).

H1, the primary hypothesis, says the spectral model reaches a mean validation bits per byte "no worse than" the parameter-matched control "by more than 0.02", and the protocol states that "it is rejected if the mean difference over seeds exceeds 0.02". It has every part of a falsifiable hypothesis: a quantity, a comparison, a margin, a rule and a sample of three seeds per model. Its rule is the mean rule of the opening, so its operating characteristic is the one-half curve at the margin. The paper reports the verdict by this registered rule and adds that the claim also holds under the stricter reading, the interval rule. Both verdicts are recomputed from the records below, and both say "supported".

H2 says the spectral model "does not reach the quality" of the arithmetic-matched control, and that "the gap is reported with its uncertainty either way". It is directional and has no margin: any outcome in which the spectral model's mean is lower would refute it, and a gap of any size in the predicted direction confirms it. Falsifiable, then, but at its weakest, because it can only be told apart from "the two are equal" by its interval, which the protocol promises to report. The interval below is $+0.0882$ to $+0.0937$ bits per byte, far from zero, so the weakness costs nothing here; it would have cost a great deal had the gap been small.

H3 is a claim about a pretrained model's hidden states: that the lowest $m$ modes of a Fourier basis retain "no more energy, and no more next-token information, than $m$ random orthogonal directions, and less than the top $m$ principal components". The comparison with a single random draw has no margin, so a Fourier value larger than the random one by any amount refutes the first clause, and if the Fourier basis and a random one really are equivalent, a single draw lands below the Fourier value about as often as above it. The clause is refuted, as written, by noise of the size the claim's intent would tolerate. That is what happened: the paper reports that the energy clause "fails narrowly", the Fourier basis keeping a fraction of the total energy larger by 0.013 to 0.021 than the one random draw at every width tested (four widths, from the spectrum record of commit `5badba61d`), and then discusses the result in substance in a separate paragraph (commit `cf0f07e67`). A margin, and several random draws to estimate the random basis's own spread, would have made the clause say what was meant.

H4 says the fast-transform path computes the same function as the dense product within single-precision tolerance and "is faster at every evaluated" shape, five shapes listed in the driver. A universal claim over a finite listed set is the cleanest kind there is: one counterexample refutes it and nothing else can. The records of the five shapes contain one, and the paper reports H4 rejected (commit `a0466345d`; the paper's table generator computes the verdict from the records).

The shape and the size of the measured gap are told in Volume VI, [Chapter 9](../v06-language-models/09-spectral-lm.md); the hypothesis's verdict depends only on the sign of the gap at that one shape, which the records give.

Of the four, H1 and H4 decide themselves from their records by a rule a program can apply, H2 needs its interval to mean anything, and H3 needed prose to rescue its intent. The lesson is about writing, not about statistics: a hypothesis is falsifiable only to the extent that its sentence fixes the rule.

## Writing the test down before the data

A hypothesis formed after the results are in is not a test of anything; it is a description of the results that happens to fit them. The remedy is a **registration**, a protocol written and dated before the runs that fixes what will be measured and how it will be decided, so that the analysis cannot drift towards the outcome. In a repository the date is a commit. The four hypotheses entered the spectral-lm paper in its first commit, `d643c87a5`; its protocol fixes the corpus and the metric, the four model configurations with their exact parameter counts, the training budget (4,096 steps of 16 sequences of 512 tokens, 33,554,432 tokens per run), the learning-rate selection rule, the number of seeds, the scoring rule for validation documents and the four hypotheses. Commit `ed5ac8f30` added the configurations and the measurement driver, and the comparison records arrived later, in commit `609bb62bc`; the order of the first commit and the last is the registration.

A protocol is sometimes wrong, and then it is amended. What matters is when, and whether the amendment is visible. The spectral-lm protocol lists its exceptions at the top: the byte limit on scored documents was lowered from 2,000 to 1,500 before any run was scored (commit `ee7d5fa80`, because a 2,000-byte document can exceed the 512-token context at the corpus's tokenizer); the training budget of the pretrained reference model was set from a throughput probe while the primary runs were in progress but before its own run (commit `08b7bd453`); and an extension with more seeds was added after the registered analysis and is labelled post hoc wherever it appears (commit `c8f490b98`). An amendment made before the data it affects is as good as the original protocol. One made after is an exploratory analysis (Volume VI, Chapter 9 tells the case in which the first write-up of H1 used a stricter rule than the registered one, and commit `b68af93c1` restored the registered rule), worth doing and worth reporting, as long as it is reported as exploratory and the registered analysis is reported too.

The registration also has to fix the decisions that the experiment itself makes along the way. The learning rate of each model family was chosen by a registered rule, the lowest final training loss of a 300-step run at each of three rates, and the driver's comparison plan does not take the rate as an argument: it reads the selection runs' records from the repository and applies the rule (`chosen_lr` in `papers/spectral-lm/scripts/run.py`). The records of the selection were committed in commit `90a318a78`, and the nine comparison records name that commit as the one they ran at. So the rate each comparison run used is a function of committed data at a named commit, and a reader can check that the rule chose it rather than an analyst. The paper reports the selection table and states that the spectral model's selection losses were not monotone in the rate, "which one seed per rate cannot separate from seed variation; the rule was applied as registered".

## What "the same validation text" means

Before a result can be recomputed, everyone involved has to be talking about the same data, and the names people use for data are not identities. "The validation split" of a corpus can arrive with different line endings after a trip through another operating system, be filtered by a length rule, or differ by one character after a correction upstream. Each of these changes the number an evaluation reports. Here is a tiny version: a model of bytes fitted on three short lines of text, scored on three validation documents in four forms that a person would describe with the same words.

<!-- worked: research.data_identity -->
The model is the add-one byte model of the training text 'the cat sat on the mat.', 'the dog sat on the log.', 'a cat and a dog sat.', each line ending in LF (byte 10), 69 bytes: p(b) = (count of b + 1) / (69 + 256) for each of the 256 byte values. The validation documents are 'the cat sat on the log.', 'a dog sat on the mat and the cat sat on the dog.' and 'the mat sat.'. Bits per byte = the sum over the scored bytes of -log2 p(b), divided by their number (log2 is the library function; four decimals). The checksum is the first 16 hexadecimal digits of the SHA-256 of the scored bytes.

| validation text | documents | bytes | SHA-256 (first 16 hex digits) | bits per byte |
|---|---:|---:|---:|---:|
| as published (each document ends in LF) | 3 | 86 | 9b821ce394648cb6 | 5.4699 |
| the same documents ending in CR LF | 3 | 89 | 36c864058fb3b60e | 5.5668 |
| documents of at most 30 bytes only | 2 | 37 | 2e0273d76b808f7f | 5.5283 |
| first 'log' changed to 'lot' | 3 | 86 | 7535e5f412e59d40 | 5.4530 |
<!-- /worked -->

The second row has three more bytes, one carriage return (byte 13) per document, a byte the training text never contained, and its score rises from 5.4699 to 5.5668 bits per byte. The third keeps the two documents of at most 30 bytes and scores 5.5283; the fourth changes one letter and scores 5.4530. Bits per byte is the summed negative log-likelihood in bits divided by the number of bytes scored, so both the numerator and the denominator move with the bytes, and a real model behaves the same way. None of the four can be told from the others by its description. Each is told from the others by its checksum: the SHA-256 digest, a cryptographic hash from the standard libraries of most languages, changes completely when a single byte changes, so two people who compare sixteen hexadecimal digits know whether they evaluated the same bytes.

The spectral-lm paper treats its corpus this way. The corpus record `papers/spectral-lm/data/corpus/tinystories.json` names the source with its pinned revision and licence, the command that prepared the files, the SHA-256 of the raw training and validation files, and the counts of the prepared corpus (27,630 validation documents). The scoring rule is part of the same identity: the evaluation scores only documents of at most 1,500 UTF-8 bytes, 26,439 of them, and the rule is stated on bytes rather than tokens so that a model with another tokenizer scores exactly the same documents. Every comparison record repeats the number of documents and bytes it scored, which is how a reader checks that nine runs were measured on one text.

## A record for every run

Recomputing a result needs more than the data. It needs the code at the state that ran, the exact command with its arguments, the seed, the device, and the measurement itself, kept together. A **reproducibility package** is that collection for a whole study: the code at a named commit, the identity of the data, the registered protocol, one result record per run holding its commit, command, seed, device and measurements, and the scripts that turn the records into every table and number of the report, so that someone else can recompute each number from the records, rerun each record from its command, and repeat the study with new seeds. The package is not a promise to share materials later. It is the set of files that make the report a function of the runs.

Here is one record of the package of the spectral-lm paper, shortened to the fields that matter.

| field | value in the record of the spectral model with seed 287849435889518350 |
|---|---|
| commit | `90a318a78499fbd6ce7825a9406dfc0d71595c43` |
| device | gpu1 Tesla P40 |
| command | the training command of the sample, with every shape, schedule and seed argument |
| train | the configuration as the model wrote it, the step count, tokens seen, initial and final training loss, parameter counts, the analytic forward FLOPs per token |
| peak_mem_mib | the device memory of the run, measured |
| eval_command | the scoring command, with the 1,500-byte document limit |
| eval | documents and tokens scored, bytes, summed loss in nats, bits per byte |

Each field answers a question that a later reader will otherwise have to guess. Four design choices in the driver that writes these records, `papers/spectral-lm/scripts/run.py`, are worth taking as rules for any package.

A record names a commit that reproduces it. The driver asks git for the current commit and for the tracked files with uncommitted changes, and if there are any it refuses to run: "the tracked tree has uncommitted changes; records must name a commit that reproduces them". A record written from a modified tree would name a commit whose code did not produce it, which is worse than naming none. The same check guards the script that writes the correctness record (`papers/spectral-lm/scripts/correctness.py`).

A record keeps its commit valid when history moves. The branch that held these records was rebased while they were being made, so the commits they named were no longer in its history, and commit `581b6c839` rewrote each record's commit to the rebased commit with the same subject and the same measured code, a change the paper's reproduction appendix states. A record that names a commit nobody can check out has lost the link that made it a record.

A value nothing measured is absent, never zero. The driver first placed its two parallel jobs on two cards with an environment variable that the engine's graphics backend does not read, so both ran on the first card, and it looked up each job's memory by process number, which inside a container matches nothing, so every job recorded a peak memory of 0 MiB (knowledge #359). A zero is a plausible number. The fix was twofold: the runs are pinned by the engine's own device selection (commit `057425e4b`), and memory is measured as the card's use above its idle use with the job alone on the card (commit `ee7d5fa80`). The records written before the fix, the learning-rate selection runs, were rewritten with `null` in the field (commit `90a318a78`).

A record is read strictly. The configuration inside each record is the model's own `to_json`, and the matching `from_json` in `crates/spectrallm/src/config.rs` refuses an unknown or missing field rather than defaulting it, and refuses a spectral configuration whose basis is not the one named `real-fourier-orthonormal-frequency-ordered`, the coefficient convention the build implements. A record that is silently completed with defaults describes a run that never happened.

Seeds deserve a sentence of their own. The driver draws each seed at random (63 random bits) and writes it into the record and into the command; the other common practice, used by the worked examples here, derives each seed from a fixed name with a hash (Volume I, Chapter 17). Both are reproducible, because both put the seed where a reader can find it. They differ in what a rerun does: running a plan of the paper's driver again draws new seeds, so it repeats the study rather than any one run, and running a record's own command repeats that run.

## Recomputing a headline from its records

The test of a package is whether a reader can rebuild the headline number from it without the authors. The paper's headline is the comparison of the spectral model with its two controls, three seeds each, and its nine records are enough to recompute it. First the measurement itself: each record holds the summed next-token loss over the scored text, in nats, and the number of bytes scored, so the bits per byte is $L/(B \ln 2)$ for a summed loss $L$ and $B$ bytes, and it can be recomputed and compared with the value the record also holds.

<!-- worked: research.record_runs -->
The nine records of papers/spectral-lm/data/primary/, one per trained model, in the paper's order and by seed within a model. Bits per byte recomputed = summed next-token loss in nats / ln 2 / UTF-8 bytes of the scored text, from the two fields the record holds; every record scored the same 26,439 documents (4,935,406 tokens). The last column prints the bits_per_byte field the record also holds; the two agree to 1e-12 in every row.

| model | seed | commit | device | summed loss (nats) | bytes scored | bits per byte | field in the record |
|---|---:|---:|---:|---:|---:|---:|---:|
| spectral mixer, C = 24 | 287849435889518350 | 90a318a78 | gpu1 Tesla P40 | 9019499.163 | 19,721,848 | 0.659795508 | 0.659795508 |
| spectral mixer, C = 24 | 5651191025060493230 | 90a318a78 | gpu0 Tesla P40 | 9003065.978 | 19,721,848 | 0.658593385 | 0.658593385 |
| spectral mixer, C = 24 | 8986012006834498100 | 90a318a78 | gpu1 Tesla P40 | 9019734.418 | 19,721,848 | 0.659812717 | 0.659812717 |
| SwiGLU, f = 24 | 1257447695936767018 | 90a318a78 | gpu0 Tesla P40 | 9287350.740 | 19,721,848 | 0.679389419 | 0.679389419 |
| SwiGLU, f = 24 | 3753482583031048465 | 90a318a78 | gpu0 Tesla P40 | 9233352.218 | 19,721,848 | 0.675439313 | 0.675439313 |
| SwiGLU, f = 24 | 6035840459909840765 | 90a318a78 | gpu0 Tesla P40 | 9119945.655 | 19,721,848 | 0.667143382 | 0.667143382 |
| SwiGLU, f = 1024 | 2603556909225610885 | 90a318a78 | gpu1 Tesla P40 | 7781509.340 | 19,721,848 | 0.569233924 | 0.569233924 |
| SwiGLU, f = 1024 | 3127860714495365997 | 90a318a78 | gpu1 Tesla P40 | 7749608.526 | 19,721,848 | 0.566900312 | 0.566900312 |
| SwiGLU, f = 1024 | 6444700745050942377 | 90a318a78 | gpu0 Tesla P40 | 7780987.685 | 19,721,848 | 0.569195764 | 0.569195764 |
<!-- /worked -->

All nine values agree with the records' own field, every run scored the same 19,721,848 bytes, and all nine ran at commit `90a318a78` on one of two cards of the same model. The records also hold counts that do not depend on training at all: the number of parameters outside the tied embedding, and the analytic forward arithmetic per token. Those can be re-derived from the configuration each record names, by a counting rule stated in full. Per block, the parameters are two normalization gains of $m$ each, the query and output projections of $h_q m$ each, the key and value projections of $h_{kv} m$ each, two head-wise normalization gains of $h_d$ each, and the sublayer, $3Cm$ for the spectral mixer (three filter banks of $C$ by $m$) and $3mf$ for a SwiGLU of hidden width $f$; one final gain of $m$ is added after the blocks, and the embedding of $Vm$ is not counted. Here $m$ is the width of the residual stream, $h_q$ and $h_{kv}$ the query and key-value widths (heads times head dimension $h_d$) and $V$ the vocabulary. Per block and token, the FLOPs count a multiply-add as 2: $2m(2h_q + 2h_{kv})$ for the four projections, $\lfloor 4h_q(t+1)/2\rfloor$ for the attention products at the mean causal length of a context of $t$ tokens (the product is even whenever $h_q$ is, so no rounding happens at these sizes), and $6mf$ for a SwiGLU, or $4Cm + 5CD + 3C\cdot 5D\log_2 D$ for the spectral mixer with its transforms of length $D$ (the signal length the record's configuration names) counted as fast transforms; the head adds $2mV$ once. Normalizations and other elementwise operations are not counted. This is the convention of `forward_flops_per_token` in `crates/spectrallm/src/config.rs`.

<!-- worked: research.record_counts -->
Counts re-derived from the configuration each record names (6 blocks, residual width m = 256, 8 query heads and 4 key-value heads of dimension 32, vocabulary 4,096, context t = 512, and for the spectral mixer a transform length D = 512, the signal_len of its record) and compared with the counts the training program wrote into the same record; the three runs of a model hold identical counts.

| model | non-embedding parameters, re-derived | in the record | forward FLOPs per token, re-derived | in the record | equal |
|---|---:|---:|---:|---:|---:|
| spectral mixer, C = 24 | 1,293,952 | 1,293,952 | 16,501,760 | 16,501,760 | yes |
| SwiGLU, f = 24 | 1,293,952 | 1,293,952 | 6,253,568 | 6,253,568 | yes |
| SwiGLU, f = 1024 | 5,901,952 | 5,901,952 | 15,469,568 | 15,469,568 | yes |
<!-- /worked -->

The counts agree to the last digit, which says that the configuration in each record is the one the counts were computed from, and that the parameter-matched control really has exactly the spectral model's parameter count, which is what makes it the baseline of H1. Then the comparison.

<!-- worked: research.record_verdict -->
From the nine bits per byte of the record table (computed at full precision; the nine printed decimals give the same printed results), three runs per model. Standard deviations use the divisor n - 1. Welch's standard error is sqrt(s_a^2 / n_a + s_b^2 / n_b), its degrees of freedom nu = (s_a^2 / n_a + s_b^2 / n_b)^2 / ((s_a^2 / n_a)^2 / (n_a - 1) + (s_b^2 / n_b)^2 / (n_b - 1)), the multiplier is t_quantile_real(0.975, nu), and p is the two-sided tail 2 (1 - t_cdf_real(|d| / se, nu)).

| model | mean bits per byte | standard deviation over 3 runs |
|---|---:|---:|
| spectral mixer | 0.659401 | 0.000699 |
| SwiGLU f = 24 | 0.673991 | 0.006250 |
| SwiGLU f = 1024 | 0.568443 | 0.001336 |

| contrast | mean difference | standard error | nu | multiplier | 95% interval | p |
|---|---:|---:|---:|---:|---:|---:|
| spectral minus SwiGLU f = 24 (H1) | -0.0146 | 0.0036 | 2.05 | 4.204 | -0.0299 to +0.0007 | 0.054 |
| spectral minus SwiGLU f = 1024 (H2) | +0.0910 | 0.0009 | 3.02 | 3.172 | +0.0882 to +0.0937 | below 0.001 |

H1 against the margin 0.02 bits per byte:

| decision rule | verdict |
|---|---:|
| registered: mean difference at most +0.02 | supported |
| stricter: upper end of the interval at most +0.02 | supported |
| superiority: upper end of the interval below 0 | not shown |

The root mean square of the two standard deviations of H1's arms is 0.0044 bits per byte.
<!-- /worked -->

These are the numbers the paper prints: a difference of $-0.0146$ bits per byte against the parameter-matched control with an interval from $-0.0299$ to $+0.0007$, a difference of $+0.0910$ against the arithmetic-matched control with an interval from $+0.0882$ to $+0.0937$. Recomputing them from the records with an independent implementation is what turns them from numbers in a paper into numbers anyone can check.

The degrees of freedom deserve a look. With three runs per arm a pooled interval would use 4, but the spectral model's three runs agree to within 0.0013 bits per byte and the control's spread is nine times larger, so almost all of the uncertainty comes from the control's three runs, and Welch's formula gives $\nu = 2.05$, barely more than the 2 of three numbers alone. The multiplier for $\nu = 2.05$ is 4.204, and the interval is wide enough to reach just past zero. So H1 is supported by its registered rule, supported by the interval rule too, and the spectral model's superiority, which the means suggest, is not shown. Three runs per model were enough for the claim the protocol made and not for a stronger one.

The last line of the table checks the opening's assumption: the root mean square of the two arms' spreads is 0.0044 bits per byte, so the 0.005 used to draw the operating characteristic was of the right size for this comparison, if a little generous.

## Reproduce, rerun, replicate

Three different things can be asked of a package, and they are often confused. To **reproduce** a result in the narrow sense is to recompute the report's numbers from the recorded measurements, as was just done for H1 and H2: it needs only the records and the analysis, and it must agree to every digit, because it is arithmetic. To **rerun** a record is to execute its command at its commit on the same kind of device and obtain the same measurement; that tests whether the record describes what the code does, and it agrees to every digit only if the computation is deterministic on that hardware, which a package should state rather than assume. To **replicate** the study is to repeat the registered plan with new seeds, or new data, and ask whether the verdict survives. Only the last says whether the finding is a property of the method rather than of the three seeds that happened to be drawn.

The records say how a replication of H1 would be likely to come out, under one strong assumption: that the true means and spreads of the two models are what the three records of each show.

<!-- worked: research.replication -->
Suppose the true bits per byte of each model and its spread from seed to seed were exactly what the three records show: spectral mixer mean 0.659401, standard deviation 0.000699; SwiGLU f = 24 mean 0.673991, standard deviation 0.006250 (these six-decimal values are the simulation's inputs). A replication trains n new seeds of each, run i scoring mean + sd z. Generator `research.replication` (0x56976918bb0c7199), `rng.normal()`, n = 3, 6, 10 in turn, 10,000 replications each, the n spectral draws before the n SwiGLU draws. Each column is the share of replications reaching that verdict on H1 (Monte Carlo standard error at most 0.005); 'superiority' means the upper end of the Welch 95% interval of the difference lies below 0, tested as t_cdf_real(-d / se, nu) >= 0.975 with d <= 0.

| seeds per model | registered rule supported | stricter rule supported | superiority shown |
|---|---:|---:|---:|
| 3 | 1.0000 | 0.9883 | 0.5774 |
| 6 | 1.0000 | 1.0000 | 0.9926 |
| 10 | 1.0000 | 1.0000 | 1.0000 |
<!-- /worked -->

With three new seeds per model the registered verdict is reached in all 10,000 simulated replications and the stricter one in 98.8 percent, so the finding of non-inferiority is robust. Superiority, which the original three seeds did not show, would be shown by a replication of the same size in only about 58 percent of replications, and by one with six seeds per model in 99 percent. The paper's post hoc extension is a replication of this kind, three more seeds of each of the two models, and it asks whether the gap the means suggest is real; the table says that six seeds would very likely settle it if the gap is what the first three runs suggest. The assumption is the weak point. Estimated means and spreads from three runs are noisy, and a result chosen because it looked interesting tends to be an overestimate (the winner's curse of Volume II, Chapter 5), so the simulation states what the records predict, not what a replication will find. The extension is registered in the driver as its own plan; its records are not in the repository, and the paper's status records what that means.

## The anatomy of a paper

A paper is how a result travels to people who were not there, and its structure is a sequence of answers to the questions such a reader asks, in the order they ask them. The **research paper structure** used for empirical work in machine learning is that sequence: an abstract that states the question, the method and the main numbers; an introduction that states the question, the scope and the contributions; related work; the method; the implementation; the evaluation protocol; the results; a discussion with the limitations; a conclusion; references; and appendices for what a reader needs to rebuild the work. Each part has a job that can be checked, and the spectral-lm paper's sections, under `papers/spectral-lm/sections/`, show each job done.

| part | the reader's question | what in it must be checkable |
|---|---|---|
| abstract | what was found, and how strongly? | every number it quotes is one the results section reports, with its interval and sample size |
| introduction | what question, what is outside it, what is claimed? | each contribution is a claim that points to the section that supports it |
| related work | what was known, and what is new here? | each cited work exists and says what the paper says it says |
| method | what exactly is the idea? | enough to reimplement it from the text alone |
| implementation | what was built, and how was it shown to be correct? | the correctness gates and what they measured |
| protocol | what was decided before the data? | the hypotheses, margins, rules, budgets and seeds, with every amendment marked |
| results | what happened? | every number from a record, each comparison with its interval and sample size |
| discussion and limitations | why, what else could explain it, how far does it reach? | alternative explanations named; every condition that was not varied listed |
| references | where can I read the sources? | each entry checked against its primary record |
| appendices | how do I rebuild it? | the commands that produce each record and the build that produces the paper |

The introduction carries more weight than its length suggests. The spectral-lm paper's introduction has a paragraph that states its scope ("It does not study Fourier token mixing along the sequence axis or spectral compression of weights"), a paragraph that states the question as one sentence ("does a feed-forward mixer that is diagonal in a Fourier basis around a signal-domain gated product reach the language-modeling quality of a dense feed-forward network, and at what parameter and arithmetic cost?"), and a list of four contributions, each with a cross-reference to the section that establishes it. A reader can check each contribution against its section, and a reviewer can reject a paper whose contributions point at nothing. The scope sentence also does practical work in brain: the papers directory keeps one scope per paper, listed in `papers/README.md`, and a new finding goes into the paper whose scope it falls in (commit `d643c87a5` set up the convention with the first paper).

Related work is where an honest paper gives credit, and getting it right can take a second pass. The paper's basis-absorption proposition, that a decoder whose maps on the residual stream are dense absorbs any orthogonal change of basis into its weights, is the same computational invariance that an earlier paper on compressing pretrained models had stated and used; commit `72e845013` added the citation, a paragraph on that work in the related-work section and a sentence after the proof saying so. A result that rediscovers a known fact is still useful, but only once it says whose fact it is. The bibliography file opens with a note that every entry was checked against its arXiv record, identifier, title and author order, on a stated date: a reference is a claim like any other, and it can be wrong.

The limitations section is the most useful part of a paper for the next person, because it lists the conditions under which the findings were not tested. The spectral-lm paper's limitations name the scale (decoders of one to twenty-four million non-embedding parameters), the single corpus, the single token budget and context length, single-precision training on one model of card, three seeds per primary model, the untuned rate of the pretrained reference, the transform-cost measurements being of its own kernels on one card, and the members of the architecture family that were not evaluated. Each item is a hypothesis someone else can test, which is how a paper becomes something others build on.

## Every number backed by a record

The last link of the chain is the one most often broken: the numbers in the text. A number typed into a paper by hand is copied from somewhere, and copies drift; a sentence that interprets a number can say more than the number supports. brain's rule for papers, in `.agents/rules/papers.md`, is that "every number in a paper's tables is generated from a committed result record under the paper's `data/` directory by the paper's own scripts, never typed by hand", and the spectral-lm paper extends it to the prose: the script `papers/spectral-lm/scripts/build_tables.py` writes every table and also a file of LaTeX macros, one per number the text quotes, so a sentence of the results section reads "the difference is `\DiffSQpar{}` bits per byte (Welch 95% interval `\DiffSQparCI{}`)" and the build fills in the values recomputed above. The verdicts are macros as well, computed from the records by the registered rules: H1's from the mean difference against the margin, H4's from whether any shape's records show the dense path ahead. The gates that establish correctness feed the same way: each gate test prints what it measured as one `GATE` line beside asserting its tolerance, the correctness script collects those lines into a record, and the correctness table is built from it (commit `b5cb9e4a3`; the printing is in `crates/spectrallm/tests/it/model.rs` and `crates/gradcheck/tests/it/spectrallm_fd.rs`).

![A chain of boxes from the registered protocol through the driver, one record per run and the committed records to the table generator, its tables and macros, the paper's sections and the PDF, with a dashed arrow from the committed records back to the driver for the next plan's chosen learning rate.](img/research-evidence-chain.svg)

Generation keeps the numbers right; it does not keep the sentences right. A sentence can still claim more than its records hold, and the only check is to read each sentence against its evidence. One sentence of the results section explained the spectral model's lower training throughput by saying that its transform kernels "run far below the card's arithmetic peak", and no record measured the card's arithmetic peak or the kernels' share of it. Commit `6292fdddd`, titled "cite only what the records hold about transform cost", replaced the clause with a pointer to the section that reports the measured transform cost. The deleted clause may well have been true. It was not shown, and so it went.

The status of the paper as a whole is computed the same way. The generator declares the paper final only when every registered plan has its records: the correctness gates, all six model families, both measurements of the pretrained reference, the spectrum study, the transform cost and, since the extension was registered, the extension's seven runs. Until then three macros insert a sentence into the abstract, the results and the conclusion saying that the paper is a draft whose registered results are not all measured. The history shows the rule at work in both directions. Commit `9de0d3fef` wrote the discussion, limitations and conclusion and set the status to final, every registered measurement having its record; commit `c8f490b98` then added the post hoc extension and set the status back to draft "until its records exist". A paper that registers more work than it has done says so in its own text.

Even the digits of a generated number are a decision. The spectrum table prints a change of loss to two decimals, and it once printed a tiny negative change as `-0.00`, which reads as a measured decrease where the value is zero at the printed precision; commit `dcdc876e9` made the generator print it as `0.00`. A package that generates its numbers can fix such a thing once, for every table it builds.

## Summary

A claim becomes testable when it is stated before the data with the quantity measured, the comparison, the margin, the decision rule and the sample, so that each possible result is assigned to a verdict in advance; that is a falsifiable hypothesis. Its decision rule has an operating characteristic, the chance of saying "supported" at each true difference, which can be computed before any run: the mean rule passes a claim at its boundary half the time whatever the sample, while the rule that requires the whole interval to clear the margin rarely passes a false claim and pays in power that only more runs buy. Registered hypotheses fail in different ways, and a hypothesis without a margin can be refuted by noise its author would have tolerated. A registration fixes the protocol in a dated commit, amendments are marked before or after the data, and decisions the experiment makes along the way, such as the chosen learning rate, are taken by rules from committed records.

A reproducibility package makes a report a function of its runs: the data named by checksums and filters, one record per run holding its commit, command, seed, device and measurements, written only from a committed tree, absent where nothing was measured, and parsed strictly, plus the scripts that build every number from the records. With brain's spectral-lm records the headline was recomputed independently, the counts re-derived from the configurations, and the replication odds estimated: the registered non-inferiority is robust, and superiority would need about six seeds per model if the observed gap is real. A paper is the structure that carries such a result to readers, each part answering a question they ask, and it is trustworthy when every number in it is generated from a record and every sentence claims no more than its record holds.

The methods of this volume now have a complete chain from a reference implementation to a defended result. The next chapter turns to what happens when that chain breaks: a catalogue of the expensive mistakes made while building a real engine, each traced to its root cause and to the gate that now keeps it from coming back.

## Check your understanding

1. A protocol registers the claim "our model is at least as good as the baseline" with the rule "supported if our mean score is better". What is the rule's chance of supporting the claim when the two models are exactly equal, and what would you change?
2. Predict, without simulating: if the spread $\sigma$ in the opening table were halved to 0.0025 and nothing else changed, would the mean rule's share at $\delta = +0.025$ for three runs rise or fall, and would its share at $\delta = +0.02$ change?
3. A colleague reruns a record's command at its commit and gets bits per byte that differ in the fifth decimal. Is the package broken? What would you check first?
4. A paper reports "the method is 0.015 bits per byte better (three seeds)". Using the record verdict table, say what is missing from the sentence and how it could mislead.
5. Find the error: "H3's energy clause failed, so the Fourier basis is a better coordinate system than a random one for this model."
6. Why does the driver refuse to run on a tree with uncommitted changes to tracked files, but not on untracked files? What would break if it refused nothing?
7. Explain why Welch's degrees of freedom for H1 are close to 2 rather than to 4, and what that does to the interval.

## References

- B. L. Welch. The generalization of 'Student's' problem when several different population variances are involved. Biometrika 34, 28-35, 1947.
- F. E. Satterthwaite. An approximate distribution of estimates of variance components. Biometrics Bulletin 2, 110-114, 1946.
- Karl Popper. The Logic of Scientific Discovery. 1959 (first published in German in 1934).
- Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, James Hensman. SliceGPT: Compress Large Language Models by Deleting Rows and Columns. International Conference on Learning Representations, 2024.

## Exercises

1. **[Calculate]** With $\Delta = 0.02$, $\sigma = 0.005$ and $n = 8$, compute the mean rule's chance of supporting the claim at $\delta = 0.025$ from $P_{\text{mean}}(\delta)$, and check it against the bracketed value of `research.rule_pass_rates`.
2. **[Calculate]** For the mean rule, find the number of runs per arm $n$ at which a claim false by $\delta - \Delta = 0.005$ is supported in at most 5 percent of experiments, with $\sigma = 0.005$. ($\Phi(-1.645) = 0.05$.)
3. **[Calculate]** From the three spectral and three SwiGLU f = 24 bits per byte of `research.record_runs`, compute by hand the two standard deviations, Welch's standard error and $\nu$, and confirm the row of `research.record_verdict`.
4. **[Calculate]** Re-derive the non-embedding parameter count of the SwiGLU model with $f = 1024$ by the counting rule stated with `research.record_counts`, block by block, and the forward FLOPs per token at $t = 512$.
5. **[Implement]** Implement `welch`, `t_cdf_real` and `t_quantile_real` and reproduce `research.record_verdict` from the nine bits per byte printed in `research.record_runs` (nine decimals), to every printed digit.
6. **[Implement]** Implement the simulation of `research.rule_pass_rates` with the book's generator and the stated draw order and reproduce the table's shares exactly. Then add a third rule, "supported if the lower end of the interval is below 0.02", and describe its operating characteristic.
7. **[Implement]** Implement the add-one byte model and reproduce `research.data_identity`, including the checksums.
8. **[Investigate]** Rewrite H3 of the spectral-lm protocol so that it says what its authors meant: name the margin, the number of random draws and the rule, and say which result would refute it.
9. **[Investigate]** Take a paper you know and write the table of the anatomy section for it: for each part, the reader's question and whether what must be checkable is checkable from the paper. Which part fails first?
10. **[Investigate]** Design the record format for an experiment of your own: list its fields, say which of them a rerun must reproduce exactly, which only approximately, and how a value that was not measured is written.

## Key terms

- **falsifiable hypothesis**: a claim stated before the data together with the quantity measured, the comparison, the margin that matters, the decision rule and the sample, so that every possible result is assigned to a verdict in advance and some results would refute it; its decision rule's chance of supporting the claim at each true difference can be computed before any run. <!-- concept: falsifiable-hypothesis -->
- **reproducibility package**: the files that make a report a function of its runs: the code at a named commit, the identity of the data, the registered protocol, one record per run with its commit, command, seed, device and measurements (absent where nothing was measured), and the scripts that build every table and quoted number from the records. <!-- concept: reproducibility-package -->
- **research paper structure**: the sequence of parts of an empirical paper (abstract, introduction with scope and contributions, related work, method, implementation, protocol, results, discussion and limitations, references, appendices), each answering a question a reader asks and each holding something the reader can check. <!-- concept: research-paper-structure -->

## Answers

**Recall questions.**

1. One baseline matches the candidate's parameters and the other its arithmetic, because "equal" has two meanings and the candidate can win at one and lose at the other; with one baseline the comparison would answer only one of the two questions "better per parameter" and "better per operation". A baseline that differs in exactly one thing lets the reader attribute any difference to that thing, since everything else was held equal.
2. The components interact: the effect of each depends on whether the other is present, so the single-removal losses (5 and 1) do not add up to the loss of removing both (3). Report the whole grid of runs that were made and do not divide the gain into shares.
3. Any five of: the interval over seeds; the interval over the test sample; the number of seeds and how they were produced; the number of test examples; what was tuned, on which split, with what budget for each arm; how many variants, metrics and seeds were tried; what was decided in advance; and the seeds, commit, configuration and command that let someone repeat it.
4. With few numbers the standard deviation is itself estimated from the data, and the standardized mean follows Student's t rather than the Gaussian; t has heavier tails when the degrees of freedom are few. Three numbers give $n - 1 = 2$ degrees of freedom, and the 97.5 percent point of t with 2 degrees of freedom is 4.303.

**Check your understanding.**

1. When the models are equal the mean difference is equally likely to fall on either side of zero, so the rule supports the claim half the time, and it also supports a slightly worse model nearly half the time. Give it a margin of what difference matters and decide by the interval: for non-inferiority, require the interval's worse end to lie within the margin.
2. It falls: the standard deviation of the difference halves, so the argument $(\Delta - \delta)/(\sigma\sqrt{2/n})$ at $\delta = 0.025$ doubles in size to $-2.449$ and $\Phi$ drops from 0.110 to about 0.007. At $\delta = 0.02$ the argument is zero for any $\sigma$, so the share stays at one half.
3. Not necessarily. A rerun agrees exactly only if the computation is deterministic on that device; parallel reductions on a graphics card can add in a different order from run to run. Check first that the commit, command, seed, device model and data checksum match the record, then whether the training is deterministic on that backend; a fifth-decimal difference is a statement about rerun determinism, while the reproduced numbers (recomputed from the records) must agree exactly.
4. It omits the interval and the comparison. The interval of the difference against the parameter-matched control is $-0.0299$ to $+0.0007$, which includes zero, so "better" is not shown by three seeds; the sentence also does not say which baseline (against the arithmetic-matched one the method is worse by 0.091) nor what was registered (non-inferiority, which holds).
5. The clause had no margin and compared with a single random draw, so its failure says only that the Fourier basis kept slightly more energy, 0.013 to 0.021, than one random basis; both kept about the fraction $m/1024$, and the information clause showed the Fourier projection costing as much loss as the random one or more at the smaller widths. The failure of a clause written without a margin is not evidence of a meaningful difference.
6. Tracked files are the code and configuration that a commit names; if any of them were modified, the record would name a commit whose code did not produce it. Untracked files (logs, checkpoints, scratch output) are not part of what the commit reproduces. If it refused nothing, a record could name a commit that cannot reproduce it, and the link between record and code would be unverifiable.
7. Welch's formula weights each arm's variance by its share of the total; the control's variance is about eighty times the spectral model's ($0.00625^2$ against $0.000699^2$), so almost all of the uncertainty comes from three numbers with 2 degrees of freedom, and $\nu = 2.05$. The multiplier is then 4.204 instead of the 2.776 of 4 degrees of freedom, a multiplier half again as large, and the interval reaches past zero.

**Exercises.**

1. The argument is $-0.005/(0.005\sqrt{2/8}) = -2$ and $\Phi(-2) = 0.0228$, which rounds to the bracketed 0.0228 of the table.
2. Need $-0.005/(0.005\sqrt{2/n}) \le -1.645$, that is $\sqrt{n/2} \ge 1.645$, $n \ge 5.41$, so six runs per arm.
3. The spectral standard deviation is 0.000699 and the control's 0.006250 (divisor 2); $s_a^2/3 = 1.63\times10^{-7}$, $s_b^2/3 = 1.302\times10^{-5}$, the standard error $\sqrt{1.318\times10^{-5}} = 0.0036$, and $\nu = (1.318\times10^{-5})^2 / ((1.63\times10^{-7})^2/2 + (1.302\times10^{-5})^2/2) = 2.05$.
4. Per block: $2 \cdot 256$ norm gains, $256 \cdot 256$ query, $2 \cdot 128 \cdot 256$ key and value, $2 \cdot 32$ head norms, $256 \cdot 256$ output and $3 \cdot 256 \cdot 1024 = 786{,}432$ for the SwiGLU, so $512 + 65{,}536 + 65{,}536 + 64 + 65{,}536 + 786{,}432 = 983{,}616$; six blocks give $5{,}901{,}696$ and the final gain $256$ makes $5{,}901{,}952$. FLOPs per block: $2\cdot256\cdot(512 + 256) = 393{,}216$, $4\cdot256\cdot513/2 = 262{,}656$, $6\cdot256\cdot1024 = 1{,}572{,}864$, total $2{,}228{,}736$; six blocks $13{,}372{,}416$ plus the head $2\cdot256\cdot4096 = 2{,}097{,}152$ gives $15{,}469{,}568$.
5. See `research.record_verdict`; the H1 row is $-0.0146$, standard error 0.0036, $\nu = 2.05$, multiplier 4.204, interval $-0.0299$ to $+0.0007$, $p = 0.054$.
6. See `research.rule_pass_rates`. The third rule passes whenever the interval reaches below the margin, which includes every experiment the mean rule passes; its curve lies above the mean rule's everywhere, passing false claims more often than one half near the margin, and it is a rule that favours the claim.
7. See `research.data_identity`; the four bits per byte are 5.4699, 5.5668, 5.5283 and 5.4530.
8. One version: "the energy kept by the lowest $m$ Fourier modes exceeds the mean energy kept by 20 random orthonormal bases of the same dimension by at most 0.02 at every width tested; refuted if, at any width, the Fourier value exceeds the upper end of the 95 percent interval of the 20 random values plus 0.02". The number of draws and the margin are the parts the original lacked.
9. The answer depends on the paper. A common first failure is the results part: numbers without intervals or sample sizes, or a comparison against a baseline whose tuning is not stated.
10. The answer is a design. A sound one keeps commit, command, seed, device, data checksum and every measurement; says that counts and configuration must rerun exactly, while trained scores agree exactly only on a deterministic backend and otherwise to a stated tolerance; and writes an unmeasured value as an explicit null, never as 0.
