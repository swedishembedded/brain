<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Testing Numerical Code

*Why do tests of numerical code pass while the code is wrong, and how do we write tests that cannot be fooled?*

Here is an attention head with three mistakes in it. It forgets to divide the scores by $\sqrt{d_k}$. It normalises each column of the score matrix instead of each row, so the weights a query puts on the keys no longer add up to one. And its causal mask is off by one, so every token can see the token after it. Test it against a correct reference on a prompt of one token, comparing every output to a tolerance of $10^{-9}$, and it passes. It does not pass narrowly: the two outputs are equal to the last bit.

Nothing is wrong with the comparison. With one token there is one query and one key, so the score matrix is a single number, and the softmax of a single number is 1 whatever the number is. The scale multiplies a number nobody reads; the row and the column of a $1 \times 1$ matrix are the same thing; a mask that lets the token see "the next one" finds nothing there to see. Every one of the three wrong programs computes $\mathbf{Y} = \mathbf{V}$ on this input, and so does the right one. The test asked a question that every candidate answers correctly.

This is the general shape of a numerical test that is fooled. A test is a chain of four links: an instance (sizes and values to run on), the code under test beside a reference that says what the answer should be, a measure of the difference with a threshold, and a verdict that reaches whoever reads the test report. A wrong program gets through when any link has a hole in it. The instance can be one on which the wrong program and the right one compute the same function. The reference can share the mistake. The measure can be blind to a whole family of errors, so that a wrong output scores as perfectly as a right one. And the comparison can simply not happen, while the report still says "passed". Each hole has a cure, and the cures are of one kind: they ask the test to prove that it could have failed.

![The chain of a numerical test, drawn as four boxes left to right inside one frame titled "The chain of a numerical test and where a wrong program gets through": test instance (sizes and values), code under test and its reference, measure and threshold, and the verdict the runner reports, joined by arrows. Under each box a note names the way that link lets a wrong program through: a degenerate instance on which a wrong program computes the same; a shared reference whose oracle repeats the same mistake; an invariant measure under which a family of errors scores perfectly; a gate that never runs, so that no comparison is made and the result is reported as a pass.](img/fooled-test.svg)

The instruments for this have all been built already, and four of them are worth bringing back before they are leaned on. From "Vectors: Lists of Numbers with Geometry" (Volume I, Chapter 3): how is the cosine similarity of two vectors defined, and what change of one vector leaves it unchanged? From "Proving Gradients Correct" (Volume III, Chapter 6): what does a gradient check compare with what, and what does a passing check prove about the objective? From the same chapter: by which rule are the sizes of a test model chosen, and why may none of them be 1? And from "Importing Pretrained Models and Proving Parity" (Volume VI, Chapter 2): where does a parity tolerance come from, and why is it not set a little above the error the run happened to show? All four are answered in the last section of the chapter.

## When the wrong program computes the right function

The one-token prompt is the extreme case of something that happens at every size. A test instance is a choice, and some choices make a mistake disappear. To see which, take the attention head and six things a person can get wrong in it, and run each of them on five instances. The reference is causal attention as Volume V defined it, one head:

$$
\mathbf{Y} = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V},
$$

with $\mathbf{Q}$ and $\mathbf{K}$ of shape $T \times d_k$, $\mathbf{V}$ of shape $T \times d_v$, $T$ the number of tokens, the mask $\mathbf{M}$ zero where key $j$ is at or before query $i$ and $-\infty$ elsewhere, and the softmax taken along each row. A wrong copy of a program with one deliberate change is a mutant, the planted defect of the test-the-test exercise of "Project: Your Own Small Language Model" (Volume V, Chapter 15), and each of the six below changes one step:

```text
function attention_variant(Q: array<real>[T, dk], K: array<real>[T, dk],
                           V: array<real>[T, dv], m: int) -> array<real>[T, dv]:
    # m = 0 the reference; 1 scale omitted; 2 softmax over queries; 3 mask omitted;
    # 4 mask one key late; 5 values read from K; 6 score (i, j) computed as k_i . q_j
    require T >= 1 and 0 <= m and m <= 6, "a sequence and a known variant"
    require m != 5 or dv == dk, "K in place of V needs d_v = d_k: a shape error"
    c = 1.0 if m == 1 else 1.0 / sqrt(dk)
    scores = zeros(real, [T, T])
    for i in 0 .. T:
        for j in 0 .. T:
            visible = j <= i or m == 3 or (m == 4 and j == i + 1)
            if not visible:
                scores[i, j] = -infinity
            else if m == 6:
                scores[i, j] = c * dot(K[i, :], Q[j, :])
            else:
                scores[i, j] = c * dot(Q[i, :], K[j, :])
    P = transpose(softmax(transpose(scores))) if m == 2 else softmax(scores)
    return matmul(P, K) if m == 5 else matmul(P, V)
```

The instances differ in their sizes $(T, d_k, d_v)$ and, in one case, in their values: "identical tokens" repeats one row of each matrix $T$ times, which is what a test gets when it fills every position of a prompt with the same token.

<!-- worked: testing_numerical_code.degenerate_instances -->
One causal attention head, Y = softmax(Q K^T / sqrt(d_k) + M) V, with M = 0 where key j <= query i and minus infinity elsewhere (rows and columns numbered from 0), the softmax taken along each row, in double precision. One generator, the book's SplitMix64 with seed 0xf489693257f4b4f6 (17620730682008712438), draws the instances one after another in the order of the table; for each, Q (T x d_k), then K (T x d_k), then V (T x d_v), every entry normal() in row-major order, except for 'identical tokens', where one row of Q, one of K and one of V are drawn (d_k, d_k and d_v values) and repeated T times. Each mutant (numbered m as in attention_variant) replaces one step: 1, the scale 1/sqrt(d_k) left out; 2, each column of the masked scores normalised over the queries that may see that key instead of each row over its keys; 3, every key visible to every query; 4, query i seeing keys up to i + 1; 5, K in place of V (a product that exists only when d_v = d_k); 6, score (i, j) computed as k_i . q_j instead of q_i . k_j. Dot products and the sums of the softmax and of P V are taken in increasing index. A mutant is caught when some entry of its Y differs from the reference's by more than 1e-09; the number is that largest absolute difference over all entries, to two significant digits.

| instance (T, d_k, d_v) | 1: no scale | 2: column softmax | 3: no mask | 4: late mask | 5: V from K | 6: K Q^T |
|---|---|---|---|---|---|---|
| one token (1, 4, 3) | same output | same output | same output | same output | shape error | same output |
| head width one (3, 1, 2) | same output | caught, 2.6 | caught, 3.8 | caught, 2.6 | shape error | caught, 0.66 |
| identical tokens (3, 4, 3) | same output | caught, 1.8 | same output | same output | shape error | same output |
| square sizes (3, 3, 3) | caught, 0.20 | caught, 1.5 | caught, 1.5 | caught, 1.2 | caught, 0.89 | caught, 0.45 |
| distinct sizes (4, 5, 3) | caught, 0.28 | caught, 2.5 | caught, 0.97 | caught, 0.77 | shape error | caught, 0.37 |
<!-- /worked -->

Read the table by rows. The one-token instance passes five of the six mutants and rejects the sixth only because the product $\mathbf{P}\mathbf{K}$ has the wrong width, the same rejection by shape that Volume III, Chapter 6 found for transposed weights. A head width of one hides the missing scale, because $1/\sqrt{1} = 1$: the mutant is wrong at every other width and right at this one. The identical-token instance is the subtle row. Each query's scores are all equal, so each row of the softmax is uniform over the keys it may see; and since every row of $\mathbf{V}$ is the same vector, any average of them is that vector. The scale, the mask and the order of the product change which rows are averaged or with what weights, and none of them changes the average. Only the column normalisation shows, and only because its weights no longer add up to one. The square instance catches all six, including the one that reads values from $\mathbf{K}$, which is legal there because $d_v = d_k$, but it catches that one by value only, and a structure in the values (a test that sets $\mathbf{V} = \mathbf{K}$ to save a draw) would hide it again. The distinct instance catches five by value and rejects the sixth by its shape, the earliest and most legible failure there is.

The **degenerate-dimension trap** is a test instance on which some wrong program computes the same function as the right one, because a size is 1, two sizes are equal, or the values have a symmetry or a restricted range (all equal, all ones, all zero, all of one sign), so the test cannot fail for that mistake however tight its tolerance. The everyday sense of "degenerate" is "fallen from its kind", and the mathematical sense is the one that matters: a degenerate case is one where distinct things coincide, a circle of radius zero, a matrix whose rows are equal, a softmax over one score.

Chapter 6 of Volume III named the cure for sizes, test-dimension choice: every size distinct and above 1, so that no two roles can be exchanged and no sum has a single term. The table shows that the trap is wider than sizes, and brain's history has instances of every kind.

*Values with a restricted range.* About forty test files of brain generated their random inputs with a private copy of one line of arithmetic, a linear congruential generator whose 64-bit state was shifted right by 33 bits, divided by $2^{31}$ and reduced by one. The shift keeps 31 bits, so the division lands in $[0, 1)$ and the subtraction puts every sample in $[-1, 0)$: no test that used it ever fed a positive number to a ReLU, a parametric ReLU, a leaky ReLU or a maximum, the one branch those kernels exist to get right. Commit `55eb76184` (4 August 2026) replaced the copies with one shared generator, `data::rng::Lcg` in `crates/data/src/rng.rs`, whose `signed` shifts by 32 bits and covers $[-1, 1)$; its doc comment records why, and the commit notes that three files had found and patched the defect on their own while nothing compared the copies with each other. A test input of one sign is a degenerate instance, and it is invisible from the outside, because every test that used it passed.

*Sizes below a threshold.* A size can also be degenerate for the code rather than for the mathematics. A kernel library usually chooses between implementations by size: below some width a simple loop, above it a faster path built from different pieces. A test model small enough to be cheap may never cross the threshold, and then the path every real model takes has no test at all. Commit `83990db00` (10 August 2026) found exactly that in brain's vector-quantised image autoencoder: its gradient-check model had at most 8 output channels, the convolution backward switches, on a device with cooperative reductions, to a lowered matrix-product path at 32 (`GEMM_CONV_BWD_MIN_COUT` in `crates/backend-api/src/select.rs`), and so the path of a training step with hundreds of channels was reached by no finite-difference check. The fix, `lowered_config` in `crates/gradcheck/src/vqgan.rs`, is a model of 32 channels whose only purpose is to cross the threshold, and its test skips by name on a device where the lowered path cannot be selected, because there, as its comment says, re-running the simple path "would be a vacuous green".

*Magnitudes starved below a constant.* A toy model's activations can be so small that a fixed constant swamps them. Knowledge #40 records a gradient check of a hybrid recurrent decoder that passed cleanly while every parameter of its recurrent layers reported a numeric derivative of exactly 0. The toy model's small initial weights, passed through a depthwise convolution of 4 taps and a nonlinearity, left the vectors entering an L2 normalisation at about $10^{-6}$, the size of the normalisation's own $\epsilon$; the "normalised" output was the input divided by $\sqrt{\epsilon}$, still tiny, and nothing downstream depended on the recurrent parameters any more. The harness `qwen35moe_gradcheck_harness` in `crates/gradcheck/src/lib.rs` now redraws the two weight tensors in front of the normalisation (the input projection and the convolution of each recurrent layer) with `Lcg::scaled(1.0)`, which is uniform on $[-1, 1)$ and so has a standard deviation of $1/\sqrt{3} \approx 0.58$; the entry and the harness's comment describe the override as a standard deviation of 1.0, and either is some thirty to fifty times the 0.02 of the model's own initialisation, which is what puts the test instance in the regime the real model is in. The model's own initialisation and $\epsilon$ are unchanged, because they are right for the model.

*A real model's coincidences.* The trap is not only in toy instances. T5-XXL, the text encoder of an image generator that brain ports, has 64 heads of width 64, and 64 times 64 is its model width of 4096. Commit `a3a479800` (4 August 2026) records the consequence: an exchange of the head count with the head width, or of the model width with the attention's inner width, would have passed a 19 GB parity test at cosine 1.0, because at those sizes the exchanged quantities are the same number. The same commit added `tests/tiny_ref.rs` of the T5 encoder crate (now `crates/t5encoder/tests/it/tiny_ref.rs`), a two-layer random reference at 2 heads of width 64 and a model width of 64, so that the inner width is 128, and knowledge #4 turns the case into the rule that both the sizes and the choice of which fixtures to compare must differ in every role. Another of its examples is a model with attention sites of two widths whose first set of goldens happened to be two sites of the same width; the test now discovers the widths from the goldens and asserts that it saw at least two.

That last move generalises, and it is the defence that cannot itself decay. A test that depends on an instance being non-degenerate should say so in code, so that a later edit to the instance fails the test instead of quietly hollowing it out. `tiny_ref.rs` asserts `cfg.inner() != cfg.d_model` and `cfg.heads != cfg.d_kv` before it compares anything, and checks the golden's own tensor widths against the configuration. `check_gpt_tied` in `crates/gradcheck/src/lib.rs`, which checks a language model whose output layer shares its table with the input embedding, asserts that the batch leaves some vocabulary rows unused as inputs, because only on those rows is the output layer's share of the gradient alone and visible; a batch that used every token would make the check unable to tell a missing share from a present one.

The table itself is a way to measure an instance, and it can be run on any instance before it is trusted: list the mistakes you fear, plant each one, and see which ones the instance cannot tell from the reference.

```text
function blind_mutants(Q: array<real>[T, dk], K: array<real>[T, dk],
                       V: array<real>[T, dv], tol: real) -> list<int>:
    # the variants of attention_variant this one instance cannot tell from the reference
    require tol >= 0, "a tolerance is not negative"
    want = attention_variant(Q, K, V, 0)
    blind = []
    for m in 1 ..= 6:
        if m == 5 and dv != dk:
            continue                             # rejected by its shape: caught
        got = attention_variant(Q, K, V, m)
        if max(abs(got - want)) <= tol:
            append m to blind
    return blind
```

An empty list is evidence about the instance, not about the code: it says that this instance can fail for each of the planted mistakes. It is cheap, one forward pass per mutant, and it is the only way to know that a test of a small model is not a test of nothing.

## Measures that forgive a family of errors

An instance that can tell two programs apart is half of a test. The other half is a number that says how far apart they are, and here a different trap waits. Every measure of difference is blind to something, and the useful way to say what is to name the transformations it ignores.

Write $\mathrm{dist}(\mathbf{a}, \mathbf{b})$ for a measure of how far an output $\mathbf{a}$ is from a reference $\mathbf{b}$, and say that the measure is invariant under a transformation $g$ when $\mathrm{dist}(g(\mathbf{b}), \mathbf{b})$ is the value $\mathrm{dist}(\mathbf{b}, \mathbf{b})$ of a perfect match. Then a mistake that happens to act on the output as $g$ acts scores as a perfect match, whatever its size. Five kinds of measure in common use, and what each one forgives:

- cosine similarity is invariant under multiplying the whole output by any positive number;
- agreement of the largest entry of each row (top-1 agreement, the measure of "does it still predict the same token") is invariant under any transformation that preserves the order within each row: a positive scale, an added constant, a squashing $z \mapsto z + 0.05 z^3$;
- a comparison of probabilities after a softmax is invariant under adding a constant to every logit of a row, because the softmax subtracts it back out;
- any comparison made after a normalisation of the rows (a root-mean-square norm, a layer norm, an L2 normalisation) is invariant under scaling each row by its own positive factor;
- relative L2 error and the largest absolute difference, `rel_l2` and `max_abs` of Volume VI, Chapter 2, are invariant under nothing but the identity: they forgive only what is smaller than their threshold.

Six outputs show the list at work: one correct (the reference rounded to 32 bits), and five wrong in the ways the list names, each gated by the six measures.

<!-- worked: testing_numerical_code.invariant_measures -->
Reference logits b: 3 rows of 5, drawn by the book's SplitMix64 with seed 0x20fe97f2eb0176cc (2377504722878822092) as normal() in row-major order. Each output is b changed as the first column says: rounded to the nearest 32-bit number; multiplied by 1.05; row r plus 0.3, -0.2, 0.5; row r times 1.0, 1.2, 0.8; the two smallest entries of row 0 swapped; every z replaced by z + 0.05 z^3. Six gates, each a pass or FAIL: cos, the cosine of the two flattened 15-vectors at least 0.9999; rel, rel_l2 at most 1e-05; max, max_abs at most 1e-05; top-1, the largest entry in the same column in every row; probs, max_abs of the row softmaxes at most 1e-05; normed, rel_l2 at most 1e-05 after each row of both is divided by its root mean square. Sums in index order, double precision.

| output | 1 - cos | rel_l2 | cos | rel | max | top-1 | probs | normed |
|---|---:|---:|---|---|---|---|---|---|
| correct, rounded to 32 bits | below 1e-12 | 2.42e-8 | pass | pass | pass | pass | pass | pass |
| all logits times 1.05 | below 1e-12 | 5.00e-2 | pass | FAIL | FAIL | pass | FAIL | pass |
| a constant added to each row | 4.16e-2 | 3.03e-1 | FAIL | FAIL | FAIL | pass | pass | FAIL |
| each row scaled by its own factor | 1.28e-2 | 1.61e-1 | FAIL | FAIL | FAIL | pass | FAIL | pass |
| two smallest of row 0 exchanged | 1.01e-2 | 1.42e-1 | FAIL | FAIL | FAIL | pass | FAIL | FAIL |
| z + 0.05 z^3 on every logit | 7.65e-4 | 1.28e-1 | FAIL | FAIL | FAIL | pass | FAIL | FAIL |
| errors caught, of 5 |  |  | 4 | 5 | 5 | 0 | 4 | 3 |
<!-- /worked -->

The correct output passes every gate, which shows that none of them is too tight to be used. Then each gate fails exactly where its invariance says it must. Cosine waves through the uniform scale of 1.05, with $1 - \cos$ below $10^{-12}$, and catches the other four, because they change the direction of the 15-vector. Top-1 agreement catches nothing at all, 0 of 5: every one of the five wrong outputs keeps the largest entry of every row where it was. The probability comparison passes the row constants, which a missing or doubled bias on the logits would produce, and catches the rest. The comparison after normalising each row passes both scale errors, the uniform one and the per-row one, which are precisely the errors a per-channel scale factor makes. Only `rel_l2` and `max_abs` catch all five.

The **scale-invariant-metric trap** is a test that gates on a measure invariant under a transformation that a plausible mistake applies to the output, so that the mistake scores as a perfect match: cosine under a dropped or doubled scale, top-1 agreement under a wrong temperature or bias, probabilities under a constant shift of the logits, any comparison under whatever a normalisation between the mistake and the measurement removes. In ordinary speech "invariant" means "never changing"; here it means unchanged by a particular change, which is useful in a feature and fatal in a test.

The fourth case is the one that catches people who know the first. Knowledge #47 is the record, with the figures below as it gives them, of a gate on a new 8-bit tier of a text encoder in brain, written after knowledge #2 (the cosine lesson) and asserting cosine and `rel_l2` on a layer's output and cosine on its self-attention output. It was checked by planting the most likely defect of such a tier, every per-channel weight scale halved, and the gate passed. On the attention output the cosine was 0.998157036 with and without the mutation, while `rel_l2` there, which the gate did not assert, rose from 0.061 to 0.503. On the layer output, which the gate did assert with both measures, `rel_l2` moved only from 0.067 to 0.082, inside the tolerance, because the attention output passes through a normalisation that divides the factor of one half back out, and is then added to a residual that dominates what is left. A measure that sees scale cannot see it after something that removes scale. The repair was to assert `rel_l2` on the attention output itself, upstream of the normalisation.

The cure has two parts, and the table justifies both. First, every gate asserts at least one measure whose invariance is trivial, `rel_l2` or `max_abs` or both, so that no family of errors is forgiven above the tolerance. Second, the measurement is taken upstream of every operation that removes what the error changes: a tap on the output of a sublayer, before its normalisation and before its residual add, and logits compared as logits, not as probabilities or as the index of their largest entry. Cosine stays useful, as a diagnosis rather than a gate: a failure with cosine 1.000000 says the direction is right and the size is wrong, which narrows the search to scales, gains and normalisations.

```text
function parity_failures(got: array<real>[N], want: array<real>[N],
                         cos_floor: real, rel_ceiling: real) -> list<text>:
    require N > 0, "an empty tap compares nothing"
    require max(abs(want)) > 0, "a reference of zeros matches anything in direction"
    failures = []
    c = dot(got, want) / (norm(got) * norm(want))
    r = norm(got - want) / norm(want)
    if not is_finite(c) or c < cos_floor:
        append "cosine" to failures
    if not is_finite(r) or r > rel_ceiling:
        append "rel_l2" to failures
    return failures
```

Two lines in it are defences against degenerate references rather than against wrong outputs. A reference tap that is all zeros makes the cosine undefined and the relative error meaningless, so it is refused before anything is compared. And a cosine or a relative error that is not finite (an output of zeros gives $0/0$) is a failure, written as "not finite or below", because the comparison `c < cos_floor` is false for a not-a-number value and would let it pass.

brain has measured how often each measure carries the verdict alone. Commit `d16050463` (25 August 2026) made three parity tests of a speech codec and a speaker encoder actually run, gated each on two measures, and then checked each gate by applying a mutation and reverting it. Each mutation was caught by a different measure: a uniform scale on the speaker embedding by `rel_l2` alone, with the cosine printing 1.0000000000 unchanged; a uniform scale on the decoded waveform by a log-mel distance (a spectral one) alone, the scaled signal having slipped under the `max_abs` ceiling; and a scaled input to the codec's residual quantiser by the agreement of the codes, the damage landing in the later codebooks. Every single-measure gate would have missed one of the three.

## Tests that pass without comparing anything

The third trap needs no subtle mistake and no blind measure. The test does not compare anything, and the test runner reports it as a pass.

Many test runners know three outcomes, passed, failed and ignored, and a test that decides for itself not to run, because a file it needs is missing or a device is absent, has no way to report a fourth. If it returns early it has passed. A suite of nine such tests run on a machine where seven of them find no fixture reports "9 passed", and that report is identical, character for character, to the report of a machine on which all nine compared their outputs and all nine agreed. Knowledge #1 records exactly this: nine tests of brain's image tokenizer reported as passing, seven of which had printed a skip message because a weights file was named by an environment variable nobody had set.

The cases come in more shapes than the missing file. The table runs one parity test of three taps, with the sound gate of the last section, in eight states, on the correct output and on the mutant that multiplies everything by 1.05:

<!-- worked: testing_numerical_code.silent_gates -->
One parity test of three taps (embed, block0, block1; the golden holds the three rows of the reference logits of the previous table, one tap each) run in eight states, on the correct output (rounded to 32 bits) and on the mutant that multiplies every value by 1.05. The sound test asserts cosine at least 0.9999 and rel_l2 at most 1e-05 on every tap. 'returns early' and 'skip printed' return without comparing when the golden is absent; 'demands fixtures' turns that return into a failure; 'no tap name matches' looks up names the golden does not contain (the golden's names were changed) and skips each missing one; 'only cos asserted' computes and prints rel_l2 but asserts only the cosine; 'plus a count' adds a failure when fewer than 3 taps were compared. The status is what a test runner reports.

| state of the test | taps compared | correct code | code times 1.05 | runner also shows |
|---|---:|---|---|---|
| golden present, cos and rel_l2 asserted | 3 | pass | FAIL |  |
| golden absent, test returns early | 0 | pass | pass | nothing printed |
| golden absent, skip printed | 0 | pass | pass | SKIP printed |
| golden absent, run demands fixtures | 0 | FAIL | FAIL | refused: fixture absent |
| golden present, no tap name matches | 0 | pass | pass |  |
| as above, plus a count of compared taps | 0 | FAIL | FAIL | too few taps |
| rel_l2 printed, only cos asserted | 3 | pass | pass |  |
| as above, plus a count of compared taps | 3 | pass | pass |  |
<!-- /worked -->

Only the sound test, in the first row, tells the mutant from the correct code. Two of the other rows fail on both outputs, and that is the right result for a test that cannot do its job: it should be red, so that the person reading the report learns that nothing was proven. The remaining five rows pass both, and they are five different holes. The early return prints nothing. The skip prints a line that a runner hides unless asked, so the report is the same. The renamed golden is the quietest: the test runs, finds none of the tap names it looks for, skips each one as absent, and compares nothing. The counted version of that row refuses it, because it knows how many taps it meant to compare. And the gate that prints `rel_l2` but asserts only the cosine is the trap of the last section in disguise; no count can find it, because it compares every tap and every comparison passes. Only the mutation in the last column finds it.

A **gate that never runs** is a test whose comparison does not happen, or cannot fail, while the report shows it as passing: a test that returns early when its fixture or device is absent, a check over a list that turned out empty, an assertion that is computed and printed but never made, a test target that is never built or never invoked by the command people run before they call something done. It is worse than no test, because no test at least leaves the question open.

The cures follow from the rows of the table.

*A test that cannot run says so, by name, and a run that must prove something refuses to skip.* In brain a skip goes through one of three functions in `crates/testutil/src/lib.rs` (the legacy early returns that remain are counted, below). `skip(reason)` is for a missing fixture: it prints `SKIP:` and the reason, and it panics when the environment variable `BRAIN_REQUIRE_FIXTURES` is set to a non-empty value other than `0`, so that a run whose purpose is to prove parity fails for every comparison that did not happen. `skip_unavailable(reason)` is for missing hardware, a machine with no graphics processor or no neural accelerator, and no flag can make it fatal, since "the data is on this box" has nothing to say about silicon that is not. `skip_unvalidated_capability(cap, reason)` is the third bucket, for code that exists for a capability this machine does not have: it prints a warning that the result is unvalidated here and may fail where the capability exists, appends a row to a ledger that `make test/capability-report` renders, and is fatal on a machine that names the capability in `BRAIN_REQUIRE_CAPABILITIES`. The doc comment of `skip_unavailable` gives the reason for three functions rather than one flag: which bucket a skip belongs in is a judgement, and a reviewer must be able to check it by reading the call. The Makefile target `test/parity/strict` runs the reference-parity suites with `BRAIN_REQUIRE_FIXTURES=1`, and it is part of the release gate `test/full` for the reason its comment gives: it is the one member of that gate that can fail for want of data, and every other member is green on a machine with no fixtures at all.

The history behind it is a sequence of discoveries that the problem was bigger than thought. Commit `64bcd5d69` (17 August 2026) added the panic under `BRAIN_REQUIRE_FIXTURES` after the parity suite of a video model was found reporting success while 7 of its 9 comparisons of the video autoencoder's stages and its comparison of the real transformer were all skipping, because their weights resolved from the environment. Commit `d2a345388` (the same day) moved 66 test files to the helper and records that the premise was wrong by an order of magnitude: the change had been scoped as seven files, and an audit found 102 files and 195 sites, plus about 55 more phrased differently enough that the original search missed them. Commit `1f7d7333e` (18 August 2026) finished the sweep, 235 sites in 152 files, and found eight sites that were completely silent, a bare early return with no message at all, which no search for the word "skip" could ever have found.

*A list that must not be empty is counted.* A parity test that looks up its taps by name, a check that loops over the tensors a model reports, a suite that loops over registered models: each passes vacuously over nothing. The fix is to know the expected count and assert it, as the counted rows of the table do, or at least to assert that it is not zero. In brain `parity::Table::has` (`crates/testutil/src/parity.rs`) answers whether a stage of a given name was compared, "the check that a golden is not sitting in the fixture with no tap pointed at it", and the conformance suite of the next section asserts that its registry is not empty. A golden that belongs to another checkpoint is a case of the same kind: `golden::Source::require` (`crates/testutil/src/golden.rs`) compares the architectural identity the golden recorded with the checkpoint under test and turns a mismatch into a named skip, fatal under `BRAIN_REQUIRE_FIXTURES`, instead of a comparison against the wrong reference.

*A test that is not built or not invoked is wired in where people look.* Cargo builds a test target that requires a feature only when the feature is turned on, and the reinforcement-learning objectives of brain are gradient-checked by targets that need such a feature; the comment above the Makefile target `test/rl` calls the result of a plain run "not a reported skip, an absence", and the target exists so that the checks run. Knowledge #106 records a gate that existed, was correct and found two backend-only bugs once someone ran it by hand, but was not part of the command people ran before calling something done; commit `2c32bfe72` (11 September 2026) made it part of `test/full`, a fix of one line in the Makefile.

*A count of skips that may only fall.* `scripts/gates/check-test-skips.sh` (`make check/test-skips`, from commit `49865fe93`, 16 October 2026) counts the remaining silent early returns per file against an allowlist and fails on a new one, and also fails when the list claims more than the file has, so the list can only shrink.

*A gate is not trusted until it has been seen to fail.* None of the measures above finds the row whose `rel_l2` is computed and never asserted. A mutation does: plant the defect the gate exists to catch, run the gate, see it fail, revert. Knowledge #47 states the conclusion that the 8-bit tier of the last section forced: "a gate nobody has seen fail is a hypothesis." The idea is old in software testing, where it is called mutation testing (DeMillo, Lipton and Sayward, 1978), and in numerical code it is unusually cheap, because the mutations that matter are few and familiar: a factor of 2, a scale, a transpose, an off-by-one in a mask, a sum over the wrong axis.

```text
function parity_test(golden: map<text, array<real>[N]>, present: bool, taps: list<text>,
                     produce: function(text) -> array<real>[N], prove: bool,
                     cos_floor: real, rel_ceiling: real) -> list<text>:
    if not present:
        require not prove, "this run must prove parity, and the golden is absent"
        print "SKIP: golden absent (name the command that makes it)"
        return []
    require len(taps) > 0, "a test of no taps proves nothing"
    failures = []
    compared = 0
    for name in taps:
        require name in golden, "the golden holds no tap " + name
        for f in parity_failures(produce(name), golden[name], cos_floor, rel_ceiling):
            append name + ": " + f to failures
        compared = compared + 1
    require compared == len(taps), "fewer taps compared than listed"
    return failures
```

Every `require` in it turns a hole of the table into a red result: the absent golden in a run that must prove, the empty list, the tap the golden does not hold. What it cannot do is check itself; that remains the mutation's job.

## One suite that every implementation answers to

A large engine has many implementations of the same promise. Each model computes a loss and its gradient; each serving engine answers a batch of requests; each backend runs the same kernels. Testing each with its own hand-written test leaves a gap wherever one author forgot a case another author remembered, and it multiplies the three traps above by the number of authors. The alternative is to write the tests once, against the promise.

A weighted cross-entropy is a small promise of this kind, the loss that reward-weighted fine-tuning trains through. For a sequence of $N$ positions with logits $\mathbf{z}_i$, targets $t_i$ (some positions ignored) and per-position weights $w_i$, the loss is

$$
L_{\mathrm{w}} = \frac{1}{n_{\mathrm{sup}}}\sum_{i\ \text{supervised}} w_i\,\mathrm{CE}_i, \qquad \mathrm{CE}_i = \mathrm{lse}(\mathbf{z}_i) - z_{i, t_i},
$$

with $n_{\mathrm{sup}}$ the number of supervised positions and lse the log-sum-exp of Volume III. Its gradient with respect to $\mathbf{z}_i$ is $(w_i / n_{\mathrm{sup}})(\mathrm{softmax}(\mathbf{z}_i) - \mathbf{e}_{t_i})$ for a supervised position and zero for an ignored one. Five contracts say what any implementation must do, each of them a property of the promise and not of any one implementation:

- K1: with every weight equal to 1, the loss and the gradient equal those of plain training, the unweighted masked mean that is already proven;
- K2: the gradient agrees entry by entry with central differences of the implementation's own loss, at weights that are neither uniform nor all nonzero (the per-entry check of Volume III, Chapter 6);
- K3: a position of weight 0 contributes a gradient row of exactly 0;
- K4: an ignored position contributes a gradient row of exactly 0;
- K5: the loss equals the definition, evaluated by a separate implementation of the formula above.

Five implementations are put through them: the definition, and four plausible ways to get it wrong.

<!-- worked: testing_numerical_code.conformance_suite -->
Weighted cross-entropy over N = 6 positions and C = 4 classes: L = (1/n_sup) sum over the supervised positions i of w_i (lse(z_i) - z_i[t_i]), n_sup the number of supervised positions. Logits Z (6 x 4) drawn by the book's SplitMix64 with seed 0x075728879255576c (528936043980085100) as normal() in row-major order (to four places: +0.9474 +0.6307 +1.1983 +1.5392; +1.5054 +0.2023 -0.4928 +0.0536; -0.9466 +1.0495 -2.5319 -0.7805; -1.5568 -0.2282 +1.1444 -0.6948; -0.4106 -1.6361 +0.7251 -0.6144; +0.0271 +0.7365 -0.5476 -1.9913); targets t = [2, 0, 3, 1, -1, 2] with -1 marking the one ignored position; weights w_i = 0 when i is a multiple of 3, otherwise 0.5 + 0.1 i, so w = [0, 0.6, 0.7, 0, 0.9, 1]. The five implementations return a loss and its gradient with respect to Z: the definition; plain cross-entropy that ignores w; the weighted sum divided by the sum of the supervised weights instead of n_sup; w applied to the loss but not to the gradient; the ignored position counted as a supervised one of class 0 (dividing by N). K1 compares, at w = 1 everywhere, the loss and every gradient entry with the plain masked mean (within 1e-12); K2 differences the implementation's own loss in each of the 24 logits (h = 1e-05, gate 1e-07 + 1e-05 max(|a|, |n|)); K3 and K4 require the gradient rows of position 0 and 3 (weight 0) and of position 4 (ignored) to be exactly 0; K5 compares the loss with the definition evaluated independently (relative 1e-12). lse is the stable form max + log(sum(exp(z - max))) with the library exp and log, and every sum runs in increasing index, in double precision. The suite ran 25 checks, and the four wrong implementations were also passed as known_bad.

| implementation | K1 | K2 | K3 | K4 | K5 | suite |
|---|---|---|---|---|---|---|
| correct | pass | pass | pass | pass | pass | conforms |
| weights ignored | pass | pass | FAIL | pass | FAIL | rejected |
| divides by the sum of weights | pass | pass | pass | pass | FAIL | rejected |
| weights in the loss only | pass | FAIL | FAIL | pass | pass | rejected |
| mask ignored | FAIL | pass | pass | FAIL | FAIL | rejected |
<!-- /worked -->

The correct implementation passes everything, and every wrong one fails something, which is what a suite is for. What each contract can see on its own is more instructive:

<!-- worked: testing_numerical_code.contract_blindness -->
Each contract of the suite above on its own, against the four wrong implementations:

| contract alone | wrong ones rejected, of 4 | which |
|---|---:|---|
| K1: all-ones weights equal plain training | 1 | mask ignored |
| K2: gradient matches central differences | 1 | weights in the loss only |
| K3: a zero weight gives a zero gradient row | 2 | weights ignored, weights in the loss only |
| K4: an ignored position gives a zero gradient row | 1 | mask ignored |
| K5: loss equals the definition | 3 | weights ignored, divides by the sum of weights, mask ignored |
<!-- /worked -->

K1 is the contract of the degenerate instance. All-ones weights are exactly the input on which "ignore the weights" and "divide by the sum of the weights" coincide with the right answer, because the sum of $n_{\mathrm{sup}}$ ones is $n_{\mathrm{sup}}$; it catches only the implementation that forgot the mask, whose fault does not depend on the weights. K2 is the limit of the gradient check that Volume III, Chapter 6 stated: it proves that the gradient is the derivative of the loss the implementation computed, and three of the four wrong implementations compute a wrong loss and its exact derivative. Knowledge #3 says it in one line, that finite differences gate the backward pass against whatever forward pass is emitted. K3 and K4 are structural, cheap, and blind to everything that leaves zeros alone. K5 catches the most, three of the four, because it is the only contract that consults an independent reference, and it still misses the implementation whose loss is right and whose gradient is not. No contract catches more than three, and the five together catch all four.

A **conformance suite** is a set of contracts written once against an interface, run unchanged over every registered implementation of that interface, each on a small instance of its own with distinct sizes, so that every implementation answers the same questions and a new one joins by registering rather than by writing tests. Its everyday sense, the conformance tests by which an implementation of a published standard shows that it follows the standard, is the same idea within one program.

A suite is a test like any other and falls into the same traps, so its runner guards against them in the same ways: it refuses an empty registry, counts the checks that ran, and checks that subjects known to be wrong are rejected, a negative control that proves the suite can fail.

```text
function run_conformance(subjects: list<Subject>,
                         contracts: list<function(Subject) -> bool>,
                         known_bad: list<Subject>) -> array<bool>[R, C]:
    # R = len(subjects), C = len(contracts);
    # entry [s, k] is true when subject s honours contract k
    require len(subjects) > 0, "a suite over no subject proves nothing"
    require len(contracts) > 0, "a suite of no contract proves nothing"
    results = zeros(bool, [len(subjects), len(contracts)])
    ran = 0
    for s in 0 .. len(subjects):
        for k in 0 .. len(contracts):
            results[s, k] = contracts[k](subjects[s])
            ran = ran + 1
    require ran == len(subjects) * len(contracts), "a contract did not run"
    for b in known_bad:
        honoured = [contracts[k](b) for k in 0 .. len(contracts)]
        require not all(honoured), "a known-wrong subject conforms"
    return results
```

A `Subject` is whatever the interface needs, here a function from logits, targets and weights to a loss and a gradient; in the worked table all five implementations are registered and the four wrong ones are also passed as `known_bad`, so the run ends in no `require` failure only because every one of them is rejected. The cost is easy to count. K2 is the expensive contract, two loss evaluations per logit entry and one for the gradient, 49 for the 24 logits here; the others need one or two each. For a real model the gradient contract is bounded by tensors and directions, not by entries, which is what the budget of brain's suite, below, does.

brain has two suites of this shape. `crates/model/src/conformance.rs` (built in commits `cb184dec5` and `2f0a40d89`, 16 October 2026) holds the training contracts that every model declaring the capability must honour, numbered in its module comment (the objective-level check C7 is in `crates/rl/src/conformance.rs`), among them the two that the worked table copies: C2, an all-ones weighted batch reproduces plain training, and C3, the weighted loss's gradient matches finite differences with non-uniform weights including exact zeros, so that the weights are also tested away from the all-ones instance of C2 (`crates/model/src/conformance_grad.rs`). A model registers a `Subject`: a tiny configuration whose doc comment demands distinct, non-square dimensions, a declaration of which capabilities it offers, and a `Budget` that bounds the finite differences by tensors and by random directions per tensor. The declaration is checked in both directions: the suite fails a model that offers a capability it says it does not, and one that says it offers what it does not. The registration lives in `crates/rl/tests/conformance.rs`, one line per model, with a list `REGISTERED` that a test asserts is not empty and that every registered subject must appear in, a fake model that every objective must refuse with a typed error, and two negative controls: a correct model given a false declaration must fail the suite, and a configuration that silently trains the base weights instead of the adapter weights must fail its audit. The Makefile target `test/conformance` runs it, and it is part of `test/full`.

The second suite covers serving. `model::serve::conformance` (commit `8a708e7e7`, 19 October 2026) runs, over any engine that implements the scheduler's decoder interface, properties the scheduler promises (the module has since gained more, for scoring, embedded prompts and speculation): that a batched request produces what the same request alone produces, that a short request is not held behind a long one, that a cached prefix answers as a cold one does, and that adapter requests never share cached keys and values with each other or with the base model. Beside it, `scripts/gates/check-serve-conformance.sh` (`make check/serve-conformance`) closes the registry's last hole, an implementation that nobody registered: it finds every implementation of the interface in the source and fails unless a test of that crate names the engine and calls the suite, or the implementation carries a `conformance-exempt:` comment whose reason names the test that covers it instead. A suite that an implementation can avoid by not joining is a gate that never runs for that implementation.

## The gradient checker and the test helpers of brain

Two crates hold most of what this chapter describes, and each is easier to read with the traps in mind.

`crates/gradcheck` is the gradient check of Volume III, Chapter 6, applied to every model brain trains, and its design records the traps one by one. The assertion every model check goes through, `assert_grad_gate` in the test module of `crates/gradcheck/src/lib.rs`, makes two assertions, not one: every tensor within the shared tolerance $(4 \times 10^{-3}, 8 \times 10^{-2})$, and no dead gradient, an analytic value of exactly 0 against a numeric value above $10^{-3}$, the case the absolute floor of the tolerance waves through (`Report::dead_gradients`); `zero_grad_params` is the structural guard that returns every parameter whose whole gradient came back within $10^{-9}$ of zero. Both came from commit `83990db00`, the same change that lifted the image autoencoder's check over its size threshold and that replaced, for a group of split matrix-product kernels, an "oracle" that was a print statement, "a comparison that cannot fail", by tests against 64-bit references. Several checks exist to avoid a degenerate instance. `check_qwen3_weighted` uses non-uniform weights including exact zeros so that, in its doc comment's words, the check does not degenerate to the all-ones case; `check_gpt_tied` asserts the unused rows that make the tied gradient visible; the harness of the hybrid decoder redraws the weights in front of its normalisation. Each model's module begins with a comment stating which parameters the check covers and which are frozen, because a passing check covers only what it touched. Its device-dependent tests open with `brain_testutil::gpu_tests_skipped()`, the one reader of the variable that lets a machine without a device skip them, which names the test it skips and is fatal under `BRAIN_REQUIRE_FIXTURES` (commit `49865fe93`). And the integration tests in `crates/gradcheck/tests/it/` are linked as one binary whose `main.rs` declares each file as a module, so a file of tests is compiled only once it is declared there.

`crates/testutil` is the shared kit of brain's tests, and its first function explains why it exists: `testdata`, which resolves a fixture's path, replaced 36 copies of the same four lines, one per crate, and the crate's doc comment names the risk of drift between copies as the reason. The rest of it is the machinery of this chapter. The three skip functions and `env_lock`, a single lock that every test which changes the process environment must hold, since tests of one binary run on parallel threads and two of them that set and clear the same variable would race each other's assertions, a test that passes or fails by timing. `parity.rs` holds `compare`, which accumulates the cosine and `max_abs` in 64 bits so that a long tensor's dot product keeps the low bits it is asked to certify; `rel_l2`, which returns infinity for a nonzero output against a reference of zeros, "as maximally wrong rather than as a perfect score"; and three report types that gate on a cosine floor (`Report`, which refuses a reference tap that is all near zero as degenerate and treats a cosine that is not a number as a failure) or on a cosine floor and a `rel_l2` ceiling together (`Table` and `WorstTable`). `golden.rs` holds the provenance check of a golden. `rounding.rs` derives a tolerance from the round-off bound of a sum of $k$ terms in 32 bits, $\gamma_k = k u / (1 - k u)$ with $u = 2^{-24}$ (Higham, 2002, Chapter 3), and refuses $k u \ge 1$, where the bound says nothing: the parity tolerance of Volume VI, Chapter 2, as a function. `fd.rs` holds a directional check that refuses to compare where it cannot: before it compares a direction it compares the two one-sided differences along it, and a direction whose two sides disagree by more than the tolerance has a kink of a piecewise-linear activation inside its step and is drawn again, up to eight times, after which the check reports the kink instead of a verdict. And `hfquant.rs` writes quantized checkpoint fixtures with encoders that, in its own words, share no code with brain's codecs, so that a load can be checked against values the fixture decoded independently: the reference does not share the code it is meant to judge, which is the second link of the chain closed by construction.

That second link deserves its sentence, because it is the trap this chapter has named only in the figure. A reference built from the same simplification as the code under test agrees with it whatever both get wrong. Knowledge #48 records a video model whose components had each passed parity against the reference, some bit for bit, whose reduced-depth tests agreed with the reference's goldens at cosine 0.999999 or better, and whose output was still noise: the positions fed to its rotary embedding were indices of the latent grid where the reference uses pixel coordinates, and the generator of those goldens had been written, as a documented simplification, with the same grid indices. The lesson there is the general one for references: a golden is only as independent as the code that produced it, and the part of a pipeline that a golden does not exercise is untested however good the agreement.

## Check your understanding

1. **Explain.** A test of a layer normalisation runs on a batch in which every row is the same vector. Name a mistake in the code that this instance cannot see, and the change to the instance that fixes it.
2. **Predict.** A parity test compares a model's next-token probabilities with a reference's, `max_abs` at most $10^{-5}$. The model's output layer adds a bias that the reference does not have, the same bias for every vocabulary entry. Does the test fail? What should it compare instead?
3. **Find the error.** "The gate asserts cosine at least 0.9999 and `rel_l2` at most $10^{-4}$ on the output of the block, after the final RMS norm, so a mis-scaled attention projection inside the block cannot pass." Where is the gap, and where should the tap be?
4. **Find the error.** "Our suite has 412 tests and all of them passed on the release machine, so every reference comparison matched." Name two ways this can be true while no comparison ran, and what the release run should set or check.
5. **Explain.** In the conformance table, K2 passes the implementation that ignores the weights. Why does a correct gradient check pass it, and which contract is needed to reject it?
6. **Predict.** A conformance suite's registry is a list of model names, and one model's registration line is deleted by accident. What happens to that model's tests, and which two guards of this chapter would turn the deletion into a failure?
7. **Explain.** Why is a planted mutation stronger evidence that a gate works than a review of the gate's code, and why is it cheap for numerical code in particular?

## Summary

A numerical test is a chain of an instance, a reference, a measure with a threshold and a verdict, and a wrong program passes when any link has a hole. The **degenerate-dimension trap** is an instance on which a wrong program computes the right function: a size of 1 or two equal sizes, values with a symmetry or a restricted range (identical tokens, all-ones weights, inputs of one sign), a size below the threshold at which the code switches path, magnitudes starved below a fixed constant, or the coincidences of a real model's sizes. On one token an attention head with three mistakes is exact; on identical tokens four of six planted mistakes vanish; on distinct sizes with random values every one is caught. The defence is to plant the mistakes and see which ones an instance cannot tell apart, and to assert the instance's non-degeneracy in the test itself.

The **scale-invariant-metric trap** is a gate on a measure that is invariant under what a plausible mistake does: cosine under a scale, top-1 agreement under anything that keeps the order of a row (it caught 0 of 5 errors in the table), probabilities under a constant shift, and any comparison under what a normalisation upstream of it removes. Every gate asserts a measure with trivial invariance, `rel_l2` or `max_abs`, on a tap upstream of every normalisation, and keeps cosine for diagnosis.

A **gate that never runs** compares nothing and is reported as a pass: an early return, a skip that only prints, an empty list, an assertion computed and not made, a target never built or never invoked. The defences are skips that name themselves and become failures in a run that must prove something, counts of what was compared, a ratchet on the number of silent skips, wiring into the command people run, and the mutation that shows each gate can fail.

A **conformance suite** writes the contracts of an interface once and runs them over every registered implementation on its own distinct instance; it refuses an empty registry, counts its checks, rejects known-wrong subjects and, in brain, is enforced by a gate that finds implementations nobody registered. In the weighted cross-entropy example no single contract caught more than three of four wrong implementations and the five together caught all four: the degenerate contract catches one, the gradient check sees only inconsistency between a loss and its gradient, and only an independent reference value sees a wrong objective.

These defences make a test trustworthy about correctness. A test of speed has traps of its own, a clock that measures the wrong thing and a peak that was never reached, and the next chapter, Benchmarking and Performance Claims, takes them up.

## References

- DeMillo, R. A., Lipton, R. J. and Sayward, F. G., Hints on test data selection: help for the practicing programmer, *Computer* 11(4), 34-41, 1978.
- Higham, N. J., *Accuracy and Stability of Numerical Algorithms*, second edition, 2002.

## Exercises

1. **[Calculate]** A single-head attention test uses $T = 2$, $d_k = 2$, $d_v = 2$ with $\mathbf{Q} = \mathbf{K}$. Which of the six mutants of `attention_variant` are certainly invisible on it, whatever the values, and which depend on the values?
2. **[Calculate]** For the reference logits of `testing_numerical_code.invariant_measures`, row 0 shifted by 0.3: compute $1 - \cos$ for that row alone, its `rel_l2`, and the largest change in its softmax probabilities. Explain each from the invariances listed in the chapter.
3. **[Calculate]** An output is $\mathbf{a} = s\,\mathbf{b}$ with $s = 1 + \delta$. Show that `rel_l2` is $|\delta|$ and $1 - \cos$ is 0, and find the smallest $|\delta|$ the gate of `parity_failures` with a ceiling of $10^{-5}$ rejects.
4. **[Calculate]** The weighted cross-entropy with all weights equal to $c > 0$: compare the loss of the correct implementation, of the one that divides by the sum of the weights, and of the one that ignores the weights. For which $c$ do all three agree, and what does that say about K1?
5. **[Implement]** Implement `attention_variant` and `blind_mutants` and reproduce `testing_numerical_code.degenerate_instances` with the generator, seed and draw order printed with it. Then add an instance with $T = 4$, $d_k = 4$, $d_v = 4$ and report which variant becomes shape-legal.
6. **[Implement]** Reproduce `testing_numerical_code.invariant_measures`. Then add a seventh gate, the Pearson correlation of the flattened outputs at least 0.9999, work out its invariance before you run it, and check your prediction against the six rows.
7. **[Implement]** Implement the five weighted cross-entropy implementations, the five contracts and `run_conformance`, and reproduce `testing_numerical_code.conformance_suite` and `testing_numerical_code.contract_blindness`. Then add a sixth implementation that applies the weights to the gradient but not to the loss, and say which contracts reject it.
8. **[Implement]** Reproduce `testing_numerical_code.silent_gates`: write the sound state as `parity_test` and each other state as the one change its row describes. Then add a state in which the golden holds the right tap names but every tap of the reference is all zeros, and say which `require` of the pseudocode stops it.
9. **[Investigate]** Read `crates/model/src/conformance.rs` and `crates/model/src/conformance_grad.rs`. For each contract of the module comment (C1 to C4 and C6 to C10), say which of the three traps of this chapter it guards against, and which contract plays the role of K5, the independent reference value, if any does.
10. **[Investigate]** Read `scripts/gates/check-serve-conformance.sh`. Describe precisely what it searches for, and construct an implementation of the interface that it would accept while the suite never runs over it, or argue that none exists.

## Key terms

- **degenerate-dimension trap**: a test instance on which some wrong program computes the same function as the right one, because a size is 1, two sizes are equal, a size lies below the threshold where the code changes path, or the values have a symmetry or a restricted range, so the test cannot fail for that mistake at any tolerance. <!-- concept: degenerate-dimension-trap -->
- **scale-invariant-metric trap**: a test that gates on a measure unchanged by a transformation that a plausible mistake applies to the output (cosine under a scale, top-1 agreement under an order-preserving change, probabilities under a constant shift of the logits, any comparison after a normalisation), so the mistake scores as a perfect match. <!-- concept: scale-invariant-metric-trap -->
- **gate that never runs**: a test whose comparison does not happen or cannot fail while the report shows it as passing, such as an early return on a missing fixture, a check over an empty list, an assertion computed but never made, or a target that is never built or invoked. <!-- concept: gate-that-never-runs -->
- **conformance suite**: a set of contracts written once against an interface and run unchanged over every registered implementation on a small instance of its own, which refuses an empty registry, counts its checks and rejects known-wrong subjects. <!-- concept: conformance-suite -->

## Answers

**Recall.**

1. $\cos(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b} / (\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert)$, a number from $-1$ to $1$; multiplying either vector by a positive number leaves it unchanged.
2. It compares the gradient the backward pass returns with finite-difference estimates of the same forward pass, entry by entry or along directions, under a stated step and tolerance. A passing check proves the derivative of the forward pass that was run, not that the forward pass computes the intended objective.
3. Every size distinct from every other and none equal to 1 (small primes such as 3, 5, 7, 2), so that no two roles are interchangeable, no sum has one term, and a transpose changes the shape.
4. From the rounding model: the count of roundings along the longest chain of dependent operations times the unit round-off $u$. A tolerance read off the run passes by construction and says nothing about the mistakes it admits.

**Check your understanding.**

1. Any mistake that confuses one row with another: the mean and variance of row 0 used for every row, or a row stride off by one that reads a neighbour's statistics, gives exactly the right output when all rows are equal. Draw every row independently, so that no two rows share their statistics.
2. It does not fail: a constant added to every logit of a row leaves the softmax unchanged, so the probabilities are equal to rounding. Compare the logits themselves with `rel_l2` or `max_abs`.
3. The final RMS norm divides a scale factor out of everything after it, and the residual dominates what remains, so a scaled attention projection can change the block output by less than the tolerance; this is knowledge #47. Tap the attention sublayer's own output, before the norm and before the residual add, and assert `rel_l2` there.
4. Tests that return early or skip when their fixture or device is absent are reported as passes, and a target that needs a feature or is not part of the release command never runs. The release run sets the variable that makes a missing-fixture skip fatal (`BRAIN_REQUIRE_FIXTURES` in brain), and the report is read for skip lines and compared counts, not for the number passed.
5. That implementation's loss is plain cross-entropy and its gradient is the exact derivative of that loss, so the finite differences agree with it. A contract that consults something outside the implementation is needed: K5, the loss against the definition, rejects it (and K3 does, since its weight-0 rows are not zero).
6. Nothing fails: the model is simply not tested, a gate that never runs for it. A list of the expected models that every registered subject must appear in and that is checked against the models in the source (as `check-serve-conformance.sh` does for serving engines) turns the deletion into a failure; an assertion that the registry is not empty catches only the extreme case of deleting all of them.
7. A review shows what the gate's author meant it to check; a mutation shows what it actually rejects, and knowledge #47 is a gate that looked complete to an author who knew the trap and still passed the planted defect. In numerical code the likely mistakes are few and well known (a factor, a scale, a transpose, a mask off by one, a wrong axis), so a handful of mutations covers most of what a gate is for, at the cost of one run each.
