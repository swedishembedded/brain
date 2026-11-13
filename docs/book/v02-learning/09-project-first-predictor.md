<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Project: A Predictor You Can Trust

*Can you train a model on real measurements, evaluate it honestly and explain exactly where it fails?*

Eight chapters have taught ideas about learning: a line fitted by solving equations and by descent, a probability fitted by cross-entropy, a penalty as a prior, a split that keeps the test set honest, a metric chosen for the question, a classical method as a comparison, an experiment whose result someone else can repeat. This chapter asks you to turn them into one predictor and one report. You will write, in a language of your choice, on top of the numerical library you built at the end of Volume I, a program with seven parts:

1. **Honest splits.** Divide the data by pump, never by row, draw cross-validation folds of whole pumps, and prove with a leakage check that no pump is on both sides of a split.
2. **Baselines.** A trivial model that uses no features and a simple model that uses one, for both of the targets below.
3. **Linear regression** for the continuous target, solved by the normal equations and by gradient descent, which must agree.
4. **A regularized logistic classifier** for the binary outcome, with its penalty chosen by cross-validation.
5. **Evaluation** (Chapter 6). Accuracy, precision, recall and $F_1$ at a threshold derived from costs; the area under the ROC curve computed two ways; a reliability table, the expected calibration error and temperature scaling.
6. **One classical comparison model**, $k$ nearest neighbours.
7. **A report** (Chapter 8). Effect sizes with bootstrap intervals, a multi-seed study with a paired comparison, an ablation of one design choice, and an honest statement of where the model fails.

Nothing here is new. Every part was taught; the project asks whether the parts work together, and, more than any chapter, whether you will believe a number before it has been checked. The failures it is designed to expose are the ordinary ones: a split that lets the model recognise a machine instead of learning a pattern, a baseline that is weaker than it looks, a probability that is confident without being right, a comparison that one seed could have reversed, and a test set that quietly stopped resembling the world.

## Three hundred and sixty pumps

A plant keeps 360 pumps in two sites. Each pump is tested four times, and each test records five measurements (load, temperature, vibration, age and humidity) together with two results: whether the pump failed during the following season (`fail`, 0 or 1) and how much it wore (`wear`, a number). The goal is to predict both from the five measurements, for pumps the model has never seen. Site A holds 300 pumps and site B, a second plant with heavier loads, holds 60. The book plays the part of nature: the data are generated from a stated law with the book's own generator (the milestones print the law and the order of every draw), so that you can regenerate every row in your own language and compare your numbers with the ones printed here.

Each milestone below says what to build, in the book's pseudocode, which fixes the interface and nothing else. It says why the piece matters to the steps after it, names a check to run before moving on, and warns about the mistakes readers actually make. The **acceptance tests** at the end give inputs and the outputs a correct implementation produces, written so that any language can be tested against them. Every expected value was computed by a second, independent implementation, `scripts/book/worked/project_predictor.py`, and is regenerated whenever the book is built. Use only your language's standard library plus the library of Volume I (the generator, the gradient check, `percentile`, `correlation` and the bootstrap).

![Seven boxes labelled with the milestones 1 to 7. Milestone 1 sits at the top with arrows to milestones 2, 3 and 4 in a row below it; milestone 4 leads down to milestone 5 and then to milestone 6 in the right column; milestone 7, highlighted, sits at the bottom centre and receives arrows from milestones 2, 3, 5 and 6.](img/predictor-milestones.svg)

The figure shows which milestone feeds which, so the order 1 to 7 is a valid build order. Milestone 1 is used by every other one: the splits, the folds and the standardiser are the ground the rest stands on. Build and test each milestone before starting the next. A failure in milestone 7 that traces back to milestone 1 costs an afternoon; the same failure caught by milestone 1's own check costs a minute.

Before the first milestone, six questions bring back what the project leans on. They reach across the whole of Volume II, and two reach into Volume I. Answer each from memory before reading on; the answers are at the end.

1. (Chapter 5, [Generalization, Overfitting and Honest Evaluation](05-generalization.md)) A dataset has several rows from each of many machines. A model is scored on rows chosen at random for testing and again on rows from machines it never saw. Which score is higher, and what does the difference measure?
2. (Chapter 2, [Linear Regression and the First Descent](02-linear-regression.md)) For a design matrix $\mathbf{X}$ and targets $\mathbf{y}$, what equation do the least-squares weights satisfy, and what decides whether gradient descent on the mean squared error is stable?
3. (Chapter 3, [Classification and Logistic Regression](03-logistic-regression.md), and Chapter 4, [Maximum Likelihood, Priors and Regularization](04-likelihood.md)) Write the gradient of the mean binary cross-entropy plus $\frac{\lambda}{2}\lVert\mathbf{w}\rVert^2$ with respect to one weight. Which weight is usually left out of the penalty, and why?
4. (Chapter 6, [Measuring Models: Metrics and Calibration](06-metrics.md)) A missed failure costs $5$ and a false alarm costs $1$. At which probability threshold should a calibrated model raise the alarm? What does the area under the ROC curve count?
5. (Chapter 8, [Experiments, Baselines and Claims](08-experiments.md)) Two models are compared on the same split, then again on the same $16$ splits. Why is the second comparison paired, and why must both models see the same splits?
6. (Volume I, Chapter 12, [Statistics: Learning Facts from Samples](../v01-mathematics/12-statistics.md), and Chapter 17, [Randomness on a Deterministic Machine](../v01-mathematics/17-randomness.md)) A bootstrap resamples units with replacement. Why must the units be independent, and what is the unit when four rows come from one pump?

## From honest splits to the report

A note on language and style. The pseudocode follows [Pseudocode Conventions](../pseudocode.md): 0-indexed, row-major, explicit broadcasts, explicit randomness, `require` for every input assumption. Use 64-bit floats. Write the checks of each milestone as a test file that you rerun after every change, and compare tables to the digits printed (a difference below half a unit in the last printed place), unless a milestone states another tolerance. Every generator below is the book's SplitMix64, started at the printed seed with `Generator(seed)`. Every random number is drawn in the printed order, so a reader whose generator is right sees the same numbers.

Two letters are used for different things in different milestones, as in Chapters 2 and 3. In milestone 3, $\mathbf{y}$ is the vector of the continuous target `wear`. In milestones 4 to 7, $y_n \in \{0, 1\}$ is the binary outcome `fail`. The tables name the column.

### The data you receive

The first piece of code is a function that generates the dataset, and nothing else yet. The law is stated here because you must regenerate the data; a real project would not know it, and the project does not use it to choose features or models. It appears again in two places only: in milestone 3, as the smallest error that any model could reach on a new pump (the noise of the law itself), which the regression is compared with, and in milestone 7, where it explains one failure.

There are $G = 360$ pumps, numbered $0, \dots, 359$. Pumps $0$ to $299$ are site A and pumps $300$ to $359$ are site B. Pump $g$ has four rows, numbered $4g, \dots, 4g + 3$, so there are $1440$ rows. Draw, for $g = 0, \dots, 359$ in order, from one generator:

1. the **profile** of the pump, $\mathrm{prof}_{g0}, \dots, \mathrm{prof}_{g4}$ (the centre of its measurements, one number per feature): five calls of `rng.normal()`;
2. the pump effect $\mathrm{eff}^{\mathrm{fail}}_g = 1.5 \cdot \mathrm{rng.normal()}$ on the failure logit;
3. the pump effect $\mathrm{eff}^{\mathrm{wear}}_g = 0.4 \cdot \mathrm{rng.normal()}$ on the wear;
4. for each of the pump's four rows, in order: five calls of `rng.normal()`, the $j$-th giving the measurement $x_j = \mathrm{prof}_{gj} + 0.2\,\mathrm{rng.normal()}$; then, for pumps of site B only, $x_0 \leftarrow x_0 + 2.0$; then the true logit $e^\star$ of the row and the draw

$$
\begin{aligned}
e^\star &= -0.5 + 2.0\,x_0 - 1.2\,x_1 + 0.8\,x_2 - 0.8\,x_0^2 + \mathrm{eff}^{\mathrm{fail}}_g,\\
\mathrm{fail} &= \begin{cases} 1 & \text{if } \mathrm{rng.uniform()} < \sigma(e^\star), \\ 0 & \text{otherwise;} \end{cases}
\end{aligned}
$$

   then one more call of `rng.normal()`, $\varepsilon$, and the wear

$$
\mathrm{wear} = 3.0 + 2.0\,x_0 - 1.0\,x_1 + 0.5\,x_2 + \mathrm{eff}^{\mathrm{wear}}_g + 0.5\,\varepsilon.
$$

```text
function generate(rng: Generator) -> (X: array<real>[1440, 5], fail: array<int>[1440],
                                      wear: array<real>[1440], pump: array<int>[1440]):
    X = zeros(real, [1440, 5])
    fail = zeros(int, [1440])
    wear = zeros(real, [1440])
    pump = zeros(int, [1440])
    for g in 0 .. 360:
        profile = [rng.normal() for j in 0 .. 5]       # five draws, in order
        effect_fail = 1.5 * rng.normal()
        effect_wear = 0.4 * rng.normal()
        for r in 0 .. 4:
            n = 4 * g + r
            for j in 0 .. 5:
                X[n, j] = profile[j] + 0.2 * rng.normal()
            if g >= 300:
                X[n, 0] = X[n, 0] + 2.0
            e_true = -0.5 + 2.0 * X[n, 0] - 1.2 * X[n, 1] + 0.8 * X[n, 2] - 0.8 * X[n, 0] ^ 2 + effect_fail
            fail[n] = 1 if rng.uniform() < sigmoid(e_true) else 0
            wear[n] = 3.0 + 2.0 * X[n, 0] - 1.0 * X[n, 1] + 0.5 * X[n, 2] + effect_wear + 0.5 * rng.normal()
            pump[n] = g
    return (X, fail, wear, pump)
```

(Letters in this chapter: $z_{nj}$ is reserved for the standardised measurements of milestone 1, so the logit that Chapters 3 and 6 write $z$ is written $e$ here, $e^\star$ for the logit of the law and $\sigma(a\,e)$ for a rescaled one; the pump effects carry labels, not the letters $u$ and $v$, which the book uses for other things.) Three facts about the source matter for what follows. The rows of one pump are alike (their spread around the profile is $0.2$, while the profiles themselves spread by $1$), and the pump effects are shared by its four rows, so the four rows are not four independent examples. Features $x_3$ and $x_4$ have no effect on either result. And the load $x_0$ enters the failure probability through a curved term $-0.8\,x_0^2$, which a model linear in the measurements cannot represent; at site A the curve is gentle over most of the data, at site B, whose loads are $2.0$ higher, it is not.

Everything downstream depends on regenerating exactly these rows, and on knowing what the unit of independence is: the pump.

To check the generator, seed it with the printed seed. The table gives rows you can compare, and the sums let you test the rest.

<!-- worked: project_predictor.dataset_fingerprint -->
Generator: the book's SplitMix64, seed 0x6678aaee123fc739 (7383839528559036217); the draws are those of `generate` in the order printed in the chapter. Selected rows (x0 .. x4, then fail and wear):

| row | pump | x0 | x1 | x2 | x3 | x4 | fail | wear |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | -0.336310 | -0.060979 | -0.056807 | +1.635146 | +1.578391 | 0 | +1.789348 |
| 1 | 0 | -0.483398 | -0.083118 | -0.083159 | +1.783889 | +1.535122 | 0 | +1.758760 |
| 2 | 0 | -0.326185 | -0.569788 | +0.322002 | +1.500731 | +1.639031 | 1 | +2.349518 |
| 3 | 0 | -0.637088 | -0.108087 | -0.176494 | +1.495540 | +1.359129 | 0 | +1.564872 |
| 4 | 1 | -2.120150 | -0.102250 | +1.314263 | -0.481486 | +0.314950 | 0 | -0.891108 |
| 1199 | 299 | +1.745556 | +1.278546 | -1.281036 | -0.859609 | +1.515464 | 0 | +4.518587 |
| 1200 | 300 | +1.684460 | +0.375463 | +0.654254 | +0.920714 | +2.107920 | 0 | +6.328378 |
| 1439 | 359 | +3.362109 | +0.089771 | +0.760320 | -0.204692 | -1.460333 | 0 | +10.255229 |

Trace of the first draws (pump 0, row 0): profile = (-0.305871, -0.184682, -0.009254, +1.515063, +1.316486), pump effects effect_fail = +0.553026 and effect_wear = -0.386881 (the standard deviations 1.5 and 0.4 already applied), true logit e* = -0.682348, label draw 0.784715, target noise draw -0.367455 (before the factor 0.5).

| quantity | value |
|---|---:|
| rows, site A (pumps 0 to 299) and site B (pumps 300 to 359) | 1200 and 240 |
| sum over all rows of x0, x1, x2, x3, x4 | +488.449270, +15.404431, +61.111885, -59.132319, -56.598423 |
| rows with fail = 1, site A and site B | 468 and 107 |
| sum of wear over all rows | +5321.495936 |
| sum of the pump effects effect_fail (pumps 0 to 359) and effect_wear | +41.163717 and +1.326514 |
<!-- /worked -->


### Milestone 1: honest splits and a leakage check

This milestone builds the splitting machinery of Chapter 5, applied to groups, together with the checks that it did what it should.

```text
function fold_assignment(n: int, k: int, rng: Generator) -> array<int>[n]      # Chapter 5
function split_pumps(G: int, n_train: int, n_val: int, rng: Generator)       # the pumps are 0 .. G - 1
        -> (train: array<int>, val: array<int>, test: array<int>):
    require n_train + n_val < G, "no pumps left for the test set"
    order = [g for g in 0 .. G]
    shuffle(order, rng)
    return (sort(order[0:n_train]), sort(order[n_train:n_train + n_val]), sort(order[n_train + n_val:G]))
function rows_of(pumps: array<int>[m], pump: array<int>[N]) -> array<int>      # row numbers, ascending
function fit_standardiser(X: array<real>[N, d]) -> (mean: array<real>[d], sd: array<real>[d])
        # sd has divisor N; require every sd > 0
function standardise(X: array<real>[N, d], mean: array<real>[d], sd: array<real>[d]) -> array<real>[N, d]
function check_disjoint(a: array<int>, b: array<int>):
    for g_a in a:
        for g_b in b:
            require g_a != g_b, "a pump appears on both sides"
```

Site A has $300$ pumps: `split_pumps(300, 180, 60, rng)` gives $180$ training, $60$ validation and $60$ test pumps. The $60$ pumps of site B are never split; they form the **shift slice** that no earlier step looks at, and the report reads them last. The $180$ training pumps are dealt into $k = 5$ folds of whole pumps: number them $0, \dots, 179$ in ascending pump order, call `fold_assignment(180, 5, rng)` with its own generator, and give every row the fold of its pump. The picture shows the layout.

![Boxes proportional to the number of pumps. The top row divides the 360 pumps into a training box of 180 pumps, a validation box of 60 and a test box of 60 (together site A, 300 pumps) and a shift box of 60 pumps (site B). Below the training box, five equal boxes of 36 pumps are the cross-validation folds.](img/pump-splits.svg)

The standardiser (the z-score of Chapter 2) is fitted on a set of training rows and applied unchanged to everything else. This is the rule of Chapter 5 for every statistic computed from data: it is a parameter of the model, and the model is fitted on the training rows.

Every score in the report is a statement about pumps the model has not seen. If rows of one pump sit on both sides of a split, every later number is optimistic in a way that no later step can detect.

Three checks confirm the work, in this order. First, the sizes and the folds: the table below prints the generator seeds, the sets, the shared-pump count (which must be zero for every pair) and the rows per fold. Second, the standardiser: applied to the training rows it gives columns of mean $0$ and root mean square $1$; applied to the test rows it gives columns whose mean is *not* $0$, which is what you want, because the test rows must not influence the numbers they are scaled by. Third, the **leakage probe**: a model that can only memorise (label each held-out row with the label of its nearest training row) is scored under two protocols, a random split of rows and a split of pumps. If the first is clearly better, rows of one pump are leaking across it.

<!-- worked: project_predictor.split_and_probe -->
Generator for the split: the book's SplitMix64, seed 0x149d40d46e5ee440 (1485414733231088704). The pump numbers 0 .. 299 are shuffled (299 integer draws, the book's shuffle); the first 180 shuffled numbers train, the next 60 validate, the last 60 test; each set is then sorted ascending. Rows of a set are those of its pumps in ascending row number.

| set | pumps | rows | rows with fail = 1 | first six pumps |
|---|---:|---:|---:|---|
| training | 180 | 720 | 274 | 0, 2, 3, 4, 5, 7 |
| validation | 60 | 240 | 108 | 8, 17, 21, 24, 34, 38 |
| test | 60 | 240 | 86 | 1, 6, 14, 18, 23, 25 |
| shift slice (site B) | 60 | 240 | 107 | 300, 301, 302, 303, 304, 305 |

Pumps shared between training and validation, training and test, validation and test: 0, 0, 0.

Folds of the 180 training pumps: the book's SplitMix64, seed 0xd36d816960b808bd (15234975404058478781); the training pumps, numbered 0 .. 179 in ascending pump order, are shuffled (179 integer draws) and the pump at shuffled position p goes to fold p mod 5. Pumps per fold: [36, 36, 36, 36, 36]. Rows per fold: [144, 144, 144, 144, 144]. First eight training pumps and their folds: 0: 3, 2: 1, 3: 2, 4: 4, 5: 1, 7: 1, 9: 0, 10: 2.

Leakage probe. A model that can only memorise: label each held-out row with the label of its nearest training row (Euclidean distance on the training-standardised features, the standardiser fitted on the training rows of that protocol; on a tie the training row listed first wins; continuous data have no ties, so this never matters here). One draw of 240 scored rows is noisy, so the probe is repeated 20 times. Master generator: seed 0x90058fde7b56c620 (10377859102059972128); repetition r uses the r-th raw value (rng.next(), counting from 0) as the seed of its own SplitMix64, which draws, in order, the shuffle of the 1200 site A row numbers (1199 integer draws; random-row protocol: the first 720 shuffled rows train, listed in shuffled order, and the next 240 are scored) and then the shuffle of the 300 site A pump numbers (299 integer draws; pump protocol: the rows of the first 180 shuffled pumps train, listed in ascending row number, and the rows of the last 60 shuffled pumps, in ascending row number, are scored). The sd over repetitions has the divisor 20 - 1.

| protocol | mean accuracy of the nearest-row label | sd over repetitions | lowest | highest |
|---|---:|---:|---:|---:|
| random rows | 0.7465 | 0.0296 | 0.6958 | 0.7958 |
| whole pumps | 0.6410 | 0.0307 | 0.5625 | 0.7000 |
<!-- /worked -->

<!-- worked: project_predictor.standardiser_check -->
The standardiser is fitted on the 720 training rows (mean, and standard deviation with divisor N) and applied unchanged to the test rows. The mean of the training z is zero up to round-off (about 1e-17, whose digits depend on the order of summation), so the table tests it against 1e-12.

| feature | training mean | training sd | abs(mean of training z) below 1e-12 | rms of training z | mean of test z |
|---|---:|---:|---:|---:|---:|
| x0 | 0.013543 | 1.041330 | yes | 1.000000 | -0.162007 |
| x1 | 0.016727 | 0.970340 | yes | 1.000000 | 0.118007 |
| x2 | 0.078828 | 1.155400 | yes | 1.000000 | 0.014098 |
| x3 | -0.034371 | 1.008792 | yes | 1.000000 | -0.247290 |
| x4 | -0.137214 | 0.967832 | yes | 1.000000 | 0.313914 |
<!-- /worked -->

Under the random-row protocol the memoriser labels $74.65\%$ of the scored rows correctly, and under the pump protocol $64.10\%$. The pump protocol is the honest number, and the $10.55$ points between them are the leak: with rows of the same pump on both sides, the nearest training row is usually a sibling with a similar label, and the memoriser is rewarded for recognising a pump. The probe's pass criterion is the stated one, not a number to copy: the random-row accuracy must exceed the pump accuracy by more than $5$ points on the average of the $20$ repetitions. A single repetition can show a smaller gap (the sd over repetitions is about $3$ points), which is why the probe is repeated.

Splitting by row passes every size check and fails only the probe. Numbering the training pumps in the order of the shuffle instead of ascending pump order before dealing the folds changes the fold of every pump, and the later tables differ. Forgetting to sort the three pump sets leaves the set of pumps right but the rows in another order, so sums over rows differ in the last digits. Fitting the standardiser on all rows (or on the test rows too) "for convenience" has a small effect, which is why the mistake survives; it is still a leak (Chapter 5). Computing the folds from the rows and not from the pumps is the same trap: do it once, from the pumps, and carry the fold number with the row.

### Milestone 2: baselines

This milestone builds two baselines for each target, fitted on training rows and scored on validation rows (Chapter 8).

```text
function mean_probability(y: array<int>[N]) -> real                  # the share of ones, written $\bar y$ in the text
function best_feature(X: array<real>[N, d], y: array<real>[N]) -> (j: int, rho: real)
        # the j with the largest abs(correlation(X[:, j], y)) over these rows (rho is that absolute value); on a tie the first
function bin_table(x: array<real>[N], y: array<int>[N]) -> (cuts: array<real>[3], p: array<real>[4])
        # sort x; cut q is (v[q*N//4 - 1] + v[q*N//4]) / 2 for q = 1, 2, 3; a row goes to bin
        # (the number of cuts it exceeds); p[b] = (ones in bin b + 1) / (rows in bin b + 2)
function fit_line(x: array<real>[N], y: array<real>[N]) -> (intercept: real, slope: real)   # Chapter 2
```

The **trivial** baselines use no features: for `fail` the training share $\bar y$ as the predicted probability of every row, for `wear` the training mean. The **simple** baselines use the one feature that correlates most with the target: a four-bin frequency table for `fail` (every bin predicts its own smoothed share, $(n_1 + 1)/(n + 2)$ with $n$ the training rows in the bin and $n_1$ those of them with `fail` $= 1$, so no bin ever says $0$ or $1$), and the least-squares line for `wear`. The feature is chosen on the training rows only.

Milestone 7's numbers mean nothing until a model with no skill and a model a colleague could have written in an afternoon have been scored on the same rows. The simple baselines also show what a single feature is worth, so that later gains can be attributed.

Both baselines are scored on the same validation rows, for classification first and then for regression.

<!-- worked: project_predictor.baselines -->
Classification, 180 training pumps (720 rows) against the 240 validation rows. The constant model predicts the training share of fail = 1 for every row: p = 0.380556. The simple model is a four-bin frequency table on the one feature with the largest absolute correlation with fail over the training rows (x0, load, |correlation| 0.4019). Its cuts are -0.7931, -0.0212, 0.7439; a row goes to bin 0 to 3 by the number of cuts it exceeds. Decisions are 'predict 1 when p >= 1/6' (the cost threshold of milestone 5).

| bin | training rows | prediction p |
|---|---:|---:|
| 0 | 180 | 0.082418 |
| 1 | 180 | 0.324176 |
| 2 | 180 | 0.516484 |
| 3 | 180 | 0.604396 |

| model | log loss | ROC AUC | accuracy at 0.5 | cost per row at p >= 1/6 |
|---|---:|---:|---:|---:|
| constant probability | 0.6982 | 0.5000 | 0.5500 | 0.5500 |
| four-bin table on x0 | 0.5832 | 0.7122 | 0.6917 | 0.4292 |

Cost uses 1 per false alarm and 5 per missed failure. A model that always raises the alarm (the constant model at this threshold) has the cost per row of the share of rows with fail = 0.

Regression, the same rows. The constant model predicts the training mean of wear. The simple model is the least-squares line on the one feature with the largest absolute correlation with wear (x0, load, |correlation| 0.8385).

| model | RMSE | R^2 (around the training mean) |
|---|---:|---:|
| constant 3.033682 | 2.7163 | 0.0000 |
| line on x0: wear = 3.0077 + 1.9162 x0 | 1.3564 | 0.7506 |
<!-- /worked -->

The constant probability, the training share $\bar y = 0.3806$, has a log loss of $0.6982$, a little above $\ln 2 = 0.6931$ because the validation share ($108/240 = 0.45$) differs from the training share, and a ROC area of exactly $0.5$, since all scores tie. The four-bin table already reaches an area of $0.7122$ and a log loss of $0.5832$ from one feature, which are the numbers every later classifier has to beat by enough to matter. The cost column anticipates milestone 5: at the cost threshold $1/6$ the constant model raises every alarm, and costs the share of rows with `fail` $= 0$ per row. The line on one feature has $R^2 = 0.7506$ on the validation rows, because load dominates the law, against $R^2 = 0$ for the constant by definition (the total sum of squares is taken around the training mean, which is the constant's prediction).

Taking the base rate or the mean from the rows being scored hands the baseline the answer it is scored on (Chapter 8), and so does choosing the best feature on all rows, or on the validation rows. A bin table can have a bin with no training rows (impossible here, but not for a smaller $N$); the smoothing $(n_1 + 1)/(n + 2)$ is what makes an empty bin predict $1/2$ and not divide by zero. And reporting a baseline's accuracy without its log loss or its area misleads, since the constant model's accuracy of $0.55$ is a statement about the base rate of the validation rows, not about skill.

### Milestone 3: linear regression two ways

This milestone fits a regression of `wear` on the five measurements and solves it twice.

```text
function normal_equations(Z: array<real>[N, d], y: array<real>[N]) -> w: array<real>[d + 1]
        # X = augment(ones, Z), the design [1, Z]; w = solve(matmul(transpose(X), X), matmul(transpose(X), y), true)
function mse_gradient(w: array<real>[d + 1], X: array<real>[N, d + 1], y: array<real>[N]) -> (real, array<real>[d + 1])
        # loss = mean of (w . x_n - y_n)^2, gradient = (2 / N) X^T (X w - y)
function gradient_descent_linear(Z, y, eta: real, steps: int) -> w: array<real>[d + 1]
        # from w = 0; steps times: w = w - eta * gradient; no stopping rule
function to_raw_units(w, mean, sd) -> (intercept: real, slope: array<real>[d])
        # slope_j = w_j / sd_j, intercept = w_0 - sum_j w_j mean_j / sd_j
```

Fit on the standardised training rows, $\eta = 0.3$, $100$ steps from zero. The loss is the mean squared error $\frac{1}{N}\sum_n (\mathbf{w}\cdot\tilde{\mathbf{x}}_n - y_n)^2$ of Chapter 2, so its Hessian is $\frac{2}{N}\mathbf{X}^\top\mathbf{X}$. On standardised features that matrix has eigenvalues of order $2$, which is why a step of $0.3$ is stable (the limit is $2/\lambda_{\max}$); the table below shows that $100$ steps reach the solution. Check your gradient against a numerical one before trusting either solver.

Two routes to one answer are the cheapest correctness test there is. When milestone 4 has no closed form, the gradient check and the descent loop are what you will trust, and they are exercised here first, where the answer can be verified.

The two solvers must agree, and the gradient must pass its numerical check.

<!-- worked: project_predictor.regression_fit -->
Training rows standardised as in the previous table; design matrix [1, z] of shape [720, 6]; loss mean((w.[1, z] - wear)^2). Normal equations (X^T X) w = X^T wear solved by Gaussian elimination with partial pivoting; gradient descent from w = 0 with eta = 0.3 and 100 steps.

| weight | normal equations | gradient descent |
|---|---:|---:|
| intercept | 3.0336823357 | 3.0336823357 |
| z0 | 2.0475355378 | 2.0475355378 |
| z1 | -0.9819908458 | -0.9819908458 |
| z2 | 0.5365956784 | 0.5365956784 |
| z3 | 0.0172370170 | 0.0172370170 |
| z4 | 0.0466615867 | 0.0466615867 |

The two solutions agree to 1e-8 in every weight: True. The gradient norm at the gradient-descent solution is below 1e-8: True. Loss at both: 0.3571024324.

Gradient check at w = (0.5, -0.25, 0.1, 0.3, -0.2, 0.05) on the first 200 training rows, central differences with h = 1e-5, close with rtol = 1e-6 and atol = 1e-8:

| component | analytic | numerical |
|---|---:|---:|
| intercept | -4.96512981 | -4.96512981 |
| z0 | -4.34064448 | -4.34064448 |
| z1 | 1.16074845 | 1.16074845 |
| z2 | -0.46046668 | -0.46046668 |
| z3 | 0.91555346 | 0.91555346 |
| z4 | -1.52885356 | -1.52885356 |

Check passes: True. The same check against a gradient whose z0 component is replaced by 0 passes: False.

Mapped back to the raw features (slope w_j / sd_j, intercept w_0 - sum w_j mean_j / sd_j), beside the coefficients of the law that generated wear:

| coefficient | fitted | law |
|---|---:|---:|
| intercept | 2.9946 | 3.0000 |
| x0 (load) | +1.9663 | +2.0000 |
| x1 (temperature) | -1.0120 | -1.0000 |
| x2 (vibration) | +0.4644 | +0.5000 |
| x3 (age) | +0.0171 | +0.0000 |
| x4 (humidity) | +0.0482 | +0.0000 |

On the 240 validation rows: RMSE 0.6343, R^2 0.9455 (around the training mean 3.0337). The noise floor of the law for a new pump is sqrt(0.4^2 + 0.5^2) = 0.6403.
<!-- /worked -->

The two solutions agree to $10^{-8}$ in every weight (they agree in all ten printed digits). The gradient check passes, and fails when one component of the gradient is replaced by zero, so it can tell a right gradient from a wrong one. In raw units the fitted coefficients $1.9663$ for load and $-1.0120$ for temperature are close to the law's $2$ and $-1$, and the two measurements with no effect receive slopes of $0.0171$ and $0.0482$: not exactly zero, because $720$ rows from $180$ pumps estimate each slope only to a few hundredths. The validation RMSE is $0.6343$, almost exactly the noise floor $\sqrt{0.4^2 + 0.5^2} = 0.6403$ of the law (the pump effect $v$ plus the noise of the row): the model has found everything there is to find, and no model can do much better on a new pump. The simple baseline's RMSE of $1.3564$ is more than twice that, and the constant's $2.7163$ more than four times.

Solving the normal equations on unstandardised features and comparing with descent on standardised ones gives weights that differ by the transformation, not by an error; compare in the same coordinates, then map back. Descent that stops because the loss "stopped moving" at six digits has not run the stated steps. A step size copied from another problem is dangerous when $2/\lambda_{\max}$ of the Hessian has not been computed first (Chapter 2). And dividing the gradient by $N$ in one solver and not in the other makes the two disagree for no deep reason.

### Milestone 4: a regularized logistic classifier

This milestone builds the classifier of Chapters 3 and 4 and the selection of its penalty by cross-validation.

```text
function logistic_loss(w: array<real>[d + 1], Z: array<real>[N, d], y: array<int>[N], lam: real) -> real
        # mean of (softplus(e_n) - y_n e_n) with e_n = w[0] + w[1:] . z_n,  plus (lam / 2) * sum of w[1:]^2
function logistic_gradient(w, Z, y, lam) -> array<real>[d + 1]
        # component 0: mean(sigmoid(e_n) - y_n);  component j: mean((sigmoid(e_n) - y_n) z_nj) + lam * w[j]
function fit_logistic(Z, y, lam, eta: real, steps: int) -> w        # from w = 0, steps exactly
function cv_log_loss(rows: array<int>, fold: array<int>[len(rows)], lams: array<real>) -> array<real>[len(lams), 5]
        # for each lam and fold j: standardise on the other folds, fit on them, score the log loss on fold j
function choose(table: array<real>[len(values), 5], values: array<real>) -> real
        # the value (a penalty here, a neighbour count in milestone 6) whose row has the smallest mean over the 5 folds; the larger value on a tie
function largest_eigenvalue(A: array<real>[d, d], iterations: int) -> real        # power iteration for the step-size check below
        # v = all ones; repeat `iterations` times: av = matmul(A, v); lam = sqrt(sum(av^2)); v = av / lam; return the last lam
```

The loss is the mean binary cross-entropy of Chapter 3 computed from the logit (the softplus form of "Making it numerically safe") plus the L2 penalty of Chapter 4 on the slopes. **The intercept is not penalised**: a prior that pulls the intercept to zero would pull the predicted base rate toward one half, which is a statement about nothing. Chapter 4's separable-data example penalised it for simplicity and said so; here it is left free. The gradient has a component for every weight and one extra term $\lambda w_j$ for each slope, and the descent is Chapter 3's loop with $\eta = 2$ and exactly $100$ steps from $\mathbf{w} = \mathbf{0}$.

The penalty grid is $\lambda \in \{0, 0.003, 0.01, 0.03, 0.1\}$. Inside the cross-validation, **every step that uses data is inside the loop**: the standardiser is fitted on the four training folds of each round and applied to the held-out fold. A step $\eta$ that is stable for one penalty can diverge for a larger one, so the step is checked against the curvature first: the penalised loss has curvature at most $\lambda_{\mathrm{cap}} + \lambda$ with $\lambda_{\mathrm{cap}} = \lambda_{\max}/4$ (the bound of Chapter 3, with $\lambda_{\max}$ the largest eigenvalue of $\frac{1}{N}\mathbf{X}^\top\mathbf{X}$ for the design $[\mathbf{1}, \mathbf{Z}]$), and descent is guaranteed stable below $2/(\lambda_{\mathrm{cap}} + \lambda)$.

<!-- worked: project_predictor.logistic_step_size -->
The matrix of the design [1, z] over the 720 training rows, A = X^T X / N (6 by 6); largest eigenvalue by power iteration (300 iterations from the all-ones vector; each iteration forms A v, takes its Euclidean norm as the eigenvalue estimate and divides A v by it to get the next v; the estimate of the last iteration is used): 1.099815. The curvature bound of the unpenalised loss is lambda_cap = lambda_max / 4 = 0.274954; the penalty adds lambda to it. Gradient descent with step eta is stable when eta < 2 / (lambda_cap + lambda); the step 1 / (lambda_cap + lambda) is the classical choice. The first five penalties are the grid of the cross-validation; 0.3 and 1 are added to show where the guarantee ends.

| lambda | in the grid | 1 / (lambda_cap + lambda) | stability limit 2 / (lambda_cap + lambda) | eta = 2 stable? |
|---|---|---:|---:|---|
| 0 | yes | 3.6370 | 7.2740 | yes |
| 0.003 | yes | 3.5977 | 7.1954 | yes |
| 0.01 | yes | 3.5093 | 7.0187 | yes |
| 0.03 | yes | 3.2792 | 6.5584 | yes |
| 0.1 | yes | 2.6670 | 5.3340 | yes |
| 0.3 | no | 1.7393 | 3.4785 | yes |
| 1 | no | 0.7843 | 1.5687 | no (outside the guarantee) |
<!-- /worked -->

The step $\eta = 2$ lies inside the guarantee for every penalty of the grid, and for $0.3$, which is not in the grid; it lies outside at $\lambda = 1$, which is why the grid stops at $0.1$ and not further up.

This is the model that the rest of the project judges. It is also the first model whose gradient you cannot verify by comparing with a closed form; the numerical check is the only protection against a derivative that is wrong in a way the loss never reveals (a missing penalty term still gives a descent that converges, to the wrong place).

Check the gradient first, at a fixed point on the first $200$ training rows, and then the cross-validation table and the refit.

<!-- worked: project_predictor.logistic_gradient_check -->
First 200 training rows, standardised with the training statistics; w = (-0.3, 0.2, -0.1, 0.05, 0.0, 0.15); lambda = 0.1; loss = mean(softplus(e) - y e) + (lambda / 2) sum of the squared slopes; central differences with h = 1e-5, rtol = 1e-6, atol = 1e-8.

| component | analytic | numerical |
|---|---:|---:|
| intercept | 0.13819838 | 0.13819838 |
| z0 | -0.05205619 | -0.05205619 |
| z1 | 0.06662198 | 0.06662198 |
| z2 | -0.08054588 | -0.08054588 |
| z3 | -0.01466625 | -0.01466625 |
| z4 | 0.02949772 | 0.02949772 |

Check passes: True. With the penalty term left out of the gradient (lambda = 0 in the analytic gradient only) the check passes: False.
<!-- /worked -->

<!-- worked: project_predictor.logistic_cv -->
Five folds of whole pumps (the fold assignment printed with the split). For each fold: the standardiser and the model are fitted on the other four folds (gradient descent from w = 0, eta = 2, 100 steps) and the log loss of the held-out fold is recorded. The chosen lambda has the smallest mean (the larger lambda on a tie); sd uses the divisor 4.

| lambda | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean | sd |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5349 | 0.4426 | 0.5745 | 0.5226 | 0.4530 | 0.5055 | 0.0562 |
| 0.003 | 0.5302 | 0.4442 | 0.5736 | 0.5214 | 0.4551 | 0.5049 | 0.0543 |
| 0.01 (chosen) | 0.5225 | 0.4480 | 0.5724 | 0.5200 | 0.4598 | 0.5046 | 0.0509 |
| 0.03 | 0.5132 | 0.4586 | 0.5726 | 0.5220 | 0.4722 | 0.5077 | 0.0451 |
| 0.1 | 0.5172 | 0.4879 | 0.5822 | 0.5424 | 0.5045 | 0.5268 | 0.0368 |

Refit on all 720 training rows with lambda = 0.01: weights (intercept, then z0 .. z4) -0.653920, 1.091077, -0.806235, 0.484311, 0.079537, 0.059636. Gradient norm below 1e-6: True. Training log loss 0.4899; validation log loss 0.5839; validation ROC AUC 0.7613.
<!-- /worked -->

The check passes for the right gradient and fails when the penalty term is left out, so it has the power to catch the most likely mistake. The cross-validation table is a lesson in how little the penalty matters here: the mean log losses of the first four penalties are within $0.004$ of each other ($0.5055$, $0.5049$, $0.5046$, $0.5077$), while the fold-to-fold spread of one penalty is $0.05$. The grid has found a flat valley. The choice of $\lambda = 0.01$ is the smallest mean, not a measurement that $0.003$ or $0.03$ is worse. With $720$ training rows and five features the data decide almost everything and the prior little; a smaller training set would make the table steeper (Chapter 4). Larger $\lambda$ does hurt: $0.1$ costs $0.022$.

Penalising the intercept changes the fit a little and the validation log loss with it, but nothing is flagged. Standardising once before the folds are drawn lets the folds share statistics with the rows they score (Chapter 5). A step size that diverges for the largest penalty shows as a loss that rises or turns to NaN, and the table above says where the limit is. Choosing the penalty on the log loss of the *training* rows prefers $\lambda = 0$ always. Checking the gradient at $\mathbf{w} = \mathbf{0}$, where several terms vanish, lets a wrong gradient pass; check at a generic point.

### Milestone 5: evaluation

This milestone builds the metrics of Chapter 6, as functions of scores and labels.

```text
function confusion(pred: array<int>[N], y: array<int>[N]) -> (tp: int, fp: int, fn: int, tn: int)
function precision_recall_f1(tp: int, fp: int, fn: int) -> (optional real, optional real, optional real)
        # undefined (not zero) where a denominator is zero
function auc_rank(score: array<real>[N], y: array<int>[N]) -> real
        # (sum of the positives' ranks - P (P + 1) / 2) / (P (N - P)), P the number of positives;
        # ranks ascending from 1, tied scores share their average rank
function roc_points(score, y) -> array<(real, real)>[n_points]  # (FPR, TPR) from the top, tied scores together
function auc_trapezoid(score, y) -> real
function reliability_bins(p: array<real>[N], y, n_bins: int) -> array<real>[n_bins, 3]
        # row b = (count, mean p, observed share) of the rows with b = min(floor(p * n_bins), n_bins - 1), so p = 1 is in the last bin
function ece(p, y, n_bins) -> real                       # sum over bins of (count / N) |observed - mean p|
function fit_scale(e: array<real>[N], y) -> a: real       # Chapter 6; slope(a) = mean((sigmoid(a * e) - y) * e)
        # require slope(0) < 0; lo = 0, hi = 1; up to 60 times: stop if slope(hi) > 0, else lo = hi and hi = 2 * hi (error if it never stops);
        # then 80 times: mid = (lo + hi) / 2; if slope(mid) < 0 then lo = mid else hi = mid; return (lo + hi) / 2
function cost_threshold(c_fp: real, c_fn: real) -> real   # c_fp / (c_fp + c_fn)
```

The decision threshold comes from the costs, not from a search: with a false alarm costing $c_{\mathrm{FP}} = 1$ and a missed failure $c_{\mathrm{FN}} = 5$, a calibrated probability $p$ should raise the alarm when $(1 - p)\,c_{\mathrm{FP}} < p\,c_{\mathrm{FN}}$, that is, when $p \ge t^\star = c_{\mathrm{FP}}/(c_{\mathrm{FP}} + c_{\mathrm{FN}}) = 1/6$ (Chapter 6). The cost of a decision rule is $(c_{\mathrm{FP}}\,\mathrm{FP} + c_{\mathrm{FN}}\,\mathrm{FN})/N$ per row. Temperature scaling fits one number $a$ on the validation rows and replaces $\sigma(e)$ by $\sigma(a\,e)$, with $e$ the logit.

The report quotes these numbers with intervals; a metric that is wrong by a tie rule or by a bin edge is wrong in every row of every table after it.

The functions are tested first on twelve scored items, small enough to verify by hand, and only then on the real validation rows. The first fixture has two tied pairs of scores (at $0.8$ and at $0.5$), each holding one positive and one negative, which is where the three ways of computing the area disagree if one of them mishandles ties.

<!-- worked: project_predictor.metric_fixture -->
Twelve scored items (score, label): (0.9, 1), (0.8, 1), (0.8, 0), (0.7, 1), (0.55, 0), (0.5, 1), (0.5, 0), (0.4, 0), (0.3, 1), (0.2, 0), (0.15, 0), (0.05, 0).

| quantity | value |
|---|---:|
| ROC AUC, rank statistic | 0.7714285714 |
| ROC AUC, trapezoid on the ROC points | 0.7714285714 |
| ROC AUC, pairs counted (a tie counts 1/2): sum / (n+ n-) | 27 / 35 = 0.7714285714 |

ROC points (FPR, TPR), tied scores moving together: (0.0000, 0.0000), (0.0000, 0.2000), (0.1429, 0.4000), (0.1429, 0.6000), (0.2857, 0.6000), (0.4286, 0.8000), (0.5714, 0.8000), (0.5714, 1.0000), (0.7143, 1.0000), (0.8571, 1.0000), (1.0000, 1.0000).

Decisions: predict 1 when score >= t. Cost uses 1 per false alarm and 5 per missed failure; t = 1/6 is the cost threshold c_FP / (c_FP + c_FN).

| t | TP | FP | FN | TN | accuracy | precision | recall | F1 | cost per item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.500000 | 4 | 3 | 1 | 4 | 0.6667 | 0.5714 | 0.8000 | 0.6667 | 0.6667 |
| 0.300000 | 5 | 4 | 0 | 3 | 0.6667 | 0.5556 | 1.0000 | 0.7143 | 0.3333 |
| 0.166667 | 5 | 5 | 0 | 2 | 0.5833 | 0.5000 | 1.0000 | 0.6667 | 0.4167 |
<!-- /worked -->

The area is $27/35 = 0.7714$ by pairs (a tie counts one half), by the ranks and by the trapezoid on the ROC points; the three computations must agree to $10^{-12}$. The second fixture tests the calibration functions on twelve logits.

<!-- worked: project_predictor.calibration_fixture -->
Twelve logits and labels: (-2, 0), (-1.2, 0), (-0.4, 1), (0.3, 0), (0.9, 1), (1.6, 1), (2.4, 1), (-0.8, 0), (1.1, 0), (3.1, 1), (-3, 0), (0.1, 1).

Fitted scale a (bisection as in the metrics chapter): 1.2423296616; temperature 1/a = 0.8049393256.

| probabilities | log loss | ECE, 10 bins | ECE, 5 bins | ROC AUC |
|---|---:|---:|---:|---:|
| raw, a = 1 | 0.4387463095 | 0.1801425058 | 0.1635722217 | 0.8611111111 |
| scaled, a = 1.2423 | 0.4332643076 | 0.1699410616 | 0.1249161292 | 0.8611111111 |

Reliability bins of the raw probabilities (10 equal-width bins; empty bins omitted):

| bin of p | items | mean p | observed share |
|---|---:|---:|---:|
| 0.0 to 0.1 | 1 | 0.047426 | 0.000000 |
| 0.1 to 0.2 | 1 | 0.119203 | 0.000000 |
| 0.2 to 0.3 | 1 | 0.231475 | 0.000000 |
| 0.3 to 0.4 | 1 | 0.310026 | 0.000000 |
| 0.4 to 0.5 | 1 | 0.401312 | 1.000000 |
| 0.5 to 0.6 | 2 | 0.549711 | 0.500000 |
| 0.7 to 0.8 | 2 | 0.730605 | 0.500000 |
| 0.8 to 0.9 | 1 | 0.832018 | 1.000000 |
| 0.9 to 1.0 | 2 | 0.936860 | 1.000000 |
<!-- /worked -->

Then the real validation rows, with the model of milestone 4:

<!-- worked: project_predictor.validation_report -->
Validation rows (240, 60 pumps), logistic model with lambda = 0.01. Temperature scaling: the single scale a is fitted on these rows by bisection, a = 0.752044 (temperature 1.329709).

| probabilities | log loss | ECE, 10 bins | ROC AUC, rank | ROC AUC, trapezoid |
|---|---:|---:|---:|---:|
| raw | 0.5839 | 0.0638 | 0.7613 | 0.7613 |
| scaled | 0.5757 | 0.0629 | 0.7613 | 0.7613 |

Reliability of the raw probabilities on the validation rows (10 equal-width bins, empty bins omitted):

| bin of p | rows | mean p | observed share |
|---|---:|---:|---:|
| 0.0 to 0.1 | 29 | 0.0588 | 0.0690 |
| 0.1 to 0.2 | 25 | 0.1527 | 0.1600 |
| 0.2 to 0.3 | 35 | 0.2491 | 0.2857 |
| 0.3 to 0.4 | 20 | 0.3530 | 0.4500 |
| 0.4 to 0.5 | 20 | 0.4474 | 0.5500 |
| 0.5 to 0.6 | 34 | 0.5420 | 0.5882 |
| 0.6 to 0.7 | 23 | 0.6477 | 0.5652 |
| 0.7 to 0.8 | 21 | 0.7511 | 0.7619 |
| 0.8 to 0.9 | 22 | 0.8412 | 0.7273 |
| 0.9 to 1.0 | 11 | 0.9413 | 0.6364 |

Decisions at the cost threshold t = 0.166667 on the scaled probabilities, and at 0.5 for comparison:

| t | TP | FP | FN | TN | accuracy | precision | recall | F1 | cost per row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 72 | 39 | 36 | 93 | 0.6875 | 0.6486 | 0.6667 | 0.6575 | 0.9125 |
| 0.166667 | 106 | 103 | 2 | 29 | 0.5625 | 0.5072 | 0.9815 | 0.6688 | 0.4708 |
<!-- /worked -->

Read this table with the chapter's own warnings in mind. The area ($0.7613$) is identical for the raw and the scaled probabilities, because a positive scale changes no ranking; the trapezoid and the rank statistic agree to the digits printed. The scale $a = 0.7520$ (temperature $1.3297$) says the raw probabilities are overconfident, and the log loss falls from $0.5839$ to $0.5757$. The ECE barely moves ($0.0638$ to $0.0629$) because, with $240$ rows in ten bins, it is dominated by the noise of bins with a few dozen rows (Chapter 6). The reliability rows show where the raw model is wrong: at the top, the $11$ rows with $p \ge 0.9$ fail only $63.64\%$ of the time. And the last table is the cost threshold at work: at $t = 0.5$ the decisions cost $0.9125$ per row; at the derived $t = 1/6$ they cost $0.4708$, the accuracy falls from $0.6875$ to $0.5625$, and the recall rises from $0.6667$ to $0.9815$. Accuracy and cost disagree, because accuracy does not know that a missed failure costs five false alarms.

Giving a tie the rank of its first element, or the score $0$ or $1$ in the pair count, changes the area. Putting $p = 1$ in a bin of its own (out of range) instead of the last bin loses rows. Fitting the scale on the test rows, or searching the threshold on them, breaks the rule that the scale and the cost threshold use validation rows only. A precision of $0/0$ is undefined, and reading it as $0$ is a mistake. And a positive-class convention that differs between functions (alarm = 1 in one, alarm = 0 in another) corrupts every table downstream.

### Milestone 6: one classical comparison model

This milestone builds $k$ nearest neighbours (Chapter 7) as a probability model, to be compared with the logistic classifier on the same footing.

```text
function knn_probabilities(Ztr: array<real>[N, d], ytr: array<int>[N], Zq: array<real>[Q, d],
                           ks: array<int>) -> array<real>[len(ks), Q]
        # for each query: order the training rows by Euclidean distance (equal distances keep row order);
        # for each k: p = (ones among the k nearest + 1) / (k + 2)
function cv_knn(rows, fold, ks) -> array<real>[len(ks), 5]       # the same folds as milestone 4, the standardiser refitted per round; `choose(table, ks)` as in milestone 4 (the larger k on a tie)
```

The candidate neighbour counts are $k \in \{1, 5, 15, 31, 61, 121\}$ (this $k$ is the neighbour count of Chapter 7, not the number of folds of Chapter 5, which is $5$ throughout), chosen by the mean log loss over the same five folds of pumps. The probability is the share of the $k$ nearest training labels, smoothed by one pseudo-count for each class so that no probability is $0$ or $1$ (Chapter 7's Laplace smoothing; a log loss of $\infty$ is not a score). Sort the distances once per query and read every $k$ from the same order.

A comparison is only as convincing as the effort behind the thing compared with (Chapter 8): the neighbour count must be chosen as carefully as the penalty, on the same folds, with the same metric.

The neighbour count is chosen on the same folds as the penalty, and both models are then scored on the validation rows.

<!-- worked: project_predictor.knn_comparison -->
The same five grouped folds. For each fold the standardiser is fitted on the other four, each held-out row gets the probability (n1 + 1) / (k + 2) from its k nearest rows (Euclidean distance, equal distances in row order), and the log loss of the fold is recorded.

| k | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean | sd |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6846 | 0.6750 | 0.7376 | 0.6124 | 0.6558 | 0.6731 | 0.0455 |
| 5 | 0.7036 | 0.5935 | 0.6946 | 0.5335 | 0.6554 | 0.6361 | 0.0720 |
| 15 | 0.5947 | 0.5398 | 0.6325 | 0.5047 | 0.5871 | 0.5718 | 0.0499 |
| 31 | 0.5602 | 0.5108 | 0.6142 | 0.5251 | 0.5300 | 0.5481 | 0.0411 |
| 61 (chosen) | 0.5421 | 0.4971 | 0.5893 | 0.5174 | 0.5176 | 0.5327 | 0.0354 |
| 121 | 0.5458 | 0.5149 | 0.5930 | 0.5388 | 0.5259 | 0.5437 | 0.0300 |

On the 240 validation rows, models fitted on all 720 training rows:

| model | log loss | ROC AUC | ECE, 10 bins |
|---|---:|---:|---:|
| logistic, lambda = 0.01 | 0.5839 | 0.7613 | 0.0638 |
| k nearest neighbours, k = 61 | 0.5615 | 0.7761 | 0.0547 |
<!-- /worked -->

The fold log loss falls from $0.6731$ at $k = 1$ to $0.5327$ at $k = 61$ and rises again at $121$: Chapter 7's hump. The model with $k = 61$ is slightly better than the logistic classifier on the validation rows ($0.5615$ against $0.5839$ in log loss, $0.7761$ against $0.7613$ in area), which is a statement about those $240$ rows and not yet about the models; the report will say what the comparison is worth.

Letting a training row be its own neighbour when scoring it flatters the model, so score held-out rows only. A tie order that depends on the sorting routine breaks reproducibility; use a stable sort, or break ties by row number. Fitting the standardiser once is a particular danger here, since nearest neighbours are the most scale-sensitive model of the project. And choosing $k$ on accuracy and the penalty on log loss compares two models chosen by different rules, which is not a fair comparison.

### Milestone 7: the report

The last milestone builds four things, in this order, using only choices already made.

1. **The test table.** Read the $240$ test rows once, for the models of milestones 2 to 6, and report log loss, area, ECE, $F_1$, recall and cost per row at the derived threshold. Report the regression on the test rows too.
2. **Intervals.** A cluster bootstrap over the $60$ test pumps: each resample draws $60$ pump numbers with `rng.integer(60)` in order, takes all four rows of each drawn pump, and recomputes every statistic. The same resamples serve every quantity, so differences are paired. This is the bootstrap of Volume I with the **pump as the unit**, since the four rows of a pump are not independent draws; `cluster_replicates` returns every replicate of every quantity, because the report needs the standard deviation of the replicates and the differences between columns, and Volume I's `bootstrap_interval` is its special case with one quantity.
3. **A multi-seed study with an ablation.** Repeat the whole pipeline (`run_pipeline`: split, folds, choice of $\lambda$ and $k$, refit, test) for $16$ runs whose seeds come from one master generator, with a paired comparison of the models over the runs. As the ablation, redo the choice of $\lambda$ and $k$ with random row folds in place of folds of pumps and see what changes.
4. **Where it fails.** Score the shift slice, and say, with numbers, what changed.

```text
function cluster_replicates(n_units: int, statistics: function(array<int>[n_units]) -> array<real>[S],
                            n_resamples: int, g: inout Generator) -> array<real>[n_resamples, S]
        # replicate r = 0 .. n_resamples - 1 is statistics(picks) with picks = [g.integer(n_units) for i in 0 .. n_units]
        # (n_units draws, in order, repeats allowed); the replicates are drawn one after the other from the one generator g.
        # All S numbers of a row come from the same picks, so a difference of two columns is a paired difference.
        # Here a unit is a test pump (numbered 0 .. 59 in ascending pump order); `statistics` takes the four rows of every
        # picked pump, pumps in the order of the picks, and returns every score of every model that the report needs
function percentile_interval(column: array<real>[n_resamples], level: real) -> (real, real)
        # (percentile(column, (1 - level) / 2), percentile(column, 1 - (1 - level) / 2)); the sd of a column is sd(column)
record Choice:                         # what one way of drawing the folds chose, and how the choice scored on the test rows
    lam: real                          # penalty, by choose(): smallest mean fold log loss, the larger on a tie
    k: int                             # neighbour count, by the same rule
    cv_logistic: real                  # mean fold log loss of the chosen lam
    cv_knn: real                       # mean fold log loss of the chosen k
    test_log_loss_logistic: real       # models refitted on the 720 training rows, scored on the 240 test rows
    test_log_loss_knn: real
    test_auc_logistic: real
    test_auc_knn: real
record PipelineRun:
    seed: u64
    grouped: Choice                    # choices made with the folds of pumps (the pipeline's own)
    row: Choice                        # the same choices made with random row folds (the ablation only)
    scale: real                        # a of fit_scale on the 240 validation logits of the grouped-fold logistic model
    test_log_loss_scaled: real         # log loss on the test rows of sigmoid(a * e), e the logit of the grouped-fold model
    test_log_loss_constant: real       # the training share of ones for every row
    site_b_log_loss_logistic: real     # the grouped-fold logistic model on the 240 site B rows
    site_b_auc_logistic: real
    site_b_log_loss_constant: real
function run_pipeline(seed: u64) -> PipelineRun
        # rng = Generator(seed); then, in this order, all from rng: split_pumps(300, 180, 60, rng) (299 draws);
        # fold_assignment(180, 5, rng) over the training pumps numbered 0 .. 179 in ascending pump order (179 draws);
        # fold_assignment(720, 5, rng) over the training rows numbered 0 .. 719 in ascending row number (719 draws,
        # made in every run); then choose, refit and score as in milestones 4 to 6, once per kind of fold
```

This is what the whole volume was for. A number is not a result until the reader has the interval, the number of runs, the comparison it was measured against and the place where it stops being true.

The check is the test rows, read once. Every choice below was made before this table was computed: $\lambda = 0.01$, $k = 61$, $a = 0.7520$ and the threshold $1/6$.

<!-- worked: project_predictor.test_report -->
The test rows (240 rows, 60 pumps), read once, with every choice already made on training and validation rows: lambda = 0.01, k = 61, scale a = 0.752044, decision threshold 0.166667 on the scaled probability (and on the other probabilities as they are).

| model | log loss | ROC AUC | ECE, 10 bins | F1 at t | recall at t | cost per row at t |
|---|---:|---:|---:|---:|---:|---:|
| constant probability | 0.6535 | 0.5000 | 0.0222 | 0.5276 | 1.0000 | 0.6417 |
| four-bin table on x0 | 0.5688 | 0.7068 | 0.0378 | 0.6194 | 0.9651 | 0.4750 |
| logistic, raw | 0.4939 | 0.8173 | 0.0571 | 0.6378 | 0.9419 | 0.4667 |
| logistic, temperature scaled | 0.5047 | 0.8173 | 0.0676 | 0.6050 | 0.9884 | 0.4792 |
| k nearest neighbours, k = 61 | 0.5262 | 0.7969 | 0.0631 | 0.5952 | 1.0000 | 0.4875 |

Cluster bootstrap over the 60 test pumps: seed 0x6e7dd69d0d03b2ee (7961755686277460718). One resample is 60 calls of rng.integer(60), in order, each the index (in ascending pump order) of a pump whose four rows are all taken, the pumps in draw order and the rows of a pump ascending; 1000 resamples from one generator, all six quantities of the table and the quantities they are differences of recomputed on the same resample (so the differences are paired). Interval ends are the `percentile` of the replicates at q = 0.025 and q = 0.975 (linear interpolation at position q (number of replicates - 1)); sd is the standard deviation of the replicates with divisor 1000 - 1.

| quantity | estimate | sd of replicates | 95 percent lower | 95 percent upper |
|---|---:|---:|---:|---:|
| logistic, ROC AUC | 0.8173 | 0.0367 | 0.7427 | 0.8841 |
| logistic, log loss | 0.4939 | 0.0435 | 0.4081 | 0.5759 |
| logistic, ECE | 0.0571 | 0.0210 | 0.0512 | 0.1318 |
| log loss, logistic minus constant | -0.1596 | 0.0361 | -0.2328 | -0.0930 |
| log loss, logistic minus k-NN | -0.0323 | 0.0161 | -0.0644 | -0.0008 |
| ROC AUC, logistic minus k-NN | 0.0204 | 0.0147 | -0.0063 | 0.0513 |
<!-- /worked -->

On $240$ test rows from $60$ pumps the logistic classifier reaches a ROC area of $0.8173$ and a log loss of $0.4939$, against $0.5000$ and $0.6535$ for the constant model and $0.7068$ and $0.5688$ for the four-bin table on one feature. The bootstrap says how far to trust the first two numbers: $0.7427$ to $0.8841$ for the area and $0.4081$ to $0.5759$ for the log loss. The paired differences are the sharper statement. Against the constant model the log loss is lower by $0.1596$ (interval $-0.2328$ to $-0.0930$), clear of zero. Against $k$ nearest neighbours it is lower by $0.0323$, with an interval, $-0.0644$ to $-0.0008$, that touches zero; and the difference in area, $+0.0204$ (interval $-0.0063$ to $+0.0513$), contains it. On this test set the two models are close, and this table cannot rank them reliably. Two further things the table says. The temperature-scaled probabilities are worse on the test rows ($0.5047$ against $0.4939$ in log loss): a scale fitted on $240$ validation rows from $60$ pumps is itself an estimate, and here it moved the wrong way. And the cost per row at $t = 1/6$ is $0.4667$ for the raw logistic probabilities, against $0.6417$ for raising every alarm and $0.4750$ for the one-feature table, so the decision rule gains little over the simple baseline here.

<!-- worked: project_predictor.regression_test -->
The 240 test rows, read once, with the weights of milestone 3 (fitted on the training rows only). Interval: the cluster bootstrap over the 60 test pumps as in the classification report (seed 0xa65146519b753764 (11984437399724046180), 1000 resamples, 60 calls of rng.integer(60) each, taken as in the classification report; percentile interval of the RMSE at q = 0.025 and 0.975).

| model | test RMSE | R^2 (around the training mean) | RMSE 95 percent lower | RMSE 95 percent upper |
|---|---:|---:|---:|---:|
| constant (training mean) | 2.2458 | 0.0000 | - | - |
| line on x0 | 1.4277 | 0.5959 | 1.1980 | 1.6500 |
| linear regression on x0 .. x4 | 0.6294 | 0.9215 | 0.5673 | 0.6915 |

RMSE of the linear regression minus RMSE of the line, same resamples: -0.7983, 95 percent interval -1.0159 to -0.5841.
<!-- /worked -->

For the continuous target the linear regression has a test RMSE of $0.6294$ (interval $0.5673$ to $0.6915$), at the noise floor, against $1.4277$ for the line on one feature: the difference of $-0.7983$ in RMSE has an interval of $-1.0159$ to $-0.5841$ and is not in doubt.

#### Sixteen runs, to put one test split in its place

One test set is one draw. The study below repeats the whole pipeline $16$ times, each with its own split of the $300$ pumps, so that the spread over runs includes the choice of test pumps, of folds and of the neighbour count.

<!-- worked: project_predictor.seed_study -->
16 runs. Master generator: seed 0x0ffb3472a2ba3e69 (1151571796684324457); run s uses the s-th raw value (rng.next(), counting from 0) as the seed of its own SplitMix64, which draws, in order: the shuffle of the 300 site A pump numbers (299 integer draws; 180 pumps train, 60 validate, 60 test, each set sorted ascending), the shuffle of the 180 training pumps for the grouped folds (179 integer draws; the pumps numbered 0 .. 179 in ascending pump order, the pump at shuffled position p goes to fold p mod 5), and the shuffle of the 720 training rows for the row folds used only by the ablation below (719 integer draws; the rows numbered 0 .. 719 in ascending row number, the row at shuffled position p goes to fold p mod 5). The 719 draws are made in every run, whether or not the ablation is read. In each run the penalty lambda and the neighbour count k are chosen by grouped five-fold cross-validation on the mean fold log loss (the smallest mean; the larger value on a tie, for lambda and for k alike), the models are refitted on the 720 training rows and scored on the 240 test rows and on the 240 rows of site B. The scale a of 'scaled' is fitted by the doubling-and-bisection routine of milestone 5 on the 240 validation rows of that run, and the 'constant' model is the share of ones among that run's 720 training rows.

| run | seed | lambda | k | test log loss, logistic | k-NN | constant | test AUC, logistic | k-NN | site B log loss | site B AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0x36c4c4ac98220070 | 0.01 | 61 | 0.5181 | 0.5406 | 0.6799 | 0.8118 | 0.8173 | 0.8783 | 0.5971 |
| 1 | 0xe63eccf223012c78 | 0.01 | 61 | 0.5084 | 0.5394 | 0.6827 | 0.8263 | 0.8116 | 0.9973 | 0.5893 |
| 2 | 0x8579437648e13e00 | 0.01 | 61 | 0.5210 | 0.5510 | 0.6572 | 0.7976 | 0.7798 | 0.9114 | 0.5893 |
| 3 | 0xe45c2dcb082fadff | 0.01 | 31 | 0.5326 | 0.5704 | 0.6684 | 0.7926 | 0.7473 | 0.9295 | 0.5923 |
| 4 | 0x09feb7be332b4744 | 0.01 | 31 | 0.4471 | 0.5060 | 0.6874 | 0.8746 | 0.8289 | 0.9512 | 0.5898 |
| 5 | 0x2b057a0302f40a72 | 0.01 | 61 | 0.5449 | 0.5478 | 0.6654 | 0.7953 | 0.7913 | 0.9176 | 0.5911 |
| 6 | 0x31b464d13eb96075 | 0.01 | 61 | 0.4751 | 0.5078 | 0.6618 | 0.8474 | 0.8471 | 0.9419 | 0.5805 |
| 7 | 0x1cf9e8d19d406d15 | 0.01 | 61 | 0.5541 | 0.5515 | 0.6974 | 0.8098 | 0.7884 | 0.9366 | 0.5952 |
| 8 | 0xfcfcacb61a996b2d | 0.01 | 31 | 0.5334 | 0.5354 | 0.6731 | 0.7907 | 0.8005 | 0.8635 | 0.6176 |
| 9 | 0x62bbdeef4f4b92ea | 0.01 | 31 | 0.5873 | 0.5769 | 0.6865 | 0.7638 | 0.7530 | 1.0313 | 0.5898 |
| 10 | 0x114c6e5250749323 | 0.01 | 61 | 0.5687 | 0.5632 | 0.6527 | 0.7652 | 0.7560 | 0.9331 | 0.5927 |
| 11 | 0x3164c0a3fbbf9ddf | 0.01 | 31 | 0.5663 | 0.6142 | 0.6695 | 0.7692 | 0.7117 | 0.9573 | 0.5762 |
| 12 | 0x1e139d382cb1c476 | 0.01 | 61 | 0.5109 | 0.5382 | 0.6713 | 0.8231 | 0.8220 | 0.9878 | 0.5788 |
| 13 | 0xe12dcf4b21249c99 | 0.01 | 61 | 0.4670 | 0.5136 | 0.6582 | 0.8509 | 0.8354 | 0.8790 | 0.5837 |
| 14 | 0x407c1f62a9485c1e | 0.03 | 61 | 0.4987 | 0.5202 | 0.6718 | 0.8415 | 0.8209 | 0.7953 | 0.6067 |
| 15 | 0x552a42d5ef74a31d | 0.01 | 31 | 0.5583 | 0.5510 | 0.6554 | 0.7938 | 0.7588 | 0.9424 | 0.5969 |

Over the 16 runs: mean, standard deviation (divisor n - 1) and a bootstrap 95 percent interval of the mean (resample the 16 runs with replacement, 2000 resamples, interval ends at the 2.5th and 97.5th percentiles of the replicated means; each row has its own generator, started at the seed in the last column; a resample is 16 calls of rng.integer(16), each the index of the next run to include).

| quantity | mean | sd | 95 percent lower | 95 percent upper | bootstrap seed |
|---|---:|---:|---:|---:|---|
| test log loss, logistic | 0.5245 | 0.0393 | 0.5053 | 0.5420 | 0x78c0a5fe3f7259f0 |
| test log loss, scaled minus raw logistic | +0.0094 | 0.0160 | 0.0025 | 0.0179 | 0x3ed50322ed61cd2d |
| test log loss, logistic minus constant | -0.1467 | 0.0414 | -0.1678 | -0.1267 | 0xef1431fb42271c5d |
| test log loss, logistic minus k-NN | -0.0210 | 0.0219 | -0.0317 | -0.0111 | 0x5798c410d5146f19 |
| test ROC AUC, logistic | 0.8096 | 0.0324 | 0.7953 | 0.8253 | 0x708d3f124f265b93 |
| test ROC AUC, logistic minus k-NN | +0.0177 | 0.0194 | 0.0089 | 0.0274 | 0x54e67c895886be33 |
| site B log loss, logistic | 0.9284 | 0.0566 | 0.8990 | 0.9528 | 0x8518ec8dc0260bb0 |
| site B log loss, logistic minus constant | +0.2334 | 0.0580 | 0.2047 | 0.2586 | 0x025fdaab5aa89fd5 |
| site B ROC AUC, logistic | 0.5917 | 0.0103 | 0.5870 | 0.5969 | 0xed27029b613d7831 |

Runs in which the logistic model has the smaller test log loss than k-NN: 12 of 16.
<!-- /worked -->

The logistic classifier's mean test log loss over the $16$ runs is $0.5245$ (sd $0.0393$, bootstrap interval of the mean $0.5053$ to $0.5420$). Against the constant model it is lower by $0.1467$ on average over the runs, and the interval of the mean difference ($-0.1678$ to $-0.1267$) is far from zero. Against $k$ nearest neighbours the mean difference is $-0.0210$ with an interval of $-0.0317$ to $-0.0111$; the logistic model has the smaller log loss in $12$ of the $16$ runs, and the area is higher by $0.0177$ ($0.0089$ to $0.0274$). So the weak evidence of the single test split ($-0.0323$, an interval touching zero) becomes, over sixteen splits, a small but consistent advantage; the run-to-run standard deviation of the difference is $0.0219$, as large as the difference itself, which is why one split could not show it. Scaling the probabilities did not help in general either: the mean of (scaled minus raw) is $+0.0094$ with an interval of $0.0025$ to $0.0179$, so the scale fitted on the validation rows makes the log loss slightly worse on average, and the pipeline reports the raw probabilities. These intervals describe the uncertainty from the splits of one dataset of $300$ pumps; they do not include the uncertainty from having drawn these $300$ pumps from the world.

#### An ablation of the fold design

Folds drawn at random over rows, instead of over pumps, put the sibling rows of a pump into the training folds of the round that scores it.

<!-- worked: project_predictor.ablation_cv -->
The 16 runs of the seed study. The design choice: how the folds that select lambda (logistic) and k (k-NN) are drawn (grouped folds: the folds of pumps of the seed study; random row folds: the 720 training rows dealt by the row-fold shuffle of the seed study). The chosen value is the one with the smallest mean fold log loss over the five folds of the drawn kind (the larger value on a tie). 'CV estimate' is that mean fold log loss of the chosen value, the number a report would claim; 'test' is the log loss of the refitted model on the 240 test rows of the same run, from pumps never seen; gap = CV estimate minus test. The interval is a bootstrap 95 percent interval of the mean gap over the runs (2000 resamples as in the seed study, each row with the generator seed of the last column).

| folds | model | mean CV estimate | mean test log loss | mean gap | 95 percent lower | 95 percent upper | bootstrap seed |
|---|---|---:|---:|---:|---:|---:|---|
| grouped folds (by pump) | logistic | 0.5227 | 0.5245 | -0.0018 | -0.0234 | 0.0191 | 0x76b31ef327924d57 |
| grouped folds (by pump) | k-NN | 0.5419 | 0.5455 | -0.0036 | -0.0199 | 0.0128 | 0x6de21e2ac8f84e4d |
| random row folds | logistic | 0.5131 | 0.5242 | -0.0112 | -0.0330 | 0.0086 | 0xda2bd5b028b76813 |
| random row folds | k-NN | 0.5181 | 0.5845 | -0.0665 | -0.0968 | -0.0391 | 0x8c0c387249a59440 |

Test log loss of the logistic model chosen with row folds minus the one chosen with grouped folds, over the runs: mean -0.00027, 95 percent interval -0.00241 to +0.00169 (bootstrap seed 0x9c0573f3fba8b3ba); chosen lambda differs in 13 runs.

Test log loss of the k-NN model chosen with row folds minus the one chosen with grouped folds, over the runs: mean +0.03909, 95 percent interval +0.02340 to +0.05488 (bootstrap seed 0xeb6947b8a24d39f4); chosen k differs in 15 runs.
<!-- /worked -->

![A scatter plot of the test log loss of the k-NN model, vertical, against the cross-validation log loss of the chosen k, horizontal, for 16 runs and two ways of drawing the folds. Points from folds of whole pumps scatter around the dashed diagonal; points from random row folds lie mostly above it (14 of the 16), with test log loss between 0.51 and 0.71 against claims between 0.48 and 0.57.](img/predictor-fold-ablation.svg)

For the nearest-neighbour model the effect is large. With folds of pumps the cross-validation estimate is $0.5419$ and the test loss $0.5455$, a gap of $-0.0036$ (interval $-0.0199$ to $+0.0128$): honest. With random row folds the estimate is $0.5181$ and the test loss $0.5845$, a gap of $-0.0665$ (interval $-0.0968$ to $-0.0391$): the report would have claimed $0.518$ for a model that delivers $0.585$, and the chosen $k$ differs in $15$ of $16$ runs, ending up in a model whose test log loss is $0.0391$ worse (interval $+0.0234$ to $+0.0549$). For the logistic classifier the same change does nothing visible: the gap is $-0.0112$ (interval $-0.0330$ to $+0.0086$) and the test loss of the chosen model differs by $-0.0003$ (interval $-0.0024$ to $+0.0017$). A model with six parameters cannot memorise pumps; a model that stores the training rows can, and it is the memoriser of milestone 1's probe, rewarded there for recognising a pump. That is no argument for splitting by group only when the model is flexible: you do not know in advance which of your models can exploit a leak, so the split is by group for all of them.

#### Where the model fails: the second site

Site B was never touched. Its pumps carry loads $2.0$ higher than site A's. Score the model there:

<!-- worked: project_predictor.shift_slice -->
The model of the report (logistic, no recalibration) on the test rows and on the 240 rows of site B, which no earlier step looked at.

| rows | n | observed share fail = 1 | mean predicted p | log loss, logistic | log loss, constant | log loss, k-NN | ROC AUC, logistic | ROC AUC, k-NN | ECE, logistic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test (site A) | 240 | 0.3583 | 0.3323 | 0.4939 | 0.6535 | 0.5262 | 0.8173 | 0.7969 | 0.0571 |
| shift slice (site B) | 240 | 0.4458 | 0.7181 | 1.0066 | 0.6961 | 0.6855 | 0.5768 | 0.6859 | 0.2934 |

Reliability of the logistic probabilities on the shift slice and on the test rows (10 equal-width bins; empty bins omitted):

| bin of p | test rows | mean p | observed | site B rows | mean p | observed |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 to 0.1 | 44 | 0.0598 | 0.0227 | 1 | 0.0844 | 0.0000 |
| 0.1 to 0.2 | 38 | 0.1497 | 0.1579 | 10 | 0.1555 | 0.1000 |
| 0.2 to 0.3 | 44 | 0.2424 | 0.2727 | 8 | 0.2345 | 0.0000 |
| 0.3 to 0.4 | 35 | 0.3486 | 0.4000 | 10 | 0.3470 | 0.6000 |
| 0.4 to 0.5 | 23 | 0.4416 | 0.4783 | 15 | 0.4471 | 0.3333 |
| 0.5 to 0.6 | 17 | 0.5485 | 0.7647 | 18 | 0.5552 | 0.3889 |
| 0.6 to 0.7 | 14 | 0.6505 | 0.5000 | 25 | 0.6510 | 0.4800 |
| 0.7 to 0.8 | 13 | 0.7434 | 0.8462 | 39 | 0.7621 | 0.4615 |
| 0.8 to 0.9 | 11 | 0.8556 | 0.9091 | 60 | 0.8572 | 0.5667 |
| 0.9 to 1.0 | 1 | 0.9129 | 1.0000 | 54 | 0.9487 | 0.4444 |

Where the slice differs: the mean of each standardised feature (training statistics) over the training rows and over the slice.

| feature | training rows | site B rows |
|---|---:|---:|
| x0 (load) | -0.0000 | 1.7545 |
| x1 (temperature) | 0.0000 | -0.0669 |
| x2 (vibration) | -0.0000 | -0.1324 |
| x3 (age) | -0.0000 | 0.1336 |
| x4 (humidity) | 0.0000 | -0.0162 |

Range of the raw (unstandardised) x0 over the training rows: -2.7576 to 3.2405; site B rows outside it: 17 of 240. Rows with raw x0 > 2: 17 of 720 training rows, 110 of 240 site B rows.
<!-- /worked -->

![Reliability diagram: share of rows with fail = 1 against mean predicted probability in ten bins, for 240 test rows and for 240 rows of site B. The test curve stays near the diagonal, its largest departure in a bin of only 17 rows; the site B curve stays between 0.33 and 0.57 for mean predicted probabilities from 0.45 to 0.95, far below the diagonal at the top.](img/predictor-reliability.svg)

The logistic classifier's log loss on site B is $1.0066$, against $0.4939$ on the test rows and $0.6961$ for the constant model: **worse than no model at all.** Its area falls from $0.8173$ to $0.5768$, close to chance. Its mean predicted probability is $0.7181$ where $44.58\%$ of the rows failed, and its ECE rises from $0.0571$ to $0.2934$. The reliability table shows the mechanism: in the two highest bins, $114$ site B rows, the model says $0.86$ to $0.95$ and the failures are $57\%$ and $44\%$. The failure is extrapolation, not noise. The table of feature means shows where the slice differs from training: load, standardised by the training statistics, is $1.7545$ standard deviations higher, and the other four features are within $0.14$. In the training rows only $17$ of $720$ have a raw load $x_0 > 2$; at site B $110$ of $240$ do. That is the region where the law's curved term $-0.8\,x_0^2$ bends the failure probability back down (the law is printed above only so that the data can be regenerated; a project without it would find the same fact by plotting the residuals against load) and where a linear logit keeps rising. $k$ nearest neighbours average the outcomes of the training rows closest to each query, which for site B are the highest-load training rows, where the curve has already bent. They are less wrong: their site B log loss is $0.6855$ and their area $0.6859$, against $1.0066$ and $0.5768$ for the logistic model, though neither is good. Over the $16$ runs the picture is the same: the mean site B log loss of the logistic model is $0.9284$, which exceeds the constant model's by $0.2334$ (interval $0.2047$ to $0.2586$), and the mean area is $0.5917$.

#### The honest statement

A sentence in the form of Chapter 8's report card, from the numbers above:

> On $240$ test rows from $60$ pumps unseen in training, validation and model selection, the regularized logistic classifier (penalty $0.01$ chosen by five-fold cross-validation over pumps) has a ROC area of $0.817$ (95% cluster-bootstrap interval $0.743$ to $0.884$) and a log loss of $0.494$ ($0.408$ to $0.576$), against $0.654$ for a constant model. Over $16$ repeated splits its mean log loss is $0.5245$, lower than $k$ nearest neighbours by $0.021$ (interval $0.011$ to $0.032$; $12$ of $16$ runs); a single split could not separate the two. Temperature scaling does not help and is not used. At the cost threshold $1/6$ the decisions cost $0.467$ per row, against $0.642$ for raising every alarm. **The model must not be used at site B**: there its log loss is $1.007$, worse than the constant's $0.696$, and its probabilities are overconfident (mean $0.718$ against an observed $0.446$), because the loads of site B lie beyond the range in which a linear logit describes the failure probability. For the continuous target the linear regression has an RMSE of $0.629$ ($0.567$ to $0.692$), at the noise floor of $0.640$ for new pumps. Everything was fitted on $180$ pumps; the intervals do not include the uncertainty from the choice of those pumps from the world.

Bootstrapping rows instead of pumps gives an interval that is too narrow, because it treats $240$ rows from $60$ pumps as $240$ independent ones (an extension measures how much). Using a different resample for each model puts the resampling noise of two draws into the paired difference instead of none. Reading the test set before every choice is made, including the neighbour count, and then changing something, spoils the test. Letting the shift slice decide anything (a threshold, a feature, a penalty) turns it into a validation set, and the failure is no longer a finding. Reporting the best of the sixteen runs, or the run whose seed was "lucky", should give way to the mean and the interval over all of them. And the temperature-scaling gain must not be stated from the validation rows on which the scale was fitted.

### Where brain's own code enforces the same habits

The habits of this chapter are ones brain's own code enforces. `crates/decisions/src/evaluation.rs` evaluates a decision model against the majority and chance baselines and, in its module documentation, takes the majority label from the training split because taking it from the evaluated rows flatters the baseline (the trivial-baseline mistake of milestone 2); commit `b3da5ef07` added it with a negative control, a model trained on shuffled labels that must not beat the baseline. `crates/data/src/episode.rs` splits recorded episodes into training and validation sets by whole episode, never by frame, with a test (`episode_split_by_episode_disjoint_and_covers_all`) that the two parts are disjoint: the group split of milestone 1, with the episode as the pump. `crates/rlcd/src/metrics.rs` holds `reliability_bins` and `ece` with the same rule for a probability of exactly $1$ as milestone 5.

## What your predictor must reproduce

The tests are the blocks printed in the milestones above, collected here as a checklist. An implementation passes when every statement holds. "Exact" means integers or text equal; "to $t$" means the stated absolute tolerance.

| milestone | test | expected | tolerance |
|---|---|---|---|
| data | the first rows, the sums of the features and of `wear`, the pump effects, the counts of `fail` | as printed (6 decimals) | $5 \times 10^{-7}$, half a unit of the last printed place, for the sums and rows; counts exact |
| 1 | sizes of the four sets, pumps shared between sets, rows per fold, first pumps of each set and the folds of the first eight training pumps | as printed; zero pumps shared | exact |
| 1 | standardiser: mean and root mean square of training $z$ | $0$ and $1$ | $10^{-12}$; the printed means of the test $z$ to $10^{-6}$ |
| 1 | leakage probe, 20 repetitions | means as printed; random-row mean exceeds pump mean by more than $0.05$ | $10^{-4}$ for a bit-exact generator; the stated margin otherwise |
| 2 | bin cuts, bin probabilities, both baseline tables | as printed | $10^{-4}$ |
| 3 | weights by the normal equations and by descent; the gradient check; the mapped-back coefficients; validation RMSE | as printed; solutions agree to $10^{-8}$; the check passes for the right gradient and fails for the broken one | $10^{-8}$ for weights; $10^{-4}$ for the rest |
| 4 | step-size table (eigenvalue by power iteration) | as printed | $10^{-6}$ |
| 4 | gradient check with the penalty; failure of the check without it | as printed | $10^{-6}$ relative |
| 4 | fold log losses for each penalty, the chosen penalty, the refit weights | as printed; the same $\lambda$ | $10^{-4}$ for log losses; $10^{-6}$ for weights; $\lambda$ exact |
| 5 | fixture 1: area three ways, ROC points, confusion and cost at three thresholds | as printed; the three areas agree with each other | $10^{-12}$ between the three areas; $10^{-10}$ against the printed values |
| 5 | fixture 2: scale, log loss, ECE with 10 and 5 bins, reliability rows | as printed | $10^{-8}$ |
| 5 | validation report: scale, log loss, ECE, area, reliability, decisions | as printed | $10^{-4}$ |
| 6 | fold log losses for each $k$, the chosen $k$, validation scores | as printed; the same $k$ | $10^{-4}$; $k$ exact |
| 7 | the test table and the cluster bootstrap (seed given) | as printed | $10^{-4}$ for a bit-exact generator |
| 7 | regression on the test rows and its bootstrap | as printed | $10^{-4}$ |
| 7 | the 16 runs: seeds, chosen $\lambda$ and $k$, scores; the summary rows | as printed | $10^{-4}$ for scores; seeds and choices exact |
| 7 | ablation: gaps and intervals | as printed | $10^{-4}$ |
| 7 | shift slice: scores, reliability rows, feature means, ranges | as printed | $10^{-4}$ |

Two conventions apply to the whole list. A number computed from a generator started at a printed seed is reproduced only by a bit-exact generator, so test in order: the raw outputs, then the samplers, then the data, then the splits, then everything after them. A mismatch in the data table points at the generator or at the order of the draws; a mismatch in a later table with correct data points at the step that table tests. And where the table says "the stated margin", the criterion is part of the test and not a number to copy.

A compact fingerprint of a correct pipeline, to check at the end:

<!-- worked: project_predictor.acceptance_fingerprint -->
| milestone | quantity | value | tolerance |
|---|---|---:|---|
| 1 | sum of x0 over all 1440 rows | +488.449270 | 5e-7 |
| 1 | rows with fail = 1 over all 1440 rows | 575 | exact |
| 1 | training pumps, first six | 0, 2, 3, 4, 5, 7 | exact |
| 3 | normal-equation weights, intercept and z0 | 3.03368234, 2.04753554 | 1e-8 |
| 4 | chosen lambda; mean CV log loss at lambda = 0 | 0.01; 0.505518 | exact; 1e-6 |
| 4 | logistic weights, intercept and z0 | -0.65391993, 1.09107731 | 1e-6 |
| 5 | scale a fitted on the validation rows | 0.75204425 | 1e-6 |
| 6 | chosen k | 61 | exact |
| 7 | test RMSE, linear regression | 0.629391 | 1e-6 |
| 7 | test ROC AUC, logistic | 0.817276 | 1e-6 |
| 7 | test log loss, logistic | 0.493936 | 1e-6 |
<!-- /worked -->

## Summary

If every acceptance test passes you have built and checked: a split that keeps whole pumps together, with a probe that shows what happens if it does not; two baselines for each target; a linear regression solved two ways that agree; a penalised logistic classifier whose penalty was chosen on folds of pumps, with a gradient verified numerically; the metrics of Chapter 6 computed two ways where two ways exist; a classical comparison model chosen as carefully as the first; and a report with intervals over pumps, over runs and over seeds, an ablation and a statement of where the model fails.

What the project established, in order. The honest number for the memoriser is $64.10\%$, not $74.65\%$. The simple baselines set a floor ($0.7122$ area from one feature; an RMSE of $1.3564$) that the models clear by a wide margin for `wear` and a narrow one for `fail`. The penalty barely matters at this data size. On one test split the logistic classifier and $k$ nearest neighbours are indistinguishable; over sixteen splits one is slightly better; and the gap between them is smaller than the gap between honest and dishonest folds for the model that can memorise. The temperature scale fitted on a small validation set did not generalise. And the model fails at a second site, where it is worse than no model, for a reason the test set could never have shown, because the test pumps came from the same world as the training pumps.

Where this stops. The project has one source, one pair of targets and no tuning beyond one penalty and one neighbour count; the intervals describe the uncertainty from the splits of $300$ pumps, not from the choice of those pumps; the shift slice is a single, known shift. A real project would also monitor the incoming data for the kind of shift found here, and would decide in advance what to do when the feature means move (an extension builds the detector of Chapter 5). The loop of this volume, a loss, its gradient, a step and an honest evaluation, is exactly the loop that Volume III runs on models with millions of parameters. There, nothing can be checked by hand, and the habits built here (the numerical gradient check, the baseline, the interval over pumps, the slice that is read last) are what stand in for checking.

## Questions the predictor leaves open

These are optional. Each is a bigger test of the same ideas.

1. **Repair the failure.** Add $x_0^2$ as a sixth feature (standardised like the others), redo milestones 4 to 7 and report the shift slice again. Does the repair work, and what would a reader who did not know the law have used to find it?
2. **The naive interval.** Repeat the test-set bootstrap resampling rows instead of pumps. How much narrower is the interval of the area, and in how many of $200$ simulated test sets (regenerate the source with other seeds) does each interval contain the area measured on a large fresh sample?
3. **Shift detection.** Following Chapter 5, train a classifier to tell training rows from site B rows and report its area. Write the monitoring rule you would deploy and its false-alarm rate on fresh site A pumps.
4. **Costs.** Redo the report with costs $1:2$ and $1:20$. How does the derived threshold, the recall and the cost per row move, and does the temperature scale matter more or less?
5. **Learning curve.** Train on $30, 60, 90, 120, 150$ and $180$ training pumps and plot validation log loss for both models. Where does the penalty start to matter, and where do the two models cross?
6. **Leave-one-pump-out.** Replace the five folds by one fold per pump ($180$ folds) and compare the chosen penalty, the estimate and the cost with milestone 4.
7. **A second classical model.** Add a Gaussian naive Bayes or a small tree (Chapter 7) as a third model in the paired comparison. Does it change the conclusion?
8. **Newton's method.** Replace the gradient descent of milestone 4 by Newton's method (Chapter 3) and compare the number of steps to a gradient norm below $10^{-8}$ for each $\lambda$.

## Answers

**Recall questions.**

1. The random-row score is higher. Rows of one machine are alike, so a held-out row has siblings in the training set and a model that memorises machines is rewarded; the score on unseen machines is the one that describes use. The difference measures how much of the first score is recognition of machines rather than learning of the pattern (Chapter 5; the leakage probe of milestone 1 shows $74.65\%$ against $64.10\%$).
2. $\mathbf{X}^\top\mathbf{X}\,\mathbf{w} = \mathbf{X}^\top\mathbf{y}$. Gradient descent on the mean squared error is stable when the step is below $2/\lambda_{\max}$, with $\lambda_{\max}$ the largest eigenvalue of the Hessian $\frac{2}{N}\mathbf{X}^\top\mathbf{X}$ (Chapter 2); the ratio of largest to smallest eigenvalue sets how many steps are needed.
3. $\frac{1}{N}\sum_n(\sigma(\mathbf{w}\cdot\mathbf{x}_n) - y_n)\,x_{nj} + \lambda w_j$. The intercept is usually left out of the penalty: a prior that centres it at zero says that the average outcome is probably one half, which is arbitrary, and leaving it free makes the fit independent of the base rate (Chapter 4).
4. $t^\star = c_{\mathrm{FP}}/(c_{\mathrm{FP}} + c_{\mathrm{FN}}) = 1/6$. The area under the ROC curve is the share of (positive, negative) pairs in which the positive has the higher score, a tie counting one half (Chapter 6).
5. Because the two scores in a run are measured on the same pumps, and the same choice of split affects both; the difference of the two removes what they share (Chapter 8). If the models saw different splits, the differences would contain the split noise of both, and pairing would remove nothing.
6. The bootstrap's interval comes from the claim that the units are independent draws; if they are not, resampling them understates the spread (Volume I, Chapter 12). The four rows of a pump share a fingerprint and the pump effects, so the unit is the pump. In the project a resample of the test set is $60$ calls of `rng.integer(60)`, one per pump, each bringing the four rows of the pump it names (Volume I, Chapter 17, for the generator).
