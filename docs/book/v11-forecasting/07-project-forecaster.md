<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Project: A Calibrated Forecasting System

*Can you build a forecaster whose uncertainty estimates are calibrated and honestly evaluated?*

Draw forty random walks of three hundred bars each, one after another from one generator. Nothing in them can be forecast: the best forecast of the next value is the last value, and a forecaster that beats it on bars it has not seen has been lucky or has been told something. Now give them to the forecasting system of this project, with one change in how its windows are scaled. The mean and standard deviation that scale a window are taken over the window's context and over the six bars it is about to forecast, together, where the correct system takes them over the context alone. The system reports a skill of $+0.529$ over the naive forecast on 14,400 forecast bars, with a 95 percent interval from $+0.511$ to $+0.548$, and the intervals it states for those bars contain 81.2 percent of the outcomes where they promise 80. By its own numbers it is a calibrated forecaster with real skill. With the scaler corrected, on the same walks, the skill is $-0.0002$ and the interval runs from $-0.004$ to $+0.003$.

The walks are the control of the project, the panel where the truth is known to be zero. The panel the system is built for is another one, forty series that each have a trend, a season of twelve bars, a regime change at a bar of their own and a stock-out process whose hazard depends on how high the series stood and on how long ago the last stock-out was. The table builds the same system four times and judges each build the way its row says. 'Skill' is measured against the seasonal naive forecast, 'inside' is the share of outcomes that fall inside the central 80 percent interval, the fourth column is the skill the same build shows on the random walks, and the last two columns are the 55 origins that follow the study, with the models left as they were.

<!-- worked: project_forecaster.opening -->
The study panel and the control panel are those of milestones 1 and 5; the same pooled ridge map, Gaussian head and recalibration of the quantile levels (milestones 3, 4 and 6) are built four times, differing only in the way the row says. 'Skill' is one minus the mean scaled quantile loss of the recalibrated learned model over that of the recalibrated seasonal naive forecast, over the origins of the block; the control column is the skill of the recalibrated learned model against the recalibrated naive forecast on 40 random walks, where the truth is zero. 'Inside' is the share of outcomes inside the central 80 percent interval (nominal 80). 'Test' is the test block of the study panel, 'deployment' the 55 origins that follow it. Every number carries the digits on which the run and its four nudged copies agree.

| how the system was built and judged | skill, test | inside, test (%) | skill, control | skill, deployment | inside, deployment (%) |
|---|---:|---:|---:|---:|---:|
| the system as specified: forward split, recalibration on the calibration block | +0.248 | 78.8 | -0.0002 | +0.240 | 78.9 |
| windows scaled by their context and their target bars together | +0.579 | 79.2 | +0.5287 | -0.727 | 25.0 |
| fit, calibration and test blocks drawn at random from all the windows | +0.244 | 79.3 | -0.0032 | +0.244 | 79.8 |
| recalibration fitted on the test block itself | +0.247 | 80.0 | -0.0002 | +0.241 | 79.9 |
<!-- /worked -->

Read it a column at a time. The first row is the system as it will be specified: a skill of $0.248$ on the test block, intervals that hold $78.8$ percent where they promise 80, nothing to report on the random walks, and the same skill and the same coverage in the block that follows. The second row is the mistake of the first paragraph. On the test block it looks better than the system as specified ($+0.579$ against $+0.248$) and just as calibrated, and the two columns that distinguish it are the control, which shows $+0.5287$ on noise, and the deployment block, where the windows can no longer be scaled by bars that have not happened: the skill is $-0.727$ and the intervals hold $25.0$ percent. The third and fourth rows are the opposite case. A split that scatters the windows of every block through the series, and a recalibration fitted on the block that is then used to judge it, leave every one of these columns where they were. Nothing in the table says they are wrong. They are, and where that shows can be measured: in a count of forecast bars that two blocks share, in a calibration error that is exactly zero on the data it was fitted on and is not zero afterwards, and in a control that cannot see either of them and so cannot be the only guard.

That is the shape of the project. You build a small end-to-end forecasting system, and then you build the tests that tell you which of its mistakes they can see. The pieces are those of this volume in the order a forecast is made: a panel of related series, each with an event, and a leak-safe cut of it into windows ([Chapter 1](01-time-series.md)); baselines and one learned model, each stating its forecast as quantiles ([Chapters 2](02-probabilistic-forecasting.md) and [3](03-foundation-forecasters.md)); an evaluation by skill, with an interval that respects the order of the origins and a negative control ([Chapter 4](04-evaluating-forecasts.md)); a recalibration fitted on a window of its own ([Chapter 6](06-calibrated-decisions.md)); an event forecast read over a short horizon ([Chapter 5](05-survival-analysis.md)); and a decision layer that turns quantiles into orders and probabilities into alerts, and is scored by what it actually cost.

![A two-row diagram. Upper row, left to right: the panel of 40 series of 300 bars with season, regime change and stock-outs; the blocks fit, calibration, test and deployment with a purge and an embargo (marked as the place of mistake 1, a random split); the four forecasters naive, seasonal naive, seasonal AR(1) and a pooled ridge map, each with a Gaussian head (marked as the place of mistake 2, a scaler that sees the future); the scaled quantile loss, skill with a block bootstrap, coverage and the random-walk control. A path leads from the evaluation down to the lower row, labelled the forecasts of the test block. Lower row, left to right: the repair of the quantile levels by the probability integral transform and the isotonic fit of the event probability (marked as the place of mistake 3, fitted on the test block); orders at the 0.9 quantile and alerts at the cost threshold; the cost actually incurred in the test and the deployment blocks.](img/forecaster-loop.svg)

`scripts/book/worked/project_forecaster.py`, the module that produced every number below, imports the worked modules of the chapters instead of copying them: the sample-series recipe, the baselines, the window scaling and the purge-and-embargo split of Chapter 1 (`time_series`), the pinball loss, the Gaussian quantile and the percentile of Chapter 2 (`probabilistic_forecasting`), the pooled ridge map over scaled windows of Chapter 3 (`foundation_forecasters`), the block bootstrap and the random walk of Chapter 4 (`evaluating_forecasts`), and the isotonic fit, the Brier score and the calibration error of Chapter 6 (`calibrated_decisions`), with the book's generator and the stable-digits rule. What is written here is the glue: the panel and its events, the way the blocks are cut and scored, what the learned model and its head are, how the repair is fitted, what a decision costs, and what the planted mistakes are.

You build it in any language. You need double-precision arithmetic with a correctly rounded `sqrt`, `exp`, `log`, `sin` and `cos`, the standard Gaussian distribution function and its inverse from a library (any routine accurate to $10^{-12}$ gives the printed digits), the book's generator ([Volume I, Chapter 17](../v01-mathematics/17-randomness.md)), and the routines of the six chapters, each named where it is used. Six questions bring back what the milestones lean on.

1. (Chapter 1, the scale of a window.) A window has the context $4, 6, 8, 10$ and $\epsilon = 10^{-5}$. What are $\mu$ and the scale $d = \sigma + \epsilon$ (divisor $c$), and the scaled context?
2. (Chapter 1, purge and embargo.) For $n = 240$, $c = 24$, $h = 6$, $f_{\mathrm{train}} = 0.5$, $f_{\mathrm{val}} = 0.2$ and an embargo of 6, what are the gap, the two cuts, and the first and last origin of the validation band?
3. (Chapter 2, the pinball loss.) An order costs 1 per unit left over and 9 per unit short. Which quantile level minimizes the expected cost, and what are the pinball loss at that level and the actual cost of an order of 10 when the demand turns out to be 12, and when it turns out to be 8?
4. (Chapter 4, blocks and a control.) Adjacent origins forecast six bars each. How many forecast bars do two neighbouring origins share, why does a bootstrap that resamples single origins state an interval that is too narrow, and what skill against naive can an honest forecaster have on a driftless random walk?
5. (Chapter 6, from a probability to an action.) A false alarm costs 1 and a miss costs 5. At what probability is an alert as cheap as no alert, and why must a probability be calibrated before that threshold is applied to it?
6. (Volume II, [Chapter 2](../v02-learning/02-linear-regression.md), least squares.) What are the normal equations of a ridge fit with a constant, and what does the penalty do to them when two input columns are copies of each other?

## The rules every number follows

Every random input comes from the book's SplitMix64 at the seed `seed_of(name)` printed beside its table, drawn in the order the text states: the panel from `project_forecaster.panel`, the events from `project_forecaster.events`, the control from `project_forecaster.placebo`, the random split from `project_forecaster.random_split`, the bootstrap from `project_forecaster.bootstrap`. Arithmetic is in 64-bit floating point. Bars count from 0. A window has $c = 24$ context bars and $H = 6$ forecast bars, its origin $o$ is the index of its first forecast bar, and the forecast bars are numbered $h = 0, \dots, 5$ from the origin, as in Chapter 1.

Nothing in this system is chaotic: the ridge fit is a direct solve, the event model converges by Newton's method, and the bootstrap draws from a fixed stream. The stable-digits rule is applied all the same, because a claim of stability that is not checked is the usual small lie of a system like this. Every number in the tables is computed five times, from the panels as generated and from four copies whose first value, the first bar of the first series, is changed by $+10^{-12}$, $-10^{-12}$, $+3 \times 10^{-12}$ and $-3 \times 10^{-12}$ (the events are drawn once from the panel as generated, and the control panel is nudged in the same way), and a digit is printed only if all five runs agree on it. They agree on every digit shown, so the printed values are the full values. Counts of operations and bytes, counts of bars and the sizes of blocks are exact.

The system is judged on two things that are kept apart on purpose. The **study** is the first 240 bars of the panel; it is cut into a fit, a calibration and a test block, and everything the system is allowed to know it learns from the first two. The last 60 bars are the **deployment block**: the forecasts of origins 240 to 294 are made by the models as they stand at the end of the study, and nothing about them is fitted again. A number measured there is what happened next, and it is the one thing in the project that no choice made during the study can touch.

## Building the forecaster in nine steps

The pseudocode follows [Pseudocode Conventions](../pseudocode.md). Build and check each milestone before the next: a disagreement at milestone 6 that follows agreement at 1 to 5 points at the repair and not at the panel.

### Milestone 1: a panel with a season, a regime change and stock-outs

A forecasting system needs many related series and one thing in each of them that it can be asked about besides its next value. The panel has forty series of three hundred bars. A series is a level, a slope, a season of twelve bars with an amplitude and a phase, and a remainder that remembers 70 percent of its last value, as in the series of Chapter 1, in units of the level; at a bar of its own, drawn between 100 and 299, the whole series moves by a factor $1 + \mathrm{jump}$. The bar of the move is late on purpose. The fit block ends at bar 114, and only 3 of the 40 series have moved by then, so that the model is fitted on a nearly calm world and the calibration and test blocks and the deployment block see the moves arrive: a model that has seen almost no regime change is the usual case, and one whose intervals were fitted on a calm window is the usual way to be miscalibrated.

```text
function draw_series(rng: Generator, n: int) -> (array<real>[n], int):
    level  = 20.0 * 5.0 ^ rng.uniform()                  # the draws, in this order
    slope  = 0.0005 + 0.0025 * rng.uniform()
    amp    = 0.05 + 0.20 * rng.uniform()
    phase  = 2.0 * pi * rng.uniform()
    shock  = 0.01 + 0.03 * rng.uniform()
    change = 100 + rng.integer(200)
    jump   = -0.25 + 0.50 * rng.uniform()
    v = 0.0
    y = zeros(real, [n])
    for t in 0 .. n:
        v = 0.7 * v + shock * rng.normal()               # one normal() per bar
        scale = 1.0 + (jump if t >= change else 0.0)
        y[t] = level * (1.0 + slope * t + amp * sin(2.0 * pi * t / 12 + phase) + v) * scale
    return (y, change)
```

The second process is the event. A stock-out is a bar at which something happens that the series itself does not show directly, and the useful question about it is whether one will happen soon. Its hazard rises with how high the last bar stood in its window ($z_t$, the last bar of the 24 before $t$ measured from their mean in units of their scale), rises with the time since the last stock-out ($\mathrm{age}_t$, capped at 12 bars and divided by 12 so that the feature stays in $[0, 1]$), and its log-odds rise by a fixed amount once the series has changed regime. The system will see $z$ and the age and never the regime. The events are drawn from a generator of their own so that the series do not depend on them.

```text
function draw_events(y: array<real>[n], change: int, c: int, rng: Generator)
        -> (array<int>[n], array<real>[n], array<real>[n]):
    b = -6.0 + 0.8 * rng.uniform()                       # the series' own intercept, drawn first
    age = 0
    ev = zeros(int, [n]); zs = zeros(real, [n]); ages = zeros(real, [n])
    for t in 0 .. n:
        z = 0.0
        if t >= c:
            (mu, d) = window_scale(y[t - c:t], 1e-5)     # Chapter 1; d = standard deviation + 1e-5
            z = (y[t - 1] - mu) / d
        a = min(age, 12) / 12
        eta = b + 1.6 * z + 1.2 * a + (0.9 if t >= change else 0.0)
        hit = 1 if rng.uniform() < sigmoid(eta) else 0   # one uniform() per bar
        ev[t] = hit; zs[t] = z; ages[t] = a
        age = 0 if hit == 1 else age + 1                 # age counts the bars since the last stock-out
    return (ev, zs, ages)
```

The panel is the 40 calls of `draw_series` in turn on one generator, $n = 300$; the events are the 40 calls of `draw_events` in turn on a second one. (The 300 calls of `normal()` of a series are an even number, so no Gaussian value is held from one series to the next; `rng.integer(200)` rejects raw values below $2^{64} \bmod 200$, as the book's generator does.) The block gives the generators and prints enough of the first series to check a draw order: the parameters of series 0, 1 and 2, the first six values of series 0, the bars at which series 0 has its stock-outs, three values of its $z$, and the counts that the other series add up to.

<!-- worked: project_forecaster.panel_setup -->
The panel has 40 series of 300 bars. Generator `project_forecaster.panel` (SplitMix64, seed 0xef57210e805cc99a), series after series; for series i the draws, in this order, are seven values (six calls of uniform() and one of integer(200)) and then 300 calls of normal(): u1 for level = 20 * 5^u1; u2 for slope = 0.0005 + 0.0025 u2; u3 for amp = 0.05 + 0.20 u3; u4 for phase = 2 pi u4; u5 for shock = 0.01 + 0.03 u5; integer(200) for change = 100 + that integer; u6 for jump = -0.25 + 0.50 u6. Then v_t = 0.7 v_(t-1) + shock normal() with v_(-1) = 0, and y_t = level (1 + slope t + amp sin(2 pi t / 12 + phase) + v_t) (1 + jump [t >= change]) for t = 0 .. 299 (the library sine; the 300 calls of normal() are an even number, so no Gaussian value is held from one series to the next). Events: generator `project_forecaster.events` (seed 0x2c05f27c6f4d6798), series after series, one uniform b for the intercept b = -6.0 + 0.8 u and then one uniform per bar t = 0 .. 299; with z_t = (y_(t-1) - mean) / (sd + 1e-5) over the 24 bars y_(t-24) .. y_(t-1) (mean and sd with divisor 24; 0 for t < 24), age_t the bars since the last event capped at 12 and divided by 12 (age_0 = 0, 0 after an event) the bar is an event when its uniform is below sigmoid(b + 1.6 z_t + 1.2 age_t + 0.9 [t >= change]).

First table: the parameters of series 0, 1 and 2, with the sum of all 300 values, the last value and the number of events.

| i | level | slope | amp | phase | shock | change | jump | sum of y | y at bar 299 | events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30.650565 | 0.001019 | 0.223259 | 2.392362 | 0.031425 | 189 | -0.029610 | 10431.398759 | 44.661397 | 16 |
| 1 | 61.256929 | 0.002406 | 0.056313 | 3.946935 | 0.037977 | 243 | 0.077211 | 25333.015051 | 116.606868 | 16 |
| 2 | 51.844175 | 0.001894 | 0.148433 | 1.691854 | 0.026869 | 125 | 0.024416 | 20194.456741 | 91.464565 | 14 |

Series 0 starts 35.990220, 31.657905, 28.652900, 26.009594, 22.567062, 23.702273; its events fall on bars 32, 58, 71, 98, 132, 142, 158, 205, 216, 227, 228, 238, 251, 263, 264, 275; its z_24, z_25, z_26 are 1.415716, 1.087904, 0.421068. Change bars by fifties: 100 .. 149: 11, 150 .. 199: 9, 200 .. 249: 10, 250 .. 299: 10 series. Series whose change bar is at most 114 (the last bar a fit window reads): 3; at most 119, 179, 239 and 299: 6, 17, 27 and 40. Events in all: 487; in bars 0 .. 119: 135, 120 .. 179: 101, 180 .. 239: 113, 240 .. 299: 138.
<!-- /worked -->

The stock-out count by block of bars shows the regime changes working: 135 in the first 120 bars, where 6 series have moved, and then 101, 113 and 138 in the three blocks of 60 bars after them, by the ends of which 17, 27 and 40 of the series have moved (the counts of series with a change bar of at most 119, 179, 239 and 299 are 6, 17, 27 and 40). A stock-out in the 3 bars $o, o+1, o+2$ is the event the system will forecast at origin $o$, so the share of windows that have one drifts upwards from block to block, and what the model learns in the fit block understates it later; milestone 6 measures how much.

**Milestone 1 ends** when your 40 series reproduce the sums and last values of the first table to six decimals, and your series 0 has the same 16 stock-outs on the same bars. Nothing that follows can be checked if the panel is not the same panel.

### Milestone 2: blocks that cannot leak

The windows of the study are the origins 24 to 234 of every series, 211 per series. They are cut into blocks by the `temporal_split` of Chapter 1 with $n = 240$, $c = 24$, $h = 6$, $f_{\mathrm{train}} = 0.5$, $f_{\mathrm{val}} = 0.2$ and an embargo of 6: the gap is $\max(6, 6) = 6$, the cuts are bars 120 and 168, and the three bands are the fit block, the calibration block (the validation band of the rule) and the test block. The windows that fall in an embargo band are dropped. Every block contains all forty series, so that the origins of a block are shared across the panel, and a block is scored origin by origin as Chapter 4 does: the loss of an origin is the mean over the forty series.

The deployment block is cut by hand: origins 240 to 294, whose forecast bars are bars 240 to 299. It needs no purge, because nothing is fitted on it.

The table that matters is not the one of origins. It is the one that counts, for each block, how many of its distinct forecast bars are also a bar of a window of an earlier block, context or target. The rule of Chapter 1 promises none for the target bars, and the module checks the promise. It also draws the random split of the planted mistake of milestone 8, so that the same count can be read for it: the origins 24 to 234 are put in increasing order, shuffled with the book's `shuffle`, and cut into blocks of the sizes above, 86, 31 and 60, each sorted again afterwards.

<!-- worked: project_forecaster.split_table -->
The study is bars 0 .. 239. A window has 24 context bars and 6 forecast bars; its origin o is the index of its first forecast bar, the origins are 24 .. 234 (211 windows per series) and the rule is the `temporal_split` of Chapter 1 with n = 240, c = 24, h = 6, f_train = 0.5, f_val = 0.2, embargo 6: gap = 6, cuts floor(240 * 0.5) = 120 and floor(240 * 0.7) = 168, the validation side of the rule is the calibration block. The deployment block is the origins 240 .. 294, whose forecast bars are bars 240 .. 299. Every block contains all 40 series. Second table: for each block, how many of its distinct forecast bars are also a bar (context or forecast) of a window of the earlier block named in the column. Third table: the same count for the test block against the fit block under the forward split and under the random split `project_forecaster.random_split` (seed 0xb69f481c8ac3efac): the origins 24 .. 234 in increasing order are shuffled by the book's `shuffle`, the first 86 (sorted) are the fit origins, the next 31 the calibration origins and the next 60 the test origins; the first five shuffled origins are 222, 156, 116, 169, 231.

| block | origins | windows per series | rows | bars read (context and forecast) |
|---|---:|---:|---:|---:|
| fit | 24 .. 109 | 86 | 3440 | 0 .. 114 |
| calibration | 127 .. 157 | 31 | 1240 | 103 .. 162 |
| test | 175 .. 234 | 60 | 2400 | 151 .. 239 |
| deployment | 240 .. 294 | 55 | 2200 | 216 .. 299 |

| forecast bars of the block | also a bar of a fit window | also a bar of a calibration window | also a bar of a test window |
|---|---:|---:|---:|
| calibration | 0 | - | - |
| test | 0 | 0 | - |
| deployment | 0 | 0 | 0 |

| split | distinct forecast bars of the test block | that are fit forecast bars | that are any fit bar |
|---|---:|---:|---:|
| forward split | 65 | 0 | 0 |
| random split | 195 | 179 | 194 |
<!-- /worked -->

The test block's forecast bars share nothing with any bar a fit window reads. (The calibration block's context starts at bar 103 and the fit block's last forecast bar is bar 114, so a window of one block reads bars that a window of the other forecast; the answer to question 4 of Chapter 1's check of understanding applies. No bar that is a target in the later block was ever a bar of the earlier one, and that is what the count measures.) In the random split the test block has 195 distinct forecast bars, and 179 of them are forecast bars of fit windows as well; 194 are read by some fit window in one way or the other.

**Milestone 2 ends** when your blocks have the origins and row counts of the first table, the overlap counts of the second table are zero, and your random split has the same first five shuffled origins and the overlap counts of the third table.

### Milestone 3: baselines that state their uncertainty

A skill is a statement about two forecasters, and the baseline has to state its forecast in the same form as the model. A quantile forecast of Chapter 2 needs a point and a spread; the forecasters of this milestone differ only in their point, and share the way their spread is made. The four points are the forecasts of Chapter 1, each from the bars before the origin and no others. Naive repeats $y_{o-1}$. Seasonal naive repeats the last season, $y_{o-12+h}$ for $h = 0, \dots, 5$. The seasonal AR(1) is fitted again at every origin to the seasonal differences $y_t - y_{t-12}$ of the whole past, an AR(1) with a constant by least squares, and adds the forecast differences back to the season before. The fourth is the learned model of milestone 4.

The spread is a Gaussian head. At each horizon $h$ the head has one number $\hat\sigma_h$, the square root of the mean over the rows of the fit block of the squared error of the point, in units of the window's scale $d = \sigma + \epsilon$:

$$
\hat\sigma_h = \sqrt{\frac{1}{N_{\mathrm{fit}}}\sum_{\text{fit rows}} \Bigl(\frac{y_{o+h} - \mathrm{point}_h}{d}\Bigr)^2}, \qquad q_{\tau, h} = \mathrm{point}_h + d\,\hat\sigma_h\,m_\tau,
$$

with $m_\tau = \Phi^{-1}(\tau)$ the Gaussian quantile for a raw head and the multipliers of milestone 6 for a repaired one. A **row** is a series and an origin; the fit block has $40 \times 86 = 3440$ of them, accumulated series by series with the origins in increasing order. The quantile levels are $\tau = 0.1, 0.25, 0.5, 0.75, 0.9$.

The score of a forecaster at a row is the mean over the six horizons and the five levels of the pinball loss, in units of the context's own scale (the same $d$, with divisor $c$, whether or not the head used it):

```text
function scaled_quantile_loss(q: array<real>[H, L], y: array<real>[H], levels: array<real>[L],
        d_loss: real) -> real:
    total = 0.0
    for h in 0 .. H:
        for k in 0 .. L:
            total = total + pinball(q[h, k], y[h], levels[k])      # Chapter 2
    return total / (H * L * d_loss)
```

The loss $E_o$ of an origin is the mean of this over the forty series, and the loss of a block is the mean of $E_o$ over its origins, as in Chapter 4. Dividing by the context's scale makes the forty series, whose levels run from 20 to 100 and further, comparable and averageable; it is the in-window cousin of the scaled errors of Chapter 4. The table gives $\hat\sigma_h$ for the four forecasters: it grows with the horizon for each of them, from $0.5777$ to $1.9574$ for naive, whose error is the change since the last bar and grows as the season and the trend move away from it, and only from $0.6353$ to $0.6692$ for seasonal naive, whose error is a season's difference whatever the horizon.

<!-- worked: project_forecaster.heads -->
The fit block has 3440 rows (40 series times 86 origins), accumulated series by series with the origins in increasing order. Each of the four forecasters has a Gaussian head: the forecast of bar o + h is the point forecast plus d sigma_h z, where d = sd + 1e-5 is the context's scale and sigma_h is the root of the mean over the fit rows of ((y_(o+h) - point_h) / d)^2. The table gives sigma_h for h = 0 .. 5 (the first forecast bar is h = 0); the learned map's point is the forecast of milestone 4.

| forecaster | h = 0 | h = 1 | h = 2 | h = 3 | h = 4 | h = 5 |
|---|---:|---:|---:|---:|---:|---:|
| naive | 0.5777 | 1.0222 | 1.4091 | 1.7039 | 1.8902 | 1.9574 |
| seasonal naive | 0.6353 | 0.6459 | 0.6562 | 0.6644 | 0.6671 | 0.6692 |
| seasonal AR(1) | 0.4210 | 0.5249 | 0.5845 | 0.6234 | 0.6465 | 0.6639 |
| pooled ridge map | 0.3097 | 0.4056 | 0.4743 | 0.5161 | 0.5386 | 0.5510 |
<!-- /worked -->

**Milestone 3 ends** when your four rows of $\hat\sigma_h$ agree with the table to the digits shown. The naive row is the one to look at first: it needs only the panel and the fit rows.

### Milestone 4: a learned model that is shared by all the series

The learned model is the shared map of Chapter 3, fitted to this panel. Every window is scaled by its own context, so that a series at level 20 and one at level 100 look alike; the map takes the 24 scaled context bars and returns the 6 scaled forecast bars; one map serves all forty series, and it is fitted by ridge regression with a penalty of 1 on every weight but the constant. The forecast is mapped back with the window's own $\mu$ and $d$.

```text
function scaled_window(y: array<real>[n], o: int, c: int, h: int, future: bool)
        -> (array<real>[c], array<real>[h], real, real, real):
    require o >= c and o + h <= n, "the window is outside the series"
    ctx = y[o - c:o]
    (_, d_loss) = window_scale(ctx, 1e-5)
    span = y[o - c:o + h] if future else ctx             # future = true is mistake 2 of milestone 8
    (mu, d) = window_scale(span, 1e-5)
    return ((ctx - mu) / d, (y[o:o + h] - mu) / d, mu, d, d_loss)

function fit_map(panel: list<array<real>[n]>, origins: list<int>, c: int, h: int, lam: real,
        future: bool) -> array<real>[h, c + 1]:
    p = c + 1
    A = zeros(real, [p, p])
    B = zeros(real, [p, h])
    for y in panel:                                      # series in order, then origins in increasing order
        for o in origins:
            (x, t, _, _, _) = scaled_window(y, o, c, h, future)
            row = [1.0] + x
            for i in 0 .. p:
                for j in i .. p:                         # the upper triangle, accumulated row after row
                    A[i, j] = A[i, j] + row[i] * row[j]
                for k in 0 .. h:
                    B[i, k] = B[i, k] + row[i] * t[k]
    for i in 1 .. p:
        A[i, i] = A[i, i] + lam                          # the constant (index 0) is not penalized
    for i in 0 .. p:
        for j in 0 .. i:
            A[i, j] = A[j, i]
    W = zeros(real, [h, p])
    for k in 0 .. h:
        W[k, :] = solve(A, B[:, k], true)                # elimination with partial pivoting
    return W
```

The forecast of bar $o + k$ is $\mu + d\,(W_{k,0} + \sum_j W_{k,j+1} x_j)$ with $x$ the scaled context, and its head is the Gaussian head of milestone 3 with its own $\hat\sigma_h$. The map has $25 \times 6 = 150$ weights, fitted on 3440 rows. The block prints the weights of its first output. They do not read like a recipe, and a ridge fit does not promise that they will: the newest bar has weight $+0.8073$ and nearly all the others are small, so whatever the map knows of the season is spread over many of them. What the map has learned is checked by what it forecasts, in the next milestone, and the check is not a weight.

<!-- worked: project_forecaster.learned_map -->
The ridge map of 25 inputs and 6 outputs, penalty 1 on every weight but the constant, fitted on the 3440 rows of the fit block. The table gives the weights of its first output (h = 0), w0 the constant and w1 .. w24 the weights of the scaled context bars from the oldest to the newest; the row labelled w5 holds w5 .. w9, and so on.

| weights | +0 | +1 | +2 | +3 | +4 |
|---|---:|---:|---:|---:|---:|
| w0 | +0.0705 | -0.0332 | -0.0375 | +0.0459 | -0.0162 |
| w5 | -0.0315 | -0.0408 | -0.0057 | -0.1473 | -0.0031 |
| w10 | -0.1381 | -0.0063 | -0.0331 | +0.0024 | -0.0406 |
| w15 | +0.0208 | -0.0216 | +0.0580 | -0.0371 | -0.1088 |
| w20 | -0.0573 | -0.0834 | -0.1055 | +0.0127 | +0.8073 |
<!-- /worked -->

**Milestone 4 ends** when your weights $w_0$ to $w_{24}$ of the first output agree with the table to four decimals, and your $\hat\sigma_h$ of the learned row with the fourth row of the table of milestone 3. A disagreement there is in the Gram matrix or in the order in which it was accumulated, since the elimination is the one of Chapter 3.

### Milestone 5: skill with an interval, and a control

The evaluation takes the test block, 60 origins of 40 series, and makes three statements about each forecaster: the scaled quantile loss, the skill against the seasonal naive forecast, and the share of outcomes inside the central 80 and 50 percent intervals. The skill of a forecaster is $\mathrm{Sk} = 1 - \bar E^{\mathrm{A}} / \bar E^{\mathrm{SN}}$ over the origins of the block, and its interval is the paired block bootstrap of Chapter 4 with a block of 6 origins (the horizon, since neighbouring origins share five of their six forecast bars) and 2000 resamples, the 2.5th and 97.5th percentiles, from the generator `project_forecaster.bootstrap`. Each forecaster is scored with its head raw and with its head repaired by milestone 6, and the skill of a repaired forecaster is taken against the repaired seasonal naive forecast, which is what is meant below by the skill of the system. (Skill against a raw baseline would credit the model with the baseline's miscalibration.)

<!-- worked: project_forecaster.evaluation -->
The test block: 40 series times 60 origins, 2400 rows. The loss of a row is the mean over the 6 horizons and the 5 levels 0.1, 0.25, 0.5, 0.75, 0.9 of the pinball loss of the quantile against the outcome, divided by d_loss = sd + 1e-5 of the row's 24 context bars (divisor 24); the loss of an origin is the mean over the 40 series, and the tables use the mean over the 60 origins. 'Raw' quantiles are point + d sigma_h z_tau; 'recalibrated' ones use the multipliers of milestone 6. Skill is 1 - mean loss / mean loss of the seasonal naive forecast with the same kind of head. The interval is the block bootstrap of Chapter 4 with blk = 6 and 2000 resamples over the 60 origins, 2.5th and 97.5th percentiles, generator `project_forecaster.bootstrap` (seed 0x4c6f325f52edd4f3) drawing the rows of the third table in its order and then the four rows of the control table of milestone 5. 'Inside' is the share of the 14400 outcomes (rows times horizons) between the two quantile forecasts.

| forecaster | loss, raw head | loss, recalibrated | skill against seasonal naive, raw | recalibrated |
|---|---:|---:|---:|---:|
| naive | 0.4041 | 0.3988 |  |  |
| seasonal naive | 0.1867 | 0.1740 | - | - |
| seasonal AR(1) | 0.1577 | 0.1554 | +0.155 | +0.107 |
| pooled ridge map | 0.1325 | 0.1309 | +0.290 | +0.248 |

| forecaster | head | inside 80% (nominal 80) | inside 50% (nominal 50) |
|---|---:|---:|---:|
| seasonal naive | raw | 83.9 | 61.7 |
| seasonal naive | recal | 78.0 | 47.2 |
| seasonal AR(1) | raw | 85.3 | 63.8 |
| seasonal AR(1) | recal | 77.8 | 46.6 |
| pooled ridge map | raw | 84.9 | 61.8 |
| pooled ridge map | recal | 78.8 | 46.5 |

| comparison | head | 95% interval of the skill |
|---|---:|---:|
| learned map against seasonal naive | raw | [+0.269, +0.318] |
| learned map against seasonal naive | recalibrated | [+0.227, +0.278] |
| seasonal AR(1) against seasonal naive | raw | [+0.134, +0.181] |
| seasonal AR(1) against seasonal naive | recalibrated | [+0.087, +0.131] |
<!-- /worked -->

The learned map beats seasonal naive by $0.290$ with raw heads and by $0.248$ with repaired ones, and the interval of the second, $[0.227, 0.278]$, lies far from zero; the seasonal AR(1), which refits at every origin, has $0.155$ raw and $0.107$ repaired. The learned map's loss $0.1325$ is $29.0$ percent below seasonal naive's $0.1867$ before repair, and the repair moves it by only $0.0016$ ($0.1309$): what the model gets right is the point, and the repair changes only the head. Where the width is wrong is in the second table. Raw, the central 80 percent intervals of the three forecasters hold $83.9$, $85.3$ and $84.9$ percent of the outcomes, too wide by four to five points, and the central 50 percent intervals hold $61.7$, $63.8$ and $61.8$ percent, too wide by about twelve to fourteen. Repaired on the calibration block, the same three hold $78.0$, $77.8$ and $78.8$ percent and $47.2$, $46.6$ and $46.5$: they now err on the narrow side, because the test block has heavier tails than the calibration block (milestone 6 shows it).

The control uses the same machinery on a panel where the answer is zero. Forty driftless random walks of 300 bars (the first value `normal()` and each next value the last plus `normal()`), the same blocks, windows, ridge map, heads and repairs; the best forecast of a random walk is naive, so the skill of any forecaster against naive should be at most zero, and an evaluation that shows more has found something that is not there. The four rows are the system as specified and the three mistakes of milestone 8, which reads them, with the bootstrap interval of the skill of the repaired learned model against the repaired naive forecast. The seasonal naive forecast, which is bad on a random walk, has a skill of $-0.670$ against naive, and the seasonal AR(1) of $-0.262$.

<!-- worked: project_forecaster.control -->
40 driftless random walks of 300 bars, generator `project_forecaster.placebo` (seed 0x04179795a8d72a1e), series after series: x_0 = normal(), x_t = x_(t-1) + normal() (the 300 calls are an even number). The same splits, windows, ridge map, heads and recalibrations as milestones 2 to 6; the naive forecast is the best forecast there is. The skill is that of the recalibrated learned model against the recalibrated naive forecast on the test block, with the block-bootstrap interval of milestone 5 (the generator continues from the rows of that table); the rows differ in the way the system was built, as in the opening table. Skill of the seasonal naive forecast against naive on the same walks: -0.670; of the seasonal AR(1): -0.262.

| how the system was built | skill against naive | 95% interval | inside 80% (%) |
|---|---:|---:|---:|
| honest | -0.0002 | [-0.0039, +0.0033] | 78.4 |
| future scaler | +0.5287 | [+0.5113, +0.5476] | 81.2 |
| random split | -0.0032 | [-0.0083, +0.0029] | 78.9 |
| recalibration on the test block | -0.0002 | [-0.0037, +0.0029] | 80.0 |
<!-- /worked -->

The system as specified shows $-0.0002$ with an interval from $-0.0039$ to $+0.0033$: no skill, and not a loss either, which is the fitted map sitting where naive does. This is the number that has to hold before any skill on the real panel is believed. The rest of the table is read in milestone 8.

**Milestone 5 ends** when your losses, skills, intervals and shares agree with the tables to the digits shown. The interval depends on the stream of the generator: it draws the rows of the third table in order, 2000 resamples of 10 block starts each, then the four rows of the control table.

### Milestone 6: repairing the probabilities on a window of their own

The heads of milestone 3 are Gaussians with a width taken from the fit block, a block in which almost no series has moved and in which the learned map has been fitted to the very rows its width is measured on. Neither is a reason to expect the quantiles to hold on the blocks that follow, and the second table of milestone 5 shows they do not. The repair uses a window the models have not been fitted on, the calibration block, and nothing that comes after it.

For each of the four forecasters with a head (naive included), the **probability integral transform** of an outcome is $u = \Phi\bigl((y_{o+h} - \mathrm{point}_h)/(d\hat\sigma_h)\bigr)$, the level at which the head's Gaussian puts the outcome. A calibrated head would have these uniform. The repair collects them over the calibration rows and all six horizons ($1240 \times 6 = 7440$ values), sorts them, and reads the Gaussian quantile of their empirical level for each $\tau$: if the head puts only 6.7 percent of the outcomes below its 0.1 quantile, that quantile is moved up until the outcomes below it are 10 percent of the calibration data. In symbols, with the sorted values $u_{(1)} \le \dots \le u_{(n)}$ extended by $u_{(0)} = 0$ and $u_{(n+1)} = 1$, the multiplier of level $\tau$ is $m_\tau = \Phi^{-1}$ of the value at rank $\tau(n + 1)$, interpolated linearly between neighbouring ranks:

```text
function recal_multipliers(u: array<real>[n], levels: array<real>[L]) -> array<real>[L]:
    require n >= 1, "no calibration data"
    ext = [0.0] + sort(u) + [1.0]                       # ext[r] is u_(r), r = 0 .. n + 1
    m = zeros(real, [L])
    for k in 0 .. L:
        require 0.0 < levels[k] and levels[k] < 1.0, "a level must be inside (0, 1)"
        pos = levels[k] * (n + 1)
        j = floor(pos)
        m[k] = Phi_inv(ext[j] + (pos - j) * (ext[j + 1] - ext[j]))
    return m
```

The same multipliers apply to every row and every horizon, so the repair is five numbers per forecaster, and it can change the width and the tilt of a head but not its centre or the way its width follows the horizon. The multipliers of the four heads are in the first table, and the second table reads the learned map's quantile levels on the data they were fitted on and on the test block. On the calibration block the repaired shares are $10.0$, $25.0$, $50.0$, $75.0$ and $90.0$ percent, exactly, which is what fitting five numbers to 7440 values does and proves nothing. The test block is the check: $10.7$, $27.0$, $50.4$, $73.5$ and $89.5$. The repair has brought every level to within two points of its nominal share, and at four of the five levels to the far side of it from where the raw head was: the test block has heavier tails than the calibration block, with $10.7$ percent of its outcomes below the 0.1 quantile and $10.5$ above the 0.9 one, and a repair fitted on one window describes that window.

![A line plot of the share of test-block outcomes below the quantile forecast against the quantile level, for the learned head. The dashed diagonal is perfect calibration. The raw Gaussian head lies below the diagonal at the two low levels (6.5 percent at 0.1, 19.0 at 0.25) and above it at the two high ones (80.8 at 0.75, 91.3 at 0.9). The head repaired on the calibration block lies close to the diagonal at every level, slightly above it at the low levels and below it at the high ones.](img/quantile-reliability.svg)

The event has its own model and its own repair. At each row the features are $[1, z_o, \mathrm{age}_o]$ of milestone 1, the label is a stock-out in the three bars $o, o+1, o+2$ (the system never sees the regime or the intercept $b$ of the series), and the model is the logistic regression of Volume II fitted on the fit rows by 12 steps of Newton's method from zero weights. The probabilities it gives are repaired by the isotonic regression of Chapter 6 fitted on the calibration rows. The third table shows the event drifting away from its model, and the repair catching part of it:

<!-- worked: project_forecaster.recalibration -->
Quantile levels. The recalibration data are the 1240 rows of the calibration block (origins 127 .. 157) and their 6 horizons, 7440 probability integral transforms Phi((y_(o+h) - point_h) / (d sigma_h)) per forecaster, with Phi the library distribution function of the standard Gaussian. Sorted as p_(1) <= .. <= p_(n) with p_(0) = 0 and p_(n+1) = 1, the recalibrated level of tau is the p at rank tau (n + 1), linearly interpolated between neighbouring ranks, and the multiplier is the library inverse Gaussian distribution function of it. First table: the multipliers. Second table: the share of outcomes below the quantile forecast of each level, on the calibration block (the data the multipliers were fitted on) and on the test block, for the learned map. Event probabilities. The event of a row is a stock-out in the 3 bars o, o+1, o+2; its features are [1, z_o, age_o] of milestone 1; the model is the logistic regression fitted by 12 steps of Newton's method from zero weights on the 3440 fit rows (sums over rows in order, the largest gradient entry below 1e-8 after the last step), with the weights -4.2688, +0.9754, +1.7308. The share of fit windows with a stock-out is 9.7% and of calibration windows 12.0%. The repair is the isotonic fit of Chapter 6 (pool adjacent violators, equal probabilities keeping their input order) of the stock-out label on the raw probability over the calibration rows, read at the largest fitted point not above p, and at the first block's value below all of them. Third table: the share of windows that had a stock-out, the means of the raw and the repaired probabilities, the Brier score and the ten-bin ECE (equal-width bins of the probability, as in Chapter 6) in each block.

| level tau | raw multiplier z_tau | naive | seasonal naive | seasonal AR(1) | learned map |
|---|---:|---:|---:|---:|---:|
| 0.1 | -1.2816 | -1.2270 | -0.6384 | -1.0246 | -1.0098 |
| 0.25 | -0.6745 | -0.8195 | -0.1371 | -0.4119 | -0.4652 |
| 0.5 | 0.0000 | +0.0283 | +0.2749 | +0.0155 | -0.0501 |
| 0.75 | 0.6745 | +0.9311 | +0.7096 | +0.4450 | +0.4336 |
| 0.9 | 1.2816 | +1.3857 | +1.3068 | +0.9696 | +1.1222 |

| level tau (%) | calibration, raw | calibration, recalibrated | test, raw | test, recalibrated |
|---|---:|---:|---:|---:|
| 10 | 6.7 | 10.0 | 6.5 | 10.7 |
| 25 | 17.4 | 25.0 | 19.0 | 27.0 |
| 50 | 53.0 | 50.0 | 53.3 | 50.4 |
| 75 | 82.3 | 75.0 | 80.8 | 73.5 |
| 90 | 92.0 | 90.0 | 91.3 | 89.5 |

| block | share of windows with a stock-out (%) | mean raw p (%) | mean recalibrated p (%) | Brier, raw | Brier, recal | ECE, raw | ECE, recal |
|---|---:|---:|---:|---:|---:|---:|---:|
| cal | 12.0 | 9.2 | 12.0 | 0.0954 | 0.0911 | 0.0281 | 0.0000 |
| test | 13.5 | 9.0 | 11.8 | 0.1081 | 0.1069 | 0.0453 | 0.0210 |
| deploy | 15.6 | 8.1 | 10.7 | 0.1219 | 0.1160 | 0.0750 | 0.0515 |
<!-- /worked -->

The share of windows with a stock-out rises from $12.0$ percent in the calibration block to $13.5$ and then $15.6$, while the mean probability of the model falls, from $9.2$ to $9.0$ and $8.1$ percent, because the model learned the hazard of a world in which almost no series had moved, and more of the panel has moved in every block. The isotonic repair, fitted on the calibration block, lifts the mean to $12.0$, $11.8$ and $10.7$ percent, and the calibration error from $0.0453$ to $0.0210$ on the test block and from $0.0750$ to $0.0515$ in the deployment block. It cannot follow the drift, because it was fitted at one place in time, and a repair that is fitted once is a repair of the block it was fitted on: the first extension makes it roll.

**Milestone 6 ends** when your multipliers, the shares below each quantile and the three rows of the event table agree with the block to the digits shown. If the shares on the calibration block are not exactly the nominal levels, the rank rule is wrong.

### Milestone 7: orders and alerts, scored by what they cost

A forecast is an input to a decision, and a decision has a cost that the forecast's score does not know about. The system makes two. An order is placed for the third bar after the origin ($h = 2$, a lead time of three bars): too much stock costs 1 per unit of the window's scale and too little costs 9, so the order that minimizes the expected cost is the quantile at $\tau = c_{\mathrm{short}}/(c_{\mathrm{short}} + c_{\mathrm{over}}) = 0.9$ (Chapter 2). An alert is raised for a stock-out in the next three bars: a false alarm costs 1 and a miss costs 5, so an alert is worth raising when $p \ge 1/(1 + r)$ with $r = 5$, that is $p \ge 0.1667$ (Chapter 6).

```text
function order_cost(q: real, y: real, d_loss: real, c_over: real, c_short: real) -> real:
    e = y - q                                            # positive when the order was short
    return (c_short * e if e > 0.0 else -c_over * e) / d_loss

function alert_cost(p: real, event: int, threshold: real, c_fp: real, c_fn: real) -> real:
    if p >= threshold:                                   # alert; a tie alerts
        return c_fp if event == 0 else 0.0
    return c_fn if event == 1 else 0.0
```

Scoring by realized cost is stricter than scoring by a proper rule, and it is what the decision was for: a model that is better on the pinball loss can be worse on a cost that weights one side nine times as much, and an alert threshold that is wrong by a factor of two costs real money. The costs are in the tables as means per row, so that a policy that was never wrong would cost $0$ for an order and for an alert, which is what perfect foresight is.

<!-- worked: project_forecaster.decisions -->
Alerts. The probabilities and the repair are those of milestone 6. An alert costs 1 when no stock-out follows and nothing when one does; no alert costs 5 when one follows and nothing otherwise; the threshold is 1 / (1 + 5) = 0.1667, and a window is alerted when p >= the threshold. First table: mean cost per window (series and origin) in the three blocks, then the counts of alerts.

| policy | calibration block | test block | deployment block |
|---|---:|---:|---:|
| never alert | 0.6008 | 0.6750 | 0.7818 |
| always alert | 0.8798 | 0.8650 | 0.8436 |
| alert when p >= 0.5, raw p | 0.5887 | 0.6696 | 0.7795 |
| alert when p >= 0.1667, raw p | 0.4629 | 0.5246 | 0.5591 |
| alert when p >= 0.1667, recalibrated p | 0.4226 | 0.4763 | 0.4623 |
| number of alerts, raw p | 255 | 485 | 362 |
| number of alerts, recalibrated p | 469 | 873 | 695 |
| number of alerts at p >= 0.5, raw p | 3 | 5 | 1 |

Orders. For each row the order is the quantile 0.9 (or the median) of the third forecast bar (h = 2) of the head named in the row, raw (point + d sigma_h z_tau) or recalibrated (the multipliers of milestone 6 of that forecaster), and its cost is 1 per unit of d_loss ordered above the outcome and 9 per unit of d_loss below it (the pinball loss at 0.9 times ten). Second table: the mean cost per row.

| order | test block | deployment block |
|---|---:|---:|
| seasonal naive, raw head, quantile 0.9 | 1.4176 | 1.3758 |
| seasonal AR(1), raw head, quantile 0.9 | 1.2310 | 1.1815 |
| learned map, raw head, quantile 0.9 | 1.0883 | 1.0626 |
| learned map, raw head, median | 1.8462 | 1.7178 |
| seasonal naive, recalibrated head, quantile 0.9 | 1.4143 | 1.3750 |
| seasonal AR(1), recalibrated head, quantile 0.9 | 1.2245 | 1.1551 |
| learned map, recalibrated head, quantile 0.9 | 1.0855 | 1.0503 |
| learned map, recalibrated head, median | 1.9354 | 1.8044 |
| perfect foresight | 0 | 0 |
<!-- /worked -->

The alert is worth having. On the test block the repaired model costs $0.4763$ per window, against $0.6750$ for never alerting and $0.8650$ for always alerting, and in the deployment block $0.4623$ against $0.7818$ and $0.8436$; the cost threshold applied to the raw probabilities already costs $0.5246$ and $0.5591$, and the repair takes off a further $9.2$ percent on the test block and $17.3$ percent in deployment. The rule that is easiest to reach for, alert when $p \ge 0.5$, is as bad as never alerting ($0.6696$ and $0.7795$): the model gives a probability of one half or more to 5 of the 2400 windows of the test block and to 1 of the 2200 in deployment, so the rule almost never alerts. The repaired model alerts 873 times on the test block and the raw one 485: moving the probabilities up moves more windows across a threshold that was set for calibrated ones.

The order shows the other side of calibration. The repaired quantile at 0.9 costs $1.0855$ per row on the test block and the raw one $1.0883$, a difference in the third decimal, because the 0.9 level of the raw head was already close ($91.3$ percent of the outcomes below it). The point of the table is the rest of it, read head against head, repaired against repaired: with recalibrated heads the learned map's order costs $23.2$ percent less than seasonal naive's ($1.4143$) and $11.4$ percent less than the AR's ($1.2245$), and with raw heads $23.2$ and $11.6$ percent less ($1.4176$ and $1.2310$), so repairing the baselines' heads changes the gap in the first decimal and not the conclusion. Ordering the median, which is the order a point forecaster places, costs $1.9354$ with the repaired head, $78$ percent more than the repaired quantile for the same forecast ($1.8462$ and $70$ percent with the raw head). A forecast that is right on average and is used as a point is the wrong forecast for this decision.

**Milestone 7 ends** when your alert and order costs agree with the two tables of the block, and a policy of never alerting costs exactly the share of windows with a stock-out times 5.

### Milestone 8: three planted mistakes

A system that works is not yet tested. Each of the mistakes below is the kind that a code review signs off, and each is planted alone in an otherwise correct system. The project asks the system to catch each by the check that can see it, and to say which checks cannot.

1. **A random split.** The fit, calibration and test blocks are drawn at random from the 211 origins of the study (the sizes of milestone 2) instead of cut in time. The windows of a block are then scattered through the series between those of the others.
2. **A scaler that sees the future.** Every window is scaled by the mean and the standard deviation of its 24 context bars and its 6 forecast bars together (`future = true` in `scaled_window`), in the fit, in the head and in the test. At deployment the scaler cannot see the forecast bars, since they have not happened, and the windows are scaled by their context alone.
3. **Recalibration on the test block.** The multipliers of milestone 6 are fitted on the rows of the test block, and so is the isotonic repair of the event probability, instead of on the calibration block.

<!-- worked: project_forecaster.mistakes -->
Each mistake is planted alone in the system as specified. The random split is the one of milestone 2 (its test block holds 60 origins as the forward one does, so the rows are the same in number). The future scaler scales every window by the mean and the standard deviation (divisor 30) of its 24 context bars and its 6 forecast bars together: the same scaling for the fit windows, the head and the test windows, the point forecast is mu + d times the map's output with those mu and d, and losses are still divided by the context's own d_loss; in deployment the windows are scaled by their context alone, since the future is not there to be read. The test-block recalibration computes the multipliers of milestone 6 from the test rows, and the isotonic repair of the event probability from the test windows. 'Deployment' is the 55 origins after the study with the models unchanged. In the random split, 179 of the test block's forecast bars are also forecast bars of fit windows. The isotonic repair fitted on the test block takes 19 distinct values. First table: the learned model's skill against the seasonal naive forecast, each with its own head and recalibration, and the share inside the central 80 percent interval; second table: the event model, whose probabilities are repaired by the isotonic fit of the calibration block (the test-block row: of the test block itself). The head scale d of every forecaster, baselines included, is the window's scaling, so under the future scaler the baselines' spread uses the leaky d too.

| mistake | skill, test | skill, deployment | inside 80% raw, test (%) | raw, deployment (%) | recalibrated, test (%) | recalibrated, deployment (%) |
|---|---:|---:|---:|---:|---:|---:|
| none: the system as specified | +0.248 | +0.240 | 84.9 | 84.3 | 78.8 | 78.9 |
| a random split | +0.244 | +0.244 | 87.2 | 86.8 | 79.3 | 79.8 |
| a scaler that sees the future | +0.579 | -0.727 | 82.6 | 28.2 | 79.2 | 25.0 |
| recalibration fitted on the test block | +0.247 | +0.241 | 84.9 | 84.3 | 80.0 | 79.9 |

| mistake | probability | ECE, test | ECE, deployment | Brier, test | Brier, deployment | alert cost, test | alert cost, deployment |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | raw p | 0.0453 | 0.0750 | 0.1081 | 0.1219 | 0.5246 | 0.5591 |
| none | repaired p | 0.0210 | 0.0515 | 0.1069 | 0.1160 | 0.4763 | 0.4623 |
| random split | raw p | 0.0168 | 0.0552 | 0.0884 | 0.1177 | 0.4500 | 0.5068 |
| random split | repaired p | 0.0208 | 0.0574 | 0.0891 | 0.1174 | 0.4383 | 0.4486 |
| recalibration on the test block | raw p | 0.0453 | 0.0750 | 0.1081 | 0.1219 | 0.5246 | 0.5591 |
| recalibration on the test block | repaired p | 0.0000 | 0.0349 | 0.1038 | 0.1148 | 0.4571 | 0.4509 |
<!-- /worked -->

![A dot plot of skill for four builds of the system, numbered 1 as specified, 2 future scaler, 3 random split, 4 test-block recalibration. For each build three dots: the skill on the test block, the skill in the deployment block, and the skill on the random-walk control. Build 1 has all three near 0.25, 0.24 and 0. Build 2 has 0.579 on the test block, -0.727 in deployment and 0.529 on the control. Builds 3 and 4 are near 0.24 and 0 as build 1.](img/skill-reported-deployed.svg)

The scaler is the one every check sees. A structural audit finds it in one line of code (the scaler's inputs include bars after the origin). The control shows $+0.5287$ on noise with an interval from $+0.5113$ to $+0.5476$. The test block shows $+0.579$, the best of the four, and the deployment block shows $-0.727$, with intervals that hold $25.0$ percent of the outcomes. The model has learned to read a number that is not there: with the 30 bars scaled to a mean of 0 and a standard deviation of 1, the six forecast bars are constrained by the 24 context bars, and a linear map finds the constraint. Without it, as in deployment, the map's forecasts are scaled by the wrong numbers and every interval is in the wrong place. A coverage of $79.2$ percent on the test block, which looks perfect, is the reason not to read coverage alone: the head was fitted on the same constraint.

The random split is invisible to the numbers that depend on one linear map. The skill is $0.244$ against $0.248$, the control shows $-0.0032$ with an interval from $-0.0083$ to $+0.0029$, and the deployment block agrees with the test block. The map has 150 weights and the fit has 3440 rows, which is too little capacity to memorize neighbouring windows; a nearest-window forecaster of Chapter 1 does memorize them, and the control of Chapter 4 shows what such a forecaster makes of windows that overlap. The test the split fails is the structural one: 179 of the 195 forecast bars of its test block are forecast bars of fit windows as well. It also fails where something drifts. The raw event probabilities of the model fitted on the random fit block report a calibration error of $0.0168$ on their test block, and $0.0552$ in deployment, more than three times as much; the specified system reports $0.0453$ and has $0.0750$ at deployment, one and two thirds times as much. A random split cannot see the drift because the drift is spread over every block it makes, and a report made with it carries no warning that the world moves.

The recalibration on the test block is the quietest. Its quantile intervals hold exactly $80.0$ percent on the test block and $79.9$ percent in deployment, against $78.8$ and $78.9$ for the specified system, and its skill is $+0.247$ against $+0.248$: five multipliers fitted to 14,400 values cannot overfit them, and the control, $-0.0002$, cannot see it. The damage is in the part of the system with many degrees of freedom. The isotonic repair of the event probability takes 19 distinct values (its fitted levels, not its blocks) when fitted on the test block, and it reports a calibration error of $0.0000$ on the data it was fitted on, which is not what a repair fitted elsewhere reports ($0.0210$ for the specified system). In deployment the same repair has $0.0349$, and the specified system's repair, fitted on the calibration block, reports $0.0210$ and has $0.0515$. The alert cost of the mistaken system in deployment, $0.4509$, is slightly lower than the specified system's $0.4623$, and the reason is not that the mistake helps: the repair fitted on the test block is fitted nearer in time to the deployment block, in a world where the hazard is higher; the first extension tests that explanation. What the mistake took away is the ability to know. A calibration error of exactly zero on a block, an interval that holds exactly its nominal share on 14,400 outcomes, are numbers that fitting makes and no honest evaluation does; the standard error of a share near 0.8 on that many outcomes is $0.0033$, though the outcomes of a window are not independent and the true spread is wider.

**Milestone 8 ends** when each of the three planted builds produces the numbers of its row in both tables and of its row in the control table, and the specified build does not.

### Milestone 9: what the system costs

The last milestone counts the work, in FLOPs and in bytes, from the shapes alone, for the system as specified: milestones 1 to 7, without the planted mistakes, the control and the seasonal AR(1) forecaster. The counting rule is in the block: a multiply-add is two FLOPs, a library call is not counted, the bytes are 8-byte doubles read once per stage. The counts of the learned path are exact formulas in $c$, $H$, the number of levels and the number of rows, and the formulas of the stages are printed so that a reader's own counts can be compared with them.

<!-- worked: project_forecaster.bill -->
Work counted from the shapes alone, for the system as specified (milestones 1 to 7, without the planted mistakes, the control and the seasonal AR(1) forecaster). One FLOP is an addition, subtraction, multiplication, division or square root; a multiply-add is 2; comparisons, loads, stores and library calls (the sine, the logarithm, the Gaussian distribution and its inverse, the sigmoid, the draws of the generator) are not counted. With p = c + 1 = 25 inputs of the map, H = 6 outputs, L = 5 levels and K = 60 test origins: scaling a window is 6c + 4 FLOPs (mean c + 1, variance 3c + 1, the square root, the added 1e-5, and 2c for the scaled bars) and its targets 2H; one Gram update is 2 (p (p + 1) / 2 + p H) = 950; one elimination of order p is 11950 (per pivot column c = 0 .. p - 1 and row below it one division and 2 FLOPs for each of the p - c + 1 entries from the pivot column itself through the right-hand side, which is updated like the rest, then 2 (p - 1 - i) + 2 for row i of the back substitution); the map's output for a row is 2 p H + 2 H = 312; scoring one head on one row is H (1 + 7 L) = 216 (one product d sigma_h per horizon, then per level 2 for the quantile, 4 for the loss and 1 to add it up); one bootstrap resample is 2K + 2 = 122 (two sums of K losses, a division and a subtraction); one Newton step of the event model is 40 FLOPs a row (the logit, the gradient and the 3 x 3 Hessian). Bytes are 8-byte doubles of the arrays a stage reads, each counted once however often the stage passes over it, and arrays that the stage before just wrote and that are still resident are not counted again: scaling reads the c + H = 30 raw bars of each fit window; the Gram stage reads its accumulator, the p x p matrix and the p x H right-hand sides (775 doubles); elimination reads that same matrix and right-hand sides once for all H solves (775 doubles); the forward pass reads the c = 24 scaled context bars of each row of the four blocks (the constant input and the 150 weights are resident); scoring reads nothing new (the rows just passed forward); a bootstrap reads two vectors of K origin losses, the baseline's and the model's; the Newton fit reads the 3 features and the label of each fit row. Rows of the four blocks: 3440, 1240, 2400, 2200.

| stage | count | FLOPs | bytes read | share of FLOPs (%) |
|---|---:|---:|---:|---:|
| scale a fit window and its targets | 3440 x (6c + 4 + 2H) | 550,400 | 825,600 | 3.6 |
| accumulate the Gram matrix and the right-hand sides | 3440 x 2 (p(p+1)/2 + pH) | 3,268,000 | 6,200 | 21.1 |
| eliminate: H solves of order p | 6 x elimination(p = 25) | 71,700 | 6,200 | 0.5 |
| forward pass of the map, all four blocks | 9280 x (2pH + 2H) | 2,895,360 | 1,781,760 | 18.7 |
| score a head on a block: 8 variants on test, 4 on deployment | 28000 x H (1 + 7 L) | 6,048,000 | - | 39.1 |
| four block bootstraps of the skill | 4 x 2000 x (2 K + 2) | 976,000 | 3,840 | 6.3 |
| Newton fit of the event model | 12 x 3440 x 40 | 1,651,200 | 110,080 | 10.7 |
| total |  | 15,460,660 |  | 100.0 |

One fit window is read once, 240 bytes for 1,110 FLOPs, 4.62 FLOPs per byte; the whole panel is 96,000 bytes.
<!-- /worked -->

The fit of the learned model, the scaling, the Gram accumulation and the elimination together, is $25.2$ percent of the $1.55 \times 10^{7}$ FLOPs of the system; scoring the heads on the two blocks is $39.1$ percent and the bootstraps of the skill another $6.3$ (together $45.4$). The system spends more arithmetic judging its forecasts than learning them, so that a change in the number of variants scored changes the bill more than a change in the model. Within the fit the Gram accumulation is $3.27 \times 10^{6}$ FLOPs, 46 times the elimination that solves it ($7.17 \times 10^{4}$): the cost of a ridge fit is in the rows, not in the weights. The accumulation grows with the square of the number of inputs $p = c + 1$ for every row; the elimination grows with its cube and is paid once. One fit window is read once, 240 bytes for 1110 FLOPs, 4.6 FLOPs per byte. The whole panel is 96,000 bytes, and the windows the scaling and the forward pass read add up to $2.6 \times 10^{6}$ bytes, 27 times as many, since each bar of the panel sits in up to 30 windows.

## What the forecaster must report

The forecaster is tested against the values below, in the order of the milestones. 'Exact' means equal integers; a number with decimals is matched to the digits shown, which are the digits on which the run and its four nudged copies agree.

<!-- worked: project_forecaster.acceptance_tests -->
Each row is a number of a block above, collected in the order of the milestones. 'Exact' means equal integers or equal text; a number with decimals is to be matched to the digits shown, which are the digits on which the run and its four nudged copies agree.

| milestone | quantity | value | how to compare |
|---|---|---|---|
| 1 | y at bar 299 of series 0, 39 | 44.661397, 54.987004 | 6 decimals |
| 1 | sum of the 300 values of series 0 | 10431.398759 | 6 decimals |
| 1 | events in all, in series 0, in bars 0 .. 119 | 487, 16, 135 | exact |
| 2 | origins of the fit, calibration, test and deployment blocks | 24..109, 127..157, 175..234, 240..294 | exact |
| 2 | test forecast bars that are fit forecast bars: forward, random | 0, 179 | exact |
| 3 | sigma_h of the seasonal naive head, h = 0 and 5 | 0.6353, 0.6692 | 4 decimals |
| 4 | w0 and w24 of the map's first output | +0.0705, +0.8073 | 4 decimals |
| 4 | sigma_h of the learned head, h = 0 and 5 | 0.3097, 0.5510 | 4 decimals |
| 5 | test loss of the seasonal naive, AR(1) and learned heads (raw) | 0.1867, 0.1577, 0.1325 | 4 decimals |
| 5 | skill of the learned map, raw and recalibrated | +0.290, +0.248 | 3 decimals |
| 5 | 95% interval of the recalibrated skill | [+0.227, +0.278] | 3 decimals; the generator's rows in table order |
| 5 | share inside the central 80 percent: learned map raw, recalibrated | 84.9, 78.8 | 1 decimal, percent |
| 5 | control: skill of the honest system against naive | -0.0002 in [-0.0039, +0.0033] | 4 decimals |
| 5 | control: skill with the future scaler | +0.5287 | 4 decimals |
| 6 | recalibrated multipliers of the learned head at tau = 0.1, 0.5, 0.9 | -1.0098, -0.0501, +1.1222 | 4 decimals |
| 6 | share below the 0.1 and 0.9 quantiles on the test block, raw and recalibrated | 6.5, 10.7; 91.3, 89.5 | 1 decimal, percent |
| 7 | event model weights | -4.2688, +0.9754, +1.7308 | 4 decimals |
| 7 | ECE on the test block, raw and repaired | 0.0453, 0.0210 | 4 decimals |
| 7 | alert cost on the test block: never, always, raw, repaired | 0.6750, 0.8650, 0.5246, 0.4763 | 4 decimals |
| 7 | order cost on the test block, quantile 0.9, raw head: seasonal naive, learned map | 1.4176, 1.0883 | 4 decimals |
| 7 | order cost on the test block, quantile 0.9, recalibrated head: seasonal naive, AR(1), learned map | 1.4143, 1.2245, 1.0855 | 4 decimals |
| 8 | future scaler: skill on the test block and in deployment | +0.579, -0.727 | 3 decimals |
| 8 | future scaler: share inside the central 80 percent in deployment (recalibrated) | 25.0 | 1 decimal, percent |
| 8 | random split: raw event ECE on its test block and in deployment | 0.0168, 0.0552 | 4 decimals |
| 8 | test-block recalibration: share inside the central 80 percent on the test block, in deployment | 80.0, 79.9 | 1 decimal, percent |
| 8 | test-block recalibration: repaired event ECE on the test block, in deployment | 0.0000, 0.0349 | 4 decimals |
| 9 | FLOPs of the system as specified | 15,460,660 | exact |
<!-- /worked -->

Beyond the table an implementation passes when it also meets these, each of which is in a block above: the naive, seasonal naive, AR(1) and learned rows of $\hat\sigma_h$ at all six horizons; the 25 weights of the learned map's first output; the five multipliers of each head; the shares below each quantile on the calibration and the test blocks; the three event weights and the three blocks of the event table; the five alert policies on each block and the eight order rows (three forecasters, raw and recalibrated head, and the median of the learned map) on the test and deployment blocks; the four rows of the control; and the three planted builds each producing the numbers of its row. A system that passes the clean rows and cannot be made to fail the planted ones has tests that cannot fail, and the control column of the opening table is the one that says which of the planted builds your tests can see: if your control passes the random split as it did here, the overlap count of milestone 2 is the test that must not be skipped.

## Where the system's pieces live in brain

The project is small and shaped like the real thing, so that each piece of it corresponds to a crate of the repository. The correspondence is closest in the middle and loosest at the ends, and brain's history records two of the project's own lessons, a control that passed when it had not run and quantile levels served under the wrong label.

**The windows and the split.** `crates/forecast/src/train_data.rs` is the leak-safe split of milestone 2. `enumerate_windows` lists every origin of a series, and `temporal_split` is the rule of Chapter 1 with the gap `cfg.embargo.max(horizon)`, but it assigns a window to a block by the date of its last context bar along one calendar shared by all the series, where the project cuts by the bar index of a panel whose series all start together. It does no scaling itself: each model normalizes a window from its own past bars, never globally, which is the line the project's second mistake crosses. The code arrived with the forecasting subsystem in commit `5ce448795` (21 July 2026) with a test that no forecast horizon leaks across the embargo.

**The baselines, the harness and the control.** `crates/fcbench/src/baselines.rs` holds the forecasters that every learned model is measured against: `RandomWalk` is the project's naive, `SeasonalNaive` its seasonal naive, `Drift`, `Arima` (an autoregression on the differenced series) and `Garch11` are the others, each a `ForecastModel` that returns a Gaussian. Their spread is made from the window itself (`RandomWalk` takes the standard deviation of the differences of the context, grown with the square root of the horizon), so that they need no fit block, where the project fits one $\hat\sigma_h$ per horizon on a block of its own. `crates/fcbench/src/scenarios.rs` has a `RandomWalkScenario` marked as a negative control, and `Comparison::negative_control_violations` in `crates/fcbench/src/harness.rs` returns every model whose paired MASE difference from naive has its whole $99$ percent interval below zero, so that a comparison with a violation fails instead of printing a table. That is the control of milestone 5 decided on an interval, an independent-window or Newey-West standard error of the paired difference where the project uses a block bootstrap. Commit `a7b2d5fb6` (12 November 2026) records the kind of mistake the control is there to catch in the evaluator itself: the report printed a pass for the control when no random-walk scenario had been run, since an empty list of violations reads as no violation, and averaged a model's score over fewer windows than the run had with nothing in the cell to say so.

**The scores.** `crates/forecast/src/metrics.rs` has the pinball loss (`pinball`, `mean_pinball`), the coverage of an interval (`coverage`) and a skill score. Its skill is clamped to $[0, 1]$, so a forecaster worse than its baseline prints $0$; the project's skill keeps its sign, because the sign is what the control column of the opening table is read for. `crates/forecast/src/lib.rs` defines the `Representation` a forecast is stated in (quantiles, samples, distribution parameters, a point path, class probabilities) and `check_quantile_levels`, which refuses a requested level outside the range of the quantiles a model's head emits. Commit `f66ddb876` (12 November 2026) made all three quantile adapters call it, after a request for the 0.05 and 0.95 quantiles of a head that emits 0.1 to 0.9 had come back as the 80 percent band labelled 90. The project's repair, which moves each level by a multiplier, needs the same care: with a head of nine fixed levels the repaired quantile of a level between two of them is an interpolation, and a level beyond them is not available.

**The learned model.** The project's pooled ridge map stands in for the pretrained forecasters of Chapter 3, all of which sit behind the same trait. `crates/timesfm3`, `crates/chronos2` and `crates/fincast` have a `forecaster.rs` that implements `ForecastModel` and stores its forecast as quantiles, and `crates/kronos` implements it with sampled paths from its tokenizer and decoder. Each normalizes its context in its own `preprocess.rs`, from the bars before the origin alone, and none needs the project's `scaled_window`. To build the project on brain one removes milestone 4 and asks a model for the quantiles at the levels of its head, keeps the heads of the baselines, and runs everything else as written: the study, the control, the repair and the decisions are indifferent to where the quantiles come from, and that is the reason the system is built around quantiles.

**The event over a horizon.** The stock-out is a discrete event with no censoring, because every window of the panel is observed in full, and so a logistic regression and an isotonic fit are enough. Where a real series ends in the middle of a window, brain has the machinery for the censored case. `crates/survival/src/recalibration.rs` fits `logit P = a + b logit F` by the inverse probability of censoring weights, with a slope kept only where the data show it differs from one (commit `6d2ad5d50`, 6 October 2026), and `crates/survival/src/venn_abers.rs` gives a probability as an interval whose width says how much calibration data stands behind it. `crates/horizon/src/evaluation.rs` evaluates a saved model at the horizons the caller names, leaves out a code and horizon that has fewer than 30 events and lists it as absent, and resamples whole groups in its bootstrap. The rule of leaving out a number that few events support is one the project does not need at $2400$ rows and about 300 events in the test block, and that a project with a rare event would.

**The decision.** `crates/rlcd/src/cost.rs` is the decision layer of milestone 7. `CostMatrix::binary(cost_fp, cost_fn)` is the matrix of the alert and `bayes_action` its minimizer, which picks the first action on a tie, as the project's `>=` does, and lists the others it tied with; for it the action is an alert exactly when $p \ge c_{\mathrm{fp}}/(c_{\mathrm{fp}} + c_{\mathrm{fn}})$, the threshold the project uses. Commit `05dcd9d04` (22 September 2026) added it with the observation that a search of the repository had found no cost matrix and no expected-loss minimization anywhere, and gated its numbers against a hand-worked example. `crates/rlcd/src/metrics.rs` has the calibration error and the Brier score. Its `ece` takes a confidence and a flag for whether the decision was right and bins the confidence in equal-width bins; nothing in it requires the confidence to be the top probability, so called with the probability of the event as the confidence and a stock-out as the flag, it computes the project's number.

**What a real series would add to the panel.** The project's world is closed: the series are generated, their answers are known, and the one non-stationarity is a level shift of known size. Its model is one linear map and one Gaussian head per horizon; brain's models are pretrained, have a head of fixed levels, and normalize each context in their own way. Its repair is five numbers fitted once; a deployed system would refit as the world drifts. And its control is one panel of random walks, where a real evaluation has the real panel's own history, which is not a random walk and has no control of its own.

## Check your understanding

1. A colleague reads the row of the opening table for the future scaler: "skill $+0.579$ on the test block, better than the specified $+0.248$, and $79.2$ percent inside the 80 percent interval, so just as calibrated; we should keep it". Find the error, and name the check that shows it without a deployment block and the one that shows it with one.
2. The random split leaves the skill at $0.244$, the control at $-0.0032$ with an interval that contains zero, and the deployment skill at $0.244$. A reviewer concludes that the split is harmless for this system. What do the overlap count and the event calibration errors say, and what property of the learned map hides the harm in the skill?
3. A recalibration fitted on the test block reports an interval coverage of $80.0$ percent on it, and in deployment $79.9$. The honest repair reports $78.8$ and has $78.9$. Why is the test-block repair nonetheless a mistake, and why does the quantile repair show no damage where the event repair does?
4. The alert cost of the repaired model is $0.4763$ on the test block and $0.4623$ in the deployment block, lower, while the calibration error of the same probabilities rose from $0.0210$ to $0.0515$. Explain how both can be true, and which numbers of the decisions block to read with it.
5. The cost of a miss is raised so that the order quantile is 0.95. What does the repair need that the five levels of the project do not have, and what does a pretrained head with the levels 0.1 to 0.9 do when asked?
6. The context is lengthened to 48 bars, with everything else unchanged. Predict which rows of the bill change and by roughly how much, and which do not.

## Summary

If every acceptance test passes you have a complete forecasting system on a world whose answers are known: a panel of forty series with a season, a regime change and a stock-out process; blocks cut in time with a purge and an embargo, on which no forecast bar of a later block was ever read by an earlier one; three baselines and one pooled ridge map, each stating its forecast as quantiles; a skill of $0.248$ for the map against seasonal naive with an interval from $0.227$ to $0.278$, and a negative control that shows $-0.0002$ on random walks; a repair of the quantile levels and of the event probability fitted on a window of their own; alerts that cost $0.4763$ a window against $0.6750$ for never alerting, and orders that cost $1.0855$ a row against $1.4176$ for the seasonal naive forecast; three planted mistakes, each measured on the test block and in the 55 origins that follow it; and a bill of $1.55 \times 10^{7}$ FLOPs in which judging the forecasts costs more than learning them.

What the project established, in order. A control on a panel where the truth is zero makes the largest of the mistakes visible with no model of the mistake at all: the scaler that sees the future shows $+0.529$ on noise. A control is not a proof of honesty, because it passed the random split and the recalibration on the test block, whose damage was in a count of shared bars and in a calibration error that is exactly zero where it was fitted; the control, the count of shared bars and the calibration error on fresh data see three different things, and a system that has only one of them can miss two of the three mistakes. A coverage that is nominal on the data a repair was fitted on says nothing, and the five multipliers of a quantile repair are too few to overfit, while the nineteen distinct levels of an isotonic repair are enough. Skill against a raw baseline credits a model with the baseline's miscalibration, and the skill of the specified system dropped from $0.290$ to $0.248$ once both were repaired. A decision scored by the cost it incurred answers questions that a proper score does not: the same quantile forecast costs $70$ percent more when it is used as a point. And what a calibration repair cannot follow, a world whose hazard moves from block to block, it leaves behind: the isotonic repair that took the test block's error from $0.0453$ to $0.0210$ left $0.0515$ in the block after it.

Where this stops. The panel is generated, its regime changes are all of one kind, and the one learned model is linear with a Gaussian head; the event has no censoring, and the costs are given, not estimated from a business. The repair is fitted once. The 55 origins of deployment are one sample of the future and their numbers are one draw of it. Each extension below lifts one of these limits.

## References

- T. Gneiting and A. E. Raftery, *Strictly proper scoring rules, prediction, and estimation*, 2007. The scores of the milestones and why the pinball loss is one.
- T. Gneiting, F. Balabdaoui and A. E. Raftery, *Probabilistic forecasts, calibration and sharpness*, 2007. The probability integral transform and the repair of milestone 6.
- V. Kuleshov, N. Fenner and S. Ermon, *Accurate uncertainties for deep learning using calibrated regression*, 2018. A recalibration of quantile levels from the empirical distribution of the transforms.
- B. Zadrozny and C. Elkan, *Transforming classifier scores into accurate multiclass probability estimates*, 2002. The isotonic repair of a probability.
- H. R. Künsch, *The jackknife and the bootstrap for general stationary observations*, 1989. The block bootstrap of milestone 5.
- R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd edition, 2021. The baselines and the walk-forward evaluation.

## Beyond the generated panel

These are optional. Each changes one thing of the project and asks you to predict the outcome before you run it.

1. **A repair that rolls.** Fit the isotonic repair of the event probability and the multipliers of the quantile levels on the most recent 31 origins before each deployment origin, with the gap of milestone 2. Predict the calibration error of the repaired event probability in deployment, relative to the $0.0515$ of the repair fitted once, and whether the quantile coverage moves.
2. **A conformal width.** Replace the multipliers by the conformal half-width of Chapter 2, fitted on the absolute errors of the learned map on the calibration block, for the central 80 percent interval. Predict the coverage on the test block and the width relative to the Gaussian head's.
3. **A head for each series.** Give each series its own $\hat\sigma_h$ from its fit rows, shrunk towards the pooled one. Predict the effect on the coverage of the series with the largest shocks and on the pooled coverage.
4. **A quantile regression head.** Fit the learned model's quantiles by the pinball loss, one linear map per level, in place of the Gaussian head. Predict how often the five quantiles cross, and what a sort does to the loss.
5. **A refit at every origin.** Fit the ridge map again at each deployment origin on all the windows that end before it, with the purge. Predict the skill in deployment and the cost of the refits in FLOPs from the bill.
6. **A hazard instead of a window.** Fit the discrete hazard of Chapter 5 on person-period rows of $[1, z_t, \mathrm{age}_t]$ and compute the probability of a stock-out in three bars from it, in place of the direct classifier. Predict which one is better calibrated at the fit and which in deployment.
7. **A pretrained model for the map.** Replace milestone 4 with a forecaster of Chapter 3 behind `ForecastModel`, using its own quantile levels. Predict its skill against the ridge map on the test block, and say which part of milestone 6 you can no longer carry out.

## Answers

**Recall questions.**

1. $\mu = (4 + 6 + 8 + 10)/4 = 7$, the deviations are $-3, -1, 1, 3$ with mean square $5$, so $\sigma = \sqrt{5} = 2.2361$ and $d = 2.23607$ with the $10^{-5}$ added. The scaled context is $-1.3416, -0.4472, 0.4472, 1.3416$.
2. The gap is $\max(6, 6) = 6$, the cuts are $\lfloor 240 \cdot 0.5 \rfloor = 120$ and $\lfloor 240 \cdot 0.7 \rfloor = 168$. Origins run from 24 to 234. With $g = o - 1$ the last context bar of a window, validation windows need $g \ge 120 + 6 = 126$, that is $o \ge 127$, and $g + 6 \le 168 - 6$, that is $o \le 157$: the band is 127 to 157, 31 windows. The fit band is 24 to 109 and the holdout (the test block) starts at $g \ge 174$, $o \ge 175$.
3. The cost-minimizing level is $c_{\mathrm{short}}/(c_{\mathrm{short}} + c_{\mathrm{over}}) = 9/10 = 0.9$. The pinball loss at $0.9$ of the order 10 is $0.9 \cdot 2 = 1.8$ when the demand is 12 and $0.1 \cdot 2 = 0.2$ when it is 8; the actual costs are $9 \cdot 2 = 18$ and $1 \cdot 2 = 2$, ten times the pinball loss.
4. Two neighbouring origins share five of their six forecast bars. A bootstrap that resamples single origins treats losses that are strongly dependent as independent, so the effective number of origins is overstated and the interval is too narrow. An honest forecaster has a skill of zero or less against naive on a driftless random walk, since naive is the best forecast there is.
5. The alert is as cheap as no alert when $(1 - p) \cdot 1 = p \cdot 5$, that is $p = 1/6 = 0.1667$. The threshold compares the probability with a cost ratio, so it is right only when the probability is a probability: if it is too low by a third, the alert fires at a true risk that is higher than the threshold intended.
6. $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}')\mathbf{w} = \mathbf{X}^\top\mathbf{y}$ with $\mathbf{I}'$ the identity with a zero in the place of the constant. Two copies of a column make $\mathbf{X}^\top\mathbf{X}$ singular, and the penalty makes the matrix invertible and picks, among the many solutions, the one that splits the weight equally between the copies.

**Check your understanding.**

1. The test block is the block the scaler's leak is in: with the 30 bars scaled together the six forecast bars are constrained by the 24 context bars, and a linear map finds the constraint. The control shows $+0.5287$ on random walks, where the truth is zero, and the structural audit of the scaler's inputs shows that they include the forecast bars; the deployment block, where the scaler cannot see them, shows $-0.727$ and $25.0$ percent coverage. The coverage of $79.2$ percent is no evidence: the head was fitted on the same constraint.
2. The overlap count says 179 of the 195 forecast bars of the test block are forecast bars of fit windows, and the event calibration error of the raw probabilities is $0.0168$ on the random test block against $0.0552$ in deployment, more than three times, where the specified system's rose by a factor of about one and two thirds. The learned map has 150 weights for 3440 rows and cannot memorize neighbouring windows, so the leak does not reach its loss; a nearest-window forecaster of Chapter 1 would have exploited it.
3. The test block is the data the evaluation is made of, and a repair fitted on them is graded on its own exam: the coverage is nominal by construction and reports nothing about the future. The five multipliers of the quantile repair are too few to overfit 14,400 outcomes, so its quantile intervals look the same in deployment. The isotonic repair takes 19 distinct values on 2400 rows, enough to reach a calibration error of $0.0000$ on them, and it has $0.0349$ in deployment.
4. A calibration error is a gap between probability and frequency; a cost depends also on where the threshold falls and on how many events there are. The share of windows with a stock-out rises ($13.5$ to $15.6$ percent), so never alerting costs more ($0.6750$ to $0.7818$), and an alert that catches the extra events is worth more, while the probabilities that make the decision drift further from the frequencies. Read the cost beside the never and always rows of the same block: $0.4623$ against $0.7818$ and $0.8436$ is the saving in deployment, $0.4763$ against $0.6750$ and $0.8650$ on the test block.
5. A cost of 19 per unit short and 1 over puts the optimum at $\tau = 19/20 = 0.95$, a level the repair does not have: it needs a multiplier for $\tau = 0.95$, which is a sixth rank of the probability integral transforms. A head with the levels 0.1 to 0.9 cannot give the 0.95 quantile at all: since commit `f66ddb876` the adapters refuse a level outside the head's range instead of returning the 0.9 quantile under the 0.95 label.
6. The window scaling, the Gram accumulation and the forward pass change: $p$ goes from 25 to 49, a Gram update from 950 to $2(49 \cdot 50/2 + 49 \cdot 6) = 3038$ FLOPs, 3.2 times as many, the scaling from 148 to 292 FLOPs, and the elimination of order 49 from 11,950 FLOPs to 84,378, 7.1 times as many (near the cube of the ratio of the orders); the map's forward pass per row from 312 to 600. The scoring of the heads, the bootstraps and the Newton fit of the event model are untouched, since they see the forecasts and the three event features.

