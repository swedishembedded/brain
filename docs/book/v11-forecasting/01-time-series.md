<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Time Series Fundamentals

*What makes data ordered in time different, and how do we avoid cheating with the future?*

Take two lists of a hundred numbers that have nothing to do with each other. Each is a random walk: it starts at one draw from a standard Gaussian and then adds a fresh draw at every step, so that the value at step $t$ is the value at step $t - 1$ plus noise. The two lists are drawn one after the other from one generator, and no number in the first is used to make any number in the second. Now compute the correlation between them, the quantity of Volume I that says how closely two lists move together, and do it for a thousand such pairs.

<!-- worked: time_series.spurious -->
Generator: the book's SplitMix64, seed 0x72b83e00a61b0635 (8266425288546256437). A normal() call that finds a held value returns it without drawing, and a held value left at the end of a series is carried into the next one (the generator is never reset).
1000 pairs of walks of 100 steps. A walk is s_0 = normal(), then s_t = s_(t-1) + normal() for t = 1 .. 99; the draws of a pair are the 100 steps of the first walk and then the 100 steps of the second, pair after pair, and no walk has any connection to any other. r is the Pearson correlation of the two lists (the covariance over the product of the standard deviations, every sum over the same positions in order); 'steps' means the 99 first differences s_t - s_(t-1) of each walk.

| what is correlated | mean abs r | share abs r > 0.3 | share abs r > 0.5 |
|---|---:|---:|---:|
| the walks themselves | 0.422 | 0.639 | 0.398 |
| their steps | 0.074 | 0.002 | 0.000 |

The first pair has r = -0.070 between the walks and r = 0.064 between their steps. For independent values the standard deviation of r is close to 1 / sqrt(100) = 0.100.
<!-- /worked -->

For the walks themselves the correlation is far from zero. Its absolute value averages $0.422$, it exceeds $0.3$ in $63.9\%$ of the pairs and exceeds $0.5$ in $39.8\%$ of them: two things with no connection at all look strongly related in two pairs out of five. Take the same walks and correlate their steps, the $99$ differences between neighbours, and the effect is gone: the absolute value averages $0.074$ and it exceeds $0.3$ in two pairs in a thousand. The standard deviation of a correlation between independent samples of a hundred values is about $0.1$, and the step correlations behave as that theory says, while the correlations between the walks do not. The first pair drawn, with $r = -0.070$ between its walks, shows that the effect is a tendency and not a rule; the trouble is how often the correlation of two unrelated walks is large.

The reason is that the statistics of Volume I came with an assumption that this experiment breaks. The formula that says a correlation of a hundred pairs is accurate to about $0.1$ treats the pairs as independent draws, and the values of a walk are not independent: each is the one before it plus a small step, so a hundred of them carry the information of far fewer than a hundred independent numbers, and two walks that each wander away from where they began will often wander in the same direction by chance. Yule described this "nonsense correlation" between time series in 1926, and Granger and Newbold showed in 1974 that regressions between unrelated walks reach high fits and impressive test statistics for the same reason. The cure is not a better formula but a different look at the data: the steps of the walks are independent, and the question about them has the answer that theory gives.

That experiment holds the whole subject in small. Data ordered in time cannot be shuffled without losing something, its values depend on each other in a way that has a shape, and every part of the machinery of Volumes I and II that quietly assumed independent samples has to be re-examined: the standard errors, the correlations, and above all the split between the rows a model learns from and the rows it is judged on. The tools for that re-examination are few: the autocorrelation, which reads the memory of a series; differencing, which removes what a model should not have to learn; forecasts built from nothing but the series itself, which set the bar a fitted model has to beat; an autoregressive model fitted by least squares; and a rule for cutting a series into training and test windows so that no future leaks into either.

Each of those tools rebuilds something from Volumes I and II, and five of those ideas are worth having to hand before they are re-examined.

1. (Volume I, Chapter 11, "The Distributions Machine Learning Uses", covariance.) Two quantities have variances $4$ and $9$ and covariance $3$. What is the variance of their sum, and what is their correlation?
2. (Volume II, Chapter 2, "Linear Regression and the First Descent", least squares.) Fit $y = a + bx$ by least squares to the points $(0, 1)$, $(1, 2)$, $(2, 4)$: write the normal equations and solve them.
3. (Volume II, Chapter 5, "Generalization, Overfitting and Honest Evaluation", data leakage.) A daily series is cut into training and test rows by a random shuffle of its days. What goes wrong with the test score, and what rule does a chronological split follow instead?
4. (Volume II, Chapter 8, "Experiments, Baselines and Claims", baselines.) What does a baseline protect a claim from?
5. (Volume I, Chapter 14, "Complex Numbers, Frequencies and the Fourier Transform", the discrete Fourier transform.) A list of $n = 120$ values contains $A\cos(2\pi k t / n + \varphi)$ with $k$ a whole number. In which bin of the transform does it appear, and how does $A$ come out of the bin's value?

## What the order adds

A **time series** is a list of values $y_0, y_1, \dots, y_{n-1}$ taken at equally spaced moments, in which the position of a value is part of the data. Each position is a bar, whatever the spacing is (a day, an hour, a month), and bars are counted from zero. What makes the series different from a table of rows is that the joint distribution of $y_t$ and $y_{t+k}$ is in general not the product of their distributions: the values depend on each other, and the nature of that dependence is what a forecast uses.

A **forecast** is a statement about bars that have not happened yet, made from the bars that have. The moment of the statement is the **origin** $o$, the index of the first bar to be forecast, so that $y_0, \dots, y_{o-1}$ are known and $y_o, y_{o+1}, \dots$ are not; the number of bars ahead is the **horizon**. The formulas below number the forecast bars $h = 0, 1, \dots$ from the origin, so that bar $o + h$ is $h + 1$ bars after the last known one, while a table that speaks of "horizon $h$" counts from one: horizon one is the bar at the origin. The information set at an origin is exactly the past, and every forecast and every statistic computed for one has to be a function of the past alone. That one rule is the whole discipline. Most of what goes wrong with forecasts, every leak described below included, is a statistic or a model that, somewhere, looked at the other side of the origin.

The dependence between values at different bars is measured by the tool of Volume I. For two bars $k$ apart, the covariance $\mathrm{Cov}[y_t, y_{t+k}]$ says how the pair moves together. For it to be one number per $k$ rather than one number per pair of bars $(t, t+k)$, the series must look the same wherever you stand in it. That property has a name and a precise definition, which are easier to give once there is a series to look at.

## A series with a trend, a season and a memory

Most of the examples below use one small series, built from three parts that real series show in combination: a slow drift upward, a regular cycle, and a remainder that is not independent from one bar to the next. It has $n = 120$ values, which a reader can think of as ten years of monthly readings, and it is fully specified by its recipe.

<!-- worked: time_series.sample_series -->
Generator: the book's SplitMix64, seed 0xbfe805a9e24a1b3c (13828308883045227324). A normal() call that finds a held value returns it without drawing, and a held value left at the end of a series is carried into the next one (the generator is never reset).
The series has n = 120 values, t = 0 .. 119: y_t = 10 + 0.05 t + 3 sin(2 pi t / 12) + u_t, where sin is the library sine of the real number 2 pi t / 12 in radians and u_t = 0.7 u_(t-1) + 0.6 normal(), with u_(-1) = 0. The only draws are the 120 calls of normal(), one per t in increasing order. First 14 values:

| t | trend | season | remainder u_t | y_t |
|---|---:|---:|---:|---:|
| 0 | 10.000 | +0.000 | +0.364 | 10.364 |
| 1 | 10.050 | +1.500 | +0.296 | 11.846 |
| 2 | 10.100 | +2.598 | +0.567 | 13.265 |
| 3 | 10.150 | +3.000 | -0.236 | 12.914 |
| 4 | 10.200 | +2.598 | -0.586 | 12.212 |
| 5 | 10.250 | +1.500 | -1.074 | 10.676 |
| 6 | 10.300 | +0.000 | -1.497 | 8.803 |
| 7 | 10.350 | -1.500 | -1.122 | 7.728 |
| 8 | 10.400 | -2.598 | -0.775 | 7.027 |
| 9 | 10.450 | -3.000 | -1.177 | 6.273 |
| 10 | 10.500 | -2.598 | -1.384 | 6.518 |
| 11 | 10.550 | -1.500 | -0.700 | 8.350 |
| 12 | 10.600 | -0.000 | -0.697 | 9.903 |
| 13 | 10.650 | +1.500 | -1.184 | 10.966 |

The mean of all 120 values is 12.733, their standard deviation (divisor n) 2.776, the smallest 6.273 (t = 9) and the largest 18.637 (t = 111); the last value is y_119 = 15.952. Once it has settled (the memory of the start fades), the remainder u_t has the long-run standard deviation 0.6 / sqrt(1 - 0.7^2) = 0.840, derived in the text below.
<!-- /worked -->

The trend column of the table is the **trend**, a line that rises $0.05$ per bar. The season column is a sinusoid with period $12$ and amplitude $3$, a **seasonality**: a pattern that repeats after a fixed number of bars, here $m = 12$. It is the cycle of the year in monthly data, of the day in hourly data (where $m = 24$) and of the week in daily data (where $m = 7$). The remainder column is $u_t = 0.7\,u_{t-1} + 0.6\,\varepsilon_t$ in which each value keeps seventy per cent of the one before it and adds a new Gaussian shock $\varepsilon_t$ (the call `normal()`), so that high values come in runs and low values come in runs. Once it has settled, its variance $v$ is the same at every bar, and because the new shock is independent of the old value, $v = 0.7^2\,v + 0.6^2$, so $v = 0.36/0.51$ and the standard deviation is $0.6/\sqrt{1 - 0.7^2} = 0.840$, which is small beside the amplitude $3$ of the season; the season is what the eye sees, and the remainder is what a forecaster has to be right about.

The series ends at $y_{119} = 15.952$, rising from a start near $10$ with a swing of about $\pm 3$ around the line. From here on the three parts are recovered from the data alone: the season by the autocorrelation and the Fourier transform, the trend by differencing, and the memory of the remainder by an autoregressive model.

## Autocorrelation: how a value remembers its past

Covariance across a gap is only useful once it is made unitless and compared across gaps. The **autocorrelation** at **lag** $k$ is the correlation of the series with itself shifted by $k$ bars. With $\bar y$ the mean of all $n$ values, the sample **autocovariance** at lag $k$ is

$$\gamma_k = \frac{1}{n} \sum_{t=0}^{n-1-k} (y_t - \bar y)(y_{t+k} - \bar y), \qquad r_k = \frac{\gamma_k}{\gamma_0}.$$

Two conventions are fixed here, because a printed digit depends on them. The mean is the mean of all $n$ values for every lag, and the divisor is $n$ for every lag, not $n - k$: the sum has $n - k$ terms, but dividing by $n$ keeps the sample autocovariances those of a valid covariance sequence and shrinks the noisy high lags towards zero, and it is the convention of the book's pseudocode. Then $r_0 = 1$ by definition and $|r_k| \le 1$.

```text
function autocorrelation(x: array<real>[n], kmax: int) -> array<real>[kmax + 1]:
    require n > kmax and kmax >= 0, "need more values than lags"
    xbar = mean(x)
    c = zeros(real, [kmax + 1])
    for k in 0 ..= kmax:
        for t in 0 .. n - k:
            c[k] = c[k] + (x[t] - xbar) * (x[t + k] - xbar)
        c[k] = c[k] / n
    require c[0] > 0, "a constant list has no autocorrelation"
    return c / c[0]
```

How large must $r_k$ be to mean anything? For a list of $n$ independent values of mean zero and one common variance, called white noise here and in the rest of the volume (a list with no memory at all), $r_k$ for $k \ge 1$ has a mean near zero and a standard deviation near $1/\sqrt{n}$ (the standard error of Volume I, Chapter 12, applied to a correlation of $n$ independent pairs), so about $95\%$ of the $r_k$ of such a list lie within $\pm 1.96/\sqrt{n}$. Values outside that band are evidence of memory. Because one looks at many lags at once, one lag in twenty will leave the band by chance (the many comparisons of Volume I again), so a single mild excursion at a high lag says nothing and a pattern of several does.

<!-- worked: time_series.acf_table -->
The series of the first table. r_k = c_k / c_0, where c_k = (1/n) sum over t = 0 .. n-1-k of (x_t - xbar)(x_(t+k) - xbar), xbar is the mean of all n values of the list and the divisor is n for every lag. 'first difference' is x_t = y_t - y_(t-1) for t = 1 .. 119 (n = 119); 'seasonal difference' is x_t = y_t - y_(t-12) for t = 12 .. 119 (n = 108). For white noise (independent values of mean zero and one common variance) r_k has standard deviation about 1/sqrt(n), so the 95 percent band is 1.96/sqrt(n): 0.179 for n = 120, 0.180 for n = 119 and 0.189 for n = 108.

| lag k | y | first difference | seasonal difference |
|---|---:|---:|---:|
| 1 | +0.89 | +0.61 | +0.69 |
| 2 | +0.68 | +0.34 | +0.48 |
| 3 | +0.40 | +0.00 | +0.31 |
| 4 | +0.13 | -0.39 | +0.13 |
| 5 | -0.08 | -0.68 | -0.07 |
| 6 | -0.16 | -0.74 | -0.11 |
| 7 | -0.10 | -0.61 | -0.14 |
| 8 | +0.06 | -0.35 | -0.20 |
| 9 | +0.28 | +0.01 | -0.26 |
| 10 | +0.51 | +0.35 | -0.27 |
| 11 | +0.67 | +0.63 | -0.23 |
| 12 | +0.72 | +0.64 | -0.41 |
| 13 | +0.65 | +0.54 | -0.33 |
| 14 | +0.49 | +0.32 | -0.20 |
<!-- /worked -->

The $y$ column of the table is the series itself. Its autocorrelation starts at $0.89$ at lag 1, which is the memory of a smooth curve: neighbouring values are nearly equal. It falls, crosses zero near lag 5, reaches $-0.16$ at lag 6, which is half a season, where the cycle has moved from a peak to a trough, and climbs back to a maximum of $0.72$ at lag 12, one full season. That return of a large positive correlation at a lag of $m$ and again, if the series were long enough, at $2m$, is how autocorrelation shows a season: the series resembles itself shifted by one period. The first difference, the next column, which turns the trend into a constant, shows the same oscillation more purely, with $-0.74$ at lag 6 and $+0.64$ at lag 12.

The last column is what remains after a seasonal difference, $y_t - y_{t-12}$, which subtracts from each value the one a season earlier. The return at lag 12 is gone and so is the oscillation: the correlation falls steadily from $0.69$ at lag 1 to $0.13$ at lag 4 and then drifts slightly negative. What is left is the memory of the remainder $u_t$, whose decay is the geometric $0.7, 0.49, 0.34, \dots$ of its recipe, blurred by the subtraction. The $-0.41$ at lag 12 is a trace of the differencing itself: $z_t = u_t - u_{t-12}$ and $z_{t-12} = u_{t-12} - u_{t-24}$ share the term $u_{t-12}$ with opposite signs. A reader who sees a negative spike at the seasonal lag after seasonal differencing should expect it, and not read it as a second season.

![Autocorrelation of the sample series at lags 1 to 14, before and after a seasonal difference, with the 95 per cent band of white noise as dashed lines at plus and minus 0.179. The first curve falls from 0.89 to a trough near lag 6 and rises to 0.72 at lag 12; the second falls from 0.69 and has no return at lag 12.](img/acf-lags.svg)

### The same season read by the Fourier transform

The autocorrelation found the period by its repetition. The discrete Fourier transform of Volume I finds it by projection onto cosines and sines, and the two must agree. Remove the trend by a least-squares line (the normal equations of Volume II) and transform the residuals; a cosine that completes a whole number $k$ of turns in the $n$ bars appears in bin $k$ with $|X[k]| = An/2$, so $2|X[k]|/n$ is its amplitude $A$, and the period is $n/k$ bars.

<!-- worked: time_series.spectrum -->
The series of the first table. The least-squares line through (t, y_t), t = 0 .. 119 (normal equations, solved with partial pivoting) is 9.817 + 0.0490 t; its residuals are transformed by the discrete Fourier transform X[k] = sum over t of x_t exp(-2 pi i k t / 120), and the table lists the five largest amplitudes 2 |X[k]| / 120 among the bins k = 1 .. 59, largest first (the library cosine and sine, sums in order of t).

| bin k | period 120 / k (bars) | amplitude |
|---|---:|---:|
| 10 | 12.00 | 2.900 |
| 6 | 20.00 | 0.493 |
| 7 | 17.14 | 0.364 |
| 5 | 24.00 | 0.334 |
| 2 | 60.00 | 0.318 |
<!-- /worked -->

The largest amplitude is in bin $10$, a period of $120/10 = 12$ bars, and it is $2.900$, close to the $3$ that the recipe put in and a little off it because the remainder and the fitted line also contribute. The next amplitude, $0.493$ in bin 6, is a sixth of that: the slow wandering of the remainder shows up as a little energy in low bins, none of it a clear period. Because $n = 120$ is a whole number of seasons, the season lands exactly on a bin; for a series whose length is not a multiple of the period it spreads over neighbouring bins, the spectral leakage of Volume I, and the autocorrelation then reads the period more cleanly than the spectrum does. The two tools see the same thing, and either will do.

## Stationarity, and what differencing removes

The autocorrelation $r_k$ summarises the dependence at lag $k$ by one number for the whole series. That summary is meaningful only if the dependence at lag $k$ is the same in every part of the series. A series is (weakly) **stationary** if its mean does not depend on $t$, its variance does not depend on $t$, and the covariance between $y_t$ and $y_{t+k}$ depends only on $k$. Its statistical character is then the same wherever one stands in it, which is exactly what lets the past of a series say something about its future: if the mean climbed without limit, no stretch of the past would be a sample of the future's mean.

The sample series is not stationary, for a plain reason: it rises, and its season repeats. A random walk is not stationary either, because its variance grows with $t$, which is what the opening experiment turned on. The practical remedy has the same shape in both cases: replace the series by changes in it. The **first difference** is $\nabla y_t = y_t - y_{t-1}$, which removes a linear trend (the line $0.05\,t$ becomes the constant $0.05$) and turns a random walk into its independent steps. The **seasonal difference** of period $m$ is $\nabla_m y_t = y_t - y_{t-m}$, which removes any pattern that repeats exactly every $m$ bars and turns the trend into the constant $0.05\,m$. A difference shortens the list, by one value and by $m$ values respectively; the two can be combined, and a trend that is not a line needs a second difference.

<!-- worked: time_series.stationarity -->
The series of the first table and three differences of it. Each list of n values is cut after its first floor(n/2) values (the first part; the second part is the remaining n - floor(n/2) values); the mean and the variance (divisor: the number of values of the part, around the part's own mean) of each part are printed, and the autocorrelation at lag 1 of the whole list as in the autocorrelation table.

| series | n | mean, first part | mean, second part | variance, first part | variance, second part | r_1 |
|---|---:|---:|---:|---:|---:|---:|
| y | 120 | 11.218 | 14.249 | 5.996 | 4.819 | +0.89 |
| first difference | 119 | 0.013 | 0.081 | 1.688 | 1.326 | +0.61 |
| seasonal difference | 108 | 0.574 | 0.729 | 1.629 | 1.299 | +0.69 |
| seasonal difference of the first difference | 107 | 0.017 | 0.053 | 1.125 | 0.557 | -0.17 |
<!-- /worked -->

Read the table by cutting each list in two and comparing the halves. The series itself has a mean of $11.218$ in its first half and $14.249$ in its second, a rise that a stationary series would not show. The first difference has halves with means $0.013$ and $0.081$, both near the trend's $0.05$ and close to each other. The seasonal difference has means $0.574$ and $0.729$, near the $0.6 = 12 \times 0.05$ that the recipe predicts, and variances $1.629$ and $1.299$. The variances of the first two differences differ between halves by a quarter, and those of the last row by a factor of two. Neighbouring values are correlated, so a variance over sixty of them is a less steady estimate than sixty independent values would give, and the reader should not hunt for a trend in such differences: comparing halves cannot settle a close call, and formal tests of stationarity exist for series where it matters. What it can do is rule out the large violation, a series whose level moves by more than its own spread. The series fails that test and its differences pass it.

Differencing is not free. A difference amplifies fast noise and removes information at low frequency, and a series that was already stationary becomes worse for it: the last row, the seasonal difference of the first difference, has a first-lag autocorrelation of $-0.17$, and a negative value there is the usual sign of a series that has been differenced once too often. The rule of thumb that follows is to difference as little as makes the series look the same in its halves, and to read the autocorrelation afterwards for the signs of overdoing it.

## Forecasts that need no model

Before fitting anything, a forecaster should know what the series offers for free, because a model that cannot beat these is not worth fitting. The sample series is cut at bar $96$: the first $96$ values are the past, and the last $24$ are held out as the future to be forecast, all from the past alone. Every forecast below is a few lines of arithmetic.

The **naive** forecast repeats the last value, $\hat y_{o+h} = y_{o-1}$, which is the best point forecast of a random walk. The **drift** forecast extends the line through the first and last bars of the past, $\hat y_{o+h} = y_{o-1} + (h + 1)\,\delta$ with $\delta = (y_{o-1} - y_0)/(o - 1)$ the average change per bar. The **moving average** repeats the mean of the last $w$ values (here $w = 12$, one season). The **seasonal naive** forecast repeats the last whole season, $\hat y_{o+h} = y_{o - m + (h \bmod m)}$, and adding the drift to it, $\hat y_{o+h} = y_{o-m+(h \bmod m)} + \delta\,m\,(\lfloor h/m \rfloor + 1)$, shifts each repeated season up by the trend accumulated since the season it copies.

<!-- worked: time_series.baselines -->
The series of the first table. The past is y_0 .. y_95, the 24 values to forecast are y_96 .. y_119; h = 0 .. 23 numbers the forecast bars and every forecast is made from the past only. naive: y_95. drift: y_95 + (h + 1) delta with delta = (y_95 - y_0) / 95. moving average: the mean of the last 12 values of the past, repeated. seasonal naive: y_(84 + (h mod 12)). seasonal naive with drift: that value plus delta * 12 * (floor(h / 12) + 1). The last two rows are defined in the next two sections. MAE and RMSE are over the 24 forecast bars (divisor 24). The last column is the MAE averaged over 200 further series drawn by the same recipe from one SplitMix64 generator (seed 0xc1ab4c4f1389c8c7 (13955331772851538119)), each with its 120 draws in turn, cut and forecast in the same way.

| forecast | MAE | RMSE | mean MAE over 200 series |
|---|---:|---:|---:|
| naive | 2.301 | 2.820 | 2.522 |
| drift | 2.188 | 2.638 | 2.393 |
| moving average, 12 bars | 2.035 | 2.372 | 1.999 |
| seasonal naive | 1.211 | 1.421 | 1.203 |
| seasonal naive with drift | 1.012 | 1.166 | 0.986 |
| exponential smoothing, best alpha | 2.301 | 2.820 | 2.522 |
| AR(1) on seasonal differences | 1.029 | 1.188 | 0.976 |
<!-- /worked -->

The table scores each on the $24$ held-out bars by the mean absolute error (MAE) and the root mean square error (RMSE) of Volume II, and, to show that the one holdout is not luck, on $200$ further series from the same recipe. The ranking is clear on the one series and on average. The naive forecast has an MAE of $2.301$ here and $2.522$ on average. The moving average of a whole season, which averages out the season and so removes the very thing it ought to forecast, is only a little better, $2.035$ and $1.999$. Seasonal naive halves the error, $1.211$ and $1.203$, and adding the drift takes off about a sixth more, $1.012$ and $0.986$. Drift alone, $2.188$ and $2.393$, is better than naive for a series that rises, and nowhere near what a model of the season gives. The lesson of the table is the ordering: the largest gain is in modelling the season, the next in the trend, and nothing that ignores the season gets near.

![The last 60 bars of the sample series and, for the final 24, four forecasts made from the first 96: naive and moving average as flat dashed lines, seasonal naive with drift and the AR(1) model on seasonal differences as curves that follow the season with a smaller swing than the series.](img/holdout-forecasts.svg)

## Exponential smoothing, and why its best weight is one here

The moving average gives equal weight to the last $w$ values and none to older ones, an abrupt rule. **Exponential smoothing** keeps a running level $\ell_t$ that moves a fraction $\alpha \in (0, 1]$ of the way towards each new value:

$$\ell_0 = y_0, \qquad \ell_t = \alpha\,y_t + (1 - \alpha)\,\ell_{t-1} \quad (t \ge 1),$$

and forecasts every future bar with the last level, $\hat y_{o+h} = \ell_{o-1}$. Unrolling the recursion shows the weights it puts on the past: $\ell_t = \sum_{j=0}^{t-1} \alpha (1 - \alpha)^j\,y_{t-j} + (1 - \alpha)^t\,\ell_0$, a geometric decay whose last term vanishes as $t$ grows. For $\alpha = 0.3$ the weights on $y_t, y_{t-1}, \dots$ are $0.3000, 0.2100, 0.1470, 0.1029, 0.0720, \dots$, and they sum to one in the limit. A large $\alpha$ forgets quickly, a small one slowly, and $\alpha = 1$ keeps only the last value and is the naive forecast.

How is $\alpha$ chosen? By least squares on the past, the way the weights of Volume II were: run the recursion over the past and score the one-step error $y_t - \ell_{t-1}$ at each bar, summing its square over $t = 1, \dots, o-1$. One parameter on a grid needs no calculus.

<!-- worked: time_series.ses_search -->
The past y_0 .. y_95 of the first table. l_0 = y_0 and l_t = alpha y_t + (1 - alpha) l_(t-1) for t = 1 .. 95; the one-step error at t is y_t - l_(t-1) and the SSE is the sum of its squares over t = 1 .. 95. alpha runs over 0.05, 0.10, ..., 1.00 and the smallest SSE wins (the lower alpha on a tie). The forecast is l_(95) repeated 24 times; its MAE is over y_96 .. y_119.

| alpha | one-step SSE | level l_95 | MAE of the flat forecast |
|---|---:|---:|---:|
| 0.05 | 513.1 | 13.432 | 2.379 |
| 0.25 | 438.7 | 13.628 | 2.278 |
| 0.50 | 302.6 | 13.250 | 2.501 |
| 0.75 | 196.9 | 13.341 | 2.440 |
| 0.95 | 148.0 | 13.528 | 2.326 |
| 1.00 | 139.3 | 13.576 | 2.301 |

The winner of the grid is alpha = 1.00. For alpha = 0.3 the weights alpha (1 - alpha)^j on y_(t-j), j = 0 .. 4, are 0.3000, 0.2100, 0.1470, 0.1029, 0.0720.
<!-- /worked -->

The sum of squared one-step errors falls all the way from $513.1$ at $\alpha = 0.05$ to $139.3$ at $\alpha = 1.00$, so the grid chooses its largest value, and exponential smoothing on this series is the naive forecast, with the naive forecast's MAE of $2.301$. The reason is the season. A level that smooths over many bars averages across the cycle and lags a series that swings by $\pm 3$ in six bars; the best one-step guess is the last value. The holdout column is not monotone and, among the rows shown, would have preferred $\alpha = 0.25$ (MAE $2.278$), but the difference from the chosen rule is a few hundredths against an error of two, and the criterion that was legitimately allowed to look at the past is the one to trust. A smoother is not a model of a seasonal series; it earns its place on series without strong cycles, and as a smoother of the level it is the building block of the seasonal variants that add a season to it.

## An autoregressive model fitted by least squares

The remainder $u_t$ of the recipe has a memory: a high value is followed by another high value. A model that uses it forecasts the next value as a weighted sum of the last few, plus a constant. An **autoregressive model** of order $p$, written AR($p$), says

$$z_t = \phi_0 + \phi_1 z_{t-1} + \phi_2 z_{t-2} + \dots + \phi_p z_{t-p} + e_t,$$

where $e_t$ is an error to be made small. It is a linear regression of the series on its own past, so Volume II's least squares applies without change: each bar $t = p, \dots, n-1$ of a list $z_0, \dots, z_{n-1}$ is one row, with the target $z_t$ and the features $1, z_{t-1}, \dots, z_{t-p}$, and the normal equations $A\boldsymbol\phi = \mathbf b$ built from those rows give the coefficients.

The model also needs a stationary series, which is what differencing provides. The model is not fitted to $y$ but to its seasonal difference $z_t = y_t - y_{t-12}$, which has a constant mean and the short memory of the remainder. A forecast of $z$ is then turned back into one of $y$ by adding it to the value one season before, which is a value of the past for the first season of the horizon and an earlier forecast for the second.

```text
function fit_ar(z: array<real>[n], p: int) -> array<real>[p + 1]:
    require p >= 1 and n - p >= p + 1, "fewer rows than coefficients"
    k = p + 1
    A = zeros(real, [k, k])
    b = zeros(real, [k])
    for t in p .. n:                      # one row per target z[t]
        row = zeros(real, [k])
        row[0] = 1.0
        for j in 1 ..= p:
            row[j] = z[t - j]
        for i in 0 .. k:
            b[i] = b[i] + row[i] * z[t]
            for j in 0 .. k:
                A[i, j] = A[i, j] + row[i] * row[j]
    return solve(A, b, true)

function seasonal_ar_forecast(y: array<real>[n], m: int, p: int, H: int) -> array<real>[H]:
    require n > m, "less than one season of data"
    z = zeros(real, [n - m + H])
    ys = zeros(real, [n + H])
    ys[0:n] = y
    for t in m .. n:
        z[t - m] = y[t] - y[t - m]
    coef = fit_ar(z[0:n - m], p)
    for h in 0 .. H:
        zf = coef[0]
        for j in 1 ..= p:
            zf = zf + coef[j] * z[n - m + h - j]
        z[n - m + h] = zf
        ys[n + h] = ys[n + h - m] + zf
    return ys[n:n + H]
```

In `fit_ar` the entries of $A$ and $\mathbf b$ are sums over the rows in increasing $t$, so that a reader's digits agree with the table. In `seasonal_ar_forecast` the list `z` first holds the seasonal differences of the past and then, one cell per step, each forecast difference, which is what the next step reads back; the loop over `h` runs $H$ times and writes cells $n - m$ to $n - m + H - 1$ of `z` and $n$ to $n + H - 1$ of `ys`.

<!-- worked: time_series.ar_models -->
The past of the first table. z_t = y_t - y_(t-12) for t = 12 .. 95 (84 values, indexed 0 .. 83 below). For each order p the coefficients (phi_0, phi_1, ..., phi_p) minimise the sum of squares of z_t - phi_0 - sum_j phi_j z_(t-j) over t = p .. 83 (normal equations, solved with partial pivoting; the sums run over t in increasing order). 'training RMS' is the root mean square of those residuals (divisor: their number). The forecast rolls the fitted recursion forward 24 bars from the last p values of z, with each forecast z fed back, and adds each to the y of 12 bars before (a forecast y once that bar is itself in the future). MAE over y_96 .. y_119.

| order p | phi_0, phi_1 .. phi_p | training RMS | holdout MAE |
|---|---:|---:|---:|
| 1 | +0.270, +0.635 | 0.873 | 1.029 |
| 2 | +0.274, +0.589, +0.056 | 0.871 | 1.041 |
| 3 | +0.267, +0.595, +0.059, -0.005 | 0.876 | 1.036 |
| 4 | +0.284, +0.590, +0.080, +0.090, -0.155 | 0.867 | 1.025 |
<!-- /worked -->

The first model, AR(1), has $z_t \approx 0.270 + 0.635\,z_{t-1}$. Its constant is not zero because the seasonal difference of a trend of $0.05$ per bar is $0.6$ per bar of $z$, and a stationary $z$ with coefficient $0.635$ has a mean of $0.270/(1 - 0.635) = 0.74$, a little above $0.6$ because the fit is on a small sample; the coefficient $0.635$ is near the lag-1 autocorrelation of the seasonal difference, $0.69$, as it should be for a first-order model. Raising the order changes the training error very little ($0.873$, $0.871$, $0.876$, $0.867$, which is not even monotone because every order drops one more row from the front of the fit) and the holdout MAE hardly at all ($1.029, 1.041, 1.036, 1.025$): the extra coefficients are small ($0.056$, $-0.005$ and so on) and fitted to noise. The more complex model does not buy skill, and the order that a reader would choose from the table is $1$, for the reason given in Volume II, Chapter 5: a larger model fits the past more closely by construction, and that says nothing about the future.

The model's value is not seen in the total error. The single holdout MAE of $1.029$ is a little above that of seasonal naive with drift ($1.012$), and on the $200$ series the model averages $0.976$ and seasonal naive with drift $0.986$: a tie. The difference is in when each is right.

<!-- worked: time_series.ar_horizons -->
The 200 series of the baseline table (seed 0xc1ab4c4f1389c8c7), each cut at 96. For the horizon h (1 = the first forecast bar, y_96) the table gives the mean over the series of the absolute error |forecast - y| at that one bar.

| horizon h (bars) | seasonal naive with drift | AR(1) on seasonal differences |
|---|---:|---:|
| 1 | 0.877 | 0.648 |
| 2 | 0.855 | 0.768 |
| 3 | 0.897 | 0.886 |
| 6 | 1.054 | 1.092 |
| 12 | 1.022 | 0.988 |
| 24 | 1.104 | 1.046 |
<!-- /worked -->

At horizon $1$ the autoregressive forecast is wrong by $0.648$ on average and the seasonal naive with drift by $0.877$, a gain of a quarter: the first forecast bar still remembers the last value of the remainder, because $0.7$ of it survives one step. At horizon $3$ the two are level ($0.886$ and $0.897$), because $0.7^3 \approx 0.34$ of the memory is left and the model's recursion has decayed most of the way to its mean. Beyond that the two stay close: the autoregressive forecast is behind at horizon 6 ($1.092$ against $1.054$) and slightly ahead at the long horizons ($0.988$ against $1.022$ at 12 and $1.046$ against $1.104$ at 24). A plausible reason for the long-horizon edge is that the fitted constant estimates the trend of the seasonal difference from all the past, where the drift uses two bars; the table does not test it. A memory of a remainder pays at the short horizons only, which is where the value of a model of the dependence between bars lies; for horizons beyond the memory, forecasting is about the season and the trend.

## Windows, and the splits that do not leak

So far the series was cut once, at bar $96$, and everything before the cut was the past. A model that learns from many examples, rather than fitting one series, needs a way to make a great many (input, target) examples from one series, and the way it is done decides whether its evaluation can be believed.

A **window** takes $c$ consecutive bars as the context and the next $h$ bars as the target, and a window is named by its origin $o$, so that its context is bars $o - c, \dots, o - 1$ and its target bars $o, \dots, o + h - 1$. A series of $n$ bars has windows at the origins $o = c, c + 1, \dots, n - h$, which is $n - h - c + 1$ windows: with $n = 120$, $c = 24$ and $h = 12$ it is $85$. Neighbouring windows share almost everything: two windows one bar apart have contexts that overlap in all but one bar and targets that overlap in all but one bar, and the context of the later one contains the first target bar of the earlier. That overlap is the new danger. In Volume II, Chapter 5, a model that interpolates between neighbours was fooled by a random split of the bars. A random split of windows is worse, because a held-out window has near copies of itself among the training windows.

A **leakage-safe window** split is one in which no bar that a model is judged on, in the target of a held-out window, has been used by the fitting: not as a target and not as a context. Two rules produce one. First, assign each window to a side by its origin, and never by its bars, so that a window belongs wholly to the past or wholly to the future of a cut. Second, keep the targets of training windows on their side of the cut, by discarding (purging) the windows whose targets would cross it, and put an **embargo** of a further gap of bars between the sides, because a remainder with memory, such as the $u_t$ of the recipe, links bars near the cut even when no bar is shared. The rule below, restated for a single series of $n$ bars, is also the one brain's `forecast` crate uses: a window's position is the index $g = o - 1$ of its last context bar, the cuts are the fractions $f_{\mathrm{train}}$ and $f_{\mathrm{train}} + f_{\mathrm{val}}$ of the length, and the gap is the larger of the requested embargo and the horizon, so that no caller can ask for less than a purge.

```text
function temporal_split(n: int, c: int, h: int, f_train: real, f_val: real, embargo: int)
        -> (list<int>, list<int>, list<int>):
    require c >= 1 and h >= 1 and n >= c + h, "series shorter than one window"
    require embargo >= 0 and f_train > 0.0 and f_val >= 0.0 and f_train + f_val < 1.0, "bad fractions"
    gap = max(embargo, h)                         # never below the purge minimum
    train_end = floor(n * f_train)
    val_end = floor(n * (f_train + f_val))
    train = []
    val = []
    holdout = []
    for o in c ..= n - h:                         # every window with a full context and target
        g = o - 1                                 # index of the last context bar
        if g + h <= train_end - gap:
            append o to train
        else if g >= train_end + gap and g + h <= val_end - gap:
            append o to val
        else if g >= val_end + gap:
            append o to holdout
        # a window in none of the three bands is purged
    return (train, val, holdout)
```

The first test says the last target bar of a training window, at index $g + h$, is at or before index $T - \mathrm{gap}$ with $T$ the cut, so the training targets stay in the past and a further `gap` bars separate them from it. The second and third say the last context bar of a window on the far side of a cut is at index $T + \mathrm{gap}$ or later. A window that falls in the embargo band, or whose target would cross a cut, is in none of the bands and is dropped. With $n = 120$ the cut $\lfloor 120 \cdot 0.7 \rfloor$ is $84$ and no validation set is used.

<!-- worked: time_series.window_policies -->
A series of n = 120 bars, windows of c = 24 bars of context and h = 12 bars of forecast. The origin o of a window is the index of its first forecast bar: its context is bars o - 24 .. o - 1 and its forecast bars o .. o + 11, so the origins are 24 .. 108 (85 windows). The train fraction is 0.7 and the cut is floor(120 * 0.7) = 84; the purge-and-embargo rows follow the rule of the pseudocode above (temporal_split) with no validation set. The random row shuffles the 85 origins (in increasing order before the shuffle; SplitMix64 seed 0xd7f86c756932a746 (15562307763910649670)) and takes the first 38 as training windows and the next 12 as holdout windows. 'holdout label bars' counts the distinct bars among the forecast bars of the holdout windows; the last two columns count how many of them are also a forecast bar of a training window, and how many are a bar of the context or the forecast of a training window.

| split | training windows | holdout windows | holdout label bars | also a training label bar | also a training bar of any kind |
|---|---:|---:|---:|---:|---:|
| at random, the sizes of the third row | 38 | 12 | 86 | 85 | 86 |
| a cut after bar 83, no purge | 61 | 24 | 35 | 11 | 11 |
| purge and embargo, embargo 0 (gap 12) | 38 | 12 | 23 | 0 | 0 |
| purge and embargo, embargo 16 (gap 16) | 34 | 8 | 19 | 0 | 0 |
<!-- /worked -->

![The 85 window origins of the sample series under four splits, one row each: at random (training windows scattered through the origins with holdout windows among them), a cut after bar 83 with no purge, and purge-and-embargo splits with gaps of 12 and 16 bars, in which a block of unused windows lies between the training windows on the left and the holdout windows on the right.](img/window-splits.svg)

Count what each split lets through. The random split has the same $38$ training windows and $12$ holdout windows as the third row, and the $12$ holdout windows contain $86$ distinct target bars, of which $85$ are also target bars of some training window: nearly everything the holdout is scored on was seen as a label in training. The cut after bar $83$ with no purge is a chronological split and looks safe, and it still lets $11$ target bars through: the training windows with origins up to $84$ have targets that run to bar $95$, while the holdout windows begin at origin $85$, so bars $85$ to $95$ are labels on both sides. The two purge-and-embargo rows let nothing through, in targets or in contexts: no target bar of a holdout window is a bar of any training window at all. The price is data: the gap leaves $38$ training windows and $12$ holdout windows from $85$, and a gap of $16$ leaves $34$ and $8$. On a series this short the embargo costs a visible share of the holdout, and a larger gap than the memory of the series requires is waste.

Why are the zeros in the last two rows guaranteed, and what would a longer context change? A training window with last context bar $g$ has its last target bar at index $g + h$, and the rule keeps that at or before $T - \mathrm{gap}$, so every training bar, as context or as target, is no later than $T - \mathrm{gap}$. A held-out window has $g' \ge T + \mathrm{gap}$, so its target bars start no earlier than $T + \mathrm{gap} + 1$, and no held-out target bar is a training bar for any gap. The held-out contexts are a different matter: the first context bar is $g' + 1 - c \ge T + \mathrm{gap} + 1 - c$, which is after every training bar exactly when $T - \mathrm{gap} < T + \mathrm{gap} + 1 - c$, that is when $c \le 2\,\mathrm{gap}$. Here $c = 24 = 2 \cdot 12$: the contexts just fit, and a gap of $16$ has room to spare. With a longer context and no larger gap, the held-out contexts reach back into the training targets. That does no harm to the honesty of the score (a deployed model also reads the recent past as input) and does make the two sets less independent than the table's zeros suggest. The floor at the horizon is a guard against a caller who asks for no separation at all.

### What a random split flatters

The effect on a reported score can be measured. A forecaster that exploits overlap is the simplest one that can: the **nearest window** forecaster stores the training windows and answers a query by the target of the stored window whose context is closest in Euclidean distance. It has no parameters and, with the overlap of a random split, plenty to copy.

<!-- worked: time_series.window_leak -->
Generator: the book's SplitMix64, seed 0x0815e4f1131971a1 (582623450873033121). A normal() call that finds a held value returns it without drawing, and a held value left at the end of a series is carried into the next one (the generator is never reset). 100 series of the recipe of the first table, each 120 calls of normal() in turn, followed by the shuffle of its 85 origins 24 .. 108 (in increasing order before the shuffle). Windows have c = 24 and h = 12. The forward split is the purge-and-embargo rule with embargo 0 and no validation set, so 38 training origins and the holdout origins 97 .. 108; the random split takes the first of the shuffled origins as training windows and the next 12 as holdout windows. The nearest-window forecaster predicts the h forecast bars of a window as the h forecast bars of the training window whose context has the smallest squared Euclidean distance to its own (the earliest origin on a tie); seasonal naive predicts bar o + k as bar o - 12 + (k mod 12). The error of a series is the mean over its holdout windows of the mean absolute error over the 12 forecast bars; the table gives the mean and the sample standard deviation (divisor n - 1) of that error over the 100 series.

| forecaster | random split: mean | sd | forward split: mean | sd |
|---|---:|---:|---:|---:|
| nearest window | 1.217 | 0.158 | 2.306 | 0.610 |
| seasonal naive | 1.038 | 0.141 | 1.000 | 0.343 |
<!-- /worked -->

On a random split the nearest-window forecaster has a mean error of $1.217$, within a fifth of the seasonal naive forecast's $1.038$ on the same holdout windows. On the forward split, which is the one that describes how the forecaster would be used, the same forecaster has an error of $2.306$, and the seasonal naive forecast, which needs no split because it fits nothing, has $1.000$. The random split reported a model competitive with the baseline and the honest split shows it more than twice as bad. The sample standard deviations show that this is not one lucky series: they are $0.158$ and $0.610$ for the nearest window and $0.141$ and $0.343$ for the baseline over the hundred series.

## Scaling each series by what came before it

The nearest-window forecaster did badly on the forward split for a reason other than leakage: the series has a trend, so the context of a holdout window has a mean about two units higher than that of any training window, and the nearest training window is one from the lower end of the climb, whatever its shape. The remedy is to remove the level before comparing: subtract the window's own mean and divide by its own spread. The same step decides how a model reads a series whose scale is not the one it was fitted on, and it is where a second, quieter leak enters.

A **scaler** is a pair $(\mu, \sigma)$ applied as $(x - \mu)/(\sigma + \epsilon)$, with a tiny $\epsilon = 10^{-5}$ so that a constant window does not divide by zero. The question is where $(\mu, \sigma)$ come from. A scaler fitted to all of the data, with the holdout inside it, uses the future. A scaler fitted to the training span, a single pair for the whole series, uses only the past, but does not follow a series that moves. A scaler fitted per window, on the window's own context, uses the past of that window alone and follows the level wherever it goes:

```text
function window_scale(ctx: array<real>[c], eps: real) -> (real, real):
    require c >= 1 and eps >= 0.0, "need a context and a nonnegative eps"
    mu = mean(ctx)
    sd = sqrt(sum((ctx[i] - mu)^2 for i in 0 .. c) / c)   # divisor c, not c - 1
    return (mu, sd + eps)                                 # scaled: (x - mu) / (sd + eps)
```

A scaler is itself a forecast. Take a forecaster that does nothing at all but repeat one number, the number a scaler would map to zero, and see how the choice of number changes its score.

<!-- worked: time_series.scaling_constant -->
The series, forward holdout windows and shuffles of the two previous tables. The forecaster has no model at all: it repeats one number 12 times, and the table says which number. Mean over the 100 series of the mean MAE over the 12 holdout windows, with the sample standard deviation (divisor n - 1).

| the repeated number | mean MAE | sd |
|---|---:|---:|
| mean of all 120 bars (includes the holdout) | 2.687 | 0.269 |
| mean of bars 0 .. 83 (the training span) | 3.336 | 0.378 |
| mean of the 24 context bars of the window | 2.102 | 0.267 |
| last context bar | 2.662 | 0.418 |
<!-- /worked -->

The mean of all $120$ bars, which includes the holdout, gives an error of $2.687$. The mean of the training span only, bars $0$ to $83$, gives $3.336$. The difference, $0.65$, is a gain in skill that is made of nothing but the future entering the scaler: a model standardised with the full-series mean is a better forecaster of the holdout than one standardised honestly, without having learned anything. The mean of the window's own $24$ context bars gives $2.102$, the best of the four and honest, because the series rises and a recent mean is the one close to the future; the last context bar, $2.662$, is the naive forecast.

<!-- worked: time_series.scaling_windows -->
The 100 series, windows, forward split and shuffles of the previous table, forward holdout only. A scaled window is (x - mu) / (sigma + 1e-05) with sigma the standard deviation (divisor n); for the per-window scaler mu and sigma come from the 24 context bars of that window, the distance is taken between scaled contexts and the neighbour's forecast bars are scaled with the neighbour's context statistics and mapped back with the query's (mu_q + (sigma_q + 1e-05) (v - mu_n) / (sigma_n + 1e-05)); for a global scaler the distance is taken between scaled contexts and the neighbour's forecast bars are returned as they are. Mean over the series of the mean MAE over the holdout windows, with the sample standard deviation (divisor n - 1).

| scaler | forecaster | mean MAE | sd |
|---|---|---:|---:|
| none | nearest window, raw values | 2.306 | 0.610 |
| one mean and sd from bars 0 .. 83 of the series | nearest window | 2.306 | 0.610 |
| one mean and sd from all 120 bars of the series | nearest window | 2.306 | 0.610 |
| mean and sd of each window's own 24 context bars | nearest window | 1.094 | 0.311 |
| none | seasonal naive | 1.000 | 0.343 |
<!-- /worked -->

The scalers do not act alike on the nearest-window forecaster. A scaler that is one pair for the whole series, whether from the training span or from all $120$ bars, changes every distance by the same factor and every context by the same shift, so that the nearest window is the same window as without scaling: the three rows are identical, $2.306$ with a spread of $0.610$. A global scaler is invisible to a method that compares windows, and the leak in it has no effect there; it matters for a model that is not invariant to a shift and stretch of its inputs, such as a network with nonlinearities, a penalty on its weights or a clipped input range. The per-window scaler, which maps every context to mean zero and spread one and the neighbour's target by the same two numbers, brings the error to $1.094$ (spread $0.311$), within a tenth of the seasonal naive's $1.000$. The level had been nearly the whole problem.

A single scaler across several series has a third consequence, which a panel of three series shows. Take the same relative pattern at levels $5$, $50$ and $500$ and scale them all by one mean and one standard deviation.

<!-- worked: time_series.scaling_panel -->
Three series of 120 bars, one per level L in 5, 50 and 500, each y_t = L (1 + 0.004 t + 0.25 sin(2 pi t / 12) + v_t) with v_t = 0.7 v_(t-1) + 0.04 normal(), v_(-1) = 0 and the 120 draws of normal() in turn from its own generator (seeds 0x030cdc63c553a94e, 0x2381fd8266fd3c31, 0xc62f7e8de5899c0d for L = 5, 50, 500). Only bars 0 .. 83 are used. The pooled scaler subtracts the mean (218.87) and divides by the standard deviation (270.23, divisor n) of all 252 of these values together; sd(y) and sd(step) are the standard deviations (divisor n) of the series and of its first differences in the units the column header says. 'share' is sd(step)^2 over the sum of the three values of sd(step)^2, the part of a squared-error total that each series contributes when the error is measured in pooled units.

| L | sd(y) raw | sd(y) pooled units | sd(step) pooled units | sd(step) / sd(y), per series | share of squared step error |
|---|---:|---:|---:|---:|---:|
| 5 | 0.971 | 0.0036 | 0.00190 | 0.5294 | 0.000112 |
| 50 | 9.316 | 0.0345 | 0.01781 | 0.5165 | 0.009829 |
| 500 | 95.106 | 0.3519 | 0.17871 | 0.5078 | 0.990059 |
<!-- /worked -->

In pooled units the series at level $5$ varies by $0.0036$ and moves by $0.0019$ from one bar to the next, the one at level $500$ varies by $0.352$ and moves by $0.179$, and every number the model sees from the small series is a rounding error beside the large. The per-series columns are the same for all three, $0.53$, $0.52$ and $0.51$ for the step against the spread: the three series have the same shape. A squared error in pooled units is made of the large series almost entirely: $99.0\%$ of the total comes from the level-$500$ series and $0.011\%$ from the level-$5$ one, so a model trained to minimise it learns the large series and ignores the small. A single global scaler on a panel throws away the information that is the same across series, the shape, in favour of the information that is different, the size, and a per-series or per-window scaler is the remedy that keeps the shape.

## Where this lives in brain

The brain repository once had a crate named for this subject, `timeseries`: six lines describing a "Float-I/O time-series transformer with MSE loss" as a "Placeholder crate" to be "implemented in a later phase", created with a similar placeholder for autodiff in commit `d9bfa70d4` (24 June 2026) and never touched again. The ideas above live in the forecasting crates that came after: the windowing and the split in `forecast`, the baselines in `fcbench`, the scaling in each model's `preprocess.rs`. Commit `8dd2f7a6a` removed the placeholder, so that a reader who goes looking for time series in brain finds the code that does the work.

The leak-safe split is `crates/forecast/src/train_data.rs`, introduced in commit `5ce448795` (21 July 2026) with the weekly fine-tuning pipeline. `enumerate_windows` lists every origin from `context` to `len - horizon`, as the origin count above does; `temporal_split` is the rule of the pseudocode, with the line `cfg.embargo.max(horizon)` for the gap and the assignment of a window to a side by the date of its last context bar along one calendar shared by all the series, where the pseudocode used the bar index of one series. Its test `temporal_split_has_no_horizon_leak_across_the_embargo` asserts that the last target bar of any training window comes strictly before the first context bar of any validation window, for a context of $20$ bars and a gap of $10$: exactly the boundary case $c = 2\,\mathrm{gap}$ of the inequality above, so the test sits exactly on the boundary. The module's header also says what the scaling must do, that it is "computed per window from PAST bars only", and leaves it to each model.

The baselines are `crates/fcbench/src/baselines.rs`, from the same commit: `RandomWalk` is the naive forecast, `SeasonalNaive` repeats the last `season` bars as the formula above does, `Drift` extends the straight line through the first and last values of the context, and `Arima` fits an autoregression by ordinary least squares on the differenced series with the elimination of Volume I (`solve` in `crates/fcbench/src/util.rs`, with partial pivoting), then integrates the forecast back. Where the model above differenced seasonally, brain's `Arima` differences `d` times with the ordinary difference and leaves the season to `SeasonalNaive`. Each baseline also returns a Gaussian with a spread that grows with the horizon, which Chapter 2 takes up. The comments of the file call them "the controls every foundation model must beat before its skill is believed", which is the use made of them here.

The scalers are in the models. `crates/kronos/src/preprocess.rs` has `normalize`, which z-scores each feature column over the context, divisor $n$ and a $10^{-5}$ added to the deviation, then clips to a stated range (the callers pass $5$): the `window_scale` above, written for a window of bars with several features. `crates/timesfm3/src/preprocess.rs` takes the idea further with running statistics. Its `running_stats` scans a series patch by patch and keeps a running count, mean and standard deviation, merged patch by patch with the parallel-variance formula, and each patch is scaled by the statistics from the start of the context through that patch (`revin` treats a deviation below $10^{-6}$ as one rather than dividing by it). A patch is therefore never scaled by anything after it, which is the per-window rule in a causal form that needs no window. That file arrived with the model's whole decoding pipeline in commit `64624e4a7` (1 September 2026).

Reading these files turned up three places that did not say or do what they should, all fixed in commit `2d38d96c5`. In `crates/kronos/src/preprocess.rs`, the module header and the doc comment of `normalize` said the deviation is floored at $10^{-5}$, while the code divides by `std[c] + 1e-5`, an addition; the two agree for a constant feature and differ by at most $10^{-5}$ elsewhere, so only the comments changed. In `crates/fcbench/src/baselines.rs`, `integrate` took one cumulative sum for every difference order of one or more, with a comment calling higher orders "approximated by repeated cumsum": with `Arima { p, d: 2 }` the forecast of second differences was added to the last level as if it were a first difference, which is wrong, not an approximation, though every caller in the repository used `d = 1` or `d = 0`. The commit integrates order by order, each sum anchored at the last value of its own order, with a test that continues $t^2$ from its constant second difference. The same commit corrects a comment that called the interval width a residual scale: it is the standard deviation of the differenced series itself, floored by that of the first differences, which makes this baseline's intervals wider than a residual scale would.

## Check your understanding

1. An analyst reports a correlation of $0.9$ between daily sales of two products over a year and concludes that they are linked. Name the check that would be made before believing it and what a negative result would look like.
2. The autocorrelation of a monthly series has peaks at lags $12$, $24$ and $36$ that decrease slowly. A seasonal difference is taken and the new autocorrelation has a clear negative value at lag $12$. Is there still a season in the data?
3. An analyst standardises a hundred daily series by the mean and standard deviation of each over its whole history, trains on the first $80\%$ of the days and tests on the last $20\%$. What has leaked, how should the scaler have been fitted, and what does a per-window scaler do about the question?
4. A window split uses a context of $48$ bars and a horizon of $6$. The gap is $6$. Is a training bar ever also a context bar of a held-out window, and does that make the evaluation dishonest?
5. Exponential smoothing with $\alpha = 1$ is chosen for a series by the one-step sum of squares. What does this say about the series, and what forecast does it give?
6. Find the error: "The AR(4) model fits the past more closely than the AR(1) model, so it will forecast the holdout better."
7. A random split of the windows of one series gives a model a test error half that of a forward split. Give the mechanism, and what to report instead.
8. A model standardises each window by its own context mean and deviation. Can this scaler leak the future, and if not, what does the model lose against a global scaler?

## Summary

A time series differs from a table of rows in that its order is data. Correlation, variance and the standard errors of Volume I assume independent draws, and applied to series that wander they invent relationships: independent walks correlate at $0.422$ in absolute value on average, and their steps at $0.074$. Autocorrelation measures the memory of a series by lag, its return at a fixed lag identifies a season, and the discrete Fourier transform finds the same period from another side. Stationarity, the same mean, variance and lag covariance everywhere, is what lets the past speak for the future, and differencing removes a trend or a season to get it, at the price of information and the risk of overdoing it.

Forecasts without models, the naive, drift, moving-average and seasonal-naive rules, set the bar: on the chapter's series the season is worth a halving of the error and the trend about a sixth more, and exponential smoothing, which chooses its weight by least squares, picks the naive forecast. An autoregressive model fitted by least squares on seasonal differences ties seasonal naive with drift over a long horizon and beats it clearly only in the first few bars, where the memory of the remainder is still alive. And no number means anything until the windows that make and score it are separated honestly: a random split of windows lets $85$ of $86$ held-out target bars be seen in training and turns a model more than twice as bad as the baseline into one that looks competitive, a cut without a purge lets $11$ bars through, and a purge with an embargo lets none. A scaler is a parameter fitted from data, so it obeys the same rule: a mean taken over the whole series scores better than an honest one without having learned anything.

Everything here gave a forecast as a single number per bar. A forecast of a series with memory is uncertain, and uncertain in a way that grows with the horizon, from the noise at the first bar to the full spread of the series. The next chapter asks how a forecast should state that uncertainty, and how to score a forecast that is a distribution rather than a number.

## References

- G. U. Yule, "Why do we sometimes get nonsense-correlations between time-series?", Journal of the Royal Statistical Society, 1926.
- C. W. J. Granger and P. Newbold, "Spurious regressions in econometrics", Journal of Econometrics, 1974.
- G. E. P. Box and G. M. Jenkins, "Time Series Analysis: Forecasting and Control", 1970.
- R. J. Hyndman and G. Athanasopoulos, "Forecasting: Principles and Practice", 3rd edition, 2021.

## Exercises

1. **[Calculate]** For $n = 5$ and the list $2, 4, 6, 8, 10$, compute $\bar y$, $\gamma_0$, $\gamma_1$, $\gamma_2$ and $r_1$, $r_2$ with the chapter's divisor $n$ and the mean of all five values. Then give $r_1$ and $r_2$ again with the divisor $n - k$ at lag $k$ (and $n$ at lag 0).
2. **[Calculate]** A series has the trend $3 + 0.5\,t$ and a seasonal pattern of period $4$ with the values $+2, 0, -2, 0$, and no noise. Give its first difference for $t = 1, \dots, 8$, its seasonal difference, and the series obtained by taking both.
3. **[Calculate]** For $\alpha = 0.4$ give the first four weights $\alpha(1 - \alpha)^j$ of the unrolled recursion, the weight that $\ell_3$ puts on $\ell_0 = y_0$, and the level $\ell_3$ for the list $10, 12, 11, 15$.
4. **[Calculate]** Fit AR(1) with a constant by least squares to $z = 1, 3, 2, 4, 3, 5$: write the five rows, the normal equations $A\boldsymbol\phi = \mathbf b$, solve them, and give the forecast of the next two values.
5. **[Calculate]** With $n = 200$, $c = 30$, $h = 10$, $f_{\mathrm{train}} = 0.6$, $f_{\mathrm{val}} = 0.2$ and an embargo of $5$, give the gap, the two cuts, the first and last origin of each of the three bands and the number of windows in each. How many of the $161$ windows are purged?
6. **[Calculate]** For the same $n$, $c$ and $h$, what is the smallest gap for which no training bar, context or target, is a context bar of a held-out window? Which of the rules of the chapter would change if one wanted that?
7. **[Calculate]** A window of context $4, 6, 8, 10$ is scaled with $\epsilon = 10^{-5}$. Give $\mu$, $\sigma$ (divisor $c$) and the scaled values, and then the unscaled forecast when a neighbour's target is scaled to $0.5, 1.0$.
8. **[Implement]** Reproduce `time_series.spurious`: the generator, the two walks and the two correlations, from the seed and the draw order the block prints.
9. **[Implement]** Reproduce `time_series.sample_series`, `time_series.acf_table` and `time_series.stationarity` from the recipe, the divisor rule and the halves rule that the blocks state.
10. **[Implement]** Reproduce `time_series.baselines`, `time_series.ses_search` and `time_series.ar_models`, with the normal equations solved by partial pivoting, and check that your AR(1) coefficients match the first row of the AR table.
11. **[Implement]** Reproduce `time_series.window_policies` with the `temporal_split` of the chapter, and then `time_series.window_leak` and `time_series.scaling_windows`, including the shuffle of the origins.
12. **[Investigate]** Rerun `time_series.window_leak` with a context of $12$ instead of $24$ and with a nearest-window forecaster that uses only the last $6$ bars of the context. How do the random and forward errors change, and does the gap between them grow or shrink? Explain it with the overlap of the windows.

## Key terms

- **time series**: a list of values taken at equally spaced moments in which the order is part of the data, so that its values depend on each other and cannot be shuffled. <!-- concept: time-series -->
- **stationarity**: the property that a series has the same mean and variance at every bar and a covariance between two bars that depends only on how far apart they are. <!-- concept: stationarity -->
- **autocorrelation**: the correlation of a series with itself shifted by a number of bars, a measure of its memory at that lag. <!-- concept: autocorrelation -->
- **seasonality**: a pattern that repeats after a fixed number of bars, shown by a return of large autocorrelation at that lag and by a peak in the spectrum. <!-- concept: seasonality -->
- **autoregressive model**: a model that forecasts a value as a constant plus a weighted sum of the values just before it, fitted by least squares. <!-- concept: autoregressive-model -->
- **leakage-safe window**: a (context, target) window assigned to training or evaluation by its origin, with a purge and an embargo so that no bar a model is judged on was used in fitting it. <!-- concept: leakage-safe-window -->

## Answers

**Recall questions.**

1. The variance of a sum is $\mathrm{Var}[A] + \mathrm{Var}[B] + 2\,\mathrm{Cov}[A, B] = 4 + 9 + 6 = 19$. The correlation is the covariance over the product of the standard deviations, $3/(2 \cdot 3) = 0.5$.
2. The sums are $n = 3$, $\sum x = 3$, $\sum x^2 = 5$, $\sum y = 7$, $\sum xy = 0 + 2 + 8 = 10$, so the normal equations are $3a + 3b = 7$ and $3a + 5b = 10$. Subtracting gives $2b = 3$, so $b = 1.5$ and $a = 7/3 - 1.5 = 5/6$.
3. Test days sit between training days, so a model that interpolates, or that was fitted on neighbours of the test day, is judged on days it has effectively seen, and in use there are no later days to lean on. A chronological split trains on the past and tests on the later days, with an embargo of at least the horizon between them when the quantity predicted reaches into the future.
4. A baseline protects a claim from being a claim about nothing: a number such as an error means little until it is compared with what a trivial method gets, and an improvement is only as large as its distance from the best simple method run under the same protocol.
5. It appears in bin $k$ (and in bin $n - k$ as its mirror image), and $A = 2\,|X[k]|/n$, so a cosine of amplitude $A$ has $|X[k]| = An/2$.

**Check your understanding.**

1. Take the first differences (or changes) of both series and correlate those, as in the opening experiment. If the correlation of the changes is near zero, within about $\pm 1.96/\sqrt{n}$, the relation of the levels was the common drift of two series with trends or walks and not a link; a link shows in the changes as well.
2. Yes, the season is still there; it is not what the autocorrelation shows. The seasonal difference removes a repeating pattern, so what remains is the rest of the series, and a negative value at lag $12$ is the trace of the differencing itself ($y_t - y_{t-12}$ and $y_{t-12} - y_{t-24}$ share a term with opposite signs). The season was removed on purpose, and the negative value is not evidence of a second one.
3. The mean and standard deviation of each series cover the whole history, including the last $20\%$ of the days, so the test days entered the scaler: a series that rises has a full-history mean that lies nearer the test days than a training mean would, which is a gain in skill from nothing but the future (the scaler table of the chapter, $2.687$ against $3.336$). The scaler should have been fitted on the training span only and applied unchanged to the test days. A per-window scaler, from each window's own context, cannot see beyond the window's origin and so raises no question.
4. A training window with origin $o$ has target bars $o, \dots, o + 5$, and a held-out window has first context bar $g' + 1 - 48$ with $g' \ge T + 6$ and $T$ the cut; the training targets are at or before $T - 6$, which is after $T + 6 + 1 - 48 = T - 41$. So yes, many training target bars are context bars of held-out windows. It does not make the evaluation dishonest, because no held-out target bar was ever a training bar and a deployed model also reads the recent past as its input; the two sets are only less independent.
5. The best guess of the next value is the last value: whatever level the smoothing averaged over lagged behind the series, which is what a seasonal or trending series does to a level. The forecast is the naive one, the last value repeated.
6. A closer fit to the past is guaranteed for a larger model by construction (it contains the smaller one, apart from the rows the fit loses), and says nothing about the holdout. In the chapter the training error went from $0.873$ to $0.867$ while the holdout MAE went from $1.029$ to $1.025$, a difference far below the spread between series; the evidence for a better forecast has to come from windows the model did not fit.
7. The random split puts, in training, windows that overlap the held-out windows, so that the neighbour copied from training contains most of the held-out target; the score measures the copying. Report the score of a forward split with a purge and an embargo, and the same split's score for the seasonal naive forecast beside it.
8. No: the context is the past of its own window, so nothing from after the origin enters. The model loses the absolute level, the information in how high the series is compared with other times and other series; the window's mean and deviation are removed, so a model can only recover them if they are passed to it as separate inputs.

**Exercises.**

1. $\bar y = 6$. $\gamma_0 = (16 + 4 + 0 + 4 + 16)/5 = 8$. $\gamma_1 = ((-4)(-2) + (-2)(0) + 0 \cdot 2 + 2 \cdot 4)/5 = 16/5 = 3.2$. $\gamma_2 = ((-4)(0) + (-2)(2) + 0 \cdot 4)/5 = -4/5 = -0.8$. So $r_1 = 0.4$ and $r_2 = -0.1$. With the divisor $n - k$, $\gamma_1 = 4$ and $\gamma_2 = -4/3$, so $r_1 = 0.5$ and $r_2 = -1/6$; with that divisor $r_k$ can exceed one in absolute value at high lags, which the divisor $n$ prevents.
2. The series is $t \mapsto 3 + 0.5t + s_{t \bmod 4}$ with $s = (+2, 0, -2, 0)$. The first difference for $t = 1 \dots 8$ is $0.5 + s_t - s_{t-1}$: $-1.5, -1.5, 2.5, 2.5, -1.5, -1.5, 2.5, 2.5$. The seasonal difference of period 4 is the constant $0.5 \cdot 4 = 2$ (for $t = 4 \dots 8$ five values of $2$). Taking both gives $0$ for every $t = 5 \dots 8$ (the first difference of a constant).
3. The weights are $0.4, 0.24, 0.144, 0.0864$, and $\ell_3$ puts the weight $0.6^3 = 0.216$ on $\ell_0$ (the first three weights and that one sum to one). Then $\ell_1 = 0.4 \cdot 12 + 0.6 \cdot 10 = 10.8$, $\ell_2 = 0.4 \cdot 11 + 0.6 \cdot 10.8 = 10.88$ and $\ell_3 = 0.4 \cdot 15 + 0.6 \cdot 10.88 = 12.528$.
4. The rows $(1, z_{t-1} \mid z_t)$ for $t = 1 \dots 5$ are $(1, 1 \mid 3)$, $(1, 3 \mid 2)$, $(1, 2 \mid 4)$, $(1, 4 \mid 3)$, $(1, 3 \mid 5)$. The sums give $A = \begin{pmatrix} 5 & 13 \\ 13 & 39 \end{pmatrix}$ and $\mathbf b = (17, 44)^\top$. The determinant is $5 \cdot 39 - 169 = 26$, so $\phi_0 = (39 \cdot 17 - 13 \cdot 44)/26 = 91/26 = 3.5$ and $\phi_1 = (5 \cdot 44 - 13 \cdot 17)/26 = -1/26 \approx -0.0385$. The first forecast is $3.5 - 5/26 = 3.308$ and the second $3.5 - 3.308/26 = 3.373$: the coefficient is almost zero because this list has no memory beyond its level.
5. The gap is $\max(5, 10) = 10$. The cuts are $\lfloor 200 \cdot 0.6 \rfloor = 120$ and $\lfloor 200 \cdot 0.8 \rfloor = 160$. Origins run from $30$ to $190$ ($161$ windows), $g = o - 1$. Training needs $g + 10 \le 110$, so $g \le 100$, origins $30 \dots 101$ ($72$ windows). Validation needs $g \ge 130$ and $g + 10 \le 150$, so $g \le 140$, origins $131 \dots 141$ ($11$ windows). The holdout needs $g \ge 170$, origins $171 \dots 190$ ($20$ windows). The purged windows are $161 - 72 - 11 - 20 = 58$.
6. The condition of the chapter is $c \le 2\,\mathrm{gap}$, so the smallest gap with no shared context is $15$ for $c = 30$ (and it is at least $h = 10$). The gap line of the pseudocode, `max(embargo, h)`, would change to also take $\lceil c/2 \rceil$.
7. $\mu = 7$, the deviations are $-3, -1, 1, 3$ with mean square $5$, so $\sigma = \sqrt{5} + 10^{-5} = 2.23607$, and the scaled values are $-1.3416, -0.4472, 0.4472, 1.3416$. For the neighbour target $0.5, 1.0$ the forecast is $7 + 2.23607 \cdot 0.5 = 8.118$ and $7 + 2.23607 = 9.236$.
8. to 11. The checks are the tables of the blocks, to the printed digits. 12. A context of $12$ gives $97$ windows instead of $85$. The size of the gap between the random and the forward error follows how much of a target the nearest training window repeats, which a reader finds by counting the bars shared by the two windows, as `time_series.window_policies` does, and the fixed gap $c \le 2\,\mathrm{gap}$ of the chapter is now met with room to spare.
