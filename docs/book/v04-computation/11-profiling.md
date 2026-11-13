<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Measuring Performance Honestly

*Why do most performance numbers lie, and how do we produce ones that do not?*

A colleague reports that their kernel adds two arrays of $2^{26}$ floats in 5.4 microseconds. Count the work before believing anything ([Chapter 3](03-roofline.md)). The addition $c = a + b$ moves 12 bytes per element, 805,306,368 bytes in all. At 646 (GB/s), the measured bandwidth of the model GPU unit of [Chapter 10](10-kernel-selection.md), the bytes alone need 1,246.6 microseconds to arrive. The report is faster than the memory can possibly deliver the data, by a factor of more than 200. Nothing is wrong with the colleague's arithmetic or with their kernel. Their clock was stopped at the wrong moment.

Here is what three clocks read on a model device, a seeded model and not a measurement of any machine. All three time the same program, a dispatch of one kernel, but they stop at different moments: when the call returns, after the host has waited for the device, and from the device's own timestamps. The table prints every parameter of the model (the next section explains each), the seed and the order in which the random numbers are drawn.

<!-- worked: profiling.enqueue_lie -->
The model of this chapter, all parameters labelled and none a measurement. Device: the model GPU unit of Chapter 10, true rates 36,000 (GFLOP/s) and 664 (GB/s), graded against its measured roofline 34,469 (GFLOP/s) and 646 (GB/s). A dispatch submitted alone costs the host t_sub + t_dsp = 4 + 1 = 5 microseconds of enqueue, times (1 + 0.20 u); the wait costs 40 microseconds after the device has finished, times (1 + 0.25 u); the device runs for t_dsp = 1 microsecond plus the roofline time of the kernel at its hidden efficiency (0.85 for the additions c = a + b, 0.55 for the matrix product), times the noise factor of Chapter 10, (1 + 0.04 u0) (1 - 0.25 ln(1 - u2) if u1 < 0.2, else 1), with ln the natural logarithm. Bytes and FLOPs: the addition of N floats does N FLOPs and moves 12 N bytes, the product C[m, n] = A[m, k] B[k, n] in fp32 does 2 m n k FLOPs and moves 4 (m k + k n + m n) bytes.

Generator: the book's SplitMix64, seed 0x1d1ef5ca73c25ed2 (`seed_of` of the name `profiling.enqueue_lie`, defined on the pseudocode page). Draw order: for each kernel in the order of the table, 11 runs; each run draws five uniforms in the order enqueue jitter, wait jitter, u0, u1, u2. Each timer reports the median of the 11 runs. 'Roofline time' is the bound of Chapter 3 on the measured roofline, max(FLOPs / 34,469 (GFLOP/s), bytes / 646 (GB/s)); the fraction is that bound divided by the time read.

| kernel | roofline time (microseconds) | timer | time read (microseconds) | fraction of the measured roofline |
|---|---:|---|---:|---:|
| add, 2^20 floats | 19.48 | enqueue only (clock stopped when the call returns) | 5.21 | 3.741 |
| add, 2^20 floats | 19.48 | enqueue and wait (clock stopped after the wait) | 74.12 | 0.263 |
| add, 2^20 floats | 19.48 | device stamps (the dispatch's own start and end) | 24.03 | 0.811 |
| add, 2^26 floats | 1,246.60 | enqueue only (clock stopped when the call returns) | 5.37 | 232.156 |
| add, 2^26 floats | 1,246.60 | enqueue and wait (clock stopped after the wait) | 1,503.60 | 0.829 |
| add, 2^26 floats | 1,246.60 | device stamps (the dispatch's own start and end) | 1,455.02 | 0.857 |
| matmul, 512 x 2560 x 2048 | 155.75 | enqueue only (clock stopped when the call returns) | 5.49 | 28.373 |
| matmul, 512 x 2560 x 2048 | 155.75 | enqueue and wait (clock stopped after the wait) | 330.69 | 0.471 |
| matmul, 512 x 2560 x 2048 | 155.75 | device stamps (the dispatch's own start and end) | 280.85 | 0.555 |
<!-- /worked -->

Read the "enqueue only" rows first. A dispatch returns as soon as the host has handed the work over ([Chapter 5](05-compute-kernels.md)); a clock stopped then reads about 5 microseconds whatever the kernel does. For the addition of $2^{26}$ floats that is a fraction of 232.156 of the roofline, and for the matrix product 28.373. A fraction above 1 cannot be earned: the roofline time is a lower bound ([Chapter 3](03-roofline.md)). It can only be a sign that the number is not a time of the kernel at all, and it is the same sign that knowledge #6 recorded for a real card, a bandwidth above what the card can physically move, which [Chapter 4](04-gpu.md) told.

The other two timers agree to within 5 percent on the large addition (0.829 and 0.857) and disagree widely on the small one: for $2^{20}$ floats, waiting for the device makes the kernel look 3.1 times slower than its stamps say (0.263 against 0.811 of the roofline). Nothing is a lie in that row either, but the two numbers are answers to two different questions. A timing is a number together with the question it answers.

So the question comes first. Speed is best stated as work (FLOPs and bytes) and the achieved fraction of the device's own roofline, and the time that goes into that fraction has to deserve its place there. Five habits make it do so, each the answer to one way a timing goes wrong:

1. Stop the clock only when the device has finished (the clock of the table above).
2. Keep the start-up transient out of the number.
3. Repeat, and report how much the number moves.
4. When explaining a time, divide it among the kernels, and then check the division against the whole.
5. Decide what to optimize by what the whole pass would gain.

Those habits lean on six ideas from earlier chapters, from the dispatch to seeded randomness.

1. (Chapter 5, the dispatch.) A host program calls `dispatch` for a kernel and reads the host clock the moment the call returns. What has the clock measured?
2. (Chapter 3, the achieved fraction.) A computation moves 100 MB and does far fewer FLOPs than the machine could do in that time. The machine has $\mathrm{BW} = 800$ (GB/s) and the run takes 200 microseconds. What fraction of the roofline is that?
3. (Chapter 8, launch overhead.) With $t_{\mathrm{sub}} = 4$ microseconds for a submission and $t_{\mathrm{dsp}} = 1$ microsecond for starting a dispatch, what do ten dispatches recorded into one submission cost the host and the device to start, and what would they cost submitted one by one?
4. (Chapter 10, noise only lengthens a run.) In the noise model of Chapter 10, which of the mean and the minimum of repeated runs estimates the device's floor, and why does the mean not?
5. (Volume I, Chapter 12, the standard error.) One hundred timings have a sample standard deviation of 0.2 (microseconds). What is the standard error of their mean?
6. (Volume I, Chapter 17, seeded randomness.) Two runs of a simulation use the same generator seed and draw their numbers in the same order. What can be said about their results, and why does a worked example in this book print its seed and its draw order?

## Where the clock stops: device-synchronized timing

A time is only the time of a kernel if the clock starts and stops around the kernel's work, so the first job is to separate the host's time from the device's and to find out what it costs to make the host wait.

A GPU program has two processors that run at different times. The host runs the program: it prepares a dispatch, hands it to the device and goes on to the next line. The device runs the dispatch later, on its own. The host is free again after $t_{\mathrm{enq}}$, the time it takes to hand over one dispatch, which is the sum of the submission and the start of one dispatch from [Chapter 8](08-fusion.md), $t_{\mathrm{enq}} = t_{\mathrm{sub}} + t_{\mathrm{dsp}} = 4 + 1 = 5$ microseconds in the model. The device then runs for $t_{\mathrm{dev}}$. The figure puts both on one time axis for the addition of $2^{20}$ floats, with the widths to scale.

![Two timelines, each with a host lane and a device lane. Top: the host enqueues for 5 microseconds and the clock stops there, while the device kernel (23.3 microseconds) has only just started. Bottom: the host enqueues, is blocked until the device finishes, then spends 40 microseconds noticing, and the clock stops after 68.3 microseconds; the device's own stamps read 23.3.](img/sync-timeline.svg)

A clock on the host can stop at two moments. Stopped when the call returns, it reads $t_{\mathrm{enq}}$, the cost of the host's own bookkeeping, which does not depend on the kernel at all (the first rows of the opening table: 5.21, 5.37 and 5.49 microseconds for three kernels whose device times differ by a factor of 60). Stopped after the host has *waited* for the device, it reads

$$
t_{\mathrm{sync}} = t_{\mathrm{enq}} + t_{\mathrm{dev}} + t_{\mathrm{wait}},
$$

where $t_{\mathrm{wait}}$ is the cost of the wait itself: after the device has written its last result the host must still notice (a fence is signalled, the operating system schedules the waiting thread, the call returns). In the model that is 40 microseconds. **Device-synchronized timing** is timing in which the clock stops only after the device has finished the work being timed, whether by waiting for it (the second reading) or by reading the device's own clock, a timestamp the device writes before and after the work (the third). The pseudocode of the first is a bracket with two waits:

```text
function time_synchronized(work: function(), reps: int) -> array<real>[reps]:
    require reps >= 1
    t = zeros(real, [reps])
    for r in 0 .. reps:
        wait_idle()                  # start from an idle device: nothing earlier is still running
        t0 = now()
        work()                       # enqueues; returns at once
        wait_idle()                  # the clock must not stop before the device has finished
        t[r] = now() - t0
    return t
```

`now` and `wait_idle`, and the two stamp operations below, are defined in `docs/book/pseudocode.md`. The first `wait_idle` is as necessary as the second: without it the device may still be running the previous repetition when the clock starts, and the reading includes work that belongs to someone else.

### What the wait costs

The bracket is honest, and it is not free. Its reading is the kernel's time plus a floor of $t_{\mathrm{enq}} + t_{\mathrm{wait}}$, about 45 microseconds here, whatever the kernel does. The next table takes the same device and times the addition of $N$ floats for sizes from $2^{12}$ to $2^{26}$ with both clocks.

<!-- worked: profiling.wait_cost -->
The addition c = a + b on N floats (N FLOPs, 12 N bytes), hidden efficiency 0.85, the model and the timers of `profiling.enqueue_lie`. Each row: 11 runs, medians. 'Device stamps' read only the device's time; 'enqueue and wait' reads the enqueue, the device and the wait. Generator: the book's SplitMix64, seed 0xb24f53f3c4395fef (`seed_of` of the name `profiling.wait_cost`, defined on the pseudocode page). Draw order: the sizes of the table in order, 11 runs each, five uniforms per run as before.

| N | bytes | device stamps (microseconds) | enqueue and wait (microseconds) | wait reading / stamps | fraction by stamps | fraction by enqueue and wait |
|---|---:|---:|---:|---:|---:|---:|
| 2^12 | 49,152 | 1.11 | 50.66 | 45.60 | 0.068 | 0.002 |
| 2^14 | 196,608 | 1.37 | 52.76 | 38.41 | 0.222 | 0.006 |
| 2^16 | 786,432 | 2.45 | 54.61 | 22.27 | 0.497 | 0.022 |
| 2^18 | 3,145,728 | 6.70 | 56.25 | 8.39 | 0.726 | 0.087 |
| 2^20 | 12,582,912 | 23.84 | 73.88 | 3.10 | 0.817 | 0.264 |
| 2^22 | 50,331,648 | 93.54 | 147.47 | 1.58 | 0.833 | 0.528 |
| 2^24 | 201,326,592 | 364.16 | 414.62 | 1.14 | 0.856 | 0.752 |
| 2^26 | 805,306,368 | 1,460.73 | 1,515.22 | 1.04 | 0.853 | 0.823 |
<!-- /worked -->

At $N = 2^{12}$ the device needs 1.11 microseconds and the bracketed reading says 50.66, forty-five times as much; the fraction of the roofline is 0.068 by the stamps and 0.002 by the bracket. The bracket's floor is not a property of the kernel and it would be the same for any kernel, so the reading says nothing about this one. As the kernel grows the floor shrinks in proportion: the readings agree to 14 percent at $2^{24}$ (1.14) and to 4 percent at $2^{26}$ (1.04). Both are correct measurements of different quantities, and the rule is to know which one a reading is. A bracket reads the cost of *asking for this result and waiting for it*, which is what a program that must read the result pays. A pair of device stamps reads the cost of the *device's work*, which is what a kernel author optimizes.

The same table explains why a benchmark of small kernels should not be a loop of brackets: in a loop that brackets each of $n$ dispatches the floor is paid $n$ times. Two remedies keep the floor out of the number. Time the dispatches of a whole program with *one* bracket (the next sections come back to it), or have the device write its clock between the dispatches and read the stamps afterwards:

```text
function stamped_times(launches: list<Launch>, reps: int) -> array<real>[reps, len(launches)]:
    require reps >= 1 and len(launches) >= 1
    t = zeros(real, [reps, len(launches)])
    for r in 0 .. reps:
        c = begin_commands()
        s = [record_stamp(c)]                       # stamp 0, before the first launch
        for i in 0 .. len(launches):
            L = launches[i]
            record(c, L.k, L.buffers, L.params, L.n_grp)
            append record_stamp(c) to s             # stamp i + 1, after launch i
        submit(c)
        wait_idle()                                 # stamps mean something only once the list has run
        for i in 0 .. len(launches):
            t[r, i] = stamp_span(c, s[i], s[i + 1])
    return t
```

with `Launch` the record of the four arguments of `record`. A list of $n$ launches has $n + 1$ stamps, and launch $i$ costs the span between stamp $i$ and stamp $i + 1$. Because the stamps sit *between* launches, each span contains the start of its own dispatch, $t_{\mathrm{dsp}}$, and the model's device times (the third timer above) include it. One list, one submission and one wait, with the wait outside every span: the floor is paid once for the whole list, and the list is the same single submission that production runs, so the stamps measure the structure that ships ([Chapter 8](08-fusion.md)). A mode that gives every dispatch its own submission changes the execution it is measuring.

Two cautions complete the picture. A device stamp is still a measurement and can be wrong: drivers have been seen to return a timestamp pair that is off by orders of magnitude for one dispatch in dozens, and a profiler must refuse a reading it cannot believe (the section on brain shows where). And a stamp pair belongs to a device that has them: a device without timestamp queries leaves only the bracket, with its floor, and a report must say which one it has.

## Warm-up runs

The first runs of a program are not the runs a user will see. To say how far they mislead, we model the transient, show what it does to a mean, and give a rule that finds where it ends.

The first call of a kernel does work no later call does. Pipelines are built (compiled and specialized for this device), memory is allocated and touched for the first time, caches are cold, and the device's clock has not yet risen to the frequency it will hold ([Chapter 10](10-kernel-selection.md) found the cost of that last one: a roofline probe that ran on an idle device measured a ceiling 0.4 times the real one, and every fraction built on it was wrong). The runs that pay for these things are the **warm-up runs** of a measurement: the leading runs of a series whose cost is dominated by a start-up transient and is not the cost of the steady state.

The model of the next example is two transients added to the floor. The clock ramp multiplies the time of run $i$ by $1 + A e^{-i/\tau}$, with $A = 0.8$ and $\tau = 5$ runs: the first run takes 1.8 times the floor and the excess falls by a factor $e$ every 5 runs. Run 0 also builds its pipelines and pays $C = 20$ floors extra. Everything is multiplied by the noise factor of Chapter 10. Time is in units of the steady floor, so the series is the same on any machine with this shape of start. The first 24 runs of a seeded series of 200:

<!-- worked: profiling.warmup_runs -->
A series of 200 runs of one pass, in units of its steady floor. Run i costs (1 + 0.8 e^(-i / 5)) f_i, plus 20 on run 0 (the pipelines are built on the first call), where e is the base of the natural logarithm, exp the library function, and f_i the noise factor of Chapter 10: (1 + 0.04 u0) (1 - 0.25 ln(1 - u2) if u1 < 0.2, else 1). Generator: the book's SplitMix64, seed 0xbe8eebd02e9899da (`seed_of` of the name `profiling.warmup_series`, defined on the pseudocode page). Draw order: runs 0 to 199, three uniforms each (u0, u1, u2). The floor is estimated as the minimum of runs 100 to 199, 1.0009; a run is 'within tolerance' when it is at most 1.05 times that (1.0509); the warm-up ends at the first run that starts 5 consecutive runs each within tolerance. 'Transient' is the noise-free factor 1 + 0.8 e^(-i / 5) (without the compile cost).

| run i | time (floors) | transient | within tolerance |  |
|---:|---:|---:|---|---|
| 0 | 21.9320 | 1.8000 | no |  |
| 1 | 1.6896 | 1.6550 | no |  |
| 2 | 1.5972 | 1.5363 | no |  |
| 3 | 1.4979 | 1.4390 | no |  |
| 4 | 2.6822 | 1.3595 | no |  |
| 5 | 1.2991 | 1.2943 | no |  |
| 6 | 1.2575 | 1.2410 | no |  |
| 7 | 1.2171 | 1.1973 | no |  |
| 8 | 1.2055 | 1.1615 | no |  |
| 9 | 1.1379 | 1.1322 | no |  |
| 10 | 1.1322 | 1.1083 | no |  |
| 11 | 1.1193 | 1.0886 | no |  |
| 12 | 1.1107 | 1.0726 | no |  |
| 13 | 1.9244 | 1.0594 | no |  |
| 14 | 1.0531 | 1.0486 | no |  |
| 15 | 1.0747 | 1.0398 | no |  |
| 16 | 1.0435 | 1.0326 | yes |  |
| 17 | 1.0626 | 1.0267 | no |  |
| 18 | 1.0452 | 1.0219 | yes |  |
| 19 | 1.0561 | 1.0179 | no |  |
| 20 | 1.0347 | 1.0147 | yes | <- warm-up ends |
| 21 | 1.0426 | 1.0120 | yes |  |
| 22 | 1.0390 | 1.0098 | yes |  |
| 23 | 1.0307 | 1.0080 | yes |  |

The warm-up ends at run 20.
<!-- /worked -->

![The first 60 runs of the seeded series: the dots fall along the dashed noise-free transient from 1.8 toward 1, with occasional spikes above it, and settle around the mean after the warm-up at 1.074; run 0 is off the scale at 21.9; a vertical line marks run 20, where the detection rule ends the warm-up.](img/warmup-series.svg)

The picture shows what the table does: a few runs in which the time falls, a settled region in which it scatters above a floor, and the one very large first run. The settled region is not flat. Its scatter is the noise of Chapter 10, which only lengthens a run. The transient is a different thing, a trend, and the next section's statistics assume there is none.

### Finding where it ends

A rule that finds the end must not be fooled by the noise. A single run that happens to fall near the floor is no evidence that the transient is over, and a single spike in the settled region is no evidence that it is not. The rule used here asks for $k$ runs in a row, each within a tolerance of an estimate of the floor, and takes the minimum of the second half of the series as that estimate (noise only lengthens a run, so the minimum of many is the floor):

```text
function warmup_end(t: array<real>[N], k: int, tol: real) -> int:
    require N >= 2 and 1 <= k <= N // 2 and tol > 0
    ref = min(t[N // 2 : N])                 # the floor, estimated on the second half
    limit = (1 + tol) * ref
    for i in 0 .. N - k + 1:
        if all([t[j] <= limit for j in i .. i + k]):
            return i                         # the first run that starts k runs within tolerance
    return N                                 # the series never settled: it is not a measurement yet
```

The tolerance must exceed the jitter of the noise model, which lengthens a run by up to 4 percent, so $tol = 0.05$; and $k = 5$ makes a lucky early run harmless, since a run inside the transient cannot be within tolerance until $0.8 e^{-i/5} \le 0.05$, which first holds at $i = 14$. On the series of the table the rule ends the warm-up at run 20. It is conservative on purpose: it errs by discarding a few good runs and never by keeping a transient one. The `return N` branch is the rule's honesty clause. A series that never settles does not get a number, and the reader of the report is told so.

### What the transient does to a mean

Now the statistics of the series, computed on all 200 runs and on the 180 after the warm-up:

<!-- worked: profiling.warmup_series -->
The 200 runs of `profiling.warmup_runs` (the same series, the same seed 0xbe8eebd02e9899da). Left: all 200 runs. Right: runs 20 to 199, after the warm-up detected by the rule of the text. Units: floors. The noise factor's expected value is (1 + 0.04 / 2) (1 + 0.2 x 0.25) = 1.071, so a steady mean near that is the right answer.

| statistic | all 200 runs | runs 20 to 199 |
|---|---:|---:|
| mean | 1.2025 | 1.0742 |
| median | 1.0277 | 1.0256 |
| minimum | 1.0002 | 1.0002 |
| standard deviation | 1.4867 | 0.1455 |
| standard error of the mean, s / sqrt(n) | 0.1051 | 0.0108 |
| runs | 200 | 180 |

The mean over all runs is 0.1194 above the mean after the warm-up (a bias of 11.9 percent); the median is 0.0020 above and the minimum is the same.
<!-- /worked -->

The mean over all runs is 11.9 percent above the steady mean. The standard deviation is 1.4867 against 0.1455: a single run, the compile at run 0, accounts for nearly all of it, and the standard error 0.1051 is ten times the standard error after the warm-up, 0.0108. The median, the middle of the sorted runs, barely notices the transient (1.0277 against 1.0256): a few leading runs are a few runs in 200 and the median is defined by the middle of the series. The minimum is untouched. That is a good reason to like the median and the minimum, and not a reason to leave the warm-up in: it works only because the transient is a small share of the runs. With 10 runs and a transient that covers 5 of them the median is a transient run.

The bias of a mean over the first $n$ runs can be derived. The expected time of run $i$ is $\bar f\,(1 + A e^{-i/\tau})$ plus $C$ on run 0, where $\bar f = 1.071$ is the mean of the noise factor (Chapter 10, and the table below). Dividing the mean of $n$ runs by the steady mean $\bar f$ gives

$$
\begin{aligned}
1 + b(n), \qquad b(n) &= \frac{1}{n}\Bigl(A \sum_{i=0}^{n-1} e^{-i/\tau} + \frac{C}{\bar f}\Bigr)\\
&\longrightarrow\; \frac{1}{n}\Bigl(\frac{A}{1 - e^{-1/\tau}} + \frac{C}{\bar f}\Bigr) = \frac{4.413 + 18.674}{n} = \frac{23.09}{n}.
\end{aligned}
$$

<!-- worked: profiling.warmup_bias -->
The same series, the mean of the first n runs against the steady mean 1.0742 (runs 20 to 199). 'Expected bias' is what the warm-up adds to the mean of n runs on average, relative to the steady mean: (0.8 times the sum over i < n of e^(-i / 5), plus 20 / 1.071) / n, where 1.071 is the noise factor's mean; the measured column differs from it by the noise of one series.

| n | mean of the first n runs | measured: relative to the steady mean, minus 1 | expected bias |
|---|---:|---:|---:|
| 5 | 5.8798 | 4.4735 | 4.2928 |
| 10 | 3.5516 | 2.3062 | 2.2490 |
| 20 | 2.3569 | 1.1940 | 1.1503 |
| 50 | 1.5792 | 0.4701 | 0.4617 |
| 100 | 1.3321 | 0.2401 | 0.2309 |
| 200 | 1.2025 | 0.1194 | 0.1154 |
<!-- /worked -->

The bias falls as $1/n$ and the noise of the mean falls as $1/\sqrt{n}$, so a transient cannot be averaged away at the pace noise can. To bring the bias below 1 percent by averaging alone takes $23.09 / 0.01 = 2{,}309$ runs; discarding the 20 runs that the rule found removes it at a cost of 20. That is the argument for detecting the end of the warm-up and not for running long: running long is what one does when one does not know the transient's length. A measurement that must include the start-up (the first request to a server, the first token after loading a model, as brain's cold-start scenario below does) is a legitimate measurement of a different thing, and it reports the cold run as a separate row, never mixed into the steady mean.

## Many runs, one summary, one comparison

A single run is a draw from a distribution, so we must choose what to report about the draws, say how well we know it with the standard error of Volume I, and derive how many runs a given resolution costs.

With the warm-up gone, the remaining runs scatter because of noise that only lengthens them. The noise of Chapter 10 has an analytic mean and standard deviation, and the seeded sample agrees with them:

<!-- worked: profiling.noise_moments -->
The noise factor of Chapter 10, (1 + 0.04 u0) (1 - 0.25 ln(1 - u2) if u1 < 0.2, else 1). Analytic: with a = 1 + 0.04 U and m = 1 + 0.25 X on a spike (X exponential with mean 1, probability 0.2), E[a] = 1.02, Var[a] = 0.04^2 / 12, E[m] = 1.05, E[m^2] = 0.8 + 0.2 (1 + 0.5 + 0.125) = 1.125, and the factor is a m with a and m independent. Generator: the book's SplitMix64, seed 0xa111d2c8dcd41448 (`seed_of` of the name `profiling.noise_moments`, defined on the pseudocode page). Draw order: 100,000 factors, three uniforms each (u0, u1, u2). Sample standard deviation with divisor n - 1.

| quantity | analytic | sample of 100,000 |
|---|---:|---:|
| mean of the factor | 1.0710 | 1.0703 |
| standard deviation of the factor | 0.1535 | 0.1515 |
| relative standard deviation cv = sd / mean | 0.1433 | 0.1416 |
| median of the factor | - | 1.0248 |
| minimum of the 100,000 draws | 1 (the floor) | 1.0000 |
<!-- /worked -->

A single run has a relative standard deviation of $\mathrm{cv} = 0.1433$, and its mean sits 7.1 percent above the floor. Which statistic of the $n$ runs answers which question?

- The **minimum** estimates the floor, the best the device can do. Noise only lengthens a run, so it is the least contaminated draw, and it is the right quantity to rank two kernels by on a device another process shares ([Chapter 10](10-kernel-selection.md), and brain's rule of replaying a program and keeping each dispatch's minimum, below). It is also an optimistic number that no user is promised.
- The **median** is what a typical run costs. It is robust to a spike and to a few leftover transient runs, and it answers "what did most runs take?".
- The **mean** is the right number when the quantity of interest is a total: a thousand requests cost a thousand means, spikes included. It is also the statistic with a standard error, $\mathrm{SE} = s/\sqrt{n}$ (Volume I, Chapter 12), which tells how well $n$ runs pin it down.

A report names the statistic, $n$, and the spread. "1.4 milliseconds" is a number with no meaning. "The minimum of 20 runs after a warm-up of 20, the mean 1.07 times that with a standard error of 1 percent" is a measurement.

### Comparing two variants

The question that matters in practice is whether a change helped. Two variants A and B are timed $n$ times each, and the difference of their means $\Delta = \bar t_A - \bar t_B$ has a standard error, because the two samples are independent and their variances add (Volume I, Chapter 12):

$$
\mathrm{SE}_\Delta = \sqrt{\frac{s_A^2}{n} + \frac{s_B^2}{n}} \;\approx\; \mathrm{cv}\,\bar t\,\sqrt{\frac{2}{n}}.
$$

B is declared faster only when $\Delta$ exceeds twice $\mathrm{SE}_\Delta$, the rule that [Chapter 10](10-kernel-selection.md) used to keep a cache from flipping on noise:

```text
function faster(a: array<real>[n], b: array<real>[n], z: real) -> bool:
    # B is faster than A only when the difference of the means exceeds z standard errors of the difference
    require len(a) >= 2 and len(b) >= 2 and z > 0
    se = sqrt(sd(a)^2 / len(a) + sd(b)^2 / len(b))
    return mean(a) - mean(b) > z * se
```

How many runs does a given resolution cost? Let the gap to resolve be $\delta$ as a fraction of the mean, so that $\Delta = \delta\,\bar t$. The gap is twice its standard error when $\delta\,\bar t = 2\,\mathrm{cv}\,\bar t\sqrt{2/n}$, and solving for $n$,

$$
n_2 = \frac{8\,\mathrm{cv}^2}{\delta^2},
$$

and four times as many ($n_4 = 4 n_2$, which halves the standard error) when the gap is to be four standard errors, the point at which a Gaussian difference passes the 2 SE rule 97.7 times in 100. Resolution is bought with the *square* of the precision: ten times as fine a gap costs a hundred times the runs. The table gives $n_2$ and $n_4$ for four gaps and checks them in simulation:

<!-- worked: profiling.repetitions_needed -->
Two variants, A with floor 1 and B with floor 1 - gap, each timed n times with the noise of Chapter 10 (relative standard deviation cv = 0.143314, printed as 0.1433 in the table above). The runs needed so that the gap equals twice the standard error of the difference, 2 cv sqrt(2 / n) = gap, are n = 8 cv^2 / gap^2 (rounded up), and four times as many when the gap is to equal four standard errors. The last two columns are the share of 300 trials in which the mean of A exceeded the mean of B by more than 2 sqrt(s_A^2 / n + s_B^2 / n), with s the sample standard deviation (divisor n - 1). Generator: the book's SplitMix64, seed 0xbc097d370c9e4db2 (`seed_of` of the name `profiling.repetitions`, defined on the pseudocode page). Draw order: for each gap in the order of the table, for n = n_2 and then n = 4 n_2, for each trial, A's n factors and then B's n factors, three uniforms each; B's run is (1 - gap) times its factor. The 0.005 row is not simulated. Each n_2 is the ceiling of its own formula, and n for four standard errors is 4 n_2 of the same row; the n_2 of the 0.005 row (6,573) is computed from the formula directly, so it is not 4 times the n_2 of the 0.01 row (4 x 1,644 = 6,576, because 1,644 was itself rounded up).

| gap | n (gap = 2 SE) | n (gap = 4 SE) | flagged at n_2 | flagged at 4 n_2 |
|---|---:|---:|---:|---:|
| 0.050 | 66 | 264 | 0.560 | 0.973 |
| 0.020 | 411 | 1,644 | 0.503 | 0.973 |
| 0.010 | 1,644 | 6,576 | 0.513 | 0.973 |
| 0.005 | 6,573 | 26,292 | not simulated | not simulated |
<!-- /worked -->

At $n_2$ the rule flags the gap in about half of the trials (0.560, 0.503, 0.513), as it should, since the true gap equals the threshold, and at $n_4$ in 0.973, the 97.7 percent of the Gaussian argument to within the noise of 300 trials. A 1 percent difference needs about 1,644 runs of each variant to be a coin's worth of evidence and 6,576 to be reliable, and a half-percent difference 26,292. Most claims of a few percent that rest on a handful of runs are inside the noise, and nobody can tell whether they are. The remedy is not always more runs: a quieter device has a smaller $\mathrm{cv}$, and a paired design that runs both variants on the same inputs back to back cancels part of the noise (Volume I, Chapter 12, the paired test).

### The best of many

One more habit looks like thoroughness. "I ran it 100 times and report the best." The table computes what it reports.

<!-- worked: profiling.best_of_k -->
Reporting the best of k runs. The device has floor 1 and the noise of Chapter 10. For each k, 1000 trials of k runs; 'best' is the mean over trials of the minimum of the k runs, 'typical' the mean over trials of the mean of the k runs. Generator: the book's SplitMix64, seed 0xc432865443cd0dd8 (`seed_of` of the name `profiling.best_of_k`, defined on the pseudocode page). Draw order: for each k in the order of the table, each of the 1000 trials in turn, its k factors, three uniforms each.

| k | best of k | typical (mean of k) | typical / best - 1 |
|---|---:|---:|---:|
| 1 | 1.0704 | 1.0704 | 0.0000 |
| 3 | 1.0134 | 1.0697 | 0.0556 |
| 10 | 1.0045 | 1.0718 | 0.0669 |
| 30 | 1.0016 | 1.0712 | 0.0695 |
| 100 | 1.0005 | 1.0707 | 0.0702 |
| 300 | 1.0002 | 1.0713 | 0.0711 |
<!-- /worked -->

The best of $k$ approaches the floor (1.0002 at $k = 300$) while the typical run, the mean of the same $k$ runs, stays at 1.071. A report that quotes the best of 300 quotes a number the device reaches in a small fraction of runs and that a user will essentially never see: the report flatters by 7.1 percent. This is the selection effect of Volume I, Chapter 12 ("the best of twenty") with a stopwatch. The minimum of many runs is the right number to compare two kernels in a shared race (the floor is a property of the kernel), and the wrong number to promise to a user (the mean is). The honest report gives both and says which is which.

## The per-kernel profile

A pass is made of many dispatches, and "make it faster" needs to know which ones to look at. The table that divides the time among them and attaches the work to each makes every row a fraction of the roofline.

Consider one layer of a transformer for 512 tokens, as a model of the kind of pass brain runs: ten dispatches in dependency order, four of them matrix products. For each dispatch the profile needs three things: the counted work (FLOPs and bytes by the rules of [Chapter 3](03-roofline.md)), its device time as the stamps read it, and the roofline time of the same work on the measured roofline. A **per-kernel profile** is the table of those for every dispatch or kernel kind of a pass: its counted FLOPs and bytes, its device time and its share of the pass, and the achieved fraction of the measured roofline, graded against the roof of its own class (bandwidth or compute). In the model pass, with the hidden efficiencies printed in the table's header, the replays' minima are:

<!-- worked: profiling.kernel_profile -->
One layer for m = 512 tokens, model width d = 2048, feed-forward width 6144 = 3 d, fp32. Ten dispatches in dependency order. Counts: rmsnorm 4 m d FLOPs and 4 (2 m d + d) bytes; a matrix product [m, k] times [k, n] 2 m n k FLOPs and 4 (m k + k n + m n) bytes; attention (fused, scores never leave the chip) 4 m^2 d FLOPs and 16 m d bytes; silu_mul 5 m (feed-forward width) FLOPs and 12 m (feed-forward width) bytes; add m d FLOPs and 12 m d bytes. The device is the model unit of `profiling.enqueue_lie`; the hidden efficiencies are rmsnorm 0.55, matmul_qkv 0.62, attention 0.30, matmul_out 0.58, add 0.85, matmul_gateup 0.66, silu_mul 0.80, matmul_down 0.60. A dispatch's time is its device stamp pair (t_dsp = 1 microsecond plus the roofline time on the true rates at the efficiency) times the noise factor of Chapter 10, and the table keeps the minimum over 15 replays of the pass. Generator: the book's SplitMix64, seed 0x005091e793e7185b (`seed_of` of the name `profiling.kernel_profile`, defined on the pseudocode page). Draw order: replay 0 to 14, dispatches 0 to 9 in each, three uniforms (u0, u1, u2). 'Bound' compares the intensity with the measured ridge 34,469 / 646 = 53.36 FLOP per byte; the fraction is the bound on the measured roofline over the time; the share is of the sum of the ten times.

| # | dispatch | FLOPs | bytes | FLOP per byte | bound | device time (microseconds) | share | fraction of the roofline |
|---:|---|---:|---:|---:|---|---:|---:|---:|
| 0 | rmsnorm_1 | 4,194,304 | 8,396,800 | 0.50 | bandwidth | 24.04 | 0.008 | 0.541 |
| 1 | matmul_qkv | 12,884,901,888 | 67,108,864 | 192.00 | compute | 578.86 | 0.204 | 0.646 |
| 2 | attention | 2,147,483,648 | 16,777,216 | 128.00 | compute | 200.24 | 0.070 | 0.311 |
| 3 | matmul_out | 4,294,967,296 | 25,165,824 | 170.67 | compute | 207.20 | 0.073 | 0.601 |
| 4 | add_1 | 1,048,576 | 12,582,912 | 0.08 | bandwidth | 23.32 | 0.008 | 0.835 |
| 5 | rmsnorm_2 | 4,194,304 | 8,396,800 | 0.50 | bandwidth | 24.02 | 0.008 | 0.541 |
| 6 | matmul_gateup | 25,769,803,776 | 130,023,424 | 198.19 | compute | 1,091.11 | 0.384 | 0.685 |
| 7 | silu_mul | 15,728,640 | 37,748,736 | 0.42 | bandwidth | 72.22 | 0.025 | 0.809 |
| 8 | matmul_down | 12,884,901,888 | 67,108,864 | 192.00 | compute | 598.43 | 0.211 | 0.625 |
| 9 | add_2 | 1,048,576 | 12,582,912 | 0.08 | bandwidth | 23.31 | 0.008 | 0.835 |

Sum of the ten device times: 2,842.77 microseconds.
<!-- /worked -->

The bound column compares each dispatch's intensity with the measured ridge $34{,}469 / 646 = 53.36$ FLOP per byte. The four matrix products and the attention are compute-bound with intensities of 128 to 198, and the norms, the additions and the activation are bandwidth-bound with intensities below one. The last column is the useful one:

$$
\mathrm{frac}_i = \frac{t_{\mathrm{bound},i}}{t_i}, \qquad t_{\mathrm{bound},i} = \max\!\Bigl(\frac{N_{\mathrm{flop},i}}{F_{\mathrm{ms}}},\ \frac{N_{\mathrm{byte},i}}{\mathrm{BW}_{\mathrm{ms}}}\Bigr),
$$

the achieved fraction of Chapter 3 with the denominator the measured roofline of Chapter 10. It needs no machine in it. Two kernels that run on different devices are comparable by their fractions, and a fraction can be given a target: brain's profiler holds compute-bound kernels to 0.60 and bandwidth-bound ones to 0.70, and calls a kernel a defect below 0.30 and 0.35. By those bands the model pass has nothing below its defect line, `attention` is the lowest at 0.311, and the products sit at 0.601 to 0.685.

The same ten times, grouped by kernel kind as brain's profiler groups them:

<!-- worked: profiling.profile_by_kind -->
The same ten times grouped by kernel kind, as brain's profiler groups them. The fraction of a kind is the sum of its dispatches' roofline times (on the measured roofline) over the sum of their device times, so a long dispatch weighs more.

| kind | calls | FLOPs | bytes | device time (microseconds) | share | fraction of the roofline |
|---|---:|---:|---:|---:|---:|---:|
| rmsnorm | 2 | 8,388,608 | 16,793,600 | 48.06 | 0.017 | 0.541 |
| matmul | 4 | 55,834,574,848 | 289,406,976 | 2,475.61 | 0.871 | 0.654 |
| attention | 1 | 2,147,483,648 | 16,777,216 | 200.24 | 0.070 | 0.311 |
| add | 2 | 2,097,152 | 25,165,824 | 46.63 | 0.016 | 0.835 |
| silu_mul | 1 | 15,728,640 | 37,748,736 | 72.22 | 0.025 | 0.809 |
<!-- /worked -->

The two ways to rank the rows disagree, and the disagreement is the point of the profile. Ranked by *fraction*, `attention` is the worst kernel of the pass (0.311). Ranked by *share*, it is second among the kinds, at 0.070 of the time, and the matrix products are 0.871 of the stamps' sum. A kernel that is far from its roofline and takes little time is not where the time is. The next sections turn that sentence into arithmetic.

Two rules keep the table honest, and brain's profiler enforces both. A kind with no count of its work (no FLOP or byte formula) reports no fraction at all, not zero: dividing a partial count by a whole time reports a low rate that is false. And a fraction above 1 beyond noise is printed as an error and not as a percentage, for the reason the opening showed.

### The trap: the rows do not add up to the pass

A profile's device times are a property of dispatches, and the quantity a user waits for is a property of the pass. The table's own last line, the sum of the ten times, is not the time of the pass. Three cases show three ways the two differ:

<!-- worked: profiling.sum_vs_pass -->
Per-dispatch device stamps against the time of the pass, noise-free. Rows 1 and 2 use the ten minimum times of `profiling.kernel_profile`. One submission with one wait costs t_sub + sum of the device times + t_wait = 4 + sum + 40 microseconds; a dispatch submitted alone and waited for costs t_enq + its time + t_wait = 5 + time + 40. Row 3: two dispatches with no dependence between them, each using at most half the device's lanes, so they run at the same time (a model assumption); their stamps are 600 and 400 microseconds, and the pass lasts the longer plus the host's 4 + 40.

| case | sum of the device stamps (microseconds) | time of the pass (microseconds) | pass minus sum |
|---|---:|---:|---:|
| ten dependent dispatches, one submission, one wait | 2,842.77 | 2,886.77 | +44.00 |
| the same ten, each submitted alone and waited for | 2,842.77 | 3,292.77 | +450.00 |
| two independent dispatches of 600 and 400, overlapped | 1,000.00 | 644.00 | -356.00 |
<!-- /worked -->

In the first row the pass is *longer* than the sum by 44 microseconds, which is the host: the submission and the wait of the model that no dispatch's stamp contains. In the second the ten dispatches are measured one at a time, each submitted alone and waited for, the way a profiler that has no device stamps must work. The sum of the parts is unchanged at 2,842.77 microseconds but the *measured* time of the pass is 450 microseconds longer, ten floors of 45, and each dispatch's reading would be inflated by the floor: a profile built from those readings ranks small kernels as large ones (the section on brain tells the real case). In the third row two independent dispatches overlap: their stamps sum to 1,000 and the pass takes 644, *less* than the sum, because the device ran them at the same time.

So the sum of per-kernel times can be larger or smaller than the pass, and it is never the pass. The table is for explaining where the device's time goes. Its sum is not a measurement of how long the thing takes, and a fix that makes a row smaller has not made the pass faster until the pass says so.

## The whole pass

The number a user feels is the time from asking to having the result. We measure that number and judge a change by it, and then show a change that improves a row and worsens the pass.

A **whole-pass measurement** is the time of the unit of work a user asks for (a layer, a forward pass, a generated image, a request), taken from the moment the request starts to the moment its result is available to the program, with every dispatch, every wait, every host-side step and every copy that the real use performs inside the bracket, and nothing else. It is measured with a synchronized clock around the entire unit, repeated and summarized as the previous sections describe. Per-kernel profiles come after it and explain it.

Here is why. A change to a single kernel can make that kernel's row smaller and the pass larger. Suppose a variant B of the output projection of the layer runs the same product 15 percent faster by its device stamp, by splitting the inner dimension, and that the split needs three small extra dispatches (to zero the partial results, to reduce them and to scale) and two points at which the host has to read a result before going on, each of which costs a submission and a wait. The model charges B for exactly those:

<!-- worked: profiling.whole_pass_change -->
Changing the output projection (`matmul_out`) of the layer of `profiling.kernel_profile`. Variant B runs the product with the start of the dispatch unchanged (t_dsp = 1 microsecond) and the rest of its time, the work at the hidden efficiency, multiplied by 0.85 (by its device stamp: 1 + 0.85 x (206.70 - 1) = 175.84 against 206.70 microseconds, noise-free) but needs 3 more dispatches of t_dsp + 2 = 3 microseconds each and 2 points where the host must wait for a result, each costing a submission (4) and a wait (40 microseconds, times (1 + 0.25 u)). A pass is timed with a wait at its end: the sum over its dispatches of the noise-free time times the noise factor, plus 4 per submission and one wait per submission. 1200 passes of each variant are drawn; the first 300 are one row and all 1200 the other. The mean is the cost a user pays. Generator: the book's SplitMix64, seed 0xb7dec26213ad7b63 (`seed_of` of the name `profiling.whole_pass`, defined on the pseudocode page). Draw order: the 1200 passes of A, then the 1200 passes of B; in a pass, the dispatches in order (A: the ten of the layer; B: the same with `matmul_out` replaced, then the three extra), three uniforms each (u0, u1, u2), then one uniform for each wait. Standard error s / sqrt(n) with divisor n - 1; the standard error of the difference is sqrt(SE_A^2 + SE_B^2).

| passes per variant | mean A (microseconds) | SE A | mean B (microseconds) | SE B | B minus A | SE of the difference | difference / SE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 300 | 3,110.59 | 15.43 | 3,163.24 | 12.70 | +52.64 | 19.98 | +2.63 |
| 1200 | 3,094.26 | 6.87 | 3,169.74 | 6.37 | +75.47 | 9.37 | +8.05 |

By its stamp the product is 14.9 percent faster in B; the pass is +2.44 percent (1200 passes). The relative standard deviation of one pass of A is cv = 0.0769, so the formula n = 8 cv^2 / gap^2 with gap = 0.0244 asks for 80 passes to bring the gap to twice its standard error.
<!-- /worked -->

By its stamp B's product is 14.9 percent faster, and the pass is 2.44 percent slower. The table shows the evidence at two sample sizes, and the comparison rule of the previous section applies to both: with 300 passes of each, $\Delta$ is +52.64 microseconds, 2.63 standard errors, and the 2 SE rule says "B is slower"; with 1,200 passes it is +75.47 microseconds, 8.05 standard errors. The formula for the number of runs predicted the need: with $\mathrm{cv} = 0.0769$ for one pass, a gap of 0.0244 reaches twice its standard error at 80 passes and four times at about 320, where the expected ratio is near 4. The 300-pass row, at 2.63, drew an estimate of $\Delta$ about one standard error below the 1,200-pass one (the expected ratio at 300 passes is near 3.9): the unlucky side of one draw, which is what a standard error of 19.98 allows. The honest conclusion at 300 passes was "probably slower, not yet pinned down", and 1,200 settled it.

The parameters of the model explain the sign. In expectation B's two extra syncs cost $2 \times (4 + 45) = 98$ microseconds (the wait averages 40 times 1.125), its three small dispatches another $3 \times 3 \times 1.071 = 9.6$, and the product saves $(206.70 - 175.84) \times 1.071 = 33.1$ (the noise factor's mean is 1.071): a net cost of about 74.6 per pass, which the measured +75.47 (standard error 9.37) agrees with. Nothing in the kernel's own row showed the cost. That is why knowledge #21 is stated as a rule: a fix does not count until the whole-pass number moves. [Chapter 8](08-fusion.md) retold the real case, a two-stage reduction that was reverted because 88 more dispatches bought nothing end to end.

The pass is also where the host's work lives. If a pass includes sampling on the CPU, a readback, a tokenizer or a copy over the bus ([Chapter 4](04-gpu.md)), the whole-pass time has terms that no kernel's stamp contains and that a profile of kernels will never show. A pass measured by the user's own bracket includes them and a sum of stamps does not. Comparing the two (the first row of the table) is a cheap check that the bracket contains what it should: a pass much longer than the sum of its stamps is a pass with host time in it.

## Amdahl's law

A profile says where the time is, and does not say how much a pass can gain from improving one part, and so which row deserves attention. Amdahl's law does.

Suppose a pass takes time $t_{\mathrm{pass}}$ and a part of it takes the share $p$ of that time (a number in $[0, 1]$, not a probability). We make that part $g$ times as fast, so its time falls from $p\,t_{\mathrm{pass}}$ to $p\,t_{\mathrm{pass}}/g$, and nothing else changes. The new pass takes

$$
t_{\mathrm{pass}}' = (1 - p)\,t_{\mathrm{pass}} + \frac{p\,t_{\mathrm{pass}}}{g},
$$

and the speedup, the old time over the new, is

$$
A(p, g) = \frac{t_{\mathrm{pass}}}{t_{\mathrm{pass}}'} = \frac{1}{(1 - p) + p/g}.
$$

This is **Amdahl's law**: the speedup of a whole from improving one part depends only on the share of the part and the factor of the improvement. Two consequences follow from the formula without any new idea. As $g$ grows without bound the term $p/g$ vanishes and

$$
A(p, \infty) = \frac{1}{1 - p},
$$

the **ceiling**: even a part that takes no time at all leaves the rest, and no change to that part can pass $1/(1 - p)$. For $p = 0.5$ the ceiling is 2, for 0.9 it is 10, for 0.95 it is 20. And the ceiling is approached slowly. Setting $g = p/(1 - p)$ makes the improved part exactly as long as the unimproved rest, so the pass takes twice the unimprovable time and the speedup is half the ceiling; to get halfway to the ceiling a part with $p = 0.9$ must already be 9 times as fast. With several parts of shares $p_j$ made $g_j$ times faster each, the same derivation gives $A = 1/\sum_j p_j/g_j$, with the shares of the unimproved parts taking $g_j = 1$.

```text
function amdahl(p: real, g: real) -> real:
    require 0 <= p and p <= 1 and g > 0
    return 1 / ((1 - p) + p / g)
```

![Speedup of the whole against the base-2 logarithm of the factor g, for shares p of 0.5, 0.8, 0.9 and 0.95 and for the matrix products of the profiled layer (p = 0.858, dashed): every curve rises slowly and flattens at 1/(1-p). Two marks on the dashed curve, g = 2 at 1.75 and the roofline limit g = 1.53 at 1.42.](img/amdahl-curves.svg)

Apply it to the profile. The table uses the shares of the *pass*, so the host's 44 microseconds of submission and wait are in the denominator (and are a part of their own):

<!-- worked: profiling.amdahl_table -->
The profile of `profiling.kernel_profile` and the pass it makes: 2,886.77 microseconds = 4 (submission) + the sum of the ten device times 2,842.77 + 40 (wait). For each kind, p is its time over the pass; A(2) is the speedup if its time is halved, 1 / (1 - p) the ceiling if it took no time at all, p / (1 - p) the factor g at which the speedup reaches half of the ceiling (the improved part then takes as long as all the rest; none exists for p below one half), 'g to the roofline' the factor by which the kind's time falls if every dispatch reached the measured roofline (the kind's time over the sum of its roofline times), and the last column the speedup of that.

| part | time (microseconds) | p | A(p, 2) | ceiling 1 / (1 - p) | g for half the ceiling | g to the roofline | A at that g |
|---|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm | 48.06 | 0.0166 | 1.0084 | 1.0169 | - | 1.849 | 1.0077 |
| matmul | 2,475.61 | 0.8576 | 1.7507 | 7.0210 | 6.02 | 1.528 | 1.4213 |
| attention | 200.24 | 0.0694 | 1.0359 | 1.0745 | - | 3.214 | 1.0502 |
| add | 46.63 | 0.0162 | 1.0081 | 1.0164 | - | 1.197 | 1.0027 |
| silu_mul | 72.22 | 0.0250 | 1.0127 | 1.0257 | - | 1.236 | 1.0048 |
| submission and wait | 44.00 | 0.0152 | - | 1.0155 | - | - | - |

Every dispatch at the measured roofline and the host's costs unchanged: the pass would take 1,849.54 microseconds, a speedup of 1.5608.
<!-- /worked -->

"Make the matrix multiplication twice as fast" is a request with answer $A(0.8576, 2) = 1.7507$: the pass becomes 1.75 times as fast, and the ceiling for the products is 7.02. The three smaller kernels are nearly irrelevant for time: halving the normalizations gains 0.8 percent, and a perfect attention kernel (to the roofline, a factor 3.214 on that row) gains 5.0 percent, in spite of attention being the lowest fraction of the profile. That is the arithmetic behind the remark of the profile section: *the payoff of a row is its share times how much of it can be removed*.

The last two columns show what the roofline adds. A kernel cannot be made faster than its roofline time, so its $g$ has a ceiling of its own, the reciprocal of its fraction. The products run at 0.654 of the roofline (their time-weighted fraction), so $g \le 1/0.654 = 1.528$ on this device, and "twice as fast" is not available: the best case for the products is $A(0.8576, 1.528) = 1.4213$. Everything at the roofline together, with the host's costs unchanged, would give a pass of 1,849.54 microseconds, a speedup of 1.5608. Amdahl's law bounds the gain by the share, the roofline bounds each part's factor by its distance from the device's limit, and the two together say what is worth doing before anyone writes a line.

A word on what the law does not claim. Its derivation assumes the improved part and the rest do not overlap and that improving the part changes nothing else. A change that adds dispatches, or syncs, as variant B did, adds terms of its own, and the law applies to the time of the whole after the change, not to a hope for one.

## Other ways to lie

Five more ways for a time to go wrong are cheap to demonstrate on the model of the opening, each with a small seeded example and the check that catches it, or the observation that no check can. Each row is a way of reading a time; the last column says whether the roofline test of [Chapter 3](03-roofline.md), "a fraction above 1.05 is a defect of the measurement", catches it.

<!-- worked: profiling.lies_catalogue -->
A catalogue of the ways to read a wrong time, on the model of `profiling.enqueue_lie`. The kernel is the addition c = a + b on 2^26 floats (67,108,864 FLOPs, 805,306,368 bytes, hidden efficiency 0.85), except where the row says otherwise; each row is the median of 11 runs (the best of 100 in the last row). 'Removed' means the kernel body is gone, so the dispatch costs only t_dsp = 1 microsecond on the device; 'cached' costs 3 microseconds on the host (times (1 + 0.20 u)) and no dispatch; 'not yet warmed up' multiplies the work time by 1 / 0.4; the roof probed cold is the arithmetic roof times 0.4. The fraction is the roofline time of the row's own work over the time read; 'caught' means the fraction is above 1.05, the threshold at which brain's profiler prints an error instead of a percentage (Chapter 3). Generator: the book's SplitMix64, seed 0x46cde49cc64a4c0e (`seed_of` of the name `profiling.lies`, defined on the pseudocode page). Draw order: rows in the order of the table; a row of runs draws five uniforms per run (enqueue jitter, wait jitter, u0, u1, u2); the cached row draws one uniform per run; the best-of-100 row draws three per run (u0, u1, u2).

| situation | what the timer saw | time read (microseconds) | fraction of the roofline | caught by the roof test |
|---|---|---:|---:|---|
| honest: production size, device stamps | the device's time | 1,473.12 | 0.846 | no |
| clock stopped at enqueue | the host queueing the work | 5.57 | 223.784 | yes |
| result unused, kernel removed | an empty dispatch and the wait | 53.10 | 23.476 | yes |
| same call on same inputs, result cached | a table lookup on the host | 3.33 | 373.857 | yes |
| benchmark at 2^14 floats, enqueue and wait | the wait, mostly | 51.59 | 0.006 | no |
| production size, device not yet warmed up | the device at 0.4 of its clock | 3,637.21 | 0.343 | no |
| matmul against a roof probed cold | the device's time, against 0.4 of the arithmetic roof | 278.93 | 1.396 | yes |
| best of 100 runs, reported as the result | the luckiest run | 1,429.76 | 0.872 | no |
<!-- /worked -->

**Dead code.** A compiler or a driver removes work whose result is never used. A benchmark that dispatches a kernel and never reads its output may be timing an empty dispatch; the third row reads 53.10 microseconds for the work of 805 MB, a fraction of 23. The defence is to make the result matter (read it back, or fold it into a value the program prints) and to feed the kernel inputs the compiler cannot see at build time: the compute probe of Chapter 10 passes its constants in a parameter block for this reason, because a chain of multiply-adds whose operands are known at compile time can be folded to a constant and reports a spectacular rate.

**A cached result.** A library that memoizes returns the previous result for the same inputs, and a benchmark that calls it repeatedly on one input times the lookup (the fourth row, 3.33 microseconds). The same applies to a pipeline cache, a page cache and a persisted selection ([Chapter 10](10-kernel-selection.md)): the measurement times the state it left behind and not the work. The defence is to vary the input and to say which caches are part of the claim, since a cold-cache number and a warm-cache number are both honest and are different quantities.

**The wrong size.** A kernel timed at one size says little about another: at $2^{14}$ floats the wait-bracketed time is almost all floor, and the fraction is 0.006 (the fifth row) against 0.846 at the size of production. The benchmark that decides is the one with the shapes, data layout and batch size of the real use, and a claim names them.

**A device that has not ramped up.** The sixth row times the production kernel with the clock at 0.4 of its steady value, and reads a fraction of 0.343, a third of what the device does when warm. The seventh does the opposite, as Chapter 10 found: the kernel is timed warm against a roof that was probed cold, and the matrix product comes out at a fraction of 1.396. Clock state, temperature and the power limit are part of the setup. The performance suite of brain records the device's throttle reasons and frequency beside each result for that purpose, and a reading of "unknown" is stored as unknown and never as zero.

**The best of many.** The last row reports the luckiest of 100 runs. It is the one row of the table that the roofline test cannot catch and cannot suspect: 0.872 is a plausible fraction. Only the discipline of reporting the summary and its spread catches it, which is why the earlier sections spent their time on that discipline.

Four rows of the eight are caught by the roof test: the timer that never waited, the dead code, the cached result and the roof probed cold. The honest row is rightly not flagged. The other three (the wrong size, the cold clock, the best of 100) are not, because each of them is *plausible*: a fraction below 1 that nobody can tell from a real result. That asymmetry is the reason for process. A test that only catches impossible numbers leaves the plausible lies to habits: a wait around every timed region, a warm-up that is detected and not assumed, a statistic with its $n$, and a record of the setup. Hoefler and Belli's survey of benchmarking practice in high-performance computing found the same kinds of omission in published results, stated rules for avoiding them, and gave them as a checklist of twelve; the list here is shorter and is the part that applies to one device. Other effects can change a time by more than the optimization being measured while having nothing to do with the program's meaning: the order in which a program is linked and the size of its environment were shown to do it (Mytkowicz and coauthors, 2009). The defence is the same, to vary what should not matter and to report the spread.

## Reporting speed as work

What remains is what to write down, so that a number about speed means the same thing next year on other hardware.

A time is a fact about a run: this program, this input, this device, this driver, this clock state, this day. It tells nothing about a different setup, and it ages with every change of driver and every update of the code around it. The hardware-independent statement of speed is the *work* the computation requires, which does not change with the machine, and the *fraction* of what the device can do that the run achieved, which has the machine divided out:

- the work, counted by a stated rule: FLOPs and bytes, which are as exact as the shapes;
- the device's measured roofline, the two rates and how they were found ([Chapter 10](10-kernel-selection.md));
- the achieved fraction of that roofline, $t_{\mathrm{bound}}/t$, graded against the roof of the kernel's class;
- the way the time was taken: which clock (device stamps or a bracket), the warm-up and how it was found, the statistic and $n$, and the standard error;
- the setup, with a date, in case anyone needs the time itself.

For the matrix product of the opening table that reads: "$5.37 \times 10^9$ FLOPs and 30.4 million bytes; 0.555 of the roofline (34,469 (GFLOP/s), 646 (GB/s), measured on this unit), by device stamps, median of 11 runs after warm-up; model GPU unit of the book, November 2026." The 280.85 microseconds belong in that report only as the dated observation they are. A claim of the form "twice as fast" is a claim about two fractions or about two amounts of work, and a claim of the form "5.4 microseconds" with no work beside it cannot be checked at all, which is the claim of the opening.

The same discipline has a name in brain's repository, and it is mechanical: a gate that fails any bare number beside a performance unit in documentation or in source commentary, unless a reviewer has signed it off with a stated reason. The last section describes what it enforces.

### Nine habits in one list

- **Count first.** Write $N_{\mathrm{flop}}$ and $N_{\mathrm{byte}}$ and the roofline time before timing, and distrust a result above it. A fraction above 1 is the first thing to check, never a result.
- **Bracket every timed region with a wait, or use the device's stamps.** State which.
- **Detect the warm-up.** Keep cold runs as their own row.
- **Repeat enough.** $n = 8\,\mathrm{cv}^2/\delta^2$ runs resolve a gap $\delta$ at two standard errors, and four times as many make it reliable.
- **Compare by the difference over its standard error,** not by the order of two means.
- **Profile to explain; decide with the pass.**
- **Choose the part by its share times its distance to the roofline,** not by how bad its fraction looks.
- **Time what the user runs:** the sizes, the data, the caches and the clock state of the real use.
- **Report work and fraction; write times as dated observations.**

## Profilers, the performance suite and the number gate in brain

The performance code of brain is in three places with three jobs. `crates/gpu-core/src/profile.rs` times kernels and passes on the device. `crates/perf` measures what a served engine delivers over time and compares runs. `crates/trace` records what a run did, as events. The first is the per-kernel profile above, the second its statistics and its refusal to compare what cannot be compared, the third a tool for finding where a run went when a table of times is not enough.

**The profile: `crates/gpu-core/src/profile.rs`.** `PassProfile` carries both numbers above side by side: `total_secs`, one submit of the whole step list, "the number a change is judged by", and `summed_secs`, the sum of the per-group times, with the header's rule "rank with the table, decide with the pass". `best_of` is the bracket of the first section: it submits once to warm up, then for each repetition takes the clock, submits, calls `poll_wait` (the host-side `wait_idle`) and takes the clock again, keeping the minimum; its comment records why, a bare-submit loop timing the host (knowledge #6, told in [Chapter 4](04-gpu.md)). `profile` prefers device time: it switches the backend's kernel timing on, runs one timed submit of the whole pass, and reads per-kernel totals, with `device_timed` recording which path was taken; only a backend without timestamps falls back to timing contiguous groups with a bracket each, which the table then labels as inflated. `profile_live` is for a pass that submits itself: one unmeasured call first, "a warm-up must never enter the statistics", then `reps` timed calls with the minimum kept as the whole-pass time, then one more run with the per-kernel ledgers armed after the timed runs, so that the online tally's cost is not in the number a change is judged by. The printed table carries the guards above. A row whose rate exceeds 105 percent of its roof prints as an error (`IMPOSSIBLE_PCT`), a row graded against a roof that was not measured for its kind is marked, a pass with a kernel that has no cost formula prints "rate unavailable" and names the kernel, and the footer prints the sum of the kernel times beside the whole pass and says whether the times are device times or host-bracketed. `device_table` gives the per-kernel device time of a stage under the application's `--kernel-profile` flag; `program_table` replays a captured program a number of times (the flag `--profile-replays`) and reports each dispatch at its minimum, the statistic of the section on summaries: another context on a shared card only lengthens whichever dispatch it interrupts.

**The time source: `crates/backend-wgpu/src/lib.rs` and `crates/backend-cuda/src/backend.rs`.** The backend side of `set_kernel_timing` and `kernel_times` is the $n + 1$ stamps of the first section. In the wgpu backend `flush_timed` writes a timestamp before each dispatch and one after the last, inside the single compute pass that production runs, so dispatch $i$ costs `ticks[i + 1] - ticks[i]`. It resolves the stamps into a buffer and reads them later, so that timing a flush does not itself drain the device, and it spreads the stamps over several query sets when a pass has more dispatches than one set holds. A device without in-pass timestamps falls back to a pass per dispatch, which `flush_profiled` implements and which changes the execution it measures. `fold_ticks` is the "refuse what you cannot believe" clause: a batch containing a dispatch longer than `IMPLAUSIBLE_DISPATCH_MS` (2,000) is discarded as a whole, because a driver glitch that produced one impossible stamp pair is evidence about the neighbours' too. The CUDA backend times each launch with events and issues submissions launch by launch while timing is on; its `profile_program` launches a captured program once to warm it and then `reps` times, keeping each dispatch's minimum (commit 580746c00, October 2026, whose message gives the reason, and whose code agrees: the minimum over runs, since another context only ever lengthens a span).

**What the suite measures and refuses: `crates/perf`.** The suite's header states its rules, which are the habits above: warm-up requests never enter a statistic (`ReqRecord::warmup` in `crates/perf/src/metrics.rs`; `Summary` filters on it, and the driver in `crates/perf/src/driver.rs` takes its wall-clock window from the first *measured* submission, so that warm-up never inflates the denominator); a failed or unfinished request is never counted as good output; an unmeasured field is `null` and never zero (`crates/perf/src/schema.rs`, `devicetel.rs`: a missing sysfs file is `None`, not "not throttling"); a run whose correctness gate failed is marked `valid: false` and refused by comparison (`fidelity.rs`, `gate.rs`: speed is only accepted for a computation that was still right); and a result carries a fingerprint of device, backend, build and cores (`env.rs`) so that `report.rs` warns on every axis that differs between two compared runs and `gate.rs` refuses to gate runs that differ in scenario, unit, device axes or workload seed, and refuses a smoke-sized candidate. A scenario repeats and keeps the best of $N$ with the observed spread (`BestOf` in `crates/stats/src/latency.rs`, `scenarios/mod.rs`): the suite reports the best (the stable signal on a shared or throttled machine) together with the spread that says how far to trust it, the two numbers the section on the best of many asks for together. The `startup` scenario in `scenarios/startup.rs` reports cold and warm start-up as separate rows, the cold start being a legitimate measurement of its own thing. The driver is deliberately single-threaded so that the harness does not add its own scheduling noise to the numbers. `crates/perf` does not itself grade a run against a roofline; that grading is `gpu-core`'s, and the two are joined by the habit of reporting work and fraction together.

**The trace: `crates/trace`.** `brain-trace` is the front end of the workspace's structured logging. Library crates emit events through the `tracing` facade, and one binary installs the one subscriber, from the `--trace-<family>` flags (or the `BRAIN_TRACE` variable for entry points with no command line). The 0-to-5 scale maps onto `tracing`'s levels, level 4 adding "per-step decisions and timings" and level 5 everything per iteration, and a level of 0 installs nothing, so a run that does not ask pays nothing. The families in `crates/trace/src/registry.rs` name the crates a user means by "the GPU". Output is text or one JSON object per line (`--trace-format json`) to the standard output or a file, with target, level, fields and the span list as real members, and the events' timestamps are the host's clock when each event was emitted. That makes the trace the right tool for a different question from the profile's: where in a run a decision was taken or a stage ran (host time, in order, with the context), not how long the device spent in a kernel. A trace line's time is a host time and obeys the first section: a line emitted when work is enqueued says when it was handed over.

Three stories from the history, each confirmed against the code, show these rules at work in brain. Three more are told elsewhere in the book and are only pointed at here. The bare-submit timing loop that reported a bandwidth above the card's roof (knowledge #6, commit 883b40b0c) is told in [Chapter 4](04-gpu.md); the per-kernel table that gave credit that the whole pass did not (knowledge #21) in [Chapter 8](08-fusion.md); and the roofline probe that timed an idle device (commit 1bbcfc77e) in [Chapter 3](03-roofline.md) and [Chapter 10](10-kernel-selection.md).

**Host-bracketed slices rank kernels wrongly (knowledge #31, commits e13693ba7 and a8cadf8c2, August 2026).** Before the profiler used device stamps it timed each group of kernels by submitting that slice alone and bracketing it with a wait, the method of the second row of `profiling.sum_vs_pass`. The floor of 45 microseconds in the model has a real counterpart, and the commit message of e13693ba7 compares the two sources on one workload:

<!-- perf-number: historical measurement, commit e13693ba7 -->
In August 2026, on one GPU, the host-timed slice of a normalization kernel read about 20 times its device-timestamp time (0.240 against 0.0122 milliseconds per call), and the two paged-attention kernels about 17 and 29 times theirs; by device time the matrix product was 94.8 percent of that pass where the host-timed table put it at 53.8 percent, and two kernels with under 2 percent of the time had been promoted to "defect" rows by the host-timed table.

The effect was not a scale factor but an inversion of the ranking, as the second row of the table above predicts for small kernels. Commit a8cadf8c2 is the fix, confirmed in the code described above: stamps written between the dispatches of the production pass. Its stated validation is the check of the parts against the whole described above:

<!-- perf-number: historical measurement, commit a8cadf8c2 -->
In August 2026 on one GPU, the summed kernel device time of one pass was 80.28 milliseconds against 80.85 for the whole pass, 0.7 percent apart, where the earlier mode that gave every dispatch its own compute pass had reported 330.7 against 114.9.

**A profiler that drove the graph wrongly measured a graph that did nothing (knowledge #32, same commit).**

<!-- perf-number: historical measurement, commit a8cadf8c2 -->
In August 2026, with the timing fixed, one kernel still reported the impossible bandwidth of 7,060 (GB/s). The kernel was timed correctly and really was that fast, because the benchmark fed the serving tape in the mode `Input::Resident` (`crates/qwen3/src/serve.rs`), which is documented there as the mode in which the forward pass performs *no host writes at all*, because the previous step is supposed to have produced the token ids and the paged-attention metadata on the device already. A benchmark has no such previous step, the sequence lengths stayed zero, and the commit reports that 61 percent of the pass did no work. The lesson was costly: the benchmark had been used to publish a speedup that was corrected, in that commit, to 11.8 times from a figure between 27 and 29 times, and the commit's own words are "reproducibility is not validity": the ratio had been stable over four runs because both arms ran the same empty attention. The check that caught it was the roof test: the impossible-rate guard of the profile (`IMPOSSIBLE_PCT`) refused to print the number as a percentage, and a weight-bandwidth budget printed by the same tool said a served step could not be that cheap. A stable number can still be a number about the wrong computation.

**The minimum over replays on a shared card (commits d307f7ad4, 237034e8d and 580746c00, October 2026; `.agents/rules/kernels.md`, section E.0a).** A card that another process uses bills that process's time slices to whichever of your kernels it interrupts, which only ever lengthens a kernel. The remedy, confirmed in `profile_program`, is the statistic of the summary section: replay the same captured program and keep each dispatch's *minimum*, asked for with `--profile-replays` and never with an environment variable (commit 237034e8d moved it to a flag). The per-kernel timestamp mode of knowledge #31 was first switched on by an environment variable, `BRAIN_PROFILE`; commit 0fd8c51e5 (November 2026) replaced it by the application's `--kernel-profile` flag, which `backend-api` stores once and never reads from the environment.

**The institutional form of these rules: `scripts/gates/check-no-perf-numbers.sh`.** The gate, introduced by commit 8430aa763 (August 2026), turns the rule of reporting speed as work into a check. It denies, in every documentation page and in the comments, doc comments and string literals of source, a bare number beside a performance unit or claim: durations, rates in FLOP or bytes per second, speed-up multipliers and percentages near measurement words, and percentile values. The reasoning in its header is the one argued above: such a number is "a claim that outlives the hardware, driver, and code that produced it", and the day the kernel path changes or someone reads it on other silicon it becomes a lie its reader cannot detect. A number is let through only by a reviewed marker, `<!-- perf-number: <reason> -->` in prose or a comment in source with the reason, on the line or the line before: a vendor constant, a fixture, or, in this book, a historical measurement with its commit. Commit 0cbb46735 (August 2026) extended it from documents to source narration (code itself is exempt: a timeout is a value the program depends on, and not a claim), and the make target `check/scripts` runs it. In the book the percentages and multipliers are gated on performance vocabulary only, since the book's statistics chapters print percentages that are not about speed. This is why the tables above print times in a model with its parameters beside them, and why every real measurement quoted carries its commit and its setup.

## Check your understanding

1. A student times 100 dispatches of one kernel in a loop, with the clock before the loop and after it, divides by 100, and reports 3.1 microseconds per dispatch for a kernel that moves 805 MB. Find the error, say which of the three timers of the opening it resembles, and say what to check first.
2. A kernel's device time is 4.5 microseconds. Using the model of this chapter, about what will a bracket with two waits read, and by what factor will it overstate the kernel? What would you do to measure a program of many such kernels?
3. A warm-up of $\tau = 5$ runs is handled by discarding only run 0, the compile, and averaging the remaining 199. What bias remains, and why is "discard the first run" a poor rule?
4. A profile's per-kernel device times sum to 4 milliseconds while the whole pass measured with a bracket is 3 milliseconds. Another pass has per-kernel times that sum to 2 milliseconds against a bracket of 3. Give the most likely cause of each, and say what to look at next.
5. Kernel X runs at a fraction of 0.2 of its roofline and takes 2 percent of a pass. Kernel Y runs at 0.6 and takes 50 percent. Which would you work on first, and what does Amdahl's law say you can gain from bringing each to its roofline?
6. Find the error: "I ran both variants 10 times. B's mean is 4 percent below A's, so B is faster." The relative standard deviation of one run is 0.14.
7. A report says "the new kernel reaches 3.2 times the device's measured bandwidth." List, in order, the three things you check before believing or refuting it.
8. Find the error: "A part takes 90 percent of the time, so making it 100 times as fast makes the whole 10 times as fast."

## Summary

A time is a number with a question attached, and most timings that lie answer a different question from the one asked. The clock must stop only after the device has finished, by waiting for it or by reading the device's own stamps; a clock stopped at the call's return reads the host's hand-over cost, a fraction of the roofline above 1 is the sign, and a bracket that waits adds a floor ($t_{\mathrm{enq}} + t_{\mathrm{wait}}$, about 45 microseconds in the model, 45.6 times a small kernel's own time) that stamps between the dispatches of one submission do not pay. The first runs belong to a transient, and a rule that asks for $k$ consecutive runs within a tolerance of the floor ends the warm-up (at run 20 on the seeded series): averaged in, it biases a mean like $23.09/n$, by 11.9 percent at $n = 200$. Repetition gives the standard error $s/\sqrt{n}$, a comparison is the difference over its standard error with $n = 8\,\mathrm{cv}^2/\delta^2$ runs for a gap $\delta$ (1,644 runs for 1 percent on the noise of Chapter 10) and four times as many for confidence, and the best of many runs is a number the device reaches rarely (7.1 percent flattering on this model). A per-kernel profile divides a pass by dispatch with counted FLOPs and bytes and a fraction of the measured roofline, and its sum is never the pass (it can be 44 microseconds short, or 356 over); the whole pass, bracketed around everything the user waits for, is the number a change is judged by, and a kernel 14.9 percent faster by its stamps made the pass 2.44 percent slower. Amdahl's law, $A = 1/((1 - p) + p/g)$, bounds the gain of a part by its share, with ceiling $1/(1 - p)$, and the roofline bounds the part's own factor by $1/\mathrm{frac}$: on the model layer "twice as fast products" gives 1.75 but the device allows 1.4213. The conclusion is one rule. State speed as the counted work and the achieved fraction of the device's measured roofline, with the way the time was taken, and keep every time as a dated observation of a stated setup.

Measurement tells what is. The next chapter, [Quantization](12-quantization.md), changes what the weights are: it stores a model in four bits per weight, so that the bytes of the roofline fall, and judges the result by the error it introduces and by speed measured with these habits.

## References

- Gene M. Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities", 1967.
- Torsten Hoefler and Roberto Belli, "Scientific benchmarking of parallel computing systems: twelve ways to tell the masses when reporting performance results", 2015.
- Todd Mytkowicz, Amer Diwan, Matthias Hauswirth and Peter F. Sweeney, "Producing wrong data without doing anything obviously wrong!", 2009.
- Samuel Williams, Andrew Waterman and David Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", 2009.

## Exercises

1. **[Calculate]** For the matrix product $512 \times 2560 \times 2048$ of `profiling.enqueue_lie`, find $N_{\mathrm{flop}}$, $N_{\mathrm{byte}}$ and the roofline time on the measured roofline, and then the fraction that each of the three timers reads from the medians of the table.
2. **[Calculate]** With $\mathrm{cv} = 0.143314$ (printed to four places as 0.1433), find the number of runs of each variant that resolves a gap of 3 percent at two standard errors and at four. Check them against the pattern of `profiling.repetitions_needed`.
3. **[Calculate]** Derive the limit of $b(n)$ for the warm-up model, $A = 0.8$, $\tau = 5$, $C = 20$, $\bar f = 1.071$. How many runs bring the bias below 1 percent by averaging alone, and what is the bias if the first 20 runs are discarded and the transient there is $0.8 e^{-20/5}$? Compare with the table of `profiling.warmup_bias`.
4. **[Calculate]** A pass has three parts with shares of 0.6, 0.3 and 0.1 of its time. The first is made 3 times as fast and the second 1.5 times as fast. What is the speedup? What factor on the first part alone gives the same speedup?
5. **[Calculate]** A matrix product at a fraction of 0.55 of the roofline takes 40 percent of a pass. What is the largest speedup of the pass from improving it, with and without the roofline limit on the part, and what speedup would "twice as fast" promise?
6. **[Implement]** Reproduce `profiling.enqueue_lie` and `profiling.wait_cost`: the model (the enqueue, the wait, the device time with the noise factor of Chapter 10), the five uniforms per run in the printed order, the median of 11 runs and the three fractions. Use the book's SplitMix64 with the printed seeds.
7. **[Implement]** Reproduce `profiling.warmup_runs` and `profiling.warmup_series`: the series of 200 runs, the detection rule `warmup_end` with $k = 5$ and $tol = 0.05$, and the two columns of statistics. Then change $k$ to 1 and describe how the detected end moves on this series and why.
8. **[Implement]** Reproduce `profiling.kernel_profile`, `profiling.profile_by_kind` and `profiling.amdahl_table`: the counts of FLOPs and bytes of the ten dispatches, the minimum over 15 replays, the fractions of the measured roofline and the Amdahl columns. Add a dispatch of your own and say what it does to the shares.
9. **[Implement]** Reproduce `profiling.whole_pass_change`: both variants, the 1,200 passes of each in the printed order, and the two rows of the table. Find by experiment the smallest number of passes at which the difference exceeds 3 standard errors on this series, and compare with the formula of the chapter.
10. **[Investigate]** On a machine of your own, time the addition of two large arrays with a clock stopped at the call's return and with a bracket that waits, counting the bytes and taking the machine's bandwidth from a copy of your own. Report the fractions and see which is above 1. Do not quote the times as properties of your machine; write them as dated observations with the setup.
11. **[Investigate]** Read `print_top` in `crates/gpu-core/src/profile.rs` and list each guard that prevents one of the lies of this chapter (a rate above the roof, a rate over an incomplete count, a roof not measured for the kind, host-bracketed times that inflate small kernels). For each, say what the printed table would hide without it.
12. **[Investigate]** Run `scripts/gates/check-no-perf-numbers.sh` on a page of your own that quotes a time, a rate and a speedup, then add the reviewed marker to one of them. State, for each of the three, whether the number should have been a counted quantity (work, a fraction) instead of a time.

## Key terms

- **device-synchronized timing**: timing in which the clock stops only after the device has finished the work being timed, either by the host waiting for it or by reading timestamps the device wrote itself; a clock stopped when the call returns reads the host's cost of handing over the work. <!-- concept: device-synchronized-timing -->
- **warmup runs**: the leading runs of a repeated measurement whose cost is dominated by a start-up transient (building pipelines, first-touch memory, cold caches, a clock that has not ramped up) and is not the cost of the steady state; they are detected and kept out of the summary, or reported as a separate cold row. <!-- concept: warmup-runs -->
- **per-kernel profile**: the table that divides the time of a pass among its dispatches or kernel kinds, with each row's counted FLOPs and bytes, device time, share of the pass and achieved fraction of the device's measured roofline; it explains where the time goes and does not add up to the pass. <!-- concept: per-kernel-profile -->
- **whole-pass measurement**: the time of the whole unit of work a user waits for, with every dispatch, wait, copy and host-side step the real use performs inside the bracket, repeated and summarized with its spread; the number by which a change is judged. <!-- concept: whole-pass-measurement -->
- **Amdahl's law**: the speedup of a whole from making a part that takes the share $p$ of the time $g$ times as fast, $1/((1 - p) + p/g)$, bounded by the ceiling $1/(1 - p)$ however fast the part becomes. <!-- concept: amdahls-law -->

## Answers

**Recall questions.**

1. The host's cost of handing the dispatch over (building the request and queueing it), $t_{\mathrm{enq}}$. The call returns without waiting, so the kernel has barely started; the reading is nearly independent of the kernel.
2. $t_{\mathrm{bound}} = 100 \times 10^6 / (800 \times 10^9) = 125$ microseconds, so the fraction is $125/200 = 0.625$.
3. Recorded into one list: one submission and ten dispatch starts, $4 + 10 \times 1 = 14$ microseconds. Submitted one by one: $10 \times (4 + 1) = 50$ microseconds.
4. The minimum. A run never beats the floor of the device and noise only lengthens it, so the least contaminated draw is the lowest. The mean includes the spikes and sits above the floor by their average (7.1 percent in the model).
5. $0.2/\sqrt{100} = 0.02$ microseconds.
6. They are identical, to the last bit, because the generator is a deterministic function of its seed and every draw is consumed in the same order. A worked example prints its seed and its draw order so that a reader can seed their own generator the same way and reproduce its tables from the text alone.

**Check your understanding**

1. A bracket around a loop of dispatches with no wait after the loop reads the host's enqueue cost: 3.1 microseconds per dispatch is the host handing over work, and 805 MB cannot move in that time. It resembles the "enqueue only" timer. Check the fraction first: $805{,}306{,}368 / (646 \times 10^9) \approx 1{,}247$ microseconds per dispatch is the floor, so 3.1 microseconds is 400 times faster than possible. Then put a wait after the loop (and one before it) and report the time per dispatch of the whole.
2. The floor of a bracket is $t_{\mathrm{enq}} + t_{\mathrm{wait}} \approx 45$ microseconds (the table shows readings of about 50 for the smallest kernels), so the reading is about $4.5 + 45$ to $4.5 + 50$ microseconds, 11 to 12 times the kernel's time. For a program of many such kernels, take one bracket around the whole list and attribute the parts by device stamps between the dispatches; do not bracket each kernel.
3. The remaining transient is $0.8\sum_{i \ge 1} e^{-i/5}/199 = 0.8 \times 4.517/199 = 0.0182$, 1.8 percent, because the clock ramp lasts about 14 runs beyond run 0. The rule discards by position and not by evidence: it keeps the ramp when the ramp is long and discards good runs when it is short. Detect the end from the series itself.
4. The first (the sum exceeds the pass): kernels overlapped, so some stamp intervals cover the same time. Look for independent dispatches whose stamps overlap and for a device that runs two at once; a sum that exceeds the pass says the profile double counts. The second (the pass exceeds the sum by a millisecond): host time and gaps, one submission's worth of waiting, readbacks, a sync inside the pass, or work on the CPU between dispatches. Compare the number of submissions and waits and look at the host's timeline.
5. Gaining from X: $g = 1/0.2 = 5$, $p = 0.02$, $A = 1/(0.98 + 0.004) = 1.0163$. From Y: $g = 1/0.6 = 1.667$, $A = 1/(0.5 + 0.3) = 1.25$. Work on Y first, although X is the worse fraction: the payoff of a part is its share times how much of it can be removed.
6. $\mathrm{SE}_\Delta \approx 0.14\sqrt{2/10}\,\bar t = 0.063\,\bar t$, so a gap of 0.04 is 0.64 standard errors: far inside the noise and the order of the two means carries no information. About $8 \times 0.14^2/0.04^2 \approx 98$ runs of each variant would put it at two standard errors.
7. First the timer: a clock stopped at the call's return or without a wait before it reads the host's cost, so a fraction above 1 almost always means "the device had not finished". Then the work: whether the byte count is the bytes the device actually had to move (a cache can serve a small working set above the main-memory bandwidth) and whether the result was used so that nothing was removed. Then the roof: whether the measured bandwidth was itself measured on a warmed-up device with a probe that streams past the caches.
8. Amdahl's law gives $A(0.9, 100) = 1/(0.1 + 0.009) = 9.17$, not 10: 10 is the ceiling at infinite speedup and is never reached, and the roofline of the part will stop $g$ well short of 100 for most kernels.
