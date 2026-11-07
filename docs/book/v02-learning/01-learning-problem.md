<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# The Learning Problem

*What exactly is a model, what is it learning, and from what?*

## The problem

A small electronics workshop builds a temperature sensor. It is a cheap
component that puts out a voltage, and the voltage depends on the temperature
of the thing it touches, but nobody has a formula that turns one into the
other. The datasheet gives a rough curve, the parts vary from one batch to the
next, and the workshop wants the real relationship for the real parts.

So someone does the obvious experiment. Ten times, they warm the sensor to some
temperature, read its voltage, and read the temperature on a trusted reference
thermometer at the same moment. Here are the ten measurements.

<!-- worked: learning_problem.sensor_table -->
N = 10 measured rows (sorted by voltage).

| row n | voltage v (V) | temperature y (deg C) |
|---|---:|---:|
| 1 | 0.78 | 27.5 |
| 2 | 1.29 | 44.0 |
| 3 | 1.46 | 48.5 |
| 4 | 1.46 | 49.5 |
| 5 | 1.73 | 57.1 |
| 6 | 1.94 | 59.7 |
| 7 | 2.05 | 59.8 |
| 8 | 2.35 | 68.6 |
| 9 | 2.52 | 68.9 |
| 10 | 2.56 | 69.7 |
<!-- /worked -->

The sensors will soon be installed in machines where there is no reference
thermometer, only the voltage. The question the workshop actually has is this:
a sensor reads $1.80$ V. What temperature is it?

There is no row in the table with $1.80$ V. Row 5 says that $1.73$ V was
$57.1$ degrees and row 6 says that $1.94$ V was $59.7$, so the answer is
somewhere near $58$, and you can say that without being able to
write down a rule for how you got there. That small act, producing an answer
for an input you have never seen from a handful of inputs you have, is what
this volume is about. This chapter builds the vocabulary for it precisely:
what the table is, what a **model** is, what it means for a model to be
chosen using data, what makes a choice good, and the three broad situations
(with labels, without them, and with only rewards) in which the same idea
appears.

Everything later in the book, from fitting a line to training a model with
billions of numbers in it, is a refinement of the picture drawn here. The
chapter does not teach any method well; it names the parts and shows each
one at a size where every number can be checked by hand.

You may use from earlier chapters: functions with parameters (Volume I,
chapter 2), vectors (chapter 3), probability and its distributions
(chapters 10 and 11), and the idea that a measurement taken from a sample
estimates a number about the whole source (chapter 12).

## Before you begin

Five questions on ideas from Volume I, to answer from memory. The answers are
at the end of the chapter. If one is hard, go back to the chapter named before
reading on.

1. (Chapter 2, "Functions and Their Shapes", families and parameters.) The line $f(x) = w x + b$ has the numbers $w = 3$ and $b = 1$ set. What is $f(2)$, and which of $w$, $b$, $x$ is changed to move from one member of the family of lines to another?
2. (Chapter 2, "Functions and Their Shapes", polynomials.) How many points with different $x$ values determine exactly one parabola through them, and what happens to the count for a family with $k$ parameters?
3. (Chapter 3, "Vectors: Lists of Numbers with Geometry".) The vector $\mathbf{x} = (84, 3, 12)$ has how many entries, and what is the distance between $(0, 0)$ and $(3, 4)$?
4. (Chapter 10, "Probability: Reasoning Under Uncertainty", a dataset is a distribution.) Five examples have errors $0, 1, 1, 4, 9$. What is their mean, and what probability does each example get when the mean is read as an expectation?
5. (Chapter 12, "Statistics: Learning Facts from Samples", estimates.) A model's average error is measured on a sample of examples. Why is that number an estimate, and what makes it more trustworthy?

## Examples, features and labels

Look at one row of the table: a voltage of $1.73$ V and a temperature of
$57.1$ degrees. The two numbers play different roles, and the vocabulary is
about telling the roles apart.

The voltage is what we will always have: it is the thing that is measured
at the moment of use. It is a **feature** of the example, one measurable
property that the prediction is allowed to depend on. The temperature is what
we want to know and, for these ten rows only, have been told by the reference
thermometer. It is the **label** of the example: the answer attached to it.
One row, a feature together with its label, is an **example**.

A real problem rarely has just one feature. A house has an area, a number of
rooms and an age, and the thing to predict is its selling price. Write the
features of one example as a list of numbers, a vector $\mathbf{x}$ with
$d$ entries (here $d = 3$), and its label as $y$. A collection of such
examples looks like this.

<!-- worked: learning_problem.feature_vectors -->
N = 5 examples, d = 3 features each: the feature matrix has shape [5, 3] and the labels have shape [5].

| n | area (m^2) | rooms | age (years) | price (thousand SEK) |
|---|---:|---:|---:|---:|
| 1 | 84 | 3 | 12 | 3150 |
| 2 | 52 | 2 | 41 | 1980 |
| 3 | 126 | 5 | 7 | 5240 |
| 4 | 67 | 3 | 25 | 2470 |
| 5 | 95 | 4 | 3 | 4110 |
<!-- /worked -->

The **dataset** is the collection of $N$ examples, written as pairs,

$$
\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_N, y_N)\}
= \{(\mathbf{x}_n, y_n)\}_{n=1}^{N},
$$

as the notation page has it. For the sensor $N = 10$ and $d = 1$: each
$\mathbf{x}_n$ is a vector with a single entry, the voltage, which we simply
write $x_n$ or $v_n$. For the houses $N = 5$ and $d = 3$. Stacking the
vectors $\mathbf{x}_n$ as rows gives a matrix $\mathbf{X}$ of shape $[N, d]$
(the layout of chapter 5: the first axis counts examples), and the labels form
a vector $\mathbf{y}$ of shape $[N]$. This is how every dataset in the rest of
the book is held in memory, and it is why the first axis of an array in this
book is so often called the batch: it is the axis of examples.

Two things about labels are worth saying early. First, a label does not have
to be a number measured on a scale. It can be one choice out of a short list
(a message is spam or it is not; a photograph shows one of a thousand kinds of
object), and the later chapters handle that case with its own tools. Second,
a label does not have to be written by a person. In the calculator task that
brain uses for tests, each line of the data is an arithmetic problem and its
answer, like `12+7=19`. Everything before the equals sign is the feature and
everything after it is the label, and a program that knows arithmetic wrote
both. A line of English text can be cut the same way: the words so far are the
features and the next word is the label, a source of an enormous number of
labelled examples that needs no one to label anything.

What counts as a feature is a decision, and a good deal of this book's history
is about moving that decision from people to the model. Here it is a measured
voltage. For an image it could be a few hand-chosen numbers (average
brightness, the length of the longest edge) or it could be every pixel and
nothing else, leaving the model to find what matters; the second choice is the
one the rest of the book makes.

## A model is a function with parameters

Now we can state the task exactly. Given a dataset of examples, produce a
function that takes the features of a new example, one the dataset has not
seen, and returns a good guess for its label. The function is the **model**,
and the guess is a **prediction**, written $\hat{y}$ (the hat means "an
estimate of").

Chapter 2 showed how to describe whole collections of functions with a few
numbers: the line $f(x) = wx + b$ is a family, and choosing $w$ and $b$ picks
one member. The model of this book is always such a thing. It is a function
whose rule has a shape fixed in advance and whose behaviour is set by a list of
numbers, the **parameters**, written together as one vector $\theta$. We write
the model as $f_\theta$:

$$
\hat{y} = f_\theta(\mathbf{x}).
$$

For the sensor, take the family of lines, $f_\theta(v) = w v + b$ with
$\theta = (w, b)$. Choose $\theta = (24, 9)$. Then the model answers any
voltage we ask it about.

<!-- worked: learning_problem.line_predictions -->
The model f(v) = w v + b with parameters theta = (w, b) = (24, 9).

| new voltage v | arithmetic | prediction (deg C) |
|---|---:|---:|
| 0.50 | 24 * 0.50 + 9 | 21.0 |
| 1.20 | 24 * 1.20 + 9 | 37.8 |
| 2.00 | 24 * 2.00 + 9 | 57.0 |
| 2.60 | 24 * 2.60 + 9 | 71.4 |
<!-- /worked -->

The distinction between the two kinds of number matters in everything that
follows. The feature $v$ changes from question to question: it is the input.
The parameters $w$ and $b$ do not change between questions; they are the
settings of the model, fixed while it predicts. Learning will be the activity
of changing the parameters, and nothing else, so that the predictions improve.

A line has two parameters. Real models have the same form with more. A
network that reads text can have seven billion numbers in $\theta$ (the
notation page writes this as a 7B model), and every one of them is a
parameter in exactly the sense above: a setting that the data is going to pick.
The rule that turns $\mathbf{x}$ and $\theta$ into $\hat{y}$ is much more
elaborate in such a model, and Volume III builds it, but the picture is
unchanged: a function with a shape fixed in advance and numbers left open.

## The hypothesis space

Before choosing parameters it helps to ask what choices are available at all.
If the model is a line, every possible choice of $(w, b)$ gives a different
line, and the set of all of them is everything the model can ever say. This set
of functions is the **hypothesis space**: all the functions $f_\theta$ that can
be reached by varying $\theta$,

$$
\mathcal{H} = \{ f_\theta \;:\; \theta \text{ any allowed list of numbers} \}.
$$

(The word hypothesis is the traditional one: each member of $\mathcal{H}$ is a
hypothesis about how labels depend on features.) Choosing a model family
chooses $\mathcal{H}$, before a single row of data has been looked at, and the
choice limits what can ever be learned. Three spaces for the sensor problem
show how.

- **Lines**, $f_\theta(v) = w v + b$. Two parameters. Every member is straight,
  so no setting of $(w, b)$ can bend.
- **Parabolas**, $f_\theta(v) = a v^2 + w v + b$. Three parameters. A parabola
  can curve once; setting $a = 0$ recovers the lines, so $\mathcal{H}$ for
  parabolas contains $\mathcal{H}$ for lines.
- **Lookup tables.** Keep the ten rows. To answer a new voltage, find the
  stored row with the closest voltage and return its temperature. The numbers
  that define this model are the ten stored voltages and the ten stored
  temperatures: twenty parameters. Its members are staircases that jump from
  one stored label to the next.

By the counting of chapter 2 (a family with $k$ parameters can pass through $k$
chosen points), the lookup table can reproduce any ten points exactly. A line
cannot even reproduce three that do not lie on one. Bigger spaces can say more,
and that sounds like a pure advantage; the next sections show that it is not,
and that choosing the size of $\mathcal{H}$ is one of the main design decisions
in the field.

## Learning: choosing parameters with data

We now have a model with open parameters and a dataset. **Learning** is the use
of the dataset to choose the parameters. For that, we need a way to say
whether a particular choice is good, and the simplest sensible one is to ask
how far off the predictions are on the examples we have.

Take the line through the first and last measurement: $\theta = (24, 9)$, the
same one as above. For each of the ten rows, the model predicts a temperature
from the voltage, and we put the prediction next to the label. The difference
between them, ignoring its sign, is how far off the prediction is on that row,
and the average of those ten numbers is how far off the model is on this
dataset.

<!-- worked: learning_problem.first_attempt -->
Attempt 1: w = 24, b = 9.

| v | label y | prediction | prediction - y | off by |
|---|---:|---:|---:|---:|
| 0.78 | 27.5 | 27.7 | +0.2 | 0.2 |
| 1.29 | 44.0 | 40.0 | -4.0 | 4.0 |
| 1.46 | 48.5 | 44.0 | -4.5 | 4.5 |
| 1.46 | 49.5 | 44.0 | -5.5 | 5.5 |
| 1.73 | 57.1 | 50.5 | -6.6 | 6.6 |
| 1.94 | 59.7 | 55.6 | -4.1 | 4.1 |
| 2.05 | 59.8 | 58.2 | -1.6 | 1.6 |
| 2.35 | 68.6 | 65.4 | -3.2 | 3.2 |
| 2.52 | 68.9 | 69.5 | +0.6 | 0.6 |
| 2.56 | 69.7 | 70.4 | +0.7 | 0.7 |
|  |  |  | total | 31.0 |
|  |  |  | average over N = 10 | 3.10 |
<!-- /worked -->

Writing that average as a function of the parameters,

$$
E(\theta) = \frac{1}{N} \sum_{n=1}^{N} \bigl| f_\theta(\mathbf{x}_n) - y_n \bigr| ,
$$

the first attempt scores $E(24, 9) = 3.10$ degrees. This number, a single
measure of how well one setting of the parameters explains the data, is what
makes the search for parameters a well-defined problem: **learning is finding
$\theta$ for which $E(\theta)$ is small.** The next chapter gives such
measures a name, asks which of the many possible ones to use (there is a good
reason to square the difference instead of ignoring its sign), and derives a
method that finds the best parameters of a line without trying them. Here the
only requirement is that a smaller $E$ means a better fit to these rows.

Now try to improve on it by hand. The table showed where the line is wrong: it
is below the data in the middle and close to it at both ends. Three attempts,
each a change a person might try after reading the previous score: the second
lowers $w$ to pull the right-hand end down and raises $b$ to lift the middle,
and the third keeps that $w$ and raises $b$ a little more.

<!-- worked: learning_problem.hand_tuning -->
| attempt | w | b | average off by (deg C) |
|---|---:|---:|---:|
| 1 | 24 | 9 | 3.10 |
| 2 | 20 | 15 | 4.67 |
| 3 | 20 | 18 | 2.27 |
<!-- /worked -->

The second attempt is worse than the first. Both knobs moved at once, and their
effects interact: a lower $w$ with a higher $b$ is a different line that pivots
through a different point, and here it ended up below the data nearly everywhere.
The third attempt, with $w$ held and only $b$ raised, is better than the first. This is
the character of the problem: with two parameters one can still tune by eye,
with a table of errors and some patience; with seven billion nobody can, and a
procedure has to do it.

The simplest procedure is to try many settings and keep the best one. For a
line, lay a grid over the plausible values of each parameter and compute $E$
at every point of the grid.

```text
function best_on_grid(x: array<real>[N], y: array<real>[N],
                      w_values: list<real>, b_values: list<real>)
        -> (real, real, real):
    require len(x) == len(y) and len(x) > 0, "need matching, non-empty data"
    require len(w_values) > 0 and len(b_values) > 0, "need a non-empty grid"
    best_w = w_values[0]
    best_b = b_values[0]
    best_e = infinity
    for w in w_values:
        for b in b_values:
            total = 0.0
            for n in 0 .. N:
                total = total + abs((w * x[n] + b) - y[n])
            e = total / N
            if e < best_e:
                best_w = w
                best_b = b
                best_e = e
    return (best_w, best_b, best_e)
```

<!-- worked: learning_problem.grid_search -->
Grid: w from 0 to 60 in steps of 0.25 (241 values), b from -20 to 40 in steps of 0.25 (241 values): 58081 settings, 580810 single predictions.

|  | w | b | average off by (deg C) |
|---|---:|---:|---:|
| best of the grid | 19.5 | 20 | 1.86 |
| attempt 3 by hand | 20 | 18 | 2.27 |
<!-- /worked -->

The search found a better line than any of the three attempts, and it paid for
it with a large amount of arithmetic: every one of the grid's settings costs a
pass over the data. For a grid of $m$ values per parameter and $k$ parameters
the count is $m^k$ settings, which is already hopeless at ten parameters and
unthinkable at seven billion. Every learning method in this book can be read as
a way around that count: a rule that, from the current parameters and the
data, says which way to change them (the gradient of Volume I, used in the next
chapter), so that the search takes a few steps instead of visiting everything.

The figure shows the four lines of the table inside the data.

![The ten measured rows as dots, with three hand-tuned lines drawn dashed and the best line found by the grid search drawn solid. Each line is labelled with its two parameters and its average error on the ten rows.](img/candidate-lines.svg)

## Too small, too big

The grid search found the best line, but is a line the right hypothesis space?
Run the same kind of search over parabolas (a grid in three parameters; its
size is in the table below), and build the lookup table, which needs no search.
Then score all three the way we scored the first attempt, on the ten rows that
were used to choose them.

To see how they behave on rows they were not chosen on, we need rows that were
not used. In a real project we would have to set some aside or collect more.
Here the source of the data is a program, so we can draw two thousand more rows
from the same source, never looked at while choosing parameters, and score the
three on those as well.

<!-- worked: learning_problem.hypothesis_spaces -->
Best member of each space found on the 10 known rows; then scored on 2000 fresh rows from the same source.

| hypothesis space | numbers that define a member | average off by on the 10 known rows | average off by on 2000 fresh rows |
|---|---:|---:|---:|
| line  w v + b | 2 | 1.86 | 3.70 |
| parabola  a v^2 + w v + b | 3 | 0.73 | 1.19 |
| lookup table (nearest stored row) | 20 | 0.10 | 3.30 |
<!-- /worked -->

<!-- worked: learning_problem.parabola_grid -->
Best parabola on the grid (241865 settings): a = -7.0, w = 47.0, b = -5.0, average off by 0.73 deg C on the known rows.
<!-- /worked -->

Read the table by columns. On the ten known rows the ordering is exactly what
the sizes suggest: the more parameters, the smaller the error, down to $0.10$
for the lookup table (not zero only because two of the stored voltages, $1.46$
V, have different temperatures and the table has to answer one value for both).
On the fresh rows the ordering changes. The parabola is best at $1.19$ degrees,
although on the known rows the lookup table looked far better. The line is stuck at $3.70$: no setting of it can
bend, and the sensor's curve bends. And the lookup table has become almost as
bad as the line, $3.30$, even though it was nearly perfect on the rows it was
built from.

The reason is the same one in each case, and it is worth stating plainly. The
labels contain two things: the real relationship between voltage and
temperature, which is smooth, and the measurement noise of the reference
thermometer and the sensor, which is not and which does not repeat. The line is
too small a space to hold the smooth part, so it is wrong everywhere in the same
way. The lookup table is large enough to hold the smooth part and the noise, and
it dutifully stores the noise of each of its ten rows, so each stored answer is
the real temperature plus an error that will not be there for the next sensor.
The parabola is big enough for the smooth part and too small to store much of the
noise, and that is the sweet spot of this dataset.

![The best line, the best parabola and the lookup table, all chosen using the same ten known rows (large dots), drawn over eighty of the fresh rows (small grey dots). The line misses the curvature, and the lookup table's steps follow the individual known rows away from the cloud of fresh ones.](img/hypothesis-spaces.svg)

A hypothesis space that is too small cannot represent the pattern. One that is
too large represents the pattern and also the accidents of the particular
rows. The next chapters turn this plain observation into a theory. The
chapter "Generalization, Overfitting and Honest Evaluation" gives the second
failure a name, explains why the score on the rows used for choosing
parameters is always too optimistic, and shows how to keep a set of rows apart
for an honest score. What this chapter needs is only the consequence: the goal
cannot be a small error on the known rows. It has to be a small error on rows
the model has not seen.

## Data comes from a distribution

What does "rows the model has not seen" mean? The rows of the sensor dataset
are not the only rows that could have been measured. The workshop could have
chosen other temperatures, other sensors, other days, and got other ten rows.
We describe this by saying that each example is a draw from a **probability
distribution**, in the sense of chapters 10 and 11, over pairs of a feature
and a label: $(\mathbf{x}, y) \sim p$. The distribution $p$ is whatever
produces the world's examples (here: sensors from this batch, a temperature
chosen by the experimenter, a noisy thermometer), and the dataset is $N$ draws
from it:

$$
(\mathbf{x}_n, y_n) \sim p, \qquad n = 1, \dots, N, \text{ drawn independently.}
$$

We never see $p$. We see only the draws. In this book's tables $p$ is a short
program, and the program is not shown to the learner: to the code that picks
parameters, it is as unknown as the physics of the sensor is to the workshop.

This changes the meaning of the goal. What we want is not a small error on
the ten rows but a small error on *a new draw from $p$*, on average:

$$
\mathbb{E}_{(\mathbf{x}, y) \sim p}\bigl[\, \lvert f_\theta(\mathbf{x}) - y \rvert \,\bigr].
$$

This is an expectation in the sense of chapter 10, and it cannot be computed
without $p$. What we computed instead, $E(\theta)$, is the same expectation
taken under the empirical distribution of the dataset, which gives each of the
$N$ examples probability $1/N$ (chapter 10 called this "a dataset is a
distribution"). That is an estimate of the quantity we want, and chapter 12
described how such an estimate behaves: it is itself random, it has a sampling
distribution, and it improves as the sample grows. The two thousand fresh rows
of the previous section are a second, much larger sample, used as a better
estimate of the same expectation.

The consequence applies to the parameters too. The parameters we choose are a
function of the rows we happened to draw, so they are random as well. Draw
five different datasets of ten rows from the same source and pick the best line
for each.

<!-- worked: learning_problem.resampled_lines -->
| dataset | best w | best b | off by on its own 10 rows | off by on 2000 fresh rows |
|---|---:|---:|---:|---:|
| 1 | 15.25 | 30.5 | 2.10 | 5.93 |
| 2 | 26.75 | 7.5 | 2.49 | 3.12 |
| 3 | 19.75 | 20.5 | 3.16 | 3.72 |
| 4 | 21.75 | 16.75 | 2.84 | 3.23 |
| 5 | 22.25 | 13.75 | 2.28 | 3.26 |
<!-- /worked -->

Five datasets from one source give five different best lines, with $w$ anywhere
from $15$ to $27$, and the fifth column shows that the lines are not equally
good on new rows: dataset 1, whose ten voltages happened to include none below $1.06$ V, chose a
line that is much worse on fresh data than the others do, because nothing in its
rows showed it the steep low end of the curve. Nothing was done wrong: the line
is the best one for its ten rows, and the ten rows were a poor sample. A model's parameters carry the uncertainty of the data they came from.
More data narrows this spread, in the way chapter 12's standard error shrinks as
the sample grows, and it is the main reason the large models of the later
volumes want datasets of billions of examples.

There is one more thing the sentence "new draws from the same distribution"
quietly requires, and it is worth saying aloud because it is where most real
projects go wrong: the examples the model will meet in use must come from the
same $p$ as the ones it learned from. If the sensors in the machines run hotter
than the ones in the experiment, or the model will be used in a different way
than the way its data was collected, the guarantee in this section does not
apply. The chapter on generalization names this problem (distribution shift),
and the "Where this appears in brain" section below shows a real instance.

## Three ways to learn

The sensor problem has a specific shape: every example has a label, and the
task is to predict labels. That shape has a name, and there are two others that
differ in what the learner is given.

![Three learning set-ups by what the learner is given. Supervised: rows with features and labels, a model with parameters, and predictions for new features. Unsupervised: rows with features only, a model of their structure. Reinforcement: an agent sends actions to an environment and receives observations and rewards back, so it produces the data it learns from.](img/learning-paradigms.svg)

### Supervised learning

When every example in the dataset comes with its label, and the task is to
predict the label of a new example, the problem is **supervised learning**.
The name comes from the labels playing the part of a teacher, who supervises by
saying, for each example, what the right answer was. Everything in this chapter
so far is supervised learning.

Its forms are very widely used, and it is worth naming a few in plain terms.
Deciding whether an email is spam, what object a photograph shows, or which
language a sentence is in are problems whose label is a choice from a short
list. Predicting a price, a temperature, or tomorrow's electricity demand from
what is known today are problems whose label is a number. Next-word prediction
for text, the task that trains the language models of Volume V, has the next
word as its label, and the data supplies the labels itself. Forecasting is the
same with time as the ordering: the features are the last few values of a
series and the label is the next value. The next chapters give these their
technical names and tools.

### Unsupervised learning

Sometimes the dataset has no labels at all, only features, and what is wanted
is to find some structure in them: groups of similar rows, a few numbers that
summarise many, rows that do not look like the others. This is **unsupervised
learning**. There is nothing to predict, so there is no error to compute
against a label, and the model is a description of the data instead.

A concrete case. The workshop records two numbers from each of twelve machines
(the vibration frequency and amplitude, say) and suspects there are two kinds of
machine, but nobody has written down which is which. The twelve rows are
two-feature vectors, with no labels. A procedure that finds two groups can be
written with nothing beyond the vectors of chapter 3 (the distance between two
of them) and averages. Pick two starting centres. Then repeat two steps:
assign every row to the nearest centre, and move each centre to the mean of the
rows assigned to it. Stop when the assignment no longer changes. The starting
centres here are deliberately poor, both in the lower left corner.

<!-- worked: learning_problem.grouping_loop -->
| round | centre A | centre B | points assigned to A and B |
|---|---:|---:|---:|
| 1 | (0.00, 0.00) | (1.00, 0.50) | 1 and 11 |
| 2 | (-0.12, 1.34) | (2.37, 2.17) | 4 and 8 |
| 3 | (0.50, 1.05) | (2.99, 2.62) | 6 and 6 |
| 4 | (0.84, 1.14) | (3.48, 3.06) | 6 and 6 |
<!-- /worked -->

The first round puts almost everything in one group, because both centres are on
the same side of the data. Moving the centres to the group means pulls them
apart, the second round splits four and eight, the third splits six and six,
and in the fourth the assignment does not change: the procedure has stopped. The
groups it found, with the rows that fell into each:

<!-- worked: learning_problem.grouping_points -->
| n | feature 1 | feature 2 | group found |
|---|---:|---:|---|
| 1 | 3.51 | 3.22 | B |
| 2 | 1.54 | 0.82 | A |
| 3 | 2.83 | 3.65 | B |
| 4 | 3.97 | 2.98 | B |
| 5 | 0.77 | 1.27 | A |
| 6 | -0.12 | 1.34 | A |
| 7 | 1.04 | 0.88 | A |
| 8 | 0.32 | 0.70 | A |
| 9 | 3.56 | 2.80 | B |
| 10 | 3.34 | 2.43 | B |
| 11 | 3.66 | 3.29 | B |
| 12 | 1.51 | 1.80 | A |
<!-- /worked -->

![Twelve unlabelled readings coloured by the group the procedure found, with the final group centres as large dark dots and the earlier positions of the centres as small grey dots.](img/groups-found.svg)

The parameters of this model are the positions of the two centres, four numbers
in all, and "learning" chose them from the data without any label. The same
two-step procedure is a well-known method with a name of its own, which a later
chapter (on classical methods) gives and examines; what this chapter needs is
the shape. Unsupervised learning is also how a model can learn from far more
data than anyone could label, and many of the large models later in the book
use it as a first stage.

### Reinforcement learning

In the third setting there is no fixed dataset at all. An **agent** (the thing
that decides) acts in an **environment** (everything else): it picks an action,
the environment responds, and the agent receives a number, the **reward**, that
says how well that went. Nobody tells the agent what it should have done. It
learns which actions lead to good rewards by trying them. This is
**reinforcement learning**, and this chapter gives only an overview of it (the
concept is introduced here as the overview; the full treatment is Volume XII).

Two things set it apart from the other settings. The data depends on what the
agent does: if the agent never tries an action it never learns what the action
would have given. And the reward may arrive late: in a game, most moves are not
immediately rewarded, and the signal comes at the end.

The smallest instance is a machine with two buttons. Pressing button A pays one
unit with probability $0.3$ and nothing otherwise; button B pays one unit with
probability $0.7$. The agent does not know these probabilities. It presses the
buttons alternately for ten steps to find out, compares the average reward
each one has paid, and then keeps pressing the one that looked better.

<!-- worked: learning_problem.button_game -->
| step | button pressed | reward |
|---|---:|---:|
| 1 | A | 0 |
| 2 | B | 1 |
| 3 | A | 0 |
| 4 | B | 0 |
| 5 | A | 1 |
| 6 | B | 1 |
| 7 | A | 0 |
| 8 | B | 0 |
| 9 | A | 1 |
| 10 | B | 1 |

Average reward: A 0.4, B 0.6. The agent now presses B for 10 more steps and collects 9 reward. Total over 20 steps: 14.
<!-- /worked -->

The agent measured B as better ($0.6$ against $0.4$, the true values being
$0.7$ and $0.3$), chose it, and collected most of the possible reward in the
second half. Ten presses is a small sample, in the sense of chapter 12: with a
slightly different run of luck it might have preferred A. Every part of this tiny
game returns in Volume XII at a larger size, including the tension it shows
between spending steps to learn which button is better and spending them to
collect the reward.

### Which setting, and how they combine

The three settings differ in what is given: examples with answers, examples
alone, or only the consequences of one's own actions. The same data can often
be viewed in more than one. A language model trained to predict the next word
is doing supervised learning on labels the text supplied; a model that is later
tuned from human preferences is learning from rewards. A project usually begins
with a question ("what will the temperature be?", "which machines are alike?",
"what should the robot do?") and the question picks the setting. The first two
settings are the subject of Volume II. The rest of this volume is supervised
learning, because it is the setting in which "how good is this model" has the
most direct answer, and because the tools it develops carry over to the other
two.

## Making it work

**Know the unit of an example.** The ten rows were ten separate measurements.
If the same sensor had been measured ten times in a row, the rows would not be
independent draws of anything, and the dataset would hold one example ten times
over. Before counting $N$, decide what one example is.

**Keep features in comparable units.** A feature measured in thousands (price in
kronor) next to one measured in fractions (a rate) will make some learning
procedures struggle for reasons that have nothing to do with the problem.
Chapter 12's z-score is the standard remedy and the next chapter shows why it
matters.

**State the question the numbers answer.** The average error on a dataset is
an estimate of the average error on the distribution the dataset came from.
The two thousand fresh rows were drawn from the same source, so they answer
"how well will it do on sensors like these?". They say nothing about sensors
that are hotter, older or built differently.

**Common mistakes.**

- Reporting the error on the rows used to choose the parameters as if it were the
  error on new data. The lookup table's $0.10$ against its $3.30$ is the whole
  lesson.
- Taking a bigger hypothesis space to be better by definition. It contains more
  functions, including the one that is right, and a great many that are wrong
  in ways the known rows cannot reveal.
- Treating a label as ground truth. The reference thermometer had its own error;
  a model that learns the labels well learns the labels' errors with them.

## Where this appears in brain

brain is a collection of models, but each one starts from the same three things
this chapter named: examples, a function with parameters, and a rule for picking
the parameters.

**Datasets are generated, labels and all.** `crates/data` is brain's data layer,
and its generators play for the models the part that a program played for the
sensor here. `crates/data/src/gen_calculator.rs` emits lines of the form
`EXPRESSION=RESULT` from random one- and two-operator arithmetic, each at most 64
characters, and its doc comment states that the output is deterministic for a fixed
random-number generator, a property its test `deterministic_for_fixed_seed`
checks. Other generators produce a string
reverser (`gen_reverser.rs`), arithmetic written in words (`gen_wordcalc.rs`) and a
three-phase electrical signal with noise (`gen_timeseries.rs`). In each, the
distribution $p$ of this chapter is the generator and its random stream, and a
dataset is what comes out of it. Commit `00121704e` (24 June 2026) introduced
the crate with these generators, "ported from" a reference data pipeline, and
the loader beside them: `crates/data/src/loader.rs` cuts a stream into
feature-and-label windows, with the label being the next token, and its
`mask_before_token` option, given the character `=`, marks every target up to and
including the first one on a line as ignored, so the loss counts only what comes
after the equals sign. That is this chapter's split of a calculator line into features
(before `=`) and a label (after), written as code.

**A model with parameters, trained by a loop.** The model of this chapter is the
`Model` trait in `crates/model/src/lib.rs`: a type holds its parameters and
exposes `forward`, which returns the scalar loss for the current batch,
`backward`, which accumulates the gradient of that loss with respect to the
parameters, `zero_grads`, and `adamw_step`, which updates the parameters. That
loop (compute how far off, find which way to move every parameter, move) is the
subject of the next chapter and is derived for networks in
[Backpropagation Derived from Scratch](../v03-neural-networks/04-backpropagation.md).
Models trained through brain's generic trainer, from the small autoencoder in
`crates/toyautoencoder` to the language models, implement it.

**The third setting.** `crates/data/src/gen_pong.rs` is a two-paddle game in
which the agent controls the left paddle: three actions (do nothing, up, down),
and a reward at every step, $+1$ when the agent scores, $-1$ when it concedes and
$0$ otherwise. Commit `558cdaa24` (15 July 2026) introduced it, with the episode
format of `crates/data/src/episode.rs` that records the actions and rewards. The
reinforcement-learning side of the repository, `crates/rl`, trains a model from
rewards: its `fit_weighted` is the ordinary training loop with one change, that
each position's contribution carries a weight taken from a reward.

**The distribution you are scored on.** The same commit holds a small instance of
the section on distributions. `crates/data/src/episode.rs` splits a dataset into
training and held-out episodes by whole episode, never by single frame, and its
comment gives the reason: neighbouring frames of one game are almost identical,
so a split that scattered them would put near-copies of the training rows into
the held-out set and make the score look better than it is. A larger version of
the lesson is knowledge #145. A sample (`samples/decision/rubiks`) generated its
labelled examples by walking scrambled cubes home, a method that produces many
examples one move from solved and few far away. A policy fitted to them scored 87%
on a held-out set drawn the same way, and picked a shortest move on 37% of the
turns it took when driving itself, solving 9 cubes in 50 (the figures are the
entry's and the sample's own comment's, from one run). The held-out number was a
correct average over the distribution it was drawn from; it was not a number
about the task. The fix, in commit `3cbc3bbfb` (23 September 2026), was to stop
letting the sampling method decide the distribution: every distance from solved
now gets the same quota of examples. A model is only ever as good as the match
between the distribution it learned from and the one it is used on.

## Check your understanding

Answers are at the end of the chapter.

1. **Explain.** In $\hat{y} = f_\theta(\mathbf{x})$, which of $\hat{y}$, $\mathbf{x}$ and $\theta$ are changed (a) when we ask the same model about a new example and (b) when we say the model is learning?
2. **Predict.** The line $f(v) = 24 v + 9$ is asked about $v = 2.20$ V. What does it predict? If the true temperature was $62.0$ degrees, by how much is it off?
3. **Find the error.** "A parabola has three parameters and a line has two, so the best parabola is guaranteed to have a smaller error than the best line on any rows." For which rows is the claim true, and for which is it not necessarily true? (Hint: use the fact that the lines are parabolas with $a = 0$.)
4. **Explain.** The lookup table scores $0.10$ degrees on the known rows and $3.30$ on fresh rows. A colleague says that this proves the lookup table is a bad model. What does it show, and what would you want to know before calling it bad?
5. **Find the error.** "Dataset 1 of the five resampled datasets chose a very different line from the others, so there must have been a bug in its search." What is the more likely explanation, and what does it say about the parameters?
6. **Predict.** A grid for a model with $k = 3$ parameters has $m = 100$ values per parameter. How many settings does it contain, and how many for $k = 10$?
7. **Explain.** For each of the following, say which of the three settings it is and what the data or feedback consists of: (a) grouping customers by their purchases with no categories given; (b) predicting a stock's next price from its history; (c) a robot learning to balance by being told only whether it fell.
8. **Explain.** Why does the "data comes from a distribution" assumption make the average error on the known rows an estimate rather than the answer?

## Exercises

1. **[Calculate]** For the line $f(v) = 24v + 9$ compute by hand the prediction and the off-by amount for rows 1, 5 and 10 of the sensor table, and check them against the table of `learning_problem.first_attempt`. Then compute the average error of the same line on only those three rows, and say whether the difference from $3.10$ is a surprise.

2. **[Calculate]** Starting from $\theta = (20, 18)$ (attempt 3), raise $w$ by $1$ and recompute the average error over the ten rows. Then lower $b$ by $1$ in the original and recompute. Which single change is better, and what does the comparison say about trying one parameter at a time?

3. **[Calculate]** A model has $d$ features and predicts $\hat{y} = w_1 x_1 + \dots + w_d x_d + b$. How many parameters does it have? How many for the lookup table over $N$ examples of $d$ features each? Evaluate both for $d = 3$ and $N = 5$ (the houses table) and for $d = 1000$ and $N = 10^6$.

4. **[Calculate]** Count the work of a grid search. For $N = 10$ rows, $k = 2$ parameters and $m = 241$ values each, how many settings and how many single predictions? Compare with the number printed by `learning_problem.grid_search`. Then for $k = 3$ with $m = 241$ and $N = 10$.

5. **[Calculate]** Do two rounds of the grouping loop by hand on the four one-feature rows $1, 2, 8, 9$ with starting centres $1$ and $2$. Give the assignment, the new centres, and say whether a third round changes anything.

6. **[Implement]** In your language, generate the sensor dataset of this chapter's described source (voltage drawn uniformly from $[0.4, 2.8]$ V, temperature $-6 + 50v - 8v^2$ plus Gaussian noise of standard deviation $1.2$, rounded to two and one decimals) with any random generator, and implement the grid search of the pseudocode. Reproduce the structure of `learning_problem.grid_search` (a best line whose average error is clearly below that of the three hand attempts); your numbers will differ because your random generator does.

7. **[Implement]** Implement the grouping loop of this chapter and reproduce `learning_problem.grouping_loop` and `learning_problem.grouping_points`: take the twelve rows of the second table as input, start from the centres $(0, 0)$ and $(1, 0.5)$, and your centres and group sizes round by round must match, to the digits printed.

8. **[Implement]** Implement the decision rule of the button game: from the ten rewards of the table of `learning_problem.button_game` (use them as input rather than generating them), reproduce the two average rewards and the button chosen. Then run your own game for $10^5$ steps with the explore-then-choose rule, with exploration lengths $2$, $10$ and $100$, and report how often the agent picks the wrong button. Which length is best, and why does the answer depend on how close the two probabilities are?

9. **[Investigate]** Reproduce the three rows of `learning_problem.hypothesis_spaces` with your own data from exercise 6. Then add a fourth hypothesis space: cubic polynomials. Does the error on the fresh rows fall below the parabola's? Repeat with a dataset of $N = 100$ rows instead of $10$ and say what changed.

10. **[Investigate]** Redo `learning_problem.resampled_lines` with $N = 100$ rows per dataset instead of $10$. Report the range of the best $w$ across five datasets and compare with the range at $N = 10$. How does the spread seem to scale with $N$, and what does chapter 12 predict?

11. **[Investigate]** Make the fresh rows come from a slightly different distribution (for example, voltages from $[2.0, 3.2]$ V, partly outside the range of the known rows) and score the line, the parabola and the lookup table on them. Which of the three degrades most, and why?

## Summary

A **dataset** is a collection of $N$ examples, each a vector of **features**
$\mathbf{x}$ and a **label** $y$. A **model** is a function $f_\theta$ whose
shape is fixed in advance and whose behaviour is set by its **parameters**
$\theta$; its output for a new example is a **prediction** $\hat{y}$. The set
of all functions the model can express as $\theta$ varies is its **hypothesis
space**, chosen before the data is seen. Learning is using the dataset to choose
$\theta$, which needs a number that says how far off a choice is, here the
average of the absolute differences between predictions and labels on the known
rows. By hand, and then by trying a grid of $58{,}081$ settings, we improved a
line's score from $3.10$ to $1.86$ degrees, and saw that the grid's cost grows
as $m^k$ and so cannot be the method.

Comparing three hypothesis spaces showed that bigger is not simply better. The
line was too small to hold the curve of the sensor. The lookup table of twenty
numbers was almost perfect on the rows it was chosen on and almost as poor as the
line on new ones, because it stored the noise of those rows. The goal must
therefore be stated against new examples: the data are draws from an unknown
distribution, the quantity we want is an expectation under it, the average on the
known rows is an estimate of that expectation, and the parameters are estimates
too, different for every dataset. Three settings differ in what the learner is
given: **supervised learning** has labels, **unsupervised learning** has
features alone and finds structure, and **reinforcement learning** (here an
overview, Volume XII has the rest) has an agent acting and receiving rewards.

Limits: the score we used, the absolute difference, was chosen because it is
easy to read, not because it is the best; the search by grid cannot scale; and
the check on rows that were not used, though it exposed the lookup table, was
done with a conveniently large set of fresh rows that real projects do not have.
The next chapter, Linear Regression and the First Descent, takes the question of
the score seriously (it gives it a name and a justification), solves the line
exactly, and replaces the grid by following the gradient.

## Key terms

- **dataset**: a collection of $N$ examples, each a vector of features paired with a label, written as the pairs $(\mathbf{x}_n, y_n)$. <!-- concept: dataset -->
- **feature**: one measurable property of an example that a prediction may depend on; the features of one example form a vector. <!-- concept: feature -->
- **label**: the answer attached to an example, the quantity a supervised model is asked to predict. <!-- concept: label -->
- **model**: a function whose rule has a fixed shape and whose behaviour is set by a list of parameters, used to turn the features of an example into a prediction. <!-- concept: model -->
- **parameter**: one of the numbers that set a model's behaviour; fixed while the model predicts and changed while it learns. <!-- concept: parameter -->
- **prediction**: the value $\hat{y} = f_\theta(\mathbf{x})$ a model returns for an example. <!-- concept: prediction -->
- **hypothesis space**: the set of all functions a model can express as its parameters vary. <!-- concept: hypothesis-space -->
- **supervised learning**: learning from examples that all carry labels, in order to predict the label of a new example. <!-- concept: supervised-learning -->
- **unsupervised learning**: learning from examples without labels, to find structure such as groups in the features themselves. <!-- concept: unsupervised-learning -->
- **reinforcement learning (overview)**: learning by an agent that acts in an environment and receives rewards, without being told the right action; full treatment in Volume XII. <!-- concept: reinforcement-learning-overview -->

## Answers

**Before you begin.**

1. $f(2) = 3 \cdot 2 + 1 = 7$. Moving between members of the family changes the parameters $w$ and $b$; $x$ is the input and is not part of the choice of member.
2. Three points with different $x$ values determine one parabola; in general a family with $k$ parameters can be made to pass through $k$ chosen points (given the usual conditions), and $n + 1$ points determine a polynomial of degree at most $n$.
3. Three entries. The distance between $(0, 0)$ and $(3, 4)$ is $\sqrt{3^2 + 4^2} = 5$.
4. The mean is $(0 + 1 + 1 + 4 + 9) / 5 = 3$. Each example has probability $1/5$; the equal values pool, so the value $1$ has probability $2/5$.
5. The sample is a random draw from the source, so its average is a random variable with a sampling distribution around the true average. A larger sample (and independent examples) narrows it, with the standard error shrinking as one over the square root of the sample size.

**Check your understanding.**

1. (a) Only $\mathbf{x}$ (and so $\hat{y}$) changes; $\theta$ stays fixed. (b) Only $\theta$ changes, and with it the whole function $f_\theta$; the data stay as they are.
2. $24 \cdot 2.20 + 9 = 61.8$ degrees. It is off by $\lvert 61.8 - 62.0 \rvert = 0.2$.
3. It is true for the rows used to choose the parameters, when the error is the one being minimised: every line is a parabola with $a = 0$, so the best parabola on those rows is at least as good as the best line (up to the grid's resolution). It does not follow for new rows: the extra parameter is fitted to the noise of the known rows too and can make the predictions worse on others.
4. It shows only that the table is excellent on rows it was built from and mediocre on others, so the known-row score is not informative about new rows. Before judging it, ask for the error on fresh rows from the same source (it was $3.30$) and compare with the alternatives on those (the parabola was $1.19$). On those rows it is worse than the parabola and about as poor as the line.
5. The more likely explanation is a poor sample: its ten voltages included none below $1.06$ V, so nothing in its rows showed the steep low end of the curve and the best line for those rows was a poor one there. The parameters depend on the particular rows drawn, so they vary from dataset to dataset.
6. $100^3 = 10^6$ settings, and $100^{10} = 10^{20}$ for $k = 10$.
7. (a) Unsupervised: features only, the structure (the customer groups) is to be found. (b) Supervised: the label is the next price, which the history itself supplies. (c) Reinforcement: no examples of correct actions, only a reward signal (a fall or not) after the agent acts.
8. We care about new draws from the distribution, and the known rows are only $N$ of them. Their average error is the expectation under the empirical distribution of the sample, which varies with the sample and, for parameters chosen on these same rows, is also biased toward being too small. It is an estimate of the quantity we want, not the quantity itself.

## References

- Mitchell, T. M., *Machine Learning*, McGraw-Hill, 1997 (the formulation of learning as improvement from data on a task, and the term hypothesis space).
- Vapnik, V. N., *The Nature of Statistical Learning Theory*, 1995 (learning as choosing a function from a space using draws from an unknown distribution).
- Hastie, T., Tibshirani, R. and Friedman, J., *The Elements of Statistical Learning*, second edition, 2009 (supervised and unsupervised learning).
- Sutton, R. S. and Barto, A. G., *Reinforcement Learning: An Introduction*, second edition, 2018.
