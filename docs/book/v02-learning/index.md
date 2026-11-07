<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Volume II: What Does It Mean to Learn?

A machine learning model is a function with numbers left open, and learning is
the use of examples to choose those numbers. This volume makes that sentence
precise: what the examples are, what a model and its parameters are, how to say
that one choice of parameters is better than another, how to find a good choice
without trying all of them, and how to find out honestly whether the result
works on examples it has never seen. It does so with models small enough to
solve by hand (a straight line, a yes-or-no classifier) so that every step can
be computed and checked before the models grow.

## Why it comes here

Volume I supplied the language: functions, vectors, derivatives, probability.
Volume III builds neural networks, and a neural network is a model with a very
large number of parameters, trained by exactly the loop this volume develops
with two. A reader who has not seen that loop on a line, with the loss, the
gradient and the learning rate all visible, will meet it for the first time
inside a network, where nothing can be checked by hand. The volume also carries
the habits that keep the later work honest: separating the data used to choose
parameters from the data used to judge them, choosing a metric before looking at
a result, and comparing against a baseline.

## What you can do after it

You can state a learning problem precisely: the data, the model, the score and
the goal on new examples. You can fit a line by solving equations and by
gradient descent and say how the two relate, build a model that outputs a
probability and train it, explain why the losses in use are the right ones
(maximum likelihood) and what it means to prefer simple explanations
(regularization), split data so that an evaluation is trustworthy, choose a
metric that answers the question actually being asked, recognise when a
classical method is the right tool, and run an experiment whose result someone
else can repeat. The project that ends the volume does all of these on real
measurements.

## Chapters

1. [The Learning Problem](01-learning-problem.md): what a model is, what it
   learns and from what, and the three settings (labels, no labels, rewards) in
   which the same idea appears.
2. [Linear Regression and the First Descent](02-linear-regression.md): how to find the straight line that
   best explains a set of measurements, and what "best" means.
3. [Classification and Logistic Regression](03-logistic-regression.md): how a model outputs a probability
   for a yes-or-no question, and how it is trained.
4. [Maximum Likelihood, Priors and Regularization](04-likelihood.md): why the
   losses in use are the right ones, and what it means to prefer simple
   explanations.
5. [Generalization, Overfitting and Honest Evaluation](05-generalization.md): how
   to know that a model has learned the pattern rather than memorized the
   examples.
6. [Measuring Models: Metrics and Calibration](06-metrics.md): which number says
   whether a model is good, and why accuracy is so often the wrong one.
7. [Machine Learning Before Deep Learning](07-classical-methods.md): what the field
   knew before neural networks took over, and when those methods are still the
   right tool.
8. [Experiments, Baselines and Claims](08-experiments.md): how to run an experiment
   whose result can be believed and repeated.
9. [Project: A Predictor You Can Trust](09-project-first-predictor.md): train a
   model on real measurements, evaluate it honestly and explain exactly where it
   fails.
