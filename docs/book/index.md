<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# From First Principles to Intelligent Machines

*The mathematics, algorithms and engineering of machine learning, told
through the building of brain.*

This is a textbook. It starts from arithmetic and ends with a complete
machine learning engine: one that trains and serves language, vision,
audio, video, forecasting and decision models on CPUs, GPUs and dedicated
accelerators. Every idea is derived before it is used, written as
language-neutral pseudocode, checked with worked numbers you can reproduce,
and then followed into a real implementation in brain, including the
mistakes that implementation made on the way.

You can read it without installing anything, and you can implement every
algorithm in it in the programming language of your choice.

Start with the [Preface](preface.md) and [How to Read This Book](how-to-read.md).
The symbols are collected in [Notation](notation.md), the algorithm language
in [Pseudocode Conventions](pseudocode.md), and the engineering timeline the
historical sections draw on in the [Chronology](history/chronology.md).

## Contents

<!-- contents -->
### [Volume I: Thinking in Mathematics](v01-mathematics/index.md)

1. [Numbers, Quantities and Notation](v01-mathematics/01-numbers.md): What are the numbers a machine learns with, and how do we write about them precisely?
2. [Functions and Their Shapes](v01-mathematics/02-functions.md): How do we describe a rule that turns inputs into outputs, and what shapes can such rules take?
3. [Vectors: Lists of Numbers with Geometry](v01-mathematics/03-vectors.md): How can a list of numbers have a length and a direction, and how do we measure how alike two lists are?
4. [Matrices as Linear Maps](v01-mathematics/04-matrices.md): What does it mean to multiply a matrix by a vector, and why is that the central operation of machine learning?
5. [Arrays, Shapes, Strides and Broadcasting](v01-mathematics/05-arrays.md): How are vectors, matrices and higher-dimensional arrays actually laid out in memory and indexed?
6. [Eigenvalues, Rotations and Low-Rank Structure](v01-mathematics/06-decompositions.md): What hidden structure does a matrix have, and how can a big matrix be approximated by small ones?
7. [Derivatives: The Mathematics of Change](v01-mathematics/07-derivatives.md): How do we measure how fast a function's output changes when its input moves a tiny bit?
8. [Gradients, Jacobians and Many Variables](v01-mathematics/08-multivariable-calculus.md): When a quantity depends on millions of numbers, which way should we move them all to make it smaller?
9. [Integrals and Differential Equations](v01-mathematics/09-integrals-and-odes.md): How do we add up infinitely many small pieces, and how do we follow a system that changes continuously?
10. [Probability: Reasoning Under Uncertainty](v01-mathematics/10-probability.md): How do we put numbers on uncertainty and update them when we see evidence?
11. [The Distributions Machine Learning Uses](v01-mathematics/11-distributions.md): Which few probability distributions keep appearing, and why?
12. [Statistics: Learning Facts from Samples](v01-mathematics/12-statistics.md): How much can a finite sample tell us, and how do we know whether a difference is real?
13. [Information, Entropy and Surprise](v01-mathematics/13-information-theory.md): How many bits does it take to describe what we expect to see, and what does it cost to expect the wrong thing?
14. [Complex Numbers, Frequencies and the Fourier Transform](v01-mathematics/14-fourier.md): How can any signal be written as a sum of waves, and why does that make convolution cheap?
15. [Floating Point and Numerical Error](v01-mathematics/15-floating-point.md): Why does a computer get arithmetic slightly wrong, and when does slightly wrong become completely wrong?
16. [Algorithms, Cost and Complexity](v01-mathematics/16-algorithms.md): How do we predict how long a computation will take and how much memory it needs before we run it?
17. [Randomness on a Deterministic Machine](v01-mathematics/17-randomness.md): How does a deterministic computer produce random numbers, and how do we make random experiments repeatable?
18. [Project: A Numerical Library from Nothing](v01-mathematics/18-project-numerical-library.md): Can you build the numerical toolkit every later chapter relies on, and prove each piece correct?

### [Volume II: What Does It Mean to Learn?](v02-learning/index.md)

1. [The Learning Problem](v02-learning/01-learning-problem.md): What exactly is a model, what is it learning, and from what?
2. [Linear Regression and the First Descent](v02-learning/02-linear-regression.md): How do we find the straight line that best explains our measurements, and what does 'best' mean?
3. [Classification and Logistic Regression](v02-learning/03-logistic-regression.md): How does a model output a probability for a yes-or-no question, and how do we train it?
4. [Maximum Likelihood, Priors and Regularization](v02-learning/04-likelihood.md): Why are the losses we use the right ones, and what does it mean to prefer simple explanations?
5. [Generalization, Overfitting and Honest Evaluation](v02-learning/05-generalization.md): How do we know a model has learned the pattern rather than memorized the examples?
6. [Measuring Models: Metrics and Calibration](v02-learning/06-metrics.md): Which number tells us whether a model is good, and why is accuracy so often the wrong one?
7. [Machine Learning Before Deep Learning](v02-learning/07-classical-methods.md): What did the field know before neural networks took over, and when are those methods still the right tool?
8. [Experiments, Baselines and Claims](v02-learning/08-experiments.md): How do we run an experiment whose result we can believe and someone else can repeat?
9. [Project: A Predictor You Can Trust](v02-learning/09-project-first-predictor.md): Can you train a model on real measurements, evaluate it honestly and explain exactly where it fails?

### [Volume III: Building Neural Networks from Nothing](v03-neural-networks/index.md)

1. [The Neuron and the Perceptron](v03-neural-networks/01-perceptron.md): What is an artificial neuron, and what can a single one never learn?
2. [Multilayer Networks](v03-neural-networks/02-mlp.md): Why does stacking layers with a nonlinearity between them let a network represent almost anything?
3. [Softmax, Logits and Cross-Entropy](v03-neural-networks/03-softmax.md): How does a network choose among many classes, and why is its gradient so simple?
4. [Backpropagation Derived from Scratch](v03-neural-networks/04-backpropagation.md): How do we compute the gradient of one loss with respect to millions of parameters in about the cost of one forward pass?
5. [The Backward Pass of Every Layer](v03-neural-networks/05-layer-gradients.md): What is the backward pass of a matrix multiply, a sum, a broadcast and an elementwise function, written out exactly?
6. [Proving Gradients Correct](v03-neural-networks/06-gradient-checking.md): How do we know a backward pass is right without trusting the person who wrote it?
7. [Optimizers: From SGD to AdamW](v03-neural-networks/07-optimizers.md): Once we have a gradient, what is the smartest way to use it?
8. [Learning-Rate Schedules and Gradient Clipping](v03-neural-networks/08-schedules-and-clipping.md): Why should the step size change during training, and what do we do when a gradient explodes?
9. [Initialization: Where Training Starts](v03-neural-networks/09-initialization.md): Why does the random starting point decide whether a deep network trains at all?
<!-- /contents -->
