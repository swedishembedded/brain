<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Volume III: Building Neural Networks from Nothing

Volume II ended with models that learn: a linear model fitted by gradient
descent, a logistic model that classifies, and an honest way to measure
whether either of them has learned anything. Those models have a few
parameters, and their gradients can be written down by hand in a line.

This volume builds the models that made machine learning a different field:
networks with millions of parameters arranged in layers, trained by the same
gradient descent. Nothing about the update rule changes. What changes is how
the gradient is obtained, how the starting point is chosen, how the step is
shaped, and how we find out that a network of this size is silently wrong.

## Why it comes here

A neural network is a composition of simple functions, so its gradient is a
chain rule applied many times. That sentence is easy to say and hides almost
all of the engineering. How is the chain rule organised so that the cost is
one extra pass instead of one pass per parameter? What must the forward
computation remember? How do we know a hand-written backward pass is correct?
Which optimiser, which schedule, which initial weights? Every later volume
(language models, vision, diffusion, audio, forecasting, reinforcement
learning) trains networks built from the pieces of this one, and every one of
them fails in the ways this volume teaches you to diagnose.

The volume needs Volume I (vectors, matrices, the chain rule, Jacobians,
floating-point numbers) and Volume II (loss functions, gradient descent,
generalisation). It is needed by everything that follows.

## What you can do after it

You can build a multilayer network from a blank file in any language, derive
and implement its backward pass layer by layer, prove the gradients correct
against finite differences, train it with AdamW under a warmed-up schedule
with gradient clipping, initialise it so that signals neither vanish nor
explode, regularise it, and read a loss curve well enough to say what is wrong
with a run that is going badly.

## Chapters

The chapters that exist in this edition are linked; the others are specified
in the curriculum and appear here when they are written.

1. [The Neuron and the Perceptron](01-perceptron.md): what an artificial
   neuron is, and what a single one can never learn.
2. [Multilayer Networks](02-mlp.md): why stacking layers with a nonlinearity between them
   lets a network represent almost anything.
3. [Softmax, Logits and Cross-Entropy](03-softmax.md): how a network chooses
   among many classes, and why the gradient of that choice is so simple.
4. [Backpropagation Derived from Scratch](04-backpropagation.md): how the
   gradient of one loss with respect to millions of parameters costs about
   one extra pass.
5. [The Backward Pass of Every Layer](05-layer-gradients.md): the backward
   pass of a matrix multiply, a sum, a broadcast and an elementwise function,
   written out exactly.
6. [Proving Gradients Correct](06-gradient-checking.md): how to know a backward
   pass is right without trusting the person who wrote it.
7. [Optimizers: From SGD to AdamW](07-optimizers.md): the smartest way to use
   a gradient once we have one.
8. [Learning-Rate Schedules and Gradient Clipping](08-schedules-and-clipping.md):
   why the step size should change during training, and what to do when a
   gradient explodes.
9. Initialization: Where Training Starts: why the random starting point
   decides whether a deep network trains at all.
10. Normalization Layers: how rescaling activations inside a network makes it
    train faster and more stably.
11. Regularization in Neural Networks: how to stop a network with millions of
    parameters from memorizing its training set.
12. Training Dynamics and Debugging: how to find out why a loss curve looks
    wrong.
13. First Complete Models: what it takes to build, train and verify a small
    network end to end.
