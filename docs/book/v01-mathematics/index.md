<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Volume I: Thinking in Mathematics

Every equation in the rest of the book is built from a small set of
mathematical ideas: numbers and the notation for writing about them,
functions, vectors and matrices, rates of change, probability, and the
arithmetic a real computer actually performs. This volume teaches those ideas
from school arithmetic upward, in the order in which each one is needed by the
next, and always with the machine learning problem that makes it necessary.

## Why it comes here

Machine learning is mathematics executed on hardware, and the failures that
cost the most time are rarely failures of an algorithm. They are a product of
probabilities that underflows to zero, a matrix whose shape was misread, a
derivative applied to the wrong variable, a sum whose terms were added in the
wrong order. A reader who can only follow the mathematics, or who can only run
the code, cannot find these. This volume gives the reader both: every idea is
derived, and then computed with real numbers that a program produced, so that
statement and arithmetic can be checked against one another.

The volume assumes nothing beyond school arithmetic. It is needed by every
later volume: Volume II uses functions, derivatives and probability to define
what it means to learn, and Volume III uses the chain rule, matrices and
floating-point numbers to build and train networks.

## What you can do after it

You can read and write the notation of the rest of the book: sums, products,
sets, functions, vectors, matrices and tensors, derivatives, gradients,
expectations. You can derive the gradient of a function of many variables,
reason about the shape and memory layout of an array, describe a data
distribution, quantify how much a finite sample can be trusted, explain
why a computer's arithmetic is slightly wrong and when it becomes completely
wrong, estimate the cost of a computation before running it, and explain how
a deterministic machine produces random numbers and why experiments must be
repeatable.

## Chapters

The chapters that exist in this edition are linked; the others are specified
in the curriculum and appear here when they are written.

1. [Numbers, Quantities and Notation](01-numbers.md): the numbers a machine
   learns with, and the notation (exponents, logarithms, sums, products, sets,
   function notation) for writing about them precisely.
2. Functions and Their Shapes: how to describe a rule that turns inputs into
   outputs, and what shapes such rules can take.
3. Vectors: Lists of Numbers with Geometry: how a list of numbers gains a
   length and a direction, and how to measure how alike two lists are.
4. Matrices as Linear Maps: what multiplying a matrix by a vector means, and
   why it is the central operation of machine learning.
5. Arrays, Shapes, Strides and Broadcasting: how vectors, matrices and
   higher-dimensional arrays are laid out in memory and indexed.
6. Eigenvalues, Rotations and Low-Rank Structure: the hidden structure of a
   matrix, and how a large one is approximated by small ones.
7. Derivatives: The Mathematics of Change: how fast a function's output moves
   when its input moves a tiny bit.
8. Gradients, Jacobians and Many Variables: which way to move millions of
   numbers at once to make a quantity smaller.
9. Integrals and Differential Equations: adding up infinitely many small
   pieces, and following a system that changes continuously.
10. Probability: Reasoning Under Uncertainty: putting numbers on uncertainty
    and updating them on evidence.
11. The Distributions Machine Learning Uses: the few distributions that keep
    appearing, and why.
12. Statistics: Learning Facts from Samples: how much a finite sample can tell
    us, and how to know whether a difference is real.
13. Information, Entropy and Surprise: how many bits it takes to describe what
    we expect to see, and what expecting the wrong thing costs.
14. Complex Numbers, Frequencies and the Fourier Transform: writing any signal
    as a sum of waves, and why that makes convolution cheap.
15. Floating Point and Numerical Error: why a computer gets arithmetic
    slightly wrong, and when slightly wrong becomes completely wrong.
16. Algorithms, Cost and Complexity: predicting how long a computation takes
    and how much memory it needs before running it.
17. Randomness on a Deterministic Machine: how a deterministic computer
    produces random numbers, and how to make random experiments repeatable.
