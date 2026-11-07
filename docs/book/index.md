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

### [Volume III: Building Neural Networks from Nothing](v03-neural-networks/index.md)

4. [Backpropagation Derived from Scratch](v03-neural-networks/04-backpropagation.md): How do we compute the gradient of one loss with respect to millions of parameters in about the cost of one forward pass?
<!-- /contents -->
