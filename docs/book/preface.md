<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Preface

This book teaches machine learning from the beginning: from what a number is
to how a trained model with billions of parameters answers thousands of users
at once. Along the way it builds everything a working engine needs: the
mathematics, the algorithms, the numerical methods, the code that runs on the
hardware, the training methods and the serving system. By the end you should
be able to build such an engine yourself, in whatever programming language
you prefer, and to understand every decision inside it well enough to change
it.

## Why another machine learning book

Most machine learning texts sit in one of two places. Some teach the theory
and stop before the engineering: they derive backpropagation but never show
why a correct derivation can still produce a wrong gradient on a GPU. Others
teach a framework: they show which function to call, and the reader learns
the framework rather than the field. Neither leaves you able to build the
thing yourself.

This book goes the whole distance. Every important idea is introduced by the
problem it solves, explained in plain language, derived from things you have
already learned, written down as a precise algorithm, and then followed into
a real implementation, including the ways that implementation went wrong.

## Where the material comes from

The engineering half of this book comes from **brain**, a machine learning
engine written from scratch. brain trains and serves neural networks on
CPUs, GPUs and neural processing units from one code base. It implements
language models, vision models, image, audio and video generators,
forecasters, reinforcement learning, 3D reconstruction and more, all on top
of its own kernels rather than an existing framework. Over its first months
its developers made several thousand commits, and wrote down hundreds of
expensive mistakes as numbered engineering findings.

That history is unusual teaching material. A finished system hides how it was
made; its history does not. When this book explains why attention is computed
in tiles, it can show the moment a measurement revealed that one part of the
computation was running on a single GPU block while thousands sat idle. When
it explains why a gradient check needs more than one test size, it can show
the defect that a test with all dimensions equal to one let through. These
stories appear throughout the book, each placed where you have learned
enough to understand it. The chronology appendix shows the order in which
things actually happened.

## What this book is not

It is not brain's manual. brain's own documentation explains how to install
it and use it. This book uses brain as evidence: that each idea can be
implemented, tested, made fast and put in front of users. Every concept is
taught so that you could implement it without brain, and in another language.
Where the book points at brain's source code, it does so the way a physics
text points at an experiment: to show that the theory holds up, not to ask
you to memorise the apparatus.

It is also not a survey of the latest papers. Model families change every few
months; the ideas underneath them change far more slowly. The book covers the
families brain implements in enough depth that you could reproduce them, but
it organises them by the ideas they share, so that the next family to appear
will look familiar.

## How it is built

The book is generated as carefully as the engine it describes. Its outline is
a specification in which every chapter declares the concepts it introduces
and the concepts it relies on; a check refuses any chapter that uses an idea
before the book has taught it. Every number printed in a worked example is
computed by a short reference program written independently of the engine,
and checked again whenever the book is built. Every figure is generated from
a source file, and the figures that show data are drawn from the same
arithmetic the text describes. Every historical claim cites the commit or
engineering finding it rests on, and those citations are checked against the
repository.

None of this makes the book free of errors. It does mean that most errors can
be found and fixed mechanically, and that the remaining ones can be reported
precisely.

## Acknowledgements

The engine this book describes stands on the published work of a great many
researchers and engineers, whose papers and reference implementations are
cited where their ideas appear. Any mistakes in explaining their work are
ours.

Martin Schröder
