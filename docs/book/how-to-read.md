<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# How to Read This Book

## Who it is for

The book assumes that you can read carefully and are willing to work
through examples with pencil and paper. It does not assume any mathematics
beyond arithmetic, and it does not assume any particular programming
language. If you already know some of the early material, you can skim it,
but read the chapter summaries: later chapters use the exact notation and
terms defined there.

To do the implementation exercises you need to be able to write and run
small programs in some language: loops, functions, arrays. Any general
purpose language works. The book never relies on a library that you would
not be able to write yourself.

## The order of the volumes

The sixteen volumes are ordered so that each one only needs what came
before it.

| Volume | Topic | You will be able to |
|---|---|---|
| I | Mathematics | read and derive every equation in the rest of the book |
| II | Learning | state what learning is, train a simple model, evaluate it honestly |
| III | Neural networks | build, train and verify a neural network from nothing |
| IV | Computation | make that network run fast on CPUs, GPUs and accelerators |
| V | Sequences and transformers | build and train a small language model |
| VI | Advanced language models | implement modern large language model architectures |
| VII | Vision | build models that detect, segment and describe images |
| VIII | Image generation | build diffusion and flow-matching image generators |
| IX | Audio | build speech recognition, speech synthesis and audio codecs |
| X | Video, 3D and world models | generate video, reconstruct scenes, simulate environments |
| XI | Forecasting | build and honestly evaluate forecasting systems |
| XII | Reinforcement learning | train agents and align language models with feedback |
| XIII | Training at scale | train large models within limited memory and across devices |
| XIV | Inference and serving | serve many models to many users efficiently and safely |
| XV | Methods | port, test, benchmark and publish machine learning work |
| XVI | Systems | design and build a complete machine learning engine |

Volumes I to IV are the foundation; everything later depends on them.
After Volume V the volumes branch: vision, generation, audio and forecasting
can be read in any order, though each one assumes Volumes I to V, and
some borrow from each other (video generation, for example, uses both the
image generation and the vision volumes). Volumes XIII to XVI assume you
know at least one model family well.

## The shape of a chapter

Every chapter answers one question, stated under its title, and every
chapter is laid out the same way, so that after the first few you always
know where you are.

1. **The problem.** What we cannot yet do, why it matters, and a small
   concrete case you can see and compute.
2. **Before you begin.** A handful of short questions about ideas from
   earlier chapters that this one builds on, each naming where the idea was
   taught. Answer them from memory before reading on. If one is hard, go
   back to that chapter first: the new material will not make sense on top
   of a shaky foundation, and it is much cheaper to repair the foundation
   now.
3. **The explanation.** One new idea at a time: first in plain language with
   a concrete example and a picture, then derived properly from what you
   already know, then written as a precise algorithm with the shape and type
   of every array. New terms appear in **bold** where they are defined, and
   never before.
4. **Making it work.** Memory layout, numerical pitfalls, cost, and what goes
   wrong in practice.
5. **In brain.** Where the idea lives in a real engine, how it is tested
   there, and often the story of a mistake that shaped it.
6. **Check your understanding.** Short questions that ask you to explain,
   predict or spot an error, with answers at the end of the chapter.
7. **Exercises.** Calculations by hand, implementations in your own
   language, and investigations that ask you to change something and
   explain what happens.
8. **Summary** and **Key terms.** What the chapter established and why the
   next one follows, then every new term with a plain-language definition.
   The same definitions, from every chapter, make up the
   [Glossary](glossary.md).

Each volume ends with a **project**: a larger piece of work, built from
nothing in the language of your choice, that uses everything the volume
taught. The projects are where knowledge becomes ability. Do not move to
the next volume until its project works and you can explain why it works.

## How to study with it

Read with something to write on. When a derivation appears, cover the next
line and try to produce it yourself. When a worked example appears, compute
at least its first row by hand before reading the table.

Answer the "Before you begin" questions honestly, from memory. Recalling an
idea is what makes it stay; rereading it is much weaker. The questions
deliberately reach back several chapters, so that ideas come back to you
after you have had time to forget them a little.

If an explanation stops making sense, do not push on. Go back to the last
point where everything was clear and look for the first word or symbol you
cannot define precisely; the [Glossary](glossary.md) and
[Notation](notation.md) are there for this. Almost always the difficulty
started there, not where you noticed it.

## Conventions in the text

Definitions are set apart and named in **bold** the first time they appear.
Every term is defined before it is used; if a chapter needs an idea from an
earlier chapter it says where. The glossary of symbols is in
[Notation](notation.md), and the conventions every algorithm follows are in
[Pseudocode Conventions](pseudocode.md). Read both before Volume III.

Worked examples show real numbers. They are not illustrative: each was
computed by a reference program and is checked whenever the book is built.
When you implement an algorithm, reproduce the worked example first. If your
numbers differ in more than the last digit or two, your implementation (or,
occasionally, the book) is wrong, and finding out which is the exercise.

A table agrees with yours when each value matches after rounding to the
digits printed (a difference below half a unit in the last place printed),
unless the text states a tolerance. A worked example that uses random numbers
prints its seed and the order of its draws; the generator that produces them
is built in Chapter 17, and the earlier chapters print the drawn values when
they are few.

References to brain look like `crates/gradcheck` (a directory in brain's
source tree), commit `afbbae41d` (a change in its history, identified by the
first characters of its hash) or knowledge #3 (a numbered engineering
finding). You do not need brain to read the book; these references are for
the reader who wants to see the idea in a working system.

## Exercises

Exercises are marked by level:

- **[Calculate]** work it out by hand. These check that you can follow and
  apply the derivation.
- **[Implement]** write it in a language of your choice and reproduce the
  chapter's worked example.
- **[Investigate]** change an assumption, measure what happens and explain it.
  These have no single answer; they are where understanding is tested.

Do the implementation exercises. Machine learning is unusual in how often a
program runs without error, produces plausible numbers and is still wrong.
Almost every chapter shows a real example. The only defence is to have built
and checked the pieces yourself.
