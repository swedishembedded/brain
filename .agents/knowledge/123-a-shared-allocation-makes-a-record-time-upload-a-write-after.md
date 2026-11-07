<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 123. A shared allocation makes a record-time upload a write-after-write race

**Lesson:** Uploading step parameters at record time is only correct while each step's parameter
storage is private to it. Once storage is shared, the upload must move to where the ordering
already exists.

## What happened
A dispatch-recording API where each step carries its own parameters invites uploading them when the
step is recorded. That is only correct while every step's parameter storage is private to it.

The tell is not a crash. Both steps compute, with the same parameters, and the wrong one of the two
is the survivor. The failure looks like a model that is subtly off rather than a backend that is
broken.

## Why
The moment that storage is shared between steps of the same shape, a record-time upload becomes a
host write that races the device. Any address-stability scheme needs this sharing, graph capture
most obviously.

Two steps recorded before the submission runs both write the same buffer, and the second wins for
both.

## What to do
Move the upload to where the ordering already exists: between the two dispatches, in the queue that
separates them.
