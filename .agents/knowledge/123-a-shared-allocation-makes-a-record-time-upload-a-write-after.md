<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 123. A shared allocation makes a record-time upload a write-after-write race

A dispatch-recording API where each step carries its own parameters invites
uploading them when the step is recorded. That is only correct while every
step's parameter storage is private to it. The moment that storage is shared
between steps of the same shape - which is what any address-stability scheme
needs, graph capture most obviously - a record-time upload becomes a host
write that races the device: two steps recorded before the submission runs
both write the same buffer, and the second wins for both.

The upload has to move to where the ordering already exists, between the two
dispatches in the queue that separates them. The tell is not a crash. Both
steps compute, with the same parameters, and the wrong one of the two is the
survivor - so the failure looks like a model that is subtly off rather than a
backend that is broken.
