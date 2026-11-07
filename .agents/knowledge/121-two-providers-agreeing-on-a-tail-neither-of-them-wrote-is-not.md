<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 121. Two providers agreeing on a tail neither of them wrote is not parity

**Lesson:** A self-parity fixture proves the harness's plumbing and nothing about the dispatch-count
formula. The discriminating case is always the one that under-covers.

## What happened
A cross-provider parity harness compares two implementations' output buffers over seeded inputs. It
cannot see an output element that NEITHER side wrote: both buffers start zeroed, so an
under-dispatched thread count leaves the same zeros on both sides and the comparison passes.

## Why
That failure mode is precisely the one a dispatch-count formula has. It is silent corruption, not a
crash, because every kernel in this catalogue bounds itself.

This is the same reasoning as the existing rule that an over-dispatched grid is invisible where
every kernel self-masks.

## What to do
Hold at least one case of the operators added to such a harness against a HOST oracle. Use a shape
whose extent is not a multiple of the work-group size, so that a tail exists for an under-count to
leave behind.
