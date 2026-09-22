<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 113. An idle device makes an uninitialised-memory test pass

The first version of the `__shared__` zero-init gate read zeros with the
zeroing removed, because on a quiet GPU the block was the first tenant of that
scratch and the memory happened to be zero. A test for "this memory is not
initialised for you" has to DIRTY the memory first - here, a preceding kernel
declaring the same amount of `__shared__` and filling every element of it -
otherwise it measures how busy the device was, and reports that as correctness.
The same trap applies to any freshly-mapped host allocation.
