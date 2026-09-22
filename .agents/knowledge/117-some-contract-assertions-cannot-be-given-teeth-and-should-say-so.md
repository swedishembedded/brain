<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 117. Some contract assertions cannot be given teeth, and should say so

`cuMemAlloc` guarantees nothing about its contents, every other backend in this
engine hands back zeroed storage, and model code depends on it - so the CUDA
backend zeroes explicitly. The test for it cannot be made to fail: with the
zeroing removed, the driver still returned zeros across twenty dirty-free-
reallocate rounds, because it scrubs a freed allocation before reissuing it.
That is a driver's courtesy, not an API guarantee, so the zeroing stays.

The thing to avoid is letting such an assertion sit in the suite looking like
evidence. Either it says in its own doc comment that it was NOT shown to
discriminate, or a later reader counts it among the tests that prove something.
This is lesson 113's failure mode one level up: there, dirtying the memory gave
the test teeth; here, dirtying it was tried and the teeth still are not there.
