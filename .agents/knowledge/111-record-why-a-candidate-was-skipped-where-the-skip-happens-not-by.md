<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 111. Record why a candidate was skipped where the skip happens, not by asking again afterwards
**Lesson:** Have the loop that makes a decision emit the record of why candidates were skipped, as
it makes it. Reconstructing the reasons afterwards by asking again gives an answer that was not the
one acted on.

## What happened

`ProviderRegistry::resolve` chose an implementation with a loop that `continue`d past every provider
that was disabled, whose requirements the device did not meet, or that declined the request, and
returned only the winner. The reasons existed, for one stack frame, and were thrown away.

The failure mode is that "the tuned provider declined this shape", "the device cannot satisfy it"
and "someone disabled it in the environment" become one indistinguishable silence whose only
symptom is being slower than expected.

## Why

Reconstructing that afterwards by re-asking each provider is worse than useless. `accepts` may
consult measured state that has since changed, so the answer you report is not the answer that was
acted on, and you pay for the whole chain twice.

## What to do

Have the loop that makes the decision emit the record as it makes it. The cost is one `Vec` that
stays empty (no allocation) in the common case where the first candidate takes the work.
