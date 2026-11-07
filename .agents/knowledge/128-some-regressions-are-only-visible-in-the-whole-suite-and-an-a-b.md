<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 128. Some regressions are only visible in the whole suite, and an A/B can be blind to them

**Lesson:** A test that passes standalone and fails in the suite is evidence about the code, not
only about the machine. An A/B only exonerates what its switch actually switches.

## What happened
Two things happened together here and both are worth keeping.

The regression above passed every targeted run of the affected test and failed only inside a
full-package run.

The A/B that was supposed to isolate the cause reported "no difference" and was right. The flag it
toggled was the new feature, while the regression was in a *supporting* change the flag did not
touch.

## Why
On an idle box the stall the regression introduced is cheap. Under load it is not.

## What to do
- Bisect a test that passes standalone and fails in the suite rather than re-running it.
- Before concluding "not my change", check that the control path is the old code and not merely the
  new code with one feature disabled.

Here the honest control was the previous commit's files, restored into the tree and run in the same
context.
