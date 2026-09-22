<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 138. An evaluation horizon that is a constant measures a different task

`ControlPipeline`'s evaluation and play stages both ran episodes with
`max_steps` hardcoded to 40, while training used `ControlSpec::max_steps`.
For the environment the constant was written against - a corridor shooter whose
episodes are a couple of dozen decisions - the two agreed and nothing was
wrong.

The first environment whose episodes are longer than that broke silently.
DOOM needs a couple of hundred decisions to cross a level, so evaluation
truncated every episode at 40, scored "did it reach the exit" as no, and
reported a win rate of 0.000 over 200 episodes. That number is not a policy
failure. It is the harness measuring whether the policy can finish a level in
less than a fifth of the decisions it takes to walk there - and it looks
exactly like a policy failure, which is the dangerous part.

THE RULE. An episode horizon belongs to the environment, not to the harness. A
default is fine; a constant the caller cannot reach is a measurement of
something the caller did not ask for. The fix is one field threaded from the
spec and a builder method for a pipeline that only loads weights.
