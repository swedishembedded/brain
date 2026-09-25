<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 163. A parameter sweep that picks its own bound has not found the optimum

Structure from motion chooses a metadata-less camera's focal length by
solving the capture under a sweep of candidates and keeping the best
(#159). The sweep ran from 0.5x to 1.3x the long side - a range chosen for
phone MAIN cameras. On a 16-photo capture from a phone ULTRA-WIDE (3264x2448)
it chose 1632 px, exactly 0.50x: the lowest candidate. Bundle adjustment then
walked the focal length to 1559 px (0.48x), outside the range the sweep had
looked at, and the log said nothing - "focal length 1632.0 px" read like an
answer.

The run still converged because 5% was inside bundle adjustment's basin. A
wider lens (fisheyes sit near 0.3x) would have been seeded through a focal
length the sweep could not reach, with the projective distortion #159
describes.

**Rule:** a grid search whose winner is the first or last grid point has
found a bound, not an optimum - widen the range or say so. The sweep in
`crates/sfm/src/incremental.rs` now runs from 0.3x to 1.3x and logs a
winner on either edge as "AT THE EDGE of the sweep".
