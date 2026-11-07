<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 161. A 3D split of a coarse blob lands half its children behind the surface

**Lesson:** Splitting a coarse blob by its own 3D density puts about half the children out of sight.
Fill the room refinement leaves at the top-scored sites instead, and call a gaussian starved only
when it renders essentially nothing anywhere.

## What happened

A scene grown from a sparse structure-from-motion cloud starts as a few thousand large, overlapping
blobs. On a 16-photo capture the population SHRANK for three rounds (8.8k -> 6.6k) while the budget
schedule asked for tenfold growth. Refinement can at most double the gaussians that are visible and
ranked, and the rest of the room went unspent.

## Why

Splitting a blob draws its children from its own 3D density, so about half land BEHIND the surface,
where the front layers saturate transmittance in every view. They composite nothing, get no
gradient, and are (correctly) reclaimed at the next round.

Two thresholds made it worse, in opposite directions:

- starvation relative to the median contribution condemns gaussians for being small (11k of 26k
  flagged in one round of a blob scene);
- an absolute starvation threshold of 0.25 px per view is an ordinary contribution once there are
  about as many gaussians as pixels. It suppressed 150k of 344k in one round and doubled the loss.

## What to do

**Rule:** `splat::density` fills whatever room refinement leaves with samples at the top-scored
sites (3DGS-MCMC's growth, aimed by credit instead of opacity, with its opacity/scale correction),
and calls a gaussian starved only below 0.01 px per view: rendering essentially nothing anywhere.

## Evidence

3DGS-MCMC itself collapsed on the same capture: 434k samples added onto 66k sites corrected every
site's opacity below the relocation threshold, and the next round relocated 460k of 500k.
