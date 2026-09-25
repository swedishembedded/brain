<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 161. A 3D split of a coarse blob lands half its children behind the surface

A scene grown from a sparse structure-from-motion cloud starts as a few
thousand large, overlapping blobs. Splitting one draws its children from its
own 3D density, so about half land BEHIND the surface, where the front
layers saturate transmittance in every view: they composite nothing, get no
gradient, and are (correctly) reclaimed at the next round. On a 16-photo
capture that made the population SHRINK for three rounds (8.8k -> 6.6k) while
the budget schedule asked for tenfold growth - refinement can at most double
the gaussians that are visible and ranked, and the rest of the room went
unspent.

Two thresholds made it worse, in opposite directions:

- starvation relative to the median contribution condemns gaussians for
  being small (11k of 26k flagged in one round of a blob scene);
- an absolute starvation threshold of 0.25 px per view is an ordinary
  contribution once there are about as many gaussians as pixels - it
  suppressed 150k of 344k in one round and doubled the loss.

**Rule:** `splat::density` fills whatever room refinement leaves with
samples at the top-scored sites (3DGS-MCMC's growth, aimed by credit instead
of opacity, with its opacity/scale correction), and calls a gaussian starved
only below 0.01 px per view - rendering essentially nothing anywhere.
3DGS-MCMC itself collapsed on the same capture: 434k samples added onto 66k
sites corrected every site's opacity below the relocation threshold, and the
next round relocated 460k of 500k.
