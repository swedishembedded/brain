<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 162. A raw learning rate on linear parameters turns a sparse start into fog

`splat::opt` optimizes positions and scales LINEARLY, and Adam's step is
about `lr` whatever the gradient. The fit normalizes the scene to about one
unit across - and on a real capture that extent is set by the distant
background, so the subject's gaussians are ~0.002-0.01 units. A raw
`lr = 5e-3` then moves positions ~30x faster than 3DGS does and lets a small
gaussian grow several-fold in ONE step.

`FitCfg::from_sparse_points` shipped with exactly that: it switched the pixel
size cap off and left the radius-relative budgets unset, reasoning that a
sparse cloud must grow a long way. On a 16-photo capture, 60 iterations
produced uniform grey fog (every render needed ~2M tile instances for 50k
gaussians - each covering ~10% of the frame), the loss plateaued at 0.21-0.3
for hours of runs, and the backward's cost exploded with the overlap. Density
control got the blame first, and several real defects were found and fixed
there (#160, #161) without moving the plateau at all.

Every subsystem in the preset had its own passing test; the COMPOSITION had
none, and the one synthetic scene that would have been natural to write
first does not reproduce the failure - without distant background its
gaussians are large in normalized units and the raw rate is harmless.

**Rule:** a preset that composes subsystems gets its own end-to-end test on a
scene with the property that matters (here: distant background). Geometry
learning rates are expressed relative to the gaussians' own size
(`position_budget`, `scale_budget`, `rotation_budget`) and sizes are capped
in pixels (`max_scale_pixels`). With them: 200 iterations at 384 px went
from 0.31 to 0.087 loss on the same capture, and the render needed 81k tile
instances for 50k gaussians.
