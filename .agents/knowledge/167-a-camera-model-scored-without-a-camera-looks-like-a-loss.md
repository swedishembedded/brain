<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 167. A camera model scored without a camera looks like a loss

A photometric camera model (per-view exposure and white balance,
vignetting, response) was measured to cost 1.3 dB on held-out photographs
of a 16-photo capture, and was left off by default on that evidence. The
held-out views were rendered from the scene alone, with no camera at all:
the scene fitted WITH a camera model is radiance in the capture's average
camera, and a held-out photograph was shot through its own camera. Scored
through the average camera at the photo's EXIF exposure, and with the
standard appearance-fitted protocol (exposure and white balance fitted on
one half of the frame, scored on the other), the same model is +0.08 dB and
+0.011 SSIM on that capture (`recon::eval::score_held_out`).

**Rule:** evaluate a model component through the same forward the fit used.
A held-out score that drops the part of the model which maps the scene to
pixels measures the protocol, not the component - and a decision made on
it switches off exactly the capability that matters for the captures that
need it.
