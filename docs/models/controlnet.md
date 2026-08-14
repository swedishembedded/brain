# ControlNet (not yet servable)

A backbone-agnostic control seam (`ControlAdapter` declares a backbone's
named injection points; producer and consumer match by name and element
count) plus the SDXL `ControlNetModel` that is its first producer, built on
[SDXL UNet](sdxlunet.md)'s blocks (the trainable copy is recorded directly
from them, adding no new kernel).

This is a real, verified port - imported (844 -> 810 tensors, 5.00 GB fp32)
and residual-parity-gated against a hooked diffusers reference (140
comparisons, worst 1-cos 1.914e-11) on both a real GPU and CPU - but
forward/residuals only: no backward, no int8, no batch > 1, no sampler
loop, no CLI and no serving surface. "InstantID works" is not claimed
(see [InstantID](instantid.md)). Not something you can run as a model
today.

Package: `brain-controlnet`.
