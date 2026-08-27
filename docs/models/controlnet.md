# ControlNet

A backbone-agnostic control seam (`ControlAdapter` declares a backbone's
named injection points; producer and consumer match by name and element
count) plus the SDXL `ControlNetModel` that is its first producer, built on
[SDXL UNet](sdxlunet.md)'s blocks (the trainable copy is recorded directly
from them, adding no new kernel).

This is a real, verified port - imported (844 -> 810 tensors, 5.00 GB fp32)
and residual-parity-gated against a hooked diffusers reference (140
comparisons, worst 1-cos 1.914e-11) on both a real GPU and CPU.

It is **served**: a capability manifest (`text2image`, model id
`brain/sdxl-controlnet`), a residency adapter (`BRAIN_SDXL_DIR` for the SDXL
backbone plus `BRAIN_CONTROLNET_DIR` for the ControlNet checkpoint), D-Bus
`Run`, and a runnable example (`examples/imagegen/controlnet_generate.py`),
on top of the same complete sampler loop [SDXL UNet](sdxlunet.md) uses - a
prompt plus a conditioning image (edge map, depth map, pose, ...) in, an
image out.

Two things it still does not do: batching (every request is its own
multi-step sample, so concurrent requests are served serially) and INT8;
there is also no backward/training path here. "InstantID works" is not
claimed (see [InstantID](instantid.md)).

Package: `brain-controlnet`.
