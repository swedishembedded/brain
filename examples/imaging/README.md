# Imaging pipeline examples

## The whole edit in one call (`brain/imgpipe`'s `run` action, over D-Bus)

`edit_pipeline.py` drives `brain/imgpipe` over D-Bus: segment → refine the mask →
restore → (optionally) upscale, as a **single** `run`.

```sh
BRAIN_SAM2_WEIGHTS=... BRAIN_CODEFORMER_WEIGHTS=... BRAIN_ESRGAN_WEIGHTS=... \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 3
    python3 examples/imaging/edit_pipeline.py --image photo.ppm --point 614,430 --upscale'
```

Why one call rather than four:

* the intermediate mask and image **never cross the bus** — four separate calls
  would marshal a full-resolution image out and back three times;
* the composite happens **once**, at the end, so pixels outside the mask come
  back bit-identical instead of surviving three lossy round trips;
* a stage whose model is not configured fails with **that model's** own
  `set BRAIN_…` message, because the pipeline dispatches through the capability
  registry rather than linking the models.

`upscale` is a **tail**: it changes the image size, so it must be the last stage
and runs after the composite. Asking for it in the middle is rejected by
position rather than silently reordered. The returned mask travels at the
*output* size, so it still describes the image it came back with.

The stages and their parameters are the ones `brain caps brain/imgpipe` lists —
the example builds the JSON, it does not define it.
