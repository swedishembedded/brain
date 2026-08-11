# Image generation: text-to-image and image editing

brain runs two diffusion-transformer models for producing images from a text
prompt, and for editing existing images with a prompt. Both are served the
same way — CLI, `brain do`, D-Bus, and the `/v1/images/generations` HTTP
route — and both support LoRA fine-tuning; the difference is in speed and
editing style.

## Capabilities

### Fast text-to-image and editing — `Tongyi-MAI/Z-Image-Turbo`

A distilled text-to-image diffusion transformer: generates an image from a
prompt in a handful of steps, trading a little quality headroom for much
lower latency. It also does image-to-image, inpainting, and outpainting
against an existing image, and can be LoRA-finetuned on your own images. A
slower, non-distilled base variant (`Tongyi-MAI/Z-Image`) is available for
more quality headroom. See [the Z-Image page](../models/zimage.md).

### Text-to-image and reference-image editing — `brain/flux2-klein`

Generates from a prompt alone, or edits one or more reference images in
place — recolor a photo, add weather, change style — driven by a short
instruction rather than a scene description. The Klein variant is fast and
distilled; an undistilled `base` variant trades that speed for classifier-
free guidance, useful when an edit needs to fight the source image's content
harder than Klein's guidance-free steering can. See
[the FLUX.2 page](../models/flux2.md).

## Which one to reach for

Z-Image Turbo is the faster of the two out of the box and its 4-bit/INT8
default footprint fits comfortably on one mid-range GPU. FLUX.2 Klein adds
reference-image editing as a first-class mode, with an undistilled `base`
variant available when an edit needs stronger steering. Both pages document
their own model ids, environment variables, and flags — start there for the
exact invocation.
