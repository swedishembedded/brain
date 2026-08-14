# vlm - roadmap

Vision-language models: an image (and, for one variant, more) plus text in,
text out.

## Moondream 3 - not yet done

- [ ] A capability manifest / `brain moondream3 <verb>` action surface
- [ ] A CLI reference path
- [ ] A servable end-to-end pipeline (vision encoder → decoder, wired together)

Moondream 3's decoder is gradient-checked and its weights import correctly,
but it isn't reachable from any user-facing surface yet - it exists only in
tests.
