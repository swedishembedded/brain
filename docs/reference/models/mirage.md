# Visual perception (Mirage)

A 51M-parameter vision model that reads an image and produces structured
output: a scene description, object boxes and masks, text lines read from the
image, relations and form fields. It is small enough to run in real time on a
phone-class CPU. Both its weights and its published code are Apache-2.0.

## Support

| Capability | Supported |
|---|---|
| Inference             | [ ] |
| Training from scratch | [ ] |
| CLI                    | [ ] |
| HTTP API               | [ ] |
| D-Bus                  | [ ] |
| Batched serving        | [ ] |

Nothing is servable yet: brain can read and validate the released checkpoint
files, but the model does not run. This page grows as capabilities land.

## Getting the weights

Model id: `Cactus-Compute/mirage`. Not auto-fetched. Put the released
repository (`config.json`, `mirage.safetensors`, `mirage.cact`) under the
models directory as `Cactus-Compute/mirage/`; brain resolves it from there.
