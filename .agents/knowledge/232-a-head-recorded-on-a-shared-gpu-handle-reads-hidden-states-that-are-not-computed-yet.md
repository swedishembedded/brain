<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 232. A head recorded on a shared GPU handle reads hidden states that are not computed yet

`Ops::new(gpu.share())` gives the `Ops` façade a second handle onto the same
device with its own command stream. Everything the façade only RECORDS
(`matmul`, `matmul_dx`, `embed`, the head's `*_steps`) is submitted by the model
on its own handle and so stays in order with the layer work. The immediate
entry points of `HeadCe` (`forward`, `backward`) instead submit on the
façade's handle, which is not ordered after the model handle's pending submits.

On the CPU backend, which runs every submit eagerly, that is invisible. On a
GPU the head ran against `xn_final` before the final norm had executed: all
logits were zero, the loss was exactly `ln(vocab)` for any adapter, and every
gradient was zero, so a LoRA run "trained" without moving a weight and no
assertion on shapes or finiteness noticed.

qwen35 therefore records the head with `forward_steps` / `backward_steps` and
submits them on its own handle, reading the per-token loss from the target
buffer it already holds. `tests/lora_head.rs` compares the default device
against the CPU backend for this reason (a uniform loss fails it by name), and
the older convergence and freeze tests, which build on the default device,
caught it first.

The general rule: code that mixes the façade's immediate APIs with the model's
own submits has to share one handle, or the model has to `poll_wait` before
handing a buffer over.
