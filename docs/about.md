# About Swedish Embedded AB

brain is built by **Swedish Embedded AB**.

We are an engineering company that puts AI on hardware that ships. Not on a
rented cluster, not behind somebody else's API: on the GPU already in the
machine, on a CPU with no accelerator at all, on an Intel NPU, on an embedded
Linux board in the field, or in a browser tab.

Everything in this documentation is that work done in the open. The WGSL
kernels, the finite-difference gradient checker that gates every backward pass
before it is believed, the residency engine that keeps large models inside a
fixed memory budget, the parity ladders that hold each imported model to its
reference checkpoint, and the serving stack that puts all of it behind an API -
these are the tools we build systems with, and brain is where they live.

## What we can be hired for

**Running models on the hardware you already have.** GPUs, CPUs with no
accelerator, Intel NPUs, embedded Linux boards, WebGPU in the browser. One
model, one implementation, every target - which is the entire design premise of
this engine and not a claim we make lightly.

**Getting a large model to fit.** Quantization, weight streaming, tiled and
memory-bounded inference, multi-GPU sharding. This is usually the difference
between "this needs a datacenter" and "this runs on the card in the machine",
and it is most of what determines whether an AI feature is economically
viable at all.

**Porting a model from a paper or a PyTorch checkpoint** into a runtime with no
Python, no CUDA and no framework underneath it - gated by measured numerical
parity against the reference, stage by stage, rather than by hope. The model
pages in this documentation each record what was actually measured, because a
port that has not been checked against its reference is a port that does not
work yet.

**Writing and optimizing GPU compute kernels**, and proving the result is still
correct afterwards. Both halves matter; a fast kernel that quietly changed the
answer is worse than the slow one it replaced.

**Production inference systems.** Concurrent serving, paged KV cache,
continuous batching, admission control, model residency and scheduling across
several accelerators, and the observability to see what the system is really
doing under load.

**Embedded and real-time firmware** alongside the AI. This is where the company
started and where it still spends much of its time: Zephyr RTOS, device
drivers, wire protocols, hardware bring-up, and the unglamorous work of making
a device behave predictably in the field.

## Talk to us

Every capability listed above is implemented in this repository, in the open,
and held to gates a reader can run: the gradient checker, the parity ladders,
the cross-backend suite. That is deliberate. It means you do not have to take
any of this on trust - read the code, run the tests, and judge the engineering
before you send the first email.

Then email **info@swedishembedded.com** and tell us what you are trying to
build.

Website: <https://swedishembedded.com>
