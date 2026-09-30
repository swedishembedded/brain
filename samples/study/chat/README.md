<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: study/chat

LoRA fine-tune a chat model on a handful of a product's own conversations,
measure on held-out questions whether it helped, show that a cancelled run
resumes to exactly the adapter an uninterrupted run produces, and chat with
the result.

```bash
make samples/study/chat/build
make samples/study/chat/run ARGS="--base Qwen/Qwen3-0.6B"
```

## What it demonstrates

* **Fine-tuning is a library call.** Everything here is `brain::ChatFineTune`
  (the `study` surface) and `brain::ChatPipeline` (the `text` surface). The
  sample writes a tiny `generic-messages-v2` dataset - eight support
  questions about a fictional router, only the assistant turns supervised -
  and a held-out set of three questions asked differently.
* **Held-out measurement, before and after.** The fine-tune scores the base
  and the base-plus-adapter on the held-out records with `brain::score_chat`'s
  arithmetic: mean per-token cross-entropy and greedy token accuracy over the
  supervised positions. Anything not measured is printed as `not measured`,
  never as zero.
* **Exact resume.** The same fine-tune is run again with a `CancelToken` that
  fires after `--cancel-after` steps. That run exports nothing and leaves its
  resume state; starting it again continues from the saved step
  (`resumed_at`) to `steps_completed == steps`, and its adapter digest is
  compared with the uninterrupted run's. The program exits non-zero if they
  differ.
* **Serving what was trained.** The adapter is folded into the base in a
  `ChatPipeline`, whose identity reports the adapter digest it loaded - the
  same `sha256:` digest the fine-tune reported and the one
  `brain serve --adapter FILE` prints - and it answers one held-out question.

## Usage

```text
--base BASE         a Qwen3 checkpoint file (tokenizer.json and
                    tokenizer_config.json beside it) or a vendor/repo
                    reference in the model store (default Qwen/Qwen3-0.6B)
--out DIR           where the dataset, runs and adapters go
                    (default: <tmp>/sample-study-chat, recreated every run)
--steps N           optimizer steps per run (default 24)
--cancel-after N    cancel the interrupted run after this step (default steps/2)
--lr X              peak learning rate (default 5e-4)
--device SPEC       cpu, gpu, gpu:1, ... (default: auto)
--models-dir DIR    the model store a vendor/repo base resolves in
```

## Expected output

The shape, with values elided (they depend on the base, device and steps):

```text
base      Qwen/Qwen3-0.6B
dataset   <out>/train.jsonl (8 records), held-out <out>/held_out.jsonl (3 records)

[1/3] uninterrupted fine-tune, 24 steps
  step   1/24  loss ...  lr ...
  ...
  completed: steps_completed 24/24, resumed_at none (fresh start), loss ... -> ...
  adapter <out>/uninterrupted/adapter.safetensors sha256:<hex>
  held-out base   loss ...  token accuracy ...  (N positions, 3 records, 0 skipped)
  held-out tuned  loss ...  token accuracy ...  (N positions, 3 records, 0 skipped)

[2/3] the same fine-tune, cancelled after step 12
  ...
  cancelled: steps_completed 12/24, resumed_at none (fresh start), loss ... -> ...
  no adapter exported; resume state <out>/resumed/train.state

[3/3] started again: it continues from the saved state
  ...
  completed: steps_completed 24/24, resumed_at step 12, loss ... -> ...
  adapter <out>/resumed/adapter.safetensors sha256:<hex>
  same adapter as the uninterrupted run: yes, byte for byte

chat      adapter digest as loaded: sha256:<hex>
  user       On which port is the Harbor-7 admin UI reachable?
  assistant  ...
  (held-out reference: The Harbor-7 admin UI is reachable on port 8443.)
```

One measured run, with every default (`Qwen/Qwen3-0.6B` from the model
store, 24 steps, cancelled after 12, learning rate 5e-4) on one Tesla P40
through the Vulkan backend, release build: held-out loss 3.8107 -> 0.6507
and token accuracy 0.526 -> 0.868 over 76 supervised positions, the resumed
run's adapter digest identical to the uninterrupted run's, the tuned model
answering the held-out question with the reference sentence, and 3 min 36 s
wall time for the whole program. Other hardware, bases or settings will
differ; this is one run, not a benchmark.

## What it needs, and its limits

* **A Qwen3 chat checkpoint.** `ChatFineTune` trains Qwen3 decoders only
  (dense GQA + QK-norm; a checkpoint `brain` imported or downloaded, e.g.
  `brain pull Qwen/Qwen3-0.6B`). Its directory must hold `tokenizer.json` and
  a `tokenizer_config.json` carrying the chat template: every record is
  checked against that template before a device is claimed. Other
  architectures are refused.
* **A GPU, in practice.** It runs on the CPU backend too, but training a
  0.6B-parameter model there is slow; the sample makes no claim about how
  long either takes on your hardware. The three runs add up to two full
  runs' worth of steps, and every run scores the base on the held-out set
  (each completed run also scores its adapter).
* **Memory.** The base is held at fp32 while it trains, plus the LoRA
  weights, their optimizer state and activations for one row of the longest
  record.
* **What eight examples can teach.** This is a demonstration of the
  mechanics, not of a useful fine-tune: a few dozen steps on eight records
  moves the held-out loss, and whether the one sampled answer repeats the
  trained facts depends on the base, the steps and the rate. The held-out
  numbers are the measurement; the chat line is an illustration.
