<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 334. Full fine-tune gradients accumulated across steps, and a process-wide variable chose the optimiser

`a_full_fine_tune_records_the_seed_it_drew` failed about one run in three when its file's tests ran together:
two runs of one seed gave different weights. Finding it took the training step's own numbers, not the code:
hashing every gradient per step for six concurrent runs of one seed showed all of them equal at step 0, and at
step 1 five runs agreeing and one not, in every tensor, with the same loss. The odd run's gradient was the
other five's minus the step-0 gradient, exactly.

Two defects, one hiding the other:

* `ParamStore::zero_grads` cleared the trainable and frozen gradients and left out the `Role::Offload` ones. A
  full fine-tune with the AdamW moments in host RAM reads each offloaded gradient and never clears it, so from step 1
  on it trained on the sum of its own gradient and every earlier step's. Every full fine-tune that ran through
  the offloading optimiser (`ChatFullFineTune`, `finetune_full_controlled`, the data-parallel and sharded
  trainers) trained that way; LoRA runs do not offload and were not affected.
* Whether a build offloads was the process environment variable `BRAIN_OFFLOAD_ADAM`, set to `1` before
  the model was built and restored after. Two trainings in one process read each other's setting: one built
  with the device optimiser (correct gradients), the other with the offloading one (accumulating gradients), so
  the same seed gave two different runs depending on timing. `qwen3::model::with_offloaded_adam` now scopes the
  choice to the building thread; the variable is only the default.

Also fixed on the way: `par::sum_sq_f64` added its per-tensor partials with rayon's adaptive sum, whose last bit
depended on free threads (a grad-norm that differs in its last bit scales every update differently); it now cuts each
tensor into fixed chunks and adds the partials in order.

A weights digest equal across concurrent runs of one seed is the check that finds this class; an assertion on
the loss would not, because the loss at the divergent step was bit-identical.
