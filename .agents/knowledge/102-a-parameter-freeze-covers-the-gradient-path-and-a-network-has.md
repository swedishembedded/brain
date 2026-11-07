<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 102. A parameter freeze covers the gradient path, and a network has more than one way to move
**Lesson:** "Frozen" is a claim about every path by which a number can change, not about the one
path the optimiser owns. The only proof that covers all of them is to train for real and compare
bit patterns.

## What happened

Adding an 81st class to a COCO-pretrained `yolov8n` without wrecking the other 80 needs a real
freeze, and `--freeze-backbone` had been a printed notice with nothing behind it.

The obvious implementation, withdrawing those tensors from the optimiser (which
`ParamStore::freeze_where` now does), is correct and insufficient. Written as a test that trains
for five real steps and then compares BITS, it fails on its first assertion, and the tensor it
names is `backbone.0.bn.run_mean`.

## Why

There were three routes by which a "frozen" network still moved.

**BatchNorm running statistics.** Running mean/variance are updated by the FORWARD pass. No
gradient is involved, no optimiser touches them, and every existing freeze mechanism in this
workspace reasons about gradients. A "frozen" backbone fed a narrow fine-tune set quietly
re-normalizes itself to that set's statistics, which is precisely the distribution shift the freeze
existed to prevent.

**The detection loss.** It scores BCE over the FULL `[anchors x nc]` grid against a target that is
zero everywhere except assigned `(anchor, class)` cells. A class absent from the fine-tune data is
therefore not merely unsupervised, it is actively trained, on every anchor of every image, toward
"never fire". Fine-tuning a COCO detector on people-only images destroys the other 78 classes while
the backbone sits there bit-for-bit perfect and every freeze test passes. Nothing moves that you
thought to check.

**Weight decay.** AdamW's decoupled weight decay is not a gradient. Zeroing a gradient still leaves
`w -= lr*wd*w`, so "gated off" is bit-exact only at `wd = 0`.

## What to do

- Pin the module to eval-mode BN as well as withdrawing tensors from the optimiser. It has to
  survive the `set_eval(false)` / `set_update_running(true)` that any training loop calls
  afterwards, hence a flag the broadcast setters re-assert rather than a one-shot toggle they
  silently undo.
- Gate the absent classes' gradient at the head with `Yolo::train_only_classes`. This also stops
  them reshaping the shared layers ABOVE the freeze.
- Keep the negative control as the regression guard for that gate: a test asserting that WITHOUT
  the gate those classes measurably get driven down, so the mechanism cannot be quietly removed.
- Prove a freeze by training for real and comparing bit patterns. A tolerance would have accepted
  all three of these.
