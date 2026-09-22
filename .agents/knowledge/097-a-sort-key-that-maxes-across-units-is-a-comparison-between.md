<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 97. A sort key that maxes across units is a comparison between incompatible things

`plan`'s largest-first ordering was
`Reverse(peak_vram.max(cost.ram).max(cost.npu))`. It reads as "how big is
this part", and it type-checks, because all three are `u64` bytes. But VRAM
bytes, host bytes and NPU bytes are capacity in three different budgets on
three different devices, and the number that decides who gets first pick of
the CARDS must be made of card bytes only. A part declaring 6 GiB of VRAM
and 20 GiB of host staging sorted ahead of a 14 GiB DiT and took the only
card the DiT could have used - on a machine where `dit->gpu0, te->gpu1` was
sitting right there.

The same file had the mirror-image bug in its charging: `MemCost::ram` is
documented as "host bytes it will hold REGARDLESS of where it is placed",
and only `cost.on(device)` was ever charged - so a part landing on a card
was charged its VRAM and its host RAM was neither checked nor charged
against anything. Two GPU-placed parts holding 20 GiB of host RAM each left
the host tier reporting itself completely empty.

A multi-dimensional cost needs its dimensions kept apart in BOTH directions:
each dimension charged to its own budget, and each ordering decision made
from the dimension it is actually about. `u64` is not a unit.
