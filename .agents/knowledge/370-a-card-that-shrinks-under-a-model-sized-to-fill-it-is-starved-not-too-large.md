<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 370. A card that shrinks under a model sized to fill it is starved, not too large

Question: why did every request fail with "too large for that device's budget even with nothing resident" after one
out-of-memory fault, and keep failing after the neighbouring process had exited?

Finding (a defect): the default `--ctx` is the longest the cards hold, so the model's cost equals the card's usable bytes at
startup. The live capacity probe re-sizes a budget to `used + free`, so a neighbour taking a few GiB, or this process holding
memory it leaked (a device that reported a hard fault is deliberately never destroyed, see `DeviceShared::drop`), pushes the
usable bytes below that cost. The claim path compared the cost with the *current* usable bytes and called the refusal
`TooLarge`, the permanent class: every queued job failed, and nothing said the shortfall was a measurement rather than the
model. The leaked bytes never return, so the model was unservable until restart, in silence beyond the per-request 500.

The fix keeps the two apart. A budget remembers the most it has been (`Budget::peak_total`); a cost above that ceiling is
`TooLarge`, a cost above the current usable bytes only is `ClaimError::Starved`, which names the shortage, is retried on
every request and clears itself when the memory comes back. The executor logs the transition once (`UNSERVABLE <key>` and
`<key> is servable again`) and lists the key in `Stats::unservable`. What it cannot do is give the leaked bytes back: a
process whose device faulted needs a restart to serve a model sized to fill the card.

Rule: a refusal that depends on a live measurement is never the permanent class, and a model sized to the whole card has no
margin for anything the measurement can move.
