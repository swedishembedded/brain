<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 77. "Ported and unit-tested" is not "reachable": a whole decoder sat finished and unreferenced, and the gap that kept it out was one composition function and a number nobody had computed

A user ran a real, complete, correctly-named LTX-2.5 download through
`examples/videogen/images_to_video.sh` and got `missing tensor
decoder.conv_in.conv.weight`. The download was fine. The official Lightricks
repo ships the video VAE as TWO files with different decoder architectures
under the same `decoder.*` top-level name, and `import_vae` was built against
exactly one of them.

The immediate fix - detect the other architecture's tell-tale prefixes and
name it - was right and shipped. But it was a better error message for a
capability the repo already had: `crates/ltxv/src/na_decoder.rs` was a
complete, real-weight, cosine-1.000000000-verified port of that exact
decoder. `grep` found `import_na_decoder`/`NADecoder` referenced in precisely
two places: its own tests, and one doc comment explaining why it was not
wired in. Every part was proven. Nothing could call it.

**A module with passing parity tests and zero non-test callers is not a
finished feature, and its test suite will never tell you that.** The parity
suite drove `forward_context` and `forward_diff` directly, so it stayed green
forever while the thing had no entry point that turned a latent into pixels.
The missing code was about twenty lines of composition. What made it look
bigger than it was:

**The "differs enough to be a separate integration" note was half right, and
the half that was wrong had never been checked.** The standing comment said
the NA decoder's "tiling/scale contract" differed from the conv decoder's.
Scale: composing the four real upsample strides with each temporal-stride-2
shuffle's leading-frame drop gives `1 + 8*(lat_t-1)` frames at `lh*32 x lw*32`
- character for character the conv decoder's formula. There was no scale
difference at all, and ten minutes of arithmetic would have said so at any
point. Tiling: entirely real, see below. One vague sentence bundled a
non-problem with a real one and deterred anyone from separating them. When a
doc comment gives two reasons not to do something, check them separately.

**The real blocker was a number, and computing it was the actual deliverable.**
`na_attention` materialises a dense `[heads, nq, window]` f32 score buffer,
and stage 5's window is `11x11x11` = 1331 over the FULL output context. That
is 1082 MiB for the smallest clip the decoder can run and 5928 MiB for
512x512, against a measured `max_storage_buffer_binding_size` of **2047 MiB**
on this Tesla P40 - not the 4 GiB the `max_buffer_size` of 4094 MiB would
suggest, which is why guessing from the card's 24 GiB or from the wrong limit
field gets the answer wrong in both directions. So the honest state is "works
to about 17 frames at 288x288 on this card, refused above that by name". That
is a real, useful, shippable capability plus a precisely bounded gap - which
is only expressible once someone computes the bound. "Can't wire it in" was
never true; "can't wire it in above N pixels" was, and N was knowable all
along.

**Two byte-identical halves are a design input, not a coincidence.** Both
releases carry the SAME conv encoder and the same `per_channel_statistics` -
86 tensors, verified by range-reading both headers and comparing bytes, not
by trusting matching names and shapes. That is what lets one loader validate
the encoder half against one manifest whichever decoder came with it, instead
of a second config per release. Diffing two real checkpoints before designing
the loader cost minutes and removed an entire axis of duplication.

**A two-importer split is where two-way coverage quietly dies.** The
conv path had always errored on unused source tensors. The obvious NA
implementation - filter `encoder.*` to one importer, `decoder.*` to the other
- silently drops anything that is neither, which is exactly the corrupted-
download case the check exists for. Routing everything the decoder importer
does not claim into the encoder check as-is keeps one importer's leftovers
visible to the other. When splitting a validated whole into validated parts,
name where the remainder goes, or it goes nowhere.
