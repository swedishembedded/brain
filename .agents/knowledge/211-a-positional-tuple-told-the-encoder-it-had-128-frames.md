# 211. A positional tuple told the audio encoder it had 128 frames

`audio::asr_frontend::qwen_logmel` returns `(mel, n_mels, n_frames)`. The three
Qwen3-ASR call sites read it as `(mel, valid, _n)`, so the encoder was told
`valid = 128`, the mel-bin count, whatever the clip: it saw the first 128 mel
frames (about 1.3 s) of every utterance. The decoder was assembled for the
same wrong count (17 audio tokens for a window that holds about 100).

How it showed: `brain qwen3asr transcribe` returned the first two to four words
of 3 to 6 second utterances ("The people must." for a 15-word sentence, empty
text for an 8 s clip) while Nemotron transcribed the same clips correctly. The
window size (5, 10, 30 s) changed nothing. Cropping the audio showed it: a crop
of 0 to 1 s gave "What did you think?", and one starting at 1.0 s gave "Think
of taxation without representation." The later audio was fine and was simply
never encoded.

Nothing caught it because the only end-to-end test of this path is `#[ignore]`d
and takes its frame count from the reference mask, not from the front end, and
the only front-end test uses the tuple correctly. Round-tripping 120 spoken
sentences through it gave a corpus word error rate of 0.78 and lost 120 of 120
sentences; the same sentences through Nemotron gave 0.038.

Fix: `caps::window_features` is the one place the padded window meets the front
end, and a spec pins that the encoder is told every frame of the window
(`window / 160` frames, 128 bins each). The SDK has a real-weights round trip
for Qwen3-ASR. A positional triple of two integers and a buffer invites this
swap again; a named struct from `qwen_logmel` would remove it.
