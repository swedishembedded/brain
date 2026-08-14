# World models

A world model lets you "play" inside a learned simulation of a game or
environment: you reset it with a few context frames, then step it forward
one discrete action at a time, and it predicts the next frame - rendered
live in a window so you can watch and steer it like a game.

| Architecture | Environment | Status |
|---|---|---|
| [DIAMOND](diamond.md) | Atari-100k | playable end to end |
| [GenieRedux-G](genieredux.md) | CoinRun | not yet playable (tokenizer/MaskGIT dynamics in progress) |

DIAMOND is the one to reach for today - see its page for getting started,
recording/replaying episodes, and fine-tuning on your own data.
