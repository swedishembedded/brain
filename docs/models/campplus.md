# CAM++ (speaker encoder, component)

The 192-d x-vector speaker encoder [CosyVoice](cosyvoice.md) uses for
zero-shot voice cloning - turns a reference clip's 80-dim kaldi-style fbank
into the x-vector CosyVoice's flow model conditions its CFM decoder on. Not
independently servable: it has no capability manifest or CLI verb of its own,
reached only as part of CosyVoice's own actions.

Package: `brain-campplus`.
